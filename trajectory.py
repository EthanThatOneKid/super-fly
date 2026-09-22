"""Checksummed trajectory shards for offline pretraining and DAgger aggregation.

A dataset directory holds one or more ``shard-XXXXX.npz`` files plus a
``manifest.json`` that records, for every shard, its sample count, its SHA-256
checksum and a provenance block describing how the shard was produced (teacher
plan, on-policy rollout, DAgger recovery window, seed, cadence, source
checkpoint). ``iter_dataset`` refuses to load a dataset whose shards do not
match their recorded checksums, so a reported metric can always be traced back
to exact bytes on disk.
"""

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, Iterator, List

import numpy as np

SCHEMA_VERSION = 1


def checksum_file(path) -> str:
    """SHA-256 of a file on disk."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Backwards-compatible private alias.
_checksum = checksum_file


def load_manifest(dataset_dir) -> Dict[str, object]:
    """Read ``manifest.json`` for a dataset directory."""
    return json.loads((Path(dataset_dir) / "manifest.json").read_text())


def write_manifest(dataset_dir, manifest: Dict[str, object]) -> Path:
    """Atomically persist a dataset manifest."""
    root = Path(dataset_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    tmp_path = root / "manifest.json.tmp"
    tmp_path.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(tmp_path, manifest_path)
    return manifest_path


def next_shard_name(dataset_dir) -> str:
    """Return the next free ``shard-XXXXX.npz`` name for a dataset directory."""
    root = Path(dataset_dir)
    index = 0
    if root.exists():
        existing = sorted(p.name for p in root.glob("shard-*.npz"))
        if existing:
            try:
                index = int(existing[-1].split("-")[-1].split(".")[0]) + 1
            except ValueError:
                index = len(existing)
    return f"shard-{index:05d}.npz"


def write_shard(output_dir, frames: List[np.ndarray], actions: List[int], ram: List[np.ndarray],
                terminated: List[bool], truncated: List[bool], metadata: Dict[str, object] | None = None,
                shard_name: str | None = None, provenance: Dict[str, object] | None = None) -> Path:
    """Write one checksummed shard, appending it to the dataset manifest.

    ``provenance`` documents how this shard was produced; it is stored per shard so
    an aggregated DAgger dataset can mix teacher and on-policy shards while keeping
    each sample's origin recoverable.
    """
    if not frames or not (len(frames) == len(actions) == len(ram) == len(terminated) == len(truncated)):
        raise ValueError("trajectory arrays must be non-empty and have equal lengths")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    shard_path = root / (shard_name or next_shard_name(root))

    np.savez_compressed(
        shard_path,
        frames=np.asarray(frames, dtype=np.uint8),
        actions=np.asarray(actions, dtype=np.uint8),
        ram=np.asarray(ram, dtype=np.uint8),
        terminated=np.asarray(terminated, dtype=np.bool_),
        truncated=np.asarray(truncated, dtype=np.bool_),
    )

    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = load_manifest(root)
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported trajectory schema: {manifest.get('schema_version')}")
        manifest.setdefault("shards", [])
        manifest.setdefault("metadata", {})
        if metadata:
            manifest["metadata"].update(metadata)
    else:
        manifest = {"schema_version": SCHEMA_VERSION, "shards": [], "metadata": metadata or {}}

    entry: Dict[str, object] = {
        "path": shard_path.name,
        "samples": len(actions),
        "sha256": checksum_file(shard_path),
    }
    if provenance:
        entry["provenance"] = dict(provenance)
    manifest["shards"].append(entry)
    write_manifest(root, manifest)
    return shard_path


def add_shard_file(dataset_dir, source_path, provenance: Dict[str, object] | None = None) -> Path:
    """Copy an existing shard into a dataset directory and register its checksum."""
    root = Path(dataset_dir)
    root.mkdir(parents=True, exist_ok=True)
    source = Path(source_path)
    with np.load(source) as data:
        samples = int(len(data["actions"]))
    target = root / source.name
    shutil.copyfile(source, target)

    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = load_manifest(root)
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported trajectory schema: {manifest.get('schema_version')}")
        manifest.setdefault("shards", [])
        manifest.setdefault("metadata", {})
    else:
        manifest = {"schema_version": SCHEMA_VERSION, "shards": [], "metadata": {}}

    entry: Dict[str, object] = {
        "path": target.name,
        "samples": samples,
        "sha256": checksum_file(target),
    }
    if provenance:
        entry["provenance"] = dict(provenance)
    manifest["shards"].append(entry)
    write_manifest(root, manifest)
    return target


def slice_dataset(source_dir, target_dir, start: int = 0, stop: int | None = None) -> Dict[str, object]:
    """Write samples ``[start:stop)`` of a single-shard dataset into a new directory.

    A recorded episode is one shard, so measuring anything on data the weights never
    saw means materialising the two halves as separate, checksummed datasets. The
    slice keeps the source provenance and records the offsets it came from, so a
    held-out number still traces back to the same bytes as the training shard.

    Callers slicing at an action cadence should put the boundary on a cadence
    multiple: the sampled chunk boundaries are ``frames[::cadence]``, so an unaligned
    boundary shifts every held-out decision relative to the teacher's own chunks.

    Returns:
        summary dict with the sample range, the number of shards written and the
        target directory.
    """
    shards = list(iter_dataset(source_dir))
    if len(shards) != 1:
        raise ValueError(f"slice_dataset needs a single-shard dataset, got {len(shards)}")
    shard = shards[0]
    total = len(shard["actions"])
    start = int(start)
    end = total if stop is None else int(stop)
    if not (0 <= start < end <= total):
        raise ValueError(f"sample range [{start}, {end}) is outside a {total}-sample shard")

    source_manifest = load_manifest(source_dir)
    source_entry = source_manifest["shards"][0]
    provenance = {
        **source_entry.get("provenance", {}),
        "slice": {"source_sha256": source_entry["sha256"], "start": start, "stop": end,
                  "source_samples": total},
        "samples": end - start,
    }
    write_shard(
        target_dir,
        list(shard["frames"][start:end]),
        list(shard["actions"][start:end]),
        list(shard["ram"][start:end]),
        list(shard["terminated"][start:end]),
        list(shard["truncated"][start:end]),
        metadata=source_manifest.get("metadata", {}),
        provenance=provenance,
    )
    return {
        "source": str(Path(source_dir).resolve()),
        "target": str(Path(target_dir).resolve()),
        "start": start,
        "stop": end,
        "samples": end - start,
        "source_sha256": source_entry["sha256"],
    }


def iter_dataset(dataset_dir) -> Iterator[Dict[str, np.ndarray]]:
    """Yield verified shard payloads for a dataset directory."""
    root = Path(dataset_dir)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported trajectory schema: {manifest.get('schema_version')}")
    for shard in manifest.get("shards", []):
        path = root / shard["path"]
        if checksum_file(path) != shard["sha256"]:
            raise ValueError(f"trajectory checksum mismatch: {path}")
        with np.load(path) as data:
            payload = {name: data[name].copy() for name in data.files}
        expected = int(shard["samples"])
        if len(payload["actions"]) != expected:
            raise ValueError(f"trajectory sample count mismatch: {path}")
        yield payload


def dataset_provenance(dataset_dir) -> Dict[str, object]:
    """Summarize a dataset for experiment reports, verifying every checksum."""
    manifest = load_manifest(dataset_dir)
    shards = []
    total_samples = 0
    for shard in manifest.get("shards", []):
        path = Path(dataset_dir) / shard["path"]
        digest = checksum_file(path)
        if digest != shard["sha256"]:
            raise ValueError(f"trajectory checksum mismatch: {path}")
        total_samples += int(shard["samples"])
        entry = {
            "path": shard["path"],
            "samples": int(shard["samples"]),
            "sha256": shard["sha256"],
        }
        if "provenance" in shard:
            entry["provenance"] = shard["provenance"]
        shards.append(entry)
    return {
        "schema_version": manifest.get("schema_version"),
        "shard_count": len(shards),
        "total_samples": total_samples,
        "shards": shards,
        "metadata": manifest.get("metadata", {}),
        "checksums_verified": True,
    }
