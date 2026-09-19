import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Iterable, Iterator, List

import numpy as np

SCHEMA_VERSION = 1


def _checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_shard(output_dir: str, frames: List[np.ndarray], actions: List[int], ram: List[np.ndarray],
                terminated: List[bool], truncated: List[bool], metadata: Dict[str, object] | None = None) -> Path:
    if not frames or not (len(frames) == len(actions) == len(ram) == len(terminated) == len(truncated)):
        raise ValueError("trajectory arrays must be non-empty and have equal lengths")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    shard_path = root / "shard-00000.npz"
    np.savez_compressed(
        shard_path,
        frames=np.asarray(frames, dtype=np.uint8),
        actions=np.asarray(actions, dtype=np.uint8),
        ram=np.asarray(ram, dtype=np.uint8),
        terminated=np.asarray(terminated, dtype=np.bool_),
        truncated=np.asarray(truncated, dtype=np.bool_),
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "shards": [{"path": shard_path.name, "samples": len(actions), "sha256": _checksum(shard_path)}],
        "metadata": metadata or {},
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return shard_path


def iter_dataset(dataset_dir: str) -> Iterator[Dict[str, np.ndarray]]:
    root = Path(dataset_dir)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported trajectory schema: {manifest.get('schema_version')}")
    for shard in manifest.get("shards", []):
        path = root / shard["path"]
        if _checksum(path) != shard["sha256"]:
            raise ValueError(f"trajectory checksum mismatch: {path}")
        with np.load(path) as data:
            payload = {name: data[name].copy() for name in data.files}
        expected = int(shard["samples"])
        if len(payload["actions"]) != expected:
            raise ValueError(f"trajectory sample count mismatch: {path}")
        yield payload
