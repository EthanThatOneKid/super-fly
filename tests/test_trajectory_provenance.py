import tempfile
import unittest
from pathlib import Path

import numpy as np

from trajectory import (
    add_shard_file,
    dataset_provenance,
    iter_dataset,
    load_manifest,
    slice_dataset,
    write_shard,
)


def _shard(dataset_dir, samples=3, name=None, provenance=None):
    frames = [np.full((2, 3, 3), fill_value=i, dtype=np.uint8) for i in range(samples)]
    actions = [1] * samples
    ram = [np.zeros(0x800, dtype=np.uint8) for _ in range(samples)]
    return write_shard(
        dataset_dir,
        frames,
        actions,
        ram,
        [False] * samples,
        [False] * samples,
        provenance=provenance,
        shard_name=name,
    )


class TestTrajectoryProvenance(unittest.TestCase):

    def test_shard_provenance_is_recorded_per_shard(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _shard(tmpdir, provenance={"origin": "teacher", "source_sha256": "abc"})
            _shard(tmpdir, provenance={"origin": "dagger_recovery_windows", "window_frames": 8})

            manifest = load_manifest(tmpdir)
            origins = [shard["provenance"]["origin"] for shard in manifest["shards"]]
            self.assertEqual(origins, ["teacher", "dagger_recovery_windows"])

            info = dataset_provenance(tmpdir)
            self.assertTrue(info["checksums_verified"])
            self.assertEqual(info["shard_count"], 2)
            self.assertEqual(info["total_samples"], 6)
            self.assertEqual(info["shards"][0]["provenance"]["origin"], "teacher")

    def test_add_shard_file_copies_bytes_and_keeps_checksum(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "source"
            target_dir = Path(tmp) / "target"
            shard_path = _shard(source_dir, samples=4)

            copied = add_shard_file(
                target_dir,
                shard_path,
                provenance={"origin": "teacher", "source_sha256": "deadbeef", "source_samples": 4},
            )

            self.assertEqual(copied.read_bytes(), shard_path.read_bytes())
            entry = load_manifest(target_dir)["shards"][0]
            self.assertEqual(entry["samples"], 4)
            self.assertEqual(entry["provenance"]["origin"], "teacher")
            self.assertEqual(len(list(iter_dataset(target_dir))), 1)

    def test_dataset_provenance_rejects_tampered_shard(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            shard_path = _shard(tmpdir)
            with shard_path.open("ab") as handle:
                handle.write(b"tamper")

            with self.assertRaises(ValueError):
                dataset_provenance(tmpdir)


class TestDatasetSlicing(unittest.TestCase):
    """Held-out halves of an episode.

    One recorded episode is one shard, so measuring on data the weights never trained
    on means materialising the halves as separate, checksummed datasets.
    """

    def test_slice_materialises_exactly_the_requested_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            head, tail = Path(tmp) / "head", Path(tmp) / "tail"
            _shard(source, samples=30)

            summary = slice_dataset(source, tail, start=20, stop=30)
            slice_dataset(source, head, start=0, stop=20)

            self.assertEqual(summary["samples"], 10)
            self.assertEqual(len(list(iter_dataset(head))[0]["actions"]), 20)
            tail_frames = list(iter_dataset(tail))[0]["frames"]
            self.assertEqual(len(tail_frames), 10)
            # The frames carry their sample index, so this is the requested range and
            # not merely a correctly-sized slice of the wrong frames.
            self.assertEqual(int(tail_frames[0][0, 0, 0]), 20)
            self.assertEqual(int(tail_frames[-1][0, 0, 0]), 29)

    def test_slice_keeps_provenance_and_records_the_offsets(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp) / "source", Path(tmp) / "target"
            _shard(source, samples=10, provenance={"origin": "teacher", "env_kind": "stable-retro"})
            source_sha = load_manifest(source)["shards"][0]["sha256"]

            slice_dataset(source, target, start=4, stop=10)

            provenance = load_manifest(target)["shards"][0]["provenance"]
            self.assertEqual(provenance["origin"], "teacher")
            self.assertEqual(provenance["env_kind"], "stable-retro")
            self.assertEqual(provenance["slice"], {
                "source_sha256": source_sha, "start": 4, "stop": 10, "source_samples": 10,
            })
            self.assertTrue(dataset_provenance(target)["checksums_verified"])

    def test_impossible_ranges_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            _shard(source, samples=5)
            for start, stop in ((3, 2), (-1, 3), (0, 6), (5, 5), (5, None)):
                with self.assertRaises(ValueError):
                    slice_dataset(source, Path(tmp) / "out", start=start, stop=stop)

    def test_a_multi_shard_source_is_refused(self):
        """Sample offsets are only meaningful within one episode's shard."""
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            _shard(source, samples=3)
            _shard(source, samples=3)
            with self.assertRaises(ValueError):
                slice_dataset(source, Path(tmp) / "out", start=0, stop=2)


if __name__ == "__main__":
    unittest.main()
