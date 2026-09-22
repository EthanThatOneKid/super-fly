import tempfile
import unittest
from pathlib import Path

import numpy as np

from trajectory import add_shard_file, dataset_provenance, iter_dataset, load_manifest, write_shard


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


if __name__ == "__main__":
    unittest.main()
