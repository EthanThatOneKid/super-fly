import json
import os
import tempfile
import unittest

import numpy as np

from pretrain import target_rate_vector
from trajectory import iter_dataset, write_shard


class TestTrainingPipeline(unittest.TestCase):
    def test_target_rate_vector(self):
        target = target_rate_vector(3)
        np.testing.assert_allclose(target.numpy(), [0.05, 0.05, 0.05, 0.9])

    def test_trajectory_manifest_and_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frames = [np.zeros((2, 3, 3), dtype=np.uint8), np.ones((2, 3, 3), dtype=np.uint8)]
            actions = [1, 3]
            ram = [np.zeros(0x800, dtype=np.uint8), np.ones(0x800, dtype=np.uint8)]
            shard = write_shard(tmpdir, frames, actions, ram, [False, True], [False, False], {"level": "Level1-1"})

            with open(os.path.join(tmpdir, "manifest.json")) as handle:
                manifest = json.load(handle)
            self.assertEqual(manifest["schema_version"], 1)
            self.assertEqual(manifest["shards"][0]["samples"], 2)
            self.assertEqual(shard.name, "shard-00000.npz")

            rows = list(iter_dataset(tmpdir))
            self.assertEqual(rows[0]["actions"].tolist(), actions)
            self.assertEqual(rows[0]["frames"].shape, (2, 2, 3, 3))
            self.assertEqual(rows[0]["ram"].shape, (2, 0x800))

    def test_trajectory_checksum_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            write_shard(
                tmpdir,
                [np.zeros((1, 1, 3), dtype=np.uint8)],
                [1],
                [np.zeros(0x800, dtype=np.uint8)],
                [False],
                [False],
            )
            with open(os.path.join(tmpdir, "shard-00000.npz"), "ab") as handle:
                handle.write(b"tampered")
            with self.assertRaises(ValueError):
                list(iter_dataset(tmpdir))


if __name__ == "__main__":
    unittest.main()
