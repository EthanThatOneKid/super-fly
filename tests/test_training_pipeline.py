import json
import os
import tempfile
import unittest

import numpy as np

import torch

from pretrain import save_checkpoint, target_rate_vector, pretrain_motor_layer
from simulation import Simulation
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

    def test_sequence_aware_pretraining_and_settling_bounds(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frames = [np.zeros((240, 256, 3), dtype=np.uint8) for _ in range(5)]
            actions = [1, 1, 3, 3, 1]
            ram = [np.zeros(0x800, dtype=np.uint8) for _ in range(5)]
            write_shard(tmpdir, frames, actions, ram, [False]*5, [False]*5, {"level": "Level1-1"})

            # Test invalid settle_steps bounds
            with self.assertRaises(ValueError):
                pretrain_motor_layer(tmpdir, settle_steps=0)
            with self.assertRaises(ValueError):
                pretrain_motor_layer(tmpdir, settle_steps=11)

            # Test valid sequence-aware pretraining
            model, metadata = pretrain_motor_layer(tmpdir, epochs=2, settle_steps=3, seed=42)
            self.assertEqual(metadata["settle_steps"], 3)
            self.assertEqual(metadata["samples"], 5)
            self.assertEqual(metadata["updates"], 10)

    def test_pretraining_calibrates_the_macro_jump_margin(self):
        """Pretraining has to publish a calibrated decoder, not just weights.

        The readout's jump/run evidence is offset, so the margin that separates the
        two chunk types must come from the data and travel with the checkpoint.
        Otherwise evaluation rebuilds an uncalibrated decoder at margin 0 and every
        learned jump decision is discarded.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            frames = [np.zeros((240, 256, 3), dtype=np.uint8) for _ in range(8)]
            actions = [1, 1, 3, 3, 1, 1, 3, 1]
            ram = [np.zeros(0x800, dtype=np.uint8) for _ in range(8)]
            write_shard(tmpdir, frames, actions, ram, [False] * 8, [False] * 8, {"level": "Level1-1"})

            model, metadata = pretrain_motor_layer(tmpdir, epochs=1, settle_steps=1, stride=1, seed=42)
            decoder = metadata["decoder"]
            calibration = decoder["calibration"]
            self.assertEqual(calibration["method"], "balanced_accuracy")
            self.assertEqual(calibration["samples"], 8 * metadata["decoder"]["calibration"]["replays"])
            self.assertEqual(metadata["jump_margin"], decoder["jump_margin"])
            self.assertEqual(decoder["chunk_frames"], 1)
            self.assertEqual(decoder["max_chunk_frames"], 30)

            # A single-class shard cannot be calibrated, and says so rather than
            # quietly reporting margin 0 as if it were a decision.
            uniform = os.path.join(tmpdir, "uniform")
            write_shard(uniform, frames, [1] * 8, ram, [False] * 8, [False] * 8, {"level": "Level1-1"})
            _, uniform_metadata = pretrain_motor_layer(uniform, epochs=1, settle_steps=1, stride=1)
            self.assertEqual(
                uniform_metadata["decoder"]["calibration"]["method"], "insufficient_classes"
            )

    def test_calibrated_margin_is_honoured_when_the_checkpoint_is_loaded(self):
        """``Simulation`` must adopt the checkpoint's decoder configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            frames = [np.zeros((240, 256, 3), dtype=np.uint8) for _ in range(8)]
            actions = [1, 1, 3, 3, 1, 1, 3, 1]
            ram = [np.zeros(0x800, dtype=np.uint8) for _ in range(8)]
            write_shard(tmpdir, frames, actions, ram, [False] * 8, [False] * 8, {"level": "Level1-1"})

            model, metadata = pretrain_motor_layer(tmpdir, epochs=1, settle_steps=1, stride=4, seed=42)
            checkpoint = os.path.join(tmpdir, "checkpoint.pth")
            save_checkpoint(model, checkpoint, metadata)

            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            published = payload["policy_config"]["macro_decoder"]
            self.assertEqual(published["jump_margin"], metadata["jump_margin"])
            self.assertEqual(published["chunk_frames"], 4)
            self.assertEqual(payload["policy_config"]["action_cadence"], 4)

            sim = Simulation(
                save_path=checkpoint,
                bootstrap_episodes=0,
                max_bootstrap_step=0,
                policy="macro",
                states=["Level1-1"],
                settle_steps=3,
                action_cadence=15,
                seed=42,
                runs_dir=os.path.join(tmpdir, "runs"),
            )
            self.assertEqual(sim.macro_jump_margin, metadata["jump_margin"])
            self.assertEqual(sim.macro_decoder.jump_margin, metadata["jump_margin"])
            self.assertEqual(sim.action_cadence, 4)
            self.assertEqual(sim.macro_decoder.chunk_frames, 4)

    def test_calibration_replays_are_rejected_when_below_one(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frames = [np.zeros((240, 256, 3), dtype=np.uint8) for _ in range(4)]
            ram = [np.zeros(0x800, dtype=np.uint8) for _ in range(4)]
            write_shard(tmpdir, frames, [1, 3, 1, 3], ram, [False] * 4, [False] * 4, {"level": "Level1-1"})
            with self.assertRaises(ValueError):
                pretrain_motor_layer(tmpdir, epochs=1, settle_steps=1, calibration_replays=0)


if __name__ == "__main__":
    unittest.main()
