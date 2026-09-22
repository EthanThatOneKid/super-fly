import json
import os
import tempfile
import unittest

import numpy as np

import torch

from connectome import DrosophilaConnectomeSNN
from pretrain import output_errors, save_checkpoint, target_rate_vector, pretrain_motor_layer
from simulation import Simulation
from trajectory import iter_dataset, slice_dataset, write_shard


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


class TestVisualPathwayPretraining(unittest.TestCase):
    """Training the visual pathway, not only the motor readout.

    The closed-loop ceiling was structural: pretraining moved ``layer3_4`` and nothing
    else, so every DAgger round refit a linear probe on a frozen random connectome and
    the features feeding the decision never changed. These tests pin that the new mode
    is real (the visual layers move), that it is opt-in (frozen stays exactly at its
    initialization), and that the two modes are measurably different.
    """

    SAMPLES = 12
    ACTIONS = [1, 3, 1, 1, 3, 1, 3, 3, 1, 1, 3, 1]

    def _vision_shard(self, dataset_dir, env_kind="stable-retro"):
        """A shard whose frames carry real visual structure (not blank frames)."""
        frames = [
            np.random.RandomState(i).randint(0, 255, (64, 64, 3), dtype=np.uint8)
            for i in range(self.SAMPLES)
        ]
        ram = [np.zeros(0x800, dtype=np.uint8) for _ in range(self.SAMPLES)]
        write_shard(dataset_dir, frames, list(self.ACTIONS), ram,
                    [False] * self.SAMPLES, [False] * self.SAMPLES, {"env_kind": env_kind})
        return dataset_dir

    def test_frozen_mode_leaves_the_visual_pathway_at_its_initialization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._vision_shard(tmpdir)
            torch.manual_seed(42)
            reference = DrosophilaConnectomeSNN()

            model, metadata = pretrain_motor_layer(
                tmpdir, epochs=2, stride=1, settle_steps=2, seed=42
            )

            self.assertEqual(metadata["visual_pathway"], "frozen")
            self.assertEqual(metadata["trained_layers"], ["layer3_4"])
            self.assertIsNone(metadata["visual_learning_rate"])
            for name in ("layer1_2", "layer2_3", "feedback_3_2"):
                # Checked against a fresh model under the same seed, so this is the
                # initialization heuristic rather than merely "small delta".
                torch.testing.assert_close(
                    getattr(model, name).weight, getattr(reference, name).weight
                )
                self.assertEqual(metadata["weight_deltas"][name]["l2_delta"], 0.0, name)
            self.assertGreater(metadata["weight_deltas"]["layer3_4"]["l2_delta"], 0.0)

    def test_visual_pathway_mode_trains_the_visual_layers_under_one_rule(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._vision_shard(tmpdir)

            model, metadata = pretrain_motor_layer(
                tmpdir, epochs=2, stride=1, settle_steps=2, seed=42,
                visual_pathway="linear_feedback", visual_lr=0.005,
            )

            self.assertEqual(metadata["visual_pathway"], "linear_feedback")
            self.assertEqual(metadata["visual_learning_rate"], 0.005)
            self.assertEqual(
                metadata["trained_layers"],
                ["feedback_3_2", "layer1_2", "layer2_3", "layer3_4"],
            )
            for name in ("layer1_2", "layer2_3", "feedback_3_2"):
                self.assertGreater(metadata["weight_deltas"][name]["relative"], 0.0, name)
                weight = getattr(model, name).weight
                self.assertTrue(bool(torch.isfinite(weight).all()), name)
                # The readout's invariants hold for every trained layer.
                self.assertLessEqual(float(weight.detach().abs().max()), 3.0, name)
                self.assertLess(float(weight.detach().mean(dim=1).abs().max()), 1e-4, name)

    def test_the_two_modes_reach_different_models(self):
        """A flag that changed nothing would make the comparison meaningless."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._vision_shard(tmpdir)
            frozen, _ = pretrain_motor_layer(tmpdir, epochs=1, stride=1, settle_steps=1, seed=42)
            visual, _ = pretrain_motor_layer(
                tmpdir, epochs=1, stride=1, settle_steps=1, seed=42,
                visual_pathway="linear_feedback", visual_lr=0.005,
            )
            self.assertGreater(
                float((frozen.layer1_2.weight - visual.layer1_2.weight).norm().detach()), 0.0
            )
            # The readout sees different features, so its own weights diverge too.
            self.assertFalse(
                torch.allclose(frozen.layer3_4.weight, visual.layer3_4.weight)
            )

    def test_credit_for_each_layer_is_chained_through_the_transposed_weights(self):
        torch.manual_seed(7)
        model = DrosophilaConnectomeSNN()
        motor_error = torch.tensor([0.4, -0.2, 0.1, 0.3])

        errors = output_errors(model, motor_error)

        torch.testing.assert_close(errors["layer3_4"], motor_error)
        central = model.layer3_4.weight.t().matmul(motor_error)
        torch.testing.assert_close(errors["layer2_3"], central)
        torch.testing.assert_close(errors["layer1_2"], model.layer2_3.weight.t().matmul(central))
        # Both writers into the optic lobe read the error measured at that layer.
        torch.testing.assert_close(errors["feedback_3_2"], errors["layer1_2"])
        self.assertEqual(tuple(errors["layer1_2"].shape), (model.num_optic_lobe,))

    def test_visual_pathway_configuration_is_validated(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._vision_shard(tmpdir)
            with self.assertRaises(ValueError):
                pretrain_motor_layer(tmpdir, epochs=1, settle_steps=1,
                                     visual_pathway="surrogate_gradient")
            with self.assertRaises(ValueError):
                pretrain_motor_layer(tmpdir, epochs=1, settle_steps=1,
                                     visual_pathway="linear_feedback", visual_lr=0.0)

    def test_held_out_reporting_applies_the_training_margin(self):
        """One episode is one shard, so the held-out half is materialised first."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source = os.path.join(tmpdir, "source")
            train_dir = os.path.join(tmpdir, "train")
            eval_dir = os.path.join(tmpdir, "eval")
            self._vision_shard(source)
            slice_dataset(source, train_dir, 0, 8)
            slice_dataset(source, eval_dir, 8, self.SAMPLES)

            _, metadata = pretrain_motor_layer(
                train_dir, epochs=1, stride=1, settle_steps=1, seed=42,
                report_dataset_dir=eval_dir,
            )

            training = metadata["decoder"]["calibration"]
            held_out = metadata["held_out_decision_quality"]
            self.assertEqual(training["method"], "balanced_accuracy")
            self.assertEqual(held_out["method"], "fixed_margin")
            self.assertEqual(held_out["margin"], training["margin"])
            self.assertEqual(held_out["decisions_per_replay"], 4)
            self.assertEqual(held_out["samples"], 4 * training["replays"])
            self.assertEqual(held_out["env_kind"], "stable-retro")
            self.assertTrue(held_out["dataset"]["checksums_verified"])

    def test_a_held_out_dataset_from_another_environment_is_refused(self):
        """A synthetic shard has the same schema, so only provenance can catch it."""
        with tempfile.TemporaryDirectory() as tmpdir:
            train_dir = os.path.join(tmpdir, "train")
            self._vision_shard(train_dir)
            synthetic = os.path.join(tmpdir, "synthetic")
            self._vision_shard(synthetic, env_kind="offline_synthetic")

            with self.assertRaises(ValueError):
                pretrain_motor_layer(
                    train_dir, epochs=1, stride=1, settle_steps=1,
                    report_dataset_dir=synthetic,
                )


if __name__ == "__main__":
    unittest.main()
