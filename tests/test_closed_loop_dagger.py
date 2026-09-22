import json
import unittest
from pathlib import Path

import numpy as np

from closed_loop_dagger import OFFLINE_ENV_KIND, baseline_verdict, build_arg_parser, build_report
from trajectory import load_manifest, write_shard


TEACHER_PX_PER_FRAME = 8


def write_small_teacher_dataset(dataset_dir, count=40):
    """Small env-free teacher shard so the runner test stays fast and deterministic.

    The teacher advances faster than a candidate that simply holds RUN, so the default
    divergence thresholds fire and the DAgger round has genuine recovery windows.
    """
    frames = [np.zeros((240, 256, 3), dtype=np.uint8) for _ in range(count)]
    actions = [1 if i % 2 else 3 for i in range(count)]
    ram = [np.zeros(0x800, dtype=np.uint8) for _ in range(count)]
    for index, snapshot in enumerate(ram):
        x_pos = 40 + TEACHER_PX_PER_FRAME * index
        snapshot[0x006D] = (x_pos // 256) % 256
        snapshot[0x0086] = x_pos % 256
    write_shard(
        dataset_dir,
        frames,
        actions,
        ram,
        [False] * count,
        [False] * count,
        metadata={"env_kind": "offline_synthetic"},
        provenance={"origin": "teacher", "source": "test_teacher", "samples": count},
    )


def runner_args(tmpdir, **overrides):
    argv = [
        "--offline-env",
        "--teacher-dataset", str(Path(tmpdir) / "teacher"),
        "--save-path", str(Path(tmpdir) / "missing_checkpoint.pth"),
        "--runs-dir", str(Path(tmpdir) / "runs"),
        "--train-seeds", "0",
        "--eval-seeds", "42",
        "--episodes", "1",
        "--max-steps", "40",
        "--rollout-steps", "40",
        "--settle-steps", "1",
        "--action-cadence", "4",
        "--epochs", "1",
    ]
    for key, value in overrides.items():
        argv.extend([f"--{key.replace('_', '-')}", str(value)])
    return build_arg_parser().parse_args(argv)


class TestBaselineVerdict(unittest.TestCase):

    def test_verdict_never_meets_the_gate_without_the_rom(self):
        evaluation = {
            "model_only": True,
            "best_x": 900,
            "completion_rate": 1.0,
            "env_kind": OFFLINE_ENV_KIND,
        }
        verdict = baseline_verdict(evaluation)
        self.assertTrue(verdict["best_x_above_baseline"])
        self.assertTrue(verdict["improved_completion_behavior"])
        # Completion on the synthetic stand-in is plumbing validation only.
        self.assertFalse(verdict["p0_gate_met"])

    def test_verdict_flags_model_x_at_or_below_baseline(self):
        verdict = baseline_verdict({
            "model_only": True, "best_x": 400, "completion_rate": 0.0, "env_kind": "stable-retro",
        })
        self.assertFalse(verdict["best_x_above_baseline"])
        self.assertFalse(verdict["improved_completion_behavior"])
        self.assertFalse(verdict["p0_gate_met"])


class TestRunnerOffline(unittest.TestCase):

    def test_baseline_mode_records_a_report_without_iterations(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            report, report_path = build_report(runner_args(tmpdir, mode="baseline"))

            self.assertEqual(report["mode"], "baseline")
            self.assertEqual(report["environment"]["env_kind"], OFFLINE_ENV_KIND)
            self.assertTrue(report["environment"]["results_are_pipeline_validation_only"])
            self.assertIsNone(report["environment"]["rom_sha256"])
            self.assertEqual(report["iterations"], [])
            self.assertEqual(report["verdict"]["stop_reason"], "baseline_only")
            self.assertFalse(report["verdict"]["p0_gate_met"])
            self.assertFalse(report["verdict"]["gate_eligible_env"])
            self.assertTrue(report_path.exists())
            self.assertTrue((Path(tmpdir) / "runs" / "report.md").exists())
            self.assertEqual(json.loads(report_path.read_text())["mode"], "baseline")

    def test_dagger_mode_runs_a_bounded_round_with_full_provenance(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            teacher_dir = Path(tmpdir) / "teacher"
            write_small_teacher_dataset(teacher_dir)
            report, _ = build_report(runner_args(tmpdir, mode="dagger", iterations=1))

            self.assertEqual(len(report["iterations"]), 1)
            self.assertIn(report["verdict"]["stop_reason"], {"iteration_budget_exhausted", "model_only_completion_detected"})
            self.assertTrue(report["verdict"]["gate_eligible_env"] is False)
            self.assertFalse(report["verdict"]["p0_gate_met"])

            teacher_info = report["teacher_dataset"]
            self.assertTrue(teacher_info["checksums_verified"])
            self.assertEqual(teacher_info["shards"][0]["provenance"]["origin"], "teacher")
            self.assertEqual(teacher_info["shards"][0]["provenance"]["source"], "test_teacher")

            iteration = report["iterations"][0]
            self.assertTrue(iteration["rollouts"]["model_only"])
            self.assertEqual(iteration["rollouts"]["deaths"], 0)
            self.assertGreaterEqual(iteration["rollouts"]["recovery_windows"], 1)

            origins = {shard["provenance"]["origin"] for shard in iteration["dataset"]["shards"]}
            self.assertEqual(origins, {"teacher", "dagger_recovery_windows"})
            self.assertTrue(iteration["dataset"]["checksums_verified"])

            pretraining = iteration["pretraining"]
            self.assertEqual(pretraining["action_cadence"], 4)
            self.assertEqual(pretraining["settle_steps"], 1)
            self.assertEqual(pretraining["episode_boundary_resets"], len(iteration["dataset"]["shards"]))
            self.assertEqual(set(pretraining["samples_by_origin"]), {"teacher", "dagger_recovery_windows"})
            # Supervised samples are strided at the action cadence, but every sample in
            # the aggregated dataset is still attributed to the shard that produced it.
            self.assertEqual(
                sum(pretraining["samples_by_origin"].values()),
                iteration["dataset"]["total_samples"],
            )
            self.assertLessEqual(pretraining["samples"], iteration["dataset"]["total_samples"])

            self.assertTrue(iteration["evaluation"]["model_only"])
            self.assertEqual(iteration["evaluation"]["seeds"], [42])
            self.assertTrue(iteration["evaluation"]["cadence_consistent"])
            self.assertEqual(iteration["evaluation"]["total_assisted_jump_frames"], 0)

            self.assertEqual(report["config"]["eval_seeds"], [42])
            self.assertEqual(report["config"]["train_seeds"], [0])

            # The iteration dataset is on disk and its manifest references a real checkpoint.
            iteration_dir = Path(tmpdir) / "runs" / "iter-01"
            self.assertTrue((iteration_dir / "checkpoint.pth").exists())
            self.assertTrue((iteration_dir / "dataset" / "manifest.json").exists())
            self.assertEqual(
                Path(iteration["dataset"]["metadata"]["teacher_dataset"]),
                teacher_dir.resolve(),
            )
            self.assertEqual(len(load_manifest(iteration_dir / "dataset")["shards"]), 2)

    def test_dagger_mode_requires_a_teacher_dataset(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(SystemExit):
                build_report(runner_args(tmpdir, mode="dagger", iterations=1))


if __name__ == "__main__":
    unittest.main()
