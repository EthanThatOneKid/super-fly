import tempfile
import unittest
from pathlib import Path

import numpy as np

from dagger import (
    TeacherLabeler,
    aggregate_rollout_metadata,
    build_dagger_dataset,
    collect_on_policy_rollout,
    write_rollout_shard,
    x_pos_from_ram,
)
from offline_env import RAM_SIZE, OfflineMarioEnv
from simulation import Simulation
from trajectory import dataset_provenance, iter_dataset, load_manifest, write_shard

TEACHER_ACTIONS = (1, 3)


def ram_at(x_pos):
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[0x006D] = (x_pos // 256) % 256
    ram[0x0086] = x_pos % 256
    return ram


def synthetic_teacher(count=400, start=40, step=2):
    """Build (x_pos, action) teacher arrays with monotone progress.

    ``step`` is how many pixels the teacher advances per frame. The rollout tests use
    a fast teacher so a candidate holding RUN cannot simply keep pace with it.
    """
    xs = [start + step * i for i in range(count)]
    actions = [TEACHER_ACTIONS[i % len(TEACHER_ACTIONS)] for i in range(count)]
    return xs, actions


def write_teacher_dataset(dataset_dir, count=40):
    """Tiny teacher shard: enough for labelling and aggregation, cheap to pretrain on."""
    frames = [np.zeros((240, 256, 3), dtype=np.uint8) for _ in range(count)]
    xs, actions = synthetic_teacher(count=count)
    ram = [ram_at(x) for x in xs]
    write_shard(
        dataset_dir,
        frames,
        actions,
        ram,
        [False] * count,
        [False] * count,
        metadata={"env_kind": "offline_synthetic"},
        provenance={"origin": "teacher", "env_kind": "offline_synthetic"},
    )
    return xs, actions


def make_sim(tmpdir, action_cadence=4):
    return Simulation(
        save_path=str(Path(tmpdir) / "missing_checkpoint.pth"),
        bootstrap_episodes=0,
        max_bootstrap_step=0,
        policy="macro",
        states=["Level1-1"],
        settle_steps=1,
        action_cadence=action_cadence,
        seed=0,
        runs_dir=str(Path(tmpdir) / "runs"),
    )


class TestTeacherLabeler(unittest.TestCase):

    def test_x_pos_from_ram_combines_page_and_sub_page(self):
        self.assertEqual(x_pos_from_ram(ram_at(594)), 594)

    def test_label_matches_progress_not_step(self):
        xs, actions = synthetic_teacher(count=100)
        labeler = TeacherLabeler(xs, actions)

        self.assertEqual(labeler.label(40).action, actions[0])
        self.assertEqual(labeler.label(50).action, actions[5])
        # Progress between recorded samples snaps back to the last reached sample.
        self.assertEqual(labeler.label(51).index, 5)
        self.assertLessEqual(labeler.label(51).teacher_x, 51)
        self.assertEqual(labeler.final_x, xs[-1])

    def test_schedule_lag_and_progress_deficit_are_non_negative(self):
        xs, actions = synthetic_teacher(count=200)
        labeler = TeacherLabeler(xs, actions)

        # Reaching the teacher's progress in exactly the teacher's own frame count
        # is on schedule; every extra frame spent getting there is lag.
        self.assertEqual(labeler.schedule_lag(step=labeler.label(100).index, x_pos=100), 0)
        self.assertEqual(labeler.schedule_lag(step=labeler.label(100).index + 5, x_pos=100), 5)
        self.assertEqual(labeler.schedule_lag(step=200, x_pos=100), 200 - labeler.label(100).index)
        self.assertEqual(labeler.progress_deficit(step=1, x_pos=10_000), 0)
        self.assertGreater(labeler.progress_deficit(step=200, x_pos=100), 0)
        # Beyond the recorded trajectory the teacher schedule is clamped, not extrapolated.
        self.assertEqual(labeler.progress_at_step(10_000), xs[-1])

    def test_monotonises_wobbly_progress_for_bisect(self):
        labeler = TeacherLabeler([10, 12, 11, 20], [1, 3, 1, 3])
        self.assertEqual(labeler.teacher_x, [10, 12, 12, 20])
        self.assertEqual(labeler.label(15).action, 1)

    def test_rejects_malformed_teacher_arrays(self):
        with self.assertRaises(ValueError):
            TeacherLabeler([1, 2], [1])
        with self.assertRaises(ValueError):
            TeacherLabeler([], [])

    def test_from_dataset_reads_shard_progress(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            xs, _ = write_teacher_dataset(tmpdir, count=10)
            labeler = TeacherLabeler.from_dataset(tmpdir)
            self.assertEqual(labeler.length, 10)
            self.assertEqual(labeler.final_x, xs[-1])


class TestOnPolicyRollout(unittest.TestCase):

    def test_rollout_is_bounded_model_only_and_labels_windows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            labeler = TeacherLabeler(*synthetic_teacher(step=8))
            sim = make_sim(tmpdir, action_cadence=4)
            env = OfflineMarioEnv(state="Level1-1")
            try:
                result = collect_on_policy_rollout(
                    sim,
                    env,
                    labeler,
                    max_steps=60,
                    recovery_lag_steps=10,
                    recovery_lag_px=8,
                    window_frames=4,
                    max_windows=2,
                    window_cooldown_frames=4,
                    seed=7,
                )
            finally:
                env.close()

            summary = result.summary()
            self.assertLessEqual(result.steps, 60)
            self.assertEqual(result.action_cadence, 4)
            self.assertEqual(result.seed, 7)
            self.assertTrue(summary["model_only"])
            self.assertEqual(result.assisted_jump_frames, 0)
            self.assertEqual(result.bootstrap_episodes, 0)
            self.assertEqual(result.policy, "macro")
            # The candidate may only act through its own macro-chunk decisions.
            self.assertEqual(result.model_jump_frames + result.assisted_jump_frames, result.model_jump_frames)

            # Divergence must be detected and bounded, with complete windows.
            self.assertGreaterEqual(result.divergence_events, 1)
            self.assertGreaterEqual(len(result.windows), 1)
            self.assertLessEqual(len(result.windows), 2)
            for window in result.windows:
                # A window is the trailing frames plus the current one, bounded by
                # window_frames (early divergence fires before the buffer is full).
                self.assertLessEqual(len(window.frames), 5)
                self.assertGreaterEqual(len(window.frames), 1)
                self.assertEqual(len(window.actions), len(window.frames))
                self.assertEqual(len(window.ram), len(window.frames))
                self.assertIn(window.target_action, TEACHER_ACTIONS)
                self.assertIn(window.reason, {"divergence", "unrecoverable_death", "unrecoverable_stagnation"})
            self.assertGreater(summary["max_progress_deficit_px"], 0)

    def test_recovery_windows_are_written_as_checksummed_shards(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            teacher_dir = Path(tmpdir) / "teacher"
            write_teacher_dataset(teacher_dir)
            labeler = TeacherLabeler(*synthetic_teacher(step=8))
            sim = make_sim(tmpdir)
            env = OfflineMarioEnv(state="Level1-1")
            try:
                rollout = collect_on_policy_rollout(
                    sim,
                    env,
                    labeler,
                    max_steps=40,
                    recovery_lag_steps=5,
                    recovery_lag_px=4,
                    window_frames=3,
                    max_windows=2,
                )
                empty_rollout = collect_on_policy_rollout(
                    sim,
                    env,
                    labeler,
                    max_steps=40,
                    recovery_lag_steps=5,
                    recovery_lag_px=4,
                    window_frames=3,
                    max_windows=0,
                )
            finally:
                env.close()

            aggregated = Path(tmpdir) / "aggregated"
            shard_path = write_rollout_shard(aggregated, rollout)
            self.assertIsNotNone(shard_path)
            self.assertIsNone(write_rollout_shard(aggregated, empty_rollout))

            shard = next(iter_dataset(aggregated))
            expected = sum(len(window.frames) for window in rollout.windows)
            self.assertEqual(len(shard["actions"]), expected)
            entry = load_manifest(aggregated)["shards"][0]
            self.assertEqual(entry["provenance"]["origin"], "dagger_recovery_windows")
            self.assertEqual(entry["provenance"]["rollout"]["model_only"], True)

    def test_build_dagger_dataset_mixes_teacher_and_rollout_provenance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            teacher_dir = Path(tmpdir) / "teacher"
            write_teacher_dataset(teacher_dir, count=20)
            labeler = TeacherLabeler(*synthetic_teacher(step=8))
            sim = make_sim(tmpdir)
            env = OfflineMarioEnv(state="Level1-1")
            try:
                rollout = collect_on_policy_rollout(
                    sim, env, labeler, max_steps=40, recovery_lag_px=4, window_frames=3, max_windows=1
                )
            finally:
                env.close()

            aggregated = Path(tmpdir) / "aggregated"
            info = build_dagger_dataset(aggregated, teacher_dir, [rollout], metadata={"iteration": 1})

            self.assertTrue(info["checksums_verified"])
            self.assertEqual(info["shard_count"], 2)
            origins = [shard["provenance"]["origin"] for shard in info["shards"]]
            self.assertEqual(origins, ["teacher", "dagger_recovery_windows"])
            self.assertEqual(info["shards"][0]["provenance"]["origin"], "teacher")
            self.assertEqual(info["metadata"]["iteration"], 1)
            self.assertEqual(info["metadata"]["rollout_shards"], 1)
            self.assertEqual(info["total_samples"], 20 + sum(len(w.frames) for w in rollout.windows))

            # The teacher shard is copied byte-for-byte, so its checksum still verifies.
            teacher_manifest = load_manifest(teacher_dir)
            self.assertEqual(info["shards"][0]["sha256"], teacher_manifest["shards"][0]["sha256"])
            self.assertEqual(dataset_provenance(aggregated)["total_samples"], info["total_samples"])
            self.assertEqual(len(info["shards"]), len(list(iter_dataset(aggregated))))

    def test_rebuilding_a_round_dataset_leaves_no_stale_shards(self):
        """Re-running a round must not inherit the previous round's shards.

        The aggregated directory is reused across runs. Appending to the existing
        manifest leaves entries for shards that the rebuild has since overwritten or
        superseded, which shows up later as a checksum mismatch -- or as a round
        silently training on another dataset's rollout windows.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            teacher_dir = Path(tmpdir) / "teacher"
            write_teacher_dataset(teacher_dir, count=20)
            labeler = TeacherLabeler(*synthetic_teacher(step=8))
            sim = make_sim(tmpdir)
            env = OfflineMarioEnv(state="Level1-1")
            try:
                first = collect_on_policy_rollout(
                    sim, env, labeler, max_steps=40, recovery_lag_px=4, window_frames=3, max_windows=1
                )
                second = collect_on_policy_rollout(
                    sim, env, labeler, max_steps=60, recovery_lag_px=4, window_frames=3, max_windows=2
                )
            finally:
                env.close()

            aggregated = Path(tmpdir) / "aggregated"
            first_info = build_dagger_dataset(aggregated, teacher_dir, [first], metadata={"iteration": 1})
            # A different teacher dataset, as a resumed run against another ROM/shard
            # would provide, plus extra rollout windows.
            other_teacher = Path(tmpdir) / "teacher_other"
            write_teacher_dataset(other_teacher, count=30)
            second_info = build_dagger_dataset(aggregated, other_teacher, [first, second], metadata={"iteration": 2})

            self.assertTrue(second_info["checksums_verified"])
            self.assertEqual(dataset_provenance(aggregated)["total_samples"], second_info["total_samples"])
            # Both rounds wrote a shard-00000.npz; only the current one may be registered.
            paths = [shard["path"] for shard in second_info["shards"]]
            self.assertEqual(len(paths), len(set(paths)))
            self.assertEqual(second_info["shard_count"], 3)
            self.assertEqual(len(list(iter_dataset(aggregated))), 3)
            self.assertEqual(second_info["metadata"]["stale_shards_cleared"], first_info["shard_count"])
            self.assertEqual(
                second_info["metadata"]["teacher_shards"],
                [load_manifest(other_teacher)["shards"][0]["sha256"]],
            )
            self.assertGreater(second_info["total_samples"], first_info["total_samples"])

    def test_aggregate_rollout_metadata_reports_assistance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            labeler = TeacherLabeler(*synthetic_teacher(step=8))
            sim = make_sim(tmpdir)
            env = OfflineMarioEnv(state="Level1-1")
            try:
                rollout = collect_on_policy_rollout(
                    sim, env, labeler, max_steps=30, recovery_lag_px=4, window_frames=2, max_windows=1
                )
            finally:
                env.close()

            meta = aggregate_rollout_metadata([rollout])
            self.assertEqual(meta["rollouts"], 1)
            self.assertTrue(meta["model_only"])
            self.assertGreaterEqual(meta["recovery_windows"], 1)
            self.assertEqual(sum(meta["recovery_reasons"].values()), meta["recovery_windows"])

    def test_model_only_actions_are_independent_of_ram(self):
        """Issue #30 forbids RAM / privileged state from influencing model-only actions."""

        class PoisonedRAMEnv(OfflineMarioEnv):
            """Identical physics and frames, but every RAM read is garbage."""

            def get_ram(self):
                return np.full(RAM_SIZE, 0xFF, dtype=np.uint8)

        with tempfile.TemporaryDirectory() as tmpdir:
            env = OfflineMarioEnv(state="Level1-1")
            sim = make_sim(tmpdir)
            obs = sim.reset_episode(env)
            frames = []
            try:
                for _ in range(5):
                    frames.append(np.asarray(obs).copy())
                    obs = sim.step(env, obs, train=False)["obs"]
            finally:
                env.close()

            # Replay the exact same frames into fresh, identically seeded models: one
            # with honest RAM and one that only ever sees garbage. If RAM leaked into
            # action selection, the chosen actions would differ.
            chosen_actions = []
            for env_cls in (OfflineMarioEnv, PoisonedRAMEnv):
                replay_env = env_cls(state="Level1-1")
                replay_sim = make_sim(tmpdir)
                replay_sim.reset_episode(replay_env)
                try:
                    chosen_actions.append([
                        replay_sim.step(replay_env, frame, train=False)["action_idx"]
                        for frame in frames
                    ])
                finally:
                    replay_env.close()

            self.assertEqual(chosen_actions[0], chosen_actions[1])

    def test_invalid_bounds_are_rejected(self):
        with self.assertRaises(ValueError):
            collect_on_policy_rollout(None, None, TeacherLabeler([1], [1]), max_steps=0)
        with self.assertRaises(ValueError):
            collect_on_policy_rollout(None, None, TeacherLabeler([1], [1]), window_frames=0)
        with self.assertRaises(ValueError):
            collect_on_policy_rollout(None, None, TeacherLabeler([1], [1]), max_windows=-1)


if __name__ == "__main__":
    unittest.main()
