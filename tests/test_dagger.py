import tempfile
import unittest
from pathlib import Path

import numpy as np

from dagger import (
    AIRBORNE_ADDR,
    PHASE_AIR,
    PHASE_ANY,
    PHASE_GROUND,
    TeacherLabeler,
    aggregate_rollout_metadata,
    build_dagger_dataset,
    collect_on_policy_rollout,
    label_aliasing_report,
    position_only_disagrees,
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


# --- a chunk-aligned teacher with a mid-flight stretch ------------------------
#
# Mirrors the shape of the recorded ROM teacher: four macro chunks, of which one is
# a jump taken from the ground and one is a run chunk decided while airborne. The
# airborne run chunk is the aliasing population -- its recorded action is ``run``
# even though the state it was chosen from was off the ground.
PHASE_TEACHER_FRAMES = 60
PHASE_TEACHER_MACRO_FRAMES = 15
PHASE_TEACHER_STEP = 7
#: Chunks 0/3 run on the ground, chunk 1 jumps from it, chunk 2 holds run mid-flight.
PHASE_TEACHER_CHUNK_ACTIONS = (1, 3, 1, 1)
#: The flight: the jump chunk leaves the ground at frame 17 and lands at frame 42.
PHASE_TEACHER_TAKEOFF_FRAME = 17
PHASE_TEACHER_LANDING_FRAME = 42
PHASE_TEACHER_GROUND_FRAMES = 35
PHASE_TEACHER_AIRBORNE_FRAMES = 25


def phase_teacher_chunk(frame):
    return frame // PHASE_TEACHER_MACRO_FRAMES


def phase_teacher_airborne():
    """Whether each frame ends airborne."""
    return [PHASE_TEACHER_TAKEOFF_FRAME <= frame < PHASE_TEACHER_LANDING_FRAME
            for frame in range(PHASE_TEACHER_FRAMES)]


def phase_teacher():
    """(xs, actions, phase-before-action) for the synthetic chunk-aligned teacher."""
    airborne = phase_teacher_airborne()
    xs = [30 + PHASE_TEACHER_STEP * frame for frame in range(PHASE_TEACHER_FRAMES)]
    actions = [PHASE_TEACHER_CHUNK_ACTIONS[phase_teacher_chunk(frame)]
               for frame in range(PHASE_TEACHER_FRAMES)]
    before = [False] + airborne[:-1]
    return xs, actions, before


def make_phase_labeler():
    xs, actions, before = phase_teacher()
    return TeacherLabeler(xs, actions, teacher_airborne=before,
                          macro_frames=PHASE_TEACHER_MACRO_FRAMES)


def write_phase_teacher_dataset(dataset_dir):
    """Persist the synthetic teacher as a shard, phase included in the RAM.

    Frame i's recorded RAM is the state *after* action i, which is the phase the
    teacher chose action i+1 in -- exactly how ``teacher.py`` records the real shard.
    """
    xs, actions, before = phase_teacher()
    airborne = phase_teacher_airborne()
    ram = []
    for index, x_pos in enumerate(xs):
        snapshot = ram_at(x_pos)
        snapshot[AIRBORNE_ADDR] = 1 if airborne[index] else 0
        ram.append(snapshot)
    write_shard(
        dataset_dir,
        [np.zeros((240, 256, 3), dtype=np.uint8) for _ in xs],
        actions,
        ram,
        [False] * len(xs),
        [False] * len(xs),
        metadata={"env_kind": "offline_synthetic"},
        provenance={"origin": "teacher", "env_kind": "offline_synthetic",
                    "macro_frames": PHASE_TEACHER_MACRO_FRAMES},
    )
    return xs, actions, before


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


class TestPhaseAwareLabelling(unittest.TestCase):
    """The labeler must match the *phase* the teacher decided in, not just progress."""

    #: Inside the teacher's mid-flight run chunk: position-only says ``run``.
    ALIASED_X = 300
    #: The ground-phase decision that carried the teacher over this stretch.
    TAKEOFF_X = 135

    def test_ground_label_uses_the_same_phase_decision(self):
        labeler = make_phase_labeler()
        legacy = labeler.label(self.ALIASED_X)
        grounded = labeler.label(self.ALIASED_X, airborne=False)

        # The teacher held ``run`` mid-flight over this stretch, so progress-only
        # matching teaches a grounded candidate to run at the obstacle.
        self.assertEqual(legacy.action, 1)
        self.assertEqual(legacy.phase, PHASE_ANY)
        self.assertFalse(legacy.phase_matched)
        # Phase-matched matching returns the takeoff decision that clears it.
        self.assertEqual(grounded.action, 3)
        self.assertEqual(grounded.phase, PHASE_GROUND)
        self.assertTrue(grounded.phase_matched)
        self.assertEqual(grounded.chunk_start_x, self.TAKEOFF_X)
        # The last ground-phase decision at or before x=300 is frame 17, the top of
        # the jump chunk that carries the teacher across this progress.
        self.assertEqual(grounded.index, 17)

    def test_chunk_commitment_horizon_is_reported(self):
        grounded = make_phase_labeler().label(self.ALIASED_X, airborne=False)
        self.assertEqual(grounded.chunk_index, 1)
        self.assertEqual(grounded.chunk_end_x, 233)
        # Frame 17 of a 15-frame chunk starting at 15 leaves 13 frames of the chunk.
        self.assertEqual(grounded.chunk_frames_remaining, 13)
        self.assertTrue(grounded.is_jump)

    def test_airborne_label_uses_the_airborne_decision(self):
        labeler = make_phase_labeler()
        airborne = labeler.label(self.ALIASED_X, airborne=True)
        # Matching in the air is legitimate: the teacher really was flying over this
        # progress and really was holding run. Only the grounded case was wrong.
        self.assertEqual(airborne.action, 1)
        self.assertEqual(airborne.phase, PHASE_AIR)
        self.assertTrue(airborne.phase_matched)

    def test_phase_matching_only_ever_adds_jumps(self):
        """The fix can never remove a jump the teacher actually performed.

        Every disagreement has to be ``run`` -> ``jump``: a grounded candidate is
        only ever upgraded to the jump the teacher used to clear the spot.
        """
        labeler = make_phase_labeler()
        flips = 0
        for x_pos in range(0, labeler.final_x + 1):
            legacy = labeler.label(x_pos).action
            phased = labeler.label(x_pos, airborne=False).action
            if legacy != phased:
                flips += 1
                self.assertEqual((legacy, phased), (1, 3), f"x={x_pos}")
        self.assertGreater(flips, 0)

    def test_without_phase_data_labelling_is_position_only(self):
        xs, actions, _ = phase_teacher()
        labeler = TeacherLabeler(xs, actions)
        self.assertFalse(labeler.has_phase)
        # No phase recorded means no phase to match on, so the old rule stands and
        # says so, rather than silently claiming a match it cannot make.
        label = labeler.label(self.ALIASED_X, airborne=False)
        self.assertEqual(label.action, 1)
        self.assertFalse(label.phase_matched)
        self.assertEqual(label.phase, PHASE_ANY)

    def test_rejects_mismatched_phase_array(self):
        with self.assertRaises(ValueError):
            TeacherLabeler([10, 20], [1, 3], teacher_airborne=[False])

    def test_without_phase_reproduces_the_old_behaviour(self):
        """The ablation is what a progress-only labeler would have produced."""
        phased = make_phase_labeler()
        ablated = phased.without_phase()

        self.assertFalse(ablated.has_phase)
        self.assertEqual(ablated.macro_frames, phased.macro_frames)
        self.assertEqual(ablated.final_x, phased.final_x)
        self.assertEqual(ablated.label(self.ALIASED_X, airborne=False).action, 1)
        self.assertEqual(ablated.label(self.ALIASED_X).action, phased.label(self.ALIASED_X).action)

    def test_from_dataset_recovers_phase_and_declared_cadence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            write_phase_teacher_dataset(tmpdir)
            labeler = TeacherLabeler.from_dataset(tmpdir)
            self.assertTrue(labeler.has_phase)
            # The cadence comes from the shard's own provenance, not a guess.
            self.assertEqual(labeler.macro_frames, PHASE_TEACHER_MACRO_FRAMES)
            self.assertEqual(len(labeler.ground_index), PHASE_TEACHER_GROUND_FRAMES)
            self.assertEqual(len(labeler.air_index), PHASE_TEACHER_AIRBORNE_FRAMES)
            self.assertEqual(labeler.label(self.ALIASED_X, airborne=False).action, 3)

    def test_chunk_decision_summary_counts_mid_flight_decisions(self):
        summary = make_phase_labeler().chunk_decision_summary()
        self.assertEqual(summary["macro_decisions"], 4)
        self.assertEqual(summary["jump_decisions"], 1)
        self.assertEqual(summary["run_decisions"], 3)
        # The aliasing population: run chunks the teacher decided while airborne.
        self.assertEqual(summary["airborne_run_decisions"], 1)
        self.assertEqual(summary["airborne_decision_rate"], 0.25)


class TestLabellingAudit(unittest.TestCase):
    """The offline audit is the evidence that the labelling bug is real."""

    def test_audit_quantifies_aliasing_without_an_emulator(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            write_phase_teacher_dataset(tmpdir)
            report = label_aliasing_report(tmpdir)

        self.assertTrue(report["has_phase"])
        self.assertEqual(report["decision_summary"]["airborne_run_decisions"], 1)
        # Every swept ground state inside the teacher's flight is mislabelled, and
        # every disagreement is a missing jump -- never the other way round.
        self.assertEqual(report["mislabelled_positions"], 91)
        self.assertEqual(report["action_transitions"], {"1->3": 91})
        self.assertEqual(report["disagreement_clusters"], 1)
        cluster = report["largest_clusters"][0]
        self.assertEqual((cluster["x_start"], cluster["x_end"]), (240, 330))
        self.assertEqual(cluster["position_only_actions"], [1])
        self.assertEqual(cluster["phase_aware_actions"], [3])
        # The recovery target is anchored on the takeoff chunk that clears the gap.
        self.assertEqual(cluster["recovery_chunk_starts"], [135])

    def test_audit_refuses_to_pretend_a_phaseless_shard_has_phase(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            write_teacher_dataset(tmpdir)
            report = label_aliasing_report(tmpdir)

        self.assertFalse(report["has_phase"])
        self.assertIn("no phase information", report["note"])
        self.assertNotIn("mislabelled_positions", report)


class ScriptedEnv:
    """Just enough env surface for a rollout driven by a scripted trajectory."""

    def __init__(self):
        self.ram = np.zeros(RAM_SIZE, dtype=np.uint8)

    def get_ram(self):
        return self.ram

    def close(self):
        pass


class ScriptedSim:
    """Replays a fixed (x, airborne) trace instead of running a model.

    Lets a rollout be tested against an exact candidate trajectory, which is the only
    way to hold the labelling rule fixed while varying one state.
    """

    class _Tracker:
        def __init__(self):
            self.last_x_pos = 0

    def __init__(self, trace, action_cadence=4):
        self.trace = list(trace)
        self.ram_tracker = ScriptedSim._Tracker()
        self.action_cadence = action_cadence
        self.settle_steps = 1
        self.policy = "macro"
        self.bootstrap_episodes = 0
        self.index = 0
        self.max_x = 0
        self.obs = np.zeros((4, 4, 3), dtype=np.uint8)

    def reset_episode(self, env):
        self.index = 0
        self.max_x = 0
        self.ram_tracker.last_x_pos = 0
        return self.obs

    def step(self, env, obs, train=False):
        x_pos, airborne = self.trace[min(self.index, len(self.trace) - 1)]
        self.index += 1
        self.max_x = max(self.max_x, x_pos)
        self.ram_tracker.last_x_pos = x_pos
        env.ram[0x006D] = (x_pos // 256) % 256
        env.ram[0x0086] = x_pos % 256
        env.ram[AIRBORNE_ADDR] = 1 if airborne else 0
        return {
            "obs": obs,
            "action_idx": 1,
            "action_source": "macro",
            "ram_info": {"x_pos": x_pos, "max_x_pos": self.max_x, "is_airborne": airborne},
            "died": False,
            "completed": False,
            "terminated": False,
            "truncated": False,
        }


def scripted_rollout(labeler, trace, **kwargs):
    # Divergence by stagnation only, so the teacher's pace cannot be what decides
    # which state gets a window; the rollout stops the frame the window is cut.
    params = {"max_steps": 4, "stagnation_limit": 2, "window_frames": 3,
              "max_windows": 1, "recovery_lag_steps": 1000, "recovery_lag_px": 10_000}
    params.update(kwargs)
    env = ScriptedEnv()
    try:
        return collect_on_policy_rollout(ScriptedSim(trace), env, labeler, **params)
    finally:
        env.close()


class TestRecoveryWindowLabelling(unittest.TestCase):
    """Same scripted trajectory, two labelers: only the phase-aware one is right."""

    #: Stalled on the ground inside the teacher's flight; the teacher's own record
    #: at this progress is ``run`` because it was mid-air when it passed through.
    GROUND_STALL = [(300, False)] * 5
    #: Stalls mid-flight just past the point where the teacher's recorded action
    #: stops being the jump and becomes the airborne ``run``.
    AIRBORNE_STALL = [(233, False), (240, True), (240, True), (240, True), (240, True)]

    def test_grounded_window_target_is_the_jump_not_the_mid_flight_run(self):
        phased = scripted_rollout(make_phase_labeler(), self.GROUND_STALL)
        legacy = scripted_rollout(TeacherLabeler(*phase_teacher()[:2]), self.GROUND_STALL)

        self.assertEqual(len(phased.windows), 1)
        self.assertEqual(phased.windows[0].reason, "unrecoverable_stagnation")
        # The teacher cleared this progress by jumping from the ground at x=135.
        self.assertEqual(phased.windows[0].target_action, 3)
        self.assertEqual(phased.windows[0].phase, PHASE_GROUND)
        self.assertEqual(phased.windows[0].chunk_start_x, 135)
        self.assertGreater(phased.windows[0].label_flips, 0)
        self.assertGreater(phased.summary()["phase_aware_labels"], 0)
        # The same stall under progress-only matching is taught to run.
        self.assertEqual(legacy.windows[0].target_action, 1)
        self.assertEqual(legacy.summary()["phase_aware_labels"], 0)

    def test_held_chunk_target_does_not_chatter_mid_flight(self):
        phased = scripted_rollout(make_phase_labeler(), self.AIRBORNE_STALL)
        legacy = scripted_rollout(TeacherLabeler(*phase_teacher()[:2]), self.AIRBORNE_STALL)

        # A committed jump is held for its chunk, so every target in the window is
        # the decision that started the flight.
        self.assertEqual(set(phased.windows[0].actions), {3})
        self.assertEqual(phased.summary()["committed_frames"], 2)
        self.assertGreater(phased.windows[0].committed_frames, 0)
        # Progress-only matching re-decides every frame and flips jump -> run mid-air.
        self.assertEqual(set(legacy.windows[0].actions), {1, 3})
        self.assertEqual(legacy.summary()["committed_frames"], 1)

    def test_position_only_disagrees_flags_the_alias(self):
        labeler = make_phase_labeler()
        self.assertTrue(position_only_disagrees(labeler, 300, 3))
        self.assertFalse(position_only_disagrees(labeler, 300, 1))

    def test_aggregate_reports_labelling_provenance(self):
        phased = scripted_rollout(make_phase_labeler(), self.GROUND_STALL)
        meta = aggregate_rollout_metadata([phased])
        self.assertEqual(meta["phase_aware_labels"], phased.summary()["phase_aware_labels"])
        self.assertEqual(meta["label_flips"], phased.summary()["label_flips"])
        self.assertIn("committed_frames", meta)


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
