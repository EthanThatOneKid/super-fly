"""Closed-loop DAgger data collection for the Super Fly temporal policy.

The measured bottleneck is closed-loop distribution shift: the teacher trajectory
reaches the flagpole, but a policy trained only on teacher frames leaves that
narrow corridor and never returns to it. This module implements one DAgger round:

1. **On-policy rollout** — run the current candidate policy model-only (no RAM
   input, no teacher actions, no bootstrap pulses) and record every visited
   frame, its RAM-visible progress and the teacher's action target for that
   progress.
2. **Divergence / unrecoverable detection** — compare the candidate's progress
   against the teacher schedule. A lag above ``recovery_lag_px`` is a
   divergence; death or a stagnation streak is an unrecoverable state.
3. **Recovery windows** — when either fires, add the preceding temporal window
   of frames plus the teacher action target for the state the candidate is now
   in to the next training set.

The recovery target is **phase-aware**. Matching the teacher by progress alone is
wrong wherever the teacher was *airborne*: 45 of the recorded teacher's 99 macro
decisions are run chunks the teacher took mid-flight, so a candidate standing on
the ground at x=594 was labelled ``run`` -- the action the teacher took while
flying over that spot, and the one action that cannot get it over the pipe. The
labeler therefore matches on (progress, ground/airborne phase) and returns the
teacher's decision from the last state in the *same* phase, which for that state
is the jump at x=549 that actually clears the obstacle. On top of that, a
recovery target is held for the remainder of the teacher chunk that produced it
(``chunk_frames_remaining``, capped by the decoder's bounded cadence), so a flight
in progress is never relabelled mid-air. ``label_aliasing_report`` measures the
old position-only behaviour against the phase-aware one on a recorded shard.

Every rollout is bounded (``max_steps``, ``window_frames``, ``max_windows``) and
the aggregation step records full provenance (teacher checksums, candidate
checkpoint checksum, seed, horizon, action cadence), so the resulting dataset is
reproducible from its manifest alone.
"""

import argparse
import bisect
import json
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

from macro_decoder import JUMP_ACTION, MAX_CHUNK_FRAMES
from trajectory import add_shard_file, dataset_provenance, iter_dataset, load_manifest, write_manifest, write_shard

DEFAULT_RECOVERY_LAG_STEPS = 90
DEFAULT_RECOVERY_LAG_PX = 64
DEFAULT_WINDOW_FRAMES = 8
DEFAULT_STAGNATION_LIMIT = 90
DEFAULT_MAX_WINDOWS = 64
DEFAULT_WINDOW_COOLDOWN_FRAMES = 8

#: Teacher trajectory macro cadence (teacher.py MACRO_FRAMES).
DEFAULT_MACRO_FRAMES = 15

RAM_SIZE = 0x0800
LEVEL_LENGTH_PX = 3200
#: Player vertical state: 0 on the ground, 1 jumping, 2 falling.
AIRBORNE_ADDR = 0x001D

PHASE_GROUND = "ground"
PHASE_AIR = "air"
#: Position-only matching, used when a shard records no phase information.
PHASE_ANY = "any"


def x_pos_from_ram(ram: np.ndarray) -> int:
    """Screen-space X position straight from SMB RAM (page * 256 + sub-page)."""
    page = int(ram[0x006D]) if len(ram) > 0x006D else 0
    sub_x = int(ram[0x0086]) if len(ram) > 0x0086 else 0
    return page * 256 + sub_x


def is_airborne_from_ram(ram: np.ndarray) -> bool:
    """Whether the player is off the ground, straight from 0x001D."""
    if len(ram) <= AIRBORNE_ADDR:
        return False
    return int(ram[AIRBORNE_ADDR]) in (1, 2)


def position_only_disagrees(labeler: "TeacherLabeler", x_pos: int, action: int) -> bool:
    """Whether the legacy position-only rule would have labelled ``x_pos`` differently.

    Used purely as evidence: it counts how many training targets the phase-aware fix
    changed, which is otherwise invisible in a run's metrics.
    """
    return int(action) != labeler.label(x_pos).action


@dataclass(frozen=True)
class TeacherLabel:
    """The teacher's action target for a candidate state (progress and phase).

    ``chunk_index`` / ``chunk_frames_remaining`` describe the teacher macro chunk the
    target came from. The horizon matters because the target is a *chunk* decision:
    once a candidate is committed to a jump, relabelling it one frame later (when the
    teacher's own record at that progress reads ``run``) is exactly the aliasing this
    labeler exists to prevent.
    """

    index: int
    action: int
    teacher_x: int
    schedule_index: int
    phase: str = PHASE_ANY
    chunk_index: int = 0
    chunk_frames_remaining: int = 1
    chunk_start_x: int = 0
    chunk_end_x: int = 0
    phase_matched: bool = False

    @property
    def is_jump(self) -> bool:
        return self.action == JUMP_ACTION


class TeacherLabeler:
    """Maps candidate state to the teacher action target from a recorded shard.

    The teacher shard stores, for every frame, the action the teacher executed and
    the resulting RAM snapshot. Progress is monotonic by construction, so a
    candidate that has reached ``x`` can be labelled with a teacher action -- but
    only if the *phase* matches too. The teacher spends much of Level 1-1 airborne,
    and while airborne its recorded action is usually ``run`` (it is holding right
    mid-flight); labelling a grounded candidate from an airborne frame therefore
    teaches it to run at obstacles.

    ``label(x, airborne=...)`` restricts the search to teacher frames whose phase
    before the action matches the candidate's, so the target is the decision the
    teacher took from a state the candidate could actually be in. Passing
    ``airborne=None`` keeps the original position-only behaviour, which is what a
    shard with no phase information can support.
    """

    def __init__(self, teacher_x: Sequence[int], teacher_actions: Sequence[int],
                 teacher_airborne: Sequence[bool] | None = None,
                 macro_frames: int = DEFAULT_MACRO_FRAMES):
        if len(teacher_x) != len(teacher_actions):
            raise ValueError("teacher trajectory arrays must have equal lengths")
        if not len(teacher_x):
            raise ValueError("teacher trajectory must contain at least one frame")
        if teacher_airborne is not None and len(teacher_airborne) != len(teacher_actions):
            raise ValueError("teacher phase array must match the action array length")
        self.teacher_actions = [int(a) for a in teacher_actions]
        self.macro_frames = max(1, int(macro_frames))
        # Monotonize progress so bisect is well defined even if RAM wobbles.
        monotone: List[int] = []
        running = 0
        for value in teacher_x:
            running = max(running, int(value))
            monotone.append(running)
        self.teacher_x = monotone

        # Phase *before* each recorded action: the decision the teacher made at frame
        # i was taken in the state frame i-1 left behind.
        self.has_phase = teacher_airborne is not None
        self.teacher_airborne: List[bool] = (
            [bool(a) for a in teacher_airborne] if teacher_airborne is not None else []
        )
        self._ground_index = [i for i, air in enumerate(self.teacher_airborne) if not air]
        self._air_index = [i for i, air in enumerate(self.teacher_airborne) if air]
        self._ground_x = [self.teacher_x[i] for i in self._ground_index]
        self._air_x = [self.teacher_x[i] for i in self._air_index]

    @classmethod
    def from_dataset(cls, dataset_dir, macro_frames: int | None = None) -> "TeacherLabeler":
        """Rebuild a labeler (progress + phase) from a recorded teacher shard.

        ``macro_frames`` defaults to the cadence the shard declares in its provenance,
        so the chunk-commitment horizon matches the plan that produced the data.
        """
        if macro_frames is None:
            macro_frames = DEFAULT_MACRO_FRAMES
            for shard in load_manifest(dataset_dir).get("shards", []):
                declared = (shard.get("provenance") or {}).get("macro_frames")
                if declared:
                    macro_frames = int(declared)
                    break

        xs: List[int] = []
        actions: List[int] = []
        airborne_before: List[bool] = []
        for shard in iter_dataset(dataset_dir):
            previous_airborne = False
            for index, ram in enumerate(shard["ram"]):
                xs.append(x_pos_from_ram(ram))
                actions.append(int(shard["actions"][index]))
                airborne_before.append(previous_airborne)
                previous_airborne = is_airborne_from_ram(ram)
        return cls(xs, actions, teacher_airborne=airborne_before, macro_frames=macro_frames)

    def without_phase(self) -> "TeacherLabeler":
        """This labeler with the phase array dropped: the pre-fix labelling behaviour.

        Exists so the fix can be ablated from the public API, reporting exactly the
        targets a progress-only labeler would have produced on identical rollouts.
        """
        return TeacherLabeler(self.teacher_x, self.teacher_actions,
                              macro_frames=self.macro_frames)

    @property
    def final_x(self) -> int:
        return self.teacher_x[-1]

    @property
    def length(self) -> int:
        return len(self.teacher_x)

    @property
    def ground_index(self) -> List[int]:
        """Teacher frame indices whose action was chosen on the ground."""
        return list(self._ground_index)

    @property
    def air_index(self) -> List[int]:
        """Teacher frame indices whose action was chosen while airborne."""
        return list(self._air_index)

    def _label_for(self, index: int, phase: str, phase_matched: bool) -> TeacherLabel:
        chunk_index = index // self.macro_frames
        chunk_start = chunk_index * self.macro_frames
        chunk_end = min(chunk_start + self.macro_frames, len(self.teacher_x)) - 1
        return TeacherLabel(
            index=index,
            action=self.teacher_actions[index],
            teacher_x=self.teacher_x[index],
            schedule_index=index,
            phase=phase,
            chunk_index=chunk_index,
            chunk_frames_remaining=max(1, chunk_end - index + 1),
            chunk_start_x=self.teacher_x[chunk_start],
            chunk_end_x=self.teacher_x[chunk_end],
            phase_matched=phase_matched,
        )

    def label(self, x_pos: int, airborne: bool | None = None) -> TeacherLabel:
        """Teacher target for a candidate at ``x_pos`` in the given phase.

        Args:
            x_pos: candidate's RAM-visible progress.
            airborne: candidate's phase. ``None`` (or a shard with no phase data)
                falls back to position-only matching -- the behaviour that mislabels
                grounded candidates from the teacher's mid-flight frames.

        With a phase, the target is the teacher's action recorded the last time the
        teacher was at (or before) ``x_pos`` *in that phase*. If the candidate is
        grounded where the teacher was only ever airborne, that is the ground-phase
        decision that carried the teacher over the spot: the candidate is behind a
        jump it should already have taken, so the recovery target is that jump.
        """
        x_pos = int(x_pos)
        if airborne is None or not self.has_phase:
            index = max(0, bisect.bisect_right(self.teacher_x, x_pos) - 1)
            return self._label_for(index, PHASE_ANY, phase_matched=False)

        phase = PHASE_AIR if airborne else PHASE_GROUND
        indices = self._air_index if airborne else self._ground_index
        phase_x = self._air_x if airborne else self._ground_x
        if not indices:
            # The teacher never occupied this phase; position-only is all there is.
            index = max(0, bisect.bisect_right(self.teacher_x, x_pos) - 1)
            return self._label_for(index, phase, phase_matched=False)
        position = bisect.bisect_right(phase_x, x_pos) - 1
        # Behind every recorded same-phase state: clamp to the earliest one rather
        # than extrapolating a target from a state the teacher never visited.
        return self._label_for(indices[max(0, position)], phase, phase_matched=True)

    def progress_at_step(self, step: int) -> int:
        """Teacher progress at the same emulator step (clamped to the trajectory)."""
        index = min(max(int(step) - 1, 0), len(self.teacher_x) - 1)
        return self.teacher_x[index]

    def progress_deficit(self, step: int, x_pos: int) -> int:
        """Pixels behind the teacher's position at the same step (>= 0)."""
        return max(0, self.progress_at_step(step) - int(x_pos))

    def schedule_lag(self, step: int, x_pos: int) -> int:
        """Steps of teacher progress the candidate is behind at its current position.

        The teacher reached the candidate's current progress after ``index`` frames;
        if the candidate needed ``step > index`` frames to get there, it is behind
        the teacher schedule — the closed-loop divergence this experiment targets.

        This is a pure *progress* comparison, so it deliberately uses the full
        position-only index. Divergence detection must not shift when the labelling
        fix changes which target a state is taught.
        """
        index = max(0, bisect.bisect_right(self.teacher_x, int(x_pos)) - 1)
        return max(0, int(step) - index)

    def chunk_decision_summary(self) -> Dict[str, object]:
        """How the teacher's macro decisions split by the phase they were made in.

        The ``airborne_run_decisions`` count is the aliasing population: run chunks
        the teacher decided on mid-flight, whose recorded action is ``run`` even
        though the state they were taken from was off the ground.
        """
        chunks = (self.length + self.macro_frames - 1) // self.macro_frames
        jumps = jumps_airborne = runs = runs_airborne = 0
        for chunk_index in range(chunks):
            start = chunk_index * self.macro_frames
            airborne = bool(self.teacher_airborne[start]) if self.has_phase else False
            if int(self.teacher_actions[start]) == JUMP_ACTION:
                jumps += 1
                jumps_airborne += airborne
            else:
                runs += 1
                runs_airborne += airborne
        return {
            "macro_frames": self.macro_frames,
            "macro_decisions": chunks,
            "jump_decisions": jumps,
            "run_decisions": runs,
            "jump_rate": round(jumps / chunks, 4) if chunks else None,
            "airborne_run_decisions": runs_airborne,
            "airborne_jump_decisions": jumps_airborne,
            "airborne_decision_rate": round((runs_airborne + jumps_airborne) / chunks, 4) if chunks else None,
            "has_phase": self.has_phase,
            "ground_frames": len(self._ground_index),
            "airborne_frames": len(self._air_index),
        }


@dataclass
class RecoveryWindow:
    """The temporal window preceding a divergence plus its recovery target."""

    reason: str
    lag_px: int
    x_pos: int
    teacher_index: int
    target_action: int
    phase: str = PHASE_ANY
    chunk_index: int = 0
    chunk_start_x: int = 0
    committed_frames: int = 0
    label_flips: int = 0
    frames: List[np.ndarray] = field(default_factory=list)
    ram: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)

    def metadata(self) -> Dict[str, object]:
        return {
            "reason": self.reason,
            "lag_px": self.lag_px,
            "x_pos": self.x_pos,
            "teacher_index": self.teacher_index,
            "target_action": self.target_action,
            "window_frames": len(self.frames),
            # Provenance for the labelling fix: which phase the target was matched
            # in, which teacher chunk it came from, and how many frames of the
            # window were served by the held chunk commitment rather than a fresh
            # lookup, plus how many of its targets the position-only rule would
            # have got wrong.
            "phase": self.phase,
            "chunk_index": self.chunk_index,
            "chunk_start_x": self.chunk_start_x,
            "committed_frames": self.committed_frames,
            "label_flips": self.label_flips,
        }


@dataclass
class RolloutResult:
    """Outcome of one bounded model-only on-policy rollout."""

    steps: int = 0
    max_x: int = 0
    died: bool = False
    completed: bool = False
    truncated: bool = False
    windows: List[RecoveryWindow] = field(default_factory=list)
    divergence_events: int = 0
    unrecoverable_events: int = 0
    model_jump_frames: int = 0
    assisted_jump_frames: int = 0
    action_cadence: int = 0
    settle_steps: int = 0
    policy: str = "macro"
    bootstrap_episodes: int = 0
    seed: int | None = None
    phase_aware_labels: int = 0
    committed_frames: int = 0
    label_flips: int = 0
    schedule_lag_samples: List[int] = field(default_factory=list)
    progress_deficit_samples: List[int] = field(default_factory=list)

    def summary(self) -> Dict[str, object]:
        reasons = Counter(window.reason for window in self.windows)
        return {
            "steps": self.steps,
            "max_x": self.max_x,
            "died": self.died,
            "completed": self.completed,
            "truncated": self.truncated,
            "recovery_windows": len(self.windows),
            "recovery_reasons": dict(reasons),
            "divergence_events": self.divergence_events,
            "unrecoverable_events": self.unrecoverable_events,
            "model_jump_frames": self.model_jump_frames,
            "assisted_jump_frames": self.assisted_jump_frames,
            "model_only": self.assisted_jump_frames == 0 and self.bootstrap_episodes == 0,
            "action_cadence": self.action_cadence,
            "settle_steps": self.settle_steps,
            "policy": self.policy,
            "seed": self.seed,
            # Labelling provenance: how many frames were labelled on phase rather
            # than progress alone, how many were served by a held chunk commitment,
            # and how many targets the position-only rule would have got wrong.
            "phase_aware_labels": self.phase_aware_labels,
            "committed_frames": self.committed_frames,
            "label_flips": self.label_flips,
            "max_schedule_lag_steps": max(self.schedule_lag_samples, default=0),
            "max_progress_deficit_px": max(self.progress_deficit_samples, default=0),
            "mean_progress_deficit_px": (
                round(float(np.mean(self.progress_deficit_samples)), 2)
                if self.progress_deficit_samples else 0.0
            ),
        }


def collect_on_policy_rollout(
    sim,
    env,
    labeler: TeacherLabeler,
    max_steps: int = 1000,
    recovery_lag_steps: int = DEFAULT_RECOVERY_LAG_STEPS,
    recovery_lag_px: int = DEFAULT_RECOVERY_LAG_PX,
    window_frames: int = DEFAULT_WINDOW_FRAMES,
    stagnation_limit: int = DEFAULT_STAGNATION_LIMIT,
    max_windows: int = DEFAULT_MAX_WINDOWS,
    window_cooldown_frames: int = DEFAULT_WINDOW_COOLDOWN_FRAMES,
    seed: int | None = None,
) -> RolloutResult:
    """Run one bounded model-only rollout and extract DAgger recovery windows.

    The candidate may only use its own visual/SNN state: ``sim.step`` is called with
    ``train=False``, and no teacher action, RAM value, privileged emulator state or
    bootstrap pulse influences the action selection. RAM is read *after* the fact
    purely as a measurement for divergence detection and dataset provenance.

    Labels are matched on the candidate's phase as well as its progress, and a jump
    target is held for the remainder of the teacher chunk that produced it, so a
    flight in progress is never relabelled mid-air from the teacher's mid-flight
    ``run`` frames.
    """
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if window_frames < 1:
        raise ValueError("window_frames must be at least 1")
    if max_windows < 0:
        raise ValueError("max_windows must be non-negative")

    result = RolloutResult(
        action_cadence=getattr(sim, "action_cadence", 0),
        settle_steps=getattr(sim, "settle_steps", 0),
        policy=getattr(sim, "policy", "macro"),
        bootstrap_episodes=getattr(sim, "bootstrap_episodes", 0),
        seed=seed,
    )

    obs = sim.reset_episode(env)
    buffer = deque(maxlen=window_frames)
    last_x = 0
    stagnation = 0
    cooldown = 0
    # Phase the candidate is in when it chooses this frame's action, i.e. the state
    # the previous step left behind. Seeded on the ground at the episode's spawn.
    previous_airborne = False
    commitment: TeacherLabel | None = None
    commitment_remaining = 0
    # The held target must not outlive the decoder's own bounded cadence, so the
    # commitment horizon is capped by the chunk length the controller can commit to.
    hold_cap = min(int(getattr(sim, "action_cadence", 0) or MAX_CHUNK_FRAMES), MAX_CHUNK_FRAMES)

    for step in range(1, max_steps + 1):
        result.steps = step
        current_x = int(getattr(sim.ram_tracker, "last_x_pos", 0))
        frame = np.asarray(obs)
        if commitment is not None and commitment_remaining > 0 and previous_airborne:
            # Chunk-commitment horizon: the candidate is mid-flight on a jump it
            # already committed to. Holding the target keeps the window's action
            # labels on the decision that started the flight instead of flipping
            # them to ``run`` because the teacher's record here reads ``run``.
            frame_label = commitment
            commitment_remaining -= 1
            result.committed_frames += 1
        else:
            frame_label = labeler.label(current_x, airborne=previous_airborne)
            if frame_label.phase_matched:
                result.phase_aware_labels += 1
                if position_only_disagrees(labeler, current_x, frame_label.action):
                    result.label_flips += 1
            if previous_airborne or frame_label.is_jump:
                commitment = frame_label
                commitment_remaining = min(frame_label.chunk_frames_remaining, hold_cap)
            else:
                commitment = None
                commitment_remaining = 0
        ram_snapshot = env.get_ram().copy() if hasattr(env, "get_ram") else np.zeros(RAM_SIZE, dtype=np.uint8)
        buffer.append((frame, ram_snapshot, current_x, frame_label.action))

        outcome = sim.step(env, obs, train=False)
        obs = outcome["obs"]
        x_pos = int(outcome["ram_info"].get("x_pos", 0))
        airborne_now = bool(outcome["ram_info"].get("is_airborne", False))
        result.max_x = max(result.max_x, int(outcome["ram_info"].get("max_x_pos", x_pos)))
        schedule_lag = labeler.schedule_lag(step, x_pos)
        progress_deficit = labeler.progress_deficit(step, x_pos)
        result.schedule_lag_samples.append(schedule_lag)
        result.progress_deficit_samples.append(progress_deficit)

        if outcome["action_source"] in {"model", "model_hold", "macro", "macro_hold"}:
            result.model_jump_frames += 1
        if outcome["action_source"] in {"bootstrap", "bootstrap_hold"}:
            result.assisted_jump_frames += 1

        if x_pos > last_x:
            stagnation = 0
        else:
            stagnation += 1
        last_x = x_pos

        # Divergence = the candidate is behind the teacher schedule (in steps) or
        # behind the teacher's progress at the same step (in pixels).
        diverged = schedule_lag >= recovery_lag_steps or progress_deficit >= recovery_lag_px
        reason = None
        if outcome["died"]:
            reason = "unrecoverable_death"
        elif stagnation >= stagnation_limit:
            reason = "unrecoverable_stagnation"
        elif diverged:
            reason = "divergence"

        if reason == "divergence":
            result.divergence_events += 1
        elif reason is not None:
            result.unrecoverable_events += 1

        if cooldown > 0:
            cooldown -= 1

        if reason is not None and cooldown == 0 and len(result.windows) < max_windows:
            # The target describes the state the candidate is in *after* this frame.
            # While a jump is in flight the committed target stands; otherwise the
            # phase-aware lookup gives the decision the teacher took from a state
            # the candidate could actually be in.
            in_flight = airborne_now or previous_airborne
            committed_frames = 0
            if commitment is not None and commitment_remaining > 0 and in_flight:
                recovery_target = commitment
                committed_frames = commitment_remaining
            else:
                recovery_target = labeler.label(x_pos, airborne=airborne_now)
            frames = [entry[0] for entry in buffer] + [np.asarray(obs)]
            ram = [entry[1] for entry in buffer] + [ram_snapshot]
            actions = [int(entry[3]) for entry in buffer] + [recovery_target.action]
            # How many of this window's targets the position-only rule had wrong.
            window_pairs = [(entry[2], int(entry[3])) for entry in buffer] + [(x_pos, recovery_target.action)]
            result.windows.append(
                RecoveryWindow(
                    reason=reason,
                    lag_px=progress_deficit,
                    x_pos=x_pos,
                    teacher_index=recovery_target.index,
                    target_action=recovery_target.action,
                    phase=recovery_target.phase,
                    chunk_index=recovery_target.chunk_index,
                    chunk_start_x=recovery_target.chunk_start_x,
                    committed_frames=committed_frames,
                    label_flips=sum(
                        1 for x_at, action in window_pairs
                        if position_only_disagrees(labeler, x_at, action)
                    ),
                    frames=frames,
                    ram=ram,
                    actions=actions,
                )
            )
            cooldown = window_cooldown_frames

        previous_airborne = airborne_now

        if outcome["died"]:
            result.died = True
        if outcome["completed"]:
            result.completed = True
        if outcome["terminated"] or outcome["truncated"]:
            result.truncated = bool(outcome["truncated"])
            break

    return result


def write_rollout_shard(dataset_dir, rollout: RolloutResult, provenance: Dict[str, object] | None = None):
    """Write one rollout's recovery windows as a single checksummed shard."""
    frames: List[np.ndarray] = []
    actions: List[int] = []
    ram: List[np.ndarray] = []
    terminated: List[bool] = []
    truncated: List[bool] = []
    for window in rollout.windows:
        frames.extend(window.frames)
        actions.extend(window.actions)
        ram.extend(window.ram)
        terminated.extend([False] * len(window.frames))
        truncated.extend([False] * len(window.frames))
    if not frames:
        return None

    shard_provenance = {
        "origin": "dagger_recovery_windows",
        "rollout": rollout.summary(),
        "windows": [window.metadata() for window in rollout.windows],
    }
    if provenance:
        shard_provenance.update(provenance)
    return write_shard(
        dataset_dir,
        frames,
        actions,
        ram,
        terminated,
        truncated,
        metadata=None,
        provenance=shard_provenance,
    )


def clear_dataset_dir(dataset_dir) -> int:
    """Drop shards and manifest so a round is rebuilt solely from its declared inputs.

    A round's dataset is a *derived* artifact: teacher shards plus this round's
    rollout windows. Appending to whatever happens to be in the directory makes a
    re-run depend on history -- a second round reusing the same directory keeps the
    previous round's shard entries (and, when the teacher dataset changed, an entry
    whose recorded checksum no longer matches the bytes that were just copied over
    it). ``dataset_provenance`` then fails on a stale entry, or worse, a round trains
    on leftover rollout windows from a different dataset. Rebuilding is idempotent
    and leaves nothing to go stale.

    Only ``shard-*.npz`` and ``manifest.json`` are removed; nothing else in the
    directory is touched.
    """
    root = Path(dataset_dir)
    if not root.exists():
        return 0
    removed = 0
    for shard in sorted(root.glob("shard-*.npz")):
        shard.unlink()
        removed += 1
    manifest = root / "manifest.json"
    if manifest.exists():
        manifest.unlink()
    return removed


def build_dagger_dataset(dataset_dir, teacher_dataset_dir, rollouts: Iterable[RolloutResult],
                         metadata: Dict[str, object] | None = None) -> Dict[str, object]:
    """Aggregate teacher shards plus on-policy recovery-window shards.

    The teacher shards are copied byte-for-byte so their checksums stay valid, and
    the aggregated manifest records the full provenance of both sources. The
    directory is rebuilt from scratch first (see ``clear_dataset_dir``) so re-running
    a round cannot inherit stale shards. Returns the verified dataset provenance
    summary.
    """
    root = Path(dataset_dir)
    root.mkdir(parents=True, exist_ok=True)
    cleared = clear_dataset_dir(root)

    teacher_manifest = load_manifest(teacher_dataset_dir)
    for shard in teacher_manifest.get("shards", []):
        source = Path(teacher_dataset_dir) / shard["path"]
        add_shard_file(
            root,
            source,
            provenance={
                "origin": "teacher",
                "source_dataset": str(Path(teacher_dataset_dir).resolve()),
                "source_sha256": shard["sha256"],
                "source_samples": int(shard["samples"]),
            },
        )

    written_rollouts = 0
    for rollout in rollouts:
        if write_rollout_shard(root, rollout) is not None:
            written_rollouts += 1

    manifest = load_manifest(root)
    combined_metadata = dict(manifest.get("metadata", {}))
    combined_metadata.update({
        "teacher_dataset": str(Path(teacher_dataset_dir).resolve()),
        "teacher_shards": [shard["sha256"] for shard in teacher_manifest.get("shards", [])],
        "rollout_shards": written_rollouts,
        "stale_shards_cleared": cleared,
    })
    combined_metadata.update(metadata or {})
    manifest["metadata"] = combined_metadata
    write_manifest(root, manifest)

    return dataset_provenance(root)


def _contiguous_runs(values: Sequence[int], gap: int) -> List[tuple]:
    """Group a sparse sorted position list into (start, end) runs no further than ``gap`` apart."""
    runs: List[tuple] = []
    start = previous = None
    for value in values:
        if start is None:
            start = previous = value
            continue
        if value - previous <= gap:
            previous = value
            continue
        runs.append((start, previous))
        start = previous = value
    if start is not None:
        runs.append((start, previous))
    return runs


def label_aliasing_report(dataset_dir, sweep_step: int = 1, macro_frames: int | None = None) -> Dict[str, object]:
    """Measure position-only vs phase-aware labelling on a recorded teacher shard.

    This is the offline evidence that the labelling bug is real and that the fix
    changes it: it sweeps every ground-state progress value, compares the legacy
    progress-only target with the phase-aware one, and reports where they disagree --
    the airborne stretches where the teacher's own action was ``run`` because it was
    mid-flight, and a grounded candidate was being taught to run at an obstacle.

    Pure shard reading: no emulator, no model, so it runs anywhere the teacher
    dataset does.
    """
    labeler = TeacherLabeler.from_dataset(dataset_dir, macro_frames=macro_frames)
    summary = labeler.chunk_decision_summary()
    report: Dict[str, object] = {
        "dataset": str(Path(dataset_dir).resolve()),
        "teacher_frames": labeler.length,
        "final_x": labeler.final_x,
        "sweep_step": max(1, int(sweep_step)),
        "decision_summary": summary,
    }
    if not labeler.has_phase or not labeler.air_index or not labeler.ground_index:
        # No airborne frames means no phase *contrast*: the two rules agree by
        # construction, so reporting an aliasing count would be meaningless rather
        # than reassuring.
        report["has_phase"] = False
        report["note"] = (
            "shard records no phase information (its RAM shows no airborne frames), "
            "so position-only and phase-aware labelling are identical by construction"
        )
        return report

    report["has_phase"] = True
    step = max(1, int(sweep_step))
    positions = list(range(0, labeler.final_x + 1, step))
    flips: List[tuple] = []
    transitions = Counter()
    for x_pos in positions:
        legacy = labeler.label(x_pos)
        phased = labeler.label(x_pos, airborne=False)
        if legacy.action == phased.action:
            continue
        flips.append((x_pos, legacy.action, phased.action, phased.chunk_start_x))
        transitions[f"{legacy.action}->{phased.action}"] += 1

    clusters = []
    for start, end in _contiguous_runs([flip[0] for flip in flips], step):
        inside = [flip for flip in flips if start <= flip[0] <= end]
        clusters.append({
            "x_start": start,
            "x_end": end,
            "width": end - start + step,
            "positions": len(inside),
            "position_only_actions": sorted({flip[1] for flip in inside}),
            "phase_aware_actions": sorted({flip[2] for flip in inside}),
            "recovery_chunk_starts": sorted({flip[3] for flip in inside}),
        })
    clusters.sort(key=lambda cluster: cluster["positions"], reverse=True)

    report.update({
        "positions_swept": len(positions),
        "mislabelled_positions": len(flips),
        "mislabelled_fraction": round(len(flips) / len(positions), 4) if positions else 0.0,
        "action_transitions": dict(transitions),
        "disagreement_clusters": len(clusters),
        "largest_clusters": clusters[:5],
        "examples": [
            {"x": x_pos, "position_only_action": legacy, "phase_aware_action": phased,
             "recovery_chunk_start_x": chunk_start}
            for x_pos, legacy, phased, chunk_start in flips[:10]
        ],
    })
    return report


def main() -> None:
    """Audit a recorded teacher shard's labelling (no emulator or model required)."""
    parser = argparse.ArgumentParser(description="Audit DAgger teacher labelling (issue #30)")
    parser.add_argument("--teacher-dataset", default="data/teacher",
                        help="Directory holding the teacher trajectory shard")
    parser.add_argument("--sweep-step", type=int, default=1, help="Pixels between sampled candidate states")
    parser.add_argument("--macro-frames", type=int, default=None, help="Override the shard's declared macro cadence")
    parser.add_argument("--output", help="Write the audit JSON here as well as to stdout")
    args = parser.parse_args()

    report = label_aliasing_report(args.teacher_dataset, sweep_step=args.sweep_step,
                                   macro_frames=args.macro_frames)
    text = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n")
    print(text)


def aggregate_rollout_metadata(rollouts: Iterable[RolloutResult]) -> Dict[str, object]:
    """Summarize a batch of rollouts for a report."""
    rollouts = list(rollouts)
    reasons = Counter()
    windows = 0
    for rollout in rollouts:
        windows += len(rollout.windows)
        reasons.update(window.reason for window in rollout.windows)
    return {
        "rollouts": len(rollouts),
        "steps": sum(r.steps for r in rollouts),
        "recovery_windows": windows,
        "recovery_reasons": dict(reasons),
        "completed": sum(1 for r in rollouts if r.completed),
        "deaths": sum(1 for r in rollouts if r.died),
        "model_only": all(r.summary()["model_only"] for r in rollouts) if rollouts else True,
        "best_x": max((r.max_x for r in rollouts), default=0),
        # Labelling provenance for the round, so a report shows how much of the
        # round's supervision came from phase-matched rather than position-only
        # targets, and how many targets the old rule had wrong.
        "phase_aware_labels": sum(r.phase_aware_labels for r in rollouts),
        "committed_frames": sum(r.committed_frames for r in rollouts),
        "label_flips": sum(r.label_flips for r in rollouts),
    }


if __name__ == "__main__":
    main()
