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

Every rollout is bounded (``max_steps``, ``window_frames``, ``max_windows``) and
the aggregation step records full provenance (teacher checksums, candidate
checkpoint checksum, seed, horizon, action cadence), so the resulting dataset is
reproducible from its manifest alone.
"""

from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

from trajectory import add_shard_file, dataset_provenance, iter_dataset, load_manifest, write_manifest, write_shard

DEFAULT_RECOVERY_LAG_STEPS = 90
DEFAULT_RECOVERY_LAG_PX = 64
DEFAULT_WINDOW_FRAMES = 8
DEFAULT_STAGNATION_LIMIT = 90
DEFAULT_MAX_WINDOWS = 64
DEFAULT_WINDOW_COOLDOWN_FRAMES = 8

RAM_SIZE = 0x0800
LEVEL_LENGTH_PX = 3200


def x_pos_from_ram(ram: np.ndarray) -> int:
    """Screen-space X position straight from SMB RAM (page * 256 + sub-page)."""
    page = int(ram[0x006D]) if len(ram) > 0x006D else 0
    sub_x = int(ram[0x0086]) if len(ram) > 0x0086 else 0
    return page * 256 + sub_x


@dataclass(frozen=True)
class TeacherLabel:
    """The teacher's action target for a candidate progress value."""

    index: int
    action: int
    teacher_x: int
    schedule_index: int


class TeacherLabeler:
    """Maps candidate progress to the teacher action target from a recorded shard.

    The teacher shard stores, for every frame, the action the teacher executed and
    the resulting RAM snapshot. Progress is monotonic by construction, so a
    candidate that has reached ``x`` is labelled with the teacher action recorded
    the last time the teacher was at (or before) ``x``.
    """

    def __init__(self, teacher_x: Sequence[int], teacher_actions: Sequence[int]):
        if len(teacher_x) != len(teacher_actions):
            raise ValueError("teacher trajectory arrays must have equal lengths")
        if not len(teacher_x):
            raise ValueError("teacher trajectory must contain at least one frame")
        self.teacher_actions = [int(a) for a in teacher_actions]
        # Monotonize progress so bisect is well defined even if RAM wobbles.
        monotone: List[int] = []
        running = 0
        for value in teacher_x:
            running = max(running, int(value))
            monotone.append(running)
        self.teacher_x = monotone

    @classmethod
    def from_dataset(cls, dataset_dir) -> "TeacherLabeler":
        xs: List[int] = []
        actions: List[int] = []
        for shard in iter_dataset(dataset_dir):
            xs.extend(x_pos_from_ram(ram) for ram in shard["ram"])
            actions.extend(int(a) for a in shard["actions"])
        return cls(xs, actions)

    @property
    def final_x(self) -> int:
        return self.teacher_x[-1]

    @property
    def length(self) -> int:
        return len(self.teacher_x)

    def label(self, x_pos: int) -> TeacherLabel:
        """Teacher target for a candidate at ``x_pos``.

        Matching is by progress, so the labelled action is the one the teacher took
        when it occupied the same part of the level as the candidate does now.
        """
        import bisect

        index = max(0, bisect.bisect_right(self.teacher_x, int(x_pos)) - 1)
        return TeacherLabel(
            index=index,
            action=self.teacher_actions[index],
            teacher_x=self.teacher_x[index],
            schedule_index=index,
        )

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
        """
        return max(0, int(step) - self.label(x_pos).index)


@dataclass
class RecoveryWindow:
    """The temporal window preceding a divergence plus its recovery target."""

    reason: str
    lag_px: int
    x_pos: int
    teacher_index: int
    target_action: int
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

    for step in range(1, max_steps + 1):
        result.steps = step
        current_x = int(getattr(sim.ram_tracker, "last_x_pos", 0))
        frame = np.asarray(obs)
        frame_label = labeler.label(current_x)
        ram_snapshot = env.get_ram().copy() if hasattr(env, "get_ram") else np.zeros(RAM_SIZE, dtype=np.uint8)
        buffer.append((frame, ram_snapshot, current_x, frame_label.action))

        outcome = sim.step(env, obs, train=False)
        obs = outcome["obs"]
        x_pos = int(outcome["ram_info"].get("x_pos", 0))
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
            recovery_target = labeler.label(x_pos)
            frames = [entry[0] for entry in buffer] + [np.asarray(obs)]
            ram = [entry[1] for entry in buffer] + [ram_snapshot]
            actions = [int(entry[3]) for entry in buffer] + [recovery_target.action]
            result.windows.append(
                RecoveryWindow(
                    reason=reason,
                    lag_px=progress_deficit,
                    x_pos=x_pos,
                    teacher_index=recovery_target.index,
                    target_action=recovery_target.action,
                    frames=frames,
                    ram=ram,
                    actions=actions,
                )
            )
            cooldown = window_cooldown_frames

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
    }
