"""Bounded macro-action decoding for the Super Fly motor controller.

The mainline frame-level controller asked the SNN for an independent jump
decision on every emulator frame. That made the learned policy brittle in
closed loop: a single spurious motor spike triggered a 4-frame jump with a
24-frame refractory period, and the decision cadence was unbounded relative to
the temporal window the SNN actually integrates over.

This module replaces that with a *bounded macro-action decoder*: the SNN's motor
evidence consumed over one settle window selects a short temporal chunk

* ``run``  -> ``RIGHT + B`` for ``chunk_frames`` frames, or
* ``jump`` -> ``RIGHT + B + A`` for ``jump_chunk_frames`` frames,

and the chunk is then executed frame by frame. Every chunk length is bounded by
``MAX_CHUNK_FRAMES``, so inference can never loop unboundedly and the action
cadence is a reproducible hyperparameter rather than an emergent property of
the spike train.

The decision is read off the two motor channels the supervised readout is
actually trained on -- ``RUN_ACTION`` (RIGHT+B) and ``JUMP_ACTION`` (RIGHT+B+A),
which are exactly the actions ``action_idx`` can take. What the decoder must not
do is treat a jump channel as evidence for *both* alternatives, or decide on a
channel nothing is trained to drive: see ``MacroActionDecoder.step``.
"""

import math
from dataclasses import asdict, dataclass, replace

# NES action indices from simulation.ACTION_MAP
RUN_ACTION = 1   # RIGHT + B (run)
JUMP_ACTION = 3  # RIGHT + B + A (running jump)

#: Hard upper bound on how many frames a single macro chunk may occupy.
MAX_CHUNK_FRAMES = 30
#: Teacher trajectory macro cadence (teacher.py MACRO_FRAMES), used as the default.
DEFAULT_CHUNK_FRAMES = 15
#: Hard upper bound on the run-chunk spacing enforced between jump chunks.
MAX_MACRO_REFRACTORY_FRAMES = MAX_CHUNK_FRAMES * 4

MACRO_ACTIONS = ("run", "jump")


@dataclass(frozen=True)
class MacroDecision:
    """One bounded macro-action chunk decision."""

    macro: str
    action_idx: int
    chunk_frames: int
    frames_remaining: int
    jump_evidence: float
    run_evidence: float
    is_new_decision: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _require_bounded(name: str, value: int, lower: int, upper: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if not (lower <= value <= upper):
        raise ValueError(f"{name} must be between {lower} and {upper}")
    return value


class MacroActionDecoder:
    """Deterministic bounded action-repeat policy over SNN motor evidence.

    ``step`` is called exactly once per emulator frame with the motor-ganglion
    spikes accumulated over the frame's settle window and returns the chunk the
    controller should be executing. A chunk always runs to completion, which is
    what keeps the action cadence fixed instead of re-deciding every frame.
    """

    def __init__(self, chunk_frames: int = DEFAULT_CHUNK_FRAMES,
                 jump_chunk_frames: int | None = None,
                 jump_margin: float = 0.0,
                 refractory_frames: int = 0,
                 max_chunk_frames: int = MAX_CHUNK_FRAMES):
        self.max_chunk_frames = _require_bounded("max_chunk_frames", max_chunk_frames, 1, MAX_CHUNK_FRAMES)
        self.chunk_frames = _require_bounded("chunk_frames", chunk_frames, 1, self.max_chunk_frames)
        if jump_chunk_frames is None:
            jump_chunk_frames = self.chunk_frames
        self.jump_chunk_frames = _require_bounded("jump_chunk_frames", jump_chunk_frames, 1, self.max_chunk_frames)
        self.refractory_frames = _require_bounded(
            "refractory_frames", int(refractory_frames), 0, MAX_MACRO_REFRACTORY_FRAMES
        )
        if not isinstance(jump_margin, (int, float)) or isinstance(jump_margin, bool):
            raise ValueError("jump_margin must be a number")
        self.jump_margin = float(jump_margin)
        if math.isnan(self.jump_margin):
            raise ValueError("jump_margin must be a number, not NaN")
        self.reset()

    # -- episode lifecycle -------------------------------------------------
    def reset(self) -> None:
        """Clear chunk state at an episode boundary (no cross-episode leakage)."""
        self.frames_remaining = 0
        self.current_macro: str | None = None
        self.frames_since_jump = self.refractory_frames
        self.decisions = 0
        self.jump_decisions = 0
        self.frames_executed = 0
        self.last_decision: MacroDecision | None = None

    # -- decision loop -----------------------------------------------------
    def step(self, accumulated_motor_spikes) -> MacroDecision:
        """Advance one emulator frame and return the active macro chunk.

        Args:
            accumulated_motor_spikes: length-4 motor spike tensor summed over the
                frame's settle window, ordered ``[NOOP, RIGHT, JUMP, RIGHT+JUMP]``.
        """
        if len(accumulated_motor_spikes) < 4:
            raise ValueError("accumulated_motor_spikes must contain the 4 motor neurons")

        # One channel per executable macro action, each counted exactly once.
        #
        # The supervised targets (``pretrain.target_rate_vector``) drive the motor
        # neuron whose index *is* the action index: action 1 for a run frame and
        # action 3 for a jump frame. So the decision boundary has to compare those
        # two channels. Two plausible-looking alternatives both fail:
        #
        # * counting ``spikes[3]`` in *both* terms cancels it, leaving
        #   ``spikes[2] - spikes[1]``; unit 2 is 0.05 in every target vector and is
        #   never the positively trained channel for a jump, so a network that has
        #   learned to jump reads as a tie and the controller never jumps;
        # * deciding on ``spikes[2]`` (JUMP without RIGHT) asks for a standing jump
        #   the macro never executes -- ``jump`` always means RIGHT+B+A here.
        jump_evidence = float(accumulated_motor_spikes[JUMP_ACTION])
        run_evidence = float(accumulated_motor_spikes[RUN_ACTION])

        is_new_decision = False
        if self.frames_remaining <= 0:
            jump_ready = self.frames_since_jump >= self.refractory_frames
            # A tie falls back to running, and ``refractory_frames`` bounds how
            # often a jump chunk may reopen, so a jump-happy network still cannot
            # produce an unbounded jump rate.
            wants_jump = (jump_evidence - run_evidence) > self.jump_margin and jump_ready
            macro = "jump" if wants_jump else "run"
            chunk_frames = self.jump_chunk_frames if macro == "jump" else self.chunk_frames

            self.current_macro = macro
            self.frames_remaining = chunk_frames
            self.decisions += 1
            is_new_decision = True
            if macro == "jump":
                self.jump_decisions += 1
                self.frames_since_jump = 0
            self.last_decision = MacroDecision(
                macro=macro,
                action_idx=JUMP_ACTION if macro == "jump" else RUN_ACTION,
                chunk_frames=chunk_frames,
                frames_remaining=chunk_frames,
                jump_evidence=jump_evidence,
                run_evidence=run_evidence,
                is_new_decision=True,
            )

        # Execute one frame of the active chunk.
        self.frames_remaining -= 1
        self.frames_since_jump += 1
        self.frames_executed += 1

        # Evidence fields describe the decision that opened the active chunk.
        return replace(
            self.last_decision,
            frames_remaining=self.frames_remaining,
            is_new_decision=is_new_decision,
        )

    # -- introspection -----------------------------------------------------
    @property
    def active_macro(self) -> str | None:
        return self.current_macro

    @property
    def action_cadence(self) -> int:
        """Frames per run-chunk decision (the reproducibility-critical cadence)."""
        return self.chunk_frames

    def config(self) -> dict:
        """Reproducibility-critical decoder parameters (checkpoint-stable)."""
        return {
            "decoder": "bounded_macro_action",
            "chunk_frames": self.chunk_frames,
            "jump_chunk_frames": self.jump_chunk_frames,
            "jump_margin": self.jump_margin,
            "refractory_frames": self.refractory_frames,
            "max_chunk_frames": self.max_chunk_frames,
        }

    def metadata(self) -> dict:
        """Serializable provenance (parameters plus per-episode counters)."""
        return {
            **self.config(),
            "decisions": self.decisions,
            "jump_decisions": self.jump_decisions,
            "frames_executed": self.frames_executed,
        }


def calibrate_jump_margin(evidence_diffs, jump_labels, method: str = "balanced_accuracy") -> dict:
    """Pick ``jump_margin`` from labelled decision evidence instead of assuming zero.

    A freshly trained readout is *offset*: on the teacher's own chunks the mean
    ``jump - run`` evidence is negative for run chunks *and* for jump chunks, so a
    zero margin reads as "never jump" no matter how well the two classes separate.
    Calibrating the margin on the supervised chunks places the boundary where the
    classes actually divide.

    Args:
        evidence_diffs: per-decision ``jump_evidence - run_evidence`` values.
        jump_labels: per-decision truth (True if a jump chunk was correct).
        method: ``balanced_accuracy`` (mean of per-class recall, so the majority
            class cannot win) or ``jump_rate`` (match the labelled jump frequency).

    Returns:
        dict with ``margin`` plus the recall/jump-rate achieved at it. When only one
        class is present the calibration is undefined and margin 0.0 is returned
        with ``method: insufficient_classes`` -- an honest failure, not a silent
        default that happens to look calibrated.
    """
    diffs = [float(d) for d in evidence_diffs]
    labels = [bool(label) for label in jump_labels]
    if len(diffs) != len(labels):
        raise ValueError("evidence_diffs and jump_labels must have the same length")
    if method not in ("balanced_accuracy", "jump_rate"):
        raise ValueError("method must be 'balanced_accuracy' or 'jump_rate'")

    n_jump = sum(labels)
    if not diffs or n_jump == 0 or n_jump == len(labels):
        return {
            "margin": 0.0,
            "method": "insufficient_classes",
            "samples": len(diffs),
            "jump_samples": n_jump,
            "balanced_accuracy": None,
            "jump_recall": None,
            "run_recall": None,
            "jump_rate": None,
            "target_jump_rate": round(n_jump / len(labels), 4) if labels else None,
        }

    # Candidate boundaries: every observed value (predicting "jump" strictly above it).
    candidates = sorted(set(diffs + [-math.inf, math.inf]))
    best = None
    for margin in candidates:
        predicted = [d > margin for d in diffs]
        jump_recall = sum(p and l for p, l in zip(predicted, labels)) / n_jump
        run_recall = sum((not p) and (not l) for p, l in zip(predicted, labels)) / (len(labels) - n_jump)
        balanced = 0.5 * (jump_recall + run_recall)
        jump_rate = sum(predicted) / len(predicted)
        score = balanced if method == "balanced_accuracy" else -abs(jump_rate - n_jump / len(labels))
        if best is None or score > best["score"]:
            best = {
                "score": score,
                "margin": margin,
                "balanced_accuracy": balanced,
                "jump_recall": jump_recall,
                "run_recall": run_recall,
                "jump_rate": jump_rate,
            }

    return {
        "margin": float(best["margin"]),
        "method": method,
        "samples": len(diffs),
        "jump_samples": n_jump,
        "balanced_accuracy": round(best["balanced_accuracy"], 4),
        "jump_recall": round(best["jump_recall"], 4),
        "run_recall": round(best["run_recall"], 4),
        "jump_rate": round(best["jump_rate"], 4),
        "target_jump_rate": round(n_jump / len(labels), 4),
    }
