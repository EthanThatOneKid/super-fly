"""Teacher-forced sequential replay: the pre-screen's episode-shaped outcome.

The pre-screen used to score decisions one at a time: stride the shard by the action
cadence, ask the network what it thinks of every chunk in isolation, and compare the
evidence values. That throws away the three things that decide a closed-loop run:

* **state.** Each decision is made by an SNN whose recurrent state came from the decisions
  before it, not from a cold start at every chunk.
* **commitment.** The decoder holds a chunk to completion, so the controller cannot change
  its mind one frame after choosing ``run`` -- and a refractory period can *forbid* a jump
  that the evidence asks for.
* **consequence.** A missed jump where the teacher jumped is not one wrong label out of 99.
  It ends the run. The arms measure that difference directly: the closed-loop deaths are at
  x 313, 549, 594, 624 and 839, and every one of them is a teacher takeoff.

This module replays the teacher's frames in order, driving the same ``MacroActionDecoder``
the closed loop drives, and reports an *episode*: where the run dies, how many required
jumps it missed, how many it took that were not required, and the ``best_x`` that follows --
the same units the emulator reports, so the numbers are directly comparable to the reference
and to an arm's own closed-loop score.

The frames are the teacher's, because an offline replay cannot fabricate frames for a
trajectory the model did not take. Everything else in the replay is real: the frames are fed
one at a time (the emulator's own cadence), the settle window, the recurrent state, the
chunk commitment and the refractory counter all evolve as they do in closed loop.

The cost model, stated so it can be argued with:

* a **missed jump** -- the teacher was airborne out of a chunk and the controller ran -- is
  treated as fatal at that position, which is what makes ``offline_best_x`` episode-shaped;
* a **spurious** jump -- taken where the teacher was on the ground -- is *not* fatal here.
  On the ROM it lands and continues, so the cost is a lost chunk rather than a death, and
  this replay cannot show the desync it causes because the frames belong to the teacher. It
  is counted and reported, never scored.

**The fatal model is a pessimistic lower bound, and the first measurement said so.** Replaying
DAgger round 1 (whose emulator run reached ``best_x`` 1247) kills it at **x 249**, the first
required jump it missed: the teacher's jumps are *sufficient* to cross the level, not
*necessary*, and a real run can clear some obstacles at speed without them. So
``offline_best_x`` is not comparable to an emulator ``best_x`` in absolute terms, and worse, it
**saturates**: with ~20 required jumps and a per-jump success rate below ~0.9, the first miss
lands early for every arm and they all pile up at the same number (both of the first two arms
measured died at exactly 249).

Hence two statistics, and they answer different questions:

* ``jump_sequence_recall`` -- of the jumps the teacher actually took, how many the controller
took, decided in sequence. Teacher-forced, so a miss does not stop later jumps from being
counted, which is what keeps it informative at competence levels where ``offline_best_x`` has
saturated. This is the gate's progress measure.
* ``offline_best_x`` / ``offline_completion`` -- the compounded episode under the pessimistic
  model. Only the high-competence regime can move it, and that is the point: it is the
  strictest reading of the same evidence, and the emulator's own gate (completion) in
  offline form.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from connectome import DrosophilaConnectomeSNN
from dagger import is_airborne_from_ram
from evidence import evidence_config, rule_from_config
from macro_decoder import MacroActionDecoder
from ram_tracker import LEVEL_END_MIN_X, MarioRAMTracker
from seed_policy import SeedSet, resolve_seeds
from vision import OmmatidiaVisionPreprocessor

#: NES player vertical state (0x001D): 0 on the ground, 1 rising, 2 falling.
AIR_STATE_ADDR = 0x001D


@dataclass(frozen=True)
class JumpWindow:
    """One stretch the teacher spent off the ground, and whether it began with a jump.

    A stretch that began with an A press (air state 1) is a jump the level *required*: the
    teacher is a beam search optimizing for completion, so it did not jump for nothing. A
    stretch that began already falling (air state 2) is something that happened to the
    teacher rather than a decision, and is tracked separately so it cannot be mistaken for
    a required jump.
    """

    start: int
    end: int
    jumped: bool

    @property
    def frames(self) -> int:
        return self.end - self.start + 1


def teacher_traces(shards: Sequence[dict], cadence: int = 15) -> List[Dict[str, object]]:
    """Position, phase and required jumps per shard, from the recorded RAM alone.

    Computed once per pre-screen and handed to every arm: all arms are scored against the
    same teacher, so their outcomes are comparable decision for decision.
    """
    tracker = MarioRAMTracker()
    traces: List[Dict[str, object]] = []
    for shard in shards:
        ram = shard["ram"]
        frames = int(len(shard["frames"]))
        xs = [tracker.get_x_pos(row) for row in ram]
        air_states = [
            int(row[AIR_STATE_ADDR]) if len(row) > AIR_STATE_ADDR else 0 for row in ram
        ]
        airborne = [is_airborne_from_ram(row) for row in ram]
        windows: List[JumpWindow] = []
        start: Optional[int] = None
        for index in range(frames):
            if airborne[index] and start is None:
                start = index
            elif not airborne[index] and start is not None:
                windows.append(JumpWindow(start, index - 1, air_states[start] == 1))
                start = None
        if start is not None:
            windows.append(JumpWindow(start, frames - 1, air_states[start] == 1))

        traces.append({
            "frames": frames,
            "cadence": cadence,
            "x": xs,
            "airborne": airborne,
            "air_states": air_states,
            "windows": windows,
            "required": [window for window in windows if window.jumped],
            "fall_windows": [window for window in windows if not window.jumped],
            "max_x": max(xs) if xs else 0,
            "reached_end": bool(xs) and max(xs) >= LEVEL_END_MIN_X,
        })
    return traces


def macro_decoder(cadence: int, config: Optional[Dict[str, object]] = None) -> MacroActionDecoder:
    """The decoder a checkpoint ships, falling back to this run's cadence.

    The closed loop adopts ``policy_config.macro_decoder`` when it loads weights, so the
    replay has to as well -- otherwise an arm would be pre-screened with a decoder it will
    never run.
    """
    config = dict(config or {})
    chunk_frames = int(config.get("chunk_frames") or cadence)
    jump_chunk_frames = config.get("jump_chunk_frames")
    return MacroActionDecoder(
        chunk_frames=chunk_frames,
        jump_chunk_frames=int(jump_chunk_frames) if jump_chunk_frames else chunk_frames,
        jump_margin=float(config.get("jump_margin", 0.0)),
        refractory_frames=int(config.get("refractory_frames") or 0),
        evidence_rule=config.get("evidence_rule"),
        evidence_decay=config.get("evidence_decay"),
    )


def _covers(decision: Dict[str, object], window: JumpWindow) -> bool:
    """Whether a decision's chunk overlaps a stretch of teacher flight."""
    start = int(decision["start_frame"])
    end = start + int(decision["frames"]) - 1
    return start <= window.end and end >= window.start


def episode_outcome(decisions: Sequence[Dict[str, object]], trace: Dict[str, object]) -> Dict[str, object]:
    """Turn one replay's decisions into the outcome an episode would have had.

    A required jump is *taken* if any jump chunk the controller committed to overlaps the
    stretch of teacher flight -- which is the physical requirement (be airborne over the
    gap), and is robust to the controller's chunks drifting out of step with the teacher's.
    """
    windows: List[JumpWindow] = trace["windows"]
    required: List[JumpWindow] = trace["required"]
    jumps = [decision for decision in decisions if decision["macro"] == "jump"]

    taken = [window for window in required if any(_covers(decision, window) for decision in jumps)]
    missed = [window for window in required if not any(_covers(decision, window) for decision in jumps)]
    spurious = [
        decision for decision in jumps
        if not any(_covers(decision, window) for window in windows)
    ]
    first_miss = min(missed, key=lambda window: window.start) if missed else None

    xs = trace["x"]
    if first_miss is not None:
        best_x = xs[min(first_miss.start, len(xs) - 1)] if xs else 0
    else:
        best_x = trace["max_x"]

    return {
        "decisions": len(decisions),
        "required_jumps": len(required),
        "jumps_taken": len(taken),
        "missed_jumps": len(missed),
        "spurious_jumps": len(spurious),
        "jump_rate": round(len(jumps) / len(decisions), 4) if decisions else None,
        # Counted over the whole sequence, teacher-forced: an earlier miss does not stop
        # later required jumps from being measured, which is what keeps this informative
        # where the compounded best_x has already saturated at the first miss.
        "jump_sequence_recall": round(len(taken) / len(required), 4) if required else None,
        "fatality_model": "any_missed_teacher_jump_ends_run",
        "first_miss_frame": first_miss.start if first_miss is not None else None,
        "first_miss_x": best_x if first_miss is not None else None,
        "offline_best_x": best_x,
        # No required jump missed -- but a shard that stops short of the level end cannot
        # produce completion evidence, so it is not credited as one.
        "offline_completion": first_miss is None and bool(trace["reached_end"]),
        "survival_frames": first_miss.start if first_miss is not None else trace["frames"],
        "teacher_frames": trace["frames"],
        "teacher_max_x": trace["max_x"],
        # A flight that began as a fall rather than a jump is not a required jump; it is
        # reported so an unexpected one cannot hide inside the required count.
        "fall_windows": len(trace["fall_windows"]),
    }


def replay_sequence(model: DrosophilaConnectomeSNN, shards: Sequence[dict],
                    traces: Sequence[Dict[str, object]], cadence: int, settle_steps: int,
                    seed: int, decoder_config: Optional[Dict[str, object]] = None,
                    preprocessor: Optional[OmmatidiaVisionPreprocessor] = None) -> List[Dict[str, object]]:
    """Replay the teacher's frames in order, deciding as the controller would.

    One frame at a time through the emulator's own loop -- preprocess, settle, accumulate
    motor spikes, hand them to the decoder -- with no weight updates and both RNGs seeded
    from ``seed``, so a replay is reproducible and two arms under one seed see the same
    Poisson draws over the same frames.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    model.eval()
    decoder = macro_decoder(cadence, decoder_config)
    pre = preprocessor if preprocessor is not None else OmmatidiaVisionPreprocessor(grid_h=28, grid_w=28)
    episodes: List[Dict[str, object]] = []

    with torch.no_grad():
        for shard, trace in zip(shards, traces):
            model.reset_state()
            pre.reset()
            decoder.reset()
            decisions: List[Dict[str, object]] = []
            frames = shard["frames"]
            for index in range(len(frames)):
                features, _ = pre.process_frame(frames[index])
                # The decoder's own evidence rule, so an arm is replayed with the statistic it
                # will run under rather than with a reconstruction of it.
                evidence = decoder.new_evidence()
                for _ in range(settle_steps):
                    spikes = pre.generate_poisson_spikes(features)
                    motor_spikes, activations = model(spikes)
                    evidence.observe(motor_spikes, activations, model)

                decision = decoder.step(evidence.channels())
                if decision.is_new_decision:
                    decisions.append({
                        "index": len(decisions),
                        "start_frame": index,
                        "frames": int(decision.chunk_frames),
                        "macro": decision.macro,
                        "jump_evidence": round(float(decision.jump_evidence), 6),
                        "run_evidence": round(float(decision.run_evidence), 6),
                    })

            episodes.append({
                "decisions": decisions,
                "outcome": episode_outcome(decisions, trace),
            })
    return episodes


def _spread(values: Sequence[Optional[float]]) -> Dict[str, object]:
    present = [float(value) for value in values if value is not None]
    if not present:
        return {"values": [], "mean": None, "min": None, "max": None}
    return {
        "values": [round(value, 4) for value in present],
        "mean": round(sum(present) / len(present), 4),
        "min": round(min(present), 4),
        "max": round(max(present), 4),
    }


def sequence_report(name: str, model: DrosophilaConnectomeSNN, shards: Sequence[dict],
                    cadence: int, settle_steps: int, seeds,
                    decoder_config: Optional[Dict[str, object]] = None,
                    traces: Optional[Sequence[Dict[str, object]]] = None,
                    detail: Optional[Dict[str, object]] = None,
                    preprocessor: Optional[OmmatidiaVisionPreprocessor] = None) -> Dict[str, object]:
    """Sequential outcomes for one arm, per replay, plus the teacher it was scored against.

    ``seeds`` is a :class:`~seed_policy.SeedSet` or anything ``resolve_seeds`` accepts; the
    set carries the role it is allowed to play into the report, so a reader can tell a
    tuning measurement from one that was reported on the reserved set.
    """
    if not shards:
        raise ValueError("a sequence report needs at least one shard")
    seed_set = seeds if isinstance(seeds, SeedSet) else resolve_seeds(seeds)
    traces = list(traces) if traces is not None else teacher_traces(shards, cadence)

    required_jumps = sum(len(trace["required"]) for trace in traces)
    per_replay: List[Dict[str, object]] = []
    for replay_seed in seed_set.seeds:
        episodes = replay_sequence(
            model, shards, traces, cadence, settle_steps, replay_seed, decoder_config, preprocessor
        )
        outcomes = [dict(episode["outcome"]) for episode in episodes]
        decisions = [episode["decisions"] for episode in episodes]
        required = sum(o["required_jumps"] for o in outcomes)
        taken = sum(o["jumps_taken"] for o in outcomes)
        per_replay.append({
            "replay_seed": replay_seed,
            "shards": len(outcomes),
            "outcomes": outcomes,
            "decisions": [len(entry) for entry in decisions],
            # One shard is one episode segment; the arm-level figure is their mean, and the
            # per-shard outcomes stay attached so a single bad segment cannot hide.
            "offline_best_x": round(sum(o["offline_best_x"] for o in outcomes) / len(outcomes), 4),
            "offline_completion": all(o["offline_completion"] for o in outcomes),
            "required_jumps": required,
            "jumps_taken": taken,
            "missed_jumps": sum(o["missed_jumps"] for o in outcomes),
            "spurious_jumps": sum(o["spurious_jumps"] for o in outcomes),
            "jump_sequence_recall": round(taken / required, 4) if required else None,
            "jump_rate": round(sum(o["jumps_taken"] + o["spurious_jumps"] for o in outcomes)
                               / sum(o["decisions"] for o in outcomes), 4)
            if sum(o["decisions"] for o in outcomes) else None,
        })

    return {
        "arm": name,
        "metric": "teacher_forced_sequence",
        "replays": len(per_replay),
        "replay_seeds": [row["replay_seed"] for row in per_replay],
        "seed_role": seed_set.role,
        "cadence": cadence,
        "settle_steps": settle_steps,
        "decoder": dict(decoder_config or {}),
        # Which statistic the margin was applied to, stated outright: the two numbers are
        # only interpretable together, and the replay above ran exactly this rule.
        "evidence": evidence_config(*rule_from_config(decoder_config)),
        "per_replay": per_replay,
        # Headline progress measure: the teacher's jumps, taken in sequence.
        "jump_sequence_recall": _spread([row["jump_sequence_recall"] for row in per_replay]),
        # The compounded episode under the pessimistic fatality model. Reported, and only
        # meaningful once jump recall is high enough for a run to survive its first miss.
        "offline_best_x": _spread([row["offline_best_x"] for row in per_replay]),
        "offline_completion_rate": round(
            sum(1 for row in per_replay if row["offline_completion"]) / len(per_replay), 4
        ),
        "missed_jumps": _spread([row["missed_jumps"] for row in per_replay]),
        "spurious_jumps": _spread([row["spurious_jumps"] for row in per_replay]),
        "required_jumps": _spread([row["required_jumps"] for row in per_replay]),
        "jump_rate": _spread([row["jump_rate"] for row in per_replay]),
        "teacher": {
            "shards": len(traces),
            "frames": sum(int(trace["frames"]) for trace in traces),
            "max_x": max(int(trace["max_x"]) for trace in traces),
            "required_jumps": required_jumps,
            "fall_windows": sum(len(trace["fall_windows"]) for trace in traces),
            "reached_end": all(bool(trace["reached_end"]) for trace in traces),
            # What the teacher spent to cross the level, for reading an arm's own jump rate
            # against (it jumps on ~20% of its decisions on Level 1-1).
            "jump_rate": round(
                required_jumps / sum(max(1, int(trace["frames"]) // cadence) for trace in traces), 4
            ) if traces else None,
        },
        "detail": dict(detail or {}),
    }
