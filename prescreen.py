"""Emulator-free pre-screen: would this arm survive the level, judged as a sequence?

A closed-loop round is the expensive, hard-to-reproduce part of the work, so the decision
to spend one has to be made offline. Two offline measurements have already been tried and
replaced here, and each failure is worth naming, because the next attempt has to avoid it:

* *calibrated balanced accuracy* over pooled Poisson replays refitted the decision boundary
  on the same evidence it then scored, reported 0.52 for an untrained network against
  0.54-0.58 for a trained one, and pooled away the variance that mattered -- the same frozen
  model showed a separation of d=0.62 on one replay and d=0.06 over three;
* *jump-chunk recall at a bounded jump rate* fixed the fitting (a budget is a definition,
  not a threshold, and a random ordering of the same evidence catches exactly the rate it
  spends) and the pooling (per replay, paired on the shared seed, untrained baseline in
  every table), but it still scored **one decision at a time**: every chunk was judged in
  isolation, from a cold SNN state, ignoring the decoder's chunk commitment and the
  refractory period, with no notion that a missed jump ends the run. It ranked DAgger round
  2 above round 1 (0.427 vs 0.413) where the emulator ranked them the other way (899 vs
  1247) -- and a pre-screen that cannot rank two arms the emulator separates is not a gate.

So the default measurement is now an **episode**. ``offline_episode.py`` replays the
teacher's frames in order, drives the same decoder the closed loop drives, and reports where
the run dies, how many required jumps it missed, and the ``best_x`` that follows -- the
emulator's own units, directly comparable to the reference (594) and to the arm's measured
closed-loop score. Recall at a bounded rate is still computed for every arm, as a supporting
view under ``budget``: it answers whether the arm can catch jumps at all, while the sequence
answers whether catching them survives being a sequence.

Both views are reported per replay (never pooled into a single figure), paired across arms
on the replay seed they share, and every table carries an untrained baseline measured from
the initialization a trained arm starts from.
"""

import argparse
import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from connectome import DrosophilaConnectomeSNN
from macro_decoder import JUMP_ACTION, RUN_ACTION, calibrate_jump_margin, decision_quality
from offline_episode import sequence_report, teacher_traces
from simulation import DEFAULT_SETTLE_STEPS
from trajectory import dataset_provenance, iter_dataset
from vision import OmmatidiaVisionPreprocessor

#: Fraction of decisions the controller is allowed to spend on jumps, for the supporting
#: chunk-decision view. The teacher jumps on 25.3% of its Level 1-1 macro chunks, so a cap a
#: little above that leaves an arm room to over-fire deliberately without turning into a
#: jump-happy policy; the number is recorded in every report so a reader never has to guess.
DEFAULT_MAX_JUMP_RATE = 0.35
#: Replications per arm. Each one is an independent Poisson draw over the same chunks,
#: which is the dominant source of noise in this measurement, so one replay is not a
#: measurement -- it is a sample.
DEFAULT_REPLAYS = 3
#: Required-jump recall the sequence has to hold. The teacher needs every one of its jumps to
#: reach the flagpole, so the only principled way to set this is the survival it implies: at
#: 0.90 across ~20 required jumps a run still survives only ~12% of the time, and anything
#: much lower cannot chain into a completion at all.
PRESCREEN_MIN_JUMP_RECALL = 0.90
#: Minimum paired mean gain in sequence recall over the untrained baseline.
PRESCREEN_MIN_RECALL_DELTA = 0.10
#: Reserved name for the baseline row; a run cannot call a checkpoint by it.
UNTRAINED_ARM = "untrained"


@dataclass
class Arm:
    """One candidate to pre-screen: a model, and optionally the margin it ships with.

    ``detail`` carries whatever provenance should travel with the numbers (training
    configuration, weight deltas, checkpoint path). It is copied into the report
    verbatim, so a measured row can always be traced back to what produced it.
    """

    name: str
    model: DrosophilaConnectomeSNN
    margin: Optional[float] = None
    #: The rest of the checkpoint's ``policy_config.macro_decoder`` (chunk lengths and
    #: refractory period). The replay has to use the decoder the closed loop would adopt, so
    #: an arm carries its own rather than inheriting the run's defaults.
    decoder: Optional[Dict[str, object]] = None
    detail: Dict[str, object] = field(default_factory=dict)


def vision_preprocessor() -> OmmatidiaVisionPreprocessor:
    """A fresh preprocessor; one per arm, so no arm inherits another's adaptation."""
    return OmmatidiaVisionPreprocessor(grid_h=28, grid_w=28)


def untrained_model(seed: int = 42) -> DrosophilaConnectomeSNN:
    """The initialization a trained arm *starts* from, with no training at all.

    Every table carries this arm. The chunk-decision evidence is offset and the metric
    has a high no-skill floor, so a candidate's recall is only readable next to a
    network that was never trained -- and next to the chance line at the same rate.
    """
    torch.manual_seed(seed)
    model = DrosophilaConnectomeSNN()
    model.eval()
    return model


def recall_at_bounded_rate(evidence_diffs: Sequence[float], jump_labels: Sequence[bool],
                           max_jump_rate: float = DEFAULT_MAX_JUMP_RATE) -> Dict[str, object]:
    """Jump-chunk recall achievable while spending at most ``max_jump_rate`` jumps.

    The shipped rule is ``jump when evidence_diff > margin``, but fitting that margin to
    a score is what made the previous pre-screen unfalsifiable on the data it was fitted
    on. This fixes the *budget* instead: with ``k = floor(max_jump_rate * n)`` decisions
    allowed to be jumps, the recall-maximising choice is the ``k`` highest-evidence
    decisions -- any other subset of the same size catches no more -- so the operating
    point is derived from the budget alone and no margin is fitted to the labels.

    ``margin`` is still reported, because the decoder needs one. It is placed midway
    between the last included and the first excluded decision, so the strict ``>`` rule
    spends the budget rather than rounding down to a gap in the evidence, and
    ``margin_jump_rate`` is what that threshold actually spends: with ties at the boundary
    it can miss the budget, which is reported rather than smoothed over.

    Args:
        evidence_diffs: per-decision ``jump_evidence - run_evidence`` values.
        jump_labels: per-decision truth (True if a jump chunk was correct).
        max_jump_rate: the jump budget, as a fraction of decisions.

    Returns:
        dict with the recall at the budget, the rate it spends, the chance reference at
        that rate, and the margin realizing it. When the evidence contains no jump chunk
        at all the recall is reported as ``None`` rather than 0.0 -- a zero there would
        look like a measured failure instead of an unmeasurable one.
    """
    diffs = [float(d) for d in evidence_diffs]
    labels = [bool(label) for label in jump_labels]
    if len(diffs) != len(labels):
        raise ValueError("evidence_diffs and jump_labels must have the same length")
    if not isinstance(max_jump_rate, (int, float)) or isinstance(max_jump_rate, bool):
        raise ValueError("max_jump_rate must be a number")
    if math.isnan(float(max_jump_rate)):
        raise ValueError("max_jump_rate must be a number, not NaN")
    if not 0.0 <= float(max_jump_rate) <= 1.0:
        raise ValueError("max_jump_rate must be between 0 and 1")
    max_jump_rate = float(max_jump_rate)

    chunks = len(diffs)
    jump_chunks = sum(labels)
    budget = int(math.floor(max_jump_rate * chunks))
    base = {
        "metric": "jump_recall_at_bounded_rate",
        "method": "bounded_jump_rate",
        "max_jump_rate": max_jump_rate,
        "chunks": chunks,
        "jump_chunks": jump_chunks,
        "budget_chunks": budget,
        "jump_rate": round(budget / chunks, 4) if chunks else None,
        "teacher_jump_rate": round(jump_chunks / chunks, 4) if chunks else None,
    }
    if not chunks or not jump_chunks:
        return {
            **base,
            "jump_recall": None,
            "jump_chunks_caught": None,
            "precision": None,
            "random_recall": base["jump_rate"],
            "lift_over_random": None,
            "margin": None,
            "margin_jump_rate": None,
        }

    ranked = sorted(range(chunks), key=lambda index: diffs[index], reverse=True)
    caught = sum(1 for index in ranked[:budget] if labels[index])
    recall = caught / jump_chunks
    # Midway between the last included and the first excluded decision, so the strict
    # ``>`` rule spends the budget instead of rounding down to the nearest gap.
    if budget <= 0:
        margin = diffs[ranked[0]]
    elif budget >= chunks:
        margin = None
    else:
        margin = (diffs[ranked[budget - 1]] + diffs[ranked[budget]]) / 2
    margin_rate = None if margin is None else sum(1 for diff in diffs if diff > margin) / chunks

    return {
        **base,
        "jump_recall": round(recall, 4),
        "jump_chunks_caught": caught,
        "precision": round(caught / budget, 4) if budget else None,
        # What an uninformative ordering of this same evidence would catch at this rate.
        "random_recall": base["jump_rate"],
        "lift_over_random": round(recall - budget / chunks, 4),
        "margin": margin,
        "margin_jump_rate": round(margin_rate, 4) if margin_rate is not None else None,
    }


def collect_replays(model: DrosophilaConnectomeSNN, shards: Sequence[dict], stride: int,
                    settle_steps: int, seed: int, replays: int = DEFAULT_REPLAYS) -> List[Dict[str, object]]:
    """One evidence row per replay, keyed by replay seed.

    The replay seed is the pairing key: ``decision_evidence`` reseeds both RNGs from it
    before every replay, so two arms scored with the same seed see the *same* Poisson
    draws over the same chunk order and differ only in what their weights make of them.
    That is what makes a seed-wise delta a paired comparison rather than two independent
    samples that happen to be adjacent in a table.
    """
    if replays < 1:
        raise ValueError("replays must be at least 1")
    preprocessor = vision_preprocessor()
    rows = []
    for replay in range(replays):
        replay_seed = seed + replay
        diffs, labels = decision_evidence(model, preprocessor, shards, stride, settle_steps, replay_seed)
        rows.append({"replay_seed": replay_seed, "diffs": diffs, "labels": labels})
    return rows


def pooled_evidence(replays: Sequence[Dict[str, object]]) -> Tuple[List[float], List[bool]]:
    """Concatenate the replays' decisions (the old pooled view, kept for calibration)."""
    diffs: List[float] = []
    labels: List[bool] = []
    for row in replays:
        diffs.extend(row["diffs"])
        labels.extend(row["labels"])
    return diffs, labels


def calibration_record(replays: Sequence[Dict[str, object]],
                       method: str = "balanced_accuracy") -> Dict[str, object]:
    """Calibrate the decoder's jump margin on pooled replays, recording where it came from.

    The margin is only interpretable next to the measurement that produced it, so the
    replay count, the seeds and the per-replay decision count travel with it into the
    checkpoint's ``policy_config``.
    """
    if not replays:
        raise ValueError("calibration needs at least one replay")
    calibration = calibrate_jump_margin(*pooled_evidence(replays), method=method)
    calibration["replays"] = len(replays)
    calibration["replay_seeds"] = [row["replay_seed"] for row in replays]
    calibration["decisions_per_replay"] = len(replays[0]["diffs"])
    return calibration


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


def summarise_arm(name: str, replays: Sequence[Dict[str, object]],
                  max_jump_rate: float = DEFAULT_MAX_JUMP_RATE,
                  margin: Optional[float] = None,
                  detail: Optional[Dict[str, object]] = None,
                  stride: Optional[int] = None, settle_steps: Optional[int] = None) -> Dict[str, object]:
    """Turn collected replays into a reportable arm: per replay, plus the pooled view.

    ``margin`` adds what the *shipped* threshold does on this same evidence. The two are
    deliberately different questions -- the bounded-rate figure asks what the arm could
    catch with its jump budget, the margin figure asks what the checkpoint in hand will
    actually do -- and a gap between them means the shipped margin is leaving recall on
    the table.
    """
    if not replays:
        raise ValueError("an arm report needs at least one replay")

    per_replay = []
    for row in replays:
        entry = {
            "replay_seed": row["replay_seed"],
            "chunks": len(row["diffs"]),
            "jump_chunks": sum(row["labels"]),
            **recall_at_bounded_rate(row["diffs"], row["labels"], max_jump_rate),
        }
        if margin is not None:
            entry["at_margin"] = decision_quality(row["diffs"], row["labels"], margin)
        per_replay.append(entry)

    pooled_diffs, pooled_labels = pooled_evidence(replays)
    report = {
        "arm": name,
        "stride": stride,
        "settle_steps": settle_steps,
        "replays": len(replays),
        "replay_seeds": [row["replay_seed"] for row in replays],
        "decisions_per_replay": len(replays[0]["diffs"]),
        "max_jump_rate": max_jump_rate,
        "per_replay": per_replay,
        # Headline: the per-replay mean, never a pooled figure that hides the spread.
        "jump_recall": _spread([row["jump_recall"] for row in per_replay]),
        "jump_rate": _spread([row["jump_rate"] for row in per_replay]),
        "random_recall": _spread([row["random_recall"] for row in per_replay]),
        "lift_over_random": _spread([row["lift_over_random"] for row in per_replay]),
        "pooled": recall_at_bounded_rate(pooled_diffs, pooled_labels, max_jump_rate),
        "detail": dict(detail or {}),
    }
    if margin is not None:
        report["margin"] = margin
        report["at_margin"] = decision_quality(pooled_diffs, pooled_labels, margin)
    return report


def arm_report(arm: Arm, dataset, stride: int, settle_steps: int, seed: int,
               replays: int = DEFAULT_REPLAYS,
               max_jump_rate: float = DEFAULT_MAX_JUMP_RATE) -> Dict[str, object]:
    """Collect and summarise one arm against ``dataset`` (a directory or loaded shards)."""
    shards = dataset if isinstance(dataset, list) else _load_shards(dataset)
    rows = collect_replays(arm.model, shards, stride, settle_steps, seed, replays)
    return summarise_arm(
        arm.name, rows, max_jump_rate, margin=arm.margin, detail=arm.detail,
        stride=stride, settle_steps=settle_steps,
    )


def paired_deltas(baseline: Dict[str, object], arm: Dict[str, object],
                  metric: str = "jump_recall") -> Dict[str, object]:
    """Seed-wise differences between one arm and the baseline, on shared replay seeds.

    Only seeds present in *both* arms are compared, so the two columns are never
    measured on different data; the count of pairs is reported for the same reason.
    """
    base_by_seed = {row["replay_seed"]: row for row in baseline["per_replay"]}
    rows = []
    for row in arm["per_replay"]:
        seed = row["replay_seed"]
        if seed not in base_by_seed or row.get(metric) is None or base_by_seed[seed].get(metric) is None:
            continue
        rows.append({
            "replay_seed": seed,
            "arm": row[metric],
            "baseline": base_by_seed[seed][metric],
            "delta": round(row[metric] - base_by_seed[seed][metric], 4),
        })

    deltas = [row["delta"] for row in rows]
    return {
        "arm": arm["arm"],
        "baseline": baseline["arm"],
        "metric": metric,
        "paired_replays": len(rows),
        "rows": rows,
        "mean_delta": round(sum(deltas) / len(deltas), 4) if deltas else None,
        "min_delta": round(min(deltas), 4) if deltas else None,
        "max_delta": round(max(deltas), 4) if deltas else None,
        "improved": sum(1 for delta in deltas if delta > 0),
        "all_improved": bool(deltas) and all(delta > 0 for delta in deltas),
        # The paired means as well as the deltas: the verdict has to place a bar relative to
        # the baseline it was actually compared against, not to a constant.
        "baseline_mean": round(sum(row["baseline"] for row in rows) / len(rows), 4) if rows else None,
        "arm_mean": round(sum(row["arm"] for row in rows) / len(rows), 4) if rows else None,
    }


def prescreen_verdict(arm: Dict[str, object], paired: Optional[Dict[str, object]],
                      min_recall: float = PRESCREEN_MIN_JUMP_RECALL,
                      min_delta: float = PRESCREEN_MIN_RECALL_DELTA,
                      max_jump_rate: float = DEFAULT_MAX_JUMP_RATE) -> Dict[str, object]:
    """Does this arm clear the bar for an emulator run? Every criterion is itemised.

    A gate that reports only "fail" invites re-running until the number looks good, so each
    criterion carries the value it judged and the threshold it judged against. Five must hold,
    and they are the emulator's own criteria expressed offline: the run may not miss a jump the
    teacher took (``offline_completion`` -- the pessimistic reading of survival), it must take
    the teacher's jumps at a rate that could chain into one (``jump_sequence_recall``), it may
    not spend more than the jump budget doing it (``jump_rate_within_budget``), it must beat the
    untrained network it shares an architecture with by a real margin, and it must do so on
    *every* replay -- a mean propped up by one lucky draw is the noise this rewrite exists to
    expose.

    The rate criterion is not decoration: it is the only thing standing between this metric and
    its simplest exploit. An always-jump policy covers every stretch of teacher flight, so it
    scores a perfect ``jump_sequence_recall`` and an ``offline_completion`` -- measured, on a
    checkpoint whose calibration degenerated to "jump on everything": recall 1.000 and
    completion 1.00 on all three replays, by spending **49.5%** of its decisions on jumps and
    29 spurious ones. Covering 20 required jumps inside a 35% budget needs at least 20 of ~34
    jumps on target, which is a real constraint; jumping constantly is not.

    ``offline_best_x`` is deliberately **not** a criterion. It is incommensurable with an
    emulator ``best_x`` (it is teacher-forced, so it stops at the first miss instead of
    continuing under the controller's own dynamics) and it saturates at low competence, where
    arms that differ die at the same first jump: measured per replay it was 249 / 249 / 2200 for
    an untrained network. It is reported as the pessimistic scenario, not gated on.
    """
    mean_recall = arm["jump_sequence_recall"]["mean"]
    mean_jump_rate = arm["jump_rate"]["mean"]
    completion_rate = arm["offline_completion_rate"]
    paired_replays = paired["paired_replays"] if paired else 0
    baseline_recall = paired.get("baseline_mean") if paired else None
    bar = min_recall
    if baseline_recall is not None:
        bar = max(bar, baseline_recall + min_delta)
    criteria = [
        {
            "criterion": "offline_completion",
            "detail": "no required jump missed, on every replay",
            "pass": completion_rate == 1.0,
            "observed": completion_rate,
            "threshold": 1.0,
        },
        {
            "criterion": "jump_sequence_recall",
            "detail": ("teacher jumps taken in sequence, clear of the untrained baseline by "
                       f"{min_delta:g}"),
            "pass": mean_recall is not None and mean_recall >= bar,
            "observed": mean_recall,
            "threshold": round(bar, 4),
        },
        {
            "criterion": "jump_rate_within_budget",
            "detail": "the sequence spends no more than the jump budget on jumps",
            "pass": mean_jump_rate is not None and mean_jump_rate <= max_jump_rate,
            "observed": mean_jump_rate,
            "threshold": max_jump_rate,
        },
        {
            "criterion": "beats_untrained",
            "detail": "paired mean gain over the untrained baseline",
            "pass": bool(paired) and paired["mean_delta"] is not None and paired["mean_delta"] >= min_delta,
            "observed": paired["mean_delta"] if paired else None,
            "threshold": min_delta,
        },
        {
            "criterion": "consistent_across_replays",
            "detail": "every paired replay improves, not just the average",
            "pass": bool(paired) and paired["all_improved"],
            "observed": f"{paired['improved']}/{paired_replays}" if paired else None,
            "threshold": "all paired replays",
        },
    ]
    return {
        "arm": arm["arm"],
        "pass": all(criterion["pass"] for criterion in criteria),
        "criteria": criteria,
        "note": _verdict_note(paired),
    }


def _verdict_note(paired: Optional[Dict[str, object]]) -> Optional[str]:
    """Why a paired comparison is missing, when it is -- the two causes are not the same.

    An absent baseline is a bug in the caller. A baseline with nothing to compare is usually
    a dataset with no teacher jumps in it at all (a synthetic shard, say), where the sequence
    metric is undefined for every arm and the arm should not be read as "failed" so much as
    "unmeasurable".
    """
    if paired is None:
        return "no untrained baseline in this table: no paired comparison"
    if not paired.get("paired_replays"):
        return ("no teacher jumps in this dataset: the sequence metric is undefined for "
                "every arm")
    return None


def prescreen(arms: Sequence[Arm], dataset, stride: int = 15,
              settle_steps: int = DEFAULT_SETTLE_STEPS, seed: int = 42,
              replays: int = DEFAULT_REPLAYS,
              max_jump_rate: float = DEFAULT_MAX_JUMP_RATE,
              dataset_label: Optional[str] = None) -> Dict[str, object]:
    """Score every arm and the untrained baseline on one dataset, as one table.

    Each arm gets two views of the same data, and they answer different questions:

    * ``offline_best_x`` and the rest of the sequence outcome are the **gate** -- how far a
      run gets when its decisions are a sequence, in the emulator's own units;
    * ``budget`` is the supporting chunk-decision view -- whether the arm can catch the
      teacher's jumps at all within a jump budget.

    The teacher's trajectory is traced once and handed to every arm, so all arms are scored
    against the same windows and their outcomes are comparable decision for decision. The
    baseline is not optional: it is built here, from the same initialization a trained arm
    starts from, and it is the row every paired delta is measured against.
    """
    arms = list(arms)
    if any(arm.name == UNTRAINED_ARM for arm in arms):
        raise ValueError(f"{UNTRAINED_ARM!r} is reserved for the baseline arm")

    shards = dataset if isinstance(dataset, list) else _load_shards(dataset)
    if not shards:
        raise ValueError("pre-screen needs at least one shard")
    if stride < 1:
        raise ValueError("stride must be positive")

    traces = teacher_traces(shards, stride)
    table = [Arm(UNTRAINED_ARM, untrained_model(seed), detail={"role": "untrained_baseline"})]
    table.extend(arms)

    reports = []
    for arm in table:
        report = sequence_report(
            arm.name, arm.model, shards, stride, settle_steps, seed, replays,
            decoder_config=arm.decoder, traces=traces, detail=arm.detail,
        )
        # The budget view reads the same frames, so it is cheap next to the full replay and
        # stays attached to the arm it belongs to.
        report["budget"] = arm_report(arm, shards, stride, settle_steps, seed, replays, max_jump_rate)
        report["role"] = arm.detail.get("role", "candidate")
        reports.append(report)

    baseline = reports[0]
    paired = [paired_deltas(baseline, report, metric="jump_sequence_recall") for report in reports[1:]]
    by_arm = {row["arm"]: row for row in paired}
    verdicts = [
        prescreen_verdict(report, by_arm.get(report["arm"]), max_jump_rate=max_jump_rate)
        for report in reports[1:]
    ]
    paired_budget = [
        paired_deltas(baseline["budget"], report["budget"], metric="jump_recall")
        for report in reports[1:]
    ]

    return {
        "protocol": {
            "metric": "teacher_forced_sequence",
            "supporting_metric": "jump_recall_at_bounded_rate",
            "dataset": dataset_label,
            "dataset_provenance": dataset_provenance(dataset) if isinstance(dataset, str) else None,
            "stride": stride,
            "settle_steps": settle_steps,
            "seed": seed,
            "replays": replays,
            "replay_seeds": [seed + replay for replay in range(replays)],
            "max_jump_rate": max_jump_rate,
            "decisions_per_replay": reports[0]["budget"]["decisions_per_replay"],
            "teacher": reports[0]["teacher"],
            "thresholds": {
                "min_jump_recall": PRESCREEN_MIN_JUMP_RECALL,
                "min_recall_delta": PRESCREEN_MIN_RECALL_DELTA,
            },
            "baseline": UNTRAINED_ARM,
        },
        "table": reports,
        "paired": paired,
        "paired_budget": paired_budget,
        "verdicts": verdicts,
    }


def format_table(report: Dict[str, object]) -> str:
    """Render the pre-screen as the tables a reader actually compares rows in."""
    protocol = report["protocol"]
    teacher = protocol["teacher"]
    paired_by_arm = {row["arm"]: row for row in report["paired"]}
    verdict_by_arm = {row["arm"]: row for row in report["verdicts"]}

    lines = [
        "pre-screen: teacher-forced sequence (the gate), with the chunk-decision budget "
        "view below it",
        (f"dataset {protocol['dataset']}  cadence {protocol['stride']}  "
         f"settle {protocol['settle_steps']}  seed {protocol['seed']}  replays {protocol['replays']}  "
         f"teacher {teacher['required_jumps']} required jumps, max_x {teacher['max_x']}, "
         f"reached end {teacher['reached_end']}"),
        "",
        f"{'arm':<22}{'jump rec':>9}{'per replay':>22}{'paired delta vs untrained':>28}"
        f"{'missed':>8}{'spurious':>10}{'done':>6}{'best_x':>9}  verdict",
    ]
    for row in report["table"]:
        per_replay = " ".join(_decimal(value, 2) for value in row["jump_sequence_recall"]["values"])
        paired = paired_by_arm.get(row["arm"])
        if paired and paired["mean_delta"] is not None:
            delta_text = f"{paired['mean_delta']:+.3f} ({paired['improved']}/{paired['paired_replays']} seeds)"
        else:
            delta_text = "-"
        verdict = verdict_by_arm.get(row["arm"])
        lines.append(
            f"{row['arm']:<22}{_number(row['jump_sequence_recall']['mean']):>9}{per_replay:>22}"
            f"{delta_text:>28}{_number(row['missed_jumps']['mean']):>8}"
            f"{_number(row['spurious_jumps']['mean']):>10}"
            f"{_number(row['offline_completion_rate']):>6}"
            f"{_integer(row['offline_best_x']['mean']):>9}  "
            f"{'-' if verdict is None else ('PASS' if verdict['pass'] else 'fail')}"
        )
    lines.append(
        "  jump rec = teacher jumps taken, decided in sequence. best_x is the pessimistic "
        "scenario:"
    )
    lines.append(
        "  teacher-forced, so it stops at the first missed teacher jump and saturates for "
        "weak arms."
    )

    lines.append("")
    lines.append(
        f"budget view: jump-chunk recall at max_jump_rate {protocol['max_jump_rate']} "
        f"(chance = the rate itself)"
    )
    lines.append(f"{'arm':<22}{'recall':>8}{'rate':>8}{'lift':>8}  per replay")
    for row in report["table"]:
        budget = row["budget"]
        per_replay = " ".join(_decimal(value, 3) for value in budget["jump_recall"]["values"])
        lines.append(
            f"{row['arm']:<22}{_number(budget['jump_recall']['mean']):>8}"
            f"{_number(budget['jump_rate']['mean']):>8}"
            f"{_number(budget['lift_over_random']['mean']):>8}  {per_replay}"
        )

    for verdict in report["verdicts"]:
        if verdict["pass"]:
            continue
        failed = [f"{c['criterion']}: {c['observed']} vs {c['threshold']}"
                  for c in verdict["criteria"] if not c["pass"]]
        lines.append(f"  {verdict['arm']}: " + "; ".join(failed))
    return "\n".join(lines)


def _integer(value) -> str:
    return "-" if value is None else f"{value:.0f}"


def _decimal(value, places: int) -> str:
    """A per-replay figure that may be undefined when a replay had nothing to measure."""
    return "-" if value is None else f"{value:.{places}f}"


def _number(value) -> str:
    return "-" if value is None else f"{value:.3f}"


def _load_shards(dataset) -> List[dict]:
    shards = list(iter_dataset(dataset))
    if not shards:
        raise ValueError("trajectory dataset contains no samples")
    return shards


def decision_evidence(model: DrosophilaConnectomeSNN, preprocessor: OmmatidiaVisionPreprocessor,
                      shards: Sequence[dict], stride: int, settle_steps: int, seed: int):
    """Replay the supervised chunks and record the macro decision evidence.

    For every supervised decision point this returns ``jump_evidence - run_evidence`` as
    the *decoder* computes it, together with the labelled action. Both RNGs are reseeded
    from ``seed`` first and no weights are updated, so the draws are a pure function of
    the seed -- which is what lets two arms be compared seed by seed.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    model.eval()
    diffs, labels = [], []
    with torch.no_grad():
        for shard in shards:
            model.reset_state()
            preprocessor.reset()
            frames = shard["frames"][::stride]
            actions = shard["actions"][::stride]
            for frame, action in zip(frames, actions):
                features, _ = preprocessor.process_frame(frame)
                accumulated_motor = torch.zeros(model.num_motor_ganglion)
                for _ in range(settle_steps):
                    spikes = preprocessor.generate_poisson_spikes(features)
                    motor_spikes, _ = model(spikes)
                    accumulated_motor += motor_spikes
                diffs.append(float(accumulated_motor[JUMP_ACTION] - accumulated_motor[RUN_ACTION]))
                labels.append(int(action) == JUMP_ACTION)
    return diffs, labels


def arm_from_checkpoint(name: str, path: str, margin: Optional[float] = None) -> Arm:
    """Rebuild an arm from a checkpoint written by ``pretrain.save_checkpoint``.

    The shipped margin is read from the checkpoint itself -- ``policy_config`` first,
    because that is what the closed loop honours when it loads the weights, then the
    training metadata -- so the pre-screen reports the decoder the controller would
    actually run rather than a reconstructed approximation of it.
    """
    payload = torch.load(path, weights_only=False)
    state = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
    model = DrosophilaConnectomeSNN()
    model.load_state_dict(state)
    model.eval()
    metadata = payload.get("pretraining", {}) if isinstance(payload, dict) else {}
    decoder = metadata.get("decoder", {}) if isinstance(metadata, dict) else {}
    shipped = (payload.get("policy_config", {}) or {}).get("macro_decoder", {}) \
        if isinstance(payload, dict) else {}
    if margin is None:
        margin = shipped.get("jump_margin", decoder.get("jump_margin"))
    return Arm(
        name=name,
        model=model,
        margin=margin,
        # The sequence replay runs the decoder this checkpoint would actually adopt, so the
        # chunk lengths and refractory period travel with the arm rather than defaulting.
        decoder=shipped,
        detail={
            "checkpoint": os.path.abspath(path),
            "visual_pathway": metadata.get("visual_pathway") if isinstance(metadata, dict) else None,
            "epochs": metadata.get("epochs") if isinstance(metadata, dict) else None,
            "learning_rate": metadata.get("learning_rate") if isinstance(metadata, dict) else None,
            "jump_margin": margin,
            "refractory_frames": shipped.get("refractory_frames"),
            "weight_deltas": metadata.get("weight_deltas") if isinstance(metadata, dict) else None,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score arms on the emulator-free jump-recall pre-screen"
    )
    parser.add_argument("--dataset", required=True, help="Trajectory dataset directory")
    parser.add_argument("--checkpoint", action="append", default=[],
                        metavar="NAME=PATH", help="Candidate checkpoint; repeatable")
    parser.add_argument("--stride", type=int, default=15, help="Action cadence")
    parser.add_argument("--settle-steps", type=int, default=DEFAULT_SETTLE_STEPS)
    parser.add_argument("--seed", type=int, default=42, help="First replay seed (arms pair on it)")
    parser.add_argument("--replays", type=int, default=DEFAULT_REPLAYS)
    parser.add_argument("--max-jump-rate", type=float, default=DEFAULT_MAX_JUMP_RATE)
    parser.add_argument("--output", default=None, help="Write the full report as JSON here")
    args = parser.parse_args()

    arms = []
    for spec in args.checkpoint:
        if "=" not in spec:
            raise SystemExit(f"--checkpoint expects NAME=PATH, got {spec!r}")
        name, path = spec.split("=", 1)
        arms.append(arm_from_checkpoint(name, path))

    report = prescreen(
        arms, args.dataset, stride=args.stride, settle_steps=args.settle_steps,
        seed=args.seed, replays=args.replays, max_jump_rate=args.max_jump_rate,
        dataset_label=os.path.abspath(args.dataset),
    )
    print(format_table(report))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"\nwrote {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
