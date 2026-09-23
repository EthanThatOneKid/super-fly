"""Is the ceiling in the decision rule, or in the representation?

Level 1-1 needs ~20 jumps and, under the replay's own fatality model, a missed one ends
the run, so completion needs per-jump reliability near 1: at 0.90 a run survives 20
decisions about 12% of the time, and at 0.84 -- where the best closed-loop arm sits,
teacher-forced -- about 3%. Closing that gap is either a **decoder** change (the rule
that turns motor spikes into a decision is discarding a distinction the network does
make) or a **representation** change (the distinction was never there to read). Those
have very different price tags, and the difference is measurable offline, with no
emulator and no training.

What this module measures is an *oracle ceiling*: for a fixed statistic, the best score
any decision rule could reach on the teacher's own chunks if the threshold and the
timing were chosen with the labels in hand. Because it is allowed to peek, it can only
overstate what a causal policy achieves -- which is what makes it a bound. If the bound
is below what completion needs, no amount of tuning that statistic will get there, and
the honest next move is somewhere else.

Three families are compared, all from one recording pass over the same teacher chunks:

* **motor evidence**, the two channels the shipped decoder reads (``RUN_ACTION``,
  ``JUMP_ACTION``). The decoder is handed their *sum* over the settle window; that sum is
  one of nine statistics tested here, so the family brackets the shipped rule and the
  obvious alternatives to it.
* **the population one layer upstream** (the central complex), reduced to label-free
  one-dimensional projections (top principal components, mean activity). If a projection
  of the central population separates the two chunk classes sharply while no statistic of
  the motor channels does, the information is present and the *readout* is what loses it.
* **a linear readout** of the same evidence, cross-validated, as supporting evidence
  only. It is deliberately *not* used as a ceiling: with ~99 chunks and hundreds of
  features an in-sample linear fit separates anything, which bounds nothing.

The decision rule is fixed before the measurement, so the result cannot be read as
confirmation either way:

* motor ceiling at or above :data:`COMPLETION_RELIABILITY` -> ``decoder``: the evidence
  can support completion-level decisions and the shipped rule is leaving them on the floor;
* motor ceiling below it but a central-population projection at or above -> ``readout``:
  the information is upstream and the motor readout is the bottleneck;
* both below -> ``representation``: nothing that reads this network can complete the level,
  and the next lever is the features or the weights, not the decision rule.

"At or above" is measured at *bounded* jump cost -- the best jump-chunk recall reachable
while keeping run-chunk recall at or above :data:`MIN_RUN_RECALL` -- because a statistic
that catches every jump by jumping on everything is the always-jump defect this project
has already been bitten by, and it is not a ceiling worth quoting.
"""

import argparse
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from macro_decoder import JUMP_ACTION, RUN_ACTION
from prescreen import Arm, arm_from_checkpoint, untrained_model, vision_preprocessor
from seed_policy import (
    DEFAULT_REPLAYS,
    DEV_SEEDS,
    GATE_ROLE,
    GATE_SEEDS,
    SeedSet,
    parse_seeds,
    resolve_seeds,
)
from simulation import DEFAULT_SETTLE_STEPS
from trajectory import dataset_provenance, iter_dataset

#: Per-jump reliability a controller needs for the ~20 required jumps to chain. 0.97 leaves
#: about a 54% chance of surviving all twenty, which is the first value where completion is
#: a coin flip rather than an outlier; 0.90 leaves 12% and 0.84 leaves 3%.
COMPLETION_RELIABILITY = 0.97
#: Run-chunk recall a statistic has to hold while catching jumps. Quoting a ceiling that
#: jumps on everything would repeat the always-jump exploit the sequence gate had to close.
MIN_RUN_RECALL = 0.95
#: How far a held-out population readout has to beat the motor ceiling before the loss is
#: placed upstream of the motor layer. Matches the pre-screen's paired-gain bar, so the two
#: instruments ask for the same size of effect.
READOUT_EVIDENCE_MARGIN = 0.10
#: Weighting for the ``leaky_early`` rule: an exponentially decaying accumulator, which is
#: what a membrane would do, as against the boxcar sum the shipped decoder uses.
LEAKY_DECAY = 0.5
#: Ridge strength for the Fisher direction, applied after normalising the covariance trace:
#: the population is 128-dimensional and there are ~500 frames, so the estimate is ill
#: conditioned without it.
LDA_RIDGE = 1e-3
#: Shrinkage options for the held-out population readout. The sweep exists so that a verdict
#: of "not separable" cannot be an artefact of an under-regularised direction.
LDA_SHRINKAGES = (1e-3, 1e-2, 1e-1, 1.0)
#: Ridge strength and fold count for the supporting cross-validated readout.
RIDGE_LAMBDA = 1.0
CV_FOLDS = 5
#: Motor statistics tested. ``sum`` is the shipped rule, so the family brackets it.
MOTOR_RULES = (
    "sum", "mean", "last_frame", "max_frame", "prefix_max", "prefix_min",
    "leaky_early", "contrast", "latency",
)
#: Statistics over the motor layer's *analog* drive -- the pre-threshold input current the
#: LIF integrates, read at the same instant as the spikes and quantised by nothing. The
#: shipped rule sums a *binary* spike train, so a five-frame window can only produce six
#: values per channel; a threshold on top of that can only sit between integers.
DRIVE_RULES = (
    "drive_sum", "drive_mean", "drive_leaky_recency", "drive_leaky_early",
    "drive_normalized", "drive_recency_normalized", "drive_range", "drive_last",
)
#: Rules that take a decay parameter. The decay is *reported*, not chosen: every value here
#: appears in the table, so the one that wins is visible rather than selected out of sight.
DECAY_RULES = ("drive_leaky_recency", "drive_leaky_early", "drive_recency_normalized")
DRIVE_DECAYS = (0.0, 0.25, 0.5, 0.75)
#: The decay a shipped decoder would use if it adopted one of the leaky rules -- the middle
#: of :data:`DRIVE_DECAYS`, quoted so the choice is a parameter and not a search result.
DEFAULT_DRIVE_DECAY = 0.5
#: Statistics over the *binary spike* series with the shapes the drive rules test, so a
#: difference between the two families is attributable to the quantisation and not the shape.
SPIKE_SHAPE_RULES = ("spike_range", "spike_recency_normalized")
#: How a per-frame trajectory is reduced to one number per chunk. Every rule here maps a
#: chunk's frame sequence to a single statistic; ``oracle_frame`` is the timing oracle and
#: is handled separately because it reads the label.
REDUCTIONS = ("sum", "mean", "last_frame", "max_frame", "prefix_max", "prefix_min",
              "leaky_early", "latency")
UNTRAINED_ARM = "untrained"


def load_shards(dataset) -> List[dict]:
    """Load a trajectory dataset directory, refusing an empty one."""
    shards = list(iter_dataset(dataset))
    if not shards:
        raise ValueError("oracle ceiling needs at least one shard")
    return shards


def _int_list(value: str) -> List[int]:
    """Parse a comma-separated integer list, the form every CLI here accepts."""
    return [int(token.strip()) for token in str(value).split(",") if token.strip()]


# --------------------------------------------------------------------------- reduction


def reduce_trajectory(trajectory: np.ndarray, rule: str) -> float:
    """Collapse one chunk's per-frame evidence into the statistic a rule would decide on.

    ``trajectory`` is ``(frames, channels)``: the per-frame evidence the network produced
    *before* anything summed it, so the shipped decoder's statistic is reconstructible and
    every alternative is measured on identical draws. A rule that is undefined for a single
    frame (a prefix, a latency) degenerates to the frame itself rather than raising, because
    a one-frame window is a legitimate configuration.
    """
    if trajectory.ndim == 1:
        trajectory = trajectory[:, None]
    series = trajectory[:, 0] if trajectory.shape[1] == 1 else _series(trajectory)
    n = len(series)
    if rule == "sum":
        return float(series.sum())
    if rule == "mean":
        return float(series.mean())
    if rule == "last_frame":
        return float(series[-1])
    if rule == "max_frame":
        return float(series.max())
    if rule == "prefix_max":
        return float(np.cumsum(series).max())
    if rule == "prefix_min":
        return float(np.cumsum(series).min())
    if rule == "leaky_early":
        weights = LEAKY_DECAY ** np.arange(n)
        return float(np.sum(series * weights))
    if rule == "contrast":
        return float(_contrast(trajectory))
    if rule == "latency":
        # The frame index of the first crossing, negated so that larger means "sooner".
        # A chunk that never crosses ranks last rather than raising.
        crossed = np.flatnonzero(np.cumsum(series) > 0.0)
        return -(float(crossed[0]) if len(crossed) else float(n + 1))
    if rule == "spike_range":
        return float(series.max() - series.min())
    if rule == "spike_recency_normalized":
        weights = _recency_weights(n)
        total = float(np.abs(trajectory[:, JUMP_ACTION]).sum() + np.abs(trajectory[:, RUN_ACTION]).sum())
        return (float(np.sum(series * weights)) / total) if total > 0.0 else 0.0
    raise ValueError(f"unknown reduction rule {rule!r}")


def _recency_weights(n: int, decay: float = DEFAULT_DRIVE_DECAY) -> np.ndarray:
    """Weights that make a rule *timing-aware*: later frames in the window count more.

    A decision is committed on the frame it is made on, so the end of the settle window is
    the most recent evidence the controller has. A boxcar sum is timing-*blind*: it cannot
    tell a drive that rose into the decision from one that fell out of it.
    """
    return decay ** np.arange(n)[::-1]


def reduce_drive(trajectory: np.ndarray, rule: str, decay: float = DEFAULT_DRIVE_DECAY) -> float:
    """Collapse one chunk's per-frame *analog motor drive* into the statistic a rule reads.

    ``trajectory`` is ``(frames, channels)`` of the motor layer's pre-threshold input current:
    the same quantity the LIF integrates into its membrane. It is continuous, so a statistic
    of it is not limited to the few levels a spike count over the window can take -- which is
    exactly what a fitted decision threshold needs in order to sit where the classes divide.
    """
    trajectory = np.asarray(trajectory, dtype=np.float64)
    if trajectory.ndim == 1:
        trajectory = trajectory[:, None]
    jump = trajectory[:, JUMP_ACTION]
    run = trajectory[:, RUN_ACTION]
    diff = jump - run
    n = len(diff)
    weights = _recency_weights(n, decay)
    if rule == "drive_sum":
        return float(diff.sum())
    if rule == "drive_mean":
        return float(diff.mean())
    if rule == "drive_leaky_recency":
        return float((jump * weights).sum() - (run * weights).sum())
    if rule == "drive_leaky_early":
        return float((jump * weights[::-1]).sum() - (run * weights[::-1]).sum())
    if rule == "drive_normalized":
        total = float(np.abs(jump).sum() + np.abs(run).sum())
        return float(diff.sum() / total) if total > 0.0 else 0.0
    if rule == "drive_recency_normalized":
        # Both asks at once: recency-weighted, and divided by the window's own activity so a
        # network that drives one channel hard cannot lift the statistic without lifting the
        # thing it is divided by.
        total = float((np.abs(jump) * weights).sum() + (np.abs(run) * weights).sum())
        weighted = float((jump * weights).sum() - (run * weights).sum())
        return weighted / total if total > 0.0 else 0.0
    if rule == "drive_range":
        # Within-window spread, and the causal cousin of the timing oracle: the oracle picks the
        # best frame *with the label*, and this picks how strongly the window moved at all.
        return float(diff.max() - diff.min())
    if rule == "drive_last":
        return float(diff[-1])
    raise ValueError(f"unknown drive rule {rule!r}")


def _series(trajectory: np.ndarray) -> np.ndarray:
    """The scalar series a multi-channel trajectory reduces to: jump minus run."""
    if trajectory.shape[1] < 4:
        raise ValueError("a motor trajectory carries the 4 motor neurons")
    return trajectory[:, JUMP_ACTION] - trajectory[:, RUN_ACTION]


def _contrast(trajectory: np.ndarray) -> float:
    """Normalised contrast between the two channels, in [-1, 1].

    Scale-free on purpose: the shipped statistic is a *sum* of bounded per-frame spikes, so a
    network that drives one channel hard raises both the signal and the ceiling it saturates
    at. Dividing by the total activity keeps the ordering while removing the level, and a
    chunk with no activity at all reads as zero rather than dividing by it.
    """
    jump = float(np.abs(trajectory[:, JUMP_ACTION]).sum())
    run = float(np.abs(trajectory[:, RUN_ACTION]).sum())
    total = jump + run
    return (jump - run) / total if total > 0.0 else 0.0


def oracle_frame_statistic(trajectory: np.ndarray, label: bool) -> float:
    """The best single frame of the chunk, chosen *using the label*.

    This is the timing oracle, and it is generous on purpose: it hands every chunk the
    frame that most supports the answer it is supposed to give, which no causal policy can
    do. It bounds the family from above.
    """
    series = _series(trajectory)
    return float(series.max() if label else series.min())


# --------------------------------------------------------------------------- scoring


def auc(scores: Sequence[float], labels: Sequence[bool]) -> Optional[float]:
    """Rank-based AUC (Mann-Whitney), with ties sharing their average rank.

    0.5 is chance and 1.0 is perfect ordering, independent of where a threshold sits --
    which separates "the statistic orders the two classes" from "the boundary is in the
    wrong place", the distinction the balanced-accuracy sweep alone cannot make.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    pos_rank_sum = ranks[labels].sum()
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _margins(scores: np.ndarray) -> List[float]:
    """Candidate boundaries: every observed value, plus a sentinel strictly beyond each end.

    The decision rule is ``score > margin``, so the sentinels are what express the two
    constant rules -- and they are finite, because an infinite margin is not valid JSON and
    was already the shape of one shipped defect.
    """
    return sorted(set(scores.tolist() + [float(scores.min()) - 1.0, float(scores.max()) + 1.0]))


def best_margin(scores: Sequence[float], labels: Sequence[bool],
                method: str = "balanced_accuracy") -> Dict[str, object]:
    """The boundary that scores best on the decisions it is handed.

    ``balanced_accuracy`` maximises the mean of the two per-class recalls, so the majority
    class cannot win; ``jump_rate`` matches the labelled jump frequency instead. Both are
    *fits*: the caller is responsible for saying whose data they were fitted on.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    n_jump, n_run = int(labels.sum()), int((~labels).sum())
    if n_jump == 0 or n_run == 0:
        raise ValueError("a decision boundary needs both chunk classes present")
    target = n_jump / len(labels)
    best = None
    for margin in _margins(scores):
        predicted = scores > margin
        jump_recall = float(np.sum(predicted & labels) / n_jump)
        run_recall = float(np.sum(~predicted & ~labels) / n_run)
        balanced = 0.5 * (jump_recall + run_recall)
        score = balanced if method == "balanced_accuracy" else -abs(float(predicted.mean()) - target)
        # Ties keep the *larger* margin, i.e. the boundary that jumps less. The shipped
        # decoder documents the same tie-break ("a tie falls back to running"), and it is the
        # opposite of the one that shipped an always-jump decoder here: a statistic with no
        # separation must read as "never jump", not as "jump on everything".
        if best is None or score >= best["score"]:
            best = {"score": score, "margin": float(margin), "balanced_accuracy": balanced,
                    "jump_recall": jump_recall, "run_recall": run_recall,
                    "jump_rate": float(predicted.mean())}
    return best


def bounded_margin(scores: Sequence[float], labels: Sequence[bool],
                   min_run_recall: float = MIN_RUN_RECALL) -> Optional[Dict[str, object]]:
    """The boundary that catches the most jumps while keeping run recall at or above the floor.

    This is the decisive operating point: a statistic that catches every jump by jumping on
    everything saves no run chunks, and this is what refuses to call that a ceiling.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    n_jump, n_run = int(labels.sum()), int((~labels).sum())
    if n_jump == 0 or n_run == 0:
        return None
    best = None
    for margin in _margins(scores):
        predicted = scores > margin
        jump_recall = float(np.sum(predicted & labels) / n_jump)
        run_recall = float(np.sum(~predicted & ~labels) / n_run)
        if run_recall + 1e-12 < min_run_recall:
            continue
        if best is None or jump_recall > best["jump_recall"]:
            best = {"jump_recall": jump_recall, "run_recall": run_recall,
                    "margin": float(margin), "jump_rate": float(predicted.mean())}
    return best


def sweep_threshold(scores: Sequence[float], labels: Sequence[bool],
                    min_run_recall: float = MIN_RUN_RECALL) -> Dict[str, object]:
    """Best achievable operating points for one statistic, with the threshold fitted.

    Two numbers come out. ``balanced_accuracy`` is the best mean of the two per-class
    recalls over every candidate boundary -- the standard "how well does this statistic
    separate the classes" reading, and optimistic here because the boundary is chosen on
    the same decisions it then scores (which is the correct direction for a ceiling).
    ``jump_recall_at_min_run_recall`` is the decisive one: the most jump chunks any
    boundary can catch while still recognising at least ``min_run_recall`` of the run
    chunks, so a jump-on-everything rule cannot win.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    n_jump, n_run = int(labels.sum()), int((~labels).sum())
    if n_jump == 0 or n_run == 0:
        return {"balanced_accuracy": None, "jump_recall": None, "run_recall": None,
                "jump_recall_at_min_run_recall": None, "margin": None, "jump_rate": None,
                "note": "only one chunk class is present"}

    best = best_margin(scores, labels)
    best_bounded = bounded_margin(scores, labels, min_run_recall)
    return {
        "balanced_accuracy": round(best["balanced_accuracy"], 4),
        "jump_recall": round(best["jump_recall"], 4),
        "run_recall": round(best["run_recall"], 4),
        "margin": round(best["margin"], 6),
        "jump_rate": round(best["jump_rate"], 4),
        "jump_recall_at_min_run_recall": (round(best_bounded["jump_recall"], 4)
                                          if best_bounded else None),
        "jump_rate_at_min_run_recall": (round(best_bounded["jump_rate"], 4)
                                        if best_bounded else None),
        "min_run_recall": min_run_recall,
    }


def ceiling(scores: Sequence[float], labels: Sequence[bool],
            min_run_recall: float = MIN_RUN_RECALL) -> Dict[str, object]:
    """Ordering quality plus the fitted-boundary operating points for one statistic."""
    return {
        "auc": None if auc(scores, labels) is None else round(auc(scores, labels), 4),
        **sweep_threshold(scores, labels, min_run_recall),
    }


# --------------------------------------------------------------------------- recording


def record_chunks(model, preprocessor, shards: Sequence[dict], stride: int, settle_steps: int,
                  seed: int) -> List[Dict[str, object]]:
    """Replay the supervised chunks and keep the *per-frame* evidence, not the sum.

    Recording every frame of the settle window is what makes the aggregation rules
    comparable: the shipped decoder's statistic is one reduction of these traces and every
    alternative rule is a different reduction of the same draws. The central-complex
    population is recorded alongside, from the same forward passes, so the upstream family
    costs no extra emulation.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    model.eval()
    rows: List[Dict[str, object]] = []
    with torch.no_grad():
        for shard in shards:
            model.reset_state()
            preprocessor.reset()
            frames = shard["frames"][::stride]
            actions = shard["actions"][::stride]
            for frame, action in zip(frames, actions):
                features, _ = preprocessor.process_frame(frame)
                motor = np.zeros((settle_steps, 4), dtype=np.float64)
                drive = np.zeros((settle_steps, 4), dtype=np.float64)
                central = np.zeros((settle_steps, model.num_central_complex), dtype=np.float64)
                for step in range(settle_steps):
                    spikes = preprocessor.generate_poisson_spikes(features)
                    motor_spikes, activations = model(spikes)
                    motor[step] = motor_spikes.detach().cpu().numpy()
                    central[step] = activations["central_complex"].detach().cpu().numpy()
                    drive[step] = motor_drive(model, activations["central_complex"])
                rows.append({"label": int(action) == JUMP_ACTION, "motor": motor,
                             "drive": drive, "central": central})
    return rows


# --------------------------------------------------------------------------- families


def motor_drive(model, central_spikes) -> np.ndarray:
    """The motor layer's pre-threshold input current -- exactly what the LIF integrates.

    ``connectome.LIFNeuronLayer.forward`` computes this, thresholds it, and throws the value
    away: the spike is a one-bit report of a continuous quantity. Reading it costs one matrix
    product inside a forward pass that has already happened, it uses no labels, and it reads
    no frame the spike does not also come from -- so it is causal and free.
    """
    current = torch.matmul(model.layer3_4.weight, central_spikes)
    current = current - current.mean()
    return (current * model.layer3_4.current_gain).detach().cpu().numpy().astype(np.float64)


def _drive_oracle(trajectory: np.ndarray, label: bool) -> float:
    """Timing oracle over the drive series, so the analog family has the same upper bound."""
    series = np.asarray(trajectory, dtype=np.float64)
    series = series[:, JUMP_ACTION] - series[:, RUN_ACTION]
    return float(series.max() if label else series.min())


def _chunk_features(rows: Sequence[Dict[str, object]], field: str) -> np.ndarray:
    """One flattened feature row per *chunk* (the readout is per decision, not per frame)."""
    return np.stack([np.asarray(row[field], dtype=np.float64).reshape(-1) for row in rows])


def principal_components(rows: Sequence[Dict[str, object]], components: int = 3) -> np.ndarray:
    """Label-free principal directions of the central-complex population.

    The basis is fitted without the labels -- a projection chosen *with* them would answer
    its own question -- so a projection that separates the chunk classes is evidence that
    the population carries the distinction, not evidence that a search found one.
    """
    stacked = np.concatenate([row["central"] for row in rows], axis=0)
    centred = stacked - stacked.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    return vt[:components]


def projection_trajectory(row: Dict[str, object], direction: np.ndarray,
                          reference: np.ndarray) -> np.ndarray:
    """One chunk's per-frame scalar projection of its central-complex activity."""
    return (row["central"] - reference) @ direction


def _ridge_weights(features: np.ndarray, labels: np.ndarray,
                   lam: float = RIDGE_LAMBDA) -> np.ndarray:
    """Closed-form ridge weights onto +/-1 targets.

    No new dependency: the solver is a dense linear solve, and centring keeps the
    intercept out of the weights.
    """
    centred = features - features.mean(axis=0, keepdims=True)
    targets = labels.astype(np.float64) * 2.0 - 1.0
    targets = targets - targets.mean()
    gram = centred.T @ centred + lam * np.eye(centred.shape[1])
    return np.linalg.solve(gram, centred.T @ targets)


def ridge_scores(features: np.ndarray, labels: np.ndarray,
                 lam: float = RIDGE_LAMBDA) -> np.ndarray:
    """In-sample ridge scores; the projection uses the sample's own mean."""
    return ((features - features.mean(axis=0, keepdims=True))
            @ _ridge_weights(features, labels, lam))


def cross_validated_boundary(features: np.ndarray, labels: np.ndarray,
                             folds: int = CV_FOLDS, lam: float = RIDGE_LAMBDA) -> Dict[str, object]:
    """Fold-wise fit of a readout, scored on the folds it never saw.

    This is the honest companion to the oracle sweeps: it says what a linear readout of
    this evidence actually achieves at this sample size. It is reported as evidence, never
    as a bound -- an in-sample linear fit of ~600 features on ~99 chunks can separate
    anything, which is why it is not quoted as a ceiling.
    """
    labels = np.asarray(labels, dtype=bool)
    n = len(labels)
    if n < folds or labels.all() or not labels.any():
        return {"folds": folds, "auc": None, "jump_recall": None, "run_recall": None,
                "balanced_accuracy": None, "note": "not enough chunks to fold"}
    fold_of = contiguous_folds(n, folds)
    scores = np.zeros(n, dtype=np.float64)
    for fold in range(folds):
        train, test = fold_of != fold, fold_of == fold
        if labels[train].all() or not labels[train].any() or not test.any():
            continue
        # Weights are fitted on the training folds only, and the projection is centred with
        # the *training* mean: nothing about the held-out folds reaches the fit.
        weights = _ridge_weights(features[train], labels[train], lam)
        scores[test] = (features[test] - features[train].mean(axis=0, keepdims=True)) @ weights
    return {
        # Marked as fitted, and skipped by the family ceilings: a linear readout of ~600
        # features is not a bound on what a threshold rule could do, so letting it set the
        # family's headline number would quote a fit as a ceiling.
        "fitted": True,
        "folds": folds,
        "auc": None if auc(scores, labels) is None else round(auc(scores, labels), 4),
        **sweep_threshold(scores, labels),
    }


def lda_direction(rows: Sequence[Dict[str, object]], ridge: float = LDA_RIDGE) -> np.ndarray:
    """The single best linear direction for separating the two chunk classes.

    Label-aware on purpose. The central population has 128 dimensions and 99 chunks, so a
    *label-free* projection (a principal component) can easily miss the separating direction
    and would understate what the population carries. The Fisher direction is the optimal
    one-dimensional linear readout, so scoring it makes the central family an honest ceiling
    for one-dimensional population readouts rather than a search over lucky directions.
    Heavier readouts than that cannot be bounded from 99 chunks, and the report says so.
    """
    pooled = np.concatenate([row["central"] for row in rows], axis=0)
    labels = np.concatenate([np.full(len(row["central"]), bool(row["label"])) for row in rows])
    mean_jump = pooled[labels].mean(axis=0)
    mean_run = pooled[~labels].mean(axis=0)
    centred = pooled - pooled.mean(axis=0, keepdims=True)
    covariance = centred.T @ centred / max(len(pooled) - 1, 1)
    covariance = covariance / max(float(np.trace(covariance)) / covariance.shape[0], 1e-12)
    return np.linalg.solve(covariance + ridge * np.eye(covariance.shape[0]),
                           mean_jump - mean_run)


def _reduction_table(trajectories: Sequence[np.ndarray], labels: Sequence[bool]) -> Dict[str, object]:
    """Score every reduction rule, plus the timing oracle, on one projected trajectory set."""
    table = {rule: ceiling([reduce_trajectory(np.asarray(trajectory)[:, None], rule)
                            for trajectory in trajectories], labels)
             for rule in REDUCTIONS}
    table["oracle_frame"] = ceiling(
        [float(np.asarray(trajectory).max() if label else np.asarray(trajectory).min())
         for trajectory, label in zip(trajectories, labels)], labels)
    return table


def contiguous_folds(n: int, folds: int) -> np.ndarray:
    """Fold ids in contiguous blocks, not interleaved.

    Chunks are consecutive stretches of the teacher's trajectory, so chunk ``i`` and chunk
    ``i + 1`` are adjacent frames of the same level. A modulo split puts a held-out chunk's
    immediate neighbours in the training set, and a readout can then score well by leaning on
    the block of trajectory the chunk sits in rather than on the distinction being measured.
    Contiguous blocks make the held-out chunks genuinely unseen -- measured, not assumed:
    interleaving lifted one saturated arm's held-out readout to a perfect 1.000 that does not
    survive the block split.
    """
    return (np.arange(n) * folds // max(n, 1)).astype(int)


def _held_out_recalls(predictions: np.ndarray, scored: np.ndarray,
                      labels: np.ndarray) -> Dict[str, object]:
    """Per-class recall over the chunks a fold actually scored, plus their ordering."""
    jump = scored & labels
    run = scored & ~labels
    n_jump, n_run = int(labels[scored].sum()), int((~labels[scored]).sum())
    jump_recall = float(predictions[jump].sum() / n_jump) if n_jump else None
    run_recall = float((~predictions[run]).sum() / n_run) if n_run else None
    balanced = (0.5 * (jump_recall + run_recall)
                if jump_recall is not None and run_recall is not None else None)
    return {"chunks_scored": int(scored.sum()), "jump_recall": jump_recall,
            "run_recall": run_recall, "balanced_accuracy": balanced}


def nearest_neighbour_readout(rows: Sequence[Dict[str, object]],
                              folds: int = CV_FOLDS) -> Dict[str, object]:
    """A nonlinear probe of the population, held out by chunk: 1-NN on mean activity.

    A linear direction is the best *linear* readout, so a weak held-out Fisher result could
    in principle mean the distinction is nonlinear rather than absent. One nearest neighbour
    makes no linearity assumption and is the cheapest way to check that objection, so the
    verdict does not rest on the linear probe alone.
    """
    features = np.stack([np.asarray(row["central"], dtype=np.float64).mean(axis=0) for row in rows])
    labels = np.array([bool(row["label"]) for row in rows])
    n = len(labels)
    if n < folds or labels.all() or not labels.any():
        return {"fitted": True, "auc": None, "jump_recall": None, "run_recall": None,
                "balanced_accuracy": None, "note": "not enough chunks"}
    fold_of = contiguous_folds(n, folds)
    predictions = np.zeros(n, dtype=bool)
    scored = np.zeros(n, dtype=bool)
    for fold in range(folds):
        test = np.flatnonzero(fold_of == fold)
        train = np.flatnonzero(fold_of != fold)
        if not len(test) or labels[train].all() or not labels[train].any():
            continue
        distances = ((features[test][:, None, :] - features[train][None, :, :]) ** 2).sum(axis=2)
        predictions[test] = labels[train[distances.argmin(axis=1)]]
        scored[test] = True
    return {"fitted": True, "probe": "1-nearest-neighbour on mean central activity",
            "folds": folds, **_held_out_recalls(predictions, scored, labels)}


def cross_validated_lda(rows: Sequence[Dict[str, object]], reduction: str = "mean",
                        folds: int = CV_FOLDS,
                        shrinkages: Sequence[float] = LDA_SHRINKAGES) -> Dict[str, object]:
    """The population readout, measured on chunks it was not fitted on.

    This is the number the decoder-versus-representation verdict actually turns on. The
    Fisher direction fitted on all frames scores 1.000 against those same frames by
    construction, and a 128-dimensional direction fitted over ~500 frames is optimistic even
    when the distinction is real, so quoting that would be a fit quoted as evidence. Here the
    direction comes from the training chunks' frames only, the boundary comes from the
    training chunks only, and the recalls are measured on the held-out chunks -- the same
    discipline the pre-screen had to adopt after a refitted measurement read the same for an
    untrained network as for a trained one.

    Shrinkage is swept and the *held-out* score picks the winner, which is generous by
    construction: a verdict of "not separable" then cannot be an artefact of an under-regularised
    direction, it can only overstate how separable the population is.
    """
    labels = np.array([bool(row["label"]) for row in rows])
    n = len(labels)
    if n < folds or labels.all() or not labels.any():
        return {"fitted": True, "folds": folds, "auc": None, "jump_recall": None,
                "run_recall": None, "balanced_accuracy": None, "note": "not enough chunks"}
    fold_of = contiguous_folds(n, folds)
    trials = []
    for ridge in shrinkages:
        predictions = np.zeros(n, dtype=bool)
        scored = np.zeros(n, dtype=bool)
        fold_aucs: List[float] = []
        for fold in range(folds):
            train = np.flatnonzero(fold_of != fold)
            test = np.flatnonzero(fold_of == fold)
            if not len(test) or labels[train].all() or not labels[train].any():
                continue
            train_rows = [rows[index] for index in train]
            direction = lda_direction(train_rows, ridge)
            reference = np.concatenate([row["central"] for row in train_rows], axis=0).mean(axis=0)
            train_scores = np.array([
                reduce_trajectory(np.asarray(projection_trajectory(row, direction, reference))[:, None],
                                  reduction) for row in train_rows])
            test_scores = np.array([
                reduce_trajectory(np.asarray(projection_trajectory(rows[index], direction, reference))[:, None],
                                  reduction) for index in test])
            # The boundary is placed at the *bounded* operating point on the training folds,
            # so the held-out recalls answer the same question the motor ceiling answers and a
            # statistic with no operating point at all is left unscored rather than reported
            # at a jump-on-everything boundary.
            boundary = bounded_margin(train_scores, labels[train])
            if boundary is None:
                continue
            predictions[test] = test_scores > boundary["margin"]
            scored[test] = True
            fold_auc = auc(test_scores, labels[test])
            if fold_auc is not None:
                fold_aucs.append(fold_auc)
        recalls = _held_out_recalls(predictions, scored, labels)
        trials.append({
            "shrinkage": ridge,
            "auc": round(float(np.mean(fold_aucs)), 4) if fold_aucs else None,
            **{key: (None if value is None else round(float(value), 4))
               for key, value in recalls.items() if key != "chunks_scored"},
            "chunks_scored": recalls["chunks_scored"],
        })
    scored_trials = [trial for trial in trials if trial["balanced_accuracy"] is not None]
    if not scored_trials:
        return {"fitted": True, "folds": folds, "auc": None, "jump_recall": None,
                "run_recall": None, "balanced_accuracy": None,
                "note": "no fold could place a boundary at the run-recall floor: this readout "
                        "has no operating point to score"}
    # The winner is picked by the *held-out* balanced accuracy, so the reported number is an
    # upper bound on what a properly tuned Fisher readout achieves at this sample size.
    best = max(scored_trials, key=lambda trial: trial["balanced_accuracy"])
    return {"fitted": True, "folds": folds, "reduction": reduction,
            "shrinkage": best["shrinkage"], "shrinkage_sweep": trials,
            "auc": best["auc"], "jump_recall": best["jump_recall"],
            "run_recall": best["run_recall"], "balanced_accuracy": best["balanced_accuracy"],
            "chunks_scored": best["chunks_scored"],
            "note": "direction and boundary fitted on training chunks; shrinkage picked on "
                    "held-out balanced accuracy, so this is an upper bound on a tuned readout"}


def family_report(rows: Sequence[Dict[str, object]], rules: Sequence[str] = MOTOR_RULES) -> Dict[str, object]:
    """Score the motor evidence and the central population with one instrument.

    The motor family is what the shipped decoder reads: fixed one-dimensional statistics of
    the two trained channels, each with its threshold fitted, plus the timing oracle and the
    best linear functional of the traces. The central family is the population one layer
    upstream, reduced to one dimension -- through the Fisher direction and, as supporting
    label-free points of comparison, the leading principal components and mean activity -- and
    then reduced by the same rules, so the families differ in their evidence and not in how
    they are scored.
    """
    labels = [bool(row["label"]) for row in rows]
    motor: Dict[str, object] = {}
    for rule in rules:
        motor[rule] = ceiling([reduce_trajectory(row["motor"], rule) for row in rows], labels)
    # Statistics over the *binary spike* series with the shapes the drive rules test, so a
    # difference between the two families is attributable to the quantisation and not the
    # shape. They read the spikes, so they belong to this family, not the drive one.
    for rule in SPIKE_SHAPE_RULES:
        motor[rule] = ceiling([reduce_trajectory(row["motor"], rule) for row in rows], labels)
    motor["oracle_frame"] = ceiling(
        [oracle_frame_statistic(row["motor"], label) for row, label in zip(rows, labels)], labels)
    # The best linear functional of the motor traces, threshold fitted in sample. Twenty
    # features against ninety-nine chunks, so the fit is mildly optimistic -- and that is the
    # direction a ceiling should be wrong in: it is the last word on what any one-dimensional
    # linear statistic of this evidence can do.
    motor["linear_oracle"] = ceiling(ridge_scores(_chunk_features(rows, "motor"), np.asarray(labels)),
                                     labels)
    motor["linear_cv"] = cross_validated_boundary(_chunk_features(rows, "motor"), np.asarray(labels))

    # The analog drive family: the same question asked of the quantity the spikes quantise.
    # Every decay in ``DRIVE_DECAYS`` is reported rather than searched, so a rule that only wins
    # at one decay is visible as such, and a leaky rule is not silently the outcome of a sweep.
    # A caller may hand in rows recorded without the drive (synthetic fixtures, or an older
    # recording); the family is then simply absent rather than a family of nothing, so the
    # ceiling it reports stays honest and the verdict falls back to the spike statistics.
    drive: Dict[str, object] = {}
    if all("drive" in row for row in rows):
        for decay in DRIVE_DECAYS:
            for rule in DRIVE_RULES:
                if rule not in DECAY_RULES and decay != DRIVE_DECAYS[0]:
                    continue
                name = f"{rule}@{decay}" if rule in DECAY_RULES else rule
                drive[name] = ceiling([reduce_drive(row["drive"], rule, decay) for row in rows], labels)
        drive["oracle_frame"] = ceiling(
            [_drive_oracle(row["drive"], label) for row, label in zip(rows, labels)], labels)

    reference = np.concatenate([row["central"] for row in rows], axis=0).mean(axis=0)
    projections = {"lda": lda_direction(rows)}
    for index, direction in enumerate(principal_components(rows)):
        projections[f"pc{index + 1}"] = direction
    central: Dict[str, object] = {}
    for name, direction in projections.items():
        trajectories = [projection_trajectory(row, direction, reference) for row in rows]
        central[name] = _reduction_table(trajectories, labels)
    central["mean_activity"] = _reduction_table(
        [row["central"].sum(axis=1) for row in rows], labels)
    central["linear_cv"] = cross_validated_boundary(_chunk_features(rows, "central"), np.asarray(labels))
    central["lda_cv"] = cross_validated_lda(rows)
    central["nearest_neighbour_cv"] = nearest_neighbour_readout(rows)
    return {"motor": motor, "motor_drive": drive, "central": central}


def _family_values(entry: Dict[str, object], key: str) -> List[float]:
    """Collect one statistic from a family, descending into nested projection families.

    A cross-validated readout is skipped wherever it appears: it is a fit at this sample
    size, not a bound, so it may not set a family's ceiling.
    """
    values: List[float] = []
    for value in entry.values():
        if not isinstance(value, dict) or value.get("fitted"):
            continue
        if value.get(key) is not None:
            values.append(float(value[key]))
        else:
            values.extend(_family_values(value, key))
    return values


def best_bounded(entry: Dict[str, object]) -> Optional[float]:
    """The highest bounded jump recall anywhere in a family."""
    values = _family_values(entry, "jump_recall_at_min_run_recall")
    return max(values) if values else None


def best_auc(entry: Dict[str, object]) -> Optional[float]:
    """The best threshold-free ordering anywhere in a family."""
    values = _family_values(entry, "auc")
    return max(values) if values else None


def summarise_statistics(entry: Dict[str, object]) -> Dict[str, object]:
    """Collapse one family (or one chain of nested families) to its two headline numbers."""
    return {"best_bounded_jump_recall": best_bounded(entry), "best_auc": best_auc(entry)}


def decide(motor_ceiling: Optional[float], holdout_readout: Optional[float],
           in_sample_readout: Optional[float] = None,
           required: float = COMPLETION_RELIABILITY,
           margin: float = READOUT_EVIDENCE_MARGIN) -> Dict[str, object]:
    """Turn the ceilings into the pre-committed verdict.

    Fixed before the measurement so the result cannot be read as confirmation either way.
    ``holdout_readout`` is the population readout scored on chunks it was not fitted on, so
    "the distinction is upstream" is only ever claimed from a held-out number; the in-sample
    figure travels alongside it in the reason because the gap between the two *is* evidence --
    about how much of the readout is fit rather than found.
    """
    if motor_ceiling is None:
        return {"decision": "undetermined", "readout_evidence": "unknown",
                "reason": "the motor family produced no bounded recall"}
    if motor_ceiling >= required:
        return {
            "decision": "decoder",
            "readout_evidence": "not needed",
            "reason": (f"a statistic of the motor evidence catches {motor_ceiling:.3f} of the "
                       f"teacher's jumps at run recall >= {MIN_RUN_RECALL:.2f}, at or above the "
                       f"{required:.2f} completion needs: the evidence carries the decision and "
                       "the shipped rule is discarding it. The ceiling is what a rule reaches "
                       "with the threshold *and the timing* chosen from the labels, so the work "
                       "left is finding a causal rule that reads the same window as well -- "
                       "the bound says it exists, not that it is already built"),
        }
    if holdout_readout is not None and holdout_readout >= required:
        return {
            "decision": "motor_readout",
            "readout_evidence": "certified",
            "reason": (f"no motor statistic clears {required:.2f} (best {motor_ceiling:.3f}), but a "
                       f"one-dimensional readout of the central-complex population reaches "
                       f"{holdout_readout:.3f} on chunks it was not fitted on: the distinction is "
                       "present upstream and the motor readout is what loses it"),
        }
    if holdout_readout is not None and holdout_readout >= motor_ceiling + margin:
        return {
            "decision": "motor_readout_unproven",
            "readout_evidence": "suggested",
            "reason": (f"the population separates the classes better than any motor statistic "
                       f"({holdout_readout:.3f} held out vs {motor_ceiling:.3f} bounded), which puts "
                       f"the loss upstream of the motor layer -- but the held-out readout is short "
                       f"of the {required:.2f} completion needs, so at this sample size the "
                       f"distinction is not certified as separable. The fix is a better readout and "
                       "more labelled chunks, not the connectome"),
        }
    return {
        "decision": "representation",
        "readout_evidence": "absent",
        "reason": (f"neither the motor evidence ({motor_ceiling:.3f}, threshold and timing "
                   f"oracled) nor a held-out readout of the central population ({holdout_readout}) "
                   f"reaches the {required:.2f} completion needs at bounded jump cost, and the "
                   f"fitted population readout ({in_sample_readout}) does not survive holding the "
                   "chunks out: the lever is the features or the weights, not the decision rule"),
    }


# --------------------------------------------------------------------------- assembly


def oracle_ceiling(arms, dataset, stride: int = 15,
                   settle_steps: Sequence[int] = (DEFAULT_SETTLE_STEPS,),
                   seeds: Optional[Sequence[int]] = None,
                   replays: Optional[int] = None,
                   init_seed: int = 42,
                   min_run_recall: float = MIN_RUN_RECALL,
                   dataset_label: Optional[str] = None) -> Dict[str, object]:
    """Measure the ceiling for every arm, at every window length, on one dataset.

    Seeds default to the dev range and the reserved gate set is refused, for the same
    reason the pre-screen refuses it: this instrument exists to decide where to spend
    effort, so it is iteration by definition, and a ceiling read off the seeds the claim
    is reported on would be tuned into the claim.
    """
    arms = list(arms)
    if any(arm.name == UNTRAINED_ARM for arm in arms):
        raise ValueError(f"{UNTRAINED_ARM!r} is reserved for the baseline arm")
    seed_set = resolve_seeds(seeds, replays)
    if seed_set.role == GATE_ROLE:
        raise ValueError(
            f"the oracle ceiling is a tuning instrument and may not be measured on the reserved "
            f"gate seeds {list(GATE_SEEDS)}: it decides where the next attempt goes, so it is "
            f"measured on dev seeds {list(DEV_SEEDS)}."
        )
    windows = [int(window) for window in settle_steps]
    if not windows or any(window < 1 for window in windows):
        raise ValueError("settle steps must be positive")

    shards = load_shards(dataset)
    # The baseline is rebuilt from the same initialization a trained arm started from, so the
    # comparison is against this arm's own starting point rather than a different null.
    table = [Arm(name=UNTRAINED_ARM, model=untrained_model(init_seed),
                 detail={"role": "untrained_baseline"})] + arms

    results = []
    for arm in table:
        per_window = {}
        for window in windows:
            replays_out = []
            for replay_seed in seed_set.seeds:
                preprocessor = vision_preprocessor()
                rows = record_chunks(arm.model, preprocessor, shards, stride, window, replay_seed)
                families = family_report(rows)
                motor = summarise_statistics(families["motor"])
                drive = summarise_statistics(families["motor_drive"])
                central = summarise_statistics(families["central"])
                holdout = families["central"].get("lda_cv") or {}
                neighbours = families["central"].get("nearest_neighbour_cv") or {}
                # The verdict takes the *better* held-out probe, so an upstream verdict has to
                # beat both a linear and a nonlinear readout rather than fall between them.
                upstream = [value for value in (holdout.get("jump_recall"),
                                                neighbours.get("jump_recall"))
                            if value is not None]
                replays_out.append({
                    "replay_seed": replay_seed,
                    "chunks": len(rows),
                    "jump_chunks": int(sum(bool(row["label"]) for row in rows)),
                    "motor": motor,
                    "motor_drive": drive,
                    "central": central,
                    "population_readout_held_out": max(upstream) if upstream else None,
                    "population_readout_held_out_lda": holdout.get("jump_recall"),
                    "population_readout_held_out_1nn": neighbours.get("jump_recall"),
                    "population_readout_held_out_auc": holdout.get("auc"),
                    "population_readout_in_sample": best_bounded(families["central"]),
                    "detail": families,
                })
            per_window[str(window)] = {
                "settle_steps": window,
                "replays": replays_out,
                "motor_best_bounded_jump_recall": _mean([r["motor"]["best_bounded_jump_recall"] for r in replays_out]),
                "motor_best_auc": _mean([r["motor"]["best_auc"] for r in replays_out]),
                "motor_drive_best_bounded_jump_recall": _mean([r["motor_drive"]["best_bounded_jump_recall"] for r in replays_out]),
                "motor_drive_best_auc": _mean([r["motor_drive"]["best_auc"] for r in replays_out]),
                "central_best_bounded_jump_recall": _mean([r["central"]["best_bounded_jump_recall"] for r in replays_out]),
                "central_best_auc": _mean([r["central"]["best_auc"] for r in replays_out]),
                "population_readout_held_out": _mean([r["population_readout_held_out"] for r in replays_out]),
                "population_readout_held_out_lda": _mean([r["population_readout_held_out_lda"] for r in replays_out]),
                "population_readout_held_out_1nn": _mean([r["population_readout_held_out_1nn"] for r in replays_out]),
                "population_readout_held_out_auc": _mean([r["population_readout_held_out_auc"] for r in replays_out]),
                "population_readout_in_sample": _mean([r["population_readout_in_sample"] for r in replays_out]),
            }
            # The verdict takes the *best statistic the motor evidence supports*, spikes or the
            # analog drive they quantise, so adding a family can only raise the bound it judges.
            measured = per_window[str(window)]
            measured["motor_evidence_best_bounded_jump_recall"] = max(
                [value for value in (measured["motor_best_bounded_jump_recall"],
                                     measured["motor_drive_best_bounded_jump_recall"])
                 if value is not None], default=None
            )
            measured["motor_evidence_best_auc"] = max(
                [value for value in (measured["motor_best_auc"], measured["motor_drive_best_auc"])
                 if value is not None], default=None
            )
            measured["verdict"] = decide(
                measured["motor_evidence_best_bounded_jump_recall"],
                measured["population_readout_held_out"],
                measured["population_readout_in_sample"],
                required=COMPLETION_RELIABILITY,
            )
        results.append({"arm": arm.name, "detail": arm.detail, "windows": per_window})

    primary = str(windows[0])
    return {
        "protocol": {
            "metric": "oracle_ceiling",
            "dataset": dataset_label,
            "dataset_provenance": dataset_provenance(dataset) if isinstance(dataset, str) else None,
            "stride": stride,
            "settle_steps": windows,
            "primary_settle_steps": windows[0],
            "seed_role": seed_set.role,
            "replays": len(seed_set.seeds),
            "replay_seeds": list(seed_set.seeds),
            "seeds": seed_set.to_dict(),
            "init_seed": init_seed,
            "completion_reliability": COMPLETION_RELIABILITY,
            "min_run_recall": min_run_recall,
            "motor_rules": list(MOTOR_RULES),
            "drive_rules": list(DRIVE_RULES),
            "drive_decays": list(DRIVE_DECAYS),
            "spike_shape_rules": list(SPIKE_SHAPE_RULES),
            "reductions": list(REDUCTIONS),
            "decision_rule": (
                "best motor statistic (spike or analog drive) >= completion reliability -> "
                "decoder; else central projection >= it -> motor_readout; else representation"
            ),
        },
        "arms": results,
        "verdicts": {row["arm"]: row["windows"][primary]["verdict"] for row in results},
    }


def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return round(float(np.mean(present)), 4)


# --------------------------------------------------------------------------- rendering


def format_table(report: Dict[str, object]) -> str:
    """Render the ceilings as the table a reader compares statistics in."""
    protocol = report["protocol"]
    primary = str(protocol["primary_settle_steps"])
    lines = [
        "oracle ceiling: the best score any rule reaches with the threshold and the timing "
        "chosen from the labels",
        (f"dataset {protocol['dataset']}  cadence {protocol['stride']}  windows "
         f"{protocol['settle_steps']}  replays {protocol['replays']}  "
         f"completion needs {protocol['completion_reliability']} per-jump reliability"),
        f"seeds {list(protocol['replay_seeds'])} -- {protocol['seeds']['note']}",
        f"bounded recall = best jump-chunk recall at run-chunk recall >= {protocol['min_run_recall']} "
        "(so jumping on everything cannot win)",
        "",
    ]
    for row in report["arms"]:
        window = row["windows"][primary]
        lines.append(
            f"{row['arm']}: motor spikes {_fmt(window['motor_best_bounded_jump_recall'])} "
            f"(auc {_fmt(window['motor_best_auc'])})  "
            f"analog drive {_fmt(window.get('motor_drive_best_bounded_jump_recall'))} "
            f"(auc {_fmt(window.get('motor_drive_best_auc'))})  "
            f"population readout held-out {_fmt(window['population_readout_held_out'])} "
            f"(lda {_fmt(window['population_readout_held_out_lda'])}, 1nn "
            f"{_fmt(window['population_readout_held_out_1nn'])}, in-sample "
            f"{_fmt(window['population_readout_in_sample'])})  "
            f"-> {window['verdict']['decision']}")
    lines.append("")
    lines.append(f"{'statistic':<26}{'auc':>8}{'bal acc':>9}{'jump rec':>10}{'run rec':>9}"
                 f"{'bounded':>9}  rule")
    baseline = report["arms"][0]
    detail = baseline["windows"][primary]["replays"][0]["detail"]
    lines.append("motor evidence: the two channels the shipped decoder reads")
    for rule in (list(protocol["motor_rules"]) + list(protocol["spike_shape_rules"])
                 + ["oracle_frame", "linear_oracle", "linear_cv"]):
        lines.append(_row(f"motor.{rule}", detail["motor"].get(rule), fitted=rule == "linear_cv"))
    lines.append("")
    lines.append("analog motor drive: the pre-threshold current the spikes quantise -- the "
                 "non-saturating family, and the timing-aware reductions over it")
    drive_rules = list(protocol["drive_rules"])
    decays = list(protocol["drive_decays"])
    ordered = [name for rule in drive_rules for name in
               ([f"{rule}@{decay}" for decay in decays]
                if rule in
                ("drive_leaky_recency", "drive_leaky_early", "drive_recency_normalized")
                else [rule])]
    for name in ordered + ["oracle_frame"]:
        lines.append(_row(f"drive.{name}", detail["motor_drive"].get(name)))
    lines.append("")
    lines.append("central-complex population, one-dimensional projections -- cell is "
                 f"auc (bounded jump recall at run recall >= {protocol['min_run_recall']})")
    projections = [name for name in detail["central"] if name != "linear_cv"]
    lines.append(f"{'reduction':<20}" + "".join(f"{name:>18}" for name in projections))
    for rule in list(protocol["reductions"]) + ["oracle_frame"]:
        cells = "".join(_cell(detail["central"][name].get(rule)) for name in projections)
        lines.append(f"{rule:<20}{cells}")
    lines.append(_row("central.lda_cv", detail["central"].get("lda_cv"), fitted=True))
    lines.append(_row("central.nearest_neighbour_cv",
                      detail["central"].get("nearest_neighbour_cv"), fitted=True))
    lines.append(_row("central.linear_cv", detail["central"].get("linear_cv"), fitted=True))
    lines.append("")
    lines.append("('lda' is the label-aware Fisher direction -- the best possible one-dimensional "
                 "population readout, so the \"best\" column bounds it; 'pc1..3' are label-free and "
                 "only comparable. linear_cv is cross-validated and is not a ceiling: ~600 "
                 "features against ~99 chunks can separate anything in a fit)")
    lines.append("")
    lines.append("(detail rows are the untrained arm, replay 1; the JSON carries every arm, "
                 "window and replay)")
    for arm, verdict in report["verdicts"].items():
        lines.append(f"verdict {arm}: {verdict['decision']} -- {verdict['reason']}")
    return "\n".join(lines)


def _cell(entry: Optional[Dict[str, object]]) -> str:
    if not entry:
        return f"{'-':>18}"
    return f"{_fmt(entry.get('auc'))} ({_fmt(entry.get('jump_recall_at_min_run_recall'))})".rjust(18)


def _row(name: str, entry: Optional[Dict[str, object]], fitted: bool = False) -> str:
    if not entry:
        return f"{name:<26}{'-':>8}"
    how = "cross-validated" if fitted else "threshold fitted"
    return (f"{name:<26}{_fmt(entry.get('auc')):>8}{_fmt(entry.get('balanced_accuracy')):>9}"
            f"{_fmt(entry.get('jump_recall')):>10}{_fmt(entry.get('run_recall')):>9}"
            f"{_fmt(entry.get('jump_recall_at_min_run_recall')):>9}  {how}")


def _fmt(value) -> str:
    return "-" if value is None else f"{float(value):.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure the oracle ceiling of the decision evidence (decoder or "
                    "representation?)"
    )
    parser.add_argument("--dataset", required=True, help="Trajectory dataset directory")
    parser.add_argument("--checkpoint", action="append", default=[],
                        metavar="NAME=PATH", help="Candidate checkpoint; repeatable")
    parser.add_argument("--stride", type=int, default=15, help="Action cadence")
    parser.add_argument("--settle-steps", default=str(DEFAULT_SETTLE_STEPS),
                        help=f"Comma-separated window lengths to compare (default "
                             f"{DEFAULT_SETTLE_STEPS}); a longer window is finer evidence")
    parser.add_argument("--seeds", default=None,
                        help=f"Comma-separated dev-seed list (default {list(DEV_SEEDS[:DEFAULT_REPLAYS])}); "
                             f"the reserved gate seeds {list(GATE_SEEDS)} are refused")
    parser.add_argument("--replays", type=int, default=None,
                        help=f"How many dev seeds to spend (default {DEFAULT_REPLAYS})")
    parser.add_argument("--init-seed", type=int, default=42,
                        help="Seed for the untrained baseline's weights (not an evaluation seed)")
    parser.add_argument("--min-run-recall", type=float, default=MIN_RUN_RECALL)
    parser.add_argument("--output", default=None, help="Write the full report as JSON here")
    args = parser.parse_args()

    arms = []
    for spec in args.checkpoint:
        if "=" not in spec:
            raise SystemExit(f"--checkpoint expects NAME=PATH, got {spec!r}")
        name, path = spec.split("=", 1)
        arms.append(arm_from_checkpoint(name, path))

    report = oracle_ceiling(
        arms, args.dataset, stride=args.stride,
        settle_steps=_int_list(args.settle_steps),
        seeds=parse_seeds(args.seeds) if args.seeds else None,
        replays=args.replays, init_seed=args.init_seed,
        min_run_recall=args.min_run_recall, dataset_label=os.path.abspath(args.dataset),
    )
    print(format_table(report))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"\nwrote {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
