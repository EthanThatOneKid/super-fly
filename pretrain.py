import argparse
import json
import os
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from connectome import DrosophilaConnectomeSNN
from evidence import DEFAULT_RULE as DEFAULT_EVIDENCE_RULE
from evidence import evidence_config
from macro_decoder import MAX_CHUNK_FRAMES
from offline_episode import sequence_report
from prescreen import (
    DEFAULT_MAX_JUMP_RATE,
    Arm,
    arm_report,
    calibration_record,
    collect_replays,
    prescreen,
)
from seed_policy import (
    DEFAULT_REPLAYS,
    DEV_SEEDS,
    GATE_SEEDS,
    SeedSet,
    assert_not_reserved,
    parse_seeds,
    resolve_seeds,
)
from simulation import DEFAULT_SAVE_PATH, MAX_SETTLE_STEPS, DEFAULT_SETTLE_STEPS
from trajectory import dataset_provenance, iter_dataset
from vision import OmmatidiaVisionPreprocessor

TARGET_RATE = 0.05
CHOSEN_RATE = 0.90
DEFAULT_LR = 0.0005
#: Bound on every trainable weight, shared with online STDP.
WEIGHT_CLAMP = 3.0

#: Which layers supervised pretraining may move.
#:
#: ``frozen`` fits only the ``layer3_4`` motor readout, leaving the connectome at its
#: initialization heuristic -- a linear probe on fixed random features, which is the
#: ceiling the closed-loop runs kept hitting. ``linear_feedback`` also trains the
#: visual pathway, chaining the motor error back to each layer's *output* units
#: through the transposed next-layer weights (``W_next.T @ e_next``). That is the
#: linear part of backprop with no autograd and no surrogate derivative, it reduces
#: exactly to the rule the readout has always used, and it needs no extra feedback
#: matrices, so nothing new enters the checkpoint.
VISUAL_PATHWAY_MODES = ("frozen", "linear_feedback")
#: Layers the visual-pathway mode trains, beyond the ``layer3_4`` readout.
VISUAL_PATHWAY_LAYERS = ("layer2_3", "layer1_2", "feedback_3_2")
#: Replays per arm, shared with the pre-screen (``prescreen.DEFAULT_REPLAYS``). A single
#: replay is one Poisson draw per decision point, which is noisy enough that both the
#: chosen margin and the reported recall swing across evaluation seeds; repeating the
#: measurement is what makes either stable, and the pre-screen reports every replay so the
#: swing stays visible instead of being averaged away.
DEFAULT_CALIBRATION_REPLAYS = DEFAULT_REPLAYS


def target_rate_vector(action: int, width: int = 4) -> torch.Tensor:
    target = torch.full((width,), TARGET_RATE, dtype=torch.float32)
    target[int(action)] = CHOSEN_RATE
    return target


def _samples_by_origin(provenance: Dict[str, object]) -> Dict[str, int]:
    """Count dataset samples by shard provenance origin (teacher vs DAgger rollouts)."""
    counts: Dict[str, int] = {}
    for shard in provenance["shards"]:
        origin = str(shard.get("provenance", {}).get("origin", "unknown"))
        counts[origin] = counts.get(origin, 0) + int(shard["samples"])
    return counts


def _dataset_env_kind(dataset_dir) -> str | None:
    """The environment a dataset declares, from dataset metadata or its first shard."""
    info = dataset_provenance(dataset_dir)
    kind = info.get("metadata", {}).get("env_kind")
    if kind:
        return str(kind)
    for shard in info["shards"]:
        kind = shard.get("provenance", {}).get("env_kind")
        if kind:
            return str(kind)
    return None


def _apply_delta_rule(layer, error: torch.Tensor, lr: float) -> None:
    """One row-centred, clamped delta step on a LIF layer's weights.

    ``error`` is the correction for the layer's *output* units and ``trace_pre`` is
    that layer's own pre-synaptic eligibility trace, so the update stays local to the
    layer. Rows are re-centred and clamped to the same bound online STDP uses, which
    is what keeps a deep chain of these updates from drifting into a constant offset.
    """
    eligibility = layer.trace_pre.clone()
    layer.weight.add_(lr * error.unsqueeze(1) * eligibility.unsqueeze(0))
    layer.weight.sub_(layer.weight.mean(dim=1, keepdim=True))
    layer.weight.clamp_(-WEIGHT_CLAMP, WEIGHT_CLAMP)


def output_errors(model: DrosophilaConnectomeSNN, motor_error: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Credit for every trainable layer's output units, derived from the motor error.

    The transposed weight matrices already form the chain the forward pass walks --
    ``motor(4) -> central(128) -> optic(256)`` -- so ``e_layer = W_next.T @ e_next``
    needs no random feedback matrices and adds no state to the checkpoint.
    ``layer1_2`` and ``feedback_3_2`` both write into the optic-lobe layer that
    ``layer2_3`` reads, so they share the error measured there.
    """
    central_error = model.layer3_4.weight.t().matmul(motor_error)
    optic_error = model.layer2_3.weight.t().matmul(central_error)
    return {
        "layer3_4": motor_error,
        "layer2_3": central_error,
        "layer1_2": optic_error,
        "feedback_3_2": optic_error,
    }


def _weight_deltas(model: DrosophilaConnectomeSNN,
                   initial: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, float | None]]:
    """How far each trainable layer actually moved (audit, not a metric).

    A null result is only interpretable next to this: a mode that claims to train the
    visual pathway but leaves it within noise of its initialization has not been
    tested, it has been skipped.
    """
    deltas: Dict[str, Dict[str, float | None]] = {}
    for name, before in initial.items():
        after = getattr(model, name).weight.detach()
        delta = float((after - before).norm())
        norm = float(before.norm())
        deltas[name] = {
            "l2_delta": round(delta, 6),
            "l2_initial": round(norm, 6),
            "relative": round(delta / norm, 6) if norm else None,
        }
    return deltas


# The offline chunk-decision measurement used to live here, as ``measure_chunk_decisions``
# plus ``decision_evidence``. It moved to ``prescreen``, which scores jump-chunk recall at
# a bounded jump rate, reports every replay rather than a pooled figure, pairs each arm
# seed by seed against an untrained baseline, and can be run without training anything
# (``python prescreen.py --dataset ... --checkpoint NAME=PATH``).



def pretrain_motor_layer(dataset_dir: str, epochs: int = 3, lr: float = DEFAULT_LR,
                         stride: int = 1, settle_steps: int = DEFAULT_SETTLE_STEPS,
                         seed: int = 42,
                         calibration_replays: int = DEFAULT_CALIBRATION_REPLAYS,
                         visual_pathway: str = "frozen",
                         visual_lr: float | None = None,
                         report_dataset_dir: str | None = None,
                         eval_seeds: Optional[Sequence[int]] = None,
                         evidence_rule: str | None = None,
                         evidence_decay: float | None = None) -> Tuple[DrosophilaConnectomeSNN, Dict[str, object]]:
    """Supervised pretraining over trajectory shards.

    ``stride`` is the action cadence: one supervised target per cadence frames,
    matching the macro-action chunk length the controller executes. Each shard is a
    separate episode segment, so the recurrent SNN state and eligibility traces are
    explicitly reset at every shard boundary.

    ``visual_pathway`` selects what is allowed to move: ``frozen`` fits only the
    ``layer3_4`` motor readout (the long-standing behaviour) and ``linear_feedback``
    also trains ``layer1_2``/``layer2_3``/``feedback_3_2``, so the decision is made on
    learned visual features rather than on fixed random ones. Both modes share one
    learning rule; see ``output_errors``.

    ``report_dataset_dir`` adds a quality report on a dataset the weights never
    trained on, using the margin fitted on the training shards. One episode is one
    shard, so a held-out half has to be materialised first (``trajectory.slice_dataset``).

    ``seed`` is the *initialization and training* seed -- the connectome the arm starts
    from and the Poisson draws the updates see -- which is a different axis from the
    ``eval_seeds`` the arm is then calibrated and pre-screened on. Those default to the dev
    range and may not be the reserved gate set: fitting the jump margin on the seeds the
    published claim is reported on would leak the claim back into the thing being claimed.

    ``evidence_rule`` selects the statistic the calibrated margin thresholds (see
    :mod:`evidence`). It is calibrated *under that rule* and written into the checkpoint
    beside it, because a margin in spike counts means nothing applied to a drive statistic.
    """
    if epochs <= 0 or lr <= 0 or stride <= 0:
        raise ValueError("epochs, lr, and stride must be positive")
    if not (1 <= settle_steps <= MAX_SETTLE_STEPS):
        raise ValueError(f"settle_steps must be between 1 and {MAX_SETTLE_STEPS}")
    if calibration_replays < 1:
        raise ValueError("calibration_replays must be at least 1")
    # An explicit seed list defines the count; ``calibration_replays`` is the count used
    # when the caller only said "a few".
    eval_seed_set: SeedSet = resolve_seeds(
        eval_seeds, None if eval_seeds is not None else calibration_replays
    )
    assert_not_reserved(
        eval_seed_set.seeds, "calibrating the jump margin and pre-screening this arm"
    )
    if visual_pathway not in VISUAL_PATHWAY_MODES:
        raise ValueError(f"visual_pathway must be one of {VISUAL_PATHWAY_MODES}")
    visual_lr_used = lr if visual_lr is None else float(visual_lr)
    if visual_pathway != "frozen" and visual_lr_used <= 0:
        raise ValueError("visual_lr must be positive when the visual pathway is trained")

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = DrosophilaConnectomeSNN()
    model.eval()
    preprocessor = OmmatidiaVisionPreprocessor(grid_h=28, grid_w=28)

    layer_lrs = {"layer3_4": lr}
    if visual_pathway != "frozen":
        layer_lrs.update({name: visual_lr_used for name in VISUAL_PATHWAY_LAYERS})
    # Snapshot every layer the audit reports on, not only the ones this mode trains:
    # a frozen run then *shows* the visual pathway at zero delta instead of leaving
    # the reader to infer it from an absent key.
    audited_layers = ("layer3_4",) + VISUAL_PATHWAY_LAYERS
    initial_weights = {name: getattr(model, name).weight.detach().clone() for name in audited_layers}

    report_env_kind = (None if report_dataset_dir is None
                       else _require_matching_env_kind(dataset_dir, report_dataset_dir))

    shards = list(iter_dataset(dataset_dir))
    if not shards:
        raise ValueError("trajectory dataset contains no samples")

    total_samples = sum(len(s["actions"][::stride]) for s in shards)
    updates = 0
    correct = 0
    episode_boundary_resets = 0

    with torch.no_grad():
        for _ in range(epochs):
            for shard in shards:
                # Explicit episode-boundary reset: membrane potentials, spikes,
                # pre/post eligibility traces, rate traces and the recurrent
                # feedback buffer all start clean for every shard.
                model.reset_state()
                preprocessor.reset()
                episode_boundary_resets += 1
                frames = shard["frames"][::stride]
                actions = shard["actions"][::stride]

                for frame, action in zip(frames, actions):
                    action_int = int(action)
                    features, _ = preprocessor.process_frame(frame)
                    accumulated_motor = torch.zeros(model.num_motor_ganglion)

                    for step_idx in range(settle_steps):
                        spikes = preprocessor.generate_poisson_spikes(features)
                        motor_spikes, _ = model(spikes)
                        accumulated_motor += motor_spikes

                        motor_error = target_rate_vector(action_int) - motor_spikes
                        # Every layer's error is read off the weights *before* any of
                        # them moves, so the update order cannot change the result.
                        step_errors = ({"layer3_4": motor_error} if visual_pathway == "frozen"
                                       else output_errors(model, motor_error))
                        for layer_name, layer_error in step_errors.items():
                            _apply_delta_rule(
                                getattr(model, layer_name), layer_error, layer_lrs[layer_name]
                            )

                    correct += int(int(accumulated_motor.argmax()) == action_int)
                    updates += 1

    # Calibrate the macro decoder's jump threshold on the supervised chunks. The
    # readout is offset (run evidence dominates on both classes), so a hard-coded
    # zero margin reads as "never jump" and the controller cannot clear an obstacle.
    # Pooled replays are what make the boundary reflect expected evidence rather than
    # one noisy draw. The same replays then feed the pre-screen, so the arm is measured
    # on exactly the evidence its margin was chosen from -- which is why both run on the
    # *dev* seeds: it is a fitted number, and the reserved set is report-only.
    rule_config = evidence_config(evidence_rule, evidence_decay)
    replays = collect_replays(model, shards, stride, settle_steps, eval_seed_set, rule_config)
    calibration = calibration_record(replays)
    decoder_config = {
        "decoder": "bounded_macro_action",
        "chunk_frames": stride,
        "jump_chunk_frames": stride,
        "jump_margin": calibration["margin"],
        "refractory_frames": 0,
        "max_chunk_frames": MAX_CHUNK_FRAMES,
        "calibration": calibration,
        # The statistic the margin above belongs to. Recorded beside it so the closed loop
        # cannot apply one rule's threshold to another rule's evidence.
        **rule_config,
    }

    dataset_info = dataset_provenance(dataset_dir)
    metadata = {
        "pretraining": "supervised_motor_eligibility",
        "epochs": epochs,
        "learning_rate": lr,
        "stride": stride,
        "action_cadence": stride,
        "settle_steps": settle_steps,
        "seed": seed,
        "eval_seeds": list(eval_seed_set.seeds),
        "eval_seed_role": eval_seed_set.role,
        "samples": total_samples,
        "updates": updates,
        "episode_boundary_resets": episode_boundary_resets,
        "motor_argmax_accuracy": correct / updates if updates > 0 else 0.0,
        "samples_by_origin": _samples_by_origin(dataset_info),
        "dataset": dataset_info,
        "decoder": decoder_config,
        "jump_margin": calibration["margin"],
        "visual_pathway": visual_pathway,
        "visual_learning_rate": visual_lr_used if visual_pathway != "frozen" else None,
        "trained_layers": sorted(layer_lrs),
        "weight_deltas": _weight_deltas(model, initial_weights),
    }
    # The pre-screen that decides whether this checkpoint is worth an emulator run: a
    # teacher-forced sequential replay (frames in order, the shipped decoder driving them)
    # reporting where the run dies and the best_x that follows, paired seed by seed against
    # an untrained network of the same architecture. Assembled by the same entry point as the
    # standalone ``prescreen.py`` CLI, so the training path and the CLI cannot disagree about
    # what was measured.
    metadata["prescreen"] = prescreen(
        [Arm(
            "candidate",
            model,
            margin=calibration["margin"],
            decoder=decoder_config,
            detail={
                "role": "candidate",
                "visual_pathway": visual_pathway,
                "epochs": epochs,
                "learning_rate": lr,
                "visual_learning_rate": visual_lr_used if visual_pathway != "frozen" else None,
                "trained_layers": sorted(layer_lrs),
                "weight_deltas": metadata["weight_deltas"],
            },
        )],
        shards,
        stride=stride,
        settle_steps=settle_steps,
        seeds=eval_seed_set,
        # The null arm has to be this arm's own initialization, or the paired delta is
        # measured against a different network.
        init_seed=seed,
        dataset_label=dataset_dir,
    )
    if report_dataset_dir is not None:
        metadata["held_out_decision_quality"] = _held_out_decision_quality(
            model, report_dataset_dir, report_env_kind, stride, settle_steps,
            eval_seed_set, calibration["margin"], decoder_config,
        )
    return model, metadata


def _require_matching_env_kind(train_dataset_dir: str, report_dataset_dir: str) -> str | None:
    """Refuse a held-out dataset collected from a different environment.

    A synthetic shard has the same schema as a real-ROM one, so nothing downstream
    would notice the frames are fabricated -- the reported number would simply be
    meaningless. Checked before training, so a mislabelled report costs nothing.
    """
    train_kind = _dataset_env_kind(train_dataset_dir)
    report_kind = _dataset_env_kind(report_dataset_dir)
    if train_kind and report_kind and train_kind != report_kind:
        raise ValueError(
            f"report dataset env_kind {report_kind!r} does not match training dataset {train_kind!r}"
        )
    return report_kind


def _held_out_decision_quality(model: DrosophilaConnectomeSNN, report_dataset_dir: str,
                               env_kind: str | None, stride: int, settle_steps: int,
                               seeds: SeedSet, margin: float,
                               decoder_config: Dict[str, object]) -> Dict[str, object]:
    """Pre-screen report on a dataset the weights never trained on.

    The same two views as the training-side table: the sequence outcome (the gate) and the
    chunk-decision budget view. The margin comes from the training shards and is applied
    unchanged, so ``budget.at_margin`` shows what the checkpoint actually ships; the sequence
    replay uses the decoder the checkpoint ships, chunk lengths and refractory period included.
    """
    shards = list(iter_dataset(report_dataset_dir))
    report = sequence_report(
        "candidate", model, shards, stride, settle_steps, seeds,
        decoder_config=decoder_config, detail={"role": "held_out"},
    )
    report["budget"] = arm_report(
        Arm("candidate", model, margin=margin, decoder=decoder_config), shards,
        stride, settle_steps, seeds, DEFAULT_MAX_JUMP_RATE,
    )
    return {"env_kind": env_kind, "dataset": dataset_provenance(report_dataset_dir), **report}


def save_checkpoint(model: DrosophilaConnectomeSNN, path: str, metadata: Dict[str, object]) -> None:
    """Persist weights plus the policy configuration they were trained under.

    ``Simulation.load_checkpoint`` honours ``policy_config.macro_decoder``, so the
    calibrated jump margin and the cadence travel with the weights. Without this the
    evaluated controller would silently fall back to an uncalibrated decoder and
    throw away every jump decision the readout learned to make.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    decoder_config = metadata.get("decoder") if isinstance(metadata, dict) else None
    policy_config = {
        "source": "pretrain_motor_layer",
        "action_cadence": metadata.get("action_cadence"),
        "settle_steps": metadata.get("settle_steps"),
        "seed": metadata.get("seed"),
    }
    if isinstance(decoder_config, dict):
        policy_config["macro_decoder"] = {
            key: decoder_config[key]
            for key in ("decoder", "chunk_frames", "jump_chunk_frames", "jump_margin",
                        "refractory_frames", "max_chunk_frames", "calibration",
                        "evidence_rule", "evidence_decay")
            if key in decoder_config
        }
    torch.save({
        "model_state_dict": model.state_dict(),
        "pretraining": metadata,
        "policy_config": policy_config,
    }, target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-train the Super Fly motor layer from trajectory shards")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", default=DEFAULT_SAVE_PATH)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--settle-steps", type=int, default=DEFAULT_SETTLE_STEPS)
    parser.add_argument("--seed", type=int, default=42,
                        help="Initialization and training seed (weight init and update draws), "
                             "not an evaluation seed")
    parser.add_argument("--eval-seeds", default=None,
                        help=f"Seeds the margin is calibrated on and the arm pre-screened on; "
                             f"defaults to the dev range {list(DEV_SEEDS[:DEFAULT_REPLAYS])}. The "
                             f"reserved gate seeds {list(GATE_SEEDS)} are refused")
    parser.add_argument("--calibration-replays", type=int, default=DEFAULT_CALIBRATION_REPLAYS,
                        help="How many dev seeds to spend on calibration and the pre-screen")
    parser.add_argument("--visual-pathway", choices=VISUAL_PATHWAY_MODES, default="frozen",
                        help="'frozen' fits only the motor readout; 'linear_feedback' trains the visual pathway too")
    parser.add_argument("--visual-lr", type=float, default=None,
                        help="Learning rate for the visual pathway (defaults to --lr)")
    parser.add_argument("--report-dataset", default=None,
                        help="Held-out dataset to pre-screen at the training margin")
    parser.add_argument("--evidence-rule", default=None,
                        help=f"Statistic the calibrated margin thresholds (default "
                             f"{DEFAULT_EVIDENCE_RULE}); 'spike_sum' is the rule that shipped")
    parser.add_argument("--evidence-decay", type=float, default=None,
                        help="Recency decay for a leaky evidence rule")
    args = parser.parse_args()
    model, metadata = pretrain_motor_layer(
        args.dataset, args.epochs, args.lr, args.stride, args.settle_steps, args.seed,
        calibration_replays=args.calibration_replays,
        visual_pathway=args.visual_pathway, visual_lr=args.visual_lr,
        report_dataset_dir=args.report_dataset,
        eval_seeds=parse_seeds(args.eval_seeds) if args.eval_seeds else None,
        evidence_rule=args.evidence_rule, evidence_decay=args.evidence_decay,
    )
    save_checkpoint(model, args.output, metadata)
    calibration = metadata["decoder"]["calibration"]
    measurement = metadata["prescreen"]
    baseline, candidate = measurement["table"][0], measurement["table"][1]
    paired = measurement["paired"][0]
    verdict = measurement["verdicts"][0]
    print(json.dumps({
        "checkpoint": os.path.abspath(args.output),
        "visual_pathway": metadata["visual_pathway"],
        "motor_argmax_accuracy": round(metadata["motor_argmax_accuracy"], 4),
        "prescreen": {
            "offline_best_x": candidate["offline_best_x"]["mean"],
            "per_replay_best_x": candidate["offline_best_x"]["values"],
            "untrained_best_x": baseline["offline_best_x"]["mean"],
            "required_jumps": candidate["teacher"]["required_jumps"],
            "missed_jumps": candidate["missed_jumps"]["mean"],
            "spurious_jumps": candidate["spurious_jumps"]["mean"],
            "offline_completion_rate": candidate["offline_completion_rate"],
            "paired_mean_delta": paired["mean_delta"],
            "paired_improved": f"{paired['improved']}/{paired['paired_replays']}",
            "pass": verdict["pass"],
            "failed_criteria": [c["criterion"] for c in verdict["criteria"] if not c["pass"]],
        },
        "budget_view": {
            "max_jump_rate": measurement["protocol"]["max_jump_rate"],
            "jump_recall": candidate["budget"]["jump_recall"]["mean"],
            "per_replay_recall": candidate["budget"]["jump_recall"]["values"],
            "random_recall": candidate["budget"]["random_recall"]["mean"],
            "untrained_recall": baseline["budget"]["jump_recall"]["mean"],
        },
        "shipping_margin": calibration["margin"],
        "evidence_rule": metadata["decoder"]["evidence_rule"],
        "shipping_margin_quality": {
            key: candidate["budget"]["at_margin"][key]
            for key in ("jump_recall", "jump_rate", "balanced_accuracy")
        },
        "held_out": metadata.get("held_out_decision_quality", {}).get("offline_best_x"),
        "weight_deltas": metadata["weight_deltas"],
    }, indent=2))


if __name__ == "__main__":
    main()
