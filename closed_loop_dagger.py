"""Issue #30 experiment runner: closed-loop DAgger with a bounded macro-action decoder.

This script is the reproducible driver for the "next materially different
experiment" toward model-only Level 1-1 completion:

* ``--mode baseline`` records a deterministic model-only baseline for the
  current checkpoint across the reserved seeds (42, 43, 44) with the same
  horizon, settle window and action cadence used for candidates.
* ``--mode dagger`` runs bounded DAgger rounds: model-only on-policy rollouts →
  divergence / unrecoverable recovery windows → aggregated teacher + rollout
  dataset → supervised motor-eligibility pretraining at the chunk cadence →
  model-only evaluation on the reserved seeds, stopping early as soon as the
  real completion detector fires without assistance.

Provenance is written next to every result: ROM and checkpoint checksums, git
commit, seeds, horizon, settle steps, action cadence, dataset shard checksums and
per-shard origins. Use ``--offline-env`` to validate the whole loop without the
emulator; offline results are labelled ``offline_synthetic`` and can never count
toward the P0 gate.

Examples::

    python closed_loop_dagger.py --mode baseline --save-path drosophila_snn.pth
    python closed_loop_dagger.py --mode dagger --teacher-dataset data/teacher \\
        --save-path drosophila_snn.pth --iterations 3
    python closed_loop_dagger.py --mode dagger --offline-env --iterations 2
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from dagger import (
    DEFAULT_RECOVERY_LAG_PX,
    DEFAULT_RECOVERY_LAG_STEPS,
    DEFAULT_WINDOW_FRAMES,
    TeacherLabeler,
    aggregate_rollout_metadata,
    build_dagger_dataset,
    collect_on_policy_rollout,
    label_aliasing_report,
)
from eval_harness import evaluate_policy
from offline_env import OfflineMarioEnv
from pretrain import pretrain_motor_layer, save_checkpoint
from simulation import (
    DEFAULT_ACTION_CADENCE,
    DEFAULT_MAX_STEPS,
    DEFAULT_ROM_PATH,
    DEFAULT_SAVE_PATH,
    DEFAULT_SETTLE_STEPS,
    Simulation,
    make_env,
)
from storage import compute_sha256, get_git_commit_sha
from trajectory import dataset_provenance

#: Current measured mainline temporal-decoder baseline recorded in issue #30.
BASELINE_REFERENCE_BEST_X = 594.0
RESERVED_SEEDS = "42,43,44"
DEFAULT_TRAIN_SEEDS = "0,1,2"
OFFLINE_ENV_KIND = "offline_synthetic"


def build_env_factory(offline: bool, rom_path: str):
    """Return (env_factory, env_kind) for the requested environment."""
    if offline:
        return (lambda state: OfflineMarioEnv(state=state)), OFFLINE_ENV_KIND

    def factory(state):
        env, _ = make_env(rom_path, state=state)
        return env

    return factory, "stable-retro"


def evaluate_over_seeds(env_factory, env_kind, checkpoint, seeds, episodes, max_steps, settle_steps,
                        action_cadence, policy, states, rom_path=None):
    """Model-only evaluation of one checkpoint across the requested seeds."""
    per_seed = []
    for seed in seeds:
        report = evaluate_policy(
            env_factory,
            save_path=checkpoint,
            episodes=episodes,
            max_steps=max_steps,
            bootstrap_episodes=0,
            max_bootstrap_step=0,
            seed=seed,
            settle_steps=settle_steps,
            policy=policy,
            states=states,
            action_cadence=action_cadence,
            env_kind=env_kind,
            rom_path=rom_path if env_kind != OFFLINE_ENV_KIND else None,
        )
        per_seed.append(report)

    completion_rates = [r["completion_rate"] for r in per_seed]
    completion_steps = [
        r["completion_step"] for r in per_seed if r["completion_step"] is not None
    ]
    return {
        "policy": policy,
        "seeds": list(seeds),
        "episodes_per_seed": episodes,
        "horizon": max_steps,
        "settle_steps": settle_steps,
        "action_cadence": action_cadence,
        "env_kind": env_kind,
        "model_only": all(r["model_only"] for r in per_seed),
        "bootstrap_assistance_used": any(r["bootstrap_assistance_used"] for r in per_seed),
        "teacher_assistance_used": any(r["teacher_assistance_used"] for r in per_seed),
        "completion_rate": round(float(np.mean(completion_rates)), 4),
        "completion_step": round(float(np.mean(completion_steps)), 2) if completion_steps else None,
        "best_x": max(r["best_x"] for r in per_seed),
        "mean_best_x": round(float(np.mean([r["best_x"] for r in per_seed])), 2),
        "death_rate": round(float(np.mean([r["death_rate"] for r in per_seed])), 4),
        "total_assisted_jump_frames": sum(r["total_assisted_jump_frames"] for r in per_seed),
        "assisted_jump_frames": sum(r["total_assisted_jump_frames"] for r in per_seed),
        "cadence_consistent": all(r["cadence_consistent"] for r in per_seed),
        "per_seed": per_seed,
    }


def baseline_verdict(evaluation, baseline_x: float = BASELINE_REFERENCE_BEST_X):
    """Compare a model-only evaluation against the issue #30 baseline."""
    return {
        "baseline_best_x": baseline_x,
        "baseline_source": "issue #30 measured mainline temporal-decoder baseline",
        "model_best_x": evaluation["best_x"],
        "best_x_above_baseline": bool(evaluation["best_x"] > baseline_x),
        "completion_rate": evaluation["completion_rate"],
        "improved_completion_behavior": bool(evaluation["completion_rate"] > 0.0),
        "model_only": evaluation["model_only"],
        "p0_gate_met": bool(
            evaluation["model_only"]
            and evaluation["completion_rate"] > 0.0
            and evaluation["env_kind"] != OFFLINE_ENV_KIND
        ),
    }


def run_iteration(iteration, checkpoint, teacher_dataset, labeler, env_factory, env_kind, config):
    """Run one DAgger round: rollouts -> dataset -> pretraining -> evaluation."""
    iteration_dir = Path(config["runs_dir"]) / f"iter-{iteration:02d}"
    iteration_dir.mkdir(parents=True, exist_ok=True)

    rollouts = []
    for seed in config["train_seeds"]:
        sim = Simulation(
            rom_path=config["rom"],
            save_path=checkpoint,
            bootstrap_episodes=0,
            max_bootstrap_step=0,
            policy=config["policy"],
            states=config["states"],
            settle_steps=config["settle_steps"],
            action_cadence=config["action_cadence"],
            seed=seed,
        )
        env = env_factory(config["states"][0])
        try:
            rollouts.append(
                collect_on_policy_rollout(
                    sim,
                    env,
                    labeler,
                    max_steps=config["rollout_steps"],
                    recovery_lag_steps=config["recovery_lag_steps"],
                    recovery_lag_px=config["recovery_lag_px"],
                    window_frames=config["window_frames"],
                    seed=seed,
                )
            )
        finally:
            env.close()

    dataset_dir = iteration_dir / "dataset"
    dataset_info = build_dagger_dataset(
        dataset_dir,
        teacher_dataset,
        rollouts,
        metadata={
            "iteration": iteration,
            "candidate_checkpoint_sha256": compute_sha256(checkpoint) if os.path.exists(checkpoint) else None,
            "env_kind": env_kind,
            "seed": config["pretrain_seed"],
            "horizon": config["rollout_steps"],
            "action_cadence": config["action_cadence"],
            "settle_steps": config["settle_steps"],
            "recovery_lag_steps": config["recovery_lag_steps"],
            "recovery_lag_px": config["recovery_lag_px"],
            "window_frames": config["window_frames"],
            "git_commit_sha": get_git_commit_sha(),
            "agents": "teacher_trajectory + on_policy_dagger_recovery_windows",
        },
    )

    model, pretrain_metadata = pretrain_motor_layer(
        str(dataset_dir),
        epochs=config["epochs"],
        lr=config["lr"],
        stride=config["action_cadence"],
        settle_steps=config["settle_steps"],
        seed=config["pretrain_seed"],
    )
    checkpoint_out = iteration_dir / "checkpoint.pth"
    save_checkpoint(model, str(checkpoint_out), pretrain_metadata)

    evaluation = evaluate_over_seeds(
        env_factory,
        env_kind,
        str(checkpoint_out),
        config["eval_seeds"],
        config["episodes"],
        config["max_steps"],
        config["settle_steps"],
        config["action_cadence"],
        config["policy"],
        config["states"],
        rom_path=config["rom"],
    )

    return {
        "iteration": iteration,
        "checkpoint": str(checkpoint_out),
        "checkpoint_sha256": compute_sha256(str(checkpoint_out)),
        "rollouts": aggregate_rollout_metadata(rollouts),
        "dataset": dataset_info,
        "pretraining": {
            key: pretrain_metadata[key]
            for key in (
                "epochs",
                "learning_rate",
                "stride",
                "action_cadence",
                "settle_steps",
                "seed",
                "samples",
                "updates",
                "episode_boundary_resets",
                "motor_argmax_accuracy",
                "samples_by_origin",
                "decoder",
                "jump_margin",
            )
        },
        "dataset_metadata": dataset_info.get("metadata", {}),
        "evaluation": evaluation,
        "verdict": baseline_verdict(evaluation, config["baseline_x"]),
    }


def run_dagger(config, teacher_dataset, labeler, env_factory, env_kind):
    """Run bounded DAgger rounds from the current checkpoint, stopping at the gate."""
    checkpoint = config["save_path"]
    history = []
    stop_reason = "iteration_budget_exhausted"
    for iteration in range(1, config["iterations"] + 1):
        entry = run_iteration(iteration, checkpoint, teacher_dataset, labeler, env_factory, env_kind, config)
        history.append(entry)
        checkpoint = entry["checkpoint"]
        evaluation = entry["evaluation"]
        if evaluation["model_only"] and evaluation["completion_rate"] > 0.0:
            stop_reason = "model_only_completion_detected"
            break
    return history, stop_reason


def check_teacher_env_kind(teacher_info, env_kind):
    """Refuse to train on a teacher shard collected from a different environment.

    A synthetic teacher shard left over from an offline plumbing run is a valid
    dataset with the same schema as a real-ROM one, so nothing downstream notices
    the frames are fabricated -- the model simply learns nothing and the run dies at
    the first obstacle. Fail loudly instead of reporting a mislabelled number.
    """
    kinds = {
        shard.get("provenance", {}).get("env_kind")
        for shard in teacher_info.get("shards", [])
    }
    kinds.discard(None)
    if kinds and env_kind not in kinds:
        raise SystemExit(
            f"teacher dataset was collected on {sorted(kinds)} but this run uses "
            f"'{env_kind}'; recollect the teacher shard with teacher.py against the "
            "same environment (mixed provenance would make the result meaningless)"
        )


def parse_seeds(value):
    return [int(token.strip()) for token in value.split(",") if token.strip()]


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Closed-loop DAgger + macro-action experiment runner (issue #30)")
    parser.add_argument("--mode", choices=("baseline", "dagger"), default="dagger")
    parser.add_argument("--rom", default=DEFAULT_ROM_PATH, help="Path to the Super Mario Bros NES ROM")
    parser.add_argument("--save-path", default=DEFAULT_SAVE_PATH, help="Checkpoint to evaluate / start DAgger from")
    parser.add_argument("--teacher-dataset", default="data/teacher", help="Directory holding the teacher trajectory shard")
    parser.add_argument("--offline-env", action="store_true", help="Validate plumbing on the offline synthetic env (never gate-eligible)")
    parser.add_argument("--synthesize-teacher", action="store_true", help="Collect a teacher shard when --teacher-dataset is missing (used with --offline-env)")
    parser.add_argument("--policy", choices=("macro", "agent"), default="macro", help="Model-only controller under test")
    parser.add_argument("--iterations", type=int, default=3, help="DAgger rounds to run")
    parser.add_argument("--episodes", type=int, default=3, help="Evaluation episodes per reserved seed")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS, help="Evaluation horizon per episode")
    parser.add_argument("--rollout-steps", type=int, default=1000, help="On-policy DAgger rollout horizon")
    parser.add_argument("--train-seeds", default=DEFAULT_TRAIN_SEEDS, help="Seeds for on-policy DAgger rollouts")
    parser.add_argument("--eval-seeds", default=RESERVED_SEEDS, help="Reserved seeds for model-only evaluation")
    parser.add_argument("--settle-steps", type=int, default=DEFAULT_SETTLE_STEPS, help="Bounded settle window per frame")
    parser.add_argument("--action-cadence", type=int, default=DEFAULT_ACTION_CADENCE, help="Frames per macro-action chunk")
    parser.add_argument("--label-matching", choices=("phase", "position_only"), default="phase",
                        help="Recovery-target matching: on the candidate's phase (default) or progress only")
    parser.add_argument("--recovery-lag-steps", type=int, default=DEFAULT_RECOVERY_LAG_STEPS, help="Teacher-schedule steps behind which a rollout counts as diverged")
    parser.add_argument("--recovery-lag-px", type=int, default=DEFAULT_RECOVERY_LAG_PX, help="Progress deficit in pixels that marks a divergence")
    parser.add_argument("--window-frames", type=int, default=DEFAULT_WINDOW_FRAMES, help="Frames in each recovery window")
    parser.add_argument("--epochs", type=int, default=3, help="Supervised pretraining epochs per round")
    parser.add_argument("--lr", type=float, default=0.0005, help="Supervised pretraining learning rate")
    parser.add_argument("--pretrain-seed", type=int, default=42, help="Seed for pretraining and dataset metadata")
    parser.add_argument("--baseline-x", type=float, default=BASELINE_REFERENCE_BEST_X, help="Baseline best_x to beat")
    parser.add_argument("--states", default="Level1-1", help="Comma-separated level states")
    parser.add_argument("--runs-dir", default=os.path.join("runs", "closed_loop_dagger"), help="Output directory for reports and checkpoints")
    return parser


def _write_markdown_report(report, path):
    audit = report.get("teacher_labelling") or {}
    lines = [
        "# Closed-loop DAgger experiment report (issue #30)",
        "",
        f"- mode: `{report['mode']}`",
        f"- environment: `{report['environment']['env_kind']}`",
        f"- recovery-target matching: `{report['config'].get('label_matching', 'phase')}`",
        (
            f"- teacher labelling audit: {audit.get('mislabelled_positions')} of "
            f"{audit.get('positions_swept')} ground states labelled differently by "
            f"progress only ({audit.get('action_transitions')})"
            if audit.get("has_phase") else
            "- teacher labelling audit: no phase contrast in this shard"
        ),
        f"- model-only policy: `{report['config']['policy']}`",
        f"- reserved seeds: {report['config']['eval_seeds']}",
        f"- settle steps: {report['config']['settle_steps']}, action cadence: {report['config']['action_cadence']}",
        f"- gate eligible environment: {report['verdict']['gate_eligible_env']}",
        f"- P0 gate met: {report['verdict']['p0_gate_met']}",
        f"- stop reason: `{report['verdict']['stop_reason']}`",
        "",
        f"Baseline reference best_x: {report['verdict']['baseline_best_x']}",
        "",
        "| round | best_x | completion_rate | death_rate | model_only | windows | samples |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for entry in report.get("iterations", []):
        evaluation = entry["evaluation"]
        lines.append(
            "| {iteration} | {best_x} | {completion_rate} | {death_rate} | {model_only} | {windows} | {samples} |".format(
                iteration=entry["iteration"],
                best_x=evaluation["best_x"],
                completion_rate=evaluation["completion_rate"],
                death_rate=evaluation["death_rate"],
                model_only=evaluation["model_only"],
                windows=entry["rollouts"]["recovery_windows"],
                samples=entry["pretraining"]["samples"],
            )
        )
    lines.append("")
    lines.append(f"Artifacts: `{report['report_dir']}`")
    lines.append("")
    Path(path).write_text("\n".join(lines))


def build_report(args):
    """Execute the requested mode and assemble the reproducible report."""
    config = {
        "policy": args.policy,
        "iterations": args.iterations,
        "settle_steps": args.settle_steps,
        "action_cadence": args.action_cadence,
        "states": [s.strip() for s in args.states.split(",") if s.strip()] or ["Level1-1"],
        "train_seeds": parse_seeds(args.train_seeds),
        "eval_seeds": parse_seeds(args.eval_seeds),
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "rollout_steps": args.rollout_steps,
        "label_matching": args.label_matching,
        "recovery_lag_steps": args.recovery_lag_steps,
        "recovery_lag_px": args.recovery_lag_px,
        "window_frames": args.window_frames,
        "epochs": args.epochs,
        "lr": args.lr,
        "pretrain_seed": args.pretrain_seed,
        "baseline_x": args.baseline_x,
        "runs_dir": args.runs_dir,
        "save_path": args.save_path,
        "rom": args.rom,
    }

    env_factory, env_kind = build_env_factory(args.offline_env, args.rom)
    report_dir = Path(args.runs_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    teacher_dataset = args.teacher_dataset
    teacher_info = None
    if args.mode == "dagger":
        if not os.path.exists(os.path.join(teacher_dataset, "manifest.json")):
            if args.synthesize_teacher:
                from teacher import write_teacher_shard

                write_teacher_shard(teacher_dataset, args.rom, env_factory=env_factory, state=config["states"][0])
            else:
                raise SystemExit(
                    f"teacher dataset '{teacher_dataset}' not found; run teacher.py first "
                    "or pass --synthesize-teacher with --offline-env"
                )
        teacher_info = dataset_provenance(teacher_dataset)
        check_teacher_env_kind(teacher_info, env_kind)

    baseline_checkpoint = config["save_path"]
    baseline_evaluation = evaluate_over_seeds(
        env_factory,
        env_kind,
        baseline_checkpoint,
        config["eval_seeds"],
        config["episodes"],
        config["max_steps"],
        config["settle_steps"],
        config["action_cadence"],
        config["policy"],
        config["states"],
        rom_path=config["rom"],
    )

    iterations = []
    stop_reason = "baseline_only"
    label_audit = None
    if args.mode == "dagger":
        labeler = TeacherLabeler.from_dataset(teacher_dataset)
        # The audit is the evidence for whatever ``--label-matching`` chose, computed
        # from the teacher shard alone rather than from the run it is explaining.
        label_audit = label_aliasing_report(teacher_dataset)
        if config["label_matching"] == "position_only":
            labeler = labeler.without_phase()
        iterations, stop_reason = run_dagger(config, teacher_dataset, labeler, env_factory, env_kind)

    final_evaluation = iterations[-1]["evaluation"] if iterations else baseline_evaluation
    final_verdict = baseline_verdict(final_evaluation, config["baseline_x"])
    gate_eligible_env = env_kind != OFFLINE_ENV_KIND

    report = {
        "issue": "https://github.com/EthanThatOneKid/super-fly/issues/30",
        "experiment": "closed_loop_dagger_macro_action",
        "mode": args.mode,
        "report_dir": str(report_dir.resolve()),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {
            "env_kind": env_kind,
            "rom": os.path.abspath(args.rom) if env_kind != OFFLINE_ENV_KIND else None,
            "rom_sha256": compute_sha256(args.rom) if env_kind != OFFLINE_ENV_KIND and os.path.exists(args.rom) else None,
            "git_commit_sha": get_git_commit_sha(),
            "results_are_pipeline_validation_only": env_kind == OFFLINE_ENV_KIND,
        },
        "config": config,
        "teacher_dataset": teacher_info,
        "teacher_labelling": label_audit,
        "baseline": {
            "checkpoint": baseline_checkpoint,
            "checkpoint_sha256": compute_sha256(baseline_checkpoint) if os.path.exists(baseline_checkpoint) else None,
            "evaluation": baseline_evaluation,
            "verdict": baseline_verdict(baseline_evaluation, config["baseline_x"]),
        },
        "iterations": iterations,
        "verdict": {
            **final_verdict,
            "stop_reason": stop_reason,
            "gate_eligible_env": gate_eligible_env,
            "p0_gate_met": bool(final_verdict["p0_gate_met"] and gate_eligible_env),
            "assistance_flags": {
                "bootstrap_assistance_used": final_evaluation["bootstrap_assistance_used"],
                "teacher_assistance_used": final_evaluation["teacher_assistance_used"],
            },
            "notes": (
                "Offline synthetic environment: plumbing validation only, never a research result."
                if env_kind == OFFLINE_ENV_KIND
                else "Model-only evaluation on the real ROM; the P0 gate requires the completion detector without assistance."
            ),
        },
    }

    report_path = report_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    _write_markdown_report(report, report_dir / "report.md")
    return report, report_path


def main():
    args = build_arg_parser().parse_args()
    report, report_path = build_report(args)
    print(json.dumps({
        "report": str(report_path),
        "mode": report["mode"],
        "env_kind": report["environment"]["env_kind"],
        "iterations": len(report["iterations"]),
        "stop_reason": report["verdict"]["stop_reason"],
        "baseline_best_x": report["baseline"]["evaluation"]["best_x"],
        "final_best_x": report["verdict"]["model_best_x"],
        "final_completion_rate": report["verdict"]["completion_rate"],
        "model_only": report["verdict"]["model_only"],
        "p0_gate_met": report["verdict"]["p0_gate_met"],
    }, indent=2))


if __name__ == "__main__":
    main()
