import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch

from connectome import DrosophilaConnectomeSNN
from macro_decoder import (
    JUMP_ACTION,
    MAX_CHUNK_FRAMES,
    RUN_ACTION,
    calibrate_jump_margin,
)
from simulation import DEFAULT_SAVE_PATH, MAX_SETTLE_STEPS, DEFAULT_SETTLE_STEPS
from trajectory import dataset_provenance, iter_dataset
from vision import OmmatidiaVisionPreprocessor

TARGET_RATE = 0.05
CHOSEN_RATE = 0.90
DEFAULT_LR = 0.0005


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


def decision_evidence(model: DrosophilaConnectomeSNN, preprocessor: OmmatidiaVisionPreprocessor,
                      shards, stride: int, settle_steps: int, seed: int):
    """Replay the supervised chunks and record the macro decision evidence.

    For every supervised decision point this returns ``jump_evidence - run_evidence``
    as the *decoder* computes it, together with the labelled action, so the jump
    threshold can be calibrated on data rather than assumed. Runs with no weight
    updates and a fixed seed, so calibration is reproducible.
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


#: Calibration replays pooled into one margin. A single replay is one Poisson draw
#: per decision point, which is noisy enough that the chosen boundary swings wildly
#: across evaluation seeds; averaging a few draws is what makes the margin stable.
DEFAULT_CALIBRATION_REPLAYS = 3


def pretrain_motor_layer(dataset_dir: str, epochs: int = 3, lr: float = DEFAULT_LR,
                         stride: int = 1, settle_steps: int = DEFAULT_SETTLE_STEPS,
                         seed: int = 42,
                         calibration_replays: int = DEFAULT_CALIBRATION_REPLAYS) -> Tuple[DrosophilaConnectomeSNN, Dict[str, object]]:
    """Supervised motor-eligibility pretraining over trajectory shards.

    ``stride`` is the action cadence: one supervised target per cadence frames,
    matching the macro-action chunk length the controller executes. Each shard is a
    separate episode segment, so the recurrent SNN state and eligibility traces are
    explicitly reset at every shard boundary.
    """
    if epochs <= 0 or lr <= 0 or stride <= 0:
        raise ValueError("epochs, lr, and stride must be positive")
    if not (1 <= settle_steps <= MAX_SETTLE_STEPS):
        raise ValueError(f"settle_steps must be between 1 and {MAX_SETTLE_STEPS}")
    if calibration_replays < 1:
        raise ValueError("calibration_replays must be at least 1")

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = DrosophilaConnectomeSNN()
    model.eval()
    preprocessor = OmmatidiaVisionPreprocessor(grid_h=28, grid_w=28)

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

                        target = target_rate_vector(action_int)
                        error = target - motor_spikes
                        eligibility = model.layer3_4.trace_pre.clone()
                        model.layer3_4.weight.add_(lr * error.unsqueeze(1) * eligibility.unsqueeze(0))
                        model.layer3_4.weight.sub_(model.layer3_4.weight.mean(dim=1, keepdim=True))
                        model.layer3_4.weight.clamp_(-3.0, 3.0)

                    correct += int(int(accumulated_motor.argmax()) == action_int)
                    updates += 1

    # Calibrate the macro decoder's jump threshold on the supervised chunks. The
    # readout is offset (run evidence dominates on both classes), so a hard-coded
    # zero margin reads as "never jump" and the controller cannot clear an obstacle.
    # Pool several replays so the boundary reflects expected evidence rather than
    # one noisy draw; replay seeds are derived from ``seed`` and thus reproducible.
    evidence_diffs, jump_labels = [], []
    for replay in range(calibration_replays):
        replay_diffs, replay_labels = decision_evidence(
            model, preprocessor, shards, stride, settle_steps, seed + replay
        )
        evidence_diffs.extend(replay_diffs)
        jump_labels.extend(replay_labels)
    calibration = calibrate_jump_margin(evidence_diffs, jump_labels)
    calibration["replays"] = calibration_replays
    calibration["replay_seeds"] = [seed + replay for replay in range(calibration_replays)]
    decoder_config = {
        "decoder": "bounded_macro_action",
        "chunk_frames": stride,
        "jump_chunk_frames": stride,
        "jump_margin": calibration["margin"],
        "refractory_frames": 0,
        "max_chunk_frames": MAX_CHUNK_FRAMES,
        "calibration": calibration,
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
        "samples": total_samples,
        "updates": updates,
        "episode_boundary_resets": episode_boundary_resets,
        "motor_argmax_accuracy": correct / updates if updates > 0 else 0.0,
        "samples_by_origin": _samples_by_origin(dataset_info),
        "dataset": dataset_info,
        "decoder": decoder_config,
        "jump_margin": calibration["margin"],
    }
    return model, metadata


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
                        "refractory_frames", "max_chunk_frames", "calibration")
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
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    model, metadata = pretrain_motor_layer(
        args.dataset, args.epochs, args.lr, args.stride, args.settle_steps, args.seed
    )
    save_checkpoint(model, args.output, metadata)
    print(json.dumps({"checkpoint": os.path.abspath(args.output), **metadata}, indent=2))


if __name__ == "__main__":
    main()
