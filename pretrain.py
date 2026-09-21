import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch

from connectome import DrosophilaConnectomeSNN
from simulation import DEFAULT_SAVE_PATH, MAX_SETTLE_STEPS, DEFAULT_SETTLE_STEPS
from trajectory import iter_dataset
from vision import OmmatidiaVisionPreprocessor

TARGET_RATE = 0.05
CHOSEN_RATE = 0.90
DEFAULT_LR = 0.0005


def target_rate_vector(action: int, width: int = 4) -> torch.Tensor:
    target = torch.full((width,), TARGET_RATE, dtype=torch.float32)
    target[int(action)] = CHOSEN_RATE
    return target


def pretrain_motor_layer(dataset_dir: str, epochs: int = 3, lr: float = DEFAULT_LR,
                         stride: int = 1, settle_steps: int = DEFAULT_SETTLE_STEPS,
                         seed: int = 42) -> Tuple[DrosophilaConnectomeSNN, Dict[str, object]]:
    if epochs <= 0 or lr <= 0 or stride <= 0:
        raise ValueError("epochs, lr, and stride must be positive")
    if not (1 <= settle_steps <= MAX_SETTLE_STEPS):
        raise ValueError(f"settle_steps must be between 1 and {MAX_SETTLE_STEPS}")

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

    with torch.no_grad():
        for _ in range(epochs):
            for shard in shards:
                model.reset_state()
                preprocessor.reset()
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

                        # Update Layer 4 -> Layer 3 Efference Copy weights
                        if action_int in (1, 3):
                            eff_target = torch.zeros(model.num_central_complex)
                            eff_target[:model.num_central_complex // 2] = 0.5
                            eff_error = eff_target - model.recurrent_central_spikes
                            eff_elig = model.feedback_4_3.trace_pre.clone()
                            model.feedback_4_3.weight.add_(lr * eff_error.unsqueeze(1) * eff_elig.unsqueeze(0))
                            model.feedback_4_3.weight.sub_(model.feedback_4_3.weight.mean(dim=1, keepdim=True))
                            model.feedback_4_3.weight.clamp_(-3.0, 3.0)

                    correct += int(int(accumulated_motor.argmax()) == action_int)
                    updates += 1

    metadata = {
        "pretraining": "supervised_motor_eligibility",
        "epochs": epochs,
        "learning_rate": lr,
        "stride": stride,
        "settle_steps": settle_steps,
        "seed": seed,
        "samples": total_samples,
        "updates": updates,
        "motor_argmax_accuracy": correct / updates if updates > 0 else 0.0,
    }
    return model, metadata


def save_checkpoint(model: DrosophilaConnectomeSNN, path: str, metadata: Dict[str, object]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "pretraining": metadata}, target)


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
