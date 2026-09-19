import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch

from connectome import DrosophilaConnectomeSNN
from simulation import DEFAULT_SAVE_PATH
from trajectory import iter_dataset
from vision import OmmatidiaVisionPreprocessor

TARGET_RATE = 0.05
CHOSEN_RATE = 0.90
DEFAULT_LR = 0.0005


def target_rate_vector(action: int, width: int = 4) -> torch.Tensor:
    target = torch.full((width,), TARGET_RATE, dtype=torch.float32)
    target[int(action)] = CHOSEN_RATE
    return target


def iter_samples(dataset_dir: str, stride: int) -> Iterable[Tuple[np.ndarray, int]]:
    for shard in iter_dataset(dataset_dir):
        for frame, action in zip(shard["frames"][::stride], shard["actions"][::stride]):
            yield frame, int(action)


def pretrain_motor_layer(dataset_dir: str, epochs: int = 3, lr: float = DEFAULT_LR,
                         stride: int = 4, seed: int = 42) -> Tuple[DrosophilaConnectomeSNN, Dict[str, object]]:
    if epochs <= 0 or lr <= 0 or stride <= 0:
        raise ValueError("epochs, lr, and stride must be positive")
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = DrosophilaConnectomeSNN()
    model.eval()
    preprocessor = OmmatidiaVisionPreprocessor(grid_h=28, grid_w=28)
    samples = list(iter_samples(dataset_dir, stride))
    if not samples:
        raise ValueError("trajectory dataset contains no samples")
    updates = 0
    correct = 0
    with torch.no_grad():
        for _ in range(epochs):
            model.reset_state()
            preprocessor.reset()
            for frame, action in samples:
                features, _ = preprocessor.process_frame(frame)
                spikes = preprocessor.generate_poisson_spikes(features)
                motor_spikes, _ = model(spikes)
                target = target_rate_vector(action)
                error = target - motor_spikes
                eligibility = model.layer3_4.trace_pre.clone()
                model.layer3_4.weight.add_(lr * error.unsqueeze(1) * eligibility.unsqueeze(0))
                model.layer3_4.weight.sub_(model.layer3_4.weight.mean(dim=1, keepdim=True))
                model.layer3_4.weight.clamp_(-3.0, 3.0)
                correct += int(int(motor_spikes.argmax()) == action)
                updates += 1
    metadata = {
        "pretraining": "supervised_motor_eligibility",
        "epochs": epochs,
        "learning_rate": lr,
        "stride": stride,
        "seed": seed,
        "samples": len(samples),
        "updates": updates,
        "motor_argmax_accuracy": correct / updates,
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
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    model, metadata = pretrain_motor_layer(args.dataset, args.epochs, args.lr, args.stride, args.seed)
    save_checkpoint(model, args.output, metadata)
    print(json.dumps({"checkpoint": os.path.abspath(args.output), **metadata}, indent=2))


if __name__ == "__main__":
    main()
