import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from connectome import DrosophilaConnectomeSNN, surrogate_spike
from simulation import DEFAULT_SAVE_PATH, MAX_SETTLE_STEPS, DEFAULT_SETTLE_STEPS
from trajectory import iter_dataset
from vision import OmmatidiaVisionPreprocessor

TARGET_RATE = 0.05
CHOSEN_RATE = 0.90
DEFAULT_LR = 0.001


def target_rate_vector(action: int, width: int = 4) -> torch.Tensor:
    target = torch.full((width,), TARGET_RATE, dtype=torch.float32)
    target[int(action)] = CHOSEN_RATE
    return target


def _step_lif_functional(layer, input_spikes, state_v, state_spikes):
    current = torch.matmul(layer.weight, input_spikes)
    current = current - current.mean()
    v_next = layer.alpha * state_v * (1.0 - state_spikes) + current * layer.current_gain
    spikes_next = surrogate_spike(v_next, layer.v_thresh)
    v_after_reset = torch.where(spikes_next > 0, torch.tensor(layer.v_reset, device=v_next.device), v_next)
    return v_after_reset, spikes_next


def pretrain_motor_layer(dataset_dir: str, epochs: int = 5, lr: float = DEFAULT_LR,
                         stride: int = 1, settle_steps: int = DEFAULT_SETTLE_STEPS,
                         seed: int = 42) -> Tuple[DrosophilaConnectomeSNN, Dict[str, object]]:
    """
    Sequence-aware end-to-end BPTT pretraining of Drosophila connectome SNN
    and TemporalMotorDecoder on checksummed trajectory shards.
    """
    if epochs <= 0 or lr <= 0 or stride <= 0:
        raise ValueError("epochs, lr, and stride must be positive")
    if not (1 <= settle_steps <= MAX_SETTLE_STEPS):
        raise ValueError(f"settle_steps must be between 1 and {MAX_SETTLE_STEPS}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = DrosophilaConnectomeSNN()
    preprocessor = OmmatidiaVisionPreprocessor(grid_h=28, grid_w=28)

    shards = list(iter_dataset(dataset_dir))
    if not shards:
        raise ValueError("trajectory dataset contains no samples")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Class weights for actions: [NOOP: 1.0, RIGHT: 1.0, JUMP: 1.0, RIGHT+JUMP: 4.0]
    action_weights = torch.tensor([1.0, 1.0, 1.0, 4.0])
    criterion = nn.CrossEntropyLoss(weight=action_weights)

    total_samples = sum(len(s["actions"][::stride]) for s in shards)
    updates = 0
    correct = 0

    chunk_size = 15  # BPTT sequence chunk size

    for ep in range(epochs):
        for shard in shards:
            model.reset_state()
            preprocessor.reset()

            frames = shard["frames"][::stride]
            actions = shard["actions"][::stride]

            # Pre-extract frame visual features for fast training
            features_list = []
            for f in frames:
                feat, _ = preprocessor.process_frame(f)
                features_list.append(feat)

            # Functional dynamic state buffers for BPTT
            v_1_2 = torch.zeros(model.num_optic_lobe)
            spikes_1_2 = torch.zeros(model.num_optic_lobe)
            v_2_3 = torch.zeros(model.num_central_complex)
            spikes_2_3 = torch.zeros(model.num_central_complex)
            v_3_4 = torch.zeros(model.num_motor_ganglion)
            spikes_3_4 = torch.zeros(model.num_motor_ganglion)
            v_fb = torch.zeros(model.num_optic_lobe)
            spikes_fb = torch.zeros(model.num_optic_lobe)
            recurrent_cc = torch.zeros(model.num_central_complex)

            chunk_loss = torch.tensor(0.0)
            chunk_count = 0

            for i in range(len(features_list)):
                feat = features_list[i]
                act = int(actions[i])

                accumulated_motor = torch.zeros(model.num_motor_ganglion)
                last_cc = torch.zeros(model.num_central_complex)

                for _ in range(settle_steps):
                    sensory_spikes = preprocessor.generate_poisson_spikes(feat)

                    v_fb, spikes_fb = _step_lif_functional(model.feedback_3_2, recurrent_cc, v_fb, spikes_fb)
                    v_1_2, sensory_optic = _step_lif_functional(model.layer1_2, sensory_spikes, v_1_2, spikes_1_2)
                    optic_spikes = torch.clamp(sensory_optic + spikes_fb, 0.0, 1.0)
                    spikes_1_2 = optic_spikes

                    v_2_3, central_spikes = _step_lif_functional(model.layer2_3, optic_spikes, v_2_3, spikes_2_3)
                    spikes_2_3 = central_spikes
                    recurrent_cc = central_spikes
                    last_cc = central_spikes

                    v_3_4, motor_spikes = _step_lif_functional(model.layer3_4, central_spikes, v_3_4, spikes_3_4)
                    spikes_3_4 = motor_spikes

                    accumulated_motor = accumulated_motor + motor_spikes

                logits = model.decode_action(last_cc, accumulated_motor).unsqueeze(0)
                target = torch.tensor([act], dtype=torch.long)
                loss = criterion(logits, target)

                chunk_loss = chunk_loss + loss
                chunk_count += 1

                pred = logits.argmax(dim=-1).item()
                correct += int(pred == act)
                updates += 1

                if chunk_count >= chunk_size or i == len(features_list) - 1:
                    optimizer.zero_grad()
                    (chunk_loss / chunk_count).backward()
                    optimizer.step()

                    # Enforce zero-mean row centering & [-3.0, 3.0] clamping
                    model.enforce_bio_constraints()

                    # Detach dynamic state tensors for truncated BPTT
                    v_1_2 = v_1_2.detach()
                    spikes_1_2 = spikes_1_2.detach()
                    v_2_3 = v_2_3.detach()
                    spikes_2_3 = spikes_2_3.detach()
                    v_3_4 = v_3_4.detach()
                    spikes_3_4 = spikes_3_4.detach()
                    v_fb = v_fb.detach()
                    spikes_fb = spikes_fb.detach()
                    recurrent_cc = recurrent_cc.detach()

                    chunk_loss = torch.tensor(0.0)
                    chunk_count = 0

    metadata = {
        "pretraining": "sequence_aware_bptt",
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
    parser = argparse.ArgumentParser(description="Pre-train the Super Fly SNN & TemporalMotorDecoder from trajectory shards")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", default=DEFAULT_SAVE_PATH)
    parser.add_argument("--epochs", type=int, default=5)
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
