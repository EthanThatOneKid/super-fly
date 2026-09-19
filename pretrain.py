import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from connectome import DrosophilaConnectomeSNN
from simulation import DEFAULT_SAVE_PATH, MAX_SETTLE_STEPS, DEFAULT_SETTLE_STEPS
from trajectory import iter_dataset
from vision import OmmatidiaVisionPreprocessor

DEFAULT_LR = 0.001


class FastSigmoid(torch.autograd.Function):
    """Fast sigmoid surrogate gradient for spiking LIF neurons."""
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input >= 0.0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        gamma = 2.0
        return grad_output / (1.0 + gamma * torch.abs(input)) ** 2


fast_sigmoid = FastSigmoid.apply


def pretrain_motor_layer(dataset_dir: str, epochs: int = 5, lr: float = DEFAULT_LR,
                         stride: int = 1, settle_steps: int = DEFAULT_SETTLE_STEPS,
                         seed: int = 42, chunk_size: int = 50) -> Tuple[DrosophilaConnectomeSNN, Dict[str, object]]:
    """
    Sequence-aware pretraining using Backpropagation Through Time (BPTT) with surrogate gradients.
    Transfers temporal dynamics from teacher trajectory shards across all connectome layers:
    layer1_2, layer2_3, layer3_4, and feedback_3_2.
    """
    if epochs <= 0 or lr <= 0 or stride <= 0:
        raise ValueError("epochs, lr, and stride must be positive")
    if not (1 <= settle_steps <= MAX_SETTLE_STEPS):
        raise ValueError(f"settle_steps must be between 1 and {MAX_SETTLE_STEPS}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = DrosophilaConnectomeSNN()
    model.train()
    preprocessor = OmmatidiaVisionPreprocessor(grid_h=28, grid_w=28)

    shards = list(iter_dataset(dataset_dir))
    if not shards:
        raise ValueError("trajectory dataset contains no samples")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    total_samples = 0
    total_correct = 0
    total_updates = 0

    for _ in range(epochs):
        for shard in shards:
            model.reset_state()
            preprocessor.reset()

            frames = shard["frames"][::stride]
            actions = shard["actions"][::stride]
            T = len(frames)

            v1_2 = torch.zeros(model.layer1_2.out_features)
            v2_3 = torch.zeros(model.layer2_3.out_features)
            v3_4 = torch.zeros(model.layer3_4.out_features)
            v_fb = torch.zeros(model.feedback_3_2.out_features)
            rec_cc = torch.zeros(model.num_central_complex)

            for t_start in range(0, T, chunk_size):
                t_end = min(t_start + chunk_size, T)

                optimizer.zero_grad()
                chunk_logits = []
                chunk_targets = []

                for t in range(t_start, t_end):
                    frame = frames[t]
                    action = int(actions[t])
                    features, _ = preprocessor.process_frame(frame)

                    acc_motor_spikes = 0.0
                    for _ in range(settle_steps):
                        sensory_spikes = preprocessor.generate_poisson_spikes(features)

                        # Recurrent feedback step (3 -> 2)
                        cur_fb = torch.matmul(model.feedback_3_2.weight, rec_cc)
                        cur_fb = cur_fb - cur_fb.mean()
                        v_fb = model.feedback_3_2.alpha * v_fb + cur_fb * model.feedback_3_2.current_gain
                        fb_spikes = fast_sigmoid(v_fb - model.feedback_3_2.v_thresh)
                        v_fb = v_fb * (1.0 - fb_spikes)

                        # Layer 1 -> 2 (Sensory to Optic Lobe)
                        cur1_2 = torch.matmul(model.layer1_2.weight, sensory_spikes)
                        cur1_2 = cur1_2 - cur1_2.mean()
                        v1_2 = model.layer1_2.alpha * v1_2 + cur1_2 * model.layer1_2.current_gain
                        s1_2 = fast_sigmoid(v1_2 - model.layer1_2.v_thresh)
                        v1_2 = v1_2 * (1.0 - s1_2)

                        optic_spikes = torch.clamp(s1_2 + fb_spikes, 0.0, 1.0)

                        # Layer 2 -> 3 (Optic Lobe to Central Complex)
                        cur2_3 = torch.matmul(model.layer2_3.weight, optic_spikes)
                        cur2_3 = cur2_3 - cur2_3.mean()
                        v2_3 = model.layer2_3.alpha * v2_3 + cur2_3 * model.layer2_3.current_gain
                        s2_3 = fast_sigmoid(v2_3 - model.layer2_3.v_thresh)
                        v2_3 = v2_3 * (1.0 - s2_3)

                        # Layer 3 -> 4 (Central Complex to Motor Ganglion)
                        cur3_4 = torch.matmul(model.layer3_4.weight, s2_3)
                        cur3_4 = cur3_4 - cur3_4.mean()
                        v3_4 = model.layer3_4.alpha * v3_4 + cur3_4 * model.layer3_4.current_gain
                        s3_4 = fast_sigmoid(v3_4 - model.layer3_4.v_thresh)
                        v3_4 = v3_4 * (1.0 - s3_4)

                        rec_cc = s2_3
                        acc_motor_spikes = acc_motor_spikes + s3_4

                    chunk_logits.append(acc_motor_spikes)
                    chunk_targets.append(action)

                logits_tensor = torch.stack(chunk_logits)
                targets_tensor = torch.tensor(chunk_targets, dtype=torch.long)

                loss = nn.functional.cross_entropy(logits_tensor, targets_tensor)
                loss.backward()

                optimizer.step()

                # Maintain row zero-mean and weight clamping constraints
                with torch.no_grad():
                    for layer in [model.layer1_2, model.layer2_3, model.layer3_4, model.feedback_3_2]:
                        layer.weight.sub_(layer.weight.mean(dim=1, keepdim=True))
                        layer.weight.clamp_(-3.0, 3.0)

                # Detach recurrent and membrane potential states for truncated BPTT across chunk boundaries
                v1_2 = v1_2.detach()
                v2_3 = v2_3.detach()
                v3_4 = v3_4.detach()
                v_fb = v_fb.detach()
                rec_cc = rec_cc.detach()

                preds = logits_tensor.argmax(dim=1)
                total_correct += (preds == targets_tensor).sum().item()
                total_samples += len(targets_tensor)
                total_updates += 1

    model.eval()

    metadata = {
        "pretraining": "sequence_bptt_surrogate",
        "epochs": epochs,
        "learning_rate": lr,
        "stride": stride,
        "settle_steps": settle_steps,
        "seed": seed,
        "chunk_size": chunk_size,
        "samples": total_samples,
        "updates": total_updates,
        "motor_argmax_accuracy": total_correct / total_samples if total_samples > 0 else 0.0,
    }
    return model, metadata


def save_checkpoint(model: DrosophilaConnectomeSNN, path: str, metadata: Dict[str, object]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "pretraining": metadata}, target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-train the Super Fly SNN connectome from trajectory shards using sequence-aware surrogate gradients")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", default=DEFAULT_SAVE_PATH)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--settle-steps", type=int, default=DEFAULT_SETTLE_STEPS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=50)
    args = parser.parse_args()
    model, metadata = pretrain_motor_layer(
        args.dataset, args.epochs, args.lr, args.stride, args.settle_steps, args.seed, args.chunk_size
    )
    save_checkpoint(model, args.output, metadata)
    print(json.dumps({"checkpoint": os.path.abspath(args.output), **metadata}, indent=2))


if __name__ == "__main__":
    main()
