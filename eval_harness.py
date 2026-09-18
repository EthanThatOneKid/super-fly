import argparse
import os
import json
import torch

from simulation import Simulation, make_env, DEFAULT_ROM_PATH, DEFAULT_SAVE_PATH

import numpy as np

def evaluate_agent(rom_path=DEFAULT_ROM_PATH, save_path=DEFAULT_SAVE_PATH, episodes=5, max_steps=1000,
                   bootstrap_episodes=0, max_bootstrap_step=0, seed=42):
    """
    Evaluates the fly SNN agent deterministically in an isolated environment without updating weights.
    """
    if not os.path.exists(rom_path):
        print(f"ROM path '{rom_path}' not found. Skipping evaluation harness execution.")
        return None

    # Seed random number generators for reproducible evaluation trajectories
    np.random.seed(seed)
    torch.manual_seed(seed)

    env, game_id = make_env(rom_path)
    sim = Simulation(rom_path=rom_path, save_path=save_path,
                     bootstrap_episodes=bootstrap_episodes, max_bootstrap_step=max_bootstrap_step)

    results = []

    try:
        for ep in range(1, episodes + 1):
            obs = sim.reset_episode(env)
            ep_pam = 0.0
            ep_ppl1 = 0.0
            step = 0

            while step < max_steps:
                step += 1
                # Run step with train=False to disable STDP weight updates and trace injection
                outcome = sim.step(env, obs, train=False)
                obs = outcome["obs"]
                ep_pam += outcome["d_pam"]
                ep_ppl1 += outcome["d_ppl1"]

                if outcome["terminated"] or outcome["truncated"]:
                    break

            ep_result = {
                "episode": ep,
                "steps": step,
                "max_x": outcome["ram_info"]["max_x_pos"],
                "pam": round(ep_pam, 2),
                "ppl1": round(ep_ppl1, 2),
                "model_jumps": outcome["model_jumps"],
                "assisted_jumps": outcome["assisted_jumps"],
                "action_source": outcome["action_source"],
                "bootstrap_active": outcome["bootstrap_active"],
            }
            results.append(ep_result)
    finally:
        env.close()

    avg_max_x = sum(r["max_x"] for r in results) / len(results) if results else 0
    total_model_jumps = sum(r["model_jumps"] for r in results)
    total_assisted_jumps = sum(r["assisted_jumps"] for r in results)

    summary = {
        "episodes_evaluated": len(results),
        "avg_max_x": round(avg_max_x, 2),
        "best_max_x": max(r["max_x"] for r in results) if results else 0,
        "total_model_jumps": total_model_jumps,
        "total_assisted_jumps": total_assisted_jumps,
        "episodes": results,
    }

    return summary

def main():
    parser = argparse.ArgumentParser(description="Deterministic Evaluation Harness for Drosophila SMB SNN Agent")
    parser.add_argument("--rom", type=str, default=DEFAULT_ROM_PATH, help="Path to NES ROM")
    parser.add_argument("--save-path", type=str, default=DEFAULT_SAVE_PATH, help="Path to SNN weights checkpoint")
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max steps per episode")
    parser.add_argument("--bootstrap-episodes", type=int, default=0, help="Bootstrap episodes override (0 to evaluate post-bootstrap)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible evaluation trajectories")
    args = parser.parse_args()

    summary = evaluate_agent(
        rom_path=args.rom,
        save_path=args.save_path,
        episodes=args.episodes,
        max_steps=args.max_steps,
        bootstrap_episodes=args.bootstrap_episodes,
        max_bootstrap_step=0 if args.bootstrap_episodes == 0 else 600,
        seed=args.seed,
    )

    if summary is not None:
        print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
