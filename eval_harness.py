import argparse
import json
import os

import numpy as np
import torch

from simulation import Simulation, make_env, DEFAULT_ROM_PATH, DEFAULT_SAVE_PATH, DEFAULT_SETTLE_STEPS


POLICIES = ("agent", "right_only", "bootstrap_only")
LAYER_NAMES = ("ommatidia", "optic_lobe", "central_complex", "motor_ganglion")


def evaluate_agent(
    rom_path=DEFAULT_ROM_PATH,
    save_path=DEFAULT_SAVE_PATH,
    episodes=5,
    max_steps=1000,
    bootstrap_episodes=0,
    max_bootstrap_step=0,
    seed=42,
    settle_steps=DEFAULT_SETTLE_STEPS,
    policy="agent",
    states=None,
):
    """Evaluate one policy without changing checkpoint weights."""
    if policy not in POLICIES:
        raise ValueError(f"Unknown evaluation policy: {policy}")
    if episodes < 0 or max_steps < 0:
        raise ValueError("episodes and max_steps must be non-negative")
    if not os.path.exists(rom_path):
        print(f"ROM path '{rom_path}' not found. Skipping evaluation harness execution.")
        return None

    np.random.seed(seed)
    torch.manual_seed(seed)

    states_list = [s.strip() for s in states.split(",") if s.strip()] if isinstance(states, str) else (list(states) if states else ["Level1-1"])
    initial_state = states_list[0] if states_list else "Level1-1"

    env, _ = make_env(rom_path, state=initial_state)
    sim = Simulation(
        rom_path=rom_path,
        save_path=save_path,
        bootstrap_episodes=bootstrap_episodes,
        max_bootstrap_step=max_bootstrap_step,
        policy=policy,
        states=states_list,
        settle_steps=settle_steps,
        seed=seed,
    )

    results = []
    try:
        for ep in range(1, episodes + 1):
            obs = sim.reset_episode(env)
            ep_pam = 0.0
            ep_ppl1 = 0.0
            step = 0
            model_jump_frames = 0
            assisted_jump_frames = 0
            outcome = None
            layer_spike_counts = {name: 0 for name in LAYER_NAMES}

            while step < max_steps:
                step += 1
                outcome = sim.step(env, obs, train=False)
                obs = outcome["obs"]
                ep_pam += outcome["d_pam"]
                ep_ppl1 += outcome["d_ppl1"]
                if outcome["action_source"] in {"model", "model_hold"}:
                    model_jump_frames += 1
                if outcome["action_source"] in {"bootstrap", "bootstrap_hold"}:
                    assisted_jump_frames += 1
                for name in LAYER_NAMES:
                    layer_spike_counts[name] += int(outcome["layer_acts"][name].sum().item())
                if outcome["terminated"] or outcome["truncated"]:
                    break

            terminated = bool(outcome["terminated"]) if outcome else False
            truncated = bool(outcome["truncated"]) if outcome else False
            ram_info = outcome["ram_info"] if outcome else {"max_x_pos": 0, "max_sub_page": 0, "max_page": 0}
            completed = bool(outcome["completed"]) if outcome else False
            model_jumps = outcome["model_jumps"] if outcome else 0
            assisted_jumps = outcome["assisted_jumps"] if outcome else 0
            model_jump_freq = round(model_jumps / step, 6) if step else 0.0

            results.append(
                {
                    "episode": ep,
                    "state": sim.current_state,
                    "steps": step,
                    "max_x": ram_info.get("max_x_pos", 0),
                    "max_sub_page": ram_info.get("max_sub_page", 0),
                    "max_page": ram_info.get("max_page", 0),
                    "pam": round(ep_pam, 2),
                    "ppl1": round(ep_ppl1, 2),
                    "terminated": terminated,
                    "truncated": truncated,
                    "died": bool(outcome["died"]) if outcome else False,
                    "completed": completed,
                    "completion_step": step if completed else None,
                    "survived_to_limit": step >= max_steps and not terminated,
                    "model_jumps": model_jumps,
                    "assisted_jumps": assisted_jumps,
                    "model_jump_frames": model_jump_frames,
                    "assisted_jump_frames": assisted_jump_frames,
                    "model_jump_frequency": model_jump_freq,
                    "action_source": outcome["action_source"] if outcome else "right",
                    "bootstrap_active": outcome["bootstrap_active"] if outcome else False,
                    "layer_spike_counts": layer_spike_counts,
                    "layer_spike_rates": {
                        name: round(count / step, 6) if step else 0
                        for name, count in layer_spike_counts.items()
                    },
                }
            )
    finally:
        env.close()

    x_vals = [r["max_x"] for r in results]
    completed_steps = [r["completion_step"] for r in results if r["completion_step"] is not None]
    avg_completion_step = round(float(np.mean(completed_steps)), 2) if completed_steps else None
    death_count = sum(r["died"] for r in results)
    assisted_count = sum(r["assisted_jumps"] for r in results)
    bootstrap_assistance_used = assisted_count > 0 or bootstrap_episodes > 0 or max_bootstrap_step > 0

    summary = {
        "policy": policy,
        "seed": seed,
        "settle_steps": sim.settle_steps,
        "bootstrap_assistance_used": bootstrap_assistance_used,
        "teacher_assistance_used": False,
        "episodes_evaluated": len(results),
        "best_x": max(x_vals, default=0),
        "best_max_x": max(x_vals, default=0),
        "avg_max_x": round(sum(x_vals) / len(results), 2) if results else 0.0,
        "min_max_x": min(x_vals, default=0),
        "std_max_x": round(float(np.std(x_vals)), 2) if results else 0.0,
        "avg_max_sub_page": round(sum(r["max_sub_page"] for r in results) / len(results), 2) if results else 0.0,
        "avg_max_page": round(sum(r["max_page"] for r in results) / len(results), 2) if results else 0.0,
        "best_sub_page": max((r["max_sub_page"] for r in results), default=0),
        "best_page": max((r["max_page"] for r in results), default=0),
        "sub_page_milestone_rate": round(sum(1 for r in results if r["max_sub_page"] > 0) / len(results), 4) if results else 0.0,
        "page_milestone_rate": round(sum(1 for r in results if r["max_page"] > 0) / len(results), 4) if results else 0.0,
        "avg_steps": round(sum(r["steps"] for r in results) / len(results), 2) if results else 0,
        "episodes_died": death_count,
        "death_rate": round(death_count / len(results), 4) if results else 0.0,
        "episodes_completed": sum(r["completed"] for r in results),
        "completion_rate": round(sum(r["completed"] for r in results) / len(results), 4) if results else 0.0,
        "completion_step": avg_completion_step,
        "episodes_terminated": sum(r["terminated"] for r in results),
        "episodes_truncated": sum(r["truncated"] for r in results),
        "episodes_survived_to_limit": sum(r["survived_to_limit"] for r in results),
        "total_model_jumps": sum(r["model_jumps"] for r in results),
        "total_assisted_jumps": assisted_count,
        "total_model_jump_frames": sum(r["model_jump_frames"] for r in results),
        "total_assisted_jump_frames": sum(r["assisted_jump_frames"] for r in results),
        "avg_model_jump_frequency": round(sum(r["model_jump_frequency"] for r in results) / len(results), 6) if results else 0.0,
        "avg_layer_spike_rates": {
            name: round(
                sum(r["layer_spike_rates"][name] for r in results) / len(results),
                6,
            ) if results else 0
            for name in LAYER_NAMES
        },
        "episodes": results,
    }
    return summary


def evaluate_policies(
    rom_path=DEFAULT_ROM_PATH,
    save_path=DEFAULT_SAVE_PATH,
    episodes=5,
    max_steps=1000,
    seed=42,
    settle_steps=DEFAULT_SETTLE_STEPS,
    states=None,
):
    """Evaluate the learned policy against matched RIGHT and bootstrap controls."""
    reports = {
        "agent": evaluate_agent(rom_path, save_path, episodes, max_steps, 0, 0, seed, settle_steps, "agent", states=states),
        "right_only": evaluate_agent(rom_path, save_path, episodes, max_steps, 0, 0, seed, settle_steps, "right_only", states=states),
        "bootstrap_only": evaluate_agent(
            rom_path, save_path, episodes, max_steps, episodes, 600, seed, settle_steps, "bootstrap_only", states=states
        ),
    }
    agent_x = reports["agent"]["avg_max_x"]
    return {
        "seed": seed,
        "episodes": episodes,
        "max_steps": max_steps,
        "policies": reports,
        "agent_delta_vs_right_only": round(agent_x - reports["right_only"]["avg_max_x"], 2),
        "agent_delta_vs_bootstrap_only": round(agent_x - reports["bootstrap_only"]["avg_max_x"], 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Deterministic Evaluation Harness for Drosophila SMB SNN Agent")
    parser.add_argument("--rom", type=str, default=DEFAULT_ROM_PATH, help="Path to NES ROM")
    parser.add_argument("--save-path", type=str, default=DEFAULT_SAVE_PATH, help="Path to SNN weights checkpoint")
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max steps per episode")
    parser.add_argument("--policy", choices=POLICIES, default="agent", help="Policy to evaluate")
    parser.add_argument("--compare-policies", action="store_true", help="Evaluate agent, RIGHT-only, and bootstrap-only controls")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible evaluation trajectories")
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated list of seeds for multi-seed evaluation (e.g. 42,43,44)")
    parser.add_argument("--settle-steps", type=int, default=DEFAULT_SETTLE_STEPS, help="Number of temporal settling steps per frame (1-10)")
    parser.add_argument("--states", type=str, default="Level1-1", help="Comma-separated list of level states to evaluate (e.g. Level1-1,Level1-2)")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()] if args.seeds else [args.seed]

    if len(seeds) > 1:
        seed_results = []
        for s in seeds:
            if args.compare_policies:
                res = evaluate_policies(args.rom, args.save_path, args.episodes, args.max_steps, s, settle_steps=args.settle_steps, states=args.states)
            else:
                res = evaluate_agent(
                    rom_path=args.rom,
                    save_path=args.save_path,
                    episodes=args.episodes,
                    max_steps=args.max_steps,
                    seed=s,
                    settle_steps=args.settle_steps,
                    policy=args.policy,
                    states=args.states,
                )
            if res is not None:
                seed_results.append(res)

        if not seed_results:
            summary = None
        else:
            all_comp = [r["completion_rate"] for r in seed_results if "completion_rate" in r]
            all_best_x = [r["best_x"] for r in seed_results if "best_x" in r]
            all_death = [r["death_rate"] for r in seed_results if "death_rate" in r]
            summary = {
                "seeds_evaluated": seeds,
                "episodes_per_seed": args.episodes,
                "policy": args.policy,
                "settle_steps": args.settle_steps,
                "bootstrap_assistance_used": any(r.get("bootstrap_assistance_used", False) for r in seed_results),
                "teacher_assistance_used": False,
                "overall_completion_rate": round(float(np.mean(all_comp)), 4) if all_comp else 0.0,
                "overall_best_x": max(all_best_x, default=0),
                "overall_death_rate": round(float(np.mean(all_death)), 4) if all_death else 0.0,
                "per_seed_results": seed_results,
            }
    elif args.compare_policies:
        summary = evaluate_policies(args.rom, args.save_path, args.episodes, args.max_steps, seeds[0], settle_steps=args.settle_steps, states=args.states)
    else:
        summary = evaluate_agent(
            rom_path=args.rom,
            save_path=args.save_path,
            episodes=args.episodes,
            max_steps=args.max_steps,
            seed=seeds[0],
            settle_steps=args.settle_steps,
            policy=args.policy,
            states=args.states,
        )

    if summary is not None:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
