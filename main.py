import argparse
import os
import sys

import cv2

from simulation import (
    Simulation,
    make_env,
    DEFAULT_ROM_PATH,
    DEFAULT_SAVE_PATH,
    DEFAULT_LR,
    DEFAULT_MAX_STEPS,
)

def parse_args():
    parser = argparse.ArgumentParser(description="Drosophila Melanogaster Connectome SNN - Super Mario Bros")
    parser.add_argument("--rom", type=str, default=DEFAULT_ROM_PATH, help="Path to Super Mario Bros NES ROM file")
    parser.add_argument("--episodes", type=int, default=50, help="Number of training episodes")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS, help="Max steps per episode")
    parser.add_argument("--headless", action="store_true", help="Run without UI overlay rendering")
    parser.add_argument("--render", action="store_true", help="Enable live visualization telemetry window")
    parser.add_argument("--web", action="store_true", help="Launch live web browser streaming dashboard")
    parser.add_argument("--port", type=int, default=5000, help="Port for web browser dashboard")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Learning rate for dopamine STDP")
    parser.add_argument("--save-path", type=str, default=DEFAULT_SAVE_PATH, help="Path to save/load SNN model weights")
    parser.add_argument("--runs-dir", type=str, default="runs", help="Directory for durable run storage and manifests")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for PyTorch, NumPy, and Python")
    parser.add_argument("--no-curriculum", action="store_true", help="Disable structured curriculum bootstrap decay")
    parser.add_argument("--states", type=str, default="Level1-1", help="Comma-separated list of level states for multi-level curriculum (e.g. Level1-1,Level1-2,Level1-3)")
    return parser.parse_args()

def main():
    args = parse_args()

    # Launch Web Server if --web flag passed
    if args.web:
        from web_server import start_server
        print(f"Starting Drosophila Fly SNN Web Dashboard on http://localhost:{args.port} ...")
        start_server(port=args.port, rom=args.rom, lr=args.lr, save_path=args.save_path, max_steps=args.max_steps)
        return

    states_list = [s.strip() for s in args.states.split(",") if s.strip()] or ["Level1-1"]
    initial_state = states_list[0]

    # Import ROM if provided and create the environment
    try:
        env, game_id = make_env(args.rom if args.rom and os.path.exists(args.rom) else None, state=initial_state)
    except Exception as e:
        print(f"Error loading SuperMarioBros-Nes: {e}")
        print("Please pass your Super Mario Bros NES ROM file via '--rom path/to/rom.nes' to initialize the environment.")
        sys.exit(1)

    sim = Simulation(
        rom_path=args.rom,
        save_path=args.save_path,
        lr=args.lr,
        curriculum=not args.no_curriculum,
        states=states_list,
        runs_dir=args.runs_dir,
        seed=args.seed,
    )

    print(f"Starting Drosophila Melanogaster SNN Training Loop on {game_id}...")

    for episode in range(1, args.episodes + 1):
        obs = sim.reset_episode(env)

        ep_pam = 0.0
        ep_ppl1 = 0.0
        step = 0

        while step < args.max_steps:
            step += 1

            outcome = sim.step(env, obs)
            obs = outcome["obs"]
            ep_pam += outcome["d_pam"]
            ep_ppl1 += outcome["d_ppl1"]

            # Live Telemetry Rendering
            if args.render:
                canvas = sim.telemetry.render_overlay(
                    obs, outcome["layer_acts"], outcome["d_pam"], outcome["d_ppl1"], outcome["telemetry_info"]
                )
                cv2.imshow("Drosophila Fly Brain SNN Telemetry", canvas)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("User requested exit.")
                    env.close()
                    cv2.destroyAllWindows()
                    return

            if outcome["terminated"] or outcome["truncated"]:
                break

        x_reached = sim.ram_tracker.max_x_pos
        if sim.maybe_save_record():
            print(f"New Record! Distance: {x_reached} (Saved model to {args.save_path})")

        print(f"Episode {episode}/{args.episodes} | Steps: {step} | Max X: {x_reached} | Best X: {sim.best_x} | PAM: {ep_pam:.2f} | PPL1: {ep_ppl1:.2f}")

    sim.run_storage.set_status("completed")
    env.close()
    if args.render:
        cv2.destroyAllWindows()
    print("Training finished successfully.")

if __name__ == "__main__":
    main()
