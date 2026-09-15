import argparse
import os
import sys
import time
import cv2
import numpy as np
import torch

from vision import OmmatidiaVisionPreprocessor
from connectome import DrosophilaConnectomeSNN
from stdp import DualDopamineSTDP
from ram_tracker import MarioRAMTracker
from rom_importer import import_nes_rom
from telemetry import DrosophilaTelemetryOverlay

# NES Action mapping: [NOOP, RIGHT, JUMP, RIGHT+JUMP]
# NES retro action array (12 buttons): [B, Y, SELECT, START, UP, DOWN, LEFT, RIGHT, A, MODE, L, R]
# Index 0: B, Index 7: RIGHT, Index 8: A (JUMP)
ACTION_MAP = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0: NOOP
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 1: RIGHT
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # 2: JUMP (A button)
    [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],  # 3: RIGHT + JUMP (A button + RIGHT + B run)
]

def parse_args():
    parser = argparse.ArgumentParser(description="Drosophila Melanogaster Connectome SNN - Super Mario Bros")
    parser.add_argument("--rom", type=str, default="roms/Super Mario Bros. (World).nes", help="Path to Super Mario Bros NES ROM file")
    parser.add_argument("--episodes", type=int, default=50, help="Number of training episodes")
    parser.add_argument("--max-steps", type=int, default=2000, help="Max steps per episode")
    parser.add_argument("--headless", action="store_true", help="Run without UI overlay rendering")
    parser.add_argument("--render", action="store_true", help="Enable live visualization telemetry window")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate for dopamine STDP")
    parser.add_argument("--save-path", type=str, default="drosophila_snn.pth", help="Path to save/load SNN model weights")
    return parser.parse_args()

def main():
    args = parse_args()

    # Import ROM if provided
    if args.rom and os.path.exists(args.rom):
        import_nes_rom(args.rom)

    import stable_retro

    # Check games list
    game_id = "SuperMarioBros-Nes-v0" if "SuperMarioBros-Nes-v0" in stable_retro.data.list_games() else "SuperMarioBros-Nes"
    try:
        env = stable_retro.make(game=game_id, state="Level1-1", render_mode=None, use_restricted_actions=stable_retro.Actions.FILTERED)
    except Exception as e:
        print(f"Error loading SuperMarioBros-Nes: {e}")
        print("Please pass your Super Mario Bros NES ROM file via '--rom path/to/rom.nes' to initialize the environment.")
        sys.exit(1)

    preprocessor = OmmatidiaVisionPreprocessor(grid_h=28, grid_w=28)
    model = DrosophilaConnectomeSNN(num_ommatidia=784, channels_per_ommatidium=5)
    stdp = DualDopamineSTDP(model, lr=args.lr)
    ram_tracker = MarioRAMTracker()
    telemetry = DrosophilaTelemetryOverlay()

    if os.path.exists(args.save_path):
        print(f"Loading existing model weights from {args.save_path}...")
        model.load_state_dict(torch.load(args.save_path))

    best_x_pos = 0
    print(f"Starting Drosophila Melanogaster SNN Training Loop on {game_id}...")

    for episode in range(1, args.episodes + 1):
        obs, info = env.reset()
        preprocessor.reset()
        model.reset_state()
        ram_tracker.reset()

        ep_pam = 0.0
        ep_ppl1 = 0.0
        step = 0

        while step < args.max_steps:
            step += 1

            # 1. Vision preprocessing
            features, debug_info = preprocessor.process_frame(obs)
            spikes = preprocessor.generate_poisson_spikes(features)

            # 2. SNN forward pass
            motor_spikes, layer_acts = model(spikes)

            active_actions = torch.where(motor_spikes > 0)[0]
            if len(active_actions) > 0:
                action_idx = active_actions[0].item()
            else:
                action_idx = 1  # Default to RIGHT

            retro_action = ACTION_MAP[action_idx]

            # 3. Environment Step
            obs, reward, terminated, truncated, env_info = env.step(retro_action)

            # Extract RAM state
            ram = env.get_ram()
            d_pam, d_ppl1, ram_info = ram_tracker.compute_dopamine(ram, terminated, truncated)

            # 4. Dopamine STDP update
            stdp.step(d_pam, d_ppl1)

            ep_pam += d_pam
            ep_ppl1 += d_ppl1

            # 5. Live Telemetry Rendering
            if args.render:
                canvas = telemetry.render_overlay(obs, layer_acts, d_pam, d_ppl1, ram_info)
                cv2.imshow("Drosophila Fly Brain SNN Telemetry", canvas)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("User requested exit.")
                    env.close()
                    cv2.destroyAllWindows()
                    return

            if terminated or truncated:
                break

        x_reached = ram_tracker.max_x_pos
        if x_reached > best_x_pos:
            best_x_pos = x_reached
            torch.save(model.state_dict(), args.save_path)
            print(f"New Record! Distance: {x_reached} (Saved model to {args.save_path})")

        print(f"Episode {episode}/{args.episodes} | Steps: {step} | Max X: {x_reached} | Best X: {best_x_pos} | PAM: {ep_pam:.2f} | PPL1: {ep_ppl1:.2f}")

    env.close()
    if args.render:
        cv2.destroyAllWindows()
    print("Training finished successfully.")

if __name__ == "__main__":
    main()
