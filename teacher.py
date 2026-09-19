import argparse
import json
from pathlib import Path

import stable_retro

from ram_tracker import MarioRAMTracker
from rom_importer import import_nes_rom
from simulation import DEFAULT_ROM_PATH, get_game_id
from trajectory import write_shard

TEACHER_PLAN = """R R R R R R R J R R J R R R R J R J J R R R R R J R R R R R R J R R R R R R J R R R J R R R J J R J R R R R J R R R R J R J R R R J R J R R R J R R R J R R J J R J J R J J R R R R R R R R R R R R R R""".split()
PLAN_ACTIONS = {"R": 1, "J": 3}
TEACHER_ACTIONS = {
    "R": [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    "J": [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
}
MACRO_FRAMES = 15


def collect_teacher_episode(rom_path: str):
    import_nes_rom(rom_path)
    env = stable_retro.make(
        game=get_game_id(),
        state="Level1-1",
        render_mode=None,
        use_restricted_actions=stable_retro.Actions.FILTERED,
    )
    tracker = MarioRAMTracker()
    frames, actions, ram, terminated, truncated = [], [], [], [], []
    obs, _ = env.reset()
    env.unwrapped.load_state("Level1-1")
    obs, _ = env.reset()
    completed = False
    try:
        for token in TEACHER_PLAN:
            action_idx = PLAN_ACTIONS[token]
            for _ in range(MACRO_FRAMES):
                frames.append(obs.copy())
                actions.append(action_idx)
                obs, _, env_terminated, env_truncated, _ = env.step(TEACHER_ACTIONS[token])
                after_ram = env.get_ram().copy()
                completed = tracker.update_completion(after_ram)
                ram.append(after_ram)
                episode_terminated = bool(env_terminated or tracker.is_dead(after_ram) or completed)
                terminated.append(episode_terminated)
                truncated.append(bool(env_truncated))
                if episode_terminated or env_truncated:
                    return frames, actions, ram, terminated, truncated, completed
    finally:
        env.close()
    return frames, actions, ram, terminated, truncated, completed


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect a deterministic successful Level 1-1 teacher trajectory")
    parser.add_argument("--rom", default=DEFAULT_ROM_PATH)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    frames, actions, ram, terminated, truncated, completed = collect_teacher_episode(args.rom)
    shard = write_shard(args.output, frames, actions, ram, terminated, truncated, {
        "source": "deterministic_beam_teacher",
        "level": "Level1-1",
        "completed": completed,
        "macro_frames": MACRO_FRAMES,
    })
    print(json.dumps({"shard": str(Path(shard).resolve()), "samples": len(actions), "completed": completed}, indent=2))


if __name__ == "__main__":
    main()
