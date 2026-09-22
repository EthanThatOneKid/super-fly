"""Deterministic teacher trajectory collection for Level 1-1.

The teacher is a fixed macro plan (RUN / RUN+JUMP chunks of ``MACRO_FRAMES``
frames) recorded as a checksummed shard. It is an *upper bound* reference, not
model learning: its shard supplies the supervised action targets used by
``pretrain.py`` and by the DAgger recovery windows in ``dagger.py``.

Collection is env-factory driven so the same plan can be replayed on the real
stable-retro ROM or on the offline synthetic stand-in used to validate the
experiment plumbing (``offline_env.OfflineMarioEnv``).
"""

import argparse
import json
from pathlib import Path

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


def make_stable_retro_env(state: str = "Level1-1"):
    """Create a stable-retro environment for the requested level state."""
    import stable_retro  # imported lazily so this module loads without the emulator

    return stable_retro.make(
        game=get_game_id(),
        state=state,
        render_mode=None,
        use_restricted_actions=stable_retro.Actions.FILTERED,
    )


def collect_teacher_episode(rom_path: str, env_factory=None, state: str = "Level1-1"):
    """Replay the deterministic teacher plan and record frames, actions and RAM.

    Args:
        rom_path: Path to the SMB ROM (imported into stable-retro data dir).
        env_factory: callable(state) -> env. Defaults to stable-retro.
        state: Level state to load before replaying the plan.

    Returns:
        (frames, actions, ram, terminated, truncated, completed)
    """
    if env_factory is None:
        import_nes_rom(rom_path)
        env_factory = make_stable_retro_env

    env = env_factory(state)
    tracker = MarioRAMTracker()
    frames, actions, ram, terminated, truncated = [], [], [], [], []
    obs, _ = env.reset()
    if hasattr(env.unwrapped, "load_state"):
        env.unwrapped.load_state(state)
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


def collect_scripted_teacher_episode(env_factory, state: str = "Level1-1", chunk_frames: int = MACRO_FRAMES,
                                     jump_lookahead: int = 24, obstacle_xs=None, max_frames: int = 6000):
    """Chunked, RAM-following scripted teacher used to validate offline plumbing.

    Unlike the recorded research teacher (a fixed macro plan over the real level
    geometry), this teacher makes one chunked decision per ``chunk_frames`` frames
    from the RAM-visible ground state and the next obstacle distance. It exists so
    the DAgger dataset, pretraining and evaluation paths can be exercised without
    the emulator; it is never evidence about learning and its shard records
    ``env_kind: offline_synthetic``.
    """
    if obstacle_xs is None:
        from offline_env import OBSTACLE_XS

        obstacle_xs = OBSTACLE_XS

    env = env_factory(state)
    tracker = MarioRAMTracker()
    frames, actions, ram, terminated, truncated = [], [], [], [], []
    obs, _ = env.reset()
    completed = False
    pending = 0
    macro = "R"
    try:
        while len(frames) < max_frames:
            if pending <= 0:
                current_ram = env.get_ram()
                x_pos = tracker.get_x_pos(current_ram)
                on_ground = current_ram[0x001D] == 0
                imminent = any(0 < obstacle_x - x_pos <= jump_lookahead for obstacle_x in obstacle_xs)
                macro = "J" if (on_ground and imminent) else "R"
                pending = chunk_frames
            pending -= 1

            frames.append(obs.copy())
            actions.append(PLAN_ACTIONS[macro])
            obs, _, env_terminated, env_truncated, _ = env.step(TEACHER_ACTIONS[macro])
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


def write_teacher_shard(output_dir: str, rom_path: str = DEFAULT_ROM_PATH, env_factory=None, state: str = "Level1-1"):
    """Collect a teacher episode and persist it as a checksummed shard."""
    offline = env_factory is not None
    if offline:
        collected = collect_scripted_teacher_episode(env_factory, state=state)
        source = "scripted_offline_teacher"
    else:
        collected = collect_teacher_episode(rom_path, env_factory=None, state=state)
        source = "deterministic_beam_teacher"
    frames, actions, ram, terminated, truncated, completed = collected
    env_kind = "offline_synthetic" if offline else "stable-retro"
    shard = write_shard(output_dir, frames, actions, ram, terminated, truncated, {
        "source": source,
        "level": state,
        "completed": completed,
        "macro_frames": MACRO_FRAMES,
        "env_kind": env_kind,
    }, provenance={
        # Per-shard origin, so an aggregated DAgger dataset can attribute every
        # supervised sample back to the teacher plan or to a rollout window.
        "origin": "teacher",
        "source": source,
        "level": state,
        "completed": completed,
        "macro_frames": MACRO_FRAMES,
        "env_kind": env_kind,
        "samples": len(actions),
    })
    return shard, completed, len(actions)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect a deterministic successful Level 1-1 teacher trajectory")
    parser.add_argument("--rom", default=DEFAULT_ROM_PATH)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    shard, completed, samples = write_teacher_shard(args.output, args.rom)
    print(json.dumps({"shard": str(Path(shard).resolve()), "samples": samples, "completed": completed}, indent=2))


if __name__ == "__main__":
    main()
