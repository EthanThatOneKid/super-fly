import os

import torch

from vision import OmmatidiaVisionPreprocessor
from connectome import DrosophilaConnectomeSNN
from stdp import DualDopamineSTDP
from ram_tracker import MarioRAMTracker
from rom_importer import import_nes_rom
from telemetry import DrosophilaTelemetryOverlay

DEFAULT_ROM_PATH = "roms/Super Mario Bros. (World).nes"
DEFAULT_SAVE_PATH = "drosophila_snn.pth"
DEFAULT_LR = 0.005
DEFAULT_MAX_STEPS = 2000

# NES Action mapping: [NOOP, RIGHT, JUMP, RIGHT+JUMP]
# NES retro action array (12 buttons): [B, Y, SELECT, START, UP, DOWN, LEFT, RIGHT, A, MODE, L, R]
ACTION_MAP = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0: NOOP
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 1: RIGHT
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # 2: JUMP (A button)
    [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],  # 3: RIGHT + JUMP (A button + RIGHT + B run)
]

DEFAULT_ACTION = 1  # Default to RIGHT when no motor neurons fire

def get_game_id():
    """Resolve the Super Mario Bros game id available in the stable-retro install."""
    import stable_retro
    return "SuperMarioBros-Nes-v0" if "SuperMarioBros-Nes-v0" in stable_retro.data.list_games() else "SuperMarioBros-Nes"

def make_env(rom_path=None):
    """Import the ROM (if present) and create a Super Mario Bros stable-retro env.

    Returns:
        env: Fresh stable-retro environment on the Level1-1 state.
        game_id: Resolved stable-retro game id.
    """
    import stable_retro

    if rom_path and os.path.exists(rom_path):
        import_nes_rom(rom_path)

    game_id = get_game_id()
    env = stable_retro.make(
        game=game_id,
        state="Level1-1",
        render_mode=None,
        use_restricted_actions=stable_retro.Actions.FILTERED,
    )
    return env, game_id


class Simulation:
    """
    Shared Drosophila connectome SNN simulation core used by both the CLI
    trainer and the live web dashboard.

    Owns the visual preprocessor, the SNN model, the dopamine STDP learner,
    RAM-based reward tracking, and the telemetry overlay, and exposes a single
    brain-driven step primitive so both entrypoints share identical wiring.
    """

    def __init__(self, rom_path=DEFAULT_ROM_PATH, save_path=DEFAULT_SAVE_PATH, lr=DEFAULT_LR,
                 bootstrap_episodes=20, max_bootstrap_step=600):
        self.rom_path = rom_path
        self.save_path = save_path
        self.lr = lr
        self.bootstrap_episodes = bootstrap_episodes
        self.max_bootstrap_step = max_bootstrap_step

        self.preprocessor = OmmatidiaVisionPreprocessor(grid_h=28, grid_w=28)
        self.model = DrosophilaConnectomeSNN(num_ommatidia=784, channels_per_ommatidium=5)
        self.stdp = DualDopamineSTDP(self.model, lr=lr)
        self.ram_tracker = MarioRAMTracker()
        self.telemetry = DrosophilaTelemetryOverlay()

        self.best_x = 0
        self.current_episode = 0
        self.current_step = 0

        # Jump controller state
        self.hold_jump_counter = 0
        self.refractory_counter = 0
        self.bootstrap_pulse_counter = 0

        # Diagnostics & Action stats
        self.action_source = "right"
        self.model_jumps = 0
        self.assisted_jumps = 0
        self.bootstrap_active = False

        if os.path.exists(save_path):
            print(f"Loading existing model weights from {save_path}...")
            self.model.load_state_dict(torch.load(save_path))

    def reset_episode(self, env):
        """Reset the environment and all temporal simulation state for a new episode."""
        obs, _ = env.reset()
        self.preprocessor.reset()
        self.model.reset_state()
        self.ram_tracker.reset()

        self.current_episode += 1
        self.current_step = 0

        self.hold_jump_counter = 0
        self.refractory_counter = 0
        self.bootstrap_pulse_counter = 0

        self.action_source = "right"
        self.model_jumps = 0
        self.assisted_jumps = 0
        self.bootstrap_active = self.current_episode <= self.bootstrap_episodes

        return obs

    def save_checkpoint(self, path=None):
        """Atomically persist SNN weights to disk."""
        target_path = path if path is not None else self.save_path
        tmp_path = f"{target_path}.tmp"
        torch.save(self.model.state_dict(), tmp_path)
        os.replace(tmp_path, target_path)

    def maybe_save_record(self):
        """Persist model weights when a new best distance is reached.

        Returns True if a new record was set this call.
        """
        x = self.ram_tracker.max_x_pos
        if x <= self.best_x:
            return False
        self.best_x = x
        self.save_checkpoint()
        return True

    def step(self, env, obs):
        """Advance the agent by one step.

        Returns a dict with the new observation plus per-step telemetry:
        obs, action_idx, reward, d_pam, d_ppl1, ram_info, layer_acts, terminated, truncated,
        action_source, model_jumps, assisted_jumps, bootstrap_active.
        """
        self.current_step += 1
        self.bootstrap_active = self.current_episode <= self.bootstrap_episodes and self.current_step <= self.max_bootstrap_step

        features, _ = self.preprocessor.process_frame(obs)
        spikes = self.preprocessor.generate_poisson_spikes(features)

        motor_spikes, layer_acts = self.model(spikes)

        # Check model motor outputs (2: JUMP, 3: RIGHT+JUMP)
        jump_requested = (motor_spikes[2] > 0 or motor_spikes[3] > 0)
        action_source = "right"
        execute_jump = False
        is_assisted = False

        if self.refractory_counter > 0:
            self.refractory_counter -= 1

        if self.hold_jump_counter > 0:
            self.hold_jump_counter -= 1
            execute_jump = True
            action_source = "model_hold"
        elif jump_requested and self.refractory_counter == 0:
            execute_jump = True
            self.hold_jump_counter = 3  # Hold for 4 frames total (current frame + 3)
            self.refractory_counter = 24  # 24-step refractory period
            action_source = "model"
            self.model_jumps += 1
        elif self.bootstrap_pulse_counter > 0:
            self.bootstrap_pulse_counter -= 1
            execute_jump = True
            action_source = "bootstrap_hold"
            is_assisted = True
        elif self.bootstrap_active and self.current_step >= 128 and (self.current_step - 128) % 84 == 0:
            execute_jump = True
            self.bootstrap_pulse_counter = 19  # 20 frames pulse total
            action_source = "bootstrap"
            self.assisted_jumps += 1
            is_assisted = True

        action_idx = 3 if execute_jump else 1
        self.action_source = action_source

        # Teaching / Exploration motor trace injection:
        # If jump is assisted (forced by bootstrap), update layer3_4 post-eligibility trace
        # so STDP associates active sensory patterns with jump timing.
        if is_assisted:
            with torch.no_grad():
                self.model.layer3_4.trace_post[3] = self.model.layer3_4.decay_trace * self.model.layer3_4.trace_post[3] + 1.0

        obs, reward, terminated, truncated, _ = env.step(ACTION_MAP[action_idx])

        ram = env.get_ram()
        # Override terminated if SMB death state detected in RAM
        if self.ram_tracker.is_dead(ram):
            terminated = True

        d_pam, d_ppl1, ram_info = self.ram_tracker.compute_dopamine(ram, terminated, truncated)
        self.stdp.step(d_pam, d_ppl1)

        return {
            "obs": obs,
            "action_idx": action_idx,
            "reward": reward,
            "d_pam": d_pam,
            "d_ppl1": d_ppl1,
            "ram_info": ram_info,
            "layer_acts": layer_acts,
            "terminated": terminated,
            "truncated": truncated,
            "action_source": self.action_source,
            "model_jumps": self.model_jumps,
            "assisted_jumps": self.assisted_jumps,
            "bootstrap_active": self.bootstrap_active,
        }
