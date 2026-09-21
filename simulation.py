import os
import random
import numpy as np
import torch

from vision import OmmatidiaVisionPreprocessor
from connectome import DrosophilaConnectomeSNN
from stdp import DualDopamineSTDP
from ram_tracker import MarioRAMTracker
from rom_importer import import_nes_rom
from telemetry import DrosophilaTelemetryOverlay
from storage import RunStorage

DEFAULT_ROM_PATH = "roms/Super Mario Bros. (World).nes"
DEFAULT_SAVE_PATH = "drosophila_snn.pth"
DEFAULT_LR = 0.005
DEFAULT_MAX_STEPS = 2000
MAX_SETTLE_STEPS = 10
DEFAULT_SETTLE_STEPS = 3

# NES Action mapping: [NOOP, RIGHT+RUN, JUMP, RIGHT+RUN+JUMP]
# NES retro action array (12 buttons): [B, Y, SELECT, START, UP, DOWN, LEFT, RIGHT, A, MODE, L, R]
ACTION_MAP = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0: NOOP
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 1: RIGHT + B run
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # 2: JUMP (A button)
    [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],  # 3: RIGHT + JUMP (A button + B run)
]

DEFAULT_ACTION = 1  # Default to RIGHT when no motor neurons fire

def get_game_id():
    """Resolve the Super Mario Bros game id available in the stable-retro install."""
    import stable_retro
    return "SuperMarioBros-Nes-v0" if "SuperMarioBros-Nes-v0" in stable_retro.data.list_games() else "SuperMarioBros-Nes"

def make_env(rom_path=None, state="Level1-1"):
    """Import the ROM (if present) and create a Super Mario Bros stable-retro env.

    Args:
        rom_path: Path to Super Mario Bros ROM file.
        state: Initial state level name (e.g. 'Level1-1', 'Level1-2', 'Level1-3', 'Level1-4', etc.).

    Returns:
        env: Fresh stable-retro environment on the requested state.
        game_id: Resolved stable-retro game id.
    """
    import stable_retro

    if rom_path and os.path.exists(rom_path):
        import_nes_rom(rom_path)

    game_id = get_game_id()
    env = stable_retro.make(
        game=game_id,
        state=state,
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
                 bootstrap_episodes=20, max_bootstrap_step=600, curriculum=True, policy="agent",
                 states=None, runs_dir="runs", run_id=None, seed=None, settle_steps=DEFAULT_SETTLE_STEPS):
        self.rom_path = rom_path
        self.save_path = save_path
        self.lr = lr
        self.bootstrap_episodes = bootstrap_episodes
        self.max_bootstrap_step = max_bootstrap_step
        self.curriculum = curriculum
        self.seed = seed
        if policy not in {"agent", "right_only", "bootstrap_only"}:
            raise ValueError(f"Unknown evaluation policy: {policy}")
        self.policy = policy

        if not (1 <= settle_steps <= MAX_SETTLE_STEPS):
            raise ValueError(f"settle_steps must be between 1 and {MAX_SETTLE_STEPS}")
        self.settle_steps = settle_steps

        if states is None:
            self.states = ["Level1-1"]
        elif isinstance(states, str):
            self.states = [s.strip() for s in states.split(",") if s.strip()]
        else:
            self.states = list(states)
        if not self.states:
            self.states = ["Level1-1"]
        self.current_state = self.states[0]

        # Peek at save_path if present to check for existing run_id
        if run_id is None and os.path.exists(save_path):
            try:
                peek_payload = torch.load(save_path, weights_only=False)
                if isinstance(peek_payload, dict) and "run_id" in peek_payload:
                    run_id = peek_payload["run_id"]
            except Exception:
                pass

        self.run_storage = RunStorage(runs_dir=runs_dir, run_id=run_id)

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

        # Population rate temporal motor decoder state
        self.motor_rate_trace = torch.zeros(self.model.num_motor_ganglion)
        self.tau_motor_rate = 10.0
        self.decay_motor_rate = float(np.exp(-1.0 / self.tau_motor_rate))

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        if os.path.exists(save_path):
            print(f"Loading existing model weights from {save_path}...")
            self.run_storage.migrate_legacy_checkpoint(save_path)
            self.load_checkpoint(save_path)

    def load_checkpoint(self, path_or_dir=None):
        """Load full simulation state from a checkpoint path or run directory."""
        payload = self.run_storage.load_checkpoint(path_or_dir)

        if isinstance(payload, dict) and "model_state_dict" in payload:
            # Rebind run_storage to loaded run_id if adopting existing run identity
            if "run_id" in payload and payload["run_id"] and self.run_storage.run_id != payload["run_id"]:
                self.run_storage = RunStorage(
                    runs_dir=self.run_storage.runs_dir,
                    run_id=payload["run_id"],
                    tenant_scope=payload.get("tenant_scope", "default"),
                    repository_id=payload.get("repository_id", "super-fly"),
                )

            self.model.load_state_dict(payload["model_state_dict"])
            if "episode" in payload:
                self.current_episode = payload["episode"]
            if "step" in payload:
                self.current_step = payload["step"]
            if "best_x" in payload:
                self.best_x = payload["best_x"]

            # Restore policy configuration if serialized
            if "policy_config" in payload:
                p_cfg = payload["policy_config"]
                if "policy" in p_cfg:
                    self.policy = p_cfg["policy"]
                if "lr" in p_cfg:
                    self.lr = p_cfg["lr"]
                    self.stdp.lr = self.lr
                if "settle_steps" in p_cfg:
                    self.settle_steps = p_cfg["settle_steps"]
                if "seed" in p_cfg and p_cfg["seed"] is not None:
                    self.seed = p_cfg["seed"]

            # Restore curriculum control state if serialized
            if "curriculum_state" in payload:
                cs = payload["curriculum_state"]
                self.current_state = cs.get("current_state", self.current_state)
                self.states = cs.get("states", self.states)
                self.curriculum = cs.get("curriculum", self.curriculum)
                self.bootstrap_episodes = cs.get("bootstrap_episodes", self.bootstrap_episodes)
                self.max_bootstrap_step = cs.get("max_bootstrap_step", self.max_bootstrap_step)

            if "rng_states" in payload:
                rngs = payload["rng_states"]
                if rngs.get("python") is not None:
                    random.setstate(rngs["python"])
                if rngs.get("numpy") is not None:
                    np.random.set_state(rngs["numpy"])
                if rngs.get("torch") is not None:
                    torch.set_rng_state(rngs["torch"])
                if rngs.get("torch_cuda") is not None and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(rngs["torch_cuda"])
        else:
            self.model.load_state_dict(payload)

    def save_checkpoint(self, path=None, is_best=False):
        """Atomically persist simulation state to latest_checkpoint.pth and option best_checkpoint.pth."""
        target_path = path if path is not None else self.save_path
        self.run_storage.save_checkpoint(
            self,
            is_best=is_best,
            save_path=target_path,
            seed=self.seed,
        )

    def maybe_save_record(self):
        """Persist model weights when a new best distance is reached.

        Returns True if a new record was set this call.
        """
        x = self.ram_tracker.max_x_pos
        if x <= self.best_x:
            return False
        self.best_x = x
        self.save_checkpoint(is_best=True)
        return True

    def get_effective_max_bootstrap_step(self, episode: int) -> int:
        """
        Calculate effective max bootstrap step limit based on multi-stage curriculum schedule:
        - Phase 1 (Full assistance): Initial episodes (ep <= bootstrap_episodes / 2) get full max_bootstrap_step.
        - Phase 2 (Curriculum decay): Intermediate episodes linearly decay max bootstrap step limit.
        - Phase 3 (Autonomous execution): Later episodes (ep > bootstrap_episodes) have 0 bootstrap step limit.
        """
        if episode <= 0 or self.bootstrap_episodes <= 0 or episode > self.bootstrap_episodes:
            return 0
        if not self.curriculum:
            return self.max_bootstrap_step

        phase1_episodes = max(1, self.bootstrap_episodes // 2)
        if episode <= phase1_episodes:
            return self.max_bootstrap_step

        phase2_total = max(1, self.bootstrap_episodes - phase1_episodes)
        phase2_ep = episode - phase1_episodes
        decay_factor = max(0.0, 1.0 - (phase2_ep / phase2_total))
        return int(self.max_bootstrap_step * decay_factor)

    def get_state_for_episode(self, episode: int) -> str:
        """Determine level state for given episode in multi-level curriculum."""
        if not self.states:
            return "Level1-1"
        idx = (episode - 1) % len(self.states)
        return self.states[idx]

    def reset_episode(self, env, state: str = None):
        """Reset the environment and all temporal simulation state for a new episode.

        Args:
            env: stable-retro environment.
            state: Optional state name override for this episode.
        """
        self.current_episode += 1

        target_state = state if state is not None else self.get_state_for_episode(self.current_episode)
        self.current_state = target_state

        if hasattr(env, "unwrapped") and hasattr(env.unwrapped, "load_state"):
            env.unwrapped.load_state(target_state)

        obs, _ = env.reset()
        self.preprocessor.reset()
        self.model.reset_state()
        self.ram_tracker.reset()

        self.current_step = 0

        self.hold_jump_counter = 0
        self.refractory_counter = 0
        self.bootstrap_pulse_counter = 0
        self.motor_rate_trace.zero_()

        self.action_source = "right"
        self.model_jumps = 0
        self.assisted_jumps = 0
        effective_max = self.get_effective_max_bootstrap_step(self.current_episode)
        self.bootstrap_active = self.policy != "right_only" and effective_max > 0

        return obs

    def step(self, env, obs, train: bool = True):
        """Advance the agent by one step.

        Args:
            env: stable-retro environment
            obs: current RGB frame
            train: if True, applies STDP learning updates and trace injection;
                   if False (eval_mode), learning updates are skipped.

        Returns a dict with the new observation plus per-step telemetry:
        obs, action_idx, reward, d_pam, d_ppl1, ram_info, telemetry_info, layer_acts, terminated, truncated,
        action_source, model_jumps, assisted_jumps, bootstrap_active.
        """
        self.current_step += 1
        effective_max = self.get_effective_max_bootstrap_step(self.current_episode)
        self.bootstrap_active = self.policy != "right_only" and effective_max > 0 and self.current_step <= effective_max

        features, _ = self.preprocessor.process_frame(obs)
        accumulated_motor_spikes = torch.zeros(self.model.num_motor_ganglion)
        layer_acts = None

        for _ in range(self.settle_steps):
            spikes = self.preprocessor.generate_poisson_spikes(features)
            if train:
                self.model.train()
                m_spikes, layer_acts = self.model(spikes)
            else:
                self.model.eval()
                with torch.no_grad():
                    m_spikes, layer_acts = self.model(spikes)
            accumulated_motor_spikes += m_spikes

        # Update population rate temporal trace across settling frame window
        mean_step_spikes = accumulated_motor_spikes / float(self.settle_steps)
        self.motor_rate_trace = self.decay_motor_rate * self.motor_rate_trace + mean_step_spikes

        # Decode motor request using both frame-step spikes and integrated population rate trace
        rate_jump_signal = float(self.motor_rate_trace[2] + self.motor_rate_trace[3])
        jump_requested = self.policy == "agent" and (
            accumulated_motor_spikes[2] > 0 or accumulated_motor_spikes[3] > 0 or rate_jump_signal >= 0.15
        )
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

        # Teaching / Exploration motor trace injection (training mode only):
        # If jump is assisted (forced by bootstrap), update layer3_4 post-eligibility trace
        # so STDP associates active sensory patterns with jump timing.
        if train and is_assisted:
            with torch.no_grad():
                self.model.layer3_4.trace_post[3] = self.model.layer3_4.decay_trace * self.model.layer3_4.trace_post[3] + 1.0

        obs, reward, terminated, truncated, _ = env.step(ACTION_MAP[action_idx])

        ram = env.get_ram()
        # Override termination with RAM-level death or stable Level 1-1 completion detection.
        died = self.ram_tracker.is_dead(ram)
        completed = self.ram_tracker.update_completion(ram)
        if died or completed:
            terminated = True

        d_pam, d_ppl1, ram_info = self.ram_tracker.compute_dopamine(ram, terminated, truncated)

        # Apply STDP learning update only during training
        if train:
            self.stdp.step(d_pam, d_ppl1)

        telemetry_info = dict(ram_info)
        telemetry_info.update({
            "action_source": self.action_source,
            "model_jumps": self.model_jumps,
            "assisted_jumps": self.assisted_jumps,
            "bootstrap_active": self.bootstrap_active,
            "policy": self.policy,
            "settle_steps": self.settle_steps,
            "died": died,
            "completed": completed,
        })

        return {
            "obs": obs,
            "action_idx": action_idx,
            "reward": reward,
            "d_pam": d_pam,
            "d_ppl1": d_ppl1,
            "ram_info": ram_info,
            "telemetry_info": telemetry_info,
            "layer_acts": layer_acts,
            "terminated": terminated,
            "terminated": terminated,
            "truncated": truncated,
            "action_source": self.action_source,
            "model_jumps": self.model_jumps,
            "assisted_jumps": self.assisted_jumps,
            "bootstrap_active": self.bootstrap_active,
            "policy": self.policy,
            "died": died,
            "completed": completed,
        }
