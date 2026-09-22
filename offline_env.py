"""Deterministic offline stand-in for the Super Mario Bros stable-retro environment.

This module exists so the closed-loop DAgger experiment (issue #30) can be
validated end to end on machines without the stable-retro native build — CI and
the dev checkout — and so unit tests can drive the real ``Simulation``,
``MacroActionDecoder``, DAgger and evaluation code paths.

It is **not** a research environment. Its physics are a caricature of Level 1-1
(straight ground, evenly spaced obstacle blocks, three pits, a flagpole at
x=3200), it exists only to exercise plumbing, and every metric produced with it
is reported with ``env_kind="offline_synthetic"`` and
``gate_eligible=False``. Real claims require the ROM and ``stable-retro``.
"""

import numpy as np

is_synthetic = True

FRAME_HEIGHT = 240
FRAME_WIDTH = 256
RAM_SIZE = 0x0800

# Button indices inside the 12-button NES action array (see simulation.ACTION_MAP).
BUTTON_RUN = 0      # B
BUTTON_RIGHT = 7    # RIGHT
BUTTON_JUMP = 8     # A

# SMB RAM addresses mirrored from ram_tracker.MarioRAMTracker.
ADDR_PLAYER_STATE = 0x000E
ADDR_AIRBORNE = 0x001D
ADDR_PAGE_X = 0x006D
ADDR_SUB_X = 0x0086
ADDR_SUB_Y = 0x00CE
ADDR_PAGE_Y = 0x00B5
ADDR_OPER_MODE = 0x0770

GROUND_Y = 160
LEVEL_END_X = 3200
LEVEL_MAX_X = 3400
#: No pits in the stand-in: one clean failure mode (ground-level obstacles) keeps
#: the plumbing validation deterministic. The real ROM supplies real hazards.
PIT_SEGMENTS = ()
OBSTACLE_XS = tuple(range(320, LEVEL_END_X, 320))


class OfflineMarioEnv:
    """Minimal deterministic SMB-like environment with real RAM semantics."""

    is_synthetic = True
    env_kind = "offline_synthetic"

    def __init__(self, state: str = "Level1-1", seed: int = 0, max_frames: int = 4000):
        self.state = state
        self.seed = seed
        self.max_frames = max_frames
        self.ram = np.zeros(RAM_SIZE, dtype=np.uint8)
        self._reset_dynamics()

    # -- stable-retro compatible surface ----------------------------------
    @property
    def unwrapped(self):
        return self

    def load_state(self, state: str) -> None:
        self.state = state

    def close(self) -> None:
        pass

    def get_ram(self) -> np.ndarray:
        return self.ram

    def reset(self):
        self._reset_dynamics()
        self._sync_ram()
        return self._render(), {}

    def step(self, action):
        right = bool(action[BUTTON_RIGHT])
        run = bool(action[BUTTON_RUN])
        jump = bool(action[BUTTON_JUMP])

        previous_x = self.x
        self.frames += 1

        if jump and self.on_ground and not self.dead:
            self.vy = -3.4
            self.on_ground = False
            self.airborne_state = 1

        if self.on_ground and not self.dead:
            if right:
                self.vx = min(self.vx + 1, 2 if run else 1)
            else:
                self.vx = max(0, self.vx - 1)

        if not self.on_ground and not self.dead:
            self.vy += 0.36
            self.y += self.vy
            self.airborne_state = 1 if self.vy < 0 else 2
            if self.y >= GROUND_Y:
                self.y = GROUND_Y
                self.vy = 0.0
                self.on_ground = True
                self.airborne_state = 0

        if not self.dead:
            self.x += self.vx

        collided = False
        if self.on_ground and not self.dead:
            for obstacle_x in OBSTACLE_XS:
                if abs(self.x - obstacle_x) <= 4:
                    collided = True
                    break
        if collided:
            # Blocked by an obstacle: position clamps but forward intent is kept, so
            # a jump while pressed against the wall still carries the player over it.
            self.x = previous_x
            self.collision_frames += 1
        else:
            self.collision_frames = 0

        for start, end in PIT_SEGMENTS:
            if start <= self.x <= end and self.on_ground and not self.dead:
                self._die()

        if self.collision_frames > 45 and not self.dead:
            self._die()

        if self.x >= LEVEL_END_X:
            self.player_state = 0x05  # flagpole / level-clear state held for the detector
            self.completed = True

        if self.frames >= self.max_frames:
            self.truncated = True

        self._sync_ram()
        reward = max(0.0, float(self.x - previous_x))
        # Reaching the flagpole does not end the episode by itself: the RAM completion
        # detector has to observe the level-clear state held for 30 frames, exactly as
        # on the real ROM. The episode ends when the player walks past LEVEL_MAX_X.
        terminated = self.dead or self.x >= LEVEL_MAX_X
        return self._render(), reward, terminated, self.truncated, {}

    # -- internals ---------------------------------------------------------
    def _reset_dynamics(self) -> None:
        self.x = 40.0
        self.y = float(GROUND_Y)
        self.vx = 0.0
        self.vy = 0.0
        self.on_ground = True
        self.airborne_state = 0
        self.frames = 0
        self.dead = False
        self.completed = False
        self.truncated = False
        self.collision_frames = 0
        self.player_state = 0x08

    def _die(self) -> None:
        self.dead = True
        self.player_state = 0x0B
        self.vx = 0.0

    def _sync_ram(self) -> None:
        x = int(self.x)
        self.ram[ADDR_PAGE_X] = (x // 256) % 256
        self.ram[ADDR_SUB_X] = x % 256
        self.ram[ADDR_PLAYER_STATE] = 0x06 if self.dead else self.player_state
        self.ram[ADDR_AIRBORNE] = self.airborne_state
        self.ram[ADDR_SUB_Y] = int(self.y) % 256
        self.ram[ADDR_PAGE_Y] = int(self.y) // 256
        self.ram[ADDR_OPER_MODE] = 0x01 if self.completed else 0x00

    def _render(self) -> np.ndarray:
        """Deterministic frame: ground, obstacles and Mario translated by progress."""
        frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        frame[:, :] = (108, 148, 252)          # sky
        frame[GROUND_Y:, :] = (140, 88, 60)    # ground
        cam = int(self.x) - 100

        for obstacle_x in OBSTACLE_XS:
            screen_x = obstacle_x - cam
            if -20 <= screen_x <= FRAME_WIDTH:
                left = max(0, screen_x)
                right = min(FRAME_WIDTH, screen_x + 20)
                if left < right:
                    frame[GROUND_Y - 40:GROUND_Y, left:right] = (40, 40, 200)

        for start, end in PIT_SEGMENTS:
            screen_left = max(0, start - cam)
            screen_right = min(FRAME_WIDTH, end - cam)
            if screen_left < screen_right:
                frame[GROUND_Y:, screen_left:screen_right] = (0, 0, 0)

        mario_x = int(self.x) - cam
        left = max(0, min(FRAME_WIDTH - 1, mario_x))
        right = min(FRAME_WIDTH, left + 12)
        top = max(0, int(self.y) - 24)
        frame[top:top + 24, left:right] = (220, 40, 40)
        return frame
