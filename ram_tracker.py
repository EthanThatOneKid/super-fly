import numpy as np
import torch
from typing import Tuple, Dict, Any

class MarioRAMTracker:
    """
    Extracts Mario's state (horizontal position, death, level completion, airborne state, velocity)
    from Super Mario Bros RAM to compute dual-pathway dopamine signals (PAM reward, PPL1 punishment).

    RAM Addresses for NES Super Mario Bros:
    - 0x006D: Level page (each page = 256 pixels)
    - 0x0086: Screen X position within current page (0-255)
    - 0x00CE: Screen Y position (0-255, higher is lower on screen)
    - 0x00B5: Page Y position (0 for normal ground/air, >0 when falling off screen into pits)
    - 0x000E: Player state (0x06 or 0x0B = Dying / Dead)
    - 0x001D: Player vertical/airborne state (0 = on ground, 1 = jumping, 2 = falling)
    """
    def __init__(self):
        self.max_x_pos = 0
        self.last_x_pos = 0
        self.max_sub_page = 0
        self.max_page = 0
        self.stagnant_steps = 0
        self.last_speed = 0
        self.in_air = False
        self.jump_start_x = 0
        self.cleared_obstacle = False

    def reset(self):
        self.max_x_pos = 0
        self.last_x_pos = 0
        self.max_sub_page = 0
        self.max_page = 0
        self.stagnant_steps = 0
        self.last_speed = 0
        self.in_air = False
        self.jump_start_x = 0
        self.cleared_obstacle = False

    def get_x_pos(self, ram: np.ndarray) -> int:
        page = int(ram[0x006D]) if len(ram) > 0x006D else 0
        sub_x = int(ram[0x0086]) if len(ram) > 0x0086 else 0
        return page * 256 + sub_x

    def get_y_pos(self, ram: np.ndarray) -> int:
        y_page = int(ram[0x00B5]) if len(ram) > 0x00B5 else 0
        sub_y = int(ram[0x00CE]) if len(ram) > 0x00CE else 0
        return y_page * 256 + sub_y

    def is_airborne(self, ram: np.ndarray) -> bool:
        if len(ram) > 0x001D:
            player_air_state = int(ram[0x001D])
            if player_air_state in (1, 2):
                return True
        return False

    def is_dead(self, ram: np.ndarray) -> bool:
        if len(ram) > 0x000E:
            player_state = int(ram[0x000E])
            if player_state in (0x06, 0x0B):
                return True
        return False

    def compute_dopamine(self, ram: np.ndarray, terminated: bool, truncated: bool) -> Tuple[float, float, Dict[str, Any]]:
        """
        Computes PAM (positive reward) and PPL1 (aversive punishment) dopamine signals.
        Returns:
            d_pam: PAM positive dopamine scalar
            d_ppl1: PPL1 aversive dopamine scalar
            info: tracking details including dopamine sub-type breakdowns
        """
        x_pos = self.get_x_pos(ram)
        airborne = self.is_airborne(ram)
        d_pam_progress = 0.0
        d_pam_obstacle = 0.0
        d_ppl1_stagnation = 0.0
        d_ppl1_collision = 0.0
        d_ppl1_death = 0.0

        current_speed = x_pos - self.last_x_pos if self.last_x_pos > 0 else 0

        # 1. Forward progress & Milestone rewards (PAM)
        if x_pos > self.max_x_pos:
            progress_delta = x_pos - self.max_x_pos
            d_pam_progress = min(1.0, progress_delta / 10.0)

            # Sub-page milestone check (every 64 pixels)
            sub_page = x_pos // 64
            if sub_page > self.max_sub_page:
                d_pam_progress = min(1.0, d_pam_progress + 0.3)
                self.max_sub_page = sub_page

            # Level page milestone check (every 256 pixels)
            page = x_pos // 256
            if page > self.max_page:
                d_pam_progress = min(1.0, d_pam_progress + 0.5)
                self.max_page = page

            self.max_x_pos = x_pos
            self.stagnant_steps = 0
        else:
            self.stagnant_steps += 1
            if self.stagnant_steps > 60:  # Stagnant for ~1 second (60 fps)
                d_ppl1_stagnation += 0.05

        # 2. Airborne / Jump Obstacle Clearance & Mid-jump Stagnation
        if airborne:
            if not self.in_air:
                self.in_air = True
                self.jump_start_x = self.last_x_pos if self.last_x_pos > 0 else x_pos
                self.cleared_obstacle = False

            # Mid-jump stagnation check (airborne with zero or negative forward speed)
            if current_speed <= 0:
                d_ppl1_stagnation = min(1.0, d_ppl1_stagnation + 0.2)

            # Obstacle/pipe clearance mid-jump
            air_distance = x_pos - self.jump_start_x
            if air_distance >= 24 and not self.cleared_obstacle:
                d_pam_obstacle = min(1.0, d_pam_obstacle + 0.4)
                self.cleared_obstacle = True
        else:
            if self.in_air:
                # Just landed from jump
                air_distance = x_pos - self.jump_start_x
                if air_distance >= 24 and not self.cleared_obstacle:
                    d_pam_obstacle = min(1.0, d_pam_obstacle + 0.4)
                self.in_air = False

        # 3. Collision-induced speed drop check (PPL1)
        # Bumping into wall or pipe while moving forward on ground
        if not airborne and self.last_speed >= 2 and current_speed <= 0 and not self.is_dead(ram) and not terminated:
            d_ppl1_collision = min(1.0, d_ppl1_collision + 0.3)

        # 4. Death or Termination (PPL1)
        if self.is_dead(ram) or terminated:
            d_ppl1_death = 1.0

        self.last_speed = current_speed
        self.last_x_pos = x_pos

        d_pam = float(np.clip(d_pam_progress + d_pam_obstacle, 0.0, 1.0))
        d_ppl1 = float(np.clip(d_ppl1_stagnation + d_ppl1_collision + d_ppl1_death, 0.0, 1.0))

        info = {
            'x_pos': x_pos,
            'max_x_pos': self.max_x_pos,
            'stagnant_steps': self.stagnant_steps,
            'sub_page': x_pos // 64,
            'max_sub_page': self.max_sub_page,
            'page': x_pos // 256,
            'max_page': self.max_page,
            'is_airborne': airborne,
            'current_speed': current_speed,
            'dopamine_breakdown': {
                'progress': float(np.clip(d_pam_progress, 0.0, 1.0)),
                'obstacle_clearance': float(np.clip(d_pam_obstacle, 0.0, 1.0)),
                'stagnation': float(np.clip(d_ppl1_stagnation, 0.0, 1.0)),
                'collision': float(np.clip(d_ppl1_collision, 0.0, 1.0)),
                'death': float(np.clip(d_ppl1_death, 0.0, 1.0)),
            }
        }

        return d_pam, d_ppl1, info
