import numpy as np
import torch
from typing import Tuple, Dict, Any

class MarioRAMTracker:
    """
    Extracts Mario's state (horizontal position, death, level completion) from Super Mario Bros RAM.

    RAM Addresses for NES Super Mario Bros:
    - 0x06D0: Level page (each page = 256 pixels)
    - 0x0086: Screen X position within current page (0-255)
    - 0x000E: Player state (0x0B or 0x06 = Dying / Dead)
    - 0x001D: Game mode / screen state
    """
    def __init__(self):
        self.max_x_pos = 0
        self.last_x_pos = 0
        self.stagnant_steps = 0

    def reset(self):
        self.max_x_pos = 0
        self.last_x_pos = 0
        self.stagnant_steps = 0

    def get_x_pos(self, ram: np.ndarray) -> int:
        page = int(ram[0x06D0]) if len(ram) > 0x06D0 else 0
        sub_x = int(ram[0x0086]) if len(ram) > 0x0086 else 0
        return page * 256 + sub_x

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
            info: tracking details
        """
        x_pos = self.get_x_pos(ram)
        d_pam = 0.0
        d_ppl1 = 0.0

        # Check forward progress
        if x_pos > self.max_x_pos:
            progress_delta = x_pos - self.max_x_pos
            d_pam = min(1.0, progress_delta / 10.0)
            self.max_x_pos = x_pos
            self.stagnant_steps = 0
        else:
            self.stagnant_steps += 1
            if self.stagnant_steps > 60:  # Stagnant for ~1 second (60 fps)
                d_ppl1 += 0.05

        # Check death or termination
        if self.is_dead(ram) or terminated:
            d_ppl1 = 1.0

        self.last_x_pos = x_pos

        info = {
            'x_pos': x_pos,
            'max_x_pos': self.max_x_pos,
            'stagnant_steps': self.stagnant_steps
        }

        return d_pam, d_ppl1, info
