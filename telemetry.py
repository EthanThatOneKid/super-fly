import cv2
import numpy as np
import torch
from typing import Dict, Any

class DrosophilaTelemetryOverlay:
    """
    Renders layer-by-layer fly SNN spike activity heatmaps and dopamine telemetry
    side-by-side with the NES Super Mario Bros gameplay frame.
    """
    def __init__(self, width: int = 1280, height: int = 720):
        self.width = width
        self.height = height

    def render_overlay(self, obs_frame: np.ndarray,
                       layer_activations: Dict[str, torch.Tensor],
                       d_pam: float, d_ppl1: float,
                       info: Dict[str, Any]) -> np.ndarray:
        """
        Constructs side-by-side telemetry view:
        Left panel: Mario NES Game Screen
        Right panel: 4-Layer Fly Brain Spike Heatmaps & Dopamine Gauges
        """
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # 1. Game Frame (Resize NES 256x240 to 600x560)
        game_w, game_h = 600, 560
        game_bgr = cv2.cvtColor(obs_frame, cv2.COLOR_RGB2BGR)
        game_resized = cv2.resize(game_bgr, (game_w, game_h), interpolation=cv2.INTER_NEAREST)

        canvas[40:40+game_h, 30:30+game_w] = game_resized

        # Game Header
        cv2.putText(canvas, "Drosophila Melanogaster SNN - NES SMB 1-1", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 2. Right Panel: SNN Brain Layer Visualizations
        right_x = 660

        # Layer 1: Ommatidia Sensory Grid (28x28)
        ommatidia_spikes = layer_activations['ommatidia'].detach().cpu().numpy()
        # Reshape or sum across channels for 28x28 spatial visualization
        if len(ommatidia_spikes) == 3920:
            grid = ommatidia_spikes.reshape(28, 28, 5).sum(axis=-1)
            grid = np.clip(grid * 80, 0, 255).astype(np.uint8)
        else:
            grid = np.zeros((28, 28), dtype=np.uint8)

        grid_colored = cv2.applyColorMap(grid, cv2.COLORMAP_JET)
        grid_resized = cv2.resize(grid_colored, (150, 150), interpolation=cv2.INTER_NEAREST)

        canvas[70:220, right_x:right_x+150] = grid_resized
        cv2.putText(canvas, "Layer 1: Ommatidia Grid", (right_x, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Layer 2: Optic Lobe / Medulla Heatmap (256 units -> 16x16)
        optic_spikes = layer_activations['optic_lobe'].detach().cpu().numpy()
        optic_grid = (optic_spikes.reshape(16, 16) * 255).astype(np.uint8)
        optic_colored = cv2.applyColorMap(optic_grid, cv2.COLORMAP_HOT)
        optic_resized = cv2.resize(optic_colored, (150, 150), interpolation=cv2.INTER_NEAREST)

        canvas[70:220, right_x+180:right_x+330] = optic_resized
        cv2.putText(canvas, "Layer 2: Optic Lobe", (right_x+180, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Layer 3: Central Complex (128 units -> 8x16)
        central_spikes = layer_activations['central_complex'].detach().cpu().numpy()
        central_grid = (central_spikes.reshape(8, 16) * 255).astype(np.uint8)
        central_colored = cv2.applyColorMap(central_grid, cv2.COLORMAP_VIRIDIS)
        central_resized = cv2.resize(central_colored, (330, 80), interpolation=cv2.INTER_NEAREST)

        canvas[260:340, right_x:right_x+330] = central_resized
        cv2.putText(canvas, "Layer 3: Central Complex / Mushroom Body", (right_x, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Layer 4: Motor Output Neurons
        motor_spikes = layer_activations['motor_ganglion'].detach().cpu().numpy()
        actions = ["NOOP", "RIGHT", "JUMP", "R+JUMP"]
        cv2.putText(canvas, "Layer 4: Thoracic Motor Output", (right_x, 370),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        for i, (act, spk) in enumerate(zip(actions, motor_spikes)):
            color = (0, 255, 0) if spk > 0 else (100, 100, 100)
            cv2.rectangle(canvas, (right_x + i*80, 385), (right_x + i*80 + 70, 420), color, -1)
            cv2.putText(canvas, act, (right_x + i*80 + 5, 410),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0) if spk > 0 else (255, 255, 255), 1)

        # 3. Dual Dopamine Pathway Gauges
        cv2.putText(canvas, "Dual Dopamine Pathways", (right_x, 460),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # PAM (Positive Progress) Bar
        pam_bar_w = int(np.clip(d_pam, 0.0, 1.0) * 300)
        cv2.rectangle(canvas, (right_x, 480), (right_x + 300, 500), (50, 50, 50), -1)
        cv2.rectangle(canvas, (right_x, 480), (right_x + pam_bar_w, 500), (0, 255, 0), -1)
        cv2.putText(canvas, f"PAM (Reward): {d_pam:.2f}", (right_x + 310, 495),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        # PPL1 (Aversion / Death) Bar
        ppl1_bar_w = int(np.clip(d_ppl1, 0.0, 1.0) * 300)
        cv2.rectangle(canvas, (right_x, 520), (right_x + 300, 540), (50, 50, 50), -1)
        cv2.rectangle(canvas, (right_x, 520), (right_x + ppl1_bar_w, 540), (0, 0, 255), -1)
        cv2.putText(canvas, f"PPL1 (Aversion): {d_ppl1:.2f}", (right_x + 310, 535),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

        # Telemetry Text
        cv2.putText(canvas, f"Horizontal Progress X: {info.get('x_pos', 0)}", (right_x, 580),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(canvas, f"Max X Reached: {info.get('max_x_pos', 0)}", (right_x, 605),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        return canvas
