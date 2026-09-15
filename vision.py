import cv2
import numpy as np
import torch

class OmmatidiaVisionPreprocessor:
    """
    Preprocesses NES emulator frames into Drosophila ommatidial sensory signals.
    Employs Farneback optical flow for motion vector extraction and Canny edge
    detection, resampling onto a spatial ommatidia grid (~800 channels), and
    converting continuous intensities into Poisson spike probabilities.
    """
    def __init__(self, grid_h=28, grid_w=28, canny_thresh1=50, canny_thresh2=150):
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.num_ommatidia = grid_h * grid_w  # e.g., 28x28 = 784 ommatidia (~800)
        self.canny_thresh1 = canny_thresh1
        self.canny_thresh2 = canny_thresh2

        self.prev_gray = None

        # Channels per ommatidium:
        # 0: Canny Edge intensity
        # 1: Horizontal Motion Right (vx > 0)
        # 2: Horizontal Motion Left (vx < 0)
        # 3: Vertical Motion Down (vy > 0)
        # 4: Vertical Motion Up (vy < 0)
        self.num_channels = 5
        self.total_inputs = self.num_ommatidia * self.num_channels

    def reset(self):
        """Reset temporal state between episodes."""
        self.prev_gray = None

    def process_frame(self, frame_rgb: np.ndarray):
        """
        Processes an RGB NES frame (256x240x3) and computes ommatidial feature values.
        Returns:
            features: np.ndarray of shape (num_ommatidia, num_channels) with normalized values [0, 1]
            debug_info: dict containing raw intermediate images for visualization
        """
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

        # 1. Canny Edge Detection
        edges = cv2.Canny(gray, self.canny_thresh1, self.canny_thresh2) / 255.0

        # 2. Farneback Optical Flow
        if self.prev_gray is None:
            flow_x = np.zeros_like(gray, dtype=np.float32)
            flow_y = np.zeros_like(gray, dtype=np.float32)
        else:
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            flow_x = flow[..., 0]
            flow_y = flow[..., 1]

        self.prev_gray = gray

        # Separate motion directional components and apply soft non-linear scaling
        vx_right = np.maximum(0, flow_x)
        vx_left = np.maximum(0, -flow_x)
        vy_down = np.maximum(0, flow_y)
        vy_up = np.maximum(0, -flow_y)

        # Normalize motion channels
        vx_right = np.clip(vx_right / 10.0, 0.0, 1.0)
        vx_left = np.clip(vx_left / 10.0, 0.0, 1.0)
        vy_down = np.clip(vy_down / 10.0, 0.0, 1.0)
        vy_up = np.clip(vy_up / 10.0, 0.0, 1.0)

        # 3. Resample onto Ommatidial Grid
        edge_grid = cv2.resize(edges, (self.grid_w, self.grid_h), interpolation=cv2.INTER_AREA)
        vx_r_grid = cv2.resize(vx_right, (self.grid_w, self.grid_h), interpolation=cv2.INTER_AREA)
        vx_l_grid = cv2.resize(vx_left, (self.grid_w, self.grid_h), interpolation=cv2.INTER_AREA)
        vy_d_grid = cv2.resize(vy_down, (self.grid_w, self.grid_h), interpolation=cv2.INTER_AREA)
        vy_u_grid = cv2.resize(vy_up, (self.grid_w, self.grid_h), interpolation=cv2.INTER_AREA)

        # Stack channels: shape (grid_h, grid_w, 5)
        features_grid = np.stack([edge_grid, vx_r_grid, vx_l_grid, vy_d_grid, vy_u_grid], axis=-1)

        # Flatten to (num_ommatidia, num_channels)
        features = features_grid.reshape(-1, self.num_channels)

        debug_info = {
            'gray': gray,
            'edges': edges,
            'flow_x': flow_x,
            'flow_y': flow_y,
            'features_grid': features_grid
        }

        return features, debug_info

    def generate_poisson_spikes(self, features: np.ndarray, max_firing_prob: float = 0.8) -> torch.Tensor:
        """
        Converts ommatidial feature values [0, 1] into Poisson spike train tensor (1D vector).
        Args:
            features: np.ndarray of shape (num_ommatidia, num_channels)
            max_firing_prob: Maximum probability of a spike occurring in a time step
        Returns:
            spikes: torch.FloatTensor of shape (total_inputs,) containing 1.0 for spike, 0.0 otherwise
        """
        flat_features = features.flatten()
        probs = flat_features * max_firing_prob
        random_draws = np.random.rand(*probs.shape)
        spikes = (random_draws < probs).astype(np.float32)
        return torch.from_numpy(spikes)
