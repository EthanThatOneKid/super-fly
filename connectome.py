import torch
import torch.nn as nn
import numpy as np

class LIFNeuronLayer(nn.Module):
    r"""
    Leaky Integrate-and-Fire (LIF) Neuron Layer with support for eligibility traces
    and dynamic homeostatic intrinsic plasticity.
    V[t] = \alpha V[t-1] + I[t] - S[t-1] V_{reset}
    S[t] = \mathbb{I}(V[t] \ge V_{thresh})
    """
    def __init__(self, in_features: int, out_features: int,
                 tau_m: float = 20.0, v_thresh: float = 1.0,
                 v_reset: float = 0.0, dt: float = 1.0,
                 target_rate: float = 0.05, eta_homeo: float = 0.001,
                 gamma_homeo: float = 0.01, v_thresh_min: float = 0.2,
                 v_thresh_max: float = 5.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Synaptic weights
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * (2.0 / in_features)**0.5)

        # LIF parameters
        self.alpha = float(np.exp(-dt / tau_m))
        self.v_thresh_init = float(v_thresh)
        self.v_reset = v_reset
        self.current_gain = 1.0

        # Homeostatic intrinsic plasticity parameters
        self.target_rate = target_rate
        self.eta_homeo = eta_homeo
        self.gamma_homeo = gamma_homeo
        self.v_thresh_min = v_thresh_min
        self.v_thresh_max = v_thresh_max

        # Dynamic states
        self.register_buffer('v', torch.zeros(out_features))
        self.register_buffer('spikes', torch.zeros(out_features))
        self.register_buffer('v_thresh', torch.full((out_features,), self.v_thresh_init))
        self.register_buffer('rate_trace', torch.zeros(out_features))

        # Eligibility traces for pre and post spikes (for 3-factor STDP)
        self.register_buffer('trace_pre', torch.zeros(in_features))
        self.register_buffer('trace_post', torch.zeros(out_features))

        self.tau_trace = 20.0
        self.decay_trace = float(np.exp(-dt / self.tau_trace))

    def reset_state(self):
        """Reset membrane potential, eligibility traces, and firing rate trace."""
        self.v.zero_()
        self.spikes.zero_()
        self.trace_pre.zero_()
        self.trace_post.zero_()
        self.rate_trace.zero_()

    def reset_homeostasis(self):
        """Reset dynamic thresholds and firing rate trace back to initial values."""
        self.v_thresh.fill_(self.v_thresh_init)
        self.rate_trace.zero_()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        v_thresh_key = prefix + 'v_thresh'
        if v_thresh_key in state_dict:
            val = state_dict[v_thresh_key]
            if val.dim() == 0:  # scalar tensor from older state dicts
                state_dict[v_thresh_key] = torch.full((self.out_features,), val.item(), device=val.device)
        else:
            state_dict[v_thresh_key] = torch.full((self.out_features,), self.v_thresh_init)

        rate_trace_key = prefix + 'rate_trace'
        if rate_trace_key not in state_dict:
            state_dict[rate_trace_key] = torch.zeros(self.out_features)

        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def forward(self, input_spikes: torch.Tensor) -> torch.Tensor:
        """
        Forward step for 1D spike tensor of shape (in_features,).
        Returns output spike tensor of shape (out_features,).
        """
        # Synaptic current
        current = torch.matmul(self.weight, input_spikes)
        current = current - current.mean()

        # Membrane potential integration with leak
        # Reset potential for neurons that fired in the previous step
        self.v = self.alpha * self.v * (1.0 - self.spikes) + current * self.current_gain

        # Firing condition
        self.spikes = (self.v >= self.v_thresh).float()

        # Soft / Hard reset
        self.v = torch.where(self.spikes > 0, torch.tensor(self.v_reset, device=self.v.device), self.v)

        # Update eligibility traces: trace = decay * trace + spike
        self.trace_pre = self.decay_trace * self.trace_pre + input_spikes
        self.trace_post = self.decay_trace * self.trace_post + self.spikes

        # Homeostatic intrinsic plasticity: update threshold toward target firing rate during training
        if self.training:
            self.rate_trace = (1.0 - self.gamma_homeo) * self.rate_trace + self.gamma_homeo * self.spikes
            delta_v = self.eta_homeo * (self.rate_trace - self.target_rate)
            self.v_thresh = (self.v_thresh + delta_v).clamp(self.v_thresh_min, self.v_thresh_max)

        return self.spikes


class DrosophilaConnectomeSNN(nn.Module):
    """
    4-Layer Drosophila Connectome SNN with Recurrent Central Complex -> Optic Lobe Feedback:
    - Layer 1: Sensory Ommatidia Input (~800 ommatidia x 5 channels = 3920 units)
    - Layer 2: Optic Lobe / Medulla / Lobula (Motion & Edge Neuropils)
    - Layer 3: Central Complex / Mushroom Body (Recurrent Interneurons)
    - Layer 4: Thoracic Motor Ganglion (Action Output Neurons: NOOP, RIGHT, JUMP, RIGHT+JUMP)
    - Feedback 3->2: Central Complex / Mushroom Body to Optic Lobe Recurrent Loop
    """
    def __init__(self, num_ommatidia: int = 784, channels_per_ommatidium: int = 5):
        super().__init__()

        self.num_inputs = num_ommatidia * channels_per_ommatidium
        self.num_optic_lobe = 256
        self.num_central_complex = 128
        self.num_motor_ganglion = 4  # [NOOP, RIGHT, JUMP, RIGHT+JUMP]

        # Layer 1 -> Layer 2: Sensory Ommatidia to Optic Lobe
        self.layer1_2 = LIFNeuronLayer(self.num_inputs, self.num_optic_lobe, tau_m=15.0)

        # Layer 2 -> Layer 3: Optic Lobe to Central Complex / Mushroom Body
        self.layer2_3 = LIFNeuronLayer(self.num_optic_lobe, self.num_central_complex, tau_m=20.0)
        self.layer2_3.current_gain = 6.0

        # Layer 3 -> Layer 4: Central Complex to Thoracic Motor Ganglion
        self.layer3_4 = LIFNeuronLayer(self.num_central_complex, self.num_motor_ganglion, tau_m=25.0)
        self.layer3_4.current_gain = 6.0

        # Recurrent Feedback 3 -> 2: Central Complex to Optic Lobe
        self.feedback_3_2 = LIFNeuronLayer(self.num_central_complex, self.num_optic_lobe, tau_m=20.0)
        self.register_buffer('recurrent_central_spikes', torch.zeros(self.num_central_complex))

        self.initialize_weights()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        recurrent_key = prefix + 'recurrent_central_spikes'
        if recurrent_key not in state_dict:
            state_dict[recurrent_key] = torch.zeros(self.num_central_complex)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def initialize_weights(self):
        """
        Pre-training weight initialization heuristics for Drosophila connectome layers.
        Structures sensory feature gain, optic-central pathways, and motor ganglion jump excitation
        while maintaining zero-mean row alignment across weight matrices.
        """
        with torch.no_grad():
            # Layer 1->2: Sensory Ommatidia to Optic Lobe
            # Shape: (num_optic_lobe=256, num_inputs=3920)
            w1 = torch.randn(self.num_optic_lobe, self.num_inputs) * (2.0 / self.num_inputs)**0.5
            # Apply feature-gain weighting for Canny edges (ch 0) and RIGHT motion (ch 1)
            # Channel ordering per ommatidium: 0=edges, 1=vx_right, 2=vx_left, 3=vy_down, 4=vy_up
            reshaped_w1 = w1.view(self.num_optic_lobe, self.num_inputs // 5, 5)
            reshaped_w1[:, :, 0] *= 1.25  # Edge sensitivity
            reshaped_w1[:, :, 1] *= 1.25  # Right motion sensitivity
            w1 = reshaped_w1.view(self.num_optic_lobe, self.num_inputs)
            w1 -= w1.mean(dim=1, keepdim=True)
            self.layer1_2.weight.copy_(w1)

            # Layer 2->3: Optic Lobe to Central Complex / Mushroom Body
            # Shape: (num_central_complex=128, num_optic_lobe=256)
            w2 = torch.randn(self.num_central_complex, self.num_optic_lobe) * (2.0 / self.num_optic_lobe)**0.5
            w2 -= w2.mean(dim=1, keepdim=True)
            self.layer2_3.weight.copy_(w2)

            # Layer 3->4: Central Complex to Thoracic Motor Ganglion
            # Shape: (num_motor_ganglion=4, num_central_complex=128)
            # Actions: [0: NOOP, 1: RIGHT, 2: JUMP, 3: RIGHT+JUMP]
            w3 = torch.randn(self.num_motor_ganglion, self.num_central_complex) * (2.0 / self.num_central_complex)**0.5
            half_cc = self.num_central_complex // 2
            # Structure central complex interneuron pathways to excite movement and jump actions:
            w3[0, :half_cc] -= 0.30  # NOOP suppression on primary pathways
            w3[0, half_cc:] += 0.30
            w3[1, :half_cc] += 0.15  # RIGHT motor excitation on primary pathways
            w3[1, half_cc:] -= 0.15
            w3[2, :half_cc] += 0.20  # JUMP motor excitation on primary pathways
            w3[2, half_cc:] -= 0.20
            w3[3, :half_cc] += 0.25  # RIGHT+JUMP motor excitation on primary pathways
            w3[3, half_cc:] -= 0.25
            w3 -= w3.mean(dim=1, keepdim=True)
            self.layer3_4.weight.copy_(w3)

            # Feedback 3->2: Central Complex to Optic Lobe
            # Shape: (num_optic_lobe=256, num_central_complex=128)
            w_fb = torch.randn(self.num_optic_lobe, self.num_central_complex) * (2.0 / self.num_central_complex)**0.5
            w_fb -= w_fb.mean(dim=1, keepdim=True)
            self.feedback_3_2.weight.copy_(w_fb)

    def reset_state(self):
        """Reset internal state of all LIF layers and recurrent buffers."""
        self.layer1_2.reset_state()
        self.layer2_3.reset_state()
        self.layer3_4.reset_state()
        self.feedback_3_2.reset_state()
        self.recurrent_central_spikes.zero_()

    def reset_homeostasis(self):
        """Reset threshold values and firing rate traces across all layers."""
        self.layer1_2.reset_homeostasis()
        self.layer2_3.reset_homeostasis()
        self.layer3_4.reset_homeostasis()
        self.feedback_3_2.reset_homeostasis()

    def forward(self, sensory_spikes: torch.Tensor):
        """
        Process single timestep sensory spikes with recurrent feedback.
        Returns:
            motor_spikes: Tensor of shape (4,) containing spikes for actions.
            layer_activations: dict containing spiking activity across layers for visualization.
        """
        # Process recurrent feedback from Central Complex spikes from previous timestep
        fb_spikes = self.feedback_3_2(self.recurrent_central_spikes)

        # Optic Lobe integrates sensory input and recurrent feedback spikes
        sensory_optic_spikes = self.layer1_2(sensory_spikes)
        optic_spikes = torch.clamp(sensory_optic_spikes + fb_spikes, 0.0, 1.0)

        central_spikes = self.layer2_3(optic_spikes)
        motor_spikes = self.layer3_4(central_spikes)

        # Update recurrent state buffer with current timestep Central Complex activity
        self.recurrent_central_spikes.copy_(central_spikes.detach())

        layer_activations = {
            'ommatidia': sensory_spikes,
            'optic_lobe': optic_spikes,
            'central_complex': central_spikes,
            'motor_ganglion': motor_spikes,
            'feedback_3_2': fb_spikes
        }

        return motor_spikes, layer_activations
