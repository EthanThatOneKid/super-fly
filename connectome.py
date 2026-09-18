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
    4-Layer Drosophila Connectome SNN:
    - Layer 1: Sensory Ommatidia Input (~800 ommatidia x 5 channels = 3920 units)
    - Layer 2: Optic Lobe / Medulla / Lobula (Motion & Edge Neuropils)
    - Layer 3: Central Complex / Mushroom Body (Recurrent Interneurons)
    - Layer 4: Thoracic Motor Ganglion (Action Output Neurons: NOOP, RIGHT, JUMP, RIGHT+JUMP)
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

    def reset_state(self):
        """Reset internal state of all LIF layers."""
        self.layer1_2.reset_state()
        self.layer2_3.reset_state()
        self.layer3_4.reset_state()

    def reset_homeostasis(self):
        """Reset threshold values and firing rate traces across all layers."""
        self.layer1_2.reset_homeostasis()
        self.layer2_3.reset_homeostasis()
        self.layer3_4.reset_homeostasis()

    def forward(self, sensory_spikes: torch.Tensor):
        """
        Process single timestep sensory spikes.
        Returns:
            motor_spikes: Tensor of shape (4,) containing spikes for actions.
            layer_activations: dict containing spiking activity across layers for visualization.
        """
        optic_spikes = self.layer1_2(sensory_spikes)
        central_spikes = self.layer2_3(optic_spikes)
        motor_spikes = self.layer3_4(central_spikes)

        layer_activations = {
            'ommatidia': sensory_spikes,
            'optic_lobe': optic_spikes,
            'central_complex': central_spikes,
            'motor_ganglion': motor_spikes
        }

        return motor_spikes, layer_activations
