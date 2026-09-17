import torch
import torch.nn as nn
import numpy as np

class LIFNeuronLayer(nn.Module):
    r"""
    Leaky Integrate-and-Fire (LIF) Neuron Layer with support for eligibility traces.
    V[t] = \alpha V[t-1] + I[t] - S[t-1] V_{reset}
    S[t] = \mathbb{I}(V[t] \ge V_{thresh})
    """
    def __init__(self, in_features: int, out_features: int,
                 tau_m: float = 20.0, v_thresh: float = 1.0,
                 v_reset: float = 0.0, dt: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Synaptic weights
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * (2.0 / in_features)**0.5)

        # LIF parameters
        self.alpha = float(np.exp(-dt / tau_m))
        self.v_thresh = v_thresh
        self.v_reset = v_reset

        # Dynamic states
        self.register_buffer('v', torch.zeros(out_features))
        self.register_buffer('spikes', torch.zeros(out_features))

        # Eligibility traces for pre and post spikes (for 3-factor STDP)
        self.register_buffer('trace_pre', torch.zeros(in_features))
        self.register_buffer('trace_post', torch.zeros(out_features))

        self.tau_trace = 20.0
        self.decay_trace = float(np.exp(-dt / self.tau_trace))

    def reset_state(self):
        """Reset membrane potential and eligibility traces."""
        self.v.zero_()
        self.spikes.zero_()
        self.trace_pre.zero_()
        self.trace_post.zero_()

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
        self.v = self.alpha * self.v * (1.0 - self.spikes) + current

        # Firing condition
        self.spikes = (self.v >= self.v_thresh).float()

        # Soft / Hard reset
        self.v = torch.where(self.spikes > 0, torch.tensor(self.v_reset, device=self.v.device), self.v)

        # Update eligibility traces: trace = decay * trace + spike
        self.trace_pre = self.decay_trace * self.trace_pre + input_spikes
        self.trace_post = self.decay_trace * self.trace_post + self.spikes

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

        # Layer 3 -> Layer 4: Central Complex to Thoracic Motor Ganglion
        self.layer3_4 = LIFNeuronLayer(self.num_central_complex, self.num_motor_ganglion, tau_m=25.0)

    def reset_state(self):
        """Reset internal state of all LIF layers."""
        self.layer1_2.reset_state()
        self.layer2_3.reset_state()
        self.layer3_4.reset_state()

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
