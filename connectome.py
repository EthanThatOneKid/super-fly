import torch
import torch.nn as nn
import numpy as np


class FastSigmoidSpike(torch.autograd.Function):
    r"""
    FastSigmoid surrogate gradient function for spike generation during backpropagation.
    Forward: $S = \mathbb{I}(V \ge V_{\text{thresh}})$
    Backward: $\frac{\partial S}{\partial V} = \frac{1}{(1 + \beta |V - V_{\text{thresh}}|)^2}$
    """
    @staticmethod
    def forward(ctx, v: torch.Tensor, v_thresh: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(v, v_thresh)
        return (v >= v_thresh).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        v, v_thresh = ctx.saved_tensors
        beta = 5.0
        grad_v = grad_output / (1.0 + beta * torch.abs(v - v_thresh)).pow(2)
        return grad_v, None


def surrogate_spike(v: torch.Tensor, v_thresh: torch.Tensor) -> torch.Tensor:
    return FastSigmoidSpike.apply(v, v_thresh)


class LIFNeuronLayer(nn.Module):
    r"""
    Leaky Integrate-and-Fire (LIF) Neuron Layer with support for eligibility traces,
    dynamic homeostatic intrinsic plasticity, and surrogate gradient sequence learning.
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
        current = torch.matmul(self.weight, input_spikes)
        current = current - current.mean()

        if torch.is_grad_enabled():
            v_next = self.alpha * self.v * (1.0 - self.spikes) + current * self.current_gain
            spikes_next = surrogate_spike(v_next, self.v_thresh)
            v_after_reset = torch.where(spikes_next > 0, torch.tensor(self.v_reset, device=v_next.device), v_next)
            self.v = v_after_reset
            self.spikes = spikes_next
        else:
            self.v = self.alpha * self.v * (1.0 - self.spikes) + current * self.current_gain
            self.spikes = (self.v >= self.v_thresh).float()
            self.v = torch.where(self.spikes > 0, torch.tensor(self.v_reset, device=self.v.device), self.v)

        # Update eligibility traces
        self.trace_pre = self.decay_trace * self.trace_pre + input_spikes.detach()
        self.trace_post = self.decay_trace * self.trace_post + self.spikes.detach()

        # Homeostatic intrinsic plasticity: update threshold toward target firing rate during training
        if self.training:
            self.rate_trace = (1.0 - self.gamma_homeo) * self.rate_trace + self.gamma_homeo * self.spikes.detach()
            delta_v = self.eta_homeo * (self.rate_trace - self.target_rate)
            self.v_thresh = (self.v_thresh + delta_v).clamp(self.v_thresh_min, self.v_thresh_max)

        return self.spikes


class TemporalMotorDecoder(nn.Module):
    """
    Temporal Motor Decoder / Readout layer over temporal Drosophila SNN state:
    Decodes action decisions (NOOP, RIGHT, JUMP, RIGHT+JUMP) from Central Complex
    interneuron activity, Thoracic Motor Ganglion output spikes, and recurrent dynamics.
    """
    def __init__(self, cc_dim: int = 128, motor_dim: int = 4, hidden_dim: int = 64, num_actions: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(cc_dim + motor_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_actions)

    def forward(self, cc_spikes: torch.Tensor, motor_spikes: torch.Tensor) -> torch.Tensor:
        state = torch.cat([cc_spikes, motor_spikes], dim=-1)
        return self.fc2(self.relu(self.fc1(state)))


class DrosophilaConnectomeSNN(nn.Module):
    """
    4-Layer Drosophila Connectome SNN with Recurrent Central Complex -> Optic Lobe Feedback
    and Temporal Motor Decoder:
    - Layer 1: Sensory Ommatidia Input (~800 ommatidia x 5 channels = 3920 units)
    - Layer 2: Optic Lobe / Medulla / Lobula (Motion & Edge Neuropils)
    - Layer 3: Central Complex / Mushroom Body (Recurrent Interneurons)
    - Layer 4: Thoracic Motor Ganglion (Action Output Neurons: NOOP, RIGHT, JUMP, RIGHT+JUMP)
    - Feedback 3->2: Central Complex / Mushroom Body to Optic Lobe Recurrent Loop
    - Temporal Motor Decoder: Action readout layer over temporal SNN states
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

        # Temporal Motor Decoder Readout
        self.decoder = TemporalMotorDecoder(
            cc_dim=self.num_central_complex,
            motor_dim=self.num_motor_ganglion,
            hidden_dim=64,
            num_actions=4,
        )

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
            w1 = torch.randn(self.num_optic_lobe, self.num_inputs) * (2.0 / self.num_inputs)**0.5
            reshaped_w1 = w1.view(self.num_optic_lobe, self.num_inputs // 5, 5)
            reshaped_w1[:, :, 0] *= 1.25  # Edge sensitivity
            reshaped_w1[:, :, 1] *= 1.25  # Right motion sensitivity
            w1 = reshaped_w1.view(self.num_optic_lobe, self.num_inputs)
            w1 -= w1.mean(dim=1, keepdim=True)
            self.layer1_2.weight.copy_(w1)

            w2 = torch.randn(self.num_central_complex, self.num_optic_lobe) * (2.0 / self.num_optic_lobe)**0.5
            w2 -= w2.mean(dim=1, keepdim=True)
            self.layer2_3.weight.copy_(w2)

            w3 = torch.randn(self.num_motor_ganglion, self.num_central_complex) * (2.0 / self.num_central_complex)**0.5
            half_cc = self.num_central_complex // 2
            w3[0, :half_cc] -= 0.30
            w3[0, half_cc:] += 0.30
            w3[1, :half_cc] += 0.15
            w3[1, half_cc:] -= 0.15
            w3[2, :half_cc] += 0.20
            w3[2, half_cc:] -= 0.20
            w3[3, :half_cc] += 0.25
            w3[3, half_cc:] -= 0.25
            w3 -= w3.mean(dim=1, keepdim=True)
            self.layer3_4.weight.copy_(w3)

            w_fb = torch.randn(self.num_optic_lobe, self.num_central_complex) * (2.0 / self.num_central_complex)**0.5
            w_fb -= w_fb.mean(dim=1, keepdim=True)
            self.feedback_3_2.weight.copy_(w_fb)

    def enforce_bio_constraints(self):
        """Apply zero-mean row centering and [-3.0, 3.0] clamping on SNN weight matrices."""
        with torch.no_grad():
            for layer in (self.layer1_2, self.layer2_3, self.layer3_4, self.feedback_3_2):
                layer.weight.sub_(layer.weight.mean(dim=1, keepdim=True))
                layer.weight.clamp_(-3.0, 3.0)

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
        fb_spikes = self.feedback_3_2(self.recurrent_central_spikes)

        sensory_optic_spikes = self.layer1_2(sensory_spikes)
        optic_spikes = torch.clamp(sensory_optic_spikes + fb_spikes, 0.0, 1.0)

        central_spikes = self.layer2_3(optic_spikes)
        motor_spikes = self.layer3_4(central_spikes)

        self.recurrent_central_spikes.copy_(central_spikes.detach())

        layer_activations = {
            'ommatidia': sensory_spikes,
            'optic_lobe': optic_spikes,
            'central_complex': central_spikes,
            'motor_ganglion': motor_spikes,
            'feedback_3_2': fb_spikes
        }

        return motor_spikes, layer_activations

    def decode_action(self, central_spikes: torch.Tensor, motor_spikes: torch.Tensor) -> torch.Tensor:
        """Decode action logits from Central Complex and Thoracic Motor Ganglion states."""
        return self.decoder(central_spikes, motor_spikes)
