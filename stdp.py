import torch
import torch.nn as nn
from connectome import LIFNeuronLayer, DrosophilaConnectomeSNN

class DualDopamineSTDP:
    r"""
    Drosophila dual-pathway dopamine-modulated STDP with Inverted LTD learning rule.

    Pathways:
    - PAM Cluster (Reward): Dopamine signal $D_{\text{PAM}}$ released upon positive forward progress,
      obstacle clearance, or milestone completion.
    - PPL1 Cluster (Punishment/Aversion): Dopamine signal $D_{\text{PPL1}}$ released upon death,
      stagnation, collision speed drops, or mid-jump stagnation.

    Inverted Update Sign:
    $\Delta w_{ij} = -\eta \cdot D(t) \cdot E_{ij}(t)$
    where $D(t) = D_{\text{PAM}}(t) - D_{\text{PPL1}}(t)$ and $E_{ij}(t) = \text{trace}_{\text{post}} \otimes \text{trace}_{\text{pre}}$
    is the synaptic eligibility trace.
    Inverting the sign depresses Mushroom Body Output Neurons (MBONs) driving avoidance/inappropriate actions,
    aligning with biological Drosophila learning mechanisms.
    """
    def __init__(self, model: DrosophilaConnectomeSNN, lr: float = 0.005, w_min: float = -3.0, w_max: float = 3.0):
        self.model = model
        self.lr = lr
        self.w_min = w_min
        self.w_max = w_max
        self.last_dopamine_signal = 0.0

    def step(self, d_pam: float, d_ppl1: float):
        """
        Applies dopamine-modulated STDP weight update to trainable LIF layers in the SNN.
        Args:
            d_pam: Positive dopamine scalar [0, 1] from forward progress / milestone / clearance.
            d_ppl1: Aversive dopamine scalar [0, 1] from death / stagnation / collision / mid-jump stall.
        """
        # Net dopamine modulation D(t)
        dopamine_signal = float(d_pam - d_ppl1)
        self.last_dopamine_signal = dopamine_signal

        if abs(dopamine_signal) < 1e-6:
            return

        layers = [self.model.layer1_2, self.model.layer2_3, self.model.layer3_4, self.model.feedback_3_2]

        with torch.no_grad():
            for layer in layers:
                # Eligibility trace E_ij = post_trace (out) x pre_trace (in)
                eligibility = torch.outer(layer.trace_post, layer.trace_pre)

                # Inverted update sign: \Delta w = - \eta * D(t) * E_ij
                delta_w = - self.lr * dopamine_signal * eligibility

                # Update weights and clip
                layer.weight.add_(delta_w)
                layer.weight.clamp_(self.w_min, self.w_max)
                layer.weight -= layer.weight.mean(dim=1, keepdim=True)
                layer.weight.clamp_(self.w_min, self.w_max)
