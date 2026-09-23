"""How one settle window of motor activity becomes the decoder's evidence vector.

The shipped decoder was handed ``sum(motor_spikes)`` over the settle window: four integers,
one per motor neuron, each in ``[0, settle_steps]``. At the shipped window of five frames
that is six possible values per channel, so a decision threshold on top of it can only sit
between integers -- and the oracle ceiling measurement found exactly that damage. Measured on
the dev seeds at settle 5, over the same forward passes, with the boundary fitted for each
statistic (``oracle.py``):

| statistic | bounded jump recall | AUC |
| --- | --- | --- |
| ``spike_sum`` -- what shipped, and what this module still defaults to | 0.056 | 0.657 |
| ``drive_leaky_recency`` @ 0.25 | 0.160 | 0.611 |
| ``drive_sum`` -- the analog drive, flat | 0.128 | 0.640 |

The ordering is the same (AUC is unchanged to within noise, and slightly *worse*), so the
gain is not better ranking: it is that a continuous statistic has somewhere for a threshold
to sit. Rounding a five-frame window to six levels forces the boundary onto an integer, and on
this evidence that costs about two thirds of the jumps a fitted boundary could catch. Two
mechanisms are available and only the first is measured to matter:

* **resolution (measured to matter).** Spikes are one-bit reports of a continuous quantity. A
  frame whose motor drive is 0.99 of threshold and one whose drive is 0.01 are both a zero,
  so the evidence loses its resolution exactly where a threshold needs it. The motor layer's
  *pre-threshold input current* -- what ``connectome.LIFNeuronLayer`` integrates and then
  discards -- is continuous, and reading it costs one matrix product inside a forward pass
  that has already happened.
* **timing (in the rule, not yet separable from noise).** A boxcar sum is timing-blind: a
  drive that rose into the decision and one that fell out of it sum to the same number, and a
  decision is committed on the frame it is made on, so recency weighting is the principled
  shape. It is *not* established by this measurement: across the three trained arms at settle
  5 the decay sweep is inconsistent (0.25 beats 0.5 on one arm and loses on another, by less
  than one jump chunk), and the flat ``drive_sum`` does as well as most decays. The rule is
  timing-aware because that is the right shape for a committed decision, and the decay is
  reported rather than hidden; the *gain* over the shipped statistic is the resolution.

So the evidence vector is now built by an explicit, checkpoint-carried rule over the frames of
the window, and ``spike_sum`` is kept as one of those rules -- the shipped one -- so the
change can be measured against exactly what it replaces rather than against a memory of it.
The whole family, and every decay in :data:`DECAY_GRID`, is scored in the oracle table, so a
reader can see the selection rather than the winner alone.

**The replacement was then measured against the gate, and it did not earn its place.** The
ceiling's metric fits a boundary at a run-chunk recall floor; the gate fits a margin for
balanced accuracy and then replays the teacher's frames in sequence, and the two disagree. Dev
seeds 45-49, five replays, the real teacher shard, each statistic given a margin fitted for
itself (``prescreen.py``, one run per rule):

| statistic | rate-free budget view (untrained / round 1) | sequential view (untrained / round 1) |
| --- | --- | --- |
| ``spike_sum`` | 0.360 / 0.496 | 0.660 / **0.840** |
| ``spike_leaky_recency`` @ 0.25 | 0.344 / 0.480 | 0.760 / 0.720 |
| ``drive_sum`` | 0.336 / 0.488 | **0.960** / 0.740 |
| ``drive_leaky_recency`` @ 0.25 | 0.336 / 0.480 | 0.730 / 0.540 |
| ``drive_leaky_recency`` @ 0.75 | 0.336 / 0.544 | **0.960** / 0.510 |

The rate-free view -- recall at a fixed jump budget, which no margin touches -- barely moves:
0.336-0.360 for the untrained arm and 0.480-0.544 for round 1 across all five. The AUC agrees
from the other direction, because the drive orders the chunks *no better* than the sum it would
replace (0.611 against 0.657). What moves is the sequential view, by 0.30 on the untrained arm
and by 0.33 on round 1, and it moves with the jump rate a calibrated margin spends: the untrained
arm's 0.960 appears twice, at rates 0.374 and 0.404 -- both above the 0.35 budget, where at its
own budget its recall is 0.336, the same as the rules it appears to beat. So the resolution the
ceiling measured is real and confined to the one metric that fits its boundary at a run-recall
floor; it does not transfer to the statistic the controller ships. ``spike_sum`` therefore stays
the default and the drive rules stay reachable per arm: this module is what made the question
askable on identical draws, and the answer was that the statistic is not the lever -- the window
length is (see the README).

The rule is a *decoder* parameter, not a training one: it changes what the threshold is
applied to, which is why it travels in ``policy_config.macro_decoder`` beside the margin
fitted for it. A checkpoint that predates it carries no ``evidence_rule`` key and falls back
to :data:`DEFAULT_RULE`, and a margin in spike units means nothing to a drive statistic, so
the pre-screen re-fits the margin and says so (``margin_source``) instead of applying a
threshold whose units it cannot know.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch

from macro_decoder import JUMP_ACTION, RUN_ACTION

MOTOR_CHANNELS = 4

#: Evidence reductions. ``spike_sum`` is the statistic that shipped; the rest are measured
#: against it. ``drive_*`` reads the motor layer's pre-threshold current, ``spike_*`` reads
#: the binary spike train the same frames produced, and the ``*_normalized`` rules divide by
#: the window's own activity so the statistic cannot be lifted by driving everything harder.
RULES = (
    "spike_sum",
    "spike_leaky_recency",
    "spike_recency_normalized",
    "drive_sum",
    "drive_normalized",
    "drive_leaky_recency",
    "drive_recency_normalized",
    "drive_last",
)
#: Rules whose meaning depends on a decay, and the range it may take.
DECAY_RULES = ("spike_leaky_recency", "spike_recency_normalized",
               "drive_leaky_recency", "drive_recency_normalized")
#: The statistic the decoder shipped, and still ships by default. It is both the fallback for a
#: checkpoint that names no rule and -- since it was the only statistic that existed when a
#: rule could not be named -- the rule such a checkpoint's margin was fitted for.
SHIPPED_RULE = "spike_sum"
#: The statistic a decoder uses when its checkpoint does not name one: the spike sum, which is
#: what shipped. The drive statistics stay reachable per arm and per checkpoint, and the module
#: docstring records why the default did not move -- replacing it won nothing on the rate-free
#: view and cost 0.30 of recall on the sequential one.
DEFAULT_RULE = SHIPPED_RULE
#: Recency decay of a leaky rule. 0 would mean "the last frame only", 1 a boxcar sum.
DEFAULT_DECAY = 0.25
#: Every decay the oracle table scores, so the sweep is visible and the choice is reviewable.
DECAY_GRID = (0.0, 0.25, 0.5, 0.75)


def validate_rule(rule: str) -> str:
    """Refuse an unknown rule rather than silently deciding on something else."""
    if rule not in RULES:
        raise ValueError(f"unknown evidence rule {rule!r}; known rules are {list(RULES)}")
    return rule


def validate_decay(decay: float) -> float:
    """A decay is a weight in ``[0, 1]``; anything else is not a leaky accumulator."""
    if not isinstance(decay, (int, float)) or isinstance(decay, bool):
        raise ValueError("evidence decay must be a number")
    value = float(decay)
    if not 0.0 <= value <= 1.0:
        raise ValueError("evidence decay must be between 0 and 1")
    return value


def decay_weights(frames: int, decay: float = DEFAULT_DECAY) -> np.ndarray:
    """Weights that make a rule timing-aware: ``decay ** (frames - 1 - t)`` per frame.

    The controller commits its chunk on the last frame of the window, so that is the frame
    whose evidence is most recent -- and, measured over the teacher's chunks, recency
    weighting is what turns the analog drive into the best causal statistic of the family.
    """
    return np.asarray(decay, dtype=np.float64) ** np.arange(frames - 1, -1, -1)


def motor_drive(model, central_spikes) -> np.ndarray:
    """The motor layer's pre-threshold input current, as ``(motor_channels,)`` floats.

    Exactly the quantity ``connectome.LIFNeuronLayer.forward`` computes before its spike
    nonlinearity -- the mean-centred central-complex drive times the layer's current gain --
    and then thresholds away. It uses no labels, reads no frame the spikes do not also come
    from, and costs one matrix product inside a forward pass already done.
    """
    with torch.no_grad():
        current = torch.matmul(model.layer3_4.weight, central_spikes)
        current = current - current.mean()
        return (current * model.layer3_4.current_gain).detach().cpu().numpy().astype(np.float64)


def _powers(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(values * weights))


def _positive(values: np.ndarray) -> float:
    return float(np.abs(values).sum())


def reduce_channels(rule: str, spike_frames: np.ndarray, drive_frames: Optional[np.ndarray],
                    decay: float = DEFAULT_DECAY) -> Tuple[float, float]:
    """Reduce one settle window to ``(run_evidence, jump_evidence)`` for the decoder.

    ``spike_frames`` and ``drive_frames`` are ``(frames, motor_channels)`` and must cover the
    same frames. Each channel is reduced on its own, which is the structure the decoder
    compares: a jump is taken when the jump channel's evidence beats the run channel's by
    more than the margin.
    """
    validate_rule(rule)
    decay = validate_decay(decay)
    spikes = np.asarray(spike_frames, dtype=np.float64)
    if spikes.ndim != 2 or spikes.shape[1] < MOTOR_CHANNELS or len(spikes) == 0:
        raise ValueError("spike_frames must be a non-empty (frames, motor_channels) array")
    weights = decay_weights(len(spikes), decay)

    if rule.startswith("spike_"):
        run = spikes[:, RUN_ACTION]
        jump = spikes[:, JUMP_ACTION]
        if rule == "spike_sum":
            return float(run.sum()), float(jump.sum())
        if rule == "spike_leaky_recency":
            return _powers(run, weights), _powers(jump, weights)
        if rule == "spike_recency_normalized":
            total = _powers(np.abs(run), weights) + _powers(np.abs(jump), weights)
            if total <= 0.0:
                return 0.0, 0.0
            return _powers(run, weights) / total, _powers(jump, weights) / total
        raise ValueError(f"unknown spike rule {rule!r}")

    if drive_frames is None:
        raise ValueError(f"rule {rule!r} reads the motor drive; pass drive_frames")
    drive = np.asarray(drive_frames, dtype=np.float64)
    if drive.shape != spikes.shape:
        raise ValueError("drive_frames must cover the same frames as spike_frames")
    run = drive[:, RUN_ACTION]
    jump = drive[:, JUMP_ACTION]

    if rule == "drive_sum":
        return float(run.sum()), float(jump.sum())
    if rule == "drive_last":
        return float(run[-1]), float(jump[-1])
    if rule == "drive_normalized":
        total = _positive(run) + _positive(jump)
        if total <= 0.0:
            return 0.0, 0.0
        return float(run.sum()) / total, float(jump.sum()) / total
    if rule == "drive_leaky_recency":
        return _powers(run, weights), _powers(jump, weights)
    if rule == "drive_recency_normalized":
        # Both asks at once: timing-aware, and scale-free, so a network that drives every
        # channel harder cannot lift the statistic without lifting the denominator.
        total = _powers(np.abs(run), weights) + _powers(np.abs(jump), weights)
        if total <= 0.0:
            return 0.0, 0.0
        return _powers(run, weights) / total, _powers(jump, weights) / total
    raise ValueError(f"unknown drive rule {rule!r}")


def evidence_config(rule: Optional[str] = None, decay: Optional[float] = None) -> Dict[str, object]:
    """The checkpoint-stable form of an evidence rule, for ``policy_config``."""
    return {
        "evidence_rule": validate_rule(rule or DEFAULT_RULE),
        "evidence_decay": validate_decay(DEFAULT_DECAY if decay is None else decay),
    }


def rule_from_config(config: Optional[Dict[str, object]]) -> Tuple[str, float]:
    """Read the rule a checkpoint ships, falling back to the default for one that predates it."""
    config = dict(config or {})
    return (
        validate_rule(str(config.get("evidence_rule") or DEFAULT_RULE)),
        validate_decay(config.get("evidence_decay", DEFAULT_DECAY)),
    )


class MotorEvidence:
    """Accumulates a settle window's frames and reports the decoder's evidence vector.

    One ``observe`` call per settle frame, then ``channels()`` for the decision. The drive is
    only computed when the configured rule needs it, so a ``spike_*`` rule costs nothing
    extra over the shipped statistic.
    """

    def __init__(self, rule: Optional[str] = None, decay: Optional[float] = None):
        self.rule = validate_rule(rule or DEFAULT_RULE)
        self.decay = validate_decay(DEFAULT_DECAY if decay is None else decay)
        self.reset()

    @property
    def needs_drive(self) -> bool:
        return self.rule.startswith("drive_")

    def reset(self) -> None:
        """Clear the window (called at every decision, and at every episode boundary)."""
        self.spikes: list = []
        self.drives: list = []

    def observe(self, motor_spikes, layer_activations, model=None) -> None:
        """Record one frame of the window.

        Args:
            motor_spikes: the model's ``(motor_channels,)`` spike output for this frame.
            layer_activations: the model's activation dict, for the central-complex population.
            model: the network the drive is read from; required by a ``drive_*`` rule.
        """
        self.spikes.append(np.asarray(motor_spikes.detach().cpu().numpy(), dtype=np.float64))
        if self.needs_drive:
            if model is None or layer_activations is None:
                raise ValueError(f"rule {self.rule!r} reads the motor drive, so it needs the "
                                 "model and its layer activations")
            self.drives.append(motor_drive(model, layer_activations["central_complex"]))

    @property
    def frames(self) -> int:
        return len(self.spikes)

    def reduce(self) -> Tuple[float, float]:
        """``(run_evidence, jump_evidence)`` over the frames observed so far."""
        if not self.spikes:
            raise ValueError("no frames observed: a decision needs at least one settle frame")
        drives = None
        if self.needs_drive:
            drives = np.stack(self.drives) if self.drives else None
        return reduce_channels(self.rule, np.stack(self.spikes), drives, self.decay)

    def channels(self) -> torch.Tensor:
        """The ``(motor_channels,)`` evidence vector ``MacroActionDecoder.step`` reads."""
        run, jump = self.reduce()
        vector = torch.zeros(MOTOR_CHANNELS)
        vector[RUN_ACTION] = run
        vector[JUMP_ACTION] = jump
        return vector

    def config(self) -> Dict[str, object]:
        return evidence_config(self.rule, self.decay)

    def provenance(self) -> Dict[str, object]:
        """What a report needs in order to say which statistic produced its numbers."""
        return {**self.config(), "frames": self.frames,
                "reads_motor_drive": self.needs_drive}
