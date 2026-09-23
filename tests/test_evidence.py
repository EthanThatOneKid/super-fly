"""Tests for the settle-window evidence rules.

The shipped statistic summed binary motor spikes over the settle window, and the oracle
ceiling measurement put a number on what that costs: a five-frame window can only produce six
values per channel, so a fitted threshold on it caught 0.056 of the teacher's jump chunks
while a statistic of the same frames' analog drive caught several times that. These tests pin
the two properties the replacement has to have, and — more importantly — pin that the rule it
replaced is still reachable and unchanged, so the comparison is against the shipped statistic
rather than against a memory of it.

* **non-saturating**: the drive statistics are continuous and scale-free, so driving every
  channel harder cannot lift them.
* **timing-aware**: recency weighting distinguishes a window that rose into the decision from
  one that fell out of it, which a boxcar sum provably cannot.
"""

import unittest

import numpy as np
import torch

from evidence import (
    DEFAULT_DECAY,
    DEFAULT_RULE,
    RULES,
    MotorEvidence,
    decay_weights,
    evidence_config,
    motor_drive,
    reduce_channels,
    rule_from_config,
    validate_decay,
    validate_rule,
)
from macro_decoder import JUMP_ACTION, RUN_ACTION, MacroActionDecoder

RAM_SIZE = 0x0800


def frames(spikes, drives):
    """``(frames, 4)`` spike and drive windows from per-frame jump/run pairs."""
    spike_rows, drive_rows = [], []
    for (jump_s, run_s), (jump_d, run_d) in zip(spikes, drives):
        spike_row = np.zeros(4)
        drive_row = np.zeros(4)
        spike_row[JUMP_ACTION], spike_row[RUN_ACTION] = jump_s, run_s
        drive_row[JUMP_ACTION], drive_row[RUN_ACTION] = jump_d, run_d
        spike_rows.append(spike_row)
        drive_rows.append(drive_row)
    return np.stack(spike_rows), np.stack(drive_rows)


class TestRules(unittest.TestCase):
    def test_an_unknown_rule_is_refused_rather_than_substituted(self):
        with self.assertRaises(ValueError):
            validate_rule("spike_something")
        with self.assertRaises(ValueError):
            reduce_channels("nonsense", np.zeros((3, 4)), None)

    def test_the_decay_is_a_weight_in_the_unit_interval(self):
        self.assertEqual(validate_decay(0.5), 0.5)
        for bad in (-0.1, 1.1, "half", True):
            with self.assertRaises(ValueError):
                validate_decay(bad)

    def test_the_shipped_statistic_is_still_reachable_and_unchanged(self):
        """``spike_sum`` has to reproduce the statistic it replaced, exactly.

        Everything measured under the new rule is compared against this one, so if the two
        ever drifted apart the comparison would silently become two changes at once.
        """
        spikes, drives = frames([(1, 0), (0, 1), (1, 0)], [(0.0, 0.0)] * 3)
        run, jump = reduce_channels("spike_sum", spikes, None)
        self.assertEqual(run, 1.0)
        self.assertEqual(jump, 2.0)

    def test_a_drive_rule_needs_the_drive_and_says_which_rule_asked(self):
        spikes, _ = frames([(1, 0)], [(0.0, 0.0)])
        with self.assertRaises(ValueError) as caught:
            reduce_channels("drive_sum", spikes, None)
        self.assertIn("drive_sum", str(caught.exception))

    def test_the_drive_rules_are_reductions_of_the_frames_they_are_given(self):
        spikes, drives = frames(
            [(0, 0), (0, 0), (0, 0)],
            [(1.0, -1.0), (2.0, 1.0), (3.0, 0.5)],
        )
        run, jump = reduce_channels("drive_sum", spikes, drives)
        self.assertAlmostEqual(jump, 6.0)
        self.assertAlmostEqual(run, 0.5)
        run, jump = reduce_channels("drive_last", spikes, drives)
        self.assertAlmostEqual(jump, 3.0)
        self.assertAlmostEqual(run, 0.5)

    def test_a_decay_of_one_is_a_boxcar_and_zero_reads_only_the_newest_frame(self):
        spikes, drives = frames([(0, 0)] * 3, [(3.0, 1.0), (2.0, 1.0), (1.0, 1.0)])
        self.assertAlmostEqual(reduce_channels("drive_leaky_recency", spikes, drives, 1.0)[1], 6.0)
        self.assertAlmostEqual(reduce_channels("drive_leaky_recency", spikes, drives, 0.0)[1], 1.0)
        np.testing.assert_allclose(decay_weights(3, 1.0), [1.0, 1.0, 1.0])

    def test_a_normalized_rule_is_scale_free(self):
        """The non-saturating property: raising the whole drive cannot lift the statistic."""
        spikes, drives = frames([(0, 0)] * 3, [(1.0, -1.0), (2.0, 1.0), (1.5, 0.5)])
        plain = reduce_channels("drive_recency_normalized", spikes, drives)
        louder = reduce_channels("drive_recency_normalized", spikes, drives * 10.0)
        self.assertAlmostEqual(plain[1], louder[1])
        self.assertLessEqual(abs(plain[1]), 1.0)

    def test_recency_weighting_separates_two_windows_a_sum_cannot(self):
        """The timing defect, pinned: same sum, different order, different evidence.

        A drive that rose into the committed frame and one that fell out of it are the same
        number to a boxcar sum, and they are the opposite evidence to the controller that has
        to act now.
        """
        spikes, rising = frames([(0, 0)] * 3, [(0.0, 0.0), (1.0, 1.0), (4.0, 1.0)])
        _, falling = frames([(0, 0)] * 3, [(4.0, 1.0), (1.0, 1.0), (0.0, 0.0)])

        flat_rising = reduce_channels("drive_sum", spikes, rising)
        flat_falling = reduce_channels("drive_sum", spikes, falling)
        self.assertAlmostEqual(flat_rising[1] - flat_rising[0], flat_falling[1] - flat_falling[0])

        timed_rising = reduce_channels("drive_leaky_recency", spikes, rising)
        timed_falling = reduce_channels("drive_leaky_recency", spikes, falling)
        self.assertGreater(timed_rising[1] - timed_rising[0],
                           timed_falling[1] - timed_falling[0])

    def test_every_declared_rule_is_reachable(self):
        spikes, drives = frames([(1, 1), (0, 0)], [(1.0, -1.0), (0.5, 0.5)])
        for rule in RULES:
            with self.subTest(rule=rule):
                run, jump = reduce_channels(rule, spikes, drives, DEFAULT_DECAY)
                self.assertTrue(np.isfinite(run) and np.isfinite(jump))

    def test_an_empty_window_has_no_evidence(self):
        with self.assertRaises(ValueError):
            reduce_channels("spike_sum", np.zeros((0, 4)), None)
        with self.assertRaises(ValueError):
            reduce_channels("spike_sum", np.zeros((3, 2)), None)

    def test_a_window_of_silence_is_zero_rather_than_a_division(self):
        spikes, drives = frames([(0, 0)] * 2, [(0.0, 0.0)] * 2)
        for rule in ("drive_normalized", "drive_recency_normalized", "spike_recency_normalized"):
            with self.subTest(rule=rule):
                self.assertEqual(reduce_channels(rule, spikes, drives), (0.0, 0.0))


class _StubLayer:
    """A motor layer that records how often the drive is read."""

    def __init__(self):
        self.weight = torch.zeros(4, 3)
        self.current_gain = 6.0
        self.reads = 0

    def __getattr__(self, name):  # pragma: no cover - only reached on a wrong attribute
        raise AssertionError(f"the motor layer was touched (missing {name!r})")


class _StubModel:
    def __init__(self):
        self.layer3_4 = _StubLayer()


class TestAccumulator(unittest.TestCase):
    def test_the_vector_carries_the_two_trained_channels(self):
        evidence = MotorEvidence("spike_sum")
        for jump, run in ((1, 0), (1, 1)):
            spikes, _ = frames([(jump, run)], [(0.0, 0.0)])
            evidence.observe(torch.tensor(spikes[0]), None)

        vector = evidence.channels()
        self.assertAlmostEqual(float(vector[RUN_ACTION]), 1.0)
        self.assertAlmostEqual(float(vector[JUMP_ACTION]), 2.0)
        self.assertEqual(evidence.frames, 2)

    def test_a_spike_rule_never_reads_the_motor_layer(self):
        """A spike statistic must cost nothing more than it did before."""
        model = _StubModel()
        evidence = MotorEvidence("spike_sum")
        evidence.observe(torch.ones(4), {"central_complex": torch.zeros(3)}, model)
        self.assertEqual(model.layer3_4.reads, 0)
        self.assertFalse(evidence.needs_drive)

    def test_a_drive_rule_reads_the_pre_threshold_current(self):
        model = _StubModel()
        drive = motor_drive(model, torch.zeros(3))
        self.assertEqual(drive.shape, (4,))
        # Mean-centred, so the four channels always sum to zero: the layer's own arithmetic.
        self.assertAlmostEqual(float(drive.sum()), 0.0, places=6)

    def test_the_drive_is_continuous_where_a_spike_is_one_bit(self):
        """A frame just short of threshold still carries evidence, and the spike throws it away."""
        model = _StubModel()
        model.layer3_4.weight[JUMP_ACTION, 0] = 0.1
        drive = motor_drive(model, torch.tensor([1.0, 0.0, 0.0]))
        self.assertNotEqual(float(drive[JUMP_ACTION]), 0.0)
        # Not a spike count: the value a fitted threshold reads is not an integer.
        self.assertFalse(np.allclose(drive, np.round(drive)))

    def test_a_drive_rule_refuses_a_missing_model_rather_than_deciding_blind(self):
        evidence = MotorEvidence("drive_sum")
        with self.assertRaises(ValueError):
            evidence.observe(torch.ones(4), None, None)

    def test_a_decision_needs_at_least_one_frame(self):
        with self.assertRaises(ValueError):
            MotorEvidence("spike_sum").channels()


class TestDecoderWiring(unittest.TestCase):
    def test_the_rule_is_part_of_the_decoder_config(self):
        decoder = MacroActionDecoder(chunk_frames=5, evidence_rule="drive_normalized",
                                     evidence_decay=0.25)
        config = decoder.config()
        self.assertEqual(config["evidence_rule"], "drive_normalized")
        self.assertEqual(config["evidence_decay"], 0.25)
        accumulator = decoder.new_evidence()
        self.assertEqual(accumulator.rule, "drive_normalized")
        self.assertEqual(accumulator.decay, 0.25)

    def test_a_decoder_without_a_rule_takes_the_default_one(self):
        self.assertEqual(MacroActionDecoder(chunk_frames=5).evidence_rule, DEFAULT_RULE)
        self.assertEqual(rule_from_config(None), (DEFAULT_RULE, DEFAULT_DECAY))
        self.assertEqual(evidence_config(), {"evidence_rule": DEFAULT_RULE,
                                             "evidence_decay": DEFAULT_DECAY})

    def test_a_config_round_trips_through_the_checkpoint_form(self):
        config = evidence_config("drive_leaky_recency", 0.75)
        self.assertEqual(rule_from_config(config), ("drive_leaky_recency", 0.75))

    def test_an_unusable_rule_is_refused_at_construction(self):
        with self.assertRaises(ValueError):
            MacroActionDecoder(chunk_frames=5, evidence_rule="spike_guessing")
        with self.assertRaises(ValueError):
            MacroActionDecoder(chunk_frames=5, evidence_decay=2.0)


if __name__ == "__main__":
    unittest.main()
