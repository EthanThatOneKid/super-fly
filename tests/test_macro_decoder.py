import json
import math
import unittest

import torch

from macro_decoder import (
    JUMP_ACTION,
    MAX_CHUNK_FRAMES,
    MAX_MACRO_REFRACTORY_FRAMES,
    RUN_ACTION,
    MacroActionDecoder,
    calibrate_jump_margin,
    decision_quality,
)


def motor(right=0.0, jump=0.0, right_jump=0.0, noop=0.0):
    return torch.tensor([noop, right, jump, right_jump])


class TestMacroActionDecoder(unittest.TestCase):

    def test_chunk_is_held_to_completion(self):
        """A committed chunk must never be re-decided mid-chunk."""
        decoder = MacroActionDecoder(chunk_frames=4)
        decisions = [decoder.step(motor(right_jump=1.0)) for _ in range(6)]

        self.assertTrue(decisions[0].is_new_decision)
        self.assertEqual(decisions[0].macro, "jump")
        self.assertEqual(decisions[0].action_idx, JUMP_ACTION)
        # Chunk runs to completion: frames 1-3 are held, frame 4 opens a new chunk.
        self.assertEqual([d.is_new_decision for d in decisions], [True, False, False, False, True, False])
        self.assertEqual([d.action_idx for d in decisions], [JUMP_ACTION] * 4 + [JUMP_ACTION, JUMP_ACTION])
        self.assertEqual(decoder.decisions, 2)

    def test_run_evidence_selects_run_chunk(self):
        decoder = MacroActionDecoder(chunk_frames=3)
        decision = decoder.step(motor(right=2.0))
        self.assertEqual(decision.macro, "run")
        self.assertEqual(decision.action_idx, RUN_ACTION)

    def test_jump_channel_decides_jump_chunks(self):
        """The RIGHT+B+A channel -- the one supervised jump targets drive -- opens a
        jump chunk, and a tie falls back to running right."""
        jump_decoder = MacroActionDecoder(chunk_frames=3)
        decision = jump_decoder.step(motor(right_jump=1.0))
        self.assertEqual(decision.macro, "jump")
        self.assertEqual(decision.jump_evidence, 1.0)
        self.assertEqual(decision.run_evidence, 0.0)

        # Equal evidence on both executable channels defaults to running, which is
        # what keeps a jump-happy network from an unbounded jump rate.
        tie_decoder = MacroActionDecoder(chunk_frames=3)
        tie = tie_decoder.step(motor(right=1.0, right_jump=1.0))
        self.assertEqual(tie.macro, "run")
        self.assertEqual(tie.jump_evidence, 1.0)
        self.assertEqual(tie.run_evidence, 1.0)

    def test_standing_jump_neuron_does_not_open_a_jump_chunk(self):
        """Unit 2 (JUMP without RIGHT) is not a macro action and nothing trains it
        positively, so it must not be able to drive the decoder."""
        decoder = MacroActionDecoder(chunk_frames=3)
        decision = decoder.step(motor(jump=3.0))
        self.assertEqual(decision.macro, "run")
        self.assertEqual(decision.action_idx, RUN_ACTION)
        self.assertEqual(decision.jump_evidence, 0.0)

    def test_jump_channel_is_counted_once(self):
        """A jump channel carried in the run term as well would cancel itself out and
        read as a tie -- the failure this decoder has to avoid."""
        decoder = MacroActionDecoder(chunk_frames=3)
        decision = decoder.step(motor(right=0.02, right_jump=0.9))
        self.assertEqual(decision.macro, "jump")
        self.assertAlmostEqual(decision.jump_evidence - decision.run_evidence, 0.88, places=6)

    def test_jump_margin_and_refractory_bound_jump_rate(self):
        # A 1.0 vs 1.0 tie does not clear the 0.5 margin, so the chunk runs.
        tied = MacroActionDecoder(chunk_frames=4, jump_margin=0.5, refractory_frames=8)
        tied_first = tied.step(motor(right=1.0, right_jump=1.0))
        self.assertEqual(tied_first.macro, "run")
        self.assertEqual(tied_first.jump_evidence, 1.0)
        self.assertEqual(tied_first.run_evidence, 1.0)

        # Once jump evidence clears the margin the jump chunk length is used.
        cleared = MacroActionDecoder(chunk_frames=4, jump_chunk_frames=6, jump_margin=0.5)
        cleared_first = cleared.step(motor(right_jump=1.0))
        self.assertEqual(cleared_first.macro, "jump")
        self.assertEqual(cleared_first.chunk_frames, 6)

        decoder = MacroActionDecoder(chunk_frames=4, jump_margin=0.5, refractory_frames=8)
        decisions = [decoder.step(motor(right_jump=3.0)) for _ in range(12)]
        jump_starts = [i for i, d in enumerate(decisions) if d.is_new_decision and d.macro == "jump"]
        # refractory_frames=8 with a 4-frame cadence: the first chunk opens at frame
        # 0, the hold suppresses the second decision, and the next jump chunk only
        # opens at frame 8. Jump rate is therefore bounded, not frame-wise.
        self.assertEqual(jump_starts, [0, 8])

    def test_chunk_lengths_are_bounded(self):
        with self.assertRaises(ValueError):
            MacroActionDecoder(chunk_frames=0)
        with self.assertRaises(ValueError):
            MacroActionDecoder(chunk_frames=MAX_CHUNK_FRAMES + 1)
        with self.assertRaises(ValueError):
            MacroActionDecoder(chunk_frames=4, jump_chunk_frames=MAX_CHUNK_FRAMES + 1)
        with self.assertRaises(ValueError):
            MacroActionDecoder(chunk_frames=4, refractory_frames=MAX_MACRO_REFRACTORY_FRAMES + 1)
        with self.assertRaises(ValueError):
            MacroActionDecoder(chunk_frames=4).step(torch.zeros(3))

    def test_reset_clears_chunk_state_at_episode_boundary(self):
        decoder = MacroActionDecoder(chunk_frames=5)
        for _ in range(3):
            decoder.step(motor(right_jump=1.0))
        self.assertGreater(decoder.decisions, 0)
        self.assertGreater(decoder.jump_decisions, 0)

        decoder.reset()

        self.assertIsNone(decoder.active_macro)
        self.assertEqual(decoder.decisions, 0)
        self.assertEqual(decoder.jump_decisions, 0)
        self.assertEqual(decoder.frames_executed, 0)
        first = decoder.step(motor(right=1.0))
        self.assertTrue(first.is_new_decision)
        self.assertEqual(decoder.decisions, 1)

    def test_metadata_reports_bounded_configuration(self):
        decoder = MacroActionDecoder(chunk_frames=7, jump_chunk_frames=9, jump_margin=0.25, refractory_frames=11)
        config = decoder.config()
        self.assertEqual(config["chunk_frames"], 7)
        self.assertEqual(config["jump_chunk_frames"], 9)
        self.assertEqual(config["jump_margin"], 0.25)
        self.assertEqual(config["refractory_frames"], 11)
        self.assertEqual(config["max_chunk_frames"], MAX_CHUNK_FRAMES)

        metadata = decoder.metadata()
        self.assertIn("decisions", metadata)
        self.assertIn("frames_executed", metadata)


class TestJumpMarginCalibration(unittest.TestCase):
    """The jump threshold has to come from labelled evidence, not from zero.

    A trained readout is offset: on the teacher's own chunks the mean
    ``jump - run`` evidence is negative for *both* classes, so a hard-coded zero
    margin means "never jump" however well the classes separate.
    """

    #: 6 run chunks comfortably below the boundary, 6 jump chunks above it -- but
    #: every value is negative, exactly the offset seen on the real teacher shard.
    OFFSET_RUN = [-3.0, -2.8, -2.6, -2.4, -2.2, -2.0]
    OFFSET_JUMP = [-1.8, -1.6, -1.4, -1.2, -1.0, -0.8]

    def test_calibration_finds_a_boundary_that_zero_cannot(self):
        diffs = self.OFFSET_RUN + self.OFFSET_JUMP
        labels = [False] * len(self.OFFSET_RUN) + [True] * len(self.OFFSET_JUMP)

        calibrated = calibrate_jump_margin(diffs, labels)

        self.assertGreaterEqual(calibrated["balanced_accuracy"], 0.99)
        self.assertAlmostEqual(calibrated["jump_recall"], 1.0)
        self.assertAlmostEqual(calibrated["run_recall"], 1.0)
        self.assertLess(calibrated["margin"], 0.0)

        # The observed evidence is negative even for jump chunks, so at the
        # uncalibrated zero margin the controller reads it as "run" and never jumps.
        uncalibrated = MacroActionDecoder(chunk_frames=3, jump_margin=0.0)
        decision = uncalibrated.step(motor(right_jump=self.OFFSET_JUMP[-1]))
        self.assertEqual(decision.macro, "run")
        self.assertLess(decision.jump_evidence - decision.run_evidence, 0.0)

    def test_calibrated_margin_reproduces_the_jump_schedule(self):
        diffs = self.OFFSET_RUN + self.OFFSET_JUMP
        labels = [False] * len(self.OFFSET_RUN) + [True] * len(self.OFFSET_JUMP)
        calibrated = calibrate_jump_margin(diffs, labels)

        # One-frame chunks so every replayed decision is made independently.
        decoder = MacroActionDecoder(chunk_frames=1, jump_margin=calibrated["margin"])
        macros = [
            decoder.step(motor(right_jump=diff)).macro
            for diff in diffs
        ]
        self.assertEqual(macros, ["run"] * len(self.OFFSET_RUN) + ["jump"] * len(self.OFFSET_JUMP))

    def test_balanced_accuracy_ignores_class_imbalance(self):
        """A majority-class rule scores 0.5 balanced accuracy no matter how skewed the
        labels are, so the calibrator cannot win by always predicting run."""
        diffs = [-2.0] * 90 + [-1.0] * 10
        labels = [False] * 90 + [True] * 10
        genuine = calibrate_jump_margin(diffs, labels)
        self.assertAlmostEqual(genuine["balanced_accuracy"], 1.0)

        inseparable = calibrate_jump_margin([-1.0] * 100, labels)
        # Every margin either fires on both classes or on neither: no more than 0.5.
        self.assertLessEqual(inseparable["balanced_accuracy"], 0.5)

    def test_rate_matching_bounds_over_jumping(self):
        """Matching the labelled jump rate is available because over-jumping is its own
        failure mode, not merely a precision problem."""
        diffs = [-3.0, -2.9, -2.8, -2.7, -1.5, -1.4, -1.3, -1.2]
        labels = [False, False, False, False, True, True, True, True]
        matched = calibrate_jump_margin(diffs, labels, method="jump_rate")
        self.assertEqual(matched["method"], "jump_rate")
        self.assertAlmostEqual(matched["jump_rate"], 0.5, places=6)
        self.assertAlmostEqual(matched["target_jump_rate"], 0.5)

    def test_the_calibrated_margin_is_always_finite_and_json_safe(self):
        """An infinite margin prints as ``Infinity``, which is not JSON.

        The boundary meaning "jump on every decision" used to be ``-math.inf``, so an
        inseparable readout published a decoder that jumps on everything and wrote a
        checkpoint no strict JSON reader would accept.
        """
        labels = [False] * 90 + [True] * 10

        inseparable = calibrate_jump_margin([-1.0] * 100, labels)

        self.assertTrue(math.isfinite(inseparable["margin"]))
        self.assertEqual(json.loads(json.dumps(inseparable))["margin"], inseparable["margin"])

    def test_a_calibration_that_does_not_separate_says_so(self):
        """Ties are broken toward jumping on everything, so degeneracy must be visible."""
        labels = [False] * 90 + [True] * 10

        # Identical evidence for both classes: no boundary beats the constant rules.
        degenerate = calibrate_jump_margin([-1.0] * 100, labels)
        self.assertTrue(degenerate["degenerate"])
        self.assertIn("does not separate", degenerate["degenerate_reason"])
        self.assertEqual(degenerate["balanced_accuracy"], 0.5)
        self.assertEqual(degenerate["jump_rate"], 1.0)

        separable = calibrate_jump_margin(
            self.OFFSET_RUN + self.OFFSET_JUMP, [False] * 6 + [True] * 6
        )
        self.assertFalse(separable["degenerate"])
        self.assertIsNone(separable["degenerate_reason"])

        single_class = calibrate_jump_margin([-1.0, -2.0], [True, True])
        self.assertTrue(single_class["degenerate"])
        self.assertIn("one chunk class", single_class["degenerate_reason"])

    def test_single_class_is_reported_not_guessed(self):
        calibrated = calibrate_jump_margin([-1.0, -2.0, -3.0], [False, False, False])
        self.assertEqual(calibrated["method"], "insufficient_classes")
        self.assertEqual(calibrated["margin"], 0.0)
        self.assertIsNone(calibrated["balanced_accuracy"])
        self.assertEqual(calibrated["samples"], 3)

        empty = calibrate_jump_margin([], [])
        self.assertEqual(empty["method"], "insufficient_classes")

    def test_malformed_calibration_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            calibrate_jump_margin([-1.0, -2.0], [True])
        with self.assertRaises(ValueError):
            calibrate_jump_margin([-1.0], [True], method="guess")


class TestFixedMarginQuality(unittest.TestCase):
    """Reporting a score at a margin fitted *elsewhere*.

    A calibration fits the best boundary on whatever evidence it is handed, so using
    the same evidence to then report a score is optimistic. The held-out protocol
    therefore applies the training margin unchanged, and these tests pin that the two
    reportings agree at one margin.
    """

    RUN = [-2.4, -2.2, -2.0, -1.95, -2.1, -2.3]
    JUMP = [-1.8, -1.6, -1.4, -1.2, -1.0, -0.8]

    def _labelled(self):
        diffs = self.RUN + self.JUMP
        return diffs, [False] * len(self.RUN) + [True] * len(self.JUMP)

    def test_quality_at_the_calibrated_margin_matches_the_calibration(self):
        diffs, labels = self._labelled()
        calibrated = calibrate_jump_margin(diffs, labels)
        fixed = decision_quality(diffs, labels, calibrated["margin"])

        self.assertEqual(fixed["method"], "fixed_margin")
        for key in ("balanced_accuracy", "jump_recall", "run_recall", "jump_rate",
                    "target_jump_rate", "samples", "jump_samples"):
            self.assertEqual(fixed[key], calibrated[key], key)

    def test_a_margin_from_elsewhere_is_applied_not_re_fitted(self):
        diffs, labels = self._labelled()

        never_jump = decision_quality(diffs, labels, max(diffs) + 1.0)
        self.assertEqual(never_jump["margin"], max(diffs) + 1.0)
        self.assertEqual(never_jump["jump_rate"], 0.0)
        self.assertEqual(never_jump["jump_recall"], 0.0)
        self.assertEqual(never_jump["run_recall"], 1.0)
        self.assertAlmostEqual(never_jump["balanced_accuracy"], 0.5)
        # Re-fitting would have found the separating boundary and scored ~1.0.
        self.assertLess(never_jump["balanced_accuracy"],
                        calibrate_jump_margin(diffs, labels)["balanced_accuracy"])

    def test_an_absent_class_reports_none_rather_than_zero(self):
        """A held-out split with no jump chunk scores no recall for jumps: saying so
        is honest where a 0.0 would look like a measured failure."""
        only_run = decision_quality([-1.0, -2.0], [False, False], 0.0)
        self.assertIsNone(only_run["jump_recall"])
        self.assertIsNone(only_run["balanced_accuracy"])
        self.assertEqual(only_run["run_recall"], 1.0)

        only_jump = decision_quality([1.0, 2.0], [True, True], 0.0)
        self.assertIsNone(only_jump["run_recall"])
        self.assertIsNone(only_jump["balanced_accuracy"])
        self.assertEqual(only_jump["jump_recall"], 1.0)

    def test_malformed_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            decision_quality([-1.0], [True, False], 0.0)
        with self.assertRaises(ValueError):
            decision_quality([-1.0], [True], None)


if __name__ == "__main__":
    unittest.main()
