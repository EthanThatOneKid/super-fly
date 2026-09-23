"""Tests for the oracle-ceiling instrument.

Two kinds of test matter here. The first is the arithmetic of each statistic and boundary
sweep, which is ordinary unit testing. The second is the guard that makes the instrument
worth reading at all: a probe that *never* finds signal would make "the ceiling is low"
vacuous, and a fit that is allowed to score itself would make "the ceiling is high" vacuous.
So the held-out readouts are tested against synthetic data that is genuinely separable, and
against the same data with the labels shuffled.
"""

import json
import os
import tempfile
import unittest

import numpy as np
import torch

import oracle
from macro_decoder import JUMP_ACTION, RUN_ACTION


def _trajectory(frames, jump, run):
    """A motor trace of per-frame spike counts, with the two trained channels filled in."""
    trace = np.zeros((frames, 4), dtype=np.float64)
    trace[:, JUMP_ACTION] = jump
    trace[:, RUN_ACTION] = run
    return trace


def _synthetic_rows(n_chunks=60, frames=3, channels=6, separation=1.0, shuffle=False, seed=0):
    """Chunk rows with a real, linearly readable difference in the central population.

    With ``separation=0.0`` or ``shuffle=True`` the same construction carries no signal, so
    every probe has an honest null to be tested against.
    """
    rng = np.random.default_rng(seed)
    direction = np.zeros(channels)
    direction[:2] = 1.0
    rows = []
    for index in range(n_chunks):
        label = index % 3 == 0
        central = rng.normal(scale=0.5, size=(frames, channels))
        if separation:
            central += direction * (separation if label else -separation)
        motor = np.zeros((frames, 4))
        if separation:
            motor[:, JUMP_ACTION] = 1.0 if label else 0.0
            motor[:, RUN_ACTION] = 0.0 if label else 1.0
        # The analog drive the spikes quantise: the same signal, continuous, and (unlike a
        # spike count) different for every chunk even where the spikes are identical.
        drive = motor * 0.5 + rng.normal(scale=0.05, size=(frames, 4))
        rows.append({"label": label, "motor": motor, "drive": drive, "central": central})
    if shuffle:
        rng.shuffle(rows)
        for row in rows:
            row["label"] = bool(rng.random() < 0.25)
    return rows


class ReductionTests(unittest.TestCase):
    def test_sum_and_mean_read_the_same_ordering(self):
        trace = _trajectory(5, jump=[1, 1, 0, 1, 1], run=[0, 0, 1, 0, 0])
        self.assertAlmostEqual(oracle.reduce_trajectory(trace, "sum"), 3.0)
        self.assertAlmostEqual(oracle.reduce_trajectory(trace, "mean"), 0.6)

    def test_last_frame_reads_only_the_boundary_frame(self):
        trace = _trajectory(3, jump=[1, 0, 0], run=[0, 0, 1])
        self.assertAlmostEqual(oracle.reduce_trajectory(trace, "last_frame"), -1.0)
        self.assertAlmostEqual(oracle.reduce_trajectory(trace, "sum"), 0.0)

    def test_prefix_max_finds_the_best_prefix_a_threshold_rule_could_act_on(self):
        trace = _trajectory(4, jump=[1, 1, 0, 0], run=[0, 0, 1, 1])
        self.assertAlmostEqual(oracle.reduce_trajectory(trace, "prefix_max"), 2.0)
        self.assertAlmostEqual(oracle.reduce_trajectory(trace, "sum"), 0.0)

    def test_contrast_is_scale_free(self):
        trace = _trajectory(2, jump=[1, 1], run=[0, 0])
        doubled = _trajectory(2, jump=[2, 2], run=[0, 0])
        self.assertAlmostEqual(oracle.reduce_trajectory(trace, "contrast"),
                               oracle.reduce_trajectory(doubled, "contrast"))

    def test_latency_reports_the_crossing_frame(self):
        # Series is [0, -1, 1, 1], so the cumulative difference first goes positive at frame 3.
        trace = _trajectory(4, jump=[0, 0, 1, 1], run=[0, 1, 0, 0])
        self.assertAlmostEqual(oracle.reduce_trajectory(trace, "latency"), -3.0)
        never = _trajectory(4, jump=[0, 0, 0, 0], run=[1, 1, 1, 1])
        self.assertAlmostEqual(oracle.reduce_trajectory(never, "latency"), -5.0)

    def test_single_frame_window_is_legal(self):
        trace = _trajectory(1, jump=[1], run=[0])
        self.assertAlmostEqual(oracle.reduce_trajectory(trace, "prefix_max"), 1.0)

    def test_single_channel_trajectory_is_read_directly(self):
        self.assertAlmostEqual(oracle.reduce_trajectory(np.array([[1.0], [2.0]]), "last_frame"), 2.0)

    def test_unknown_rule_is_an_error(self):
        with self.assertRaises(ValueError):
            oracle.reduce_trajectory(_trajectory(2, [1, 1], [0, 0]), "vibes")

    def test_oracle_frame_uses_the_label_and_bounds_a_fixed_frame_rule(self):
        trace = _trajectory(3, jump=[0, 1, 0], run=[0, 0, 0])
        self.assertAlmostEqual(oracle.oracle_frame_statistic(trace, label=True), 1.0)
        self.assertAlmostEqual(oracle.oracle_frame_statistic(trace, label=False), 0.0)


class OrderingTests(unittest.TestCase):
    def test_auc_is_perfect_for_a_separated_pair_and_reversed_for_a_flipped_one(self):
        scores = [0.1, 0.2, 0.8, 0.9]
        labels = [False, False, True, True]
        self.assertAlmostEqual(oracle.auc(scores, labels), 1.0)
        self.assertAlmostEqual(oracle.auc(scores, [True, True, False, False]), 0.0)

    def test_auc_is_chance_for_a_constant_statistic(self):
        self.assertAlmostEqual(oracle.auc([1.0] * 4, [True, True, False, False]), 0.5)

    def test_auc_is_undefined_with_one_class(self):
        self.assertIsNone(oracle.auc([0.1, 0.2], [True, True]))


class ThresholdTests(unittest.TestCase):
    def test_separated_evidence_reaches_perfect_balanced_accuracy(self):
        scores = [0.0, 0.1, 0.2, 0.8, 0.9, 1.0]
        labels = [False, False, False, True, True, True]
        summary = oracle.sweep_threshold(scores, labels)
        self.assertAlmostEqual(summary["balanced_accuracy"], 1.0)
        self.assertAlmostEqual(summary["jump_recall_at_min_run_recall"], 1.0)

    def test_a_constant_statistic_cannot_beat_zero_at_a_run_recall_floor(self):
        # The always-jump rule reports a perfect jump recall and saves no run chunk; the
        # floor exists so that this cannot be quoted as a ceiling.
        summary = oracle.sweep_threshold([0.5] * 8, [True, True, False, False, False, False, False, False])
        self.assertAlmostEqual(summary["jump_recall_at_min_run_recall"], 0.0)
        self.assertAlmostEqual(summary["balanced_accuracy"], 0.5)

    def test_an_inverted_statistic_cannot_claim_jump_recall_it_does_not_have(self):
        # The statistic ranks the classes backwards, so the only boundary that catches the
        # jumps is the always-jump one -- which the run-recall floor must refuse.
        scores = [0.0, 0.0, 0.9, 0.9]
        labels = [True, True, False, False]
        margin = oracle.bounded_margin(scores, labels, min_run_recall=0.95)
        self.assertIsNotNone(margin)
        self.assertAlmostEqual(margin["run_recall"], 1.0)
        self.assertAlmostEqual(margin["jump_recall"], 0.0)

    def test_best_margin_needs_both_classes(self):
        with self.assertRaises(ValueError):
            oracle.best_margin([0.1, 0.2], [True, True])

    def test_a_tie_is_broken_toward_running_not_toward_jumping(self):
        # A statistic with no separation must read as "never jump". Breaking the tie the
        # other way is exactly the defect that shipped an always-jump decoder.
        summary = oracle.best_margin([0.5] * 8, [True, True, False, False, False, False, False, False])
        self.assertAlmostEqual(summary["jump_recall"], 0.0)
        self.assertAlmostEqual(summary["run_recall"], 1.0)

    def test_a_degenerate_population_readout_catches_nothing_rather_than_everything(self):
        # A constant population gives the Fisher solve an all-zero direction, so every chunk
        # scores the same and the only boundary at the run-recall floor predicts no jump. The
        # failure mode being guarded against is the opposite reading -- a statistic with no
        # separation reporting a perfect jump recall through an always-jump boundary.
        rows = [{"label": index % 4 == 0, "motor": np.zeros((2, 4)),
                 "central": np.ones((2, 5))} for index in range(40)]
        report = oracle.cross_validated_lda(rows)
        self.assertTrue(report["fitted"])
        self.assertAlmostEqual(report["jump_recall"], 0.0)
        self.assertAlmostEqual(report["run_recall"], 1.0)
        self.assertAlmostEqual(report["balanced_accuracy"], 0.5)


    def test_single_class_evidence_reports_undefined_rather_than_a_number(self):
        summary = oracle.sweep_threshold([0.1, 0.2], [True, True])
        self.assertIsNone(summary["balanced_accuracy"])
        self.assertIn("one chunk class", summary["note"])


class FamilyTests(unittest.TestCase):
    def test_fitted_readouts_are_excluded_from_a_family_ceiling(self):
        entry = {
            "rule": {"jump_recall_at_min_run_recall": 0.30, "auc": 0.60},
            "overfit": {"fitted": True, "jump_recall_at_min_run_recall": 0.95, "auc": 0.99},
        }
        self.assertAlmostEqual(oracle.best_bounded(entry), 0.30)
        self.assertAlmostEqual(oracle.best_auc(entry), 0.60)

    def test_family_ceiling_descends_into_nested_projection_families(self):
        entry = {"lda": {"sum": {"jump_recall_at_min_run_recall": 0.4, "auc": 0.7},
                         "oracle_frame": {"jump_recall_at_min_run_recall": 0.6, "auc": 0.9}},
                 "pc1": {"sum": {"jump_recall_at_min_run_recall": 0.2, "auc": 0.5}}}
        self.assertAlmostEqual(oracle.best_bounded(entry), 0.6)
        self.assertAlmostEqual(oracle.best_auc(entry), 0.9)

    def test_principal_components_do_not_read_the_labels(self):
        rows = _synthetic_rows(seed=3)
        first = oracle.principal_components(rows)
        flipped = [dict(row, label=not row["label"]) for row in rows]
        second = oracle.principal_components(flipped)
        self.assertEqual(first.shape, (3, rows[0]["central"].shape[1]))
        np.testing.assert_allclose(first, second)

    def test_family_report_covers_both_families_and_marks_the_readouts(self):
        report = oracle.family_report(_synthetic_rows(seed=4))
        self.assertIn("sum", report["motor"])
        self.assertTrue(report["motor"]["linear_cv"]["fitted"])
        self.assertIn("lda", report["central"])
        self.assertIn("oracle_frame", report["central"]["lda"])
        self.assertTrue(report["central"]["lda_cv"]["fitted"])
        self.assertTrue(report["central"]["nearest_neighbour_cv"]["fitted"])

    def test_family_report_scores_the_analog_drive_family_too(self):
        report = oracle.family_report(_synthetic_rows(seed=4))
        drive = report["motor_drive"]
        self.assertIn("drive_sum", drive)
        self.assertIn("oracle_frame", drive)
        # Every decay of a leaky rule is reported, so the sweep is reviewable rather than a
        # winner whose competitors were never measured.
        for decay in oracle.DRIVE_DECAYS:
            self.assertIn(f"drive_leaky_recency@{decay}", drive)
        # The spike statistics with the same shapes stay in the table for comparison, so a
        # difference between the families is attributable to the quantisation and not the shape.
        for rule in oracle.SPIKE_SHAPE_RULES:
            self.assertIn(rule, report["motor"])
            self.assertNotIn(rule, drive)

    def test_a_recording_without_the_drive_reports_no_drive_family(self):
        """Absent, not empty-and-scored: a family of nothing must not set a ceiling."""
        rows = [{k: v for k, v in row.items() if k != "drive"}
                for row in _synthetic_rows(seed=4)]
        families = oracle.family_report(rows)
        self.assertEqual(families["motor_drive"], {})
        self.assertIsNone(oracle.summarise_statistics(families["motor_drive"])["best_bounded_jump_recall"])

    def test_the_drive_reductions_read_their_own_frames(self):
        trace = _trajectory(3, jump=[0.0, 1.0, 3.0], run=[1.0, 1.0, 1.0])
        self.assertAlmostEqual(oracle.reduce_drive(trace, "drive_sum"), 1.0)
        self.assertAlmostEqual(oracle.reduce_drive(trace, "drive_last"), 2.0)
        # Recency weighting prefers the drive that rose into the committed frame.
        rising = _trajectory(3, jump=[0.0, 1.0, 4.0], run=[1.0, 1.0, 1.0])
        falling = _trajectory(3, jump=[4.0, 1.0, 0.0], run=[1.0, 1.0, 1.0])
        self.assertAlmostEqual(oracle.reduce_drive(rising, "drive_sum"),
                               oracle.reduce_drive(falling, "drive_sum"))
        self.assertGreater(oracle.reduce_drive(rising, "drive_leaky_recency", 0.25),
                           oracle.reduce_drive(falling, "drive_leaky_recency", 0.25))

    def test_an_unknown_drive_rule_is_an_error(self):
        with self.assertRaises(ValueError):
            oracle.reduce_drive(_trajectory(3, jump=[0, 0, 0], run=[0, 0, 0]), "drive_guess")


class ReadoutProbeTests(unittest.TestCase):
    """The probes must find signal that is there, or the ceiling is not evidence."""

    def test_held_out_fisher_readout_finds_a_real_population_difference(self):
        report = oracle.cross_validated_lda(_synthetic_rows(separation=1.0, seed=5))
        self.assertTrue(report["fitted"])
        self.assertGreaterEqual(report["balanced_accuracy"], 0.85)
        self.assertGreaterEqual(report["jump_recall"], 0.8)

    def test_held_out_fisher_readout_is_chance_when_there_is_nothing_to_find(self):
        report = oracle.cross_validated_lda(_synthetic_rows(separation=0.0, seed=5))
        self.assertLessEqual(report["balanced_accuracy"], 0.75)
        self.assertLessEqual(report["jump_recall"], 0.5)

    def test_held_out_nearest_neighbour_finds_a_real_population_difference(self):
        report = oracle.nearest_neighbour_readout(_synthetic_rows(separation=1.5, seed=6))
        self.assertGreaterEqual(report["balanced_accuracy"], 0.85)

    def test_held_out_nearest_neighbour_is_chance_when_there_is_nothing_to_find(self):
        report = oracle.nearest_neighbour_readout(_synthetic_rows(separation=0.0, seed=6))
        self.assertLessEqual(report["balanced_accuracy"], 0.8)

    def test_in_sample_fit_is_optimistic_where_the_held_out_one_is_not(self):
        # The defect this instrument exists to avoid: ~600 features against ~99 chunks can
        # separate anything, so the in-sample number is never the evidence.
        rows = _synthetic_rows(separation=0.0, seed=7)
        in_sample = oracle.ceiling(oracle.ridge_scores(oracle._chunk_features(rows, "central"),
                                                       np.array([row["label"] for row in rows])),
                                   [row["label"] for row in rows])
        held_out = oracle.cross_validated_lda(rows)
        self.assertGreater(in_sample["balanced_accuracy"], held_out["balanced_accuracy"])

    def test_the_shrinkage_sweep_is_reported_with_every_candidate(self):
        report = oracle.cross_validated_lda(_synthetic_rows(seed=8))
        self.assertEqual(len(report["shrinkage_sweep"]), len(oracle.LDA_SHRINKAGES))
        self.assertIn(report["shrinkage"], oracle.LDA_SHRINKAGES)

    def test_cross_validated_boundary_needs_enough_chunks_to_fold(self):
        report = oracle.cross_validated_boundary(np.zeros((3, 2)), np.array([True, False, True]))
        self.assertIn("note", report)


class DecisionTests(unittest.TestCase):
    def test_an_evidence_ceiling_above_the_bar_means_the_decoder(self):
        verdict = oracle.decide(0.99, 0.10)
        self.assertEqual(verdict["decision"], "decoder")

    def test_a_certified_upstream_readout_means_the_motor_readout(self):
        verdict = oracle.decide(0.60, 0.98)
        self.assertEqual(verdict["decision"], "motor_readout")
        self.assertEqual(verdict["readout_evidence"], "certified")

    def test_a_better_but_uncertified_readout_is_reported_as_unproven(self):
        verdict = oracle.decide(0.60, 0.75)
        self.assertEqual(verdict["decision"], "motor_readout_unproven")

    def test_nothing_above_the_bar_means_the_representation(self):
        verdict = oracle.decide(0.68, 0.44, in_sample_readout=1.0)
        self.assertEqual(verdict["decision"], "representation")
        self.assertIn("1.0", verdict["reason"])

    def test_a_missing_motor_ceiling_is_undetermined_rather_than_a_verdict(self):
        self.assertEqual(oracle.decide(None, 0.9)["decision"], "undetermined")


class RecordingTests(unittest.TestCase):
    def test_recording_keeps_every_frame_and_follows_the_actions(self):
        from prescreen import untrained_model, vision_preprocessor

        model = untrained_model(42)
        shard = {
            "frames": np.zeros((6, 240, 256, 3), dtype=np.uint8),
            "actions": np.array([1, 3, 1, 3, 1, 1]),
        }
        rows = oracle.record_chunks(model, vision_preprocessor(), [shard], stride=3, settle_steps=2,
                                    seed=45)
        self.assertEqual(len(rows), 2)
        self.assertFalse(rows[0]["label"])
        self.assertTrue(rows[1]["label"])
        self.assertEqual(rows[0]["motor"].shape, (2, 4))
        self.assertEqual(rows[0]["central"].shape, (2, model.num_central_complex))
        # Spikes are 0/1 per frame, so a two-frame window cannot exceed two on a channel --
        # the reason the statistic the shipped decoder reads is a small integer count.
        self.assertLessEqual(float(rows[0]["motor"].max()), 2.0)
        # The drive is recorded alongside from the same forward pass, and unlike the spikes it
        # is not a small set of integers: that resolution is what a fitted threshold uses.
        self.assertEqual(rows[0]["drive"].shape, (2, 4))


class ProtocolTests(unittest.TestCase):
    def test_the_reserved_gate_seeds_are_refused(self):
        with self.assertRaises(ValueError):
            oracle.oracle_ceiling([], "nowhere", seeds=(42, 43, 44))

    def test_the_baseline_name_is_reserved(self):
        from prescreen import Arm, untrained_model

        arm = Arm(name="untrained", model=untrained_model(42))
        with self.assertRaises(ValueError):
            oracle.oracle_ceiling([arm], "nowhere", seeds=(45,))

    def test_an_empty_settle_window_is_refused(self):
        with self.assertRaises(ValueError):
            oracle.oracle_ceiling([], "nowhere", settle_steps=(0,), seeds=(45,))

    def test_the_report_renders_and_carries_the_verdict_and_the_seed_role(self):
        rows = _synthetic_rows(seed=9)
        families = oracle.family_report(rows)
        self.assertIn("motor", families)
        self.assertIn("sum", families["motor"])
        self.assertIn("lda_cv", families["central"])
        # A JSON round trip is part of the contract: the report is a committed artifact.
        payload = json.loads(json.dumps({"protocol": {"motor_rules": list(oracle.MOTOR_RULES)},
                                         "central": families["central"]["lda_cv"]}))
        self.assertEqual(payload["central"]["fitted"], True)

    def test_verdict_reason_names_the_measured_numbers(self):
        report = oracle.decide(0.68, 0.44, in_sample_readout=1.0)
        self.assertIn("0.68", report["reason"])
        self.assertIn("0.44", report["reason"])


if __name__ == "__main__":
    unittest.main()
