"""Tests for the emulator-free pre-screen that gates an emulator run.

The measurement it replaced scored calibrated balanced accuracy over pooled replays,
which was unbounded by construction (it refitted the boundary on the evidence it then
scored), insensitive (an untrained network landed inside the trained range), and silent
about per-replay variance. These tests pin the properties that make the replacement
usable as a gate: a budget rather than a fit, one row per replay rather than a pooled
figure, seed-wise pairing rather than two independent columns, and an untrained baseline
that is always present and always measured.
"""

import os
import tempfile
import unittest

import numpy as np

import torch

from macro_decoder import decision_quality
from pretrain import pretrain_motor_layer, save_checkpoint
from prescreen import (
    PRESCREEN_MIN_JUMP_RECALL,
    PRESCREEN_MIN_RECALL_DELTA,
    UNTRAINED_ARM,
    Arm,
    arm_from_checkpoint,
    arm_report,
    calibration_record,
    collect_replays,
    format_table,
    paired_deltas,
    pooled_evidence,
    prescreen,
    prescreen_verdict,
    recall_at_bounded_rate,
    summarise_arm,
    untrained_model,
)
from trajectory import iter_dataset, write_shard


class TestBoundedRateRecall(unittest.TestCase):
    """Jump-chunk recall inside a jump budget: a definition, not a fitted threshold."""

    #: 5 jump chunks, a perfect evidence ordering for them, and 15 run chunks below.
    JUMP = [4.0, 3.5, 3.0, 2.5, 2.0]
    RUN = [1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0, -3.5, -4.0, -4.5, -5.0, -5.5]

    def _labelled(self):
        return self.JUMP + self.RUN, [True] * len(self.JUMP) + [False] * len(self.RUN)

    def test_a_perfect_ranking_catches_every_jump_chunk_inside_the_budget(self):
        diffs, labels = self._labelled()

        report = recall_at_bounded_rate(diffs, labels, 0.35)

        # floor(0.35 * 20) = 7 decisions may be jumps; all 5 jump chunks fit inside them.
        self.assertEqual(report["budget_chunks"], 7)
        self.assertEqual(report["jump_chunks_caught"], 5)
        self.assertEqual(report["jump_recall"], 1.0)
        self.assertEqual(report["jump_rate"], 0.35)
        self.assertEqual(report["teacher_jump_rate"], 0.25)

    def test_the_budget_beats_every_fixed_margin_inside_it(self):
        """No threshold spending the same rate recalls more -- the budget is the ceiling."""
        diffs, labels = self._labelled()
        budget = recall_at_bounded_rate(diffs, labels, 0.35)

        for margin in sorted(set(diffs)):
            fixed = decision_quality(diffs, labels, margin)
            if fixed["jump_rate"] <= 0.35:
                self.assertLessEqual(fixed["jump_recall"], budget["jump_recall"], margin)

    def test_the_reported_margin_realizes_the_budget(self):
        diffs, labels = self._labelled()

        report = recall_at_bounded_rate(diffs, labels, 0.35)

        # The 7th and 8th largest decisions are 1.0 and 0.5, so the margin sits between
        # them and the strict ``>`` rule spends exactly the 7-decision budget. Placing it
        # just above the first excluded value instead would spend only 6 here.
        self.assertEqual(report["margin"], 0.75)
        self.assertEqual(report["margin_jump_rate"], report["jump_rate"])
        self.assertEqual(report["jump_chunks_caught"], 5)

    def test_a_budget_of_zero_still_reports_a_margin_that_never_jumps(self):
        diffs, labels = self._labelled()

        report = recall_at_bounded_rate(diffs, labels, 0.0)

        self.assertEqual(report["jump_rate"], 0.0)
        self.assertEqual(report["margin"], max(diffs))
        self.assertEqual(report["margin_jump_rate"], 0.0)
        self.assertEqual(report["jump_recall"], 0.0)

    def test_a_budget_covering_every_decision_reports_no_margin(self):
        diffs, labels = self._labelled()

        report = recall_at_bounded_rate(diffs, labels, 1.0)

        self.assertIsNone(report["margin"])
        self.assertEqual(report["jump_recall"], 1.0)

    def test_chance_is_the_rate_it_spends(self):
        """An uninformative ordering catches the rate, so the report says what that rate is."""
        rng = np.random.RandomState(0)
        diffs = [float(value) for value in rng.randn(100)]
        labels = [index < 25 for index in range(100)]

        report = recall_at_bounded_rate(diffs, labels, 0.35)

        self.assertEqual(report["random_recall"], report["jump_rate"])
        self.assertAlmostEqual(
            report["lift_over_random"], report["jump_recall"] - report["jump_rate"], places=4
        )
        # Nowhere near a perfect ranking: this is the floor every arm has to clear.
        self.assertLess(report["jump_recall"], 0.75)

    def test_more_budget_never_buys_less_recall(self):
        diffs, labels = self._labelled()

        recalls = [
            recall_at_bounded_rate(diffs, labels, cap)["jump_recall"]
            for cap in (0.0, 0.1, 0.25, 0.5, 1.0)
        ]

        self.assertEqual(recalls, sorted(recalls))
        self.assertEqual(recalls[0], 0.0)
        self.assertEqual(recalls[-1], 1.0)

    def test_an_unmeasurable_recall_is_reported_as_none_not_zero(self):
        """A replay without a jump chunk has no recall; 0.0 would read as a measured failure."""
        no_jumps = recall_at_bounded_rate([-1.0, -2.0], [False, False], 0.35)
        self.assertIsNone(no_jumps["jump_recall"])
        self.assertEqual(no_jumps["jump_chunks"], 0)

        empty = recall_at_bounded_rate([], [], 0.35)
        self.assertIsNone(empty["jump_recall"])
        self.assertIsNone(empty["jump_rate"])

    def test_malformed_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            recall_at_bounded_rate([1.0], [True, False], 0.35)
        with self.assertRaises(ValueError):
            recall_at_bounded_rate([1.0], [True], 1.5)
        with self.assertRaises(ValueError):
            recall_at_bounded_rate([1.0], [True], -0.1)
        with self.assertRaises(ValueError):
            recall_at_bounded_rate([1.0], [True], float("nan"))
        with self.assertRaises(ValueError):
            recall_at_bounded_rate([1.0], [True], "0.35")
        # A bool is an int in Python; it is not a rate.
        with self.assertRaises(ValueError):
            recall_at_bounded_rate([1.0], [True], True)


class TestArmReporting(unittest.TestCase):
    """One row per replay, and a headline that is not the pooled figure."""

    def _replay(self, seed, shift=5.0):
        rng = np.random.RandomState(seed)
        labels = [False] * 15 + [True] * 5
        diffs = [float(value) + (shift if label else 0.0)
                 for value, label in zip(rng.randn(20), labels)]
        return {"replay_seed": seed, "diffs": diffs, "labels": labels}

    def test_every_replay_is_reported_and_the_headline_is_their_mean(self):
        rows = [self._replay(seed) for seed in (42, 43, 44)]

        report = summarise_arm("candidate", rows, 0.35)

        self.assertEqual([row["replay_seed"] for row in report["per_replay"]], [42, 43, 44])
        self.assertEqual(report["replays"], 3)
        self.assertEqual(report["replay_seeds"], [42, 43, 44])
        self.assertEqual(len(report["jump_recall"]["values"]), 3)
        self.assertEqual(
            report["jump_recall"]["values"],
            [row["jump_recall"] for row in report["per_replay"]],
        )
        self.assertEqual(
            report["jump_recall"]["mean"],
            round(sum(report["jump_recall"]["values"]) / 3, 4),
        )

    def test_the_pooled_view_is_reported_but_does_not_replace_the_per_replay_mean(self):
        rows = [self._replay(seed) for seed in (42, 43, 44)]

        report = summarise_arm("candidate", rows, 0.35)

        self.assertEqual(report["pooled"]["chunks"], 60)
        self.assertEqual(report["decisions_per_replay"], 20)
        # Pooling three replays is a different statistic, and it is kept as its own field.
        self.assertNotIn(report["pooled"]["jump_recall"], (None, False))

    def test_the_shipped_margin_is_measured_next_to_the_budget_figure(self):
        rows = [self._replay(seed) for seed in (42, 43)]

        report = summarise_arm("candidate", rows, 0.35, margin=99.0)

        self.assertEqual(report["at_margin"]["margin"], 99.0)
        self.assertEqual(report["at_margin"]["method"], "fixed_margin")
        self.assertEqual(report["per_replay"][0]["at_margin"]["jump_rate"], 0.0)
        # A margin nothing clears stops the arm jumping altogether, whatever the budget says.
        self.assertEqual(report["per_replay"][0]["at_margin"]["jump_recall"], 0.0)

    def test_an_arm_needs_at_least_one_replay(self):
        with self.assertRaises(ValueError):
            summarise_arm("candidate", [], 0.35)

    def test_calibration_records_the_replays_it_was_fitted_on(self):
        """The margin is only interpretable next to the measurement that produced it."""
        rows = [self._replay(seed) for seed in (42, 43)]

        calibration = calibration_record(rows)

        self.assertEqual(calibration["method"], "balanced_accuracy")
        self.assertEqual(calibration["replays"], 2)
        self.assertEqual(calibration["replay_seeds"], [42, 43])
        self.assertEqual(calibration["decisions_per_replay"], 20)
        self.assertEqual(calibration["samples"], 40)
        with self.assertRaises(ValueError):
            calibration_record([])

    def test_pooled_evidence_concatenates_every_replay(self):
        rows = [
            {"replay_seed": 1, "diffs": [1.0], "labels": [True]},
            {"replay_seed": 2, "diffs": [2.0, 3.0], "labels": [False, True]},
        ]

        diffs, labels = pooled_evidence(rows)

        self.assertEqual(diffs, [1.0, 2.0, 3.0])
        self.assertEqual(labels, [True, False, True])


class TestPairedReporting(unittest.TestCase):
    """Deltas are taken seed by seed, so a mean cannot hide a single lucky replay."""

    def _arm(self, name, recall_by_seed):
        return {
            "arm": name,
            "per_replay": [
                {"replay_seed": seed, "jump_recall": recall}
                for seed, recall in recall_by_seed.items()
            ],
        }

    def test_deltas_are_taken_on_the_shared_replay_seed(self):
        baseline = self._arm(UNTRAINED_ARM, {42: 0.2, 43: 0.2, 44: 0.2})
        arm = self._arm("candidate", {42: 0.5, 43: 0.4, 44: 0.6})

        paired = paired_deltas(baseline, arm)

        self.assertEqual([row["delta"] for row in paired["rows"]], [0.3, 0.2, 0.4])
        self.assertEqual(paired["mean_delta"], 0.3)
        self.assertEqual(paired["improved"], 3)
        self.assertTrue(paired["all_improved"])
        self.assertEqual(paired["baseline"], UNTRAINED_ARM)

    def test_a_mean_lifted_by_one_replay_is_visible(self):
        """Exactly the failure the pooled metric hid: one good draw carrying an average."""
        baseline = self._arm(UNTRAINED_ARM, {42: 0.2, 43: 0.2, 44: 0.2})
        arm = self._arm("candidate", {42: 0.8, 43: 0.1, 44: 0.1})

        paired = paired_deltas(baseline, arm)

        self.assertEqual(paired["improved"], 1)
        self.assertFalse(paired["all_improved"])
        self.assertGreater(paired["mean_delta"], 0.0)
        self.assertEqual(paired["min_delta"], -0.1)

    def test_only_shared_replay_seeds_are_compared(self):
        baseline = self._arm(UNTRAINED_ARM, {42: 0.2, 43: 0.2})
        arm = self._arm("candidate", {43: 0.5, 44: 0.9})

        paired = paired_deltas(baseline, arm)

        self.assertEqual(paired["paired_replays"], 1)
        self.assertEqual(paired["rows"][0]["replay_seed"], 43)

    def test_a_replay_without_a_measurable_recall_is_skipped(self):
        baseline = self._arm(UNTRAINED_ARM, {42: 0.2, 43: 0.2})
        absent = {"arm": "candidate", "per_replay": [
            {"replay_seed": 42, "jump_recall": None},
            {"replay_seed": 43, "jump_recall": 0.5},
        ]}
        missing_key = {"arm": "candidate", "per_replay": [
            {"replay_seed": 42},
            {"replay_seed": 43, "jump_recall": 0.5},
        ]}

        self.assertEqual(paired_deltas(baseline, absent)["paired_replays"], 1)
        self.assertEqual(paired_deltas(baseline, missing_key)["paired_replays"], 1)


class TestPrescreenVerdict(unittest.TestCase):
    """The gate: operationally useful, better than untrained, better on every replay."""

    def _arm_report(self, recalls, completion=1.0, jump_rate=0.28):
        return {
            "arm": "candidate",
            "jump_sequence_recall": {"values": list(recalls), "mean": sum(recalls) / len(recalls)},
            "jump_rate": {"values": [jump_rate], "mean": jump_rate},
            "offline_completion_rate": completion,
        }

    def _paired(self, deltas, baseline_mean=0.0):
        return {
            "arm": "candidate",
            "baseline": UNTRAINED_ARM,
            "paired_replays": len(deltas),
            "rows": [],
            "mean_delta": sum(deltas) / len(deltas),
            "min_delta": min(deltas),
            "max_delta": max(deltas),
            "improved": sum(1 for delta in deltas if delta > 0),
            "all_improved": all(delta > 0 for delta in deltas),
            "baseline_mean": baseline_mean,
            "arm_mean": baseline_mean + sum(deltas) / len(deltas),
        }

    def test_a_completed_and_consistent_gain_passes(self):
        verdict = prescreen_verdict(
            self._arm_report([0.95, 0.96, 0.95]),
            self._paired([0.35, 0.36, 0.35], baseline_mean=0.60),
        )

        self.assertTrue(verdict["pass"])
        self.assertEqual(verdict["arm"], "candidate")
        self.assertIsNone(verdict["note"])

    def test_every_criterion_is_itemised_with_the_value_it_judged(self):
        verdict = prescreen_verdict(
            self._arm_report([0.50, 0.55, 0.54], completion=0.0),
            self._paired([0.02, -0.03, 0.01], baseline_mean=0.52),
        )

        self.assertFalse(verdict["pass"])
        failed = {c["criterion"] for c in verdict["criteria"] if not c["pass"]}
        self.assertEqual(
            failed,
            {"offline_completion", "jump_sequence_recall", "beats_untrained",
             "consistent_across_replays"},
        )
        by_name = {criterion["criterion"]: criterion for criterion in verdict["criteria"]}
        self.assertEqual(by_name["offline_completion"]["threshold"], 1.0)
        self.assertEqual(by_name["offline_completion"]["observed"], 0.0)
        self.assertEqual(by_name["jump_sequence_recall"]["threshold"], PRESCREEN_MIN_JUMP_RECALL)
        self.assertAlmostEqual(by_name["jump_sequence_recall"]["observed"], 0.53, places=4)
        self.assertEqual(by_name["beats_untrained"]["threshold"], PRESCREEN_MIN_RECALL_DELTA)
        self.assertEqual(by_name["consistent_across_replays"]["observed"], "2/3")

    def test_a_missed_required_jump_disqualifies_however_good_the_recall_is(self):
        """Completion is the emulator's own bar, so a near-perfect sequence is not enough."""
        verdict = prescreen_verdict(
            self._arm_report([0.99, 0.99, 0.99], completion=0.0),
            self._paired([0.39, 0.39, 0.39], baseline_mean=0.60),
        )

        self.assertFalse(verdict["pass"])
        failed = [c["criterion"] for c in verdict["criteria"] if not c["pass"]]
        self.assertEqual(failed, ["offline_completion"])

    def test_an_always_jump_arm_is_caught_by_the_rate_budget(self):
        """The metric's simplest exploit, measured on a real checkpoint.

        A policy that jumps on every chunk covers every stretch of teacher flight, so it scores
        a perfect sequence recall and a completion -- the 30-epoch visual-pathway arm does
        exactly that (recall 1.000, completion 1.00, 49.5% of decisions spent jumping, 29
        spurious). Only the jump budget distinguishes it from an arm that jumps where the
        teacher did.
        """
        verdict = prescreen_verdict(
            self._arm_report([1.0, 1.0, 1.0], completion=1.0, jump_rate=0.495),
            self._paired([0.27, 0.27, 0.27], baseline_mean=0.733),
        )

        self.assertFalse(verdict["pass"])
        failed = [c["criterion"] for c in verdict["criteria"] if not c["pass"]]
        self.assertEqual(failed, ["jump_rate_within_budget"])
        by_name = {c["criterion"]: c for c in verdict["criteria"]}
        self.assertEqual(by_name["jump_rate_within_budget"]["threshold"], 0.35)
        self.assertEqual(by_name["jump_rate_within_budget"]["observed"], 0.495)

    def test_an_unmeasurable_jump_rate_fails_rather_than_passing(self):
        verdict = prescreen_verdict(
            self._arm_report([0.95, 0.95, 0.95], jump_rate=None),
            self._paired([0.20, 0.20, 0.20], baseline_mean=0.60),
        )

        self.assertFalse(verdict["pass"])

    def test_the_bar_rises_above_the_floor_when_the_baseline_is_strong(self):
        """A strong untrained arm has to be beaten by the minimum gain, not merely matched."""
        verdict = prescreen_verdict(
            self._arm_report([0.93, 0.93, 0.93]),
            self._paired([0.05, 0.05, 0.05], baseline_mean=0.88),
        )

        bar = [c for c in verdict["criteria"] if c["criterion"] == "jump_sequence_recall"][0]
        self.assertEqual(bar["threshold"], 0.88 + PRESCREEN_MIN_RECALL_DELTA)
        self.assertFalse(bar["pass"])

    def test_an_unmeasurable_arm_fails_rather_than_passing_by_default(self):
        verdict = prescreen_verdict(
            {"arm": "candidate", "jump_sequence_recall": {"values": [], "mean": None},
             "jump_rate": {"values": [], "mean": None}, "offline_completion_rate": 0.0},
            self._paired([0.30]),
        )

        self.assertFalse(verdict["pass"])

    def test_no_baseline_means_no_paired_comparison_and_no_pass(self):
        verdict = prescreen_verdict(self._arm_report([0.95, 0.95, 0.95]), None)

        self.assertFalse(verdict["pass"])
        self.assertIn("no untrained baseline", verdict["note"])


def teacher_ram(samples, takeoff_at=5):
    """RAM for a teacher that walks right and jumps once.

    ``0x001D`` is the vertical state, so 1 then 2 is a jump the level required and a lone 2
    would be a fall. Without this the shard has no required jumps and the sequence metric is
    undefined for every arm -- which is a property worth having a test for, not the default.
    """
    rows = []
    for index in range(samples):
        row = np.zeros(0x800, dtype=np.uint8)
        x = 40 + index * 110
        row[0x006D] = (x // 256) % 256
        row[0x0086] = x % 256
        row[0x001D] = 1 if index == takeoff_at else (2 if index == takeoff_at + 1 else 0)
        rows.append(row)
    return rows


class TestPrescreenTable(unittest.TestCase):
    """End to end on a real (synthetic) shard: the baseline is never optional."""

    SAMPLES = 12
    ACTIONS = [1, 3, 1, 1, 3, 1, 3, 3, 1, 1, 3, 1]

    def _shard(self, dataset_dir, env_kind="stable-retro"):
        frames = [
            np.random.RandomState(index).randint(0, 255, (64, 64, 3), dtype=np.uint8)
            for index in range(self.SAMPLES)
        ]
        write_shard(dataset_dir, frames, list(self.ACTIONS), teacher_ram(self.SAMPLES),
                    [False] * self.SAMPLES, [False] * self.SAMPLES, {"env_kind": env_kind})
        return dataset_dir

    def test_every_table_carries_a_measured_untrained_baseline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._shard(tmpdir)

            report = prescreen(
                [Arm("candidate", untrained_model(seed=42))], tmpdir,
                stride=1, settle_steps=1, seed=42, replays=2,
            )

            self.assertEqual([row["arm"] for row in report["table"]], [UNTRAINED_ARM, "candidate"])
            self.assertEqual(report["table"][0]["role"], "untrained_baseline")
            self.assertEqual(report["table"][1]["role"], "candidate")
            baseline = report["table"][0]
            self.assertEqual(baseline["replays"], 2)
            self.assertIsNotNone(baseline["offline_best_x"]["mean"])
            self.assertIsNotNone(baseline["budget"]["jump_recall"]["mean"])
            # A model compared against its own initialization is the null arm: a delta of
            # exactly zero, and no verdict it could honestly pass.
            self.assertEqual(report["paired"][0]["mean_delta"], 0.0)
            self.assertEqual(report["paired"][0]["improved"], 0)
            self.assertFalse(report["verdicts"][0]["pass"])

    def test_the_reserved_baseline_name_cannot_be_a_candidate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._shard(tmpdir)

            with self.assertRaises(ValueError):
                prescreen(
                    [Arm(UNTRAINED_ARM, untrained_model(seed=42))], tmpdir,
                    stride=1, settle_steps=1, replays=1,
                )

    def test_replays_are_shared_seeds_and_a_seed_reproduces_the_same_draws(self):
        """Pairing means nothing unless a seed reproduces the same decisions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._shard(tmpdir)
            shards = list(iter_dataset(tmpdir))
            model = untrained_model(seed=42)

            first = collect_replays(model, shards, stride=1, settle_steps=1, seed=7, replays=2)
            second = collect_replays(model, shards, stride=1, settle_steps=1, seed=7, replays=2)

            self.assertEqual([row["replay_seed"] for row in first], [7, 8])
            for left, right in zip(first, second):
                self.assertEqual(left["diffs"], right["diffs"])
                self.assertEqual(left["labels"], right["labels"])
            # And the replays are genuinely independent draws, not one measurement twice.
            self.assertNotEqual(first[0]["diffs"], first[1]["diffs"])

    def test_two_arms_are_measured_on_the_same_decision_points(self):
        """Different weights under one seed: same chunks, same labels, different evidence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._shard(tmpdir)
            shards = list(iter_dataset(tmpdir))

            left = collect_replays(untrained_model(seed=1), shards, 1, 1, 11, 1)[0]
            right = collect_replays(untrained_model(seed=2), shards, 1, 1, 11, 1)[0]

            self.assertEqual(left["labels"], right["labels"])
            self.assertEqual(left["diffs"] != right["diffs"], True)

    def test_the_table_pairs_an_episode_outcome_with_the_budget_view(self):
        """Two views of one dataset: the sequence outcome gates, the budget supports it."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._shard(tmpdir)

            report = prescreen(
                [Arm("candidate", untrained_model(seed=42))], tmpdir,
                stride=1, settle_steps=1, seed=42, replays=2,
            )
            row = report["table"][1]

            self.assertEqual(row["metric"], "teacher_forced_sequence")
            self.assertEqual(len(row["per_replay"]), 2)
            for replay_row in row["per_replay"]:
                for key in ("offline_best_x", "offline_completion", "missed_jumps",
                            "spurious_jumps", "jump_rate", "jump_sequence_recall"):
                    self.assertIn(key, replay_row)
            self.assertEqual(row["teacher"]["shards"], 1)
            # One takeoff in the shard, and the shard stops well short of the flagpole, so
            # no replay can be credited with a completion.
            self.assertEqual(row["teacher"]["required_jumps"], 1)
            self.assertEqual(row["offline_completion_rate"], 0.0)
            self.assertIsNotNone(row["jump_sequence_recall"]["mean"])
            self.assertEqual(row["budget"]["replays"], 2)
            self.assertEqual(report["paired"][0]["metric"], "jump_sequence_recall")
            self.assertEqual(report["paired_budget"][0]["metric"], "jump_recall")

    def test_a_dataset_with_no_teacher_jumps_says_why_it_cannot_be_compared(self):
        """A shard without a single takeoff is unmeasurable, and the report has to say so."""
        with tempfile.TemporaryDirectory() as tmpdir:
            frames = [
                np.random.RandomState(index).randint(0, 255, (64, 64, 3), dtype=np.uint8)
                for index in range(self.SAMPLES)
            ]
            write_shard(tmpdir, frames, list(self.ACTIONS),
                        [np.zeros(0x800, dtype=np.uint8) for _ in range(self.SAMPLES)],
                        [False] * self.SAMPLES, [False] * self.SAMPLES, {"env_kind": "stable-retro"})

            report = prescreen(
                [Arm("candidate", untrained_model(seed=42))], tmpdir,
                stride=1, settle_steps=1, seed=42, replays=2,
            )

            self.assertIsNone(report["table"][1]["jump_sequence_recall"]["mean"])
            self.assertEqual(report["paired"][0]["paired_replays"], 0)
            self.assertIn("no teacher jumps", report["verdicts"][0]["note"])
            self.assertFalse(report["verdicts"][0]["pass"])

    def test_the_shipped_margin_is_applied_not_refitted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._shard(tmpdir)

            report = arm_report(
                Arm("candidate", untrained_model(seed=42), margin=99.0), tmpdir, 1, 1, 42, 2
            )

            self.assertEqual(report["margin"], 99.0)
            self.assertEqual(report["at_margin"]["margin"], 99.0)
            self.assertEqual(report["at_margin"]["jump_rate"], 0.0)
            self.assertEqual(report["at_margin"]["jump_recall"], 0.0)
            # The bounded-rate figure is unaffected by the margin: it is a budget, not a fit.
            self.assertGreater(report["jump_recall"]["mean"], 0.0)

    def test_the_report_renders_and_records_its_protocol(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._shard(tmpdir)

            report = prescreen(
                [Arm("candidate", untrained_model(seed=42), margin=0.0)], tmpdir,
                stride=1, settle_steps=1, seed=42, replays=2,
            )
            text = format_table(report)

            self.assertIn(UNTRAINED_ARM, text)
            self.assertIn("candidate", text)
            self.assertIn("chance", text)
            self.assertEqual(report["protocol"]["replay_seeds"], [42, 43])
            self.assertEqual(report["protocol"]["stride"], 1)
            self.assertEqual(report["protocol"]["settle_steps"], 1)
            self.assertEqual(report["protocol"]["baseline"], UNTRAINED_ARM)
            self.assertEqual(report["protocol"]["decisions_per_replay"], self.SAMPLES)
            self.assertEqual(report["protocol"]["metric"], "teacher_forced_sequence")
            self.assertEqual(
                report["protocol"]["thresholds"]["min_jump_recall"], PRESCREEN_MIN_JUMP_RECALL
            )

    def test_replays_and_stride_are_validated(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._shard(tmpdir)
            shards = list(iter_dataset(tmpdir))

            with self.assertRaises(ValueError):
                collect_replays(untrained_model(seed=42), shards, 1, 1, 42, 0)
            with self.assertRaises(ValueError):
                prescreen([], tmpdir, stride=0, settle_steps=1, replays=1)

    def test_a_checkpoint_arm_runs_the_decoder_the_controller_would_run(self):
        """A pre-screened arm has to be the shipped one, not a rebuilt approximation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._shard(tmpdir)
            model, metadata = pretrain_motor_layer(
                tmpdir, epochs=1, settle_steps=1, stride=1, seed=42
            )
            checkpoint = os.path.join(tmpdir, "checkpoint.pth")
            save_checkpoint(model, checkpoint, metadata)

            arm = arm_from_checkpoint("candidate", checkpoint)
            report = prescreen([arm], tmpdir, stride=1, settle_steps=1, seed=42, replays=2)

            self.assertEqual(arm.margin, metadata["jump_margin"])
            self.assertEqual(report["table"][1]["budget"]["margin"], metadata["jump_margin"])
            # The replay runs the decoder the checkpoint ships, chunk lengths included.
            self.assertEqual(arm.decoder["chunk_frames"], 1)
            self.assertEqual(report["table"][1]["decoder"]["chunk_frames"], 1)
            self.assertEqual(arm.detail["checkpoint"], os.path.abspath(checkpoint))
            self.assertEqual(arm.detail["visual_pathway"], metadata["visual_pathway"])
            self.assertEqual(report["table"][1]["detail"]["learning_rate"], metadata["learning_rate"])


if __name__ == "__main__":
    unittest.main()
