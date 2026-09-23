"""Tests for the sweep merge and its cross-host drift check.

A sweep summary is only as good as its key: if two runs that differ in the statistic, the
window, or the data collide, the comparison table silently averages a measurement into
another one, and if a run's artifact is missing the summary reads as though it was never
supposed to be there. These tests pin the key, the loud failure on a malformed artifact, and
the difference between "the number moved" and "the number moved too far".
"""

import json
import os
import tempfile
import unittest

from sweep import (
    METRICS,
    cell_key,
    compare,
    flatten,
    forced_statistic,
    load_table,
    main,
    render,
    row_key,
    shard_checksums,
    statistic_of,
)

SHARD_A = "a" * 64
SHARD_B = "b" * 64
SEEDS = [45, 46, 47, 48, 49]


def arm_row(arm, recall, budget=0.4, rate=0.2, spurious=6.0, best_x=700.0, completion=0.0,
            rule=None, decay=None, role=None):
    row = {
        "arm": arm,
        "role": role or ("untrained_baseline" if arm == "untrained" else "candidate"),
        "jump_sequence_recall": {"mean": recall, "min": recall, "max": recall},
        "jump_rate": {"mean": rate},
        "spurious_jumps": {"mean": spurious},
        "offline_best_x": {"mean": best_x},
        "offline_completion_rate": completion,
        "budget": {"jump_recall": {"mean": budget}},
    }
    if rule:
        row["evidence"] = {"evidence_rule": rule, "evidence_decay": decay}
    return row


def table(rows, sha=SHARD_A, override=None, seeds=SEEDS, settle=5, stride=15, role="dev",
          paired=None):
    protocol = {
        "dataset_provenance": {"shards": [{"sha256": sha, "samples": 1477}]},
        "stride": stride,
        "settle_steps": settle,
        "replay_seeds": list(seeds),
        "seed_role": role,
        "evidence_override": override,
    }
    return {"protocol": protocol, "table": rows, "paired": paired or [], "verdicts": []}


def override(rule, decay=0.25):
    return {"evidence_rule": rule, "evidence_decay": decay, "forced_on_every_arm": True}


def write(directory, name, report):
    path = os.path.join(directory, name)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle)
    return path


class TestTableLoading(unittest.TestCase):

    def test_something_that_is_not_a_table_is_refused_with_its_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = write(tmpdir, "notes.json", {"hello": "world"})

            with self.assertRaises(ValueError) as caught:
                load_table(path)

            self.assertIn("notes.json", str(caught.exception))

    def test_a_table_without_a_data_identity_cannot_key_itself(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report = table([arm_row("untrained", 0.7)])
            report["protocol"]["dataset_provenance"] = {}

            with self.assertRaises(ValueError) as caught:
                shard_checksums(report["protocol"])

            self.assertIn("checksums", str(caught.exception))

    def test_a_forced_statistic_is_read_from_the_protocol_and_absent_without_one(self):
        self.assertIsNone(forced_statistic(table([])["protocol"]))
        self.assertEqual(forced_statistic(table([], override=override("drive_sum"))["protocol"]),
                         "drive_sum@0.25")

    def test_a_row_without_a_recorded_statistic_is_labelled_rather_than_defaulted(self):
        """A table from before the statistic was recorded is a different measurement."""
        report = table([arm_row("untrained", 0.7)])

        self.assertEqual(statistic_of(report["protocol"], report["table"][0]), "unspecified")
        report = table([arm_row("untrained", 0.7, rule="spike_sum", decay=0.25)])
        self.assertEqual(statistic_of(report["protocol"], report["table"][0]), "spike_sum@0.25")


class TestCellKeys(unittest.TestCase):

    def test_the_same_protocol_on_the_same_data_is_the_same_cell(self):
        first = flatten([table([arm_row("untrained", 0.7)])])
        second = flatten([table([arm_row("untrained", 0.7)])])

        self.assertEqual(first[0]["cell_key"], second[0]["cell_key"])

    def test_a_different_forced_statistic_is_a_different_cell(self):
        spike = flatten([table([arm_row("round1", 0.84)], override=override("spike_sum"))])
        drive = flatten([table([arm_row("round1", 0.74)], override=override("drive_sum"))])

        self.assertNotEqual(spike[0]["cell_key"], drive[0]["cell_key"])

    def test_a_different_shard_is_a_different_cell(self):
        left = flatten([table([arm_row("round1", 0.84)], sha=SHARD_A)])
        right = flatten([table([arm_row("round1", 0.84)], sha=SHARD_B)])

        self.assertNotEqual(left[0]["cell_key"], right[0]["cell_key"])

    def test_a_different_window_or_seed_set_is_a_different_cell(self):
        base = flatten([table([arm_row("round1", 0.84)])])
        window = flatten([table([arm_row("round1", 0.84)], settle=10)])
        seeds = flatten([table([arm_row("round1", 0.84)], seeds=[45, 46, 47])])

        self.assertNotEqual(base[0]["cell_key"], window[0]["cell_key"])
        self.assertNotEqual(base[0]["cell_key"], seeds[0]["cell_key"])

    def test_seed_role_separates_tuning_readings_from_claim_readings(self):
        """The same numbers on the reserved seeds are not the same cell as on the dev seeds."""
        dev = flatten([table([arm_row("round1", 0.84)], role="dev")])
        gate = flatten([table([arm_row("round1", 0.84)], role="gate")])

        self.assertNotEqual(dev[0]["cell_key"], gate[0]["cell_key"])
        self.assertEqual(gate[0]["seed_role"], "gate")

    def test_two_per_arm_runs_that_carried_different_rules_are_different_rows(self):
        """Same protocol and same arm name, different statistic: not the same measurement."""
        spike = flatten([table([arm_row("round1", 0.84, rule="spike_sum", decay=0.25)])])
        drive = flatten([table([arm_row("round1", 0.54, rule="drive_leaky_recency", decay=0.25)])])

        self.assertEqual(spike[0]["cell_key"], drive[0]["cell_key"])
        self.assertNotEqual(row_key(spike[0]), row_key(drive[0]))

    def test_the_arm_set_is_not_part_of_a_row_key(self):
        """A partial run is comparable with the rows it does share.

        This is what makes a baseline-only dispatch usable as a reproduce check against a
        published table that also carries checkpoints. The statistic is in the key, so the
        shortcut cannot confuse an arm measured one way with the same arm measured another.
        """
        partial = flatten([table([arm_row("untrained", 0.66)], override=override("spike_sum"))])
        whole = flatten([table([arm_row("untrained", 0.66), arm_row("round1", 0.84)],
                               override=override("spike_sum"))])

        self.assertEqual(row_key(partial[0]), row_key(whole[0]))


class TestFlatten(unittest.TestCase):

    def test_every_arm_of_every_table_becomes_a_row(self):
        reports = [table([arm_row("untrained", 0.66), arm_row("round1", 0.84)],
                         override=override("spike_sum")),
                   table([arm_row("untrained", 0.96), arm_row("round1", 0.74)],
                         override=override("drive_sum"))]

        rows = flatten(reports)

        self.assertEqual(len(rows), 4)
        self.assertEqual({row["arm"] for row in rows}, {"untrained", "round1"})

    def test_the_untrained_baseline_leads_each_cell(self):
        rows = flatten([table([arm_row("round1", 0.84), arm_row("untrained", 0.66)],
                              override=override("spike_sum"))])

        self.assertEqual([row["arm"] for row in rows], ["untrained", "round1"])

    def test_the_paired_delta_and_verdict_travel_with_their_arm(self):
        report = table(
            [arm_row("untrained", 0.66), arm_row("round1", 0.84)],
            paired=[{"arm": "round1", "mean_delta": 0.18, "improved": 5, "paired_replays": 5}],
        )
        report["verdicts"] = [{"arm": "round1", "pass": False}]

        rows = {row["arm"]: row for row in flatten([report])}

        self.assertEqual(rows["round1"]["paired_delta"], 0.18)
        self.assertEqual(rows["round1"]["paired_improved"], 5)
        self.assertFalse(rows["round1"]["verdict"])
        self.assertIsNone(rows["untrained"]["paired_delta"])


class TestRendering(unittest.TestCase):

    def test_the_rendered_table_names_the_statistic_the_role_and_every_quoted_number(self):
        report = table([arm_row("untrained", 0.66, rule="spike_sum", decay=0.25),
                        arm_row("round1", 0.84, rule="spike_sum", decay=0.25)],
                       paired=[{"arm": "round1", "mean_delta": 0.18, "improved": 5,
                                "paired_replays": 5}])
        report["verdicts"] = [{"arm": "round1", "pass": False}]

        text = render(flatten([report]))

        self.assertIn("spike_sum@0.25", text)
        self.assertIn("0.840", text)
        self.assertIn("0.180 (5/5)", text)
        self.assertIn("fail", text)

    def test_a_missing_number_renders_as_a_dash_rather_than_a_zero(self):
        row = arm_row("round1", 0.84)
        row["jump_sequence_recall"]["mean"] = None

        text = render(flatten([table([row])]))

        self.assertIn("| - |", text)


class TestDrift(unittest.TestCase):

    def _cell(self, recall, budget=0.4, rate=0.2, completion=0.0, override_rule="spike_sum"):
        return flatten([table([arm_row("untrained", 0.66), arm_row("round1", recall,
                                                                  budget=budget, rate=rate,
                                                                  completion=completion)],
                              override=override(override_rule))])

    def test_an_identical_sweep_does_not_drift(self):
        report = compare(self._cell(0.84), self._cell(0.84), tolerance=0.0)

        self.assertTrue(report["ok"])
        self.assertEqual(report["compared"], 2)
        self.assertEqual(report["moved"], [])

    def test_a_moved_number_is_reported_with_both_values_and_the_delta(self):
        report = compare(self._cell(0.54), self._cell(0.84), tolerance=0.01)

        self.assertFalse(report["ok"])
        moved = report["moved"][0]
        self.assertEqual(moved["arm"], "round1")
        detail = moved["metrics"]["jump_sequence_recall"]
        self.assertEqual(detail["reference"], 0.84)
        self.assertEqual(detail["current"], 0.54)
        self.assertAlmostEqual(detail["delta"], -0.30)

    def test_a_move_inside_the_tolerance_is_not_drift(self):
        report = compare(self._cell(0.845), self._cell(0.84), tolerance=0.01)

        self.assertTrue(report["ok"])
        self.assertEqual(report["moved"], [])

    def test_zero_tolerance_demands_an_exact_reproduction(self):
        report = compare(self._cell(0.845), self._cell(0.84), tolerance=0.0)

        self.assertFalse(report["ok"])
        self.assertTrue(report["moved"][0]["metrics"]["jump_sequence_recall"]["moved"])

    def test_completion_is_compared_exactly_not_with_a_tolerance(self):
        """A replay count is not a float: 0.4 of a completed replay is not a thing."""
        report = compare(self._cell(0.84, completion=0.6), self._cell(0.84, completion=0.4),
                         tolerance=0.5)

        self.assertFalse(report["ok"])
        self.assertTrue(report["moved"][0]["metrics"]["offline_completion_rate"]["moved"])

    def test_a_run_missing_from_this_sweep_is_reported_rather_than_ignored(self):
        reference = self._cell(0.84)
        current = [row for row in reference if row["arm"] != "round1"]

        report = compare(current, reference, tolerance=0.01)

        self.assertTrue(report["ok"])
        self.assertEqual([entry["arm"] for entry in report["missing"]], ["round1"])

    def test_a_sweep_the_reference_cannot_address_is_not_a_pass(self):
        """Nothing compared is not a clean sweep, it is an unchecked one.

        The usual cause is that the data differs, since the shard checksum is part of the key,
        and reporting that as a pass is the silent failure every other guard here prevents.
        """
        report = compare(self._cell(0.84, override_rule="drive_sum"), self._cell(0.84),
                         tolerance=0.01)

        self.assertFalse(report["ok"])
        self.assertTrue(report["unaddressed"])
        self.assertEqual(report["compared"], 0)
        self.assertIn("shard checksums", report["reason"])
        self.assertNotEqual(report["reference_data"], report["current_data"])

    def test_a_row_the_reference_does_not_contain_is_reported_as_extra(self):
        reference = self._cell(0.84)
        current = flatten([table([arm_row("untrained", 0.66), arm_row("round3", 0.9)],
                                 override=override("spike_sum"))])

        report = compare(current, reference, tolerance=0.01)

        self.assertTrue(report["ok"])
        self.assertEqual(report["compared"], 1)
        self.assertEqual([entry["arm"] for entry in report["extra"]], ["round3"])

    def test_a_partial_sweep_checks_the_rows_it_shares(self):
        reference = self._cell(0.84)
        partial = [row for row in reference if row["arm"] == "untrained"]

        report = compare(partial, reference, tolerance=0.0)

        self.assertTrue(report["ok"])
        self.assertEqual(report["compared"], 1)
        self.assertEqual([entry["arm"] for entry in report["missing"]], ["round1"])

    def test_every_quoted_metric_is_compared(self):
        self.assertEqual(len(METRICS), 6)


class TestCli(unittest.TestCase):

    def _sweep(self, directory, name, recall):
        return write(directory, name, table(
            [arm_row("untrained", 0.66), arm_row("round1", recall)],
            override=override("spike_sum"),
        ))

    def test_a_reproduced_sweep_exits_zero(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            current = self._sweep(tmpdir, "current.json", 0.84)
            reference = self._sweep(tmpdir, "reference.json", 0.84)

            self.assertEqual(main(["--tables", current, "--reference", reference]), 0)

    def test_a_sweep_whose_published_number_moved_exits_non_zero(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            current = self._sweep(tmpdir, "current.json", 0.54)
            reference = self._sweep(tmpdir, "reference.json", 0.84)

            self.assertEqual(main(["--tables", current, "--reference", reference]), 1)

    def test_a_directory_of_reference_tables_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            current = self._sweep(tmpdir, "current.json", 0.84)
            reference_dir = os.path.join(tmpdir, "reference")
            os.makedirs(reference_dir)
            self._sweep(reference_dir, "reference.json", 0.84)

            self.assertEqual(main(["--tables", current, "--reference", reference_dir]), 0)

    def test_a_sweep_without_a_reference_only_reports(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            current = self._sweep(tmpdir, "current.json", 0.54)

            self.assertEqual(main(["--tables", current]), 0)

    def test_the_report_is_written_where_it_is_asked_for(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            current = self._sweep(tmpdir, "current.json", 0.84)
            reference = self._sweep(tmpdir, "reference.json", 0.54)
            output = os.path.join(tmpdir, "sweep.json")

            self.assertEqual(
                main(["--tables", current, "--reference", reference, "--output", output]), 1)

            with open(output, "r", encoding="utf-8") as handle:
                report = json.load(handle)
            self.assertEqual(len(report["rows"]), 2)
            self.assertFalse(report["drift"]["ok"])

    def test_a_negative_tolerance_is_refused(self):
        with self.assertRaises(ValueError):
            compare([], [], tolerance=-1.0)


class TestRealArtifacts(unittest.TestCase):
    """The sweep over the runs this worktree actually produced, when they are present."""

    TABLES = ("spike_recalibrated_table.json", "drive_sum_table.json",
              "drive_sequence_table.json")

    def test_the_sweep_keys_the_real_tables_apart_and_keeps_their_baselines(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        paths = [os.path.join(root, "runs", "prescreen", name) for name in self.TABLES]
        if not all(os.path.exists(path) for path in paths):
            self.skipTest("pre-screen tables are regenerated artifacts and are not in the tree")

        rows = flatten([load_table(path) for path in paths])
        spike = [row for row in rows if row["statistic"] == "spike_sum@0.25"]
        drive = [row for row in rows if row["statistic"] == "drive_sum@0.25"]

        # One table per forced statistic, four arms each, the untrained baseline always among
        # them, and the two statistics in separate cells rather than averaged together.
        self.assertEqual(len(spike), 4)
        self.assertEqual(len(drive), 4)
        self.assertEqual(spike[0]["arm"], "untrained")
        self.assertNotEqual(spike[0]["cell_key"], drive[0]["cell_key"])
        self.assertEqual(spike[0]["cell_key"][0], drive[0]["cell_key"][0])


if __name__ == "__main__":
    unittest.main()
