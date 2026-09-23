import unittest

from seed_policy import (
    DEFAULT_REPLAYS,
    DEV_ROLE,
    DEV_SEEDS,
    GATE_ROLE,
    GATE_SEEDS,
    SeedSet,
    assert_not_reserved,
    describe,
    dev_seeds,
    gate_seed_set,
    parse_seeds,
    reserved_seeds,
    resolve_seeds,
)


class TestSeedSets(unittest.TestCase):
    """The split between the seeds iteration runs on and the seeds the claim runs on.

    This is the one place the experiment's seed hygiene can be stated as an invariant
    rather than a convention, so it is tested like one: the default is the dev range,
    the reserved set is reachable only in full, and a set may not mix the two.
    """

    def test_the_default_is_the_dev_range(self):
        resolved = resolve_seeds()

        self.assertEqual(resolved.seeds, DEV_SEEDS[:DEFAULT_REPLAYS])
        self.assertEqual(resolved.role, DEV_ROLE)
        self.assertFalse(resolved.report_only)
        self.assertFalse(set(resolved.seeds) & set(GATE_SEEDS))

    def test_a_replay_count_walks_down_the_dev_range_never_up_from_a_base(self):
        """Counting up from a base seed is how iteration silently became the reserved triple."""
        self.assertEqual(resolve_seeds(replays=1).seeds, (45,))
        self.assertEqual(resolve_seeds(replays=5).seeds, DEV_SEEDS)
        # And the old derivation, spelled out, is exactly what is now refused.
        self.assertEqual(set(GATE_SEEDS), {42, 43, 44})

    def test_asking_for_more_replays_than_the_dev_range_refuses_rather_than_borrowing(self):
        with self.assertRaises(ValueError) as caught:
            dev_seeds(len(DEV_SEEDS) + 1)

        self.assertIn("report-only", str(caught.exception))

    def test_the_reserved_set_is_reachable_only_in_full(self):
        self.assertEqual(resolve_seeds(GATE_SEEDS).role, GATE_ROLE)
        self.assertTrue(resolve_seeds(GATE_SEEDS).report_only)

        for subset in ((42,), (42, 43), (44,)):
            with self.assertRaises(ValueError) as caught:
                resolve_seeds(subset)
            self.assertIn("whole reserved set", str(caught.exception))

    def test_a_set_may_not_straddle_the_two_roles(self):
        for straddle in ((42, 45), (43, 44, 45), (42, 43, 44, 45)):
            with self.assertRaises(ValueError) as caught:
                resolve_seeds(straddle)
            self.assertIn("straddle", str(caught.exception))

    def test_the_role_is_inferred_and_a_mismatch_is_an_error(self):
        with self.assertRaises(ValueError) as caught:
            resolve_seeds(DEV_SEEDS[:2], role=GATE_ROLE)
        self.assertIn("does not match", str(caught.exception))

        with self.assertRaises(ValueError):
            resolve_seeds(GATE_SEEDS, role=DEV_ROLE)

        # Asserting the role that follows from the seeds is allowed and changes nothing.
        self.assertEqual(resolve_seeds(DEV_SEEDS[:2], role=DEV_ROLE).role, DEV_ROLE)

    def test_duplicate_and_empty_sets_are_refused(self):
        for bad in ((45, 45), ()):
            with self.assertRaises(ValueError):
                resolve_seeds(bad)

    def test_a_replay_count_contradicting_the_seed_list_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            resolve_seeds((45, 46), replays=3)

        self.assertIn("does not match", str(caught.exception))

    def test_fitting_on_the_reserved_seeds_is_refused(self):
        assert_not_reserved(DEV_SEEDS, "calibrating the jump margin")

        for reserved in (GATE_SEEDS, (42, 45)):
            with self.assertRaises(ValueError) as caught:
                assert_not_reserved(reserved, "calibrating the jump margin")
            self.assertIn("may not use the reserved gate seeds", str(caught.exception))

    def test_reserved_seeds_reports_what_is_present_in_reserved_order(self):
        self.assertEqual(reserved_seeds((44, 45, 42)), (42, 44))
        self.assertEqual(reserved_seeds(DEV_SEEDS), ())

    def test_a_seed_set_carries_both_ranges_so_a_report_can_be_audited(self):
        dev = resolve_seeds(replays=2).to_dict()
        gate = gate_seed_set().to_dict()

        self.assertEqual(dev["seeds"], list(DEV_SEEDS[:2]))
        self.assertEqual(dev["dev_seeds"], list(DEV_SEEDS))
        self.assertEqual(dev["reserved_gate_seeds"], list(GATE_SEEDS))
        self.assertFalse(dev["report_only"])
        self.assertTrue(gate["report_only"])
        self.assertIn("tuning only", dev["note"])
        self.assertIn("report-only", gate["note"])
        self.assertEqual(describe(gate_seed_set()), gate["note"])

    def test_the_gate_seed_set_is_the_reserved_one(self):
        self.assertEqual(gate_seed_set().seeds, GATE_SEEDS)
        self.assertTrue(gate_seed_set().report_only)

    def test_parse_seeds_accepts_the_cli_form(self):
        self.assertEqual(parse_seeds("45, 46 ,47"), (45, 46, 47))
        self.assertEqual(parse_seeds(""), ())

    def test_seed_sets_are_comparable_by_value(self):
        self.assertEqual(resolve_seeds(replays=2), SeedSet(DEV_SEEDS[:2], DEV_ROLE))


if __name__ == "__main__":
    unittest.main()
