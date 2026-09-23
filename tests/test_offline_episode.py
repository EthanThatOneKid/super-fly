"""Tests for the teacher-forced sequential pre-screen.

This layer exists because scoring decisions independently could not rank two arms the
emulator separated (round 2 scored 0.427 against round 1's 0.413 while dying 350 px
earlier). What has to be true for the replacement to be a gate:

* a *required* jump is one the teacher actually made -- a stretch of flight that began with
  a jump, not a fall;
* missing one is fatal *at that position*, so the outcome is an episode's ``best_x`` and not
  a count of wrong labels;
* jumping where the teacher was grounded is not fatal (it lands and continues) but is
  counted, so an over-firing arm is visible;
* a shard that stops short of the flagpole cannot produce completion evidence.
"""

import unittest

import numpy as np

from offline_episode import JumpWindow, episode_outcome, macro_decoder, teacher_traces

RAM_SIZE = 0x0800
PAGE_X = 0x006D
SUB_X = 0x0086
AIR_STATE = 0x001D


def shard(xs, air_states):
    """A shard whose RAM carries the teacher's positions and vertical states."""
    ram = []
    for x, air in zip(xs, air_states):
        row = np.zeros(RAM_SIZE, dtype=np.uint8)
        row[PAGE_X] = (int(x) // 256) % 256
        row[SUB_X] = int(x) % 256
        row[AIR_STATE] = int(air)
        ram.append(row)
    frames = np.zeros((len(xs), 2, 2, 3), dtype=np.uint8)
    actions = np.ones(len(xs), dtype=np.uint8)
    return {"frames": frames, "actions": actions, "ram": np.stack(ram)}


def decision(index, start_frame, macro, frames=5):
    return {
        "index": index,
        "start_frame": start_frame,
        "frames": frames,
        "macro": macro,
        "jump_evidence": 0.0,
        "run_evidence": 0.0,
    }


def built(*macros, frames=5):
    """One decision per cadence chunk, all of the given macros."""
    return [decision(index, index * frames, macro, frames) for index, macro in enumerate(macros)]


class TestTeacherTraces(unittest.TestCase):
    def test_position_and_phase_come_from_the_recorded_ram(self):
        xs = [40, 60, 80, 100, 120, 140]
        air = [0, 0, 1, 1, 0, 0]

        trace = teacher_traces([shard(xs, air)])[0]

        self.assertEqual(trace["x"], xs)
        self.assertEqual(trace["airborne"], [False, False, True, True, False, False])
        self.assertEqual(trace["max_x"], 140)
        self.assertEqual(trace["frames"], 6)

    def test_a_required_jump_is_flight_that_began_with_a_jump(self):
        # Frames 2-3 are a jump (state 1 then 2); frames 5-6 are a fall (state 2 only).
        air = [0, 0, 1, 2, 0, 2, 2, 0]
        xs = [0, 10, 20, 30, 40, 50, 60, 70]

        trace = teacher_traces([shard(xs, air)])[0]

        self.assertEqual(trace["required"], [JumpWindow(2, 3, True)])
        self.assertEqual(trace["fall_windows"], [JumpWindow(5, 6, False)])
        self.assertEqual(trace["windows"], [JumpWindow(2, 3, True), JumpWindow(5, 6, False)])

    def test_a_shard_that_never_leaves_the_ground_requires_nothing(self):
        trace = teacher_traces([shard([0, 10, 20, 30], [0, 0, 0, 0])])[0]

        self.assertEqual(trace["required"], [])
        self.assertEqual(trace["windows"], [])
        self.assertFalse(trace["reached_end"])

    def test_a_stretch_of_flight_running_to_the_last_frame_is_closed(self):
        trace = teacher_traces([shard([0, 10, 20], [0, 1, 2])])[0]

        self.assertEqual(trace["required"], [JumpWindow(1, 2, True)])


class TestEpisodeOutcome(unittest.TestCase):
    #: A teacher that jumps out of chunk 1 (frames 5-9) and out of chunk 4 (frames 20-24),
    #: reaching the flagpole, and is grounded everywhere else.
    def _trace(self, required_chunks=(1, 3)):
        length = 30
        xs = [40 + index * 110 for index in range(length)]
        air = [0] * length
        for chunk in required_chunks:
            air[chunk * 5] = 1
            air[chunk * 5 + 1] = 2
        return teacher_traces([shard(xs, air)])[0]

    def test_taking_every_required_jump_completes_the_shard(self):
        trace = self._trace()
        # Chunk 1 opens at frame 5, chunk 3 at frame 15: jump on both.
        decisions = built("run", "jump", "run", "jump", "run", "run")

        outcome = episode_outcome(decisions, trace)

        self.assertEqual(outcome["required_jumps"], 2)
        self.assertEqual(outcome["jumps_taken"], 2)
        self.assertEqual(outcome["missed_jumps"], 0)
        self.assertEqual(outcome["spurious_jumps"], 0)
        self.assertEqual(outcome["jump_sequence_recall"], 1.0)
        self.assertTrue(outcome["offline_completion"])
        self.assertEqual(outcome["offline_best_x"], trace["max_x"])
        self.assertIsNone(outcome["first_miss_x"])

    def test_running_where_the_teacher_jumped_is_fatal_at_that_position(self):
        trace = self._trace()
        decisions = built("run", "run", "run", "jump", "run", "run")

        outcome = episode_outcome(decisions, trace)

        self.assertEqual(outcome["missed_jumps"], 1)
        self.assertEqual(outcome["jumps_taken"], 1)
        self.assertEqual(outcome["jump_sequence_recall"], 0.5)
        self.assertFalse(outcome["offline_completion"])
        # The run ends where the flight it did not take began.
        self.assertEqual(outcome["first_miss_frame"], 5)
        self.assertEqual(outcome["first_miss_x"], trace["x"][5])
        self.assertEqual(outcome["offline_best_x"], trace["x"][5])
        self.assertLess(outcome["offline_best_x"], trace["max_x"])
        self.assertEqual(outcome["survival_frames"], 5)

    def test_the_first_miss_is_what_ends_the_run(self):
        trace = self._trace()
        decisions = built("run", "jump", "run", "run", "run", "run")

        outcome = episode_outcome(decisions, trace)

        self.assertEqual(outcome["missed_jumps"], 1)
        # Missed the second flight, not the first, so the run got as far as chunk 3.
        self.assertEqual(outcome["first_miss_frame"], 15)
        self.assertEqual(outcome["offline_best_x"], trace["x"][15])

    def test_sequence_recall_keeps_counting_after_a_miss(self):
        """The reason this statistic exists: it does not saturate where ``best_x`` does.

        Teacher-forced, later jumps are still measurable, so two arms that both die at the
        first miss are still separated by how many of the teacher's jumps they took.
        """
        trace = self._trace()
        decisions = built("run", "run", "run", "jump", "run", "run")

        outcome = episode_outcome(decisions, trace)

        # Died at the first flight (the compounded reading) but took half the jumps.
        self.assertEqual(outcome["first_miss_frame"], 5)
        self.assertEqual(outcome["jump_sequence_recall"], 0.5)
        self.assertEqual(outcome["fatality_model"], "any_missed_teacher_jump_ends_run")

    def test_a_jump_where_the_teacher_was_grounded_is_counted_and_not_fatal(self):
        trace = self._trace()
        decisions = built("jump", "jump", "run", "jump", "run", "run")

        outcome = episode_outcome(decisions, trace)

        self.assertEqual(outcome["spurious_jumps"], 1)
        self.assertEqual(outcome["missed_jumps"], 0)
        self.assertTrue(outcome["offline_completion"])
        self.assertEqual(outcome["jump_rate"], round(3 / 6, 4))

    def test_a_jump_started_one_chunk_early_still_counts_as_taken(self):
        """The requirement is to be airborne *over* the gap, not to press A at a frame."""
        trace = self._trace(required_chunks=(1,))
        # A 5-frame chunk opening at frame 4 overlaps the flight at frames 5-6.
        decisions = [decision(0, 0, "run"), decision(1, 4, "jump")]

        outcome = episode_outcome(decisions, trace)

        self.assertEqual(outcome["missed_jumps"], 0)
        self.assertEqual(outcome["spurious_jumps"], 0)

    def test_a_shard_short_of_the_flagpole_is_not_credited_with_completion(self):
        trace = teacher_traces([shard([40, 60, 80, 100], [0, 1, 2, 0])])[0]

        outcome = episode_outcome(built("jump", "run"), trace)

        self.assertEqual(outcome["missed_jumps"], 0)
        self.assertFalse(trace["reached_end"])
        self.assertFalse(outcome["offline_completion"])
        # The outcome still reports how far it got, so a partial shard is readable.
        self.assertEqual(outcome["offline_best_x"], 100)

    def test_a_fall_is_reported_but_is_not_a_miss(self):
        air = [0, 0, 2, 0]
        trace = teacher_traces([shard([40, 60, 80, 100], air)])[0]

        outcome = episode_outcome(built("run", "run"), trace)

        self.assertEqual(outcome["required_jumps"], 0)
        self.assertEqual(outcome["missed_jumps"], 0)
        self.assertIsNone(outcome["jump_sequence_recall"])
        self.assertEqual(outcome["fall_windows"], 1)

    def test_an_arm_that_never_decides_reports_no_rate(self):
        outcome = episode_outcome([], self._trace())

        self.assertIsNone(outcome["jump_rate"])
        self.assertEqual(outcome["decisions"], 0)
        self.assertEqual(outcome["missed_jumps"], 2)
        self.assertEqual(outcome["jump_sequence_recall"], 0.0)


class TestDecoderSelection(unittest.TestCase):
    """The replay has to run the decoder the checkpoint would adopt in closed loop."""

    def test_chunk_lengths_and_refractory_come_from_the_shipped_config(self):
        decoder = macro_decoder(15, {"chunk_frames": 4, "jump_chunk_frames": 6,
                                     "jump_margin": -2.0, "refractory_frames": 8})

        self.assertEqual(decoder.chunk_frames, 4)
        self.assertEqual(decoder.jump_chunk_frames, 6)
        self.assertEqual(decoder.jump_margin, -2.0)
        self.assertEqual(decoder.refractory_frames, 8)

    def test_an_arm_without_a_config_falls_back_to_the_run_cadence(self):
        decoder = macro_decoder(15, None)

        self.assertEqual(decoder.chunk_frames, 15)
        self.assertEqual(decoder.jump_chunk_frames, 15)
        self.assertEqual(decoder.jump_margin, 0.0)
        self.assertEqual(decoder.refractory_frames, 0)


if __name__ == "__main__":
    unittest.main()
