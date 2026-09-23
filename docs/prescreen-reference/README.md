# Pre-screen reference tables

Published pre-screen tables, in the form `sweep.py` reads, so the numbers quoted in `README.md`
and `REACH_1_1_PLAN.md` can be checked rather than remembered.

`sweep.py --reference <one of these>` gates a sweep against them. A row whose metric moves
further than `--tolerance` fails the run, and so does a sweep that shares no row with the
reference at all -- a comparison that addressed nothing has not checked anything. Rows are
matched by their data (the teacher shard's SHA-256, not its path), their protocol (stride, settle
steps, replay seeds and their seed *role*), the arm, and the statistic that arm was measured
under. A partial sweep is therefore checked against the rows it does share, and the rest are
reported as "in the reference, not in this sweep" rather than dropped.

`spike_sum-settle5-dev.json` is the shipped statistic's control run: settle 5, stride 15, dev
seeds 45-49, five replays, `--evidence-rule spike_sum` forced across every arm including the
untrained baseline. It is the table the statistic sweep in the README is compared against, and
its untrained row is reproducible **without any checkpoint**, which is what makes it usable as a
first dispatch's reproduce check on a runner that has nothing but the repository:

```bash
python sweep.py --tables runs/prescreen/<cell>.json \
  --reference docs/prescreen-reference/spike_sum-settle5-dev.json --tolerance 0
```

At `--tolerance 0` the comparison demands an exact reproduction, which is the interesting question
across hosts: floating-point reductions depend on BLAS threading and CPU features, so a clean
result licenses later cross-host numbers and a dirty one is itself the finding. The three
checkpoint-bearing rows of that table need the checkpoints on the runner to be reproduced; the
`prescreen-sweep` workflow takes them as an input and reports them as missing when they are absent.
Checkpoints are declared per cell, so a cell that reads none (the untrained baseline, or any arm
measured from scratch) is not held up by weights another cell needs: give it `"checkpoints":
"none"` in the dispatch's `cells`, and the preflight requires exactly what that cell reads.

## Cross-host reproduction: measured, exact

The first reference dispatch ([run 35933466810](https://github.com/EthanThatOneKid/super-fly/actions/runs/35933466810),
`ubuntu-latest`, Python 3.11) ran one cell -- the untrained row of `spike_sum-settle5-dev.json`,
forced `spike_sum@0.25`, settle 5, stride 15, dev seeds 45-49, five replays, no checkpoints -- and
asked for an exact reproduction. It got one, on both axes of the question:

- **The data reproduced byte for byte.** `data/` is gitignored, so the runner regenerated the
  teacher shard from the tracked ROM with `teacher.py` and hashed `shard-00000.npz` to
  `c050a0a0…4290b8` -- the same bytes this table was built on, from a different numpy/zlib build.
  The workflow checks the shard against the reference's recorded hash before the run for exactly
  this reason, since rows are matched on file bytes, and a mismatched shard and a drifting float
  are the same "nothing compared" from `sweep.py` while meaning different things.
- **The numbers reproduced exactly at `--tolerance 0`.** `1 row(s) compared, 0 moved beyond 0.0`:
  per-replay sequence recall 0.70/0.65/0.70/0.70/0.55 (mean 0.660), jump rate
  0.2121/0.1717/0.1717/0.202/0.202 (mean 0.1919), spurious jumps 7/4/3/6/9 (mean 5.8),
  `best_x` 549/362/249/249/249 (mean 331.6), and the re-fitted margin's calibration block
  (balanced accuracy 0.519, run recall 0.8459). Every field in the row compares equal, not just
  the six the gate watches: the same five draws drive both, so a 2-vCPU runner's BLAS reduction
  order is not moving a digit against the eight-core local run.

What that licenses: a pre-screen number is a property of the shard, the protocol and the seeds,
not of the machine it was computed on, so later sweeps can be dispatched and compared against
local readings without a per-host caveat. The untrained row is the cheap half of the check because
it needs no weights; the remaining three rows of this table are the same question with checkpoints
loaded, and they need the ~13 MB of checkpoints on the runner (they live under the gitignored
`runs/`) before the gate can cover them.
