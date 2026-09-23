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
