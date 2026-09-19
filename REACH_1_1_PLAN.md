# Reach World 1-1 plan

Updated 2026-09-19 from the open GitHub issues and current `origin/main`.

## Triage

The shortest path to a credible Level 1-1 result is to make completion measurable, create reproducible teacher trajectories, pre-train before online STDP, and evaluate model-only behavior against matched controls. Biology and browser presentation are deferred until that loop works.

| Priority | Issue | Status | Decision |
| --- | --- | --- | --- |
| P0 | #3 autonomous jump timing | Partially addressed | Durable run/checkpoint storage is merged. The remaining gate is autonomous model-only completion with a frozen checkpoint and a real completion detector. |
| P0 | #10 offline trajectory pretraining | In progress | Add versioned compressed trajectory shards, a deterministic successful teacher collector, and supervised motor eligibility pretraining. Keep model-only evaluation separate from the teacher upper bound. |
| P1 | #20 telemetry and distributed training | Prerequisites largely present | Use it after the P0 benchmark is reproducible; do not distribute an unvalidated learner. |
| P2 | #11 biological FlyWire/Hemibrain topology | Not started | Defer until the synthetic controller can solve the task; otherwise topology changes confound the motor-learning bottleneck. |
| P2 | #14 explicit descending-neuron groups | Not started | Defer until motor-output failure is characterized with the P0 baselines. |
| P3 | #13 client-side WebGPU execution | Not started | Presentation and latency work, not a reachability blocker. |
| P3 | #22 prickly fly arms | Not started | Keep as a dashboard/controller presentation feature after the learning loop is credible. |

## Acceptance gates

1. The teacher trajectory must reach the Level 1-1 flagpole and hold the completion state for 30 frames.
2. `eval_harness.py` must report `completed` and `completion_rate`; max X is not success.
3. Compare model-only, right-only, bootstrap-only, and teacher upper-bound policies with matched reset, horizon, and seed.
4. Freeze the earliest checkpoint that passes the model-only completion gate. Do not call bootstrap success learning.

## Current evidence

- The deterministic beam-found teacher reaches x=3200+ and holds RAM player state `0x05` for the completion detector; the generated shard contains 1,477 frames and is marked completed.
- The current supervised motor eligibility prototype runs with the requested 0.0005 learning rate, three epochs, and 4-frame sampling stride.
- The first model-only checkpoint evaluation did not complete: it died at x=313. This is useful negative evidence, not a solved learner.
- All 47 regression and training-pipeline tests pass on the feature worktree.

## Next experiment

Replace motor-only one-layer pretraining with sequence-aware supervised training or a bounded temporal-settling loop, then run the checkpoint budgets `0, 10, 20, 50, 100, 200` across independent seeds. Keep the successful teacher as an upper bound and report completion rate, Wilson lower bound, max X, death rate, and control provenance.
