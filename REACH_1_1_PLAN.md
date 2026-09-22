# Reach World 1-1 plan

Updated 2026-09-21 from the open GitHub issues and current `origin/main`.

## Triage

The shortest path to a credible Level 1-1 result is to make completion measurable, create reproducible teacher trajectories, pre-train before online STDP, and evaluate model-only behavior against matched controls. Biology and browser presentation are deferred until that loop works.

| Priority | Issue | Status | Decision |
| --- | --- | --- | --- |
| P0 | #3 autonomous jump timing | Partially addressed | Durable run/checkpoint storage is merged. The remaining gate is autonomous model-only completion with a frozen checkpoint and a real completion detector. |
| P0 | #10 offline trajectory pretraining | In progress | Add versioned compressed trajectory shards, a deterministic successful teacher collector, and supervised motor eligibility pretraining. Keep model-only evaluation separate from the teacher upper bound. |
| P0 | #30 closed-loop DAgger | Implemented; ROM baseline beaten, completion gate still open | Teacher-only pretraining is confirmed insufficient (the policy leaves the teacher corridor and never returns). Added the bounded macro-action decoder, model-only on-policy rollouts with divergence detection, labelled recovery windows, an aggregated provenance-tracked dataset, and the `closed_loop_dagger.py` runner. The real-ROM run beats the reference baseline 2.1x model-only but does not complete the level; see below for the measured reason. |
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
- Issue #30 instrumentation: the mainline temporal decoder's measured model-only baseline is best_x 594, i.e. far short of the teacher's 3200+ flagpole. The failure mode is closed-loop: on-policy rollouts of a model-only candidate sit behind the teacher schedule by up to 112 px and then die at x≈635 (cadence 15, settle 3).
- Offline (synthetic, non-gate-eligible) DAgger validation, reserved seeds 42/43/44, cadence 15, settle 3, 2 epochs: baseline best_x 635 / death rate 1.00 / completion 0.00 → round 1 best_x 815 / death rate 0.33 → round 2 best_x 839 / death rate 0.33. Recovery windows mined per round: 4 then 31 (36 then 279 labelled samples), aggregated dataset 1,683 → 1,926 samples with per-shard origins (`teacher`, `dagger_recovery_windows`). No episode completed, so the P0 gate is still open and the synthetic numbers are labelled `offline_synthetic`.
- All 89 regression tests pass (`python -m unittest discover -s tests`, Linux container with `stable-retro` installed), including new coverage for the macro decoder and jump-margin calibration, DAgger rollouts/recovery windows, dataset provenance and rebuild-on-rerun, the experiment runner, the checkpoint round trip of the calibrated decoder config, and an explicit check that model-only action selection is unchanged when the environment returns garbage RAM.
- **Gate-eligible real-ROM run** (container build, cadence 15, settle 5, horizon 1600, one episode on each of seeds 42/43/44), `env_kind: stable-retro`, `model_only: true`, `p0_gate_met: false`: untrained baseline best_x 898 / mean 504.0; DAgger round 1 best_x **1247** / mean 875.7 (2.1x the mainline temporal-decoder reference of 594, 1.4x its own untrained baseline); round 2 continued from round 1 and regressed to 899. No episode completed in any arm. The teacher plan itself completes the level in 1,477 frames (max_x 3243, no death), so both the upper bound and the completion detector are real.
- Two decoder defects blocked the gate and are fixed: the jump evidence was summed so that the jump channel cancelled itself out of the comparison (a learned jump read as a tie, so the controller never jumped and died at the first obstacle), and the jump margin was hard-coded to zero on a readout whose evidence is offset for both chunk classes. Calibrating the margin from labelled evidence moved round-1 best_x from 313 to 1247. Supervising every frame rather than one target per cadence chunk was tested and is worse (decision-point separation 0.24-0.29 versus 2.4-5.8).
- Known ceiling: pretraining only fits the `layer3_4` readout over a fixed random connectome, reaching ~0.66-0.70 calibrated balanced accuracy at chunk decisions. Level 1-1 needs 100 sequential decisions with no fatal misjudgement, so a ~30% per-decision error rate cannot chain into completion. Raising this ceiling requires learning the visual features, not another DAgger round.
- **Recovery targets were matched on progress only, which taught the wrong action wherever the teacher was airborne** (`dagger.py`). On the recorded ROM shard the teacher takes 43 of its 74 run decisions mid-flight, so progress-only matching mislabels **1159 of 3244 ground states (35.7%)** as `run`; every one of those disagreements is `run` -> `jump`, i.e. a missing jump the teacher actually performed. At the x 594 stall it labelled `run` from the teacher's mid-flight frame there instead of the jump at x 549 that clears the pipe. Matching on (progress, ground/airborne phase) recovers the takeoff chunk, and a jump target is now held for the remainder of its teacher chunk so a flight in progress is not relabelled mid-air. Reporting: `teacher_labelling` in every round report, plus `--label-matching position_only` to ablate the fix on identical rollouts.
- **Ablating that fix rules it out as the gate blocker.** Same seeds, cadence 15, settle 5, horizon 1600, 30 epochs: the model-only rollout visits the same 131 frames and dies at the same x 312 in both arms, so only the window's supervision differs. `position_only` labels the window `run` from chunk start x 287; `phase` labels it `jump` from x 249 and every one of the window's 9 frames was previously mislabelled. Closed-loop round 1: best_x **1247 vs 1246**, per-seed 700/680/1247 vs 700/680/1246, motor argmax accuracy 0.647 vs 0.643. Nine corrected frames out of 1486 do not move a ~0.64-accuracy readout, so the labelling defect is a correctness fix that would compound in a longer-surviving run, not the binding constraint. The hypothesis that mislabelled recovery windows explained round 2's regression is falsified.

## Next experiment

Attack the readout ceiling rather than adding DAgger rounds, since round 2 already showed
that better calibration metrics can come with a worse closed-loop score:

1. ~~Audit what the recovery windows are actually teaching.~~ Done: `python dagger.py --teacher-dataset data/teacher_rom` is now a standing pre-flight check, and its findings are recorded in every report as `teacher_labelling`.
2. ~~Match recovery targets on phase rather than progress alone.~~ Done and ablated: see the two bullets above. It corrected all 9 frames of the mined window and changed round-1 best_x by 1 pixel, so the labelling rule is no longer a candidate explanation for the gate.
3. Record the teacher shard on the target machine: `python teacher.py --output data/teacher` (the shard's provenance must match the environment under test; the runner now refuses a mismatch), then reproduce round 1: `python closed_loop_dagger.py --mode dagger --teacher-dataset data/teacher --save-path drosophila_snn.pth --iterations 1 --settle-steps 5 --epochs 30`.
4. ~~Train something other than the fixed random readout — supervising the visual pathway (`layer1_2`/`layer2_3`) instead of `layer3_4` alone — and re-measure calibrated balanced accuracy at the chunk decisions before spending an emulator run on it.~~ Done: `--visual-pathway linear_feedback` trains `layer1_2`, `layer2_3` and `feedback_3_2` alongside the readout, and it does learn features (uncalibrated per-chunk argmax accuracy 0.548 -> 0.654 as its learning rate rises). The gating metric does **not** move: 0.522 / 0.531 / 0.540 at 5e-4 / 2e-3 / 1e-2, against 0.541 frozen and **0.521 for an untrained model**, all on the same teacher shard and seeds. No emulator run was spent. See "Training the visual pathway" in the README for the table.
5. **Fix the measurement before the next arm.** The chunk-decision metric cannot gate anything as it stands: an untrained model scores 0.521, its companion metric never beats always-run (0.748), a single Poisson replay of the same frozen model reads `d` 0.62 where three pooled read 0.06 (replays are separate trajectories of a chaotic network, not resamples), and a faithful continuous replay is at chance too. Report per-replay decisions across several model seeds, and score jump-chunk recall at a bounded jump rate rather than threshold-optimal accuracy on ~100 decisions.
6. Only then sweep the cadence (`1, 4, 8, 15`) and recovery-window budgets (`0, 4, 8, 16`) at matched seeds, and use the jump-rate-matched calibration (`method="jump_rate"`) as a second arm, to see whether the residual spread is cadence, window count, or threshold placement.

**Ruled out so far, with measurements:** teacher-only pretraining (the policy leaves the
corridor); decoder channel arithmetic and a hard-coded zero jump margin (fixed; 313 -> 1247);
per-frame instead of per-cadence supervision (separation 0.24-0.29 vs 2.4-5.8);
progress-only recovery labelling (fixed; 1247 -> 1246, i.e. not the blocker); and training
the visual pathway (0.531 vs 0.541 frozen and 0.521 untrained at the runner's own settings,
plus a new evidence-saturation failure that collapses the decision to always-run). Every one
of those was a real defect or a real design question, and none of them was the gate.

The measurement itself is now the binding constraint, not the architecture. Three of the
numbers this plan has leaned on do not survive their own error bars: the "~0.66-0.70" readout
ceiling is 0.541 at the epochs the runner actually uses (it needs 10-30 epochs), an untrained
model scores the same as a trained one, and per-replay variance is as large as any arm
difference measured so far. Pre-screening is still the right discipline — it takes seconds
and caught the 313 -> 1247 decoder defect — but its output has to be a *paired, multi-seed,
per-replay* comparison with a trivial baseline next to it, never a single calibrated number. Report completion rate, Wilson lower bound, best
X, death rate, divergence-window counts and provenance (`model_only`, assistance flags, ROM
and dataset checksums) for every arm. Do not treat bootstrap-assisted or synthetic-env
numbers as learning evidence, and do not call a cadence change a learned policy.
