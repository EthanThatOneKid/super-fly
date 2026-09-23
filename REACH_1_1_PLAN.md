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
- All 174 regression tests pass (`python -m unittest discover -s tests`, Linux container with `stable-retro` installed), including coverage for the macro decoder and jump-margin calibration (now pinned finite and JSON-safe, with degeneracy reported), the bounded-rate pre-screen (budget vs every fixed margin, per-replay spread, seed pairing, the untrained baseline being non-optional, checkpoint arms honouring the shipped margin), DAgger rollouts/recovery windows, dataset provenance and rebuild-on-rerun, the experiment runner, the checkpoint round trip of the calibrated decoder config, and an explicit check that model-only action selection is unchanged when the environment returns garbage RAM.
- **Gate-eligible real-ROM run** (container build, cadence 15, settle 5, horizon 1600, one episode on each of seeds 42/43/44), `env_kind: stable-retro`, `model_only: true`, `p0_gate_met: false`: untrained baseline best_x 898 / mean 504.0; DAgger round 1 best_x **1247** / mean 875.7 (2.1x the mainline temporal-decoder reference of 594, 1.4x its own untrained baseline); round 2 continued from round 1 and regressed to 899. No episode completed in any arm. The teacher plan itself completes the level in 1,477 frames (max_x 3243, no death), so both the upper bound and the completion detector are real.
- Two decoder defects blocked the gate and are fixed: the jump evidence was summed so that the jump channel cancelled itself out of the comparison (a learned jump read as a tie, so the controller never jumped and died at the first obstacle), and the jump margin was hard-coded to zero on a readout whose evidence is offset for both chunk classes. Calibrating the margin from labelled evidence moved round-1 best_x from 313 to 1247. Supervising every frame rather than one target per cadence chunk was tested and is worse (decision-point separation 0.24-0.29 versus 2.4-5.8).
- Known ceiling, now measured against a null arm: pretraining only fits the `layer3_4` readout over a fixed random connectome, and the bounded-rate pre-screen puts it at **0.413** jump-chunk recall inside a 0.35 jump budget, against **0.333** for an untrained model and 0.343 for chance (round 2: 0.427). The earlier "~0.66-0.70 calibrated balanced accuracy" claim was measured at 10-30 epochs; at the epochs the runner uses it is 0.541, which is why the pre-screen no longer uses balanced accuracy. Level 1-1 needs 100 sequential decisions with no fatal misjudgement, so a 7-9 point edge over chance cannot chain into completion. Raising this ceiling requires a decision that actually separates the teacher's jumps, not another DAgger round.
- **Recovery targets were matched on progress only, which taught the wrong action wherever the teacher was airborne** (`dagger.py`). On the recorded ROM shard the teacher takes 43 of its 74 run decisions mid-flight, so progress-only matching mislabels **1159 of 3244 ground states (35.7%)** as `run`; every one of those disagreements is `run` -> `jump`, i.e. a missing jump the teacher actually performed. At the x 594 stall it labelled `run` from the teacher's mid-flight frame there instead of the jump at x 549 that clears the pipe. Matching on (progress, ground/airborne phase) recovers the takeoff chunk, and a jump target is now held for the remainder of its teacher chunk so a flight in progress is not relabelled mid-air. Reporting: `teacher_labelling` in every round report, plus `--label-matching position_only` to ablate the fix on identical rollouts.
- **Ablating that fix rules it out as the gate blocker.** Same seeds, cadence 15, settle 5, horizon 1600, 30 epochs: the model-only rollout visits the same 131 frames and dies at the same x 312 in both arms, so only the window's supervision differs. `position_only` labels the window `run` from chunk start x 287; `phase` labels it `jump` from x 249 and every one of the window's 9 frames was previously mislabelled. Closed-loop round 1: best_x **1247 vs 1246**, per-seed 700/680/1247 vs 700/680/1246, motor argmax accuracy 0.647 vs 0.643. Nine corrected frames out of 1486 do not move a ~0.64-accuracy readout, so the labelling defect is a correctness fix that would compound in a longer-surviving run, not the binding constraint. The hypothesis that mislabelled recovery windows explained round 2's regression is falsified.

## Next experiment

Attack the readout ceiling rather than adding DAgger rounds, since round 2 already showed
that better calibration metrics can come with a worse closed-loop score:

1. ~~Audit what the recovery windows are actually teaching.~~ Done: `python dagger.py --teacher-dataset data/teacher_rom` is now a standing pre-flight check, and its findings are recorded in every report as `teacher_labelling`.
2. ~~Match recovery targets on phase rather than progress alone.~~ Done and ablated: see the two bullets above. It corrected all 9 frames of the mined window and changed round-1 best_x by 1 pixel, so the labelling rule is no longer a candidate explanation for the gate.
3. Record the teacher shard on the target machine: `python teacher.py --output data/teacher` (the shard's provenance must match the environment under test; the runner now refuses a mismatch), then reproduce round 1: `python closed_loop_dagger.py --mode dagger --teacher-dataset data/teacher --save-path drosophila_snn.pth --iterations 1 --settle-steps 5 --epochs 30`.
4. ~~Train something other than the fixed random readout — supervising the visual pathway (`layer1_2`/`layer2_3`) instead of `layer3_4` alone — and re-measure calibrated balanced accuracy at the chunk decisions before spending an emulator run on it.~~ Done: `--visual-pathway linear_feedback` trains `layer1_2`, `layer2_3` and `feedback_3_2` alongside the readout, and it does learn features (uncalibrated per-chunk argmax accuracy 0.548 -> 0.654 as its learning rate rises). The gating metric does **not** move: 0.522 / 0.531 / 0.540 at 5e-4 / 2e-3 / 1e-2, against 0.541 frozen and **0.521 for an untrained model**, all on the same teacher shard and seeds. No emulator run was spent. Re-measured once the measurement was replaced (30 epochs, settle 5): recall@0.35 **0.280** against **0.333** for an untrained model, 0/3 replays improved, so the arm is not merely un-gated -- it is worse than its own initialization. See "Training the visual pathway" and "The pre-screen: jump recall at a bounded jump rate" in the README.
5. ~~Fix the measurement before the next arm.~~ Done: `prescreen.py` replaces the pooled calibrated number with **jump-chunk recall at a bounded jump rate** (`recall@0.35`), reported per replay, paired seed by seed against an untrained arm that now appears in every table. On the real teacher shard (99 decisions/replay, 3 replays, settle 5, chance = 0.343): untrained 0.333 (**at chance**), DAgger round 1 **0.413**, round 2 0.427, visual-pathway arm 0.280 (**below chance**, paired delta -0.053 on 0/3 replays). No arm passes the gate (recall floor 0.60, paired gain >= +0.10, every replay improving), so no emulator run is justified yet. It also exposed two defects: an inseparable readout shipped an always-jump decoder (margin `-Infinity`, now a finite sentinel plus a `degenerate: true` flag), and the visual-pathway arm is measurably worse than not training at all.

   That view was then replaced in turn, because it still scored one decision at a time from a cold SNN state, ignoring the decoder's chunk commitment and the refractory period, with no notion that a missed jump ends the run. The default is now a **teacher-forced sequential replay** (`offline_episode.py`): the teacher's frames are fed in order through the same decoder the closed loop drives, and an arm is scored by `jump_sequence_recall` (of the 20 jumps the teacher actually took, how many the controller took, decided in sequence). Real shard, settle 5, seeds 42/43/44, against the same checkpoints' emulator scores: untrained 0.733 (emulator 898), round 1 0.817 (emulator **1247**), round 2 0.850 (emulator 899), visual-pathway arm 1.000 (never run). No arm passes the five-criterion gate, and the report says which way each one is wrong: round 2 fails only completion and the 0.90 recall floor, round 1 fails four of five, and the visual-pathway arm fails **only** `jump_rate_within_budget` (0.495 of decisions spent jumping, 29 spurious) while passing recall and completion -- the always-jump exploit that the budget criterion exists to close. See "The pre-screen: how an arm earns an emulator run" in the README.
6. Only then sweep the cadence (`1, 4, 8, 15`) and recovery-window budgets (`0, 4, 8, 16`) at matched seeds, and use the jump-rate-matched calibration (`method="jump_rate"`) as a second arm, to see whether the residual spread is cadence, window count, or threshold placement.
7. **Give the replay consequences, or stop asking it to predict dynamics.** Teacher forcing is what stopped the sequential pre-screen ranking the arms: the frames belong to the teacher, so the arm's own trajectory never exists and jumping constantly is unpunished (measured: one checkpoint scores a perfect recall and a completion while spending 49.5% of its decisions jumping). The cheap path is to keep it as a gate and a diagnostic -- which is what it now is, and what caught the exploit. The real path is to model what a jump *does*, so that taking one early, late, or constantly has an outcome; that is a job for the offline stand-in (`offline_env.py`), which is plumbing today rather than physics. Either way the emulator remains the only arbiter of dynamics, and its run stays the thing an arm has to earn.

**Ruled out so far, with measurements:** teacher-only pretraining (the policy leaves the
corridor); decoder channel arithmetic and a hard-coded zero jump margin (fixed; 313 -> 1247);
per-frame instead of per-cadence supervision (separation 0.24-0.29 vs 2.4-5.8);
progress-only recovery labelling (fixed; 1247 -> 1246, i.e. not the blocker); and training
the visual pathway (recall@0.35 **0.280** vs 0.333 untrained, paired -0.053 on 0/3 replays,
i.e. worse than not training at all, plus an evidence-saturation failure that leaves the
shipped margin unable to separate the chunk classes); and teacher-forced sequential scoring
itself as a *predictor* of the emulator's ordering (round 2 0.850 vs round 1 0.817 recall,
while the emulator has round 1 ahead 1247 vs 899 -- the sequence view remains useful as a gate
and a diagnostic, and it is what caught the always-jump exploit). Every one of those was a real
defect or a real design question, and none of them was the gate.

The measurement is fixed; the **decision ceiling** is now the binding constraint, and fixing
the measurement sharpened that reading rather than weakening it. The sequential view is the
third attempt and the most honest: it measures the right thing (decisions made in context) and
it still ranks round 2 above round 1 on recall, consistency and `best_x`, where the emulator
ranked round 1 first (1247 vs 899). That is a structural limit, not a scoring bug -- with the
frames teacher-forced, an arm's own trajectory never exists, so no offline replay can model the
state its decisions would create. The emulator stays the only arbiter of dynamics; the
pre-screen's job is to stop wasting runs on arms that are wrong for reasons it *can* see. Three numbers this plan leaned
on do not survive their own error bars: the "~0.66-0.70" readout ceiling is 0.541 at the epochs
the runner actually uses (it needs 10-30 epochs), an untrained model scores the same as a
trained one, and per-replay variance is *larger* than any arm's paired gain. What the new
pre-screen adds is a falsifiable statement of where the arms actually are: 7-9 points above
chance at catching the teacher's jumps inside a 35% jump budget, never above 0.43 recall,
against a 0.60 floor. Pre-screening remains the right discipline — it takes seconds and caught
the 313 -> 1247 decoder defect — and its output is now a *paired, multi-seed, per-replay*
comparison with a trivial baseline next to it, never a single calibrated number.

One honest limit stays on the record: the pre-screen still ranks round 2 (0.427, closed-loop
899) above round 1 (0.413, closed-loop 1247), because it scores one decision per cadence point
along a successful teacher trajectory while the closed loop is sequential and a single
misjudgement is fatal. It is a gate and a diagnostic, not a surrogate for the emulator — and
the arms it is gating are all currently on the wrong side of it. Report completion rate, Wilson lower bound, best
X, death rate, divergence-window counts and provenance (`model_only`, assistance flags, ROM
and dataset checksums) for every arm. Do not treat bootstrap-assisted or synthetic-env
numbers as learning evidence, and do not call a cadence change a learned policy.
