# super-fly

A PyTorch spiking neural network (SNN) model of the *Drosophila melanogaster*
(common fruit fly) connectome learning to play NES **Super Mario Bros** using
dual-pathway dopamine-modulated STDP.

The "brain" drives the controller, and a live web dashboard streams the
telemetry (game frame + per-layer spike heatmaps + dopamine gauges) to your
browser in real time.

## Features

- **4-layer Drosophila connectome SNN** — ommatidia sensory grid → optic lobe /
  medulla → central complex / mushroom body → thoracic motor ganglion.
- **Ommatidial vision preprocessing** — Farneback optical flow + Canny edges on
  the NES frame, resampled to a 28×28 ommatidia grid.
- **Dual dopamine-modulated STDP** — PAM (reward/progress) and PPL1
  (punishment/death) pathways with an inverted update sign.
- **RAM-based reward and completion** — progress, stagnation, death, and a 30-frame Level 1-1 completion detector are read straight from SMB RAM addresses (`0x006D` level page, `0x0086` sub-page X, `0x000E` player state, `0x0770` operating mode).
- **Autonomous Jump & Bootstrap Controller** — model-driven jump priority with 4-frame hold and 24-step refractory timing, plus periodic bootstrap pulses and STDP teaching trace injection for assisted jumps.
- **Deterministic Evaluation Harness** — isolated evaluation script (`eval_harness.py`) for benchmarking progress and actual Level 1-1 completions across episodes.
- **Offline teacher pipeline** — `teacher.py` records a successful, checksummed Level 1-1 trajectory; `pretrain.py` applies supervised motor eligibility-trace updates before online STDP.
- **Bounded macro-action decoder** — `macro_decoder.py` commits one RUN / RUN+JUMP chunk per action cadence (bounded by `MAX_CHUNK_FRAMES`), is held to completion, and is explicitly reset at every episode boundary, replacing unbounded frame-level jump decisions in the `macro` policy.
- **Closed-loop DAgger (#30)** — `dagger.py` rolls out the candidate model-only, measures divergence against the teacher schedule, and mines labelled recovery windows matched on the candidate's ground/airborne phase; `closed_loop_dagger.py` chains rollouts → aggregated checksummed dataset → supervised pretraining → model-only evaluation on reserved seeds, with full provenance in the report. Each round's dataset directory is rebuilt from scratch, so re-running a round cannot inherit another dataset's shards, the runner refuses a teacher shard collected in a different environment from the one under test, and `python dagger.py --teacher-dataset <dir>` audits the shard's labelling with no emulator or model.
- **Offline plumbing validation** — `offline_env.py` is a deterministic synthetic stand-in (no `stable-retro` build needed) so the whole DAgger loop and its tests can run in CI; every result produced with it is labelled `offline_synthetic` and can never satisfy the P0 gate.
- **Live web streaming dashboard** — MJPEG video feed + JSON stats endpoint via
  Flask, with the shared SNN core in `simulation.py`.

## Requirements

- Python 3.9+
- A GPU is **not required** — the network is small enough for CPU. For GPU
  acceleration, install the CUDA build of PyTorch matching your driver from
  <https://pytorch.org> instead of the `requirements.txt` default.

## Installation

```sh
git clone https://github.com/EthanThatOneKid/super-fly.git
cd super-fly
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
```

Notes:

- `stable-retro` ships prebuilt wheels for common platforms; if yours has none,
  it builds from source and needs a C++ compiler.
- The Super Mario Bros ROM is already vendored at
  `roms/Super Mario Bros. (World).nes` and is imported into the `stable-retro`
  data directory on first run.

## Quick start

Train headless (defaults: 50 episodes, 2000 steps/episode):

```sh
python main.py
```

Train with a live OpenCV telemetry window (press `q` to quit):

```sh
python main.py --render
```

Run the live web dashboard:

```sh
python main.py --web
# or: python web_server.py
# then open http://localhost:5000
```

## Usage

### CLI (`main.py`)

| Flag           | Default                                   | Description                                   |
| -------------- | ----------------------------------------- | --------------------------------------------- |
| `--rom`        | `roms/Super Mario Bros. (World).nes`      | Path to a Super Mario Bros NES ROM            |
| `--episodes`   | `50`                                      | Number of training episodes                   |
| `--max-steps`  | `2000`                                    | Max steps per episode                         |
| `--headless`   | *(off)*                                   | Reserved placeholder flag                     |
| `--render`     | *(off)*                                   | Show live telemetry overlay window (`q` quit) |
| `--web`        | *(off)*                                   | Launch the web dashboard instead of training  |
| `--port`       | `5000`                                    | Port for the web dashboard                    |
| `--lr`         | `0.005`                                   | STDP learning rate                            |
| `--save-path`  | `drosophila_snn.pth`                      | Path to load (if present) and save weights    |

### Web dashboard (`--web`)

The dashboard runs the same `Simulation` loop (`simulation.py`) in a background
thread and exposes:

- `/` — dashboard page with `#video_feed` stream and polled stats
- `/video_feed` — MJPEG stream of the telemetry overlay
- `/stats` — JSON episode stats (`episode`, `max_x`, `best_x`, `completed`, `completion_streak`, `pam`, `ppl1`, `step`)

Model weights are checkpointed to `--save-path` whenever a new best distance is
reached, in both CLI and web modes.

## Project layout

| File             | Purpose                                                              |
| ---------------- | -------------------------------------------------------------------- |
| `simulation.py`  | **Shared core**: `Simulation` (model wiring + `step()`) and `make_env()` |
| `main.py`        | CLI trainer; also the `--web` entry point                            |
| `web_server.py`  | Flask dashboard streaming the `Simulation` loop                      |
| `vision.py`      | `OmmatidiaVisionPreprocessor` — frame → ommatidial spike trains      |
| `connectome.py`  | `DrosophilaConnectomeSNN` + `LIFNeuronLayer`                         |
| `stdp.py`        | `DualDopamineSTDP` — PAM/PPL1-modulated weight updates               |
| `ram_tracker.py` | `MarioRAMTracker` — dopamine from SMB RAM (progress/death)           |
| `eval_harness.py`| Isolated deterministic evaluation harness for SNN performance      |
| `teacher.py`     | Deterministic Level 1-1 teacher trajectory collector              |
| `trajectory.py`  | Versioned compressed trajectory shards, per-shard provenance and checksum validation |
| `pretrain.py`   | Supervised motor eligibility-trace pretraining                    |
| `macro_decoder.py` | Bounded RUN/JUMP macro-action chunk decoder (`macro` policy)     |
| `dagger.py`      | Closed-loop DAgger rollouts, divergence detection, phase-aware recovery windows, teacher-label audit CLI |
| `closed_loop_dagger.py` | Issue #30 experiment runner: baseline + DAgger rounds + report |
| `offline_env.py` | Deterministic synthetic env used to validate the loop without `stable-retro` |
| `REACH_1_1_PLAN.md` | Issue triage and the reach-1-1 acceptance gate                 |
| `telemetry.py`   | `DrosophilaTelemetryOverlay` — layer heatmaps + dopamine gauges      |
| `rom_importer.py`| Copies/imports a NES ROM into stable-retro's data dir               |

## Evaluation & Testing

Run the unit test suite:

```sh
python -m unittest discover -s tests
```

Run the deterministic evaluation harness (eval_mode with seed for reproducible evaluation trajectories without updating weights):

```sh
python eval_harness.py --episodes 5 --max-steps 2000 --seed 42
```

Create a reproducible successful teacher shard, then pre-train the motor layer:

```sh
python teacher.py --rom "roms/Super Mario Bros. (World).nes" --output /tmp/super-fly-teacher-dataset
python pretrain.py --dataset /tmp/super-fly-teacher-dataset --output /tmp/super-fly-pretrained.pth
python eval_harness.py --save-path /tmp/super-fly-pretrained.pth --episodes 5 --max-steps 2000 --seed 42
```

The teacher trajectory is an upper-bound and data-generation tool, not evidence that the SNN has learned. The learned checkpoint must be evaluated with `completion_rate`; max X alone is not a Level 1-1 success.

### Closed-loop DAgger (issue #30)

The measured bottleneck is **closed-loop distribution shift**: the teacher trajectory reaches
the flagpole, but a policy trained only on teacher frames leaves that narrow corridor and
never returns to it. The fix is to train on the states the candidate actually visits.

One DAgger round is: model-only on-policy rollouts → divergence / unrecoverable detection
against the teacher schedule → labelled recovery windows → aggregated teacher + rollout
dataset → supervised pretraining at the chunk cadence → model-only evaluation on the seed set in
play (`--eval-seeds`, dev seeds by default; the reserved gate set is opt-in, because the rounds
are tuned by reading their scores -- see "Tuning seeds and the reported claim" below).

```sh
# Record the teacher shard, then run bounded DAgger rounds on the real ROM
python teacher.py --output data/teacher
python closed_loop_dagger.py --mode dagger --teacher-dataset data/teacher \
    --save-path drosophila_snn.pth --iterations 3

# Model-only baseline only
python closed_loop_dagger.py --mode baseline --save-path drosophila_snn.pth

# Validate the entire loop without the emulator (never gate-eligible)
python closed_loop_dagger.py --mode dagger --offline-env --synthesize-teacher --iterations 2
```

The controller only ever sees visual spikes: a regression test replays identical frames into
two identically seeded models, one of which only ever receives garbage RAM, and requires the
same actions from both. RAM is read after the fact for reward, divergence measurement and
provenance only.

Every result is written to `runs/closed_loop_dagger/report.json` and `report.md` with
checkpoint, ROM and dataset-shard checksums, the git commit, seeds, horizon, settle steps,
action cadence, per-shard origins and explicit `model_only` / `p0_gate_met` flags. A run
that used bootstrap pulses, teacher actions or the synthetic environment can never report
`p0_gate_met: true`.

#### Teacher labelling: progress alone is not enough

The teacher spends most of Level 1-1 airborne, and while it is airborne its recorded action
is usually `run` — it is holding right mid-flight. Matching a candidate to the teacher by
progress alone therefore teaches a *grounded* candidate to run wherever the teacher happened
to fly overhead, which is exactly the state where running is fatal.

Measured on the recorded ROM teacher shard (`python dagger.py --teacher-dataset data/teacher_rom`):

| Quantity | Value |
| --- | --- |
| Teacher macro decisions | 99 (25 jump, 74 run) |
| Run decisions taken mid-flight | **43** (58% of run decisions) |
| Ground states mislabelled by progress-only matching | **1159 of 3244 (35.7%)** |
| Direction of every disagreement | `run` → `jump` (1159 of 1159) |
| Largest contiguous mislabelled stretch | x 399–503, recovered from the takeoff at x 362 |

`TeacherLabeler` now matches on progress **and** phase: the target is the teacher's action
recorded the last time the teacher was at (or before) that progress *in the same ground /
airborne state*. The recovered target is the takeoff chunk that actually clears the obstacle
— for the stall at x 594 that is the jump at x 549, not the mid-flight `run` the old rule
returned. On top of that, a jump target is held for the remainder of the teacher chunk that
produced it (bounded by the decoder's own `MAX_CHUNK_FRAMES`), so a flight in progress is
never relabelled mid-air.

Two properties keep this honest and checkable:

* Phase matching **can only ever add jumps**, never remove one — a grounded candidate is only
  upgraded to the jump the teacher itself used to clear that progress (asserted in the tests).
* Divergence detection still uses the full position-only index, so fixing the labels cannot
  silently change which states are flagged as diverged.

```sh
# Audit a teacher shard's labelling from the shard alone: no emulator, no model
python dagger.py --teacher-dataset data/teacher_rom --sweep-step 1 --output runs/label-audit.json

# Ablate the fix: identical rollouts, progress-only recovery targets
python closed_loop_dagger.py --mode dagger --teacher-dataset data/teacher_rom \
    --label-matching position_only
```

Every round's report records `teacher_labelling` (the audit above) and, per recovery window,
which phase the target was matched in, which teacher chunk it came from, how many frames were
served by the held commitment, and how many of the window's targets progress-only matching
would have got wrong (`label_flips`).

**Ablation on the real ROM** (identical seeds, cadence 15, settle 5, horizon 1600, 30 epochs,
one episode per seed on the reserved set 42/43/44 -- measured before the split, so these are the
seeds the arm was being selected against at the time; the runner now iterates on dev seeds). The rollout is model-only, so it visits the same 131 frames in
both arms and dies in the same place; only the recovery window's supervision differs:

| Arm | window target | window frames mislabelled | motor argmax acc | round-1 best_x | per-seed |
| --- | --- | --- | --- | --- | --- |
| `position_only` | `run` (chunk start x 287) | 0 | 0.647 | 1247 | 700 / 680 / 1247 |
| `phase` (default) | `jump` (chunk start x 249) | **9 of 9** | 0.643 | 1246 | 700 / 680 / 1246 |

**The fix corrects the supervision and does not move the number.** Every frame of the only
window the round mined was being taught the wrong action, and the closed-loop result is
unchanged (1246 vs 1247 — one pixel on one seed). Nine corrected training frames out of 1486
cannot move a readout that sits at ~0.64 chunk accuracy, so this is a *correctness* fix, not the
gate blocker. It is worth keeping for the reason the audit gives: the defect scales with how
much of the level a policy survives (35.7% of ground states are affected), so it would silently
compound in any run that got further than x 312.

Offline validation of this loop (synthetic env, cadence 15, settle 3, seeds 42/43/44 -- also from
before the seed split, 400-step horizon, 2 epochs):

| round | best_x | mean best_x | death rate | model only | windows | dataset samples |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 635 | 528.3 | 1.00 | yes | – | – |
| 1 | 815 | 755.0 | 0.33 | yes | 4 | 1683 |
| 2 | 839 | 663.0 | 0.33 | yes | 31 | 1926 |

These numbers validate plumbing only: the synthetic env is a caricature, no episode
completed, and the results are explicitly excluded from the P0 gate. What they do show is
that recovery windows are mined from genuine divergence (36 and 279 labelled samples),
that the death rate drops from 1/1 to 1/3 episodes, and that the recorded provenance
attributes every sample back to the teacher shard or to a specific rollout window.

The improvement is not monotone, and the table should not be read as one: round 2 has the
best single run (839) but a *worse* mean (663 vs 755), and one of the three reserved seeds
still dies in both rounds. A real result needs more seeds, a longer horizon, and the ROM.

### Real-ROM run (issue #30)

Run inside a Linux container (`python:3.11-slim`, CPU torch, `stable-retro` manylinux
wheel) against the committed ROM, cadence 15, settle 5, horizon 1600, one episode per seed on the
reserved set (42/43/44). **This run predates the seed split** and was part of how the arms were
selected, so it reads as a tuning measurement today; the runner reaches those seeds with
`--eval-seeds 42,43,44`, and `claim_eligible` is what records that a run did. The teacher plan
completes Level 1-1 on this ROM in 1,477 frames
(`max_x` 3243, no death), so the upper bound and the completion detector are both real:

| arm | best_x | mean best_x | death rate | completion | jump decisions (per seed) |
| --- | --- | --- | --- | --- | --- |
| baseline (untrained, same harness) | 898 | 504.0 | 1.00 | 0.00 | 15 / 0 / 2 |
| DAgger round 1 | **1247** | 875.7 | 1.00 | 0.00 | 15 / 14 / 17 |
| reference (mainline temporal decoder, issue #30) | 594 | – | – | – | – |

Round 1 clears the reference by 2.1x and its own untrained baseline by 1.4x, model-only.
It does **not** complete the level, so `p0_gate_met` stays `false`. A second round
continued from round 1's checkpoint *regressed* (1247 -> 899): its calibration metrics
improved (balanced accuracy 0.57 -> 0.66) while its closed-loop score fell, and on one
reserved seed it made zero jump decisions at all. The improvement is therefore not
monotone and the report says so per seed.

**Why the completion gate is still shut.** Pretraining fits `layer3_4`, the readout over a
connectome whose visual layers are normally frozen, and level 1-1 is 100 macro decisions
with a single fatal misjudgement at an obstacle ending the episode, so a per-decision error
rate of ~30% cannot be chained into completion. Two things this run did establish:

* The decision must be read off the two channels supervision trains (`RUN_ACTION` and
  `JUMP_ACTION`). The earlier aggregate counted the jump channel in *both* terms, which
  cancels it, so a learned jump request read as a tie and the controller never jumped:
  zero jump decisions and death at the first obstacle.
* The jump margin has to be **calibrated**, not fixed at zero. The trained evidence is
  offset -- on the teacher's own chunks the mean jump-minus-run evidence is negative for run
  chunks *and* for jump chunks -- so a zero threshold reads as "never jump" however well the
  classes separate. Calibration lifted closed-loop `best_x` from 313 to 1247.

Supervising every frame instead of one target per cadence chunk was tested and is worse
(decision-point separation 0.24-0.29 versus 2.4-5.8), which is why stride is tied to the
action cadence.

### Training the visual pathway (issue #30, step 4)

`pretrain.py --visual-pathway {frozen,linear_feedback}`. `frozen` (the default) fits only
the `layer3_4` motor readout; `linear_feedback` also trains `layer1_2`, `layer2_3` and
`feedback_3_2`, chaining the motor error back to each layer's *output* units through the
transposed next-layer weights (`W_next.T @ e_next`). That is the linear part of backprop
with no autograd and no surrogate derivative, it reduces exactly to the rule the readout
already used, and it adds no state to the checkpoint. Trained layers keep the readout's
invariants (rows re-centred, weights clamped to ±3), and `weight_deltas` records how far
each layer actually moved, so a mode that leaves the pathway at its initialization is
visible rather than assumed.

#### Tuning seeds and the reported claim

Every offline number here used to be measured on whatever seeds the caller passed, and the
default was the same three the closed loop reserves for its published model-only evaluation. That
put the pre-screen's verdicts on the seeds the eventual claim has to be made on, while the arms
were being *chosen* by reading those same numbers -- a learning-rate sweep, a visual-pathway arm,
a margin-calibration method. Re-scoring an arm on the seeds it was selected against does not
un-contaminate it, and a table a reader can re-run is worth little if the arm in it was picked
from the numbers in it.

`seed_policy.py` splits the two roles and refuses to let one set straddle them:

| role | seeds | what may happen on them |
| --- | --- | --- |
| **dev** | 45, 46, 47, 48, 49 | calibration, sweeps, ablations and every "has this arm earned an emulator run" verdict. Re-measurable as often as the work needs. |
| **gate** (reserved) | 42, 43, 44 | the closed loop's model-only evaluation -- the only place a completion claim can be made. **Report-only**: fitting a margin, calibrating a decoder or gating an arm on these raises. |

Concretely:

* `prescreen.py` and `pretrain.py` iterate on dev seeds and **refuse** the reserved set (a
  `ValueError`, not a warning), because the pre-screen's output *is* a gate verdict;
* `closed_loop_dagger.py` defaults `--eval-seeds` to the head of the dev range, and its
  `claim_eligible` flag -- now required by `p0_gate_met` -- is true only for a real ROM run on the
  **whole** reserved set, so a completion seen on dev seeds is a tuning reading, not a claim;
* a seed set that mixes the two, or that is a strict subset of the reserved set, is an error; and
* `replays=N` walks down the dev range instead of counting up from a base seed -- counting up from
  42 is exactly how a tuning run silently landed on the reserved triple in the first place.

`init_seed` (default 42) is a separate axis and deliberately so: it seeds the *weights* of the
untrained baseline, which has to be the initialization the candidate actually started from, so
that the null arm is the same null. It is not an evaluation seed and does not enter the split.

### The pre-screen: how an arm earns an emulator run (issue #30)

`prescreen.py` decides whether an arm is worth an emulator run, and it fits nothing. The default
measurement replays the **teacher's frames in order** with the same decoder the closed loop
drives (`offline_episode.py`): the settle window, the SNN's recurrent state, the decoder's chunk
commitment and its refractory period all evolve as they do in a real episode, and the question
is no longer "how many chunks does it classify correctly" but "how many of the teacher's jumps
does it take, in sequence". Every table still carries an **untrained** arm built from the same
initialization, and every arm is reported per replay and paired against that baseline on the
shared replay seed.

Two statistics come out of one replay, and they answer different questions:

* **`jump_sequence_recall`** -- of the jumps the teacher actually took, how many the controller
took. Teacher-forced, so a miss does not stop later jumps from being counted, which keeps it
informative at the competence levels where a first-miss statistic has saturated. This is the
gate's progress measure.
* **`offline_best_x` / `offline_completion`** -- the compounded episode under a *pessimistic*
  fatality model ("any missed teacher jump ends the run"): where the run dies, and whether it
gets to the flagpole. Reported and **not** gated on, for the reasons below.

Real teacher shard, 20 required jumps over 1,477 frames (99 decisions per replay), **dev seeds
45-49** (5 replays), cadence 15, settle 5 -- the seeds iteration is allowed to use, so this is the
table the verdicts below are read off. `emulator best_x` is the closed-loop score the *same
checkpoint* produced on the ROM:

| arm | jump recall | per replay | jumps taken | spurious | jump rate | paired Δ vs untrained | `best_x` (pessimistic) | per replay | emulator `best_x` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| untrained | 0.800 | 0.80 / 0.80 / 0.90 / 0.80 / 0.70 | 16.0 | 10.4 | 0.267 | – | 437 | 549 / 887 / 249 / 249 / 249 | 898 |
| DAgger round 1 (readout only, 30 epochs) | 0.840 | 0.85 / 0.85 / 0.90 / 0.80 / 0.80 | 16.8 | 12.0 | 0.291 | +0.040 (3/5) | 1277 | 2125 / 1750 / 1075 / 549 / 887 | **1247** |
| DAgger round 2 (continued from round 1) | 0.810 | 0.90 / 0.60 / 0.85 / 0.80 / 0.90 | 16.2 | 12.2 | 0.287 | +0.010 (2/5) | 1322 | 2350 / 362 / 887 / 887 / 2125 | 899 |
| visual pathway @2e-3, 30 epochs | **1.000** | 1.00 x5 | 20.0 | **29.0** | **0.495** | +0.200 (5/5) | 3243 | 3243 / 3243 / 3243 / 3243 / 3243 | not run |

Read honestly:

* **Nothing passes, and the shape of each failure is unchanged.** Round 1 fails four of five
  criteria (completion 0.00, recall 0.840 against the 0.90 floor, paired gain +0.040 against
  +0.10, consistency on 3/5 replays); round 2 fails the same four (0.00, 0.810, +0.010, 2/5); the
  visual-pathway arm fails **only** the jump budget. **No arm has earned an emulator run.**
* **The baseline moves with the seed set, which is why it is in the table.** Untrained recall is
  0.800 on the dev seeds against 0.733 on the reserved three -- same shard, same decoder, same
  metric -- so an absolute recall figure is partly a statement about which seeds were spent. A
  verdict read against a fixed floor alone would be a verdict about the seeds, which is why the
  gate compares every arm to the baseline measured on the *same* replays.
* **The round-1/round-2 ordering does not survive three more replays.** On the reserved three,
  recall was 0.817 vs 0.850: round 2 ahead, against the emulator's 1247 vs 899. On the dev five it
  is 0.840 vs 0.810 -- round 1 ahead, *matching* the emulator. So the "the sequential pre-screen
  does not rank the arms like the emulator" result previously recorded here was measured on the
  seeds the arms were being selected against, and it does not survive the move to the tuning set.
  The honest statement is that the ordering is not stable at this sample size, not that an offline
  replay cannot in principle track a closed loop. What does hold on both sets: `best_x` still
  prefers round 2 (1322 vs 1277) where the emulator has it behind, and teacher forcing still means
  the arm's own trajectory never exists.
* **The trained arms' edge over their own initialization is weak.** Paired deltas are +0.040
  (3/5 replays improved) and +0.010 (2/5), both far short of the +0.10 floor. Round 2's *mean*
  recall sits above the untrained *mean*, but on the paired view it is a coin flip, so the fair
  description of this readout is "at or barely above its initialization". The budget view agrees:
  untrained 0.360, round 1 0.496, round 2 0.504, chance 0.343.
* **`offline_best_x` still saturates and is still high-variance**: 549 / 887 / 249 / 249 / 249 for
  the untrained arm, 2350 / 362 / 887 / 887 / 2125 for round 2. With ~20 required jumps and a
  per-jump success rate below ~0.9, the first miss lands early for every weak arm. Reported and
  not gated on.
* **The visual-pathway arm "completes" the level offline on every replay -- and it is not a
  learner.** Recall 1.000, completion 1.00 on all five replays, reaching the teacher's max_x, paired
  gain +0.200. Its calibration degenerated to "jump on everything", so it spends **49.5%** of its
  decisions jumping, 29 of them spurious. Covering the teacher's 20 jumps inside a 35% budget needs
  at least 20 of ~34 jumps on target, and jumping constantly is not a strategy this replay can
  punish, because the model's own airborne state is never simulated. The rate budget is the *only*
  criterion that catches it, and it does so on both seed sets -- the exploit is a property of the
  metric, not of the seeds.
* The per-replay spread (round 2: 0.60 -> 0.90 on the recall scale; round 1: 549 -> 2125 on
  `best_x`) is **larger than any arm's paired gain**, which is why the verdict demands that every
  replay improve and not just the mean -- and why a five-replay table imposes a stricter
  consistency requirement than the three-replay one it replaces.

**The reserved set carries no pre-screen table any more.** The numbers that used to stand here
(untrained 0.733, round 1 0.817, round 2 0.850 over seeds 42/43/44) were measured on the seeds the
arms were selected against, and `prescreen.py` now refuses that set, so they are not reproducible
through it by design. They stay on the record *as history*, not as a verdict: the verdict is the
dev table above, and the reserved set is where the closed-loop model-only evaluation is reported.

**Two measurements this replaced, for the record.** *Calibrated balanced accuracy* pooled over
replays refitted the decision boundary on the evidence it then scored; at 3 epochs it read
0.54-0.58 for a trained readout against 0.52-0.54 for an untrained one, so it could not separate
learning from the initialization heuristic, and on round 2 it *rose* (0.57 → 0.66) while the
closed-loop score fell (1247 → 899). *Jump-chunk recall at a bounded jump rate* fixed the
fitting and the pooling, and was then replaced because it scored one decision at a time from a
cold state; its numbers are still reported under `budget` in every pre-screen (dev seeds:
untrained 0.360, round 1 0.496, round 2 0.504, visual pathway 0.280, chance 0.343), and it is
the view that correctly rated the visual-pathway arm lowest. Balanced accuracy survives only as the *margin
chooser* for the shipped decoder (`calibrate_jump_margin`), which is validated closed-loop
(313 → 1247).

**What the visual pathway does do**, on the same shard: it raises the uncalibrated per-chunk
argmax accuracy monotonically with its learning rate (0.548 frozen → 0.600 / 0.632 / 0.654 at
5e-4 / 2e-3 / 1e-2) and it moves the layers (relative `weight_deltas` 10x / 34x / 114x for
`layer3_4` / `layer2_3` / `layer1_2`), so features are genuinely being learned. What it does not
do is reach a decision the controller can use: at settle 5 the trained layers saturate the
readout's drive until the evidence stops separating the two chunk classes, so the decision
degenerates -- at 3 epochs as a collapse to always-run, and at 30 epochs as a calibration that
cannot separate the classes at all and therefore jumps on everything.

**Defects this work exposed, both fixed:**

* **An inseparable readout published an always-jump decoder.** When no boundary beats the two
  constant rules the calibration ties, and the tie was broken toward the lowest candidate:
  `-math.inf`, i.e. jump on every decision. The 30-epoch visual-pathway arm hits this, and an
  infinite margin also serialized into the checkpoint as `Infinity`, which is not valid JSON.
  Candidate boundaries are now finite sentinels, and a calibration that cannot separate the
  classes reports `degenerate: true` with the reason next to `jump_rate: 1.0`.
* **A sequence metric that an always-jump policy can win**, caught only because a real checkpoint
  was run through it: covering every stretch of teacher flight scores a perfect recall and a
  completion. The gate now spends a jump budget to close it (see above).

Reproduce the table, with no emulator and no training for the readout arms. The sequential
replay costs about 30 s per arm-replay at settle 5 (1,477 frames × 5 forward passes), which is
~15× the budget view it sits next to:

```bash
python prescreen.py --dataset data/teacher_rom --stride 15 --settle-steps 5 --replays 5 \
  --checkpoint round1=runs/closed_loop_dagger/iter-01/checkpoint.pth \
  --checkpoint round2=runs/closed_loop_dagger/round2/iter-01/checkpoint.pth \
  --checkpoint feedback=runs/prescreen/feedback.pth \
  --output runs/prescreen/dev_sequence_table.json
```

That is the table above, read off the dev seeds. No `--seed` flag: the pre-screen spends the
**dev** seeds by default, `--replays` says how many (3 by default, 5 above; `--seeds
45,46,47,48,49` is the same thing spelled out), and passing the reserved gate set (42-44) to it is
an error rather than an option -- see "Tuning seeds and the reported claim" above. Every table
prints its role in the header, so a reader can tell which set a number came from without reading
the command that produced it.

Every `pretrain.py` run assembles the same table -- candidate plus untrained baseline, paired,
with the verdict -- and records it as `metadata["prescreen"]` and under
`pretraining.prescreen` in each closed-loop round report, so a round always carries the offline
measurement that justified it. `prescreen.py` builds each arm from its checkpoint with the
margin that checkpoint actually ships (`policy_config.macro_decoder.jump_margin`), so a
pre-screened arm is the decoder the controller would run:

```bash
python pretrain.py --dataset data/teacher_rom --stride 15 --settle-steps 5 --epochs 30 --seed 42 \
  --visual-pathway linear_feedback --visual-lr 0.002 \
  --report-dataset <held-out tail dataset> --output runs/prescreen/feedback.pth
```

`data/` (teacher shards) and `runs/` (reports, checkpoints, aggregated datasets) are
regenerated artifacts and are not committed.

### Where the ceiling is: the decision rule, or the representation?

Level 1-1 needs ~20 jumps and, under the replay's fatality model, a missed one ends the run, so
completion needs per-jump reliability near 1 -- at 0.90 a run survives its twenty decisions about
12% of the time, and at 0.84, where the best closed-loop arm sits teacher-forced, about 3%. The
answer to "why not" is either a **decoder** change (the rule reading motor spikes is discarding a
distinction the network makes) or a **representation** change (the distinction was never there),
and they have very different price tags. `oracle.py` prices them offline, with no emulator and no
training.

It records every frame of the settle window rather than the sum the shipped decoder reads, so
every aggregation rule can be applied to identical draws; it probes the central-complex population
one layer upstream from the same forward passes; and it reports the best score any of them reaches
on the teacher's own chunks with the threshold *and* the timing chosen from the labels. Because it
peeks, it can only overstate what a causal policy reaches, which is what makes it a ceiling.
Everything is quoted at **bounded jump cost** -- the most jumps catchable while keeping run-chunk
recall at 0.95 -- so a statistic that catches every jump by jumping on everything cannot win, which
is the same exploit the sequence gate had to close.

Two families of motor statistic are scored, and the difference between them turned out to be the
largest decoder finding in the experiment:

* the **binary spike** evidence the shipped decoder reads: the sum, the mean, the last frame, the
  per-frame max, the best and the worst prefix, a leaky accumulator, a normalised contrast, a
  crossing latency, and the two spread/recency shapes below;
* the **analog motor drive** -- the pre-threshold input current `connectome.LIFNeuronLayer`
  integrates and then throws away. A five-frame spike sum takes six values per channel, so a
  threshold on it can only sit between integers; the drive is continuous, so it cannot.

Real teacher shard, 99 chunks (25 of them jumps), dev seeds 45-49, cadence 15. Bounded per-jump
recall, threshold and timing oracled; each cell is the spike family's best -> the drive family's
best (AUC in brackets):

| arm | settle 5 (shipped) | settle 10 | settle 20 |
| --- | --- | --- | --- |
| untrained | 0.440 (0.844) -> **0.784** (0.957) | 0.624 -> 0.976 | 0.952 -> **1.000** |
| DAgger round 1 | 0.656 (0.933) -> **0.920** (0.987) | 0.808 -> 1.000 | 0.928 -> 1.000 |
| DAgger round 2 | 0.648 (0.924) -> **0.928** (0.987) | 0.800 -> 0.984 | 0.944 -> 1.000 |
| visual pathway @2e-3, 30 epochs | 0.000 -> 0.000 | 0.000 -> 0.000 | 0.008 -> 0.024 |

The rule-by-rule view at the window the controller actually ships (round 1, settle 5, five-seed
means) is where the second finding is:

| statistic | AUC | bounded jump recall |
| --- | --- | --- |
| `spike_sum` -- what the decoder shipped with | 0.657 | **0.056** |
| `spike_leaky_early` (best spike rule) | 0.653 | 0.104 |
| `spike_range` (within-window spread, spike) | 0.531 | 0.000 |
| `drive_sum` -- the analog drive, flat | 0.640 | 0.128 |
| `drive_normalized` (scale-free) | 0.633 | 0.104 |
| `drive_leaky_early` @ 0.75 | 0.634 | 0.152 |
| `drive_leaky_recency` @ 0.25 | 0.611 | **0.160** |
| `drive_leaky_recency` @ 0.5 | 0.627 | 0.120 |
| `drive_leaky_recency` @ 0.75 | 0.640 | 0.112 |
| `drive_recency_normalized` @ 0.5 | 0.615 | 0.088 |
| drive timing oracle (best frame per chunk) | 0.987 | 0.920 |
| best linear functional of the drive traces | 0.803 | 0.360 |
| **family ceiling (drive)** | **0.987** | **0.920** |

Read it as four separate facts:

* **The evidence is not saturated -- it is starved.** The ceiling climbs monotonically with the
  number of draws per decision (0.44 -> 0.62 -> 0.95 for the untrained arm, 0.66 -> 0.81 -> 0.93 for
  round 1), and at 20 draws the ordering is essentially perfect (AUC 0.99). Whatever limits this
  arm is not a network that has run out of range; it is a decision taken from very few looks.
* **Reading the sum instead of the drive costs about two thirds of the jumps a fitted threshold
  could catch.** On identical draws at the shipped window, `spike_sum` reaches 0.056 bounded recall
  and `drive_leaky_recency` 0.160 -- with *no* better ordering at all (AUC 0.657 vs 0.611, and the
  drive is the worse of the two). The gain is not ranking; it is that a continuous statistic has
  somewhere for a threshold to sit, and rounding a five-frame window to six levels costs more than
  everything the nine spike rules were searching for. **It also does not transfer to the gate** -- the
  rule that gains here loses on the sequential pre-screen, and `spike_sum` stays the default: see
  "What the decoder reads" below.
* **Which reduction of the drive is used is settled by resolution, not by timing.** The decay sweep
  is inconsistent across arms -- 0.25 beats 0.5 on round 1 and loses on round 2, by less than one
  jump chunk of 25 -- and the flat `drive_sum` (0.128) is close to the best decayed rule (0.160). The
  shipped statistic is recency-weighted because a decision is committed on the frame it is made on
  and that is the defensible shape; the *measurement* above establishes the resolution and nothing
  about the weighting. Every decay in the grid is in the table so the selection is reviewable.
* **At the shipped window the ceiling is 0.92 and the verdict is still `representation`; at ten or
  twenty draws it crosses the 0.97 bar and the verdict becomes `decoder`.** That is worth stating
  plainly because it moves the argument: at settle 10 the drive ceiling is 1.000 for round 1 and
  0.976 for the untrained network, while at settle 5 it is 0.920 and 0.784. The limit at the
  shipped window is the window. The `decoder` verdict rests on a *label-aware timing oracle*, so it
  says a rule that reads the same window as well exists; the best causal rule reaches 0.16, and when
  that rule was actually put through the gate it moved nothing that is rate-free and lost 0.30 of
  sequential recall (see "What the decoder reads" below), which is what leaves the **window** as the
  only lever here that has moved a ceiling anywhere near the 0.97 completion needs.

**The population one layer upstream does not bail the readout out.** A one-dimensional readout of
the central-complex population, with its direction and boundary fitted on training chunks and
scored on held-out ones, reaches 0.448 bounded recall at settle 5 and 0.168 at settle 20, and a
1-nearest-neighbour probe (which assumes no linearity) reaches 0.200 / 0.152. Both are *worse* than
the motor evidence's own ceiling. The Fisher direction fitted on all frames scores 1.000 -- and
that number is a fit, not evidence: 128 dimensions against ~500 frames separates anything. The
shrinkage is swept and the winner picked on the held-out score, so a "not separable" reading cannot
be an artefact of an under-regularised direction.

**The readout is also the only thing that has ever changed.** Re-recording with identical seeds
shows the central population is **bit-identical** between the untrained baseline, DAgger round 1
and DAgger round 2 -- they differ only in `layer3_4`, 512 weights over 128 central units, because
the connectome below the readout is frozen and nothing downstream feeds back into it. The arm that
does change the population (training the visual pathway) saturates it: its motor layer emits no
spikes at all (`motor mean 0.0`, AUC 0.500 at settle 5 and 10), which is why its calibration
degenerated to always-jump and why the sequence pre-screen had to close that hole with a rate
budget. "We trained the network" has so far meant "we trained 512 weights".

Two limits stay on the record. `settle 20` is a *different controller* -- twenty SNN steps per
emulator frame, four times the inference cost, and different adaptation -- so its ceiling is not
something to switch on without re-measuring closed-loop, where only the emulator arbitrates. And
99 chunks is not enough to certify *any* 128-dimensional readout of the population: the held-out
numbers bound what a readout achieves here, not what one would achieve with more labelled chunks,
which is why the verdict field is a pre-committed reading rather than a proof.

```bash
python oracle.py --dataset data/teacher_rom --stride 15 --settle-steps 5,10,20 --replays 5 \
  --checkpoint round1=runs/closed_loop_dagger/iter-01/checkpoint.pth \
  --checkpoint round2=runs/closed_loop_dagger/round2/iter-01/checkpoint.pth \
  --checkpoint feedback=runs/prescreen/feedback.pth \
  --output runs/oracle/ceiling.json
```

### What the decoder reads: the settle-window statistic (issue #30)

The decision rule was never the whole decoder: the controller compares two numbers, and until now
those two numbers were the **sum of binary motor spikes** over the settle window -- six possible
values per channel at settle 5. The ceiling measurement above found that this quantisation, not the
network, was worth the largest single margin in the offline pipeline (0.056 of the jumps a fitted
boundary could catch, against 0.160 for a rule reading the motor layer's pre-threshold *drive* on
identical draws). So the statistic became a parameter:

* `evidence.py` owns the reduction, and `spike_sum` is one of eight rules -- so the change can be
  measured against exactly what it replaces rather than against a memory of it. The rest are
  `spike_*` variants (recency-weighted, and recency-weighted and scale-free) and `drive_*` variants
  that read the pre-threshold input current `connectome.LIFNeuronLayer` integrates and then
  thresholds away: one matrix product, inside a forward pass that has already happened.
* The rule travels **in the checkpoint**, as `policy_config.macro_decoder.evidence_rule` /
  `evidence_decay`, because a threshold and the statistic it thresholds only mean anything together.
  The closed loop restores it with the rest of the policy, so an arm runs the statistic its own
  margin was fitted for.
* A checkpoint that names no rule is one from before the rule travelled in a checkpoint, and the
  only statistic that existed then is the spike sum. When that is the rule being measured its own
  margin is applied unchanged -- a pre-existing checkpoint re-screened under the default reproduces
  the number it always had. Measuring it under a *different* statistic re-fits the margin on the dev
  seeds and records both values: `margin_source` on every row says `checkpoint` or
  `recalibrated_for_evidence_rule`, with the rule, the margin that was replaced and the calibration
  that replaced it. That is the guard against the silent failure this replaces -- a spike-count
  threshold applied to a drive statistic reads as "never jump", and nothing in the report would have
  said which.
* `prescreen.py --evidence-rule NAME` forces one statistic across every arm, baseline included,
  which is how the *statistic* is isolated from each arm's own threshold.

**The replacement was measured, and it did not earn its place.** Dev seeds 45-49, five replays, the
real teacher shard, one run per rule with the margin fitted for whichever statistic is applied.
`seq` is the sequential view the gate reads; `budget` is the rate-free view beside it (jump-chunk
recall at a fixed 0.35 jump budget), which no margin touches:

| statistic forced on every arm | untrained seq / budget | round 1 seq / budget | round 2 seq / budget | jump rate spent (untrained / round 1) |
| --- | --- | --- | --- | --- |
| `spike_sum` (shipped) | 0.660 / **0.360** | **0.840** / 0.496 | 0.520 / **0.504** | 0.192 / 0.291 |
| `spike_leaky_recency` @ 0.25 | 0.760 / 0.344 | 0.720 / 0.480 | 0.750 / 0.416 | 0.248 / 0.234 |
| `drive_sum` | **0.960** / 0.336 | 0.740 / 0.488 | 0.440 / 0.464 | **0.374** / 0.242 |
| `drive_leaky_recency` @ 0.25 | 0.730 / 0.336 | 0.540 / 0.480 | 0.510 / 0.472 | 0.212 / 0.164 |
| `drive_leaky_recency` @ 0.75 | **0.960** / 0.336 | 0.510 / **0.544** | 0.680 / 0.464 | **0.404** / 0.168 |

* **The rate-free view barely moves.** Across all five statistics the untrained baseline's budget
  recall spans 0.336-0.360, round 1's spans 0.480-0.544 and round 2's spans 0.416-0.504, and the AUC
  from the ceiling measurement says the same thing from the other direction: the drive orders the
  chunks *no better* than the sum it would replace (0.611 against 0.657). The evidence does not
  improve; only the *operating point* moves, and there is no more of it than that.
* **The sequential view swings by 0.30 on every arm, and the swing tracks the jump rate spent.**
  Untrained 0.660-0.960 (rate 0.192-0.404), round 1 0.510-0.840 (0.164-0.291), round 2 0.440-0.750
  (0.131-0.232): each statistic's calibrated margin lands somewhere else on the same ROC, and the
  arms that look best are the ones spending the most. The clearest case is the untrained baseline
  with the flat drive: 0.960 while spending **0.374** of its decisions jumping, above the 0.35
  budget, where at its own budget its recall is 0.336 -- the same as the two rules it appears to
  beat. The paired criterion sees the same effect from the other end: against that inflated baseline
  every trained arm "fails" `beats_untrained` (-0.22 and -0.52), where with the shipped sum round 1
  passes it (+0.18 on 5/5 replays) and round 2 does not (+0.01 on 2/5).
* **No arm passes under any statistic**, so the gate's verdicts do not move; and the statistic whose
  settings are *not* re-fitted for it is the one that shipped. `spike_sum` therefore stays the
  default, and the drive rules stay reachable per arm and per checkpoint -- that is what made the
  comparison possible on identical draws. The ordering inside the sequential view is no more stable
  than it was between the DAgger rounds: `spike_leaky_recency` puts round 2 ahead of round 1
  (0.750 vs 0.720) where the shipped sum puts round 1 ahead (0.840 vs 0.520), on the same data and
the same seeds. The ceiling's resolution finding remains a reading of the ceiling's own metric (a
  boundary fitted at a run-chunk recall floor), not yet a claim about the controller.

**A defect in the paired comparison that this exposed, not yet fixed.** The baseline's margin is
calibrated the same way an arm's is, but nothing constrains the *rate* it spends, and the paired
criterion compares the two as if it did. With the flat drive the untrained baseline spends 37.4% of
its decisions jumping against a 35% budget, so an arm that stays inside the budget is measured
against a bar that overspends it. No verdict flips here -- both trained arms fail completion and
recall on their own merits -- but a baseline free to exploit the budget is the same exploit the
budget criterion was added to close, one level up. The fix is to hold the baseline to the same
budget (or compare at a matched rate) before the paired criterion is trusted again.

```bash
# one run per statistic; --evidence-rule forces it on every arm, baseline included
for rule in spike_sum spike_leaky_recency drive_sum drive_leaky_recency; do
  python prescreen.py --dataset data/teacher_rom --stride 15 --settle-steps 5 --replays 5 \
    --evidence-rule "$rule" \
    --checkpoint round1=runs/closed_loop_dagger/iter-01/checkpoint.pth \
    --checkpoint round2=runs/closed_loop_dagger/round2/iter-01/checkpoint.pth \
    --checkpoint feedback=runs/prescreen/feedback.pth \
    --output "runs/prescreen/$rule.json"
done

# and one table for all of them, with a drift check against a published one
python sweep.py --tables runs/prescreen/*.json --reference runs/prescreen/dev_sequence_table.json
```

`sweep.py` is what does the merging by hand otherwise: rows are keyed by the shard checksum, the
protocol, the seed *role* and the statistic each arm ran under, so two cells that differ in any of
those stay apart instead of being averaged together, and a table that produced no file is reported
rather than omitted. `.github/workflows/prescreen-sweep.yml` runs a set of cells as a matrix on
dispatch, one job per cell, and calls the same merge -- with a `reference` input it fails the run
when a published number moves.

## How it works

1. Each NES frame is converted to a synthetic ommatidial grid (28×28 × 5
   channels: edges, right/left motion, down/up motion) and then to Poisson
   spike trains.
2. Spikes propagate through the 4-layer connectome (`connectome.py`); motor output
   is filtered through jump hold/refractory timing.
3. SMB RAM is read each step to produce PAM (progress, obstacle clearance, and
   completion) and PPL1 (death / stagnation) dopamine signals, which update weights
   via inverted-sign STDP (`stdp.py`). Completion requires 30 qualifying flagpole
   frames and is never counted as a death.
4. Offline teacher trajectories can initialize the motor layer with supervised
   eligibility-trace updates before online dopamine-modulated STDP. The pre-trained
   checkpoint is still required to pass model-only evaluation.

## How we teach the fly

The fly is not given a recording of a human player or a list of correct button presses. We teach it with a constrained curriculum:

1. **Keep moving right.** RIGHT is the default action, so the model can focus on learning when to jump instead of learning movement and jumping at the same time.
2. **Create jump opportunities.** During the first 20 episodes, the controller adds short RIGHT+JUMP bootstrap pulses during the first 600 steps of each episode. Model-produced jumps take priority and are held for four frames.
3. **Learn from consequences.** Each frame becomes visual spikes, the SNN chooses an action, and Mario's RAM reports the result. Forward progress produces PAM reward; stagnation and death produce PPL1 punishment; dopamine-modulated STDP adjusts the connections.
4. **Remove the training wheels.** After the first 20 episodes, scheduled jump pulses stop. Model-selected jumps must carry the run, while telemetry distinguishes model jumps from bootstrap assistance.

Bootstrap jumps are exploration, not demonstrations or proof that the model has learned jump timing. The meaningful test is whether model-selected jumps continue to improve progress after bootstrap assistance ends.

## Contributing

Keep the shared brain logic in `simulation.py` and the entrypoints
(`main.py`, `web_server.py`) thin — they should only orchestrate loops, not
reimplement the SNN wiring. Run `python -m py_compile` on changed modules, and
run `python -m unittest discover -s tests` to verify unit test passes.