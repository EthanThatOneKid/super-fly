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
dataset → supervised pretraining at the chunk cadence → model-only evaluation on the
reserved seeds.

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
one episode per reserved seed). The rollout is model-only, so it visits the same 131 frames in
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

Offline validation of this loop (synthetic env, cadence 15, settle 3, seeds 42/43/44,
400-step horizon, 2 epochs):

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
wheel) against the committed ROM, cadence 15, settle 5, horizon 1600, one episode per
reserved seed. The teacher plan completes Level 1-1 on this ROM in 1,477 frames
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

Real teacher shard, 20 required jumps over 1,477 frames (98 decisions), 3 replays (seeds
42/43/44), cadence 15, settle 5. `emulator best_x` is the closed-loop score the *same
checkpoint* produced on the ROM:

| arm | jump recall | per replay | jumps taken | spurious | jump rate | `best_x` (pessimistic) | per replay | emulator `best_x` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| untrained | 0.733 | 0.60 / 0.70 / 0.90 | 14.7 | 11.7 | 0.266 | 899 | 249 / 249 / 2200 | 898 |
| DAgger round 1 (readout only, 30 epochs) | 0.817 | 0.75 / 0.85 / 0.85 | 16.3 | 11.7 | 0.283 | 374 | 249 / 249 / 624 | **1247** |
| DAgger round 2 (continued from round 1) | 0.850 | 0.85 / 0.75 / 0.95 | 17.0 | 12.0 | 0.293 | 712 | 249 / 249 / 1638 | 899 |
| visual pathway @2e-3, 30 epochs | **1.000** | 1.00 / 1.00 / 1.00 | 20.0 | **29.0** | **0.495** | 3243 | 3243 ×3 | not run |

Read honestly:

* **The sequence view separates the arms, and still does not rank them like the emulator.** Jump
  recall climbs 0.733 (untrained) → 0.817 → 0.850, with the ceiling effect gone -- and round 2
  still beats round 1 on recall, on consistency against the untrained baseline (3/3 seeds vs
  2/3) and on `best_x` (712 vs 374), while its emulator `best_x` is the *worse* one (899 vs
  1247). Two offline views have now failed to reproduce that ordering, and the second failure is
  not a scoring artefact: the decisions here really are made in sequence. **Teacher forcing is
  the limit** -- the arm's own trajectory never exists in the replay, so nothing models the
  state its decisions would create, and the emulator's ordering is about dynamics, not decisions.
* **`offline_best_x` saturates and is high-variance**: 249 / 249 / 2200 for the untrained arm,
  249 / 249 / 624 for round 1. With ~20 required jumps and a per-jump success rate below ~0.9,
  the first miss lands early for every weak arm. That is why it is reported and not gated on.
* **The visual-pathway arm "completes" the level offline -- and it is not a learner.** Recall
  1.000, completion 1.00 on all three replays, reaching the teacher's max_x. Its calibration
  degenerated to "jump on everything", so it spends **49.5%** of its decisions jumping, 29 of
  them spurious. Covering the teacher's 20 jumps inside a 35% budget needs at least 20 of ~34
  jumps on target; jumping constantly is not a strategy this replay can punish, because the
  model's own airborne state is never simulated.
* **The rate budget closes that hole, and it is the only criterion that does.** All five gate
  criteria hold for the visual-pathway arm *except* `jump_rate_within_budget` (0.495 vs 0.35).
  Round 2 fails only completion and the 0.90 recall floor (0.850); round 1 fails four of five
  (completion, recall 0.817, paired gain +0.083, and consistency at 2/3 seeds). **No arm passes,
  and the report says which way each one is wrong** -- which is the useful outcome: no emulator
  run is justified yet, and the two arms that look best offline are best for different reasons,
  one of which is a defect.
* The per-replay spread (round 1: 0.62 → 0.90 on the recall scale, 0.249 → 0.624 on `best_x`) is
  **larger than any arm's paired gain**, which is why the verdict demands that every replay
  improve and not just the mean.

**Two measurements this replaced, for the record.** *Calibrated balanced accuracy* pooled over
replays refitted the decision boundary on the evidence it then scored; at 3 epochs it read
0.54-0.58 for a trained readout against 0.52-0.54 for an untrained one, so it could not separate
learning from the initialization heuristic, and on round 2 it *rose* (0.57 → 0.66) while the
closed-loop score fell (1247 → 899). *Jump-chunk recall at a bounded jump rate* fixed the
fitting and the pooling, and was then replaced because it scored one decision at a time from a
cold state; its numbers are still reported under `budget` in every pre-screen (untrained 0.333,
round 1 0.413, round 2 0.427, visual pathway 0.280, chance 0.343), and it is the view that
correctly rated the visual-pathway arm lowest. Balanced accuracy survives only as the *margin
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
python prescreen.py --dataset data/teacher_rom --stride 15 --settle-steps 5 --seed 42 --replays 3 \
  --checkpoint round1=runs/closed_loop_dagger/iter-01/checkpoint.pth \
  --checkpoint round2=runs/closed_loop_dagger/round2/iter-01/checkpoint.pth \
  --output runs/prescreen/sequence_table.json
```

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