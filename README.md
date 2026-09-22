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

**Why the completion gate is still shut.** Pretraining only ever fits `layer3_4`, the
readout over a fixed random connectome; the visual features feeding it never change. On
the teacher's own chunk decisions that readout reaches a calibrated balanced accuracy of
roughly 0.66-0.70. Level 1-1 is 100 macro decisions and a single fatal misjudgement at an
obstacle ends the episode, so a per-decision error rate of ~30% cannot be chained into
completion. Two things this run did establish:

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