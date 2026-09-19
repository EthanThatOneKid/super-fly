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
| `trajectory.py`  | Versioned compressed trajectory shards and checksum validation    |
| `pretrain.py`   | Supervised motor eligibility-trace pretraining                    |
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