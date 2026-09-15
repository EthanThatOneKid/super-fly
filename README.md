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
- **RAM-based reward** — progress, stagnation, and death are read straight from
  SMB RAM addresses.
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
- `/stats` — JSON episode stats (`episode`, `max_x`, `best_x`, `pam`, `ppl1`, `step`)

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
| `telemetry.py`   | `DrosophilaTelemetryOverlay` — layer heatmaps + dopamine gauges      |
| `rom_importer.py`| Copies/imports a NES ROM into stable-retro's data dir               |

## How it works

1. Each NES frame is converted to a synthetic ommatidial grid (28×28 × 5
   channels: edges, right/left motion, down/up motion) and then to Poisson
   spike trains.
2. Spikes propagate through the 4-layer connectome (`connectome.py`); whichever
   motor neuron fires (or `RIGHT` by default when none do) becomes the action.
3. SMB RAM is read each step to produce PAM (progress) and PPL1 (death /
   stagnation) dopamine signals, which update weights via inverted-sign STDP
   (`stdp.py`).

## Contributing

Keep the shared brain logic in `simulation.py` and the entrypoints
(`main.py`, `web_server.py`) thin — they should only orchestrate loops, not
reimplement the SNN wiring. Run `python -m py_compile` on changed modules; there
is no test suite yet.