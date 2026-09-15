import os
import sys
import time
import threading
import cv2
import numpy as np
import torch
from flask import Flask, Response, render_template_string, jsonify

from vision import OmmatidiaVisionPreprocessor
from connectome import DrosophilaConnectomeSNN
from stdp import DualDopamineSTDP
from ram_tracker import MarioRAMTracker
from rom_importer import import_nes_rom
from telemetry import DrosophilaTelemetryOverlay

# Action mapping: [NOOP, RIGHT, JUMP, RIGHT+JUMP]
ACTION_MAP = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0: NOOP
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 1: RIGHT
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # 2: JUMP
    [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],  # 3: RIGHT + JUMP
]

class FlyBrainWebRunner:
    def __init__(self, rom_path="roms/Super Mario Bros. (World).nes", model_path="drosophila_snn.pth"):
        self.rom_path = rom_path
        self.model_path = model_path
        self.latest_jpeg = None
        self.lock = threading.Lock()
        self.running = False
        self.stats = {"episode": 0, "max_x": 0, "pam": 0.0, "ppl1": 0.0, "step": 0}

    def start_simulation(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def _run_loop(self):
        import stable_retro
        if self.rom_path and os.path.exists(self.rom_path):
            import_nes_rom(self.rom_path)

        game_id = "SuperMarioBros-Nes-v0" if "SuperMarioBros-Nes-v0" in stable_retro.data.list_games() else "SuperMarioBros-Nes"
        env = stable_retro.make(game=game_id, state="Level1-1", render_mode=None, use_restricted_actions=stable_retro.Actions.FILTERED)

        preprocessor = OmmatidiaVisionPreprocessor(grid_h=28, grid_w=28)
        model = DrosophilaConnectomeSNN(num_ommatidia=784, channels_per_ommatidium=5)
        stdp = DualDopamineSTDP(model, lr=0.005)
        ram_tracker = MarioRAMTracker()
        telemetry = DrosophilaTelemetryOverlay()

        if os.path.exists(self.model_path):
            model.load_state_dict(torch.load(self.model_path))

        episode = 0
        best_x = 0

        while self.running:
            episode += 1
            obs, info = env.reset()
            preprocessor.reset()
            model.reset_state()
            ram_tracker.reset()

            ep_pam = 0.0
            ep_ppl1 = 0.0
            step = 0

            while self.running and step < 2000:
                step += 1
                features, _ = preprocessor.process_frame(obs)
                spikes = preprocessor.generate_poisson_spikes(features)

                motor_spikes, layer_acts = model(spikes)
                active = torch.where(motor_spikes > 0)[0]
                action_idx = active[0].item() if len(active) > 0 else 1

                retro_action = ACTION_MAP[action_idx]
                obs, reward, terminated, truncated, env_info = env.step(retro_action)

                ram = env.get_ram()
                d_pam, d_ppl1, ram_info = ram_tracker.compute_dopamine(ram, terminated, truncated)
                stdp.step(d_pam, d_ppl1)

                ep_pam += d_pam
                ep_ppl1 += d_ppl1

                # Render canvas & encode JPEG
                canvas = telemetry.render_overlay(obs, layer_acts, d_pam, d_ppl1, ram_info)
                _, jpeg_bytes = cv2.imencode('.jpg', canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

                with self.lock:
                    self.latest_jpeg = jpeg_bytes.tobytes()
                    self.stats = {
                        "episode": episode,
                        "max_x": ram_info['max_x_pos'],
                        "best_x": max(best_x, ram_info['max_x_pos']),
                        "pam": round(ep_pam, 2),
                        "ppl1": round(ep_ppl1, 2),
                        "step": step
                    }

                if ram_info['max_x_pos'] > best_x:
                    best_x = ram_info['max_x_pos']
                    torch.save(model.state_dict(), self.model_path)

                if terminated or truncated:
                    break

                time.sleep(0.01)  # Frame rate limit (~60 fps cap)

        env.close()

runner = FlyBrainWebRunner()
app = Flask(__name__)

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Drosophila Brain SNN - NES Super Mario Bros</title>
    <style>
        body { background-color: #121212; color: #ffffff; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; text-align: center; margin: 0; padding: 20px; }
        h1 { color: #00ffcc; margin-bottom: 5px; }
        p { color: #aaaaaa; }
        .container { display: flex; flex-direction: column; align-items: center; justify-content: center; margin-top: 15px; }
        .stream-card { border: 2px solid #333; border-radius: 8px; box-shadow: 0 4px 20px rgba(0, 255, 204, 0.2); overflow: hidden; max-width: 1280px; }
        img { display: block; width: 100%; height: auto; }
        .stats-panel { display: flex; gap: 30px; margin-top: 20px; font-size: 1.1em; background: #1e1e1e; padding: 15px 30px; border-radius: 8px; }
        .stat-item { display: flex; flex-direction: column; align-items: center; }
        .stat-value { font-weight: bold; font-size: 1.4em; color: #00ffcc; }
    </style>
</head>
<body>
    <h1>Drosophila melanogaster Connectome SNN</h1>
    <p>NES Super Mario Bros 1-1 Online Dopamine STDP Simulation</p>

    <div class="container">
        <div class="stream-card">
            <img src="/video_feed" alt="Fly Brain Telemetry Live Stream" />
        </div>
    </div>

    <script>
        function updateStats() {
            fetch('/stats')
                .then(r => r.json())
                .then(data => {
                    if (data) {
                        console.log(data);
                    }
                });
        }
        setInterval(updateStats, 1000);
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_PAGE)

def generate_frames():
    runner.start_simulation()
    while True:
        with runner.lock:
            frame = runner.latest_jpeg
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.03)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def stats():
    with runner.lock:
        return jsonify(runner.stats)

def start_server(port=5000, rom="roms/Super Mario Bros. (World).nes"):
    runner.rom_path = rom
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)

if __name__ == '__main__':
    start_server()
