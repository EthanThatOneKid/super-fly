import threading
import time

import cv2
from flask import Flask, Response, render_template_string, jsonify

from simulation import (
    Simulation,
    make_env,
    DEFAULT_ROM_PATH,
    DEFAULT_SAVE_PATH,
    DEFAULT_LR,
    DEFAULT_MAX_STEPS,
)

class FlyBrainWebRunner:
    def __init__(self, rom_path=DEFAULT_ROM_PATH, save_path=DEFAULT_SAVE_PATH,
                 lr=DEFAULT_LR, max_steps=DEFAULT_MAX_STEPS):
        self.rom_path = rom_path
        self.save_path = save_path
        self.lr = lr
        self.max_steps = max_steps
        self.latest_jpeg = None
        self.lock = threading.Lock()
        self.running = False
        self.stats = {"episode": 0, "max_x": 0, "best_x": 0, "pam": 0.0, "ppl1": 0.0, "step": 0}

    def start_simulation(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def _run_loop(self):
        env, _ = make_env(self.rom_path)
        sim = Simulation(rom_path=self.rom_path, save_path=self.save_path, lr=self.lr)

        episode = 0
        try:
            while self.running:
                episode += 1
                obs = sim.reset_episode(env)

                ep_pam = 0.0
                ep_ppl1 = 0.0
                step = 0

                while self.running and step < self.max_steps:
                    step += 1
                    outcome = sim.step(env, obs)
                    obs = outcome["obs"]
                    ep_pam += outcome["d_pam"]
                    ep_ppl1 += outcome["d_ppl1"]

                    sim.maybe_save_record()

                    # Render canvas & encode JPEG
                    canvas = sim.telemetry.render_overlay(
                        obs, outcome["layer_acts"], outcome["d_pam"], outcome["d_ppl1"], outcome["ram_info"]
                    )
                    _, jpeg_bytes = cv2.imencode('.jpg', canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

                    with self.lock:
                        self.latest_jpeg = jpeg_bytes.tobytes()
                        self.stats = {
                            "episode": episode,
                            "max_x": outcome["ram_info"]["max_x_pos"],
                            "best_x": sim.best_x,
                            "pam": round(ep_pam, 2),
                            "ppl1": round(ep_ppl1, 2),
                            "step": step,
                        }

                    if outcome["terminated"] or outcome["truncated"]:
                        break

                    time.sleep(0.01)  # Frame rate limit (~60 fps cap)
        finally:
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

def start_server(port=5000, rom=DEFAULT_ROM_PATH, save_path=DEFAULT_SAVE_PATH,
                 lr=DEFAULT_LR, max_steps=DEFAULT_MAX_STEPS):
    runner.rom_path = rom
    runner.save_path = save_path
    runner.lr = lr
    runner.max_steps = max_steps
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)

if __name__ == '__main__':
    start_server()
