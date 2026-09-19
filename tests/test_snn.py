import unittest
import numpy as np
import torch
import os
import tempfile

from ram_tracker import MarioRAMTracker
from connectome import DrosophilaConnectomeSNN, LIFNeuronLayer
from stdp import DualDopamineSTDP
from simulation import Simulation, ACTION_MAP
from telemetry import DrosophilaTelemetryOverlay
from eval_harness import evaluate_agent, evaluate_policies
from web_server import FlyBrainWebRunner, app

class MockEnv:
    """Mock environment for testing Simulation step logic without stable-retro GUI/ROM dependencies."""
    def __init__(self):
        self.ram = np.zeros(0x0700, dtype=np.uint8)
        self.step_count = 0

    def reset(self):
        self.ram = np.zeros(0x0700, dtype=np.uint8)
        self.step_count = 0
        dummy_obs = np.zeros((240, 256, 3), dtype=np.uint8)
        return dummy_obs, {}

    def step(self, action):
        self.step_count += 1
        dummy_obs = np.zeros((240, 256, 3), dtype=np.uint8)
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        return dummy_obs, reward, terminated, truncated, info

    def get_ram(self):
        return self.ram


class TestSuperFlyRegression(unittest.TestCase):

    def test_lif_current_centering_restores_relative_activity(self):
        layer = LIFNeuronLayer(2, 2)
        with torch.no_grad():
            layer.weight.copy_(torch.tensor([[-1.0, -1.0], [-2.0, -2.0]]))

        output = layer(torch.ones(2))

        self.assertEqual(output.tolist(), [1.0, 0.0])

    def test_downstream_layers_use_activity_gain(self):
        model = DrosophilaConnectomeSNN()
        self.assertEqual(model.layer1_2.current_gain, 1.0)
        self.assertEqual(model.layer2_3.current_gain, 6.0)
        self.assertEqual(model.layer3_4.current_gain, 6.0)

    def test_stdp_keeps_weight_rows_centered(self):
        model = DrosophilaConnectomeSNN()
        stdp = DualDopamineSTDP(model)
        with torch.no_grad():
            for layer in (model.layer1_2, model.layer2_3, model.layer3_4):
                layer.trace_pre.fill_(1.0)
                layer.trace_post.fill_(1.0)

        stdp.step(1.0, 0.0)

        for layer in (model.layer1_2, model.layer2_3, model.layer3_4):
            row_means = layer.weight.mean(dim=1)
            self.assertTrue(torch.allclose(row_means, torch.zeros_like(row_means), atol=1e-6))

    def test_ram_tracker_x_pos(self):
        tracker = MarioRAMTracker()
        ram = np.zeros(0x0700, dtype=np.uint8)
        ram[0x006D] = 3  # Level page 3
        ram[0x0086] = 50 # Sub-page X 50
        x_pos = tracker.get_x_pos(ram)
        self.assertEqual(x_pos, 3 * 256 + 50)

    def test_ram_tracker_death_states(self):
        tracker = MarioRAMTracker()
        ram = np.zeros(0x0700, dtype=np.uint8)

        ram[0x000E] = 0x0B # Dying state
        self.assertTrue(tracker.is_dead(ram))

        ram[0x000E] = 0x06 # Dead state
        self.assertTrue(tracker.is_dead(ram))

        ram[0x000E] = 0x08 # Normal state
        self.assertFalse(tracker.is_dead(ram))

    def test_milestone_progression_rewards(self):
        tracker = MarioRAMTracker()
        ram = np.zeros(0x0700, dtype=np.uint8)

        # Initial position
        ram[0x006D] = 0 # Page 0
        ram[0x0086] = 10 # Sub X 10
        d_pam, d_ppl1, info = tracker.compute_dopamine(ram, False, False)
        self.assertEqual(info['sub_page'], 0)
        self.assertEqual(info['page'], 0)

        # Advance to sub-page 1 (x_pos = 64)
        ram[0x0086] = 64
        d_pam, d_ppl1, info = tracker.compute_dopamine(ram, False, False)
        self.assertGreaterEqual(d_pam, 0.3)
        self.assertEqual(info['sub_page'], 1)
        self.assertEqual(info['max_sub_page'], 1)

        # Advance to page 1 (x_pos = 256)
        ram[0x006D] = 1 # Page 1
        ram[0x0086] = 0
        d_pam, d_ppl1, info = tracker.compute_dopamine(ram, False, False)
        self.assertGreaterEqual(d_pam, 0.5)
        self.assertEqual(info['page'], 1)
        self.assertEqual(info['max_page'], 1)

    def test_obstacle_clearance_rewards(self):
        tracker = MarioRAMTracker()
        ram = np.zeros(0x0700, dtype=np.uint8)

        # Ground position x = 100
        ram[0x006D] = 0
        ram[0x0086] = 100
        ram[0x001D] = 0 # Ground
        tracker.compute_dopamine(ram, False, False)

        # Airborne jump start at x = 100
        ram[0x001D] = 1 # Airborne
        ram[0x0086] = 105
        d_pam, d_ppl1, info = tracker.compute_dopamine(ram, False, False)
        self.assertTrue(info['is_airborne'])

        # Mid-air forward progress past obstacle threshold (distance >= 24)
        ram[0x0086] = 128
        d_pam, d_ppl1, info = tracker.compute_dopamine(ram, False, False)
        self.assertGreaterEqual(d_pam, 0.4)

        # Landing
        ram[0x001D] = 0 # On ground
        ram[0x0086] = 130
        d_pam, d_ppl1, info = tracker.compute_dopamine(ram, False, False)
        self.assertFalse(info['is_airborne'])

    def test_collision_speed_drop_punishment(self):
        tracker = MarioRAMTracker()
        ram = np.zeros(0x0700, dtype=np.uint8)

        # Moving forward fast on ground
        ram[0x0086] = 10
        tracker.compute_dopamine(ram, False, False)

        ram[0x0086] = 15 # Speed = 5
        d_pam, d_ppl1, info = tracker.compute_dopamine(ram, False, False)
        self.assertEqual(info['current_speed'], 5)

        # Abrupt collision stop (Speed = 0)
        ram[0x0086] = 15
        d_pam, d_ppl1, info = tracker.compute_dopamine(ram, False, False)
        self.assertGreaterEqual(d_ppl1, 0.3)
        self.assertEqual(info['current_speed'], 0)

    def test_mid_jump_stagnation_punishment(self):
        tracker = MarioRAMTracker()
        ram = np.zeros(0x0700, dtype=np.uint8)

        # Airborne at x = 50
        ram[0x001D] = 1 # Airborne
        ram[0x0086] = 50
        tracker.compute_dopamine(ram, False, False)

        # Mid-jump with zero forward movement
        ram[0x0086] = 50
        d_pam, d_ppl1, info = tracker.compute_dopamine(ram, False, False)
        self.assertGreaterEqual(d_ppl1, 0.2)
        self.assertTrue(info['is_airborne'])

    def test_stdp_dopamine_weight_updates(self):
        model = DrosophilaConnectomeSNN()
        stdp = DualDopamineSTDP(model, lr=0.01)

        # Set non-uniform eligibility traces to prevent row-mean subtraction cancellation
        with torch.no_grad():
            model.layer1_2.trace_post.fill_(1.0)
            model.layer1_2.trace_pre.zero_()
            model.layer1_2.trace_pre[:1000] = 1.0

        initial_w = model.layer1_2.weight.clone()

        # PAM positive reward (d_pam = 1.0, d_ppl1 = 0.0)
        stdp.step(1.0, 0.0)
        self.assertEqual(stdp.last_dopamine_signal, 1.0)

        # Inverted update sign: \Delta w = - \eta * D * E
        # D = 1.0 -> weight values should update
        self.assertFalse(torch.allclose(model.layer1_2.weight, initial_w))

        w_after_pam = model.layer1_2.weight.clone()

        # PPL1 aversive punishment (d_pam = 0.0, d_ppl1 = 1.0)
        stdp.step(0.0, 1.0)
        self.assertEqual(stdp.last_dopamine_signal, -1.0)
        self.assertFalse(torch.allclose(model.layer1_2.weight, w_after_pam))

    def test_jump_refractory_and_hold(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "test_model.pth")
            sim = Simulation(save_path=save_path, bootstrap_episodes=0)
            env = MockEnv()
            obs = sim.reset_episode(env)

            # Force model to output jump on motor layer
            # Thoracic motor ganglion output 2 or 3 triggers jump
            def force_jump_forward(sensory_spikes):
                optic = sim.model.layer1_2(sensory_spikes)
                central = sim.model.layer2_3(optic)
                motor = sim.model.layer3_4(central)
                motor[3] = 1.0 # Force RIGHT+JUMP spike
                return motor, {'ommatidia': sensory_spikes, 'optic_lobe': optic, 'central_complex': central, 'motor_ganglion': motor}

            sim.model.forward = force_jump_forward

            # First step should trigger a model jump
            res1 = sim.step(env, obs)
            self.assertEqual(res1["action_idx"], 3)
            self.assertEqual(res1["action_source"], "model")
            self.assertEqual(res1["model_jumps"], 1)

            # Subsequent 3 steps should hold the jump
            for _ in range(3):
                res = sim.step(env, obs)
                self.assertEqual(res["action_idx"], 3)
                self.assertEqual(res["action_source"], "model_hold")

            # Reset model forward to normal (no jump spikes)
            def normal_forward(sensory_spikes):
                optic = sim.model.layer1_2(sensory_spikes)
                central = sim.model.layer2_3(optic)
                motor = sim.model.layer3_4(central)
                motor.zero_()
                return motor, {'ommatidia': sensory_spikes, 'optic_lobe': optic, 'central_complex': central, 'motor_ganglion': motor}

            sim.model.forward = normal_forward

            # Next steps during refractory period should revert to default action RIGHT (1)
            res_ref = sim.step(env, obs)
            self.assertEqual(res_ref["action_idx"], 1)

    def test_assisted_jump_trace_injection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "test_model.pth")
            sim = Simulation(save_path=save_path, bootstrap_episodes=1, max_bootstrap_step=600)
            env = MockEnv()
            obs = sim.reset_episode(env)

            # Fast forward steps to reach step 128 (first bootstrap pulse)
            for _ in range(127):
                sim.step(env, obs)

            initial_post_trace = sim.model.layer3_4.trace_post[3].item()
            res_pulse = sim.step(env, obs) # Step 128

            self.assertEqual(res_pulse["action_idx"], 3)
            self.assertEqual(res_pulse["action_source"], "bootstrap")
            self.assertGreater(sim.model.layer3_4.trace_post[3].item(), initial_post_trace)

    def test_atomic_checkpoint_recovery(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "test_model.pth")
            sim = Simulation(save_path=save_path)

            # Modify weights
            sim.model.layer1_2.weight.data.add_(1.0)
            sim.save_checkpoint()

            self.assertTrue(os.path.exists(save_path))

            # Load into a new simulation
            sim2 = Simulation(save_path=save_path)
            torch.testing.assert_close(sim.model.layer1_2.weight, sim2.model.layer1_2.weight)

    def test_eval_mode_disables_weight_updates_and_trace_injection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "test_model.pth")
            sim = Simulation(save_path=save_path, bootstrap_episodes=1, max_bootstrap_step=600)
            env = MockEnv()
            obs = sim.reset_episode(env)

            # Fast forward steps to step 128
            for _ in range(127):
                sim.step(env, obs, train=True)

            initial_post_trace = sim.model.layer3_4.trace_post[3].item()
            initial_weight = sim.model.layer1_2.weight.clone()

            # Execute step at 128 with train=False
            res = sim.step(env, obs, train=False)

            self.assertEqual(res["action_source"], "bootstrap")
            # In eval mode, trace_post should NOT be updated by trace injection
            self.assertEqual(sim.model.layer3_4.trace_post[3].item(), initial_post_trace)
            # Weights should remain unchanged
            torch.testing.assert_close(sim.model.layer1_2.weight, initial_weight)
            self.assertTrue(all(not activation.requires_grad for activation in res["layer_acts"].values()))

    def test_telemetry_info_structure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "test_model.pth")
            sim = Simulation(save_path=save_path)
            env = MockEnv()
            obs = sim.reset_episode(env)

            res = sim.step(env, obs)
            self.assertIn("telemetry_info", res)
            telemetry = res["telemetry_info"]
            self.assertIn("action_source", telemetry)
            self.assertIn("model_jumps", telemetry)
            self.assertIn("assisted_jumps", telemetry)
            self.assertIn("bootstrap_active", telemetry)
            self.assertIn("died", telemetry)

    def test_baseline_policies_isolate_jump_sources(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            def forced_jump(_):
                motor = torch.tensor([0.0, 0.0, 0.0, 1.0])
                return motor, {}

            right_only = Simulation(
                save_path=os.path.join(tmpdir, "right_only.pth"),
                bootstrap_episodes=1,
                max_bootstrap_step=600,
                policy="right_only",
            )
            env = MockEnv()
            obs = right_only.reset_episode(env)
            right_only.model.forward = forced_jump
            result = right_only.step(env, obs, train=False)
            self.assertEqual(result["action_idx"], 1)
            self.assertEqual(result["action_source"], "right")
            self.assertFalse(result["bootstrap_active"])
            self.assertEqual(result["model_jumps"], 0)
            self.assertEqual(result["assisted_jumps"], 0)

            bootstrap_only = Simulation(
                save_path=os.path.join(tmpdir, "bootstrap_only.pth"),
                bootstrap_episodes=1,
                max_bootstrap_step=600,
                policy="bootstrap_only",
            )
            env = MockEnv()
            obs = bootstrap_only.reset_episode(env)
            bootstrap_only.model.forward = forced_jump
            for _ in range(127):
                bootstrap_only.step(env, obs, train=False)
            result = bootstrap_only.step(env, obs, train=False)
            self.assertEqual(result["action_idx"], 3)
            self.assertEqual(result["action_source"], "bootstrap")
            self.assertTrue(result["bootstrap_active"])
            self.assertEqual(result["model_jumps"], 0)
            self.assertEqual(result["assisted_jumps"], 1)
            self.assertEqual(result["policy"], "bootstrap_only")

    def test_homeostatic_plasticity_threshold_adaptation(self):
        # Test silent neuron lowers threshold and hyperactive neuron raises threshold
        layer = LIFNeuronLayer(2, 2, v_thresh=1.0, target_rate=0.05, eta_homeo=0.01)
        layer.train()

        # Force weight matrix so neuron 0 never spikes and neuron 1 always spikes
        with torch.no_grad():
            layer.weight.copy_(torch.tensor([[-10.0, -10.0], [10.0, 10.0]]))

        initial_v_thresh = layer.v_thresh.clone()

        # Step forward multiple times
        for _ in range(50):
            layer(torch.ones(2))

        # Neuron 0 is silent -> rate_trace < target_rate -> threshold should decrease
        self.assertLess(layer.v_thresh[0].item(), initial_v_thresh[0].item())
        # Neuron 1 is hyperactive -> rate_trace > target_rate -> threshold should increase
        self.assertGreater(layer.v_thresh[1].item(), initial_v_thresh[1].item())

    def test_homeostasis_disabled_in_eval_mode(self):
        layer = LIFNeuronLayer(2, 2, v_thresh=1.0)
        layer.eval()

        initial_v_thresh = layer.v_thresh.clone()
        for _ in range(20):
            layer(torch.ones(2))

        torch.testing.assert_close(layer.v_thresh, initial_v_thresh)

    def test_weight_initialization_heuristics(self):
        model = DrosophilaConnectomeSNN()

        # Verify shapes
        self.assertEqual(model.layer1_2.weight.shape, (256, 3920))
        self.assertEqual(model.layer2_3.weight.shape, (128, 256))
        self.assertEqual(model.layer3_4.weight.shape, (4, 128))

        # Verify zero-mean row centering across all layers
        for layer in (model.layer1_2, model.layer2_3, model.layer3_4):
            row_means = layer.weight.mean(dim=1)
            self.assertTrue(torch.allclose(row_means, torch.zeros_like(row_means), atol=1e-5))

        # Verify elevated excitation bias for motor neurons (RIGHT, JUMP, RIGHT+JUMP vs NOOP) on primary central complex pathways
        w3 = model.layer3_4.weight
        half_cc = 64
        noop_primary = w3[0, :half_cc].mean().item()
        right_primary = w3[1, :half_cc].mean().item()
        jump_primary = w3[2, :half_cc].mean().item()
        right_jump_primary = w3[3, :half_cc].mean().item()

        self.assertGreater(right_primary, noop_primary)
        self.assertGreater(jump_primary, noop_primary)
        self.assertGreater(right_jump_primary, noop_primary)

    def test_dopamine_breakdown_in_ram_tracker(self):
        tracker = MarioRAMTracker()
        ram = np.zeros(0x0700, dtype=np.uint8)

        # Forward movement
        ram[0x0086] = 50
        d_pam, d_ppl1, info = tracker.compute_dopamine(ram, False, False)
        self.assertIn("dopamine_breakdown", info)
        breakdown = info["dopamine_breakdown"]
        self.assertIn("progress", breakdown)
        self.assertIn("obstacle_clearance", breakdown)
        self.assertIn("stagnation", breakdown)
        self.assertIn("collision", breakdown)
        self.assertIn("death", breakdown)
        self.assertGreater(breakdown["progress"], 0.0)

    def test_telemetry_overlay_rendering(self):
        overlay = DrosophilaTelemetryOverlay()
        obs_frame = np.zeros((240, 256, 3), dtype=np.uint8)
        layer_acts = {
            'ommatidia': torch.zeros(3920),
            'optic_lobe': torch.zeros(256),
            'central_complex': torch.zeros(128),
            'motor_ganglion': torch.zeros(4)
        }
        info = {
            'x_pos': 120,
            'max_x_pos': 120,
            'max_sub_page': 1,
            'max_page': 0,
            'action_source': 'model',
            'model_jumps': 2,
            'assisted_jumps': 0,
            'bootstrap_active': False,
            'dopamine_breakdown': {
                'progress': 0.5,
                'obstacle_clearance': 0.4,
                'stagnation': 0.0,
                'collision': 0.0,
                'death': 0.0
            }
        }
        canvas = overlay.render_overlay(obs_frame, layer_acts, 0.9, 0.0, info)
        self.assertEqual(canvas.shape, (720, 1280, 3))

    def test_eval_harness_benchmark_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "eval_model.pth")
            sim = Simulation(save_path=save_path)
            sim.save_checkpoint()

            # Mock ROM check in evaluate_agent by pointing to non-existent ROM -> returns None
            summary_none = evaluate_agent(rom_path="non_existent_rom.nes", save_path=save_path)
            self.assertIsNone(summary_none)

    def test_web_server_runner_stats(self):
        runner = FlyBrainWebRunner()
        self.assertIn("max_sub_page", runner.stats)
        self.assertIn("max_page", runner.stats)
        self.assertIn("dopamine_breakdown", runner.stats)

        # Flask client test
        app.config['TESTING'] = True
        with app.test_client() as client:
            resp = client.get('/stats')
            self.assertEqual(resp.status_code, 200)
            data = resp.get_json()
            self.assertIn("max_sub_page", data)
            self.assertIn("dopamine_breakdown", data)

    def test_curriculum_decay_schedule(self):
        sim = Simulation(bootstrap_episodes=20, max_bootstrap_step=600, curriculum=True)

        # Phase 1: Episodes 1-10 receive full max_bootstrap_step
        self.assertEqual(sim.get_effective_max_bootstrap_step(1), 600)
        self.assertEqual(sim.get_effective_max_bootstrap_step(10), 600)

        # Phase 2: Episodes 11-20 linearly decay max bootstrap step
        step_11 = sim.get_effective_max_bootstrap_step(11)
        step_15 = sim.get_effective_max_bootstrap_step(15)
        step_20 = sim.get_effective_max_bootstrap_step(20)

        self.assertLess(step_11, 600)
        self.assertLess(step_15, step_11)
        self.assertLess(step_20, step_15)

        # Phase 3: Episode 21+ has 0 bootstrap step limit
        self.assertEqual(sim.get_effective_max_bootstrap_step(21), 0)
        self.assertEqual(sim.get_effective_max_bootstrap_step(50), 0)

        # Curriculum disabled check
        sim_no_curr = Simulation(bootstrap_episodes=20, max_bootstrap_step=600, curriculum=False)
        self.assertEqual(sim_no_curr.get_effective_max_bootstrap_step(15), 600)

    def test_reset_homeostasis(self):
        model = DrosophilaConnectomeSNN()
        model.train()

        for _ in range(10):
            model(torch.rand(3920))

        model.reset_homeostasis()

        self.assertTrue(torch.allclose(model.layer1_2.v_thresh, torch.full_like(model.layer1_2.v_thresh, model.layer1_2.v_thresh_init)))
        self.assertTrue(torch.all(model.layer1_2.rate_trace == 0))

    def test_legacy_state_dict_loading(self):
        model = DrosophilaConnectomeSNN()
        state_dict = model.state_dict()

        # Remove homeostatic keys to simulate legacy checkpoint
        legacy_state_dict = {
            k: v for k, v in state_dict.items()
            if "v_thresh" not in k and "rate_trace" not in k
        }

        new_model = DrosophilaConnectomeSNN()
        # Should load without missing key error due to custom _load_from_state_dict hook
        new_model.load_state_dict(legacy_state_dict, strict=True)
        self.assertEqual(new_model.layer1_2.v_thresh.shape, (256,))


if __name__ == "__main__":
    unittest.main()
