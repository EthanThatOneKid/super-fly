import unittest
import numpy as np
import torch
import os
import tempfile

from ram_tracker import MarioRAMTracker
from connectome import DrosophilaConnectomeSNN, LIFNeuronLayer
from stdp import DualDopamineSTDP
from simulation import Simulation, ACTION_MAP

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


if __name__ == "__main__":
    unittest.main()
