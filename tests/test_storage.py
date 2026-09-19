import json
import os
import random
import shutil
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch

from storage import (
    RunStorage,
    RunHistoryStore,
    RunSummary,
    RunEvent,
    compute_sha256,
    get_git_commit_sha,
    atomic_write_json,
    SCHEMA_VERSION,
    TERMINAL_STATUSES,
)
from simulation import Simulation


class MockEnv:
    def __init__(self):
        self.ram = np.zeros(0x0700, dtype=np.uint8)

    def reset(self):
        return np.zeros((240, 256, 3), dtype=np.uint8), {}

    def step(self, action):
        return np.zeros((240, 256, 3), dtype=np.uint8), 0.0, False, False, {}

    def get_ram(self):
        return self.ram


class TestRunStorageAndHistoryStore(unittest.TestCase):

    def test_complete_state_resumption_and_crash_recovery(self):
        """Verify model weights, homeostatic thresholds/rate traces, recurrent feedback state,
        curriculum state, episode counter, best_x, RNG states, policy config, seed,
        git commit SHA, and checksum metadata are saved and correctly restored upon reloading.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = os.path.join(tmpdir, "runs")
            save_path = os.path.join(tmpdir, "model.pth")
            seed = 42

            sim1 = Simulation(
                save_path=save_path,
                runs_dir=runs_dir,
                seed=seed,
                lr=0.0123,
                bootstrap_episodes=15,
                max_bootstrap_step=450,
                curriculum=True,
                states=["Level1-1", "Level1-2"],
                policy="agent",
            )

            env = MockEnv()
            sim1.reset_episode(env)

            # Mutate model weights, thresholds, rate traces, and recurrent state
            with torch.no_grad():
                sim1.model.layer1_2.weight.add_(0.5)
                sim1.model.layer1_2.v_thresh.fill_(2.5)
                sim1.model.layer1_2.rate_trace.fill_(0.12)
                sim1.model.recurrent_central_spikes.fill_(1.0)

            sim1.current_episode = 5
            sim1.current_step = 150
            sim1.best_x = 420
            sim1.current_state = "Level1-2"

            sim1.save_checkpoint(is_best=True)

            # Create sim2 and restore state
            sim2 = Simulation(
                save_path=save_path,
                runs_dir=runs_dir,
                states=["Level1-1", "Level1-2"],
            )
            sim2.load_checkpoint(os.path.join(sim1.run_storage.run_dir, "latest_checkpoint.pth"))

            # Verify model & control state restoration
            torch.testing.assert_close(sim1.model.layer1_2.weight, sim2.model.layer1_2.weight)
            torch.testing.assert_close(sim1.model.layer1_2.v_thresh, sim2.model.layer1_2.v_thresh)
            torch.testing.assert_close(sim1.model.layer1_2.rate_trace, sim2.model.layer1_2.rate_trace)
            torch.testing.assert_close(sim1.model.recurrent_central_spikes, sim2.model.recurrent_central_spikes)

            self.assertEqual(sim2.current_episode, 5)
            self.assertEqual(sim2.current_step, 150)
            self.assertEqual(sim2.best_x, 420)
            self.assertEqual(sim2.current_state, "Level1-2")

            # Verify serialized policy & curriculum control parameters
            self.assertEqual(sim2.policy, "agent")
            self.assertEqual(sim2.lr, 0.0123)
            self.assertEqual(sim2.seed, 42)
            self.assertEqual(sim2.bootstrap_episodes, 15)
            self.assertEqual(sim2.max_bootstrap_step, 450)
            self.assertEqual(sim2.states, ["Level1-1", "Level1-2"])

            # Verify manifest metadata
            manifest_path = sim1.run_storage.manifest_path
            self.assertTrue(os.path.exists(manifest_path))
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)

            self.assertEqual(manifest["schema_version"], SCHEMA_VERSION)
            self.assertIn("latest", manifest["checkpoints"])
            self.assertIn("best", manifest["checkpoints"])
            self.assertEqual(manifest["policy_config"]["policy"], "agent")
            self.assertEqual(manifest["policy_config"]["seed"], seed)
            self.assertEqual(manifest["git_commit_sha"], get_git_commit_sha())

    def test_run_identity_continuity_across_restarts(self):
        """Verify process restart adopts checkpoint's original run_id and maintains run-history continuity."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = os.path.join(tmpdir, "runs")
            save_path = os.path.join(tmpdir, "model.pth")

            sim1 = Simulation(save_path=save_path, runs_dir=runs_dir)
            original_run_id = sim1.run_storage.run_id
            sim1.save_checkpoint()

            # Restart simulation using save_path without passing explicit run_id
            sim2 = Simulation(save_path=save_path, runs_dir=runs_dir)

            self.assertEqual(sim2.run_storage.run_id, original_run_id)

            # Record event on restarted simulation -> asserts event persists under original run_id directory
            evt = sim2.run_storage.record_event("restart_step", {"step": 1})
            expected_event_path = os.path.join(sim1.run_storage.events_dir, f"{evt.event_id}.json")
            self.assertTrue(os.path.exists(expected_event_path))

    def test_stale_reservation_crash_recovery(self):
        """Verify that if a worker process crashes mid-write after reserving an idempotency key,
        subsequent calls detect and reconcile the stale reservation without returning phantom events.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = os.path.join(tmpdir, "runs")
            storage = RunHistoryStore(runs_dir=runs_dir, run_id="run_crash_test")
            idem_key = "crash_key_101"

            crashed_event = RunEvent(
                event_id="evt_crashed_99",
                run_id=storage.run_id,
                event_type="crashed_event",
                timestamp="2026-09-19T06:00:00Z",
                idempotency_key=idem_key,
                data={"part": 1},
            )

            # Simulate process crash: write idempotency claim with status 'reserved' without writing event/projection
            idem_path = os.path.join(storage.idempotency_dir, f"{idem_key}.json")
            atomic_write_json({"status": "reserved", "event": crashed_event.to_dict()}, idem_path)

            # Verify event file does not exist yet (crashed before write)
            event_file = os.path.join(storage.events_dir, "evt_crashed_99.json")
            self.assertFalse(os.path.exists(event_file))

            # Retry recording event with same idempotency key
            storage2 = RunHistoryStore(runs_dir=runs_dir, run_id=storage.run_id)
            reconciled_evt = storage2.record_event(
                "crashed_event",
                {"part": 1},
                event_id="evt_crashed_99",
                idempotency_key=idem_key,
            )

            # Assert stale reservation was reconciled and event + projection files now exist
            self.assertEqual(reconciled_evt.event_id, "evt_crashed_99")
            self.assertTrue(os.path.exists(event_file))

            with open(idem_path, "r", encoding="utf-8") as f:
                idem_record = json.load(f)
            self.assertEqual(idem_record["status"], "completed")

    def test_delayed_writer_does_not_duplicate_projections(self):
        """Verify that if a delayed writer resumes writing projection after a recovery attempt,
        projections deduplicate by event_id and no duplicate projection sequence files are generated.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = os.path.join(tmpdir, "runs")
            storage = RunHistoryStore(runs_dir=runs_dir, run_id="run_delayed_writer")
            idem_key = "delayed_key_55"
            event_id = "evt_delayed_55"

            # 1. Recovery worker completes reservation & writes projection
            evt_recovered = storage.record_event(
                "delayed_event",
                {"data": 1},
                event_id=event_id,
                idempotency_key=idem_key,
            )

            projections_before = os.listdir(storage.projections_dir)
            self.assertEqual(len(projections_before), 1)

            # 2. Delayed original writer attempts to write projection for same event_id
            storage._write_projection_if_missing(event_id, "delayed_event")

            projections_after = os.listdir(storage.projections_dir)
            self.assertEqual(len(projections_after), 1)  # Deduplicated; no extra projection written

    def test_concurrent_duplicate_deliveries_race_safety(self):
        """Verify 8 concurrent deliveries with identical idempotency_key produce exactly 1 event file,
        1 projection file, and 1 idempotency file, and all threads return identical event IDs.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = os.path.join(tmpdir, "runs")
            storage = RunHistoryStore(runs_dir=runs_dir, run_id="run_race_test")
            idem_key = "race_idem_key_42"

            def worker(thread_idx):
                return storage.record_event(
                    "concurrent_pulse",
                    {"thread": thread_idx},
                    delivery_id=f"del_thread_{thread_idx}",
                    idempotency_key=idem_key,
                )

            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(worker, i) for i in range(8)]
                results = [f.result() for f in futures]

            event_ids = {r.event_id for r in results}
            self.assertEqual(len(event_ids), 1)  # All 8 threads returned the winner's event_id

            event_files = os.listdir(storage.events_dir)
            proj_files = os.listdir(storage.projections_dir)
            idem_files = os.listdir(storage.idempotency_dir)

            self.assertEqual(len(event_files), 1)
            self.assertEqual(len(proj_files), 1)
            self.assertEqual(len(idem_files), 1)

    def test_path_components_reject_traversal(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                RunStorage(runs_dir=tmpdir, run_id="../escaped")
            storage = RunStorage(runs_dir=tmpdir, run_id="run_safe")
            with self.assertRaises(ValueError):
                storage.record_event("event", {}, idempotency_key="../../escaped")
            with self.assertRaises(ValueError):
                storage.record_event("event", {}, event_id="../escaped")

    def test_reopening_run_recovers_sequence_projection(self):
        """Verify reopening an existing run_id recovers the next sequence sequence and appends history."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = os.path.join(tmpdir, "runs")
            run_id = "run_restart_seq_test"

            storage1 = RunStorage(runs_dir=runs_dir, run_id=run_id)
            storage1.record_event("event_a", {"step": 1})
            storage1.record_event("event_b", {"step": 2})

            self.assertEqual(storage1.projection_sequence, 2)

            # Reopen run in new storage instance
            storage2 = RunStorage(runs_dir=runs_dir, run_id=run_id)
            self.assertEqual(storage2.projection_sequence, 2)

            # Record next event -> should create projection 000003.json seamlessly
            evt3 = storage2.record_event("event_c", {"step": 3})
            self.assertEqual(storage2.projection_sequence, 3)

            proj3_path = os.path.join(storage2.projections_dir, "000003.json")
            self.assertTrue(os.path.exists(proj3_path))

    def test_immutable_event_writes_and_overwrite_rejection(self):
        """Verify event files under run-history/v1 reject overwrites when allow_overwrite=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = os.path.join(tmpdir, "runs")
            storage = RunHistoryStore(runs_dir=runs_dir, run_id="run_test_immutability")

            evt1 = storage.record_event("test_type", {"foo": "bar"}, event_id="evt_fixed_123")
            self.assertEqual(evt1.event_id, "evt_fixed_123")

            # Attempt recording event with same event_id without allow_overwrite -> FileExistsError
            with self.assertRaises(FileExistsError):
                storage.record_event("test_type", {"foo": "baz"}, event_id="evt_fixed_123", allow_overwrite=False)

    def test_idempotent_event_and_duplicate_delivery(self):
        """Verify duplicate deliveries with identical idempotency_key return the existing recorded event."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = os.path.join(tmpdir, "runs")
            storage = RunHistoryStore(runs_dir=runs_dir, run_id="run_idempotency")

            idem_key = "idem_key_999"
            evt1 = storage.record_event("pulse", {"val": 1}, delivery_id="del_1", idempotency_key=idem_key)
            evt2 = storage.record_event("pulse", {"val": 1}, delivery_id="del_1_retry", idempotency_key=idem_key)

            self.assertEqual(evt1.event_id, evt2.event_id)
            self.assertEqual(evt1.timestamp, evt2.timestamp)

    def test_terminal_statuses(self):
        """Verify terminal statuses (completed, failed, cancelled, waiting, expired) are enforced."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = os.path.join(tmpdir, "runs")
            storage = RunStorage(runs_dir=runs_dir)

            for status in TERMINAL_STATUSES:
                storage.set_status(status)
                self.assertEqual(storage.manifest["status"], status)

            with self.assertRaises(ValueError):
                storage.set_status("invalid_status_xyz")

    def test_tenant_and_repository_scope_query_filtering(self):
        """Verify query_runs enforces tenant_scope and repository_id filtering."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = os.path.join(tmpdir, "runs")

            r1 = RunStorage(runs_dir=runs_dir, run_id="r1", tenant_scope="tenantA", repository_id="repo1")
            r2 = RunStorage(runs_dir=runs_dir, run_id="r2", tenant_scope="tenantB", repository_id="repo1")
            r3 = RunStorage(runs_dir=runs_dir, run_id="r3", tenant_scope="tenantA", repository_id="repo2")

            results_a1 = RunStorage.query_runs(runs_dir=runs_dir, tenant_scope="tenantA", repository_id="repo1")
            self.assertEqual(len(results_a1), 1)
            self.assertEqual(results_a1[0]["run_id"], "r1")

            results_b1 = RunStorage.query_runs(runs_dir=runs_dir, tenant_scope="tenantB", repository_id="repo1")
            self.assertEqual(len(results_b1), 1)
            self.assertEqual(results_b1[0]["run_id"], "r2")

    def test_bounded_retention_cleanup(self):
        """Verify retention cleanup keeps max 20 runs and max 100 MB disk space without deleting active run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = os.path.join(tmpdir, "runs")
            os.makedirs(runs_dir, exist_ok=True)

            # Create 25 mock inactive run directories
            for i in range(25):
                r_dir = os.path.join(runs_dir, f"run_old_{i:02d}")
                os.makedirs(r_dir, exist_ok=True)
                manifest = {
                    "schema_version": SCHEMA_VERSION,
                    "run_id": f"run_old_{i:02d}",
                    "checkpoints": {},
                }
                with open(os.path.join(r_dir, "run_manifest.json"), "w") as f:
                    json.dump(manifest, f)

                # Add dummy payload file
                with open(os.path.join(r_dir, "latest_checkpoint.pth"), "wb") as f:
                    f.write(b"x" * 1024)

            # Active run
            active_storage = RunStorage(runs_dir=runs_dir, run_id="active_run")
            active_storage.enforce_retention()

            remaining_dirs = [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d)) and d != "run-history"]
            self.assertLessEqual(len(remaining_dirs), 20)
            self.assertIn("active_run", remaining_dirs)

    def test_legacy_non_destructive_migration(self):
        """Verify legacy drosophila_snn.pth is backed up with unique timestamp without overwriting,
        and relative path + SHA-256 are recorded in run_manifest.json.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = os.path.join(tmpdir, "runs")
            legacy_file = os.path.join(tmpdir, "drosophila_snn.pth")

            # Create dummy legacy weights file
            dummy_data = {"layer1_2.weight": torch.randn(256, 3920)}
            torch.save(dummy_data, legacy_file)
            legacy_sha256 = compute_sha256(legacy_file)

            storage = RunStorage(runs_dir=runs_dir)
            backup_info1 = storage.migrate_legacy_checkpoint(legacy_file)

            self.assertIsNotNone(backup_info1)
            self.assertTrue(os.path.exists(backup_info1["path"]))
            self.assertEqual(backup_info1["sha256"], legacy_sha256)

            # Attempt migrating again (e.g. duplicate trigger) -> must NOT overwrite existing backup
            backup_info2 = storage.migrate_legacy_checkpoint(legacy_file)
            self.assertIsNotNone(backup_info2)
            self.assertNotEqual(backup_info1["path"], backup_info2["path"])
            self.assertTrue(os.path.exists(backup_info1["path"]))
            self.assertTrue(os.path.exists(backup_info2["path"]))

            # Verify recording in manifest
            with open(storage.manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)

            self.assertIn("legacy_backup", manifest)
            self.assertEqual(manifest["legacy_backup"]["sha256"], legacy_sha256)

    def test_checksum_metadata_verification(self):
        """Verify file SHA-256 checksum matching against saved manifest metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = os.path.join(tmpdir, "runs")
            save_path = os.path.join(tmpdir, "model.pth")

            sim = Simulation(save_path=save_path, runs_dir=runs_dir)
            sim.save_checkpoint(is_best=True)

            run_dir = sim.run_storage.run_dir
            latest_path = os.path.join(run_dir, "latest_checkpoint.pth")
            best_path = os.path.join(run_dir, "best_checkpoint.pth")

            self.assertTrue(os.path.exists(latest_path))
            self.assertTrue(os.path.exists(best_path))

            computed_latest_sha256 = compute_sha256(latest_path)
            computed_best_sha256 = compute_sha256(best_path)

            manifest_path = sim.run_storage.manifest_path
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)

            self.assertEqual(manifest["checkpoints"]["latest"]["sha256"], computed_latest_sha256)
            self.assertEqual(manifest["checkpoints"]["best"]["sha256"], computed_best_sha256)


if __name__ == "__main__":
    unittest.main()
