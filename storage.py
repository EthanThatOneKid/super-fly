import hashlib
import json
import os
import random
import shutil
import subprocess
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List
import numpy as np
import torch

SCHEMA_VERSION = "1.0"
MAX_RUNS = 20
MAX_STORAGE_BYTES = 100 * 1024 * 1024  # 100 MB
TERMINAL_STATUSES = {"completed", "failed", "cancelled", "waiting", "expired"}


def get_git_commit_sha() -> str:
    """Safely retrieve the current git commit SHA if in a git repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def compute_sha256(filepath: str) -> str:
    """Compute the SHA-256 hex digest of a file on disk."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def atomic_save_torch(data: dict, filepath: str, allow_overwrite: bool = True) -> None:
    """Atomically write a PyTorch object using a temporary file in the same directory."""
    dir_name = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(dir_name, exist_ok=True)
    tmp_path = f"{filepath}.tmp.{os.getpid()}_{time.time_ns()}_{os.urandom(2).hex()}"
    try:
        torch.save(data, tmp_path)
        if not allow_overwrite:
            try:
                os.link(tmp_path, filepath)
            except OSError as e:
                if e.errno == 17 or isinstance(e, FileExistsError):
                    raise FileExistsError(f"File already exists and allow_overwrite is False: {filepath}")
                raise
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        else:
            os.replace(tmp_path, filepath)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def atomic_write_json(data: dict, filepath: str, allow_overwrite: bool = True) -> None:
    """Atomically write JSON metadata using a temporary file in the same directory."""
    dir_name = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(dir_name, exist_ok=True)
    tmp_path = f"{filepath}.tmp.{os.getpid()}_{time.time_ns()}_{os.urandom(2).hex()}"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        if not allow_overwrite:
            try:
                os.link(tmp_path, filepath)
            except OSError as e:
                if e.errno == 17 or isinstance(e, FileExistsError):
                    raise FileExistsError(f"File already exists and allow_overwrite is False: {filepath}")
                raise
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        else:
            os.replace(tmp_path, filepath)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def get_directory_size(path: str) -> int:
    """Calculate total size of all files inside directory in bytes."""
    total_size = 0
    if not os.path.exists(path):
        return 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if not os.path.islink(fp) and os.path.exists(fp):
                total_size += os.path.getsize(fp)
    return total_size


@dataclass
class RunSummary:
    schema_version: str = SCHEMA_VERSION
    run_id: str = ""
    tenant_scope: str = "default"
    repository_id: str = "super-fly"
    status: str = "active"  # active, completed, failed, cancelled, waiting, expired
    created_at: str = ""
    updated_at: str = ""
    git_commit_sha: str = "unknown"
    policy_config: Dict[str, Any] = field(default_factory=dict)
    legacy_backup: Optional[Dict[str, Any]] = None
    checkpoints: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RunEvent:
    event_id: str
    run_id: str
    event_type: str
    timestamp: str
    tenant_scope: str = "default"
    repository_id: str = "super-fly"
    delivery_id: Optional[str] = None
    idempotency_key: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class RunStorage:
    """
    Schema-versioned run and checkpoint manager with bounded retention,
    atomic checkpointing, non-destructive legacy migration, immutable event storage,
    idempotency handling, and complete resume state.
    """

    def __init__(
        self,
        runs_dir: str = "runs",
        run_id: str = None,
        tenant_scope: str = "default",
        repository_id: str = "super-fly",
    ):
        self.runs_dir = os.path.abspath(runs_dir)
        os.makedirs(self.runs_dir, exist_ok=True)
        self.tenant_scope = tenant_scope
        self.repository_id = repository_id

        if run_id is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            rand_suffix = os.urandom(3).hex()
            run_id = f"run_{timestamp}_{rand_suffix}"

        self.run_id = run_id
        self.run_dir = os.path.join(self.runs_dir, self.run_id)
        os.makedirs(self.run_dir, exist_ok=True)

        # Durable layout directories under run-history/v1/<scope>/runs/<runId>/
        scope_str = f"{self.tenant_scope}_{self.repository_id}"
        self.history_v1_dir = os.path.join(
            self.runs_dir, "run-history", "v1", scope_str, "runs", self.run_id
        )
        self.events_dir = os.path.join(self.history_v1_dir, "events")
        self.projections_dir = os.path.join(self.history_v1_dir, "projections")
        self.idempotency_dir = os.path.join(self.history_v1_dir, "idempotency")

        os.makedirs(self.events_dir, exist_ok=True)
        os.makedirs(self.projections_dir, exist_ok=True)
        os.makedirs(self.idempotency_dir, exist_ok=True)

        self.manifest_path = os.path.join(self.run_dir, "run_manifest.json")
        if os.path.exists(self.manifest_path):
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    self.manifest = json.load(f)
            except Exception:
                self.manifest = self._create_initial_manifest()
        else:
            self.manifest = self._create_initial_manifest()
            atomic_write_json(self.manifest, self.manifest_path)

        # Recover projection sequence from existing durable state
        self.projection_sequence = 0
        if os.path.exists(self.projections_dir):
            seqs = []
            for f in os.listdir(self.projections_dir):
                if f.endswith(".json"):
                    stem = f[:-5]
                    if stem.isdigit():
                        seqs.append(int(stem))
            if seqs:
                self.projection_sequence = max(seqs)

    def _create_initial_manifest(self) -> dict:
        now_str = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        summary = RunSummary(
            schema_version=SCHEMA_VERSION,
            run_id=self.run_id,
            tenant_scope=self.tenant_scope,
            repository_id=self.repository_id,
            status="active",
            created_at=now_str,
            updated_at=now_str,
            git_commit_sha=get_git_commit_sha(),
            policy_config={},
            legacy_backup=None,
            checkpoints={},
        )
        return summary.to_dict()

    def record_event(
        self,
        event_type: str,
        data: dict,
        event_id: str = None,
        delivery_id: str = None,
        idempotency_key: str = None,
        allow_overwrite: bool = False,
    ) -> RunEvent:
        """
        Record an immutable event under run-history/v1/<scope>/runs/<runId>/events/<eventId>.json.
        Rejects overwrites if allow_overwrite is False and handles duplicate deliveries safely.
        """
        now_str = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        # Atomic idempotency claim
        if idempotency_key:
            idem_path = os.path.join(self.idempotency_dir, f"{idempotency_key}.json")
            if os.path.exists(idem_path):
                return self._wait_and_load_idempotent_event(idem_path)

            if event_id is None:
                event_id = f"evt_{time.time_ns()}_{os.urandom(2).hex()}"

            tentative_event = RunEvent(
                event_id=event_id,
                run_id=self.run_id,
                event_type=event_type,
                timestamp=now_str,
                tenant_scope=self.tenant_scope,
                repository_id=self.repository_id,
                delivery_id=delivery_id,
                idempotency_key=idempotency_key,
                data=data,
            )

            # Claim idempotency key atomically before writing event/projection files
            claim_data = {"status": "reserved", "event": tentative_event.to_dict()}
            try:
                atomic_write_json(claim_data, idem_path, allow_overwrite=False)
            except FileExistsError:
                # Lost race to concurrent claim -> load winner's record
                return self._wait_and_load_idempotent_event(idem_path)

            # Successfully reserved idempotency key -> write event & projection
            event = tentative_event
            event_file = os.path.join(self.events_dir, f"{event.event_id}.json")
            atomic_write_json(event.to_dict(), event_file, allow_overwrite=allow_overwrite)

            written_proj = False
            while not written_proj:
                self.projection_sequence += 1
                proj_file = os.path.join(self.projections_dir, f"{self.projection_sequence:06d}.json")
                try:
                    atomic_write_json(
                        {"sequence": self.projection_sequence, "event_id": event.event_id, "type": event_type},
                        proj_file,
                        allow_overwrite=False,
                    )
                    written_proj = True
                except FileExistsError:
                    continue

            # Update idempotency claim to complete status
            atomic_write_json({"status": "completed", "event": event.to_dict()}, idem_path, allow_overwrite=True)
            return event

        if event_id is None:
            event_id = f"evt_{time.time_ns()}_{os.urandom(2).hex()}"

        event = RunEvent(
            event_id=event_id,
            run_id=self.run_id,
            event_type=event_type,
            timestamp=now_str,
            tenant_scope=self.tenant_scope,
            repository_id=self.repository_id,
            delivery_id=delivery_id,
            idempotency_key=idempotency_key,
            data=data,
        )

        event_file = os.path.join(self.events_dir, f"{event_id}.json")
        atomic_write_json(event.to_dict(), event_file, allow_overwrite=allow_overwrite)

        written = False
        while not written:
            self.projection_sequence += 1
            proj_file = os.path.join(self.projections_dir, f"{self.projection_sequence:06d}.json")
            try:
                atomic_write_json(
                    {"sequence": self.projection_sequence, "event_id": event_id, "type": event_type},
                    proj_file,
                    allow_overwrite=False,
                )
                written = True
            except FileExistsError:
                continue

        return event

    def _wait_and_load_idempotent_event(self, idem_path: str) -> RunEvent:
        """Poll and load idempotent event record once written by winning worker.
        Reconciles stale/crashed reservations if winning process crashed mid-write.
        """
        for _ in range(50):
            if os.path.exists(idem_path):
                try:
                    with open(idem_path, "r", encoding="utf-8") as f:
                        rec = json.load(f)
                    event_dict = rec.get("event")
                    if event_dict:
                        event_id = event_dict.get("event_id")
                        event_file = os.path.join(self.events_dir, f"{event_id}.json")
                        # If reservation completed and event file exists, return directly
                        if rec.get("status") == "completed" and os.path.exists(event_file):
                            return RunEvent(**event_dict)
                except Exception:
                    pass
            time.sleep(0.01)

        # Timed out waiting or winner crashed while reservation was pending -> reconcile stale reservation
        with open(idem_path, "r", encoding="utf-8") as f:
            rec = json.load(f)

        event_dict = rec["event"]
        event = RunEvent(**event_dict)
        event_file = os.path.join(self.events_dir, f"{event.event_id}.json")

        if not os.path.exists(event_file):
            atomic_write_json(event.to_dict(), event_file, allow_overwrite=True)

        written_proj = False
        while not written_proj:
            self.projection_sequence += 1
            proj_file = os.path.join(self.projections_dir, f"{self.projection_sequence:06d}.json")
            try:
                atomic_write_json(
                    {"sequence": self.projection_sequence, "event_id": event.event_id, "type": event.event_type},
                    proj_file,
                    allow_overwrite=False,
                )
                written_proj = True
            except FileExistsError:
                continue

        atomic_write_json({"status": "completed", "event": event.to_dict()}, idem_path, allow_overwrite=True)
        return event

    def migrate_legacy_checkpoint(self, legacy_path: str) -> dict:
        """
        Non-destructively migrate a legacy checkpoint file (e.g. drosophila_snn.pth).
        Creates a unique timestamped backup without ever overwriting an existing backup,
        and records its relative path and SHA-256 in run_manifest.json.
        """
        if not os.path.exists(legacy_path):
            return None

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_path = f"{legacy_path}.bak.{timestamp}"

        # Ensure uniqueness if a backup with the same timestamp already exists
        if os.path.exists(backup_path):
            counter = 1
            while os.path.exists(f"{backup_path}_{counter}"):
                counter += 1
            backup_path = f"{backup_path}_{counter}"

        # Copy original legacy file to unique backup path
        shutil.copy2(legacy_path, backup_path)
        sha256_val = compute_sha256(backup_path)

        backup_info = {
            "path": os.path.abspath(backup_path),
            "relative_path": os.path.relpath(backup_path, start=self.run_dir),
            "sha256": sha256_val,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        self.manifest["legacy_backup"] = backup_info
        self.manifest["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        atomic_write_json(self.manifest, self.manifest_path)

        self.record_event("legacy_migration", backup_info)
        return backup_info

    def save_checkpoint(
        self,
        simulation,
        is_best: bool = False,
        save_path: str = None,
        seed: int = None,
    ) -> dict:
        """
        Save complete resume state into latest_checkpoint.pth and optionally best_checkpoint.pth.
        """
        model_state = simulation.model.state_dict()

        curriculum_state = {
            "current_state": getattr(simulation, "current_state", "Level1-1"),
            "states": list(getattr(simulation, "states", ["Level1-1"])),
            "curriculum": getattr(simulation, "curriculum", True),
            "bootstrap_episodes": getattr(simulation, "bootstrap_episodes", 20),
            "max_bootstrap_step": getattr(simulation, "max_bootstrap_step", 600),
        }

        policy_config = {
            "policy": getattr(simulation, "policy", "agent"),
            "lr": getattr(simulation, "lr", 0.005),
            "seed": seed,
        }

        rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

        checkpoint_data = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "tenant_scope": self.tenant_scope,
            "repository_id": self.repository_id,
            "model_state_dict": model_state,
            "episode": getattr(simulation, "current_episode", 0),
            "step": getattr(simulation, "current_step", 0),
            "best_x": getattr(simulation, "best_x", 0),
            "curriculum_state": curriculum_state,
            "policy_config": policy_config,
            "rng_states": rng_states,
            "git_commit_sha": get_git_commit_sha(),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        # 1. Save latest_checkpoint.pth inside run_dir
        latest_path = os.path.join(self.run_dir, "latest_checkpoint.pth")
        atomic_save_torch(checkpoint_data, latest_path, allow_overwrite=True)
        latest_sha256 = compute_sha256(latest_path)

        if "checkpoints" not in self.manifest:
            self.manifest["checkpoints"] = {}

        self.manifest["checkpoints"]["latest"] = {
            "filename": "latest_checkpoint.pth",
            "sha256": latest_sha256,
            "episode": checkpoint_data["episode"],
            "step": checkpoint_data["step"],
            "best_x": checkpoint_data["best_x"],
            "updated_at": checkpoint_data["created_at"],
        }

        # 2. Save best_checkpoint.pth if this step achieved a new record
        if is_best:
            best_path = os.path.join(self.run_dir, "best_checkpoint.pth")
            atomic_save_torch(checkpoint_data, best_path, allow_overwrite=True)
            best_sha256 = compute_sha256(best_path)
            self.manifest["checkpoints"]["best"] = {
                "filename": "best_checkpoint.pth",
                "sha256": best_sha256,
                "episode": checkpoint_data["episode"],
                "step": checkpoint_data["step"],
                "best_x": checkpoint_data["best_x"],
                "updated_at": checkpoint_data["created_at"],
            }

        # 3. Synchronize with legacy / custom save_path if provided
        target_path = save_path or getattr(simulation, "save_path", None)
        if target_path:
            atomic_save_torch(checkpoint_data, target_path, allow_overwrite=True)

        self.manifest["policy_config"] = policy_config
        self.manifest["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        atomic_write_json(self.manifest, self.manifest_path, allow_overwrite=True)

        self.record_event(
            "checkpoint_saved",
            {
                "episode": checkpoint_data["episode"],
                "step": checkpoint_data["step"],
                "is_best": is_best,
                "sha256": latest_sha256,
            },
        )

        self.enforce_retention()
        return checkpoint_data

    def load_checkpoint(self, path_or_dir: str = None) -> dict:
        """
        Load checkpoint payload from path or run_dir.
        Handles both schema-versioned checkpoint dicts and legacy raw state dicts.
        """
        target_path = path_or_dir
        if target_path is None:
            target_path = os.path.join(self.run_dir, "latest_checkpoint.pth")
        elif os.path.isdir(target_path):
            latest = os.path.join(target_path, "latest_checkpoint.pth")
            best = os.path.join(target_path, "best_checkpoint.pth")
            if os.path.exists(latest):
                target_path = latest
            elif os.path.exists(best):
                target_path = best

        if not os.path.exists(target_path):
            raise FileNotFoundError(f"Checkpoint not found at: {target_path}")

        payload = torch.load(target_path, weights_only=False)
        return payload

    def set_status(self, status: str) -> None:
        """Set terminal or active status on run manifest (completed, failed, cancelled, waiting, expired)."""
        valid_statuses = {"active", "completed", "failed", "cancelled", "waiting", "expired"}
        if status not in valid_statuses:
            raise ValueError(f"Invalid status '{status}'. Must be one of {valid_statuses}")
        self.manifest["status"] = status
        self.manifest["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        atomic_write_json(self.manifest, self.manifest_path, allow_overwrite=True)
        self.record_event("status_changed", {"status": status})

    @classmethod
    def query_runs(
        cls,
        runs_dir: str = "runs",
        tenant_scope: str = "default",
        repository_id: str = "super-fly",
    ) -> List[dict]:
        """Query runs matching tenant_scope and repository_id filter."""
        abs_runs = os.path.abspath(runs_dir)
        if not os.path.exists(abs_runs):
            return []

        results = []
        for entry in os.listdir(abs_runs):
            if entry == "run-history":
                continue
            full_path = os.path.join(abs_runs, entry)
            manifest_file = os.path.join(full_path, "run_manifest.json")
            if os.path.isdir(full_path) and os.path.exists(manifest_file):
                try:
                    with open(manifest_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if (
                        data.get("tenant_scope") == tenant_scope
                        and data.get("repository_id") == repository_id
                    ):
                        results.append(data)
                except Exception:
                    continue
        return results

    def enforce_retention(self) -> list:
        """
        Keep latest 20 runs OR 100 MB max storage size, whichever limit is reached first.
        Never deletes the active run or the shared run-history directory.
        Returns list of deleted run IDs.
        """
        if not os.path.exists(self.runs_dir):
            return []

        run_dirs = []
        for entry in os.listdir(self.runs_dir):
            if entry == "run-history":
                continue
            full_path = os.path.join(self.runs_dir, entry)
            manifest_file = os.path.join(full_path, "run_manifest.json")
            if os.path.isdir(full_path) and not os.path.islink(full_path) and os.path.exists(manifest_file):
                run_dirs.append(full_path)

        active_real_path = os.path.realpath(self.run_dir)

        def get_mtime(path):
            manifest = os.path.join(path, "run_manifest.json")
            if os.path.exists(manifest):
                return os.path.getmtime(manifest)
            return os.path.getmtime(path)

        inactive_runs = [
            p for p in run_dirs if os.path.realpath(p) != active_real_path
        ]
        inactive_runs.sort(key=get_mtime)

        deleted_runs = []

        # Rule 1: Bound count of runs <= MAX_RUNS (20)
        total_runs_count = len(inactive_runs) + 1  # +1 for active run
        while total_runs_count > MAX_RUNS and inactive_runs:
            oldest_run = inactive_runs.pop(0)
            shutil.rmtree(oldest_run, ignore_errors=True)
            deleted_runs.append(os.path.basename(oldest_run))
            total_runs_count -= 1

        # Rule 2: Bound total storage size across runs <= MAX_STORAGE_BYTES (100 MB)
        def total_bytes():
            return sum(get_directory_size(p) for p in [self.run_dir] + inactive_runs)

        while total_bytes() > MAX_STORAGE_BYTES and inactive_runs:
            oldest_run = inactive_runs.pop(0)
            shutil.rmtree(oldest_run, ignore_errors=True)
            deleted_runs.append(os.path.basename(oldest_run))

        return deleted_runs


# Alias for canonical storage naming
RunHistoryStore = RunStorage
