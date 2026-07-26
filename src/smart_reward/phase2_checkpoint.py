"""Crash-safe, identity-bound checkpoints for long Phase-2 head training.

This module deliberately contains no training-policy decisions.  It only
persists a caller-supplied controller payload, binds it to immutable input
identities, records compact progress receipts, and turns scheduler signals
into a flag that the training loop can service at a safe update boundary.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import signal
import stat
import tempfile
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from types import FrameType
from typing import Any, Literal

import torch

CHECKPOINT_SCHEMA = "prorm-phase2-durable-training-checkpoint/v1"
CHECKPOINT_MANIFEST_SCHEMA = "prorm-phase2-durable-training-checkpoint-manifest/v1"
PROGRESS_SCHEMA = "prorm-phase2-training-progress/v1"
SIGNAL_RECEIPT_SCHEMA = "prorm-phase2-training-signal-receipt/v1"
PLANNED_BOUNDARY_RECEIPT_SCHEMA = "prorm-phase2-planned-segment-boundary-receipt/v1"
TRAINING_PROGRESS_DETAILS_SCHEMA = "prorm-phase2-training-progress-details/v1"
_OBJECTIVE_PATTERN = re.compile(r"[a-z0-9][a-z0-9_.-]*\Z")
_GENERATION_PATTERN = re.compile(r"generation-([0-9]{8})\Z")
_PROGRESS_EVENT_PATTERN = re.compile(r"event-([0-9]{8})\.json\Z")
_HEX = frozenset("0123456789abcdef")
_SAVE_REASONS = frozenset({"interval", "signal", "stage_boundary", "manual"})
_PROGRESS_STATUSES = frozenset(
    {
        "initialized",
        "checkpoint_loaded",
        "resumed",
        "running",
        "finalizing",
        "checkpointed",
        "signal_checkpointed",
        "checkpoint_discovered_after_crash",
        "completed",
        "failed",
    }
)
_TRAINING_PROGRESS_STATUSES = frozenset({"running", "finalizing", "completed", "failed"})
_UNSET = object()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def _nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _finite_nonnegative_float(value: object, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{name} must be a finite non-negative number")
    return float(value)


def _regular_file(path: Path, *, name: str) -> Path:
    if path.is_symlink() or not path.exists() or not stat.S_ISREG(path.stat().st_mode):
        raise ValueError(f"{name} must be an existing regular non-symlink file")
    return path


def _json_mapping(
    value: Mapping[str, object],
    *,
    name: str,
    allow_empty: bool = False,
) -> dict[str, object]:
    if not isinstance(value, Mapping) or (not value and not allow_empty):
        qualifier = "" if allow_empty else " non-empty"
        raise ValueError(f"{name} must be a{qualifier} mapping")
    copied = json.loads(_canonical_json_bytes(dict(value)).decode("utf-8"))
    if not isinstance(copied, dict):
        raise TypeError(f"{name} must serialize to a JSON object")
    return copied


def _read_json(path: Path, *, name: str) -> dict[str, Any]:
    _regular_file(path, name=name)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{name} root must be a JSON object")
    return value


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_canonical_json_bytes(dict(value)))
            stream.write(b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_torch(path: Path, value: Mapping[str, object]) -> None:
    with path.open("wb") as stream:
        torch.save(dict(value), stream)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    """Fsync a publication directory on POSIX; Windows lacks portable dir fsync."""

    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError as error:
        if os.name == "nt":
            return
        raise RuntimeError(f"failed to open checkpoint directory for fsync: {path}") from error
    try:
        os.fsync(descriptor)
    except OSError as error:
        if os.name != "nt":
            raise RuntimeError(f"failed to fsync checkpoint directory: {path}") from error
    finally:
        os.close(descriptor)


def _torch_rng_payload() -> dict[str, object]:
    cuda_states: list[torch.Tensor] = []
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        cuda_states = [
            state.detach().to(device="cpu").contiguous().clone()
            for state in torch.cuda.get_rng_state_all()
        ]
    numpy_state: dict[str, object] | None
    try:
        import numpy as np
    except ModuleNotFoundError:
        numpy_state = None
    else:
        raw_numpy = np.random.get_state()
        numpy_state = {
            "bit_generator": raw_numpy[0],
            "keys": torch.from_numpy(raw_numpy[1].copy()),
            "position": int(raw_numpy[2]),
            "has_gauss": int(raw_numpy[3]),
            "cached_gaussian": float(raw_numpy[4]),
        }
    return {
        "python": random.getstate(),
        "numpy": numpy_state,
        "cpu": torch.get_rng_state().detach().to(device="cpu").contiguous().clone(),
        "cuda": cuda_states,
        "cuda_device_count": len(cuda_states),
    }


def _restore_torch_rng(value: object) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "python",
        "numpy",
        "cpu",
        "cuda",
        "cuda_device_count",
    }:
        raise ValueError("checkpoint torch_rng_state is malformed")
    python_state = value["python"]
    numpy_state = value["numpy"]
    cpu = value["cpu"]
    cuda = value["cuda"]
    count = _nonnegative_integer(value["cuda_device_count"], name="cuda_device_count")
    if (
        not isinstance(cpu, torch.Tensor)
        or cpu.device.type != "cpu"
        or cpu.dtype != torch.uint8
        or cpu.ndim != 1
    ):
        raise ValueError("checkpoint CPU RNG state is malformed")
    if (
        not isinstance(cuda, Sequence)
        or isinstance(cuda, (str, bytes))
        or len(cuda) != count
        or any(
            not isinstance(item, torch.Tensor)
            or item.device.type != "cpu"
            or item.dtype != torch.uint8
            or item.ndim != 1
            for item in cuda
        )
    ):
        raise ValueError("checkpoint CUDA RNG states are malformed")
    if not isinstance(python_state, tuple):
        raise ValueError("checkpoint Python RNG state is malformed")
    random.setstate(python_state)
    if numpy_state is not None:
        if (
            not isinstance(numpy_state, Mapping)
            or set(numpy_state)
            != {
                "bit_generator",
                "keys",
                "position",
                "has_gauss",
                "cached_gaussian",
            }
            or not isinstance(numpy_state["bit_generator"], str)
            or not isinstance(numpy_state["keys"], torch.Tensor)
            or numpy_state["keys"].device.type != "cpu"
            or numpy_state["keys"].ndim != 1
            or numpy_state["keys"].dtype != torch.uint32
        ):
            raise ValueError("checkpoint NumPy RNG state is malformed")
        try:
            import numpy as np
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "checkpoint contains NumPy RNG state but NumPy is unavailable"
            ) from error
        np.random.set_state(
            (
                numpy_state["bit_generator"],
                numpy_state["keys"].numpy().copy(),
                int(numpy_state["position"]),
                int(numpy_state["has_gauss"]),
                float(numpy_state["cached_gaussian"]),
            )
        )
    torch.set_rng_state(cpu)
    if count:
        if not torch.cuda.is_available() or torch.cuda.device_count() != count:
            raise RuntimeError(
                "checkpoint CUDA RNG state count does not match visible CUDA devices"
            )
        torch.cuda.set_rng_state_all(list(cuda))


class DurableCheckpointStore:
    """Immutable checkpoint generations plus identity-bound progress receipts."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        objective: str,
        binding: Mapping[str, object],
    ) -> None:
        if not isinstance(objective, str) or not _OBJECTIVE_PATTERN.fullmatch(objective):
            raise ValueError("objective must match [a-z0-9][a-z0-9_.-]* for safe checkpoint paths")
        self.root = Path(root).resolve()
        if self.root.exists() and (self.root.is_symlink() or not self.root.is_dir()):
            raise ValueError("checkpoint root must be a directory or an absent path")
        self.root.mkdir(parents=True, exist_ok=True)
        self.objective = objective
        self.binding = _json_mapping(binding, name="binding")
        self.binding_sha256 = hashlib.sha256(_canonical_json_bytes(self.binding)).hexdigest()
        self._pending_rng_state: object | None = None
        self.generations_path = self.root / f"{objective}.checkpoints"
        if self.generations_path.exists() and (
            self.generations_path.is_symlink() or not self.generations_path.is_dir()
        ):
            raise ValueError("checkpoint generations path must be a directory")
        self.generations_path.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / f"{objective}.latest.json"
        self.progress_directory = self.root / f"{objective}.progress"
        if self.progress_directory.exists() and (
            self.progress_directory.is_symlink() or not self.progress_directory.is_dir()
        ):
            raise ValueError("checkpoint progress path must be a directory")
        self.progress_directory.mkdir(parents=True, exist_ok=True)
        self.progress_path = self.root / f"{objective}.progress.latest.json"
        self.signal_directory = self.root / f"{objective}.signals"
        if self.signal_directory.exists() and (
            self.signal_directory.is_symlink() or not self.signal_directory.is_dir()
        ):
            raise ValueError("checkpoint signal-receipt path must be a directory")
        self.signal_directory.mkdir(parents=True, exist_ok=True)
        self.planned_boundary_directory = self.root / f"{objective}.planned-boundaries"
        if self.planned_boundary_directory.exists() and (
            self.planned_boundary_directory.is_symlink()
            or not self.planned_boundary_directory.is_dir()
        ):
            raise ValueError("checkpoint planned-boundary receipt path must be a directory")
        self.planned_boundary_directory.mkdir(parents=True, exist_ok=True)

    def _generation_directories(self) -> list[tuple[int, Path]]:
        result: list[tuple[int, Path]] = []
        for path in self.generations_path.iterdir():
            match = _GENERATION_PATTERN.fullmatch(path.name)
            if match is None:
                continue
            if path.is_symlink() or not path.is_dir():
                raise ValueError("checkpoint generation must be a non-symlink directory")
            result.append((int(match.group(1)), path))
        return sorted(result)

    def _next_generation(self) -> int:
        generations = self._generation_directories()
        return 1 if not generations else generations[-1][0] + 1

    def _previous_checkpoint_metadata_sha256(self, generation: int) -> str | None:
        if generation == 1:
            return None
        previous = self.generations_path / f"generation-{generation - 1:08d}"
        if not previous.is_dir() or previous.is_symlink():
            raise RuntimeError("checkpoint generation chain is not contiguous")
        previous_metadata = self._load_generation_metadata(
            generation - 1,
            previous,
            verify_checkpoint_bytes=False,
        )
        if generation > 2:
            predecessor = self.generations_path / f"generation-{generation - 2:08d}"
            predecessor_metadata = self._load_generation_metadata(
                generation - 2,
                predecessor,
                verify_checkpoint_bytes=False,
            )
            expected = _sha256_file(predecessor / "metadata.json")
            if previous_metadata["previous_checkpoint_metadata_sha256"] != expected:
                raise RuntimeError("checkpoint predecessor hash chain is invalid")
            if predecessor_metadata["generation"] != generation - 2:
                raise RuntimeError("checkpoint predecessor generation is invalid")
        return _sha256_file(previous / "metadata.json")

    def save(
        self,
        payload: Mapping[str, object],
        *,
        completed_steps: int,
        reason: Literal["interval", "signal", "stage_boundary", "manual"],
    ) -> dict[str, object]:
        """Atomically publish a new immutable generation and a LATEST hint."""

        steps = _nonnegative_integer(completed_steps, name="completed_steps")
        if reason not in _SAVE_REASONS:
            raise ValueError(f"reason must be one of {sorted(_SAVE_REASONS)!r}")
        if not isinstance(payload, Mapping):
            raise TypeError("checkpoint payload must be a mapping")
        generation = self._next_generation()
        previous_checkpoint_metadata_sha256 = self._previous_checkpoint_metadata_sha256(generation)
        envelope: dict[str, object] = {
            "schema_version": CHECKPOINT_SCHEMA,
            "objective": self.objective,
            "binding": self.binding,
            "binding_sha256": self.binding_sha256,
            "generation": generation,
            "previous_checkpoint_metadata_sha256": (previous_checkpoint_metadata_sha256),
            "completed_steps": steps,
            "save_reason": reason,
            "torch_rng_state": _torch_rng_payload(),
            "payload": dict(payload),
        }
        generation_name = f"generation-{generation:08d}"
        final_directory = self.generations_path / generation_name
        if final_directory.exists() or final_directory.is_symlink():
            raise FileExistsError(f"checkpoint generation already exists: {generation_name}")
        temporary_directory = Path(
            tempfile.mkdtemp(
                prefix=f".{generation_name}.",
                suffix=".tmp",
                dir=self.generations_path,
            )
        )
        checkpoint_path = temporary_directory / "state.pt"
        metadata_path = temporary_directory / "metadata.json"
        committed_path = temporary_directory / "COMMITTED"
        _write_torch(checkpoint_path, envelope)
        checkpoint_sha256 = _sha256_file(checkpoint_path)
        generation_metadata: dict[str, object] = {
            "schema_version": CHECKPOINT_MANIFEST_SCHEMA,
            "objective": self.objective,
            "binding": self.binding,
            "binding_sha256": self.binding_sha256,
            "generation": generation,
            "previous_checkpoint_metadata_sha256": (previous_checkpoint_metadata_sha256),
            "completed_steps": steps,
            "save_reason": reason,
            "generation_directory": generation_name,
            "checkpoint_file": "state.pt",
            "checkpoint_sha256": checkpoint_sha256,
        }
        _atomic_json(metadata_path, generation_metadata)
        metadata_sha256 = _sha256_file(metadata_path)
        with committed_path.open("wb") as stream:
            stream.write(
                _canonical_json_bytes(
                    {
                        "schema_version": "prorm-phase2-checkpoint-commit/v1",
                        "metadata_sha256": metadata_sha256,
                    }
                )
            )
            stream.write(b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_directory(temporary_directory)
        os.replace(temporary_directory, final_directory)
        _fsync_directory(self.generations_path)
        latest: dict[str, object] = {
            **generation_metadata,
            "metadata_sha256": metadata_sha256,
        }
        _atomic_json(self.manifest_path, latest)
        self.record_progress(
            status=("signal_checkpointed" if reason == "signal" else "checkpointed"),
            completed_steps=steps,
            details={
                "generation": generation,
                "save_reason": reason,
                "checkpoint_sha256": checkpoint_sha256,
            },
        )
        return latest

    def _load_generation_metadata(
        self,
        generation: int,
        path: Path,
        *,
        expected_previous_sha256: object = _UNSET,
        verify_checkpoint_bytes: bool = True,
    ) -> dict[str, Any]:
        generation_name = f"generation-{generation:08d}"
        metadata_path = path / "metadata.json"
        committed_path = path / "COMMITTED"
        checkpoint_path = path / "state.pt"
        if not committed_path.exists():
            raise RuntimeError(f"checkpoint {generation_name} lacks COMMITTED marker")
        committed = _read_json(committed_path, name="checkpoint commit marker")
        if (
            set(committed) != {"schema_version", "metadata_sha256"}
            or committed.get("schema_version") != "prorm-phase2-checkpoint-commit/v1"
        ):
            raise ValueError("checkpoint commit marker is invalid")
        expected_metadata_sha = _validate_sha256(
            committed.get("metadata_sha256"),
            name="checkpoint commit metadata_sha256",
        )
        _regular_file(metadata_path, name="checkpoint generation metadata")
        if _sha256_file(metadata_path) != expected_metadata_sha:
            raise RuntimeError("checkpoint generation metadata does not match COMMITTED")
        metadata = _read_json(metadata_path, name="checkpoint generation metadata")
        required = {
            "schema_version",
            "objective",
            "binding",
            "binding_sha256",
            "generation",
            "previous_checkpoint_metadata_sha256",
            "completed_steps",
            "save_reason",
            "generation_directory",
            "checkpoint_file",
            "checkpoint_sha256",
        }
        if set(metadata) != required:
            raise ValueError("checkpoint generation metadata keys are invalid")
        if (
            metadata["schema_version"] != CHECKPOINT_MANIFEST_SCHEMA
            or metadata["objective"] != self.objective
            or metadata["binding"] != self.binding
            or metadata["binding_sha256"] != self.binding_sha256
            or metadata["generation"] != generation
            or metadata["generation_directory"] != generation_name
            or metadata["checkpoint_file"] != "state.pt"
        ):
            raise ValueError("checkpoint generation identity does not match this run")
        previous = metadata["previous_checkpoint_metadata_sha256"]
        if generation == 1:
            if previous is not None:
                raise ValueError("first checkpoint generation cannot name a predecessor")
        else:
            _validate_sha256(
                previous,
                name="previous_checkpoint_metadata_sha256",
            )
        if expected_previous_sha256 is not _UNSET and previous != expected_previous_sha256:
            raise RuntimeError("checkpoint predecessor hash chain is invalid")
        _nonnegative_integer(metadata["completed_steps"], name="completed_steps")
        if metadata["save_reason"] not in _SAVE_REASONS:
            raise ValueError("checkpoint generation save_reason is invalid")
        expected_checkpoint_sha = _validate_sha256(
            metadata["checkpoint_sha256"],
            name="checkpoint generation checkpoint_sha256",
        )
        _regular_file(checkpoint_path, name="checkpoint data")
        if verify_checkpoint_bytes and _sha256_file(checkpoint_path) != expected_checkpoint_sha:
            raise RuntimeError("checkpoint bytes do not match committed metadata")
        return metadata

    def load(
        self,
        *,
        map_location: str | torch.device | None = None,
    ) -> dict[str, object] | None:
        """Load a committed payload; restore RNG only after caller objects exist.

        RNG is never restored inside ``load``.  The caller must first restore
        model, optimizer, and controller state, then call
        :meth:`restore_pending_rng_state` immediately before the next audit or
        update.  This prevents restoration code from advancing the stream.

        ``map_location=None`` is also intentional for resumable GPU state:
        Torch must restore each tensor to its recorded device because the
        controller fingerprint and ProRM dual-state validator bind device
        placement.  ``map_location="cpu"`` is allowed for read-only inspection,
        but such a payload is not valid controller resume input.
        """

        generations = self._generation_directories()
        if not generations:
            if self.manifest_path.exists() or self.manifest_path.is_symlink():
                raise RuntimeError("LATEST checkpoint pointer exists without any generation")
            return None
        if [generation for generation, _ in generations] != list(range(1, len(generations) + 1)):
            raise RuntimeError("checkpoint generation sequence is not contiguous")
        expected_previous_sha256: str | None = None
        metadata: dict[str, Any] | None = None
        for index, (observed_generation, observed_path) in enumerate(generations):
            metadata = self._load_generation_metadata(
                observed_generation,
                observed_path,
                expected_previous_sha256=expected_previous_sha256,
                verify_checkpoint_bytes=index == len(generations) - 1,
            )
            expected_previous_sha256 = _sha256_file(observed_path / "metadata.json")
        if metadata is None:
            raise RuntimeError("checkpoint generation audit produced no metadata")
        generation, generation_path = generations[-1]
        if self.manifest_path.exists() or self.manifest_path.is_symlink():
            latest = _read_json(self.manifest_path, name="LATEST checkpoint pointer")
            required_latest = {
                "schema_version",
                "objective",
                "binding",
                "binding_sha256",
                "generation",
                "previous_checkpoint_metadata_sha256",
                "completed_steps",
                "save_reason",
                "generation_directory",
                "checkpoint_file",
                "checkpoint_sha256",
                "metadata_sha256",
            }
            if set(latest) != required_latest:
                raise ValueError("LATEST checkpoint pointer keys are invalid")
            latest_generation = _nonnegative_integer(
                latest["generation"],
                name="LATEST checkpoint generation",
            )
            if latest_generation > generation:
                raise RuntimeError("LATEST checkpoint pointer names a missing generation")
            # A stale pointer is an allowed crash state: committed generations are
            # authoritative and immutable.  A pointer to the newest generation
            # must, however, agree byte-for-byte with its metadata.
            if latest_generation == generation:
                expected_latest = {
                    **metadata,
                    "metadata_sha256": _sha256_file(generation_path / "metadata.json"),
                }
                if latest != expected_latest:
                    raise RuntimeError("LATEST checkpoint pointer disagrees with generation")
        checkpoint_path = generation_path / "state.pt"
        envelope = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=True,
        )
        if not isinstance(envelope, dict):
            raise ValueError("checkpoint envelope must be a mapping")
        required_envelope = {
            "schema_version",
            "objective",
            "binding",
            "binding_sha256",
            "generation",
            "previous_checkpoint_metadata_sha256",
            "completed_steps",
            "save_reason",
            "torch_rng_state",
            "payload",
        }
        if set(envelope) != required_envelope:
            raise ValueError("checkpoint envelope keys are invalid")
        if (
            envelope["schema_version"] != CHECKPOINT_SCHEMA
            or envelope["objective"] != self.objective
            or envelope["binding"] != self.binding
            or envelope["binding_sha256"] != self.binding_sha256
            or envelope["generation"] != generation
            or envelope["previous_checkpoint_metadata_sha256"]
            != metadata["previous_checkpoint_metadata_sha256"]
            or envelope["completed_steps"] != metadata["completed_steps"]
            or envelope["save_reason"] != metadata["save_reason"]
        ):
            raise ValueError("checkpoint envelope does not match committed metadata")
        payload = envelope["payload"]
        if not isinstance(payload, dict):
            raise ValueError("checkpoint controller payload must be a mapping")
        if self._pending_rng_state is not None:
            raise RuntimeError("a previously loaded checkpoint RNG state is still pending")
        self._pending_rng_state = envelope["torch_rng_state"]
        acknowledged = False
        for path in self.progress_directory.iterdir():
            if _PROGRESS_EVENT_PATTERN.fullmatch(path.name) is None:
                continue
            event = _read_json(path, name="progress event")
            details = event.get("details")
            if (
                event.get("status") in {"checkpointed", "signal_checkpointed"}
                and isinstance(details, Mapping)
                and details.get("generation") == generation
                and details.get("checkpoint_sha256") == metadata["checkpoint_sha256"]
            ):
                acknowledged = True
        if not acknowledged:
            self.record_progress(
                status="checkpoint_discovered_after_crash",
                completed_steps=int(metadata["completed_steps"]),
                details={
                    "generation": generation,
                    "checkpoint_sha256": metadata["checkpoint_sha256"],
                    "committed_generation_was_authoritative": True,
                },
            )
        self.record_progress(
            status="checkpoint_loaded",
            completed_steps=int(metadata["completed_steps"]),
            details={
                "generation": generation,
                "checkpoint_sha256": metadata["checkpoint_sha256"],
                "rng_restored": False,
                "rng_restore_deferred_until_after_trainer_state": True,
            },
        )
        return payload

    def audit_generations(
        self,
        *,
        verify_all_checkpoint_bytes: bool,
    ) -> tuple[dict[str, object], ...]:
        """Verify the complete immutable chain for terminal/finalization evidence."""

        if not isinstance(verify_all_checkpoint_bytes, bool):
            raise TypeError("verify_all_checkpoint_bytes must be bool")
        generations = self._generation_directories()
        if [generation for generation, _ in generations] != list(range(1, len(generations) + 1)):
            raise RuntimeError("checkpoint generation sequence is not contiguous")
        expected_previous_sha256: str | None = None
        audited: list[dict[str, object]] = []
        for generation, path in generations:
            metadata = self._load_generation_metadata(
                generation,
                path,
                expected_previous_sha256=expected_previous_sha256,
                verify_checkpoint_bytes=verify_all_checkpoint_bytes,
            )
            audited.append(dict(metadata))
            expected_previous_sha256 = _sha256_file(path / "metadata.json")
        return tuple(audited)

    def restore_pending_rng_state(self) -> None:
        """Restore RNG after trainer/controller state, then consume the token."""

        if self._pending_rng_state is None:
            raise RuntimeError("no pending checkpoint RNG state is available")
        state = self._pending_rng_state
        _restore_torch_rng(state)
        self._pending_rng_state = None

    def record_progress(
        self,
        *,
        status: Literal[
            "initialized",
            "checkpoint_loaded",
            "resumed",
            "running",
            "finalizing",
            "checkpointed",
            "signal_checkpointed",
            "checkpoint_discovered_after_crash",
            "completed",
            "failed",
        ],
        completed_steps: int,
        details: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        """Publish a compact crash-safe progress snapshot for operator diagnosis."""

        if status not in _PROGRESS_STATUSES:
            raise ValueError(f"invalid progress status {status!r}")
        steps = _nonnegative_integer(completed_steps, name="completed_steps")
        detail_copy = (
            {}
            if details is None
            else _json_mapping(details, name="progress details", allow_empty=True)
        )
        events: list[tuple[int, Path]] = []
        for path in self.progress_directory.iterdir():
            match = _PROGRESS_EVENT_PATTERN.fullmatch(path.name)
            if match is None:
                continue
            _regular_file(path, name="progress event")
            events.append((int(match.group(1)), path))
        events.sort()
        previous_sha256: str | None = None
        for expected_sequence, (sequence, path) in enumerate(events, start=1):
            if sequence != expected_sequence:
                raise RuntimeError("progress event sequence is not contiguous")
            event = _read_json(path, name="progress event")
            required_event = {
                "schema_version",
                "objective",
                "binding",
                "binding_sha256",
                "sequence",
                "previous_progress_sha256",
                "recorded_at_utc",
                "status",
                "completed_steps",
                "details",
            }
            if (
                set(event) != required_event
                or event["schema_version"] != PROGRESS_SCHEMA
                or event["objective"] != self.objective
                or event["binding"] != self.binding
                or event["binding_sha256"] != self.binding_sha256
                or event["sequence"] != sequence
                or event["previous_progress_sha256"] != previous_sha256
            ):
                raise ValueError("progress event identity or predecessor chain is invalid")
            previous_sha256 = _sha256_file(path)
        sequence = len(events) + 1
        record: dict[str, object] = {
            "schema_version": PROGRESS_SCHEMA,
            "objective": self.objective,
            "binding": self.binding,
            "binding_sha256": self.binding_sha256,
            "sequence": sequence,
            "previous_progress_sha256": previous_sha256,
            "recorded_at_utc": datetime.now(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z"),
            "status": status,
            "completed_steps": steps,
            "details": detail_copy,
        }
        event_path = self.progress_directory / f"event-{sequence:08d}.json"
        if event_path.exists() or event_path.is_symlink():
            raise FileExistsError("refusing to overwrite an existing progress event")
        _atomic_json(event_path, record)
        event_sha256 = _sha256_file(event_path)
        latest: dict[str, object] = {
            "schema_version": PROGRESS_SCHEMA,
            "objective": self.objective,
            "binding": self.binding,
            "binding_sha256": self.binding_sha256,
            "sequence": sequence,
            "event_file": event_path.name,
            "event_sha256": event_sha256,
            "previous_progress_sha256": previous_sha256,
        }
        _atomic_json(self.progress_path, latest)
        return record

    def record_training_progress(
        self,
        *,
        status: Literal["running", "finalizing", "completed", "failed"],
        completed_steps: int,
        next_update: int | None,
        learning_rate: float | None,
        gradient_ratio: float | None,
        consecutive_passes: int,
        pcg: Mapping[str, object] | None,
        current_gpu_memory_bytes: int,
        peak_gpu_memory_bytes: int,
        cumulative_training_seconds: float,
        cumulative_audit_seconds: float,
        cumulative_checkpoint_io_seconds: float,
        checkpoint_metadata_sha256: str | None,
        signal_state: Mapping[str, object],
        scheduler_segment: int,
        remaining_allocation_seconds: float | None,
    ) -> dict[str, object]:
        """Validate and append the strict R3 train-only progress contract."""

        if not isinstance(status, str) or status not in _TRAINING_PROGRESS_STATUSES:
            raise ValueError(f"invalid training progress status {status!r}")
        steps = _nonnegative_integer(completed_steps, name="completed_steps")
        if next_update is None:
            if status == "running":
                raise ValueError("running progress requires next_update")
        else:
            update = _nonnegative_integer(next_update, name="next_update")
            if update != steps + 1:
                raise ValueError("next_update must equal completed_steps + 1")
            if status in {"finalizing", "completed"}:
                raise ValueError(f"{status} progress cannot advertise a next_update")
        if learning_rate is None:
            if next_update is not None:
                raise ValueError("an active next_update requires learning_rate")
            validated_learning_rate = None
        else:
            if next_update is None:
                raise ValueError("learning_rate requires an active next_update")
            rate = _finite_nonnegative_float(learning_rate, name="learning_rate")
            if rate == 0.0:
                raise ValueError("learning_rate must be positive")
            validated_learning_rate = rate
        if gradient_ratio is not None:
            gradient_ratio = _finite_nonnegative_float(
                gradient_ratio,
                name="gradient_ratio",
            )
        passes = _nonnegative_integer(
            consecutive_passes,
            name="consecutive_passes",
        )
        pcg_copy: dict[str, object] | None
        if pcg is None:
            pcg_copy = None
        else:
            pcg_copy = _json_mapping(pcg, name="pcg")
            if set(pcg_copy) != {
                "iterations",
                "relative_residual",
                "reason",
                "converged",
            }:
                raise ValueError("pcg progress fields are invalid")
            _nonnegative_integer(pcg_copy["iterations"], name="pcg.iterations")
            _finite_nonnegative_float(
                pcg_copy["relative_residual"],
                name="pcg.relative_residual",
            )
            if not isinstance(pcg_copy["reason"], str) or not pcg_copy["reason"]:
                raise ValueError("pcg.reason must be non-empty")
            if not isinstance(pcg_copy["converged"], bool):
                raise TypeError("pcg.converged must be bool")
        current_memory = _nonnegative_integer(
            current_gpu_memory_bytes,
            name="current_gpu_memory_bytes",
        )
        peak_memory = _nonnegative_integer(
            peak_gpu_memory_bytes,
            name="peak_gpu_memory_bytes",
        )
        if current_memory > peak_memory:
            raise ValueError("current GPU memory cannot exceed peak GPU memory")
        training_seconds = _finite_nonnegative_float(
            cumulative_training_seconds,
            name="cumulative_training_seconds",
        )
        audit_seconds = _finite_nonnegative_float(
            cumulative_audit_seconds,
            name="cumulative_audit_seconds",
        )
        checkpoint_seconds = _finite_nonnegative_float(
            cumulative_checkpoint_io_seconds,
            name="cumulative_checkpoint_io_seconds",
        )
        if checkpoint_metadata_sha256 is not None:
            _validate_sha256(
                checkpoint_metadata_sha256,
                name="checkpoint_metadata_sha256",
            )
        signal_copy = _json_mapping(signal_state, name="signal_state")
        if set(signal_copy) != {
            "requested",
            "signal_name",
            "received_at_utc",
            "additional_signal_count",
        }:
            raise ValueError("signal_state fields are invalid")
        if not isinstance(signal_copy["requested"], bool):
            raise TypeError("signal_state.requested must be bool")
        signal_name = signal_copy["signal_name"]
        received_at = signal_copy["received_at_utc"]
        if signal_copy["requested"]:
            if (
                not isinstance(signal_name, str)
                or not signal_name
                or not isinstance(received_at, str)
                or not received_at.endswith("Z")
            ):
                raise ValueError("requested signal_state lacks name or UTC timestamp")
        elif signal_name is not None or received_at is not None:
            raise ValueError("unrequested signal_state cannot contain signal metadata")
        _nonnegative_integer(
            signal_copy["additional_signal_count"],
            name="signal_state.additional_signal_count",
        )
        segment = _nonnegative_integer(
            scheduler_segment,
            name="scheduler_segment",
        )
        if segment < 1:
            raise ValueError("scheduler_segment must be positive")
        if remaining_allocation_seconds is not None:
            remaining_allocation_seconds = _finite_nonnegative_float(
                remaining_allocation_seconds,
                name="remaining_allocation_seconds",
            )
        details: dict[str, object] = {
            "schema_version": TRAINING_PROGRESS_DETAILS_SCHEMA,
            "monotonic_ns": time.monotonic_ns(),
            "completed_steps": steps,
            "next_update": next_update,
            "learning_rate": validated_learning_rate,
            "gradient_ratio": gradient_ratio,
            "consecutive_passes": passes,
            "pcg": pcg_copy,
            "gpu_memory": {
                "current_bytes": current_memory,
                "peak_bytes": peak_memory,
            },
            "cumulative_elapsed_seconds": {
                "training": training_seconds,
                "audit": audit_seconds,
                "checkpoint_io": checkpoint_seconds,
            },
            "checkpoint_metadata_sha256": checkpoint_metadata_sha256,
            "last_progress_sha256_before_this_event": (
                None
                if not any(self.progress_directory.iterdir())
                else self.latest_progress_sha256()
            ),
            "signal_state": signal_copy,
            "scheduler_segment": segment,
            "remaining_allocation_seconds": remaining_allocation_seconds,
            "information_boundary": "train_only",
        }
        return self.record_progress(
            status=status,
            completed_steps=steps,
            details=details,
        )

    def record_signal_receipt(
        self,
        *,
        head_name: str,
        signal_name: str,
        received_at_utc: str,
        additional_signal_count: int,
        completed_steps: int,
        in_flight_update: int | None,
        reached_safe_boundary: bool,
        checkpoint_metadata_sha256: str | None,
        checkpoint_flush_succeeded: bool,
        checkpoint_verified: bool,
        last_progress_sha256: str,
        scheduler_identity: Mapping[str, object],
        planned_action: Literal[
            "continue_same_logical_run",
            "terminate_completed",
            "fail_closed",
        ],
    ) -> dict[str, object]:
        """Append an immutable receipt after servicing a scheduler signal."""

        if head_name != self.objective:
            raise ValueError("signal receipt head_name must equal the checkpoint objective")
        if not isinstance(signal_name, str) or not signal_name:
            raise ValueError("signal_name must be non-empty")
        if not isinstance(received_at_utc, str) or not received_at_utc.endswith("Z"):
            raise ValueError("received_at_utc must be an explicit UTC timestamp")
        steps = _nonnegative_integer(completed_steps, name="completed_steps")
        extra_signals = _nonnegative_integer(
            additional_signal_count,
            name="additional_signal_count",
        )
        if in_flight_update is not None:
            update = _nonnegative_integer(in_flight_update, name="in_flight_update")
            if update < 1:
                raise ValueError("in_flight_update must be positive when present")
        if not isinstance(reached_safe_boundary, bool):
            raise TypeError("reached_safe_boundary must be bool")
        if not isinstance(checkpoint_flush_succeeded, bool) or not isinstance(
            checkpoint_verified,
            bool,
        ):
            raise TypeError("checkpoint flush/verification flags must be bool")
        if checkpoint_metadata_sha256 is not None:
            _validate_sha256(
                checkpoint_metadata_sha256,
                name="checkpoint_metadata_sha256",
            )
        if checkpoint_verified and (
            not checkpoint_flush_succeeded or checkpoint_metadata_sha256 is None
        ):
            raise ValueError("checkpoint verification requires a flushed checkpoint hash")
        if checkpoint_flush_succeeded and checkpoint_metadata_sha256 is None:
            raise ValueError("checkpoint flush success requires a checkpoint hash")
        _validate_sha256(last_progress_sha256, name="last_progress_sha256")
        scheduler = _json_mapping(scheduler_identity, name="scheduler_identity")
        if planned_action not in {
            "continue_same_logical_run",
            "terminate_completed",
            "fail_closed",
        }:
            raise ValueError("planned_action is invalid")
        continuation_checkpoint_usable = (
            reached_safe_boundary
            and checkpoint_flush_succeeded
            and checkpoint_verified
            and checkpoint_metadata_sha256 is not None
        )
        if planned_action == "continue_same_logical_run" and not continuation_checkpoint_usable:
            raise ValueError("continuation requires a safe, flushed, and verified checkpoint")
        existing: list[tuple[int, Path]] = []
        for path in self.signal_directory.iterdir():
            match = _PROGRESS_EVENT_PATTERN.fullmatch(path.name)
            if match is None:
                continue
            _regular_file(path, name="signal receipt")
            existing.append((int(match.group(1)), path))
        existing.sort()
        previous_sha256: str | None = None
        for expected_sequence, (sequence, path) in enumerate(existing, start=1):
            if sequence != expected_sequence:
                raise RuntimeError("signal receipt sequence is not contiguous")
            previous = _read_json(path, name="signal receipt")
            required_receipt = {
                "schema_version",
                "objective",
                "binding",
                "binding_sha256",
                "sequence",
                "previous_signal_receipt_sha256",
                "head_name",
                "signal_name",
                "received_at_utc",
                "additional_signal_count",
                "receipt_recorded_at_utc",
                "completed_steps",
                "in_flight_update",
                "reached_safe_boundary",
                "checkpoint_metadata_sha256",
                "checkpoint_flush_succeeded",
                "checkpoint_verified",
                "continuation_checkpoint_usable",
                "last_progress_sha256",
                "scheduler_identity",
                "planned_action",
                "terminal_success_claimed",
            }
            if (
                set(previous) != required_receipt
                or previous.get("schema_version") != SIGNAL_RECEIPT_SCHEMA
                or previous.get("objective") != self.objective
                or previous.get("binding") != self.binding
                or previous.get("binding_sha256") != self.binding_sha256
                or previous.get("sequence") != sequence
                or previous.get("previous_signal_receipt_sha256") != previous_sha256
            ):
                raise ValueError("signal receipt identity or predecessor chain is invalid")
            previous_sha256 = _sha256_file(path)
        sequence = len(existing) + 1
        receipt: dict[str, object] = {
            "schema_version": SIGNAL_RECEIPT_SCHEMA,
            "objective": self.objective,
            "binding": self.binding,
            "binding_sha256": self.binding_sha256,
            "sequence": sequence,
            "previous_signal_receipt_sha256": previous_sha256,
            "head_name": head_name,
            "signal_name": signal_name,
            "received_at_utc": received_at_utc,
            "additional_signal_count": extra_signals,
            "receipt_recorded_at_utc": datetime.now(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z"),
            "completed_steps": steps,
            "in_flight_update": in_flight_update,
            "reached_safe_boundary": reached_safe_boundary,
            "checkpoint_metadata_sha256": checkpoint_metadata_sha256,
            "checkpoint_flush_succeeded": checkpoint_flush_succeeded,
            "checkpoint_verified": checkpoint_verified,
            "continuation_checkpoint_usable": continuation_checkpoint_usable,
            "last_progress_sha256": last_progress_sha256,
            "scheduler_identity": scheduler,
            "planned_action": planned_action,
            "terminal_success_claimed": False,
        }
        path = self.signal_directory / f"event-{sequence:08d}.json"
        if path.exists() or path.is_symlink():
            raise FileExistsError("refusing to overwrite an existing signal receipt")
        _atomic_json(path, receipt)
        return receipt

    def latest_progress_sha256(self) -> str:
        """Return the hash of the newest immutable progress event."""

        events: list[tuple[int, Path]] = []
        for path in self.progress_directory.iterdir():
            match = _PROGRESS_EVENT_PATTERN.fullmatch(path.name)
            if match is not None:
                _regular_file(path, name="progress event")
                events.append((int(match.group(1)), path))
        if not events:
            raise RuntimeError("no progress event has been published")
        events.sort()
        if [sequence for sequence, _ in events] != list(range(1, len(events) + 1)):
            raise RuntimeError("progress event sequence is not contiguous")
        return _sha256_file(events[-1][1])

    def record_planned_boundary_receipt(
        self,
        *,
        head_name: str,
        completed_steps: int,
        checkpoint_metadata_sha256: str,
        checkpoint_verified: bool,
        last_progress_sha256: str,
        scheduler_identity: Mapping[str, object],
        execution_slice_sha256: str,
        update_blocks_consumed: int,
        update_blocks_remaining: int,
        planned_action: Literal["continue_same_logical_run", "fail_closed"],
    ) -> dict[str, object]:
        """Append an immutable non-signal segment-boundary receipt.

        A planned allocation boundary is deliberately represented separately
        from a scheduler signal.  This prevents a normal segment cap from
        fabricating signal provenance while still giving the successor segment
        an exact checkpoint/progress/slice binding to validate.
        """

        if head_name != self.objective:
            raise ValueError("planned-boundary head_name must equal the checkpoint objective")
        steps = _nonnegative_integer(completed_steps, name="completed_steps")
        _validate_sha256(
            checkpoint_metadata_sha256,
            name="checkpoint_metadata_sha256",
        )
        if not isinstance(checkpoint_verified, bool):
            raise TypeError("checkpoint_verified must be bool")
        if not checkpoint_verified:
            raise ValueError("planned continuation requires a verified checkpoint")
        _validate_sha256(last_progress_sha256, name="last_progress_sha256")
        scheduler = _json_mapping(scheduler_identity, name="scheduler_identity")
        _validate_sha256(execution_slice_sha256, name="execution_slice_sha256")
        consumed = _nonnegative_integer(
            update_blocks_consumed,
            name="update_blocks_consumed",
        )
        remaining = _nonnegative_integer(
            update_blocks_remaining,
            name="update_blocks_remaining",
        )
        if planned_action not in {"continue_same_logical_run", "fail_closed"}:
            raise ValueError("planned_action is invalid")
        if planned_action == "continue_same_logical_run" and remaining != 0:
            raise ValueError(
                "planned continuation may be published only after the slice "
                "update-block budget is exhausted"
            )

        existing: list[tuple[int, Path]] = []
        for path in self.planned_boundary_directory.iterdir():
            match = _PROGRESS_EVENT_PATTERN.fullmatch(path.name)
            if match is None:
                continue
            _regular_file(path, name="planned-boundary receipt")
            existing.append((int(match.group(1)), path))
        existing.sort()
        previous_sha256: str | None = None
        required_receipt = {
            "schema_version",
            "objective",
            "binding",
            "binding_sha256",
            "sequence",
            "previous_planned_boundary_receipt_sha256",
            "head_name",
            "receipt_recorded_at_utc",
            "completed_steps",
            "checkpoint_metadata_sha256",
            "checkpoint_verified",
            "continuation_checkpoint_usable",
            "last_progress_sha256",
            "scheduler_identity",
            "execution_slice_sha256",
            "update_blocks_consumed",
            "update_blocks_remaining",
            "planned_action",
            "terminal_success_claimed",
        }
        for expected_sequence, (sequence, path) in enumerate(existing, start=1):
            if sequence != expected_sequence:
                raise RuntimeError("planned-boundary receipt sequence is not contiguous")
            previous = _read_json(path, name="planned-boundary receipt")
            if (
                set(previous) != required_receipt
                or previous.get("schema_version") != PLANNED_BOUNDARY_RECEIPT_SCHEMA
                or previous.get("objective") != self.objective
                or previous.get("binding") != self.binding
                or previous.get("binding_sha256") != self.binding_sha256
                or previous.get("sequence") != sequence
                or previous.get("previous_planned_boundary_receipt_sha256") != previous_sha256
            ):
                raise ValueError(
                    "planned-boundary receipt identity or predecessor chain is invalid"
                )
            previous_sha256 = _sha256_file(path)

        sequence = len(existing) + 1
        receipt: dict[str, object] = {
            "schema_version": PLANNED_BOUNDARY_RECEIPT_SCHEMA,
            "objective": self.objective,
            "binding": self.binding,
            "binding_sha256": self.binding_sha256,
            "sequence": sequence,
            "previous_planned_boundary_receipt_sha256": previous_sha256,
            "head_name": head_name,
            "receipt_recorded_at_utc": datetime.now(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z"),
            "completed_steps": steps,
            "checkpoint_metadata_sha256": checkpoint_metadata_sha256,
            "checkpoint_verified": True,
            "continuation_checkpoint_usable": True,
            "last_progress_sha256": last_progress_sha256,
            "scheduler_identity": scheduler,
            "execution_slice_sha256": execution_slice_sha256,
            "update_blocks_consumed": consumed,
            "update_blocks_remaining": remaining,
            "planned_action": planned_action,
            "terminal_success_claimed": False,
        }
        path = self.planned_boundary_directory / f"event-{sequence:08d}.json"
        if path.exists() or path.is_symlink():
            raise FileExistsError("refusing to overwrite an existing planned-boundary receipt")
        _atomic_json(path, receipt)
        return receipt


class CheckpointSignal:
    """Signal handler that only latches a request for the training loop."""

    def __init__(self) -> None:
        self.requested = False
        self.signal_name: str | None = None
        self.received_at_utc: str | None = None
        self.received_in_flight_update: int | None = None
        self.additional_signal_count = 0
        self._active_update: int | None = None
        self._previous: dict[int, Any] = {}

    def _handle(self, signum: int, _frame: FrameType | None) -> None:
        if self.requested:
            self.additional_signal_count += 1
            return
        self.requested = True
        self.received_in_flight_update = self._active_update
        try:
            self.signal_name = signal.Signals(signum).name
        except ValueError:
            self.signal_name = str(signum)
        self.received_at_utc = (
            datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        )

    @property
    def handlers_installed(self) -> bool:
        return bool(self._previous)

    @property
    def active_update(self) -> int | None:
        return self._active_update

    def begin_update(self, update: int) -> bool:
        """Mark an optimizer update active, or refuse a pre-existing signal.

        Python signal handlers can run between any two bytecodes.  The second
        requested check closes the small window between the initial poll and
        publication of the active-update marker: a signal observed before that
        marker prevents the update, while one observed after it is durably
        attributed to that exact update and the update may finish.
        """

        update = _nonnegative_integer(update, name="update")
        if update < 1:
            raise ValueError("update must be positive")
        if self._active_update is not None:
            raise RuntimeError("an update lifecycle is already active")
        if self.requested:
            return False
        self._active_update = update
        if self.requested and self.received_in_flight_update is None:
            self._active_update = None
            return False
        return True

    def end_update(self, update: int) -> None:
        update = _nonnegative_integer(update, name="update")
        if self._active_update != update:
            raise RuntimeError("update lifecycle end does not match the active update")
        self._active_update = None

    def install(self) -> None:
        if self._previous:
            raise RuntimeError("checkpoint signal handlers are already installed")
        supported = [
            value
            for value in (
                getattr(signal, "SIGUSR1", None),
                getattr(signal, "SIGTERM", None),
                getattr(signal, "SIGINT", None),
            )
            if isinstance(value, signal.Signals)
        ]
        for value in supported:
            self._previous[int(value)] = signal.getsignal(value)
            signal.signal(value, self._handle)

    def restore(self) -> None:
        for signum, previous in self._previous.items():
            signal.signal(signum, previous)
        self._previous.clear()

    def __enter__(self) -> CheckpointSignal:
        self.install()
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.restore()


class CheckpointInterruption(RuntimeError):
    """Raised after a scheduler-requested checkpoint reaches durable storage."""


class PlannedSegmentBoundary(CheckpointInterruption):
    """Raised after exhausting a frozen segment slice at a durable boundary."""


__all__ = [
    "CHECKPOINT_MANIFEST_SCHEMA",
    "CHECKPOINT_SCHEMA",
    "PROGRESS_SCHEMA",
    "PLANNED_BOUNDARY_RECEIPT_SCHEMA",
    "SIGNAL_RECEIPT_SCHEMA",
    "TRAINING_PROGRESS_DETAILS_SCHEMA",
    "CheckpointInterruption",
    "CheckpointSignal",
    "DurableCheckpointStore",
    "PlannedSegmentBoundary",
]
