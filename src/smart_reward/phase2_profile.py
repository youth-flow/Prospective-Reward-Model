"""Claim-free Gate-P workload profiling core for the two primary learners.

The profile is deliberately not a training result.  It runs the frozen
BT-MLE then ProRM+ workloads for a predeclared update cap, records only
operational measurements, and destroys every transient checkpoint probe.
No trained parameter, optimizer state, random-generator state, target, or
evaluation outcome is serialised into the returned payload.

The payload is intentionally not an R3 profile identity or authorization.
A formal Gate-P wrapper must bind verified materialization provenance and a
validated R3 profile-run design before executing this core.
"""

from __future__ import annotations

import math
import os
import tempfile
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TypeAlias

import torch

from . import phase2_training as _training
from .contracts import BT_MLE, PRORM_PLUS
from .phase2_primary import (
    NeutralPhase2TrainingContext,
    PrimaryLearner,
    PrimaryTrainer,
    build_primary_core_trainer,
)
from .training import BTMLETrainer, ProRMPlusTrainer, TrainingStepDiagnostics

PHASE2_PROFILE_BINDING_SCHEMA = "phase2-r4-gate-p-profile-core-binding/v1"
PHASE2_PROFILE_RESULT_SCHEMA = "phase2-r4-gate-p-profile-core-result/v1"
PHASE2_PROFILE_CAMPAIGN_KIND = "phase2_r4_gate_p_profile_core_unclaimed"
PHASE2_PROFILE_EXECUTION_REVISION = 0
PHASE2_PROFILE_ROLE = "unclaimed_profile_core_nonreusable"
PHASE2_PROFILE_SEED = 20260801
PHASE2_PROFILE_UPDATES = 100
PHASE2_PROFILE_AUDIT_UPDATES = (0, 20, 40, 60, 80, 100)
PHASE2_PROFILE_LEARNER_ORDER = (BT_MLE, PRORM_PLUS)
PHASE2_PROFILE_STOP_REASON = "predeclared_profile_update_cap"
LiveBoundaryProbe = Callable[
    [PrimaryLearner, int, Mapping[str, object], Path],
    None,
]

_PCG_REASONS = frozenset({"converged", "zero_rhs", "max_iterations"})
_FORBIDDEN_OUTPUT_TOKENS = (
    "head",
    "optimizer",
    "rng",
    "raw_reward",
    "raw_oracle",
    "label",
    "beta",
    "heldout",
    "outcome",
)
_TOP_LEVEL_FIELDS = {
    "schema_version",
    "campaign_kind",
    "execution_revision",
    "role",
    "profile_nonreusable",
    "seed",
    "context_sha256",
    "settings_sha256",
    "input_training_sha256",
    "binding_sha256",
    "learner_order",
    "update_cap_per_learner",
    "audit_update_indices",
    "stop_reason",
    "device_type",
    "formal_cuda_profile",
    "setup",
    "learners",
    "information_boundary",
    "profile_sha256",
}
_LEARNER_FIELDS = {
    "learner",
    "updates_executed",
    "stop_reason",
    "build_wall_seconds",
    "phase_wall_seconds",
    "gradient_selection_applied",
    "steps",
    "audits",
    "ephemeral_checkpoint_io",
}
_STEP_FIELDS = {"update", "wall_seconds", "cuda_memory"}
_PCG_STEP_FIELDS = _STEP_FIELDS | {"pcg"}
_PCG_FIELDS = {
    "iterations",
    "residual_norm",
    "relative_residual",
    "converged",
    "reason",
}
_AUDIT_FIELDS = {
    "update",
    "wall_seconds",
    "trainer_state_unchanged",
}
_IO_FIELDS = {
    "update",
    "serialized_bytes",
    "serialize_wall_seconds",
    "fsync_wall_seconds",
    "reload_wall_seconds",
    "roundtrip_verified",
    "artifact_retained",
    "reusable",
    "filesystem_scope",
}
_CUDA_MEMORY_FIELDS = {"measurement", "current_bytes", "peak_bytes"}
_SETUP_FIELDS = {"wall_seconds", "cuda_memory"}
_BOUNDARY_FIELDS = {
    "train_only",
    "validation_or_test_data_accessed",
    "policy_session_opened",
    "policy_rollout_performed",
    "controls_executed",
    "serialized_training_state_retained",
    "profile_consumable_as_primary_evidence",
}

ProfilePayload: TypeAlias = dict[str, object]


class PCGReasonUnavailableError(RuntimeError):
    """Raised before profiling when the solver's original reason is unavailable."""


def _nonnegative_seconds(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _elapsed_seconds(start_ns: int, end_ns: int, *, name: str) -> float:
    if isinstance(start_ns, bool) or not isinstance(start_ns, int):
        raise TypeError(f"{name} start timestamp must be an integer")
    if isinstance(end_ns, bool) or not isinstance(end_ns, int) or end_ns < start_ns:
        raise RuntimeError(f"{name} monotonic timestamp moved backwards")
    return (end_ns - start_ns) / 1_000_000_000.0


def _require_exact_fields(
    value: object,
    expected: set[str],
    *,
    name: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if set(value) != expected:
        raise ValueError(
            f"{name} fields are not the strict schema: "
            f"missing={sorted(expected.difference(value))}, "
            f"extra={sorted(set(value).difference(expected))}"
        )
    return value


def _assert_no_forbidden_output(value: object, *, path: str = "profile") -> None:
    if isinstance(value, torch.Tensor):
        raise TypeError(f"{path} must not contain tensors")
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings")
            lowered = key.casefold()
            if any(token in lowered for token in _FORBIDDEN_OUTPUT_TOKENS):
                raise ValueError(f"{path}.{key} contains a forbidden profile-output token")
            _assert_no_forbidden_output(item, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_no_forbidden_output(item, path=f"{path}[{index}]")
        return
    if isinstance(value, str):
        lowered = value.casefold()
        if any(token in lowered for token in _FORBIDDEN_OUTPUT_TOKENS):
            raise ValueError(f"{path} contains a forbidden profile-output token")
        return
    if value is not None and not isinstance(value, (bool, int, float)):
        raise TypeError(f"{path} contains a non-JSON value")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} contains a non-finite float")


def _preflight_pcg_reason_contract() -> None:
    descriptor = getattr(ProRMPlusTrainer, "last_pcg_reason", None)
    if not isinstance(descriptor, property) or descriptor.fset is not None:
        raise PCGReasonUnavailableError(
            "Gate-P requires the read-only ProRMPlusTrainer.last_pcg_reason "
            "property carrying the original PCGResult.reason; inference is forbidden"
        )


def profile_core_binding(
    context: NeutralPhase2TrainingContext,
) -> dict[str, object]:
    """Return a non-reusable, claim-free profile-core binding."""

    if not isinstance(context, NeutralPhase2TrainingContext):
        raise TypeError("context must be NeutralPhase2TrainingContext")
    context.validate_integrity()
    if context.seed != PHASE2_PROFILE_SEED:
        raise ValueError(f"Gate-P seed must equal {PHASE2_PROFILE_SEED}")
    return {
        "schema_version": PHASE2_PROFILE_BINDING_SCHEMA,
        "campaign_kind": PHASE2_PROFILE_CAMPAIGN_KIND,
        "execution_revision": PHASE2_PROFILE_EXECUTION_REVISION,
        "role": PHASE2_PROFILE_ROLE,
        "profile_nonreusable": True,
        "seed": PHASE2_PROFILE_SEED,
        "context_sha256": context.context_sha256,
        "settings_sha256": context.settings.sha256,
        "input_training_sha256": context.input_training_sha256,
        "learner_order": list(PHASE2_PROFILE_LEARNER_ORDER),
        "update_cap_per_learner": PHASE2_PROFILE_UPDATES,
        "audit_update_indices": list(PHASE2_PROFILE_AUDIT_UPDATES),
        "stop_reason": PHASE2_PROFILE_STOP_REASON,
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _cuda_memory(device: torch.device) -> dict[str, object]:
    if device.type != "cuda":
        return {
            "measurement": "nonformal_cpu",
            "current_bytes": None,
            "peak_bytes": None,
        }
    return {
        "measurement": "cuda_allocator",
        "current_bytes": int(torch.cuda.memory_allocated(device)),
        "peak_bytes": int(torch.cuda.max_memory_allocated(device)),
    }


def _timed_audit(
    trainer: PrimaryTrainer,
    learner: PrimaryLearner,
    *,
    device: torch.device,
) -> dict[str, object]:
    before_sha = _training._checkpoint_value_sha256(trainer.state_dict())
    _synchronize(device)
    start_ns = time.perf_counter_ns()
    if learner == BT_MLE:
        if not isinstance(trainer, BTMLETrainer):
            raise TypeError("BT profile trainer has the wrong concrete type")
        _training._bt_first_order_measurement(trainer)
    else:
        if not isinstance(trainer, ProRMPlusTrainer):
            raise TypeError("ProRM+ profile trainer has the wrong concrete type")
        _training._prorm_first_order_measurement(trainer)
    _synchronize(device)
    end_ns = time.perf_counter_ns()
    after_sha = _training._checkpoint_value_sha256(trainer.state_dict())
    if before_sha != after_sha:
        raise RuntimeError("Gate-P first-order audit mutated trainer state")
    return {
        "wall_seconds": _elapsed_seconds(start_ns, end_ns, name="audit"),
        "trainer_state_unchanged": True,
    }


def _ephemeral_checkpoint_probe(
    trainer: PrimaryTrainer,
    *,
    update: int,
    directory: Path | None,
) -> dict[str, object]:
    source_state = trainer.state_dict()
    source_sha = _training._checkpoint_value_sha256(source_state)
    file_directory = None if directory is None else str(directory)
    with tempfile.TemporaryFile(mode="w+b", dir=file_directory) as handle:
        start_ns = time.perf_counter_ns()
        torch.save(source_state, handle)
        serialized_ns = time.perf_counter_ns()
        serialized_bytes = handle.tell()
        if serialized_bytes <= 0:
            raise RuntimeError("ephemeral checkpoint probe serialized no bytes")

        fsync_start_ns = time.perf_counter_ns()
        handle.flush()
        os.fsync(handle.fileno())
        fsync_end_ns = time.perf_counter_ns()

        handle.seek(0)
        reload_start_ns = time.perf_counter_ns()
        restored_state = torch.load(handle, weights_only=True)
        reload_end_ns = time.perf_counter_ns()
        restored_sha = _training._checkpoint_value_sha256(restored_state)
        if restored_sha != source_sha:
            raise RuntimeError("ephemeral checkpoint probe did not round-trip exactly")
        del restored_state
    del source_state
    return {
        "update": update,
        "serialized_bytes": serialized_bytes,
        "serialize_wall_seconds": _elapsed_seconds(
            start_ns,
            serialized_ns,
            name="checkpoint serialization",
        ),
        "fsync_wall_seconds": _elapsed_seconds(
            fsync_start_ns,
            fsync_end_ns,
            name="checkpoint fsync",
        ),
        "reload_wall_seconds": _elapsed_seconds(
            reload_start_ns,
            reload_end_ns,
            name="checkpoint reload",
        ),
        "roundtrip_verified": True,
        "artifact_retained": False,
        "reusable": False,
        "filesystem_scope": (
            "system_temporary_directory" if directory is None else "declared_profile_directory"
        ),
    }


def _pcg_step_payload(
    diagnostic: TrainingStepDiagnostics,
    *,
    trainer: ProRMPlusTrainer,
) -> dict[str, object]:
    # This read must immediately follow the update.  The value is the solver's
    # original categorical result, not a classification derived from residuals.
    reason = trainer.last_pcg_reason
    if not isinstance(reason, str) or reason not in _PCG_REASONS:
        raise PCGReasonUnavailableError(
            "ProRM+ update did not expose its original PCGResult.reason"
        )
    if (
        diagnostic.pcg_iterations is None
        or diagnostic.pcg_residual_norm is None
        or diagnostic.pcg_relative_residual is None
        or diagnostic.pcg_converged is None
    ):
        raise RuntimeError("ProRM+ update omitted required PCG diagnostics")
    iterations = _nonnegative_integer(
        diagnostic.pcg_iterations,
        name="pcg_iterations",
    )
    residual_norm = _nonnegative_seconds(
        diagnostic.pcg_residual_norm,
        name="pcg_residual_norm",
    )
    relative_residual = _nonnegative_seconds(
        diagnostic.pcg_relative_residual,
        name="pcg_relative_residual",
    )
    converged = diagnostic.pcg_converged
    if not isinstance(converged, bool):
        raise TypeError("pcg_converged must be bool")
    if reason == "zero_rhs" and (
        not converged or iterations != 0 or residual_norm != 0.0 or relative_residual != 0.0
    ):
        raise RuntimeError("raw zero_rhs reason conflicts with PCG diagnostics")
    if reason == "converged" and not converged:
        raise RuntimeError("raw converged reason conflicts with PCG diagnostics")
    if reason == "max_iterations" and converged:
        raise RuntimeError("raw max_iterations reason conflicts with PCG diagnostics")
    return {
        "iterations": iterations,
        "residual_norm": residual_norm,
        "relative_residual": relative_residual,
        "converged": converged,
        "reason": reason,
    }


def _run_live_boundary_probe(
    trainer: PrimaryTrainer,
    learner: PrimaryLearner,
    *,
    update: int,
    directory: Path | None,
    probe: LiveBoundaryProbe | None,
) -> None:
    if probe is None:
        return
    if directory is None:
        raise ValueError("live boundary probe requires a declared profile directory")
    before_sha = _training._checkpoint_value_sha256(trainer.state_dict())
    copied_state = trainer.state_dict()
    probe(learner, update, copied_state, directory)
    del copied_state
    after_sha = _training._checkpoint_value_sha256(trainer.state_dict())
    if after_sha != before_sha:
        raise RuntimeError("live boundary probe mutated the profile trainer")


def _profile_learner(
    context: NeutralPhase2TrainingContext,
    learner: PrimaryLearner,
    *,
    device: torch.device,
    io_probe_directory: Path | None,
    live_boundary_probe: LiveBoundaryProbe | None,
) -> dict[str, object]:
    phase_start_ns = time.perf_counter_ns()
    _synchronize(device)
    build_start_ns = time.perf_counter_ns()
    trainer = build_primary_core_trainer(context, learner)
    _synchronize(device)
    build_end_ns = time.perf_counter_ns()
    if trainer.completed_steps != 0:
        raise RuntimeError("Gate-P trainer was not fresh")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    audits: list[dict[str, object]] = []
    probes: list[dict[str, object]] = []
    initial_audit = _timed_audit(trainer, learner, device=device)
    audits.append({"update": 0, **initial_audit})
    probes.append(
        _ephemeral_checkpoint_probe(
            trainer,
            update=0,
            directory=io_probe_directory,
        )
    )
    _run_live_boundary_probe(
        trainer,
        learner,
        update=0,
        directory=io_probe_directory,
        probe=live_boundary_probe,
    )

    steps: list[dict[str, object]] = []
    for update in range(1, PHASE2_PROFILE_UPDATES + 1):
        _synchronize(device)
        step_start_ns = time.perf_counter_ns()
        diagnostic = trainer.step()
        pcg_payload: dict[str, object] | None = None
        if learner == PRORM_PLUS:
            if not isinstance(trainer, ProRMPlusTrainer):
                raise TypeError("ProRM+ profile trainer has the wrong concrete type")
            pcg_payload = _pcg_step_payload(diagnostic, trainer=trainer)
        _synchronize(device)
        step_end_ns = time.perf_counter_ns()
        if diagnostic.step != update or trainer.completed_steps != update:
            raise RuntimeError("Gate-P trainer update count is not exact")
        step_payload: dict[str, object] = {
            "update": update,
            "wall_seconds": _elapsed_seconds(
                step_start_ns,
                step_end_ns,
                name="training update",
            ),
            "cuda_memory": _cuda_memory(device),
        }
        if pcg_payload is not None:
            step_payload["pcg"] = pcg_payload
        steps.append(step_payload)
        if update in PHASE2_PROFILE_AUDIT_UPDATES[1:]:
            audit_payload = _timed_audit(trainer, learner, device=device)
            audits.append({"update": update, **audit_payload})
            probes.append(
                _ephemeral_checkpoint_probe(
                    trainer,
                    update=update,
                    directory=io_probe_directory,
                )
            )
            _run_live_boundary_probe(
                trainer,
                learner,
                update=update,
                directory=io_probe_directory,
                probe=live_boundary_probe,
            )

    _synchronize(device)
    phase_end_ns = time.perf_counter_ns()
    if trainer.completed_steps != PHASE2_PROFILE_UPDATES:
        raise RuntimeError("Gate-P did not execute the predeclared update cap")
    result = {
        "learner": learner,
        "updates_executed": PHASE2_PROFILE_UPDATES,
        "stop_reason": PHASE2_PROFILE_STOP_REASON,
        "build_wall_seconds": _elapsed_seconds(
            build_start_ns,
            build_end_ns,
            name="trainer build",
        ),
        "phase_wall_seconds": _elapsed_seconds(
            phase_start_ns,
            phase_end_ns,
            name="learner profile",
        ),
        "gradient_selection_applied": False,
        "steps": steps,
        "audits": audits,
        "ephemeral_checkpoint_io": probes,
    }
    del trainer
    return result


def run_gate_p_profile_core(
    context: NeutralPhase2TrainingContext,
    *,
    io_probe_directory: str | Path | None = None,
    live_boundary_probe: LiveBoundaryProbe | None = None,
) -> ProfilePayload:
    """Run the isolated fixed-work Gate-P profile.

    This call never resumes and never returns a trained state.  If a directory
    is supplied, anonymous temporary checkpoint probes are created on that
    filesystem so the I/O timings reflect the declared profile storage.
    """

    if live_boundary_probe is not None and not callable(live_boundary_probe):
        raise TypeError("live_boundary_probe must be callable")

    _preflight_pcg_reason_contract()
    setup_start_ns = time.perf_counter_ns()
    binding = profile_core_binding(context)
    directory: Path | None
    if io_probe_directory is None:
        directory = None
    else:
        directory = Path(io_probe_directory).resolve()
        if not directory.is_dir():
            raise ValueError("io_probe_directory must be an existing directory")
    device = context.training.reward_features.device
    formal_cuda = device.type == "cuda"
    if formal_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA tensors are present but CUDA is unavailable")
    setup_end_ns = time.perf_counter_ns()

    learners = [
        _profile_learner(
            context,
            learner,
            device=device,
            io_probe_directory=directory,
            live_boundary_probe=live_boundary_probe,
        )
        for learner in PHASE2_PROFILE_LEARNER_ORDER
    ]
    payload: ProfilePayload = {
        "schema_version": PHASE2_PROFILE_RESULT_SCHEMA,
        "campaign_kind": PHASE2_PROFILE_CAMPAIGN_KIND,
        "execution_revision": PHASE2_PROFILE_EXECUTION_REVISION,
        "role": PHASE2_PROFILE_ROLE,
        "profile_nonreusable": True,
        "seed": PHASE2_PROFILE_SEED,
        "context_sha256": context.context_sha256,
        "settings_sha256": context.settings.sha256,
        "input_training_sha256": context.input_training_sha256,
        "binding_sha256": _training._canonical_sha256(binding),
        "learner_order": list(PHASE2_PROFILE_LEARNER_ORDER),
        "update_cap_per_learner": PHASE2_PROFILE_UPDATES,
        "audit_update_indices": list(PHASE2_PROFILE_AUDIT_UPDATES),
        "stop_reason": PHASE2_PROFILE_STOP_REASON,
        "device_type": device.type,
        "formal_cuda_profile": formal_cuda,
        "setup": {
            "wall_seconds": _elapsed_seconds(
                setup_start_ns,
                setup_end_ns,
                name="profile setup",
            ),
            "cuda_memory": _cuda_memory(device),
        },
        "learners": learners,
        "information_boundary": {
            "train_only": True,
            "validation_or_test_data_accessed": False,
            "policy_session_opened": False,
            "policy_rollout_performed": False,
            "controls_executed": False,
            "serialized_training_state_retained": False,
            "profile_consumable_as_primary_evidence": False,
        },
    }
    _assert_no_forbidden_output(payload)
    payload["profile_sha256"] = _training._canonical_sha256(payload)
    validate_gate_p_profile_core_result(payload)
    return payload


def validate_gate_p_profile_core_result(value: object) -> None:
    """Validate an exact, non-sensitive but unclaimed profile-core payload."""

    profile = _require_exact_fields(value, _TOP_LEVEL_FIELDS, name="profile")
    _assert_no_forbidden_output(profile)
    expected_scalars = {
        "schema_version": PHASE2_PROFILE_RESULT_SCHEMA,
        "campaign_kind": PHASE2_PROFILE_CAMPAIGN_KIND,
        "execution_revision": PHASE2_PROFILE_EXECUTION_REVISION,
        "role": PHASE2_PROFILE_ROLE,
        "profile_nonreusable": True,
        "seed": PHASE2_PROFILE_SEED,
        "learner_order": list(PHASE2_PROFILE_LEARNER_ORDER),
        "update_cap_per_learner": PHASE2_PROFILE_UPDATES,
        "audit_update_indices": list(PHASE2_PROFILE_AUDIT_UPDATES),
        "stop_reason": PHASE2_PROFILE_STOP_REASON,
    }
    for field, expected in expected_scalars.items():
        if profile[field] != expected:
            raise ValueError(f"profile {field} is not frozen Gate-P evidence")
    for field in (
        "context_sha256",
        "settings_sha256",
        "input_training_sha256",
        "binding_sha256",
        "profile_sha256",
    ):
        _training._validate_digest(profile[field], name=field)
    device_type = profile["device_type"]
    if device_type not in {"cpu", "cuda"}:
        raise ValueError("profile device_type must be cpu or cuda")
    formal_cuda = profile["formal_cuda_profile"]
    if not isinstance(formal_cuda, bool) or formal_cuda != (device_type == "cuda"):
        raise ValueError("formal_cuda_profile does not match device_type")

    setup = _require_exact_fields(profile["setup"], _SETUP_FIELDS, name="profile.setup")
    _nonnegative_seconds(setup["wall_seconds"], name="profile.setup.wall_seconds")
    _validate_cuda_memory(
        setup["cuda_memory"],
        formal_cuda=formal_cuda,
        name="profile.setup.cuda_memory",
    )
    learners = profile["learners"]
    if not isinstance(learners, list) or len(learners) != 2:
        raise ValueError("profile must contain exactly two learner records")
    for expected_learner, raw_learner in zip(
        PHASE2_PROFILE_LEARNER_ORDER,
        learners,
        strict=True,
    ):
        _validate_learner_profile(
            raw_learner,
            expected_learner=expected_learner,
            formal_cuda=formal_cuda,
        )

    boundary = _require_exact_fields(
        profile["information_boundary"],
        _BOUNDARY_FIELDS,
        name="profile.information_boundary",
    )
    expected_boundary = {
        "train_only": True,
        "validation_or_test_data_accessed": False,
        "policy_session_opened": False,
        "policy_rollout_performed": False,
        "controls_executed": False,
        "serialized_training_state_retained": False,
        "profile_consumable_as_primary_evidence": False,
    }
    if dict(boundary) != expected_boundary:
        raise ValueError("profile information boundary is invalid")
    unsigned = dict(profile)
    observed_sha = unsigned.pop("profile_sha256")
    if _training._canonical_sha256(unsigned) != observed_sha:
        raise ValueError("profile_sha256 does not match the strict payload")


def _validate_cuda_memory(
    value: object,
    *,
    formal_cuda: bool,
    name: str,
) -> None:
    memory = _require_exact_fields(value, _CUDA_MEMORY_FIELDS, name=name)
    expected_measurement = "cuda_allocator" if formal_cuda else "nonformal_cpu"
    if memory["measurement"] != expected_measurement:
        raise ValueError(f"{name}.measurement is invalid")
    if formal_cuda:
        current = _nonnegative_integer(memory["current_bytes"], name=f"{name}.current_bytes")
        peak = _nonnegative_integer(memory["peak_bytes"], name=f"{name}.peak_bytes")
        if peak < current:
            raise ValueError(f"{name}.peak_bytes must be at least current_bytes")
    elif memory["current_bytes"] is not None or memory["peak_bytes"] is not None:
        raise ValueError(f"{name} CPU measurements must be explicitly unavailable")


def _validate_learner_profile(
    value: object,
    *,
    expected_learner: str,
    formal_cuda: bool,
) -> None:
    learner = _require_exact_fields(
        value,
        _LEARNER_FIELDS,
        name=f"profile.learners[{expected_learner}]",
    )
    if (
        learner["learner"] != expected_learner
        or learner["updates_executed"] != PHASE2_PROFILE_UPDATES
        or learner["stop_reason"] != PHASE2_PROFILE_STOP_REASON
        or learner["gradient_selection_applied"] is not False
    ):
        raise ValueError(f"{expected_learner} fixed-work contract is invalid")
    _nonnegative_seconds(
        learner["build_wall_seconds"],
        name=f"{expected_learner}.build_wall_seconds",
    )
    _nonnegative_seconds(
        learner["phase_wall_seconds"],
        name=f"{expected_learner}.phase_wall_seconds",
    )
    steps = learner["steps"]
    if not isinstance(steps, list) or len(steps) != PHASE2_PROFILE_UPDATES:
        raise ValueError(f"{expected_learner} must contain exactly 100 step records")
    expected_step_fields = _STEP_FIELDS if expected_learner == BT_MLE else _PCG_STEP_FIELDS
    for expected_update, raw_step in enumerate(steps, start=1):
        step = _require_exact_fields(
            raw_step,
            expected_step_fields,
            name=f"{expected_learner}.steps[{expected_update}]",
        )
        if step["update"] != expected_update:
            raise ValueError(f"{expected_learner} step sequence is not exact")
        _nonnegative_seconds(
            step["wall_seconds"],
            name=f"{expected_learner}.steps[{expected_update}].wall_seconds",
        )
        _validate_cuda_memory(
            step["cuda_memory"],
            formal_cuda=formal_cuda,
            name=f"{expected_learner}.steps[{expected_update}].cuda_memory",
        )
        if expected_learner == PRORM_PLUS:
            _validate_pcg_payload(
                step["pcg"],
                name=f"{expected_learner}.steps[{expected_update}].pcg",
            )

    audits = learner["audits"]
    if not isinstance(audits, list) or len(audits) != len(PHASE2_PROFILE_AUDIT_UPDATES):
        raise ValueError(f"{expected_learner} audit count is invalid")
    for expected_update, raw_audit in zip(
        PHASE2_PROFILE_AUDIT_UPDATES,
        audits,
        strict=True,
    ):
        audit = _require_exact_fields(
            raw_audit,
            _AUDIT_FIELDS,
            name=f"{expected_learner}.audits[{expected_update}]",
        )
        if audit["update"] != expected_update or audit["trainer_state_unchanged"] is not True:
            raise ValueError(f"{expected_learner} audit boundary is invalid")
        _nonnegative_seconds(
            audit["wall_seconds"],
            name=f"{expected_learner}.audits[{expected_update}].wall_seconds",
        )

    probes = learner["ephemeral_checkpoint_io"]
    if not isinstance(probes, list) or len(probes) != len(PHASE2_PROFILE_AUDIT_UPDATES):
        raise ValueError(f"{expected_learner} checkpoint-probe count is invalid")
    for expected_update, raw_probe in zip(
        PHASE2_PROFILE_AUDIT_UPDATES,
        probes,
        strict=True,
    ):
        probe = _require_exact_fields(
            raw_probe,
            _IO_FIELDS,
            name=f"{expected_learner}.ephemeral_checkpoint_io[{expected_update}]",
        )
        if (
            probe["update"] != expected_update
            or isinstance(probe["serialized_bytes"], bool)
            or not isinstance(probe["serialized_bytes"], int)
            or probe["serialized_bytes"] <= 0
            or probe["roundtrip_verified"] is not True
            or probe["artifact_retained"] is not False
            or probe["reusable"] is not False
            or probe["filesystem_scope"]
            not in {"system_temporary_directory", "declared_profile_directory"}
        ):
            raise ValueError(f"{expected_learner} checkpoint probe is reusable or malformed")
        for field in (
            "serialize_wall_seconds",
            "fsync_wall_seconds",
            "reload_wall_seconds",
        ):
            _nonnegative_seconds(
                probe[field],
                name=f"{expected_learner}.ephemeral_checkpoint_io[{expected_update}].{field}",
            )


def _validate_pcg_payload(value: object, *, name: str) -> None:
    pcg = _require_exact_fields(value, _PCG_FIELDS, name=name)
    iterations = _nonnegative_integer(pcg["iterations"], name=f"{name}.iterations")
    residual = _nonnegative_seconds(pcg["residual_norm"], name=f"{name}.residual_norm")
    relative = _nonnegative_seconds(
        pcg["relative_residual"],
        name=f"{name}.relative_residual",
    )
    converged = pcg["converged"]
    reason = pcg["reason"]
    if not isinstance(converged, bool) or reason not in _PCG_REASONS:
        raise ValueError(f"{name} convergence evidence is invalid")
    if reason == "zero_rhs" and (
        not converged or iterations != 0 or residual != 0.0 or relative != 0.0
    ):
        raise ValueError(f"{name} zero_rhs evidence is inconsistent")
    if reason == "converged" and not converged:
        raise ValueError(f"{name} converged reason is inconsistent")
    if reason == "max_iterations" and converged:
        raise ValueError(f"{name} max_iterations reason is inconsistent")


__all__ = [
    "LiveBoundaryProbe",
    "PCGReasonUnavailableError",
    "PHASE2_PROFILE_AUDIT_UPDATES",
    "PHASE2_PROFILE_BINDING_SCHEMA",
    "PHASE2_PROFILE_CAMPAIGN_KIND",
    "PHASE2_PROFILE_EXECUTION_REVISION",
    "PHASE2_PROFILE_LEARNER_ORDER",
    "PHASE2_PROFILE_RESULT_SCHEMA",
    "PHASE2_PROFILE_ROLE",
    "PHASE2_PROFILE_SEED",
    "PHASE2_PROFILE_STOP_REASON",
    "PHASE2_PROFILE_UPDATES",
    "profile_core_binding",
    "run_gate_p_profile_core",
    "validate_gate_p_profile_core_result",
]
