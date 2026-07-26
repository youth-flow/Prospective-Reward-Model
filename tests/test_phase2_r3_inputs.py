from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest
import torch

from smart_reward import phase2_r3_inputs as inputs
from smart_reward.config import config_hash, load_config
from smart_reward.data import CandidateNode
from smart_reward.phase2_r3_config import load_r3_science_config

ROOT = Path(__file__).resolve().parents[1]
SCIENCE_PATH = ROOT / "configs" / "phase2_recovery_r3_science.yaml"
SOURCE_CONFIG_PATH = ROOT / "configs" / "common_beta_pilot_base.yaml"
SEEDS = (20260801, 20260802, 20260803)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _token_sha(value: str) -> str:
    return _sha(value.encode("utf-8"))


def _canonical_raw(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_raw(value))


def _training_tensors() -> dict[str, torch.Tensor]:
    node = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    train_policy = torch.stack(
        [torch.sin(0.11 * node), torch.cos(0.17 * node), 0.03 * node],
        dim=-1,
    )
    train_reward = torch.stack(
        [torch.cos(0.13 * node), torch.sin(0.19 * node)],
        dim=-1,
    )
    validation_node = torch.arange(4, dtype=torch.float32).reshape(1, 4)
    test_node = validation_node + 0.5
    return {
        "train.policy_scores": train_policy,
        "train.reward_features": train_reward,
        "train.h": torch.linspace(-0.4, 0.3, 4),
        "train.left_wins": torch.tensor([1, 2, 1, 3], dtype=torch.int64),
        "train.num_annotations": torch.tensor([3, 4, 2, 5], dtype=torch.int64),
        "validation.policy_scores": torch.stack(
            [
                torch.sin(0.11 * validation_node),
                torch.cos(0.17 * validation_node),
                0.03 * validation_node,
            ],
            dim=-1,
        ),
        "validation.reward_features": torch.stack(
            [
                torch.cos(0.13 * validation_node),
                torch.sin(0.19 * validation_node),
            ],
            dim=-1,
        ),
        "validation.true_rewards": torch.zeros((1, 4), dtype=torch.float32),
        "test.policy_scores": torch.stack(
            [
                torch.sin(0.11 * test_node),
                torch.cos(0.17 * test_node),
                0.03 * test_node,
            ],
            dim=-1,
        ),
        "test.reward_features": torch.stack(
            [torch.cos(0.13 * test_node), torch.sin(0.19 * test_node)],
            dim=-1,
        ),
        "test.true_rewards": torch.zeros((1, 4), dtype=torch.float32),
    }


def _candidate_lines() -> tuple[bytes, ...]:
    result: list[bytes] = []
    for prompt_index in range(4):
        prompt_id = f"train-{prompt_index}"
        for candidate_index in range(4):
            candidate = CandidateNode(
                prompt_id=prompt_id,
                candidate_id=f"{prompt_id}::candidate::{candidate_index}",
                prompt=f"prompt {prompt_index}",
                response=f"response {prompt_index}/{candidate_index}",
                token_ids=(1, 10 + prompt_index, 20 + candidate_index),
                response_mask=(0, 1, 1),
                terminated_by_eos=True,
                reached_max_length=False,
            )
            result.append(_canonical_raw(candidate.to_dict()))
    return tuple(result)


def _tensor_specs(tensors: dict[str, torch.Tensor]) -> dict[str, dict[str, object]]:
    return {
        name: {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).removeprefix("torch."),
        }
        for name, tensor in sorted(tensors.items())
    }


@dataclass
class _ParentFixture:
    root: Path
    source_config: Path
    registry: Path
    run_root: Path
    artifact_root: Path
    candidate_lines: tuple[bytes, ...]
    candidate_suffix: bytes
    registry_sha256: str


def _dummy_hash_mapping(names: set[str] | frozenset[str], token: str) -> dict[str, str]:
    return {name: _token_sha(f"{token}:{name}") for name in sorted(names)}


def _build_parent_fixture(tmp_path: Path) -> _ParentFixture:
    safetensors = pytest.importorskip("safetensors.torch")
    root = (tmp_path / "project").resolve()
    root.mkdir(parents=True)
    source_config_path = root / "configs" / "common_beta_pilot_base.yaml"
    source_config_path.parent.mkdir()
    shutil.copyfile(SOURCE_CONFIG_PATH, source_config_path)
    source_config_hash = config_hash(load_config(source_config_path))

    producer = {
        "git_commit": "a" * 40,
        "image_sha256": _token_sha("image"),
        "hf_inventory_sha256": _token_sha("inventory"),
    }
    parent_design_sha = _token_sha("parent-design")
    source_array = "1647491"
    selected_run_relative = "runs/seed-20260801"
    selected_artifact_relative = "artifacts/seed-20260801"
    run_root = root / selected_run_relative
    artifact_root = root / selected_artifact_relative
    run_root.mkdir(parents=True)
    artifact_root.mkdir(parents=True)

    tensors = _training_tensors()
    tensor_path = artifact_root / "tensors.safetensors"
    safetensors.save_file(tensors, str(tensor_path))
    tensor_sha = _sha(tensor_path.read_bytes())

    candidate_lines = _candidate_lines()
    malicious_suffix = b'\xff{"heldout":"must never be decoded"}\n'
    candidate_path = artifact_root / "candidates.jsonl"
    candidate_path.write_bytes(b"".join(candidate_lines) + malicious_suffix)
    candidate_sha = _sha(candidate_path.read_bytes())
    other_jsonl = {
        name: _token_sha(name)
        for name in (
            "prompts.jsonl",
            "training_edges.jsonl",
            "evaluation_edges.jsonl",
        )
    }
    metadata = {
        "schema": inputs.ARTIFACT_SCHEMA,
        "config_hash": source_config_hash,
        "seed": SEEDS[0],
        "splits": {
            "train": {"prompt_ids": [f"train-{index}" for index in range(4)]},
            "validation": {"prompt_ids": ["validation-0"]},
            "test": {"prompt_ids": ["test-0"]},
        },
        "tensors": _tensor_specs(tensors),
        "tensor_sha256": tensor_sha,
        "evidence": {
            "schema": inputs.MATERIALIZATION_SCHEMA,
            "config_sha256": source_config_hash,
            "seed": SEEDS[0],
            "producer": producer,
            "jsonl_sha256": {
                "candidates.jsonl": candidate_sha,
                **other_jsonl,
            },
            "oracle_chat_template_sha256": _token_sha("oracle-template"),
            "oracle_transform": {"b": 0.125, "tau": 1.75},
        },
    }
    metadata_path = artifact_root / "metadata.json"
    _write_json(metadata_path, metadata)
    metadata_sha = _sha(metadata_path.read_bytes())

    manifest = {
        "schema_version": "smart-reward-run/v1",
        "config_hash": source_config_hash,
        "selected_seed": SEEDS[0],
        "git": {"commit": producer["git_commit"], "dirty": False},
        "slurm": {
            "PRORM_GIT_COMMIT": producer["git_commit"],
            "PRORM_IMAGE_SHA256": producer["image_sha256"],
            "PRORM_HF_INVENTORY_SHA256": producer["hf_inventory_sha256"],
            "SLURM_CLUSTER_NAME": "hpc4",
            "SLURM_JOB_ACCOUNT": "sigroup",
            "SLURM_JOB_PARTITION": "gpu-l20",
            "SLURM_ARRAY_JOB_ID": source_array,
            "SLURM_ARRAY_TASK_ID": "0",
            "SLURM_GPUS_ON_NODE": "1",
            "SLURM_NNODES": "1",
            "SLURM_NTASKS": "1",
            "CUDA_VISIBLE_DEVICES": "0",
        },
    }
    _write_json(run_root / "run-manifest.json", manifest)
    binding = {
        "schema_version": inputs.ARTIFACT_BINDING_SCHEMA,
        "mode": "materialized",
        "base_config_hash": source_config_hash,
        "phase2_design_sha256": parent_design_sha,
        "seed": SEEDS[0],
        "artifact_metadata_sha256": metadata_sha,
        "producer": producer,
    }
    _write_json(run_root / "artifact-materialization.json", binding)
    verification = {
        "status": "ok",
        "seed": SEEDS[0],
        "phase2_design_sha256": parent_design_sha,
        "base_config_hash": source_config_hash,
        "formal_environment": True,
        "artifact_metadata_sha256": metadata_sha,
    }
    _write_json(run_root / "artifact-verification.json", verification)

    artifact_hashes = {
        "metadata.json": metadata_sha,
        "tensors.safetensors": tensor_sha,
        "candidates.jsonl": candidate_sha,
        **other_jsonl,
        "policy_prompt_semantics_records": _token_sha("prompt-semantics"),
        "selected_prompt_ids": _token_sha("selected-prompts"),
    }
    evidence_hashes = _dummy_hash_mapping(inputs._EVIDENCE_FILES, "unused")
    for name in (
        "run-manifest.json",
        "artifact-materialization.json",
        "artifact-verification.json",
    ):
        evidence_hashes[name] = _sha((run_root / name).read_bytes())
    campaign = {
        "source_job_array_id": source_array,
        "parent_phase2_design_sha256": parent_design_sha,
        "base_config_hash": source_config_hash,
        "producer": producer,
        "failure_class": "primary_bt_mle_first_order_convergence_gate_not_met",
        "failed_optimizer_updates": 5760,
        "first_order_tolerance": 0.001,
        "consecutive_passes_required": 3,
        "failure_aggregate": {
            "present": False,
            "reason": "no structured aggregate",
            "replacement_evidence": "registry binds immutable source evidence",
        },
        "one_shot_no_further_adaptation": True,
        "allowed_recovery_scope": "train_only_same_materialized_artifacts",
    }
    entries: list[dict[str, object]] = []
    for index, seed in enumerate(SEEDS):
        if index == 0:
            entry_evidence = evidence_hashes
            entry_artifacts = artifact_hashes
            source_run = selected_run_relative
            source_artifact = selected_artifact_relative
        else:
            entry_evidence = _dummy_hash_mapping(
                inputs._EVIDENCE_FILES,
                f"seed-{seed}",
            )
            entry_artifacts = _dummy_hash_mapping(
                inputs._ARTIFACT_FILES | inputs._ARTIFACT_DERIVED_DIGESTS,
                f"seed-{seed}",
            )
            source_run = f"runs/seed-{seed}"
            source_artifact = f"artifacts/seed-{seed}"
        entries.append(
            {
                "seed": seed,
                "array_task_id": index,
                "source_run": source_run,
                "source_artifact": source_artifact,
                "evidence_sha256": entry_evidence,
                "artifact_sha256": entry_artifacts,
            }
        )
    common_schema = {
        "num_tensor_keys": len(tensors),
        "train_policy_scores_shape": list(tensors["train.policy_scores"].shape),
        "train_reward_features_shape": list(tensors["train.reward_features"].shape),
        "validation_policy_scores_shape": list(tensors["validation.policy_scores"].shape),
        "validation_reward_features_shape": list(tensors["validation.reward_features"].shape),
        "test_policy_scores_shape": list(tensors["test.policy_scores"].shape),
        "test_reward_features_shape": list(tensors["test.reward_features"].shape),
    }
    registry = {
        "schema_version": inputs.PARENT_REGISTRY_SCHEMA,
        "campaign": campaign,
        "common_artifact_identities": {
            "eligible_prompt_ids_sha256": _token_sha("eligible-prompts"),
            "tensor_schema": common_schema,
        },
        "optimizer_diagnostic": {
            "sha256": _token_sha("diagnostic"),
            "artifact_metadata_sha256": metadata_sha,
        },
        "seeds": entries,
    }
    registry_path = root / "configs" / "phase2_recovery_parent_failures.json"
    _write_json(registry_path, registry)
    return _ParentFixture(
        root=root,
        source_config=source_config_path,
        registry=registry_path,
        run_root=run_root,
        artifact_root=artifact_root,
        candidate_lines=candidate_lines,
        candidate_suffix=malicious_suffix,
        registry_sha256=_sha(registry_path.read_bytes()),
    )


def _refresh_selected_closure(fixture: _ParentFixture) -> None:
    metadata_path = fixture.artifact_root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    tensor_sha = _sha((fixture.artifact_root / "tensors.safetensors").read_bytes())
    candidate_sha = _sha((fixture.artifact_root / "candidates.jsonl").read_bytes())
    metadata["tensor_sha256"] = tensor_sha
    metadata["evidence"]["jsonl_sha256"]["candidates.jsonl"] = candidate_sha
    _write_json(metadata_path, metadata)
    metadata_sha = _sha(metadata_path.read_bytes())

    binding_path = fixture.run_root / "artifact-materialization.json"
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    binding["artifact_metadata_sha256"] = metadata_sha
    _write_json(binding_path, binding)
    verification_path = fixture.run_root / "artifact-verification.json"
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    verification["artifact_metadata_sha256"] = metadata_sha
    _write_json(verification_path, verification)

    registry = json.loads(fixture.registry.read_text(encoding="utf-8"))
    selected = registry["seeds"][0]
    selected["artifact_sha256"]["metadata.json"] = metadata_sha
    selected["artifact_sha256"]["tensors.safetensors"] = tensor_sha
    selected["artifact_sha256"]["candidates.jsonl"] = candidate_sha
    selected["evidence_sha256"]["artifact-materialization.json"] = _sha(binding_path.read_bytes())
    selected["evidence_sha256"]["artifact-verification.json"] = _sha(verification_path.read_bytes())
    registry["optimizer_diagnostic"]["artifact_metadata_sha256"] = metadata_sha
    _write_json(fixture.registry, registry)
    fixture.registry_sha256 = _sha(fixture.registry.read_bytes())


class _TrackingSafeOpen:
    def __init__(
        self,
        safe_open: Any,
        decoded_keys: list[str],
        *args: object,
        **kwargs: object,
    ) -> None:
        self._context = safe_open(*args, **kwargs)
        self._decoded_keys = decoded_keys
        self._handle: Any = None

    def __enter__(self) -> _TrackingSafeOpen:
        self._handle = self._context.__enter__()
        return self

    def __exit__(self, *args: object) -> object:
        return self._context.__exit__(*args)

    def keys(self) -> list[str]:
        return list(self._handle.keys())

    def get_tensor(self, name: str) -> torch.Tensor:
        self._decoded_keys.append(name)
        return self._handle.get_tensor(name)


class _TrackingLineStream:
    def __init__(self, path: Path, calls: list[int]) -> None:
        self._stream = path.open("rb", buffering=0)
        self._calls = calls

    def __enter__(self) -> _TrackingLineStream:
        return self

    def __exit__(self, *args: object) -> None:
        self._stream.close()

    def fileno(self) -> int:
        return self._stream.fileno()

    def readline(self, size: int = -1) -> bytes:
        self._calls.append(size)
        return self._stream.readline(size)


def _run_cpu_materialization(
    fixture: _ParentFixture,
    *,
    oracle_values: torch.Tensor | None = None,
    control_only: bool = False,
    clock_values: tuple[int, int, int, int] = (
        0,
        1_000_000_000,
        3_000_000_000,
        6_000_000_000,
    ),
) -> tuple[
    inputs.R3TrainOnlyMaterializationResult | inputs.R3ControlTrainOnlyMaterializationResult,
    list[str],
    list[int],
    dict[str, object],
]:
    safetensors = pytest.importorskip("safetensors")
    science = load_r3_science_config(SCIENCE_PATH)
    decoded_keys: list[str] = []
    line_calls: list[int] = []
    oracle_observation: dict[str, object] = {}
    clock = iter(clock_values)

    def safe_open_factory(*args: object, **kwargs: object) -> _TrackingSafeOpen:
        return _TrackingSafeOpen(
            safetensors.safe_open,
            decoded_keys,
            *args,
            **kwargs,
        )

    def line_stream_factory(path: Path) -> _TrackingLineStream:
        return _TrackingLineStream(path, line_calls)

    def oracle_rescorer(**kwargs: object) -> torch.Tensor:
        candidates = kwargs["candidates"]
        oracle_observation.update(kwargs)
        assert isinstance(candidates, tuple)
        assert len(candidates) == 16
        if oracle_values is not None:
            return oracle_values
        return torch.tensor(
            [0.20, -0.20, 0.10, -0.10],
            dtype=torch.float32,
        ).repeat(4)

    result = inputs._materialize_r3_train_only_from_parent(
        project_root=fixture.root,
        parent_registry_path=fixture.registry,
        expected_parent_registry_file_sha256=fixture.registry_sha256,
        source_config_path=fixture.source_config,
        science_bundle=science,
        seed=SEEDS[0],
        target_device=torch.device("cpu"),
        oracle_rescorer=oracle_rescorer,
        clock_ns=lambda: next(clock),
        safe_open_factory=safe_open_factory,
        line_stream_factory=line_stream_factory,
        control_only=control_only,
    )
    return result, decoded_keys, line_calls, oracle_observation


def test_control_materialization_returns_before_any_primary_label_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_parent_fixture(tmp_path)

    def forbidden_primary_context(*args: object, **kwargs: object) -> object:
        raise AssertionError("control-only materialization reached primary label construction")

    monkeypatch.setattr(
        inputs,
        "prepare_neutral_phase2_context",
        forbidden_primary_context,
    )
    result, decoded_keys, line_calls, _ = _run_cpu_materialization(
        fixture,
        control_only=True,
    )
    assert type(result) is inputs.R3ControlTrainOnlyMaterializationResult
    assert decoded_keys == list(inputs._STORAGE_TRAIN_KEYS)
    assert len(line_calls) == 16
    result.validate_integrity()
    serialized = result.to_dict()
    capability = serialized["control_input_capability"]
    assert isinstance(capability, dict)
    assert capability["primary_label_stream_constructed"] is False
    assert capability["primary_label_stream_accessed"] is False
    assert capability["heldout_tensor_values_decoded"] is False
    assert capability["policy_session_opened"] is False
    assert capability["raw_tensors_serialized"] is False
    assert capability["raw_oracle_rewards_serialized"] is False
    assert "materialization_capability" not in serialized


def test_real_closure_selectively_decodes_train_prefix_and_returns_timings(
    tmp_path: Path,
) -> None:
    fixture = _build_parent_fixture(tmp_path)
    result, decoded_keys, line_calls, oracle = _run_cpu_materialization(fixture)

    assert decoded_keys == list(inputs._STORAGE_TRAIN_KEYS)
    assert len(line_calls) == 16
    assert all(size == -1 for size in line_calls)
    assert fixture.candidate_suffix.startswith(b"\xff")
    assert result.materialization.heldout_bytes_decoded is False
    assert result.capability.heldout_tensor_values_decoded is False
    assert result.capability.candidate_suffix_decoded is False
    assert result.capability.policy_session_opened is False
    assert result.capability.candidate_train_prefix_sha256 == _sha(
        b"".join(fixture.candidate_lines)
    )
    assert result.capability.parent_registry_file_sha256 == fixture.registry_sha256
    assert result.capability.materialization is result.materialization
    assert "_seal" not in result.capability.to_dict()
    assert result.to_dict()["gate_p_capability_issued"] is False
    assert oracle["device"] == torch.device("cpu")
    assert oracle["batch_size"] == 16
    timings = result.preparation_timings
    assert timings.artifact_verification_wall_seconds == 1.0
    assert timings.oracle_rescore_wall_seconds == 2.0
    assert timings.label_reconstruction_wall_seconds == 3.0
    assert timings.total_preparation_wall_seconds == 6.0
    result.validate_integrity()


def test_capability_cannot_be_constructed_replaced_or_substituted(
    tmp_path: Path,
) -> None:
    fixture = _build_parent_fixture(tmp_path)
    result, *_ = _run_cpu_materialization(fixture)
    capability = result.capability
    serialized = capability.to_dict()
    unsigned = dict(serialized)
    unsigned.pop("capability_sha256")
    unsigned["artifact_tensors_sha256"] = "f" * 64
    forged_sha = inputs._canonical_sha256(unsigned)

    with pytest.raises(TypeError, match="issued by"):
        replace(
            capability,
            artifact_tensors_sha256="f" * 64,
            capability_sha256=forged_sha,
        )

    constructor = dict(serialized)
    constructor["train_tensor_keys_decoded"] = tuple(constructor["train_tensor_keys_decoded"])
    with pytest.raises(TypeError, match="issued by"):
        inputs.R3TrainMaterializationCapability(
            materialization=result.materialization,
            **constructor,
        )

    with pytest.raises(TypeError, match="capability"):
        inputs.R3TrainOnlyMaterializationResult(
            capability=result.materialization,  # type: ignore[arg-type]
            preparation_timings=result.preparation_timings,
        )


def test_duplicate_registry_and_candidate_json_fail_closed(tmp_path: Path) -> None:
    fixture = _build_parent_fixture(tmp_path)
    registry_raw = fixture.registry.read_bytes()
    duplicate = b'{"schema_version":"caller-duplicate",' + registry_raw.removeprefix(b"{")
    fixture.registry.write_bytes(duplicate)
    fixture.registry_sha256 = _sha(duplicate)
    with pytest.raises(ValueError, match="strict UTF-8 JSON"):
        _run_cpu_materialization(fixture)

    fixture = _build_parent_fixture(tmp_path / "candidate")
    first = fixture.candidate_lines[0]
    duplicated_first = first.replace(
        b'{"candidate_id"',
        b'{"prompt_id":"duplicate","candidate_id"',
        1,
    )
    (fixture.artifact_root / "candidates.jsonl").write_bytes(
        duplicated_first + b"".join(fixture.candidate_lines[1:]) + fixture.candidate_suffix
    )
    _refresh_selected_closure(fixture)
    with pytest.raises(ValueError, match="invalid train candidate"):
        _run_cpu_materialization(fixture)


def test_symlink_and_byte_hash_tamper_fail_before_decode(tmp_path: Path) -> None:
    fixture = _build_parent_fixture(tmp_path)
    tensor_path = fixture.artifact_root / "tensors.safetensors"
    tensor_path.write_bytes(tensor_path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="tensor SHA256"):
        _run_cpu_materialization(fixture)

    fixture = _build_parent_fixture(tmp_path / "symlink")
    metadata = fixture.artifact_root / "metadata.json"
    target = fixture.artifact_root / "metadata.real.json"
    metadata.replace(target)
    try:
        os.symlink(target.name, metadata)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(ValueError, match="regular non-symlink"):
        _run_cpu_materialization(fixture)


def test_wrong_seed_nonfinite_tensor_and_oracle_fail_closed(tmp_path: Path) -> None:
    fixture = _build_parent_fixture(tmp_path)
    science = load_r3_science_config(SCIENCE_PATH)
    with pytest.raises(ValueError, match="not declared"):
        inputs._materialize_r3_train_only_from_parent(
            project_root=fixture.root,
            parent_registry_path=fixture.registry,
            expected_parent_registry_file_sha256=fixture.registry_sha256,
            source_config_path=fixture.source_config,
            science_bundle=science,
            seed=20269999,
            target_device=torch.device("cpu"),
            oracle_rescorer=lambda **_: torch.zeros(16),
            clock_ns=iter((0, 1, 2, 3)).__next__,
            safe_open_factory=inputs._default_safe_open,
            line_stream_factory=inputs._default_line_stream,
        )

    fixture = _build_parent_fixture(tmp_path / "nonfinite-tensor")
    safetensors = pytest.importorskip("safetensors.torch")
    tensor_path = fixture.artifact_root / "tensors.safetensors"
    tensors = _training_tensors()
    tensors["train.h"][0] = float("nan")
    replacement = tensor_path.with_suffix(".replacement")
    safetensors.save_file(tensors, str(replacement))
    replacement.replace(tensor_path)
    _refresh_selected_closure(fixture)
    with pytest.raises(ValueError, match="NaN or infinity"):
        _run_cpu_materialization(fixture)

    fixture = _build_parent_fixture(tmp_path / "nonfinite-oracle")
    bad_oracle = torch.zeros(16, dtype=torch.float32)
    bad_oracle[3] = float("inf")
    with pytest.raises(ValueError, match="oracle rescore returned malformed"):
        _run_cpu_materialization(fixture, oracle_values=bad_oracle)


def test_candidate_order_and_timing_regression_fail_closed(tmp_path: Path) -> None:
    fixture = _build_parent_fixture(tmp_path)
    candidate_path = fixture.artifact_root / "candidates.jsonl"
    swapped = (
        fixture.candidate_lines[1]
        + fixture.candidate_lines[0]
        + b"".join(fixture.candidate_lines[2:])
        + fixture.candidate_suffix
    )
    candidate_path.write_bytes(swapped)
    _refresh_selected_closure(fixture)
    with pytest.raises(ValueError, match="canonical tensor order"):
        _run_cpu_materialization(fixture)

    fixture = _build_parent_fixture(tmp_path / "clock")
    with pytest.raises(ValueError, match="strictly monotonic"):
        _run_cpu_materialization(
            fixture,
            clock_values=(10, 20, 20, 30),
        )


@pytest.mark.parametrize(
    ("available", "count", "device", "message"),
    [
        (False, 0, "cuda", "allocated CUDA"),
        (True, 2, "cuda", "exactly one visible CUDA"),
        (True, 1, "cpu", "cuda:0"),
        (True, 1, "cuda:1", "cuda:0"),
    ],
)
def test_public_production_entry_requires_exactly_one_cuda(
    monkeypatch: pytest.MonkeyPatch,
    available: bool,
    count: int,
    device: str,
    message: str,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)
    with pytest.raises(RuntimeError, match=message):
        inputs.materialize_r3_train_only_from_parent(
            project_root="never-read",
            parent_registry_path="never-read",
            expected_parent_registry_file_sha256="0" * 64,
            source_config_path="never-read",
            science_bundle=load_r3_science_config(SCIENCE_PATH),
            seed=SEEDS[0],
            device=device,
        )
