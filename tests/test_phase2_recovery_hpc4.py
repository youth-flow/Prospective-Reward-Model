from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch

from smart_reward.phase2_config import load_phase2_config_bundle
from smart_reward.phase2_recovery import (
    TRAIN_TENSOR_KEYS,
    _load_train_candidate_prefix_only,
    _load_train_tensors_only,
)
from smart_reward.phase2_training import compile_phase2_training_settings

ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT / "configs" / "common_beta_recovery_pilot.yaml"
PARENT_REGISTRY = ROOT / "configs" / "phase2_recovery_parent_failures.json"
SUBMIT = ROOT / "scripts" / "hpc4" / "submit_phase2_recovery_pilot.sh"
JOB = ROOT / "scripts" / "hpc4" / "phase2_recovery_pilot.sbatch"
VALIDATOR = ROOT / "scripts" / "hpc4" / "validate_phase2_recovery_parent.py"
RUNNER = ROOT / "scripts" / "hpc4" / "run_phase2_recovery_train.py"
REGISTRY_SHA = "7be4ee90b1f494d32f96214f407a57cbee54be86a77dacc1206d2acd527857dc"
DESIGN_SHA = "9602b0f00a73880545fd57ce1886ec65d7901385cce2b919fd72f3efec4592d4"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_recovery_overlay_and_parent_registry_have_frozen_identities() -> None:
    bundle = load_phase2_config_bundle(OVERLAY)
    settings = compile_phase2_training_settings(bundle)
    assert bundle.design_identity == DESIGN_SHA
    assert _sha(PARENT_REGISTRY) == REGISTRY_SHA
    assert settings.convergence.max_steps == 12760
    protocol = settings.convergence.optimizer_protocol
    assert protocol is not None
    assert [stage.learning_rate for stage in protocol.stages] == [
        1.0e-3,
        3.0e-4,
        1.0e-4,
        3.0e-5,
        1.0e-5,
    ]
    control = bundle.config["recovery_control"]
    assert control["parent_failure_registry_sha256"] == REGISTRY_SHA
    assert control["execution_scope"] == "train_only"
    assert control["one_shot_no_further_adaptation"] is True
    assert control["policy_rollout_allowed"] is False
    assert control["validation_or_test_access_allowed"] is False
    assert control["final_oracle_allowed"] is False
    assert control["downstream_utility_allowed"] is False


def test_old_pilot_overlay_and_failed_registry_are_not_conflated() -> None:
    assert _sha(ROOT / "configs" / "common_beta_pilot.yaml") == (
        "b855883b744ed87c998e8771fe8c4f736ed132c97977ddcf672c5eeed143fb29"
    )
    registry = json.loads(PARENT_REGISTRY.read_text(encoding="utf-8"))
    assert [entry["seed"] for entry in registry["seeds"]] == [
        20260801,
        20260802,
        20260803,
    ]
    assert registry["campaign"]["source_job_array_id"] == "1647491"
    assert registry["campaign"]["failure_aggregate"]["present"] is False
    assert registry["optimizer_diagnostic"]["sha256"] == (
        "bd7c3d80c26500ee273b14bb1ea8bc3428f71fdb319a49c792bf4de567e2c6a9"
    )


def test_recovery_hpc_control_plane_is_fail_closed_and_train_only() -> None:
    submit = SUBMIT.read_text(encoding="utf-8")
    job = JOB.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    implementation = (ROOT / "src" / "smart_reward" / "phase2_recovery.py").read_text(
        encoding="utf-8"
    )
    assert '--array="0-2%${concurrency}"' in submit
    assert 'concurrency="${3:-3}"' in submit
    assert "--export=NONE" not in submit
    assert 'export_spec="PATH=/usr/local/bin:/usr/bin:/bin,' in submit
    assert '--export="${export_spec}"' in submit
    assert "may not contain commas or newlines" in submit
    assert "may not contain a colon" in submit
    assert "slurm-logs/phase2-recovery-pilot" in submit
    assert '--output="${log_dir}/%x-%A_%a.out"' in submit
    assert '--error="${log_dir}/%x-%A_%a.err"' in submit
    assert "merge-base --is-ancestor" in submit
    assert "materialization-relevant blobs changed" in submit
    assert "configs/identities.json" in submit
    assert "one-shot recovery already has a terminal namespace" in submit
    assert "--verify-sources" in submit
    assert "${artifact}:/parent-artifact:ro" in job
    assert "artifact-snapshot-before.json" in job
    assert "artifact-snapshot-after.json" in job
    assert "parent-run-snapshot-before.json" in job
    assert "parent-run-snapshot-after.json" in job
    assert "cmp -s" in job
    assert "phase2-recovery-pilot/${PRORM_PHASE2_RECOVERY_DESIGN_SHA256}" in job
    assert "one_shot_no_further_adaptation=true" in job
    assert "#SBATCH --no-requeue" in job
    assert "SLURM_RESTART_COUNT" in job
    assert "atomic_copy_required" in job
    assert 'if [[ "${workload}" = 0 ]]' in job
    assert '[[ ! -e "${job_dir}/recovery-failure-evidence.json"' in job
    assert 'convergence.get("schema_version")!="objective-first-order-convergence/v2"' in job
    assert 'training.get("schema_version")!="phase2-fresh-head-training/v3"' in job
    assert '"low_dimensional_prorm_plus"' in job
    assert '"exact_margin_prorm_plus"' in job
    assert '"exact_soft_label_bt"' in job
    assert "selected_primary_optimizer_state_restored_and_verified" in job
    assert "all_updates_checked_before_and_after" in job
    assert "diagnostic seed oracle/label reproducibility anchor failed" in job
    assert "7a7d7b005ec7e377205d6f40743bed950ad38154dec6f54516f7ced8ffca0b1a" in job
    assert "container image bytes changed before job execution" in job
    assert "HF inventory bytes changed before job execution" in job
    assert "HF inventory changed during in-container verification" in job
    assert "34574e1b1dc22a9503b89249059596d92aa5c3df074022ecfc8ff008dc4bc3af" in job
    assert "46e0b0fdc70c507b0325c445068326ba7bc30326d70b65fa33803cc3c876c216" in job
    assert "controlled-materialize" not in job
    assert "python -m smart_reward.cli phase2-run" not in job
    assert "policy_session(" not in runner
    assert "policy_session(" not in implementation
    assert "prepare_phase2_inputs(" not in implementation
    assert "load_controlled_feature_artifact" not in implementation
    assert "run_common_beta_rollouts" not in implementation
    assert "final_oracle" in implementation
    assert "downstream_utility_computed" in implementation


def test_shell_entry_points_parse() -> None:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is unavailable on this Windows test host")
    for path in (SUBMIT, JOB):
        completed = subprocess.run(
            [bash, "-n", str(path)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr


def test_parent_registry_validates_without_touching_external_sources() -> None:
    namespace: dict[str, object] = {"__name__": "phase2_recovery_parent_validator"}
    exec(compile(VALIDATOR.read_text(encoding="utf-8"), str(VALIDATOR), "exec"), namespace)
    result = namespace["load_and_validate_registry"](
        PARENT_REGISTRY,
        project_root=None,
        expected_registry_sha256=REGISTRY_SHA,
        expected_parent_design_sha256=(
            "0c8820b67b8ca85c23cd5c31b8d25001018b31a7271f6a861d88cfae2f85d7ca"
        ),
        expected_base_config_hash=(
            "81ccbd3bc9d745d3792d1834116ba2480d34a7201d6f4e463a61d8d8ab0baefa"
        ),
        seed=20260802,
        verify_sources=False,
    )
    assert result["status"] == "ok"
    assert result["selected_seed"]["seed"] == 20260802
    assert result["all_three_sources_verified"] is False


def test_train_tensor_loader_requests_only_train_keys(tmp_path: Path, monkeypatch) -> None:
    tensor_path = tmp_path / "tensors.safetensors"
    tensor_path.write_bytes(b"integrity-only-placeholder")
    tensors = {
        "train.policy_scores": torch.zeros(2, 2, 3),
        "train.reward_features": torch.zeros(2, 2, 4),
        "train.h": torch.zeros(2),
        "train.left_wins": torch.ones(2, dtype=torch.int64),
        "train.num_annotations": torch.full((2,), 2, dtype=torch.int64),
    }
    requested: list[str] = []

    class Handle:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def get_tensor(self, key: str) -> torch.Tensor:
            requested.append(key)
            assert not key.startswith(("validation.", "test."))
            return tensors[key]

    fake = types.ModuleType("safetensors")
    fake.safe_open = lambda *args, **kwargs: Handle()
    monkeypatch.setitem(sys.modules, "safetensors", fake)
    specs = {
        key: (tuple(value.shape), str(value.dtype).removeprefix("torch."))
        for key, value in tensors.items()
    }
    metadata = {"splits": {"train": {"prompt_ids": ["p0", "p1"]}}}
    entry = {
        "artifact_sha256": {
            "tensors.safetensors": hashlib.sha256(tensor_path.read_bytes()).hexdigest()
        }
    }
    train = _load_train_tensors_only(
        tmp_path,
        metadata=metadata,
        specs=specs,
        entry=entry,
    )
    assert tuple(requested) == TRAIN_TENSOR_KEYS
    assert train.prompt_ids == ("p0", "p1")


def test_candidate_loader_does_not_decode_heldout_suffix(tmp_path: Path) -> None:
    rows = []
    for prompt_id in ("p0", "p1"):
        for candidate_index in range(2):
            rows.append(
                {
                    "prompt_id": prompt_id,
                    "candidate_id": f"{prompt_id}::candidate::{candidate_index}",
                    "prompt": f"prompt {prompt_id}",
                    "response": "response",
                    "token_ids": [1, 2],
                    "response_mask": [0, 1],
                    "terminated_by_eos": True,
                    "reached_max_length": False,
                    "schema_version": "candidate-node/v1",
                }
            )
    payload = (
        b"".join(json.dumps(row, separators=(",", ":")).encode() + b"\n" for row in rows)
        + b"this heldout suffix is intentionally not JSON\n"
    )
    path = tmp_path / "candidates.jsonl"
    path.write_bytes(payload)
    from smart_reward.experiment import TrainingTensorData

    train = TrainingTensorData(
        prompt_ids=("p0", "p1"),
        policy_scores=torch.zeros(2, 2, 3),
        reward_features=torch.zeros(2, 2, 4),
        h=torch.zeros(2),
        left_wins=torch.ones(2, dtype=torch.int64),
        num_annotations=torch.full((2,), 2, dtype=torch.int64),
    )
    entry = {
        "artifact_sha256": {
            "candidates.jsonl": hashlib.sha256(payload).hexdigest(),
        }
    }
    candidates = _load_train_candidate_prefix_only(tmp_path, train=train, entry=entry)
    assert len(candidates) == 4
