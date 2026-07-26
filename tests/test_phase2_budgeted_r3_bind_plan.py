from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from smart_reward import phase2_r3_authorization as gate_r_module
from smart_reward.phase2_r3_artifacts import publish_canonical_artifact

ROOT = Path(__file__).resolve().parents[1]
BIND_SCRIPT = ROOT / "scripts" / "hpc4" / "phase2_budgeted_r3_bind_plan_stdlib.py"
AUTHORIZATION_SUPPORT = ROOT / "tests" / "test_phase2_r3_authorization.py"
SCHEDULE = "46e0b0fdc70c507b0325c445068326ba7bc30326d70b65fa33803cc3c876c216"


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _semantic(value: dict[str, object], *, newline: bool = False) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    if newline:
        encoded += "\n"
    return hashlib.sha256(encoded.encode()).hexdigest()


def _combined_fixture(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, str]:
    support = _load(AUTHORIZATION_SUPPORT, "_gate_r_bind_support")
    gate_r_output = root / gate_r_module.R3_SUCCESS_AUTHORIZATION_RELATIVE
    gate_r_output.parent.mkdir(parents=True)
    histories = support._histories(root, monkeypatch)
    gate_r = gate_r_module.publish_r3_success_authorization(
        histories,
        project_root=root,
    )

    gate_c_body = {
        "schema_version": "phase2-recovery-r3-gate-c-aggregate/v1",
        "role": "exact_three_families_by_three_seeds_train_only_gate_c_closure",
        "optimizer_schedule_sha256": SCHEDULE,
        "ordered_seeds": [20260801, 20260802, 20260803],
        "matrix_shape": [3, 3],
        "gate_c_passed": True,
        "fresh_calibration_authorized": False,
        "result_reusable_for_training": False,
    }
    gate_c_payload = {
        **gate_c_body,
        "aggregate_sha256": _semantic(gate_c_body),
    }
    (root / "runs" / "phase2-recovery-r3-controls").mkdir(parents=True)
    gate_c = publish_canonical_artifact(
        root / "runs/phase2-recovery-r3-controls/gate-c-aggregate.json",
        gate_c_payload,
    )
    transport = {
        "parameters": False,
        "optimizer_moments": False,
        "checkpoints": False,
        "labels_or_data": False,
        "gradients_or_directions": False,
        "validation_or_test_values": False,
        "policy_outputs": False,
        "utility_values": False,
        "beta_values": False,
    }
    body: dict[str, object] = {
        "schema_version": "phase2-recovery-r3-gate-c-success-authorization/v1",
        "role": "head_free_exact_three_by_three_gate_c_success_capability",
        "recovery_design_sha256": gate_r.payload["recovery_design_sha256"],
        "optimizer_schedule_sha256": SCHEDULE,
        "optimizer_schedule_is_unique": True,
        "execution_revision": 3,
        "ordered_seeds": [20260801, 20260802, 20260803],
        "gate_r_authorization_path": (
            "runs/phase2-recovery-r3/recovery-success-authorization.json"
        ),
        "gate_r_authorization_file_sha256": gate_r.file_sha256,
        "gate_r_authorization_sha256": gate_r.payload["authorization_sha256"],
        "gate_c_aggregate_path": ("runs/phase2-recovery-r3-controls/gate-c-aggregate.json"),
        "gate_c_aggregate_file_sha256": gate_c.file_sha256,
        "gate_c_aggregate_sha256": gate_c_payload["aggregate_sha256"],
        "gate_r_passed": True,
        "gate_c_passed": True,
        "fresh_calibration_authorized": True,
        "authorized_information": ("gate_r_design_optimizer_schedule_and_gate_source_hashes_only"),
        "authorized_next_action": "materialize_fresh_common_beta_calibration",
        "formal_efficacy_claim_authorized": False,
        "recovery_or_control_outputs_reusable": False,
        "validation_or_heldout_access_authorized": False,
        "policy_or_final_utility_access_authorized": False,
        "transport_boundary": transport,
        "gate_c_source_set_sha256": "e" * 64,
    }
    final = publish_canonical_artifact(
        root / "runs" / "phase2-recovery-r3-controls" / "gate-c-success-authorization.json",
        {**body, "authorization_sha256": _semantic(body)},
    )
    return final.artifact_path, final.file_sha256


def test_minimal_bind_plan_is_complete_and_excludes_trained_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load(BIND_SCRIPT, "_budgeted_bind_plan")
    root = tmp_path.resolve()
    authorization, sha256 = _combined_fixture(root, monkeypatch)
    plan = module.build_bind_plan(
        authorization,
        expected_sha256=sha256,
        project_root=root,
    )
    assert plan["trained_outputs_bound"] is False
    assert len(plan["bind_paths"]) >= 10
    assert all(
        token not in path
        for path in plan["bind_paths"]
        for token in ("/heads/", "/checkpoints/", "/models/")
    )
    assert any(
        Path(path).as_posix().endswith("gatep-attempt-001/gatep-operational-bundle.json")
        for path in plan["bind_paths"]
    )


def test_bind_plan_fails_when_one_dependency_is_missing_or_terminal_is_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load(BIND_SCRIPT, "_budgeted_bind_plan_missing")
    root = tmp_path.resolve()
    authorization, sha256 = _combined_fixture(root, monkeypatch)
    closure = next(root.glob("runs/phase2-recovery-r3/**/runtime-closures/task-0.json"))
    closure.unlink()
    with pytest.raises(ValueError, match="unavailable"):
        module.build_bind_plan(
            authorization,
            expected_sha256=sha256,
            project_root=root,
        )

    second_root = (tmp_path / "second").resolve()
    authorization, sha256 = _combined_fixture(second_root, monkeypatch)
    terminal = next(
        second_root.glob("runs/phase2-recovery-r3/**/terminal-evidence/task-0-segment-1")
    )
    (terminal / "head.pt").write_bytes(b"forbidden")
    with pytest.raises(ValueError, match="exactly"):
        module.build_bind_plan(
            authorization,
            expected_sha256=sha256,
            project_root=second_root,
        )


def test_bind_plan_rejects_rehashed_open_boundary_and_non_ascii_canonical_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load(BIND_SCRIPT, "_budgeted_bind_plan_boundary")
    root = tmp_path.resolve()
    authorization, _ = _combined_fixture(root, monkeypatch)
    payload = json.loads(authorization.read_text(encoding="utf-8"))
    opened = copy.deepcopy(payload)
    opened["transport_boundary"]["checkpoints"] = True
    unsigned = dict(opened)
    unsigned.pop("authorization_sha256")
    opened["authorization_sha256"] = _semantic(unsigned)
    authorization.unlink()
    artifact = publish_canonical_artifact(authorization, opened)
    with pytest.raises(ValueError, match="envelope"):
        module.build_bind_plan(
            authorization,
            expected_sha256=artifact.file_sha256,
            project_root=root,
        )

    with pytest.raises(ValueError, match="canonical"):
        module._decode_strict_json(
            '{"name":"é"}\n'.encode(),
            name="non-ASCII alias",
            canonical=True,
        )


def test_gatep_attempt_001_is_admissible_but_000_is_not(tmp_path: Path) -> None:
    module = _load(BIND_SCRIPT, "_budgeted_bind_plan_gatep")
    root = tmp_path.resolve()
    valid = (
        root
        / "runs"
        / "phase2-recovery-r3"
        / "gatep"
        / "profile"
        / "gatep-attempt-001"
        / "gatep-operational-bundle.json"
    )
    valid.parent.mkdir(parents=True)
    valid.write_text("{}\n", encoding="utf-8")
    module._validate_bundle_namespace(valid, root=root)
    invalid = valid.parents[1] / "gatep-attempt-000" / valid.name
    invalid.parent.mkdir()
    invalid.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="canonical namespace"):
        module._validate_bundle_namespace(invalid, root=root)
