from __future__ import annotations

import hashlib
import importlib.util
import os
import stat
from copy import deepcopy
from pathlib import Path

import pytest

from smart_reward import phase2_r3_authorization as gate_r_module
from smart_reward import phase2_r3_controls_hpc4 as gate_c
from smart_reward.config import ConfigError, load_config
from smart_reward.phase2_config import (
    build_post_recovery_authorization_reference,
    load_phase2_config,
    validate_phase2_config,
    validate_post_recovery_authorization_reference,
)
from smart_reward.phase2_post_recovery_control import (
    verify_recovery_authorization_file,
)
from smart_reward.phase2_r3_artifacts import (
    canonical_json_bytes,
    publish_canonical_artifact,
    read_canonical_artifact,
)
from smart_reward.phase2_r3_post_recovery_authorization import (
    publish_r3_final_authorization,
    verify_r3_final_authorization,
)
from smart_reward.phase2_r3_post_recovery_contract import (
    R3_FINAL_AUTHORIZATION_PROJECTION_SCHEMA,
    R3_FINAL_AUTHORIZATION_REFERENCE_SCHEMA,
    R3_FINAL_AUTHORIZATION_RELATIVE,
    R3_GATE_C_AGGREGATE_RELATIVE,
    R3_GATE_R_AUTHORIZATION_RELATIVE,
    R3_OPTIMIZER_SCHEDULE_SHA256,
    R3_ORDERED_RECOVERY_SEEDS,
)

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = (
    "exact_margin_prorm_plus",
    "exact_soft_label_bt",
    "low_dimensional_prorm_plus",
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _aggregate() -> dict[str, object]:
    sources = []
    task_id = 0
    for family in FAMILIES:
        for seed in R3_ORDERED_RECOVERY_SEEDS:
            sources.append(
                {
                    "task_id": task_id,
                    "family": family,
                    "seed": seed,
                    "family_result_file_sha256": _digest(f"result-file:{task_id}"),
                    "family_result_sha256": _digest(f"result:{task_id}"),
                    "closure_sha256": _digest(f"closure:{task_id}"),
                    "terminal_sha256": _digest(f"terminal:{task_id}"),
                    "raw_sacct_sha256": _digest(f"sacct:{task_id}"),
                }
            )
            task_id += 1
    body: dict[str, object] = {
        "schema_version": gate_c.R3_CONTROLS_AGGREGATE_SCHEMA,
        "role": "exact_three_families_by_three_seeds_train_only_gate_c_closure",
        "plan_sha256": _digest("plan"),
        "profile_sha256": _digest("profile"),
        "optimizer_schedule_sha256": R3_OPTIMIZER_SCHEDULE_SHA256,
        "ordered_families": list(FAMILIES),
        "ordered_seeds": list(R3_ORDERED_RECOVERY_SEEDS),
        "matrix_shape": [3, 3],
        "all_nine_compute_complete": True,
        "all_nine_scheduler_success": True,
        "gate_c_passed": True,
        "fresh_calibration_authorized": False,
        "result_reusable_for_training": False,
        "information_boundary": "train_only_local_mechanism_evidence",
        "sources": sources,
        "source_set_sha256": gate_c._semantic_sha256({"sources": sources}),
    }
    return {**body, "aggregate_sha256": gate_c._semantic_sha256(body)}


def _gate_r() -> dict[str, object]:
    return {
        "schema_version": "phase2-recovery-r3-success-authorization/v1",
        "recovery_design_sha256": _digest("r3-design"),
        "optimizer_schedule_sha256": R3_OPTIMIZER_SCHEDULE_SHA256,
        "execution_revision": 3,
        "ordered_seeds": list(R3_ORDERED_RECOVERY_SEEDS),
        "gate_r_passed": True,
        "fresh_calibration_authorized": False,
        "authorization_sha256": _digest("gate-r-semantic"),
    }


def _combined(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    aggregate = _aggregate()
    gate_r = _gate_r()
    monkeypatch.setattr(gate_c, "_validated_gate_r", lambda value: deepcopy(value))
    combined = gate_c.build_controls_authorization(
        aggregate,
        gate_r_authorization=gate_r,
        gate_r_authorization_file_sha256=_digest("gate-r-file"),
    )
    return gate_r, aggregate, combined


def _load_script(name: str):
    path = ROOT / "scripts" / "hpc4" / name
    spec = importlib.util.spec_from_file_location(f"_r3_bridge_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rewrite_canonical(path: Path, value: dict[str, object]) -> str:
    if os.name == "posix":
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    path.write_bytes(canonical_json_bytes(value))
    if os.name == "posix":
        os.chmod(path, stat.S_IRUSR | stat.S_IRGRP)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_r3_reference_is_closed_roundtrips_and_cannot_downgrade_to_r2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, authorization = _combined(monkeypatch)
    artifact_sha256 = _digest("combined-file")
    reference = build_post_recovery_authorization_reference(
        authorization,
        artifact_sha256=artifact_sha256,
    )
    assert reference["schema_version"] == R3_FINAL_AUTHORIZATION_REFERENCE_SCHEMA
    projection = reference["authorization_projection"]
    assert projection["schema_version"] == R3_FINAL_AUTHORIZATION_PROJECTION_SCHEMA
    assert projection["recovery_design_sha256"] == authorization["recovery_design_sha256"]
    assert (
        projection["gate_r_authorization_sha256"] == (authorization["gate_r_authorization_sha256"])
    )
    assert projection["gate_c_aggregate_sha256"] == (authorization["gate_c_aggregate_sha256"])
    assert projection["ordered_seeds"] == list(R3_ORDERED_RECOVERY_SEEDS)

    validated = validate_post_recovery_authorization_reference(
        reference,
        authorization_payload_sha256=artifact_sha256,
        authorization_payload=authorization,
    )
    assert validated == reference

    downgrade = deepcopy(reference)
    downgrade["schema_version"] = "prorm-phase2-recovery-success-reference/v1"
    with pytest.raises(ConfigError, match="unknown keys|must equal"):
        validate_post_recovery_authorization_reference(downgrade)

    heldout = deepcopy(reference)
    heldout["authorization_projection"]["validation_or_heldout_access_authorized"] = True
    with pytest.raises(ConfigError, match="must be false"):
        validate_post_recovery_authorization_reference(heldout)


def test_r3_materializer_candidate_keeps_fresh_five_head_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, authorization = _combined(monkeypatch)
    artifact_sha256 = _digest("combined-file")
    materializer = _load_script("materialize_phase2_post_recovery_calibration.py")
    template = load_phase2_config(ROOT / "configs" / "common_beta_pilot.yaml")
    recovery = load_phase2_config(ROOT / "configs" / "common_beta_recovery_pilot.yaml")
    base = load_config(ROOT / "configs" / "common_beta_pilot_base.yaml")

    candidate = materializer._candidate(
        template=template,
        recovery=recovery,
        authorization=authorization,
        authorization_sha256=artifact_sha256,
    )
    normalized = validate_phase2_config(candidate, base_config=base)
    reference = normalized["recovery_success_reference"]
    protocol = normalized["reward_model"]["optimizer_protocol"]
    assert reference["schema_version"] == R3_FINAL_AUTHORIZATION_REFERENCE_SCHEMA
    assert reference["artifact_sha256"] == artifact_sha256
    assert protocol["source_recovery_authorization_sha256"] == artifact_sha256
    assert protocol["initialization"] == "exact_zero_head_and_fresh_optimizer_state"
    assert normalized["positive_controls"]["low_dimensional_tangent"]["enabled"] is True
    assert normalized["positive_controls"]["exact_soft_label_bt"]["enabled"] is True
    assert protocol["validation_or_test_selection"] is False

    stage_materializer = _load_script("materialize_phase2_post_recovery_stage.py")
    freeze = stage_materializer._freeze_projection(
        normalized,
        beta=2.0,
        beta_source_sha256=_digest("calibration-aggregate"),
        horizon_parent_sha256=_digest("calibration-aggregate"),
        beta_grid_index=0,
    )
    normalized_freeze = validate_phase2_config(freeze, base_config=base)
    confirmatory, confirmatory_base = stage_materializer._confirmatory_projection(
        normalized_freeze,
        base,
        freeze_sha256=_digest("freeze-aggregate"),
        frozen_beta=2.0,
    )
    normalized_confirmatory = validate_phase2_config(
        confirmatory,
        base_config=confirmatory_base,
    )
    budgeted_materializer = _load_script("materialize_phase2_budgeted_end_to_end.py")
    budgeted, budgeted_base = budgeted_materializer._budgeted_projection(
        normalized_freeze,
        base,
        freeze_sha256=_digest("freeze-aggregate"),
        frozen_beta=2.0,
    )
    normalized_budgeted = validate_phase2_config(
        budgeted,
        base_config=budgeted_base,
    )
    for propagated in (
        normalized_freeze,
        normalized_confirmatory,
        normalized_budgeted,
    ):
        assert propagated["recovery_success_reference"] == reference
        assert (
            propagated["reward_model"]["optimizer_protocol"]["source_recovery_authorization_sha256"]
            == artifact_sha256
        )


def test_r3_file_verifier_reopens_exact_gate_r_and_gate_c_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path.resolve()
    for relative in (
        R3_GATE_R_AUTHORIZATION_RELATIVE,
        R3_GATE_C_AGGREGATE_RELATIVE,
        R3_FINAL_AUTHORIZATION_RELATIVE,
    ):
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
    gate_r, aggregate, _ = _combined(monkeypatch)
    gate_r_artifact = publish_canonical_artifact(
        root / R3_GATE_R_AUTHORIZATION_RELATIVE,
        {"fixture": "gate-r-source-bytes"},
    )
    gate_c_artifact = publish_canonical_artifact(
        root / R3_GATE_C_AGGREGATE_RELATIVE,
        aggregate,
    )

    def fake_gate_r_verifier(
        path: str | os.PathLike[str],
        *,
        expected_sha256: str,
        project_root: str | os.PathLike[str] | None,
    ) -> dict[str, object]:
        assert Path(path) == root / R3_GATE_R_AUTHORIZATION_RELATIVE
        assert Path(project_root) == root
        read_canonical_artifact(path, expected_file_sha256=expected_sha256)
        return deepcopy(gate_r)

    monkeypatch.setattr(
        gate_r_module,
        "verify_r3_success_authorization",
        fake_gate_r_verifier,
    )
    artifact = publish_r3_final_authorization(
        gate_r_authorization=root / R3_GATE_R_AUTHORIZATION_RELATIVE,
        gate_r_authorization_file_sha256=gate_r_artifact.file_sha256,
        gate_c_aggregate=root / R3_GATE_C_AGGREGATE_RELATIVE,
        gate_c_aggregate_file_sha256=gate_c_artifact.file_sha256,
        output=root / R3_FINAL_AUTHORIZATION_RELATIVE,
        project_root=root,
    )
    verified = verify_r3_final_authorization(
        artifact.artifact_path,
        expected_sha256=artifact.file_sha256,
        project_root=root,
    )
    dispatched = verify_recovery_authorization_file(
        artifact.artifact_path,
        expected_sha256=artifact.file_sha256,
        project_root=root,
    )
    assert dispatched == verified
    assert verified["fresh_calibration_authorized"] is True
    assert verified["gate_r_authorization_file_sha256"] == gate_r_artifact.file_sha256
    assert verified["gate_c_aggregate_file_sha256"] == gate_c_artifact.file_sha256

    tampered = artifact.payload
    tampered["fresh_calibration_authorized"] = False
    tampered_file_sha256 = _rewrite_canonical(artifact.artifact_path, tampered)
    with pytest.raises(ValueError, match="self-hash"):
        verify_r3_final_authorization(
            artifact.artifact_path,
            expected_sha256=tampered_file_sha256,
            project_root=root,
        )


def test_r3_file_verifier_rejects_gate_c_source_byte_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path.resolve()
    for relative in (
        R3_GATE_R_AUTHORIZATION_RELATIVE,
        R3_GATE_C_AGGREGATE_RELATIVE,
        R3_FINAL_AUTHORIZATION_RELATIVE,
    ):
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
    gate_r, aggregate, _ = _combined(monkeypatch)
    gate_r_artifact = publish_canonical_artifact(
        root / R3_GATE_R_AUTHORIZATION_RELATIVE,
        {"fixture": "gate-r-source-bytes"},
    )
    gate_c_artifact = publish_canonical_artifact(
        root / R3_GATE_C_AGGREGATE_RELATIVE,
        aggregate,
    )

    monkeypatch.setattr(
        gate_r_module,
        "verify_r3_success_authorization",
        lambda path, *, expected_sha256, project_root: deepcopy(gate_r),
    )
    authorization = gate_c.build_controls_authorization(
        aggregate,
        gate_r_authorization=gate_r,
        gate_r_authorization_file_sha256=gate_r_artifact.file_sha256,
    )
    final_artifact = publish_canonical_artifact(
        root / R3_FINAL_AUTHORIZATION_RELATIVE,
        authorization,
    )
    modified = deepcopy(aggregate)
    modified["plan_sha256"] = _digest("tampered-plan")
    _rewrite_canonical(gate_c_artifact.artifact_path, modified)

    with pytest.raises(ValueError, match="file SHA-256"):
        verify_r3_final_authorization(
            final_artifact.artifact_path,
            expected_sha256=final_artifact.file_sha256,
            project_root=root,
        )
