from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest

from smart_reward import phase2_r3_authorization as gate_r
from smart_reward import phase2_r3_controls_hpc4 as gate_c
from smart_reward.phase2_r3_artifacts import publish_canonical_artifact
from smart_reward.phase2_r3_post_recovery_authorization import (
    publish_r3_final_authorization,
)
from smart_reward.phase2_r3_post_recovery_contract import (
    R3_FINAL_AUTHORIZATION_RELATIVE,
    R3_GATE_C_AGGREGATE_RELATIVE,
    R3_OPTIMIZER_SCHEDULE_SHA256,
    R3_ORDERED_RECOVERY_SEEDS,
)

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = (
    "exact_margin_prorm_plus",
    "exact_soft_label_bt",
    "low_dimensional_prorm_plus",
)


def _load_path(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _gate_c_aggregate() -> dict[str, object]:
    sources: list[dict[str, object]] = []
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


def test_real_builders_roundtrip_to_exact_head_free_bind_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path.resolve()
    authorization_support = _load_path(
        ROOT / "tests" / "test_phase2_r3_authorization.py",
        "_r3_authorization_support",
    )
    inspector = _load_path(
        ROOT / "scripts" / "hpc4" / "inspect_phase2_post_recovery_stdlib.py",
        "_post_recovery_stdlib_inspector",
    )

    histories = authorization_support._histories(root, monkeypatch)
    bundle = histories[0][0].operational_bundle
    monkeypatch.setattr(
        gate_r,
        "reopen_verified_gate_p_operational_bundle",
        lambda _path, *, expected_file_sha256: bundle,
    )
    gate_r_artifact = gate_r.publish_r3_success_authorization(
        histories,
        project_root=root,
    )
    gate_c_path = root / R3_GATE_C_AGGREGATE_RELATIVE
    gate_c_path.parent.mkdir(parents=True, exist_ok=True)
    gate_c_artifact = publish_canonical_artifact(
        gate_c_path,
        _gate_c_aggregate(),
    )
    final_artifact = publish_r3_final_authorization(
        gate_r_authorization=gate_r_artifact.artifact_path,
        gate_r_authorization_file_sha256=gate_r_artifact.file_sha256,
        gate_c_aggregate=gate_c_artifact.artifact_path,
        gate_c_aggregate_file_sha256=gate_c_artifact.file_sha256,
        output=root / R3_FINAL_AUTHORIZATION_RELATIVE,
        project_root=root,
    )

    inspected = inspector.inspect_authorization(
        final_artifact.artifact_path,
        expected_sha256=final_artifact.file_sha256,
        project_root=root,
    )
    bindings = inspected["bindings"]
    assert inspected["schema_version"] == inspector.R3_AUTHORIZATION_SCHEMA
    assert len(bindings) == 11
    assert sum(binding["kind"] == "directory" for binding in bindings) == 4
    assert all("head.pt" not in binding["path"] for binding in bindings)
    assert all("checkpoint" not in Path(binding["path"]).name for binding in bindings)


@pytest.mark.parametrize(
    ("path", "task_id"),
    (
        ("runs/phase2-recovery-r3/attempt/runtime-closures/head.pt", 0),
        ("runs/phase2-recovery-r3/attempt/runtime-closures/task-1.json", 0),
        ("runs/phase2-recovery-r3/attempt/checkpoints/task-0.json", 0),
    ),
)
def test_runtime_closure_namespace_cannot_bind_model_payloads(
    path: str,
    task_id: int,
) -> None:
    inspector = _load_path(
        ROOT / "scripts" / "hpc4" / "inspect_phase2_post_recovery_stdlib.py",
        "_post_recovery_stdlib_inspector_rejection",
    )
    with pytest.raises(ValueError, match="exact evidence namespace"):
        inspector._require_closure_namespace(path, task_id=task_id)
