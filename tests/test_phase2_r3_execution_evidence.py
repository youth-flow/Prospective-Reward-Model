from __future__ import annotations

import hashlib
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from conftest import make_shared_gate_p_evidence

from smart_reward import phase2_r3_orchestrator as orchestrator
from smart_reward.phase2_r3_artifacts import canonical_json_bytes, publish_canonical_artifact
from smart_reward.phase2_r3_execution_evidence import (
    publish_primary_identity_receipt,
    publish_segment_evidence_receipt,
    reopen_primary_identity_receipt,
    reopen_segment_evidence_receipt,
)
from smart_reward.phase2_r3_identity import (
    VERIFIED_CONTINUATION_CHECKPOINT_ROLE,
    VERIFIED_CONTINUATION_CHECKPOINT_SCHEMA,
    ArtifactRef,
    admit_primary_segment,
    authorize_gate_p,
    create_r3_primary_design,
)
from smart_reward.phase2_r3_primary import capture_slurm_segment_runtime
from smart_reward.phase2_r3_terminal import publish_primary_segment_runtime_closure


def test_identity_and_segment_receipts_reopen_exact_design_and_task_bytes(
    tmp_path: Path,
) -> None:
    shared = make_shared_gate_p_evidence()
    authorization = authorize_gate_p(
        operational_bundle=shared.operational_bundle,
        successful_terminal=shared.successful_terminal,
    )
    design = create_r3_primary_design(
        science=shared.science,
        gate0_capability=shared.gate0_capability,
        gate1_capabilities=shared.gate1_capabilities,
        profile_authorization=authorization,
        operational_bundle=shared.operational_bundle,
    )
    capability = shared.materialization_capability
    admission = admit_primary_segment(
        design=design,
        materialization_capability=capability,
        task_id=0,
        seed=20260801,
        segment_index=1,
        continuation_evidence=None,
    )
    project_root = tmp_path.resolve()
    plan = project_root / "primary-plan.json"
    plan_body = {
        "schema_version": "phase2-recovery-r3-primary-submission-plan/v2",
        "segment_index": 1,
        "array_task_ids": [0, 1, 2],
        "science_config_file_sha256": shared.science.file_sha256,
        "parent_registry_file_sha256": capability.parent_registry_file_sha256,
    }
    plan_semantic_sha = hashlib.sha256(canonical_json_bytes(plan_body)).hexdigest()
    plan_artifact = publish_canonical_artifact(
        plan,
        {**plan_body, "submission_plan_sha256": plan_semantic_sha},
    )
    plan_file_sha = plan_artifact.file_sha256
    identity_artifact = publish_primary_identity_receipt(
        project_root,
        base_primary_submission_plan_path=plan,
        base_primary_submission_plan_file_sha256=plan_file_sha,
        base_primary_submission_plan_sha256=plan_semantic_sha,
        design=design,
        materialization_capability=capability,
        admission=admission,
    )
    identity = reopen_primary_identity_receipt(
        project_root,
        task_id=0,
        expected_file_sha256=identity_artifact.file_sha256,
        expected_design=design,
        expected_materialization_capability=capability,
    )
    assert identity["segment_1_admission"]["logical_run_id"] == admission.logical_run_id

    environment = {
        "SLURM_CLUSTER_NAME": "hpc4",
        "SLURM_JOB_ID": "701001",
        "SLURM_ARRAY_JOB_ID": "701000",
        "SLURM_ARRAY_TASK_ID": "0",
        "SLURM_JOB_ACCOUNT": shared.operational_bundle.slurm_account,
        "SLURM_JOB_PARTITION": shared.operational_bundle.partition,
    }
    with patch.dict("os.environ", environment, clear=False):
        runtime = capture_slurm_segment_runtime(
            admission,
            requested_walltime_seconds=(
                shared.operational_bundle.requested_walltime_seconds_per_segment
            ),
        )
    task_root = project_root / "scratch-task"
    outcome_directory = task_root / "segment-outcomes"
    outcome_directory.mkdir(parents=True)
    checkpoint_ref = ArtifactRef(
        schema_version=VERIFIED_CONTINUATION_CHECKPOINT_SCHEMA,
        artifact_sha256="b" * 64,
        role=VERIFIED_CONTINUATION_CHECKPOINT_ROLE,
    )
    outcome = orchestrator._materialize_outcome(
        path=outcome_directory / "segment-0001.json",
        admission=admission,
        runtime=runtime,
        operational_policy=shared.operational_bundle,
        status="continuation_required_after_safe_checkpoint",
        receipts={},
        active_learner="bt_mle",
        continuation_checkpoint=checkpoint_ref,
        continuation_reason="test boundary",
    )
    closure_path = project_root / "attempt-1" / "runtime-closures" / "task-0.json"
    closure_path.parent.mkdir(parents=True)
    closure = publish_primary_segment_runtime_closure(
        closure_path,
        admission=admission,
        runtime=runtime,
        outcome=outcome,
        operational_bundle=shared.operational_bundle,
    )
    segment_artifact = publish_segment_evidence_receipt(
        project_root,
        task_root=task_root,
        identity_receipt_file_sha256=identity_artifact.file_sha256,
        closure=closure,
        runtime=runtime,
    )
    reopened = reopen_segment_evidence_receipt(
        project_root,
        task_id=0,
        segment_index=1,
        expected_file_sha256=segment_artifact.file_sha256,
        runtime_closure=closure,
        require_exact_current_manifest=True,
    )
    assert reopened["completed_head_receipts"] == []

    if os.name == "posix":
        outcome.artifact_path.chmod(0o640)
    outcome.artifact_path.write_text('{"tampered":true}\n', encoding="utf-8")
    if os.name == "posix":
        outcome.artifact_path.chmod(0o440)
    with pytest.raises(ValueError, match="bytes changed"):
        reopen_segment_evidence_receipt(
            project_root,
            task_id=0,
            segment_index=1,
            expected_file_sha256=segment_artifact.file_sha256,
            runtime_closure=closure,
            require_exact_current_manifest=True,
        )
