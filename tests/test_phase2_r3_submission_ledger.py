from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from smart_reward.phase2_r3_artifacts import publish_canonical_artifact

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "hpc4" / "submit_phase2_r3_once.py"
PRIMARY_LAUNCHER = ROOT / "scripts" / "hpc4" / "submit_phase2_r3_primary.sh"
CONTINUATION_LAUNCHER = ROOT / "scripts" / "hpc4" / "submit_phase2_r3_continuation.sh"
PRIMARY_SUBMIT = ROOT / "scripts" / "hpc4" / "phase2_r3_primary_submission.sbatch"
CONTINUATION_SUBMIT = ROOT / "scripts" / "hpc4" / "phase2_r3_continuation_submission.sbatch"
AUTHORIZATION_VALIDATOR = ROOT / "scripts" / "hpc4" / "validate_phase2_r3_authorization.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_r3_submit_once_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_authorization_validator() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_r3_authorization_ledger_test",
        AUTHORIZATION_VALIDATOR,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_complete_submission_ledger_reopens_and_binds_plan_bytes(tmp_path: Path) -> None:
    module = _load()
    root = tmp_path.resolve()
    plan = root / "plan.json"
    plan_body = {"schema_version": "phase2-recovery-r3-primary-submission-plan/v2"}
    semantic = module._semantic_sha256(plan_body)
    plan_artifact = publish_canonical_artifact(
        plan,
        {**plan_body, "submission_plan_sha256": semantic},
    )
    plan_file_sha = plan_artifact.file_sha256
    sbatch_script = root / "primary.sbatch"
    sbatch_script.write_text("#!/bin/bash\n", encoding="utf-8")
    sbatch_script_sha = module._sha256(sbatch_script.read_bytes())
    sbatch_command = [
        "sbatch",
        "--hold",
        "--parsable",
        "--no-requeue",
        f"--comment=prorm-r3-{semantic}",
        "--job-name=prorm-r3-primary-s1",
        "--array=0-2%2",
        (
            "--export=PATH=/usr/bin:/bin,"
            f"PRORM_R3_GIT_COMMIT={'b' * 40},"
            f"PRORM_R3_IMAGE_SHA256={'c' * 64},"
            f"PRORM_R3_PRIMARY_SUBMISSION_PLAN={plan},"
            f"PRORM_R3_PRIMARY_SUBMISSION_PLAN_FILE_SHA256={plan_file_sha},"
            f"PRORM_R3_PRIMARY_SUBMISSION_PLAN_SHA256={semantic}"
        ),
        str(sbatch_script),
    ]
    ledger = root / "runs" / "phase2-recovery-r3" / "submission-ledgers" / semantic
    ledger.mkdir(parents=True)
    intent_body = {
        "schema_version": module._INTENT_SCHEMA,
        "role": "one_exact_r3_array_submission_intent",
        "plan_kind": "primary",
        "plan_path": "plan.json",
        "plan_file_sha256": plan_file_sha,
        "plan_semantic_sha256": semantic,
        "array_task_ids": [0, 1, 2],
        "dependency_array_job_ids": [],
        "attempt_root": "attempt",
        "git_commit": "b" * 40,
        "container_image_file_sha256": "c" * 64,
        "sbatch_script_path": str(sbatch_script),
        "sbatch_script_file_sha256": sbatch_script_sha,
        "sbatch_command": sbatch_command,
        "sbatch_command_sha256": module._semantic_sha256({"argv": sbatch_command}),
        "job_name": "prorm-r3-primary-s1",
        "slurm_comment": f"prorm-r3-{semantic}",
    }
    intent = {
        **intent_body,
        "submission_intent_sha256": module._semantic_sha256(intent_body),
    }
    publish_canonical_artifact(ledger / "intent.json", intent)
    held_inspection = (
        "JobId=1234 JobName=prorm-r3-primary-s1 JobState=PENDING "
        f"Reason=JobHeldUser Priority=0 Comment=prorm-r3-{semantic}"
    )
    submission_body = {
        "schema_version": module._SUBMISSION_SCHEMA,
        "role": "held_array_allocation_bound_to_immutable_intent",
        "submission_intent_sha256": intent["submission_intent_sha256"],
        "array_job_id": "1234",
        "array_task_ids": [0, 1, 2],
        "dependency_array_job_ids": [],
        "held_before_ledger_publication": True,
        "scontrol_show_job": held_inspection,
        "scontrol_show_job_sha256": module._sha256(held_inspection.encode("utf-8")),
    }
    submission = {
        **submission_body,
        "submission_receipt_sha256": module._semantic_sha256(submission_body),
    }
    publish_canonical_artifact(ledger / "submission.json", submission)
    release_observation = "released\n\n"
    release_body = {
        "schema_version": module._RELEASE_SCHEMA,
        "role": "ledgered_r3_array_released_for_execution",
        "submission_receipt_sha256": submission["submission_receipt_sha256"],
        "array_job_id": "1234",
        "released_after_submission_ledger_fsync": True,
        "release_observation": release_observation,
        "release_observation_sha256": module._sha256(release_observation.encode("utf-8")),
    }
    release = {
        **release_body,
        "release_receipt_sha256": module._semantic_sha256(release_body),
    }
    publish_canonical_artifact(ledger / "release.json", release)

    reopened = module.reopen_submission_ledger(
        root,
        plan_semantic_sha256=semantic,
    )
    assert reopened["submission"]["array_job_id"] == "1234"
    with pytest.raises(ValueError, match="plan identity"):
        module._validate_plan_identity(
            plan,
            plan_kind="primary",
            plan_file_sha256=plan_file_sha,
            plan_semantic_sha256="f" * 64,
        )
    plan.write_text('{"changed":true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="file SHA-256"):
        module.reopen_submission_ledger(root, plan_semantic_sha256=semantic)


def test_held_inspection_requires_exact_pending_user_hold() -> None:
    module = _load()
    valid = (
        "JobId=1234 JobName=prorm-r3-primary-s1 JobState=PENDING "
        "Reason=JobHeldUser Priority=0 Comment=prorm-r3-deadbeef"
    )
    module._validate_held_inspection(
        valid,
        job_id="1234",
        job_name="prorm-r3-primary-s1",
        comment="prorm-r3-deadbeef",
    )
    for invalid in (
        valid.replace("JobState=PENDING", "JobState=RUNNING"),
        valid.replace("Reason=JobHeldUser", "Reason=Dependency"),
        valid.replace("Priority=0", "Priority=10"),
        valid.replace("Comment=prorm-r3-deadbeef", "Comment=other"),
    ):
        with pytest.raises(ValueError, match="not verifiably held"):
            module._validate_held_inspection(
                invalid,
                job_id="1234",
                job_name="prorm-r3-primary-s1",
                comment="prorm-r3-deadbeef",
            )


def test_authorization_binds_each_continuation_ledger_to_exact_plan_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_authorization_validator()
    root = tmp_path.resolve()
    monkeypatch.setattr(module, "_PROJECT_ROOT", root)
    base_plan = root / "retained" / "base.json"
    continuation_plan = root / "retained" / "continuation-1.json"
    base_semantic = "a" * 64
    continuation_semantic = "b" * 64
    final_semantic = "c" * 64
    base_file_sha = "d" * 64
    continuation_file_sha = "e" * 64
    commit = "f" * 40
    image_sha = "1" * 64
    base_job = "510001"
    continuation_job = "510002"
    ledgers = {
        base_semantic: {
            "intent": {
                "plan_kind": "primary",
                "plan_path": base_plan.relative_to(root).as_posix(),
                "plan_file_sha256": base_file_sha,
                "array_task_ids": [0, 1, 2],
                "dependency_array_job_ids": [],
                "git_commit": commit,
                "container_image_file_sha256": image_sha,
            },
            "submission": {"array_job_id": base_job},
        },
        continuation_semantic: {
            "intent": {
                "plan_kind": "continuation",
                "plan_path": continuation_plan.relative_to(root).as_posix(),
                "plan_file_sha256": continuation_file_sha,
                "array_task_ids": [0],
                "dependency_array_job_ids": [base_job],
                "git_commit": commit,
                "container_image_file_sha256": image_sha,
            },
            "submission": {"array_job_id": continuation_job},
        },
    }

    class FakeLedgerModule:
        @staticmethod
        def reopen_submission_ledger(
            _project_root: Path,
            *,
            plan_semantic_sha256: str,
        ) -> dict[str, dict[str, object]]:
            return ledgers[plan_semantic_sha256]

    monkeypatch.setattr(module, "_submission_ledger_module", FakeLedgerModule)
    segment_one_routes = [{"history": [{"array_job_id": base_job}]} for _task_id in range(3)]
    successor_routes = [
        {"history": [{"array_job_id": base_job}, {"array_job_id": continuation_job}]},
        {"history": [{"array_job_id": base_job}]},
        {"history": [{"array_job_id": base_job}]},
    ]
    lineage = (
        {
            "base_primary_submission_plan_path": str(base_plan),
            "base_primary_submission_plan_file_sha256": base_file_sha,
            "continuation_plan_sha256": continuation_semantic,
            "continuation_array_required": True,
            "active_array_task_ids": [0],
            "dependency_array_job_ids": [base_job],
            "task_routes": segment_one_routes,
        },
        {
            "previous_continuation_plan_path": str(continuation_plan),
            "previous_continuation_plan_file_sha256": continuation_file_sha,
            "continuation_plan_sha256": final_semantic,
            "all_tasks_complete": True,
            "task_routes": successor_routes,
        },
    )
    base = {
        "submission_plan_sha256": base_semantic,
        "git_commit": commit,
        "container_image_file_sha256": image_sha,
    }

    module._validate_submission_ledger_lineage(base=base, lineage=lineage)
    ledgers[continuation_semantic]["intent"]["plan_path"] = "retained/copy.json"
    with pytest.raises(ValueError, match="sealed plan"):
        module._validate_submission_ledger_lineage(base=base, lineage=lineage)


def test_shell_submitters_use_held_ledger_and_only_active_continuation_tasks() -> None:
    primary = PRIMARY_SUBMIT.read_text(encoding="utf-8")
    continuation = CONTINUATION_SUBMIT.read_text(encoding="utf-8")
    primary_launcher = PRIMARY_LAUNCHER.read_text(encoding="utf-8")
    continuation_launcher = CONTINUATION_LAUNCHER.read_text(encoding="utf-8")
    helper = SCRIPT.read_text(encoding="utf-8")
    assert "submit_phase2_r3_once.py" in primary
    assert "submit_phase2_r3_once.py" in continuation
    assert '--array-task-ids "0,1,2"' in primary
    assert '--array-task-ids "${active_array_task_ids}"' in continuation
    assert '--array="${active_array_task_ids}%${array_concurrency}"' in continuation
    assert '"--hold"' in helper
    assert '["scontrol", "release", job_id]' in helper
    assert '"sbatch_command": command' in helper
    assert '"scontrol_show_job": inspection' in helper
    assert '"release_observation": release_observation' in helper
    for submitter in (primary, continuation):
        assert (
            "/opt/shared/spack/local/linux-rocky9-x86_64_v4/gcc-11.4.1/"
            "miniconda3-24.3.0-quc3pyudmzikgo2r4qsyqpwnrvzpin63/bin/python3.12" in submitter
        )
        assert "9c91f9aa231c61c6bf2eabb9b93ebc5a8269a4126a36125e9548d8853e32da9c" in (submitter)
        assert '== "Python 3.12.2"' in submitter
        assert '"${host_python}" -I -S "${submit_once}"' in submitter
        assert "ambient sbatch overrides are forbidden" in submitter
        assert "--no-requeue" in submitter
        assert 'export_spec="PATH=/usr/bin:/bin"' in submitter
        assert 'export_spec="NONE"' not in submitter
    for launcher, driver_name in (
        (primary_launcher, "phase2_r3_primary_submission.sbatch"),
        (continuation_launcher, "phase2_r3_continuation_submission.sbatch"),
    ):
        assert "apptainer exec" not in launcher
        assert driver_name in launcher
        assert "exec srun" in launcher
        assert "--partition=gpu-l20" in launcher
        assert "--gpus-per-node=1" in launcher
        assert 'export_spec="PATH=/usr/bin:/bin"' in launcher
    assert "from smart_reward" not in helper
    assert "phase2_r3_artifacts.py" in helper


def test_no_r3_login_side_submitter_executes_apptainer() -> None:
    launchers = sorted((ROOT / "scripts" / "hpc4").glob("submit_phase2_r3*.sh"))
    assert launchers
    for launcher in launchers:
        text = launcher.read_text(encoding="utf-8")
        assert "apptainer exec" not in text, launcher
        assert '"${APPTAINER}" exec' not in text, launcher

    for driver in (
        ROOT / "scripts" / "hpc4" / "phase2_r3_gatep_submission.sbatch",
        PRIMARY_SUBMIT,
        CONTINUATION_SUBMIT,
    ):
        assert "apptainer exec" in driver.read_text(encoding="utf-8")
