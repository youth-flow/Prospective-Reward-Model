from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "scripts" / "hpc4" / "validate_phase2_recovery_infrastructure_failure.py"
PRODUCTION_REGISTRY = ROOT / "configs" / "phase2_recovery_infrastructure_failure.json"
DESIGN = "9602b0f00a73880545fd57ce1886ec65d7901385cce2b919fd72f3efec4592d4"
BASE = "81ccbd3bc9d745d3792d1834116ba2480d34a7201d6f4e463a61d8d8ab0baefa"
COMMIT = "734d2a27473f974431b96d5d196f9793e14b2755"
REGISTRY_SHA = "e09eefa403f72044192c58f19e06b1e89b939c1d35ddba5081b13693e995cafd"


def _load_validator() -> ModuleType:
    specification = importlib.util.spec_from_file_location(
        "prorm_recovery_infrastructure_validator", VALIDATOR
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("could not load recovery infrastructure validator")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _materialize_fixture(tmp_path: Path) -> tuple[Path, Path]:
    project_root = tmp_path / "project"
    evidence_root = project_root / "runs" / "phase2-recovery-pilot" / DESIGN
    log_root = project_root / "slurm-logs" / "phase2-recovery-pilot" / DESIGN
    log_root.mkdir(parents=True)
    tasks = []
    for task_id in range(3):
        seed = 20260801 + task_id
        run_dir = evidence_root / f"seed-{seed}" / f"job-1648094_{task_id}"
        run_dir.mkdir(parents=True)
        workload_exit_code = 1 if task_id < 2 else 0
        marker = (
            "schema_version=prorm-phase2-recovery-run-status/v1\n"
            "status=FAILED\n"
            f"workload_exit_code={workload_exit_code}\n"
            "final_exit_code=1\n"
            "array_job_id=1648094\n"
            f"array_task_id={task_id}\n"
            f"seed={seed}\n"
            f"recovery_design_sha256={DESIGN}\n"
            f"base_config_hash={BASE}\n"
            f"recovery_git_commit={COMMIT}\n"
            "one_shot_no_further_adaptation=true\n"
            "created_at_utc=2026-07-25T00:00:00Z\n"
        )
        failed = run_dir / "FAILED"
        failed.write_text(marker, encoding="utf-8", newline="\n")
        preflight = run_dir / "phase2-config-check.json"
        preflight.write_text('{"status":"ok"}\n', encoding="utf-8", newline="\n")
        stdout = log_root / f"prorm-p2-recovery-1648094_{task_id}.out"
        stderr = log_root / f"prorm-p2-recovery-1648094_{task_id}.err"
        stdout.write_text("preflight only\n", encoding="utf-8", newline="\n")
        failure_text = (
            "OSError: [Errno 30] Read-only file system: "
            "'/project/cache/hf-cache/datasets/builder.lock'\n"
            if task_id < 2
            else "slurmstepd: error: *** JOB 1648094 CANCELLED ***\n"
        )
        stderr.write_text(failure_text, encoding="utf-8", newline="\n")
        tasks.append(
            {
                "array_task_id": task_id,
                "failure_class": (
                    "hf_datasets_read_only_runtime_lock"
                    if task_id < 2
                    else "cancelled_before_training_after_sibling_preflight_failure"
                ),
                "files": {
                    "FAILED": _sha(failed),
                    "phase2-config-check.json": _sha(preflight),
                },
                "final_exit_code": 1,
                "seed": seed,
                "stderr_sha256": _sha(stderr),
                "stdout_sha256": _sha(stdout),
                "workload_exit_code": workload_exit_code,
            }
        )
    registry = {
        "execution_revision": 1,
        "next_execution_revision": 2,
        "next_execution_reason": "pretrainer_hf_datasets_runtime_lock",
        "recovery_design_sha256": DESIGN,
        "recovery_git_commit": COMMIT,
        "schema_version": "prorm-phase2-recovery-infrastructure-failure/v1",
        "source_array_job_id": "1648094",
        "tasks": tasks,
        "trainer_entered": False,
    }
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps(registry, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return project_root, registry_path


def test_production_registry_has_frozen_identity_and_pretrainer_contract() -> None:
    payload = json.loads(PRODUCTION_REGISTRY.read_text(encoding="utf-8"))

    assert _sha(PRODUCTION_REGISTRY) == REGISTRY_SHA
    assert payload["trainer_entered"] is False
    assert payload["source_array_job_id"] == "1648094"
    assert payload["next_execution_revision"] == 2
    assert [task["seed"] for task in payload["tasks"]] == [
        20260801,
        20260802,
        20260803,
    ]
    forbidden = _load_validator().FORBIDDEN_TRAINING_FILES
    assert all(not (set(task["files"]) & forbidden) for task in payload["tasks"])


def test_validator_accepts_exact_pretrainer_failure_evidence(tmp_path: Path) -> None:
    validator = _load_validator()
    project_root, registry = _materialize_fixture(tmp_path)

    result = validator.validate(
        registry,
        project_root=project_root,
        expected_registry_sha256=_sha(registry),
    )

    assert result["status"] == "ok"
    assert result["trainer_entered"] is False
    assert result["next_execution_revision"] == 2
    assert len(result["tasks"]) == 3


@pytest.mark.parametrize(
    "mutation",
    ["training_result", "changed_marker", "changed_log", "extra_file"],
)
def test_validator_fails_closed_on_changed_or_trainer_evidence(
    tmp_path: Path,
    mutation: str,
) -> None:
    validator = _load_validator()
    project_root, registry = _materialize_fixture(tmp_path)
    run_dir = (
        project_root / "runs" / "phase2-recovery-pilot" / DESIGN / "seed-20260801" / "job-1648094_0"
    )
    if mutation == "training_result":
        (run_dir / "recovery-result.json").write_text("{}\n", encoding="utf-8")
    elif mutation == "changed_marker":
        with (run_dir / "FAILED").open("a", encoding="utf-8") as stream:
            stream.write("unexpected=true\n")
    elif mutation == "changed_log":
        stderr = (
            project_root
            / "slurm-logs"
            / "phase2-recovery-pilot"
            / DESIGN
            / "prorm-p2-recovery-1648094_0.err"
        )
        with stderr.open("a", encoding="utf-8") as stream:
            stream.write("changed\n")
    else:
        (run_dir / "unexpected").write_text("changed\n", encoding="utf-8")

    with pytest.raises(ValueError):
        validator.validate(
            registry,
            project_root=project_root,
            expected_registry_sha256=_sha(registry),
        )


def test_validator_rejects_intermediate_symlink_in_evidence_path(tmp_path: Path) -> None:
    validator = _load_validator()
    project_root, registry = _materialize_fixture(tmp_path)
    real_runs = project_root / "runs"
    moved_runs = project_root / "real-runs"
    real_runs.rename(moved_runs)
    try:
        real_runs.symlink_to(moved_runs, target_is_directory=True)
    except OSError:
        moved_runs.rename(real_runs)
        pytest.skip("host does not permit directory symlink creation")

    with pytest.raises(ValueError, match="no symlink component"):
        validator.validate(
            registry,
            project_root=project_root,
            expected_registry_sha256=_sha(registry),
        )
