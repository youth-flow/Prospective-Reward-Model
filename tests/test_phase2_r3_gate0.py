from __future__ import annotations

import inspect
import os
import shutil
import stat
import subprocess
import sys
import textwrap
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from smart_reward import phase2_r3_gate0 as gate0

ROOT = Path(__file__).resolve().parents[1]


def _sacct_raw(
    *,
    task0_state: str = "TIMEOUT",
    task1_elapsed: str = "12:00:04",
    task2_state: str = "CANCELLED",
) -> bytes:
    rows = [
        (
            "1648125_0",
            "1648126",
            task0_state,
            "12:00:04",
            "0:15",
            "0:0",
        ),
        (
            "1648125_1",
            "1648203",
            "TIMEOUT",
            task1_elapsed,
            "0:15",
            "0:0",
        ),
        (
            "1648125_2",
            "1648125",
            task2_state,
            "08:58:42",
            "0:15",
            "0:0",
        ),
    ]
    suffix = (
        "hpc4",
        "sigroup",
        "gpu-l20",
        "1",
        "8",
        gate0._REQUESTED_TRES,
        gate0._ALLOCATED_TRES,
    )
    return ("\n".join("|".join((*prefix, *suffix)) for prefix in rows) + "\n").encode("utf-8")


def _failed_raw(*, task: int, seed: int, workload: int = 143, final: int = 143) -> bytes:
    values = {
        "schema_version": "prorm-phase2-recovery-run-status/v1",
        "status": "FAILED",
        "workload_exit_code": str(workload),
        "final_exit_code": str(final),
        "array_job_id": gate0.SOURCE_ARRAY_JOB_ID,
        "array_task_id": str(task),
        "seed": str(seed),
        "execution_revision": str(gate0.R2_EXECUTION_REVISION),
        "retry_reason": "pretrainer_hf_datasets_runtime_lock",
        "recovery_design_sha256": gate0.R2_RECOVERY_DESIGN_SHA256,
        "base_config_hash": gate0.R2_BASE_CONFIG_HASH,
        "recovery_git_commit": gate0.R2_RECOVERY_GIT_COMMIT,
        "parent_design_sha256": gate0.R2_PARENT_DESIGN_SHA256,
        "parent_registry_sha256": gate0.R2_PARENT_REGISTRY_SHA256,
        "parent_producer_git_commit": gate0.R2_PARENT_PRODUCER_GIT_COMMIT,
        "one_shot_no_further_adaptation": "true",
        "created_at_utc": "2026-07-26T03:04:05Z",
    }
    return "".join(f"{key}={value}\n" for key, value in values.items()).encode("utf-8")


def _live_scontrol_raw(*, task2_state: str = "PENDING") -> bytes:
    states = ("RUNNING", "RUNNING", task2_state)
    lines = [
        f"schema={gate0._LIVE_CONTROL_SCHEMA}",
        f"captured_at={gate0._LIVE_CAPTURED_AT}",
        f"command={gate0._LIVE_COMMAND}",
    ]
    for (task, _, raw_job_id, _, _), state in zip(
        gate0.ORDERED_TASKS,
        states,
        strict=True,
    ):
        lines.append(
            " ".join(
                (
                    f"JobId={raw_job_id}",
                    f"ArrayJobId={gate0.SOURCE_ARRAY_JOB_ID}",
                    f"ArrayTaskId={task}",
                    f"JobState={state}",
                    "Account=sigroup",
                    "QOS=l20_qos",
                    "Partition=gpu-l20",
                    "Requeue=0",
                    "Restarts=0",
                    "TimeLimit=12:00:00",
                )
            )
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def test_gate0_sacct_parser_accepts_only_frozen_failure_terminal() -> None:
    rows = gate0._parse_sacct_raw(_sacct_raw())

    assert [row["terminal_state"] for row in rows] == [
        "TIMEOUT",
        "TIMEOUT",
        "CANCELLED",
    ]
    assert [row["elapsed"] for row in rows] == ["12:00:04", "12:00:04", "08:58:42"]
    assert [row["job_id_raw"] for row in rows] == ["1648126", "1648203", "1648125"]


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        (_sacct_raw(task0_state="COMPLETED"), "task 0"),
        (_sacct_raw(task1_elapsed="11:59:59"), "task 1"),
        (_sacct_raw(task2_state="TIMEOUT"), "task 2"),
        (_sacct_raw().replace(b"sigroup", b"other", 1), "task 0"),
        (_sacct_raw().replace(b"gres/gpu:l20=1", b"gres/gpu:a30=1", 1), "task 0"),
    ],
)
def test_gate0_sacct_parser_rejects_success_or_identity_drift(
    replacement: bytes,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        gate0._parse_sacct_raw(replacement)


def test_gate0_sacct_parser_rejects_missing_duplicate_or_extra_rows() -> None:
    raw = _sacct_raw()
    lines = raw.splitlines(keepends=True)
    for invalid in (b"".join(lines[:2]), b"".join((lines[0], lines[0], lines[2])), raw + lines[2]):
        with pytest.raises(ValueError):
            gate0._parse_sacct_raw(invalid)


def test_failed_marker_parser_binds_nonzero_failure_and_r2_identity() -> None:
    projection = gate0._parse_failed_marker(
        _failed_raw(task=1, seed=20260802),
        task=1,
        seed=20260802,
    )

    assert projection["status"] == "FAILED"
    assert projection["workload_exit_code"] == 143
    assert projection["array_task_id"] == 1
    assert projection["recovery_git_commit"] == gate0.R2_RECOVERY_GIT_COMMIT


def test_failed_marker_rejects_zero_exit_or_success_status() -> None:
    with pytest.raises(ValueError, match="failing exit"):
        gate0._parse_failed_marker(
            _failed_raw(task=0, seed=20260801, workload=0, final=0),
            task=0,
            seed=20260801,
        )
    success = _failed_raw(task=0, seed=20260801).replace(
        b"status=FAILED",
        b"status=SUCCESS",
    )
    with pytest.raises(ValueError, match="execution identity"):
        gate0._parse_failed_marker(success, task=0, seed=20260801)


def test_run_inventory_binds_failed_and_explicit_absence(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    failed = _failed_raw(task=0, seed=20260801)
    (run / "FAILED").write_bytes(failed)
    (run / "run-manifest.json").write_text("evidence\n", encoding="utf-8")
    (run / "recovery-train.log").write_bytes(b"")

    result = gate0._inventory_run(run, task=0, seed=20260801)

    assert result["failed_bytes"] == failed
    assert [item["name"] for item in result["entries"]] == [
        "FAILED",
        "recovery-train.log",
        "run-manifest.json",
    ]
    assert result["absence_audit"] == {
        "SUCCESS_present": False,
        "recovery_result_present": False,
        "durable_checkpoint_present": False,
        "checkpoint_candidates": [],
    }


@pytest.mark.parametrize(
    "forbidden",
    ["SUCCESS", "recovery-result.json", "checkpoint-20.pt", "state.pt", "model.ckpt"],
)
def test_run_inventory_rejects_success_result_or_checkpoint(
    tmp_path: Path,
    forbidden: str,
) -> None:
    run = tmp_path / forbidden.replace(".", "-")
    run.mkdir()
    (run / "FAILED").write_bytes(_failed_raw(task=2, seed=20260803))
    (run / forbidden).write_bytes(b"forbidden")

    with pytest.raises(ValueError, match="success/result/checkpoint"):
        gate0._inventory_run(run, task=2, seed=20260803)


def test_run_inventory_rejects_nested_or_special_entries(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "FAILED").write_bytes(_failed_raw(task=0, seed=20260801))
    (run / "nested").mkdir()

    with pytest.raises(ValueError, match="nested/special"):
        gate0._inventory_run(run, task=0, seed=20260801)


def test_live_scontrol_projection_is_submission_only() -> None:
    rows = gate0._parse_live_scontrol(
        _live_scontrol_raw(),
        require_frozen_bytes=False,
    )

    assert [row["state_at_capture"] for row in rows] == ["RUNNING", "RUNNING", "PENDING"]
    assert all(row["terminal_status_authority"] is False for row in rows)
    assert [row["job_id_raw"] for row in rows] == ["1648126", "1648203", "1648125"]


def test_live_scontrol_frozen_digest_cannot_be_replaced_by_local_fixture() -> None:
    with pytest.raises(ValueError, match="bytes changed"):
        gate0._parse_live_scontrol(
            _live_scontrol_raw(),
            require_frozen_bytes=True,
        )


def test_scontrol_config_requires_exact_hpc4_cluster() -> None:
    gate0._parse_scontrol_config(b"ClusterName = hpc4\nSlurmctldHost = login\n")
    with pytest.raises(ValueError, match="not exactly cluster hpc4"):
        gate0._parse_scontrol_config(b"ClusterName = local-test\n")


def test_strict_json_rejects_duplicate_and_noncanonical_bytes() -> None:
    with pytest.raises(ValueError, match="duplicate JSON key"):
        gate0._decode_json(b'{"a":1,"a":2}\n', name="test", require_canonical=True)
    with pytest.raises(ValueError, match="not canonical"):
        gate0._decode_json(b'{ "a": 1 }\n', name="test", require_canonical=True)


def test_artifact_self_hash_is_checked_before_embedded_claims() -> None:
    payload = {
        "schema_version": gate0.GATE0_ARTIFACT_SCHEMA,
        "role": gate0.GATE0_ARTIFACT_ROLE,
        "captured_at_utc": "2026-07-26T03:04:05Z",
        "source_array_job_id": gate0.SOURCE_ARRAY_JOB_ID,
        "r2_execution_identity": {},
        "scheduler_terminal": {},
        "original_live_scontrol": {},
        "capture_environment": {},
        "tasks": [],
        "raw_sources": [],
        "producer": {},
        "container": {},
        "parent_registries": [],
        "failure_parent_policy": {},
        "artifact_sha256": "0" * 64,
    }

    with pytest.raises(ValueError, match="self-hash"):
        gate0._validate_payload(payload)


def test_offline_inspection_type_is_explicitly_non_authorizing() -> None:
    report = gate0.R3Gate0Inspection(
        schema_version=gate0.GATE0_ARTIFACT_SCHEMA,
        artifact_sha256="a" * 64,
        file_sha256="b" * 64,
        scheduler_rows=tuple(gate0._parse_sacct_raw(_sacct_raw())),
    )

    assert report.formal_authorization is False
    with pytest.raises(FrozenInstanceError):
        report.formal_authorization = True  # type: ignore[misc]


def test_gate0_capability_cannot_be_caller_constructed() -> None:
    with pytest.raises(TypeError, match="live HPC4 verification"):
        gate0.R3Gate0Capability(
            schema_version=gate0.GATE0_ARTIFACT_SCHEMA,
            role=gate0.GATE0_ARTIFACT_ROLE,
            artifact_sha256="a" * 64,
            file_sha256="b" * 64,
            production_relative=gate0._GATE0_RELATIVE.as_posix(),
        )


def test_gate0_capability_seal_is_not_inherited_by_dataclass_replace() -> None:
    capability = gate0.R3Gate0Capability(
        schema_version=gate0.GATE0_ARTIFACT_SCHEMA,
        role=gate0.GATE0_ARTIFACT_ROLE,
        artifact_sha256="a" * 64,
        file_sha256="b" * 64,
        production_relative=gate0._GATE0_RELATIVE.as_posix(),
        _factory_token=gate0._FACTORY_TOKEN,
    )

    with pytest.raises(TypeError, match="live HPC4 verification"):
        replace(capability, artifact_sha256="c" * 64)


def test_formal_entrypoints_have_no_root_or_output_override() -> None:
    capture_parameters = set(inspect.signature(gate0.capture_live_r3_gate0_bundle).parameters)
    verify_parameters = set(inspect.signature(gate0.verify_live_r3_gate0_bundle).parameters)
    inside_parameters = set(inspect.signature(gate0.verify_live_r3_gate0_in_container).parameters)

    assert capture_parameters == {"container", "now"}
    assert verify_parameters == {"container"}
    assert inside_parameters == {"expected_file_sha256"}
    script = (ROOT / "scripts" / "hpc4" / "capture_phase2_r3_gate0.py").read_text(encoding="utf-8")
    assert "--project-root" not in script
    assert "--output" not in script
    gatep_runner = (ROOT / "scripts" / "hpc4" / "run_phase2_r3_gatep.py").read_text(
        encoding="utf-8"
    )
    gatep_sbatch = (ROOT / "scripts" / "hpc4" / "phase2_r3_gatep.sbatch").read_text(
        encoding="utf-8"
    )
    gatep_launcher = (ROOT / "scripts" / "hpc4" / "submit_phase2_r3_gatep.sh").read_text(
        encoding="utf-8"
    )
    gatep_submitter = (ROOT / "scripts" / "hpc4" / "phase2_r3_gatep_submission.sbatch").read_text(
        encoding="utf-8"
    )
    gatep_environment = (ROOT / "scripts" / "hpc4" / "r3_env.example").read_text(encoding="utf-8")
    assert "verify_live_r3_gate0_in_container" in gatep_runner
    assert "verify_live_r3_gate0_bundle" not in gatep_runner
    assert "--gate0-file-sha256" in gatep_runner
    assert "--container" not in gatep_runner
    assert "PRORM_R3_GATE0_FILE_SHA256" in gatep_sbatch
    assert '--gate0-file-sha256 "${PRORM_R3_GATE0_FILE_SHA256}"' in gatep_sbatch
    assert "PRORM_R3_GATE0_FILE_SHA256" in gatep_submitter
    assert "apptainer exec" not in gatep_launcher
    assert "phase2_r3_gatep_submission.sbatch" in gatep_launcher
    assert "exec srun" in gatep_launcher
    assert "--partition=gpu-l20" in gatep_launcher
    assert "--gpus-per-node=1" in gatep_launcher
    assert 'export_spec="PATH=/usr/bin:/bin"' in gatep_launcher
    assert 'Path("/home/yyangjo/Smart-Reward-Model")' in gatep_runner
    assert 'Path("/project/sigroup/smart-reward-model")' in gatep_runner
    assert 'parser.add_argument("--project-root"' not in gatep_runner
    assert "source_test_receipt=arguments." not in gatep_runner
    for script in (gatep_sbatch, gatep_submitter):
        assert "PRORM_R3_REPO_ROOT" in script
        assert '"/home/yyangjo/Smart-Reward-Model"' in script
        assert '"/project/sigroup/smart-reward-model"' in script
        assert 'git -C "${repo_root}"' in script
        assert '"${repo_root}:${repo_root}:ro"' in script
        assert '"${project_root}:${project_root}:rw"' in script
        assert '"/project/sigroup:/project/sigroup:rw"' not in script
        assert "\n  PRORM_R3_SOURCE_TEST_RECEIPT\n" not in script
        assert (
            "${project_root}/runs/phase2-recovery-r3/gate1/r3-source-test-receipt.json"
        ) in script
    assert 'runner="${repo_root}/scripts/hpc4/run_phase2_r3_gatep.py"' in gatep_sbatch
    assert 'terminal_cli="${repo_root}/scripts/hpc4/' in gatep_submitter
    assert '"${repo_root}/scripts/hpc4/phase2_r3_gatep.sbatch"' in gatep_submitter
    assert 'ln -- "${input_temp}" "${destination}"' in gatep_submitter
    assert "retained input copy must have mode 0440" in gatep_submitter
    assert "retained input SHA-256 differs from clean repository bytes" in (gatep_submitter)
    assert "retained input copy differs from clean repository bytes" in gatep_submitter
    assert "runs/phase2-recovery-r3/inputs/${PRORM_R3_GIT_COMMIT}" in gatep_sbatch
    assert "retained clean-commit copy" in gatep_runner
    assert "--nodes=1" in gatep_submitter
    assert "--ntasks=1" in gatep_submitter
    assert "--gpus-per-node=1" in gatep_submitter
    assert 'export_spec="PATH=/usr/bin:/bin"' in gatep_submitter
    assert 'export_spec="NONE"' not in gatep_submitter
    assert '--export="${export_spec}"' in gatep_submitter
    assert "contains a comma or newline" in gatep_submitter
    assert "--no-requeue" in gatep_submitter
    assert "    --gpus-per-task=1 \\" not in gatep_submitter
    assert '[[ "${SLURM_NNODES:-}" == "1" ]]' in gatep_sbatch
    assert '[[ "${SLURM_NTASKS:-}" == "1" ]]' in gatep_sbatch
    assert '[[ -n "${CUDA_VISIBLE_DEVICES:-}"' in gatep_sbatch
    assert gatep_runner.count("required_mode=0o440") == 2
    assert (
        "${PRORM_R3_PROJECT_ROOT}/runs/phase2-recovery-r3/inputs/${PRORM_R3_INPUT_GIT_COMMIT}"
    ) in gatep_environment
    assert "${PRORM_R3_INPUT_ROOT}/common_beta_recovery_pilot.yaml" in (gatep_environment)
    assert "${PRORM_R3_INPUT_ROOT}/phase2_recovery_parent_failures.json" in (gatep_environment)
    assert '    --project-root "${project_root}" \\' not in gatep_sbatch
    assert '    --source-test-receipt "${PRORM_R3_SOURCE_TEST_RECEIPT}" \\' not in (gatep_sbatch)


@pytest.mark.parametrize(
    "relative",
    [
        Path("scripts/hpc4/capture_phase2_r3_gate0.py"),
        Path("scripts/hpc4/capture_phase2_r3_gate1.py"),
    ],
)
def test_capture_cli_help_bypasses_package_init_and_poisoned_pythonpath(
    tmp_path: Path,
    relative: Path,
) -> None:
    poison = tmp_path / "poison"
    poison_package = poison / "smart_reward"
    poison_package.mkdir(parents=True)
    (poison_package / "__init__.py").write_text(
        "raise RuntimeError('poisoned smart_reward package imported')\n",
        encoding="utf-8",
    )
    launcher = textwrap.dedent(
        """
        import builtins
        import runpy
        import sys

        real_import = builtins.__import__

        def blocked(name, *args, **kwargs):
            if name == "torch" or name.startswith("torch."):
                raise RuntimeError("torch import is forbidden for capture CLI startup")
            if name == "smart_reward" or name.startswith("smart_reward."):
                raise RuntimeError("smart_reward package import is forbidden")
            return real_import(name, *args, **kwargs)

        builtins.__import__ = blocked
        script = sys.argv[1]
        sys.argv = [script, "--help"]
        runpy.run_path(script, run_name="__main__")
        """
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.fspath(poison)
    environment["PYTHONNOUSERSITE"] = "1"
    completed = subprocess.run(
        (sys.executable, "-S", "-c", launcher, os.fspath(ROOT / relative)),
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout
    assert "poisoned smart_reward" not in completed.stderr

    copied = tmp_path / "copied" / relative
    copied.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / relative, copied)
    rejected = subprocess.run(
        (sys.executable, os.fspath(copied), "--help"),
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert rejected.returncode != 0
    assert "real Git checkout" in rejected.stderr


def test_gate0_production_dual_roots_are_fixed_disjoint_and_not_swappable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = Path("/home/yyangjo/Smart-Reward-Model")
    project = Path("/project/sigroup/smart-reward-model")
    assert repo == gate0.PRODUCTION_REPO_ROOT
    assert project == gate0.PRODUCTION_PROJECT_ROOT
    assert repo != project and repo not in project.parents and project not in repo.parents

    with monkeypatch.context() as patch:
        patch.setattr(gate0, "PRODUCTION_REPO_ROOT", project)
        patch.setattr(gate0, "PRODUCTION_PROJECT_ROOT", repo)
        with pytest.raises(RuntimeError, match="fixed HPC4 path"):
            gate0._assert_production_roots()

    with monkeypatch.context() as patch:
        patch.setattr(gate0, "PRODUCTION_PROJECT_ROOT", repo / "persistent")
        with pytest.raises(RuntimeError, match="fixed HPC4 path"):
            gate0._assert_production_roots()


@pytest.mark.skipif(os.name != "posix", reason="requires real POSIX directory modes")
def test_gate_output_namespaces_are_multilevel_owner_writable_0750(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from smart_reward import phase2_r3_gate1 as gate1

    project = (tmp_path / "project").resolve()
    project.mkdir(mode=0o750)
    os.chmod(project, 0o2750)
    retained_runs = project / "runs"
    retained_runs.mkdir(mode=0o755)
    os.chmod(retained_runs, 0o2755)
    retained_r3 = retained_runs / "phase2-recovery-r3"
    retained_r3.mkdir(mode=0o550)
    os.chmod(retained_r3, 0o550)
    monkeypatch.setattr(gate0, "PRODUCTION_PROJECT_ROOT", project)
    monkeypatch.setattr(gate1, "PRODUCTION_PROJECT_ROOT", project)

    with pytest.raises(ValueError, match="mode 0750"):
        gate0._ensure_output_parent()

    os.chmod(retained_r3, 0o2750)
    gate0_parent = gate0._ensure_output_parent()
    gate1_parent = gate1._ensure_output_parent()
    source_receipt_parent = gate1._ensure_output_parent(
        gate1._SOURCE_TEST_RECEIPT_RELATIVE,
        namespace_name="source-test receipt",
    )
    assert gate0_parent == project / "runs/phase2-recovery-r3/gate0"
    assert gate1_parent == project / "runs/phase2-recovery-r3/gate1"
    assert source_receipt_parent == gate1_parent

    expected_modes = {
        Path("runs"): 0o2755,
        Path("runs/phase2-recovery-r3"): 0o2750,
        Path("runs/phase2-recovery-r3/gate0"): 0o2750,
        Path("runs/phase2-recovery-r3/gate1"): 0o2750,
    }
    for relative, expected_mode in expected_modes.items():
        directory = project / relative
        assert stat.S_IMODE(directory.stat().st_mode) == expected_mode
        assert os.access(directory, os.W_OK | os.X_OK)


def test_local_machine_cannot_issue_live_hpc4_capability() -> None:
    if os.name == "posix" and gate0.PRODUCTION_PROJECT_ROOT.exists():
        pytest.skip("this assertion is specifically for a non-production local test host")
    with pytest.raises((RuntimeError, FileNotFoundError, ValueError)):
        gate0.verify_live_r3_gate0_bundle(container="not-a-live-r2-container.sif")
    with pytest.raises((RuntimeError, FileNotFoundError, ValueError)):
        gate0.verify_live_r3_gate0_in_container(expected_file_sha256="0" * 64)


def _immutable_revalidation_payload() -> dict[str, object]:
    raw_sources = [
        {
            "name": "scheduler-sacct-X.psv",
            "bytes_base64": gate0.base64.b64encode(_sacct_raw()).decode("ascii"),
        },
        {
            "name": "original-live-scontrol.txt",
            "bytes_base64": gate0.base64.b64encode(_live_scontrol_raw()).decode("ascii"),
        },
    ]
    raw_sources.extend(
        {
            "name": name,
            "bytes_base64": gate0.base64.b64encode(name.encode()).decode("ascii"),
        }
        for task, *_ in gate0.ORDERED_TASKS
        for name in (
            f"task-{task}-FAILED",
            f"task-{task}-slurm.out",
            f"task-{task}-slurm.err",
        )
    )
    tasks = [
        {
            "inventory": [{"name": f"task-{task}"}],
            "failed_marker": {"task": task},
            "absence_audit": {"task": task, "closed": True},
        }
        for task, *_ in gate0.ORDERED_TASKS
    ]
    return {
        "producer": {"commit": "clean"},
        "parent_registries": [{"registry": "frozen"}],
        "container": {"sha256": gate0.R2_IMAGE_SHA256},
        "raw_sources": raw_sources,
        "tasks": tasks,
    }


def test_in_container_immutable_revalidation_never_queries_live_slurm(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = _immutable_revalidation_payload()
    compared: list[str] = []
    frozen_live_control_checks: list[bool] = []

    monkeypatch.setattr(
        gate0,
        "_run_command",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("in-container revalidation queried a live command")
        ),
    )
    monkeypatch.setattr(gate0, "_capture_source_identity", lambda: payload["producer"])
    monkeypatch.setattr(
        gate0,
        "_parent_registry_records",
        lambda: payload["parent_registries"],
    )
    monkeypatch.setattr(gate0, "_container_record", lambda _path: payload["container"])
    monkeypatch.setattr(
        gate0,
        "_parse_live_scontrol",
        lambda _raw, *, require_frozen_bytes: (
            frozen_live_control_checks.append(require_frozen_bytes) or []
        ),
    )
    monkeypatch.setattr(
        gate0,
        "_production_path",
        lambda *_args, **_kwargs: tmp_path / "retained",
    )
    monkeypatch.setattr(
        gate0,
        "_run_path",
        lambda task, seed: tmp_path / f"run-{task}-{seed}",
    )
    monkeypatch.setattr(
        gate0,
        "_log_path",
        lambda task, suffix: tmp_path / f"log-{task}.{suffix}",
    )
    monkeypatch.setattr(
        gate0,
        "_inventory_run",
        lambda _path, *, task, seed: {
            "entries": payload["tasks"][task]["inventory"],
            "failed_projection": payload["tasks"][task]["failed_marker"],
            "absence_audit": payload["tasks"][task]["absence_audit"],
            "seed": seed,
        },
    )
    monkeypatch.setattr(
        gate0,
        "_compare_file_to_raw_record",
        lambda _path, _record, *, name: compared.append(name),
    )

    gate0._revalidate_immutable_sources(payload, container=tmp_path / "image.sif")

    assert len(compared) == 10
    assert frozen_live_control_checks == [True]
    assert any("live-scontrol" in name for name in compared)
    assert any("Slurm err" in name for name in compared)


def test_in_container_frozen_scheduler_tamper_fails_without_live_query() -> None:
    payload = _immutable_revalidation_payload()
    payload["raw_sources"][0]["bytes_base64"] = gate0.base64.b64encode(
        b"forged scheduler row\n"
    ).decode("ascii")

    with pytest.raises(ValueError):
        gate0._revalidate_frozen_scheduler_bytes(payload)


def test_in_container_closure_pins_file_sha_and_issues_sealed_capability(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    raw = b"canonical Gate-0 fixture\n"
    payload = {"closed": True}
    artifact_path = tmp_path / "gate0.json"
    observed: list[Path] = []
    monkeypatch.setattr(
        gate0,
        "_production_path",
        lambda *_args, **_kwargs: artifact_path,
    )
    monkeypatch.setattr(
        gate0,
        "_stable_file",
        lambda *_args, **_kwargs: (raw, {"mode_octal": "0440"}),
    )
    monkeypatch.setattr(gate0, "_decode_json", lambda *_args, **_kwargs: payload)
    monkeypatch.setattr(
        gate0,
        "_validate_payload",
        lambda value: ("a" * 64, ()) if value is payload else AssertionError(),
    )
    monkeypatch.setattr(
        gate0,
        "_revalidate_immutable_sources",
        lambda value, *, container: (
            observed.append(container)
            if value is payload
            else (_ for _ in ()).throw(AssertionError("wrong payload"))
        ),
    )
    container = tmp_path / "image.sif"

    capability = gate0._verify_r3_gate0_in_container_closure(
        expected_file_sha256=gate0._sha256(raw),
        container=container,
    )

    capability.validate_integrity()
    assert capability.file_sha256 == gate0._sha256(raw)
    assert observed == [container]
    with pytest.raises(ValueError, match="caller expectation"):
        gate0._verify_r3_gate0_in_container_closure(
            expected_file_sha256="0" * 64,
            container=container,
        )


def test_old_terminal_capture_remains_success_only_and_is_not_gate0() -> None:
    old = (ROOT / "src" / "smart_reward" / "phase2_recovery_aggregate.py").read_text(
        encoding="utf-8"
    )
    new = (ROOT / "src" / "smart_reward" / "phase2_r3_gate0.py").read_text(encoding="utf-8")

    assert '"COMPLETED"' in old
    assert '"0:0"' in old
    assert "Elapsed" in new
    assert '"TIMEOUT"' in new
    assert '"CANCELLED"' in new
    assert gate0.GATE0_ARTIFACT_SCHEMA not in old


def test_all_bound_r2_source_blobs_exist_at_execution_commit() -> None:
    records = [
        gate0._git_blob_record(gate0.R2_RECOVERY_GIT_COMMIT, relative)
        for relative in gate0._R2_SOURCE_PATHS
    ]

    assert [record["repository_relative"] for record in records] == [
        path.as_posix() for path in gate0._R2_SOURCE_PATHS
    ]
    assert all(record["git_commit"] == gate0.R2_RECOVERY_GIT_COMMIT for record in records)


def test_parent_registry_bytes_match_r2_execution_blobs() -> None:
    records = gate0._parent_registry_records()

    assert {(record["repository_relative"], record["sha256"]) for record in records} == {
        (path.as_posix(), digest) for path, digest in gate0._R2_PARENT_REGISTRIES
    }


def test_digest_reader_does_not_require_retaining_file_bytes(tmp_path: Path) -> None:
    path = tmp_path / "container.sif"
    raw = b"0123456789" * 10_000
    path.write_bytes(raw)

    digest, size, record = gate0._stable_file_digest(
        path,
        name="test image",
        maximum_bytes=len(raw),
    )

    assert digest == gate0._sha256(raw)
    assert size == len(raw)
    assert record["size_bytes"] == len(raw)
