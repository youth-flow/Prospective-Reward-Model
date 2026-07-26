from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from smart_reward import phase2_r3_controls_hpc4 as hpc
from smart_reward.phase2_r3_artifacts import (
    publish_canonical_artifact,
    read_canonical_artifact,
)
from smart_reward.phase2_r3_controls import load_r3_controls_config

FAMILIES = (
    "exact_margin_prorm_plus",
    "exact_soft_label_bt",
    "low_dimensional_prorm_plus",
)
SEEDS = (20260801, 20260802, 20260803)
HEX = "a" * 64
ROOT = Path(__file__).resolve().parents[1]
SUBMIT = ROOT / "scripts" / "hpc4" / "submit_phase2_r3_controls.sh"
SBATCH = ROOT / "scripts" / "hpc4" / "phase2_r3_controls.sbatch"
PROFILE_RUNNER = ROOT / "scripts" / "hpc4" / "run_phase2_r3_control_profile.py"
PROFILE_SUBMIT = ROOT / "scripts" / "hpc4" / "submit_phase2_r3_controls_profile.sh"
PROFILE_SBATCH = ROOT / "scripts" / "hpc4" / "phase2_r3_controls_profile.sbatch"
FORMAL_RUNNER = ROOT / "scripts" / "hpc4" / "run_phase2_r3_control_family.py"
EVIDENCE_CLI = ROOT / "scripts" / "hpc4" / "phase2_r3_controls_evidence.py"
STDLIB_INSPECTOR = ROOT / "scripts" / "hpc4" / "inspect_phase2_r3_controls_plan_stdlib.py"
PROFILE_FINALIZE_SUBMIT = (
    ROOT / "scripts" / "hpc4" / "submit_phase2_r3_controls_profile_finalize.sh"
)
PROFILE_FINALIZE_SBATCH = ROOT / "scripts" / "hpc4" / "phase2_r3_controls_profile_finalize.sbatch"
FORMAL_FINALIZE_SUBMIT = ROOT / "scripts" / "hpc4" / "submit_phase2_r3_controls_finalize.sh"
FORMAL_FINALIZE_SBATCH = ROOT / "scripts" / "hpc4" / "phase2_r3_controls_finalize.sbatch"


class FakeConfig:
    file_sha256 = "1" * 64
    semantic_sha256 = "2" * 64
    maximum_updates = 200
    audit_interval_updates = 20


def _result(family: str, seed: int) -> dict[str, object]:
    body = {
        "schema_version": "phase2-recovery-r3-control-family-result/v1",
        "family": family,
        "seed": seed,
        "controls_config_semantic_sha256": FakeConfig.semantic_sha256,
        "controls_config_file_sha256": FakeConfig.file_sha256,
        "information_boundary": {
            "train_only": True,
            "primary_head_accessed": False,
            "heldout_accessed": False,
            "policy_accessed": False,
            "beta_accessed": False,
        },
        "completion": {
            "status": "completed",
            "completed_updates": 160,
            "stop_reason": "sustained_first_order_gate",
            "formal_family_result": True,
            "profile_only": False,
            "head_or_optimizer_state_retained": False,
        },
        "family_evidence": {"passed": True},
    }
    return {**body, "result_sha256": hpc._semantic_sha256(body)}


@pytest.fixture(autouse=True)
def fake_science(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = SimpleNamespace(
        R3_GATE_C_FAMILIES=FAMILIES,
        R3_GATE_C_SEEDS=SEEDS,
        validate_r3_control_family_result=lambda value, config: deepcopy(value),
    )
    monkeypatch.setattr(hpc, "_core_module", lambda: fake)


def _measurement(family: str, *, task: int) -> dict[str, object]:
    raw_sha = hashlib.sha256(f"profile-{family}".encode()).hexdigest()
    return hpc.build_profile_family_measurement(
        family=family,
        seed=SEEDS[0],
        git_commit="3" * 40,
        container_sha256="4" * 64,
        controls_config_file_sha256=FakeConfig.file_sha256,
        controls_config_semantic_sha256=FakeConfig.semantic_sha256,
        input_training_sha256=hashlib.sha256(f"input-{family}".encode()).hexdigest(),
        oracle_reward_sha256=hashlib.sha256(f"oracle-{family}".encode()).hexdigest(),
        setup_wall_seconds=2.0,
        training_wall_seconds=100.0,
        audit_wall_seconds=5.0,
        checkpoint_roundtrip_wall_seconds=2.0,
        peak_gpu_memory_bytes=1024**3,
        gpu_total_memory_bytes=hpc.R3_CONTROLS_PROFILE_GPU_MEMORY_CAPACITY_BYTES,
        scheduler_terminal={
            "array_job_id": "410000",
            "array_task_id": task,
            "job_id": f"410000_{task}",
            "job_id_raw": str(410100 + task),
            "raw_sacct_sha256": raw_sha,
            "elapsed_seconds": 120,
        },
    )


def _profile(
    measurements: list[dict[str, object]] | None = None,
    *,
    checkpoint_cadence_updates: int = 200,
) -> dict[str, object]:
    return hpc.build_controls_operational_profile(
        (
            measurements
            if measurements is not None
            else [_measurement(family, task=index) for index, family in enumerate(FAMILIES)]
        ),
        controls_config=FakeConfig(),
        optimizer_schedule_sha256=hpc.R3_OPTIMIZER_SCHEDULE_SHA256,
        checkpoint_cadence_updates=checkpoint_cadence_updates,
        walltime_safety_margin_fraction=0.25,
        fixed_walltime_margin_seconds=30.0,
        memory_safety_margin_fraction=0.25,
        cluster="hpc4",
        account="sigroup",
        partition="gpu-l20",
        gpu_name="NVIDIA L20",
        cpus_per_task=hpc.R3_CONTROLS_PROFILE_CPUS_PER_TASK,
        memory_bytes=hpc.R3_CONTROLS_PROFILE_MEMORY_BYTES,
        array_concurrency=1,
        requested_walltime_seconds_per_segment=3600,
        signal_lead_seconds=300,
        max_scheduler_segments=1,
    )


def _raw(
    *,
    job_id: str,
    job_id_raw: str,
    state: str = "COMPLETED",
    cpus: int = hpc.R3_CONTROLS_PROFILE_CPUS_PER_TASK,
    memory: str = "96G",
) -> bytes:
    req = f"billing={cpus},cpu={cpus},gres/gpu=1,mem={memory},node=1"
    alloc = f"{req},gres/gpu:l20=1"
    return (
        "|".join(
            (
                job_id,
                job_id_raw,
                state,
                "0:0",
                "0:0",
                "hpc4",
                "sigroup",
                "gpu-l20",
                "l20_qos",
                "1",
                str(cpus),
                req,
                alloc,
                "120",
            )
        )
        + "\n"
    ).encode()


def _entries(
    plan: dict[str, object],
    profile: dict[str, object],
) -> list[tuple[dict[str, object], dict[str, object], dict[str, object], bytes]]:
    entries = []
    tasks = plan["tasks"]
    assert isinstance(tasks, list)
    for task in tasks:
        assert isinstance(task, dict)
        result = _result(str(task["family"]), int(task["seed"]))
        closure = hpc.build_controls_task_closure(
            plan,
            profile=profile,
            controls_config=FakeConfig(),
            task_id=int(task["task_id"]),
            segment_index=1,
            family_result=result,
        )
        array_job_id = str(500000 + int(task["family_index"]))
        raw_job_id = str(600000 + int(task["task_id"]))
        raw = _raw(
            job_id=f"{array_job_id}_{task['array_task_id']}",
            job_id_raw=raw_job_id,
        )
        terminal = hpc.build_controls_task_terminal(
            raw,
            expected_raw_sacct_sha256=hashlib.sha256(raw).hexdigest(),
            plan=plan,
            profile=profile,
            controls_config=FakeConfig(),
            closure=closure,
            family_result=result,
            array_job_id=array_job_id,
            job_id_raw=raw_job_id,
        )
        entries.append((result, closure, terminal, raw))
    return entries


def test_profile_is_exact_nonreusable_three_family_and_enforces_two_day_ceiling() -> None:
    profile = _profile()
    assert len(profile["measurements"]) == 3
    assert profile["resource_plan"]["requested_walltime_seconds_per_segment"] == 3600
    assert profile["resource_plan"]["max_scheduler_segments"] == 1
    assert hpc.R3_CONTROLS_L20_PHYSICAL_GPU_MEMORY_BYTES == 46_068 * 1024**2
    assert hpc.R3_CONTROLS_PROFILE_TORCH_VISIBLE_GPU_MEMORY_BYTES == 47_676_129_280
    assert hpc.R3_CONTROLS_PROFILE_GPU_MEMORY_CAPACITY_BYTES == 47_676_129_280
    assert profile["resource_plan"]["observed_gpu_memory_capacity_bytes"] == 47_676_129_280
    assert profile["resource_plan"]["checkpoint_cadence_updates"] == 200
    with pytest.raises(ValueError, match="frozen 200-update policy"):
        _profile(checkpoint_cadence_updates=40)

    wrong_host_memory = deepcopy(profile)
    wrong_resource = wrong_host_memory["resource_plan"]
    wrong_resource["memory_bytes"] = 4 * 1024**3
    unsigned_resource = dict(wrong_resource)
    unsigned_resource.pop("resource_plan_sha256")
    wrong_resource["resource_plan_sha256"] = hpc._semantic_sha256(unsigned_resource)
    unsigned_profile = dict(wrong_host_memory)
    unsigned_profile.pop("profile_sha256")
    wrong_host_memory["profile_sha256"] = hpc._semantic_sha256(unsigned_profile)
    with pytest.raises(ValueError, match="fixed profiled host allocation"):
        hpc.validate_controls_operational_profile(
            wrong_host_memory,
            controls_config=FakeConfig(),
        )

    oversized_gpu = [_measurement(family, task=index) for index, family in enumerate(FAMILIES)]
    oversized_gpu[0]["peak_gpu_memory_bytes"] = 47 * 1024**3
    unsigned_measurement = dict(oversized_gpu[0])
    unsigned_measurement.pop("measurement_sha256")
    oversized_gpu[0]["measurement_sha256"] = hpc._semantic_sha256(unsigned_measurement)
    with pytest.raises(ValueError, match="exceeds one NVIDIA L20"):
        _profile(oversized_gpu)

    with pytest.raises(ValueError, match="each family once"):
        hpc.build_controls_operational_profile(
            [_measurement(FAMILIES[0], task=0), _measurement(FAMILIES[1], task=1)],
            controls_config=FakeConfig(),
            optimizer_schedule_sha256="5" * 64,
            checkpoint_cadence_updates=200,
            walltime_safety_margin_fraction=0.25,
            fixed_walltime_margin_seconds=30,
            memory_safety_margin_fraction=0.25,
            cluster="hpc4",
            account="sigroup",
            partition="gpu-l20",
            gpu_name="NVIDIA L20",
            cpus_per_task=hpc.R3_CONTROLS_PROFILE_CPUS_PER_TASK,
            memory_bytes=hpc.R3_CONTROLS_PROFILE_MEMORY_BYTES,
            array_concurrency=1,
            requested_walltime_seconds_per_segment=172801,
            signal_lead_seconds=300,
            max_scheduler_segments=1,
        )
    bad = deepcopy(profile)
    bad["resource_plan"]["requested_walltime_seconds_per_segment"] = 172801
    resource = bad["resource_plan"]
    unsigned_resource = dict(resource)
    unsigned_resource.pop("resource_plan_sha256")
    resource["resource_plan_sha256"] = hpc._semantic_sha256(unsigned_resource)
    unsigned = dict(bad)
    unsigned.pop("profile_sha256")
    bad["profile_sha256"] = hpc._semantic_sha256(unsigned)
    with pytest.raises(ValueError, match="two days"):
        hpc.validate_controls_operational_profile(bad, controls_config=FakeConfig())


def test_profile_compute_receipt_terminalizes_only_after_exact_fixed_hpc4_row() -> None:
    family = FAMILIES[1]
    receipt = hpc.build_profile_compute_receipt(
        family=family,
        seed=SEEDS[0],
        git_commit="3" * 40,
        container_sha256="4" * 64,
        controls_config_file_sha256=FakeConfig.file_sha256,
        controls_config_semantic_sha256=FakeConfig.semantic_sha256,
        input_training_sha256="5" * 64,
        oracle_reward_sha256="6" * 64,
        setup_wall_seconds=2.0,
        training_wall_seconds=10.0,
        audit_wall_seconds=1.0,
        checkpoint_roundtrip_wall_seconds=0.5,
        peak_gpu_memory_bytes=1024**3,
        gpu_total_memory_bytes=hpc.R3_CONTROLS_PROFILE_GPU_MEMORY_CAPACITY_BYTES,
    )
    assert receipt["completed_updates"] == 100
    assert receipt["result_reusable_for_training"] is False
    assert "scheduler_terminal" not in receipt
    with pytest.raises(ValueError, match="Torch L20 capacity"):
        hpc.build_profile_compute_receipt(
            family=family,
            seed=SEEDS[0],
            git_commit="3" * 40,
            container_sha256="4" * 64,
            controls_config_file_sha256=FakeConfig.file_sha256,
            controls_config_semantic_sha256=FakeConfig.semantic_sha256,
            input_training_sha256="5" * 64,
            oracle_reward_sha256="6" * 64,
            setup_wall_seconds=2.0,
            training_wall_seconds=10.0,
            audit_wall_seconds=1.0,
            checkpoint_roundtrip_wall_seconds=0.5,
            peak_gpu_memory_bytes=1024**3,
            gpu_total_memory_bytes=hpc.R3_CONTROLS_L20_PHYSICAL_GPU_MEMORY_BYTES,
        )

    raw = _raw(
        job_id="410000_1",
        job_id_raw="410101",
        cpus=hpc.R3_CONTROLS_PROFILE_CPUS_PER_TASK,
        memory="96G",
    )
    terminal = hpc.build_profile_scheduler_terminal(
        raw,
        expected_raw_sacct_sha256=hashlib.sha256(raw).hexdigest(),
        family=family,
        array_job_id="410000",
        job_id_raw="410101",
    )
    measurement = hpc.build_profile_family_measurement_from_compute_receipt(
        receipt,
        scheduler_terminal=terminal,
    )
    assert measurement["family"] == family
    assert measurement["scheduler_terminal"]["array_task_id"] == 1
    assert measurement["result_reusable_for_training"] is False

    wrong_cpu = _raw(
        job_id="410000_1",
        job_id_raw="410101",
        cpus=4,
        memory="96G",
    )
    with pytest.raises(ValueError, match="allocation identity"):
        hpc.build_profile_scheduler_terminal(
            wrong_cpu,
            expected_raw_sacct_sha256=hashlib.sha256(wrong_cpu).hexdigest(),
            family=family,
            array_job_id="410000",
            job_id_raw="410101",
        )

    reusable = deepcopy(receipt)
    reusable["result_reusable_for_training"] = True
    unsigned = dict(reusable)
    unsigned.pop("compute_receipt_sha256")
    reusable["compute_receipt_sha256"] = hpc._semantic_sha256(unsigned)
    with pytest.raises(ValueError, match="non-reusable role"):
        hpc.validate_profile_compute_receipt(reusable)


def test_profile_measurement_cli_closes_compute_with_external_sacct(
    tmp_path: Path,
) -> None:
    receipt = hpc.build_profile_compute_receipt(
        family=FAMILIES[0],
        seed=SEEDS[0],
        git_commit="3" * 40,
        container_sha256="4" * 64,
        controls_config_file_sha256=FakeConfig.file_sha256,
        controls_config_semantic_sha256=FakeConfig.semantic_sha256,
        input_training_sha256="5" * 64,
        oracle_reward_sha256="6" * 64,
        setup_wall_seconds=2.0,
        training_wall_seconds=10.0,
        audit_wall_seconds=1.0,
        checkpoint_roundtrip_wall_seconds=0.5,
        peak_gpu_memory_bytes=1024**3,
        gpu_total_memory_bytes=hpc.R3_CONTROLS_PROFILE_GPU_MEMORY_CAPACITY_BYTES,
    )
    receipt_path = (tmp_path / "compute.json").resolve()
    receipt_artifact = hpc.publish_canonical_artifact(receipt_path, receipt)
    raw = _raw(
        job_id="410000_0",
        job_id_raw="410100",
        cpus=hpc.R3_CONTROLS_PROFILE_CPUS_PER_TASK,
        memory="96G",
    )
    raw_source = (tmp_path / "sacct.capture").resolve()
    raw_source.write_bytes(raw)
    raw_path = (tmp_path / "sacct.raw").resolve()
    raw_sha = hashlib.sha256(raw).hexdigest()
    output = (tmp_path / "measurement.json").resolve()

    spec = importlib.util.spec_from_file_location("_gate_c_evidence_cli", EVIDENCE_CLI)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert (
        module.main(
            [
                "publish-captured-sacct",
                "--source",
                str(raw_source),
                "--source-sha256",
                raw_sha,
                "--output",
                str(raw_path),
            ]
        )
        == 0
    )
    assert raw_path.read_bytes() == raw
    assert (
        module.main(
            [
                "profile-measurement-finalize",
                "--compute-receipt",
                str(receipt_path),
                "--compute-receipt-file-sha256",
                receipt_artifact.file_sha256,
                "--raw-sacct",
                str(raw_path),
                "--raw-sacct-sha256",
                raw_sha,
                "--family",
                FAMILIES[0],
                "--array-job-id",
                "410000",
                "--job-id-raw",
                "410100",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    measurement = read_canonical_artifact(
        output,
        expected_file_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
    ).payload
    assert measurement["family"] == FAMILIES[0]
    assert measurement["scheduler_terminal"]["raw_sacct_sha256"] == raw_sha
    assert measurement["result_reusable_for_training"] is False


def test_failed_raw_sacct_is_published_before_success_promotion(
    tmp_path: Path,
) -> None:
    raw = _raw(
        job_id="410000_0",
        job_id_raw="410100",
        state="TIMEOUT",
    )
    source = (tmp_path / "failed.capture").resolve()
    source.write_bytes(raw)
    output = (tmp_path / "failed.raw").resolve()
    raw_sha256 = hashlib.sha256(raw).hexdigest()
    spec = importlib.util.spec_from_file_location("_gate_c_failure_cli", EVIDENCE_CLI)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert (
        module.main(
            [
                "publish-captured-sacct",
                "--source",
                str(source),
                "--source-sha256",
                raw_sha256,
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert output.read_bytes() == raw
    with pytest.raises(FileExistsError):
        module.main(
            [
                "publish-captured-sacct",
                "--source",
                str(source),
                "--source-sha256",
                raw_sha256,
                "--output",
                str(output),
            ]
        )


def test_plan_is_three_rolling_arrays_and_exact_nine_namespaces() -> None:
    profile = _profile()
    plan = hpc.build_controls_execution_plan(profile, controls_config=FakeConfig())
    assert [array["array_task_range"] for array in plan["arrays"]] == ["0-2%1"] * 3
    assert len(plan["tasks"]) == 9
    assert len({task["namespace"] for task in plan["tasks"]}) == 9
    assert plan["optimizer_schedule_sha256"] == profile["optimizer_schedule_sha256"]


def test_terminal_requires_exact_completed_raw_slurm_row() -> None:
    profile = _profile()
    plan = hpc.build_controls_execution_plan(profile, controls_config=FakeConfig())
    entry = _entries(plan, profile)[0]
    result, closure, terminal, raw = entry
    checked = hpc.validate_controls_task_terminal(
        terminal,
        raw_sacct_bytes=raw,
        plan=plan,
        profile=profile,
        controls_config=FakeConfig(),
        closure=closure,
        family_result=result,
    )
    assert checked["elapsed_seconds"] == 120

    timeout = _raw(job_id="500000_0", job_id_raw="600000", state="TIMEOUT")
    with pytest.raises(ValueError, match="COMPLETED"):
        hpc.build_controls_task_terminal(
            timeout,
            expected_raw_sacct_sha256=hashlib.sha256(timeout).hexdigest(),
            plan=plan,
            profile=profile,
            controls_config=FakeConfig(),
            closure=closure,
            family_result=result,
            array_job_id="500000",
            job_id_raw="600000",
        )


def test_aggregate_requires_all_nine_and_authorization_is_head_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _profile()
    plan = hpc.build_controls_execution_plan(profile, controls_config=FakeConfig())
    entries = _entries(plan, profile)
    with pytest.raises(ValueError, match="exactly nine"):
        hpc.build_controls_aggregate(
            entries[:-1],
            plan=plan,
            profile=profile,
            controls_config=FakeConfig(),
        )
    aggregate = hpc.build_controls_aggregate(
        entries,
        plan=plan,
        profile=profile,
        controls_config=FakeConfig(),
    )
    assert aggregate["matrix_shape"] == [3, 3]
    assert aggregate["fresh_calibration_authorized"] is False

    gate_r = {
        "schema_version": "phase2-recovery-r3-success-authorization/v1",
        "recovery_design_sha256": "7" * 64,
        "execution_revision": 3,
        "ordered_seeds": list(SEEDS),
        "gate_r_passed": True,
        "fresh_calibration_authorized": False,
        "optimizer_schedule_sha256": aggregate["optimizer_schedule_sha256"],
        "authorization_sha256": "8" * 64,
    }
    monkeypatch.setattr(hpc, "_validated_gate_r", lambda value: deepcopy(value))
    authorization = hpc.build_controls_authorization(
        aggregate,
        gate_r_authorization=gate_r,
        gate_r_authorization_file_sha256="9" * 64,
    )
    assert authorization["fresh_calibration_authorized"] is True
    serialized = str(authorization).lower()
    for forbidden in ("head_weight", "optimizer_state", "checkpoint_bytes", "beta_values': true"):
        assert forbidden not in serialized


def test_publication_is_no_overwrite(tmp_path: Path) -> None:
    profile = _profile()
    output = (tmp_path / "profile.json").resolve()
    hpc.publish_canonical_artifact(output, profile)
    with pytest.raises(FileExistsError):
        hpc.publish_canonical_artifact(output, profile)


def test_submit_surface_is_profile_owned_rolling_and_preflight_fail_closed() -> None:
    text = SUBMIT.read_text(encoding="utf-8")
    assert '"${host_python}" -I -S "${plan_inspector}"' in text
    assert "inspect_phase2_r3_controls_plan_stdlib.py" in text
    assert '[[ "${array_concurrency}" == "1"' in text
    assert '[[ "${max_scheduler_segments}" == "1" ]]' in text
    assert "walltime_seconds <= 172800" in text
    assert "squeue -r" in text
    assert "submitted_count <= 1" in text
    assert "running_count <= 1" in text
    assert '--array="${array_spec}"' in text
    assert '--time="${slurm_walltime}"' in text
    assert '--signal="B:USR1@${signal_lead_seconds}"' in text
    assert "--no-requeue" in text
    assert "missing committed Gate-C family science runner; no job was submitted" in text
    assert 'readonly HOST_PYTHON="/opt/shared/' in text
    assert 'readonly HOST_PYTHON_SHA256="9c91f9aa' in text
    assert '"${host_python}" --version' in text
    assert "apptainer exec" not in text
    assert "SIF_PYTHON" not in text
    assert 'PYTHONPATH="${repo_root}/src" python3' not in text
    assert 'export_spec="PATH=/usr/bin:/bin"' in text
    assert "status --porcelain --untracked-files=all" in text
    assert "container SHA-256 mismatch" in text
    assert "PRORM_R3_GATEC_HF_CACHE" in text
    for forbidden in (
        "--walltime-seconds",
        "--cpus-per-task",
        "--memory-bytes",
        "--array-concurrency",
        "--checkpoint-cadence",
        "--max-scheduler-segments",
    ):
        assert (
            forbidden
            not in text.split("sbatch --parsable", 1)[0].split(
                "inspect-plan",
                1,
            )[0]
        )


def test_sbatch_has_independent_task_namespace_and_no_science_overrides() -> None:
    text = SBATCH.read_text(encoding="utf-8")
    assert "#SBATCH" not in text
    assert "--cleanenv" in text
    assert "PRORM_R3_GATEC_HF_CACHE" in text
    assert '--env "HF_HOME=${hf_cache}"' in text
    assert '--env "TRANSFORMERS_CACHE=${hf_cache}"' in text
    assert '--env "HF_HUB_OFFLINE=1"' in text
    assert '--env "TRANSFORMERS_OFFLINE=1"' in text
    assert 'task_id="$((PRORM_R3_GATEC_FAMILY_INDEX * 3' in text
    assert 'task_root="${submission_root}/task-${SLURM_ARRAY_TASK_ID}-seed-${seed}"' in text
    assert "task namespace already exists; refusing replacement" in text
    assert "run_phase2_r3_control_family.py" in text
    assert 'readonly SIF_PYTHON="/opt/conda/bin/python"' in text
    assert 'python3 "${science_runner}"' not in text
    assert '"${SIF_PYTHON}" "${science_runner}"' in text
    assert "trap forward_usr1 USR1" in text
    assert 'kill -USR1 "${runner_pid}"' in text
    assert '--task-id "${task_id}"' in text
    assert '--family "${family}"' in text
    assert '--seed "${seed}"' in text
    assert "pending_external_sacct_terminal_finalization" in text
    for forbidden in (
        "--head",
        "--heldout",
        "--validation",
        "--policy",
        "--rollout",
        "--utility",
        "--beta",
    ):
        assert forbidden not in text


def test_real_control_runners_and_disposable_profile_surface_are_separated() -> None:
    formal = FORMAL_RUNNER.read_text(encoding="utf-8")
    profile = PROFILE_RUNNER.read_text(encoding="utf-8")
    evidence = EVIDENCE_CLI.read_text(encoding="utf-8")
    submit = PROFILE_SUBMIT.read_text(encoding="utf-8")
    sbatch = PROFILE_SBATCH.read_text(encoding="utf-8")
    profile_finalize_submit = PROFILE_FINALIZE_SUBMIT.read_text(encoding="utf-8")
    profile_finalize_sbatch = PROFILE_FINALIZE_SBATCH.read_text(encoding="utf-8")
    formal_finalize_submit = FORMAL_FINALIZE_SUBMIT.read_text(encoding="utf-8")
    formal_finalize_sbatch = FORMAL_FINALIZE_SBATCH.read_text(encoding="utf-8")

    assert "materialize_r3_control_train_only_from_parent" in formal
    assert "run_r3_control_family" in formal
    assert "prepare_neutral_phase2_context" not in formal
    assert "materialize_r3_control_train_only_from_parent" in profile
    assert "profile_r3_control_family" in profile
    assert "build_profile_compute_receipt" in profile
    assert "torch.cuda.get_device_properties(device).total_memory" in profile
    assert "R3_CONTROLS_PROFILE_TORCH_VISIBLE_GPU_MEMORY_BYTES" in profile
    assert "47,676,129,280-byte Torch-visible HPC4 L20 capacity" in profile
    assert '"formal_result_issued": False' in profile
    assert '"primary_label_stream_constructed": False' in profile
    assert "profile-measurement-finalize" in evidence
    assert "build_profile_scheduler_terminal" in evidence
    assert "build_profile_family_measurement_from_compute_receipt" in evidence

    assert "--array=0-2%1" in submit
    assert "--time=0-12:00:00" in submit
    assert 'export_spec="PATH=/usr/bin:/bin"' in submit
    assert 'readonly HOST_PYTHON="/opt/shared/' in submit
    assert '"${host_python}" --version' in submit
    assert "build_controls_operational_profile" in submit
    assert "not a formal-family walltime source" in submit
    assert "common_beta_pilot_base.yaml" in submit
    assert "common_beta_recovery_pilot.yaml" not in submit
    assert 'chmod 0440 -- "${destination}"' in submit
    assert "retained Gate-C profile input must have mode 0440" in submit
    assert "PRORM_R3_GATEC_PROFILE_HF_CACHE" in submit
    assert 'readonly SIF_PYTHON="/opt/conda/bin/python"' in sbatch
    assert 'python3 "${runner}"' not in sbatch
    assert '"${SIF_PYTHON}" "${runner}"' in sbatch
    assert "common_beta_pilot_base.yaml" in sbatch
    assert "common_beta_recovery_pilot.yaml" not in sbatch
    assert "PRORM_R3_GATEC_PROFILE_HF_CACHE" in sbatch
    assert '--env "HF_HOME=${hf_cache}"' in sbatch
    assert '--env "TRANSFORMERS_CACHE=${hf_cache}"' in sbatch

    assert "submit_phase2_r3_controls_profile_finalize.sh" in submit
    assert '--dependency="afterany:${profile_array_job_id}"' in profile_finalize_submit
    assert "--time=0-01:00:00" in profile_finalize_submit
    assert "apptainer exec" not in profile_finalize_submit
    assert "apptainer exec" in profile_finalize_sbatch
    assert "profile-measurement-finalize" in profile_finalize_sbatch
    assert "profile-finalize" in profile_finalize_sbatch
    assert "--walltime-seconds 86400" in profile_finalize_sbatch
    assert "--checkpoint-cadence-updates 200" in profile_finalize_sbatch
    assert "--format=${SACCT_FORMAT}" in profile_finalize_sbatch

    assert '"${host_python}" -I -S "${inspector}"' in formal_finalize_submit
    assert "apptainer exec" not in formal_finalize_submit
    assert '--dependency="afterany:${array0}:${array1}:${array2}"' in (formal_finalize_submit)
    assert "apptainer exec" in formal_finalize_sbatch
    assert "for family_index in 0 1 2" in formal_finalize_sbatch
    assert "for seed_index in 0 1 2" in formal_finalize_sbatch
    assert "publish-captured-sacct" in formal_finalize_sbatch
    assert "run_in_sif terminal" in formal_finalize_sbatch
    assert "run_in_sif aggregate" in formal_finalize_sbatch
    assert "run_in_sif authorize" in formal_finalize_sbatch


def test_finalizers_capture_complete_raw_sets_before_success_artifacts() -> None:
    profile_submit = PROFILE_FINALIZE_SUBMIT.read_text(encoding="utf-8")
    profile = PROFILE_FINALIZE_SBATCH.read_text(encoding="utf-8")
    formal = FORMAL_FINALIZE_SBATCH.read_text(encoding="utf-8")

    assert "profile-compute-receipt.json" not in profile_submit
    assert 'evidence_root="${submission_root}/terminal-evidence"' in profile
    assert "# Failure evidence comes first." in profile
    assert "# Success promotion starts only after" in profile
    assert profile.index("run_in_sif publish-captured-sacct") < profile.index('receipt="$(realpath')
    assert 'if [[ -e "${raw}" || -L "${raw}" ]]' in profile

    assert 'evidence_root="${submission_parent}/terminal-evidence"' in formal
    assert "# Capture every scheduler outcome before" in formal
    assert "# Only a complete successful matrix can enter" in formal
    assert formal.index("run_in_sif publish-captured-sacct") < formal.index('result="$(realpath')
    assert formal.index("run_in_sif publish-captured-sacct") < formal.index('closure="$(realpath')
    assert 'if [[ -e "${raw}" || -L "${raw}" ]]' in formal


def test_pure_stdlib_plan_inspector_accepts_real_contract_and_rejects_tamper(
    tmp_path: Path,
) -> None:
    controls_path = ROOT / "configs" / "phase2_recovery_r3_controls.yaml"
    controls = load_r3_controls_config(controls_path)
    measurements = [
        hpc.build_profile_family_measurement(
            family=family,
            seed=SEEDS[0],
            git_commit="3" * 40,
            container_sha256="4" * 64,
            controls_config_file_sha256=controls.file_sha256,
            controls_config_semantic_sha256=controls.semantic_sha256,
            input_training_sha256=hashlib.sha256(f"real-input-{family}".encode()).hexdigest(),
            oracle_reward_sha256=hashlib.sha256(f"real-oracle-{family}".encode()).hexdigest(),
            setup_wall_seconds=1.0,
            training_wall_seconds=10.0,
            audit_wall_seconds=1.0,
            checkpoint_roundtrip_wall_seconds=0.5,
            peak_gpu_memory_bytes=1024**3,
            gpu_total_memory_bytes=(hpc.R3_CONTROLS_PROFILE_GPU_MEMORY_CAPACITY_BYTES),
            scheduler_terminal={
                "array_job_id": "410000",
                "array_task_id": index,
                "job_id": f"410000_{index}",
                "job_id_raw": str(410100 + index),
                "raw_sacct_sha256": hashlib.sha256(f"real-profile-{family}".encode()).hexdigest(),
                "elapsed_seconds": 120,
            },
        )
        for index, family in enumerate(FAMILIES)
    ]
    profile = hpc.build_controls_operational_profile(
        measurements,
        controls_config=controls,
        optimizer_schedule_sha256=hpc.R3_OPTIMIZER_SCHEDULE_SHA256,
        checkpoint_cadence_updates=200,
        walltime_safety_margin_fraction=0.25,
        fixed_walltime_margin_seconds=1800.0,
        memory_safety_margin_fraction=0.25,
        cluster="hpc4",
        account="sigroup",
        partition="gpu-l20",
        gpu_name="NVIDIA L20",
        cpus_per_task=8,
        memory_bytes=96 * 1024**3,
        array_concurrency=1,
        requested_walltime_seconds_per_segment=86400,
        signal_lead_seconds=1800,
        max_scheduler_segments=1,
    )
    plan = hpc.build_controls_execution_plan(profile, controls_config=controls)
    profile_artifact = publish_canonical_artifact(
        (tmp_path / "profile.json").resolve(),
        profile,
    )
    plan_artifact = publish_canonical_artifact(
        (tmp_path / "plan.json").resolve(),
        plan,
    )
    command = [
        sys.executable,
        "-I",
        "-S",
        str(STDLIB_INSPECTOR),
        "--controls-config",
        str(controls_path),
        "--profile",
        str(profile_artifact.artifact_path),
        "--profile-file-sha256",
        profile_artifact.file_sha256,
        "--plan",
        str(plan_artifact.artifact_path),
        "--plan-file-sha256",
        plan_artifact.file_sha256,
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    inspected = json.loads(completed.stdout)
    assert inspected["plan_sha256"] == plan["plan_sha256"]
    assert inspected["resources"]["observed_gpu_memory_capacity_bytes"] == 47_676_129_280

    tampered = deepcopy(plan)
    tampered["tasks"][0]["seed"] = SEEDS[1]
    unsigned = dict(tampered)
    unsigned.pop("plan_sha256")
    tampered["plan_sha256"] = hpc._semantic_sha256(unsigned)
    tampered_artifact = publish_canonical_artifact(
        (tmp_path / "tampered-plan.json").resolve(),
        tampered,
    )
    command[-1] = tampered_artifact.file_sha256
    command[-3] = str(tampered_artifact.artifact_path)
    rejected = subprocess.run(command, check=False, capture_output=True, text=True)
    assert rejected.returncode != 0
    assert "exact 3x3 task matrix" in rejected.stderr
