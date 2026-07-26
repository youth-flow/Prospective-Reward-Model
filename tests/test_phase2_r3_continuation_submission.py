from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
PREPARE = ROOT / "scripts" / "hpc4" / "prepare_phase2_r3_continuation_submission.py"
RUNNER = ROOT / "scripts" / "hpc4" / "run_phase2_r3_primary_continuation.py"
LAUNCHER = ROOT / "scripts" / "hpc4" / "submit_phase2_r3_continuation.sh"
SUBMIT = ROOT / "scripts" / "hpc4" / "phase2_r3_continuation_submission.sbatch"
SBATCH = ROOT / "scripts" / "hpc4" / "phase2_r3_primary.sbatch"


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _digest(character: str) -> str:
    return character * 64


def _base(tmp_path: Path) -> dict[str, object]:
    return {
        "submission_plan_sha256": _digest("a"),
        "operational_bundle_path": str((tmp_path / "bundle.json").resolve()),
        "operational_bundle_file_sha256": _digest("b"),
        "operational_bundle_semantic_sha256": _digest("c"),
        "resource_plan_sha256": _digest("d"),
        "slurm_account": "sigroup",
        "partition": "gpu-l20",
        "gpu_name": "NVIDIA L20",
        "gpus_per_task": 1,
        "cpus_per_task": 8,
        "memory_bytes": 64 * 1024 * 1024 * 1024,
        "memory_mib": 65536,
        "requested_walltime_seconds": 43200,
        "slurm_walltime": "0-12:00:00",
        "array_concurrency": 2,
        "max_scheduler_segments": 4,
        "advance_signal_lead_seconds": 900,
        "audit_cadence_updates": 20,
        "durable_checkpoint_cadence_updates": 200,
    }


def _create_argv(
    tmp_path: Path,
    output: Path,
    *,
    previous: Path | None = None,
) -> list[str]:
    base_path = tmp_path / "base-plan.json"
    result = [
        "create",
        "--primary-submission-plan",
        str(base_path),
        "--primary-submission-plan-file-sha256",
        _digest("1"),
    ]
    if previous is not None:
        result += [
            "--previous-continuation-plan",
            str(previous),
            "--previous-continuation-plan-file-sha256",
            hashlib.sha256(previous.read_bytes()).hexdigest(),
        ]
    for task_id in range(3):
        result += [
            f"--task-{task_id}-runtime-closure",
            str(tmp_path / f"closure-{task_id}.json"),
            f"--task-{task_id}-runtime-closure-file-sha256",
            _digest(str(task_id + 2)),
            f"--task-{task_id}-terminal-evidence-directory",
            str(tmp_path / f"terminal-{task_id}"),
            f"--task-{task_id}-terminal-manifest-file-sha256",
            _digest(str(task_id + 5)),
            f"--task-{task_id}-terminal-raw-sacct-sha256",
            _digest(str(task_id + 7)),
        ]
    return [*result, "--output", str(output)]


def test_continuation_plan_extends_full_sealed_history_and_stops_at_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepare = _load(PREPARE, "_prepare_continuation")
    (tmp_path / "base-plan.json").write_text("{}\n", encoding="utf-8")
    base = _base(tmp_path)
    bundle = SimpleNamespace(
        artifact_path=Path(base["operational_bundle_path"]),
        file_sha256=base["operational_bundle_file_sha256"],
        bundle_semantic_sha256=base["operational_bundle_semantic_sha256"],
        resource_plan_sha256=base["resource_plan_sha256"],
        max_scheduler_segments=4,
    )
    monkeypatch.setattr(prepare, "_base_plan", lambda *_args, **_kwargs: base)
    monkeypatch.setattr(
        prepare,
        "reopen_verified_gate_p_operational_bundle",
        lambda *_args, **_kwargs: bundle,
    )
    state = {"segment": 1, "kind": "continuable"}

    def latest(*, task_id: int, bundle: object, arguments: object) -> dict[str, object]:
        del bundle, arguments
        segment = state["segment"]
        kind = state["kind"]
        return prepare._validate_entry(
            {
                "segment_index": segment,
                "runtime_closure_path": str(
                    (tmp_path / f"closure-s{segment}-t{task_id}.json").resolve()
                ),
                "runtime_closure_file_sha256": _digest(str(task_id + 1)),
                "runtime_closure_sha256": _digest(str(task_id + 4)),
                "terminal_kind": kind,
                "terminal_evidence_directory": str(
                    (tmp_path / f"terminal-s{segment}-t{task_id}").resolve()
                ),
                "terminal_manifest_file_sha256": _digest(str(task_id + 6)),
                "terminal_raw_sacct_sha256": _digest(str(task_id + 7)),
                "terminal_sha256": _digest(str(task_id + 3)),
                "array_job_id": str(100 + segment),
                "job_id": str(200 + segment * 10 + task_id),
                "selected_checkpoint": (
                    None
                    if kind == "completed"
                    else {
                        "schema_version": "checkpoint/v1",
                        "artifact_sha256": _digest(str(task_id + 1)),
                        "role": "verified",
                    }
                ),
            },
            task_id=task_id,
        )

    monkeypatch.setattr(prepare, "_latest_entry", latest)
    first = tmp_path / "continuation-1.json"
    assert prepare.main(_create_argv(tmp_path, first)) == 0
    first_plan = json.loads(first.read_bytes())
    assert [route["next_segment_index"] for route in first_plan["task_routes"]] == [
        2,
        2,
        2,
    ]
    assert first_plan["active_array_task_ids"] == [0, 1, 2]

    state["segment"] = 2
    second = tmp_path / "continuation-2.json"
    assert prepare.main(_create_argv(tmp_path, second, previous=first)) == 0
    second_plan = json.loads(second.read_bytes())
    assert all(len(route["history"]) == 2 for route in second_plan["task_routes"])
    assert [route["next_segment_index"] for route in second_plan["task_routes"]] == [
        3,
        3,
        3,
    ]
    lineage = prepare.reopen_continuation_plan_lineage(
        second,
        expected_file_sha256=hashlib.sha256(second.read_bytes()).hexdigest(),
    )
    assert len(lineage) == 2

    forged = dict(second_plan)
    forged["previous_continuation_plan_path"] = None
    forged["previous_continuation_plan_file_sha256"] = None
    forged_unsigned = dict(forged)
    forged_unsigned.pop("continuation_plan_sha256")
    forged["continuation_plan_sha256"] = prepare._semantic_sha256(forged_unsigned)
    forged_path = tmp_path / "forged-skipped-lineage.json"
    forged_artifact = prepare.publish_canonical_artifact(forged_path, forged)
    with pytest.raises(ValueError, match="initial continuation plan"):
        prepare.reopen_continuation_plan_lineage(
            forged_path,
            expected_file_sha256=forged_artifact.file_sha256,
        )

    state["kind"] = "completed"
    completed = tmp_path / "completed.json"
    assert prepare.main(_create_argv(tmp_path, completed, previous=first)) == 0
    completed_plan = json.loads(completed.read_bytes())
    assert completed_plan["all_tasks_complete"] is True
    assert completed_plan["continuation_array_required"] is False
    assert completed_plan["active_array_task_ids"] == []
    assert completed_plan["dependency_array_job_ids"] == []
    assert all(route["next_segment_index"] is None for route in completed_plan["task_routes"])


def _runner_argv(tmp_path: Path | None = None) -> list[str]:
    project_root = "project" if tmp_path is None else str(tmp_path)
    primary_plan = "primary-plan.json" if tmp_path is None else str(tmp_path / "primary-plan.json")
    return [
        "--project-root",
        project_root,
        "--science-config",
        "science.yaml",
        "--source-config",
        "source.yaml",
        "--parent-registry",
        "registry.json",
        "--parent-registry-file-sha256",
        _digest("1"),
        "--gate0-file-sha256",
        _digest("2"),
        "--gate1-file-sha256",
        _digest("3"),
        "--source-test-receipt-file-sha256",
        _digest("4"),
        "--operational-bundle",
        "bundle.json",
        "--operational-bundle-file-sha256",
        _digest("5"),
        "--profile-allocation-intent",
        "intent.json",
        "--profile-allocation-intent-file-sha256",
        _digest("6"),
        "--profile-runtime-receipt",
        "runtime.json",
        "--profile-runtime-receipt-file-sha256",
        _digest("7"),
        "--profile-terminal-evidence-directory",
        "profile-terminal",
        "--profile-terminal-manifest-file-sha256",
        _digest("8"),
        "--profile-terminal-raw-sacct-sha256",
        _digest("9"),
        "--primary-submission-plan",
        primary_plan,
        "--primary-submission-plan-file-sha256",
        _digest("b"),
        "--primary-submission-plan-sha256",
        _digest("c"),
        "--continuation-plan",
        "continuation.json",
        "--continuation-plan-file-sha256",
        _digest("a"),
        "--task-root",
        "task-root",
        "--runtime-closure",
        "closure.json",
    ]


def test_runner_rebuilds_every_predecessor_before_admitting_next_segment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(RUNNER.parent))
    runner = _load(RUNNER, "_run_continuation")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "1")
    history = [
        {
            "terminal_kind": "continuable",
            "runtime_closure_path": f"closure-{index}.json",
            "runtime_closure_file_sha256": _digest(str(index)),
            "runtime_closure_sha256": _digest(str(index + 2)),
            "terminal_evidence_directory": f"terminal-{index}",
            "terminal_manifest_file_sha256": _digest(str(index + 4)),
            "terminal_raw_sacct_sha256": _digest(str(index + 6)),
            "terminal_sha256": _digest(str(index + 7)),
            "selected_checkpoint": {"segment": index},
        }
        for index in (1, 2)
    ]
    plan = {
        "operational_bundle_file_sha256": _digest("5"),
        "operational_bundle_semantic_sha256": _digest("b"),
        "resource_plan_sha256": _digest("c"),
        "continuation_plan_sha256": _digest("d"),
        "task_routes": [
            {"action": "complete"},
            {"action": "continue", "next_segment_index": 3, "history": history},
            {"action": "complete"},
        ],
    }
    bundle = SimpleNamespace(
        file_sha256=_digest("5"),
        bundle_semantic_sha256=_digest("b"),
        resource_plan_sha256=_digest("c"),
        requested_walltime_seconds_per_segment=43200,
    )
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(runner, "reopen_continuation_plan", lambda *_a, **_k: plan)
    for name, value in (
        ("load_r3_science_config", object()),
        ("verify_live_r3_gate0_in_container", object()),
        ("verify_live_r3_gate1_in_container", object()),
        ("reopen_verified_gate_p_operational_bundle", bundle),
        ("reopen_profile_allocation_intent", object()),
        ("reopen_profile_slurm_runtime_receipt", object()),
        ("revalidate_successful_profile_terminal", object()),
        ("authorize_gate_p", object()),
    ):
        monkeypatch.setattr(
            runner,
            name,
            lambda *_a, _name=name, _value=value, **_k: (
                calls.append((_name, None)),
                _value,
            )[1],
        )
    design = SimpleNamespace(design_sha256=_digest("e"))
    monkeypatch.setattr(runner, "create_r3_primary_design", lambda **_k: design)
    capability = object()
    monkeypatch.setattr(
        runner,
        "materialize_r3_train_only_from_parent",
        lambda **_k: SimpleNamespace(capability=capability),
    )
    closures = []
    terminals = []
    evidences = []
    for index, entry in enumerate(history, start=1):
        closure = SimpleNamespace(
            closure_sha256=entry["runtime_closure_sha256"],
            admission_payload={
                "segment_index": index,
                "task_id": 1,
                "seed": 20260802,
            },
            outcome_payload={"continuation_checkpoint": entry["selected_checkpoint"]},
        )
        closures.append(closure)
        terminals.append(SimpleNamespace(terminal_sha256=entry["terminal_sha256"]))
        evidences.append(SimpleNamespace(evidence_sha256=_digest(str(index + 5))))
    monkeypatch.setattr(
        runner,
        "reopen_primary_segment_runtime_closure",
        lambda *_a, **_k: closures.pop(0),
    )
    predecessor_payloads = [{"segment_index": 1}, {"segment_index": 2}]
    predecessors = [
        SimpleNamespace(to_dict=lambda payload=payload: payload) for payload in predecessor_payloads
    ]
    monkeypatch.setattr(
        runner,
        "rehydrate_primary_segment_admission",
        lambda *_a, **_k: predecessors.pop(0),
    )
    primary_plan = tmp_path / "primary-plan.json"
    primary_plan.write_text("{}\n", encoding="utf-8")
    identity_path = tmp_path / "identity.json"
    identity_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(runner, "identity_receipt_path", lambda *_a, **_k: identity_path)
    monkeypatch.setattr(
        runner,
        "reopen_primary_identity_receipt",
        lambda *_a, **_k: {
            "base_primary_submission_plan": {
                "path": "primary-plan.json",
                "file_sha256": _digest("b"),
                "submission_plan_sha256": _digest("c"),
            },
            "segment_1_admission": predecessor_payloads[0],
        },
    )
    monkeypatch.setattr(
        runner,
        "revalidate_continuable_primary_terminal",
        lambda *_a, **_k: terminals.pop(0),
    )
    evidence_queue = list(evidences)
    monkeypatch.setattr(
        runner,
        "validate_continuation_evidence",
        lambda **_k: evidence_queue.pop(0),
    )
    admission = SimpleNamespace(segment_index=3, admission_sha256=_digest("f"))
    admitted: list[dict[str, object]] = []

    def admit(**kwargs: object) -> object:
        admitted.append(kwargs)
        return admission

    monkeypatch.setattr(runner, "admit_primary_segment", admit)
    runtime = SimpleNamespace(runtime_sha256=_digest("0"))
    monkeypatch.setattr(runner, "capture_slurm_segment_runtime", lambda *_a, **_k: runtime)
    outcome = SimpleNamespace(status="done", outcome_sha256=_digest("1"))
    monkeypatch.setattr(runner, "run_r3_primary_task_segment", lambda *_a, **_k: outcome)
    closure = SimpleNamespace(closure_sha256=_digest("2"), file_sha256=_digest("3"))
    monkeypatch.setattr(
        runner,
        "publish_primary_segment_runtime_closure",
        lambda *_a, **_k: closure,
    )
    segment_receipt = SimpleNamespace(
        file_sha256=_digest("4"),
        payload={"segment_evidence_receipt_sha256": _digest("5")},
    )
    monkeypatch.setattr(
        runner,
        "publish_segment_evidence_receipt",
        lambda *_a, **_k: segment_receipt,
    )

    class Signal:
        def __enter__(self) -> Signal:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(runner, "CheckpointSignal", Signal)
    monkeypatch.setattr(runner, "_emit", lambda _value: None)
    assert runner.main(_runner_argv(tmp_path)) == 0
    assert admitted[0]["segment_index"] == 3
    assert admitted[0]["task_id"] == 1
    assert admitted[0]["seed"] == 20260802
    assert admitted[0]["continuation_evidence"] is evidences[-1]
    assert admitted[0]["materialization_capability"] is capability


def test_completed_task_and_unsealed_failure_paths_never_run_or_resubmit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(RUNNER.parent))
    runner = _load(RUNNER, "_run_completed_guard")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "0")
    monkeypatch.setattr(
        runner,
        "reopen_continuation_plan",
        lambda *_a, **_k: {"task_routes": [{"action": "complete"}]},
    )
    destinations = {action.dest for action in runner._parser()._actions}
    assert destinations.isdisjoint(
        {
            "seed",
            "segment_index",
            "head",
            "primary_heads",
            "continuation_checkpoint",
            "walltime_seconds",
            "memory_bytes",
            "array_concurrency",
            "heldout",
            "control",
        }
    )
    with pytest.raises(RuntimeError, match="must never enter"):
        runner.main(_runner_argv())

    submit = SUBMIT.read_text(encoding="utf-8")
    launcher = LAUNCHER.read_text(encoding="utf-8")
    sbatch = SBATCH.read_text(encoding="utf-8")
    assert "/project/sigroup:/project/sigroup:" not in submit
    assert '--bind "${project_root}:${project_root}:ro"' in submit
    assert '--bind "${project_root}:${project_root}:rw"' in submit
    assert '[[ "${image}" == "${project_root}/"* ]]' in submit
    assert '[[ "${hf_cache}" == "${project_root}/"* ]]' in submit
    assert "apptainer exec" not in launcher
    assert "phase2_r3_continuation_submission.sbatch" in launcher
    assert "exec srun" in launcher
    assert "--partition=gpu-l20" in launcher
    assert "--gpus-per-node=1" in launcher
    assert '--dependency="afterok:${dependency_job_ids}"' in submit
    assert "afternotok:%s" in submit
    assert '--array="${active_array_task_ids}%${array_concurrency}"' in submit
    assert "all_three_tasks_compute_complete_no_resubmission" in submit
    assert "sealed_completed_task_not_resubmitted" in sbatch
    assert "kill-on-invalid-dep=yes" in submit
    assert "--nodes=1" in submit
    assert "--ntasks=1" in submit
    assert "--gpus-per-node=1" in submit
    assert "--gpus-per-task" not in submit
    assert "/home/yyangjo/Smart-Reward-Model" in submit
    assert "/project/sigroup/smart-reward-model" in submit
    assert "PYTHONPATH=${repo_root}/src" in submit
    assert "cp --no-clobber" in submit
    assert "cmp --silent" in submit
    assert 'input_root="${input_parent}/${commit}"' in submit
    assert 'science_config="${input_root}/phase2_recovery_r3_science.yaml"' in submit
    assert 'source_config="${input_root}/common_beta_pilot_base.yaml"' in submit
    assert "common_beta_recovery_pilot.yaml" not in submit
    assert 'parent_registry="${input_root}/phase2_recovery_parent_failures.json"' in submit
    assert (
        'gate1="${project_root}/runs/phase2-recovery-r3/gate1/'
        '${commit}/r3-implementation-closure.json"'
    ) in submit
    assert (
        'source_test_receipt="${project_root}/runs/phase2-recovery-r3/gate1/'
        '${commit}/r3-source-test-receipt.json"'
    ) in submit
    assert submit.index('commit="$(git -C "${repo_root}" rev-parse HEAD)"') < submit.index(
        "${commit}/r3-implementation-closure.json"
    )
    assert 'export PRORM_R3_SCIENCE_CONFIG="${science_config}"' in submit
    assert 'export PRORM_R3_CONTINUATION_PLAN_SHA256="${continuation_plan_sha256}"' in submit
    assert "--format binding-lines" in submit
    assert 'die "base primary ${base_binding_names[index]} binding drifted"' in submit
    for binding_name in (
        "git_commit",
        "container_image_file_sha256",
        "science_config_path",
        "operational_bundle_path",
        "profile_terminal_raw_sacct_sha256",
    ):
        assert f"\n  {binding_name}\n" in submit
    assert "PRORM_R3_CONTINUATION_ATTEMPT_ROOT|PRORM_R3_TASK_*" in submit
    for canonical_export in (
        'export PRORM_R3_IMAGE="${image}"',
        'export PRORM_R3_REPO_ROOT="${repo_root}"',
        'export PRORM_R3_PROJECT_ROOT="${project_root}"',
        'export PRORM_R3_SCRATCH_ROOT="${scratch_root}"',
        'export PRORM_R3_HF_CACHE="${hf_cache}"',
        'export PRORM_R3_OPERATIONAL_BUNDLE="${operational_bundle}"',
        'export PRORM_R3_PROFILE_INTENT="${profile_intent}"',
        'export PRORM_R3_PROFILE_RUNTIME_RECEIPT="${profile_runtime_receipt}"',
        ('export PRORM_R3_PROFILE_TERMINAL_EVIDENCE_DIRECTORY="${profile_terminal_directory}"'),
    ):
        assert canonical_export in submit
    assert "retained input copy differs from clean repository bytes" in submit
    assert "runs/phase2-recovery-r3/inputs/${PRORM_R3_GIT_COMMIT}" in sbatch
    assert "common_beta_pilot_base.yaml" in sbatch
    assert "common_beta_recovery_pilot.yaml" not in sbatch
    assert '    --source-config "${source_config}" \\' in sbatch
    assert '    --parent-registry "${parent_registry}" \\' in sbatch
    assert '    --source-config "${PRORM_R3_SOURCE_CONFIG}" \\' not in sbatch
    assert '    --parent-registry "${PRORM_R3_PARENT_REGISTRY}" \\' not in sbatch
    assert '[[ "${SLURM_NNODES:-}" == "1" ]]' in sbatch
    assert '[[ "${SLURM_NTASKS:-}" == "1" ]]' in sbatch
    assert 'case "${SLURM_GPUS_ON_NODE:-}" in' in sbatch
    assert '--project-root "${project_root}"' in sbatch
    for forbidden in (
        "--seed",
        "--head",
        "--segment-index",
        "--continuation-checkpoint",
        "--heldout",
        "--control",
    ):
        assert forbidden not in submit
        assert forbidden not in sbatch
