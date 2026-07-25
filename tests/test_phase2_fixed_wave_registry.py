from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).parents[1]
HPC4 = ROOT / "scripts" / "hpc4"
RESOLVER = HPC4 / "resolve_phase2_campaign_registry.py"
WAVE_TASKS = (
    (0, 1, 2, 3),
    (4, 5, 6, 7),
    (8, 9, 10, 11),
    (12, 13, 14, 15),
    (16, 17, 18, 19),
    (20, 21, 22, 23),
    (24, 25, 26, 27),
    (28, 29),
)


def _load_resolver() -> ModuleType:
    module_name = "_test_phase2_fixed_wave_resolver"
    sys.path.insert(0, str(HPC4))
    try:
        spec = importlib.util.spec_from_file_location(module_name, RESOLVER)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(HPC4))


def _write_json(path: Path, value: object) -> bytes:
    raw = (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode()
    path.write_bytes(raw)
    return raw


def _make_registry(tmp_path: Path) -> tuple[Path, dict[str, str], dict[str, object], str]:
    identities = {
        "design": "a" * 64,
        "base": "b" * 64,
        "commit": "c" * 40,
        "image": "d" * 64,
        "inventory": "e" * 64,
        "freeze": "f" * 64,
    }
    root = tmp_path / identities["design"]
    registry = root / "campaign-registry"
    for name in (
        ".staging",
        "admissions",
        "submissions",
        "executions",
        "recoveries",
        "scheduler-terminals",
    ):
        (registry / name).mkdir(parents=True)
    (registry / "registry.lock").write_bytes(b"")
    plan: dict[str, object] = {
        "schema_version": "prorm-phase2-fixed-wave-campaign-plan/v1",
        "status": "precommitted_before_first_slurm_submission",
        "phase2_design_sha256": identities["design"],
        "base_config_hash": identities["base"],
        "git_commit": identities["commit"],
        "accepted_freeze_aggregate_sha256": identities["freeze"],
        "ordered_seeds": list(range(20260901, 20260931)),
        "attempt_index": 1,
        "retry_policy": "single_predeclared_attempt_no_retry",
        "replacement_seed_allowed": False,
        "optional_stopping_allowed": False,
        "max_submitted_tasks": 4,
        "max_running_tasks": 2,
        "waves": [
            {
                "wave_index": index,
                "array_spec": f"{tasks[0]}-{tasks[-1]}%2",
                "array_task_ids": list(tasks),
                "seeds": [20260901 + task for task in tasks],
            }
            for index, tasks in enumerate(WAVE_TASKS)
        ],
        "job_tuple": {
            "account": "sigroup",
            "partition": "gpu-l20",
            "qos": "l20_qos",
            "nodes": 1,
            "tasks": 1,
            "cpus_per_task": 8,
            "memory": "64G",
            "gpus_per_node": 1,
            "walltime": "08:00:00",
            "no_requeue": True,
            "held_before_registry_commit": True,
            "script": "scripts/hpc4/phase2_confirmatory.sbatch",
            "script_file_sha256": hashlib.sha256(
                (HPC4 / "phase2_confirmatory.sbatch").read_bytes()
            ).hexdigest(),
        },
        "producer": {
            "overlay_file_sha256": "1" * 64,
            "base_file_sha256": "2" * 64,
            "identities_file_sha256": "3" * 64,
            "image_sha256": identities["image"],
            "hf_inventory_sha256": identities["inventory"],
        },
        "created_at_utc": "2026-07-25T00:00:00Z",
    }
    plan_raw = _write_json(registry / "campaign-plan.json", plan)
    return root, identities, plan, hashlib.sha256(plan_raw).hexdigest()


def _write_submission(
    root: Path,
    *,
    plan: dict[str, object],
    plan_sha: str,
    wave_index: int,
    array_job_id: str,
    admission_sha: str | None = None,
) -> None:
    tasks = WAVE_TASKS[wave_index]
    if admission_sha is None:
        admission = root / "campaign-registry" / "admissions" / f"wave-{wave_index}.json"
        admission_sha = hashlib.sha256(admission.read_bytes()).hexdigest()
    command = str(HPC4 / "phase2_confirmatory.sbatch")
    work_dir = str(ROOT)
    array_spec = f"{tasks[0]}-{tasks[-1]}%2"
    raw_scontrol = " ".join(
        (
            f"JobId={array_job_id}",
            f"ArrayJobId={array_job_id}",
            f"JobName=prorm-p2-{plan_sha[:12]}-w{wave_index}",
            f"ArrayTaskId={array_spec}",
            "ArrayTaskThrottle=2",
            "JobState=PENDING",
            "Reason=JobHeldUser",
            "Account=sigroup",
            "Partition=gpu-l20",
            "QOS=l20_qos",
            "Requeue=0",
            "Restarts=0",
            "NumNodes=1-1",
            "NumTasks=1",
            "NumCPUs=8",
            "CPUs/Task=8",
            "MinMemoryNode=64G",
            f"TimeLimit={plan['job_tuple']['walltime']}",
            "TRES=cpu=8,mem=64G,node=1,billing=8,gres/gpu=1",
            "TresPerNode=gres:gpu:1",
            f"Command={command}",
            f"WorkDir={work_dir}",
        )
    )
    scheduler_request = {
        "schema_version": "prorm-phase2-held-scheduler-request/v1",
        "captured_while_held": True,
        "raw_scontrol_record": raw_scontrol,
        "raw_scontrol_sha256": hashlib.sha256(raw_scontrol.encode()).hexdigest(),
        "normalized": {
            "array_job_id": array_job_id,
            "job_name": f"prorm-p2-{plan_sha[:12]}-w{wave_index}",
            "array_spec": array_spec,
            "array_task_throttle": 2,
            "account": "sigroup",
            "partition": "gpu-l20",
            "qos": "l20_qos",
            "nodes": 1,
            "tasks": 1,
            "cpus": 8,
            "cpus_per_task": 8,
            "memory": "64G",
            "gpus_per_node": 1,
            "walltime": plan["job_tuple"]["walltime"],
            "tres": {"cpu": "8", "gres/gpu": "1", "mem": "64G", "node": "1"},
            "tres_per_node": "gres:gpu:1",
            "requeue": False,
            "restarts": 0,
            "command": command,
            "work_dir": work_dir,
        },
    }
    scheduler_request_sha = hashlib.sha256(
        (
            json.dumps(
                scheduler_request,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode()
    ).hexdigest()
    value = {
        "schema_version": "prorm-phase2-campaign-submission/v3",
        "status": "committed_while_slurm_held",
        "campaign_plan_sha256": plan_sha,
        "wave_admission_sha256": admission_sha,
        "scheduler_request_sha256": scheduler_request_sha,
        "scheduler_request": scheduler_request,
        "wave_index": wave_index,
        "phase2_design_sha256": plan["phase2_design_sha256"],
        "base_config_hash": plan["base_config_hash"],
        "git_commit": plan["git_commit"],
        "accepted_freeze_aggregate_sha256": plan["accepted_freeze_aggregate_sha256"],
        "array_job_id": array_job_id,
        "submitted_cluster": "hpc4",
        "array_spec": array_spec,
        "attempt_index": 1,
        "entries": [
            {
                "seed": 20260901 + task,
                "attempt_index": 1,
                "array_job_id": array_job_id,
                "array_task_id": task,
            }
            for task in tasks
        ],
        "job_tuple": plan["job_tuple"],
        "producer": plan["producer"],
        "replacement_seed_allowed": False,
        "created_at_utc": "2026-07-25T00:00:00Z",
    }
    _write_json(
        root / "campaign-registry" / "submissions" / f"array-{array_job_id}.json",
        value,
    )


def _snapshot_entry(root: Path, seed: int, terminal: Path) -> dict[str, object]:
    marker = terminal.parent / ("SUCCESS" if seed % 2 else "FAILED")
    return {
        "seed": seed,
        "terminal_relative_path": terminal.relative_to(root).as_posix(),
        "terminal_sha256": hashlib.sha256(terminal.read_bytes()).hexdigest(),
        "marker_relative_path": marker.relative_to(root).as_posix(),
        "marker_sha256": hashlib.sha256(marker.read_bytes()).hexdigest(),
    }


def _write_admission(
    root: Path,
    *,
    plan_sha: str,
    wave_index: int,
    predecessor_admission_sha: str | None = None,
    predecessor_submission_sha: str | None = None,
    predecessor_terminals: list[Path] | None = None,
) -> str:
    tasks = WAVE_TASKS[wave_index]
    predecessor_index = None if wave_index == 0 else wave_index - 1
    predecessor_tasks = () if predecessor_index is None else WAVE_TASKS[predecessor_index]
    terminals = predecessor_terminals or []
    snapshot = [
        _snapshot_entry(root, 20260901 + task, terminal)
        for task, terminal in zip(predecessor_tasks, terminals, strict=True)
    ]
    value = {
        "schema_version": "prorm-phase2-wave-admission/v1",
        "status": "committed_before_current_wave_scheduler_submission",
        "campaign_plan_sha256": plan_sha,
        "wave_index": wave_index,
        "wave": {
            "wave_index": wave_index,
            "array_spec": f"{tasks[0]}-{tasks[-1]}%2",
            "array_task_ids": list(tasks),
            "seeds": [20260901 + task for task in tasks],
        },
        "admission_rule": "predecessor_terminal_completeness_only_outcome_independent",
        "predecessor_wave_index": predecessor_index,
        "predecessor_admission_sha256": predecessor_admission_sha,
        "predecessor_submission_sha256": predecessor_submission_sha,
        "predecessor_terminal_snapshot": snapshot,
        "predecessor_terminal_snapshot_sha256": hashlib.sha256(
            (
                json.dumps(
                    snapshot,
                    allow_nan=False,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
            ).encode()
        ).hexdigest(),
        "created_at_utc": "2026-07-25T00:00:00Z",
    }
    raw = _write_json(
        root / "campaign-registry" / "admissions" / f"wave-{wave_index}.json",
        value,
    )
    return hashlib.sha256(raw).hexdigest()


def _resolve(
    resolver: ModuleType,
    root: Path,
    identities: dict[str, str],
) -> tuple[dict[str, object], list[Path]]:
    return resolver._resolve(
        root,
        design=identities["design"],
        base=identities["base"],
        commit=identities["commit"],
        image=identities["image"],
        inventory=identities["inventory"],
    )


def test_fixed_wave_boundaries_and_initial_state_are_exact(tmp_path: Path) -> None:
    resolver = _load_resolver()
    root, identities, _, plan_sha = _make_registry(tmp_path)

    assert resolver.WAVE_TASKS == WAVE_TASKS
    assert [resolver._wave_payload(index)["array_spec"] for index in range(8)] == [
        "0-3%2",
        "4-7%2",
        "8-11%2",
        "12-15%2",
        "16-19%2",
        "20-23%2",
        "24-27%2",
        "28-29%2",
    ]
    state, terminals = _resolve(resolver, root, identities)
    assert state == {
        "status": "ready",
        "campaign_plan_sha256": plan_sha,
        "wave_admission_sha256": None,
        "walltime": "08:00:00",
        "plan_created_at_utc": "2026-07-25T00:00:00Z",
        "wave_index": 0,
        "array_spec": "0-3%2",
        "array_task_ids": [0, 1, 2, 3],
        "seeds": [20260901, 20260902, 20260903, 20260904],
    }
    assert terminals == []


def test_wave_progression_depends_only_on_complete_terminal_presence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolver = _load_resolver()
    root, identities, plan, plan_sha = _make_registry(tmp_path)
    _write_admission(root, plan_sha=plan_sha, wave_index=0)
    _write_submission(
        root,
        plan=plan,
        plan_sha=plan_sha,
        wave_index=0,
        array_job_id="900",
    )

    state, _ = _resolve(resolver, root, identities)
    assert state["status"] == "active"

    def mixed_terminal(*_: object, seed: int, **__: object) -> Path | None:
        if seed > 20260904:
            return None
        outcome = "success" if seed % 2 else "failure"
        return Path(f"/terminal/{seed}/phase2-{outcome}-terminal.json")

    monkeypatch.setattr(resolver, "_terminal_for_seed", mixed_terminal)
    state, terminals = _resolve(resolver, root, identities)
    assert state["status"] == "ready"
    assert state["wave_index"] == 1
    assert state["array_spec"] == "4-7%2"
    assert len(terminals) == 4
    assert any("success" in path.name for path in terminals)
    assert any("failure" in path.name for path in terminals)


def test_submission_prefix_gaps_and_reordered_entries_fail_closed(tmp_path: Path) -> None:
    resolver = _load_resolver()
    root, identities, plan, plan_sha = _make_registry(tmp_path)
    _write_submission(
        root,
        plan=plan,
        plan_sha=plan_sha,
        wave_index=1,
        array_job_id="901",
        admission_sha="9" * 64,
    )
    with pytest.raises(SystemExit, match="gap-free ordered prefix"):
        _resolve(resolver, root, identities)

    submission = root / "campaign-registry" / "submissions" / "array-901.json"
    submission.unlink()
    _write_admission(root, plan_sha=plan_sha, wave_index=0)
    _write_submission(
        root,
        plan=plan,
        plan_sha=plan_sha,
        wave_index=0,
        array_job_id="900",
    )
    submission = root / "campaign-registry" / "submissions" / "array-900.json"
    value = json.loads(submission.read_text(encoding="utf-8"))
    value["entries"] = list(reversed(value["entries"]))
    _write_json(submission, value)
    with pytest.raises(SystemExit, match="submission entries are invalid"):
        _resolve(resolver, root, identities)


def test_complete_registry_orders_terminals_by_wave_not_array_filename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolver = _load_resolver()
    root, identities, plan, plan_sha = _make_registry(tmp_path)
    array_ids = ("90", "100", "11", "82", "13", "74", "15", "66")
    terminal_paths: dict[int, Path] = {}
    for seed in range(20260901, 20260931):
        job = root / f"seed-{seed}" / "attempt-1" / f"job-{seed}"
        job.mkdir(parents=True)
        terminal = job / (
            "phase2-success-terminal.json" if seed % 2 else "phase2-failure-terminal.json"
        )
        terminal.write_text(f"{seed}\n", encoding="utf-8")
        marker = job / ("SUCCESS" if seed % 2 else "FAILED")
        marker.write_text(f"{seed}:marker\n", encoding="utf-8")
        terminal_paths[seed] = terminal
    previous_admission_sha = None
    previous_submission_sha = None
    for wave_index, array_job_id in enumerate(array_ids):
        predecessor_terminals = (
            []
            if wave_index == 0
            else [terminal_paths[20260901 + task] for task in WAVE_TASKS[wave_index - 1]]
        )
        admission_sha = _write_admission(
            root,
            plan_sha=plan_sha,
            wave_index=wave_index,
            predecessor_admission_sha=previous_admission_sha,
            predecessor_submission_sha=previous_submission_sha,
            predecessor_terminals=predecessor_terminals,
        )
        _write_submission(
            root,
            plan=plan,
            plan_sha=plan_sha,
            wave_index=wave_index,
            array_job_id=array_job_id,
            admission_sha=admission_sha,
        )
        submission_path = root / "campaign-registry" / "submissions" / f"array-{array_job_id}.json"
        previous_admission_sha = admission_sha
        previous_submission_sha = hashlib.sha256(submission_path.read_bytes()).hexdigest()

    def terminal(*_: object, seed: int, **__: object) -> Path:
        return terminal_paths[seed]

    monkeypatch.setattr(resolver, "_terminal_for_seed", terminal)
    state, terminals = _resolve(resolver, root, identities)
    assert state["status"] == "complete"
    assert [path.parent.parent.parent.name for path in terminals] == [
        f"seed-{seed}" for seed in range(20260901, 20260931)
    ]


@pytest.mark.parametrize(
    "tamper",
    ("normalized_qos", "normalized_walltime", "raw_qos"),
)
def test_coordinated_scheduler_request_tamper_fails_closed(
    tmp_path: Path,
    tamper: str,
) -> None:
    resolver = _load_resolver()
    root, identities, plan, plan_sha = _make_registry(tmp_path)
    admission_sha = _write_admission(root, plan_sha=plan_sha, wave_index=0)
    _write_submission(
        root,
        plan=plan,
        plan_sha=plan_sha,
        wave_index=0,
        array_job_id="900",
        admission_sha=admission_sha,
    )
    submission = root / "campaign-registry" / "submissions" / "array-900.json"
    value = json.loads(submission.read_text(encoding="utf-8"))
    if tamper == "normalized_qos":
        value["scheduler_request"]["normalized"]["qos"] = "forged_qos"
    elif tamper == "normalized_walltime":
        value["scheduler_request"]["normalized"]["walltime"] = "09:00:00"
    else:
        raw = value["scheduler_request"]["raw_scontrol_record"]
        value["scheduler_request"]["raw_scontrol_record"] = raw.replace(
            "QOS=l20_qos",
            "QOS=forged_qos",
        )
        value["scheduler_request"]["raw_scontrol_sha256"] = hashlib.sha256(
            value["scheduler_request"]["raw_scontrol_record"].encode()
        ).hexdigest()
    value["scheduler_request_sha256"] = hashlib.sha256(
        (
            json.dumps(
                value["scheduler_request"],
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode()
    ).hexdigest()
    _write_json(submission, value)

    with pytest.raises(
        SystemExit,
        match="(?:scheduler request evidence|raw held scontrol evidence) disagrees",
    ):
        _resolve(resolver, root, identities)


def test_predecessor_marker_tamper_breaks_admission_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolver = _load_resolver()
    root, identities, plan, plan_sha = _make_registry(tmp_path)
    admission0_sha = _write_admission(root, plan_sha=plan_sha, wave_index=0)
    _write_submission(
        root,
        plan=plan,
        plan_sha=plan_sha,
        wave_index=0,
        array_job_id="900",
        admission_sha=admission0_sha,
    )
    terminals: dict[int, Path] = {}
    for seed in range(20260901, 20260905):
        job = root / f"seed-{seed}" / "attempt-1" / f"job-{seed}"
        job.mkdir(parents=True)
        terminal = job / (
            "phase2-success-terminal.json" if seed % 2 else "phase2-failure-terminal.json"
        )
        terminal.write_text(f"{seed}\n", encoding="utf-8")
        (job / ("SUCCESS" if seed % 2 else "FAILED")).write_text(
            f"{seed}:marker\n",
            encoding="utf-8",
        )
        terminals[seed] = terminal

    def terminal(*_: object, seed: int, **__: object) -> Path:
        return terminals[seed]

    monkeypatch.setattr(resolver, "_terminal_for_seed", terminal)
    state, terminal_list = _resolve(resolver, root, identities)
    assert state["status"] == "ready" and state["wave_index"] == 1
    staging = root / "campaign-registry" / ".staging" / "admission-wave-1.TEST"
    staging.write_bytes(b"")
    admission1_sha = resolver._materialize_admission(
        staging,
        root=root,
        state=state,
        terminals=terminal_list,
    )
    admission1 = root / "campaign-registry" / "admissions" / "wave-1.json"
    staging.replace(admission1)
    state, _ = _resolve(resolver, root, identities)
    assert state["wave_admission_sha256"] == admission1_sha

    original_admission = admission1.read_bytes()
    admission_value = json.loads(original_admission)
    admission_value["predecessor_submission_sha256"] = "0" * 64
    _write_json(admission1, admission_value)
    with pytest.raises(SystemExit, match="wave-admission registry identity is invalid"):
        _resolve(resolver, root, identities)
    admission1.write_bytes(original_admission)

    marker = terminals[20260901].parent / "SUCCESS"
    marker.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="wave-admission registry identity is invalid"):
        _resolve(resolver, root, identities)
