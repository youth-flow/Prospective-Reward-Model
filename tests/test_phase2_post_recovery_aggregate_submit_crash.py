from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import threading
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "scripts" / "hpc4" / "submit_phase2_post_recovery_aggregate_attempt.py"


class InjectedCrash(RuntimeError):
    pass


def _load_helper() -> Any:
    name = f"_post_recovery_aggregate_submit_{os.urandom(6).hex()}"
    spec = importlib.util.spec_from_file_location(name, HELPER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _FakeFcntl(types.ModuleType):
    LOCK_EX = 1
    LOCK_UN = 2

    def __init__(self) -> None:
        super().__init__("fcntl")
        self._lock = threading.Lock()

    def flock(self, _descriptor: int, operation: int) -> None:
        if operation == self.LOCK_EX:
            self._lock.acquire()
        elif operation == self.LOCK_UN:
            self._lock.release()
        else:  # pragma: no cover - protects the test double itself
            raise AssertionError(f"unexpected flock operation: {operation}")


def _option(arguments: tuple[str, ...], prefix: str) -> str:
    matches = [item.removeprefix(prefix) for item in arguments if item.startswith(prefix)]
    assert len(matches) == 1
    return matches[0]


class FakeSlurm:
    """Small state machine for the held-submit/ledger/release protocol."""

    def __init__(self, *, script: Path, user: str) -> None:
        self.script = script
        self.committed_script = script.read_bytes()
        self.user = user
        self.jobs: dict[str, dict[str, Any]] = {}
        self.next_job_id = 3000
        self.sbatch_calls = 0
        self.release_calls: list[str] = []
        self._mutex = threading.Lock()
        self._faults: dict[str, int] = {}
        self.failure_query_error = False
        self.failure_query_override: bytes | None = None
        self.controller_override: bytes | None = None
        self.mutate_worktree_after_git: bytes | None = None
        self.sbatch_warning = False
        self.controller_queries: list[str] = []
        self.unavailable_controller_ids: set[str] = set()

    def crash_once(self, point: str, *, occurrence: int = 1) -> None:
        self._faults[point] = occurrence

    def _trip(self, point: str) -> None:
        remaining = self._faults.get(point)
        if remaining is None:
            return
        if remaining == 1:
            del self._faults[point]
            raise InjectedCrash(point)
        self._faults[point] = remaining - 1

    def _history_raw(self, job: dict[str, Any]) -> str:
        state = str(job["state"])
        successful = state == "COMPLETED"
        exit_code = "0:0" if successful else "1:0"
        fields = (
            job["job_id"],
            job["job_id"],
            job["job_name"],
            state,
            exit_code,
            exit_code,
            "hpc4",
            "sigroup",
            job["partition"],
            "1",
            "4",
            "2026-07-25T00:00:00",
            job["walltime"],
            "billing=4,cpu=4,mem=16G,node=1",
            "billing=4,cpu=4,mem=16G,node=1",
        )
        return "|".join(fields) + "\n"

    def _scontrol_raw(self, job: dict[str, Any]) -> str:
        held = bool(job["held"])
        state = "PENDING" if held else str(job.get("live_state", "RUNNING"))
        reason = "JobHeldUser" if held else "Resources"
        return (
            f"JobId={job['job_id']} JobName={job['job_name']} "
            f"UserId={self.user}(1000) Account=sigroup "
            f"Partition={job['partition']} Requeue=0 Restarts=0 "
            "NumNodes=1 NumTasks=1 NumCPUs=4 CPUs/Task=4 "
            f"MinMemoryNode=16G TimeLimit={job['walltime']} "
            f"Command=(null) WorkDir={job['work_dir']} "
            f"Comment={job['comment']} TRES=cpu=4,mem=16G,node=1 "
            f"JobState={state} Reason={reason} BatchFlag=1\n"
        )

    def finish(self, job_id: str, state: str) -> None:
        assert state in {"COMPLETED", "FAILED", "TIMEOUT", "PENDING"}
        with self._mutex:
            job = self.jobs[job_id]
            job["live"] = False
            job["held"] = False
            job["state"] = state

    def add_unknown_history(self, job_name: str, job_id: str = "3999") -> None:
        self.jobs[job_id] = {
            "job_id": job_id,
            "job_name": job_name,
            "partition": "amd",
            "walltime": "01:00:00",
            "script_bytes": self.committed_script,
            "work_dir": os.fspath(self.script.parents[2]),
            "comment": "unregistered",
            "live": False,
            "held": False,
            "state": "COMPLETED",
        }

    def add_unknown_live(self, job_name: str, job_id: str) -> None:
        self.jobs[job_id] = {
            "job_id": job_id,
            "job_name": job_name,
            "partition": "amd",
            "walltime": "01:00:00",
            "script_bytes": self.committed_script,
            "work_dir": os.fspath(self.script.parents[2]),
            "comment": "unregistered",
            "live": True,
            "held": True,
            "state": "PENDING",
        }

    def run(
        self,
        arguments: Any,
        *,
        name: str,
        text: bool = True,
        input_bytes: bytes | None = None,
        require_empty_stderr: bool = False,
    ) -> str | bytes:
        args = tuple(os.fspath(item) for item in arguments)
        command = args[0]
        if command == "git":
            assert not text
            committed = self.committed_script
            if self.mutate_worktree_after_git is not None:
                self.script.write_bytes(self.mutate_worktree_after_git)
                self.mutate_worktree_after_git = None
            return committed
        if command == "squeue":
            job_name = _option(args, "--name=")
            with self._mutex:
                live = sorted(
                    (
                        job_id
                        for job_id, job in self.jobs.items()
                        if job["live"] and job["job_name"] == job_name
                    ),
                    key=int,
                )
            return "".join(f"{job_id}\n" for job_id in live)
        if command == "sacct":
            job_name = _option(args, "--name=")
            with self._mutex:
                rows = [
                    self._history_raw(job)
                    for job in self.jobs.values()
                    if not job["live"] and job["job_name"] == job_name
                ]
            return "".join(rows)
        if command == "sbatch":
            assert not text
            assert input_bytes is not None
            assert args[-1].startswith("--export=")
            with self._mutex:
                self.sbatch_calls += 1
                job_id = str(self.next_job_id)
                self.next_job_id += 1
                self.jobs[job_id] = {
                    "job_id": job_id,
                    "job_name": _option(args, "--job-name="),
                    "partition": _option(args, "--partition="),
                    "walltime": _option(args, "--time="),
                    "script_bytes": input_bytes,
                    "work_dir": _option(args, "--chdir="),
                    "comment": _option(args, "--comment="),
                    "export": _option(args, "--export="),
                    "live": True,
                    "held": True,
                    "state": "PENDING",
                }
            self._trip("after_sbatch")
            if self.sbatch_warning and require_empty_stderr:
                self.sbatch_warning = False
                raise RuntimeError("held aggregate attempt submission failed: warning")
            return f"{job_id};hpc4\n".encode()
        if command == "scontrol" and args[1:4] == ("show", "job", "--oneliner"):
            job_id = args[4]
            with self._mutex:
                raw = self._scontrol_raw(self.jobs[job_id])
            self._trip("after_held_query")
            return raw
        if command == "scontrol" and args[1:3] == ("write", "batch_script"):
            job_id = args[3]
            assert args[4] == "-"
            self.controller_queries.append(job_id)
            if job_id in self.unavailable_controller_ids:
                raise RuntimeError("controller batch script is unavailable")
            with self._mutex:
                raw = self.jobs[job_id]["script_bytes"]
            self._trip("after_controller_readback")
            return self.controller_override if self.controller_override is not None else raw
        if command == "scontrol" and args[1] == "release":
            job_id = args[2]
            with self._mutex:
                self.jobs[job_id]["held"] = False
                self.release_calls.append(job_id)
            self._trip("after_release")
            return ""
        raise AssertionError(f"unexpected {name}: {args!r}")

    def subprocess_run(self, arguments: Any, **_kwargs: Any) -> subprocess.CompletedProcess:
        args = tuple(os.fspath(item) for item in arguments)
        assert args[0] == "sacct" and "-j" in args
        job_id = args[args.index("-j") + 1]
        if self.failure_query_error:
            return subprocess.CompletedProcess(args, 1, b"", b"sacct failed")
        with self._mutex:
            raw = self._history_raw(self.jobs[job_id]).encode("utf-8")
        if self.failure_query_override is not None:
            raw = self.failure_query_override
        return subprocess.CompletedProcess(args, 0, raw, b"")


@dataclass
class Scenario:
    helper: Any
    slurm: FakeSlurm
    project: Path
    repository: Path
    output: Path
    script: Path
    export_spec: str
    argv: list[str]

    @property
    def root(self) -> Path:
        return self.project / "runs" / "phase2-post-recovery-aggregate-attempts" / self.output.name

    @property
    def registry(self) -> Path:
        return self.root / "submission-registry"

    @property
    def job_name(self) -> str:
        identity = f"calibration\0{'d' * 64}\0{self.output}".encode()
        suffix = hashlib.sha256(identity).hexdigest()[:10]
        return f"prorm-p2-post-agg-{'d' * 12}-{suffix}"

    def run(self) -> int:
        return self.helper.main(self.argv)


@pytest.fixture
def scenario(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Scenario:
    helper = _load_helper()
    project = tmp_path / "project"
    repository = tmp_path / "repository"
    (project / "aggregates").mkdir(parents=True)
    script = repository / "scripts" / "hpc4" / "phase2_post_recovery_aggregate.sbatch"
    script.parent.mkdir(parents=True)
    script.write_bytes(b"#!/usr/bin/env bash\nexit 0\n")
    output = project / "aggregates" / "phase2-post-recovery-calibration-aggregate.json"
    export_spec = ",".join(
        (
            f"PRORM_PROJECT_ROOT={project}",
            f"PRORM_REPO_ROOT={repository}",
            f"PRORM_POST_RECOVERY_DESIGN_SHA256={'d' * 64}",
            "PRORM_POST_RECOVERY_ARRAY_JOB_ID=2000",
            f"PRORM_AGGREGATOR_GIT_COMMIT={'c' * 40}",
            f"PRORM_POST_RECOVERY_AGGREGATE_OUTPUT={output}",
            "PRORM_POST_RECOVERY_PILOT_PHASE=calibration",
        )
    )
    argv = [
        "--project-root",
        os.fspath(project),
        "--repo-root",
        os.fspath(repository),
        "--pilot-phase",
        "calibration",
        "--design-sha256",
        "d" * 64,
        "--pilot-array-job-id",
        "2000",
        "--aggregator-git-commit",
        "c" * 40,
        "--output",
        os.fspath(output),
        "--partition",
        "amd",
        "--walltime",
        "01:00:00",
        "--export-spec",
        export_spec,
        "--sbatch-script",
        os.fspath(script),
    ]
    slurm = FakeSlurm(script=script, user="researcher")
    monkeypatch.setattr(helper, "_run", slurm.run)
    monkeypatch.setattr(helper.subprocess, "run", slurm.subprocess_run)
    monkeypatch.setattr(helper, "_effective_user", lambda: "researcher")
    monkeypatch.setattr(helper, "_fsync_directory", lambda _path: None)
    monkeypatch.setitem(sys.modules, "fcntl", _FakeFcntl())
    return Scenario(
        helper=helper,
        slurm=slurm,
        project=project,
        repository=repository,
        output=output,
        script=script,
        export_spec=export_spec,
        argv=argv,
    )


def _post_install_crash(
    monkeypatch: pytest.MonkeyPatch,
    module: Any,
    attribute: str,
    *,
    target_name: str,
) -> None:
    original = getattr(module, attribute)
    fired = False

    def wrapped(path: Path, value: Any, *, name: str) -> str:
        nonlocal fired
        result = original(path, value, name=name)
        if name == target_name and not fired:
            fired = True
            assert path.exists()
            raise InjectedCrash(target_name)
        return result

    monkeypatch.setattr(module, attribute, wrapped)


@pytest.mark.parametrize(
    ("point", "write_name", "expected_sbatch_after_crash"),
    [
        ("after_intent", "aggregate submission intent", 0),
        ("after_sbatch", None, 1),
        ("after_held_query", None, 1),
        ("after_attempt", "aggregate submission attempt ledger", 1),
        ("after_release", None, 1),
    ],
)
def test_initial_attempt_recovers_at_every_durable_boundary(
    scenario: Scenario,
    monkeypatch: pytest.MonkeyPatch,
    point: str,
    write_name: str | None,
    expected_sbatch_after_crash: int,
) -> None:
    if write_name is not None:
        _post_install_crash(
            monkeypatch,
            scenario.helper,
            "_write_exclusive",
            target_name=write_name,
        )
    else:
        scenario.slurm.crash_once(point)

    with pytest.raises(InjectedCrash):
        scenario.run()
    assert scenario.slurm.sbatch_calls == expected_sbatch_after_crash

    assert scenario.run() == 0
    assert scenario.slurm.sbatch_calls == 1
    assert len(scenario.slurm.jobs) == 1
    attempt = scenario.registry / "attempts" / "attempt-0001.json"
    assert attempt.is_file()
    assert scenario.slurm.release_calls == ["3000"]


@pytest.mark.parametrize(
    ("point", "write_attribute", "write_name"),
    [
        ("after_failure_raw", "_write_bytes_exclusive", "aggregate failure raw sacct evidence"),
        ("after_failure_json", "_write_exclusive", "aggregate terminal failure evidence"),
        ("after_retry_sbatch", None, None),
    ],
)
def test_retry_recovers_without_duplicate_cpu_attempt(
    scenario: Scenario,
    monkeypatch: pytest.MonkeyPatch,
    point: str,
    write_attribute: str | None,
    write_name: str | None,
) -> None:
    assert scenario.run() == 0
    scenario.slurm.finish("3000", "FAILED")
    if write_attribute is not None and write_name is not None:
        _post_install_crash(
            monkeypatch,
            scenario.helper,
            write_attribute,
            target_name=write_name,
        )
    else:
        scenario.slurm.crash_once("after_sbatch", occurrence=1)

    with pytest.raises(InjectedCrash):
        scenario.run()
    assert scenario.run() == 0

    assert scenario.slurm.sbatch_calls == 2
    assert sorted(scenario.slurm.jobs) == ["3000", "3001"]
    assert scenario.slurm.release_calls == ["3000", "3001"]
    assert (scenario.registry / "attempts" / "attempt-0002.json").is_file()
    assert (scenario.registry / "failures" / "job-3000.sacct.psv").is_file()
    assert (scenario.registry / "failures" / "job-3000.json").is_file()


@pytest.mark.parametrize("after_failure", [False, True])
def test_two_concurrent_callers_submit_exactly_one_authorized_attempt(
    scenario: Scenario,
    after_failure: bool,
) -> None:
    if after_failure:
        assert scenario.run() == 0
        scenario.slurm.finish("3000", "FAILED")

    barrier = threading.Barrier(3)
    outcomes: list[object] = []

    def invoke() -> None:
        barrier.wait()
        try:
            outcomes.append(scenario.run())
        except BaseException as error:  # pragma: no cover - asserted below
            outcomes.append(error)

    threads = [threading.Thread(target=invoke) for _ in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive()

    assert outcomes == [0, 0]
    assert scenario.slurm.sbatch_calls == (2 if after_failure else 1)
    assert sorted(scenario.slurm.jobs) == (["3000", "3001"] if after_failure else ["3000"])


def test_success_and_nonterminal_history_never_authorize_replacement(
    scenario: Scenario,
) -> None:
    assert scenario.run() == 0
    scenario.slurm.finish("3000", "COMPLETED")
    assert scenario.run() == 0
    assert scenario.slurm.sbatch_calls == 1

    scenario.slurm.jobs["3000"]["state"] = "PENDING"
    with pytest.raises(RuntimeError, match="not exact success or terminal failure"):
        scenario.run()
    assert scenario.slurm.sbatch_calls == 1


def test_unknown_scheduler_history_and_multiple_live_jobs_fail_closed(
    scenario: Scenario,
) -> None:
    scenario.slurm.add_unknown_history(scenario.job_name)
    with pytest.raises(RuntimeError, match="historical unregistered"):
        scenario.run()
    assert scenario.slurm.sbatch_calls == 0

    scenario.slurm.jobs.clear()
    scenario.slurm.add_unknown_live(scenario.job_name, "3998")
    scenario.slurm.add_unknown_live(scenario.job_name, "3999")
    with pytest.raises(RuntimeError, match="multiple unregistered"):
        scenario.run()
    assert scenario.slurm.sbatch_calls == 0


def test_registry_gap_tamper_and_truncated_precreation_are_never_repaired(
    scenario: Scenario,
) -> None:
    scenario.registry.mkdir(parents=True)
    intent_path = scenario.registry / "intent.json"
    intent_path.write_bytes(b"{")
    with pytest.raises(ValueError, match="strict JSON"):
        scenario.run()
    assert intent_path.read_bytes() == b"{"
    assert scenario.slurm.sbatch_calls == 0

    intent_path.unlink()
    assert scenario.run() == 0
    attempt_one = scenario.registry / "attempts" / "attempt-0001.json"
    attempt_two = attempt_one.with_name("attempt-0002.json")
    attempt_one.rename(attempt_two)
    before = attempt_two.read_bytes()
    with pytest.raises(ValueError, match="non-contiguous"):
        scenario.run()
    assert attempt_two.read_bytes() == before
    assert scenario.slurm.sbatch_calls == 1


def test_hidden_staging_residue_is_ignored_and_roots_remain_distinct(
    scenario: Scenario,
) -> None:
    scenario.registry.mkdir(parents=True)
    (scenario.registry / ".intent.json.staged-crash").write_bytes(b"partial")
    assert scenario.run() == 0
    intent = json.loads((scenario.registry / "intent.json").read_text(encoding="utf-8"))
    assert Path(intent["project_root"]) == scenario.project
    assert Path(intent["repository_root"]) == scenario.repository
    assert scenario.project != scenario.repository


def test_wrong_export_and_script_outside_repository_fail_before_sbatch(
    scenario: Scenario,
    tmp_path: Path,
) -> None:
    bad_export = list(scenario.argv)
    export_index = bad_export.index("--export-spec") + 1
    bad_export[export_index] = scenario.export_spec.replace(
        "PRORM_POST_RECOVERY_ARRAY_JOB_ID=2000",
        "PRORM_POST_RECOVERY_ARRAY_JOB_ID=9999",
    )
    with pytest.raises(ValueError, match="export"):
        scenario.helper.main(bad_export)
    assert scenario.slurm.sbatch_calls == 0

    external = tmp_path / "outside.sbatch"
    external.write_bytes(scenario.script.read_bytes())
    bad_script = list(scenario.argv)
    script_index = bad_script.index("--sbatch-script") + 1
    bad_script[script_index] = os.fspath(external)
    with pytest.raises(ValueError):
        scenario.helper.main(bad_script)
    assert scenario.slurm.sbatch_calls == 0


def test_failure_raw_query_error_or_mismatch_blocks_retry(
    scenario: Scenario,
) -> None:
    assert scenario.run() == 0
    scenario.slurm.finish("3000", "FAILED")
    scenario.slurm.failure_query_error = True
    with pytest.raises(RuntimeError, match="sacct query failed"):
        scenario.run()
    assert scenario.slurm.sbatch_calls == 1

    scenario.slurm.failure_query_error = False
    raw = scenario.slurm._history_raw(scenario.slurm.jobs["3000"])
    scenario.slurm.failure_query_override = raw.replace("|FAILED|", "|TIMEOUT|").encode()
    with pytest.raises(RuntimeError, match="differs between locked"):
        scenario.run()
    assert scenario.slurm.sbatch_calls == 1


def test_verified_retry_registry_binds_complete_failure_chain(
    scenario: Scenario,
) -> None:
    assert scenario.run() == 0
    scenario.slurm.finish("3000", "FAILED")
    assert scenario.run() == 0
    intent_sha = hashlib.sha256((scenario.registry / "intent.json").read_bytes()).hexdigest()
    workload_sha = hashlib.sha256(scenario.export_spec.encode()).hexdigest()

    verified = scenario.helper.verify_aggregate_submission_registry(
        scenario.registry,
        expected_intent_sha256=intent_sha,
        expected_attempt_index=2,
        expected_job_id="3001",
        expected_project_root=scenario.project,
        expected_repository_root=scenario.repository,
        expected_output=scenario.output,
        expected_workload_export_sha256=workload_sha,
    )

    assert verified["attempt_index"] == 2
    assert verified["slurm_job_id"] == "3001"
    assert [entry["slurm_job_id"] for entry in verified["failure_entries"]] == ["3000"]


def test_mutated_worktree_after_git_read_still_submits_committed_bytes(
    scenario: Scenario,
) -> None:
    committed = scenario.slurm.committed_script
    scenario.slurm.mutate_worktree_after_git = b"#!/usr/bin/env bash\nexit 99\n"

    assert scenario.run() == 0

    assert scenario.script.read_bytes() != committed
    assert scenario.slurm.jobs["3000"]["script_bytes"] == committed
    assert (scenario.registry / scenario.helper.SCRIPT_EVIDENCE_FILENAME).read_bytes() == committed
    intent = json.loads((scenario.registry / "intent.json").read_text(encoding="utf-8"))
    script = intent["sbatch_script"]
    assert script["sha256"] == hashlib.sha256(committed).hexdigest()
    assert script["git_blob_sha1"] == scenario.helper._git_blob_sha1(committed)
    assert script["size_bytes"] == len(committed)


def test_controller_mismatch_fails_before_ledger_and_release(
    scenario: Scenario,
) -> None:
    scenario.slurm.controller_override = b"#!/usr/bin/env bash\nexit 88\n"

    with pytest.raises(RuntimeError, match="controller batch script differs"):
        scenario.run()

    assert scenario.slurm.sbatch_calls == 1
    assert scenario.slurm.release_calls == []
    assert not (scenario.registry / "attempts" / "attempt-0001.json").exists()
    assert not (
        scenario.registry / scenario.helper.CONTROLLER_READBACK_DIRECTORY / "attempt-0001.sbatch"
    ).exists()


def test_success_with_stderr_is_adopted_without_duplicate_submission(
    scenario: Scenario,
) -> None:
    scenario.slurm.sbatch_warning = True

    with pytest.raises(RuntimeError, match="warning"):
        scenario.run()

    assert scenario.slurm.sbatch_calls == 1
    assert scenario.slurm.release_calls == []
    assert scenario.run() == 0
    assert scenario.slurm.sbatch_calls == 1
    assert scenario.slurm.release_calls == ["3000"]


def test_terminal_prior_controller_can_be_purged_before_retry(
    scenario: Scenario,
) -> None:
    assert scenario.run() == 0
    initial_queries = scenario.slurm.controller_queries.count("3000")
    scenario.slurm.finish("3000", "FAILED")
    scenario.slurm.unavailable_controller_ids.add("3000")

    assert scenario.run() == 0

    assert scenario.slurm.controller_queries.count("3000") == initial_queries
    assert scenario.slurm.controller_queries.count("3001") == 2
    assert scenario.slurm.sbatch_calls == 2


def test_legacy_v1_intent_is_rejected_without_resubmission(
    scenario: Scenario,
) -> None:
    assert scenario.run() == 0
    intent_path = scenario.registry / "intent.json"
    intent = json.loads(intent_path.read_text(encoding="utf-8"))
    intent["schema_version"] = "prorm-phase2-post-recovery-aggregate-submit-intent/v1"
    intent_path.write_bytes(scenario.helper._canonical_json(intent))

    with pytest.raises(ValueError, match="intent identity differs"):
        scenario.run()

    assert scenario.slurm.sbatch_calls == 1


@pytest.mark.skipif(not hasattr(os, "link"), reason="hard links unavailable")
def test_exclusive_install_pre_link_crash_leaves_no_visible_file(
    scenario: Scenario,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = scenario.project / "pre-install.json"
    original_link = scenario.helper.os.link

    def crash_before_link(*_args: Any, **_kwargs: Any) -> None:
        raise InjectedCrash("before-link")

    monkeypatch.setattr(scenario.helper.os, "link", crash_before_link)
    with pytest.raises(InjectedCrash, match="before-link"):
        scenario.helper._write_bytes_exclusive(
            destination,
            b"exact\n",
            name="pre-install test",
        )
    assert not destination.exists()
    assert list(destination.parent.glob(".staged-*"))

    monkeypatch.setattr(scenario.helper.os, "link", original_link)
    scenario.helper._write_bytes_exclusive(
        destination,
        b"exact\n",
        name="pre-install test",
    )
    assert destination.read_bytes() == b"exact\n"


@pytest.mark.skipif(not hasattr(os, "link"), reason="hard links unavailable")
def test_exclusive_install_post_link_crash_preserves_exact_no_clobber_file(
    scenario: Scenario,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = scenario.project / "post-install.json"
    original_link = scenario.helper.os.link

    def crash_after_link(source: Any, target: Any, **kwargs: Any) -> None:
        original_link(source, target, **kwargs)
        assert Path(target).read_bytes() == b"exact\n"
        raise InjectedCrash("after-link")

    monkeypatch.setattr(scenario.helper.os, "link", crash_after_link)
    with pytest.raises(InjectedCrash, match="after-link"):
        scenario.helper._write_bytes_exclusive(
            destination,
            b"exact\n",
            name="post-install test",
        )
    assert destination.read_bytes() == b"exact\n"

    monkeypatch.setattr(scenario.helper.os, "link", original_link)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        scenario.helper._write_bytes_exclusive(
            destination,
            b"different\n",
            name="post-install test",
        )
    assert destination.read_bytes() == b"exact\n"
