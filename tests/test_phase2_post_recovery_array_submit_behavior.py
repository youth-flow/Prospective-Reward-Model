from __future__ import annotations

import importlib.util
import subprocess
import sys
import threading
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
HPC4 = ROOT / "scripts" / "hpc4"
SUBMITTER = HPC4 / "submit_phase2_post_recovery_array_once.py"
CALIBRATION_ENTRYPOINT = HPC4 / "submit_phase2_post_recovery_calibration.sh"
GENERIC_ENTRYPOINT = HPC4 / "submit_phase2_post_recovery_pilot.sh"
CONFIRMATORY_ENTRYPOINT = HPC4 / "submit_phase2_confirmatory.sh"
WALLTIME = "2-00:00:00"


def _load_submitter() -> ModuleType:
    name = "_test_phase2_post_recovery_array_submitter"
    spec = importlib.util.spec_from_file_location(name, SUBMITTER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _FakeFcntl(ModuleType):
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
        else:  # pragma: no cover - the submitter owns the only call sites
            raise AssertionError(f"unexpected flock operation: {operation}")


class _FakeScheduler:
    def __init__(
        self,
        *,
        repo: Path,
        script: Path,
        project: Path,
        user: str = "tester",
    ) -> None:
        self.repo = repo
        self.script = script
        self.project = project
        self.user = user
        self.jobs: dict[str, str] = {}
        self.history: set[str] = set()
        self.next_id = 1000
        self.sbatch_calls = 0
        self.release_calls = 0
        self.squeue_calls = 0
        self.sacct_formats: list[str] = []
        self.fail_next_release = False
        self.add_extra_on_squeue_call: int | None = None
        self._lock = threading.Lock()

    @property
    def registry(self) -> Path:
        return (
            self.project
            / "runs"
            / "phase2-post-recovery-calibration"
            / ("a" * 64)
            / "submission-registry"
        )

    @staticmethod
    def _option(arguments: tuple[str, ...], prefix: str) -> str:
        return next(value.removeprefix(prefix) for value in arguments if value.startswith(prefix))

    def add_held(self, job_id: str) -> None:
        self.jobs[job_id] = "held"

    def add_history(self, job_id: str) -> None:
        self.history.add(job_id)

    def _job_name(self) -> str:
        return f"prorm-p2-post-calibration-{'a' * 12}"

    def _scontrol(self, job_id: str) -> str:
        state = self.jobs[job_id]
        reason = "JobHeldUser" if state == "held" else "Priority"
        return " ".join(
            (
                f"JobId={job_id}",
                f"ArrayJobId={job_id}",
                "ArrayTaskId=0-2%2",
                "ArrayTaskThrottle=2",
                f"JobName={self._job_name()}",
                f"UserId={self.user}(1000)",
                "Account=sigroup",
                "Partition=gpu-l20",
                "QOS=l20_qos",
                "Requeue=0",
                "Restarts=0",
                "NumNodes=1-1",
                "NumTasks=1",
                "NumCPUs=8",
                "CPUs/Task=8",
                "MinMemoryNode=96G",
                f"TimeLimit={WALLTIME}",
                "TRES=cpu=8,mem=96G,node=1,billing=8,gres/gpu=1",
                "TresPerNode=gres/gpu:1",
                f"Command={self.script}",
                f"WorkDir={self.repo}",
                "JobState=PENDING",
                f"Reason={reason}",
            )
        )

    def _sacct(self) -> str:
        rows = []
        for job_id in sorted(self.history, key=int):
            rows.append(
                "|".join(
                    (
                        job_id,
                        job_id,
                        self._job_name(),
                        "FAILED",
                        "2026-07-25T00:00:01",
                        WALLTIME,
                        "billing=8,cpu=8,gres/gpu=1,mem=96G,node=1",
                        "",
                    )
                )
            )
        return "\n".join(rows)

    def run(
        self,
        raw_arguments: tuple[str, ...] | list[str],
        *,
        name: str,
        timeout: int = 60,
    ) -> str:
        del name, timeout
        arguments = tuple(raw_arguments)
        with self._lock:
            command = arguments[0]
            if command == "squeue":
                self.squeue_calls += 1
                if self.add_extra_on_squeue_call == self.squeue_calls:
                    self.add_held("9999")
                return "\n".join(sorted(self.jobs, key=int))
            if command == "sacct":
                self.sacct_formats.append(
                    next(value for value in arguments if value.startswith("--format="))
                )
                return self._sacct()
            if command == "sbatch":
                self.sbatch_calls += 1
                assert "--hold" in arguments
                assert self._option(arguments, "--array=") == "0-2%2"
                assert self._option(arguments, "--job-name=") == self._job_name()
                job_id = str(self.next_id)
                self.next_id += 1
                self.add_held(job_id)
                return f"{job_id};hpc4\n"
            if command == "scontrol" and arguments[1:4] == ("show", "job", "--oneliner"):
                job_id = arguments[4]
                if job_id not in self.jobs:
                    raise RuntimeError("registered pilot array query failed")
                return self._scontrol(job_id)
            if command == "scontrol" and arguments[1] == "release":
                job_id = arguments[2]
                assert (self.registry / "submission.json").is_file()
                if self.fail_next_release:
                    self.fail_next_release = False
                    raise RuntimeError("injected release failure")
                self.release_calls += 1
                self.jobs[job_id] = "released"
                return ""
        raise AssertionError(f"unexpected scheduler command: {arguments}")


def _arguments(project: Path, repo: Path, script: Path) -> list[str]:
    digest = "a" * 64
    return [
        "--project-root",
        str(project),
        "--repo-root",
        str(repo),
        "--pilot-phase",
        "calibration",
        "--design-sha256",
        digest,
        "--base-config-hash",
        "b" * 64,
        "--authorization-sha256",
        "c" * 64,
        "--optimizer-schedule-sha256",
        "46e0b0fdc70c507b0325c445068326ba7bc30326d70b65fa33803cc3c876c216",
        "--git-commit",
        "d" * 40,
        "--image-sha256",
        "e" * 64,
        "--inventory-sha256",
        "f" * 64,
        "--overlay-sha256",
        "1" * 64,
        "--base-file-sha256",
        "2" * 64,
        "--walltime",
        WALLTIME,
        "--export-spec",
        "PATH=/usr/local/bin:/usr/bin:/bin",
        "--sbatch-script",
        str(script),
    ]


def _harness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[ModuleType, _FakeScheduler, list[str]]:
    module = _load_submitter()
    repo = tmp_path / "repo"
    project = tmp_path / "project"
    script = repo / "scripts" / "hpc4" / "phase2_post_recovery_calibration.sbatch"
    script.parent.mkdir(parents=True)
    project.mkdir()
    script.write_bytes(b"#!/usr/bin/env bash\n")
    scheduler = _FakeScheduler(repo=repo, script=script, project=project)
    monkeypatch.setitem(sys.modules, "fcntl", _FakeFcntl())
    monkeypatch.setattr(module, "_effective_user", lambda: scheduler.user)
    monkeypatch.setattr(module, "_utc_now", lambda: "2026-07-25T00:00:00Z")
    monkeypatch.setattr(module, "_fsync_directory", lambda _path: None)
    monkeypatch.setattr(module, "_run", scheduler.run)
    monkeypatch.setattr(
        module,
        "_run_bytes",
        lambda _arguments, *, name, timeout=60: script.read_bytes(),
    )
    return module, scheduler, _arguments(project, repo, script)


def test_fresh_submission_commits_ledger_before_releasing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, scheduler, arguments = _harness(tmp_path, monkeypatch)

    assert module.main(arguments) == 0

    assert scheduler.sbatch_calls == 1
    assert scheduler.release_calls == 1
    assert scheduler.squeue_calls == 2
    assert scheduler.sacct_formats
    assert all(
        "JobName%128" in value and "AllocTRES%256" in value for value in scheduler.sacct_formats
    )
    assert (scheduler.registry / "intent.json").is_file()
    assert (scheduler.registry / "submission.json").is_file()


def test_orphan_held_submission_is_adopted_without_another_sbatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, scheduler, arguments = _harness(tmp_path, monkeypatch)
    scheduler.add_held("1234")

    assert module.main(arguments) == 0

    assert scheduler.sbatch_calls == 0
    assert scheduler.release_calls == 1


def test_committed_ledger_recovers_after_release_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, scheduler, arguments = _harness(tmp_path, monkeypatch)
    scheduler.fail_next_release = True

    with pytest.raises(RuntimeError, match="injected release failure"):
        module.main(arguments)
    assert (scheduler.registry / "submission.json").is_file()

    assert module.main(arguments) == 0
    assert scheduler.sbatch_calls == 1
    assert scheduler.release_calls == 1


def test_history_without_ledger_forbids_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, scheduler, arguments = _harness(tmp_path, monkeypatch)
    scheduler.add_history("7777")

    with pytest.raises(RuntimeError, match="historical unregistered"):
        module.main(arguments)

    assert scheduler.sbatch_calls == 0
    assert scheduler.release_calls == 0


def test_extra_id_in_second_snapshot_keeps_registered_array_held(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, scheduler, arguments = _harness(tmp_path, monkeypatch)
    scheduler.add_extra_on_squeue_call = 2

    with pytest.raises(RuntimeError, match="another scheduler array appeared"):
        module.main(arguments)

    assert scheduler.sbatch_calls == 1
    assert scheduler.release_calls == 0
    assert (scheduler.registry / "submission.json").is_file()
    assert scheduler.jobs["1000"] == "held"


@pytest.mark.parametrize(
    "entrypoint_order",
    (("calibration", "generic"), ("generic", "calibration")),
)
def test_both_entrypoint_orders_resolve_to_one_submission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entrypoint_order: tuple[str, str],
) -> None:
    module, scheduler, arguments = _harness(tmp_path, monkeypatch)

    for entrypoint in entrypoint_order:
        assert entrypoint in {"calibration", "generic"}
        assert module.main(arguments) == 0

    assert scheduler.sbatch_calls == 1
    assert scheduler.release_calls == 1


def test_concurrent_entrypoints_submit_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, scheduler, arguments = _harness(tmp_path, monkeypatch)
    outcomes: list[int] = []
    failures: list[BaseException] = []

    def invoke() -> None:
        try:
            outcomes.append(module.main(arguments))
        except BaseException as error:  # pragma: no cover - asserted below
            failures.append(error)

    threads = [threading.Thread(target=invoke) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not any(thread.is_alive() for thread in threads)
    assert failures == []
    assert outcomes == [0, 0]
    assert scheduler.sbatch_calls == 1
    assert scheduler.release_calls == 1


def test_calibration_entrypoint_execs_the_generic_locked_path() -> None:
    calibration = CALIBRATION_ENTRYPOINT.read_text(encoding="utf-8")
    generic = GENERIC_ENTRYPOINT.read_text(encoding="utf-8")

    delegated = 'exec bash "${generic_submit}" "${overlay}" "${authorization}" "${walltime}"'
    assert delegated in calibration
    assert "configs/common_beta_post_recovery_calibration.yaml" in calibration
    for source in (
        "submit_phase2_post_recovery_calibration.sh",
        "submit_phase2_post_recovery_pilot.sh",
        "phase2_post_recovery_output.py",
        "validate_phase2_recovery_authorization.py",
        "validate_phase2_post_recovery_output.py",
    ):
        assert source in generic


def test_formal_history_parser_preserves_an_empty_alloc_tres_field() -> None:
    submit = CONFIRMATORY_ENTRYPOINT.read_text(encoding="utf-8")
    marker = "record, expected_name, expected_walltime = sys.argv[1:]"
    marker_at = submit.index(marker)
    parser_start = submit.rfind("import re\n", 0, marker_at)
    parser_end = submit.index("\nPY\n", marker_at)
    parser = submit[parser_start:parser_end]
    name = "prorm-p2-fixed-wave"
    record = "|".join(
        (
            "1234",
            "1234",
            name,
            "FAILED",
            "2026-07-25T00:00:01",
            WALLTIME,
            "billing=8,cpu=8,gres/gpu=1,mem=96G,node=1",
            "",
        )
    )

    completed = subprocess.run(
        (sys.executable, "-I", "-S", "-", record, name, WALLTIME),
        input=parser,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "1234"
