from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import threading
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SUBMITTER = ROOT / "scripts" / "hpc4" / "submit_phase2_budgeted_end_to_end_once.py"
WALLTIME = "2-00:00:00"
DESIGN = "a" * 64
SCHEDULE = "46e0b0fdc70c507b0325c445068326ba7bc30326d70b65fa33803cc3c876c216"
EXPORT_SPEC = "PATH=/usr/local/bin:/usr/bin:/bin"
EXPORT_SPEC_SHA256 = hashlib.sha256(EXPORT_SPEC.encode()).hexdigest()
SCHEDULER_COMMENT = f"prorm-budgeted:{EXPORT_SPEC_SHA256}"


def _load_submitter() -> ModuleType:
    name = f"_test_budgeted_submitter_{id(object())}"
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
        else:  # pragma: no cover - the submitter owns both call sites
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
        self.comments: dict[str, str | None] = {}
        self.history: dict[str, str] = {}
        self.next_id = 1000
        self.sbatch_calls = 0
        self.release_calls = 0
        self.squeue_calls = 0
        self.sacct_calls = 0
        self.fail_show_once = False
        self.fail_release_once = False
        self.fail_sbatch_after_accept_once = False
        self.add_collision_on_squeue_call: int | None = None
        self.events: list[str] = []
        self.sbatch_arguments: tuple[str, ...] | None = None
        self._lock = threading.Lock()

    @property
    def ledger(self) -> Path:
        return self.project / "runs" / "phase2-budgeted-end-to-end" / DESIGN / "submission-ledger"

    @property
    def job_name(self) -> str:
        return f"prorm-p2-budgeted-{DESIGN[:12]}"

    def add_held(self, job_id: str, *, comment: str | None = SCHEDULER_COMMENT) -> None:
        self.jobs[job_id] = "held"
        self.comments[job_id] = comment

    def add_released(self, job_id: str, *, comment: str | None = SCHEDULER_COMMENT) -> None:
        self.jobs[job_id] = "released"
        self.comments[job_id] = comment

    @staticmethod
    def _option(arguments: tuple[str, ...], prefix: str) -> str:
        return next(item.removeprefix(prefix) for item in arguments if item.startswith(prefix))

    def scontrol_record(self, job_id: str) -> str:
        state = self.jobs[job_id]
        job_state = "PENDING" if state == "held" else "RUNNING"
        reason = "JobHeldUser" if state == "held" else "None"
        fields = [
            f"JobId={job_id}",
            f"ArrayJobId={job_id}",
            "ArrayTaskId=0-4%2",
            "ArrayTaskThrottle=2",
            f"JobName={self.job_name}",
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
            "TRES=billing=8,cpu=8,gres/gpu=1,mem=96G,node=1",
            "TresPerNode=gres/gpu:1",
            f"Command={self.script}",
            f"WorkDir={self.repo}",
            f"JobState={job_state}",
            f"Reason={reason}",
        ]
        comment = self.comments[job_id]
        if comment is not None:
            fields.insert(5, f"Comment={comment}")
        return " ".join(fields)

    def squeue_raw(self) -> str:
        return "".join(
            f"{job_id}|0-4%2|{self.job_name}|{self.user}|gpu-l20|l20_qos\n"
            for job_id in sorted(self.jobs, key=int)
        )

    def sacct_raw(self) -> str:
        return "".join(
            "|".join(
                (
                    job_id,
                    job_id,
                    self.job_name,
                    self.user,
                    "hpc4",
                    "sigroup",
                    "gpu-l20",
                    "l20_qos",
                    state,
                    "2026-07-26T00:00:01",
                    WALLTIME,
                    "billing=8,cpu=8,gres/gpu=1,mem=96G,node=1",
                    "",
                )
            )
            + "\n"
            for job_id, state in sorted(
                self.history.items(),
                key=lambda item: int(item[0]),
            )
        )

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
                self.events.append(f"squeue:{self.squeue_calls}")
                assert self._option(arguments, "--format=") == "%F|%K|%j|%u|%P|%q"
                assert self._option(arguments, "--user=") == self.user
                assert self._option(arguments, "--name=") == self.job_name
                if self.add_collision_on_squeue_call == self.squeue_calls:
                    self.add_held("9999")
                return self.squeue_raw()
            if command == "sacct":
                self.sacct_calls += 1
                self.events.append(f"sacct:{self.sacct_calls}")
                assert "-X" in arguments
                assert "--clusters=hpc4" in arguments
                assert "--starttime=2026-01-01T00:00:00" in arguments
                assert self._option(arguments, "--user=") == self.user
                assert self._option(arguments, "--name=") == self.job_name
                assert "AllocTRES%256" in self._option(arguments, "--format=")
                return self.sacct_raw()
            if command == "sbatch":
                self.sbatch_calls += 1
                self.events.append("sbatch")
                self.sbatch_arguments = arguments
                assert "--hold" in arguments
                assert self._option(arguments, "--array=") == "0-4%2"
                assert self._option(arguments, "--job-name=") == self.job_name
                comment = self._option(arguments, "--comment=")
                assert comment == SCHEDULER_COMMENT
                assert self._option(arguments, "--clusters=") == "hpc4"
                assert self._option(arguments, "--chdir=") == str(self.repo)
                assert arguments[-1] == str(self.script)
                job_id = str(self.next_id)
                self.next_id += 1
                self.add_held(job_id, comment=comment)
                if self.fail_sbatch_after_accept_once:
                    self.fail_sbatch_after_accept_once = False
                    raise RuntimeError("injected ambiguous accepted sbatch failure")
                return f"{job_id};hpc4\n"
            if command == "scontrol" and arguments[1:4] == ("show", "job", "--oneliner"):
                self.events.append("scontrol-show")
                if self.fail_show_once:
                    self.fail_show_once = False
                    raise RuntimeError("injected post-sbatch scontrol crash")
                job_id = arguments[4]
                if job_id not in self.jobs:
                    raise RuntimeError("scheduler no longer has this job")
                return self.scontrol_record(job_id)
            if command == "scontrol" and arguments[1] == "release":
                self.events.append("release")
                job_id = arguments[2]
                assert (self.ledger / "intent.json").is_file()
                assert (self.ledger / "submission.json").is_file()
                if self.fail_release_once:
                    self.fail_release_once = False
                    raise RuntimeError("injected release failure")
                assert self.jobs[job_id] == "held"
                self.jobs[job_id] = "released"
                self.release_calls += 1
                return ""
        raise AssertionError(f"unexpected scheduler command: {arguments!r}")


def _arguments(project: Path, repo: Path, script: Path) -> list[str]:
    return [
        "--project-root",
        str(project),
        "--repo-root",
        str(repo),
        "--design-sha256",
        DESIGN,
        "--base-config-hash",
        "b" * 64,
        "--authorization-sha256",
        "c" * 64,
        "--optimizer-schedule-sha256",
        SCHEDULE,
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
        EXPORT_SPEC,
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
    script = repo / "scripts" / "hpc4" / "phase2_budgeted_end_to_end.sbatch"
    script.parent.mkdir(parents=True)
    project.mkdir()
    script.write_bytes(b"#!/usr/bin/env bash\n")
    scheduler = _FakeScheduler(
        repo=repo.absolute(),
        script=script.absolute(),
        project=project.absolute(),
    )
    monkeypatch.setitem(sys.modules, "fcntl", _FakeFcntl())
    monkeypatch.setattr(module, "_effective_user", lambda: scheduler.user)
    monkeypatch.setattr(module, "_utc_now", lambda: "2026-07-26T00:00:00Z")
    monkeypatch.setattr(module, "_fsync_directory", lambda _path: None)
    monkeypatch.setattr(module, "_run", scheduler.run)
    monkeypatch.setattr(
        module,
        "_run_bytes",
        lambda _arguments, *, name, timeout=60: script.read_bytes(),
    )
    return module, scheduler, _arguments(project, repo, script)


def test_fresh_fixed_five_submission_has_exact_order_and_namespace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, scheduler, arguments = _harness(tmp_path, monkeypatch)
    original_write = module._write_exclusive

    def observed_write(path: Path, value: Any, *, name: str) -> str:
        installed_sha256 = original_write(path, value, name=name)
        scheduler.events.append(f"write:{path.name}")
        return installed_sha256

    monkeypatch.setattr(module, "_write_exclusive", observed_write)

    assert module.main(arguments) == 0

    assert scheduler.sbatch_calls == 1
    assert scheduler.release_calls == 1
    assert scheduler.squeue_calls == 2
    assert scheduler.sacct_calls == 2
    assert scheduler.events.index("sbatch") < scheduler.events.index("scontrol-show")
    assert scheduler.events.index("scontrol-show") < scheduler.events.index("write:intent.json")
    assert scheduler.events.index("write:intent.json") < scheduler.events.index(
        "write:submission.json"
    )
    assert scheduler.events.index("write:submission.json") < scheduler.events.index("squeue:2")
    assert scheduler.events.index("sacct:2") < scheduler.events.index("release")

    assert scheduler.sbatch_arguments is not None
    assert "--array=0-4%2" in scheduler.sbatch_arguments
    assert "--no-requeue" in scheduler.sbatch_arguments
    assert f"--comment={SCHEDULER_COMMENT}" in scheduler.sbatch_arguments
    assert scheduler.sbatch_arguments[-1] == str(
        scheduler.repo / "scripts/hpc4/phase2_budgeted_end_to_end.sbatch"
    )
    intent = json.loads((scheduler.ledger / "intent.json").read_bytes())
    submission = json.loads((scheduler.ledger / "submission.json").read_bytes())
    assert intent["schema_version"] == module.INTENT_SCHEMA
    assert submission["schema_version"] == module.SUBMISSION_SCHEMA
    assert intent["ordered_seeds"] == list(range(20261001, 20261006))
    assert submission["ordered_seeds"] == list(range(20261001, 20261006))
    assert intent["array_spec"] == submission["array_spec"] == "0-4%2"
    assert intent["replacement_array_allowed"] is False
    assert intent["replacement_seed_allowed"] is False
    assert submission["replacement_array_allowed"] is False
    assert submission["replacement_seed_allowed"] is False
    assert not (scheduler.project / "runs" / "phase2-post-recovery-calibration" / DESIGN).exists()


def test_ledger_deep_verifier_binds_exact_source_and_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, scheduler, arguments = _harness(tmp_path, monkeypatch)
    assert module.main(arguments) == 0
    verified = module.verify_submission_ledger(
        scheduler.ledger,
        project_root=scheduler.project,
        repo_root=scheduler.repo,
        design_sha256=DESIGN,
        base_config_hash="b" * 64,
        authorization_sha256="c" * 64,
        optimizer_schedule_sha256=SCHEDULE,
        git_commit="d" * 40,
        image_sha256="e" * 64,
        inventory_sha256="f" * 64,
        overlay_sha256="1" * 64,
        base_file_sha256="2" * 64,
        export_spec_sha256=EXPORT_SPEC_SHA256,
        array_job_id="1000",
        submitter_user=scheduler.user,
    )

    assert verified["status"] == "verified"
    assert verified["array_job_id"] == "1000"
    assert verified["ordered_seeds"] == list(range(20261001, 20261006))


def test_crash_after_sbatch_recovers_held_orphan_without_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, scheduler, arguments = _harness(tmp_path, monkeypatch)
    scheduler.fail_show_once = True

    with pytest.raises(RuntimeError, match="post-sbatch scontrol crash"):
        module.main(arguments)
    assert scheduler.sbatch_calls == 1
    assert scheduler.jobs == {"1000": "held"}
    assert not (scheduler.ledger / "intent.json").exists()
    assert not (scheduler.ledger / "submission.json").exists()

    assert module.main(arguments) == 0
    assert scheduler.sbatch_calls == 1
    assert scheduler.release_calls == 1
    assert scheduler.jobs == {"1000": "released"}


def test_ambiguous_accepted_sbatch_failure_is_recovered_without_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, scheduler, arguments = _harness(tmp_path, monkeypatch)
    scheduler.fail_sbatch_after_accept_once = True

    with pytest.raises(RuntimeError, match="ambiguous accepted sbatch"):
        module.main(arguments)
    assert scheduler.sbatch_calls == 1
    assert scheduler.jobs == {"1000": "held"}
    assert not (scheduler.ledger / "submission.json").exists()

    assert module.main(arguments) == 0
    assert scheduler.sbatch_calls == 1
    assert scheduler.release_calls == 1
    assert scheduler.jobs == {"1000": "released"}


@pytest.mark.parametrize("comment", (None, "prorm-budgeted:" + ("0" * 64)))
def test_unregistered_held_orphan_with_missing_or_wrong_export_comment_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    comment: str | None,
) -> None:
    module, scheduler, arguments = _harness(tmp_path, monkeypatch)
    scheduler.add_held("1234", comment=comment)

    with pytest.raises(ValueError, match="immutable budgeted"):
        module.main(arguments)

    assert scheduler.sbatch_calls == 0
    assert scheduler.release_calls == 0
    assert scheduler.jobs == {"1234": "held"}
    assert not (scheduler.ledger / "intent.json").exists()
    assert not (scheduler.ledger / "submission.json").exists()


def test_crash_after_intent_install_resumes_without_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, scheduler, arguments = _harness(tmp_path, monkeypatch)
    original_write = module._write_exclusive
    fired = False

    def crash_after_intent(path: Path, value: Any, *, name: str) -> str:
        nonlocal fired
        result = original_write(path, value, name=name)
        if path.name == "intent.json" and not fired:
            fired = True
            raise RuntimeError("injected crash after intent fsync")
        return result

    monkeypatch.setattr(module, "_write_exclusive", crash_after_intent)
    with pytest.raises(RuntimeError, match="after intent fsync"):
        module.main(arguments)
    assert (scheduler.ledger / "intent.json").is_file()
    assert not (scheduler.ledger / "submission.json").exists()
    assert scheduler.jobs == {"1000": "held"}

    monkeypatch.setattr(module, "_write_exclusive", original_write)
    assert module.main(arguments) == 0
    assert scheduler.sbatch_calls == 1
    assert scheduler.release_calls == 1


def test_committed_ledger_recovers_release_failure_without_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, scheduler, arguments = _harness(tmp_path, monkeypatch)
    scheduler.fail_release_once = True

    with pytest.raises(RuntimeError, match="injected release failure"):
        module.main(arguments)
    assert (scheduler.ledger / "intent.json").is_file()
    assert (scheduler.ledger / "submission.json").is_file()
    assert scheduler.jobs == {"1000": "held"}

    scheduler.fail_show_once = True
    with pytest.raises(RuntimeError, match="live budgeted array could not be verified"):
        module.main(arguments)
    assert scheduler.release_calls == 0
    assert scheduler.jobs == {"1000": "held"}

    assert module.main(arguments) == 0
    assert scheduler.sbatch_calls == 1
    assert scheduler.release_calls == 1


def test_historical_or_externally_released_orphan_is_never_replaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, scheduler, arguments = _harness(tmp_path, monkeypatch)
    scheduler.history["7777"] = "FAILED"
    with pytest.raises(RuntimeError, match="historical unregistered"):
        module.main(arguments)
    assert scheduler.sbatch_calls == 0

    scheduler.history.clear()
    scheduler.add_released("8888")
    with pytest.raises(RuntimeError, match="externally released"):
        module.main(arguments)
    assert scheduler.sbatch_calls == 0
    assert scheduler.release_calls == 0


def test_second_collision_snapshot_leaves_registered_array_held(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, scheduler, arguments = _harness(tmp_path, monkeypatch)
    scheduler.add_collision_on_squeue_call = 2

    with pytest.raises(RuntimeError, match="another scheduler array appeared"):
        module.main(arguments)

    assert scheduler.sbatch_calls == 1
    assert scheduler.release_calls == 0
    assert scheduler.jobs["1000"] == "held"
    assert (scheduler.ledger / "submission.json").is_file()


def test_concurrent_invocations_submit_once(
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


@pytest.mark.parametrize(
    "raw",
    [
        f"1000|0-4%2|prorm-p2-budgeted-{DESIGN[:12]}|intruder|gpu-l20|l20_qos\n",
        f"1000|0-4%2|prorm-p2-budgeted-{DESIGN[:12]}|tester|gpu-l20|wrong\n",
        f"1000|0-9%2|prorm-p2-budgeted-{DESIGN[:12]}|tester|gpu-l20|l20_qos\n",
        f"1000|0-4%2|prorm-p2-budgeted-{DESIGN[:12]}|tester|gpu-l20\n",
        f"1000|0-4%2|prorm-p2-budgeted-{DESIGN[:12]}|tester|gpu-l20|l20_qos\r\n",
        f"1000|0-4%2|prorm-p2-budgeted-{DESIGN[:12]}|tester|gpu-l20|l20_qos\n\n2000",
    ],
)
def test_squeue_parser_rejects_every_malformed_or_foreign_row(raw: str) -> None:
    module = _load_submitter()
    with pytest.raises(ValueError):
        module._parse_squeue_ids(
            raw,
            expected_name=f"prorm-p2-budgeted-{DESIGN[:12]}",
            expected_user="tester",
        )


def test_squeue_parser_uses_array_root_and_accepts_only_fixed_wave_subsets() -> None:
    module = _load_submitter()
    name = f"prorm-p2-budgeted-{DESIGN[:12]}"
    raw = f"1000|1-4%2|{name}|tester|gpu-l20|l20_qos\n1000|0|{name}|tester|gpu-l20|l20_qos\n"

    assert module._parse_squeue_ids(
        raw,
        expected_name=name,
        expected_user="tester",
    ) == ("1000",)


def test_sacct_parser_preserves_empty_alloc_tres_and_rejects_identity_drift() -> None:
    module = _load_submitter()
    name = f"prorm-p2-budgeted-{DESIGN[:12]}"
    valid = "|".join(
        (
            "2000",
            "1000_4",
            name,
            "tester",
            "hpc4",
            "sigroup",
            "gpu-l20",
            "l20_qos",
            "FAILED",
            "2026-07-26T00:00:01",
            WALLTIME,
            "billing=8,cpu=8,gres/gpu=1,mem=96G,node=1",
            "",
        )
    )
    assert module._parse_sacct_ids(
        valid,
        expected_name=name,
        expected_user="tester",
        expected_walltime=WALLTIME,
    ) == ("1000",)
    cancelled = valid.replace("|FAILED|", "|CANCELLED by 1000|")
    assert module._parse_sacct_ids(
        cancelled,
        expected_name=name,
        expected_user="tester",
        expected_walltime=WALLTIME,
    ) == ("1000",)

    for invalid in (
        valid.replace("cpu=8", "cpu=7"),
        valid.replace(name, f"{name}-other"),
        valid.replace("|tester|", "|intruder|"),
        valid.replace("|hpc4|", "|other|"),
        valid + "|extra",
        valid.replace("|FAILED|", "|FAILED by user|"),
        valid.replace("2000|1000_4", "bad|also-bad"),
    ):
        with pytest.raises(ValueError):
            module._parse_sacct_ids(
                invalid,
                expected_name=name,
                expected_user="tester",
                expected_walltime=WALLTIME,
            )


@pytest.mark.parametrize(
    "old,new",
    [
        ("UserId=tester(1000)", "UserId=intruder(1000)"),
        (
            "TRES=billing=8,cpu=8,gres/gpu=1,mem=96G,node=1",
            "TRES=billing=8,cpu=7,gres/gpu=1,mem=96G,node=1",
        ),
        ("ArrayTaskId=0-4%2", "ArrayTaskId=0-2%2"),
        ("Command=", "Command=/wrong"),
        ("WorkDir=", "WorkDir=/wrong"),
    ],
)
def test_scontrol_requires_full_user_tres_command_workdir_and_array_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    old: str,
    new: str,
) -> None:
    module, scheduler, _ = _harness(tmp_path, monkeypatch)
    scheduler.add_held("1000")
    raw = scheduler.scontrol_record("1000")
    if old == "Command=":
        raw = raw.replace(f"Command={scheduler.script}", f"{new}{scheduler.script}")
    elif old == "WorkDir=":
        raw = raw.replace(f"WorkDir={scheduler.repo}", f"{new}{scheduler.repo}")
    else:
        raw = raw.replace(old, new)

    with pytest.raises(
        ValueError,
        match="immutable budgeted|unexpected scheduler authority",
    ):
        module._parse_scontrol_records(
            raw,
            array_job_id="1000",
            expected_name=scheduler.job_name,
            expected_export_spec_sha256=EXPORT_SPEC_SHA256,
            expected_walltime=WALLTIME,
            expected_command=scheduler.script,
            expected_workdir=scheduler.repo,
            expected_user=scheduler.user,
        )


def test_scontrol_held_evidence_reparses_exactly_and_binds_effective_user(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, scheduler, _ = _harness(tmp_path, monkeypatch)
    scheduler.add_held("1000")
    raw = scheduler.scontrol_record("1000")

    state, evidence = module._parse_scontrol_records(
        raw,
        array_job_id="1000",
        expected_name=scheduler.job_name,
        expected_export_spec_sha256=EXPORT_SPEC_SHA256,
        expected_walltime=WALLTIME,
        expected_command=scheduler.script,
        expected_workdir=scheduler.repo,
        expected_user=scheduler.user,
    )

    assert state == "HELD"
    assert evidence is not None
    assert evidence["schema_version"] == module.SCHEDULER_REQUEST_SCHEMA
    assert evidence["normalized"]["user_id"] == "tester(1000)"
    assert evidence["normalized"]["comment"] == SCHEDULER_COMMENT
    assert evidence["normalized"]["export_spec_sha256"] == EXPORT_SPEC_SHA256
    assert evidence["normalized"]["command"] == str(scheduler.script)
    assert evidence["normalized"]["work_dir"] == str(scheduler.repo)
    assert evidence["normalized"]["tres"] == {
        "billing": "8",
        "cpu": "8",
        "gres/gpu": "1",
        "mem": "96G",
        "node": "1",
    }


@pytest.mark.parametrize(
    "comment_token",
    (
        "",
        "Comment=prorm-budgeted:" + ("0" * 64),
    ),
)
def test_scontrol_rejects_missing_or_wrong_export_identity_comment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    comment_token: str,
) -> None:
    module, scheduler, _ = _harness(tmp_path, monkeypatch)
    scheduler.add_held("1000")
    raw = scheduler.scontrol_record("1000")
    raw = raw.replace(f" Comment={SCHEDULER_COMMENT}", f" {comment_token}").replace("  ", " ")

    with pytest.raises(ValueError, match="immutable budgeted"):
        module._parse_scontrol_records(
            raw,
            array_job_id="1000",
            expected_name=scheduler.job_name,
            expected_export_spec_sha256=EXPORT_SPEC_SHA256,
            expected_walltime=WALLTIME,
            expected_command=scheduler.script,
            expected_workdir=scheduler.repo,
            expected_user=scheduler.user,
        )


def test_scontrol_accepts_released_partial_array_with_allocated_l20_tres(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, scheduler, _ = _harness(tmp_path, monkeypatch)
    scheduler.add_released("1000")
    raw = scheduler.scontrol_record("1000")
    raw = raw.replace("ArrayTaskId=0-4%2", "ArrayTaskId=1-4%2")
    raw = raw.replace(
        "TRES=billing=8,cpu=8,gres/gpu=1,mem=96G,node=1",
        "TRES=billing=8,cpu=8,gres/gpu=1,mem=96G,node=1,gres/gpu:l20=1",
    )

    state, evidence = module._parse_scontrol_records(
        raw,
        array_job_id="1000",
        expected_name=scheduler.job_name,
        expected_export_spec_sha256=EXPORT_SPEC_SHA256,
        expected_walltime=WALLTIME,
        expected_command=scheduler.script,
        expected_workdir=scheduler.repo,
        expected_user=scheduler.user,
    )

    assert state == "ALREADY_RELEASED"
    assert evidence is None


def test_exact_sbatch_repository_path_is_mandatory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, scheduler, arguments = _harness(tmp_path, monkeypatch)
    wrong = scheduler.repo / "scripts" / "hpc4" / "other.sbatch"
    wrong.write_bytes(scheduler.script.read_bytes())
    arguments[-1] = str(wrong)

    with pytest.raises(ValueError, match="exact locked sbatch path"):
        module.main(arguments)
    assert scheduler.sbatch_calls == 0


def test_tampered_no_replacement_policy_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, scheduler, arguments = _harness(tmp_path, monkeypatch)
    assert module.main(arguments) == 0
    submission_path = scheduler.ledger / "submission.json"
    submission = json.loads(submission_path.read_bytes())
    submission["replacement_seed_allowed"] = True
    submission_path.write_bytes(module._canonical_json(submission))

    with pytest.raises(ValueError, match="submission policy"):
        module.main(arguments)
    assert scheduler.sbatch_calls == 1


def test_tampered_ledger_scheduler_comment_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, scheduler, arguments = _harness(tmp_path, monkeypatch)
    assert module.main(arguments) == 0
    submission_path = scheduler.ledger / "submission.json"
    submission = json.loads(submission_path.read_bytes())
    submission["scheduler_request"]["normalized"]["comment"] = "prorm-budgeted:" + ("0" * 64)
    submission_path.write_bytes(module._canonical_json(submission))

    with pytest.raises(ValueError, match="scheduler comment differs"):
        module.main(arguments)
    assert scheduler.sbatch_calls == 1
