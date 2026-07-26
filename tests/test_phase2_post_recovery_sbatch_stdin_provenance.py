from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SUBMIT_SUPPORT_FILE = ROOT / "tests" / "test_phase2_post_recovery_aggregate_submit_crash.py"
FINAL_SUPPORT_FILE = ROOT / "tests" / "test_phase2_post_recovery_crash_atomicity.py"


def _load_support(path: Path, stem: str) -> Any:
    name = f"_{stem}_{os.urandom(6).hex()}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SUBMIT_SUPPORT = _load_support(SUBMIT_SUPPORT_FILE, "stdin_provenance_submit_support")
FINAL_SUPPORT = _load_support(FINAL_SUPPORT_FILE, "stdin_provenance_final_support")


@pytest.fixture
def submission_scenario(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Any:
    return SUBMIT_SUPPORT.scenario.__wrapped__(tmp_path, monkeypatch)


@pytest.fixture
def built_final_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Any:
    return FINAL_SUPPORT.built_attempt.__wrapped__(tmp_path, monkeypatch)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def test_binary_run_forwards_exact_committed_bytes_as_subprocess_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper = SUBMIT_SUPPORT._load_helper()
    committed = b"#!/usr/bin/env bash\nprintf 'committed\\n'\n"
    observed: dict[str, Any] = {}

    def fake_subprocess_run(arguments: Any, **kwargs: Any) -> Any:
        observed["arguments"] = tuple(arguments)
        observed["kwargs"] = dict(kwargs)
        return subprocess.CompletedProcess(arguments, 0, b"3000;hpc4\n", b"")

    monkeypatch.setattr(helper.subprocess, "run", fake_subprocess_run)
    result = helper._run(
        ("sbatch", "--parsable", "--hold"),
        name="held aggregate attempt submission",
        text=False,
        input_bytes=committed,
        require_empty_stderr=True,
    )

    assert result == b"3000;hpc4\n"
    assert observed["arguments"] == ("sbatch", "--parsable", "--hold")
    assert observed["kwargs"] == {
        "check": False,
        "capture_output": True,
        "text": False,
        "input": committed,
        "timeout": 60,
    }


def test_binary_run_rejects_nonempty_stderr_when_exact_bytes_are_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper = SUBMIT_SUPPORT._load_helper()

    def fake_subprocess_run(arguments: Any, **kwargs: Any) -> Any:
        return subprocess.CompletedProcess(arguments, 0, b"committed bytes\n", b"warning\n")

    monkeypatch.setattr(helper.subprocess, "run", fake_subprocess_run)
    with pytest.raises(RuntimeError, match="committed Git blob query failed: warning"):
        helper._run(
            ("git", "cat-file", "blob", "commit:path"),
            name="committed Git blob query",
            text=False,
            require_empty_stderr=True,
        )


def test_fresh_submission_is_bound_to_git_blob_despite_worktree_rewrite(
    submission_scenario: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = submission_scenario
    helper = scenario.helper
    committed = scenario.slurm.committed_script
    rewritten = b"#!/usr/bin/env bash\nprintf 'uncommitted rewrite\\n'\n"
    scenario.slurm.mutate_worktree_after_git = rewritten
    sbatch_calls: list[tuple[tuple[str, ...], bytes | None]] = []
    controller_queries: list[tuple[str, ...]] = []
    git_requires_empty_stderr: list[bool] = []
    original_run = scenario.slurm.run

    def tracking_run(
        arguments: Any,
        *,
        name: str,
        text: bool = True,
        input_bytes: bytes | None = None,
        require_empty_stderr: bool = False,
    ) -> str | bytes:
        command = tuple(os.fspath(item) for item in arguments)
        if command[0] == "sbatch":
            sbatch_calls.append((command, input_bytes))
        if command[0] == "git":
            git_requires_empty_stderr.append(require_empty_stderr)
        if command[:3] == ("scontrol", "write", "batch_script"):
            controller_queries.append(command)
        return original_run(
            arguments,
            name=name,
            text=text,
            input_bytes=input_bytes,
            require_empty_stderr=require_empty_stderr,
        )

    monkeypatch.setattr(helper, "_run", tracking_run)
    assert scenario.run() == 0

    assert scenario.script.read_bytes() == rewritten
    assert len(sbatch_calls) == 1
    command, submitted_stdin = sbatch_calls[0]
    assert submitted_stdin == committed
    assert all(argument.startswith("--") for argument in command[1:])
    assert os.fspath(scenario.script) not in command
    assert scenario.slurm.jobs["3000"]["script_bytes"] == committed

    intent = _json(scenario.registry / "intent.json")
    relative = scenario.script.relative_to(scenario.repository).as_posix()
    assert intent["sbatch_script"] == {
        "repo_relative_path": relative,
        "sha256": _sha256(committed),
        "git_blob_sha1": helper._git_blob_sha1(committed),
        "size_bytes": len(committed),
        "git_object": f"{'c' * 40}:{relative}",
        "evidence_filename": helper.SCRIPT_EVIDENCE_FILENAME,
        "transport": helper.SCRIPT_TRANSPORT,
    }
    assert (scenario.registry / helper.SCRIPT_EVIDENCE_FILENAME).read_bytes() == committed

    attempt = _json(scenario.registry / "attempts" / "attempt-0001.json")
    binding = attempt["batch_script"]
    readback_relative = helper._controller_readback_relative(1)
    readback = scenario.registry.joinpath(*readback_relative.split("/"))
    assert binding == {
        "schema_version": helper.SCRIPT_BINDING_SCHEMA,
        "transport": helper.SCRIPT_TRANSPORT,
        "submission_command": list(command),
        "stdin_sha256": _sha256(committed),
        "stdin_size_bytes": len(committed),
        "controller_readback": {
            "query": list(helper._controller_readback_query("3000")),
            "relative_path": readback_relative,
            "sha256": _sha256(committed),
            "size_bytes": len(committed),
        },
        "controller_matches_committed": True,
    }
    assert readback.read_bytes() == committed
    assert controller_queries == [
        helper._controller_readback_query("3000"),
        helper._controller_readback_query("3000"),
    ]
    assert git_requires_empty_stderr == [True]


def test_registry_verifier_requires_empty_stderr_from_git_blob_query(
    submission_scenario: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = submission_scenario
    assert scenario.run() == 0
    intent_sha256 = _sha256((scenario.registry / "intent.json").read_bytes())
    workload_sha256 = _sha256(scenario.export_spec.encode())
    git_requires_empty_stderr: list[bool] = []
    original_run = scenario.slurm.run

    def tracking_run(
        arguments: Any,
        *,
        name: str,
        text: bool = True,
        input_bytes: bytes | None = None,
        require_empty_stderr: bool = False,
    ) -> str | bytes:
        command = tuple(os.fspath(item) for item in arguments)
        if command[0] == "git":
            git_requires_empty_stderr.append(require_empty_stderr)
        return original_run(
            arguments,
            name=name,
            text=text,
            input_bytes=input_bytes,
            require_empty_stderr=require_empty_stderr,
        )

    monkeypatch.setattr(scenario.helper, "_run", tracking_run)
    verified = scenario.helper.verify_aggregate_submission_registry(
        scenario.registry,
        expected_intent_sha256=intent_sha256,
        expected_attempt_index=1,
        expected_job_id="3000",
        expected_project_root=scenario.project,
        expected_repository_root=scenario.repository,
        expected_output=scenario.output,
        expected_workload_export_sha256=workload_sha256,
    )

    assert verified["slurm_job_id"] == "3000"
    assert git_requires_empty_stderr == [True]


def test_controller_mismatch_creates_neither_attempt_ledger_nor_release(
    submission_scenario: Any,
) -> None:
    scenario = submission_scenario
    scenario.slurm.controller_override = b"#!/usr/bin/env bash\nexit 99\n"

    with pytest.raises(
        RuntimeError,
        match="controller batch script differs from the committed Git blob",
    ):
        scenario.run()

    assert scenario.slurm.sbatch_calls == 1
    assert scenario.slurm.release_calls == []
    assert not (scenario.registry / "attempts" / "attempt-0001.json").exists()
    relative = scenario.helper._controller_readback_relative(1)
    assert not scenario.registry.joinpath(*relative.split("/")).exists()


@pytest.mark.parametrize("controller_matches", [True, False], ids=["exact", "mismatch"])
def test_orphan_adoption_rechecks_controller_bytes_before_ledger_and_release(
    submission_scenario: Any,
    controller_matches: bool,
) -> None:
    scenario = submission_scenario
    scenario.slurm.crash_once("after_sbatch")
    with pytest.raises(SUBMIT_SUPPORT.InjectedCrash, match="after_sbatch"):
        scenario.run()
    assert scenario.slurm.sbatch_calls == 1
    assert scenario.slurm.release_calls == []
    assert not (scenario.registry / "attempts" / "attempt-0001.json").exists()

    if not controller_matches:
        scenario.slurm.controller_override = b"#!/usr/bin/env bash\nexit 98\n"
        with pytest.raises(RuntimeError, match="differs from the committed Git blob"):
            scenario.run()
        assert scenario.slurm.release_calls == []
        assert not (scenario.registry / "attempts" / "attempt-0001.json").exists()
        return

    assert scenario.run() == 0
    assert scenario.slurm.sbatch_calls == 1
    assert scenario.slurm.release_calls == ["3000"]
    assert (scenario.registry / "attempts" / "attempt-0001.json").is_file()
    relative = scenario.helper._controller_readback_relative(1)
    assert (
        scenario.registry.joinpath(*relative.split("/")).read_bytes()
        == scenario.slurm.committed_script
    )


@pytest.mark.parametrize("controller_matches", [True, False], ids=["exact", "mismatch"])
def test_ledger_resume_freshly_rechecks_controller_before_release(
    submission_scenario: Any,
    monkeypatch: pytest.MonkeyPatch,
    controller_matches: bool,
) -> None:
    scenario = submission_scenario
    SUBMIT_SUPPORT._post_install_crash(
        monkeypatch,
        scenario.helper,
        "_write_exclusive",
        target_name="aggregate submission attempt ledger",
    )
    with pytest.raises(
        SUBMIT_SUPPORT.InjectedCrash,
        match="aggregate submission attempt ledger",
    ):
        scenario.run()

    attempt = scenario.registry / "attempts" / "attempt-0001.json"
    assert attempt.is_file()
    assert scenario.slurm.sbatch_calls == 1
    assert scenario.slurm.release_calls == []

    if not controller_matches:
        scenario.slurm.controller_override = b"#!/usr/bin/env bash\nexit 97\n"
        with pytest.raises(RuntimeError, match="changed after capture"):
            scenario.run()
        assert scenario.slurm.release_calls == []
        return

    assert scenario.run() == 0
    assert scenario.slurm.sbatch_calls == 1
    assert scenario.slurm.release_calls == ["3000"]


def test_final_evidence_preserves_every_script_transport_binding(
    built_final_evidence: Any,
) -> None:
    built = built_final_evidence
    built.capture()
    control = FINAL_SUPPORT.control
    bundle = built.final_evidence / "aggregate-submission"
    committed_file = bundle / control.POST_RECOVERY_AGGREGATE_SCRIPT_EVIDENCE
    committed = committed_file.read_bytes()
    intent = _json(bundle / "intent.json")
    script = intent["sbatch_script"]

    assert script["sha256"] == _sha256(committed)
    assert script["git_blob_sha1"] == control._git_blob_sha1(committed)
    assert script["size_bytes"] == len(committed)
    assert script["evidence_filename"] == committed_file.name
    assert script["transport"] == control.POST_RECOVERY_AGGREGATE_SCRIPT_TRANSPORT

    for index, job_id in ((1, "2999"), (2, "3000")):
        attempt = _json(bundle / "attempts" / f"attempt-{index:04d}.json")
        binding = attempt["batch_script"]
        controller = binding["controller_readback"]
        relative = (
            f"{control.POST_RECOVERY_AGGREGATE_CONTROLLER_READBACKS}/attempt-{index:04d}.sbatch"
        )
        readback = bundle.joinpath(*relative.split("/"))
        command = binding["submission_command"]

        assert command[0] == "sbatch"
        assert all(argument.startswith("--") for argument in command[1:])
        assert script["repo_relative_path"] not in command
        assert binding["transport"] == control.POST_RECOVERY_AGGREGATE_SCRIPT_TRANSPORT
        assert binding["stdin_sha256"] == _sha256(committed)
        assert binding["stdin_size_bytes"] == len(committed)
        assert binding["controller_matches_committed"] is True
        assert controller == {
            "query": ["scontrol", "write", "batch_script", job_id, "-"],
            "relative_path": relative,
            "sha256": _sha256(committed),
            "size_bytes": len(committed),
        }
        assert readback.read_bytes() == committed

    built.verify()


@pytest.mark.parametrize(
    ("relative", "mutation"),
    [
        ("script.sbatch", "tamper"),
        ("script.sbatch", "missing"),
        ("controller/attempt-0002.sbatch", "tamper"),
        ("controller/attempt-0002.sbatch", "missing"),
    ],
)
def test_embedded_script_or_controller_evidence_tamper_fails_closed(
    built_final_evidence: Any,
    relative: str,
    mutation: str,
) -> None:
    built = built_final_evidence
    built.capture()
    target = built.final_evidence / "aggregate-submission"
    target = target.joinpath(*relative.split("/"))
    if mutation == "tamper":
        if os.name == "posix":
            target.chmod(0o640)
        target.write_bytes(target.read_bytes() + b"# tampered\n")
        if os.name == "posix":
            target.chmod(0o440)
    else:
        target.unlink()

    with pytest.raises(ValueError):
        built.verify()


def test_success_fast_path_is_offline_but_still_validates_embedded_binding(
    built_final_evidence: Any,
) -> None:
    built = built_final_evidence
    completed = built.capture()
    archive = built.campaign_root.with_name(f"{built.campaign_root.name}.archived")
    built.campaign_root.rename(archive)
    built.scheduler.fail_all = True

    verified = built.verify()
    replayed = built.capture()
    assert verified["receipt_sha256"] == completed["receipt_sha256"]
    assert replayed["receipt_sha256"] == completed["receipt_sha256"]
    assert archive.is_dir()

    control = FINAL_SUPPORT.control
    embedded = (
        built.final_evidence
        / "aggregate-submission"
        / control.POST_RECOVERY_AGGREGATE_SCRIPT_EVIDENCE
    )
    if os.name == "posix":
        embedded.chmod(0o640)
    embedded.write_bytes(embedded.read_bytes() + b"# post-success tamper\n")
    if os.name == "posix":
        embedded.chmod(0o440)
    with pytest.raises(ValueError):
        built.verify()
    with pytest.raises(ValueError):
        built.capture()
