from __future__ import annotations

import dataclasses
import inspect
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

from smart_reward import phase2_r3_gate1 as gate1
from smart_reward.phase2_r3_identity import ArtifactRef

_PROBE = {
    "python_implementation": "CPython",
    "python_version": "3.11.13",
    "torch_version": "2.7.1+cu126",
    "torch_cuda_version": "12.6",
    "cuda_available": True,
    "cuda_device_count": 1,
    "cuda_device_name": "NVIDIA L20",
    "cuda_compute_capability": [8, 9],
}

_FORMAL_SCRIPTS = {
    "scripts/hpc4/capture_phase2_r3_gate1.py",
    "scripts/hpc4/validate_phase2_r3_authorization.py",
    "scripts/hpc4/submit_phase2_r3_profile.sh",
    "scripts/hpc4/phase2_r3_profile.sbatch",
    "scripts/hpc4/capture_phase2_r3_terminal.py",
}


def _run_git(root: Path, *arguments: str) -> None:
    subprocess.run(
        ("git", *arguments),
        cwd=root,
        check=True,
        stdin=subprocess.DEVNULL,
        capture_output=True,
    )


def _contents(relative: str) -> bytes:
    suffix = Path(relative).suffix
    if suffix == ".py":
        return b'"""Temporary Gate-1 fixture."""\n\nVALUE = 1\n'
    if suffix in {".sh", ".sbatch"}:
        return b"#!/usr/bin/env bash\nset -euo pipefail\n"
    if suffix in {".yaml", ".yml"}:
        return b"fixture: true\n"
    if suffix == ".toml":
        return b'[project]\nname = "gate1-fixture"\nversion = "0.0.0"\n'
    return f"Gate-1 fixture for {relative}\n".encode()


@dataclass
class _FixtureRunner:
    engine: Path
    fail_command: str | None = None
    probe: dict[str, object] | None = None

    def __post_init__(self) -> None:
        self.observed: list[tuple[str, ...]] = []

    @staticmethod
    def _command_name(command: tuple[str, ...]) -> str:
        executable = Path(command[0]).stem.lower()
        if executable == "ruff" and command[1:2] == ("check",):
            return "ruff_check"
        if executable == "ruff" and command[1:3] == ("format", "--check"):
            return "ruff_format_check"
        if executable.startswith("python") and command[1:4] == (
            "-m",
            "compileall",
            "-q",
        ):
            return "python_compileall"
        if command[:6] == (
            "env",
            "PYTHONPYCACHEPREFIX=/tmp/prorm-r3-gate1-pyc",
            "python",
            "-m",
            "compileall",
            "-q",
        ):
            return "sif_python_compileall"
        if executable.startswith("python") and command[1:4] == (
            "-m",
            "pytest",
            "-q",
        ):
            return "pytest"
        if command[:2] == ("bash", "-n"):
            return "sif_bash_syntax"
        return "unknown"

    def __call__(
        self,
        command: tuple[str, ...],
        *,
        cwd: Path,
        maximum_bytes: int,
    ) -> gate1._CommandResult:
        del maximum_bytes
        self.observed.append(command)
        if command[0] == "git":
            return gate1._subprocess_runner(
                command,
                cwd=cwd,
                maximum_bytes=gate1._MAX_COMMAND_BYTES,
            )
        if command == (str(self.engine), "--version"):
            return gate1._CommandResult(0, b"apptainer version 1.3.6\n", b"")
        executable = Path(command[0]).stem.lower()
        if command[1:] == ("--version",) and executable.startswith("python"):
            return gate1._CommandResult(0, b"Python 3.11.13\n", b"")
        if command[1:] == ("--version",) and executable == "ruff":
            return gate1._CommandResult(0, b"ruff 0.15.22\n", b"")
        if executable.startswith("python") and command[1:] == ("-m", "pytest", "--version"):
            return gate1._CommandResult(0, b"pytest 7.4.4\n", b"")
        if (
            command[0] == str(self.engine)
            and command[1:5]
            == (
                "exec",
                "--cleanenv",
                "--containall",
                "--nv",
            )
            and len(command) == 9
        ):
            raw = gate1._canonical_bytes(self.probe or _PROBE)
            return gate1._CommandResult(0, raw, b"")
        if command[0] == str(self.engine) and command[1:6] == (
            "exec",
            "--cleanenv",
            "--containall",
            "--nv",
            "--env",
        ):
            if command[7] != "--bind" or command[9] != "--bind" or command[11] != "--pwd":
                raise AssertionError(f"verification wrapper omitted bind/pwd: {command!r}")
            command = command[14:]
        name = self._command_name(command)
        if name == self.fail_command:
            stdout = (
                b"progress\n"
                b"FAILED tests/test_phase2_r3_gate1.py::test_fixture_failure"
                b" - AssertionError\n"
                if name == "pytest"
                else b""
            )
            return gate1._CommandResult(7, stdout, f"{name} failed\n".encode())
        if name == "unknown":
            raise AssertionError(f"unexpected fixture command: {command!r}")
        return gate1._CommandResult(0, f"{name} passed\n".encode(), b"")


@dataclass(frozen=True)
class _RepositoryFixture:
    root: Path
    container: Path
    engine: Path
    runner: _FixtureRunner


def _commit_all(root: Path, message: str) -> None:
    _run_git(root, "add", "-A")
    _run_git(root, "commit", "-m", message)


def _repository(tmp_path: Path) -> _RepositoryFixture:
    root = (tmp_path / "repo").resolve()
    root.mkdir()
    paths = set(gate1._REQUIRED_TRACKED_PATHS) | _FORMAL_SCRIPTS
    for relative in sorted(paths):
        path = root.joinpath(*Path(relative).parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_contents(relative))
    _run_git(root, "init", "-q")
    _run_git(root, "config", "user.email", "gate1@example.invalid")
    _run_git(root, "config", "user.name", "Gate One Test")
    _commit_all(root, "formal fixture")

    external = (tmp_path / "runtime").resolve()
    external.mkdir()
    container = external / "prorm-hpc4.sif"
    container.write_bytes(b"canonical-sif-fixture\n")
    engine = external / "apptainer"
    engine.write_bytes(b"fixture executable\n")
    runner = _FixtureRunner(engine=engine)
    return _RepositoryFixture(
        root=root,
        container=container,
        engine=engine,
        runner=runner,
    )


def _capture(fixture: _RepositoryFixture) -> bytes:
    source_test_raw = gate1._capture_source_test_receipt_for_inspection(
        root=fixture.root,
        runner=fixture.runner,
        captured_at_utc="2026-07-26T11:59:00Z",
    )
    return gate1._capture_gate1_evidence_for_inspection(
        root=fixture.root,
        container_path=fixture.container,
        engine_path=fixture.engine,
        runner=fixture.runner,
        captured_at_utc="2026-07-26T12:00:00Z",
        source_test_receipt_raw=source_test_raw,
        expected_source_test_receipt_file_sha256=gate1._sha256(source_test_raw),
    )


def _write_evidence(tmp_path: Path, raw: bytes) -> Path:
    path = (tmp_path / "gate1.json").resolve()
    path.write_bytes(raw)
    return path


def _payload(raw: bytes) -> dict[str, object]:
    value = json.loads(raw)
    assert isinstance(value, dict)
    return value


def _rehash(payload: dict[str, object]) -> bytes:
    unsigned = dict(payload)
    unsigned.pop("artifact_sha256", None)
    payload["artifact_sha256"] = gate1._canonical_sha256(unsigned)
    return gate1._canonical_bytes(payload)


def test_offline_inspection_binds_clean_git_commands_and_container_without_authority(
    tmp_path: Path,
) -> None:
    fixture = _repository(tmp_path)
    raw = _capture(fixture)
    path = _write_evidence(tmp_path, raw)
    inspection = gate1.inspect_r3_gate1_bundle(
        path,
        expected_file_sha256=gate1._sha256(raw),
    )

    assert inspection.formal_authorization is False
    assert inspection.file_sha256 == gate1._sha256(raw)
    payload = _payload(raw)
    assert inspection.source_commit == payload["source"]["commit"]
    assert inspection.source_artifact_sha256 == payload["source"]["artifact_sha256"]
    assert inspection.container_artifact_sha256 == payload["container"]["artifact_sha256"]
    receipts = payload["verification"]["commands"]
    assert [receipt["name"] for receipt in receipts] == [
        "sif_python_compileall",
        "sif_bash_syntax",
    ]
    assert all(receipt["exit_code"] == 0 for receipt in receipts)
    assert all(
        receipt["argv"][:6]
        == [
            str(fixture.engine),
            "exec",
            "--cleanenv",
            "--containall",
            "--nv",
            "--env",
        ]
        for receipt in receipts
    )
    assert all(receipt["argv"][7] == "--bind" for receipt in receipts)
    assert all(receipt["argv"][9] == "--bind" for receipt in receipts)
    assert all(
        receipt["argv"][10] == f"{gate1.PRODUCTION_PROJECT_ROOT}:{gate1.PRODUCTION_PROJECT_ROOT}:ro"
        for receipt in receipts
    )
    source_test = payload["verification"]["source_test_receipt"]
    assert [receipt["name"] for receipt in source_test["verification"]["commands"]] == [
        "ruff_check",
        "ruff_format_check",
        "python_compileall",
        "pytest",
    ]
    assert [receipt["name"] for receipt in source_test["toolchain"]["version_receipts"]] == [
        "python_version",
        "ruff_version",
        "pytest_version",
    ]
    source_test_raw = gate1._canonical_bytes(source_test)
    source_test_path = (tmp_path / "source-test.json").resolve()
    source_test_path.write_bytes(source_test_raw)
    source_test_inspection = gate1.inspect_r3_source_test_receipt(
        source_test_path,
        expected_file_sha256=gate1._sha256(source_test_raw),
    )
    assert source_test_inspection.formal_authorization is False
    assert source_test_inspection.source_artifact_sha256 == inspection.source_artifact_sha256
    with pytest.raises(ValueError, match="source-test receipt file differs"):
        gate1._capture_gate1_evidence_for_inspection(
            root=fixture.root,
            container_path=fixture.container,
            engine_path=fixture.engine,
            runner=fixture.runner,
            captured_at_utc="2026-07-26T12:01:00Z",
            source_test_receipt_raw=source_test_raw,
            expected_source_test_receipt_file_sha256="0" * 64,
        )
    assert {role for entry in payload["source"]["inventory"] for role in entry["roles"]}.issuperset(
        gate1._REQUIRED_SCRIPT_COVERAGE
    )
    runtime = payload["container"]["runtime"]
    assert runtime["probe"] == _PROBE
    assert runtime["probe_receipt"]["stdout"]["sha256"] == gate1._sha256(
        gate1._canonical_bytes(_PROBE)
    )


def test_inspection_rejects_wrong_file_sha_noncanonical_and_nested_tamper(
    tmp_path: Path,
) -> None:
    fixture = _repository(tmp_path)
    raw = _capture(fixture)
    path = _write_evidence(tmp_path, raw)
    with pytest.raises(ValueError, match="caller expectation"):
        gate1.inspect_r3_gate1_bundle(path, expected_file_sha256="0" * 64)

    path.write_bytes(json.dumps(_payload(raw), indent=2).encode())
    with pytest.raises(ValueError, match="not canonical"):
        gate1.inspect_r3_gate1_bundle(path)

    payload = _payload(raw)
    payload["source"]["inventory"][0]["working_sha256"] = "1" * 64
    path.write_bytes(_rehash(payload))
    with pytest.raises(ValueError, match="source inventory closure"):
        gate1.inspect_r3_gate1_bundle(path)


def test_clean_committed_source_and_all_script_categories_fail_closed(
    tmp_path: Path,
) -> None:
    fixture = _repository(tmp_path)
    (fixture.root / "untracked.txt").write_text("dirty", encoding="utf-8")
    with pytest.raises(ValueError, match="clean committed"):
        _capture(fixture)

    (fixture.root / "untracked.txt").unlink()
    terminal = fixture.root / "scripts/hpc4/capture_phase2_r3_terminal.py"
    terminal.unlink()
    _commit_all(fixture.root, "remove terminal capture")
    with pytest.raises(ValueError, match="terminal_capture"):
        _capture(fixture)


def test_any_failed_verification_command_prevents_even_offline_evidence(
    tmp_path: Path,
) -> None:
    fixture = _repository(tmp_path)
    fixture.runner.fail_command = "pytest"
    with pytest.raises(
        RuntimeError,
        match=(
            "pytest failed with exit 7; "
            "stdout_size_bytes=.*failure_tests="
            "tests/test_phase2_r3_gate1.py::test_fixture_failure"
        ),
    ):
        _capture(fixture)
    assert any(
        Path(command[0]).stem.lower().startswith("python")
        and command[1:4] == ("-m", "pytest", "-q")
        for command in fixture.runner.observed
    )
    assert not any(command[0] == str(fixture.engine) for command in fixture.runner.observed)


def test_gate1_source_closure_includes_shared_non_r3_tests() -> None:
    assert gate1._is_formal_tracked_path("tests/test_training.py")
    assert gate1._is_formal_tracked_path("tests/test_rollout.py")
    assert gate1._is_formal_tracked_path("tests/conftest.py")
    assert not gate1._is_formal_tracked_path("tests/fixtures/result.json")


def test_runtime_probe_requires_exact_python_torch_cuda_and_one_l20(
    tmp_path: Path,
) -> None:
    fixture = _repository(tmp_path)
    bad = dict(_PROBE)
    bad["cuda_available"] = False
    fixture.runner.probe = bad
    with pytest.raises(ValueError, match="frozen HPC4 L20"):
        _capture(fixture)


def test_probe_raw_bytes_cannot_be_replaced_by_rehashed_claims(
    tmp_path: Path,
) -> None:
    fixture = _repository(tmp_path)
    payload = _payload(_capture(fixture))
    runtime = payload["container"]["runtime"]
    runtime["probe"] = {**_PROBE, "cuda_device_name": "claimed L20"}
    runtime_unsigned = dict(runtime)
    runtime_unsigned.pop("runtime_probe_sha256")
    runtime["runtime_probe_sha256"] = gate1._canonical_sha256(runtime_unsigned)
    container = payload["container"]
    container_unsigned = dict(container)
    container_unsigned.pop("artifact_sha256")
    container["artifact_sha256"] = gate1._canonical_sha256(container_unsigned)
    path = _write_evidence(tmp_path, _rehash(payload))
    with pytest.raises(ValueError, match="parsed probe differs"):
        gate1.inspect_r3_gate1_bundle(path)


def test_publication_is_no_overwrite_and_inspection_uses_caller_sha(
    tmp_path: Path,
) -> None:
    fixture = _repository(tmp_path)
    raw = _capture(fixture)
    parent = (tmp_path / "publication").resolve()
    parent.mkdir()
    path = parent / "gate1.json"
    digest = gate1._publish_exclusive(path, raw)
    assert digest == gate1._sha256(raw)
    assert (
        gate1.inspect_r3_gate1_bundle(
            path,
            expected_file_sha256=digest,
        ).file_sha256
        == digest
    )
    with pytest.raises(FileExistsError, match="overwrite"):
        gate1._publish_exclusive(path, raw)


def test_publication_cleans_only_its_owned_temp_after_fsync_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    parent = (tmp_path / "publication-failure").resolve()
    parent.mkdir()
    path = parent / "gate1.json"

    monkeypatch.setattr(
        gate1.os,
        "fsync",
        lambda _descriptor: (_ for _ in ()).throw(OSError("forced fsync failure")),
    )

    with pytest.raises(OSError, match="forced fsync failure"):
        gate1._publish_exclusive(path, b"evidence\n")

    assert not path.exists()
    assert list(parent.iterdir()) == []


def test_portable_cleanup_preserves_a_different_inode(tmp_path: Path) -> None:
    path = tmp_path / "replacement.json"
    path.write_bytes(b"replacement\n")
    info = path.stat()

    gate1._unlink_if_identity(path, (info.st_dev, info.st_ino + 1))

    assert path.read_bytes() == b"replacement\n"


def test_reopen_reruns_checks_and_rejects_new_clean_commit_or_changed_sif(
    tmp_path: Path,
) -> None:
    fixture = _repository(tmp_path)
    raw = _capture(fixture)
    payload = gate1._validate_payload(_payload(raw))
    source_test = payload["verification"]["source_test_receipt"]
    source_test_file_sha = payload["verification"]["source_test_receipt_file_sha256"]
    baseline = len(fixture.runner.observed)
    current, live_sha = gate1._reverify_current(
        payload,
        root=fixture.root,
        container_path=fixture.container,
        engine_path=fixture.engine,
        runner=fixture.runner,
        file_sha256=gate1._sha256(raw),
        source_test_receipt=source_test,
        source_test_receipt_file_sha256=source_test_file_sha,
    )
    assert len(fixture.runner.observed) > baseline
    assert current["source"] == payload["source"]
    assert len(live_sha) == 64

    fixture.container.write_bytes(b"different-sif-bytes\n")
    with pytest.raises(ValueError, match="SIF/definition/lock differs"):
        gate1._reverify_current(
            payload,
            root=fixture.root,
            container_path=fixture.container,
            engine_path=fixture.engine,
            runner=fixture.runner,
            file_sha256=gate1._sha256(raw),
            source_test_receipt=source_test,
            source_test_receipt_file_sha256=source_test_file_sha,
        )

    fixture.container.write_bytes(b"canonical-sif-fixture\n")
    science = fixture.root / "configs/phase2_recovery_r3_science.yaml"
    science.write_text("fixture: changed\n", encoding="utf-8")
    _commit_all(fixture.root, "new science config")
    with pytest.raises(ValueError, match="clean commit/tree/blob inventory differs"):
        gate1._reverify_current(
            payload,
            root=fixture.root,
            container_path=fixture.container,
            engine_path=fixture.engine,
            runner=fixture.runner,
            file_sha256=gate1._sha256(raw),
            source_test_receipt=source_test,
            source_test_receipt_file_sha256=source_test_file_sha,
        )


def _constructor_kwargs(capability: object) -> dict[str, object]:
    return {
        item.name: getattr(capability, item.name)
        for item in dataclasses.fields(capability)
        if item.init
    }


def test_capabilities_are_distinct_sealed_and_dto_conversion_is_typed(
    tmp_path: Path,
) -> None:
    fixture = _repository(tmp_path)
    raw = _capture(fixture)
    payload = gate1._validate_payload(_payload(raw))
    current = {name: payload[name] for name in ("source", "verification", "container", "contract")}
    capabilities = gate1._issue_capabilities(
        payload,
        current,
        file_sha256=gate1._sha256(raw),
        live_reverification_sha256="a" * 64,
    )
    source_commit = str(payload["source"]["commit"])
    assert capabilities.gate1.production_relative == gate1._gate1_relative(source_commit).as_posix()
    assert gate1._gate1_namespace_commit(capabilities.gate1.production_relative) == (source_commit)
    cross_commit = _constructor_kwargs(capabilities.gate1)
    cross_commit["production_relative"] = gate1._gate1_relative("f" * 40).as_posix()
    with pytest.raises(ValueError, match="namespace"):
        gate1.R3Gate1Capability(
            **cross_commit,
            _factory_token=gate1._GATE1_FACTORY_TOKEN,
        )
    mismatched_source = _constructor_kwargs(capabilities.gate1)
    mismatched_source["source_git_commit"] = "f" * 40
    mismatched_source["production_relative"] = gate1._gate1_relative("f" * 40).as_posix()
    unsigned = dict(mismatched_source)
    del unsigned["capability_sha256"]
    mismatched_source["capability_sha256"] = gate1._canonical_sha256(unsigned)
    foreign_gate1 = gate1.R3Gate1Capability(
        **mismatched_source,
        _factory_token=gate1._GATE1_FACTORY_TOKEN,
    )
    with pytest.raises(ValueError, match="close over one evidence bundle"):
        gate1.R3Gate1Capabilities(
            gate1=foreign_gate1,
            source=capabilities.source,
            container=capabilities.container,
        )

    assert gate1.r3_gate1_artifact_ref(capabilities.gate1).schema_version == (
        gate1.GATE1_ARTIFACT_SCHEMA
    )
    assert gate1.r3_source_artifact_ref(capabilities.source).schema_version == (
        gate1.SOURCE_ARTIFACT_SCHEMA
    )
    assert (
        gate1.r3_container_artifact_ref(capabilities.container).schema_version
        == gate1.CONTAINER_ARTIFACT_SCHEMA
    )

    for capability in (
        capabilities.gate1,
        capabilities.source,
        capabilities.container,
    ):
        with pytest.raises(TypeError, match="live Gate-1 verification"):
            type(capability)(**_constructor_kwargs(capability))
        with pytest.raises(TypeError, match="live Gate-1 verification"):
            dataclasses.replace(capability, capability_sha256="b" * 64)

    generic = ArtifactRef(
        schema_version=gate1.SOURCE_ARTIFACT_SCHEMA,
        artifact_sha256=capabilities.source.artifact_sha256,
        role=gate1.SOURCE_ARTIFACT_ROLE,
    )
    with pytest.raises(TypeError, match="exact R3SourceCapability"):
        gate1.r3_source_artifact_ref(generic)  # type: ignore[arg-type]


@pytest.mark.skipif(os.name != "posix", reason="requires exact POSIX modes")
def test_stable_bytes_checks_mode_on_the_open_descriptor(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    path.write_bytes(b"evidence")
    os.chmod(path, 0o600)
    with pytest.raises(ValueError, match="retain mode 0440"):
        gate1._stable_bytes(
            path,
            name="receipt",
            maximum_bytes=1024,
            required_mode=0o440,
        )
    os.chmod(path, 0o440)
    assert (
        gate1._stable_bytes(
            path,
            name="receipt",
            maximum_bytes=1024,
            required_mode=0o440,
        )
        == b"evidence"
    )


def test_public_authorizers_have_no_runner_injection_and_fail_off_hpc4(
    tmp_path: Path,
) -> None:
    fixture = _repository(tmp_path)
    assert "runner" not in gate1.capture_live_r3_gate1_evidence.__annotations__
    assert "runner" not in gate1.verify_live_r3_gate1_bundle.__annotations__
    assert "runner" not in gate1.verify_live_r3_gate1_in_container.__annotations__
    with pytest.raises((RuntimeError, ValueError)):
        gate1.capture_live_r3_gate1_evidence(
            container=fixture.container,
            expected_source_test_receipt_file_sha256="0" * 64,
        )
    with pytest.raises((RuntimeError, ValueError)):
        gate1.verify_live_r3_gate1_bundle(
            container=fixture.container,
            expected_file_sha256="0" * 64,
            expected_source_test_receipt_file_sha256="0" * 64,
        )
    with pytest.raises((RuntimeError, ValueError)):
        gate1.verify_live_r3_gate1_in_container(
            expected_file_sha256="0" * 64,
            expected_source_test_receipt_file_sha256="0" * 64,
        )


def test_gate1_formal_surfaces_fix_disjoint_roots_and_receipt_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = Path("/home/yyangjo/Smart-Reward-Model")
    project = Path("/project/sigroup/smart-reward-model")
    assert repo == gate1.PRODUCTION_REPO_ROOT
    assert project == gate1.PRODUCTION_PROJECT_ROOT
    assert repo != project and repo not in project.parents and project not in repo.parents
    commit = "1" * 40
    assert gate1._gate1_relative(commit) == (
        Path("runs/phase2-recovery-r3/gate1") / commit / "r3-implementation-closure.json"
    )
    assert gate1._source_test_receipt_relative(commit) == (
        Path("runs/phase2-recovery-r3/gate1") / commit / "r3-source-test-receipt.json"
    )
    assert gate1._gate1_namespace_commit(gate1._gate1_relative(commit).as_posix()) == commit
    with pytest.raises(ValueError, match="namespace"):
        gate1._gate1_namespace_commit(
            "runs/phase2-recovery-r3/gate1/r3-implementation-closure.json"
        )
    with pytest.raises(ValueError, match="namespace commit"):
        gate1._gate1_relative("../mutable")
    assert set(inspect.signature(gate1.capture_r3_gate1_source_test_receipt).parameters) == set()
    assert set(inspect.signature(gate1.capture_live_r3_gate1_evidence).parameters) == {
        "container",
        "expected_source_test_receipt_file_sha256",
    }
    assert set(inspect.signature(gate1.verify_live_r3_gate1_bundle).parameters) == {
        "container",
        "expected_file_sha256",
        "expected_source_test_receipt_file_sha256",
    }
    assert set(inspect.signature(gate1.verify_live_r3_gate1_in_container).parameters) == {
        "expected_file_sha256",
        "expected_source_test_receipt_file_sha256",
    }
    cli = (
        Path(__file__).resolve().parents[1] / "scripts" / "hpc4" / "capture_phase2_r3_gate1.py"
    ).read_text(encoding="utf-8")
    assert "--project-root" not in cli
    assert "--output" not in cli
    assert 'add_argument("--source-test-receipt",' not in cli

    with monkeypatch.context() as patch:
        patch.setattr(gate1, "PRODUCTION_REPO_ROOT", project)
        patch.setattr(gate1, "PRODUCTION_PROJECT_ROOT", repo)
        with pytest.raises(RuntimeError, match="fixed HPC4 path"):
            gate1._assert_production_roots()

    with monkeypatch.context() as patch:
        patch.setattr(gate1, "PRODUCTION_PROJECT_ROOT", repo / "persistent")
        with pytest.raises(RuntimeError, match="fixed HPC4 path"):
            gate1._assert_production_roots()
