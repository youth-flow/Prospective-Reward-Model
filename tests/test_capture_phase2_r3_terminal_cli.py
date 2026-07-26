from __future__ import annotations

import importlib.util
import os
import stat
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "hpc4" / "capture_phase2_r3_terminal.py"


def _load_cli() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_capture_phase2_r3_terminal_cli",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _argv(command: str) -> list[str]:
    return [
        command,
        "--operational-bundle",
        "bundle.json",
        "--operational-bundle-file-sha256",
        "1" * 64,
        "--runtime-closure",
        "runtime-closure.json",
        "--runtime-closure-file-sha256",
        "2" * 64,
        "--raw-sacct",
        "raw-sacct.psv",
        "--raw-sacct-sha256",
        "3" * 64,
        "--evidence-directory",
        "terminal-evidence",
    ]


def test_profile_intent_dispatch_binds_failure_lineage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _load_cli()
    calls: list[tuple[Path, dict[str, object]]] = []

    def publish(path: Path, **kwargs: object) -> SimpleNamespace:
        calls.append((path, kwargs))
        return SimpleNamespace(
            allocation_intent_sha256="3" * 64,
            file_sha256="4" * 64,
        )

    emitted: list[dict[str, object]] = []
    monkeypatch.setattr(cli, "publish_profile_allocation_intent", publish)
    monkeypatch.setattr(cli, "_emit", emitted.append)
    argv = [
        "profile-intent",
        "--output",
        "intent.json",
        "--attempt-lineage-file-sha256",
        "1" * 64,
        "--attempt-lineage-sha256",
        "2" * 64,
        "--cluster",
        "hpc4",
        "--account",
        "sigroup",
        "--partition",
        "gpu-l20",
        "--gpu-name",
        "NVIDIA L20",
        "--gpus-per-task",
        "1",
        "--cpus-per-task",
        "8",
        "--memory-bytes",
        "68719476736",
        "--walltime-seconds",
        "21600",
    ]

    assert cli.main(argv) == 0
    assert calls == [
        (
            Path("intent.json"),
            {
                "attempt_lineage_file_sha256": "1" * 64,
                "attempt_lineage_sha256": "2" * 64,
                "cluster": "hpc4",
                "account": "sigroup",
                "partition": "gpu-l20",
                "gpu_name": "NVIDIA L20",
                "gpus_per_task": 1,
                "cpus_per_task": 8,
                "memory_bytes": 68719476736,
                "requested_walltime_seconds": 21600,
            },
        )
    ]
    assert emitted == [
        {
            "status": "r3_gate_p_allocation_intent_published",
            "allocation_intent_sha256": "3" * 64,
            "file_sha256": "4" * 64,
        }
    ]


@pytest.mark.skipif(os.name != "posix", reason="publisher is an HPC4 POSIX-only surface")
def test_raw_publisher_enforces_mode_under_restrictive_umask(tmp_path: Path) -> None:
    cli = _load_cli()
    output = tmp_path / "raw-sacct.psv"
    raw = b"terminal scheduler evidence\n"

    previous_umask = os.umask(0o077) if os.name == "posix" else None
    try:
        digest = cli._publish_exclusive(output, raw)
    finally:
        if previous_umask is not None:
            os.umask(previous_umask)

    assert len(digest) == 64
    assert output.read_bytes() == raw
    if os.name == "posix":
        assert stat.S_IMODE(output.stat().st_mode) == 0o440
    with pytest.raises(FileExistsError):
        cli._publish_exclusive(output, raw)


@pytest.mark.parametrize(
    ("command", "finalizer_name", "status"),
    [
        (
            "primary-continuable-finalize",
            "finalize_continuable_primary_terminal_from_files",
            "r3_primary_continuable_scheduler_terminal_validated",
        ),
        (
            "primary-completed-finalize",
            "finalize_completed_primary_terminal_from_files",
            "r3_primary_completed_scheduler_terminal_validated",
        ),
    ],
)
def test_primary_finalize_parser_and_dispatch_smoke(
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    finalizer_name: str,
    status: str,
) -> None:
    cli = _load_cli()
    parsed = cli._parser().parse_args(_argv(command))
    assert set(vars(parsed)) == {
        "command",
        "operational_bundle",
        "operational_bundle_file_sha256",
        "runtime_closure",
        "runtime_closure_file_sha256",
        "raw_sacct",
        "raw_sacct_sha256",
        "evidence_directory",
    }

    calls: list[dict[str, object]] = []

    def finalize(**kwargs: object) -> SimpleNamespace:
        calls.append(kwargs)
        return SimpleNamespace(
            manifest_file_sha256="4" * 64,
            terminal_sha256="5" * 64,
        )

    emitted: list[dict[str, object]] = []
    monkeypatch.setattr(cli, finalizer_name, finalize)
    monkeypatch.setattr(cli, "_emit", emitted.append)

    assert cli.main(_argv(command)) == 0
    assert calls == [
        {
            "operational_bundle_path": Path("bundle.json"),
            "expected_operational_bundle_file_sha256": "1" * 64,
            "runtime_closure_path": Path("runtime-closure.json"),
            "expected_runtime_closure_file_sha256": "2" * 64,
            "raw_sacct_path": Path("raw-sacct.psv"),
            "expected_raw_sacct_sha256": "3" * 64,
            "evidence_directory": Path("terminal-evidence"),
        }
    ]
    assert emitted == [
        {
            "status": status,
            "manifest_file_sha256": "4" * 64,
            "terminal_sha256": "5" * 64,
        }
    ]
