from __future__ import annotations

import importlib.util
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
