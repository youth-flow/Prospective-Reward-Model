from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "hpc4" / "capture_phase2_budgeted_end_to_end_terminal.py"
ARRAY_JOB_ID = "7000"
REQ_TRES = "billing=8,cpu=8,gres/gpu=1,mem=96G,node=1"
ALLOC_TRES = "billing=8,cpu=8,gres/gpu:l20=1,gres/gpu=1,mem=96G,node=1"


def _load() -> ModuleType:
    name = "_budgeted_terminal_test"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _raw(*, mutation: tuple[int, int, str] | None = None) -> bytes:
    rows: list[list[str]] = []
    for task in range(5):
        rows.append(
            [
                f"{ARRAY_JOB_ID}_{task}",
                str(7100 + task),
                "COMPLETED",
                "0:0",
                "0:0",
                "hpc4",
                "sigroup",
                "gpu-l20",
                "l20_qos",
                "1",
                "8",
                REQ_TRES,
                ALLOC_TRES,
            ]
        )
    if mutation is not None:
        row, column, value = mutation
        rows[row][column] = value
    return ("".join("|".join(row) + "\n" for row in rows)).encode()


def _capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    raw: bytes | None = None,
) -> tuple[ModuleType, Path]:
    module = _load()
    observed: list[tuple[str, ...]] = []

    def run(arguments: tuple[str, ...], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        observed.append(arguments)
        return subprocess.CompletedProcess(arguments, 0, raw or _raw(), b"")

    monkeypatch.setattr(module.subprocess, "run", run)
    output = tmp_path / "terminal.json"
    payload = module.capture_terminal_evidence(ARRAY_JOB_ID, output)
    assert observed == [module.sacct_command(ARRAY_JOB_ID)]
    assert payload["ordered_seeds"] == list(range(20261001, 20261006))
    return module, output


def test_capture_and_verify_exact_fixed_five_terminal_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, output = _capture(tmp_path, monkeypatch)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()

    payload = module.verify_terminal_evidence(
        output,
        expected_sha256=digest,
        expected_array_job_id=ARRAY_JOB_ID,
    )

    assert [row["array_task_id"] for row in payload["rows"]] == list(range(5))
    assert [row["seed"] for row in payload["rows"]] == list(range(20261001, 20261006))
    assert {row["qos"] for row in payload["rows"]} == {"l20_qos"}
    assert json.loads(output.read_bytes()) == payload


@pytest.mark.parametrize(
    ("row", "column", "value"),
    [
        (0, 0, f"{ARRAY_JOB_ID}_1"),
        (1, 1, "7100"),
        (2, 2, "FAILED"),
        (2, 3, "1:0"),
        (2, 4, "1:0"),
        (3, 5, "other"),
        (3, 6, "other"),
        (3, 7, "amd"),
        (3, 8, "wrong_qos"),
        (4, 11, "billing=7,cpu=7,gres/gpu=1,mem=96G,node=1"),
        (4, 12, "billing=8,cpu=8,gres/gpu=1,mem=96G,node=1"),
    ],
)
def test_capture_rejects_terminal_or_scheduler_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    row: int,
    column: int,
    value: str,
) -> None:
    module = _load()
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda arguments, **_kwargs: subprocess.CompletedProcess(
            arguments,
            0,
            _raw(mutation=(row, column, value)),
            b"",
        ),
    )
    with pytest.raises(ValueError, match="exact successful HPC4 allocation"):
        module.capture_terminal_evidence(ARRAY_JOB_ID, tmp_path / "terminal.json")


def test_raw_tamper_and_cross_array_replay_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, output = _capture(tmp_path, monkeypatch)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    raw_path = tmp_path / "terminal.sacct.psv"
    raw_path.write_bytes(_raw(mutation=(4, 8, "other_qos")))

    with pytest.raises(ValueError, match="does not bind its raw bytes"):
        module.verify_terminal_evidence(
            output,
            expected_sha256=digest,
            expected_array_job_id=ARRAY_JOB_ID,
        )
    with pytest.raises(ValueError, match="identity"):
        module.verify_terminal_evidence(
            output,
            expected_sha256=digest,
            expected_array_job_id="8000",
        )


def test_capture_never_overwrites_either_publication_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, output = _capture(tmp_path, monkeypatch)
    original = output.read_bytes()
    raw_path = tmp_path / "terminal.sacct.psv"
    original_raw = raw_path.read_bytes()

    with pytest.raises(FileExistsError, match="overwrite"):
        module.capture_terminal_evidence(ARRAY_JOB_ID, output)

    assert output.read_bytes() == original
    assert raw_path.read_bytes() == original_raw


def test_noncanonical_or_duplicate_envelope_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, output = _capture(tmp_path, monkeypatch)
    value = json.loads(output.read_bytes())
    output.write_text(json.dumps(value, indent=2), encoding="utf-8")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="canonical"):
        module.verify_terminal_evidence(
            output,
            expected_sha256=digest,
            expected_array_job_id=ARRAY_JOB_ID,
        )


def test_tres_semantics_are_order_independent_but_duplicate_closed() -> None:
    module = _load()
    reordered = (
        _raw()
        .replace(
            REQ_TRES.encode(),
            b"node=1,mem=96G,gres/gpu=1,cpu=8,billing=8",
        )
        .replace(
            ALLOC_TRES.encode(),
            b"node=1,mem=96G,gres/gpu=1,gres/gpu:l20=1,cpu=8,billing=8",
        )
    )
    assert len(module._parse_sacct(reordered, array_job_id=ARRAY_JOB_ID)) == 5

    duplicated = _raw().replace(
        REQ_TRES.encode(),
        b"billing=8,billing=8,cpu=8,gres/gpu=1,mem=96G,node=1",
    )
    with pytest.raises(ValueError, match="duplicate"):
        module._parse_sacct(duplicated, array_job_id=ARRAY_JOB_ID)
