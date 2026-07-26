from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "hpc4"
    / "prepare_phase2_r3_primary_submission.py"
)


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_prepare_r3_primary", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _digest(character: str) -> str:
    return character * 64


def _argv(tmp_path: Path) -> list[str]:
    retained: dict[str, Path] = {}
    for name in (
        "image.sif",
        "science.yaml",
        "source.yaml",
        "parent.json",
        "gate0.json",
        "gate1.json",
        "source-test.json",
    ):
        path = (tmp_path / name).absolute()
        path.write_bytes(f"{name}\n".encode())
        retained[name] = path

    def digest(name: str) -> str:
        return hashlib.sha256(retained[name].read_bytes()).hexdigest()

    return [
        "create",
        "--operational-bundle",
        str(tmp_path / "bundle.json"),
        "--operational-bundle-file-sha256",
        _digest("1"),
        "--profile-allocation-intent",
        str(tmp_path / "profile-intent.json"),
        "--profile-allocation-intent-file-sha256",
        _digest("2"),
        "--profile-runtime-receipt",
        str(tmp_path / "profile-runtime.json"),
        "--profile-runtime-receipt-file-sha256",
        _digest("3"),
        "--profile-terminal-evidence-directory",
        str(tmp_path / "profile-terminal"),
        "--profile-terminal-manifest-file-sha256",
        _digest("4"),
        "--profile-terminal-raw-sacct-sha256",
        _digest("5"),
        "--git-commit",
        "a" * 40,
        "--container-image",
        str(retained["image.sif"]),
        "--container-image-file-sha256",
        digest("image.sif"),
        "--science-config",
        str(retained["science.yaml"]),
        "--science-config-file-sha256",
        digest("science.yaml"),
        "--source-config",
        str(retained["source.yaml"]),
        "--source-config-file-sha256",
        digest("source.yaml"),
        "--parent-registry",
        str(retained["parent.json"]),
        "--parent-registry-file-sha256",
        digest("parent.json"),
        "--gate0",
        str(retained["gate0.json"]),
        "--gate0-file-sha256",
        digest("gate0.json"),
        "--gate1",
        str(retained["gate1.json"]),
        "--gate1-file-sha256",
        digest("gate1.json"),
        "--source-test-receipt",
        str(retained["source-test.json"]),
        "--source-test-receipt-file-sha256",
        digest("source-test.json"),
        "--output",
        str(tmp_path / "primary-plan.json"),
    ]


def _bind_pure_data_dependencies(
    module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[
    SimpleNamespace,
    SimpleNamespace,
    SimpleNamespace,
    SimpleNamespace,
    list[tuple[str, tuple[object, ...], dict[str, object]]],
]:
    bundle = SimpleNamespace(
        artifact_path=(tmp_path / "bundle.json").absolute(),
        file_sha256=_digest("1"),
        bundle_semantic_sha256=_digest("6"),
        resource_plan_sha256=_digest("7"),
        resource_plan={"array_concurrency": 2},
        slurm_account="sigroup",
        partition="gpu-l20",
        gpu_name="NVIDIA L20",
        gpus_per_task=1,
        cpus_per_task=8,
        memory_bytes=64 * 1024 * 1024 * 1024,
        requested_walltime_seconds_per_segment=43200,
        advance_signal_lead_seconds=900,
        max_scheduler_segments=3,
        audit_cadence_updates=20,
        durable_checkpoint_cadence_updates=200,
    )
    intent = SimpleNamespace(
        artifact_path=(tmp_path / "profile-intent.json").absolute(),
        file_sha256=_digest("2"),
    )
    runtime = SimpleNamespace(
        artifact_path=(tmp_path / "profile-runtime.json").absolute(),
        file_sha256=_digest("3"),
    )
    terminal = SimpleNamespace(
        evidence_directory=(tmp_path / "profile-terminal").absolute(),
        manifest_file_sha256=_digest("4"),
        inspection=SimpleNamespace(raw_sacct_sha256=_digest("5")),
        terminal_sha256=_digest("8"),
    )
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def bind(name: str, result: object):
        def call(*args: object, **kwargs: object) -> object:
            calls.append((name, args, kwargs))
            return result

        return call

    monkeypatch.setattr(
        module,
        "reopen_verified_gate_p_operational_bundle",
        bind("bundle", bundle),
    )
    monkeypatch.setattr(
        module,
        "reopen_profile_allocation_intent",
        bind("intent", intent),
    )
    monkeypatch.setattr(
        module,
        "reopen_profile_slurm_runtime_receipt",
        bind("runtime", runtime),
    )
    monkeypatch.setattr(
        module,
        "revalidate_successful_profile_terminal",
        bind("terminal", terminal),
    )
    return bundle, intent, runtime, terminal, calls


def test_create_revalidates_successful_gatep_and_derives_every_sbatch_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script()
    bundle, intent, runtime, terminal, calls = _bind_pure_data_dependencies(
        module,
        monkeypatch,
        tmp_path,
    )

    assert module.main(_argv(tmp_path)) == 0
    output = json.loads(capsys.readouterr().out)
    plan_path = tmp_path / "primary-plan.json"
    raw = plan_path.read_bytes()
    plan = json.loads(raw)

    assert output["file_sha256"] == hashlib.sha256(raw).hexdigest()
    assert output["resource_plan_sha256"] == _digest("7")
    assert [name for name, _, _ in calls] == [
        "bundle",
        "intent",
        "runtime",
        "terminal",
    ]
    assert calls[0][2] == {"expected_file_sha256": _digest("1")}
    assert calls[1][2] == {"expected_file_sha256": _digest("2")}
    assert calls[2][2] == {
        "expected_file_sha256": _digest("3"),
        "operational_bundle": bundle,
        "allocation_intent": intent,
    }
    assert calls[3][1] == (bundle,)
    assert calls[3][2]["runtime_receipt"] is runtime
    assert calls[3][2]["evidence_directory"] == terminal.evidence_directory
    assert calls[3][2]["expected_manifest_file_sha256"] == _digest("4")
    assert calls[3][2]["expected_raw_sacct_sha256"] == _digest("5")

    assert plan["segment_index"] == 1
    assert plan["array_task_ids"] == [0, 1, 2]
    assert plan["array_concurrency"] == 2
    assert plan["slurm_account"] == "sigroup"
    assert plan["partition"] == "gpu-l20"
    assert plan["gpu_name"] == "NVIDIA L20"
    assert plan["gpus_per_task"] == 1
    assert plan["cpus_per_task"] == 8
    assert plan["memory_bytes"] == 64 * 1024 * 1024 * 1024
    assert plan["memory_mib"] == 65536
    assert plan["requested_walltime_seconds"] == 43200
    assert plan["slurm_walltime"] == "0-12:00:00"
    assert plan["advance_signal_lead_seconds"] == 900
    assert plan["max_scheduler_segments"] == 3
    assert plan["audit_cadence_updates"] == 20
    assert plan["durable_checkpoint_cadence_updates"] == 200
    assert plan["resource_plan_sha256"] == _digest("7")

    semantic = plan.pop("submission_plan_sha256")
    assert semantic == module._semantic_sha256(plan)


def test_inspect_requires_the_caller_pinned_file_and_emits_fixed_field_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script()
    _bind_pure_data_dependencies(module, monkeypatch, tmp_path)
    assert module.main(_argv(tmp_path)) == 0
    capsys.readouterr()

    plan_path = tmp_path / "primary-plan.json"
    plan_file_sha256 = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    assert (
        module.main(
            [
                "inspect",
                "--plan",
                str(plan_path),
                "--plan-file-sha256",
                plan_file_sha256,
                "--format",
                "sbatch-lines",
            ]
        )
        == 0
    )
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == len(module._SBATCH_FIELD_ORDER)
    assert [line.split("=", 1)[0] for line in lines] == list(module._SBATCH_FIELD_ORDER)
    assert lines[2] == "slurm_account=sigroup"
    assert lines[11] == "array_concurrency=2"

    assert (
        module.main(
            [
                "inspect",
                "--plan",
                str(plan_path),
                "--plan-file-sha256",
                plan_file_sha256,
                "--format",
                "binding-lines",
            ]
        )
        == 0
    )
    binding_lines = capsys.readouterr().out.splitlines()
    assert len(binding_lines) == len(module._BINDING_FIELD_ORDER)
    assert [line.split("=", 1)[0] for line in binding_lines] == list(module._BINDING_FIELD_ORDER)
    assert binding_lines[0] == f"git_commit={'a' * 40}"

    with pytest.raises(ValueError, match="file SHA-256"):
        module.main(
            [
                "inspect",
                "--plan",
                str(plan_path),
                "--plan-file-sha256",
                _digest("f"),
            ]
        )


def test_prepare_parser_has_no_resource_science_seed_or_head_override(
    tmp_path: Path,
) -> None:
    module = _load_script()
    prohibited = (
        "--walltime-seconds",
        "--memory-bytes",
        "--cpus-per-task",
        "--array-concurrency",
        "--seed",
        "--head",
        "--heldout",
        "--control",
    )
    for option in prohibited:
        with pytest.raises(SystemExit):
            module._parser().parse_args([*_argv(tmp_path), option, "1"])
