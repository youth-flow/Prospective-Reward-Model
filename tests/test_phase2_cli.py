from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

import smart_reward.cli as cli_module
from smart_reward.config import config_hash, load_config

ROOT = Path(__file__).resolve().parents[1]


def _fake_module(name: str, **attributes: object) -> ModuleType:
    module = ModuleType(name)
    for attribute, value in attributes.items():
        setattr(module, attribute, value)
    return module


def test_importing_cli_does_not_import_phase2_runtime_modules() -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    command = (
        "import sys; import smart_reward.cli; "
        "names=('smart_reward.phase2_inputs','smart_reward.phase2_training',"
        "'smart_reward.phase2_hf','smart_reward.phase2_rollout',"
        "'smart_reward.phase2_aggregate','smart_reward.phase2_pilot_aggregate',"
        "'smart_reward.phase2_campaign'); "
        "assert not any(name in sys.modules for name in names)"
    )

    completed = subprocess.run(
        [sys.executable, "-c", command],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_phase2_config_check_reports_both_bound_identities(
    capsys: pytest.CaptureFixture[str],
) -> None:
    overlay = ROOT / "configs" / "common_beta_pilot.yaml"
    source = ROOT / "configs" / "common_beta_pilot_base.yaml"

    assert cli_module.main(["phase2-config-check", str(overlay)]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "design_stage": "pilot",
        "formal_eligibility": False,
        "overlay_path": str(overlay),
        "phase2_design_sha256": payload["phase2_design_sha256"],
        "source_config_hash": config_hash(load_config(source)),
        "source_config_path": str(source),
        "status": "ok",
    }
    assert len(payload["phase2_design_sha256"]) == 64
    assert payload["phase2_design_sha256"] != payload["source_config_hash"]


def test_phase2_run_chains_fresh_training_backend_design_and_rollout(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    seed = 20260801
    overlay_config = {
        "design": {
            "stage": "pilot",
            "pilot_phase": "calibration",
            "formal_eligibility": False,
        },
        "run": {"seeds": [20260801, 20260802, 20260803]},
    }
    bundle = SimpleNamespace(
        config=overlay_config,
        base_config={"base": "validated"},
        design_identity="d" * 64,
    )
    prepared = object()
    trainer = object()
    backend = object()
    design = object()
    events: list[str] = []
    observed: dict[str, object] = {}

    def load_bundle(path: str) -> object:
        events.append("load")
        observed["overlay"] = path
        return bundle

    def prepare(
        path: str,
        *,
        seed: int,
        artifact_dir: str,
        run_manifest: str,
        training_device: str,
    ) -> object:
        events.append("prepare")
        observed["prepare"] = (
            path,
            seed,
            artifact_dir,
            run_manifest,
            training_device,
        )
        return prepared

    def make_trainer(settings: object) -> object:
        events.append("trainer")
        assert settings is bundle
        return trainer

    def make_backend(
        source_config: object,
        *,
        device: str,
        local_files_only: bool,
    ) -> object:
        events.append("backend")
        observed["backend"] = (source_config, device, local_files_only)
        return backend

    class FakeDesign:
        @classmethod
        def from_phase2_config(cls, config: object) -> object:
            events.append("design")
            assert config is overlay_config
            return design

    def run_rollouts(
        inputs: object,
        head_trainer: object,
        runtime_backend: object,
        *,
        output_json: str,
        design: object,
    ) -> dict[str, object]:
        events.append("run")
        observed["run"] = (
            inputs,
            head_trainer,
            runtime_backend,
            output_json,
            design,
        )
        return {
            "design_stage": "pilot",
            "phase2_design_sha256": "d" * 64,
            "diagnostics_jsonl": "seed.diagnostics.jsonl",
            "measured_kl_safety": {"passed": False},
            "pilot_kl_safety_gate": {
                "gate_passed": False,
                "measure_only": True,
            },
        }

    modules = {
        "smart_reward.phase2_config": _fake_module(
            "smart_reward.phase2_config",
            load_phase2_config_bundle=load_bundle,
        ),
        "smart_reward.phase2_inputs": _fake_module(
            "smart_reward.phase2_inputs",
            prepare_phase2_inputs=prepare,
        ),
        "smart_reward.phase2_training": _fake_module(
            "smart_reward.phase2_training",
            FreshPhase2HeadTrainer=make_trainer,
        ),
        "smart_reward.phase2_hf": _fake_module(
            "smart_reward.phase2_hf",
            HuggingFacePhase2Backend=make_backend,
        ),
        "smart_reward.phase2_rollout": _fake_module(
            "smart_reward.phase2_rollout",
            Phase2Design=FakeDesign,
            run_common_beta_rollouts=run_rollouts,
        ),
        "smart_reward.phase2_pilot_aggregate": _fake_module(
            "smart_reward.phase2_pilot_aggregate",
            verify_beta_source_aggregate=lambda _config, _source: None,
            verify_horizon_parent_aggregate=lambda _config, _source: None,
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    overlay = str(tmp_path / "overlay.yaml")
    artifact = str(tmp_path / "artifact")
    manifest = str(tmp_path / "run-manifest.json")
    output = str(tmp_path / "seed.json")
    assert (
        cli_module.main(
            [
                "phase2-run",
                overlay,
                artifact,
                manifest,
                output,
                "--seed",
                str(seed),
                "--device",
                "cuda:0",
            ]
        )
        == 0
    )

    assert events == ["load", "prepare", "trainer", "backend", "design", "run"]
    assert observed["overlay"] == overlay
    assert observed["prepare"] == (overlay, seed, artifact, manifest, "cuda:0")
    assert observed["backend"] == (bundle.base_config, "cuda:0", True)
    assert observed["run"] == (prepared, trainer, backend, output, design)
    assert json.loads(capsys.readouterr().out) == {
        "design_stage": "pilot",
        "formal_eligibility": False,
        "kl_gate_measure_only": True,
        "kl_gate_passed": False,
        "output": "seed.json",
        "phase2_design_sha256": "d" * 64,
        "sidecar_jsonl": "seed.diagnostics.jsonl",
        "seed": seed,
        "status": "ok",
    }


def test_phase2_aggregate_delegates_to_strict_aggregator(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    overlay_config = {
        "design": {"stage": "pilot", "formal_eligibility": False},
        "validated": "overlay",
    }
    bundle = SimpleNamespace(config=overlay_config)
    results = [str(tmp_path / f"seed-{index}.json") for index in range(3)]
    output = str(tmp_path / "aggregate.json")
    observed: dict[str, object] = {}
    aggregate = SimpleNamespace(
        evidence=SimpleNamespace(to_dict=lambda: {"status": "passed"}),
        seeds=tuple(range(3)),
        phase2_design_sha256="e" * 64,
    )

    def load_bundle(path: str) -> object:
        observed["overlay"] = path
        return bundle

    def write_aggregate(
        config: object,
        result_jsons: list[str],
        output_json: str,
    ) -> object:
        observed["write"] = (config, result_jsons, output_json)
        return aggregate

    monkeypatch.setitem(
        sys.modules,
        "smart_reward.phase2_config",
        _fake_module(
            "smart_reward.phase2_config",
            load_phase2_config_bundle=load_bundle,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "smart_reward.phase2_aggregate",
        _fake_module(
            "smart_reward.phase2_aggregate",
            write_common_beta_seed_aggregate=write_aggregate,
        ),
    )

    overlay = str(tmp_path / "overlay.yaml")
    assert cli_module.main(["phase2-aggregate", overlay, output, *results]) == 0

    assert observed["overlay"] == overlay
    assert observed["write"] == (overlay_config, results, output)
    assert json.loads(capsys.readouterr().out) == {
        "design_stage": "pilot",
        "formal_eligibility": False,
        "evidence_status": "passed",
        "num_seeds": 3,
        "output": "aggregate.json",
        "phase2_design_sha256": "e" * 64,
        "status": "ok",
    }


@pytest.mark.parametrize(
    ("command", "runtime_module", "runner_name"),
    [
        (
            "phase2-sensitivity-run",
            "smart_reward.phase2_sensitivity",
            "run_phase2_sensitivity_seed",
        ),
        (
            "phase2-mechanism-run",
            "smart_reward.phase2_mechanism",
            "run_phase2_mechanism_seed",
        ),
    ],
)
def test_secondary_phase2_seed_cli_binds_primary_and_runtime(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    command: str,
    runtime_module: str,
    runner_name: str,
) -> None:
    seed = 20260901
    config = {
        "design": {"stage": "confirmatory", "formal_eligibility": True},
        "run": {"seeds": [seed]},
    }
    bundle = SimpleNamespace(
        config=config,
        base_config={"base": "validated"},
    )
    inputs = object()
    binding = object()
    backend = object()
    design = object()
    observed: dict[str, object] = {}

    def prepare(
        overlay: str,
        *,
        seed: int,
        artifact_dir: str,
        run_manifest: str,
        training_device: str | None = None,
    ) -> object:
        observed["prepare"] = (
            overlay,
            seed,
            artifact_dir,
            run_manifest,
            training_device,
        )
        return inputs

    class FakeDesign:
        @classmethod
        def from_phase2_config(cls, value: object) -> object:
            assert value is config
            return design

    def make_backend(
        value: object,
        *,
        device: str,
        local_files_only: bool,
    ) -> object:
        observed["backend"] = (value, device, local_files_only)
        return backend

    def load_binding(value: object, path: str) -> object:
        observed["binding"] = (value, path)
        return binding

    def run_sensitivity(
        value: object,
        frozen: object,
        runtime: object,
        *,
        settings: object,
        design: object,
        output_json: str,
    ) -> dict[str, object]:
        observed["run"] = (
            value,
            frozen,
            runtime,
            settings,
            design,
            output_json,
        )
        Path(output_json).write_text("{}\n", encoding="utf-8")
        return {
            "phase2_design_sha256": "d" * 64,
            "ridge_cells": [{}, {}, {}],
            "beta_cells": [
                {"multiplier": 0.5, "status": "completed"},
                {"multiplier": 1.0, "status": "primary_reference"},
                {"multiplier": 2.0, "status": "completed"},
            ],
        }

    def run_mechanism(
        value: object,
        frozen: object,
        runtime: object,
        *,
        design: object,
        output_json: str,
    ) -> dict[str, object]:
        observed["run"] = (
            value,
            frozen,
            runtime,
            design,
            output_json,
        )
        Path(output_json).write_text("{}\n", encoding="utf-8")
        return {"phase2_design_sha256": "d" * 64}

    monkeypatch.setitem(
        sys.modules,
        "smart_reward.phase2_config",
        _fake_module(
            "smart_reward.phase2_config",
            load_phase2_config_bundle=lambda _: bundle,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "smart_reward.phase2_inputs",
        _fake_module(
            "smart_reward.phase2_inputs",
            prepare_phase2_inputs=prepare,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "smart_reward.phase2_hf",
        _fake_module(
            "smart_reward.phase2_hf",
            HuggingFacePhase2Backend=make_backend,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "smart_reward.phase2_rollout",
        _fake_module("smart_reward.phase2_rollout", Phase2Design=FakeDesign),
    )
    attributes = {
        "load_primary_sensitivity_binding": load_binding,
        runner_name: (run_sensitivity if command == "phase2-sensitivity-run" else run_mechanism),
    }
    monkeypatch.setitem(
        sys.modules,
        runtime_module,
        _fake_module(runtime_module, **attributes),
    )
    if command == "phase2-mechanism-run":
        monkeypatch.setitem(
            sys.modules,
            "smart_reward.phase2_sensitivity",
            _fake_module(
                "smart_reward.phase2_sensitivity",
                load_primary_sensitivity_binding=load_binding,
            ),
        )

    overlay = str(tmp_path / "overlay.yaml")
    artifact = str(tmp_path / "artifact")
    manifest = str(tmp_path / "manifest.json")
    primary = str(tmp_path / "primary.json")
    output = tmp_path / "secondary.json"
    assert (
        cli_module.main(
            [
                command,
                overlay,
                artifact,
                manifest,
                primary,
                str(output),
                "--seed",
                str(seed),
                "--device",
                "cuda:0",
            ]
        )
        == 0
    )
    expected_training_device = "cuda:0"
    assert observed["prepare"] == (
        overlay,
        seed,
        artifact,
        manifest,
        expected_training_device,
    )
    assert observed["binding"] == (config, primary)
    assert observed["backend"] == (bundle.base_config, "cuda:0", True)
    printed = json.loads(capsys.readouterr().out)
    assert printed["formal_eligibility"] is False
    assert printed["seed"] == seed
    assert printed["supports_primary_claim"] is False
    assert printed["output_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    ("command", "runtime_module", "writer_name", "scope_key"),
    [
        (
            "phase2-sensitivity-aggregate",
            "smart_reward.phase2_sensitivity",
            "write_phase2_sensitivity_aggregate",
            None,
        ),
        (
            "phase2-mechanism-aggregate",
            "smart_reward.phase2_mechanism",
            "write_phase2_mechanism_aggregate",
            "mechanism_qualified",
        ),
    ],
)
def test_secondary_phase2_aggregate_cli_delegates_without_primary_rewrite(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    command: str,
    runtime_module: str,
    writer_name: str,
    scope_key: str | None,
) -> None:
    config = {"design": {"stage": "confirmatory"}}
    bundle = SimpleNamespace(config=config)
    observed: dict[str, object] = {}

    def writer(
        value: object,
        results: list[str],
        output: str,
        *,
        primary_aggregate_json: str,
    ) -> dict[str, object]:
        observed["write"] = (value, results, output, primary_aggregate_json)
        Path(output).write_text("{}\n", encoding="utf-8")
        payload: dict[str, object] = {
            "num_seeds": 30,
            "phase2_design_sha256": "d" * 64,
            "primary_aggregate": {"efficacy_status": "not_passed"},
        }
        if scope_key is not None:
            payload["claim_scope"] = {"status": scope_key}
        return payload

    monkeypatch.setitem(
        sys.modules,
        "smart_reward.phase2_config",
        _fake_module(
            "smart_reward.phase2_config",
            load_phase2_config_bundle=lambda _: bundle,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        runtime_module,
        _fake_module(runtime_module, **{writer_name: writer}),
    )
    overlay = str(tmp_path / "overlay.yaml")
    primary = str(tmp_path / "primary.json")
    output = tmp_path / "secondary-aggregate.json"
    results = [str(tmp_path / f"seed-{index}.json") for index in range(30)]
    assert cli_module.main([command, overlay, primary, str(output), *results]) == 0
    assert observed["write"] == (config, results, str(output), primary)
    printed = json.loads(capsys.readouterr().out)
    assert printed["num_seeds"] == 30
    assert printed["primary_efficacy_status"] == "not_passed"
    assert printed["supports_primary_claim"] is False
    assert printed["output_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()


def test_phase2_pilot_aggregate_delegates_to_target_free_aggregator(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    overlay_config = {
        "design": {
            "stage": "pilot",
            "pilot_phase": "freeze",
            "formal_eligibility": False,
        },
    }
    bundle = SimpleNamespace(config=overlay_config)
    results = [str(tmp_path / f"seed-{index}.json") for index in range(3)]
    output = tmp_path / "pilot-aggregate.json"
    source = str(tmp_path / "calibration-aggregate.json")
    observed: dict[str, object] = {}

    def load_bundle(path: str) -> object:
        observed["overlay"] = path
        return bundle

    def write_aggregate(
        config: object,
        result_jsons: list[str],
        output_json: str,
        *,
        beta_source_aggregate: str | None,
        horizon_parent_aggregate: str | None,
    ) -> dict[str, object]:
        observed["write"] = (
            config,
            result_jsons,
            output_json,
            beta_source_aggregate,
            horizon_parent_aggregate,
        )
        Path(output_json).write_text("{}\n", encoding="utf-8")
        return {
            "pilot_phase": "freeze",
            "phase2_design_sha256": "e" * 64,
            "selection": {
                "selection_accepted": True,
                "next_action": "freeze_confirmatory_design_identity",
            },
        }

    monkeypatch.setitem(
        sys.modules,
        "smart_reward.phase2_config",
        _fake_module(
            "smart_reward.phase2_config",
            load_phase2_config_bundle=load_bundle,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "smart_reward.phase2_pilot_aggregate",
        _fake_module(
            "smart_reward.phase2_pilot_aggregate",
            write_phase2_pilot_aggregate=write_aggregate,
        ),
    )

    overlay = str(tmp_path / "overlay.yaml")
    assert (
        cli_module.main(
            [
                "phase2-pilot-aggregate",
                overlay,
                str(output),
                *results,
                "--beta-source-aggregate",
                source,
            ]
        )
        == 0
    )

    assert observed["overlay"] == overlay
    assert observed["write"] == (
        overlay_config,
        results,
        str(output),
        source,
        None,
    )
    printed = json.loads(capsys.readouterr().out)
    assert printed == {
        "formal_eligibility": False,
        "next_action": "freeze_confirmatory_design_identity",
        "output": output.name,
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "phase2_design_sha256": "e" * 64,
        "pilot_phase": "freeze",
        "selection_accepted": True,
        "status": "ok",
        "supports_formal_claim": False,
    }


def test_phase2_run_rejects_a_seed_not_declared_by_overlay(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bundle = SimpleNamespace(
        config={"run": {"seeds": list(range(10))}},
        base_config={},
    )
    monkeypatch.setitem(
        sys.modules,
        "smart_reward.phase2_config",
        _fake_module(
            "smart_reward.phase2_config",
            load_phase2_config_bundle=lambda _: bundle,
        ),
    )
    for name, attributes in {
        "smart_reward.phase2_inputs": {"prepare_phase2_inputs": object()},
        "smart_reward.phase2_training": {"FreshPhase2HeadTrainer": object()},
        "smart_reward.phase2_hf": {"HuggingFacePhase2Backend": object()},
        "smart_reward.phase2_rollout": {
            "Phase2Design": object(),
            "run_common_beta_rollouts": object(),
        },
        "smart_reward.phase2_pilot_aggregate": {
            "verify_beta_source_aggregate": lambda _config, _source: None,
            "verify_horizon_parent_aggregate": lambda _config, _source: None,
        },
    }.items():
        monkeypatch.setitem(sys.modules, name, _fake_module(name, **attributes))

    assert (
        cli_module.main(
            [
                "phase2-run",
                "overlay.yaml",
                "artifact",
                "manifest.json",
                "output.json",
                "--seed",
                "99",
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "seed 99 is not declared by the configuration" in captured.err


def test_phase2_failure_manifest_cli_delegates_to_immutable_writer(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    overlay_config = {"design": {"stage": "confirmatory"}}
    bundle = SimpleNamespace(config=overlay_config)
    spec = tmp_path / "failure-spec.json"
    spec.write_text('{"seed":20260901}\n', encoding="utf-8")
    output = tmp_path / "FAILED.json"
    observed: dict[str, object] = {}

    monkeypatch.setitem(
        sys.modules,
        "smart_reward.phase2_config",
        _fake_module(
            "smart_reward.phase2_config",
            load_phase2_config_bundle=lambda path: observed.update({"overlay": path}) or bundle,
        ),
    )

    def write_manifest(
        config: object,
        value: object,
        destination: str,
    ) -> dict[str, object]:
        observed["write"] = (config, value, destination)
        Path(destination).write_text("{}\n", encoding="utf-8")
        return {
            "phase2_design_sha256": "d" * 64,
            "seed": 20260901,
        }

    monkeypatch.setitem(
        sys.modules,
        "smart_reward.phase2_campaign",
        _fake_module(
            "smart_reward.phase2_campaign",
            write_phase2_seed_failure_manifest=write_manifest,
        ),
    )

    assert (
        cli_module.main(
            [
                "phase2-failure-manifest",
                "overlay.yaml",
                str(spec),
                str(output),
            ]
        )
        == 0
    )
    assert observed["overlay"] == "overlay.yaml"
    assert observed["write"] == (
        overlay_config,
        {"seed": 20260901},
        str(output),
    )
    assert json.loads(capsys.readouterr().out) == {
        "output": "FAILED.json",
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "phase2_design_sha256": "d" * 64,
        "seed": 20260901,
        "status": "failed_seed_terminal_recorded",
        "supports_formal_claim": False,
    }


def test_phase2_success_manifest_cli_binds_result_and_attempt_spec(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    overlay_config = {"design": {"stage": "confirmatory"}}
    bundle = SimpleNamespace(config=overlay_config)
    spec = tmp_path / "success-spec.json"
    result = tmp_path / "result.json"
    output = tmp_path / "SUCCESS.json"
    attempt_ledger = {"attempts": [{"attempt_index": 1}]}
    observed: dict[str, object] = {}
    monkeypatch.setitem(
        sys.modules,
        "smart_reward.phase2_config",
        _fake_module(
            "smart_reward.phase2_config",
            load_phase2_config_bundle=lambda value: (
                observed.__setitem__("overlay", value) or bundle
            ),
        ),
    )

    def load_spec(value: str) -> dict[str, object]:
        observed["spec"] = value
        return {"attempt_ledger": attempt_ledger}

    def write_manifest(
        config: object,
        result_json: str,
        ledger: object,
        destination: str,
    ) -> dict[str, object]:
        observed["write"] = (config, result_json, ledger, destination)
        Path(destination).write_text("{}\n", encoding="utf-8")
        return {
            "phase2_design_sha256": "d" * 64,
            "seed": 20260901,
            "result": {"sha256": "e" * 64},
            "rollout": {"sha256": "f" * 64},
            "attempt_ledger": attempt_ledger,
        }

    monkeypatch.setitem(
        sys.modules,
        "smart_reward.phase2_campaign",
        _fake_module(
            "smart_reward.phase2_campaign",
            load_phase2_seed_success_spec=load_spec,
            write_phase2_seed_success_manifest=write_manifest,
        ),
    )

    assert (
        cli_module.main(
            [
                "phase2-success-manifest",
                "overlay.yaml",
                str(result),
                str(spec),
                str(output),
            ]
        )
        == 0
    )
    assert observed == {
        "overlay": "overlay.yaml",
        "spec": str(spec),
        "write": (
            overlay_config,
            str(result),
            attempt_ledger,
            str(output),
        ),
    }
    assert json.loads(capsys.readouterr().out) == {
        "attempt_count": 1,
        "output": "SUCCESS.json",
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "phase2_design_sha256": "d" * 64,
        "result_sha256": "e" * 64,
        "rollout_sha256": "f" * 64,
        "seed": 20260901,
        "status": "successful_seed_terminal_recorded",
        "supports_formal_claim": False,
    }


def test_phase2_campaign_finalize_cli_preserves_no_ci_failure_status(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    overlay_config = {"design": {"stage": "confirmatory"}}
    bundle = SimpleNamespace(config=overlay_config)
    terminals = [str(tmp_path / f"seed-{index}.json") for index in range(30)]
    output = tmp_path / "campaign-terminal.json"
    aggregate = tmp_path / "primary-aggregate.json"
    observed: dict[str, object] = {}
    monkeypatch.setitem(
        sys.modules,
        "smart_reward.phase2_config",
        _fake_module(
            "smart_reward.phase2_config",
            load_phase2_config_bundle=lambda _: bundle,
        ),
    )

    def finalize(
        config: object,
        values: list[str],
        destination: str,
        *,
        aggregate_output_json: str,
    ) -> dict[str, object]:
        observed["write"] = (
            config,
            values,
            destination,
            aggregate_output_json,
        )
        Path(destination).write_text("{}\n", encoding="utf-8")
        return {
            "failed_seeds": [20260907],
            "primary_ci_computed": False,
            "status": "not_passed_due_to_seed_failure",
            "supports_formal_claim": False,
        }

    monkeypatch.setitem(
        sys.modules,
        "smart_reward.phase2_campaign",
        _fake_module(
            "smart_reward.phase2_campaign",
            write_phase2_campaign_terminal=finalize,
        ),
    )

    assert (
        cli_module.main(
            [
                "phase2-campaign-finalize",
                "overlay.yaml",
                str(output),
                str(aggregate),
                *terminals,
            ]
        )
        == 0
    )
    assert observed["write"] == (
        overlay_config,
        terminals,
        str(output),
        str(aggregate),
    )
    assert not aggregate.exists()
    assert json.loads(capsys.readouterr().out) == {
        "failed_seeds": [20260907],
        "output": "campaign-terminal.json",
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "primary_ci_computed": False,
        "status": "not_passed_due_to_seed_failure",
        "supports_formal_claim": False,
    }
