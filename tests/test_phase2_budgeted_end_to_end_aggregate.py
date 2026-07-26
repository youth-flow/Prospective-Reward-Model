from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "hpc4" / "aggregate_phase2_budgeted_end_to_end.py"
SEEDS = (20261001, 20261002, 20261003)
ARRAY = "7000"
DESIGN = "a" * 64
BASE = "b" * 64
GIT = "c" * 40
IMAGE = "d" * 64
INVENTORY = "e" * 64
FREEZE = "f" * 64
RUNTIME = "1" * 64
INTENT = "2" * 64
SUBMISSION = "3" * 64
BETA = 2.5


def _load() -> ModuleType:
    name = f"_budgeted_aggregate_test_{os.urandom(4).hex()}"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalized(seed: int, offset: int) -> dict[str, object]:
    endpoints: dict[str, dict[str, float]] = {}
    for index, endpoint in enumerate(
        (
            "heldout_local_regret",
            "finite_policy_utility",
            "oracle_pairwise_cross_entropy",
            "oracle_probability_mae",
            "pairwise_order_accuracy",
        )
    ):
        bt = float(offset + index + 1)
        endpoints[endpoint] = {"bt_mle": bt, "prorm_plus": bt + 0.25}
    return {
        "seed": seed,
        "phase2_design_sha256": DESIGN,
        "phase2_runtime_contract_sha256": RUNTIME,
        "beta_source_aggregate_sha256": FREEZE,
        "frozen_global_beta": BETA,
        "admissible": True,
        "endpoints": endpoints,
    }


def _success(fields: dict[str, str], keys: tuple[str, ...]) -> bytes:
    return ("".join(f"{key}={fields[key]}\n" for key in keys)).encode()


@dataclass
class Campaign:
    module: ModuleType
    project: Path
    repository: Path
    terminal: Path
    runs: list[Path]
    output: Path

    def publish(self) -> tuple[dict[str, object], dict[str, object]]:
        rows = [
            {
                "job_id": f"{ARRAY}_{task}",
                "job_id_raw": str(7100 + task),
                "array_job_id": ARRAY,
                "array_task_id": task,
                "seed": seed,
                "state": "COMPLETED",
                "exit_code": "0:0",
                "derived_exit_code": "0:0",
                "cluster": "hpc4",
                "account": "sigroup",
                "partition": "gpu-l20",
                "qos": "l20_qos",
            }
            for task, seed in enumerate(SEEDS)
        ]

        def terminal_verifier(
            path: Path,
            *,
            expected_sha256: str,
            expected_array_job_id: str,
        ) -> dict[str, object]:
            assert path == self.terminal
            assert expected_sha256 == _sha(self.terminal)
            assert expected_array_job_id == ARRAY
            raw = self.terminal.with_name("terminal.sacct.psv")
            return {
                "rows": rows,
                "raw_sacct": {
                    "filename": raw.name,
                    "sha256": _sha(raw),
                    "size_bytes": raw.stat().st_size,
                },
            }

        def ledger_verifier(
            ledger: Path,
            **kwargs: Any,
        ) -> tuple[dict[str, object], tuple[object, ...]]:
            assert ledger.name == "submission-ledger"
            assert kwargs["array_job_id"] == ARRAY
            assert kwargs["identity"]["phase2_design_sha256"] == DESIGN
            return (
                {
                    "status": "verified",
                    "array_job_id": ARRAY,
                    "phase2_design_sha256": DESIGN,
                    "ordered_seeds": list(SEEDS),
                    "intent_sha256": INTENT,
                    "submission_sha256": SUBMISSION,
                },
                (),
            )

        def checkout_verifier(
            repository: Path,
            *,
            expected_git_commit: str,
        ) -> tuple[dict[str, object], tuple[object, ...]]:
            assert repository == self.repository
            assert expected_git_commit == GIT
            return (
                {
                    "git_head": GIT,
                    "repository_clean": True,
                    "critical_files": {"fixture": {"sha256": "4" * 64, "size_bytes": 1}},
                },
                (),
            )

        def git_bytes(
            _repository: Path,
            *arguments: str,
            name: str,
        ) -> bytes:
            del name
            if arguments[0] == "rev-parse":
                return f"{GIT}\n".encode()
            if arguments[0] == "status":
                return b""
            raise AssertionError(arguments)

        self.module._verify_terminal_evidence = terminal_verifier
        self.module._verify_submission_ledger = ledger_verifier
        self.module._verify_publication_checkout = checkout_verifier
        self.module._git_bytes = git_bytes
        return self.module.write_budgeted_end_to_end_aggregate(
            self.project,
            self.repository,
            self.terminal,
            self.runs,
            self.output,
            terminal_evidence_sha256=_sha(self.terminal),
            array_job_id=ARRAY,
        )


@pytest.fixture
def campaign(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Campaign:
    module = _load()
    project = tmp_path / "p"
    repository = tmp_path / "r"
    (project / "aggregates").mkdir(parents=True)
    repository.mkdir()
    campaign_root = project / "runs" / "phase2-budgeted-end-to-end" / DESIGN
    (campaign_root / "submission-ledger").mkdir(parents=True)
    terminal = campaign_root / "terminal.json"
    terminal.write_bytes(b"terminal evidence fixture\n")
    terminal.with_name("terminal.sacct.psv").write_bytes(b"terminal raw fixture\n")
    artifact_root = project / "artifacts" / "phase2-budgeted-end-to-end" / DESIGN
    runs: list[Path] = []
    symlink_unavailable = False

    for task, seed in enumerate(SEEDS):
        run = campaign_root / f"seed-{seed}" / f"job-{ARRAY}_{task}"
        run.mkdir(parents=True)
        artifact = artifact_root / f"seed-{seed}" / f"job-{ARRAY}_{task}"
        artifact.mkdir(parents=True)
        (artifact / "metadata.json").write_bytes(_canonical({"seed": seed, "kind": "artifact"}))
        try:
            (run / "artifact").symlink_to(
                os.path.relpath(artifact, run),
                target_is_directory=True,
            )
        except OSError:
            symlink_unavailable = True
            (run / "artifact").mkdir()

        normalized = _normalized(seed, task)
        (run / "phase2-result.json").write_bytes(
            _canonical({"seed": seed, "normalized": normalized})
        )
        (run / "phase2-result.rollouts.jsonl").write_bytes(
            (json.dumps({"seed": seed, "row": 0}, separators=(",", ":")) + "\n").encode()
        )
        (run / "run-manifest.json").write_bytes(_canonical({"seed": seed, "kind": "manifest"}))
        (run / "artifact-materialization.json").write_bytes(
            _canonical({"seed": seed, "mode": "fresh"})
        )
        hashes = {
            "result": _sha(run / "phase2-result.json"),
            "rollouts": _sha(run / "phase2-result.rollouts.jsonl"),
            "manifest": _sha(run / "run-manifest.json"),
            "artifact": _sha(artifact / "metadata.json"),
            "materialization": _sha(run / "artifact-materialization.json"),
        }
        verification = {
            "schema_version": module.VERIFICATION_SCHEMA,
            "status": "verified",
            "design_stage": module.STAGE,
            "evidence_role": module.EVIDENCE_ROLE,
            "formal_eligibility": False,
            "formal_claim_eligible": False,
            "supports_formal_claim": False,
            "inferential_or_significance_claim_produced": False,
            "seed": seed,
            "phase2_design_sha256": DESIGN,
            "base_config_hash": BASE,
            "accepted_freeze_aggregate_sha256": FREEZE,
            "frozen_global_beta": BETA,
            "phase2_runtime_contract_sha256": RUNTIME,
            "git_commit": GIT,
            "image_sha256": IMAGE,
            "hf_inventory_sha256": INVENTORY,
            "slurm_job_id_raw": str(7100 + task),
            "array_job_id": ARRAY,
            "array_task_id": task,
            "result_sha256": hashes["result"],
            "rollouts_sha256": hashes["rollouts"],
            "run_manifest_sha256": hashes["manifest"],
            "artifact_metadata_sha256": hashes["artifact"],
            "artifact_materialization_sha256": hashes["materialization"],
            "slurm": {
                "job_id_raw": str(7100 + task),
                "array_job_id": ARRAY,
                "array_task_id": task,
                "account": "sigroup",
                "cluster": "hpc4",
                "partition": "gpu-l20",
            },
            "relative_files": {
                "result": "phase2-result.json",
                "rollouts": "phase2-result.rollouts.jsonl",
                "run_manifest": "run-manifest.json",
                "artifact_metadata": "artifact/metadata.json",
                "artifact_materialization": "artifact-materialization.json",
            },
            "input_sha256": {
                "overlay": "4" * 64,
                "result": hashes["result"],
                "rollouts": hashes["rollouts"],
                "run_manifest": hashes["manifest"],
                "artifact_metadata": hashes["artifact"],
                "artifact_materialization": hashes["materialization"],
            },
            "rollout_geometry": {
                "row_count": 8,
                "rows_per_arm": {
                    "zero_b": 2,
                    "bt_mle": 2,
                    "prorm_plus": 2,
                    "oracle_step": 2,
                },
                "test_prompt_count": 1,
                "candidates_per_prompt": 2,
                "arm_order": ["zero_b", "bt_mle", "prorm_plus", "oracle_step"],
            },
            "environment_identity": {
                "formal": True,
                "git_commit": GIT,
                "image_sha256": IMAGE,
                "hf_inventory_sha256": INVENTORY,
                "account": "sigroup",
                "partition": "gpu-l20",
                "gpu_models": ["NVIDIA L20"],
            },
            "normalized_seed_record": normalized,
        }
        (run / "phase2-budgeted-output-verification.json").write_bytes(_canonical(verification))
        fields = {
            "schema_version": module.RUN_STATUS_SCHEMA,
            "status": "SUCCESS",
            "formal": "false",
            "evidence_role": module.EVIDENCE_ROLE,
            "stage": module.STAGE,
            "seed": str(seed),
            "slurm_job_id": str(7100 + task),
            "array_job_id": ARRAY,
            "array_task_id": str(task),
            "cluster": "hpc4",
            "account": "sigroup",
            "partition": "gpu-l20",
            "restart_count": "0",
            "phase2_design_sha256": DESIGN,
            "base_config_hash": BASE,
            "git_commit": GIT,
            "submission_intent_sha256": INTENT,
            "submission_ledger_sha256": SUBMISSION,
            "freeze_evidence_sha256": FREEZE,
            "frozen_global_beta": str(BETA),
            "optimizer_schedule_sha256": module.OPTIMIZER_SCHEDULE_SHA256,
            "artifact_metadata_sha256": hashes["artifact"],
            "phase2_result_sha256": hashes["result"],
            "rollouts_sha256": hashes["rollouts"],
            "verification_sha256": _sha(run / "phase2-budgeted-output-verification.json"),
            "manifest_sha256": hashes["manifest"],
            "workload_exit_code": "0",
            "final_exit_code": "0",
            "created_at_utc": "2026-07-26T00:00:00Z",
        }
        (run / "SUCCESS").write_bytes(_success(fields, module._SUCCESS_KEYS))
        runs.append(run)

    monkeypatch.setattr(
        module,
        "normalize_budgeted_end_to_end_seed_result",
        lambda result: result["normalized"],
    )
    if symlink_unavailable:
        monkeypatch.setattr(
            module,
            "_artifact_metadata",
            lambda _run, *, expected: expected / "metadata.json",
        )
    return Campaign(
        module=module,
        project=project,
        repository=repository,
        terminal=terminal,
        runs=runs,
        output=project / "aggregates" / "budgeted-fixed-three.json",
    )


def test_publishes_only_complete_descriptive_effects_and_receipt(campaign: Campaign) -> None:
    aggregate, receipt = campaign.publish()

    assert aggregate["aggregation_state"] == "complete_descriptive_aggregate"
    assert aggregate["formal_claim_eligible"] is False
    assert set(aggregate["effect_summaries"]) == {
        "heldout_local_regret",
        "finite_policy_utility",
        "oracle_pairwise_cross_entropy",
        "oracle_probability_mae",
        "pairwise_order_accuracy",
    }
    serialized = campaign.output.read_text(encoding="utf-8").lower()
    assert '"p_value"' not in serialized
    assert '"significance"' not in serialized
    receipt_path = Path(f"{campaign.output}.evidence.json")
    assert receipt_path.is_file()
    assert receipt["schema_version"] == (
        "prorm-phase2-budgeted-end-to-end-fixed-three-descriptive-publication/v1"
    )
    assert receipt["analysis_role"] == "fixed_three_exploratory_descriptive_only"
    assert receipt["ordered_seeds"] == [20261001, 20261002, 20261003]
    assert receipt["aggregate"]["sha256"] == _sha(campaign.output)
    assert [row["seed"] for row in receipt["seed_evidence"]] == list(SEEDS)


def test_cross_seed_run_order_and_missing_seed_fail_closed(campaign: Campaign) -> None:
    campaign.runs[0], campaign.runs[1] = campaign.runs[1], campaign.runs[0]
    with pytest.raises(ValueError, match="marker identity|run path"):
        campaign.publish()
    assert not campaign.output.exists()

    campaign.runs[0], campaign.runs[1] = campaign.runs[1], campaign.runs[0]
    campaign.runs.pop()
    with pytest.raises(ValueError, match="exactly three"):
        campaign.publish()
    assert not campaign.output.exists()


def test_result_and_cross_seed_verifier_tampering_are_rejected(campaign: Campaign) -> None:
    result = campaign.runs[0] / "phase2-result.json"
    result.write_bytes(result.read_bytes() + b" ")
    with pytest.raises(ValueError, match="does not bind"):
        campaign.publish()
    assert not campaign.output.exists()


def test_old_fixed_five_seed_verification_schema_is_rejected(campaign: Campaign) -> None:
    run = campaign.runs[0]
    verification_path = run / "phase2-budgeted-output-verification.json"
    verification = json.loads(verification_path.read_bytes())
    verification["schema_version"] = "prorm-phase2-budgeted-seed-output-verification/v1"
    verification_path.write_bytes(_canonical(verification))
    marker, _ = campaign.module._parse_success(run / "SUCCESS")
    marker["verification_sha256"] = _sha(verification_path)
    (run / "SUCCESS").write_bytes(_success(marker, campaign.module._SUCCESS_KEYS))

    with pytest.raises(ValueError, match="verification identity"):
        campaign.publish()
    assert not campaign.output.exists()


def test_old_fixed_five_run_status_schema_is_rejected(campaign: Campaign) -> None:
    marker, _ = campaign.module._parse_success(campaign.runs[0] / "SUCCESS")
    marker["schema_version"] = "prorm-phase2-budgeted-end-to-end-run-status/v1"
    (campaign.runs[0] / "SUCCESS").write_bytes(_success(marker, campaign.module._SUCCESS_KEYS))

    with pytest.raises(ValueError, match="SUCCESS marker identity"):
        campaign.publish()
    assert not campaign.output.exists()


def test_inadmissible_recomputation_withholds_publication(
    campaign: Campaign,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = campaign.module.normalize_budgeted_end_to_end_seed_result

    def reject_one(result: dict[str, object]) -> dict[str, object]:
        value = dict(original(result))
        if value["seed"] == SEEDS[2]:
            value["admissible"] = False
            value.pop("endpoints")
        return value

    monkeypatch.setattr(campaign.module, "normalize_budgeted_end_to_end_seed_result", reject_one)
    with pytest.raises(ValueError, match="inadmissible"):
        campaign.publish()
    assert not campaign.output.exists()


def test_atomic_publication_never_overwrites_existing_output(campaign: Campaign) -> None:
    campaign.publish()
    output_raw = campaign.output.read_bytes()
    receipt_path = Path(f"{campaign.output}.evidence.json")
    receipt_raw = receipt_path.read_bytes()

    campaign.publish()

    assert campaign.output.read_bytes() == output_raw
    assert receipt_path.read_bytes() == receipt_raw
    campaign.output.write_bytes(b"conflicting bytes\n")
    with pytest.raises(FileExistsError, match="conflicting"):
        campaign.publish()


def test_cross_seed_freeze_splice_is_rejected_even_after_rehash(campaign: Campaign) -> None:
    run = campaign.runs[1]
    verification_path = run / "phase2-budgeted-output-verification.json"
    verification = json.loads(verification_path.read_bytes())
    verification["accepted_freeze_aggregate_sha256"] = "5" * 64
    verification_path.write_bytes(_canonical(verification))
    marker, _ = campaign.module._parse_success(run / "SUCCESS")
    marker["verification_sha256"] = _sha(verification_path)
    (run / "SUCCESS").write_bytes(_success(marker, campaign.module._SUCCESS_KEYS))

    with pytest.raises(ValueError, match="cross-bound"):
        campaign.publish()
    assert not campaign.output.exists()


def test_toctou_change_after_normalization_is_detected(
    campaign: Campaign,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_builder = campaign.module.build_fixed_three_exploratory_aggregate
    result = campaign.runs[0] / "phase2-result.json"

    def mutate_after_normalization(*args: object, **kwargs: object) -> dict[str, object]:
        value = original_builder(*args, **kwargs)
        result.write_bytes(result.read_bytes() + b" ")
        return value

    monkeypatch.setattr(
        campaign.module,
        "build_fixed_three_exploratory_aggregate",
        mutate_after_normalization,
    )
    with pytest.raises(ValueError, match="changed after authentication"):
        campaign.publish()
    assert not campaign.output.exists()


def test_cross_seed_submission_ledger_splice_is_rejected(campaign: Campaign) -> None:
    run = campaign.runs[2]
    marker, _ = campaign.module._parse_success(run / "SUCCESS")
    marker["submission_ledger_sha256"] = "5" * 64
    (run / "SUCCESS").write_bytes(_success(marker, campaign.module._SUCCESS_KEYS))

    with pytest.raises(ValueError, match="immutable submission ledger"):
        campaign.publish()
    assert not campaign.output.exists()


def test_crash_after_aggregate_resumes_exact_receipt(
    campaign: Campaign,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_write = campaign.module._write_exclusive
    failed = False

    def fail_receipt_once(path: Path, raw: bytes, *, name: str) -> None:
        nonlocal failed
        if "receipt" in name and not failed:
            failed = True
            raise RuntimeError("injected receipt crash")
        original_write(path, raw, name=name)

    monkeypatch.setattr(campaign.module, "_write_exclusive", fail_receipt_once)
    with pytest.raises(RuntimeError, match="receipt crash"):
        campaign.publish()
    assert campaign.output.is_file()
    assert not Path(f"{campaign.output}.evidence.json").exists()

    monkeypatch.setattr(campaign.module, "_write_exclusive", original_write)
    campaign.publish()
    assert Path(f"{campaign.output}.evidence.json").is_file()


def test_cli_requires_exactly_three_ordered_run_paths(tmp_path: Path) -> None:
    module = _load()
    output = tmp_path / "aggregate.json"
    shared_options = [
        "--project-root",
        str(tmp_path),
        "--repo-root",
        str(tmp_path),
        "--terminal-evidence",
        str(tmp_path / "terminal.json"),
        "--terminal-evidence-sha256",
        "0" * 64,
        "--array-job-id",
        ARRAY,
    ]
    runs = [str(tmp_path / f"run-{index}") for index in range(5)]

    parsed = module.build_parser().parse_args([str(output), *runs[:3], *shared_options])
    assert parsed.runs == [Path(path) for path in runs[:3]]
    for invalid in (runs[:2], runs[:4], runs):
        with pytest.raises(SystemExit):
            module.build_parser().parse_args([str(output), *invalid, *shared_options])


def test_dirty_or_wrong_head_publication_checkout_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load()
    repository = tmp_path / "repository"
    repository.mkdir()

    def dirty_git(
        _repository: Path,
        *arguments: str,
        name: str,
    ) -> bytes:
        del name
        if arguments[0] == "rev-parse":
            return f"{GIT}\n".encode()
        if arguments[0] == "status":
            return b" M src/smart_reward/phase2_exploratory_aggregate.py\n"
        raise AssertionError(arguments)

    monkeypatch.setattr(module, "_git_bytes", dirty_git)
    with pytest.raises(ValueError, match="completely clean"):
        module._verify_publication_checkout(repository, expected_git_commit=GIT)

    monkeypatch.setattr(
        module,
        "_git_bytes",
        lambda _repository, *arguments, name: (
            b"9" * 40 + b"\n" if arguments[0] == "rev-parse" else b""
        ),
    )
    with pytest.raises(ValueError, match="HEAD differs"):
        module._verify_publication_checkout(repository, expected_git_commit=GIT)


def test_bootstrap_contract_is_locked_and_not_cli_selectable(campaign: Campaign) -> None:
    aggregate, _ = campaign.publish()
    assert aggregate["bootstrap"] == {
        "method": "paired_seed_percentile_bootstrap",
        "unit": "seed",
        "resamples": campaign.module.FIXED_BOOTSTRAP_RESAMPLES,
        "seed": campaign.module.FIXED_BOOTSTRAP_SEED,
        "confidence_level": campaign.module.FIXED_CONFIDENCE_LEVEL,
        "interpretation": "descriptive_only",
    }
    parser = campaign.module.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                str(campaign.output),
                *(str(run) for run in campaign.runs),
                "--project-root",
                str(campaign.project),
                "--repo-root",
                str(campaign.repository),
                "--terminal-evidence",
                str(campaign.terminal),
                "--terminal-evidence-sha256",
                _sha(campaign.terminal),
                "--array-job-id",
                ARRAY,
                "--bootstrap-seed",
                "1",
            ]
        )
