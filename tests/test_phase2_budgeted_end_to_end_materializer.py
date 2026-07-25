from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from smart_reward.config import config_hash, load_config
from smart_reward.phase2_config import (
    PHASE2_BUDGETED_END_TO_END_BASE_CONFIG,
    PHASE2_BUDGETED_END_TO_END_CONFIG,
    PHASE2_BUDGETED_END_TO_END_EVIDENCE_ROLE,
    PHASE2_BUDGETED_END_TO_END_SEEDS,
    PHASE2_BUDGETED_END_TO_END_STAGE,
    PHASE2_RECOVERY_LR_SCHEDULE_SHA256,
    phase2_design_identity,
    validate_phase2_config,
)
from smart_reward.phase2_training import compile_phase2_training_settings

ROOT = Path(__file__).resolve().parents[1]
MATERIALIZER_PATH = ROOT / "scripts" / "hpc4" / "materialize_phase2_budgeted_end_to_end.py"
BASE_PATH = ROOT / "configs" / "common_beta_pilot_base.yaml"


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _materializer():
    return _module(MATERIALIZER_PATH, "_phase2_budgeted_end_to_end_materializer")


def _post_recovery_config() -> dict[str, Any]:
    helpers = _module(
        ROOT / "tests" / "test_phase2_aggregate.py",
        "_budgeted_materializer_config_helpers",
    )
    return helpers._post_recovery_config()


def _freeze_source(
    *,
    beta: float = 2.5,
    beta_grid_index: int = 0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    materializer = _materializer()
    source = _post_recovery_config()
    base = load_config(BASE_PATH)
    design = source["design"]
    design.update(
        {
            "name": (
                "common-beta-post-recovery-freeze-v1"
                if beta_grid_index == 0
                else f"common-beta-post-recovery-freeze-retry-{beta_grid_index}-v1"
            ),
            "pilot_phase": "freeze",
        }
    )
    common = source["objective"]["common_beta"]
    common.update(
        {
            "calibration_split": "excluded_pilot_calibration",
            "calibration_source": (
                "frozen_calibration_aggregate_candidate_in_pilot_freeze_design_identity"
            ),
            "rule": "pilot_fixed_global_beta_target_free_safety_rehearsal",
            "frozen_global_beta": beta,
            "beta_source_aggregate_sha256": "b" * 64,
            "sensitivity_k_cal": None,
            "sensitivity_frozen_global_beta_multipliers": None,
            "primary_execution_role": "pilot_frozen_global_beta_safety_rehearsal",
            "sensitivity_execution_role": ("new_pilot_freeze_design_identity_double_beta_grid"),
        }
    )
    maximum = source["evaluation"]["max_length"]
    maximum.update(
        {
            "role": "pilot_frozen_global_beta_safety_selection",
            "parent_pilot_aggregate_sha256": "c" * 64,
            "post_pilot_requirement": (
                "freeze_confirmatory_identity_if_passed_else_double_beta_new_identity"
            ),
        }
    )
    source["evaluation"]["decision_gates"]["application"] = (
        "pilot_freeze_target_free_safety_selection"
    )
    validated = validate_phase2_config(source, base_config=base)
    assert materializer._expected_freeze_overlay_relative(beta_grid_index).name.startswith(
        "common_beta_post_recovery_freeze"
    )
    return validated, base


def _accepted_aggregate(
    source: dict[str, Any],
    *,
    beta: float = 2.5,
    beta_grid_index: int = 0,
) -> dict[str, object]:
    maximum = source["evaluation"]["max_length"]
    tokens = source["policy"]["max_response_tokens"]
    return {
        "schema_version": "common-beta-pilot-selection-aggregate/v3",
        "pilot_phase": "freeze",
        "formal_eligibility": False,
        "supports_formal_claim": False,
        "evidence_role": "target_free_design_selection_only",
        "phase2_design_sha256": phase2_design_identity(source),
        "information_boundary": {
            "validation_metrics_read": False,
            "test_metrics_read": False,
            "learner_ordering_read": False,
            "downstream_utility_read": False,
        },
        "horizon": {
            "candidate_horizon_tokens": tokens,
            "horizon_grid_index": maximum["horizon_grid_index"],
            "all_seed_length_gates_passed": True,
        },
        "selection": {
            "schema_version": "pilot-freeze-selection/v1",
            "frozen_global_beta": beta,
            "all_seeds_and_arms_used_same_beta": True,
            "beta_grid_index": beta_grid_index,
            "all_pre_oracle_safety_gates_passed": True,
            "all_length_gates_passed": True,
            "all_non_length_safety_gates_passed": True,
            "selection_accepted": True,
            "accepted_for_confirmatory_identity": True,
            "next_horizon_tokens": tokens,
            "next_global_beta": beta,
            "next_action": "freeze_confirmatory_design_identity",
        },
    }


def _authorization_payload(source: dict[str, Any]) -> dict[str, object]:
    projection = copy.deepcopy(source["recovery_success_reference"]["authorization_projection"])
    projection["schema_version"] = projection.pop("source_schema_version")
    return projection


def _git(repo: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )


def _committed_source_repo(
    tmp_path: Path,
    *,
    beta: float = 2.5,
    beta_grid_index: int = 0,
) -> tuple[Path, Path, dict[str, Any], dict[str, Any]]:
    repo = (tmp_path / "repo").resolve()
    configs = repo / "configs"
    configs.mkdir(parents=True)
    source, base = _freeze_source(
        beta=beta,
        beta_grid_index=beta_grid_index,
    )
    source_relative = (
        Path("configs/common_beta_post_recovery_freeze.yaml")
        if beta_grid_index == 0
        else Path(f"configs/common_beta_post_recovery_freeze_retry_{beta_grid_index}.yaml")
    )
    source_path = repo / source_relative
    base_path = repo / "configs" / "common_beta_pilot_base.yaml"
    source_path.write_text(
        yaml.safe_dump(source, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
        newline="\n",
    )
    base_path.write_text(
        yaml.safe_dump(base, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
        newline="\n",
    )
    _git(repo, "init")
    _git(repo, "config", "user.name", "Budgeted Materializer Test")
    _git(repo, "config", "user.email", "budgeted-materializer@example.invalid")
    _git(repo, "config", "core.autocrlf", "false")
    _git(repo, "add", "--", "configs")
    _git(repo, "commit", "-m", "source")
    return repo, source_path, source, base


def _main_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    inject_second_write_failure: bool = False,
) -> tuple[object, Path, Path, Path, str]:
    materializer = _materializer()
    repo, source_path, source, _ = _committed_source_repo(tmp_path)
    aggregate_root = (tmp_path / "production" / "aggregates").resolve()
    aggregate_root.mkdir(parents=True)
    aggregate_path = aggregate_root / "accepted-freeze.json"
    aggregate_path.write_text(
        json.dumps(
            _accepted_aggregate(source),
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
        newline="\n",
    )
    aggregate_sha256 = materializer._sha256(aggregate_path)
    authorization = _authorization_payload(source)
    authorization_path = (tmp_path / "authorization.json").resolve()
    authorization_path.write_text("{}\n", encoding="utf-8", newline="\n")
    authorization_sha256 = source["recovery_success_reference"]["artifact_sha256"]

    monkeypatch.setattr(materializer, "_AGGREGATE_ROOT", aggregate_root)
    receipt_calls: list[Path] = []

    def verify_receipt(path: Path) -> dict[str, object]:
        receipt_calls.append(path)
        return {"aggregate_sha256": aggregate_sha256}

    monkeypatch.setattr(
        materializer,
        "verify_post_recovery_aggregate_success_receipt",
        verify_receipt,
    )
    monkeypatch.setattr(
        materializer,
        "verify_recovery_authorization_file",
        lambda path, *, expected_sha256: authorization,
    )
    predecessor_calls: list[tuple[str, Path]] = []

    def beta_binding(config: object, path: Path) -> dict[str, object]:
        predecessor_calls.append(("beta", path))
        return {"sha256": aggregate_sha256, "accepted_beta": 2.5}

    def horizon_binding(config: object, path: Path) -> dict[str, object]:
        predecessor_calls.append(("horizon", path))
        return {"sha256": aggregate_sha256, "source_pilot_phase": "freeze"}

    monkeypatch.setattr(materializer, "verify_beta_source_aggregate", beta_binding)
    monkeypatch.setattr(
        materializer,
        "verify_horizon_parent_aggregate",
        horizon_binding,
    )
    if inject_second_write_failure:
        original_publish = materializer._publish_bytes_no_replace
        write_calls = 0

        def fail_second(
            path: Path,
            raw: bytes,
            *,
            staging_directory: Path,
        ) -> bool:
            nonlocal write_calls
            write_calls += 1
            if write_calls == 2:
                raise OSError("injected overlay publication failure")
            return original_publish(
                path,
                raw,
                staging_directory=staging_directory,
            )

        monkeypatch.setattr(materializer, "_publish_bytes_no_replace", fail_second)

    materializer._test_receipt_calls = receipt_calls
    materializer._test_predecessor_calls = predecessor_calls
    argv = [
        str(source_path),
        str(aggregate_path),
        "--repo-root",
        str(repo),
        "--authorization",
        str(authorization_path),
        "--authorization-sha256",
        authorization_sha256,
    ]
    materializer._test_argv = argv
    return materializer, repo, aggregate_path, source_path, aggregate_sha256


def test_projection_is_exactly_five_seed_nonformal_and_freeze_bound() -> None:
    materializer = _materializer()
    freeze, base = _freeze_source()
    freeze_sha256 = "d" * 64
    assert PHASE2_BUDGETED_END_TO_END_EVIDENCE_ROLE == "budgeted_end_to_end_exploratory_only"

    candidate, candidate_base = materializer._budgeted_projection(
        freeze,
        base,
        freeze_sha256=freeze_sha256,
        frozen_beta=2.5,
    )
    validated = validate_phase2_config(candidate, base_config=candidate_base)
    settings = compile_phase2_training_settings(
        {"config": validated, "base_config": candidate_base}
    )

    assert validated["design"] == {
        **freeze["design"],
        "name": "common-beta-post-recovery-budgeted-end-to-end-v1",
        "stage": PHASE2_BUDGETED_END_TO_END_STAGE,
        "pilot_phase": None,
        "formal_eligibility": False,
        "evidence_role": PHASE2_BUDGETED_END_TO_END_EVIDENCE_ROLE,
        "source_config": PHASE2_BUDGETED_END_TO_END_BASE_CONFIG,
        "source_config_hash": config_hash(candidate_base),
    }
    assert validated["run"]["seeds"] == list(PHASE2_BUDGETED_END_TO_END_SEEDS)
    assert validated["run"]["confirmatory"] is False
    assert validated["run"]["formal_eligibility"] is False
    assert validated["run"]["excluded_from_confirmatory_evidence"] is True
    assert candidate_base["run"]["seeds"] == list(PHASE2_BUDGETED_END_TO_END_SEEDS)
    assert settings.stage == PHASE2_BUDGETED_END_TO_END_STAGE
    assert settings.seeds == PHASE2_BUDGETED_END_TO_END_SEEDS
    assert settings.formal_eligibility is False

    identifiability = validated["reward_model"]["identifiability"]
    assert identifiability["role"] == (
        "budgeted_end_to_end_exploratory_frozen_identifiability_audit"
    )
    assert identifiability["require_full_column_rank"] is False
    assert identifiability["confirmatory_freeze_requirement"] == (
        "satisfied_by_accepted_freeze_budgeted_end_to_end_identity"
    )
    common = validated["objective"]["common_beta"]
    maximum = validated["evaluation"]["max_length"]
    assert common["rule"] == "single_accepted_freeze_global_beta_scalar"
    assert common["calibration_source"] == (
        "accepted_freeze_global_beta_in_budgeted_end_to_end_design_identity"
    )
    assert common["frozen_global_beta"] == 2.5
    assert common["beta_source_aggregate_sha256"] == freeze_sha256
    assert maximum["parent_pilot_aggregate_sha256"] == freeze_sha256
    assert maximum["role"] == "budgeted_end_to_end_pre_oracle_safety_gate"
    assert maximum["measure_only"] is False
    assert maximum["formal_gate"] is False
    assert validated["evaluation"]["decision_gates"]["supports_formal_claim"] is False

    protocol = settings.convergence.optimizer_protocol
    assert protocol is not None
    assert protocol.mode == "adopted"
    assert protocol.schedule_sha256 == PHASE2_RECOVERY_LR_SCHEDULE_SHA256
    assert protocol.source_recovery_authorization_sha256 == ("a" * 64)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("pilot_phase",), "calibration"),
        (("schema_version",), "common-beta-pilot-selection-aggregate/v2"),
        (("selection", "selection_accepted"), False),
        (("selection", "accepted_for_confirmatory_identity"), False),
        (("selection", "all_length_gates_passed"), False),
        (("selection", "all_non_length_safety_gates_passed"), False),
        (("selection", "next_action"), "issue_new_pilot_freeze_identity_at_double_beta"),
    ],
)
def test_only_production_v3_fully_accepted_freeze_is_admissible(
    path: tuple[str, ...],
    replacement: object,
) -> None:
    materializer = _materializer()
    source, _ = _freeze_source()
    aggregate = _accepted_aggregate(source)
    target: dict[str, object] = aggregate
    for component in path[:-1]:
        target = target[component]  # type: ignore[assignment]
    target[path[-1]] = replacement

    if path in {("schema_version",), ("pilot_phase",)}:
        with pytest.raises(ValueError, match="production-v3"):
            materializer._source_matches_aggregate(source, aggregate)
    else:
        materializer._source_matches_aggregate(source, aggregate)
        with pytest.raises(ValueError, match="fully accepted production-v3 freeze"):
            materializer._accepted_freeze(aggregate, source)


def test_freeze_source_identity_and_target_free_boundary_are_strict() -> None:
    materializer = _materializer()
    source, _ = _freeze_source()
    aggregate = _accepted_aggregate(source)
    aggregate["phase2_design_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="supplied source identity"):
        materializer._source_matches_aggregate(source, aggregate)

    aggregate = _accepted_aggregate(source)
    aggregate["information_boundary"]["test_metrics_read"] = True  # type: ignore[index]
    with pytest.raises(ValueError, match="information boundary"):
        materializer._source_matches_aggregate(source, aggregate)


def test_freeze_semantic_paths_are_independent_from_formal_materializer() -> None:
    materializer = _materializer()
    assert materializer._expected_freeze_overlay_relative(0) == Path(
        "configs/common_beta_post_recovery_freeze.yaml"
    )
    assert materializer._expected_freeze_overlay_relative(3) == Path(
        "configs/common_beta_post_recovery_freeze_retry_3.yaml"
    )
    with pytest.raises(ValueError, match="non-negative"):
        materializer._expected_freeze_overlay_relative(-1)

    source = MATERIALIZER_PATH.read_text(encoding="utf-8")
    for forbidden in (
        "materialize_phase2_post_recovery_stage",
        "submit_phase2_confirmatory",
        "resolve_phase2_campaign_registry",
        "phase2_campaign_finalize",
        "PHASE2_CONFIRMATORY_SEEDS",
    ):
        assert forbidden not in source


def test_cli_materializes_exclusive_pair_and_binds_one_freeze_everywhere(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    materializer, repo, aggregate_path, _, aggregate_sha256 = _main_fixture(
        tmp_path,
        monkeypatch,
    )

    assert materializer.main(materializer._test_argv) == 0
    report = json.loads(capsys.readouterr().out)
    base_output = repo / PHASE2_BUDGETED_END_TO_END_BASE_CONFIG
    overlay_output = repo / PHASE2_BUDGETED_END_TO_END_CONFIG
    assert base_output.is_file()
    assert overlay_output.is_file()
    assert materializer._test_receipt_calls == [aggregate_path]
    assert materializer._test_predecessor_calls == [
        ("beta", aggregate_path),
        ("horizon", aggregate_path),
    ]
    assert report["stage"] == PHASE2_BUDGETED_END_TO_END_STAGE
    assert report["formal"] is False
    assert report["formal_eligibility"] is False
    assert report["supports_formal_claim"] is False
    assert report["evidence_role"] == PHASE2_BUDGETED_END_TO_END_EVIDENCE_ROLE
    assert report["ordered_seeds"] == list(PHASE2_BUDGETED_END_TO_END_SEEDS)
    assert report["seed_count"] == 5
    assert report["accepted_freeze_aggregate_sha256"] == aggregate_sha256
    assert report["beta_source_aggregate_sha256"] == aggregate_sha256
    assert report["horizon_parent_aggregate_sha256"] == aggregate_sha256
    assert report["optimizer_schedule_sha256"] == PHASE2_RECOVERY_LR_SCHEDULE_SHA256
    assert report["materialization_mode"] == "recoverable_atomic_link_no_overwrite"
    assert report["resumed_publications"] == []
    receipt_output = repo / materializer._BUDGETED_RECEIPT_RELATIVE
    assert receipt_output.is_file()
    assert report["materialization_receipt"] == str(receipt_output)
    assert report["materialization_receipt_sha256"] == materializer._sha256(receipt_output)

    bundle = materializer.load_phase2_config_bundle(overlay_output)
    assert bundle.base_config_path == base_output
    assert (
        bundle.config["objective"]["common_beta"]["beta_source_aggregate_sha256"]
        == aggregate_sha256
    )
    assert (
        bundle.config["evaluation"]["max_length"]["parent_pilot_aggregate_sha256"]
        == aggregate_sha256
    )
    base_before = base_output.read_bytes()
    overlay_before = overlay_output.read_bytes()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        materializer.main(materializer._test_argv)
    assert base_output.read_bytes() == base_before
    assert overlay_output.read_bytes() == overlay_before


def test_cli_resumes_after_process_dies_between_base_and_overlay_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    materializer, repo, _, _, _ = _main_fixture(
        tmp_path,
        monkeypatch,
        inject_second_write_failure=True,
    )

    with pytest.raises(OSError, match="injected overlay publication failure"):
        materializer.main(materializer._test_argv)

    base_output = repo / PHASE2_BUDGETED_END_TO_END_BASE_CONFIG
    overlay_output = repo / PHASE2_BUDGETED_END_TO_END_CONFIG
    receipt_output = repo / materializer._BUDGETED_RECEIPT_RELATIVE
    assert base_output.is_file()
    assert not overlay_output.exists()
    assert not receipt_output.exists()

    assert materializer.main(materializer._test_argv) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["resumed_publications"] == [PHASE2_BUDGETED_END_TO_END_BASE_CONFIG]
    assert base_output.is_file()
    assert overlay_output.is_file()
    assert receipt_output.is_file()


def test_cli_rejects_dirty_or_uncommitted_source_before_consuming_aggregate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer, repo, _, _, _ = _main_fixture(tmp_path, monkeypatch)
    (repo / "untracked.txt").write_text("dirty\n", encoding="utf-8")

    with pytest.raises(ValueError, match="clean committed worktree"):
        materializer.main(materializer._test_argv)

    assert materializer._test_receipt_calls == []
    assert not (repo / PHASE2_BUDGETED_END_TO_END_BASE_CONFIG).exists()
    assert not (repo / PHASE2_BUDGETED_END_TO_END_CONFIG).exists()


def test_authorization_projection_and_optimizer_schedule_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _materializer()
    freeze, base = _freeze_source()
    candidate, candidate_base = materializer._budgeted_projection(
        freeze,
        base,
        freeze_sha256="d" * 64,
        frozen_beta=2.5,
    )
    normalized = validate_phase2_config(candidate, base_config=candidate_base)
    authorization = _authorization_payload(freeze)
    authorization["optimizer_schedule_sha256"] = "0" * 64
    monkeypatch.setattr(
        materializer,
        "verify_recovery_authorization_file",
        lambda path, *, expected_sha256: authorization,
    )

    with pytest.raises(
        ValueError,
        match="optimizer_schedule_sha256|authorization payload projection",
    ):
        materializer._verify_authorization_and_optimizer(
            normalized,
            candidate_base,
            authorization_path=tmp_path / "authorization.json",
            authorization_sha256="a" * 64,
        )
