from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from test_phase2_sensitivity import (
    _canonical_sha256,
    _confirmatory,
    _environment,
    _primary_aggregate,
    _primary_sha,
    _write,
)

from smart_reward.contracts import BT_MLE, PRORM_PLUS
from smart_reward.phase2_config import PHASE2_CONFIRMATORY_SEEDS
from smart_reward.phase2_mechanism import (
    PHASE2_MECHANISM_AGGREGATE_SCHEMA,
    PHASE2_MECHANISM_SEED_SCHEMA,
    _ridge_free_regret,
    build_phase2_mechanism_aggregate,
)


def _mechanism_seed(
    path: Path,
    config: dict[str, object],
    seed: int,
    *,
    exact_prorm: float = 0.4,
    low_prorm: float = 0.5,
) -> None:
    from smart_reward.phase2_config import phase2_design_identity
    from smart_reward.phase2_rollout import Phase2Design

    environment = _environment()
    runtime = Phase2Design.from_phase2_config(config)
    primary_path = path.parent / "primary" / f"{seed}.json"
    primary_path.parent.mkdir(parents=True, exist_ok=True)
    primary_path.write_bytes(f"primary:{seed}".encode())
    assert hashlib.sha256(primary_path.read_bytes()).hexdigest() == _primary_sha(seed)
    payload: dict[str, object] = {
        "schema_version": PHASE2_MECHANISM_SEED_SCHEMA,
        "seed": seed,
        "source_config_hash": config["design"]["source_config_hash"],
        "phase2_design_sha256": phase2_design_identity(config),
        "phase2_runtime_contract_sha256": runtime.sha256,
        "environment_identity": environment,
        "environment_identity_sha256": _canonical_sha256(environment),
        "primary_result": {
            "path": f"primary/{seed}.json",
            "sha256": _primary_sha(seed),
            "heads_sha256": "a" * 64,
            "label_stream_sha256": "b" * 64,
            "beta0": 2.5,
            "read_only": True,
            "modified": False,
        },
        "heldout_test": {
            "deferred_input_sha256": "c" * 64,
            "transformed_target_sha256": "d" * 64,
            "num_prompts": 256,
            "num_candidates": 4,
            "raw_oracle_logits_serialized": False,
            "target_vector_serialized": False,
        },
        "exact_noise_free": {
            "schema_version": "phase2-exact-noise-free-mechanism/v1",
            "comparison": "exact_margin_prorm_plus_vs_exact_soft_label_bt",
            "geometry": "full_tangent_primary_ridge",
            "beta": 2.5,
            "relative_ridge": 0.001,
            "absolute_ridge": 0.0003,
            "learners": {
                BT_MLE: {
                    "head_source": "exact_soft_label_bt_control",
                    "head_sha256": "e" * 64,
                    "heldout_local_regret": 1.0 + 0.01 * (seed % 5),
                },
                PRORM_PLUS: {
                    "head_source": "exact_margin_control",
                    "head_sha256": "f" * 64,
                    "heldout_local_regret": exact_prorm + 0.005 * (seed % 5),
                },
            },
            "label_noise_present": False,
            "eligible_for_primary_claim": False,
        },
        "low_dimensional_ridge_free": {
            "schema_version": "phase2-low-dimensional-mechanism/v1",
            "comparison": "r4_prorm_plus_vs_r4_bt_mle",
            "geometry": "seeded_projected_tangent_full_rank_ridge_free",
            "beta": 2.5,
            "projection": {
                "sha256": "1" * 64,
                "namespace": "test",
                "effective_seed": seed,
                "selected_dimension": 2,
                "projection_matrix_serialized": False,
            },
            "fisher": {
                "schema_version": "phase2-ridge-free-heldout-geometry/v1",
                "dimension": 2,
                "numerical_rank": 2,
                "full_rank": True,
                "ridge_enabled": False,
                "ridge_coefficient": 0.0,
            },
            "learners": {
                BT_MLE: {
                    "head_source": "primary_r4_bt_mle",
                    "head_sha256": "2" * 64,
                    "heldout_local_regret": 1.1 + 0.01 * (seed % 5),
                },
                PRORM_PLUS: {
                    "head_source": "low_dimensional_r4_prorm_plus_control",
                    "head_sha256": "3" * 64,
                    "heldout_local_regret": low_prorm + 0.005 * (seed % 5),
                },
            },
            "same_r4_label_stream": True,
            "ridge_enabled": False,
            "full_rank_gate_passed": True,
            "eligible_for_primary_claim": False,
        },
        "claim_contract": {
            "role": "mechanism_scope_qualifier_only",
            "may_support_misspecification_geometry_interpretation": True,
            "eligible_to_modify_primary_efficacy_status": False,
            "primary_efficacy_status_read": False,
            "primary_efficacy_status_modified": False,
        },
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    _write(path, payload)


def _campaign(
    tmp_path: Path,
    config: dict[str, object],
    *,
    exact_prorm: float = 0.4,
    low_prorm: float = 0.5,
) -> list[Path]:
    paths: list[Path] = []
    for seed in PHASE2_CONFIRMATORY_SEEDS:
        path = tmp_path / f"mechanism-{seed}.json"
        _mechanism_seed(
            path,
            config,
            seed,
            exact_prorm=exact_prorm,
            low_prorm=low_prorm,
        )
        paths.append(path)
    return paths


def test_mechanism_aggregate_qualifies_scope_but_preserves_primary_status(
    tmp_path: Path,
) -> None:
    config = _confirmatory()
    primary = _primary_aggregate(tmp_path, config, status="not_passed")
    paths = _campaign(tmp_path, config)
    payload = build_phase2_mechanism_aggregate(
        config,
        paths,
        primary_aggregate_json=primary,
        reference_base=tmp_path,
    )

    assert payload["schema_version"] == PHASE2_MECHANISM_AGGREGATE_SCHEMA
    assert payload["seeds"] == list(PHASE2_CONFIRMATORY_SEEDS)
    assert payload["claim_scope"]["status"] == "mechanism_qualified"
    assert payload["claim_scope"]["both_pre_registered_qualifiers_passed"] is True
    assert payload["primary_aggregate"]["efficacy_status"] == "not_passed"
    assert payload["primary_aggregate"]["modified"] is False
    assert payload["claim_scope"]["eligible_to_modify_primary_efficacy_status"] is False


def test_failed_mechanism_qualifier_downgrades_only_claim_scope(
    tmp_path: Path,
) -> None:
    config = _confirmatory()
    primary = _primary_aggregate(tmp_path, config, status="passed")
    paths = _campaign(tmp_path, config, low_prorm=2.0)
    payload = build_phase2_mechanism_aggregate(
        config,
        paths,
        primary_aggregate_json=primary,
        reference_base=tmp_path,
    )

    assert payload["claim_scope"]["status"] == "finite_procedure_only_no_geometry_attribution"
    assert payload["claim_scope"]["both_pre_registered_qualifiers_passed"] is False
    assert payload["primary_aggregate"]["efficacy_status"] == "passed"
    assert payload["primary_aggregate"]["modified"] is False


def test_mechanism_aggregate_rejects_missing_or_selected_seed_subset(
    tmp_path: Path,
) -> None:
    config = _confirmatory()
    primary = _primary_aggregate(tmp_path, config)
    paths = _campaign(tmp_path, config)
    with pytest.raises(ValueError, match="exactly 30"):
        build_phase2_mechanism_aggregate(
            config,
            paths[1:],
            primary_aggregate_json=primary,
            reference_base=tmp_path,
        )


def test_mechanism_aggregate_rejects_non_full_rank_low_dimensional_claim(
    tmp_path: Path,
) -> None:
    config = _confirmatory()
    primary = _primary_aggregate(tmp_path, config)
    paths = _campaign(tmp_path, config)
    value = json.loads(paths[0].read_text(encoding="utf-8"))
    value["low_dimensional_ridge_free"]["full_rank_gate_passed"] = False
    unsigned = {key: item for key, item in value.items() if key != "artifact_sha256"}
    value["artifact_sha256"] = _canonical_sha256(unsigned)
    _write(paths[0], value)
    with pytest.raises(ValueError, match="not full-rank/ridge-free"):
        build_phase2_mechanism_aggregate(
            config,
            paths,
            primary_aggregate_json=primary,
            reference_base=tmp_path,
        )


def test_mechanism_artifact_must_bind_its_exact_primary_result(
    tmp_path: Path,
) -> None:
    config = _confirmatory()
    primary = _primary_aggregate(tmp_path, config)
    paths = _campaign(tmp_path, config)
    value = json.loads(paths[0].read_text(encoding="utf-8"))
    value["primary_result"]["sha256"] = "9" * 64
    unsigned = {key: item for key, item in value.items() if key != "artifact_sha256"}
    value["artifact_sha256"] = _canonical_sha256(unsigned)
    _write(paths[0], value)
    with pytest.raises(ValueError, match="byte hash does not match"):
        build_phase2_mechanism_aggregate(
            config,
            paths,
            primary_aggregate_json=primary,
            reference_base=tmp_path,
        )


def test_mechanism_artifact_rejects_changed_control_beta(tmp_path: Path) -> None:
    config = _confirmatory()
    primary = _primary_aggregate(tmp_path, config)
    paths = _campaign(tmp_path, config)
    value = json.loads(paths[0].read_text(encoding="utf-8"))
    value["exact_noise_free"]["beta"] = 9.0
    unsigned = {key: item for key, item in value.items() if key != "artifact_sha256"}
    value["artifact_sha256"] = _canonical_sha256(unsigned)
    _write(paths[0], value)
    with pytest.raises(ValueError, match="control identity is invalid"):
        build_phase2_mechanism_aggregate(
            config,
            paths,
            primary_aggregate_json=primary,
            reference_base=tmp_path,
        )


def test_ridge_free_mechanism_metric_uses_exact_full_rank_inverse() -> None:
    policy_scores = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
            [[0.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    reward_features = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.5]],
            [[0.5, -1.0], [1.0, 1.0], [-0.5, 1.0]],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor(
        [[0.2, -0.3, 0.1], [-0.2, 0.4, 0.3]],
        dtype=torch.float32,
    )
    value, geometry = _ridge_free_regret(
        policy_scores,
        reward_features,
        targets,
        (0.2, -0.1),
        beta=2.5,
        relative_eigenvalue_tolerance=1.0e-10,
    )
    assert value >= 0.0
    assert geometry["full_rank"] is True
    assert geometry["numerical_rank"] == 2
    assert geometry["ridge_enabled"] is False
    assert geometry["ridge_coefficient"] == 0.0
    assert geometry["solve_relative_residual"] < 1.0e-12


def test_ridge_free_mechanism_metric_fails_closed_on_rank_deficiency() -> None:
    policy_scores = torch.ones((2, 3, 2), dtype=torch.float32)
    reward_features = torch.randn((2, 3, 2), generator=torch.Generator().manual_seed(4))
    targets = torch.zeros((2, 3), dtype=torch.float32)
    with pytest.raises(ValueError, match="full-rank held-out Fisher"):
        _ridge_free_regret(
            policy_scores,
            reward_features,
            targets,
            (0.1, 0.2),
            beta=2.5,
            relative_eigenvalue_tolerance=1.0e-10,
        )


def test_mechanism_reader_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    config = _confirmatory()
    primary = _primary_aggregate(tmp_path, config)
    paths = _campaign(tmp_path, config)
    paths[0].write_text(
        '{"schema_version":"phase2-mechanism-qualifiers-seed/v1",'
        '"schema_version":"phase2-mechanism-qualifiers-seed/v1"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid mechanism JSON"):
        build_phase2_mechanism_aggregate(
            config,
            paths,
            primary_aggregate_json=primary,
            reference_base=tmp_path,
        )


@pytest.mark.parametrize("invalid_seed", [True, 0.0])
def test_mechanism_reader_rejects_non_integer_seed(
    tmp_path: Path,
    invalid_seed: object,
) -> None:
    config = _confirmatory()
    primary = _primary_aggregate(tmp_path, config)
    paths = _campaign(tmp_path, config)
    value = json.loads(paths[0].read_text(encoding="utf-8"))
    value["seed"] = invalid_seed
    unsigned = {key: item for key, item in value.items() if key != "artifact_sha256"}
    value["artifact_sha256"] = _canonical_sha256(unsigned)
    _write(paths[0], value)
    with pytest.raises(TypeError, match="must be an integer"):
        build_phase2_mechanism_aggregate(
            config,
            paths,
            primary_aggregate_json=primary,
            reference_base=tmp_path,
        )
