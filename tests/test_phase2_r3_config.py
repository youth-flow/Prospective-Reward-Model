from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml

from smart_reward.phase2_r3_config import (
    R3_PRIMARY_HEADS,
    R3_PRIMARY_SEEDS,
    R3_RECOVERY_SCHEDULE_SHA256,
    R3_SCIENCE_CONFIG_SCHEMA,
    R3ScienceConfigBundle,
    R3ScienceConfigError,
    load_r3_science_config,
)
from smart_reward.phase2_training import Phase2TrainingSettings

ROOT = Path(__file__).parents[1]
SCIENCE_PATH = ROOT / "configs" / "phase2_recovery_r3_science.yaml"
LEGACY_R2_PATH = ROOT / "configs" / "common_beta_recovery_pilot.yaml"
R2_DESIGN_SHA256 = "9602b0f00a73880545fd57ce1886ec65d7901385cce2b919fd72f3efec4592d4"


def _canonical_sha256(value: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _load_yaml() -> dict[str, Any]:
    value = yaml.safe_load(SCIENCE_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write_yaml(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        yaml.safe_dump(value, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _leaf_paths(value: object, prefix: tuple[object, ...] = ()) -> list[tuple[object, ...]]:
    if isinstance(value, dict):
        result: list[tuple[object, ...]] = []
        for key, child in value.items():
            result.extend(_leaf_paths(child, (*prefix, key)))
        return result
    if isinstance(value, list):
        result = []
        for index, child in enumerate(value):
            result.extend(_leaf_paths(child, (*prefix, index)))
        return result
    return [prefix]


def _replace_leaf(root: object, path: tuple[object, ...]) -> None:
    parent = root
    for component in path[:-1]:
        parent = parent[component]  # type: ignore[index]
    key = path[-1]
    original = parent[key]  # type: ignore[index]
    if original is None:
        replacement: object = "tampered"
    elif type(original) is bool:
        replacement = not original
    elif type(original) is int:
        replacement = original + 1
    elif type(original) is float:
        replacement = original + 0.125
    elif type(original) is str:
        replacement = f"{original}-tampered"
    else:
        raise AssertionError(f"unhandled leaf type {type(original).__name__}")
    parent[key] = replacement  # type: ignore[index]


def test_loads_standalone_r3_science_and_compiles_exact_settings() -> None:
    bundle = load_r3_science_config(SCIENCE_PATH)

    assert isinstance(bundle, R3ScienceConfigBundle)
    assert isinstance(bundle.settings, Phase2TrainingSettings)
    assert bundle.source_path == SCIENCE_PATH.absolute()
    assert bundle.file_sha256 == hashlib.sha256(SCIENCE_PATH.read_bytes()).hexdigest()
    assert bundle.semantic_sha256 == _canonical_sha256(bundle.normalized)
    assert bundle.semantic_sha256 != R2_DESIGN_SHA256
    assert R2_DESIGN_SHA256 not in json.dumps(bundle.normalized, sort_keys=True)
    assert bundle.normalized["schema_version"] == R3_SCIENCE_CONFIG_SCHEMA
    assert bundle.validate_integrity() is None

    campaign = bundle.normalized["campaign"]
    assert campaign == {
        "campaign_kind": "phase2_recovery_revision3_primary_only",
        "confirmatory": False,
        "evidence_role": "train_only_nonconfirmatory_recovery",
        "execution_revision": 3,
        "execution_scope": "primary_only",
        "formal_eligibility": False,
        "inherited_r2_design": False,
        "name": "phase2-recovery-r3-primary-only",
        "primary_heads": list(R3_PRIMARY_HEADS),
    }
    assert bundle.normalized["run"] == {"seeds": list(R3_PRIMARY_SEEDS)}
    assert bundle.normalized["execution_boundary"] == {
        "controls_executed_in_primary_run": False,
        "downstream_utility_allowed": False,
        "execution_scope": "train_only",
        "final_oracle_allowed": False,
        "policy_rollout_allowed": False,
        "profile_evidence_reusable": False,
        "validation_or_test_access_allowed": False,
    }

    settings = bundle.settings
    assert settings.phase2_config_hash == bundle.semantic_sha256
    assert settings.stage == "pilot"
    assert settings.formal_eligibility is False
    assert settings.seeds == R3_PRIMARY_SEEDS
    assert settings.outer_steps == 720
    assert settings.num_label_replicates == 4
    assert settings.annotation_gamma == 0.9
    assert settings.label_rng_namespace == "prorm-common-beta-r4-labels-v1"
    assert settings.optimizer == "adamw"
    assert settings.learning_rate == 1.0e-3
    assert settings.weight_decay == 0.0
    assert settings.training_beta == 1.0
    assert settings.pcg_dtype == "float64"
    assert settings.pcg_tolerance == 1.0e-5
    assert settings.pcg_max_iterations == 8192
    assert settings.convergence.gradient_ratio_tolerance == 1.0e-3
    assert settings.convergence.min_steps == 100
    assert settings.convergence.max_steps == 12760
    assert settings.convergence.check_interval == 20
    assert settings.convergence.consecutive_checks == 3
    protocol = settings.convergence.optimizer_protocol
    assert protocol is not None
    assert protocol.mode == "recovery"
    assert protocol.schedule_sha256 == R3_RECOVERY_SCHEDULE_SHA256
    assert protocol.maximum_update == 12760
    assert protocol.legacy_boundary_snapshot_steps == 5760
    assert protocol.reward_head_dtype == "float32"
    assert protocol.first_order_audit_dtype == "float64"
    assert protocol.learning_rate_for_update(1) == 1.0e-3
    assert protocol.learning_rate_for_update(5760) == 1.0e-3
    assert protocol.learning_rate_for_update(5761) == 3.0e-4
    assert protocol.learning_rate_for_update(6761) == 1.0e-4
    assert protocol.learning_rate_for_update(8761) == 3.0e-5
    assert protocol.learning_rate_for_update(10761) == 1.0e-5
    assert protocol.learning_rate_for_update(12760) == 1.0e-5
    assert bundle.normalized["diagnostics"] == {
        "may_change_science_config": False,
        "may_select_primary_iterate": False,
        "snapshots": [
            {"role": "compute_matched_diagnostic_only", "update": 720},
            {"role": "legacy_boundary_diagnostic_only", "update": 5760},
        ],
    }


def test_loader_rejects_legacy_r2_overlay_and_non_path_inputs() -> None:
    with pytest.raises(R3ScienceConfigError):
        load_r3_science_config(LEGACY_R2_PATH)

    bundle = load_r3_science_config(SCIENCE_PATH)
    for invalid in (bundle, bundle.settings, bundle.normalized, None, 17, True):
        with pytest.raises(TypeError):
            load_r3_science_config(invalid)  # type: ignore[arg-type]


def test_every_declared_leaf_is_locked_against_tamper(tmp_path: Path) -> None:
    original = _load_yaml()
    leaf_paths = _leaf_paths(original)
    assert len(leaf_paths) > 100

    for index, leaf_path in enumerate(leaf_paths):
        tampered = copy.deepcopy(original)
        _replace_leaf(tampered, leaf_path)
        candidate = tmp_path / f"tampered-{index}.yaml"
        _write_yaml(candidate, tampered)
        with pytest.raises(R3ScienceConfigError):
            load_r3_science_config(candidate)


def test_closed_schema_rejects_missing_unknown_duplicate_and_fake_hash(
    tmp_path: Path,
) -> None:
    original = _load_yaml()
    mutations: list[dict[str, Any]] = []

    missing = copy.deepcopy(original)
    del missing["diagnostics"]
    mutations.append(missing)

    unknown = copy.deepcopy(original)
    unknown["semantic_sha256"] = "f" * 64
    mutations.append(unknown)

    nested_unknown = copy.deepcopy(original)
    nested_unknown["campaign"]["legacy_design_hash"] = R2_DESIGN_SHA256
    mutations.append(nested_unknown)

    for index, mutation in enumerate(mutations):
        candidate = tmp_path / f"closed-{index}.yaml"
        _write_yaml(candidate, mutation)
        with pytest.raises(R3ScienceConfigError):
            load_r3_science_config(candidate)

    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_bytes(SCIENCE_PATH.read_bytes() + b"\nschema_version: duplicate\n")
    with pytest.raises(R3ScienceConfigError, match="duplicate key"):
        load_r3_science_config(duplicate)


def test_bundle_reloads_source_and_rejects_hash_normalized_or_settings_replacement(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "science.yaml"
    candidate.write_bytes(SCIENCE_PATH.read_bytes())
    bundle = load_r3_science_config(candidate)

    with pytest.raises(R3ScienceConfigError):
        replace(bundle, file_sha256="f" * 64)
    with pytest.raises(R3ScienceConfigError):
        replace(bundle, semantic_sha256="f" * 64)

    replacement_settings = replace(bundle.settings, phase2_config_hash="f" * 64)
    with pytest.raises(R3ScienceConfigError):
        replace(bundle, settings=replacement_settings)

    replacement_normalized = copy.deepcopy(bundle.normalized)
    replacement_normalized["campaign"]["execution_revision"] = 4
    with pytest.raises(R3ScienceConfigError):
        replace(bundle, normalized=replacement_normalized)

    candidate.write_bytes(SCIENCE_PATH.read_bytes() + b"\n# byte-level change\n")
    with pytest.raises(R3ScienceConfigError, match="source file bytes changed"):
        bundle.validate_integrity()


def test_source_must_be_a_non_symlink_regular_file(tmp_path: Path) -> None:
    directory = tmp_path / "directory"
    directory.mkdir()
    with pytest.raises(R3ScienceConfigError, match="non-symlink regular file"):
        load_r3_science_config(directory)

    link = tmp_path / "science-link.yaml"
    try:
        link.symlink_to(SCIENCE_PATH)
    except OSError:
        pytest.skip("symbolic links are unavailable on this platform")
    with pytest.raises(R3ScienceConfigError, match="symbolic link"):
        load_r3_science_config(link)
