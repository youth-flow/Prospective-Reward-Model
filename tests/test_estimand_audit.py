from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import smart_reward.estimand_audit as audit_module
from smart_reward.cli import main
from smart_reward.config import config_hash, load_config
from smart_reward.contracts import BT_MLE, PRORM_PLUS
from smart_reward.estimand_audit import (
    audit_phase1_estimands,
    evaluate_saved_policy_directions,
)
from smart_reward.experiment import EvaluationTensorData

ROOT = Path(__file__).resolve().parents[1]


def _test_split() -> EvaluationTensorData:
    scores = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[2.0, 1.0], [1.0, -1.0]],
        ],
        dtype=torch.float64,
    )
    return EvaluationTensorData(
        prompt_ids=("test-0", "test-1"),
        policy_scores=scores,
        reward_features=torch.zeros(2, 2, 1, dtype=torch.float64),
        true_rewards=torch.tensor([[2.0, 0.0], [1.0, -1.0]], dtype=torch.float64),
    )


def _learner_evidence(
    direction: list[float],
    *,
    beta: float,
    relative_damping: float,
    train_curvature: float,
    alpha: float,
    target_kl: float,
) -> dict[str, object]:
    direction_tensor = torch.tensor(direction, dtype=torch.float64)
    return {
        "direction": {
            "schema_version": "policy-direction/v1",
            "direction": direction,
            "beta": beta,
            "relative_damping": relative_damping,
            "absolute_damping": relative_damping,
            "mean_fisher_diagonal": 1.0,
            "moment_norm": 1.0,
            "direction_norm": float(torch.linalg.vector_norm(direction_tensor).item()),
            "fisher_curvature": train_curvature,
            "damped_curvature": train_curvature + relative_damping,
            "moment_alignment": 1.0,
            "pcg": {
                "iterations": 2,
                "residual_norm": 1.0e-10,
                "relative_residual": 1.0e-10,
                "converged": True,
                "reason": "converged",
            },
        },
        "measured_kl_update": {
            "schema_version": "measured-kl-update/v1",
            "target_kl": target_kl,
            "initialization": "train_fisher_quadratic",
            "initial_step_size": alpha,
            "fisher_curvature": train_curvature,
            "best_step_size": alpha,
            "best_measured_kl": target_kl,
            "applied_step_size": alpha,
            "applied_measured_kl": target_kl,
            "line_search_evaluations": 3,
            "converged": True,
            "applied": True,
            "reference_forward_evaluations": 1,
            "tangent_dimension": len(direction),
            "a_state_sha256": "a" * 64,
        },
    }


def _evidence(
    *,
    beta: float = 2.0,
    relative_damping: float = 0.1,
    target_kl: float = 0.1,
) -> dict[str, object]:
    return {
        BT_MLE: _learner_evidence(
            [1.0, 0.0],
            beta=beta,
            relative_damping=relative_damping,
            train_curvature=4.0,
            alpha=0.5,
            target_kl=target_kl,
        ),
        PRORM_PLUS: _learner_evidence(
            [0.0, 2.0],
            beta=beta,
            relative_damping=relative_damping,
            train_curvature=9.0,
            alpha=0.25,
            target_kl=target_kl,
        ),
    }


def test_actual_train_directions_are_evaluated_on_test_geometry() -> None:
    result = evaluate_saved_policy_directions(
        _test_split(),
        _evidence(),
        beta=2.0,
        relative_damping=0.1,
        fixed_k=0.1,
        prompt_chunk_size=1,
    )

    assert result["geometry"]["target_moment_l2_norm"] == pytest.approx(math.sqrt(1.25))
    bt = result["learners"][BT_MLE]
    prorm = result["learners"][PRORM_PLUS]
    bt_test = bt["test_estimands"]
    prorm_test = prorm["test_estimands"]

    # For this fixture g*=(1, .5).  The raw-node test Fisher curvatures are
    # d_bt^T F d_bt=1.5 and d_prorm^T F d_prorm=3.
    assert bt_test["target_linear_term"] == pytest.approx(1.0)
    assert bt_test["native_fisher_norm"] == pytest.approx(math.sqrt(1.5))
    assert bt_test["native_quadratic_kl"] == pytest.approx(0.75)
    assert bt_test["native_fixed_beta_utility"] == pytest.approx(-0.5)
    assert bt_test["applied_quadratic_kl"] == pytest.approx(0.1875)
    assert bt_test["applied_fixed_beta_utility"] == pytest.approx(0.125)
    assert bt_test["fixed_k_normalization_step"] == pytest.approx(math.sqrt(0.2 / 1.5))
    assert bt_test["fixed_k_linear_gain"] == pytest.approx(math.sqrt(0.2 / 1.5))
    assert bt_test["fixed_k_fixed_beta_utility"] == pytest.approx(math.sqrt(0.2 / 1.5) - 0.2)
    assert bt["saved_train_update"]["beta_eff"] == pytest.approx(4.0)
    assert bt["saved_train_update"]["train_native_quadratic_kl"] == pytest.approx(2.0)

    assert prorm_test["target_linear_term"] == pytest.approx(1.0)
    assert prorm_test["native_quadratic_kl"] == pytest.approx(1.5)
    assert prorm_test["native_fixed_beta_utility"] == pytest.approx(-2.0)
    assert prorm_test["applied_quadratic_kl"] == pytest.approx(0.09375)
    assert prorm_test["applied_fixed_beta_utility"] == pytest.approx(0.0625)
    assert prorm["saved_train_update"]["beta_eff"] == pytest.approx(8.0)

    differences = result["paired_differences_prorm_plus_minus_bt"]
    assert differences["native_fixed_beta_utility"] == pytest.approx(-1.5)
    assert differences["applied_fixed_beta_utility"] == pytest.approx(-0.0625)
    assert differences["fixed_k_linear_gain"] == pytest.approx(
        math.sqrt(0.2 / 3.0) - math.sqrt(0.2 / 1.5)
    )


def test_chunking_does_not_change_the_estimand() -> None:
    unchunked = evaluate_saved_policy_directions(
        _test_split(),
        _evidence(),
        beta=2.0,
        relative_damping=0.1,
        fixed_k=0.1,
        prompt_chunk_size=16,
    )
    chunked = evaluate_saved_policy_directions(
        _test_split(),
        _evidence(),
        beta=2.0,
        relative_damping=0.1,
        fixed_k=0.1,
        prompt_chunk_size=1,
    )

    for learner in (BT_MLE, PRORM_PLUS):
        assert chunked["learners"][learner]["test_estimands"] == pytest.approx(
            unchunked["learners"][learner]["test_estimands"]
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value[BT_MLE]["direction"].__setitem__(
                "direction",
                [1.0],
            ),
            "expected 2",
        ),
        (
            lambda value: value[BT_MLE]["direction"].__setitem__("beta", 3.0),
            "objective.beta",
        ),
        (
            lambda value: value[BT_MLE]["measured_kl_update"].__setitem__(
                "applied",
                False,
            ),
            "converged, applied",
        ),
    ],
)
def test_invalid_saved_update_evidence_is_rejected(mutation, message: str) -> None:
    evidence = copy.deepcopy(_evidence())
    mutation(evidence)

    with pytest.raises((TypeError, ValueError), match=message):
        evaluate_saved_policy_directions(
            _test_split(),
            evidence,
            beta=2.0,
            relative_damping=0.1,
            fixed_k=0.1,
        )


def test_bound_audit_writes_new_file_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(ROOT / "configs" / "smoke.yaml")
    digest = config_hash(config)
    seed = 20260722
    artifact_digest = "d" * 64
    run_manifest_digest = "e" * 64
    environment = {"formal": True}
    comparison_path = tmp_path / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "run_manifest_sha256": run_manifest_digest,
                "environment_identity": environment,
            }
        ),
        encoding="utf-8",
    )
    comparison_digest = hashlib.sha256(comparison_path.read_bytes()).hexdigest()
    beta = float(config["objective"]["beta"])
    relative_damping = float(config["objective"]["damping_relative_to_mean_fisher_diagonal"])
    target_kl = float(config["evaluation"]["kl_budget"])
    rollout_path = tmp_path / "rollout.json"
    rollout_path.write_text(
        json.dumps(
            {
                "schema_version": "matched-kl-rollout/v2",
                "config_hash": digest,
                "seed": seed,
                "artifact_metadata_sha256": artifact_digest,
                "comparison_sha256": comparison_digest,
                "run_manifest_sha256": run_manifest_digest,
                "environment_identity": environment,
                "learners": _evidence(
                    beta=beta,
                    relative_damping=relative_damping,
                    target_kl=target_kl,
                ),
                "train_oracle_values_accessed": False,
                "raw_oracle_values_serialized": False,
            }
        ),
        encoding="utf-8",
    )
    experiment = SimpleNamespace(
        train=SimpleNamespace(reward_dimension=1),
        test=_test_split(),
    )
    monkeypatch.setattr(
        audit_module,
        "artifact_metadata_sha256",
        lambda *args, **kwargs: artifact_digest,
    )
    monkeypatch.setattr(
        audit_module,
        "load_controlled_feature_artifact",
        lambda *args, **kwargs: experiment,
    )
    monkeypatch.setattr(
        audit_module,
        "parse_comparison_heads",
        lambda *args, **kwargs: {BT_MLE: (0.0,), PRORM_PLUS: (0.0,)},
    )

    output = tmp_path / "audit" / "estimands.json"
    payload = audit_phase1_estimands(
        config,
        seed=seed,
        artifact_dir=tmp_path / "artifact",
        comparison_json=comparison_path,
        rollout_json=rollout_path,
        output_json=output,
    )
    assert payload["schema_version"] == "estimand-audit/v1"
    assert payload["computation"]["policy_or_oracle_model_loaded"] is False
    assert payload["computation"]["uses_serialized_actual_train_directions"] is True
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        audit_phase1_estimands(
            config,
            seed=seed,
            artifact_dir=tmp_path / "artifact",
            comparison_json=comparison_path,
            rollout_json=rollout_path,
            output_json=output,
        )


def test_cli_wires_cpu_estimand_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[dict[str, object]] = []

    def fake_audit(config, **kwargs):
        calls.append(kwargs)
        return {"config_hash": config_hash(config)}

    monkeypatch.setattr(audit_module, "audit_phase1_estimands", fake_audit)
    output = tmp_path / "audit.json"
    assert (
        main(
            [
                "estimand-audit",
                str(ROOT / "configs" / "smoke.yaml"),
                str(tmp_path / "artifact"),
                str(tmp_path / "comparison.json"),
                str(tmp_path / "rollout.json"),
                str(output),
                "--seed",
                "20260722",
                "--prompt-chunk-size",
                "3",
            ]
        )
        == 0
    )
    announcement = json.loads(capsys.readouterr().out)
    assert announcement["device"] == "cpu"
    assert announcement["status"] == "ok"
    assert calls == [
        {
            "seed": 20260722,
            "artifact_dir": str(tmp_path / "artifact"),
            "comparison_json": str(tmp_path / "comparison.json"),
            "rollout_json": str(tmp_path / "rollout.json"),
            "output_json": str(output),
            "prompt_chunk_size": 3,
        }
    ]
