from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import pytest
import torch

import smart_reward.optimization_audit as audit_module
from smart_reward.baseline import repeated_btl_nll
from smart_reward.cli import main
from smart_reward.config import config_hash, load_config
from smart_reward.contracts import BT_MLE, PRORM_PLUS
from smart_reward.experiment import (
    ControlledFeatureExperiment,
    EvaluationTensorData,
    TrainingTensorData,
    compile_feature_experiment_config,
    run_feature_experiment,
)
from smart_reward.linear import DampedEmpiricalFisher
from smart_reward.objective import empirical_moment, envelope_weights
from smart_reward.optimization_audit import (
    audit_phase1_head_optimization,
    evaluate_saved_head_optimization,
)
from smart_reward.pcg import pcg

ROOT = Path(__file__).resolve().parents[1]


def _train(dtype: torch.dtype = torch.float64) -> TrainingTensorData:
    return TrainingTensorData(
        prompt_ids=("train-0", "train-1", "train-2", "train-3"),
        policy_scores=torch.tensor(
            [
                [[1.0, 0.0], [0.0, 1.0], [0.2, -0.4]],
                [[2.0, 1.0], [1.0, -1.0], [-0.5, 0.7]],
                [[0.5, -0.2], [-0.4, 0.3], [1.1, 0.2]],
                [[-0.7, 0.8], [0.6, 0.1], [0.3, -1.0]],
            ],
            dtype=dtype,
        ),
        reward_features=torch.tensor(
            [
                [[1.0, 0.0], [0.0, 1.0], [0.3, -0.1]],
                [[1.0, 1.0], [-1.0, 0.5], [0.4, 0.8]],
                [[0.2, 0.7], [-0.3, -0.1], [1.2, -0.5]],
                [[-0.4, 0.2], [0.8, -0.6], [0.1, 0.9]],
            ],
            dtype=dtype,
        ),
        h=torch.tensor([0.2, -0.4, 0.6, -0.1], dtype=dtype),
        left_wins=torch.tensor([3, 1, 4, 2], dtype=torch.int64),
        num_annotations=torch.tensor([4, 3, 5, 6], dtype=torch.int64),
    )


def _heads() -> dict[str, torch.Tensor]:
    return {
        BT_MLE: torch.tensor([0.2, -0.1], dtype=torch.float64),
        PRORM_PLUS: torch.tensor([0.1, 0.3], dtype=torch.float64),
    }


def _evaluation(
    prefix: str,
    *,
    offset: float,
    dtype: torch.dtype,
) -> EvaluationTensorData:
    base = torch.arange(9, dtype=dtype).reshape(3, 3) + offset
    scores = torch.stack(
        (
            torch.sin(base) + 0.2,
            torch.cos(0.7 * base) - 0.1,
        ),
        dim=-1,
    )
    features = torch.stack(
        (
            0.15 * base,
            torch.sin(0.4 * base),
        ),
        dim=-1,
    )
    rewards = 0.8 * features[..., 0] - 0.3 * features[..., 1]
    return EvaluationTensorData(
        prompt_ids=tuple(f"{prefix}-{index}" for index in range(3)),
        policy_scores=scores,
        reward_features=features,
        true_rewards=rewards,
    )


def _experiment(dtype: torch.dtype = torch.float32) -> ControlledFeatureExperiment:
    return ControlledFeatureExperiment(
        train=_train(dtype),
        validation=_evaluation("validation", offset=0.3, dtype=dtype),
        test=_evaluation("test", offset=1.7, dtype=dtype),
    )


def test_bt_audit_is_exact_repeated_label_objective_and_unclipped_gradient() -> None:
    train = _train()
    heads = _heads()
    original = {name: value.clone() for name, value in heads.items()}
    result = evaluate_saved_head_optimization(
        train,
        heads,
        beta=1.2,
        absolute_damping=0.07,
        pcg_max_iterations=20,
        pcg_tolerance=1.0e-12,
    )

    batch = train.to_training_batch()
    expected_head = heads[BT_MLE].detach().clone().requires_grad_(True)
    margins = (batch.left_features - batch.right_features) @ expected_head
    expected_objective = repeated_btl_nll(
        margins,
        batch.left_wins,
        batch.num_annotations,
    )
    (expected_gradient,) = torch.autograd.grad(expected_objective, expected_head)
    observed = result["learners"][BT_MLE]

    assert observed["objective"] == pytest.approx(expected_objective.item(), rel=1.0e-14)
    assert observed["gradient_l2_norm"] == pytest.approx(
        torch.linalg.vector_norm(expected_gradient).item(),
        rel=1.0e-14,
    )
    assert observed["gradient_to_head_norm_ratio"] == pytest.approx(
        observed["gradient_l2_norm"] / observed["head_l2_norm"]
    )
    assert observed["gradient"] == "full_data_unclipped"
    assert result["optimizer_constructed"] is False
    assert result["optimizer_step_called"] is False
    assert all(torch.equal(heads[name], original[name]) for name in heads)


def test_prorm_audit_uses_fresh_fp64_dual_and_envelope_gradient() -> None:
    train = _train()
    heads = _heads()
    beta = 1.2
    damping = 0.07
    result = evaluate_saved_head_optimization(
        train,
        heads,
        beta=beta,
        absolute_damping=damping,
        pcg_max_iterations=20,
        pcg_tolerance=1.0e-12,
        pcg_residual_recompute_interval=3,
    )

    batch = train.to_training_batch()
    x = (batch.left_features - batch.right_features).to(torch.float64)
    z = batch.edge_scores.to(torch.float64)
    nodes = batch.node_scores.to(torch.float64)
    h = batch.h.to(torch.float64)
    margins = x @ heads[PRORM_PLUS]
    moment = empirical_moment(z, margins, h)
    solved = pcg(
        DampedEmpiricalFisher(nodes, damping).matvec,
        moment,
        inverse_diagonal=None,
        max_iterations=20,
        tolerance=1.0e-12,
        residual_recompute_interval=3,
    )
    weights = envelope_weights(z, solved.solution, beta=beta)
    expected_gradient = x.mT @ weights / x.shape[0]
    observed = result["learners"][PRORM_PLUS]

    assert solved.converged
    assert observed["fresh_inner_pcg"]["converged"] is True
    assert observed["fresh_inner_pcg"]["warm_start_used"] is False
    assert observed["fresh_inner_pcg"]["dtype"] == "float64"
    assert observed["gradient_l2_norm"] == pytest.approx(
        torch.linalg.vector_norm(expected_gradient).item(),
        rel=1.0e-12,
    )
    assert abs(observed["dual_loss_minus_saddle_value"]) < 1.0e-20
    assert observed["gradient_definition"] == "fresh_dual_envelope_gradient"


def test_zero_head_reports_undefined_ratio_without_infinity() -> None:
    heads = _heads()
    heads[BT_MLE] = torch.zeros(2, dtype=torch.float64)
    result = evaluate_saved_head_optimization(
        _train(),
        heads,
        beta=1.0,
        absolute_damping=0.1,
        pcg_max_iterations=20,
        pcg_tolerance=1.0e-10,
    )

    bt = result["learners"][BT_MLE]
    assert bt["head_norm_is_zero"] is True
    assert bt["gradient_to_head_norm_ratio"] is None


def _comparison_payload(
    config: dict[str, object],
    experiment: ControlledFeatureExperiment,
    *,
    artifact_digest: str,
) -> dict[str, object]:
    runtime = compile_feature_experiment_config(config)
    result = run_feature_experiment(experiment, runtime)
    return {
        "schema_version": "controlled-comparison/v2",
        "config_hash": config_hash(config),
        "seed": 20260722,
        "artifact_dir": "artifact",
        "artifact_metadata_sha256": artifact_digest,
        "run_manifest": "run-manifest.json",
        "run_manifest_sha256": "e" * 64,
        "environment_identity": {
            "formal": True,
            "git_commit": "a" * 40,
            "image_sha256": "b" * 64,
            "hf_inventory_sha256": "c" * 64,
            "account": "sigroup",
            "partition": "gpu-l20",
            "gpu_models": ["NVIDIA L20"],
        },
        "damping_runs": [
            {
                "damping_multiplier": 1.0,
                "result": result.to_dict(),
            }
        ],
    }


def test_bound_audit_checks_comparison_objectives_and_writes_new_file_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(ROOT / "configs" / "smoke.yaml")
    experiment = _experiment(torch.float32)
    artifact_digest = "d" * 64
    comparison = _comparison_payload(
        config,
        experiment,
        artifact_digest=artifact_digest,
    )
    comparison_path = tmp_path / "comparison.json"
    comparison_path.write_text(json.dumps(comparison), encoding="utf-8")
    comparison_digest = hashlib.sha256(comparison_path.read_bytes()).hexdigest()
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

    output = tmp_path / "audit" / "optimization.json"
    payload = audit_phase1_head_optimization(
        config,
        seed=20260722,
        artifact_dir=tmp_path / "artifact",
        comparison_json=comparison_path,
        output_json=output,
    )
    assert payload["schema_version"] == "optimization-audit/v1"
    assert payload["sources"]["comparison_sha256"] == comparison_digest
    assert payload["diagnostic_contract"]["optimizer_step_called"] is False
    assert payload["diagnostic_contract"]["optimization_convergence_threshold_declared"] is False
    assert payload["diagnostic_contract"]["optimization_convergence_claimed"] is False
    for learner in (BT_MLE, PRORM_PLUS):
        learner_result = payload["learners"][learner]
        assert learner_result["comparison_training_record"]["initial_train_objective"] >= 0.0
        assert learner_result["comparison_training_record"]["final_train_objective"] >= 0.0
        assert math.isfinite(learner_result["objective_binding"]["audit_minus_comparison_final"])
    assert json.loads(output.read_text(encoding="utf-8")) == payload

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        audit_phase1_head_optimization(
            config,
            seed=20260722,
            artifact_dir=tmp_path / "artifact",
            comparison_json=comparison_path,
            output_json=output,
        )


def test_corrupt_initial_head_binding_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(ROOT / "configs" / "smoke.yaml")
    experiment = _experiment(torch.float32)
    artifact_digest = "d" * 64
    comparison = _comparison_payload(
        config,
        experiment,
        artifact_digest=artifact_digest,
    )
    corrupt = copy.deepcopy(comparison)
    corrupt["damping_runs"][0]["result"][BT_MLE]["initial_head_sha256"] = "f" * 64
    comparison_path = tmp_path / "corrupt.json"
    comparison_path.write_text(json.dumps(corrupt), encoding="utf-8")
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

    with pytest.raises(ValueError, match="zero initialization"):
        audit_phase1_head_optimization(
            config,
            seed=20260722,
            artifact_dir=tmp_path / "artifact",
            comparison_json=comparison_path,
            output_json=tmp_path / "output.json",
        )


def test_cli_wires_cpu_optimization_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[dict[str, object]] = []

    def fake_audit(config, **kwargs):
        calls.append(kwargs)
        return {"config_hash": config_hash(config)}

    monkeypatch.setattr(audit_module, "audit_phase1_head_optimization", fake_audit)
    output = tmp_path / "optimization.json"
    assert (
        main(
            [
                "optimization-audit",
                str(ROOT / "configs" / "smoke.yaml"),
                str(tmp_path / "artifact"),
                str(tmp_path / "comparison.json"),
                str(output),
                "--seed",
                "20260722",
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
            "output_json": str(output),
        }
    ]
