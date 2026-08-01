from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from smart_reward.artifacts import (
    exact_delta_artifact_metadata_sha256,
    save_exact_delta_artifact,
)
from smart_reward.config import TRPO_PROTOCOL, config_hash, load_config
from smart_reward.exact import ExactDeltaExperiment, ExactSplitData
from smart_reward.ngd_evaluation import (
    BETAS,
    POLICIES,
    PROTOCOL,
    evaluate_candidate_pool,
    run_ngd_evaluation,
)
from smart_reward.trpo_run import SCHEMA as TRPO_REWARD_SCHEMA

ROOT = Path(__file__).parents[1]


def _split() -> ExactSplitData:
    generator = torch.Generator().manual_seed(41)
    scores = torch.randn(9, 6, 5, generator=generator, dtype=torch.float64)
    rewards = torch.randn(9, 6, generator=generator, dtype=torch.float64)
    return ExactSplitData(
        prompt_ids=tuple(f"test-{index}" for index in range(9)),
        policy_scores=scores,
        reward_features=torch.randn(9, 6, 3, generator=generator, dtype=torch.float64),
        true_rewards=rewards,
    )


@pytest.mark.parametrize("beta", BETAS)
def test_five_policy_evaluation_satisfies_exact_gibbs_identities(beta: float) -> None:
    split = _split()
    directions = {
        "mle_rm": torch.tensor([0.2, -0.1, 0.3, 0.4, -0.2], dtype=torch.float64),
        "pro_rm": torch.tensor([0.1, 0.3, -0.2, 0.2, 0.5], dtype=torch.float64),
        "oracle": torch.tensor([0.4, -0.2, 0.1, 0.3, 0.2], dtype=torch.float64),
    }
    result = evaluate_candidate_pool(split, directions, beta=beta)
    assert tuple(result["policies"]) == POLICIES
    assert result["policies"]["pi0"]["K"] == pytest.approx(0.0, abs=1.0e-15)
    assert result["policies"]["tabular"]["J"] == pytest.approx(result["J_close"])
    assert result["policies"]["tabular"]["delta_J"] == pytest.approx(0.0, abs=1.0e-14)
    assert result["policies"]["tabular"]["beta_KL"] == pytest.approx(0.0, abs=1.0e-14)
    for metrics in result["policies"].values():
        assert metrics["delta_J"] == pytest.approx(metrics["beta_KL"], abs=1.0e-12)
        assert metrics["delta_J"] >= -1.0e-12


def test_beta_grid_is_frozen() -> None:
    split = _split()
    zero = torch.zeros(split.policy_dimension, dtype=torch.float64)
    with pytest.raises(ValueError, match="frozen"):
        evaluate_candidate_pool(
            split,
            {"mle_rm": zero, "pro_rm": zero, "oracle": zero},
            beta=3.0,
        )


def test_seed_evaluator_validates_and_bridges_fisher_trpo_ancestors(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs" / "fisher_trpo_smoke.yaml")
    digest = config_hash(config)
    train = _split()
    validation_raw = _split()
    test_raw = _split()
    validation = ExactSplitData(
        prompt_ids=tuple(f"validation-{index}" for index in range(validation_raw.num_prompts)),
        policy_scores=validation_raw.policy_scores,
        reward_features=validation_raw.reward_features,
        true_rewards=validation_raw.true_rewards,
    )
    test = ExactSplitData(
        prompt_ids=tuple(f"formal-test-{index}" for index in range(test_raw.num_prompts)),
        policy_scores=test_raw.policy_scores,
        reward_features=test_raw.reward_features,
        true_rewards=test_raw.true_rewards,
    )
    experiment = ExactDeltaExperiment(train, validation, test)
    artifact = tmp_path / "artifact"
    seed = int(config["run"]["seeds"][0])
    save_exact_delta_artifact(
        experiment,
        artifact,
        config_hash=digest,
        seed=seed,
    )
    dimension = test.policy_dimension
    direction = [0.01 * (index + 1) for index in range(dimension)]
    update_record = {"update": direction}
    reward_result = tmp_path / "reward.json"
    reward_result.write_text(
        json.dumps(
            {
                "schema": TRPO_REWARD_SCHEMA,
                "protocol": TRPO_PROTOCOL,
                "config_sha256": digest,
                "artifact_metadata_sha256": exact_delta_artifact_metadata_sha256(artifact),
                "fisher_selection_sha256": "1" * 64,
                "selected_relative_damping": 1.0,
                "seed": seed,
                "methods": {
                    "MLE-RM": {"head_weight": [0.1, -0.2, 0.3]},
                    "Pro-RM": {"head_weight": [-0.1, 0.3, 0.2]},
                },
                "policy_directions": {
                    "mle_rm": direction,
                    "pro_rm": list(reversed(direction)),
                    "oracle": [2.0 * value for value in direction],
                },
                "policy_updates": {
                    method: {"0.001": update_record}
                    for method in ("mle_rm", "pro_rm", "oracle")
                },
                "dimensions": {"policy_tangent": dimension},
                "producer": {"git_commit": "0" * 40},
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "evaluation.json"
    result = run_ngd_evaluation(
        config,
        artifact,
        reward_result,
        output,
        seed=seed,
        device="cpu",
    )
    assert output.exists()
    assert result["protocol"] == PROTOCOL
    assert result["betas"] == list(BETAS)
    assert result["test_usage"] == "formal_evaluation_only_no_hyperparameter_selection"
    assert set(result["reward"]) == {"mle", "pro"}
    assert set(result["policy"]) == {str(beta) for beta in BETAS}
    assert result["provenance_bridge"]["source_config_sha256"] == digest
