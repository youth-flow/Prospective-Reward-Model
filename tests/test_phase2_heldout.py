from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Sequence

import pytest
import torch
import torch.nn.functional as F

from smart_reward.contracts import BT_MLE, CANONICAL_LEARNERS, PRORM_PLUS
from smart_reward.data import CandidateNode
from smart_reward.experiment import EvaluationTensorData
from smart_reward.metrics import local_regret, natural_direction_metrics
from smart_reward.oracle import RobustOracleTransform
from smart_reward.phase2_heldout import (
    PHASE2_HELDOUT_SCHEMA_V2,
    DeferredHeldoutInputs,
    DeferredHeldoutSplit,
    FrozenHeldoutEvaluationState,
    _preference_fit,
    heldout_evaluation_sha256,
    score_and_evaluate_deferred_heldout,
    verify_heldout_evaluation_payload,
)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _candidate(split: str, prompt_id: str, candidate_index: int) -> CandidateNode:
    return CandidateNode(
        prompt_id=prompt_id,
        candidate_id=f"{prompt_id}::candidate::{candidate_index}",
        prompt=f"{split} prompt {prompt_id}",
        response=f"{split}:{prompt_id}:{candidate_index}",
        token_ids=(1, 10 + candidate_index),
        response_mask=(0, 1),
        terminated_by_eos=True,
        reached_max_length=False,
    )


def _split(
    split: str,
    *,
    score_scale: float,
    target_offset: float = 0.0,
) -> tuple[DeferredHeldoutSplit, torch.Tensor]:
    prompt_ids = (f"{split}-0", f"{split}-1")
    base_scores = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
            [[1.5, 0.0], [0.0, 1.5], [-1.5, 0.0], [0.0, -1.5]],
        ],
        dtype=torch.float32,
    )
    features = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
            [[2.0, 0.0], [0.0, 2.0], [-2.0, 0.0], [0.0, -2.0]],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor(
        [
            [1.0, 0.25, -1.0, -0.25],
            [0.75, -0.5, -0.75, 0.5],
        ],
        dtype=torch.float32,
    )
    targets = targets + target_offset
    return (
        DeferredHeldoutSplit(
            split=split,
            prompt_ids=prompt_ids,
            policy_scores=base_scores * score_scale,
            reward_features=features,
            candidates=tuple(
                _candidate(split, prompt_id, candidate_index)
                for prompt_id in prompt_ids
                for candidate_index in range(4)
            ),
        ),
        targets,
    )


def _deferred() -> tuple[DeferredHeldoutInputs, dict[str, torch.Tensor]]:
    validation, validation_targets = _split("validation", score_scale=0.5)
    test, test_targets = _split("test", score_scale=1.0)
    return (
        DeferredHeldoutInputs(validation=validation, test=test),
        {"validation": validation_targets, "test": test_targets},
    )


def _state(*, beta: float = 2.0) -> FrozenHeldoutEvaluationState:
    heads = {
        BT_MLE: (0.5, -0.25),
        PRORM_PLUS: (0.75, 0.125),
    }
    heads_sha = _canonical_sha256({learner: list(heads[learner]) for learner in CANONICAL_LEARNERS})
    return FrozenHeldoutEvaluationState(
        source_config_hash="1" * 64,
        phase2_design_sha256="2" * 64,
        phase2_runtime_contract_sha256="3" * 64,
        seed=20260801,
        heads=heads,
        heads_sha256=heads_sha,
        training_design_sha256="2" * 64,
        beta_common=beta,
        deployment_identity={
            "arm_order": ["zero_b", BT_MLE, PRORM_PLUS, "oracle_step"],
            "arms": {
                "zero_b": {"displacement_sha256": "4" * 64},
                BT_MLE: {"displacement_sha256": "5" * 64},
                PRORM_PLUS: {"displacement_sha256": "6" * 64},
                "oracle_step": {"displacement_sha256": "7" * 64},
            },
        },
    )


class _Oracle:
    def __init__(
        self,
        deferred: DeferredHeldoutInputs,
        targets: dict[str, torch.Tensor],
        *,
        malformed: str | None = None,
    ) -> None:
        self.deferred = deferred
        self.targets = targets
        self.malformed = malformed
        self.calls: list[tuple[tuple[str, ...], tuple[str, ...]]] = []

    def score_transformed(
        self,
        prompts: Sequence[str],
        responses: Sequence[str],
        *,
        transform: RobustOracleTransform,
        batch_size: int,
    ) -> torch.Tensor:
        assert transform == RobustOracleTransform(b=0.0, tau=1.0)
        assert batch_size == 3
        self.calls.append((tuple(prompts), tuple(responses)))
        values = torch.cat(
            [self.targets["validation"].reshape(-1), self.targets["test"].reshape(-1)]
        )
        if self.malformed == "shape":
            return values[:-1]
        if self.malformed == "nan":
            values[0] = float("nan")
        return values


def _evaluate(
    deferred: DeferredHeldoutInputs,
    targets: dict[str, torch.Tensor],
    *,
    beta: float = 2.0,
    malformed: str | None = None,
    result_schema_version: str = "phase2-heldout-fixed-beta/v1",
) -> dict[str, object]:
    return score_and_evaluate_deferred_heldout(
        _Oracle(deferred, targets, malformed=malformed),
        deferred,
        _state(beta=beta),
        transform=RobustOracleTransform(b=0.0, tau=1.0),
        oracle_chat_template_sha256="8" * 64,
        batch_size=3,
        relative_damping=0.01,
        pcg_dtype="float64",
        pcg_max_iterations=100,
        pcg_tolerance=1.0e-10,
        result_schema_version=result_schema_version,
    )


def test_deferred_payload_drops_old_artifact_targets_by_construction() -> None:
    split, _ = _split("validation", score_scale=1.0)
    old_a = EvaluationTensorData(
        prompt_ids=split.prompt_ids,
        policy_scores=split.policy_scores,
        reward_features=split.reward_features,
        true_rewards=torch.zeros((2, 4), dtype=torch.float32),
    )
    old_b = EvaluationTensorData(
        prompt_ids=split.prompt_ids,
        policy_scores=split.policy_scores,
        reward_features=split.reward_features,
        true_rewards=torch.full((2, 4), 1.0e6, dtype=torch.float32),
    )

    deferred_a = DeferredHeldoutSplit.from_evaluation_tensor(
        "validation",
        old_a,
        split.candidates,
    )
    deferred_b = DeferredHeldoutSplit.from_evaluation_tensor(
        "validation",
        old_b,
        split.candidates,
    )

    assert deferred_a.identity_sha256 == deferred_b.identity_sha256
    assert not hasattr(deferred_a, "true_rewards")
    assert not hasattr(deferred_a, "target_rewards")
    assert deferred_a.identity_payload()["contains_oracle_targets"] is False


def test_common_beta_arithmetic_matches_phase1_definition_and_split_damping() -> None:
    deferred, targets = _deferred()
    result = _evaluate(deferred, targets, beta=2.0)

    assert result["beta_common"] == 2.0
    validation = result["splits"]["validation"]
    test = result["splits"]["test"]
    # Test scores are exactly twice validation scores, so mean(diag(F)) and
    # the split-specific absolute damping are four times larger.
    assert test["absolute_damping"] == pytest.approx(4.0 * validation["absolute_damping"])
    assert result["formal_gate_split"] == "test"

    split = deferred.test
    head = torch.tensor(_state().heads[PRORM_PLUS], dtype=torch.float32)
    predicted = split.reward_features @ head
    flat_scores = split.policy_scores.to(torch.float64).reshape(-1, split.policy_dimension)
    expected_damping = 0.01 * float(flat_scores.square().mean(dim=0).mean().item())
    expected_regret = local_regret(
        split.policy_scores,
        predicted,
        targets["test"],
        damping=expected_damping,
        beta=2.0,
        pcg_tolerance=1.0e-10,
        pcg_max_iterations=100,
        pcg_dtype="float64",
    )
    expected_directions = natural_direction_metrics(
        split.policy_scores,
        predicted,
        targets["test"],
        damping=expected_damping,
        pcg_tolerance=1.0e-10,
        pcg_max_iterations=100,
        pcg_dtype="float64",
    )
    observed = test["learners"][PRORM_PLUS]
    assert observed["local_regret_at_frozen_global_beta"] == pytest.approx(expected_regret.item())
    assert observed["native_beta1_squared_fisher_direction_error"] == pytest.approx(
        expected_directions.squared_fisher_error.item()
    )
    assert observed["native_beta1_fisher_cosine"] == pytest.approx(
        expected_directions.fisher_cosine.item()
    )


def test_formal_v1_preserves_float32_reward_prediction_before_fp64_solve() -> None:
    deferred, targets = _deferred()
    heads = {
        BT_MLE: (12345.678901234, -9876.543210987),
        PRORM_PLUS: (-23456.789012345, 34567.890123456),
    }
    heads_sha = _canonical_sha256({learner: list(heads[learner]) for learner in CANONICAL_LEARNERS})
    state = FrozenHeldoutEvaluationState(
        source_config_hash="1" * 64,
        phase2_design_sha256="2" * 64,
        phase2_runtime_contract_sha256="3" * 64,
        seed=20260801,
        heads=heads,
        heads_sha256=heads_sha,
        training_design_sha256="2" * 64,
        beta_common=2.0,
        deployment_identity=_state().deployment_identity,
    )
    formal_v1 = score_and_evaluate_deferred_heldout(
        _Oracle(deferred, targets),
        deferred,
        state,
        transform=RobustOracleTransform(b=0.0, tau=1.0),
        oracle_chat_template_sha256="8" * 64,
        batch_size=3,
        relative_damping=0.01,
        pcg_dtype="float64",
        pcg_max_iterations=100,
        pcg_tolerance=1.0e-10,
    )

    split = deferred.test
    predicted_v1 = split.reward_features @ torch.tensor(
        heads[PRORM_PLUS],
        dtype=split.reward_features.dtype,
    )
    flat_scores = split.policy_scores.to(torch.float64).reshape(-1, split.policy_dimension)
    damping = 0.01 * float(flat_scores.square().mean(dim=0).mean().item())
    expected = local_regret(
        split.policy_scores,
        predicted_v1,
        targets["test"],
        damping=damping,
        beta=state.beta_common,
        pcg_tolerance=1.0e-10,
        pcg_max_iterations=100,
        pcg_dtype="float64",
    )

    observed = formal_v1["splits"]["test"]["learners"][PRORM_PLUS][
        "local_regret_at_frozen_global_beta"
    ]
    assert observed == float(expected.item())

    budgeted_v2 = score_and_evaluate_deferred_heldout(
        _Oracle(deferred, targets),
        deferred,
        state,
        transform=RobustOracleTransform(b=0.0, tau=1.0),
        oracle_chat_template_sha256="8" * 64,
        batch_size=3,
        relative_damping=0.01,
        pcg_dtype="float64",
        pcg_max_iterations=100,
        pcg_tolerance=1.0e-10,
        result_schema_version=PHASE2_HELDOUT_SCHEMA_V2,
    )
    assert (
        budgeted_v2["splits"]["test"]["learners"][PRORM_PLUS]["local_regret_at_frozen_global_beta"]
        != observed
    )


def test_primary_regret_uses_frozen_global_beta_while_native_diagnostics_do_not() -> None:
    deferred, targets = _deferred()
    beta_one = _evaluate(deferred, targets, beta=1.0)
    beta_two = _evaluate(deferred, targets, beta=2.0)

    for split_name in ("validation", "test"):
        for learner in CANONICAL_LEARNERS:
            one = beta_one["splits"][split_name]["learners"][learner]
            two = beta_two["splits"][split_name]["learners"][learner]
            assert two["local_regret_at_frozen_global_beta"] == pytest.approx(
                0.5 * one["local_regret_at_frozen_global_beta"]
            )
            assert two["native_beta1_squared_fisher_direction_error"] == pytest.approx(
                one["native_beta1_squared_fisher_direction_error"]
            )
            assert two["native_beta1_fisher_cosine"] == pytest.approx(
                one["native_beta1_fisher_cosine"]
            )


def test_oracle_call_is_validation_then_test_candidate_order() -> None:
    deferred, targets = _deferred()
    oracle = _Oracle(deferred, targets)

    score_and_evaluate_deferred_heldout(
        oracle,
        deferred,
        _state(),
        transform=RobustOracleTransform(b=0.0, tau=1.0),
        oracle_chat_template_sha256="8" * 64,
        batch_size=3,
        relative_damping=0.01,
        pcg_dtype="float64",
        pcg_max_iterations=100,
        pcg_tolerance=1.0e-10,
    )

    assert len(oracle.calls) == 1
    prompts, responses = oracle.calls[0]
    expected = (*deferred.validation.candidates, *deferred.test.candidates)
    assert prompts == tuple(candidate.prompt for candidate in expected)
    assert responses == tuple(candidate.response for candidate in expected)


@pytest.mark.parametrize(("malformed", "match"), [("shape", "one finite"), ("nan", "one finite")])
def test_malformed_final_oracle_output_is_rejected(
    malformed: str,
    match: str,
) -> None:
    deferred, targets = _deferred()
    with pytest.raises(ValueError, match=match):
        _evaluate(deferred, targets, malformed=malformed)


def test_deferred_tensor_tampering_is_detected_before_oracle_call() -> None:
    deferred, targets = _deferred()
    oracle = _Oracle(deferred, targets)
    deferred.test.policy_scores[0, 0, 0] += 1.0

    with pytest.raises(RuntimeError, match="policy scores changed"):
        score_and_evaluate_deferred_heldout(
            oracle,
            deferred,
            _state(),
            transform=RobustOracleTransform(b=0.0, tau=1.0),
            oracle_chat_template_sha256="8" * 64,
            batch_size=3,
            relative_damping=0.01,
            pcg_dtype="float64",
            pcg_max_iterations=100,
            pcg_tolerance=1.0e-10,
        )

    assert oracle.calls == []


def test_result_hash_detects_metric_tampering_and_no_raw_vector_is_serialized() -> None:
    deferred, targets = _deferred()
    result = _evaluate(deferred, targets)
    digest = heldout_evaluation_sha256(result)

    verify_heldout_evaluation_payload(result, expected_sha256=digest)
    rendered = json.dumps(result, allow_nan=False, sort_keys=True)
    assert "raw_oracle_logits_serialized" in rendered
    assert '"direction":' not in rendered
    assert all(
        result["splits"][split]["raw_oracle_logits_serialized"] is False
        for split in ("validation", "test")
    )

    tampered = copy.deepcopy(result)
    tampered["splits"]["test"]["learners"][PRORM_PLUS]["local_regret_at_frozen_global_beta"] += 0.1
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        verify_heldout_evaluation_payload(tampered, expected_sha256=digest)


def test_v2_serializes_prompt_equal_operational_preference_fit_and_explicit_pcg() -> None:
    deferred, targets = _deferred()
    result = _evaluate(
        deferred,
        targets,
        result_schema_version=PHASE2_HELDOUT_SCHEMA_V2,
    )
    digest = heldout_evaluation_sha256(result)
    verify_heldout_evaluation_payload(result, expected_sha256=digest)

    assert result["schema_version"] == PHASE2_HELDOUT_SCHEMA_V2
    assert result["formal_claim_eligible"] is False
    assert result["formal_gate_split"] is None
    assert result["primary_descriptive_split"] == "test"
    contract = result["operational_oracle_preference_fit"]
    assert contract["expected_pairs_per_prompt_for_four_candidates"] == 6
    assert contract["aggregation"] == "mean_pairs_within_prompt_then_mean_prompts"
    assert contract["oracle_or_predicted_tie_accuracy_credit"] == 0.5

    split = deferred.test
    head = torch.tensor(_state().heads[PRORM_PLUS], dtype=torch.float64)
    predicted = split.reward_features.to(torch.float64) @ head
    target = targets["test"].to(torch.float64)
    pairs = torch.combinations(torch.arange(split.num_candidates), r=2)
    predicted_margins = predicted[:, pairs[:, 0]] - predicted[:, pairs[:, 1]]
    target_margins = target[:, pairs[:, 0]] - target[:, pairs[:, 1]]
    oracle_probabilities = torch.sigmoid(target_margins)
    per_pair_ce = F.binary_cross_entropy_with_logits(
        predicted_margins,
        oracle_probabilities,
        reduction="none",
    )
    per_pair_mae = torch.abs(torch.sigmoid(predicted_margins) - oracle_probabilities)
    per_pair_accuracy = (torch.sign(predicted_margins) == torch.sign(target_margins)).to(
        torch.float64
    )
    per_pair_accuracy[(predicted_margins == 0.0) | (target_margins == 0.0)] = 0.5

    fit = result["splits"]["test"]["preference_fit"][PRORM_PLUS]
    assert set(fit) == {
        "oracle_pairwise_cross_entropy",
        "oracle_probability_mae",
        "pairwise_order_accuracy",
    }
    assert fit["oracle_pairwise_cross_entropy"] == pytest.approx(
        per_pair_ce.mean(dim=1).mean().item()
    )
    assert fit["oracle_probability_mae"] == pytest.approx(per_pair_mae.mean(dim=1).mean().item())
    assert fit["pairwise_order_accuracy"] == pytest.approx(
        per_pair_accuracy.mean(dim=1).mean().item()
    )

    for split_name in ("validation", "test"):
        pcg_evidence = result["splits"][split_name]["heldout_pcg_evidence"]
        assert pcg_evidence["operator"] == (
            "node_empirical_fisher_plus_split_specific_isotropic_damping"
        )
        assert pcg_evidence["pcg_dtype"] == "float64"
        assert pcg_evidence["preconditioner"] == "none"
        assert pcg_evidence["all_solves_cold_start"] is True
        assert pcg_evidence["all_solves_converged"] is True
        assert pcg_evidence["target_direction"]["converged"] is True
        for learner in CANONICAL_LEARNERS:
            assert set(pcg_evidence["learners"][learner]) == {
                "predicted_direction",
                "reward_error_direction",
            }
            assert all(
                solve["converged"] is True for solve in pcg_evidence["learners"][learner].values()
            )

    tampered = copy.deepcopy(result)
    tampered["splits"]["test"]["heldout_pcg_evidence"]["learners"][PRORM_PLUS][
        "predicted_direction"
    ]["converged"] = False
    with pytest.raises(ValueError, match="does not prove PCG convergence"):
        verify_heldout_evaluation_payload(
            tampered,
            expected_sha256=heldout_evaluation_sha256(tampered),
        )


def test_operational_preference_fit_awards_half_credit_to_pairwise_ties() -> None:
    predicted = torch.zeros((2, 4), dtype=torch.float64)
    target = torch.tensor(
        [[3.0, 2.0, 1.0, 0.0], [0.0, 1.0, 2.0, 3.0]],
        dtype=torch.float64,
    )

    fit = _preference_fit(predicted, target)

    assert fit["pairwise_order_accuracy"] == 0.5


def test_frozen_state_rejects_head_identity_tampering() -> None:
    with pytest.raises(ValueError, match="heads do not match"):
        FrozenHeldoutEvaluationState(
            source_config_hash="1" * 64,
            phase2_design_sha256="2" * 64,
            phase2_runtime_contract_sha256="3" * 64,
            seed=20260801,
            heads={BT_MLE: (1.0, 0.0), PRORM_PLUS: (0.0, 1.0)},
            heads_sha256="9" * 64,
            training_design_sha256="2" * 64,
            beta_common=2.0,
            deployment_identity={},
        )
