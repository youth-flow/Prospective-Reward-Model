from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import smart_reward.hf as hf
import smart_reward.rollout as rollout_module
from smart_reward.experiment import TrainingTensorData
from smart_reward.hf import ExactTokenCandidates, FixedALoRASetup
from smart_reward.metrics import policy_reward_moment
from smart_reward.rollout import (
    match_fixed_a_measured_kl,
    oracle_rollout_improvement,
    policy_direction_from_head,
    policy_direction_from_node_rewards,
)
from smart_reward.scores import ParameterLayout


def _training_data(dtype: torch.dtype = torch.float64) -> TrainingTensorData:
    policy_scores = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.5]],
            [[0.5, -1.0], [1.5, 0.5], [-0.5, 1.0]],
        ],
        dtype=dtype,
    )
    reward_features = torch.tensor(
        [
            [[1.0, 2.0], [0.0, -1.0], [2.0, 0.5]],
            [[-1.0, 1.0], [1.5, 0.0], [0.5, 2.0]],
        ],
        dtype=dtype,
    )
    return TrainingTensorData(
        prompt_ids=("p0", "p1"),
        policy_scores=policy_scores,
        reward_features=reward_features,
        h=torch.zeros(2, dtype=dtype),
        left_wins=torch.tensor([1, 1], dtype=torch.int64),
        num_annotations=torch.tensor([2, 2], dtype=torch.int64),
    )


def test_policy_direction_matches_full_matrix_covariance_formula() -> None:
    train = _training_data()
    head_weight = torch.tensor([0.7, -0.2], dtype=torch.float64)
    relative_damping = 0.3
    beta = 1.7

    result = policy_direction_from_head(
        train,
        head_weight,
        relative_damping=relative_damping,
        beta=beta,
        pcg_tolerance=1.0e-13,
        pcg_absolute_tolerance=1.0e-14,
    )

    rewards = train.reward_features @ head_weight
    moment = policy_reward_moment(train.policy_scores, rewards)
    flat_scores = train.policy_scores.reshape(-1, train.policy_dimension)
    fisher = flat_scores.mT @ flat_scores / flat_scores.shape[0]
    expected_lambda = relative_damping * torch.diagonal(fisher).mean().item()
    expected = (
        torch.linalg.solve(
            fisher + expected_lambda * torch.eye(2, dtype=torch.float64),
            moment,
        )
        / beta
    )

    torch.testing.assert_close(result.direction, expected, rtol=1.0e-11, atol=1.0e-12)
    assert result.absolute_damping == pytest.approx(expected_lambda)
    assert result.mean_fisher_diagonal == pytest.approx(torch.diagonal(fisher).mean().item())
    assert result.fisher_curvature == pytest.approx(torch.dot(expected, fisher @ expected).item())
    assert result.pcg_converged
    json.dumps(result.to_dict(), allow_nan=False)


def test_policy_direction_preserves_low_rank_plus_damping_krylov_structure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_pcg = rollout_module.pcg
    observed_preconditioners: list[torch.Tensor | None] = []

    def recording_pcg(*args, **kwargs):
        observed_preconditioners.append(kwargs.get("inverse_diagonal"))
        return original_pcg(*args, **kwargs)

    monkeypatch.setattr(rollout_module, "pcg", recording_pcg)
    policy_direction_from_head(
        _training_data(),
        torch.tensor([0.7, -0.2], dtype=torch.float64),
        relative_damping=0.3,
        pcg_tolerance=1.0e-13,
        pcg_absolute_tolerance=1.0e-14,
    )

    assert observed_preconditioners == [None]


def test_policy_direction_promotes_fp32_geometry_to_fp64(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_pcg = rollout_module.pcg
    observed_rhs_dtypes: list[torch.dtype] = []

    def recording_pcg(*args, **kwargs):
        observed_rhs_dtypes.append(args[1].dtype)
        return original_pcg(*args, **kwargs)

    monkeypatch.setattr(rollout_module, "pcg", recording_pcg)
    result = policy_direction_from_head(
        _training_data(torch.float32),
        torch.tensor([0.7, -0.2]),
        relative_damping=0.3,
    )

    assert observed_rhs_dtypes == [torch.float64]
    assert result.direction.dtype == torch.float64


def test_node_reward_direction_is_identical_to_linear_head_direction() -> None:
    train = _training_data()
    head_weight = torch.tensor([0.7, -0.2], dtype=torch.float64)
    node_rewards = (train.reward_features @ head_weight).detach()
    solve_options = {
        "relative_damping": 0.3,
        "beta": 1.7,
        "pcg_tolerance": 1.0e-13,
        "pcg_absolute_tolerance": 1.0e-14,
    }

    from_head = policy_direction_from_head(train, head_weight, **solve_options)
    from_nodes = policy_direction_from_node_rewards(train, node_rewards, **solve_options)

    torch.testing.assert_close(from_nodes.direction, from_head.direction, rtol=0.0, atol=0.0)
    assert from_nodes.to_dict() == from_head.to_dict()


def test_node_reward_direction_is_invariant_to_prompt_constant_gauge() -> None:
    train = _training_data()
    node_rewards = torch.tensor(
        [[0.75, -0.5, 1.25], [-1.0, 0.25, 0.5]],
        dtype=torch.float64,
    )
    prompt_constants = torch.tensor([[8.0], [-3.0]], dtype=torch.float64)
    options = {
        "relative_damping": 0.3,
        "beta": 1.0,
        "pcg_tolerance": 1.0e-13,
        "pcg_absolute_tolerance": 1.0e-14,
    }

    original = policy_direction_from_node_rewards(train, node_rewards, **options)
    shifted = policy_direction_from_node_rewards(
        train,
        node_rewards + prompt_constants,
        **options,
    )

    torch.testing.assert_close(shifted.direction, original.direction, rtol=1.0e-12, atol=1.0e-13)
    assert shifted.moment_norm == pytest.approx(original.moment_norm, rel=1.0e-13)
    assert shifted.fisher_curvature == pytest.approx(
        original.fisher_curvature,
        rel=1.0e-12,
    )


def test_node_reward_direction_fails_closed_when_pcg_does_not_converge() -> None:
    train = _training_data()
    node_rewards = torch.tensor(
        [[0.75, -0.5, 1.25], [-1.0, 0.25, 0.5]],
        dtype=torch.float64,
    )

    with pytest.raises(RuntimeError, match="PCG did not converge"):
        policy_direction_from_node_rewards(
            train,
            node_rewards,
            relative_damping=0.3,
            pcg_max_iterations=1,
            pcg_tolerance=0.0,
            pcg_absolute_tolerance=0.0,
            require_pcg_convergence=True,
        )


def test_node_reward_direction_requires_exact_frozen_training_geometry() -> None:
    train = _training_data()
    valid = torch.zeros(
        train.num_prompts,
        train.num_candidates,
        dtype=train.policy_scores.dtype,
        device=train.policy_scores.device,
    )

    with pytest.raises(TypeError, match="torch.Tensor"):
        policy_direction_from_node_rewards(  # type: ignore[arg-type]
            train,
            [[0.0] * train.num_candidates] * train.num_prompts,
            relative_damping=0.3,
        )
    with pytest.raises(ValueError, match="shape"):
        policy_direction_from_node_rewards(
            train,
            valid[:, :-1],
            relative_damping=0.3,
        )
    with pytest.raises(TypeError, match="floating-point"):
        policy_direction_from_node_rewards(
            train,
            torch.zeros_like(valid, dtype=torch.int64),
            relative_damping=0.3,
        )
    with pytest.raises(ValueError, match="dtype and device"):
        policy_direction_from_node_rewards(
            train,
            valid.to(torch.float32),
            relative_damping=0.3,
        )
    with pytest.raises(ValueError, match="frozen and detached"):
        policy_direction_from_node_rewards(
            train,
            valid.clone().requires_grad_(True),
            relative_damping=0.3,
        )
    nonfinite = valid.clone()
    nonfinite[0, 0] = torch.inf
    with pytest.raises(ValueError, match="finite"):
        policy_direction_from_node_rewards(
            train,
            nonfinite,
            relative_damping=0.3,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_node_reward_direction_rejects_device_mismatch() -> None:
    train = _training_data()
    rewards = torch.zeros(
        train.num_prompts,
        train.num_candidates,
        dtype=train.policy_scores.dtype,
        device="cuda",
    )
    with pytest.raises(ValueError, match="dtype and device"):
        policy_direction_from_node_rewards(
            train,
            rewards,
            relative_damping=0.3,
        )


class _TinyFixedAPolicy(nn.Module):
    def __init__(self, *, raise_on_nonzero: bool = False) -> None:
        super().__init__()
        self.proj_lora_A = nn.Parameter(
            torch.tensor([0.75], dtype=torch.float64), requires_grad=False
        )
        self.proj_lora_B = nn.Parameter(torch.zeros((), dtype=torch.float64))
        self.raise_on_nonzero = raise_on_nonzero
        self.forward_b_values: list[float] = []

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
    ) -> SimpleNamespace:
        del attention_mask, use_cache
        value = float(self.proj_lora_B.detach().item())
        self.forward_b_values.append(value)
        if self.raise_on_nonzero and value != 0.0:
            raise RuntimeError("synthetic trial failure")
        positive = self.proj_lora_B.expand(input_ids.shape)
        return SimpleNamespace(logits=torch.stack((positive, -positive), dim=-1))


def _setup_and_candidates(
    *,
    raise_on_nonzero: bool = False,
) -> tuple[_TinyFixedAPolicy, FixedALoRASetup, ExactTokenCandidates]:
    model = _TinyFixedAPolicy(raise_on_nonzero=raise_on_nonzero).eval()
    named_tangent = (("proj_lora_B", model.proj_lora_B),)
    layout = ParameterLayout.from_named_parameters(named_tangent)
    setup = FixedALoRASetup(
        model=model,
        layout=layout,
        a_state_sha256=hf._fingerprint_named_tensors((("proj_lora_A", model.proj_lora_A),)),
        trainable_names=layout.names,
    )
    candidates = ExactTokenCandidates(
        input_ids=torch.tensor([[0, 1, 0], [1, 0, 1]], dtype=torch.int64),
        attention_mask=torch.ones((2, 3), dtype=torch.bool),
        response_mask=torch.tensor([[False, True, True], [False, True, True]], dtype=torch.bool),
        terminated_by_eos=torch.tensor([True, True]),
        reached_max_length=torch.tensor([False, False]),
        prompt_width=1,
        source_model_id=id(model),
        source_trainable_sha256=hf._fingerprint_named_tensors(named_tangent),
    )
    return model, setup, candidates


def test_measured_kl_trials_overwrite_zero_origin_and_hit_target() -> None:
    model, setup, candidates = _setup_and_candidates()
    # This train Fisher gives an intentionally imperfect quadratic initializer
    # of 0.02.  Actual KL needs about 0.10, forcing several observable trials.
    train_node_scores = torch.full((2, 2, 1), math.sqrt(50.0), dtype=torch.float64)

    result = match_fixed_a_measured_kl(
        model,
        setup,
        candidates,
        torch.ones(1, dtype=torch.float64),
        target_kl=0.01,
        train_node_scores=train_node_scores,
        relative_tolerance=0.01,
    )

    assert result.converged and result.applied
    assert result.initial_step_size == pytest.approx(0.02)
    assert result.applied_measured_kl == pytest.approx(0.01, rel=0.01)
    assert model.proj_lora_B.item() == pytest.approx(result.applied_step_size)
    # Reference logits are evaluated once at zero.  Doubling trials are 0.02,
    # 0.04, 0.08, 0.16—not cumulative 0.02, 0.06, 0.14, 0.30.
    assert model.forward_b_values[:5] == pytest.approx([0.0, 0.02, 0.04, 0.08, 0.16])
    assert sum(value == 0.0 for value in model.forward_b_values) == 1
    assert result.reference_forward_evaluations == 1
    assert (
        hf._fingerprint_named_tensors((("proj_lora_A", model.proj_lora_A),)) == setup.a_state_sha256
    )
    json.dumps(result.to_dict(), allow_nan=False)


def test_nonconverged_measured_kl_search_restores_zero_b() -> None:
    model, setup, candidates = _setup_and_candidates()

    result = match_fixed_a_measured_kl(
        model,
        setup,
        candidates,
        torch.ones(1, dtype=torch.float64),
        target_kl=0.01,
        initial_step=1.0e-3,
        max_iterations=2,
    )

    assert not result.converged
    assert not result.applied
    assert result.applied_step_size == 0.0
    assert result.applied_measured_kl == 0.0
    assert model.proj_lora_B.item() == 0.0


def test_measured_kl_forward_exception_restores_zero_b() -> None:
    model, setup, candidates = _setup_and_candidates(raise_on_nonzero=True)

    with pytest.raises(RuntimeError, match="synthetic trial failure"):
        match_fixed_a_measured_kl(
            model,
            setup,
            candidates,
            torch.ones(1, dtype=torch.float64),
            target_kl=0.01,
            initial_step=0.1,
        )

    assert model.proj_lora_B.item() == 0.0


def test_oracle_rollout_improvement_is_paired_mean_and_sample_se() -> None:
    reference = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    updated = torch.tensor([2.0, 2.0, 5.0], dtype=torch.float64)

    result = oracle_rollout_improvement(reference, updated)

    assert result.num_pairs == 3
    assert result.mean_difference == pytest.approx(1.0)
    assert result.sample_standard_error == pytest.approx(1.0 / math.sqrt(3.0))
    assert not result.significance_claimed
    json.dumps(result.to_dict(), allow_nan=False)


def test_oracle_rollout_improvement_rejects_unpaired_or_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="equal-length"):
        oracle_rollout_improvement(torch.ones(2), torch.ones(3))
    with pytest.raises(ValueError, match="finite"):
        oracle_rollout_improvement(
            torch.tensor([0.0, float("inf")]),
            torch.zeros(2),
        )
