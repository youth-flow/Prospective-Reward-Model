from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from smart_reward.hf import (
    build_oracle_chat,
    extract_scalar_oracle_logits,
    generate_exact_candidates,
    pool_final_response_hidden_state,
    score_exact_candidates,
    sequence_forward_kl,
    validate_exact_generation_kwargs,
)


class _RequiresGradTogglingPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.25))
        self.generation_config = SimpleNamespace(eos_token_id=4, pad_token_id=0)

    def generate(self, input_ids: torch.Tensor, **_: object) -> torch.Tensor:
        response = torch.full(
            (input_ids.shape[0], 1),
            2,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        return torch.cat((input_ids, response), dim=1)

    def forward(self, input_ids: torch.Tensor, **_: object) -> SimpleNamespace:
        logits = self.weight * torch.ones(
            *input_ids.shape,
            5,
            dtype=self.weight.dtype,
            device=input_ids.device,
        )
        return SimpleNamespace(logits=logits)

    @contextmanager
    def disable_adapter(self):
        self.weight.requires_grad_(False)
        try:
            yield
        finally:
            self.weight.requires_grad_(True)


def test_exact_generation_contract_rejects_distribution_changes() -> None:
    result = validate_exact_generation_kwargs({"max_new_tokens": 8, "num_return_sequences": 6})
    assert result["temperature"] == 1.0
    assert result["top_p"] == 1.0
    with pytest.raises(ValueError, match="temperature"):
        validate_exact_generation_kwargs({"temperature": 0.8, "max_new_tokens": 8})
    with pytest.raises(ValueError, match="fail closed"):
        validate_exact_generation_kwargs({"custom_sampler": True})


def test_policy_fingerprint_tracks_same_tensors_across_trainability_changes() -> None:
    model = _RequiresGradTogglingPolicy().eval()
    candidates = generate_exact_candidates(
        model,
        torch.tensor([[1, 3]]),
        generation_kwargs={"max_new_tokens": 1},
    )

    with model.disable_adapter():
        scores = score_exact_candidates(model, candidates)
    assert scores.shape == (1,)

    with torch.no_grad():
        model.weight.add_(1.0)
    with pytest.raises(ValueError, match="changed between generation and scoring"):
        score_exact_candidates(model, candidates)


def test_sequence_forward_kl_integrates_next_token_actions_exactly() -> None:
    updated = torch.tensor([[[2.0, 0.0], [0.0, 2.0], [9.0, -9.0]]])
    reference = torch.zeros_like(updated)
    response_mask = torch.tensor([[0, 1, 1]])

    result = sequence_forward_kl(updated, reference, response_mask)
    updated_log_prob = updated[:, :2].log_softmax(dim=-1, dtype=torch.float64)
    reference_log_prob = reference[:, :2].log_softmax(dim=-1, dtype=torch.float64)
    expected = (
        (updated_log_prob.exp() * (updated_log_prob - reference_log_prob)).sum(dim=-1).sum(dim=-1)
    )
    assert torch.allclose(result, expected)
    assert torch.equal(
        sequence_forward_kl(reference, reference, response_mask),
        torch.zeros(1, dtype=torch.float64),
    )


def test_sequence_forward_kl_is_stable_for_nearly_identical_policies() -> None:
    generator = torch.Generator().manual_seed(140)
    updated = torch.randn((1, 257, 1000), generator=generator, dtype=torch.float32)
    reference = updated + 3.0e-5 * torch.randn(
        updated.shape, generator=generator, dtype=torch.float32
    )
    response_mask = torch.ones((1, 257), dtype=torch.long)

    # The sign and magnitude of cancellation in the former float32 computation
    # depend on the backend kernel.  Test the portable contract instead: the
    # implementation performs the reduction in float64 and matches an
    # independently evaluated float64 expression.
    updated_log_prob_f64 = updated[:, :-1].log_softmax(dim=-1, dtype=torch.float64)
    reference_log_prob_f64 = reference[:, :-1].log_softmax(dim=-1, dtype=torch.float64)
    expected = (
        (updated_log_prob_f64.exp() * (updated_log_prob_f64 - reference_log_prob_f64))
        .sum(dim=-1)
        .sum(dim=-1)
    )
    result = sequence_forward_kl(updated, reference, response_mask)

    assert expected.item() > 0.0
    assert result.dtype == torch.float64
    assert result.item() >= 0.0
    assert torch.allclose(result, expected, rtol=1.0e-12, atol=1.0e-14)


def test_reward_feature_pooling_selects_final_response_token() -> None:
    hidden = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
    mask = torch.tensor([[0, 0, 1, 1, 0], [0, 1, 1, 1, 1]])
    pooled = pool_final_response_hidden_state((torch.zeros_like(hidden), hidden), mask)
    assert torch.equal(pooled, torch.stack((hidden[0, 3], hidden[1, 4])))


def test_reward_feature_pooling_rejects_disjoint_response_spans() -> None:
    with pytest.raises(ValueError, match="contiguous"):
        pool_final_response_hidden_state(
            torch.zeros(1, 4, 2),
            torch.tensor([[0, 1, 0, 1]]),
        )


def test_oracle_contract_uses_two_message_chat_and_scalar_logits() -> None:
    assert build_oracle_chat("prompt", "response") == [
        {"role": "user", "content": "prompt"},
        {"role": "assistant", "content": "response"},
    ]
    logits = extract_scalar_oracle_logits(SimpleNamespace(logits=torch.tensor([[1.0], [2.0]])))
    assert torch.equal(logits, torch.tensor([1.0, 2.0]))
