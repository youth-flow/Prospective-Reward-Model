from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from smart_reward.hf import (
    build_oracle_chat,
    extract_scalar_oracle_logits,
    pool_final_response_hidden_state,
    validate_exact_generation_kwargs,
)


def test_exact_generation_contract_rejects_distribution_changes() -> None:
    result = validate_exact_generation_kwargs({"max_new_tokens": 8, "num_return_sequences": 6})
    assert result["temperature"] == 1.0
    assert result["top_p"] == 1.0
    with pytest.raises(ValueError, match="temperature"):
        validate_exact_generation_kwargs({"temperature": 0.8, "max_new_tokens": 8})
    with pytest.raises(ValueError, match="fail closed"):
        validate_exact_generation_kwargs({"custom_sampler": True})


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
