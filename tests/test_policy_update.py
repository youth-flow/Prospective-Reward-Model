import math

import pytest
import torch

from smart_reward.policy_update import (
    add_tangent_update_,
    fisher_quadratic,
    line_search_measured_kl,
    masked_causal_forward_kl,
    masked_causal_updated_to_reference_kl,
    masked_causal_updated_to_reference_kl_per_sequence,
    select_causal_response_logits,
    selected_causal_forward_kl,
    selected_causal_updated_to_reference_kl,
    selected_causal_updated_to_reference_kl_per_sequence,
    set_tangent_update_,
    step_size_for_kl_budget,
    unflatten_tangent_vector,
)
from smart_reward.scores import ParameterLayout


def _parameters():
    first = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
    second = torch.nn.Parameter(torch.zeros(1, 2, dtype=torch.float64))
    named = (("first_lora_B", first), ("second_lora_B", second))
    return named, ParameterLayout.from_named_parameters(named)


def test_unflatten_and_apply_update_respect_layout() -> None:
    named, layout = _parameters()
    direction = torch.tensor([1.0, -2.0, 3.0, 4.0], dtype=torch.float64)
    pieces = unflatten_tangent_vector(direction, layout)
    assert pieces[0].shape == (2,)
    assert pieces[1].shape == (1, 2)

    add_tangent_update_(named, layout, direction, step_size=0.25)
    torch.testing.assert_close(named[0][1], torch.tensor([0.25, -0.5], dtype=torch.float64))
    torch.testing.assert_close(named[1][1], torch.tensor([[0.75, 1.0]], dtype=torch.float64))
    with pytest.raises(ValueError, match="not at the reference"):
        add_tangent_update_(named, layout, direction, step_size=0.25)


def test_set_update_is_reference_based_and_never_accumulates_trials() -> None:
    named, layout = _parameters()
    direction = torch.tensor([1.0, -2.0, 3.0, 4.0], dtype=torch.float64)

    set_tangent_update_(named, layout, direction, step_size=0.25)
    set_tangent_update_(named, layout, direction, step_size=0.5)

    torch.testing.assert_close(named[0][1], torch.tensor([0.5, -1.0], dtype=torch.float64))
    torch.testing.assert_close(named[1][1], torch.tensor([[1.5, 2.0]], dtype=torch.float64))


def test_masked_causal_forward_kl_is_zero_at_reference_and_positive_after_update() -> None:
    reference = torch.tensor(
        [[[0.0, 1.0], [0.5, -0.5], [2.0, -1.0]]],
        dtype=torch.float64,
    )
    mask = torch.tensor([[0, 1, 1]], dtype=torch.bool)
    assert masked_causal_forward_kl(reference, reference.clone(), mask).item() == pytest.approx(
        0.0, abs=1.0e-15
    )

    updated = reference.clone()
    updated[:, :2, 0] += 0.7
    measured = masked_causal_forward_kl(reference, updated, mask)
    assert measured.item() > 0.0


def test_selected_chunked_kl_matches_direct_full_sequence_formula() -> None:
    generator = torch.Generator().manual_seed(17)
    reference = torch.randn(3, 7, 19, dtype=torch.float64, generator=generator)
    updated = reference + 0.2 * torch.randn(3, 7, 19, dtype=torch.float64, generator=generator)
    mask = torch.tensor(
        [
            [0, 0, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    shifted_mask = mask[:, 1:]
    reference_log_probs = reference[:, :-1].log_softmax(dim=-1)
    updated_log_probs = updated[:, :-1].log_softmax(dim=-1)
    direct_tokens = (reference_log_probs.exp() * (reference_log_probs - updated_log_probs)).sum(
        dim=-1
    )
    direct = direct_tokens.masked_fill(~shifted_mask, 0.0).sum(dim=-1).mean()

    selected_reference = select_causal_response_logits(reference, mask)
    selected_updated = select_causal_response_logits(updated, mask)
    assert selected_reference.logits.shape == (int(mask.sum().item()), 19)
    chunked = selected_causal_forward_kl(
        selected_reference,
        selected_updated,
        token_chunk_size=2,
    )
    torch.testing.assert_close(chunked, direct, rtol=1.0e-12, atol=1.0e-12)
    torch.testing.assert_close(
        masked_causal_forward_kl(reference, updated, mask, token_chunk_size=2),
        direct,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_updated_to_reference_kl_has_the_requested_asymmetric_direction() -> None:
    updated_probability = torch.tensor([0.8, 0.2], dtype=torch.float64)
    reference_probability = torch.tensor([0.5, 0.5], dtype=torch.float64)
    updated = torch.zeros(1, 2, 2, dtype=torch.float64)
    reference = torch.zeros_like(updated)
    updated[0, 0] = updated_probability.log()
    reference[0, 0] = reference_probability.log()
    mask = torch.tensor([[0, 1]], dtype=torch.bool)

    expected_updated_to_reference = (
        updated_probability * (updated_probability.log() - reference_probability.log())
    ).sum()
    expected_reference_to_updated = (
        reference_probability * (reference_probability.log() - updated_probability.log())
    ).sum()

    actual = masked_causal_updated_to_reference_kl(updated, reference, mask)
    locked_phase1_direction = masked_causal_forward_kl(reference, updated, mask)
    torch.testing.assert_close(actual, expected_updated_to_reference)
    torch.testing.assert_close(locked_phase1_direction, expected_reference_to_updated)
    assert actual.item() != pytest.approx(locked_phase1_direction.item())


def test_updated_to_reference_kl_sums_through_eos_then_averages_sequences() -> None:
    updated_probability = torch.tensor(
        [
            [0.8, 0.6, 0.999, 0.3],
            [0.999, 0.9, 0.001, 0.3],
        ],
        dtype=torch.float64,
    )
    reference_probability = torch.tensor(
        [
            [0.5, 0.5, 0.001, 0.7],
            [0.001, 0.4, 0.999, 0.7],
        ],
        dtype=torch.float64,
    )
    updated = torch.stack(
        (updated_probability.log(), (1.0 - updated_probability).log()),
        dim=-1,
    )
    reference = torch.stack(
        (reference_probability.log(), (1.0 - reference_probability).log()),
        dim=-1,
    )
    # Sequence 0 has two response predictions: the second is the EOS token.
    # Sequence 1 has one response prediction.  The other large logit
    # differences are deliberately unselected and must not affect the result.
    mask = torch.tensor(
        [
            [0, 1, 1, 0],
            [0, 0, 1, 0],
        ],
        dtype=torch.bool,
    )

    def bernoulli_kl(probability: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return probability * (probability.log() - target.log()) + (1.0 - probability) * (
            (1.0 - probability).log() - (1.0 - target).log()
        )

    expected_per_sequence = torch.stack(
        (
            bernoulli_kl(updated_probability[0, 0], reference_probability[0, 0])
            + bernoulli_kl(updated_probability[0, 1], reference_probability[0, 1]),
            bernoulli_kl(updated_probability[1, 1], reference_probability[1, 1]),
        )
    )
    actual_per_sequence = masked_causal_updated_to_reference_kl_per_sequence(
        updated,
        reference,
        mask,
        token_chunk_size=1,
    )
    torch.testing.assert_close(actual_per_sequence, expected_per_sequence)
    torch.testing.assert_close(
        masked_causal_updated_to_reference_kl(
            updated,
            reference,
            mask,
            token_chunk_size=1,
        ),
        expected_per_sequence.mean(),
    )

    selected_updated = select_causal_response_logits(updated, mask)
    selected_reference = select_causal_response_logits(reference, mask)
    torch.testing.assert_close(
        selected_causal_updated_to_reference_kl_per_sequence(
            selected_updated,
            selected_reference,
            token_chunk_size=1,
        ),
        expected_per_sequence,
    )
    torch.testing.assert_close(
        selected_causal_updated_to_reference_kl(
            selected_updated,
            selected_reference,
            token_chunk_size=1,
        ),
        expected_per_sequence.mean(),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_updated_to_reference_kl_preserves_compute_dtype_and_device(
    dtype: torch.dtype,
) -> None:
    updated = torch.tensor(
        [[[0.3, -0.2], [0.0, 0.0]]],
        dtype=dtype,
    )
    reference = torch.tensor(
        [[[-0.1, 0.4], [0.0, 0.0]]],
        dtype=dtype,
    )
    mask = torch.tensor([[0, 1]], dtype=torch.bool, device=updated.device)

    per_sequence = masked_causal_updated_to_reference_kl_per_sequence(
        updated,
        reference,
        mask,
    )
    assert per_sequence.shape == (1,)
    assert per_sequence.dtype == dtype
    assert per_sequence.device == updated.device


def test_updated_to_reference_kl_rejects_invalid_inputs() -> None:
    logits = torch.zeros(1, 2, 3)
    mask = torch.tensor([[0, 1]], dtype=torch.bool)

    with pytest.raises(TypeError, match="torch.Tensor"):
        masked_causal_updated_to_reference_kl([], logits, mask)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="share shape"):
        masked_causal_updated_to_reference_kl(logits, torch.zeros(1, 3, 3), mask)
    with pytest.raises(ValueError, match="binary"):
        masked_causal_updated_to_reference_kl(
            logits,
            logits,
            torch.tensor([[0, 2]]),
        )
    with pytest.raises(ValueError, match="at least one"):
        masked_causal_updated_to_reference_kl(
            logits,
            logits,
            torch.zeros_like(mask),
        )
    with pytest.raises(ValueError, match="positive integer"):
        masked_causal_updated_to_reference_kl(
            logits,
            logits,
            mask,
            token_chunk_size=True,
        )
    nonfinite = logits.clone()
    nonfinite[0, 0, 0] = torch.inf
    with pytest.raises(ValueError, match="finite"):
        masked_causal_updated_to_reference_kl(nonfinite, logits, mask)


def test_kl_budget_scaling_matches_quadratic_target() -> None:
    matrix = torch.tensor([[2.0, 0.5], [0.5, 1.0]], dtype=torch.float64)
    direction = torch.tensor([1.0, -0.5], dtype=torch.float64)
    budget = 0.01
    curvature = fisher_quadratic(direction, lambda vector: matrix @ vector)
    step = step_size_for_kl_budget(
        direction,
        lambda vector: matrix @ vector,
        kl_budget=budget,
    )
    assert 0.5 * step**2 * curvature.item() == pytest.approx(budget)
    assert step == pytest.approx(math.sqrt(2.0 * budget / curvature.item()))


def test_null_direction_cannot_spend_positive_budget() -> None:
    direction = torch.ones(2)
    with pytest.raises(ValueError, match="Fisher-null"):
        step_size_for_kl_budget(direction, lambda vector: torch.zeros_like(vector), kl_budget=0.01)


def test_measured_kl_line_search_hits_target_without_using_quadratic_formula() -> None:
    target = 0.01
    result = line_search_measured_kl(
        lambda step: 0.4 * step**2 + 0.1 * step**4,
        target_kl=target,
        initial_step=1.0,
        relative_tolerance=1.0e-5,
    )
    assert result.converged
    assert result.measured_kl == pytest.approx(target, rel=1.0e-5)
    assert result.iterations <= 30


def test_measured_kl_line_search_rejects_nonzero_reference() -> None:
    with pytest.raises(ValueError, match=r"measure_kl\(0\)"):
        line_search_measured_kl(
            lambda step: 0.1 + step**2,
            target_kl=0.01,
            initial_step=0.1,
        )
