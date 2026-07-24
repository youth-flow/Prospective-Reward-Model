"""Safe utilities for writing a local direction back into fixed-A LoRA-B."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch

from .scores import NamedParameter, ParameterLayout


@dataclass(frozen=True, slots=True)
class KLLineSearchResult:
    """Result of a monotone measured-KL bisection."""

    step_size: float
    measured_kl: float
    iterations: int
    converged: bool


def unflatten_tangent_vector(
    vector: torch.Tensor,
    layout: ParameterLayout,
) -> tuple[torch.Tensor, ...]:
    """View one flat tangent vector using the saved canonical layout."""

    if not isinstance(vector, torch.Tensor) or vector.ndim != 1:
        raise TypeError("vector must be a one-dimensional torch.Tensor")
    if not vector.is_floating_point() or not bool(torch.isfinite(vector).all()):
        raise ValueError("vector must be finite and floating point")
    if not isinstance(layout, ParameterLayout):
        raise TypeError("layout must be a ParameterLayout")
    if vector.numel() != layout.dimension:
        raise ValueError(f"vector must have length {layout.dimension}")
    return tuple(
        vector[entry.offset : entry.offset + entry.numel].reshape(entry.shape)
        for entry in layout.entries
    )


@torch.no_grad()
def add_tangent_update_(
    named_parameters: Iterable[NamedParameter],
    layout: ParameterLayout,
    direction: torch.Tensor,
    *,
    step_size: float,
    require_zero_base: bool = True,
) -> None:
    """Apply ``theta <- theta + step_size * direction`` in layout order."""

    values = tuple(named_parameters)
    layout.validate_named_parameters(values)
    step = float(step_size)
    if not math.isfinite(step) or step < 0.0:
        raise ValueError("step_size must be finite and non-negative")
    pieces = unflatten_tangent_vector(direction, layout)
    for (name, parameter), piece in zip(values, pieces, strict=True):
        if parameter.device != direction.device:
            raise ValueError(f"parameter {name!r} and direction must share a device")
        if require_zero_base and not bool(torch.count_nonzero(parameter) == 0):
            raise ValueError(f"parameter {name!r} is not at the reference zero LoRA-B state")
        parameter.add_(piece.to(dtype=parameter.dtype), alpha=step)


@torch.no_grad()
def set_tangent_update_(
    named_parameters: Iterable[NamedParameter],
    layout: ParameterLayout,
    direction: torch.Tensor,
    *,
    step_size: float,
) -> None:
    """Set LoRA-B to ``step_size * direction`` from the saved reference origin.

    Unlike :func:`add_tangent_update_`, this operation is idempotent.  It is the
    required primitive for measured-KL line-search callbacks, where every
    trial must start from the same zero-B reference rather than accumulating
    prior trial steps.
    """

    values = tuple(named_parameters)
    layout.validate_named_parameters(values)
    step = float(step_size)
    if not math.isfinite(step) or step < 0.0:
        raise ValueError("step_size must be finite and non-negative")
    pieces = unflatten_tangent_vector(direction, layout)
    for (name, parameter), piece in zip(values, pieces, strict=True):
        if parameter.device != direction.device:
            raise ValueError(f"parameter {name!r} and direction must share a device")
        parameter.copy_(piece.to(dtype=parameter.dtype), non_blocking=False)
        parameter.mul_(step)


@dataclass(frozen=True, slots=True)
class SelectedResponseLogits:
    """Only causal prediction rows whose next token is in the response."""

    logits: torch.Tensor
    sequence_indices: torch.Tensor
    batch_size: int

    def __post_init__(self) -> None:
        if not isinstance(self.logits, torch.Tensor) or self.logits.ndim != 2:
            raise TypeError("logits must have shape (selected_tokens, vocabulary_size)")
        if min(self.logits.shape) < 1 or not self.logits.is_floating_point():
            raise ValueError("selected logits must be non-empty and floating point")
        if (
            not isinstance(self.sequence_indices, torch.Tensor)
            or self.sequence_indices.shape != (self.logits.shape[0],)
            or self.sequence_indices.dtype != torch.int64
            or self.sequence_indices.device != self.logits.device
        ):
            raise ValueError("sequence_indices must be int64 and match selected token rows")
        if (
            isinstance(self.batch_size, bool)
            or not isinstance(self.batch_size, int)
            or self.batch_size < 1
        ):
            raise ValueError("batch_size must be a positive integer")
        if bool((self.sequence_indices < 0).any()) or bool(
            (self.sequence_indices >= self.batch_size).any()
        ):
            raise ValueError("sequence_indices are outside the declared batch")
        counts = torch.bincount(self.sequence_indices, minlength=self.batch_size)
        if bool((counts < 1).any()):
            raise ValueError("every sequence must select at least one predicted response token")


def select_causal_response_logits(
    logits: torch.Tensor,
    response_mask: torch.Tensor,
) -> SelectedResponseLogits:
    """Copy only logits that predict active response tokens.

    The returned tensor is typically much smaller than ``B x L x V``.  It is
    an owning advanced-indexing result, so callers may immediately release the
    full model output before computing KL.
    """

    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise TypeError("logits must have shape (batch, sequence_length, vocabulary_size)")
    if min(logits.shape) < 1 or not logits.is_floating_point():
        raise ValueError("logits must be non-empty and floating point")
    if not isinstance(response_mask, torch.Tensor) or response_mask.shape != logits.shape[:2]:
        raise ValueError("response_mask must match the logits batch and sequence axes")
    if response_mask.device != logits.device:
        raise ValueError("response_mask and logits must share a device")
    if not bool(((response_mask == 0) | (response_mask == 1)).all()):
        raise ValueError("response_mask must be binary")
    shifted_mask = response_mask[:, 1:].to(torch.bool)
    if bool((shifted_mask.sum(dim=1) < 1).any()):
        raise ValueError("every sequence must select at least one predicted response token")
    batch_size = logits.shape[0]
    sequence_indices = (
        torch.arange(batch_size, device=logits.device)
        .unsqueeze(1)
        .expand_as(shifted_mask)[shifted_mask]
        .to(torch.int64)
    )
    selected = logits[:, :-1][shifted_mask].contiguous()
    return SelectedResponseLogits(selected, sequence_indices, batch_size)


def _selected_causal_kl_per_sequence(
    numerator: SelectedResponseLogits,
    denominator: SelectedResponseLogits,
    *,
    token_chunk_size: int = 8,
) -> torch.Tensor:
    """Compute per-sequence exact KL in small chunks of selected token rows."""

    if not isinstance(numerator, SelectedResponseLogits) or not isinstance(
        denominator, SelectedResponseLogits
    ):
        raise TypeError("numerator and denominator must be SelectedResponseLogits")
    if numerator.logits.shape != denominator.logits.shape:
        raise ValueError("numerator and denominator selected logits must share shape")
    if numerator.logits.device != denominator.logits.device:
        raise ValueError("numerator and denominator selected logits must share a device")
    if numerator.batch_size != denominator.batch_size or not torch.equal(
        numerator.sequence_indices, denominator.sequence_indices
    ):
        raise ValueError("numerator and denominator response positions must be identical")
    if (
        isinstance(token_chunk_size, bool)
        or not isinstance(token_chunk_size, int)
        or token_chunk_size < 1
    ):
        raise ValueError("token_chunk_size must be a positive integer")

    compute_dtype = (
        torch.float64
        if numerator.logits.dtype == torch.float64 and denominator.logits.dtype == torch.float64
        else torch.float32
    )
    sequence_kl = torch.zeros(
        numerator.batch_size,
        dtype=compute_dtype,
        device=numerator.logits.device,
    )
    for start in range(0, numerator.logits.shape[0], token_chunk_size):
        stop = min(start + token_chunk_size, numerator.logits.shape[0])
        numerator_chunk = numerator.logits[start:stop].to(compute_dtype)
        denominator_chunk = denominator.logits[start:stop].to(compute_dtype)
        if not bool(torch.isfinite(numerator_chunk).all()) or not bool(
            torch.isfinite(denominator_chunk).all()
        ):
            raise ValueError("selected logits must be finite")
        numerator_log_probs = numerator_chunk.log_softmax(dim=-1)
        denominator_log_probs = denominator_chunk.log_softmax(dim=-1)
        token_kl = (numerator_log_probs.exp() * (numerator_log_probs - denominator_log_probs)).sum(
            dim=-1
        )
        sequence_kl.index_add_(
            0,
            numerator.sequence_indices[start:stop],
            token_kl,
        )
    tolerance = 64.0 * torch.finfo(sequence_kl.dtype).eps
    if float(sequence_kl.min().item()) < -tolerance:
        raise FloatingPointError("measured forward KL is numerically negative")
    return sequence_kl.clamp_min(0.0)


def selected_causal_forward_kl(
    reference: SelectedResponseLogits,
    updated: SelectedResponseLogits,
    *,
    token_chunk_size: int = 8,
) -> torch.Tensor:
    """Compute mean sequence ``KL(pi_reference || pi_updated)``."""

    return _selected_causal_kl_per_sequence(
        reference,
        updated,
        token_chunk_size=token_chunk_size,
    ).mean()


def selected_causal_updated_to_reference_kl_per_sequence(
    updated: SelectedResponseLogits,
    reference: SelectedResponseLogits,
    *,
    token_chunk_size: int = 8,
) -> torch.Tensor:
    """Return one ``KL(pi_updated || pi_reference)`` sum per sequence.

    The vector has shape ``(batch_size,)`` and retains the logits' device.  It
    is suitable for prompt-level reward-minus-beta-KL aggregation.  It is an
    on-policy Monte Carlo quantity only when histories come from ``updated``.
    """

    return _selected_causal_kl_per_sequence(
        updated,
        reference,
        token_chunk_size=token_chunk_size,
    )


def selected_causal_updated_to_reference_kl(
    updated: SelectedResponseLogits,
    reference: SelectedResponseLogits,
    *,
    token_chunk_size: int = 8,
) -> torch.Tensor:
    """Compute ``KL(pi_updated || pi_reference)`` on selected histories.

    The token KLs are summed within each sequence and the sequence totals are
    averaged across the batch.  This is an on-policy Monte Carlo estimate of
    ``KL(pi_updated || pi_reference)`` only when the selected histories were
    sampled from ``pi_updated``.  The source of those histories cannot be
    inferred from logits and is therefore an explicit caller responsibility.
    """

    return selected_causal_updated_to_reference_kl_per_sequence(
        updated,
        reference,
        token_chunk_size=token_chunk_size,
    ).mean()


def masked_causal_forward_kl(
    reference_logits: torch.Tensor,
    updated_logits: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    token_chunk_size: int = 8,
) -> torch.Tensor:
    """Estimate sequence KL on reference response histories.

    At every selected causal prediction position this computes the exact
    vocabulary KL ``KL(pi_0(.|history) || pi_delta(.|history))``.  Token KLs
    are summed per response (including the EOS prediction) and then averaged
    over reference-policy sequences.  This is a measured, non-negative
    forward-KL estimate; it is not the Fisher quadratic approximation.
    """

    if not isinstance(reference_logits, torch.Tensor) or not isinstance(
        updated_logits, torch.Tensor
    ):
        raise TypeError("logits must be torch.Tensor objects")
    if reference_logits.shape != updated_logits.shape:
        raise ValueError("logits must share shape (batch, sequence_length, vocabulary_size)")
    reference = select_causal_response_logits(reference_logits, response_mask)
    updated = select_causal_response_logits(updated_logits, response_mask)
    return selected_causal_forward_kl(
        reference,
        updated,
        token_chunk_size=token_chunk_size,
    )


def masked_causal_updated_to_reference_kl(
    updated_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    token_chunk_size: int = 8,
) -> torch.Tensor:
    """Compute sequence ``KL(pi_updated || pi_reference)`` on given histories.

    ``response_mask`` marks response tokens, including EOS when EOS is part of
    the sampled response.  At every corresponding causal prediction position,
    this computes the exact vocabulary KL from the updated distribution to the
    reference distribution.  Token KLs are summed per sequence and sequence
    totals are averaged over the batch.

    To estimate the on-policy divergence
    ``E_{y ~ pi_updated} sum_t KL(pi_updated(.|h_t) || pi_reference(.|h_t))``,
    callers must supply histories sampled from ``pi_updated`` and evaluate both
    policies on those same histories.  Histories sampled from the reference
    policy do not estimate that quantity.
    """

    return masked_causal_updated_to_reference_kl_per_sequence(
        updated_logits,
        reference_logits,
        response_mask,
        token_chunk_size=token_chunk_size,
    ).mean()


def masked_causal_updated_to_reference_kl_per_sequence(
    updated_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    token_chunk_size: int = 8,
) -> torch.Tensor:
    """Return per-sequence updated-to-reference KL on the supplied histories.

    Histories must be sampled from the updated policy for this vector to be an
    on-policy Monte Carlo estimate.  Both policies must be evaluated on those
    exact same token histories.
    """

    if not isinstance(updated_logits, torch.Tensor) or not isinstance(
        reference_logits, torch.Tensor
    ):
        raise TypeError("logits must be torch.Tensor objects")
    if updated_logits.shape != reference_logits.shape:
        raise ValueError("logits must share shape (batch, sequence_length, vocabulary_size)")
    updated = select_causal_response_logits(updated_logits, response_mask)
    reference = select_causal_response_logits(reference_logits, response_mask)
    return selected_causal_updated_to_reference_kl_per_sequence(
        updated,
        reference,
        token_chunk_size=token_chunk_size,
    )


def fisher_quadratic(
    direction: torch.Tensor,
    fisher_operator: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Return ``direction.T @ F @ direction`` with strict PSD checks."""

    if not isinstance(direction, torch.Tensor) or direction.ndim != 1:
        raise TypeError("direction must be a one-dimensional torch.Tensor")
    if not direction.is_floating_point() or not bool(torch.isfinite(direction).all()):
        raise ValueError("direction must be finite and floating point")
    if not callable(fisher_operator):
        raise TypeError("fisher_operator must be callable")
    product = fisher_operator(direction)
    if not isinstance(product, torch.Tensor) or product.shape != direction.shape:
        raise ValueError("fisher_operator returned an incompatible tensor")
    if product.dtype != direction.dtype or product.device != direction.device:
        raise ValueError("fisher_operator output must match direction dtype and device")
    if not bool(torch.isfinite(product).all()):
        raise ValueError("fisher_operator output must be finite")
    curvature = torch.dot(direction, product)
    tolerance = (
        32.0
        * torch.finfo(curvature.dtype).eps
        * max(1.0, torch.linalg.vector_norm(direction).square().item())
    )
    if curvature.item() < -tolerance:
        raise ValueError("Fisher curvature is negative")
    return curvature.clamp_min(0.0)


def step_size_for_kl_budget(
    direction: torch.Tensor,
    fisher_operator: Callable[[torch.Tensor], torch.Tensor],
    *,
    kl_budget: float,
) -> float:
    """Choose ``eta`` so ``eta^2 direction.T F direction / 2 = kl_budget``."""

    budget = float(kl_budget)
    if not math.isfinite(budget) or budget <= 0.0:
        raise ValueError("kl_budget must be finite and strictly positive")
    curvature = float(fisher_quadratic(direction, fisher_operator).item())
    if curvature == 0.0:
        raise ValueError("cannot scale a Fisher-null direction to a positive KL budget")
    return math.sqrt(2.0 * budget / curvature)


def line_search_measured_kl(
    measure_kl: Callable[[float], float],
    *,
    target_kl: float,
    initial_step: float,
    relative_tolerance: float = 0.05,
    max_iterations: int = 30,
) -> KLLineSearchResult:
    """Bisect a non-negative step to match an actually measured KL.

    ``measure_kl(step)`` must evaluate from the same reference parameters on
    every call.  The routine first brackets the target by doubling or halving
    the initial step, then bisects.  It never substitutes quadratic KL for the
    measured value; the quadratic step is only a suitable initializer.
    """

    if not callable(measure_kl):
        raise TypeError("measure_kl must be callable")
    target = float(target_kl)
    step = float(initial_step)
    tolerance = float(relative_tolerance)
    if not math.isfinite(target) or target <= 0.0:
        raise ValueError("target_kl must be finite and strictly positive")
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("initial_step must be finite and strictly positive")
    if not math.isfinite(tolerance) or not 0.0 < tolerance < 1.0:
        raise ValueError("relative_tolerance must lie in (0, 1)")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        raise TypeError("max_iterations must be an integer")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")

    evaluations = 0

    def checked_measure(candidate_step: float) -> float:
        nonlocal evaluations
        value = float(measure_kl(candidate_step))
        evaluations += 1
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("measure_kl must return a finite non-negative value")
        return value

    zero_kl = checked_measure(0.0)
    if zero_kl > max(1.0e-12, tolerance * target):
        raise ValueError("measure_kl(0) must be zero relative to the reference policy")

    measured = checked_measure(step)
    lower_step, lower_kl = 0.0, zero_kl
    upper_step, upper_kl = step, measured
    while upper_kl < target and evaluations < max_iterations:
        lower_step, lower_kl = upper_step, upper_kl
        upper_step *= 2.0
        upper_kl = checked_measure(upper_step)

    if upper_kl < target:
        return KLLineSearchResult(upper_step, upper_kl, evaluations, False)

    best_step, best_kl = min(
        ((lower_step, lower_kl), (upper_step, upper_kl)),
        key=lambda item: abs(item[1] - target),
    )
    while evaluations < max_iterations:
        candidate_step = 0.5 * (lower_step + upper_step)
        candidate_kl = checked_measure(candidate_step)
        if abs(candidate_kl - target) < abs(best_kl - target):
            best_step, best_kl = candidate_step, candidate_kl
        if abs(candidate_kl - target) / target <= tolerance:
            return KLLineSearchResult(candidate_step, candidate_kl, evaluations, True)
        if candidate_kl < target:
            lower_step, lower_kl = candidate_step, candidate_kl
        else:
            upper_step, upper_kl = candidate_step, candidate_kl
    return KLLineSearchResult(best_step, best_kl, evaluations, False)


__all__ = [
    "KLLineSearchResult",
    "SelectedResponseLogits",
    "add_tangent_update_",
    "fisher_quadratic",
    "line_search_measured_kl",
    "masked_causal_forward_kl",
    "masked_causal_updated_to_reference_kl",
    "masked_causal_updated_to_reference_kl_per_sequence",
    "select_causal_response_logits",
    "selected_causal_forward_kl",
    "selected_causal_updated_to_reference_kl",
    "selected_causal_updated_to_reference_kl_per_sequence",
    "set_tangent_update_",
    "step_size_for_kl_budget",
    "unflatten_tangent_vector",
]
