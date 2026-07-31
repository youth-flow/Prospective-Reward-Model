"""Write a solved natural-gradient direction into fixed LoRA-B coordinates."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable

import torch

from .scores import NamedParameter, ParameterLayout


def unflatten_tangent_vector(
    vector: torch.Tensor,
    layout: ParameterLayout,
) -> tuple[torch.Tensor, ...]:
    if not isinstance(vector, torch.Tensor) or vector.ndim != 1:
        raise TypeError("vector must be one-dimensional")
    if not vector.is_floating_point() or not bool(torch.isfinite(vector).all()):
        raise ValueError("vector must be finite and floating point")
    if vector.numel() != layout.dimension:
        raise ValueError(f"vector must have length {layout.dimension}")
    return tuple(
        vector[entry.offset : entry.offset + entry.numel].reshape(entry.shape)
        for entry in layout.entries
    )


def scale_direction_to_quadratic_kl(
    direction: torch.Tensor,
    fisher_matvec: Callable[[torch.Tensor], torch.Tensor],
    *,
    kl_target: float,
) -> tuple[torch.Tensor, float, float]:
    """Scale a direction to ``0.5 * delta.T F delta == kl_target``.

    ``F`` is the raw, undamped empirical Fisher.  Damping determines the
    direction but is deliberately excluded from the trust-region constraint.
    """

    if not isinstance(direction, torch.Tensor) or direction.ndim != 1:
        raise TypeError("direction must be one-dimensional")
    if not direction.is_floating_point() or not bool(torch.isfinite(direction).all()):
        raise ValueError("direction must be finite and floating point")
    target = float(kl_target)
    if not math.isfinite(target) or target <= 0.0:
        raise ValueError("kl_target must be finite and positive")
    fisher_direction = fisher_matvec(direction)
    if (
        not isinstance(fisher_direction, torch.Tensor)
        or fisher_direction.shape != direction.shape
        or not bool(torch.isfinite(fisher_direction).all())
    ):
        raise ValueError("fisher_matvec returned an invalid vector")
    curvature = float(torch.dot(direction, fisher_direction).item())
    if not math.isfinite(curvature) or curvature <= 0.0:
        raise ValueError("direction must have positive finite raw-Fisher curvature")
    scale = math.sqrt(2.0 * target / curvature)
    update = direction * scale
    realized = 0.5 * float(torch.dot(update, fisher_matvec(update)).item())
    tolerance = max(1.0e-12, 1.0e-10 * target)
    if not math.isfinite(realized) or abs(realized - target) > tolerance:
        raise RuntimeError("quadratic KL scaling failed its numerical identity gate")
    return update, scale, realized


@torch.no_grad()
def set_tangent_update_(
    named_parameters: Iterable[NamedParameter],
    layout: ParameterLayout,
    direction: torch.Tensor,
    *,
    step_size: float,
) -> None:
    """Set LoRA-B to ``step_size * direction`` from the reference origin."""

    values = tuple(named_parameters)
    layout.validate_named_parameters(values)
    step = float(step_size)
    if not math.isfinite(step) or step < 0.0:
        raise ValueError("step_size must be finite and non-negative")
    pieces = unflatten_tangent_vector(direction, layout)
    for (name, parameter), piece in zip(values, pieces, strict=True):
        if parameter.device != direction.device:
            raise ValueError(f"parameter {name!r} and direction must share a device")
        parameter.copy_(piece.to(dtype=parameter.dtype))
        parameter.mul_(step)


__all__ = [
    "scale_direction_to_quadratic_kl",
    "set_tangent_update_",
    "unflatten_tangent_vector",
]
