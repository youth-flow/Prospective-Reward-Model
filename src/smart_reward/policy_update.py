"""Write a solved natural-gradient direction into fixed LoRA-B coordinates."""

from __future__ import annotations

import math
from collections.abc import Iterable

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


__all__ = ["set_tangent_update_", "unflatten_tangent_vector"]
