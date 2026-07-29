"""Train-only affine standardization of Skywork oracle scores."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

import torch

_MAD_NORMALIZATION = 1.482602218505602


def _tensor(value: torch.Tensor, name: str) -> None:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise TypeError(f"{name} must be a floating-point tensor")
    if value.numel() == 0 or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be non-empty and finite")


@dataclass(frozen=True, slots=True)
class AffineOracleTransform:
    b: float
    tau: float

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool) or not isinstance(value, Real) for value in (self.b, self.tau)
        ):
            raise TypeError("b and tau must be real scalars")
        if not math.isfinite(float(self.b)) or not math.isfinite(float(self.tau)) or self.tau <= 0:
            raise ValueError("b must be finite and tau must be finite and positive")

    def __call__(self, scores: torch.Tensor) -> torch.Tensor:
        _tensor(scores, "scores")
        return (scores - self.b) / self.tau


def fit_affine_oracle_transform(
    train_scores: torch.Tensor,
    *,
    scale_floor: float = 1.0e-6,
) -> AffineOracleTransform:
    _tensor(train_scores, "train_scores")
    floor = float(scale_floor)
    if not math.isfinite(floor) or floor <= 0.0:
        raise ValueError("scale_floor must be finite and positive")
    center = torch.median(train_scores.detach())
    mad = torch.median(torch.abs(train_scores.detach() - center))
    return AffineOracleTransform(
        b=float(center.item()),
        tau=max(floor, float((_MAD_NORMALIZATION * mad).item())),
    )


__all__ = ["AffineOracleTransform", "fit_affine_oracle_transform"]
