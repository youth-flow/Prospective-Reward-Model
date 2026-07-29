"""Matrix-free empirical Fisher linear algebra."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Literal

import torch

FisherSolveDType = Literal["float64"]


def resolve_fisher_solve_dtype(value: str) -> torch.dtype:
    """Resolve the locked high-accuracy Fisher solve dtype."""

    if value != "float64":
        raise ValueError("pcg_dtype must be 'float64'")
    return torch.float64


def _validate_score_matrix(score_matrix: torch.Tensor) -> tuple[int, int]:
    if not isinstance(score_matrix, torch.Tensor):
        raise TypeError("score_matrix must be a torch.Tensor")
    if score_matrix.ndim != 2:
        raise ValueError("score_matrix must have shape (num_samples, num_parameters)")
    if not score_matrix.is_floating_point():
        raise TypeError("score_matrix must have a floating-point dtype")
    num_samples, num_parameters = score_matrix.shape
    if num_samples < 1 or num_parameters < 1:
        raise ValueError("score_matrix dimensions must both be positive")
    if not bool(torch.isfinite(score_matrix).all()):
        raise ValueError("score_matrix must be finite")
    return num_samples, num_parameters


def _validate_damping(damping: float) -> float:
    value = float(damping)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("damping must be finite and non-negative")
    return value


def _validate_vector(
    vector: torch.Tensor,
    score_matrix: torch.Tensor,
    num_parameters: int,
) -> None:
    if not isinstance(vector, torch.Tensor):
        raise TypeError("vector must be a torch.Tensor")
    if vector.ndim != 1 or vector.numel() != num_parameters:
        raise ValueError(f"vector must have shape ({num_parameters},)")
    if vector.device != score_matrix.device or vector.dtype != score_matrix.dtype:
        raise ValueError("vector and score_matrix must have the same dtype and device")
    if not bool(torch.isfinite(vector).all()):
        raise ValueError("vector must be finite")


def damped_fisher_matvec(
    vector: torch.Tensor,
    score_matrix: torch.Tensor,
    damping: float = 0.0,
) -> torch.Tensor:
    """Apply ``S.T @ S / n + damping * I`` without forming the matrix.

    The normalization matches the empirical Fisher convention
    ``F_hat = S.T S / n`` used throughout the ProRM objective.
    """

    num_samples, num_parameters = _validate_score_matrix(score_matrix)
    damping_value = _validate_damping(damping)
    _validate_vector(vector, score_matrix, num_parameters)

    fisher_product = score_matrix.mT @ (score_matrix @ vector)
    return fisher_product / num_samples + damping_value * vector


def damped_fisher_diagonal(
    score_matrix: torch.Tensor,
    damping: float = 0.0,
) -> torch.Tensor:
    """Return ``diag(S.T S / n + damping * I)`` exactly."""

    num_samples, _ = _validate_score_matrix(score_matrix)
    damping_value = _validate_damping(damping)
    return score_matrix.square().sum(dim=0) / num_samples + damping_value


# Concise aliases used by training code and in the derivation.
fisher_matvec: Final = damped_fisher_matvec
fisher_diagonal: Final = damped_fisher_diagonal


@dataclass(frozen=True)
class DampedEmpiricalFisher:
    """A lightweight callable representation of a damped empirical Fisher."""

    score_matrix: torch.Tensor
    damping: float = 0.0

    def __post_init__(self) -> None:
        _validate_score_matrix(self.score_matrix)
        _validate_damping(self.damping)

    @property
    def dimension(self) -> int:
        """Number of tangent parameters."""

        return self.score_matrix.shape[1]

    def matvec(self, vector: torch.Tensor) -> torch.Tensor:
        """Apply the operator without rescanning the fixed score matrix."""

        _validate_vector(vector, self.score_matrix, self.dimension)
        product = self.score_matrix.mT @ (self.score_matrix @ vector)
        return product / self.score_matrix.shape[0] + self.damping * vector

    def diagonal(self) -> torch.Tensor:
        """Return the represented operator's diagonal."""

        return self.score_matrix.square().mean(dim=0) + self.damping

    def inverse_diagonal(self) -> torch.Tensor:
        """Return the Jacobi inverse diagonal, requiring strict positivity."""

        diagonal = self.diagonal()
        if bool((diagonal <= 0.0).any()):
            raise ValueError(
                "the Jacobi preconditioner is not positive; use positive damping "
                "or remove structurally zero score columns"
            )
        return diagonal.reciprocal()

    def pcg_inverse_diagonal(self) -> torch.Tensor | None:
        """Choose Jacobi only when it does not destroy low-rank ridge structure.

        When the score matrix has fewer rows than columns, the operator is a
        low-rank Gram matrix plus ``damping * I``. Unpreconditioned CG then has
        at most ``num_rows + 1`` distinct eigenvalues in exact arithmetic,
        while coordinate-wise Jacobi scaling can turn it into a full-spectrum
        problem. Once the empirical Fisher can be full rank, Jacobi is the
        useful default.
        """

        if self.score_matrix.shape[0] < self.dimension:
            return None
        return self.inverse_diagonal()


__all__ = [
    "DampedEmpiricalFisher",
    "FisherSolveDType",
    "damped_fisher_diagonal",
    "damped_fisher_matvec",
    "fisher_diagonal",
    "fisher_matvec",
    "resolve_fisher_solve_dtype",
]
