"""Named random streams for reproducible ProRM experiments."""

from __future__ import annotations

import hashlib
import random
from dataclasses import asdict, dataclass

import torch


def derive_seed(base_seed: int, namespace: str) -> int:
    """Derive a stable 63-bit seed without depending on Python's salted hash."""

    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or base_seed < 0:
        raise ValueError("base_seed must be a non-negative integer")
    if not isinstance(namespace, str) or not namespace.strip():
        raise ValueError("namespace must be a non-empty string")
    payload = f"prospective-reward-model/v1\0{base_seed}\0{namespace}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


@dataclass(frozen=True, slots=True)
class SeedBundle:
    """Independent streams shared by both objectives in one paired run."""

    candidate_generation: int
    policy_lora_a: int
    rollout: int

    @classmethod
    def from_base_seed(cls, base_seed: int) -> SeedBundle:
        return cls(**{field: derive_seed(base_seed, field) for field in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def seed_python_and_torch(seed: int, *, deterministic_algorithms: bool = False) -> None:
    """Seed local Python/PyTorch state; CUDA is seeded when available."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic_algorithms)


__all__ = ["SeedBundle", "derive_seed", "seed_python_and_torch"]
