"""Serialized node records for the reference-policy candidate pool."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

NODE_SCHEMA = "candidate-node/v2"


class SchemaError(ValueError):
    """A JSON record does not satisfy its exact schema."""


def _integer_tuple(value: object, name: str, *, binary: bool) -> tuple[int, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be an integer sequence")
    result = tuple(value)
    for item in result:
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError(f"{name} must contain integers")
        if item < 0 or (binary and item not in {0, 1}):
            raise ValueError(f"{name} contains an invalid value")
    return result


@dataclass(frozen=True, slots=True, kw_only=True)
class CandidateNode:
    prompt_id: str
    candidate_id: str
    candidate_index: int
    split: str
    prompt: str
    response: str
    raw_oracle_score: float
    oracle_reward: float
    token_ids: tuple[int, ...]
    response_mask: tuple[int, ...]
    terminated_by_eos: bool
    reached_max_length: bool
    schema_version: str = NODE_SCHEMA

    def __post_init__(self) -> None:
        for name in ("prompt_id", "candidate_id", "prompt"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.response, str):
            raise TypeError("response must be a string")
        if (
            isinstance(self.candidate_index, bool)
            or not isinstance(self.candidate_index, int)
            or self.candidate_index < 0
        ):
            raise ValueError("candidate_index must be a non-negative integer")
        if self.split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, or test")
        for name in ("raw_oracle_score", "oracle_reward"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        token_ids = _integer_tuple(self.token_ids, "token_ids", binary=False)
        response_mask = _integer_tuple(self.response_mask, "response_mask", binary=True)
        if not token_ids or len(token_ids) != len(response_mask) or not any(response_mask):
            raise ValueError("token_ids and response_mask must align and select a response")
        object.__setattr__(self, "token_ids", token_ids)
        object.__setattr__(self, "response_mask", response_mask)
        if not isinstance(self.terminated_by_eos, bool) or not isinstance(
            self.reached_max_length, bool
        ):
            raise TypeError("termination flags must be boolean")
        if self.terminated_by_eos and self.reached_max_length:
            raise ValueError("a response cannot terminate by EOS and the length limit")
        if self.schema_version != NODE_SCHEMA:
            raise SchemaError(f"schema_version must equal {NODE_SCHEMA!r}")

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> CandidateNode:
        expected = {field.name for field in fields(cls)}
        if not isinstance(value, Mapping) or set(value) != expected:
            raise SchemaError("invalid CandidateNode keys")
        payload = dict(value)
        payload["token_ids"] = _integer_tuple(payload["token_ids"], "token_ids", binary=False)
        payload["response_mask"] = _integer_tuple(
            payload["response_mask"], "response_mask", binary=True
        )
        return cls(**payload)

    def to_dict(self) -> dict[str, object]:
        return {
            "prompt_id": self.prompt_id,
            "candidate_id": self.candidate_id,
            "candidate_index": self.candidate_index,
            "split": self.split,
            "prompt": self.prompt,
            "response": self.response,
            "raw_oracle_score": self.raw_oracle_score,
            "oracle_reward": self.oracle_reward,
            "token_ids": list(self.token_ids),
            "response_mask": list(self.response_mask),
            "terminated_by_eos": self.terminated_by_eos,
            "reached_max_length": self.reached_max_length,
            "schema_version": self.schema_version,
        }


Record = TypeVar("Record")


def save_jsonl(path: str | os.PathLike[str], records: Iterable[Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            for record in records:
                to_dict = getattr(record, "to_dict", None)
                if not callable(to_dict):
                    raise TypeError("JSONL records must expose to_dict()")
                json.dump(
                    to_dict(), stream, ensure_ascii=False, allow_nan=False, separators=(",", ":")
                )
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_jsonl(path: str | os.PathLike[str], record_type: type[Record]) -> list[Record]:
    result: list[Record] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise SchemaError(f"blank JSONL line {line_number}")
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise SchemaError(f"JSONL line {line_number} must be an object")
            result.append(record_type.from_dict(value))
    return result


__all__ = ["NODE_SCHEMA", "CandidateNode", "SchemaError", "load_jsonl", "save_jsonl"]
