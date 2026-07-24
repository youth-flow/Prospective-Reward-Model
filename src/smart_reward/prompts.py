"""Deterministic prompt preparation for the controlled on-policy experiment."""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

PROMPT_SCHEMA_VERSION = "prompt/v1"
PROMPT_POOL_SELECTION_SCHEMA = "policy-token-length-eligible-prompt-pool/v1"
POLICY_TOKEN_LENGTH_ELIGIBILITY_RULE = (
    "policy_chat_template_tokens_lte_max_prompt_tokens_before_seeded_shuffle"
)
Split = Literal["train", "validation", "test"]
_SPLIT_ORDER: tuple[Split, ...] = ("train", "validation", "test")


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One immutable chat-template message."""

    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in {"developer", "system", "user", "assistant"}:
            raise ValueError(f"unsupported chat role: {self.role!r}")
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError("message content must be a non-empty string")

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True, slots=True)
class PromptRecord:
    """A prompt assigned to exactly one split before candidate generation."""

    prompt_id: str
    messages: tuple[ChatMessage, ...]
    split: Split
    schema_version: str = PROMPT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.prompt_id, str) or not self.prompt_id.strip():
            raise ValueError("prompt_id must be a non-empty string")
        if not isinstance(self.messages, tuple) or not self.messages:
            raise ValueError("messages must be a non-empty tuple")
        if not all(isinstance(message, ChatMessage) for message in self.messages):
            raise TypeError("messages must contain ChatMessage objects")
        if self.split not in _SPLIT_ORDER:
            raise ValueError(f"unsupported split: {self.split!r}")
        if self.schema_version != PROMPT_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {PROMPT_SCHEMA_VERSION!r}")

    def to_dict(self) -> dict[str, object]:
        return {
            "prompt_id": self.prompt_id,
            "messages": [message.to_dict() for message in self.messages],
            "split": self.split,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> PromptRecord:
        expected = {"prompt_id", "messages", "split", "schema_version"}
        if set(value) != expected:
            raise ValueError(
                f"invalid prompt schema: expected {sorted(expected)}, got {sorted(value)}"
            )
        raw_messages = value["messages"]
        if not isinstance(raw_messages, list) or not raw_messages:
            raise TypeError("messages must be a non-empty list")
        messages: list[ChatMessage] = []
        for raw_message in raw_messages:
            if not isinstance(raw_message, Mapping) or set(raw_message) != {"role", "content"}:
                raise ValueError("each message must contain exactly role and content")
            messages.append(
                ChatMessage(
                    role=str(raw_message["role"]),
                    content=str(raw_message["content"]),
                )
            )
        return cls(
            prompt_id=str(value["prompt_id"]),
            messages=tuple(messages),
            split=value["split"],  # type: ignore[arg-type]
            schema_version=str(value["schema_version"]),
        )


@dataclass(frozen=True, slots=True)
class PolicyTokenEligiblePromptPool:
    """A deduplicated prompt population filtered with one frozen policy tokenizer."""

    prompt_texts: tuple[tuple[str, str], ...]
    token_counts: tuple[tuple[str, int], ...]
    eligible_prompt_ids: tuple[str, ...]
    excluded_prompt_ids: tuple[str, ...]
    max_prompt_tokens: int
    policy_chat_template_sha256: str

    def select(
        self,
        *,
        split_sizes: Mapping[str, int],
        seed: int,
    ) -> tuple[list[PromptRecord], dict[str, object]]:
        """Draw one deterministic split and return materializer-identical evidence."""

        prompt_texts = dict(self.prompt_texts)
        token_counts = dict(self.token_counts)
        prompts = prepare_multipref_prompts(
            (
                {"prompt_id": prompt_id, "text": prompt_texts[prompt_id]}
                for prompt_id in self.eligible_prompt_ids
            ),
            split_sizes=split_sizes,
            seed=seed,
        )
        selected_ids = tuple(prompt.prompt_id for prompt in prompts)
        selected_counts = tuple(token_counts[prompt_id] for prompt_id in selected_ids)
        if not selected_counts or max(selected_counts) > self.max_prompt_tokens:
            raise RuntimeError("prompt eligibility filtering failed before the seeded split")
        evidence: dict[str, object] = {
            "schema_version": PROMPT_POOL_SELECTION_SCHEMA,
            "rule": POLICY_TOKEN_LENGTH_ELIGIBILITY_RULE,
            "dataset_unique_prompt_count": len(self.prompt_texts),
            "eligible_prompt_count": len(self.eligible_prompt_ids),
            "excluded_over_limit_prompt_count": len(self.excluded_prompt_ids),
            "selected_prompt_count": len(selected_ids),
            "max_prompt_tokens": self.max_prompt_tokens,
            "filter_applied_before_seeded_shuffle": True,
            "split_seed": seed,
            "policy_chat_template_sha256": self.policy_chat_template_sha256,
            "eligible_prompt_ids_sha256": _string_sequence_sha256(self.eligible_prompt_ids),
            "excluded_prompt_ids_sha256": _string_sequence_sha256(self.excluded_prompt_ids),
            "selected_prompt_ids_sha256": _string_sequence_sha256(selected_ids),
            "selected_minimum_policy_chat_token_count": min(selected_counts),
            "selected_maximum_policy_chat_token_count": max(selected_counts),
            "truncation": False,
        }
        return prompts, evidence


def _string_sequence_sha256(values: Sequence[str]) -> str:
    if isinstance(values, (str, bytes, bytearray)) or not all(
        isinstance(value, str) and value for value in values
    ):
        raise ValueError("digest values must be a sequence of non-empty strings")
    encoded = json.dumps(
        list(values),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def policy_chat_token_count(tokenizer: object, prompt_text: str) -> int:
    """Count one complete policy chat without truncation or tensor allocation."""

    if not isinstance(prompt_text, str) or not prompt_text.strip():
        raise ValueError("prompt_text must be a non-empty string")
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_chat_template):
        raise TypeError("policy tokenizer must expose callable apply_chat_template")
    encoded = apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        tokenize=True,
        add_generation_prompt=True,
        truncation=False,
    )
    if isinstance(encoded, torch.Tensor):
        if encoded.ndim == 1:
            count = int(encoded.numel())
        elif encoded.ndim == 2 and encoded.shape[0] == 1:
            count = int(encoded.shape[1])
        else:
            raise ValueError("policy chat template returned an invalid token tensor")
    elif isinstance(encoded, Sequence) and not isinstance(encoded, (str, bytes, bytearray)):
        count = len(encoded)
    else:
        raise TypeError("policy chat template must return one token-ID sequence")
    if count < 1:
        raise ValueError("policy chat template returned an empty token sequence")
    return count


def build_policy_token_eligible_prompt_pool(
    rows: Iterable[Mapping[str, Any]],
    *,
    policy_tokenizer: object,
    max_prompt_tokens: int,
    eligibility_rule: object,
) -> PolicyTokenEligiblePromptPool:
    """Apply the frozen token-length eligibility rule to the full unique pool."""

    if eligibility_rule != POLICY_TOKEN_LENGTH_ELIGIBILITY_RULE:
        raise ValueError(
            "data.prompt_eligibility must explicitly freeze policy-token-length "
            "filtering before the seeded prompt split"
        )
    if (
        isinstance(max_prompt_tokens, bool)
        or not isinstance(max_prompt_tokens, int)
        or max_prompt_tokens < 1
    ):
        raise ValueError("max_prompt_tokens must be a positive integer")
    chat_template = getattr(policy_tokenizer, "chat_template", None)
    if not isinstance(chat_template, str) or not chat_template:
        raise ValueError("policy tokenizer must provide a non-empty chat_template")

    prompt_texts = deduplicate_multipref_prompt_texts(rows)
    ordered_prompt_texts = tuple(sorted(prompt_texts.items()))
    token_counts = tuple(
        (
            prompt_id,
            policy_chat_token_count(policy_tokenizer, prompt_text),
        )
        for prompt_id, prompt_text in ordered_prompt_texts
    )
    eligible_ids = tuple(
        prompt_id for prompt_id, count in token_counts if count <= max_prompt_tokens
    )
    excluded_ids = tuple(
        prompt_id for prompt_id, count in token_counts if count > max_prompt_tokens
    )
    return PolicyTokenEligiblePromptPool(
        prompt_texts=ordered_prompt_texts,
        token_counts=token_counts,
        eligible_prompt_ids=eligible_ids,
        excluded_prompt_ids=excluded_ids,
        max_prompt_tokens=max_prompt_tokens,
        policy_chat_template_sha256=hashlib.sha256(chat_template.encode("utf-8")).hexdigest(),
    )


def prepare_multipref_prompts(
    rows: Iterable[Mapping[str, Any]],
    *,
    split_sizes: Mapping[str, int],
    seed: int,
) -> list[PromptRecord]:
    """Deduplicate MultiPref rows and split prompts deterministically.

    Rows sharing ``prompt_id`` must also share exactly the same prompt text;
    conflicting duplicates are rejected instead of silently selecting one.
    IDs are sorted before seeded shuffling so input iteration order cannot
    change the split.
    """

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if set(split_sizes) != set(_SPLIT_ORDER):
        raise ValueError(f"split_sizes must contain exactly {_SPLIT_ORDER}")
    normalized_sizes: dict[Split, int] = {}
    for split in _SPLIT_ORDER:
        size = split_sizes[split]
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise ValueError(f"split size for {split!r} must be a positive integer")
        normalized_sizes[split] = size

    prompt_text = deduplicate_multipref_prompt_texts(rows)

    required = sum(normalized_sizes.values())
    if len(prompt_text) < required:
        raise ValueError(f"need {required} unique prompts, found only {len(prompt_text)}")
    prompt_ids = sorted(prompt_text)
    random.Random(seed).shuffle(prompt_ids)
    prompt_ids = prompt_ids[:required]

    records: list[PromptRecord] = []
    offset = 0
    for split in _SPLIT_ORDER:
        for prompt_id in prompt_ids[offset : offset + normalized_sizes[split]]:
            records.append(
                PromptRecord(
                    prompt_id=prompt_id,
                    messages=(ChatMessage(role="user", content=prompt_text[prompt_id]),),
                    split=split,
                )
            )
        offset += normalized_sizes[split]
    return records


def deduplicate_multipref_prompt_texts(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, str]:
    """Validate MultiPref rows and return one immutable-text entry per prompt ID.

    The returned mapping is intentionally not shuffled.  Callers that impose a
    tokenizer-length eligibility rule can therefore filter the complete unique
    prompt population *before* the seeded split is drawn.
    """

    prompt_text: dict[str, str] = {}
    for row_number, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            raise TypeError(f"row {row_number} must be a mapping")
        try:
            prompt_id = row["prompt_id"]
            text = row["text"]
        except KeyError as error:
            raise ValueError(f"row {row_number} is missing {error.args[0]!r}") from error
        if not isinstance(prompt_id, str) or not prompt_id.strip():
            raise ValueError(f"row {row_number} has an invalid prompt_id")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"row {row_number} has invalid prompt text")
        previous = prompt_text.setdefault(prompt_id, text)
        if previous != text:
            raise ValueError(f"prompt_id {prompt_id!r} maps to conflicting prompt text")
    return prompt_text


def load_multipref_parquet_snapshot(
    datasets_module: Any,
    snapshot_path: str | os.PathLike[str],
    *,
    datasets_cache: str | os.PathLike[str] | None = None,
) -> Any:
    """Load the pinned MultiPref train parquet without any Hub metadata call.

    ``datasets.load_dataset(repo_id, local_files_only=True)`` still consults
    Hub metadata in Datasets 3.6.  Formal jobs instead resolve the immutable
    dataset snapshot with ``huggingface_hub`` and pass its local parquet shards
    through this function.
    """

    snapshot = Path(snapshot_path).resolve(strict=True)
    if not snapshot.is_dir():
        raise NotADirectoryError(f"MultiPref snapshot is not a directory: {snapshot}")
    train_files = sorted(
        (path for path in (snapshot / "data").glob("train-*.parquet") if path.is_file()),
        key=lambda path: path.name,
    )
    if not train_files:
        raise FileNotFoundError(f"MultiPref snapshot has no train parquet shards: {snapshot}")
    download_config_type = getattr(datasets_module, "DownloadConfig", None)
    load_dataset = getattr(datasets_module, "load_dataset", None)
    if download_config_type is None or not callable(load_dataset):
        raise RuntimeError("installed datasets package lacks the required loading interface")
    kwargs: dict[str, Any] = {
        "data_files": {"train": [str(path) for path in train_files]},
        "download_config": download_config_type(local_files_only=True),
        "split": "train",
    }
    if datasets_cache is not None:
        kwargs["cache_dir"] = str(Path(datasets_cache).resolve())
    return load_dataset("parquet", **kwargs)


def load_multipref_prompts(
    *,
    dataset_name: str,
    revision: str,
    split_sizes: Mapping[str, int],
    seed: int,
) -> list[PromptRecord]:
    """Load a pinned MultiPref revision and prepare deterministic prompts."""

    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "datasets is required for prompt download; install smart-reward-model[llm]"
        ) from error
    dataset = load_dataset(dataset_name, revision=revision, split="train")
    return prepare_multipref_prompts(dataset, split_sizes=split_sizes, seed=seed)


def save_prompt_jsonl(path: str | os.PathLike[str], records: Iterable[PromptRecord]) -> None:
    """Atomically persist prompt records as strict UTF-8 JSONL."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary_name = handle.name
            for record in records:
                if not isinstance(record, PromptRecord):
                    raise TypeError("records must contain PromptRecord objects")
                json.dump(record.to_dict(), handle, ensure_ascii=False, separators=(",", ":"))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
        temporary_name = None
    finally:
        if temporary_name is not None and Path(temporary_name).exists():
            Path(temporary_name).unlink()


def load_prompt_jsonl(path: str | os.PathLike[str]) -> list[PromptRecord]:
    """Load strict prompt JSONL and reject duplicate IDs or blank lines."""

    source = Path(path)
    records: list[PromptRecord] = []
    seen: set[str] = set()
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{source}:{line_number}: blank lines are forbidden")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{source}:{line_number}: expected a JSON object")
            record = PromptRecord.from_dict(value)
            if record.prompt_id in seen:
                raise ValueError(
                    f"{source}:{line_number}: duplicate prompt_id {record.prompt_id!r}"
                )
            seen.add(record.prompt_id)
            records.append(record)
    return records


__all__ = [
    "POLICY_TOKEN_LENGTH_ELIGIBILITY_RULE",
    "PROMPT_POOL_SELECTION_SCHEMA",
    "PROMPT_SCHEMA_VERSION",
    "ChatMessage",
    "PolicyTokenEligiblePromptPool",
    "PromptRecord",
    "build_policy_token_eligible_prompt_pool",
    "deduplicate_multipref_prompt_texts",
    "load_multipref_parquet_snapshot",
    "load_multipref_prompts",
    "load_prompt_jsonl",
    "policy_chat_token_count",
    "prepare_multipref_prompts",
    "save_prompt_jsonl",
]
