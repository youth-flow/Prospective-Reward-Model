"""Deterministic prompt preparation for the controlled on-policy experiment."""

from __future__ import annotations

import json
import os
import random
import tempfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

PROMPT_SCHEMA_VERSION = "prompt/v1"
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


def prepare_multipref_prompts(
    rows: Iterable[Mapping[str, Any]],
    *,
    split_sizes: Mapping[str, int],
    seed: int,
    text_filter: Callable[[str], bool] | None = None,
    split_offsets: Mapping[str, int] | None = None,
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

    if text_filter is not None and not callable(text_filter):
        raise TypeError("text_filter must be callable or None")
    if text_filter is not None:
        prompt_text = {
            prompt_id: text for prompt_id, text in prompt_text.items() if text_filter(text)
        }

    if split_offsets is None:
        normalized_offsets: dict[Split, int] = {}
        running_offset = 0
        for split in _SPLIT_ORDER:
            normalized_offsets[split] = running_offset
            running_offset += normalized_sizes[split]
    else:
        if set(split_offsets) != set(_SPLIT_ORDER):
            raise ValueError(f"split_offsets must contain exactly {_SPLIT_ORDER}")
        normalized_offsets = {}
        for split in _SPLIT_ORDER:
            offset = split_offsets[split]
            if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
                raise ValueError(f"split offset for {split!r} must be a non-negative integer")
            normalized_offsets[split] = offset
        intervals = sorted(
            (
                normalized_offsets[split],
                normalized_offsets[split] + normalized_sizes[split],
                split,
            )
            for split in _SPLIT_ORDER
        )
        for (_, previous_end, previous_split), (next_start, _, next_split) in zip(
            intervals, intervals[1:], strict=False
        ):
            if previous_end > next_start:
                raise ValueError(f"split intervals overlap: {previous_split!r} and {next_split!r}")

    required = max(normalized_offsets[split] + normalized_sizes[split] for split in _SPLIT_ORDER)
    if len(prompt_text) < required:
        raise ValueError(f"need {required} unique prompts, found only {len(prompt_text)}")
    prompt_ids = sorted(prompt_text)
    random.Random(seed).shuffle(prompt_ids)

    records: list[PromptRecord] = []
    for split in _SPLIT_ORDER:
        offset = normalized_offsets[split]
        for prompt_id in prompt_ids[offset : offset + normalized_sizes[split]]:
            records.append(
                PromptRecord(
                    prompt_id=prompt_id,
                    messages=(ChatMessage(role="user", content=prompt_text[prompt_id]),),
                    split=split,
                )
            )
    return records


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
    split_offsets: Mapping[str, int] | None = None,
) -> list[PromptRecord]:
    """Load a pinned MultiPref revision and prepare deterministic prompts."""

    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "datasets is required for prompt download; install prospective-reward-model[llm]"
        ) from error
    dataset = load_dataset(dataset_name, revision=revision, split="train")
    return prepare_multipref_prompts(
        dataset,
        split_sizes=split_sizes,
        seed=seed,
        split_offsets=split_offsets,
    )


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
    "PROMPT_SCHEMA_VERSION",
    "ChatMessage",
    "PromptRecord",
    "load_multipref_parquet_snapshot",
    "load_multipref_prompts",
    "load_prompt_jsonl",
    "prepare_multipref_prompts",
    "save_prompt_jsonl",
]
