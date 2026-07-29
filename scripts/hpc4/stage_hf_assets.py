#!/usr/bin/env python3
"""Stage and verify the exact public Hugging Face assets used by ProRM.

The script is intentionally separate from formal experiment jobs.  It may use
the network while staging, but it finishes by resolving every configured asset
with both Transformers and Datasets forced offline.  The emitted inventory
contains only cache-relative POSIX paths and content hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

from smart_reward.config import config_hash, load_config
from smart_reward.prompts import (
    load_multipref_parquet_snapshot,
    prepare_multipref_prompts,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    """Atomically create deterministic JSON without replacing existing bytes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(_canonical_json_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        try:
            # A same-directory hard link is an atomic O_EXCL-style publish:
            # an existing inventory wins and is never silently replaced.
            os.link(temporary_name, path)
        except FileExistsError as error:
            raise FileExistsError(f"refusing to overwrite existing inventory: {path}") from error
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _read_inventory(path: Path) -> tuple[dict[str, object], bytes]:
    if not path.is_file():
        raise FileNotFoundError(f"inventory does not exist: {path}")
    raw = path.read_bytes()

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate inventory key {key!r}: {path}")
            value[key] = item
        return value

    try:
        parsed = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"inventory is not strict UTF-8 JSON: {path}") from error
    if not isinstance(parsed, dict):
        raise ValueError(f"inventory root must be an object: {path}")
    return parsed, raw


@contextmanager
def _offline_huggingface_environment(
    *,
    hub_cache: Path,
    datasets_cache: Path,
) -> Iterator[None]:
    values = {
        "HF_HOME": str(hub_cache.parent),
        "HF_HUB_CACHE": str(hub_cache),
        "HF_DATASETS_CACHE": str(datasets_cache),
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _asset_contract(config: Mapping[str, object]) -> tuple[dict[str, str], ...]:
    data = config["data"]
    policy = config["policy"]
    oracle = config["oracle"]
    if not all(isinstance(section, Mapping) for section in (data, policy, oracle)):
        raise TypeError("validated config sections must be mappings")
    raw_assets = (
        {
            "kind": "dataset",
            "repo_id": str(data["prompt_dataset"]),
            "revision": str(data["prompt_revision"]),
        },
        {
            "kind": "model",
            "repo_id": str(policy["model"]),
            "revision": str(policy["revision"]),
        },
        {
            "kind": "model",
            "repo_id": str(oracle["model"]),
            "revision": str(oracle["revision"]),
        },
    )
    unique: dict[tuple[str, str, str], dict[str, str]] = {}
    for asset in raw_assets:
        key = (asset["kind"], asset["repo_id"], asset["revision"])
        unique.setdefault(key, asset)
    return tuple(unique.values())


def _snapshot_inventory(snapshot: Path, cache_root: Path) -> dict[str, object]:
    snapshot_absolute = snapshot.resolve()
    cache_absolute = cache_root.resolve()
    try:
        relative_snapshot = snapshot_absolute.relative_to(cache_absolute).as_posix()
    except ValueError as error:
        raise ValueError("Hugging Face snapshot escaped the declared cache root") from error

    files: list[dict[str, object]] = []
    total_bytes = 0
    for path in sorted(
        snapshot.rglob("*"),
        key=lambda value: value.relative_to(snapshot).as_posix(),
    ):
        if not path.is_file():
            continue
        size = path.stat().st_size
        total_bytes += size
        files.append(
            {
                "path": path.relative_to(snapshot).as_posix(),
                "bytes": size,
                "sha256": _sha256_file(path),
            }
        )
    if not files:
        raise RuntimeError(f"snapshot contains no files: {snapshot}")
    return {
        "snapshot": relative_snapshot,
        "file_count": len(files),
        "total_bytes": total_bytes,
        "files": files,
    }


def _package_versions() -> dict[str, str]:
    names = (
        "accelerate",
        "datasets",
        "huggingface-hub",
        "peft",
        "safetensors",
        "torch",
        "transformers",
    )
    return {name: importlib.metadata.version(name) for name in names}


def _stage_snapshots(
    assets: tuple[dict[str, str], ...],
    *,
    hub_cache: Path,
    local_files_only: bool,
) -> tuple[tuple[dict[str, str], Path], ...]:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError("huggingface-hub is required to stage assets") from error

    staged: list[tuple[dict[str, str], Path]] = []
    for asset in assets:
        snapshot = Path(
            snapshot_download(
                repo_id=asset["repo_id"],
                repo_type=asset["kind"],
                revision=asset["revision"],
                cache_dir=hub_cache,
                local_files_only=local_files_only,
                token=False,
                max_workers=4,
            )
        )
        if snapshot.name != asset["revision"]:
            raise RuntimeError(
                f"{asset['repo_id']} resolved to {snapshot.name}, expected {asset['revision']}"
            )
        staged.append((asset, snapshot))
    return tuple(staged)


def _verify_offline_resolution(
    config: Mapping[str, object],
    *,
    hub_cache: Path,
    datasets_cache: Path,
    staged: tuple[tuple[dict[str, str], Path], ...],
) -> dict[str, object]:
    for name in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if os.environ.get(name) != "1":
            raise RuntimeError(f"offline verification requires {name}=1")
    try:
        import datasets
        from transformers import AutoConfig, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "prospective-reward-model[llm] is required for offline verification"
        ) from error

    model_checks: list[dict[str, object]] = []
    seen_models: set[tuple[str, str]] = set()
    for section_name in ("policy", "oracle"):
        section = config[section_name]
        if not isinstance(section, Mapping):
            raise TypeError(f"{section_name} must be a mapping")
        model_id = str(section["model"])
        revision = str(section["revision"])
        identity = (model_id, revision)
        if identity in seen_models:
            continue
        seen_models.add(identity)
        loaded_config = AutoConfig.from_pretrained(
            model_id,
            revision=revision,
            cache_dir=hub_cache,
            local_files_only=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            cache_dir=hub_cache,
            local_files_only=True,
            use_fast=True,
        )
        chat_template = getattr(tokenizer, "chat_template", None)
        if not isinstance(chat_template, str) or not chat_template.strip():
            raise RuntimeError(
                f"pinned tokenizer has no non-empty chat template: {model_id}@{revision}"
            )
        model_checks.append(
            {
                "repo_id": model_id,
                "revision": revision,
                "model_type": str(getattr(loaded_config, "model_type", "")),
                "tokenizer_class": type(tokenizer).__name__,
                "chat_template_present": True,
            }
        )

    data = config["data"]
    if not isinstance(data, Mapping):
        raise TypeError("data must be a mapping")
    dataset_identity = (
        "dataset",
        str(data["prompt_dataset"]),
        str(data["prompt_revision"]),
    )
    dataset_snapshots = [
        path
        for asset, path in staged
        if (asset["kind"], asset["repo_id"], asset["revision"]) == dataset_identity
    ]
    if len(dataset_snapshots) != 1:
        raise RuntimeError("staged assets do not contain exactly one pinned prompt dataset")
    dataset = load_multipref_parquet_snapshot(
        datasets,
        dataset_snapshots[0],
        datasets_cache=datasets_cache,
    )
    run = config["run"]
    if not isinstance(run, Mapping):
        raise TypeError("run must be a mapping")
    prompt_split_seed = int(run["prompt_split_seed"])
    prompts = prepare_multipref_prompts(
        dataset,
        split_sizes=run["split_sizes"],  # type: ignore[arg-type]
        seed=prompt_split_seed,
    )
    prompt_digest = hashlib.sha256()
    split_counts: dict[str, int] = {}
    for prompt in prompts:
        split_counts[prompt.split] = split_counts.get(prompt.split, 0) + 1
        prompt_digest.update(
            json.dumps(
                prompt.to_dict(),
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        )
        prompt_digest.update(b"\n")
    prompt_checks = [
        {
            "prompt_split_seed": prompt_split_seed,
            "prepared_prompts": len(prompts),
            "split_counts": split_counts,
            "prepared_prompts_sha256": prompt_digest.hexdigest(),
        }
    ]
    return {
        "models": model_checks,
        "dataset": {
            "repo_id": str(data["prompt_dataset"]),
            "revision": str(data["prompt_revision"]),
            "rows": len(dataset),
            "prompt_checks": prompt_checks,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage pinned public Hugging Face assets and prove offline resolution."
    )
    parser.add_argument("config", help="relative path to configs/smoke.yaml or configs/main.yaml")
    parser.add_argument("cache_root", help="Hugging Face cache root")
    parser.add_argument(
        "--inventory",
        help="inventory output; default is <cache_root>/inventories/<config-sha>.json",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="forbid network access and verify the existing cache",
    )
    return parser


def _inventory_path(arguments: argparse.Namespace, *, cache_root: Path, digest: str) -> Path:
    candidate = (
        Path(arguments.inventory)
        if arguments.inventory
        else cache_root / "inventories" / f"{digest}.json"
    ).resolve()
    try:
        candidate.relative_to(cache_root)
    except ValueError as error:
        raise ValueError("inventory must remain inside the declared cache root") from error
    return candidate


def _build_inventory(
    *,
    digest: str,
    cache_root: Path,
    staged: tuple[tuple[dict[str, str], Path], ...],
    offline: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": "prorm-hf-assets/v1",
        "config_hash": digest,
        "cache_layout": {
            "hub": "hub",
            "datasets": "datasets",
        },
        "packages": _package_versions(),
        "assets": [
            {
                **asset,
                **_snapshot_inventory(snapshot, cache_root),
            }
            for asset, snapshot in staged
        ],
        "offline_resolution": dict(offline),
    }


def _execute(arguments: argparse.Namespace) -> dict[str, object]:
    config = load_config(arguments.config)
    digest = config_hash(config)
    cache_root = Path(arguments.cache_root).resolve()
    hub_cache = cache_root / "hub"
    datasets_cache = cache_root / "datasets"
    inventory_path = _inventory_path(arguments, cache_root=cache_root, digest=digest)
    if arguments.verify_only:
        if not hub_cache.is_dir() or not datasets_cache.is_dir():
            raise FileNotFoundError(
                "verify-only requires existing hub and datasets cache directories"
            )
        recorded, recorded_bytes = _read_inventory(inventory_path)
    else:
        if inventory_path.exists():
            raise FileExistsError(f"refusing to overwrite existing inventory: {inventory_path}")
        hub_cache.mkdir(parents=True, exist_ok=True)
        datasets_cache.mkdir(parents=True, exist_ok=True)
        recorded = None
        recorded_bytes = None

    assets = _asset_contract(config)
    if arguments.verify_only:
        # Set all three library-level offline switches before importing or
        # resolving any Hugging Face object.
        with _offline_huggingface_environment(
            hub_cache=hub_cache,
            datasets_cache=datasets_cache,
        ):
            staged = _stage_snapshots(
                assets,
                hub_cache=hub_cache,
                local_files_only=True,
            )
            offline = _verify_offline_resolution(
                config,
                hub_cache=hub_cache,
                datasets_cache=datasets_cache,
                staged=staged,
            )
    else:
        staged = _stage_snapshots(
            assets,
            hub_cache=hub_cache,
            local_files_only=False,
        )
        # The inventory certifies a second resolution with every network path
        # disabled, not merely a successful networked download.
        with _offline_huggingface_environment(
            hub_cache=hub_cache,
            datasets_cache=datasets_cache,
        ):
            offline_staged = _stage_snapshots(
                assets,
                hub_cache=hub_cache,
                local_files_only=True,
            )
            if tuple(path for _, path in offline_staged) != tuple(path for _, path in staged):
                raise RuntimeError("offline snapshot resolution changed staged asset paths")
            offline = _verify_offline_resolution(
                config,
                hub_cache=hub_cache,
                datasets_cache=datasets_cache,
                staged=staged,
            )

    payload = _build_inventory(
        digest=digest,
        cache_root=cache_root,
        staged=staged,
        offline=offline,
    )
    canonical_bytes = _canonical_json_bytes(payload)
    if arguments.verify_only:
        if recorded != payload:
            raise RuntimeError(
                "staged Hugging Face assets or runtime package versions do not match inventory"
            )
        if recorded_bytes != canonical_bytes:
            raise RuntimeError("inventory bytes are not in the canonical deterministic encoding")
    else:
        _atomic_write_json(inventory_path, payload)
        recorded_bytes = canonical_bytes

    inventory_sha256 = hashlib.sha256(recorded_bytes).hexdigest()
    return {
        "config_hash": digest,
        "inventory": inventory_path.relative_to(cache_root).as_posix(),
        "inventory_sha256": inventory_sha256,
        "status": "ok",
        "verify_only": bool(arguments.verify_only),
    }


def main() -> int:
    result = _execute(build_parser().parse_args())
    print(
        json.dumps(
            result,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
