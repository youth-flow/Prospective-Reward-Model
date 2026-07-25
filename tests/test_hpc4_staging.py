from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from smart_reward.config import load_config


def _load_staging_module() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "hpc4" / "stage_hf_assets.py"
    specification = importlib.util.spec_from_file_location("prorm_hpc4_staging", path)
    if specification is None or specification.loader is None:
        raise RuntimeError("could not load the HPC4 staging module")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_asset_contract_matches_unique_pinned_repositories() -> None:
    staging = _load_staging_module()
    config = load_config(Path(__file__).parents[1] / "configs" / "main.yaml")

    assets = staging._asset_contract(config)

    assert assets == (
        {
            "kind": "dataset",
            "repo_id": "allenai/multipref",
            "revision": "12910233a0238a997ebe425656e9dfed7b0ff031",
        },
        {
            "kind": "model",
            "repo_id": "Qwen/Qwen2.5-0.5B-Instruct",
            "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
        },
        {
            "kind": "model",
            "repo_id": "Skywork/Skywork-Reward-V2-Qwen3-0.6B",
            "revision": "8c14a4e9e6321deaf572544339b16b8d6bbe8886",
        },
    )


def test_snapshot_inventory_is_cache_relative_and_content_hashed(tmp_path: Path) -> None:
    staging = _load_staging_module()
    cache = tmp_path / "hf-cache"
    snapshot = cache / "hub" / "models--test" / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    first = snapshot / "config.json"
    second = snapshot / "nested" / "weights.safetensors"
    first.write_bytes(b'{"model_type":"test"}\n')
    second.parent.mkdir()
    second.write_bytes(b"weights")

    inventory = staging._snapshot_inventory(snapshot, cache)

    assert inventory["snapshot"] == f"hub/models--test/snapshots/{'a' * 40}"
    assert inventory["file_count"] == 2
    assert inventory["total_bytes"] == first.stat().st_size + second.stat().st_size
    files = {item["path"]: item for item in inventory["files"]}
    assert files["config.json"]["sha256"] == hashlib.sha256(first.read_bytes()).hexdigest()
    assert (
        files["nested/weights.safetensors"]["sha256"]
        == hashlib.sha256(second.read_bytes()).hexdigest()
    )
    assert all("\\" not in path for path in files)


def test_snapshot_inventory_rejects_path_outside_cache(tmp_path: Path) -> None:
    staging = _load_staging_module()
    cache = tmp_path / "hf-cache"
    snapshot = tmp_path / "outside"
    cache.mkdir()
    snapshot.mkdir()
    (snapshot / "file").write_text("content", encoding="utf-8")

    with pytest.raises(ValueError, match="escaped"):
        staging._snapshot_inventory(snapshot, cache)


def test_inventory_writer_is_atomic_utf8_json(tmp_path: Path) -> None:
    staging = _load_staging_module()
    destination = tmp_path / "inventories" / "assets.json"
    payload = {"schema_version": "test/v1", "name": "相对路径"}

    staging._atomic_write_json(destination, payload)

    assert json.loads(destination.read_text(encoding="utf-8")) == payload
    assert list(destination.parent.glob(".*.tmp")) == []


def test_inventory_writer_refuses_to_replace_existing_bytes(tmp_path: Path) -> None:
    staging = _load_staging_module()
    destination = tmp_path / "inventory.json"
    destination.write_bytes(b"original\n")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        staging._atomic_write_json(destination, {"replacement": True})

    assert destination.read_bytes() == b"original\n"
    assert list(tmp_path.glob(".*.tmp")) == []


def test_verify_only_is_offline_deterministic_and_does_not_write_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = _load_staging_module()
    cache = tmp_path / "hf-cache"
    offline_observations: list[tuple[bool, str | None, str | None, str | None]] = []

    def fake_stage(
        assets: tuple[dict[str, str], ...],
        *,
        hub_cache: Path,
        local_files_only: bool,
    ) -> tuple[tuple[dict[str, str], Path], ...]:
        offline_observations.append(
            (
                local_files_only,
                os.environ.get("HF_HUB_OFFLINE"),
                os.environ.get("HF_DATASETS_OFFLINE"),
                os.environ.get("TRANSFORMERS_OFFLINE"),
            )
        )
        result = []
        for index, asset in enumerate(assets):
            snapshot = hub_cache / f"repo-{index}" / "snapshots" / asset["revision"]
            snapshot.mkdir(parents=True, exist_ok=True)
            (snapshot / "asset.bin").write_bytes(f"asset-{index}".encode())
            result.append((asset, snapshot))
        return tuple(result)

    def fake_offline(*_: object, **__: object) -> dict[str, object]:
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert os.environ["HF_DATASETS_OFFLINE"] == "1"
        assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
        return {"models": [], "dataset": {"prompt_checks": []}}

    monkeypatch.setattr(staging, "_stage_snapshots", fake_stage)
    monkeypatch.setattr(staging, "_verify_offline_resolution", fake_offline)
    monkeypatch.setattr(staging, "_package_versions", lambda: {"test": "1"})
    arguments = argparse.Namespace(
        config=str(Path(__file__).parents[1] / "configs" / "smoke.yaml"),
        cache_root=str(cache),
        inventory=None,
        verify_only=False,
        verification_datasets_cache=None,
    )
    created = staging._execute(arguments)
    inventory = cache / str(created["inventory"])
    original = inventory.read_bytes()
    original_stat = inventory.stat()

    arguments.verify_only = True
    verified = staging._execute(arguments)

    assert verified["inventory_sha256"] == hashlib.sha256(original).hexdigest()
    assert inventory.read_bytes() == original
    assert inventory.stat().st_mtime_ns == original_stat.st_mtime_ns
    # The first call may access the network once; every local/verify-only
    # resolution observes all three offline switches before it starts.
    assert offline_observations[0][0] is False
    assert all(observation == (True, "1", "1", "1") for observation in offline_observations[1:])


def test_verify_only_can_use_empty_per_job_datasets_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = _load_staging_module()
    cache = tmp_path / "hf-cache"
    work_cache = tmp_path / "job" / "datasets"
    work_cache.mkdir(parents=True)
    observed: list[Path] = []

    def fake_stage(
        assets: tuple[dict[str, str], ...],
        *,
        hub_cache: Path,
        local_files_only: bool,
    ) -> tuple[tuple[dict[str, str], Path], ...]:
        result = []
        for index, asset in enumerate(assets):
            snapshot = hub_cache / f"repo-{index}" / "snapshots" / asset["revision"]
            snapshot.mkdir(parents=True, exist_ok=True)
            (snapshot / "asset.bin").write_bytes(f"asset-{index}".encode())
            result.append((asset, snapshot))
        return tuple(result)

    def fake_offline(
        *_: object,
        datasets_cache: Path,
        **__: object,
    ) -> dict[str, object]:
        observed.append(datasets_cache)
        (datasets_cache / "runtime.lock").write_text("ephemeral", encoding="utf-8")
        return {"models": [], "dataset": {"prompt_checks": []}}

    monkeypatch.setattr(staging, "_stage_snapshots", fake_stage)
    monkeypatch.setattr(staging, "_verify_offline_resolution", fake_offline)
    monkeypatch.setattr(staging, "_package_versions", lambda: {"test": "1"})
    arguments = argparse.Namespace(
        config=str(Path(__file__).parents[1] / "configs" / "smoke.yaml"),
        cache_root=str(cache),
        inventory=None,
        verify_only=False,
        verification_datasets_cache=None,
    )
    created = staging._execute(arguments)
    frozen_datasets_before = tuple((cache / "datasets").iterdir())

    arguments.verify_only = True
    arguments.verification_datasets_cache = str(work_cache)
    verified = staging._execute(arguments)

    assert verified["inventory_sha256"] == created["inventory_sha256"]
    assert observed[-1] == work_cache.resolve()
    assert (work_cache / "runtime.lock").read_text(encoding="utf-8") == "ephemeral"
    assert tuple((cache / "datasets").iterdir()) == frozen_datasets_before


def test_verification_datasets_cache_is_verify_only_and_must_start_empty(
    tmp_path: Path,
) -> None:
    staging = _load_staging_module()
    work_cache = tmp_path / "job" / "datasets"
    work_cache.mkdir(parents=True)
    arguments = argparse.Namespace(
        config=str(Path(__file__).parents[1] / "configs" / "smoke.yaml"),
        cache_root=str(tmp_path / "hf-cache"),
        inventory=None,
        verify_only=False,
        verification_datasets_cache=str(work_cache),
    )
    with pytest.raises(ValueError, match="requires --verify-only"):
        staging._execute(arguments)

    arguments.verify_only = True
    (work_cache / "unexpected").write_text("bytes", encoding="utf-8")
    with pytest.raises(ValueError, match="must start empty"):
        staging._execute(arguments)


def test_verification_datasets_cache_must_be_absolute_canonical_and_outside_frozen_root(
    tmp_path: Path,
) -> None:
    staging = _load_staging_module()
    cache = tmp_path / "hf-cache"
    (cache / "hub").mkdir(parents=True)
    (cache / "datasets").mkdir()
    inventory = cache / "inventories" / "placeholder.json"
    inventory.parent.mkdir()
    inventory.write_text("{}\n", encoding="utf-8")
    arguments = argparse.Namespace(
        config=str(Path(__file__).parents[1] / "configs" / "smoke.yaml"),
        cache_root=str(cache),
        inventory=str(inventory),
        verify_only=True,
        verification_datasets_cache="relative-cache",
    )
    with pytest.raises(ValueError, match="must be absolute"):
        staging._execute(arguments)

    inside = cache / "runtime"
    inside.mkdir()
    arguments.verification_datasets_cache = str(inside)
    with pytest.raises(ValueError, match="outside the frozen cache"):
        staging._execute(arguments)

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    try:
        linked_parent.symlink_to(real_parent, target_is_directory=True)
    except OSError:
        pytest.skip("host does not permit directory symlink creation")
    through_symlink = linked_parent / "datasets"
    (real_parent / "datasets").mkdir()
    arguments.verification_datasets_cache = str(through_symlink)
    with pytest.raises(ValueError, match="canonical and contain no symlink"):
        staging._execute(arguments)


def test_offline_verification_uses_formal_prompt_preparation_and_local_only_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = _load_staging_module()
    config = load_config(Path(__file__).parents[1] / "configs" / "smoke.yaml")
    config["data"]["prompt_eligibility"] = (
        "policy_chat_template_tokens_lte_max_prompt_tokens_before_seeded_shuffle"
    )
    rows = [
        {
            "prompt_id": f"p-{index:02d}",
            "text": f"{'long ' if index < 6 else ''}prompt {index}",
        }
        for index in range(70)
    ]
    load_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    tokenizer_calls: list[tuple[list[dict[str, str]], dict[str, object]]] = []
    dataset_asset = staging._asset_contract(config)[0]
    dataset_snapshot = (
        tmp_path / "hub" / "datasets--allenai--multipref" / "snapshots" / dataset_asset["revision"]
    )
    parquet = dataset_snapshot / "data" / "train-00000-of-00001.parquet"
    parquet.parent.mkdir(parents=True)
    parquet.write_bytes(b"test parquet placeholder")
    staged = ((dataset_asset, dataset_snapshot),)

    class DownloadConfig:
        def __init__(self, *, local_files_only: bool) -> None:
            self.local_files_only = local_files_only

    def load_dataset(*args: object, **kwargs: object) -> list[dict[str, str]]:
        load_calls.append((args, kwargs))
        assert args == ("parquet",)
        assert isinstance(kwargs["download_config"], DownloadConfig)
        assert kwargs["download_config"].local_files_only is True
        assert kwargs["data_files"] == {"train": [str(parquet)]}
        assert kwargs["split"] == "train"
        return rows

    class ConfigFactory:
        @staticmethod
        def from_pretrained(*_: object, **kwargs: object) -> object:
            assert kwargs["local_files_only"] is True
            return SimpleNamespace(model_type="test")

    class Tokenizer:
        chat_template = "{{ messages }}"

        @staticmethod
        def apply_chat_template(messages: list[dict[str, str]], **kwargs: object) -> list[int]:
            tokenizer_calls.append((messages, kwargs))
            assert kwargs == {
                "tokenize": True,
                "add_generation_prompt": True,
                "truncation": False,
            }
            return [1] * (400 if messages[0]["content"].startswith("long ") else 8)

    class TokenizerFactory:
        @staticmethod
        def from_pretrained(*_: object, **kwargs: object) -> object:
            assert kwargs["local_files_only"] is True
            assert kwargs["use_fast"] is True
            return Tokenizer()

    datasets_module = ModuleType("datasets")
    datasets_module.DownloadConfig = DownloadConfig
    datasets_module.load_dataset = load_dataset
    transformers_module = ModuleType("transformers")
    transformers_module.AutoConfig = ConfigFactory
    transformers_module.AutoTokenizer = TokenizerFactory
    monkeypatch.setitem(sys.modules, "datasets", datasets_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)

    with staging._offline_huggingface_environment(
        hub_cache=tmp_path / "hub",
        datasets_cache=tmp_path / "datasets",
    ):
        evidence = staging._verify_offline_resolution(
            config,
            hub_cache=tmp_path / "hub",
            datasets_cache=tmp_path / "datasets",
            staged=staged,
        )

    prompt_check = evidence["dataset"]["prompt_checks"][0]
    assert prompt_check["prepared_prompts"] == 64
    assert prompt_check["split_counts"] == {"train": 48, "validation": 8, "test": 8}
    assert len(prompt_check["prepared_prompts_sha256"]) == 64
    selection = prompt_check["prompt_pool_selection"]
    assert selection["dataset_unique_prompt_count"] == 70
    assert selection["eligible_prompt_count"] == 64
    assert selection["excluded_over_limit_prompt_count"] == 6
    assert selection["selected_prompt_count"] == 64
    assert selection["filter_applied_before_seeded_shuffle"] is True
    assert selection["selected_minimum_policy_chat_token_count"] == 8
    assert selection["selected_maximum_policy_chat_token_count"] == 8
    assert selection["truncation"] is False
    expected_eligible_ids = [f"p-{index:02d}" for index in range(6, 70)]
    expected_excluded_ids = [f"p-{index:02d}" for index in range(6)]
    assert (
        selection["eligible_prompt_ids_sha256"]
        == hashlib.sha256(
            json.dumps(
                expected_eligible_ids,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    )
    assert (
        selection["excluded_prompt_ids_sha256"]
        == hashlib.sha256(
            json.dumps(
                expected_excluded_ids,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    )
    assert len(tokenizer_calls) == 70
    assert len(load_calls) == 1


def test_offline_verification_rejects_empty_chat_template(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = _load_staging_module()
    config = load_config(Path(__file__).parents[1] / "configs" / "smoke.yaml")

    class DownloadConfig:
        def __init__(self, *, local_files_only: bool) -> None:
            self.local_files_only = local_files_only

    class ConfigFactory:
        @staticmethod
        def from_pretrained(*_: object, **__: object) -> object:
            return SimpleNamespace(model_type="test")

    class TokenizerFactory:
        @staticmethod
        def from_pretrained(*_: object, **__: object) -> object:
            return SimpleNamespace(chat_template="")

    datasets_module = ModuleType("datasets")
    datasets_module.DownloadConfig = DownloadConfig
    datasets_module.load_dataset = lambda *_args, **_kwargs: []
    transformers_module = ModuleType("transformers")
    transformers_module.AutoConfig = ConfigFactory
    transformers_module.AutoTokenizer = TokenizerFactory
    monkeypatch.setitem(sys.modules, "datasets", datasets_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)

    with (
        staging._offline_huggingface_environment(
            hub_cache=tmp_path / "hub",
            datasets_cache=tmp_path / "datasets",
        ),
        pytest.raises(RuntimeError, match="no non-empty chat template"),
    ):
        staging._verify_offline_resolution(
            config,
            hub_cache=tmp_path / "hub",
            datasets_cache=tmp_path / "datasets",
            staged=(),
        )
