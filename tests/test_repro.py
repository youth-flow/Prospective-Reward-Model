from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

import smart_reward.repro as repro_module
from smart_reward.config import load_config
from smart_reward.repro import (
    atomic_write_json,
    build_run_manifest,
    collect_execution_identity,
    collect_git_state,
    collect_slurm_environment,
)

ROOT = Path(__file__).resolve().parents[1]


def test_git_state_uses_commit_and_porcelain_status(monkeypatch: pytest.MonkeyPatch) -> None:
    commands: list[list[str]] = []

    def fake_run(command: list[str], **_: object) -> SimpleNamespace:
        commands.append(command)
        output = "a" * 40 if "rev-parse" in command else " M README.md\n"
        return SimpleNamespace(returncode=0, stdout=output)

    monkeypatch.setattr(repro_module.subprocess, "run", fake_run)

    assert collect_git_state(ROOT) == {"commit": "a" * 40, "dirty": True}
    assert commands == [
        ["git", "rev-parse", "--verify", "HEAD"],
        ["git", "status", "--porcelain", "--untracked-files=normal"],
    ]


def test_manifest_is_utc_complete_and_never_copies_secret_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(ROOT / "configs" / "smoke.yaml")
    environment = {
        "SLURM_JOB_ID": "230642",
        "SLURM_CPUS_PER_TASK": "8",
        "SLURM_SUBMIT_DIR": "/home/researcher/Smart-Reward-Model",
        "PRORM_IMAGE": "/project/sigroup/smart-reward-model/images/prorm.sif",
        "SRM_IMAGE": "/project/sigroup/smart-reward-model/images/prorm.sif",
        "PRORM_IMAGE_SHA256": "a" * 64,
        "PRORM_HF_INVENTORY_SHA256": "c" * 64,
        "PRORM_GIT_COMMIT": "b" * 40,
        "HF_TOKEN": "hf_super_secret",
        "HUGGING_FACE_HUB_TOKEN": "also_secret",
        "WANDB_API_KEY": "wandb_super_secret",
        "AWS_SECRET_ACCESS_KEY": "cloud_secret",
    }
    monkeypatch.setattr(
        repro_module,
        "collect_git_state",
        lambda _: {"commit": "b" * 40, "dirty": False},
    )
    monkeypatch.setattr(
        repro_module,
        "collect_torch_state",
        lambda: {
            "installed": True,
            "version": "2.test",
            "cuda_available": True,
            "cuda_version": "12.test",
            "cudnn_version": 90000,
            "gpu_count": 1,
            "gpus": [{"index": 0, "name": "Mock GPU"}],
        },
    )
    china_time = timezone(timedelta(hours=8))

    manifest = build_run_manifest(
        config,
        repo_path=ROOT,
        environ=environment,
        now=datetime(2026, 7, 22, 18, 2, 29, tzinfo=china_time),
    )
    payload = manifest.to_dict()
    serialized = json.dumps(payload, sort_keys=True)

    assert payload["created_at_utc"] == "2026-07-22T10:02:29Z"
    assert payload["git"] == {"commit": "b" * 40, "dirty": False}
    assert payload["slurm"] == {
        "PRORM_GIT_COMMIT": "b" * 40,
        "PRORM_HF_INVENTORY_SHA256": "c" * 64,
        "PRORM_IMAGE_SHA256": "a" * 64,
        "SLURM_CPUS_PER_TASK": "8",
        "SLURM_JOB_ID": "230642",
    }
    assert not {"SLURM_SUBMIT_DIR", "PRORM_IMAGE", "SRM_IMAGE"} & set(payload["slurm"])
    assert payload["seed"] == 20260722
    assert payload["selected_seed"] == 20260722
    assert payload["normalized_config"] == config
    assert set(payload["named_seeds"]["20260722"]) >= {
        "prompt_split",
        "candidate_generation",
        "annotations",
    }
    assert set(payload["packages"]) >= {"torch", "transformers", "peft"}
    assert payload["revisions"]["policy_model"]["revision"] == config["policy"]["revision"]
    for forbidden in (
        "HF_TOKEN",
        "hf_super_secret",
        "HUGGING_FACE_HUB_TOKEN",
        "also_secret",
        "WANDB_API_KEY",
        "wandb_super_secret",
        "AWS_SECRET_ACCESS_KEY",
        "cloud_secret",
        "/home/researcher/Smart-Reward-Model",
        "/project/sigroup/smart-reward-model/images/prorm.sif",
    ):
        assert forbidden not in serialized


def test_slurm_collector_is_an_allowlist() -> None:
    assert collect_slurm_environment(
        {
            "SLURM_JOB_ID": "1",
            "SLURM_JOB_ACCOUNT": "sigroup",
            "CUDA_VISIBLE_DEVICES": "3",
            "PRORM_HF_INVENTORY_SHA256": "c" * 64,
            "SLURM_FAKE_SECRET": "no",
            "HF_TOKEN": "no",
        }
    ) == {
        "SLURM_JOB_ID": "1",
        "SLURM_JOB_ACCOUNT": "sigroup",
        "CUDA_VISIBLE_DEVICES": "3",
        "PRORM_HF_INVENTORY_SHA256": "c" * 64,
    }


def test_execution_identity_requires_sigroup_and_exactly_one_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        repro_module,
        "collect_torch_state",
        lambda: {
            "cuda_available": True,
            "gpu_count": 1,
            "gpus": [{"name": "NVIDIA L20"}],
        },
    )
    environment = {
        "PRORM_GIT_COMMIT": "a" * 40,
        "PRORM_IMAGE_SHA256": "b" * 64,
        "PRORM_HF_INVENTORY_SHA256": "c" * 64,
        "SLURM_JOB_ACCOUNT": "sigroup",
        "SLURM_JOB_PARTITION": "gpu-l20",
    }

    assert collect_execution_identity(environment) == {
        "formal": True,
        "git_commit": "a" * 40,
        "image_sha256": "b" * 64,
        "hf_inventory_sha256": "c" * 64,
        "account": "sigroup",
        "partition": "gpu-l20",
        "gpu_models": ["NVIDIA L20"],
    }
    environment["SLURM_JOB_ACCOUNT"] = "another-account"
    assert collect_execution_identity(environment)["formal"] is False


def test_execution_identity_is_not_formal_without_hf_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        repro_module,
        "collect_torch_state",
        lambda: {
            "cuda_available": True,
            "gpu_count": 1,
            "gpus": [{"name": "NVIDIA L20"}],
        },
    )
    identity = collect_execution_identity(
        {
            "PRORM_GIT_COMMIT": "a" * 40,
            "PRORM_IMAGE_SHA256": "b" * 64,
            "SLURM_JOB_ACCOUNT": "sigroup",
            "SLURM_JOB_PARTITION": "gpu-l20",
        }
    )

    assert identity["formal"] is False
    assert identity["hf_inventory_sha256"] is None


def test_execution_identity_accepts_legacy_keys_but_rejects_conflicts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        repro_module,
        "collect_torch_state",
        lambda: {
            "cuda_available": True,
            "gpu_count": 1,
            "gpus": [{"name": "NVIDIA L20"}],
        },
    )
    environment = {
        "SRM_GIT_COMMIT": "a" * 40,
        "SRM_IMAGE_SHA256": "b" * 64,
        "SRM_HF_INVENTORY_SHA256": "c" * 64,
        "SLURM_JOB_ACCOUNT": "sigroup",
        "SLURM_JOB_PARTITION": "gpu-l20",
    }
    assert collect_execution_identity(environment)["formal"] is True
    environment["PRORM_IMAGE_SHA256"] = "c" * 64
    with pytest.raises(ValueError, match="conflicting PRORM_IMAGE_SHA256"):
        collect_execution_identity(environment)

    environment["PRORM_IMAGE_SHA256"] = ""
    with pytest.raises(ValueError, match="conflicting PRORM_IMAGE_SHA256"):
        collect_execution_identity(environment)


def test_atomic_json_write_replaces_destination_and_leaves_no_temp_files(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "manifest.json"
    destination.write_text("old", encoding="utf-8")

    atomic_write_json(destination, {"z": 1, "a": "值"})

    assert json.loads(destination.read_text(encoding="utf-8")) == {"a": "值", "z": 1}
    assert list(tmp_path.glob(".manifest.json.*.tmp")) == []


def test_failed_atomic_write_preserves_existing_destination(tmp_path: Path) -> None:
    destination = tmp_path / "manifest.json"
    destination.write_text("old\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Out of range float"):
        atomic_write_json(destination, {"invalid": float("nan")})

    assert destination.read_text(encoding="utf-8") == "old\n"
    assert list(tmp_path.glob(".manifest.json.*.tmp")) == []


def test_exclusive_atomic_json_write_creates_once_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "aggregate.json"

    atomic_write_json(destination, {"version": 1}, overwrite=False)

    assert json.loads(destination.read_text(encoding="utf-8")) == {"version": 1}
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        atomic_write_json(destination, {"version": 2}, overwrite=False)
    assert json.loads(destination.read_text(encoding="utf-8")) == {"version": 1}
    assert list(tmp_path.glob(".aggregate.json.*.tmp")) == []


def test_manifest_does_not_implicitly_read_supplied_secret_process_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "process_secret")
    monkeypatch.setenv("WANDB_API_KEY", "wandb_process_secret")
    monkeypatch.setenv("SLURM_JOB_ID", "7")

    collected = collect_slurm_environment(os.environ)
    serialized = json.dumps(collected)

    assert collected["SLURM_JOB_ID"] == "7"
    assert "HF_TOKEN" not in collected
    assert "WANDB_API_KEY" not in collected
    assert "secret" not in serialized
