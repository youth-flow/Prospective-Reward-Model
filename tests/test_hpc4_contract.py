from __future__ import annotations

from pathlib import Path

import pytest

from smart_reward.runtime import producer_identity

ROOT = Path(__file__).parents[1]


def test_hpc4_gpu_jobs_request_only_the_primary_resource() -> None:
    for name in ("stage_gpu.sbatch", "gpu_smoke.sbatch"):
        text = (ROOT / "scripts" / "hpc4" / name).read_text(encoding="utf-8")
        assert "#SBATCH --account=sigroup" in text
        assert "#SBATCH --gpus-per-node=1" in text
        assert "#SBATCH --cpus-per-task" not in text
        assert "#SBATCH --mem" not in text


def test_hpc4_contract_contains_no_retired_model_or_environment_names() -> None:
    paths = [
        *ROOT.joinpath("src").rglob("*.py"),
        *ROOT.joinpath("scripts", "hpc4").glob("*.sh"),
        *ROOT.joinpath("scripts", "hpc4").glob("*.sbatch"),
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    assert "SRM_" not in combined
    assert "Qwen3" not in combined


def test_hpc4_paths_follow_project_and_scratch_policy() -> None:
    documentation = (ROOT / "docs" / "hpc4.md").read_text(encoding="utf-8")
    controlled = (ROOT / "scripts" / "hpc4" / "submit_pipeline.sh").read_text(encoding="utf-8")
    assert "/project/sigroup/$USER" in documentation
    assert "/scratch/$USER" in documentation
    assert 'case "${run_root}" in "/scratch/${USER}/"*' in controlled
    assert "--partition=gpu-l20" in controlled


def test_hpc4_pipeline_is_dependency_ordered_and_concurrency_limited() -> None:
    text = (ROOT / "scripts" / "hpc4" / "submit_pipeline.sh").read_text(encoding="utf-8")
    assert "afterok:${materialize_job}" in text
    assert "afterok:${reward_job}" in text
    assert "afterok:${adapter_job}" in text
    assert "%${rollout_concurrency}" in text
    assert "afterok:${rollout_job}" in text
    assert "afterok:${rollout_aggregate_job}" in text
    assert not (ROOT / "scripts" / "hpc4" / "controlled.sbatch").exists()


def test_producer_identity_uses_only_current_environment_names(monkeypatch) -> None:
    for name in (
        "PRORM_GIT_COMMIT",
        "PRORM_IMAGE_SHA256",
        "PRORM_HF_INVENTORY_SHA256",
        "SLURM_JOB_ID",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PRORM_GIT_COMMIT", "a" * 40)
    monkeypatch.setenv("PRORM_IMAGE_SHA256", "b" * 64)
    monkeypatch.setenv("PRORM_HF_INVENTORY_SHA256", "c" * 64)
    assert producer_identity() == {
        "git_commit": "a" * 40,
        "image_sha256": "b" * 64,
        "hf_inventory_sha256": "c" * 64,
    }


def test_slurm_production_requires_all_three_identities(monkeypatch) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    for name in (
        "PRORM_GIT_COMMIT",
        "PRORM_IMAGE_SHA256",
        "PRORM_HF_INVENTORY_SHA256",
    ):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match="requires Git, image, and HF inventory"):
        producer_identity()
