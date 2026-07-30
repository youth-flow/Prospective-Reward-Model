from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

import smart_reward.pipeline as pipeline_module
from smart_reward.artifacts import save_exact_delta_artifact
from smart_reward.checkpoints import write_stage_receipt
from smart_reward.config import config_hash, load_config
from smart_reward.exact import ExactDeltaExperiment, ExactSplitData
from smart_reward.pipeline import (
    _validated_materialization,
    import_materialization_stage,
    run_reward_stage,
)
from smart_reward.runtime import sha256_file

ROOT = Path(__file__).parents[1]


def _identity(monkeypatch: pytest.MonkeyPatch, marker: str) -> dict[str, str]:
    result = {
        "git_commit": marker * 40,
        "image_sha256": marker * 64,
        "hf_inventory_sha256": marker * 64,
    }
    monkeypatch.setenv("PRORM_GIT_COMMIT", result["git_commit"])
    monkeypatch.setenv("PRORM_IMAGE_SHA256", result["image_sha256"])
    monkeypatch.setenv("PRORM_HF_INVENTORY_SHA256", result["hf_inventory_sha256"])
    return result


def _experiment() -> ExactDeltaExperiment:
    generator = torch.Generator().manual_seed(41)

    def split(name: str) -> ExactSplitData:
        return ExactSplitData(
            prompt_ids=(name,),
            policy_scores=torch.randn(1, 6, 2, generator=generator),
            reward_features=torch.randn(1, 6, 3, generator=generator),
            true_rewards=torch.randn(1, 6, generator=generator),
        )

    return ExactDeltaExperiment(
        train=split("train"),
        validation=split("validation"),
        test=split("test"),
    )


def _source_materialization(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, object], dict[str, str]]:
    config = load_config(ROOT / "configs" / "smoke.yaml")
    producer = _identity(monkeypatch, "a")
    artifact = root / "artifact"
    save_exact_delta_artifact(
        _experiment(),
        artifact,
        config_hash=config_hash(config),
        seed=20261001,
    )
    jsonl_sha256: dict[str, str] = {}
    for name in ("prompts.jsonl", "candidates.jsonl", "edges.jsonl"):
        path = artifact / name
        path.write_text("{}\n", encoding="utf-8")
        jsonl_sha256[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata_path = artifact / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["evidence"] = {
        "producer": producer,
        "jsonl_sha256": jsonl_sha256,
    }
    metadata_path.write_text(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    outputs = {
        "artifact_metadata": sha256_file(metadata_path),
        "artifact_tensors": sha256_file(artifact / "tensors.safetensors"),
        "prompts": jsonl_sha256["prompts.jsonl"],
        "candidates": jsonl_sha256["candidates.jsonl"],
        "edges": jsonl_sha256["edges.jsonl"],
    }
    write_stage_receipt(
        root / "stage_receipts" / "materialize.json",
        config,
        stage="materialize",
        seed=20261001,
        inputs={},
        outputs=outputs,
    )
    return config, producer


def test_materialization_import_preserves_source_and_binds_consumer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    config, source_producer = _source_materialization(source, monkeypatch)
    source_receipt = (source / "stage_receipts" / "materialize.json").read_bytes()
    consumer = _identity(monkeypatch, "b")
    analysis = tmp_path / "analysis.md"
    analysis.write_text("earliest affected stage: reward\n", encoding="utf-8")

    outputs = import_materialization_stage(
        config,
        source,
        target,
        analysis,
        seed=20261001,
    )

    assert "materialize_provenance" in outputs
    assert (target / "stage_receipts" / "materialize.json").read_bytes() == source_receipt
    bridge = json.loads(
        (target / "stage_receipts" / "materialize-provenance.json").read_text(encoding="utf-8")
    )
    assert bridge["source"]["producer"] == source_producer
    assert bridge["consumer"] == consumer
    assert _validated_materialization(config, target, seed=20261001) == outputs
    assert import_materialization_stage(config, source, target, analysis, seed=20261001) == outputs


def test_materialization_import_rejects_analysis_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    config, _ = _source_materialization(source, monkeypatch)
    _identity(monkeypatch, "b")
    analysis = tmp_path / "analysis.md"
    analysis.write_text("earliest affected stage: reward\n", encoding="utf-8")
    import_materialization_stage(config, source, target, analysis, seed=20261001)

    imported_analysis = target / "stage_receipts" / "materialize-affected-stage-analysis.md"
    imported_analysis.write_text("tampered\n", encoding="utf-8")

    with pytest.raises(ValueError, match="provenance bridge mismatch"):
        _validated_materialization(config, target, seed=20261001)


def test_materialization_import_rejects_source_receipt_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    config, _ = _source_materialization(source, monkeypatch)
    receipt_path = source / "stage_receipts" / "materialize.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["outputs"]["artifact_metadata"] = "f" * 64
    receipt_path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")
    _identity(monkeypatch, "b")
    analysis = tmp_path / "analysis.md"
    analysis.write_text("earliest affected stage: reward\n", encoding="utf-8")

    with pytest.raises(ValueError, match="source stage receipt mismatch"):
        import_materialization_stage(config, source, target, analysis, seed=20261001)


def test_reward_receipt_binds_materialization_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    config, _ = _source_materialization(source, monkeypatch)
    consumer = _identity(monkeypatch, "b")
    analysis = tmp_path / "analysis.md"
    analysis.write_text("earliest affected stage: reward\n", encoding="utf-8")
    materialization = import_materialization_stage(
        config,
        source,
        target,
        analysis,
        seed=20261001,
    )

    def write_result(
        config,
        artifact_dir,
        output,
        *,
        seed,
        device,
    ) -> None:
        del config, artifact_dir, seed, device
        Path(output).write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(pipeline_module, "run_exact_reward_comparison", write_result)
    monkeypatch.setattr(
        pipeline_module,
        "load_exact_reward_comparison",
        lambda *args, **kwargs: {
            "producer": consumer,
            "artifact_metadata_sha256": materialization["artifact_metadata"],
        },
    )

    run_reward_stage(config, target, seed=20261001, device="cpu")

    receipt = json.loads((target / "stage_receipts" / "reward.json").read_text(encoding="utf-8"))
    assert receipt["inputs"] == {
        "artifact_metadata": materialization["artifact_metadata"],
        "materialize_provenance": materialization["materialize_provenance"],
    }


def test_reward_receipt_binds_validated_pro_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    config, _ = _source_materialization(source, monkeypatch)
    consumer = _identity(monkeypatch, "b")
    analysis = tmp_path / "analysis.md"
    analysis.write_text("earliest affected stage: reward\n", encoding="utf-8")
    materialization = import_materialization_stage(
        config,
        source,
        target,
        analysis,
        seed=20261001,
    )
    pro_source = target / "reward_provenance" / "pro-source.json"
    pro_source.parent.mkdir()
    pro_source.write_text('{"failed_reward_result":true}\n', encoding="utf-8")
    pro_source_sha = sha256_file(pro_source)

    def write_result(
        config,
        artifact_dir,
        output,
        *,
        seed,
        device,
        reuse_pro_from,
    ) -> None:
        del config, artifact_dir, seed, device
        assert Path(reuse_pro_from) == pro_source
        Path(output).write_text("{}\n", encoding="utf-8")

    result = {
        "producer": consumer,
        "artifact_metadata_sha256": materialization["artifact_metadata"],
        "fit_provenance": {
            "MLE-RM": {"mode": "computed"},
            "Pro-RM": {
                "mode": "validated_reuse",
                "source_result_sha256": pro_source_sha,
            },
        },
    }
    monkeypatch.setattr(pipeline_module, "run_exact_reward_comparison", write_result)
    monkeypatch.setattr(
        pipeline_module,
        "load_exact_reward_comparison",
        lambda *args, **kwargs: result,
    )

    run_reward_stage(config, target, seed=20261001, device="cpu")

    receipt = json.loads((target / "stage_receipts" / "reward.json").read_text(encoding="utf-8"))
    assert receipt["inputs"]["pro_fit_source"] == pro_source_sha
    pro_source.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="reused Pro-RM source SHA-256 mismatch"):
        run_reward_stage(config, target, seed=20261001, device="cpu")
