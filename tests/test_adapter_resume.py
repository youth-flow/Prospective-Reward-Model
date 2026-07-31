from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch

import smart_reward.exact_policy as policy_module
import smart_reward.trpo_policy as trpo_policy_module
from smart_reward.config import load_config
from smart_reward.exact_policy import export_exact_ngd_adapters, validate_adapter_metadata
from smart_reward.trpo_policy import export_trpo_adapters, validate_trpo_adapter_metadata


class _FakeLayout:
    def to_metadata(self) -> list[dict[str, object]]:
        return [{"name": "adapter", "shape": [1, 2], "numel": 2, "offset": 0}]


class _FakeModel:
    def __init__(self, saves: list[Path]) -> None:
        self.saves = saves

    def save_pretrained(self, directory: Path, *, safe_serialization: bool) -> None:
        assert safe_serialization
        self.saves.append(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "adapter_model.safetensors").write_bytes(b"fixed-adapter")
        (directory / "adapter_config.json").write_text("{}\n", encoding="utf-8")


def _fake_setup(saves: list[Path]) -> SimpleNamespace:
    parameter = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
    layout = _FakeLayout()
    return SimpleNamespace(
        a_state_sha256="1" * 64,
        layout=layout,
        model=_FakeModel(saves),
        named_tangent_parameters=lambda: [("adapter", parameter)],
    )


def test_adapters_resume_independently_and_quarantine_only_invalid_component(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = load_config(Path("configs/main.yaml"))
    seed = 20261001
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "metadata.json").write_text(
        json.dumps(
            {
                "evidence": {
                    "policy_a_sha256": "1" * 64,
                    "policy_layout": [
                        {"name": "adapter", "shape": [1, 2], "numel": 2, "offset": 0}
                    ],
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    comparison_path = tmp_path / "reward.json"
    comparison_path.write_text("{}\n", encoding="utf-8")
    artifact_identity = "2" * 64
    comparison = {
        "artifact_metadata_sha256": artifact_identity,
        "policy_directions": {
            "mle_rm": [1.0, 0.0],
            "pro_rm": [0.0, 1.0],
            "oracle": [1.0, 1.0],
        },
        "methods": {
            "MLE-RM": {"head_sha256": "3" * 64},
            "Pro-RM": {"head_sha256": "4" * 64},
        },
    }
    identity = {
        "git_commit": "5" * 40,
        "image_sha256": "6" * 64,
        "hf_inventory_sha256": "7" * 64,
    }
    saves: list[Path] = []
    monkeypatch.setattr(
        policy_module,
        "exact_delta_artifact_metadata_sha256",
        lambda *args, **kwargs: artifact_identity,
    )
    monkeypatch.setattr(
        policy_module,
        "load_exact_delta_artifact",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        policy_module,
        "load_exact_reward_comparison",
        lambda *args, **kwargs: comparison,
    )
    monkeypatch.setattr(policy_module, "producer_identity", lambda: identity)
    monkeypatch.setattr(
        policy_module,
        "set_tangent_update_",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        policy_module,
        "_load_policy",
        lambda *args, **kwargs: _fake_setup(saves),
    )
    output = tmp_path / "adapters"

    first = export_exact_ngd_adapters(
        config,
        artifact,
        comparison_path,
        output,
        seed=seed,
        device="cpu",
    )

    assert len(saves) == 9
    assert len(first["adapters"]) == 9
    original_receipts = {
        name: record["component_receipt_sha256"] for name, record in first["adapters"].items()
    }
    assert validate_adapter_metadata(output, expected_producer=identity) == first

    monkeypatch.setattr(
        policy_module,
        "_load_policy",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("complete adapter inventory must not reload the policy")
        ),
    )
    second = export_exact_ngd_adapters(
        config,
        artifact,
        comparison_path,
        output,
        seed=seed,
        device="cpu",
    )
    assert second == first

    damaged_name = "mle_rm__beta_1"
    (output / damaged_name / "adapter_model.safetensors").write_bytes(b"tampered")
    repair_saves: list[Path] = []
    monkeypatch.setattr(
        policy_module,
        "_load_policy",
        lambda *args, **kwargs: _fake_setup(repair_saves),
    )
    repaired = export_exact_ngd_adapters(
        config,
        artifact,
        comparison_path,
        output,
        seed=seed,
        device="cpu",
    )

    assert len(repair_saves) == 1
    assert (
        repaired["adapters"][damaged_name]["component_receipt_sha256"]
        == original_receipts[damaged_name]
    )
    assert all(
        repaired["adapters"][name]["component_receipt_sha256"] == receipt
        for name, receipt in original_receipts.items()
    )
    assert any((path / "adapter").is_dir() for path in (output / ".rejected").iterdir())
    assert validate_adapter_metadata(output, expected_producer=identity) == repaired


def test_trpo_adapters_use_peft_safe_names_and_resume_by_component(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = load_config(Path("configs/fisher_trpo_smoke.yaml"))
    seed = 20261001
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    layout = [{"name": "adapter", "shape": [1, 2], "numel": 2, "offset": 0}]
    (artifact / "metadata.json").write_text(
        json.dumps(
            {
                "evidence": {
                    "policy_a_sha256": "1" * 64,
                    "policy_layout": layout,
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    comparison_path = tmp_path / "reward.json"
    comparison_path.write_text("{}\n", encoding="utf-8")
    artifact_identity = "2" * 64
    targets = [0.0003, 0.001, 0.003]
    comparison = {
        "artifact_metadata_sha256": artifact_identity,
        "policy_updates": {
            method: {
                str(target): {
                    "update": [target, -target],
                    "step_scale": 1.0,
                }
                for target in targets
            }
            for method in ("mle_rm", "pro_rm", "oracle")
        },
    }
    identity = {
        "git_commit": "5" * 40,
        "image_sha256": "6" * 64,
        "hf_inventory_sha256": "7" * 64,
    }
    saves: list[Path] = []
    monkeypatch.setattr(
        trpo_policy_module,
        "exact_delta_artifact_metadata_sha256",
        lambda *args, **kwargs: artifact_identity,
    )
    monkeypatch.setattr(
        trpo_policy_module,
        "load_exact_delta_artifact",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        trpo_policy_module,
        "load_trpo_reward_comparison",
        lambda *args, **kwargs: comparison,
    )
    monkeypatch.setattr(trpo_policy_module, "producer_identity", lambda: identity)
    monkeypatch.setattr(
        trpo_policy_module,
        "set_tangent_update_",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        trpo_policy_module,
        "_load_policy",
        lambda *args, **kwargs: _fake_setup(saves),
    )
    output = tmp_path / "trpo-adapters"
    first = export_trpo_adapters(
        config,
        artifact,
        comparison_path,
        output,
        seed=seed,
        device="cpu",
    )
    assert len(saves) == 9
    assert all("." not in name for name in first["adapters"])
    assert set(first["adapters"]) == {
        f"{method}__kappa_{target}"
        for method in ("mle_rm", "pro_rm", "oracle")
        for target in ("0p0003", "0p001", "0p003")
    }
    assert validate_trpo_adapter_metadata(output, expected_producer=identity) == first

    monkeypatch.setattr(
        trpo_policy_module,
        "_load_policy",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("complete TRPO inventory must not reload the policy")
        ),
    )
    assert (
        export_trpo_adapters(
            config,
            artifact,
            comparison_path,
            output,
            seed=seed,
            device="cpu",
        )
        == first
    )
