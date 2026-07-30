from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch

import smart_reward.exact_policy as policy_module
from smart_reward.config import load_config
from smart_reward.exact_policy import export_exact_ngd_adapters, validate_adapter_metadata


class _FakeLayout:
    def to_metadata(self) -> dict[str, object]:
        return {"names": ["adapter"], "total": 2}


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
                    "policy_layout": {"names": ["adapter"], "total": 2},
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
