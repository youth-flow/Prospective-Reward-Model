from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch

import smart_reward.real_policy_evaluation as real_module
from smart_reward.config import load_config
from smart_reward.real_policy_evaluation import (
    BETA,
    _policy_metrics,
    adapter_name,
    export_real_policy_adapters,
    policy_names,
)
from smart_reward.scores import ParameterLayout


class _FakeModel:
    def __init__(self, parameter: torch.nn.Parameter, snapshots: list[torch.Tensor]) -> None:
        self.parameter = parameter
        self.snapshots = snapshots

    def save_pretrained(self, directory: Path, *, safe_serialization: bool) -> None:
        assert safe_serialization
        self.snapshots.append(self.parameter.detach().clone())
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "adapter_model.safetensors").write_bytes(b"real-policy-adapter")
        (directory / "adapter_config.json").write_text("{}\n", encoding="utf-8")


def test_real_policy_contract_is_exactly_four_policies_at_beta_point_two() -> None:
    assert BETA == 0.2
    assert policy_names() == [
        "pi0",
        "mle_rm__beta_0p2",
        "pro_rm__beta_0p2",
        "oracle__beta_0p2",
    ]


def test_adapter_export_writes_direction_divided_by_beta(tmp_path: Path, monkeypatch) -> None:
    config = load_config(Path("configs/fisher_trpo_smoke.yaml"))
    seed = config["run"]["seeds"][0]
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    parameter = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
    layout = ParameterLayout.from_named_parameters((("adapter", parameter),))
    (artifact / "metadata.json").write_text(
        json.dumps(
            {
                "evidence": {
                    "policy_a_sha256": "1" * 64,
                    "policy_layout": layout.to_metadata(),
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    reward_path = tmp_path / "reward.json"
    reward_path.write_text("{}\n", encoding="utf-8")
    reward = {
        "protocol": "prorm_fisher_trpo_v1",
        "producer": {},
        "policy_directions": {
            "mle_rm": [1.0, 2.0],
            "pro_rm": [-1.0, 3.0],
            "oracle": [2.0, -4.0],
        },
    }
    snapshots: list[torch.Tensor] = []
    setup = SimpleNamespace(
        a_state_sha256="1" * 64,
        layout=layout,
        model=_FakeModel(parameter, snapshots),
        named_tangent_parameters=lambda: (("adapter", parameter),),
    )
    monkeypatch.setattr(
        real_module,
        "_validate_source",
        lambda *args, **kwargs: (config, "2" * 64, reward),
    )
    monkeypatch.setattr(real_module, "_load_policy", lambda *args, **kwargs: setup)
    monkeypatch.setattr(real_module, "producer_identity", lambda: {})
    monkeypatch.setattr(
        real_module,
        "validate_real_policy_adapters",
        lambda *args, **kwargs: json.loads(
            (Path(args[3]) / "metadata.json").read_text(encoding="utf-8")
        ),
    )

    metadata = export_real_policy_adapters(
        config,
        artifact,
        reward_path,
        tmp_path / "adapters",
        seed=seed,
        device="cpu",
    )

    expected = [
        torch.tensor(reward["policy_directions"][method], dtype=torch.float64) / BETA
        for method in ("mle_rm", "pro_rm", "oracle")
    ]
    assert len(snapshots) == 3
    assert all(
        torch.equal(observed, target) for observed, target in zip(snapshots, expected, strict=True)
    )
    assert set(metadata["adapters"]) == {adapter_name(method) for method in real_module.METHODS}
    assert metadata["update_rule"] == "lora_B = beta_free_natural_direction / beta"


def test_rollout_summary_exposes_only_r_k_j() -> None:
    rows = [
        {"oracle_reward": 1.0, "forward_kl": 0.5},
        {"oracle_reward": 3.0, "forward_kl": 1.5},
    ]
    metrics = _policy_metrics(rows)
    assert metrics == {"R": 2.0, "K": 1.0, "J": 1.8}
    assert not any("tab" in key.lower() for key in metrics)
