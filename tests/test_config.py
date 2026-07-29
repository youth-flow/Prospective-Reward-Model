from __future__ import annotations

import copy
from pathlib import Path

import pytest

from smart_reward.config import ConfigError, config_hash, load_config, validate_config

ROOT = Path(__file__).parents[1]


def test_main_config_is_the_frozen_protocol() -> None:
    config = load_config(ROOT / "configs" / "main.yaml")
    assert config["run"]["seeds"] == [20261001, 20261002, 20261003]
    assert config["data"]["num_candidates"] == 6
    assert config["policy_update"]["beta_grid"] == [1.0, 2.0, 4.0]
    assert config["policy_update"]["reward_sources"] == ["mle_rm", "pro_rm", "oracle"]
    assert "measured_kl_budget" not in config["policy_update"]


def test_smoke_uses_same_scientific_choices() -> None:
    main = load_config(ROOT / "configs" / "main.yaml")
    smoke = load_config(ROOT / "configs" / "smoke.yaml")
    for key in ("model", "revision", "lora_rank", "lora_layers", "lora_modules"):
        assert smoke["policy"][key] == main["policy"][key]
    assert smoke["data"]["num_candidates"] == main["data"]["num_candidates"]
    assert smoke["policy_update"]["beta_grid"] == main["policy_update"]["beta_grid"]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value["data"].update(num_candidates=1), "num_candidates"),
        (lambda value: value["policy_update"].update(beta_grid=[2.0, 1.0]), "beta_grid"),
        (lambda value: value["policy_update"].update(beta_grid=[1.0, 1.0]), "beta_grid"),
        (lambda value: value["policy_update"].update(kl_budget=0.01), "keys mismatch"),
        (
            lambda value: value["evaluation"].update(validation_usage="early_stopping"),
            "diagnostics",
        ),
    ],
)
def test_invalid_config_is_rejected(mutate, message: str) -> None:
    config = copy.deepcopy(load_config(ROOT / "configs" / "smoke.yaml"))
    mutate(config)
    with pytest.raises(ConfigError, match=message):
        validate_config(config)


def test_hash_is_semantic_and_deterministic() -> None:
    config = load_config(ROOT / "configs" / "main.yaml")
    assert config_hash(config) == config_hash(copy.deepcopy(config))
