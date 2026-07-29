from __future__ import annotations

import json
from pathlib import Path

import pytest

from smart_reward.config import PROTOCOL, config_hash, load_config
from smart_reward.statistics import aggregate_results

ROOT = Path(__file__).parents[1]


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_three_seed_aggregation_preserves_pairing(tmp_path) -> None:
    config = load_config(ROOT / "configs" / "main.yaml")
    digest = config_hash(config)
    reward_paths = []
    rollout_paths = []
    for index, seed in enumerate(config["run"]["seeds"], start=1):
        reward_methods = {
            method: {
                "test": {
                    metric: float(index + (0 if method == "MLE-RM" else 2))
                    for metric in config["evaluation"]["reward_fit_metrics"]
                }
            }
            for method in ("MLE-RM", "Pro-RM")
        }
        local = {
            str(beta): {
                method: {
                    metric: float(index + (1 if method == "pro_rm" else 0))
                    for metric in config["evaluation"]["local_policy_metrics"]
                }
                for method in ("pi0", "mle_rm", "pro_rm", "oracle")
            }
            for beta in config["policy_update"]["beta_grid"]
        }
        rollout = {
            str(beta): {
                method: {
                    metric: float(index + (1 if method == "pro_rm" else 0))
                    for metric in config["evaluation"]["rollout"]["metrics"]
                }
                for method in ("pi0", "mle_rm", "pro_rm", "oracle")
            }
            for beta in config["policy_update"]["beta_grid"]
        }
        common = {"protocol": PROTOCOL, "config_sha256": digest, "seed": seed}
        reward_paths.append(
            _write(
                tmp_path / f"reward-{seed}.json",
                {**common, "evaluation": reward_methods, "local_policy_evaluation": local},
            )
        )
        rollout_paths.append(
            _write(tmp_path / f"rollout-{seed}.json", {**common, "metrics": rollout})
        )

    result = aggregate_results(config, reward_paths, rollout_paths)
    assert result["reward_fit"]["MLE-RM"]["pair_kl"]["mean"] == pytest.approx(2.0)
    assert result["reward_fit_pro_minus_mle"]["pair_kl"]["per_seed"] == [2.0, 2.0, 2.0]
    assert result["local_policy_pro_minus_mle"]["1.0"]["local_regret"]["mean"] == 1.0
    assert result["rollout_policy_pro_minus_mle"]["4.0"]["forward_kl"]["mean"] == 1.0
    assert result["inference_scope"] == "descriptive_three_seed_experiment"


def test_aggregation_rejects_missing_seed(tmp_path) -> None:
    config = load_config(ROOT / "configs" / "main.yaml")
    payload = {
        "protocol": PROTOCOL,
        "config_sha256": config_hash(config),
        "seed": config["run"]["seeds"][0],
    }
    path = _write(tmp_path / "one.json", payload)
    with pytest.raises(ValueError, match="every configured seed"):
        aggregate_results(config, [path], [path])
