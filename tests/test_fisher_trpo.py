from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

import smart_reward.fisher_crossfit as crossfit_module
import smart_reward.trpo_run as trpo_run_module
from smart_reward.artifacts import save_exact_delta_artifact
from smart_reward.config import ConfigError, load_config, validate_config
from smart_reward.exact import ExactDeltaExperiment, ExactSplitData
from smart_reward.fisher_crossfit import (
    crossfit_fisher_regularization,
    prompt_fold_assignment,
    select_fisher_regularization,
)
from smart_reward.kl_calibration import (
    _scaled_adapter_copy,
    next_quadratic_ratio_scale,
    summarize_prompt_kl,
)
from smart_reward.policy_update import scale_direction_to_quadratic_kl
from smart_reward.prompts import prepare_multipref_prompts
from smart_reward.trpo_run import run_trpo_reward_comparison

ROOT = Path(__file__).parents[1]


def _split(seed: int, prompts: int = 20) -> ExactSplitData:
    generator = torch.Generator().manual_seed(seed)
    scores = torch.randn(prompts, 6, 5, generator=generator, dtype=torch.float64)
    rewards = (
        0.7 * scores[..., 0]
        - 0.4 * scores[..., 1]
        + 0.2 * torch.randn(prompts, 6, generator=generator, dtype=torch.float64)
    )
    return ExactSplitData(
        prompt_ids=tuple(f"prompt-{index:03d}" for index in range(prompts)),
        policy_scores=scores,
        reward_features=torch.randn(
            prompts,
            6,
            3,
            generator=generator,
            dtype=torch.float64,
        ),
        true_rewards=rewards,
    )


def test_fresh_test_offsets_skip_legacy_test_without_changing_train_or_validation() -> None:
    rows = [{"prompt_id": f"id-{index:03d}", "text": f"text {index}"} for index in range(40)]
    legacy = prepare_multipref_prompts(
        rows,
        split_sizes={"train": 8, "validation": 4, "test": 4},
        seed=17,
    )
    fresh = prepare_multipref_prompts(
        rows,
        split_sizes={"train": 8, "validation": 4, "test": 4},
        split_offsets={"train": 0, "validation": 8, "test": 16},
        seed=17,
    )
    assert [record.prompt_id for record in fresh[:12]] == [
        record.prompt_id for record in legacy[:12]
    ]
    legacy_test = {record.prompt_id for record in legacy if record.split == "test"}
    fresh_test = {record.prompt_id for record in fresh if record.split == "test"}
    assert legacy_test.isdisjoint(fresh_test)


def test_fisher_trpo_configs_are_closed_and_share_scientific_choices() -> None:
    main = load_config(ROOT / "configs" / "fisher_trpo_main.yaml")
    smoke = load_config(ROOT / "configs" / "fisher_trpo_smoke.yaml")
    assert main["policy_update"]["kl_targets"] == [3.0e-4, 1.0e-3, 3.0e-3]
    assert main["geometry"]["damping_selection"]["relative_candidates"] == [0.1, 1.0, 10.0]
    assert main["run"]["split_offsets"]["test"] == 4096
    assert smoke["policy_update"]["kl_targets"] == main["policy_update"]["kl_targets"]
    invalid = copy.deepcopy(main)
    invalid["policy_update"]["calibration"]["max_attempts"] = 0
    with pytest.raises(ConfigError, match="max_attempts"):
        validate_config(invalid)


def test_trpo_scaling_uses_raw_fisher_and_hits_target() -> None:
    matrix = torch.diag(torch.tensor([2.0, 5.0], dtype=torch.float64))
    direction = torch.tensor([1.0, -2.0], dtype=torch.float64)
    update, scale, realized = scale_direction_to_quadratic_kl(
        direction,
        lambda value: matrix @ value,
        kl_target=1.0e-3,
    )
    assert scale > 0.0
    assert realized == pytest.approx(1.0e-3, rel=1.0e-12)
    assert 0.5 * torch.dot(update, matrix @ update).item() == pytest.approx(1.0e-3)


def test_prompt_folds_are_balanced_and_independent_of_input_order() -> None:
    prompt_ids = [f"prompt-{index}" for index in range(23)]
    assignment = dict(zip(prompt_ids, prompt_fold_assignment(prompt_ids, 5), strict=True))
    reversed_ids = list(reversed(prompt_ids))
    reversed_assignment = dict(
        zip(reversed_ids, prompt_fold_assignment(reversed_ids, 5), strict=True)
    )
    assert assignment == reversed_assignment
    sizes = [sum(value == fold for value in assignment.values()) for fold in range(5)]
    assert max(sizes) - min(sizes) <= 1


def test_crossfit_is_train_only_finite_and_hits_fit_side_kl() -> None:
    result = crossfit_fisher_regularization(
        _split(11),
        folds=5,
        relative_candidates=[0.1, 1.0, 10.0],
        kl_target=1.0e-3,
        cg_tolerance=1.0e-9,
        cg_max_iterations=100,
        residual_recompute_interval=10,
    )
    assert len(result["results"]) == 3
    for candidate in result["results"].values():
        assert len(candidate["folds"]) == 5
        for fold in candidate["folds"]:
            assert fold["fit_quadratic_forward_kl"] == pytest.approx(1.0e-3)
            assert fold["pcg_relative_residual"] <= 1.0e-9
            assert all(
                torch.isfinite(torch.tensor(float(value)))
                for key, value in fold.items()
                if key not in {"fold", "fit_prompts", "heldout_prompts"}
            )


def test_global_selection_requires_positive_each_seed_and_uses_larger_one_se(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config(ROOT / "configs" / "fisher_trpo_main.yaml")
    digest = crossfit_module.config_hash(config)
    means = {
        20261001: [0.10, 0.12, 0.115],
        20261002: [0.09, 0.11, 0.105],
        20261003: [0.11, 0.13, 0.125],
    }
    paths = []
    for seed, values in means.items():
        payload = {
            "schema": crossfit_module.CROSSFIT_SCHEMA,
            "protocol": config["protocol"],
            "config_sha256": digest,
            "seed": seed,
            "fit_split": "train",
            "fold_assignment_sha256": "same-folds",
            "results": {
                str(candidate): {"mean": {"heldout_oracle_reward_improvement": value}}
                for candidate, value in zip([0.1, 1.0, 10.0], values, strict=True)
            },
        }
        path = tmp_path / f"{seed}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths.append(path)
    monkeypatch.setattr(
        crossfit_module,
        "producer_identity",
        lambda: {
            "git_commit": "0" * 40,
            "image_sha256": "1" * 64,
            "hf_inventory_sha256": "2" * 64,
        },
    )
    result = select_fisher_regularization(config, paths, tmp_path / "selection.json")
    assert result["best_mean_candidate"] == 1.0
    assert result["selected_relative_damping"] == 10.0


def test_trpo_reward_run_refits_pro_and_builds_all_matched_kl_updates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config(ROOT / "configs" / "fisher_trpo_smoke.yaml")
    digest = crossfit_module.config_hash(config)
    train = _split(31, 16)
    validation_raw = _split(32, 4)
    test_raw = _split(33, 4)
    validation = ExactSplitData(
        prompt_ids=tuple(f"validation-{value}" for value in validation_raw.prompt_ids),
        policy_scores=validation_raw.policy_scores,
        reward_features=validation_raw.reward_features,
        true_rewards=validation_raw.true_rewards,
    )
    test = ExactSplitData(
        prompt_ids=tuple(f"test-{value}" for value in test_raw.prompt_ids),
        policy_scores=test_raw.policy_scores,
        reward_features=test_raw.reward_features,
        true_rewards=test_raw.true_rewards,
    )
    experiment = ExactDeltaExperiment(train, validation, test)
    artifact = tmp_path / "artifact"
    save_exact_delta_artifact(
        experiment,
        artifact,
        config_hash=digest,
        seed=20261001,
    )
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "schema": crossfit_module.SELECTION_SCHEMA,
                "protocol": config["protocol"],
                "config_sha256": digest,
                "selected_relative_damping": 1.0,
            }
        ),
        encoding="utf-8",
    )
    identity = {
        "git_commit": "0" * 40,
        "image_sha256": "1" * 64,
        "hf_inventory_sha256": "2" * 64,
    }
    monkeypatch.setattr(trpo_run_module, "producer_identity", lambda: identity)
    result = run_trpo_reward_comparison(
        config,
        artifact,
        selection,
        tmp_path / "reward.json",
        seed=20261001,
        device="cpu",
    )
    assert result["selected_relative_damping"] == 1.0
    assert result["fit_provenance"]["MLE-RM"] == {"mode": "computed"}
    assert set(result["policy_updates"]) == {"mle_rm", "pro_rm", "oracle"}
    for updates in result["policy_updates"].values():
        assert set(updates) == {"0.0003", "0.001", "0.003"}
        for target, record in updates.items():
            assert record["train_quadratic_forward_kl"] == pytest.approx(float(target))


def test_validation_kl_gate_is_prompt_clustered_and_fail_closed() -> None:
    accepted = summarize_prompt_kl(
        [9.0e-4, 1.0e-3, 1.1e-3, 1.0e-3],
        kl_target=1.0e-3,
        point_relative_interval=[0.8, 1.2],
        confidence_level=0.95,
        upper_confidence_multiplier=1.5,
    )
    assert accepted["accepted"]
    rejected = summarize_prompt_kl(
        [1.3e-3, 1.4e-3, 1.5e-3, 1.6e-3],
        kl_target=1.0e-3,
        point_relative_interval=[0.8, 1.2],
        confidence_level=0.95,
        upper_confidence_multiplier=1.5,
    )
    assert not rejected["accepted"]


def test_quadratic_ratio_calibration_step_is_bounded() -> None:
    assert next_quadratic_ratio_scale(1.0, 4.0e-3, kl_target=1.0e-3) == pytest.approx(0.5)
    assert next_quadratic_ratio_scale(
        1.0,
        1.0e-9,
        kl_target=1.0e-3,
        max_scale_change=4.0,
    ) == pytest.approx(4.0)


def test_calibrated_adapter_copy_scales_only_lora_b(tmp_path: Path) -> None:
    import safetensors.torch

    source = tmp_path / "source"
    source.mkdir()
    (source / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    safetensors.torch.save_file(
        {
            "layer.lora_A.weight": torch.tensor([[2.0]], dtype=torch.float32),
            "layer.lora_B.weight": torch.tensor([[3.0]], dtype=torch.float32),
        },
        str(source / "adapter_model.safetensors"),
    )
    target = tmp_path / "target"
    files = _scaled_adapter_copy(source, target, scale=0.5)
    result = safetensors.torch.load_file(str(target / "adapter_model.safetensors"))
    assert torch.equal(result["layer.lora_A.weight"], torch.tensor([[2.0]]))
    assert torch.equal(result["layer.lora_B.weight"], torch.tensor([[1.5]]))
    assert set(files) == {"adapter_config.json", "adapter_model.safetensors"}
