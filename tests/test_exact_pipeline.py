from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

import smart_reward.evaluation as evaluation_module
import smart_reward.exact as exact_module
import smart_reward.exact_run as exact_run_module
from smart_reward.artifacts import save_exact_delta_artifact
from smart_reward.config import config_hash, load_config
from smart_reward.evaluation import (
    GeometrySettings,
    evaluate_local_policy,
    evaluate_reference_policy,
    solve_natural_direction,
    summarize_rollouts,
    validate_natural_direction,
)
from smart_reward.exact import (
    ExactDeltaExperiment,
    ExactSplitData,
    MLETrainingConfig,
    ProTrainingConfig,
    empirical_fisher_score_rows,
    evaluate_reward_head,
    fit_mle_reward,
    fit_pro_reward,
    pair_indices,
    pairwise_differences,
    policy_reward_moment,
    validate_pro_reward_fit,
)
from smart_reward.exact_run import load_exact_reward_comparison, run_exact_reward_comparison
from smart_reward.linear import DampedEmpiricalFisher


def make_split(seed: int, prompts: int = 18) -> ExactSplitData:
    generator = torch.Generator().manual_seed(seed)
    scores = torch.randn(prompts, 6, 4, generator=generator, dtype=torch.float64)
    features = torch.randn(prompts, 6, 2, generator=generator, dtype=torch.float64)
    rewards = (
        1.5 * features[..., 0]
        - 0.7 * features[..., 1]
        + 0.35 * scores[..., 2]
        + 0.1 * torch.randn(prompts, 6, generator=generator, dtype=torch.float64)
    )
    return ExactSplitData(
        prompt_ids=tuple(f"prompt-{seed}-{index}" for index in range(prompts)),
        policy_scores=scores,
        reward_features=features,
        true_rewards=rewards,
    )


def make_experiment() -> ExactDeltaExperiment:
    return ExactDeltaExperiment(make_split(1), make_split(2, 6), make_split(3, 6))


def settings(estimator: str = "raw_second_moment") -> GeometrySettings:
    return GeometrySettings(estimator, 1.0e-2, 1.0e-9, 100, 10)


def test_six_candidates_produce_fifteen_edges() -> None:
    assert pair_indices(6).shape == (15, 2)
    values = torch.arange(12).reshape(2, 6)
    assert pairwise_differences(values).shape == (2, 15)


def test_fisher_estimators_have_documented_normalization() -> None:
    split = make_split(4, 5)
    raw = empirical_fisher_score_rows(split.policy_scores, "raw_second_moment")
    centered = empirical_fisher_score_rows(split.policy_scores, "prompt_centered_sample_covariance")
    assert raw.shape == centered.shape == (30, 4)
    centered_matrix = centered.mT @ centered / centered.shape[0]
    manual = (
        sum(torch.cov(split.policy_scores[index].mT) for index in range(split.num_prompts))
        / split.num_prompts
    )
    assert torch.allclose(centered_matrix, manual)


def test_fisher_pcg_preconditioner_respects_low_rank_ridge_structure() -> None:
    underdetermined = DampedEmpiricalFisher(torch.randn(3, 5, dtype=torch.float64), damping=1.0e-3)
    full_rank_capable = DampedEmpiricalFisher(
        torch.randn(5, 3, dtype=torch.float64), damping=1.0e-3
    )
    assert underdetermined.pcg_inverse_diagonal() is None
    assert torch.equal(
        full_rank_capable.pcg_inverse_diagonal(),
        full_rank_capable.inverse_diagonal(),
    )


def test_policy_reward_moment_is_prompt_shift_invariant() -> None:
    split = make_split(5)
    shifts = torch.randn(split.num_prompts, 1, dtype=torch.float64)
    original = policy_reward_moment(split.policy_scores, split.true_rewards)
    shifted = policy_reward_moment(split.policy_scores, split.true_rewards + shifts)
    assert torch.allclose(original, shifted, atol=1.0e-12)


def test_mle_and_pro_fit_exact_delta_targets() -> None:
    split = make_split(6, 24)
    mle = fit_mle_reward(
        split,
        MLETrainingConfig(
            max_iterations=100,
            history_size=20,
            gradient_tolerance=1.0e-6,
            change_tolerance=1.0e-12,
            microbatch_size=64,
        ),
    )
    pro = fit_pro_reward(
        split,
        ProTrainingConfig(
            relative_damping=1.0e-2,
            fisher_estimator="raw_second_moment",
            inner_max_iterations=100,
            inner_tolerance=1.0e-9,
            outer_max_iterations=100,
            outer_tolerance=1.0e-9,
            residual_recompute_interval=10,
        ),
    )
    assert mle.method == "MLE-RM"
    assert pro.method == "Pro-RM"
    assert mle.weight.shape == pro.weight.shape == (2,)
    assert pro.converged


def test_pro_fit_validation_recomputes_gate_without_outer_solve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    split = make_split(7, 24)
    config = ProTrainingConfig(
        relative_damping=1.0e-2,
        fisher_estimator="raw_second_moment",
        inner_max_iterations=100,
        inner_tolerance=1.0e-9,
        outer_max_iterations=100,
        outer_tolerance=1.0e-8,
        residual_recompute_interval=10,
    )
    source = fit_pro_reward(split, config)
    right_hand_side_sizes: list[int] = []
    original_pcg = exact_module.pcg

    def recording_pcg(matvec, rhs, **kwargs):
        right_hand_side_sizes.append(rhs.numel())
        return original_pcg(matvec, rhs, **kwargs)

    monkeypatch.setattr(exact_module, "pcg", recording_pcg)
    validated = validate_pro_reward_fit(
        split,
        source.weight,
        config,
        source_iterations=source.iterations,
        source_inner_pcg_calls=source.inner_pcg_calls,
    )

    assert validated.converged
    assert validated.head_sha256 == source.head_sha256
    assert validated.relative_residual is not None
    assert validated.relative_residual <= config.outer_tolerance
    assert right_hand_side_sizes == [split.policy_dimension, split.policy_dimension]


def test_mle_uses_identifiable_coordinates_for_underdetermined_head() -> None:
    generator = torch.Generator().manual_seed(17)
    prompts, candidates, reward_dimension = 4, 6, 64
    features = torch.randn(
        prompts,
        candidates,
        reward_dimension,
        generator=generator,
        dtype=torch.float64,
    )
    oracle_weight = 0.1 * torch.randn(
        reward_dimension,
        generator=generator,
        dtype=torch.float64,
    )
    split = ExactSplitData(
        prompt_ids=tuple(f"underdetermined-{index}" for index in range(prompts)),
        policy_scores=torch.randn(
            prompts,
            candidates,
            3,
            generator=generator,
            dtype=torch.float64,
        ),
        reward_features=features,
        true_rewards=features @ oracle_weight,
    )

    result = fit_mle_reward(
        split,
        MLETrainingConfig(
            max_iterations=20,
            history_size=10,
            gradient_tolerance=1.0e-6,
            change_tolerance=1.0e-10,
            microbatch_size=32,
        ),
    )

    design, _ = exact_module._edge_design(split)
    projected = torch.linalg.lstsq(design, design @ result.weight).solution
    assert result.converged
    assert result.gradient_norm <= 1.0e-6
    assert torch.allclose(result.weight, projected, atol=1.0e-8, rtol=1.0e-8)


def test_mle_full_rank_coordinates_are_exact_and_empirically_whitened() -> None:
    generator = torch.Generator().manual_seed(23)
    design = torch.randn(40, 5, generator=generator, dtype=torch.float64)
    design = design * torch.tensor(
        [1.0e-4, 1.0e-2, 1.0, 10.0, 100.0],
        dtype=torch.float64,
    )

    coordinates, head_map = exact_module._mle_parameterization(
        design,
        maximum_identifiable_rank=design.shape[1],
        source_epsilon=torch.finfo(torch.float64).eps,
    )

    assert head_map is not None
    assert torch.allclose(design @ head_map, coordinates, atol=1.0e-9, rtol=1.0e-9)
    gram = coordinates.mT @ coordinates
    assert torch.allclose(
        gram / design.shape[0],
        torch.eye(gram.shape[0], dtype=gram.dtype),
        atol=1.0e-7,
        rtol=1.0e-10,
    )


def test_mle_converges_for_ill_conditioned_full_rank_features() -> None:
    generator = torch.Generator().manual_seed(29)
    prompts, candidates, reward_dimension = 12, 6, 5
    feature_scales = torch.tensor(
        [1.0e-3, 3.0e-2, 1.0, 3.0e1, 1.0e3],
        dtype=torch.float64,
    )
    features = (
        torch.randn(
            prompts,
            candidates,
            reward_dimension,
            generator=generator,
            dtype=torch.float64,
        )
        * feature_scales
    )
    oracle_weight = torch.tensor([0.3, -0.2, 0.1, 0.2, -0.3], dtype=torch.float64) / feature_scales
    split = ExactSplitData(
        prompt_ids=tuple(f"full-rank-{index}" for index in range(prompts)),
        policy_scores=torch.randn(
            prompts,
            candidates,
            3,
            generator=generator,
            dtype=torch.float64,
        ),
        reward_features=features,
        true_rewards=features @ oracle_weight,
    )

    result = fit_mle_reward(
        split,
        MLETrainingConfig(
            max_iterations=50,
            history_size=20,
            gradient_tolerance=1.0e-7,
            change_tolerance=1.0e-12,
            microbatch_size=64,
        ),
    )

    assert result.converged
    assert result.gradient_norm <= 1.0e-7


def test_mle_uses_solve_precision_for_float32_full_rank_artifacts() -> None:
    generator = torch.Generator().manual_seed(31)
    prompts, candidates, reward_dimension = 20, 6, 5
    feature_scales = torch.tensor(
        [1.0e-4, 1.0e-2, 1.0, 10.0, 100.0],
        dtype=torch.float32,
    )
    features = (
        torch.randn(
            prompts,
            candidates,
            reward_dimension,
            generator=generator,
            dtype=torch.float32,
        )
        * feature_scales
    )
    oracle_weight = torch.tensor([0.3, -0.2, 0.1, 0.2, -0.3], dtype=torch.float32) / feature_scales
    split = ExactSplitData(
        prompt_ids=tuple(f"float32-full-rank-{index}" for index in range(prompts)),
        policy_scores=torch.randn(
            prompts,
            candidates,
            3,
            generator=generator,
            dtype=torch.float32,
        ),
        reward_features=features,
        true_rewards=features @ oracle_weight,
    )

    result = fit_mle_reward(
        split,
        MLETrainingConfig(
            max_iterations=200,
            history_size=20,
            gradient_tolerance=1.0e-7,
            change_tolerance=1.0e-12,
            microbatch_size=64,
        ),
    )

    assert result.converged
    assert result.gradient_norm <= 1.0e-7


def test_mle_does_not_stop_on_tiny_scaled_coordinate_steps() -> None:
    generator = torch.Generator().manual_seed(29)
    prompts, candidates, reward_dimension = 20, 6, 5
    feature_scales = torch.logspace(
        -4,
        4,
        reward_dimension,
        dtype=torch.float64,
    )
    features = (
        torch.randn(
            prompts,
            candidates,
            reward_dimension,
            generator=generator,
            dtype=torch.float64,
        )
        * feature_scales
    )
    oracle_weight = torch.tensor([0.3, -0.2, 0.1, 0.2, -0.3], dtype=torch.float64) / feature_scales
    split = ExactSplitData(
        prompt_ids=tuple(f"scaled-step-{index}" for index in range(prompts)),
        policy_scores=torch.randn(
            prompts,
            candidates,
            3,
            generator=generator,
            dtype=torch.float64,
        ),
        reward_features=features,
        true_rewards=features @ oracle_weight,
    )

    result = fit_mle_reward(
        split,
        MLETrainingConfig(
            max_iterations=200,
            history_size=20,
            gradient_tolerance=1.0e-7,
            change_tolerance=1.0e-12,
            microbatch_size=64,
        ),
    )

    assert result.converged
    assert result.gradient_norm <= 1.0e-7


def test_pro_nested_solve_tightens_inner_accuracy_and_preconditions_outer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[int, float, bool]] = []
    original_pcg = exact_module.pcg

    def recording_pcg(matvec, rhs, inverse_diagonal=None, x0=None, **kwargs):
        observed.append((rhs.numel(), float(kwargs["tolerance"]), inverse_diagonal is not None))
        return original_pcg(
            matvec,
            rhs,
            inverse_diagonal=inverse_diagonal,
            x0=x0,
            **kwargs,
        )

    monkeypatch.setattr(exact_module, "pcg", recording_pcg)
    result = fit_pro_reward(
        make_split(14, 24),
        ProTrainingConfig(
            relative_damping=1.0e-2,
            fisher_estimator="raw_second_moment",
            inner_max_iterations=100,
            inner_tolerance=1.0e-5,
            outer_max_iterations=100,
            outer_tolerance=1.0e-6,
            residual_recompute_interval=10,
        ),
    )

    inner_calls = [call for call in observed if call[0] == 4]
    outer_calls = [call for call in observed if call[0] == 2]
    assert inner_calls and all(call[1] == pytest.approx(1.0e-7) for call in inner_calls)
    assert outer_calls == [(2, pytest.approx(1.0e-6), True)]
    assert result.effective_inner_tolerance == pytest.approx(1.0e-7)
    assert result.relative_residual is not None
    assert result.relative_residual <= 1.0e-6


def test_reward_metrics_are_prompt_aggregated_and_finite() -> None:
    split = make_split(7)
    result = evaluate_reward_head(split, torch.tensor([1.5, -0.7], dtype=torch.float64))
    assert result.pair_kl >= -1.0e-12
    assert result.probability_mse >= 0.0
    assert 0.0 <= result.pairwise_accuracy <= 1.0
    assert result.centered_reward_nmse >= 0.0
    assert result.centered_reward_nmse_defined_fraction == 1.0
    assert all(math.isfinite(value) for value in result.to_dict().values())


def test_centered_reward_nmse_records_and_skips_zero_energy_prompts() -> None:
    split = make_split(31, 7)
    rewards = split.true_rewards.clone()
    rewards[0] = 2.0
    with_zero_energy = ExactSplitData(
        prompt_ids=split.prompt_ids,
        policy_scores=split.policy_scores,
        reward_features=split.reward_features,
        true_rewards=rewards,
    )

    result = evaluate_reward_head(
        with_zero_energy,
        torch.tensor([1.5, -0.7], dtype=torch.float64),
    )

    assert result.centered_reward_nmse >= 0.0
    assert result.centered_reward_nmse_defined_fraction == pytest.approx(6.0 / 7.0)
    assert all(math.isfinite(value) for value in result.to_dict().values())


def test_centered_reward_nmse_rejects_fully_degenerate_oracle() -> None:
    split = make_split(32, 3)
    degenerate = ExactSplitData(
        prompt_ids=split.prompt_ids,
        policy_scores=split.policy_scores,
        reward_features=split.reward_features,
        true_rewards=torch.ones_like(split.true_rewards),
    )

    with pytest.raises(ValueError, match="positive energy for some prompt"):
        evaluate_reward_head(
            degenerate,
            torch.tensor([1.5, -0.7], dtype=torch.float64),
        )


def test_oracle_direction_has_zero_local_regret() -> None:
    split = make_split(8)
    direction = solve_natural_direction(split, split.true_rewards, settings())
    result = evaluate_local_policy(
        split,
        direction,
        beta=2.0,
        settings=settings(),
    )
    assert result.local_regret == pytest.approx(0.0, abs=1.0e-12)
    assert result.fisher_cosine == pytest.approx(1.0, abs=1.0e-9)
    assert result.tabular_regret >= -1.0e-12


def test_beta_scales_local_regret_inverse_linearly() -> None:
    split = make_split(9)
    predicted = split.reward_features @ torch.tensor([1.0, -0.2], dtype=torch.float64)
    direction = solve_natural_direction(split, predicted, settings())
    first = evaluate_local_policy(split, direction, beta=1.0, settings=settings())
    fourth = evaluate_local_policy(split, direction, beta=4.0, settings=settings())
    assert fourth.local_regret == pytest.approx(first.local_regret / 4.0)


def test_reference_policy_is_a_valid_fourth_policy_family() -> None:
    result = evaluate_reference_policy(make_split(10), beta=2.0, settings=settings())
    assert result.local_regret >= 0.0
    assert result.fisher_cosine == 0.0


def test_natural_direction_is_beta_free() -> None:
    split = make_split(11)
    direction = solve_natural_direction(split, split.true_rewards, settings())
    assert direction.shape == (split.policy_dimension,)
    assert torch.allclose(direction / 4.0, direction * 0.25)


def test_saved_natural_direction_is_validated_without_repeating_pcg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    split = make_split(34)
    geometry = settings()
    direction = solve_natural_direction(split, split.true_rewards, geometry)

    def unexpected_pcg(*args, **kwargs):
        del args, kwargs
        raise AssertionError("direction validation must not repeat PCG")

    monkeypatch.setattr(evaluation_module, "pcg", unexpected_pcg)
    residual = validate_natural_direction(
        split,
        split.true_rewards,
        direction,
        geometry,
    )

    assert residual <= geometry.cg_tolerance
    with pytest.raises(RuntimeError, match="residual gate"):
        validate_natural_direction(
            split,
            split.true_rewards,
            direction + 1.0,
            geometry,
        )


def test_reward_retry_reuses_independent_validated_components(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(Path("configs/main.yaml"))
    seed = 20261001
    artifact = tmp_path / "artifact"
    save_exact_delta_artifact(
        make_experiment(),
        artifact,
        config_hash=config_hash(config),
        seed=seed,
    )
    source_identity = {
        "git_commit": "a" * 40,
        "image_sha256": "b" * 64,
        "hf_inventory_sha256": "c" * 64,
    }
    monkeypatch.setattr(
        exact_run_module,
        "producer_identity",
        lambda: source_identity,
    )
    source_path = tmp_path / "source.json"
    source = run_exact_reward_comparison(
        config,
        artifact,
        source_path,
        seed=seed,
        device="cpu",
    )

    def unexpected_pro_fit(*args, **kwargs):
        del args, kwargs
        raise AssertionError("validated Pro-RM fit must not be recomputed")

    original_direction_solve = exact_run_module.solve_natural_direction
    direction_solve_calls = 0

    def record_direction_solve(*args, **kwargs):
        nonlocal direction_solve_calls
        direction_solve_calls += 1
        return original_direction_solve(*args, **kwargs)

    monkeypatch.setattr(exact_run_module, "fit_pro_reward", unexpected_pro_fit)
    monkeypatch.setattr(
        exact_run_module,
        "solve_natural_direction",
        record_direction_solve,
    )
    monkeypatch.setattr(
        exact_run_module,
        "producer_identity",
        lambda: {
            "git_commit": "d" * 40,
            "image_sha256": "e" * 64,
            "hf_inventory_sha256": "f" * 64,
        },
    )
    retried = run_exact_reward_comparison(
        config,
        artifact,
        tmp_path / "retried.json",
        seed=seed,
        device="cpu",
        reuse_pro_from=source_path,
    )

    assert direction_solve_calls == 1
    assert retried["methods"]["Pro-RM"] == source["methods"]["Pro-RM"]
    assert retried["evaluation"]["Pro-RM"] == source["evaluation"]["Pro-RM"]
    for method in ("pro_rm", "oracle"):
        assert retried["policy_directions"][method] == source["policy_directions"][method]
    for beta in config["policy_update"]["beta_grid"]:
        for method in ("pi0", "pro_rm", "oracle"):
            assert (
                retried["local_policy_evaluation"][str(beta)][method]
                == source["local_policy_evaluation"][str(beta)][method]
            )
    provenance = retried["component_provenance"]
    assert provenance["reward_fit"]["Pro-RM"]["mode"] == "validated_reuse"
    assert provenance["reward_evaluation"]["Pro-RM"]["mode"] == "validated_reuse"
    assert provenance["natural_direction"]["pro_rm"]["mode"] == "validated_reuse"
    assert provenance["natural_direction"]["oracle"]["mode"] == "validated_reuse"
    assert provenance["natural_direction"]["mle_rm"] == {"mode": "computed"}
    assert (
        load_exact_reward_comparison(
            tmp_path / "retried.json",
            expected_config_hash=config_hash(config),
            expected_seed=seed,
        )
        == retried
    )

    tampered = json.loads(source_path.read_text(encoding="utf-8"))
    tampered["evaluation"]["Pro-RM"]["test"]["pair_kl"] += 1.0
    tampered_path = tmp_path / "tampered-source.json"
    tampered_path.write_text(json.dumps(tampered) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="numerical mismatch"):
        run_exact_reward_comparison(
            config,
            artifact,
            tmp_path / "tampered-retry.json",
            seed=seed,
            device="cpu",
            reuse_pro_from=tampered_path,
        )


def test_held_out_evaluation_does_not_refit_the_direction() -> None:
    train = make_split(12)
    test = make_split(13)
    train_direction = solve_natural_direction(train, train.true_rewards, settings())
    test_direction = solve_natural_direction(test, test.true_rewards, settings())
    deployed = evaluate_local_policy(test, train_direction, beta=2.0, settings=settings())
    refitted = evaluate_local_policy(test, test_direction, beta=2.0, settings=settings())
    assert deployed.local_regret > 0.0
    assert refitted.local_regret == pytest.approx(0.0, abs=1.0e-12)


def test_rollout_summary_uses_forward_log_ratio_and_prompt_unit() -> None:
    reward = torch.tensor([[2.0, 0.0], [1.0, 3.0]])
    ratio = torch.tensor([[0.2, 0.0], [0.1, 0.1]])
    reference = torch.zeros_like(reward)
    oracle_reward = reward + 1.0
    oracle_ratio = ratio
    result = summarize_rollouts(
        reward,
        ratio,
        beta=2.0,
        reference_oracle_rewards=reference,
        oracle_ngd_oracle_rewards=oracle_reward,
        oracle_ngd_forward_log_ratios=oracle_ratio,
    )
    assert result["oracle_reward"] == pytest.approx(1.5)
    assert result["forward_kl"] == pytest.approx(0.1)
    assert result["regularized_utility"] == pytest.approx(1.3)
    assert result["oracle_ngd_regret"] == pytest.approx(1.0)
    assert result["sampling_unit"] == "prompt"


def test_rollout_summary_requires_all_shapes_to_match() -> None:
    values = torch.zeros(2, 3)
    with pytest.raises(ValueError, match="identical shapes"):
        summarize_rollouts(
            values,
            values,
            beta=1.0,
            reference_oracle_rewards=torch.zeros(2, 2),
            oracle_ngd_oracle_rewards=values,
            oracle_ngd_forward_log_ratios=values,
        )
