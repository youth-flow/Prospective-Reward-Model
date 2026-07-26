import copy
import math

import pytest
import torch
import torch.nn.functional as functional

import smart_reward.training as training_module
from smart_reward.baseline import repeated_btl_nll
from smart_reward.objective import empirical_moment
from smart_reward.training import (
    BTMLETrainer,
    BTMLETrainingConfig,
    FeatureTrainingBatch,
    FrozenFeatureLinearReward,
    ProRMPlusTrainer,
    ProRMPlusTrainingConfig,
    SRMPlusTrainer,
    SRMPlusTrainingConfig,
    evaluate_bt_mle,
    evaluate_prorm_plus,
    evaluate_srm_plus,
)


def _misspecified_toy(dtype: torch.dtype = torch.float64) -> FeatureTrainingBatch:
    """A true edge target outside the two-dimensional reward feature span."""

    feature_differences = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, -1.0],
            [-1.0, 2.0],
            [1.5, 0.5],
            [-0.5, 1.5],
            [2.0, 1.0],
        ],
        dtype=dtype,
    )
    true_target = torch.tensor(
        [1.2, -0.8, 0.5, 1.5, -1.2, 0.7, -0.9, 1.1],
        dtype=dtype,
    )
    # Assert that this really is a restricted/misspecified reward class.
    least_squares = torch.linalg.lstsq(feature_differences, true_target).solution
    assert not torch.allclose(feature_differences @ least_squares, true_target)
    edge_scores = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, -1.0],
            [-1.0, 1.0],
            [2.0, 0.5],
            [0.5, 2.0],
            [-1.0, -1.0],
        ],
        dtype=dtype,
    )
    node_scores = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0], [-1.0, 2.0], [2.0, 1.0]],
        dtype=dtype,
    )
    counts = torch.tensor([17, 23, 19, 21, 18, 25, 22, 20], dtype=torch.int64)
    wins = torch.round(torch.sigmoid(true_target) * counts).to(torch.int64)
    return FeatureTrainingBatch(
        left_features=feature_differences,
        right_features=torch.zeros_like(feature_differences),
        edge_scores=edge_scores,
        node_scores=node_scores,
        h=true_target,
        left_wins=wins,
        num_annotations=counts,
    )


def _prorm_config(*, microbatch_size: int | None = 3) -> ProRMPlusTrainingConfig:
    return ProRMPlusTrainingConfig(
        learning_rate=0.1,
        optimizer="sgd",
        microbatch_size=microbatch_size,
        beta=1.3,
        damping=0.2,
        pcg_max_iterations=20,
        pcg_tolerance=1.0e-12,
    )


def test_legacy_srm_api_names_alias_canonical_prorm_api() -> None:
    assert SRMPlusTrainer is ProRMPlusTrainer
    assert SRMPlusTrainingConfig is ProRMPlusTrainingConfig
    assert evaluate_srm_plus is evaluate_prorm_plus


def test_prorm_dual_preserves_low_rank_plus_damping_krylov_structure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_pcg = training_module.pcg
    observed_preconditioners: list[torch.Tensor | None] = []

    def recording_pcg(*args, **kwargs):
        observed_preconditioners.append(kwargs.get("inverse_diagonal"))
        return original_pcg(*args, **kwargs)

    monkeypatch.setattr(training_module, "pcg", recording_pcg)
    evaluate_prorm_plus(
        FrozenFeatureLinearReward(2, dtype=torch.float64),
        _misspecified_toy(),
        _prorm_config(),
    )

    assert observed_preconditioners == [None]


def test_frozen_feature_head_is_bias_free_zero_or_explicitly_initialized() -> None:
    zero_model = FrozenFeatureLinearReward(3, dtype=torch.float64)
    assert list(dict(zero_model.named_parameters())) == ["weight"]
    assert torch.equal(zero_model.weight, torch.zeros(3, dtype=torch.float64))

    initial = torch.tensor([[0.2, -0.4, 0.8]], dtype=torch.float64)
    initialized = FrozenFeatureLinearReward(3, initial)
    initial.add_(10.0)
    assert torch.equal(
        initialized.weight,
        torch.tensor([0.2, -0.4, 0.8], dtype=torch.float64),
    )
    features = torch.tensor([[1.0, 2.0, -1.0]], dtype=torch.float64)
    assert initialized(features).item() == pytest.approx(-1.4)


def test_tensor_bundle_enforces_frozen_dtype_shape_counts_and_orientation() -> None:
    batch = _misspecified_toy()
    assert batch.Z is batch.edge_scores
    assert batch.S is batch.node_scores
    assert batch.N is batch.num_annotations

    values = batch.__dict__.copy()
    values["orientation"] = "right_minus_left"
    with pytest.raises(ValueError, match="left_minus_right"):
        FeatureTrainingBatch(**values)

    values = batch.__dict__.copy()
    values["h"] = batch.h.float()
    with pytest.raises(ValueError, match="same floating dtype"):
        FeatureTrainingBatch(**values)

    values = batch.__dict__.copy()
    values["left_features"] = batch.left_features.clone().requires_grad_(True)
    with pytest.raises(ValueError, match="frozen"):
        FeatureTrainingBatch(**values)

    values = batch.__dict__.copy()
    invalid_wins = batch.left_wins.clone()
    invalid_wins[0] = batch.num_annotations[0] + 1
    values["left_wins"] = invalid_wins
    with pytest.raises(ValueError, match="0 <= left_wins"):
        FeatureTrainingBatch(**values)


def test_bt_and_prorm_each_reduce_its_own_objective_on_misspecified_toy() -> None:
    batch = _misspecified_toy()
    shared_initialization = torch.tensor([0.0, 0.0], dtype=torch.float64)
    bt_model = FrozenFeatureLinearReward(2, shared_initialization)
    prorm_model = FrozenFeatureLinearReward(2, shared_initialization)
    assert torch.equal(bt_model.weight, prorm_model.weight)

    bt_before = evaluate_bt_mle(bt_model, batch)
    bt_trainer = BTMLETrainer(
        bt_model,
        batch,
        BTMLETrainingConfig(
            learning_rate=0.05,
            optimizer="sgd",
            microbatch_size=3,
        ),
    )
    bt_trainer.fit(100)
    bt_after = bt_trainer.evaluate()

    config = _prorm_config()
    prorm_before = evaluate_prorm_plus(prorm_model, batch, config).dual_loss
    prorm_trainer = ProRMPlusTrainer(prorm_model, batch, config)
    prorm_trainer.fit(100)
    prorm_after = prorm_trainer.evaluate().dual_loss

    assert bt_after < 0.9 * bt_before
    assert prorm_after < 0.1 * prorm_before
    # The restricted class forces the policy-aware and likelihood projections
    # to select genuinely different reward heads.
    assert not torch.allclose(bt_model.weight, prorm_model.weight, atol=0.03, rtol=0.0)


def test_one_prorm_envelope_step_matches_exact_quadratic_gradient() -> None:
    batch = _misspecified_toy()
    initial = torch.tensor([0.17, -0.31], dtype=torch.float64)
    model = FrozenFeatureLinearReward(2, initial)
    learning_rate = 0.07
    config = ProRMPlusTrainingConfig(
        learning_rate=learning_rate,
        optimizer="sgd",
        microbatch_size=3,
        beta=1.7,
        damping=0.3,
        pcg_max_iterations=10,
        pcg_tolerance=1.0e-13,
    )

    feature_differences = batch.feature_differences
    margins = feature_differences @ initial
    moment = empirical_moment(batch.edge_scores, margins, batch.h)
    fisher = batch.node_scores.mT @ batch.node_scores / batch.node_scores.shape[0]
    operator = fisher + config.damping * torch.eye(2, dtype=torch.float64)
    direction = torch.linalg.solve(operator, moment)
    expected_gradient = (
        feature_differences.mT
        @ (batch.edge_scores @ direction)
        / (2.0 * config.beta * batch.num_edges)
    )
    expected_weight = initial - learning_rate * expected_gradient

    trainer = ProRMPlusTrainer(model, batch, config)
    diagnostic = trainer.step()

    assert torch.allclose(model.weight, expected_weight, atol=2.0e-14, rtol=1.0e-13)
    assert diagnostic.gradient_norm == pytest.approx(
        torch.linalg.vector_norm(expected_gradient).item(), rel=1.0e-12
    )
    assert diagnostic.dual_loss == pytest.approx(diagnostic.objective)
    assert diagnostic.dual_saddle_value == pytest.approx(
        diagnostic.dual_loss, rel=1.0e-12, abs=1.0e-14
    )
    assert diagnostic.pcg_converged
    assert trainer.last_pcg_reason == "converged"
    assert trainer.dual_direction is not None
    assert not trainer.dual_direction.requires_grad


def test_prorm_refreshes_dual_once_before_every_optimizer_update(monkeypatch) -> None:
    batch = _misspecified_toy()
    model = FrozenFeatureLinearReward(2, dtype=torch.float64)
    optimizer = torch.optim.SGD([model.weight], lr=0.05)
    optimizer_updates = 0
    original_optimizer_step = optimizer.step

    def counted_optimizer_step(*args, **kwargs):
        nonlocal optimizer_updates
        optimizer_updates += 1
        return original_optimizer_step(*args, **kwargs)

    monkeypatch.setattr(optimizer, "step", counted_optimizer_step)
    dual_calls: list[tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]] = []
    original_solve = training_module._solve_prorm_dual

    def counted_solve(batch_arg, margins, config, warm_start, geometry=None):
        result = original_solve(batch_arg, margins, config, warm_start, geometry)
        dual_calls.append(
            (
                margins.detach().clone(),
                None if warm_start is None else warm_start.detach().clone(),
                result[2].solution.detach().clone(),
            )
        )
        return result

    monkeypatch.setattr(training_module, "_solve_prorm_dual", counted_solve)
    trainer = ProRMPlusTrainer(model, batch, _prorm_config(), optimizer=optimizer)
    diagnostics = trainer.fit(3)

    assert optimizer_updates == 3
    assert len(dual_calls) == 3
    assert trainer.dual_refreshes == 3
    assert [item.dual_refresh for item in diagnostics] == [1, 2, 3]
    assert dual_calls[0][1] is None
    assert dual_calls[1][1] is not None and dual_calls[2][1] is not None
    assert not torch.equal(dual_calls[0][0], dual_calls[1][0])
    assert torch.equal(dual_calls[1][1], dual_calls[0][2])
    assert torch.equal(dual_calls[2][1], dual_calls[1][2])


def test_prorm_uses_fp64_policy_geometry_with_fp32_reward_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = _misspecified_toy(torch.float32)
    model = FrozenFeatureLinearReward(2, dtype=torch.float32)
    original_pcg = training_module.pcg
    observed_rhs_dtypes: list[torch.dtype] = []

    def recording_pcg(*args, **kwargs):
        observed_rhs_dtypes.append(args[1].dtype)
        return original_pcg(*args, **kwargs)

    monkeypatch.setattr(training_module, "pcg", recording_pcg)
    trainer = ProRMPlusTrainer(model, batch, _prorm_config())
    diagnostic = trainer.step()

    assert observed_rhs_dtypes == [torch.float64]
    assert trainer.dual_direction is not None
    assert trainer.dual_direction.dtype == torch.float64
    assert trainer._policy_geometry.edge_scores.dtype == torch.float64
    assert trainer._policy_geometry.node_scores.dtype == torch.float64
    assert model.weight.dtype == torch.float32
    assert model.weight.grad is not None and model.weight.grad.dtype == torch.float32
    assert diagnostic.pcg_converged


def test_global_orientation_swap_preserves_both_objectives_and_updates() -> None:
    batch = _misspecified_toy()
    swapped = batch.swapped()
    initial = torch.tensor([0.13, -0.22], dtype=torch.float64)

    original_bt = FrozenFeatureLinearReward(2, initial)
    swapped_bt = FrozenFeatureLinearReward(2, initial)
    assert evaluate_bt_mle(original_bt, batch) == pytest.approx(
        evaluate_bt_mle(swapped_bt, swapped), rel=1.0e-15
    )
    bt_config = BTMLETrainingConfig(
        learning_rate=0.03,
        optimizer="sgd",
        microbatch_size=3,
    )
    BTMLETrainer(original_bt, batch, bt_config).step()
    BTMLETrainer(swapped_bt, swapped, bt_config).step()
    assert torch.allclose(original_bt.weight, swapped_bt.weight, atol=1.0e-15, rtol=0.0)

    original_prorm = FrozenFeatureLinearReward(2, initial)
    swapped_prorm = FrozenFeatureLinearReward(2, initial)
    config = _prorm_config()
    original_value = evaluate_prorm_plus(original_prorm, batch, config)
    swapped_value = evaluate_prorm_plus(swapped_prorm, swapped, config)
    assert original_value.dual_loss == pytest.approx(swapped_value.dual_loss, rel=1.0e-15)
    ProRMPlusTrainer(original_prorm, batch, config).step()
    ProRMPlusTrainer(swapped_prorm, swapped, config).step()
    assert torch.allclose(original_prorm.weight, swapped_prorm.weight, atol=1.0e-15, rtol=0.0)


def test_bt_count_objective_and_gradient_equal_expanded_binary_labels() -> None:
    margins = torch.tensor([0.4, -0.7, 1.1], dtype=torch.float64, requires_grad=True)
    wins = torch.tensor([2, 1, 3], dtype=torch.int64)
    counts = torch.tensor([3, 4, 5], dtype=torch.int64)
    count_loss = repeated_btl_nll(margins, wins, counts)
    (count_gradient,) = torch.autograd.grad(count_loss, margins)

    expanded_margins = torch.repeat_interleave(margins.detach(), counts).requires_grad_(True)
    labels = torch.cat(
        [
            torch.cat(
                [
                    torch.ones(int(win), dtype=torch.float64),
                    torch.zeros(int(total - win), dtype=torch.float64),
                ]
            )
            for win, total in zip(wins, counts, strict=True)
        ]
    )
    expanded_loss = functional.binary_cross_entropy_with_logits(expanded_margins, labels)
    (expanded_gradient,) = torch.autograd.grad(expanded_loss, expanded_margins)
    offsets = torch.cat([torch.zeros(1, dtype=torch.int64), counts.cumsum(dim=0)])
    collapsed_gradient = torch.stack(
        [
            expanded_gradient[int(offsets[index].item()) : int(offsets[index + 1].item())].sum()
            for index in range(counts.numel())
        ]
    )

    assert torch.allclose(count_loss, expanded_loss)
    assert torch.allclose(count_gradient, collapsed_gradient, atol=1.0e-15, rtol=0.0)


@pytest.mark.parametrize("kind", ["bt", "prorm_plus"])
def test_microbatch_accumulation_matches_one_full_batch_update(kind: str) -> None:
    batch = _misspecified_toy()
    initial = torch.tensor([0.11, -0.09], dtype=torch.float64)
    full_model = FrozenFeatureLinearReward(2, initial)
    micro_model = FrozenFeatureLinearReward(2, initial)

    if kind == "bt":
        full_config = BTMLETrainingConfig(learning_rate=0.04, optimizer="sgd")
        micro_config = BTMLETrainingConfig(
            learning_rate=0.04,
            optimizer="sgd",
            microbatch_size=3,
        )
        full_diagnostic = BTMLETrainer(full_model, batch, full_config).step()
        micro_diagnostic = BTMLETrainer(micro_model, batch, micro_config).step()
    else:
        full_diagnostic = ProRMPlusTrainer(
            full_model,
            batch,
            _prorm_config(microbatch_size=None),
        ).step()
        micro_diagnostic = ProRMPlusTrainer(micro_model, batch, _prorm_config()).step()

    assert torch.allclose(full_model.weight, micro_model.weight, atol=2.0e-15, rtol=1.0e-14)
    assert full_diagnostic.objective == pytest.approx(micro_diagnostic.objective, rel=1.0e-14)
    assert full_diagnostic.gradient_norm == pytest.approx(
        micro_diagnostic.gradient_norm, rel=1.0e-14
    )


@pytest.mark.parametrize("kind", ["bt", "prorm_plus"])
def test_in_memory_checkpoint_roundtrip_has_deterministic_continuation(kind: str) -> None:
    batch = _misspecified_toy()
    first_model = FrozenFeatureLinearReward(2, dtype=torch.float64)
    restored_model = FrozenFeatureLinearReward(2, dtype=torch.float64)
    if kind == "bt":
        config = BTMLETrainingConfig(
            learning_rate=0.03,
            optimizer="adamw",
            microbatch_size=3,
        )
        first = BTMLETrainer(first_model, batch, config)
        restored = BTMLETrainer(restored_model, batch, config)
    else:
        config = ProRMPlusTrainingConfig(
            learning_rate=0.03,
            optimizer="adamw",
            microbatch_size=3,
            damping=0.2,
            pcg_tolerance=1.0e-12,
        )
        first = ProRMPlusTrainer(first_model, batch, config)
        restored = ProRMPlusTrainer(restored_model, batch, config)

    first.fit(4)
    checkpoint = first.state_dict()
    assert checkpoint["format_version"] == 2
    assert checkpoint["trainer"] == ("bt_mle" if kind == "bt" else "prorm_plus")
    untouched_checkpoint = copy.deepcopy(checkpoint)
    restored.load_state_dict(checkpoint)
    if kind == "prorm_plus":
        assert restored.last_pcg_reason is None
    first.step()
    restored.step()

    assert torch.equal(first_model.weight, restored_model.weight)
    assert first.completed_steps == restored.completed_steps == 5
    assert first.history == restored.history
    # Loading and subsequent optimization must not mutate the caller-owned dict.
    assert torch.equal(checkpoint["model"]["weight"], untouched_checkpoint["model"]["weight"])
    if kind == "prorm_plus":
        assert first.last_pcg_reason == restored.last_pcg_reason == "converged"
        assert first.dual_refreshes == restored.dual_refreshes == 5
        assert torch.equal(first.dual_direction, restored.dual_direction)


def test_prorm_trainer_reads_v1_name_with_current_numerical_config() -> None:
    batch = _misspecified_toy()
    config = _prorm_config()
    source = ProRMPlusTrainer(FrozenFeatureLinearReward(2, dtype=torch.float64), batch, config)
    source.fit(2)
    legacy = copy.deepcopy(source.state_dict())
    legacy["format_version"] = 1
    legacy["trainer"] = "srm_plus"

    restored = ProRMPlusTrainer(FrozenFeatureLinearReward(2, dtype=torch.float64), batch, config)
    restored.load_state_dict(legacy)

    assert torch.equal(restored.model.weight, source.model.weight)
    assert torch.equal(restored.dual_direction, source.dual_direction)
    assert restored.history == source.history


def test_prorm_trainer_rejects_pre_fp64_checkpoint_config() -> None:
    batch = _misspecified_toy()
    config = _prorm_config()
    source = ProRMPlusTrainer(FrozenFeatureLinearReward(2, dtype=torch.float64), batch, config)
    source.step()
    legacy = copy.deepcopy(source.state_dict())
    legacy["format_version"] = 1
    legacy["trainer"] = "srm_plus"
    legacy["config"].pop("pcg_dtype")

    restored = ProRMPlusTrainer(FrozenFeatureLinearReward(2, dtype=torch.float64), batch, config)
    with pytest.raises(ValueError, match="checkpoint config"):
        restored.load_state_dict(legacy)


def test_prorm_checkpoint_rejects_non_solver_dtype_dual_direction() -> None:
    batch = _misspecified_toy()
    config = _prorm_config()
    source = ProRMPlusTrainer(FrozenFeatureLinearReward(2, dtype=torch.float64), batch, config)
    source.step()
    checkpoint = source.state_dict()
    checkpoint["dual_direction"] = checkpoint["dual_direction"].to(torch.float32)

    restored = ProRMPlusTrainer(FrozenFeatureLinearReward(2, dtype=torch.float64), batch, config)
    with pytest.raises(ValueError, match="solver dtype"):
        restored.load_state_dict(checkpoint)


def test_prorm_diagnostics_include_residual_dual_value_and_gradient_norm() -> None:
    trainer = ProRMPlusTrainer(
        FrozenFeatureLinearReward(2, dtype=torch.float64),
        _misspecified_toy(),
        _prorm_config(),
    )
    diagnostic = trainer.step()

    assert diagnostic.pcg_iterations is not None
    assert diagnostic.pcg_residual_norm is not None
    assert diagnostic.pcg_relative_residual is not None
    assert diagnostic.pcg_relative_residual < 1.0e-11
    assert diagnostic.dual_loss is not None and diagnostic.dual_loss >= 0.0
    assert diagnostic.gradient_norm > 0.0
    assert math.isfinite(diagnostic.gradient_norm)


@pytest.mark.parametrize("kind", ["bt", "prorm_plus"])
def test_optional_gradient_clipping_bounds_one_sgd_update(kind: str) -> None:
    batch = _misspecified_toy()
    model = FrozenFeatureLinearReward(2, dtype=torch.float64)
    learning_rate = 0.1
    max_norm = 1.0e-3
    if kind == "bt":
        config = BTMLETrainingConfig(
            learning_rate=learning_rate,
            optimizer="sgd",
            max_grad_norm=max_norm,
        )
        diagnostic = BTMLETrainer(model, batch, config).step()
    else:
        config = ProRMPlusTrainingConfig(
            learning_rate=learning_rate,
            optimizer="sgd",
            max_grad_norm=max_norm,
            damping=0.2,
            pcg_tolerance=1.0e-12,
        )
        diagnostic = ProRMPlusTrainer(model, batch, config).step()

    assert diagnostic.gradient_norm > max_norm
    assert torch.linalg.vector_norm(model.weight).item() <= learning_rate * max_norm * 1.000001
