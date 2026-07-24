import math
from dataclasses import FrozenInstanceError, fields

import pytest
import torch

from smart_reward.annotations import randomized_truncation_u_statistic_from_counts
from smart_reward.experiment import TrainingTensorData
from smart_reward.phase2_controls import (
    ALL_SIX_PAIR_LAYOUT,
    PRIMARY_GAMMA,
    PRIMARY_NUM_REPLICATES,
    TangentCoordinateLayout,
    build_direct_oracle_geometry_control,
    build_exact_margin_canonical_arm,
    generate_seeded_orthonormal_projection,
    sample_all_six_prompt_u_stat_arm,
    sample_canonical_r4_noisy_arm,
    select_low_dimensional_tangent,
    select_seeded_orthonormal_tangent,
    slice_low_dimensional_tangent,
)


def _training(
    *,
    num_prompts: int = 3,
    num_candidates: int = 4,
    policy_dimension: int = 5,
    reward_dimension: int = 3,
) -> TrainingTensorData:
    dtype = torch.float64
    node = torch.arange(
        num_prompts * num_candidates,
        dtype=dtype,
    ).reshape(num_prompts, num_candidates)
    policy_scores = torch.stack(
        [
            torch.sin((coordinate + 1.0) * 0.13 * node) + 0.07 * coordinate
            for coordinate in range(policy_dimension)
        ],
        dim=-1,
    )
    reward_features = torch.stack(
        [
            torch.cos((coordinate + 1.0) * 0.11 * node) - 0.04 * coordinate
            for coordinate in range(reward_dimension)
        ],
        dim=-1,
    )
    return TrainingTensorData(
        prompt_ids=tuple(f"train-{index}" for index in range(num_prompts)),
        policy_scores=policy_scores,
        reward_features=reward_features,
        h=torch.linspace(-0.3, 0.4, num_prompts, dtype=dtype),
        left_wins=torch.arange(num_prompts, dtype=torch.int64).remainder(9),
        num_annotations=torch.full((num_prompts,), 8, dtype=torch.int64),
    )


def _oracle_rewards(training: TrainingTensorData) -> torch.Tensor:
    candidate = torch.arange(
        training.num_candidates,
        dtype=training.policy_scores.dtype,
        device=training.policy_scores.device,
    )
    prompt = torch.arange(
        training.num_prompts,
        dtype=training.policy_scores.dtype,
        device=training.policy_scores.device,
    ).unsqueeze(1)
    return 0.2 * prompt + torch.sin(0.7 * candidate) + 0.1 * candidate.square()


def _all_six_probabilities(node_utilities: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [
            torch.sigmoid(node_utilities[:, left] - node_utilities[:, right])
            for left, right in ALL_SIX_PAIR_LAYOUT
        ],
        dim=1,
    )


def test_exact_margin_arm_is_gauge_invariant_leakage_safe_and_storage_independent() -> None:
    training = _training()
    rewards = _oracle_rewards(training)
    arm = build_exact_margin_canonical_arm(training, rewards)

    torch.testing.assert_close(arm.training.h, rewards[:, 0] - rewards[:, 1])
    assert torch.equal(arm.training.left_wins, training.left_wins)
    assert torch.equal(arm.training.num_annotations, training.num_annotations)
    assert arm.training.policy_scores.data_ptr() != training.policy_scores.data_ptr()
    assert arm.training.reward_features.data_ptr() != training.reward_features.data_ptr()
    assert arm.training.h.data_ptr() != rewards.data_ptr()
    assert arm.audit.raw_node_rewards_retained is False
    assert arm.audit.bt_counts_source == "input_training_passthrough"
    assert arm.audit.reward_head_fit_required is True
    assert arm.audit.oracle_direction_identity_expected is False
    assert len(arm.audit.source_node_rewards_sha256) == 64
    assert len(arm.audit.exact_margin_sha256) == 64
    assert {field.name for field in fields(arm.training)} == {
        "prompt_ids",
        "policy_scores",
        "reward_features",
        "h",
        "left_wins",
        "num_annotations",
    }
    assert all(edge.edge_id == (edge.prompt_id, 0, 1) for edge in arm.edges)

    prompt_gauge = torch.tensor([[11.0], [-4.0], [0.75]], dtype=torch.float64)
    gauged = build_exact_margin_canonical_arm(training, rewards + prompt_gauge)
    torch.testing.assert_close(gauged.training.h, arm.training.h, rtol=0.0, atol=1.0e-14)
    assert gauged.audit.source_node_rewards_sha256 != arm.audit.source_node_rewards_sha256

    with pytest.raises(FrozenInstanceError):
        arm.audit.raw_node_rewards_retained = True  # type: ignore[misc]


def test_exact_margin_and_geometry_reverse_together_when_candidates_are_swapped() -> None:
    training = _training()
    rewards = _oracle_rewards(training)
    forward = build_exact_margin_canonical_arm(training, rewards)
    permutation = torch.tensor([1, 0, 2, 3], dtype=torch.int64)
    swapped_training = TrainingTensorData(
        prompt_ids=training.prompt_ids,
        policy_scores=training.policy_scores.index_select(1, permutation),
        reward_features=training.reward_features.index_select(1, permutation),
        h=-training.h,
        left_wins=training.num_annotations - training.left_wins,
        num_annotations=training.num_annotations.clone(),
    )
    reverse = build_exact_margin_canonical_arm(
        swapped_training,
        rewards.index_select(1, permutation),
    )

    torch.testing.assert_close(reverse.training.h, -forward.training.h)
    torch.testing.assert_close(
        reverse.training.to_training_batch().edge_scores,
        -forward.training.to_training_batch().edge_scores,
    )
    torch.testing.assert_close(
        reverse.training.to_training_batch().feature_differences,
        -forward.training.to_training_batch().feature_differences,
    )


def test_direct_oracle_control_proves_complete_pair_moment_identity_and_bypasses_head() -> None:
    training = _training()
    rewards = _oracle_rewards(training)
    control = build_direct_oracle_geometry_control(
        training,
        rewards,
        relative_damping=0.05,
        pcg_tolerance=1.0e-12,
    )

    torch.testing.assert_close(
        control.canonical_margins,
        rewards[:, 0] - rewards[:, 1],
    )
    torch.testing.assert_close(
        control.complete_pair_u_stat_moment,
        control.all_node_covariance_moment,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert control.identity_absolute_error < 1.0e-12
    assert control.native_oracle_direction.beta == 1.0
    assert control.native_oracle_direction.pcg_converged is True
    assert control.reward_head_bypassed is True
    assert control.complete_pair_identity_is_algebraic is True
    assert control.trained_exact_margin_head_required_to_match is False
    assert control.raw_node_rewards_retained is False
    assert "node_rewards" not in {field.name for field in fields(control)}

    # The direct geometry control does not depend on the misspecified
    # reward-feature class because it never fits a reward head.
    changed_features = TrainingTensorData(
        prompt_ids=training.prompt_ids,
        policy_scores=training.policy_scores.clone(),
        reward_features=torch.zeros_like(training.reward_features),
        h=training.h.clone(),
        left_wins=training.left_wins.clone(),
        num_annotations=training.num_annotations.clone(),
    )
    changed = build_direct_oracle_geometry_control(
        changed_features,
        rewards,
        relative_damping=0.05,
        pcg_tolerance=1.0e-12,
    )
    torch.testing.assert_close(
        changed.all_node_covariance_moment,
        control.all_node_covariance_moment,
    )
    torch.testing.assert_close(
        changed.native_oracle_direction.direction,
        control.native_oracle_direction.direction,
    )


def test_direct_oracle_control_is_prompt_reward_gauge_invariant() -> None:
    training = _training()
    rewards = _oracle_rewards(training)
    gauge = torch.tensor([[17.0], [-9.0], [2.0]], dtype=torch.float64)
    base = build_direct_oracle_geometry_control(
        training,
        rewards,
        relative_damping=0.1,
        pcg_tolerance=1.0e-12,
    )
    shifted = build_direct_oracle_geometry_control(
        training,
        rewards + gauge,
        relative_damping=0.1,
        pcg_tolerance=1.0e-12,
    )

    torch.testing.assert_close(
        shifted.canonical_margins,
        base.canonical_margins,
        rtol=0.0,
        atol=2.0e-15,
    )
    torch.testing.assert_close(
        shifted.all_node_covariance_moment,
        base.all_node_covariance_moment,
        rtol=0.0,
        atol=2.0e-14,
    )
    torch.testing.assert_close(
        shifted.native_oracle_direction.direction,
        base.native_oracle_direction.direction,
        rtol=0.0,
        atol=2.0e-13,
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("wrong_shape", "shape"),
        ("integer", "floating-point"),
        ("nan", "finite"),
        ("requires_grad", "frozen"),
    ],
)
def test_exact_margin_rejects_invalid_oracle_inputs(mutation: str, message: str) -> None:
    training = _training()
    rewards = _oracle_rewards(training)
    if mutation == "wrong_shape":
        rewards = rewards[:, :2]
    elif mutation == "integer":
        rewards = rewards.to(torch.int64)
    elif mutation == "nan":
        rewards = rewards.clone()
        rewards[0, 0] = float("nan")
    elif mutation == "requires_grad":
        rewards = rewards.clone().requires_grad_(True)

    with pytest.raises((TypeError, ValueError), match=message):
        build_exact_margin_canonical_arm(training, rewards)


def test_canonical_r4_routes_mean_h_and_raw_bt_counts_without_pooling_h() -> None:
    training = _training()
    probabilities = torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64)
    arm = sample_canonical_r4_noisy_arm(
        training,
        probabilities,
        generator=torch.Generator().manual_seed(20_260_725),
    )

    labels = arm.repeated_labels
    assert labels.num_replicates == PRIMARY_NUM_REPLICATES
    assert labels.gamma == PRIMARY_GAMMA
    assert labels.counts.shape == (4, training.num_prompts)
    assert labels.replicate_h.shape == (4, training.num_prompts)
    assert tuple(batch.counts.shape for batch in labels.replicates) == ((3,),) * 4
    assert torch.equal(arm.training.h, labels.replicate_h.mean(dim=0))
    assert torch.equal(arm.training.left_wins, labels.wins.sum(dim=0))
    assert torch.equal(arm.training.num_annotations, labels.counts.sum(dim=0))
    assert arm.audit.replicate_boundaries_preserved is True
    assert arm.audit.pooled_counts_reused_as_one_truncation is False
    assert arm.audit.prorm_target == "mean_of_per_replicate_h"
    assert arm.audit.bt_target == "pooled_raw_wins_and_totals"
    assert arm.audit.realized_total_annotations == int(labels.counts.sum())
    assert arm.audit.expected_annotations_per_edge == 40.0

    invalid_pooled_h = randomized_truncation_u_statistic_from_counts(
        labels.pooled_wins,
        labels.pooled_totals,
        gamma=PRIMARY_GAMMA,
    )
    assert not torch.equal(arm.training.h, invalid_pooled_h)


def test_canonical_r4_estimator_is_unbiased_and_has_the_locked_expected_cost() -> None:
    num_edges = 20_000
    probability = 0.4
    training = _training(
        num_prompts=num_edges,
        num_candidates=2,
        policy_dimension=2,
        reward_dimension=1,
    )
    probabilities = torch.full((num_edges,), probability, dtype=torch.float64)
    arm = sample_canonical_r4_noisy_arm(
        training,
        probabilities,
        generator=torch.Generator().manual_seed(91_337),
    )

    truth = math.log(probability / (1.0 - probability))
    assert float(arm.training.h.mean()) == pytest.approx(truth, abs=0.02)
    assert arm.audit.realized_annotations_per_edge == pytest.approx(40.0, rel=0.015)


def test_canonical_noisy_orientation_reverses_labels_counts_and_h() -> None:
    training = _training()
    left = sample_canonical_r4_noisy_arm(
        training,
        torch.ones(training.num_prompts, dtype=torch.float64),
        generator=torch.Generator().manual_seed(44),
    )
    right = sample_canonical_r4_noisy_arm(
        training,
        torch.zeros(training.num_prompts, dtype=torch.float64),
        generator=torch.Generator().manual_seed(44),
    )

    assert torch.equal(left.repeated_labels.counts, right.repeated_labels.counts)
    assert torch.equal(
        left.training.left_wins,
        right.training.num_annotations - right.training.left_wins,
    )
    torch.testing.assert_close(left.training.h, -right.training.h)


@pytest.mark.parametrize(
    ("probabilities", "message"),
    [
        (torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float64), "shape"),
        (torch.tensor([0.5, -0.1, 0.5], dtype=torch.float64), r"\[0, 1\]"),
        (torch.tensor([0.5, 1.1, 0.5], dtype=torch.float64), r"\[0, 1\]"),
        (torch.tensor([0, 1, 0], dtype=torch.int64), "floating-point"),
    ],
)
def test_canonical_noisy_arm_rejects_invalid_probabilities(
    probabilities: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        sample_canonical_r4_noisy_arm(_training(), probabilities)


def test_all_six_arm_uses_every_oriented_edge_and_preserves_prompt_clusters() -> None:
    training = _training(num_prompts=2)
    probabilities = _all_six_probabilities(_oracle_rewards(training))
    arm = sample_all_six_prompt_u_stat_arm(
        training,
        probabilities,
        generator=torch.Generator().manual_seed(6_006),
    )

    batch = arm.training_batch
    assert batch.num_edges == 12
    assert batch.node_scores.shape == (8, training.policy_dimension)
    torch.testing.assert_close(
        batch.node_scores,
        training.policy_scores.reshape(8, training.policy_dimension),
    )
    assert arm.repeated_labels.counts.shape == (4, 2, 6)
    assert arm.label_audit.edge_shape == (2, 6)
    assert arm.edge_prompt_indices == (0,) * 6 + (1,) * 6
    assert arm.prompt_edge_slice(0) == slice(0, 6)
    assert arm.prompt_edge_slice(1) == slice(6, 12)
    assert len({edge.edge_id for edge in arm.edges}) == 12

    for prompt_index in range(2):
        for edge_index, (left, right) in enumerate(ALL_SIX_PAIR_LAYOUT):
            flat_index = prompt_index * 6 + edge_index
            edge = arm.edges[flat_index]
            assert edge.prompt_id == training.prompt_ids[prompt_index]
            assert edge.edge_index_within_prompt == edge_index
            assert edge.left_node_id == (training.prompt_ids[prompt_index], left)
            assert edge.right_node_id == (training.prompt_ids[prompt_index], right)
            torch.testing.assert_close(
                batch.left_features[flat_index],
                training.reward_features[prompt_index, left],
            )
            torch.testing.assert_close(
                batch.right_features[flat_index],
                training.reward_features[prompt_index, right],
            )
            torch.testing.assert_close(
                batch.edge_scores[flat_index],
                training.policy_scores[prompt_index, left]
                - training.policy_scores[prompt_index, right],
            )

    assert arm.metadata.cluster_unit == "prompt"
    assert arm.metadata.construction == "complete_unordered_pair_u_statistic"
    assert arm.metadata.edges_share_nodes_within_prompt is True
    assert arm.metadata.edges_independent_within_prompt is False
    assert arm.metadata.prompt_is_experimental_unit is True
    assert arm.metadata.annotation_work_multiplier == 6
    assert arm.metadata.reward_edge_work_multiplier == 6
    assert arm.metadata.generation_work_multiplier == 1
    assert arm.metadata.oracle_forward_work_multiplier == 1
    assert arm.metadata.independent_sample_multiplier == 1


def test_all_six_r4_targets_use_replicate_means_and_pooled_raw_counts() -> None:
    training = _training(num_prompts=2)
    probabilities = torch.full((2, 6), 0.55, dtype=torch.float64)
    arm = sample_all_six_prompt_u_stat_arm(
        training,
        probabilities,
        generator=torch.Generator().manual_seed(1_234),
    )
    labels = arm.repeated_labels

    assert torch.equal(
        arm.training_batch.h,
        labels.replicate_h.mean(dim=0).reshape(-1),
    )
    assert torch.equal(
        arm.training_batch.left_wins,
        labels.wins.sum(dim=0).reshape(-1),
    )
    assert torch.equal(
        arm.training_batch.num_annotations,
        labels.counts.sum(dim=0).reshape(-1),
    )
    invalid_pooled_h = randomized_truncation_u_statistic_from_counts(
        labels.pooled_wins,
        labels.pooled_totals,
        gamma=PRIMARY_GAMMA,
    )
    assert not torch.equal(arm.training_batch.h, invalid_pooled_h.reshape(-1))


def test_all_six_rejects_non_four_node_or_misoriented_probability_layout() -> None:
    with pytest.raises(ValueError, match="exactly four"):
        sample_all_six_prompt_u_stat_arm(
            _training(num_candidates=3),
            torch.full((3, 6), 0.5, dtype=torch.float64),
        )
    with pytest.raises(ValueError, match="shape"):
        sample_all_six_prompt_u_stat_arm(
            _training(),
            torch.full((3, 4, 4), 0.5, dtype=torch.float64),
        )
    with pytest.raises(ValueError, match="out of range"):
        arm = sample_all_six_prompt_u_stat_arm(
            _training(),
            torch.full((3, 6), 0.5, dtype=torch.float64),
            generator=torch.Generator().manual_seed(2),
        )
        arm.prompt_edge_slice(3)


def test_low_dimensional_selection_is_auditable_independent_and_scatterable() -> None:
    training = _training(policy_dimension=5)
    layout = TangentCoordinateLayout(
        layout_id="fixed-a-lora-b/full-v1",
        coordinate_ids=("q0", "q1", "v0", "v1", "q2"),
    )
    control = select_low_dimensional_tangent(
        training,
        coordinate_indices=(4, 1),
        coordinate_layout=layout,
    )
    sliced = control.training

    torch.testing.assert_close(
        sliced.policy_scores,
        training.policy_scores[:, :, [4, 1]],
    )
    assert control.source_dimension == 5
    assert control.selected_indices == (4, 1)
    assert control.selected_coordinate_ids == ("q2", "q1")
    assert control.num_fisher_nodes == training.num_prompts * training.num_candidates
    assert sliced.policy_scores.data_ptr() != training.policy_scores.data_ptr()
    assert sliced.reward_features.data_ptr() != training.reward_features.data_ptr()
    assert sliced.h.data_ptr() != training.h.data_ptr()
    assert sliced.left_wins.data_ptr() != training.left_wins.data_ptr()
    assert sliced.num_annotations.data_ptr() != training.num_annotations.data_ptr()

    selected_direction = torch.tensor([0.75, -0.25], dtype=torch.float64)
    full_direction = control.scatter_direction_to_full(selected_direction)
    torch.testing.assert_close(
        full_direction,
        torch.tensor([0.0, -0.25, 0.0, 0.0, 0.75], dtype=torch.float64),
    )
    assert full_direction.data_ptr() != selected_direction.data_ptr()

    direct = slice_low_dimensional_tangent(
        training,
        coordinate_indices=(4, 1),
        coordinate_layout=layout,
    )
    assert isinstance(direct, TrainingTensorData)
    torch.testing.assert_close(direct.policy_scores, sliced.policy_scores)
    assert direct.policy_scores.data_ptr() != sliced.policy_scores.data_ptr()


@pytest.mark.parametrize(
    ("indices", "message"),
    [
        ((), "at least one"),
        ((0, 0), "unique"),
        ((5,), "out of range"),
        ((True,), "non-boolean"),
        ((0, 1, 2, 3, 4), "strict coordinate subset"),
    ],
)
def test_low_dimensional_selection_rejects_invalid_indices(
    indices: tuple[object, ...],
    message: str,
) -> None:
    training = _training(policy_dimension=5)
    layout = TangentCoordinateLayout("layout", tuple(range(5)))
    with pytest.raises((TypeError, ValueError), match=message):
        select_low_dimensional_tangent(
            training,
            coordinate_indices=indices,  # type: ignore[arg-type]
            coordinate_layout=layout,
        )


def test_low_dimensional_selection_validates_layout_and_d_below_fisher_nodes() -> None:
    training = _training(
        num_prompts=1,
        num_candidates=2,
        policy_dimension=4,
    )
    layout = TangentCoordinateLayout("full", ("a", "b", "c", "d"))
    with pytest.raises(ValueError, match=r"d < n_F"):
        select_low_dimensional_tangent(
            training,
            coordinate_indices=(0, 1),
            coordinate_layout=layout,
        )
    with pytest.raises(ValueError, match="layout dimension"):
        select_low_dimensional_tangent(
            _training(policy_dimension=5),
            coordinate_indices=(0,),
            coordinate_layout=layout,
        )
    with pytest.raises(ValueError, match="unique"):
        TangentCoordinateLayout("bad", ("a", "a"))
    with pytest.raises(ValueError, match="non-empty"):
        TangentCoordinateLayout("", ("a",))


def test_scatter_rejects_wrong_direction_shape_dtype_or_grad_state() -> None:
    training = _training(policy_dimension=5)
    control = select_low_dimensional_tangent(
        training,
        coordinate_indices=(0, 3),
        coordinate_layout=TangentCoordinateLayout("full", tuple(range(5))),
    )
    with pytest.raises(ValueError, match="shape"):
        control.scatter_direction_to_full(torch.zeros(3, dtype=torch.float64))
    with pytest.raises(TypeError, match="floating-point"):
        control.scatter_direction_to_full(torch.zeros(2, dtype=torch.int64))
    with pytest.raises(ValueError, match="frozen"):
        control.scatter_direction_to_full(torch.zeros(2, dtype=torch.float64, requires_grad=True))


def test_seeded_orthonormal_projection_is_reproducible_audited_and_deployable() -> None:
    training = _training(policy_dimension=6)
    layout = TangentCoordinateLayout(
        "fixed-a-lora-b/full-v1",
        tuple(f"coordinate-{index}" for index in range(6)),
    )
    kwargs = {
        "selected_dimension": 3,
        "coordinate_layout": layout,
        "seed": 202_607_25,
        "namespace": "prorm-common-beta-low-dimensional-tangent-v1",
    }
    first = select_seeded_orthonormal_tangent(training, **kwargs)
    second = select_seeded_orthonormal_tangent(training, **kwargs)

    assert torch.equal(first.projection, second.projection)
    assert first.projection_sha256 == second.projection_sha256
    assert first.effective_seed == second.effective_seed
    assert first.declared_seed == 202_607_25
    assert first.selected_dimension == 3
    assert first.source_dimension == 6
    assert first.num_fisher_nodes == 12
    torch.testing.assert_close(
        first.projection.mT @ first.projection,
        torch.eye(3, dtype=torch.float64),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    torch.testing.assert_close(
        first.training.policy_scores,
        training.policy_scores @ first.projection,
    )
    assert first.training.policy_scores.data_ptr() != training.policy_scores.data_ptr()
    assert first.training.reward_features.data_ptr() != training.reward_features.data_ptr()

    low_direction = torch.tensor([0.3, -0.5, 0.8], dtype=torch.float64)
    full_direction = first.scatter_direction_to_full(low_direction)
    torch.testing.assert_close(
        training.policy_scores @ full_direction,
        first.training.policy_scores @ low_direction,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    torch.testing.assert_close(
        torch.linalg.vector_norm(full_direction),
        torch.linalg.vector_norm(low_direction),
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    float32_direction = low_direction.to(torch.float32)
    assert first.scatter_direction_to_full(float32_direction).dtype == torch.float32
    different_namespace = select_seeded_orthonormal_tangent(
        training,
        selected_dimension=3,
        coordinate_layout=layout,
        seed=202_607_25,
        namespace="different-independent-stream",
    )
    assert not torch.equal(first.projection, different_namespace.projection)
    assert first.projection_sha256 != different_namespace.projection_sha256


def test_seeded_projection_validates_dimension_seed_namespace_dtype_and_layout() -> None:
    training = _training(
        num_prompts=1,
        num_candidates=2,
        policy_dimension=5,
    )
    layout = TangentCoordinateLayout("full", tuple(range(5)))
    with pytest.raises(ValueError, match=r"d < n_F"):
        select_seeded_orthonormal_tangent(
            training,
            selected_dimension=2,
            coordinate_layout=layout,
            seed=1,
            namespace="projection",
        )
    with pytest.raises(ValueError, match="selected_dimension < source_dimension"):
        generate_seeded_orthonormal_projection(
            3,
            3,
            seed=1,
            namespace="projection",
            dtype=torch.float64,
            device="cpu",
        )
    with pytest.raises(ValueError, match="seed"):
        generate_seeded_orthonormal_projection(
            3,
            1,
            seed=-1,
            namespace="projection",
            dtype=torch.float64,
            device="cpu",
        )
    with pytest.raises(ValueError, match="namespace"):
        generate_seeded_orthonormal_projection(
            3,
            1,
            seed=1,
            namespace="",
            dtype=torch.float64,
            device="cpu",
        )
    with pytest.raises(TypeError, match="dtype"):
        generate_seeded_orthonormal_projection(
            3,
            1,
            seed=1,
            namespace="projection",
            dtype=torch.float16,
            device="cpu",
        )
    with pytest.raises(ValueError, match="layout dimension"):
        select_seeded_orthonormal_tangent(
            _training(policy_dimension=6),
            selected_dimension=2,
            coordinate_layout=layout,
            seed=1,
            namespace="projection",
        )


def test_control_builders_preserve_formal_float32_training_contract() -> None:
    source = _training(policy_dimension=6)
    training = TrainingTensorData(
        prompt_ids=source.prompt_ids,
        policy_scores=source.policy_scores.to(torch.float32),
        reward_features=source.reward_features.to(torch.float32),
        h=source.h.to(torch.float32),
        left_wins=source.left_wins.clone(),
        num_annotations=source.num_annotations.clone(),
    )
    rewards = _oracle_rewards(training)
    exact = build_exact_margin_canonical_arm(training, rewards)
    canonical = sample_canonical_r4_noisy_arm(
        training,
        torch.full((training.num_prompts,), 0.5, dtype=torch.float32),
        generator=torch.Generator().manual_seed(10),
    )
    all_six = sample_all_six_prompt_u_stat_arm(
        training,
        _all_six_probabilities(rewards),
        generator=torch.Generator().manual_seed(11),
    )
    direct = build_direct_oracle_geometry_control(
        training,
        rewards,
        relative_damping=0.1,
    )
    projected = select_seeded_orthonormal_tangent(
        training,
        selected_dimension=3,
        coordinate_layout=TangentCoordinateLayout("full", tuple(range(6))),
        seed=12,
        namespace="float32-control",
    )

    assert exact.training.policy_scores.dtype == torch.float32
    assert exact.training.h.dtype == torch.float32
    assert canonical.training.h.dtype == torch.float32
    assert all_six.training_batch.h.dtype == torch.float32
    assert all_six.training_batch.edge_scores.dtype == torch.float32
    assert direct.all_node_covariance_moment.dtype == torch.float64
    assert direct.native_oracle_direction.direction.dtype == torch.float64
    assert projected.projection.dtype == torch.float64
    assert projected.training.policy_scores.dtype == torch.float32
    assert projected.to_dict()["orthonormality_max_absolute_error"] <= 1.0e-10
    assert (
        projected.scatter_direction_to_full(torch.ones(3, dtype=torch.float64)).dtype
        == torch.float64
    )


def test_probability_and_layout_objects_do_not_accept_grad_or_implicit_inputs() -> None:
    training = _training()
    probabilities = torch.full(
        (training.num_prompts,),
        0.5,
        dtype=torch.float64,
        requires_grad=True,
    )
    with pytest.raises(ValueError, match="frozen"):
        sample_canonical_r4_noisy_arm(training, probabilities)
    with pytest.raises(TypeError, match="explicit integer sequence"):
        select_low_dimensional_tangent(
            training,
            coordinate_indices=3,  # type: ignore[arg-type]
            coordinate_layout=TangentCoordinateLayout(
                "full",
                tuple(range(training.policy_dimension)),
            ),
        )

    detached = probabilities.detach()
    with pytest.raises(RuntimeError, match="max_total_annotations"):
        sample_canonical_r4_noisy_arm(
            training,
            detached,
            generator=torch.Generator().manual_seed(7),
            max_total_annotations=training.num_prompts * 4 - 1,
        )
