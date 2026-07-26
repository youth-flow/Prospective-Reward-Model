from __future__ import annotations

import json
from pathlib import Path

import torch

from smart_reward.experiment import TrainingTensorData
from smart_reward.phase2_r3_config import load_r3_science_config
from smart_reward.phase2_r3_control_training import (
    _prepare_control,
    profile_r3_control_family,
    run_r3_control_family,
    validate_r3_control_profile_observation,
)
from smart_reward.phase2_r3_controls import (
    load_r3_controls_config,
    validate_r3_control_family_result,
)
from smart_reward.phase2_r3_inputs import _issue_control_train_input_capability

ROOT = Path(__file__).resolve().parents[1]
SCIENCE = ROOT / "configs" / "phase2_recovery_r3_science.yaml"
CONTROLS = ROOT / "configs" / "phase2_recovery_r3_controls.yaml"
SEED = 20260801


def _training(
    *,
    num_prompts: int = 4,
    num_candidates: int = 4,
    policy_dimension: int = 3,
) -> TrainingTensorData:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(7301 + num_prompts + num_candidates + policy_dimension)
    policy_scores = torch.randn(
        (num_prompts, num_candidates, policy_dimension),
        generator=generator,
        dtype=torch.float32,
    )
    reward_features = torch.randn(
        (num_prompts, num_candidates, 2),
        generator=generator,
        dtype=torch.float32,
    )
    return TrainingTensorData(
        prompt_ids=tuple(f"train-{index}" for index in range(num_prompts)),
        policy_scores=policy_scores,
        reward_features=reward_features,
        h=torch.linspace(-0.3, 0.3, num_prompts, dtype=torch.float32),
        left_wins=torch.ones(num_prompts, dtype=torch.int64),
        num_annotations=torch.full((num_prompts,), 2, dtype=torch.int64),
    )


def _oracle_rewards(training: TrainingTensorData) -> torch.Tensor:
    nodes = torch.arange(
        training.num_prompts * training.num_candidates,
        dtype=torch.float32,
    ).reshape(training.num_prompts, training.num_candidates)
    return 0.15 * torch.sin(0.17 * nodes)


def _capability(training: TrainingTensorData):
    science = load_r3_science_config(SCIENCE)
    token = "a" * 64
    return _issue_control_train_input_capability(
        training=training,
        train_oracle_rewards=_oracle_rewards(training),
        science_bundle=science,
        seed=SEED,
        source_config_hash=science.settings.source_config_hash,
        parent_registry_file_sha256=token,
        parent_seed_entry_sha256=token,
        artifact_metadata_sha256=token,
        artifact_tensors_sha256=token,
        artifact_candidates_sha256=token,
        candidate_train_prefix_sha256=token,
        candidate_train_prefix_count=(training.num_prompts * training.num_candidates),
        artifact_materialization_sha256=token,
        artifact_verification_sha256=token,
        source_run_manifest_sha256=token,
        source_producer_identity_sha256=token,
        oracle_chat_template_sha256=token,
        oracle_transform_sha256=token,
    )


def test_exact_soft_profile_executes_exactly_100_disposable_updates(
    tmp_path: Path,
) -> None:
    controls = load_r3_controls_config(CONTROLS)
    observation = validate_r3_control_profile_observation(
        profile_r3_control_family(
            _capability(_training()),
            "exact_soft_label_bt",
            controls_config=controls,
            checkpoint_directory=tmp_path,
        )
    )
    assert observation["completed_updates"] == 100
    assert observation["information_boundary"] == {
        "train_only": True,
        "primary_label_stream_constructed": False,
        "primary_label_stream_accessed": False,
        "primary_head_accessed": False,
        "heldout_or_validation_accessed": False,
        "policy_optimization_executed": False,
        "result_reusable_for_training": False,
        "head_or_optimizer_state_retained": False,
    }
    assert list(tmp_path.iterdir()) == []
    serialized = json.dumps(observation, sort_keys=True)
    for forbidden in (
        '"head_weight":',
        '"optimizer_state":',
        '"checkpoint_bytes":',
        '"raw_oracle_rewards":',
        '"raw_labels":',
    ):
        assert forbidden not in serialized


def test_exact_soft_formal_runner_uses_real_controller_and_validates() -> None:
    controls = load_r3_controls_config(CONTROLS)
    result = run_r3_control_family(
        _capability(_training()),
        "exact_soft_label_bt",
        controls_config=controls,
    )
    checked = validate_r3_control_family_result(result, controls)
    assert checked["family"] == "exact_soft_label_bt"
    assert checked["seed"] == SEED
    assert checked["completion"]["status"] == "completed"
    assert checked["completion"]["formal_family_result"] is True
    assert checked["completion"]["profile_only"] is False
    assert checked["information_boundary"]["primary_head_accessed"] is False
    assert checked["family_evidence"]["gates"] == {
        "exact_soft_label_objective_decrease": True,
        "exact_soft_label_first_order_convergence": True,
        "saved_head_objective_binding": True,
    }


def test_low_dimensional_family_prepares_real_rank_256_geometry() -> None:
    controls = load_r3_controls_config(CONTROLS)
    prepared = _prepare_control(
        _capability(
            _training(
                num_prompts=129,
                num_candidates=2,
                policy_dimension=257,
            )
        ),
        controls,
        "low_dimensional_prorm_plus",
    )
    assert prepared.projection_control is not None
    assert prepared.projection_control.selected_dimension == 256
    assert prepared.dense_geometry is not None
    assert prepared.dense_geometry.rank == 256
    assert prepared.family_local_label_stream_sha256 is not None
    assert prepared.exact_arm is None
    assert prepared.direct_control is None
