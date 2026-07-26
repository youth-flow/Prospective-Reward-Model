from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import smart_reward.phase2_profile as phase2_profile
from smart_reward.contracts import BT_MLE, PRORM_PLUS
from smart_reward.experiment import TrainingTensorData
from smart_reward.phase2_checkpoint import DurableCheckpointStore
from smart_reward.phase2_config import load_phase2_config_bundle
from smart_reward.phase2_primary import (
    NeutralPhase2TrainingContext,
    prepare_neutral_phase2_context,
    primary_core_checkpoint_binding,
    train_primary_head_core,
)
from smart_reward.phase2_profile import (
    PHASE2_PROFILE_AUDIT_UPDATES,
    PHASE2_PROFILE_CAMPAIGN_KIND,
    PHASE2_PROFILE_LEARNER_ORDER,
    PHASE2_PROFILE_ROLE,
    PHASE2_PROFILE_SEED,
    PHASE2_PROFILE_STOP_REASON,
    PHASE2_PROFILE_UPDATES,
    PCGReasonUnavailableError,
    profile_core_binding,
    run_gate_p_profile_core,
    validate_gate_p_profile_core_result,
)
from smart_reward.phase2_training import (
    FirstOrderConvergenceSpec,
    Phase2TrainingSettings,
    compile_phase2_training_settings,
)
from smart_reward.training import ProRMPlusTrainer

ROOT = Path(__file__).resolve().parents[1]


def _training() -> TrainingTensorData:
    num_prompts, num_candidates, policy_dimension, reward_dimension = 4, 4, 3, 2
    node = torch.arange(
        num_prompts * num_candidates,
        dtype=torch.float32,
    ).reshape(num_prompts, num_candidates)
    return TrainingTensorData(
        prompt_ids=tuple(f"profile-train-{index}" for index in range(num_prompts)),
        policy_scores=torch.stack(
            [
                torch.sin((coordinate + 1.0) * 0.11 * node) + 0.1 * coordinate
                for coordinate in range(policy_dimension)
            ],
            dim=-1,
        ),
        reward_features=torch.stack(
            [
                torch.cos((coordinate + 1.0) * 0.13 * node) - 0.05 * coordinate
                for coordinate in range(reward_dimension)
            ],
            dim=-1,
        ),
        h=torch.linspace(-0.4, 0.3, num_prompts),
        left_wins=torch.tensor([1, 2, 1, 3], dtype=torch.int64),
        num_annotations=torch.tensor([3, 4, 2, 5], dtype=torch.int64),
    )


def _oracle_rewards(training: TrainingTensorData) -> torch.Tensor:
    node = torch.arange(
        training.num_prompts * training.num_candidates,
        dtype=training.policy_scores.dtype,
        device=training.policy_scores.device,
    ).reshape(training.num_prompts, training.num_candidates)
    return 0.2 * torch.sin(0.3 * node)


@pytest.fixture(scope="module")
def profile_settings() -> Phase2TrainingSettings:
    compiled = compile_phase2_training_settings(
        load_phase2_config_bundle(ROOT / "configs" / "common_beta_pilot.yaml")
    )
    return replace(
        compiled,
        phase2_config_hash="d" * 64,
        low_dimensional_selected_dimension=2,
        convergence=FirstOrderConvergenceSpec(
            gradient_ratio_tolerance=1.0e6,
            min_steps=20,
            max_steps=720,
            check_interval=20,
            consecutive_checks=2,
        ),
    )


@pytest.fixture(scope="module")
def profile_context(profile_settings: Phase2TrainingSettings) -> NeutralPhase2TrainingContext:
    training = _training()
    return prepare_neutral_phase2_context(
        training,
        _oracle_rewards(training),
        seed=PHASE2_PROFILE_SEED,
        settings=profile_settings,
    )


@pytest.fixture(scope="module")
def profile_result(
    profile_context: NeutralPhase2TrainingContext,
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[dict[str, object], Path]:
    probe_directory = tmp_path_factory.mktemp("gate-p-io-probe")
    result = run_gate_p_profile_core(
        profile_context,
        io_probe_directory=probe_directory,
    )
    return result, probe_directory


def _walk(value: object):
    yield value
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item)


def test_profile_binding_is_independent_and_primary_validator_rejects_it(
    profile_context: NeutralPhase2TrainingContext,
    tmp_path: Path,
) -> None:
    binding = profile_core_binding(profile_context)
    assert binding["campaign_kind"] == PHASE2_PROFILE_CAMPAIGN_KIND
    assert binding["role"] == PHASE2_PROFILE_ROLE
    assert binding["profile_nonreusable"] is True
    assert binding["seed"] == PHASE2_PROFILE_SEED
    assert binding["learner_order"] == [BT_MLE, PRORM_PLUS]
    assert binding["update_cap_per_learner"] == 100
    assert binding["audit_update_indices"] == [0, 20, 40, 60, 80, 100]
    assert binding["stop_reason"] == PHASE2_PROFILE_STOP_REASON
    for learner in PHASE2_PROFILE_LEARNER_ORDER:
        assert binding != primary_core_checkpoint_binding(profile_context, learner)

    store = DurableCheckpointStore(
        tmp_path / "profile-cannot-be-primary",
        objective=BT_MLE,
        binding=binding,
    )
    with pytest.raises(ValueError, match="binding"):
        train_primary_head_core(
            profile_context,
            BT_MLE,
            checkpoint_store=store,
        )


def test_gate_p_fails_before_any_training_without_raw_pcg_reason_contract(
    profile_context: NeutralPhase2TrainingContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built = False

    def forbidden_build(*_args, **_kwargs):
        nonlocal built
        built = True
        raise AssertionError("preflight must run before trainer construction")

    monkeypatch.delattr(ProRMPlusTrainer, "last_pcg_reason")
    monkeypatch.setattr(phase2_profile, "build_primary_core_trainer", forbidden_build)
    with pytest.raises(PCGReasonUnavailableError, match="inference is forbidden"):
        run_gate_p_profile_core(profile_context)
    assert built is False


def test_gate_p_runs_exact_fixed_work_and_strict_train_only_schema(
    profile_result: tuple[dict[str, object], Path],
) -> None:
    result, probe_directory = profile_result
    validate_gate_p_profile_core_result(result)
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert result["learner_order"] == list(PHASE2_PROFILE_LEARNER_ORDER)
    assert result["update_cap_per_learner"] == PHASE2_PROFILE_UPDATES
    assert result["audit_update_indices"] == list(PHASE2_PROFILE_AUDIT_UPDATES)
    assert result["stop_reason"] == PHASE2_PROFILE_STOP_REASON
    assert result["profile_nonreusable"] is True
    assert result["seed"] == PHASE2_PROFILE_SEED
    assert result["formal_cuda_profile"] is False
    assert result["device_type"] == "cpu"
    assert result["setup"]["cuda_memory"] == {
        "measurement": "nonformal_cpu",
        "current_bytes": None,
        "peak_bytes": None,
    }

    learners = result["learners"]
    assert [item["learner"] for item in learners] == [BT_MLE, PRORM_PLUS]
    for learner in learners:
        assert learner["updates_executed"] == 100
        assert learner["stop_reason"] == "predeclared_profile_update_cap"
        assert learner["gradient_selection_applied"] is False
        assert [item["update"] for item in learner["steps"]] == list(range(1, 101))
        assert [item["update"] for item in learner["audits"]] == [0, 20, 40, 60, 80, 100]
        assert all(
            set(item) == {"update", "wall_seconds", "trainer_state_unchanged"}
            for item in learner["audits"]
        )
        assert [item["update"] for item in learner["ephemeral_checkpoint_io"]] == [
            0,
            20,
            40,
            60,
            80,
            100,
        ]
        assert all(item["trainer_state_unchanged"] for item in learner["audits"])
        assert all(
            item["artifact_retained"] is False
            and item["reusable"] is False
            and item["roundtrip_verified"] is True
            for item in learner["ephemeral_checkpoint_io"]
        )
        assert all(
            item["cuda_memory"]["measurement"] == "nonformal_cpu"
            and item["cuda_memory"]["current_bytes"] is None
            and item["cuda_memory"]["peak_bytes"] is None
            for item in learner["steps"]
        )
    bt, prorm = learners
    assert all(set(step) == {"update", "wall_seconds", "cuda_memory"} for step in bt["steps"])
    assert all(
        set(step) == {"update", "wall_seconds", "cuda_memory", "pcg"} for step in prorm["steps"]
    )
    assert all(
        set(step["pcg"])
        == {
            "iterations",
            "residual_norm",
            "relative_residual",
            "converged",
            "reason",
        }
        and step["pcg"]["reason"] in {"converged", "zero_rhs", "max_iterations"}
        for step in prorm["steps"]
    )
    assert list(probe_directory.iterdir()) == []


def test_profile_payload_contains_no_forbidden_data(
    profile_result: tuple[dict[str, object], Path],
) -> None:
    result, _ = profile_result
    forbidden = {
        "head",
        "optimizer",
        "rng",
        "raw_reward",
        "raw_oracle",
        "label",
        "beta",
        "heldout",
        "outcome",
    }
    for item in _walk(result):
        assert not isinstance(item, torch.Tensor)
        if isinstance(item, str):
            lowered = item.casefold()
            assert all(token not in lowered for token in forbidden)


def test_profile_validator_rejects_gradient_selection_missing_reason_and_leakage(
    profile_result: tuple[dict[str, object], Path],
) -> None:
    result, _ = profile_result

    selected = copy.deepcopy(result)
    selected["learners"][0]["gradient_selection_applied"] = True
    with pytest.raises(ValueError, match="fixed-work contract"):
        validate_gate_p_profile_core_result(selected)

    missing_reason = copy.deepcopy(result)
    del missing_reason["learners"][1]["steps"][0]["pcg"]["reason"]
    with pytest.raises(ValueError, match="strict schema"):
        validate_gate_p_profile_core_result(missing_reason)

    leaked = copy.deepcopy(result)
    leaked["head"] = [1.0]
    with pytest.raises(ValueError, match="strict schema"):
        validate_gate_p_profile_core_result(leaked)


def test_profile_rejects_every_nonpredeclared_seed(
    profile_settings: Phase2TrainingSettings,
) -> None:
    other_seed = PHASE2_PROFILE_SEED + 1
    training = _training()
    context = prepare_neutral_phase2_context(
        training,
        _oracle_rewards(training),
        seed=other_seed,
        settings=profile_settings,
    )
    with pytest.raises(ValueError, match=str(PHASE2_PROFILE_SEED)):
        profile_core_binding(context)
