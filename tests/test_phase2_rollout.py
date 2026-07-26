from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import smart_reward.phase1 as phase1
import smart_reward.phase2_rollout as phase2_rollout_module
from smart_reward.config import config_hash
from smart_reward.contracts import BT_MLE, PRORM_PLUS
from smart_reward.data import CandidateNode
from smart_reward.experiment import TrainingTensorData
from smart_reward.oracle import RobustOracleTransform
from smart_reward.phase2_config import load_phase2_config_bundle
from smart_reward.phase2_heldout import (
    DeferredHeldoutInputs,
    DeferredHeldoutSplit,
)
from smart_reward.phase2_rollout import (
    BUDGETED_COMMON_BETA_RULE,
    CONFIRMATORY_COMMON_BETA_RULE,
    KL_HISTORY_SOURCE,
    KL_ORIENTATION,
    PHASE2_ARM_ORDER,
    PHASE2_BUDGETED_RESULT_SCHEMA,
    PHASE2_BUDGETED_ROLLOUT_SCHEMA,
    PHASE2_PILOT_DIAGNOSTIC_SCHEMA,
    PHASE2_PILOT_RESULT_SCHEMA,
    PILOT_FREEZE_COMMON_BETA_RULE,
    Phase2ArmDeployment,
    Phase2Design,
    Phase2HeadTrainingResult,
    Phase2PolicyRollout,
    Phase2PreOracleSafetyError,
    Phase2PreparedInputs,
    Phase2Trajectory,
    assess_phase2_pre_oracle_safety,
    run_common_beta_rollouts,
)
from smart_reward.prompts import ChatMessage, PromptRecord
from smart_reward.repro import collect_execution_identity
from smart_reward.scores import ParameterLayout

_SOURCE_CONFIG = {
    "source": "strict-phase1-config",
    "policy": {"max_prompt_tokens": 1024},
}
_DIGEST = config_hash(_SOURCE_CONFIG)


def _candidate(prompt_id: str, index: int) -> CandidateNode:
    return CandidateNode(
        prompt_id=prompt_id,
        candidate_id=f"{prompt_id}::candidate::{index}",
        prompt=f"train prompt {prompt_id}",
        response=f"train response {prompt_id}/{index}",
        token_ids=(1, 10 + index),
        response_mask=(0, 1),
        terminated_by_eos=True,
        reached_max_length=False,
    )


def _heldout_candidate(split: str, prompt_id: str, index: int) -> CandidateNode:
    prompt_index = int(prompt_id.rsplit("-", 1)[1])
    return CandidateNode(
        prompt_id=prompt_id,
        candidate_id=f"{prompt_id}::candidate::{index}",
        prompt=f"{split} prompt {prompt_index}",
        response=f"heldout:{split}:{prompt_id}:{index}",
        token_ids=(1, 20 + index),
        response_mask=(0, 1),
        terminated_by_eos=True,
        reached_max_length=False,
    )


def _synthetic_safety_rollouts(
    *,
    prorm_kl: Sequence[float],
    prorm_max_length_indices: frozenset[int] = frozenset(),
) -> dict[str, Phase2PolicyRollout]:
    if len(prorm_kl) % 4 != 0:
        raise ValueError("synthetic safety KL values must have four candidates per prompt")
    prompts = len(prorm_kl) // 4
    rollouts: dict[str, Phase2PolicyRollout] = {}
    for arm_name in PHASE2_ARM_ORDER:
        trajectories: list[Phase2Trajectory] = []
        for flat_index in range(len(prorm_kl)):
            prompt_index, candidate_index = divmod(flat_index, 4)
            prompt = f"safety prompt {prompt_index}"
            reached_max_length = arm_name == PRORM_PLUS and flat_index in prorm_max_length_indices
            trajectories.append(
                Phase2Trajectory(
                    arm_name=arm_name,
                    prompt_id=f"safety-{prompt_index}",
                    candidate_index=candidate_index,
                    prompt=prompt,
                    raw_prompt_sha256=phase1._prompt_text_sha256(prompt),
                    policy_chat_token_count=1,
                    policy_prompt_token_ids_sha256=phase1._prompt_token_ids_sha256((1,)),
                    max_prompt_tokens=1024,
                    prompt_truncated=False,
                    raw_prompt_preserved=True,
                    response=f"{arm_name}:{flat_index}",
                    token_ids=(1, 2),
                    response_mask=(0, 1),
                    terminated_by_eos=not reached_max_length,
                    reached_max_length=reached_max_length,
                    prompt_rollout_seed=prompt_index,
                )
            )
        if arm_name == PRORM_PLUS:
            kl = torch.tensor(prorm_kl, dtype=torch.float32)
        elif arm_name == "zero_b":
            kl = torch.zeros(len(prorm_kl), dtype=torch.float32)
        else:
            kl = torch.full((len(prorm_kl),), 0.001, dtype=torch.float32)
        rollouts[arm_name] = Phase2PolicyRollout(
            arm_name=arm_name,
            trajectories=tuple(trajectories),
            per_sequence_kl_updated_to_reference=kl,
        )
    assert prompts >= 1
    return rollouts


def _deferred_heldout() -> DeferredHeldoutInputs:
    scores = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
            [[2.0, 0.0], [0.0, 2.0], [-2.0, 0.0], [0.0, -2.0]],
        ],
        dtype=torch.float32,
    )
    features = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
            [[1.5, 0.0], [0.0, 1.5], [-1.5, 0.0], [0.0, -1.5]],
        ],
        dtype=torch.float32,
    )

    def split(name: str, scale: float) -> DeferredHeldoutSplit:
        prompt_ids = (f"{name}-0", f"{name}-1")
        return DeferredHeldoutSplit(
            split=name,
            prompt_ids=prompt_ids,
            policy_scores=scores * scale,
            reward_features=features,
            candidates=tuple(
                _heldout_candidate(name, prompt_id, candidate_index)
                for prompt_id in prompt_ids
                for candidate_index in range(4)
            ),
        )

    return DeferredHeldoutInputs(
        validation=split("validation", 0.5),
        test=split("test", 1.0),
    )


def _inputs(tmp_path: Path) -> Phase2PreparedInputs:
    run_manifest = tmp_path / "run_manifest.json"
    run_manifest.write_text('{"phase2":"test"}\n', encoding="utf-8")
    policy_scores = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
            [[2.0, 0.0], [0.0, 2.0], [-2.0, 0.0], [0.0, -2.0]],
        ],
        dtype=torch.float32,
    )
    reward_features = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
            [[1.5, 0.0], [0.0, 1.5], [-1.5, 0.0], [0.0, -1.5]],
        ],
        dtype=torch.float32,
    )
    train = TrainingTensorData(
        prompt_ids=("train-0", "train-1"),
        policy_scores=policy_scores,
        reward_features=reward_features,
        h=torch.tensor([0.25, -0.5], dtype=torch.float32),
        left_wins=torch.tensor([3, 2], dtype=torch.int64),
        num_annotations=torch.tensor([4, 4], dtype=torch.int64),
    )
    train_candidates = tuple(
        _candidate(prompt_id, index)
        for prompt_id in train.prompt_ids
        for index in range(train.num_candidates)
    )
    test_prompts = tuple(
        PromptRecord(
            prompt_id=f"test-{index}",
            messages=(ChatMessage(role="user", content=f"test prompt {index}"),),
            split="test",
        )
        for index in range(2)
    )
    tangent = torch.zeros(2, dtype=torch.float32, requires_grad=True)
    layout = ParameterLayout.from_named_parameters((("adapter.lora_B", tangent),))
    heldout = _deferred_heldout()
    materialized_candidates = (
        *train_candidates,
        *heldout.validation.candidates,
        *heldout.test.candidates,
    )
    first_by_prompt: dict[str, CandidateNode] = {}
    for candidate in materialized_candidates:
        first_by_prompt.setdefault(str(candidate.prompt_id), candidate)
    prompt_order = (
        *train.prompt_ids,
        *heldout.validation.prompt_ids,
        *heldout.test.prompt_ids,
    )
    records = [
        {
            "prompt_id": prompt_id,
            "raw_prompt_sha256": phase1._prompt_text_sha256(first_by_prompt[prompt_id].prompt),
            "policy_chat_token_count": 1,
            "policy_prompt_token_ids_sha256": phase1._prompt_token_ids_sha256((1,)),
            "max_prompt_tokens": 1024,
            "truncated": False,
            "raw_prompt_preserved": True,
        }
        for prompt_id in prompt_order
    ]
    prompt_semantics = {
        "schema_version": phase1._POLICY_PROMPT_SEMANTICS_SCHEMA,
        "policy_chat_template_sha256": "c" * 64,
        "encoding": "policy_tokenizer_apply_chat_template",
        "add_generation_prompt": True,
        "truncation": False,
        "fail_closed_above_max_prompt_tokens": True,
        "max_prompt_tokens": 1024,
        "num_prompts": len(records),
        "minimum_policy_chat_token_count": 1,
        "maximum_policy_chat_token_count": 1,
        "mean_policy_chat_token_count": 1.0,
        "over_limit_prompt_count": 0,
        "truncated_prompt_count": 0,
        "raw_prompt_preserved_count": len(records),
        "records_sha256": phase1._prompt_semantics_records_sha256(records),
        "candidate_prefixes_verified": True,
        "records": records,
    }
    return Phase2PreparedInputs(
        source_config=_SOURCE_CONFIG,
        source_config_hash=_DIGEST,
        phase2_config_hash="2" * 64,
        seed=20260722,
        train=train,
        train_candidates=train_candidates,
        test_prompts=test_prompts,
        heldout=heldout,
        oracle_transform=RobustOracleTransform(b=0.0, tau=1.0),
        policy_layout=layout,
        policy_a_sha256="b" * 64,
        policy_chat_template_sha256="c" * 64,
        oracle_chat_template_sha256="d" * 64,
        artifact_dir=tmp_path / "artifact",
        artifact_metadata_sha256="e" * 64,
        run_manifest=run_manifest,
        run_manifest_sha256=_sha256(run_manifest),
        environment_identity=collect_execution_identity(),
        materialization_prompt_semantics=prompt_semantics,
    )


class _FakeOracleSession:
    def __init__(self, backend: _FakeBackend, phase: str) -> None:
        self.backend = backend
        self.phase = phase

    def score_transformed(
        self,
        prompts: Sequence[str],
        responses: Sequence[str],
        *,
        transform: RobustOracleTransform,
        batch_size: int,
    ) -> torch.Tensor:
        assert isinstance(transform, RobustOracleTransform)
        assert batch_size == 16
        assert len(prompts) == len(responses)
        score_phase = (
            "heldout"
            if self.phase == "test"
            and all(response.startswith("heldout:") for response in responses)
            else self.phase
        )
        self.backend.events.append(f"oracle_score:{score_phase}")
        self.backend.oracle_inputs.append((score_phase, tuple(prompts), tuple(responses)))
        if self.phase == "train":
            # Prompt-major train values; the nonzero first-coordinate moment
            # gives a valid oracle calibration direction.
            return self.backend.train_oracle_scale * torch.tensor(
                [1.0, 0.25, -1.0, -0.25, 1.5, 0.5, -1.5, -0.5],
                dtype=torch.float32,
            )
        if score_phase == "heldout":
            target_by_index = (1.0, 0.25, -1.0, -0.25)
            return torch.tensor(
                [
                    self.backend.heldout_target_scale
                    * target_by_index[int(response.rsplit(":", 1)[1])]
                    for response in responses
                ],
                dtype=torch.float32,
            )
        rewards = []
        for response in responses:
            arm_name = response.split(":", 1)[0]
            rewards.append(
                {
                    "zero_b": 0.5,
                    BT_MLE: 0.7,
                    PRORM_PLUS: 0.9,
                    "oracle_step": 1.0,
                }[arm_name]
            )
        return torch.tensor(rewards, dtype=torch.float32)


class _FakePolicySession:
    def __init__(self, backend: _FakeBackend) -> None:
        self.backend = backend

    def rollout(
        self,
        deployment: Phase2ArmDeployment,
        test_prompts: Sequence[PromptRecord],
        *,
        candidates_per_prompt: int,
        max_response_tokens: int,
        rollout_seed: int,
        kl_token_chunk_size: int,
    ) -> Phase2PolicyRollout:
        assert max_response_tokens == 256
        assert kl_token_chunk_size == 4
        self.backend.events.append(f"rollout:{deployment.arm_name}")
        self.backend.deployments.append(deployment)
        trajectories = []
        for prompt in test_prompts:
            prompt_seed = rollout_seed + int(prompt.prompt_id.rsplit("-", 1)[1])
            if self.backend.mismatched_crn and deployment.arm_name == PRORM_PLUS:
                prompt_seed += 1
            for candidate_index in range(candidates_per_prompt):
                reached_max_length = candidate_index in self.backend.reached_max_length_by_arm.get(
                    deployment.arm_name,
                    frozenset(),
                )
                trajectories.append(
                    Phase2Trajectory(
                        arm_name=deployment.arm_name,
                        prompt_id=prompt.prompt_id,
                        candidate_index=candidate_index,
                        prompt=prompt.messages[0].content,
                        raw_prompt_sha256=phase1._prompt_text_sha256(prompt.messages[0].content),
                        policy_chat_token_count=1,
                        policy_prompt_token_ids_sha256=phase1._prompt_token_ids_sha256((1,)),
                        max_prompt_tokens=1024,
                        prompt_truncated=False,
                        raw_prompt_preserved=True,
                        response=(f"{deployment.arm_name}:{prompt.prompt_id}:{candidate_index}"),
                        token_ids=(1, 2),
                        response_mask=(0, 1),
                        terminated_by_eos=not reached_max_length,
                        reached_max_length=reached_max_length,
                        prompt_rollout_seed=prompt_seed,
                    )
                )
        kl_values = self.backend.kl_values_by_arm.get(deployment.arm_name)
        if kl_values is None:
            kl = torch.full(
                (len(trajectories),),
                self.backend.kl_by_arm[deployment.arm_name],
                dtype=torch.float32,
            )
        else:
            if len(kl_values) != len(trajectories):
                raise RuntimeError("fake KL vector does not match trajectory geometry")
            kl = torch.tensor(kl_values, dtype=torch.float32)
        return Phase2PolicyRollout(
            arm_name=deployment.arm_name,
            trajectories=tuple(trajectories),
            per_sequence_kl_updated_to_reference=kl,
            kl_orientation=self.backend.kl_orientation,
            history_source=self.backend.history_source,
        )


class _FakeBackend:
    def __init__(
        self,
        *,
        kl_by_arm: dict[str, float] | None = None,
        kl_orientation: str = KL_ORIENTATION,
        history_source: str = KL_HISTORY_SOURCE,
        mismatched_crn: bool = False,
        heldout_target_scale: float = 1.0,
        train_oracle_scale: float = 1.0,
        kl_values_by_arm: dict[str, Sequence[float]] | None = None,
        reached_max_length_by_arm: dict[str, frozenset[int]] | None = None,
        expected_seed: int = 20260722,
    ) -> None:
        self.events: list[str] = []
        self.deployments: list[Phase2ArmDeployment] = []
        self.oracle_inputs: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = []
        self.resident: str | None = None
        self.oracle_session_count = 0
        self.kl_by_arm = kl_by_arm or {
            "zero_b": 0.0,
            BT_MLE: 0.001,
            PRORM_PLUS: 0.002,
            "oracle_step": 0.003,
        }
        self.kl_values_by_arm = kl_values_by_arm or {}
        self.reached_max_length_by_arm = reached_max_length_by_arm or {}
        self.kl_orientation = kl_orientation
        self.history_source = history_source
        self.mismatched_crn = mismatched_crn
        self.heldout_target_scale = heldout_target_scale
        self.train_oracle_scale = train_oracle_scale
        self.expected_seed = expected_seed

    @contextmanager
    def oracle_session(
        self,
        *,
        expected_chat_template_sha256: str,
    ) -> Iterator[_FakeOracleSession]:
        assert expected_chat_template_sha256 == "d" * 64
        assert self.resident is None
        phase = "train" if self.oracle_session_count == 0 else "test"
        self.oracle_session_count += 1
        self.resident = "oracle"
        self.events.append(f"oracle_open:{phase}")
        try:
            yield _FakeOracleSession(self, phase)
        finally:
            self.events.append(f"oracle_close:{phase}")
            self.resident = None

    @contextmanager
    def policy_session(
        self,
        *,
        seed: int,
        expected_a_sha256: str,
        expected_layout: ParameterLayout,
        expected_chat_template_sha256: str,
    ) -> Iterator[_FakePolicySession]:
        assert seed == self.expected_seed
        assert expected_a_sha256 == "b" * 64
        assert expected_layout.dimension == 2
        assert expected_chat_template_sha256 == "c" * 64
        assert self.resident is None
        self.resident = "policy"
        self.events.append("policy_open")
        try:
            yield _FakePolicySession(self)
        finally:
            self.events.append("policy_close")
            self.resident = None


class _FakeHeadTrainer:
    def __init__(self, backend: _FakeBackend) -> None:
        self.backend = backend
        self.received_oracle_rewards: torch.Tensor | None = None
        self.result: Phase2HeadTrainingResult | None = None

    def train_heads(
        self,
        train: TrainingTensorData,
        train_oracle_rewards: torch.Tensor,
        *,
        seed: int,
    ) -> Phase2HeadTrainingResult:
        assert self.backend.resident is None
        assert seed == self.backend.expected_seed
        assert train_oracle_rewards.shape == (
            train.num_prompts,
            train.num_candidates,
        )
        self.backend.events.append("train_heads:r4")
        self.received_oracle_rewards = train_oracle_rewards.detach().clone()
        self.result = Phase2HeadTrainingResult(
            heads={
                BT_MLE: (1.0, 0.25),
                PRORM_PLUS: (0.25, 1.0),
            },
            training_design_sha256="2" * 64,
            training_arm="r4_independent_gamma_0.9",
            audit={
                "independent_replicates_per_edge": 4,
                "aggregation": "arithmetic_mean",
                "clipping": False,
                "bt_uses_all_bernoulli_labels": True,
                "nested_vector_probe": {
                    "head_weight": [1.0, 2.0],
                    "direction": [3.0, 4.0],
                    "moment": [5.0, 6.0],
                    "gradient_l2_norm": 0.125,
                },
            },
        )
        return self.result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _confirmatory_design() -> Phase2Design:
    return replace(
        Phase2Design(),
        stage="confirmatory",
        formal_eligibility=True,
        pilot_phase=None,
        common_beta_rule=CONFIRMATORY_COMMON_BETA_RULE,
        common_beta_calibration_split="excluded_pilot",
        common_beta_source="frozen_pilot_global_beta_in_confirmatory_design_identity",
        frozen_global_beta=2.5,
        beta_source_aggregate_sha256="f" * 64,
        parent_pilot_aggregate_sha256="f" * 64,
        k_cal_sensitivity_values=None,
        frozen_global_beta_sensitivity_multipliers=(0.5, 2.0),
        max_length_formal_gate=True,
        max_length_formal_threshold=0.05,
    )


def _freeze_design() -> Phase2Design:
    return replace(
        Phase2Design(),
        pilot_phase="freeze",
        common_beta_rule=PILOT_FREEZE_COMMON_BETA_RULE,
        common_beta_calibration_split="excluded_pilot_calibration",
        common_beta_source=(
            "frozen_calibration_aggregate_candidate_in_pilot_freeze_design_identity"
        ),
        frozen_global_beta=3.0,
        beta_source_aggregate_sha256="a" * 64,
        parent_pilot_aggregate_sha256="a" * 64,
        k_cal_sensitivity_values=None,
    )


def _budgeted_design() -> Phase2Design:
    return replace(
        Phase2Design(),
        stage="budgeted_end_to_end",
        formal_eligibility=False,
        pilot_phase=None,
        common_beta_rule=BUDGETED_COMMON_BETA_RULE,
        common_beta_calibration_split="excluded_pilot",
        common_beta_source=("accepted_freeze_global_beta_in_budgeted_end_to_end_design_identity"),
        frozen_global_beta=2.5,
        beta_source_aggregate_sha256="f" * 64,
        parent_pilot_aggregate_sha256="f" * 64,
        k_cal_sensitivity_values=None,
        frozen_global_beta_sensitivity_multipliers=None,
        max_length_formal_gate=False,
        max_length_formal_threshold=0.05,
    )


def test_runtime_design_is_extracted_from_the_validated_phase2_overlay() -> None:
    repository = Path(__file__).resolve().parents[1]
    bundle = load_phase2_config_bundle(repository / "configs" / "common_beta_pilot.yaml")

    design = Phase2Design.from_phase2_config(bundle.config)

    assert design.stage == "pilot"
    assert design.formal_eligibility is False
    assert design.pilot_phase == "calibration"
    assert design.frozen_global_beta is None
    assert design.beta_source_aggregate_sha256 is None
    assert design.target_oracle_quadratic_kl == 0.003
    assert design.measured_kl_safety_cap == 0.02
    assert design.prompt_mean_p95_kl_cap == 0.02
    assert design.prompt_mean_p99_kl_cap == 0.05
    assert design.prompt_mean_maximum_kl_cap == 0.10
    assert design.per_sequence_maximum_kl_cap == 0.20
    assert design.max_response_tokens == 256
    assert design.allowed_horizon_sequence == (256, 512, 1024)
    assert design.horizon_grid_index == 0
    assert design.parent_pilot_aggregate_sha256 is None
    assert design.previous_horizon_failed_length_gate is False
    assert design.rollout_candidates_per_prompt == 4
    assert design.k_cal_sensitivity_values == (0.001, 0.01)
    assert design.frozen_global_beta_sensitivity_multipliers is None
    assert design.ridge_sensitivity_multipliers == (0.1, 1.0, 10.0)
    assert design.to_dict()["learner_specific_line_search"] is False
    assert design.to_dict()["common_beta_rule"] == (
        "pilot_seed_candidate_from_oracle_train_fisher_quadratic_for_future_global_beta"
    )
    assert design.to_dict()["current_seed_oracle_curvature_role"] == (
        "pilot_beta_candidate_calibration"
    )
    assert design.to_dict()["max_length_gate"] == {
        "formal_gate": False,
        "formal_threshold": 0.05,
        "measure_only": True,
    }


def test_runtime_design_requires_identity_bound_monotone_horizon_escalation() -> None:
    escalated = replace(
        Phase2Design(),
        max_response_tokens=512,
        horizon_grid_index=1,
        parent_pilot_aggregate_sha256="e" * 64,
        previous_horizon_failed_length_gate=True,
    )

    assert escalated.max_response_tokens == 512
    assert escalated.to_dict()["allowed_horizon_sequence"] == [256, 512, 1024]
    assert escalated.to_dict()["horizon_grid_index"] == 1
    assert escalated.to_dict()["parent_pilot_aggregate_sha256"] == "e" * 64
    assert escalated.to_dict()["previous_horizon_failed_length_gate"] is True
    with pytest.raises(ValueError, match="max_response_tokens must equal"):
        replace(Phase2Design(), max_response_tokens=512)
    with pytest.raises(ValueError, match="parent_pilot_aggregate_sha256"):
        replace(
            Phase2Design(),
            max_response_tokens=512,
            horizon_grid_index=1,
            previous_horizon_failed_length_gate=True,
        )
    with pytest.raises(ValueError, match="failed previous length gate"):
        replace(
            Phase2Design(),
            max_response_tokens=512,
            horizon_grid_index=1,
            parent_pilot_aggregate_sha256="e" * 64,
        )
    with pytest.raises(ValueError, match="initial pilot calibration"):
        replace(
            Phase2Design(),
            parent_pilot_aggregate_sha256="e" * 64,
        )


def test_runtime_design_rejects_stage_inconsistent_global_beta_contracts() -> None:
    with pytest.raises(ValueError, match="pilot calibration frozen_global_beta"):
        Phase2Design(frozen_global_beta=2.5)
    with pytest.raises(ValueError, match="common_beta_rule"):
        replace(
            Phase2Design(),
            stage="confirmatory",
            formal_eligibility=True,
            pilot_phase=None,
            frozen_global_beta=2.5,
            beta_source_aggregate_sha256="f" * 64,
            parent_pilot_aggregate_sha256="f" * 64,
            max_length_formal_gate=True,
            max_length_formal_threshold=0.05,
        )
    with pytest.raises((TypeError, ValueError), match="frozen_global_beta"):
        replace(
            Phase2Design(),
            stage="confirmatory",
            formal_eligibility=True,
            pilot_phase=None,
            common_beta_rule=CONFIRMATORY_COMMON_BETA_RULE,
            common_beta_calibration_split="excluded_pilot",
            common_beta_source="frozen_pilot_global_beta_in_confirmatory_design_identity",
            beta_source_aggregate_sha256="f" * 64,
            max_length_formal_gate=True,
            max_length_formal_threshold=0.05,
        )
    with pytest.raises(ValueError, match="confirmatory K_cal sensitivities"):
        replace(
            Phase2Design(),
            stage="confirmatory",
            formal_eligibility=True,
            pilot_phase=None,
            common_beta_rule=CONFIRMATORY_COMMON_BETA_RULE,
            common_beta_calibration_split="excluded_pilot",
            common_beta_source="frozen_pilot_global_beta_in_confirmatory_design_identity",
            frozen_global_beta=2.5,
            beta_source_aggregate_sha256="f" * 64,
            parent_pilot_aggregate_sha256="f" * 64,
            max_length_formal_gate=True,
            max_length_formal_threshold=0.05,
        )


def test_pre_oracle_gate_enforces_every_preregistered_tail_and_length_cap() -> None:
    values = [0.0] * 400
    for prompt_index in range(10):
        values[prompt_index * 4 : prompt_index * 4 + 4] = [0.03] * 4
    for prompt_index in range(10, 13):
        values[prompt_index * 4 : prompt_index * 4 + 4] = [0.06] * 4
    values[13 * 4 : 13 * 4 + 4] = [0.11] * 4
    values[14 * 4] = 0.21
    gate = assess_phase2_pre_oracle_safety(
        _synthetic_safety_rollouts(
            prorm_kl=values,
            prorm_max_length_indices=frozenset(range(21)),
        ),
        design=_confirmatory_design(),
    )

    assert gate.mean_kl_safety.passed is True
    assert gate.passed is False
    assert gate.to_dict()["thresholds"] == {
        "mean_policy_to_reference_kl_cap": 0.02,
        "prompt_mean_p95_kl_cap": 0.02,
        "prompt_mean_p99_kl_cap": 0.05,
        "prompt_mean_maximum_kl_cap": 0.10,
        "per_sequence_maximum_kl_cap": 0.20,
        "reached_max_length_rate_cap": 0.05,
    }
    assert set(gate.violations) == {
        f"{PRORM_PLUS}:prompt_mean_p95_kl",
        f"{PRORM_PLUS}:prompt_mean_p99_kl",
        f"{PRORM_PLUS}:prompt_mean_maximum_kl",
        f"{PRORM_PLUS}:per_sequence_maximum_kl",
        f"{PRORM_PLUS}:reached_max_length_rate",
    }


def test_end_to_end_state_machine_common_beta_information_boundary_and_output(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    backend = _FakeBackend(
        reached_max_length_by_arm={arm_name: frozenset({1, 3}) for arm_name in PHASE2_ARM_ORDER}
    )
    head_trainer = _FakeHeadTrainer(backend)
    destination = tmp_path / "phase2.json"

    result = run_common_beta_rollouts(
        inputs,
        head_trainer,
        backend,
        output_json=destination,
    )

    assert result["design_stage"] == "pilot"
    assert result["pilot_phase"] == "calibration"
    assert result["formal_eligibility"] is False
    assert result["per_seed_supports_formal_claim"] is False
    assert backend.resident is None
    assert backend.events == [
        "oracle_open:train",
        "oracle_score:train",
        "oracle_close:train",
        "train_heads:r4",
        "policy_open",
        "rollout:zero_b",
        f"rollout:{BT_MLE}",
        f"rollout:{PRORM_PLUS}",
        "rollout:oracle_step",
        "policy_close",
    ]
    assert backend.oracle_session_count == 1
    assert [deployment.arm_name for deployment in backend.deployments] == list(PHASE2_ARM_ORDER)
    beta_values = {deployment.beta_common for deployment in backend.deployments}
    assert len(beta_values) == 1
    beta = beta_values.pop()
    assert beta == result["train_only_global_beta_calibration_candidate"]["candidate_beta"]
    assert torch.count_nonzero(backend.deployments[0].displacement) == 0
    for deployment in backend.deployments[1:]:
        natural = torch.tensor(deployment.direction_evidence["direction"], dtype=torch.float64)
        assert torch.allclose(
            deployment.displacement.to(torch.float64),
            natural / beta,
        )

    train_phase, train_prompts, train_responses = backend.oracle_inputs[0]
    assert train_phase == "train"
    assert len(train_prompts) == inputs.train.num_prompts * inputs.train.num_candidates
    assert all(prompt.startswith("train prompt") for prompt in train_prompts)
    assert all(response.startswith("train response") for response in train_responses)
    assert not any(prompt.startswith("test prompt") for prompt in train_prompts)
    assert result["schema_version"] == PHASE2_PILOT_RESULT_SCHEMA
    expected_boundary = {
        "calibration_split": "train_only",
        "new_rollout_prompts_used_for_calibration": False,
        "final_oracle_session_opened": False,
        "rollout_responses_oracle_scored": False,
        "heldout_evaluator_called": False,
        "oracle_outcomes_serialized": False,
        "prompt_or_response_text_serialized": False,
        "token_ids_or_response_masks_serialized": False,
        "source_artifact_heldout_targets_exposed_by_phase2_prepared_inputs": False,
    }
    assert {
        key: result["information_boundary"][key] for key in expected_boundary
    } == expected_boundary
    prompt_semantics = result["information_boundary"]["prompt_semantics"]
    assert prompt_semantics["materialization"]["truncated_prompt_count"] == 0
    assert prompt_semantics["rollout"]["truncated_prompt_count"] == 0
    assert prompt_semantics["rollout"]["matches_materialization_token_prefix_evidence"] is True
    assert prompt_semantics["oracle"]["rerendered_with_independent_oracle_chat_template"] is True
    assert prompt_semantics["oracle"]["policy_chat_tokens_reused_by_oracle"] is False
    assert result["learner_specific_line_search"] is False
    assert head_trainer.result is not None
    assert result["head_training"] == {
        "training_arm": "r4_independent_gamma_0.9",
        "training_design_sha256": "2" * 64,
        "heads_sha256": head_trainer.result.heads_sha256,
        "head_weights_serialized": False,
        "audit": {
            "independent_replicates_per_edge": 4,
            "aggregation": "arithmetic_mean",
            "clipping": False,
            "bt_uses_all_bernoulli_labels": True,
            "nested_vector_probe": {"gradient_l2_norm": 0.125},
        },
        "audit_vector_fields_redacted": [
            "direction",
            "displacement",
            "head_weight",
            "head_weights",
            "moment",
            "natural_direction",
            "operator_direction",
            "oracle_displacement",
            "projection_matrix",
            "true_rewards",
        ],
        "source": "trained_after_train_oracle_rescore",
        "old_phase1_comparison_heads_reused": False,
        "test_data_accessed": False,
    }
    assert result["source_config_hash"] == _DIGEST
    assert result["phase2_design_sha256"] == "2" * 64
    assert result["run_manifest"] == "run_manifest.json"
    assert result["run_manifest_sha256"] == _sha256(inputs.run_manifest)
    assert result["environment_identity"] == collect_execution_identity()
    assert result["current_process_identity"] == collect_execution_identity()
    assert result["phase2_runtime_contract_sha256"] == Phase2Design().sha256
    assert result["phase2_runtime_contract"]["sensitivity_scope"] == {
        "pilot_k_cal_candidates": [0.001, 0.01],
        "frozen_global_beta_multipliers": None,
        "sensitivity_step_rule": "recalibrate_pilot_seed_candidate_from_k_cal",
        "ridge_multipliers_configured": [0.1, 1.0, 10.0],
        "executed_by_this_runner_invocation": False,
        "result_role": "primary_only",
    }
    assert result["common_random_numbers"] == {
        "named_stream": "rollout",
        "seed": result["common_random_numbers"]["seed"],
        "same_per_prompt_seed_reset_across_arms": True,
        "candidate_index_alignment": True,
    }
    assert result["measured_kl_safety"]["passed"] is True
    pre_oracle = result["pre_oracle_safety_gate"]
    assert pre_oracle["schema_version"] == "phase2-pre-oracle-safety-gate/v1"
    assert pre_oracle["design_stage"] == "pilot"
    assert pre_oracle["pilot_phase"] == "calibration"
    assert pre_oracle["measure_only"] is True
    assert pre_oracle["formal_gate"] is False
    assert pre_oracle["thresholds"] == {
        "mean_policy_to_reference_kl_cap": 0.02,
        "prompt_mean_p95_kl_cap": 0.02,
        "prompt_mean_p99_kl_cap": 0.05,
        "prompt_mean_maximum_kl_cap": 0.10,
        "per_sequence_maximum_kl_cap": 0.20,
        "reached_max_length_rate_cap": 0.05,
    }
    assert pre_oracle["passed"] is False
    assert pre_oracle["beta_retuned"] is False
    assert any(
        violation.endswith(":reached_max_length_rate") for violation in pre_oracle["violations"]
    )
    assert result["pilot_kl_safety_gate"] == {
        "schema_version": "pilot-measured-kl-gate/v1",
        "gate_passed": True,
        "measure_only": True,
        "supports_formal_claim": False,
        "violations": [],
        "on_violation": "publish_target_free_diagnostics_without_final_oracle",
    }
    assert list(result["arms"]) == list(PHASE2_ARM_ORDER)
    assert result["arms"][PRORM_PLUS]["mean_on_policy_kl_pi_updated_to_pi0"] == pytest.approx(0.002)
    kl_tail = result["arms"][PRORM_PLUS]["on_policy_kl_tail"]
    assert kl_tail["unit"] == "prompt_mean_over_candidates"
    assert kl_tail["num_prompts"] == 2
    assert kl_tail["candidates_per_prompt"] == 4
    assert kl_tail["p95"] == pytest.approx(0.002)
    assert kl_tail["maximum"] == pytest.approx(0.002)
    assert kl_tail["formal_gate_applied"] is False
    assert result["arms"][BT_MLE]["rollout_length"]["terminated_by_eos_rate"] == 0.5
    assert result["arms"][BT_MLE]["rollout_length"]["reached_max_length_rate"] == 0.5

    diagnostics_path = tmp_path / "phase2.diagnostics.jsonl"
    assert destination.exists() and diagnostics_path.exists()
    assert result["diagnostics_sha256"] == _sha256(diagnostics_path)
    assert json.loads(destination.read_text(encoding="utf-8")) == result
    records = [
        json.loads(line) for line in diagnostics_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == len(PHASE2_ARM_ORDER) * 2 * 4
    assert [records[index * 8]["arm"] for index in range(4)] == list(PHASE2_ARM_ORDER)
    assert all(record["schema_version"] == PHASE2_PILOT_DIAGNOSTIC_SCHEMA for record in records)
    assert all(record["pilot_phase"] == "calibration" for record in records)
    assert all(record["beta_common"] == beta for record in records)
    assert all(record["beta_role"] == "seed_calibration_candidate" for record in records)
    assert all(record["kl_orientation"] == KL_ORIENTATION for record in records)
    assert all(record["kl_history_source"] == KL_HISTORY_SOURCE for record in records)
    forbidden = {
        "head_weight",
        "head_weights",
        "prompt",
        "response",
        "token_ids",
        "response_mask",
        "reward",
        "transformed_oracle_reward",
        "target_utility",
        "utility",
        "regret",
        "heldout",
        "heldout_fixed_beta",
        "true_rewards",
        "local_regret_at_frozen_global_beta",
    }

    def nested_keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return set(value).union(
                *(nested_keys(item) for item in value.values()),
            )
        if isinstance(value, list):
            return set().union(*(nested_keys(item) for item in value))
        return set()

    assert all(forbidden.isdisjoint(nested_keys(record)) for record in records)
    assert forbidden.isdisjoint(nested_keys(result))


def test_pilot_kl_violation_publishes_target_free_measure_only_diagnostics(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    backend = _FakeBackend(
        kl_by_arm={
            "zero_b": 0.0,
            BT_MLE: 0.001,
            PRORM_PLUS: 0.021,
            "oracle_step": 0.003,
        }
    )
    destination = tmp_path / "unsafe.json"
    head_trainer = _FakeHeadTrainer(backend)

    result = run_common_beta_rollouts(
        inputs,
        head_trainer,
        backend,
        output_json=destination,
    )

    assert result["measured_kl_safety"]["passed"] is False
    assert result["measured_kl_safety"]["violations"] == [PRORM_PLUS]
    assert result["pilot_kl_safety_gate"] == {
        "schema_version": "pilot-measured-kl-gate/v1",
        "gate_passed": False,
        "measure_only": True,
        "supports_formal_claim": False,
        "violations": [PRORM_PLUS],
        "on_violation": "publish_target_free_diagnostics_without_final_oracle",
    }
    assert result["arms"][PRORM_PLUS]["mean_on_policy_kl_pi_updated_to_pi0"] == pytest.approx(0.021)
    assert result["arms"][PRORM_PLUS]["on_policy_kl_tail"]["maximum"] == pytest.approx(0.021)
    unified = result["pre_oracle_safety_gate"]
    assert unified["measure_only"] is True
    assert unified["formal_gate"] is False
    assert unified["passed"] is False
    assert f"{PRORM_PLUS}:mean_policy_to_reference_kl" in unified["violations"]
    assert f"{PRORM_PLUS}:prompt_mean_p95_kl" in unified["violations"]
    assert backend.oracle_session_count == 1
    assert "oracle_open:test" not in backend.events
    diagnostics = tmp_path / "unsafe.diagnostics.jsonl"
    assert destination.exists()
    assert diagnostics.exists()
    assert result["diagnostics_sha256"] == _sha256(diagnostics)
    records = [json.loads(line) for line in diagnostics.read_text(encoding="utf-8").splitlines()]
    assert len(records) == len(PHASE2_ARM_ORDER) * 2 * 4
    assert all(record["contains_oracle_outcome"] is False for record in records)


def test_pilot_freeze_uses_fixed_global_beta_and_remains_target_free(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    backend = _FakeBackend()
    destination = tmp_path / "freeze.json"

    result = run_common_beta_rollouts(
        inputs,
        _FakeHeadTrainer(backend),
        backend,
        output_json=destination,
        design=_freeze_design(),
    )

    assert result["pilot_phase"] == "freeze"
    assert "train_only_global_beta_calibration_candidate" not in result
    rehearsal = result["pilot_fixed_global_beta_rehearsal"]
    assert rehearsal == {
        "schema_version": "pilot-frozen-global-beta-rehearsal/v1",
        "rule": PILOT_FREEZE_COMMON_BETA_RULE,
        "beta_common": 3.0,
        "frozen_global_beta": 3.0,
        "beta_matches_frozen_global_beta": True,
        "beta_source_aggregate_sha256": "a" * 64,
        "current_seed_oracle_natural_curvature": rehearsal["current_seed_oracle_natural_curvature"],
        "reference_target_oracle_quadratic_kl": 0.003,
        "predicted_current_seed_oracle_quadratic_kl": rehearsal[
            "predicted_current_seed_oracle_quadratic_kl"
        ],
        "current_seed_curvature_role": "predicted_kl_diagnostic_only",
        "beta_selected_from_current_seed_curvature": False,
        "frozen_in_phase2_design_identity": True,
        "learner_specific_rescaling": False,
        "post_evaluation_retuning": False,
    }
    assert {deployment.beta_common for deployment in backend.deployments} == {3.0}
    assert result["pre_oracle_safety_gate"]["measure_only"] is True
    assert result["pre_oracle_safety_gate"]["formal_gate"] is False
    assert backend.oracle_session_count == 1
    assert "oracle_open:test" not in backend.events
    rows = [
        json.loads(line)
        for line in (tmp_path / "freeze.diagnostics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert all(row["pilot_phase"] == "freeze" for row in rows)
    assert all(row["beta_common"] == 3.0 for row in rows)
    assert all(row["beta_role"] == "frozen_global_beta_candidate" for row in rows)


def test_confirmatory_kl_violation_fails_before_final_oracle_and_publishes_nothing(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    backend = _FakeBackend(
        kl_by_arm={
            "zero_b": 0.0,
            BT_MLE: 0.001,
            PRORM_PLUS: 0.021,
            "oracle_step": 0.003,
        }
    )
    destination = tmp_path / "unsafe-confirmatory.json"
    head_trainer = _FakeHeadTrainer(backend)

    with pytest.raises(Phase2PreOracleSafetyError) as error:
        run_common_beta_rollouts(
            inputs,
            head_trainer,
            backend,
            output_json=destination,
            design=_confirmatory_design(),
        )

    assert error.value.safety.passed is False
    assert error.value.safety.violations == (PRORM_PLUS,)
    assert f"{PRORM_PLUS}:mean_policy_to_reference_kl" in error.value.pre_oracle_safety.violations
    assert backend.oracle_session_count == 1
    assert "oracle_open:test" not in backend.events
    assert not destination.exists()
    assert not (tmp_path / "unsafe-confirmatory.rollouts.jsonl").exists()


def test_confirmatory_kl_tail_violation_fails_before_final_oracle(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    backend = _FakeBackend(
        kl_values_by_arm={
            PRORM_PLUS: (0.03, 0.03, 0.03, 0.03, 0.0, 0.0, 0.0, 0.0),
        }
    )
    destination = tmp_path / "unsafe-tail-confirmatory.json"

    with pytest.raises(Phase2PreOracleSafetyError) as error:
        run_common_beta_rollouts(
            inputs,
            _FakeHeadTrainer(backend),
            backend,
            output_json=destination,
            design=_confirmatory_design(),
        )

    gate = error.value.pre_oracle_safety
    assert gate.mean_kl_safety.passed is True
    assert gate.mean_kl_safety.violations == ()
    assert f"{PRORM_PLUS}:prompt_mean_p95_kl" in gate.violations
    assert backend.oracle_session_count == 1
    assert "oracle_open:test" not in backend.events
    assert not destination.exists()
    assert not (tmp_path / "unsafe-tail-confirmatory.rollouts.jsonl").exists()


def test_confirmatory_max_length_violation_fails_before_final_oracle(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    backend = _FakeBackend(
        reached_max_length_by_arm={PRORM_PLUS: frozenset({0})},
    )
    destination = tmp_path / "unsafe-length-confirmatory.json"

    with pytest.raises(Phase2PreOracleSafetyError) as error:
        run_common_beta_rollouts(
            inputs,
            _FakeHeadTrainer(backend),
            backend,
            output_json=destination,
            design=_confirmatory_design(),
        )

    gate = error.value.pre_oracle_safety
    assert gate.mean_kl_safety.passed is True
    assert f"{PRORM_PLUS}:reached_max_length_rate" in gate.violations
    assert backend.oracle_session_count == 1
    assert "oracle_open:test" not in backend.events
    assert not destination.exists()
    assert not (tmp_path / "unsafe-length-confirmatory.rollouts.jsonl").exists()


def test_heldout_targets_cannot_change_heads_beta_or_policy_deployments(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    backend_a = _FakeBackend(heldout_target_scale=1.0)
    backend_b = _FakeBackend(heldout_target_scale=7.0)
    trainer_a = _FakeHeadTrainer(backend_a)
    trainer_b = _FakeHeadTrainer(backend_b)

    result_a = run_common_beta_rollouts(
        inputs,
        trainer_a,
        backend_a,
        output_json=tmp_path / "heldout-a.json",
        design=_confirmatory_design(),
    )
    result_b = run_common_beta_rollouts(
        inputs,
        trainer_b,
        backend_b,
        output_json=tmp_path / "heldout-b.json",
        design=_confirmatory_design(),
    )

    assert result_a["head_training"] == result_b["head_training"]
    for result in (result_a, result_b):
        assert result["pre_oracle_safety_gate"]["passed"] is True
        assert result["pre_oracle_safety_gate"]["formal_gate"] is True
        assert result["pre_oracle_safety_gate"]["measure_only"] is False
        assert result["arms"][PRORM_PLUS]["on_policy_kl_tail"]["formal_gate_applied"] is True
    assert result_a["common_beta_calibration"] == result_b["common_beta_calibration"]
    assert result_a["train_oracle_direction"] == result_b["train_oracle_direction"]
    assert result_a["arms"] == result_b["arms"]
    assert len(backend_a.deployments) == len(backend_b.deployments)
    for deployment_a, deployment_b in zip(
        backend_a.deployments,
        backend_b.deployments,
        strict=True,
    ):
        assert deployment_a.arm_name == deployment_b.arm_name
        assert deployment_a.beta_common == deployment_b.beta_common
        assert torch.equal(deployment_a.displacement, deployment_b.displacement)
    assert (
        result_a["heldout_fixed_beta"]["splits"]["test"]["learners"][BT_MLE][
            "local_regret_at_frozen_global_beta"
        ]
        != result_b["heldout_fixed_beta"]["splits"]["test"]["learners"][BT_MLE][
            "local_regret_at_frozen_global_beta"
        ]
    )
    assert backend_a.events.index("oracle_score:heldout") > backend_a.events.index(
        "rollout:oracle_step"
    )
    assert backend_b.events.index("oracle_score:heldout") > backend_b.events.index(
        "rollout:oracle_step"
    )


def test_budgeted_end_to_end_reuses_confirmatory_numerical_event_sequence_but_not_claim_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    budgeted_dir = tmp_path / "budgeted"
    confirmatory_dir = tmp_path / "confirmatory"
    budgeted_dir.mkdir()
    confirmatory_dir.mkdir()
    budgeted_inputs = replace(_inputs(budgeted_dir), seed=20261001)
    confirmatory_inputs = _inputs(confirmatory_dir)
    budgeted_backend = _FakeBackend(expected_seed=20261001)
    confirmatory_backend = _FakeBackend()
    stage_traces: dict[str, list[str]] = {
        "budgeted_end_to_end": [],
        "confirmatory": [],
    }

    def traced(name: str, function: object) -> object:
        def wrapper(*args: object, **kwargs: object) -> object:
            design = kwargs["design"]
            assert isinstance(design, Phase2Design)
            stage_traces[design.stage].append(name)
            return function(*args, **kwargs)

        return wrapper

    for name in (
        "_freeze_heldout_evaluation_state",
        "_rollout_policy_arms",
        "assess_phase2_pre_oracle_safety",
        "_score_final_rollouts",
    ):
        original = getattr(phase2_rollout_module, name)
        monkeypatch.setattr(phase2_rollout_module, name, traced(name, original))

    budgeted = run_common_beta_rollouts(
        budgeted_inputs,
        _FakeHeadTrainer(budgeted_backend),
        budgeted_backend,
        output_json=budgeted_dir / "result.json",
        design=_budgeted_design(),
    )
    confirmatory = run_common_beta_rollouts(
        confirmatory_inputs,
        _FakeHeadTrainer(confirmatory_backend),
        confirmatory_backend,
        output_json=confirmatory_dir / "result.json",
        design=_confirmatory_design(),
    )

    assert (
        budgeted_backend.events
        == confirmatory_backend.events
        == [
            "oracle_open:train",
            "oracle_score:train",
            "oracle_close:train",
            "train_heads:r4",
            "policy_open",
            "rollout:zero_b",
            f"rollout:{BT_MLE}",
            f"rollout:{PRORM_PLUS}",
            "rollout:oracle_step",
            "policy_close",
            "oracle_open:test",
            "oracle_score:test",
            "oracle_score:heldout",
            "oracle_close:test",
        ]
    )
    expected_trace = [
        "_freeze_heldout_evaluation_state",
        "_rollout_policy_arms",
        "assess_phase2_pre_oracle_safety",
        "_score_final_rollouts",
    ]
    assert stage_traces["budgeted_end_to_end"] == expected_trace
    assert stage_traces["confirmatory"] == expected_trace
    assert budgeted["schema_version"] == PHASE2_BUDGETED_RESULT_SCHEMA
    assert confirmatory["schema_version"] != PHASE2_BUDGETED_RESULT_SCHEMA
    assert budgeted["design_stage"] == "budgeted_end_to_end"
    assert budgeted["formal_eligibility"] is False
    assert budgeted["formal_claim_eligible"] is False
    assert budgeted["supports_formal_claim"] is False
    assert budgeted["per_seed_supports_formal_claim"] is False
    assert budgeted["excluded_from_confirmatory_evidence"] is True
    assert budgeted["confirmatory_authorization_created"] is False
    assert budgeted["numerical_event_sequence_matches_confirmatory"] is True
    assert "common_beta_calibration" not in budgeted
    beta_evidence = budgeted["common_beta_frozen_evidence"]
    assert beta_evidence["schema_version"] == "common-beta-frozen-global-budgeted/v1"
    assert beta_evidence["beta_common"] == 2.5
    assert beta_evidence["frozen_global_beta"] == 2.5
    assert beta_evidence["accepted_freeze_beta_reused_without_recalibration"] is True
    assert beta_evidence["beta_selected_from_current_seed_curvature"] is False
    assert beta_evidence["current_seed_can_change_beta"] is False
    assert budgeted["phase2_runtime_contract"]["sensitivity_scope"] == {
        "pilot_k_cal_candidates": None,
        "frozen_global_beta_multipliers": None,
        "sensitivity_step_rule": ("deploy_config_frozen_beta_without_seed_curvature_calibration"),
        "ridge_multipliers_configured": [0.1, 1.0, 10.0],
        "executed_by_this_runner_invocation": False,
        "result_role": "budgeted_end_to_end_exploratory_primary",
    }

    gate = budgeted["pre_oracle_safety_gate"]
    assert gate["schema_version"] == "phase2-pre-oracle-safety-gate/v2"
    assert gate["measure_only"] is False
    assert gate["formal_gate"] is False
    assert gate["enforced_before_final_oracle"] is True
    assert gate["supports_formal_claim"] is False
    assert gate["passed"] is True
    assert budgeted["arms"][PRORM_PLUS]["on_policy_kl_tail"]["formal_gate_applied"] is False

    heldout = budgeted["heldout_fixed_beta"]
    assert heldout["schema_version"] == "phase2-heldout-fixed-beta/v2"
    assert heldout["formal_claim_eligible"] is False
    for split_name in ("validation", "test"):
        split = heldout["splits"][split_name]
        assert set(split["preference_fit"]) == {BT_MLE, PRORM_PLUS}
        for learner in (BT_MLE, PRORM_PLUS):
            assert set(split["preference_fit"][learner]) == {
                "oracle_pairwise_cross_entropy",
                "oracle_probability_mae",
                "pairwise_order_accuracy",
            }
        assert split["heldout_pcg_evidence"]["all_solves_converged"] is True
        assert split["heldout_pcg_evidence"]["all_solves_cold_start"] is True
    assert confirmatory["heldout_fixed_beta"]["schema_version"] == ("phase2-heldout-fixed-beta/v1")
    assert "preference_fit" not in confirmatory["heldout_fixed_beta"]["splits"]["test"]

    rollout_rows = [
        json.loads(line)
        for line in (budgeted_dir / "result.rollouts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert rollout_rows
    assert all(row["schema_version"] == PHASE2_BUDGETED_ROLLOUT_SCHEMA for row in rollout_rows)
    assert all(row["formal_claim_eligible"] is False for row in rollout_rows)
    assert all(row["supports_formal_claim"] is False for row in rollout_rows)


def test_budgeted_end_to_end_safety_is_nonformal_but_fail_closed_before_oracle(
    tmp_path: Path,
) -> None:
    inputs = replace(_inputs(tmp_path), seed=20261001)
    backend = _FakeBackend(
        expected_seed=20261001,
        kl_by_arm={
            "zero_b": 0.0,
            BT_MLE: 0.001,
            PRORM_PLUS: 0.021,
            "oracle_step": 0.003,
        },
    )
    destination = tmp_path / "unsafe-budgeted.json"

    with pytest.raises(Phase2PreOracleSafetyError) as error:
        run_common_beta_rollouts(
            inputs,
            _FakeHeadTrainer(backend),
            backend,
            output_json=destination,
            design=_budgeted_design(),
        )

    serialized_gate = error.value.pre_oracle_safety.to_dict()
    assert serialized_gate["measure_only"] is False
    assert serialized_gate["formal_gate"] is False
    assert serialized_gate["enforced_before_final_oracle"] is True
    assert serialized_gate["supports_formal_claim"] is False
    assert serialized_gate["passed"] is False
    assert backend.oracle_session_count == 1
    assert "oracle_open:test" not in backend.events
    assert not destination.exists()
    assert not (tmp_path / "unsafe-budgeted.rollouts.jsonl").exists()


@pytest.mark.parametrize("invalid_seed", [20261004, 20261005])
def test_budgeted_end_to_end_rejects_historical_fixed_five_only_seed_before_backend_access(
    tmp_path: Path,
    invalid_seed: int,
) -> None:
    backend = _FakeBackend()
    with pytest.raises(ValueError, match="fixed exploratory seed list"):
        run_common_beta_rollouts(
            replace(_inputs(tmp_path), seed=invalid_seed),
            _FakeHeadTrainer(backend),
            backend,
            output_json=tmp_path / "wrong-budgeted-seed.json",
            design=_budgeted_design(),
        )
    assert backend.events == []


def test_confirmatory_seed_curvature_cannot_change_the_frozen_global_beta(
    tmp_path: Path,
) -> None:
    design = _confirmatory_design()
    seed_a = tmp_path / "seed-a"
    seed_b = tmp_path / "seed-b"
    seed_a.mkdir()
    seed_b.mkdir()
    inputs_a = _inputs(seed_a)
    inputs_b = _inputs(seed_b)
    backend_a = _FakeBackend(train_oracle_scale=1.0)
    backend_b = _FakeBackend(train_oracle_scale=3.0)

    result_a = run_common_beta_rollouts(
        inputs_a,
        _FakeHeadTrainer(backend_a),
        backend_a,
        output_json=tmp_path / "seed-a.json",
        design=design,
    )
    result_b = run_common_beta_rollouts(
        inputs_b,
        _FakeHeadTrainer(backend_b),
        backend_b,
        output_json=tmp_path / "seed-b.json",
        design=design,
    )

    for result in (result_a, result_b):
        evidence = result["common_beta_calibration"]
        assert evidence["schema_version"] == "common-beta-frozen-global/v1"
        assert evidence["rule"] == CONFIRMATORY_COMMON_BETA_RULE
        assert evidence["beta_selection_split"] == "excluded_pilot"
        assert evidence["beta_source"] == (
            "frozen_pilot_global_beta_in_confirmatory_design_identity"
        )
        assert evidence["beta_common"] == 2.5
        assert evidence["frozen_global_beta"] == 2.5
        assert evidence["beta_matches_frozen_global_beta"] is True
        assert evidence["beta_selected_from_current_seed_curvature"] is False
        assert evidence["current_seed_curvature_role"] == "predicted_kl_diagnostic_only"
        assert result["phase2_runtime_contract"]["frozen_global_beta"] == 2.5
        assert (
            result["phase2_runtime_contract"]["sensitivity_scope"]["pilot_k_cal_candidates"] is None
        )
        assert result["phase2_runtime_contract"]["sensitivity_scope"][
            "frozen_global_beta_multipliers"
        ] == [0.5, 2.0]
        assert (
            result["phase2_runtime_contract"]["sensitivity_scope"]["sensitivity_step_rule"]
            == "multiply_config_frozen_global_beta_without_seed_curvature_calibration"
        )
    assert (
        result_a["common_beta_calibration"]["current_seed_oracle_natural_curvature"]
        != result_b["common_beta_calibration"]["current_seed_oracle_natural_curvature"]
    )
    assert (
        result_a["common_beta_calibration"]["predicted_current_seed_oracle_quadratic_kl"]
        != result_b["common_beta_calibration"]["predicted_current_seed_oracle_quadratic_kl"]
    )
    assert {deployment.beta_common for deployment in backend_a.deployments} == {2.5}
    assert {deployment.beta_common for deployment in backend_b.deployments} == {2.5}
    for index in (1, 2):
        torch.testing.assert_close(
            backend_a.deployments[index].displacement,
            backend_b.deployments[index].displacement,
        )


def test_head_training_must_bind_the_full_phase2_design(tmp_path: Path) -> None:
    inputs = replace(_inputs(tmp_path), phase2_config_hash="9" * 64)
    backend = _FakeBackend()

    with pytest.raises(ValueError, match="trainer design identity"):
        run_common_beta_rollouts(
            inputs,
            _FakeHeadTrainer(backend),
            backend,
            output_json=tmp_path / "wrong-head-design.json",
        )

    assert backend.events == [
        "oracle_open:train",
        "oracle_score:train",
        "oracle_close:train",
        "train_heads:r4",
    ]
    assert "policy_open" not in backend.events


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("kl_orientation", "pi0_to_pi_updated", r"KL\(pi_updated \|\| pi0\)"),
        ("history_source", "reference_policy", "histories"),
    ],
)
def test_reverse_kl_or_reference_histories_fail_closed(
    tmp_path: Path,
    field: str,
    value: str,
    match: str,
) -> None:
    inputs = _inputs(tmp_path)
    kwargs = {field: value}
    backend = _FakeBackend(**kwargs)
    head_trainer = _FakeHeadTrainer(backend)
    destination = tmp_path / f"bad-{field}.json"

    with pytest.raises(ValueError, match=match):
        run_common_beta_rollouts(
            inputs,
            head_trainer,
            backend,
            output_json=destination,
        )

    assert backend.oracle_session_count == 1
    assert not destination.exists()


@pytest.mark.parametrize("existing", ["result", "diagnostics"])
def test_existing_output_is_refused_before_any_backend_session(
    tmp_path: Path,
    existing: str,
) -> None:
    inputs = _inputs(tmp_path)
    backend = _FakeBackend()
    head_trainer = _FakeHeadTrainer(backend)
    destination = tmp_path / "locked.json"
    target = destination if existing == "result" else tmp_path / "locked.diagnostics.jsonl"
    target.write_text("owned by another run\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_common_beta_rollouts(
            inputs,
            head_trainer,
            backend,
            output_json=destination,
        )

    assert backend.events == []
    assert target.read_text(encoding="utf-8") == "owned by another run\n"


def test_zero_b_nonzero_kl_is_rejected_at_the_backend_contract(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    backend = _FakeBackend(
        kl_by_arm={
            "zero_b": 1.0e-6,
            BT_MLE: 0.001,
            PRORM_PLUS: 0.002,
            "oracle_step": 0.003,
        }
    )
    head_trainer = _FakeHeadTrainer(backend)

    with pytest.raises(ValueError, match="zero-B"):
        run_common_beta_rollouts(
            inputs,
            head_trainer,
            backend,
            output_json=tmp_path / "bad-zero.json",
        )


def test_common_random_number_seed_mismatch_fails_closed(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    backend = _FakeBackend(mismatched_crn=True)
    head_trainer = _FakeHeadTrainer(backend)

    with pytest.raises(ValueError, match="identical per-prompt rollout seeds"):
        run_common_beta_rollouts(
            inputs,
            head_trainer,
            backend,
            output_json=tmp_path / "bad-crn.json",
        )

    assert backend.oracle_session_count == 1
    assert not (tmp_path / "bad-crn.json").exists()


def test_tampered_run_manifest_is_rejected_before_backend_session(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    inputs.run_manifest.write_text('{"tampered":true}\n', encoding="utf-8")
    backend = _FakeBackend()

    with pytest.raises(ValueError, match="run manifest bytes"):
        run_common_beta_rollouts(
            inputs,
            _FakeHeadTrainer(backend),
            backend,
            output_json=tmp_path / "tampered.json",
        )

    assert backend.events == []


def test_tampered_deferred_heldout_geometry_is_rejected_before_backend_session(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    inputs.heldout.validation.reward_features[0, 0, 0] += 1.0
    backend = _FakeBackend()

    with pytest.raises(RuntimeError, match="reward features changed"):
        run_common_beta_rollouts(
            inputs,
            _FakeHeadTrainer(backend),
            backend,
            output_json=tmp_path / "tampered-heldout.json",
        )

    assert backend.events == []


def test_formal_environment_identity_must_match_current_process(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    formal_identity = {
        "formal": True,
        "git_commit": "3" * 40,
        "image_sha256": "4" * 64,
        "hf_inventory_sha256": "5" * 64,
        "account": "sigroup",
        "partition": "gpu",
        "gpu_models": ["fake"],
    }
    inputs = replace(inputs, environment_identity=formal_identity)
    backend = _FakeBackend()

    with pytest.raises(RuntimeError, match="current process identity"):
        run_common_beta_rollouts(
            inputs,
            _FakeHeadTrainer(backend),
            backend,
            output_json=tmp_path / "wrong-environment.json",
        )

    assert backend.events == []
