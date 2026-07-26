from __future__ import annotations

from pathlib import Path

import pytest
import torch

import smart_reward.phase2_training as phase2_training
from smart_reward.phase2_checkpoint import (
    CheckpointInterruption,
    DurableCheckpointStore,
)
from smart_reward.phase2_training import FirstOrderConvergenceSpec
from smart_reward.training import (
    BTMLETrainer,
    BTMLETrainingConfig,
    FeatureTrainingBatch,
    FrozenFeatureLinearReward,
    ProRMPlusTrainer,
    ProRMPlusTrainingConfig,
)


def _tiny_cpu_batch() -> FeatureTrainingBatch:
    dtype = torch.float64
    left_features = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, -1.0],
        ],
        dtype=dtype,
        device="cpu",
    )
    return FeatureTrainingBatch(
        left_features=left_features,
        right_features=torch.zeros_like(left_features),
        edge_scores=torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [1.0, -1.0],
            ],
            dtype=dtype,
            device="cpu",
        ),
        node_scores=torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [1.0, -1.0],
            ],
            dtype=dtype,
            device="cpu",
        ),
        h=torch.tensor([1.0, -0.7, 0.4, 1.3], dtype=dtype, device="cpu"),
        left_wins=torch.tensor([8, 2, 6, 7], dtype=torch.int64, device="cpu"),
        num_annotations=torch.tensor([10, 10, 10, 10], dtype=torch.int64, device="cpu"),
    )


def _trainer(
    kind: str,
    batch: FeatureTrainingBatch,
) -> BTMLETrainer | ProRMPlusTrainer:
    model = FrozenFeatureLinearReward(
        batch.reward_dimension,
        dtype=torch.float64,
        device="cpu",
    )
    if kind == "bt_mle":
        return BTMLETrainer(
            model,
            batch,
            BTMLETrainingConfig(
                learning_rate=0.03,
                optimizer="adamw",
                weight_decay=0.01,
                microbatch_size=2,
            ),
        )
    if kind == "prorm_plus":
        return ProRMPlusTrainer(
            model,
            batch,
            ProRMPlusTrainingConfig(
                learning_rate=0.03,
                optimizer="adamw",
                weight_decay=0.01,
                microbatch_size=2,
                beta=1.2,
                damping=0.2,
                pcg_dtype="float64",
                pcg_max_iterations=10,
                pcg_tolerance=1.0e-12,
                pcg_residual_recompute_interval=2,
                require_pcg_convergence=True,
            ),
        )
    raise AssertionError(f"unhandled trainer kind {kind!r}")


def _audit(
    trainer: BTMLETrainer | ProRMPlusTrainer,
) -> phase2_training._FirstOrderMeasurement:
    if isinstance(trainer, BTMLETrainer):
        return phase2_training._bt_first_order_measurement(trainer)
    return phase2_training._prorm_first_order_measurement(trainer)


def _two_update_spec() -> FirstOrderConvergenceSpec:
    return FirstOrderConvergenceSpec(
        gradient_ratio_tolerance=1.0e6,
        min_steps=2,
        max_steps=2,
        check_interval=1,
        consecutive_checks=1,
    )


def _adamw_moment_state(
    trainer: ProRMPlusTrainer,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert isinstance(trainer.optimizer, torch.optim.AdamW)
    state = trainer.optimizer.state[trainer.model.weight]
    assert set(state) >= {"step", "exp_avg", "exp_avg_sq"}
    return state["step"], state["exp_avg"], state["exp_avg_sq"]


@pytest.mark.parametrize("kind", ["bt_mle", "prorm_plus"])
def test_core_trainer_durable_resume_matches_two_uninterrupted_updates(
    kind: str,
    tmp_path: Path,
) -> None:
    batch = _tiny_cpu_batch()
    spec = _two_update_spec()
    objective_name = f"core_resume_{kind}"
    execution_role = "phase2_recovery_r3_primary"
    rng_seed = 20260801

    torch.manual_seed(rng_seed)
    uninterrupted_trainer = _trainer(kind, batch)
    uninterrupted = phase2_training._run_trainer_to_first_order_convergence(
        uninterrupted_trainer,
        audit=lambda: _audit(uninterrupted_trainer),
        spec=spec,
        fixed_snapshot_steps=2,
        objective_name=objective_name,
        execution_role=execution_role,
    )
    uninterrupted_rng_state = torch.get_rng_state().clone()

    store = DurableCheckpointStore(
        tmp_path / kind,
        objective=kind,
        binding={
            "schema_version": "phase2-core-resume-test-binding/v1",
            "trainer": kind,
            "seed": rng_seed,
            "execution_role": execution_role,
        },
    )
    torch.manual_seed(rng_seed)
    interrupted_trainer = _trainer(kind, batch)
    checkpoint_rng_state: torch.Tensor | None = None

    def save_checkpoint(payload: dict[str, object], *, reason: str) -> None:
        nonlocal checkpoint_rng_state
        assert reason == "signal"
        checkpoint_rng_state = torch.get_rng_state().clone()
        controller_state = payload["controller_state"]
        assert isinstance(controller_state, dict)
        assert controller_state["completed_steps"] == 1
        store.save(
            payload,
            completed_steps=1,
            reason="signal",
        )

    with pytest.raises(CheckpointInterruption, match="SIGUSR1"):
        phase2_training._run_trainer_to_first_order_convergence(
            interrupted_trainer,
            audit=lambda: _audit(interrupted_trainer),
            spec=spec,
            fixed_snapshot_steps=2,
            objective_name=objective_name,
            checkpoint_hook=save_checkpoint,
            stop_requested=(
                lambda: "SIGUSR1" if interrupted_trainer.completed_steps == 1 else None
            ),
            execution_role=execution_role,
        )

    assert checkpoint_rng_state is not None
    torch.rand(32)
    polluted_rng_state = torch.get_rng_state().clone()
    assert not torch.equal(polluted_rng_state, checkpoint_rng_state)
    resume_state = store.load()
    assert resume_state is not None

    resumed_trainer = _trainer(kind, batch)
    callback_observations: list[int] = []

    def restore_deferred_rng() -> None:
        assert resumed_trainer.completed_steps == 1
        assert not torch.equal(torch.get_rng_state(), checkpoint_rng_state)
        store.restore_pending_rng_state()
        assert torch.equal(torch.get_rng_state(), checkpoint_rng_state)
        callback_observations.append(resumed_trainer.completed_steps)

    resumed = phase2_training._run_trainer_to_first_order_convergence(
        resumed_trainer,
        audit=lambda: _audit(resumed_trainer),
        spec=spec,
        fixed_snapshot_steps=2,
        objective_name=objective_name,
        resume_state=resume_state,
        after_resume_state_restored=restore_deferred_rng,
        execution_role=execution_role,
    )

    assert callback_observations == [1]
    assert torch.equal(torch.get_rng_state(), uninterrupted_rng_state)
    assert resumed.history == uninterrupted.history
    assert resumed.initial == uninterrupted.initial
    assert resumed.final == uninterrupted.final
    assert resumed.evidence == uninterrupted.evidence
    assert resumed.evidence["checks"] == uninterrupted.evidence["checks"]
    assert (
        resumed.evidence["fixed_step_compute_matched_snapshot"]
        == uninterrupted.evidence["fixed_step_compute_matched_snapshot"]
    )
    assert phase2_training._checkpoint_value_sha256(
        resumed_trainer.state_dict()
    ) == phase2_training._checkpoint_value_sha256(uninterrupted_trainer.state_dict())

    if kind == "prorm_plus":
        assert isinstance(uninterrupted_trainer, ProRMPlusTrainer)
        assert isinstance(resumed_trainer, ProRMPlusTrainer)
        assert uninterrupted_trainer.dual_refreshes == resumed_trainer.dual_refreshes == 2
        assert uninterrupted_trainer.dual_direction is not None
        assert resumed_trainer.dual_direction is not None
        assert torch.equal(
            resumed_trainer.dual_direction,
            uninterrupted_trainer.dual_direction,
        )
        uninterrupted_step, uninterrupted_exp_avg, uninterrupted_exp_avg_sq = _adamw_moment_state(
            uninterrupted_trainer
        )
        resumed_step, resumed_exp_avg, resumed_exp_avg_sq = _adamw_moment_state(resumed_trainer)
        assert int(uninterrupted_step.item()) == int(resumed_step.item()) == 2
        assert torch.count_nonzero(uninterrupted_exp_avg)
        assert torch.count_nonzero(uninterrupted_exp_avg_sq)
        assert torch.equal(resumed_exp_avg, uninterrupted_exp_avg)
        assert torch.equal(resumed_exp_avg_sq, uninterrupted_exp_avg_sq)
