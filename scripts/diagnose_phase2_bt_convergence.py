#!/usr/bin/env python3
"""Diagnose Phase-2 BT convergence on an existing immutable artifact.

This is a non-evidentiary, train-only diagnostic.  It re-scores only the saved
train candidates, reconstructs the exact named R=4 label stream, reproduces the
configured AdamW convergence controller, and then probes two deterministic
optimization remedies without serializing rewards, labels, or head vectors.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from smart_reward.phase2_config import load_phase2_config_bundle
from smart_reward.phase2_controls import sample_canonical_r4_noisy_arm
from smart_reward.phase2_hf import HuggingFacePhase2Backend
from smart_reward.phase2_inputs import prepare_phase2_inputs
from smart_reward.phase2_rollout import Phase2Design, _score_training_oracle
from smart_reward.phase2_training import (
    OptimizationConvergenceError,
    _bt_config,
    _bt_first_order_measurement,
    _canonical_sha256,
    _generator_for_training,
    _run_trainer_to_first_order_convergence,
    _tensor_sha256,
    _zero_model,
    compile_phase2_training_settings,
)
from smart_reward.repro import atomic_write_json
from smart_reward.training import BTMLETrainer


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("overlay")
    parser.add_argument("artifact")
    parser.add_argument("manifest")
    parser.add_argument("output")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    return parser


def _finite(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise FloatingPointError(f"{name} is not finite")
    return result


def _measurement(
    trainer: BTMLETrainer,
    *,
    initial_gradient: float,
) -> dict[str, object]:
    observed = _bt_first_order_measurement(trainer)
    ratio = observed.gradient_l2_norm / max(initial_gradient, 1.0e-30)
    return {
        "step": trainer.completed_steps,
        "learning_rate": float(trainer.optimizer.param_groups[0]["lr"]),
        "objective": observed.objective,
        "gradient_l2_norm": observed.gradient_l2_norm,
        "gradient_ratio_to_zero_initialization": _finite(
            ratio,
            name="gradient ratio",
        ),
        "head_sha256": _tensor_sha256(trainer.model.weight),
    }


def _adamw_decay_probe(
    trainer: BTMLETrainer,
    *,
    initial_gradient: float,
    tolerance: float,
    check_interval: int,
    required_consecutive: int,
) -> dict[str, object]:
    stages = (
        (3.0e-4, 1000),
        (1.0e-4, 2000),
        (3.0e-5, 2000),
        (1.0e-5, 2000),
    )
    checks: list[dict[str, object]] = []
    consecutive = 0
    selected_step: int | None = None
    for learning_rate, updates in stages:
        for group in trainer.optimizer.param_groups:
            group["lr"] = learning_rate
        stop = trainer.completed_steps + updates
        while trainer.completed_steps < stop:
            trainer.step()
            if trainer.completed_steps % check_interval != 0:
                continue
            check = _measurement(
                trainer,
                initial_gradient=initial_gradient,
            )
            passed = float(check["gradient_ratio_to_zero_initialization"]) <= tolerance
            consecutive = consecutive + 1 if passed else 0
            check["threshold_passed"] = passed
            check["consecutive_threshold_passes"] = consecutive
            checks.append(check)
            if consecutive >= required_consecutive:
                selected_step = trainer.completed_steps
                break
        if selected_step is not None:
            break
    return {
        "schema_version": "phase2-bt-adamw-decay-probe/v1",
        "starts_after_configured_max_steps": True,
        "stages": [
            {"learning_rate": learning_rate, "maximum_updates": updates}
            for learning_rate, updates in stages
        ],
        "check_interval": check_interval,
        "gradient_ratio_tolerance": tolerance,
        "required_consecutive_checks": required_consecutive,
        "converged": selected_step is not None,
        "selected_step": selected_step,
        "checks": checks,
        "final": _measurement(
            trainer,
            initial_gradient=initial_gradient,
        ),
    }


def _lbfgs_probe(
    trainer: BTMLETrainer,
    *,
    initial_gradient: float,
) -> dict[str, object]:
    batch = trainer.batch
    features = (batch.left_features - batch.right_features).detach().to(torch.float64)
    counts = batch.num_annotations.detach().to(torch.float64)
    wins = batch.left_wins.detach().to(torch.float64)
    denominator = counts.sum()
    head = torch.zeros(
        features.shape[1],
        dtype=torch.float64,
        device=features.device,
        requires_grad=True,
    )
    optimizer = torch.optim.LBFGS(
        [head],
        lr=1.0,
        max_iter=1000,
        max_eval=1250,
        tolerance_grad=1.0e-12,
        tolerance_change=1.0e-15,
        history_size=100,
        line_search_fn="strong_wolfe",
    )
    closure_calls = 0

    def objective() -> torch.Tensor:
        margins = features @ head
        return (counts * torch.nn.functional.softplus(margins) - wins * margins).sum() / (
            denominator
        )

    def closure() -> torch.Tensor:
        nonlocal closure_calls
        closure_calls += 1
        optimizer.zero_grad(set_to_none=True)
        loss = objective()
        loss.backward()
        return loss

    started = time.monotonic()
    optimizer.step(closure)
    elapsed = time.monotonic() - started
    final_objective = objective()
    gradient = torch.autograd.grad(final_objective, head)[0]
    gradient_norm = float(torch.linalg.vector_norm(gradient).item())
    ratio = gradient_norm / max(initial_gradient, 1.0e-30)
    if not bool(torch.isfinite(head).all()) or not math.isfinite(ratio):
        raise FloatingPointError("L-BFGS probe produced a non-finite result")
    return {
        "schema_version": "phase2-bt-lbfgs-probe/v1",
        "initialization": "exact_zero_head",
        "dtype": "float64",
        "full_batch": True,
        "line_search": "strong_wolfe",
        "maximum_iterations": 1000,
        "maximum_evaluations": 1250,
        "history_size": 100,
        "closure_calls": closure_calls,
        "elapsed_seconds": elapsed,
        "final_objective": float(final_objective.item()),
        "final_gradient_l2_norm": gradient_norm,
        "gradient_ratio_to_zero_initialization": ratio,
        "configured_gate_passed": ratio <= 1.0e-3,
        "head_sha256": _tensor_sha256(head.detach()),
        "head_vector_serialized": False,
    }


def main() -> int:
    arguments = _parser().parse_args()
    destination = Path(arguments.output)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite diagnostic output: {destination}")

    started = time.monotonic()
    bundle = load_phase2_config_bundle(arguments.overlay)
    settings = compile_phase2_training_settings(bundle)
    if arguments.seed not in settings.seeds:
        raise ValueError("seed is not declared by the Phase-2 overlay")
    inputs = prepare_phase2_inputs(
        arguments.overlay,
        seed=arguments.seed,
        artifact_dir=arguments.artifact,
        run_manifest=arguments.manifest,
        training_device=arguments.device,
        require_formal=False,
        match_current_environment=False,
    )
    backend = HuggingFacePhase2Backend(
        bundle.base_config,
        device=arguments.device,
        local_files_only=True,
    )
    rewards = _score_training_oracle(
        inputs,
        backend,
        design=Phase2Design.from_phase2_config(bundle.config),
    )
    oracle_elapsed = time.monotonic() - started
    probabilities = torch.sigmoid(rewards[:, 0] - rewards[:, 1])
    generator, derived_seed, derivation_sha256 = _generator_for_training(
        inputs.train,
        base_seed=arguments.seed,
        namespace=settings.label_rng_namespace,
    )
    initial_generator_sha256 = _tensor_sha256(generator.get_state())
    noisy_arm = sample_canonical_r4_noisy_arm(
        inputs.train,
        probabilities,
        generator=generator,
        max_total_annotations=settings.max_total_annotations,
    )
    final_generator_sha256 = _tensor_sha256(generator.get_state())
    batch = noisy_arm.training.to_training_batch()
    model = _zero_model(noisy_arm.training)
    trainer = BTMLETrainer(model, batch, _bt_config(settings))
    current_failure: dict[str, object] | None = None
    current_success: dict[str, object] | None = None
    try:
        run = _run_trainer_to_first_order_convergence(
            trainer,
            audit=lambda: _bt_first_order_measurement(trainer),
            spec=settings.convergence,
            fixed_snapshot_steps=settings.outer_steps,
            objective_name="bt_mle",
            rank_diagnostic=None,
        )
    except OptimizationConvergenceError as error:
        current_failure = dict(error.evidence)
    else:
        current_success = dict(run.evidence)

    evidence = current_failure if current_failure is not None else current_success
    if evidence is None:
        raise RuntimeError("configured convergence probe produced no evidence")
    initial_gradient = float(
        evidence["initial_zero_head_measurement"]["gradient_l2_norm"]  # type: ignore[index]
    )
    decay = (
        _adamw_decay_probe(
            trainer,
            initial_gradient=initial_gradient,
            tolerance=settings.convergence.gradient_ratio_tolerance,
            check_interval=settings.convergence.check_interval,
            required_consecutive=settings.convergence.consecutive_checks,
        )
        if current_failure is not None
        else None
    )
    lbfgs = _lbfgs_probe(trainer, initial_gradient=initial_gradient)
    label_identity = {
        "namespace": settings.label_rng_namespace,
        "base_seed": arguments.seed,
        "derived_seed": derived_seed,
        "derivation_sha256": derivation_sha256,
        "initial_generator_state_sha256": initial_generator_sha256,
        "final_generator_state_sha256": final_generator_sha256,
        "mean_h_sha256": _tensor_sha256(noisy_arm.training.h),
        "replicate_count_sha256": _tensor_sha256(noisy_arm.repeated_labels.counts),
        "replicate_win_sha256": _tensor_sha256(noisy_arm.repeated_labels.wins),
        "replicate_h_sha256": _tensor_sha256(noisy_arm.repeated_labels.replicate_h),
        "realized_total_annotations": noisy_arm.repeated_labels.total_annotations,
    }
    payload = {
        "schema_version": "phase2-bt-convergence-diagnostic/v1",
        "evidence_role": "nonconfirmatory_train_only_optimizer_diagnostic",
        "phase2_design_sha256": bundle.design_identity,
        "source_config_hash": bundle.config["design"]["source_config_hash"],
        "seed": arguments.seed,
        "artifact_metadata_sha256": inputs.artifact_metadata_sha256,
        "run_manifest_sha256": inputs.run_manifest_sha256,
        "training_settings_sha256": settings.sha256,
        "input_training_sha256": _canonical_sha256(
            {
                "policy_scores_sha256": _tensor_sha256(inputs.train.policy_scores),
                "reward_features_sha256": _tensor_sha256(inputs.train.reward_features),
                "prompt_ids": list(inputs.train.prompt_ids),
            }
        ),
        "label_stream_identity": label_identity,
        "oracle_rescore": {
            "transformed_rewards_sha256": _tensor_sha256(rewards),
            "elapsed_seconds": oracle_elapsed,
            "raw_values_serialized": False,
        },
        "configured_adamw": {
            "converged": current_success is not None,
            "failure": current_failure,
            "success": current_success,
        },
        "adamw_decay_probe": decay,
        "lbfgs_probe": lbfgs,
        "information_boundary": {
            "train_only": True,
            "validation_or_test_targets_accessed": False,
            "raw_oracle_values_serialized": False,
            "raw_labels_serialized": False,
            "head_vectors_serialized": False,
            "eligible_for_primary_claim": False,
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(destination, payload, overwrite=False)
    print(
        json.dumps(
            {
                "configured_adamw_converged": current_success is not None,
                "adamw_decay_converged": (None if decay is None else decay["converged"]),
                "lbfgs_gate_passed": lbfgs["configured_gate_passed"],
                "output": str(destination),
                "seed": arguments.seed,
                "status": "ok",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
