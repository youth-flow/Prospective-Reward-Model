import json
from pathlib import Path

import pytest
import torch

from smart_reward.direct_preference import (
    _initialize_plateau_baseline,
    _plateau_converged,
    auxdpo_global_regularizer,
    auxdpo_loss,
    candidate_policy_metrics,
    centered,
    extension_hash,
    import_reference_logps,
    load_direct_preference_config,
    pair_indices,
    resolve_source_config,
    soft_preference_loss,
)
from smart_reward.runtime import sha256_file

ROOT = Path(__file__).resolve().parents[1]
EXTENSION = ROOT / "configs" / "dpo_auxdpo_main.yaml"
SMOKE_EXTENSION = ROOT / "configs" / "dpo_auxdpo_smoke.yaml"
CONVERGED_EXTENSION = ROOT / "configs" / "dpo_auxdpo_converged.yaml"
CONVERGED_V2_EXTENSION = ROOT / "configs" / "dpo_auxdpo_converged_v2.yaml"
CONVERGED_V3_EXTENSION = ROOT / "configs" / "dpo_auxdpo_converged_v3.yaml"
CONVERGED_V4_EXTENSION = ROOT / "configs" / "dpo_auxdpo_converged_v4.yaml"
CONVERGED_V5_EXTENSION = ROOT / "configs" / "dpo_auxdpo_converged_v5.yaml"
CONVERGED_SMOKE_EXTENSION = ROOT / "configs" / "dpo_auxdpo_converged_smoke.yaml"


def test_formal_direct_preference_config_is_bound_to_source() -> None:
    extension = load_direct_preference_config(EXTENSION)
    source, source_config = resolve_source_config(EXTENSION, extension)
    assert source.name == "fisher_trpo_main.yaml"
    assert source_config["run"]["seeds"] == [20261001, 20261002, 20261003]
    assert len(extension_hash(extension)) == 64


def test_smoke_config_is_a_bounded_subset_of_the_formal_source() -> None:
    extension = load_direct_preference_config(SMOKE_EXTENSION)
    _, source_config = resolve_source_config(SMOKE_EXTENSION, extension)
    assert extension["training"]["limit_prompts_per_split"] == 4
    assert set(extension["experiment"]["seeds"]).issubset(source_config["run"]["seeds"])


def test_converged_config_uses_validation_only_adaptive_stopping() -> None:
    extension = load_direct_preference_config(CONVERGED_EXTENSION)
    _, source_config = resolve_source_config(CONVERGED_EXTENSION, extension)
    training = extension["training"]
    assert extension["experiment"]["betas"] == [0.2]
    assert extension["experiment"]["seeds"] == source_config["run"]["seeds"]
    assert training["validation_selection_metric"] == "policy_implied_soft_btl_nll"
    assert training["test_usage"] == "final_evaluation_only"
    assert training["gradient_accumulation_steps"] == 1
    assert training["min_epochs"] < training["max_epochs"]
    assert training["minimum_lr_reductions"] == 2
    assert training["restore_best_validation_checkpoint"] is True


def test_converged_smoke_changes_only_budget_and_prompt_limit() -> None:
    formal = load_direct_preference_config(CONVERGED_EXTENSION)
    smoke = load_direct_preference_config(CONVERGED_SMOKE_EXTENSION)
    assert smoke["experiment"]["seeds"] == [formal["experiment"]["seeds"][0]]
    assert smoke["experiment"]["betas"] == formal["experiment"]["betas"]
    assert smoke["training"]["prompt_batch_size"] == formal["training"]["prompt_batch_size"]
    assert smoke["training"]["policy_learning_rate"] == formal["training"]["policy_learning_rate"]
    assert smoke["training"]["limit_prompts_per_split"] == 8


def test_memory_safe_converged_config_preserves_science_and_halves_physical_batch() -> None:
    first = load_direct_preference_config(CONVERGED_EXTENSION)
    second = load_direct_preference_config(CONVERGED_V2_EXTENSION)
    assert second["experiment"]["seeds"] == first["experiment"]["seeds"]
    assert second["experiment"]["betas"] == first["experiment"]["betas"]
    assert second["training"]["prompt_batch_size"] == 2
    for key in (
        "policy_learning_rate",
        "max_epochs",
        "min_epochs",
        "validation_min_delta",
        "early_stopping_patience",
        "validation_selection_metric",
        "test_usage",
    ):
        assert second["training"][key] == first["training"][key]


def test_scaled_lr_config_changes_only_batch_dependent_optimizer_rates() -> None:
    second = load_direct_preference_config(CONVERGED_V2_EXTENSION)
    third = load_direct_preference_config(CONVERGED_V3_EXTENSION)
    assert third["experiment"]["seeds"] == second["experiment"]["seeds"]
    assert third["experiment"]["betas"] == second["experiment"]["betas"]
    assert third["training"]["prompt_batch_size"] == second["training"]["prompt_batch_size"]
    assert third["training"]["policy_learning_rate"] == pytest.approx(
        second["training"]["policy_learning_rate"] / 2.0
    )
    assert third["auxdpo"]["auxiliary_learning_rate"] == pytest.approx(
        second["auxdpo"]["auxiliary_learning_rate"] / 2.0
    )
    for key in (
        "max_epochs",
        "min_epochs",
        "validation_min_delta",
        "minimum_validation_improvement",
        "early_stopping_patience",
        "minimum_lr_reductions",
        "validation_selection_metric",
        "test_usage",
    ):
        assert third["training"][key] == second["training"][key]


def test_plateau_convergence_is_not_conflated_with_generalization() -> None:
    config = load_direct_preference_config(CONVERGED_V4_EXTENSION)
    training = config["training"]
    assert not _plateau_converged(
        epochs_completed=4, bad_epochs=5, lr_reductions=1, training=training
    )
    assert _plateau_converged(epochs_completed=6, bad_epochs=5, lr_reductions=2, training=training)


def test_full_gradient_config_accumulates_exactly_one_train_epoch() -> None:
    config = load_direct_preference_config(CONVERGED_V5_EXTENSION)
    training = config["training"]
    assert config["experiment"]["seeds"] == [20261001, 20261002, 20261003]
    assert config["experiment"]["betas"] == [0.2]
    assert training["prompt_batch_size"] == 2
    assert training["gradient_accumulation_steps"] == 1536
    assert training["minimum_training_improvement"] > 0.0
    assert training["test_usage"] == "final_evaluation_only"


def test_reference_cache_import_records_a_byte_identical_provenance_bridge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    policy_a_sha = "b" * 64
    (artifact / "metadata.json").write_text(
        json.dumps({"evidence": {"policy_a_sha256": policy_a_sha}}), encoding="utf-8"
    )
    (artifact / "candidates.jsonl").write_text("candidate\n", encoding="utf-8")
    artifact_sha = "a" * 64
    monkeypatch.setattr(
        "smart_reward.direct_preference.exact_delta_artifact_metadata_sha256",
        lambda *args, **kwargs: artifact_sha,
    )

    source_root = tmp_path / "source-reference"
    source_root.mkdir()
    source_tensor = source_root / "reference_logps.safetensors"
    source_tensor.write_bytes(b"immutable-reference-tensor")
    source_extension = load_direct_preference_config(CONVERGED_V4_EXTENSION)
    source_metadata = {
        "schema": "prorm-direct-preference-reference-logps/v1",
        "status": "complete",
        "seed": 20261001,
        "source_config_sha256": source_extension["source_config_sha256"],
        "extension_config_sha256": extension_hash(source_extension),
        "artifact_metadata_sha256": artifact_sha,
        "artifact_candidates_sha256": sha256_file(artifact / "candidates.jsonl"),
        "lora_a_sha256": policy_a_sha,
        "response_log_probability": "sum over response tokens",
        "compute_dtype": "bfloat16",
        "limit_prompts_per_split": None,
        "tensors_sha256": sha256_file(source_tensor),
        "shapes": {"train": [1, 1], "validation": [1, 1], "test": [1, 1]},
        "producer": {"git_commit": "c" * 40},
    }
    source_metadata_path = source_root / "metadata.json"
    source_metadata_path.write_text(json.dumps(source_metadata), encoding="utf-8")

    target_root = tmp_path / "target-reference"
    imported = import_reference_logps(
        CONVERGED_V4_EXTENSION,
        CONVERGED_V5_EXTENSION,
        artifact,
        source_root,
        target_root,
        seed=20261001,
    )
    target_extension = load_direct_preference_config(CONVERGED_V5_EXTENSION)
    assert imported["extension_config_sha256"] == extension_hash(target_extension)
    assert imported["tensors_sha256"] == sha256_file(source_tensor)
    assert (target_root / "reference_logps.safetensors").read_bytes() == source_tensor.read_bytes()
    assert imported["provenance_bridge"] == {
        "schema": "prorm-reference-cache-import/v1",
        "mode": "byte_identical_copy",
        "source_metadata_sha256": sha256_file(source_metadata_path),
        "source_extension_config_sha256": extension_hash(source_extension),
        "target_extension_config_sha256": extension_hash(target_extension),
        "tensors_sha256": sha256_file(source_tensor),
    }


def test_soft_preference_loss_uses_every_unordered_edge() -> None:
    oracle = torch.tensor([[1.0, 0.0, -1.0]], dtype=torch.float64)
    pairs = pair_indices(3)
    margins = oracle[:, pairs[0]] - oracle[:, pairs[1]]
    targets = torch.sigmoid(margins)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(margins, targets)
    assert torch.equal(soft_preference_loss(oracle, oracle), expected)
    assert pairs.tolist() == [[0, 0, 1], [1, 2, 2]]


def test_auxiliary_delta_enters_reward_but_zero_moment_is_policy_invisible() -> None:
    implicit = torch.zeros((1, 3), dtype=torch.float64, requires_grad=True)
    oracle = torch.tensor([[1.0, -0.5, -0.5]], dtype=torch.float64)
    # Candidate score rows sum to zero.  This delta is orthogonal to the score
    # coordinate, so it changes the preference fit but has zero policy moment.
    scores = torch.tensor([[[-1.0], [0.0], [1.0]]], dtype=torch.float64)
    delta_raw = torch.tensor([[0.4, -0.8, 0.4]], dtype=torch.float64, requires_grad=True)
    loss, diagnostics = auxdpo_loss(
        implicit,
        oracle,
        delta_raw,
        scores,
        nullspace_weight=1.0,
        amplitude_weight=0.01,
        delta_cap=1.0,
    )
    assert diagnostics["delta"].abs().max() > 0
    assert diagnostics["nullspace_moment"].abs().item() == pytest.approx(0.0, abs=1e-15)
    loss.backward()
    assert implicit.grad is not None
    assert delta_raw.grad is not None


def test_global_auxiliary_regularizer_uses_square_of_full_moment() -> None:
    delta_raw = torch.tensor(
        [[0.2, -0.1, -0.1], [-0.3, 0.1, 0.2]], dtype=torch.float64, requires_grad=True
    )
    scores = torch.tensor([[[-1.0], [0.0], [1.0]], [[2.0], [-1.0], [-1.0]]], dtype=torch.float64)
    regularizer, diagnostics = auxdpo_global_regularizer(
        delta_raw,
        scores,
        nullspace_weight=1.0,
        amplitude_weight=0.01,
        delta_cap=1.0,
    )
    delta = centered(torch.tanh(delta_raw))
    centered_scores = scores - scores.mean(dim=1, keepdim=True)
    expected_moment = torch.einsum("bmd,bm->d", centered_scores, delta) / delta.numel()
    expected = expected_moment.square().sum() - 0.01 * delta.square().mean()
    assert torch.allclose(diagnostics["nullspace_moment"], expected_moment)
    assert torch.allclose(regularizer, expected)
    regularizer.backward()
    assert delta_raw.grad is not None


def test_accumulated_preference_and_global_aux_gradients_match_full_objective() -> None:
    oracle = torch.tensor([[0.8, -0.2, -0.6], [-0.5, 0.1, 0.4]], dtype=torch.float64)
    features = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]],
            [[0.5, -0.5], [-0.5, 0.5], [1.0, 0.0]],
        ],
        dtype=torch.float64,
    )
    scores = features.clone()

    full_policy = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    full_delta = torch.zeros((2, 3), dtype=torch.float64, requires_grad=True)
    full_implicit = torch.einsum("bmd,d->bm", features, full_policy)
    full_preference = soft_preference_loss(full_implicit + centered(full_delta), oracle)
    full_regularizer, _ = auxdpo_global_regularizer(
        full_delta,
        scores,
        nullspace_weight=1.0,
        amplitude_weight=0.01,
        delta_cap=1.0,
    )
    (full_preference + full_regularizer).backward()

    accumulated_policy = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    accumulated_delta = torch.zeros((2, 3), dtype=torch.float64, requires_grad=True)
    for index in range(2):
        implicit = torch.einsum("bmd,d->bm", features[index : index + 1], accumulated_policy)
        delta = centered(accumulated_delta[index : index + 1])
        soft_preference_loss(implicit + delta, oracle[index : index + 1]).backward()
    accumulated_regularizer, _ = auxdpo_global_regularizer(
        accumulated_delta,
        scores,
        nullspace_weight=1.0,
        amplitude_weight=0.01,
        delta_cap=1.0,
    )
    (2.0 * accumulated_regularizer).backward()
    accumulated_policy.grad.div_(2.0)
    accumulated_delta.grad.div_(2.0)

    assert torch.allclose(accumulated_policy.grad, full_policy.grad, atol=1.0e-12)
    assert torch.allclose(accumulated_delta.grad, full_delta.grad, atol=1.0e-12)


def test_candidate_policy_metrics_satisfy_gibbs_identities() -> None:
    rewards = torch.tensor([[1.2, 0.1, -0.7], [0.3, 0.2, -0.2]], dtype=torch.float64)
    beta = 0.2
    # The tabular policy has log ratios equal to reward/beta up to a prompt constant.
    metrics = candidate_policy_metrics(rewards / beta, rewards, beta=beta)
    assert metrics["delta_J"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["beta_KL"] == pytest.approx(0.0, abs=1e-12)
    assert max(metrics["identity_residuals"].values()) < 1e-12


def test_centering_removes_only_prompt_constants() -> None:
    values = torch.tensor([[3.0, 4.0, 5.0], [-2.0, 0.0, 2.0]])
    shifted = values + torch.tensor([[91.0], [-17.0]])
    assert torch.allclose(centered(values), centered(shifted))
    assert torch.allclose(centered(values).mean(dim=1), torch.zeros(2))


def test_plateau_scheduler_is_initialized_against_epoch_zero_policy() -> None:
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=0, threshold=1.0e-5, threshold_mode="abs"
    )
    _initialize_plateau_baseline(scheduler, 0.693147)
    scheduler.step(0.694)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.5)
