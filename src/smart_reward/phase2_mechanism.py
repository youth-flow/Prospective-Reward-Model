"""Held-out mechanism qualifiers for the Phase-2 ProRM claim.

These controls answer a narrower question than the primary finite-policy
experiment:

1. with label noise removed, does exact-margin ProRM have lower held-out local
   regret than exact-soft-label BT under the same full-tangent geometry?
2. in a full-rank, ridge-free projected tangent where the Moore-Penrose theorem
   is directly evaluable, does R=4 ProRM have lower held-out local regret than
   the corresponding BT head?

The resulting gate may qualify the *mechanistic interpretation* of a primary
result.  It is structurally unable to change primary efficacy status.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import Final

import torch

from .contracts import BT_MLE, PRORM_PLUS
from .linear import resolve_fisher_solve_dtype
from .metrics import empirical_fisher_matrix, local_regret, policy_reward_moment
from .paths import relative_posix_reference
from .phase2_config import (
    PHASE2_CONFIRMATORY_SEEDS,
    phase2_design_identity,
    validate_phase2_config,
)
from .phase2_controls import generate_seeded_orthonormal_projection
from .phase2_rollout import Phase2Design, Phase2PreparedInputs, Phase2RuntimeBackend
from .phase2_sensitivity import (
    PrimarySensitivityBinding,
    _validate_primary_aggregate,
    _verify_relative_file_reference,
)
from .phase2_training import _tensor_sha256
from .repro import atomic_write_json
from .statistics import aggregate_paired_metrics

PHASE2_MECHANISM_SEED_SCHEMA: Final = "phase2-mechanism-qualifiers-seed/v1"
PHASE2_MECHANISM_AGGREGATE_SCHEMA: Final = "phase2-mechanism-qualifiers-aggregate/v1"


def _canonical_sha256(value: object) -> str:
    rendered = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _reject_duplicate_object_pairs(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _digest(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def _finite(value: object, *, name: str, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        qualifier = " finite and non-negative" if nonnegative else " finite"
        raise ValueError(f"{name} must be{qualifier}")
    return result


def _seed(value: object, *, name: str = "seed") -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0 or value > 2**63 - 1:
        raise ValueError(f"{name} must be in [0, 2**63 - 1]")
    return value


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _head(
    value: object,
    *,
    expected_method: str,
    expected_dimension: int,
    name: str,
) -> tuple[tuple[float, ...], str]:
    record = _mapping(value, name=name)
    if record.get("method") != expected_method:
        raise ValueError(f"{name} method identity is invalid")
    raw = record.get("head_weight")
    if (
        isinstance(raw, (str, bytes, bytearray))
        or not isinstance(raw, Sequence)
        or len(raw) != expected_dimension
    ):
        raise ValueError(f"{name} head has the wrong reward dimension")
    head = tuple(
        _finite(item, name=f"{name}.head_weight[{index}]") for index, item in enumerate(raw)
    )
    recorded = _digest(record.get("head_sha256"), name=f"{name}.head_sha256")
    observed = _tensor_sha256(torch.tensor(head, dtype=torch.float32))
    if observed != recorded:
        raise ValueError(f"{name} head SHA256 does not match its values")
    return head, recorded


def _predict(
    reward_features: torch.Tensor,
    head: Sequence[float],
) -> torch.Tensor:
    weight = torch.tensor(
        tuple(float(value) for value in head),
        dtype=reward_features.dtype,
        device=reward_features.device,
    )
    if weight.shape != (reward_features.shape[-1],):
        raise ValueError("mechanism reward head has the wrong dimension")
    return reward_features @ weight


def _full_tangent_regret(
    policy_scores: torch.Tensor,
    reward_features: torch.Tensor,
    targets: torch.Tensor,
    head: Sequence[float],
    *,
    beta: float,
    relative_ridge: float,
    design: Phase2Design,
) -> tuple[float, float]:
    flat = policy_scores.reshape(-1, policy_scores.shape[-1]).to(
        dtype=resolve_fisher_solve_dtype(design.pcg_dtype)
    )
    mean_diagonal = float(flat.square().mean(dim=0).mean().item())
    damping = relative_ridge * mean_diagonal
    if not math.isfinite(damping) or damping <= 0.0:
        raise ValueError("mechanism full-tangent damping is invalid")
    value = local_regret(
        policy_scores,
        _predict(reward_features, head),
        targets,
        damping=damping,
        beta=beta,
        pcg_dtype=design.pcg_dtype,
        pcg_max_iterations=design.pcg_max_iterations,
        pcg_tolerance=design.pcg_tolerance,
    )
    result = float(value.item())
    if not math.isfinite(result) or result < 0.0:
        raise FloatingPointError("mechanism full-tangent local regret is invalid")
    return result, damping


def _ridge_free_regret(
    projected_scores: torch.Tensor,
    reward_features: torch.Tensor,
    targets: torch.Tensor,
    head: Sequence[float],
    *,
    beta: float,
    relative_eigenvalue_tolerance: float,
) -> tuple[float, dict[str, object]]:
    scores = projected_scores.to(dtype=torch.float64)
    target64 = targets.to(dtype=torch.float64)
    predicted = _predict(
        reward_features.to(dtype=torch.float64),
        head,
    )
    fisher = empirical_fisher_matrix(scores)
    eigenvalues, eigenvectors = torch.linalg.eigh(fisher)
    if not bool(torch.isfinite(eigenvalues).all()):
        raise FloatingPointError("low-dimensional held-out Fisher spectrum is non-finite")
    largest = float(eigenvalues[-1].item())
    if largest <= 0.0:
        raise ValueError("low-dimensional held-out Fisher is degenerate")
    tolerance = _finite(
        relative_eigenvalue_tolerance,
        name="relative_eigenvalue_tolerance",
    )
    if tolerance <= 0.0:
        raise ValueError("relative_eigenvalue_tolerance must be positive")
    threshold = tolerance * largest
    retained = eigenvalues > threshold
    rank = int(retained.sum().item())
    dimension = int(projected_scores.shape[-1])
    if rank != dimension:
        raise ValueError(
            "ridge-free mechanism qualifier requires a full-rank held-out Fisher; "
            f"rank={rank}, dimension={dimension}"
        )
    inverse = (eigenvectors * eigenvalues.reciprocal().unsqueeze(0)) @ eigenvectors.mT
    error_moment = policy_reward_moment(
        scores,
        predicted - target64,
        center_candidates=True,
        candidate_dim=1,
    )
    solution = inverse @ error_moment
    residual = fisher @ solution - error_moment
    regret = float((0.5 / beta * torch.dot(error_moment, solution)).item())
    if not math.isfinite(regret) or regret < -1.0e-12:
        raise FloatingPointError("ridge-free mechanism local regret is invalid")
    regret = max(0.0, regret)
    denominator = max(
        float(torch.linalg.vector_norm(error_moment).item()),
        torch.finfo(torch.float64).tiny,
    )
    evidence = {
        "schema_version": "phase2-ridge-free-heldout-geometry/v1",
        "dimension": dimension,
        "numerical_rank": rank,
        "full_rank": True,
        "ridge_enabled": False,
        "ridge_coefficient": 0.0,
        "solver": "torch.linalg.eigh_exact_inverse",
        "relative_eigenvalue_tolerance": tolerance,
        "absolute_eigenvalue_threshold": threshold,
        "smallest_eigenvalue": float(eigenvalues[0].item()),
        "largest_eigenvalue": largest,
        "fisher_sha256": _tensor_sha256(fisher),
        "inverse_sha256": _tensor_sha256(inverse),
        "solve_relative_residual": float(torch.linalg.vector_norm(residual).item()) / denominator,
    }
    return regret, evidence


def _auxiliary_heads_and_projection(
    inputs: Phase2PreparedInputs,
    binding: PrimarySensitivityBinding,
) -> dict[str, object]:
    head_training = _mapping(
        binding.raw_result.get("head_training"),
        name="primary.head_training",
    )
    audit = _mapping(head_training.get("audit"), name="primary.head_training.audit")
    reward_dimension = inputs.train.reward_dimension

    exact_soft = _mapping(
        audit.get("exact_soft_label_bt_control"),
        name="audit.exact_soft_label_bt_control",
    )
    exact_margin = _mapping(
        audit.get("exact_margin_control"),
        name="audit.exact_margin_control",
    )
    low = _mapping(
        audit.get("low_dimensional_control"),
        name="audit.low_dimensional_control",
    )
    exact_soft_head, exact_soft_sha = _head(
        _mapping(exact_soft.get("head"), name="exact_soft.head"),
        expected_method=BT_MLE,
        expected_dimension=reward_dimension,
        name="exact_soft.head",
    )
    exact_margin_head, exact_margin_sha = _head(
        _mapping(exact_margin.get("head"), name="exact_margin.head"),
        expected_method=PRORM_PLUS,
        expected_dimension=reward_dimension,
        name="exact_margin.head",
    )
    low_head, low_head_sha = _head(
        _mapping(low.get("head"), name="low.head"),
        expected_method=PRORM_PLUS,
        expected_dimension=reward_dimension,
        name="low.head",
    )
    low_bt_head = tuple(binding.heads[BT_MLE])
    low_bt_sha = _tensor_sha256(torch.tensor(low_bt_head, dtype=torch.float32))
    low_bt_evidence = _mapping(low.get("bt_head"), name="low.bt_head")
    if (
        low_bt_evidence.get("head_sha256") != low_bt_sha
        or low_bt_evidence.get("retrained") is not False
        or low_bt_evidence.get("reason") != "bt_objective_is_independent_of_policy_tangent_geometry"
    ):
        raise ValueError("low-dimensional BT head is not the frozen primary BT head")

    projection_record = _mapping(low.get("projection"), name="low.projection")
    selected_dimension = projection_record.get("selected_dimension")
    source_dimension = projection_record.get("source_dimension")
    namespace = projection_record.get("namespace")
    if (
        isinstance(selected_dimension, bool)
        or not isinstance(selected_dimension, int)
        or isinstance(source_dimension, bool)
        or not isinstance(source_dimension, int)
        or source_dimension != inputs.train.policy_dimension
        or not isinstance(namespace, str)
        or not namespace
        or projection_record.get("declared_seed") != binding.seed
        or projection_record.get("algorithm") != "gaussian_qr_sign_canonical_v1"
    ):
        raise ValueError("low-dimensional projection identity is invalid")
    projection, effective_seed = generate_seeded_orthonormal_projection(
        source_dimension,
        selected_dimension,
        seed=binding.seed,
        namespace=namespace,
        dtype=torch.float64,
        # The formal training control generated this matrix on the training
        # tensor device.  CPU and CUDA Gaussian/QR kernels are not promised to
        # be byte-identical, so regeneration must use that same device before
        # checking the frozen projection hash.
        device=inputs.train.policy_scores.device,
    )
    projection_sha = _tensor_sha256(projection)
    if projection_sha != projection_record.get(
        "projection_sha256"
    ) or effective_seed != projection_record.get("effective_seed"):
        raise ValueError("regenerated low-dimensional projection differs from training")
    geometry = _mapping(low.get("geometry"), name="low.geometry")
    if (
        geometry.get("ridge_enabled") is not False
        or geometry.get("ridge_coefficient") != 0.0
        or geometry.get("regularization") != "moore_penrose_pseudoinverse"
        or geometry.get("selected_dimension") != selected_dimension
    ):
        raise ValueError("low-dimensional training geometry is not ridge-free")
    tolerance = _finite(
        geometry.get("relative_eigenvalue_tolerance"),
        name="low.geometry.relative_eigenvalue_tolerance",
    )
    return {
        "exact_soft_bt_head": exact_soft_head,
        "exact_soft_bt_head_sha256": exact_soft_sha,
        "exact_margin_prorm_head": exact_margin_head,
        "exact_margin_prorm_head_sha256": exact_margin_sha,
        "low_bt_head": low_bt_head,
        "low_bt_head_sha256": low_bt_sha,
        "low_prorm_head": low_head,
        "low_prorm_head_sha256": low_head_sha,
        "projection": projection.to(device=inputs.heldout.test.policy_scores.device)
        .detach()
        .clone(),
        "projection_sha256": projection_sha,
        "projection_namespace": namespace,
        "projection_effective_seed": effective_seed,
        "selected_dimension": selected_dimension,
        "relative_eigenvalue_tolerance": tolerance,
    }


def run_phase2_mechanism_seed(
    inputs: Phase2PreparedInputs,
    binding: PrimarySensitivityBinding,
    backend: Phase2RuntimeBackend,
    *,
    design: Phase2Design,
    output_json: str | os.PathLike[str],
) -> dict[str, object]:
    """Score fresh held-out targets and publish both mechanism qualifiers."""

    if not isinstance(inputs, Phase2PreparedInputs):
        raise TypeError("inputs must be Phase2PreparedInputs")
    if not isinstance(binding, PrimarySensitivityBinding):
        raise TypeError("binding must be PrimarySensitivityBinding")
    if not isinstance(design, Phase2Design):
        raise TypeError("design must be Phase2Design")
    if (
        inputs.seed != binding.seed
        or inputs.seed not in PHASE2_CONFIRMATORY_SEEDS
        or inputs.source_config_hash != binding.source_config_hash
        or inputs.phase2_config_hash != binding.phase2_design_sha256
        or design.sha256 != binding.runtime_contract_sha256
        or design.stage != "confirmatory"
        or design.formal_eligibility is not True
        or dict(inputs.environment_identity) != dict(binding.environment_identity)
    ):
        raise ValueError("mechanism inputs, primary result, and design identities differ")
    destination = Path(output_json).resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite mechanism artifact: {destination}")

    auxiliaries = _auxiliary_heads_and_projection(inputs, binding)
    split = inputs.heldout.test
    with backend.oracle_session(
        expected_chat_template_sha256=inputs.oracle_chat_template_sha256
    ) as oracle:
        targets = oracle.score_transformed(
            tuple(candidate.prompt for candidate in split.candidates),
            tuple(candidate.response for candidate in split.candidates),
            transform=inputs.oracle_transform,
            batch_size=design.oracle_batch_size,
        )
    if (
        targets.shape != (len(split.candidates),)
        or targets.requires_grad
        or not bool(torch.isfinite(targets).all())
    ):
        raise ValueError("mechanism oracle returned malformed held-out targets")
    target_matrix = (
        targets.to(
            device=split.policy_scores.device,
            dtype=split.policy_scores.dtype,
        )
        .reshape(split.num_prompts, split.num_candidates)
        .detach()
        .clone()
    )
    exact_bt, full_damping = _full_tangent_regret(
        split.policy_scores,
        split.reward_features,
        target_matrix,
        auxiliaries["exact_soft_bt_head"],
        beta=binding.beta0,
        relative_ridge=binding.relative_ridge0,
        design=design,
    )
    exact_prorm, second_damping = _full_tangent_regret(
        split.policy_scores,
        split.reward_features,
        target_matrix,
        auxiliaries["exact_margin_prorm_head"],
        beta=binding.beta0,
        relative_ridge=binding.relative_ridge0,
        design=design,
    )
    if second_damping != full_damping:
        raise RuntimeError("exact mechanism learners used different held-out geometry")

    projection = auxiliaries["projection"]
    projected_scores = (split.policy_scores.to(dtype=torch.float64) @ projection).to(
        dtype=split.policy_scores.dtype
    )
    low_bt, low_geometry = _ridge_free_regret(
        projected_scores,
        split.reward_features,
        target_matrix,
        auxiliaries["low_bt_head"],
        beta=binding.beta0,
        relative_eigenvalue_tolerance=auxiliaries["relative_eigenvalue_tolerance"],
    )
    low_prorm, low_geometry_second = _ridge_free_regret(
        projected_scores,
        split.reward_features,
        target_matrix,
        auxiliaries["low_prorm_head"],
        beta=binding.beta0,
        relative_eigenvalue_tolerance=auxiliaries["relative_eigenvalue_tolerance"],
    )
    if low_geometry_second != low_geometry:
        raise RuntimeError("low-dimensional mechanism learners used different geometry")

    payload: dict[str, object] = {
        "schema_version": PHASE2_MECHANISM_SEED_SCHEMA,
        "seed": binding.seed,
        "source_config_hash": binding.source_config_hash,
        "phase2_design_sha256": binding.phase2_design_sha256,
        "phase2_runtime_contract_sha256": binding.runtime_contract_sha256,
        "environment_identity": dict(binding.environment_identity),
        "environment_identity_sha256": binding.environment_identity_sha256,
        "primary_result": {
            "path": relative_posix_reference(binding.path, base=destination.parent),
            "sha256": binding.result_sha256,
            "heads_sha256": binding.heads_sha256,
            "label_stream_sha256": binding.label_stream_sha256,
            "beta0": binding.beta0,
            "read_only": True,
            "modified": False,
        },
        "heldout_test": {
            "deferred_input_sha256": split.identity_sha256,
            "transformed_target_sha256": _tensor_sha256(target_matrix),
            "num_prompts": split.num_prompts,
            "num_candidates": split.num_candidates,
            "raw_oracle_logits_serialized": False,
            "target_vector_serialized": False,
        },
        "exact_noise_free": {
            "schema_version": "phase2-exact-noise-free-mechanism/v1",
            "comparison": "exact_margin_prorm_plus_vs_exact_soft_label_bt",
            "geometry": "full_tangent_primary_ridge",
            "beta": binding.beta0,
            "relative_ridge": binding.relative_ridge0,
            "absolute_ridge": full_damping,
            "learners": {
                BT_MLE: {
                    "head_source": "exact_soft_label_bt_control",
                    "head_sha256": auxiliaries["exact_soft_bt_head_sha256"],
                    "heldout_local_regret": exact_bt,
                },
                PRORM_PLUS: {
                    "head_source": "exact_margin_control",
                    "head_sha256": auxiliaries["exact_margin_prorm_head_sha256"],
                    "heldout_local_regret": exact_prorm,
                },
            },
            "label_noise_present": False,
            "eligible_for_primary_claim": False,
        },
        "low_dimensional_ridge_free": {
            "schema_version": "phase2-low-dimensional-mechanism/v1",
            "comparison": "r4_prorm_plus_vs_r4_bt_mle",
            "geometry": "seeded_projected_tangent_full_rank_ridge_free",
            "beta": binding.beta0,
            "projection": {
                "sha256": auxiliaries["projection_sha256"],
                "namespace": auxiliaries["projection_namespace"],
                "effective_seed": auxiliaries["projection_effective_seed"],
                "selected_dimension": auxiliaries["selected_dimension"],
                "projection_matrix_serialized": False,
            },
            "fisher": low_geometry,
            "learners": {
                BT_MLE: {
                    "head_source": "primary_r4_bt_mle",
                    "head_sha256": auxiliaries["low_bt_head_sha256"],
                    "heldout_local_regret": low_bt,
                },
                PRORM_PLUS: {
                    "head_source": "low_dimensional_r4_prorm_plus_control",
                    "head_sha256": auxiliaries["low_prorm_head_sha256"],
                    "heldout_local_regret": low_prorm,
                },
            },
            "same_r4_label_stream": True,
            "ridge_enabled": False,
            "full_rank_gate_passed": True,
            "eligible_for_primary_claim": False,
        },
        "claim_contract": {
            "role": "mechanism_scope_qualifier_only",
            "may_support_misspecification_geometry_interpretation": True,
            "eligible_to_modify_primary_efficacy_status": False,
            "primary_efficacy_status_read": False,
            "primary_efficacy_status_modified": False,
        },
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(destination, payload, overwrite=False)
    return payload


def _read_strict_json(path: str | os.PathLike[str]) -> tuple[dict[str, object], str]:
    source = Path(path).resolve()
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"mechanism JSON must be a regular non-symlink file: {source}")
    raw = source.read_bytes()
    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token {token}")
            ),
            object_pairs_hook=_reject_duplicate_object_pairs,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid mechanism JSON: {source}") from error
    if not isinstance(value, dict):
        raise TypeError("mechanism JSON must encode an object")
    json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)
    return value, hashlib.sha256(raw).hexdigest()


def _validate_mechanism_seed(
    path: str | os.PathLike[str],
    *,
    expected_source_hash: str,
    expected_design_sha: str,
    expected_runtime_sha: str,
    expected_beta: float,
    expected_relative_ridge: float,
    reference_base: Path,
) -> dict[str, object]:
    value, file_sha = _read_strict_json(path)
    source = Path(path).resolve()
    required = {
        "schema_version",
        "seed",
        "source_config_hash",
        "phase2_design_sha256",
        "phase2_runtime_contract_sha256",
        "environment_identity",
        "environment_identity_sha256",
        "primary_result",
        "heldout_test",
        "exact_noise_free",
        "low_dimensional_ridge_free",
        "claim_contract",
        "artifact_sha256",
    }
    if set(value) != required:
        raise ValueError(f"{source} mechanism fields do not match the strict schema")
    artifact_sha = _digest(
        value.pop("artifact_sha256"),
        name=f"{source}:artifact_sha256",
    )
    if _canonical_sha256(value) != artifact_sha:
        raise ValueError(f"{source} mechanism artifact SHA256 mismatch")
    value["artifact_sha256"] = artifact_sha
    if (
        value["schema_version"] != PHASE2_MECHANISM_SEED_SCHEMA
        or value["source_config_hash"] != expected_source_hash
        or value["phase2_design_sha256"] != expected_design_sha
        or value["phase2_runtime_contract_sha256"] != expected_runtime_sha
    ):
        raise ValueError(f"{source} mechanism artifact identity is invalid")
    environment = _mapping(value["environment_identity"], name=f"{source}:environment")
    environment_sha = _digest(
        value["environment_identity_sha256"],
        name=f"{source}:environment_identity_sha256",
    )
    if _canonical_sha256(environment) != environment_sha:
        raise ValueError(f"{source} environment identity hash mismatch")
    primary = _mapping(value["primary_result"], name=f"{source}:primary_result")
    if (
        primary.get("read_only") is not True
        or primary.get("modified") is not False
        or _finite(primary.get("beta0"), name=f"{source}:beta0") != expected_beta
    ):
        raise ValueError(f"{source} mechanism artifact could mutate primary evidence")
    primary_sha = _digest(
        primary.get("sha256"),
        name=f"{source}:primary_result.sha256",
    )
    _digest(
        primary.get("heads_sha256"),
        name=f"{source}:primary_result.heads_sha256",
    )
    _digest(
        primary.get("label_stream_sha256"),
        name=f"{source}:primary_result.label_stream_sha256",
    )
    _verify_relative_file_reference(
        primary.get("path"),
        source=source,
        expected_sha256=primary_sha,
        name=f"{source}:primary_result.path",
    )
    heldout = _mapping(value["heldout_test"], name=f"{source}:heldout_test")
    _digest(
        heldout.get("deferred_input_sha256"),
        name=f"{source}:heldout_test.deferred_input_sha256",
    )
    _digest(
        heldout.get("transformed_target_sha256"),
        name=f"{source}:heldout_test.transformed_target_sha256",
    )
    _positive_integer(
        heldout.get("num_prompts"),
        name=f"{source}:heldout_test.num_prompts",
    )
    _positive_integer(
        heldout.get("num_candidates"),
        name=f"{source}:heldout_test.num_candidates",
    )
    if (
        heldout.get("raw_oracle_logits_serialized") is not False
        or heldout.get("target_vector_serialized") is not False
    ):
        raise ValueError(f"{source} mechanism artifact leaked held-out oracle values")
    claim = _mapping(value["claim_contract"], name=f"{source}:claim_contract")
    if dict(claim) != {
        "role": "mechanism_scope_qualifier_only",
        "may_support_misspecification_geometry_interpretation": True,
        "eligible_to_modify_primary_efficacy_status": False,
        "primary_efficacy_status_read": False,
        "primary_efficacy_status_modified": False,
    }:
        raise ValueError(f"{source} mechanism claim boundary is invalid")

    metrics: dict[str, dict[str, float]] = {}
    for key, schema in (
        ("exact_noise_free", "phase2-exact-noise-free-mechanism/v1"),
        ("low_dimensional_ridge_free", "phase2-low-dimensional-mechanism/v1"),
    ):
        control = _mapping(value[key], name=f"{source}:{key}")
        if (
            control.get("schema_version") != schema
            or control.get("eligible_for_primary_claim") is not False
            or _finite(
                control.get("beta"),
                name=f"{source}:{key}.beta",
            )
            != expected_beta
        ):
            raise ValueError(f"{source} {key} control identity is invalid")
        if key == "exact_noise_free":
            if (
                control.get("comparison") != "exact_margin_prorm_plus_vs_exact_soft_label_bt"
                or control.get("geometry") != "full_tangent_primary_ridge"
                or _finite(
                    control.get("relative_ridge"),
                    name=f"{source}:{key}.relative_ridge",
                )
                != expected_relative_ridge
                or _finite(
                    control.get("absolute_ridge"),
                    name=f"{source}:{key}.absolute_ridge",
                )
                <= 0.0
                or control.get("label_noise_present") is not False
            ):
                raise ValueError(f"{source} exact-noise-free control identity is invalid")
            expected_sources = {
                BT_MLE: "exact_soft_label_bt_control",
                PRORM_PLUS: "exact_margin_control",
            }
        else:
            if (
                control.get("comparison") != "r4_prorm_plus_vs_r4_bt_mle"
                or control.get("geometry") != "seeded_projected_tangent_full_rank_ridge_free"
                or control.get("same_r4_label_stream") is not True
                or control.get("ridge_enabled") is not False
                or control.get("full_rank_gate_passed") is not True
            ):
                raise ValueError(f"{source} low-dimensional control is not full-rank/ridge-free")
            projection = _mapping(
                control.get("projection"),
                name=f"{source}:{key}.projection",
            )
            _digest(
                projection.get("sha256"),
                name=f"{source}:{key}.projection.sha256",
            )
            namespace = projection.get("namespace")
            _seed(
                projection.get("effective_seed"),
                name=f"{source}:{key}.projection.effective_seed",
            )
            selected_dimension = _positive_integer(
                projection.get("selected_dimension"),
                name=f"{source}:{key}.projection.selected_dimension",
            )
            if (
                not isinstance(namespace, str)
                or not namespace
                or projection.get("projection_matrix_serialized") is not False
            ):
                raise ValueError(f"{source} low-dimensional projection identity is invalid")
            fisher = _mapping(
                control.get("fisher"),
                name=f"{source}:{key}.fisher",
            )
            dimension = _positive_integer(
                fisher.get("dimension"),
                name=f"{source}:{key}.fisher.dimension",
            )
            if (
                fisher.get("schema_version") != "phase2-ridge-free-heldout-geometry/v1"
                or dimension != selected_dimension
                or fisher.get("numerical_rank") != dimension
                or fisher.get("full_rank") is not True
                or fisher.get("ridge_enabled") is not False
                or fisher.get("ridge_coefficient") != 0.0
            ):
                raise ValueError(f"{source} low-dimensional held-out Fisher identity is invalid")
            expected_sources = {
                BT_MLE: "primary_r4_bt_mle",
                PRORM_PLUS: "low_dimensional_r4_prorm_plus_control",
            }
        learners = _mapping(control.get("learners"), name=f"{source}:{key}.learners")
        if set(learners) != {BT_MLE, PRORM_PLUS}:
            raise ValueError(f"{source} {key} learners are incomplete")
        metrics[key] = {}
        for learner in (BT_MLE, PRORM_PLUS):
            record = _mapping(
                learners[learner],
                name=f"{source}:{key}.{learner}",
            )
            if record.get("head_source") != expected_sources[learner]:
                raise ValueError(f"{source} {key} learner source identity is invalid")
            _digest(
                record.get("head_sha256"),
                name=f"{source}:{key}.{learner}.head_sha256",
            )
            metrics[key][learner] = _finite(
                record.get("heldout_local_regret"),
                name=f"{source}:{key}.{learner}.heldout_local_regret",
                nonnegative=True,
            )
    return {
        "seed": _seed(value["seed"], name=f"{source}:seed"),
        "source_path": relative_posix_reference(source, base=reference_base),
        "source_sha256": file_sha,
        "artifact_sha256": artifact_sha,
        "primary_result_sha256": primary_sha,
        "environment_identity": dict(environment),
        "metrics": metrics,
    }


def _upper_bound_negative(aggregate: Mapping[str, object], metric: str) -> bool:
    metrics = _mapping(aggregate.get("metrics"), name="aggregate.metrics")
    summary = _mapping(metrics.get(metric), name=f"aggregate.metrics.{metric}")
    interval = _mapping(summary.get("bootstrap_ci"), name="aggregate.bootstrap_ci")
    return _finite(interval.get("upper"), name="aggregate.bootstrap_ci.upper") < 0.0


def build_phase2_mechanism_aggregate(
    overlay_config: Mapping[str, object],
    mechanism_jsons: Sequence[str | os.PathLike[str]],
    *,
    primary_aggregate_json: str | os.PathLike[str],
    reference_base: str | os.PathLike[str],
) -> dict[str, object]:
    """Aggregate both exact-30 qualifiers without touching primary status."""

    validated = validate_phase2_config(overlay_config)
    design_config = _mapping(validated["design"], name="overlay.design")
    seeds = tuple(int(value) for value in validated["run"]["seeds"])
    if (
        design_config.get("stage") != "confirmatory"
        or design_config.get("formal_eligibility") is not True
        or seeds != PHASE2_CONFIRMATORY_SEEDS
    ):
        raise ValueError("mechanism aggregation requires the exact formal 30-seed design")
    if (
        isinstance(mechanism_jsons, (str, bytes, bytearray))
        or not isinstance(mechanism_jsons, Sequence)
        or len(mechanism_jsons) != len(seeds)
    ):
        raise ValueError("mechanism aggregation requires exactly 30 explicit artifacts")
    paths = tuple(str(Path(path).resolve()) for path in mechanism_jsons)
    if len(set(paths)) != len(paths):
        raise ValueError("mechanism artifact paths must be unique")
    design_sha = phase2_design_identity(validated)
    runtime = Phase2Design.from_phase2_config(validated)
    source_hash = str(design_config["source_config_hash"])
    beta = _finite(runtime.frozen_global_beta, name="frozen beta")
    relative_ridge = _finite(
        validated["objective"]["full_tangent"]["ridge"]["relative_coefficient"],
        name="relative ridge",
    )
    if relative_ridge <= 0.0:
        raise ValueError("relative ridge must be positive")
    base = Path(reference_base).resolve()
    primary = _validate_primary_aggregate(
        primary_aggregate_json,
        design_sha256=design_sha,
        runtime_sha256=runtime.sha256,
        source_config_hash=source_hash,
        expected_seeds=seeds,
    )
    loaded: dict[int, dict[str, object]] = {}
    for path in mechanism_jsons:
        item = _validate_mechanism_seed(
            path,
            expected_source_hash=source_hash,
            expected_design_sha=design_sha,
            expected_runtime_sha=runtime.sha256,
            expected_beta=beta,
            expected_relative_ridge=relative_ridge,
            reference_base=base,
        )
        seed = int(item["seed"])
        if seed in loaded:
            raise ValueError(f"duplicate mechanism artifact for seed {seed}")
        loaded[seed] = item
    if set(loaded) != set(seeds):
        raise ValueError(
            "mechanism artifacts must exactly match the formal seed set; "
            f"missing={sorted(set(seeds) - set(loaded))!r}, "
            f"unexpected={sorted(set(loaded) - set(seeds))!r}"
        )
    for seed in seeds:
        if loaded[seed]["primary_result_sha256"] != primary["result_sha_by_seed"][seed]:
            raise ValueError(
                f"mechanism artifact for seed {seed} is not bound to its primary result"
            )
    environment = loaded[seeds[0]]["environment_identity"]
    if any(loaded[seed]["environment_identity"] != environment for seed in seeds[1:]):
        raise ValueError("mechanism artifacts do not share the formal environment identity")
    if environment != primary["environment_identity"]:
        raise ValueError("mechanism environment identity differs from the primary aggregate")
    bootstrap_seed = int(validated["evaluation"]["paired_bootstrap_seed"])
    bootstrap_resamples = int(validated["evaluation"]["paired_bootstrap_resamples"])
    aggregates: dict[str, object] = {}
    for qualifier in ("exact_noise_free", "low_dimensional_ridge_free"):
        bt = {
            seed: {"heldout_local_regret": loaded[seed]["metrics"][qualifier][BT_MLE]}
            for seed in seeds
        }
        prorm = {
            seed: {"heldout_local_regret": loaded[seed]["metrics"][qualifier][PRORM_PLUS]}
            for seed in seeds
        }
        aggregates[qualifier] = aggregate_paired_metrics(
            bt,
            prorm,
            directions={"heldout_local_regret": "lower_is_better"},
            bootstrap_seed=bootstrap_seed,
            num_resamples=bootstrap_resamples,
        ).to_dict()
    exact_passed = _upper_bound_negative(
        _mapping(aggregates["exact_noise_free"], name="exact aggregate"),
        "heldout_local_regret",
    )
    low_passed = _upper_bound_negative(
        _mapping(
            aggregates["low_dimensional_ridge_free"],
            name="low aggregate",
        ),
        "heldout_local_regret",
    )
    if exact_passed and low_passed:
        scope_status = "mechanism_qualified"
    else:
        scope_status = "finite_procedure_only_no_geometry_attribution"
    payload: dict[str, object] = {
        "schema_version": PHASE2_MECHANISM_AGGREGATE_SCHEMA,
        "source_config_hash": source_hash,
        "phase2_design_sha256": design_sha,
        "phase2_runtime_contract_sha256": runtime.sha256,
        "seeds": list(seeds),
        "num_seeds": len(seeds),
        "experimental_unit": "seed",
        "environment_identity": dict(environment),
        "primary_aggregate": {
            "path": relative_posix_reference(primary["path"], base=base),
            "sha256": primary["sha256"],
            "efficacy_status": primary["evidence_status"],
            "supports_pre_registered_claim": primary["supports_pre_registered_claim"],
            "read_only": True,
            "modified": False,
        },
        "qualifiers": {
            "exact_noise_free": {
                "paired_prorm_plus_minus_bt": aggregates["exact_noise_free"],
                "ci_upper_below_zero": exact_passed,
            },
            "low_dimensional_ridge_free": {
                "paired_prorm_plus_minus_bt": aggregates["low_dimensional_ridge_free"],
                "ci_upper_below_zero": low_passed,
            },
        },
        "claim_scope": {
            "status": scope_status,
            "both_pre_registered_qualifiers_passed": exact_passed and low_passed,
            "may_attribute_primary_advantage_to_misspecification_geometry": (
                exact_passed and low_passed
            ),
            "primary_efficacy_status_copied_without_recomputation": True,
            "eligible_to_modify_primary_efficacy_status": False,
            "if_not_qualified": (
                "restrict_claim_to_finite_procedure_improvement_or_primary_negative_result"
            ),
        },
        "integrity": {
            "exact_30_seed_set": True,
            "all_seed_artifacts_used": True,
            "prompt_or_candidate_pseudoreplication": False,
            "missing_or_selected_seed_subset_allowed": False,
            "both_qualifiers_required_for_geometry_attribution": True,
        },
        "sources": [
            {
                "seed": seed,
                "path": loaded[seed]["source_path"],
                "file_sha256": loaded[seed]["source_sha256"],
                "artifact_sha256": loaded[seed]["artifact_sha256"],
                "primary_result_sha256": loaded[seed]["primary_result_sha256"],
            }
            for seed in seeds
        ],
    }
    payload["aggregate_sha256"] = _canonical_sha256(payload)
    return payload


def write_phase2_mechanism_aggregate(
    overlay_config: Mapping[str, object],
    mechanism_jsons: Sequence[str | os.PathLike[str]],
    output_json: str | os.PathLike[str],
    *,
    primary_aggregate_json: str | os.PathLike[str],
) -> dict[str, object]:
    destination = Path(output_json).resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite mechanism aggregate: {destination}")
    payload = build_phase2_mechanism_aggregate(
        overlay_config,
        mechanism_jsons,
        primary_aggregate_json=primary_aggregate_json,
        reference_base=destination.parent,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(destination, payload, overwrite=False)
    return payload


__all__ = [
    "PHASE2_MECHANISM_AGGREGATE_SCHEMA",
    "PHASE2_MECHANISM_SEED_SCHEMA",
    "build_phase2_mechanism_aggregate",
    "run_phase2_mechanism_seed",
    "write_phase2_mechanism_aggregate",
]
