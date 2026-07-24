"""Command-line control plane for Prospective Reward Modeling experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict
from numbers import Real
from pathlib import Path
from types import ModuleType

from .config import ConfigError, config_hash, load_config
from .contracts import (
    BT_MLE,
    CANONICAL_LEARNERS,
    CONTROLLED_COMPARISON_SCHEMA_V1,
    CONTROLLED_COMPARISON_SCHEMA_V2,
    LEGACY_SRM_PLUS,
    MATCHED_KL_ROLLOUT_SCHEMA_V1,
    MATCHED_KL_ROLLOUT_SCHEMA_V2,
    PRORM_PLUS,
    compatibility_value,
)
from .data import SchemaError, TrainingEdgeRecord, load_jsonl
from .paths import relative_posix_reference
from .repro import (
    atomic_write_json,
    build_run_manifest,
    collect_execution_identity,
    collect_git_state,
)


def _resolve_run_seed(config: dict[str, object], requested_seed: int | None) -> int:
    run = config["run"]
    if not isinstance(run, dict):  # load_config already guarantees this.
        raise ConfigError("run must be a mapping")
    allowed = [int(run["seed"])] if "seed" in run else [int(seed) for seed in run["seeds"]]
    if requested_seed is None:
        if len(allowed) != 1:
            raise ConfigError("--seed is required when run.seeds contains multiple seeds")
        return allowed[0]
    if requested_seed not in allowed:
        raise ConfigError(f"seed {requested_seed} is not declared by the configuration")
    return requested_seed


def _print_json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True))


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_result_float(value: object, *, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{path} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path} must be finite")
    return result


def _serialized_learners(
    schema_version: object,
    *,
    legacy_schema: str,
    canonical_schema: str,
    path: Path,
) -> tuple[str, str]:
    """Return serialized BT/ProRM+ keys for one supported input schema."""

    if schema_version == canonical_schema:
        return CANONICAL_LEARNERS
    if schema_version == legacy_schema:
        return BT_MLE, LEGACY_SRM_PLUS
    raise ValueError(
        f"{path} has unsupported schema {schema_version!r}; "
        f"expected {legacy_schema!r} or {canonical_schema!r}"
    )


def _read_json_object(path: str | Path) -> dict[str, object]:
    source = Path(path)
    if source.stat().st_size > 64 * 1024 * 1024:
        raise ValueError(f"JSON input exceeds 64 MiB: {source}")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {source}")
            result[key] = value
        return result

    try:
        value = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid UTF-8 JSON: {source}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {source}")
    return value


def _lower_hex_digest(value: object, *, path: str, lengths: set[int]) -> str:
    if (
        not isinstance(value, str)
        or len(value) not in lengths
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{path} must be a lowercase hexadecimal digest")
    return value


def _run_environment_identity(
    manifest_path: str | Path,
    *,
    expected_config_hash: str,
    expected_seed: int,
    require_formal: bool,
    match_current_environment: bool = False,
) -> tuple[str, dict[str, object]]:
    path = Path(manifest_path)
    value = _read_json_object(path)
    if value.get("schema_version") != "smart-reward-run/v1":
        raise ValueError(f"{path} is not a smart-reward-run/v1 manifest")
    if value.get("config_hash") != expected_config_hash:
        raise ValueError(f"{path} config hash does not match the controlled run")
    if value.get("selected_seed") != expected_seed:
        raise ValueError(f"{path} selected_seed does not match the controlled run")

    git = value.get("git")
    slurm = value.get("slurm")
    torch_state = value.get("torch")
    if not all(isinstance(item, dict) for item in (git, slurm, torch_state)):
        raise ValueError(f"{path} has invalid git/slurm/torch evidence")
    commit = _lower_hex_digest(git.get("commit"), path=f"{path}:git.commit", lengths={40, 64})
    dirty = git.get("dirty")
    if not isinstance(dirty, bool):
        raise ValueError(f"{path}:git.dirty must be boolean")

    image = compatibility_value(slurm, "PRORM_IMAGE_SHA256", "SRM_IMAGE_SHA256")
    hf_inventory = compatibility_value(
        slurm,
        "PRORM_HF_INVENTORY_SHA256",
        "SRM_HF_INVENTORY_SHA256",
    )
    environment_commit = compatibility_value(slurm, "PRORM_GIT_COMMIT", "SRM_GIT_COMMIT")
    account = slurm.get("SLURM_JOB_ACCOUNT")
    partition = slurm.get("SLURM_JOB_PARTITION")
    gpus = torch_state.get("gpus")
    gpu_names = (
        [gpu.get("name") for gpu in gpus if isinstance(gpu, dict)] if isinstance(gpus, list) else []
    )
    complete = (
        dirty is False
        and isinstance(image, str)
        and isinstance(hf_inventory, str)
        and isinstance(environment_commit, str)
        and environment_commit == commit
        and isinstance(partition, str)
        and bool(partition)
        and account == "sigroup"
        and torch_state.get("cuda_available") is True
        and torch_state.get("gpu_count") == 1
        and len(gpu_names) == 1
        and isinstance(gpu_names[0], str)
        and bool(gpu_names[0])
    )
    if require_formal and not complete:
        raise ValueError(
            f"{path} lacks a clean, single-GPU Slurm/image/Git/HF-inventory environment identity"
        )
    if complete:
        image = _lower_hex_digest(image, path=f"{path}:slurm.PRORM_IMAGE_SHA256", lengths={64})
        hf_inventory = _lower_hex_digest(
            hf_inventory,
            path=f"{path}:slurm.PRORM_HF_INVENTORY_SHA256",
            lengths={64},
        )
    else:
        commit = None
        image = None
        hf_inventory = None
        account = None
        partition = None
        gpu_names = []
    identity = {
        "formal": complete,
        "git_commit": commit,
        "image_sha256": image,
        "hf_inventory_sha256": hf_inventory,
        "account": account,
        "partition": partition,
        "gpu_models": gpu_names,
    }
    if match_current_environment and collect_execution_identity() != identity:
        raise ValueError(f"{path} environment identity does not match the current process")
    return _sha256_file(path), identity


def _formal_execution_requested() -> bool:
    """Return whether the caller explicitly entered the formal Slurm protocol."""

    return any(
        bool(os.environ.get(name))
        for name in (
            "SLURM_JOB_ID",
            "PRORM_GIT_COMMIT",
            "PRORM_IMAGE_SHA256",
            "PRORM_HF_INVENTORY_SHA256",
            "SRM_GIT_COMMIT",
            "SRM_IMAGE_SHA256",
            "SRM_HF_INVENTORY_SHA256",
        )
    )


def _start_cuda_memory_tracking() -> ModuleType:
    """Reset the single visible GPU's PyTorch peak-memory counters."""

    import importlib

    torch = importlib.import_module("torch")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("PRORM_MEMORY_REPORT requires exactly one visible CUDA GPU")
    # reset_peak_memory_stats does not reliably trigger lazy CUDA
    # initialization itself.  On HPC4's PyTorch/CUDA stack it otherwise
    # raises ``RuntimeError: Invalid device argument`` in a fresh process.
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(0)
    return torch


def _write_cuda_memory_report(
    torch: ModuleType,
    destination: str | Path,
    *,
    command: str,
    status: str,
) -> None:
    """Persist allocator peak evidence after a model-stage CLI invocation."""

    torch.cuda.synchronize(0)
    atomic_write_json(
        destination,
        {
            "schema_version": "cuda-memory-peak/v1",
            "command": command,
            "status": status,
            "device_index": 0,
            "device_name": str(torch.cuda.get_device_name(0)),
            "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
            "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
        },
    )


def _config_check(arguments: argparse.Namespace) -> int:
    config = load_config(arguments.config)
    _print_json(
        {
            "config_hash": config_hash(config),
            "path": str(Path(arguments.config)),
            "status": "ok",
        }
    )
    return 0


def _phase2_config_check(arguments: argparse.Namespace) -> int:
    from .phase2_config import load_phase2_config_bundle

    bundle = load_phase2_config_bundle(arguments.overlay)
    _print_json(
        {
            "design_stage": bundle.config["design"]["stage"],
            "formal_eligibility": bundle.config["design"]["formal_eligibility"],
            "overlay_path": str(bundle.source_path),
            "phase2_design_sha256": bundle.design_identity,
            "source_config_hash": config_hash(bundle.base_config),
            "source_config_path": str(bundle.base_config_path),
            "status": "ok",
        }
    )
    return 0


def _env_report(arguments: argparse.Namespace) -> int:
    config = load_config(arguments.config)
    selected_seed = (
        None
        if arguments.seed is None and isinstance(config["run"].get("seeds"), list)
        else _resolve_run_seed(config, arguments.seed)
    )
    manifest = build_run_manifest(
        config,
        repo_path=arguments.repo_root,
        selected_seed=selected_seed,
    )
    if arguments.output is None:
        _print_json(manifest.to_dict())
    else:
        atomic_write_json(arguments.output, manifest)
        _print_json(
            {
                "config_hash": manifest.config_hash,
                "output": Path(arguments.output).name,
                "status": "ok",
            }
        )
    return 0


def _data_check(arguments: argparse.Namespace) -> int:
    records = load_jsonl(arguments.jsonl, TrainingEdgeRecord)
    if not records:
        raise SchemaError("training JSONL must contain at least one record")
    seen: set[str | int] = set()
    duplicates: list[str | int] = []
    for record in records:
        if record.edge_id in seen:
            duplicates.append(record.edge_id)
        seen.add(record.edge_id)
    if duplicates:
        ordered = sorted({repr(item) for item in duplicates})
        raise SchemaError(f"duplicate edge_id values are forbidden: {ordered!r}")
    _print_json(
        {
            "annotations": sum(record.num_annotations for record in records),
            "edges": len(records),
            "path": str(Path(arguments.jsonl)),
            "prompts": len({record.prompt_id for record in records}),
            "schema_version": records[0].schema_version,
            "status": "ok",
        }
    )
    return 0


def _prepare_prompts(arguments: argparse.Namespace) -> int:
    from .prompts import load_multipref_prompts, save_prompt_jsonl
    from .seeding import SeedBundle

    config = load_config(arguments.config)
    seed = _resolve_run_seed(config, arguments.seed)
    split_seed = SeedBundle.from_base_seed(seed).prompt_split
    run = config["run"]
    data = config["data"]
    records = load_multipref_prompts(
        dataset_name=str(data["prompt_dataset"]),
        revision=str(data["prompt_revision"]),
        split_sizes=run["split_sizes"],
        seed=split_seed,
    )
    save_prompt_jsonl(arguments.output, records)
    _print_json(
        {
            "config_hash": config_hash(config),
            "output": str(Path(arguments.output)),
            "prompts": len(records),
            "seed": seed,
            "prompt_split_seed": split_seed,
            "status": "ok",
        }
    )
    return 0


def _synthetic_check(arguments: argparse.Namespace) -> int:
    from .synthetic import run_synthetic_experiment

    payload = asdict(run_synthetic_experiment(seed=arguments.seed))
    payload["schema_version"] = "prorm-synthetic-benchmark/v1"
    payload["benchmark_only"] = True
    payload["status"] = "ok"
    if arguments.output is None:
        _print_json(payload)
    else:
        destination = Path(arguments.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(destination, payload)
        _print_json({"output": str(destination), "seed": arguments.seed, "status": "ok"})
    return 0


def _closed_form_check(arguments: argparse.Namespace) -> int:
    """Emit the deterministic audited four-response ProRM calculation."""

    from .closed_form import (
        F0,
        THREE_EDGE_DISTRIBUTION,
        TRUE_REWARD,
        apply_a0,
        bt_rm_optimal_w,
        closed_form_results,
        exact_true_regret,
        fisher_from_reference_policy,
        fixed_kl_learned_theta,
        fixed_kl_local_regret,
        local_prorm_regret,
        ordered_iid_pair_distribution,
        pair_fisher,
        pair_reward_moment,
        reward_vector,
    )

    def vectors_close(left: Sequence[float], right: Sequence[float]) -> bool:
        return len(left) == len(right) and all(
            math.isclose(float(a), float(b), rel_tol=1.0e-12, abs_tol=1.0e-12)
            for a, b in zip(left, right, strict=True)
        )

    def matrices_close(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> bool:
        return len(left) == len(right) and all(
            vectors_close(left_row, right_row)
            for left_row, right_row in zip(left, right, strict=True)
        )

    audit_rewards = (
        TRUE_REWARD,
        reward_vector(bt_rm_optimal_w()),
        reward_vector(3.0, 6.0),
        (1.25, -0.75, 2.5, -4.0),
    )
    natural_q0 = ordered_iid_pair_distribution()
    natural_q0_identity = matrices_close(pair_fisher(natural_q0), F0) and all(
        vectors_close(pair_reward_moment(reward, natural_q0), apply_a0(reward))
        for reward in audit_rewards
    )
    three_edge_identity = matrices_close(
        pair_fisher(THREE_EDGE_DISTRIBUTION), fisher_from_reference_policy()
    ) and all(
        vectors_close(
            pair_reward_moment(reward, THREE_EDGE_DISTRIBUTION),
            apply_a0(reward),
        )
        for reward in audit_rewards
    )

    beta_grid = []
    for beta in (4.0, 8.0, 16.0, 32.0, 64.0):
        method_rows: dict[str, dict[str, float]] = {}
        for method, weight in (("BT-RM", bt_rm_optimal_w()), ("ProRM", 3.0)):
            local = local_prorm_regret(weight, beta=beta)
            exact = exact_true_regret(weight, beta=beta)
            method_rows[method] = {
                "local_regret": local,
                "exact_regret": exact,
                "absolute_error": abs(local - exact),
                "relative_error": abs(local - exact) / exact,
            }
        beta_grid.append({"beta": beta, "methods": method_rows})

    fixed_k = 0.01
    bt_weight = bt_rm_optimal_w()
    bt_fixed_k_theta = fixed_kl_learned_theta(bt_weight, kl_budget=fixed_k)
    prorm_fixed_k_theta = fixed_kl_learned_theta(3.0, kl_budget=fixed_k)
    bt_fixed_k_regret = fixed_kl_local_regret(bt_weight, kl_budget=fixed_k)
    prorm_fixed_k_regret = fixed_kl_local_regret(3.0, kl_budget=fixed_k)
    payload = {
        "schema_version": "prorm-closed-form/v1",
        "audited_table": [asdict(row) for row in closed_form_results()],
        "natural_q0_identity": natural_q0_identity,
        "three_edge_identity": three_edge_identity,
        "beta_grid_local_approximation": beta_grid,
        "fixed_k_estimand_audit": {
            "kl_budget": fixed_k,
            "bt_theta": list(bt_fixed_k_theta),
            "prorm_theta": list(prorm_fixed_k_theta),
            "bt_target_regret": bt_fixed_k_regret,
            "prorm_target_regret": prorm_fixed_k_regret,
            "directions_equal": vectors_close(bt_fixed_k_theta, prorm_fixed_k_theta),
            "regrets_equal": math.isclose(
                bt_fixed_k_regret,
                prorm_fixed_k_regret,
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            ),
            "interpretation": (
                "this example separates BT and ProRM under common beta, "
                "not after learner-specific fixed-K normalization"
            ),
        },
        "population_example_only": True,
        "benchmark_only": True,
        "status": "ok",
    }
    if arguments.output is None:
        _print_json(payload)
    else:
        destination = Path(arguments.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(destination, payload)
        _print_json({"output": str(destination), "status": "ok"})
    return 0


def _controlled_materialize(arguments: argparse.Namespace) -> int:
    from .phase1 import materialize_phase1

    config = load_config(arguments.config)
    seed = _resolve_run_seed(config, arguments.seed)
    materialization = materialize_phase1(
        config,
        seed=seed,
        artifact_dir=arguments.artifact_dir,
        device=arguments.device,
        local_files_only=not arguments.allow_download,
    )
    _print_json(
        {
            "artifact_dir": materialization.artifact_directory.name,
            "config_hash": config_hash(config),
            "seed": seed,
            "status": "ok",
        }
    )
    return 0


def _damping_multipliers(config: dict[str, object]) -> tuple[float, ...]:
    objective = config["objective"]
    if not isinstance(objective, dict):
        raise ConfigError("objective must be a mapping")
    raw = objective.get("damping_sensitivity_multipliers", [1.0])
    values = tuple(float(value) for value in raw)
    # The primary run is required by every downstream stage and must be
    # completed before an ill-conditioned sensitivity solve is attempted.
    return (1.0, *(value for value in values if value != 1.0))


def _controlled_compare(arguments: argparse.Namespace) -> int:
    from .artifacts import artifact_metadata_sha256, load_controlled_feature_artifact
    from .experiment import (
        ControlledFeatureExperiment,
        EvaluationTensorData,
        TrainingTensorData,
        compile_feature_experiment_config,
        run_feature_experiment,
    )

    config = load_config(arguments.config)
    seed = _resolve_run_seed(config, arguments.seed)
    digest = config_hash(config)
    destination = Path(arguments.output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {destination}")
    formal_execution = _formal_execution_requested()
    manifest_sha256, environment_identity = _run_environment_identity(
        arguments.run_manifest,
        expected_config_hash=digest,
        expected_seed=seed,
        require_formal=formal_execution,
        # Always match the parsed manifest to this process.  An old formal
        # manifest cannot be replayed from an unrelated local workstation.
        match_current_environment=True,
    )
    experiment = load_controlled_feature_artifact(
        arguments.artifact_dir,
        expected_config_hash=digest,
        expected_seed=seed,
    )
    artifact_identity = artifact_metadata_sha256(
        arguments.artifact_dir,
        expected_config_hash=digest,
        expected_seed=seed,
    )
    artifact_metadata = _read_json_object(Path(arguments.artifact_dir) / "metadata.json")
    artifact_evidence = artifact_metadata.get("evidence")
    artifact_producer = (
        artifact_evidence.get("producer") if isinstance(artifact_evidence, dict) else None
    )
    if not isinstance(artifact_producer, dict):
        raise ValueError("artifact metadata is missing producer identity")
    if environment_identity["formal"] and artifact_producer != {
        "git_commit": environment_identity["git_commit"],
        "image_sha256": environment_identity["image_sha256"],
        "hf_inventory_sha256": environment_identity["hf_inventory_sha256"],
    }:
        raise ValueError("artifact producer does not match the run manifest environment")
    if arguments.device != "cpu":
        train = experiment.train
        validation = experiment.validation
        test = experiment.test
        experiment = ControlledFeatureExperiment(
            train=TrainingTensorData(
                prompt_ids=train.prompt_ids,
                policy_scores=train.policy_scores.to(arguments.device),
                reward_features=train.reward_features.to(arguments.device),
                h=train.h.to(arguments.device),
                left_wins=train.left_wins.to(arguments.device),
                num_annotations=train.num_annotations.to(arguments.device),
            ),
            validation=EvaluationTensorData(
                prompt_ids=validation.prompt_ids,
                policy_scores=validation.policy_scores.to(arguments.device),
                reward_features=validation.reward_features.to(arguments.device),
                true_rewards=validation.true_rewards.to(arguments.device),
            ),
            test=EvaluationTensorData(
                prompt_ids=test.prompt_ids,
                policy_scores=test.policy_scores.to(arguments.device),
                reward_features=test.reward_features.to(arguments.device),
                true_rewards=test.true_rewards.to(arguments.device),
            ),
        )
    damping_runs = []
    for multiplier in _damping_multipliers(config):
        runtime = compile_feature_experiment_config(
            config,
            damping_multiplier=multiplier,
        )
        try:
            result = run_feature_experiment(experiment, runtime).to_dict()
        except RuntimeError as error:
            if multiplier == 1.0 or "PCG did not converge" not in str(error):
                raise
            result = {
                "status": "failed",
                "failure_type": "pcg_nonconvergence",
                "message": str(error),
            }
        damping_runs.append({"damping_multiplier": multiplier, "result": result})
    payload = {
        "schema_version": CONTROLLED_COMPARISON_SCHEMA_V2,
        "config_hash": digest,
        "seed": seed,
        "artifact_dir": relative_posix_reference(
            arguments.artifact_dir,
            base=destination.parent,
        ),
        "artifact_metadata_sha256": artifact_identity,
        "run_manifest": relative_posix_reference(
            arguments.run_manifest,
            base=destination.parent,
        ),
        "run_manifest_sha256": manifest_sha256,
        "environment_identity": environment_identity,
        "damping_runs": damping_runs,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(destination, payload)
    _print_json(
        {
            "config_hash": digest,
            "damping_runs": len(damping_runs),
            "output": destination.name,
            "seed": seed,
            "status": "ok",
        }
    )
    return 0


def _controlled_rollout(arguments: argparse.Namespace) -> int:
    from .phase1_rollout import evaluate_matched_kl_rollouts

    config = load_config(arguments.config)
    seed = _resolve_run_seed(config, arguments.seed)
    payload = evaluate_matched_kl_rollouts(
        config,
        seed=seed,
        artifact_dir=arguments.artifact_dir,
        comparison_json=arguments.comparison,
        output_json=arguments.output,
        device=arguments.device,
        local_files_only=not arguments.allow_download,
    )
    _print_json(
        {
            "config_hash": payload["config_hash"],
            "output": Path(arguments.output).name,
            "seed": seed,
            "status": "ok",
            "updated_rollouts": "updated_rollouts.jsonl",
        }
    )
    return 0


def _phase2_run(arguments: argparse.Namespace) -> int:
    from .phase2_config import load_phase2_config_bundle
    from .phase2_hf import HuggingFacePhase2Backend
    from .phase2_inputs import prepare_phase2_inputs
    from .phase2_pilot_aggregate import (
        verify_beta_source_aggregate,
        verify_horizon_parent_aggregate,
    )
    from .phase2_rollout import Phase2Design, run_common_beta_rollouts
    from .phase2_training import FreshPhase2HeadTrainer

    bundle = load_phase2_config_bundle(arguments.overlay)
    seed = _resolve_run_seed(bundle.config, arguments.seed)
    beta_source_aggregate = getattr(arguments, "beta_source_aggregate", None)
    horizon_parent_aggregate = getattr(arguments, "horizon_parent_aggregate", None)
    verify_beta_source_aggregate(bundle.config, beta_source_aggregate)
    verify_horizon_parent_aggregate(
        bundle.config,
        (
            beta_source_aggregate
            if horizon_parent_aggregate is None
            and bundle.config["design"]["pilot_phase"] != "calibration"
            else horizon_parent_aggregate
        ),
    )
    inputs = prepare_phase2_inputs(
        arguments.overlay,
        seed=seed,
        artifact_dir=arguments.artifact,
        run_manifest=arguments.manifest,
        training_device=arguments.device,
    )
    head_trainer = FreshPhase2HeadTrainer(bundle)
    backend = HuggingFacePhase2Backend(
        bundle.base_config,
        device=arguments.device,
        local_files_only=True,
    )
    design = Phase2Design.from_phase2_config(bundle.config)
    payload = run_common_beta_rollouts(
        inputs,
        head_trainer,
        backend,
        output_json=arguments.output,
        design=design,
    )
    safety = payload.get("measured_kl_safety")
    if not isinstance(safety, dict) or not isinstance(safety.get("passed"), bool):
        raise RuntimeError("Phase-2 runner returned malformed measured-KL safety evidence")
    kl_gate_passed = safety["passed"]
    kl_gate_measure_only = payload["design_stage"] == "pilot"
    if kl_gate_measure_only:
        pilot_gate = payload.get("pilot_kl_safety_gate")
        if (
            not isinstance(pilot_gate, dict)
            or pilot_gate.get("gate_passed") is not kl_gate_passed
            or pilot_gate.get("measure_only") is not True
        ):
            raise RuntimeError("Phase-2 pilot returned inconsistent measure-only KL evidence")
    _print_json(
        {
            "design_stage": bundle.config["design"]["stage"],
            "formal_eligibility": bundle.config["design"]["formal_eligibility"],
            "kl_gate_measure_only": kl_gate_measure_only,
            "kl_gate_passed": kl_gate_passed,
            "output": Path(arguments.output).name,
            "phase2_design_sha256": payload["phase2_design_sha256"],
            "sidecar_jsonl": (
                payload["diagnostics_jsonl"]
                if payload["design_stage"] == "pilot"
                else payload["rollouts_jsonl"]
            ),
            "seed": seed,
            "status": "ok",
        }
    )
    return 0


def _phase2_sensitivity_run(arguments: argparse.Namespace) -> int:
    from .phase2_config import load_phase2_config_bundle
    from .phase2_hf import HuggingFacePhase2Backend
    from .phase2_inputs import prepare_phase2_inputs
    from .phase2_rollout import Phase2Design
    from .phase2_sensitivity import (
        load_primary_sensitivity_binding,
        run_phase2_sensitivity_seed,
    )

    bundle = load_phase2_config_bundle(arguments.overlay)
    seed = _resolve_run_seed(bundle.config, arguments.seed)
    inputs = prepare_phase2_inputs(
        arguments.overlay,
        seed=seed,
        artifact_dir=arguments.artifact,
        run_manifest=arguments.manifest,
        training_device=arguments.device,
    )
    binding = load_primary_sensitivity_binding(
        bundle.config,
        arguments.primary_result,
    )
    backend = HuggingFacePhase2Backend(
        bundle.base_config,
        device=arguments.device,
        local_files_only=True,
    )
    payload = run_phase2_sensitivity_seed(
        inputs,
        binding,
        backend,
        settings=bundle,
        design=Phase2Design.from_phase2_config(bundle.config),
        output_json=arguments.output,
    )
    _print_json(
        {
            "beta_cell_statuses": {
                str(cell["multiplier"]): cell["status"] for cell in payload["beta_cells"]
            },
            "formal_eligibility": False,
            "output": Path(arguments.output).name,
            "output_sha256": _sha256_file(arguments.output),
            "phase2_design_sha256": payload["phase2_design_sha256"],
            "ridge_cells": len(payload["ridge_cells"]),
            "seed": seed,
            "status": "ok",
            "supports_primary_claim": False,
        }
    )
    return 0


def _phase2_mechanism_run(arguments: argparse.Namespace) -> int:
    from .phase2_config import load_phase2_config_bundle
    from .phase2_hf import HuggingFacePhase2Backend
    from .phase2_inputs import prepare_phase2_inputs
    from .phase2_mechanism import run_phase2_mechanism_seed
    from .phase2_rollout import Phase2Design
    from .phase2_sensitivity import load_primary_sensitivity_binding

    bundle = load_phase2_config_bundle(arguments.overlay)
    seed = _resolve_run_seed(bundle.config, arguments.seed)
    inputs = prepare_phase2_inputs(
        arguments.overlay,
        seed=seed,
        artifact_dir=arguments.artifact,
        run_manifest=arguments.manifest,
        training_device=arguments.device,
    )
    binding = load_primary_sensitivity_binding(
        bundle.config,
        arguments.primary_result,
    )
    backend = HuggingFacePhase2Backend(
        bundle.base_config,
        device=arguments.device,
        local_files_only=True,
    )
    payload = run_phase2_mechanism_seed(
        inputs,
        binding,
        backend,
        design=Phase2Design.from_phase2_config(bundle.config),
        output_json=arguments.output,
    )
    _print_json(
        {
            "formal_eligibility": False,
            "output": Path(arguments.output).name,
            "output_sha256": _sha256_file(arguments.output),
            "phase2_design_sha256": payload["phase2_design_sha256"],
            "qualifiers": [
                "exact_noise_free",
                "low_dimensional_ridge_free",
            ],
            "seed": seed,
            "status": "ok",
            "supports_primary_claim": False,
        }
    )
    return 0


def _estimand_audit(arguments: argparse.Namespace) -> int:
    from .estimand_audit import audit_phase1_estimands

    config = load_config(arguments.config)
    seed = _resolve_run_seed(config, arguments.seed)
    payload = audit_phase1_estimands(
        config,
        seed=seed,
        artifact_dir=arguments.artifact_dir,
        comparison_json=arguments.comparison,
        rollout_json=arguments.rollout,
        output_json=arguments.output,
        prompt_chunk_size=arguments.prompt_chunk_size,
    )
    _print_json(
        {
            "config_hash": payload["config_hash"],
            "device": "cpu",
            "output": Path(arguments.output).name,
            "seed": seed,
            "status": "ok",
        }
    )
    return 0


def _optimization_audit(arguments: argparse.Namespace) -> int:
    from .optimization_audit import audit_phase1_head_optimization

    config = load_config(arguments.config)
    seed = _resolve_run_seed(config, arguments.seed)
    payload = audit_phase1_head_optimization(
        config,
        seed=seed,
        artifact_dir=arguments.artifact_dir,
        comparison_json=arguments.comparison,
        output_json=arguments.output,
    )
    _print_json(
        {
            "config_hash": payload["config_hash"],
            "device": "cpu",
            "output": Path(arguments.output).name,
            "seed": seed,
            "status": "ok",
        }
    )
    return 0


def _load_comparison_metrics(
    paths: Sequence[str],
    *,
    expected_config_hash: str,
    expected_damping_multipliers: tuple[float, ...],
    reference_base: str | os.PathLike[str],
) -> tuple[
    dict[int, dict[str, float]],
    dict[int, dict[str, float]],
    dict[int, dict[str, object]],
    dict[float, dict[int, dict[str, object]]],
]:
    bt_by_seed: dict[int, dict[str, float]] = {}
    prorm_plus_by_seed: dict[int, dict[str, float]] = {}
    sources: dict[int, dict[str, object]] = {}
    damping_evidence: dict[float, dict[int, dict[str, object]]] = {
        multiplier: {} for multiplier in expected_damping_multipliers
    }
    for raw_path in paths:
        path = Path(raw_path)
        value = _read_json_object(path)
        if not isinstance(value, dict):
            raise ValueError(f"{path} comparison result must be an object")
        schema_version = value.get("schema_version")
        serialized_learners = _serialized_learners(
            schema_version,
            legacy_schema=CONTROLLED_COMPARISON_SCHEMA_V1,
            canonical_schema=CONTROLLED_COMPARISON_SCHEMA_V2,
            path=path,
        )
        serialized_prorm_plus = serialized_learners[1]
        if value.get("config_hash") != expected_config_hash:
            raise ValueError(f"{path} config_hash does not match the aggregation config")
        artifact_identity = value.get("artifact_metadata_sha256")
        if (
            not isinstance(artifact_identity, str)
            or len(artifact_identity) != 64
            or any(character not in "0123456789abcdef" for character in artifact_identity)
        ):
            raise ValueError(f"{path} has an invalid artifact_metadata_sha256")
        seed = value.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"{path} has an invalid seed")
        if seed in bt_by_seed:
            raise ValueError(f"duplicate comparison result for seed {seed}")
        manifest_name = Path(str(value.get("run_manifest", ""))).name
        if manifest_name != "run-manifest.json":
            raise ValueError(f"{path} records an invalid run manifest filename")
        manifest_path = path.parent / manifest_name
        recorded_manifest_sha = _lower_hex_digest(
            value.get("run_manifest_sha256"),
            path=f"{path}:run_manifest_sha256",
            lengths={64},
        )
        manifest_sha, environment_identity = _run_environment_identity(
            manifest_path,
            expected_config_hash=expected_config_hash,
            expected_seed=seed,
            require_formal=True,
        )
        if manifest_sha != recorded_manifest_sha:
            raise ValueError(f"{path} run-manifest.json SHA256 mismatch")
        if value.get("environment_identity") != environment_identity:
            raise ValueError(f"{path} environment identity does not match its run manifest")
        raw_runs = value.get("damping_runs")
        if not isinstance(raw_runs, list):
            raise ValueError(f"{path} damping_runs must be a list")
        runs: dict[float, dict[str, object]] = {}
        for run_index, run in enumerate(raw_runs):
            if not isinstance(run, dict) or set(run) != {"damping_multiplier", "result"}:
                raise ValueError(f"{path} damping_runs[{run_index}] has an invalid schema")
            multiplier = _finite_result_float(
                run["damping_multiplier"], path=f"{path}:damping_runs[{run_index}]"
            )
            if multiplier in runs:
                raise ValueError(f"{path} repeats damping multiplier {multiplier}")
            if multiplier not in damping_evidence:
                raise ValueError(f"{path} contains undeclared damping multiplier {multiplier}")
            result_value = run["result"]
            if not isinstance(result_value, dict):
                raise ValueError(f"{path} damping result must be an object")
            runs[multiplier] = result_value
        if set(runs) != set(expected_damping_multipliers):
            raise ValueError(f"{path} damping multipliers do not exactly match the config")

        for multiplier, damping_result in runs.items():
            if damping_result.get("status") == "failed":
                if multiplier == 1.0:
                    raise ValueError(f"{path} primary damping run may not be failed")
                damping_evidence[multiplier][seed] = {
                    "status": "failed",
                    "failure_type": str(damping_result.get("failure_type", "unknown")),
                    "message": str(damping_result.get("message", "")),
                }
                continue
            local_regret: dict[str, float] = {}
            for learner_name, serialized_name in zip(
                CANONICAL_LEARNERS, serialized_learners, strict=True
            ):
                learner_value = damping_result.get(serialized_name)
                test_value = learner_value.get("test") if isinstance(learner_value, dict) else None
                if not isinstance(test_value, dict):
                    raise ValueError(
                        f"{path} damping={multiplier} is missing {serialized_name}.test"
                    )
                local_regret[learner_name] = _finite_result_float(
                    test_value.get("local_regret"),
                    path=f"{path}:damping={multiplier}:{serialized_name}.test.local_regret",
                )
            prorm_value = damping_result.get(serialized_prorm_plus)
            final_pcg = prorm_value.get("final_pcg") if isinstance(prorm_value, dict) else None
            pcg_converged = isinstance(final_pcg, dict) and final_pcg.get("converged") is True
            damping_evidence[multiplier][seed] = {
                "status": "ok",
                "bt_local_regret": local_regret[BT_MLE],
                "prorm_plus_local_regret": local_regret[PRORM_PLUS],
                "pcg_converged": pcg_converged,
            }

        result = runs[1.0]
        if not isinstance(result, dict):
            raise ValueError(f"{path} contains an invalid main result")
        learners: dict[str, dict[str, float]] = {}
        for key, serialized_key in zip(CANONICAL_LEARNERS, serialized_learners, strict=True):
            learner = result.get(serialized_key)
            test = learner.get("test") if isinstance(learner, dict) else None
            if not isinstance(test, dict):
                raise ValueError(f"{path} is missing {serialized_key}.test metrics")
            learner_metrics = {
                "test_local_regret": _finite_result_float(
                    test.get("local_regret"),
                    path=f"{path}:{serialized_key}.test.local_regret",
                ),
                "test_squared_fisher_error": _finite_result_float(
                    test.get("squared_fisher_error"),
                    path=f"{path}:{serialized_key}.test.squared_fisher_error",
                ),
                "test_fisher_cosine": _finite_result_float(
                    test.get("fisher_cosine"),
                    path=f"{path}:{serialized_key}.test.fisher_cosine",
                ),
                "test_pairwise_accuracy": _finite_result_float(
                    test.get("pairwise_accuracy"),
                    path=f"{path}:{serialized_key}.test.pairwise_accuracy",
                ),
            }
            descriptive_names = ("oracle_pairwise_nll", "oracle_probability_mae")
            has_descriptive = [name in test for name in descriptive_names]
            if schema_version == CONTROLLED_COMPARISON_SCHEMA_V2 or any(has_descriptive):
                if not all(has_descriptive):
                    raise ValueError(
                        f"{path}:{serialized_key}.test must contain both oracle descriptive metrics"
                    )
                learner_metrics.update(
                    {
                        "test_oracle_pairwise_nll": _finite_result_float(
                            test["oracle_pairwise_nll"],
                            path=(f"{path}:{serialized_key}.test.oracle_pairwise_nll"),
                        ),
                        "test_oracle_probability_mae": _finite_result_float(
                            test["oracle_probability_mae"],
                            path=(f"{path}:{serialized_key}.test.oracle_probability_mae"),
                        ),
                    }
                )
            learners[key] = learner_metrics
        bt_by_seed[seed] = learners[BT_MLE]
        prorm_plus_by_seed[seed] = learners[PRORM_PLUS]
        sources[seed] = {
            "comparison_path": relative_posix_reference(path, base=reference_base),
            "comparison_sha256": _sha256_file(path),
            "artifact_metadata_sha256": artifact_identity,
            "run_manifest_path": relative_posix_reference(
                manifest_path,
                base=reference_base,
            ),
            "run_manifest_sha256": manifest_sha,
            "environment_identity": environment_identity,
        }
    return bt_by_seed, prorm_plus_by_seed, sources, damping_evidence


def _load_rollout_metrics(
    paths: Sequence[str],
    *,
    expected_config_hash: str,
    comparison_sources: dict[int, dict[str, object]],
    expected_kl: float,
    kl_relative_tolerance: float,
    reference_base: str | os.PathLike[str],
) -> tuple[dict[int, float], dict[int, float], dict[int, dict[str, object]]]:
    bt_by_seed: dict[int, float] = {}
    prorm_plus_by_seed: dict[int, float] = {}
    sources: dict[int, dict[str, object]] = {}
    for raw_path in paths:
        path = Path(raw_path)
        value = _read_json_object(path)
        if not isinstance(value, dict):
            raise ValueError(f"{path} rollout result must be an object")
        serialized_learners = _serialized_learners(
            value.get("schema_version"),
            legacy_schema=MATCHED_KL_ROLLOUT_SCHEMA_V1,
            canonical_schema=MATCHED_KL_ROLLOUT_SCHEMA_V2,
            path=path,
        )
        if value.get("config_hash") != expected_config_hash:
            raise ValueError(f"{path} config_hash does not match the aggregation config")
        seed = value.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"{path} has an invalid seed")
        if seed in bt_by_seed:
            raise ValueError(f"duplicate rollout result for seed {seed}")
        comparison_source = comparison_sources.get(seed)
        if comparison_source is None:
            raise ValueError(f"{path} has no same-seed controlled comparison")
        if value.get("artifact_metadata_sha256") != comparison_source["artifact_metadata_sha256"]:
            raise ValueError(f"{path} is bound to a different artifact")
        if value.get("comparison_sha256") != comparison_source["comparison_sha256"]:
            raise ValueError(f"{path} is bound to different comparison bytes")
        if value.get("run_manifest_sha256") != comparison_source["run_manifest_sha256"]:
            raise ValueError(f"{path} is bound to a different run manifest")
        if value.get("environment_identity") != comparison_source["environment_identity"]:
            raise ValueError(f"{path} is bound to a different execution environment")

        recorded_rollouts = value.get("updated_rollouts_sha256")
        if (
            not isinstance(recorded_rollouts, str)
            or len(recorded_rollouts) != 64
            or any(character not in "0123456789abcdef" for character in recorded_rollouts)
        ):
            raise ValueError(f"{path} has an invalid updated_rollouts_sha256")
        rollouts_name = Path(str(value.get("updated_rollouts_jsonl", ""))).name
        if rollouts_name != "updated_rollouts.jsonl":
            raise ValueError(f"{path} records an invalid updated rollouts filename")
        rollouts_path = path.parent / rollouts_name
        if _sha256_file(rollouts_path) != recorded_rollouts:
            raise ValueError(f"{path} updated_rollouts.jsonl SHA256 mismatch")

        reference = value.get("test_reference")
        if not isinstance(reference, dict) or reference.get("source") != (
            "zero_b_common_random_number_rollout"
        ):
            raise ValueError(f"{path} does not use the required zero-B CRN reference")
        num_prompts = reference.get("num_prompts")
        if isinstance(num_prompts, bool) or not isinstance(num_prompts, int) or num_prompts < 2:
            raise ValueError(f"{path} has an invalid test prompt count")
        learners = value.get("learners")
        if not isinstance(learners, dict) or set(learners) != set(serialized_learners):
            raise ValueError(
                f"{path} must contain exactly the serialized BT-MLE and ProRM+ rollout results"
            )
        parsed: dict[str, float] = {}
        for learner_name, serialized_name in zip(
            CANONICAL_LEARNERS, serialized_learners, strict=True
        ):
            learner = learners[serialized_name]
            direction = learner.get("direction") if isinstance(learner, dict) else None
            direction_pcg = direction.get("pcg") if isinstance(direction, dict) else None
            if not isinstance(direction_pcg, dict) or direction_pcg.get("converged") is not True:
                raise ValueError(f"{path} {serialized_name} policy-direction PCG did not converge")
            update = learner.get("measured_kl_update") if isinstance(learner, dict) else None
            if (
                not isinstance(update, dict)
                or update.get("converged") is not True
                or update.get("applied") is not True
            ):
                raise ValueError(f"{path} {serialized_name} measured-KL update did not converge")
            target_kl = _finite_result_float(
                update.get("target_kl"), path=f"{path}:{serialized_name}.target_kl"
            )
            applied_kl = _finite_result_float(
                update.get("applied_measured_kl"),
                path=f"{path}:{serialized_name}.applied_measured_kl",
            )
            if target_kl != expected_kl:
                raise ValueError(f"{path} {serialized_name} used the wrong KL target")
            if abs(applied_kl - expected_kl) / expected_kl > kl_relative_tolerance:
                raise ValueError(f"{path} {serialized_name} did not meet the measured-KL tolerance")
            paired = (
                learner.get("paired_improvement_over_zero_b_reference")
                if isinstance(learner, dict)
                else None
            )
            if not isinstance(paired, dict) or paired.get("schema_version") != (
                "oracle-rollout-improvement/v1"
            ):
                raise ValueError(f"{path} has invalid {serialized_name} paired improvement")
            if paired.get("num_pairs") != num_prompts:
                raise ValueError(f"{path} {serialized_name} uncertainty is not prompt-level")
            if paired.get("significance_claimed") is not False:
                raise ValueError(f"{path} {serialized_name} must not claim significance")
            parsed[learner_name] = _finite_result_float(
                paired.get("mean_difference"),
                path=f"{path}:{serialized_name}.paired_improvement.mean_difference",
            )
        bt_by_seed[seed] = parsed[BT_MLE]
        prorm_plus_by_seed[seed] = parsed[PRORM_PLUS]
        sources[seed] = {
            **comparison_source,
            "rollout_path": relative_posix_reference(path, base=reference_base),
            "rollout_sha256": _sha256_file(path),
            "updated_rollouts_path": relative_posix_reference(
                rollouts_path,
                base=reference_base,
            ),
            "updated_rollouts_sha256": recorded_rollouts,
        }
        if value.get("train_oracle_values_accessed") is not False:
            raise ValueError(f"{path} accessed train oracle values")
        if value.get("raw_oracle_values_serialized") is not False:
            raise ValueError(f"{path} serialized raw oracle values")
    return bt_by_seed, prorm_plus_by_seed, sources


def _aggregate_damping_evidence(
    damping_evidence: dict[float, dict[int, dict[str, object]]],
    *,
    declared_seeds: set[int],
    bootstrap_seed: int,
    bootstrap_resamples: int,
) -> tuple[list[dict[str, object]], bool, bool]:
    from .statistics import aggregate_paired_metrics

    rows: list[dict[str, object]] = []
    all_pcg_converged = True
    sensitivity_nonreversal = True
    for multiplier in sorted(damping_evidence):
        per_seed = damping_evidence[multiplier]
        if set(per_seed) != declared_seeds:
            raise ValueError(f"damping={multiplier} seed set does not match config run.seeds")
        failures: list[dict[str, object]] = []
        bt: dict[int, dict[str, float]] = {}
        prorm_plus: dict[int, dict[str, float]] = {}
        for seed in sorted(declared_seeds):
            record = per_seed[seed]
            if record.get("status") != "ok" or record.get("pcg_converged") is not True:
                all_pcg_converged = False
                failures.append({"seed": seed, **record})
                continue
            bt[seed] = {"test_local_regret": float(record["bt_local_regret"])}
            prorm_plus[seed] = {"test_local_regret": float(record["prorm_plus_local_regret"])}
        if failures:
            if multiplier != 1.0:
                sensitivity_nonreversal = False
            rows.append(
                {
                    "damping_multiplier": multiplier,
                    "status": "incomplete",
                    "all_pcg_converged": False,
                    "local_regret_nonreversal": False,
                    "failures": failures,
                }
            )
            continue
        aggregate = aggregate_paired_metrics(
            bt,
            prorm_plus,
            bootstrap_seed=bootstrap_seed,
            num_resamples=bootstrap_resamples,
        ).to_dict()
        summary = aggregate["metrics"]["test_local_regret"]
        # The preregistration uses the strict negative sign.  An exact zero is
        # inconclusive rather than counted as robustness evidence.
        nonreversal = float(summary["paired_mean"]) < 0.0
        if multiplier != 1.0 and not nonreversal:
            sensitivity_nonreversal = False
        rows.append(
            {
                "damping_multiplier": multiplier,
                "status": "ok",
                "all_pcg_converged": True,
                "local_regret_nonreversal": nonreversal,
                "paired_local_regret": summary,
                "failures": [],
            }
        )
    return rows, all_pcg_converged, sensitivity_nonreversal


def _pre_registered_evidence_status(
    paired_metrics: dict[str, object],
    *,
    all_pcg_converged: bool,
    sensitivity_nonreversal: bool,
) -> dict[str, object]:
    metrics = paired_metrics["metrics"]

    def summary(name: str) -> dict[str, object]:
        value = metrics[name]
        if not isinstance(value, dict):
            raise ValueError(f"aggregate metric {name!r} has an invalid schema")
        return value

    local = summary("test_local_regret")
    error = summary("test_squared_fisher_error")
    cosine = summary("test_fisher_cosine")
    rollout = summary("test_rollout_improvement")
    criteria = {
        "main_local_regret_negative_with_ci": (
            float(local["paired_mean"]) < 0.0 and float(local["bootstrap_ci"]["upper"]) < 0.0
        ),
        "main_direction_error_negative_with_ci": (
            float(error["paired_mean"]) < 0.0 and float(error["bootstrap_ci"]["upper"]) < 0.0
        ),
        "main_fisher_cosine_positive": float(cosine["paired_mean"]) > 0.0,
        "matched_kl_rollout_positive_with_ci": (
            float(rollout["paired_mean"]) > 0.0 and float(rollout["bootstrap_ci"]["lower"]) > 0.0
        ),
        "sensitivity_local_regret_nonreversal": sensitivity_nonreversal,
        "all_pcg_converged": all_pcg_converged,
        # Non-converged/out-of-tolerance KL runs are rejected by the rollout
        # loader before an aggregate can be written.
        "all_measured_kl_updates_converged": True,
    }
    passed = all(criteria.values())
    return {
        "status": "passed" if passed else "not_passed",
        "supports_pre_registered_claim": passed,
        "criteria": criteria,
    }


def _aggregation_source_identity(
    repo_root: str | os.PathLike[str],
    config_path: str | os.PathLike[str],
    *,
    expected_commit: object,
) -> dict[str, object]:
    """Bind aggregation code and config to the formal producer commit."""

    if (
        not isinstance(expected_commit, str)
        or len(expected_commit) not in {40, 64}
        or any(character not in "0123456789abcdef" for character in expected_commit)
    ):
        raise ValueError("formal seed identity has an invalid Git commit")
    root = Path(repo_root).resolve()
    config = Path(config_path).resolve()
    if not (root / ".git").exists():
        raise ValueError(f"aggregation repo root is not a Git worktree: {root}")
    try:
        relative_config = config.relative_to(root)
    except ValueError as error:
        raise ValueError("aggregation config must be inside the declared Git worktree") from error
    git_state = collect_git_state(root)
    if git_state != {"commit": expected_commit, "dirty": False}:
        raise ValueError(
            "aggregation requires a clean checkout at the exact formal producer commit"
        )
    try:
        tracked = subprocess.run(
            [
                "git",
                "ls-files",
                "--error-unmatch",
                "--",
                relative_config.as_posix(),
            ],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError("could not verify the aggregation config in Git") from error
    if tracked.returncode != 0:
        raise ValueError("aggregation config must be tracked by the producer commit")
    return {
        "git_commit": expected_commit,
        "git_dirty": False,
        "config_path": relative_config.as_posix(),
    }


def _aggregate_results(arguments: argparse.Namespace) -> int:
    from .statistics import aggregate_paired_metrics

    config = load_config(arguments.config)
    destination = Path(arguments.output)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing aggregate: {destination}")
    declared = config["run"].get("seeds")
    if not isinstance(declared, list):
        raise ConfigError("aggregate-results requires a config with run.seeds")
    digest = config_hash(config)
    damping_multipliers = _damping_multipliers(config)
    (
        bt_by_seed,
        prorm_plus_by_seed,
        comparison_sources,
        damping_evidence,
    ) = _load_comparison_metrics(
        arguments.results,
        expected_config_hash=digest,
        expected_damping_multipliers=damping_multipliers,
        reference_base=destination.parent,
    )
    if set(bt_by_seed) != set(int(seed) for seed in declared):
        raise ValueError("comparison result seeds must exactly match config run.seeds")
    bt_rollout, prorm_plus_rollout, sources = _load_rollout_metrics(
        arguments.rollouts,
        expected_config_hash=digest,
        comparison_sources=comparison_sources,
        expected_kl=float(config["evaluation"]["kl_budget"]),
        kl_relative_tolerance=float(config["evaluation"]["kl_relative_tolerance"]),
        reference_base=destination.parent,
    )
    if set(bt_rollout) != set(bt_by_seed):
        raise ValueError("rollout result seeds must exactly match comparison seeds")
    ordered_seeds = sorted(sources)
    shared_environment = sources[ordered_seeds[0]].get("environment_identity")
    if not isinstance(shared_environment, dict) or shared_environment.get("formal") is not True:
        raise ValueError("aggregate-results requires complete formal environment identities")
    if any(
        sources[seed].get("environment_identity") != shared_environment
        for seed in ordered_seeds[1:]
    ):
        raise ValueError(
            "all paired seeds must use the same Git commit, image, HF inventory, "
            "account, partition, and GPU model"
        )
    aggregation_source = _aggregation_source_identity(
        arguments.repo_root,
        arguments.config,
        expected_commit=shared_environment.get("git_commit"),
    )
    for seed in bt_by_seed:
        bt_by_seed[seed]["test_rollout_improvement"] = bt_rollout[seed]
        prorm_plus_by_seed[seed]["test_rollout_improvement"] = prorm_plus_rollout[seed]
    evaluation = config["evaluation"]
    descriptive_directions = {
        name: "lower_is_better"
        for name in ("test_oracle_pairwise_nll", "test_oracle_probability_mae")
        if all(name in metrics for metrics in bt_by_seed.values())
    }
    aggregate = aggregate_paired_metrics(
        bt_by_seed,
        prorm_plus_by_seed,
        directions={
            **descriptive_directions,
            "test_pairwise_accuracy": "higher_is_better",
            "test_rollout_improvement": "higher_is_better",
        },
        bootstrap_seed=int(evaluation["paired_bootstrap_seed"]),
        num_resamples=int(evaluation["paired_bootstrap_resamples"]),
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = aggregate.to_dict()
    sensitivity, all_pcg, nonreversal = _aggregate_damping_evidence(
        damping_evidence,
        declared_seeds=set(int(seed) for seed in declared),
        bootstrap_seed=int(evaluation["paired_bootstrap_seed"]),
        bootstrap_resamples=int(evaluation["paired_bootstrap_resamples"]),
    )
    payload["config_hash"] = digest
    payload["environment_identity"] = shared_environment
    payload["aggregation_source"] = aggregation_source
    payload["damping_evidence"] = sensitivity
    payload["pre_registered_evidence"] = _pre_registered_evidence_status(
        payload,
        all_pcg_converged=all_pcg,
        sensitivity_nonreversal=nonreversal,
    )
    payload["sources"] = [{"seed": seed, **sources[seed]} for seed in sorted(sources)]
    atomic_write_json(destination, payload, overwrite=False)
    _print_json(
        {
            "evidence_status": payload["pre_registered_evidence"]["status"],
            "num_seeds": len(bt_by_seed),
            "output": destination.name,
            "status": "ok",
        }
    )
    return 0


def _phase2_aggregate(arguments: argparse.Namespace) -> int:
    from .phase2_aggregate import write_common_beta_seed_aggregate
    from .phase2_config import load_phase2_config_bundle

    bundle = load_phase2_config_bundle(arguments.overlay)
    aggregate = write_common_beta_seed_aggregate(
        bundle.config,
        arguments.results,
        arguments.output,
    )
    _print_json(
        {
            "design_stage": bundle.config["design"]["stage"],
            "formal_eligibility": bundle.config["design"]["formal_eligibility"],
            "evidence_status": aggregate.evidence.to_dict()["status"],
            "num_seeds": len(aggregate.seeds),
            "output": Path(arguments.output).name,
            "phase2_design_sha256": aggregate.phase2_design_sha256,
            "status": "ok",
        }
    )
    return 0


def _phase2_pilot_aggregate(arguments: argparse.Namespace) -> int:
    from .phase2_config import load_phase2_config_bundle
    from .phase2_pilot_aggregate import write_phase2_pilot_aggregate

    bundle = load_phase2_config_bundle(arguments.overlay)
    payload = write_phase2_pilot_aggregate(
        bundle.config,
        arguments.results,
        arguments.output,
        aggregation_identity={
            "schema_version": "phase2-pilot-aggregation-identity/v1",
            "aggregator_git_commit": arguments.aggregator_git_commit,
            "producer_git_commit": arguments.producer_git_commit,
            "image_sha256": arguments.aggregation_image_sha256,
            "hf_inventory_sha256": arguments.aggregation_hf_inventory_sha256,
            "validator_source_sha256": arguments.validator_source_sha256,
        },
        beta_source_aggregate=arguments.beta_source_aggregate,
        horizon_parent_aggregate=arguments.horizon_parent_aggregate,
    )
    selection = payload["selection"]
    if not isinstance(selection, dict):
        raise RuntimeError("pilot aggregate returned malformed selection evidence")
    _print_json(
        {
            "formal_eligibility": False,
            "output": Path(arguments.output).name,
            "output_sha256": _sha256_file(arguments.output),
            "phase2_design_sha256": payload["phase2_design_sha256"],
            "pilot_phase": payload["pilot_phase"],
            "selection_accepted": selection["selection_accepted"],
            "next_action": selection["next_action"],
            "status": "ok",
            "supports_formal_claim": False,
        }
    )
    return 0


def _phase2_sensitivity_aggregate(arguments: argparse.Namespace) -> int:
    from .phase2_config import load_phase2_config_bundle
    from .phase2_sensitivity import write_phase2_sensitivity_aggregate

    bundle = load_phase2_config_bundle(arguments.overlay)
    payload = write_phase2_sensitivity_aggregate(
        bundle.config,
        arguments.results,
        arguments.output,
        primary_aggregate_json=arguments.primary_aggregate,
    )
    _print_json(
        {
            "formal_eligibility": False,
            "num_seeds": payload["num_seeds"],
            "output": Path(arguments.output).name,
            "output_sha256": _sha256_file(arguments.output),
            "phase2_design_sha256": payload["phase2_design_sha256"],
            "primary_efficacy_status": payload["primary_aggregate"]["efficacy_status"],
            "status": "ok",
            "supports_primary_claim": False,
        }
    )
    return 0


def _phase2_mechanism_aggregate(arguments: argparse.Namespace) -> int:
    from .phase2_config import load_phase2_config_bundle
    from .phase2_mechanism import write_phase2_mechanism_aggregate

    bundle = load_phase2_config_bundle(arguments.overlay)
    payload = write_phase2_mechanism_aggregate(
        bundle.config,
        arguments.results,
        arguments.output,
        primary_aggregate_json=arguments.primary_aggregate,
    )
    _print_json(
        {
            "claim_scope_status": payload["claim_scope"]["status"],
            "formal_eligibility": False,
            "num_seeds": payload["num_seeds"],
            "output": Path(arguments.output).name,
            "output_sha256": _sha256_file(arguments.output),
            "phase2_design_sha256": payload["phase2_design_sha256"],
            "primary_efficacy_status": payload["primary_aggregate"]["efficacy_status"],
            "status": "ok",
            "supports_primary_claim": False,
        }
    )
    return 0


def _phase2_failure_manifest(arguments: argparse.Namespace) -> int:
    from .phase2_campaign import write_phase2_seed_failure_manifest
    from .phase2_config import load_phase2_config_bundle

    bundle = load_phase2_config_bundle(arguments.overlay)
    spec = json.loads(Path(arguments.spec).read_text(encoding="utf-8"))
    if not isinstance(spec, dict):
        raise TypeError("Phase-2 failure spec must contain one JSON object")
    payload = write_phase2_seed_failure_manifest(
        bundle.config,
        spec,
        arguments.output,
    )
    _print_json(
        {
            "output": Path(arguments.output).name,
            "output_sha256": _sha256_file(arguments.output),
            "phase2_design_sha256": payload["phase2_design_sha256"],
            "seed": payload["seed"],
            "status": "failed_seed_terminal_recorded",
            "supports_formal_claim": False,
        }
    )
    return 0


def _phase2_success_manifest(arguments: argparse.Namespace) -> int:
    from .phase2_campaign import (
        load_phase2_seed_success_spec,
        write_phase2_seed_success_manifest,
    )
    from .phase2_config import load_phase2_config_bundle

    bundle = load_phase2_config_bundle(arguments.overlay)
    spec = load_phase2_seed_success_spec(arguments.spec)
    attempt_ledger = spec["attempt_ledger"]
    if not isinstance(attempt_ledger, dict):
        raise TypeError("Phase-2 success attempt ledger must be one JSON object")
    payload = write_phase2_seed_success_manifest(
        bundle.config,
        arguments.result,
        attempt_ledger,
        arguments.output,
    )
    attempts = payload["attempt_ledger"]["attempts"]
    if not isinstance(attempts, list):
        raise RuntimeError("success terminal manifest returned malformed attempt evidence")
    _print_json(
        {
            "attempt_count": len(attempts),
            "output": Path(arguments.output).name,
            "output_sha256": _sha256_file(arguments.output),
            "phase2_design_sha256": payload["phase2_design_sha256"],
            "result_sha256": payload["result"]["sha256"],
            "rollout_sha256": payload["rollout"]["sha256"],
            "seed": payload["seed"],
            "status": "successful_seed_terminal_recorded",
            "supports_formal_claim": False,
        }
    )
    return 0


def _phase2_campaign_finalize(arguments: argparse.Namespace) -> int:
    from .phase2_campaign import write_phase2_campaign_terminal
    from .phase2_config import load_phase2_config_bundle

    bundle = load_phase2_config_bundle(arguments.overlay)
    payload = write_phase2_campaign_terminal(
        bundle.config,
        arguments.terminals,
        arguments.output,
        aggregate_output_json=arguments.aggregate_output,
    )
    _print_json(
        {
            "failed_seeds": payload["failed_seeds"],
            "output": Path(arguments.output).name,
            "output_sha256": _sha256_file(arguments.output),
            "primary_ci_computed": payload["primary_ci_computed"],
            "status": payload["status"],
            "supports_formal_claim": payload["supports_formal_claim"],
        }
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser without touching network or GPU state."""

    parser = argparse.ArgumentParser(
        prog="prorm",
        description="Validated controls and experiment entry points for ProRM/ProRM+.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    config_parser = subparsers.add_parser(
        "config-check",
        help="strictly validate a YAML config and print its canonical SHA-256 hash",
    )
    config_parser.add_argument("config", help="path to configs/smoke.yaml or configs/main.yaml")
    config_parser.set_defaults(handler=_config_check)

    phase2_config_parser = subparsers.add_parser(
        "phase2-config-check",
        help="validate the Phase-2 overlay and print both bound configuration identities",
    )
    phase2_config_parser.add_argument(
        "overlay",
        help="path to a validated Phase-2 pilot or confirmatory overlay",
    )
    phase2_config_parser.set_defaults(handler=_phase2_config_check)

    environment_parser = subparsers.add_parser(
        "env-report",
        help="emit an offline run manifest for a validated config",
    )
    environment_parser.add_argument("config", help="path to a YAML experiment config")
    environment_parser.add_argument(
        "--repo-root",
        default=".",
        help="Git repository root (default: current directory)",
    )
    environment_parser.add_argument(
        "--seed",
        type=int,
        help="selected seed for this run (required later by controlled-compare)",
    )
    environment_parser.add_argument(
        "--output",
        "-o",
        help="atomically write JSON here instead of printing the manifest",
    )
    environment_parser.set_defaults(handler=_env_report)

    data_parser = subparsers.add_parser(
        "data-check",
        help="validate an exact TrainingEdgeRecord JSONL file",
    )
    data_parser.add_argument("jsonl", help="path to training-edge/v1 JSONL")
    data_parser.set_defaults(handler=_data_check)

    prompt_parser = subparsers.add_parser(
        "prepare-prompts",
        help="download a pinned prompt dataset and write deterministic prompt splits",
    )
    prompt_parser.add_argument("config", help="path to a validated YAML experiment config")
    prompt_parser.add_argument("output", help="destination prompt/v1 JSONL")
    prompt_parser.add_argument("--seed", type=int, help="one seed declared by run.seed(s)")
    prompt_parser.set_defaults(handler=_prepare_prompts)

    synthetic_parser = subparsers.add_parser(
        "synthetic-check",
        help="run the CPU-only end-to-end numerical integration benchmark",
    )
    synthetic_parser.add_argument("--seed", type=int, default=0)
    synthetic_parser.add_argument("--output", "-o", help="atomically write JSON here")
    synthetic_parser.set_defaults(handler=_synthetic_check)

    closed_form_parser = subparsers.add_parser(
        "closed-form-check",
        help="emit the audited four-response population ProRM calculation",
    )
    closed_form_parser.add_argument(
        "--output",
        "-o",
        help="atomically write JSON here instead of printing the full audit",
    )
    closed_form_parser.set_defaults(handler=_closed_form_check)

    materialize_parser = subparsers.add_parser(
        "controlled-materialize",
        help="materialize pinned Phase-1 candidates, geometry, oracle labels, and features",
    )
    materialize_parser.add_argument("config")
    materialize_parser.add_argument("artifact_dir")
    materialize_parser.add_argument("--seed", type=int, required=True)
    materialize_parser.add_argument("--device", default="cuda")
    materialize_parser.add_argument(
        "--allow-download",
        action="store_true",
        help="allow Hugging Face network access (formal HPC jobs remain offline by default)",
    )
    materialize_parser.set_defaults(handler=_controlled_materialize)

    compare_parser = subparsers.add_parser(
        "controlled-compare",
        help="run paired fixed-step BT/ProRM+ training over an integrity-checked artifact",
    )
    compare_parser.add_argument("config")
    compare_parser.add_argument("artifact_dir")
    compare_parser.add_argument("output")
    compare_parser.add_argument("--seed", type=int, required=True)
    compare_parser.add_argument("--device", default="cpu")
    compare_parser.add_argument(
        "--run-manifest",
        required=True,
        help="selected-seed compatibility manifest to bind into the result",
    )
    compare_parser.set_defaults(handler=_controlled_compare)

    rollout_parser = subparsers.add_parser(
        "controlled-rollout",
        help="match BT/ProRM+ policy updates to measured KL and run paired oracle rollouts",
    )
    rollout_parser.add_argument("config")
    rollout_parser.add_argument("artifact_dir")
    rollout_parser.add_argument("comparison")
    rollout_parser.add_argument("output")
    rollout_parser.add_argument("--seed", type=int, required=True)
    rollout_parser.add_argument("--device", default="cuda")
    rollout_parser.add_argument(
        "--allow-download",
        action="store_true",
        help="allow Hugging Face network access (formal HPC jobs remain offline by default)",
    )
    rollout_parser.set_defaults(handler=_controlled_rollout)

    phase2_run_parser = subparsers.add_parser(
        "phase2-run",
        help="run one fresh-head common-beta seed over a bound source artifact",
    )
    phase2_run_parser.add_argument(
        "overlay",
        help="path to a validated Phase-2 pilot or confirmatory overlay",
    )
    phase2_run_parser.add_argument("artifact", help="source materialization artifact directory")
    phase2_run_parser.add_argument("manifest", help="selected-seed source run manifest")
    phase2_run_parser.add_argument("output", help="new common-beta result JSON")
    phase2_run_parser.add_argument("--seed", type=int, required=True)
    phase2_run_parser.add_argument("--device", default="cuda")
    phase2_run_parser.add_argument(
        "--beta-source-aggregate",
        help=(
            "identity-bound calibration/freeze aggregate; required by pilot-freeze "
            "and confirmatory designs"
        ),
    )
    phase2_run_parser.add_argument(
        "--horizon-parent-aggregate",
        help=(
            "identity-bound failed prior-horizon aggregate; required by escalated "
            "calibration designs (pilot-freeze defaults to --beta-source-aggregate)"
        ),
    )
    phase2_run_parser.set_defaults(handler=_phase2_run)

    phase2_sensitivity_run_parser = subparsers.add_parser(
        "phase2-sensitivity-run",
        help="run the complete fixed ridge/frozen-beta secondary grid for one formal seed",
    )
    phase2_sensitivity_run_parser.add_argument(
        "overlay",
        help="validated formal Phase-2 confirmatory overlay",
    )
    phase2_sensitivity_run_parser.add_argument(
        "artifact",
        help="source materialization artifact directory",
    )
    phase2_sensitivity_run_parser.add_argument(
        "manifest",
        help="selected-seed source run manifest",
    )
    phase2_sensitivity_run_parser.add_argument(
        "primary_result",
        help="immutable successful primary result for the same seed",
    )
    phase2_sensitivity_run_parser.add_argument(
        "output",
        help="new per-seed sensitivity artifact",
    )
    phase2_sensitivity_run_parser.add_argument("--seed", type=int, required=True)
    phase2_sensitivity_run_parser.add_argument("--device", default="cuda")
    phase2_sensitivity_run_parser.set_defaults(handler=_phase2_sensitivity_run)

    phase2_mechanism_run_parser = subparsers.add_parser(
        "phase2-mechanism-run",
        help="evaluate both held-out mechanism qualifiers for one formal seed",
    )
    phase2_mechanism_run_parser.add_argument(
        "overlay",
        help="validated formal Phase-2 confirmatory overlay",
    )
    phase2_mechanism_run_parser.add_argument(
        "artifact",
        help="source materialization artifact directory",
    )
    phase2_mechanism_run_parser.add_argument(
        "manifest",
        help="selected-seed source run manifest",
    )
    phase2_mechanism_run_parser.add_argument(
        "primary_result",
        help="immutable successful primary result for the same seed",
    )
    phase2_mechanism_run_parser.add_argument(
        "output",
        help="new per-seed mechanism artifact",
    )
    phase2_mechanism_run_parser.add_argument("--seed", type=int, required=True)
    phase2_mechanism_run_parser.add_argument("--device", default="cuda")
    phase2_mechanism_run_parser.set_defaults(handler=_phase2_mechanism_run)

    audit_parser = subparsers.add_parser(
        "estimand-audit",
        help="evaluate saved train directions on test F/g* without loading an LLM or GPU",
    )
    audit_parser.add_argument("config")
    audit_parser.add_argument("artifact_dir")
    audit_parser.add_argument("comparison")
    audit_parser.add_argument("rollout")
    audit_parser.add_argument("output")
    audit_parser.add_argument("--seed", type=int, required=True)
    audit_parser.add_argument(
        "--prompt-chunk-size",
        type=int,
        default=16,
        help="number of test prompts per CPU float64 geometry chunk (default: 16)",
    )
    audit_parser.set_defaults(handler=_estimand_audit)

    optimization_audit_parser = subparsers.add_parser(
        "optimization-audit",
        help="recompute full-train unclipped head gradients without an optimizer or GPU",
    )
    optimization_audit_parser.add_argument("config")
    optimization_audit_parser.add_argument("artifact_dir")
    optimization_audit_parser.add_argument("comparison")
    optimization_audit_parser.add_argument("output")
    optimization_audit_parser.add_argument("--seed", type=int, required=True)
    optimization_audit_parser.set_defaults(handler=_optimization_audit)

    aggregate_parser = subparsers.add_parser(
        "aggregate-results",
        help="aggregate all declared paired seeds with a deterministic bootstrap CI",
    )
    aggregate_parser.add_argument("config")
    aggregate_parser.add_argument("output")
    aggregate_parser.add_argument("results", nargs="+")
    aggregate_parser.add_argument(
        "--repo-root",
        required=True,
        help="clean Git checkout at the exact producer commit used by every input seed",
    )
    aggregate_parser.add_argument(
        "--rollouts",
        nargs="+",
        required=True,
        help="matched-kl-rollout/v1 or v2 JSON files, one for every declared seed",
    )
    aggregate_parser.set_defaults(handler=_aggregate_results)

    phase2_aggregate_parser = subparsers.add_parser(
        "phase2-aggregate",
        help="strictly validate and aggregate the complete declared-seed common-beta campaign",
    )
    phase2_aggregate_parser.add_argument(
        "overlay",
        help="path to a validated Phase-2 pilot or confirmatory overlay",
    )
    phase2_aggregate_parser.add_argument("output", help="new aggregate JSON")
    phase2_aggregate_parser.add_argument(
        "results",
        nargs="+",
        help="one Phase-2 result JSON for every seed declared by the overlay",
    )
    phase2_aggregate_parser.set_defaults(handler=_phase2_aggregate)

    phase2_pilot_aggregate_parser = subparsers.add_parser(
        "phase2-pilot-aggregate",
        help="strictly aggregate all three target-free calibration or freeze pilot seeds",
    )
    phase2_pilot_aggregate_parser.add_argument(
        "overlay",
        help="validated Phase-2 pilot calibration or freeze overlay",
    )
    phase2_pilot_aggregate_parser.add_argument("output", help="new pilot aggregate JSON")
    phase2_pilot_aggregate_parser.add_argument(
        "results",
        nargs="+",
        help="one target-free pilot result JSON for every declared pilot seed",
    )
    phase2_pilot_aggregate_parser.add_argument(
        "--aggregator-git-commit",
        required=True,
        help="exact commit executing and validating this aggregate",
    )
    phase2_pilot_aggregate_parser.add_argument(
        "--producer-git-commit",
        required=True,
        help="exact commit that produced all three immutable pilot seeds",
    )
    phase2_pilot_aggregate_parser.add_argument(
        "--aggregation-image-sha256",
        required=True,
        help="validated SIF digest used by the producer and aggregation job",
    )
    phase2_pilot_aggregate_parser.add_argument(
        "--aggregation-hf-inventory-sha256",
        required=True,
        help="validated base-config HF inventory digest",
    )
    phase2_pilot_aggregate_parser.add_argument(
        "--validator-source-sha256",
        required=True,
        help="SHA256 of the committed pilot aggregate validator source",
    )
    phase2_pilot_aggregate_parser.add_argument(
        "--beta-source-aggregate",
        help=(
            "immediate beta parent: accepted calibration for the first freeze, "
            "or the preceding failed freeze aggregate for a beta-grid retry"
        ),
    )
    phase2_pilot_aggregate_parser.add_argument(
        "--horizon-parent-aggregate",
        help=(
            "aggregate authorizing the frozen horizon: accepted calibration "
            "for freeze, or failed prior-horizon evidence for escalation"
        ),
    )
    phase2_pilot_aggregate_parser.set_defaults(handler=_phase2_pilot_aggregate)

    phase2_sensitivity_aggregate_parser = subparsers.add_parser(
        "phase2-sensitivity-aggregate",
        help="validate and aggregate the complete exact-30 fixed sensitivity grid",
    )
    phase2_sensitivity_aggregate_parser.add_argument(
        "overlay",
        help="validated formal Phase-2 confirmatory overlay",
    )
    phase2_sensitivity_aggregate_parser.add_argument(
        "primary_aggregate",
        help="immutable exact-30 primary aggregate",
    )
    phase2_sensitivity_aggregate_parser.add_argument(
        "output",
        help="new sensitivity aggregate JSON",
    )
    phase2_sensitivity_aggregate_parser.add_argument(
        "results",
        nargs="+",
        help="one complete sensitivity artifact for every formal seed",
    )
    phase2_sensitivity_aggregate_parser.set_defaults(handler=_phase2_sensitivity_aggregate)

    phase2_mechanism_aggregate_parser = subparsers.add_parser(
        "phase2-mechanism-aggregate",
        help="validate and aggregate both exact-30 mechanism qualifiers",
    )
    phase2_mechanism_aggregate_parser.add_argument(
        "overlay",
        help="validated formal Phase-2 confirmatory overlay",
    )
    phase2_mechanism_aggregate_parser.add_argument(
        "primary_aggregate",
        help="immutable exact-30 primary aggregate",
    )
    phase2_mechanism_aggregate_parser.add_argument(
        "output",
        help="new mechanism aggregate JSON",
    )
    phase2_mechanism_aggregate_parser.add_argument(
        "results",
        nargs="+",
        help="one complete mechanism artifact for every formal seed",
    )
    phase2_mechanism_aggregate_parser.set_defaults(handler=_phase2_mechanism_aggregate)

    phase2_failure_parser = subparsers.add_parser(
        "phase2-failure-manifest",
        help="immutably bind one failed formal seed and its attempt ledger",
    )
    phase2_failure_parser.add_argument(
        "overlay",
        help="validated formal Phase-2 confirmatory overlay",
    )
    phase2_failure_parser.add_argument(
        "spec",
        help="strict JSON failure/attempt/evidence specification",
    )
    phase2_failure_parser.add_argument("output", help="new immutable seed failure manifest")
    phase2_failure_parser.set_defaults(handler=_phase2_failure_manifest)

    phase2_success_parser = subparsers.add_parser(
        "phase2-success-manifest",
        help="immutably bind one successful formal seed result and its attempt ledger",
    )
    phase2_success_parser.add_argument(
        "overlay",
        help="validated formal Phase-2 confirmatory overlay",
    )
    phase2_success_parser.add_argument(
        "result",
        help="identity-bound successful Phase-2 result JSON",
    )
    phase2_success_parser.add_argument(
        "spec",
        help="strict JSON object containing exactly one attempt_ledger field",
    )
    phase2_success_parser.add_argument("output", help="new immutable seed success manifest")
    phase2_success_parser.set_defaults(handler=_phase2_success_manifest)

    phase2_finalize_parser = subparsers.add_parser(
        "phase2-campaign-finalize",
        help="finalize exactly 30 success- or failure-manifest terminal slots",
    )
    phase2_finalize_parser.add_argument(
        "overlay",
        help="validated formal Phase-2 confirmatory overlay",
    )
    phase2_finalize_parser.add_argument("output", help="new campaign terminal JSON")
    phase2_finalize_parser.add_argument(
        "aggregate_output",
        help="reserved primary aggregate path; remains absent after any seed failure",
    )
    phase2_finalize_parser.add_argument(
        "terminals",
        nargs="+",
        help="exactly one success or failure terminal manifest for every declared seed",
    )
    phase2_finalize_parser.set_defaults(handler=_phase2_campaign_finalize)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command, returning zero on success and two on invalid input."""

    parser = build_parser()
    arguments = parser.parse_args(argv)
    memory_destination: str | None = None
    torch: ModuleType | None = None
    status = "ok"
    try:
        raw_memory_destination = compatibility_value(
            os.environ,
            "PRORM_MEMORY_REPORT",
            "SRM_MEMORY_REPORT",
        )
        if raw_memory_destination is not None and not isinstance(raw_memory_destination, str):
            raise TypeError("PRORM_MEMORY_REPORT must be a string path")
        memory_destination = raw_memory_destination
        torch = _start_cuda_memory_tracking() if memory_destination else None
        exit_code = int(arguments.handler(arguments))
        status = "ok" if exit_code == 0 else "error"
    except (
        ConfigError,
        ImportError,
        RuntimeError,
        SchemaError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        status = "error"
        print(f"error: {error}", file=sys.stderr)
        exit_code = 2
    if memory_destination and torch is not None:
        try:
            _write_cuda_memory_report(
                torch,
                memory_destination,
                command=str(arguments.command),
                status=status,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            print(f"error: failed to write CUDA memory report: {error}", file=sys.stderr)
            return 2
    return exit_code


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess entry points.
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
