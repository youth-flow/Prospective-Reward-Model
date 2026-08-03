"""Command-line entry points for the single ProRM experiment workflow."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .audit import audit_fisher_trpo_run
from .config import TRPO_PROTOCOL, ConfigError, config_hash, load_config
from .direct_policy_evaluation import (
    aggregate_six_policy,
    assemble_six_policy_seed,
    audit_six_policy_run,
    export_direct_policy_adapter,
    run_direct_policy_rollout,
    smoke_direct_policy,
)
from .direct_preference import (
    aggregate_direct_preference,
    audit_direct_preference,
    compute_reference_logps,
    evaluate_direct_preference_seed,
    import_reference_logps,
    train_direct_preference,
)
from .exact_phase import materialize_exact_delta
from .exact_policy import export_exact_ngd_adapters
from .exact_run import run_exact_reward_comparison
from .fisher_crossfit import run_fisher_crossfit, select_fisher_regularization
from .ngd_evaluation import (
    aggregate_ngd_evaluations,
    audit_ngd_run,
    run_ngd_evaluation,
)
from .pipeline import (
    import_materialization_stage,
    run_adapter_stage,
    run_fisher_crossfit_stage,
    run_kl_calibration_aggregate_stage,
    run_kl_calibration_policy_stage,
    run_materialization_stage,
    run_policy_rollout_stage,
    run_reward_stage,
    run_rollout_aggregate_stage,
)
from .real_policy_evaluation import (
    aggregate_real_policy,
    assemble_real_policy_seed,
    audit_real_policy_run,
    export_real_policy_adapters,
    run_real_policy_rollout,
    smoke_real_policy_writeback,
)
from .real_policy_evaluation import (
    policy_names as real_policy_names,
)
from .real_policy_extension import (
    aggregate_extended_real_policy,
    assemble_extended_real_policy_seed,
    audit_extended_real_policy_run,
    extend_real_policy_rollout,
)
from .rollout import evaluate_policy_rollouts, policy_instance_names
from .statistics import aggregate_results
from .trpo_policy import export_trpo_adapters
from .trpo_run import run_trpo_reward_comparison


def _write_json(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _config_check(arguments: argparse.Namespace) -> int:
    config = load_config(arguments.config)
    print(json.dumps({"config_sha256": config_hash(config), "config": config}, ensure_ascii=False))
    return 0


def _materialize(arguments: argparse.Namespace) -> int:
    materialize_exact_delta(
        load_config(arguments.config),
        seed=arguments.seed,
        artifact_dir=arguments.artifact_dir,
        device=arguments.device,
        local_files_only=not arguments.allow_download,
        reuse_splits_from=arguments.reuse_splits_from,
    )
    return 0


def _import_materialization(arguments: argparse.Namespace) -> int:
    import_materialization_stage(
        load_config(arguments.config),
        arguments.source_seed_root,
        arguments.target_seed_root,
        arguments.affected_stage_analysis,
        seed=arguments.seed,
    )
    return 0


def _train_rewards(arguments: argparse.Namespace) -> int:
    config = load_config(arguments.config)
    if config["protocol"] == TRPO_PROTOCOL:
        if arguments.fisher_selection is None:
            raise ValueError("--fisher-selection is required by the Fisher-TRPO protocol")
        run_trpo_reward_comparison(
            config,
            arguments.artifact_dir,
            arguments.fisher_selection,
            arguments.output,
            seed=arguments.seed,
            device=arguments.device,
            reuse_mle_from=arguments.reuse_mle_from,
        )
    else:
        run_exact_reward_comparison(
            config,
            arguments.artifact_dir,
            arguments.output,
            seed=arguments.seed,
            device=arguments.device,
            reuse_pro_from=arguments.reuse_pro_from,
        )
    return 0


def _fisher_crossfit(arguments: argparse.Namespace) -> int:
    run_fisher_crossfit(
        load_config(arguments.config),
        arguments.artifact_dir,
        arguments.output,
        seed=arguments.seed,
        device=arguments.device,
    )
    return 0


def _select_fisher(arguments: argparse.Namespace) -> int:
    select_fisher_regularization(
        load_config(arguments.config),
        arguments.crossfit_results,
        arguments.output,
    )
    return 0


def _export_policies(arguments: argparse.Namespace) -> int:
    config = load_config(arguments.config)
    exporter = (
        export_trpo_adapters if config["protocol"] == TRPO_PROTOCOL else export_exact_ngd_adapters
    )
    exporter(
        config,
        arguments.artifact_dir,
        arguments.reward_result,
        arguments.output_dir,
        seed=arguments.seed,
        device=arguments.device,
        local_files_only=not arguments.allow_download,
    )
    return 0


def _evaluate_rollouts(arguments: argparse.Namespace) -> int:
    evaluate_policy_rollouts(
        load_config(arguments.config),
        arguments.artifact_dir,
        arguments.adapter_dir,
        arguments.output_dir,
        seed=arguments.seed,
        device=arguments.device,
        local_files_only=not arguments.allow_download,
    )
    return 0


def _run_seed(arguments: argparse.Namespace) -> int:
    config = load_config(arguments.config)
    if config["protocol"] == TRPO_PROTOCOL:
        raise ValueError(
            "Fisher-TRPO requires the cross-seed staged workflow; run-seed is disabled"
        )
    root = Path(arguments.output_dir)
    run_materialization_stage(
        config,
        root,
        seed=arguments.seed,
        device=arguments.device,
        local_files_only=not arguments.allow_download,
    )
    run_reward_stage(
        config,
        root,
        seed=arguments.seed,
        device=arguments.device,
    )
    run_adapter_stage(
        config,
        root,
        seed=arguments.seed,
        device=arguments.device,
        local_files_only=not arguments.allow_download,
    )
    for policy_name in policy_instance_names(config):
        run_policy_rollout_stage(
            config,
            root,
            policy_name=policy_name,
            seed=arguments.seed,
            device=arguments.device,
            local_files_only=not arguments.allow_download,
            reuse_splits_from=arguments.reuse_splits_from,
        )
    run_rollout_aggregate_stage(config, root, seed=arguments.seed)
    return 0


def _run_stage(arguments: argparse.Namespace) -> int:
    config = load_config(arguments.config)
    common = {"seed": arguments.seed}
    if arguments.stage == "materialize":
        run_materialization_stage(
            config,
            arguments.seed_root,
            **common,
            device=arguments.device,
            local_files_only=not arguments.allow_download,
            reuse_splits_from=arguments.reuse_splits_from,
        )
    elif arguments.stage == "fisher-crossfit":
        run_fisher_crossfit_stage(
            config,
            arguments.seed_root,
            **common,
            device=arguments.device,
        )
    elif arguments.stage == "reward":
        run_reward_stage(
            config,
            arguments.seed_root,
            **common,
            device=arguments.device,
            fisher_selection=arguments.fisher_selection,
            reuse_mle_from=arguments.reuse_mle_from,
        )
    elif arguments.stage == "adapters":
        run_adapter_stage(
            config,
            arguments.seed_root,
            **common,
            device=arguments.device,
            local_files_only=not arguments.allow_download,
        )
    elif arguments.stage == "kl-calibration-policy":
        if arguments.policy_name is None or arguments.policy_name == "pi0":
            raise ValueError("--policy-name must name an updated policy for kl-calibration-policy")
        run_kl_calibration_policy_stage(
            config,
            arguments.seed_root,
            **common,
            policy_name=arguments.policy_name,
            device=arguments.device,
            local_files_only=not arguments.allow_download,
        )
    elif arguments.stage == "kl-calibration-aggregate":
        run_kl_calibration_aggregate_stage(
            config,
            arguments.seed_root,
            **common,
        )
    elif arguments.stage == "rollout-policy":
        if arguments.policy_name is None:
            raise ValueError("--policy-name is required for rollout-policy")
        run_policy_rollout_stage(
            config,
            arguments.seed_root,
            **common,
            policy_name=arguments.policy_name,
            device=arguments.device,
            local_files_only=not arguments.allow_download,
        )
    elif arguments.stage == "rollout-aggregate":
        run_rollout_aggregate_stage(config, arguments.seed_root, **common)
    else:  # pragma: no cover - argparse enforces choices
        raise ValueError(f"unknown stage: {arguments.stage}")
    return 0


def _policy_names(arguments: argparse.Namespace) -> int:
    for name in policy_instance_names(load_config(arguments.config)):
        print(name)
    return 0


def _aggregate(arguments: argparse.Namespace) -> int:
    payload = aggregate_results(
        load_config(arguments.config),
        arguments.reward_results,
        arguments.rollout_results,
    )
    output = Path(arguments.output)
    if output.exists():
        with output.open("r", encoding="utf-8") as stream:
            if json.load(stream) != payload:
                raise ValueError(f"existing aggregate does not match validated inputs: {output}")
        print("stage=three-seed-aggregate status=reused", flush=True)
    else:
        _write_json(output, payload)
        print("stage=three-seed-aggregate status=complete", flush=True)
    return 0


def _audit(arguments: argparse.Namespace) -> int:
    audit_fisher_trpo_run(
        load_config(arguments.config),
        arguments.run_root,
        arguments.source_run_root,
        arguments.output,
    )
    print("stage=integrity-audit status=complete", flush=True)
    return 0


def _evaluate_ngd(arguments: argparse.Namespace) -> int:
    run_ngd_evaluation(
        load_config(arguments.config),
        arguments.artifact_dir,
        arguments.reward_result,
        arguments.output,
        seed=arguments.seed,
        device=arguments.device,
    )
    print("stage=ngd-evaluation status=complete", flush=True)
    return 0


def _aggregate_ngd(arguments: argparse.Namespace) -> int:
    aggregate_ngd_evaluations(
        load_config(arguments.config),
        arguments.results,
        arguments.output,
    )
    print("stage=ngd-aggregate status=complete", flush=True)
    return 0


def _audit_ngd(arguments: argparse.Namespace) -> int:
    audit_ngd_run(
        load_config(arguments.config),
        arguments.run_root,
        arguments.source_run_root,
        arguments.output,
    )
    print("stage=ngd-audit status=complete", flush=True)
    return 0


def _cache_direct_reference(arguments: argparse.Namespace) -> int:
    compute_reference_logps(
        arguments.extension_config,
        arguments.artifact_dir,
        arguments.output_dir,
        seed=arguments.seed,
        device=arguments.device,
    )
    print("stage=direct-reference status=complete", flush=True)
    return 0


def _import_direct_reference(arguments: argparse.Namespace) -> int:
    import_reference_logps(
        arguments.source_extension_config,
        arguments.target_extension_config,
        arguments.artifact_dir,
        arguments.source_reference_dir,
        arguments.target_reference_dir,
        seed=arguments.seed,
    )
    return 0


def _train_direct(arguments: argparse.Namespace) -> int:
    train_direct_preference(
        arguments.extension_config,
        arguments.artifact_dir,
        arguments.reference_dir,
        arguments.output_dir,
        seed=arguments.seed,
        beta=arguments.beta,
        method=arguments.method,
        device=arguments.device,
    )
    print("stage=direct-train status=complete", flush=True)
    return 0


def _evaluate_direct(arguments: argparse.Namespace) -> int:
    evaluate_direct_preference_seed(
        arguments.extension_config,
        arguments.artifact_dir,
        arguments.source_reward_result,
        arguments.reference_dir,
        arguments.fits_dir,
        arguments.baseline_evaluation,
        arguments.output,
        seed=arguments.seed,
    )
    print("stage=direct-evaluation status=complete", flush=True)
    return 0


def _aggregate_direct(arguments: argparse.Namespace) -> int:
    aggregate_direct_preference(
        arguments.extension_config,
        arguments.results,
        arguments.output,
    )
    print("stage=direct-aggregate status=complete", flush=True)
    return 0


def _audit_direct(arguments: argparse.Namespace) -> int:
    audit_direct_preference(
        arguments.extension_config,
        arguments.run_root,
        arguments.output,
    )
    print("stage=direct-audit status=complete", flush=True)
    return 0


def _real_policy_names(arguments: argparse.Namespace) -> int:
    for name in real_policy_names():
        print(name)
    return 0


def _export_real_policy_adapters(arguments: argparse.Namespace) -> int:
    export_real_policy_adapters(
        load_config(arguments.config),
        arguments.artifact_dir,
        arguments.reward_result,
        arguments.output_dir,
        seed=arguments.seed,
        device=arguments.device,
        local_files_only=not arguments.allow_download,
    )
    print("stage=real-policy-adapters status=complete", flush=True)
    return 0


def _rollout_real_policy(arguments: argparse.Namespace) -> int:
    run_real_policy_rollout(
        load_config(arguments.config),
        arguments.artifact_dir,
        arguments.reward_result,
        arguments.adapter_dir,
        arguments.output_dir,
        policy_name=arguments.policy_name,
        seed=arguments.seed,
        device=arguments.device,
        local_files_only=not arguments.allow_download,
    )
    print("stage=real-policy-rollout status=complete", flush=True)
    return 0


def _smoke_real_policy(arguments: argparse.Namespace) -> int:
    smoke_real_policy_writeback(
        load_config(arguments.config),
        arguments.artifact_dir,
        arguments.reward_result,
        arguments.adapter_dir,
        arguments.output,
        seed=arguments.seed,
        device=arguments.device,
        local_files_only=not arguments.allow_download,
    )
    print("stage=real-policy-smoke status=complete", flush=True)
    return 0


def _assemble_real_policy(arguments: argparse.Namespace) -> int:
    assemble_real_policy_seed(
        load_config(arguments.config),
        arguments.artifact_dir,
        arguments.reward_result,
        arguments.adapter_dir,
        arguments.policy_root,
        arguments.output,
        seed=arguments.seed,
    )
    print("stage=real-policy-seed-aggregate status=complete", flush=True)
    return 0


def _aggregate_real_policy(arguments: argparse.Namespace) -> int:
    aggregate_real_policy(load_config(arguments.config), arguments.results, arguments.output)
    print("stage=real-policy-three-seed-aggregate status=complete", flush=True)
    return 0


def _audit_real_policy(arguments: argparse.Namespace) -> int:
    audit_real_policy_run(
        load_config(arguments.config),
        arguments.source_run_root,
        arguments.run_root,
        arguments.output,
    )
    print("stage=real-policy-audit status=complete", flush=True)
    return 0


def _extend_real_policy(arguments: argparse.Namespace) -> int:
    extend_real_policy_rollout(
        arguments.extension_config,
        arguments.artifact_dir,
        arguments.reward_result,
        arguments.adapter_dir,
        arguments.source_rollout_dir,
        arguments.output_dir,
        policy_name=arguments.policy_name,
        seed=arguments.seed,
        device=arguments.device,
        local_files_only=not arguments.allow_download,
    )
    print("stage=real-policy-rollout-extension-4-to-6 status=complete", flush=True)
    return 0


def _assemble_extended_real_policy(arguments: argparse.Namespace) -> int:
    assemble_extended_real_policy_seed(
        arguments.extension_config,
        arguments.artifact_dir,
        arguments.reward_result,
        arguments.adapter_dir,
        arguments.source_policy_root,
        arguments.policy_root,
        arguments.output,
        seed=arguments.seed,
    )
    print("stage=real-policy-m6-seed-aggregate status=complete", flush=True)
    return 0


def _aggregate_extended_real_policy(arguments: argparse.Namespace) -> int:
    aggregate_extended_real_policy(arguments.extension_config, arguments.results, arguments.output)
    print("stage=real-policy-m6-three-seed-aggregate status=complete", flush=True)
    return 0


def _audit_extended_real_policy(arguments: argparse.Namespace) -> int:
    audit_extended_real_policy_run(
        arguments.extension_config,
        arguments.source_run_root,
        arguments.base_real_run_root,
        arguments.run_root,
        arguments.output,
    )
    print("stage=real-policy-m6-audit status=complete", flush=True)
    return 0


def _export_direct_policy_adapter(arguments: argparse.Namespace) -> int:
    export_direct_policy_adapter(
        arguments.config,
        arguments.artifact_dir,
        arguments.fit_dir,
        arguments.output_dir,
        seed=arguments.seed,
        method=arguments.method,
        device=arguments.device,
    )
    print("stage=direct-policy-adapter status=complete", flush=True)
    return 0


def _rollout_direct_policy(arguments: argparse.Namespace) -> int:
    run_direct_policy_rollout(
        arguments.config,
        arguments.artifact_dir,
        arguments.fit_dir,
        arguments.adapter_dir,
        arguments.output_dir,
        seed=arguments.seed,
        method=arguments.method,
        device=arguments.device,
    )
    print("stage=direct-policy-rollout-m6 status=complete", flush=True)
    return 0


def _smoke_direct_policy(arguments: argparse.Namespace) -> int:
    smoke_direct_policy(
        arguments.config,
        arguments.artifact_dir,
        arguments.fit_dir,
        arguments.adapter_dir,
        arguments.output,
        seed=arguments.seed,
        method=arguments.method,
        device=arguments.device,
    )
    print("stage=direct-policy-smoke status=complete", flush=True)
    return 0


def _assemble_six_policy(arguments: argparse.Namespace) -> int:
    assemble_six_policy_seed(
        arguments.config,
        arguments.artifact_dir,
        arguments.reward_result,
        arguments.direct_seed_root,
        arguments.source_evaluation,
        arguments.output,
        seed=arguments.seed,
    )
    print("stage=six-policy-seed-aggregate status=complete", flush=True)
    return 0


def _aggregate_six_policy(arguments: argparse.Namespace) -> int:
    aggregate_six_policy(arguments.config, arguments.results, arguments.output)
    print("stage=six-policy-three-seed-aggregate status=complete", flush=True)
    return 0


def _audit_six_policy(arguments: argparse.Namespace) -> int:
    audit_six_policy_run(arguments.config, arguments.run_root, arguments.output)
    print("stage=six-policy-audit status=complete", flush=True)
    return 0


def _add_execution_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-download", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prorm",
        description=(
            "Exact-delta ProRM experiments with legacy common-beta NGD and "
            "Fisher-corrected matched-KL TRPO protocols."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    check = commands.add_parser("config-check")
    check.add_argument("config")
    check.set_defaults(handler=_config_check)

    materialize = commands.add_parser("materialize")
    materialize.add_argument("config")
    materialize.add_argument("artifact_dir")
    _add_execution_options(materialize)
    materialize.set_defaults(handler=_materialize)
    materialize.add_argument("--reuse-splits-from")

    materialize_import = commands.add_parser("import-materialization")
    materialize_import.add_argument("config")
    materialize_import.add_argument("source_seed_root")
    materialize_import.add_argument("target_seed_root")
    materialize_import.add_argument("affected_stage_analysis")
    materialize_import.add_argument("--seed", type=int, required=True)
    materialize_import.set_defaults(handler=_import_materialization)

    train = commands.add_parser("train-rewards")
    train.add_argument("config")
    train.add_argument("artifact_dir")
    train.add_argument("output")
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--device", default="cuda")
    train.add_argument("--reuse-pro-from")
    train.add_argument("--reuse-mle-from")
    train.add_argument("--fisher-selection")
    train.set_defaults(handler=_train_rewards)

    crossfit = commands.add_parser("fisher-crossfit")
    crossfit.add_argument("config")
    crossfit.add_argument("artifact_dir")
    crossfit.add_argument("output")
    crossfit.add_argument("--seed", type=int, required=True)
    crossfit.add_argument("--device", default="cpu")
    crossfit.set_defaults(handler=_fisher_crossfit)

    select_fisher = commands.add_parser("select-fisher")
    select_fisher.add_argument("config")
    select_fisher.add_argument("output")
    select_fisher.add_argument("--crossfit-results", nargs="+", required=True)
    select_fisher.set_defaults(handler=_select_fisher)

    policies = commands.add_parser("export-policies")
    policies.add_argument("config")
    policies.add_argument("artifact_dir")
    policies.add_argument("reward_result")
    policies.add_argument("output_dir")
    _add_execution_options(policies)
    policies.set_defaults(handler=_export_policies)

    rollout = commands.add_parser("evaluate-rollouts")
    rollout.add_argument("config")
    rollout.add_argument("artifact_dir")
    rollout.add_argument("adapter_dir")
    rollout.add_argument("output_dir")
    _add_execution_options(rollout)
    rollout.set_defaults(handler=_evaluate_rollouts)

    seed = commands.add_parser("run-seed")
    seed.add_argument("config")
    seed.add_argument("output_dir")
    _add_execution_options(seed)
    seed.set_defaults(handler=_run_seed)

    stage = commands.add_parser("run-stage")
    stage.add_argument("config")
    stage.add_argument("seed_root")
    stage.add_argument(
        "--stage",
        required=True,
        choices=(
            "materialize",
            "fisher-crossfit",
            "reward",
            "adapters",
            "kl-calibration-policy",
            "kl-calibration-aggregate",
            "rollout-policy",
            "rollout-aggregate",
        ),
    )
    stage.add_argument("--policy-name")
    stage.add_argument("--fisher-selection")
    stage.add_argument("--reuse-mle-from")
    stage.add_argument("--reuse-splits-from")
    _add_execution_options(stage)
    stage.set_defaults(handler=_run_stage)

    policies = commands.add_parser("policy-names")
    policies.add_argument("config")
    policies.set_defaults(handler=_policy_names)

    aggregate = commands.add_parser("aggregate")
    aggregate.add_argument("config")
    aggregate.add_argument("output")
    aggregate.add_argument("--reward-results", nargs="+", required=True)
    aggregate.add_argument("--rollout-results", nargs="+", required=True)
    aggregate.set_defaults(handler=_aggregate)

    audit = commands.add_parser("audit-fisher-trpo")
    audit.add_argument("config")
    audit.add_argument("run_root")
    audit.add_argument("source_run_root")
    audit.add_argument("output")
    audit.set_defaults(handler=_audit)

    ngd = commands.add_parser("evaluate-ngd")
    ngd.add_argument("config")
    ngd.add_argument("artifact_dir")
    ngd.add_argument("reward_result")
    ngd.add_argument("output")
    ngd.add_argument("--seed", type=int, required=True)
    ngd.add_argument("--device", default="cpu")
    ngd.set_defaults(handler=_evaluate_ngd)

    ngd_aggregate = commands.add_parser("aggregate-ngd")
    ngd_aggregate.add_argument("config")
    ngd_aggregate.add_argument("output")
    ngd_aggregate.add_argument("--results", nargs="+", required=True)
    ngd_aggregate.set_defaults(handler=_aggregate_ngd)

    ngd_audit = commands.add_parser("audit-ngd")
    ngd_audit.add_argument("config")
    ngd_audit.add_argument("run_root")
    ngd_audit.add_argument("source_run_root")
    ngd_audit.add_argument("output")
    ngd_audit.set_defaults(handler=_audit_ngd)

    direct_reference = commands.add_parser("cache-direct-reference")
    direct_reference.add_argument("extension_config")
    direct_reference.add_argument("artifact_dir")
    direct_reference.add_argument("output_dir")
    direct_reference.add_argument("--seed", type=int, required=True)
    direct_reference.add_argument("--device", default="cuda")
    direct_reference.set_defaults(handler=_cache_direct_reference)

    direct_reference_import = commands.add_parser("import-direct-reference")
    direct_reference_import.add_argument("source_extension_config")
    direct_reference_import.add_argument("target_extension_config")
    direct_reference_import.add_argument("artifact_dir")
    direct_reference_import.add_argument("source_reference_dir")
    direct_reference_import.add_argument("target_reference_dir")
    direct_reference_import.add_argument("--seed", type=int, required=True)
    direct_reference_import.set_defaults(handler=_import_direct_reference)

    direct_train = commands.add_parser("train-direct-preference")
    direct_train.add_argument("extension_config")
    direct_train.add_argument("artifact_dir")
    direct_train.add_argument("reference_dir")
    direct_train.add_argument("output_dir")
    direct_train.add_argument("--seed", type=int, required=True)
    direct_train.add_argument("--beta", type=float, required=True)
    direct_train.add_argument("--method", choices=("dpo", "auxdpo"), required=True)
    direct_train.add_argument("--device", default="cuda")
    direct_train.set_defaults(handler=_train_direct)

    direct_evaluate = commands.add_parser("evaluate-direct-preference")
    direct_evaluate.add_argument("extension_config")
    direct_evaluate.add_argument("artifact_dir")
    direct_evaluate.add_argument("source_reward_result")
    direct_evaluate.add_argument("reference_dir")
    direct_evaluate.add_argument("fits_dir")
    direct_evaluate.add_argument("baseline_evaluation")
    direct_evaluate.add_argument("output")
    direct_evaluate.add_argument("--seed", type=int, required=True)
    direct_evaluate.set_defaults(handler=_evaluate_direct)

    direct_aggregate = commands.add_parser("aggregate-direct-preference")
    direct_aggregate.add_argument("extension_config")
    direct_aggregate.add_argument("output")
    direct_aggregate.add_argument("--results", nargs="+", required=True)
    direct_aggregate.set_defaults(handler=_aggregate_direct)

    direct_audit = commands.add_parser("audit-direct-preference")
    direct_audit.add_argument("extension_config")
    direct_audit.add_argument("run_root")
    direct_audit.add_argument("output")
    direct_audit.set_defaults(handler=_audit_direct)

    real_names = commands.add_parser("real-policy-names")
    real_names.set_defaults(handler=_real_policy_names)

    real_adapters = commands.add_parser("export-real-policy-adapters")
    real_adapters.add_argument("config")
    real_adapters.add_argument("artifact_dir")
    real_adapters.add_argument("reward_result")
    real_adapters.add_argument("output_dir")
    _add_execution_options(real_adapters)
    real_adapters.set_defaults(handler=_export_real_policy_adapters)

    real_rollout = commands.add_parser("rollout-real-policy")
    real_rollout.add_argument("config")
    real_rollout.add_argument("artifact_dir")
    real_rollout.add_argument("reward_result")
    real_rollout.add_argument("adapter_dir")
    real_rollout.add_argument("output_dir")
    real_rollout.add_argument("--policy-name", required=True)
    _add_execution_options(real_rollout)
    real_rollout.set_defaults(handler=_rollout_real_policy)

    real_smoke = commands.add_parser("smoke-real-policy")
    real_smoke.add_argument("config")
    real_smoke.add_argument("artifact_dir")
    real_smoke.add_argument("reward_result")
    real_smoke.add_argument("adapter_dir")
    real_smoke.add_argument("output")
    _add_execution_options(real_smoke)
    real_smoke.set_defaults(handler=_smoke_real_policy)

    real_seed = commands.add_parser("assemble-real-policy-seed")
    real_seed.add_argument("config")
    real_seed.add_argument("artifact_dir")
    real_seed.add_argument("reward_result")
    real_seed.add_argument("adapter_dir")
    real_seed.add_argument("policy_root")
    real_seed.add_argument("output")
    real_seed.add_argument("--seed", type=int, required=True)
    real_seed.set_defaults(handler=_assemble_real_policy)

    real_aggregate = commands.add_parser("aggregate-real-policy")
    real_aggregate.add_argument("config")
    real_aggregate.add_argument("output")
    real_aggregate.add_argument("--results", nargs="+", required=True)
    real_aggregate.set_defaults(handler=_aggregate_real_policy)

    real_audit = commands.add_parser("audit-real-policy")
    real_audit.add_argument("config")
    real_audit.add_argument("source_run_root")
    real_audit.add_argument("run_root")
    real_audit.add_argument("output")
    real_audit.set_defaults(handler=_audit_real_policy)

    real_extend = commands.add_parser("extend-real-policy-rollout-to-six")
    real_extend.add_argument("extension_config")
    real_extend.add_argument("artifact_dir")
    real_extend.add_argument("reward_result")
    real_extend.add_argument("adapter_dir")
    real_extend.add_argument("source_rollout_dir")
    real_extend.add_argument("output_dir")
    real_extend.add_argument("--policy-name", required=True)
    _add_execution_options(real_extend)
    real_extend.set_defaults(handler=_extend_real_policy)

    real_m6_seed = commands.add_parser("assemble-extended-real-policy-seed")
    real_m6_seed.add_argument("extension_config")
    real_m6_seed.add_argument("artifact_dir")
    real_m6_seed.add_argument("reward_result")
    real_m6_seed.add_argument("adapter_dir")
    real_m6_seed.add_argument("source_policy_root")
    real_m6_seed.add_argument("policy_root")
    real_m6_seed.add_argument("output")
    real_m6_seed.add_argument("--seed", type=int, required=True)
    real_m6_seed.set_defaults(handler=_assemble_extended_real_policy)

    real_m6_aggregate = commands.add_parser("aggregate-extended-real-policy")
    real_m6_aggregate.add_argument("extension_config")
    real_m6_aggregate.add_argument("output")
    real_m6_aggregate.add_argument("--results", nargs="+", required=True)
    real_m6_aggregate.set_defaults(handler=_aggregate_extended_real_policy)

    real_m6_audit = commands.add_parser("audit-extended-real-policy")
    real_m6_audit.add_argument("extension_config")
    real_m6_audit.add_argument("source_run_root")
    real_m6_audit.add_argument("base_real_run_root")
    real_m6_audit.add_argument("run_root")
    real_m6_audit.add_argument("output")
    real_m6_audit.set_defaults(handler=_audit_extended_real_policy)

    direct_adapter = commands.add_parser("export-direct-policy-adapter")
    direct_adapter.add_argument("config")
    direct_adapter.add_argument("artifact_dir")
    direct_adapter.add_argument("fit_dir")
    direct_adapter.add_argument("output_dir")
    direct_adapter.add_argument("--method", choices=("dpo", "auxdpo"), required=True)
    _add_execution_options(direct_adapter)
    direct_adapter.set_defaults(handler=_export_direct_policy_adapter)

    direct_rollout = commands.add_parser("rollout-direct-policy-m6")
    direct_rollout.add_argument("config")
    direct_rollout.add_argument("artifact_dir")
    direct_rollout.add_argument("fit_dir")
    direct_rollout.add_argument("adapter_dir")
    direct_rollout.add_argument("output_dir")
    direct_rollout.add_argument("--method", choices=("dpo", "auxdpo"), required=True)
    _add_execution_options(direct_rollout)
    direct_rollout.set_defaults(handler=_rollout_direct_policy)

    direct_smoke = commands.add_parser("smoke-direct-policy")
    direct_smoke.add_argument("config")
    direct_smoke.add_argument("artifact_dir")
    direct_smoke.add_argument("fit_dir")
    direct_smoke.add_argument("adapter_dir")
    direct_smoke.add_argument("output")
    direct_smoke.add_argument("--method", choices=("dpo", "auxdpo"), required=True)
    _add_execution_options(direct_smoke)
    direct_smoke.set_defaults(handler=_smoke_direct_policy)

    six_seed = commands.add_parser("assemble-six-policy-seed")
    six_seed.add_argument("config")
    six_seed.add_argument("artifact_dir")
    six_seed.add_argument("reward_result")
    six_seed.add_argument("direct_seed_root")
    six_seed.add_argument("source_evaluation")
    six_seed.add_argument("output")
    six_seed.add_argument("--seed", type=int, required=True)
    six_seed.set_defaults(handler=_assemble_six_policy)

    six_aggregate = commands.add_parser("aggregate-six-policy")
    six_aggregate.add_argument("config")
    six_aggregate.add_argument("output")
    six_aggregate.add_argument("--results", nargs="+", required=True)
    six_aggregate.set_defaults(handler=_aggregate_six_policy)

    six_audit = commands.add_parser("audit-six-policy")
    six_audit.add_argument("config")
    six_audit.add_argument("run_root")
    six_audit.add_argument("output")
    six_audit.set_defaults(handler=_audit_six_policy)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        return int(arguments.handler(arguments))
    except (ConfigError, ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
