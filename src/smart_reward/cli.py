"""Command-line entry points for the single ProRM experiment workflow."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .audit import audit_fisher_trpo_run
from .config import TRPO_PROTOCOL, ConfigError, config_hash, load_config
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
