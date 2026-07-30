"""Command-line entry points for the single ProRM experiment workflow."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .config import ConfigError, config_hash, load_config
from .exact_phase import materialize_exact_delta
from .exact_policy import export_exact_ngd_adapters
from .exact_run import run_exact_reward_comparison
from .pipeline import (
    import_materialization_stage,
    run_adapter_stage,
    run_materialization_stage,
    run_policy_rollout_stage,
    run_reward_stage,
    run_rollout_aggregate_stage,
)
from .rollout import evaluate_policy_rollouts, policy_instance_names
from .statistics import aggregate_results


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
    run_exact_reward_comparison(
        load_config(arguments.config),
        arguments.artifact_dir,
        arguments.output,
        seed=arguments.seed,
        device=arguments.device,
        reuse_pro_from=arguments.reuse_pro_from,
    )
    return 0


def _export_policies(arguments: argparse.Namespace) -> int:
    export_exact_ngd_adapters(
        load_config(arguments.config),
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
        )
    elif arguments.stage == "reward":
        run_reward_stage(
            config,
            arguments.seed_root,
            **common,
            device=arguments.device,
        )
    elif arguments.stage == "adapters":
        run_adapter_stage(
            config,
            arguments.seed_root,
            **common,
            device=arguments.device,
            local_files_only=not arguments.allow_download,
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


def _add_execution_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-download", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prorm",
        description="Exact-delta MLE-RM/Pro-RM training and common-beta NGD evaluation.",
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
    train.set_defaults(handler=_train_rewards)

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
        choices=("materialize", "reward", "adapters", "rollout-policy", "rollout-aggregate"),
    )
    stage.add_argument("--policy-name")
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
