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
from .rollout import evaluate_policy_rollouts
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


def _train_rewards(arguments: argparse.Namespace) -> int:
    run_exact_reward_comparison(
        load_config(arguments.config),
        arguments.artifact_dir,
        arguments.output,
        seed=arguments.seed,
        device=arguments.device,
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
    if root.exists():
        raise FileExistsError(f"refusing to overwrite seed directory: {root}")
    artifact = root / "artifact"
    reward_result = root / "reward_result.json"
    adapters = root / "adapters"
    rollout = root / "policy_utility"
    materialize_exact_delta(
        config,
        seed=arguments.seed,
        artifact_dir=artifact,
        device=arguments.device,
        local_files_only=not arguments.allow_download,
    )
    run_exact_reward_comparison(
        config,
        artifact,
        reward_result,
        seed=arguments.seed,
        device=arguments.device,
    )
    export_exact_ngd_adapters(
        config,
        artifact,
        reward_result,
        adapters,
        seed=arguments.seed,
        device=arguments.device,
        local_files_only=not arguments.allow_download,
    )
    evaluate_policy_rollouts(
        config,
        artifact,
        adapters,
        rollout,
        seed=arguments.seed,
        device=arguments.device,
        local_files_only=not arguments.allow_download,
    )
    return 0


def _aggregate(arguments: argparse.Namespace) -> int:
    payload = aggregate_results(
        load_config(arguments.config),
        arguments.reward_results,
        arguments.rollout_results,
    )
    _write_json(Path(arguments.output), payload)
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

    train = commands.add_parser("train-rewards")
    train.add_argument("config")
    train.add_argument("artifact_dir")
    train.add_argument("output")
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--device", default="cuda")
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
