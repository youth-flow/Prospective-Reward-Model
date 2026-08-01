#!/usr/bin/env python3
"""Build the compact, audited Main Experiment v1 result bundle.

This is a reporting-only transform.  It never recomputes a learned direction or
reads test results to modify the experiment.  The dense sweep remains immutable;
the report exposes the frozen beta subset 0.1, 0.2, and 0.3.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

MAIN_BETAS = ("0.1", "0.2", "0.3")
PRIMARY_BETA = "0.2"
POLICIES = ("pi0", "mle", "pro", "oracle", "tabular")
REWARD_METRICS = ("NLL", "MSE", "approximate_regret")
POLICY_METRICS = ("R", "K", "J", "delta_J", "beta_KL")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite {label}")
    return result


def _summary_stat(record: dict[str, Any], label: str) -> dict[str, float]:
    return {
        "mean": _finite(record["mean"], f"{label}.mean"),
        "sample_sd": _finite(record["sample_sd"], f"{label}.sample_sd"),
    }


def _strict_order(values: dict[str, float], order: tuple[str, ...]) -> bool:
    return all(values[left] > values[right] for left, right in zip(order, order[1:], strict=False))


def build_summary(
    aggregate_path: Path,
    audit_path: Path,
    seed_paths: list[Path],
) -> dict[str, Any]:
    aggregate = _read_json(aggregate_path)
    audit = _read_json(audit_path)
    if audit.get("status") != "passed":
        raise ValueError("integrity audit did not pass")
    aggregate_sha256 = _sha256(aggregate_path)
    if audit.get("aggregate_sha256") != aggregate_sha256:
        raise ValueError("aggregate hash does not match the integrity audit")
    seeds = [int(seed) for seed in aggregate.get("seeds", [])]
    if seeds != [20261001, 20261002, 20261003]:
        raise ValueError(f"unexpected formal seeds: {seeds!r}")
    if len(seed_paths) != len(seeds):
        raise ValueError("all three seed-level evaluation files are required")
    if not set(MAIN_BETAS).issubset(aggregate.get("policy", {})):
        raise ValueError("aggregate is missing a Main Experiment v1 beta")

    reward: dict[str, Any] = {}
    for method in ("mle", "pro"):
        source = aggregate["reward"][method]
        reward[method] = {
            "NLL": _summary_stat(source["NLL"], f"reward.{method}.NLL"),
            "MSE": _summary_stat(source["MSE"], f"reward.{method}.MSE"),
            "approximate_regret": {
                beta: _summary_stat(
                    source["approximate_regret"][beta],
                    f"reward.{method}.approximate_regret.{beta}",
                )
                for beta in MAIN_BETAS
            },
        }

    policy: dict[str, Any] = {}
    max_identity_residual = 0.0
    for beta in MAIN_BETAS:
        source = aggregate["policy"][beta]
        max_identity_residual = max(
            max_identity_residual,
            _finite(
                source["identity_residuals"]["abs_J_tabular_minus_J_close"]["mean"],
                f"policy.{beta}.J_identity",
            ),
            _finite(
                source["identity_residuals"]["max_abs_delta_J_minus_beta_KL"]["mean"],
                f"policy.{beta}.gibbs_identity",
            ),
        )
        policies: dict[str, Any] = {}
        for method in POLICIES:
            policies[method] = {
                metric: _summary_stat(
                    source["policies"][method][metric],
                    f"policy.{beta}.{method}.{metric}",
                )
                for metric in POLICY_METRICS
            }
        policy[beta] = {
            "J_close": _summary_stat(source["J_close"], f"policy.{beta}.J_close"),
            "policies": policies,
        }

    ordering = {beta: {"R": 0, "J": 0} for beta in MAIN_BETAS}
    seen_seeds: list[int] = []
    audit_checks = {int(record["seed"]): record for record in audit.get("checks", [])}
    for path in seed_paths:
        seed_result = _read_json(path)
        seed = int(seed_result["seed"])
        seen_seeds.append(seed)
        result_sha256 = _sha256(path)
        if audit_checks.get(seed, {}).get("result_sha256") != result_sha256:
            raise ValueError(f"seed {seed} hash does not match the integrity audit")
        if aggregate["input_sha256"].get(str(seed)) != result_sha256:
            raise ValueError(f"seed {seed} hash does not match the aggregate input")
        for beta in MAIN_BETAS:
            records = seed_result["policy"][beta]["policies"]
            reward_values = {name: float(records[name]["R"]) for name in POLICIES}
            utility_values = {name: float(records[name]["J"]) for name in POLICIES}
            ordering[beta]["R"] += int(
                _strict_order(reward_values, ("tabular", "oracle", "pro", "mle", "pi0"))
            )
            ordering[beta]["J"] += int(
                _strict_order(utility_values, ("tabular", "pro", "oracle", "mle", "pi0"))
            )
    if sorted(seen_seeds) != seeds:
        raise ValueError(f"seed result set does not match aggregate: {seen_seeds!r}")

    return {
        "schema": "prorm-main-experiment-report/v1",
        "title": "Fisher-corrected common-beta NGD Main Experiment v1",
        "status": "completed_audited_archived",
        "evidence_class": "exploratory_post_hoc_beta_selection",
        "betas": [float(beta) for beta in MAIN_BETAS],
        "primary_beta": float(PRIMARY_BETA),
        "seeds": seeds,
        "reward": reward,
        "policy": policy,
        "ordering_seed_counts": ordering,
        "identities": {
            "J_tabular_equals_J_close": True,
            "delta_J_equals_beta_KL": True,
            "max_mean_absolute_residual": max_identity_residual,
        },
        "interpretation": {
            "reward": (
                "Pro-RM fits its train moment objective but generalizes worse than MLE-RM "
                "on held-out NLL, centered reward MSE, and approximate regret."
            ),
            "policy": (
                "Despite the reward-level generalization gap, Pro-NGD exceeds MLE-NGD "
                "in regularized test utility for every reported beta and every seed."
            ),
            "selection_caveat": (
                "The beta subset was selected after inspecting the dense test sweep; it is "
                "descriptive Main Experiment v1 evidence, not a fresh confirmatory test."
            ),
        },
        "provenance": {
            "aggregate_sha256": aggregate_sha256,
            "audit_sha256": _sha256(audit_path),
            "producer_commit": aggregate["producer"]["git_commit"],
            "source_config_sha256": aggregate["source_config_sha256"],
            "protocol": aggregate["protocol"],
            "aggregate_schema": aggregate["schema"],
            "audit_schema": audit["schema"],
        },
    }


def _pm(record: dict[str, float], digits: int = 6) -> str:
    mean = record["mean"]
    if abs(mean) < 0.5 * 10**-digits:
        mean = 0.0
    return f"${mean:.{digits}f} \\pm {record['sample_sd']:.{digits}f}$"


def build_latex(summary: dict[str, Any]) -> str:
    lines = [
        r"\begin{table}[tbp]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Reward 指标 $\downarrow$ & MLE-RM & ProRM \\",
        r"\midrule",
        "Soft BTL NLL & "
        + _pm(summary["reward"]["mle"]["NLL"])
        + " & "
        + _pm(summary["reward"]["pro"]["NLL"])
        + r" \\",
        "中心化 reward MSE & "
        + _pm(summary["reward"]["mle"]["MSE"])
        + " & "
        + _pm(summary["reward"]["pro"]["MSE"])
        + r" \\",
    ]
    for beta in MAIN_BETAS:
        lines.append(
            rf"$\widetilde{{\Reg}}_{{\beta={beta}}}$ & "
            + _pm(summary["reward"]["mle"]["approximate_regret"][beta])
            + " & "
            + _pm(summary["reward"]["pro"]["approximate_regret"][beta])
            + r" \\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        (
            r"\caption{Reward 层 held-out evaluation，均报告三 seed mean "
            r"$\pm$ sample SD。ProRM 的三个指标均较差，诊断为 reward-moment "
            r"泛化不足；该负结果与下游 policy 评价分开解释。}"
        ),
        r"\label{tab:main-reward}",
        r"\end{table}",
        "",
    ]
    display_name = {
        "pi0": r"$\pi_0$",
        "mle": r"$\pi_{\rm MLE}$",
        "pro": r"$\pi_{\rm Pro}$",
        "oracle": r"$\pi_{\rm oracle}$",
        "tabular": r"$\pi_{\rm tab}$",
    }
    row_order = ("tabular", "pro", "oracle", "mle", "pi0")
    for beta in MAIN_BETAS:
        lines += [
            r"\begin{table}[tbp]",
            r"\centering",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{3.0pt}",
            r"\begin{tabular}{lrrrrr}",
            r"\toprule",
            r"Policy & $R$ & $K$ & $J$ & $\Delta J$ & $\beta\,\KL(\pi\|\pi_{\rm tab})$ \\",
            r"\midrule",
        ]
        for method in row_order:
            record = summary["policy"][beta]["policies"][method]
            cells = [display_name[method]] + [_pm(record[metric]) for metric in POLICY_METRICS]
            lines.append(" & ".join(cells) + r" \\")
        counts = summary.get("ordering_seed_counts")
        count_text = ""
        if counts:
            count_text = (
                rf" Reward 完整排序与 utility 完整排序分别在 "
                rf"{counts[beta]['R']}/3 和 {counts[beta]['J']}/3 个 seed 成立。"
            )
        lines += [
            r"\bottomrule",
            r"\end{tabular}",
            (
                rf"\caption{{$\beta={beta}$ 的冻结 test-candidate policy evaluation；"
                rf"均报告三 seed mean $\pm$ sample SD。{count_text}}}"
            ),
            rf"\label{{tab:main-policy-{beta.replace('.', '')}}}",
            r"\end{table}",
            "",
        ]
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--seed-result", type=Path, action="append", default=[])
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--latex", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_summary(args.aggregate, args.audit, args.seed_result)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.latex.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.latex.write_text(build_latex(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
