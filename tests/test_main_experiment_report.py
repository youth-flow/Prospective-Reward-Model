from __future__ import annotations

import json
from pathlib import Path

from smart_reward.main_experiment_report import build_latex, build_summary

ROOT = Path(__file__).parents[1]
RESULT = ROOT / "results" / "main_experiment_v1"
EVIDENCE = RESULT / "evidence"
REPORT = ROOT / "reports" / "main_experiment_v1"


def _seed_results() -> list[Path]:
    return [EVIDENCE / f"seed-{seed}-evaluation.json" for seed in (20261001, 20261002, 20261003)]


def test_main_experiment_summary_is_synced_with_audited_evidence() -> None:
    expected = json.loads((RESULT / "summary.json").read_text(encoding="utf-8"))
    actual = build_summary(
        EVIDENCE / "aggregate.json",
        EVIDENCE / "integrity-audit.json",
        _seed_results(),
    )
    assert actual == expected
    assert actual["status"] == "completed_audited_archived"
    assert actual["evidence_class"] == "exploratory_post_hoc_beta_selection"
    assert actual["betas"] == [0.1, 0.2, 0.3]
    assert actual["primary_beta"] == 0.2
    assert actual["ordering_seed_counts"] == {
        "0.1": {"J": 2, "R": 3},
        "0.2": {"J": 3, "R": 3},
        "0.3": {"J": 3, "R": 3},
    }


def test_main_experiment_latex_tables_are_generated_not_hand_edited() -> None:
    summary = json.loads((RESULT / "summary.json").read_text(encoding="utf-8"))
    expected = (REPORT / "ProRM_main_results.tex").read_text(encoding="utf-8")
    assert build_latex(summary) == expected
