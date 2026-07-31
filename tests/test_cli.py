from pathlib import Path

from smart_reward.cli import build_parser, main

ROOT = Path(__file__).parents[1]


def test_cli_only_exposes_current_workflow() -> None:
    parser = build_parser()
    help_text = parser.format_help()
    assert "Exact-delta" in help_text
    subcommands = parser._subparsers._group_actions[0].choices
    assert set(subcommands) == {
        "config-check",
        "import-materialization",
        "materialize",
        "fisher-crossfit",
        "select-fisher",
        "train-rewards",
        "export-policies",
        "evaluate-rollouts",
        "run-seed",
        "run-stage",
        "policy-names",
        "aggregate",
        "audit-fisher-trpo",
    }


def test_config_check_succeeds(capsys) -> None:
    assert main(["config-check", str(ROOT / "configs" / "main.yaml")]) == 0
    assert "config_sha256" in capsys.readouterr().out


def test_policy_names_exposes_reference_and_nine_updates(capsys) -> None:
    assert main(["policy-names", str(ROOT / "configs" / "main.yaml")]) == 0
    names = capsys.readouterr().out.splitlines()
    assert names[0] == "pi0"
    assert len(names) == 10
    assert set(names[1:]) == {
        f"{method}__beta_{beta}"
        for method in ("mle_rm", "pro_rm", "oracle")
        for beta in ("1", "2", "4")
    }
