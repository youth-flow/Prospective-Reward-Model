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
        "materialize",
        "train-rewards",
        "export-policies",
        "evaluate-rollouts",
        "run-seed",
        "aggregate",
    }


def test_config_check_succeeds(capsys) -> None:
    assert main(["config-check", str(ROOT / "configs" / "main.yaml")]) == 0
    assert "config_sha256" in capsys.readouterr().out
