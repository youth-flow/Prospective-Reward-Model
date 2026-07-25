from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from smart_reward.config import config_hash
from smart_reward.phase2_config import (
    PHASE2_BUDGETED_END_TO_END_BASE_CONFIG,
    PHASE2_BUDGETED_END_TO_END_CONFIG,
    PHASE2_BUDGETED_END_TO_END_EVIDENCE_ROLE,
    PHASE2_BUDGETED_END_TO_END_SEEDS,
    PHASE2_BUDGETED_END_TO_END_STAGE,
)
from smart_reward.phase2_post_recovery_control import OPTIMIZER_SCHEDULE_SHA256

ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts" / "hpc4" / "submit_phase2_budgeted_end_to_end.sh"
SBATCH = ROOT / "scripts" / "hpc4" / "phase2_budgeted_end_to_end.sbatch"

EXPECTED_EXPORT_ORDER = (
    "PATH",
    "PRORM_PROJECT_ROOT",
    "PRORM_SCRATCH_ROOT",
    "PRORM_REPO_ROOT",
    "PRORM_IMAGE",
    "PRORM_IMAGE_SHA256",
    "PRORM_HF_CACHE",
    "PRORM_HF_INVENTORY",
    "PRORM_HF_INVENTORY_SHA256",
    "PRORM_BUDGETED_OVERLAY_REL",
    "PRORM_BUDGETED_BASE_REL",
    "PRORM_BUDGETED_OVERLAY_SHA256",
    "PRORM_BUDGETED_BASE_SHA256",
    "PRORM_BUDGETED_DESIGN_SHA256",
    "PRORM_BUDGETED_BASE_CONFIG_HASH",
    "PRORM_RECOVERY_AUTHORIZATION",
    "PRORM_RECOVERY_AUTHORIZATION_SHA256",
    "PRORM_OPTIMIZER_SCHEDULE_SHA256",
    "PRORM_BUDGETED_FROZEN_BETA",
    "PRORM_BUDGETED_FREEZE_EVIDENCE",
    "PRORM_BUDGETED_FREEZE_EVIDENCE_SHA256",
    "PRORM_GIT_COMMIT",
)


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _assignment(source: str, name: str) -> str:
    match = re.search(rf"^{re.escape(name)}=\"([^\n\"]+)\"$", source, flags=re.MULTILINE)
    assert match is not None
    return match.group(1)


def _export_keys(template: str) -> tuple[str, ...]:
    return tuple(re.findall(r"(?:^|,)([A-Z][A-Z0-9_]*)=", template))


def _validator_source() -> str:
    source = _source(WRAPPER)
    begin = source.index("# BEGIN BUDGETED_DEEP_VALIDATOR")
    end = source.index("# END BUDGETED_DEEP_VALIDATOR")
    return source[begin:end]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_canonical_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _execute_deep_validator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    wrong_stage: bool = False,
    wrong_freeze: bool = False,
    wrong_receipt: bool = False,
) -> tuple[BaseException | None, tuple[str, ...]]:
    import smart_reward.phase2_config as phase2_config
    import smart_reward.phase2_pilot_aggregate as pilot_aggregate
    import smart_reward.phase2_post_recovery_control as post_recovery_control
    import smart_reward.phase2_training as phase2_training

    overlay = tmp_path / Path(PHASE2_BUDGETED_END_TO_END_CONFIG).name
    base = tmp_path / Path(PHASE2_BUDGETED_END_TO_END_BASE_CONFIG).name
    receipt_path = tmp_path / "materialized.json"
    authorization = tmp_path / "authorization.json"
    freeze = tmp_path / "accepted-freeze.json"
    overlay.write_text("overlay\n", encoding="utf-8", newline="\n")
    base.write_text("base\n", encoding="utf-8", newline="\n")
    authorization.write_text("{}\n", encoding="utf-8", newline="\n")
    authorization_sha256 = _sha256(authorization)

    freeze_value: dict[str, Any] = {
        "schema_version": "common-beta-pilot-selection-aggregate/v3",
        "pilot_phase": "freeze",
        "formal_eligibility": False,
        "supports_formal_claim": False,
        "evidence_role": "target_free_design_selection_only",
        "information_boundary": {
            "validation_metrics_read": False,
            "test_metrics_read": False,
            "learner_ordering_read": False,
            "downstream_utility_read": False,
        },
        "horizon": {
            "all_seed_length_gates_passed": True,
        },
        "selection": {
            "schema_version": "pilot-freeze-selection/v1",
            "frozen_global_beta": 2.5,
            "next_global_beta": 2.5,
            "selection_accepted": True,
            "accepted_for_confirmatory_identity": True,
            "all_seeds_and_arms_used_same_beta": True,
            "all_pre_oracle_safety_gates_passed": True,
            "all_length_gates_passed": True,
            "all_non_length_safety_gates_passed": True,
            "next_action": "freeze_confirmatory_design_identity",
        },
    }
    if wrong_freeze:
        freeze_value["selection"]["selection_accepted"] = False
    _write_canonical_json(freeze, freeze_value)
    freeze_sha256 = _sha256(freeze)

    base_config = {
        "schema_version": "test-base/v1",
        "run": {"seeds": list(PHASE2_BUDGETED_END_TO_END_SEEDS)},
    }
    base_hash = config_hash(base_config)
    design_sha256 = "d" * 64
    config: dict[str, Any] = {
        "design": {
            "stage": "confirmatory" if wrong_stage else PHASE2_BUDGETED_END_TO_END_STAGE,
            "pilot_phase": None,
            "formal_eligibility": False,
            "evidence_role": PHASE2_BUDGETED_END_TO_END_EVIDENCE_ROLE,
            "source_config": PHASE2_BUDGETED_END_TO_END_BASE_CONFIG,
            "source_config_hash": base_hash,
        },
        "run": {
            "seeds": list(PHASE2_BUDGETED_END_TO_END_SEEDS),
            "confirmatory": False,
            "formal_eligibility": False,
            "excluded_from_confirmatory_evidence": True,
        },
        "objective": {
            "common_beta": {
                "frozen_global_beta": 2.5,
                "beta_source_aggregate_sha256": freeze_sha256,
            }
        },
        "evaluation": {
            "max_length": {
                "parent_pilot_aggregate_sha256": freeze_sha256,
            }
        },
    }
    bundle = SimpleNamespace(
        config=config,
        base_config=base_config,
        base_config_path=base,
        design_identity=design_sha256,
    )
    receipt = {
        "schema_version": "budgeted-end-to-end-materialization-receipt/v1",
        "stage": "pilot" if wrong_receipt else PHASE2_BUDGETED_END_TO_END_STAGE,
        "formal_claim_eligible": False,
        "git_commit_used_for_source": "a" * 40,
        "base_relative_path": PHASE2_BUDGETED_END_TO_END_BASE_CONFIG,
        "base_file_sha256": _sha256(base),
        "overlay_relative_path": PHASE2_BUDGETED_END_TO_END_CONFIG,
        "overlay_file_sha256": _sha256(overlay),
        "phase2_design_sha256": design_sha256,
        "accepted_freeze_aggregate_sha256": freeze_sha256,
        "authorization_sha256": authorization_sha256,
    }
    _write_canonical_json(receipt_path, receipt)

    protocol = SimpleNamespace(
        mode="adopted",
        schedule_sha256=OPTIMIZER_SCHEDULE_SHA256,
        source_recovery_authorization_sha256=authorization_sha256,
        to_dict=lambda: {"scope": "every_phase2_first_order_convergence_trainer"},
    )
    settings = SimpleNamespace(
        stage=PHASE2_BUDGETED_END_TO_END_STAGE,
        formal_eligibility=False,
        seeds=PHASE2_BUDGETED_END_TO_END_SEEDS,
        convergence=SimpleNamespace(
            max_steps=12760,
            check_interval=20,
            consecutive_checks=3,
            optimizer_protocol=protocol,
        ),
    )
    monkeypatch.setattr(
        phase2_config,
        "load_phase2_config_bundle",
        lambda _path: bundle,
    )
    monkeypatch.setattr(
        post_recovery_control,
        "verify_recovery_authorization_config_binding",
        lambda *_args, **_kwargs: {
            "authorization_sha256": authorization_sha256,
            "phase2_design_sha256": design_sha256,
            "base_config_hash": base_hash,
            "optimizer_schedule_sha256": OPTIMIZER_SCHEDULE_SHA256,
            "stage": PHASE2_BUDGETED_END_TO_END_STAGE,
            "pilot_phase": None,
        },
    )
    monkeypatch.setattr(
        post_recovery_control,
        "verify_post_recovery_aggregate_success_receipt",
        lambda _path: {
            "aggregate_sha256": freeze_sha256,
            "pilot_phase": "freeze",
        },
    )
    monkeypatch.setattr(
        pilot_aggregate,
        "verify_beta_source_aggregate",
        lambda _config, _path: {
            "sha256": freeze_sha256,
            "accepted_beta": 2.5,
        },
    )
    monkeypatch.setattr(
        pilot_aggregate,
        "verify_horizon_parent_aggregate",
        lambda _config, _path: {
            "sha256": freeze_sha256,
            "source_pilot_phase": "freeze",
        },
    )
    monkeypatch.setattr(
        phase2_training,
        "compile_phase2_training_settings",
        lambda _bundle: settings,
    )

    arguments = [
        "budgeted-deep-validator",
        os.fspath(overlay),
        os.fspath(base),
        os.fspath(receipt_path),
        os.fspath(authorization),
        authorization_sha256,
        os.fspath(freeze),
        freeze_sha256,
        PHASE2_BUDGETED_END_TO_END_CONFIG,
        PHASE2_BUDGETED_END_TO_END_BASE_CONFIG,
        "configs/.common_beta_post_recovery_budgeted_end_to_end.materialized.json",
    ]
    previous_argv = sys.argv
    output = io.StringIO()
    failure: BaseException | None = None
    try:
        sys.argv = arguments
        with contextlib.redirect_stdout(output):
            exec(compile(_validator_source(), "<budgeted-deep-validator>", "exec"), {})
    except BaseException as error:  # the assertion examines the exact fail-closed path
        failure = error
    finally:
        sys.argv = previous_argv
    return failure, tuple(output.getvalue().splitlines())


def test_wrapper_and_sbatch_share_one_canonical_export_field_order() -> None:
    wrapper_template = _assignment(_source(WRAPPER), "export_spec")
    sbatch_template = _assignment(_source(SBATCH), "runtime_export_spec")

    assert _export_keys(wrapper_template) == EXPECTED_EXPORT_ORDER
    assert _export_keys(sbatch_template) == EXPECTED_EXPORT_ORDER
    assert "PRORM_BUDGETED_EXPORT_SPEC_SHA256" not in wrapper_template
    assert "PRORM_BUDGETED_EXPORT_SPEC_SHA256" not in sbatch_template


def test_wrapper_locks_exact_budgeted_files_and_deep_production_gates() -> None:
    source = _source(WRAPPER)
    for required in (
        'readonly OVERLAY_RELATIVE="configs/common_beta_post_recovery_budgeted_end_to_end.yaml"',
        (
            'readonly BASE_RELATIVE="configs/'
            'common_beta_post_recovery_budgeted_end_to_end_base.yaml"'
        ),
        (
            'readonly MATERIALIZATION_RECEIPT_RELATIVE="configs/'
            '.common_beta_post_recovery_budgeted_end_to_end.materialized.json"'
        ),
        'readonly AUTHORIZATION_RELATIVE="runs/phase2-recovery-pilot/'
        'recovery-success-authorization.json"',
        "verify_recovery_authorization_config_binding(",
        "verify_post_recovery_aggregate_success_receipt(",
        "verify_beta_source_aggregate(config, freeze)",
        "verify_horizon_parent_aggregate(config, freeze)",
        "expected_stage=PHASE2_BUDGETED_END_TO_END_STAGE",
        "settings.convergence.max_steps != 12760",
        "settings.convergence.check_interval != 20",
        "settings.convergence.consecutive_checks != 3",
        "submit_phase2_budgeted_end_to_end_once.py",
    ):
        assert required in source
    safety_start = source.index("for name in \\\n  project_root")
    export_safety_loop = source[safety_start : source.index("\n\ncritical_paths=(", safety_start)]
    assert "walltime" not in export_safety_loop
    assert re.search(
        r"\[\[ \"\$\{walltime\}\" =~ .+\]\] \\\n"
        r"  \|\| die \"walltime must be HH:MM:SS or D-HH:MM:SS\"",
        source,
    )


@pytest.mark.parametrize("walltime", ("12:00:00", "2-00:00:00"))
def test_legal_walltimes_pass_the_dedicated_nonexport_validation(walltime: str) -> None:
    source = _source(WRAPPER)
    match = re.search(
        r'\[\[ "\$\{walltime\}" =~ \^\(([^\\\n]+)\)\$ \]\]',
        source,
    )
    assert match is not None
    assert re.fullmatch(match.group(1), walltime)
    safety_start = source.index("for name in \\\n  project_root")
    export_safety_loop = source[safety_start : source.index("\n\ncritical_paths=(", safety_start)]
    assert "walltime" not in export_safety_loop


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ({"wrong_stage": True}, "budgeted_end_to_end"),
        ({"wrong_freeze": True}, "fully accepted"),
        ({"wrong_receipt": True}, "materialization receipt"),
    ),
)
def test_executable_deep_validator_rejects_wrong_identity_layers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: dict[str, bool],
    message: str,
) -> None:
    failure, output = _execute_deep_validator(tmp_path, monkeypatch, **mutation)

    assert failure is not None
    assert message in str(failure)
    assert output == ()


def test_executable_deep_validator_emits_only_locked_submission_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure, output = _execute_deep_validator(tmp_path, monkeypatch)

    assert failure is None
    assert output == (
        "d" * 64,
        config_hash(
            {
                "schema_version": "test-base/v1",
                "run": {"seeds": list(PHASE2_BUDGETED_END_TO_END_SEEDS)},
            }
        ),
        OPTIMIZER_SCHEDULE_SHA256,
        "2.5",
        "a" * 40,
    )


def test_executable_wrapper_rejects_a_dirty_checkout_before_submission(
    tmp_path: Path,
) -> None:
    bash = shutil.which("bash")
    git = shutil.which("git")
    if bash is None or git is None:
        pytest.skip("bash and git are required for the executable shell check")
    repo = tmp_path / "repo"
    script = repo / "scripts" / "hpc4" / WRAPPER.name
    script.parent.mkdir(parents=True)
    shutil.copyfile(WRAPPER, script)
    subprocess.run([git, "-C", os.fspath(repo), "init"], check=True, capture_output=True)
    subprocess.run(
        [git, "-C", os.fspath(repo), "config", "user.name", "Budgeted Wrapper Test"],
        check=True,
    )
    subprocess.run(
        [
            git,
            "-C",
            os.fspath(repo),
            "config",
            "user.email",
            "budgeted-wrapper@example.invalid",
        ],
        check=True,
    )
    subprocess.run(
        [git, "-C", os.fspath(repo), "add", "--", "scripts"],
        check=True,
    )
    subprocess.run(
        [git, "-C", os.fspath(repo), "commit", "-m", "wrapper"],
        check=True,
        capture_output=True,
    )
    (repo / "dirty.txt").write_text("untracked\n", encoding="utf-8")
    environment = {
        key: value
        for key, value in os.environ.items()
        if not (
            key.startswith(("APPTAINER", "SINGULARITY", "SBATCH_", "PRORM_BUDGETED_"))
            or key
            in {
                "PRORM_RECOVERY_AUTHORIZATION",
                "PRORM_OPTIMIZER_SCHEDULE_SHA256",
                "PRORM_GIT_COMMIT",
                "PRORM_HF_INVENTORY",
                "PRORM_HF_INVENTORY_SHA256",
                "PRORM_REPO_ROOT",
            }
        )
    }
    environment.update(
        {
            "PRORM_PROJECT_ROOT": "/not-reached/project",
            "PRORM_SCRATCH_ROOT": "/not-reached/scratch",
            "PRORM_IMAGE": "images/prorm.sif",
            "PRORM_IMAGE_SHA256": "a" * 64,
            "PRORM_HF_CACHE": "hf-cache",
        }
    )

    completed = subprocess.run(
        [
            bash,
            os.fspath(script),
            "overlay.yaml",
            "authorization.json",
            "accepted-freeze.json",
            "12:00:00",
        ],
        cwd=repo,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "clean committed worktree" in completed.stderr
    assert "sbatch" not in completed.stderr
