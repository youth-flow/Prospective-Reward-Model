from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest

from smart_reward.config import config_hash, load_config
from smart_reward.phase2_config import (
    PHASE2_CONFIRMATORY_SEEDS,
    validate_phase2_config,
)
from smart_reward.phase2_post_recovery_aggregate import (
    write_phase2_post_recovery_aggregate,
)

ROOT = Path(__file__).resolve().parents[1]
MATERIALIZER_PATH = ROOT / "scripts" / "hpc4" / "materialize_phase2_post_recovery_stage.py"


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _materializer():
    return _module(MATERIALIZER_PATH, "_post_recovery_stage_materializer")


def _post_recovery_config() -> dict[str, object]:
    helpers = _module(
        ROOT / "tests" / "test_phase2_aggregate.py",
        "_post_recovery_stage_config_helpers",
    )
    return helpers._post_recovery_config()


def test_horizon_escalation_materializes_a_new_bound_base_identity() -> None:
    materializer = _materializer()
    source = _post_recovery_config()
    base = load_config(ROOT / "configs" / "common_beta_pilot_base.yaml")
    candidate = materializer._calibration_projection(
        source,
        next_tokens=512,
        next_index=1,
        parent_sha256="a" * 64,
    )
    candidate_base = copy.deepcopy(base)
    candidate_base["run"]["name"] = "common-beta-post-recovery-calibration-horizon-1"
    candidate_base["policy"]["max_response_tokens"] = 512
    candidate["design"].update(
        {
            "source_config": ("configs/common_beta_post_recovery_calibration_horizon_1_base.yaml"),
            "source_config_hash": config_hash(candidate_base),
        }
    )

    validated = validate_phase2_config(candidate, base_config=candidate_base)

    assert validated["policy"]["max_response_tokens"] == 512
    assert validated["evaluation"]["max_length"]["horizon_grid_index"] == 1
    assert validated["evaluation"]["max_length"]["parent_pilot_aggregate_sha256"] == ("a" * 64)
    assert validated["objective"]["common_beta"]["frozen_global_beta"] is None


def test_freeze_length_failure_can_issue_new_horizon_calibration_identity() -> None:
    materializer = _materializer()
    calibration = _post_recovery_config()
    base = load_config(ROOT / "configs" / "common_beta_pilot_base.yaml")
    freeze = materializer._freeze_projection(
        calibration,
        beta=2.0,
        beta_source_sha256="b" * 64,
        horizon_parent_sha256="a" * 64,
        beta_grid_index=0,
    )
    candidate = materializer._calibration_projection(
        freeze,
        next_tokens=512,
        next_index=1,
        parent_sha256="c" * 64,
    )
    candidate_base = copy.deepcopy(base)
    candidate_base["run"]["name"] = "common-beta-post-recovery-calibration-horizon-1"
    candidate_base["policy"]["max_response_tokens"] = 512
    candidate["design"].update(
        {
            "source_config": ("configs/common_beta_post_recovery_calibration_horizon_1_base.yaml"),
            "source_config_hash": config_hash(candidate_base),
        }
    )

    validated = validate_phase2_config(candidate, base_config=candidate_base)

    assert validated["design"]["pilot_phase"] == "calibration"
    assert validated["evaluation"]["max_length"]["horizon_grid_index"] == 1
    assert validated["evaluation"]["max_length"]["parent_pilot_aggregate_sha256"] == ("c" * 64)
    assert validated["objective"]["common_beta"]["frozen_global_beta"] is None
    assert validated["objective"]["common_beta"]["beta_source_aggregate_sha256"] is None


@pytest.mark.parametrize(
    ("phase", "index", "expected"),
    [
        (
            "calibration",
            0,
            Path("configs/common_beta_post_recovery_calibration.yaml"),
        ),
        (
            "calibration",
            4,
            Path("configs/common_beta_post_recovery_calibration_horizon_4.yaml"),
        ),
        (
            "freeze",
            0,
            Path("configs/common_beta_post_recovery_freeze.yaml"),
        ),
        (
            "freeze",
            7,
            Path("configs/common_beta_post_recovery_freeze_retry_7.yaml"),
        ),
    ],
)
def test_materializer_requires_the_exact_semantic_source_overlay(
    phase: str,
    index: int,
    expected: Path,
) -> None:
    materializer = _materializer()
    source = _post_recovery_config()
    source["design"]["pilot_phase"] = phase
    source["evaluation"]["max_length"]["horizon_grid_index"] = (
        index if phase == "calibration" else 0
    )
    aggregate = {"selection": {"beta_grid_index": index}}

    assert materializer._source_semantic_overlay_relative(source, aggregate) == expected


def test_materializer_rejects_forged_semantic_source_index() -> None:
    materializer = _materializer()
    source = _post_recovery_config()
    source["evaluation"]["max_length"]["horizon_grid_index"] = True

    with pytest.raises(ValueError, match="horizon grid index"):
        materializer._source_semantic_overlay_relative(source, {"selection": {}})


@pytest.mark.parametrize("grid_index", [0, 1, 3])
def test_freeze_projection_supports_only_sequential_identity_inputs(
    grid_index: int,
) -> None:
    materializer = _materializer()
    source = _post_recovery_config()
    base = load_config(ROOT / "configs" / "common_beta_pilot_base.yaml")
    beta = 2.0 * (2.0**grid_index)
    candidate = materializer._freeze_projection(
        source,
        beta=beta,
        beta_source_sha256="b" * 64,
        horizon_parent_sha256="a" * 64,
        beta_grid_index=grid_index,
    )

    validated = validate_phase2_config(candidate, base_config=base)

    assert validated["design"]["pilot_phase"] == "freeze"
    assert validated["objective"]["common_beta"]["frozen_global_beta"] == beta
    assert validated["objective"]["common_beta"]["beta_source_aggregate_sha256"] == ("b" * 64)
    assert validated["evaluation"]["max_length"]["parent_pilot_aggregate_sha256"] == ("a" * 64)


def test_confirmatory_projection_is_exact_30_and_keeps_post_recovery_schedule() -> None:
    materializer = _materializer()
    source = _post_recovery_config()
    pilot_base = load_config(ROOT / "configs" / "common_beta_pilot_base.yaml")
    freeze = materializer._freeze_projection(
        source,
        beta=4.0,
        beta_source_sha256="a" * 64,
        horizon_parent_sha256="a" * 64,
        beta_grid_index=1,
    )
    candidate, base = materializer._confirmatory_projection(
        freeze,
        pilot_base,
        freeze_sha256="c" * 64,
        frozen_beta=4.0,
    )

    validated = validate_phase2_config(candidate, base_config=base)

    assert tuple(validated["run"]["seeds"]) == PHASE2_CONFIRMATORY_SEEDS
    assert validated["design"]["stage"] == "confirmatory"
    assert validated["design"]["pilot_phase"] is None
    assert validated["objective"]["common_beta"]["frozen_global_beta"] == 4.0
    assert validated["objective"]["common_beta"]["beta_source_aggregate_sha256"] == ("c" * 64)
    assert (
        validated["reward_model"]["optimizer_protocol"]["learning_rate_schedule"]["schedule_sha256"]
        == "46e0b0fdc70c507b0325c445068326ba7bc30326d70b65fa33803cc3c876c216"
    )


def test_production_writer_rejects_nonproduction_publication_parent(
    tmp_path: Path,
) -> None:
    output = tmp_path / "phase2-post-recovery-calibration-aggregate.json"
    with pytest.raises(ValueError, match="production.*aggregates directory"):
        write_phase2_post_recovery_aggregate(
            ROOT / "configs" / "common_beta_pilot.yaml",
            [],
            output,
            require_production_output_path=True,
            publication_output_path=output,
        )


def test_shell_control_plane_covers_all_v3_phases_and_keeps_v2_replay() -> None:
    pilot_submit = (ROOT / "scripts" / "hpc4" / "submit_phase2_post_recovery_pilot.sh").read_text(
        encoding="utf-8"
    )
    pilot_job = (ROOT / "scripts" / "hpc4" / "phase2_post_recovery_calibration.sbatch").read_text(
        encoding="utf-8"
    )
    aggregate_submit = (
        ROOT / "scripts" / "hpc4" / "submit_phase2_post_recovery_aggregate.sh"
    ).read_text(encoding="utf-8")
    aggregate_job = (ROOT / "scripts" / "hpc4" / "phase2_post_recovery_aggregate.sbatch").read_text(
        encoding="utf-8"
    )
    legacy_pilot = (ROOT / "scripts" / "hpc4" / "submit_phase2_pilot.sh").read_text(
        encoding="utf-8"
    )
    legacy_aggregate = (ROOT / "scripts" / "hpc4" / "submit_phase2_pilot_aggregate.sh").read_text(
        encoding="utf-8"
    )
    pilot_submit_once = (
        ROOT / "scripts" / "hpc4" / "submit_phase2_post_recovery_array_once.py"
    ).read_text(encoding="utf-8")

    for text in (pilot_submit, aggregate_submit):
        assert "verify_beta_source_aggregate" in text
        assert "verify_horizon_parent_aggregate" in text
        assert "common_beta_post_recovery_calibration_horizon_" in text
        assert "common_beta_post_recovery_freeze_retry_" in text
    assert "submit_phase2_post_recovery_array_once.py" in pilot_submit
    assert 'ARRAY_SPEC = "0-2%2"' in pilot_submit_once
    assert "--no-requeue" in pilot_submit_once
    assert "post-recovery-predecessor-check" in pilot_job
    assert "reward_head_or_optimizer_state_reused" in pilot_job
    assert "--beta-source-aggregate" in pilot_job
    assert "--horizon-parent-aggregate" in pilot_job
    assert "PRORM_POST_RECOVERY_PILOT_PHASE" in aggregate_job
    assert "--publication-output" in aggregate_job
    assert "PRORM_PHASE2_BETA_SOURCE_AGGREGATE_PRESENT" in aggregate_job
    for text in (pilot_job, aggregate_job):
        assert "freeze_run_evidence_root" in text
        assert "freeze_artifact_evidence_root" in text
        assert "New-horizon calibration" in text or "new-horizon calibration" in text
    assert (
        'if [[ "${PRORM_POST_RECOVERY_PILOT_PHASE}" = "freeze" \\\n'
        '    && "${PRORM_POST_RECOVERY_OVERLAY_REL}"' not in pilot_job
    )
    assert "configs/common_beta_post_recovery_freeze_retry_[1-9]*.yaml" not in aggregate_job
    assert "prorm-common-beta-config/v2" in legacy_pilot
    assert "phase2_pilot_aggregate.sbatch" in legacy_aggregate
    assert "submit_phase2_post_recovery_pilot.sh" in legacy_pilot
    assert "submit_phase2_post_recovery_aggregate.sh" in legacy_aggregate


def test_confirmatory_and_downstream_consumers_are_schema_agnostic() -> None:
    confirmatory_submit = (ROOT / "scripts" / "hpc4" / "submit_phase2_confirmatory.sh").read_text(
        encoding="utf-8"
    )
    confirmatory_job = (ROOT / "scripts" / "hpc4" / "phase2_confirmatory.sbatch").read_text(
        encoding="utf-8"
    )
    assert "prorm-common-beta-config/v2" in confirmatory_submit
    assert "prorm-common-beta-post-recovery-experiment/v1" in confirmatory_submit
    assert "verify_beta_source_aggregate" in confirmatory_submit
    assert "verify_horizon_parent_aggregate" in confirmatory_submit
    assert "accepted-freeze-preflight.json" in confirmatory_job
    assert "prorm-phase2-fixed-wave-campaign-plan/v1" in confirmatory_submit
    assert "prorm-phase2-campaign-submission/v3" in confirmatory_submit
    assert '"max_submitted_tasks": 4' in confirmatory_submit
    assert '"max_running_tasks": 2' in confirmatory_submit
    assert '"optional_stopping_allowed": False' in confirmatory_submit
    assert "--no-requeue" in confirmatory_submit
    assert "attempt_index=1" in confirmatory_job

    consumers = (
        "resolve_phase2_campaign_registry.py",
        "submit_phase2_campaign_finalize.sh",
        "phase2_campaign_finalize.sbatch",
        "terminalize_phase2_scheduler_failure.sh",
        "terminalize_phase2_compute_failure.sh",
        "publish_phase2_terminal_bundle.py",
    )
    for name in consumers:
        text = (ROOT / "scripts" / "hpc4" / name).read_text(encoding="utf-8")
        assert "common-beta-pilot-selection-aggregate/v2" not in text
    resolver = (ROOT / "scripts" / "hpc4" / "resolve_phase2_campaign_registry.py").read_text(
        encoding="utf-8"
    )
    finalizer = (ROOT / "scripts" / "hpc4" / "phase2_campaign_finalize.sbatch").read_text(
        encoding="utf-8"
    )
    assert "accepted_freeze_aggregate_sha256" in resolver
    assert "accepted_freeze_aggregate_sha256" in finalizer
    assert "phase2-config-check" in finalizer
