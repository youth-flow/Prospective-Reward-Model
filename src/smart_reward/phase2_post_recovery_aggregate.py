"""Target-free aggregation with post-recovery authorization and Slurm evidence."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

from .paths import relative_posix_reference
from .phase2_config import load_phase2_config_bundle
from .phase2_pilot_aggregate import (
    PHASE2_PILOT_AGGREGATION_IDENTITY_SCHEMA,
    build_phase2_pilot_aggregate,
)
from .phase2_post_recovery_control import (
    OPTIMIZER_SCHEDULE_SHA256,
    ORDERED_SEEDS,
    POST_RECOVERY_OUTPUT_VERIFICATION_SCHEMA,
    verify_post_recovery_aggregate_success_receipt,
    verify_post_recovery_success_marker,
    verify_post_recovery_terminal_evidence,
    verify_recovery_authorization_config_binding,
)
from .repro import atomic_write_json

PHASE2_POST_RECOVERY_AGGREGATE_SCHEMA = "common-beta-pilot-selection-aggregate/v3"
PHASE2_POST_RECOVERY_CONTROL_SCHEMA = "phase2-post-recovery-aggregation-control/v1"
_HEX = frozenset("0123456789abcdef")


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _digest(value: object, *, name: str, lengths: frozenset[int] = frozenset({64})) -> str:
    if (
        not isinstance(value, str)
        or len(value) not in lengths
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase hexadecimal digest")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, *, name: str) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{name} must be a regular non-symlink file")

    def reject_duplicates(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"{name} contains duplicate JSON key {key!r}")
            value[key] = item
        return value

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicates,
        parse_constant=lambda item: (_ for _ in ()).throw(
            ValueError(f"{name} contains non-finite constant {item!r}")
        ),
    )
    if not isinstance(value, dict):
        raise TypeError(f"{name} must contain one JSON object")
    return value


def _validator_source_sha256() -> str:
    return _sha256_file(Path(__file__))


def _legacy_validator_source_sha256() -> str:
    from . import phase2_pilot_aggregate

    return _sha256_file(Path(phase2_pilot_aggregate.__file__))


def _deep_validator_source_sha256() -> str:
    from . import phase2_aggregate

    return _sha256_file(Path(phase2_aggregate.__file__))


def _overlay_git_binding(
    overlay_path: str | os.PathLike[str],
    *,
    aggregator_git_commit: str,
    producer_git_commit: str,
    expected_repo_relative: str,
) -> dict[str, str]:
    """Bind the parsed overlay to exact producer/aggregator Git blob bytes."""

    overlay = Path(overlay_path).absolute()
    if not overlay.is_file() or overlay.is_symlink():
        raise ValueError("post-recovery overlay must be a regular non-symlink file")

    def git(*arguments: str, binary: bool = False) -> str | bytes:
        completed = subprocess.run(
            ["git", "-C", os.fspath(overlay.parent), *arguments],
            check=False,
            capture_output=True,
            text=not binary,
            timeout=30,
        )
        if completed.returncode != 0 or completed.stderr:
            raise ValueError("could not verify the post-recovery overlay Git binding")
        return completed.stdout

    raw_root = git("rev-parse", "--show-toplevel")
    if not isinstance(raw_root, str):
        raise TypeError("Git root query unexpectedly returned bytes")
    repository_root = Path(raw_root.strip()).resolve()
    resolved_overlay = overlay.resolve(strict=True)
    try:
        relative = resolved_overlay.relative_to(repository_root).as_posix()
    except ValueError as error:
        raise ValueError("post-recovery overlay is outside its Git repository") from error
    if relative != expected_repo_relative:
        raise ValueError("post-recovery overlay repository-relative path is not locked")
    raw_head = git("rev-parse", "--verify", "HEAD")
    if not isinstance(raw_head, str) or raw_head.strip() != aggregator_git_commit:
        raise ValueError("aggregate checkout HEAD differs from aggregator commit")
    raw_status = git("status", "--porcelain", "--untracked-files=normal")
    if not isinstance(raw_status, str) or raw_status:
        raise ValueError("aggregate checkout must be clean for overlay Git binding")
    blob_oids: list[str] = []
    for commit in (producer_git_commit, aggregator_git_commit):
        raw_blob = git("rev-parse", "--verify", f"{commit}:{relative}")
        if not isinstance(raw_blob, str):
            raise TypeError("Git blob query unexpectedly returned bytes")
        blob = raw_blob.strip()
        if len(blob) != 40 or any(character not in _HEX for character in blob):
            raise ValueError("post-recovery overlay Git blob must be a SHA-1 object ID")
        blob_oids.append(blob)
    if blob_oids[0] != blob_oids[1]:
        raise ValueError("producer and aggregator commits contain different overlay blobs")
    raw_blob_bytes = git("cat-file", "blob", blob_oids[0], binary=True)
    if not isinstance(raw_blob_bytes, bytes) or raw_blob_bytes != overlay.read_bytes():
        raise ValueError("post-recovery overlay bytes differ from the committed Git blob")
    raw_hash_object = git("hash-object", "--", os.fspath(overlay))
    if not isinstance(raw_hash_object, str) or raw_hash_object.strip() != blob_oids[0]:
        raise ValueError("post-recovery overlay hash-object differs from its Git blob")
    return {
        "phase2_overlay_repo_relative": relative,
        "phase2_overlay_sha256": _sha256_file(overlay),
        "phase2_overlay_git_blob_sha1": blob_oids[0],
        "phase2_overlay_git_commit": producer_git_commit,
    }


def _strict_output_receipt(
    path: Path,
    *,
    seed: int,
    design_sha256: str,
    base_config_hash: str,
    authorization_sha256: str,
    result_sha256: str,
    old_output_sha256: str,
    diagnostics_sha256: str,
    allocation_job_id_raw: str,
    array_job_id: str,
    array_task_id: int,
    pilot_phase: str,
) -> dict[str, object]:
    value = _load_json(path, name="post-recovery output verification")
    expected_keys = {
        "schema_version",
        "status",
        "pilot_phase",
        "slurm_job_id_raw",
        "allocation_job_id_raw",
        "slurm_array_task_job_id",
        "array_job_id",
        "array_task_id",
        "seed",
        "phase2_design_sha256",
        "source_config_hash",
        "result_sha256",
        "phase2_output_verification_sha256",
        "diagnostics_sha256",
        "recovery_authorization_sha256",
        "optimizer_schedule_sha256",
        "materialization_mode",
        "recovery_outputs_reused",
        "five_head_adopted_schedule_verified",
        "five_head_training",
        "target_free_information_boundary_verified",
    }
    if set(value) != expected_keys:
        raise ValueError("post-recovery output verification fields differ")
    training = _mapping(
        value["five_head_training"],
        name="post-recovery output verification.five_head_training",
    )
    if (
        value["schema_version"] != POST_RECOVERY_OUTPUT_VERIFICATION_SCHEMA
        or value["status"] != "passed"
        or value["pilot_phase"] != pilot_phase
        or value["slurm_job_id_raw"] != allocation_job_id_raw
        or value["allocation_job_id_raw"] != allocation_job_id_raw
        or value["slurm_array_task_job_id"] != f"{array_job_id}_{array_task_id}"
        or value["array_job_id"] != array_job_id
        or value["array_task_id"] != str(array_task_id)
        or value["seed"] != seed
        or value["phase2_design_sha256"] != design_sha256
        or value["source_config_hash"] != base_config_hash
        or value["result_sha256"] != result_sha256
        or value["phase2_output_verification_sha256"] != old_output_sha256
        or value["diagnostics_sha256"] != diagnostics_sha256
        or value["recovery_authorization_sha256"] != authorization_sha256
        or value["optimizer_schedule_sha256"] != OPTIMIZER_SCHEDULE_SHA256
        or value["materialization_mode"] != "fresh"
        or value["recovery_outputs_reused"] is not False
        or value["five_head_adopted_schedule_verified"] is not True
        or value["target_free_information_boundary_verified"] is not True
        or set(training)
        != {
            "primary_bt_mle",
            "primary_prorm_plus",
            "low_dimensional_prorm_plus",
            "exact_margin_prorm_plus",
            "exact_soft_label_bt",
        }
    ):
        raise ValueError("post-recovery output verification identity is invalid")
    for name, raw in training.items():
        head = _mapping(raw, name=f"post-recovery output verification.{name}")
        if (
            head.get("schedule_sha256") != OPTIMIZER_SCHEDULE_SHA256
            or head.get("fresh_zero_head") is not True
            or head.get("fresh_optimizer_state") is not True
            or head.get("per_update_state_checks_passed") is not True
        ):
            raise ValueError(f"post-recovery output verification {name} is invalid")
    return value


def build_phase2_post_recovery_aggregate(
    overlay_path: str | os.PathLike[str],
    result_jsons: Sequence[str | os.PathLike[str]],
    *,
    authorization_path: str | os.PathLike[str],
    authorization_sha256: str,
    terminal_evidence_path: str | os.PathLike[str],
    terminal_evidence_sha256: str,
    array_job_id: str,
    submission_intent_sha256: str,
    submission_ledger_sha256: str,
    submission_intent_reference_path: str | os.PathLike[str],
    submission_ledger_reference_path: str | os.PathLike[str],
    aggregator_git_commit: str,
    producer_git_commit: str,
    image_sha256: str,
    hf_inventory_sha256: str,
    reference_base: str | os.PathLike[str],
    phase2_overlay_reference_path: str | os.PathLike[str] | None = None,
    beta_source_aggregate_path: str | os.PathLike[str] | None = None,
    horizon_parent_aggregate_path: str | os.PathLike[str] | None = None,
) -> dict[str, object]:
    """Build v3 only after auth, three SUCCESS receipts, and sacct all agree."""

    for value, name, lengths in (
        (authorization_sha256, "authorization_sha256", frozenset({64})),
        (terminal_evidence_sha256, "terminal_evidence_sha256", frozenset({64})),
        (submission_intent_sha256, "submission_intent_sha256", frozenset({64})),
        (submission_ledger_sha256, "submission_ledger_sha256", frozenset({64})),
        (aggregator_git_commit, "aggregator_git_commit", frozenset({40, 64})),
        (producer_git_commit, "producer_git_commit", frozenset({40, 64})),
        (image_sha256, "image_sha256", frozenset({64})),
        (hf_inventory_sha256, "hf_inventory_sha256", frozenset({64})),
    ):
        _digest(value, name=name, lengths=lengths)
    binding = verify_recovery_authorization_config_binding(
        authorization_path,
        overlay_path,
        expected_sha256=authorization_sha256,
    )
    verified_predecessors: set[Path] = set()
    for raw_predecessor in (
        beta_source_aggregate_path,
        horizon_parent_aggregate_path,
    ):
        if raw_predecessor is None:
            continue
        predecessor = Path(raw_predecessor).absolute()
        if predecessor in verified_predecessors:
            continue
        verify_post_recovery_aggregate_success_receipt(predecessor)
        verified_predecessors.add(predecessor)
    pilot_phase = str(binding["pilot_phase"])
    terminal = verify_post_recovery_terminal_evidence(
        terminal_evidence_path,
        expected_sha256=terminal_evidence_sha256,
        expected_array_job_id=array_job_id,
        expected_pilot_phase=pilot_phase,
    )
    if len(result_jsons) != len(ORDERED_SEEDS):
        raise ValueError("post-recovery aggregation requires exactly three ordered results")
    design_sha256 = str(binding["phase2_design_sha256"])
    base_config_hash = str(binding["base_config_hash"])
    source_records: list[dict[str, object]] = []
    rows = terminal["rows"]
    if not isinstance(rows, list):
        raise TypeError("terminal evidence rows must be a list")
    observed_allocation_job_ids: set[str] = set()
    for task, (seed, raw_result, raw_row) in enumerate(
        zip(ORDERED_SEEDS, result_jsons, rows, strict=True)
    ):
        result = Path(raw_result).absolute()
        if not result.is_file() or result.is_symlink():
            raise ValueError(f"post-recovery result {task} is missing or unsafe")
        run = result.parent
        if (
            result.name != "phase2-pilot-diagnostics.json"
            or run.name != f"job-{array_job_id}_{task}"
            or run.parent.name != f"seed-{seed}"
        ):
            raise ValueError("post-recovery result path does not match array task/seed")
        row = _mapping(raw_row, name=f"terminal row {task}")
        allocation_job_id_raw = str(row.get("job_id_raw"))
        if allocation_job_id_raw in observed_allocation_job_ids:
            raise ValueError("terminal allocation JobIDRaw values must be unique")
        observed_allocation_job_ids.add(allocation_job_id_raw)
        marker = verify_post_recovery_success_marker(
            run / "SUCCESS",
            expected_array_job_id=array_job_id,
            expected_task_id=task,
            expected_seed=seed,
            expected_design_sha256=design_sha256,
            expected_base_config_hash=base_config_hash,
            expected_git_commit=producer_git_commit,
            expected_authorization_sha256=authorization_sha256,
            expected_submission_intent_sha256=submission_intent_sha256,
            expected_submission_ledger_sha256=submission_ledger_sha256,
            expected_allocation_job_id_raw=allocation_job_id_raw,
            expected_pilot_phase=pilot_phase,
        )
        if (
            marker["slurm_job_id"] != row.get("job_id_raw")
            or marker["allocation_job_id_raw"] != row.get("job_id_raw")
            or marker["slurm_array_task_job_id"] != row.get("job_id")
        ):
            raise ValueError("SUCCESS marker job IDs differ from terminal sacct evidence")
        result_sha256 = _sha256_file(result)
        old_output = run / "phase2-output-verification.json"
        strict_output = run / "post-recovery-output-verification.json"
        sidecar = run / "phase2-pilot-diagnostics.diagnostics.jsonl"
        manifest = run / "run-manifest.json"
        artifact_metadata = run / "artifact" / "metadata.json"
        for path, name in (
            (old_output, "phase2 output verification"),
            (strict_output, "post-recovery output verification"),
            (sidecar, "diagnostics sidecar"),
            (manifest, "run manifest"),
            (artifact_metadata, "artifact metadata"),
        ):
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"{name} is missing or unsafe")
        old_output_sha256 = _sha256_file(old_output)
        strict_output_sha256 = _sha256_file(strict_output)
        diagnostics_sha256 = _sha256_file(sidecar)
        if (
            marker["phase2_result_sha256"] != result_sha256
            or marker["phase2_output_verification_sha256"] != old_output_sha256
            or marker["post_recovery_output_verification_sha256"] != strict_output_sha256
            or marker["artifact_metadata_sha256"] != _sha256_file(artifact_metadata)
        ):
            raise ValueError("SUCCESS marker does not bind its immutable source files")
        _strict_output_receipt(
            strict_output,
            seed=seed,
            design_sha256=design_sha256,
            base_config_hash=base_config_hash,
            authorization_sha256=authorization_sha256,
            result_sha256=result_sha256,
            old_output_sha256=old_output_sha256,
            diagnostics_sha256=diagnostics_sha256,
            allocation_job_id_raw=allocation_job_id_raw,
            array_job_id=array_job_id,
            array_task_id=task,
            pilot_phase=pilot_phase,
        )
        source_records.append(
            {
                "seed": seed,
                "array_task_id": task,
                "job_id": row["job_id"],
                "result": result,
                "result_sha256": result_sha256,
                "diagnostics_jsonl": sidecar,
                "diagnostics_sha256": diagnostics_sha256,
                "artifact_metadata": artifact_metadata,
                "artifact_metadata_sha256": _sha256_file(artifact_metadata),
                "run_manifest": manifest,
                "run_manifest_sha256": _sha256_file(manifest),
                "output_verification": old_output,
                "output_verification_sha256": old_output_sha256,
                "post_recovery_output_verification": strict_output,
                "post_recovery_output_verification_sha256": strict_output_sha256,
                "success_receipt": run / "SUCCESS",
                "success_receipt_sha256": _sha256_file(run / "SUCCESS"),
            }
        )

    bundle = load_phase2_config_bundle(overlay_path)
    legacy = build_phase2_pilot_aggregate(
        bundle.config,
        [Path(path).absolute() for path in result_jsons],
        aggregation_identity={
            "schema_version": PHASE2_PILOT_AGGREGATION_IDENTITY_SCHEMA,
            "aggregator_git_commit": aggregator_git_commit,
            "producer_git_commit": producer_git_commit,
            "image_sha256": image_sha256,
            "hf_inventory_sha256": hf_inventory_sha256,
            "validator_source_sha256": _legacy_validator_source_sha256(),
        },
        reference_base=reference_base,
        beta_source_aggregate=beta_source_aggregate_path,
        horizon_parent_aggregate=horizon_parent_aggregate_path,
    )
    overlay_filename, _ = _semantic_lineage_filenames(legacy)
    overlay_repo_relative = f"configs/{overlay_filename}"
    overlay_git = _overlay_git_binding(
        overlay_path,
        aggregator_git_commit=aggregator_git_commit,
        producer_git_commit=producer_git_commit,
        expected_repo_relative=overlay_repo_relative,
    )

    base = Path(reference_base).resolve()
    public_sources = [
        {
            key: (relative_posix_reference(value, base=base) if isinstance(value, Path) else value)
            for key, value in source.items()
        }
        for source in source_records
    ]
    result = dict(legacy)
    if result.get("pilot_phase") != pilot_phase:
        raise ValueError("pilot aggregate phase differs from the authorization/config binding")
    result["schema_version"] = PHASE2_POST_RECOVERY_AGGREGATE_SCHEMA
    result["sources"] = public_sources
    result["post_recovery_control"] = {
        "schema_version": PHASE2_POST_RECOVERY_CONTROL_SCHEMA,
        "pilot_phase": pilot_phase,
        "phase2_overlay": relative_posix_reference(
            (
                overlay_path
                if phase2_overlay_reference_path is None
                else phase2_overlay_reference_path
            ),
            base=base,
        ),
        **overlay_git,
        "normalized_phase2_config": bundle.config,
        "normalized_phase2_config_sha256": bundle.design_identity,
        "recovery_authorization": relative_posix_reference(
            authorization_path,
            base=base,
        ),
        "recovery_authorization_sha256": authorization_sha256,
        "optimizer_schedule_sha256": OPTIMIZER_SCHEDULE_SHA256,
        "submission_intent": relative_posix_reference(
            submission_intent_reference_path,
            base=base,
        ),
        "submission_intent_sha256": submission_intent_sha256,
        "submission_ledger": relative_posix_reference(
            submission_ledger_reference_path,
            base=base,
        ),
        "submission_ledger_sha256": submission_ledger_sha256,
        "pilot_terminal_evidence": relative_posix_reference(
            terminal_evidence_path,
            base=base,
        ),
        "pilot_terminal_evidence_sha256": terminal_evidence_sha256,
        "pilot_array_job_id": array_job_id,
        "ordered_seeds": list(ORDERED_SEEDS),
        "materialization_mode": "fresh",
        "recovery_outputs_reused": False,
        "all_tasks_terminal_completed_zero_exit": True,
        "post_recovery_validator_source_sha256": _validator_source_sha256(),
        "phase2_deep_validator_source_sha256": _deep_validator_source_sha256(),
    }
    return result


def _semantic_lineage_filenames(
    payload: Mapping[str, object],
) -> tuple[str, str]:
    pilot_phase = payload.get("pilot_phase")
    if pilot_phase == "calibration":
        horizon = _mapping(payload.get("horizon"), name="post-recovery aggregate.horizon")
        horizon_grid_index = horizon.get("horizon_grid_index")
        if (
            isinstance(horizon_grid_index, bool)
            or not isinstance(horizon_grid_index, int)
            or horizon_grid_index < 0
        ):
            raise ValueError("post-recovery calibration horizon grid index is invalid")
        expected_name = (
            "phase2-post-recovery-calibration-aggregate.json"
            if horizon_grid_index == 0
            else (f"phase2-post-recovery-calibration-horizon-{horizon_grid_index}-aggregate.json")
        )
        overlay_name = (
            "common_beta_post_recovery_calibration.yaml"
            if horizon_grid_index == 0
            else f"common_beta_post_recovery_calibration_horizon_{horizon_grid_index}.yaml"
        )
    elif pilot_phase == "freeze":
        selection = _mapping(
            payload.get("selection"),
            name="post-recovery aggregate.selection",
        )
        beta_grid_index = selection.get("beta_grid_index")
        if (
            isinstance(beta_grid_index, bool)
            or not isinstance(beta_grid_index, int)
            or beta_grid_index < 0
        ):
            raise ValueError("post-recovery freeze beta grid index is invalid")
        expected_name = (
            "phase2-post-recovery-freeze-aggregate.json"
            if beta_grid_index == 0
            else f"phase2-post-recovery-freeze-retry-{beta_grid_index}-aggregate.json"
        )
        overlay_name = (
            "common_beta_post_recovery_freeze.yaml"
            if beta_grid_index == 0
            else f"common_beta_post_recovery_freeze_retry_{beta_grid_index}.yaml"
        )
    else:
        raise ValueError("post-recovery aggregate pilot phase is invalid")
    return overlay_name, expected_name


def _semantic_aggregate_filename(payload: Mapping[str, object]) -> str:
    _, aggregate_name = _semantic_lineage_filenames(payload)
    return aggregate_name


def write_phase2_post_recovery_aggregate(
    overlay_path: str | os.PathLike[str],
    result_jsons: Sequence[str | os.PathLike[str]],
    output_path: str | os.PathLike[str],
    *,
    reference_base: str | os.PathLike[str] | None = None,
    require_production_output_path: bool = False,
    publication_output_path: str | os.PathLike[str] | None = None,
    **kwargs: object,
) -> dict[str, object]:
    destination = Path(output_path)
    publication = destination if publication_output_path is None else Path(publication_output_path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite post-recovery aggregate: {destination}")
    if not destination.parent.is_dir() or destination.parent.is_symlink():
        raise ValueError("post-recovery aggregate output parent is unsafe")
    if require_production_output_path:
        expected_parent = Path("/project/sigroup/smart-reward-model/aggregates")
        try:
            resolved_parent = publication.parent.resolve(strict=True)
        except OSError as error:
            raise ValueError("production post-recovery aggregate parent is inaccessible") from error
        if resolved_parent != expected_parent:
            raise ValueError(
                "production post-recovery aggregate output parent must be the "
                "locked project aggregates directory"
            )
    payload = build_phase2_post_recovery_aggregate(
        overlay_path,
        result_jsons,
        reference_base=destination.parent if reference_base is None else reference_base,
        **kwargs,
    )
    expected_name = _semantic_aggregate_filename(payload)
    if publication.name != expected_name:
        raise ValueError(
            "post-recovery aggregate output filename differs from its semantic lineage"
        )
    atomic_write_json(destination, payload, overwrite=False)
    return payload


__all__ = [
    "PHASE2_POST_RECOVERY_AGGREGATE_SCHEMA",
    "PHASE2_POST_RECOVERY_CONTROL_SCHEMA",
    "build_phase2_post_recovery_aggregate",
    "write_phase2_post_recovery_aggregate",
]
