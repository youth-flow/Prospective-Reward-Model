#!/usr/bin/env python3
"""Authenticate and publish the fixed-three exploratory descriptive aggregate."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import os
import re
import stat
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any

from smart_reward.phase2_exploratory_aggregate import (
    FIXED_THREE_EXPLORATORY_SEEDS,
    assert_exploratory_payload_has_no_inferential_fields,
    build_fixed_three_exploratory_aggregate,
    normalize_budgeted_end_to_end_seed_result,
    validate_fixed_three_exploratory_aggregate,
)

PUBLICATION_SCHEMA = "prorm-phase2-budgeted-end-to-end-fixed-three-descriptive-publication/v1"
VERIFICATION_SCHEMA = "prorm-phase2-budgeted-fixed-three-seed-output-verification/v1"
RUN_STATUS_SCHEMA = "prorm-phase2-budgeted-end-to-end-fixed-three-run-status/v1"
EVIDENCE_ROLE = "budgeted_end_to_end_fixed_three_exploratory_only"
STAGE = "budgeted_end_to_end"
OPTIMIZER_SCHEDULE_SHA256 = "46e0b0fdc70c507b0325c445068326ba7bc30326d70b65fa33803cc3c876c216"
FIXED_BOOTSTRAP_SEED = 20260801
FIXED_BOOTSTRAP_RESAMPLES = 10000
FIXED_CONFIDENCE_LEVEL = 0.95

_HEX = frozenset("0123456789abcdef")
_SUCCESS_KEYS = (
    "schema_version",
    "status",
    "formal",
    "evidence_role",
    "stage",
    "seed",
    "slurm_job_id",
    "array_job_id",
    "array_task_id",
    "cluster",
    "account",
    "partition",
    "restart_count",
    "phase2_design_sha256",
    "base_config_hash",
    "git_commit",
    "submission_intent_sha256",
    "submission_ledger_sha256",
    "freeze_evidence_sha256",
    "frozen_global_beta",
    "optimizer_schedule_sha256",
    "artifact_metadata_sha256",
    "phase2_result_sha256",
    "rollouts_sha256",
    "verification_sha256",
    "manifest_sha256",
    "workload_exit_code",
    "final_exit_code",
    "created_at_utc",
)
_VERIFICATION_KEYS = {
    "schema_version",
    "status",
    "design_stage",
    "evidence_role",
    "formal_eligibility",
    "formal_claim_eligible",
    "supports_formal_claim",
    "inferential_or_significance_claim_produced",
    "seed",
    "phase2_design_sha256",
    "base_config_hash",
    "accepted_freeze_aggregate_sha256",
    "frozen_global_beta",
    "phase2_runtime_contract_sha256",
    "git_commit",
    "image_sha256",
    "hf_inventory_sha256",
    "slurm_job_id_raw",
    "array_job_id",
    "array_task_id",
    "result_sha256",
    "rollouts_sha256",
    "run_manifest_sha256",
    "artifact_metadata_sha256",
    "artifact_materialization_sha256",
    "slurm",
    "relative_files",
    "input_sha256",
    "rollout_geometry",
    "environment_identity",
    "normalized_seed_record",
}
_IDENTITY_KEYS = (
    "phase2_design_sha256",
    "base_config_hash",
    "git_commit",
    "image_sha256",
    "hf_inventory_sha256",
    "accepted_freeze_aggregate_sha256",
    "phase2_runtime_contract_sha256",
    "frozen_global_beta",
)
_ENDPOINT_KEYS = {
    "heldout_local_regret",
    "finite_policy_utility",
    "oracle_pairwise_cross_entropy",
    "oracle_probability_mae",
    "pairwise_order_accuracy",
}
_PUBLICATION_CODE_FILES = (
    "scripts/hpc4/aggregate_phase2_budgeted_end_to_end.py",
    "scripts/hpc4/capture_phase2_budgeted_end_to_end_terminal.py",
    "scripts/hpc4/submit_phase2_budgeted_end_to_end_once.py",
    "scripts/hpc4/verify_phase2_budgeted_end_to_end_seed_output.py",
    "src/smart_reward/phase2_exploratory_aggregate.py",
    "src/smart_reward/statistics.py",
)


@dataclass(frozen=True, slots=True)
class _FileSnapshot:
    path: Path
    sha256: str
    size_bytes: int
    identity: tuple[int, int, int, int, int]
    raw: bytes | None


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _digest(
    value: object,
    *,
    name: str,
    lengths: frozenset[int] = frozenset({64}),
) -> str:
    if (
        not isinstance(value, str)
        or len(value) not in lengths
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{name} must be a full lowercase hexadecimal digest")
    return value


def _positive_job_id(value: object, *, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[1-9][0-9]*", value) is None:
        raise ValueError(f"{name} must be a positive decimal Slurm ID")
    return value


def _finite_positive(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite positive scalar")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive scalar")
    return result


def _valid_utc(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return False
    return True


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _decode_json(raw: bytes, *, name: str) -> dict[str, Any]:
    def reject_duplicates(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"{name} contains non-finite JSON constant {value}")

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _real_directory(path: Path, *, name: str) -> Path:
    absolute = path.absolute()
    try:
        metadata = absolute.lstat()
        resolved = absolute.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{name} is missing or inaccessible") from error
    if not stat.S_ISDIR(metadata.st_mode) or absolute.is_symlink() or resolved != absolute:
        raise ValueError(f"{name} must be a canonical real directory")
    return absolute


def _real_file(path: Path, *, name: str) -> Path:
    absolute = path.absolute()
    try:
        metadata = absolute.lstat()
        resolved = absolute.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{name} is missing or inaccessible") from error
    if not stat.S_ISREG(metadata.st_mode) or absolute.is_symlink() or resolved != absolute:
        raise ValueError(f"{name} must be a canonical regular non-symlink file")
    return absolute


def _snapshot_file(
    path: Path,
    *,
    name: str,
    retain_raw: bool,
    maximum_bytes: int,
) -> _FileSnapshot:
    """Read one immutable view through one no-follow descriptor and hash it."""

    source = Path(path).absolute()
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source, flags)
    except OSError as error:
        raise ValueError(f"{name} is missing, inaccessible, or a symlink") from error
    digest = hashlib.sha256()
    chunks: list[bytes] | None = [] if retain_raw else None
    size = 0
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"{name} must be a regular non-symlink file")
        try:
            path_metadata = source.lstat()
            resolved = source.resolve(strict=True)
        except OSError as error:
            raise ValueError(f"{name} changed while it was opened") from error
        if (
            not stat.S_ISREG(path_metadata.st_mode)
            or source.is_symlink()
            or resolved != source
            or path_metadata.st_dev != metadata.st_dev
            or path_metadata.st_ino != metadata.st_ino
        ):
            raise ValueError(f"{name} changed identity while it was opened")
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            size += len(block)
            if size > maximum_bytes:
                raise ValueError(f"{name} exceeds its locked byte limit")
            digest.update(block)
            if chunks is not None:
                chunks.append(block)
        final_metadata = os.fstat(descriptor)
        initial_identity = (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        )
        final_identity = (
            final_metadata.st_dev,
            final_metadata.st_ino,
            final_metadata.st_size,
            final_metadata.st_mtime_ns,
            final_metadata.st_ctime_ns,
        )
        if initial_identity != final_identity or size != metadata.st_size:
            raise ValueError(f"{name} changed during its single-descriptor snapshot")
    finally:
        os.close(descriptor)
    return _FileSnapshot(
        path=source,
        sha256=digest.hexdigest(),
        size_bytes=size,
        identity=initial_identity,
        raw=b"".join(chunks) if chunks is not None else None,
    )


def _json_snapshot(
    path: Path,
    *,
    name: str,
    canonical: bool = False,
) -> tuple[dict[str, Any], _FileSnapshot]:
    snapshot = _snapshot_file(
        path,
        name=name,
        retain_raw=True,
        maximum_bytes=64 * 1024 * 1024,
    )
    assert snapshot.raw is not None
    value = _decode_json(snapshot.raw, name=name)
    if canonical and snapshot.raw != _canonical_json(value):
        raise ValueError(f"{name} must use canonical JSON bytes")
    return value, snapshot


def _verify_snapshot_unchanged(snapshot: _FileSnapshot, *, name: str) -> None:
    try:
        current = _snapshot_file(
            snapshot.path,
            name=name,
            retain_raw=False,
            maximum_bytes=max(snapshot.size_bytes, 1),
        )
    except ValueError as error:
        raise ValueError(f"{name} changed after authentication and before publication") from error
    if (
        current.sha256 != snapshot.sha256
        or current.size_bytes != snapshot.size_bytes
        or current.identity != snapshot.identity
    ):
        raise ValueError(f"{name} changed after authentication and before publication")


def _contained(path: Path, root: Path, *, name: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{name} leaves the project root") from error


def _parse_success(path: Path) -> tuple[dict[str, str], _FileSnapshot]:
    snapshot = _snapshot_file(
        path,
        name="budgeted SUCCESS marker",
        retain_raw=True,
        maximum_bytes=64 * 1024,
    )
    assert snapshot.raw is not None
    raw = snapshot.raw
    if not raw or len(raw) > 64 * 1024 or not raw.endswith(b"\n"):
        raise ValueError("budgeted SUCCESS marker bytes are invalid")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("budgeted SUCCESS marker must be UTF-8") from error
    if "\r" in text or "\x00" in text:
        raise ValueError("budgeted SUCCESS marker contains unsafe characters")
    fields: dict[str, str] = {}
    ordered: list[str] = []
    for line in text.splitlines():
        if not line or "=" not in line:
            raise ValueError("budgeted SUCCESS marker is not strict key=value text")
        key, value = line.split("=", 1)
        if not key or key in fields or "\n" in value:
            raise ValueError("budgeted SUCCESS marker contains duplicate or invalid fields")
        fields[key] = value
        ordered.append(key)
    if tuple(ordered) != _SUCCESS_KEYS:
        raise ValueError("budgeted SUCCESS marker fields or order differ from the job contract")
    return fields, snapshot


def _load_sibling(name: str) -> ModuleType:
    path = Path(__file__).resolve().with_name(name)
    spec = importlib.util.spec_from_file_location(f"_prorm_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load required control-plane module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _verify_terminal_evidence(
    path: Path,
    *,
    expected_sha256: str,
    expected_array_job_id: str,
) -> dict[str, object]:
    module = _load_sibling("capture_phase2_budgeted_end_to_end_terminal.py")
    return module.verify_terminal_evidence(
        path,
        expected_sha256=expected_sha256,
        expected_array_job_id=expected_array_job_id,
    )


def _verify_submission_ledger(
    ledger: Path,
    *,
    project_root: Path,
    repo_root: Path,
    identity: Mapping[str, object],
    marker: Mapping[str, str],
    array_job_id: str,
) -> tuple[dict[str, object], tuple[_FileSnapshot, _FileSnapshot]]:
    intent, intent_snapshot = _json_snapshot(
        ledger / "intent.json",
        name="submission intent",
        canonical=True,
    )
    _submission, submission_snapshot = _json_snapshot(
        ledger / "submission.json",
        name="submission ledger",
        canonical=True,
    )
    required = {
        "recovery_authorization_sha256",
        "phase2_overlay_sha256",
        "phase2_base_sha256",
        "export_spec",
        "submitter_user",
    }
    if not required.issubset(intent):
        raise ValueError("submission intent lacks fields required for deep verification")
    export_spec = intent["export_spec"]
    if not isinstance(export_spec, str):
        raise ValueError("submission intent export_spec is invalid")
    module = _load_sibling("submit_phase2_budgeted_end_to_end_once.py")
    verified = module.verify_submission_ledger(
        ledger,
        project_root=project_root,
        repo_root=repo_root,
        design_sha256=identity["phase2_design_sha256"],
        base_config_hash=identity["base_config_hash"],
        authorization_sha256=intent["recovery_authorization_sha256"],
        optimizer_schedule_sha256=marker["optimizer_schedule_sha256"],
        git_commit=identity["git_commit"],
        image_sha256=identity["image_sha256"],
        inventory_sha256=identity["hf_inventory_sha256"],
        overlay_sha256=intent["phase2_overlay_sha256"],
        base_file_sha256=intent["phase2_base_sha256"],
        export_spec_sha256=_sha256(export_spec.encode("utf-8")),
        array_job_id=array_job_id,
        submitter_user=intent["submitter_user"],
    )
    if (
        verified.get("intent_sha256") != intent_snapshot.sha256
        or verified.get("submission_sha256") != submission_snapshot.sha256
    ):
        raise ValueError("deep ledger verification changed between its byte snapshots")
    return verified, (intent_snapshot, submission_snapshot)


def _git_bytes(repository: Path, *arguments: str, name: str) -> bytes:
    try:
        completed = subprocess.run(
            ("git", "-C", os.fspath(repository), *arguments),
            check=False,
            capture_output=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"could not execute {name}") from error
    if completed.returncode != 0 or completed.stderr:
        raise ValueError(f"{name} failed or emitted stderr")
    return completed.stdout


def _verify_publication_checkout(
    repository: Path,
    *,
    expected_git_commit: str,
) -> tuple[dict[str, object], tuple[_FileSnapshot, ...]]:
    """Bind the normalizer/publication implementation to the producer commit."""

    expected = _digest(
        expected_git_commit,
        name="seed git commit",
        lengths=frozenset({40, 64}),
    )
    head_raw = _git_bytes(repository, "rev-parse", "--verify", "HEAD", name="Git HEAD query")
    try:
        head = head_raw.decode("ascii").strip()
    except UnicodeDecodeError as error:
        raise ValueError("Git HEAD query was not ASCII") from error
    if head != expected or head_raw != f"{head}\n".encode():
        raise ValueError("publication checkout HEAD differs from the seed producer commit")
    status = _git_bytes(
        repository,
        "status",
        "--porcelain",
        "--untracked-files=normal",
        name="Git cleanliness query",
    )
    if status:
        raise ValueError("publication checkout must be completely clean")
    executing_script = Path(__file__).resolve()
    expected_script = repository.joinpath(
        "scripts",
        "hpc4",
        "aggregate_phase2_budgeted_end_to_end.py",
    )
    if executing_script != expected_script:
        raise ValueError("publication must execute the aggregate script from the bound repository")
    normalizer_source = inspect.getsourcefile(normalize_budgeted_end_to_end_seed_result)
    expected_normalizer = repository / "src" / "smart_reward" / "phase2_exploratory_aggregate.py"
    if normalizer_source is None or Path(normalizer_source).resolve() != expected_normalizer:
        raise ValueError("publication imported its normalizer outside the bound repository")

    files: dict[str, dict[str, object]] = {}
    snapshots: list[_FileSnapshot] = []
    for relative in _PUBLICATION_CODE_FILES:
        local = _snapshot_file(
            repository.joinpath(*relative.split("/")),
            name=f"publication source {relative}",
            retain_raw=True,
            maximum_bytes=16 * 1024 * 1024,
        )
        committed = _git_bytes(
            repository,
            "cat-file",
            "blob",
            f"{head}:{relative}",
            name=f"committed publication source query for {relative}",
        )
        assert local.raw is not None
        if local.raw != committed:
            raise ValueError(f"publication source {relative} differs from the seed commit")
        files[relative] = {
            "sha256": local.sha256,
            "size_bytes": local.size_bytes,
        }
        snapshots.append(local)
    return (
        {
            "git_head": head,
            "repository_clean": True,
            "critical_files": files,
        },
        tuple(snapshots),
    )


def _unique_success(run: Path) -> None:
    seed_root = run.parent
    entries = list(seed_root.iterdir())
    if entries != [run]:
        raise ValueError("each fixed seed must have exactly one immutable job directory")
    successes: list[Path] = []
    failures: list[Path] = []
    for directory, directory_names, file_names in os.walk(run, followlinks=False):
        directory_names[:] = [
            name for name in directory_names if not (Path(directory) / name).is_symlink()
        ]
        successes.extend(Path(directory) / name for name in file_names if name == "SUCCESS")
        failures.extend(Path(directory) / name for name in file_names if name == "FAILED")
    if successes != [run / "SUCCESS"] or failures:
        raise ValueError("run must contain one root SUCCESS marker and no FAILED marker")


def _artifact_metadata(run: Path, *, expected: Path) -> Path:
    link = run / "artifact"
    try:
        metadata = link.lstat()
    except OSError as error:
        raise ValueError("run artifact link is missing") from error
    if not stat.S_ISLNK(metadata.st_mode):
        raise ValueError("run artifact must be the job-published symlink")
    try:
        resolved = link.resolve(strict=True)
    except OSError as error:
        raise ValueError("run artifact link is broken") from error
    expected_directory = _real_directory(expected, name="published seed artifact")
    if resolved != expected_directory:
        raise ValueError("run artifact link crosses the fixed seed/job identity")
    return _real_file(resolved / "metadata.json", name="artifact metadata")


def _verify_sidecar(
    value: Mapping[str, object],
    *,
    seed: int,
    task: int,
    row: Mapping[str, object],
    marker: Mapping[str, str],
    hashes: Mapping[str, str],
    normalized: Mapping[str, object],
) -> dict[str, object]:
    if set(value) != _VERIFICATION_KEYS:
        raise ValueError(f"seed {seed} output verification fields differ from the locked schema")
    false_fields = (
        "formal_eligibility",
        "formal_claim_eligible",
        "supports_formal_claim",
        "inferential_or_significance_claim_produced",
    )
    if (
        value.get("schema_version") != VERIFICATION_SCHEMA
        or value.get("status") != "verified"
        or value.get("design_stage") != STAGE
        or value.get("evidence_role") != EVIDENCE_ROLE
        or any(value.get(field) is not False for field in false_fields)
        or value.get("seed") != seed
        or value.get("array_job_id") != marker["array_job_id"]
        or value.get("array_task_id") != task
        or value.get("slurm_job_id_raw") != row["job_id_raw"]
    ):
        raise ValueError(f"seed {seed} output verification identity is invalid")
    expected_hash_fields = {
        "result_sha256": hashes["result"],
        "rollouts_sha256": hashes["rollouts"],
        "run_manifest_sha256": hashes["manifest"],
        "artifact_metadata_sha256": hashes["artifact"],
        "artifact_materialization_sha256": hashes["materialization"],
    }
    if any(value.get(key) != expected for key, expected in expected_hash_fields.items()):
        raise ValueError(f"seed {seed} output verification does not bind its files")
    relative_files = value.get("relative_files")
    if relative_files != {
        "result": "phase2-result.json",
        "rollouts": "phase2-result.rollouts.jsonl",
        "run_manifest": "run-manifest.json",
        "artifact_metadata": "artifact/metadata.json",
        "artifact_materialization": "artifact-materialization.json",
    }:
        raise ValueError(f"seed {seed} output verification filenames are invalid")
    input_sha256 = value.get("input_sha256")
    if not isinstance(input_sha256, Mapping) or set(input_sha256) != {
        "overlay",
        "result",
        "rollouts",
        "run_manifest",
        "artifact_metadata",
        "artifact_materialization",
    }:
        raise ValueError(f"seed {seed} output verification input hashes are invalid")
    input_to_file = {
        "result": "result",
        "rollouts": "rollouts",
        "run_manifest": "manifest",
        "artifact_metadata": "artifact",
        "artifact_materialization": "materialization",
    }
    for key, file_key in input_to_file.items():
        if input_sha256[key] != hashes[file_key]:
            raise ValueError(f"seed {seed} verification input hash {key!r} changed")
    slurm = value.get("slurm")
    if slurm != {
        "job_id_raw": row["job_id_raw"],
        "array_job_id": marker["array_job_id"],
        "array_task_id": task,
        "account": "sigroup",
        "cluster": "hpc4",
        "partition": "gpu-l20",
    }:
        raise ValueError(f"seed {seed} output verification Slurm identity is invalid")
    geometry = value.get("rollout_geometry")
    if not isinstance(geometry, Mapping) or set(geometry) != {
        "row_count",
        "rows_per_arm",
        "test_prompt_count",
        "candidates_per_prompt",
        "arm_order",
    }:
        raise ValueError(f"seed {seed} rollout geometry evidence is invalid")
    environment = value.get("environment_identity")
    if environment != {
        "formal": True,
        "git_commit": value["git_commit"],
        "image_sha256": value["image_sha256"],
        "hf_inventory_sha256": value["hf_inventory_sha256"],
        "account": "sigroup",
        "partition": "gpu-l20",
        "gpu_models": ["NVIDIA L20"],
    }:
        raise ValueError(f"seed {seed} environment identity is missing")
    if value.get("normalized_seed_record") != dict(normalized):
        raise ValueError(f"seed {seed} serialized normalized record differs from recomputation")
    for key in (
        "phase2_design_sha256",
        "base_config_hash",
        "image_sha256",
        "hf_inventory_sha256",
        "accepted_freeze_aggregate_sha256",
        "phase2_runtime_contract_sha256",
    ):
        _digest(value[key], name=f"seed {seed} output verification.{key}")
    _digest(
        value["git_commit"],
        name=f"seed {seed} output verification.git_commit",
        lengths=frozenset({40, 64}),
    )
    _finite_positive(
        value["frozen_global_beta"],
        name=f"seed {seed} output verification.frozen_global_beta",
    )
    return {key: value[key] for key in _IDENTITY_KEYS}


def _validate_normalized(value: Mapping[str, object], *, seed: int) -> None:
    if value.get("seed") != seed or value.get("admissible") is not True:
        raise ValueError(f"seed {seed} is inadmissible; descriptive effects are withheld")
    endpoints = value.get("endpoints")
    if not isinstance(endpoints, Mapping) or set(endpoints) != _ENDPOINT_KEYS:
        raise ValueError(f"seed {seed} normalized endpoints are incomplete")


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
    except PermissionError:
        if os.name == "nt":
            return
        raise
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_exclusive(path: Path, raw: bytes, *, name: str) -> None:
    destination = path.absolute()
    parent = _real_directory(destination.parent, name=f"{name} parent")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite {name}: {destination}")
    descriptor, staged_text = tempfile.mkstemp(prefix=f".{destination.name}.staged-", dir=parent)
    staged = Path(staged_text).absolute()
    os.chmod(staged, 0o640)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(staged, destination, follow_symlinks=False)
    except FileExistsError:
        raise FileExistsError(f"refusing to overwrite {name}: {destination}") from None
    finally:
        with suppress(OSError):
            staged.unlink()
    _fsync_directory(parent)


def _require_exact_existing(path: Path, raw: bytes, *, name: str) -> None:
    snapshot = _snapshot_file(
        path,
        name=name,
        retain_raw=True,
        maximum_bytes=max(len(raw), 1),
    )
    if snapshot.raw != raw:
        raise FileExistsError(f"refusing to replace conflicting {name}: {path}")


def _publish_recoverable_pair(
    output: Path,
    aggregate_raw: bytes,
    receipt: Path,
    receipt_raw: bytes,
) -> str:
    """Install aggregate then receipt, resuming only an exact first-phase inode."""

    output_present = output.exists() or output.is_symlink()
    receipt_present = receipt.exists() or receipt.is_symlink()
    if receipt_present and not output_present:
        raise ValueError("publication receipt exists without its aggregate")
    if output_present:
        _require_exact_existing(output, aggregate_raw, name="fixed-three descriptive aggregate")
        if receipt_present:
            _require_exact_existing(receipt, receipt_raw, name="fixed-three publication receipt")
            return "already_published"
        _write_exclusive(receipt, receipt_raw, name="fixed-three publication receipt")
        return "resumed_after_aggregate"
    _write_exclusive(output, aggregate_raw, name="fixed-three descriptive aggregate")
    _write_exclusive(receipt, receipt_raw, name="fixed-three publication receipt")
    return "published"


def write_budgeted_end_to_end_aggregate(
    project_root: Path,
    repo_root: Path,
    terminal_evidence_path: Path,
    run_directories: Sequence[Path],
    output_path: Path,
    *,
    terminal_evidence_sha256: str,
    array_job_id: str,
) -> tuple[dict[str, object], dict[str, object]]:
    """Verify all immutable evidence and exclusively publish aggregate plus receipt."""

    project = _real_directory(project_root, name="project root")
    repository = _real_directory(repo_root, name="repository root")
    array_id = _positive_job_id(array_job_id, name="array_job_id")
    terminal_digest = _digest(
        terminal_evidence_sha256,
        name="terminal_evidence_sha256",
    )
    terminal_file = _real_file(terminal_evidence_path, name="terminal scheduler evidence")
    _contained(terminal_file, project, name="terminal scheduler evidence")
    terminal_snapshot = _snapshot_file(
        terminal_file,
        name="terminal scheduler evidence",
        retain_raw=True,
        maximum_bytes=1024 * 1024,
    )
    if terminal_snapshot.sha256 != terminal_digest:
        raise ValueError("terminal scheduler evidence changed before verification")
    terminal = _verify_terminal_evidence(
        terminal_file,
        expected_sha256=terminal_digest,
        expected_array_job_id=array_id,
    )
    rows = terminal.get("rows")
    if not isinstance(rows, list) or len(rows) != len(FIXED_THREE_EXPLORATORY_SEEDS):
        raise ValueError("terminal scheduler evidence does not contain the fixed three rows")
    input_snapshots: list[tuple[_FileSnapshot, str]] = [
        (terminal_snapshot, "terminal scheduler evidence")
    ]
    raw_binding = terminal.get("raw_sacct")
    if not isinstance(raw_binding, Mapping) or set(raw_binding) != {
        "filename",
        "sha256",
        "size_bytes",
    }:
        raise ValueError("terminal scheduler evidence lacks its raw sacct binding")
    raw_name = raw_binding.get("filename")
    if not isinstance(raw_name, str) or Path(raw_name).name != raw_name:
        raise ValueError("terminal scheduler evidence raw filename is invalid")
    raw_snapshot = _snapshot_file(
        terminal_file.with_name(raw_name),
        name="terminal raw sacct evidence",
        retain_raw=False,
        maximum_bytes=128 * 1024,
    )
    if raw_snapshot.sha256 != raw_binding.get(
        "sha256"
    ) or raw_snapshot.size_bytes != raw_binding.get("size_bytes"):
        raise ValueError("terminal raw sacct evidence changed after verification")
    input_snapshots.append((raw_snapshot, "terminal raw sacct evidence"))

    runs = tuple(Path(path).absolute() for path in run_directories)
    if len(runs) != len(FIXED_THREE_EXPLORATORY_SEEDS) or len(set(runs)) != len(runs):
        raise ValueError("exactly three unique run directories are required")

    normalized_records: list[dict[str, object]] = []
    seed_receipts: list[dict[str, object]] = []
    marker_controls: list[dict[str, str]] = []
    run_rechecks: list[tuple[Path, Path, Path]] = []
    shared_identity: dict[str, object] | None = None
    shared_marker: dict[str, str] | None = None
    publication_code: dict[str, object] | None = None
    for task, (seed, raw_run, row) in enumerate(
        zip(FIXED_THREE_EXPLORATORY_SEEDS, runs, rows, strict=True)
    ):
        if not isinstance(row, Mapping):
            raise ValueError(f"terminal row {task} is invalid")
        if (
            row.get("job_id") != f"{array_id}_{task}"
            or row.get("array_job_id") != array_id
            or row.get("array_task_id") != task
            or row.get("seed") != seed
            or row.get("state") != "COMPLETED"
            or row.get("exit_code") != "0:0"
            or row.get("derived_exit_code") != "0:0"
            or row.get("cluster") != "hpc4"
            or row.get("account") != "sigroup"
            or row.get("partition") != "gpu-l20"
            or row.get("qos") != "l20_qos"
        ):
            raise ValueError(f"terminal row {task} is cross-bound or not successful")
        run = _real_directory(raw_run, name=f"seed {seed} run directory")
        _contained(run, project, name=f"seed {seed} run directory")
        _unique_success(run)
        marker, success_snapshot = _parse_success(run / "SUCCESS")
        input_snapshots.append((success_snapshot, f"seed {seed} SUCCESS marker"))
        if (
            marker["schema_version"] != RUN_STATUS_SCHEMA
            or marker["status"] != "SUCCESS"
            or marker["formal"] != "false"
            or marker["evidence_role"] != EVIDENCE_ROLE
            or marker["stage"] != STAGE
            or marker["seed"] != str(seed)
            or marker["array_job_id"] != array_id
            or marker["array_task_id"] != str(task)
            or marker["slurm_job_id"] != row.get("job_id_raw")
            or marker["cluster"] != "hpc4"
            or marker["account"] != "sigroup"
            or marker["partition"] != "gpu-l20"
            or marker["restart_count"] != "0"
            or marker["workload_exit_code"] != "0"
            or marker["final_exit_code"] != "0"
            or not _valid_utc(marker["created_at_utc"])
            or marker["optimizer_schedule_sha256"] != OPTIMIZER_SCHEDULE_SHA256
        ):
            raise ValueError(f"seed {seed} SUCCESS marker identity is invalid")
        for key in (
            "phase2_design_sha256",
            "base_config_hash",
            "submission_intent_sha256",
            "submission_ledger_sha256",
            "freeze_evidence_sha256",
            "optimizer_schedule_sha256",
            "artifact_metadata_sha256",
            "phase2_result_sha256",
            "rollouts_sha256",
            "verification_sha256",
            "manifest_sha256",
        ):
            _digest(marker[key], name=f"seed {seed} SUCCESS.{key}")
        _digest(
            marker["git_commit"],
            name=f"seed {seed} SUCCESS.git_commit",
            lengths=frozenset({40, 64}),
        )
        if task == 0:
            checkout_verification = _verify_publication_checkout(
                repository,
                expected_git_commit=marker["git_commit"],
            )
            if (
                not isinstance(checkout_verification, tuple)
                or len(checkout_verification) != 2
                or not isinstance(checkout_verification[0], Mapping)
                or not isinstance(checkout_verification[1], tuple)
            ):
                raise ValueError("publication checkout verifier returned an invalid contract")
            publication_code = dict(checkout_verification[0])
            if (
                publication_code.get("git_head") != marker["git_commit"]
                or publication_code.get("repository_clean") is not True
            ):
                raise ValueError("publication code is not the clean seed producer commit")
            input_snapshots.extend(
                (snapshot, f"publication source file {index}")
                for index, snapshot in enumerate(checkout_verification[1])
            )
        frozen_beta = _finite_positive(
            float(marker["frozen_global_beta"]),
            name=f"seed {seed} SUCCESS.frozen_global_beta",
        )
        expected_run = (
            project
            / "runs"
            / "phase2-budgeted-end-to-end"
            / marker["phase2_design_sha256"]
            / f"seed-{seed}"
            / f"job-{array_id}_{task}"
        )
        if run != expected_run:
            raise ValueError(f"seed {seed} run path is not the fixed campaign namespace")
        expected_artifact = (
            project
            / "artifacts"
            / "phase2-budgeted-end-to-end"
            / marker["phase2_design_sha256"]
            / f"seed-{seed}"
            / f"job-{array_id}_{task}"
        )
        artifact_metadata = _artifact_metadata(run, expected=expected_artifact)
        run_rechecks.append((run, expected_artifact, artifact_metadata))
        result, result_snapshot = _json_snapshot(
            run / "phase2-result.json",
            name=f"seed {seed} result",
        )
        rollouts_snapshot = _snapshot_file(
            run / "phase2-result.rollouts.jsonl",
            name=f"seed {seed} rollout sidecar",
            retain_raw=False,
            maximum_bytes=4 * 1024 * 1024 * 1024,
        )
        _manifest, manifest_snapshot = _json_snapshot(
            run / "run-manifest.json",
            name=f"seed {seed} manifest",
        )
        _artifact, artifact_snapshot = _json_snapshot(
            artifact_metadata,
            name=f"seed {seed} artifact metadata",
        )
        _materialization, materialization_snapshot = _json_snapshot(
            run / "artifact-materialization.json",
            name=f"seed {seed} artifact materialization",
        )
        verification, verification_snapshot = _json_snapshot(
            run / "phase2-budgeted-output-verification.json",
            name=f"seed {seed} output verification",
            canonical=True,
        )
        snapshots = {
            "result": result_snapshot,
            "rollouts": rollouts_snapshot,
            "manifest": manifest_snapshot,
            "artifact": artifact_snapshot,
            "materialization": materialization_snapshot,
            "verification": verification_snapshot,
        }
        input_snapshots.extend(
            (snapshot, f"seed {seed} {name}") for name, snapshot in snapshots.items()
        )
        hashes = {name: snapshot.sha256 for name, snapshot in snapshots.items()}
        marker_hashes = {
            "result": marker["phase2_result_sha256"],
            "rollouts": marker["rollouts_sha256"],
            "manifest": marker["manifest_sha256"],
            "artifact": marker["artifact_metadata_sha256"],
            "verification": marker["verification_sha256"],
        }
        if any(hashes[name] != expected for name, expected in marker_hashes.items()):
            raise ValueError(f"seed {seed} SUCCESS marker does not bind all output bytes")

        normalized = dict(normalize_budgeted_end_to_end_seed_result(result))
        _validate_normalized(normalized, seed=seed)
        if verification_snapshot.sha256 != marker["verification_sha256"]:
            raise ValueError(f"seed {seed} verification SHA256 changed")
        identity = _verify_sidecar(
            verification,
            seed=seed,
            task=task,
            row=row,
            marker=marker,
            hashes=hashes,
            normalized=normalized,
        )
        expected_identity = {
            "phase2_design_sha256": marker["phase2_design_sha256"],
            "base_config_hash": marker["base_config_hash"],
            "git_commit": marker["git_commit"],
            "image_sha256": verification["image_sha256"],
            "hf_inventory_sha256": verification["hf_inventory_sha256"],
            "accepted_freeze_aggregate_sha256": marker["freeze_evidence_sha256"],
            "phase2_runtime_contract_sha256": normalized["phase2_runtime_contract_sha256"],
            "frozen_global_beta": frozen_beta,
        }
        if identity != expected_identity:
            raise ValueError(f"seed {seed} result/marker/verifier identities are cross-bound")
        if shared_identity is None:
            shared_identity = identity
            shared_marker = marker
        elif identity != shared_identity:
            raise ValueError(
                "fixed-three runs do not share one design/base/runtime/freeze identity"
            )
        elif any(
            marker[field] != shared_marker[field]
            for field in ("submission_intent_sha256", "submission_ledger_sha256")
        ):
            raise ValueError("fixed-three runs do not share one immutable submission ledger")
        normalized_records.append(normalized)
        marker_controls.append(marker)
        seed_receipts.append(
            {
                "seed": seed,
                "array_task_id": task,
                "slurm_job_id_raw": row["job_id_raw"],
                "run": run.relative_to(project).as_posix(),
                "success_sha256": success_snapshot.sha256,
                "result_sha256": hashes["result"],
                "rollouts_sha256": hashes["rollouts"],
                "run_manifest_sha256": hashes["manifest"],
                "artifact_metadata_sha256": hashes["artifact"],
                "artifact_materialization_sha256": hashes["materialization"],
                "seed_output_verification_sha256": hashes["verification"],
            }
        )

    assert shared_identity is not None and shared_marker is not None
    campaign = (
        project
        / "runs"
        / "phase2-budgeted-end-to-end"
        / str(shared_identity["phase2_design_sha256"])
    )
    ledger = _real_directory(campaign / "submission-ledger", name="submission ledger")
    ledger_verification = _verify_submission_ledger(
        ledger,
        project_root=project,
        repo_root=repository,
        identity=shared_identity,
        marker=shared_marker,
        array_job_id=array_id,
    )
    if (
        isinstance(ledger_verification, tuple)
        and len(ledger_verification) == 2
        and isinstance(ledger_verification[0], Mapping)
        and isinstance(ledger_verification[1], tuple)
    ):
        ledger_result = dict(ledger_verification[0])
        ledger_snapshots = ledger_verification[1]
    else:
        raise ValueError("submission ledger verifier returned an invalid contract")
    if (
        ledger_result.get("status") != "verified"
        or ledger_result.get("array_job_id") != array_id
        or ledger_result.get("phase2_design_sha256") != shared_identity["phase2_design_sha256"]
        or ledger_result.get("ordered_seeds") != list(FIXED_THREE_EXPLORATORY_SEEDS)
        or ledger_result.get("intent_sha256") != shared_marker["submission_intent_sha256"]
        or ledger_result.get("submission_sha256") != shared_marker["submission_ledger_sha256"]
    ):
        raise ValueError("submission ledger does not bind the successful fixed-three array")
    if any(
        marker["submission_intent_sha256"] != ledger_result["intent_sha256"]
        or marker["submission_ledger_sha256"] != ledger_result["submission_sha256"]
        for marker in marker_controls
    ):
        raise ValueError("one or more seed SUCCESS markers cross-bind the submission ledger")
    input_snapshots.extend(
        (snapshot, f"submission ledger file {index}")
        for index, snapshot in enumerate(ledger_snapshots)
    )

    assert publication_code is not None

    aggregate = build_fixed_three_exploratory_aggregate(
        normalized_records,
        bootstrap_seed=FIXED_BOOTSTRAP_SEED,
        bootstrap_resamples=FIXED_BOOTSTRAP_RESAMPLES,
        confidence_level=FIXED_CONFIDENCE_LEVEL,
    )
    validate_fixed_three_exploratory_aggregate(aggregate)
    assert_exploratory_payload_has_no_inferential_fields(aggregate)
    if aggregate.get("aggregation_state") != "complete_descriptive_aggregate":
        raise ValueError("all three admissible seeds are required for descriptive publication")

    output = Path(output_path).absolute()
    if output.suffix != ".json":
        raise ValueError("aggregate output must have a .json filename")
    aggregate_parent = _real_directory(project / "aggregates", name="aggregate root")
    if output.parent != aggregate_parent:
        raise ValueError("aggregate output must be a direct child of the project aggregate root")
    receipt_path = Path(f"{output}.evidence.json")

    aggregate_raw = _canonical_json(aggregate)
    receipt = {
        "schema_version": PUBLICATION_SCHEMA,
        "status": "published",
        "analysis_role": "fixed_three_exploratory_descriptive_only",
        "formal_claim_eligible": False,
        "array_job_id": array_id,
        "ordered_seeds": list(FIXED_THREE_EXPLORATORY_SEEDS),
        "identity": shared_identity,
        "terminal_evidence": {
            "path": terminal_file.relative_to(project).as_posix(),
            "sha256": terminal_digest,
        },
        "submission_ledger": {
            "path": ledger.relative_to(project).as_posix(),
            "intent_sha256": ledger_result["intent_sha256"],
            "submission_sha256": ledger_result["submission_sha256"],
        },
        "seed_evidence": seed_receipts,
        "publication_code": publication_code,
        "aggregate": {
            "filename": output.name,
            "sha256": _sha256(aggregate_raw),
            "size_bytes": len(aggregate_raw),
        },
        "bootstrap": dict(aggregate["bootstrap"]),
    }
    receipt_raw = _canonical_json(receipt)
    final_head = _git_bytes(
        repository,
        "rev-parse",
        "--verify",
        "HEAD",
        name="final Git HEAD query",
    )
    final_status = _git_bytes(
        repository,
        "status",
        "--porcelain",
        "--untracked-files=normal",
        name="final Git cleanliness query",
    )
    if final_head != f"{shared_identity['git_commit']}\n".encode() or final_status:
        raise ValueError("publication checkout changed before atomic publication")
    for run, expected_artifact, metadata_path in run_rechecks:
        _unique_success(_real_directory(run, name="final seed run directory"))
        if _artifact_metadata(run, expected=expected_artifact) != metadata_path:
            raise ValueError("seed artifact link changed before atomic publication")
    for snapshot, name in input_snapshots:
        _verify_snapshot_unchanged(snapshot, name=name)
    _publish_recoverable_pair(output, aggregate_raw, receipt_path, receipt_raw)
    return aggregate, receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("runs", type=Path, nargs=5)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--terminal-evidence", type=Path, required=True)
    parser.add_argument("--terminal-evidence-sha256", required=True)
    parser.add_argument("--array-job-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    aggregate, receipt = write_budgeted_end_to_end_aggregate(
        arguments.project_root,
        arguments.repo_root,
        arguments.terminal_evidence,
        arguments.runs,
        arguments.output,
        terminal_evidence_sha256=arguments.terminal_evidence_sha256,
        array_job_id=arguments.array_job_id,
    )
    print(
        json.dumps(
            {
                "status": "published",
                "analysis_role": aggregate["analysis_role"],
                "formal_claim_eligible": False,
                "output": str(arguments.output),
                "output_sha256": receipt["aggregate"]["sha256"],
                "receipt": f"{arguments.output}.evidence.json",
            },
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
