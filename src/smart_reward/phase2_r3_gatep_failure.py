"""Append-only, explicitly non-authorizing Gate-P failure lineage.

Gate-P can fail before the operational bundle or runtime receipt exists.  The
successful scheduler terminalizer cannot close such an attempt because its
inputs are intentionally success-only.  This module preserves that gap as
evidence without promoting it into authority:

* a failure receipt binds one terminal failed allocation row, the original
  raw ``sacct`` bytes, stdout/stderr, and the complete pre-receipt attempt
  inventory;
* a later attempt-lineage artifact binds the previous receipt by both file and
  semantic SHA-256 and derives the next attempt index; and
* neither artifact authorizes Gate-P, primary training, retry, or science
  changes.

Publication uses the existing canonical/no-overwrite artifact transport and
the existing claim-free ``sacct`` parser.  Existing attempt bytes are never
rewritten.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

from .phase2_r3_artifacts import (
    decode_canonical_json_bytes,
    publish_canonical_artifact,
    read_canonical_artifact,
)
from .phase2_r3_terminal import (
    _publish_immutable_bytes,
    inspect_sacct_terminal_bytes,
    sacct_terminal_command,
)

GATE_P_FAILURE_RECEIPT_SCHEMA: Final = "phase2-recovery-r3-gate-p-failure-receipt/v1"
GATE_P_FAILURE_RECEIPT_ROLE: Final = "terminal_failed_gate_p_attempt_evidence_non_authorizing"
GATE_P_ATTEMPT_LINEAGE_SCHEMA: Final = "phase2-recovery-r3-gate-p-attempt-lineage/v1"
GATE_P_ATTEMPT_LINEAGE_ROLE: Final = "pre_submit_gate_p_failure_lineage_non_authorizing"
GATE_P_CAMPAIGN_IDENTITY_SCHEMA: Final = "phase2-recovery-r3-gate-p-campaign-identity/v1"

FAILURE_EVIDENCE_DIRECTORY: Final = "failure-evidence"
FAILURE_RAW_SACCT_FILENAME: Final = "raw-sacct.psv"
FAILURE_RECEIPT_FILENAME: Final = "gatep-failure-receipt.json"
ATTEMPT_LINEAGE_FILENAME: Final = "gatep-attempt-lineage.json"

_GATEP_RELATIVE = Path("runs/phase2-recovery-r3/gatep")
_ATTEMPT_RE = re.compile(r"gatep-attempt-(?!000)([0-9]{3})\Z")
_IDENTITY_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_GIT_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
_UTC_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z\Z")
_POSITIVE_JOB_ID_RE = re.compile(r"[1-9][0-9]*\Z")
_MAX_INVENTORY_FILE_BYTES: Final = 256 * 1024 * 1024
_MAX_INVENTORY_ENTRIES: Final = 4096
_TERMINAL_FAILURE_STATES: Final = frozenset(
    {
        "BOOT_FAIL",
        "CANCELLED",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "REQUEUED",
        "REVOKED",
        "SPECIAL_EXIT",
        "STOPPED",
        "TIMEOUT",
    }
)
FAILURE_STAGES: Final = frozenset(
    {
        "submission_preflight",
        "gate0_revalidation",
        "gate1_revalidation",
        "train_materialization",
        "profile_preparation",
        "bt_mle_profile",
        "prorm_plus_profile",
        "operational_bundle_publication",
        "runtime_receipt_publication",
        "unknown_pre_authorization",
    }
)
_SOURCE_BINDING_FIELDS: Final = frozenset(
    {
        "source_git_commit",
        "gate0_file_sha256",
        "gate1_file_sha256",
        "source_test_receipt_file_sha256",
        "science_config_file_sha256",
        "container_file_sha256",
        "campaign_identity_sha256",
    }
)
_NO_AUTHORITY: Final = {
    "authorizes_gate_p_success": False,
    "authorizes_retry": False,
    "authorizes_primary": False,
    "authorizes_science_change": False,
    "reusable_training_state": False,
}
_SOURCE_BINDING_EVIDENCE: Final = {
    "classification": "operator_declared_historical_hashes_bound_as_identity_only",
    "mechanically_reverified_source_files_by_failure_receipt": False,
}


def _semantic_sha256(value: Mapping[str, object]) -> str:
    encoded = json.dumps(
        dict(value),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _digest(value: object, *, name: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _git_commit(value: object, *, name: str = "source Git commit") -> str:
    if type(value) is not str or _GIT_COMMIT_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase forty-character Git commit")
    return value


def _job_id(value: object, *, name: str = "Slurm job ID") -> str:
    if type(value) is not str or _POSITIVE_JOB_ID_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a positive decimal Slurm job ID")
    return value


def _utc(value: object) -> str:
    if type(value) is not str or _UTC_RE.fullmatch(value) is None:
        raise ValueError("captured_at_utc must use canonical UTC second precision")
    parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise ValueError("captured_at_utc is not canonical")
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def _exact_mapping(
    value: object,
    fields: set[str] | frozenset[str],
    *,
    name: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != set(fields):
        raise ValueError(f"{name} has an invalid closed field set")
    return dict(value)


def _canonical_directory(path: Path, *, name: str) -> Path:
    absolute = path.absolute()
    info = absolute.lstat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or absolute.is_symlink()
        or absolute.resolve(strict=True) != absolute
    ):
        raise ValueError(f"{name} must be a canonical real directory")
    return absolute


def _contained(path: Path, root: Path, *, name: str) -> Path:
    try:
        return path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{name} escaped its canonical root") from error


def _attempt_descriptor(project_root: Path, attempt_root: Path) -> dict[str, object]:
    project = _canonical_directory(project_root, name="project root")
    attempt = _canonical_directory(attempt_root, name="Gate-P attempt root")
    gatep_root = project / _GATEP_RELATIVE
    relative_to_gatep = _contained(attempt, gatep_root, name="Gate-P attempt root")
    if len(relative_to_gatep.parts) != 2:
        raise ValueError("Gate-P attempt root must have identity/attempt depth")
    identity, attempt_name = relative_to_gatep.parts
    match = _ATTEMPT_RE.fullmatch(attempt_name)
    if _IDENTITY_RE.fullmatch(identity) is None or match is None:
        raise ValueError("Gate-P attempt root has an invalid identity or attempt index")
    return {
        "project_relative": attempt.relative_to(project).as_posix(),
        "identity": identity,
        "attempt_index": int(match.group(1)),
    }


def _source_bindings(
    *,
    source_git_commit: str,
    gate0_file_sha256: str,
    gate1_file_sha256: str,
    source_test_receipt_file_sha256: str,
    science_config_file_sha256: str,
    container_file_sha256: str,
) -> dict[str, object]:
    unsigned: dict[str, object] = {
        "schema_version": GATE_P_CAMPAIGN_IDENTITY_SCHEMA,
        "source_git_commit": _git_commit(source_git_commit),
        "gate0_file_sha256": _digest(gate0_file_sha256, name="Gate-0 file SHA-256"),
        "gate1_file_sha256": _digest(gate1_file_sha256, name="Gate-1 file SHA-256"),
        "source_test_receipt_file_sha256": _digest(
            source_test_receipt_file_sha256,
            name="source-test receipt file SHA-256",
        ),
        "science_config_file_sha256": _digest(
            science_config_file_sha256,
            name="science config file SHA-256",
        ),
        "container_file_sha256": _digest(
            container_file_sha256,
            name="container file SHA-256",
        ),
    }
    return {
        **unsigned,
        "campaign_identity_sha256": _semantic_sha256(unsigned),
    }


def derive_gate_p_campaign_identity(
    *,
    source_git_commit: str,
    gate0_file_sha256: str,
    gate1_file_sha256: str,
    source_test_receipt_file_sha256: str,
    science_config_file_sha256: str,
    container_file_sha256: str,
) -> dict[str, object]:
    """Return the content identity used as the Gate-P attempt parent name."""

    return _source_bindings(
        source_git_commit=source_git_commit,
        gate0_file_sha256=gate0_file_sha256,
        gate1_file_sha256=gate1_file_sha256,
        source_test_receipt_file_sha256=source_test_receipt_file_sha256,
        science_config_file_sha256=science_config_file_sha256,
        container_file_sha256=container_file_sha256,
    )


def _validate_source_bindings(value: object) -> dict[str, object]:
    fields = {*_SOURCE_BINDING_FIELDS, "schema_version"}
    source = _exact_mapping(value, fields, name="Gate-P campaign source bindings")
    expected = _source_bindings(
        source_git_commit=_git_commit(source["source_git_commit"]),
        gate0_file_sha256=_digest(source["gate0_file_sha256"], name="Gate-0 file SHA-256"),
        gate1_file_sha256=_digest(source["gate1_file_sha256"], name="Gate-1 file SHA-256"),
        source_test_receipt_file_sha256=_digest(
            source["source_test_receipt_file_sha256"],
            name="source-test receipt file SHA-256",
        ),
        science_config_file_sha256=_digest(
            source["science_config_file_sha256"],
            name="science config file SHA-256",
        ),
        container_file_sha256=_digest(
            source["container_file_sha256"],
            name="container file SHA-256",
        ),
    )
    if source != expected:
        raise ValueError("Gate-P campaign identity SHA-256 is invalid")
    return source


def _file_record(path: Path, *, relative: str, maximum_bytes: int) -> dict[str, object]:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or path.is_symlink():
        raise ValueError(f"attempt inventory file is unsafe: {relative}")
    if before.st_size > maximum_bytes:
        raise ValueError(f"attempt inventory file is too large: {relative}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
        after_open = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = path.lstat()
    identities = {
        (before.st_dev, before.st_ino),
        (opened.st_dev, opened.st_ino),
        (after_open.st_dev, after_open.st_ino),
        (after.st_dev, after.st_ino),
    }
    if (
        len(identities) != 1
        or before.st_size != opened.st_size
        or opened.st_size != after_open.st_size
        or after_open.st_size != after.st_size
        or size != after.st_size
        or not stat.S_ISREG(after.st_mode)
        or path.is_symlink()
    ):
        raise ValueError(f"attempt inventory file changed while reading: {relative}")
    return {
        "relative": relative,
        "kind": "file",
        "mode": f"{stat.S_IMODE(after.st_mode):04o}",
        "size_bytes": size,
        "sha256": digest.hexdigest(),
    }


def _attempt_inventory(attempt_root: Path) -> list[dict[str, object]]:
    attempt = _canonical_directory(attempt_root, name="Gate-P attempt root")
    records: list[dict[str, object]] = []
    paths = sorted(attempt.rglob("*"), key=lambda item: item.relative_to(attempt).as_posix())
    if len(paths) > _MAX_INVENTORY_ENTRIES:
        raise ValueError("Gate-P failure attempt inventory is unreasonably large")
    for path in paths:
        relative_path = path.relative_to(attempt)
        if relative_path.parts[0] == FAILURE_EVIDENCE_DIRECTORY:
            continue
        relative = relative_path.as_posix()
        info = path.lstat()
        if path.is_symlink():
            raise ValueError(f"Gate-P failure attempt contains a symlink: {relative}")
        if stat.S_ISDIR(info.st_mode):
            records.append(
                {
                    "relative": relative,
                    "kind": "directory",
                    "mode": f"{stat.S_IMODE(info.st_mode):04o}",
                    "size_bytes": 0,
                    "sha256": None,
                }
            )
        elif stat.S_ISREG(info.st_mode):
            records.append(
                _file_record(
                    path,
                    relative=relative,
                    maximum_bytes=_MAX_INVENTORY_FILE_BYTES,
                )
            )
        else:
            raise ValueError(f"Gate-P failure attempt contains a special file: {relative}")
    return records


def _inventory_file(
    inventory: Sequence[Mapping[str, object]],
    *,
    relative: str,
    name: str,
) -> dict[str, object]:
    matches = [dict(record) for record in inventory if record.get("relative") == relative]
    if len(matches) != 1 or matches[0].get("kind") != "file":
        raise ValueError(f"{name} is missing from the failure attempt inventory")
    return matches[0]


def _ensure_failure_directory(attempt_root: Path) -> Path:
    failure = attempt_root / FAILURE_EVIDENCE_DIRECTORY
    if failure.exists() or failure.is_symlink():
        return _canonical_directory(failure, name="Gate-P failure evidence directory")
    os.mkdir(failure, mode=0o750)
    if os.name == "posix":
        os.chmod(failure, 0o750)
    return _canonical_directory(failure, name="Gate-P failure evidence directory")


def _stable_input_bytes(path: Path, *, maximum_bytes: int, name: str) -> bytes:
    absolute = path.absolute()
    parent = _canonical_directory(absolute.parent, name=f"{name} parent")
    before_parent = parent.stat()
    before = absolute.lstat()
    if not stat.S_ISREG(before.st_mode) or absolute.is_symlink() or before.st_size > maximum_bytes:
        raise ValueError(f"{name} must be one bounded regular non-symlink file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    descriptor = os.open(absolute, flags)
    try:
        opened = os.fstat(descriptor)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after_open = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = absolute.lstat()
    after_parent = parent.stat()
    if (
        len(
            {
                (before.st_dev, before.st_ino),
                (opened.st_dev, opened.st_ino),
                (after_open.st_dev, after_open.st_ino),
                (after.st_dev, after.st_ino),
            }
        )
        != 1
        or before.st_size != opened.st_size
        or opened.st_size != after_open.st_size
        or after_open.st_size != after.st_size
        or (before_parent.st_dev, before_parent.st_ino)
        != (after_parent.st_dev, after_parent.st_ino)
    ):
        raise ValueError(f"{name} changed while it was being read")
    raw = b"".join(chunks)
    if len(raw) != after.st_size:
        raise ValueError(f"{name} byte count changed")
    return raw


def _terminal_failure_state(state: object) -> str:
    if type(state) is not str or not state:
        raise ValueError("failed Gate-P sacct state is invalid")
    base = state.split(maxsplit=1)[0].removesuffix("+")
    if base not in _TERMINAL_FAILURE_STATES:
        raise ValueError("Gate-P failure receipt requires an exact terminal failure state")
    return base


def _publication_state(inventory: Sequence[Mapping[str, object]]) -> dict[str, bool]:
    names = {str(record["relative"]) for record in inventory if record.get("kind") == "file"}
    return {
        "profile_allocation_intent_present": "profile-allocation-intent.json" in names,
        "operational_bundle_present": "gatep-operational-bundle.json" in names,
        "runtime_receipt_present": "profile-runtime-receipt.json" in names,
        "successful_terminal_evidence_present": any(
            name.startswith("terminal-evidence/") for name in names
        ),
    }


def _validate_inventory(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise ValueError("partial inventory must be a list")
    result: list[dict[str, object]] = []
    previous = ""
    for index, item in enumerate(value):
        record = _exact_mapping(
            item,
            {"relative", "kind", "mode", "size_bytes", "sha256"},
            name=f"partial inventory record {index}",
        )
        relative = record["relative"]
        if (
            type(relative) is not str
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or relative <= previous
            or relative.startswith(f"{FAILURE_EVIDENCE_DIRECTORY}/")
        ):
            raise ValueError("partial inventory paths are invalid or unordered")
        previous = relative
        if (
            record["kind"] not in {"file", "directory"}
            or type(record["mode"]) is not str
            or re.fullmatch(r"[0-7]{4}", record["mode"]) is None
            or type(record["size_bytes"]) is not int
            or record["size_bytes"] < 0
        ):
            raise ValueError("partial inventory record metadata is invalid")
        if record["kind"] == "file":
            _digest(record["sha256"], name="partial inventory file SHA-256")
        elif record["sha256"] is not None or record["size_bytes"] != 0:
            raise ValueError("partial inventory directory record is invalid")
        result.append(record)
    return result


def _validate_failure_payload(value: object) -> dict[str, object]:
    payload = _exact_mapping(
        value,
        {
            "schema_version",
            "role",
            "captured_at_utc",
            "source",
            "source_binding_evidence",
            "attempt",
            "scheduler",
            "logs",
            "partial_inventory",
            "failure",
            "authority",
            "receipt_sha256",
        },
        name="Gate-P failure receipt",
    )
    if (
        payload["schema_version"] != GATE_P_FAILURE_RECEIPT_SCHEMA
        or payload["role"] != GATE_P_FAILURE_RECEIPT_ROLE
        or payload["authority"] != _NO_AUTHORITY
        or payload["source_binding_evidence"] != _SOURCE_BINDING_EVIDENCE
    ):
        raise ValueError("Gate-P failure receipt schema/role/authority/source evidence is invalid")
    _utc(payload["captured_at_utc"])
    source = _validate_source_bindings(payload["source"])
    attempt = _exact_mapping(
        payload["attempt"],
        {"project_relative", "identity", "attempt_index"},
        name="Gate-P failure attempt",
    )
    if (
        type(attempt["project_relative"]) is not str
        or not attempt["project_relative"]
        or _IDENTITY_RE.fullmatch(str(attempt["identity"])) is None
        or type(attempt["attempt_index"]) is not int
        or not 1 <= attempt["attempt_index"] <= 999
    ):
        raise ValueError("Gate-P failure attempt identity is invalid")
    scheduler = _exact_mapping(
        payload["scheduler"],
        {"job_id", "locked_command", "raw_sacct", "parsed_sacct"},
        name="Gate-P failure scheduler evidence",
    )
    job_id = _job_id(scheduler["job_id"])
    if scheduler["locked_command"] != list(sacct_terminal_command(job_id)):
        raise ValueError("Gate-P failure sacct command is not locked")
    raw_binding = _exact_mapping(
        scheduler["raw_sacct"],
        {"project_relative", "sha256", "size_bytes"},
        name="Gate-P failure raw sacct binding",
    )
    if (
        type(raw_binding["project_relative"]) is not str
        or type(raw_binding["size_bytes"]) is not int
        or raw_binding["size_bytes"] < 1
    ):
        raise ValueError("Gate-P failure raw sacct binding is invalid")
    _digest(raw_binding["sha256"], name="Gate-P failure raw sacct SHA-256")
    parsed = _exact_mapping(
        scheduler["parsed_sacct"],
        {
            "schema_version",
            "formal_claim_eligible",
            "locked_invocation_flags",
            "locked_fields",
            "raw_sacct",
            "row",
            "inspection_sha256",
        },
        name="Gate-P failure parsed sacct evidence",
    )
    row = _exact_mapping(
        parsed["row"],
        {
            "job_id",
            "job_id_raw",
            "state",
            "exit_code",
            "derived_exit_code",
            "cluster",
            "account",
            "partition",
            "qos",
            "n_nodes",
            "n_cpus",
            "req_tres_raw",
            "alloc_tres_raw",
            "req_tres",
            "alloc_tres",
            "elapsed_seconds",
        },
        name="Gate-P failure parsed sacct row",
    )
    if (
        row["job_id"] != job_id
        or row["job_id_raw"] != job_id
        or row["cluster"] != "hpc4"
        or row["account"] != "sigroup"
    ):
        raise ValueError("Gate-P failure scheduler identity differs from HPC4 attempt")
    state = _terminal_failure_state(row["state"])
    inventory = _validate_inventory(payload["partial_inventory"])
    logs = _exact_mapping(payload["logs"], {"stdout", "stderr"}, name="Gate-P failure logs")
    for name in ("stdout", "stderr"):
        record = _exact_mapping(
            logs[name],
            {"relative", "kind", "mode", "size_bytes", "sha256"},
            name=f"Gate-P {name} log binding",
        )
        expected_relative = f"logs/gatep-{job_id}.{'out' if name == 'stdout' else 'err'}"
        if (
            record.get("relative") != expected_relative
            or record.get("kind") != "file"
            or record not in inventory
        ):
            raise ValueError(f"Gate-P {name} log is not inventory-bound")
    failure = _exact_mapping(
        payload["failure"],
        {"stage", "classification_source", "terminal_state", "publication_state"},
        name="Gate-P failure classification",
    )
    if (
        failure["stage"] not in FAILURE_STAGES
        or failure["classification_source"]
        != "operator_stage_bound_to_terminal_sacct_logs_and_partial_inventory"
        or failure["terminal_state"] != state
        or failure["publication_state"] != _publication_state(inventory)
    ):
        raise ValueError("Gate-P failure classification is invalid")
    _digest(payload["receipt_sha256"], name="Gate-P failure receipt semantic SHA-256")
    unsigned = dict(payload)
    observed_receipt_sha = unsigned.pop("receipt_sha256")
    if observed_receipt_sha != _semantic_sha256(unsigned):
        raise ValueError("Gate-P failure receipt semantic SHA-256 is invalid")
    # Keep a live reference to avoid accepting a source mapping with unused fields.
    if source["campaign_identity_sha256"] != payload["source"]["campaign_identity_sha256"]:
        raise ValueError("Gate-P failure source binding changed during validation")
    return payload


def _failure_receipt_from_bytes(
    artifact_path: Path,
    raw: bytes,
    *,
    file_sha256: str,
) -> GatePFailureReceipt:
    payload = _validate_failure_payload(decode_canonical_json_bytes(raw))
    result = GatePFailureReceipt(
        artifact_path=artifact_path,
        file_sha256=file_sha256,
        size_bytes=len(raw),
        receipt_sha256=str(payload["receipt_sha256"]),
        campaign_identity_sha256=str(payload["source"]["campaign_identity_sha256"]),
        source_git_commit=str(payload["source"]["source_git_commit"]),
        attempt_index=int(payload["attempt"]["attempt_index"]),
        job_id=str(payload["scheduler"]["job_id"]),
        _canonical_bytes=raw,
    )
    result.validate_integrity()
    return result


@dataclass(frozen=True, slots=True)
class GatePFailureReceipt:
    """Revalidated non-authorizing record of one failed Gate-P attempt."""

    artifact_path: Path
    file_sha256: str
    size_bytes: int
    receipt_sha256: str
    campaign_identity_sha256: str
    source_git_commit: str
    attempt_index: int
    job_id: str
    _canonical_bytes: bytes = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.artifact_path.is_absolute():
            raise ValueError("Gate-P failure receipt path must be absolute")
        _digest(self.file_sha256, name="Gate-P failure receipt file SHA-256")
        _digest(self.receipt_sha256, name="Gate-P failure receipt semantic SHA-256")
        _digest(self.campaign_identity_sha256, name="Gate-P campaign identity SHA-256")
        _git_commit(self.source_git_commit)
        _job_id(self.job_id)
        if type(self.attempt_index) is not int or not 1 <= self.attempt_index <= 999:
            raise ValueError("Gate-P failure receipt attempt index is invalid")
        if type(self.size_bytes) is not int or self.size_bytes < 1:
            raise ValueError("Gate-P failure receipt size is invalid")
        if (
            type(self._canonical_bytes) is not bytes
            or len(self._canonical_bytes) != self.size_bytes
            or _sha256(self._canonical_bytes) != self.file_sha256
        ):
            raise ValueError("Gate-P failure receipt byte binding is invalid")

    @property
    def payload(self) -> dict[str, object]:
        return _validate_failure_payload(decode_canonical_json_bytes(self._canonical_bytes))

    def validate_integrity(self) -> None:
        self.__post_init__()
        transport = read_canonical_artifact(
            self.artifact_path,
            expected_file_sha256=self.file_sha256,
        )
        if transport.canonical_bytes != self._canonical_bytes:
            raise ValueError("Gate-P failure receipt bytes changed")
        payload = _validate_failure_payload(transport.payload)
        exact = {
            "receipt_sha256": payload["receipt_sha256"],
            "campaign_identity_sha256": payload["source"]["campaign_identity_sha256"],
            "source_git_commit": payload["source"]["source_git_commit"],
            "attempt_index": payload["attempt"]["attempt_index"],
            "job_id": payload["scheduler"]["job_id"],
        }
        for name, expected in exact.items():
            if getattr(self, name) != expected:
                raise ValueError(f"Gate-P failure receipt {name} is inconsistent")


def publish_gate_p_failure_receipt(
    *,
    project_root: str | os.PathLike[str],
    attempt_root: str | os.PathLike[str],
    raw_sacct_path: str | os.PathLike[str],
    stdout_path: str | os.PathLike[str],
    stderr_path: str | os.PathLike[str],
    job_id: str,
    failure_stage: str,
    source_git_commit: str,
    gate0_file_sha256: str,
    gate1_file_sha256: str,
    source_test_receipt_file_sha256: str,
    science_config_file_sha256: str,
    container_file_sha256: str,
    captured_at_utc: str | None = None,
) -> GatePFailureReceipt:
    """Publish one no-overwrite failure receipt and copied raw sacct evidence."""

    project = _canonical_directory(Path(project_root), name="project root")
    attempt = _canonical_directory(Path(attempt_root), name="Gate-P attempt root")
    attempt_descriptor = _attempt_descriptor(project, attempt)
    checked_job_id = _job_id(job_id)
    if failure_stage not in FAILURE_STAGES:
        raise ValueError("Gate-P failure stage is not a closed supported value")
    source = _source_bindings(
        source_git_commit=source_git_commit,
        gate0_file_sha256=gate0_file_sha256,
        gate1_file_sha256=gate1_file_sha256,
        source_test_receipt_file_sha256=source_test_receipt_file_sha256,
        science_config_file_sha256=science_config_file_sha256,
        container_file_sha256=container_file_sha256,
    )
    inventory = _attempt_inventory(attempt)
    expected_stdout = (attempt / "logs" / f"gatep-{checked_job_id}.out").absolute()
    expected_stderr = (attempt / "logs" / f"gatep-{checked_job_id}.err").absolute()
    if (
        Path(stdout_path).absolute() != expected_stdout
        or Path(stderr_path).absolute() != expected_stderr
    ):
        raise ValueError("Gate-P stdout/stderr paths must be the exact Slurm attempt logs")
    stdout_record = _inventory_file(
        inventory,
        relative=f"logs/gatep-{checked_job_id}.out",
        name="Gate-P stdout log",
    )
    stderr_record = _inventory_file(
        inventory,
        relative=f"logs/gatep-{checked_job_id}.err",
        name="Gate-P stderr log",
    )
    raw = _stable_input_bytes(
        Path(raw_sacct_path),
        maximum_bytes=128 * 1024,
        name="Gate-P raw sacct input",
    )
    raw_sha = _sha256(raw)
    inspection = inspect_sacct_terminal_bytes(raw, expected_raw_sha256=raw_sha)
    if (
        inspection.row.job_id != checked_job_id
        or inspection.row.job_id_raw != checked_job_id
        or inspection.row.cluster != "hpc4"
        or inspection.row.account != "sigroup"
    ):
        raise ValueError("Gate-P raw sacct row does not identify the failed HPC4 job")
    terminal_state = _terminal_failure_state(inspection.row.state)

    failure_directory = _ensure_failure_directory(attempt)
    receipt_path = failure_directory / FAILURE_RECEIPT_FILENAME
    if receipt_path.exists() or receipt_path.is_symlink():
        raise FileExistsError("refusing to overwrite a Gate-P failure receipt")
    retained_raw = failure_directory / FAILURE_RAW_SACCT_FILENAME
    if retained_raw.exists() or retained_raw.is_symlink():
        retained = _stable_input_bytes(
            retained_raw,
            maximum_bytes=128 * 1024,
            name="retained Gate-P raw sacct",
        )
        if retained != raw:
            raise ValueError("retained Gate-P raw sacct bytes differ")
    else:
        _publish_immutable_bytes(retained_raw, raw)
    retained_raw_relative = retained_raw.relative_to(project).as_posix()
    unsigned: dict[str, object] = {
        "schema_version": GATE_P_FAILURE_RECEIPT_SCHEMA,
        "role": GATE_P_FAILURE_RECEIPT_ROLE,
        "captured_at_utc": _utc(captured_at_utc or _utc_now()),
        "source": source,
        "source_binding_evidence": dict(_SOURCE_BINDING_EVIDENCE),
        "attempt": attempt_descriptor,
        "scheduler": {
            "job_id": checked_job_id,
            "locked_command": list(sacct_terminal_command(checked_job_id)),
            "raw_sacct": {
                "project_relative": retained_raw_relative,
                "sha256": raw_sha,
                "size_bytes": len(raw),
            },
            "parsed_sacct": inspection.to_dict(),
        },
        "logs": {
            "stdout": stdout_record,
            "stderr": stderr_record,
        },
        "partial_inventory": inventory,
        "failure": {
            "stage": failure_stage,
            "classification_source": (
                "operator_stage_bound_to_terminal_sacct_logs_and_partial_inventory"
            ),
            "terminal_state": terminal_state,
            "publication_state": _publication_state(inventory),
        },
        "authority": dict(_NO_AUTHORITY),
    }
    payload = {
        **unsigned,
        "receipt_sha256": _semantic_sha256(unsigned),
    }
    _validate_failure_payload(payload)
    transport = publish_canonical_artifact(receipt_path, payload)
    return reopen_gate_p_failure_receipt(
        receipt_path,
        project_root=project,
        expected_file_sha256=transport.file_sha256,
    )


def reopen_gate_p_failure_receipt(
    artifact_path: str | os.PathLike[str],
    *,
    project_root: str | os.PathLike[str],
    expected_file_sha256: str,
) -> GatePFailureReceipt:
    """Reopen a receipt and mechanically revalidate every retained byte."""

    project = _canonical_directory(Path(project_root), name="project root")
    path = Path(artifact_path).absolute()
    if path.name != FAILURE_RECEIPT_FILENAME or path.parent.name != FAILURE_EVIDENCE_DIRECTORY:
        raise ValueError("Gate-P failure receipt is outside its fixed attempt path")
    attempt = _canonical_directory(path.parent.parent, name="Gate-P attempt root")
    descriptor = _attempt_descriptor(project, attempt)
    transport = read_canonical_artifact(
        path,
        expected_file_sha256=_digest(
            expected_file_sha256,
            name="expected Gate-P failure receipt file SHA-256",
        ),
    )
    payload = _validate_failure_payload(transport.payload)
    if payload["attempt"] != descriptor:
        raise ValueError("Gate-P failure receipt attempt path differs")
    failure_entries = {child.name for child in path.parent.iterdir()}
    if failure_entries != {FAILURE_RAW_SACCT_FILENAME, FAILURE_RECEIPT_FILENAME}:
        raise ValueError("Gate-P failure evidence directory is not append-only closed")
    raw_binding = payload["scheduler"]["raw_sacct"]
    retained_raw = path.parent / FAILURE_RAW_SACCT_FILENAME
    expected_raw_relative = retained_raw.relative_to(project).as_posix()
    if raw_binding["project_relative"] != expected_raw_relative:
        raise ValueError("Gate-P failure raw sacct path binding differs")
    raw = _stable_input_bytes(
        retained_raw,
        maximum_bytes=128 * 1024,
        name="retained Gate-P raw sacct",
    )
    if _sha256(raw) != raw_binding["sha256"] or len(raw) != raw_binding["size_bytes"]:
        raise ValueError("retained Gate-P raw sacct bytes changed")
    inspection = inspect_sacct_terminal_bytes(
        raw,
        expected_raw_sha256=str(raw_binding["sha256"]),
    )
    if inspection.to_dict() != payload["scheduler"]["parsed_sacct"]:
        raise ValueError("Gate-P parsed sacct evidence differs from retained raw bytes")
    if _attempt_inventory(attempt) != payload["partial_inventory"]:
        raise ValueError("Gate-P failure attempt changed after receipt publication")
    result = _failure_receipt_from_bytes(
        path,
        transport.canonical_bytes,
        file_sha256=transport.file_sha256,
    )
    return result


def _predecessor_reference(
    receipt: GatePFailureReceipt,
    *,
    project_root: Path,
) -> dict[str, object]:
    receipt.validate_integrity()
    payload = receipt.payload
    relative = _contained(
        receipt.artifact_path,
        project_root,
        name="predecessor Gate-P failure receipt",
    ).as_posix()
    return {
        "project_relative": relative,
        "schema_version": payload["schema_version"],
        "role": payload["role"],
        "file_sha256": receipt.file_sha256,
        "receipt_sha256": receipt.receipt_sha256,
        "source_git_commit": receipt.source_git_commit,
        "campaign_identity_sha256": receipt.campaign_identity_sha256,
        "attempt_project_relative": payload["attempt"]["project_relative"],
        "attempt_identity": payload["attempt"]["identity"],
        "attempt_index": receipt.attempt_index,
        "job_id": receipt.job_id,
    }


def plan_next_gate_p_attempt(
    *,
    project_root: str | os.PathLike[str],
    predecessor_receipt: GatePFailureReceipt,
    source_git_commit: str,
    gate0_file_sha256: str,
    gate1_file_sha256: str,
    source_test_receipt_file_sha256: str,
    science_config_file_sha256: str,
    container_file_sha256: str,
) -> dict[str, object]:
    """Derive the exact identity/index before creating the next attempt root."""

    if type(predecessor_receipt) is not GatePFailureReceipt:
        raise TypeError("predecessor_receipt must be an exact GatePFailureReceipt")
    project = _canonical_directory(Path(project_root), name="project root")
    predecessor_receipt = reopen_gate_p_failure_receipt(
        predecessor_receipt.artifact_path,
        project_root=project,
        expected_file_sha256=predecessor_receipt.file_sha256,
    )
    current = _source_bindings(
        source_git_commit=source_git_commit,
        gate0_file_sha256=gate0_file_sha256,
        gate1_file_sha256=gate1_file_sha256,
        source_test_receipt_file_sha256=source_test_receipt_file_sha256,
        science_config_file_sha256=science_config_file_sha256,
        container_file_sha256=container_file_sha256,
    )
    same_campaign = (
        predecessor_receipt.campaign_identity_sha256 == current["campaign_identity_sha256"]
    )
    attempt_index = predecessor_receipt.attempt_index + 1 if same_campaign else 1
    if attempt_index > 999:
        raise ValueError("Gate-P attempt index exceeds the fixed three-digit namespace")
    return {
        "campaign_identity_sha256": current["campaign_identity_sha256"],
        "attempt_index": attempt_index,
        "attempt_name": f"gatep-attempt-{attempt_index:03d}",
        "predecessor_relation": (
            "same_campaign_next_attempt"
            if same_campaign
            else "new_campaign_attempt_001_with_cross_campaign_predecessor"
        ),
        "predecessor_receipt_file_sha256": predecessor_receipt.file_sha256,
        "predecessor_receipt_sha256": predecessor_receipt.receipt_sha256,
    }


def _validate_lineage_payload(value: object) -> dict[str, object]:
    payload = _exact_mapping(
        value,
        {
            "schema_version",
            "role",
            "current_campaign",
            "attempt",
            "predecessor_failure",
            "authority",
            "lineage_sha256",
        },
        name="Gate-P attempt lineage",
    )
    if (
        payload["schema_version"] != GATE_P_ATTEMPT_LINEAGE_SCHEMA
        or payload["role"] != GATE_P_ATTEMPT_LINEAGE_ROLE
        or payload["authority"] != _NO_AUTHORITY
    ):
        raise ValueError("Gate-P attempt lineage schema/role/authority is invalid")
    current = _validate_source_bindings(payload["current_campaign"])
    attempt = _exact_mapping(
        payload["attempt"],
        {
            "project_relative",
            "identity",
            "attempt_index",
            "predecessor_relation",
        },
        name="lineaged Gate-P attempt",
    )
    predecessor = _exact_mapping(
        payload["predecessor_failure"],
        {
            "project_relative",
            "schema_version",
            "role",
            "file_sha256",
            "receipt_sha256",
            "source_git_commit",
            "campaign_identity_sha256",
            "attempt_project_relative",
            "attempt_identity",
            "attempt_index",
            "job_id",
        },
        name="predecessor Gate-P failure receipt reference",
    )
    if (
        predecessor["schema_version"] != GATE_P_FAILURE_RECEIPT_SCHEMA
        or predecessor["role"] != GATE_P_FAILURE_RECEIPT_ROLE
    ):
        raise ValueError("predecessor Gate-P failure receipt schema/role is invalid")
    for name in ("file_sha256", "receipt_sha256", "campaign_identity_sha256"):
        _digest(predecessor[name], name=f"predecessor {name}")
    _git_commit(predecessor["source_git_commit"], name="predecessor source Git commit")
    _job_id(predecessor["job_id"], name="predecessor Slurm job ID")
    if (
        type(predecessor["project_relative"]) is not str
        or type(predecessor["attempt_project_relative"]) is not str
        or _IDENTITY_RE.fullmatch(str(predecessor["attempt_identity"])) is None
        or type(predecessor["attempt_index"]) is not int
        or not 1 <= predecessor["attempt_index"] <= 999
    ):
        raise ValueError("predecessor Gate-P failure attempt reference is invalid")
    same_campaign = predecessor["campaign_identity_sha256"] == current["campaign_identity_sha256"]
    expected_index = predecessor["attempt_index"] + 1 if same_campaign else 1
    expected_relation = (
        "same_campaign_next_attempt"
        if same_campaign
        else "new_campaign_attempt_001_with_cross_campaign_predecessor"
    )
    if (
        type(attempt["project_relative"]) is not str
        or attempt["identity"] != current["campaign_identity_sha256"]
        or attempt["attempt_index"] != expected_index
        or attempt["predecessor_relation"] != expected_relation
        or expected_index > 999
    ):
        raise ValueError("Gate-P attempt lineage index/identity relation is invalid")
    _digest(payload["lineage_sha256"], name="Gate-P attempt lineage semantic SHA-256")
    unsigned = dict(payload)
    observed = unsigned.pop("lineage_sha256")
    if observed != _semantic_sha256(unsigned):
        raise ValueError("Gate-P attempt lineage semantic SHA-256 is invalid")
    return payload


@dataclass(frozen=True, slots=True)
class GatePAttemptLineage:
    """Non-authorizing predecessor binding for one future Gate-P submission."""

    artifact_path: Path
    file_sha256: str
    size_bytes: int
    lineage_sha256: str
    campaign_identity_sha256: str
    attempt_index: int
    predecessor_file_sha256: str
    predecessor_receipt_sha256: str
    _canonical_bytes: bytes = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.artifact_path.is_absolute():
            raise ValueError("Gate-P attempt lineage path must be absolute")
        for name in (
            "file_sha256",
            "lineage_sha256",
            "campaign_identity_sha256",
            "predecessor_file_sha256",
            "predecessor_receipt_sha256",
        ):
            _digest(getattr(self, name), name=f"Gate-P attempt lineage {name}")
        if type(self.attempt_index) is not int or not 1 <= self.attempt_index <= 999:
            raise ValueError("Gate-P lineaged attempt index is invalid")
        if (
            type(self.size_bytes) is not int
            or self.size_bytes < 1
            or type(self._canonical_bytes) is not bytes
            or len(self._canonical_bytes) != self.size_bytes
            or _sha256(self._canonical_bytes) != self.file_sha256
        ):
            raise ValueError("Gate-P attempt lineage byte binding is invalid")

    @property
    def payload(self) -> dict[str, object]:
        return _validate_lineage_payload(decode_canonical_json_bytes(self._canonical_bytes))

    def validate_integrity(self) -> None:
        self.__post_init__()
        transport = read_canonical_artifact(
            self.artifact_path,
            expected_file_sha256=self.file_sha256,
        )
        if transport.canonical_bytes != self._canonical_bytes:
            raise ValueError("Gate-P attempt lineage bytes changed")
        payload = _validate_lineage_payload(transport.payload)
        exact = {
            "lineage_sha256": payload["lineage_sha256"],
            "campaign_identity_sha256": payload["current_campaign"]["campaign_identity_sha256"],
            "attempt_index": payload["attempt"]["attempt_index"],
            "predecessor_file_sha256": payload["predecessor_failure"]["file_sha256"],
            "predecessor_receipt_sha256": payload["predecessor_failure"]["receipt_sha256"],
        }
        for name, expected in exact.items():
            if getattr(self, name) != expected:
                raise ValueError(f"Gate-P attempt lineage {name} is inconsistent")

    def profile_intent_binding(self) -> dict[str, str]:
        self.validate_integrity()
        return {
            "attempt_lineage_file_sha256": self.file_sha256,
            "attempt_lineage_sha256": self.lineage_sha256,
        }


def _lineage_from_transport(
    artifact_path: Path,
    *,
    file_sha256: str,
    raw: bytes,
) -> GatePAttemptLineage:
    payload = _validate_lineage_payload(decode_canonical_json_bytes(raw))
    result = GatePAttemptLineage(
        artifact_path=artifact_path,
        file_sha256=file_sha256,
        size_bytes=len(raw),
        lineage_sha256=str(payload["lineage_sha256"]),
        campaign_identity_sha256=str(payload["current_campaign"]["campaign_identity_sha256"]),
        attempt_index=int(payload["attempt"]["attempt_index"]),
        predecessor_file_sha256=str(payload["predecessor_failure"]["file_sha256"]),
        predecessor_receipt_sha256=str(payload["predecessor_failure"]["receipt_sha256"]),
        _canonical_bytes=raw,
    )
    result.validate_integrity()
    return result


def publish_gate_p_attempt_lineage(
    *,
    project_root: str | os.PathLike[str],
    attempt_root: str | os.PathLike[str],
    predecessor_receipt: GatePFailureReceipt,
    source_git_commit: str,
    gate0_file_sha256: str,
    gate1_file_sha256: str,
    source_test_receipt_file_sha256: str,
    science_config_file_sha256: str,
    container_file_sha256: str,
) -> GatePAttemptLineage:
    """Publish the predecessor edge before any new profile job is submitted."""

    if type(predecessor_receipt) is not GatePFailureReceipt:
        raise TypeError("predecessor_receipt must be an exact GatePFailureReceipt")
    project = _canonical_directory(Path(project_root), name="project root")
    predecessor_receipt = reopen_gate_p_failure_receipt(
        predecessor_receipt.artifact_path,
        project_root=project,
        expected_file_sha256=predecessor_receipt.file_sha256,
    )
    attempt = _canonical_directory(Path(attempt_root), name="Gate-P attempt root")
    descriptor = _attempt_descriptor(project, attempt)
    current = _source_bindings(
        source_git_commit=source_git_commit,
        gate0_file_sha256=gate0_file_sha256,
        gate1_file_sha256=gate1_file_sha256,
        source_test_receipt_file_sha256=source_test_receipt_file_sha256,
        science_config_file_sha256=science_config_file_sha256,
        container_file_sha256=container_file_sha256,
    )
    predecessor = _predecessor_reference(predecessor_receipt, project_root=project)
    same_campaign = predecessor["campaign_identity_sha256"] == current["campaign_identity_sha256"]
    expected_index = predecessor_receipt.attempt_index + 1 if same_campaign else 1
    if (
        descriptor["identity"] != current["campaign_identity_sha256"]
        or descriptor["attempt_index"] != expected_index
    ):
        raise ValueError("new Gate-P attempt path does not match its campaign/predecessor")
    lineage_path = attempt / ATTEMPT_LINEAGE_FILENAME
    if lineage_path.exists() or lineage_path.is_symlink():
        raise FileExistsError("refusing to overwrite Gate-P attempt lineage")
    unsigned: dict[str, object] = {
        "schema_version": GATE_P_ATTEMPT_LINEAGE_SCHEMA,
        "role": GATE_P_ATTEMPT_LINEAGE_ROLE,
        "current_campaign": current,
        "attempt": {
            **descriptor,
            "predecessor_relation": (
                "same_campaign_next_attempt"
                if same_campaign
                else "new_campaign_attempt_001_with_cross_campaign_predecessor"
            ),
        },
        "predecessor_failure": predecessor,
        "authority": dict(_NO_AUTHORITY),
    }
    payload = {
        **unsigned,
        "lineage_sha256": _semantic_sha256(unsigned),
    }
    _validate_lineage_payload(payload)
    transport = publish_canonical_artifact(lineage_path, payload)
    return reopen_gate_p_attempt_lineage(
        lineage_path,
        project_root=project,
        expected_file_sha256=transport.file_sha256,
    )


def reopen_gate_p_attempt_lineage(
    artifact_path: str | os.PathLike[str],
    *,
    project_root: str | os.PathLike[str],
    expected_file_sha256: str,
) -> GatePAttemptLineage:
    """Reopen a lineaged submission and its immutable predecessor receipt."""

    project = _canonical_directory(Path(project_root), name="project root")
    path = Path(artifact_path).absolute()
    if path.name != ATTEMPT_LINEAGE_FILENAME:
        raise ValueError("Gate-P attempt lineage filename is not fixed")
    attempt = _canonical_directory(path.parent, name="Gate-P attempt root")
    descriptor = _attempt_descriptor(project, attempt)
    transport = read_canonical_artifact(
        path,
        expected_file_sha256=_digest(
            expected_file_sha256,
            name="expected Gate-P attempt lineage file SHA-256",
        ),
    )
    payload = _validate_lineage_payload(transport.payload)
    expected_attempt = dict(payload["attempt"])
    expected_attempt.pop("predecessor_relation")
    if expected_attempt != descriptor:
        raise ValueError("Gate-P attempt lineage path differs from its payload")
    predecessor = payload["predecessor_failure"]
    predecessor_path = project / str(predecessor["project_relative"])
    receipt = reopen_gate_p_failure_receipt(
        predecessor_path,
        project_root=project,
        expected_file_sha256=str(predecessor["file_sha256"]),
    )
    if _predecessor_reference(receipt, project_root=project) != predecessor:
        raise ValueError("Gate-P predecessor receipt reference changed")
    return _lineage_from_transport(
        path,
        file_sha256=transport.file_sha256,
        raw=transport.canonical_bytes,
    )


__all__ = [
    "ATTEMPT_LINEAGE_FILENAME",
    "FAILURE_EVIDENCE_DIRECTORY",
    "FAILURE_RAW_SACCT_FILENAME",
    "FAILURE_RECEIPT_FILENAME",
    "FAILURE_STAGES",
    "GATE_P_ATTEMPT_LINEAGE_ROLE",
    "GATE_P_ATTEMPT_LINEAGE_SCHEMA",
    "GATE_P_CAMPAIGN_IDENTITY_SCHEMA",
    "GATE_P_FAILURE_RECEIPT_ROLE",
    "GATE_P_FAILURE_RECEIPT_SCHEMA",
    "GatePAttemptLineage",
    "GatePFailureReceipt",
    "derive_gate_p_campaign_identity",
    "plan_next_gate_p_attempt",
    "publish_gate_p_attempt_lineage",
    "publish_gate_p_failure_receipt",
    "reopen_gate_p_attempt_lineage",
    "reopen_gate_p_failure_receipt",
]
