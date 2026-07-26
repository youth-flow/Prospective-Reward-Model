"""Fail-closed external Slurm terminal validators for recovery revision 3.

This module deliberately separates three different authority levels:

* :func:`inspect_sacct_terminal_bytes` parses one caller-hashed ``sacct`` row
  and produces claim-free evidence;
* the two ``produce_*`` functions validate that evidence against sealed domain
  evidence and mint an in-process sealed capability; and
* the two ``revalidate_*`` functions can regain such a capability in another
  process only by reading the immutable raw bytes and manifest again.

No JSON ``schema_version`` or ``role`` is trusted as proof of scheduler
success.  A capability is minted only after the original ``sacct -X -n -P``
bytes have been reparsed and checked against the corresponding job identity,
predeclared/derived resources, and domain outcome.  The profile finalizer is a
pure-data path: it never reconstructs profile tensors, oracle state, CUDA
objects, or a live resource-plan object.  This module does not issue Gate-P or
continuation authorization.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import stat
import tempfile
import time
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import Final

from .phase2_r3_artifacts import (
    canonical_json_bytes,
    decode_canonical_json_bytes,
    publish_canonical_artifact,
    read_canonical_artifact,
)
from .phase2_r3_identity import (
    CONTINUABLE_PRIMARY_TERMINAL_ROLE,
    CONTINUABLE_PRIMARY_TERMINAL_SCHEMA,
    R3_PRIMARY_HEADS,
    R3_PRIMARY_SEGMENT_ADMISSION_SCHEMA,
    R3_TASK_SEED_MAP,
    SUCCESSFUL_PROFILE_TERMINAL_ROLE,
    SUCCESSFUL_PROFILE_TERMINAL_SCHEMA,
    VERIFIED_CONTINUATION_CHECKPOINT_ROLE,
    VERIFIED_CONTINUATION_CHECKPOINT_SCHEMA,
    ArtifactRef,
    PrimarySegmentAdmission,
)
from .phase2_r3_orchestrator import (
    SEGMENT_OUTCOME_SCHEMA,
    R3PrimarySegmentOutcome,
    primary_outcome_semantic_sha256,
)
from .phase2_r3_primary import SlurmSegmentRuntime
from .phase2_r3_profile_artifacts import (
    VerifiedGatePOperationalBundle,
    reopen_verified_gate_p_operational_bundle,
)

SACCT_TERMINAL_INSPECTION_SCHEMA: Final = (
    "phase2-recovery-r3-claim-free-sacct-terminal-inspection/v1"
)
SACCT_TERMINAL_PARSED_EVIDENCE_SCHEMA: Final = (
    "phase2-recovery-r3-claim-free-sacct-terminal-parsed-evidence/v1"
)
SACCT_TERMINAL_MANIFEST_SCHEMA: Final = "phase2-recovery-r3-sacct-terminal-evidence-manifest/v1"
PROFILE_SLURM_RUNTIME_RECEIPT_SCHEMA: Final = (
    "phase2-recovery-r3-formal-profile-slurm-runtime-receipt/v1"
)
PROFILE_SLURM_RUNTIME_SCHEMA: Final = PROFILE_SLURM_RUNTIME_RECEIPT_SCHEMA
PROFILE_ALLOCATION_INTENT_SCHEMA: Final = (
    "phase2-recovery-r3-predeclared-profile-allocation-intent/v1"
)
PROFILE_ALLOCATION_INTENT_ROLE: Final = "predeclared_profile_allocation_not_primary_resource_plan"
PRIMARY_SEGMENT_RUNTIME_CLOSURE_SCHEMA: Final = (
    "phase2-recovery-r3-primary-segment-runtime-closure/v1"
)
PRIMARY_SEGMENT_RUNTIME_CLOSURE_PRODUCER_SCHEMA: Final = (
    "phase2-recovery-r3-terminal-primary-closure-producer/v1"
)
COMPLETED_PRIMARY_TERMINAL_SCHEMA: Final = (
    "phase2-recovery-r3-external-primary-segment-completed-terminal/v1"
)
COMPLETED_PRIMARY_TERMINAL_ROLE: Final = (
    "external_scheduler_terminal_completed_zero_exit_compute_complete"
)

_PROFILE_PRODUCER_KIND: Final = "successful_profile_terminal"
_CONTINUABLE_PRIMARY_PRODUCER_KIND: Final = "continuable_primary_terminal"
_COMPLETED_PRIMARY_PRODUCER_KIND: Final = "completed_primary_terminal"
_RAW_FILENAME: Final = "raw-sacct.psv"
_PARSED_FILENAME: Final = "parsed-sacct.json"
_MANIFEST_FILENAME: Final = "terminal-manifest.json"
_MAX_RAW_BYTES: Final = 128 * 1024
_HPC4_CLUSTER: Final = "hpc4"
_HPC4_ACCOUNT: Final = "sigroup"

SACCT_TERMINAL_FIELDS: Final = (
    "JobID",
    "JobIDRaw",
    "State",
    "ExitCode",
    "DerivedExitCode",
    "Cluster",
    "Account",
    "Partition",
    "QOS",
    "NNodes",
    "NCPUS",
    "ReqTRES",
    "AllocTRES",
    "ElapsedRaw",
)
_SACCT_FORMAT_FIELDS: Final = (
    "JobID%64",
    "JobIDRaw%32",
    "State%64",
    "ExitCode%32",
    "DerivedExitCode%32",
    "Cluster%64",
    "Account%64",
    "Partition%64",
    "QOS%64",
    "NNodes%16",
    "NCPUS%16",
    "ReqTRES%512",
    "AllocTRES%512",
    "ElapsedRaw%32",
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_POSITIVE_JOB_ID_RE = re.compile(r"[1-9][0-9]*\Z")
_NONNEGATIVE_DECIMAL_RE = re.compile(r"(?:0|[1-9][0-9]*)\Z")
_MEMORY_RE = re.compile(r"([1-9][0-9]*)([KMGTP]?)\Z", re.IGNORECASE)
_GPU_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_-]*\Z")
_CAPABILITY_SEAL = object()
_PROFILE_RUNTIME_SEAL = object()
_PROFILE_INTENT_SEAL = object()
_PRIMARY_CLOSURE_SEAL = object()

_OUTCOME_FIELDS: Final = frozenset(
    {
        "schema_version",
        "status",
        "design_sha256",
        "admission_sha256",
        "logical_run_id",
        "scheduler_segment_id",
        "runtime_sha256",
        "segment_index",
        "task_id",
        "seed",
        "gate_p_resource_plan_sha256",
        "completed_heads",
        "active_learner",
        "continuation_checkpoint",
        "continuation_reason",
        "all_primary_heads_compute_complete",
        "external_scheduler_terminal_required",
        "external_scheduler_success_claimed",
        "r3_success_authorization_created",
        "information_boundary",
        "outcome_sha256",
    }
)
_PRIMARY_CLOSURE_FIELDS: Final = frozenset(
    {
        "schema_version",
        "producer_schema",
        "operational_bundle",
        "admission",
        "runtime",
        "outcome",
        "status",
        "continuation_checkpoint",
        "completed_head_receipts",
        "external_scheduler_terminal_required",
        "external_scheduler_success_claimed",
        "continuation_authorization_issued",
        "final_three_seed_authorization_issued",
        "information_boundary",
        "closure_sha256",
    }
)
_EMBEDDED_CANONICAL_FIELDS: Final = frozenset({"encoding", "file_sha256", "size_bytes", "payload"})
_PRIMARY_CLOSURE_BUNDLE_FIELDS: Final = frozenset(
    {
        "file_sha256",
        "size_bytes",
        "bundle_semantic_sha256",
        "profile_run_sha256",
        "formal_profile_sha256",
        "resource_plan_sha256",
    }
)


def _canonical_sha256(value: Mapping[str, object]) -> str:
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


def _positive_job_id(value: object, *, name: str) -> str:
    if type(value) is not str or _POSITIVE_JOB_ID_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a positive decimal Slurm job ID")
    return value


def _nonnegative_int_text(value: object, *, name: str) -> int:
    if type(value) is not str or _NONNEGATIVE_DECIMAL_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be canonical non-negative decimal text")
    return int(value)


def _positive_int(value: object, *, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _safe_field(value: object, *, name: str, allow_empty: bool = False) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be text")
    if not allow_empty and not value:
        raise ValueError(f"{name} must be non-empty")
    if any(ord(character) < 0x20 or ord(character) > 0x7E for character in value):
        raise ValueError(f"{name} contains non-printable or non-ASCII characters")
    if "|" in value:
        raise ValueError(f"{name} contains the sacct delimiter")
    return value


def _parse_tres(value: str, *, name: str) -> tuple[tuple[str, str], ...]:
    _safe_field(value, name=name)
    result: list[tuple[str, str]] = []
    seen: set[str] = set()
    for item in value.split(","):
        if item.count("=") != 1:
            raise ValueError(f"{name} is not strict comma-separated key=value data")
        key, scalar = item.split("=", 1)
        if not key or not scalar or key in seen:
            raise ValueError(f"{name} has an empty or duplicate field")
        _safe_field(key, name=f"{name} key")
        _safe_field(scalar, name=f"{name}[{key}]")
        seen.add(key)
        result.append((key, scalar))
    return tuple(result)


def _memory_bytes(value: str, *, name: str) -> int:
    match = _MEMORY_RE.fullmatch(value)
    if match is None:
        raise ValueError(f"{name} must be an integral Slurm K/M/G/T/P memory scalar")
    amount = int(match.group(1))
    unit = match.group(2).upper()
    exponent = {"": 0, "K": 1, "M": 2, "G": 3, "T": 4, "P": 5}[unit]
    return amount * (1024**exponent)


def _canonical_directory(path: Path, *, name: str) -> Path:
    if not path.is_absolute():
        raise ValueError(f"{name} must be absolute")
    try:
        metadata = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{name} is missing or inaccessible") from error
    if not stat.S_ISDIR(metadata.st_mode) or path.is_symlink() or resolved != path:
        raise ValueError(f"{name} must be a canonical real directory")
    return path


def _canonical_file_parent(path: Path, *, name: str) -> Path:
    if not path.is_absolute():
        raise ValueError(f"{name} must be absolute")
    return _canonical_directory(path.parent, name=f"{name} parent")


def _fsync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_stable_bytes(
    path: Path,
    *,
    expected_file_sha256: str,
    require_published_mode: bool,
) -> bytes:
    parent = _canonical_file_parent(path, name="sacct evidence path")
    expected = _digest(expected_file_sha256, name="expected raw sacct SHA-256")
    parent_before = parent.stat()
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or path.is_symlink():
        raise ValueError("raw sacct evidence must be a regular non-symlink file")
    if require_published_mode and os.name == "posix" and stat.S_IMODE(before.st_mode) != 0o440:
        raise ValueError("published raw sacct evidence must retain mode 0440")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags)
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
    after_path = path.lstat()
    parent_after = parent.stat()
    identities = {
        (before.st_dev, before.st_ino),
        (opened.st_dev, opened.st_ino),
        (after_open.st_dev, after_open.st_ino),
        (after_path.st_dev, after_path.st_ino),
    }
    if (
        len(identities) != 1
        or not stat.S_ISREG(after_path.st_mode)
        or path.is_symlink()
        or before.st_size != opened.st_size
        or opened.st_size != after_open.st_size
        or after_open.st_size != after_path.st_size
        or (parent_before.st_dev, parent_before.st_ino)
        != (parent_after.st_dev, parent_after.st_ino)
    ):
        raise ValueError("raw sacct evidence changed while it was being read")
    raw = b"".join(chunks)
    if len(raw) != after_open.st_size or _sha256(raw) != expected:
        raise ValueError("raw sacct evidence does not match its caller-supplied SHA-256")
    return raw


def _publish_immutable_bytes(path: Path, raw: bytes) -> None:
    if type(raw) is not bytes:
        raise TypeError("raw sacct evidence must be exact bytes")
    parent = _canonical_file_parent(path, name="raw sacct evidence path")
    parent_before = parent.stat()
    parent_identity = (parent_before.st_dev, parent_before.st_ino)
    if path.exists() or path.is_symlink():
        raise FileExistsError("refusing to overwrite raw sacct evidence")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.",
        suffix=".tmp",
        dir=parent,
    )
    temporary = Path(temporary_name)
    temporary_identity: tuple[int, int] | None = None
    destination_linked = False
    publication_complete = False
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            if os.name == "posix":
                os.fchmod(stream.fileno(), 0o440)
            os.fsync(stream.fileno())
            metadata = os.fstat(stream.fileno())
            temporary_identity = (metadata.st_dev, metadata.st_ino)
            if metadata.st_size != len(raw):
                raise OSError("temporary raw sacct evidence has the wrong size")
        current_parent = parent.stat()
        if (current_parent.st_dev, current_parent.st_ino) != parent_identity:
            raise ValueError("raw sacct evidence parent changed before publication")
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as error:
            raise FileExistsError("refusing to overwrite raw sacct evidence") from error
        destination_linked = True
        published = path.lstat()
        if (
            temporary_identity is None
            or (published.st_dev, published.st_ino) != temporary_identity
            or not stat.S_ISREG(published.st_mode)
            or path.is_symlink()
        ):
            raise ValueError("published raw evidence is not the verified staged inode")
        _fsync_directory(parent)
        observed = _read_stable_bytes(
            path,
            expected_file_sha256=_sha256(raw),
            require_published_mode=True,
        )
        if observed != raw:
            raise OSError("published raw sacct evidence differs from source bytes")
        publication_complete = True
    finally:
        if destination_linked and not publication_complete:
            try:
                linked = path.lstat()
            except FileNotFoundError:
                linked = None
            if (
                linked is not None
                and temporary_identity is not None
                and (linked.st_dev, linked.st_ino) == temporary_identity
            ):
                path.unlink()
                _fsync_directory(parent)
        with suppress(FileNotFoundError):
            temporary.unlink()


def sacct_terminal_command(job_selector: str) -> tuple[str, ...]:
    """Return the locked allocation-only query for one exact job/task row."""

    selector = _safe_field(job_selector, name="sacct job selector")
    if (
        _POSITIVE_JOB_ID_RE.fullmatch(selector) is None
        and re.fullmatch(r"[1-9][0-9]*_(?:0|[1-9][0-9]*)", selector) is None
    ):
        raise ValueError("sacct job selector must identify one exact job or array task")
    return (
        "sacct",
        "-X",
        "-n",
        "-P",
        "-j",
        selector,
        f"--format={','.join(_SACCT_FORMAT_FIELDS)}",
    )


@dataclass(frozen=True, slots=True)
class SacctAllocationRow:
    """One structurally parsed allocation row, without a success claim."""

    job_id: str
    job_id_raw: str
    state: str
    exit_code: str
    derived_exit_code: str
    cluster: str
    account: str
    partition: str
    qos: str
    n_nodes: int
    n_cpus: int
    req_tres_raw: str
    alloc_tres_raw: str
    req_tres: tuple[tuple[str, str], ...]
    alloc_tres: tuple[tuple[str, str], ...]
    elapsed_seconds: int

    def __post_init__(self) -> None:
        for name in (
            "job_id",
            "job_id_raw",
            "state",
            "exit_code",
            "derived_exit_code",
            "cluster",
            "account",
            "partition",
            "qos",
        ):
            _safe_field(getattr(self, name), name=f"sacct {name}")
        _positive_int(self.n_nodes, name="sacct NNodes")
        _positive_int(self.n_cpus, name="sacct NCPUS")
        _nonnegative_int(self.elapsed_seconds, name="sacct ElapsedRaw")
        if type(self.req_tres) is not tuple or self.req_tres != _parse_tres(
            self.req_tres_raw,
            name="sacct ReqTRES",
        ):
            raise ValueError("parsed ReqTRES differs from its raw sacct field")
        if type(self.alloc_tres) is not tuple or self.alloc_tres != _parse_tres(
            self.alloc_tres_raw,
            name="sacct AllocTRES",
        ):
            raise ValueError("parsed AllocTRES differs from its raw sacct field")

    def to_dict(self) -> dict[str, object]:
        return {
            "job_id": self.job_id,
            "job_id_raw": self.job_id_raw,
            "state": self.state,
            "exit_code": self.exit_code,
            "derived_exit_code": self.derived_exit_code,
            "cluster": self.cluster,
            "account": self.account,
            "partition": self.partition,
            "qos": self.qos,
            "n_nodes": self.n_nodes,
            "n_cpus": self.n_cpus,
            "req_tres_raw": self.req_tres_raw,
            "alloc_tres_raw": self.alloc_tres_raw,
            "req_tres": dict(self.req_tres),
            "alloc_tres": dict(self.alloc_tres),
            "elapsed_seconds": self.elapsed_seconds,
        }


def _parse_single_row(raw: bytes) -> SacctAllocationRow:
    if type(raw) is not bytes:
        raise TypeError("raw sacct evidence must be exact bytes")
    if not raw or len(raw) > _MAX_RAW_BYTES or not raw.endswith(b"\n"):
        raise ValueError("raw sacct evidence must be non-empty, bounded, and newline-terminated")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("raw sacct evidence must be UTF-8") from error
    if "\r" in text or "\x00" in text:
        raise ValueError("raw sacct evidence contains unsafe characters")
    lines = text.splitlines()
    if len(lines) != 1 or not lines[0]:
        raise ValueError("raw sacct evidence must contain exactly one allocation row")
    fields = lines[0].split("|")
    if len(fields) != len(SACCT_TERMINAL_FIELDS):
        raise ValueError(f"raw sacct row must contain exactly {len(SACCT_TERMINAL_FIELDS)} columns")
    (
        job_id,
        job_id_raw,
        state,
        exit_code,
        derived_exit_code,
        cluster,
        account,
        partition,
        qos,
        n_nodes,
        n_cpus,
        req_tres,
        alloc_tres,
        elapsed_raw,
    ) = fields
    return SacctAllocationRow(
        job_id=job_id,
        job_id_raw=job_id_raw,
        state=state,
        exit_code=exit_code,
        derived_exit_code=derived_exit_code,
        cluster=cluster,
        account=account,
        partition=partition,
        qos=qos,
        n_nodes=_nonnegative_int_text(n_nodes, name="sacct NNodes"),
        n_cpus=_nonnegative_int_text(n_cpus, name="sacct NCPUS"),
        req_tres_raw=req_tres,
        alloc_tres_raw=alloc_tres,
        req_tres=_parse_tres(req_tres, name="sacct ReqTRES"),
        alloc_tres=_parse_tres(alloc_tres, name="sacct AllocTRES"),
        elapsed_seconds=_nonnegative_int_text(elapsed_raw, name="sacct ElapsedRaw"),
    )


def _inspection_payload(
    *,
    row: SacctAllocationRow,
    raw_sha256: str,
    raw_size_bytes: int,
) -> dict[str, object]:
    return {
        "schema_version": SACCT_TERMINAL_INSPECTION_SCHEMA,
        "formal_claim_eligible": False,
        "locked_invocation_flags": ["-X", "-n", "-P"],
        "locked_fields": list(SACCT_TERMINAL_FIELDS),
        "raw_sacct": {
            "sha256": raw_sha256,
            "size_bytes": raw_size_bytes,
        },
        "row": row.to_dict(),
    }


@dataclass(frozen=True, slots=True)
class ClaimFreeSacctTerminalInspection:
    """Self-hashed parse result which intentionally carries no authority."""

    row: SacctAllocationRow
    raw_sacct_sha256: str
    raw_size_bytes: int
    inspection_sha256: str
    _raw_bytes: bytes = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if type(self.row) is not SacctAllocationRow:
            raise TypeError("inspection row must be exactly SacctAllocationRow")
        self.row.__post_init__()
        _digest(self.raw_sacct_sha256, name="raw_sacct_sha256")
        _positive_int(self.raw_size_bytes, name="raw_size_bytes")
        _digest(self.inspection_sha256, name="inspection_sha256")
        if type(self._raw_bytes) is not bytes:
            raise TypeError("inspection raw bytes must be exact bytes")
        if (
            len(self._raw_bytes) != self.raw_size_bytes
            or _sha256(self._raw_bytes) != self.raw_sacct_sha256
            or _parse_single_row(self._raw_bytes) != self.row
        ):
            raise ValueError("claim-free inspection differs from its original raw bytes")
        payload = _inspection_payload(
            row=self.row,
            raw_sha256=self.raw_sacct_sha256,
            raw_size_bytes=self.raw_size_bytes,
        )
        if _canonical_sha256(payload) != self.inspection_sha256:
            raise ValueError("claim-free inspection SHA-256 is invalid")

    @property
    def raw_bytes(self) -> bytes:
        return self._raw_bytes

    def validate_integrity(self) -> None:
        self.__post_init__()

    def to_dict(self) -> dict[str, object]:
        self.validate_integrity()
        return {
            **_inspection_payload(
                row=self.row,
                raw_sha256=self.raw_sacct_sha256,
                raw_size_bytes=self.raw_size_bytes,
            ),
            "inspection_sha256": self.inspection_sha256,
        }


SacctTerminalInspection = ClaimFreeSacctTerminalInspection


def inspect_sacct_terminal_bytes(
    raw_sacct_bytes: bytes,
    *,
    expected_raw_sha256: str,
) -> ClaimFreeSacctTerminalInspection:
    """Parse exactly one row under a caller-supplied digest, without promotion."""

    if type(raw_sacct_bytes) is not bytes:
        raise TypeError("raw_sacct_bytes must be exact bytes")
    expected = _digest(expected_raw_sha256, name="expected_raw_sha256")
    observed = _sha256(raw_sacct_bytes)
    if observed != expected:
        raise ValueError("raw sacct bytes do not match expected_raw_sha256")
    row = _parse_single_row(raw_sacct_bytes)
    payload = _inspection_payload(
        row=row,
        raw_sha256=observed,
        raw_size_bytes=len(raw_sacct_bytes),
    )
    result = ClaimFreeSacctTerminalInspection(
        row=row,
        raw_sacct_sha256=observed,
        raw_size_bytes=len(raw_sacct_bytes),
        inspection_sha256=_canonical_sha256(payload),
        _raw_bytes=raw_sacct_bytes,
    )
    result.validate_integrity()
    return result


def inspect_sacct_terminal_file(
    raw_sacct_path: str | os.PathLike[str],
    *,
    expected_raw_sha256: str,
) -> ClaimFreeSacctTerminalInspection:
    """Read and parse a stable regular file under its caller-supplied digest."""

    path = Path(raw_sacct_path)
    raw = _read_stable_bytes(
        path,
        expected_file_sha256=expected_raw_sha256,
        require_published_mode=False,
    )
    return inspect_sacct_terminal_bytes(
        raw,
        expected_raw_sha256=expected_raw_sha256,
    )


inspect_raw_sacct_terminal = inspect_sacct_terminal_bytes


def _validated_operational_bundle(
    value: object,
) -> VerifiedGatePOperationalBundle:
    if type(value) is not VerifiedGatePOperationalBundle:
        raise TypeError("operational_bundle must be exactly VerifiedGatePOperationalBundle")
    value.validate_integrity()
    return value


def _resource_expectation(
    operational_bundle: VerifiedGatePOperationalBundle,
) -> dict[str, object]:
    bundle = _validated_operational_bundle(operational_bundle)
    partition = _safe_field(bundle.partition, name="resource plan partition")
    if not partition.startswith("gpu-"):
        raise ValueError("resource plan partition must be a concrete GPU partition")
    gpu_token = partition.removeprefix("gpu-").lower()
    if _GPU_TOKEN_RE.fullmatch(gpu_token) is None:
        raise ValueError("resource plan partition has no canonical GPU token")
    normalized_gpu_name = re.sub(r"[^a-z0-9]+", "", bundle.gpu_name.lower())
    normalized_gpu_token = re.sub(r"[^a-z0-9]+", "", gpu_token)
    if normalized_gpu_token not in normalized_gpu_name:
        raise ValueError("resource plan GPU name and partition token disagree")
    return {
        "cluster": _HPC4_CLUSTER,
        "account": _HPC4_ACCOUNT,
        "partition": partition,
        "gpu_name": bundle.gpu_name,
        "slurm_gpu_tres": f"gres/gpu:{gpu_token}",
        "gpus_per_task": bundle.gpus_per_task,
        "cpus_per_task": bundle.cpus_per_task,
        "memory_bytes": bundle.memory_bytes,
        "nodes": 1,
        "requested_walltime_seconds": (bundle.requested_walltime_seconds_per_segment),
    }


def _profile_intent_payload(
    *,
    cluster: str,
    account: str,
    partition: str,
    gpu_name: str,
    gpus_per_task: int,
    cpus_per_task: int,
    memory_bytes: int,
    requested_walltime_seconds: int,
) -> dict[str, object]:
    return {
        "schema_version": PROFILE_ALLOCATION_INTENT_SCHEMA,
        "role": PROFILE_ALLOCATION_INTENT_ROLE,
        "declared_before_profile": True,
        "cluster": cluster,
        "account": account,
        "partition": partition,
        "gpu_name": gpu_name,
        "gpus_per_task": gpus_per_task,
        "cpus_per_task": cpus_per_task,
        "memory_bytes": memory_bytes,
        "requested_walltime_seconds": requested_walltime_seconds,
        "information_boundary": "scheduler_allocation_only_no_profile_measurements",
    }


def _validated_profile_intent_payload(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "role",
        "declared_before_profile",
        "cluster",
        "account",
        "partition",
        "gpu_name",
        "gpus_per_task",
        "cpus_per_task",
        "memory_bytes",
        "requested_walltime_seconds",
        "information_boundary",
        "allocation_intent_sha256",
    }:
        raise ValueError("profile allocation intent fields are invalid")
    payload = dict(value)
    unsigned = dict(payload)
    intent_sha = unsigned.pop("allocation_intent_sha256")
    if (
        payload["schema_version"] != PROFILE_ALLOCATION_INTENT_SCHEMA
        or payload["role"] != PROFILE_ALLOCATION_INTENT_ROLE
        or payload["declared_before_profile"] is not True
        or payload["information_boundary"] != "scheduler_allocation_only_no_profile_measurements"
        or intent_sha != _canonical_sha256(unsigned)
    ):
        raise ValueError("profile allocation intent identity/self-hash is invalid")
    if payload["cluster"] != _HPC4_CLUSTER or payload["account"] != _HPC4_ACCOUNT:
        raise ValueError("profile allocation intent must target hpc4/sigroup")
    partition = _safe_field(
        payload["partition"],
        name="profile allocation partition",
    )
    gpu_name = _safe_field(
        payload["gpu_name"],
        name="profile allocation GPU name",
    )
    if not partition.startswith("gpu-"):
        raise ValueError("profile allocation intent requires a GPU partition")
    token = partition.removeprefix("gpu-").lower()
    if _GPU_TOKEN_RE.fullmatch(token) is None or re.sub(r"[^a-z0-9]+", "", token) not in re.sub(
        r"[^a-z0-9]+", "", gpu_name.lower()
    ):
        raise ValueError("profile allocation GPU name/partition disagree")
    for name in (
        "gpus_per_task",
        "cpus_per_task",
        "memory_bytes",
        "requested_walltime_seconds",
    ):
        _positive_int(payload[name], name=f"profile allocation {name}")
    return payload


@dataclass(frozen=True, slots=True)
class ProfileAllocationIntent:
    """Canonical pre-submit resource declaration for the profile job."""

    artifact_path: Path
    file_sha256: str
    size_bytes: int
    allocation_intent_sha256: str
    cluster: str
    account: str
    partition: str
    gpu_name: str
    gpus_per_task: int
    cpus_per_task: int
    memory_bytes: int
    requested_walltime_seconds: int
    _canonical_bytes: bytes = field(repr=False, compare=False)
    _seal: object | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not self.artifact_path.is_absolute():
            raise ValueError("profile allocation intent path must be absolute")
        _digest(self.file_sha256, name="profile allocation intent file SHA-256")
        _digest(
            self.allocation_intent_sha256,
            name="allocation_intent_sha256",
        )
        _positive_int(self.size_bytes, name="profile allocation intent size")
        if type(self._canonical_bytes) is not bytes:
            raise TypeError("profile allocation intent bytes must be exact bytes")

    def validate_integrity(self) -> None:
        self.__post_init__()
        if self._seal is not _PROFILE_INTENT_SEAL:
            raise TypeError("profile allocation intent must be produced by publish/reopen")
        if (
            len(self._canonical_bytes) != self.size_bytes
            or _sha256(self._canonical_bytes) != self.file_sha256
        ):
            raise ValueError("profile allocation intent bytes/file binding is invalid")
        transport = read_canonical_artifact(
            self.artifact_path,
            expected_file_sha256=self.file_sha256,
        )
        if transport.canonical_bytes != self._canonical_bytes:
            raise ValueError("live profile allocation intent bytes changed")
        payload = _validated_profile_intent_payload(transport.payload)
        exact = {
            "allocation_intent_sha256": payload["allocation_intent_sha256"],
            "cluster": payload["cluster"],
            "account": payload["account"],
            "partition": payload["partition"],
            "gpu_name": payload["gpu_name"],
            "gpus_per_task": payload["gpus_per_task"],
            "cpus_per_task": payload["cpus_per_task"],
            "memory_bytes": payload["memory_bytes"],
            "requested_walltime_seconds": payload["requested_walltime_seconds"],
        }
        for name, expected in exact.items():
            if getattr(self, name) != expected:
                raise ValueError(f"profile allocation intent {name} is inconsistent")

    def to_dict(self) -> dict[str, object]:
        self.validate_integrity()
        return read_canonical_artifact(
            self.artifact_path,
            expected_file_sha256=self.file_sha256,
        ).payload

    def expected_slurm_resources(self) -> dict[str, object]:
        self.validate_integrity()
        gpu_token = self.partition.removeprefix("gpu-").lower()
        return {
            "cluster": self.cluster,
            "account": self.account,
            "partition": self.partition,
            "gpu_name": self.gpu_name,
            "slurm_gpu_tres": f"gres/gpu:{gpu_token}",
            "gpus_per_task": self.gpus_per_task,
            "cpus_per_task": self.cpus_per_task,
            "memory_bytes": self.memory_bytes,
            "nodes": 1,
            "requested_walltime_seconds": self.requested_walltime_seconds,
        }


def _profile_intent_from_transport(
    artifact_path: Path,
    *,
    expected_file_sha256: str,
) -> ProfileAllocationIntent:
    transport = read_canonical_artifact(
        artifact_path,
        expected_file_sha256=expected_file_sha256,
    )
    payload = _validated_profile_intent_payload(transport.payload)
    result = ProfileAllocationIntent(
        artifact_path=transport.artifact_path,
        file_sha256=transport.file_sha256,
        size_bytes=transport.size_bytes,
        allocation_intent_sha256=str(payload["allocation_intent_sha256"]),
        cluster=str(payload["cluster"]),
        account=str(payload["account"]),
        partition=str(payload["partition"]),
        gpu_name=str(payload["gpu_name"]),
        gpus_per_task=int(payload["gpus_per_task"]),
        cpus_per_task=int(payload["cpus_per_task"]),
        memory_bytes=int(payload["memory_bytes"]),
        requested_walltime_seconds=int(payload["requested_walltime_seconds"]),
        _canonical_bytes=transport.canonical_bytes,
    )
    object.__setattr__(result, "_seal", _PROFILE_INTENT_SEAL)
    result.validate_integrity()
    return result


def publish_profile_allocation_intent(
    artifact_path: str | os.PathLike[str],
    *,
    cluster: str,
    account: str,
    partition: str,
    gpu_name: str,
    gpus_per_task: int,
    cpus_per_task: int,
    memory_bytes: int,
    requested_walltime_seconds: int,
) -> ProfileAllocationIntent:
    """Before submission, publish the exact profile allocation request."""

    unsigned = _profile_intent_payload(
        cluster=cluster,
        account=account,
        partition=partition,
        gpu_name=gpu_name,
        gpus_per_task=gpus_per_task,
        cpus_per_task=cpus_per_task,
        memory_bytes=memory_bytes,
        requested_walltime_seconds=requested_walltime_seconds,
    )
    payload = {
        **unsigned,
        "allocation_intent_sha256": _canonical_sha256(unsigned),
    }
    _validated_profile_intent_payload(payload)
    transport = publish_canonical_artifact(artifact_path, payload)
    return _profile_intent_from_transport(
        transport.artifact_path,
        expected_file_sha256=transport.file_sha256,
    )


def reopen_profile_allocation_intent(
    artifact_path: str | os.PathLike[str],
    *,
    expected_file_sha256: str,
) -> ProfileAllocationIntent:
    """Reopen the pre-submit allocation intent under a caller file digest."""

    return _profile_intent_from_transport(
        Path(artifact_path),
        expected_file_sha256=expected_file_sha256,
    )


def _profile_runtime_receipt_payload(
    *,
    operational_bundle: VerifiedGatePOperationalBundle,
    allocation_intent: ProfileAllocationIntent,
    cluster: str,
    job_id: str,
    array_job_id: str | None,
    array_task_id: int | None,
    account: str,
    partition: str,
    captured_monotonic_ns: int,
) -> dict[str, object]:
    bundle = _validated_operational_bundle(operational_bundle)
    allocation_intent.validate_integrity()
    return {
        "schema_version": PROFILE_SLURM_RUNTIME_RECEIPT_SCHEMA,
        "operational_bundle": {
            "file_sha256": bundle.file_sha256,
            "bundle_semantic_sha256": bundle.bundle_semantic_sha256,
            "profile_run_sha256": bundle.profile_run_sha256,
            "formal_profile_sha256": bundle.formal_profile_sha256,
            "resource_plan_sha256": bundle.resource_plan_sha256,
        },
        "profile_allocation_intent": {
            "file_sha256": allocation_intent.file_sha256,
            "allocation_intent_sha256": (allocation_intent.allocation_intent_sha256),
            "intent": allocation_intent.to_dict(),
        },
        "expected_slurm_resources": (allocation_intent.expected_slurm_resources()),
        "slurm_runtime": {
            "cluster": cluster,
            "job_id": job_id,
            "array_job_id": array_job_id,
            "array_task_id": array_task_id,
            "account": account,
            "partition": partition,
            "requested_walltime_seconds": (allocation_intent.requested_walltime_seconds),
            "captured_monotonic_ns": captured_monotonic_ns,
        },
        "information_boundary": ("slurm_runtime_identity_only_no_materialization_or_cuda_state"),
    }


def _validated_runtime_receipt_payload(
    value: object,
    *,
    operational_bundle: VerifiedGatePOperationalBundle,
    allocation_intent: ProfileAllocationIntent,
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "operational_bundle",
        "profile_allocation_intent",
        "expected_slurm_resources",
        "slurm_runtime",
        "information_boundary",
        "runtime_receipt_sha256",
    }:
        raise ValueError("profile Slurm runtime receipt fields are invalid")
    payload = dict(value)
    unsigned = dict(payload)
    receipt_sha = unsigned.pop("runtime_receipt_sha256")
    if (
        payload["schema_version"] != PROFILE_SLURM_RUNTIME_RECEIPT_SCHEMA
        or payload["information_boundary"]
        != "slurm_runtime_identity_only_no_materialization_or_cuda_state"
        or receipt_sha != _canonical_sha256(unsigned)
    ):
        raise ValueError("profile Slurm runtime receipt identity/self-hash is invalid")
    bundle = _validated_operational_bundle(operational_bundle)
    expected_bundle = {
        "file_sha256": bundle.file_sha256,
        "bundle_semantic_sha256": bundle.bundle_semantic_sha256,
        "profile_run_sha256": bundle.profile_run_sha256,
        "formal_profile_sha256": bundle.formal_profile_sha256,
        "resource_plan_sha256": bundle.resource_plan_sha256,
    }
    if payload["operational_bundle"] != expected_bundle:
        raise ValueError("profile Slurm runtime receipt binds another bundle")
    if type(allocation_intent) is not ProfileAllocationIntent:
        raise TypeError("allocation_intent must be exactly ProfileAllocationIntent")
    allocation_intent.validate_integrity()
    expected_intent = {
        "file_sha256": allocation_intent.file_sha256,
        "allocation_intent_sha256": (allocation_intent.allocation_intent_sha256),
        "intent": allocation_intent.to_dict(),
    }
    if payload["profile_allocation_intent"] != expected_intent:
        raise ValueError("profile Slurm runtime receipt binds another intent")
    if payload["expected_slurm_resources"] != allocation_intent.expected_slurm_resources():
        raise ValueError("profile Slurm runtime receipt resources differ from intent")
    runtime = _require_binding(
        payload["slurm_runtime"],
        name="profile Slurm runtime",
        keys={
            "cluster",
            "job_id",
            "array_job_id",
            "array_task_id",
            "account",
            "partition",
            "requested_walltime_seconds",
            "captured_monotonic_ns",
        },
    )
    _positive_job_id(runtime["job_id"], name="profile Slurm job_id")
    if (runtime["array_job_id"] is None) is not (runtime["array_task_id"] is None):
        raise ValueError("profile Slurm array job/task must both be set or absent")
    if runtime["array_job_id"] is not None:
        _positive_job_id(
            runtime["array_job_id"],
            name="profile Slurm array_job_id",
        )
        _nonnegative_int(
            runtime["array_task_id"],
            name="profile Slurm array_task_id",
        )
    if (
        runtime["cluster"] != _HPC4_CLUSTER
        or runtime["account"] != _HPC4_ACCOUNT
        or runtime["partition"] != allocation_intent.partition
        or runtime["requested_walltime_seconds"] != allocation_intent.requested_walltime_seconds
    ):
        raise ValueError("profile Slurm runtime differs from intent/HPC4 identity")
    _positive_int(
        runtime["captured_monotonic_ns"],
        name="profile runtime captured_monotonic_ns",
    )
    return payload


@dataclass(frozen=True, slots=True)
class ProfileSlurmRuntimeReceipt:
    """Immutable job-internal receipt reopened without profile state."""

    operational_bundle: VerifiedGatePOperationalBundle = field(
        repr=False,
        compare=False,
    )
    allocation_intent: ProfileAllocationIntent = field(
        repr=False,
        compare=False,
    )
    artifact_path: Path
    file_sha256: str
    size_bytes: int
    runtime_receipt_sha256: str
    cluster: str
    job_id: str
    array_job_id: str | None
    array_task_id: int | None
    account: str
    partition: str
    requested_walltime_seconds: int
    captured_monotonic_ns: int
    _canonical_bytes: bytes = field(repr=False, compare=False)
    _seal: object | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not self.artifact_path.is_absolute():
            raise ValueError("profile runtime receipt path must be absolute")
        _digest(self.file_sha256, name="profile runtime receipt file SHA-256")
        _digest(
            self.runtime_receipt_sha256,
            name="runtime_receipt_sha256",
        )
        _positive_int(self.size_bytes, name="profile runtime receipt size")
        if type(self._canonical_bytes) is not bytes:
            raise TypeError("profile runtime receipt bytes must be exact bytes")

    @property
    def sacct_job_selector(self) -> str:
        if self.array_job_id is None:
            return self.job_id
        return f"{self.array_job_id}_{self.array_task_id}"

    def validate_integrity(self) -> None:
        self.__post_init__()
        if self._seal is not _PROFILE_RUNTIME_SEAL:
            raise TypeError("profile Slurm runtime receipt must be produced by capture/reopen")
        if (
            len(self._canonical_bytes) != self.size_bytes
            or _sha256(self._canonical_bytes) != self.file_sha256
        ):
            raise ValueError("profile runtime receipt bytes/file binding is invalid")
        transport = read_canonical_artifact(
            self.artifact_path,
            expected_file_sha256=self.file_sha256,
        )
        if transport.canonical_bytes != self._canonical_bytes:
            raise ValueError("live profile runtime receipt bytes changed")
        payload = _validated_runtime_receipt_payload(
            transport.payload,
            operational_bundle=self.operational_bundle,
            allocation_intent=self.allocation_intent,
        )
        runtime = payload["slurm_runtime"]
        if not isinstance(runtime, Mapping):
            raise TypeError("profile runtime receipt runtime is invalid")
        exact = {
            "runtime_receipt_sha256": payload["runtime_receipt_sha256"],
            "cluster": runtime["cluster"],
            "job_id": runtime["job_id"],
            "array_job_id": runtime["array_job_id"],
            "array_task_id": runtime["array_task_id"],
            "account": runtime["account"],
            "partition": runtime["partition"],
            "requested_walltime_seconds": runtime["requested_walltime_seconds"],
            "captured_monotonic_ns": runtime["captured_monotonic_ns"],
        }
        for name, expected in exact.items():
            if getattr(self, name) != expected:
                raise ValueError(f"profile runtime receipt {name} is inconsistent")

    def to_dict(self) -> dict[str, object]:
        self.validate_integrity()
        return read_canonical_artifact(
            self.artifact_path,
            expected_file_sha256=self.file_sha256,
        ).payload


ProfileSlurmRuntimeIdentity = ProfileSlurmRuntimeReceipt
FormalProfileJobRuntimeIdentity = ProfileSlurmRuntimeReceipt
ProfileJobRuntimeIdentity = ProfileSlurmRuntimeReceipt


def _runtime_receipt_from_transport(
    operational_bundle: VerifiedGatePOperationalBundle,
    allocation_intent: ProfileAllocationIntent,
    *,
    artifact_path: Path,
    expected_file_sha256: str,
) -> ProfileSlurmRuntimeReceipt:
    bundle = _validated_operational_bundle(operational_bundle)
    transport = read_canonical_artifact(
        artifact_path,
        expected_file_sha256=expected_file_sha256,
    )
    payload = _validated_runtime_receipt_payload(
        transport.payload,
        operational_bundle=bundle,
        allocation_intent=allocation_intent,
    )
    runtime = payload["slurm_runtime"]
    if not isinstance(runtime, Mapping):
        raise TypeError("profile runtime receipt runtime is invalid")
    result = ProfileSlurmRuntimeReceipt(
        operational_bundle=bundle,
        allocation_intent=allocation_intent,
        artifact_path=transport.artifact_path,
        file_sha256=transport.file_sha256,
        size_bytes=transport.size_bytes,
        runtime_receipt_sha256=str(payload["runtime_receipt_sha256"]),
        cluster=str(runtime["cluster"]),
        job_id=str(runtime["job_id"]),
        array_job_id=(None if runtime["array_job_id"] is None else str(runtime["array_job_id"])),
        array_task_id=(None if runtime["array_task_id"] is None else int(runtime["array_task_id"])),
        account=str(runtime["account"]),
        partition=str(runtime["partition"]),
        requested_walltime_seconds=int(runtime["requested_walltime_seconds"]),
        captured_monotonic_ns=int(runtime["captured_monotonic_ns"]),
        _canonical_bytes=transport.canonical_bytes,
    )
    object.__setattr__(result, "_seal", _PROFILE_RUNTIME_SEAL)
    result.validate_integrity()
    return result


def capture_profile_slurm_runtime_receipt(
    operational_bundle: VerifiedGatePOperationalBundle,
    allocation_intent: ProfileAllocationIntent,
    artifact_path: str | os.PathLike[str],
) -> ProfileSlurmRuntimeReceipt:
    """Inside the profile job, publish its pure-data Slurm runtime receipt."""

    bundle = _validated_operational_bundle(operational_bundle)
    if type(allocation_intent) is not ProfileAllocationIntent:
        raise TypeError("allocation_intent must be exactly ProfileAllocationIntent")
    allocation_intent.validate_integrity()
    required = {
        "cluster": "SLURM_CLUSTER_NAME",
        "job_id": "SLURM_JOB_ID",
        "account": "SLURM_JOB_ACCOUNT",
        "partition": "SLURM_JOB_PARTITION",
    }
    observed: dict[str, str] = {}
    for name, variable in required.items():
        value = os.environ.get(variable)
        if type(value) is not str or not value:
            raise RuntimeError(f"formal Gate-P runtime receipt requires {variable}")
        observed[name] = value
    array_job_id = os.environ.get("SLURM_ARRAY_JOB_ID")
    array_task_text = os.environ.get("SLURM_ARRAY_TASK_ID")
    if (array_job_id is None) is not (array_task_text is None):
        raise RuntimeError("SLURM_ARRAY_JOB_ID and SLURM_ARRAY_TASK_ID must both be set or absent")
    array_task_id: int | None = None
    if array_job_id is not None:
        _positive_job_id(array_job_id, name="SLURM_ARRAY_JOB_ID")
        if (
            type(array_task_text) is not str
            or _NONNEGATIVE_DECIMAL_RE.fullmatch(array_task_text) is None
        ):
            raise ValueError("SLURM_ARRAY_TASK_ID must be canonical non-negative decimal")
        array_task_id = int(array_task_text)
    _positive_job_id(observed["job_id"], name="SLURM_JOB_ID")
    if (
        observed["cluster"] != _HPC4_CLUSTER
        or observed["account"] != _HPC4_ACCOUNT
        or observed["partition"] != allocation_intent.partition
    ):
        raise ValueError("live profile Slurm environment differs from intent")
    optional_integer_resources = {
        "SLURM_CPUS_PER_TASK": allocation_intent.cpus_per_task,
        "SLURM_GPUS_PER_TASK": allocation_intent.gpus_per_task,
    }
    for variable, expected in optional_integer_resources.items():
        value = os.environ.get(variable)
        if value is not None and (
            _NONNEGATIVE_DECIMAL_RE.fullmatch(value) is None or int(value) != expected
        ):
            raise ValueError(f"live {variable} differs from profile intent")
    memory_mib = os.environ.get("SLURM_MEM_PER_NODE")
    if memory_mib is not None and (
        _NONNEGATIVE_DECIMAL_RE.fullmatch(memory_mib) is None
        or int(memory_mib) * 1024**2 != allocation_intent.memory_bytes
    ):
        raise ValueError("live SLURM_MEM_PER_NODE differs from profile intent")
    unsigned = _profile_runtime_receipt_payload(
        operational_bundle=bundle,
        allocation_intent=allocation_intent,
        cluster=observed["cluster"],
        job_id=observed["job_id"],
        array_job_id=array_job_id,
        array_task_id=array_task_id,
        account=observed["account"],
        partition=observed["partition"],
        captured_monotonic_ns=time.monotonic_ns(),
    )
    payload = {
        **unsigned,
        "runtime_receipt_sha256": _canonical_sha256(unsigned),
    }
    transport = publish_canonical_artifact(artifact_path, payload)
    return _runtime_receipt_from_transport(
        bundle,
        allocation_intent,
        artifact_path=transport.artifact_path,
        expected_file_sha256=transport.file_sha256,
    )


def reopen_profile_slurm_runtime_receipt(
    artifact_path: str | os.PathLike[str],
    *,
    expected_file_sha256: str,
    operational_bundle: VerifiedGatePOperationalBundle,
    allocation_intent: ProfileAllocationIntent,
) -> ProfileSlurmRuntimeReceipt:
    """Reopen a job receipt using only canonical bytes and a sealed bundle."""

    return _runtime_receipt_from_transport(
        operational_bundle,
        allocation_intent,
        artifact_path=Path(artifact_path),
        expected_file_sha256=expected_file_sha256,
    )


capture_profile_slurm_runtime_identity = capture_profile_slurm_runtime_receipt
capture_formal_profile_job_runtime = capture_profile_slurm_runtime_receipt


def _validate_terminal_row(
    inspection: ClaimFreeSacctTerminalInspection,
    *,
    expected_job_id: str,
    expected_job_id_raw: str,
    expected_resources: Mapping[str, object],
    requested_walltime_seconds: int,
) -> SacctAllocationRow:
    if type(inspection) is not ClaimFreeSacctTerminalInspection:
        raise TypeError("inspection must be exactly ClaimFreeSacctTerminalInspection")
    inspection.validate_integrity()
    row = inspection.row
    _safe_field(expected_job_id, name="expected sacct JobID")
    _positive_job_id(expected_job_id_raw, name="expected sacct JobIDRaw")
    if set(expected_resources) != {
        "cluster",
        "account",
        "partition",
        "gpu_name",
        "slurm_gpu_tres",
        "gpus_per_task",
        "cpus_per_task",
        "memory_bytes",
        "nodes",
        "requested_walltime_seconds",
    }:
        raise ValueError("expected Slurm resources have an invalid closed field set")
    resources = dict(expected_resources)
    if (
        type(resources["requested_walltime_seconds"]) is not int
        or resources["requested_walltime_seconds"] != requested_walltime_seconds
    ):
        raise ValueError("expected Slurm walltime differs from runtime identity")
    if (
        row.job_id != expected_job_id
        or row.job_id_raw != expected_job_id_raw
        or _POSITIVE_JOB_ID_RE.fullmatch(row.job_id_raw) is None
    ):
        raise ValueError("sacct row is not the exact corresponding job/task allocation row")
    if (
        "." in row.job_id
        or "[" in row.job_id
        or "]" in row.job_id
        or "%" in row.job_id
        or "," in row.job_id
        or "+" in row.job_id
    ):
        raise ValueError("parent, step, range, and heterogeneous sacct rows are forbidden")
    if row.state != "COMPLETED" or row.exit_code != "0:0" or row.derived_exit_code != "0:0":
        raise ValueError("scheduler terminal must be exactly COMPLETED with both exits 0:0")
    if (
        row.cluster != resources["cluster"]
        or row.account != resources["account"]
        or row.partition != resources["partition"]
        or row.n_nodes != resources["nodes"]
        or row.n_cpus != resources["cpus_per_task"]
    ):
        raise ValueError("sacct row differs from the exact HPC4 allocation identity")
    if row.elapsed_seconds < 1 or row.elapsed_seconds > requested_walltime_seconds:
        raise ValueError("sacct elapsed time is outside the admitted walltime")

    req = dict(row.req_tres)
    alloc = dict(row.alloc_tres)
    expected_req_keys = {"billing", "cpu", "gres/gpu", "mem", "node"}
    expected_alloc_keys = {
        *expected_req_keys,
        str(resources["slurm_gpu_tres"]),
    }
    expected_cpus = str(resources["cpus_per_task"])
    expected_gpus = str(resources["gpus_per_task"])
    if (
        set(req) != expected_req_keys
        or set(alloc) != expected_alloc_keys
        or req["billing"] != expected_cpus
        or req["cpu"] != expected_cpus
        or req["gres/gpu"] != expected_gpus
        or req["node"] != "1"
        or alloc["billing"] != expected_cpus
        or alloc["cpu"] != expected_cpus
        or alloc["gres/gpu"] != expected_gpus
        or alloc[str(resources["slurm_gpu_tres"])] != expected_gpus
        or alloc["node"] != "1"
        or _memory_bytes(req["mem"], name="ReqTRES mem") != resources["memory_bytes"]
        or _memory_bytes(alloc["mem"], name="AllocTRES mem") != resources["memory_bytes"]
    ):
        raise ValueError("sacct TRES differ from the exact expected Slurm resources")
    return row


def _validate_profile_dependencies(
    operational_bundle: VerifiedGatePOperationalBundle,
    *,
    runtime_receipt: ProfileSlurmRuntimeReceipt,
) -> tuple[
    VerifiedGatePOperationalBundle,
    ProfileSlurmRuntimeReceipt,
]:
    bundle = _validated_operational_bundle(operational_bundle)
    if type(runtime_receipt) is not ProfileSlurmRuntimeReceipt:
        raise TypeError("runtime_receipt must be exactly ProfileSlurmRuntimeReceipt")
    runtime_receipt.validate_integrity()
    if (
        runtime_receipt.operational_bundle.file_sha256 != bundle.file_sha256
        or runtime_receipt.operational_bundle.bundle_semantic_sha256
        != bundle.bundle_semantic_sha256
        or runtime_receipt.operational_bundle.profile_run_sha256 != bundle.profile_run_sha256
        or runtime_receipt.operational_bundle.formal_profile_sha256 != bundle.formal_profile_sha256
        or runtime_receipt.operational_bundle.resource_plan_sha256 != bundle.resource_plan_sha256
    ):
        raise ValueError("profile runtime receipt belongs to another bundle")
    return bundle, runtime_receipt


def _read_primary_outcome(
    admission: PrimarySegmentAdmission,
    *,
    runtime: SlurmSegmentRuntime,
    outcome: R3PrimarySegmentOutcome,
    operational_bundle: VerifiedGatePOperationalBundle,
) -> dict[str, object]:
    if type(outcome) is not R3PrimarySegmentOutcome:
        raise TypeError("outcome must be exactly R3PrimarySegmentOutcome")
    outcome.validate_integrity()
    if outcome.status not in {
        "continuation_required_after_safe_checkpoint",
        "compute_complete_pending_external_scheduler_terminal",
    }:
        raise ValueError("primary segment outcome status is invalid")
    transport = read_canonical_artifact(
        outcome.artifact_path,
        expected_file_sha256=outcome.file_sha256,
    )
    value = transport.payload
    if set(value) != _OUTCOME_FIELDS:
        raise ValueError("primary segment outcome has an invalid closed field set")
    unsigned = dict(value)
    outcome_sha = unsigned.pop("outcome_sha256")
    if outcome_sha != outcome.outcome_sha256 or outcome_sha != primary_outcome_semantic_sha256(
        unsigned
    ):
        raise ValueError("primary segment outcome self-hash is invalid")
    expected_bindings = {
        "schema_version": SEGMENT_OUTCOME_SCHEMA,
        "status": outcome.status,
        "design_sha256": admission.design.design_sha256,
        "admission_sha256": admission.admission_sha256,
        "logical_run_id": admission.logical_run_id,
        "scheduler_segment_id": admission.scheduler_segment_id,
        "runtime_sha256": runtime.runtime_sha256,
        "segment_index": admission.segment_index,
        "task_id": admission.task_id,
        "seed": admission.seed,
        "gate_p_resource_plan_sha256": operational_bundle.resource_plan_sha256,
        "external_scheduler_terminal_required": True,
        "external_scheduler_success_claimed": False,
        "r3_success_authorization_created": False,
        "information_boundary": "train_only_head_free_segment_outcome",
    }
    for key, expected in expected_bindings.items():
        if value.get(key) != expected:
            raise ValueError(f"primary segment outcome {key} binding is invalid")

    completed = value["completed_heads"]
    if type(completed) is not list:
        raise TypeError("primary segment completed_heads must be an exact list")
    normalized_completed: list[dict[str, object]] = []
    for index, item in enumerate(completed):
        if not isinstance(item, Mapping) or set(item) != {
            "learner",
            "head_run_id",
            "completion_receipt_sha256",
        }:
            raise ValueError("primary segment completion receipt has invalid fields")
        copied = dict(item)
        if index >= len(R3_PRIMARY_HEADS):
            raise ValueError("primary segment has too many completion receipts")
        if (
            copied["learner"] != R3_PRIMARY_HEADS[index]
            or copied["head_run_id"] != admission.head_run_ids[index]
        ):
            raise ValueError("primary segment completion receipt ordering/binding is invalid")
        _digest(
            copied["completion_receipt_sha256"],
            name="completion_receipt_sha256",
        )
        normalized_completed.append(copied)

    if outcome.status == "continuation_required_after_safe_checkpoint":
        checkpoint = outcome.continuation_checkpoint
        if checkpoint is None:
            raise ValueError("continuation outcome lacks its checkpoint ref")
        checkpoint.validate_integrity()
        checkpoint_mapping = checkpoint.to_dict()
        if (
            checkpoint.schema_version != VERIFIED_CONTINUATION_CHECKPOINT_SCHEMA
            or checkpoint.role != VERIFIED_CONTINUATION_CHECKPOINT_ROLE
            or value["continuation_checkpoint"] != checkpoint_mapping
            or value["all_primary_heads_compute_complete"] is not False
            or type(value["continuation_reason"]) is not str
            or not value["continuation_reason"]
            or value["active_learner"] not in R3_PRIMARY_HEADS
        ):
            raise ValueError("primary continuation outcome is structurally incomplete")
        active_index = R3_PRIMARY_HEADS.index(value["active_learner"])
        if len(normalized_completed) != active_index:
            raise ValueError("continuation completion receipts are not the active-head prefix")
    else:
        if (
            outcome.continuation_checkpoint is not None
            or value["continuation_checkpoint"] is not None
            or value["continuation_reason"] is not None
            or value["active_learner"] is not None
            or value["all_primary_heads_compute_complete"] is not True
            or len(normalized_completed) != len(R3_PRIMARY_HEADS)
        ):
            raise ValueError("completed primary outcome is structurally incomplete")
    return value


def _validate_primary_dependencies(
    admission: PrimarySegmentAdmission,
    *,
    runtime: SlurmSegmentRuntime,
    outcome: R3PrimarySegmentOutcome,
    operational_bundle: VerifiedGatePOperationalBundle,
) -> tuple[
    PrimarySegmentAdmission,
    SlurmSegmentRuntime,
    R3PrimarySegmentOutcome,
    VerifiedGatePOperationalBundle,
    dict[str, object],
]:
    if type(admission) is not PrimarySegmentAdmission:
        raise TypeError("admission must be exactly PrimarySegmentAdmission")
    admission.validate_integrity()
    if type(runtime) is not SlurmSegmentRuntime:
        raise TypeError("runtime must be exactly SlurmSegmentRuntime")
    runtime.validate_integrity()
    bundle = _validated_operational_bundle(operational_bundle)
    design = admission.design
    authorization = design.profile_authorization
    if (
        design.resource_policy_sha256 != bundle.resource_plan_sha256
        or authorization.resource_plan.artifact_sha256 != bundle.resource_plan_sha256
        or authorization.formal_cuda_profile_result.artifact_sha256 != bundle.formal_profile_sha256
        or authorization.profile_run_sha256 != bundle.profile_run_sha256
        or design.max_scheduler_segments != bundle.max_scheduler_segments
    ):
        raise ValueError("primary admission is bound to another operational bundle")
    runtime_bindings = {
        "design_sha256": design.design_sha256,
        "admission_sha256": admission.admission_sha256,
        "scheduler_segment_id": admission.scheduler_segment_id,
        "segment_index": admission.segment_index,
        "task_id": admission.task_id,
        "seed": admission.seed,
        "cluster": _HPC4_CLUSTER,
        "account": _HPC4_ACCOUNT,
        "partition": bundle.partition,
        "requested_walltime_seconds": (bundle.requested_walltime_seconds_per_segment),
    }
    for name, expected in runtime_bindings.items():
        if getattr(runtime, name) != expected:
            raise ValueError(f"primary runtime {name} differs from admission/resource plan")
    outcome_payload = _read_primary_outcome(
        admission,
        runtime=runtime,
        outcome=outcome,
        operational_bundle=bundle,
    )
    return admission, runtime, outcome, bundle, outcome_payload


def _embedded_canonical(payload: Mapping[str, object]) -> dict[str, object]:
    raw = canonical_json_bytes(payload)
    return {
        "encoding": "canonical-json-utf8-newline",
        "file_sha256": _sha256(raw),
        "size_bytes": len(raw),
        "payload": dict(payload),
    }


def _validate_embedded_canonical(
    value: object,
    *,
    name: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _EMBEDDED_CANONICAL_FIELDS:
        raise ValueError(f"{name} embedded canonical binding has invalid fields")
    embedded = dict(value)
    payload = embedded["payload"]
    if not isinstance(payload, Mapping):
        raise TypeError(f"{name} embedded payload must be a mapping")
    copied = dict(payload)
    raw = canonical_json_bytes(copied)
    if (
        embedded["encoding"] != "canonical-json-utf8-newline"
        or embedded["file_sha256"] != _sha256(raw)
        or embedded["size_bytes"] != len(raw)
    ):
        raise ValueError(f"{name} embedded canonical byte binding is invalid")
    return copied


def _validate_primary_closure_admission(
    value: dict[str, object],
) -> tuple[list[dict[str, object]], int, int]:
    fields = {
        "schema_version",
        "design_sha256",
        "materialization_attestation_sha256",
        "task_id",
        "seed",
        "segment_index",
        "logical_run_id",
        "head_runs",
        "scheduler_segment_id",
        "start_mode",
        "continuation_evidence_sha256",
        "admission_sha256",
    }
    if set(value) != fields:
        raise ValueError("closure admission has an invalid closed field set")
    if value["schema_version"] != R3_PRIMARY_SEGMENT_ADMISSION_SCHEMA:
        raise ValueError("closure admission schema is invalid")
    for name in (
        "design_sha256",
        "materialization_attestation_sha256",
        "logical_run_id",
        "scheduler_segment_id",
        "admission_sha256",
    ):
        _digest(value[name], name=f"closure admission {name}")
    task_id = value["task_id"]
    seed = value["seed"]
    segment_index = value["segment_index"]
    if (
        type(task_id) is not int
        or type(seed) is not int
        or type(segment_index) is not int
        or segment_index < 1
        or dict(R3_TASK_SEED_MAP).get(task_id) != seed
    ):
        raise ValueError("closure admission task/seed/segment is invalid")
    heads = value["head_runs"]
    if type(heads) is not list or len(heads) != len(R3_PRIMARY_HEADS):
        raise ValueError("closure admission head runs are invalid")
    copied_heads: list[dict[str, object]] = []
    for index, item in enumerate(heads):
        if not isinstance(item, Mapping) or set(item) != {"head", "head_run_id"}:
            raise ValueError("closure admission head run has invalid fields")
        copied = dict(item)
        if copied["head"] != R3_PRIMARY_HEADS[index]:
            raise ValueError("closure admission head order is invalid")
        _digest(copied["head_run_id"], name="closure admission head_run_id")
        copied_heads.append(copied)
    unsigned = dict(value)
    admission_sha = unsigned.pop("admission_sha256")
    if admission_sha != _canonical_sha256(unsigned):
        raise ValueError("closure admission self-hash is invalid")
    if segment_index == 1:
        if (
            value["start_mode"] != "fresh_zero_head_fresh_adamw"
            or value["continuation_evidence_sha256"] is not None
        ):
            raise ValueError("closure admission first-segment mode is invalid")
    else:
        if value["start_mode"] != "verified_state_complete_continuation":
            raise ValueError("closure admission continuation mode is invalid")
        _digest(
            value["continuation_evidence_sha256"],
            name="closure continuation_evidence_sha256",
        )
    return copied_heads, task_id, seed


def _validate_primary_closure_runtime(
    value: dict[str, object],
    *,
    admission: Mapping[str, object],
    operational_bundle: VerifiedGatePOperationalBundle,
) -> None:
    fields = {
        "schema_version",
        "design_sha256",
        "admission_sha256",
        "scheduler_segment_id",
        "segment_index",
        "task_id",
        "seed",
        "cluster",
        "job_id",
        "array_job_id",
        "array_task_id",
        "account",
        "partition",
        "requested_walltime_seconds",
        "captured_monotonic_ns",
        "runtime_sha256",
    }
    if set(value) != fields:
        raise ValueError("closure runtime has an invalid closed field set")
    if value["schema_version"] != "phase2-recovery-r3-slurm-segment-runtime/v1":
        raise ValueError("closure runtime schema is invalid")
    expected = {
        "design_sha256": admission["design_sha256"],
        "admission_sha256": admission["admission_sha256"],
        "scheduler_segment_id": admission["scheduler_segment_id"],
        "segment_index": admission["segment_index"],
        "task_id": admission["task_id"],
        "seed": admission["seed"],
        "cluster": _HPC4_CLUSTER,
        "account": _HPC4_ACCOUNT,
        "partition": operational_bundle.partition,
        "requested_walltime_seconds": (operational_bundle.requested_walltime_seconds_per_segment),
    }
    for name, expected_value in expected.items():
        if value[name] != expected_value:
            raise ValueError(f"closure runtime {name} binding is invalid")
    if (
        type(value["array_task_id"]) is not int
        or value["array_task_id"] != admission["task_id"]
        or type(value["captured_monotonic_ns"]) is not int
        or value["captured_monotonic_ns"] < 1
    ):
        raise ValueError("closure runtime array task or capture time is invalid")
    _positive_job_id(value["job_id"], name="closure runtime job_id")
    _positive_job_id(value["array_job_id"], name="closure runtime array_job_id")
    unsigned = dict(value)
    runtime_sha = unsigned.pop("runtime_sha256")
    _digest(runtime_sha, name="closure runtime_sha256")
    if runtime_sha != _canonical_sha256(unsigned):
        raise ValueError("closure runtime self-hash is invalid")


def _validate_primary_closure_outcome(
    value: dict[str, object],
    *,
    admission: Mapping[str, object],
    runtime: Mapping[str, object],
    heads: list[dict[str, object]],
    operational_bundle: VerifiedGatePOperationalBundle,
) -> None:
    if set(value) != _OUTCOME_FIELDS:
        raise ValueError("closure outcome has an invalid closed field set")
    unsigned = dict(value)
    outcome_sha = unsigned.pop("outcome_sha256")
    _digest(outcome_sha, name="closure outcome_sha256")
    if outcome_sha != primary_outcome_semantic_sha256(unsigned):
        raise ValueError("closure outcome self-hash is invalid")
    status = value["status"]
    if status not in {
        "continuation_required_after_safe_checkpoint",
        "compute_complete_pending_external_scheduler_terminal",
    }:
        raise ValueError("closure outcome status is invalid")
    expected = {
        "schema_version": SEGMENT_OUTCOME_SCHEMA,
        "design_sha256": admission["design_sha256"],
        "admission_sha256": admission["admission_sha256"],
        "logical_run_id": admission["logical_run_id"],
        "scheduler_segment_id": admission["scheduler_segment_id"],
        "runtime_sha256": runtime["runtime_sha256"],
        "segment_index": admission["segment_index"],
        "task_id": admission["task_id"],
        "seed": admission["seed"],
        "gate_p_resource_plan_sha256": operational_bundle.resource_plan_sha256,
        "external_scheduler_terminal_required": True,
        "external_scheduler_success_claimed": False,
        "r3_success_authorization_created": False,
        "information_boundary": "train_only_head_free_segment_outcome",
    }
    for name, expected_value in expected.items():
        if value[name] != expected_value:
            raise ValueError(f"closure outcome {name} binding is invalid")
    completed = value["completed_heads"]
    if type(completed) is not list or len(completed) > len(R3_PRIMARY_HEADS):
        raise ValueError("closure outcome completed heads are invalid")
    for index, item in enumerate(completed):
        if not isinstance(item, Mapping) or set(item) != {
            "learner",
            "head_run_id",
            "completion_receipt_sha256",
        }:
            raise ValueError("closure outcome completion receipt has invalid fields")
        if (
            item["learner"] != R3_PRIMARY_HEADS[index]
            or item["head_run_id"] != heads[index]["head_run_id"]
        ):
            raise ValueError("closure outcome completion receipt binding is invalid")
        _digest(
            item["completion_receipt_sha256"],
            name="closure completion_receipt_sha256",
        )
    if status == "continuation_required_after_safe_checkpoint":
        checkpoint = value["continuation_checkpoint"]
        if not isinstance(checkpoint, Mapping) or set(checkpoint) != {
            "schema_version",
            "artifact_sha256",
            "role",
        }:
            raise ValueError("closure continuation checkpoint ref is invalid")
        if (
            checkpoint["schema_version"] != VERIFIED_CONTINUATION_CHECKPOINT_SCHEMA
            or checkpoint["role"] != VERIFIED_CONTINUATION_CHECKPOINT_ROLE
        ):
            raise ValueError("closure continuation checkpoint role/schema is invalid")
        _digest(
            checkpoint["artifact_sha256"],
            name="closure continuation checkpoint SHA256",
        )
        if (
            value["active_learner"] not in R3_PRIMARY_HEADS
            or len(completed) != R3_PRIMARY_HEADS.index(value["active_learner"])
            or type(value["continuation_reason"]) is not str
            or not value["continuation_reason"]
            or value["all_primary_heads_compute_complete"] is not False
        ):
            raise ValueError("closure continuation outcome is structurally invalid")
    elif (
        value["continuation_checkpoint"] is not None
        or value["continuation_reason"] is not None
        or value["active_learner"] is not None
        or value["all_primary_heads_compute_complete"] is not True
        or len(completed) != len(R3_PRIMARY_HEADS)
    ):
        raise ValueError("closure completed outcome is structurally invalid")


def _primary_closure_payload(
    *,
    admission: PrimarySegmentAdmission,
    runtime: SlurmSegmentRuntime,
    outcome_payload: Mapping[str, object],
    operational_bundle: VerifiedGatePOperationalBundle,
) -> dict[str, object]:
    status = str(outcome_payload["status"])
    continuation = outcome_payload["continuation_checkpoint"]
    completed = outcome_payload["completed_heads"]
    body: dict[str, object] = {
        "schema_version": PRIMARY_SEGMENT_RUNTIME_CLOSURE_SCHEMA,
        "producer_schema": PRIMARY_SEGMENT_RUNTIME_CLOSURE_PRODUCER_SCHEMA,
        "operational_bundle": {
            "file_sha256": operational_bundle.file_sha256,
            "size_bytes": operational_bundle.size_bytes,
            "bundle_semantic_sha256": operational_bundle.bundle_semantic_sha256,
            "profile_run_sha256": operational_bundle.profile_run_sha256,
            "formal_profile_sha256": operational_bundle.formal_profile_sha256,
            "resource_plan_sha256": operational_bundle.resource_plan_sha256,
        },
        "admission": _embedded_canonical(admission.to_dict()),
        "runtime": _embedded_canonical(runtime.to_dict()),
        "outcome": _embedded_canonical(outcome_payload),
        "status": status,
        "continuation_checkpoint": (
            continuation if status == "continuation_required_after_safe_checkpoint" else None
        ),
        "completed_head_receipts": (
            completed if status == "compute_complete_pending_external_scheduler_terminal" else None
        ),
        "external_scheduler_terminal_required": True,
        "external_scheduler_success_claimed": False,
        "continuation_authorization_issued": False,
        "final_three_seed_authorization_issued": False,
        "information_boundary": ("pure_data_scheduler_terminal_evidence_only_no_authorization"),
    }
    return {**body, "closure_sha256": _canonical_sha256(body)}


def _validated_primary_closure_payload(
    value: object,
    *,
    operational_bundle: VerifiedGatePOperationalBundle,
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _PRIMARY_CLOSURE_FIELDS:
        raise ValueError("primary runtime closure has an invalid closed field set")
    payload = dict(value)
    unsigned = dict(payload)
    closure_sha = unsigned.pop("closure_sha256")
    _digest(closure_sha, name="closure_sha256")
    if closure_sha != _canonical_sha256(unsigned):
        raise ValueError("primary runtime closure self-hash is invalid")
    if (
        payload["schema_version"] != PRIMARY_SEGMENT_RUNTIME_CLOSURE_SCHEMA
        or payload["producer_schema"] != PRIMARY_SEGMENT_RUNTIME_CLOSURE_PRODUCER_SCHEMA
        or payload["external_scheduler_terminal_required"] is not True
        or payload["external_scheduler_success_claimed"] is not False
        or payload["continuation_authorization_issued"] is not False
        or payload["final_three_seed_authorization_issued"] is not False
        or payload["information_boundary"]
        != "pure_data_scheduler_terminal_evidence_only_no_authorization"
    ):
        raise ValueError("primary runtime closure authority boundary is invalid")
    bundle_binding = payload["operational_bundle"]
    if (
        not isinstance(bundle_binding, Mapping)
        or set(bundle_binding) != _PRIMARY_CLOSURE_BUNDLE_FIELDS
    ):
        raise ValueError("primary runtime closure bundle binding has invalid fields")
    expected_bundle = {
        "file_sha256": operational_bundle.file_sha256,
        "size_bytes": operational_bundle.size_bytes,
        "bundle_semantic_sha256": operational_bundle.bundle_semantic_sha256,
        "profile_run_sha256": operational_bundle.profile_run_sha256,
        "formal_profile_sha256": operational_bundle.formal_profile_sha256,
        "resource_plan_sha256": operational_bundle.resource_plan_sha256,
    }
    if dict(bundle_binding) != expected_bundle:
        raise ValueError("primary runtime closure belongs to another operational bundle")
    admission = _validate_embedded_canonical(payload["admission"], name="admission")
    heads, _, _ = _validate_primary_closure_admission(admission)
    runtime = _validate_embedded_canonical(payload["runtime"], name="runtime")
    _validate_primary_closure_runtime(
        runtime,
        admission=admission,
        operational_bundle=operational_bundle,
    )
    outcome = _validate_embedded_canonical(payload["outcome"], name="outcome")
    _validate_primary_closure_outcome(
        outcome,
        admission=admission,
        runtime=runtime,
        heads=heads,
        operational_bundle=operational_bundle,
    )
    status = outcome["status"]
    if payload["status"] != status:
        raise ValueError("primary runtime closure status differs from its outcome")
    if status == "continuation_required_after_safe_checkpoint":
        if (
            payload["continuation_checkpoint"] != outcome["continuation_checkpoint"]
            or payload["completed_head_receipts"] is not None
        ):
            raise ValueError("continuation closure state fields are invalid")
    elif (
        payload["continuation_checkpoint"] is not None
        or payload["completed_head_receipts"] != outcome["completed_heads"]
    ):
        raise ValueError("completed closure state fields are invalid")
    return payload


@dataclass(frozen=True, slots=True)
class PrimarySegmentRuntimeClosure:
    """Canonical pure-data closure published before the primary job exits."""

    operational_bundle: VerifiedGatePOperationalBundle = field(
        repr=False,
        compare=False,
    )
    artifact_path: Path
    file_sha256: str
    size_bytes: int
    closure_sha256: str
    status: str
    _canonical_bytes: bytes = field(repr=False, compare=False)
    _factory_token: InitVar[object] = None
    _seal: object | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _PRIMARY_CLOSURE_SEAL:
            raise TypeError("primary runtime closure must be produced by publish/reopen")
        object.__setattr__(self, "_seal", _PRIMARY_CLOSURE_SEAL)
        self._validate_sealed_payload()

    def _decoded(self) -> dict[str, object]:
        return decode_canonical_json_bytes(self._canonical_bytes)

    def _validate_sealed_payload(self) -> None:
        if self._seal is not _PRIMARY_CLOSURE_SEAL:
            raise TypeError("primary runtime closure factory seal is invalid")
        bundle = _validated_operational_bundle(self.operational_bundle)
        if not self.artifact_path.is_absolute():
            raise ValueError("primary runtime closure path must be absolute")
        _digest(self.file_sha256, name="primary closure file_sha256")
        _digest(self.closure_sha256, name="primary closure closure_sha256")
        if type(self.size_bytes) is not int or self.size_bytes < 1:
            raise ValueError("primary runtime closure size must be positive")
        if (
            type(self._canonical_bytes) is not bytes
            or len(self._canonical_bytes) != self.size_bytes
            or _sha256(self._canonical_bytes) != self.file_sha256
        ):
            raise ValueError("primary runtime closure canonical bytes are invalid")
        payload = _validated_primary_closure_payload(
            self._decoded(),
            operational_bundle=bundle,
        )
        if payload["closure_sha256"] != self.closure_sha256 or payload["status"] != self.status:
            raise ValueError("primary runtime closure selected fields are inconsistent")

    @property
    def admission_payload(self) -> dict[str, object]:
        return _validate_embedded_canonical(self._decoded()["admission"], name="admission")

    @property
    def runtime_payload(self) -> dict[str, object]:
        return _validate_embedded_canonical(self._decoded()["runtime"], name="runtime")

    @property
    def outcome_payload(self) -> dict[str, object]:
        return _validate_embedded_canonical(self._decoded()["outcome"], name="outcome")

    @property
    def job_selector(self) -> str:
        runtime = self.runtime_payload
        return f"{runtime['array_job_id']}_{runtime['array_task_id']}"

    @property
    def job_id(self) -> str:
        return str(self.runtime_payload["job_id"])

    @property
    def requested_walltime_seconds(self) -> int:
        return int(self.runtime_payload["requested_walltime_seconds"])

    @property
    def continuation_required(self) -> bool:
        return self.status == "continuation_required_after_safe_checkpoint"

    def validate_integrity(self) -> None:
        self._validate_sealed_payload()
        transport = read_canonical_artifact(
            self.artifact_path,
            expected_file_sha256=self.file_sha256,
        )
        if transport.canonical_bytes != self._canonical_bytes:
            raise ValueError("live primary runtime closure differs from sealed bytes")

    def to_dict(self) -> dict[str, object]:
        self.validate_integrity()
        return self._decoded()


def _primary_closure_from_transport(
    operational_bundle: VerifiedGatePOperationalBundle,
    *,
    artifact_path: Path,
    expected_file_sha256: str,
) -> PrimarySegmentRuntimeClosure:
    bundle = _validated_operational_bundle(operational_bundle)
    transport = read_canonical_artifact(
        artifact_path,
        expected_file_sha256=expected_file_sha256,
    )
    payload = _validated_primary_closure_payload(
        transport.payload,
        operational_bundle=bundle,
    )
    result = PrimarySegmentRuntimeClosure(
        operational_bundle=bundle,
        artifact_path=transport.artifact_path,
        file_sha256=transport.file_sha256,
        size_bytes=transport.size_bytes,
        closure_sha256=str(payload["closure_sha256"]),
        status=str(payload["status"]),
        _canonical_bytes=transport.canonical_bytes,
        _factory_token=_PRIMARY_CLOSURE_SEAL,
    )
    result.validate_integrity()
    return result


def publish_primary_segment_runtime_closure(
    artifact_path: str | os.PathLike[str],
    *,
    admission: PrimarySegmentAdmission,
    runtime: SlurmSegmentRuntime,
    outcome: R3PrimarySegmentOutcome,
    operational_bundle: VerifiedGatePOperationalBundle,
) -> PrimarySegmentRuntimeClosure:
    """Inside the job, seal exact admission/runtime/outcome canonical bytes."""

    admitted, live_runtime, _, bundle, outcome_payload = _validate_primary_dependencies(
        admission,
        runtime=runtime,
        outcome=outcome,
        operational_bundle=operational_bundle,
    )
    payload = _primary_closure_payload(
        admission=admitted,
        runtime=live_runtime,
        outcome_payload=outcome_payload,
        operational_bundle=bundle,
    )
    transport = publish_canonical_artifact(artifact_path, payload)
    return _primary_closure_from_transport(
        bundle,
        artifact_path=transport.artifact_path,
        expected_file_sha256=transport.file_sha256,
    )


def reopen_primary_segment_runtime_closure(
    artifact_path: str | os.PathLike[str],
    *,
    expected_file_sha256: str,
    operational_bundle: VerifiedGatePOperationalBundle,
) -> PrimarySegmentRuntimeClosure:
    """Reopen a job closure from caller-hashed canonical pure-data evidence."""

    return _primary_closure_from_transport(
        operational_bundle,
        artifact_path=Path(artifact_path),
        expected_file_sha256=expected_file_sha256,
    )


def _evidence_binding(
    inspection: ClaimFreeSacctTerminalInspection,
    *,
    job_selector: str,
) -> dict[str, object]:
    return {
        "locked_sacct_command": list(sacct_terminal_command(job_selector)),
        "raw_sacct_sha256": inspection.raw_sacct_sha256,
        "raw_sacct_size_bytes": inspection.raw_size_bytes,
        "inspection_sha256": inspection.inspection_sha256,
        "observed_row": inspection.row.to_dict(),
    }


def _profile_terminal_payload(
    *,
    operational_bundle: VerifiedGatePOperationalBundle,
    runtime_receipt: ProfileSlurmRuntimeReceipt,
    inspection: ClaimFreeSacctTerminalInspection,
) -> dict[str, object]:
    return {
        "schema_version": SUCCESSFUL_PROFILE_TERMINAL_SCHEMA,
        "role": SUCCESSFUL_PROFILE_TERMINAL_ROLE,
        "operational_bundle_file_sha256": operational_bundle.file_sha256,
        "operational_bundle_semantic_sha256": (operational_bundle.bundle_semantic_sha256),
        "profile_run_sha256": operational_bundle.profile_run_sha256,
        "formal_profile_sha256": operational_bundle.formal_profile_sha256,
        "resource_plan_sha256": operational_bundle.resource_plan_sha256,
        "profile_runtime_receipt_file_sha256": runtime_receipt.file_sha256,
        "profile_runtime_receipt_sha256": (runtime_receipt.runtime_receipt_sha256),
        "profile_allocation_intent_file_sha256": (runtime_receipt.allocation_intent.file_sha256),
        "profile_allocation_intent_sha256": (
            runtime_receipt.allocation_intent.allocation_intent_sha256
        ),
        "expected_slurm_resources": (runtime_receipt.allocation_intent.expected_slurm_resources()),
        "scheduler_evidence": _evidence_binding(
            inspection,
            job_selector=runtime_receipt.sacct_job_selector,
        ),
        "gate_p_authorization_issued": False,
    }


def _primary_terminal_payload(
    *,
    runtime_closure: PrimarySegmentRuntimeClosure,
    operational_bundle: VerifiedGatePOperationalBundle,
    inspection: ClaimFreeSacctTerminalInspection,
    schema_version: str,
    role: str,
    continuation_required: bool,
) -> dict[str, object]:
    runtime_closure.validate_integrity()
    admission = runtime_closure.admission_payload
    runtime = runtime_closure.runtime_payload
    outcome = runtime_closure.outcome_payload
    return {
        "schema_version": schema_version,
        "role": role,
        "primary_runtime_closure_file_sha256": runtime_closure.file_sha256,
        "primary_runtime_closure_sha256": runtime_closure.closure_sha256,
        "primary_runtime_closure_producer_schema": (
            PRIMARY_SEGMENT_RUNTIME_CLOSURE_PRODUCER_SCHEMA
        ),
        "design_sha256": admission["design_sha256"],
        "admission_sha256": admission["admission_sha256"],
        "logical_run_id": admission["logical_run_id"],
        "scheduler_segment_id": admission["scheduler_segment_id"],
        "segment_index": admission["segment_index"],
        "task_id": admission["task_id"],
        "seed": admission["seed"],
        "runtime_sha256": runtime["runtime_sha256"],
        "segment_outcome_sha256": outcome["outcome_sha256"],
        "segment_outcome_status": outcome["status"],
        "continuation_checkpoint": outcome["continuation_checkpoint"],
        "completed_head_receipts": outcome["completed_heads"],
        "continuation_required": continuation_required,
        "operational_bundle_file_sha256": operational_bundle.file_sha256,
        "operational_bundle_semantic_sha256": (operational_bundle.bundle_semantic_sha256),
        "profile_run_sha256": operational_bundle.profile_run_sha256,
        "formal_profile_sha256": operational_bundle.formal_profile_sha256,
        "resource_plan_sha256": operational_bundle.resource_plan_sha256,
        "expected_slurm_resources": _resource_expectation(operational_bundle),
        "scheduler_evidence": _evidence_binding(
            inspection,
            job_selector=runtime_closure.job_selector,
        ),
        "continuation_authorization_issued": False,
        "final_three_seed_authorization_issued": False,
    }


def _parsed_evidence_payload(
    inspection: ClaimFreeSacctTerminalInspection,
) -> dict[str, object]:
    return {
        "schema_version": SACCT_TERMINAL_PARSED_EVIDENCE_SCHEMA,
        "formal_claim_eligible": False,
        "inspection": inspection.to_dict(),
    }


def _manifest_payload(
    *,
    producer_kind: str,
    job_selector: str,
    inspection: ClaimFreeSacctTerminalInspection,
    parsed_file_sha256: str,
    parsed_size_bytes: int,
    terminal_artifact: Mapping[str, object],
) -> dict[str, object]:
    unsigned: dict[str, object] = {
        "schema_version": SACCT_TERMINAL_MANIFEST_SCHEMA,
        "producer_kind": producer_kind,
        "json_role_is_authority": False,
        "locked_sacct_command": list(sacct_terminal_command(job_selector)),
        "raw_sacct": {
            "filename": _RAW_FILENAME,
            "sha256": inspection.raw_sacct_sha256,
            "size_bytes": inspection.raw_size_bytes,
        },
        "parsed_evidence": {
            "filename": _PARSED_FILENAME,
            "file_sha256": parsed_file_sha256,
            "size_bytes": parsed_size_bytes,
            "inspection_sha256": inspection.inspection_sha256,
        },
        "terminal_artifact": dict(terminal_artifact),
    }
    return {**unsigned, "manifest_sha256": _canonical_sha256(unsigned)}


def _create_bundle_directory(path: Path) -> Path:
    if not path.is_absolute():
        raise ValueError("terminal evidence directory must be absolute")
    parent = _canonical_directory(path.parent, name="terminal evidence parent")
    if path.exists() or path.is_symlink():
        raise FileExistsError("refusing to overwrite a terminal evidence directory")
    try:
        path.mkdir(mode=0o750)
    except FileExistsError as error:
        raise FileExistsError("refusing to overwrite a terminal evidence directory") from error
    _fsync_directory(parent)
    return _canonical_directory(path, name="terminal evidence directory")


def _publish_bundle(
    bundle_directory: Path,
    *,
    producer_kind: str,
    job_selector: str,
    inspection: ClaimFreeSacctTerminalInspection,
    terminal_payload: Mapping[str, object],
) -> str:
    directory = _create_bundle_directory(bundle_directory)
    terminal_artifact = {
        **dict(terminal_payload),
        "terminal_sha256": _canonical_sha256(terminal_payload),
    }
    _publish_immutable_bytes(directory / _RAW_FILENAME, inspection.raw_bytes)
    parsed = publish_canonical_artifact(
        directory / _PARSED_FILENAME,
        _parsed_evidence_payload(inspection),
    )
    manifest_payload = _manifest_payload(
        producer_kind=producer_kind,
        job_selector=job_selector,
        inspection=inspection,
        parsed_file_sha256=parsed.file_sha256,
        parsed_size_bytes=parsed.size_bytes,
        terminal_artifact=terminal_artifact,
    )
    manifest = publish_canonical_artifact(
        directory / _MANIFEST_FILENAME,
        manifest_payload,
    )
    return manifest.file_sha256


def _require_binding(
    value: object,
    *,
    name: str,
    keys: set[str],
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{name} has an invalid closed field set")
    return dict(value)


def _load_bundle(
    bundle_directory: Path,
    *,
    expected_manifest_file_sha256: str,
    expected_raw_sacct_sha256: str,
    expected_producer_kind: str,
) -> tuple[ClaimFreeSacctTerminalInspection, dict[str, object]]:
    directory = _canonical_directory(
        bundle_directory,
        name="terminal evidence directory",
    )
    manifest_transport = read_canonical_artifact(
        directory / _MANIFEST_FILENAME,
        expected_file_sha256=expected_manifest_file_sha256,
    )
    manifest = manifest_transport.payload
    if set(manifest) != {
        "schema_version",
        "producer_kind",
        "json_role_is_authority",
        "locked_sacct_command",
        "raw_sacct",
        "parsed_evidence",
        "terminal_artifact",
        "manifest_sha256",
    }:
        raise ValueError("terminal evidence manifest has an invalid closed field set")
    unsigned = dict(manifest)
    manifest_sha = unsigned.pop("manifest_sha256")
    if (
        manifest["schema_version"] != SACCT_TERMINAL_MANIFEST_SCHEMA
        or manifest["producer_kind"] != expected_producer_kind
        or manifest["json_role_is_authority"] is not False
        or manifest_sha != _canonical_sha256(unsigned)
    ):
        raise ValueError("terminal evidence manifest identity/self-hash is invalid")
    raw_binding = _require_binding(
        manifest["raw_sacct"],
        name="raw_sacct manifest binding",
        keys={"filename", "sha256", "size_bytes"},
    )
    expected_raw = _digest(
        expected_raw_sacct_sha256,
        name="expected_raw_sacct_sha256",
    )
    if (
        raw_binding["filename"] != _RAW_FILENAME
        or raw_binding["sha256"] != expected_raw
        or type(raw_binding["size_bytes"]) is not int
        or raw_binding["size_bytes"] < 1
    ):
        raise ValueError("manifest raw sacct binding differs from caller expectation")
    raw = _read_stable_bytes(
        directory / _RAW_FILENAME,
        expected_file_sha256=expected_raw,
        require_published_mode=True,
    )
    if len(raw) != raw_binding["size_bytes"]:
        raise ValueError("manifest raw sacct size is invalid")
    inspection = inspect_sacct_terminal_bytes(
        raw,
        expected_raw_sha256=expected_raw,
    )

    parsed_binding = _require_binding(
        manifest["parsed_evidence"],
        name="parsed_evidence manifest binding",
        keys={"filename", "file_sha256", "size_bytes", "inspection_sha256"},
    )
    if (
        parsed_binding["filename"] != _PARSED_FILENAME
        or parsed_binding["inspection_sha256"] != inspection.inspection_sha256
        or type(parsed_binding["file_sha256"]) is not str
        or type(parsed_binding["size_bytes"]) is not int
        or parsed_binding["size_bytes"] < 1
    ):
        raise ValueError("manifest parsed evidence binding is invalid")
    parsed_transport = read_canonical_artifact(
        directory / _PARSED_FILENAME,
        expected_file_sha256=_digest(
            parsed_binding["file_sha256"],
            name="parsed evidence file SHA-256",
        ),
    )
    if (
        parsed_transport.size_bytes != parsed_binding["size_bytes"]
        or parsed_transport.payload != _parsed_evidence_payload(inspection)
        or parsed_transport.canonical_bytes
        != canonical_json_bytes(_parsed_evidence_payload(inspection))
    ):
        raise ValueError("parsed evidence differs from the original sacct bytes")
    terminal = manifest["terminal_artifact"]
    if not isinstance(terminal, Mapping):
        raise ValueError("manifest terminal_artifact must be a mapping")
    scheduler_evidence = terminal.get("scheduler_evidence")
    if not isinstance(scheduler_evidence, Mapping) or manifest[
        "locked_sacct_command"
    ] != scheduler_evidence.get("locked_sacct_command"):
        raise ValueError("manifest locked sacct command is not terminal-bound")
    return inspection, dict(terminal)


@dataclass(frozen=True, slots=True)
class SuccessfulProfileTerminalCapability:
    """Sealed evidence that one exact formal profile allocation succeeded."""

    operational_bundle: VerifiedGatePOperationalBundle = field(
        repr=False,
        compare=False,
    )
    runtime_receipt: ProfileSlurmRuntimeReceipt = field(
        repr=False,
        compare=False,
    )
    inspection: ClaimFreeSacctTerminalInspection = field(repr=False, compare=False)
    evidence_directory: Path
    manifest_file_sha256: str
    terminal_sha256: str
    _seal: object | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _digest(self.manifest_file_sha256, name="manifest_file_sha256")
        _digest(self.terminal_sha256, name="terminal_sha256")
        if not self.evidence_directory.is_absolute():
            raise ValueError("evidence_directory must be absolute")

    def _expected_payload(self) -> dict[str, object]:
        bundle, receipt = _validate_profile_dependencies(
            self.operational_bundle,
            runtime_receipt=self.runtime_receipt,
        )
        _validate_terminal_row(
            self.inspection,
            expected_job_id=receipt.sacct_job_selector,
            expected_job_id_raw=receipt.job_id,
            expected_resources=(receipt.allocation_intent.expected_slurm_resources()),
            requested_walltime_seconds=receipt.requested_walltime_seconds,
        )
        return _profile_terminal_payload(
            operational_bundle=bundle,
            runtime_receipt=receipt,
            inspection=self.inspection,
        )

    def validate_integrity(self) -> None:
        self.__post_init__()
        if self._seal is not _CAPABILITY_SEAL:
            raise TypeError("profile terminal capability must be produced by its private validator")
        expected = self._expected_payload()
        if self.terminal_sha256 != _canonical_sha256(expected):
            raise ValueError("profile terminal capability SHA-256 is invalid")
        loaded_inspection, terminal = _load_bundle(
            self.evidence_directory,
            expected_manifest_file_sha256=self.manifest_file_sha256,
            expected_raw_sacct_sha256=self.inspection.raw_sacct_sha256,
            expected_producer_kind=_PROFILE_PRODUCER_KIND,
        )
        if loaded_inspection.raw_bytes != self.inspection.raw_bytes or terminal != {
            **expected,
            "terminal_sha256": self.terminal_sha256,
        }:
            raise ValueError("profile terminal manifest differs from live validation")

    def to_dict(self) -> dict[str, object]:
        self.validate_integrity()
        return {**self._expected_payload(), "terminal_sha256": self.terminal_sha256}

    def artifact_ref(self) -> ArtifactRef:
        self.validate_integrity()
        return ArtifactRef(
            schema_version=SUCCESSFUL_PROFILE_TERMINAL_SCHEMA,
            artifact_sha256=self.terminal_sha256,
            role=SUCCESSFUL_PROFILE_TERMINAL_ROLE,
        )


def _validate_primary_closure_dependency(
    operational_bundle: VerifiedGatePOperationalBundle,
    *,
    runtime_closure: PrimarySegmentRuntimeClosure,
    expected_status: str,
) -> tuple[VerifiedGatePOperationalBundle, PrimarySegmentRuntimeClosure]:
    bundle = _validated_operational_bundle(operational_bundle)
    if type(runtime_closure) is not PrimarySegmentRuntimeClosure:
        raise TypeError("runtime_closure must be exactly PrimarySegmentRuntimeClosure")
    runtime_closure.validate_integrity()
    closure_bundle = runtime_closure.operational_bundle
    if (
        closure_bundle.file_sha256 != bundle.file_sha256
        or closure_bundle.bundle_semantic_sha256 != bundle.bundle_semantic_sha256
        or closure_bundle.profile_run_sha256 != bundle.profile_run_sha256
        or closure_bundle.formal_profile_sha256 != bundle.formal_profile_sha256
        or closure_bundle.resource_plan_sha256 != bundle.resource_plan_sha256
    ):
        raise ValueError("primary runtime closure belongs to another operational bundle")
    if runtime_closure.status != expected_status:
        raise ValueError("primary runtime closure status is invalid for this capability")
    return bundle, runtime_closure


@dataclass(frozen=True, slots=True)
class ContinuablePrimaryTerminalCapability:
    """Scheduler evidence only; continuation authorization is issued elsewhere."""

    operational_bundle: VerifiedGatePOperationalBundle = field(
        repr=False,
        compare=False,
    )
    runtime_closure: PrimarySegmentRuntimeClosure = field(repr=False, compare=False)
    inspection: ClaimFreeSacctTerminalInspection = field(repr=False, compare=False)
    evidence_directory: Path
    manifest_file_sha256: str
    terminal_sha256: str
    _seal: object | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _digest(self.manifest_file_sha256, name="manifest_file_sha256")
        _digest(self.terminal_sha256, name="terminal_sha256")
        if not self.evidence_directory.is_absolute():
            raise ValueError("evidence_directory must be absolute")

    def _expected_payload(self) -> dict[str, object]:
        bundle, closure = _validate_primary_closure_dependency(
            self.operational_bundle,
            runtime_closure=self.runtime_closure,
            expected_status="continuation_required_after_safe_checkpoint",
        )
        _validate_terminal_row(
            self.inspection,
            expected_job_id=closure.job_selector,
            expected_job_id_raw=closure.job_id,
            expected_resources=_resource_expectation(bundle),
            requested_walltime_seconds=closure.requested_walltime_seconds,
        )
        return _primary_terminal_payload(
            runtime_closure=closure,
            operational_bundle=bundle,
            inspection=self.inspection,
            schema_version=CONTINUABLE_PRIMARY_TERMINAL_SCHEMA,
            role=CONTINUABLE_PRIMARY_TERMINAL_ROLE,
            continuation_required=True,
        )

    @property
    def continuation_required(self) -> bool:
        return True

    def validate_integrity(self) -> None:
        self.__post_init__()
        if self._seal is not _CAPABILITY_SEAL:
            raise TypeError("primary terminal capability must be produced by its private validator")
        expected = self._expected_payload()
        if self.terminal_sha256 != _canonical_sha256(expected):
            raise ValueError("primary terminal capability SHA-256 is invalid")
        loaded_inspection, terminal = _load_bundle(
            self.evidence_directory,
            expected_manifest_file_sha256=self.manifest_file_sha256,
            expected_raw_sacct_sha256=self.inspection.raw_sacct_sha256,
            expected_producer_kind=_CONTINUABLE_PRIMARY_PRODUCER_KIND,
        )
        if loaded_inspection.raw_bytes != self.inspection.raw_bytes or terminal != {
            **expected,
            "terminal_sha256": self.terminal_sha256,
        }:
            raise ValueError("primary terminal manifest differs from fresh validation")

    def to_dict(self) -> dict[str, object]:
        self.validate_integrity()
        return {**self._expected_payload(), "terminal_sha256": self.terminal_sha256}

    def artifact_ref(self) -> ArtifactRef:
        self.validate_integrity()
        return ArtifactRef(
            schema_version=CONTINUABLE_PRIMARY_TERMINAL_SCHEMA,
            artifact_sha256=self.terminal_sha256,
            role=CONTINUABLE_PRIMARY_TERMINAL_ROLE,
        )


@dataclass(frozen=True, slots=True)
class CompletedPrimaryTerminalCapability:
    """Scheduler evidence for compute-complete state; not final R3 authorization."""

    operational_bundle: VerifiedGatePOperationalBundle = field(
        repr=False,
        compare=False,
    )
    runtime_closure: PrimarySegmentRuntimeClosure = field(repr=False, compare=False)
    inspection: ClaimFreeSacctTerminalInspection = field(repr=False, compare=False)
    evidence_directory: Path
    manifest_file_sha256: str
    terminal_sha256: str
    _seal: object | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _digest(self.manifest_file_sha256, name="manifest_file_sha256")
        _digest(self.terminal_sha256, name="terminal_sha256")
        if not self.evidence_directory.is_absolute():
            raise ValueError("evidence_directory must be absolute")

    def _expected_payload(self) -> dict[str, object]:
        bundle, closure = _validate_primary_closure_dependency(
            self.operational_bundle,
            runtime_closure=self.runtime_closure,
            expected_status="compute_complete_pending_external_scheduler_terminal",
        )
        _validate_terminal_row(
            self.inspection,
            expected_job_id=closure.job_selector,
            expected_job_id_raw=closure.job_id,
            expected_resources=_resource_expectation(bundle),
            requested_walltime_seconds=closure.requested_walltime_seconds,
        )
        return _primary_terminal_payload(
            runtime_closure=closure,
            operational_bundle=bundle,
            inspection=self.inspection,
            schema_version=COMPLETED_PRIMARY_TERMINAL_SCHEMA,
            role=COMPLETED_PRIMARY_TERMINAL_ROLE,
            continuation_required=False,
        )

    @property
    def continuation_required(self) -> bool:
        return False

    def validate_integrity(self) -> None:
        self.__post_init__()
        if self._seal is not _CAPABILITY_SEAL:
            raise TypeError("primary terminal capability must be produced by its private validator")
        expected = self._expected_payload()
        if self.terminal_sha256 != _canonical_sha256(expected):
            raise ValueError("primary terminal capability SHA-256 is invalid")
        loaded_inspection, terminal = _load_bundle(
            self.evidence_directory,
            expected_manifest_file_sha256=self.manifest_file_sha256,
            expected_raw_sacct_sha256=self.inspection.raw_sacct_sha256,
            expected_producer_kind=_COMPLETED_PRIMARY_PRODUCER_KIND,
        )
        if loaded_inspection.raw_bytes != self.inspection.raw_bytes or terminal != {
            **expected,
            "terminal_sha256": self.terminal_sha256,
        }:
            raise ValueError("primary terminal manifest differs from fresh validation")

    def to_dict(self) -> dict[str, object]:
        self.validate_integrity()
        return {**self._expected_payload(), "terminal_sha256": self.terminal_sha256}

    def artifact_ref(self) -> ArtifactRef:
        self.validate_integrity()
        return ArtifactRef(
            schema_version=COMPLETED_PRIMARY_TERMINAL_SCHEMA,
            artifact_sha256=self.terminal_sha256,
            role=COMPLETED_PRIMARY_TERMINAL_ROLE,
        )


def produce_successful_profile_terminal(
    operational_bundle: VerifiedGatePOperationalBundle,
    *,
    runtime_receipt: ProfileSlurmRuntimeReceipt,
    inspection: ClaimFreeSacctTerminalInspection,
    evidence_directory: str | os.PathLike[str],
) -> SuccessfulProfileTerminalCapability:
    """Post-job validator using only sealed operational/runtime evidence."""

    bundle, receipt = _validate_profile_dependencies(
        operational_bundle,
        runtime_receipt=runtime_receipt,
    )
    _validate_terminal_row(
        inspection,
        expected_job_id=receipt.sacct_job_selector,
        expected_job_id_raw=receipt.job_id,
        expected_resources=receipt.allocation_intent.expected_slurm_resources(),
        requested_walltime_seconds=receipt.requested_walltime_seconds,
    )
    payload = _profile_terminal_payload(
        operational_bundle=bundle,
        runtime_receipt=receipt,
        inspection=inspection,
    )
    directory = Path(evidence_directory)
    manifest_file_sha256 = _publish_bundle(
        directory,
        producer_kind=_PROFILE_PRODUCER_KIND,
        job_selector=receipt.sacct_job_selector,
        inspection=inspection,
        terminal_payload=payload,
    )
    result = SuccessfulProfileTerminalCapability(
        operational_bundle=bundle,
        runtime_receipt=receipt,
        inspection=inspection,
        evidence_directory=directory,
        manifest_file_sha256=manifest_file_sha256,
        terminal_sha256=_canonical_sha256(payload),
    )
    object.__setattr__(result, "_seal", _CAPABILITY_SEAL)
    result.validate_integrity()
    return result


def produce_continuable_primary_terminal(
    operational_bundle: VerifiedGatePOperationalBundle,
    *,
    runtime_closure: PrimarySegmentRuntimeClosure,
    inspection: ClaimFreeSacctTerminalInspection,
    evidence_directory: str | os.PathLike[str],
) -> ContinuablePrimaryTerminalCapability:
    """Mint scheduler evidence for an exact continuation closure."""

    bundle, closure = _validate_primary_closure_dependency(
        operational_bundle,
        runtime_closure=runtime_closure,
        expected_status="continuation_required_after_safe_checkpoint",
    )
    _validate_terminal_row(
        inspection,
        expected_job_id=closure.job_selector,
        expected_job_id_raw=closure.job_id,
        expected_resources=_resource_expectation(bundle),
        requested_walltime_seconds=closure.requested_walltime_seconds,
    )
    payload = _primary_terminal_payload(
        runtime_closure=closure,
        operational_bundle=bundle,
        inspection=inspection,
        schema_version=CONTINUABLE_PRIMARY_TERMINAL_SCHEMA,
        role=CONTINUABLE_PRIMARY_TERMINAL_ROLE,
        continuation_required=True,
    )
    directory = Path(evidence_directory)
    manifest_file_sha256 = _publish_bundle(
        directory,
        producer_kind=_CONTINUABLE_PRIMARY_PRODUCER_KIND,
        job_selector=closure.job_selector,
        inspection=inspection,
        terminal_payload=payload,
    )
    result = ContinuablePrimaryTerminalCapability(
        operational_bundle=bundle,
        runtime_closure=closure,
        inspection=inspection,
        evidence_directory=directory,
        manifest_file_sha256=manifest_file_sha256,
        terminal_sha256=_canonical_sha256(payload),
    )
    object.__setattr__(result, "_seal", _CAPABILITY_SEAL)
    result.validate_integrity()
    return result


def produce_completed_primary_terminal(
    operational_bundle: VerifiedGatePOperationalBundle,
    *,
    runtime_closure: PrimarySegmentRuntimeClosure,
    inspection: ClaimFreeSacctTerminalInspection,
    evidence_directory: str | os.PathLike[str],
) -> CompletedPrimaryTerminalCapability:
    """Mint scheduler evidence for an exact compute-complete closure."""

    bundle, closure = _validate_primary_closure_dependency(
        operational_bundle,
        runtime_closure=runtime_closure,
        expected_status="compute_complete_pending_external_scheduler_terminal",
    )
    _validate_terminal_row(
        inspection,
        expected_job_id=closure.job_selector,
        expected_job_id_raw=closure.job_id,
        expected_resources=_resource_expectation(bundle),
        requested_walltime_seconds=closure.requested_walltime_seconds,
    )
    payload = _primary_terminal_payload(
        runtime_closure=closure,
        operational_bundle=bundle,
        inspection=inspection,
        schema_version=COMPLETED_PRIMARY_TERMINAL_SCHEMA,
        role=COMPLETED_PRIMARY_TERMINAL_ROLE,
        continuation_required=False,
    )
    directory = Path(evidence_directory)
    manifest_file_sha256 = _publish_bundle(
        directory,
        producer_kind=_COMPLETED_PRIMARY_PRODUCER_KIND,
        job_selector=closure.job_selector,
        inspection=inspection,
        terminal_payload=payload,
    )
    result = CompletedPrimaryTerminalCapability(
        operational_bundle=bundle,
        runtime_closure=closure,
        inspection=inspection,
        evidence_directory=directory,
        manifest_file_sha256=manifest_file_sha256,
        terminal_sha256=_canonical_sha256(payload),
    )
    object.__setattr__(result, "_seal", _CAPABILITY_SEAL)
    result.validate_integrity()
    return result


def revalidate_successful_profile_terminal(
    operational_bundle: VerifiedGatePOperationalBundle,
    *,
    runtime_receipt: ProfileSlurmRuntimeReceipt,
    evidence_directory: str | os.PathLike[str],
    expected_manifest_file_sha256: str,
    expected_raw_sacct_sha256: str,
) -> SuccessfulProfileTerminalCapability:
    """Regain a profile capability only after raw+manifest revalidation."""

    bundle, receipt = _validate_profile_dependencies(
        operational_bundle,
        runtime_receipt=runtime_receipt,
    )
    directory = Path(evidence_directory)
    inspection, terminal = _load_bundle(
        directory,
        expected_manifest_file_sha256=expected_manifest_file_sha256,
        expected_raw_sacct_sha256=expected_raw_sacct_sha256,
        expected_producer_kind=_PROFILE_PRODUCER_KIND,
    )
    _validate_terminal_row(
        inspection,
        expected_job_id=receipt.sacct_job_selector,
        expected_job_id_raw=receipt.job_id,
        expected_resources=receipt.allocation_intent.expected_slurm_resources(),
        requested_walltime_seconds=receipt.requested_walltime_seconds,
    )
    payload = _profile_terminal_payload(
        operational_bundle=bundle,
        runtime_receipt=receipt,
        inspection=inspection,
    )
    terminal_sha256 = _canonical_sha256(payload)
    if terminal != {**payload, "terminal_sha256": terminal_sha256}:
        raise ValueError("manifest profile terminal differs from fresh validation")
    result = SuccessfulProfileTerminalCapability(
        operational_bundle=bundle,
        runtime_receipt=receipt,
        inspection=inspection,
        evidence_directory=directory,
        manifest_file_sha256=expected_manifest_file_sha256,
        terminal_sha256=terminal_sha256,
    )
    object.__setattr__(result, "_seal", _CAPABILITY_SEAL)
    result.validate_integrity()
    return result


def revalidate_continuable_primary_terminal(
    operational_bundle: VerifiedGatePOperationalBundle,
    *,
    runtime_closure: PrimarySegmentRuntimeClosure,
    evidence_directory: str | os.PathLike[str],
    expected_manifest_file_sha256: str,
    expected_raw_sacct_sha256: str,
) -> ContinuablePrimaryTerminalCapability:
    """Regain a primary capability only after raw+manifest revalidation."""

    bundle, closure = _validate_primary_closure_dependency(
        operational_bundle,
        runtime_closure=runtime_closure,
        expected_status="continuation_required_after_safe_checkpoint",
    )
    directory = Path(evidence_directory)
    inspection, terminal = _load_bundle(
        directory,
        expected_manifest_file_sha256=expected_manifest_file_sha256,
        expected_raw_sacct_sha256=expected_raw_sacct_sha256,
        expected_producer_kind=_CONTINUABLE_PRIMARY_PRODUCER_KIND,
    )
    _validate_terminal_row(
        inspection,
        expected_job_id=closure.job_selector,
        expected_job_id_raw=closure.job_id,
        expected_resources=_resource_expectation(bundle),
        requested_walltime_seconds=closure.requested_walltime_seconds,
    )
    payload = _primary_terminal_payload(
        runtime_closure=closure,
        operational_bundle=bundle,
        inspection=inspection,
        schema_version=CONTINUABLE_PRIMARY_TERMINAL_SCHEMA,
        role=CONTINUABLE_PRIMARY_TERMINAL_ROLE,
        continuation_required=True,
    )
    terminal_sha256 = _canonical_sha256(payload)
    if terminal != {**payload, "terminal_sha256": terminal_sha256}:
        raise ValueError("manifest primary terminal differs from fresh validation")
    result = ContinuablePrimaryTerminalCapability(
        operational_bundle=bundle,
        runtime_closure=closure,
        inspection=inspection,
        evidence_directory=directory,
        manifest_file_sha256=expected_manifest_file_sha256,
        terminal_sha256=terminal_sha256,
    )
    object.__setattr__(result, "_seal", _CAPABILITY_SEAL)
    result.validate_integrity()
    return result


def revalidate_completed_primary_terminal(
    operational_bundle: VerifiedGatePOperationalBundle,
    *,
    runtime_closure: PrimarySegmentRuntimeClosure,
    evidence_directory: str | os.PathLike[str],
    expected_manifest_file_sha256: str,
    expected_raw_sacct_sha256: str,
) -> CompletedPrimaryTerminalCapability:
    """Regain completed scheduler evidence after raw+manifest revalidation."""

    bundle, closure = _validate_primary_closure_dependency(
        operational_bundle,
        runtime_closure=runtime_closure,
        expected_status="compute_complete_pending_external_scheduler_terminal",
    )
    directory = Path(evidence_directory)
    inspection, terminal = _load_bundle(
        directory,
        expected_manifest_file_sha256=expected_manifest_file_sha256,
        expected_raw_sacct_sha256=expected_raw_sacct_sha256,
        expected_producer_kind=_COMPLETED_PRIMARY_PRODUCER_KIND,
    )
    _validate_terminal_row(
        inspection,
        expected_job_id=closure.job_selector,
        expected_job_id_raw=closure.job_id,
        expected_resources=_resource_expectation(bundle),
        requested_walltime_seconds=closure.requested_walltime_seconds,
    )
    payload = _primary_terminal_payload(
        runtime_closure=closure,
        operational_bundle=bundle,
        inspection=inspection,
        schema_version=COMPLETED_PRIMARY_TERMINAL_SCHEMA,
        role=COMPLETED_PRIMARY_TERMINAL_ROLE,
        continuation_required=False,
    )
    terminal_sha256 = _canonical_sha256(payload)
    if terminal != {**payload, "terminal_sha256": terminal_sha256}:
        raise ValueError("manifest completed primary terminal differs from fresh validation")
    result = CompletedPrimaryTerminalCapability(
        operational_bundle=bundle,
        runtime_closure=closure,
        inspection=inspection,
        evidence_directory=directory,
        manifest_file_sha256=expected_manifest_file_sha256,
        terminal_sha256=terminal_sha256,
    )
    object.__setattr__(result, "_seal", _CAPABILITY_SEAL)
    result.validate_integrity()
    return result


def finalize_successful_profile_terminal_from_files(
    *,
    operational_bundle_path: str | os.PathLike[str],
    expected_operational_bundle_file_sha256: str,
    allocation_intent_path: str | os.PathLike[str],
    expected_allocation_intent_file_sha256: str,
    runtime_receipt_path: str | os.PathLike[str],
    expected_runtime_receipt_file_sha256: str,
    raw_sacct_path: str | os.PathLike[str],
    expected_raw_sacct_sha256: str,
    evidence_directory: str | os.PathLike[str],
) -> SuccessfulProfileTerminalCapability:
    """Post-job finalizer that opens only caller-hashed pure-data evidence."""

    bundle = reopen_verified_gate_p_operational_bundle(
        operational_bundle_path,
        expected_file_sha256=expected_operational_bundle_file_sha256,
    )
    intent = reopen_profile_allocation_intent(
        allocation_intent_path,
        expected_file_sha256=expected_allocation_intent_file_sha256,
    )
    receipt = reopen_profile_slurm_runtime_receipt(
        runtime_receipt_path,
        expected_file_sha256=expected_runtime_receipt_file_sha256,
        operational_bundle=bundle,
        allocation_intent=intent,
    )
    inspection = inspect_sacct_terminal_file(
        raw_sacct_path,
        expected_raw_sha256=expected_raw_sacct_sha256,
    )
    return produce_successful_profile_terminal(
        bundle,
        runtime_receipt=receipt,
        inspection=inspection,
        evidence_directory=evidence_directory,
    )


finalize_successful_profile_terminal = finalize_successful_profile_terminal_from_files


def finalize_continuable_primary_terminal_from_files(
    *,
    operational_bundle_path: str | os.PathLike[str],
    expected_operational_bundle_file_sha256: str,
    runtime_closure_path: str | os.PathLike[str],
    expected_runtime_closure_file_sha256: str,
    raw_sacct_path: str | os.PathLike[str],
    expected_raw_sacct_sha256: str,
    evidence_directory: str | os.PathLike[str],
) -> ContinuablePrimaryTerminalCapability:
    """Post-job finalizer opening only bundle, closure, and raw sacct bytes."""

    bundle = reopen_verified_gate_p_operational_bundle(
        operational_bundle_path,
        expected_file_sha256=expected_operational_bundle_file_sha256,
    )
    closure = reopen_primary_segment_runtime_closure(
        runtime_closure_path,
        expected_file_sha256=expected_runtime_closure_file_sha256,
        operational_bundle=bundle,
    )
    inspection = inspect_sacct_terminal_file(
        raw_sacct_path,
        expected_raw_sha256=expected_raw_sacct_sha256,
    )
    return produce_continuable_primary_terminal(
        bundle,
        runtime_closure=closure,
        inspection=inspection,
        evidence_directory=evidence_directory,
    )


def finalize_completed_primary_terminal_from_files(
    *,
    operational_bundle_path: str | os.PathLike[str],
    expected_operational_bundle_file_sha256: str,
    runtime_closure_path: str | os.PathLike[str],
    expected_runtime_closure_file_sha256: str,
    raw_sacct_path: str | os.PathLike[str],
    expected_raw_sacct_sha256: str,
    evidence_directory: str | os.PathLike[str],
) -> CompletedPrimaryTerminalCapability:
    """Post-job completed finalizer; final three-seed authority remains external."""

    bundle = reopen_verified_gate_p_operational_bundle(
        operational_bundle_path,
        expected_file_sha256=expected_operational_bundle_file_sha256,
    )
    closure = reopen_primary_segment_runtime_closure(
        runtime_closure_path,
        expected_file_sha256=expected_runtime_closure_file_sha256,
        operational_bundle=bundle,
    )
    inspection = inspect_sacct_terminal_file(
        raw_sacct_path,
        expected_raw_sha256=expected_raw_sacct_sha256,
    )
    return produce_completed_primary_terminal(
        bundle,
        runtime_closure=closure,
        inspection=inspection,
        evidence_directory=evidence_directory,
    )


finalize_continuable_primary_terminal = finalize_continuable_primary_terminal_from_files
finalize_completed_primary_terminal = finalize_completed_primary_terminal_from_files


def successful_profile_terminal_artifact_ref(
    capability: SuccessfulProfileTerminalCapability,
) -> ArtifactRef:
    """Return the typed terminal ref without issuing Gate-P authorization."""

    if type(capability) is not SuccessfulProfileTerminalCapability:
        raise TypeError("capability must be exactly SuccessfulProfileTerminalCapability")
    return capability.artifact_ref()


def continuable_primary_terminal_artifact_ref(
    capability: ContinuablePrimaryTerminalCapability,
) -> ArtifactRef:
    """Return the typed terminal ref without issuing continuation authorization."""

    if type(capability) is not ContinuablePrimaryTerminalCapability:
        raise TypeError("capability must be exactly ContinuablePrimaryTerminalCapability")
    return capability.artifact_ref()


def completed_primary_terminal_artifact_ref(
    capability: CompletedPrimaryTerminalCapability,
) -> ArtifactRef:
    """Return completed scheduler evidence without final campaign authorization."""

    if type(capability) is not CompletedPrimaryTerminalCapability:
        raise TypeError("capability must be exactly CompletedPrimaryTerminalCapability")
    return capability.artifact_ref()


produce_successful_profile_terminal_capability = produce_successful_profile_terminal
produce_continuable_primary_terminal_capability = produce_continuable_primary_terminal
produce_completed_primary_terminal_capability = produce_completed_primary_terminal
reopen_successful_profile_terminal = revalidate_successful_profile_terminal
reopen_continuable_primary_terminal = revalidate_continuable_primary_terminal
reopen_completed_primary_terminal = revalidate_completed_primary_terminal


__all__ = [
    "COMPLETED_PRIMARY_TERMINAL_ROLE",
    "COMPLETED_PRIMARY_TERMINAL_SCHEMA",
    "PRIMARY_SEGMENT_RUNTIME_CLOSURE_PRODUCER_SCHEMA",
    "PRIMARY_SEGMENT_RUNTIME_CLOSURE_SCHEMA",
    "SACCT_TERMINAL_FIELDS",
    "SACCT_TERMINAL_INSPECTION_SCHEMA",
    "SACCT_TERMINAL_MANIFEST_SCHEMA",
    "SACCT_TERMINAL_PARSED_EVIDENCE_SCHEMA",
    "PROFILE_ALLOCATION_INTENT_ROLE",
    "PROFILE_ALLOCATION_INTENT_SCHEMA",
    "PROFILE_SLURM_RUNTIME_RECEIPT_SCHEMA",
    "PROFILE_SLURM_RUNTIME_SCHEMA",
    "ClaimFreeSacctTerminalInspection",
    "CompletedPrimaryTerminalCapability",
    "ContinuablePrimaryTerminalCapability",
    "FormalProfileJobRuntimeIdentity",
    "ProfileAllocationIntent",
    "ProfileJobRuntimeIdentity",
    "ProfileSlurmRuntimeIdentity",
    "PrimarySegmentRuntimeClosure",
    "SacctAllocationRow",
    "SacctTerminalInspection",
    "SuccessfulProfileTerminalCapability",
    "capture_formal_profile_job_runtime",
    "capture_profile_slurm_runtime_identity",
    "capture_profile_slurm_runtime_receipt",
    "completed_primary_terminal_artifact_ref",
    "continuable_primary_terminal_artifact_ref",
    "finalize_completed_primary_terminal",
    "finalize_completed_primary_terminal_from_files",
    "finalize_continuable_primary_terminal",
    "finalize_continuable_primary_terminal_from_files",
    "finalize_successful_profile_terminal",
    "finalize_successful_profile_terminal_from_files",
    "inspect_raw_sacct_terminal",
    "inspect_sacct_terminal_bytes",
    "inspect_sacct_terminal_file",
    "produce_continuable_primary_terminal",
    "produce_continuable_primary_terminal_capability",
    "produce_completed_primary_terminal",
    "produce_completed_primary_terminal_capability",
    "produce_successful_profile_terminal",
    "produce_successful_profile_terminal_capability",
    "publish_profile_allocation_intent",
    "publish_primary_segment_runtime_closure",
    "reopen_completed_primary_terminal",
    "reopen_continuable_primary_terminal",
    "reopen_primary_segment_runtime_closure",
    "reopen_profile_slurm_runtime_receipt",
    "reopen_profile_allocation_intent",
    "reopen_successful_profile_terminal",
    "revalidate_continuable_primary_terminal",
    "revalidate_completed_primary_terminal",
    "revalidate_successful_profile_terminal",
    "sacct_terminal_command",
    "successful_profile_terminal_artifact_ref",
]
