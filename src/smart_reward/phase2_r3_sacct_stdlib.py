"""Pure-stdlib Slurm allocation-row parsing shared by R3 evidence planes.

This module intentionally has no dependency on PyTorch, PyYAML, the package
``__init__``, or any scientific training implementation.  It is safe to load
from the fixed HPC4 host Python with ``-I -S``.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Final

SACCT_TERMINAL_INSPECTION_SCHEMA: Final = (
    "phase2-recovery-r3-claim-free-sacct-terminal-inspection/v1"
)
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
SACCT_FORMAT_FIELDS: Final = (
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

_MAX_RAW_BYTES: Final = 128 * 1024
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_POSITIVE_JOB_ID_RE = re.compile(r"[1-9][0-9]*\Z")
_NONNEGATIVE_DECIMAL_RE = re.compile(r"(?:0|[1-9][0-9]*)\Z")
_MEMORY_RE = re.compile(r"([1-9][0-9]*)([KMGTP]?)\Z", re.IGNORECASE)


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
        f"--format={','.join(SACCT_FORMAT_FIELDS)}",
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
    if any(token in row.job_id for token in (".", "[", "]", "%", ",", "+")):
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


__all__ = [
    "ClaimFreeSacctTerminalInspection",
    "SACCT_FORMAT_FIELDS",
    "SACCT_TERMINAL_FIELDS",
    "SACCT_TERMINAL_INSPECTION_SCHEMA",
    "SacctAllocationRow",
    "_validate_terminal_row",
    "inspect_sacct_terminal_bytes",
    "sacct_terminal_command",
]
