"""Fail-closed Gate-0 evidence for Phase-2 recovery revision 3.

R2 array ``1648125`` ended in two timeouts and one cancellation.  The older
``phase2_recovery_aggregate`` terminal capture is intentionally success-only
and therefore cannot represent that failure.  This module has a separate
schema and role: it freezes the failed R2 execution as a *failure parent* and
never authorizes reuse of a head, optimizer, step, RNG, PCG state, or result.

There are deliberately two verification levels:

* :func:`inspect_r3_gate0_bundle` validates canonical bytes and all embedded
  evidence, but always returns a non-authorizing inspection.  It is suitable
  for local/offline review.
* :func:`verify_live_r3_gate0_bundle` additionally requires the exact HPC4
  production namespace, live ``hpc4`` Slurm control plane, clean committed
  capture source, unchanged source inodes, container, registries, run
  inventories, logs, and scheduler bytes.  Only this path can create
  :class:`R3Gate0Capability`.

The published artifact is one canonical JSON file.  Raw sacct, the original
immutable live-scontrol receipt, Slurm logs, and FAILED markers are embedded
as base64 so its failure evidence remains self-contained.  Publication uses a
no-overwrite hard link, verifies the linked inode through a descriptor, and
fsyncs both file and parent directory.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import secrets
import stat
import subprocess
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import InitVar, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

GATE0_ARTIFACT_SCHEMA: Final = "phase2-recovery-r3-gate0-r2-failure-parent/v1"
GATE0_ARTIFACT_ROLE: Final = "validated_r2_failure_parent"

SOURCE_ARRAY_JOB_ID: Final = "1648125"
R2_EXECUTION_REVISION: Final = 2
R2_RECOVERY_DESIGN_SHA256: Final = (
    "9602b0f00a73880545fd57ce1886ec65d7901385cce2b919fd72f3efec4592d4"
)
R2_RECOVERY_GIT_COMMIT: Final = "ad7613b7cef3ff536ec62f6f80608ee29e927b1c"
R2_BASE_CONFIG_HASH: Final = "81ccbd3bc9d745d3792d1834116ba2480d34a7201d6f4e463a61d8d8ab0baefa"
R2_PARENT_DESIGN_SHA256: Final = "0c8820b67b8ca85c23cd5c31b8d25001018b31a7271f6a861d88cfae2f85d7ca"
R2_PARENT_REGISTRY_SHA256: Final = (
    "7be4ee90b1f494d32f96214f407a57cbee54be86a77dacc1206d2acd527857dc"
)
R2_INFRASTRUCTURE_REGISTRY_SHA256: Final = (
    "e09eefa403f72044192c58f19e06b1e89b939c1d35ddba5081b13693e995cafd"
)
R2_PARENT_PRODUCER_GIT_COMMIT: Final = "ae28e2a10f0bd5762899be01ce66bc5b423374cf"
R2_IMAGE_SHA256: Final = "d6fc044b4fa303747908783ea057d5b8946f613bfec6a6ca301e3a02fd7719cb"
R2_LIVE_SCONTROL_SHA256: Final = "cb61484f435747d6705ff4567257afff2c447faa16144b697e9f9dcc03f83a5e"
R2_LIVE_SCONTROL_SIZE_BYTES: Final = 4817

ORDERED_TASKS: Final = (
    (0, 20260801, "1648126", "TIMEOUT", "12:00:04"),
    (1, 20260802, "1648203", "TIMEOUT", "12:00:04"),
    (2, 20260803, "1648125", "CANCELLED", "08:58:42"),
)

PRODUCTION_REPO_ROOT: Final = Path("/home/yyangjo/Smart-Reward-Model")
PRODUCTION_PROJECT_ROOT: Final = Path("/project/sigroup/smart-reward-model")
_EXPECTED_PRODUCTION_REPO_ROOT: Final = Path("/home/yyangjo/Smart-Reward-Model")
_EXPECTED_PRODUCTION_PROJECT_ROOT: Final = Path("/project/sigroup/smart-reward-model")
_OUTPUT_DIRECTORY_MODE: Final = 0o750
_R2_EXECUTION_RELATIVE: Final = (
    Path("runs/phase2-recovery-pilot")
    / R2_RECOVERY_DESIGN_SHA256
    / f"execution-{R2_EXECUTION_REVISION}"
)
_LIVE_SCONTROL_RELATIVE: Final = _R2_EXECUTION_RELATIVE / (
    "scheduler-control-live-20260725T153801+0800/scontrol-array-1648125.txt"
)
_GATE0_RELATIVE: Final = Path("runs/phase2-recovery-r3/gate0/r2-1648125-failure-parent.json")
_LOG_ROOT_RELATIVE: Final = Path("slurm-logs/phase2-recovery-pilot") / R2_RECOVERY_DESIGN_SHA256

_R2_PARENT_REGISTRIES: Final = (
    (
        Path("configs/phase2_recovery_parent_failures.json"),
        R2_PARENT_REGISTRY_SHA256,
    ),
    (
        Path("configs/phase2_recovery_infrastructure_failure.json"),
        R2_INFRASTRUCTURE_REGISTRY_SHA256,
    ),
)
_CAPTURE_SOURCE_PATHS: Final = (
    Path("src/smart_reward/phase2_r3_gate0.py"),
    Path("scripts/hpc4/capture_phase2_r3_gate0.py"),
    Path("docs/phase2_recovery_revision3.md"),
)
_R2_SOURCE_PATHS: Final = (
    Path("scripts/hpc4/phase2_recovery_pilot.sbatch"),
    Path("scripts/hpc4/submit_phase2_recovery_pilot.sh"),
    Path("scripts/hpc4/run_phase2_recovery_train.py"),
    Path("scripts/hpc4/validate_phase2_recovery_parent.py"),
    Path("scripts/hpc4/validate_phase2_recovery_infrastructure_failure.py"),
    Path("src/smart_reward/phase2_recovery.py"),
    Path("configs/common_beta_recovery_pilot.yaml"),
)

_SACCT_FORMAT: Final = (
    "JobID,JobIDRaw,State,Elapsed,ExitCode,DerivedExitCode,Cluster,Account,"
    "Partition,NNodes,NCPUS,ReqTRES,AllocTRES"
)
_SACCT_COMMAND: Final = (
    "sacct",
    "-X",
    "-n",
    "-P",
    "-j",
    SOURCE_ARRAY_JOB_ID,
    f"--format={_SACCT_FORMAT}",
)
_SCONTROL_CONFIG_COMMAND: Final = ("scontrol", "show", "config")
_REQUESTED_TRES: Final = "billing=8,cpu=8,gres/gpu=1,mem=96G,node=1"
_ALLOCATED_TRES: Final = "billing=8,cpu=8,gres/gpu:l20=1,gres/gpu=1,mem=96G,node=1"

_LIVE_CONTROL_SCHEMA: Final = "prorm-phase2-recovery-live-scontrol-raw/v1"
_LIVE_CAPTURED_AT: Final = "2026-07-25T15:38:01+08:00"
_LIVE_COMMAND: Final = (
    "scontrol show job -o 1648125_0; scontrol show job -o 1648125_1; scontrol show job -o 1648125_2"
)
_LIVE_STATES: Final = ("RUNNING", "RUNNING", "PENDING")

_FAILED_KEYS: Final = frozenset(
    {
        "schema_version",
        "status",
        "workload_exit_code",
        "final_exit_code",
        "array_job_id",
        "array_task_id",
        "seed",
        "execution_revision",
        "retry_reason",
        "recovery_design_sha256",
        "base_config_hash",
        "recovery_git_commit",
        "parent_design_sha256",
        "parent_registry_sha256",
        "parent_producer_git_commit",
        "one_shot_no_further_adaptation",
        "created_at_utc",
    }
)
_ABSENT_NAMES: Final = ("SUCCESS", "recovery-result.json")
_CHECKPOINT_EXACT_NAMES: Final = frozenset(
    {
        "COMMITTED",
        "checkpoint",
        "latest",
        "optimizer.pt",
        "optimizer.pth",
        "state.pt",
        "trainer-state.pt",
        "trainer_state.json",
    }
)
_CHECKPOINT_SUFFIXES: Final = (".ckpt", ".pth", ".pt")
_TIMESTAMP_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_EXIT_CODE_RE = re.compile(r"[0-9]+:[0-9]+\Z")
_GIT_COMMIT_RE = re.compile(r"[0-9a-f]{40,64}\Z")
_FACTORY_TOKEN = object()


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _require_digest(value: object, *, name: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_exact_keys(
    value: object,
    *,
    name: str,
    keys: set[str] | frozenset[str],
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"{name} has an invalid closed field set")
    return value


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _decode_json(raw: bytes, *, name: str, require_canonical: bool) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token!r}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain one JSON object")
    if require_canonical and raw != _canonical_bytes(value):
        raise ValueError(f"{name} is not canonical JSON")
    return value


def _run_command(
    command: Sequence[str],
    *,
    name: str,
    maximum_bytes: int,
    allow_empty: bool = False,
) -> bytes:
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"could not execute locked {name} command") from error
    stdout = bytes(completed.stdout)
    if (
        completed.returncode != 0
        or completed.stderr
        or (not stdout and not allow_empty)
        or len(stdout) > maximum_bytes
    ):
        raise RuntimeError(f"locked {name} command failed or emitted invalid bytes")
    return stdout


def _parse_sacct_raw(raw: bytes) -> list[dict[str, object]]:
    if not raw or len(raw) > 1024 * 1024 or not raw.endswith(b"\n"):
        raise ValueError("Gate-0 sacct bytes must be bounded and newline-terminated")
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("Gate-0 sacct bytes are not UTF-8") from error
    if len(lines) != 3:
        raise ValueError("Gate-0 sacct must contain exactly three task allocations")

    rows: list[dict[str, object]] = []
    for expected, line in zip(ORDERED_TASKS, lines, strict=True):
        task, seed, raw_job_id, terminal_state, elapsed = expected
        fields = line.split("|")
        if len(fields) != 13:
            raise ValueError(f"Gate-0 sacct task {task} has an invalid parsable layout")
        (
            job_id,
            observed_raw_job_id,
            raw_state,
            observed_elapsed,
            exit_code,
            derived_exit_code,
            cluster,
            account,
            partition,
            n_nodes,
            n_cpus,
            requested_tres,
            allocated_tres,
        ) = fields
        if terminal_state == "CANCELLED":
            state_matches = re.fullmatch(r"CANCELLED(?: by [0-9]+)?", raw_state) is not None
        else:
            state_matches = raw_state == terminal_state
        if (
            job_id != f"{SOURCE_ARRAY_JOB_ID}_{task}"
            or observed_raw_job_id != raw_job_id
            or not state_matches
            or observed_elapsed != elapsed
            or _EXIT_CODE_RE.fullmatch(exit_code) is None
            or _EXIT_CODE_RE.fullmatch(derived_exit_code) is None
            or cluster != "hpc4"
            or account != "sigroup"
            or partition != "gpu-l20"
            or n_nodes != "1"
            or n_cpus != "8"
            or requested_tres != _REQUESTED_TRES
            or allocated_tres != _ALLOCATED_TRES
        ):
            raise ValueError(f"Gate-0 sacct task {task} differs from frozen R2 failure")
        rows.append(
            {
                "job_id": job_id,
                "job_id_raw": observed_raw_job_id,
                "array_job_id": SOURCE_ARRAY_JOB_ID,
                "array_task_id": task,
                "seed": seed,
                "raw_state": raw_state,
                "terminal_state": terminal_state,
                "elapsed": observed_elapsed,
                "exit_code": exit_code,
                "derived_exit_code": derived_exit_code,
                "cluster": cluster,
                "account": account,
                "partition": partition,
                "n_nodes": 1,
                "n_cpus": 8,
                "requested_tres": requested_tres,
                "allocated_tres": allocated_tres,
            }
        )
    return rows


def _parse_live_scontrol(raw: bytes, *, require_frozen_bytes: bool) -> list[dict[str, object]]:
    if not raw or len(raw) > 64 * 1024 or not raw.endswith(b"\n"):
        raise ValueError("live-scontrol receipt must be bounded and newline-terminated")
    if require_frozen_bytes and (
        len(raw) != R2_LIVE_SCONTROL_SIZE_BYTES or _sha256(raw) != R2_LIVE_SCONTROL_SHA256
    ):
        raise ValueError("original R2 live-scontrol receipt bytes changed")
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("live-scontrol receipt is not UTF-8") from error
    if len(lines) != 6 or lines[:3] != [
        f"schema={_LIVE_CONTROL_SCHEMA}",
        f"captured_at={_LIVE_CAPTURED_AT}",
        f"command={_LIVE_COMMAND}",
    ]:
        raise ValueError("live-scontrol receipt header is not the frozen capture")

    rows: list[dict[str, object]] = []
    for expected, expected_state, line in zip(
        ORDERED_TASKS,
        _LIVE_STATES,
        lines[3:],
        strict=True,
    ):
        task, seed, raw_job_id, _, _ = expected
        fields: dict[str, str] = {}
        for token in line.split():
            if "=" not in token:
                raise ValueError(f"live-scontrol task {task} has a malformed token")
            key, value = token.split("=", 1)
            if not key or key in fields:
                raise ValueError(f"live-scontrol task {task} has a duplicate field")
            fields[key] = value
        if (
            fields.get("JobId") != raw_job_id
            or fields.get("ArrayJobId") != SOURCE_ARRAY_JOB_ID
            or fields.get("ArrayTaskId") != str(task)
            or fields.get("JobState") != expected_state
            or fields.get("Account") != "sigroup"
            or fields.get("QOS") != "l20_qos"
            or fields.get("Partition") != "gpu-l20"
            or fields.get("Requeue") != "0"
            or fields.get("Restarts") != "0"
            or fields.get("TimeLimit") != "12:00:00"
        ):
            raise ValueError(f"live-scontrol task {task} differs from frozen R2 submission")
        rows.append(
            {
                "array_job_id": SOURCE_ARRAY_JOB_ID,
                "array_task_id": task,
                "seed": seed,
                "job_id_raw": raw_job_id,
                "state_at_capture": expected_state,
                "requeue": 0,
                "restarts_at_capture": 0,
                "account": "sigroup",
                "partition": "gpu-l20",
                "qos": "l20_qos",
                "time_limit": "12:00:00",
                "terminal_status_authority": False,
            }
        )
    return rows


def _parse_scontrol_config(raw: bytes) -> None:
    if not raw or len(raw) > 4 * 1024 * 1024:
        raise ValueError("scontrol configuration bytes are empty or oversized")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("scontrol configuration is not UTF-8") from error
    matches = re.findall(r"(?m)^\s*ClusterName\s*=\s*(\S+)\s*$", text)
    if matches != ["hpc4"]:
        raise ValueError("live Slurm control plane is not exactly cluster hpc4")


def _parse_failed_marker(
    raw: bytes,
    *,
    task: int,
    seed: int,
) -> dict[str, object]:
    if not raw or len(raw) > 64 * 1024 or not raw.endswith(b"\n"):
        raise ValueError(f"FAILED marker for task {task} is empty, oversized, or unterminated")
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError(f"FAILED marker for task {task} is not UTF-8") from error
    fields: dict[str, str] = {}
    for line in lines:
        if not line or "=" not in line:
            raise ValueError(f"FAILED marker for task {task} has a malformed line")
        key, value = line.split("=", 1)
        if not key or key in fields:
            raise ValueError(f"FAILED marker for task {task} has a duplicate field")
        fields[key] = value
    if set(fields) != _FAILED_KEYS:
        raise ValueError(f"FAILED marker for task {task} has an invalid field set")
    expected = {
        "schema_version": "prorm-phase2-recovery-run-status/v1",
        "status": "FAILED",
        "array_job_id": SOURCE_ARRAY_JOB_ID,
        "array_task_id": str(task),
        "seed": str(seed),
        "execution_revision": str(R2_EXECUTION_REVISION),
        "retry_reason": "pretrainer_hf_datasets_runtime_lock",
        "recovery_design_sha256": R2_RECOVERY_DESIGN_SHA256,
        "base_config_hash": R2_BASE_CONFIG_HASH,
        "recovery_git_commit": R2_RECOVERY_GIT_COMMIT,
        "parent_design_sha256": R2_PARENT_DESIGN_SHA256,
        "parent_registry_sha256": R2_PARENT_REGISTRY_SHA256,
        "parent_producer_git_commit": R2_PARENT_PRODUCER_GIT_COMMIT,
        "one_shot_no_further_adaptation": "true",
    }
    if any(fields.get(key) != value for key, value in expected.items()):
        raise ValueError(f"FAILED marker for task {task} differs from R2 execution identity")
    try:
        workload_exit = int(fields["workload_exit_code"])
        final_exit = int(fields["final_exit_code"])
    except ValueError as error:
        raise ValueError(f"FAILED marker for task {task} has non-integer exit fields") from error
    if workload_exit < 0 or final_exit < 0 or (workload_exit == 0 and final_exit == 0):
        raise ValueError(f"FAILED marker for task {task} does not encode a failing exit")
    if _TIMESTAMP_RE.fullmatch(fields["created_at_utc"]) is None:
        raise ValueError(f"FAILED marker for task {task} has an invalid UTC timestamp")
    return {
        "schema_version": fields["schema_version"],
        "status": "FAILED",
        "workload_exit_code": workload_exit,
        "final_exit_code": final_exit,
        "array_job_id": SOURCE_ARRAY_JOB_ID,
        "array_task_id": task,
        "seed": seed,
        "execution_revision": R2_EXECUTION_REVISION,
        "recovery_design_sha256": R2_RECOVERY_DESIGN_SHA256,
        "recovery_git_commit": R2_RECOVERY_GIT_COMMIT,
        "created_at_utc": fields["created_at_utc"],
    }


def _require_real_directory(path: Path, *, name: str) -> os.stat_result:
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or path.is_symlink() or path.resolve(strict=True) != path:
        raise ValueError(f"{name} must be a canonical real directory")
    return info


def _stable_file(path: Path, *, name: str, maximum_bytes: int) -> tuple[bytes, dict[str, object]]:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or path.is_symlink():
        raise ValueError(f"{name} must be a regular non-symlink file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > maximum_bytes:
                raise ValueError(f"{name} exceeds the evidence size bound")
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    named_after = path.lstat()
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_opened = (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    identity_named_after = (
        named_after.st_dev,
        named_after.st_ino,
        named_after.st_size,
        named_after.st_mtime_ns,
    )
    if (
        identity_before != identity_opened
        or identity_opened != identity_after
        or identity_after != identity_named_after
        or not stat.S_ISREG(named_after.st_mode)
    ):
        raise ValueError(f"{name} changed while evidence was read")
    raw = b"".join(chunks)
    if len(raw) != opened.st_size:
        raise ValueError(f"{name} byte count differs from its inode")
    record = {
        "kind": "regular_file",
        "mode_octal": f"{stat.S_IMODE(opened.st_mode):04o}",
        "size_bytes": len(raw),
        "sha256": _sha256(raw),
        "mtime_ns": opened.st_mtime_ns,
        "device": opened.st_dev,
        "inode": opened.st_ino,
    }
    return raw, record


def _stable_file_digest(
    path: Path,
    *,
    name: str,
    maximum_bytes: int,
) -> tuple[str, int, dict[str, object]]:
    """Hash a stable regular file without retaining its bytes in memory."""

    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or path.is_symlink():
        raise ValueError(f"{name} must be a regular non-symlink file")
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0),
    )
    digest = hashlib.sha256()
    size = 0
    try:
        opened = os.fstat(descriptor)
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > maximum_bytes:
                raise ValueError(f"{name} exceeds the evidence size bound")
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    named_after = path.lstat()
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_opened = (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    identity_named_after = (
        named_after.st_dev,
        named_after.st_ino,
        named_after.st_size,
        named_after.st_mtime_ns,
    )
    if (
        identity_before != identity_opened
        or identity_opened != identity_after
        or identity_after != identity_named_after
        or size != opened.st_size
        or not stat.S_ISREG(named_after.st_mode)
    ):
        raise ValueError(f"{name} changed while evidence was hashed")
    record = {
        "kind": "regular_file",
        "mode_octal": f"{stat.S_IMODE(opened.st_mode):04o}",
        "size_bytes": size,
        "sha256": digest.hexdigest(),
        "mtime_ns": opened.st_mtime_ns,
        "device": opened.st_dev,
        "inode": opened.st_ino,
    }
    return digest.hexdigest(), size, record


def _checkpoint_like(name: str) -> bool:
    lowered = name.lower()
    return (
        "checkpoint" in lowered
        or name in _CHECKPOINT_EXACT_NAMES
        or lowered.endswith(_CHECKPOINT_SUFFIXES)
    )


def _inventory_run(path: Path, *, task: int, seed: int) -> dict[str, object]:
    directory_before = _require_real_directory(path, name=f"R2 task {task} run directory")
    initial_names = sorted(entry.name for entry in path.iterdir())
    entries: list[dict[str, object]] = []
    raw_files: dict[str, bytes] = {}
    for entry in (path / name for name in initial_names):
        info = entry.lstat()
        if stat.S_ISREG(info.st_mode) and not entry.is_symlink():
            if entry.name == "FAILED":
                raw, record = _stable_file(
                    entry,
                    name=f"R2 task {task} inventory entry {entry.name}",
                    maximum_bytes=64 * 1024,
                )
                raw_files[entry.name] = raw
            else:
                _, _, record = _stable_file_digest(
                    entry,
                    name=f"R2 task {task} inventory entry {entry.name}",
                    maximum_bytes=8 * 1024 * 1024 * 1024,
                )
            entries.append({"name": entry.name, **record})
        elif stat.S_ISLNK(info.st_mode) and entry.name == "parent-artifact":
            target = os.readlink(entry)
            after = entry.lstat()
            if (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            ) or not stat.S_ISLNK(after.st_mode):
                raise ValueError("R2 parent-artifact changed while it was inventoried")
            if os.path.isabs(target):
                raise ValueError("R2 parent-artifact reference must remain relative")
            target_raw = os.fsencode(target)
            entries.append(
                {
                    "name": entry.name,
                    "kind": "symlink_reference",
                    "mode_octal": f"{stat.S_IMODE(info.st_mode):04o}",
                    "size_bytes": len(target_raw),
                    "sha256": _sha256(target_raw),
                    "mtime_ns": info.st_mtime_ns,
                    "device": info.st_dev,
                    "inode": info.st_ino,
                }
            )
        else:
            raise ValueError(
                f"R2 task {task} inventory contains unsupported nested/special entry {entry.name!r}"
            )
    names = {str(item["name"]) for item in entries}
    directory_after = _require_real_directory(path, name=f"R2 task {task} run directory")
    final_names = sorted(entry.name for entry in path.iterdir())
    if initial_names != final_names or (
        directory_before.st_dev,
        directory_before.st_ino,
        directory_before.st_mtime_ns,
    ) != (directory_after.st_dev, directory_after.st_ino, directory_after.st_mtime_ns):
        raise ValueError(f"R2 task {task} run directory changed during inventory")
    if len(names) != len(entries):
        raise ValueError(f"R2 task {task} inventory contains duplicate names")
    if "FAILED" not in raw_files:
        raise ValueError(f"R2 task {task} lacks its regular FAILED marker")
    forbidden_present = sorted(name for name in _ABSENT_NAMES if name in names)
    checkpoints = sorted(name for name in names if _checkpoint_like(name))
    if forbidden_present or checkpoints:
        raise ValueError(
            f"R2 task {task} contains success/result/checkpoint evidence: "
            f"{forbidden_present + checkpoints!r}"
        )
    failed_projection = _parse_failed_marker(raw_files["FAILED"], task=task, seed=seed)
    return {
        "entries": entries,
        "failed_bytes": raw_files["FAILED"],
        "failed_projection": failed_projection,
        "absence_audit": {
            "SUCCESS_present": False,
            "recovery_result_present": False,
            "durable_checkpoint_present": False,
            "checkpoint_candidates": [],
        },
    }


def _source_metadata(relative: str | None, raw: bytes, record: Mapping[str, object] | None) -> dict:
    base: dict[str, object] = {
        "source_relative": relative,
        "size_bytes": len(raw),
        "sha256": _sha256(raw),
        "bytes_base64": base64.b64encode(raw).decode("ascii"),
    }
    if record is None:
        base["source_inode"] = None
    else:
        base["source_inode"] = {
            key: record[key] for key in ("mode_octal", "mtime_ns", "device", "inode")
        }
    return base


def _raw_source(
    *,
    name: str,
    evidence_kind: str,
    source_relative: str | None,
    raw: bytes,
    record: Mapping[str, object] | None,
) -> dict[str, object]:
    return {
        "name": name,
        "evidence_kind": evidence_kind,
        **_source_metadata(source_relative, raw, record),
    }


def _repository_root() -> Path:
    root = Path(__file__).resolve().parents[2]
    _require_real_directory(root, name="Gate-0 repository root")
    return root


def _assert_production_roots() -> tuple[Path, Path]:
    """Validate the fixed, disjoint HPC4 code and persistence namespaces."""

    repo = PRODUCTION_REPO_ROOT
    project = PRODUCTION_PROJECT_ROOT
    if repo != _EXPECTED_PRODUCTION_REPO_ROOT:
        raise RuntimeError("Gate-0 production repository root is not the fixed HPC4 path")
    if project != _EXPECTED_PRODUCTION_PROJECT_ROOT:
        raise RuntimeError("Gate-0 production project root is not the fixed HPC4 path")
    if not repo.is_absolute() or not project.is_absolute():
        raise RuntimeError("Gate-0 production roots must be absolute")
    _require_real_directory(repo, name="production repository root")
    _require_real_directory(project, name="production project root")
    if repo.resolve(strict=True) != repo or project.resolve(strict=True) != project:
        raise ValueError("Gate-0 production roots must be canonical")
    if repo == project or repo in project.parents or project in repo.parents:
        raise ValueError("Gate-0 production repository and project roots must be disjoint")
    git_directory = repo / ".git"
    _require_real_directory(git_directory, name="production repository .git directory")
    project_git = project / ".git"
    if project_git.exists() or project_git.is_symlink():
        raise ValueError("production project root must not be a Git checkout")
    if _repository_root() != repo:
        raise RuntimeError("Gate-0 verifier was not imported from fixed production repository")
    return repo, project


def _git(command: Sequence[str], *, name: str, allow_empty: bool = False) -> bytes:
    return _run_command(
        ("git", "-C", os.fspath(_repository_root()), *command),
        name=name,
        maximum_bytes=64 * 1024 * 1024,
        allow_empty=allow_empty,
    )


def _git_blob_record(commit: str, relative: Path) -> dict[str, object]:
    raw = _git(("cat-file", "blob", f"{commit}:{relative.as_posix()}"), name="git cat-file")
    object_id = (
        _git(
            ("rev-parse", "--verify", f"{commit}:{relative.as_posix()}"),
            name="git blob identity",
        )
        .decode("ascii")
        .strip()
    )
    if _GIT_COMMIT_RE.fullmatch(object_id) is None:
        raise ValueError(f"Git object id is invalid for {relative.as_posix()}")
    return {
        "repository_relative": relative.as_posix(),
        "git_commit": commit,
        "git_object_id": object_id,
        "size_bytes": len(raw),
        "sha256": _sha256(raw),
    }


def _capture_source_identity() -> dict[str, object]:
    root = _repository_root()
    commit = _git(("rev-parse", "--verify", "HEAD"), name="git HEAD").decode("ascii").strip()
    if _GIT_COMMIT_RE.fullmatch(commit) is None:
        raise ValueError("Gate-0 capture commit is invalid")
    status_raw = _git(
        ("status", "--porcelain", "--untracked-files=normal"),
        name="git status",
        allow_empty=True,
    )
    if status_raw:
        raise ValueError("Gate-0 capture requires a clean committed checkout")
    records: list[dict[str, object]] = []
    for relative in _CAPTURE_SOURCE_PATHS:
        path = root / relative
        raw, _ = _stable_file(
            path,
            name=f"capture source {relative}",
            maximum_bytes=8 * 1024 * 1024,
        )
        record = _git_blob_record(commit, relative)
        if record["sha256"] != _sha256(raw) or record["size_bytes"] != len(raw):
            raise ValueError(f"capture source differs from committed blob: {relative}")
        records.append(record)
    r2_records = [_git_blob_record(R2_RECOVERY_GIT_COMMIT, path) for path in _R2_SOURCE_PATHS]
    return {
        "execution_git_commit": commit,
        "clean_worktree": True,
        "capture_source_blobs": records,
        "r2_execution_source_blobs": r2_records,
    }


def _parent_registry_records() -> list[dict[str, object]]:
    root = _repository_root()
    records: list[dict[str, object]] = []
    for relative, expected_sha256 in _R2_PARENT_REGISTRIES:
        raw, _ = _stable_file(
            root / relative,
            name=f"R2 parent registry {relative}",
            maximum_bytes=8 * 1024 * 1024,
        )
        record = _git_blob_record(R2_RECOVERY_GIT_COMMIT, relative)
        if (
            _sha256(raw) != expected_sha256
            or record["sha256"] != expected_sha256
            or record["size_bytes"] != len(raw)
        ):
            raise ValueError(f"R2 parent registry bytes changed: {relative}")
        records.append(record)
    return records


def _production_path(relative: Path, *, must_exist: bool, name: str) -> Path:
    root = PRODUCTION_PROJECT_ROOT
    if not root.is_absolute():
        raise ValueError("production project root is not absolute")
    expected = root / relative
    if expected.parts[: len(root.parts)] != root.parts:
        raise ValueError(f"{name} escaped the production root")
    if must_exist and expected.resolve(strict=True) != expected:
        raise ValueError(f"{name} is not canonical")
    return expected


def _production_repo_path(relative: Path, *, must_exist: bool, name: str) -> Path:
    root = PRODUCTION_REPO_ROOT
    if not root.is_absolute():
        raise ValueError("production repository root is not absolute")
    expected = root / relative
    if expected.parts[: len(root.parts)] != root.parts:
        raise ValueError(f"{name} escaped the production repository root")
    if must_exist and expected.resolve(strict=True) != expected:
        raise ValueError(f"{name} is not canonical")
    return expected


def _relative_to_production(path: Path) -> str:
    return path.relative_to(PRODUCTION_PROJECT_ROOT).as_posix()


def _run_path(task: int, seed: int) -> Path:
    return _production_path(
        _R2_EXECUTION_RELATIVE / f"seed-{seed}" / f"job-{SOURCE_ARRAY_JOB_ID}_{task}",
        must_exist=True,
        name=f"R2 task {task} run directory",
    )


def _log_path(task: int, suffix: str) -> Path:
    return _production_path(
        _LOG_ROOT_RELATIVE / f"prorm-p2-recovery-{SOURCE_ARRAY_JOB_ID}_{task}.{suffix}",
        must_exist=True,
        name=f"R2 task {task} Slurm {suffix}",
    )


def _container_record(container: Path) -> dict[str, object]:
    absolute = container.absolute()
    if absolute.resolve(strict=True) != absolute:
        raise ValueError("R2 container path must be canonical")
    try:
        relative = absolute.relative_to(PRODUCTION_PROJECT_ROOT)
    except ValueError as error:
        raise ValueError("R2 container must be retained inside the production root") from error
    digest, size, inode = _stable_file_digest(
        absolute,
        name="R2 container image",
        maximum_bytes=128 * 1024 * 1024 * 1024,
    )
    if digest != R2_IMAGE_SHA256:
        raise ValueError("R2 container image bytes differ from the frozen execution")
    return {
        "source_relative": relative.as_posix(),
        "sha256": R2_IMAGE_SHA256,
        "size_bytes": size,
        "source_inode": {key: inode[key] for key in ("mode_octal", "mtime_ns", "device", "inode")},
    }


def _timestamp(now: datetime | None) -> str:
    value = datetime.now(timezone.utc) if now is None else now
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Gate-0 capture timestamp must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _assemble_live_payload(
    *,
    container: Path,
    sacct_raw: bytes,
    scontrol_config_raw: bytes,
    now: datetime | None,
) -> dict[str, object]:
    scheduler_rows = _parse_sacct_raw(sacct_raw)
    _parse_scontrol_config(scontrol_config_raw)
    live_control_path = _production_path(
        _LIVE_SCONTROL_RELATIVE,
        must_exist=True,
        name="original R2 live-scontrol receipt",
    )
    live_control_raw, live_control_inode = _stable_file(
        live_control_path,
        name="original R2 live-scontrol receipt",
        maximum_bytes=64 * 1024,
    )
    if os.name == "posix" and live_control_inode["mode_octal"] != "0440":
        raise ValueError("original R2 live-scontrol receipt must retain mode 0440")
    live_rows = _parse_live_scontrol(live_control_raw, require_frozen_bytes=True)

    raw_sources: list[dict[str, object]] = [
        _raw_source(
            name="scheduler-sacct-X.psv",
            evidence_kind="scheduler_terminal_sacct_raw",
            source_relative=None,
            raw=sacct_raw,
            record=None,
        ),
        _raw_source(
            name="original-live-scontrol.txt",
            evidence_kind="scheduler_submission_live_scontrol_raw",
            source_relative=_relative_to_production(live_control_path),
            raw=live_control_raw,
            record=live_control_inode,
        ),
        _raw_source(
            name="capture-scontrol-config.txt",
            evidence_kind="capture_environment_scontrol_config_raw",
            source_relative=None,
            raw=scontrol_config_raw,
            record=None,
        ),
    ]
    tasks: list[dict[str, object]] = []
    for task, seed, _, terminal_state, elapsed in ORDERED_TASKS:
        run_path = _run_path(task, seed)
        inventory = _inventory_run(run_path, task=task, seed=seed)
        failed_name = f"task-{task}-FAILED"
        raw_sources.append(
            _raw_source(
                name=failed_name,
                evidence_kind="r2_failed_marker_raw",
                source_relative=(f"{_relative_to_production(run_path)}/FAILED"),
                raw=inventory["failed_bytes"],
                record=next(item for item in inventory["entries"] if item["name"] == "FAILED"),
            )
        )
        logs: list[dict[str, object]] = []
        for suffix in ("out", "err"):
            path = _log_path(task, suffix)
            raw, inode = _stable_file(
                path,
                name=f"R2 task {task} Slurm {suffix}",
                maximum_bytes=512 * 1024 * 1024,
            )
            source_name = f"task-{task}-slurm.{suffix}"
            raw_sources.append(
                _raw_source(
                    name=source_name,
                    evidence_kind=f"slurm_{suffix}_raw",
                    source_relative=_relative_to_production(path),
                    raw=raw,
                    record=inode,
                )
            )
            logs.append(
                {
                    "stream": suffix,
                    "raw_source_name": source_name,
                    "sha256": _sha256(raw),
                    "size_bytes": len(raw),
                }
            )
        tasks.append(
            {
                "array_task_id": task,
                "seed": seed,
                "terminal_state": terminal_state,
                "elapsed": elapsed,
                "run_relative": _relative_to_production(run_path),
                "inventory": inventory["entries"],
                "failed_raw_source_name": failed_name,
                "failed_marker": inventory["failed_projection"],
                "absence_audit": inventory["absence_audit"],
                "slurm_logs": logs,
            }
        )
    raw_sources.sort(key=lambda item: str(item["name"]))

    unsigned: dict[str, object] = {
        "schema_version": GATE0_ARTIFACT_SCHEMA,
        "role": GATE0_ARTIFACT_ROLE,
        "captured_at_utc": _timestamp(now),
        "source_array_job_id": SOURCE_ARRAY_JOB_ID,
        "r2_execution_identity": {
            "execution_revision": R2_EXECUTION_REVISION,
            "recovery_design_sha256": R2_RECOVERY_DESIGN_SHA256,
            "recovery_git_commit": R2_RECOVERY_GIT_COMMIT,
            "base_config_hash": R2_BASE_CONFIG_HASH,
            "ordered_task_seed_map": [[task, seed] for task, seed, *_ in ORDERED_TASKS],
        },
        "scheduler_terminal": {
            "source_command": list(_SACCT_COMMAND),
            "raw_source_name": "scheduler-sacct-X.psv",
            "rows": scheduler_rows,
            "all_tasks_terminal": True,
            "all_tasks_non_success": True,
            "task_2_cancelled": True,
        },
        "original_live_scontrol": {
            "raw_source_name": "original-live-scontrol.txt",
            "sha256": R2_LIVE_SCONTROL_SHA256,
            "size_bytes": R2_LIVE_SCONTROL_SIZE_BYTES,
            "rows": live_rows,
            "terminal_status_authority": False,
        },
        "capture_environment": {
            "cluster": "hpc4",
            "scontrol_config_command": list(_SCONTROL_CONFIG_COMMAND),
            "raw_source_name": "capture-scontrol-config.txt",
        },
        "tasks": tasks,
        "raw_sources": raw_sources,
        "producer": _capture_source_identity(),
        "container": _container_record(container),
        "parent_registries": _parent_registry_records(),
        "failure_parent_policy": {
            "scope": "r3_failure_parent_only",
            "profile_authorized_by_this_artifact": False,
            "training_state_reusable": False,
            "head_reusable": False,
            "optimizer_reusable": False,
            "step_reusable": False,
            "rng_reusable": False,
            "pcg_state_reusable": False,
            "beta_reusable": False,
            "scientific_effect_claim_supported": False,
        },
    }
    payload = dict(unsigned)
    payload["artifact_sha256"] = _sha256(_canonical_bytes(unsigned))
    return payload


def _embedded_raw_sources(payload: Mapping[str, object]) -> dict[str, bytes]:
    values = payload.get("raw_sources")
    if not isinstance(values, list) or len(values) != 12:
        raise ValueError("Gate-0 artifact must embed exactly twelve raw evidence sources")
    expected_names = {
        "scheduler-sacct-X.psv",
        "original-live-scontrol.txt",
        "capture-scontrol-config.txt",
        *(f"task-{task}-FAILED" for task, *_ in ORDERED_TASKS),
        *(f"task-{task}-slurm.{suffix}" for task, *_ in ORDERED_TASKS for suffix in ("out", "err")),
    }
    result: dict[str, bytes] = {}
    for index, item in enumerate(values):
        record = _require_exact_keys(
            item,
            name=f"raw_sources[{index}]",
            keys={
                "name",
                "evidence_kind",
                "source_relative",
                "size_bytes",
                "sha256",
                "bytes_base64",
                "source_inode",
            },
        )
        name = record["name"]
        if type(name) is not str or name in result:
            raise ValueError("Gate-0 raw source names must be unique strings")
        try:
            raw = base64.b64decode(record["bytes_base64"], validate=True)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Gate-0 raw source {name!r} is not strict base64") from error
        if (
            base64.b64encode(raw).decode("ascii") != record["bytes_base64"]
            or record["size_bytes"] != len(raw)
            or record["sha256"] != _sha256(raw)
        ):
            raise ValueError(f"Gate-0 raw source {name!r} has an invalid byte binding")
        _require_digest(record["sha256"], name=f"raw source {name} SHA256")
        if record["source_relative"] is not None and (
            type(record["source_relative"]) is not str
            or not record["source_relative"]
            or Path(record["source_relative"]).is_absolute()
            or ".." in Path(record["source_relative"]).parts
        ):
            raise ValueError(f"Gate-0 raw source {name!r} has an unsafe source path")
        inode = record["source_inode"]
        if inode is not None:
            inode = _require_exact_keys(
                inode,
                name=f"raw source {name} inode",
                keys={"mode_octal", "mtime_ns", "device", "inode"},
            )
            if (
                type(inode["mode_octal"]) is not str
                or re.fullmatch(r"[0-7]{4}", inode["mode_octal"]) is None
                or any(
                    type(inode[key]) is not int or inode[key] < 0
                    for key in ("mtime_ns", "device", "inode")
                )
            ):
                raise ValueError(f"Gate-0 raw source {name!r} inode binding is invalid")
        result[name] = raw
    if set(result) != expected_names:
        raise ValueError("Gate-0 artifact has missing or unexpected raw evidence names")
    if [str(item["name"]) for item in values] != sorted(expected_names):
        raise ValueError("Gate-0 raw evidence sources are not canonically ordered")
    return result


def _validate_inventory(
    value: object,
    *,
    task: int,
    failed_raw: bytes,
) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError(f"Gate-0 task {task} inventory is empty or invalid")
    names: set[str] = set()
    failed_matches = 0
    for index, item in enumerate(value):
        record = _require_exact_keys(
            item,
            name=f"task {task} inventory[{index}]",
            keys={
                "name",
                "kind",
                "mode_octal",
                "size_bytes",
                "sha256",
                "mtime_ns",
                "device",
                "inode",
            },
        )
        name = record["name"]
        if type(name) is not str or not name or "/" in name or name in names:
            raise ValueError(f"Gate-0 task {task} inventory name is invalid")
        names.add(name)
        if record["kind"] not in {"regular_file", "symlink_reference"}:
            raise ValueError(f"Gate-0 task {task} inventory kind is invalid")
        _require_digest(record["sha256"], name=f"task {task} inventory SHA256")
        if (
            type(record["mode_octal"]) is not str
            or re.fullmatch(r"[0-7]{4}", record["mode_octal"]) is None
            or any(
                type(record[key]) is not int or record[key] < 0
                for key in ("size_bytes", "mtime_ns", "device", "inode")
            )
        ):
            raise ValueError(f"Gate-0 task {task} inventory metadata is invalid")
        if name == "FAILED":
            failed_matches += 1
            if (
                record["kind"] != "regular_file"
                or record["size_bytes"] != len(failed_raw)
                or record["sha256"] != _sha256(failed_raw)
            ):
                raise ValueError(f"Gate-0 task {task} FAILED inventory binding is invalid")
    if [str(item["name"]) for item in value] != sorted(names):
        raise ValueError(f"Gate-0 task {task} inventory is not ordered")
    if failed_matches != 1:
        raise ValueError(f"Gate-0 task {task} inventory must contain exactly one FAILED")
    if any(name in names for name in _ABSENT_NAMES) or any(
        _checkpoint_like(name) for name in names
    ):
        raise ValueError(f"Gate-0 task {task} inventory contradicts explicit absence")


def _validate_git_records(value: object, *, name: str, expected_count: int) -> None:
    if not isinstance(value, list) or len(value) != expected_count:
        raise ValueError(f"{name} has an invalid record count")
    for index, item in enumerate(value):
        record = _require_exact_keys(
            item,
            name=f"{name}[{index}]",
            keys={
                "repository_relative",
                "git_commit",
                "git_object_id",
                "size_bytes",
                "sha256",
            },
        )
        if (
            type(record["repository_relative"]) is not str
            or Path(record["repository_relative"]).is_absolute()
            or ".." in Path(record["repository_relative"]).parts
            or _GIT_COMMIT_RE.fullmatch(str(record["git_commit"])) is None
            or _GIT_COMMIT_RE.fullmatch(str(record["git_object_id"])) is None
            or type(record["size_bytes"]) is not int
            or record["size_bytes"] < 0
        ):
            raise ValueError(f"{name}[{index}] is invalid")
        _require_digest(record["sha256"], name=f"{name}[{index}].sha256")


def _validate_payload(payload: dict[str, Any]) -> tuple[str, tuple[dict[str, object], ...]]:
    _require_exact_keys(
        payload,
        name="Gate-0 artifact",
        keys={
            "schema_version",
            "role",
            "captured_at_utc",
            "source_array_job_id",
            "r2_execution_identity",
            "scheduler_terminal",
            "original_live_scontrol",
            "capture_environment",
            "tasks",
            "raw_sources",
            "producer",
            "container",
            "parent_registries",
            "failure_parent_policy",
            "artifact_sha256",
        },
    )
    if (
        payload["schema_version"] != GATE0_ARTIFACT_SCHEMA
        or payload["role"] != GATE0_ARTIFACT_ROLE
        or payload["source_array_job_id"] != SOURCE_ARRAY_JOB_ID
        or _TIMESTAMP_RE.fullmatch(str(payload["captured_at_utc"])) is None
    ):
        raise ValueError("Gate-0 artifact identity is invalid")
    claimed_sha = _require_digest(payload["artifact_sha256"], name="artifact_sha256")
    unsigned = dict(payload)
    del unsigned["artifact_sha256"]
    if claimed_sha != _sha256(_canonical_bytes(unsigned)):
        raise ValueError("Gate-0 artifact self-hash does not match its closed payload")

    identity = _require_exact_keys(
        payload["r2_execution_identity"],
        name="r2_execution_identity",
        keys={
            "execution_revision",
            "recovery_design_sha256",
            "recovery_git_commit",
            "base_config_hash",
            "ordered_task_seed_map",
        },
    )
    if identity != {
        "execution_revision": R2_EXECUTION_REVISION,
        "recovery_design_sha256": R2_RECOVERY_DESIGN_SHA256,
        "recovery_git_commit": R2_RECOVERY_GIT_COMMIT,
        "base_config_hash": R2_BASE_CONFIG_HASH,
        "ordered_task_seed_map": [[task, seed] for task, seed, *_ in ORDERED_TASKS],
    }:
        raise ValueError("Gate-0 artifact names the wrong R2 execution")

    raw = _embedded_raw_sources(payload)
    scheduler_rows = _parse_sacct_raw(raw["scheduler-sacct-X.psv"])
    live_rows = _parse_live_scontrol(
        raw["original-live-scontrol.txt"],
        require_frozen_bytes=True,
    )
    _parse_scontrol_config(raw["capture-scontrol-config.txt"])

    scheduler = _require_exact_keys(
        payload["scheduler_terminal"],
        name="scheduler_terminal",
        keys={
            "source_command",
            "raw_source_name",
            "rows",
            "all_tasks_terminal",
            "all_tasks_non_success",
            "task_2_cancelled",
        },
    )
    if scheduler != {
        "source_command": list(_SACCT_COMMAND),
        "raw_source_name": "scheduler-sacct-X.psv",
        "rows": scheduler_rows,
        "all_tasks_terminal": True,
        "all_tasks_non_success": True,
        "task_2_cancelled": True,
    }:
        raise ValueError("Gate-0 scheduler projection differs from raw sacct bytes")
    original = _require_exact_keys(
        payload["original_live_scontrol"],
        name="original_live_scontrol",
        keys={
            "raw_source_name",
            "sha256",
            "size_bytes",
            "rows",
            "terminal_status_authority",
        },
    )
    if original != {
        "raw_source_name": "original-live-scontrol.txt",
        "sha256": R2_LIVE_SCONTROL_SHA256,
        "size_bytes": R2_LIVE_SCONTROL_SIZE_BYTES,
        "rows": live_rows,
        "terminal_status_authority": False,
    }:
        raise ValueError("Gate-0 live-scontrol projection is invalid")
    if payload["capture_environment"] != {
        "cluster": "hpc4",
        "scontrol_config_command": list(_SCONTROL_CONFIG_COMMAND),
        "raw_source_name": "capture-scontrol-config.txt",
    }:
        raise ValueError("Gate-0 capture environment binding is invalid")

    tasks = payload["tasks"]
    if not isinstance(tasks, list) or len(tasks) != 3:
        raise ValueError("Gate-0 artifact must contain exactly three ordered tasks")
    for expected, task_payload in zip(ORDERED_TASKS, tasks, strict=True):
        task, seed, _, state, elapsed = expected
        task_value = _require_exact_keys(
            task_payload,
            name=f"tasks[{task}]",
            keys={
                "array_task_id",
                "seed",
                "terminal_state",
                "elapsed",
                "run_relative",
                "inventory",
                "failed_raw_source_name",
                "failed_marker",
                "absence_audit",
                "slurm_logs",
            },
        )
        expected_run = (
            _R2_EXECUTION_RELATIVE / f"seed-{seed}" / f"job-{SOURCE_ARRAY_JOB_ID}_{task}"
        ).as_posix()
        failed_name = f"task-{task}-FAILED"
        failed_projection = _parse_failed_marker(raw[failed_name], task=task, seed=seed)
        if (
            task_value["array_task_id"] != task
            or task_value["seed"] != seed
            or task_value["terminal_state"] != state
            or task_value["elapsed"] != elapsed
            or task_value["run_relative"] != expected_run
            or task_value["failed_raw_source_name"] != failed_name
            or task_value["failed_marker"] != failed_projection
            or task_value["absence_audit"]
            != {
                "SUCCESS_present": False,
                "recovery_result_present": False,
                "durable_checkpoint_present": False,
                "checkpoint_candidates": [],
            }
        ):
            raise ValueError(f"Gate-0 task {task} identity/failure projection is invalid")
        _validate_inventory(task_value["inventory"], task=task, failed_raw=raw[failed_name])
        expected_logs = []
        for suffix in ("out", "err"):
            source_name = f"task-{task}-slurm.{suffix}"
            expected_logs.append(
                {
                    "stream": suffix,
                    "raw_source_name": source_name,
                    "sha256": _sha256(raw[source_name]),
                    "size_bytes": len(raw[source_name]),
                }
            )
        if task_value["slurm_logs"] != expected_logs:
            raise ValueError(f"Gate-0 task {task} Slurm logs are not bound to raw bytes")

    producer = _require_exact_keys(
        payload["producer"],
        name="producer",
        keys={
            "execution_git_commit",
            "clean_worktree",
            "capture_source_blobs",
            "r2_execution_source_blobs",
        },
    )
    if (
        _GIT_COMMIT_RE.fullmatch(str(producer["execution_git_commit"])) is None
        or producer["clean_worktree"] is not True
    ):
        raise ValueError("Gate-0 producer identity is invalid")
    _validate_git_records(
        producer["capture_source_blobs"],
        name="capture_source_blobs",
        expected_count=len(_CAPTURE_SOURCE_PATHS),
    )
    _validate_git_records(
        producer["r2_execution_source_blobs"],
        name="r2_execution_source_blobs",
        expected_count=len(_R2_SOURCE_PATHS),
    )
    for record in producer["r2_execution_source_blobs"]:
        if record["git_commit"] != R2_RECOVERY_GIT_COMMIT:
            raise ValueError("Gate-0 R2 source blob is not bound to the R2 execution commit")

    container = _require_exact_keys(
        payload["container"],
        name="container",
        keys={"source_relative", "sha256", "size_bytes", "source_inode"},
    )
    if (
        type(container["source_relative"]) is not str
        or Path(container["source_relative"]).is_absolute()
        or ".." in Path(container["source_relative"]).parts
        or container["sha256"] != R2_IMAGE_SHA256
        or type(container["size_bytes"]) is not int
        or container["size_bytes"] <= 0
    ):
        raise ValueError("Gate-0 container binding is invalid")

    registries = payload["parent_registries"]
    _validate_git_records(
        registries,
        name="parent_registries",
        expected_count=len(_R2_PARENT_REGISTRIES),
    )
    expected_registry_pairs = {
        (relative.as_posix(), digest) for relative, digest in _R2_PARENT_REGISTRIES
    }
    if {
        (record["repository_relative"], record["sha256"]) for record in registries
    } != expected_registry_pairs or any(
        record["git_commit"] != R2_RECOVERY_GIT_COMMIT for record in registries
    ):
        raise ValueError("Gate-0 parent registry binding is invalid")

    expected_policy = {
        "scope": "r3_failure_parent_only",
        "profile_authorized_by_this_artifact": False,
        "training_state_reusable": False,
        "head_reusable": False,
        "optimizer_reusable": False,
        "step_reusable": False,
        "rng_reusable": False,
        "pcg_state_reusable": False,
        "beta_reusable": False,
        "scientific_effect_claim_supported": False,
    }
    if payload["failure_parent_policy"] != expected_policy:
        raise ValueError("Gate-0 failure-parent policy is not closed")
    return claimed_sha, tuple(scheduler_rows)


@dataclass(frozen=True, slots=True)
class R3Gate0Inspection:
    """Strict offline validation result; never a Gate-P authorization."""

    schema_version: str
    artifact_sha256: str
    file_sha256: str
    scheduler_rows: tuple[dict[str, object], ...]
    formal_authorization: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.schema_version != GATE0_ARTIFACT_SCHEMA:
            raise ValueError("Gate-0 inspection schema is invalid")
        _require_digest(self.artifact_sha256, name="inspection artifact SHA256")
        _require_digest(self.file_sha256, name="inspection file SHA256")
        if len(self.scheduler_rows) != 3 or self.formal_authorization is not False:
            raise ValueError("Gate-0 inspection cannot authorize Gate P")


@dataclass(frozen=True, slots=True)
class R3Gate0Capability:
    """Live production capability consumed by the R3 Gate-P admission layer."""

    schema_version: str
    role: str
    artifact_sha256: str
    file_sha256: str
    production_relative: str
    _factory_token: InitVar[object] = None
    _seal: object = field(repr=False, compare=False, init=False)

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _FACTORY_TOKEN:
            raise TypeError("R3Gate0Capability must be produced by live HPC4 verification")
        object.__setattr__(self, "_seal", _FACTORY_TOKEN)
        self._validate_structure()

    def _validate_structure(self) -> None:
        if self.schema_version != GATE0_ARTIFACT_SCHEMA or self.role != GATE0_ARTIFACT_ROLE:
            raise ValueError("R3 Gate-0 capability schema/role is invalid")
        _require_digest(self.artifact_sha256, name="capability artifact SHA256")
        _require_digest(self.file_sha256, name="capability file SHA256")
        if self.production_relative != _GATE0_RELATIVE.as_posix():
            raise ValueError("R3 Gate-0 capability is outside its production namespace")

    def validate_integrity(self) -> None:
        if getattr(self, "_seal", None) is not _FACTORY_TOKEN:
            raise TypeError("R3Gate0Capability is not sealed by live HPC4 verification")
        self._validate_structure()

    def to_artifact_ref(self) -> object:
        """Return the exact typed reference expected by ``GatePAdmission``."""

        # Imported lazily to avoid a module cycle.
        from .phase2_r3_identity import (
            GATE0_ARTIFACT_ROLE as IDENTITY_ROLE,
        )
        from .phase2_r3_identity import (
            GATE0_ARTIFACT_SCHEMA as IDENTITY_SCHEMA,
        )
        from .phase2_r3_identity import ArtifactRef

        if IDENTITY_SCHEMA != GATE0_ARTIFACT_SCHEMA or IDENTITY_ROLE != GATE0_ARTIFACT_ROLE:
            raise RuntimeError("Gate-0 schema/role drifted from the R3 identity layer")
        self.validate_integrity()
        return ArtifactRef(
            schema_version=self.schema_version,
            artifact_sha256=self.artifact_sha256,
            role=self.role,
        )


def inspect_r3_gate0_bundle(path: str | os.PathLike[str]) -> R3Gate0Inspection:
    """Validate an artifact's canonical embedded evidence without authorizing Gate P."""

    source = Path(path)
    raw, _ = _stable_file(
        source,
        name="Gate-0 artifact",
        maximum_bytes=2 * 1024 * 1024 * 1024,
    )
    payload = _decode_json(raw, name="Gate-0 artifact", require_canonical=True)
    artifact_sha, scheduler_rows = _validate_payload(payload)
    return R3Gate0Inspection(
        schema_version=GATE0_ARTIFACT_SCHEMA,
        artifact_sha256=artifact_sha,
        file_sha256=_sha256(raw),
        scheduler_rows=scheduler_rows,
    )


def _raw_record_by_name(payload: Mapping[str, object], name: str) -> Mapping[str, object]:
    values = payload["raw_sources"]
    matches = [item for item in values if item.get("name") == name]
    if len(matches) != 1:
        raise ValueError(f"Gate-0 raw source {name!r} is not unique")
    return matches[0]


def _compare_file_to_raw_record(path: Path, record: Mapping[str, object], *, name: str) -> None:
    raw, inode = _stable_file(path, name=name, maximum_bytes=512 * 1024 * 1024)
    embedded = base64.b64decode(record["bytes_base64"], validate=True)
    expected_inode = record["source_inode"]
    observed_inode = {key: inode[key] for key in ("mode_octal", "mtime_ns", "device", "inode")}
    if (
        raw != embedded
        or record["source_relative"] != _relative_to_production(path)
        or expected_inode != observed_inode
    ):
        raise ValueError(f"{name} changed after Gate-0 capture")


def _assert_live_hpc4() -> bytes:
    if os.name != "posix":
        raise RuntimeError("formal Gate-0 verification requires POSIX HPC4")
    _assert_production_roots()
    raw = _run_command(
        _SCONTROL_CONFIG_COMMAND,
        name="scontrol show config",
        maximum_bytes=4 * 1024 * 1024,
    )
    _parse_scontrol_config(raw)
    return raw


def _revalidate_frozen_scheduler_bytes(payload: Mapping[str, object]) -> None:
    """Reparse the immutable scheduler bytes embedded in the Gate-0 bundle."""

    sacct_record = _raw_record_by_name(payload, "scheduler-sacct-X.psv")
    live_control_record = _raw_record_by_name(payload, "original-live-scontrol.txt")
    try:
        sacct_raw = base64.b64decode(sacct_record["bytes_base64"], validate=True)
        live_control_raw = base64.b64decode(
            live_control_record["bytes_base64"],
            validate=True,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("Gate-0 frozen scheduler evidence is not strict base64") from error
    _parse_sacct_raw(sacct_raw)
    _parse_live_scontrol(live_control_raw, require_frozen_bytes=True)


def _revalidate_immutable_sources(
    payload: Mapping[str, object],
    *,
    container: Path,
) -> None:
    """Revalidate all retained immutable bytes without querying live Slurm."""

    _revalidate_frozen_scheduler_bytes(payload)
    if payload["producer"] != _capture_source_identity():
        raise ValueError("Gate-0 committed capture source changed after publication")
    if payload["parent_registries"] != _parent_registry_records():
        raise ValueError("Gate-0 parent registries changed after publication")
    if payload["container"] != _container_record(container):
        raise ValueError("Gate-0 container changed after publication")

    control_path = _production_path(
        _LIVE_SCONTROL_RELATIVE,
        must_exist=True,
        name="original R2 live-scontrol receipt",
    )
    _compare_file_to_raw_record(
        control_path,
        _raw_record_by_name(payload, "original-live-scontrol.txt"),
        name="original R2 live-scontrol receipt",
    )
    for task, seed, *_ in ORDERED_TASKS:
        run_path = _run_path(task, seed)
        inventory = _inventory_run(run_path, task=task, seed=seed)
        task_payload = payload["tasks"][task]
        if (
            task_payload["inventory"] != inventory["entries"]
            or task_payload["failed_marker"] != inventory["failed_projection"]
            or task_payload["absence_audit"] != inventory["absence_audit"]
        ):
            raise ValueError(f"R2 task {task} run inventory changed after Gate-0 capture")
        _compare_file_to_raw_record(
            run_path / "FAILED",
            _raw_record_by_name(payload, f"task-{task}-FAILED"),
            name=f"R2 task {task} FAILED marker",
        )
        for suffix in ("out", "err"):
            _compare_file_to_raw_record(
                _log_path(task, suffix),
                _raw_record_by_name(payload, f"task-{task}-slurm.{suffix}"),
                name=f"R2 task {task} Slurm {suffix}",
            )


def _revalidate_live_sources(payload: Mapping[str, object], *, container: Path) -> None:
    _revalidate_immutable_sources(payload, container=container)
    sacct_raw = _run_command(_SACCT_COMMAND, name="sacct", maximum_bytes=1024 * 1024)
    sacct_record = _raw_record_by_name(payload, "scheduler-sacct-X.psv")
    if base64.b64decode(sacct_record["bytes_base64"], validate=True) != sacct_raw:
        raise ValueError("Gate-0 terminal sacct bytes changed after publication")
    _parse_sacct_raw(sacct_raw)


def verify_live_r3_gate0_bundle(
    *,
    container: str | os.PathLike[str],
) -> R3Gate0Capability:
    """Reverify the exact production artifact and issue the only Gate-0 capability."""

    _assert_live_hpc4()
    path = _production_path(_GATE0_RELATIVE, must_exist=True, name="Gate-0 artifact")
    raw, inode = _stable_file(
        path,
        name="Gate-0 artifact",
        maximum_bytes=2 * 1024 * 1024 * 1024,
    )
    if inode["mode_octal"] != "0440":
        raise ValueError("published Gate-0 artifact must retain mode 0440")
    payload = _decode_json(raw, name="Gate-0 artifact", require_canonical=True)
    artifact_sha, _ = _validate_payload(payload)
    _revalidate_live_sources(payload, container=Path(container).absolute())
    capability = R3Gate0Capability(
        schema_version=GATE0_ARTIFACT_SCHEMA,
        role=GATE0_ARTIFACT_ROLE,
        artifact_sha256=artifact_sha,
        file_sha256=_sha256(raw),
        production_relative=_GATE0_RELATIVE.as_posix(),
        _factory_token=_FACTORY_TOKEN,
    )
    capability.validate_integrity()
    return capability


def _current_container_from_environment() -> Path:
    if os.name != "posix":
        raise RuntimeError("formal in-container Gate-0 verification requires POSIX HPC4")
    if os.environ.get("SLURM_CLUSTER_NAME") != "hpc4":
        raise RuntimeError("formal in-container Gate-0 verification requires ClusterName=hpc4")
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id is None or re.fullmatch(r"[1-9][0-9]*", job_id) is None:
        raise RuntimeError("formal in-container Gate-0 verification requires a live Slurm job")
    markers = {
        name: os.environ[name]
        for name in ("APPTAINER_CONTAINER", "SINGULARITY_CONTAINER")
        if os.environ.get(name)
    }
    if not markers:
        raise RuntimeError("current process is not identified as Apptainer/Singularity")
    paths = {Path(value).absolute() for value in markers.values()}
    if len(paths) != 1:
        raise RuntimeError("container environment identifies different image paths")
    container = paths.pop()
    if container.resolve(strict=True) != container or container.is_symlink():
        raise ValueError("container environment image path is not canonical")
    return container


def _verify_r3_gate0_in_container_closure(
    *,
    expected_file_sha256: str,
    container: Path,
) -> R3Gate0Capability:
    expected_sha = _require_digest(
        expected_file_sha256,
        name="expected Gate-0 file SHA256",
    )
    path = _production_path(_GATE0_RELATIVE, must_exist=True, name="Gate-0 artifact")
    raw, inode = _stable_file(
        path,
        name="Gate-0 artifact",
        maximum_bytes=2 * 1024 * 1024 * 1024,
    )
    if inode["mode_octal"] != "0440":
        raise ValueError("published Gate-0 artifact must retain mode 0440")
    file_sha = _sha256(raw)
    if file_sha != expected_sha:
        raise ValueError("Gate-0 artifact file SHA256 differs from caller expectation")
    payload = _decode_json(raw, name="Gate-0 artifact", require_canonical=True)
    artifact_sha, _ = _validate_payload(payload)
    _revalidate_immutable_sources(
        payload,
        container=container,
    )
    capability = R3Gate0Capability(
        schema_version=GATE0_ARTIFACT_SCHEMA,
        role=GATE0_ARTIFACT_ROLE,
        artifact_sha256=artifact_sha,
        file_sha256=file_sha,
        production_relative=_GATE0_RELATIVE.as_posix(),
        _factory_token=_FACTORY_TOKEN,
    )
    capability.validate_integrity()
    return capability


def verify_live_r3_gate0_in_container(
    *,
    expected_file_sha256: str,
) -> R3Gate0Capability:
    """Issue Gate-0 authority inside the exact SIF without live Slurm calls.

    Scheduler terminal bytes are reparsed only from the frozen canonical
    bundle.  Current verification is limited to the fixed production source,
    registries, retained R2 run/log/control files, and the SIF derived from
    Apptainer/Singularity process identity.
    """

    if os.name != "posix":
        raise RuntimeError("formal in-container Gate-0 verification requires POSIX HPC4")
    _assert_production_roots()
    expected_module = _production_repo_path(
        Path("src/smart_reward/phase2_r3_gate0.py"),
        must_exist=True,
        name="Gate-0 verifier source",
    )
    if Path(__file__).resolve(strict=True) != expected_module:
        raise RuntimeError("Gate-0 verifier was not imported from fixed production source")
    return _verify_r3_gate0_in_container_closure(
        expected_file_sha256=expected_file_sha256,
        container=_current_container_from_environment(),
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _require_output_namespace_directory(
    path: Path,
    *,
    name: str,
    r3_owned: bool,
) -> os.stat_result:
    info = _require_real_directory(path, name=name)
    if os.name == "posix":
        mode = stat.S_IMODE(info.st_mode)
        if r3_owned and mode not in {_OUTPUT_DIRECTORY_MODE, 0o2750}:
            raise ValueError(f"{name} must retain mode 0750 (optional setgid accepted)")
        if not mode & stat.S_IWUSR or not mode & stat.S_IXUSR:
            raise PermissionError(f"{name} must be owner-writable and owner-searchable")
        if not os.access(path, os.W_OK | os.X_OK):
            raise PermissionError(f"{name} must be writable and searchable by the current user")
    return info


def _ensure_output_parent() -> Path:
    _require_real_directory(PRODUCTION_PROJECT_ROOT, name="production project root")
    current = PRODUCTION_PROJECT_ROOT
    for index, component in enumerate(_GATE0_RELATIVE.parent.parts):
        current = current / component
        r3_owned = index > 0
        if current.exists() or current.is_symlink():
            _require_output_namespace_directory(
                current,
                name="Gate-0 output namespace",
                r3_owned=r3_owned,
            )
            continue
        parent_info = _require_real_directory(
            current.parent,
            name="Gate-0 output namespace parent",
        )
        directory_mode = _OUTPUT_DIRECTORY_MODE
        if os.name == "posix" and parent_info.st_mode & stat.S_ISGID:
            directory_mode |= stat.S_ISGID
        os.mkdir(current, mode=directory_mode)
        if os.name == "posix":
            os.chmod(current, directory_mode, follow_symlinks=False)
        _fsync_directory(current.parent)
        _require_output_namespace_directory(
            current,
            name="new Gate-0 output namespace",
            r3_owned=r3_owned,
        )
    return current


def _publish_exclusive(path: Path, raw: bytes) -> str:
    parent = path.parent
    parent_info = _require_real_directory(parent, name="Gate-0 output parent")
    parent_identity = (parent_info.st_dev, parent_info.st_ino)
    if path.exists() or path.is_symlink():
        raise FileExistsError("refusing to overwrite existing Gate-0 artifact")
    directory_fd = os.open(
        parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    temporary_name = f".{path.name}.{os.getpid()}.{secrets.token_hex(12)}.tmp"
    destination_linked = False
    publication_complete = False
    temporary_identity: tuple[int, int] | None = None

    def require_held_parent() -> None:
        named = _require_real_directory(parent, name="Gate-0 output parent")
        opened = os.fstat(directory_fd)
        if (named.st_dev, named.st_ino) != parent_identity or (
            opened.st_dev,
            opened.st_ino,
        ) != parent_identity:
            raise ValueError("Gate-0 output parent changed during publication")

    try:
        require_held_parent()
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_BINARY", 0),
            0o440,
            dir_fd=directory_fd,
        )
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fchmod(stream.fileno(), 0o440)
            os.fsync(stream.fileno())
            info = os.fstat(stream.fileno())
            temporary_identity = (info.st_dev, info.st_ino)
            if info.st_size != len(raw) or stat.S_IMODE(info.st_mode) != 0o440:
                raise OSError("temporary Gate-0 artifact inode is invalid")
        require_held_parent()
        try:
            os.link(
                temporary_name,
                path.name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileExistsError as error:
            raise FileExistsError("refusing to overwrite existing Gate-0 artifact") from error
        destination_linked = True
        require_held_parent()
        published_fd = os.open(
            path.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0),
            dir_fd=directory_fd,
        )
        try:
            published_info = os.fstat(published_fd)
            chunks: list[bytes] = []
            while True:
                chunk = os.read(published_fd, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
        finally:
            os.close(published_fd)
        published = b"".join(chunks)
        if (
            temporary_identity is None
            or (published_info.st_dev, published_info.st_ino) != temporary_identity
            or stat.S_IMODE(published_info.st_mode) != 0o440
            or published != raw
        ):
            raise ValueError("published Gate-0 inode failed descriptor verification")
        os.unlink(temporary_name, dir_fd=directory_fd)
        temporary_name = ""
        require_held_parent()
        os.fsync(directory_fd)
        publication_complete = True
        return _sha256(raw)
    finally:
        if temporary_name:
            with suppress(FileNotFoundError):
                os.unlink(temporary_name, dir_fd=directory_fd)
        if destination_linked and not publication_complete and temporary_identity is not None:
            # A failed publication is removed only when the name still points
            # to the inode created by this call.
            with suppress(FileNotFoundError):
                current = os.stat(path.name, dir_fd=directory_fd, follow_symlinks=False)
                if (current.st_dev, current.st_ino) == temporary_identity:
                    with suppress(FileNotFoundError):
                        os.unlink(path.name, dir_fd=directory_fd)
        with suppress(OSError):
            os.fsync(directory_fd)
        os.close(directory_fd)


def capture_live_r3_gate0_bundle(
    *,
    container: str | os.PathLike[str],
    now: datetime | None = None,
) -> R3Gate0Capability:
    """Capture, publish, reverify, and issue the formal R3 Gate-0 capability."""

    scontrol_config_raw = _assert_live_hpc4()
    destination = _production_path(_GATE0_RELATIVE, must_exist=False, name="Gate-0 artifact")
    _ensure_output_parent()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("refusing to overwrite existing Gate-0 artifact")
    sacct_raw = _run_command(_SACCT_COMMAND, name="sacct", maximum_bytes=1024 * 1024)
    payload = _assemble_live_payload(
        container=Path(container).absolute(),
        sacct_raw=sacct_raw,
        scontrol_config_raw=scontrol_config_raw,
        now=now,
    )
    raw = _canonical_bytes(payload)
    _publish_exclusive(destination, raw)
    return verify_live_r3_gate0_bundle(container=container)


__all__ = [
    "GATE0_ARTIFACT_ROLE",
    "GATE0_ARTIFACT_SCHEMA",
    "R3Gate0Capability",
    "R3Gate0Inspection",
    "PRODUCTION_PROJECT_ROOT",
    "PRODUCTION_REPO_ROOT",
    "capture_live_r3_gate0_bundle",
    "inspect_r3_gate0_bundle",
    "verify_live_r3_gate0_bundle",
    "verify_live_r3_gate0_in_container",
]
