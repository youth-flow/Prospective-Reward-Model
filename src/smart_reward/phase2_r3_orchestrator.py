"""Task-level scheduler-segment orchestration for formal R3 primary recovery.

This layer runs the two primary learners in their frozen order
``bt_mle -> prorm_plus``.  It does not train a head itself and it never turns
process completion into scheduler success.  Its responsibilities are:

* one independent :class:`DurableCheckpointStore` per logical head;
* no-overwrite internal head-result files and self-hashed completion receipts;
* cross-segment revalidation and skipping of already completed heads;
* a head-free, self-hashed segment outcome;
* a no-overwrite task journal with a hash chain;
* conversion of :class:`CheckpointInterruption` into a verified continuation
  checkpoint reference, never a ``SUCCESS`` claim.

The operational checkpoint interval is consumed from the exact, reopened,
tensor-free Gate-P operational bundle authorized by the design.  The science
contract continues to drive every first-order audit/progress cadence; the full
durable interval must be a multiple of that cadence.  Learning-rate boundaries,
controller completion, and scheduler-signal safe boundaries remain mandatory
full-checkpoint boundaries inside the formal head runner.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import stat
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import Any, Final, Literal, TypeAlias

from . import phase2_r3_primary as _primary
from .phase2_checkpoint import (
    CheckpointInterruption,
    CheckpointSignal,
    DurableCheckpointStore,
    PlannedSegmentBoundary,
)
from .phase2_r3_identity import (
    R3_PRIMARY_HEADS,
    ArtifactRef,
    PrimarySegmentAdmission,
)
from .phase2_r3_primary import (
    FORMAL_PRIMARY_EXECUTION_ROLE,
    FORMAL_PRIMARY_HEAD_RESULT_SCHEMA,
    FormalR3PrimaryHeadResult,
    SlurmSegmentRuntime,
    continuation_checkpoint_artifact_ref,
    formal_primary_checkpoint_binding,
    latest_generation_is_selected_terminal,
    recover_formal_result_from_selected_terminal,
    run_formal_r3_primary_head_segment,
)
from .phase2_r3_profile_artifacts import VerifiedGatePOperationalBundle

HEAD_COMPLETION_RECEIPT_SCHEMA: Final = "phase2-recovery-r3-primary-head-completion/v1"
SEGMENT_OUTCOME_SCHEMA: Final = "phase2-recovery-r3-primary-segment-outcome/v1"
TASK_JOURNAL_EVENT_SCHEMA: Final = "phase2-recovery-r3-primary-task-journal-event/v1"

_HEAD_RESULT_FILENAME: Final = "internal-head-result.json"
_HEAD_RECEIPT_FILENAME: Final = "head-completion.json"
_HEAD_STATE_DIRECTORY: Final = "durable-state"
_TASK_JOURNAL_DIRECTORY: Final = "task-journal"
_SEGMENT_OUTCOME_DIRECTORY: Final = "segment-outcomes"

PrimaryLearner: TypeAlias = Literal["bt_mle", "prorm_plus"]
FreshOrResume: TypeAlias = Literal["fresh", "resume"]
OutcomeStatus: TypeAlias = Literal[
    "continuation_required_after_safe_checkpoint",
    "compute_complete_pending_external_scheduler_terminal",
]

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_JOURNAL_FILE_RE = re.compile(r"event-([0-9]{8})\.json\Z")
_FACTORY_TOKEN = object()
HEAD_EXECUTION_SLICE_SCHEMA: Final = "phase2-recovery-r3-primary-head-execution-slice/v1"
_CURSOR_KEYS: Final = frozenset(
    {
        "global_safe_block",
        "bt_mle_completed_updates",
        "prorm_plus_completed_updates",
        "next_head",
    }
)
_ACTUAL_CURSOR_KEYS: Final = frozenset(
    {
        "bt_mle_completed_updates",
        "bt_mle_complete",
        "prorm_plus_completed_updates",
        "prorm_plus_complete",
        "next_head",
    }
)
_PlanCursor: TypeAlias = tuple[int, int, int, PrimaryLearner | None]


def _canonical_bytes(value: object) -> bytes:
    try:
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
    except (TypeError, ValueError) as error:
        raise ValueError("orchestrator evidence must contain strict JSON data") from error


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def primary_outcome_semantic_sha256(
    unsigned_outcome: Mapping[str, object],
) -> str:
    """Hash the unsigned outcome using its sole canonical, newline-terminated bytes.

    This semantic hash deliberately excludes ``outcome_sha256``.  The separate
    ``file_sha256`` on :class:`R3PrimarySegmentOutcome` hashes the complete
    canonical artifact bytes after ``outcome_sha256`` has been embedded.
    """

    if not isinstance(unsigned_outcome, Mapping):
        raise TypeError("unsigned primary outcome must be a mapping")
    if "outcome_sha256" in unsigned_outcome:
        raise ValueError("unsigned primary outcome must exclude outcome_sha256")
    return _canonical_sha256(dict(unsigned_outcome))


def _require_digest(value: object, *, name: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_positive_integer(value: object, *, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _require_nonnegative_integer(value: object, *, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
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


def _decode_canonical(raw: bytes, *, name: str) -> dict[str, Any]:
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
    if not isinstance(value, dict) or raw != _canonical_bytes(value):
        raise ValueError(f"{name} is not canonical JSON")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_real_directory(path: Path, *, name: str) -> os.stat_result:
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or path.is_symlink() or path.resolve(strict=True) != path:
        raise ValueError(f"{name} must be a canonical real directory")
    return info


def _require_real_file(path: Path, *, name: str) -> os.stat_result:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or path.is_symlink():
        raise ValueError(f"{name} must be a regular non-symlink file")
    if os.name == "posix" and stat.S_IMODE(info.st_mode) != 0o440:
        raise ValueError(f"{name} must retain mode 0440")
    return info


def _fsync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_child_directory(parent: Path, name: str) -> Path:
    _require_real_directory(parent, name="orchestrator directory parent")
    child = parent / name
    if child.exists() or child.is_symlink():
        _require_real_directory(child, name=f"orchestrator directory {name}")
        return child
    os.mkdir(child, mode=0o550)
    _fsync_directory(parent)
    _require_real_directory(child, name=f"orchestrator directory {name}")
    return child


def _publish_no_overwrite(path: Path, raw: bytes, *, name: str) -> str:
    """Publish one immutable file, verify the linked inode, and fsync its parent."""

    parent = path.parent
    parent_info = _require_real_directory(parent, name=f"{name} parent")
    parent_identity = (parent_info.st_dev, parent_info.st_ino)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing {name}")
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
            info = os.fstat(stream.fileno())
            temporary_identity = (info.st_dev, info.st_ino)
            if info.st_size != len(raw):
                raise OSError(f"temporary {name} has the wrong byte size")
        current_parent = parent.stat()
        if (current_parent.st_dev, current_parent.st_ino) != parent_identity:
            raise ValueError(f"{name} parent changed before publication")
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as error:
            raise FileExistsError(f"refusing to overwrite existing {name}") from error
        destination_linked = True
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
        published_fd = os.open(path, flags)
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
            or published != raw
            or (os.name == "posix" and stat.S_IMODE(published_info.st_mode) != 0o440)
        ):
            raise ValueError(f"published {name} inode failed verification")
        temporary.unlink()
        _fsync_directory(parent)
        publication_complete = True
        return hashlib.sha256(raw).hexdigest()
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()
        if destination_linked and not publication_complete and temporary_identity is not None:
            with suppress(FileNotFoundError):
                destination_info = path.lstat()
                if (
                    not stat.S_ISLNK(destination_info.st_mode)
                    and (destination_info.st_dev, destination_info.st_ino) == temporary_identity
                ):
                    path.unlink()
        if destination_linked:
            _fsync_directory(parent)


def _read_canonical_file(path: Path, *, name: str) -> tuple[dict[str, Any], bytes]:
    _require_real_file(path, name=name)
    raw = path.read_bytes()
    return _decode_canonical(raw, name=name), raw


def _learner(value: object) -> PrimaryLearner:
    if value not in R3_PRIMARY_HEADS:
        raise ValueError(f"learner must be one of {R3_PRIMARY_HEADS!r}")
    return value  # type: ignore[return-value]


def _require_admission(value: object) -> PrimarySegmentAdmission:
    if type(value) is not PrimarySegmentAdmission:
        raise TypeError("admission must be an exact PrimarySegmentAdmission")
    value.validate_integrity()
    if value.segment_index == 1:
        if (
            value.start_mode != "fresh_zero_head_fresh_adamw"
            or value.continuation_evidence is not None
        ):
            raise ValueError("segment 1 must be a fresh admitted segment")
    elif (
        value.start_mode != "verified_state_complete_continuation"
        or value.continuation_evidence is None
    ):
        raise ValueError("later segments require an admission continuation capability")
    return value


def _require_runtime(
    value: object,
    *,
    admission: PrimarySegmentAdmission,
) -> SlurmSegmentRuntime:
    if type(value) is not SlurmSegmentRuntime:
        raise TypeError("runtime must be an exact SlurmSegmentRuntime")
    value.validate_integrity()
    expected = {
        "design_sha256": admission.design.design_sha256,
        "admission_sha256": admission.admission_sha256,
        "scheduler_segment_id": admission.scheduler_segment_id,
        "segment_index": admission.segment_index,
        "task_id": admission.task_id,
        "seed": admission.seed,
    }
    if any(getattr(value, name) != expected_value for name, expected_value in expected.items()):
        raise ValueError("runtime does not match the admitted scheduler segment")
    return value


def _validate_operational_policy(
    value: object,
    *,
    admission: PrimarySegmentAdmission,
    runtime: SlurmSegmentRuntime,
) -> VerifiedGatePOperationalBundle:
    """Validate the exact reopened Gate-P bundle before task artifacts."""

    if type(value) is not VerifiedGatePOperationalBundle:
        raise TypeError("operational_policy must be an exact VerifiedGatePOperationalBundle")
    plan = _primary._validate_primary_resource_plan(
        admission=admission,
        runtime=runtime,
        resource_plan=value,
    )
    science_interval = _require_positive_integer(
        admission.design.science.settings.convergence.check_interval,
        name="science first-order audit interval",
    )
    if plan.audit_cadence_updates != science_interval:
        raise ValueError("Gate-P audit cadence differs from the frozen science cadence")
    durable_interval = _require_positive_integer(
        plan.durable_checkpoint_cadence_updates,
        name="Gate-P durable checkpoint cadence",
    )
    if durable_interval % science_interval:
        raise ValueError("Gate-P durable checkpoint cadence must align to the science audit grid")
    mandatory_flags = {
        "checkpoint_before_head_transition": plan.checkpoint_before_head_transition,
        "checkpoint_on_signal_safe_boundary": (plan.checkpoint_on_signal_safe_boundary),
        "checkpoint_at_segment_terminal": plan.checkpoint_at_segment_terminal,
        "checkpoint_before_resume": plan.checkpoint_before_resume,
    }
    if any(flag is not True for flag in mandatory_flags.values()):
        raise ValueError("Gate-P mandatory checkpoint boundary policy is incomplete")
    return plan


def _plan_cursor_payload(cursor: _PlanCursor) -> dict[str, object]:
    return {
        "global_safe_block": cursor[0],
        "bt_mle_completed_updates": cursor[1],
        "prorm_plus_completed_updates": cursor[2],
        "next_head": cursor[3],
    }


def _validate_plan_cursor(
    value: object,
    *,
    name: str,
    maximum_updates_per_head: int,
    audit_cadence_updates: int,
) -> _PlanCursor:
    cursor = _require_exact_keys(value, name=name, keys=_CURSOR_KEYS)
    global_block = _require_nonnegative_integer(
        cursor["global_safe_block"],
        name=f"{name}.global_safe_block",
    )
    bt_updates = _require_nonnegative_integer(
        cursor["bt_mle_completed_updates"],
        name=f"{name}.bt_mle_completed_updates",
    )
    prorm_updates = _require_nonnegative_integer(
        cursor["prorm_plus_completed_updates"],
        name=f"{name}.prorm_plus_completed_updates",
    )
    if (
        bt_updates > maximum_updates_per_head
        or prorm_updates > maximum_updates_per_head
        or bt_updates % audit_cadence_updates
        or prorm_updates % audit_cadence_updates
    ):
        raise ValueError(f"{name} is outside the frozen science audit grid")
    if bt_updates < maximum_updates_per_head:
        expected_next: PrimaryLearner | None = "bt_mle"
        if prorm_updates != 0:
            raise ValueError(f"{name} violates strict BT -> ProRM+ order")
    elif prorm_updates < maximum_updates_per_head:
        expected_next = "prorm_plus"
    else:
        expected_next = None
    expected_global_block = (bt_updates + prorm_updates) // audit_cadence_updates
    if cursor["next_head"] != expected_next or global_block != expected_global_block:
        raise ValueError(f"{name} is not a canonical R3 plan cursor")
    return (global_block, bt_updates, prorm_updates, expected_next)


@dataclass(frozen=True, slots=True)
class _NominalSegmentWindow:
    start_cursor: _PlanCursor
    end_cursor: _PlanCursor
    max_safe_update_blocks_to_execute: int


def _nominal_segment_window(
    plan: VerifiedGatePOperationalBundle,
    *,
    admission: PrimarySegmentAdmission,
) -> _NominalSegmentWindow:
    maximum_updates = _require_positive_integer(
        admission.design.science.settings.convergence.max_steps,
        name="maximum updates per head",
    )
    audit_cadence = _require_positive_integer(
        admission.design.science.settings.convergence.check_interval,
        name="science audit cadence",
    )
    boundaries = plan.segment_boundaries
    if (
        type(boundaries) is not tuple
        or len(boundaries) != plan.max_scheduler_segments
        or len(boundaries) != admission.design.max_scheduler_segments
    ):
        raise ValueError("Gate-P segment-boundary count differs from the admission")
    previous_end: _PlanCursor | None = None
    selected: _NominalSegmentWindow | None = None
    for expected_index, raw_segment in enumerate(boundaries, start=1):
        if not isinstance(raw_segment, Mapping):
            raise TypeError("Gate-P segment boundary must be a mapping")
        if raw_segment.get("segment_index") != expected_index:
            raise ValueError("Gate-P segment-boundary indices are not contiguous")
        start = _validate_plan_cursor(
            raw_segment.get("start_boundary"),
            name=f"Gate-P segment {expected_index} start cursor",
            maximum_updates_per_head=maximum_updates,
            audit_cadence_updates=audit_cadence,
        )
        end = _validate_plan_cursor(
            raw_segment.get("end_boundary"),
            name=f"Gate-P segment {expected_index} end cursor",
            maximum_updates_per_head=maximum_updates,
            audit_cadence_updates=audit_cadence,
        )
        if previous_end is None:
            if start != (0, 0, 0, "bt_mle"):
                raise ValueError("Gate-P first segment does not start at the R3 origin")
        elif start != previous_end:
            raise ValueError("Gate-P segment cursors are not contiguous")
        block_budget = end[0] - start[0]
        if block_budget < 1 or end[1] < start[1] or end[2] < start[2]:
            raise ValueError("Gate-P segment has a non-positive or regressing budget")
        if (
            raw_segment.get("max_safe_update_blocks_to_execute") != block_budget
            or raw_segment.get("fixed_ordered_head_transition_allowed") != tuple(R3_PRIMARY_HEADS)
            or raw_segment.get("journal_actual_cursor_required") is not True
            or raw_segment.get("nominal_boundaries_are_worst_case_projection_only") is not True
            or raw_segment.get("actual_cursor_must_reach_nominal_end") is not False
        ):
            raise ValueError("Gate-P segment execution contract is inconsistent")
        continuation_required = raw_segment.get("continuation_required")
        expected_continuation = expected_index < len(boundaries)
        if continuation_required is not expected_continuation:
            raise ValueError("Gate-P segment continuation marker differs from its plan position")
        window = _NominalSegmentWindow(
            start_cursor=start,
            end_cursor=end,
            max_safe_update_blocks_to_execute=block_budget,
        )
        if expected_index == admission.segment_index:
            selected = window
        previous_end = end
    terminal = (
        2 * (maximum_updates // audit_cadence),
        maximum_updates,
        maximum_updates,
        None,
    )
    if previous_end != terminal:
        raise ValueError("Gate-P final segment does not cover both primary heads")
    if selected is None:
        raise ValueError("admission segment is absent from the Gate-P plan")
    return selected


_ActualCursor: TypeAlias = tuple[int, bool, int, bool, PrimaryLearner | None]


def _actual_cursor_payload(cursor: _ActualCursor) -> dict[str, object]:
    return {
        "bt_mle_completed_updates": cursor[0],
        "bt_mle_complete": cursor[1],
        "prorm_plus_completed_updates": cursor[2],
        "prorm_plus_complete": cursor[3],
        "next_head": cursor[4],
    }


def _validate_actual_cursor_value(
    value: object,
    *,
    name: str,
    maximum_updates_per_head: int,
    audit_cadence_updates: int,
) -> _ActualCursor:
    cursor = _require_exact_keys(value, name=name, keys=_ACTUAL_CURSOR_KEYS)
    bt_updates = _require_nonnegative_integer(
        cursor["bt_mle_completed_updates"],
        name=f"{name}.bt_mle_completed_updates",
    )
    prorm_updates = _require_nonnegative_integer(
        cursor["prorm_plus_completed_updates"],
        name=f"{name}.prorm_plus_completed_updates",
    )
    bt_complete = cursor["bt_mle_complete"]
    prorm_complete = cursor["prorm_plus_complete"]
    if type(bt_complete) is not bool or type(prorm_complete) is not bool:
        raise TypeError(f"{name} completion markers must be bool")
    if (
        bt_updates > maximum_updates_per_head
        or prorm_updates > maximum_updates_per_head
        or bt_updates % audit_cadence_updates
        or prorm_updates % audit_cadence_updates
    ):
        raise ValueError(f"{name} is outside the science audit grid")
    if not bt_complete:
        if prorm_updates != 0 or prorm_complete:
            raise ValueError(f"{name} violates strict BT -> ProRM+ order")
        expected_next: PrimaryLearner | None = "bt_mle"
    elif not prorm_complete:
        expected_next = "prorm_plus"
    else:
        expected_next = None
    if cursor["next_head"] != expected_next:
        raise ValueError(f"{name} has an invalid next head")
    return (
        bt_updates,
        bt_complete,
        prorm_updates,
        prorm_complete,
        expected_next,
    )


def _head_execution_slice_payload(
    *,
    resource_plan_sha256: str,
    formal_profile_sha256: str,
    profile_run_sha256: str,
    design_sha256: str,
    admission_sha256: str,
    logical_run_id: str,
    head_run_id: str,
    scheduler_segment_id: str,
    runtime_sha256: str,
    segment_index: int,
    task_id: int,
    seed: int,
    head: PrimaryLearner,
    fresh_or_resume: FreshOrResume,
    science_audit_cadence_updates: int,
    maximum_updates_per_head: int,
    max_safe_update_blocks_to_execute: int,
    safe_update_blocks_consumed_before_head: int,
    safe_update_blocks_available_to_head: int,
    start_completed_updates: int,
    end_completed_updates_inclusive: int,
    nominal_segment_start_cursor: _PlanCursor,
    nominal_segment_end_cursor: _PlanCursor,
    actual_cursor_before_head: _ActualCursor,
    predecessor_checkpoint: ArtifactRef | None,
) -> dict[str, object]:
    start_cursor = _plan_cursor_payload(nominal_segment_start_cursor)
    end_cursor = _plan_cursor_payload(nominal_segment_end_cursor)
    actual_cursor = _actual_cursor_payload(actual_cursor_before_head)
    return {
        "schema_version": HEAD_EXECUTION_SLICE_SCHEMA,
        "resource_plan_sha256": resource_plan_sha256,
        "formal_profile_sha256": formal_profile_sha256,
        "profile_run_sha256": profile_run_sha256,
        "design_sha256": design_sha256,
        "admission_sha256": admission_sha256,
        "logical_run_id": logical_run_id,
        "head_run_id": head_run_id,
        "scheduler_segment_id": scheduler_segment_id,
        "runtime_sha256": runtime_sha256,
        "segment_index": segment_index,
        "task_id": task_id,
        "seed": seed,
        "head": head,
        "fresh_or_resume": fresh_or_resume,
        "science_audit_cadence_updates": science_audit_cadence_updates,
        "maximum_updates_per_head": maximum_updates_per_head,
        "max_safe_update_blocks_to_execute": (max_safe_update_blocks_to_execute),
        "safe_update_blocks_consumed_before_head": (safe_update_blocks_consumed_before_head),
        "safe_update_blocks_available_to_head": (safe_update_blocks_available_to_head),
        "start_completed_updates": start_completed_updates,
        "end_completed_updates_inclusive": end_completed_updates_inclusive,
        "nominal_segment_start_cursor": start_cursor,
        "nominal_segment_start_cursor_sha256": _canonical_sha256(start_cursor),
        "nominal_segment_end_cursor": end_cursor,
        "nominal_segment_end_cursor_sha256": _canonical_sha256(end_cursor),
        "actual_cursor_before_head": actual_cursor,
        "actual_cursor_before_head_sha256": _canonical_sha256(actual_cursor),
        "predecessor_checkpoint": (
            None if predecessor_checkpoint is None else _checkpoint_ref_dict(predecessor_checkpoint)
        ),
        "information_boundary": "operational_cursor_only_no_scientific_adaptation",
    }


@dataclass(frozen=True, slots=True)
class PrimaryHeadExecutionSlice:
    """Self-hashed, pure-data budget contract passed to one formal head run."""

    head: PrimaryLearner
    fresh_or_resume: FreshOrResume
    start_completed_updates: int
    end_completed_updates_inclusive: int
    max_safe_update_blocks_to_execute: int
    safe_update_blocks_consumed_before_head: int
    safe_update_blocks_available_to_head: int
    slice_sha256: str
    _payload: Mapping[str, object] = field(repr=False, compare=False)
    _factory_token: InitVar[object] = None
    _seal: object = field(repr=False, compare=False, init=False)

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _FACTORY_TOKEN:
            raise TypeError("head execution slice must be derived from a Gate-P plan")
        object.__setattr__(self, "_seal", _FACTORY_TOKEN)
        self._validate_structure()

    def _validate_structure(self) -> None:
        _learner(self.head)
        if self.fresh_or_resume not in {"fresh", "resume"}:
            raise ValueError("head execution slice mode is invalid")
        _require_nonnegative_integer(
            self.start_completed_updates,
            name="slice start_completed_updates",
        )
        if (
            _require_positive_integer(
                self.end_completed_updates_inclusive,
                name="slice end_completed_updates_inclusive",
            )
            <= self.start_completed_updates
        ):
            raise ValueError("head execution slice must permit at least one update")
        _require_positive_integer(
            self.max_safe_update_blocks_to_execute,
            name="slice segment block budget",
        )
        _require_nonnegative_integer(
            self.safe_update_blocks_consumed_before_head,
            name="slice consumed block budget",
        )
        _require_positive_integer(
            self.safe_update_blocks_available_to_head,
            name="slice available block budget",
        )
        digest = _require_digest(self.slice_sha256, name="slice_sha256")
        payload = dict(self._payload)
        if digest != _canonical_sha256(payload):
            raise ValueError("head execution slice self-hash is invalid")

    def validate_integrity(self) -> None:
        if getattr(self, "_seal", None) is not _FACTORY_TOKEN:
            raise TypeError("head execution slice lacks its Gate-P derivation seal")
        self._validate_structure()

    def to_dict(self) -> dict[str, object]:
        self.validate_integrity()
        return json.loads(json.dumps({**self._payload, "slice_sha256": self.slice_sha256}))


def _make_head_execution_slice(
    *,
    plan: VerifiedGatePOperationalBundle,
    admission: PrimarySegmentAdmission,
    runtime: SlurmSegmentRuntime,
    window: _NominalSegmentWindow,
    actual_cursor: _ActualCursor,
    learner: PrimaryLearner,
    mode: FreshOrResume,
    consumed_blocks: int,
    predecessor_checkpoint: ArtifactRef | None,
) -> PrimaryHeadExecutionSlice:
    head = _learner(learner)
    maximum_updates = _require_positive_integer(
        admission.design.science.settings.convergence.max_steps,
        name="maximum updates per head",
    )
    audit_cadence = _require_positive_integer(
        admission.design.science.settings.convergence.check_interval,
        name="science audit cadence",
    )
    if actual_cursor[4] != head:
        raise ValueError("execution slice head differs from the actual strict-order cursor")
    start = actual_cursor[0] if head == "bt_mle" else actual_cursor[2]
    remaining_segment_blocks = window.max_safe_update_blocks_to_execute - consumed_blocks
    if remaining_segment_blocks < 1:
        raise RuntimeError("scheduler segment has exhausted its Gate-P block budget")
    remaining_head_blocks = (maximum_updates - start) // audit_cadence
    available = min(remaining_segment_blocks, remaining_head_blocks)
    if available < 1:
        raise ValueError("actual head cursor has no executable updates")
    if mode == "fresh":
        if start != 0 or predecessor_checkpoint is not None:
            raise ValueError("fresh head slice must start at zero without a checkpoint")
    else:
        if start == 0 or predecessor_checkpoint is None:
            raise ValueError("resume head slice requires nonzero verified predecessor state")
        predecessor_checkpoint.validate_integrity()
    payload = _head_execution_slice_payload(
        resource_plan_sha256=plan.resource_plan_sha256,
        formal_profile_sha256=plan.formal_profile_sha256,
        profile_run_sha256=plan.profile_run_sha256,
        design_sha256=admission.design.design_sha256,
        admission_sha256=admission.admission_sha256,
        logical_run_id=admission.logical_run_id,
        head_run_id=admission.head_run_ids[R3_PRIMARY_HEADS.index(head)],
        scheduler_segment_id=admission.scheduler_segment_id,
        runtime_sha256=runtime.runtime_sha256,
        segment_index=admission.segment_index,
        task_id=admission.task_id,
        seed=admission.seed,
        head=head,
        fresh_or_resume=mode,
        science_audit_cadence_updates=audit_cadence,
        maximum_updates_per_head=maximum_updates,
        max_safe_update_blocks_to_execute=(window.max_safe_update_blocks_to_execute),
        safe_update_blocks_consumed_before_head=consumed_blocks,
        safe_update_blocks_available_to_head=available,
        start_completed_updates=start,
        end_completed_updates_inclusive=start + available * audit_cadence,
        nominal_segment_start_cursor=window.start_cursor,
        nominal_segment_end_cursor=window.end_cursor,
        actual_cursor_before_head=actual_cursor,
        predecessor_checkpoint=predecessor_checkpoint,
    )
    result = PrimaryHeadExecutionSlice(
        head=head,
        fresh_or_resume=mode,
        start_completed_updates=start,
        end_completed_updates_inclusive=(start + available * audit_cadence),
        max_safe_update_blocks_to_execute=(window.max_safe_update_blocks_to_execute),
        safe_update_blocks_consumed_before_head=consumed_blocks,
        safe_update_blocks_available_to_head=available,
        slice_sha256=_canonical_sha256(payload),
        _payload=payload,
        _factory_token=_FACTORY_TOKEN,
    )
    result.validate_integrity()
    return result


def _blocks_consumed_through_cursor(
    execution_slice: PrimaryHeadExecutionSlice,
    cursor: _ActualCursor,
    *,
    audit_cadence_updates: int,
) -> int:
    completed = cursor[0] if execution_slice.head == "bt_mle" else cursor[2]
    delta = completed - execution_slice.start_completed_updates
    if (
        delta < 0
        or delta % audit_cadence_updates
        or completed > execution_slice.end_completed_updates_inclusive
    ):
        raise ValueError("durable actual cursor violates its head execution slice")
    consumed = (
        execution_slice.safe_update_blocks_consumed_before_head + delta // audit_cadence_updates
    )
    if consumed > execution_slice.max_safe_update_blocks_to_execute:
        raise ValueError("durable actual cursor exceeds the Gate-P segment budget")
    return consumed


def _admission_chain(
    current: PrimarySegmentAdmission,
) -> dict[int, PrimarySegmentAdmission]:
    admitted = _require_admission(current)
    result: dict[int, PrimarySegmentAdmission] = {}
    cursor = admitted
    while True:
        if cursor.segment_index in result:
            raise ValueError("continuation admission chain contains a cycle")
        result[cursor.segment_index] = cursor
        if cursor.segment_index == 1:
            break
        evidence = cursor.continuation_evidence
        if evidence is None:
            raise ValueError("continuation admission chain is incomplete")
        cursor = evidence.predecessor
        cursor.validate_integrity()
    if set(result) != set(range(1, admitted.segment_index + 1)):
        raise ValueError("continuation admission chain is not contiguous")
    return result


def _internal_result_expected_identity(
    admission: PrimarySegmentAdmission,
    learner: PrimaryLearner,
) -> dict[str, object]:
    context = admission.materialization.context
    return {
        "schema_version": FORMAL_PRIMARY_HEAD_RESULT_SCHEMA,
        "campaign_kind": admission.design.campaign_kind,
        "execution_revision": admission.design.execution_revision,
        "campaign_role": admission.design.campaign_role,
        "execution_role": FORMAL_PRIMARY_EXECUTION_ROLE,
        "design_sha256": admission.design.design_sha256,
        "logical_run_id": admission.logical_run_id,
        "head_run_id": admission.head_run_ids[R3_PRIMARY_HEADS.index(learner)],
        "task_id": admission.task_id,
        "seed": admission.seed,
        "learner": learner,
        "science_semantic_sha256": admission.design.science.semantic_sha256,
        "science_file_sha256": admission.design.science.file_sha256,
        "materialization_attestation_sha256": (admission.materialization.attestation_sha256),
        "context_sha256": context.context_sha256,
        "input_training_sha256": context.input_training_sha256,
        "prepared_training_sha256": context.primary_training_sha256,
        "oracle_reward_sha256": context.oracle_reward_sha256,
        "label_stream_sha256": context.label_stream.label_stream_sha256,
    }


_INTERNAL_RESULT_KEYS = frozenset(
    {
        "schema_version",
        "campaign_kind",
        "execution_revision",
        "campaign_role",
        "execution_role",
        "design_sha256",
        "admission_sha256",
        "logical_run_id",
        "head_run_id",
        "scheduler_segment_id",
        "runtime_sha256",
        "segment_index",
        "task_id",
        "seed",
        "learner",
        "science_semantic_sha256",
        "science_file_sha256",
        "materialization_attestation_sha256",
        "context_sha256",
        "input_training_sha256",
        "prepared_training_sha256",
        "oracle_reward_sha256",
        "label_stream_sha256",
        "selected_primary_step",
        "controller_updates_executed",
        "head_execution_slice_sha256",
        "head",
        "terminal_checkpoint_artifact_sha256",
        "resumed_from_predecessor",
        "information_boundary",
        "external_scheduler_terminal_validated",
        "formal_r3_evidence",
        "result_sha256",
    }
)


def _validate_internal_result(
    value: object,
    *,
    admission: PrimarySegmentAdmission,
    learner: PrimaryLearner,
) -> tuple[dict[str, Any], PrimarySegmentAdmission]:
    result = _require_exact_keys(
        value,
        name=f"{learner} internal result",
        keys=_INTERNAL_RESULT_KEYS,
    )
    expected = _internal_result_expected_identity(admission, learner)
    if any(result.get(name) != expected_value for name, expected_value in expected.items()):
        raise ValueError(f"{learner} internal result identity is invalid")
    segment_index = _require_positive_integer(
        result.get("segment_index"),
        name=f"{learner} result segment_index",
    )
    chain = _admission_chain(admission)
    completion_admission = chain.get(segment_index)
    if completion_admission is None:
        raise ValueError(f"{learner} result belongs to an unadmitted segment")
    if (
        result.get("admission_sha256") != completion_admission.admission_sha256
        or result.get("scheduler_segment_id") != completion_admission.scheduler_segment_id
    ):
        raise ValueError(f"{learner} result does not bind its completion admission")
    _require_digest(result.get("runtime_sha256"), name=f"{learner} result runtime SHA")
    _require_digest(
        result.get("terminal_checkpoint_artifact_sha256"),
        name=f"{learner} terminal checkpoint SHA",
    )
    selected_step = _require_positive_integer(
        result.get("selected_primary_step"),
        name=f"{learner} selected primary step",
    )
    controller_updates = _require_positive_integer(
        result.get("controller_updates_executed"),
        name=f"{learner} controller updates",
    )
    if selected_step > controller_updates:
        raise ValueError(f"{learner} selected step exceeds controller execution")
    _require_digest(
        result.get("head_execution_slice_sha256"),
        name=f"{learner} head execution slice SHA",
    )
    if not isinstance(result.get("head"), dict) or not result["head"]:
        raise ValueError(f"{learner} internal result lacks its internal head evidence")
    if type(result.get("resumed_from_predecessor")) is not bool:
        raise TypeError(f"{learner} result resume flag must be bool")
    if result.get("information_boundary") != {
        "train_only": True,
        "validation_or_test_data_accessed": False,
        "policy_session_opened": False,
        "policy_rollout_performed": False,
        "beta_outcome_computed": False,
        "controls_executed": False,
    }:
        raise ValueError(f"{learner} internal result crossed the train-only boundary")
    if (
        result.get("external_scheduler_terminal_validated") is not False
        or result.get("formal_r3_evidence") is not False
    ):
        raise ValueError(f"{learner} internal result incorrectly claims finalized R3 evidence")
    result_sha = _require_digest(
        result.get("result_sha256"),
        name=f"{learner} result_sha256",
    )
    unsigned = dict(result)
    del unsigned["result_sha256"]
    if result_sha != _canonical_sha256(unsigned):
        raise ValueError(f"{learner} internal result self-hash is invalid")
    return result, completion_admission


def _checkpoint_ref_dict(value: ArtifactRef) -> dict[str, str]:
    value.validate_integrity()
    return value.to_dict()


def _terminal_checkpoint_artifact_sha256(
    checkpoint_store: DurableCheckpointStore,
    *,
    admission: PrimarySegmentAdmission,
    learner: PrimaryLearner,
) -> str:
    """Re-audit the completed head's terminal checkpoint without continuation."""

    return _primary._latest_generation_artifact_sha256(
        checkpoint_store,
        admission=admission,
        learner=learner,
        continuation_required=False,
    )


def _completion_receipt_payload(
    *,
    current_admission: PrimarySegmentAdmission,
    completion_admission: PrimarySegmentAdmission,
    learner: PrimaryLearner,
    internal_result: Mapping[str, object],
    result_file_sha256: str,
    result_file_size_bytes: int,
    checkpoint_store: DurableCheckpointStore,
    terminal_checkpoint_artifact_sha256: str,
) -> dict[str, object]:
    terminal_checkpoint_sha = _require_digest(
        terminal_checkpoint_artifact_sha256,
        name=f"{learner} terminal checkpoint artifact SHA",
    )
    return {
        "schema_version": HEAD_COMPLETION_RECEIPT_SCHEMA,
        "design_sha256": current_admission.design.design_sha256,
        "logical_run_id": current_admission.logical_run_id,
        "head_run_id": current_admission.head_run_ids[R3_PRIMARY_HEADS.index(learner)],
        "task_id": current_admission.task_id,
        "seed": current_admission.seed,
        "learner": learner,
        "completion_segment_index": completion_admission.segment_index,
        "completion_admission_sha256": completion_admission.admission_sha256,
        "completion_scheduler_segment_id": (completion_admission.scheduler_segment_id),
        "completion_runtime_sha256": internal_result["runtime_sha256"],
        "controller_updates_executed": internal_result["controller_updates_executed"],
        "internal_result": {
            "filename": _HEAD_RESULT_FILENAME,
            "file_sha256": result_file_sha256,
            "size_bytes": result_file_size_bytes,
            "unfinalized_head_output_sha256": internal_result["result_sha256"],
        },
        "checkpoint_store_binding_sha256": checkpoint_store.binding_sha256,
        "terminal_checkpoint_artifact_sha256": terminal_checkpoint_sha,
        "information_boundary": "train_only_internal_head_evidence",
    }


def _validate_completion_receipt(
    value: object,
    *,
    admission: PrimarySegmentAdmission,
    learner: PrimaryLearner,
    result: Mapping[str, object],
    result_raw: bytes,
    checkpoint_store: DurableCheckpointStore,
    completion_admission: PrimarySegmentAdmission,
) -> dict[str, Any]:
    receipt = _require_exact_keys(
        value,
        name=f"{learner} completion receipt",
        keys={
            "schema_version",
            "design_sha256",
            "logical_run_id",
            "head_run_id",
            "task_id",
            "seed",
            "learner",
            "completion_segment_index",
            "completion_admission_sha256",
            "completion_scheduler_segment_id",
            "completion_runtime_sha256",
            "controller_updates_executed",
            "internal_result",
            "checkpoint_store_binding_sha256",
            "terminal_checkpoint_artifact_sha256",
            "information_boundary",
            "receipt_sha256",
        },
    )
    terminal_checkpoint_sha = _terminal_checkpoint_artifact_sha256(
        checkpoint_store,
        admission=completion_admission,
        learner=learner,
    )
    expected = _completion_receipt_payload(
        current_admission=admission,
        completion_admission=completion_admission,
        learner=learner,
        internal_result=result,
        result_file_sha256=hashlib.sha256(result_raw).hexdigest(),
        result_file_size_bytes=len(result_raw),
        checkpoint_store=checkpoint_store,
        terminal_checkpoint_artifact_sha256=terminal_checkpoint_sha,
    )
    if any(receipt.get(name) != expected_value for name, expected_value in expected.items()):
        raise ValueError(f"{learner} completion receipt does not match durable evidence")
    receipt_sha = _require_digest(
        receipt.get("receipt_sha256"),
        name=f"{learner} completion receipt SHA",
    )
    if receipt_sha != _canonical_sha256(expected):
        raise ValueError(f"{learner} completion receipt self-hash is invalid")
    return receipt


def _head_paths(task_root: Path, learner: PrimaryLearner) -> tuple[Path, Path, Path]:
    heads = _ensure_child_directory(task_root, "heads")
    head_root = _ensure_child_directory(heads, learner)
    return (
        head_root,
        head_root / _HEAD_RESULT_FILENAME,
        head_root / _HEAD_RECEIPT_FILENAME,
    )


def _checkpoint_store(
    head_root: Path,
    *,
    admission: PrimarySegmentAdmission,
    learner: PrimaryLearner,
) -> DurableCheckpointStore:
    return DurableCheckpointStore(
        head_root / _HEAD_STATE_DIRECTORY,
        objective=learner,
        binding=formal_primary_checkpoint_binding(admission, learner),
    )


def _head_durable_completed_updates(
    store: DurableCheckpointStore,
    *,
    learner: PrimaryLearner,
) -> int:
    audited = store.audit_generations(verify_all_checkpoint_bytes=True)
    if not audited:
        return 0
    return _require_nonnegative_integer(
        audited[-1].get("completed_steps"),
        name=f"{learner} latest durable completed_steps",
    )


def _actual_cursor(
    *,
    admission: PrimarySegmentAdmission,
    receipts: Mapping[str, Mapping[str, object]],
    stores: Mapping[str, DurableCheckpointStore],
) -> _ActualCursor:
    maximum_updates = _require_positive_integer(
        admission.design.science.settings.convergence.max_steps,
        name="maximum updates per head",
    )
    audit_cadence = _require_positive_integer(
        admission.design.science.settings.convergence.check_interval,
        name="science audit cadence",
    )
    state: dict[str, tuple[int, bool]] = {}
    for learner_name in R3_PRIMARY_HEADS:
        learner = _learner(learner_name)
        receipt = receipts.get(learner)
        if receipt is None:
            updates = _head_durable_completed_updates(
                stores[learner],
                learner=learner,
            )
            complete = False
        else:
            updates = _require_positive_integer(
                receipt.get("controller_updates_executed"),
                name=f"{learner} completed-head update count",
            )
            complete = True
        if updates > maximum_updates or updates % audit_cadence:
            raise ValueError(f"{learner} durable cursor is outside the science audit grid")
        state[learner] = (updates, complete)
    bt_updates, bt_complete = state["bt_mle"]
    prorm_updates, prorm_complete = state["prorm_plus"]
    if not bt_complete:
        if prorm_updates != 0 or prorm_complete:
            raise ValueError("durable cursor violates strict BT -> ProRM+ order")
        next_head: PrimaryLearner | None = "bt_mle"
    elif not prorm_complete:
        next_head = "prorm_plus"
    else:
        next_head = None
    return (
        bt_updates,
        bt_complete,
        prorm_updates,
        prorm_complete,
        next_head,
    )


def _load_or_recover_completion(
    *,
    admission: PrimarySegmentAdmission,
    learner: PrimaryLearner,
    result_path: Path,
    receipt_path: Path,
    checkpoint_store: DurableCheckpointStore,
) -> tuple[dict[str, Any] | None, bool]:
    result_present = result_path.exists() or result_path.is_symlink()
    receipt_present = receipt_path.exists() or receipt_path.is_symlink()
    if receipt_present and not result_present:
        raise RuntimeError(f"{learner} completion receipt exists without internal result")
    recovered_from_terminal = False
    if not result_present:
        if not latest_generation_is_selected_terminal(
            checkpoint_store,
            admission=admission,
            learner=learner,
        ):
            return None, False
        recovered_result = recover_formal_result_from_selected_terminal(
            checkpoint_store,
            admission=admission,
            learner=learner,
        )
        recovered_result.validate_integrity()
        recovered_payload = recovered_result.to_dict()
        validated_recovered, _ = _validate_internal_result(
            recovered_payload,
            admission=admission,
            learner=learner,
        )
        _publish_no_overwrite(
            result_path,
            _canonical_bytes(validated_recovered),
            name=f"{learner} recovered internal result",
        )
        result_present = True
        recovered_from_terminal = True
    result, result_raw = _read_canonical_file(
        result_path,
        name=f"{learner} internal result",
    )
    validated_result, completion_admission = _validate_internal_result(
        result,
        admission=admission,
        learner=learner,
    )
    terminal_checkpoint_sha = _terminal_checkpoint_artifact_sha256(
        checkpoint_store,
        admission=completion_admission,
        learner=learner,
    )
    expected = _completion_receipt_payload(
        current_admission=admission,
        completion_admission=completion_admission,
        learner=learner,
        internal_result=validated_result,
        result_file_sha256=hashlib.sha256(result_raw).hexdigest(),
        result_file_size_bytes=len(result_raw),
        checkpoint_store=checkpoint_store,
        terminal_checkpoint_artifact_sha256=terminal_checkpoint_sha,
    )
    expected_with_hash = {
        **expected,
        "receipt_sha256": _canonical_sha256(expected),
    }
    recovered = recovered_from_terminal
    if receipt_present:
        receipt, _ = _read_canonical_file(
            receipt_path,
            name=f"{learner} completion receipt",
        )
        validated = _validate_completion_receipt(
            receipt,
            admission=admission,
            learner=learner,
            result=validated_result,
            result_raw=result_raw,
            checkpoint_store=checkpoint_store,
            completion_admission=completion_admission,
        )
    else:
        _publish_no_overwrite(
            receipt_path,
            _canonical_bytes(expected_with_hash),
            name=f"{learner} completion receipt",
        )
        validated = expected_with_hash
        recovered = True
    return validated, recovered


def _publish_head_completion(
    *,
    result: FormalR3PrimaryHeadResult,
    admission: PrimarySegmentAdmission,
    learner: PrimaryLearner,
    result_path: Path,
    receipt_path: Path,
    checkpoint_store: DurableCheckpointStore,
    execution_slice: PrimaryHeadExecutionSlice,
) -> dict[str, Any]:
    if type(result) is not FormalR3PrimaryHeadResult:
        raise TypeError("formal head runner returned an invalid result type")
    result.validate_integrity()
    if type(execution_slice) is not PrimaryHeadExecutionSlice:
        raise TypeError("head completion requires an exact execution slice")
    execution_slice.validate_integrity()
    if (
        result.admission is not admission
        or result.runtime.admission_sha256 != admission.admission_sha256
        or result.learner != learner
    ):
        raise ValueError("formal head result differs from the active admission/head")
    result_payload = result.to_dict()
    validated_result, completion_admission = _validate_internal_result(
        result_payload,
        admission=admission,
        learner=learner,
    )
    if completion_admission is not admission:
        raise ValueError("new formal head result must complete in the active segment")
    updates = _require_positive_integer(
        validated_result["controller_updates_executed"],
        name=f"{learner} completed updates",
    )
    cadence = admission.design.science.settings.convergence.check_interval
    if (
        execution_slice.head != learner
        or updates <= execution_slice.start_completed_updates
        or updates > execution_slice.end_completed_updates_inclusive
        or (updates - execution_slice.start_completed_updates) % cadence
        or validated_result["head_execution_slice_sha256"] != execution_slice.slice_sha256
        or validated_result["resumed_from_predecessor"]
        is not (execution_slice.fresh_or_resume == "resume")
    ):
        raise ValueError("formal head result violates its Gate-P execution slice")
    result_raw = _canonical_bytes(validated_result)
    if result_path.exists() or result_path.is_symlink():
        existing, existing_raw = _read_canonical_file(
            result_path,
            name=f"{learner} internal result",
        )
        if existing != validated_result or existing_raw != result_raw:
            raise FileExistsError(f"refusing to replace a different {learner} result")
    else:
        _publish_no_overwrite(
            result_path,
            result_raw,
            name=f"{learner} internal result",
        )
    terminal_checkpoint_sha = _terminal_checkpoint_artifact_sha256(
        checkpoint_store,
        admission=admission,
        learner=learner,
    )
    if terminal_checkpoint_sha != validated_result["terminal_checkpoint_artifact_sha256"]:
        raise ValueError("formal result terminal checkpoint reference is inconsistent")
    receipt_payload = _completion_receipt_payload(
        current_admission=admission,
        completion_admission=admission,
        learner=learner,
        internal_result=validated_result,
        result_file_sha256=hashlib.sha256(result_raw).hexdigest(),
        result_file_size_bytes=len(result_raw),
        checkpoint_store=checkpoint_store,
        terminal_checkpoint_artifact_sha256=terminal_checkpoint_sha,
    )
    receipt = {
        **receipt_payload,
        "receipt_sha256": _canonical_sha256(receipt_payload),
    }
    if receipt_path.exists() or receipt_path.is_symlink():
        existing, existing_raw = _read_canonical_file(
            receipt_path,
            name=f"{learner} completion receipt",
        )
        if existing != receipt or existing_raw != _canonical_bytes(receipt):
            raise FileExistsError(f"refusing to replace a different {learner} completion receipt")
    else:
        _publish_no_overwrite(
            receipt_path,
            _canonical_bytes(receipt),
            name=f"{learner} completion receipt",
        )
    return receipt


def _checkpoint_position(
    *,
    store: DurableCheckpointStore,
    admission: PrimarySegmentAdmission,
    learner: PrimaryLearner,
    allow_later_segment_fresh: bool,
) -> tuple[str, ArtifactRef | None]:
    audited = store.audit_generations(verify_all_checkpoint_bytes=True)
    if not audited:
        if admission.segment_index > 1 and not allow_later_segment_fresh:
            raise RuntimeError(
                "continuation admission has no predecessor checkpoint for active head"
            )
        return "fresh", None
    current_error: BaseException | None = None
    try:
        current_ref = continuation_checkpoint_artifact_ref(
            store,
            predecessor=admission,
            learner=learner,
        )
    except (TypeError, ValueError, RuntimeError, OSError) as error:
        current_error = error
    else:
        return "current_segment_checkpoint", current_ref
    if admission.segment_index == 1 or admission.continuation_evidence is None:
        raise RuntimeError(
            "fresh segment checkpoint does not belong to the active admission"
        ) from current_error
    predecessor = admission.continuation_evidence.predecessor
    try:
        predecessor_ref = continuation_checkpoint_artifact_ref(
            store,
            predecessor=predecessor,
            learner=learner,
        )
    except (TypeError, ValueError, RuntimeError, OSError) as error:
        raise RuntimeError(
            "checkpoint store is neither current nor admitted predecessor state"
        ) from error
    if predecessor_ref != admission.continuation_evidence.verified_checkpoint:
        raise ValueError("active head checkpoint differs from continuation capability")
    return "admitted_predecessor_checkpoint", predecessor_ref


class _TaskJournal:
    def __init__(
        self,
        root: Path,
        *,
        admission: PrimarySegmentAdmission,
        runtime: SlurmSegmentRuntime,
        operational_policy: VerifiedGatePOperationalBundle,
        window: _NominalSegmentWindow,
    ) -> None:
        self.root = _ensure_child_directory(root, _TASK_JOURNAL_DIRECTORY)
        self.admission = admission
        self.runtime = runtime
        self.operational_policy = operational_policy
        self.window = window

    def audit(self) -> tuple[dict[str, Any], ...]:
        indexed: list[tuple[int, Path]] = []
        for path in self.root.iterdir():
            match = _JOURNAL_FILE_RE.fullmatch(path.name)
            if match is None:
                raise ValueError(f"unexpected task-journal entry: {path.name}")
            indexed.append((int(match.group(1)), path))
        indexed.sort()
        if [index for index, _ in indexed] != list(range(1, len(indexed) + 1)):
            raise ValueError("task-journal sequence is not contiguous")
        chain = _admission_chain(self.admission)
        previous: str | None = None
        result: list[dict[str, Any]] = []
        for sequence, path in indexed:
            value, _ = _read_canonical_file(path, name="task-journal event")
            event = _require_exact_keys(
                value,
                name="task-journal event",
                keys={
                    "schema_version",
                    "design_sha256",
                    "logical_run_id",
                    "task_id",
                    "seed",
                    "sequence",
                    "previous_event_sha256",
                    "segment_index",
                    "admission_sha256",
                    "scheduler_segment_id",
                    "runtime_sha256",
                    "gate_p_resource_plan_sha256",
                    "nominal_segment_start_cursor_sha256",
                    "nominal_segment_end_cursor_sha256",
                    "max_safe_update_blocks_to_execute",
                    "safe_update_blocks_consumed",
                    "actual_cursor",
                    "actual_cursor_sha256",
                    "head_execution_slice_sha256",
                    "event_type",
                    "learner",
                    "evidence_sha256",
                    "event_sha256",
                },
            )
            segment_index = _require_positive_integer(
                event["segment_index"],
                name="journal segment_index",
            )
            segment = chain.get(segment_index)
            if segment is None:
                raise ValueError("task-journal event belongs to an unadmitted segment")
            event_window = _nominal_segment_window(
                self.operational_policy,
                admission=segment,
            )
            start_cursor = _plan_cursor_payload(event_window.start_cursor)
            end_cursor = _plan_cursor_payload(event_window.end_cursor)
            exact = {
                "schema_version": TASK_JOURNAL_EVENT_SCHEMA,
                "design_sha256": self.admission.design.design_sha256,
                "logical_run_id": self.admission.logical_run_id,
                "task_id": self.admission.task_id,
                "seed": self.admission.seed,
                "sequence": sequence,
                "previous_event_sha256": previous,
                "segment_index": segment_index,
                "admission_sha256": segment.admission_sha256,
                "scheduler_segment_id": segment.scheduler_segment_id,
                "gate_p_resource_plan_sha256": (self.operational_policy.resource_plan_sha256),
                "nominal_segment_start_cursor_sha256": (_canonical_sha256(start_cursor)),
                "nominal_segment_end_cursor_sha256": (_canonical_sha256(end_cursor)),
                "max_safe_update_blocks_to_execute": (
                    event_window.max_safe_update_blocks_to_execute
                ),
            }
            if any(event.get(name) != expected for name, expected in exact.items()):
                raise ValueError("task-journal event identity/hash chain is invalid")
            _require_digest(event["runtime_sha256"], name="journal runtime_sha256")
            consumed = _require_nonnegative_integer(
                event["safe_update_blocks_consumed"],
                name="journal consumed block budget",
            )
            if consumed > event_window.max_safe_update_blocks_to_execute:
                raise ValueError("journal event exceeds the Gate-P block budget")
            actual_cursor = _validate_actual_cursor_value(
                event["actual_cursor"],
                name="journal actual cursor",
                maximum_updates_per_head=(segment.design.science.settings.convergence.max_steps),
                audit_cadence_updates=(segment.design.science.settings.convergence.check_interval),
            )
            actual_cursor_sha = _require_digest(
                event["actual_cursor_sha256"],
                name="journal actual cursor SHA",
            )
            if actual_cursor_sha != _canonical_sha256(_actual_cursor_payload(actual_cursor)):
                raise ValueError("journal actual cursor hash is invalid")
            if event["head_execution_slice_sha256"] is not None:
                _require_digest(
                    event["head_execution_slice_sha256"],
                    name="journal head execution slice SHA",
                )
            if event["learner"] is not None:
                _learner(event["learner"])
            if event["evidence_sha256"] is not None:
                _require_digest(
                    event["evidence_sha256"],
                    name="journal evidence_sha256",
                )
            event_sha = _require_digest(
                event["event_sha256"],
                name="journal event_sha256",
            )
            unsigned = dict(event)
            del unsigned["event_sha256"]
            if event_sha != _canonical_sha256(unsigned):
                raise ValueError("task-journal event self-hash is invalid")
            previous = event_sha
            result.append(event)
        return tuple(result)

    def append(
        self,
        *,
        event_type: str,
        learner: PrimaryLearner | None,
        evidence_sha256: str | None,
        actual_cursor: _ActualCursor,
        safe_update_blocks_consumed: int,
        execution_slice: PrimaryHeadExecutionSlice | None = None,
    ) -> dict[str, Any]:
        if type(event_type) is not str or not event_type:
            raise ValueError("journal event_type must be non-empty")
        method = None if learner is None else _learner(learner)
        if evidence_sha256 is not None:
            _require_digest(evidence_sha256, name="journal evidence_sha256")
        cursor_payload = _actual_cursor_payload(actual_cursor)
        _validate_actual_cursor_value(
            cursor_payload,
            name="journal actual cursor",
            maximum_updates_per_head=(self.admission.design.science.settings.convergence.max_steps),
            audit_cadence_updates=(
                self.admission.design.science.settings.convergence.check_interval
            ),
        )
        consumed = _require_nonnegative_integer(
            safe_update_blocks_consumed,
            name="journal consumed block budget",
        )
        if consumed > self.window.max_safe_update_blocks_to_execute:
            raise ValueError("journal event exceeds the Gate-P block budget")
        if execution_slice is not None:
            if type(execution_slice) is not PrimaryHeadExecutionSlice:
                raise TypeError("journal execution_slice type is invalid")
            execution_slice.validate_integrity()
            if method != execution_slice.head:
                raise ValueError("journal learner differs from its execution slice")
        existing = self.audit()
        sequence = len(existing) + 1
        previous = None if not existing else existing[-1]["event_sha256"]
        payload: dict[str, object] = {
            "schema_version": TASK_JOURNAL_EVENT_SCHEMA,
            "design_sha256": self.admission.design.design_sha256,
            "logical_run_id": self.admission.logical_run_id,
            "task_id": self.admission.task_id,
            "seed": self.admission.seed,
            "sequence": sequence,
            "previous_event_sha256": previous,
            "segment_index": self.admission.segment_index,
            "admission_sha256": self.admission.admission_sha256,
            "scheduler_segment_id": self.admission.scheduler_segment_id,
            "runtime_sha256": self.runtime.runtime_sha256,
            "gate_p_resource_plan_sha256": (self.operational_policy.resource_plan_sha256),
            "nominal_segment_start_cursor_sha256": _canonical_sha256(
                _plan_cursor_payload(self.window.start_cursor)
            ),
            "nominal_segment_end_cursor_sha256": _canonical_sha256(
                _plan_cursor_payload(self.window.end_cursor)
            ),
            "max_safe_update_blocks_to_execute": (self.window.max_safe_update_blocks_to_execute),
            "safe_update_blocks_consumed": consumed,
            "actual_cursor": cursor_payload,
            "actual_cursor_sha256": _canonical_sha256(cursor_payload),
            "head_execution_slice_sha256": (
                None if execution_slice is None else execution_slice.slice_sha256
            ),
            "event_type": event_type,
            "learner": method,
            "evidence_sha256": evidence_sha256,
        }
        event = {**payload, "event_sha256": _canonical_sha256(payload)}
        path = self.root / f"event-{sequence:08d}.json"
        _publish_no_overwrite(
            path,
            _canonical_bytes(event),
            name="task-journal event",
        )
        return event

    def unresolved_start_event(
        self,
        learner: PrimaryLearner,
    ) -> Mapping[str, object] | None:
        method = _learner(learner)
        unresolved: Mapping[str, object] | None = None
        terminal_events = {
            "head_completed",
            "head_completion_recovered_after_crash",
            "continuation_required",
        }
        for event in self.audit():
            if event["segment_index"] != self.admission.segment_index or event["learner"] != method:
                continue
            if event["event_type"] == "head_started":
                unresolved = event
            elif event["event_type"] in terminal_events:
                unresolved = None
        return unresolved

    def has_unresolved_start(self, learner: PrimaryLearner) -> bool:
        return self.unresolved_start_event(learner) is not None


def _completion_summary(
    admission: PrimarySegmentAdmission,
    receipts: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for index, learner in enumerate(R3_PRIMARY_HEADS):
        receipt = receipts.get(learner)
        if receipt is None:
            continue
        result.append(
            {
                "learner": learner,
                "head_run_id": admission.head_run_ids[index],
                "completion_receipt_sha256": receipt["receipt_sha256"],
            }
        )
    return result


def _outcome_payload(
    *,
    admission: PrimarySegmentAdmission,
    runtime: SlurmSegmentRuntime,
    operational_policy: VerifiedGatePOperationalBundle,
    status: OutcomeStatus,
    receipts: Mapping[str, Mapping[str, object]],
    active_learner: PrimaryLearner | None,
    continuation_checkpoint: ArtifactRef | None,
    continuation_reason: str | None,
) -> dict[str, object]:
    complete = status == "compute_complete_pending_external_scheduler_terminal"
    return {
        "schema_version": SEGMENT_OUTCOME_SCHEMA,
        "status": status,
        "design_sha256": admission.design.design_sha256,
        "admission_sha256": admission.admission_sha256,
        "logical_run_id": admission.logical_run_id,
        "scheduler_segment_id": admission.scheduler_segment_id,
        "runtime_sha256": runtime.runtime_sha256,
        "segment_index": admission.segment_index,
        "task_id": admission.task_id,
        "seed": admission.seed,
        "gate_p_resource_plan_sha256": operational_policy.resource_plan_sha256,
        "completed_heads": _completion_summary(admission, receipts),
        "active_learner": active_learner,
        "continuation_checkpoint": (
            None
            if continuation_checkpoint is None
            else _checkpoint_ref_dict(continuation_checkpoint)
        ),
        "continuation_reason": continuation_reason,
        "all_primary_heads_compute_complete": complete,
        "external_scheduler_terminal_required": True,
        "external_scheduler_success_claimed": False,
        "r3_success_authorization_created": False,
        "information_boundary": "train_only_head_free_segment_outcome",
    }


@dataclass(frozen=True, slots=True)
class R3PrimarySegmentOutcome:
    """Published head-free segment outcome; never an external terminal receipt."""

    status: OutcomeStatus
    outcome_sha256: str
    file_sha256: str
    artifact_path: Path
    continuation_checkpoint: ArtifactRef | None
    _factory_token: InitVar[object] = None
    _seal: object = field(repr=False, compare=False, init=False)

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _FACTORY_TOKEN:
            raise TypeError("segment outcome must be loaded from durable evidence")
        object.__setattr__(self, "_seal", _FACTORY_TOKEN)
        self._validate_structure()

    def _validate_structure(self) -> None:
        if self.status not in {
            "continuation_required_after_safe_checkpoint",
            "compute_complete_pending_external_scheduler_terminal",
        }:
            raise ValueError("segment outcome status is invalid")
        _require_digest(self.outcome_sha256, name="outcome_sha256")
        _require_digest(self.file_sha256, name="outcome file SHA256")
        _require_real_file(self.artifact_path, name="segment outcome artifact")
        if self.continuation_checkpoint is not None:
            self.continuation_checkpoint.validate_integrity()
        if (self.status == "continuation_required_after_safe_checkpoint") is (
            self.continuation_checkpoint is None
        ):
            raise ValueError("continuation outcome/ref presence is inconsistent")

    @property
    def external_scheduler_success_claimed(self) -> bool:
        return False

    def validate_integrity(self) -> None:
        if getattr(self, "_seal", None) is not _FACTORY_TOKEN:
            raise TypeError("segment outcome lacks its durable-evidence seal")
        self._validate_structure()


def _outcome_path(task_root: Path, segment_index: int) -> Path:
    directory = _ensure_child_directory(task_root, _SEGMENT_OUTCOME_DIRECTORY)
    return directory / f"segment-{segment_index:04d}.json"


def _materialize_outcome(
    *,
    path: Path,
    admission: PrimarySegmentAdmission,
    runtime: SlurmSegmentRuntime,
    operational_policy: VerifiedGatePOperationalBundle,
    status: OutcomeStatus,
    receipts: Mapping[str, Mapping[str, object]],
    active_learner: PrimaryLearner | None,
    continuation_checkpoint: ArtifactRef | None,
    continuation_reason: str | None,
) -> R3PrimarySegmentOutcome:
    payload = _outcome_payload(
        admission=admission,
        runtime=runtime,
        operational_policy=operational_policy,
        status=status,
        receipts=receipts,
        active_learner=active_learner,
        continuation_checkpoint=continuation_checkpoint,
        continuation_reason=continuation_reason,
    )
    value = {
        **payload,
        "outcome_sha256": primary_outcome_semantic_sha256(payload),
    }
    raw = _canonical_bytes(value)
    if path.exists() or path.is_symlink():
        existing, existing_raw = _read_canonical_file(path, name="segment outcome")
        if existing != value or existing_raw != raw:
            raise FileExistsError("refusing to replace a different segment outcome")
    else:
        _publish_no_overwrite(path, raw, name="segment outcome")
    result = R3PrimarySegmentOutcome(
        status=status,
        outcome_sha256=value["outcome_sha256"],
        file_sha256=hashlib.sha256(raw).hexdigest(),
        artifact_path=path,
        continuation_checkpoint=continuation_checkpoint,
        _factory_token=_FACTORY_TOKEN,
    )
    result.validate_integrity()
    return result


def _load_existing_outcome(
    *,
    path: Path,
    admission: PrimarySegmentAdmission,
    runtime: SlurmSegmentRuntime,
    operational_policy: VerifiedGatePOperationalBundle,
    receipts: Mapping[str, Mapping[str, object]],
    stores: Mapping[str, DurableCheckpointStore],
) -> R3PrimarySegmentOutcome | None:
    if not (path.exists() or path.is_symlink()):
        return None
    value, raw = _read_canonical_file(path, name="segment outcome")
    _require_exact_keys(
        value,
        name="segment outcome",
        keys={
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
        },
    )
    status = value["status"]
    if status not in {
        "continuation_required_after_safe_checkpoint",
        "compute_complete_pending_external_scheduler_terminal",
    }:
        raise ValueError("segment outcome status is invalid")
    unsigned = dict(value)
    outcome_sha = unsigned.pop("outcome_sha256")
    if outcome_sha != primary_outcome_semantic_sha256(unsigned):
        raise ValueError("segment outcome self-hash is invalid")
    active = value["active_learner"]
    continuation: ArtifactRef | None
    if status == "continuation_required_after_safe_checkpoint":
        method = _learner(active)
        continuation = continuation_checkpoint_artifact_ref(
            stores[method],
            predecessor=admission,
            learner=method,
        )
        expected_reason = value["continuation_reason"]
        if type(expected_reason) is not str or not expected_reason:
            raise ValueError("continuation outcome lacks its reason")
    else:
        method = None
        continuation = None
        expected_reason = None
        if set(receipts) != set(R3_PRIMARY_HEADS):
            raise ValueError("complete segment outcome lacks both head receipts")
    expected_payload = _outcome_payload(
        admission=admission,
        runtime=runtime,
        operational_policy=operational_policy,
        status=status,
        receipts=receipts,
        active_learner=method,
        continuation_checkpoint=continuation,
        continuation_reason=expected_reason,
    )
    if unsigned != expected_payload:
        raise ValueError("segment outcome differs from current durable evidence")
    result = R3PrimarySegmentOutcome(
        status=status,
        outcome_sha256=outcome_sha,
        file_sha256=hashlib.sha256(raw).hexdigest(),
        artifact_path=path,
        continuation_checkpoint=continuation,
        _factory_token=_FACTORY_TOKEN,
    )
    result.validate_integrity()
    return result


def run_r3_primary_task_segment(
    admission: PrimarySegmentAdmission,
    *,
    runtime: SlurmSegmentRuntime,
    task_root: str | os.PathLike[str],
    checkpoint_signal: CheckpointSignal,
    operational_policy: VerifiedGatePOperationalBundle,
) -> R3PrimarySegmentOutcome:
    """Run one admitted scheduler segment in fixed BT -> ProRM+ order.

    The return value only describes durable compute state.  External scheduler
    terminal evidence is always required before continuation or success
    authorization can be considered by a later layer.
    """

    admitted = _require_admission(admission)
    active_runtime = _require_runtime(runtime, admission=admitted)
    if type(checkpoint_signal) is not CheckpointSignal:
        raise TypeError("checkpoint_signal must be an exact CheckpointSignal")
    operational_policy = _validate_operational_policy(
        operational_policy,
        admission=admitted,
        runtime=active_runtime,
    )
    window = _nominal_segment_window(
        operational_policy,
        admission=admitted,
    )
    root = Path(task_root)
    if not root.is_absolute() or root.resolve(strict=True) != root:
        raise ValueError("task_root must be an existing canonical absolute directory")
    _require_real_directory(root, name="R3 task root")
    journal = _TaskJournal(
        root,
        admission=admitted,
        runtime=active_runtime,
        operational_policy=operational_policy,
        window=window,
    )
    journal.audit()

    receipts: dict[str, Mapping[str, object]] = {}
    stores: dict[str, DurableCheckpointStore] = {}
    paths: dict[str, tuple[Path, Path]] = {}
    recovered: set[str] = set()
    for learner_name in R3_PRIMARY_HEADS:
        learner = _learner(learner_name)
        head_root, result_path, receipt_path = _head_paths(root, learner)
        store = _checkpoint_store(
            head_root,
            admission=admitted,
            learner=learner,
        )
        stores[learner] = store
        paths[learner] = (result_path, receipt_path)
        receipt, was_recovered = _load_or_recover_completion(
            admission=admitted,
            learner=learner,
            result_path=result_path,
            receipt_path=receipt_path,
            checkpoint_store=store,
        )
        if receipt is not None:
            receipts[learner] = receipt
        if was_recovered:
            recovered.add(learner)
    cursor = _actual_cursor(
        admission=admitted,
        receipts=receipts,
        stores=stores,
    )
    for learner_name in R3_PRIMARY_HEADS:
        learner = _learner(learner_name)
        if learner in recovered:
            journal.append(
                event_type="head_completion_recovered_after_crash",
                learner=learner,
                evidence_sha256=receipts[learner]["receipt_sha256"],
                actual_cursor=cursor,
                safe_update_blocks_consumed=0,
            )

    outcome_path = _outcome_path(root, admitted.segment_index)
    existing_outcome = _load_existing_outcome(
        path=outcome_path,
        admission=admitted,
        runtime=active_runtime,
        operational_policy=operational_policy,
        receipts=receipts,
        stores=stores,
    )
    if existing_outcome is not None:
        return existing_outcome

    consumed_blocks = 0
    for learner_name in R3_PRIMARY_HEADS:
        learner = _learner(learner_name)
        cursor = _actual_cursor(
            admission=admitted,
            receipts=receipts,
            stores=stores,
        )
        if learner in receipts:
            if learner not in recovered:
                journal.append(
                    event_type="head_completion_revalidated_and_skipped",
                    learner=learner,
                    evidence_sha256=receipts[learner]["receipt_sha256"],
                    actual_cursor=cursor,
                    safe_update_blocks_consumed=consumed_blocks,
                )
            continue
        if cursor[4] != learner:
            raise ValueError("durable actual cursor differs from strict learner order")
        store = stores[learner]
        checkpoint_position, checkpoint_ref = _checkpoint_position(
            store=store,
            admission=admitted,
            learner=learner,
            allow_later_segment_fresh=(
                admission.segment_index > 1
                and (cursor[0] if learner == "bt_mle" else cursor[2]) == 0
            ),
        )
        if checkpoint_position == "current_segment_checkpoint":
            if checkpoint_ref is None:
                raise RuntimeError("current-segment checkpoint lacks its verified ref")
            unresolved = journal.unresolved_start_event(learner)
            if unresolved is None:
                raise RuntimeError("current-segment checkpoint lacks its durable head-start cursor")
            start_cursor = _validate_actual_cursor_value(
                unresolved["actual_cursor"],
                name="unresolved head-start actual cursor",
                maximum_updates_per_head=(admitted.design.science.settings.convergence.max_steps),
                audit_cadence_updates=(admitted.design.science.settings.convergence.check_interval),
            )
            start_updates = start_cursor[0] if learner == "bt_mle" else start_cursor[2]
            current_updates = cursor[0] if learner == "bt_mle" else cursor[2]
            cadence = admitted.design.science.settings.convergence.check_interval
            delta = current_updates - start_updates
            if delta < 0 or delta % cadence:
                raise ValueError("current checkpoint cursor regresses its durable head start")
            consumed_blocks = (
                _require_nonnegative_integer(
                    unresolved["safe_update_blocks_consumed"],
                    name="unresolved head-start consumed budget",
                )
                + delta // cadence
            )
            if consumed_blocks > window.max_safe_update_blocks_to_execute:
                raise ValueError("current checkpoint exceeds the Gate-P segment block budget")
            journal.append(
                event_type="checkpoint_discovered_after_process_crash",
                learner=learner,
                evidence_sha256=checkpoint_ref.artifact_sha256,
                actual_cursor=cursor,
                safe_update_blocks_consumed=consumed_blocks,
            )
            outcome = _materialize_outcome(
                path=outcome_path,
                admission=admitted,
                runtime=active_runtime,
                operational_policy=operational_policy,
                status="continuation_required_after_safe_checkpoint",
                receipts=receipts,
                active_learner=learner,
                continuation_checkpoint=checkpoint_ref,
                continuation_reason="checkpoint_discovered_after_process_crash",
            )
            journal.append(
                event_type="continuation_required",
                learner=learner,
                evidence_sha256=outcome.outcome_sha256,
                actual_cursor=cursor,
                safe_update_blocks_consumed=consumed_blocks,
            )
            return outcome
        if journal.has_unresolved_start(learner):
            raise RuntimeError("prior head attempt ended before any recoverable safe checkpoint")
        execution_slice = _make_head_execution_slice(
            plan=operational_policy,
            admission=admitted,
            runtime=active_runtime,
            window=window,
            actual_cursor=cursor,
            learner=learner,
            mode=("fresh" if checkpoint_position == "fresh" else "resume"),
            consumed_blocks=consumed_blocks,
            predecessor_checkpoint=(None if checkpoint_position == "fresh" else checkpoint_ref),
        )
        journal.append(
            event_type="head_started",
            learner=learner,
            evidence_sha256=execution_slice.slice_sha256,
            actual_cursor=cursor,
            safe_update_blocks_consumed=consumed_blocks,
            execution_slice=execution_slice,
        )
        result_path, receipt_path = paths[learner]
        try:
            formal_result = run_formal_r3_primary_head_segment(
                admitted,
                learner,
                runtime=active_runtime,
                resource_plan=operational_policy,
                checkpoint_store=store,
                checkpoint_signal=checkpoint_signal,
                head_execution_slice=execution_slice.to_dict(),
            )
        except CheckpointInterruption as interruption:
            checkpoint_ref = continuation_checkpoint_artifact_ref(
                store,
                predecessor=admitted,
                learner=learner,
            )
            interruption_cursor = _actual_cursor(
                admission=admitted,
                receipts=receipts,
                stores=stores,
            )
            interruption_consumed = _blocks_consumed_through_cursor(
                execution_slice,
                interruption_cursor,
                audit_cadence_updates=(admitted.design.science.settings.convergence.check_interval),
            )
            outcome = _materialize_outcome(
                path=outcome_path,
                admission=admitted,
                runtime=active_runtime,
                operational_policy=operational_policy,
                status="continuation_required_after_safe_checkpoint",
                receipts=receipts,
                active_learner=learner,
                continuation_checkpoint=checkpoint_ref,
                continuation_reason=(
                    "formal_head_runner_planned_segment_boundary"
                    if isinstance(interruption, PlannedSegmentBoundary)
                    else "formal_head_runner_checkpoint_interruption"
                ),
            )
            journal.append(
                event_type="continuation_required",
                learner=learner,
                evidence_sha256=outcome.outcome_sha256,
                actual_cursor=interruption_cursor,
                safe_update_blocks_consumed=interruption_consumed,
                execution_slice=execution_slice,
            )
            return outcome
        receipt = _publish_head_completion(
            result=formal_result,
            admission=admitted,
            learner=learner,
            result_path=result_path,
            receipt_path=receipt_path,
            checkpoint_store=store,
            execution_slice=execution_slice,
        )
        receipts[learner] = receipt
        cursor = _actual_cursor(
            admission=admitted,
            receipts=receipts,
            stores=stores,
        )
        consumed_blocks = _blocks_consumed_through_cursor(
            execution_slice,
            cursor,
            audit_cadence_updates=(admitted.design.science.settings.convergence.check_interval),
        )
        journal.append(
            event_type="head_completed",
            learner=learner,
            evidence_sha256=receipt["receipt_sha256"],
            actual_cursor=cursor,
            safe_update_blocks_consumed=consumed_blocks,
            execution_slice=execution_slice,
        )
        # Do not stop merely because a signal was latched at the exact boundary
        # between heads.  The next head must enter its formal runner so that the
        # signal is converted at its first safe update boundary into a durable,
        # head-specific continuation checkpoint.

    if set(receipts) != set(R3_PRIMARY_HEADS):
        raise RuntimeError("segment loop ended without both completion receipts")
    outcome = _materialize_outcome(
        path=outcome_path,
        admission=admitted,
        runtime=active_runtime,
        operational_policy=operational_policy,
        status="compute_complete_pending_external_scheduler_terminal",
        receipts=receipts,
        active_learner=None,
        continuation_checkpoint=None,
        continuation_reason=None,
    )
    journal.append(
        event_type="segment_compute_complete",
        learner=None,
        evidence_sha256=outcome.outcome_sha256,
        actual_cursor=_actual_cursor(
            admission=admitted,
            receipts=receipts,
            stores=stores,
        ),
        safe_update_blocks_consumed=consumed_blocks,
    )
    return outcome


__all__ = [
    "HEAD_EXECUTION_SLICE_SCHEMA",
    "HEAD_COMPLETION_RECEIPT_SCHEMA",
    "PrimaryHeadExecutionSlice",
    "R3PrimarySegmentOutcome",
    "SEGMENT_OUTCOME_SCHEMA",
    "TASK_JOURNAL_EVENT_SCHEMA",
    "primary_outcome_semantic_sha256",
    "run_r3_primary_task_segment",
]
