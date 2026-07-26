"""File-level R3 authorization bridge from Gate R plus Gate C to Gate F.

The combined authorization is useful only when its canonical bytes live at the
fixed production path and both source artifacts can be reopened at their fixed
paths under the exact byte hashes carried by the authorization.  Gate R is
deeply revalidated from its retained scheduler-segment evidence; Gate C is
reopened as the exact self-hashed 3x3 aggregate.  No recovery/control output is
returned as a training input.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

from .phase2_r3_artifacts import (
    CanonicalJsonArtifact,
    publish_canonical_artifact,
    read_canonical_artifact,
)
from .phase2_r3_controls_hpc4 import (
    build_controls_authorization,
    validate_controls_aggregate_structure,
    validate_controls_authorization_structure,
)
from .phase2_r3_post_recovery_contract import (
    R3_FINAL_AUTHORIZATION_RELATIVE,
    R3_GATE_C_AGGREGATE_RELATIVE,
    R3_GATE_R_AUTHORIZATION_RELATIVE,
    R3_PRODUCTION_PROJECT_ROOT,
)


def _project_root(project_root: str | os.PathLike[str] | None) -> Path:
    root = Path(R3_PRODUCTION_PROJECT_ROOT if project_root is None else project_root)
    if not root.is_absolute():
        raise ValueError("R3 final authorization project root must be absolute")
    try:
        metadata = root.lstat()
    except OSError as error:
        raise ValueError("R3 final authorization project root is unavailable") from error
    if not stat.S_ISDIR(metadata.st_mode) or root.is_symlink() or root.resolve(strict=True) != root:
        raise ValueError("R3 final authorization project root must be canonical")
    return root


def _exact_path(
    value: str | os.PathLike[str],
    *,
    root: Path,
    relative: Path,
    name: str,
) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = path.absolute()
    expected = root / relative
    if path != expected:
        raise ValueError(f"{name} is not at its exact project path")
    parent = path.parent
    metadata = parent.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or parent.is_symlink()
        or parent.resolve(strict=True) != parent
    ):
        raise ValueError(f"{name} parent is not canonical")
    return path


def _verified_sources(
    *,
    root: Path,
    gate_r_authorization_file_sha256: str,
    gate_c_aggregate_file_sha256: str,
) -> tuple[dict[str, object], dict[str, object], CanonicalJsonArtifact]:
    # Gate-R revalidation imports the scientific R3 configuration/training
    # types.  Keep it lazy so host-Python control-plane commands which do not
    # authorize Gate F remain stdlib-only.
    from .phase2_r3_authorization import (
        R3_SUCCESS_AUTHORIZATION_RELATIVE,
        verify_r3_success_authorization,
    )

    gate_r_path = root / R3_GATE_R_AUTHORIZATION_RELATIVE
    if R3_GATE_R_AUTHORIZATION_RELATIVE != R3_SUCCESS_AUTHORIZATION_RELATIVE:
        raise RuntimeError("R3 Gate-R production path contracts diverged")
    gate_r = verify_r3_success_authorization(
        gate_r_path,
        expected_sha256=gate_r_authorization_file_sha256,
        project_root=root,
    )
    gate_c_path = root / R3_GATE_C_AGGREGATE_RELATIVE
    gate_c_transport = read_canonical_artifact(
        gate_c_path,
        expected_file_sha256=gate_c_aggregate_file_sha256,
    )
    gate_c = validate_controls_aggregate_structure(gate_c_transport.payload)
    gate_c_transport.validate_integrity()
    return gate_r, gate_c, gate_c_transport


def verify_r3_final_authorization(
    path: str | os.PathLike[str],
    *,
    expected_sha256: str,
    project_root: str | os.PathLike[str] | None = None,
) -> dict[str, object]:
    """Reopen the final R3 file and both exact source artifacts."""

    root = _project_root(project_root)
    source = _exact_path(
        path,
        root=root,
        relative=R3_FINAL_AUTHORIZATION_RELATIVE,
        name="R3 final authorization",
    )
    transport = read_canonical_artifact(
        source,
        expected_file_sha256=expected_sha256,
    )
    payload = transport.payload
    gate_r_file_sha256 = payload.get("gate_r_authorization_file_sha256")
    gate_c_file_sha256 = payload.get("gate_c_aggregate_file_sha256")
    if not isinstance(gate_r_file_sha256, str) or not isinstance(
        gate_c_file_sha256,
        str,
    ):
        raise ValueError("R3 final authorization lacks exact source byte hashes")
    gate_r, gate_c, gate_c_transport = _verified_sources(
        root=root,
        gate_r_authorization_file_sha256=gate_r_file_sha256,
        gate_c_aggregate_file_sha256=gate_c_file_sha256,
    )
    validated = validate_controls_authorization_structure(
        payload,
        aggregate=gate_c,
        gate_r_authorization=gate_r,
        gate_r_authorization_file_sha256=gate_r_file_sha256,
    )
    if (
        validated["gate_r_authorization_path"] != R3_GATE_R_AUTHORIZATION_RELATIVE.as_posix()
        or validated["gate_c_aggregate_path"] != R3_GATE_C_AGGREGATE_RELATIVE.as_posix()
        or validated["gate_c_aggregate_file_sha256"] != gate_c_transport.file_sha256
    ):
        raise ValueError("R3 final authorization source-path binding is invalid")
    gate_c_transport.validate_integrity()
    transport.validate_integrity()
    return validated


def publish_r3_final_authorization(
    *,
    gate_r_authorization: str | os.PathLike[str],
    gate_r_authorization_file_sha256: str,
    gate_c_aggregate: str | os.PathLike[str],
    gate_c_aggregate_file_sha256: str,
    output: str | os.PathLike[str],
    project_root: str | os.PathLike[str] | None = None,
) -> CanonicalJsonArtifact:
    """Publish the sole fixed-path R3 final authorization without overwrite."""

    root = _project_root(project_root)
    _exact_path(
        gate_r_authorization,
        root=root,
        relative=R3_GATE_R_AUTHORIZATION_RELATIVE,
        name="Gate-R authorization",
    )
    gate_c_path = _exact_path(
        gate_c_aggregate,
        root=root,
        relative=R3_GATE_C_AGGREGATE_RELATIVE,
        name="Gate-C aggregate",
    )
    destination = _exact_path(
        output,
        root=root,
        relative=R3_FINAL_AUTHORIZATION_RELATIVE,
        name="R3 final authorization output",
    )
    gate_r, gate_c, gate_c_transport = _verified_sources(
        root=root,
        gate_r_authorization_file_sha256=gate_r_authorization_file_sha256,
        gate_c_aggregate_file_sha256=gate_c_aggregate_file_sha256,
    )
    if gate_c_transport.artifact_path != gate_c_path:
        raise ValueError("Gate-C aggregate path changed during verification")
    payload = build_controls_authorization(
        gate_c,
        gate_r_authorization=gate_r,
        gate_r_authorization_file_sha256=gate_r_authorization_file_sha256,
    )
    artifact = publish_canonical_artifact(destination, payload)
    verified = verify_r3_final_authorization(
        destination,
        expected_sha256=artifact.file_sha256,
        project_root=root,
    )
    if verified != payload:
        raise ValueError("published R3 final authorization failed exact round-trip")
    return artifact


__all__ = [
    "publish_r3_final_authorization",
    "verify_r3_final_authorization",
]
