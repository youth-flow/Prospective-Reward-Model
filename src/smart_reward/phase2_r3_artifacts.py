"""Claim-free canonical JSON transport for Phase-2 recovery R3 artifacts.

This module proves only byte identity and safe local publication.  It does not
interpret ``schema_version`` or ``role`` values and therefore cannot authorize
an artifact for any gate.  Callers must pass the decoded mapping through the
corresponding identity rehydration or domain validator.
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    """Return the sole accepted UTF-8 representation of one mapping."""

    if not isinstance(value, Mapping):
        raise TypeError("canonical artifact payload must be a mapping")
    try:
        encoded = (
            json.dumps(
                dict(value),
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("canonical artifact payload is not strict JSON") from error
    # Round-tripping also rejects mapping implementations that serialize to a
    # non-object or contain values with a lossy custom representation.
    decoded = decode_canonical_json_bytes(encoded)
    if encoded != (
        json.dumps(
            decoded,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8"):
        raise ValueError("canonical artifact payload is not stable under round-trip")
    return encoded


def decode_canonical_json_bytes(raw: bytes) -> dict[str, Any]:
    """Decode strict canonical UTF-8 JSON, rejecting aliases and non-finites."""

    if type(raw) is not bytes:
        raise TypeError("canonical artifact bytes must be exact bytes")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token!r}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("artifact is not strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError("canonical artifact top level must be an object")
    expected = (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    if raw != expected:
        raise ValueError("artifact bytes are not canonical JSON")
    return value


def _require_digest(value: object, *, name: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _require_canonical_absolute_path(path: Path, *, name: str) -> None:
    if not path.is_absolute():
        raise ValueError(f"{name} must be absolute")
    parent = path.parent
    parent_info = parent.lstat()
    if (
        not stat.S_ISDIR(parent_info.st_mode)
        or parent.is_symlink()
        or parent.resolve(strict=True) != parent
    ):
        raise ValueError(f"{name} parent must be a canonical real directory")


def _fsync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_verified_bytes(
    path: Path,
    *,
    expected_file_sha256: str,
) -> bytes:
    _require_canonical_absolute_path(path, name="artifact path")
    parent_before = path.parent.lstat()
    parent_identity = (parent_before.st_dev, parent_before.st_ino)
    expected_sha = _require_digest(
        expected_file_sha256,
        name="expected_file_sha256",
    )
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or path.is_symlink():
        raise ValueError("artifact must be a regular non-symlink file")
    if os.name == "posix" and stat.S_IMODE(before.st_mode) != 0o440:
        raise ValueError("artifact must retain mode 0440")
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
    parent_after = path.parent.stat()
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
        or (parent_after.st_dev, parent_after.st_ino) != parent_identity
    ):
        raise ValueError("artifact changed while it was being read")
    raw = b"".join(chunks)
    if len(raw) != after_open.st_size:
        raise ValueError("artifact byte count changed while it was being read")
    observed_sha = _sha256_bytes(raw)
    if observed_sha != expected_sha:
        raise ValueError("artifact file SHA-256 does not match the expected digest")
    decode_canonical_json_bytes(raw)
    return raw


@dataclass(frozen=True, slots=True)
class CanonicalJsonArtifact:
    """A verified byte transport record, deliberately not an authorization."""

    artifact_path: Path
    file_sha256: str
    size_bytes: int
    _canonical_bytes: bytes = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        _require_canonical_absolute_path(self.artifact_path, name="artifact path")
        _require_digest(self.file_sha256, name="file_sha256")
        if type(self.size_bytes) is not int or self.size_bytes < 1:
            raise ValueError("size_bytes must be a positive integer")
        if type(self._canonical_bytes) is not bytes:
            raise TypeError("_canonical_bytes must be exact bytes")
        if (
            len(self._canonical_bytes) != self.size_bytes
            or _sha256_bytes(self._canonical_bytes) != self.file_sha256
        ):
            raise ValueError("transport record does not match its canonical bytes")
        decode_canonical_json_bytes(self._canonical_bytes)

    @property
    def payload(self) -> dict[str, Any]:
        return decode_canonical_json_bytes(self._canonical_bytes)

    @property
    def canonical_bytes(self) -> bytes:
        return self._canonical_bytes

    def validate_integrity(self) -> None:
        self.__post_init__()
        observed = _read_verified_bytes(
            self.artifact_path,
            expected_file_sha256=self.file_sha256,
        )
        if observed != self._canonical_bytes:
            raise ValueError("live artifact bytes differ from the transport record")


def read_canonical_artifact(
    artifact_path: str | os.PathLike[str],
    *,
    expected_file_sha256: str,
) -> CanonicalJsonArtifact:
    """Read one immutable artifact only under a caller-supplied byte digest."""

    path = Path(artifact_path)
    raw = _read_verified_bytes(
        path,
        expected_file_sha256=expected_file_sha256,
    )
    result = CanonicalJsonArtifact(
        artifact_path=path,
        file_sha256=expected_file_sha256,
        size_bytes=len(raw),
        _canonical_bytes=raw,
    )
    result.validate_integrity()
    return result


def publish_canonical_artifact(
    artifact_path: str | os.PathLike[str],
    payload: Mapping[str, object],
) -> CanonicalJsonArtifact:
    """Atomically publish canonical bytes without replacing any path."""

    path = Path(artifact_path)
    _require_canonical_absolute_path(path, name="artifact path")
    raw = canonical_json_bytes(payload)
    expected_sha = _sha256_bytes(raw)
    parent = path.parent
    parent_info = parent.lstat()
    parent_identity = (parent_info.st_dev, parent_info.st_ino)
    if path.exists() or path.is_symlink():
        raise FileExistsError("refusing to overwrite an artifact path")
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
                raise OSError("temporary artifact has the wrong byte size")
        current_parent = parent.stat()
        if (current_parent.st_dev, current_parent.st_ino) != parent_identity:
            raise ValueError("artifact parent changed before publication")
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as error:
            raise FileExistsError("refusing to overwrite an artifact path") from error
        destination_linked = True
        published = path.lstat()
        if (
            temporary_identity is None
            or (published.st_dev, published.st_ino) != temporary_identity
            or not stat.S_ISREG(published.st_mode)
            or path.is_symlink()
        ):
            raise ValueError("published artifact is not the verified temporary inode")
        _fsync_directory(parent)
        observed = _read_verified_bytes(
            path,
            expected_file_sha256=expected_sha,
        )
        if observed != raw:
            raise OSError("published artifact differs from canonical source bytes")
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
    return read_canonical_artifact(
        path,
        expected_file_sha256=expected_sha,
    )


__all__ = [
    "CanonicalJsonArtifact",
    "canonical_json_bytes",
    "decode_canonical_json_bytes",
    "publish_canonical_artifact",
    "read_canonical_artifact",
]
