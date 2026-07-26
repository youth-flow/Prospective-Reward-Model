from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from smart_reward.phase2_r3_artifacts import (
    canonical_json_bytes,
    decode_canonical_json_bytes,
    publish_canonical_artifact,
    read_canonical_artifact,
)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _write_fixture(path: Path, raw: bytes) -> None:
    path.write_bytes(raw)
    if os.name == "posix":
        path.chmod(0o440)


def test_publish_and_read_verify_exact_canonical_bytes_without_authorizing_role(
    tmp_path: Path,
) -> None:
    path = (tmp_path / "arbitrary-role.json").resolve()
    payload = {
        "schema_version": "caller-invented/v999",
        "role": "caller_claims_success",
        "nested": {"unicode": "科研", "value": 3},
    }
    published = publish_canonical_artifact(path, payload)

    assert published.canonical_bytes == canonical_json_bytes(payload)
    assert published.file_sha256 == _sha(published.canonical_bytes)
    assert published.payload == payload
    assert not hasattr(published, "authorized")

    reopened = read_canonical_artifact(
        path,
        expected_file_sha256=published.file_sha256,
    )
    assert reopened.payload == payload
    assert reopened.canonical_bytes == published.canonical_bytes
    reopened.validate_integrity()


@pytest.mark.parametrize(
    "raw",
    [
        b'{"a":1,"a":2}\n',
        b'{"value":NaN}\n',
        b'{ "a": 1 }\n',
        b'{"a":1}',
        b'["not","an","object"]\n',
        b'{"bad":"\xff"}\n',
    ],
)
def test_read_rejects_duplicate_nonfinite_noncanonical_or_non_utf8_json(
    tmp_path: Path,
    raw: bytes,
) -> None:
    path = (tmp_path / f"invalid-{_sha(raw)}.json").resolve()
    _write_fixture(path, raw)
    with pytest.raises((TypeError, ValueError)):
        read_canonical_artifact(
            path,
            expected_file_sha256=_sha(raw),
        )


def test_decode_and_publish_reject_nonfinite_values(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="strict JSON"):
        canonical_json_bytes({"value": float("inf")})
    with pytest.raises(ValueError, match="non-finite"):
        decode_canonical_json_bytes(b'{"value":Infinity}\n')
    assert list(tmp_path.iterdir()) == []


def test_read_requires_caller_supplied_matching_file_sha(tmp_path: Path) -> None:
    path = (tmp_path / "artifact.json").resolve()
    published = publish_canonical_artifact(path, {"value": 1})
    with pytest.raises(ValueError, match="does not match"):
        read_canonical_artifact(
            path,
            expected_file_sha256="f" * 64,
        )
    published.validate_integrity()


def test_publish_is_no_overwrite_and_preserves_original_bytes(tmp_path: Path) -> None:
    path = (tmp_path / "artifact.json").resolve()
    original = publish_canonical_artifact(path, {"value": "original"})
    with pytest.raises(FileExistsError, match="overwrite"):
        publish_canonical_artifact(path, {"value": "replacement"})
    assert path.read_bytes() == original.canonical_bytes


def test_live_tampering_invalidates_transport_record(tmp_path: Path) -> None:
    path = (tmp_path / "artifact.json").resolve()
    published = publish_canonical_artifact(path, {"value": 1})
    if os.name == "posix":
        path.chmod(0o640)
    path.write_bytes(canonical_json_bytes({"value": 2}))
    if os.name == "posix":
        path.chmod(0o440)
    with pytest.raises(ValueError, match="SHA-256|differ"):
        published.validate_integrity()


def test_symlink_paths_are_rejected_when_supported(tmp_path: Path) -> None:
    target = (tmp_path / "target.json").resolve()
    target.write_bytes(canonical_json_bytes({"value": 1}))
    if os.name == "posix":
        target.chmod(0o440)
    link = (tmp_path / "link.json").resolve()
    try:
        link.symlink_to(target)
    except (NotImplementedError, OSError):
        pytest.skip("symlink creation is unavailable")
    with pytest.raises((FileExistsError, ValueError)):
        publish_canonical_artifact(link, {"value": 2})
    with pytest.raises(ValueError, match="non-symlink"):
        read_canonical_artifact(
            link,
            expected_file_sha256=_sha(target.read_bytes()),
        )


def test_publication_fsyncs_file_before_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = 0
    real_fsync = os.fsync

    def counted_fsync(descriptor: int) -> None:
        nonlocal observed
        observed += 1
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", counted_fsync)
    path = (tmp_path / "artifact.json").resolve()
    publish_canonical_artifact(path, {"value": 1})
    assert observed >= 1
