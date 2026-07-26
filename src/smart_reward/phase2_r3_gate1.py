"""Git-, test-, and container-closed Gate-1 evidence for R3.

Gate 1 is an implementation closure, not a scientific result.  A first,
non-authorizing receipt records Ruff, format, compile, and pytest on one exact
clean commit in an external source-test environment, including tool versions
and raw command bytes.  The deliberately non-injectable HPC4 verifier then
recomputes that commit/tree/blob inventory, binds the caller-pinned receipt,
re-hashes the frozen SIF, and runs compile, Bash syntax, and Torch/CUDA checks
inside that SIF before issuing validator-specific capabilities.  Offline
inspection never creates an authorization object.

The three capabilities in this module have independent schema/role/hash
namespaces.  A generic identity ``ArtifactRef`` is only a transport DTO and
cannot substitute for any of them.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import secrets
import shutil
import stat
import subprocess
import sys
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import InitVar, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Literal, Protocol

GATE1_ARTIFACT_SCHEMA: Final = "phase2-recovery-r3-gate1-implementation/v1"
GATE1_ARTIFACT_ROLE: Final = "validated_r3_implementation"
SOURCE_ARTIFACT_SCHEMA: Final = "phase2-recovery-r3-clean-source/v1"
SOURCE_ARTIFACT_ROLE: Final = "validated_clean_source"
CONTAINER_ARTIFACT_SCHEMA: Final = "phase2-recovery-r3-container/v1"
CONTAINER_ARTIFACT_ROLE: Final = "validated_container_image"

GATE1_VERIFICATION_SCHEMA: Final = "phase2-recovery-r3-gate1-command-suite/v1"
GATE1_CONTRACT_SCHEMA: Final = "phase2-recovery-r3-gate1-contract-coverage/v1"
R3_SOURCE_TEST_RECEIPT_SCHEMA: Final = "phase2-recovery-r3-source-test-receipt/v1"
R3_SOURCE_TEST_RECEIPT_ROLE: Final = "non_authorizing_clean_source_test_receipt"
R3_SOURCE_TEST_SUITE_SCHEMA: Final = "phase2-recovery-r3-source-test-suite/v1"
R3_CONTAINER_RUNTIME_SCHEMA: Final = "phase2-recovery-r3-container-runtime/v1"
R3_IN_CONTAINER_RUNTIME_SCHEMA: Final = "phase2-recovery-r3-in-container-runtime-reverification/v1"
R3_IN_CONTAINER_VERIFICATION_SCHEMA: Final = "phase2-recovery-r3-in-container-command-suite/v1"
R3_GATE1_CAPABILITY_SCHEMA: Final = "phase2-recovery-r3-gate1-capability/v1"
R3_SOURCE_CAPABILITY_SCHEMA: Final = "phase2-recovery-r3-source-capability/v1"
R3_CONTAINER_CAPABILITY_SCHEMA: Final = "phase2-recovery-r3-container-capability/v1"

PRODUCTION_REPO_ROOT: Final = Path("/home/yyangjo/Smart-Reward-Model")
PRODUCTION_PROJECT_ROOT: Final = Path("/project/sigroup/smart-reward-model")
_EXPECTED_PRODUCTION_REPO_ROOT: Final = Path("/home/yyangjo/Smart-Reward-Model")
_EXPECTED_PRODUCTION_PROJECT_ROOT: Final = Path("/project/sigroup/smart-reward-model")
_OUTPUT_DIRECTORY_MODE: Final = 0o750
R3_HPC4_IMAGE_SHA256: Final = "d6fc044b4fa303747908783ea057d5b8946f613bfec6a6ca301e3a02fd7719cb"
_GATE1_RELATIVE: Final = Path("runs/phase2-recovery-r3/gate1/r3-implementation-closure.json")
_SOURCE_TEST_RECEIPT_RELATIVE: Final = Path(
    "runs/phase2-recovery-r3/gate1/r3-source-test-receipt.json"
)
_MAX_EVIDENCE_BYTES: Final = 64 * 1024 * 1024
_MAX_COMMAND_BYTES: Final = 16 * 1024 * 1024
_MAX_SOURCE_FILE_BYTES: Final = 128 * 1024 * 1024
_MAX_SIF_BYTES: Final = 64 * 1024 * 1024 * 1024

_TIMESTAMP_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_GIT_OID_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_PYTHON_VERSION_RE = re.compile(r"3\.11\.[0-9]+\Z")

_DEFINITION_PATH: Final = "containers/prorm-hpc4.def"
_REQUIREMENTS_LOCK_PATH: Final = "containers/requirements-hpc4.lock"

# These are the non-discoverable anchors.  The remainder of the package,
# R3 tests, and R3 HPC4 scripts are discovered from Git and therefore cannot be
# silently omitted when a new formal file is added.
_REQUIRED_TRACKED_PATHS: Final = frozenset(
    {
        ".github/workflows/build-hpc4-image.yml",
        ".github/workflows/ci.yml",
        "pyproject.toml",
        "docs/phase2_recovery_revision3.md",
        "configs/phase2_recovery_r3_science.yaml",
        _DEFINITION_PATH,
        _REQUIREMENTS_LOCK_PATH,
        "src/smart_reward/phase2_checkpoint.py",
        "src/smart_reward/phase2_r3_artifacts.py",
        "src/smart_reward/phase2_r3_config.py",
        "src/smart_reward/phase2_r3_gate0.py",
        "src/smart_reward/phase2_r3_gate1.py",
        "src/smart_reward/phase2_r3_identity.py",
        "src/smart_reward/phase2_r3_inputs.py",
        "src/smart_reward/phase2_r3_materialization.py",
        "src/smart_reward/phase2_r3_orchestrator.py",
        "src/smart_reward/phase2_r3_primary.py",
        "src/smart_reward/phase2_r3_profile.py",
        "src/smart_reward/phase2_r3_profile_artifacts.py",
        "src/smart_reward/phase2_r3_terminal.py",
        "tests/test_phase2_checkpoint.py",
        "tests/test_phase2_core_checkpoint.py",
        "tests/test_phase2_r3_config.py",
        "tests/test_phase2_r3_gate1.py",
        "tests/test_phase2_r3_identity.py",
        "tests/test_phase2_r3_orchestrator.py",
        "tests/test_phase2_r3_primary.py",
        "tests/test_phase2_r3_profile.py",
        "tests/test_phase2_r3_terminal.py",
    }
)

_REQUIRED_SCRIPT_COVERAGE: Final = frozenset(
    {
        "producer",
        "validator",
        "submitter",
        "sbatch",
        "terminal_capture",
        "authorization_validator",
    }
)

_GATE1_CLAUSES: Final = (
    "primary_only_runner_excludes_all_three_controls",
    "profile_primary_controls_authorization_have_distinct_namespaces",
    "checkpoint_progress_signal_state_continuation_fault_injection",
    "science_config_validator_freezes_section_4",
    "submitter_sbatch_terminal_authorization_validator_git_bound",
    "clean_commit_container_inventory_shell_python_tests_pass",
)

_CONTAINER_PROBE_SOURCE: Final = (
    "import json,platform,torch;"
    "p={"
    "'python_implementation':platform.python_implementation(),"
    "'python_version':platform.python_version(),"
    "'torch_version':torch.__version__,"
    "'torch_cuda_version':torch.version.cuda,"
    "'cuda_available':torch.cuda.is_available(),"
    "'cuda_device_count':torch.cuda.device_count(),"
    "'cuda_device_name':torch.cuda.get_device_name(0) "
    "if torch.cuda.is_available() and torch.cuda.device_count()==1 else None,"
    "'cuda_compute_capability':list(torch.cuda.get_device_capability(0)) "
    "if torch.cuda.is_available() and torch.cuda.device_count()==1 else None"
    "};print(json.dumps(p,sort_keys=True,separators=(',',':'),allow_nan=False))"
)

_GATE1_FACTORY_TOKEN = object()
_SOURCE_FACTORY_TOKEN = object()
_CONTAINER_FACTORY_TOKEN = object()


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


def _canonical_sha256(value: object) -> str:
    return _sha256(_canonical_bytes(value))


def _require_digest(value: object, *, name: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_git_oid(value: object, *, name: str) -> str:
    if type(value) is not str or _GIT_OID_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase Git object id")
    return value


def _require_text(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{name} must be a non-empty string")
    return value


def _require_exact_int(value: object, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _require_closed(
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


def _decode_json(
    raw: bytes,
    *,
    name: str,
    require_canonical: bool,
) -> dict[str, Any]:
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


def _safe_relative(value: object, *, name: str) -> str:
    text = _require_text(value, name=name)
    path = Path(text)
    if path.is_absolute() or text != path.as_posix() or ".." in path.parts:
        raise ValueError(f"{name} must be a safe POSIX repository-relative path")
    return text


def _require_real_directory(path: Path, *, name: str) -> os.stat_result:
    if path.is_symlink():
        raise ValueError(f"{name} must not be a symlink")
    try:
        info = path.stat()
    except OSError as error:
        raise ValueError(f"{name} is unavailable") from error
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"{name} must be a real directory")
    return info


def _canonical_real_file(
    value: str | os.PathLike[str],
    *,
    name: str,
    suffix: str | None = None,
) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{name} must use an absolute canonical path")
    if path.is_symlink():
        raise ValueError(f"{name} must not be a symlink")
    try:
        resolved = path.resolve(strict=True)
        info = path.stat()
    except OSError as error:
        raise ValueError(f"{name} is unavailable") from error
    if path != resolved:
        raise ValueError(f"{name} path is not canonical")
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"{name} must be a regular file")
    if suffix is not None and path.suffix != suffix:
        raise ValueError(f"{name} must end in {suffix}")
    return path


def _assert_production_roots() -> tuple[Path, Path]:
    """Validate the fixed, disjoint HPC4 code and persistence namespaces."""

    repo = PRODUCTION_REPO_ROOT
    project = PRODUCTION_PROJECT_ROOT
    if repo != _EXPECTED_PRODUCTION_REPO_ROOT:
        raise RuntimeError("Gate-1 production repository root is not the fixed HPC4 path")
    if project != _EXPECTED_PRODUCTION_PROJECT_ROOT:
        raise RuntimeError("Gate-1 production project root is not the fixed HPC4 path")
    if not repo.is_absolute() or not project.is_absolute():
        raise RuntimeError("Gate-1 production roots must be absolute")
    _require_real_directory(repo, name="HPC4 production repository root")
    _require_real_directory(project, name="HPC4 production project root")
    if repo.resolve(strict=True) != repo or project.resolve(strict=True) != project:
        raise ValueError("Gate-1 production roots must be canonical")
    if repo == project or repo in project.parents or project in repo.parents:
        raise ValueError("Gate-1 production repository and project roots must be disjoint")
    _require_real_directory(
        repo / ".git",
        name="HPC4 production repository .git directory",
    )
    project_git = project / ".git"
    if project_git.exists() or project_git.is_symlink():
        raise ValueError("HPC4 production project root must not be a Git checkout")
    module_root = Path(__file__).resolve(strict=True).parents[2]
    if module_root != repo:
        raise RuntimeError("Gate-1 verifier was not imported from fixed production repository")
    return repo, project


def _production_project_file(
    value: str | os.PathLike[str],
    *,
    name: str,
    suffix: str | None = None,
) -> Path:
    path = _canonical_real_file(value, name=name, suffix=suffix)
    try:
        relative = path.relative_to(PRODUCTION_PROJECT_ROOT)
    except ValueError as error:
        raise ValueError(f"{name} must be retained under the production project root") from error
    if not relative.parts:
        raise ValueError(f"{name} must name a file below the production project root")
    return path


def _root_file(root: Path, relative: str, *, name: str) -> Path:
    _safe_relative(relative, name=f"{name} relative path")
    path = root.joinpath(*Path(relative).parts)
    if path.is_symlink():
        raise ValueError(f"{name} must not be a symlink")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{name} is unavailable") from error
    if resolved != path or root not in resolved.parents:
        raise ValueError(f"{name} escapes the canonical repository")
    if not resolved.is_file():
        raise ValueError(f"{name} must be a regular file")
    return resolved


def _stable_digest(
    path: Path,
    *,
    name: str,
    maximum_bytes: int,
) -> tuple[str, int]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{name} could not be opened safely") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum_bytes:
            raise ValueError(f"{name} is not a bounded regular file")
        digest = hashlib.sha256()
        observed = 0
        while True:
            chunk = os.read(descriptor, 4 * 1024 * 1024)
            if not chunk:
                break
            observed += len(chunk)
            if observed > maximum_bytes:
                raise ValueError(f"{name} exceeds its byte bound")
            digest.update(chunk)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ) or observed != before.st_size:
            raise ValueError(f"{name} changed while it was hashed")
        return digest.hexdigest(), observed
    finally:
        os.close(descriptor)


def _stable_bytes(
    path: Path,
    *,
    name: str,
    maximum_bytes: int,
) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{name} could not be opened safely") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum_bytes:
            raise ValueError(f"{name} is not a bounded regular file")
        chunks: list[bytes] = []
        observed = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum_bytes + 1 - observed))
            if not chunk:
                break
            chunks.append(chunk)
            observed += len(chunk)
            if observed > maximum_bytes:
                raise ValueError(f"{name} exceeds its byte bound")
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ) or observed != before.st_size:
            raise ValueError(f"{name} changed while it was read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


@dataclass(frozen=True, slots=True)
class _CommandResult:
    returncode: int
    stdout: bytes
    stderr: bytes

    def __post_init__(self) -> None:
        if type(self.returncode) is not int:
            raise TypeError("command returncode must be an integer")
        if type(self.stdout) is not bytes or type(self.stderr) is not bytes:
            raise TypeError("command stdout/stderr must be exact bytes")
        if len(self.stdout) > _MAX_COMMAND_BYTES or len(self.stderr) > _MAX_COMMAND_BYTES:
            raise ValueError("command output exceeded the evidence bound")


class _CommandRunner(Protocol):
    def __call__(
        self,
        command: tuple[str, ...],
        *,
        cwd: Path,
        maximum_bytes: int,
    ) -> _CommandResult: ...


def _subprocess_runner(
    command: tuple[str, ...],
    *,
    cwd: Path,
    maximum_bytes: int,
) -> _CommandResult:
    if not command or any(type(item) is not str or not item for item in command):
        raise TypeError("command must contain non-empty exact strings")
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=3600,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"command could not run: {command!r}") from error
    if len(completed.stdout) > maximum_bytes or len(completed.stderr) > maximum_bytes:
        raise RuntimeError(f"command output exceeded {maximum_bytes} bytes: {command!r}")
    return _CommandResult(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _checked_result(
    runner: _CommandRunner,
    command: Sequence[str],
    *,
    cwd: Path,
    name: str,
    maximum_bytes: int = _MAX_COMMAND_BYTES,
) -> _CommandResult:
    argv = tuple(command)
    result = runner(argv, cwd=cwd, maximum_bytes=maximum_bytes)
    if type(result) is not _CommandResult:
        raise TypeError(f"{name} runner returned the wrong result type")
    if result.returncode != 0:
        raise RuntimeError(
            f"{name} failed with exit {result.returncode}; stderr_sha256={_sha256(result.stderr)}"
        )
    return result


def _git(
    runner: _CommandRunner,
    root: Path,
    *arguments: str,
    name: str,
    maximum_bytes: int = _MAX_COMMAND_BYTES,
) -> bytes:
    return _checked_result(
        runner,
        ("git", *arguments),
        cwd=root,
        name=name,
        maximum_bytes=maximum_bytes,
    ).stdout


def _decode_line(raw: bytes, *, name: str) -> str:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{name} is not UTF-8") from error
    if not text.endswith("\n") or "\n" in text[:-1] or "\r" in text:
        raise ValueError(f"{name} must be exactly one LF-terminated line")
    return text[:-1]


def _formal_script_roles(relative: str) -> tuple[str, ...]:
    path = Path(relative)
    lower = path.name.lower()
    roles: set[str] = set()
    if path.suffix.lower() == ".sbatch":
        roles.add("sbatch")
    if "submit" in lower:
        roles.add("submitter")
    if "capture" in lower or "produce" in lower or "materialize" in lower:
        roles.add("producer")
    if "validate" in lower or "verify" in lower or "authoriz" in lower:
        roles.add("validator")
    if "terminal" in lower and (
        "capture" in lower or "terminalize" in lower or "validate" in lower
    ):
        roles.add("terminal_capture")
    if "authoriz" in lower and ("validate" in lower or "verify" in lower or "authoriz" in lower):
        roles.add("authorization_validator")
    # These exact Git-bound entry points implement live verification and
    # admission even though their filenames use capture/run terminology.
    if lower in {
        "capture_phase2_r3_gate1.py",
        "run_phase2_r3_gatep.py",
    }:
        roles.update({"validator", "authorization_validator"})
    return tuple(sorted(roles))


def _path_roles(relative: str) -> tuple[str, ...]:
    roles: set[str] = set()
    if relative == "docs/phase2_recovery_revision3.md":
        roles.add("r3_design")
    if relative == "configs/phase2_recovery_r3_science.yaml":
        roles.add("r3_science_config")
    if relative == _DEFINITION_PATH:
        roles.add("container_definition")
    if relative == _REQUIREMENTS_LOCK_PATH:
        roles.add("runtime_lock")
    if relative.startswith("src/smart_reward/"):
        roles.add("formal_python")
    if relative.startswith("tests/"):
        roles.add("formal_test")
    if relative.startswith("scripts/hpc4/"):
        roles.update(_formal_script_roles(relative))
    if relative == ".github/workflows/ci.yml":
        roles.add("ci_contract")
    if relative == "pyproject.toml":
        roles.add("python_project_contract")
    return tuple(sorted(roles))


def _is_formal_tracked_path(relative: str) -> bool:
    if relative in _REQUIRED_TRACKED_PATHS:
        return True
    if relative.startswith("src/smart_reward/") and relative.endswith(".py"):
        return True
    if relative.startswith("tests/") and Path(relative).name.startswith(
        ("test_phase2_r3_", "test_phase2_checkpoint", "test_phase2_core_checkpoint")
    ):
        return relative.endswith(".py")
    if relative.startswith("scripts/hpc4/"):
        name = Path(relative).name.lower()
        return "phase2_r3" in name or "phase2_recovery_r3" in name
    return False


def _tracked_paths(
    root: Path,
    runner: _CommandRunner,
) -> tuple[str, ...]:
    raw = _git(
        runner,
        root,
        "ls-files",
        "-z",
        name="git tracked-file inventory",
        maximum_bytes=64 * 1024 * 1024,
    )
    try:
        values = [item.decode("utf-8") for item in raw.split(b"\0") if item]
    except UnicodeDecodeError as error:
        raise ValueError("Git tracked paths are not UTF-8") from error
    if len(values) != len(set(values)):
        raise ValueError("Git tracked paths are not unique")
    for value in values:
        _safe_relative(value, name="Git tracked path")
    return tuple(sorted(values))


def _script_coverage(paths: Sequence[str]) -> dict[str, list[str]]:
    coverage = {role: [] for role in sorted(_REQUIRED_SCRIPT_COVERAGE)}
    for relative in paths:
        if not relative.startswith("scripts/hpc4/"):
            continue
        for role in _formal_script_roles(relative):
            if role in coverage:
                coverage[role].append(relative)
    for role, values in coverage.items():
        values.sort()
        if not values:
            raise ValueError(f"Gate-1 has no Git-bound HPC4 {role} file")
    return coverage


def _capture_source(
    root: Path,
    runner: _CommandRunner,
) -> dict[str, object]:
    root = root.resolve(strict=True)
    _require_real_directory(root, name="repository root")
    shown_root = Path(
        _decode_line(
            _git(runner, root, "rev-parse", "--show-toplevel", name="git root"),
            name="git root",
        )
    ).resolve(strict=True)
    if shown_root != root:
        raise ValueError("repository root differs from Git top level")
    status = _git(
        runner,
        root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        "--ignored=no",
        name="git clean-status",
    )
    if status != b"":
        raise ValueError(
            f"Gate-1 requires a clean committed source tree; status_sha256={_sha256(status)}"
        )
    commit = _require_git_oid(
        _decode_line(
            _git(runner, root, "rev-parse", "HEAD", name="git HEAD"),
            name="git HEAD",
        ),
        name="source commit",
    )
    tree = _require_git_oid(
        _decode_line(
            _git(runner, root, "rev-parse", "HEAD^{tree}", name="git tree"),
            name="git tree",
        ),
        name="source tree",
    )
    if len(commit) != len(tree):
        raise ValueError("Git commit/tree use different object formats")

    tracked = _tracked_paths(root, runner)
    missing = sorted(_REQUIRED_TRACKED_PATHS.difference(tracked))
    if missing:
        raise ValueError(f"Gate-1 required tracked files are missing: {missing}")
    formal_paths = tuple(path for path in tracked if _is_formal_tracked_path(path))
    coverage = _script_coverage(formal_paths)

    entries: list[dict[str, object]] = []
    for relative in formal_paths:
        stage_raw = _git(
            runner,
            root,
            "ls-files",
            "--stage",
            "--",
            relative,
            name=f"Git stage for {relative}",
        )
        stage_line = _decode_line(stage_raw, name=f"Git stage for {relative}")
        match = re.fullmatch(
            r"([0-7]{6}) ([0-9a-f]{40}|[0-9a-f]{64}) 0\t(.+)",
            stage_line,
        )
        if match is None or match.group(3) != relative:
            raise ValueError(f"{relative} is not one stage-0 tracked file")
        mode, blob_oid, _ = match.groups()
        if mode not in {"100644", "100755"} or len(blob_oid) != len(commit):
            raise ValueError(f"{relative} has an unsupported Git mode/object format")
        blob = _git(
            runner,
            root,
            "cat-file",
            "blob",
            blob_oid,
            name=f"Git blob for {relative}",
            maximum_bytes=_MAX_SOURCE_FILE_BYTES,
        )
        path = _root_file(root, relative, name=f"source file {relative}")
        working_sha, working_size = _stable_digest(
            path,
            name=f"source file {relative}",
            maximum_bytes=_MAX_SOURCE_FILE_BYTES,
        )
        entries.append(
            {
                "path": relative,
                "roles": list(_path_roles(relative)),
                "git_mode": mode,
                "git_blob_oid": blob_oid,
                "git_blob_sha256": _sha256(blob),
                "git_blob_size_bytes": len(blob),
                "working_sha256": working_sha,
                "working_size_bytes": working_size,
            }
        )
    inventory_sha = _canonical_sha256(entries)
    unsigned: dict[str, object] = {
        "schema_version": SOURCE_ARTIFACT_SCHEMA,
        "role": SOURCE_ARTIFACT_ROLE,
        "git_object_format": "sha1" if len(commit) == 40 else "sha256",
        "commit": commit,
        "tree": tree,
        "repository_relative_root": ".",
        "clean_status_sha256": _sha256(status),
        "inventory": entries,
        "inventory_sha256": inventory_sha,
        "formal_path_count": len(entries),
        "script_coverage": coverage,
    }
    return {**unsigned, "artifact_sha256": _canonical_sha256(unsigned)}


def _external_tool_path(name: str) -> Path:
    if name == "python":
        candidate = Path(sys.executable).resolve(strict=True)
    else:
        resolved = shutil.which(name)
        if resolved is None:
            raise RuntimeError(f"required source-test tool is absent: {name}")
        candidate = Path(resolved).resolve(strict=True)
    return _canonical_real_file(candidate, name=f"source-test {name} executable")


def _capture_source_test_toolchain(
    root: Path,
    runner: _CommandRunner,
) -> dict[str, object]:
    python_path = _external_tool_path("python")
    ruff_path = _external_tool_path("ruff")
    specifications = (
        ("python_version", (str(python_path), "--version")),
        ("ruff_version", (str(ruff_path), "--version")),
        ("pytest_version", (str(python_path), "-m", "pytest", "--version")),
    )
    receipts = [
        _command_receipt(
            name,
            argv,
            _checked_result(runner, argv, cwd=root, name=name),
        )
        for name, argv in specifications
    ]
    if any(
        not (
            _decode_raw_record(
                receipt["stdout"],
                name=f"{receipt['name']} stdout",
            )
            or _decode_raw_record(
                receipt["stderr"],
                name=f"{receipt['name']} stderr",
            )
        )
        for receipt in receipts
    ):
        raise ValueError("source-test tool version command returned no raw identity")
    unsigned: dict[str, object] = {
        "python_path": str(python_path),
        "ruff_path": str(ruff_path),
        "version_receipts": receipts,
    }
    return {**unsigned, "toolchain_sha256": _canonical_sha256(unsigned)}


def _capture_source_test_suite(
    root: Path,
    source: Mapping[str, object],
    toolchain: Mapping[str, object],
    runner: _CommandRunner,
) -> dict[str, object]:
    commands = _source_test_commands(
        source,
        python_path=Path(str(toolchain["python_path"])),
        ruff_path=Path(str(toolchain["ruff_path"])),
    )
    receipts = [
        _command_receipt(
            name,
            argv,
            _checked_result(runner, argv, cwd=root, name=name),
        )
        for name, argv in commands
    ]
    unsigned: dict[str, object] = {
        "schema_version": R3_SOURCE_TEST_SUITE_SCHEMA,
        "commands": receipts,
    }
    return {**unsigned, "suite_sha256": _canonical_sha256(unsigned)}


def _capture_source_test_receipt_for_inspection(
    *,
    root: Path,
    runner: _CommandRunner,
    captured_at_utc: str,
) -> bytes:
    """Private deterministic source-test builder; never issues authority."""

    if _TIMESTAMP_RE.fullmatch(captured_at_utc) is None:
        raise ValueError("captured_at_utc must be canonical whole-second UTC")
    root = root.resolve(strict=True)
    source = _capture_source(root, runner)
    toolchain = _capture_source_test_toolchain(root, runner)
    verification = _capture_source_test_suite(root, source, toolchain, runner)
    if _capture_source(root, runner) != source:
        raise ValueError("source changed while its source-test receipt was produced")
    unsigned: dict[str, object] = {
        "schema_version": R3_SOURCE_TEST_RECEIPT_SCHEMA,
        "role": R3_SOURCE_TEST_RECEIPT_ROLE,
        "captured_at_utc": captured_at_utc,
        "source": source,
        "toolchain": toolchain,
        "verification": verification,
    }
    payload = {**unsigned, "artifact_sha256": _canonical_sha256(unsigned)}
    _validate_source_test_receipt(payload)
    return _canonical_bytes(payload)


def _raw_record(raw: bytes) -> dict[str, object]:
    return {
        "size_bytes": len(raw),
        "sha256": _sha256(raw),
        "bytes_base64": base64.b64encode(raw).decode("ascii"),
    }


def _command_receipt(
    name: str,
    argv: Sequence[str],
    result: _CommandResult,
) -> dict[str, object]:
    if result.returncode != 0:
        raise RuntimeError(
            f"{name} failed with exit {result.returncode}; stderr_sha256={_sha256(result.stderr)}"
        )
    unsigned: dict[str, object] = {
        "name": _require_text(name, name="command receipt name"),
        "argv": list(argv),
        "cwd": ".",
        "exit_code": result.returncode,
        "stdout": _raw_record(result.stdout),
        "stderr": _raw_record(result.stderr),
    }
    return {**unsigned, "receipt_sha256": _canonical_sha256(unsigned)}


def _source_test_commands(
    source: Mapping[str, object],
    *,
    python_path: Path,
    ruff_path: Path,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    inventory = source["inventory"]
    python_paths = [str(item["path"]) for item in inventory if str(item["path"]).endswith(".py")]
    test_paths = [path for path in python_paths if path.startswith("tests/")]
    if not python_paths or not test_paths:
        raise ValueError("Gate-1 source-test path sets are incomplete")
    return (
        (
            "ruff_check",
            (str(ruff_path), "check", "--no-cache", *python_paths),
        ),
        (
            "ruff_format_check",
            (
                str(ruff_path),
                "format",
                "--check",
                "--no-cache",
                *python_paths,
            ),
        ),
        (
            "python_compileall",
            (
                str(python_path),
                "-m",
                "compileall",
                "-q",
                "-f",
                *python_paths,
            ),
        ),
        (
            "pytest",
            (
                str(python_path),
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                *test_paths,
            ),
        ),
    )


def _sif_verification_commands(
    source: Mapping[str, object],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    inventory = source["inventory"]
    python_paths = [str(item["path"]) for item in inventory if str(item["path"]).endswith(".py")]
    shell_paths = [
        str(item["path"])
        for item in inventory
        if str(item["path"]).startswith("scripts/hpc4/")
        and Path(str(item["path"])).suffix.lower() in {".sh", ".sbatch"}
    ]
    if not python_paths or not shell_paths:
        raise ValueError("Gate-1 SIF verification path sets are incomplete")
    return (
        (
            "sif_python_compileall",
            (
                "env",
                "PYTHONPYCACHEPREFIX=/tmp/prorm-r3-gate1-pyc",
                "python",
                "-m",
                "compileall",
                "-q",
                "-f",
                *python_paths,
            ),
        ),
        ("sif_bash_syntax", ("bash", "-n", *shell_paths)),
    )


def _verification_commands(
    source: Mapping[str, object],
    *,
    root: Path,
    engine_path: Path,
    container_path: Path,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    wrapper = (
        str(engine_path),
        "exec",
        "--cleanenv",
        "--containall",
        "--nv",
        "--env",
        f"PYTHONPATH={root / 'src'}",
        "--bind",
        f"{root}:{root}:ro",
        "--bind",
        f"{PRODUCTION_PROJECT_ROOT}:{PRODUCTION_PROJECT_ROOT}:ro",
        "--pwd",
        str(root),
        str(container_path),
    )
    return tuple((name, (*wrapper, *inner)) for name, inner in _sif_verification_commands(source))


def _capture_verification(
    root: Path,
    source: Mapping[str, object],
    runner: _CommandRunner,
    *,
    engine_path: Path,
    container_path: Path,
    source_test_receipt: Mapping[str, object],
    source_test_receipt_file_sha256: str,
) -> dict[str, object]:
    validated_receipt = _validate_source_test_receipt(source_test_receipt)
    if validated_receipt["source"] != source:
        raise ValueError("source-test receipt binds another clean source")
    receipts: list[dict[str, object]] = []
    for name, argv in _verification_commands(
        source,
        root=root,
        engine_path=engine_path,
        container_path=container_path,
    ):
        result = _checked_result(runner, argv, cwd=root, name=name)
        receipts.append(_command_receipt(name, argv, result))
    unsigned: dict[str, object] = {
        "schema_version": GATE1_VERIFICATION_SCHEMA,
        "repository_path": str(root),
        "engine_path": str(engine_path),
        "container_path": str(container_path),
        "source_test_receipt_file_sha256": _require_digest(
            source_test_receipt_file_sha256,
            name="source-test receipt file SHA256",
        ),
        "source_test_receipt": validated_receipt,
        "commands": receipts,
    }
    return {**unsigned, "suite_sha256": _canonical_sha256(unsigned)}


def _probe_payload(raw: bytes) -> dict[str, object]:
    probe = _decode_json(raw, name="container Torch/CUDA probe", require_canonical=True)
    _require_closed(
        probe,
        name="container Torch/CUDA probe",
        keys={
            "python_implementation",
            "python_version",
            "torch_version",
            "torch_cuda_version",
            "cuda_available",
            "cuda_device_count",
            "cuda_device_name",
            "cuda_compute_capability",
        },
    )
    if (
        probe["python_implementation"] != "CPython"
        or type(probe["python_version"]) is not str
        or _PYTHON_VERSION_RE.fullmatch(probe["python_version"]) is None
        or type(probe["torch_version"]) is not str
        or not probe["torch_version"].startswith("2.7.1")
        or probe["torch_cuda_version"] != "12.6"
        or probe["cuda_available"] is not True
        or probe["cuda_device_count"] != 1
        or probe["cuda_device_name"] != "NVIDIA L20"
        or probe["cuda_compute_capability"] != [8, 9]
    ):
        raise ValueError("container runtime does not match the frozen HPC4 L20 contract")
    return probe


def _capture_runtime(
    root: Path,
    *,
    runner: _CommandRunner,
    engine_path: Path,
    container_path: Path,
) -> dict[str, object]:
    engine_path = _canonical_real_file(engine_path, name="container runtime executable")
    engine = engine_path.name.lower()
    if engine not in {"apptainer", "singularity"}:
        raise ValueError("container runtime must be canonical Apptainer or Singularity")
    version_argv = (str(engine_path), "--version")
    version_result = _checked_result(
        runner,
        version_argv,
        cwd=root,
        name="container runtime version",
    )
    try:
        version_text = version_result.stdout.decode("utf-8").strip().lower()
    except UnicodeDecodeError as error:
        raise ValueError("container runtime version is not UTF-8") from error
    if engine not in version_text:
        raise ValueError("container runtime version output identifies another engine")
    probe_argv = (
        str(engine_path),
        "exec",
        "--cleanenv",
        "--containall",
        "--nv",
        str(container_path),
        "python",
        "-c",
        _CONTAINER_PROBE_SOURCE,
    )
    probe_result = _checked_result(
        runner,
        probe_argv,
        cwd=root,
        name="container Torch/CUDA runtime probe",
    )
    probe = _probe_payload(probe_result.stdout)
    version_receipt = _command_receipt(
        "container_runtime_version",
        version_argv,
        version_result,
    )
    probe_receipt = _command_receipt(
        "container_torch_cuda_probe",
        probe_argv,
        probe_result,
    )
    unsigned: dict[str, object] = {
        "schema_version": R3_CONTAINER_RUNTIME_SCHEMA,
        "engine": engine,
        "engine_path": str(engine_path),
        "version_receipt": version_receipt,
        "probe_receipt": probe_receipt,
        "probe": probe,
    }
    return {**unsigned, "runtime_probe_sha256": _canonical_sha256(unsigned)}


def _inventory_item(
    source: Mapping[str, object],
    relative: str,
) -> Mapping[str, object]:
    matches = [item for item in source["inventory"] if item["path"] == relative]
    if len(matches) != 1:
        raise ValueError(f"source inventory does not contain exactly one {relative}")
    return matches[0]


def _dependency_binding(
    source: Mapping[str, object],
    relative: str,
) -> dict[str, object]:
    item = _inventory_item(source, relative)
    return {
        "path": relative,
        "git_blob_oid": item["git_blob_oid"],
        "git_blob_sha256": item["git_blob_sha256"],
        "git_blob_size_bytes": item["git_blob_size_bytes"],
        "working_sha256": item["working_sha256"],
        "working_size_bytes": item["working_size_bytes"],
    }


def _capture_container(
    root: Path,
    source: Mapping[str, object],
    *,
    runner: _CommandRunner,
    engine_path: Path,
    container_path: Path,
) -> dict[str, object]:
    container_path = _canonical_real_file(
        container_path,
        name="R3 canonical container",
        suffix=".sif",
    )
    sif_sha, sif_size = _stable_digest(
        container_path,
        name="R3 canonical container",
        maximum_bytes=_MAX_SIF_BYTES,
    )
    runtime = _capture_runtime(
        root,
        runner=runner,
        engine_path=engine_path,
        container_path=container_path,
    )
    unsigned: dict[str, object] = {
        "schema_version": CONTAINER_ARTIFACT_SCHEMA,
        "role": CONTAINER_ARTIFACT_ROLE,
        "canonical_sif_path": str(container_path),
        "sif_sha256": sif_sha,
        "sif_size_bytes": sif_size,
        "definition": _dependency_binding(source, _DEFINITION_PATH),
        "requirements_lock": _dependency_binding(source, _REQUIREMENTS_LOCK_PATH),
        "runtime": runtime,
    }
    return {**unsigned, "artifact_sha256": _canonical_sha256(unsigned)}


def _contract(source: Mapping[str, object]) -> dict[str, object]:
    coverage = source["script_coverage"]
    unsigned: dict[str, object] = {
        "schema_version": GATE1_CONTRACT_SCHEMA,
        "gate1_clauses": list(_GATE1_CLAUSES),
        "script_coverage": coverage,
        "verification_basis": {
            "primary_only_and_namespace_checks": "external_source_test_pytest_receipt",
            "checkpoint_continuation_fault_injection": ("external_source_test_pytest_receipt"),
            "science_config_closure": "external_source_test_pytest_receipt",
            "python_static_and_compile_checks": [
                "external_source_test_ruff_check",
                "external_source_test_ruff_format_check",
                "external_source_test_python_compileall",
                "exact_sif_python_compileall",
            ],
            "shell_static_check": "exact_sif_bash_syntax",
            "clean_source_and_blob_closure": "git",
            "container_runtime_closure": "live_apptainer_or_singularity_torch_cuda_probe",
        },
    }
    return {**unsigned, "contract_sha256": _canonical_sha256(unsigned)}


def _capture_gate1_state(
    *,
    root: Path,
    container_path: Path,
    engine_path: Path,
    runner: _CommandRunner,
    source_test_receipt: Mapping[str, object],
    source_test_receipt_file_sha256: str,
    prevalidated_source: Mapping[str, object] | None = None,
) -> dict[str, object]:
    source = (
        _capture_source(root, runner)
        if prevalidated_source is None
        else _validate_source(dict(prevalidated_source))
    )
    container = _capture_container(
        root,
        source,
        runner=runner,
        engine_path=engine_path,
        container_path=container_path,
    )
    verification = _capture_verification(
        root,
        source,
        runner,
        engine_path=engine_path,
        container_path=container_path,
        source_test_receipt=source_test_receipt,
        source_test_receipt_file_sha256=source_test_receipt_file_sha256,
    )
    return {
        "source": source,
        "verification": verification,
        "container": container,
        "contract": _contract(source),
    }


def _gate1_payload(
    state: Mapping[str, object],
    *,
    captured_at_utc: str,
) -> dict[str, object]:
    if _TIMESTAMP_RE.fullmatch(captured_at_utc) is None:
        raise ValueError("captured_at_utc must be canonical whole-second UTC")
    unsigned: dict[str, object] = {
        "schema_version": GATE1_ARTIFACT_SCHEMA,
        "role": GATE1_ARTIFACT_ROLE,
        "captured_at_utc": captured_at_utc,
        "source": state["source"],
        "verification": state["verification"],
        "container": state["container"],
        "contract": state["contract"],
    }
    return {**unsigned, "artifact_sha256": _canonical_sha256(unsigned)}


def _capture_gate1_evidence_for_inspection(
    *,
    root: Path,
    container_path: Path,
    engine_path: Path,
    runner: _CommandRunner,
    captured_at_utc: str,
    source_test_receipt_raw: bytes,
    expected_source_test_receipt_file_sha256: str,
) -> bytes:
    """Private deterministic builder for non-authorizing tests and review."""

    receipt_file_sha = _sha256(source_test_receipt_raw)
    if receipt_file_sha != _require_digest(
        expected_source_test_receipt_file_sha256,
        name="expected source-test receipt file SHA256",
    ):
        raise ValueError("source-test receipt file differs from caller expectation")
    source_test_receipt = _validate_source_test_receipt(
        _decode_json(
            source_test_receipt_raw,
            name="source-test receipt",
            require_canonical=True,
        )
    )
    state = _capture_gate1_state(
        root=root.resolve(strict=True),
        container_path=container_path,
        engine_path=engine_path,
        runner=runner,
        source_test_receipt=source_test_receipt,
        source_test_receipt_file_sha256=receipt_file_sha,
    )
    payload = _gate1_payload(state, captured_at_utc=captured_at_utc)
    _validate_payload(payload)
    return _canonical_bytes(payload)


def _decode_raw_record(value: object, *, name: str) -> bytes:
    record = _require_closed(
        value,
        name=name,
        keys={"size_bytes", "sha256", "bytes_base64"},
    )
    size = _require_exact_int(record["size_bytes"], name=f"{name} size")
    if size > _MAX_COMMAND_BYTES:
        raise ValueError(f"{name} exceeds the evidence byte bound")
    digest = _require_digest(record["sha256"], name=f"{name} SHA256")
    if type(record["bytes_base64"]) is not str:
        raise TypeError(f"{name} base64 must be a string")
    try:
        raw = base64.b64decode(record["bytes_base64"], validate=True)
    except (ValueError, TypeError) as error:
        raise ValueError(f"{name} is not strict base64") from error
    if (
        len(raw) != size
        or _sha256(raw) != digest
        or base64.b64encode(raw).decode("ascii") != record["bytes_base64"]
    ):
        raise ValueError(f"{name} byte binding is invalid")
    return raw


def _validate_command_receipt(
    value: object,
    *,
    name: str,
) -> tuple[str, tuple[str, ...], bytes, bytes]:
    record = _require_closed(
        value,
        name=name,
        keys={
            "name",
            "argv",
            "cwd",
            "exit_code",
            "stdout",
            "stderr",
            "receipt_sha256",
        },
    )
    command_name = _require_text(record["name"], name=f"{name} command name")
    argv_value = record["argv"]
    if (
        not isinstance(argv_value, list)
        or not argv_value
        or any(type(item) is not str or not item for item in argv_value)
    ):
        raise ValueError(f"{name} argv is invalid")
    argv = tuple(argv_value)
    if record["cwd"] != "." or record["exit_code"] != 0:
        raise ValueError(f"{name} did not record a successful repository-root command")
    stdout = _decode_raw_record(record["stdout"], name=f"{name} stdout")
    stderr = _decode_raw_record(record["stderr"], name=f"{name} stderr")
    receipt_sha = _require_digest(
        record["receipt_sha256"],
        name=f"{name} receipt SHA256",
    )
    unsigned = dict(record)
    del unsigned["receipt_sha256"]
    if receipt_sha != _canonical_sha256(unsigned):
        raise ValueError(f"{name} receipt self-hash is invalid")
    return command_name, argv, stdout, stderr


def _validate_inventory(
    value: object,
    *,
    git_oid_length: int,
) -> tuple[list[dict[str, object]], dict[str, list[str]]]:
    if not isinstance(value, list) or not value:
        raise ValueError("source inventory must be a non-empty list")
    entries: list[dict[str, object]] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        record = _require_closed(
            item,
            name=f"source inventory[{index}]",
            keys={
                "path",
                "roles",
                "git_mode",
                "git_blob_oid",
                "git_blob_sha256",
                "git_blob_size_bytes",
                "working_sha256",
                "working_size_bytes",
            },
        )
        path = _safe_relative(record["path"], name=f"source inventory[{index}] path")
        if path in seen:
            raise ValueError("source inventory paths are not unique")
        seen.add(path)
        roles = record["roles"]
        if (
            not isinstance(roles, list)
            or roles != sorted(set(roles))
            or any(type(role) is not str or not role for role in roles)
            or tuple(roles) != _path_roles(path)
        ):
            raise ValueError(f"source inventory roles are invalid for {path}")
        if record["git_mode"] not in {"100644", "100755"}:
            raise ValueError(f"source inventory Git mode is invalid for {path}")
        oid = _require_git_oid(record["git_blob_oid"], name=f"{path} Git blob")
        if len(oid) != git_oid_length:
            raise ValueError(f"{path} Git object format differs from HEAD")
        _require_digest(record["git_blob_sha256"], name=f"{path} Git blob SHA256")
        _require_digest(record["working_sha256"], name=f"{path} working SHA256")
        _require_exact_int(
            record["git_blob_size_bytes"],
            name=f"{path} Git blob size",
        )
        _require_exact_int(
            record["working_size_bytes"],
            name=f"{path} working size",
        )
        entries.append(record)
    if [entry["path"] for entry in entries] != sorted(seen):
        raise ValueError("source inventory is not canonically path ordered")
    missing = sorted(_REQUIRED_TRACKED_PATHS.difference(seen))
    if missing:
        raise ValueError(f"source inventory omits required files: {missing}")
    for path in seen:
        if not _is_formal_tracked_path(path):
            raise ValueError(f"source inventory contains a non-formal path: {path}")
    coverage = _script_coverage(tuple(sorted(seen)))
    return entries, coverage


def _validate_source(value: object) -> dict[str, Any]:
    source = _require_closed(
        value,
        name="Gate-1 source",
        keys={
            "schema_version",
            "role",
            "git_object_format",
            "commit",
            "tree",
            "repository_relative_root",
            "clean_status_sha256",
            "inventory",
            "inventory_sha256",
            "formal_path_count",
            "script_coverage",
            "artifact_sha256",
        },
    )
    if (
        source["schema_version"] != SOURCE_ARTIFACT_SCHEMA
        or source["role"] != SOURCE_ARTIFACT_ROLE
        or source["repository_relative_root"] != "."
        or source["clean_status_sha256"] != _sha256(b"")
    ):
        raise ValueError("Gate-1 source identity is invalid")
    commit = _require_git_oid(source["commit"], name="source commit")
    tree = _require_git_oid(source["tree"], name="source tree")
    if len(commit) != len(tree):
        raise ValueError("source commit/tree object formats differ")
    expected_format = "sha1" if len(commit) == 40 else "sha256"
    if source["git_object_format"] != expected_format:
        raise ValueError("source Git object format is invalid")
    entries, coverage = _validate_inventory(
        source["inventory"],
        git_oid_length=len(commit),
    )
    if (
        source["formal_path_count"] != len(entries)
        or source["script_coverage"] != coverage
        or source["inventory_sha256"] != _canonical_sha256(entries)
    ):
        raise ValueError("source inventory closure is invalid")
    _require_digest(source["inventory_sha256"], name="source inventory SHA256")
    artifact_sha = _require_digest(
        source["artifact_sha256"],
        name="source artifact SHA256",
    )
    unsigned = dict(source)
    del unsigned["artifact_sha256"]
    if artifact_sha != _canonical_sha256(unsigned):
        raise ValueError("source artifact self-hash is invalid")
    return source


def _validate_source_test_toolchain(value: object) -> dict[str, Any]:
    toolchain = _require_closed(
        value,
        name="source-test toolchain",
        keys={
            "python_path",
            "ruff_path",
            "version_receipts",
            "toolchain_sha256",
        },
    )
    python_path = Path(_require_text(toolchain["python_path"], name="source-test Python path"))
    ruff_path = Path(_require_text(toolchain["ruff_path"], name="source-test Ruff path"))
    if not python_path.is_absolute() or not ruff_path.is_absolute():
        raise ValueError("source-test tool paths must be absolute")
    expected = (
        ("python_version", (str(python_path), "--version")),
        ("ruff_version", (str(ruff_path), "--version")),
        ("pytest_version", (str(python_path), "-m", "pytest", "--version")),
    )
    receipts = toolchain["version_receipts"]
    if not isinstance(receipts, list) or len(receipts) != len(expected):
        raise ValueError("source-test toolchain version receipt count is invalid")
    for index, (receipt, (expected_name, expected_argv)) in enumerate(
        zip(receipts, expected, strict=True)
    ):
        name, argv, stdout, stderr = _validate_command_receipt(
            receipt,
            name=f"source-test tool version[{index}]",
        )
        if name != expected_name or argv != expected_argv or not stdout + stderr:
            raise ValueError("source-test tool version receipt is invalid")
    toolchain_sha = _require_digest(
        toolchain["toolchain_sha256"],
        name="source-test toolchain SHA256",
    )
    unsigned = dict(toolchain)
    del unsigned["toolchain_sha256"]
    if toolchain_sha != _canonical_sha256(unsigned):
        raise ValueError("source-test toolchain self-hash is invalid")
    return toolchain


def _validate_source_test_suite(
    value: object,
    *,
    source: Mapping[str, object],
    toolchain: Mapping[str, object],
) -> dict[str, Any]:
    suite = _require_closed(
        value,
        name="source-test suite",
        keys={"schema_version", "commands", "suite_sha256"},
    )
    if suite["schema_version"] != R3_SOURCE_TEST_SUITE_SCHEMA:
        raise ValueError("source-test suite schema is invalid")
    expected = _source_test_commands(
        source,
        python_path=Path(str(toolchain["python_path"])),
        ruff_path=Path(str(toolchain["ruff_path"])),
    )
    commands = suite["commands"]
    if not isinstance(commands, list) or len(commands) != len(expected):
        raise ValueError("source-test command count is invalid")
    for index, (receipt, (expected_name, expected_argv)) in enumerate(
        zip(commands, expected, strict=True)
    ):
        name, argv, _, _ = _validate_command_receipt(
            receipt,
            name=f"source-test command[{index}]",
        )
        if name != expected_name or argv != expected_argv:
            raise ValueError("source-test command specification drifted")
    suite_sha = _require_digest(
        suite["suite_sha256"],
        name="source-test suite SHA256",
    )
    unsigned = dict(suite)
    del unsigned["suite_sha256"]
    if suite_sha != _canonical_sha256(unsigned):
        raise ValueError("source-test suite self-hash is invalid")
    return suite


def _validate_source_test_receipt(value: object) -> dict[str, Any]:
    receipt = _require_closed(
        value,
        name="source-test receipt",
        keys={
            "schema_version",
            "role",
            "captured_at_utc",
            "source",
            "toolchain",
            "verification",
            "artifact_sha256",
        },
    )
    if (
        receipt["schema_version"] != R3_SOURCE_TEST_RECEIPT_SCHEMA
        or receipt["role"] != R3_SOURCE_TEST_RECEIPT_ROLE
        or type(receipt["captured_at_utc"]) is not str
        or _TIMESTAMP_RE.fullmatch(receipt["captured_at_utc"]) is None
    ):
        raise ValueError("source-test receipt identity is invalid")
    source = _validate_source(receipt["source"])
    toolchain = _validate_source_test_toolchain(receipt["toolchain"])
    _validate_source_test_suite(
        receipt["verification"],
        source=source,
        toolchain=toolchain,
    )
    artifact_sha = _require_digest(
        receipt["artifact_sha256"],
        name="source-test receipt artifact SHA256",
    )
    unsigned = dict(receipt)
    del unsigned["artifact_sha256"]
    if artifact_sha != _canonical_sha256(unsigned):
        raise ValueError("source-test receipt self-hash is invalid")
    return receipt


def _validate_verification(
    value: object,
    source: Mapping[str, object],
    container: Mapping[str, object],
) -> dict[str, Any]:
    suite = _require_closed(
        value,
        name="Gate-1 verification suite",
        keys={
            "schema_version",
            "repository_path",
            "engine_path",
            "container_path",
            "source_test_receipt_file_sha256",
            "source_test_receipt",
            "commands",
            "suite_sha256",
        },
    )
    if suite["schema_version"] != GATE1_VERIFICATION_SCHEMA:
        raise ValueError("Gate-1 verification suite schema is invalid")
    repository_path = Path(
        _require_text(
            suite["repository_path"],
            name="Gate-1 verification repository path",
        )
    )
    engine_path = Path(
        _require_text(
            suite["engine_path"],
            name="Gate-1 verification engine path",
        )
    )
    container_path = Path(
        _require_text(
            suite["container_path"],
            name="Gate-1 verification container path",
        )
    )
    runtime = container["runtime"]
    if (
        not repository_path.is_absolute()
        or not engine_path.is_absolute()
        or not container_path.is_absolute()
        or str(engine_path) != runtime["engine_path"]
        or str(container_path) != container["canonical_sif_path"]
    ):
        raise ValueError("Gate-1 verification execution paths are invalid")
    source_test_receipt = _validate_source_test_receipt(suite["source_test_receipt"])
    source_test_file_sha = _require_digest(
        suite["source_test_receipt_file_sha256"],
        name="Gate-1 source-test receipt file SHA256",
    )
    if source_test_receipt["source"] != source or source_test_file_sha != _sha256(
        _canonical_bytes(source_test_receipt)
    ):
        raise ValueError("Gate-1 source-test receipt binding is invalid")
    commands = suite["commands"]
    if not isinstance(commands, list):
        raise TypeError("Gate-1 verification commands must be a list")
    expected = _verification_commands(
        source,
        root=repository_path,
        engine_path=engine_path,
        container_path=container_path,
    )
    if len(commands) != len(expected):
        raise ValueError("Gate-1 verification command count is invalid")
    for index, (record, (expected_name, expected_argv)) in enumerate(
        zip(commands, expected, strict=True)
    ):
        name, argv, _, _ = _validate_command_receipt(
            record,
            name=f"Gate-1 verification command[{index}]",
        )
        if name != expected_name or argv != expected_argv:
            raise ValueError("Gate-1 verification command specification drifted")
    suite_sha = _require_digest(suite["suite_sha256"], name="verification suite SHA256")
    unsigned = dict(suite)
    del unsigned["suite_sha256"]
    if suite_sha != _canonical_sha256(unsigned):
        raise ValueError("Gate-1 verification suite self-hash is invalid")
    return suite


def _validate_dependency_binding(
    value: object,
    *,
    name: str,
    source: Mapping[str, object],
    expected_path: str,
) -> dict[str, Any]:
    binding = _require_closed(
        value,
        name=name,
        keys={
            "path",
            "git_blob_oid",
            "git_blob_sha256",
            "git_blob_size_bytes",
            "working_sha256",
            "working_size_bytes",
        },
    )
    if binding["path"] != expected_path:
        raise ValueError(f"{name} path is invalid")
    expected = _dependency_binding(source, expected_path)
    if binding != expected:
        raise ValueError(f"{name} differs from the clean-source inventory")
    return binding


def _validate_runtime(value: object) -> dict[str, Any]:
    runtime = _require_closed(
        value,
        name="container runtime",
        keys={
            "schema_version",
            "engine",
            "engine_path",
            "version_receipt",
            "probe_receipt",
            "probe",
            "runtime_probe_sha256",
        },
    )
    if runtime["schema_version"] != R3_CONTAINER_RUNTIME_SCHEMA:
        raise ValueError("container runtime schema is invalid")
    engine = runtime["engine"]
    engine_path = Path(_require_text(runtime["engine_path"], name="runtime engine path"))
    if (
        engine not in {"apptainer", "singularity"}
        or not engine_path.is_absolute()
        or engine_path.name.lower() != engine
    ):
        raise ValueError("container runtime engine identity is invalid")
    version_name, version_argv, version_stdout, _ = _validate_command_receipt(
        runtime["version_receipt"],
        name="container runtime version receipt",
    )
    if (
        version_name != "container_runtime_version"
        or version_argv != (str(engine_path), "--version")
        or engine not in version_stdout.decode("utf-8").strip().lower()
    ):
        raise ValueError("container runtime version receipt is invalid")
    probe_name, probe_argv, probe_stdout, _ = _validate_command_receipt(
        runtime["probe_receipt"],
        name="container Torch/CUDA probe receipt",
    )
    if (
        probe_name != "container_torch_cuda_probe"
        or len(probe_argv) != 9
        or probe_argv[:5] != (str(engine_path), "exec", "--cleanenv", "--containall", "--nv")
        or probe_argv[6:] != ("python", "-c", _CONTAINER_PROBE_SOURCE)
    ):
        raise ValueError("container Torch/CUDA probe command is invalid")
    parsed = _probe_payload(probe_stdout)
    if runtime["probe"] != parsed:
        raise ValueError("container runtime parsed probe differs from raw evidence")
    runtime_sha = _require_digest(
        runtime["runtime_probe_sha256"],
        name="container runtime probe SHA256",
    )
    unsigned = dict(runtime)
    del unsigned["runtime_probe_sha256"]
    if runtime_sha != _canonical_sha256(unsigned):
        raise ValueError("container runtime probe self-hash is invalid")
    return runtime


def _validate_container(
    value: object,
    source: Mapping[str, object],
) -> dict[str, Any]:
    container = _require_closed(
        value,
        name="Gate-1 container",
        keys={
            "schema_version",
            "role",
            "canonical_sif_path",
            "sif_sha256",
            "sif_size_bytes",
            "definition",
            "requirements_lock",
            "runtime",
            "artifact_sha256",
        },
    )
    path = Path(_require_text(container["canonical_sif_path"], name="canonical SIF path"))
    if (
        container["schema_version"] != CONTAINER_ARTIFACT_SCHEMA
        or container["role"] != CONTAINER_ARTIFACT_ROLE
        or not path.is_absolute()
        or path.suffix != ".sif"
    ):
        raise ValueError("Gate-1 container identity is invalid")
    _require_digest(container["sif_sha256"], name="SIF SHA256")
    _require_exact_int(container["sif_size_bytes"], name="SIF size", minimum=1)
    _validate_dependency_binding(
        container["definition"],
        name="container definition",
        source=source,
        expected_path=_DEFINITION_PATH,
    )
    _validate_dependency_binding(
        container["requirements_lock"],
        name="container requirements lock",
        source=source,
        expected_path=_REQUIREMENTS_LOCK_PATH,
    )
    runtime = _validate_runtime(container["runtime"])
    probe_argv = runtime["probe_receipt"]["argv"]
    if probe_argv[5] != str(path):
        raise ValueError("container runtime probe binds another SIF path")
    artifact_sha = _require_digest(
        container["artifact_sha256"],
        name="container artifact SHA256",
    )
    unsigned = dict(container)
    del unsigned["artifact_sha256"]
    if artifact_sha != _canonical_sha256(unsigned):
        raise ValueError("container artifact self-hash is invalid")
    return container


def _validate_contract(
    value: object,
    source: Mapping[str, object],
) -> dict[str, Any]:
    contract = _require_closed(
        value,
        name="Gate-1 contract",
        keys={
            "schema_version",
            "gate1_clauses",
            "script_coverage",
            "verification_basis",
            "contract_sha256",
        },
    )
    if (
        contract["schema_version"] != GATE1_CONTRACT_SCHEMA
        or contract["gate1_clauses"] != list(_GATE1_CLAUSES)
        or contract["script_coverage"] != source["script_coverage"]
        or contract["verification_basis"]
        != {
            "primary_only_and_namespace_checks": ("external_source_test_pytest_receipt"),
            "checkpoint_continuation_fault_injection": ("external_source_test_pytest_receipt"),
            "science_config_closure": "external_source_test_pytest_receipt",
            "python_static_and_compile_checks": [
                "external_source_test_ruff_check",
                "external_source_test_ruff_format_check",
                "external_source_test_python_compileall",
                "exact_sif_python_compileall",
            ],
            "shell_static_check": "exact_sif_bash_syntax",
            "clean_source_and_blob_closure": "git",
            "container_runtime_closure": ("live_apptainer_or_singularity_torch_cuda_probe"),
        }
    ):
        raise ValueError("Gate-1 contract coverage is invalid")
    contract_sha = _require_digest(
        contract["contract_sha256"],
        name="Gate-1 contract SHA256",
    )
    unsigned = dict(contract)
    del unsigned["contract_sha256"]
    if contract_sha != _canonical_sha256(unsigned):
        raise ValueError("Gate-1 contract self-hash is invalid")
    return contract


def _validate_payload(payload: object) -> dict[str, Any]:
    value = _require_closed(
        payload,
        name="Gate-1 artifact",
        keys={
            "schema_version",
            "role",
            "captured_at_utc",
            "source",
            "verification",
            "container",
            "contract",
            "artifact_sha256",
        },
    )
    if (
        value["schema_version"] != GATE1_ARTIFACT_SCHEMA
        or value["role"] != GATE1_ARTIFACT_ROLE
        or type(value["captured_at_utc"]) is not str
        or _TIMESTAMP_RE.fullmatch(value["captured_at_utc"]) is None
    ):
        raise ValueError("Gate-1 artifact identity is invalid")
    source = _validate_source(value["source"])
    container = _validate_container(value["container"], source)
    _validate_verification(value["verification"], source, container)
    _validate_contract(value["contract"], source)
    artifact_sha = _require_digest(
        value["artifact_sha256"],
        name="Gate-1 artifact SHA256",
    )
    unsigned = dict(value)
    del unsigned["artifact_sha256"]
    if artifact_sha != _canonical_sha256(unsigned):
        raise ValueError("Gate-1 artifact self-hash is invalid")
    return value


@dataclass(frozen=True, slots=True)
class R3Gate1Inspection:
    """Canonical evidence inspection with no formal authority."""

    schema_version: str
    artifact_sha256: str
    file_sha256: str
    source_artifact_sha256: str
    container_artifact_sha256: str
    source_commit: str
    formal_path_count: int
    formal_authorization: Literal[False] = False

    def __post_init__(self) -> None:
        if self.schema_version != GATE1_ARTIFACT_SCHEMA:
            raise ValueError("Gate-1 inspection schema is invalid")
        _require_digest(self.artifact_sha256, name="Gate-1 artifact SHA256")
        _require_digest(self.file_sha256, name="Gate-1 file SHA256")
        _require_digest(
            self.source_artifact_sha256,
            name="Gate-1 source artifact SHA256",
        )
        _require_digest(
            self.container_artifact_sha256,
            name="Gate-1 container artifact SHA256",
        )
        _require_git_oid(self.source_commit, name="Gate-1 source commit")
        _require_exact_int(self.formal_path_count, name="formal path count", minimum=1)
        if self.formal_authorization is not False:
            raise ValueError("Gate-1 inspection cannot authorize formal execution")


@dataclass(frozen=True, slots=True)
class R3SourceTestInspection:
    """Canonical clean-source test receipt with no formal authority."""

    schema_version: str
    artifact_sha256: str
    file_sha256: str
    source_artifact_sha256: str
    source_commit: str
    verification_suite_sha256: str
    formal_authorization: Literal[False] = False

    def __post_init__(self) -> None:
        if self.schema_version != R3_SOURCE_TEST_RECEIPT_SCHEMA:
            raise ValueError("source-test inspection schema is invalid")
        for name, value in (
            ("source-test artifact SHA256", self.artifact_sha256),
            ("source-test file SHA256", self.file_sha256),
            ("source artifact SHA256", self.source_artifact_sha256),
            ("source-test suite SHA256", self.verification_suite_sha256),
        ):
            _require_digest(value, name=name)
        _require_git_oid(self.source_commit, name="source-test source commit")
        if self.formal_authorization is not False:
            raise ValueError("source-test inspection cannot authorize formal execution")


def inspect_r3_source_test_receipt(
    path: str | os.PathLike[str],
    *,
    expected_file_sha256: str | None = None,
) -> R3SourceTestInspection:
    """Inspect a canonical external source-test receipt without authority."""

    source_path = _canonical_real_file(path, name="R3 source-test receipt")
    raw = _stable_bytes(
        source_path,
        name="R3 source-test receipt",
        maximum_bytes=_MAX_EVIDENCE_BYTES,
    )
    file_sha = _sha256(raw)
    if expected_file_sha256 is not None and file_sha != _require_digest(
        expected_file_sha256,
        name="expected source-test receipt file SHA256",
    ):
        raise ValueError("source-test receipt file differs from caller expectation")
    payload = _validate_source_test_receipt(
        _decode_json(
            raw,
            name="R3 source-test receipt",
            require_canonical=True,
        )
    )
    return R3SourceTestInspection(
        schema_version=R3_SOURCE_TEST_RECEIPT_SCHEMA,
        artifact_sha256=payload["artifact_sha256"],
        file_sha256=file_sha,
        source_artifact_sha256=payload["source"]["artifact_sha256"],
        source_commit=payload["source"]["commit"],
        verification_suite_sha256=payload["verification"]["suite_sha256"],
    )


def inspect_r3_gate1_bundle(
    path: str | os.PathLike[str],
    *,
    expected_file_sha256: str | None = None,
) -> R3Gate1Inspection:
    """Validate canonical evidence bytes without issuing any capability."""

    source_path = _canonical_real_file(path, name="Gate-1 evidence")
    raw = _stable_bytes(
        source_path,
        name="Gate-1 evidence",
        maximum_bytes=_MAX_EVIDENCE_BYTES,
    )
    file_sha = _sha256(raw)
    if expected_file_sha256 is not None and file_sha != _require_digest(
        expected_file_sha256,
        name="expected Gate-1 file SHA256",
    ):
        raise ValueError("Gate-1 evidence file SHA256 differs from caller expectation")
    payload = _validate_payload(_decode_json(raw, name="Gate-1 evidence", require_canonical=True))
    return R3Gate1Inspection(
        schema_version=GATE1_ARTIFACT_SCHEMA,
        artifact_sha256=payload["artifact_sha256"],
        file_sha256=file_sha,
        source_artifact_sha256=payload["source"]["artifact_sha256"],
        container_artifact_sha256=payload["container"]["artifact_sha256"],
        source_commit=payload["source"]["commit"],
        formal_path_count=payload["source"]["formal_path_count"],
    )


def _source_capability_payload(
    *,
    artifact_sha256: str,
    commit: str,
    tree: str,
    inventory_sha256: str,
    formal_path_count: int,
) -> dict[str, object]:
    return {
        "schema_version": R3_SOURCE_CAPABILITY_SCHEMA,
        "role": SOURCE_ARTIFACT_ROLE,
        "artifact_sha256": artifact_sha256,
        "commit": commit,
        "tree": tree,
        "inventory_sha256": inventory_sha256,
        "formal_path_count": formal_path_count,
    }


@dataclass(frozen=True, slots=True)
class R3SourceCapability:
    """Sealed clean-commit and exact tracked-inventory authority."""

    schema_version: str
    role: str
    artifact_sha256: str
    commit: str
    tree: str
    inventory_sha256: str
    formal_path_count: int
    capability_sha256: str
    _factory_token: InitVar[object] = None
    _seal: object = field(repr=False, compare=False, init=False)

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _SOURCE_FACTORY_TOKEN:
            raise TypeError("R3SourceCapability requires live Gate-1 verification")
        object.__setattr__(self, "_seal", _SOURCE_FACTORY_TOKEN)
        self._validate_structure()

    def _validate_structure(self) -> None:
        if self.schema_version != R3_SOURCE_CAPABILITY_SCHEMA or self.role != SOURCE_ARTIFACT_ROLE:
            raise ValueError("R3 source capability schema/role is invalid")
        payload = _source_capability_payload(
            artifact_sha256=_require_digest(
                self.artifact_sha256,
                name="source artifact SHA256",
            ),
            commit=_require_git_oid(self.commit, name="source commit"),
            tree=_require_git_oid(self.tree, name="source tree"),
            inventory_sha256=_require_digest(
                self.inventory_sha256,
                name="source inventory SHA256",
            ),
            formal_path_count=_require_exact_int(
                self.formal_path_count,
                name="source formal path count",
                minimum=1,
            ),
        )
        if len(self.commit) != len(self.tree):
            raise ValueError("source capability commit/tree formats differ")
        if self.capability_sha256 != _canonical_sha256(payload):
            raise ValueError("R3 source capability self-hash is invalid")

    def validate_integrity(self) -> None:
        if getattr(self, "_seal", None) is not _SOURCE_FACTORY_TOKEN:
            raise TypeError("R3SourceCapability is not live-verifier sealed")
        self._validate_structure()


def _container_capability_payload(
    *,
    artifact_sha256: str,
    canonical_sif_path: str,
    sif_sha256: str,
    sif_size_bytes: int,
    definition_git_blob_sha256: str,
    requirements_lock_git_blob_sha256: str,
    runtime_probe_sha256: str,
    live_runtime_probe_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": R3_CONTAINER_CAPABILITY_SCHEMA,
        "role": CONTAINER_ARTIFACT_ROLE,
        "artifact_sha256": artifact_sha256,
        "canonical_sif_path": canonical_sif_path,
        "sif_sha256": sif_sha256,
        "sif_size_bytes": sif_size_bytes,
        "definition_git_blob_sha256": definition_git_blob_sha256,
        "requirements_lock_git_blob_sha256": requirements_lock_git_blob_sha256,
        "runtime_probe_sha256": runtime_probe_sha256,
        "live_runtime_probe_sha256": live_runtime_probe_sha256,
    }


@dataclass(frozen=True, slots=True)
class R3ContainerCapability:
    """Sealed SIF/definition/lock/runtime authority."""

    schema_version: str
    role: str
    artifact_sha256: str
    canonical_sif_path: str
    sif_sha256: str
    sif_size_bytes: int
    definition_git_blob_sha256: str
    requirements_lock_git_blob_sha256: str
    runtime_probe_sha256: str
    live_runtime_probe_sha256: str
    capability_sha256: str
    _factory_token: InitVar[object] = None
    _seal: object = field(repr=False, compare=False, init=False)

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _CONTAINER_FACTORY_TOKEN:
            raise TypeError("R3ContainerCapability requires live Gate-1 verification")
        object.__setattr__(self, "_seal", _CONTAINER_FACTORY_TOKEN)
        self._validate_structure()

    def _validate_structure(self) -> None:
        if (
            self.schema_version != R3_CONTAINER_CAPABILITY_SCHEMA
            or self.role != CONTAINER_ARTIFACT_ROLE
        ):
            raise ValueError("R3 container capability schema/role is invalid")
        path = Path(
            _require_text(
                self.canonical_sif_path,
                name="capability canonical SIF path",
            )
        )
        if not path.is_absolute() or path.suffix != ".sif":
            raise ValueError("capability canonical SIF path is invalid")
        payload = _container_capability_payload(
            artifact_sha256=_require_digest(
                self.artifact_sha256,
                name="container artifact SHA256",
            ),
            canonical_sif_path=str(path),
            sif_sha256=_require_digest(self.sif_sha256, name="SIF SHA256"),
            sif_size_bytes=_require_exact_int(
                self.sif_size_bytes,
                name="SIF size",
                minimum=1,
            ),
            definition_git_blob_sha256=_require_digest(
                self.definition_git_blob_sha256,
                name="container definition Git blob SHA256",
            ),
            requirements_lock_git_blob_sha256=_require_digest(
                self.requirements_lock_git_blob_sha256,
                name="requirements lock Git blob SHA256",
            ),
            runtime_probe_sha256=_require_digest(
                self.runtime_probe_sha256,
                name="artifact runtime probe SHA256",
            ),
            live_runtime_probe_sha256=_require_digest(
                self.live_runtime_probe_sha256,
                name="live runtime probe SHA256",
            ),
        )
        if self.capability_sha256 != _canonical_sha256(payload):
            raise ValueError("R3 container capability self-hash is invalid")

    def validate_integrity(self) -> None:
        if getattr(self, "_seal", None) is not _CONTAINER_FACTORY_TOKEN:
            raise TypeError("R3ContainerCapability is not live-verifier sealed")
        self._validate_structure()


def _gate1_capability_payload(
    *,
    artifact_sha256: str,
    file_sha256: str,
    source_artifact_sha256: str,
    container_artifact_sha256: str,
    source_test_receipt_artifact_sha256: str,
    source_test_receipt_file_sha256: str,
    verification_suite_sha256: str,
    live_reverification_sha256: str,
    production_relative: str,
) -> dict[str, object]:
    return {
        "schema_version": R3_GATE1_CAPABILITY_SCHEMA,
        "role": GATE1_ARTIFACT_ROLE,
        "artifact_sha256": artifact_sha256,
        "file_sha256": file_sha256,
        "source_artifact_sha256": source_artifact_sha256,
        "container_artifact_sha256": container_artifact_sha256,
        "source_test_receipt_artifact_sha256": (source_test_receipt_artifact_sha256),
        "source_test_receipt_file_sha256": source_test_receipt_file_sha256,
        "verification_suite_sha256": verification_suite_sha256,
        "live_reverification_sha256": live_reverification_sha256,
        "production_relative": production_relative,
    }


@dataclass(frozen=True, slots=True)
class R3Gate1Capability:
    """Sealed live closure of the committed implementation and its tests."""

    schema_version: str
    role: str
    artifact_sha256: str
    file_sha256: str
    source_artifact_sha256: str
    container_artifact_sha256: str
    source_test_receipt_artifact_sha256: str
    source_test_receipt_file_sha256: str
    verification_suite_sha256: str
    live_reverification_sha256: str
    production_relative: str
    capability_sha256: str
    _factory_token: InitVar[object] = None
    _seal: object = field(repr=False, compare=False, init=False)

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _GATE1_FACTORY_TOKEN:
            raise TypeError("R3Gate1Capability requires live Gate-1 verification")
        object.__setattr__(self, "_seal", _GATE1_FACTORY_TOKEN)
        self._validate_structure()

    def _validate_structure(self) -> None:
        if (
            self.schema_version != R3_GATE1_CAPABILITY_SCHEMA
            or self.role != GATE1_ARTIFACT_ROLE
            or self.production_relative != _GATE1_RELATIVE.as_posix()
        ):
            raise ValueError("R3 Gate-1 capability schema/role/namespace is invalid")
        payload = _gate1_capability_payload(
            artifact_sha256=_require_digest(
                self.artifact_sha256,
                name="Gate-1 artifact SHA256",
            ),
            file_sha256=_require_digest(
                self.file_sha256,
                name="Gate-1 file SHA256",
            ),
            source_artifact_sha256=_require_digest(
                self.source_artifact_sha256,
                name="Gate-1 source artifact SHA256",
            ),
            container_artifact_sha256=_require_digest(
                self.container_artifact_sha256,
                name="Gate-1 container artifact SHA256",
            ),
            source_test_receipt_artifact_sha256=_require_digest(
                self.source_test_receipt_artifact_sha256,
                name="Gate-1 source-test receipt artifact SHA256",
            ),
            source_test_receipt_file_sha256=_require_digest(
                self.source_test_receipt_file_sha256,
                name="Gate-1 source-test receipt file SHA256",
            ),
            verification_suite_sha256=_require_digest(
                self.verification_suite_sha256,
                name="Gate-1 verification suite SHA256",
            ),
            live_reverification_sha256=_require_digest(
                self.live_reverification_sha256,
                name="Gate-1 live reverification SHA256",
            ),
            production_relative=self.production_relative,
        )
        if self.capability_sha256 != _canonical_sha256(payload):
            raise ValueError("R3 Gate-1 capability self-hash is invalid")

    def validate_integrity(self) -> None:
        if getattr(self, "_seal", None) is not _GATE1_FACTORY_TOKEN:
            raise TypeError("R3Gate1Capability is not live-verifier sealed")
        self._validate_structure()


@dataclass(frozen=True, slots=True)
class R3Gate1Capabilities:
    """The three independent authorities issued by one live reopen."""

    gate1: R3Gate1Capability
    source: R3SourceCapability
    container: R3ContainerCapability

    def __post_init__(self) -> None:
        if type(self.gate1) is not R3Gate1Capability:
            raise TypeError("gate1 must be an exact R3Gate1Capability")
        if type(self.source) is not R3SourceCapability:
            raise TypeError("source must be an exact R3SourceCapability")
        if type(self.container) is not R3ContainerCapability:
            raise TypeError("container must be an exact R3ContainerCapability")
        self.gate1.validate_integrity()
        self.source.validate_integrity()
        self.container.validate_integrity()
        if (
            self.gate1.source_artifact_sha256 != self.source.artifact_sha256
            or self.gate1.container_artifact_sha256 != self.container.artifact_sha256
        ):
            raise ValueError("Gate-1 capabilities do not close over one evidence bundle")


def _artifact_ref(
    capability: object,
    *,
    exact_type: type,
    schema_version: str,
    role: str,
) -> object:
    if type(capability) is not exact_type:
        raise TypeError(f"expected exact {exact_type.__name__}")
    capability.validate_integrity()
    from .phase2_r3_identity import (
        CONTAINER_ARTIFACT_ROLE as ID_CONTAINER_ROLE,
    )
    from .phase2_r3_identity import (
        CONTAINER_ARTIFACT_SCHEMA as ID_CONTAINER_SCHEMA,
    )
    from .phase2_r3_identity import GATE1_ARTIFACT_ROLE as ID_GATE1_ROLE
    from .phase2_r3_identity import GATE1_ARTIFACT_SCHEMA as ID_GATE1_SCHEMA
    from .phase2_r3_identity import SOURCE_ARTIFACT_ROLE as ID_SOURCE_ROLE
    from .phase2_r3_identity import SOURCE_ARTIFACT_SCHEMA as ID_SOURCE_SCHEMA
    from .phase2_r3_identity import ArtifactRef

    expected = {
        (GATE1_ARTIFACT_SCHEMA, GATE1_ARTIFACT_ROLE): (
            ID_GATE1_SCHEMA,
            ID_GATE1_ROLE,
        ),
        (SOURCE_ARTIFACT_SCHEMA, SOURCE_ARTIFACT_ROLE): (
            ID_SOURCE_SCHEMA,
            ID_SOURCE_ROLE,
        ),
        (CONTAINER_ARTIFACT_SCHEMA, CONTAINER_ARTIFACT_ROLE): (
            ID_CONTAINER_SCHEMA,
            ID_CONTAINER_ROLE,
        ),
    }
    if expected[(schema_version, role)] != (schema_version, role):
        raise RuntimeError("Gate-1 capability schema/role drifted from identity DTOs")
    return ArtifactRef(
        schema_version=schema_version,
        artifact_sha256=capability.artifact_sha256,
        role=role,
    )


def r3_gate1_artifact_ref(capability: R3Gate1Capability) -> object:
    """Return the transport DTO for an exact sealed Gate-1 capability."""

    return _artifact_ref(
        capability,
        exact_type=R3Gate1Capability,
        schema_version=GATE1_ARTIFACT_SCHEMA,
        role=GATE1_ARTIFACT_ROLE,
    )


def r3_source_artifact_ref(capability: R3SourceCapability) -> object:
    """Return the transport DTO for an exact sealed clean-source capability."""

    return _artifact_ref(
        capability,
        exact_type=R3SourceCapability,
        schema_version=SOURCE_ARTIFACT_SCHEMA,
        role=SOURCE_ARTIFACT_ROLE,
    )


def r3_container_artifact_ref(capability: R3ContainerCapability) -> object:
    """Return the transport DTO for an exact sealed container capability."""

    return _artifact_ref(
        capability,
        exact_type=R3ContainerCapability,
        schema_version=CONTAINER_ARTIFACT_SCHEMA,
        role=CONTAINER_ARTIFACT_ROLE,
    )


def _static_container_view(value: Mapping[str, object]) -> dict[str, object]:
    return {
        key: value[key]
        for key in (
            "schema_version",
            "role",
            "canonical_sif_path",
            "sif_sha256",
            "sif_size_bytes",
            "definition",
            "requirements_lock",
        )
    }


def _inside_executable(name: str) -> Path:
    if name == "python":
        candidate = Path(sys.executable).resolve(strict=True)
    else:
        resolved = shutil.which(name)
        if resolved is None:
            raise RuntimeError(f"required in-container executable is absent: {name}")
        candidate = Path(resolved).resolve(strict=True)
    return _canonical_real_file(
        candidate,
        name=f"in-container {name} executable",
    )


def _direct_command(
    inner: tuple[str, ...],
    *,
    python_path: Path,
    bash_path: Path,
    env_path: Path,
) -> tuple[str, ...]:
    command = list(inner)
    if command[0] == "python":
        command[0] = str(python_path)
    elif command[0] == "bash":
        command[0] = str(bash_path)
    elif command[0] == "env":
        command[0] = str(env_path)
        try:
            python_index = command.index("python")
        except ValueError as error:
            raise ValueError("containerized env command omitted Python") from error
        command[python_index] = str(python_path)
    else:
        raise ValueError("unsupported direct in-container command")
    return tuple(command)


def _capture_inside_verification(
    root: Path,
    source: Mapping[str, object],
    runner: _CommandRunner,
) -> dict[str, object]:
    python_path = _inside_executable("python")
    bash_path = _inside_executable("bash")
    env_path = _inside_executable("env")
    receipts: list[dict[str, object]] = []
    for name, inner in _sif_verification_commands(source):
        argv = _direct_command(
            inner,
            python_path=python_path,
            bash_path=bash_path,
            env_path=env_path,
        )
        result = _checked_result(
            runner,
            argv,
            cwd=root,
            name=f"in-container {name}",
        )
        receipts.append(_command_receipt(name, argv, result))
    unsigned: dict[str, object] = {
        "schema_version": R3_IN_CONTAINER_VERIFICATION_SCHEMA,
        "repository_path": str(root),
        "python_path": str(python_path),
        "bash_path": str(bash_path),
        "env_path": str(env_path),
        "commands": receipts,
    }
    return {**unsigned, "suite_sha256": _canonical_sha256(unsigned)}


def _inside_container_identity() -> tuple[str, Path, dict[str, object]]:
    if os.name != "posix":
        raise RuntimeError("in-container Gate-1 verification requires POSIX HPC4")
    if os.environ.get("SLURM_CLUSTER_NAME") != "hpc4":
        raise RuntimeError("in-container Gate-1 verification requires SLURM_CLUSTER_NAME=hpc4")
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if slurm_job_id is None or re.fullmatch(r"[1-9][0-9]*", slurm_job_id) is None:
        raise RuntimeError("in-container Gate-1 verification requires a live Slurm job")

    markers = {
        name: os.environ[name]
        for name in ("APPTAINER_CONTAINER", "SINGULARITY_CONTAINER")
        if os.environ.get(name)
    }
    if not markers:
        raise RuntimeError("current process is not identified as Apptainer/Singularity")
    paths = {
        _canonical_real_file(
            value,
            name=f"{name} exact SIF",
            suffix=".sif",
        )
        for name, value in markers.items()
    }
    if len(paths) != 1:
        raise RuntimeError("container identity environment names different SIF files")
    path = paths.pop()
    engine = "apptainer" if "APPTAINER_CONTAINER" in markers else "singularity"
    marker_payload = {
        "engine": engine,
        "markers": markers,
        "slurm_cluster_name": "hpc4",
        "slurm_job_id": slurm_job_id,
    }
    return (
        engine,
        path,
        {
            **marker_payload,
            "identity_sha256": _canonical_sha256(marker_payload),
        },
    )


def _capture_inside_runtime(
    root: Path,
    *,
    runner: _CommandRunner,
    engine: str,
    container_path: Path,
    environment_identity: Mapping[str, object],
) -> dict[str, object]:
    python_path = _inside_executable("python")
    argv = (str(python_path), "-I", "-c", _CONTAINER_PROBE_SOURCE)
    result = _checked_result(
        runner,
        argv,
        cwd=root,
        name="current in-container Torch/CUDA probe",
    )
    probe = _probe_payload(result.stdout)
    receipt = _command_receipt(
        "current_in_container_torch_cuda_probe",
        argv,
        result,
    )
    unsigned: dict[str, object] = {
        "schema_version": R3_IN_CONTAINER_RUNTIME_SCHEMA,
        "engine": engine,
        "canonical_sif_path": str(container_path),
        "environment_identity": dict(environment_identity),
        "python_path": str(python_path),
        "probe_receipt": receipt,
        "probe": probe,
    }
    return {**unsigned, "runtime_probe_sha256": _canonical_sha256(unsigned)}


def _reverify_inside_current_container(
    payload: Mapping[str, object],
    *,
    root: Path,
    container_path: Path,
    engine: str,
    environment_identity: Mapping[str, object],
    runner: _CommandRunner,
    file_sha256: str,
    source_test_receipt: Mapping[str, object],
    source_test_receipt_file_sha256: str,
) -> tuple[dict[str, object], str]:
    recorded_verification = payload["verification"]
    if (
        source_test_receipt != recorded_verification["source_test_receipt"]
        or source_test_receipt_file_sha256
        != recorded_verification["source_test_receipt_file_sha256"]
    ):
        raise ValueError("caller-pinned source-test receipt differs from Gate-1")
    source = _capture_source(root, runner)
    if source != payload["source"]:
        raise ValueError("current in-container clean source differs from Gate-1")
    recorded_container = payload["container"]
    if str(container_path) != recorded_container["canonical_sif_path"]:
        raise ValueError("current in-container SIF path differs from Gate-1")
    sif_sha, sif_size = _stable_digest(
        container_path,
        name="current in-container exact SIF",
        maximum_bytes=_MAX_SIF_BYTES,
    )
    if (
        sif_sha != recorded_container["sif_sha256"]
        or sif_size != recorded_container["sif_size_bytes"]
    ):
        raise ValueError("current in-container SIF bytes differ from Gate-1")
    runtime = _capture_inside_runtime(
        root,
        runner=runner,
        engine=engine,
        container_path=container_path,
        environment_identity=environment_identity,
    )
    recorded_runtime = recorded_container["runtime"]
    if engine != recorded_runtime["engine"] or runtime["probe"] != recorded_runtime["probe"]:
        raise ValueError("current in-container runtime differs from Gate-1")
    verification = _capture_inside_verification(root, source, runner)
    contract = _contract(source)
    if contract != payload["contract"]:
        raise ValueError("current in-container Gate-1 contract differs from evidence")
    live_receipt = {
        "mode": "inside_exact_sif",
        "gate1_artifact_sha256": payload["artifact_sha256"],
        "gate1_file_sha256": file_sha256,
        "source_artifact_sha256": source["artifact_sha256"],
        "source_test_receipt_file_sha256": source_test_receipt_file_sha256,
        "source_test_receipt_artifact_sha256": source_test_receipt["artifact_sha256"],
        "sif_sha256": sif_sha,
        "sif_size_bytes": sif_size,
        "runtime_probe_sha256": runtime["runtime_probe_sha256"],
        "verification_suite_sha256": verification["suite_sha256"],
    }
    current_container = dict(recorded_container)
    current_runtime = dict(recorded_runtime)
    current_runtime["runtime_probe_sha256"] = runtime["runtime_probe_sha256"]
    current_container["runtime"] = current_runtime
    current = {
        "source": source,
        "verification": verification,
        "container": current_container,
        "contract": contract,
    }
    return current, _canonical_sha256(live_receipt)


def _reverify_current(
    payload: Mapping[str, object],
    *,
    root: Path,
    container_path: Path,
    engine_path: Path,
    runner: _CommandRunner,
    file_sha256: str,
    source_test_receipt: Mapping[str, object],
    source_test_receipt_file_sha256: str,
) -> tuple[dict[str, object], str]:
    recorded_verification = payload["verification"]
    if (
        source_test_receipt != recorded_verification["source_test_receipt"]
        or source_test_receipt_file_sha256
        != recorded_verification["source_test_receipt_file_sha256"]
    ):
        raise ValueError("caller-pinned source-test receipt differs from Gate-1")
    current_source = _capture_source(root, runner)
    if current_source != payload["source"]:
        raise ValueError("current clean commit/tree/blob inventory differs from Gate-1")
    current = _capture_gate1_state(
        root=root,
        container_path=container_path,
        engine_path=engine_path,
        runner=runner,
        source_test_receipt=source_test_receipt,
        source_test_receipt_file_sha256=source_test_receipt_file_sha256,
        prevalidated_source=current_source,
    )
    if current["contract"] != payload["contract"]:
        raise ValueError("current Gate-1 contract coverage differs from the artifact")
    recorded_container = payload["container"]
    current_container = current["container"]
    if _static_container_view(current_container) != _static_container_view(recorded_container):
        raise ValueError("current SIF/definition/lock differs from Gate-1")
    if (
        current_container["runtime"]["engine"] != recorded_container["runtime"]["engine"]
        or current_container["runtime"]["engine_path"]
        != recorded_container["runtime"]["engine_path"]
        or current_container["runtime"]["probe"] != recorded_container["runtime"]["probe"]
    ):
        raise ValueError("current container runtime identity differs from Gate-1")
    recorded_commands = [
        (item["name"], item["argv"]) for item in payload["verification"]["commands"]
    ]
    current_commands = [
        (item["name"], item["argv"]) for item in current["verification"]["commands"]
    ]
    if current_commands != recorded_commands:
        raise ValueError("current verification command set differs from Gate-1")
    live_receipt = {
        "gate1_artifact_sha256": payload["artifact_sha256"],
        "gate1_file_sha256": file_sha256,
        "source_artifact_sha256": current["source"]["artifact_sha256"],
        "source_test_receipt_file_sha256": source_test_receipt_file_sha256,
        "source_test_receipt_artifact_sha256": source_test_receipt["artifact_sha256"],
        "verification_suite_sha256": current["verification"]["suite_sha256"],
        "container_static_sha256": _canonical_sha256(_static_container_view(current_container)),
        "runtime_probe_sha256": current_container["runtime"]["runtime_probe_sha256"],
    }
    return current, _canonical_sha256(live_receipt)


def _issue_capabilities(
    payload: Mapping[str, object],
    current: Mapping[str, object],
    *,
    file_sha256: str,
    live_reverification_sha256: str,
) -> R3Gate1Capabilities:
    payload = _validate_payload(dict(payload))
    if (
        current["source"] != payload["source"]
        or current["contract"] != payload["contract"]
        or _static_container_view(current["container"])
        != _static_container_view(payload["container"])
    ):
        raise ValueError("live Gate-1 state does not close over the evidence")
    source = payload["source"]
    source_payload = _source_capability_payload(
        artifact_sha256=source["artifact_sha256"],
        commit=source["commit"],
        tree=source["tree"],
        inventory_sha256=source["inventory_sha256"],
        formal_path_count=source["formal_path_count"],
    )
    source_capability = R3SourceCapability(
        **source_payload,
        capability_sha256=_canonical_sha256(source_payload),
        _factory_token=_SOURCE_FACTORY_TOKEN,
    )
    container = payload["container"]
    current_container = current["container"]
    container_payload = _container_capability_payload(
        artifact_sha256=container["artifact_sha256"],
        canonical_sif_path=container["canonical_sif_path"],
        sif_sha256=container["sif_sha256"],
        sif_size_bytes=container["sif_size_bytes"],
        definition_git_blob_sha256=container["definition"]["git_blob_sha256"],
        requirements_lock_git_blob_sha256=container["requirements_lock"]["git_blob_sha256"],
        runtime_probe_sha256=container["runtime"]["runtime_probe_sha256"],
        live_runtime_probe_sha256=current_container["runtime"]["runtime_probe_sha256"],
    )
    container_capability = R3ContainerCapability(
        **container_payload,
        capability_sha256=_canonical_sha256(container_payload),
        _factory_token=_CONTAINER_FACTORY_TOKEN,
    )
    gate1_payload = _gate1_capability_payload(
        artifact_sha256=payload["artifact_sha256"],
        file_sha256=file_sha256,
        source_artifact_sha256=source_capability.artifact_sha256,
        container_artifact_sha256=container_capability.artifact_sha256,
        source_test_receipt_artifact_sha256=payload["verification"]["source_test_receipt"][
            "artifact_sha256"
        ],
        source_test_receipt_file_sha256=payload["verification"]["source_test_receipt_file_sha256"],
        verification_suite_sha256=payload["verification"]["suite_sha256"],
        live_reverification_sha256=live_reverification_sha256,
        production_relative=_GATE1_RELATIVE.as_posix(),
    )
    gate1_capability = R3Gate1Capability(
        **gate1_payload,
        capability_sha256=_canonical_sha256(gate1_payload),
        _factory_token=_GATE1_FACTORY_TOKEN,
    )
    result = R3Gate1Capabilities(
        gate1=gate1_capability,
        source=source_capability,
        container=container_capability,
    )
    result.__post_init__()
    return result


def _assert_live_hpc4() -> None:
    if os.name != "posix":
        raise RuntimeError("formal Gate-1 production APIs require POSIX HPC4")
    root, _ = _assert_production_roots()
    result = _checked_result(
        _subprocess_runner,
        ("scontrol", "show", "config"),
        cwd=root,
        name="HPC4 Slurm control-plane probe",
        maximum_bytes=4 * 1024 * 1024,
    )
    try:
        text = result.stdout.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RuntimeError("HPC4 Slurm control-plane output is not UTF-8") from error
    if re.search(r"(?m)^\s*ClusterName\s*=\s*hpc4\s*$", text) is None:
        raise RuntimeError("formal Gate-1 production APIs require ClusterName=hpc4")


def _production_engine() -> Path:
    for name in ("apptainer", "singularity"):
        resolved = shutil.which(name)
        if resolved is not None:
            return _canonical_real_file(
                Path(resolved).resolve(strict=True),
                name="production container runtime",
            )
    raise RuntimeError("neither Apptainer nor Singularity is available on HPC4")


def _require_frozen_production_sif(path: Path) -> None:
    digest, _ = _stable_digest(
        path,
        name="frozen R3 HPC4 SIF",
        maximum_bytes=_MAX_SIF_BYTES,
    )
    if digest != R3_HPC4_IMAGE_SHA256:
        raise ValueError("R3 HPC4 SIF differs from the frozen d6fc044 image")


def _ensure_output_parent(
    relative: Path = _GATE1_RELATIVE,
    *,
    namespace_name: str = "Gate-1",
) -> Path:
    root = PRODUCTION_PROJECT_ROOT
    _require_real_directory(root, name="HPC4 production project root")
    current = root
    for index, component in enumerate(relative.parent.parts):
        current = current / component
        r3_owned = index > 0
        if current.exists() or current.is_symlink():
            _require_output_namespace_directory(
                current,
                name=f"{namespace_name} output namespace",
                r3_owned=r3_owned,
            )
            continue
        parent_info = _require_real_directory(
            current.parent,
            name=f"{namespace_name} output namespace parent",
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
            name=f"new {namespace_name} output namespace",
            r3_owned=r3_owned,
        )
    return current


def _require_output_namespace_directory(
    path: Path,
    *,
    name: str,
    r3_owned: bool,
) -> os.stat_result:
    info = _require_real_directory(path, name=name)
    if path.resolve(strict=True) != path:
        raise ValueError(f"{name} must be a canonical real directory")
    if os.name == "posix":
        mode = stat.S_IMODE(info.st_mode)
        if r3_owned and mode not in {_OUTPUT_DIRECTORY_MODE, 0o2750}:
            raise ValueError(f"{name} must retain mode 0750 (optional setgid accepted)")
        if not mode & stat.S_IWUSR or not mode & stat.S_IXUSR:
            raise PermissionError(f"{name} must be owner-writable and owner-searchable")
        if not os.access(path, os.W_OK | os.X_OK):
            raise PermissionError(f"{name} must be writable and searchable by the current user")
    return info


def _fsync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_exclusive(path: Path, raw: bytes) -> str:
    parent = path.parent
    _require_real_directory(parent, name="Gate-1 publication parent")
    if path.exists() or path.is_symlink():
        raise FileExistsError("refusing to overwrite existing Gate-1 evidence")
    temporary = parent / f".{path.name}.{os.getpid()}.{secrets.token_hex(12)}.tmp"
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_BINARY", 0)
    )
    linked = False
    try:
        descriptor = os.open(temporary, flags, 0o440)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            if os.name == "posix":
                os.fchmod(stream.fileno(), 0o440)
            os.fsync(stream.fileno())
            before = os.fstat(stream.fileno())
            temporary_identity = (before.st_dev, before.st_ino)
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as error:
            raise FileExistsError("refusing to overwrite existing Gate-1 evidence") from error
        linked = True
        published = _stable_bytes(
            path,
            name="published Gate-1 evidence",
            maximum_bytes=_MAX_EVIDENCE_BYTES,
        )
        after = path.stat()
        if (after.st_dev, after.st_ino) != temporary_identity or published != raw:
            raise ValueError("published Gate-1 inode failed descriptor verification")
        if os.name != "posix":
            # Windows maps the creation mode to a read-only file attribute;
            # clear it before removing the temporary hard-link name.
            os.chmod(temporary, 0o600)
        temporary.unlink()
        _fsync_directory(parent)
        return _sha256(raw)
    except Exception:
        if linked and path.exists() and not path.is_symlink():
            if os.name != "posix":
                with suppress(OSError):
                    os.chmod(path, 0o600)
            with suppress(OSError):
                path.unlink()
        raise
    finally:
        if temporary.exists() and not temporary.is_symlink():
            if os.name != "posix":
                with suppress(OSError):
                    os.chmod(temporary, 0o600)
            with suppress(OSError):
                temporary.unlink()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def capture_r3_gate1_source_test_receipt() -> R3SourceTestInspection:
    """Test the fixed clean repository and publish into fixed persistence."""

    root, _ = _assert_production_roots()
    parent = _ensure_output_parent(
        _SOURCE_TEST_RECEIPT_RELATIVE,
        namespace_name="source-test receipt",
    )
    destination = parent / _SOURCE_TEST_RECEIPT_RELATIVE.name
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("refusing to overwrite existing source-test receipt")
    raw = _capture_source_test_receipt_for_inspection(
        root=root,
        runner=_subprocess_runner,
        captured_at_utc=_utc_now(),
    )
    file_sha = _publish_exclusive(destination, raw)
    return inspect_r3_source_test_receipt(
        destination,
        expected_file_sha256=file_sha,
    )


def _load_source_test_receipt(
    *,
    expected_file_sha256: str,
) -> tuple[dict[str, Any], bytes, str]:
    source_path = _production_project_file(
        PRODUCTION_PROJECT_ROOT / _SOURCE_TEST_RECEIPT_RELATIVE,
        name="R3 source-test receipt",
        suffix=".json",
    )
    if stat.S_IMODE(source_path.stat().st_mode) != 0o440:
        raise ValueError("production source-test receipt must retain mode 0440")
    expected = _require_digest(
        expected_file_sha256,
        name="expected source-test receipt file SHA256",
    )
    raw = _stable_bytes(
        source_path,
        name="R3 source-test receipt",
        maximum_bytes=_MAX_EVIDENCE_BYTES,
    )
    file_sha = _sha256(raw)
    if file_sha != expected:
        raise ValueError("source-test receipt file differs from caller expectation")
    payload = _validate_source_test_receipt(
        _decode_json(
            raw,
            name="R3 source-test receipt",
            require_canonical=True,
        )
    )
    return payload, raw, file_sha


def capture_live_r3_gate1_evidence(
    *,
    container: str | os.PathLike[str],
    expected_source_test_receipt_file_sha256: str,
) -> R3Gate1Inspection:
    """Capture and no-overwrite publish Gate-1 evidence; do not authorize."""

    _assert_live_hpc4()
    root = PRODUCTION_REPO_ROOT
    container_path = _production_project_file(
        container,
        name="R3 production container",
        suffix=".sif",
    )
    _require_frozen_production_sif(container_path)
    _, source_test_raw, source_test_file_sha = _load_source_test_receipt(
        expected_file_sha256=expected_source_test_receipt_file_sha256,
    )
    raw = _capture_gate1_evidence_for_inspection(
        root=root,
        container_path=container_path,
        engine_path=_production_engine(),
        runner=_subprocess_runner,
        captured_at_utc=_utc_now(),
        source_test_receipt_raw=source_test_raw,
        expected_source_test_receipt_file_sha256=source_test_file_sha,
    )
    parent = _ensure_output_parent()
    destination = parent / _GATE1_RELATIVE.name
    file_sha = _publish_exclusive(destination, raw)
    return inspect_r3_gate1_bundle(
        destination,
        expected_file_sha256=file_sha,
    )


def _load_production_evidence(
    *,
    expected_file_sha256: str,
) -> tuple[dict[str, Any], str]:
    file_sha = _require_digest(
        expected_file_sha256,
        name="expected Gate-1 file SHA256",
    )
    path = PRODUCTION_PROJECT_ROOT / _GATE1_RELATIVE
    inspect_r3_gate1_bundle(path, expected_file_sha256=file_sha)
    info = path.stat()
    if stat.S_IMODE(info.st_mode) != 0o440:
        raise ValueError("production Gate-1 evidence must retain mode 0440")
    raw = _stable_bytes(
        path,
        name="Gate-1 production evidence",
        maximum_bytes=_MAX_EVIDENCE_BYTES,
    )
    if _sha256(raw) != file_sha:
        raise ValueError("Gate-1 evidence changed after caller-pinned inspection")
    payload = _validate_payload(
        _decode_json(raw, name="Gate-1 production evidence", require_canonical=True)
    )
    return payload, file_sha


def verify_live_r3_gate1_bundle(
    *,
    container: str | os.PathLike[str],
    expected_file_sha256: str,
    expected_source_test_receipt_file_sha256: str,
) -> R3Gate1Capabilities:
    """Reopen caller-pinned evidence and issue live HPC4 capabilities."""

    _assert_live_hpc4()
    payload, file_sha = _load_production_evidence(
        expected_file_sha256=expected_file_sha256,
    )
    container_path = _production_project_file(
        container,
        name="R3 production container",
        suffix=".sif",
    )
    _require_frozen_production_sif(container_path)
    source_test_payload, _, source_test_file_sha = _load_source_test_receipt(
        expected_file_sha256=expected_source_test_receipt_file_sha256,
    )
    current, live_sha = _reverify_current(
        payload,
        root=PRODUCTION_REPO_ROOT,
        container_path=container_path,
        engine_path=_production_engine(),
        runner=_subprocess_runner,
        file_sha256=file_sha,
        source_test_receipt=source_test_payload,
        source_test_receipt_file_sha256=source_test_file_sha,
    )
    return _issue_capabilities(
        payload,
        current,
        file_sha256=file_sha,
        live_reverification_sha256=live_sha,
    )


def verify_live_r3_gate1_in_container(
    *,
    expected_file_sha256: str,
    expected_source_test_receipt_file_sha256: str,
) -> R3Gate1Capabilities:
    """Reverify Gate 1 from the current exact SIF without nested execution.

    The Git-bound SBATCH must enter the SIF with its canonical image path
    visible read-only at the same absolute path, the clean repository visible
    read-only at :data:`PRODUCTION_REPO_ROOT`, and persistent evidence visible
    at :data:`PRODUCTION_PROJECT_ROOT`.  This fixed entry point derives the
    container from Apptainer/Singularity environment identity; it accepts no
    caller-selected root, receipt path, image, or injectable command runner.
    """

    if os.name != "posix":
        raise RuntimeError("formal in-container Gate-1 verification requires POSIX HPC4")
    root, _ = _assert_production_roots()
    expected_module = (root / "src/smart_reward/phase2_r3_gate1.py").resolve(strict=True)
    if Path(__file__).resolve(strict=True) != expected_module:
        raise RuntimeError("Gate-1 verifier was not imported from the bound clean source")
    engine, container_path, environment_identity = _inside_container_identity()
    container_path = _production_project_file(
        container_path,
        name="current in-container exact SIF",
        suffix=".sif",
    )
    _require_frozen_production_sif(container_path)
    payload, file_sha = _load_production_evidence(
        expected_file_sha256=expected_file_sha256,
    )
    source_test_payload, _, source_test_file_sha = _load_source_test_receipt(
        expected_file_sha256=expected_source_test_receipt_file_sha256,
    )
    current, live_sha = _reverify_inside_current_container(
        payload,
        root=root,
        container_path=container_path,
        engine=engine,
        environment_identity=environment_identity,
        runner=_subprocess_runner,
        file_sha256=file_sha,
        source_test_receipt=source_test_payload,
        source_test_receipt_file_sha256=source_test_file_sha,
    )
    return _issue_capabilities(
        payload,
        current,
        file_sha256=file_sha,
        live_reverification_sha256=live_sha,
    )


__all__ = [
    "CONTAINER_ARTIFACT_ROLE",
    "CONTAINER_ARTIFACT_SCHEMA",
    "GATE1_ARTIFACT_ROLE",
    "GATE1_ARTIFACT_SCHEMA",
    "R3ContainerCapability",
    "R3Gate1Capabilities",
    "R3Gate1Capability",
    "R3Gate1Inspection",
    "R3SourceTestInspection",
    "R3SourceCapability",
    "PRODUCTION_PROJECT_ROOT",
    "PRODUCTION_REPO_ROOT",
    "R3_HPC4_IMAGE_SHA256",
    "SOURCE_ARTIFACT_ROLE",
    "SOURCE_ARTIFACT_SCHEMA",
    "capture_r3_gate1_source_test_receipt",
    "capture_live_r3_gate1_evidence",
    "inspect_r3_gate1_bundle",
    "inspect_r3_source_test_receipt",
    "r3_container_artifact_ref",
    "r3_gate1_artifact_ref",
    "r3_source_artifact_ref",
    "verify_live_r3_gate1_bundle",
    "verify_live_r3_gate1_in_container",
]
