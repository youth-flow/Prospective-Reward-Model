#!/usr/bin/env python3
"""Submit one R3 Slurm array only after an immutable held-job ledger exists."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType

try:
    import fcntl
except ModuleNotFoundError:  # pragma: no cover - local Windows test collection only
    fcntl = None  # type: ignore[assignment]


def _stdlib_artifact_module() -> ModuleType:
    """Load the stdlib-only transport module without importing smart_reward."""

    path = (
        Path(__file__).resolve(strict=True).parents[2] / "src/smart_reward/phase2_r3_artifacts.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_phase2_r3_stdlib_artifacts_for_submit_once",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the stdlib-only R3 artifact transport")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_ARTIFACTS = _stdlib_artifact_module()
canonical_json_bytes = _ARTIFACTS.canonical_json_bytes
publish_canonical_artifact = _ARTIFACTS.publish_canonical_artifact
read_canonical_artifact = _ARTIFACTS.read_canonical_artifact

_INTENT_SCHEMA = "phase2-recovery-r3-submission-intent/v2"
_SUBMISSION_SCHEMA = "phase2-recovery-r3-held-submission-receipt/v2"
_RELEASE_SCHEMA = "phase2-recovery-r3-submission-release-receipt/v2"
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_JOB_ID = re.compile(r"[1-9][0-9]*\Z")
_TASKS = re.compile(r"[0-2](?:,[0-2])*\Z")
_INTENT_FIELDS = frozenset(
    {
        "schema_version",
        "role",
        "plan_kind",
        "plan_path",
        "plan_file_sha256",
        "plan_semantic_sha256",
        "array_task_ids",
        "dependency_array_job_ids",
        "attempt_root",
        "git_commit",
        "container_image_file_sha256",
        "sbatch_script_path",
        "sbatch_script_file_sha256",
        "sbatch_command",
        "sbatch_command_sha256",
        "job_name",
        "slurm_comment",
        "submission_intent_sha256",
    }
)
_SUBMISSION_FIELDS = frozenset(
    {
        "schema_version",
        "role",
        "submission_intent_sha256",
        "array_job_id",
        "array_task_ids",
        "dependency_array_job_ids",
        "held_before_ledger_publication",
        "scontrol_show_job",
        "scontrol_show_job_sha256",
        "submission_receipt_sha256",
    }
)
_RELEASE_FIELDS = frozenset(
    {
        "schema_version",
        "role",
        "submission_receipt_sha256",
        "array_job_id",
        "released_after_submission_ledger_fsync",
        "release_observation",
        "release_observation_sha256",
        "release_receipt_sha256",
    }
)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _semantic_sha256(value: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _digest(value: object, *, name: str) -> str:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _canonical_directory(value: Path, *, name: str) -> Path:
    if not value.is_absolute() or value.is_symlink():
        raise ValueError(f"{name} must be an absolute non-symlink directory")
    resolved = value.resolve(strict=True)
    if resolved != value or not value.is_dir():
        raise ValueError(f"{name} must be a canonical directory")
    return value


def _canonical_file(value: Path, *, name: str) -> Path:
    if not value.is_absolute() or value.is_symlink():
        raise ValueError(f"{name} must be an absolute non-symlink file")
    resolved = value.resolve(strict=True)
    if resolved != value or not value.is_file():
        raise ValueError(f"{name} must be a canonical regular file")
    return value


def _read_exact(path: Path, expected: Mapping[str, object]) -> dict[str, object]:
    raw = path.read_bytes()
    artifact = read_canonical_artifact(path, expected_file_sha256=_sha256(raw))
    if artifact.payload != dict(expected):
        raise FileExistsError(f"existing immutable ledger differs: {path}")
    return artifact.payload


def _publish_or_reopen(path: Path, payload: Mapping[str, object]) -> dict[str, object]:
    if path.exists() or path.is_symlink():
        return _read_exact(path, payload)
    return publish_canonical_artifact(path, payload).payload


def _task_ids(value: str) -> list[int]:
    if _TASKS.fullmatch(value) is None:
        raise ValueError("array task IDs must be an ordered CSV subset of 0,1,2")
    result = [int(item) for item in value.split(",")]
    if result != sorted(set(result)):
        raise ValueError("array task IDs must be unique and ordered")
    return result


def _dependency_ids(value: str) -> list[str]:
    if not value:
        return []
    result = value.split(":")
    if any(_JOB_ID.fullmatch(item) is None for item in result) or result != sorted(
        set(result), key=int
    ):
        raise ValueError("dependency job IDs must be unique sorted positive IDs")
    return result


def _single_option(command: Sequence[str], prefix: str, *, name: str) -> str:
    matches = [item[len(prefix) :] for item in command if item.startswith(prefix)]
    if len(matches) != 1 or not matches[0]:
        raise ValueError(f"SBATCH argv must contain exactly one {name}")
    return matches[0]


def _command_array_task_ids(command: Sequence[str]) -> list[int]:
    selector = _single_option(command, "--array=", name="array selector")
    selector = selector.split("%", 1)[0]
    expanded: list[int] = []
    for token in selector.split(","):
        if re.fullmatch(r"[0-2]", token):
            expanded.append(int(token))
            continue
        match = re.fullmatch(r"([0-2])-([0-2])", token)
        if match is None:
            raise ValueError("SBATCH array selector is outside the fixed task set")
        start, stop = (int(item) for item in match.groups())
        if start > stop:
            raise ValueError("SBATCH array selector is descending")
        expanded.extend(range(start, stop + 1))
    if expanded != sorted(set(expanded)):
        raise ValueError("SBATCH array selector is not a unique ordered task set")
    return expanded


def _command_export_environment(command: Sequence[str]) -> dict[str, str]:
    export_spec = _single_option(command, "--export=", name="export specification")
    entries = export_spec.split(",")
    if not entries or entries[0] != "PATH=/usr/bin:/bin":
        raise ValueError("SBATCH export must start from the fixed minimal PATH")
    result: dict[str, str] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError("SBATCH export contains an unbound environment name")
        name, value = entry.split("=", 1)
        if not name or not value or name in result:
            raise ValueError("SBATCH export contains an empty or duplicate binding")
        result[name] = value
    if any(name != "PATH" and not name.startswith("PRORM_R3_") for name in result):
        raise ValueError("SBATCH export contains a non-R3 ambient binding")
    return result


def _validate_sbatch_plan_binding(
    command: Sequence[str],
    *,
    plan_kind: str,
    plan_path: Path,
    plan_file_sha256: str,
    plan_semantic_sha256: str,
    task_ids: Sequence[int],
    dependency_ids: Sequence[str],
    git_commit: str,
    container_image_file_sha256: str,
) -> None:
    if _command_array_task_ids(command) != list(task_ids):
        raise ValueError("SBATCH array selector differs from the ledgered task set")
    dependency_options = [item for item in command if item.startswith("--dependency=")]
    if dependency_ids:
        expected = f"--dependency=afterok:{':'.join(dependency_ids)}"
        if dependency_options != [expected] or "--kill-on-invalid-dep=yes" not in command:
            raise ValueError("SBATCH dependency differs from the ledgered predecessor set")
    elif dependency_options:
        raise ValueError("fresh primary SBATCH argv must not contain a dependency")
    exported = _command_export_environment(command)
    if (
        exported.get("PRORM_R3_GIT_COMMIT") != git_commit
        or exported.get("PRORM_R3_IMAGE_SHA256") != container_image_file_sha256
    ):
        raise ValueError("SBATCH environment differs from clean commit/container intent")
    if plan_kind == "primary":
        names = (
            "PRORM_R3_PRIMARY_SUBMISSION_PLAN",
            "PRORM_R3_PRIMARY_SUBMISSION_PLAN_FILE_SHA256",
            "PRORM_R3_PRIMARY_SUBMISSION_PLAN_SHA256",
        )
    else:
        names = (
            "PRORM_R3_CONTINUATION_PLAN",
            "PRORM_R3_CONTINUATION_PLAN_FILE_SHA256",
            "PRORM_R3_CONTINUATION_PLAN_SHA256",
        )
    expected_values = (
        str(plan_path),
        plan_file_sha256,
        plan_semantic_sha256,
    )
    if tuple(exported.get(name) for name in names) != expected_values:
        raise ValueError("SBATCH environment differs from the exact ledgered plan")


def _validate_plan_identity(
    plan_path: Path,
    *,
    plan_kind: str,
    plan_file_sha256: str,
    plan_semantic_sha256: str,
) -> None:
    artifact = read_canonical_artifact(
        plan_path,
        expected_file_sha256=plan_file_sha256,
    )
    semantic_field = (
        "submission_plan_sha256" if plan_kind == "primary" else "continuation_plan_sha256"
    )
    unsigned = dict(artifact.payload)
    observed_semantic = unsigned.pop(semantic_field, None)
    if observed_semantic != plan_semantic_sha256 or observed_semantic != _semantic_sha256(unsigned):
        raise ValueError("ledgered submission plan identity changed")


def _run(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        check=True,
        capture_output=True,
        text=True,
    )


def _find_existing_job(*, comment: str, job_name: str) -> str | None:
    matches: set[str] = set()
    queries = (
        ["squeue", "-h", "-n", job_name, "-o", "%A|%k"],
        [
            "sacct",
            "-X",
            "-n",
            "-P",
            "-o",
            "JobIDRaw,JobName,Comment",
        ],
    )
    for index, query in enumerate(queries):
        try:
            output = _run(query).stdout
        except (OSError, subprocess.CalledProcessError):
            continue
        for raw_line in output.splitlines():
            fields = [item.strip() for item in raw_line.split("|")]
            if index == 0 and len(fields) >= 2:
                job_id, observed_comment = fields[:2]
                observed_name = job_name
            elif index == 1 and len(fields) >= 3:
                job_id, observed_name, observed_comment = fields[:3]
                job_id = job_id.split("_", 1)[0]
            else:
                continue
            if (
                observed_name == job_name
                and observed_comment == comment
                and _JOB_ID.fullmatch(job_id) is not None
            ):
                matches.add(job_id)
    if len(matches) > 1:
        raise RuntimeError("multiple Slurm allocations match one R3 submission intent")
    return None if not matches else next(iter(matches))


def _validate_held_inspection(
    inspection: str,
    *,
    job_id: str,
    job_name: str,
    comment: str,
) -> None:
    if type(inspection) is not str or not inspection:
        raise ValueError("held Slurm allocation inspection must be non-empty text")
    matching: list[dict[str, str]] = []
    for line in inspection.splitlines():
        fields: dict[str, str] = {}
        for token in line.split():
            if "=" not in token:
                continue
            name, value = token.split("=", 1)
            fields[name] = value
        if fields.get("JobId") == job_id or fields.get("ArrayJobId") == job_id:
            matching.append(fields)
    if len(matching) != 1:
        raise ValueError("held Slurm allocation inspection has no unique array identity")
    fields = matching[0]
    if (
        fields.get("JobName") != job_name
        or fields.get("Comment") != comment
        or fields.get("JobState") != "PENDING"
        or fields.get("Reason") != "JobHeldUser"
        or fields.get("Priority") != "0"
    ):
        raise ValueError("Slurm allocation was not verifiably held for this exact intent")


def _release_or_confirm(job_id: str) -> str:
    try:
        result = _run(["scontrol", "release", job_id])
        return f"released\n{result.stdout}\n{result.stderr}"
    except (OSError, subprocess.CalledProcessError) as release_error:
        observations: list[str] = []
        for query in (
            ["scontrol", "show", "job", "-o", job_id],
            ["sacct", "-X", "-n", "-P", "-j", job_id, "-o", "JobIDRaw,State"],
        ):
            try:
                observed = _run(query).stdout.strip()
            except (OSError, subprocess.CalledProcessError):
                continue
            if observed:
                observations.append(observed)
        joined = "\n".join(observations)
        if not joined or ("JobHeldUser" in joined and "JobState=PENDING" in joined):
            raise RuntimeError(
                "held allocation could not be released or confirmed released"
            ) from release_error
        return f"already-released\n{joined}"


def _validate_receipt(
    path: Path,
    *,
    schema: str,
    semantic_field: str,
    fields: frozenset[str],
) -> dict[str, object]:
    artifact = read_canonical_artifact(path, expected_file_sha256=_sha256(path.read_bytes()))
    payload = artifact.payload
    if set(payload) != fields or payload.get("schema_version") != schema:
        raise ValueError("submission ledger schema is invalid")
    unsigned = dict(payload)
    semantic = unsigned.pop(semantic_field, None)
    if semantic != _semantic_sha256(unsigned):
        raise ValueError("submission ledger semantic hash is invalid")
    return payload


def reopen_submission_ledger(
    project_root: str | os.PathLike[str],
    *,
    plan_semantic_sha256: str,
) -> dict[str, dict[str, object]]:
    """Reopen one complete intent -> held allocation -> release ledger."""

    root = _canonical_directory(Path(project_root), name="project root")
    semantic = _digest(plan_semantic_sha256, name="plan semantic SHA-256")
    ledger_root = root / "runs" / "phase2-recovery-r3" / "submission-ledgers" / semantic
    ledger_root = _canonical_directory(ledger_root, name="submission ledger root")
    intent = _validate_receipt(
        ledger_root / "intent.json",
        schema=_INTENT_SCHEMA,
        semantic_field="submission_intent_sha256",
        fields=_INTENT_FIELDS,
    )
    submission = _validate_receipt(
        ledger_root / "submission.json",
        schema=_SUBMISSION_SCHEMA,
        semantic_field="submission_receipt_sha256",
        fields=_SUBMISSION_FIELDS,
    )
    release = _validate_receipt(
        ledger_root / "release.json",
        schema=_RELEASE_SCHEMA,
        semantic_field="release_receipt_sha256",
        fields=_RELEASE_FIELDS,
    )
    intent_tasks = intent["array_task_ids"]
    intent_dependencies = intent["dependency_array_job_ids"]
    if (
        intent["role"] != "one_exact_r3_array_submission_intent"
        or intent["plan_kind"] not in {"primary", "continuation"}
        or not isinstance(intent_tasks, list)
        or _task_ids(",".join(str(item) for item in intent_tasks)) != intent_tasks
        or not isinstance(intent_dependencies, list)
        or _dependency_ids(":".join(str(item) for item in intent_dependencies))
        != intent_dependencies
        or re.fullmatch(r"[0-9a-f]{40}", str(intent["git_commit"])) is None
    ):
        raise ValueError("submission intent identity fields are invalid")
    for name in (
        "plan_file_sha256",
        "plan_semantic_sha256",
        "container_image_file_sha256",
        "sbatch_script_file_sha256",
        "sbatch_command_sha256",
    ):
        _digest(intent[name], name=f"submission intent {name}")
    command = intent["sbatch_command"]
    if (
        not isinstance(command, list)
        or not command
        or any(type(item) is not str or "\n" in item or "\r" in item for item in command)
        or command[0] != "sbatch"
        or command.count("--hold") != 1
        or "--parsable" not in command
        or "--no-requeue" not in command
        or f"--comment={intent['slurm_comment']}" not in command
        or f"--job-name={intent['job_name']}" not in command
        or command[-1] != intent["sbatch_script_path"]
        or _semantic_sha256({"argv": command}) != intent["sbatch_command_sha256"]
    ):
        raise ValueError("submission intent does not retain its exact SBATCH argv")
    job_id = str(submission["array_job_id"])
    if (
        submission["role"] != "held_array_allocation_bound_to_immutable_intent"
        or release["role"] != "ledgered_r3_array_released_for_execution"
        or _JOB_ID.fullmatch(job_id) is None
        or intent["plan_semantic_sha256"] != semantic
        or submission["submission_intent_sha256"] != intent["submission_intent_sha256"]
        or submission["array_task_ids"] != intent_tasks
        or submission["dependency_array_job_ids"] != intent_dependencies
        or release["submission_receipt_sha256"] != submission["submission_receipt_sha256"]
        or release["array_job_id"] != submission["array_job_id"]
        or release["released_after_submission_ledger_fsync"] is not True
        or submission["held_before_ledger_publication"] is not True
    ):
        raise ValueError("submission ledger chain is inconsistent")
    _digest(
        release["release_observation_sha256"],
        name="submission release observation SHA-256",
    )
    inspection = submission["scontrol_show_job"]
    if (
        type(inspection) is not str
        or _sha256(inspection.encode("utf-8")) != submission["scontrol_show_job_sha256"]
    ):
        raise ValueError("held Slurm inspection bytes differ from their retained digest")
    _validate_held_inspection(
        inspection,
        job_id=job_id,
        job_name=str(intent["job_name"]),
        comment=str(intent["slurm_comment"]),
    )
    release_observation = release["release_observation"]
    if (
        type(release_observation) is not str
        or not (
            release_observation.startswith("released\n")
            or release_observation.startswith("already-released\n")
        )
        or _sha256(release_observation.encode("utf-8")) != release["release_observation_sha256"]
    ):
        raise ValueError("release observation bytes differ from their retained digest")
    plan_path = root.joinpath(*Path(str(intent["plan_path"])).parts)
    plan_path = _canonical_file(plan_path, name="ledgered submission plan")
    if root not in plan_path.parents:
        raise ValueError("ledgered submission plan escapes the project root")
    _validate_plan_identity(
        plan_path,
        plan_kind=str(intent["plan_kind"]),
        plan_file_sha256=str(intent["plan_file_sha256"]),
        plan_semantic_sha256=semantic,
    )
    _validate_sbatch_plan_binding(
        command,
        plan_kind=str(intent["plan_kind"]),
        plan_path=plan_path,
        plan_file_sha256=str(intent["plan_file_sha256"]),
        plan_semantic_sha256=semantic,
        task_ids=intent_tasks,
        dependency_ids=intent_dependencies,
        git_commit=str(intent["git_commit"]),
        container_image_file_sha256=str(intent["container_image_file_sha256"]),
    )
    sbatch_script = _canonical_file(
        Path(str(intent["sbatch_script_path"])),
        name="ledgered SBATCH script",
    )
    if _sha256(sbatch_script.read_bytes()) != intent["sbatch_script_file_sha256"]:
        raise ValueError("ledgered SBATCH script bytes changed")
    return {"intent": intent, "submission": submission, "release": release}


def submit_once(arguments: argparse.Namespace) -> dict[str, object]:
    if fcntl is None:
        raise RuntimeError("formal R3 submission ledger requires POSIX file locking")
    project_root = _canonical_directory(arguments.project_root, name="project root")
    attempt_root = _canonical_directory(arguments.attempt_root, name="attempt root")
    if project_root not in attempt_root.parents:
        raise ValueError("attempt root must be retained under the project root")
    plan_path = _canonical_file(arguments.plan, name="submission plan")
    if project_root not in plan_path.parents:
        raise ValueError("submission plan must be retained under the project root")
    plan_file_sha = _digest(arguments.plan_file_sha256, name="plan file SHA-256")
    if _sha256(plan_path.read_bytes()) != plan_file_sha:
        raise ValueError("submission plan bytes differ from their caller-pinned SHA-256")
    plan_semantic_sha = _digest(
        arguments.plan_semantic_sha256,
        name="plan semantic SHA-256",
    )
    _validate_plan_identity(
        plan_path,
        plan_kind=arguments.plan_kind,
        plan_file_sha256=plan_file_sha,
        plan_semantic_sha256=plan_semantic_sha,
    )
    task_ids = _task_ids(arguments.array_task_ids)
    dependencies = _dependency_ids(arguments.dependency_job_ids)
    if re.fullmatch(r"[0-9a-f]{40}", arguments.git_commit) is None:
        raise ValueError("git commit must be a lowercase 40-character object ID")
    if not arguments.command or arguments.command[0] != "sbatch":
        raise ValueError("submission command must begin with sbatch")
    if "--hold" in arguments.command:
        raise ValueError("caller must not inject --hold")
    sbatch_script = _canonical_file(
        Path(arguments.command[-1]),
        name="SBATCH script",
    )
    command = [
        "sbatch",
        "--hold",
        f"--comment=prorm-r3-{plan_semantic_sha}",
        *arguments.command[1:],
    ]
    _validate_sbatch_plan_binding(
        command,
        plan_kind=arguments.plan_kind,
        plan_path=plan_path,
        plan_file_sha256=plan_file_sha,
        plan_semantic_sha256=plan_semantic_sha,
        task_ids=task_ids,
        dependency_ids=dependencies,
        git_commit=arguments.git_commit,
        container_image_file_sha256=_digest(
            arguments.container_image_file_sha256,
            name="container image file SHA-256",
        ),
    )
    command_sha = _semantic_sha256({"argv": command})
    ledger_root = (
        project_root / "runs" / "phase2-recovery-r3" / "submission-ledgers" / plan_semantic_sha
    )
    ledger_root.mkdir(parents=True, exist_ok=True)
    if ledger_root.resolve(strict=True) != ledger_root or ledger_root.is_symlink():
        raise ValueError("submission ledger root is not canonical")
    lock_path = ledger_root / ".lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        intent_body: dict[str, object] = {
            "schema_version": _INTENT_SCHEMA,
            "role": "one_exact_r3_array_submission_intent",
            "plan_kind": arguments.plan_kind,
            "plan_path": plan_path.relative_to(project_root).as_posix(),
            "plan_file_sha256": plan_file_sha,
            "plan_semantic_sha256": plan_semantic_sha,
            "array_task_ids": task_ids,
            "dependency_array_job_ids": dependencies,
            "attempt_root": attempt_root.relative_to(project_root).as_posix(),
            "git_commit": arguments.git_commit,
            "container_image_file_sha256": _digest(
                arguments.container_image_file_sha256,
                name="container image file SHA-256",
            ),
            "sbatch_script_path": str(sbatch_script),
            "sbatch_script_file_sha256": _sha256(sbatch_script.read_bytes()),
            "sbatch_command": command,
            "sbatch_command_sha256": command_sha,
            "job_name": arguments.job_name,
            "slurm_comment": f"prorm-r3-{plan_semantic_sha}",
        }
        intent = {
            **intent_body,
            "submission_intent_sha256": _semantic_sha256(intent_body),
        }
        intent_path = ledger_root / "intent.json"
        _publish_or_reopen(intent_path, intent)

        submission_path = ledger_root / "submission.json"
        if submission_path.exists():
            submission = _validate_receipt(
                submission_path,
                schema=_SUBMISSION_SCHEMA,
                semantic_field="submission_receipt_sha256",
                fields=_SUBMISSION_FIELDS,
            )
            if (
                submission["submission_intent_sha256"] != intent["submission_intent_sha256"]
                or submission["array_task_ids"] != task_ids
            ):
                raise ValueError("existing submission receipt differs from the intent")
            job_id = str(submission["array_job_id"])
        else:
            job_id = (
                _find_existing_job(
                    comment=str(intent["slurm_comment"]),
                    job_name=arguments.job_name,
                )
                or _run(command).stdout.strip().split(";", 1)[0]
            )
            if _JOB_ID.fullmatch(job_id) is None:
                raise RuntimeError("sbatch returned an invalid array job ID")
            inspection = _run(["scontrol", "show", "job", "-o", job_id]).stdout.strip()
            if not inspection:
                raise RuntimeError("held Slurm allocation could not be inspected")
            _validate_held_inspection(
                inspection,
                job_id=job_id,
                job_name=arguments.job_name,
                comment=str(intent["slurm_comment"]),
            )
            submission_body: dict[str, object] = {
                "schema_version": _SUBMISSION_SCHEMA,
                "role": "held_array_allocation_bound_to_immutable_intent",
                "submission_intent_sha256": intent["submission_intent_sha256"],
                "array_job_id": job_id,
                "array_task_ids": task_ids,
                "dependency_array_job_ids": dependencies,
                "held_before_ledger_publication": True,
                "scontrol_show_job": inspection,
                "scontrol_show_job_sha256": _sha256(inspection.encode("utf-8")),
            }
            submission = {
                **submission_body,
                "submission_receipt_sha256": _semantic_sha256(submission_body),
            }
            publish_canonical_artifact(submission_path, submission)

        release_path = ledger_root / "release.json"
        if release_path.exists():
            release = _validate_receipt(
                release_path,
                schema=_RELEASE_SCHEMA,
                semantic_field="release_receipt_sha256",
                fields=_RELEASE_FIELDS,
            )
            if (
                release["submission_receipt_sha256"] != submission["submission_receipt_sha256"]
                or release["array_job_id"] != job_id
            ):
                raise ValueError("existing release receipt differs from held submission")
        else:
            release_observation = _release_or_confirm(job_id)
            release_body: dict[str, object] = {
                "schema_version": _RELEASE_SCHEMA,
                "role": "ledgered_r3_array_released_for_execution",
                "submission_receipt_sha256": submission["submission_receipt_sha256"],
                "array_job_id": job_id,
                "released_after_submission_ledger_fsync": True,
                "release_observation": release_observation,
                "release_observation_sha256": _sha256(release_observation.encode("utf-8")),
            }
            release = {
                **release_body,
                "release_receipt_sha256": _semantic_sha256(release_body),
            }
            publish_canonical_artifact(release_path, release)
        verified = reopen_submission_ledger(
            project_root,
            plan_semantic_sha256=plan_semantic_sha,
        )
        submission = verified["submission"]
        release = verified["release"]
        return {
            "array_job_id": job_id,
            "ledger_root": str(ledger_root),
            "submission_intent_sha256": intent["submission_intent_sha256"],
            "submission_receipt_sha256": submission["submission_receipt_sha256"],
            "release_receipt_sha256": release["release_receipt_sha256"],
        }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--attempt-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-file-sha256", required=True)
    parser.add_argument("--plan-semantic-sha256", required=True)
    parser.add_argument("--plan-kind", choices=("primary", "continuation"), required=True)
    parser.add_argument("--array-task-ids", required=True)
    parser.add_argument("--dependency-job-ids", default="")
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--container-image-file-sha256", required=True)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command and arguments.command[0] == "--":
        arguments.command = arguments.command[1:]
    result = submit_once(arguments)
    print(f"R3_ARRAY_JOB_ID={result['array_job_id']}", flush=True)
    print(f"R3_SUBMISSION_LEDGER_ROOT={result['ledger_root']}", flush=True)
    print(
        f"R3_SUBMISSION_RECEIPT_SHA256={result['submission_receipt_sha256']}",
        flush=True,
    )
    print(f"R3_RELEASE_RECEIPT_SHA256={result['release_receipt_sha256']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
