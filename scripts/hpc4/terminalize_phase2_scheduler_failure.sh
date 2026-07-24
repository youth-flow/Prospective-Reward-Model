#!/usr/bin/env bash
set -euo pipefail
umask 027

die() {
  echo "error: $*" >&2
  exit 2
}

fsync_file_and_parent() {
  python3 -I -S - "$1" <<'PY'
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
with path.open("rb") as stream:
    os.fsync(stream.fileno())
descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
try:
    os.fsync(descriptor)
finally:
    os.close(descriptor)
PY
}

fsync_directory() {
  python3 -I -S - "$1" <<'PY'
import os
import sys

descriptor = os.open(sys.argv[1], os.O_RDONLY | os.O_DIRECTORY)
try:
    os.fsync(descriptor)
finally:
    os.close(descriptor)
PY
}

fsync_tree() {
  python3 -I -S - "$1" <<'PY'
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
for directory, names, files in os.walk(root, topdown=False, followlinks=False):
    parent = Path(directory)
    for name in (*names, *files):
        path = parent / name
        if path.is_symlink():
            raise SystemExit(f"terminal staging tree contains a symlink: {path}")
    for name in files:
        path = parent / name
        if not path.is_file():
            raise SystemExit(f"terminal staging entry is not a regular file: {path}")
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
    descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
PY
}

if [[ "$#" -ne 7 ]]; then
  die "usage: $0 <confirmatory-overlay.yaml> <seed> <array-job-id> <array-task-id> <attempt-index> <failure-classification.json> <sacct-output.txt>"
fi

overlay="$1"
seed="$2"
array_job_id="$3"
array_task_id="$4"
attempt_index="$5"
classification="$6"
scheduler_evidence="$7"

for name in PRORM_PROJECT_ROOT PRORM_GIT_COMMIT PRORM_CLUSTER_NAME \
  PRORM_PHASE2_ACCEPTED_FREEZE_AGGREGATE_SHA256; do
  [[ -n "${!name:-}" ]] || die "${name} is required"
done
for command_name in flock git python3 realpath sha256sum mktemp mv find; do
  command -v "${command_name}" >/dev/null 2>&1 \
    || die "required command is unavailable: ${command_name}"
done
[[ "${seed}" =~ ^[1-9][0-9]*$ ]] || die "seed must be a positive integer"
[[ "${array_job_id}" =~ ^[1-9][0-9]*$ ]] \
  || die "array job ID must be a positive integer"
[[ "${array_task_id}" =~ ^(0|[1-9][0-9]*)$ ]] \
  || die "array task ID must be a non-negative integer"
[[ "${attempt_index}" =~ ^[1-9][0-9]*$ ]] \
  || die "attempt index must be a positive integer"
[[ "${attempt_index}" = "1" ]] \
  || die "formal Phase-2 retries are disabled; scheduler terminal must be attempt-1"
[[ "${PRORM_GIT_COMMIT}" =~ ^[0-9a-f]{40,64}$ ]] \
  || die "PRORM_GIT_COMMIT is invalid"
[[ "${PRORM_CLUSTER_NAME}" = "hpc4" ]] \
  || die "scheduler reconciliation is locked to the hpc4 cluster"
[[ "${PRORM_PHASE2_ACCEPTED_FREEZE_AGGREGATE_SHA256}" =~ ^[0-9a-f]{64}$ ]] \
  || die "accepted freeze aggregate SHA256 is invalid"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
[[ "$(git -C "${repo_root}" rev-parse --verify HEAD)" = "${PRORM_GIT_COMMIT}" ]] \
  || die "reconciliation must use the submitted Git commit"
[[ -z "$(git -C "${repo_root}" status --porcelain --untracked-files=normal)" ]] \
  || die "reconciliation requires a clean committed worktree"

project_root="$(realpath -e -- "${PRORM_PROJECT_ROOT}")" \
  || die "project root cannot be resolved"
[[ -d "${project_root}" && ! -L "${project_root}" && "${project_root}" != "/" ]] \
  || die "project root is unsafe"
for path in overlay classification scheduler_evidence; do
  resolved="$(realpath -e -- "${!path}")" || die "${path} cannot be resolved"
  [[ -f "${resolved}" && ! -L "${resolved}" ]] \
    || die "${path} must be a non-symlink regular file"
  printf -v "${path}" '%s' "${resolved}"
done
mapfile -t identity < <(
  PYTHONPATH="${repo_root}/src" PYTHONNOUSERSITE=1 \
    python3 -I -S - \
    "${repo_root}/src" "${overlay}" "${seed}" <<'PY'
import sys

sys.path.insert(0, sys.argv[1])
from smart_reward.phase2_config import load_phase2_config_bundle
from smart_reward.phase2_rollout import Phase2Design

bundle = load_phase2_config_bundle(sys.argv[2])
seed = int(sys.argv[3])
config = bundle.config
if (
    config["design"]["stage"] != "confirmatory"
    or config["design"]["formal_eligibility"] is not True
    or config["run"]["confirmatory"] is not True
    or config["run"]["formal_eligibility"] is not True
    or seed not in config["run"]["seeds"]
):
    raise SystemExit("scheduler reconciliation requires a predeclared formal seed")
print(bundle.design_identity)
print(config["design"]["source_config_hash"])
print(Phase2Design.from_phase2_config(config).sha256)
PY
)
[[ "${#identity[@]}" -eq 3 ]] || die "cannot resolve formal identities"
design_sha="${identity[0]}"
base_hash="${identity[1]}"
runtime_sha="${identity[2]}"

scheduler_sha="$(sha256sum -- "${scheduler_evidence}" | awk '{print $1}')"
job_id="${array_job_id}_${array_task_id}"
design_root="${project_root}/runs/phase2-confirmatory/${design_sha}"
seed_root="${design_root}/seed-${seed}"
attempt_parent="${seed_root}/attempt-${attempt_index}"
job_dir="${attempt_parent}/job-${job_id}"
campaign_registry="${design_root}/campaign-registry"
registry_submissions="${campaign_registry}/submissions"
registry_executions="${campaign_registry}/executions"
registry_recoveries="${campaign_registry}/recoveries"
registry_scheduler_terminals="${campaign_registry}/scheduler-terminals"
for directory in \
  "${campaign_registry}" "${registry_submissions}" "${registry_executions}" \
  "${registry_recoveries}" "${registry_scheduler_terminals}"; do
  [[ -d "${directory}" && ! -L "${directory}" ]] \
    || die "campaign registry directory is missing or unsafe: ${directory}"
done
[[ -z "$(find "${registry_recoveries}" -mindepth 1 -maxdepth 1 -print -quit)" ]] \
  || die "formal no-retry campaign contains recovery authorization"
registry_lock="${campaign_registry}/registry.lock"
[[ -f "${registry_lock}" && ! -L "${registry_lock}" ]] \
  || die "campaign registry lock is missing or unsafe"
exec {registry_lock_fd}>> "${registry_lock}"
flock -x "${registry_lock_fd}" \
  || die "failed to acquire the campaign registry reconciliation lock"
[[ -d "${design_root}" && ! -L "${design_root}" ]] \
  || die "formal design root is missing or unsafe"
if [[ ! -e "${seed_root}" && ! -L "${seed_root}" ]]; then
  mkdir -- "${seed_root}"
fi
[[ -d "${seed_root}" && ! -L "${seed_root}" ]] \
  || die "formal seed root is unsafe"
fsync_directory "${seed_root}" \
  || die "formal seed root durability barrier failed"
fsync_directory "${design_root}" \
  || die "formal seed directory-entry durability barrier failed"
if [[ ! -e "${attempt_parent}" && ! -L "${attempt_parent}" ]]; then
  mkdir -- "${attempt_parent}"
fi
[[ -d "${attempt_parent}" && ! -L "${attempt_parent}" ]] \
  || die "formal attempt root is unsafe"
fsync_directory "${attempt_parent}" \
  || die "formal attempt root durability barrier failed"
fsync_directory "${seed_root}" \
  || die "formal attempt directory-entry durability barrier failed"
mapfile -t existing_jobs < <(
  find "${attempt_parent}" -mindepth 1 -maxdepth 1 -type d -name 'job-*' -print
)
if (( ${#existing_jobs[@]} > 1 )); then
  die "attempt has multiple claimed job directories"
fi
if (( ${#existing_jobs[@]} == 1 )) && [[ "${existing_jobs[0]}" != "${job_dir}" ]]; then
  die "attempt is already claimed by a different job directory"
fi
if [[ -d "${job_dir}" && ! -L "${job_dir}" ]]; then
  success_components=0
  for filename in SUCCESS phase2-success-terminal.json; do
    if [[ -f "${job_dir}/${filename}" && ! -L "${job_dir}/${filename}" ]]; then
      success_components=$((success_components + 1))
    elif [[ -e "${job_dir}/${filename}" || -L "${job_dir}/${filename}" ]]; then
      die "existing success component is unsafe: ${filename}"
    fi
  done
  if (( success_components == 2 )); then
    [[ -f "${job_dir}/phase2-attempt-ledger.json" \
      && ! -L "${job_dir}/phase2-attempt-ledger.json" ]] \
      || die "authoritative success bundle lacks its terminal ledger"
    for conflicting in FAILURE_PENDING FAILED SCHEDULER_FAILED \
      phase2-failure-terminal.json; do
      [[ ! -e "${job_dir}/${conflicting}" && ! -L "${job_dir}/${conflicting}" ]] \
        || die "successful attempt has conflicting failure evidence"
    done
    fsync_file_and_parent "${job_dir}/SUCCESS" \
      || die "existing SUCCESS marker durability repair failed"
    fsync_directory "${attempt_parent}" \
      || die "existing SUCCESS directory-entry durability repair failed"
    printf 'Attempt is already atomically terminal-success: %s\n' "${job_dir}"
    exit 0
  elif (( success_components != 0 )); then
    die "authoritative job has a partial success bundle"
  fi
  if [[ -e "${job_dir}/FAILURE_PENDING" \
    || -L "${job_dir}/FAILURE_PENDING" ]]; then
    [[ -f "${job_dir}/FAILURE_PENDING" \
      && ! -L "${job_dir}/FAILURE_PENDING" ]] \
      || die "FAILURE_PENDING marker is unsafe"
    die "canonical FAILURE_PENDING must be closed by the compute failure terminalizer"
  fi
  if [[ -e "${job_dir}/FAILED" || -L "${job_dir}/FAILED" ]]; then
    [[ -f "${job_dir}/FAILED" && ! -L "${job_dir}/FAILED" \
      && -f "${job_dir}/phase2-failure-terminal.json" \
      && ! -L "${job_dir}/phase2-failure-terminal.json" \
      && -f "${job_dir}/phase2-attempt-ledger.json" \
      && ! -L "${job_dir}/phase2-attempt-ledger.json" ]] \
      || die "authoritative compute-failure bundle is partial or unsafe"
    fsync_file_and_parent "${job_dir}/FAILED" \
      || die "existing FAILED marker durability repair failed"
    fsync_directory "${attempt_parent}" \
      || die "existing FAILED directory-entry durability repair failed"
    printf 'Attempt is already atomically terminal-failure: %s\n' "${job_dir}"
    exit 0
  fi
  if [[ -e "${job_dir}/SCHEDULER_FAILED" \
    || -L "${job_dir}/SCHEDULER_FAILED" ]]; then
    [[ -f "${job_dir}/SCHEDULER_FAILED" \
      && ! -L "${job_dir}/SCHEDULER_FAILED" \
      && -f "${job_dir}/phase2-failure-terminal.json" \
      && ! -L "${job_dir}/phase2-failure-terminal.json" \
      && -f "${job_dir}/phase2-attempt-ledger.json" \
      && ! -L "${job_dir}/phase2-attempt-ledger.json" ]] \
      || die "authoritative scheduler-failure bundle is partial or unsafe"
    fsync_file_and_parent "${job_dir}/SCHEDULER_FAILED" \
      || die "existing SCHEDULER_FAILED marker durability repair failed"
    fsync_directory "${attempt_parent}" \
      || die "existing SCHEDULER_FAILED directory-entry durability repair failed"
    printf 'Attempt is already scheduler-terminal-failure: %s\n' "${job_dir}"
    exit 0
  fi
fi

staging="$(mktemp -d "${attempt_parent}/.job-${job_id}.scheduler.tmp.XXXXXX")"
cleanup() {
  if [[ -d "${staging}" ]]; then
    rm -f -- "${staging}"/* 2>/dev/null || true
    rmdir -- "${staging}" 2>/dev/null || true
  fi
}
trap cleanup EXIT
cp -- "${scheduler_evidence}" "${staging}/scheduler-terminal-attestation.raw"
scheduler_copy_sha="$(
  sha256sum -- "${staging}/scheduler-terminal-attestation.raw" | awk '{print $1}'
)"
[[ "${scheduler_copy_sha}" = "${scheduler_sha}" ]] \
  || die "scheduler evidence changed while staging"

python3 -I -S - \
  "${classification}" "${scheduler_evidence}" "${staging}" \
  "${seed}" "${attempt_index}" "${job_id}" "${array_job_id}" \
  "${array_task_id}" "${scheduler_sha}" "${design_sha}" "${base_hash}" \
  "${runtime_sha}" "${PRORM_GIT_COMMIT}" "${PRORM_CLUSTER_NAME}" \
  "${PRORM_PHASE2_ACCEPTED_FREEZE_AGGREGATE_SHA256}" \
  "${registry_submissions}" "${registry_executions}" <<'PY'
import hashlib
import json
import re
import sys
from pathlib import Path

(
    classification_raw,
    scheduler_raw,
    staging_raw,
    seed_raw,
    attempt_index_raw,
    job_id,
    array_job_id,
    array_task_id,
    scheduler_sha,
    design_sha,
    base_hash,
    runtime_sha,
    git_commit,
    cluster_name,
    freeze_sha,
    submissions_raw,
    executions_raw,
) = sys.argv[1:]
seed = int(seed_raw)
attempt_index = int(attempt_index_raw)
staging = Path(staging_raw)


def reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load(path):
    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicates,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant: {value}")
        ),
    )


classification = load(classification_raw)
expected_classification = {
    "failure_stage",
    "failure_class",
    "failure_type",
    "failure_message_sha256",
    "final_outcome_reveal_started",
    "evidence_availability",
}
if not isinstance(classification, dict) or set(classification) != expected_classification:
    raise SystemExit("failure classification fields differ from the locked schema")
for key in ("failure_stage", "failure_class", "failure_type"):
    if not isinstance(classification[key], str) or not re.fullmatch(
        r"[a-z0-9][a-z0-9_]{0,127}", classification[key]
    ):
        raise SystemExit(f"{key} must be an explicit safe token")
if classification["failure_class"] not in {
    "infrastructure",
    "scientific",
    "safety",
    "identity",
    "numerical",
    "software",
}:
    raise SystemExit("failure_class is unsupported")
if (
    not isinstance(classification["failure_message_sha256"], str)
    or re.fullmatch(r"[0-9a-f]{64}", classification["failure_message_sha256"])
    is None
    or not isinstance(classification["final_outcome_reveal_started"], bool)
):
    raise SystemExit("failure classification digest or reveal boundary is invalid")

raw_scheduler = Path(scheduler_raw).read_bytes()
if hashlib.sha256(raw_scheduler).hexdigest() != scheduler_sha:
    raise SystemExit("scheduler terminal evidence changed")
decoded = raw_scheduler.decode("utf-8").strip().splitlines()
if len(decoded) != 1:
    raise SystemExit("scheduler evidence must contain exactly one pipe-delimited root row")
fields = decoded[0].split("|")
if len(fields) != 5:
    raise SystemExit(
        "scheduler evidence must be Cluster|JobIDRaw|JobID|State|ExitCode"
    )
observed_cluster, slurm_job_id, observed_array_task, state, exit_code = fields
state = state.split("+", 1)[0]
if observed_cluster != cluster_name or observed_array_task != job_id:
    raise SystemExit("scheduler evidence belongs to a different array task")
if re.fullmatch(r"[1-9][0-9]*", slurm_job_id) is None:
    raise SystemExit("scheduler evidence has an invalid raw Slurm job ID")
if state not in {
    "BOOT_FAIL",
    "CANCELLED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "TIMEOUT",
}:
    raise SystemExit("scheduler state is not a terminal non-success state")
if re.fullmatch(r"[0-9]+:[0-9]+", exit_code) is None:
    raise SystemExit("scheduler exit code is invalid")

submission_path = Path(submissions_raw) / f"array-{array_job_id}.json"
if submission_path.is_symlink() or not submission_path.is_file():
    raise SystemExit("scheduler terminal lacks its held-array submission registry")
submission = load(submission_path)
entries = submission.get("entries")
selected = (
    [
        entry
        for entry in entries
        if isinstance(entry, dict)
        and entry.get("seed") == seed
        and entry.get("attempt_index") == attempt_index
        and entry.get("array_job_id") == array_job_id
        and entry.get("array_task_id") == int(array_task_id)
    ]
    if isinstance(entries, list)
    else []
)
if (
    submission.get("schema_version") != "prorm-phase2-campaign-submission/v1"
    or submission.get("status") != "committed_while_slurm_held"
    or submission.get("phase2_design_sha256") != design_sha
    or submission.get("base_config_hash") != base_hash
    or submission.get("git_commit") != git_commit
    or submission.get("accepted_freeze_aggregate_sha256") != freeze_sha
    or submission.get("submitted_cluster") != cluster_name
    or cluster_name != "hpc4"
    or submission.get("replacement_seed_allowed") is not False
    or len(selected) != 1
):
    raise SystemExit("scheduler terminal disagrees with the held-array registry")
submission_sha = hashlib.sha256(submission_path.read_bytes()).hexdigest()
execution_path = (
    Path(executions_raw) / f"seed-{seed}-attempt-{attempt_index}.json"
)
execution_sha = None
if execution_path.exists() or execution_path.is_symlink():
    if execution_path.is_symlink() or not execution_path.is_file():
        raise SystemExit("scheduler terminal execution registry is unsafe")
    execution = load(execution_path)
    if (
        execution.get("schema_version")
        != "prorm-phase2-campaign-execution/v1"
        or execution.get("status") != "compute_started_no_requeue"
        or execution.get("seed") != seed
        or execution.get("attempt_index") != attempt_index
        or execution.get("cluster_name") != cluster_name
        or execution.get("array_job_id") != array_job_id
        or execution.get("array_task_id") != int(array_task_id)
        or execution.get("slurm_job_id") != slurm_job_id
        or execution.get("slurm_restart_count") != 0
        or execution.get("phase2_design_sha256") != design_sha
        or execution.get("base_config_hash") != base_hash
        or execution.get("git_commit") != git_commit
        or execution.get("accepted_freeze_aggregate_sha256") != freeze_sha
        or execution.get("replacement_seed_allowed") is not False
        or execution.get("submission", {}).get("sha256") != submission_sha
    ):
        raise SystemExit("scheduler terminal disagrees with execution registry")
    execution_sha = hashlib.sha256(execution_path.read_bytes()).hexdigest()
scheduler_registry = {
    "schema_version": "prorm-phase2-campaign-scheduler-terminal/v1",
    "status": "terminal_non_success_no_retry",
    "seed": seed,
    "attempt_index": attempt_index,
    "phase2_design_sha256": design_sha,
    "base_config_hash": base_hash,
    "phase2_runtime_contract_sha256": runtime_sha,
    "git_commit": git_commit,
    "accepted_freeze_aggregate_sha256": freeze_sha,
    "cluster_name": cluster_name,
    "array_job_id": array_job_id,
    "array_task_id": int(array_task_id),
    "slurm_job_id": slurm_job_id,
    "slurm_restart_count": 0,
    "scheduler_state": state,
    "exit_code": exit_code,
    "scheduler_raw_evidence_sha256": scheduler_sha,
    "registry_submission_sha256": submission_sha,
    "registry_execution_sha256": execution_sha,
    "retry_authorized": False,
    "replacement_seed_allowed": False,
}
with (staging / "scheduler-registry-terminal.json").open(
    "x", encoding="utf-8", newline="\n"
) as stream:
    json.dump(
        scheduler_registry,
        stream,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    stream.write("\n")

if attempt_index != 1:
    raise SystemExit("formal scheduler terminal must be attempt-1")
attempts = []
attempts.append(
    {
        "attempt_index": attempt_index,
        "cluster_name": cluster_name,
        "array_job_id": array_job_id,
        "array_task_id": int(array_task_id),
        "slurm_job_id": slurm_job_id,
        "status": "terminal_failure",
        "final_outcome_reveal_started": classification[
            "final_outcome_reveal_started"
        ],
        "log_sha256": scheduler_sha,
    }
)
ledger = {
    "schema_version": "phase2-seed-attempt-ledger/v3",
    "retry_policy": "single_predeclared_attempt_no_retry",
    "replacement_seed_allowed": False,
    "attempts": attempts,
}
with (staging / "phase2-attempt-ledger.json").open(
    "x", encoding="utf-8", newline="\n"
) as stream:
    json.dump(ledger, stream, allow_nan=False, sort_keys=True, separators=(",", ":"))
    stream.write("\n")
spec = {
    "seed": seed,
    "failure_stage": classification["failure_stage"],
    "failure_class": classification["failure_class"],
    "failure_type": classification["failure_type"],
    "failure_message_sha256": classification["failure_message_sha256"],
    "final_outcome_reveal_started": classification[
        "final_outcome_reveal_started"
    ],
    "attempt_ledger": ledger,
    "capture_method": "scheduler_terminal_reconciliation",
    "evidence_availability": classification["evidence_availability"],
    "evidence_sha256_by_role": {
        "scheduler_terminal_attestation": scheduler_sha,
    },
}
with (staging / "phase2-failure-spec.json").open(
    "x", encoding="utf-8", newline="\n"
) as stream:
    json.dump(spec, stream, allow_nan=False, sort_keys=True, separators=(",", ":"))
    stream.write("\n")
attestation = {
    "schema_version": "phase2-scheduler-terminal-attestation/v1",
    "terminal": True,
    "supports_formal_claim": False,
    "seed": seed,
    "attempt_index": attempt_index,
    "cluster_name": cluster_name,
    "slurm_job_id": slurm_job_id,
    "array_job_id": array_job_id,
    "array_task_id": int(array_task_id),
    "scheduler_state": state,
    "exit_code": exit_code,
    "scheduler_evidence_sha256": scheduler_sha,
    "source_config_hash": base_hash,
    "phase2_design_sha256": design_sha,
    "phase2_runtime_contract_sha256": runtime_sha,
    "git_commit": git_commit,
    "accepted_freeze_aggregate_sha256": freeze_sha,
    "registry_submission_sha256": submission_sha,
    "registry_execution_sha256": execution_sha,
    "final_outcome_reveal_started": classification[
        "final_outcome_reveal_started"
    ],
}
with (staging / "scheduler-terminal-attestation.json").open(
    "x", encoding="utf-8", newline="\n"
) as stream:
    json.dump(
        attestation, stream, allow_nan=False, sort_keys=True, separators=(",", ":")
    )
    stream.write("\n")
PY

scheduler_registry_candidate="${staging}/scheduler-registry-terminal.json"
scheduler_registry_sha="$(
  sha256sum -- "${scheduler_registry_candidate}" | awk '{print $1}'
)"
scheduler_registry_record="${registry_scheduler_terminals}/seed-${seed}-attempt-${attempt_index}.json"
if [[ -e "${scheduler_registry_record}" || -L "${scheduler_registry_record}" ]]; then
  [[ -f "${scheduler_registry_record}" && ! -L "${scheduler_registry_record}" ]] \
    || die "existing scheduler-terminal registry record is unsafe"
  printf '%s  %s\n' "${scheduler_registry_sha}" "${scheduler_registry_record}" \
    | sha256sum --check --status \
    || die "scheduler-terminal registry attempt was already committed differently"
  rm -f -- "${scheduler_registry_candidate}"
else
  mv -T --no-clobber -- \
    "${scheduler_registry_candidate}" "${scheduler_registry_record}" \
    || die "atomic scheduler-terminal registry publication failed"
fi
fsync_file_and_parent "${scheduler_registry_record}" \
  || die "scheduler-terminal registry durability barrier failed"

PYTHONPATH="${repo_root}/src" PYTHONNOUSERSITE=1 \
  python3 -m smart_reward.cli phase2-failure-manifest \
  "${overlay}" "${staging}/phase2-failure-spec.json" \
  "${staging}/phase2-failure-terminal.json" \
  > "${staging}/phase2-failure-manifest.log"

existing_claim=""
if [[ -e "${attempt_parent}/CLAIM" || -L "${attempt_parent}/CLAIM" ]]; then
  [[ -f "${attempt_parent}/CLAIM" && ! -L "${attempt_parent}/CLAIM" ]] \
    || die "existing attempt claim is unsafe"
  existing_claim="${attempt_parent}/CLAIM"
fi
outcome_reveal_marker="${attempt_parent}/OUTCOME_REVEAL_STARTED"
outcome_reveal_marker_sha="none"
if [[ -e "${outcome_reveal_marker}" || -L "${outcome_reveal_marker}" ]]; then
  [[ -f "${outcome_reveal_marker}" && ! -L "${outcome_reveal_marker}" ]] \
    || die "outcome-reveal marker is unsafe"
  outcome_reveal_marker_sha="$(
    sha256sum -- "${outcome_reveal_marker}" | awk '{print $1}'
  )"
fi
python3 -I -S - \
  "${staging}" "${design_sha}" "${base_hash}" "${runtime_sha}" \
  "${seed}" "${attempt_index}" "${scheduler_sha}" \
  "${scheduler_registry_sha}" "${existing_claim}" \
  "${outcome_reveal_marker_sha}" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

staging = Path(sys.argv[1])
(
    design,
    base,
    runtime,
    seed_raw,
    attempt_raw,
    scheduler_sha,
    scheduler_registry_sha,
    existing_claim_raw,
    outcome_marker_sha,
) = sys.argv[2:]
terminal = json.loads(
    (staging / "phase2-failure-terminal.json").read_text(encoding="utf-8")
)
attestation = json.loads(
    (staging / "scheduler-terminal-attestation.json").read_text(encoding="utf-8")
)
ledger_sha = hashlib.sha256(
    (staging / "phase2-attempt-ledger.json").read_bytes()
).hexdigest()
terminal_sha = hashlib.sha256(
    (staging / "phase2-failure-terminal.json").read_bytes()
).hexdigest()
attestation_sha = hashlib.sha256(
    (staging / "scheduler-terminal-attestation.json").read_bytes()
).hexdigest()
if (
    attestation["final_outcome_reveal_started"] is True
    and outcome_marker_sha == "none"
) or (
    attestation["final_outcome_reveal_started"] is False
    and outcome_marker_sha != "none"
):
    raise SystemExit("scheduler evidence disagrees with the durable outcome boundary")
if existing_claim_raw:
    claim_path = Path(existing_claim_raw)
    claim_fields = {}
    for line in claim_path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            raise SystemExit("existing attempt claim is malformed")
        key, value = line.split("=", 1)
        if not key or key in claim_fields:
            raise SystemExit("existing attempt claim repeats an identity")
        claim_fields[key] = value
    if (
        claim_fields.get("schema_version")
        != "prorm-phase2-formal-attempt-claim/v1"
        or claim_fields.get("status") != "CLAIMED"
        or claim_fields.get("cluster_name") != attestation["cluster_name"]
        or claim_fields.get("array_job_id") != attestation["array_job_id"]
        or claim_fields.get("array_task_id") != str(attestation["array_task_id"])
        or claim_fields.get("slurm_job_id") != attestation["slurm_job_id"]
        or claim_fields.get("slurm_restart_count") != "0"
        or claim_fields.get("attempt_index") != attempt_raw
        or claim_fields.get("seed") != seed_raw
        or claim_fields.get("phase2_design_sha256") != design
        or claim_fields.get("base_config_hash") != base
        or claim_fields.get("git_commit") != attestation["git_commit"]
        or claim_fields.get("accepted_freeze_aggregate_sha256")
        != attestation["accepted_freeze_aggregate_sha256"]
        or claim_fields.get("registry_submission_sha256")
        != attestation["registry_submission_sha256"]
        or claim_fields.get("registry_execution_sha256")
        != attestation["registry_execution_sha256"]
    ):
        raise SystemExit("existing compute attempt claim disagrees with scheduler evidence")
    claim_sha = hashlib.sha256(claim_path.read_bytes()).hexdigest()
else:
    claim_lines = [
        "schema_version=prorm-phase2-formal-scheduler-attempt-claim/v1",
        "status=CLAIMED_BY_SCHEDULER_RECONCILIATION",
        f"cluster_name={attestation['cluster_name']}",
        f"array_job_id={attestation['array_job_id']}",
        f"array_task_id={attestation['array_task_id']}",
        f"slurm_job_id={attestation['slurm_job_id']}",
        "slurm_restart_count=0",
        f"attempt_index={attempt_raw}",
        f"seed={seed_raw}",
        f"phase2_design_sha256={design}",
        f"base_config_hash={base}",
        f"git_commit={attestation['git_commit']}",
        "accepted_freeze_aggregate_sha256="
        f"{attestation['accepted_freeze_aggregate_sha256']}",
        f"registry_submission_sha256={attestation['registry_submission_sha256']}",
        "registry_execution_sha256="
        f"{attestation['registry_execution_sha256'] or 'none'}",
        f"registry_scheduler_terminal_sha256={scheduler_registry_sha}",
        "created_at_utc="
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
    ]
    (staging / "CLAIM").write_text(
        "\n".join(claim_lines) + "\n", encoding="utf-8"
    )
    claim_sha = hashlib.sha256((staging / "CLAIM").read_bytes()).hexdigest()
if (
    terminal.get("schema_version") != "phase2-seed-terminal-failure/v2"
    or terminal.get("capture_method") != "scheduler_terminal_reconciliation"
    or terminal.get("seed") != int(seed_raw)
    or terminal.get("source_config_hash") != base
    or terminal.get("phase2_design_sha256") != design
    or terminal.get("phase2_runtime_contract_sha256") != runtime
    or terminal.get("evidence_sha256_by_role", {}).get(
        "scheduler_terminal_attestation"
    )
    != scheduler_sha
):
    raise SystemExit("scheduler-reconciled failure terminal is malformed")
marker = {
    "schema_version": "prorm-phase2-scheduler-terminal-status/v1",
    "status": "SCHEDULER_FAILED",
    "terminal": True,
    "supports_formal_claim": False,
    "seed": int(seed_raw),
    "attempt_index": int(attempt_raw),
    "cluster_name": attestation["cluster_name"],
    "slurm_job_id": attestation["slurm_job_id"],
    "array_job_id": attestation["array_job_id"],
    "array_task_id": attestation["array_task_id"],
    "phase2_design_sha256": design,
    "base_config_hash": base,
    "phase2_runtime_contract_sha256": runtime,
    "registry_submission_sha256": attestation["registry_submission_sha256"],
    "registry_execution_sha256": attestation["registry_execution_sha256"],
    "registry_scheduler_terminal_sha256": scheduler_registry_sha,
    "attempt_claim_sha256": claim_sha,
    "outcome_reveal_marker_sha256": outcome_marker_sha,
    "final_outcome_reveal_started": attestation[
        "final_outcome_reveal_started"
    ],
    "scheduler_terminal_attestation_sha256": attestation_sha,
    "scheduler_raw_evidence_sha256": scheduler_sha,
    "attempt_ledger_sha256": ledger_sha,
    "terminal_manifest_sha256": terminal_sha,
}
with (staging / "SCHEDULER_FAILED").open(
    "x", encoding="utf-8", newline="\n"
) as stream:
    json.dump(marker, stream, allow_nan=False, sort_keys=True, separators=(",", ":"))
    stream.write("\n")
PY

if [[ -z "${existing_claim}" ]]; then
  mv -T --no-clobber -- "${staging}/CLAIM" "${attempt_parent}/CLAIM" \
    || die "atomic scheduler attempt claim publication failed"
  fsync_file_and_parent "${attempt_parent}/CLAIM" \
    || die "scheduler attempt CLAIM durability barrier failed"
fi

if [[ -e "${job_dir}" || -L "${job_dir}" ]]; then
  [[ -d "${job_dir}" && ! -L "${job_dir}" ]] \
    || die "claimed job path is unsafe"
  for conflicting in SUCCESS FAILED phase2-success-terminal.json; do
    [[ ! -e "${job_dir}/${conflicting}" && ! -L "${job_dir}/${conflicting}" ]] \
      || die "claimed job has conflicting terminal evidence: ${conflicting}"
  done
  python3 -I -S \
    "${repo_root}/scripts/hpc4/publish_phase2_terminal_bundle.py" \
    "${staging}" "${job_dir}" SCHEDULER_FAILED \
    scheduler-terminal-attestation.raw scheduler-terminal-attestation.json \
    phase2-attempt-ledger.json phase2-failure-spec.json \
    phase2-failure-terminal.json phase2-failure-manifest.log SCHEDULER_FAILED
else
  fsync_tree "${staging}" \
    || die "scheduler terminal staging durability barrier failed"
  mv -T --no-clobber -- "${staging}" "${job_dir}" \
    || die "atomic scheduler-terminal job claim failed"
  fsync_directory "${attempt_parent}" \
    || die "scheduler terminal directory durability barrier failed"
fi
trap - EXIT
printf 'Scheduler-terminal failure recorded: seed=%s attempt=%s job=%s terminal=%s\n' \
  "${seed}" "${attempt_index}" "${job_id}" \
  "${job_dir}/phase2-failure-terminal.json"
