#!/usr/bin/env bash
#SBATCH --job-name=prorm-phase2-confirmatory
#SBATCH --account=sigroup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1

set -euo pipefail
umask 027

die() {
  echo "error: $*" >&2
  exit 2
}

reject_ambient_controls() {
  local variable=""
  while IFS= read -r variable; do
    case "${variable}" in
      APPTAINER*|SINGULARITY*)
        die "unset exported ${variable}; Phase-2 execution forbids ambient container controls"
        ;;
    esac
  done < <(compgen -e)
}

require_canonical_directory() {
  local name="$1" raw="${!1}" resolved=""
  [[ "${raw}" = /* ]] || die "${name} must be absolute"
  resolved="$(realpath -e -- "${raw}")" \
    || die "${name} does not exist or cannot be resolved: ${raw}"
  [[ -d "${resolved}" && "${resolved}" != "/" ]] \
    || die "${name} must be a non-root directory"
  [[ "${resolved}" = "${raw}" ]] || die "${name} must be canonical: ${raw}"
}

require_canonical_file() {
  local name="$1" raw="${!1}" resolved=""
  [[ "${raw}" = /* ]] || die "${name} must be absolute"
  resolved="$(realpath -e -- "${raw}")" \
    || die "${name} does not exist or cannot be resolved: ${raw}"
  [[ -f "${resolved}" && ! -L "${resolved}" ]] \
    || die "${name} must be a non-symlink regular file: ${raw}"
  [[ "${resolved}" = "${raw}" ]] || die "${name} must be canonical: ${raw}"
}

require_contained() {
  local name="$1" child="$2" parent="$3"
  case "${child}" in
    "${parent}"/*) ;;
    *) die "${name} escaped its required root: ${child}" ;;
  esac
}

roots_overlap() {
  local first="$1" second="$2"
  case "${first}" in
    "${second}"|"${second}"/*) return 0 ;;
  esac
  case "${second}" in
    "${first}"|"${first}"/*) return 0 ;;
  esac
  return 1
}

reject_path_delimiters() {
  local name="$1" value="$2"
  [[ "${value}" != *":"* && "${value}" != *","* && "${value}" != *"="* \
    && "${value}" != *$'\n'* && "${value}" != *$'\r'* ]] \
    || die "${name} contains an unsafe bind/export delimiter"
}

ensure_shared_directory() {
  local label="$1" path="$2" resolved=""
  if [[ ! -e "${path}" && ! -L "${path}" ]]; then
    if ! mkdir -- "${path}"; then
      [[ -e "${path}" || -L "${path}" ]] \
        || die "failed to create ${label}: ${path}"
    fi
  fi
  [[ -d "${path}" && ! -L "${path}" ]] \
    || die "${label} is not a non-symlink directory: ${path}"
  resolved="$(realpath -e -- "${path}")" \
    || die "${label} cannot be resolved: ${path}"
  [[ "${resolved}" = "${path}" ]] \
    || die "${label} is not canonical: ${path}"
  fsync_directory "${path}" \
    || die "failed to durably sync ${label}: ${path}"
  fsync_directory "$(dirname -- "${path}")" \
    || die "failed to durably sync the parent of ${label}: ${path}"
}

fsync_file_and_parent() {
  local path="$1"
  python3 -I -S - "${path}" <<'PY'
import os
import sys
from pathlib import Path


path = Path(sys.argv[1])
flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
file_descriptor = os.open(path, flags)
try:
    os.fsync(file_descriptor)
finally:
    os.close(file_descriptor)
directory_descriptor = os.open(
    path.parent,
    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
)
try:
    os.fsync(directory_descriptor)
finally:
    os.close(directory_descriptor)
PY
}

fsync_tree() {
  local root="$1"
  python3 -I -S - "${root}" <<'PY'
import os
import stat
import sys
from pathlib import Path


root = Path(sys.argv[1])
if root.is_symlink() or not root.is_dir():
    raise SystemExit("tree root is not a non-symlink directory")
directories = [root]
for directory, names, files in os.walk(root, topdown=True, followlinks=False):
    current = Path(directory)
    for name in list(names):
        child = current / name
        if child.is_symlink() and current == root and name == "artifact":
            names.remove(name)
            continue
        if child.is_symlink() or not child.is_dir():
            raise SystemExit(f"unsafe directory in final tree: {child}")
        directories.append(child)
    for name in files:
        child = current / name
        if child.is_symlink():
            raise SystemExit(f"unsafe symlink file in final tree: {child}")
        descriptor = os.open(
            child,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            mode = os.fstat(descriptor).st_mode
            if not stat.S_ISREG(mode):
                raise SystemExit(f"non-regular file in final tree: {child}")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
for directory in reversed(directories):
    descriptor = os.open(
        directory,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
PY
}

fsync_directory() {
  local directory="$1"
  python3 -I -S - "${directory}" <<'PY'
import os
import sys


descriptor = os.open(
    sys.argv[1],
    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
)
try:
    os.fsync(descriptor)
finally:
    os.close(descriptor)
PY
}

for name in \
  PRORM_PROJECT_ROOT PRORM_SCRATCH_ROOT PRORM_IMAGE PRORM_IMAGE_SHA256 \
  PRORM_HF_CACHE PRORM_REPO_ROOT PRORM_PHASE2_OVERLAY \
  PRORM_PHASE2_BASE_CONFIG PRORM_PHASE2_DESIGN_SHA256 \
  PRORM_PHASE2_BASE_CONFIG_HASH PRORM_PHASE2_OVERLAY_FILE_SHA256 \
  PRORM_PHASE2_BASE_FILE_SHA256 PRORM_IDENTITIES_FILE_SHA256 \
  PRORM_GIT_COMMIT PRORM_HF_INVENTORY PRORM_HF_INVENTORY_SHA256 \
  PRORM_PHASE2_ACCEPTED_FREEZE_AGGREGATE \
  PRORM_PHASE2_ACCEPTED_FREEZE_AGGREGATE_SHA256 \
  PRORM_PHASE2_FROZEN_GLOBAL_BETA \
  PRORM_PHASE2_CAMPAIGN_REGISTRY \
  PRORM_PHASE2_ATTEMPT_INDEX; do
  [[ -n "${!name:-}" ]] || die "${name} is required"
done

: "${SLURM_JOB_ID:?Phase-2 confirmatory run must execute inside Slurm}"
: "${SLURM_ARRAY_JOB_ID:?Phase-2 confirmatory run must execute as a Slurm array}"
: "${SLURM_ARRAY_TASK_ID:?Phase-2 confirmatory run requires a Slurm array task ID}"
: "${SLURM_CLUSTER_NAME:?Phase-2 confirmatory run requires a Slurm cluster identity}"
[[ "${SLURM_JOB_ID}" =~ ^[1-9][0-9]*$ ]] || die "SLURM_JOB_ID must be positive"
[[ "${SLURM_ARRAY_JOB_ID}" =~ ^[1-9][0-9]*$ ]] \
  || die "SLURM_ARRAY_JOB_ID must be positive"
[[ "${SLURM_ARRAY_TASK_ID}" =~ ^(0|[1-9][0-9]*)$ ]] \
  || die "SLURM_ARRAY_TASK_ID must be a non-negative integer"
(( SLURM_ARRAY_TASK_ID <= 29 )) \
  || die "SLURM_ARRAY_TASK_ID exceeds the exact-30 formal seed indices"
[[ "${SLURM_CLUSTER_NAME}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] \
  || die "SLURM_CLUSTER_NAME is unsafe for formal attempt evidence"
[[ "${SLURM_CLUSTER_NAME}" = "hpc4" ]] \
  || die "formal Phase-2 compute must execute on Slurm cluster hpc4"
case "${SLURM_RESTART_COUNT:-0}" in
  0) ;;
  *)
    die "automatic Slurm requeue/restart is forbidden for a formal Phase-2 attempt"
    ;;
esac
[[ "${PRORM_PHASE2_ATTEMPT_INDEX}" =~ ^[1-9][0-9]*$ ]] \
  || die "PRORM_PHASE2_ATTEMPT_INDEX must be a positive integer"
[[ "${SLURM_JOB_ACCOUNT:-}" = "sigroup" ]] \
  || die "Phase-2 confirmatory run requires Slurm account sigroup"
case "${SLURM_JOB_PARTITION:-}" in
  gpu-l20) ;;
  *)
    die "Phase-2 confirmatory design is locked to HPC4 gpu-l20"
    ;;
esac
[[ "${SLURM_NNODES:-}" = "1" ]] \
  || die "Phase-2 confirmatory run requires exactly one node"
[[ "${SLURM_NTASKS:-}" = "1" ]] \
  || die "Phase-2 confirmatory run requires exactly one task"
case "${SLURM_GPUS_ON_NODE:-}" in
  1|gpu:1) ;;
  *) die "Phase-2 confirmatory run requires exactly one Slurm GPU on the node" ;;
esac
[[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != *,* ]] \
  || die "CUDA_VISIBLE_DEVICES must identify exactly one allocated GPU"
reject_ambient_controls

for command_name in \
  apptainer flock git python3 realpath sha256sum awk grep mktemp cp mv ln; do
  command -v "${command_name}" >/dev/null 2>&1 \
    || die "required command is unavailable: ${command_name}"
done

require_canonical_directory PRORM_PROJECT_ROOT
require_canonical_directory PRORM_SCRATCH_ROOT
require_canonical_directory PRORM_HF_CACHE
require_canonical_directory PRORM_REPO_ROOT
require_canonical_file PRORM_IMAGE
require_canonical_file PRORM_PHASE2_OVERLAY
require_canonical_file PRORM_PHASE2_BASE_CONFIG
require_canonical_file PRORM_HF_INVENTORY
require_canonical_file PRORM_PHASE2_ACCEPTED_FREEZE_AGGREGATE
require_canonical_directory PRORM_PHASE2_CAMPAIGN_REGISTRY
roots_overlap "${PRORM_PROJECT_ROOT}" "${PRORM_SCRATCH_ROOT}" \
  && die "project and scratch roots may not be equal or nested"
[[ -w "${PRORM_PROJECT_ROOT}" ]] || die "project root is not writable"
[[ -w "${PRORM_SCRATCH_ROOT}" ]] || die "scratch root is not writable"
require_contained PRORM_IMAGE "${PRORM_IMAGE}" "${PRORM_PROJECT_ROOT}"
require_contained PRORM_HF_CACHE "${PRORM_HF_CACHE}" "${PRORM_PROJECT_ROOT}"
require_contained PRORM_HF_INVENTORY "${PRORM_HF_INVENTORY}" "${PRORM_HF_CACHE}"
require_contained \
  PRORM_PHASE2_ACCEPTED_FREEZE_AGGREGATE \
  "${PRORM_PHASE2_ACCEPTED_FREEZE_AGGREGATE}" "${PRORM_PROJECT_ROOT}"
require_contained \
  PRORM_PHASE2_CAMPAIGN_REGISTRY \
  "${PRORM_PHASE2_CAMPAIGN_REGISTRY}" "${PRORM_PROJECT_ROOT}"
require_contained PRORM_PHASE2_OVERLAY "${PRORM_PHASE2_OVERLAY}" "${PRORM_REPO_ROOT}"
require_contained \
  PRORM_PHASE2_BASE_CONFIG "${PRORM_PHASE2_BASE_CONFIG}" "${PRORM_REPO_ROOT}"

[[ "${PRORM_PHASE2_ATTEMPT_INDEX}" = "1" ]] \
  || die "formal no-retry attempt must have attempt_index=1"

for name in \
  PRORM_PROJECT_ROOT PRORM_SCRATCH_ROOT PRORM_IMAGE PRORM_HF_CACHE \
  PRORM_REPO_ROOT PRORM_PHASE2_OVERLAY PRORM_PHASE2_BASE_CONFIG \
  PRORM_HF_INVENTORY PRORM_PHASE2_ACCEPTED_FREEZE_AGGREGATE \
  PRORM_PHASE2_CAMPAIGN_REGISTRY; do
  reject_path_delimiters "${name}" "${!name}"
done
for digest_name in \
  PRORM_IMAGE_SHA256 PRORM_PHASE2_DESIGN_SHA256 \
  PRORM_PHASE2_BASE_CONFIG_HASH PRORM_PHASE2_OVERLAY_FILE_SHA256 \
  PRORM_PHASE2_BASE_FILE_SHA256 PRORM_IDENTITIES_FILE_SHA256 \
  PRORM_HF_INVENTORY_SHA256 PRORM_PHASE2_ACCEPTED_FREEZE_AGGREGATE_SHA256; do
  [[ "${!digest_name}" =~ ^[0-9a-f]{64}$ ]] \
    || die "${digest_name} must be lowercase SHA256"
done
[[ "${PRORM_GIT_COMMIT}" =~ ^[0-9a-f]{40,64}$ ]] \
  || die "PRORM_GIT_COMMIT must be a full lowercase Git object ID"
[[ "${PRORM_PHASE2_FROZEN_GLOBAL_BETA}" =~ ^[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$ ]] \
  || die "PRORM_PHASE2_FROZEN_GLOBAL_BETA must be a positive decimal"
printf '%s  %s\n' "${PRORM_IMAGE_SHA256}" "${PRORM_IMAGE}" \
  | sha256sum --check --status || die "image SHA256 mismatch"
printf '%s  %s\n' "${PRORM_HF_INVENTORY_SHA256}" "${PRORM_HF_INVENTORY}" \
  | sha256sum --check --status || die "HF inventory SHA256 mismatch"
printf '%s  %s\n' "${PRORM_PHASE2_OVERLAY_FILE_SHA256}" "${PRORM_PHASE2_OVERLAY}" \
  | sha256sum --check --status || die "submitted overlay bytes changed"
printf '%s  %s\n' "${PRORM_PHASE2_BASE_FILE_SHA256}" "${PRORM_PHASE2_BASE_CONFIG}" \
  | sha256sum --check --status || die "submitted base config bytes changed"
printf '%s  %s\n' \
  "${PRORM_PHASE2_ACCEPTED_FREEZE_AGGREGATE_SHA256}" \
  "${PRORM_PHASE2_ACCEPTED_FREEZE_AGGREGATE}" \
  | sha256sum --check --status || die "accepted freeze aggregate changed"

submission_head="$(git -C "${PRORM_REPO_ROOT}" rev-parse --verify HEAD)"
[[ "${submission_head}" = "${PRORM_GIT_COMMIT}" ]] \
  || die "submission repository HEAD changed after sbatch"
[[ -z "$(git -C "${PRORM_REPO_ROOT}" status --porcelain --untracked-files=normal)" ]] \
  || die "submission repository became dirty before compute execution"
overlay_relative="$(
  realpath --relative-to="${PRORM_REPO_ROOT}" "${PRORM_PHASE2_OVERLAY}"
)"
base_relative="$(
  realpath --relative-to="${PRORM_REPO_ROOT}" "${PRORM_PHASE2_BASE_CONFIG}"
)"
case "${overlay_relative}" in
  configs/*.yaml) ;;
  *) die "submitted overlay is outside tracked configs/*.yaml" ;;
esac
case "${base_relative}" in
  configs/*.yaml) ;;
  *) die "submitted base config is outside tracked configs/*.yaml" ;;
esac
[[ "${overlay_relative}" != "configs/common_beta_pilot.yaml" ]] \
  || die "formal compute refuses the pilot overlay"
[[ "${base_relative}" != "configs/common_beta_pilot_base.yaml" ]] \
  || die "formal compute refuses the pilot base config"
identity_relative="configs/identities.json"
identity_worktree="${PRORM_REPO_ROOT}/${identity_relative}"
[[ -f "${identity_worktree}" && ! -L "${identity_worktree}" ]] \
  || die "submitted identity file is missing or is a symlink"
printf '%s  %s\n' "${PRORM_IDENTITIES_FILE_SHA256}" "${identity_worktree}" \
  | sha256sum --check --status || die "submitted identity file changed"
for binding in \
  "${overlay_relative}:${PRORM_PHASE2_OVERLAY_FILE_SHA256}" \
  "${base_relative}:${PRORM_PHASE2_BASE_FILE_SHA256}" \
  "${identity_relative}:${PRORM_IDENTITIES_FILE_SHA256}"; do
  relative="${binding%%:*}"
  expected="${binding#*:}"
  observed="$(
    git -C "${PRORM_REPO_ROOT}" cat-file blob \
      "${PRORM_GIT_COMMIT}:${relative}" | sha256sum | awk '{print $1}'
  )"
  [[ "${observed}" = "${expected}" ]] \
    || die "committed bytes differ from submitted identity: ${relative}"
done

execution_id="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
seed=$((20260901 + SLURM_ARRAY_TASK_ID))
project_runs_root="${PRORM_PROJECT_ROOT}/runs"
formal_runs_root="${project_runs_root}/phase2-confirmatory"
formal_design_root="${formal_runs_root}/${PRORM_PHASE2_DESIGN_SHA256}"
seed_root="${formal_design_root}/seed-${seed}"
attempt_root="${seed_root}/attempt-${PRORM_PHASE2_ATTEMPT_INDEX}"
scratch_jobs_root="${PRORM_SCRATCH_ROOT}/phase2-confirmatory-jobs"
expected_campaign_registry="${formal_design_root}/campaign-registry"
[[ "${PRORM_PHASE2_CAMPAIGN_REGISTRY}" = "${expected_campaign_registry}" ]] \
  || die "campaign registry is not addressed by the formal design identity"
registry_submissions="${PRORM_PHASE2_CAMPAIGN_REGISTRY}/submissions"
registry_executions="${PRORM_PHASE2_CAMPAIGN_REGISTRY}/executions"
registry_recoveries="${PRORM_PHASE2_CAMPAIGN_REGISTRY}/recoveries"
registry_scheduler_terminals="${PRORM_PHASE2_CAMPAIGN_REGISTRY}/scheduler-terminals"
registry_staging="${PRORM_PHASE2_CAMPAIGN_REGISTRY}/.staging"
registry_lock="${PRORM_PHASE2_CAMPAIGN_REGISTRY}/registry.lock"
for registry_directory in \
  "${registry_submissions}" "${registry_executions}" "${registry_recoveries}" \
  "${registry_scheduler_terminals}" "${registry_staging}"; do
  [[ -d "${registry_directory}" && ! -L "${registry_directory}" \
    && "$(realpath -e -- "${registry_directory}")" = "${registry_directory}" ]] \
    || die "formal campaign registry directory is missing or unsafe"
done
[[ -f "${registry_lock}" && ! -L "${registry_lock}" ]] \
  || die "formal campaign registry lock is missing or unsafe"
exec {execution_registry_lock_fd}> "${registry_lock}"
flock -x "${execution_registry_lock_fd}" \
  || die "failed to acquire the campaign registry lock for compute admission"
python3 -I -S - "${registry_recoveries}" <<'PY'
import sys
from pathlib import Path


recoveries = Path(sys.argv[1])
if next(recoveries.iterdir(), None) is not None:
    raise SystemExit("formal no-retry campaign recovery registry must remain empty")
PY

mapfile -t registry_submission_info < <(
  python3 -I -S - \
    "${registry_submissions}" "${SLURM_ARRAY_JOB_ID}" \
    "${SLURM_ARRAY_TASK_ID}" "${seed}" "${PRORM_PHASE2_ATTEMPT_INDEX}" \
    "${PRORM_PHASE2_DESIGN_SHA256}" "${PRORM_PHASE2_BASE_CONFIG_HASH}" \
    "${PRORM_GIT_COMMIT}" "${PRORM_PHASE2_ACCEPTED_FREEZE_AGGREGATE_SHA256}" \
    "${SLURM_CLUSTER_NAME}" "${PRORM_IMAGE_SHA256}" \
    "${PRORM_HF_INVENTORY_SHA256}" "${PRORM_PHASE2_OVERLAY_FILE_SHA256}" \
    "${PRORM_PHASE2_BASE_FILE_SHA256}" "${PRORM_IDENTITIES_FILE_SHA256}" \
    "${PRORM_REPO_ROOT}/scripts/hpc4/phase2_confirmatory.sbatch" <<'PY'
import hashlib
import json
import re
import sys
from pathlib import Path


(
    submissions_raw,
    array_job_id,
    task_raw,
    seed_raw,
    attempt_raw,
    design_sha,
    base_hash,
    git_commit,
    freeze_sha,
    cluster,
    image_sha,
    inventory_sha,
    overlay_file_sha,
    base_file_sha,
    identities_file_sha,
    job_script_raw,
) = sys.argv[1:]
task_id = int(task_raw)
seed = int(seed_raw)
attempt_index = int(attempt_raw)


def reject_duplicates(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


paths = sorted(Path(submissions_raw).iterdir())
if len(paths) != 1:
    raise SystemExit("formal compute lacks one unique held-array registry submission")
path = paths[0]
if (
    path.name != f"array-{array_job_id}.json"
    or path.is_symlink()
    or not path.is_file()
):
    raise SystemExit("formal campaign submission path is unsafe or unexpected")
value = json.loads(
    path.read_text(encoding="utf-8"),
    object_pairs_hook=reject_duplicates,
    parse_constant=lambda item: (_ for _ in ()).throw(
        ValueError(f"non-finite JSON constant: {item}")
    ),
)
expected_fields = {
    "schema_version",
    "status",
    "phase2_design_sha256",
    "base_config_hash",
    "git_commit",
    "accepted_freeze_aggregate_sha256",
    "array_job_id",
    "submitted_cluster",
    "array_spec",
    "attempt_index",
    "entries",
    "job_tuple",
    "producer",
    "replacement_seed_allowed",
    "created_at_utc",
}
expected_entries = [
    {
        "seed": 20260901 + index,
        "attempt_index": 1,
        "array_job_id": array_job_id,
        "array_task_id": index,
    }
    for index in range(30)
]
job_tuple = value.get("job_tuple")
expected_job_tuple = {
    "account": "sigroup",
    "partition": "gpu-l20",
    "nodes": 1,
    "tasks": 1,
    "cpus_per_task": 8,
    "memory": "64G",
    "gpus_per_node": 1,
    "no_requeue": True,
    "held_before_registry_commit": True,
    "script": "scripts/hpc4/phase2_confirmatory.sbatch",
    "script_file_sha256": hashlib.sha256(
        Path(job_script_raw).read_bytes()
    ).hexdigest(),
}
expected_producer = {
    "overlay_file_sha256": overlay_file_sha,
    "base_file_sha256": base_file_sha,
    "identities_file_sha256": identities_file_sha,
    "image_sha256": image_sha,
    "hf_inventory_sha256": inventory_sha,
}
if (
    set(value) != expected_fields
    or value.get("schema_version") != "prorm-phase2-campaign-submission/v1"
    or value.get("status") != "committed_while_slurm_held"
    or value.get("phase2_design_sha256") != design_sha
    or value.get("base_config_hash") != base_hash
    or value.get("git_commit") != git_commit
    or value.get("accepted_freeze_aggregate_sha256") != freeze_sha
    or value.get("array_job_id") != array_job_id
    or value.get("submitted_cluster") != cluster
    or cluster != "hpc4"
    or re.fullmatch(r"0-29%[12]", str(value.get("array_spec", ""))) is None
    or value.get("attempt_index") != 1
    or attempt_index != 1
    or value.get("entries") != expected_entries
    or expected_entries[task_id]["seed"] != seed
    or value.get("replacement_seed_allowed") is not False
    or not isinstance(value.get("created_at_utc"), str)
    or not value["created_at_utc"]
    or not isinstance(job_tuple, dict)
    or set(job_tuple) != set(expected_job_tuple) | {"walltime"}
    or any(
        job_tuple.get(key) != expected
        for key, expected in expected_job_tuple.items()
    )
    or re.fullmatch(
        r"(?:[1-9][0-9]*-)?[0-9]{2}:[0-9]{2}:[0-9]{2}",
        str(job_tuple.get("walltime", "")),
    )
    is None
    or value.get("producer") != expected_producer
):
    raise SystemExit("formal compute identity differs from the exact held-array schema")
print(path)
print(hashlib.sha256(path.read_bytes()).hexdigest())
PY
)
[[ "${#registry_submission_info[@]}" -eq 2 ]] \
  || die "failed to resolve the immutable held-array registry submission"
registry_submission="${registry_submission_info[0]}"
registry_submission_sha256="${registry_submission_info[1]}"
[[ "${registry_submission_sha256}" =~ ^[0-9a-f]{64}$ ]] \
  || die "held-array registry submission SHA256 is invalid"

registry_execution="${registry_executions}/seed-${seed}-attempt-${PRORM_PHASE2_ATTEMPT_INDEX}.json"
registry_execution_staging="$(
  mktemp "${registry_staging}/execution-seed-${seed}-attempt-${PRORM_PHASE2_ATTEMPT_INDEX}.XXXXXX"
)"
python3 -I -S - \
  "${registry_execution_staging}" "${seed}" "${PRORM_PHASE2_ATTEMPT_INDEX}" \
  "${PRORM_PHASE2_DESIGN_SHA256}" "${PRORM_PHASE2_BASE_CONFIG_HASH}" \
  "${PRORM_GIT_COMMIT}" "${PRORM_PHASE2_ACCEPTED_FREEZE_AGGREGATE_SHA256}" \
  "${SLURM_CLUSTER_NAME}" "${SLURM_ARRAY_JOB_ID}" "${SLURM_ARRAY_TASK_ID}" \
  "${SLURM_JOB_ID}" "${registry_submission}" "${registry_submission_sha256}" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


(
    output_raw,
    seed_raw,
    attempt_raw,
    design_sha,
    base_hash,
    git_commit,
    freeze_sha,
    cluster,
    array_job_id,
    array_task_id,
    slurm_job_id,
    submission_raw,
    submission_sha,
) = sys.argv[1:]
payload = {
    "schema_version": "prorm-phase2-campaign-execution/v1",
    "status": "compute_started_no_requeue",
    "seed": int(seed_raw),
    "attempt_index": int(attempt_raw),
    "phase2_design_sha256": design_sha,
    "base_config_hash": base_hash,
    "git_commit": git_commit,
    "accepted_freeze_aggregate_sha256": freeze_sha,
    "cluster_name": cluster,
    "array_job_id": array_job_id,
    "array_task_id": int(array_task_id),
    "slurm_job_id": slurm_job_id,
    "slurm_restart_count": 0,
    "submission": {
        "path": Path(submission_raw).name,
        "sha256": submission_sha,
    },
    "replacement_seed_allowed": False,
    "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
}
with Path(output_raw).open("w", encoding="utf-8", newline="\n") as stream:
    json.dump(payload, stream, allow_nan=False, sort_keys=True, separators=(",", ":"))
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
PY
[[ ! -e "${registry_execution}" && ! -L "${registry_execution}" ]] \
  || die "formal seed attempt already has a registered compute execution"
mv -T --no-clobber -- "${registry_execution_staging}" "${registry_execution}" \
  || die "lost the atomic formal execution-registry claim"
fsync_file_and_parent "${registry_execution}" \
  || die "formal execution-registry durability barrier failed"
registry_execution_sha256="$(sha256sum -- "${registry_execution}" | awk '{print $1}')"
[[ "${registry_execution_sha256}" =~ ^[0-9a-f]{64}$ ]] \
  || die "formal execution-registry SHA256 is invalid"
flock -u "${execution_registry_lock_fd}" \
  || die "failed to release the campaign registry lock after compute admission"
exec {execution_registry_lock_fd}>&-

ensure_shared_directory "project runs root" "${project_runs_root}"
ensure_shared_directory "formal Phase-2 runs root" "${formal_runs_root}"
ensure_shared_directory "formal design root" "${formal_design_root}"
ensure_shared_directory "formal seed root" "${seed_root}"
ensure_shared_directory "scratch confirmatory jobs root" "${scratch_jobs_root}"

# This mkdir is the authoritative optional-stopping guard.  An attempt is a
# single-use formal resource, not a directory that may contain competing jobs.
# A hard termination after this point deliberately leaves the claim occupied.
[[ ! -e "${attempt_root}" && ! -L "${attempt_root}" ]] \
  || die "formal seed attempt is already claimed: ${attempt_root}"
mkdir -- "${attempt_root}" \
  || die "lost the atomic formal attempt claim: ${attempt_root}"
[[ -d "${attempt_root}" && ! -L "${attempt_root}" \
  && "$(realpath -e -- "${attempt_root}")" = "${attempt_root}" ]] \
  || die "atomic formal attempt claim is not a canonical directory"
fsync_directory "${attempt_root}" \
  || die "failed to durably sync the claimed formal attempt directory"
fsync_directory "${seed_root}" \
  || die "failed to durably sync the formal attempt claim parent"

attempt_claim="${attempt_root}/CLAIM"
attempt_claim_temporary="$(mktemp "${attempt_root}/.CLAIM.tmp.XXXXXX")"
{
  printf 'schema_version=prorm-phase2-formal-attempt-claim/v1\n'
  printf 'status=CLAIMED\n'
  printf 'cluster_name=%s\n' "${SLURM_CLUSTER_NAME}"
  printf 'array_job_id=%s\n' "${SLURM_ARRAY_JOB_ID}"
  printf 'array_task_id=%s\n' "${SLURM_ARRAY_TASK_ID}"
  printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID}"
  printf 'slurm_restart_count=0\n'
  printf 'attempt_index=%s\n' "${PRORM_PHASE2_ATTEMPT_INDEX}"
  printf 'seed=%s\n' "${seed}"
  printf 'phase2_design_sha256=%s\n' "${PRORM_PHASE2_DESIGN_SHA256}"
  printf 'base_config_hash=%s\n' "${PRORM_PHASE2_BASE_CONFIG_HASH}"
  printf 'git_commit=%s\n' "${PRORM_GIT_COMMIT}"
  printf 'accepted_freeze_aggregate_sha256=%s\n' \
    "${PRORM_PHASE2_ACCEPTED_FREEZE_AGGREGATE_SHA256}"
  printf 'registry_submission_sha256=%s\n' "${registry_submission_sha256}"
  printf 'registry_execution_sha256=%s\n' "${registry_execution_sha256}"
  printf 'created_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${attempt_claim_temporary}"
mv -T --no-clobber -- "${attempt_claim_temporary}" "${attempt_claim}" \
  || die "failed to publish the atomic formal attempt claim"
[[ ! -e "${attempt_claim_temporary}" && ! -L "${attempt_claim_temporary}" \
  && -f "${attempt_claim}" && ! -L "${attempt_claim}" ]] \
  || die "formal attempt claim publication is incomplete"
fsync_file_and_parent "${attempt_claim}" \
  || die "formal attempt claim durability barrier failed"
attempt_claim_sha256="$(sha256sum -- "${attempt_claim}" | awk '{print $1}')"
[[ "${attempt_claim_sha256}" =~ ^[0-9a-f]{64}$ ]] \
  || die "formal attempt claim SHA256 is invalid"

project_run_final="${attempt_root}/job-${execution_id}"
project_run="${attempt_root}/.job-${execution_id}.in-progress-${SLURM_JOB_ID}"
[[ ! -e "${project_run_final}" && ! -L "${project_run_final}" ]] \
  || die "formal attempt unexpectedly contains its unique job directory"
[[ ! -e "${project_run}" && ! -L "${project_run}" ]] \
  || die "formal attempt unexpectedly contains an in-progress job staging directory"
mkdir -- "${project_run}" \
  || die "failed to create the non-authoritative formal job staging directory"
job_dir="${scratch_jobs_root}/${execution_id}-attempt-${PRORM_PHASE2_ATTEMPT_INDEX}"
[[ ! -e "${job_dir}" && ! -L "${job_dir}" ]] \
  || die "refusing to reuse scratch Phase-2 confirmatory job directory: ${job_dir}"
mkdir -- "${job_dir}"

execution_repo="${job_dir}/source"
git clone --quiet --no-hardlinks --no-checkout \
  "${PRORM_REPO_ROOT}" "${execution_repo}"
git -C "${execution_repo}" checkout --quiet --detach "${PRORM_GIT_COMMIT}"
execution_overlay="${execution_repo}/${overlay_relative}"
execution_base="${execution_repo}/${base_relative}"
execution_identities="${execution_repo}/${identity_relative}"
for binding in \
  "${execution_overlay}:${PRORM_PHASE2_OVERLAY_FILE_SHA256}" \
  "${execution_base}:${PRORM_PHASE2_BASE_FILE_SHA256}" \
  "${execution_identities}:${PRORM_IDENTITIES_FILE_SHA256}"; do
  path="${binding%%:*}"
  expected="${binding#*:}"
  [[ -f "${path}" && ! -L "${path}" ]] \
    || die "detached execution input is missing or is a symlink: ${path}"
  printf '%s  %s\n' "${expected}" "${path}" \
    | sha256sum --check --status \
    || die "detached execution input differs from committed identity: ${path}"
done

assert_execution_checkout() {
  [[ "$(git -C "${execution_repo}" rev-parse --verify HEAD)" = "${PRORM_GIT_COMMIT}" ]] \
    || die "detached execution checkout changed commit"
  [[ -z "$(git -C "${execution_repo}" status --porcelain --untracked-files=normal)" ]] \
    || die "detached execution checkout became dirty"
}
assert_execution_checkout

container=(
  apptainer exec --cleanenv --nv
  --no-mount home,cwd,bind-paths
  --pwd "${execution_repo}"
  --bind "${job_dir}:${job_dir},${PRORM_HF_CACHE}:${PRORM_HF_CACHE}"
  --bind "${PRORM_PHASE2_ACCEPTED_FREEZE_AGGREGATE}:${PRORM_PHASE2_ACCEPTED_FREEZE_AGGREGATE}"
  --env "PYTHONPATH=${execution_repo}/src"
  --env "PYTHONNOUSERSITE=1"
  --env "PYTHONDONTWRITEBYTECODE=1"
  --env "HF_HOME=${PRORM_HF_CACHE}"
  --env "HF_HUB_CACHE=${PRORM_HF_CACHE}/hub"
  --env "HF_DATASETS_CACHE=${PRORM_HF_CACHE}/datasets"
  --env "HF_HUB_OFFLINE=1"
  --env "TRANSFORMERS_OFFLINE=1"
  --env "HF_DATASETS_OFFLINE=1"
  --env "HF_HUB_DISABLE_TELEMETRY=1"
  --env "TOKENIZERS_PARALLELISM=false"
  --env "PRORM_IMAGE=${PRORM_IMAGE}"
  --env "PRORM_IMAGE_SHA256=${PRORM_IMAGE_SHA256}"
  --env "PRORM_GIT_COMMIT=${PRORM_GIT_COMMIT}"
  --env "PRORM_HF_INVENTORY=${PRORM_HF_INVENTORY}"
  --env "PRORM_HF_INVENTORY_SHA256=${PRORM_HF_INVENTORY_SHA256}"
)
for variable in \
  SLURM_JOB_ID SLURM_ARRAY_JOB_ID SLURM_ARRAY_TASK_ID SLURM_JOB_NAME \
  SLURM_JOB_ACCOUNT SLURM_CLUSTER_NAME SLURM_JOB_PARTITION \
  SLURM_RESTART_COUNT \
  SLURM_JOB_NODELIST SLURM_NNODES SLURM_NTASKS SLURM_CPUS_PER_TASK \
  SLURM_GPUS SLURM_GPUS_ON_NODE SLURM_PROCID SLURM_LOCALID \
  SLURM_NODEID CUDA_VISIBLE_DEVICES NVIDIA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL; do
  if [[ -v "${variable}" ]]; then
    container+=(--env "${variable}=${!variable}")
  fi
done
container+=("${PRORM_IMAGE}")

phase2_config_check="${job_dir}/phase2-config-check.json"
base_config_check="${job_dir}/base-config-check.json"
assert_execution_checkout
"${container[@]}" python -m smart_reward.cli phase2-config-check \
  "${execution_overlay}" | tee "${phase2_config_check}"
"${container[@]}" python -m smart_reward.cli config-check \
  "${execution_base}" | tee "${base_config_check}"
python3 -I -S - \
  "${phase2_config_check}" "${base_config_check}" \
  "${PRORM_PHASE2_DESIGN_SHA256}" "${PRORM_PHASE2_BASE_CONFIG_HASH}" <<'PY'
import json
import sys
from pathlib import Path


phase2 = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
base = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
expected_design, expected_base = sys.argv[3:]
if (
    phase2.get("status") != "ok"
    or phase2.get("design_stage") != "confirmatory"
    or phase2.get("formal_eligibility") is not True
    or phase2.get("phase2_design_sha256") != expected_design
    or phase2.get("source_config_hash") != expected_base
):
    raise SystemExit("container-computed formal Phase-2 identities are invalid")
if base.get("status") != "ok" or base.get("config_hash") != expected_base:
    raise SystemExit("container-computed formal base identity is invalid")
PY

mapfile -t run_info < <(
  "${container[@]}" python - \
    "${execution_overlay}" "${SLURM_ARRAY_TASK_ID}" <<'PY'
import re
import sys
from smart_reward.phase2_config import (
    PHASE2_CONFIRMATORY_SEEDS,
    load_phase2_config_bundle,
)


bundle = load_phase2_config_bundle(sys.argv[1])
index = int(sys.argv[2])
seeds = bundle.config["run"]["seeds"]
if tuple(seeds) != PHASE2_CONFIRMATORY_SEEDS:
    raise SystemExit("formal seed order differs from the preregistered exact-30 sequence")
if not 0 <= index < len(seeds):
    raise SystemExit(
        f"array index {index} is outside the Phase-2 formal seed count {len(seeds)}"
    )
name = bundle.config["design"]["name"]
if not isinstance(name, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name) is None:
    raise SystemExit("Phase-2 design.name is unsafe for a project path")
print(name)
print(seeds[index])
print(bundle.design_identity)
PY
)
[[ "${#run_info[@]}" -eq 3 ]] || die "failed to resolve the selected formal seed"
run_name="${run_info[0]}"
resolved_seed="${run_info[1]}"
compute_design_sha="${run_info[2]}"
[[ "${resolved_seed}" = "${seed}" ]] \
  || die "array task does not preserve the preregistered formal seed order"
[[ "${compute_design_sha}" = "${PRORM_PHASE2_DESIGN_SHA256}" ]] \
  || die "compute-resolved design identity differs from submission"

inventory_expected="${PRORM_HF_CACHE}/inventories/${PRORM_PHASE2_BASE_CONFIG_HASH}.json"
inventory_expected="$(realpath -e -- "${inventory_expected}")" \
  || die "base-config inventory disappeared before formal execution"
[[ "${inventory_expected}" = "${PRORM_HF_INVENTORY}" ]] \
  || die "submitted inventory is not addressed by the base config identity"

assert_attempt_claim() {
  [[ -f "${attempt_claim}" && ! -L "${attempt_claim}" ]] \
    || die "formal attempt claim disappeared or became unsafe"
  printf '%s  %s\n' "${attempt_claim_sha256}" "${attempt_claim}" \
    | sha256sum --check --status \
    || die "formal attempt claim changed during execution"
}
assert_registry_bindings() {
  printf '%s  %s\n' "${registry_submission_sha256}" "${registry_submission}" \
    | sha256sum --check --status \
    || die "held-array registry submission changed during execution"
  printf '%s  %s\n' "${registry_execution_sha256}" "${registry_execution}" \
    | sha256sum --check --status \
    || die "formal execution-registry record changed during execution"
}
assert_attempt_claim
assert_registry_bindings

attempt_evidence_log="${job_dir}/attempt-evidence.log"
printf 'schema_version=prorm-phase2-attempt-evidence/v1\n' > "${attempt_evidence_log}"
printf 'cluster_name=%s\n' "${SLURM_CLUSTER_NAME}" >> "${attempt_evidence_log}"
printf 'array_job_id=%s\n' "${SLURM_ARRAY_JOB_ID}" >> "${attempt_evidence_log}"
printf 'array_task_id=%s\n' "${SLURM_ARRAY_TASK_ID}" >> "${attempt_evidence_log}"
printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID}" >> "${attempt_evidence_log}"
printf 'slurm_restart_count=0\n' >> "${attempt_evidence_log}"
printf 'seed=%s\n' "${seed}" >> "${attempt_evidence_log}"
printf 'attempt_index=%s\n' "${PRORM_PHASE2_ATTEMPT_INDEX}" >> "${attempt_evidence_log}"
printf 'registry_submission_sha256=%s\n' \
  "${registry_submission_sha256}" >> "${attempt_evidence_log}"
printf 'registry_execution_sha256=%s\n' \
  "${registry_execution_sha256}" >> "${attempt_evidence_log}"
printf 'attempt_claim_sha256=%s\n' "${attempt_claim_sha256}" >> "${attempt_evidence_log}"

record_stage() {
  printf 'stage=%s time=%s\n' \
    "$1" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${attempt_evidence_log}"
}

materialized_this_job=0
final_outcome_reveal_started=0
success_terminal_ready=0
success_ledger_source=""
success_validation_receipt="${job_dir}/phase2-success-validation-receipt.json"
outcome_reveal_marker="${attempt_root}/OUTCOME_REVEAL_STARTED"
outcome_reveal_marker_sha256="none"
outcome_boundary_publication_started=0

refresh_outcome_reveal_state() {
  if [[ -f "${outcome_reveal_marker}" && ! -L "${outcome_reveal_marker}" ]]; then
    final_outcome_reveal_started=1
    outcome_reveal_marker_sha256="$(
      sha256sum -- "${outcome_reveal_marker}" | awk '{print $1}'
    )" || {
      outcome_reveal_marker_sha256="none"
      return 1
    }
    [[ "${outcome_reveal_marker_sha256}" =~ ^[0-9a-f]{64}$ ]] || {
      outcome_reveal_marker_sha256="none"
      return 1
    }
    return 0
  fi
  if [[ -e "${outcome_reveal_marker}" || -L "${outcome_reveal_marker}" ]]; then
    # An unsafe or partial authoritative marker is conservatively post-outcome.
    final_outcome_reveal_started=1
    outcome_reveal_marker_sha256="none"
    return 1
  fi
  final_outcome_reveal_started=0
  outcome_reveal_marker_sha256="none"
  return 0
}

refresh_success_terminal_state() {
  local candidate="${job_dir}/phase2-attempt-ledger.success-candidate.json"
  local canonical="${job_dir}/phase2-attempt-ledger.json"
  local terminal="${job_dir}/phase2-success-terminal.json"
  local result="${job_dir}/phase2-result.json"
  local rollout="${job_dir}/phase2-result.rollouts.jsonl"
  local ledger_info=()
  success_terminal_ready=0
  success_ledger_source=""
  if [[ ! -e "${success_validation_receipt}" \
    && ! -L "${success_validation_receipt}" ]]; then
    return 0
  fi
  [[ -f "${success_validation_receipt}" \
    && ! -L "${success_validation_receipt}" ]] || return 1
  mapfile -t ledger_info < <(
    python3 -I -S - \
      "${success_validation_receipt}" "${candidate}" "${canonical}" \
      "${terminal}" "${result}" "${rollout}" \
      "${seed}" "${PRORM_PHASE2_DESIGN_SHA256}" \
      "${PRORM_PHASE2_BASE_CONFIG_HASH}" "${PRORM_GIT_COMMIT}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path


(
    receipt_raw,
    candidate_raw,
    canonical_raw,
    terminal_raw,
    result_raw,
    rollout_raw,
    seed_raw,
    design_sha,
    base_hash,
    git_commit,
) = sys.argv[1:]


def digest(path):
    value = Path(path)
    if value.is_symlink() or not value.is_file():
        raise SystemExit(f"sealed-success source is missing or unsafe: {value}")
    return hashlib.sha256(value.read_bytes()).hexdigest()


receipt = json.loads(Path(receipt_raw).read_text(encoding="utf-8"))
expected_fields = {
    "schema_version",
    "status",
    "seed",
    "phase2_design_sha256",
    "base_config_hash",
    "git_commit",
    "attempt_ledger_sha256",
    "success_terminal_sha256",
    "result_sha256",
    "rollout_sha256",
}
if (
    set(receipt) != expected_fields
    or receipt.get("schema_version") != "prorm-phase2-success-validation-receipt/v1"
    or receipt.get("status") != "VALIDATED_SUCCESS_READY_FOR_ATOMIC_JOB_PUBLICATION"
    or receipt.get("seed") != int(seed_raw)
    or receipt.get("phase2_design_sha256") != design_sha
    or receipt.get("base_config_hash") != base_hash
    or receipt.get("git_commit") != git_commit
    or receipt.get("success_terminal_sha256") != digest(terminal_raw)
    or receipt.get("result_sha256") != digest(result_raw)
    or receipt.get("rollout_sha256") != digest(rollout_raw)
):
    raise SystemExit("sealed-success validation receipt is malformed or stale")
candidate = Path(candidate_raw)
canonical = Path(canonical_raw)
available = [
    path
    for path in (candidate, canonical)
    if path.exists() or path.is_symlink()
]
if len(available) != 1 or receipt.get("attempt_ledger_sha256") != digest(available[0]):
    raise SystemExit("sealed-success receipt lacks one exact attempt ledger source")
print(available[0])
PY
  ) || return 1
  [[ "${#ledger_info[@]}" -eq 1 ]] || return 1
  success_ledger_source="${ledger_info[0]}"
  success_terminal_ready=1
}

atomic_sync_file() {
  local source="$1" filename="$2" destination="${project_run}/${filename}" temporary=""
  local source_sha="" destination_sha=""
  [[ -f "${source}" && ! -L "${source}" ]] || return 0
  if [[ -e "${destination}" || -L "${destination}" ]]; then
    [[ -f "${destination}" && ! -L "${destination}" ]] || {
      echo "refusing unsafe existing run evidence: ${destination}" >&2
      return 1
    }
    source_sha="$(sha256sum -- "${source}" | awk '{print $1}')" || return 1
    destination_sha="$(sha256sum -- "${destination}" | awk '{print $1}')" || return 1
    [[ "${source_sha}" = "${destination_sha}" ]] || {
      echo "refusing to overwrite differing run evidence: ${destination}" >&2
      return 1
    }
    return 0
  fi
  temporary="$(mktemp "${project_run}/.${filename}.tmp.XXXXXX")" || return 1
  cp -p -- "${source}" "${temporary}" || {
    rm -f -- "${temporary}"
    return 1
  }
  mv -T --no-clobber -- "${temporary}" "${destination}" || {
    rm -f -- "${temporary}"
    return 1
  }
}

build_attempt_ledger() {
  local terminal_status="$1" reveal="$2"
  local output="${3:-${job_dir}/phase2-attempt-ledger.json}"
  local log_sha=""
  [[ ! -e "${output}" && ! -L "${output}" ]] \
    || { echo "refusing to overwrite attempt ledger" >&2; return 1; }
  log_sha="$(sha256sum -- "${attempt_evidence_log}" | awk '{print $1}')" || return 1
  python3 -I -S - \
    "${output}" "${PRORM_PHASE2_ATTEMPT_INDEX}" "${SLURM_CLUSTER_NAME}" \
    "${SLURM_ARRAY_JOB_ID}" "${SLURM_ARRAY_TASK_ID}" "${SLURM_JOB_ID}" \
    "${terminal_status}" "${reveal}" "${log_sha}" <<'PY'
import json
import sys
from pathlib import Path


(
    output_raw,
    attempt_index_raw,
    cluster_name,
    array_job_id,
    array_task_id_raw,
    job_id,
    terminal_status,
    reveal_raw,
    log_sha,
) = sys.argv[1:]
attempt_index = int(attempt_index_raw)
array_task_id = int(array_task_id_raw)
reveal = reveal_raw == "true"
if terminal_status not in {
    "terminal_failure",
    "success_result",
}:
    raise SystemExit("unsupported attempt terminal status")
if terminal_status == "success_result" and not reveal:
    raise SystemExit("successful formal attempt must reveal its final outcome")
if len(log_sha) != 64 or any(character not in "0123456789abcdef" for character in log_sha):
    raise SystemExit("attempt evidence log SHA256 is invalid")
if attempt_index != 1:
    raise SystemExit("formal no-retry ledger must contain exactly one first attempt")
attempts = []
attempts.append(
    {
        "attempt_index": attempt_index,
        "cluster_name": cluster_name,
        "array_job_id": array_job_id,
        "array_task_id": array_task_id,
        "slurm_job_id": job_id,
        "status": terminal_status,
        "final_outcome_reveal_started": reveal,
        "log_sha256": log_sha,
    }
)
payload = {
    "schema_version": "phase2-seed-attempt-ledger/v3",
    "retry_policy": "single_predeclared_attempt_no_retry",
    "replacement_seed_allowed": False,
    "attempts": attempts,
}
with Path(output_raw).open("x", encoding="utf-8", newline="\n") as stream:
    json.dump(payload, stream, allow_nan=False, sort_keys=True, separators=(",", ":"))
    stream.write("\n")
PY
}

atomic_write_marker() {
  local marker="$1" workload_exit="$2" final_exit="$3" ledger_sha="$4" temporary=""
  [[ ! -e "${project_run}/SUCCESS" && ! -L "${project_run}/SUCCESS" ]] || return 1
  [[ ! -e "${project_run}/FAILURE_PENDING" \
    && ! -L "${project_run}/FAILURE_PENDING" ]] || return 1
  temporary="$(mktemp "${project_run}/.${marker}.tmp.XXXXXX")" || return 1
  {
    printf 'schema_version=prorm-phase2-confirmatory-run-status/v1\n'
    printf 'status=%s\n' "${marker}"
    printf 'workload_exit_code=%s\n' "${workload_exit}"
    printf 'final_exit_code=%s\n' "${final_exit}"
    printf 'array_job_id=%s\n' "${SLURM_ARRAY_JOB_ID}"
    printf 'array_task_id=%s\n' "${SLURM_ARRAY_TASK_ID}"
    printf 'cluster_name=%s\n' "${SLURM_CLUSTER_NAME}"
    printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID}"
    printf 'slurm_restart_count=0\n'
    printf 'attempt_index=%s\n' "${PRORM_PHASE2_ATTEMPT_INDEX}"
    printf 'seed=%s\n' "${seed}"
    printf 'phase2_design_sha256=%s\n' "${PRORM_PHASE2_DESIGN_SHA256}"
    printf 'base_config_hash=%s\n' "${PRORM_PHASE2_BASE_CONFIG_HASH}"
    printf 'git_commit=%s\n' "${PRORM_GIT_COMMIT}"
    printf 'accepted_freeze_aggregate_sha256=%s\n' \
      "${PRORM_PHASE2_ACCEPTED_FREEZE_AGGREGATE_SHA256}"
    printf 'registry_submission_sha256=%s\n' "${registry_submission_sha256}"
    printf 'registry_execution_sha256=%s\n' "${registry_execution_sha256}"
    printf 'attempt_claim_sha256=%s\n' "${attempt_claim_sha256}"
    printf 'outcome_reveal_marker_sha256=%s\n' \
      "${outcome_reveal_marker_sha256}"
    printf 'attempt_ledger_sha256=%s\n' "${ledger_sha}"
    if (( final_outcome_reveal_started )); then
      printf 'final_outcome_reveal_started=true\n'
    else
      printf 'final_outcome_reveal_started=false\n'
    fi
    printf 'created_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "${temporary}" || {
    rm -f -- "${temporary}"
    return 1
  }
  mv -T --no-clobber -- "${temporary}" "${project_run}/${marker}" || {
    rm -f -- "${temporary}"
    return 1
  }
  fsync_file_and_parent "${project_run}/${marker}" || return 1
}

on_pre_outcome_timeout() {
  refresh_outcome_reveal_state || true
  record_stage "slurm_usr1_received"
  exit 75
}
trap on_pre_outcome_timeout USR1

finalize_run() {
  local workload_exit=$? sync_exit=0 final_exit marker filename
  local source_sha="" destination_sha=""
  local ledger_status ledger_reveal ledger_sha="" durable_success_ready=0
  local final_registry_lock_fd=""
  local scheduler_terminal_record=""
  trap - EXIT USR1
  set +e
  refresh_outcome_reveal_state || sync_exit=1
  refresh_success_terminal_state || sync_exit=1
  if (( success_terminal_ready == 1 )) \
    && [[ "${success_ledger_source}" != "${job_dir}/phase2-attempt-ledger.json" ]]; then
    if mv -T --no-clobber -- \
      "${success_ledger_source}" "${job_dir}/phase2-attempt-ledger.json"; then
      success_ledger_source="${job_dir}/phase2-attempt-ledger.json"
    else
      sync_exit=1
    fi
  fi
  if [[ ! -f "${job_dir}/phase2-attempt-ledger.json" ]]; then
    record_stage "finalize_started"
    ledger_status="terminal_failure"
    if (( final_outcome_reveal_started )); then
      ledger_reveal="true"
    else
      ledger_reveal="false"
    fi
    build_attempt_ledger "${ledger_status}" "${ledger_reveal}" || sync_exit=1
  fi
  if [[ -f "${job_dir}/phase2-attempt-ledger.json" ]]; then
    ledger_sha="$(
      sha256sum -- "${job_dir}/phase2-attempt-ledger.json" | awk '{print $1}'
    )"
  else
    ledger_sha="$(printf 'missing-ledger' | sha256sum | awk '{print $1}')"
    sync_exit=1
  fi
  for filename in \
    phase2-config-check.json base-config-check.json \
    hf-inventory-verification.json gpu-check.log env-report.log \
    run-manifest.json materialize.log memory-materialize.json \
    artifact-materialization.json artifact-verification.json \
    phase2-run.log memory-phase2.json phase2-result.json \
    phase2-result.rollouts.jsonl phase2-output-verification.json \
    attempt-evidence.log phase2-attempt-ledger.json phase2-success-spec.json \
    phase2-success-manifest.log phase2-success-validation-receipt.json; do
    atomic_sync_file "${job_dir}/${filename}" "${filename}" || sync_exit=1
  done
  if (( success_terminal_ready == 1 )); then
    for filename in \
      phase2-config-check.json base-config-check.json \
      hf-inventory-verification.json gpu-check.log env-report.log \
      run-manifest.json artifact-materialization.json \
      artifact-verification.json phase2-run.log memory-phase2.json \
      phase2-result.json phase2-result.rollouts.jsonl \
      phase2-output-verification.json attempt-evidence.log \
      phase2-attempt-ledger.json phase2-success-spec.json \
      phase2-success-manifest.log phase2-success-validation-receipt.json \
      materialize.log memory-materialize.json; do
      [[ -f "${project_run}/${filename}" && ! -L "${project_run}/${filename}" \
        && -f "${job_dir}/${filename}" && ! -L "${job_dir}/${filename}" ]] || {
        echo "successful formal workload is missing core evidence: ${filename}" >&2
        sync_exit=1
        continue
      }
      source_sha="$(sha256sum -- "${job_dir}/${filename}" | awk '{print $1}')"
      destination_sha="$(sha256sum -- "${project_run}/${filename}" | awk '{print $1}')"
      [[ "${source_sha}" = "${destination_sha}" ]] || {
        echo "successful formal workload core evidence changed: ${filename}" >&2
        sync_exit=1
      }
    done
    [[ -L "${project_run}/artifact" && -d "${project_run}/artifact" ]] || {
      echo "successful formal workload is missing its artifact link" >&2
      sync_exit=1
    }
    (( materialized_this_job == 1 )) || {
      echo "successful formal workload did not materialize its own formal artifact" >&2
      sync_exit=1
    }
    if (( sync_exit == 0 )); then
      atomic_sync_file \
        "${job_dir}/phase2-success-terminal.json" \
        "phase2-success-terminal.json" || sync_exit=1
    fi
    if (( sync_exit == 0 )) \
      && [[ -f "${project_run}/phase2-success-terminal.json" \
        && ! -L "${project_run}/phase2-success-terminal.json" ]]; then
      source_sha="$(
        sha256sum -- "${job_dir}/phase2-success-terminal.json" | awk '{print $1}'
      )"
      destination_sha="$(
        sha256sum -- "${project_run}/phase2-success-terminal.json" | awk '{print $1}'
      )"
      if [[ "${source_sha}" = "${destination_sha}" ]]; then
        durable_success_ready=1
      else
        sync_exit=1
      fi
    fi
  elif (( workload_exit == 0 )); then
    echo "formal workload exited zero without a success terminal manifest" >&2
    sync_exit=1
  fi
  if (( sync_exit != 0 )); then
    echo "formal job prepublication validation failed; leaving only non-authoritative staging for scheduler reconciliation" >&2
    exit 1
  fi
  if (( durable_success_ready == 1 )); then
    final_exit=0
    marker="SUCCESS"
  else
    final_exit="${workload_exit}"
    if (( final_exit == 0 )); then final_exit=1; fi
    marker="FAILURE_PENDING"
  fi
  if ! atomic_write_marker "${marker}" "${workload_exit}" "${final_exit}" "${ledger_sha}"; then
    echo "failed to install atomic ${marker} marker in ${project_run}" >&2
    final_exit=1
  elif ! fsync_tree "${project_run}"; then
    echo "failed to durably sync the complete formal job staging tree" >&2
    final_exit=1
  elif ! exec {final_registry_lock_fd}> "${registry_lock}"; then
    echo "failed to open the formal campaign registry lock for job publication" >&2
    final_exit=1
  elif ! flock -x "${final_registry_lock_fd}"; then
    echo "failed to acquire the formal campaign registry lock for job publication" >&2
    final_exit=1
  else
    scheduler_terminal_record="${registry_scheduler_terminals}/seed-${seed}-attempt-${PRORM_PHASE2_ATTEMPT_INDEX}.json"
    if [[ -e "${scheduler_terminal_record}" || -L "${scheduler_terminal_record}" ]]; then
      echo "scheduler reconciliation already owns this formal attempt terminal" >&2
      final_exit=1
    elif ! mv -T --no-clobber -- "${project_run}" "${project_run_final}"; then
      echo "failed to atomically publish the complete formal job directory" >&2
      final_exit=1
    elif ! fsync_directory "${attempt_root}"; then
      echo "formal job directory rename completed but parent fsync failed" >&2
      final_exit=1
    fi
    flock -u "${final_registry_lock_fd}" >/dev/null 2>&1 || final_exit=1
    exec {final_registry_lock_fd}>&-
  fi
  exit "${final_exit}"
}
trap finalize_run EXIT

record_stage "offline_inventory_verification"
verification_result="${job_dir}/hf-inventory-verification.json"
assert_execution_checkout
"${container[@]}" python "${execution_repo}/scripts/hpc4/stage_hf_assets.py" \
  "${execution_base}" "${PRORM_HF_CACHE}" \
  --verify-only --inventory "${PRORM_HF_INVENTORY}" \
  | tee "${verification_result}"
python3 -I -S - \
  "${verification_result}" "${PRORM_PHASE2_BASE_CONFIG_HASH}" \
  "${PRORM_HF_INVENTORY_SHA256}" <<'PY'
import json
import sys
from pathlib import Path


value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if (
    value.get("status") != "ok"
    or value.get("verify_only") is not True
    or value.get("config_hash") != sys.argv[2]
    or value.get("inventory_sha256") != sys.argv[3]
):
    raise SystemExit("offline HF inventory verification did not preserve the base identity")
PY
printf '%s  %s\n' "${PRORM_HF_INVENTORY_SHA256}" "${PRORM_HF_INVENTORY}" \
  | sha256sum --check --status || die "HF inventory changed during offline verification"

record_stage "gpu_identity_verification"
gpu_check="${job_dir}/gpu-check.log"
assert_execution_checkout
"${container[@]}" python - <<'PY' | tee "${gpu_check}"
import json
import os
import subprocess
import torch


subprocess.run(["git", "--version"], check=True)
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("Phase-2 confirmatory run requires exactly one visible CUDA GPU")
partition = os.environ.get("SLURM_JOB_PARTITION")
name = torch.cuda.get_device_name(0)
if partition != "gpu-l20" or "l20" not in name.lower():
    raise SystemExit(
        "Phase-2 confirmatory design requires gpu-l20 backed by an NVIDIA L20; "
        f"observed GPU model {name!r} on partition {partition!r}"
    )
print(
    json.dumps(
        {
            "cuda_device_count": 1,
            "gpu_model": name,
            "partition": partition,
            "status": "ok",
        },
        allow_nan=False,
        sort_keys=True,
    )
)
PY

record_stage "formal_manifest"
manifest="${job_dir}/run-manifest.json"
env_report_log="${job_dir}/env-report.log"
assert_execution_checkout
"${container[@]}" python -m smart_reward.cli env-report \
  "${execution_base}" --seed "${seed}" --repo-root "${execution_repo}" \
  --output "${manifest}" | tee "${env_report_log}"

# Crossing this barrier irrevocably consumes the formal seed attempt.  The
# controlled materializer creates candidates, policy scores, and oracle scores,
# so even checking for a reusable formal-seed artifact must happen afterwards.
[[ ! -e "${outcome_reveal_marker}" && ! -L "${outcome_reveal_marker}" ]] \
  || die "formal attempt already crossed the durable outcome-reveal boundary"
outcome_boundary_publication_started=1
outcome_reveal_temporary="$(
  mktemp "${attempt_root}/.OUTCOME_REVEAL_STARTED.tmp.XXXXXX"
)"
{
  printf 'schema_version=prorm-phase2-outcome-reveal-boundary/v1\n'
  printf 'status=OUTCOME_REVEAL_STARTED\n'
  printf 'cluster_name=%s\n' "${SLURM_CLUSTER_NAME}"
  printf 'array_job_id=%s\n' "${SLURM_ARRAY_JOB_ID}"
  printf 'array_task_id=%s\n' "${SLURM_ARRAY_TASK_ID}"
  printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID}"
  printf 'slurm_restart_count=0\n'
  printf 'attempt_index=%s\n' "${PRORM_PHASE2_ATTEMPT_INDEX}"
  printf 'seed=%s\n' "${seed}"
  printf 'phase2_design_sha256=%s\n' "${PRORM_PHASE2_DESIGN_SHA256}"
  printf 'base_config_hash=%s\n' "${PRORM_PHASE2_BASE_CONFIG_HASH}"
  printf 'git_commit=%s\n' "${PRORM_GIT_COMMIT}"
  printf 'accepted_freeze_aggregate_sha256=%s\n' \
    "${PRORM_PHASE2_ACCEPTED_FREEZE_AGGREGATE_SHA256}"
  printf 'registry_submission_sha256=%s\n' "${registry_submission_sha256}"
  printf 'registry_execution_sha256=%s\n' "${registry_execution_sha256}"
  printf 'attempt_claim_sha256=%s\n' "${attempt_claim_sha256}"
  printf 'created_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${outcome_reveal_temporary}"
mv -T --no-clobber -- \
  "${outcome_reveal_temporary}" "${outcome_reveal_marker}" \
  || die "failed to atomically publish the outcome-reveal boundary"
[[ ! -e "${outcome_reveal_temporary}" && ! -L "${outcome_reveal_temporary}" \
  && -f "${outcome_reveal_marker}" && ! -L "${outcome_reveal_marker}" ]] \
  || die "outcome-reveal boundary publication is incomplete"
fsync_file_and_parent "${outcome_reveal_marker}" \
  || die "outcome-reveal boundary durability barrier failed"
refresh_outcome_reveal_state \
  || die "outcome-reveal boundary cannot be re-derived after publication"
printf 'outcome_reveal_marker_sha256=%s\n' \
  "${outcome_reveal_marker_sha256}" >> "${attempt_evidence_log}"
record_stage "formal_scientific_outcome_boundary_entered"

artifact_dir="${job_dir}/artifact"
artifact_config_root="${PRORM_PROJECT_ROOT}/artifacts/${PRORM_PHASE2_BASE_CONFIG_HASH}"
artifact_image_root="${artifact_config_root}/${PRORM_IMAGE_SHA256}"
artifact_inventory_root="${artifact_image_root}/${PRORM_HF_INVENTORY_SHA256}"
artifact_parent="${artifact_inventory_root}/${PRORM_GIT_COMMIT}"
ensure_shared_directory "project artifacts root" "${PRORM_PROJECT_ROOT}/artifacts"
ensure_shared_directory "formal artifact config root" "${artifact_config_root}"
ensure_shared_directory "formal artifact image root" "${artifact_image_root}"
ensure_shared_directory "formal artifact inventory root" "${artifact_inventory_root}"
ensure_shared_directory "formal artifact commit root" "${artifact_parent}"
project_artifact="${artifact_parent}/seed-${seed}"
record_stage "formal_artifact_materialization"
[[ ! -e "${project_artifact}" && ! -L "${project_artifact}" ]] \
  || die "pre-existing formal-seed artifact has no current-attempt provenance"
artifact_lock="${project_artifact}.publish.lock"
exec {artifact_lock_fd}> "${artifact_lock}"
flock -n "${artifact_lock_fd}" \
  || die "another job is publishing this exact formal-seed artifact"
[[ ! -e "${project_artifact}" && ! -L "${project_artifact}" ]] \
  || die "formal-seed artifact appeared after the initial provenance check"
materialize_log="${job_dir}/materialize.log"
assert_execution_checkout
"${container[@]}" env \
  "PRORM_MEMORY_REPORT=${job_dir}/memory-materialize.json" \
  python -m smart_reward.cli controlled-materialize \
  "${execution_base}" "${artifact_dir}" --seed "${seed}" --device cuda \
  2>&1 | tee "${materialize_log}"
materialized_this_job=1
[[ -d "${artifact_dir}" && ! -L "${artifact_dir}" \
  && -f "${artifact_dir}/metadata.json" \
  && ! -L "${artifact_dir}/metadata.json" ]] \
  || die "base materializer did not produce a complete artifact"
artifact_staging="$(
  mktemp -d "${artifact_parent}/.seed-${seed}.phase2-formal-${execution_id}.XXXXXX"
)"
cp -a -- "${artifact_dir}/." "${artifact_staging}/"
fsync_tree "${artifact_staging}" \
  || die "failed to durably sync the formal-seed artifact staging tree"
[[ ! -e "${project_artifact}" && ! -L "${project_artifact}" ]] \
  || die "formal-seed artifact destination appeared before atomic publication"
mv -T --no-clobber -- "${artifact_staging}" "${project_artifact}" \
  || die "atomic no-overwrite formal-seed artifact publication failed"
fsync_directory "${artifact_parent}" \
  || die "formal-seed artifact rename parent durability barrier failed"
[[ ! -e "${artifact_staging}" && ! -L "${artifact_staging}" ]] \
  || die "formal-seed artifact publication lost an atomic rename race"
flock -u "${artifact_lock_fd}"
artifact_mode="materialized_by_current_attempt"

artifact_metadata_sha256="$(
  sha256sum -- "${project_artifact}/metadata.json" | awk '{print $1}'
)"
job_artifact_metadata_sha256="$(
  sha256sum -- "${artifact_dir}/metadata.json" | awk '{print $1}'
)"
[[ "${job_artifact_metadata_sha256}" = "${artifact_metadata_sha256}" ]] \
  || die "scratch and published base artifact metadata differ"
python3 -I -S - \
  "${job_dir}/artifact-materialization.json" "${artifact_mode}" \
  "${PRORM_PHASE2_BASE_CONFIG_HASH}" "${PRORM_PHASE2_DESIGN_SHA256}" \
  "${seed}" "${artifact_metadata_sha256}" "${PRORM_GIT_COMMIT}" \
  "${PRORM_IMAGE_SHA256}" "${PRORM_HF_INVENTORY_SHA256}" <<'PY'
import json
import sys
from pathlib import Path


(
    output,
    mode,
    base_hash,
    design_hash,
    seed,
    metadata_sha,
    commit,
    image_sha,
    inventory_sha,
) = sys.argv[1:]
payload = {
    "schema_version": "prorm-phase2-artifact-binding/v1",
    "mode": mode,
    "base_config_hash": base_hash,
    "phase2_design_sha256": design_hash,
    "seed": int(seed),
    "artifact_metadata_sha256": metadata_sha,
    "producer": {
        "git_commit": commit,
        "image_sha256": image_sha,
        "hf_inventory_sha256": inventory_sha,
    },
}
with Path(output).open("x", encoding="utf-8", newline="\n") as stream:
    json.dump(payload, stream, allow_nan=False, sort_keys=True, separators=(",", ":"))
    stream.write("\n")
PY

artifact_link="${project_run}/artifact"
[[ ! -e "${artifact_link}" && ! -L "${artifact_link}" ]] \
  || die "refusing to overwrite the formal run artifact link"
artifact_relative="$(realpath --relative-to="${project_run}" "${project_artifact}")"
[[ "${artifact_relative}" != /* ]] || die "artifact link target must be relative"
ln -s -- "${artifact_relative}" "${artifact_link}"

record_stage "artifact_bridge_verification"
artifact_verification="${job_dir}/artifact-verification.json"
assert_execution_checkout
"${container[@]}" python - \
  "${execution_overlay}" "${seed}" "${artifact_dir}" "${manifest}" \
  <<'PY' | tee "${artifact_verification}"
import json
import sys
from smart_reward.phase2_inputs import prepare_phase2_inputs


inputs = prepare_phase2_inputs(
    sys.argv[1],
    seed=int(sys.argv[2]),
    artifact_dir=sys.argv[3],
    run_manifest=sys.argv[4],
)
print(
    json.dumps(
        {
            "artifact_metadata_sha256": inputs.artifact_metadata_sha256,
            "base_config_hash": inputs.source_config_hash,
            "formal_environment": inputs.environment_identity["formal"],
            "phase2_design_sha256": inputs.phase2_config_hash,
            "seed": inputs.seed,
            "status": "ok",
        },
        allow_nan=False,
        sort_keys=True,
    )
)
PY

phase2_result="${job_dir}/phase2-result.json"
phase2_rollouts="${job_dir}/phase2-result.rollouts.jsonl"
phase2_log="${job_dir}/phase2-run.log"
printf '%s  %s\n' \
  "${PRORM_PHASE2_ACCEPTED_FREEZE_AGGREGATE_SHA256}" \
  "${PRORM_PHASE2_ACCEPTED_FREEZE_AGGREGATE}" \
  | sha256sum --check --status \
  || die "accepted freeze aggregate changed before formal Phase-2 execution"
assert_execution_checkout
"${container[@]}" env \
  "PRORM_MEMORY_REPORT=${job_dir}/memory-phase2.json" \
  python -m smart_reward.cli phase2-run \
  "${execution_overlay}" "${artifact_dir}" "${manifest}" "${phase2_result}" \
  --seed "${seed}" --device cuda \
  --beta-source-aggregate "${PRORM_PHASE2_ACCEPTED_FREEZE_AGGREGATE}" \
  --horizon-parent-aggregate "${PRORM_PHASE2_ACCEPTED_FREEZE_AGGREGATE}" \
  2>&1 | tee "${phase2_log}"
assert_execution_checkout

record_stage "formal_result_verification"
output_verification="${job_dir}/phase2-output-verification.json"
python3 -I -S - \
  "${phase2_result}" "${phase2_rollouts}" "${output_verification}" \
  "${seed}" "${PRORM_PHASE2_BASE_CONFIG_HASH}" \
  "${PRORM_PHASE2_DESIGN_SHA256}" "${PRORM_GIT_COMMIT}" \
  "${PRORM_IMAGE_SHA256}" "${PRORM_HF_INVENTORY_SHA256}" \
  "${artifact_metadata_sha256}" \
  "${PRORM_PHASE2_ACCEPTED_FREEZE_AGGREGATE_SHA256}" \
  "${PRORM_PHASE2_FROZEN_GLOBAL_BETA}" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path


(
    result_raw,
    rollouts_raw,
    output_raw,
    seed_raw,
    base_hash,
    design_hash,
    commit,
    image_sha,
    inventory_sha,
    artifact_metadata_sha,
    freeze_sha,
    frozen_beta_raw,
) = sys.argv[1:]
result_path = Path(result_raw)
rollouts_path = Path(rollouts_raw)
result = json.loads(result_path.read_text(encoding="utf-8"))
environment = {
    "formal": True,
    "git_commit": commit,
    "image_sha256": image_sha,
    "hf_inventory_sha256": inventory_sha,
    "account": "sigroup",
    "partition": "gpu-l20",
    "gpu_models": result.get("environment_identity", {}).get("gpu_models"),
}
if (
    result.get("schema_version") != "common-beta-finite-policy/v2"
    or result.get("design_stage") != "confirmatory"
    or result.get("formal_eligibility") is not True
    or result.get("per_seed_supports_formal_claim") is not False
    or result.get("source_config_hash") != base_hash
    or result.get("phase2_design_sha256") != design_hash
    or result.get("seed") != int(seed_raw)
    or result.get("artifact_dir") != "artifact"
    or result.get("artifact_metadata_sha256") != artifact_metadata_sha
    or result.get("run_manifest") != "run-manifest.json"
    or result.get("rollouts_jsonl") != "phase2-result.rollouts.jsonl"
    or result.get("environment_identity") != environment
    or not rollouts_path.is_file()
):
    raise SystemExit("formal Phase-2 result identity or evidence role is invalid")
runtime = result.get("phase2_runtime_contract")
beta = result.get("common_beta_calibration")
gate = result.get("pre_oracle_safety_gate")
if (
    not isinstance(runtime, dict)
    or runtime.get("schema_version") != "common-beta-design/v4"
    or runtime.get("stage") != "confirmatory"
    or runtime.get("formal_eligibility") is not True
    or runtime.get("pilot_phase") is not None
    or runtime.get("beta_source_aggregate_sha256") != freeze_sha
    or runtime.get("parent_pilot_aggregate_sha256") != freeze_sha
    or not isinstance(beta, dict)
    or beta.get("schema_version") != "common-beta-frozen-global/v1"
    or beta.get("beta_source_aggregate_sha256") != freeze_sha
    or beta.get("beta_matches_frozen_global_beta") is not True
    or beta.get("beta_selected_from_current_seed_curvature") is not False
    or not isinstance(gate, dict)
    or gate.get("schema_version") != "phase2-pre-oracle-safety-gate/v1"
    or gate.get("formal_gate") is not True
    or gate.get("measure_only") is not False
    or gate.get("passed") is not True
    or gate.get("violations") != []
    or gate.get("beta_retuned") is not False
):
    raise SystemExit("formal Phase-2 frozen-beta or safety contract is invalid")
frozen_beta = float(frozen_beta_raw)
if (
    not math.isfinite(frozen_beta)
    or frozen_beta <= 0.0
    or float(runtime.get("frozen_global_beta")) != frozen_beta
    or float(beta.get("beta_common")) != frozen_beta
    or float(beta.get("frozen_global_beta")) != frozen_beta
):
    raise SystemExit("formal Phase-2 result did not use the exact frozen global beta")
rollouts_sha = hashlib.sha256(rollouts_path.read_bytes()).hexdigest()
if result.get("rollouts_sha256") != rollouts_sha:
    raise SystemExit("formal rollout sidecar SHA256 differs from the result binding")
verification = {
    "schema_version": "prorm-phase2-confirmatory-output-verification/v1",
    "status": "passed",
    "seed": int(seed_raw),
    "source_config_hash": base_hash,
    "phase2_design_sha256": design_hash,
    "accepted_freeze_aggregate_sha256": freeze_sha,
    "frozen_global_beta": frozen_beta,
    "pre_oracle_safety_gate_passed": True,
    "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
    "rollouts_sha256": rollouts_sha,
    "environment_identity": environment,
}
with Path(output_raw).open("x", encoding="utf-8", newline="\n") as stream:
    json.dump(
        verification,
        stream,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    stream.write("\n")
PY

printf '%s  %s\n' "${artifact_metadata_sha256}" "${project_artifact}/metadata.json" \
  | sha256sum --check --status \
  || die "published base artifact metadata changed during formal execution"
printf '%s  %s\n' "${PRORM_HF_INVENTORY_SHA256}" "${PRORM_HF_INVENTORY}" \
  | sha256sum --check --status \
  || die "HF inventory changed during formal execution"
printf '%s  %s\n' "${PRORM_IMAGE_SHA256}" "${PRORM_IMAGE}" \
  | sha256sum --check --status \
  || die "research image changed during formal execution"
printf '%s  %s\n' \
  "${PRORM_PHASE2_ACCEPTED_FREEZE_AGGREGATE_SHA256}" \
  "${PRORM_PHASE2_ACCEPTED_FREEZE_AGGREGATE}" \
  | sha256sum --check --status \
  || die "accepted freeze aggregate changed during formal execution"
assert_execution_checkout
assert_registry_bindings
assert_attempt_claim
printf '%s  %s\n' "${outcome_reveal_marker_sha256}" "${outcome_reveal_marker}" \
  | sha256sum --check --status \
  || die "outcome-reveal boundary changed during formal execution"
record_stage "success_terminal_manifest"
record_stage "formal_success_attempt_evidence_sealed"
success_ledger_candidate="${job_dir}/phase2-attempt-ledger.success-candidate.json"
build_attempt_ledger "success_result" "true" "${success_ledger_candidate}"
success_spec="${job_dir}/phase2-success-spec.json"
python3 -I -S - \
  "${success_ledger_candidate}" "${success_spec}" <<'PY'
import json
import sys
from pathlib import Path


ledger = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
with Path(sys.argv[2]).open("x", encoding="utf-8", newline="\n") as stream:
    json.dump(
        {"attempt_ledger": ledger},
        stream,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    stream.write("\n")
PY
success_terminal="${job_dir}/phase2-success-terminal.json"
success_log="${job_dir}/phase2-success-manifest.log"
"${container[@]}" python -m smart_reward.cli phase2-success-manifest \
  "${execution_overlay}" "${phase2_result}" "${success_spec}" "${success_terminal}" \
  | tee "${success_log}"
python3 -I -S - \
  "${success_terminal}" "${phase2_result}" "${phase2_rollouts}" \
  "${PRORM_PHASE2_ATTEMPT_INDEX}" "${SLURM_JOB_ID}" \
  "${seed}" "${PRORM_PHASE2_BASE_CONFIG_HASH}" \
  "${PRORM_PHASE2_DESIGN_SHA256}" "${SLURM_CLUSTER_NAME}" \
  "${SLURM_ARRAY_JOB_ID}" "${SLURM_ARRAY_TASK_ID}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path


terminal = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
attempts = terminal.get("attempt_ledger", {}).get("attempts")
rollout_path = Path(sys.argv[3])
if rollout_path.is_symlink() or not rollout_path.is_file():
    raise SystemExit("formal success rollout is missing or is a symlink")
rollout_rows = [
    json.loads(line)
    for line in rollout_path.read_text(encoding="utf-8").splitlines()
    if line
]
if not rollout_rows or any(
    not isinstance(row, dict)
    or row.get("schema_version") != "common-beta-trajectory/v2"
    for row in rollout_rows
):
    raise SystemExit("formal success rollout contains an unsupported row schema")
expected_rollout = {
    "path": rollout_path.name,
    "sha256": hashlib.sha256(rollout_path.read_bytes()).hexdigest(),
    "schema_version": "common-beta-trajectory/v2",
}
expected_terminal_keys = {
    "schema_version",
    "terminal_status",
    "terminal",
    "supports_formal_claim",
    "seed",
    "source_config_hash",
    "phase2_design_sha256",
    "phase2_runtime_contract_sha256",
    "result",
    "rollout",
    "run_manifest_sha256",
    "artifact_metadata_sha256",
    "environment_identity",
    "pre_oracle_safety_gate_passed",
    "attempt_ledger",
    "seed_replacement_allowed",
}
if (
    set(terminal) != expected_terminal_keys
    or terminal.get("schema_version") != "phase2-seed-terminal-success/v2"
    or terminal.get("terminal_status") != "success_result"
    or terminal.get("terminal") is not True
    or terminal.get("supports_formal_claim") is not False
    or terminal.get("seed_replacement_allowed") is not False
    or terminal.get("seed") != int(sys.argv[6])
    or terminal.get("source_config_hash") != sys.argv[7]
    or terminal.get("phase2_design_sha256") != sys.argv[8]
    or terminal.get("result", {}).get("path") != "phase2-result.json"
    or terminal.get("result", {}).get("sha256")
    != hashlib.sha256(Path(sys.argv[2]).read_bytes()).hexdigest()
    or terminal.get("result", {}).get("schema_version")
    != "common-beta-finite-policy/v2"
    or terminal.get("rollout") != expected_rollout
    or not isinstance(attempts, list)
    or len(attempts) != int(sys.argv[4])
    or attempts[-1].get("cluster_name") != sys.argv[9]
    or attempts[-1].get("array_job_id") != sys.argv[10]
    or attempts[-1].get("array_task_id") != int(sys.argv[11])
    or attempts[-1].get("slurm_job_id") != sys.argv[5]
    or attempts[-1].get("status") != "success_result"
    or attempts[-1].get("final_outcome_reveal_started") is not True
):
    raise SystemExit("formal success terminal manifest is malformed")
PY
success_receipt_temporary="$(
  mktemp "${job_dir}/.phase2-success-validation-receipt.tmp.XXXXXX"
)"
python3 -I -S - \
  "${success_receipt_temporary}" "${success_ledger_candidate}" \
  "${success_terminal}" "${phase2_result}" "${phase2_rollouts}" \
  "${seed}" "${PRORM_PHASE2_DESIGN_SHA256}" \
  "${PRORM_PHASE2_BASE_CONFIG_HASH}" "${PRORM_GIT_COMMIT}" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path


(
    output_raw,
    ledger_raw,
    terminal_raw,
    result_raw,
    rollout_raw,
    seed_raw,
    design_sha,
    base_hash,
    git_commit,
) = sys.argv[1:]


def digest(path):
    value = Path(path)
    if value.is_symlink() or not value.is_file():
        raise SystemExit(f"success receipt source is missing or unsafe: {value}")
    return hashlib.sha256(value.read_bytes()).hexdigest()


payload = {
    "schema_version": "prorm-phase2-success-validation-receipt/v1",
    "status": "VALIDATED_SUCCESS_READY_FOR_ATOMIC_JOB_PUBLICATION",
    "seed": int(seed_raw),
    "phase2_design_sha256": design_sha,
    "base_config_hash": base_hash,
    "git_commit": git_commit,
    "attempt_ledger_sha256": digest(ledger_raw),
    "success_terminal_sha256": digest(terminal_raw),
    "result_sha256": digest(result_raw),
    "rollout_sha256": digest(rollout_raw),
}
with Path(output_raw).open("w", encoding="utf-8", newline="\n") as stream:
    json.dump(payload, stream, allow_nan=False, sort_keys=True, separators=(",", ":"))
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
PY
mv -T --no-clobber -- \
  "${success_receipt_temporary}" "${success_validation_receipt}" \
  || die "failed to publish the validated-success receipt"
fsync_file_and_parent "${success_validation_receipt}" \
  || die "validated-success receipt durability barrier failed"

printf 'Phase-2 confirmatory seed complete: design=%s seed=%s attempt=%s run=%s\n' \
  "${run_name}" "${seed}" "${PRORM_PHASE2_ATTEMPT_INDEX}" "${project_run_final}" \
  || true
