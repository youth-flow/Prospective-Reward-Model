#!/usr/bin/env bash
#SBATCH --job-name=prorm-p2-post-agg
#SBATCH --account=sigroup
#SBATCH --partition=gpu-l20
#SBATCH --qos=l20_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --no-requeue

set -euo pipefail
umask 027
readonly HOST_PYTHON="/opt/shared/spack/local/linux-rocky9-x86_64_v4/gcc-11.4.1/miniconda3-24.3.0-quc3pyudmzikgo2r4qsyqpwnrvzpin63/bin/python3.12"
readonly HOST_PYTHON_SHA256="9c91f9aa231c61c6bf2eabb9b93ebc5a8269a4126a36125e9548d8853e32da9c"
die() { echo "error: $*" >&2; exit 2; }

for name in \
  PRORM_PROJECT_ROOT PRORM_SCRATCH_ROOT PRORM_REPO_ROOT \
  PRORM_IMAGE PRORM_IMAGE_SHA256 PRORM_HF_INVENTORY \
  PRORM_HF_INVENTORY_SHA256 PRORM_POST_RECOVERY_OVERLAY_REL \
  PRORM_PHASE2_BASE_REL PRORM_POST_RECOVERY_OVERLAY_SHA256 \
  PRORM_PHASE2_BASE_SHA256 PRORM_POST_RECOVERY_DESIGN_SHA256 \
  PRORM_PHASE2_BASE_CONFIG_HASH PRORM_RECOVERY_AUTHORIZATION \
  PRORM_RECOVERY_AUTHORIZATION_SHA256 PRORM_POST_RECOVERY_TERMINAL \
  PRORM_POST_RECOVERY_TERMINAL_RAW PRORM_POST_RECOVERY_TERMINAL_SHA256 \
  PRORM_POST_RECOVERY_ARRAY_JOB_ID PRORM_AGGREGATOR_GIT_COMMIT \
  PRORM_PRODUCER_GIT_COMMIT PRORM_POST_RECOVERY_AGGREGATE_OUTPUT \
  PRORM_POST_RECOVERY_EVIDENCE_ROOT PRORM_POST_RECOVERY_OVERLAY_EVIDENCE \
  PRORM_POST_RECOVERY_BASE_EVIDENCE \
  PRORM_POST_RECOVERY_PILOT_PHASE PRORM_POST_RECOVERY_AUTHORIZATION_MODE \
  PRORM_PHASE2_BETA_SOURCE_AGGREGATE_PRESENT \
  PRORM_PHASE2_HORIZON_PARENT_AGGREGATE_PRESENT \
  PRORM_POST_RECOVERY_AGGREGATE_SUBMISSION_REGISTRY \
  PRORM_POST_RECOVERY_AGGREGATE_INTENT_SHA256 \
  PRORM_POST_RECOVERY_AGGREGATE_ATTEMPT_INDEX \
  PRORM_POST_RECOVERY_AGGREGATE_WORKLOAD_EXPORT_SHA256 \
  PRORM_POST_RECOVERY_RUN_0 PRORM_POST_RECOVERY_RUN_1 \
  PRORM_POST_RECOVERY_RUN_2; do
  [[ -n "${!name:-}" ]] || die "${name} is required"
done
: "${SLURM_JOB_ID:?post-recovery aggregation must execute in Slurm}"
[[ -z "${SLURM_ARRAY_JOB_ID:-}" && -z "${SLURM_ARRAY_TASK_ID:-}" ]] \
  || die "post-recovery aggregation must be one non-array job"
[[ "${SLURM_CLUSTER_NAME:-}" = "hpc4" ]] || die "aggregation requires hpc4"
[[ "${SLURM_JOB_ACCOUNT:-}" = "sigroup" ]] || die "aggregation requires sigroup"
[[ "${SLURM_JOB_PARTITION:-}" = "gpu-l20" ]] \
  || die "aggregation requires the verified gpu-l20 SIF runtime"
[[ "${SLURM_RESTART_COUNT:-0}" = "0" ]] || die "aggregation refuses restart/requeue"
authorization_validation_options=()
case "${PRORM_POST_RECOVERY_AUTHORIZATION_MODE}" in
  active-r3) ;;
  legacy-r2-replay)
    authorization_validation_options+=(--legacy-r2-replay)
    ;;
  *) die "PRORM_POST_RECOVERY_AUTHORIZATION_MODE is invalid" ;;
esac
[[ "${SLURM_NNODES:-}" = "1" && "${SLURM_NTASKS:-}" = "1" ]] \
  || die "aggregation requires one node/task"
case "${SLURM_GPUS_ON_NODE:-}" in
  1|gpu:1) ;;
  *) die "aggregation requires exactly one allocated L20" ;;
esac
[[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != *,* ]] \
  || die "aggregation requires exactly one visible allocated GPU"
while IFS= read -r variable; do
  case "${variable}" in
    APPTAINER*|SINGULARITY*) die "unset exported ${variable}" ;;
  esac
done < <(compgen -e)
for command_name in \
  apptainer git realpath sha256sum awk mktemp cp mv tee id; do
  command -v "${command_name}" >/dev/null 2>&1 \
    || die "required command is unavailable: ${command_name}"
done
host_python="$(realpath -e -- "${HOST_PYTHON}")" \
  || die "fixed host Python is unavailable"
[[ "${host_python}" = "${HOST_PYTHON}" && -f "${host_python}" \
  && -x "${host_python}" && ! -L "${host_python}" ]] \
  || die "fixed host Python must be a canonical non-symlink executable"
printf '%s  %s\n' "${HOST_PYTHON_SHA256}" "${host_python}" \
  | sha256sum --check --status || die "fixed host Python SHA256 mismatch"
[[ "$("${host_python}" --version 2>&1)" = "Python 3.12.2" ]] \
  || die "fixed host Python version mismatch"

fsync_file() {
  "${host_python}" -I -S - "$1" <<'PY'
import os
import stat
import sys

path = os.path.abspath(sys.argv[1])
descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
try:
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        raise SystemExit("fsync target is not a regular file")
    os.fsync(descriptor)
finally:
    os.close(descriptor)
PY
}

fsync_directory() {
  "${host_python}" -I -S - "$1" <<'PY'
import os
import stat
import sys

path = os.path.abspath(sys.argv[1])
descriptor = os.open(
    path,
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0),
)
try:
    if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
        raise SystemExit("fsync target is not a directory")
    os.fsync(descriptor)
finally:
    os.close(descriptor)
PY
}

fsync_tree() {
  "${host_python}" -I -S - "$1" <<'PY'
import os
import stat
import sys
from pathlib import Path

root = Path(sys.argv[1]).absolute()
if root.is_symlink() or not root.is_dir() or root.resolve(strict=True) != root:
    raise SystemExit("fsync tree root is unsafe")
directories = []
for current, names, files in os.walk(root, topdown=True, followlinks=False):
    current_path = Path(current)
    directories.append(current_path)
    for name in names:
        child = current_path / name
        if child.is_symlink() or not child.is_dir():
            raise SystemExit("fsync tree contains an unsafe directory")
    for name in files:
        child = current_path / name
        if child.is_symlink():
            raise SystemExit("fsync tree contains a symlink")
        descriptor = os.open(
            child,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise SystemExit("fsync tree contains a non-regular file")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
for directory in reversed(directories):
    descriptor = os.open(
        directory,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
PY
}

require_directory() {
  local name="$1" raw="${!1}" resolved
  [[ "${raw}" = /* ]] || die "${name} must be absolute"
  resolved="$(realpath -e -- "${raw}")" || die "${name} cannot be resolved"
  [[ -d "${resolved}" && ! -L "${raw}" && "${resolved}" != "/" \
    && "${resolved}" = "${raw}" ]] || die "${name} must be canonical"
}
require_file() {
  local name="$1" raw="${!1}" resolved
  [[ "${raw}" = /* ]] || die "${name} must be absolute"
  resolved="$(realpath -e -- "${raw}")" || die "${name} cannot be resolved"
  [[ -f "${resolved}" && ! -L "${raw}" && "${resolved}" = "${raw}" ]] \
    || die "${name} must be a canonical real file"
}
contained() {
  case "$2" in "$3"/*) ;; *) die "$1 escaped its required root" ;; esac
}
reject_delimiters() {
  [[ "$2" != *","* && "$2" != *":"* && "$2" != *"="* \
    && "$2" != *$'\n'* && "$2" != *$'\r'* ]] || die "$1 has an unsafe delimiter"
}
require_directory PRORM_PROJECT_ROOT
require_directory PRORM_SCRATCH_ROOT
require_directory PRORM_REPO_ROOT
require_file PRORM_IMAGE
require_file PRORM_HF_INVENTORY
require_file PRORM_RECOVERY_AUTHORIZATION
require_file PRORM_POST_RECOVERY_TERMINAL
require_file PRORM_POST_RECOVERY_TERMINAL_RAW
for task in 0 1 2; do
  name="PRORM_POST_RECOVERY_RUN_${task}"
  require_directory "${name}"
  contained "${name}" "${!name}" "${PRORM_PROJECT_ROOT}"
done
artifacts=()
for task in 0 1 2; do
  name="PRORM_POST_RECOVERY_RUN_${task}"
  artifact="$(realpath -e -- "${!name}/artifact")" \
    || die "run ${task} artifact target cannot be resolved"
  [[ -d "${artifact}" && ! -L "${artifact}" ]] \
    || die "run ${task} artifact target is not a real directory"
  seed=$((20260801 + task))
  case "${artifact}" in
    "${PRORM_PROJECT_ROOT}/artifacts/phase2-post-recovery-${PRORM_POST_RECOVERY_PILOT_PHASE}/${PRORM_POST_RECOVERY_DESIGN_SHA256}/seed-${seed}/job-${PRORM_POST_RECOVERY_ARRAY_JOB_ID}_${task}") ;;
    *) die "run ${task} artifact target does not match its locked namespace" ;;
  esac
  artifacts+=("${artifact}")
done
case "${PRORM_POST_RECOVERY_PILOT_PHASE}" in calibration|freeze) ;; *)
  die "post-recovery aggregate pilot phase is invalid"
  ;;
esac
predecessor_bindings=()
for kind in BETA_SOURCE_AGGREGATE HORIZON_PARENT_AGGREGATE; do
  presence="PRORM_PHASE2_${kind}_PRESENT"
  path_name="PRORM_PHASE2_${kind}"
  sha_name="PRORM_PHASE2_${kind}_SHA256"
  [[ "${!presence}" =~ ^[01]$ ]] || die "${presence} must be 0 or 1"
  if [[ "${!presence}" = "1" ]]; then
    [[ -n "${!path_name:-}" && -n "${!sha_name:-}" ]] \
      || die "${path_name} and ${sha_name} are required"
    require_file "${path_name}"
    contained "${path_name}" "${!path_name}" "${PRORM_PROJECT_ROOT}"
    [[ "${!path_name}" = "${PRORM_PROJECT_ROOT}/aggregates/"*.json \
      && "${!sha_name}" =~ ^[0-9a-f]{64}$ ]] \
      || die "${path_name} is not a locked predecessor aggregate"
    predecessor_bindings+=("${!path_name}:${!sha_name}")
  else
    [[ -z "${!path_name:-}" && -z "${!sha_name:-}" ]] \
      || die "absent predecessor leaked a path or digest"
  fi
done
for name in \
  PRORM_IMAGE PRORM_HF_INVENTORY PRORM_RECOVERY_AUTHORIZATION \
  PRORM_POST_RECOVERY_TERMINAL PRORM_POST_RECOVERY_TERMINAL_RAW; do
  contained "${name}" "${!name}" "${PRORM_PROJECT_ROOT}"
done
case "${PRORM_PROJECT_ROOT}" in
  "${PRORM_SCRATCH_ROOT}"|"${PRORM_SCRATCH_ROOT}"/*) die "roots overlap" ;;
esac
case "${PRORM_SCRATCH_ROOT}" in
  "${PRORM_PROJECT_ROOT}"|"${PRORM_PROJECT_ROOT}"/*) die "roots overlap" ;;
esac
for name in \
  PRORM_PROJECT_ROOT PRORM_SCRATCH_ROOT PRORM_REPO_ROOT PRORM_IMAGE \
  PRORM_HF_INVENTORY PRORM_RECOVERY_AUTHORIZATION \
  PRORM_POST_RECOVERY_TERMINAL PRORM_POST_RECOVERY_TERMINAL_RAW \
  PRORM_POST_RECOVERY_AGGREGATE_OUTPUT PRORM_POST_RECOVERY_EVIDENCE_ROOT \
  PRORM_POST_RECOVERY_AGGREGATE_SUBMISSION_REGISTRY \
  PRORM_POST_RECOVERY_OVERLAY_EVIDENCE PRORM_POST_RECOVERY_BASE_EVIDENCE \
  PRORM_POST_RECOVERY_RUN_0 \
  PRORM_POST_RECOVERY_RUN_1 PRORM_POST_RECOVERY_RUN_2 \
  PRORM_PHASE2_BETA_SOURCE_AGGREGATE \
  PRORM_PHASE2_HORIZON_PARENT_AGGREGATE; do
  [[ -z "${!name:-}" ]] && continue
  reject_delimiters "${name}" "${!name}"
done
for name in \
  PRORM_IMAGE_SHA256 PRORM_HF_INVENTORY_SHA256 \
  PRORM_POST_RECOVERY_OVERLAY_SHA256 PRORM_PHASE2_BASE_SHA256 \
  PRORM_POST_RECOVERY_DESIGN_SHA256 PRORM_PHASE2_BASE_CONFIG_HASH \
  PRORM_RECOVERY_AUTHORIZATION_SHA256 PRORM_POST_RECOVERY_TERMINAL_SHA256 \
  PRORM_POST_RECOVERY_AGGREGATE_INTENT_SHA256 \
  PRORM_POST_RECOVERY_AGGREGATE_WORKLOAD_EXPORT_SHA256; do
  [[ "${!name}" =~ ^[0-9a-f]{64}$ ]] || die "${name} is not lowercase SHA256"
done
for name in PRORM_AGGREGATOR_GIT_COMMIT PRORM_PRODUCER_GIT_COMMIT; do
  [[ "${!name}" =~ ^[0-9a-f]{40,64}$ ]] || die "${name} is not a full commit"
done
[[ "${PRORM_POST_RECOVERY_ARRAY_JOB_ID}" =~ ^[1-9][0-9]*$ ]] \
  || die "post-recovery pilot array ID is invalid"
[[ "${PRORM_POST_RECOVERY_AGGREGATE_ATTEMPT_INDEX}" =~ ^[1-9][0-9]*$ ]] \
  || die "aggregate attempt index is invalid"
for binding in \
  "${PRORM_IMAGE}:${PRORM_IMAGE_SHA256}" \
  "${PRORM_HF_INVENTORY}:${PRORM_HF_INVENTORY_SHA256}" \
  "${PRORM_RECOVERY_AUTHORIZATION}:${PRORM_RECOVERY_AUTHORIZATION_SHA256}" \
  "${PRORM_POST_RECOVERY_TERMINAL}:${PRORM_POST_RECOVERY_TERMINAL_SHA256}" \
  "${predecessor_bindings[@]}"; do
  path="${binding%%:*}"
  expected="${binding#*:}"
  printf '%s  %s\n' "${expected}" "${path}" \
    | sha256sum --check --status || die "aggregate immutable input SHA256 mismatch"
done
[[ "$(git -C "${PRORM_REPO_ROOT}" rev-parse --verify HEAD)" = \
  "${PRORM_AGGREGATOR_GIT_COMMIT}" ]] || die "aggregate checkout changed commit"
[[ -z "$(git -C "${PRORM_REPO_ROOT}" status --porcelain --untracked-files=normal)" ]] \
  || die "aggregate checkout is dirty"
expected_aggregate_submission_registry="${PRORM_PROJECT_ROOT}/runs/phase2-post-recovery-aggregate-attempts/$(basename "${PRORM_POST_RECOVERY_AGGREGATE_OUTPUT}")/submission-registry"
[[ "${PRORM_POST_RECOVERY_AGGREGATE_SUBMISSION_REGISTRY}" = \
  "${expected_aggregate_submission_registry}" ]] \
  || die "aggregate submission registry path differs from final output identity"
aggregate_workload_export_spec="PATH=/usr/local/bin:/usr/bin:/bin,PRORM_PROJECT_ROOT=${PRORM_PROJECT_ROOT},PRORM_SCRATCH_ROOT=${PRORM_SCRATCH_ROOT},PRORM_REPO_ROOT=${PRORM_REPO_ROOT},PRORM_IMAGE=${PRORM_IMAGE},PRORM_IMAGE_SHA256=${PRORM_IMAGE_SHA256},PRORM_HF_INVENTORY=${PRORM_HF_INVENTORY},PRORM_HF_INVENTORY_SHA256=${PRORM_HF_INVENTORY_SHA256},PRORM_POST_RECOVERY_OVERLAY_REL=${PRORM_POST_RECOVERY_OVERLAY_REL},PRORM_PHASE2_BASE_REL=${PRORM_PHASE2_BASE_REL},PRORM_POST_RECOVERY_OVERLAY_SHA256=${PRORM_POST_RECOVERY_OVERLAY_SHA256},PRORM_PHASE2_BASE_SHA256=${PRORM_PHASE2_BASE_SHA256},PRORM_POST_RECOVERY_DESIGN_SHA256=${PRORM_POST_RECOVERY_DESIGN_SHA256},PRORM_PHASE2_BASE_CONFIG_HASH=${PRORM_PHASE2_BASE_CONFIG_HASH},PRORM_RECOVERY_AUTHORIZATION=${PRORM_RECOVERY_AUTHORIZATION},PRORM_RECOVERY_AUTHORIZATION_SHA256=${PRORM_RECOVERY_AUTHORIZATION_SHA256},PRORM_POST_RECOVERY_TERMINAL=${PRORM_POST_RECOVERY_TERMINAL},PRORM_POST_RECOVERY_TERMINAL_RAW=${PRORM_POST_RECOVERY_TERMINAL_RAW},PRORM_POST_RECOVERY_TERMINAL_SHA256=${PRORM_POST_RECOVERY_TERMINAL_SHA256},PRORM_POST_RECOVERY_ARRAY_JOB_ID=${PRORM_POST_RECOVERY_ARRAY_JOB_ID},PRORM_AGGREGATOR_GIT_COMMIT=${PRORM_AGGREGATOR_GIT_COMMIT},PRORM_PRODUCER_GIT_COMMIT=${PRORM_PRODUCER_GIT_COMMIT},PRORM_POST_RECOVERY_AGGREGATE_OUTPUT=${PRORM_POST_RECOVERY_AGGREGATE_OUTPUT},PRORM_POST_RECOVERY_EVIDENCE_ROOT=${PRORM_POST_RECOVERY_EVIDENCE_ROOT},PRORM_POST_RECOVERY_OVERLAY_EVIDENCE=${PRORM_POST_RECOVERY_OVERLAY_EVIDENCE},PRORM_POST_RECOVERY_BASE_EVIDENCE=${PRORM_POST_RECOVERY_BASE_EVIDENCE},PRORM_POST_RECOVERY_PILOT_PHASE=${PRORM_POST_RECOVERY_PILOT_PHASE},PRORM_POST_RECOVERY_AUTHORIZATION_MODE=${PRORM_POST_RECOVERY_AUTHORIZATION_MODE},PRORM_PHASE2_BETA_SOURCE_AGGREGATE_PRESENT=${PRORM_PHASE2_BETA_SOURCE_AGGREGATE_PRESENT},PRORM_PHASE2_HORIZON_PARENT_AGGREGATE_PRESENT=${PRORM_PHASE2_HORIZON_PARENT_AGGREGATE_PRESENT}"
if [[ "${PRORM_PHASE2_BETA_SOURCE_AGGREGATE_PRESENT}" = "1" ]]; then
  aggregate_workload_export_spec+=",PRORM_PHASE2_BETA_SOURCE_AGGREGATE=${PRORM_PHASE2_BETA_SOURCE_AGGREGATE},PRORM_PHASE2_BETA_SOURCE_AGGREGATE_SHA256=${PRORM_PHASE2_BETA_SOURCE_AGGREGATE_SHA256}"
fi
if [[ "${PRORM_PHASE2_HORIZON_PARENT_AGGREGATE_PRESENT}" = "1" ]]; then
  aggregate_workload_export_spec+=",PRORM_PHASE2_HORIZON_PARENT_AGGREGATE=${PRORM_PHASE2_HORIZON_PARENT_AGGREGATE},PRORM_PHASE2_HORIZON_PARENT_AGGREGATE_SHA256=${PRORM_PHASE2_HORIZON_PARENT_AGGREGATE_SHA256}"
fi
for task in 0 1 2; do
  name="PRORM_POST_RECOVERY_RUN_${task}"
  aggregate_workload_export_spec+=",${name}=${!name}"
done
aggregate_workload_export_sha256="$(
  printf '%s' "${aggregate_workload_export_spec}" | sha256sum | awk '{print $1}'
)"
[[ "${aggregate_workload_export_sha256}" = \
  "${PRORM_POST_RECOVERY_AGGREGATE_WORKLOAD_EXPORT_SHA256}" ]] \
  || die "aggregate workload export differs from its precommitted intent"
aggregate_submission_validation="$(
"${host_python}" \
  "${PRORM_REPO_ROOT}/scripts/hpc4/validate_phase2_post_recovery_aggregate_submission.py" \
  "${PRORM_POST_RECOVERY_AGGREGATE_SUBMISSION_REGISTRY}" \
  --intent-sha256 "${PRORM_POST_RECOVERY_AGGREGATE_INTENT_SHA256}" \
  --attempt-index "${PRORM_POST_RECOVERY_AGGREGATE_ATTEMPT_INDEX}" \
  --job-id "${SLURM_JOB_ID}" \
  --project-root "${PRORM_PROJECT_ROOT}" \
  --repo-root "${PRORM_REPO_ROOT}" \
  --output "${PRORM_POST_RECOVERY_AGGREGATE_OUTPUT}" \
  --workload-export-sha256 \
  "${PRORM_POST_RECOVERY_AGGREGATE_WORKLOAD_EXPORT_SHA256}"
)" || die "gpu-l20 aggregate submission registry validation failed"
mapfile -t aggregate_submission_fields < <(
  "${host_python}" -I -S - "${aggregate_submission_validation}" <<'PY'
import json
import re
import sys

value = json.loads(sys.argv[1])
if value.get("status") != "verified":
    raise SystemExit("aggregate submission validator did not return verified")
for key in (
    "intent_path",
    "intent_sha256",
    "attempt_path",
    "attempt_sha256",
    "attempt_index",
    "failure_chain_sha256",
    "script_path",
    "script_sha256",
    "script_size_bytes",
):
    item = value.get(key)
    if not isinstance(item, (str, int)):
        raise SystemExit(f"aggregate submission validator returned invalid {key}")
    print(item)
for key in (
    "intent_sha256",
    "attempt_sha256",
    "failure_chain_sha256",
    "script_sha256",
):
    if re.fullmatch(r"[0-9a-f]{64}", str(value[key])) is None:
        raise SystemExit(f"aggregate submission validator returned invalid {key}")
if value["attempt_index"] <= 0 or value["script_size_bytes"] <= 0:
    raise SystemExit("aggregate submission validator returned invalid attempt index")
PY
)
[[ "${#aggregate_submission_fields[@]}" = "9" ]] \
  || die "aggregate submission evidence fields could not be resolved"
aggregate_submission_intent_path="${aggregate_submission_fields[0]}"
aggregate_submission_intent_sha256="${aggregate_submission_fields[1]}"
aggregate_submission_attempt_path="${aggregate_submission_fields[2]}"
aggregate_submission_attempt_sha256="${aggregate_submission_fields[3]}"
aggregate_submission_attempt_index="${aggregate_submission_fields[4]}"
aggregate_submission_failure_chain_sha256="${aggregate_submission_fields[5]}"
aggregate_submission_script_path="${aggregate_submission_fields[6]}"
aggregate_submission_script_sha256="${aggregate_submission_fields[7]}"
aggregate_submission_script_size_bytes="${aggregate_submission_fields[8]}"
[[ "${aggregate_submission_intent_sha256}" = \
  "${PRORM_POST_RECOVERY_AGGREGATE_INTENT_SHA256}" \
  && "${aggregate_submission_attempt_index}" = \
  "${PRORM_POST_RECOVERY_AGGREGATE_ATTEMPT_INDEX}" ]] \
  || die "aggregate submission validator identity differs from running job"
require_file aggregate_submission_script "${aggregate_submission_script_path}"
contained aggregate_submission_script "${aggregate_submission_script_path}" \
  "${PRORM_POST_RECOVERY_AGGREGATE_SUBMISSION_REGISTRY}"
[[ "$(stat -c '%s' -- "${aggregate_submission_script_path}")" = \
  "${aggregate_submission_script_size_bytes}" ]] \
  || die "aggregate committed sbatch evidence size differs"
submission_registry="${PRORM_PROJECT_ROOT}/runs/phase2-post-recovery-${PRORM_POST_RECOVERY_PILOT_PHASE}/${PRORM_POST_RECOVERY_DESIGN_SHA256}/submission-registry"
require_directory submission_registry
contained submission_registry "${submission_registry}" "${PRORM_PROJECT_ROOT}"
pilot_hf_cache="$(dirname -- "$(dirname -- "${PRORM_HF_INVENTORY}")")"
[[ "${PRORM_HF_INVENTORY}" = \
  "${pilot_hf_cache}/inventories/${PRORM_PHASE2_BASE_CONFIG_HASH}.json" ]] \
  || die "pilot HF inventory path cannot reconstruct the submission projection"
pilot_export_spec="PATH=/usr/local/bin:/usr/bin:/bin,PRORM_PROJECT_ROOT=${PRORM_PROJECT_ROOT},PRORM_SCRATCH_ROOT=${PRORM_SCRATCH_ROOT},PRORM_REPO_ROOT=${PRORM_REPO_ROOT},PRORM_IMAGE=${PRORM_IMAGE},PRORM_IMAGE_SHA256=${PRORM_IMAGE_SHA256},PRORM_HF_CACHE=${pilot_hf_cache},PRORM_HF_INVENTORY=${PRORM_HF_INVENTORY},PRORM_HF_INVENTORY_SHA256=${PRORM_HF_INVENTORY_SHA256},PRORM_POST_RECOVERY_OVERLAY_REL=${PRORM_POST_RECOVERY_OVERLAY_REL},PRORM_PHASE2_BASE_REL=${PRORM_PHASE2_BASE_REL},PRORM_POST_RECOVERY_OVERLAY_SHA256=${PRORM_POST_RECOVERY_OVERLAY_SHA256},PRORM_PHASE2_BASE_SHA256=${PRORM_PHASE2_BASE_SHA256},PRORM_POST_RECOVERY_DESIGN_SHA256=${PRORM_POST_RECOVERY_DESIGN_SHA256},PRORM_PHASE2_BASE_CONFIG_HASH=${PRORM_PHASE2_BASE_CONFIG_HASH},PRORM_RECOVERY_AUTHORIZATION=${PRORM_RECOVERY_AUTHORIZATION},PRORM_RECOVERY_AUTHORIZATION_SHA256=${PRORM_RECOVERY_AUTHORIZATION_SHA256},PRORM_OPTIMIZER_SCHEDULE_SHA256=46e0b0fdc70c507b0325c445068326ba7bc30326d70b65fa33803cc3c876c216,PRORM_GIT_COMMIT=${PRORM_PRODUCER_GIT_COMMIT},PRORM_POST_RECOVERY_PILOT_PHASE=${PRORM_POST_RECOVERY_PILOT_PHASE},PRORM_POST_RECOVERY_NAMESPACE=${PRORM_POST_RECOVERY_PILOT_PHASE},PRORM_POST_RECOVERY_AUTHORIZATION_MODE=${PRORM_POST_RECOVERY_AUTHORIZATION_MODE},PRORM_PHASE2_BETA_SOURCE_AGGREGATE_PRESENT=${PRORM_PHASE2_BETA_SOURCE_AGGREGATE_PRESENT},PRORM_PHASE2_HORIZON_PARENT_AGGREGATE_PRESENT=${PRORM_PHASE2_HORIZON_PARENT_AGGREGATE_PRESENT}"
if [[ "${PRORM_PHASE2_BETA_SOURCE_AGGREGATE_PRESENT}" = "1" ]]; then
  pilot_export_spec+=",PRORM_PHASE2_BETA_SOURCE_AGGREGATE=${PRORM_PHASE2_BETA_SOURCE_AGGREGATE},PRORM_PHASE2_BETA_SOURCE_AGGREGATE_SHA256=${PRORM_PHASE2_BETA_SOURCE_AGGREGATE_SHA256}"
fi
if [[ "${PRORM_PHASE2_HORIZON_PARENT_AGGREGATE_PRESENT}" = "1" ]]; then
  pilot_export_spec+=",PRORM_PHASE2_HORIZON_PARENT_AGGREGATE=${PRORM_PHASE2_HORIZON_PARENT_AGGREGATE},PRORM_PHASE2_HORIZON_PARENT_AGGREGATE_SHA256=${PRORM_PHASE2_HORIZON_PARENT_AGGREGATE_SHA256}"
fi
pilot_export_spec_sha256="$(
  printf '%s' "${pilot_export_spec}" | sha256sum | awk '{print $1}'
)"
submission_validation="$(
"${host_python}" \
  "${PRORM_REPO_ROOT}/scripts/hpc4/validate_phase2_post_recovery_submission.py" \
  "${submission_registry}" \
  --project-root "${PRORM_PROJECT_ROOT}" \
  --repo-root "${PRORM_REPO_ROOT}" \
  --pilot-phase "${PRORM_POST_RECOVERY_PILOT_PHASE}" \
  --design-sha256 "${PRORM_POST_RECOVERY_DESIGN_SHA256}" \
  --base-config-hash "${PRORM_PHASE2_BASE_CONFIG_HASH}" \
  --authorization-sha256 "${PRORM_RECOVERY_AUTHORIZATION_SHA256}" \
  --optimizer-schedule-sha256 \
  "46e0b0fdc70c507b0325c445068326ba7bc30326d70b65fa33803cc3c876c216" \
  --git-commit "${PRORM_PRODUCER_GIT_COMMIT}" \
  --image-sha256 "${PRORM_IMAGE_SHA256}" \
  --inventory-sha256 "${PRORM_HF_INVENTORY_SHA256}" \
  --overlay-sha256 "${PRORM_POST_RECOVERY_OVERLAY_SHA256}" \
  --base-file-sha256 "${PRORM_PHASE2_BASE_SHA256}" \
  --export-spec-sha256 "${pilot_export_spec_sha256}" \
  --array-job-id "${PRORM_POST_RECOVERY_ARRAY_JOB_ID}" \
  --submitter-user "$(id -un)"
)" || die "pilot submission registry validation failed before aggregation"
mapfile -t submission_hashes < <(
  "${host_python}" -I -S - "${submission_validation}" <<'PY'
import json
import re
import sys

value = json.loads(sys.argv[1])
if value.get("status") != "verified":
    raise SystemExit("submission registry validator did not return verified")
for key in ("intent_sha256", "submission_sha256"):
    item = value.get(key)
    if not isinstance(item, str) or re.fullmatch(r"[0-9a-f]{64}", item) is None:
        raise SystemExit(f"submission registry validator returned invalid {key}")
    print(item)
PY
)
[[ "${#submission_hashes[@]}" = "2" ]] \
  || die "pilot submission registry hashes could not be resolved"
submission_intent_sha256="${submission_hashes[0]}"
submission_ledger_sha256="${submission_hashes[1]}"

output_parent="$(realpath -e -- "$(dirname "${PRORM_POST_RECOVERY_AGGREGATE_OUTPUT}")")" \
  || die "aggregate output parent is missing"
output="${output_parent}/$(basename "${PRORM_POST_RECOVERY_AGGREGATE_OUTPUT}")"
[[ "${output}" = "${PRORM_POST_RECOVERY_AGGREGATE_OUTPUT}" ]] \
  || die "aggregate output is not canonical"
contained PRORM_POST_RECOVERY_AGGREGATE_OUTPUT "${output}" "${PRORM_PROJECT_ROOT}"
[[ "${output_parent}" = "${PRORM_PROJECT_ROOT}/aggregates" ]] \
  || die "aggregate output parent is not the exact production aggregates directory"
[[ ! -e "${output}" && ! -L "${output}" ]] || die "refusing to overwrite aggregate"
for control_artifact in \
  "${output}.ATTEMPT" \
  "${output}.PUBLISHED" \
  "${output}.TERMINAL.json" "${output}.TERMINAL.sacct.psv" \
  "${output}.SUCCESS"; do
  [[ ! -e "${control_artifact}" && ! -L "${control_artifact}" ]] \
    || die "aggregate control artifact already exists: ${control_artifact}"
done
final_evidence_root="${output}.evidence"
final_overlay_evidence="${final_evidence_root}/${PRORM_POST_RECOVERY_OVERLAY_REL}"
final_base_evidence="${final_evidence_root}/${PRORM_PHASE2_BASE_REL}"
final_submission_intent_evidence="${final_evidence_root}/submission-registry/intent.json"
final_submission_ledger_evidence="${final_evidence_root}/submission-registry/submission.json"
[[ "${PRORM_POST_RECOVERY_EVIDENCE_ROOT}" = "${final_evidence_root}" \
  && "${PRORM_POST_RECOVERY_OVERLAY_EVIDENCE}" = "${final_overlay_evidence}" \
  && "${PRORM_POST_RECOVERY_BASE_EVIDENCE}" = "${final_base_evidence}" ]] \
  || die "configuration evidence paths are not the locked aggregate bundle"
contained PRORM_POST_RECOVERY_EVIDENCE_ROOT \
  "${final_evidence_root}" "${PRORM_PROJECT_ROOT}"
for final_artifact in \
  "${output}" "${final_evidence_root}" "${output}.ATTEMPT" "${output}.PUBLISHED" \
  "${output}.TERMINAL.json" "${output}.TERMINAL.sacct.psv" \
  "${output}.SUCCESS"; do
  [[ ! -e "${final_artifact}" && ! -L "${final_artifact}" ]] \
    || die "final aggregate namespace is already occupied: ${final_artifact}"
done
attempts_root="${PRORM_PROJECT_ROOT}/runs/phase2-post-recovery-aggregate-attempts/$(basename "${output}")"
attempt_root="${attempts_root}/job-${SLURM_JOB_ID}"
mkdir -p -- "${attempts_root}"
[[ ! -e "${attempt_root}" && ! -L "${attempt_root}" ]] \
  || die "aggregate attempt namespace already exists"
mkdir -- "${attempt_root}"
[[ -d "${attempt_root}" && ! -L "${attempt_root}" \
  && "$(realpath -e -- "${attempt_root}")" = "${attempt_root}" ]] \
  || die "aggregate attempt namespace is unsafe"

job_dir="${PRORM_SCRATCH_ROOT}/phase2-post-recovery-aggregate-jobs/${SLURM_JOB_ID}"
[[ ! -e "${job_dir}" && ! -L "${job_dir}" ]] || die "aggregate scratch job exists"
mkdir -p "${PRORM_SCRATCH_ROOT}/phase2-post-recovery-aggregate-jobs"
mkdir -- "${job_dir}"
execution_repo="${job_dir}/source"
git clone --quiet --no-hardlinks --no-checkout \
  "${PRORM_REPO_ROOT}" "${execution_repo}"
git -C "${execution_repo}" checkout --quiet --detach \
  "${PRORM_AGGREGATOR_GIT_COMMIT}"
git -C "${execution_repo}" merge-base --is-ancestor \
  "${PRORM_PRODUCER_GIT_COMMIT}" "${PRORM_AGGREGATOR_GIT_COMMIT}" \
  || die "producer commit is not an ancestor of aggregate commit"
execution_overlay="${execution_repo}/${PRORM_POST_RECOVERY_OVERLAY_REL}"
execution_base="${execution_repo}/${PRORM_PHASE2_BASE_REL}"
for binding in \
  "${execution_overlay}:${PRORM_POST_RECOVERY_OVERLAY_SHA256}" \
  "${execution_base}:${PRORM_PHASE2_BASE_SHA256}"; do
  path="${binding%%:*}"
  expected="${binding#*:}"
  printf '%s  %s\n' "${expected}" "${path}" \
    | sha256sum --check --status || die "detached aggregate config differs"
done
assert_checkout() {
  [[ "$(git -C "${execution_repo}" rev-parse --verify HEAD)" = \
    "${PRORM_AGGREGATOR_GIT_COMMIT}" \
    && -z "$(git -C "${execution_repo}" status --porcelain --untracked-files=normal)" ]] \
    || die "detached aggregate checkout changed"
}
assert_checkout

authorization_inspector="$(
  realpath -e -- \
    "${execution_repo}/scripts/hpc4/inspect_phase2_post_recovery_stdlib.py"
)" || die "detached authorization transport inspector is missing"
[[ -f "${authorization_inspector}" && ! -L "${authorization_inspector}" ]] \
  || die "detached authorization transport inspector is unsafe"
"${host_python}" -I -S "${authorization_inspector}" authorization \
  "${PRORM_RECOVERY_AUTHORIZATION}" \
  --expected-sha256 "${PRORM_RECOVERY_AUTHORIZATION_SHA256}" \
  --project-root "${PRORM_PROJECT_ROOT}" \
  --authorization-mode "${PRORM_POST_RECOVERY_AUTHORIZATION_MODE}" --format json \
  >"${job_dir}/authorization-check-host.json"
mapfile -t authorization_transport < <(
  "${host_python}" -I -S "${authorization_inspector}" authorization \
    "${PRORM_RECOVERY_AUTHORIZATION}" \
    --expected-sha256 "${PRORM_RECOVERY_AUTHORIZATION_SHA256}" \
    --project-root "${PRORM_PROJECT_ROOT}" \
    --authorization-mode "${PRORM_POST_RECOVERY_AUTHORIZATION_MODE}"
)
[[ "${#authorization_transport[@]}" -ge 3 \
  && "${authorization_transport[2]}" =~ ^[0-9]+$ \
  && "${#authorization_transport[@]}" = \
    "$((3 + authorization_transport[2]))" ]] \
  || die "authorization transport inspector returned an invalid bind plan"

binds="${job_dir}:${job_dir},${PRORM_RECOVERY_AUTHORIZATION}:${PRORM_RECOVERY_AUTHORIZATION}:ro,${PRORM_POST_RECOVERY_TERMINAL}:${PRORM_POST_RECOVERY_TERMINAL}:ro,${PRORM_POST_RECOVERY_TERMINAL_RAW}:${PRORM_POST_RECOVERY_TERMINAL_RAW}:ro"
for (( index=3; index<${#authorization_transport[@]}; index++ )); do
  IFS=$'\t' read -r dependency_kind dependency_path dependency_sha256 \
    <<<"${authorization_transport[index]}"
  [[ -n "${dependency_kind}" && -n "${dependency_path}" \
    && -n "${dependency_sha256}" ]] \
    || die "authorization dependency binding is malformed"
  reject_delimiters authorization_dependency_path "${dependency_path}"
  contained authorization_dependency_path "${dependency_path}" "${PRORM_PROJECT_ROOT}"
  case "${dependency_kind}" in
    file)
      [[ "${dependency_sha256}" =~ ^[0-9a-f]{64}$ ]] \
        || die "authorization file dependency lacks a SHA256"
      [[ -f "${dependency_path}" && ! -L "${dependency_path}" \
        && "$(realpath -e -- "${dependency_path}")" = "${dependency_path}" ]] \
        || die "authorization file dependency is unsafe"
      printf '%s  %s\n' "${dependency_sha256}" "${dependency_path}" \
        | sha256sum --check --status \
        || die "authorization file dependency changed after inspection"
      ;;
    directory)
      [[ "${dependency_sha256}" = "-" ]] \
        || die "authorization directory dependency unexpectedly carries a hash"
      [[ -d "${dependency_path}" && ! -L "${dependency_path}" \
        && "$(realpath -e -- "${dependency_path}")" = "${dependency_path}" ]] \
        || die "authorization directory dependency is unsafe"
      ;;
    *) die "authorization dependency kind is invalid" ;;
  esac
  binds+=",${dependency_path}:${dependency_path}:ro"
done
for task in 0 1 2; do
  name="PRORM_POST_RECOVERY_RUN_${task}"
  binds+=",${!name}:${!name}:ro,${artifacts[$task]}:${artifacts[$task]}:ro"
done
if [[ "${PRORM_PHASE2_HORIZON_PARENT_AGGREGATE_PRESENT}" = "1" ]]; then
  for evidence_root in \
    "${PRORM_PROJECT_ROOT}/aggregates" \
    "${PRORM_PROJECT_ROOT}/runs/phase2-post-recovery-calibration" \
    "${PRORM_PROJECT_ROOT}/artifacts/phase2-post-recovery-calibration"; do
    [[ -d "${evidence_root}" && ! -L "${evidence_root}" ]] \
      || die "predecessor evidence namespace is missing or unsafe"
    binds+=",${evidence_root}:${evidence_root}:ro"
  done
  # New-horizon calibration may itself descend from a freeze length failure.
  # Preserve recursive readability of that target-free lineage independent of
  # the current overlay phase.
  freeze_run_evidence_root="${PRORM_PROJECT_ROOT}/runs/phase2-post-recovery-freeze"
  freeze_artifact_evidence_root="${PRORM_PROJECT_ROOT}/artifacts/phase2-post-recovery-freeze"
  if [[ -e "${freeze_run_evidence_root}" \
    || -e "${freeze_artifact_evidence_root}" ]]; then
    for evidence_root in \
      "${freeze_run_evidence_root}" "${freeze_artifact_evidence_root}"; do
      [[ -d "${evidence_root}" && ! -L "${evidence_root}" ]] \
        || die "freeze predecessor evidence namespace is missing or unsafe"
      binds+=",${evidence_root}:${evidence_root}:ro"
    done
  fi
fi
container=(
  apptainer exec --cleanenv --no-mount home,cwd,bind-paths
  --pwd "${execution_repo}"
  --bind "${binds}"
  --env "PYTHONPATH=${execution_repo}/src"
  --env "PYTHONNOUSERSITE=1"
  --env "PYTHONDONTWRITEBYTECODE=1"
  "${PRORM_IMAGE}"
)

"${container[@]}" python \
  "${execution_repo}/scripts/hpc4/validate_phase2_recovery_authorization.py" \
  "${PRORM_RECOVERY_AUTHORIZATION}" "${execution_overlay}" \
  --expected-sha256 "${PRORM_RECOVERY_AUTHORIZATION_SHA256}" \
  --project-root "${PRORM_PROJECT_ROOT}" \
  "${authorization_validation_options[@]}" \
  >"${job_dir}/authorization-check.json"
"${container[@]}" python \
  "${execution_repo}/scripts/hpc4/capture_phase2_post_recovery_terminal.py" verify \
  "${PRORM_POST_RECOVERY_TERMINAL}" \
  --expected-sha256 "${PRORM_POST_RECOVERY_TERMINAL_SHA256}" \
  --array-job-id "${PRORM_POST_RECOVERY_ARRAY_JOB_ID}" \
  --pilot-phase "${PRORM_POST_RECOVERY_PILOT_PHASE}" \
  >"${job_dir}/terminal-check.json"
results=()
for task in 0 1 2; do
  name="PRORM_POST_RECOVERY_RUN_${task}"
  results+=("${!name}/phase2-pilot-diagnostics.json")
done
staged="${job_dir}/post-recovery-aggregate.json"
assert_checkout
aggregate_command=(
  python "${execution_repo}/scripts/hpc4/run_phase2_post_recovery_aggregate.py"
  "${execution_overlay}" "${staged}" "${results[@]}" \
  --publication-output "${output}" \
  --authorization "${PRORM_RECOVERY_AUTHORIZATION}" \
  --authorization-sha256 "${PRORM_RECOVERY_AUTHORIZATION_SHA256}" \
  --terminal-evidence "${PRORM_POST_RECOVERY_TERMINAL}" \
  --terminal-evidence-sha256 "${PRORM_POST_RECOVERY_TERMINAL_SHA256}" \
  --array-job-id "${PRORM_POST_RECOVERY_ARRAY_JOB_ID}" \
  --submission-intent-sha256 "${submission_intent_sha256}" \
  --submission-ledger-sha256 "${submission_ledger_sha256}" \
  --submission-intent-reference "${final_submission_intent_evidence}" \
  --submission-ledger-reference "${final_submission_ledger_evidence}" \
  --aggregator-git-commit "${PRORM_AGGREGATOR_GIT_COMMIT}" \
  --producer-git-commit "${PRORM_PRODUCER_GIT_COMMIT}" \
  --image-sha256 "${PRORM_IMAGE_SHA256}" \
  --hf-inventory-sha256 "${PRORM_HF_INVENTORY_SHA256}" \
  --reference-base "${output_parent}" \
  --phase2-overlay-reference "${final_overlay_evidence}" \
)
if [[ "${PRORM_PHASE2_BETA_SOURCE_AGGREGATE_PRESENT}" = "1" ]]; then
  aggregate_command+=(
    --beta-source-aggregate "${PRORM_PHASE2_BETA_SOURCE_AGGREGATE}"
  )
fi
if [[ "${PRORM_PHASE2_HORIZON_PARENT_AGGREGATE_PRESENT}" = "1" ]]; then
  aggregate_command+=(
    --horizon-parent-aggregate "${PRORM_PHASE2_HORIZON_PARENT_AGGREGATE}"
  )
fi
"${container[@]}" "${aggregate_command[@]}" \
  | tee "${job_dir}/post-recovery-aggregate.log"

"${host_python}" -I -S - \
  "${staged}" "${PRORM_POST_RECOVERY_DESIGN_SHA256}" \
  "${PRORM_RECOVERY_AUTHORIZATION_SHA256}" \
  "${PRORM_POST_RECOVERY_TERMINAL_SHA256}" \
  "${PRORM_POST_RECOVERY_ARRAY_JOB_ID}" \
  "${PRORM_POST_RECOVERY_OVERLAY_SHA256}" \
  "$(basename "${final_evidence_root}")/${PRORM_POST_RECOVERY_OVERLAY_REL}" \
  "${PRORM_PRODUCER_GIT_COMMIT}" \
  "${PRORM_POST_RECOVERY_PILOT_PHASE}" \
  "${PRORM_POST_RECOVERY_OVERLAY_REL}" \
  "${submission_intent_sha256}" \
  "${submission_ledger_sha256}" \
  "$(basename "${final_evidence_root}")/submission-registry/intent.json" \
  "$(basename "${final_evidence_root}")/submission-registry/submission.json" <<'PY'
import json,re,sys
from pathlib import Path
value=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
control=value.get("post_recovery_control",{})
selection=value.get("selection")
if (
 value.get("schema_version")!="common-beta-pilot-selection-aggregate/v3"
 or value.get("phase2_design_sha256")!=sys.argv[2]
 or not isinstance(value.get("sources"),list) or len(value["sources"])!=3
 or not isinstance(selection,dict)
 or control.get("schema_version")!="phase2-post-recovery-aggregation-control/v1"
 or control.get("pilot_phase")!=sys.argv[9]
 or control.get("recovery_authorization_sha256")!=sys.argv[3]
 or control.get("pilot_terminal_evidence_sha256")!=sys.argv[4]
 or control.get("pilot_array_job_id")!=sys.argv[5]
 or control.get("phase2_overlay_sha256")!=sys.argv[6]
 or control.get("phase2_overlay")!=sys.argv[7]
 or control.get("phase2_overlay_repo_relative")!=sys.argv[10]
 or control.get("submission_intent_sha256")!=sys.argv[11]
 or control.get("submission_ledger_sha256")!=sys.argv[12]
 or control.get("submission_intent")!=sys.argv[13]
 or control.get("submission_ledger")!=sys.argv[14]
 or re.fullmatch(r"[0-9a-f]{40}",str(control.get("phase2_overlay_git_blob_sha1"))) is None
 or control.get("phase2_overlay_git_commit")!=sys.argv[8]
 or control.get("normalized_phase2_config_sha256")!=sys.argv[2]
 or not isinstance(control.get("normalized_phase2_config"),dict)
 or control.get("ordered_seeds")!=[20260801,20260802,20260803]
 or control.get("materialization_mode")!="fresh"
 or control.get("recovery_outputs_reused") is not False
 or control.get("all_tasks_terminal_completed_zero_exit") is not True
):
 raise SystemExit("post-recovery aggregate failed bound output verification")
PY
for binding in \
  "${PRORM_IMAGE}:${PRORM_IMAGE_SHA256}" \
  "${PRORM_HF_INVENTORY}:${PRORM_HF_INVENTORY_SHA256}" \
  "${PRORM_RECOVERY_AUTHORIZATION}:${PRORM_RECOVERY_AUTHORIZATION_SHA256}" \
  "${PRORM_POST_RECOVERY_TERMINAL}:${PRORM_POST_RECOVERY_TERMINAL_SHA256}" \
  "${predecessor_bindings[@]}"; do
  path="${binding%%:*}"
  expected="${binding#*:}"
  printf '%s  %s\n' "${expected}" "${path}" \
    | sha256sum --check --status || die "aggregate immutable input changed"
done
assert_checkout
aggregate_sha256="$(sha256sum -- "${staged}" | awk '{print $1}')"
attempt_aggregate="${attempt_root}/aggregate.json"
attempt_evidence="${attempt_root}/evidence"
mkdir -p -- \
  "${attempt_evidence}/configs" \
  "${attempt_evidence}/submission-registry" \
  "${attempt_evidence}/aggregate-submission/attempts" \
  "${attempt_evidence}/aggregate-submission/failures" \
  "${attempt_evidence}/aggregate-submission/controller"
cp -p -- "${staged}" "${attempt_aggregate}"
cp -p -- "${execution_overlay}" \
  "${attempt_evidence}/${PRORM_POST_RECOVERY_OVERLAY_REL}"
cp -p -- "${execution_base}" \
  "${attempt_evidence}/${PRORM_PHASE2_BASE_REL}"
cp -p -- "${submission_registry}/intent.json" \
  "${attempt_evidence}/submission-registry/intent.json"
cp -p -- "${submission_registry}/submission.json" \
  "${attempt_evidence}/submission-registry/submission.json"
cp -p -- "${aggregate_submission_intent_path}" \
  "${attempt_evidence}/aggregate-submission/intent.json"
cp -p -- "${aggregate_submission_attempt_path}" \
  "${attempt_evidence}/aggregate-submission/attempt.json"
cp -p -- "${aggregate_submission_script_path}" \
  "${attempt_evidence}/aggregate-submission/script.sbatch"
shopt -s nullglob
for attempt_ledger in \
  "${PRORM_POST_RECOVERY_AGGREGATE_SUBMISSION_REGISTRY}"/attempts/attempt-*.json; do
  cp -p -- "${attempt_ledger}" \
    "${attempt_evidence}/aggregate-submission/attempts/$(basename "${attempt_ledger}")"
done
controller_readback_count=0
for controller_readback in \
  "${PRORM_POST_RECOVERY_AGGREGATE_SUBMISSION_REGISTRY}"/controller/attempt-*.sbatch; do
  [[ -f "${controller_readback}" && ! -L "${controller_readback}" ]] \
    || die "aggregate controller readback evidence is unsafe"
  controller_readback_count="$((controller_readback_count + 1))"
  controller_copy="${attempt_evidence}/aggregate-submission/controller/$(basename "${controller_readback}")"
  cp -p -- "${controller_readback}" "${controller_copy}"
  cmp --silent -- "${controller_readback}" "${controller_copy}" \
    || die "aggregate controller readback copy differs"
done
[[ "${controller_readback_count}" = "${aggregate_submission_attempt_index}" ]] \
  || die "aggregate controller readback evidence is non-contiguous"
for failure in \
  "${PRORM_POST_RECOVERY_AGGREGATE_SUBMISSION_REGISTRY}"/failures/job-*; do
  [[ -f "${failure}" && ! -L "${failure}" ]] \
    || die "aggregate failure evidence is unsafe"
  cp -p -- "${failure}" \
    "${attempt_evidence}/aggregate-submission/failures/$(basename "${failure}")"
done
shopt -u nullglob
"${host_python}" -I -S - \
  "${aggregate_submission_validation}" \
  "${attempt_evidence}/aggregate-submission/failure-chain.json" <<'PY'
import json
import os
import sys

value = json.loads(sys.argv[1])
raw = value["failure_chain"].encode("utf-8")
path = os.path.abspath(sys.argv[2])
descriptor = os.open(
    path,
    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
    0o640,
)
try:
    with os.fdopen(descriptor, "wb", closefd=False) as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
finally:
    os.close(descriptor)
PY
for binding in \
  "${attempt_aggregate}:${aggregate_sha256}" \
  "${attempt_evidence}/${PRORM_POST_RECOVERY_OVERLAY_REL}:${PRORM_POST_RECOVERY_OVERLAY_SHA256}" \
  "${attempt_evidence}/${PRORM_PHASE2_BASE_REL}:${PRORM_PHASE2_BASE_SHA256}" \
  "${attempt_evidence}/submission-registry/intent.json:${submission_intent_sha256}" \
  "${attempt_evidence}/submission-registry/submission.json:${submission_ledger_sha256}" \
  "${attempt_evidence}/aggregate-submission/intent.json:${aggregate_submission_intent_sha256}" \
  "${attempt_evidence}/aggregate-submission/attempt.json:${aggregate_submission_attempt_sha256}" \
  "${attempt_evidence}/aggregate-submission/script.sbatch:${aggregate_submission_script_sha256}" \
  "${attempt_evidence}/aggregate-submission/failure-chain.json:${aggregate_submission_failure_chain_sha256}"; do
  path="${binding%%:*}"
  expected="${binding#*:}"
  printf '%s  %s\n' "${expected}" "${path}" \
    | sha256sum --check --status \
    || die "persistent aggregate attempt staging changed"
done
fsync_tree "${attempt_root}" \
  || die "persistent aggregate attempt durability barrier failed"
ready="${attempt_root}/READY"
temporary="$(mktemp "${attempt_root}/.READY.XXXXXX")"
{
  printf 'schema_version=prorm-phase2-post-recovery-aggregate-attempt-ready/v1\n'
  printf 'status=READY\n'
  printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID}"
  printf 'slurm_job_is_array=false\n'
  printf 'cluster=hpc4\naccount=sigroup\npartition=%s\n' "${SLURM_JOB_PARTITION}"
  printf 'restart_count=0\n'
  printf 'pilot_array_job_id=%s\n' "${PRORM_POST_RECOVERY_ARRAY_JOB_ID}"
  printf 'pilot_phase=%s\n' "${PRORM_POST_RECOVERY_PILOT_PHASE}"
  printf 'phase2_design_sha256=%s\n' "${PRORM_POST_RECOVERY_DESIGN_SHA256}"
  printf 'base_config_hash=%s\n' "${PRORM_PHASE2_BASE_CONFIG_HASH}"
  printf 'recovery_authorization_sha256=%s\n' \
    "${PRORM_RECOVERY_AUTHORIZATION_SHA256}"
  printf 'optimizer_schedule_sha256=%s\n' \
    "46e0b0fdc70c507b0325c445068326ba7bc30326d70b65fa33803cc3c876c216"
  printf 'pilot_terminal_evidence_sha256=%s\n' \
    "${PRORM_POST_RECOVERY_TERMINAL_SHA256}"
  printf 'submission_intent_sha256=%s\n' "${submission_intent_sha256}"
  printf 'submission_ledger_sha256=%s\n' "${submission_ledger_sha256}"
  printf 'aggregate_submission_intent_sha256=%s\n' \
    "${aggregate_submission_intent_sha256}"
  printf 'aggregate_submission_attempt_sha256=%s\n' \
    "${aggregate_submission_attempt_sha256}"
  printf 'aggregate_submission_attempt_index=%s\n' \
    "${aggregate_submission_attempt_index}"
  printf 'aggregate_submission_failure_chain_sha256=%s\n' \
    "${aggregate_submission_failure_chain_sha256}"
  printf 'phase2_overlay_sha256=%s\n' \
    "${PRORM_POST_RECOVERY_OVERLAY_SHA256}"
  printf 'phase2_base_sha256=%s\n' "${PRORM_PHASE2_BASE_SHA256}"
  printf 'aggregator_git_commit=%s\n' "${PRORM_AGGREGATOR_GIT_COMMIT}"
  printf 'producer_git_commit=%s\n' "${PRORM_PRODUCER_GIT_COMMIT}"
  printf 'image_sha256=%s\n' "${PRORM_IMAGE_SHA256}"
  printf 'hf_inventory_sha256=%s\n' "${PRORM_HF_INVENTORY_SHA256}"
  printf 'final_output=%s\n' "${output}"
  printf 'final_evidence_root=%s\n' "${final_evidence_root}"
  printf 'attempt_aggregate=aggregate.json\n'
  printf 'attempt_evidence=evidence\n'
  printf 'aggregate_sha256=%s\n' "${aggregate_sha256}"
  printf 'final_namespace_untouched=true\n'
  printf 'created_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >"${temporary}"
fsync_file "${temporary}" \
  || die "aggregate READY staging durability barrier failed"
mv -T --no-clobber -- "${temporary}" "${ready}" \
  || die "aggregate READY publication failed"
fsync_file "${ready}" || die "aggregate READY durability barrier failed"
fsync_directory "${attempt_root}" \
  || die "aggregate READY parent durability barrier failed"
fsync_directory "${attempts_root}" \
  || die "aggregate attempt registry durability barrier failed"
printf 'post-recovery aggregate attempt ready for terminal publication: %s\n' \
  "${attempt_root}"
