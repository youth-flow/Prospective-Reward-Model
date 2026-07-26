#!/usr/bin/env bash
#SBATCH --job-name=prorm-p2-post-recovery
#SBATCH --account=sigroup
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gpus-per-node=1
#SBATCH --array=0-2%2
#SBATCH --no-requeue

set -euo pipefail
umask 027

readonly HOST_PYTHON="/opt/shared/spack/local/linux-rocky9-x86_64_v4/gcc-11.4.1/miniconda3-24.3.0-quc3pyudmzikgo2r4qsyqpwnrvzpin63/bin/python3.12"
readonly HOST_PYTHON_SHA256="9c91f9aa231c61c6bf2eabb9b93ebc5a8269a4126a36125e9548d8853e32da9c"

die() { echo "error: $*" >&2; exit 2; }

for name in \
  PRORM_PROJECT_ROOT PRORM_SCRATCH_ROOT PRORM_REPO_ROOT \
  PRORM_IMAGE PRORM_IMAGE_SHA256 PRORM_HF_CACHE \
  PRORM_HF_INVENTORY PRORM_HF_INVENTORY_SHA256 \
  PRORM_POST_RECOVERY_OVERLAY_REL PRORM_PHASE2_BASE_REL \
  PRORM_POST_RECOVERY_OVERLAY_SHA256 PRORM_PHASE2_BASE_SHA256 \
  PRORM_POST_RECOVERY_DESIGN_SHA256 PRORM_PHASE2_BASE_CONFIG_HASH \
  PRORM_RECOVERY_AUTHORIZATION PRORM_RECOVERY_AUTHORIZATION_SHA256 \
  PRORM_OPTIMIZER_SCHEDULE_SHA256 PRORM_GIT_COMMIT \
  PRORM_POST_RECOVERY_PILOT_PHASE PRORM_POST_RECOVERY_NAMESPACE \
  PRORM_PHASE2_BETA_SOURCE_AGGREGATE_PRESENT \
  PRORM_PHASE2_HORIZON_PARENT_AGGREGATE_PRESENT; do
  [[ -n "${!name:-}" ]] || die "${name} is required"
done
: "${SLURM_JOB_ID:?post-recovery pilot must execute in Slurm}"
: "${SLURM_ARRAY_JOB_ID:?post-recovery pilot requires one Slurm array}"
: "${SLURM_ARRAY_TASK_ID:?post-recovery pilot requires an array task}"

[[ "${SLURM_CLUSTER_NAME:-}" = "hpc4" ]] || die "post-recovery pilot requires cluster hpc4"
[[ "${SLURM_JOB_ACCOUNT:-}" = "sigroup" ]] || die "post-recovery pilot requires account sigroup"
[[ "${SLURM_JOB_PARTITION:-}" = "gpu-l20" ]] || die "post-recovery pilot requires gpu-l20"
[[ "${SLURM_RESTART_COUNT:-0}" = "0" ]] \
  || die "post-recovery pilot refuses restart/requeue"
[[ "${SLURM_NNODES:-}" = "1" && "${SLURM_NTASKS:-}" = "1" ]] \
  || die "post-recovery pilot requires exactly one node and one task"
case "${SLURM_GPUS_ON_NODE:-}" in
  1|gpu:1) ;;
  *) die "post-recovery pilot requires exactly one allocated GPU" ;;
esac
[[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != *,* ]] \
  || die "post-recovery pilot requires exactly one visible allocated GPU"
[[ "${SLURM_ARRAY_TASK_ID}" =~ ^[0-2]$ ]] || die "array task must be 0, 1, or 2"
[[ "${SLURM_ARRAY_TASK_COUNT:-}" = "3" ]] || die "array must contain exactly three tasks"
[[ "${SLURM_ARRAY_TASK_MIN:-}" = "0" ]] || die "array minimum must be 0"
[[ "${SLURM_ARRAY_TASK_MAX:-}" = "2" ]] || die "array maximum must be 2"
[[ "${SLURM_ARRAY_TASK_STEP:-}" = "1" ]] || die "array step must be 1"

while IFS= read -r variable; do
  case "${variable}" in
    APPTAINER*|SINGULARITY*)
      die "unset exported ${variable}; execution forbids ambient container controls"
      ;;
  esac
done < <(compgen -e)
for command_name in \
  apptainer git realpath sha256sum awk mktemp cp mv ln tee id; do
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

require_directory() {
  local name="$1" raw="${!1}" resolved=""
  [[ "${raw}" = /* ]] || die "${name} must be absolute"
  resolved="$(realpath -e -- "${raw}")" || die "${name} cannot be resolved"
  [[ -d "${resolved}" && ! -L "${raw}" && "${resolved}" != "/" \
    && "${resolved}" = "${raw}" ]] || die "${name} must be a canonical real directory"
}
require_file() {
  local name="$1" raw="${!1}" resolved=""
  [[ "${raw}" = /* ]] || die "${name} must be absolute"
  resolved="$(realpath -e -- "${raw}")" || die "${name} cannot be resolved"
  [[ -f "${resolved}" && ! -L "${raw}" && "${resolved}" = "${raw}" ]] \
    || die "${name} must be a canonical regular non-symlink file"
}
contained() {
  local name="$1" child="$2" parent="$3"
  case "${child}" in
    "${parent}"/*) ;;
    *) die "${name} escaped its required root" ;;
  esac
}
reject_delimiters() {
  local name="$1" value="$2"
  [[ "${value}" != *","* && "${value}" != *":"* && "${value}" != *"="* \
    && "${value}" != *$'\n'* && "${value}" != *$'\r'* ]] \
    || die "${name} contains an unsafe bind delimiter"
}

require_directory PRORM_PROJECT_ROOT
require_directory PRORM_SCRATCH_ROOT
require_directory PRORM_REPO_ROOT
require_directory PRORM_HF_CACHE
require_file PRORM_IMAGE
require_file PRORM_HF_INVENTORY
require_file PRORM_RECOVERY_AUTHORIZATION
contained PRORM_IMAGE "${PRORM_IMAGE}" "${PRORM_PROJECT_ROOT}"
contained PRORM_HF_CACHE "${PRORM_HF_CACHE}" "${PRORM_PROJECT_ROOT}"
contained PRORM_HF_INVENTORY "${PRORM_HF_INVENTORY}" "${PRORM_HF_CACHE}"
contained \
  PRORM_RECOVERY_AUTHORIZATION \
  "${PRORM_RECOVERY_AUTHORIZATION}" "${PRORM_PROJECT_ROOT}"
case "${PRORM_PROJECT_ROOT}" in
  "${PRORM_SCRATCH_ROOT}"|"${PRORM_SCRATCH_ROOT}"/*)
    die "project and scratch roots overlap"
    ;;
esac
case "${PRORM_SCRATCH_ROOT}" in
  "${PRORM_PROJECT_ROOT}"|"${PRORM_PROJECT_ROOT}"/*)
    die "project and scratch roots overlap"
    ;;
esac
for name in \
  PRORM_PROJECT_ROOT PRORM_SCRATCH_ROOT PRORM_REPO_ROOT PRORM_IMAGE \
  PRORM_HF_CACHE PRORM_HF_INVENTORY PRORM_RECOVERY_AUTHORIZATION; do
  reject_delimiters "${name}" "${!name}"
done
for name in \
  PRORM_IMAGE_SHA256 PRORM_HF_INVENTORY_SHA256 \
  PRORM_POST_RECOVERY_OVERLAY_SHA256 PRORM_PHASE2_BASE_SHA256 \
  PRORM_POST_RECOVERY_DESIGN_SHA256 PRORM_PHASE2_BASE_CONFIG_HASH \
  PRORM_RECOVERY_AUTHORIZATION_SHA256 PRORM_OPTIMIZER_SCHEDULE_SHA256; do
  [[ "${!name}" =~ ^[0-9a-f]{64}$ ]] || die "${name} must be lowercase SHA256"
done
[[ "${PRORM_GIT_COMMIT}" =~ ^[0-9a-f]{40,64}$ ]] \
  || die "PRORM_GIT_COMMIT is not a full lowercase object ID"
case "${PRORM_POST_RECOVERY_PILOT_PHASE}:${PRORM_POST_RECOVERY_NAMESPACE}" in
  calibration:calibration|freeze:freeze) ;;
  *) die "post-recovery pilot phase/namespace binding is invalid" ;;
esac
case "${PRORM_POST_RECOVERY_OVERLAY_REL}" in
  configs/common_beta_post_recovery_calibration.yaml)
    [[ "${PRORM_POST_RECOVERY_PILOT_PHASE}" = "calibration" ]] \
      || die "base calibration overlay has the wrong phase"
    ;;
  configs/common_beta_post_recovery_calibration_horizon_[1-9]*.yaml)
    [[ "${PRORM_POST_RECOVERY_PILOT_PHASE}" = "calibration" ]] \
      || die "horizon calibration overlay has the wrong phase"
    ;;
  configs/common_beta_post_recovery_freeze.yaml)
    [[ "${PRORM_POST_RECOVERY_PILOT_PHASE}" = "freeze" ]] \
      || die "freeze overlay has the wrong phase"
    ;;
  configs/common_beta_post_recovery_freeze_retry_[1-9]*.yaml)
    [[ "${PRORM_POST_RECOVERY_PILOT_PHASE}" = "freeze" ]] \
      || die "freeze retry overlay has the wrong phase"
    ;;
  *) die "submitted post-recovery overlay path is not semantic" ;;
esac
case "${PRORM_PHASE2_BASE_REL}" in configs/*.yaml) ;; *)
  die "submitted base config path is not a tracked config path"
  ;;
esac
[[ "${PRORM_OPTIMIZER_SCHEDULE_SHA256}" = \
  "46e0b0fdc70c507b0325c445068326ba7bc30326d70b65fa33803cc3c876c216" ]] \
  || die "submitted optimizer schedule is not recovery-authorized"
[[ "${PRORM_HF_INVENTORY}" = \
  "${PRORM_HF_CACHE}/inventories/${PRORM_PHASE2_BASE_CONFIG_HASH}.json" ]] \
  || die "HF inventory is not addressed by the base identity"
for binding in \
  "${PRORM_IMAGE}:${PRORM_IMAGE_SHA256}" \
  "${PRORM_HF_INVENTORY}:${PRORM_HF_INVENTORY_SHA256}" \
  "${PRORM_RECOVERY_AUTHORIZATION}:${PRORM_RECOVERY_AUTHORIZATION_SHA256}"; do
  path="${binding%%:*}"
  expected="${binding#*:}"
  printf '%s  %s\n' "${expected}" "${path}" \
    | sha256sum --check --status || die "immutable input SHA256 mismatch: ${path}"
done

for presence_name in \
  PRORM_PHASE2_BETA_SOURCE_AGGREGATE_PRESENT \
  PRORM_PHASE2_HORIZON_PARENT_AGGREGATE_PRESENT; do
  [[ "${!presence_name}" =~ ^[01]$ ]] || die "${presence_name} must be 0 or 1"
done
if [[ "${PRORM_POST_RECOVERY_OVERLAY_REL}" = \
  "configs/common_beta_post_recovery_calibration.yaml" ]]; then
  [[ "${PRORM_PHASE2_BETA_SOURCE_AGGREGATE_PRESENT}" = "0" \
    && "${PRORM_PHASE2_HORIZON_PARENT_AGGREGATE_PRESENT}" = "0" ]] \
    || die "initial calibration must not consume a predecessor"
else
  [[ "${PRORM_PHASE2_HORIZON_PARENT_AGGREGATE_PRESENT}" = "1" ]] \
    || die "non-initial pilot requires a horizon predecessor"
fi
predecessor_bindings=()
for kind in BETA_SOURCE_AGGREGATE HORIZON_PARENT_AGGREGATE; do
  presence="PRORM_PHASE2_${kind}_PRESENT"
  path_name="PRORM_PHASE2_${kind}"
  sha_name="PRORM_PHASE2_${kind}_SHA256"
  if [[ "${!presence}" = "1" ]]; then
    [[ -n "${!path_name:-}" && -n "${!sha_name:-}" ]] \
      || die "${path_name} and ${sha_name} are required"
    require_file "${path_name}"
    contained "${path_name}" "${!path_name}" "${PRORM_PROJECT_ROOT}"
    [[ "${!path_name}" = "${PRORM_PROJECT_ROOT}/aggregates/"*.json ]] \
      || die "${path_name} must use the locked aggregate namespace"
    [[ "${!sha_name}" =~ ^[0-9a-f]{64}$ ]] \
      || die "${sha_name} must be lowercase SHA256"
    printf '%s  %s\n' "${!sha_name}" "${!path_name}" \
      | sha256sum --check --status || die "${path_name} changed"
    predecessor_bindings+=("${!path_name}:${!sha_name}")
  else
    [[ -z "${!path_name:-}" && -z "${!sha_name:-}" ]] \
      || die "absent predecessor leaked a path or digest"
  fi
done

submission_head="$(git -C "${PRORM_REPO_ROOT}" rev-parse --verify HEAD)"
[[ "${submission_head}" = "${PRORM_GIT_COMMIT}" ]] \
  || die "submission checkout changed commit"
[[ -z "$(git -C "${PRORM_REPO_ROOT}" status --porcelain --untracked-files=normal)" ]] \
  || die "submission checkout is no longer clean"
for binding in \
  "${PRORM_POST_RECOVERY_OVERLAY_REL}:${PRORM_POST_RECOVERY_OVERLAY_SHA256}" \
  "${PRORM_PHASE2_BASE_REL}:${PRORM_PHASE2_BASE_SHA256}"; do
  relative="${binding%%:*}"
  expected="${binding#*:}"
  observed="$(
    git -C "${PRORM_REPO_ROOT}" cat-file blob \
      "${PRORM_GIT_COMMIT}:${relative}" | sha256sum | awk '{print $1}'
  )"
  [[ "${observed}" = "${expected}" ]] \
    || die "committed config differs from submitted bytes: ${relative}"
done

seeds=(20260801 20260802 20260803)
seed="${seeds[SLURM_ARRAY_TASK_ID]}"
execution_id="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
scratch_jobs_root="${PRORM_SCRATCH_ROOT}/phase2-post-recovery-${PRORM_POST_RECOVERY_NAMESPACE}-jobs"
campaign_root="${PRORM_PROJECT_ROOT}/runs/phase2-post-recovery-${PRORM_POST_RECOVERY_NAMESPACE}/${PRORM_POST_RECOVERY_DESIGN_SHA256}"
artifact_campaign_root="${PRORM_PROJECT_ROOT}/artifacts/phase2-post-recovery-${PRORM_POST_RECOVERY_NAMESPACE}/${PRORM_POST_RECOVERY_DESIGN_SHA256}"
submission_registry="${campaign_root}/submission-registry"
require_directory submission_registry
contained submission_registry "${submission_registry}" "${PRORM_PROJECT_ROOT}"
runtime_export_spec="PATH=/usr/local/bin:/usr/bin:/bin,PRORM_PROJECT_ROOT=${PRORM_PROJECT_ROOT},PRORM_SCRATCH_ROOT=${PRORM_SCRATCH_ROOT},PRORM_REPO_ROOT=${PRORM_REPO_ROOT},PRORM_IMAGE=${PRORM_IMAGE},PRORM_IMAGE_SHA256=${PRORM_IMAGE_SHA256},PRORM_HF_CACHE=${PRORM_HF_CACHE},PRORM_HF_INVENTORY=${PRORM_HF_INVENTORY},PRORM_HF_INVENTORY_SHA256=${PRORM_HF_INVENTORY_SHA256},PRORM_POST_RECOVERY_OVERLAY_REL=${PRORM_POST_RECOVERY_OVERLAY_REL},PRORM_PHASE2_BASE_REL=${PRORM_PHASE2_BASE_REL},PRORM_POST_RECOVERY_OVERLAY_SHA256=${PRORM_POST_RECOVERY_OVERLAY_SHA256},PRORM_PHASE2_BASE_SHA256=${PRORM_PHASE2_BASE_SHA256},PRORM_POST_RECOVERY_DESIGN_SHA256=${PRORM_POST_RECOVERY_DESIGN_SHA256},PRORM_PHASE2_BASE_CONFIG_HASH=${PRORM_PHASE2_BASE_CONFIG_HASH},PRORM_RECOVERY_AUTHORIZATION=${PRORM_RECOVERY_AUTHORIZATION},PRORM_RECOVERY_AUTHORIZATION_SHA256=${PRORM_RECOVERY_AUTHORIZATION_SHA256},PRORM_OPTIMIZER_SCHEDULE_SHA256=${PRORM_OPTIMIZER_SCHEDULE_SHA256},PRORM_GIT_COMMIT=${PRORM_GIT_COMMIT},PRORM_POST_RECOVERY_PILOT_PHASE=${PRORM_POST_RECOVERY_PILOT_PHASE},PRORM_POST_RECOVERY_NAMESPACE=${PRORM_POST_RECOVERY_NAMESPACE},PRORM_PHASE2_BETA_SOURCE_AGGREGATE_PRESENT=${PRORM_PHASE2_BETA_SOURCE_AGGREGATE_PRESENT},PRORM_PHASE2_HORIZON_PARENT_AGGREGATE_PRESENT=${PRORM_PHASE2_HORIZON_PARENT_AGGREGATE_PRESENT}"
if [[ "${PRORM_PHASE2_BETA_SOURCE_AGGREGATE_PRESENT}" = "1" ]]; then
  runtime_export_spec+=",PRORM_PHASE2_BETA_SOURCE_AGGREGATE=${PRORM_PHASE2_BETA_SOURCE_AGGREGATE},PRORM_PHASE2_BETA_SOURCE_AGGREGATE_SHA256=${PRORM_PHASE2_BETA_SOURCE_AGGREGATE_SHA256}"
fi
if [[ "${PRORM_PHASE2_HORIZON_PARENT_AGGREGATE_PRESENT}" = "1" ]]; then
  runtime_export_spec+=",PRORM_PHASE2_HORIZON_PARENT_AGGREGATE=${PRORM_PHASE2_HORIZON_PARENT_AGGREGATE},PRORM_PHASE2_HORIZON_PARENT_AGGREGATE_SHA256=${PRORM_PHASE2_HORIZON_PARENT_AGGREGATE_SHA256}"
fi
runtime_export_spec_sha256="$(
  printf '%s' "${runtime_export_spec}" | sha256sum | awk '{print $1}'
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
  --optimizer-schedule-sha256 "${PRORM_OPTIMIZER_SCHEDULE_SHA256}" \
  --git-commit "${PRORM_GIT_COMMIT}" \
  --image-sha256 "${PRORM_IMAGE_SHA256}" \
  --inventory-sha256 "${PRORM_HF_INVENTORY_SHA256}" \
  --overlay-sha256 "${PRORM_POST_RECOVERY_OVERLAY_SHA256}" \
  --base-file-sha256 "${PRORM_PHASE2_BASE_SHA256}" \
  --export-spec-sha256 "${runtime_export_spec_sha256}" \
  --array-job-id "${SLURM_ARRAY_JOB_ID}" \
  --submitter-user "$(id -un)"
)" || die "post-recovery array submission registry validation failed"
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
  || die "submission registry hashes could not be resolved"
submission_intent_sha256="${submission_hashes[0]}"
submission_ledger_sha256="${submission_hashes[1]}"
job_dir="${scratch_jobs_root}/${execution_id}"
seed_root="${campaign_root}/seed-${seed}"
project_run="${seed_root}/job-${execution_id}"
artifact_seed_root="${artifact_campaign_root}/seed-${seed}"
project_artifact="${artifact_seed_root}/job-${execution_id}"
for path in "${job_dir}" "${seed_root}" "${project_run}" "${project_artifact}"; do
  [[ ! -e "${path}" && ! -L "${path}" ]] \
    || die "fresh post-recovery namespace already exists: ${path}"
done
mkdir -p "${scratch_jobs_root}" "${campaign_root}" "${artifact_campaign_root}"
mkdir -- "${job_dir}" "${seed_root}" "${project_run}"
mkdir -p -- "${artifact_seed_root}"
for path in "${job_dir}" "${seed_root}" "${project_run}" "${artifact_seed_root}"; do
  [[ -d "${path}" && ! -L "${path}" \
    && "$(realpath -e -- "${path}")" = "${path}" ]] \
    || die "new post-recovery namespace is unsafe: ${path}"
done

atomic_copy() {
  local source="$1" name="$2" target="${project_run}/${name}" temporary=""
  [[ -f "${source}" && ! -L "${source}" ]] || return 1
  [[ ! -e "${target}" && ! -L "${target}" ]] || return 1
  temporary="$(mktemp "${project_run}/.${name}.XXXXXX")" || return 1
  cp -p -- "${source}" "${temporary}" || return 1
  mv -T -- "${temporary}" "${target}" || return 1
}
hash_or_none() {
  local path="$1"
  if [[ -f "${path}" && ! -L "${path}" ]]; then
    sha256sum -- "${path}" | awk '{print $1}'
  else
    printf 'none\n'
  fi
}
write_marker() {
  local status="$1" workload="$2" final="$3" temporary
  [[ ! -e "${project_run}/SUCCESS" && ! -L "${project_run}/SUCCESS" \
    && ! -e "${project_run}/FAILED" && ! -L "${project_run}/FAILED" ]] \
    || return 1
  temporary="$(mktemp "${project_run}/.${status}.XXXXXX")" || return 1
  {
    printf 'schema_version=prorm-phase2-post-recovery-pilot-run-status/v1\n'
    printf 'status=%s\n' "${status}"
    printf 'pilot_phase=%s\n' "${PRORM_POST_RECOVERY_PILOT_PHASE}"
    printf 'workload_exit_code=%s\n' "${workload}"
    printf 'final_exit_code=%s\n' "${final}"
    printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID}"
    printf 'allocation_job_id_raw=%s\n' "${SLURM_JOB_ID}"
    printf 'slurm_array_task_job_id=%s\n' "${execution_id}"
    printf 'array_job_id=%s\n' "${SLURM_ARRAY_JOB_ID}"
    printf 'array_task_id=%s\n' "${SLURM_ARRAY_TASK_ID}"
    printf 'seed=%s\n' "${seed}"
    printf 'cluster=hpc4\n'
    printf 'account=sigroup\n'
    printf 'partition=gpu-l20\n'
    printf 'restart_count=0\n'
    printf 'phase2_design_sha256=%s\n' "${PRORM_POST_RECOVERY_DESIGN_SHA256}"
    printf 'base_config_hash=%s\n' "${PRORM_PHASE2_BASE_CONFIG_HASH}"
    printf 'git_commit=%s\n' "${PRORM_GIT_COMMIT}"
    printf 'recovery_authorization_sha256=%s\n' \
      "${PRORM_RECOVERY_AUTHORIZATION_SHA256}"
    printf 'optimizer_schedule_sha256=%s\n' "${PRORM_OPTIMIZER_SCHEDULE_SHA256}"
    printf 'submission_intent_sha256=%s\n' "${submission_intent_sha256}"
    printf 'submission_ledger_sha256=%s\n' "${submission_ledger_sha256}"
    printf 'materialization_mode=fresh\n'
    printf 'recovery_outputs_mounted=false\n'
    printf 'hf_root_mount_mode=read_only\n'
    printf 'datasets_cache_scope=job_local\n'
    printf 'artifact_metadata_sha256=%s\n' \
      "$(hash_or_none "${job_dir}/artifact/metadata.json")"
    printf 'phase2_result_sha256=%s\n' \
      "$(hash_or_none "${job_dir}/phase2-pilot-diagnostics.json")"
    printf 'phase2_output_verification_sha256=%s\n' \
      "$(hash_or_none "${job_dir}/phase2-output-verification.json")"
    printf 'post_recovery_output_verification_sha256=%s\n' \
      "$(hash_or_none "${job_dir}/post-recovery-output-verification.json")"
    printf 'created_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } >"${temporary}" || return 1
  mv -T -- "${temporary}" "${project_run}/${status}"
}
finalize() {
  local workload=$? sync=0 final marker file
  trap - EXIT
  set +e
  for file in \
    authorization-check-host-before.json authorization-check-container.json \
    authorization-check-host-after.json authorization-check-container-after.json \
    phase2-config-check.json \
    base-config-check.json post-recovery-predecessor-check.json \
    hf-inventory-verification.json gpu-check.json \
    run-manifest.json env-report.log materialize.log memory-materialize.json \
    artifact-materialization.json artifact-verification.json phase2-run.log \
    memory-phase2.json phase2-failure-evidence.json \
    phase2-pilot-diagnostics.json phase2-pilot-diagnostics.diagnostics.jsonl \
    phase2-output-verification.json post-recovery-output-verification.json; do
    if [[ -e "${job_dir}/${file}" || -L "${job_dir}/${file}" ]]; then
      atomic_copy "${job_dir}/${file}" "${file}" || sync=1
    fi
  done
  if [[ "${workload}" = "0" ]]; then
    for file in \
      authorization-check-host-before.json authorization-check-container.json \
      authorization-check-host-after.json authorization-check-container-after.json \
      phase2-config-check.json \
      base-config-check.json post-recovery-predecessor-check.json \
      hf-inventory-verification.json gpu-check.json \
      run-manifest.json env-report.log materialize.log memory-materialize.json \
      artifact-materialization.json artifact-verification.json phase2-run.log \
      memory-phase2.json phase2-pilot-diagnostics.json \
      phase2-pilot-diagnostics.diagnostics.jsonl phase2-output-verification.json \
      post-recovery-output-verification.json; do
      [[ -f "${project_run}/${file}" && ! -L "${project_run}/${file}" ]] || sync=1
    done
    [[ -L "${project_run}/artifact" && -d "${project_run}/artifact" ]] || sync=1
  fi
  final="${workload}"
  (( sync == 0 )) || [[ "${final}" != "0" ]] || final=1
  if [[ "${final}" = "0" ]]; then marker=SUCCESS; else marker=FAILED; fi
  write_marker "${marker}" "${workload}" "${final}" || final=1
  exit "${final}"
}
trap finalize EXIT

execution_repo="${job_dir}/source"
git clone --quiet --no-hardlinks --no-checkout \
  "${PRORM_REPO_ROOT}" "${execution_repo}"
git -C "${execution_repo}" checkout --quiet --detach "${PRORM_GIT_COMMIT}"
execution_overlay="${execution_repo}/${PRORM_POST_RECOVERY_OVERLAY_REL}"
execution_base="${execution_repo}/${PRORM_PHASE2_BASE_REL}"
for binding in \
  "${execution_overlay}:${PRORM_POST_RECOVERY_OVERLAY_SHA256}" \
  "${execution_base}:${PRORM_PHASE2_BASE_SHA256}"; do
  path="${binding%%:*}"
  expected="${binding#*:}"
  [[ -f "${path}" && ! -L "${path}" ]] || die "detached config is missing"
  printf '%s  %s\n' "${expected}" "${path}" \
    | sha256sum --check --status || die "detached config differs from submission"
done
assert_checkout() {
  [[ "$(git -C "${execution_repo}" rev-parse --verify HEAD)" = "${PRORM_GIT_COMMIT}" \
    && -z "$(git -C "${execution_repo}" status --porcelain --untracked-files=normal)" ]] \
    || die "detached execution checkout changed"
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
  --project-root "${PRORM_PROJECT_ROOT}" --format json \
  >"${job_dir}/authorization-check-host-before.json"
mapfile -t authorization_transport < <(
  "${host_python}" -I -S "${authorization_inspector}" authorization \
    "${PRORM_RECOVERY_AUTHORIZATION}" \
    --expected-sha256 "${PRORM_RECOVERY_AUTHORIZATION_SHA256}" \
    --project-root "${PRORM_PROJECT_ROOT}"
)
[[ "${#authorization_transport[@]}" -ge 3 \
  && "${authorization_transport[2]}" =~ ^[0-9]+$ \
  && "${#authorization_transport[@]}" = \
    "$((3 + authorization_transport[2]))" ]] \
  || die "authorization transport inspector returned an invalid bind plan"
authorization_dependency_binds=()
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
  authorization_dependency_binds+=(
    --bind "${dependency_path}:${dependency_path}:ro"
  )
done

datasets_cache="${job_dir}/hf-datasets-cache"
tmp_dir="${job_dir}/tmp"
xdg_cache="${job_dir}/xdg-cache"
mkdir -m 700 -- "${datasets_cache}" "${tmp_dir}" "${xdg_cache}"
for path in "${datasets_cache}" "${tmp_dir}" "${xdg_cache}"; do
  [[ -d "${path}" && ! -L "${path}" \
    && "$(realpath -e -- "${path}")" = "${path}" ]] \
    || die "job-local writable cache is unsafe"
done

# Only this job directory is writable in the container.  The immutable HF
# root, final authorization, and the inspector-approved Gate-R evidence closure
# are mounted individually read-only. No recovery head, checkpoint, optimizer
# state, task-output directory, or broad Gate-R root is mounted.
container=(
  apptainer exec --cleanenv --nv --no-mount home,cwd,bind-paths
  --pwd "${execution_repo}"
  --bind "${job_dir}:${job_dir}"
  --bind "${PRORM_HF_CACHE}:${PRORM_HF_CACHE}:ro"
  --bind "${PRORM_RECOVERY_AUTHORIZATION}:${PRORM_RECOVERY_AUTHORIZATION}:ro"
  "${authorization_dependency_binds[@]}"
  --env "PYTHONPATH=${execution_repo}/src"
  --env "PYTHONNOUSERSITE=1"
  --env "PYTHONDONTWRITEBYTECODE=1"
  --env "HF_HOME=${PRORM_HF_CACHE}"
  --env "HF_HUB_CACHE=${PRORM_HF_CACHE}/hub"
  --env "HF_DATASETS_CACHE=${datasets_cache}"
  --env "TMPDIR=${tmp_dir}"
  --env "XDG_CACHE_HOME=${xdg_cache}"
  --env "HF_HUB_OFFLINE=1"
  --env "TRANSFORMERS_OFFLINE=1"
  --env "HF_DATASETS_OFFLINE=1"
  --env "HF_HUB_DISABLE_TELEMETRY=1"
  --env "TOKENIZERS_PARALLELISM=false"
  --env "PRORM_GIT_COMMIT=${PRORM_GIT_COMMIT}"
  --env "PRORM_IMAGE_SHA256=${PRORM_IMAGE_SHA256}"
  --env "PRORM_HF_INVENTORY_SHA256=${PRORM_HF_INVENTORY_SHA256}"
)
if [[ "${PRORM_PHASE2_HORIZON_PARENT_AGGREGATE_PRESENT}" = "1" ]]; then
  # Mount only target-free post-recovery evidence namespaces.  The recovery
  # run/artifact trees and all trained recovery heads remain unmounted.
  for evidence_root in \
    "${PRORM_PROJECT_ROOT}/aggregates" \
    "${PRORM_PROJECT_ROOT}/runs/phase2-post-recovery-calibration" \
    "${PRORM_PROJECT_ROOT}/artifacts/phase2-post-recovery-calibration"; do
    [[ -d "${evidence_root}" && ! -L "${evidence_root}" ]] \
      || die "required predecessor evidence namespace is missing or unsafe"
    container+=(--bind "${evidence_root}:${evidence_root}:ro")
  done
  # A freeze length failure can authorize a new-horizon calibration, and a
  # later calibration/freeze aggregate can recursively retain that freeze
  # ancestor.  Mount the freeze evidence pair whenever it exists, regardless
  # of the current phase.  A lineage that needs it cannot verify if it is
  # absent; a lineage with no freeze ancestor does not require empty dirs.
  freeze_run_evidence_root="${PRORM_PROJECT_ROOT}/runs/phase2-post-recovery-freeze"
  freeze_artifact_evidence_root="${PRORM_PROJECT_ROOT}/artifacts/phase2-post-recovery-freeze"
  if [[ -e "${freeze_run_evidence_root}" \
    || -e "${freeze_artifact_evidence_root}" ]]; then
    for evidence_root in \
      "${freeze_run_evidence_root}" "${freeze_artifact_evidence_root}"; do
      [[ -d "${evidence_root}" && ! -L "${evidence_root}" ]] \
        || die "freeze predecessor evidence namespace is missing or unsafe"
      container+=(--bind "${evidence_root}:${evidence_root}:ro")
    done
  fi
fi
for variable in \
  SLURM_JOB_ID SLURM_ARRAY_JOB_ID SLURM_ARRAY_TASK_ID \
  SLURM_ARRAY_TASK_COUNT SLURM_ARRAY_TASK_MIN SLURM_ARRAY_TASK_MAX \
  SLURM_ARRAY_TASK_STEP SLURM_JOB_NAME SLURM_JOB_ACCOUNT \
  SLURM_CLUSTER_NAME SLURM_JOB_PARTITION SLURM_RESTART_COUNT \
  SLURM_JOB_NODELIST SLURM_NNODES SLURM_NTASKS SLURM_CPUS_PER_TASK \
  SLURM_GPUS SLURM_GPUS_ON_NODE SLURM_PROCID SLURM_LOCALID \
  SLURM_NODEID CUDA_VISIBLE_DEVICES NVIDIA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL; do
  [[ -v "${variable}" ]] && container+=(--env "${variable}=${!variable}")
done
container+=("${PRORM_IMAGE}")

"${container[@]}" python \
  "${execution_repo}/scripts/hpc4/validate_phase2_recovery_authorization.py" \
  "${PRORM_RECOVERY_AUTHORIZATION}" "${execution_overlay}" \
  --expected-sha256 "${PRORM_RECOVERY_AUTHORIZATION_SHA256}" \
  --project-root "${PRORM_PROJECT_ROOT}" \
  >"${job_dir}/authorization-check-container.json"
"${container[@]}" python -m smart_reward.cli phase2-config-check \
  "${execution_overlay}" | tee "${job_dir}/phase2-config-check.json"
"${container[@]}" python -m smart_reward.cli config-check \
  "${execution_base}" | tee "${job_dir}/base-config-check.json"
"${host_python}" -I -S - \
  "${job_dir}/phase2-config-check.json" "${job_dir}/base-config-check.json" \
  "${PRORM_POST_RECOVERY_DESIGN_SHA256}" "${PRORM_PHASE2_BASE_CONFIG_HASH}" <<'PY'
import json,sys
from pathlib import Path
phase2=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
base=json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
if (
    phase2.get("status")!="ok"
    or phase2.get("phase2_design_sha256")!=sys.argv[3]
    or phase2.get("source_config_hash")!=sys.argv[4]
    or phase2.get("design_stage")!="pilot"
    or phase2.get("formal_eligibility") is not False
    or base.get("status")!="ok"
    or base.get("config_hash")!=sys.argv[4]
):
    raise SystemExit("container-computed post-recovery identities are invalid")
PY
"${container[@]}" python "${execution_repo}/scripts/hpc4/stage_hf_assets.py" \
  "${execution_base}" "${PRORM_HF_CACHE}" \
  --verify-only --inventory "${PRORM_HF_INVENTORY}" \
  --verification-datasets-cache "${datasets_cache}" \
  | tee "${job_dir}/hf-inventory-verification.json"
printf '%s  %s\n' "${PRORM_HF_INVENTORY_SHA256}" "${PRORM_HF_INVENTORY}" \
  | sha256sum --check --status || die "HF inventory changed during verification"
"${container[@]}" python - <<'PY' | tee "${job_dir}/gpu-check.json"
import json
import torch
if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
    raise SystemExit("post-recovery pilot requires exactly one CUDA GPU")
name=torch.cuda.get_device_name(0)
if name!="NVIDIA L20":
    raise SystemExit(f"expected NVIDIA L20, observed {name!r}")
print(json.dumps({"status":"ok","cuda_device_count":1,"gpu_model":name},sort_keys=True))
PY

predecessor_check="${job_dir}/post-recovery-predecessor-check.json"
"${container[@]}" python - \
  "${execution_overlay}" \
  "${PRORM_PHASE2_BETA_SOURCE_AGGREGATE_PRESENT}" \
  "${PRORM_PHASE2_BETA_SOURCE_AGGREGATE:-none}" \
  "${PRORM_PHASE2_HORIZON_PARENT_AGGREGATE_PRESENT}" \
  "${PRORM_PHASE2_HORIZON_PARENT_AGGREGATE:-none}" <<'PY' \
  >"${predecessor_check}"
import json
import sys
from smart_reward.phase2_config import load_phase2_config_bundle
from smart_reward.phase2_pilot_aggregate import (
    verify_beta_source_aggregate,
    verify_horizon_parent_aggregate,
)
from smart_reward.phase2_post_recovery_control import (
    verify_post_recovery_aggregate_success_receipt,
)

overlay, beta_present, beta_path, horizon_present, horizon_path = sys.argv[1:]
config = load_phase2_config_bundle(overlay).config
for present, path in (
    (beta_present, beta_path),
    (horizon_present, horizon_path),
):
    if present == "1":
        verify_post_recovery_aggregate_success_receipt(path)
beta = verify_beta_source_aggregate(
    config,
    None if beta_present == "0" else beta_path,
)
horizon = verify_horizon_parent_aggregate(
    config,
    None if horizon_present == "0" else horizon_path,
)
print(json.dumps({
    "schema_version": "prorm-phase2-post-recovery-predecessor-check/v1",
    "status": "passed",
    "pilot_phase": config["design"]["pilot_phase"],
    "beta_source_verified": beta is not None,
    "horizon_parent_verified": horizon is not None,
    "information_consumed": "target_free_pilot_gates_only",
    "reward_head_or_optimizer_state_reused": False,
}, allow_nan=False, sort_keys=True, separators=(",", ":")))
PY

"${container[@]}" python -m smart_reward.cli env-report \
  "${execution_base}" --seed "${seed}" --repo-root "${execution_repo}" \
  --output "${job_dir}/run-manifest.json" | tee "${job_dir}/env-report.log"

# Fresh materialization is mandatory.  There is intentionally no branch that
# copies or reuses a content-addressed artifact from the failed/recovery runs.
artifact_dir="${job_dir}/artifact"
"${container[@]}" env \
  "PRORM_MEMORY_REPORT=${job_dir}/memory-materialize.json" \
  python -m smart_reward.cli controlled-materialize \
  "${execution_base}" "${artifact_dir}" --seed "${seed}" --device cuda \
  2>&1 | tee "${job_dir}/materialize.log"
[[ -d "${artifact_dir}" && ! -L "${artifact_dir}" \
  && -f "${artifact_dir}/metadata.json" \
  && ! -L "${artifact_dir}/metadata.json" ]] \
  || die "fresh controlled materialization did not produce a complete artifact"
artifact_metadata_sha256="$(
  sha256sum -- "${artifact_dir}/metadata.json" | awk '{print $1}'
)"
"${host_python}" -I -S - \
  "${job_dir}/artifact-materialization.json" \
  "${seed}" "${PRORM_POST_RECOVERY_DESIGN_SHA256}" \
  "${PRORM_PHASE2_BASE_CONFIG_HASH}" "${artifact_metadata_sha256}" \
  "${PRORM_RECOVERY_AUTHORIZATION_SHA256}" <<'PY'
import json,sys
from pathlib import Path
out,seed,design,base,metadata,authorization=sys.argv[1:]
value={
 "schema_version":"prorm-phase2-post-recovery-artifact-binding/v1",
 "mode":"fresh",
 "seed":int(seed),
 "phase2_design_sha256":design,
 "base_config_hash":base,
 "artifact_metadata_sha256":metadata,
 "recovery_authorization_sha256":authorization,
 "recovery_artifact_reused":False,
 "recovery_reward_heads_reused":False,
 "recovery_optimizer_state_reused":False,
}
with Path(out).open("x",encoding="utf-8",newline="\n") as stream:
 json.dump(value,stream,allow_nan=False,sort_keys=True,separators=(",",":"));stream.write("\n")
PY

# Explicit bridge verification precedes the full run.  This is the required
# controlled-materialize -> prepare_phase2_inputs -> phase2-run ordering.
"${container[@]}" python - \
  "${execution_overlay}" "${seed}" "${artifact_dir}" \
  "${job_dir}/run-manifest.json" <<'PY' | tee "${job_dir}/artifact-verification.json"
import json,sys
from smart_reward.phase2_inputs import prepare_phase2_inputs
inputs=prepare_phase2_inputs(
    sys.argv[1],
    seed=int(sys.argv[2]),
    artifact_dir=sys.argv[3],
    run_manifest=sys.argv[4],
)
print(json.dumps({
    "status":"ok",
    "seed":inputs.seed,
    "base_config_hash":inputs.source_config_hash,
    "phase2_design_sha256":inputs.phase2_config_hash,
    "artifact_metadata_sha256":inputs.artifact_metadata_sha256,
    "formal_environment":inputs.environment_identity["formal"],
},allow_nan=False,sort_keys=True))
PY

result="${job_dir}/phase2-pilot-diagnostics.json"
diagnostics="${job_dir}/phase2-pilot-diagnostics.diagnostics.jsonl"
phase2_run=(
  python -m smart_reward.cli phase2-run
  "${execution_overlay}" "${artifact_dir}" "${job_dir}/run-manifest.json"
  "${result}" --seed "${seed}" --device cuda
)
if [[ "${PRORM_PHASE2_BETA_SOURCE_AGGREGATE_PRESENT}" = "1" ]]; then
  phase2_run+=(
    --beta-source-aggregate "${PRORM_PHASE2_BETA_SOURCE_AGGREGATE}"
  )
fi
if [[ "${PRORM_PHASE2_HORIZON_PARENT_AGGREGATE_PRESENT}" = "1" ]]; then
  phase2_run+=(
    --horizon-parent-aggregate "${PRORM_PHASE2_HORIZON_PARENT_AGGREGATE}"
  )
fi
"${container[@]}" env \
  "PRORM_MEMORY_REPORT=${job_dir}/memory-phase2.json" \
  "PRORM_FAILURE_EVIDENCE=${job_dir}/phase2-failure-evidence.json" \
  "${phase2_run[@]}" \
  2>&1 | tee "${job_dir}/phase2-run.log"
assert_checkout

"${container[@]}" python \
  "${execution_repo}/scripts/hpc4/validate_phase2_post_recovery_output.py" \
  "${execution_overlay}" "${PRORM_RECOVERY_AUTHORIZATION}" \
  "${result}" "${diagnostics}" \
  "${job_dir}/phase2-output-verification.json" \
  "${job_dir}/post-recovery-output-verification.json" \
  --authorization-sha256 "${PRORM_RECOVERY_AUTHORIZATION_SHA256}" \
  --seed "${seed}" \
  --design-sha256 "${PRORM_POST_RECOVERY_DESIGN_SHA256}" \
  --base-config-hash "${PRORM_PHASE2_BASE_CONFIG_HASH}" \
  --git-commit "${PRORM_GIT_COMMIT}" \
  --image-sha256 "${PRORM_IMAGE_SHA256}" \
  --hf-inventory-sha256 "${PRORM_HF_INVENTORY_SHA256}" \
  --artifact-metadata-sha256 "${artifact_metadata_sha256}" \
  --slurm-job-id-raw "${SLURM_JOB_ID}" \
  --array-job-id "${SLURM_ARRAY_JOB_ID}" \
  --array-task-id "${SLURM_ARRAY_TASK_ID}" \
  --pilot-phase "${PRORM_POST_RECOVERY_PILOT_PHASE}"

# Publish this newly materialized artifact only after the full target-free run
# and five-head schedule verification pass.  Existing namespaces are fatal.
artifact_staging="$(
  mktemp -d "${artifact_seed_root}/.${execution_id}.staged.XXXXXX"
)"
cp -a -- "${artifact_dir}/." "${artifact_staging}/"
[[ ! -e "${project_artifact}" && ! -L "${project_artifact}" ]] \
  || die "fresh artifact destination appeared before publication"
mv -T --no-clobber -- "${artifact_staging}" "${project_artifact}" \
  || die "atomic no-overwrite fresh artifact publication failed"
[[ ! -e "${artifact_staging}" && ! -L "${artifact_staging}" ]] \
  || die "fresh artifact publication lost an atomic rename race"
published_metadata_sha256="$(
  sha256sum -- "${project_artifact}/metadata.json" | awk '{print $1}'
)"
[[ "${published_metadata_sha256}" = "${artifact_metadata_sha256}" ]] \
  || die "published fresh artifact metadata differs from scratch"
artifact_relative="$(realpath --relative-to="${project_run}" "${project_artifact}")"
[[ "${artifact_relative}" != /* ]] || die "artifact link target must be relative"
ln -s -- "${artifact_relative}" "${project_run}/artifact"

"${host_python}" -I -S "${authorization_inspector}" authorization \
  "${PRORM_RECOVERY_AUTHORIZATION}" \
  --expected-sha256 "${PRORM_RECOVERY_AUTHORIZATION_SHA256}" \
  --project-root "${PRORM_PROJECT_ROOT}" --format json \
  >"${job_dir}/authorization-check-host-after.json"
"${container[@]}" python \
  "${execution_repo}/scripts/hpc4/validate_phase2_recovery_authorization.py" \
  "${PRORM_RECOVERY_AUTHORIZATION}" "${execution_overlay}" \
  --expected-sha256 "${PRORM_RECOVERY_AUTHORIZATION_SHA256}" \
  --project-root "${PRORM_PROJECT_ROOT}" \
  >"${job_dir}/authorization-check-container-after.json"
for binding in \
  "${PRORM_IMAGE}:${PRORM_IMAGE_SHA256}" \
  "${PRORM_HF_INVENTORY}:${PRORM_HF_INVENTORY_SHA256}" \
  "${PRORM_RECOVERY_AUTHORIZATION}:${PRORM_RECOVERY_AUTHORIZATION_SHA256}" \
  "${predecessor_bindings[@]}"; do
  path="${binding%%:*}"
  expected="${binding#*:}"
  printf '%s  %s\n' "${expected}" "${path}" \
    | sha256sum --check --status || die "immutable input changed during execution"
done
assert_checkout
printf 'post-recovery %s complete: seed=%s run=%s\n' \
  "${PRORM_POST_RECOVERY_PILOT_PHASE}" "${seed}" "${project_run}"
