#!/usr/bin/env bash
# R3 Gate-P compute body.  The submitter supplies the exact Slurm resources.

set -euo pipefail
umask 077

die() {
  printf 'R3 Gate-P fatal: %s\n' "$*" >&2
  exit 1
}

required_environment=(
  PRORM_R3_GIT_COMMIT
  PRORM_R3_IMAGE
  PRORM_R3_IMAGE_SHA256
  PRORM_R3_REPO_ROOT
  PRORM_R3_PROJECT_ROOT
  PRORM_R3_SCRATCH_ROOT
  PRORM_R3_HF_CACHE
  PRORM_R3_SCIENCE_CONFIG
  PRORM_R3_SOURCE_CONFIG
  PRORM_R3_PARENT_REGISTRY
  PRORM_R3_PARENT_REGISTRY_FILE_SHA256
  PRORM_R3_GATE0_FILE_SHA256
  PRORM_R3_GATE1_FILE_SHA256
  PRORM_R3_SOURCE_TEST_RECEIPT_FILE_SHA256
  PRORM_R3_SCHEDULER_RAW_EVIDENCE
  PRORM_R3_SCHEDULER_RAW_EVIDENCE_SHA256
  PRORM_R3_RESOURCE_RAW_EVIDENCE
  PRORM_R3_RESOURCE_RAW_EVIDENCE_SHA256
  PRORM_R3_PARTITION
  PRORM_R3_GPU_NAME
  PRORM_R3_GPU_TOTAL_MEMORY_BYTES
  PRORM_R3_MAX_ALLOCATION_WALL_SECONDS
  PRORM_R3_MAX_ARRAY_CONCURRENCY
  PRORM_R3_MAX_SCHEDULER_SEGMENTS
  PRORM_R3_MAX_GPUS_PER_TASK
  PRORM_R3_MAX_CPUS_PER_TASK
  PRORM_R3_MAX_MEMORY_BYTES
  PRORM_R3_WALLTIME_MARGIN_FRACTION
  PRORM_R3_FIXED_WALLTIME_MARGIN_SECONDS
  PRORM_R3_MEMORY_MARGIN_FRACTION
  PRORM_R3_SIGNAL_MARGIN_SECONDS
  PRORM_R3_CHECKPOINT_CADENCE_UPDATES
  PRORM_R3_PRIMARY_WALLTIME_SECONDS
  PRORM_R3_PRIMARY_ARRAY_CONCURRENCY
  PRORM_R3_PRIMARY_CPUS_PER_TASK
  PRORM_R3_PRIMARY_MEMORY_BYTES
  PRORM_R3_PROFILE_INTENT
  PRORM_R3_PROFILE_INTENT_FILE_SHA256
  PRORM_R3_OPERATIONAL_BUNDLE
  PRORM_R3_PROFILE_RUNTIME_RECEIPT
  PRORM_R3_IO_PROBE_DIRECTORY
)
for name in "${required_environment[@]}"; do
  [[ -n "${!name:-}" ]] || die "missing environment variable ${name}"
done
command -v cmp >/dev/null 2>&1 || die "cmp is unavailable"
command -v git >/dev/null 2>&1 || die "git is unavailable"
command -v realpath >/dev/null 2>&1 || die "realpath is unavailable"
command -v sha256sum >/dev/null 2>&1 || die "sha256sum is unavailable"
command -v stat >/dev/null 2>&1 || die "stat is unavailable"

[[ "${SLURM_JOB_ACCOUNT:-}" == "sigroup" ]] || die "unexpected Slurm account"
[[ "${SLURM_JOB_PARTITION:-}" == "${PRORM_R3_PARTITION}" ]] \
  || die "unexpected Slurm partition"
[[ -n "${SLURM_JOB_ID:-}" ]] || die "not running inside a Slurm job"
[[ "${SLURM_NNODES:-}" == "1" ]] || die "Gate-P requires exactly one Slurm node"
[[ "${SLURM_NTASKS:-}" == "1" ]] || die "Gate-P requires exactly one Slurm task"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != *","* ]] \
  || die "Gate-P requires exactly one visible CUDA device"
[[ -z "${SLURM_ARRAY_JOB_ID:-}" ]] || die "Gate-P must not be an array job"
[[ -z "${SLURM_ARRAY_TASK_ID:-}" ]] || die "Gate-P must not have an array task"

repo_root="$(realpath -e -- "${PRORM_R3_REPO_ROOT}")"
project_root="$(realpath -e -- "${PRORM_R3_PROJECT_ROOT}")"
scratch_root="$(realpath -e -- "${PRORM_R3_SCRATCH_ROOT}")"
image="$(realpath -e -- "${PRORM_R3_IMAGE}")"
hf_cache="$(realpath -e -- "${PRORM_R3_HF_CACHE}")"
science_config="$(realpath -e -- "${PRORM_R3_SCIENCE_CONFIG}")"
source_config="$(realpath -e -- "${PRORM_R3_SOURCE_CONFIG}")"
parent_registry="$(realpath -e -- "${PRORM_R3_PARENT_REGISTRY}")"
scheduler_raw_evidence="$(realpath -e -- "${PRORM_R3_SCHEDULER_RAW_EVIDENCE}")"
resource_raw_evidence="$(realpath -e -- "${PRORM_R3_RESOURCE_RAW_EVIDENCE}")"
profile_intent="$(realpath -e -- "${PRORM_R3_PROFILE_INTENT}")"
io_probe_directory="$(realpath -e -- "${PRORM_R3_IO_PROBE_DIRECTORY}")"
source_test_receipt="$(
  realpath -e -- \
    "${project_root}/runs/phase2-recovery-r3/gate1/r3-source-test-receipt.json"
)"

[[ "${PRORM_R3_REPO_ROOT}" == "${repo_root}" ]] \
  || die "production repository root must be canonical"
[[ "${PRORM_R3_PROJECT_ROOT}" == "${project_root}" ]] \
  || die "persistent project root must be canonical"
[[ "${repo_root}" == "/home/yyangjo/Smart-Reward-Model" ]] \
  || die "unexpected production Git repository root"
[[ "${project_root}" == "/project/sigroup/smart-reward-model" ]] \
  || die "unexpected persistent project root"
[[ "${repo_root}" != "${project_root}" && \
  "${repo_root}" != "${project_root}/"* && \
  "${project_root}" != "${repo_root}/"* ]] \
  || die "production repository and project roots must not overlap"
[[ -e "${repo_root}/.git" && ! -L "${repo_root}/.git" ]] \
  || die "production repository root is not a Git checkout"
[[ ! -e "${project_root}/.git" && ! -L "${project_root}/.git" ]] \
  || die "persistent project root must not be a Git checkout"
[[ "${scratch_root}" == "/scratch/yyangjo" ]] || die "unexpected scratch root"
[[ -f "${image}" ]] || die "container is not a regular file"
[[ "${image}" == "${project_root}/"* ]] \
  || die "container must be inside the persistent project root"
[[ -d "${hf_cache}" && "${hf_cache}" == "${project_root}/"* ]] \
  || die "HF cache must be inside the persistent project root"
[[ "${science_config}" == \
  "${repo_root}/configs/phase2_recovery_r3_science.yaml" ]] \
  || die "science config must be the frozen R3 production config"
input_root="${project_root}/runs/phase2-recovery-r3/inputs/${PRORM_R3_GIT_COMMIT}"
[[ "${source_config}" == "${input_root}/common_beta_pilot_base.yaml" ]] \
  || die "source config is not the retained clean-commit copy"
[[ "${parent_registry}" == \
  "${input_root}/phase2_recovery_parent_failures.json" ]] \
  || die "parent registry is not the retained clean-commit copy"
[[ "$(stat -c '%a' -- "${source_config}")" == "440" ]] \
  || die "source config retained copy must have mode 0440"
[[ "$(stat -c '%a' -- "${parent_registry}")" == "440" ]] \
  || die "parent registry retained copy must have mode 0440"
[[ "${scheduler_raw_evidence}" == "${project_root}/"* ]] \
  || die "scheduler evidence must be durable project evidence"
[[ "${resource_raw_evidence}" == "${project_root}/"* ]] \
  || die "resource evidence must be durable project evidence"
[[ "${profile_intent}" == "${project_root}/"* ]] \
  || die "profile intent must be durable project evidence"
[[ "${io_probe_directory}" == "${scratch_root}/"* ]] \
  || die "I/O probe directory escaped the fixed scratch root"

operational_parent="$(
  realpath -e -- "$(dirname -- "${PRORM_R3_OPERATIONAL_BUNDLE}")"
)"
operational_bundle="${operational_parent}/$(basename -- "${PRORM_R3_OPERATIONAL_BUNDLE}")"
[[ "${operational_bundle}" == "${PRORM_R3_OPERATIONAL_BUNDLE}" && \
  "${operational_bundle}" == "${project_root}/"* ]] \
  || die "operational bundle must be a canonical durable project path"
runtime_parent="$(
  realpath -e -- "$(dirname -- "${PRORM_R3_PROFILE_RUNTIME_RECEIPT}")"
)"
runtime_receipt="${runtime_parent}/$(basename -- "${PRORM_R3_PROFILE_RUNTIME_RECEIPT}")"
[[ "${runtime_receipt}" == "${PRORM_R3_PROFILE_RUNTIME_RECEIPT}" && \
  "${runtime_receipt}" == "${project_root}/"* ]] \
  || die "runtime receipt must be a canonical durable project path"

observed_commit="$(git -C "${repo_root}" rev-parse HEAD)"
[[ "${observed_commit}" == "${PRORM_R3_GIT_COMMIT}" ]] || die "Git commit drift"
[[ -z "$(git -C "${repo_root}" status --porcelain --untracked-files=all)" ]] \
  || die "production checkout is not clean"
verify_retained_copy() {
  local source="$1"
  local retained="$2"
  local source_digest
  source_digest="$(sha256sum -- "${source}")"
  source_digest="${source_digest%% *}"
  [[ "${source_digest}" =~ ^[0-9a-f]{64}$ ]] \
    || die "invalid clean-repository input SHA-256"
  printf '%s  %s\n' "${source_digest}" "${retained}" \
    | sha256sum --check --status \
    || die "retained input copy SHA-256 differs from clean repository bytes"
  cmp --silent -- "${source}" "${retained}" \
    || die "retained input copy differs from clean repository bytes"
}
verify_retained_copy \
  "${repo_root}/configs/common_beta_pilot_base.yaml" \
  "${source_config}"
verify_retained_copy \
  "${repo_root}/configs/phase2_recovery_parent_failures.json" \
  "${parent_registry}"
printf '%s  %s\n' "${PRORM_R3_IMAGE_SHA256}" "${image}" \
  | sha256sum --check --status || die "container SHA-256 mismatch"
printf '%s  %s\n' \
  "${PRORM_R3_PARENT_REGISTRY_FILE_SHA256}" \
  "${parent_registry}" \
  | sha256sum --check --status || die "parent registry SHA-256 mismatch"
printf '%s  %s\n' \
  "${PRORM_R3_SOURCE_TEST_RECEIPT_FILE_SHA256}" \
  "${source_test_receipt}" \
  | sha256sum --check --status || die "source-test receipt SHA-256 mismatch"

runner="${repo_root}/scripts/hpc4/run_phase2_r3_gatep.py"
[[ -f "${runner}" ]] || die "Gate-P runner is missing"

apptainer exec \
  --cleanenv \
  --nv \
  --bind "${repo_root}:${repo_root}:ro" \
  --bind "${project_root}:${project_root}:rw" \
  --bind "${scratch_root}:${scratch_root}:rw" \
  --bind "${image}:${image}:ro" \
  --env "PYTHONPATH=${repo_root}/src" \
  --env "HF_HOME=${hf_cache}" \
  --env "HF_HUB_CACHE=${hf_cache}/hub" \
  --env "HF_HUB_OFFLINE=1" \
  --env "TRANSFORMERS_OFFLINE=1" \
  --env "PRORM_R3_REPO_ROOT=${repo_root}" \
  --env "PRORM_R3_PROJECT_ROOT=${project_root}" \
  --env "PRORM_R3_GIT_COMMIT=${PRORM_R3_GIT_COMMIT}" \
  --env "SLURM_CLUSTER_NAME=${SLURM_CLUSTER_NAME:-hpc4}" \
  --env "SLURM_JOB_ID=${SLURM_JOB_ID}" \
  --env "SLURM_JOB_ACCOUNT=${SLURM_JOB_ACCOUNT}" \
  --env "SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION}" \
  "${image}" \
  python3 "${runner}" \
    --science-config "${science_config}" \
    --source-config "${source_config}" \
    --parent-registry "${parent_registry}" \
    --parent-registry-file-sha256 "${PRORM_R3_PARENT_REGISTRY_FILE_SHA256}" \
    --gate0-file-sha256 "${PRORM_R3_GATE0_FILE_SHA256}" \
    --gate1-file-sha256 "${PRORM_R3_GATE1_FILE_SHA256}" \
    --source-test-receipt-file-sha256 \
      "${PRORM_R3_SOURCE_TEST_RECEIPT_FILE_SHA256}" \
    --scheduler-raw-evidence "${scheduler_raw_evidence}" \
    --scheduler-raw-evidence-sha256 \
      "${PRORM_R3_SCHEDULER_RAW_EVIDENCE_SHA256}" \
    --resource-raw-evidence "${resource_raw_evidence}" \
    --resource-raw-evidence-sha256 \
      "${PRORM_R3_RESOURCE_RAW_EVIDENCE_SHA256}" \
    --partition "${PRORM_R3_PARTITION}" \
    --gpu-name "${PRORM_R3_GPU_NAME}" \
    --gpu-total-memory-bytes "${PRORM_R3_GPU_TOTAL_MEMORY_BYTES}" \
    --max-allocation-wall-seconds \
      "${PRORM_R3_MAX_ALLOCATION_WALL_SECONDS}" \
    --max-array-concurrency "${PRORM_R3_MAX_ARRAY_CONCURRENCY}" \
    --max-scheduler-segments "${PRORM_R3_MAX_SCHEDULER_SEGMENTS}" \
    --max-gpus-per-task "${PRORM_R3_MAX_GPUS_PER_TASK}" \
    --max-cpus-per-task "${PRORM_R3_MAX_CPUS_PER_TASK}" \
    --max-memory-bytes "${PRORM_R3_MAX_MEMORY_BYTES}" \
    --walltime-margin-fraction "${PRORM_R3_WALLTIME_MARGIN_FRACTION}" \
    --fixed-walltime-margin-seconds \
      "${PRORM_R3_FIXED_WALLTIME_MARGIN_SECONDS}" \
    --memory-margin-fraction "${PRORM_R3_MEMORY_MARGIN_FRACTION}" \
    --signal-margin-seconds "${PRORM_R3_SIGNAL_MARGIN_SECONDS}" \
    --durable-checkpoint-cadence-updates \
      "${PRORM_R3_CHECKPOINT_CADENCE_UPDATES}" \
    --requested-walltime-seconds-per-segment \
      "${PRORM_R3_PRIMARY_WALLTIME_SECONDS}" \
    --array-concurrency "${PRORM_R3_PRIMARY_ARRAY_CONCURRENCY}" \
    --cpus-per-task "${PRORM_R3_PRIMARY_CPUS_PER_TASK}" \
    --memory-bytes "${PRORM_R3_PRIMARY_MEMORY_BYTES}" \
    --io-probe-directory "${io_probe_directory}" \
    --operational-bundle "${operational_bundle}" \
    --allocation-intent "${profile_intent}" \
    --allocation-intent-file-sha256 \
      "${PRORM_R3_PROFILE_INTENT_FILE_SHA256}" \
    --runtime-receipt "${runtime_receipt}"
