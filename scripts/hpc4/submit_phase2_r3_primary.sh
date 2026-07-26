#!/usr/bin/env bash
# Submit only the fixed three-seed R3 primary segment-1 array.

set -euo pipefail
umask 077

die() {
  printf 'R3 primary submit fatal: %s\n' "$*" >&2
  exit 1
}

required_environment=(
  PRORM_R3_IMAGE
  PRORM_R3_IMAGE_SHA256
  PRORM_R3_REPO_ROOT
  PRORM_R3_PROJECT_ROOT
  PRORM_R3_SCRATCH_ROOT
  PRORM_R3_HF_CACHE
  PRORM_R3_PRIMARY_ATTEMPT_ROOT
  PRORM_R3_SCIENCE_CONFIG
  PRORM_R3_PARENT_REGISTRY_FILE_SHA256
  PRORM_R3_GATE0_FILE_SHA256
  PRORM_R3_GATE1_FILE_SHA256
  PRORM_R3_SOURCE_TEST_RECEIPT_FILE_SHA256
  PRORM_R3_OPERATIONAL_BUNDLE
  PRORM_R3_OPERATIONAL_BUNDLE_FILE_SHA256
  PRORM_R3_PROFILE_INTENT
  PRORM_R3_PROFILE_INTENT_FILE_SHA256
  PRORM_R3_PROFILE_RUNTIME_RECEIPT
  PRORM_R3_PROFILE_RUNTIME_RECEIPT_FILE_SHA256
  PRORM_R3_PROFILE_TERMINAL_EVIDENCE_DIRECTORY
  PRORM_R3_PROFILE_TERMINAL_MANIFEST_FILE_SHA256
  PRORM_R3_PROFILE_TERMINAL_RAW_SACCT_SHA256
)
for name in "${required_environment[@]}"; do
  [[ -n "${!name:-}" ]] || die "missing environment variable ${name}"
done

command -v apptainer >/dev/null 2>&1 || die "apptainer is unavailable"
command -v cmp >/dev/null 2>&1 || die "cmp is unavailable"
command -v cp >/dev/null 2>&1 || die "cp is unavailable"
command -v git >/dev/null 2>&1 || die "git is unavailable"
command -v realpath >/dev/null 2>&1 || die "realpath is unavailable"
command -v sbatch >/dev/null 2>&1 || die "sbatch is unavailable"
command -v sha256sum >/dev/null 2>&1 || die "sha256sum is unavailable"

repo_root="$(realpath -e -- "${PRORM_R3_REPO_ROOT}")"
project_root="$(realpath -e -- "${PRORM_R3_PROJECT_ROOT}")"
scratch_root="$(realpath -e -- "${PRORM_R3_SCRATCH_ROOT}")"
image="$(realpath -e -- "${PRORM_R3_IMAGE}")"
hf_cache="$(realpath -e -- "${PRORM_R3_HF_CACHE}")"
science_config="$(realpath -e -- "${PRORM_R3_SCIENCE_CONFIG}")"
repo_source_config="$(
  realpath -e -- "${repo_root}/configs/common_beta_recovery_pilot.yaml"
)"
repo_parent_registry="$(
  realpath -e -- "${repo_root}/configs/phase2_recovery_parent_failures.json"
)"
operational_bundle="$(realpath -e -- "${PRORM_R3_OPERATIONAL_BUNDLE}")"
profile_intent="$(realpath -e -- "${PRORM_R3_PROFILE_INTENT}")"
profile_runtime_receipt="$(realpath -e -- "${PRORM_R3_PROFILE_RUNTIME_RECEIPT}")"
profile_terminal_directory="$(
  realpath -e -- "${PRORM_R3_PROFILE_TERMINAL_EVIDENCE_DIRECTORY}"
)"

[[ "${repo_root}" == "/home/yyangjo/Smart-Reward-Model" ]] \
  || die "unexpected production Git repository root"
[[ "${project_root}" == "/project/sigroup/smart-reward-model" ]] \
  || die "unexpected persistent materialization root"
[[ "${scratch_root}" == "/scratch/yyangjo" ]] \
  || die "unexpected production scratch root"
[[ "${hf_cache}" == /project/sigroup/* ]] \
  || die "HF cache must be under /project/sigroup"
[[ "${science_config}" == \
  "${repo_root}/configs/phase2_recovery_r3_science.yaml" ]] \
  || die "science config must be the frozen R3 production config"
[[ "${repo_source_config}" == \
  "${repo_root}/configs/common_beta_recovery_pilot.yaml" ]] \
  || die "source config must be the fixed clean-repository input"
[[ "${repo_parent_registry}" == \
  "${repo_root}/configs/phase2_recovery_parent_failures.json" ]] \
  || die "parent registry must be the fixed clean-repository input"
[[ "${operational_bundle}" == /project/sigroup/* ]] \
  || die "operational bundle must be durable project evidence"
[[ "${profile_intent}" == /project/sigroup/* ]] \
  || die "profile intent must be durable project evidence"
[[ "${profile_runtime_receipt}" == /project/sigroup/* ]] \
  || die "profile runtime receipt must be durable project evidence"
[[ "${profile_terminal_directory}" == /project/sigroup/* ]] \
  || die "profile terminal must be durable project evidence"

digest_environment=(
  PRORM_R3_IMAGE_SHA256
  PRORM_R3_PARENT_REGISTRY_FILE_SHA256
  PRORM_R3_GATE0_FILE_SHA256
  PRORM_R3_GATE1_FILE_SHA256
  PRORM_R3_SOURCE_TEST_RECEIPT_FILE_SHA256
  PRORM_R3_OPERATIONAL_BUNDLE_FILE_SHA256
  PRORM_R3_PROFILE_INTENT_FILE_SHA256
  PRORM_R3_PROFILE_RUNTIME_RECEIPT_FILE_SHA256
  PRORM_R3_PROFILE_TERMINAL_MANIFEST_FILE_SHA256
  PRORM_R3_PROFILE_TERMINAL_RAW_SACCT_SHA256
)
for name in "${digest_environment[@]}"; do
  [[ "${!name}" =~ ^[0-9a-f]{64}$ ]] || die "${name} is not a lowercase SHA-256"
done

commit="$(git -C "${repo_root}" rev-parse HEAD)"
[[ "${commit}" =~ ^[0-9a-f]{40}$ ]] || die "invalid Git commit"
[[ -z "$(git -C "${repo_root}" status --porcelain --untracked-files=all)" ]] \
  || die "production checkout is not clean"

# Publish or reuse only the byte-identical clean-commit inputs at their fixed
# persistent paths.  Caller-provided runtime path overrides are not accepted.
input_parent="${project_root}/runs/phase2-recovery-r3/inputs"
mkdir -p -- "${input_parent}"
input_parent="$(realpath -e -- "${input_parent}")"
[[ "${input_parent}" == "${project_root}/runs/phase2-recovery-r3/inputs" ]] \
  || die "R3 input namespace is not canonical"
input_root="${input_parent}/${commit}"
if [[ -e "${input_root}" || -L "${input_root}" ]]; then
  [[ -d "${input_root}" && ! -L "${input_root}" ]] \
    || die "commit input namespace is not a real directory"
else
  mkdir -- "${input_root}"
fi
input_root="$(realpath -e -- "${input_root}")"

copy_no_overwrite_exact() {
  local source="$1"
  local destination="$2"
  if [[ -e "${destination}" || -L "${destination}" ]]; then
    [[ -f "${destination}" && ! -L "${destination}" ]] \
      || die "retained input copy is not a regular file: ${destination}"
  else
    cp --no-clobber -- "${source}" "${destination}"
  fi
  [[ -f "${destination}" && ! -L "${destination}" ]] \
    || die "input copy publication failed: ${destination}"
  cmp --silent -- "${source}" "${destination}" \
    || die "retained input copy differs from clean repository bytes"
}

source_config="${input_root}/common_beta_recovery_pilot.yaml"
parent_registry="${input_root}/phase2_recovery_parent_failures.json"
copy_no_overwrite_exact "${repo_source_config}" "${source_config}"
copy_no_overwrite_exact "${repo_parent_registry}" "${parent_registry}"
source_config="$(realpath -e -- "${source_config}")"
parent_registry="$(realpath -e -- "${parent_registry}")"

printf '%s  %s\n' "${PRORM_R3_IMAGE_SHA256}" "${image}" \
  | sha256sum --check --status || die "container SHA-256 mismatch"
printf '%s  %s\n' \
  "${PRORM_R3_PARENT_REGISTRY_FILE_SHA256}" \
  "${parent_registry}" \
  | sha256sum --check --status || die "parent registry SHA-256 mismatch"
printf '%s  %s\n' \
  "${PRORM_R3_OPERATIONAL_BUNDLE_FILE_SHA256}" \
  "${operational_bundle}" \
  | sha256sum --check --status || die "operational bundle SHA-256 mismatch"
printf '%s  %s\n' \
  "${PRORM_R3_PROFILE_INTENT_FILE_SHA256}" \
  "${profile_intent}" \
  | sha256sum --check --status || die "profile intent SHA-256 mismatch"
printf '%s  %s\n' \
  "${PRORM_R3_PROFILE_RUNTIME_RECEIPT_FILE_SHA256}" \
  "${profile_runtime_receipt}" \
  | sha256sum --check --status || die "profile runtime receipt SHA-256 mismatch"

attempt_root_text="${PRORM_R3_PRIMARY_ATTEMPT_ROOT}"
[[ "${attempt_root_text}" == /project/sigroup/* ]] \
  || die "primary attempt root must be under /project/sigroup"
[[ ! -e "${attempt_root_text}" && ! -L "${attempt_root_text}" ]] \
  || die "primary attempt root already exists"
attempt_parent="$(realpath -e -- "$(dirname -- "${attempt_root_text}")")"
attempt_root="${attempt_parent}/$(basename -- "${attempt_root_text}")"
mkdir -- "${attempt_root}"
attempt_root="$(realpath -e -- "${attempt_root}")"
[[ "${attempt_root}" == "${attempt_root_text}" ]] \
  || die "primary attempt root must already be canonical"
mkdir -- "${attempt_root}/logs"
mkdir -- "${attempt_root}/runtime-closures"

prepare_cli="${repo_root}/scripts/hpc4/prepare_phase2_r3_primary_submission.py"
sbatch_script="${repo_root}/scripts/hpc4/phase2_r3_primary.sbatch"
[[ -f "${prepare_cli}" ]] || die "primary preparation CLI is missing"
[[ -f "${sbatch_script}" ]] || die "primary SBATCH body is missing"
submission_plan="${attempt_root}/primary-segment-1-submission-plan.json"

apptainer exec \
  --cleanenv \
  --bind "${repo_root}:${repo_root}:ro" \
  --bind "/project/sigroup:/project/sigroup:rw" \
  --env "PYTHONPATH=${repo_root}/src" \
  --env "PRORM_R3_REPO_ROOT=${repo_root}" \
  --env "PRORM_R3_PROJECT_ROOT=${project_root}" \
  "${image}" \
  python3 "${prepare_cli}" create \
    --operational-bundle "${operational_bundle}" \
    --operational-bundle-file-sha256 \
      "${PRORM_R3_OPERATIONAL_BUNDLE_FILE_SHA256}" \
    --profile-allocation-intent "${profile_intent}" \
    --profile-allocation-intent-file-sha256 \
      "${PRORM_R3_PROFILE_INTENT_FILE_SHA256}" \
    --profile-runtime-receipt "${profile_runtime_receipt}" \
    --profile-runtime-receipt-file-sha256 \
      "${PRORM_R3_PROFILE_RUNTIME_RECEIPT_FILE_SHA256}" \
    --profile-terminal-evidence-directory "${profile_terminal_directory}" \
    --profile-terminal-manifest-file-sha256 \
      "${PRORM_R3_PROFILE_TERMINAL_MANIFEST_FILE_SHA256}" \
    --profile-terminal-raw-sacct-sha256 \
      "${PRORM_R3_PROFILE_TERMINAL_RAW_SACCT_SHA256}" \
    --output "${submission_plan}"

submission_plan_file_sha256="$(
  sha256sum -- "${submission_plan}" | cut -d' ' -f1
)"
[[ "${submission_plan_file_sha256}" =~ ^[0-9a-f]{64}$ ]] \
  || die "invalid primary submission-plan file SHA-256"
plan_lines="$(
  apptainer exec \
    --cleanenv \
    --bind "${repo_root}:${repo_root}:ro" \
    --bind "/project/sigroup:/project/sigroup:ro" \
    --env "PYTHONPATH=${repo_root}/src" \
    --env "PRORM_R3_REPO_ROOT=${repo_root}" \
    --env "PRORM_R3_PROJECT_ROOT=${project_root}" \
    "${image}" \
    python3 "${prepare_cli}" inspect \
      --plan "${submission_plan}" \
      --plan-file-sha256 "${submission_plan_file_sha256}" \
      --format sbatch-lines
)" || die "failed to inspect the caller-pinned primary submission plan"
mapfile -t plan_fields <<< "${plan_lines}"
[[ "${#plan_fields[@]}" -eq 16 ]] || die "primary submission plan field count is invalid"

field_names=(
  submission_plan_sha256
  resource_plan_sha256
  slurm_account
  partition
  gpu_name
  gpus_per_task
  cpus_per_task
  memory_bytes
  memory_mib
  requested_walltime_seconds
  slurm_walltime
  array_concurrency
  max_scheduler_segments
  advance_signal_lead_seconds
  audit_cadence_updates
  durable_checkpoint_cadence_updates
)
plan_values=()
for index in "${!field_names[@]}"; do
  prefix="${field_names[index]}="
  line="${plan_fields[index]}"
  [[ "${line}" == "${prefix}"* ]] || die "primary plan field order is invalid"
  value="${line#"${prefix}"}"
  [[ -n "${value}" && "${value}" != *$'\n'* && "${value}" != *$'\r'* ]] \
    || die "primary plan field ${field_names[index]} is invalid"
  plan_values+=("${value}")
done

submission_plan_sha256="${plan_values[0]}"
resource_plan_sha256="${plan_values[1]}"
account="${plan_values[2]}"
partition="${plan_values[3]}"
gpu_name="${plan_values[4]}"
gpus_per_task="${plan_values[5]}"
cpus_per_task="${plan_values[6]}"
memory_bytes="${plan_values[7]}"
memory_mib="${plan_values[8]}"
walltime_seconds="${plan_values[9]}"
slurm_walltime="${plan_values[10]}"
array_concurrency="${plan_values[11]}"
max_scheduler_segments="${plan_values[12]}"
signal_lead_seconds="${plan_values[13]}"
audit_cadence_updates="${plan_values[14]}"
checkpoint_cadence_updates="${plan_values[15]}"

[[ "${submission_plan_sha256}" =~ ^[0-9a-f]{64}$ ]] \
  || die "submission semantic SHA-256 is invalid"
[[ "${resource_plan_sha256}" =~ ^[0-9a-f]{64}$ ]] \
  || die "resource-plan SHA-256 is invalid"
[[ "${account}" == "sigroup" ]] || die "unexpected primary Slurm account"
[[ "${partition}" == gpu-* ]] || die "primary plan does not use a GPU partition"
[[ -n "${gpu_name}" ]] || die "primary GPU name is empty"
[[ "${gpus_per_task}" == "1" ]] || die "primary plan must request one GPU per task"
for numeric in \
  "${cpus_per_task}" \
  "${memory_bytes}" \
  "${memory_mib}" \
  "${walltime_seconds}" \
  "${array_concurrency}" \
  "${max_scheduler_segments}" \
  "${signal_lead_seconds}" \
  "${audit_cadence_updates}" \
  "${checkpoint_cadence_updates}"; do
  [[ "${numeric}" =~ ^[1-9][0-9]*$ ]] || die "primary plan has an invalid integer"
done
[[ "${array_concurrency}" -le 3 ]] || die "array concurrency exceeds three fixed seeds"
[[ "${signal_lead_seconds}" -lt "${walltime_seconds}" ]] \
  || die "signal lead must be smaller than segment walltime"
[[ $((memory_mib * 1024 * 1024)) -eq "${memory_bytes}" ]] \
  || die "Slurm memory MiB does not represent the exact plan bytes"
[[ $((checkpoint_cadence_updates % audit_cadence_updates)) -eq 0 ]] \
  || die "checkpoint cadence is not audit-aligned"

task_parent="${scratch_root}/phase2-r3"
mkdir -p -- "${task_parent}"
task_parent="$(realpath -e -- "${task_parent}")"
[[ "${task_parent}" == "${scratch_root}/phase2-r3" ]] \
  || die "unexpected primary task parent"
task_root_base="${task_parent}/primary-${commit}-${submission_plan_sha256}"
[[ ! -e "${task_root_base}" && ! -L "${task_root_base}" ]] \
  || die "primary task root already exists"
mkdir -- "${task_root_base}"
task_root_base="$(realpath -e -- "${task_root_base}")"

export PRORM_R3_GIT_COMMIT="${commit}"
export PRORM_R3_REPO_ROOT="${repo_root}"
export PRORM_R3_PROJECT_ROOT="${project_root}"
export PRORM_R3_SCRATCH_ROOT="${scratch_root}"
export PRORM_R3_HF_CACHE="${hf_cache}"
export PRORM_R3_SCIENCE_CONFIG="${science_config}"
export PRORM_R3_SOURCE_CONFIG="${source_config}"
export PRORM_R3_PARENT_REGISTRY="${parent_registry}"
export PRORM_R3_OPERATIONAL_BUNDLE="${operational_bundle}"
export PRORM_R3_PROFILE_INTENT="${profile_intent}"
export PRORM_R3_PROFILE_RUNTIME_RECEIPT="${profile_runtime_receipt}"
export PRORM_R3_PROFILE_TERMINAL_EVIDENCE_DIRECTORY="${profile_terminal_directory}"
export PRORM_R3_PRIMARY_ATTEMPT_ROOT="${attempt_root}"
export PRORM_R3_PRIMARY_TASK_ROOT_BASE="${task_root_base}"
export PRORM_R3_PRIMARY_SUBMISSION_PLAN="${submission_plan}"
export PRORM_R3_PRIMARY_SUBMISSION_PLAN_FILE_SHA256="${submission_plan_file_sha256}"
export PRORM_R3_PRIMARY_SUBMISSION_PLAN_SHA256="${submission_plan_sha256}"
export PRORM_R3_PRIMARY_RESOURCE_PLAN_SHA256="${resource_plan_sha256}"
export PRORM_R3_PRIMARY_ACCOUNT="${account}"
export PRORM_R3_PRIMARY_PARTITION="${partition}"
export PRORM_R3_PRIMARY_GPU_NAME="${gpu_name}"
export PRORM_R3_PRIMARY_GPUS_PER_TASK="${gpus_per_task}"
export PRORM_R3_PRIMARY_CPUS_PER_TASK="${cpus_per_task}"
export PRORM_R3_PRIMARY_MEMORY_BYTES="${memory_bytes}"
export PRORM_R3_PRIMARY_MEMORY_MIB="${memory_mib}"
export PRORM_R3_PRIMARY_WALLTIME_SECONDS="${walltime_seconds}"
export PRORM_R3_PRIMARY_ARRAY_CONCURRENCY="${array_concurrency}"
export PRORM_R3_PRIMARY_MAX_SCHEDULER_SEGMENTS="${max_scheduler_segments}"
export PRORM_R3_PRIMARY_SIGNAL_LEAD_SECONDS="${signal_lead_seconds}"
export PRORM_R3_PRIMARY_AUDIT_CADENCE_UPDATES="${audit_cadence_updates}"
export PRORM_R3_PRIMARY_CHECKPOINT_CADENCE_UPDATES="${checkpoint_cadence_updates}"

export_names=("${required_environment[@]}")
export_names+=(
  PRORM_R3_GIT_COMMIT
  PRORM_R3_SOURCE_CONFIG
  PRORM_R3_PARENT_REGISTRY
  PRORM_R3_PRIMARY_TASK_ROOT_BASE
  PRORM_R3_PRIMARY_SUBMISSION_PLAN
  PRORM_R3_PRIMARY_SUBMISSION_PLAN_FILE_SHA256
  PRORM_R3_PRIMARY_SUBMISSION_PLAN_SHA256
  PRORM_R3_PRIMARY_RESOURCE_PLAN_SHA256
  PRORM_R3_PRIMARY_ACCOUNT
  PRORM_R3_PRIMARY_PARTITION
  PRORM_R3_PRIMARY_GPU_NAME
  PRORM_R3_PRIMARY_GPUS_PER_TASK
  PRORM_R3_PRIMARY_CPUS_PER_TASK
  PRORM_R3_PRIMARY_MEMORY_BYTES
  PRORM_R3_PRIMARY_MEMORY_MIB
  PRORM_R3_PRIMARY_WALLTIME_SECONDS
  PRORM_R3_PRIMARY_ARRAY_CONCURRENCY
  PRORM_R3_PRIMARY_MAX_SCHEDULER_SEGMENTS
  PRORM_R3_PRIMARY_SIGNAL_LEAD_SECONDS
  PRORM_R3_PRIMARY_AUDIT_CADENCE_UPDATES
  PRORM_R3_PRIMARY_CHECKPOINT_CADENCE_UPDATES
)
export_spec="NONE"
for name in "${export_names[@]}"; do
  value="${!name}"
  [[ "${value}" != *","* && "${value}" != *$'\n'* && "${value}" != *$'\r'* ]] \
    || die "${name} cannot be represented safely in sbatch --export"
  export_spec+=",${name}=${value}"
done

job_id="$(
  sbatch \
    --parsable \
    --account="${account}" \
    --partition="${partition}" \
    --nodes=1 \
    --ntasks=1 \
    --gpus-per-node=1 \
    --cpus-per-task="${cpus_per_task}" \
    --mem="${memory_mib}M" \
    --time="${slurm_walltime}" \
    --array="0-2%${array_concurrency}" \
    --signal="B:USR1@${signal_lead_seconds}" \
    --job-name=prorm-r3-primary-s1 \
    --output="${attempt_root}/logs/primary-s1-%A_%a.out" \
    --error="${attempt_root}/logs/primary-s1-%A_%a.err" \
    --export="${export_spec}" \
    "${sbatch_script}"
)"
[[ "${job_id}" =~ ^[1-9][0-9]*$ ]] || die "sbatch returned an invalid job id"
printf 'R3_PRIMARY_SEGMENT_1_ARRAY_JOB_ID=%s\n' "${job_id}"
printf 'R3_PRIMARY_ATTEMPT_ROOT=%s\n' "${attempt_root}"
printf 'R3_PRIMARY_SUBMISSION_PLAN_FILE_SHA256=%s\n' \
  "${submission_plan_file_sha256}"
printf 'R3_PRIMARY_SUBMISSION_PLAN_SHA256=%s\n' "${submission_plan_sha256}"
printf 'R3_PRIMARY_RESOURCE_PLAN_SHA256=%s\n' "${resource_plan_sha256}"
