#!/usr/bin/env bash
# Compute-side driver for one sealed R3 continuation wave.

set -euo pipefail
umask 077

readonly HOST_PYTHON="/opt/shared/spack/local/linux-rocky9-x86_64_v4/gcc-11.4.1/miniconda3-24.3.0-quc3pyudmzikgo2r4qsyqpwnrvzpin63/bin/python3.12"
readonly HOST_PYTHON_SHA256="9c91f9aa231c61c6bf2eabb9b93ebc5a8269a4126a36125e9548d8853e32da9c"

die() {
  printf 'R3 continuation submit fatal: %s\n' "$*" >&2
  exit 1
}

required_environment=(
  PRORM_R3_IMAGE
  PRORM_R3_IMAGE_SHA256
  PRORM_R3_REPO_ROOT
  PRORM_R3_PROJECT_ROOT
  PRORM_R3_SCRATCH_ROOT
  PRORM_R3_HF_CACHE
  PRORM_R3_CONTINUATION_ATTEMPT_ROOT
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
  PRORM_R3_PRIMARY_SUBMISSION_PLAN
  PRORM_R3_PRIMARY_SUBMISSION_PLAN_FILE_SHA256
)
for task_id in 0 1 2; do
  required_environment+=(
    "PRORM_R3_TASK_${task_id}_RUNTIME_CLOSURE"
    "PRORM_R3_TASK_${task_id}_RUNTIME_CLOSURE_FILE_SHA256"
    "PRORM_R3_TASK_${task_id}_TERMINAL_EVIDENCE_DIRECTORY"
    "PRORM_R3_TASK_${task_id}_TERMINAL_MANIFEST_FILE_SHA256"
    "PRORM_R3_TASK_${task_id}_TERMINAL_RAW_SACCT_SHA256"
  )
done
for name in "${required_environment[@]}"; do
  [[ -n "${!name:-}" ]] || die "missing environment variable ${name}"
done

previous_plan_present=false
if [[ -n "${PRORM_R3_PREVIOUS_CONTINUATION_PLAN:-}" || \
  -n "${PRORM_R3_PREVIOUS_CONTINUATION_PLAN_FILE_SHA256:-}" ]]; then
  [[ -n "${PRORM_R3_PREVIOUS_CONTINUATION_PLAN:-}" && \
    -n "${PRORM_R3_PREVIOUS_CONTINUATION_PLAN_FILE_SHA256:-}" ]] \
    || die "previous continuation plan path and SHA must appear together"
  previous_plan_present=true
fi

for command_name in \
  apptainer cmp cp env git realpath sacct sbatch scontrol sed sha256sum squeue; do
  command -v "${command_name}" >/dev/null 2>&1 \
    || die "${command_name} is unavailable"
done

host_python="${HOST_PYTHON}"
host_python_target="$(realpath -e -- "${host_python}")"
[[ -f "${host_python}" && -x "${host_python}" && ! -L "${host_python}" ]] \
  || die "fixed host Python target is not a regular non-symlink executable"
[[ "${host_python_target}" == "${host_python}" ]] \
  || die "fixed host Python target is not canonical"
printf '%s  %s\n' "${HOST_PYTHON_SHA256}" "${host_python}" \
  | sha256sum --check --status || die "fixed host Python SHA-256 mismatch"
[[ "$("${host_python}" --version 2>&1)" == "Python 3.12.2" ]] \
  || die "fixed host Python version mismatch"

while IFS='=' read -r name _; do
  case "${name}" in
    SBATCH_*) die "unset exported ${name}; ambient sbatch overrides are forbidden" ;;
  esac
done < <(env)

repo_root="$(realpath -e -- "${PRORM_R3_REPO_ROOT}")"
project_root="$(realpath -e -- "${PRORM_R3_PROJECT_ROOT}")"
scratch_root="$(realpath -e -- "${PRORM_R3_SCRATCH_ROOT}")"
image="$(realpath -e -- "${PRORM_R3_IMAGE}")"
hf_cache="$(realpath -e -- "${PRORM_R3_HF_CACHE}")"
repo_science_config="$(realpath -e -- "${PRORM_R3_SCIENCE_CONFIG}")"
repo_source_config="$(
  realpath -e -- "${repo_root}/configs/common_beta_pilot_base.yaml"
)"
repo_parent_registry="$(
  realpath -e -- "${repo_root}/configs/phase2_recovery_parent_failures.json"
)"
base_plan="$(realpath -e -- "${PRORM_R3_PRIMARY_SUBMISSION_PLAN}")"
operational_bundle="$(realpath -e -- "${PRORM_R3_OPERATIONAL_BUNDLE}")"
profile_intent="$(realpath -e -- "${PRORM_R3_PROFILE_INTENT}")"
profile_runtime_receipt="$(realpath -e -- "${PRORM_R3_PROFILE_RUNTIME_RECEIPT}")"
profile_terminal_directory="$(
  realpath -e -- "${PRORM_R3_PROFILE_TERMINAL_EVIDENCE_DIRECTORY}"
)"
gate0="${project_root}/runs/phase2-recovery-r3/gate0/r2-1648125-failure-parent.json"
gate1="${project_root}/runs/phase2-recovery-r3/gate1/r3-implementation-closure.json"
source_test_receipt="${project_root}/runs/phase2-recovery-r3/gate1/r3-source-test-receipt.json"
gate0="$(realpath -e -- "${gate0}")"
gate1="$(realpath -e -- "${gate1}")"
source_test_receipt="$(realpath -e -- "${source_test_receipt}")"
prepare_primary="${repo_root}/scripts/hpc4/prepare_phase2_r3_primary_submission.py"
[[ -f "${prepare_primary}" ]] || die "primary submission-plan validator is missing"
[[ "${repo_root}" == "/home/yyangjo/Smart-Reward-Model" ]] \
  || die "unexpected production Git repository root"
[[ "${project_root}" == "/project/sigroup/smart-reward-model" ]] \
  || die "unexpected persistent materialization root"
[[ "${scratch_root}" == "/scratch/yyangjo" ]] \
  || die "unexpected production scratch root"
[[ "${repo_science_config}" == \
  "${repo_root}/configs/phase2_recovery_r3_science.yaml" ]] \
  || die "science config must be the frozen R3 production config"
[[ "${repo_source_config}" == \
  "${repo_root}/configs/common_beta_pilot_base.yaml" ]] \
  || die "source config must be the fixed clean-repository input"
[[ "${repo_parent_registry}" == \
  "${repo_root}/configs/phase2_recovery_parent_failures.json" ]] \
  || die "parent registry must be the fixed clean-repository input"
[[ "${base_plan}" == /project/sigroup/* ]] \
  || die "base primary plan must be durable project evidence"
for retained_path in \
  "${operational_bundle}" "${profile_intent}" "${profile_runtime_receipt}" \
  "${profile_terminal_directory}"; do
  [[ "${retained_path}" == /project/sigroup/* ]] \
    || die "Gate-P evidence must be retained under /project/sigroup"
done

digest_names=(
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
  PRORM_R3_PRIMARY_SUBMISSION_PLAN_FILE_SHA256
)
for task_id in 0 1 2; do
  digest_names+=(
    "PRORM_R3_TASK_${task_id}_RUNTIME_CLOSURE_FILE_SHA256"
    "PRORM_R3_TASK_${task_id}_TERMINAL_MANIFEST_FILE_SHA256"
    "PRORM_R3_TASK_${task_id}_TERMINAL_RAW_SACCT_SHA256"
  )
done
if [[ "${previous_plan_present}" == "true" ]]; then
  digest_names+=(PRORM_R3_PREVIOUS_CONTINUATION_PLAN_FILE_SHA256)
fi
for name in "${digest_names[@]}"; do
  [[ "${!name}" =~ ^[0-9a-f]{64}$ ]] || die "${name} is not a lowercase SHA-256"
done

commit="$(git -C "${repo_root}" rev-parse HEAD)"
[[ "${commit}" =~ ^[0-9a-f]{40}$ ]] || die "invalid Git commit"
[[ -z "$(git -C "${repo_root}" status --porcelain --untracked-files=all)" ]] \
  || die "production checkout is not clean"

# Reuse or publish only byte-identical clean-commit inputs at their exact
# persistent paths.  External runtime path overrides are intentionally ignored.
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

source_config="${input_root}/common_beta_pilot_base.yaml"
parent_registry="${input_root}/phase2_recovery_parent_failures.json"
science_config="${input_root}/phase2_recovery_r3_science.yaml"
copy_no_overwrite_exact "${repo_source_config}" "${source_config}"
copy_no_overwrite_exact "${repo_parent_registry}" "${parent_registry}"
copy_no_overwrite_exact "${repo_science_config}" "${science_config}"
source_config="$(realpath -e -- "${source_config}")"
parent_registry="$(realpath -e -- "${parent_registry}")"
science_config="$(realpath -e -- "${science_config}")"
source_config_file_sha256="$(
  sha256sum -- "${source_config}" | cut -d' ' -f1
)"
science_config_file_sha256="$(
  sha256sum -- "${science_config}" | cut -d' ' -f1
)"

printf '%s  %s\n' "${PRORM_R3_IMAGE_SHA256}" "${image}" \
  | sha256sum --check --status || die "container SHA-256 mismatch"
printf '%s  %s\n' \
  "${PRORM_R3_PARENT_REGISTRY_FILE_SHA256}" \
  "${parent_registry}" \
  | sha256sum --check --status || die "parent registry SHA-256 mismatch"
printf '%s  %s\n' \
  "${PRORM_R3_PRIMARY_SUBMISSION_PLAN_FILE_SHA256}" \
  "${base_plan}" \
  | sha256sum --check --status || die "base primary plan SHA-256 mismatch"

base_binding_lines="$(
  apptainer exec \
    --cleanenv \
    --bind "${repo_root}:${repo_root}:ro" \
    --bind "/project/sigroup:/project/sigroup:ro" \
    --env "PYTHONPATH=${repo_root}/src" \
    --env "PRORM_R3_REPO_ROOT=${repo_root}" \
    --env "PRORM_R3_PROJECT_ROOT=${project_root}" \
    "${image}" \
    python3 "${prepare_primary}" inspect \
      --plan "${base_plan}" \
      --plan-file-sha256 "${PRORM_R3_PRIMARY_SUBMISSION_PLAN_FILE_SHA256}" \
      --format binding-lines
)" || die "failed to inspect base primary provenance bindings"
mapfile -t base_binding_fields <<< "${base_binding_lines}"
base_binding_names=(
  git_commit
  container_image_path
  container_image_file_sha256
  science_config_path
  science_config_file_sha256
  source_config_path
  source_config_file_sha256
  parent_registry_path
  parent_registry_file_sha256
  gate0_path
  gate0_file_sha256
  gate1_path
  gate1_file_sha256
  source_test_receipt_path
  source_test_receipt_file_sha256
  operational_bundle_path
  operational_bundle_file_sha256
  profile_allocation_intent_path
  profile_allocation_intent_file_sha256
  profile_runtime_receipt_path
  profile_runtime_receipt_file_sha256
  profile_terminal_evidence_directory
  profile_terminal_manifest_file_sha256
  profile_terminal_raw_sacct_sha256
)
base_binding_expected=(
  "${commit}"
  "${image}"
  "${PRORM_R3_IMAGE_SHA256}"
  "${science_config}"
  "${science_config_file_sha256}"
  "${source_config}"
  "${source_config_file_sha256}"
  "${parent_registry}"
  "${PRORM_R3_PARENT_REGISTRY_FILE_SHA256}"
  "${gate0}"
  "${PRORM_R3_GATE0_FILE_SHA256}"
  "${gate1}"
  "${PRORM_R3_GATE1_FILE_SHA256}"
  "${source_test_receipt}"
  "${PRORM_R3_SOURCE_TEST_RECEIPT_FILE_SHA256}"
  "${operational_bundle}"
  "${PRORM_R3_OPERATIONAL_BUNDLE_FILE_SHA256}"
  "${profile_intent}"
  "${PRORM_R3_PROFILE_INTENT_FILE_SHA256}"
  "${profile_runtime_receipt}"
  "${PRORM_R3_PROFILE_RUNTIME_RECEIPT_FILE_SHA256}"
  "${profile_terminal_directory}"
  "${PRORM_R3_PROFILE_TERMINAL_MANIFEST_FILE_SHA256}"
  "${PRORM_R3_PROFILE_TERMINAL_RAW_SACCT_SHA256}"
)
[[ "${#base_binding_fields[@]}" -eq "${#base_binding_names[@]}" ]] \
  || die "base primary provenance binding count is invalid"
for index in "${!base_binding_names[@]}"; do
  expected_line="${base_binding_names[index]}=${base_binding_expected[index]}"
  [[ "${base_binding_fields[index]}" == "${expected_line}" ]] \
    || die "base primary ${base_binding_names[index]} binding drifted"
done

attempt_text="${PRORM_R3_CONTINUATION_ATTEMPT_ROOT}"
[[ "${attempt_text}" == \
  "${project_root}/runs/phase2-recovery-r3/"* ]] \
  || die "continuation attempt root must use the canonical R3 namespace"
attempt_parent="$(realpath -e -- "$(dirname -- "${attempt_text}")")"
attempt_root="${attempt_parent}/$(basename -- "${attempt_text}")"
attempt_reused=false
if [[ -e "${attempt_root}" || -L "${attempt_root}" ]]; then
  [[ -d "${attempt_root}" && ! -L "${attempt_root}" ]] \
    || die "existing continuation attempt root is not a real directory"
  attempt_reused=true
else
  mkdir -- "${attempt_root}"
fi
attempt_root="$(realpath -e -- "${attempt_root}")"
[[ "${attempt_root}" == "${attempt_text}" ]] \
  || die "continuation attempt root must already be canonical"
mkdir -p -- "${attempt_root}/logs"
mkdir -p -- "${attempt_root}/runtime-closures"

prepare_continuation="${repo_root}/scripts/hpc4/prepare_phase2_r3_continuation_submission.py"
sbatch_script="${repo_root}/scripts/hpc4/phase2_r3_primary.sbatch"
[[ -f "${prepare_primary}" && -f "${prepare_continuation}" ]] \
  || die "primary continuation preparation scripts are missing"
[[ -f "${sbatch_script}" ]] || die "primary SBATCH body is missing"
continuation_plan="${attempt_root}/continuation-wave-plan.json"

previous_arguments=()
if [[ "${previous_plan_present}" == "true" ]]; then
  previous_plan="$(realpath -e -- "${PRORM_R3_PREVIOUS_CONTINUATION_PLAN}")"
  previous_arguments=(
    --previous-continuation-plan "${previous_plan}"
    --previous-continuation-plan-file-sha256 \
      "${PRORM_R3_PREVIOUS_CONTINUATION_PLAN_FILE_SHA256}"
  )
fi
task_arguments=()
for task_id in 0 1 2; do
  closure_name="PRORM_R3_TASK_${task_id}_RUNTIME_CLOSURE"
  closure_sha_name="PRORM_R3_TASK_${task_id}_RUNTIME_CLOSURE_FILE_SHA256"
  terminal_name="PRORM_R3_TASK_${task_id}_TERMINAL_EVIDENCE_DIRECTORY"
  manifest_name="PRORM_R3_TASK_${task_id}_TERMINAL_MANIFEST_FILE_SHA256"
  raw_name="PRORM_R3_TASK_${task_id}_TERMINAL_RAW_SACCT_SHA256"
  task_arguments+=(
    "--task-${task_id}-runtime-closure" "${!closure_name}"
    "--task-${task_id}-runtime-closure-file-sha256" "${!closure_sha_name}"
    "--task-${task_id}-terminal-evidence-directory" "${!terminal_name}"
    "--task-${task_id}-terminal-manifest-file-sha256" "${!manifest_name}"
    "--task-${task_id}-terminal-raw-sacct-sha256" "${!raw_name}"
  )
done

if [[ ! -e "${continuation_plan}" && ! -L "${continuation_plan}" ]]; then
  apptainer exec \
    --cleanenv \
    --bind "${repo_root}:${repo_root}:ro" \
    --bind "/project/sigroup:/project/sigroup:rw" \
    --env "PYTHONPATH=${repo_root}/src" \
    --env "PRORM_R3_REPO_ROOT=${repo_root}" \
    --env "PRORM_R3_PROJECT_ROOT=${project_root}" \
    "${image}" \
    python3 "${prepare_continuation}" create \
      --primary-submission-plan "${base_plan}" \
      --primary-submission-plan-file-sha256 \
        "${PRORM_R3_PRIMARY_SUBMISSION_PLAN_FILE_SHA256}" \
      "${previous_arguments[@]}" \
      "${task_arguments[@]}" \
      --output "${continuation_plan}"
fi
[[ -f "${continuation_plan}" && ! -L "${continuation_plan}" ]] \
  || die "continuation plan is not a regular immutable file"

continuation_plan_file_sha256="$(
  sha256sum -- "${continuation_plan}" | cut -d' ' -f1
)"
[[ "${continuation_plan_file_sha256}" =~ ^[0-9a-f]{64}$ ]] \
  || die "invalid continuation plan file SHA-256"
plan_lines="$(
  apptainer exec \
    --cleanenv \
    --bind "${repo_root}:${repo_root}:ro" \
    --bind "/project/sigroup:/project/sigroup:ro" \
    --env "PYTHONPATH=${repo_root}/src" \
    --env "PRORM_R3_REPO_ROOT=${repo_root}" \
    --env "PRORM_R3_PROJECT_ROOT=${project_root}" \
    "${image}" \
    python3 "${prepare_continuation}" inspect \
      --plan "${continuation_plan}" \
      --plan-file-sha256 "${continuation_plan_file_sha256}" \
      --format sbatch-lines
)" || die "failed to inspect continuation plan"
mapfile -t fields <<< "${plan_lines}"
[[ "${#fields[@]}" -eq 21 ]] || die "continuation plan field count is invalid"
names=(
  continuation_plan_sha256
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
  continuation_array_required
  all_tasks_complete
  active_array_task_ids
  dependency_afterok
  dependency_afternotok
)
values=()
for index in "${!names[@]}"; do
  prefix="${names[index]}="
  [[ "${fields[index]}" == "${prefix}"* ]] || die "continuation plan field order is invalid"
  values+=("${fields[index]#"${prefix}"}")
done

continuation_plan_sha256="${values[0]}"
resource_plan_sha256="${values[1]}"
account="${values[2]}"
partition="${values[3]}"
gpu_name="${values[4]}"
gpus_per_task="${values[5]}"
cpus_per_task="${values[6]}"
memory_bytes="${values[7]}"
memory_mib="${values[8]}"
walltime_seconds="${values[9]}"
slurm_walltime="${values[10]}"
array_concurrency="${values[11]}"
max_segments="${values[12]}"
signal_lead="${values[13]}"
audit_cadence="${values[14]}"
checkpoint_cadence="${values[15]}"
continuation_required="${values[16]}"
all_complete="${values[17]}"
active_array_task_ids="${values[18]}"
dependency_job_ids="${values[19]}"
failure_dependency_job_ids="${values[20]}"

[[ "${continuation_plan_sha256}" =~ ^[0-9a-f]{64}$ ]] \
  || die "continuation semantic SHA-256 is invalid"
[[ "${resource_plan_sha256}" =~ ^[0-9a-f]{64}$ ]] \
  || die "resource plan SHA-256 is invalid"
[[ "${account}" == "sigroup" && "${partition}" == gpu-* ]] \
  || die "continuation Slurm account/partition is invalid"
[[ "${gpus_per_task}" == "1" ]] || die "continuation requires one GPU per task"
[[ "${continuation_required}" == "true" || "${continuation_required}" == "false" ]] \
  || die "continuation-required marker is invalid"
[[ "${all_complete}" == "true" || "${all_complete}" == "false" ]] \
  || die "all-complete marker is invalid"
if [[ "${attempt_reused}" == "true" && "${all_complete}" == "false" ]]; then
  ledger_intent="${project_root}/runs/phase2-recovery-r3/submission-ledgers/${continuation_plan_sha256}/intent.json"
  [[ -f "${ledger_intent}" && ! -L "${ledger_intent}" ]] \
    || die "existing continuation attempt lacks an immutable submission intent"
fi
if [[ "${all_complete}" == "true" ]]; then
  [[ "${continuation_required}" == "false" && -z "${dependency_job_ids}" && \
    -z "${active_array_task_ids}" ]] \
    || die "completed wave cannot retain continuation dependencies"
  printf '%s\n' 'R3_PRIMARY_STATUS=all_three_tasks_compute_complete_no_resubmission'
  printf 'R3_CONTINUATION_PLAN=%s\n' "${continuation_plan}"
  printf 'R3_CONTINUATION_PLAN_FILE_SHA256=%s\n' \
    "${continuation_plan_file_sha256}"
  exit 0
fi
[[ "${continuation_required}" == "true" ]] \
  || die "non-complete wave must require continuation"
[[ "${active_array_task_ids}" =~ ^[0-2](,[0-2])*$ ]] \
  || die "active continuation task IDs are invalid"
[[ "${dependency_job_ids}" =~ ^[1-9][0-9]*(:[1-9][0-9]*)*$ ]] \
  || die "afterok dependency job IDs are invalid"
[[ "${failure_dependency_job_ids}" == "${dependency_job_ids}" ]] \
  || die "afternotok guard differs from the sealed predecessor set"
for numeric in \
  "${cpus_per_task}" "${memory_bytes}" "${memory_mib}" "${walltime_seconds}" \
  "${array_concurrency}" "${max_segments}" "${signal_lead}" \
  "${audit_cadence}" "${checkpoint_cadence}"; do
  [[ "${numeric}" =~ ^[1-9][0-9]*$ ]] || die "continuation plan integer is invalid"
done
[[ $((memory_mib * 1024 * 1024)) -eq "${memory_bytes}" ]] \
  || die "continuation memory request differs from exact bytes"

primary_lines="$(
  apptainer exec \
    --cleanenv \
    --bind "${repo_root}:${repo_root}:ro" \
    --bind "/project/sigroup:/project/sigroup:ro" \
    --env "PYTHONPATH=${repo_root}/src" \
    --env "PRORM_R3_REPO_ROOT=${repo_root}" \
    --env "PRORM_R3_PROJECT_ROOT=${project_root}" \
    "${image}" \
    python3 "${prepare_primary}" inspect \
      --plan "${base_plan}" \
      --plan-file-sha256 "${PRORM_R3_PRIMARY_SUBMISSION_PLAN_FILE_SHA256}" \
      --format sbatch-lines
)" || die "failed to inspect base primary plan"
base_plan_sha256="$(
  printf '%s\n' "${primary_lines}" \
    | sed -n 's/^submission_plan_sha256=//p'
)"
[[ "${base_plan_sha256}" =~ ^[0-9a-f]{64}$ ]] \
  || die "base primary semantic SHA-256 is invalid"
task_root_base="${scratch_root}/phase2-r3/primary-${commit}-${base_plan_sha256}"
task_root_base="$(realpath -e -- "${task_root_base}")"

export PRORM_R3_GIT_COMMIT="${commit}"
export PRORM_R3_IMAGE="${image}"
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
export PRORM_R3_PRIMARY_SUBMISSION_PLAN="${base_plan}"
export PRORM_R3_PRIMARY_SUBMISSION_PLAN_SHA256="${base_plan_sha256}"
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
export PRORM_R3_PRIMARY_MAX_SCHEDULER_SEGMENTS="${max_segments}"
export PRORM_R3_PRIMARY_SIGNAL_LEAD_SECONDS="${signal_lead}"
export PRORM_R3_PRIMARY_AUDIT_CADENCE_UPDATES="${audit_cadence}"
export PRORM_R3_PRIMARY_CHECKPOINT_CADENCE_UPDATES="${checkpoint_cadence}"
export PRORM_R3_CONTINUATION_PLAN="${continuation_plan}"
export PRORM_R3_CONTINUATION_PLAN_FILE_SHA256="${continuation_plan_file_sha256}"
export PRORM_R3_CONTINUATION_PLAN_SHA256="${continuation_plan_sha256}"

export_names=()
for name in "${required_environment[@]}"; do
  case "${name}" in
    PRORM_R3_CONTINUATION_ATTEMPT_ROOT|PRORM_R3_TASK_*) continue ;;
    *) export_names+=("${name}") ;;
  esac
done
export_names+=(
  PRORM_R3_GIT_COMMIT
  PRORM_R3_SOURCE_CONFIG
  PRORM_R3_PARENT_REGISTRY
  PRORM_R3_PRIMARY_ATTEMPT_ROOT
  PRORM_R3_PRIMARY_TASK_ROOT_BASE
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
  PRORM_R3_CONTINUATION_PLAN
  PRORM_R3_CONTINUATION_PLAN_FILE_SHA256
  PRORM_R3_CONTINUATION_PLAN_SHA256
)
export_spec="PATH=/usr/bin:/bin"
for name in "${export_names[@]}"; do
  value="${!name}"
  [[ "${value}" != *","* && "${value}" != *$'\n'* && "${value}" != *$'\r'* ]] \
    || die "${name} cannot be represented safely in sbatch --export"
  export_spec+=",${name}=${value}"
done

submit_once="${repo_root}/scripts/hpc4/submit_phase2_r3_once.py"
[[ -f "${submit_once}" ]] || die "R3 submission-ledger helper is missing"
submission_output="$(
  "${host_python}" -I -S "${submit_once}" \
    --project-root "${project_root}" \
    --attempt-root "${attempt_root}" \
    --plan "${continuation_plan}" \
    --plan-file-sha256 "${continuation_plan_file_sha256}" \
    --plan-semantic-sha256 "${continuation_plan_sha256}" \
    --plan-kind continuation \
    --array-task-ids "${active_array_task_ids}" \
    --dependency-job-ids "${dependency_job_ids}" \
    --git-commit "${commit}" \
    --container-image-file-sha256 "${PRORM_R3_IMAGE_SHA256}" \
    --job-name prorm-r3-primary-cont \
    -- \
    sbatch \
    --parsable \
    --no-requeue \
    --dependency="afterok:${dependency_job_ids}" \
    --kill-on-invalid-dep=yes \
    --account="${account}" \
    --partition="${partition}" \
    --nodes=1 \
    --ntasks=1 \
    --gpus-per-node=1 \
    --cpus-per-task="${cpus_per_task}" \
    --mem="${memory_mib}M" \
    --time="${slurm_walltime}" \
    --array="${active_array_task_ids}%${array_concurrency}" \
    --signal="B:USR1@${signal_lead}" \
    --job-name=prorm-r3-primary-cont \
    --output="${attempt_root}/logs/continuation-%A_%a.out" \
    --error="${attempt_root}/logs/continuation-%A_%a.err" \
    --export="${export_spec}" \
    "${sbatch_script}"
)"
job_id="$(
  printf '%s\n' "${submission_output}" \
    | sed -n 's/^R3_ARRAY_JOB_ID=//p'
)"
[[ "${job_id}" =~ ^[1-9][0-9]*$ ]] || die "sbatch returned an invalid job ID"
printf '%s\n' "${submission_output}"
printf 'R3_PRIMARY_CONTINUATION_ARRAY_JOB_ID=%s\n' "${job_id}"
printf 'R3_CONTINUATION_PLAN=%s\n' "${continuation_plan}"
printf 'R3_CONTINUATION_PLAN_FILE_SHA256=%s\n' \
  "${continuation_plan_file_sha256}"
printf 'R3_PRIMARY_AFTEROK_DEPENDENCY=afterok:%s\n' "${dependency_job_ids}"
printf 'R3_PRIMARY_PREDECESSOR_FAILURE_DEPENDENCY=afternotok:%s\n' \
  "${failure_dependency_job_ids}"
printf 'R3_PRIMARY_FAILURE_GUARD_DEPENDENCY=afternotok:%s\n' "${job_id}"
