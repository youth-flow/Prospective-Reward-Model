#!/usr/bin/env bash
# Submit the one fixed R3 Gate-P profiling allocation.

set -euo pipefail
umask 077

die() {
  printf 'R3 Gate-P submit fatal: %s\n' "$*" >&2
  exit 1
}

required_environment=(
  PRORM_R3_IMAGE
  PRORM_R3_IMAGE_SHA256
  PRORM_R3_REPO_ROOT
  PRORM_R3_PROJECT_ROOT
  PRORM_R3_SCRATCH_ROOT
  PRORM_R3_HF_CACHE
  PRORM_R3_GATEP_ATTEMPT_ROOT
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
  PRORM_R3_CLUSTER
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
  PRORM_R3_PROFILE_WALLTIME_SECONDS
  PRORM_R3_PROFILE_CPUS_PER_TASK
  PRORM_R3_PROFILE_MEMORY_MIB
)
for name in "${required_environment[@]}"; do
  [[ -n "${!name:-}" ]] || die "missing environment variable ${name}"
done

command -v apptainer >/dev/null 2>&1 || die "apptainer is unavailable"
command -v chmod >/dev/null 2>&1 || die "chmod is unavailable"
command -v cmp >/dev/null 2>&1 || die "cmp is unavailable"
command -v cp >/dev/null 2>&1 || die "cp is unavailable"
command -v git >/dev/null 2>&1 || die "git is unavailable"
command -v ln >/dev/null 2>&1 || die "ln is unavailable"
command -v mktemp >/dev/null 2>&1 || die "mktemp is unavailable"
command -v realpath >/dev/null 2>&1 || die "realpath is unavailable"
command -v rm >/dev/null 2>&1 || die "rm is unavailable"
command -v sbatch >/dev/null 2>&1 || die "sbatch is unavailable"
command -v sha256sum >/dev/null 2>&1 || die "sha256sum is unavailable"
command -v stat >/dev/null 2>&1 || die "stat is unavailable"

repo_root="$(realpath -e -- "${PRORM_R3_REPO_ROOT}")"
project_root="$(realpath -e -- "${PRORM_R3_PROJECT_ROOT}")"
scratch_root="$(realpath -e -- "${PRORM_R3_SCRATCH_ROOT}")"
image="$(realpath -e -- "${PRORM_R3_IMAGE}")"
hf_cache="$(realpath -e -- "${PRORM_R3_HF_CACHE}")"
science_config="$(realpath -e -- "${PRORM_R3_SCIENCE_CONFIG}")"
requested_source_config="${PRORM_R3_SOURCE_CONFIG}"
requested_parent_registry="${PRORM_R3_PARENT_REGISTRY}"
scheduler_raw_evidence="$(realpath -e -- "${PRORM_R3_SCHEDULER_RAW_EVIDENCE}")"
resource_raw_evidence="$(realpath -e -- "${PRORM_R3_RESOURCE_RAW_EVIDENCE}")"
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
[[ -f "${image}" && "${image}" == "${project_root}/"* ]] \
  || die "container must be inside the persistent project root"
[[ -d "${hf_cache}" && "${hf_cache}" == "${project_root}/"* ]] \
  || die "HF cache must be inside the persistent project root"
[[ "${science_config}" == \
  "${repo_root}/configs/phase2_recovery_r3_science.yaml" ]] \
  || die "science config must be the frozen R3 production config"
[[ "${scheduler_raw_evidence}" == "${project_root}/"* ]] \
  || die "scheduler evidence must be durable project evidence"
[[ "${resource_raw_evidence}" == "${project_root}/"* ]] \
  || die "resource evidence must be durable project evidence"
[[ "${PRORM_R3_CLUSTER}" == "hpc4" ]] || die "unexpected cluster"
[[ "${PRORM_R3_PARTITION}" == "gpu-l20" ]] || die "unexpected Gate-P partition"
[[ "${PRORM_R3_PROFILE_WALLTIME_SECONDS}" =~ ^[1-9][0-9]*$ ]] \
  || die "profile walltime seconds must be a positive integer"
[[ "${PRORM_R3_PROFILE_CPUS_PER_TASK}" =~ ^[1-9][0-9]*$ ]] \
  || die "profile CPU count must be a positive integer"
[[ "${PRORM_R3_PROFILE_MEMORY_MIB}" =~ ^[1-9][0-9]*$ ]] \
  || die "profile memory MiB must be a positive integer"

commit="$(git -C "${repo_root}" rev-parse HEAD)"
[[ "${commit}" =~ ^[0-9a-f]{40}$ ]] || die "invalid Git commit"
[[ -z "$(git -C "${repo_root}" status --porcelain --untracked-files=all)" ]] \
  || die "production checkout is not clean"

# The materializer resolves the parent registry and source config inside the
# persistent project root.  Publish byte-identical, no-overwrite copies from
# the clean commit before exporting their runtime paths.
repo_source_config="$(
  realpath -e -- "${repo_root}/configs/common_beta_recovery_pilot.yaml"
)"
repo_parent_registry="$(
  realpath -e -- "${repo_root}/configs/phase2_recovery_parent_failures.json"
)"
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

source_config="${input_root}/common_beta_recovery_pilot.yaml"
parent_registry="${input_root}/phase2_recovery_parent_failures.json"
[[ "${requested_source_config}" == "${source_config}" ]] \
  || die "PRORM_R3_SOURCE_CONFIG must name the fixed clean-commit copy"
[[ "${requested_parent_registry}" == "${parent_registry}" ]] \
  || die "PRORM_R3_PARENT_REGISTRY must name the fixed clean-commit copy"

input_temp=""
cleanup_input_temp() {
  if [[ -n "${input_temp}" && -e "${input_temp}" ]]; then
    rm -- "${input_temp}"
  fi
}
trap cleanup_input_temp EXIT

copy_no_overwrite_exact() {
  local source="$1"
  local destination="$2"
  local source_digest
  if [[ -e "${destination}" || -L "${destination}" ]]; then
    [[ -f "${destination}" && ! -L "${destination}" ]] \
      || die "retained input copy is not a regular file: ${destination}"
  else
    input_temp="$(
      mktemp "${input_root}/.$(basename -- "${destination}").XXXXXX"
    )"
    cp -- "${source}" "${input_temp}"
    chmod 0440 -- "${input_temp}"
    if ln -- "${input_temp}" "${destination}"; then
      :
    elif [[ ! -f "${destination}" || -L "${destination}" ]]; then
      die "input copy no-overwrite publication failed: ${destination}"
    fi
    rm -- "${input_temp}"
    input_temp=""
  fi
  [[ -f "${destination}" && ! -L "${destination}" ]] \
    || die "input copy publication failed: ${destination}"
  [[ "$(stat -c '%a' -- "${destination}")" == "440" ]] \
    || die "retained input copy must have mode 0440: ${destination}"
  source_digest="$(sha256sum -- "${source}")"
  source_digest="${source_digest%% *}"
  [[ "${source_digest}" =~ ^[0-9a-f]{64}$ ]] \
    || die "invalid clean-repository input SHA-256"
  printf '%s  %s\n' "${source_digest}" "${destination}" \
    | sha256sum --check --status \
    || die "retained input SHA-256 differs from clean repository bytes"
  cmp --silent -- "${source}" "${destination}" \
    || die "retained input copy differs from clean repository bytes"
}

copy_no_overwrite_exact "${repo_source_config}" "${source_config}"
copy_no_overwrite_exact "${repo_parent_registry}" "${parent_registry}"
source_config="$(realpath -e -- "${source_config}")"
parent_registry="$(realpath -e -- "${parent_registry}")"
export PRORM_R3_SOURCE_CONFIG="${source_config}"
export PRORM_R3_PARENT_REGISTRY="${parent_registry}"
trap - EXIT

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
printf '%s  %s\n' \
  "${PRORM_R3_SCHEDULER_RAW_EVIDENCE_SHA256}" \
  "${scheduler_raw_evidence}" \
  | sha256sum --check --status || die "scheduler evidence SHA-256 mismatch"
printf '%s  %s\n' \
  "${PRORM_R3_RESOURCE_RAW_EVIDENCE_SHA256}" \
  "${resource_raw_evidence}" \
  | sha256sum --check --status || die "resource evidence SHA-256 mismatch"

attempt_root_text="${PRORM_R3_GATEP_ATTEMPT_ROOT}"
[[ "${attempt_root_text}" == "${project_root}/"* ]] \
  || die "attempt evidence must be inside the persistent project root"
[[ ! -e "${attempt_root_text}" && ! -L "${attempt_root_text}" ]] \
  || die "Gate-P attempt root already exists"
attempt_parent="$(realpath -e -- "$(dirname -- "${attempt_root_text}")")"
attempt_root="${attempt_parent}/$(basename -- "${attempt_root_text}")"
mkdir -- "${attempt_root}"
attempt_root="$(realpath -e -- "${attempt_root}")"
[[ "${attempt_root}" == "${attempt_root_text}" ]] \
  || die "Gate-P attempt root must be canonical"
mkdir -- "${attempt_root}/logs"

profile_intent="${attempt_root}/profile-allocation-intent.json"
operational_bundle="${attempt_root}/gatep-operational-bundle.json"
runtime_receipt="${attempt_root}/profile-runtime-receipt.json"
io_parent="${scratch_root}/phase2-r3"
mkdir -p -- "${io_parent}"
io_parent="$(realpath -e -- "${io_parent}")"
io_probe="${io_parent}/gatep-${commit}-$(basename -- "${attempt_root}")"
[[ ! -e "${io_probe}" && ! -L "${io_probe}" ]] \
  || die "Gate-P I/O probe directory already exists"
mkdir -- "${io_probe}"
io_probe="$(realpath -e -- "${io_probe}")"
[[ ! -e "${profile_intent}" ]] || die "profile allocation intent already exists"
[[ ! -e "${operational_bundle}" ]] || die "operational bundle already exists"
[[ ! -e "${runtime_receipt}" ]] || die "runtime receipt already exists"

profile_memory_bytes="$((PRORM_R3_PROFILE_MEMORY_MIB * 1024 * 1024))"
terminal_cli="${repo_root}/scripts/hpc4/capture_phase2_r3_terminal.py"
apptainer exec \
  --cleanenv \
  --bind "${repo_root}:${repo_root}:ro" \
  --bind "${project_root}:${project_root}:rw" \
  --bind "${image}:${image}:ro" \
  --env "PYTHONPATH=${repo_root}/src" \
  --env "PRORM_R3_REPO_ROOT=${repo_root}" \
  --env "PRORM_R3_PROJECT_ROOT=${project_root}" \
  "${image}" \
  python3 "${terminal_cli}" profile-intent \
    --output "${profile_intent}" \
    --cluster "${PRORM_R3_CLUSTER}" \
    --account sigroup \
    --partition "${PRORM_R3_PARTITION}" \
    --gpu-name "${PRORM_R3_GPU_NAME}" \
    --gpus-per-task 1 \
    --cpus-per-task "${PRORM_R3_PROFILE_CPUS_PER_TASK}" \
    --memory-bytes "${profile_memory_bytes}" \
    --walltime-seconds "${PRORM_R3_PROFILE_WALLTIME_SECONDS}"

profile_intent_sha256="$(sha256sum -- "${profile_intent}" | cut -d' ' -f1)"
[[ "${profile_intent_sha256}" =~ ^[0-9a-f]{64}$ ]] \
  || die "invalid profile intent SHA-256"

export PRORM_R3_GIT_COMMIT="${commit}"
export PRORM_R3_PROFILE_INTENT="${profile_intent}"
export PRORM_R3_PROFILE_INTENT_FILE_SHA256="${profile_intent_sha256}"
export PRORM_R3_OPERATIONAL_BUNDLE="${operational_bundle}"
export PRORM_R3_PROFILE_RUNTIME_RECEIPT="${runtime_receipt}"
export PRORM_R3_IO_PROBE_DIRECTORY="${io_probe}"

seconds="${PRORM_R3_PROFILE_WALLTIME_SECONDS}"
days="$((seconds / 86400))"
hours="$(((seconds % 86400) / 3600))"
minutes="$(((seconds % 3600) / 60))"
remaining_seconds="$((seconds % 60))"
printf -v slurm_walltime '%d-%02d:%02d:%02d' \
  "${days}" "${hours}" "${minutes}" "${remaining_seconds}"

export_names=("${required_environment[@]}")
export_names+=(
  PRORM_R3_GIT_COMMIT
  PRORM_R3_PROFILE_INTENT
  PRORM_R3_PROFILE_INTENT_FILE_SHA256
  PRORM_R3_OPERATIONAL_BUNDLE
  PRORM_R3_PROFILE_RUNTIME_RECEIPT
  PRORM_R3_IO_PROBE_DIRECTORY
)
export_spec="NONE"
for name in "${export_names[@]}"; do
  [[ "${!name}" != *","* ]] || die "${name} contains a comma"
  export_spec+=",${name}=${!name}"
done

job_id="$(
  sbatch \
    --parsable \
    --account=sigroup \
    --partition="${PRORM_R3_PARTITION}" \
    --nodes=1 \
    --ntasks=1 \
    --gpus-per-node=1 \
    --cpus-per-task="${PRORM_R3_PROFILE_CPUS_PER_TASK}" \
    --mem="${PRORM_R3_PROFILE_MEMORY_MIB}M" \
    --time="${slurm_walltime}" \
    --job-name=prorm-r3-gatep \
    --output="${attempt_root}/logs/gatep-%j.out" \
    --error="${attempt_root}/logs/gatep-%j.err" \
    --export="${export_spec}" \
    "${repo_root}/scripts/hpc4/phase2_r3_gatep.sbatch"
)"
[[ "${job_id}" =~ ^[1-9][0-9]*$ ]] || die "sbatch returned an invalid job id"
printf 'R3_GATEP_JOB_ID=%s\n' "${job_id}"
printf 'R3_GATEP_ATTEMPT_ROOT=%s\n' "${attempt_root}"
