#!/usr/bin/env bash
# R3 primary compute body.  Fresh or continuation state comes from sealed plans.

set -euo pipefail
umask 077

die() {
  printf 'R3 primary fatal: %s\n' "$*" >&2
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
  PRORM_R3_PRIMARY_ATTEMPT_ROOT
  PRORM_R3_PRIMARY_TASK_ROOT_BASE
  PRORM_R3_SCIENCE_CONFIG
  PRORM_R3_SOURCE_CONFIG
  PRORM_R3_PARENT_REGISTRY
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
for name in "${required_environment[@]}"; do
  [[ -n "${!name:-}" ]] || die "missing environment variable ${name}"
done
continuation_mode=false
if [[ -n "${PRORM_R3_CONTINUATION_PLAN:-}" || \
  -n "${PRORM_R3_CONTINUATION_PLAN_FILE_SHA256:-}" ]]; then
  [[ -n "${PRORM_R3_CONTINUATION_PLAN:-}" && \
    -n "${PRORM_R3_CONTINUATION_PLAN_FILE_SHA256:-}" ]] \
    || die "continuation plan path and SHA must appear together"
  continuation_mode=true
fi

command -v cmp >/dev/null 2>&1 || die "cmp is unavailable"
[[ "${SLURM_CLUSTER_NAME:-}" == "hpc4" ]] || die "unexpected Slurm cluster"
[[ -n "${SLURM_JOB_ID:-}" ]] || die "not running in a Slurm allocation"
[[ -n "${SLURM_ARRAY_JOB_ID:-}" ]] || die "primary execution requires an array job"
[[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^[012]$ ]] \
  || die "primary array task must be exactly 0, 1, or 2"
[[ "${SLURM_JOB_ACCOUNT:-}" == "${PRORM_R3_PRIMARY_ACCOUNT}" ]] \
  || die "Slurm account differs from the sealed primary plan"
[[ "${SLURM_JOB_PARTITION:-}" == "${PRORM_R3_PRIMARY_PARTITION}" ]] \
  || die "Slurm partition differs from the sealed primary plan"
[[ "${SLURM_NNODES:-}" == "1" ]] \
  || die "primary execution requires exactly one Slurm node"
[[ "${SLURM_NTASKS:-}" == "1" ]] \
  || die "primary execution requires exactly one Slurm task"
[[ "${SLURM_CPUS_PER_TASK:-}" == "${PRORM_R3_PRIMARY_CPUS_PER_TASK}" ]] \
  || die "Slurm CPU request differs from the sealed primary plan"
[[ "${SLURM_MEM_PER_NODE:-}" == "${PRORM_R3_PRIMARY_MEMORY_MIB}" ]] \
  || die "Slurm memory request differs from the sealed primary plan"
case "${SLURM_GPUS_ON_NODE:-}" in
  "${PRORM_R3_PRIMARY_GPUS_PER_TASK}"|"gpu:${PRORM_R3_PRIMARY_GPUS_PER_TASK}") ;;
  *) die "Slurm GPU allocation differs from the sealed primary plan" ;;
esac

repo_root="$(realpath -e -- "${PRORM_R3_REPO_ROOT}")"
project_root="$(realpath -e -- "${PRORM_R3_PROJECT_ROOT}")"
scratch_root="$(realpath -e -- "${PRORM_R3_SCRATCH_ROOT}")"
image="$(realpath -e -- "${PRORM_R3_IMAGE}")"
hf_cache="$(realpath -e -- "${PRORM_R3_HF_CACHE}")"
science_config="$(realpath -e -- "${PRORM_R3_SCIENCE_CONFIG}")"
attempt_root="$(realpath -e -- "${PRORM_R3_PRIMARY_ATTEMPT_ROOT}")"
[[ "${attempt_root}" == \
  "${project_root}/runs/phase2-recovery-r3/"* ]] \
  || die "primary attempt root is outside the canonical R3 namespace"
task_root_base="$(realpath -e -- "${PRORM_R3_PRIMARY_TASK_ROOT_BASE}")"
submission_plan="$(realpath -e -- "${PRORM_R3_PRIMARY_SUBMISSION_PLAN}")"
input_root="${project_root}/runs/phase2-recovery-r3/inputs/${PRORM_R3_GIT_COMMIT}"
[[ -d "${input_root}" && ! -L "${input_root}" ]] \
  || die "commit input namespace is not a real directory"
input_root="$(realpath -e -- "${input_root}")"
[[ "${input_root}" == \
  "${project_root}/runs/phase2-recovery-r3/inputs/${PRORM_R3_GIT_COMMIT}" ]] \
  || die "commit input namespace is not canonical"
expected_source_config="${input_root}/common_beta_pilot_base.yaml"
expected_parent_registry="${input_root}/phase2_recovery_parent_failures.json"
expected_science_config="${input_root}/phase2_recovery_r3_science.yaml"
[[ "${PRORM_R3_SCIENCE_CONFIG}" == "${expected_science_config}" ]] \
  || die "science config is not the exact retained clean-commit copy"
[[ "${PRORM_R3_SOURCE_CONFIG}" == "${expected_source_config}" ]] \
  || die "source config is not the exact retained clean-commit copy"
[[ "${PRORM_R3_PARENT_REGISTRY}" == "${expected_parent_registry}" ]] \
  || die "parent registry is not the exact retained clean-commit copy"
[[ -f "${expected_source_config}" && ! -L "${expected_source_config}" ]] \
  || die "retained source config is not a regular file"
[[ -f "${expected_parent_registry}" && ! -L "${expected_parent_registry}" ]] \
  || die "retained parent registry is not a regular file"
[[ -f "${expected_science_config}" && ! -L "${expected_science_config}" ]] \
  || die "retained science config is not a regular file"
science_config="$(realpath -e -- "${expected_science_config}")"
source_config="$(realpath -e -- "${expected_source_config}")"
parent_registry="$(realpath -e -- "${expected_parent_registry}")"

[[ "${repo_root}" == "/home/yyangjo/Smart-Reward-Model" ]] \
  || die "unexpected production Git repository root"
[[ "${project_root}" == "/project/sigroup/smart-reward-model" ]] \
  || die "unexpected persistent materialization root"
[[ "${scratch_root}" == "/scratch/yyangjo" ]] \
  || die "unexpected production scratch root"
[[ "${science_config}" == "${expected_science_config}" ]] \
  || die "science config must be the retained R3 production config"
[[ "${attempt_root}" == /project/sigroup/* ]] \
  || die "primary attempt root must be durable project evidence"
[[ "${task_root_base}" == "${scratch_root}/phase2-r3/"* ]] \
  || die "primary task root is outside the fixed scratch namespace"
[[ "${submission_plan}" == /project/sigroup/* ]] \
  || die "primary submission plan must be durable project evidence"
if [[ "${continuation_mode}" == "false" ]]; then
  [[ "${submission_plan}" == "${attempt_root}/"* ]] \
    || die "fresh primary submission plan is outside its attempt root"
fi

observed_commit="$(git -C "${repo_root}" rev-parse HEAD)"
[[ "${observed_commit}" == "${PRORM_R3_GIT_COMMIT}" ]] || die "Git commit drift"
[[ -z "$(git -C "${repo_root}" status --porcelain --untracked-files=all)" ]] \
  || die "production checkout is not clean"
cmp --silent -- \
  "${repo_root}/configs/phase2_recovery_r3_science.yaml" \
  "${science_config}" \
  || die "retained science config differs from the clean commit"
cmp --silent -- \
  "${repo_root}/configs/common_beta_pilot_base.yaml" \
  "${source_config}" \
  || die "retained source config differs from the clean commit"
cmp --silent -- \
  "${repo_root}/configs/phase2_recovery_parent_failures.json" \
  "${parent_registry}" \
  || die "retained parent registry differs from the clean commit"
printf '%s  %s\n' "${PRORM_R3_IMAGE_SHA256}" "${image}" \
  | sha256sum --check --status || die "container SHA-256 mismatch"
printf '%s  %s\n' \
  "${PRORM_R3_PARENT_REGISTRY_FILE_SHA256}" \
  "${parent_registry}" \
  | sha256sum --check --status || die "parent registry SHA-256 mismatch"
printf '%s  %s\n' \
  "${PRORM_R3_OPERATIONAL_BUNDLE_FILE_SHA256}" \
  "${PRORM_R3_OPERATIONAL_BUNDLE}" \
  | sha256sum --check --status || die "operational bundle SHA-256 mismatch"
printf '%s  %s\n' \
  "${PRORM_R3_PRIMARY_SUBMISSION_PLAN_FILE_SHA256}" \
  "${submission_plan}" \
  | sha256sum --check --status || die "primary submission-plan SHA-256 mismatch"

prepare_cli="${repo_root}/scripts/hpc4/prepare_phase2_r3_primary_submission.py"
runner="${repo_root}/scripts/hpc4/run_phase2_r3_primary.py"
[[ -f "${prepare_cli}" ]] || die "primary preparation CLI is missing"
[[ -f "${runner}" ]] || die "primary runner is missing"

observed_plan_lines="$(
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
      --plan-file-sha256 \
        "${PRORM_R3_PRIMARY_SUBMISSION_PLAN_FILE_SHA256}" \
      --format sbatch-lines
)" || die "failed to revalidate the primary submission plan"
seconds="${PRORM_R3_PRIMARY_WALLTIME_SECONDS}"
days="$((seconds / 86400))"
hours="$(((seconds % 86400) / 3600))"
minutes="$(((seconds % 3600) / 60))"
remaining_seconds="$((seconds % 60))"
printf -v expected_slurm_walltime '%d-%02d:%02d:%02d' \
  "${days}" "${hours}" "${minutes}" "${remaining_seconds}"
expected_plan_lines="$(
  printf '%s\n' \
    "submission_plan_sha256=${PRORM_R3_PRIMARY_SUBMISSION_PLAN_SHA256}" \
    "resource_plan_sha256=${PRORM_R3_PRIMARY_RESOURCE_PLAN_SHA256}" \
    "slurm_account=${PRORM_R3_PRIMARY_ACCOUNT}" \
    "partition=${PRORM_R3_PRIMARY_PARTITION}" \
    "gpu_name=${PRORM_R3_PRIMARY_GPU_NAME}" \
    "gpus_per_task=${PRORM_R3_PRIMARY_GPUS_PER_TASK}" \
    "cpus_per_task=${PRORM_R3_PRIMARY_CPUS_PER_TASK}" \
    "memory_bytes=${PRORM_R3_PRIMARY_MEMORY_BYTES}" \
    "memory_mib=${PRORM_R3_PRIMARY_MEMORY_MIB}" \
    "requested_walltime_seconds=${PRORM_R3_PRIMARY_WALLTIME_SECONDS}" \
    "slurm_walltime=${expected_slurm_walltime}" \
    "array_concurrency=${PRORM_R3_PRIMARY_ARRAY_CONCURRENCY}" \
    "max_scheduler_segments=${PRORM_R3_PRIMARY_MAX_SCHEDULER_SEGMENTS}" \
    "advance_signal_lead_seconds=${PRORM_R3_PRIMARY_SIGNAL_LEAD_SECONDS}" \
    "audit_cadence_updates=${PRORM_R3_PRIMARY_AUDIT_CADENCE_UPDATES}" \
    "durable_checkpoint_cadence_updates=${PRORM_R3_PRIMARY_CHECKPOINT_CADENCE_UPDATES}"
)"
[[ "${observed_plan_lines}" == "${expected_plan_lines}" ]] \
  || die "exported primary fields differ from the caller-pinned plan"

task_id="${SLURM_ARRAY_TASK_ID}"
task_root="${task_root_base}/task-${task_id}"
continuation_arguments=()
if [[ "${continuation_mode}" == "true" ]]; then
  continuation_plan="$(realpath -e -- "${PRORM_R3_CONTINUATION_PLAN}")"
  [[ "${continuation_plan}" == "${attempt_root}/"* ]] \
    || die "continuation plan is outside its attempt root"
  printf '%s  %s\n' \
    "${PRORM_R3_CONTINUATION_PLAN_FILE_SHA256}" \
    "${continuation_plan}" \
    | sha256sum --check --status || die "continuation plan SHA-256 mismatch"
  continuation_prepare="${repo_root}/scripts/hpc4/prepare_phase2_r3_continuation_submission.py"
  runner="${repo_root}/scripts/hpc4/run_phase2_r3_primary_continuation.py"
  [[ -f "${continuation_prepare}" ]] || die "continuation preparation CLI is missing"
  [[ -f "${runner}" ]] || die "continuation runner is missing"
  task_lines="$(
    apptainer exec \
      --cleanenv \
      --bind "${repo_root}:${repo_root}:ro" \
      --bind "/project/sigroup:/project/sigroup:ro" \
      --env "PYTHONPATH=${repo_root}/src" \
      --env "PRORM_R3_REPO_ROOT=${repo_root}" \
      --env "PRORM_R3_PROJECT_ROOT=${project_root}" \
      "${image}" \
      python3 "${continuation_prepare}" inspect \
        --plan "${continuation_plan}" \
        --plan-file-sha256 "${PRORM_R3_CONTINUATION_PLAN_FILE_SHA256}" \
        --format task-lines \
        --task-id "${task_id}"
  )" || die "failed to inspect continuation task route"
  expected_seed="$((20260801 + task_id))"
  if [[ "${task_lines}" == *$'\naction=complete\n'* ]]; then
    printf 'R3_PRIMARY_TASK_ID=%s\n' "${task_id}"
    printf '%s\n' 'R3_PRIMARY_STATUS=sealed_completed_task_not_resubmitted'
    exit 0
  fi
  [[ "${task_lines}" == \
    "task_id=${task_id}"$'\n'"seed=${expected_seed}"$'\n'"action=continue"$'\n'* ]] \
    || die "continuation task route is invalid"
  task_root="$(realpath -e -- "${task_root}")"
  continuation_arguments=(
    --continuation-plan "${continuation_plan}"
    --continuation-plan-file-sha256 \
      "${PRORM_R3_CONTINUATION_PLAN_FILE_SHA256}"
  )
else
  [[ ! -e "${task_root}" && ! -L "${task_root}" ]] \
    || die "segment-1 task root already exists"
  mkdir -- "${task_root}"
  task_root="$(realpath -e -- "${task_root}")"
fi
runtime_closure="${attempt_root}/runtime-closures/task-${task_id}.json"
[[ ! -e "${runtime_closure}" && ! -L "${runtime_closure}" ]] \
  || die "segment-1 runtime closure already exists or continuation attempt collided"

runner_pid=""
forward_usr1() {
  if [[ -n "${runner_pid}" ]] && kill -0 "${runner_pid}" 2>/dev/null; then
    kill -USR1 "${runner_pid}"
  fi
}
trap forward_usr1 USR1

apptainer exec \
  --cleanenv \
  --nv \
  --bind "${repo_root}:${repo_root}:ro" \
  --bind "/project/sigroup:/project/sigroup:rw" \
  --bind "/scratch/yyangjo:/scratch/yyangjo:rw" \
  --env "PYTHONPATH=${repo_root}/src" \
  --env "PRORM_R3_REPO_ROOT=${repo_root}" \
  --env "PRORM_R3_PROJECT_ROOT=${project_root}" \
  --env "HF_HOME=${hf_cache}" \
  --env "TRANSFORMERS_CACHE=${hf_cache}" \
  --env "SLURM_CLUSTER_NAME=${SLURM_CLUSTER_NAME}" \
  --env "SLURM_JOB_ID=${SLURM_JOB_ID}" \
  --env "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID}" \
  --env "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" \
  --env "SLURM_JOB_ACCOUNT=${SLURM_JOB_ACCOUNT}" \
  --env "SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION}" \
  "${image}" \
  python3 "${runner}" \
    --project-root "${project_root}" \
    --science-config "${science_config}" \
    --source-config "${source_config}" \
    --parent-registry "${parent_registry}" \
    --parent-registry-file-sha256 \
      "${PRORM_R3_PARENT_REGISTRY_FILE_SHA256}" \
    --gate0-file-sha256 "${PRORM_R3_GATE0_FILE_SHA256}" \
    --gate1-file-sha256 "${PRORM_R3_GATE1_FILE_SHA256}" \
    --source-test-receipt-file-sha256 \
      "${PRORM_R3_SOURCE_TEST_RECEIPT_FILE_SHA256}" \
    --operational-bundle "${PRORM_R3_OPERATIONAL_BUNDLE}" \
    --operational-bundle-file-sha256 \
      "${PRORM_R3_OPERATIONAL_BUNDLE_FILE_SHA256}" \
    --profile-allocation-intent "${PRORM_R3_PROFILE_INTENT}" \
    --profile-allocation-intent-file-sha256 \
      "${PRORM_R3_PROFILE_INTENT_FILE_SHA256}" \
    --profile-runtime-receipt "${PRORM_R3_PROFILE_RUNTIME_RECEIPT}" \
    --profile-runtime-receipt-file-sha256 \
      "${PRORM_R3_PROFILE_RUNTIME_RECEIPT_FILE_SHA256}" \
    --profile-terminal-evidence-directory \
      "${PRORM_R3_PROFILE_TERMINAL_EVIDENCE_DIRECTORY}" \
    --profile-terminal-manifest-file-sha256 \
      "${PRORM_R3_PROFILE_TERMINAL_MANIFEST_FILE_SHA256}" \
    --profile-terminal-raw-sacct-sha256 \
      "${PRORM_R3_PROFILE_TERMINAL_RAW_SACCT_SHA256}" \
    --primary-submission-plan "${submission_plan}" \
    --primary-submission-plan-file-sha256 \
      "${PRORM_R3_PRIMARY_SUBMISSION_PLAN_FILE_SHA256}" \
    --primary-submission-plan-sha256 \
      "${PRORM_R3_PRIMARY_SUBMISSION_PLAN_SHA256}" \
    "${continuation_arguments[@]}" \
    --task-root "${task_root}" \
    --runtime-closure "${runtime_closure}" &
runner_pid="$!"

runner_status=0
while true; do
  set +e
  wait "${runner_pid}"
  wait_status="$?"
  set -e
  if kill -0 "${runner_pid}" 2>/dev/null; then
    continue
  fi
  runner_status="${wait_status}"
  break
done
trap - USR1
[[ "${runner_status}" -eq 0 ]] \
  || die "primary runner failed with exit ${runner_status}"
[[ -f "${runtime_closure}" && ! -L "${runtime_closure}" ]] \
  || die "primary runner did not publish its runtime closure"
runtime_closure_file_sha256="$(
  sha256sum -- "${runtime_closure}" | cut -d' ' -f1
)"
[[ "${runtime_closure_file_sha256}" =~ ^[0-9a-f]{64}$ ]] \
  || die "invalid runtime-closure file SHA-256"

printf 'R3_PRIMARY_TASK_ID=%s\n' "${task_id}"
printf 'R3_PRIMARY_RUNTIME_CLOSURE=%s\n' "${runtime_closure}"
printf 'R3_PRIMARY_RUNTIME_CLOSURE_FILE_SHA256=%s\n' \
  "${runtime_closure_file_sha256}"
printf '%s\n' \
  'R3_PRIMARY_STATUS=pending_external_sacct_terminal_finalization'
