#!/usr/bin/env bash
# One independent Gate-C family/seed task; all resources are submitter-owned.

set -euo pipefail
umask 077

readonly SIF_PYTHON="/opt/conda/bin/python"

die() {
  printf 'R3 Gate-C task fatal: %s\n' "$*" >&2
  exit 1
}

required=(
  PRORM_R3_GATEC_REPO_ROOT
  PRORM_R3_GATEC_PROJECT_ROOT
  PRORM_R3_GATEC_IMAGE
  PRORM_R3_GATEC_IMAGE_SHA256
  PRORM_R3_GATEC_GIT_COMMIT
  PRORM_R3_GATEC_CONTROLS_CONFIG
  PRORM_R3_GATEC_PROFILE
  PRORM_R3_GATEC_PROFILE_FILE_SHA256
  PRORM_R3_GATEC_PLAN
  PRORM_R3_GATEC_PLAN_FILE_SHA256
  PRORM_R3_GATEC_PLAN_SHA256
  PRORM_R3_GATEC_FAMILY_INDEX
  PRORM_R3_GATEC_FAMILY
  PRORM_R3_GATEC_PARENT_REGISTRY_FILE_SHA256
  PRORM_R3_GATEC_SUBMISSION_ROOT
  PRORM_R3_GATEC_CHECKPOINT_CADENCE_UPDATES
  SLURM_ARRAY_JOB_ID
  SLURM_ARRAY_TASK_ID
  SLURM_JOB_ID
)
for name in "${required[@]}"; do
  [[ -n "${!name:-}" ]] || die "missing environment variable ${name}"
done
[[ "${SLURM_ARRAY_TASK_ID}" =~ ^[0-2]$ ]] || die "unexpected array task"
[[ "${PRORM_R3_GATEC_FAMILY_INDEX}" =~ ^[0-2]$ ]] \
  || die "unexpected family index"
[[ "${SLURM_JOB_ACCOUNT:-}" == "sigroup" ]] || die "unexpected Slurm account"
[[ "${SLURM_NNODES:-}" == "1" && "${SLURM_NTASKS:-}" == "1" ]] \
  || die "Gate-C requires one node and one task"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != *","* ]] \
  || die "Gate-C requires exactly one visible CUDA device"

repo_root="$(realpath -e -- "${PRORM_R3_GATEC_REPO_ROOT}")"
project_root="$(realpath -e -- "${PRORM_R3_GATEC_PROJECT_ROOT}")"
image="$(realpath -e -- "${PRORM_R3_GATEC_IMAGE}")"
controls_config="$(realpath -e -- "${PRORM_R3_GATEC_CONTROLS_CONFIG}")"
profile="$(realpath -e -- "${PRORM_R3_GATEC_PROFILE}")"
plan="$(realpath -e -- "${PRORM_R3_GATEC_PLAN}")"
submission_root="$(realpath -e -- "${PRORM_R3_GATEC_SUBMISSION_ROOT}")"
[[ "${repo_root}" == "/home/yyangjo/Smart-Reward-Model" ]] \
  || die "unexpected repository root"
[[ "${project_root}" == "/project/sigroup/smart-reward-model" ]] \
  || die "unexpected project root"
[[ "$(git -C "${repo_root}" rev-parse HEAD)" == "${PRORM_R3_GATEC_GIT_COMMIT}" ]] \
  || die "Git commit drift"
[[ -z "$(git -C "${repo_root}" status --porcelain --untracked-files=all)" ]] \
  || die "Git checkout is dirty"
printf '%s  %s\n' "${PRORM_R3_GATEC_IMAGE_SHA256}" "${image}" \
  | sha256sum --check --status || die "container SHA-256 mismatch"

input_root="$(
  realpath -e -- \
    "${project_root}/runs/phase2-recovery-r3/inputs/${PRORM_R3_GATEC_GIT_COMMIT}"
)"
[[ "${input_root}" == \
  "${project_root}/runs/phase2-recovery-r3/inputs/${PRORM_R3_GATEC_GIT_COMMIT}" ]] \
  || die "Gate-C input namespace is not canonical"
science_config="$(realpath -e -- "${input_root}/phase2_recovery_r3_science.yaml")"
source_config="$(realpath -e -- "${input_root}/common_beta_recovery_pilot.yaml")"
parent_registry="$(
  realpath -e -- "${input_root}/phase2_recovery_parent_failures.json"
)"
for path in "${science_config}" "${source_config}" "${parent_registry}"; do
  [[ -f "${path}" && ! -L "${path}" ]] \
    || die "Gate-C retained input must be a regular non-symlink file"
done
cmp --silent -- \
  "${repo_root}/configs/phase2_recovery_r3_science.yaml" \
  "${science_config}" \
  || die "retained science config differs from the clean commit"
cmp --silent -- \
  "${repo_root}/configs/common_beta_recovery_pilot.yaml" \
  "${source_config}" \
  || die "retained source config differs from the clean commit"
cmp --silent -- \
  "${repo_root}/configs/phase2_recovery_parent_failures.json" \
  "${parent_registry}" \
  || die "retained parent registry differs from the clean commit"
printf '%s  %s\n' \
  "${PRORM_R3_GATEC_PARENT_REGISTRY_FILE_SHA256}" \
  "${parent_registry}" \
  | sha256sum --check --status || die "parent registry SHA-256 mismatch"

families=(
  exact_margin_prorm_plus
  exact_soft_label_bt
  low_dimensional_prorm_plus
)
seeds=(20260801 20260802 20260803)
family="${families[PRORM_R3_GATEC_FAMILY_INDEX]}"
seed="${seeds[SLURM_ARRAY_TASK_ID]}"
[[ "${family}" == "${PRORM_R3_GATEC_FAMILY}" ]] || die "family mapping drift"
task_id="$((PRORM_R3_GATEC_FAMILY_INDEX * 3 + SLURM_ARRAY_TASK_ID))"
task_root="${submission_root}/task-${SLURM_ARRAY_TASK_ID}-seed-${seed}"
[[ ! -e "${task_root}" && ! -L "${task_root}" ]] \
  || die "task namespace already exists; refusing replacement"
mkdir -- "${task_root}"
task_root="$(realpath -e -- "${task_root}")"

science_runner="${repo_root}/scripts/hpc4/run_phase2_r3_control_family.py"
evidence_cli="${repo_root}/scripts/hpc4/phase2_r3_controls_evidence.py"
[[ -f "${science_runner}" && -f "${evidence_cli}" ]] \
  || die "committed Gate-C runner/evidence surface is incomplete"
result="${task_root}/family-result.json"
closure="${task_root}/task-closure.json"

runner_pid=""
forward_usr1() {
  if [[ -n "${runner_pid}" ]]; then
    kill -USR1 "${runner_pid}" 2>/dev/null || true
  fi
}
trap forward_usr1 USR1

apptainer exec \
  --cleanenv \
  --nv \
  --bind "${repo_root}:${repo_root}:ro" \
  --bind "${project_root}:${project_root}:rw" \
  --bind "${image}:${image}:ro" \
  --env "PYTHONPATH=${repo_root}/src" \
  --env "PYTHONNOUSERSITE=1" \
  --env "PYTHONDONTWRITEBYTECODE=1" \
  --env "PYTHONUNBUFFERED=1" \
  --env "SLURM_JOB_ID=${SLURM_JOB_ID}" \
  --env "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID}" \
  --env "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" \
  "${image}" \
  "${SIF_PYTHON}" "${science_runner}" \
    --project-root "${project_root}" \
    --science-config "${science_config}" \
    --source-config "${source_config}" \
    --parent-registry "${parent_registry}" \
    --parent-registry-file-sha256 \
      "${PRORM_R3_GATEC_PARENT_REGISTRY_FILE_SHA256}" \
    --controls-config "${controls_config}" \
    --profile "${profile}" \
    --profile-file-sha256 "${PRORM_R3_GATEC_PROFILE_FILE_SHA256}" \
    --plan "${plan}" \
    --plan-file-sha256 "${PRORM_R3_GATEC_PLAN_FILE_SHA256}" \
    --task-id "${task_id}" \
    --family "${family}" \
    --seed "${seed}" \
    --checkpoint-cadence-updates \
      "${PRORM_R3_GATEC_CHECKPOINT_CADENCE_UPDATES}" \
    --output "${result}" &
runner_pid="$!"
wait "${runner_pid}"
runner_pid=""
trap - USR1

result_sha256="$(sha256sum -- "${result}" | cut -d' ' -f1)"
apptainer exec \
  --cleanenv \
  --bind "${repo_root}:${repo_root}:ro" \
  --bind "${project_root}:${project_root}:rw" \
  --bind "${image}:${image}:ro" \
  --env "PYTHONPATH=${repo_root}/src" \
  --env "PYTHONNOUSERSITE=1" \
  --env "PYTHONDONTWRITEBYTECODE=1" \
  "${image}" \
  "${SIF_PYTHON}" "${evidence_cli}" close-task \
    --controls-config "${controls_config}" \
    --profile "${profile}" \
    --profile-file-sha256 "${PRORM_R3_GATEC_PROFILE_FILE_SHA256}" \
    --plan "${plan}" \
    --plan-file-sha256 "${PRORM_R3_GATEC_PLAN_FILE_SHA256}" \
    --result "${result}" \
    --result-file-sha256 "${result_sha256}" \
    --task-id "${task_id}" \
    --segment-index 1 \
    --output "${closure}"

printf 'pending_external_sacct_terminal_finalization\n'
