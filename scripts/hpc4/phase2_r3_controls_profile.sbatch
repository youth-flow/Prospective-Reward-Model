#!/usr/bin/env bash
# One disposable 100-update Gate-C profile task per family.

set -euo pipefail
umask 077

readonly SIF_PYTHON="/opt/conda/bin/python"

die() {
  printf 'R3 Gate-C profile fatal: %s\n' "$*" >&2
  exit 1
}

required=(
  PRORM_R3_GATEC_PROFILE_REPO_ROOT
  PRORM_R3_GATEC_PROFILE_PROJECT_ROOT
  PRORM_R3_GATEC_PROFILE_IMAGE
  PRORM_R3_GATEC_PROFILE_IMAGE_SHA256
  PRORM_R3_GATEC_PROFILE_GIT_COMMIT
  PRORM_R3_GATEC_PROFILE_CONTROLS_CONFIG
  PRORM_R3_GATEC_PROFILE_PARENT_REGISTRY_FILE_SHA256
  PRORM_R3_GATEC_PROFILE_SUBMISSION_ROOT
  SLURM_ARRAY_JOB_ID
  SLURM_ARRAY_TASK_ID
  SLURM_JOB_ID
)
for name in "${required[@]}"; do
  [[ -n "${!name:-}" ]] || die "missing environment variable ${name}"
done
[[ "${SLURM_ARRAY_TASK_ID}" =~ ^[0-2]$ ]] || die "unexpected profile array task"
[[ "${SLURM_JOB_ACCOUNT:-}" == "sigroup" ]] || die "unexpected Slurm account"
[[ "${SLURM_JOB_PARTITION:-}" == "gpu-l20" ]] || die "unexpected Slurm partition"
[[ "${SLURM_NNODES:-}" == "1" && "${SLURM_NTASKS:-}" == "1" ]] \
  || die "Gate-C profile requires one node and one task"
[[ "${SLURM_CPUS_PER_TASK:-}" == "8" ]] || die "unexpected profile CPU request"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != *","* ]] \
  || die "Gate-C profile requires exactly one visible CUDA device"

repo_root="$(realpath -e -- "${PRORM_R3_GATEC_PROFILE_REPO_ROOT}")"
project_root="$(realpath -e -- "${PRORM_R3_GATEC_PROFILE_PROJECT_ROOT}")"
image="$(realpath -e -- "${PRORM_R3_GATEC_PROFILE_IMAGE}")"
controls_config="$(realpath -e -- "${PRORM_R3_GATEC_PROFILE_CONTROLS_CONFIG}")"
submission_root="$(
  realpath -e -- "${PRORM_R3_GATEC_PROFILE_SUBMISSION_ROOT}"
)"
[[ "${repo_root}" == "/home/yyangjo/Smart-Reward-Model" ]] \
  || die "unexpected repository root"
[[ "${project_root}" == "/project/sigroup/smart-reward-model" ]] \
  || die "unexpected project root"
[[ "$(git -C "${repo_root}" rev-parse HEAD)" == \
  "${PRORM_R3_GATEC_PROFILE_GIT_COMMIT}" ]] || die "Git commit drift"
[[ -z "$(git -C "${repo_root}" status --porcelain --untracked-files=all)" ]] \
  || die "Git checkout is dirty"
printf '%s  %s\n' "${PRORM_R3_GATEC_PROFILE_IMAGE_SHA256}" "${image}" \
  | sha256sum --check --status || die "container SHA-256 mismatch"

input_root="$(
  realpath -e -- \
    "${project_root}/runs/phase2-recovery-r3/inputs/${PRORM_R3_GATEC_PROFILE_GIT_COMMIT}"
)"
science_config="$(realpath -e -- "${input_root}/phase2_recovery_r3_science.yaml")"
source_config="$(realpath -e -- "${input_root}/common_beta_pilot_base.yaml")"
parent_registry="$(
  realpath -e -- "${input_root}/phase2_recovery_parent_failures.json"
)"
cmp --silent -- \
  "${repo_root}/configs/phase2_recovery_r3_science.yaml" \
  "${science_config}" || die "retained science config drift"
cmp --silent -- \
  "${repo_root}/configs/common_beta_pilot_base.yaml" \
  "${source_config}" || die "retained source config drift"
cmp --silent -- \
  "${repo_root}/configs/phase2_recovery_parent_failures.json" \
  "${parent_registry}" || die "retained parent registry drift"
printf '%s  %s\n' \
  "${PRORM_R3_GATEC_PROFILE_PARENT_REGISTRY_FILE_SHA256}" \
  "${parent_registry}" \
  | sha256sum --check --status || die "parent registry SHA-256 mismatch"

families=(
  exact_margin_prorm_plus
  exact_soft_label_bt
  low_dimensional_prorm_plus
)
family="${families[SLURM_ARRAY_TASK_ID]}"
task_root="${submission_root}/task-${SLURM_ARRAY_TASK_ID}-${family}"
[[ ! -e "${task_root}" && ! -L "${task_root}" ]] \
  || die "profile task namespace already exists"
mkdir -- "${task_root}"
task_root="$(realpath -e -- "${task_root}")"
runner="${repo_root}/scripts/hpc4/run_phase2_r3_control_profile.py"
[[ -f "${runner}" ]] || die "Gate-C profile science runner is missing"

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
  --env "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" \
  "${image}" \
  "${SIF_PYTHON}" "${runner}" \
    --project-root "${project_root}" \
    --science-config "${science_config}" \
    --source-config "${source_config}" \
    --parent-registry "${parent_registry}" \
    --parent-registry-file-sha256 \
      "${PRORM_R3_GATEC_PROFILE_PARENT_REGISTRY_FILE_SHA256}" \
    --controls-config "${controls_config}" \
    --git-commit "${PRORM_R3_GATEC_PROFILE_GIT_COMMIT}" \
    --container-sha256 "${PRORM_R3_GATEC_PROFILE_IMAGE_SHA256}" \
    --checkpoint-directory "${task_root}" \
    --output "${task_root}/profile-compute-receipt.json"

printf 'profile_compute_complete_pending_external_sacct family=%s\n' "${family}"
