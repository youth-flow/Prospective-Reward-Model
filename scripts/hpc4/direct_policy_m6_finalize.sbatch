#!/usr/bin/env bash
#SBATCH --account=sigroup
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

set -euo pipefail
umask 027
: "${PRORM_REPO_ROOT:?}" "${PRORM_DIRECT_POLICY_CONFIG:?}" "${PRORM_IMAGE:?}"
: "${PRORM_SOURCE_RUN_ROOT:?}" "${PRORM_SOURCE_M6_ROOT:?}" "${PRORM_DIRECT_RUN_ROOT:?}" "${PRORM_RUN_ROOT:?}"
source "${PRORM_REPO_ROOT}/scripts/hpc4/verify_bound_runtime.sh"
verify_bound_runtime "${PRORM_IMAGE}" "${PRORM_IMAGE_SOURCE_COMMIT:?}" "${PRORM_GIT_COMMIT:?}" "${PRORM_REPO_ROOT}"
source "${PRORM_REPO_ROOT}/scripts/hpc4/runtime.sh"
binds="${PRORM_REPO_ROOT}:${PRORM_REPO_ROOT},${PRORM_SOURCE_RUN_ROOT}:${PRORM_SOURCE_RUN_ROOT},${PRORM_SOURCE_M6_ROOT}:${PRORM_SOURCE_M6_ROOT},${PRORM_DIRECT_RUN_ROOT}:${PRORM_DIRECT_RUN_ROOT},${PRORM_RUN_ROOT}:${PRORM_RUN_ROOT}"
container=(prorm_apptainer_exec --cleanenv --bind "${binds}" --pwd "${PRORM_REPO_ROOT}"
  --env "PYTHONPATH=${PRORM_REPO_ROOT}/src" --env "PRORM_GIT_COMMIT=${PRORM_GIT_COMMIT}"
  --env "PRORM_IMAGE_SHA256=${PRORM_IMAGE_SHA256:?}" --env "PRORM_IMAGE_SOURCE_COMMIT=${PRORM_IMAGE_SOURCE_COMMIT}"
  --env "PRORM_HF_INVENTORY_SHA256=${PRORM_HF_INVENTORY_SHA256:?}" --env "SLURM_JOB_ID=${SLURM_JOB_ID}" "${PRORM_IMAGE}")
seeds=(20261001 20261002 20261003); results=()
for seed in "${seeds[@]}"; do
  target_seed="${PRORM_RUN_ROOT}/seed-${seed}"
  # Keep the formal run self-contained without retraining: reward evaluation needs
  # the immutable DPO/AuxDPO fit tensors, while rollout wrote only exported PEFT
  # adapters and fresh generations into this run root.
  source_reference="${PRORM_DIRECT_RUN_ROOT}/seed-${seed}/reference"
  target_reference="${target_seed}/reference"
  if [[ ! -d "${target_reference}" ]]; then
    staging_reference="${target_seed}/.reference.${SLURM_JOB_ID}"
    [[ ! -e "${staging_reference}" ]]
    cp -a "${source_reference}" "${staging_reference}"
    mv "${staging_reference}" "${target_reference}"
  fi
  for method in dpo auxdpo; do
    source_fit="${PRORM_DIRECT_RUN_ROOT}/seed-${seed}/fits/${method}__beta_0p2"
    target_fit="${target_seed}/fits/${method}__beta_0p2"
    if [[ ! -d "${target_fit}" ]]; then
      mkdir -p "${target_seed}/fits"
      staging_fit="${target_seed}/fits/.${method}__beta_0p2.${SLURM_JOB_ID}"
      [[ ! -e "${staging_fit}" ]]
      cp -a "${source_fit}" "${staging_fit}"
      mv "${staging_fit}" "${target_fit}"
    fi
  done
  "${container[@]}" python -m smart_reward.cli assemble-six-policy-seed \
    "${PRORM_DIRECT_POLICY_CONFIG}" "${PRORM_SOURCE_RUN_ROOT}/seed-${seed}/artifact" \
    "${PRORM_SOURCE_RUN_ROOT}/seed-${seed}/reward_result.json" "${target_seed}" \
    "${PRORM_SOURCE_M6_ROOT}/seed-${seed}/evaluation.json" "${target_seed}/evaluation.json" --seed "${seed}"
  results+=("${target_seed}/evaluation.json")
done
"${container[@]}" python -m smart_reward.cli aggregate-six-policy \
  "${PRORM_DIRECT_POLICY_CONFIG}" "${PRORM_RUN_ROOT}/aggregate.json" --results "${results[@]}"
"${container[@]}" python -m smart_reward.cli audit-six-policy \
  "${PRORM_DIRECT_POLICY_CONFIG}" "${PRORM_RUN_ROOT}" "${PRORM_RUN_ROOT}/integrity-audit.json"
