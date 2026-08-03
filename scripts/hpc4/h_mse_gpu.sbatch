#!/usr/bin/env bash
#SBATCH --account=sigroup
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G

set -euo pipefail
umask 027

: "${PRORM_REPO_ROOT:?}" "${PRORM_H_CONFIG:?}" "${PRORM_IMAGE:?}"
: "${PRORM_HF_CACHE:?}" "${PRORM_SOURCE_RUN_ROOT:?}" "${PRORM_SOURCE_M6_ROOT:?}"
: "${PRORM_RUN_ROOT:?}" "${PRORM_H_STAGE:?}"

source "${PRORM_REPO_ROOT}/scripts/hpc4/verify_bound_runtime.sh"
verify_bound_runtime "${PRORM_IMAGE}" "${PRORM_IMAGE_SOURCE_COMMIT:?}" \
  "${PRORM_GIT_COMMIT:?}" "${PRORM_REPO_ROOT}"
source "${PRORM_REPO_ROOT}/scripts/hpc4/runtime.sh"

binds="${PRORM_REPO_ROOT}:${PRORM_REPO_ROOT},${PRORM_HF_CACHE}:${PRORM_HF_CACHE},${PRORM_SOURCE_RUN_ROOT}:${PRORM_SOURCE_RUN_ROOT},${PRORM_SOURCE_M6_ROOT}:${PRORM_SOURCE_M6_ROOT},${PRORM_RUN_ROOT}:${PRORM_RUN_ROOT}"
container_prefix=(
  prorm_apptainer_exec --cleanenv --nv --bind "${binds}" --pwd "${PRORM_REPO_ROOT}"
  --env "PYTHONPATH=${PRORM_REPO_ROOT}/src" --env "HF_HOME=${PRORM_HF_CACHE}"
  --env "HF_HUB_CACHE=${PRORM_HF_CACHE}/hub" --env "HF_DATASETS_CACHE=${PRORM_HF_CACHE}/datasets"
  --env "HF_HUB_OFFLINE=1" --env "TRANSFORMERS_OFFLINE=1" --env "HF_DATASETS_OFFLINE=1"
  --env "PRORM_GIT_COMMIT=${PRORM_GIT_COMMIT}" --env "PRORM_IMAGE_SHA256=${PRORM_IMAGE_SHA256:?}"
  --env "PRORM_IMAGE_SOURCE_COMMIT=${PRORM_IMAGE_SOURCE_COMMIT}"
  --env "PRORM_HF_INVENTORY_SHA256=${PRORM_HF_INVENTORY_SHA256:?}"
  --env "SLURM_JOB_ID=${SLURM_JOB_ID}"
)
run_container() { "${container_prefix[@]}" "${PRORM_IMAGE}" "$@"; }
run_gpu() {
  local device="$1"
  shift
  "${container_prefix[@]}" --env "CUDA_VISIBLE_DEVICES=${device}" "${PRORM_IMAGE}" "$@"
}

mapfile -t seeds < <(
  run_container python - "${PRORM_H_CONFIG}" <<'PY'
import sys
from smart_reward.h_ablation import load_h_ablation_config
for value in load_h_ablation_config(sys.argv[1])["experiment"]["seeds"]:
    print(value)
PY
)
mapfile -t methods < <(run_container python -m smart_reward.cli h-policy-names | sed 's/__beta_0p2$//')
IFS=',' read -r -a visible <<< "${CUDA_VISIBLE_DEVICES:?}"

prepare_seed() {
  local seed="$1" device="$2"
  local source_seed="${PRORM_SOURCE_RUN_ROOT}/seed-${seed}"
  local target_seed="${PRORM_RUN_ROOT}/seed-${seed}"
  mkdir -p "${target_seed}"
  run_gpu "${device}" python -m smart_reward.cli materialize-h-annotations \
    "${PRORM_H_CONFIG}" "${source_seed}/artifact" "${target_seed}/h_annotations" --seed "${seed}"
  run_gpu "${device}" python -m smart_reward.cli train-h-mse-rewards \
    "${PRORM_H_CONFIG}" "${source_seed}/artifact" "${source_seed}/reward_result.json" \
    "${PRORM_SOURCE_RUN_ROOT}/fisher_selection.json" "${target_seed}/h_annotations" \
    "${target_seed}/reward_result.json" --seed "${seed}" --device cuda
  run_gpu "${device}" python -m smart_reward.cli export-h-mse-adapters \
    "${PRORM_H_CONFIG}" "${source_seed}/artifact" "${target_seed}/reward_result.json" \
    "${target_seed}/adapters" --seed "${seed}" --device cuda
}

case "${PRORM_H_STAGE}" in
  smoke)
    (( ${#visible[@]} == 1 )) || { echo "smoke requires exactly one visible GPU" >&2; exit 2; }
    seed="${seeds[0]}"
    run_container python -m smart_reward.cli write-h-mse-provenance \
      "${PRORM_H_CONFIG}" "${PRORM_SOURCE_RUN_ROOT}" "${PRORM_SOURCE_M6_ROOT}" \
      "${PRORM_RUN_ROOT}/provenance.json"
    prepare_seed "${seed}" "${visible[0]}"
    mkdir -p "${PRORM_RUN_ROOT}/seed-${seed}/policy_rollouts"
    for method in "${methods[@]}"; do
      run_gpu "${visible[0]}" python -m smart_reward.cli rollout-h-mse-policy \
        "${PRORM_H_CONFIG}" "${PRORM_SOURCE_RUN_ROOT}/seed-${seed}/artifact" \
        "${PRORM_RUN_ROOT}/seed-${seed}/reward_result.json" \
        "${PRORM_RUN_ROOT}/seed-${seed}/adapters" \
        "${PRORM_RUN_ROOT}/seed-${seed}/policy_rollouts/${method}__beta_0p2" \
        --seed "${seed}" --method "${method}" --device cuda
    done
    run_container python -m smart_reward.cli assemble-h-mse-smoke \
      "${PRORM_H_CONFIG}" "${PRORM_SOURCE_RUN_ROOT}/seed-${seed}/artifact" \
      "${PRORM_RUN_ROOT}/seed-${seed}/reward_result.json" \
      "${PRORM_RUN_ROOT}/seed-${seed}/adapters" \
      "${PRORM_RUN_ROOT}/seed-${seed}/policy_rollouts" \
      "${PRORM_RUN_ROOT}/smoke.json" --seed "${seed}"
    ;;
  prepare)
    (( ${#visible[@]} >= ${#seeds[@]} )) || { echo "prepare requires one GPU per seed" >&2; exit 2; }
    run_container python -m smart_reward.cli write-h-mse-provenance \
      "${PRORM_H_CONFIG}" "${PRORM_SOURCE_RUN_ROOT}" "${PRORM_SOURCE_M6_ROOT}" \
      "${PRORM_RUN_ROOT}/provenance.json"
    pids=()
    for index in "${!seeds[@]}"; do
      prepare_seed "${seeds[index]}" "${visible[index]}" &
      pids+=("$!")
    done
    failed=0
    for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
    exit "${failed}"
    ;;
  rollout)
    : "${SLURM_ARRAY_TASK_ID:?}" "${PRORM_H_WORKERS:?}" "${PRORM_H_GPUS_PER_JOB:?}"
    (( ${#visible[@]} == PRORM_H_GPUS_PER_JOB )) || { echo "GPU visibility mismatch" >&2; exit 2; }
    total=$(( ${#seeds[@]} * ${#methods[@]} ))
    run_slot() {
      local worker="$1" device="$2" task seed_index method_index seed method target_seed
      for ((task=worker; task<total; task+=PRORM_H_WORKERS)); do
        seed_index=$((task / ${#methods[@]}))
        method_index=$((task % ${#methods[@]}))
        seed="${seeds[seed_index]}"
        method="${methods[method_index]}"
        target_seed="${PRORM_RUN_ROOT}/seed-${seed}"
        mkdir -p "${target_seed}/policy_rollouts"
        run_gpu "${device}" python -m smart_reward.cli rollout-h-mse-policy \
          "${PRORM_H_CONFIG}" "${PRORM_SOURCE_RUN_ROOT}/seed-${seed}/artifact" \
          "${target_seed}/reward_result.json" "${target_seed}/adapters" \
          "${target_seed}/policy_rollouts/${method}__beta_0p2" \
          --seed "${seed}" --method "${method}" --device cuda
      done
    }
    pids=()
    for ((local_worker=0; local_worker<PRORM_H_GPUS_PER_JOB; local_worker++)); do
      worker=$((SLURM_ARRAY_TASK_ID * PRORM_H_GPUS_PER_JOB + local_worker))
      (( worker < PRORM_H_WORKERS )) || continue
      run_slot "${worker}" "${visible[local_worker]}" &
      pids+=("$!")
    done
    failed=0
    for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
    exit "${failed}"
    ;;
  *) echo "unknown H/MSE GPU stage: ${PRORM_H_STAGE}" >&2; exit 2 ;;
esac
