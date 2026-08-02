#!/usr/bin/env bash
#SBATCH --account=sigroup
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

set -euo pipefail
umask 027

: "${PRORM_REPO_ROOT:?}" "${PRORM_DIRECT_POLICY_CONFIG:?}" "${PRORM_IMAGE:?}"
: "${PRORM_HF_CACHE:?}" "${PRORM_SOURCE_RUN_ROOT:?}" "${PRORM_DIRECT_RUN_ROOT:?}"
: "${PRORM_RUN_ROOT:?}" "${PRORM_DIRECT_STAGE:?}"

source "${PRORM_REPO_ROOT}/scripts/hpc4/verify_bound_runtime.sh"
verify_bound_runtime "${PRORM_IMAGE}" "${PRORM_IMAGE_SOURCE_COMMIT:?}" \
  "${PRORM_GIT_COMMIT:?}" "${PRORM_REPO_ROOT}"
source "${PRORM_REPO_ROOT}/scripts/hpc4/runtime.sh"

binds="${PRORM_REPO_ROOT}:${PRORM_REPO_ROOT},${PRORM_HF_CACHE}:${PRORM_HF_CACHE},${PRORM_SOURCE_RUN_ROOT}:${PRORM_SOURCE_RUN_ROOT},${PRORM_DIRECT_RUN_ROOT}:${PRORM_DIRECT_RUN_ROOT},${PRORM_RUN_ROOT}:${PRORM_RUN_ROOT}"
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
run_gpu() { local device="$1"; shift; "${container_prefix[@]}" --env "CUDA_VISIBLE_DEVICES=${device}" "${PRORM_IMAGE}" "$@"; }
seeds=(20261001 20261002 20261003)
methods=(dpo auxdpo)

case "${PRORM_DIRECT_STAGE}" in
  prepare)
    device="${CUDA_VISIBLE_DEVICES%%,*}"
    for seed in "${seeds[@]}"; do
      source_seed="${PRORM_SOURCE_RUN_ROOT}/seed-${seed}"
      direct_seed="${PRORM_DIRECT_RUN_ROOT}/seed-${seed}"
      target_seed="${PRORM_RUN_ROOT}/seed-${seed}"
      mkdir -p "${target_seed}/adapters" "${target_seed}/smoke"
      for method in "${methods[@]}"; do
        name="${method}__beta_0p2"
        run_gpu "${device}" python -m smart_reward.cli export-direct-policy-adapter \
          "${PRORM_DIRECT_POLICY_CONFIG}" "${source_seed}/artifact" \
          "${direct_seed}/fits/${name}" "${target_seed}/adapters/${name}" \
          --seed "${seed}" --method "${method}" --device cuda
      done
    done
    seed="${seeds[0]}"
    for method in "${methods[@]}"; do
      name="${method}__beta_0p2"
      run_gpu "${device}" python -m smart_reward.cli smoke-direct-policy \
        "${PRORM_DIRECT_POLICY_CONFIG}" "${PRORM_SOURCE_RUN_ROOT}/seed-${seed}/artifact" \
        "${PRORM_DIRECT_RUN_ROOT}/seed-${seed}/fits/${name}" \
        "${PRORM_RUN_ROOT}/seed-${seed}/adapters/${name}" \
        "${PRORM_RUN_ROOT}/seed-${seed}/smoke/${name}.json" \
        --seed "${seed}" --method "${method}" --device cuda
    done
    ;;
  rollout)
    : "${SLURM_ARRAY_TASK_ID:?}" "${PRORM_REAL_GPUS_PER_JOB:?}" "${PRORM_REAL_WORKERS:?}"
    IFS=',' read -r -a visible <<< "${CUDA_VISIBLE_DEVICES:?}"
    (( ${#visible[@]} == PRORM_REAL_GPUS_PER_JOB )) || { echo "GPU visibility mismatch" >&2; exit 2; }
    run_slot() {
      local worker="$1" device="$2" task seed_index method_index seed method name
      for ((task=worker; task<6; task+=PRORM_REAL_WORKERS)); do
        seed_index=$((task / 2)); method_index=$((task % 2))
        seed="${seeds[seed_index]}"; method="${methods[method_index]}"; name="${method}__beta_0p2"
        mkdir -p "${PRORM_RUN_ROOT}/seed-${seed}/policy_rollouts"
        run_gpu "${device}" python -m smart_reward.cli rollout-direct-policy-m6 \
          "${PRORM_DIRECT_POLICY_CONFIG}" "${PRORM_SOURCE_RUN_ROOT}/seed-${seed}/artifact" \
          "${PRORM_DIRECT_RUN_ROOT}/seed-${seed}/fits/${name}" \
          "${PRORM_RUN_ROOT}/seed-${seed}/adapters/${name}" \
          "${PRORM_RUN_ROOT}/seed-${seed}/policy_rollouts/${name}" \
          --seed "${seed}" --method "${method}" --device cuda
      done
    }
    pids=()
    for ((local_worker=0; local_worker<PRORM_REAL_GPUS_PER_JOB; local_worker++)); do
      worker=$((SLURM_ARRAY_TASK_ID * PRORM_REAL_GPUS_PER_JOB + local_worker))
      (( worker < PRORM_REAL_WORKERS )) || continue
      run_slot "${worker}" "${visible[local_worker]}" & pids+=("$!")
    done
    failed=0; for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done; exit "${failed}"
    ;;
  *) echo "unknown direct-policy GPU stage: ${PRORM_DIRECT_STAGE}" >&2; exit 2 ;;
esac
