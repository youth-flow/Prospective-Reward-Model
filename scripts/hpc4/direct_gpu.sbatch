#!/usr/bin/env bash
#SBATCH --account=sigroup
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail
umask 027

: "${PRORM_REPO_ROOT:?}"
: "${PRORM_EXTENSION_CONFIG:?}"
: "${PRORM_IMAGE:?}"
: "${PRORM_IMAGE_SHA256:?}"
: "${PRORM_HF_CACHE:?}"
: "${PRORM_HF_INVENTORY_SHA256:?}"
: "${PRORM_GIT_COMMIT:?}"
: "${PRORM_SOURCE_RUN_ROOT:?}"
: "${PRORM_RUN_ROOT:?}"
: "${PRORM_DIRECT_STAGE:?}"
: "${PRORM_DIRECT_WORKERS:?}"
: "${SLURM_ARRAY_TASK_ID:?}"

source "${PRORM_REPO_ROOT}/scripts/hpc4/verify_image_revision.sh"
verify_image_revision "${PRORM_IMAGE}" "${PRORM_GIT_COMMIT}"
source "${PRORM_REPO_ROOT}/scripts/hpc4/runtime.sh"

container=(
  prorm_apptainer_exec --cleanenv --nv
  --bind "${PRORM_REPO_ROOT}:${PRORM_REPO_ROOT},${PRORM_HF_CACHE}:${PRORM_HF_CACHE},${PRORM_SOURCE_RUN_ROOT}:${PRORM_SOURCE_RUN_ROOT},${PRORM_RUN_ROOT}:${PRORM_RUN_ROOT}"
  --pwd "${PRORM_REPO_ROOT}"
  --env "PYTHONPATH=${PRORM_REPO_ROOT}/src"
  --env "HF_HOME=${PRORM_HF_CACHE}"
  --env "HF_HUB_CACHE=${PRORM_HF_CACHE}/hub"
  --env "TRANSFORMERS_OFFLINE=1"
  --env "HF_HUB_OFFLINE=1"
  --env "PRORM_GIT_COMMIT=${PRORM_GIT_COMMIT}"
  --env "PRORM_IMAGE_SHA256=${PRORM_IMAGE_SHA256}"
  --env "PRORM_HF_INVENTORY_SHA256=${PRORM_HF_INVENTORY_SHA256}"
  --env "SLURM_JOB_ID=${SLURM_JOB_ID}"
  "${PRORM_IMAGE}"
)

mapfile -t conditions < <(
  "${container[@]}" python - "${PRORM_EXTENSION_CONFIG}" "${PRORM_DIRECT_STAGE}" <<'PY'
import itertools
import sys
from smart_reward.direct_preference import load_direct_preference_config

config = load_direct_preference_config(sys.argv[1])
stage = sys.argv[2]
if stage == "reference":
    for seed in config["experiment"]["seeds"]:
        print(seed)
elif stage == "train":
    for seed, beta, method in itertools.product(
        config["experiment"]["seeds"],
        config["experiment"]["betas"],
        config["experiment"]["methods"],
    ):
        print(seed, beta, method)
else:
    raise SystemExit(f"unsupported direct GPU stage: {stage}")
PY
)

worker="${SLURM_ARRAY_TASK_ID}"
(( worker >= 0 && worker < PRORM_DIRECT_WORKERS )) || {
  echo "invalid direct worker index" >&2
  exit 2
}

for ((index = worker; index < ${#conditions[@]}; index += PRORM_DIRECT_WORKERS)); do
  read -r seed beta method <<< "${conditions[index]}"
  seed_root="${PRORM_RUN_ROOT}/seed-${seed}"
  artifact="${PRORM_SOURCE_RUN_ROOT}/seed-${seed}/artifact"
  mkdir -p "${seed_root}"
  if [[ "${PRORM_DIRECT_STAGE}" = "reference" ]]; then
    "${container[@]}" python -m smart_reward.cli cache-direct-reference \
      "${PRORM_EXTENSION_CONFIG}" "${artifact}" "${seed_root}/reference" \
      --seed "${seed}" --device cuda
  else
    "${container[@]}" python -m smart_reward.cli train-direct-preference \
      "${PRORM_EXTENSION_CONFIG}" "${artifact}" "${seed_root}/reference" \
      "${seed_root}/fits" --seed "${seed}" --beta "${beta}" \
      --method "${method}" --device cuda
  fi
done

