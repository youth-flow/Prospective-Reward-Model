#!/usr/bin/env bash
#SBATCH --account=sigroup
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

set -euo pipefail
umask 027

: "${PRORM_REPO_ROOT:?}"
: "${PRORM_EXTENSION_CONFIG:?}"
: "${PRORM_IMAGE:?}"
: "${PRORM_RUN_ROOT:?}"
: "${PRORM_GIT_COMMIT:?}"

source "${PRORM_REPO_ROOT}/scripts/hpc4/verify_image_revision.sh"
verify_image_revision "${PRORM_IMAGE}" "${PRORM_GIT_COMMIT}"
source "${PRORM_REPO_ROOT}/scripts/hpc4/runtime.sh"
container=(
  prorm_apptainer_exec --cleanenv
  --bind "${PRORM_REPO_ROOT}:${PRORM_REPO_ROOT},${PRORM_RUN_ROOT}:${PRORM_RUN_ROOT}"
  --pwd "${PRORM_REPO_ROOT}"
  --env "PYTHONPATH=${PRORM_REPO_ROOT}/src"
  --env "PRORM_GIT_COMMIT=${PRORM_GIT_COMMIT}"
  "${PRORM_IMAGE}"
)
mapfile -t results < <(
  "${container[@]}" python - "${PRORM_EXTENSION_CONFIG}" "${PRORM_RUN_ROOT}" <<'PY'
import sys
from smart_reward.direct_preference import load_direct_preference_config
for seed in load_direct_preference_config(sys.argv[1])["experiment"]["seeds"]:
    print(f"{sys.argv[2]}/seed-{seed}/evaluation.json")
PY
)
"${container[@]}" python -m smart_reward.cli aggregate-direct-preference \
  "${PRORM_EXTENSION_CONFIG}" "${PRORM_RUN_ROOT}/aggregate.json" \
  --results "${results[@]}"

