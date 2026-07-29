#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 6 ]] || {
  echo "usage: $0 CONFIG IMAGE HF_CACHE RUN_ROOT PARTITION WALLTIME" >&2
  exit 2
}

repo_root="$(git rev-parse --show-toplevel)"
config="$(realpath -e "$1")"
image="$(realpath -e "$2")"
hf_cache="$(realpath -e "$3")"
run_root="$(realpath -m "$4")"
partition="$5"
walltime="$6"

[[ "${partition}" = "gpu-l20" ]] || {
  echo "the frozen experiment requires the HPC4 gpu-l20 partition" >&2
  exit 2
}
[[ -z "$(git -C "${repo_root}" status --porcelain)" ]] || {
  echo "formal submission requires a clean worktree" >&2
  exit 2
}
case "${config}" in "${repo_root}/configs/"*.yaml) ;; *) echo "config must be in configs/" >&2; exit 2 ;; esac
case "${hf_cache}" in /project/sigroup/*) ;; *) echo "HF cache must be under /project/sigroup" >&2; exit 2 ;; esac
case "${run_root}" in "/scratch/${USER}/"*) ;; *) echo "run root must be under /scratch/${USER}" >&2; exit 2 ;; esac
mkdir -p "${run_root}" "${run_root}/logs"
git_commit="$(git -C "${repo_root}" rev-parse HEAD)"
image_sha="$(sha256sum "${image}" | cut -d' ' -f1)"
image_commit="$(
  apptainer inspect --labels "${image}" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["org.opencontainers.image.revision"])'
)"
[[ "${image_commit}" = "${git_commit}" ]] || {
  echo "image commit ${image_commit} does not match worktree commit ${git_commit}" >&2
  exit 2
}
mapfile -t config_info < <(PYTHONPATH="${repo_root}/src" python - "${config}" <<'PY'
import sys
from smart_reward.config import config_hash, load_config
config = load_config(sys.argv[1])
print(config_hash(config))
print(len(config["run"]["seeds"]))
PY
)
config_hash="${config_info[0]}"
seed_count="${config_info[1]}"
inventory="${hf_cache}/inventories/${config_hash}.json"
[[ -f "${inventory}" ]] || { echo "missing staged inventory: ${inventory}" >&2; exit 2; }
inventory_sha="$(sha256sum "${inventory}" | cut -d' ' -f1)"

sbatch --parsable --array="0-$((seed_count - 1))" --partition="${partition}" --time="${walltime}" \
  --output="${run_root}/logs/prorm-main-%A_%a.out" \
  --export=ALL,PRORM_REPO_ROOT="${repo_root}",PRORM_CONFIG="${config}",PRORM_IMAGE="${image}",PRORM_IMAGE_SHA256="${image_sha}",PRORM_HF_CACHE="${hf_cache}",PRORM_HF_INVENTORY_SHA256="${inventory_sha}",PRORM_GIT_COMMIT="${git_commit}",PRORM_RUN_ROOT="${run_root}" \
  "${repo_root}/scripts/hpc4/controlled.sbatch"
