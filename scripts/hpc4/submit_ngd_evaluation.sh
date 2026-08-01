#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 5 ]] || {
  echo "usage: $0 CONFIG IMAGE HF_CACHE SOURCE_RUN_ROOT RUN_ROOT" >&2
  exit 2
}

repo_root="$(git rev-parse --show-toplevel)"
config="$(realpath -e "$1")"
image="$(realpath -e "$2")"
hf_cache="$(realpath -e "$3")"
source_run_root="$(realpath -e "$4")"
run_root="$(realpath -m "$5")"
[[ -z "$(git -C "${repo_root}" status --porcelain)" ]] || {
  echo "NGD submission requires a clean worktree" >&2
  exit 2
}
case "${config}" in "${repo_root}/configs/"*.yaml) ;; *) echo "config must be in configs/" >&2; exit 2 ;; esac
case "${image}" in /project/sigroup/*) ;; *) echo "image must be under /project/sigroup" >&2; exit 2 ;; esac
case "${hf_cache}" in /project/sigroup/*) ;; *) echo "HF cache must be under /project/sigroup" >&2; exit 2 ;; esac
case "${source_run_root}" in /project/sigroup/*) ;; *) echo "source run must be archived under /project/sigroup" >&2; exit 2 ;; esac
case "${run_root}" in "/scratch/${USER}/"*) ;; *) echo "run root must be under /scratch/${USER}" >&2; exit 2 ;; esac
[[ ! -e "${run_root}" ]] || { echo "refusing to reuse run root: ${run_root}" >&2; exit 2; }
mkdir -p "${run_root}/logs"

git_commit="$(git -C "${repo_root}" rev-parse HEAD)"
image_sha="$(sha256sum "${image}" | cut -d' ' -f1)"
mapfile -t config_info < <(PYTHONPATH="${repo_root}/src" python3 - "${config}" <<'PY'
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
common_export="ALL,PRORM_REPO_ROOT=${repo_root},PRORM_CONFIG=${config},PRORM_IMAGE=${image},PRORM_IMAGE_SHA256=${image_sha},PRORM_HF_INVENTORY_SHA256=${inventory_sha},PRORM_GIT_COMMIT=${git_commit},PRORM_SOURCE_RUN_ROOT=${source_run_root},PRORM_RUN_ROOT=${run_root}"

evaluation_job="$(sbatch --parsable --job-name=prorm-ngd-evaluation --partition=amd \
  --time=01:00:00 --array="0-$((seed_count - 1))" \
  --output="${run_root}/logs/evaluation-%A_%a.out" \
  --export="${common_export}" "${repo_root}/scripts/hpc4/ngd_evaluate.sbatch")"
aggregate_job="$(sbatch --parsable --job-name=prorm-ngd-aggregate --partition=amd \
  --time=00:10:00 --dependency="afterok:${evaluation_job}" \
  --output="${run_root}/logs/aggregate-%j.out" \
  --export="${common_export}" "${repo_root}/scripts/hpc4/ngd_aggregate.sbatch")"
audit_job="$(sbatch --parsable --job-name=prorm-ngd-audit --partition=amd \
  --time=00:10:00 --dependency="afterok:${aggregate_job}" \
  --output="${run_root}/logs/audit-%j.out" \
  --export="${common_export}" "${repo_root}/scripts/hpc4/ngd_audit.sbatch")"
printf 'run_root=%s\n' "${run_root}"
printf 'evaluation_job=%s\n' "${evaluation_job}"
printf 'aggregate_job=%s\n' "${aggregate_job}"
printf 'audit_job=%s\n' "${audit_job}"
