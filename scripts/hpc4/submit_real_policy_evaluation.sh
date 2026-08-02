#!/usr/bin/env bash
set -euo pipefail
umask 027

[[ $# -eq 5 ]] || {
  echo "usage: $0 CONFIG IMAGE HF_CACHE SOURCE_RUN_ROOT RUN_ROOT" >&2
  exit 2
}

repo_root="$(git rev-parse --show-toplevel)"
config="$(realpath -e "$1")"
image="$(realpath -e "$2")"
hf_cache="$(realpath -e "$3")"
source_run="$(realpath -e "$4")"
run_root="$(realpath -m "$5")"
[[ -z "$(git -C "${repo_root}" status --porcelain)" ]] || {
  echo "real-policy submission requires a clean worktree" >&2
  exit 2
}
case "${config}" in "${repo_root}/configs/"*.yaml) ;; *) echo "config must be in configs/" >&2; exit 2 ;; esac
case "${image}" in /project/sigroup/*) ;; *) echo "image must be under /project/sigroup" >&2; exit 2 ;; esac
case "${hf_cache}" in /project/sigroup/*) ;; *) echo "HF cache must be under /project/sigroup" >&2; exit 2 ;; esac
case "${source_run}" in /project/sigroup/*) ;; *) echo "source run must be archived under /project/sigroup" >&2; exit 2 ;; esac
case "${run_root}" in "/scratch/${USER}/"*) ;; *) echo "run root must be under user scratch" >&2; exit 2 ;; esac
[[ ! -e "${run_root}" ]] || { echo "refusing to reuse run root" >&2; exit 2; }
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
config_sha="${config_info[0]}"
seed_count="${config_info[1]}"
inventory="${hf_cache}/inventories/${config_sha}.json"
[[ -f "${inventory}" ]] || { echo "missing source-config HF inventory" >&2; exit 2; }
inventory_sha="$(sha256sum "${inventory}" | cut -d' ' -f1)"
common="ALL,PRORM_REPO_ROOT=${repo_root},PRORM_CONFIG=${config},PRORM_IMAGE=${image},PRORM_IMAGE_SHA256=${image_sha},PRORM_HF_CACHE=${hf_cache},PRORM_HF_INVENTORY_SHA256=${inventory_sha},PRORM_GIT_COMMIT=${git_commit},PRORM_SOURCE_RUN_ROOT=${source_run},PRORM_RUN_ROOT=${run_root}"

adapters_job="$(sbatch --parsable --job-name=prorm-real-adapters --partition=gpu-l20 \
  --exclude=gpu18,gpu19 --time=01:00:00 --array="0-$((seed_count - 1))" \
  --gpus-per-node=1 --output="${run_root}/logs/adapters-%A_%a.out" \
  --export="${common},PRORM_REAL_STAGE=adapters" \
  "${repo_root}/scripts/hpc4/real_policy_gpu.sbatch")"
smoke_job="$(sbatch --parsable --job-name=prorm-real-smoke --partition=gpu-l20 \
  --exclude=gpu18,gpu19 --time=00:20:00 --array=0 --gpus-per-node=1 \
  --dependency="afterok:${adapters_job}" --output="${run_root}/logs/smoke-%j.out" \
  --export="${common},PRORM_REAL_STAGE=smoke" \
  "${repo_root}/scripts/hpc4/real_policy_gpu.sbatch")"
rollout_job="$(sbatch --parsable --job-name=prorm-real-rollout --partition=gpu-l20 \
  --exclude=gpu18,gpu19 --time=04:00:00 --array=0-1 --gpus-per-node=3 \
  --dependency="afterok:${smoke_job}" --output="${run_root}/logs/rollout-%A_%a.out" \
  --export="${common},PRORM_REAL_STAGE=rollout,PRORM_REAL_WORKERS=6,PRORM_REAL_GPUS_PER_JOB=3" \
  "${repo_root}/scripts/hpc4/real_policy_gpu.sbatch")"
seed_aggregate_job="$(sbatch --parsable --job-name=prorm-real-seed-aggregate --partition=amd \
  --time=00:30:00 --array="0-$((seed_count - 1))" --dependency="afterok:${rollout_job}" \
  --output="${run_root}/logs/seed-aggregate-%A_%a.out" \
  --export="${common},PRORM_REAL_STAGE=seed-aggregate" \
  "${repo_root}/scripts/hpc4/real_policy_cpu.sbatch")"
aggregate_job="$(sbatch --parsable --job-name=prorm-real-aggregate --partition=amd \
  --time=00:20:00 --dependency="afterok:${seed_aggregate_job}" \
  --output="${run_root}/logs/aggregate-%j.out" \
  --export="${common},PRORM_REAL_STAGE=aggregate" \
  "${repo_root}/scripts/hpc4/real_policy_cpu.sbatch")"
audit_job="$(sbatch --parsable --job-name=prorm-real-audit --partition=amd \
  --time=00:30:00 --dependency="afterok:${aggregate_job}" \
  --output="${run_root}/logs/audit-%j.out" \
  --export="${common},PRORM_REAL_STAGE=audit" \
  "${repo_root}/scripts/hpc4/real_policy_cpu.sbatch")"

printf 'run_root=%s\n' "${run_root}"
printf 'adapters_job=%s\n' "${adapters_job}"
printf 'smoke_job=%s\n' "${smoke_job}"
printf 'rollout_job=%s\n' "${rollout_job}"
printf 'seed_aggregate_job=%s\n' "${seed_aggregate_job}"
printf 'aggregate_job=%s\n' "${aggregate_job}"
printf 'audit_job=%s\n' "${audit_job}"
