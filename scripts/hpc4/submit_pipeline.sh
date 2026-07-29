#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 4 ]] || {
  echo "usage: $0 CONFIG IMAGE HF_CACHE RUN_ROOT" >&2
  exit 2
}

repo_root="$(git rev-parse --show-toplevel)"
config="$(realpath -e "$1")"
image="$(realpath -e "$2")"
hf_cache="$(realpath -e "$3")"
run_root="$(realpath -m "$4")"
[[ -z "$(git -C "${repo_root}" status --porcelain)" ]] || {
  echo "pipeline submission requires a clean worktree" >&2
  exit 2
}
case "${config}" in "${repo_root}/configs/"*.yaml) ;; *) echo "config must be in configs/" >&2; exit 2 ;; esac
case "${image}" in /project/sigroup/*) ;; *) echo "image must be under /project/sigroup" >&2; exit 2 ;; esac
case "${hf_cache}" in /project/sigroup/*) ;; *) echo "HF cache must be under /project/sigroup" >&2; exit 2 ;; esac
case "${run_root}" in "/scratch/${USER}/"*) ;; *) echo "run root must be under /scratch/${USER}" >&2; exit 2 ;; esac
mkdir -p "${run_root}/logs"

git_commit="$(git -C "${repo_root}" rev-parse HEAD)"
image_sha="$(sha256sum "${image}" | cut -d' ' -f1)"

mapfile -t config_info < <(PYTHONPATH="${repo_root}/src" python3 - "${config}" <<'PY'
import sys
from smart_reward.config import config_hash, load_config
config = load_config(sys.argv[1])
execution = config["execution"]
print(config_hash(config))
print(len(config["run"]["seeds"]))
print(1 + len(config["policy_update"]["reward_sources"]) * len(config["policy_update"]["beta_grid"]))
for key in (
    "rollout_max_parallel_policies",
    "materialization_walltime",
    "reward_walltime",
    "adapter_walltime",
    "rollout_walltime",
    "rollout_aggregate_walltime",
    "three_seed_aggregate_walltime",
):
    print(execution[key])
PY
)
config_hash="${config_info[0]}"
seed_count="${config_info[1]}"
policy_count="${config_info[2]}"
rollout_concurrency="${config_info[3]}"
materialization_time="${config_info[4]}"
reward_time="${config_info[5]}"
adapter_time="${config_info[6]}"
rollout_time="${config_info[7]}"
rollout_aggregate_time="${config_info[8]}"
aggregate_time="${config_info[9]}"
inventory="${hf_cache}/inventories/${config_hash}.json"
[[ -f "${inventory}" ]] || { echo "missing staged inventory: ${inventory}" >&2; exit 2; }
inventory_sha="$(sha256sum "${inventory}" | cut -d' ' -f1)"
common_export="ALL,PRORM_REPO_ROOT=${repo_root},PRORM_CONFIG=${config},PRORM_IMAGE=${image},PRORM_IMAGE_SHA256=${image_sha},PRORM_HF_CACHE=${hf_cache},PRORM_HF_INVENTORY_SHA256=${inventory_sha},PRORM_GIT_COMMIT=${git_commit},PRORM_RUN_ROOT=${run_root}"
seed_array="0-$((seed_count - 1))"

materialize_job="$(sbatch --parsable --job-name=prorm-materialize --partition=gpu-l20 \
  --time="${materialization_time}" --array="${seed_array}" \
  --output="${run_root}/logs/materialize-%A_%a.out" \
  --export="${common_export},PRORM_STAGE=materialize" \
  "${repo_root}/scripts/hpc4/stage_gpu.sbatch")"
reward_job="$(sbatch --parsable --job-name=prorm-reward --partition=gpu-l20 \
  --time="${reward_time}" --array="${seed_array}" --dependency="afterok:${materialize_job}" \
  --output="${run_root}/logs/reward-%A_%a.out" \
  --export="${common_export},PRORM_STAGE=reward" \
  "${repo_root}/scripts/hpc4/stage_gpu.sbatch")"
adapter_job="$(sbatch --parsable --job-name=prorm-adapters --partition=gpu-l20 \
  --time="${adapter_time}" --array="${seed_array}" --dependency="afterok:${reward_job}" \
  --output="${run_root}/logs/adapters-%A_%a.out" \
  --export="${common_export},PRORM_STAGE=adapters" \
  "${repo_root}/scripts/hpc4/stage_gpu.sbatch")"
rollout_last=$((seed_count * policy_count - 1))
rollout_job="$(sbatch --parsable --job-name=prorm-rollout --partition=gpu-l20 \
  --time="${rollout_time}" --array="0-${rollout_last}%${rollout_concurrency}" \
  --dependency="afterok:${adapter_job}" \
  --output="${run_root}/logs/rollout-%A_%a.out" \
  --export="${common_export},PRORM_STAGE=rollout-policy" \
  "${repo_root}/scripts/hpc4/stage_gpu.sbatch")"
rollout_aggregate_job="$(sbatch --parsable --job-name=prorm-rollout-aggregate --partition=amd \
  --time="${rollout_aggregate_time}" --array="${seed_array}" \
  --dependency="afterok:${rollout_job}" \
  --output="${run_root}/logs/rollout-aggregate-%A_%a.out" \
  --export="${common_export}" "${repo_root}/scripts/hpc4/stage_cpu.sbatch")"
aggregate_job="$(sbatch --parsable --job-name=prorm-three-seed-aggregate --partition=amd \
  --time="${aggregate_time}" --dependency="afterok:${rollout_aggregate_job}" \
  --output="${run_root}/logs/aggregate-%j.out" \
  --export="${common_export}" "${repo_root}/scripts/hpc4/aggregate.sbatch")"

printf 'materialize_job=%s\n' "${materialize_job}"
printf 'reward_job=%s\n' "${reward_job}"
printf 'adapter_job=%s\n' "${adapter_job}"
printf 'rollout_job=%s\n' "${rollout_job}"
printf 'rollout_aggregate_job=%s\n' "${rollout_aggregate_job}"
printf 'aggregate_job=%s\n' "${aggregate_job}"
