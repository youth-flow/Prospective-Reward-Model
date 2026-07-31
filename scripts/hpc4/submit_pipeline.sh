#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 5 ]] || {
  echo "usage: $0 CONFIG IMAGE HF_CACHE RUN_ROOT STAGE" >&2
  exit 2
}

repo_root="$(git rev-parse --show-toplevel)"
config="$(realpath -e "$1")"
image="$(realpath -e "$2")"
hf_cache="$(realpath -e "$3")"
run_root="$(realpath -m "$4")"
stage="$5"
case "${stage}" in
  materialize|fisher-crossfit|fisher-select|reward|adapters|kl-calibration|kl-calibration-aggregate|rollout|rollout-aggregate|aggregate|audit) ;;
  *) echo "invalid pipeline stage: ${stage}" >&2; exit 2 ;;
esac
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
print(execution["rollout_max_parallel_policies"])
print(execution.get(
    "calibration_max_parallel_policies",
    execution["rollout_max_parallel_policies"],
))
print(execution["materialization_walltime"])
print(execution.get("crossfit_walltime", "01:00:00"))
print(execution["reward_walltime"])
print(execution["adapter_walltime"])
print(execution.get("kl_calibration_walltime", execution["adapter_walltime"]))
print(execution["rollout_walltime"])
print(execution["rollout_aggregate_walltime"])
print(execution["three_seed_aggregate_walltime"])
PY
)
config_hash="${config_info[0]}"
seed_count="${config_info[1]}"
rollout_concurrency="${config_info[2]}"
calibration_concurrency="${config_info[3]}"
materialization_time="${config_info[4]}"
crossfit_time="${config_info[5]}"
reward_time="${config_info[6]}"
adapter_time="${config_info[7]}"
calibration_time="${config_info[8]}"
rollout_time="${config_info[9]}"
rollout_aggregate_time="${config_info[10]}"
aggregate_time="${config_info[11]}"
inventory="${hf_cache}/inventories/${config_hash}.json"
[[ -f "${inventory}" ]] || { echo "missing staged inventory: ${inventory}" >&2; exit 2; }
inventory_sha="$(sha256sum "${inventory}" | cut -d' ' -f1)"
common_export="ALL,PRORM_REPO_ROOT=${repo_root},PRORM_CONFIG=${config},PRORM_IMAGE=${image},PRORM_IMAGE_SHA256=${image_sha},PRORM_HF_CACHE=${hf_cache},PRORM_HF_INVENTORY_SHA256=${inventory_sha},PRORM_GIT_COMMIT=${git_commit},PRORM_RUN_ROOT=${run_root}"
gpu_job_limit=2
if (( seed_count == 1 )); then
  seed_array="0"
else
  seed_array="0-$((seed_count - 1))%${gpu_job_limit}"
fi

case "${stage}" in
  materialize)
    job_id="$(sbatch --parsable --job-name=prorm-materialize --partition=gpu-l20 \
      --exclude=gpu19 \
      --time="${materialization_time}" --array="${seed_array}" \
      --output="${run_root}/logs/materialize-%A_%a.out" \
      --export="${common_export},PRORM_STAGE=materialize" \
      "${repo_root}/scripts/hpc4/stage_gpu.sbatch")"
    ;;
  fisher-crossfit)
    job_id="$(sbatch --parsable --job-name=prorm-fisher-crossfit --partition=amd \
      --time="${crossfit_time}" --array="0-$((seed_count - 1))" \
      --output="${run_root}/logs/fisher-crossfit-%A_%a.out" \
      --export="${common_export},PRORM_STAGE=fisher-crossfit" \
      "${repo_root}/scripts/hpc4/stage_cpu.sbatch")"
    ;;
  fisher-select)
    job_id="$(sbatch --parsable --job-name=prorm-fisher-select --partition=amd \
      --time="${crossfit_time}" \
      --output="${run_root}/logs/fisher-select-%j.out" \
      --export="${common_export}" "${repo_root}/scripts/hpc4/fisher_select.sbatch")"
    ;;
  reward)
    job_id="$(sbatch --parsable --job-name=prorm-reward --partition=gpu-l20 \
      --exclude=gpu19 \
      --time="${reward_time}" --array="${seed_array}" \
      --output="${run_root}/logs/reward-%A_%a.out" \
      --export="${common_export},PRORM_STAGE=reward" \
      "${repo_root}/scripts/hpc4/stage_gpu.sbatch")"
    ;;
  adapters)
    job_id="$(sbatch --parsable --job-name=prorm-adapters --partition=gpu-l20 \
      --exclude=gpu19 \
      --time="${adapter_time}" --array="${seed_array}" \
      --output="${run_root}/logs/adapters-%A_%a.out" \
      --export="${common_export},PRORM_STAGE=adapters" \
      "${repo_root}/scripts/hpc4/stage_gpu.sbatch")"
    ;;
  kl-calibration)
    (( calibration_concurrency >= 1 && calibration_concurrency <= 8 )) \
      || { echo "calibration concurrency must fit the l20_qos 8-GPU limit" >&2; exit 2; }
    calibration_jobs=$(( calibration_concurrency < gpu_job_limit ? calibration_concurrency : gpu_job_limit ))
    calibration_gpus_per_job=$(( (calibration_concurrency + calibration_jobs - 1) / calibration_jobs ))
    (( calibration_jobs * calibration_gpus_per_job <= 8 )) \
      || { echo "calibration workers exceed the l20_qos GPU limit" >&2; exit 2; }
    if (( calibration_jobs == 1 )); then calibration_array="0"; else calibration_array="0-$((calibration_jobs - 1))"; fi
    job_id="$(sbatch --parsable --job-name=prorm-kl-calibration --partition=gpu-l20 \
      --exclude=gpu19 \
      --time="${calibration_time}" --array="${calibration_array}" \
      --gpus-per-node="${calibration_gpus_per_job}" \
      --output="${run_root}/logs/kl-calibration-%A_%a.out" \
      --export="${common_export},PRORM_STAGE=kl-calibration-worker,PRORM_CALIBRATION_WORKERS=${calibration_concurrency},PRORM_CALIBRATION_GPUS_PER_JOB=${calibration_gpus_per_job}" \
      "${repo_root}/scripts/hpc4/stage_gpu.sbatch")"
    ;;
  kl-calibration-aggregate)
    job_id="$(sbatch --parsable --job-name=prorm-kl-calibration-aggregate --partition=amd \
      --time="${rollout_aggregate_time}" --array="0-$((seed_count - 1))" \
      --output="${run_root}/logs/kl-calibration-aggregate-%A_%a.out" \
      --export="${common_export},PRORM_STAGE=kl-calibration-aggregate" \
      "${repo_root}/scripts/hpc4/stage_cpu.sbatch")"
    ;;
  rollout)
    (( rollout_concurrency >= 1 && rollout_concurrency <= 8 )) \
      || { echo "rollout concurrency must fit the l20_qos 8-GPU limit" >&2; exit 2; }
    rollout_jobs=$(( rollout_concurrency < gpu_job_limit ? rollout_concurrency : gpu_job_limit ))
    gpus_per_job=$(( (rollout_concurrency + rollout_jobs - 1) / rollout_jobs ))
    (( rollout_jobs * gpus_per_job <= 8 )) \
      || { echo "rollout workers exceed the l20_qos GPU limit" >&2; exit 2; }
    if (( rollout_jobs == 1 )); then rollout_array="0"; else rollout_array="0-$((rollout_jobs - 1))"; fi
    job_id="$(sbatch --parsable --job-name=prorm-rollout --partition=gpu-l20 \
      --exclude=gpu19 \
      --time="${rollout_time}" --array="${rollout_array}" \
      --gpus-per-node="${gpus_per_job}" \
      --output="${run_root}/logs/rollout-%A_%a.out" \
      --export="${common_export},PRORM_STAGE=rollout-worker,PRORM_ROLLOUT_WORKERS=${rollout_concurrency},PRORM_ROLLOUT_GPUS_PER_JOB=${gpus_per_job}" \
      "${repo_root}/scripts/hpc4/stage_gpu.sbatch")"
    ;;
  rollout-aggregate)
    job_id="$(sbatch --parsable --job-name=prorm-rollout-aggregate --partition=amd \
      --time="${rollout_aggregate_time}" --array="0-$((seed_count - 1))" \
      --output="${run_root}/logs/rollout-aggregate-%A_%a.out" \
      --export="${common_export},PRORM_STAGE=rollout-aggregate" "${repo_root}/scripts/hpc4/stage_cpu.sbatch")"
    ;;
  aggregate)
    job_id="$(sbatch --parsable --job-name=prorm-three-seed-aggregate --partition=amd \
      --time="${aggregate_time}" \
      --output="${run_root}/logs/aggregate-%j.out" \
      --export="${common_export}" "${repo_root}/scripts/hpc4/aggregate.sbatch")"
    ;;
  audit)
    : "${PRORM_SOURCE_RUN_ROOT:?PRORM_SOURCE_RUN_ROOT is required for audit}"
    job_id="$(sbatch --parsable --job-name=prorm-integrity-audit --partition=amd \
      --time="${aggregate_time}" \
      --output="${run_root}/logs/audit-%j.out" \
      --export="${common_export},PRORM_SOURCE_RUN_ROOT=${PRORM_SOURCE_RUN_ROOT}" \
      "${repo_root}/scripts/hpc4/audit.sbatch")"
    ;;
esac

printf 'stage=%s\n' "${stage}"
printf 'job_id=%s\n' "${job_id}"
