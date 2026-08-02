#!/usr/bin/env bash
set -euo pipefail
umask 027
[[ $# -eq 7 ]] || { echo "usage: $0 CONFIG IMAGE HF_CACHE SOURCE_RUN DIRECT_RUN SOURCE_M6 RUN_ROOT" >&2; exit 2; }
repo_root="$(git rev-parse --show-toplevel)"; config="$(realpath -e "$1")"; image="$(realpath -e "$2")"
hf_cache="$(realpath -e "$3")"; source_run="$(realpath -e "$4")"; direct_run="$(realpath -e "$5")"
source_m6="$(realpath -e "$6")"; run_root="$(realpath -m "$7")"
[[ -z "$(git -C "${repo_root}" status --porcelain)" ]] || { echo "submission requires clean worktree" >&2; exit 2; }
[[ ! -e "${run_root}" ]] || { echo "refusing to reuse run root" >&2; exit 2; }; mkdir -p "${run_root}/logs"
git_commit="$(git -C "${repo_root}" rev-parse HEAD)"; image_source_commit="${PRORM_IMAGE_SOURCE_COMMIT:-${git_commit}}"
image_sha="$(sha256sum "${image}" | cut -d' ' -f1)"
source_sha="$(PYTHONPATH="${repo_root}/src" python3 -c 'import sys; from smart_reward.config import config_hash,load_config; print(config_hash(load_config(sys.argv[1])))' "${repo_root}/configs/fisher_trpo_main.yaml")"
inventory="${hf_cache}/inventories/${source_sha}.json"; [[ -f "${inventory}" ]] || { echo "missing HF inventory" >&2; exit 2; }
inventory_sha="$(sha256sum "${inventory}" | cut -d' ' -f1)"
common="ALL,PRORM_REPO_ROOT=${repo_root},PRORM_DIRECT_POLICY_CONFIG=${config},PRORM_IMAGE=${image},PRORM_IMAGE_SOURCE_COMMIT=${image_source_commit},PRORM_IMAGE_SHA256=${image_sha},PRORM_HF_CACHE=${hf_cache},PRORM_HF_INVENTORY_SHA256=${inventory_sha},PRORM_GIT_COMMIT=${git_commit},PRORM_SOURCE_RUN_ROOT=${source_run},PRORM_DIRECT_RUN_ROOT=${direct_run},PRORM_SOURCE_M6_ROOT=${source_m6},PRORM_RUN_ROOT=${run_root}"
prepare="$(sbatch --parsable --job-name=prorm-direct-prepare --partition=gpu-l20 --exclude=gpu18,gpu19 \
  --time=01:30:00 --gpus-per-node=1 --output="${run_root}/logs/prepare-%j.out" \
  --export="${common},PRORM_DIRECT_STAGE=prepare" "${repo_root}/scripts/hpc4/direct_policy_m6_gpu.sbatch")"
rollout="$(sbatch --parsable --job-name=prorm-direct-rollout --partition=gpu-l20 --exclude=gpu18,gpu19 \
  --time=05:00:00 --array=0-1 --gpus-per-node=3 --dependency="afterok:${prepare}" \
  --output="${run_root}/logs/rollout-%A_%a.out" \
  --export="${common},PRORM_DIRECT_STAGE=rollout,PRORM_REAL_WORKERS=6,PRORM_REAL_GPUS_PER_JOB=3" \
  "${repo_root}/scripts/hpc4/direct_policy_m6_gpu.sbatch")"
finalize="$(sbatch --parsable --job-name=prorm-six-finalize --partition=amd --time=01:30:00 \
  --dependency="afterok:${rollout}" --output="${run_root}/logs/finalize-%j.out" \
  --export="${common}" "${repo_root}/scripts/hpc4/direct_policy_m6_finalize.sbatch")"
printf 'run_root=%s\nprepare_job=%s\nrollout_job=%s\nfinalize_job=%s\n' "${run_root}" "${prepare}" "${rollout}" "${finalize}"
