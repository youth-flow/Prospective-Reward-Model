#!/usr/bin/env bash
# Run Gate-P submission validation on an HPC4 L20 compute node.

set -euo pipefail
umask 077

die() {
  printf 'R3 Gate-P submission launcher fatal: %s\n' "$*" >&2
  exit 1
}

required_environment=(
  PRORM_R3_IMAGE
  PRORM_R3_IMAGE_SHA256
  PRORM_R3_REPO_ROOT
  PRORM_R3_PROJECT_ROOT
  PRORM_R3_SCRATCH_ROOT
  PRORM_R3_HF_CACHE
  PRORM_R3_GATEP_ATTEMPT_ROOT
  PRORM_R3_SCIENCE_CONFIG
  PRORM_R3_SOURCE_CONFIG
  PRORM_R3_PARENT_REGISTRY
  PRORM_R3_PARENT_REGISTRY_FILE_SHA256
  PRORM_R3_GATE0_FILE_SHA256
  PRORM_R3_GATE1_FILE_SHA256
  PRORM_R3_SOURCE_TEST_RECEIPT_FILE_SHA256
  PRORM_R3_SCHEDULER_RAW_EVIDENCE
  PRORM_R3_SCHEDULER_RAW_EVIDENCE_SHA256
  PRORM_R3_RESOURCE_RAW_EVIDENCE
  PRORM_R3_RESOURCE_RAW_EVIDENCE_SHA256
  PRORM_R3_CLUSTER
  PRORM_R3_PARTITION
  PRORM_R3_GPU_NAME
  PRORM_R3_GPU_TOTAL_MEMORY_BYTES
  PRORM_R3_MAX_ALLOCATION_WALL_SECONDS
  PRORM_R3_MAX_ARRAY_CONCURRENCY
  PRORM_R3_MAX_SCHEDULER_SEGMENTS
  PRORM_R3_MAX_GPUS_PER_TASK
  PRORM_R3_MAX_CPUS_PER_TASK
  PRORM_R3_MAX_MEMORY_BYTES
  PRORM_R3_WALLTIME_MARGIN_FRACTION
  PRORM_R3_FIXED_WALLTIME_MARGIN_SECONDS
  PRORM_R3_MEMORY_MARGIN_FRACTION
  PRORM_R3_SIGNAL_MARGIN_SECONDS
  PRORM_R3_CHECKPOINT_CADENCE_UPDATES
  PRORM_R3_PRIMARY_WALLTIME_SECONDS
  PRORM_R3_PRIMARY_ARRAY_CONCURRENCY
  PRORM_R3_PRIMARY_CPUS_PER_TASK
  PRORM_R3_PRIMARY_MEMORY_BYTES
  PRORM_R3_PROFILE_WALLTIME_SECONDS
  PRORM_R3_PROFILE_CPUS_PER_TASK
  PRORM_R3_PROFILE_MEMORY_MIB
)
for name in "${required_environment[@]}"; do
  [[ -n "${!name:-}" ]] || die "missing environment variable ${name}"
done
[[ -z "${SLURM_JOB_ID:-}" ]] \
  || die "login-side launcher must not run inside an existing Slurm allocation"
for command_name in env realpath srun; do
  command -v "${command_name}" >/dev/null 2>&1 \
    || die "${command_name} is unavailable"
done
while IFS='=' read -r name _; do
  case "${name}" in
    SBATCH_*) die "unset exported ${name}; ambient sbatch overrides are forbidden" ;;
  esac
done < <(env)

repo_root="$(realpath -e -- "${PRORM_R3_REPO_ROOT}")"
[[ "${repo_root}" == "/home/yyangjo/Smart-Reward-Model" ]] \
  || die "unexpected production Git repository root"
driver="${repo_root}/scripts/hpc4/phase2_r3_gatep_submission.sbatch"
[[ -f "${driver}" && ! -L "${driver}" ]] \
  || die "Gate-P compute-side submission driver is unavailable"
driver="$(realpath -e -- "${driver}")"

export_spec="PATH=/usr/bin:/bin"
for name in "${required_environment[@]}"; do
  value="${!name}"
  [[ "${value}" != *","* && "${value}" != *$'\n'* && "${value}" != *$'\r'* ]] \
    || die "${name} cannot be represented safely in srun --export"
  export_spec+=",${name}=${value}"
done

exec srun \
  --account=sigroup \
  --partition=gpu-l20 \
  --nodes=1 \
  --ntasks=1 \
  --gpus-per-node=1 \
  --cpus-per-task=2 \
  --mem=4G \
  --time=00:30:00 \
  --job-name=prorm-r3-gatep-submit \
  --chdir="${repo_root}" \
  --export="${export_spec}" \
  /usr/bin/bash "${driver}"
