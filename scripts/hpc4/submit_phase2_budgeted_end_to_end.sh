#!/usr/bin/env bash
# Run GateE submission validation inside a short, blocking HPC4 L20 allocation.

set -euo pipefail
umask 077

die() {
  printf 'GateE submission launcher fatal: %s\n' "$*" >&2
  exit 1
}

[[ $# -eq 4 ]] \
  || die "usage: $0 <overlay.yaml> <r3-combined-authorization.json> <accepted-freeze-aggregate.json> <walltime>"
[[ "$4" =~ ^([1-9][0-9]*-[0-9]{2}:[0-9]{2}:[0-9]{2}|[0-9]{2}:[0-9]{2}:[0-9]{2})$ ]] \
  || die "walltime must be HH:MM:SS or D-HH:MM:SS"
[[ -z "${SLURM_JOB_ID:-}" ]] \
  || die "login-side launcher must not run inside an existing Slurm allocation"

required_environment=(
  PRORM_PROJECT_ROOT
  PRORM_SCRATCH_ROOT
  PRORM_IMAGE
  PRORM_IMAGE_SHA256
  PRORM_HF_CACHE
)
for name in "${required_environment[@]}"; do
  [[ -n "${!name:-}" ]] || die "${name} is required"
done
for command_name in env realpath srun; do
  command -v "${command_name}" >/dev/null 2>&1 \
    || die "${command_name} is unavailable"
done
while IFS='=' read -r name _; do
  case "${name}" in
    APPTAINER*|SINGULARITY*|SBATCH_*|PRORM_BUDGETED_*|\
    PRORM_RECOVERY_AUTHORIZATION*|PRORM_OPTIMIZER_SCHEDULE*|\
    PRORM_GIT_COMMIT|PRORM_HF_INVENTORY*|PRORM_REPO_ROOT)
      die "unset ambient control variable before GateE submission: ${name}"
      ;;
  esac
done < <(env)

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
repo_root="$(realpath -e -- "${repo_root}")"
[[ "${repo_root}" == "/home/yyangjo/Smart-Reward-Model" ]] \
  || die "unexpected production Git repository root"
driver="${repo_root}/scripts/hpc4/phase2_budgeted_end_to_end_submission.sbatch"
[[ -f "${driver}" && ! -L "${driver}" ]] \
  || die "GateE compute-side submission driver is unavailable"
driver="$(realpath -e -- "${driver}")"

export_spec="PATH=/usr/bin:/bin"
for name in "${required_environment[@]}"; do
  value="${!name}"
  [[ -n "${value}" && "${value}" != *","* \
    && "${value}" != *$'\n'* && "${value}" != *$'\r'* ]] \
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
  --job-name=prorm-gatee-submit \
  --chdir="${repo_root}" \
  --export="${export_spec}" \
  /usr/bin/bash "${driver}" "$@"
