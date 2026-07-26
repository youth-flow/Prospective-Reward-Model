#!/usr/bin/env bash
# Run the GateE materializer inside a short, blocking HPC4 L20 allocation.

set -euo pipefail
umask 077

die() {
  printf 'GateE materialization launcher fatal: %s\n' "$*" >&2
  exit 1
}

[[ $# -eq 3 ]] \
  || die "usage: $0 <accepted-freeze-overlay.yaml> <accepted-freeze-aggregate.json> <r3-authorization-sha256>"
[[ -z "${SLURM_JOB_ID:-}" ]] || die "launcher must run outside Slurm"
[[ "$3" =~ ^[0-9a-f]{64}$ ]] || die "authorization SHA-256 is invalid"
required_environment=(PRORM_PROJECT_ROOT PRORM_IMAGE PRORM_IMAGE_SHA256)
for name in "${required_environment[@]}"; do
  [[ -n "${!name:-}" ]] || die "${name} is required"
done
for command_name in env realpath srun; do
  command -v "${command_name}" >/dev/null 2>&1 || die "${command_name} is unavailable"
done
while IFS='=' read -r name _; do
  case "${name}" in
    APPTAINER*|SINGULARITY*|SBATCH_*|PRORM_BUDGETED_*|\
    PRORM_RECOVERY_AUTHORIZATION*|PRORM_REPO_ROOT)
      die "unset ambient materialization control variable: ${name}"
      ;;
  esac
done < <(env)

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
repo_root="$(realpath -e -- "${repo_root}")"
[[ "${repo_root}" == "/home/yyangjo/Smart-Reward-Model" ]] \
  || die "unexpected production Git repository root"
driver="${repo_root}/scripts/hpc4/phase2_budgeted_end_to_end_materialize.sbatch"
[[ -f "${driver}" && ! -L "${driver}" ]] || die "compute materializer is unavailable"

export_spec="PATH=/usr/bin:/bin"
for name in "${required_environment[@]}"; do
  value="${!name}"
  [[ "${value}" != *","* && "${value}" != *$'\n'* && "${value}" != *$'\r'* ]] \
    || die "${name} is unsafe for srun export"
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
  --job-name=prorm-gatee-materialize \
  --chdir="${repo_root}" \
  --export="${export_spec}" \
  /usr/bin/bash "${driver}" "$@"
