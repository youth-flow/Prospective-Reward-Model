#!/usr/bin/env bash
# Compute-side SIF driver for the fixed GateE materializer.

set -euo pipefail
umask 077

readonly AUTHORIZATION_RELATIVE="runs/phase2-recovery-r3-controls/gate-c-success-authorization.json"
readonly BIND_PLAN_RELATIVE="scripts/hpc4/phase2_budgeted_r3_bind_plan_stdlib.py"
readonly MATERIALIZER_RELATIVE="scripts/hpc4/materialize_phase2_budgeted_end_to_end.py"
readonly HOST_PYTHON="/opt/shared/spack/local/linux-rocky9-x86_64_v4/gcc-11.4.1/miniconda3-24.3.0-quc3pyudmzikgo2r4qsyqpwnrvzpin63/bin/python3.12"
readonly HOST_PYTHON_SHA256="9c91f9aa231c61c6bf2eabb9b93ebc5a8269a4126a36125e9548d8853e32da9c"

die() {
  printf 'GateE materialization fatal: %s\n' "$*" >&2
  exit 1
}

[[ $# -eq 3 ]] || die "invalid materializer argument count"
[[ -n "${SLURM_JOB_ID:-}" && "${SLURM_JOB_PARTITION:-}" == "gpu-l20" ]] \
  || die "materializer requires its dedicated gpu-l20 allocation"
for name in PRORM_PROJECT_ROOT PRORM_IMAGE PRORM_IMAGE_SHA256; do
  [[ -n "${!name:-}" ]] || die "${name} is required"
done
for command_name in apptainer git realpath sha256sum awk; do
  command -v "${command_name}" >/dev/null 2>&1 || die "${command_name} is unavailable"
done
host_python="$(realpath -e -- "${HOST_PYTHON}")" \
  || die "fixed host Python is unavailable"
[[ "${host_python}" == "${HOST_PYTHON}" && -f "${host_python}" \
  && -x "${host_python}" && ! -L "${host_python}" ]] \
  || die "fixed host Python must be a canonical executable"
printf '%s  %s\n' "${HOST_PYTHON_SHA256}" "${host_python}" \
  | sha256sum --check --status || die "fixed host Python SHA-256 mismatch"
[[ "$("${host_python}" --version 2>&1)" == "Python 3.12.2" ]] \
  || die "fixed host Python version mismatch"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
repo_root="$(realpath -e -- "${repo_root}")"
project_root="$(realpath -e -- "${PRORM_PROJECT_ROOT}")"
image="$(realpath -e -- "${PRORM_IMAGE}")"
source_overlay="$(realpath -e -- "$1")"
freeze_aggregate="$(realpath -e -- "$2")"
authorization_sha256="$3"
authorization="${project_root}/${AUTHORIZATION_RELATIVE}"
[[ "${repo_root}" == "/home/yyangjo/Smart-Reward-Model" ]] \
  || die "unexpected production Git repository root"
[[ "${project_root}" == "/project/sigroup/smart-reward-model" ]] \
  || die "unexpected production project root"
[[ "${source_overlay}" == "${repo_root}/configs/"* && -f "${source_overlay}" \
  && ! -L "${source_overlay}" ]] || die "source overlay is not a canonical repo config"
[[ "${freeze_aggregate}" == "${project_root}/aggregates/"* \
  && -f "${freeze_aggregate}" && ! -L "${freeze_aggregate}" ]] \
  || die "accepted freeze is not a canonical production aggregate"
[[ -f "${authorization}" && ! -L "${authorization}" ]] \
  || die "fixed R3 combined authorization is unavailable"
[[ "${authorization_sha256}" =~ ^[0-9a-f]{64}$ ]] \
  || die "authorization SHA-256 is invalid"
[[ "${PRORM_IMAGE_SHA256}" =~ ^[0-9a-f]{64}$ ]] \
  || die "image SHA-256 is invalid"
[[ "$(sha256sum -- "${image}" | awk '{print $1}')" == "${PRORM_IMAGE_SHA256}" ]] \
  || die "container image SHA-256 mismatch"
[[ -z "$(git -C "${repo_root}" status --porcelain --untracked-files=normal)" ]] \
  || die "materialization requires a clean committed checkout"

mapfile -t r3_bind_paths < <(
  "${host_python}" -I -S "${repo_root}/${BIND_PLAN_RELATIVE}" \
    "${authorization}" \
    --expected-sha256 "${authorization_sha256}" \
    --project-root "${project_root}" \
    --emit paths
) || die "minimal R3 verification bind-plan construction failed"
[[ "${#r3_bind_paths[@]}" -ge 10 ]] || die "R3 bind plan is incomplete"
r3_bind_arguments=()
for path in "${r3_bind_paths[@]}"; do
  [[ "${path}" == "${project_root}/"* \
    && "${path}" != *","* && "${path}" != *":"* ]] \
    || die "unsafe R3 verification bind path"
  r3_bind_arguments+=(--bind "${path}:${path}:ro")
done

apptainer exec \
  --cleanenv \
  --nv \
  --no-mount home,cwd,bind-paths \
  --pwd "${repo_root}" \
  --bind "${repo_root}:${repo_root}:rw" \
  --bind "${freeze_aggregate}:${freeze_aggregate}:ro" \
  "${r3_bind_arguments[@]}" \
  --env "PYTHONPATH=${repo_root}/src" \
  --env "PYTHONNOUSERSITE=1" \
  --env "PYTHONDONTWRITEBYTECODE=1" \
  "${image}" \
  python "${repo_root}/${MATERIALIZER_RELATIVE}" \
    "${source_overlay}" \
    "${freeze_aggregate}" \
    --repo-root "${repo_root}" \
    --authorization "${authorization}" \
    --authorization-sha256 "${authorization_sha256}"
