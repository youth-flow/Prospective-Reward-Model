#!/usr/bin/env bash
# One tracked Gate-1 allocation: clean-source tests, live SIF capture, verification.

#SBATCH --account=sigroup
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=0-12:00:00
#SBATCH --job-name=prorm-r3-gate1
#SBATCH --output=/project/sigroup/smart-reward-model/runs/phase2-recovery-r3/gate1/logs/gate1-%j.out
#SBATCH --error=/project/sigroup/smart-reward-model/runs/phase2-recovery-r3/gate1/logs/gate1-%j.err
#SBATCH --signal=B:TERM@300
#SBATCH --no-requeue

set -euo pipefail
umask 077

readonly REPO_ROOT="/home/yyangjo/Smart-Reward-Model"
readonly PROJECT_ROOT="/project/sigroup/smart-reward-model"
readonly SBATCH_RELATIVE="scripts/hpc4/phase2_r3_gate1.sbatch"
readonly SUBMIT_RELATIVE="scripts/hpc4/submit_phase2_r3_gate1.sh"
readonly CAPTURE_RELATIVE="scripts/hpc4/capture_phase2_r3_gate1.py"
readonly SOURCE_RECEIPT_RELATIVE="runs/phase2-recovery-r3/gate1/r3-source-test-receipt.json"
readonly GATE1_RELATIVE="runs/phase2-recovery-r3/gate1/r3-implementation-closure.json"
readonly FROZEN_IMAGE_SHA256="d6fc044b4fa303747908783ea057d5b8946f613bfec6a6ca301e3a02fd7719cb"
readonly SCRATCH_TOOLS="/scratch/yyangjo/r3-gate1-tools-ruff01522-pytest744"
readonly RUFF_SHA256="64aae5e444938e33121c3b940dff9b3d8ef8fc2a88c477e7f3a4fae2584a8fe8"
readonly HOST_PYTHON_MODULE="miniconda3/24.3.0-quc3pyu"
readonly HOST_PYTHON="/opt/shared/spack/local/linux-rocky9-x86_64_v4/gcc-11.4.1/miniconda3-24.3.0-quc3pyudmzikgo2r4qsyqpwnrvzpin63/bin/python3.12"
readonly HOST_PYTHON_SHA256="9c91f9aa231c61c6bf2eabb9b93ebc5a8269a4126a36125e9548d8853e32da9c"
readonly APPTAINER="/usr/bin/apptainer"
readonly SIF_PYTHON="/opt/conda/bin/python"

die() {
  printf 'R3 Gate-1 fatal: %s\n' "$*" >&2
  exit 1
}

required_environment=(
  PRORM_R3_GATE1_GIT_COMMIT
  PRORM_R3_GATE1_HOST_PYTHON
  PRORM_R3_GATE1_HOST_PYTHON_MODULE
  PRORM_R3_GATE1_HOST_PYTHON_SHA256
  PRORM_R3_GATE1_IMAGE
  PRORM_R3_GATE1_IMAGE_SHA256
  PRORM_R3_GATE1_RUFF_SHA256
  PRORM_R3_GATE1_SCRATCH_TOOLS
)
for name in "${required_environment[@]}"; do
  [[ -n "${!name:-}" ]] || die "missing environment variable ${name}"
done

for command_name in cmp git grep mkdir realpath sed sha256sum stat scontrol; do
  command -v "${command_name}" >/dev/null 2>&1 \
    || die "${command_name} is unavailable"
done

[[ "${SLURM_JOB_ACCOUNT:-}" == "sigroup" ]] || die "unexpected Slurm account"
[[ "${SLURM_JOB_PARTITION:-}" == "gpu-l20" ]] || die "unexpected Slurm partition"
[[ "${SLURM_NNODES:-}" == "1" ]] || die "Gate-1 requires exactly one node"
[[ "${SLURM_NTASKS:-}" == "1" ]] || die "Gate-1 requires exactly one task"
[[ -n "${SLURM_JOB_ID:-}" ]] || die "Gate-1 must execute inside Slurm"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != *","* ]] \
  || die "Gate-1 requires exactly one visible CUDA device"
[[ -z "${SLURM_ARRAY_JOB_ID:-}" && -z "${SLURM_ARRAY_TASK_ID:-}" ]] \
  || die "Gate-1 must not execute as an array task"
scontrol show config \
  | grep -E '^[[:space:]]*ClusterName[[:space:]]*=[[:space:]]*hpc4[[:space:]]*$' \
    >/dev/null \
  || die "Gate-1 requires ClusterName=hpc4"

repo_root="$(realpath -e -- "${REPO_ROOT}")"
project_root="$(realpath -e -- "${PROJECT_ROOT}")"
image="$(realpath -e -- "${PRORM_R3_GATE1_IMAGE}")"
scratch_tools="$(realpath -e -- "${PRORM_R3_GATE1_SCRATCH_TOOLS}")"
ruff_path="${scratch_tools}/bin/ruff"
host_python="${PRORM_R3_GATE1_HOST_PYTHON}"
host_python_target="$(realpath -e -- "${host_python}")"
[[ "${repo_root}" == "${REPO_ROOT}" ]] || die "production repository root is not canonical"
[[ "${project_root}" == "${PROJECT_ROOT}" ]] || die "project root is not canonical"
[[ -e "${repo_root}/.git" && ! -L "${repo_root}/.git" ]] \
  || die "production repository is not a real Git checkout"
[[ ! -e "${project_root}/.git" && ! -L "${project_root}/.git" ]] \
  || die "persistent project root must not be a Git checkout"
[[ "${PRORM_R3_GATE1_IMAGE}" == "${image}" ]] \
  || die "Gate-1 image path must already be canonical"
[[ -f "${image}" && ! -L "${image}" && "${image}" == "${project_root}/"* ]] \
  || die "Gate-1 image must be a non-symlink project SIF"
[[ "${scratch_tools}" == "${SCRATCH_TOOLS}" ]] \
  || die "Gate-1 scratch tools path drifted"
[[ -d "${scratch_tools}" && ! -L "${scratch_tools}" ]] \
  || die "Gate-1 scratch tools must be one canonical real directory"
[[ -f "${ruff_path}" && -x "${ruff_path}" && ! -L "${ruff_path}" ]] \
  || die "Gate-1 Ruff must be the fixed scratch executable"
[[ "${PRORM_R3_GATE1_RUFF_SHA256}" == "${RUFF_SHA256}" ]] \
  || die "caller Ruff SHA-256 differs from the frozen Gate-1 Ruff"
printf '%s  %s\n' "${RUFF_SHA256}" "${ruff_path}" \
  | sha256sum --check --status \
  || die "Gate-1 Ruff SHA-256 mismatch"
[[ "$("${ruff_path}" --version)" == "ruff 0.15.22" ]] \
  || die "Gate-1 Ruff version mismatch"
[[ "${PRORM_R3_GATE1_HOST_PYTHON_MODULE}" == "${HOST_PYTHON_MODULE}" ]] \
  || die "Gate-1 host Python provisioning module drifted"
[[ "${host_python}" == "${HOST_PYTHON}" ]] \
  || die "Gate-1 host Python path drifted"
[[ -f "${host_python}" && -x "${host_python}" ]] \
  || die "Gate-1 host Python launcher is unavailable"
[[ -f "${host_python_target}" && -x "${host_python_target}" && ! -L "${host_python_target}" ]] \
  || die "Gate-1 host Python target must be one executable regular file"
[[ "${PRORM_R3_GATE1_HOST_PYTHON_SHA256}" == "${HOST_PYTHON_SHA256}" ]] \
  || die "caller host Python SHA-256 differs from the frozen Gate-1 control Python"
printf '%s  %s\n' "${HOST_PYTHON_SHA256}" "${host_python}" \
  | sha256sum --check --status \
  || die "Gate-1 host Python SHA-256 mismatch"
[[ "$("${host_python}" --version 2>&1)" == "Python 3.12.2" ]] \
  || die "Gate-1 host Python version mismatch"
[[ -f "${APPTAINER}" && -x "${APPTAINER}" && ! -L "${APPTAINER}" ]] \
  || die "Gate-1 requires the fixed host Apptainer executable"

commit="$(git -C "${repo_root}" rev-parse HEAD)"
[[ "${commit}" == "${PRORM_R3_GATE1_GIT_COMMIT}" ]] || die "Git commit drift"
[[ "${commit}" =~ ^[0-9a-f]{40}$ ]] || die "invalid Git commit"
[[ -z "$(git -C "${repo_root}" status --porcelain --untracked-files=all)" ]] \
  || die "production checkout is not clean"

verify_committed_file() {
  local relative="$1"
  local path="${repo_root}/${relative}"
  [[ -f "${path}" && ! -L "${path}" ]] || die "missing tracked Gate-1 file ${relative}"
  git -C "${repo_root}" cat-file -e "${commit}:${relative}" \
    || die "Gate-1 file is absent from submitted commit: ${relative}"
  git -C "${repo_root}" show "${commit}:${relative}" \
    | cmp --silent -- - "${path}" \
    || die "Gate-1 working file differs from submitted commit: ${relative}"
}
verify_committed_file "${SBATCH_RELATIVE}"
verify_committed_file "${SUBMIT_RELATIVE}"
verify_committed_file "${CAPTURE_RELATIVE}"

first_line="$(sed -n '1p' -- "${repo_root}/${SBATCH_RELATIVE}")"
second_line="$(sed -n '2p' -- "${repo_root}/${SBATCH_RELATIVE}")"
[[ "${first_line}" == '#!/usr/bin/env bash' ]] \
  || die "Gate-1 SBATCH shebang is not one complete first line"
[[ "${second_line}" != "bash" ]] \
  || die "Gate-1 SBATCH contains the historical split-shebang failure"

[[ "${PRORM_R3_GATE1_IMAGE_SHA256}" == "${FROZEN_IMAGE_SHA256}" ]] \
  || die "caller image SHA-256 differs from the frozen Gate-1 image"
printf '%s  %s\n' "${FROZEN_IMAGE_SHA256}" "${image}" \
  | sha256sum --check --status \
  || die "Gate-1 image SHA-256 mismatch"

source_receipt="${project_root}/${SOURCE_RECEIPT_RELATIVE}"
gate1_artifact="${project_root}/${GATE1_RELATIVE}"
[[ ! -e "${source_receipt}" && ! -L "${source_receipt}" ]] \
  || die "source-test receipt already exists; Gate-1 is no-overwrite"
[[ ! -e "${gate1_artifact}" && ! -L "${gate1_artifact}" ]] \
  || die "Gate-1 artifact already exists; Gate-1 is no-overwrite"

export PATH="/usr/bin:/bin"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
pycache_parent="${SLURM_TMPDIR:-/tmp}"
host_pycache="${pycache_parent}/prorm-r3-gate1-${SLURM_JOB_ID}-host-pyc"
container_pycache="/tmp/prorm-r3-gate1-${SLURM_JOB_ID}-sif-pyc"
export PYTHONPYCACHEPREFIX="${host_pycache}"
mkdir -p -- "${host_pycache}"

capture_cli="${repo_root}/${CAPTURE_RELATIVE}"
container_path="${scratch_tools}/bin:/opt/conda/bin:/usr/local/bin:/usr/bin:/bin"
container_pythonpath="${repo_root}/src:${scratch_tools}"
container_python=(
  "${APPTAINER}"
  exec
  --cleanenv
  --containall
  --nv
  --bind "${repo_root}:${repo_root}:ro"
  --bind "${project_root}:${project_root}:rw"
  --bind "${scratch_tools}:${scratch_tools}:ro"
  --env "PATH=${container_path}"
  --env "PYTHONPATH=${container_pythonpath}"
  --env "PYTHONNOUSERSITE=1"
  --env "PYTHONDONTWRITEBYTECODE=1"
  --env "PYTHONUNBUFFERED=1"
  --env "PYTHONPYCACHEPREFIX=${container_pycache}"
  --pwd "${repo_root}"
  "${image}"
  "${SIF_PYTHON}"
)
[[ "$("${container_python[@]}" --version 2>&1)" == "Python 3.11.13" ]] \
  || die "frozen SIF Python version mismatch"
[[ "$("${container_python[@]}" -m pytest --version)" == "pytest 7.4.4" ]] \
  || die "bound Gate-1 pytest version mismatch"

printf 'R3 Gate-1 source-test start: commit=%s job=%s\n' \
  "${commit}" "${SLURM_JOB_ID}"
"${container_python[@]}" "${capture_cli}" source-test

[[ -f "${source_receipt}" && ! -L "${source_receipt}" ]] \
  || die "source-test receipt was not published"
[[ "$(stat -c '%a' -- "${source_receipt}")" == "440" ]] \
  || die "source-test receipt mode must be 0440"
source_receipt_sha256="$(sha256sum -- "${source_receipt}")"
source_receipt_sha256="${source_receipt_sha256%% *}"
[[ "${source_receipt_sha256}" =~ ^[0-9a-f]{64}$ ]] \
  || die "invalid source-test receipt SHA-256"

printf 'R3 Gate-1 live container capture start: source_receipt_sha256=%s\n' \
  "${source_receipt_sha256}"
export PATH="/usr/bin:/bin"
export PYTHONPATH="${repo_root}/src"
"${host_python}" "${capture_cli}" capture-live \
  --container "${image}" \
  --source-test-receipt-file-sha256 "${source_receipt_sha256}"

[[ -f "${gate1_artifact}" && ! -L "${gate1_artifact}" ]] \
  || die "Gate-1 artifact was not published"
[[ "$(stat -c '%a' -- "${gate1_artifact}")" == "440" ]] \
  || die "Gate-1 artifact mode must be 0440"
gate1_file_sha256="$(sha256sum -- "${gate1_artifact}")"
gate1_file_sha256="${gate1_file_sha256%% *}"
[[ "${gate1_file_sha256}" =~ ^[0-9a-f]{64}$ ]] \
  || die "invalid Gate-1 artifact SHA-256"

printf 'R3 Gate-1 live verification start: gate1_file_sha256=%s\n' \
  "${gate1_file_sha256}"
"${host_python}" "${capture_cli}" verify-live \
  --container "${image}" \
  --gate1-file-sha256 "${gate1_file_sha256}" \
  --source-test-receipt-file-sha256 "${source_receipt_sha256}"

printf 'R3_GATE1_JOB_ID=%s\n' "${SLURM_JOB_ID}"
printf 'R3_GATE1_GIT_COMMIT=%s\n' "${commit}"
printf 'R3_GATE1_SOURCE_TEST_RECEIPT_FILE_SHA256=%s\n' "${source_receipt_sha256}"
printf 'R3_GATE1_FILE_SHA256=%s\n' "${gate1_file_sha256}"
