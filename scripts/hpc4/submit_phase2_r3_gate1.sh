#!/usr/bin/env bash
# Submit the tracked Gate-1 SBATCH body; never synthesize a wrapper script.

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

die() {
  printf 'R3 Gate-1 submit fatal: %s\n' "$*" >&2
  exit 1
}

for command_name in chmod cmp env git mkdir realpath sbatch sed sha256sum stat; do
  command -v "${command_name}" >/dev/null 2>&1 \
    || die "${command_name} is unavailable"
done

while IFS='=' read -r name _; do
  case "${name}" in
    SBATCH_*) die "unset exported ${name}; ambient sbatch overrides are forbidden" ;;
  esac
done < <(env)

[[ -n "${PRORM_R3_IMAGE:-}" ]] || die "missing PRORM_R3_IMAGE"
[[ -n "${PRORM_R3_IMAGE_SHA256:-}" ]] || die "missing PRORM_R3_IMAGE_SHA256"

repo_root="$(realpath -e -- "${REPO_ROOT}")"
project_root="$(realpath -e -- "${PROJECT_ROOT}")"
image="$(realpath -e -- "${PRORM_R3_IMAGE}")"
scratch_tools="$(realpath -e -- "${SCRATCH_TOOLS}")"
ruff_path="${scratch_tools}/bin/ruff"
host_python="${HOST_PYTHON}"
host_python_target="$(realpath -e -- "${host_python}")"
[[ "${repo_root}" == "${REPO_ROOT}" ]] || die "production repository root is not canonical"
[[ "${project_root}" == "${PROJECT_ROOT}" ]] || die "project root is not canonical"
[[ -e "${repo_root}/.git" && ! -L "${repo_root}/.git" ]] \
  || die "production repository is not a real Git checkout"
[[ ! -e "${project_root}/.git" && ! -L "${project_root}/.git" ]] \
  || die "persistent project root must not be a Git checkout"
[[ "${PRORM_R3_IMAGE}" == "${image}" ]] \
  || die "Gate-1 image path must already be canonical"
[[ -f "${image}" && ! -L "${image}" && "${image}" == "${project_root}/"* ]] \
  || die "Gate-1 image must be a non-symlink project SIF"
[[ "${scratch_tools}" == "${SCRATCH_TOOLS}" ]] \
  || die "Gate-1 scratch tools path drifted"
[[ -d "${scratch_tools}" && ! -L "${scratch_tools}" ]] \
  || die "Gate-1 scratch tools must be one canonical real directory"
[[ -f "${ruff_path}" && -x "${ruff_path}" && ! -L "${ruff_path}" ]] \
  || die "Gate-1 Ruff must be the fixed scratch executable"
printf '%s  %s\n' "${RUFF_SHA256}" "${ruff_path}" \
  | sha256sum --check --status \
  || die "Gate-1 Ruff SHA-256 mismatch"
[[ "$("${ruff_path}" --version)" == "ruff 0.15.22" ]] \
  || die "Gate-1 Ruff version mismatch"
[[ -f "${host_python}" && -x "${host_python}" ]] \
  || die "fixed Gate-1 host Python launcher is unavailable"
[[ -f "${host_python_target}" && -x "${host_python_target}" && ! -L "${host_python_target}" ]] \
  || die "fixed Gate-1 host Python target is unavailable"
printf '%s  %s\n' "${HOST_PYTHON_SHA256}" "${host_python}" \
  | sha256sum --check --status \
  || die "Gate-1 host Python SHA-256 mismatch"
[[ "$("${host_python}" --version 2>&1)" == "Python 3.12.2" ]] \
  || die "Gate-1 host Python version mismatch"
[[ -f "${APPTAINER}" && -x "${APPTAINER}" && ! -L "${APPTAINER}" ]] \
  || die "Gate-1 requires the fixed host Apptainer executable"

commit="$(git -C "${repo_root}" rev-parse HEAD)"
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

sbatch_script="${repo_root}/${SBATCH_RELATIVE}"
first_line="$(sed -n '1p' -- "${sbatch_script}")"
second_line="$(sed -n '2p' -- "${sbatch_script}")"
[[ "${first_line}" == '#!/usr/bin/env bash' ]] \
  || die "Gate-1 SBATCH shebang is not one complete first line"
[[ "${second_line}" != "bash" ]] \
  || die "Gate-1 SBATCH contains the historical split-shebang failure"

[[ "${PRORM_R3_IMAGE_SHA256}" == "${FROZEN_IMAGE_SHA256}" ]] \
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

ensure_real_directory() {
  local label="$1"
  local path="$2"
  local mode_policy="$3"
  local resolved=""
  local mode=""
  if [[ ! -e "${path}" && ! -L "${path}" ]]; then
    mkdir -m 0750 -- "${path}" || die "failed to create ${label}: ${path}"
  fi
  [[ -d "${path}" && ! -L "${path}" ]] \
    || die "${label} is not a non-symlink directory: ${path}"
  resolved="$(realpath -e -- "${path}")"
  [[ "${resolved}" == "${path}" ]] || die "${label} is not canonical: ${path}"
  if [[ "${mode_policy}" == "r3" ]]; then
    mode="$(stat -c '%a' -- "${path}")"
    if [[ "${mode}" != "750" && "${mode}" != "2750" ]]; then
      chmod 0750 -- "${path}" || die "failed to repair ${label} mode"
      mode="$(stat -c '%a' -- "${path}")"
    fi
    [[ "${mode}" == "750" || "${mode}" == "2750" ]] \
      || die "${label} must retain mode 0750 (optional setgid accepted)"
  fi
}

runs_root="${project_root}/runs"
r3_root="${runs_root}/phase2-recovery-r3"
gate1_root="${r3_root}/gate1"
logs="${gate1_root}/logs"
ensure_real_directory "project runs root" "${runs_root}" "shared"
ensure_real_directory "R3 output root" "${r3_root}" "r3"
ensure_real_directory "Gate-1 output root" "${gate1_root}" "r3"
ensure_real_directory "Gate-1 log directory" "${logs}" "r3"

export_spec="PATH=/usr/bin:/bin"
for assignment in \
  "PRORM_R3_GATE1_GIT_COMMIT=${commit}" \
  "PRORM_R3_GATE1_HOST_PYTHON=${host_python}" \
  "PRORM_R3_GATE1_HOST_PYTHON_MODULE=${HOST_PYTHON_MODULE}" \
  "PRORM_R3_GATE1_HOST_PYTHON_SHA256=${HOST_PYTHON_SHA256}" \
  "PRORM_R3_GATE1_IMAGE=${image}" \
  "PRORM_R3_GATE1_IMAGE_SHA256=${FROZEN_IMAGE_SHA256}" \
  "PRORM_R3_GATE1_RUFF_SHA256=${RUFF_SHA256}" \
  "PRORM_R3_GATE1_SCRATCH_TOOLS=${scratch_tools}"; do
  [[ "${assignment}" != *","* && "${assignment}" != *$'\n'* ]] \
    || die "unsafe value in Gate-1 sbatch export"
  export_spec+=",${assignment}"
done

# The final positional argument is the tracked SBATCH file.  Wrapper mode and
# stdin are forbidden: Slurm must receive the exact committed shebang bytes.
job_id="$(
  sbatch \
    --parsable \
    --export="${export_spec}" \
    "${sbatch_script}"
)"
[[ "${job_id}" =~ ^[1-9][0-9]*$ ]] || die "sbatch returned an invalid job ID"
printf 'R3_GATE1_JOB_ID=%s\n' "${job_id}"
printf 'R3_GATE1_GIT_COMMIT=%s\n' "${commit}"
printf 'R3_GATE1_SBATCH_SCRIPT=%s\n' "${sbatch_script}"
printf 'R3_GATE1_HOST_PYTHON_MODULE=%s\n' "${HOST_PYTHON_MODULE}"
