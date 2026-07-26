#!/usr/bin/env bash
# Submit the one-time 3-family Gate-C throughput profile.

set -euo pipefail
umask 077

readonly HOST_PYTHON="/opt/shared/spack/local/linux-rocky9-x86_64_v4/gcc-11.4.1/miniconda3-24.3.0-quc3pyudmzikgo2r4qsyqpwnrvzpin63/bin/python3.12"
readonly HOST_PYTHON_SHA256="9c91f9aa231c61c6bf2eabb9b93ebc5a8269a4126a36125e9548d8853e32da9c"

die() {
  printf 'R3 Gate-C profile submit fatal: %s\n' "$*" >&2
  exit 1
}

[[ "$#" -eq 0 ]] || die "usage: $0"
repo_root="$(realpath -e -- "${PRORM_R3_REPO_ROOT:-/home/yyangjo/Smart-Reward-Model}")"
project_root="$(
  realpath -e -- \
    "${PRORM_R3_PROJECT_ROOT:-/project/sigroup/smart-reward-model}"
)"
image="$(realpath -e -- "${PRORM_R3_IMAGE:?missing PRORM_R3_IMAGE}")"
hf_cache="$(realpath -e -- "${PRORM_R3_HF_CACHE:?missing PRORM_R3_HF_CACHE}")"
controls_config="$(
  realpath -e -- "${repo_root}/configs/phase2_recovery_r3_controls.yaml"
)"
runner="${repo_root}/scripts/hpc4/run_phase2_r3_control_profile.py"
sbatch_script="${repo_root}/scripts/hpc4/phase2_r3_controls_profile.sbatch"
[[ "${repo_root}" == "/home/yyangjo/Smart-Reward-Model" ]] \
  || die "unexpected production repository root"
[[ "${project_root}" == "/project/sigroup/smart-reward-model" ]] \
  || die "unexpected persistent project root"
[[ -d "${hf_cache}" && "${hf_cache}" == "${project_root}/hf-cache" ]] \
  || die "Gate-C profile requires the fixed project HF cache"
[[ -f "${runner}" && -f "${sbatch_script}" ]] \
  || die "Gate-C profile execution surface is incomplete"
[[ -f "${image}" && ! -L "${image}" && "${image}" == "${project_root}/"* ]] \
  || die "container must be a regular file under the project root"

host_python="${HOST_PYTHON}"
host_python_target="$(realpath -e -- "${host_python}")"
[[ -f "${host_python}" && -x "${host_python}" && ! -L "${host_python}" ]] \
  || die "fixed host Python launcher is not a regular non-symlink executable"
[[ "${host_python_target}" == "${host_python}" ]] \
  || die "fixed host Python path is not canonical"
printf '%s  %s\n' "${HOST_PYTHON_SHA256}" "${host_python}" \
  | sha256sum --check --status || die "fixed host Python SHA-256 mismatch"
[[ "$("${host_python}" --version 2>&1)" == "Python 3.12.2" ]] \
  || die "fixed host Python version mismatch"

commit="$(git -C "${repo_root}" rev-parse HEAD)"
[[ "${commit}" =~ ^[0-9a-f]{40}$ ]] || die "invalid Git commit"
[[ -z "$(git -C "${repo_root}" status --porcelain --untracked-files=all)" ]] \
  || die "production checkout must be clean"
image_sha256="$(sha256sum -- "${image}" | cut -d' ' -f1)"
[[ "${image_sha256}" =~ ^[0-9a-f]{64}$ ]] || die "invalid container SHA-256"

input_parent="${project_root}/runs/phase2-recovery-r3/inputs"
mkdir -p -- "${input_parent}"
input_parent="$(realpath -e -- "${input_parent}")"
[[ "${input_parent}" == "${project_root}/runs/phase2-recovery-r3/inputs" ]] \
  || die "Gate-C profile input parent is not canonical"
input_root="${input_parent}/${commit}"
mkdir -p -- "${input_root}"
input_root="$(realpath -e -- "${input_root}")"
[[ "${input_root}" == "${input_parent}/${commit}" && ! -L "${input_root}" ]] \
  || die "Gate-C profile input namespace is invalid"

copy_no_overwrite_exact() {
  local source="$1"
  local destination="$2"
  if [[ -e "${destination}" || -L "${destination}" ]]; then
    [[ -f "${destination}" && ! -L "${destination}" ]] \
      || die "retained Gate-C profile input is not a regular file"
  else
    cp --no-clobber -- "${source}" "${destination}"
  fi
  cmp --silent -- "${source}" "${destination}" \
    || die "retained Gate-C profile input differs from the clean commit"
  chmod 0440 -- "${destination}"
  [[ "$(stat -c '%a' -- "${destination}")" == "440" ]] \
    || die "retained Gate-C profile input must have mode 0440"
}
copy_no_overwrite_exact \
  "${repo_root}/configs/phase2_recovery_r3_science.yaml" \
  "${input_root}/phase2_recovery_r3_science.yaml"
copy_no_overwrite_exact \
  "${repo_root}/configs/common_beta_pilot_base.yaml" \
  "${input_root}/common_beta_pilot_base.yaml"
copy_no_overwrite_exact \
  "${repo_root}/configs/phase2_recovery_parent_failures.json" \
  "${input_root}/phase2_recovery_parent_failures.json"
parent_registry_file_sha256="$(
  sha256sum -- "${input_root}/phase2_recovery_parent_failures.json" \
    | cut -d' ' -f1
)"
[[ "${parent_registry_file_sha256}" =~ ^[0-9a-f]{64}$ ]] \
  || die "invalid parent-registry SHA-256"

# The 12-hour allocation is only for measuring 100 disposable updates.  It is
# not a formal-family walltime source; the sole formal walltime is derived
# later by build_controls_operational_profile from the three terminalized
# measurements.
submitted_count="$(squeue -r -h -u "${USER}" -o '%i' | sed '/^[[:space:]]*$/d' | wc -l)"
running_count="$(
  squeue -r -h -u "${USER}" -t RUNNING,COMPLETING -o '%i' \
    | sed '/^[[:space:]]*$/d' | wc -l
)"
(( submitted_count <= 1 )) \
  || die "profile array would exceed HPC4 MaxSubmitJobsPU=4"
(( running_count <= 1 )) \
  || die "profile array would exceed HPC4 MaxJobsPU=2"

attempt="$(date -u +%Y%m%dT%H%M%SZ)"
submission_parent="${project_root}/runs/phase2-recovery-r3-controls"
mkdir -p -- "${submission_parent}"
submission_parent="$(realpath -e -- "${submission_parent}")"
profile_parent="${submission_parent}/profile-attempts"
mkdir -p -- "${profile_parent}"
profile_parent="$(realpath -e -- "${profile_parent}")"
submission_root="${profile_parent}/${commit}-${attempt}"
[[ ! -e "${submission_root}" && ! -L "${submission_root}" ]] \
  || die "Gate-C profile attempt namespace already exists"
mkdir -- "${submission_root}"
mkdir -- "${submission_root}/logs"
submission_root="$(realpath -e -- "${submission_root}")"

export_spec="PATH=/usr/bin:/bin"
for item in \
  "PRORM_R3_GATEC_PROFILE_REPO_ROOT=${repo_root}" \
  "PRORM_R3_GATEC_PROFILE_PROJECT_ROOT=${project_root}" \
  "PRORM_R3_GATEC_PROFILE_HF_CACHE=${hf_cache}" \
  "PRORM_R3_GATEC_PROFILE_IMAGE=${image}" \
  "PRORM_R3_GATEC_PROFILE_IMAGE_SHA256=${image_sha256}" \
  "PRORM_R3_GATEC_PROFILE_GIT_COMMIT=${commit}" \
  "PRORM_R3_GATEC_PROFILE_CONTROLS_CONFIG=${controls_config}" \
  "PRORM_R3_GATEC_PROFILE_PARENT_REGISTRY_FILE_SHA256=${parent_registry_file_sha256}" \
  "PRORM_R3_GATEC_PROFILE_SUBMISSION_ROOT=${submission_root}"; do
  [[ "${item}" != *","* ]] || die "export value contains a comma"
  export_spec+=",${item}"
done

job_id="$(
  sbatch --parsable \
    --account=sigroup \
    --partition=gpu-l20 \
    --nodes=1 \
    --ntasks=1 \
    --gpus-per-node=1 \
    --cpus-per-task=8 \
    --mem=96G \
    --time=0-12:00:00 \
    --no-requeue \
    --array=0-2%1 \
    --job-name=prorm-r3-gatec-profile \
    --output="${submission_root}/logs/%A_%a.out" \
    --error="${submission_root}/logs/%A_%a.err" \
    --export="${export_spec}" \
    "${sbatch_script}"
)"
[[ "${job_id}" =~ ^[1-9][0-9]*$ ]] || die "sbatch returned an invalid job ID"
printf 'R3_GATEC_PROFILE_ARRAY_JOB_ID=%s\n' "${job_id}"
printf 'R3_GATEC_PROFILE_SUBMISSION_ROOT=%s\n' "${submission_root}"
printf 'R3_GATEC_PROFILE_HOST_PYTHON=%s\n' "${host_python}"
printf 'R3_GATEC_PROFILE_FINALIZER_ENTRYPOINT=%s\n' \
  "${repo_root}/scripts/hpc4/submit_phase2_r3_controls_profile_finalize.sh"
printf 'R3_GATEC_PROFILE_NEXT=submit_fixed_compute_finalizer_after_profile_completion\n'
