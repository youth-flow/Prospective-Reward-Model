#!/usr/bin/env bash
# Submit the fixed SIF profile evidence finalizer after one profile array.

set -euo pipefail
umask 077

die() {
  printf 'R3 Gate-C profile-finalizer submit fatal: %s\n' "$*" >&2
  exit 1
}

[[ "$#" -eq 2 ]] || die "usage: $0 PROFILE_ARRAY_JOB_ID PROFILE_SUBMISSION_ROOT"
profile_array_job_id="$1"
requested_submission_root="$2"
[[ "${profile_array_job_id}" =~ ^[1-9][0-9]*$ ]] \
  || die "invalid profile array job ID"

repo_root="$(realpath -e -- "${PRORM_R3_REPO_ROOT:-/home/yyangjo/Smart-Reward-Model}")"
project_root="$(
  realpath -e -- \
    "${PRORM_R3_PROJECT_ROOT:-/project/sigroup/smart-reward-model}"
)"
image="$(realpath -e -- "${PRORM_R3_IMAGE:?missing PRORM_R3_IMAGE}")"
submission_root="$(realpath -e -- "${requested_submission_root}")"
controls_config="$(
  realpath -e -- "${repo_root}/configs/phase2_recovery_r3_controls.yaml"
)"
sbatch_script="$(
  realpath -e -- \
    "${repo_root}/scripts/hpc4/phase2_r3_controls_profile_finalize.sbatch"
)"
[[ "${repo_root}" == "/home/yyangjo/Smart-Reward-Model" ]] \
  || die "unexpected repository root"
[[ "${project_root}" == "/project/sigroup/smart-reward-model" ]] \
  || die "unexpected project root"
[[ "${submission_root}" == \
  "${project_root}/runs/phase2-recovery-r3-controls/profile-attempts/"* ]] \
  || die "profile attempt is outside the durable namespace"
[[ -f "${image}" && ! -L "${image}" && "${image}" == "${project_root}/"* ]] \
  || die "container must be a regular project file"

commit="$(git -C "${repo_root}" rev-parse HEAD)"
[[ "${commit}" =~ ^[0-9a-f]{40}$ ]] || die "invalid Git commit"
[[ -z "$(git -C "${repo_root}" status --porcelain --untracked-files=all)" ]] \
  || die "production checkout must be clean"
image_sha256="$(sha256sum -- "${image}" | cut -d' ' -f1)"
[[ "${image_sha256}" =~ ^[0-9a-f]{64}$ ]] || die "invalid container SHA-256"

mkdir -p -- "${submission_root}/finalize-logs"

submitted_count="$(squeue -r -h -u "${USER}" -o '%i' | sed '/^[[:space:]]*$/d' | wc -l)"
(( submitted_count <= 3 )) \
  || die "profile finalizer would exceed HPC4 MaxSubmitJobsPU=4"

export_spec="PATH=/usr/bin:/bin"
for item in \
  "PRORM_R3_GATEC_PROFILE_FINALIZE_REPO_ROOT=${repo_root}" \
  "PRORM_R3_GATEC_PROFILE_FINALIZE_PROJECT_ROOT=${project_root}" \
  "PRORM_R3_GATEC_PROFILE_FINALIZE_IMAGE=${image}" \
  "PRORM_R3_GATEC_PROFILE_FINALIZE_IMAGE_SHA256=${image_sha256}" \
  "PRORM_R3_GATEC_PROFILE_FINALIZE_GIT_COMMIT=${commit}" \
  "PRORM_R3_GATEC_PROFILE_FINALIZE_CONTROLS_CONFIG=${controls_config}" \
  "PRORM_R3_GATEC_PROFILE_FINALIZE_ARRAY_JOB_ID=${profile_array_job_id}" \
  "PRORM_R3_GATEC_PROFILE_FINALIZE_SUBMISSION_ROOT=${submission_root}"; do
  [[ "${item}" != *","* ]] || die "export value contains a comma"
  export_spec+=",${item}"
done

job_id="$(
  sbatch --parsable \
    --dependency="afterany:${profile_array_job_id}" \
    --account=sigroup \
    --partition=gpu-l20 \
    --nodes=1 \
    --ntasks=1 \
    --gpus-per-node=1 \
    --cpus-per-task=8 \
    --mem=96G \
    --time=0-01:00:00 \
    --no-requeue \
    --job-name=prorm-r3-gatec-profile-finalize \
    --output="${submission_root}/finalize-logs/%j.out" \
    --error="${submission_root}/finalize-logs/%j.err" \
    --export="${export_spec}" \
    "${sbatch_script}"
)"
[[ "${job_id}" =~ ^[1-9][0-9]*$ ]] || die "sbatch returned an invalid job ID"
printf 'R3_GATEC_PROFILE_FINALIZER_JOB_ID=%s\n' "${job_id}"
printf 'R3_GATEC_PROFILE=%s\n' "${submission_root}/operational-profile.json"
printf 'R3_GATEC_PLAN=%s\n' "${submission_root}/execution-plan.json"
