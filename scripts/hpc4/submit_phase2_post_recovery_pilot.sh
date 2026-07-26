#!/usr/bin/env bash
set -euo pipefail

readonly HOST_PYTHON="/opt/shared/spack/local/linux-rocky9-x86_64_v4/gcc-11.4.1/miniconda3-24.3.0-quc3pyudmzikgo2r4qsyqpwnrvzpin63/bin/python3.12"
readonly HOST_PYTHON_SHA256="9c91f9aa231c61c6bf2eabb9b93ebc5a8269a4126a36125e9548d8853e32da9c"

die() { echo "error: $*" >&2; exit 2; }

if [[ $# -lt 3 ]]; then
  die "usage: $0 <overlay.yaml> <recovery-authorization.json> <walltime> [--beta-source-aggregate <json>] [--horizon-parent-aggregate <json>]"
fi
overlay_input="$1"
authorization_input="$2"
walltime="$3"
shift 3
beta_source_input=""
horizon_parent_input=""
while (( $# )); do
  case "$1" in
    --beta-source-aggregate)
      [[ $# -ge 2 && -z "${beta_source_input}" ]] \
        || die "--beta-source-aggregate requires one path"
      beta_source_input="$2"
      shift 2
      ;;
    --horizon-parent-aggregate)
      [[ $# -ge 2 && -z "${horizon_parent_input}" ]] \
        || die "--horizon-parent-aggregate requires one path"
      horizon_parent_input="$2"
      shift 2
      ;;
    *) die "unknown post-recovery pilot option: $1" ;;
  esac
done
[[ "${walltime}" =~ ^[0-9]+-[0-9]{2}:[0-9]{2}:[0-9]{2}$|^[0-9]{2}:[0-9]{2}:[0-9]{2}$ ]] \
  || die "walltime must be HH:MM:SS or D-HH:MM:SS"

while IFS= read -r variable; do
  case "${variable}" in
    APPTAINER*|SINGULARITY*|SBATCH_*)
      die "unset exported ${variable}; post-recovery submission forbids ambient controls"
      ;;
  esac
done < <(compgen -e)
for name in \
  PRORM_PROJECT_ROOT PRORM_SCRATCH_ROOT PRORM_IMAGE PRORM_IMAGE_SHA256 \
  PRORM_HF_CACHE; do
  [[ -n "${!name:-}" ]] || die "${name} is required"
done
for command_name in \
  git realpath sha256sum awk sbatch scontrol squeue sacct id; do
  command -v "${command_name}" >/dev/null 2>&1 \
    || die "required command is unavailable: ${command_name}"
done
host_python="$(realpath -e -- "${HOST_PYTHON}")" \
  || die "fixed host Python is unavailable"
[[ "${host_python}" = "${HOST_PYTHON}" && -f "${host_python}" \
  && -x "${host_python}" && ! -L "${host_python}" ]] \
  || die "fixed host Python must be a canonical non-symlink executable"
printf '%s  %s\n' "${HOST_PYTHON_SHA256}" "${host_python}" \
  | sha256sum --check --status || die "fixed host Python SHA256 mismatch"
[[ "$("${host_python}" --version 2>&1)" = "Python 3.12.2" ]] \
  || die "fixed host Python version mismatch"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
git_commit="$(git -C "${repo_root}" rev-parse --verify HEAD)"
[[ "${git_commit}" =~ ^[0-9a-f]{40,64}$ ]] || die "Git HEAD is not a full object ID"
[[ -z "$(git -C "${repo_root}" status --porcelain --untracked-files=normal)" ]] \
  || die "post-recovery submission requires a clean committed worktree"

canonical_root() {
  local name="$1" raw="${!1}" resolved
  [[ "${raw}" = /* ]] || die "${name} must be absolute"
  resolved="$(realpath -e -- "${raw}")" || die "${name} cannot be resolved"
  [[ -d "${resolved}" && ! -L "${raw}" && "${resolved}" != "/" \
    && "${resolved}" = "${raw}" ]] || die "${name} must be a canonical real directory"
  printf '%s\n' "${resolved}"
}
project_path() {
  local raw="$1" kind="$2" candidate resolved
  if [[ "${raw}" = /* ]]; then candidate="${raw}"; else candidate="${project_root}/${raw}"; fi
  resolved="$(realpath -e -- "${candidate}")" || die "project path cannot be resolved"
  case "${resolved}" in "${project_root}"/*) ;; *) die "project path escaped root" ;; esac
  case "${kind}" in
    file) [[ -f "${resolved}" && ! -L "${candidate}" ]] || die "project file is unsafe" ;;
    directory) [[ -d "${resolved}" && ! -L "${candidate}" ]] || die "project directory is unsafe" ;;
    *) die "invalid project path kind" ;;
  esac
  printf '%s\n' "${resolved}"
}
reject_delimiters() {
  [[ "$2" != *","* && "$2" != *":"* && "$2" != *"="* \
    && "$2" != *$'\n'* && "$2" != *$'\r'* ]] || die "$1 contains an unsafe delimiter"
}

project_root="$(canonical_root PRORM_PROJECT_ROOT)"
scratch_root="$(canonical_root PRORM_SCRATCH_ROOT)"
case "${project_root}" in "${scratch_root}"|"${scratch_root}"/*) die "roots overlap" ;; esac
case "${scratch_root}" in "${project_root}"|"${project_root}"/*) die "roots overlap" ;; esac
image="$(project_path "${PRORM_IMAGE}" file)"
hf_cache="$(project_path "${PRORM_HF_CACHE}" directory)"
authorization="$(project_path "${authorization_input}" file)"
authorization_sha256="$(sha256sum -- "${authorization}" | awk '{print $1}')"
inspector="${repo_root}/scripts/hpc4/inspect_phase2_post_recovery_stdlib.py"
[[ -f "${inspector}" && ! -L "${inspector}" ]] \
  || die "post-recovery stdlib inspector is missing or unsafe"
mapfile -t authorization_transport < <(
  "${host_python}" -I -S "${inspector}" authorization "${authorization}" \
    --expected-sha256 "${authorization_sha256}" \
    --project-root "${project_root}"
)
[[ "${#authorization_transport[@]}" -ge 3 \
  && "${authorization_transport[2]}" =~ ^[0-9]+$ \
  && "${#authorization_transport[@]}" = \
    "$((3 + authorization_transport[2]))" ]] \
  || die "authorization transport inspection returned an invalid binding set"
authorization_schema="${authorization_transport[0]}"
authorization_schedule_sha256="${authorization_transport[1]}"

overlay="$(realpath -e -- "${overlay_input}")" || die "overlay cannot be resolved"
[[ -f "${overlay}" && ! -L "${overlay}" ]] || die "overlay is unsafe"
case "${overlay}" in "${repo_root}"/configs/*.yaml) ;; *) die "overlay must be configs/*.yaml" ;; esac
overlay_relative="$(realpath --relative-to="${repo_root}" "${overlay}")"

beta_source_present=0
beta_source=""
beta_source_sha256=""
if [[ -n "${beta_source_input}" ]]; then
  beta_source="$(project_path "${beta_source_input}" file)"
  [[ "$(dirname "${beta_source}")" = "${project_root}/aggregates" ]] \
    || die "beta source must be a production aggregate"
  beta_source_sha256="$(sha256sum -- "${beta_source}" | awk '{print $1}')"
  beta_source_present=1
fi
horizon_parent_present=0
horizon_parent=""
horizon_parent_sha256=""
if [[ -n "${horizon_parent_input}" ]]; then
  horizon_parent="$(project_path "${horizon_parent_input}" file)"
  [[ "$(dirname "${horizon_parent}")" = "${project_root}/aggregates" ]] \
    || die "horizon parent must be a production aggregate"
  horizon_parent_sha256="$(sha256sum -- "${horizon_parent}" | awk '{print $1}')"
  horizon_parent_present=1
fi

mapfile -t identities < <(
  inspector_arguments=(
    overlay "${overlay}"
    --repo-root "${repo_root}"
    --authorization-schema "${authorization_schema}"
    --authorization-sha256 "${authorization_sha256}"
  )
  if (( beta_source_present )); then
    inspector_arguments+=(--beta-source-sha256 "${beta_source_sha256}")
  fi
  if (( horizon_parent_present )); then
    inspector_arguments+=(--horizon-parent-sha256 "${horizon_parent_sha256}")
  fi
  "${host_python}" -I -S "${inspector}" "${inspector_arguments[@]}"
)
[[ "${#identities[@]}" = 7 ]] || die "could not inspect post-recovery identity"
pilot_phase="${identities[0]}"
design_sha256="${identities[1]}"
base_hash="${identities[2]}"
base="$(realpath -e -- "${identities[3]}")" \
  || die "declared base config cannot be resolved"
base_relative="${identities[4]}"
schedule_sha256="${identities[5]}"
[[ "${schedule_sha256}" = "${authorization_schedule_sha256}" ]] \
  || die "overlay and authorization optimizer schedules differ"
namespace="${pilot_phase}"

case "${base}" in "${repo_root}"/configs/*.yaml) ;; *)
  die "declared base config must be a tracked configs/*.yaml file"
  ;;
esac
for relative in \
  "${overlay_relative}" "${base_relative}" \
  "src/smart_reward/phase2_post_recovery_control.py" \
  "src/smart_reward/phase2_post_recovery_output.py" \
  "src/smart_reward/phase2_pilot_aggregate.py" \
  "scripts/hpc4/inspect_phase2_post_recovery_stdlib.py" \
  "scripts/hpc4/submit_phase2_post_recovery_calibration.sh" \
  "scripts/hpc4/submit_phase2_post_recovery_pilot.sh" \
  "scripts/hpc4/submit_phase2_post_recovery_array_once.py" \
  "scripts/hpc4/validate_phase2_post_recovery_submission.py" \
  "scripts/hpc4/validate_phase2_recovery_authorization.py" \
  "scripts/hpc4/validate_phase2_post_recovery_output.py" \
  "scripts/hpc4/phase2_post_recovery_calibration.sbatch"; do
  git -C "${repo_root}" ls-files --error-unmatch -- "${relative}" >/dev/null \
    || die "required source is not tracked: ${relative}"
done
overlay_sha256="$(sha256sum -- "${overlay}" | awk '{print $1}')"
base_sha256="$(sha256sum -- "${base}" | awk '{print $1}')"
for binding in "${overlay_relative}:${overlay_sha256}" "${base_relative}:${base_sha256}"; do
  relative="${binding%%:*}"
  expected="${binding#*:}"
  observed="$(
    git -C "${repo_root}" cat-file blob "${git_commit}:${relative}" \
      | sha256sum | awk '{print $1}'
  )"
  [[ "${observed}" = "${expected}" ]] \
    || die "worktree config differs from submitted commit: ${relative}"
done

inventory="$(realpath -e -- "${hf_cache}/inventories/${base_hash}.json")" \
  || die "base-identity HF inventory is missing"
[[ -f "${inventory}" && ! -L "${inventory}" ]] || die "HF inventory is unsafe"
inventory_sha256="$(sha256sum -- "${inventory}" | awk '{print $1}')"
[[ "${PRORM_IMAGE_SHA256}" =~ ^[0-9a-f]{64}$ ]] \
  || die "PRORM_IMAGE_SHA256 must be lowercase SHA256"
printf '%s  %s\n' "${PRORM_IMAGE_SHA256}" "${image}" \
  | sha256sum --check --status || die "container image SHA256 mismatch"

for name in project_root scratch_root repo_root image hf_cache inventory authorization \
  overlay base beta_source horizon_parent; do
  [[ -z "${!name:-}" ]] || reject_delimiters "${name}" "${!name}"
done
for binding in \
  "${authorization}:${authorization_sha256}" "${inventory}:${inventory_sha256}" \
  "${image}:${PRORM_IMAGE_SHA256}" \
  "${beta_source:+${beta_source}:${beta_source_sha256}}" \
  "${horizon_parent:+${horizon_parent}:${horizon_parent_sha256}}"; do
  [[ -z "${binding}" ]] && continue
  path="${binding%%:*}"
  expected="${binding#*:}"
  printf '%s  %s\n' "${expected}" "${path}" \
    | sha256sum --check --status || die "submitted immutable input changed"
done
[[ "$(git -C "${repo_root}" rev-parse --verify HEAD)" = "${git_commit}" \
  && -z "$(git -C "${repo_root}" status --porcelain --untracked-files=normal)" ]] \
  || die "checkout changed during submission"

mkdir -p \
  "${project_root}/slurm-logs" \
  "${project_root}/runs/phase2-post-recovery-${namespace}" \
  "${project_root}/artifacts/phase2-post-recovery-${namespace}" \
  "${scratch_root}/phase2-post-recovery-${namespace}-jobs"

export_spec="PATH=/usr/local/bin:/usr/bin:/bin,PRORM_PROJECT_ROOT=${project_root},PRORM_SCRATCH_ROOT=${scratch_root},PRORM_REPO_ROOT=${repo_root},PRORM_IMAGE=${image},PRORM_IMAGE_SHA256=${PRORM_IMAGE_SHA256},PRORM_HF_CACHE=${hf_cache},PRORM_HF_INVENTORY=${inventory},PRORM_HF_INVENTORY_SHA256=${inventory_sha256},PRORM_POST_RECOVERY_OVERLAY_REL=${overlay_relative},PRORM_PHASE2_BASE_REL=${base_relative},PRORM_POST_RECOVERY_OVERLAY_SHA256=${overlay_sha256},PRORM_PHASE2_BASE_SHA256=${base_sha256},PRORM_POST_RECOVERY_DESIGN_SHA256=${design_sha256},PRORM_PHASE2_BASE_CONFIG_HASH=${base_hash},PRORM_RECOVERY_AUTHORIZATION=${authorization},PRORM_RECOVERY_AUTHORIZATION_SHA256=${authorization_sha256},PRORM_OPTIMIZER_SCHEDULE_SHA256=${schedule_sha256},PRORM_GIT_COMMIT=${git_commit},PRORM_POST_RECOVERY_PILOT_PHASE=${pilot_phase},PRORM_POST_RECOVERY_NAMESPACE=${namespace},PRORM_PHASE2_BETA_SOURCE_AGGREGATE_PRESENT=${beta_source_present},PRORM_PHASE2_HORIZON_PARENT_AGGREGATE_PRESENT=${horizon_parent_present}"
if (( beta_source_present )); then
  export_spec+=",PRORM_PHASE2_BETA_SOURCE_AGGREGATE=${beta_source},PRORM_PHASE2_BETA_SOURCE_AGGREGATE_SHA256=${beta_source_sha256}"
fi
if (( horizon_parent_present )); then
  export_spec+=",PRORM_PHASE2_HORIZON_PARENT_AGGREGATE=${horizon_parent},PRORM_PHASE2_HORIZON_PARENT_AGGREGATE_SHA256=${horizon_parent_sha256}"
fi
"${host_python}" -I -S \
  "${repo_root}/scripts/hpc4/submit_phase2_post_recovery_array_once.py" \
  --project-root "${project_root}" \
  --repo-root "${repo_root}" \
  --pilot-phase "${pilot_phase}" \
  --design-sha256 "${design_sha256}" \
  --base-config-hash "${base_hash}" \
  --authorization-sha256 "${authorization_sha256}" \
  --optimizer-schedule-sha256 "${schedule_sha256}" \
  --git-commit "${git_commit}" \
  --image-sha256 "${PRORM_IMAGE_SHA256}" \
  --inventory-sha256 "${inventory_sha256}" \
  --overlay-sha256 "${overlay_sha256}" \
  --base-file-sha256 "${base_sha256}" \
  --walltime "${walltime}" \
  --export-spec "${export_spec}" \
  --sbatch-script \
  "${repo_root}/scripts/hpc4/phase2_post_recovery_calibration.sbatch"
