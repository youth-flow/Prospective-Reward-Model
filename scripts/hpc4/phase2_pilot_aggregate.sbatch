#!/usr/bin/env bash
#SBATCH --job-name=prorm-phase2-pilot-aggregate
#SBATCH --account=sigroup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

set -euo pipefail
umask 027

die() {
  echo "error: $*" >&2
  exit 2
}

while IFS= read -r variable; do
  case "${variable}" in
    APPTAINER*|SINGULARITY*)
      die "unset exported ${variable}; pilot aggregation forbids ambient container controls"
      ;;
  esac
done < <(compgen -e)

for name in \
  PRORM_PROJECT_ROOT PRORM_SCRATCH_ROOT PRORM_IMAGE PRORM_IMAGE_SHA256 \
  PRORM_HF_CACHE PRORM_HF_INVENTORY PRORM_HF_INVENTORY_SHA256 \
  PRORM_REPO_ROOT PRORM_PHASE2_OVERLAY_REL PRORM_PHASE2_BASE_REL \
  PRORM_PHASE2_OVERLAY_FILE_SHA256 PRORM_PHASE2_BASE_FILE_SHA256 \
  PRORM_IDENTITIES_FILE_SHA256 PRORM_PHASE2_DESIGN_SHA256 \
  PRORM_PHASE2_BASE_CONFIG_HASH PRORM_GIT_COMMIT \
  PRORM_PHASE2_AGGREGATE_OUTPUT PRORM_PHASE2_BETA_SOURCE_AGGREGATE_PRESENT \
  PRORM_PHASE2_HORIZON_PARENT_AGGREGATE_PRESENT \
  PRORM_PHASE2_RUN_DIR_0 PRORM_PHASE2_RESULT_SHA256_0 \
  PRORM_PHASE2_SIDECAR_SHA256_0 PRORM_PHASE2_SUCCESS_SHA256_0 \
  PRORM_PHASE2_RUN_DIR_1 PRORM_PHASE2_RESULT_SHA256_1 \
  PRORM_PHASE2_SIDECAR_SHA256_1 PRORM_PHASE2_SUCCESS_SHA256_1 \
  PRORM_PHASE2_RUN_DIR_2 PRORM_PHASE2_RESULT_SHA256_2 \
  PRORM_PHASE2_SIDECAR_SHA256_2 PRORM_PHASE2_SUCCESS_SHA256_2; do
  [[ -n "${!name:-}" ]] || die "${name} is required"
done

: "${SLURM_JOB_ID:?pilot aggregation must execute inside Slurm}"
[[ "${SLURM_JOB_ID}" =~ ^[1-9][0-9]*$ ]] || die "SLURM_JOB_ID must be positive"
[[ "${SLURM_JOB_ACCOUNT:-}" = "sigroup" ]] \
  || die "pilot aggregation requires account sigroup"
case "${SLURM_JOB_PARTITION:-}" in
  amd|intel) ;;
  *) die "pilot aggregation requires CPU partition amd or intel" ;;
esac
[[ -z "${SLURM_ARRAY_JOB_ID:-}" && -z "${SLURM_ARRAY_TASK_ID:-}" ]] \
  || die "pilot aggregation must be a single non-array job"
[[ "${SLURM_NNODES:-}" = "1" && "${SLURM_NTASKS:-}" = "1" ]] \
  || die "pilot aggregation requires exactly one node and task"
for variable in \
  SLURM_JOB_GPUS SLURM_STEP_GPUS CUDA_VISIBLE_DEVICES \
  NVIDIA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL; do
  [[ -z "${!variable:-}" ]] || die "pilot aggregation must not receive GPU visibility"
done
for variable in SLURM_JOB_GRES SLURM_STEP_GRES SLURM_TRES_PER_JOB SLURM_TRES_PER_NODE; do
  gres_value="${!variable:-}"
  [[ "${gres_value,,}" != *gpu* ]] || die "pilot aggregation must not receive GPU GRES"
done

for command_name in \
  apptainer git python3 realpath sha256sum awk grep mktemp rm mv tee \
  mkdir dirname basename; do
  command -v "${command_name}" >/dev/null 2>&1 \
    || die "required command is unavailable: ${command_name}"
done

require_directory() {
  local name="$1" raw="${!1}" resolved=""
  [[ "${raw}" = /* ]] || die "${name} must be absolute"
  resolved="$(realpath -e -- "${raw}")" || die "${name} cannot be resolved"
  [[ -d "${resolved}" && "${resolved}" != "/" && "${resolved}" = "${raw}" ]] \
    || die "${name} must be a canonical non-root directory"
}

require_file() {
  local name="$1" raw="${!1}" resolved=""
  [[ "${raw}" = /* ]] || die "${name} must be absolute"
  resolved="$(realpath -e -- "${raw}")" || die "${name} cannot be resolved"
  [[ -f "${resolved}" && ! -L "${raw}" && "${resolved}" = "${raw}" ]] \
    || die "${name} must be a canonical regular non-symlink file"
}

contained() {
  local name="$1" child="$2" parent="$3"
  case "${child}" in
    "${parent}"/*) ;;
    *) die "${name} escaped its required root" ;;
  esac
}

reject_delimiters() {
  local name="$1" value="$2"
  [[ "${value}" != *","* && "${value}" != *":"* && "${value}" != *"="* \
    && "${value}" != *$'\n'* && "${value}" != *$'\r'* ]] \
    || die "${name} contains an unsafe bind/export delimiter"
}

require_directory PRORM_PROJECT_ROOT
require_directory PRORM_SCRATCH_ROOT
require_directory PRORM_HF_CACHE
require_directory PRORM_REPO_ROOT
require_file PRORM_IMAGE
require_file PRORM_HF_INVENTORY
contained PRORM_IMAGE "${PRORM_IMAGE}" "${PRORM_PROJECT_ROOT}"
contained PRORM_HF_CACHE "${PRORM_HF_CACHE}" "${PRORM_PROJECT_ROOT}"
contained PRORM_HF_INVENTORY "${PRORM_HF_INVENTORY}" "${PRORM_HF_CACHE}"
case "${PRORM_PROJECT_ROOT}" in
  "${PRORM_SCRATCH_ROOT}"|"${PRORM_SCRATCH_ROOT}"/*) die "project and scratch roots overlap" ;;
esac
case "${PRORM_SCRATCH_ROOT}" in
  "${PRORM_PROJECT_ROOT}"|"${PRORM_PROJECT_ROOT}"/*) die "project and scratch roots overlap" ;;
esac
[[ -w "${PRORM_PROJECT_ROOT}" && -w "${PRORM_SCRATCH_ROOT}" ]] \
  || die "project and scratch roots must be writable"

[[ "${PRORM_PHASE2_OVERLAY_REL}" = "configs/common_beta_pilot.yaml" ]] \
  || die "submitted pilot overlay path is invalid"
[[ "${PRORM_PHASE2_BASE_REL}" = "configs/common_beta_pilot_base.yaml" ]] \
  || die "submitted pilot base path is invalid"
for name in \
  PRORM_PROJECT_ROOT PRORM_SCRATCH_ROOT PRORM_IMAGE PRORM_HF_CACHE \
  PRORM_HF_INVENTORY PRORM_REPO_ROOT PRORM_PHASE2_AGGREGATE_OUTPUT; do
  reject_delimiters "${name}" "${!name}"
done
for name in \
  PRORM_IMAGE_SHA256 PRORM_HF_INVENTORY_SHA256 \
  PRORM_PHASE2_OVERLAY_FILE_SHA256 PRORM_PHASE2_BASE_FILE_SHA256 \
  PRORM_IDENTITIES_FILE_SHA256 PRORM_PHASE2_DESIGN_SHA256 \
  PRORM_PHASE2_BASE_CONFIG_HASH \
  PRORM_PHASE2_RESULT_SHA256_0 PRORM_PHASE2_SIDECAR_SHA256_0 \
  PRORM_PHASE2_SUCCESS_SHA256_0 PRORM_PHASE2_RESULT_SHA256_1 \
  PRORM_PHASE2_SIDECAR_SHA256_1 PRORM_PHASE2_SUCCESS_SHA256_1 \
  PRORM_PHASE2_RESULT_SHA256_2 PRORM_PHASE2_SIDECAR_SHA256_2 \
  PRORM_PHASE2_SUCCESS_SHA256_2; do
  [[ "${!name}" =~ ^[0-9a-f]{64}$ ]] || die "${name} must be lowercase SHA256"
done
[[ "${PRORM_GIT_COMMIT}" =~ ^[0-9a-f]{40,64}$ ]] || die "invalid Git commit"
[[ "${PRORM_HF_INVENTORY}" = \
  "${PRORM_HF_CACHE}/inventories/${PRORM_PHASE2_BASE_CONFIG_HASH}.json" ]] \
  || die "HF inventory is not addressed by the base identity"
printf '%s  %s\n' "${PRORM_IMAGE_SHA256}" "${PRORM_IMAGE}" \
  | sha256sum --check --status || die "image SHA256 mismatch"
printf '%s  %s\n' "${PRORM_HF_INVENTORY_SHA256}" "${PRORM_HF_INVENTORY}" \
  | sha256sum --check --status || die "HF inventory SHA256 mismatch"

validate_optional_aggregate() {
  local present_name="$1" path_name="$2" digest_name="$3" label="$4"
  local present="${!present_name}" path="${!path_name:-}" digest="${!digest_name:-}"
  case "${present}" in
    0)
      [[ -z "${path}" && -z "${digest}" ]] || die "${label} exported while absent"
      ;;
    1)
      [[ -n "${path}" && -n "${digest}" ]] || die "${label} path/hash is missing"
      require_file "${path_name}"
      contained "${path_name}" "${path}" "${PRORM_PROJECT_ROOT}"
      reject_delimiters "${path_name}" "${path}"
      [[ "${digest}" =~ ^[0-9a-f]{64}$ ]] || die "${label} digest is invalid"
      printf '%s  %s\n' "${digest}" "${path}" \
        | sha256sum --check --status || die "${label} SHA256 mismatch"
      ;;
    *) die "${present_name} must equal 0 or 1" ;;
  esac
}

validate_optional_aggregate \
  PRORM_PHASE2_BETA_SOURCE_AGGREGATE_PRESENT \
  PRORM_PHASE2_BETA_SOURCE_AGGREGATE \
  PRORM_PHASE2_BETA_SOURCE_AGGREGATE_SHA256 \
  "beta-source aggregate"
validate_optional_aggregate \
  PRORM_PHASE2_HORIZON_PARENT_AGGREGATE_PRESENT \
  PRORM_PHASE2_HORIZON_PARENT_AGGREGATE \
  PRORM_PHASE2_HORIZON_PARENT_AGGREGATE_SHA256 \
  "horizon-parent aggregate"

repo_head="$(git -C "${PRORM_REPO_ROOT}" rev-parse --verify HEAD)"
[[ "${repo_head}" = "${PRORM_GIT_COMMIT}" ]] || die "repository HEAD changed after submission"
[[ -z "$(git -C "${PRORM_REPO_ROOT}" status --porcelain --untracked-files=normal)" ]] \
  || die "pilot aggregate repository is no longer clean"

run_dirs=()
results=()
source_bindings=()
for index in 0 1 2; do
  run_name="PRORM_PHASE2_RUN_DIR_${index}"
  result_digest_name="PRORM_PHASE2_RESULT_SHA256_${index}"
  sidecar_digest_name="PRORM_PHASE2_SIDECAR_SHA256_${index}"
  success_digest_name="PRORM_PHASE2_SUCCESS_SHA256_${index}"
  run_dir="${!run_name}"
  require_directory "${run_name}"
  contained "${run_name}" "${run_dir}" "${PRORM_PROJECT_ROOT}"
  reject_delimiters "${run_name}" "${run_dir}"
  case "${run_dir}" in
    "${PRORM_PROJECT_ROOT}"/runs/phase2-pilot/"${PRORM_PHASE2_DESIGN_SHA256}"/*) ;;
    *) die "pilot run directory is outside the submitted design identity" ;;
  esac
  result="${run_dir}/phase2-pilot-diagnostics.json"
  sidecar="${run_dir}/phase2-pilot-diagnostics.diagnostics.jsonl"
  marker="${run_dir}/SUCCESS"
  for path in "${result}" "${sidecar}" "${marker}"; do
    [[ -f "${path}" && ! -L "${path}" ]] || die "pilot SUCCESS input is incomplete"
  done
  printf '%s  %s\n' "${!result_digest_name}" "${result}" \
    | sha256sum --check --status || die "pilot result SHA256 mismatch"
  printf '%s  %s\n' "${!sidecar_digest_name}" "${sidecar}" \
    | sha256sum --check --status || die "pilot sidecar SHA256 mismatch"
  printf '%s  %s\n' "${!success_digest_name}" "${marker}" \
    | sha256sum --check --status || die "pilot SUCCESS marker SHA256 mismatch"
  grep -Fx 'status=SUCCESS' "${marker}" >/dev/null || die "pilot marker is not SUCCESS"
  grep -Fx "phase2_design_sha256=${PRORM_PHASE2_DESIGN_SHA256}" "${marker}" >/dev/null \
    || die "pilot SUCCESS design identity mismatch"
  grep -Fx "base_config_hash=${PRORM_PHASE2_BASE_CONFIG_HASH}" "${marker}" >/dev/null \
    || die "pilot SUCCESS base identity mismatch"
  grep -Fx "git_commit=${PRORM_GIT_COMMIT}" "${marker}" >/dev/null \
    || die "pilot SUCCESS Git identity mismatch"
  run_dirs+=("${run_dir}")
  results+=("${result}")
  source_bindings+=(
    "${result}:${!result_digest_name}"
    "${sidecar}:${!sidecar_digest_name}"
    "${marker}:${!success_digest_name}"
  )
done
[[ "${run_dirs[0]}" != "${run_dirs[1]}" \
  && "${run_dirs[0]}" != "${run_dirs[2]}" \
  && "${run_dirs[1]}" != "${run_dirs[2]}" ]] \
  || die "pilot aggregate inputs must be three distinct SUCCESS runs"

output_parent="$(realpath -e -- "$(dirname "${PRORM_PHASE2_AGGREGATE_OUTPUT}")")" \
  || die "pilot aggregate output parent is missing"
output_name="$(basename "${PRORM_PHASE2_AGGREGATE_OUTPUT}")"
[[ -n "${output_name}" && "${output_name}" != "." && "${output_name}" != ".." ]] \
  || die "pilot aggregate output basename is invalid"
output="${output_parent}/${output_name}"
[[ "${output}" = "${PRORM_PHASE2_AGGREGATE_OUTPUT}" ]] \
  || die "pilot aggregate output path is not canonical"
contained PRORM_PHASE2_AGGREGATE_OUTPUT "${output}" "${PRORM_PROJECT_ROOT}"
[[ ! -e "${output}" && ! -L "${output}" ]] \
  || die "refusing to overwrite pilot aggregate"
[[ -w "${output_parent}" ]] || die "pilot aggregate output parent is not writable"

job_dir="${PRORM_SCRATCH_ROOT}/phase2-pilot-aggregate-jobs/${SLURM_JOB_ID}"
[[ ! -e "${job_dir}" && ! -L "${job_dir}" ]] \
  || die "refusing to reuse pilot aggregate scratch directory"
mkdir -p "${PRORM_SCRATCH_ROOT}/phase2-pilot-aggregate-jobs"
mkdir "${job_dir}"

execution_repo="${job_dir}/source"
git clone --quiet --no-hardlinks --no-checkout \
  "${PRORM_REPO_ROOT}" "${execution_repo}"
git -C "${execution_repo}" checkout --quiet --detach "${PRORM_GIT_COMMIT}"
execution_overlay="${execution_repo}/${PRORM_PHASE2_OVERLAY_REL}"
execution_base="${execution_repo}/${PRORM_PHASE2_BASE_REL}"
execution_identities="${execution_repo}/configs/identities.json"
for binding in \
  "${execution_overlay}:${PRORM_PHASE2_OVERLAY_FILE_SHA256}" \
  "${execution_base}:${PRORM_PHASE2_BASE_FILE_SHA256}" \
  "${execution_identities}:${PRORM_IDENTITIES_FILE_SHA256}"; do
  path="${binding%%:*}"
  expected="${binding#*:}"
  [[ -f "${path}" && ! -L "${path}" ]] || die "detached identity input is missing"
  printf '%s  %s\n' "${expected}" "${path}" \
    | sha256sum --check --status || die "detached identity input differs from submission"
done

assert_execution_checkout() {
  [[ "$(git -C "${execution_repo}" rev-parse --verify HEAD)" = "${PRORM_GIT_COMMIT}" \
    && -z "$(git -C "${execution_repo}" status --porcelain --untracked-files=normal)" ]] \
    || die "detached pilot aggregate checkout changed"
}
assert_execution_checkout

container=(
  apptainer exec --cleanenv
  --no-mount home,cwd,bind-paths
  --pwd "${execution_repo}"
  --bind "${job_dir}:${job_dir},${PRORM_PROJECT_ROOT}:${PRORM_PROJECT_ROOT}"
  --env "PYTHONPATH=${execution_repo}/src"
  --env "PYTHONNOUSERSITE=1"
  --env "PYTHONDONTWRITEBYTECODE=1"
  --env "HF_HOME=${PRORM_HF_CACHE}"
  --env "HF_HUB_OFFLINE=1"
  --env "TRANSFORMERS_OFFLINE=1"
  --env "HF_DATASETS_OFFLINE=1"
  "${PRORM_IMAGE}"
)

phase2_check="${job_dir}/phase2-config-check.json"
base_check="${job_dir}/base-config-check.json"
"${container[@]}" python -m smart_reward.cli phase2-config-check \
  "${execution_overlay}" | tee "${phase2_check}"
"${container[@]}" python -m smart_reward.cli config-check \
  "${execution_base}" | tee "${base_check}"
python3 -I -S - \
  "${phase2_check}" "${base_check}" \
  "${PRORM_PHASE2_DESIGN_SHA256}" "${PRORM_PHASE2_BASE_CONFIG_HASH}" <<'PY'
import json
import sys
from pathlib import Path


phase2 = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
base = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
if (
    phase2.get("status") != "ok"
    or phase2.get("phase2_design_sha256") != sys.argv[3]
    or phase2.get("source_config_hash") != sys.argv[4]
    or base.get("status") != "ok"
    or base.get("config_hash") != sys.argv[4]
):
    raise SystemExit("container semantic identities differ from submission")
PY

staged_output="$(mktemp "${output_parent}/.$(basename "${output}").staged.XXXXXX")"
rm -f -- "${staged_output}"
[[ ! -e "${staged_output}" && ! -L "${staged_output}" ]] \
  || die "cannot reserve staged pilot aggregate path"
aggregate_flags=()
if (( PRORM_PHASE2_BETA_SOURCE_AGGREGATE_PRESENT )); then
  aggregate_flags+=(
    --beta-source-aggregate "${PRORM_PHASE2_BETA_SOURCE_AGGREGATE}"
  )
fi
if (( PRORM_PHASE2_HORIZON_PARENT_AGGREGATE_PRESENT )); then
  aggregate_flags+=(
    --horizon-parent-aggregate "${PRORM_PHASE2_HORIZON_PARENT_AGGREGATE}"
  )
fi
assert_execution_checkout
"${container[@]}" python -m smart_reward.cli phase2-pilot-aggregate \
  "${execution_overlay}" "${staged_output}" "${results[@]}" \
  "${aggregate_flags[@]}" | tee "${job_dir}/phase2-pilot-aggregate.log"

python3 -I -S - \
  "${staged_output}" "${PRORM_PHASE2_DESIGN_SHA256}" \
  "${PRORM_PHASE2_BASE_CONFIG_HASH}" "${PRORM_GIT_COMMIT}" \
  "${PRORM_IMAGE_SHA256}" "${PRORM_HF_INVENTORY_SHA256}" <<'PY'
import json
import sys
from pathlib import Path


value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
environment = value.get("environment_identity")
boundary = value.get("information_boundary")
if (
    value.get("schema_version") != "common-beta-pilot-selection-aggregate/v1"
    or value.get("phase2_design_sha256") != sys.argv[2]
    or value.get("source_config_hash") != sys.argv[3]
    or not isinstance(value.get("sources"), list)
    or len(value["sources"]) != 3
    or not isinstance(environment, dict)
    or environment.get("git_commit") != sys.argv[4]
    or environment.get("image_sha256") != sys.argv[5]
    or environment.get("hf_inventory_sha256") != sys.argv[6]
    or not isinstance(boundary, dict)
    or boundary.get("oracle_outcomes_consumed") is not False
    or boundary.get("formal_efficacy_evidence_produced") is not False
):
    raise SystemExit("staged pilot aggregate failed bound output verification")
PY

for binding in "${source_bindings[@]}"; do
  path="${binding%%:*}"
  expected="${binding#*:}"
  printf '%s  %s\n' "${expected}" "${path}" \
    | sha256sum --check --status || die "pilot aggregate input changed during execution"
done
if (( PRORM_PHASE2_BETA_SOURCE_AGGREGATE_PRESENT )); then
  printf '%s  %s\n' \
    "${PRORM_PHASE2_BETA_SOURCE_AGGREGATE_SHA256}" \
    "${PRORM_PHASE2_BETA_SOURCE_AGGREGATE}" \
    | sha256sum --check --status || die "beta-source aggregate changed"
fi
if (( PRORM_PHASE2_HORIZON_PARENT_AGGREGATE_PRESENT )); then
  printf '%s  %s\n' \
    "${PRORM_PHASE2_HORIZON_PARENT_AGGREGATE_SHA256}" \
    "${PRORM_PHASE2_HORIZON_PARENT_AGGREGATE}" \
    | sha256sum --check --status || die "horizon-parent aggregate changed"
fi
printf '%s  %s\n' "${PRORM_IMAGE_SHA256}" "${PRORM_IMAGE}" \
  | sha256sum --check --status || die "image changed during pilot aggregation"
printf '%s  %s\n' "${PRORM_HF_INVENTORY_SHA256}" "${PRORM_HF_INVENTORY}" \
  | sha256sum --check --status || die "inventory changed during pilot aggregation"
assert_execution_checkout
[[ "$(git -C "${PRORM_REPO_ROOT}" rev-parse --verify HEAD)" = "${PRORM_GIT_COMMIT}" \
  && -z "$(git -C "${PRORM_REPO_ROOT}" status --porcelain --untracked-files=normal)" ]] \
  || die "control-plane checkout changed during pilot aggregation"
[[ ! -e "${output}" && ! -L "${output}" ]] \
  || die "pilot aggregate destination appeared before publication"
[[ -f "${staged_output}" && ! -L "${staged_output}" ]] \
  || die "staged pilot aggregate is not a regular non-symlink file"
staged_output_sha256="$(sha256sum -- "${staged_output}" | awk '{print $1}')"
[[ "${staged_output_sha256}" =~ ^[0-9a-f]{64}$ ]] \
  || die "failed to hash staged pilot aggregate"
if ! mv -T --no-clobber -- "${staged_output}" "${output}"; then
  die "atomic no-overwrite pilot aggregate publication failed"
fi
[[ ! -e "${staged_output}" && ! -L "${staged_output}" ]] \
  || die "pilot aggregate destination won a publication race; staged file was not installed"
[[ -f "${output}" && ! -L "${output}" ]] \
  || die "atomic publication did not install a regular pilot aggregate"
printf '%s  %s\n' "${staged_output_sha256}" "${output}" \
  | sha256sum --check --status \
  || die "published pilot aggregate differs from the staged payload"
printf 'Phase-2 pilot aggregate published: %s sha256=%s\n' \
  "${output}" "${staged_output_sha256}"
