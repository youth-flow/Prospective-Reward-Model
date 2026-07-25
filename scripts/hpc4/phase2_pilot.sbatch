#!/usr/bin/env bash
#SBATCH --job-name=prorm-phase2-pilot
#SBATCH --account=sigroup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1

set -euo pipefail
umask 027

die() {
  echo "error: $*" >&2
  exit 2
}

reject_ambient_controls() {
  local variable=""
  while IFS= read -r variable; do
    case "${variable}" in
      APPTAINER*|SINGULARITY*)
        die "unset exported ${variable}; Phase-2 execution forbids ambient container controls"
        ;;
    esac
  done < <(compgen -e)
}

require_canonical_directory() {
  local name="$1" raw="${!1}" resolved=""
  [[ "${raw}" = /* ]] || die "${name} must be absolute"
  resolved="$(realpath -e -- "${raw}")" \
    || die "${name} does not exist or cannot be resolved: ${raw}"
  [[ -d "${resolved}" && "${resolved}" != "/" ]] \
    || die "${name} must be a non-root directory"
  [[ "${resolved}" = "${raw}" ]] || die "${name} must be canonical: ${raw}"
}

require_canonical_file() {
  local name="$1" raw="${!1}" resolved=""
  [[ "${raw}" = /* ]] || die "${name} must be absolute"
  resolved="$(realpath -e -- "${raw}")" \
    || die "${name} does not exist or cannot be resolved: ${raw}"
  [[ -f "${resolved}" ]] || die "${name} must be a regular file: ${raw}"
  [[ "${resolved}" = "${raw}" ]] || die "${name} must be canonical: ${raw}"
}

require_contained() {
  local name="$1" child="$2" parent="$3"
  case "${child}" in
    "${parent}"/*) ;;
    *) die "${name} escaped its required root: ${child}" ;;
  esac
}

roots_overlap() {
  local first="$1" second="$2"
  case "${first}" in
    "${second}"|"${second}"/*) return 0 ;;
  esac
  case "${second}" in
    "${first}"|"${first}"/*) return 0 ;;
  esac
  return 1
}

reject_path_delimiters() {
  local name="$1" value="$2"
  [[ "${value}" != *":"* && "${value}" != *","* && "${value}" != *"="* \
    && "${value}" != *$'\n'* && "${value}" != *$'\r'* ]] \
    || die "${name} contains an unsafe bind/export delimiter"
}

for name in \
  PRORM_PROJECT_ROOT PRORM_SCRATCH_ROOT PRORM_IMAGE PRORM_IMAGE_SHA256 \
  PRORM_HF_CACHE PRORM_REPO_ROOT PRORM_PHASE2_OVERLAY \
  PRORM_PHASE2_BASE_CONFIG PRORM_PHASE2_DESIGN_SHA256 \
  PRORM_PHASE2_BASE_CONFIG_HASH PRORM_PHASE2_OVERLAY_FILE_SHA256 \
  PRORM_PHASE2_BASE_FILE_SHA256 PRORM_IDENTITIES_FILE_SHA256 \
  PRORM_GIT_COMMIT PRORM_HF_INVENTORY PRORM_HF_INVENTORY_SHA256 \
  PRORM_PHASE2_BETA_SOURCE_AGGREGATE_PRESENT \
  PRORM_PHASE2_HORIZON_PARENT_AGGREGATE_PRESENT; do
  [[ -n "${!name:-}" ]] || die "${name} is required"
done

: "${SLURM_JOB_ID:?Phase-2 pilot must execute inside a Slurm job}"
: "${SLURM_ARRAY_JOB_ID:?Phase-2 pilot must execute as a Slurm array}"
: "${SLURM_ARRAY_TASK_ID:?Phase-2 pilot requires a Slurm array task ID}"
[[ "${SLURM_JOB_ID}" =~ ^[1-9][0-9]*$ ]] || die "SLURM_JOB_ID must be positive"
[[ "${SLURM_ARRAY_JOB_ID}" =~ ^[1-9][0-9]*$ ]] \
  || die "SLURM_ARRAY_JOB_ID must be positive"
[[ "${SLURM_ARRAY_TASK_ID}" =~ ^(0|[1-9][0-9]*)$ ]] \
  || die "SLURM_ARRAY_TASK_ID must be a non-negative integer"
[[ "${SLURM_JOB_ACCOUNT:-}" = "sigroup" ]] \
  || die "Phase-2 pilot requires Slurm account sigroup"
case "${SLURM_JOB_PARTITION:-}" in
  gpu-l20) ;;
  *)
    die "Phase-2 design is locked to HPC4 gpu-l20; refusing compute partition"
    ;;
esac
[[ "${SLURM_NNODES:-}" = "1" ]] || die "Phase-2 pilot requires exactly one node"
[[ "${SLURM_NTASKS:-}" = "1" ]] || die "Phase-2 pilot requires exactly one task"
case "${SLURM_GPUS_ON_NODE:-}" in
  1|gpu:1) ;;
  *) die "Phase-2 pilot requires exactly one Slurm GPU on the node" ;;
esac
[[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != *,* ]] \
  || die "CUDA_VISIBLE_DEVICES must identify exactly one allocated GPU"
reject_ambient_controls

for command_name in \
  apptainer flock git python3 realpath sha256sum awk grep mktemp cp mv ln; do
  command -v "${command_name}" >/dev/null 2>&1 \
    || die "required command is unavailable: ${command_name}"
done

require_canonical_directory PRORM_PROJECT_ROOT
require_canonical_directory PRORM_SCRATCH_ROOT
require_canonical_directory PRORM_HF_CACHE
require_canonical_directory PRORM_REPO_ROOT
require_canonical_file PRORM_IMAGE
require_canonical_file PRORM_PHASE2_OVERLAY
require_canonical_file PRORM_PHASE2_BASE_CONFIG
require_canonical_file PRORM_HF_INVENTORY
roots_overlap "${PRORM_PROJECT_ROOT}" "${PRORM_SCRATCH_ROOT}" \
  && die "project and scratch roots may not be equal or nested"
[[ -w "${PRORM_PROJECT_ROOT}" ]] || die "project root is not writable"
[[ -w "${PRORM_SCRATCH_ROOT}" ]] || die "scratch root is not writable"
require_contained PRORM_IMAGE "${PRORM_IMAGE}" "${PRORM_PROJECT_ROOT}"
require_contained PRORM_HF_CACHE "${PRORM_HF_CACHE}" "${PRORM_PROJECT_ROOT}"
require_contained PRORM_HF_INVENTORY "${PRORM_HF_INVENTORY}" "${PRORM_HF_CACHE}"
require_contained PRORM_PHASE2_OVERLAY "${PRORM_PHASE2_OVERLAY}" "${PRORM_REPO_ROOT}"
require_contained \
  PRORM_PHASE2_BASE_CONFIG "${PRORM_PHASE2_BASE_CONFIG}" "${PRORM_REPO_ROOT}"

validate_optional_aggregate() {
  local present_name="$1" path_name="$2" digest_name="$3" label="$4"
  local present="${!present_name}" path="${!path_name:-}" digest="${!digest_name:-}"
  case "${present}" in
    0)
      [[ -z "${path}" && -z "${digest}" ]] \
        || die "${label} path/hash were exported while presence=0"
      ;;
    1)
      [[ -n "${path}" && -n "${digest}" ]] \
        || die "${label} presence=1 requires path and SHA256"
      require_canonical_file "${path_name}"
      require_contained "${path_name}" "${path}" "${PRORM_PROJECT_ROOT}"
      reject_path_delimiters "${path_name}" "${path}"
      [[ "${digest}" =~ ^[0-9a-f]{64}$ ]] \
        || die "${digest_name} must be lowercase SHA256"
      printf '%s  %s\n' "${digest}" "${path}" \
        | sha256sum --check --status \
        || die "${label} SHA256 mismatch on compute node"
      ;;
    *)
      die "${present_name} must equal 0 or 1"
      ;;
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

for name in \
  PRORM_PROJECT_ROOT PRORM_SCRATCH_ROOT PRORM_IMAGE PRORM_HF_CACHE \
  PRORM_REPO_ROOT PRORM_PHASE2_OVERLAY PRORM_PHASE2_BASE_CONFIG \
  PRORM_HF_INVENTORY; do
  reject_path_delimiters "${name}" "${!name}"
done
for digest_name in \
  PRORM_IMAGE_SHA256 PRORM_PHASE2_DESIGN_SHA256 \
  PRORM_PHASE2_BASE_CONFIG_HASH PRORM_PHASE2_OVERLAY_FILE_SHA256 \
  PRORM_PHASE2_BASE_FILE_SHA256 PRORM_IDENTITIES_FILE_SHA256 \
  PRORM_HF_INVENTORY_SHA256; do
  [[ "${!digest_name}" =~ ^[0-9a-f]{64}$ ]] \
    || die "${digest_name} must be lowercase SHA256"
done
[[ "${PRORM_GIT_COMMIT}" =~ ^[0-9a-f]{40,64}$ ]] \
  || die "PRORM_GIT_COMMIT must be a full lowercase Git object ID"
printf '%s  %s\n' "${PRORM_IMAGE_SHA256}" "${PRORM_IMAGE}" \
  | sha256sum --check --status || die "image SHA256 mismatch"
printf '%s  %s\n' "${PRORM_HF_INVENTORY_SHA256}" "${PRORM_HF_INVENTORY}" \
  | sha256sum --check --status || die "HF inventory SHA256 mismatch"
printf '%s  %s\n' "${PRORM_PHASE2_OVERLAY_FILE_SHA256}" "${PRORM_PHASE2_OVERLAY}" \
  | sha256sum --check --status || die "submitted overlay bytes changed"
printf '%s  %s\n' "${PRORM_PHASE2_BASE_FILE_SHA256}" "${PRORM_PHASE2_BASE_CONFIG}" \
  | sha256sum --check --status || die "submitted base config bytes changed"

submission_head="$(git -C "${PRORM_REPO_ROOT}" rev-parse --verify HEAD)"
[[ "${submission_head}" = "${PRORM_GIT_COMMIT}" ]] \
  || die "submission repository HEAD changed after sbatch"
[[ -z "$(git -C "${PRORM_REPO_ROOT}" status --porcelain --untracked-files=normal)" ]] \
  || die "submission repository became dirty before compute execution"
overlay_relative="$(
  realpath --relative-to="${PRORM_REPO_ROOT}" "${PRORM_PHASE2_OVERLAY}"
)"
base_relative="$(
  realpath --relative-to="${PRORM_REPO_ROOT}" "${PRORM_PHASE2_BASE_CONFIG}"
)"
[[ "${overlay_relative}" = "configs/common_beta_pilot.yaml" ]] \
  || die "submitted overlay path is not the frozen pilot overlay"
[[ "${base_relative}" = "configs/common_beta_pilot_base.yaml" ]] \
  || die "submitted base path is not the frozen pilot base"
identity_relative="configs/identities.json"
identity_worktree="${PRORM_REPO_ROOT}/${identity_relative}"
[[ -f "${identity_worktree}" && ! -L "${identity_worktree}" ]] \
  || die "submitted identity file is missing or is a symlink"
printf '%s  %s\n' "${PRORM_IDENTITIES_FILE_SHA256}" "${identity_worktree}" \
  | sha256sum --check --status || die "submitted identity file changed"
for binding in \
  "${overlay_relative}:${PRORM_PHASE2_OVERLAY_FILE_SHA256}" \
  "${base_relative}:${PRORM_PHASE2_BASE_FILE_SHA256}" \
  "${identity_relative}:${PRORM_IDENTITIES_FILE_SHA256}"; do
  relative="${binding%%:*}"
  expected="${binding#*:}"
  observed="$(
    git -C "${PRORM_REPO_ROOT}" cat-file blob \
      "${PRORM_GIT_COMMIT}:${relative}" | sha256sum | awk '{print $1}'
  )"
  [[ "${observed}" = "${expected}" ]] \
    || die "committed bytes differ from submitted identity: ${relative}"
done

execution_id="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
job_dir="${PRORM_SCRATCH_ROOT}/phase2-pilot-jobs/${execution_id}"
[[ ! -e "${job_dir}" && ! -L "${job_dir}" ]] \
  || die "refusing to reuse scratch Phase-2 job directory: ${job_dir}"
mkdir -p "${PRORM_SCRATCH_ROOT}/phase2-pilot-jobs" \
  "${PRORM_PROJECT_ROOT}/runs/phase2-pilot" "${PRORM_PROJECT_ROOT}/artifacts"
mkdir "${job_dir}"

# The mutable submission checkout is used only as a local object source.  All
# Python and config execution below uses this exact detached, clean commit.
execution_repo="${job_dir}/source"
git clone --quiet --no-hardlinks --no-checkout \
  "${PRORM_REPO_ROOT}" "${execution_repo}"
git -C "${execution_repo}" checkout --quiet --detach "${PRORM_GIT_COMMIT}"
execution_overlay="${execution_repo}/${overlay_relative}"
execution_base="${execution_repo}/${base_relative}"
execution_identities="${execution_repo}/${identity_relative}"
for binding in \
  "${execution_overlay}:${PRORM_PHASE2_OVERLAY_FILE_SHA256}" \
  "${execution_base}:${PRORM_PHASE2_BASE_FILE_SHA256}" \
  "${execution_identities}:${PRORM_IDENTITIES_FILE_SHA256}"; do
  path="${binding%%:*}"
  expected="${binding#*:}"
  [[ -f "${path}" && ! -L "${path}" ]] \
    || die "detached execution input is missing or is a symlink: ${path}"
  printf '%s  %s\n' "${expected}" "${path}" \
    | sha256sum --check --status \
    || die "detached execution input differs from committed identity: ${path}"
done

assert_execution_checkout() {
  [[ "$(git -C "${execution_repo}" rev-parse --verify HEAD)" = "${PRORM_GIT_COMMIT}" ]] \
    || die "detached execution checkout changed commit"
  [[ -z "$(git -C "${execution_repo}" status --porcelain --untracked-files=normal)" ]] \
    || die "detached execution checkout became dirty"
}
assert_execution_checkout

container=(
  apptainer exec --cleanenv --nv
  --no-mount home,cwd,bind-paths
  --pwd "${execution_repo}"
  --bind "${job_dir}:${job_dir},${PRORM_HF_CACHE}:${PRORM_HF_CACHE}"
  --env "PYTHONPATH=${execution_repo}/src"
  --env "PYTHONNOUSERSITE=1"
  --env "PYTHONDONTWRITEBYTECODE=1"
  --env "HF_HOME=${PRORM_HF_CACHE}"
  --env "HF_HUB_CACHE=${PRORM_HF_CACHE}/hub"
  --env "HF_DATASETS_CACHE=${PRORM_HF_CACHE}/datasets"
  --env "HF_HUB_OFFLINE=1"
  --env "TRANSFORMERS_OFFLINE=1"
  --env "HF_DATASETS_OFFLINE=1"
  --env "HF_HUB_DISABLE_TELEMETRY=1"
  --env "TOKENIZERS_PARALLELISM=false"
  --env "PRORM_IMAGE=${PRORM_IMAGE}"
  --env "PRORM_IMAGE_SHA256=${PRORM_IMAGE_SHA256}"
  --env "PRORM_GIT_COMMIT=${PRORM_GIT_COMMIT}"
  --env "PRORM_HF_INVENTORY=${PRORM_HF_INVENTORY}"
  --env "PRORM_HF_INVENTORY_SHA256=${PRORM_HF_INVENTORY_SHA256}"
)
for variable in \
  SLURM_JOB_ID SLURM_ARRAY_JOB_ID SLURM_ARRAY_TASK_ID SLURM_JOB_NAME \
  SLURM_JOB_ACCOUNT SLURM_CLUSTER_NAME SLURM_JOB_PARTITION \
  SLURM_JOB_NODELIST SLURM_NNODES SLURM_NTASKS SLURM_CPUS_PER_TASK \
  SLURM_GPUS SLURM_GPUS_ON_NODE SLURM_PROCID SLURM_LOCALID \
  SLURM_NODEID CUDA_VISIBLE_DEVICES NVIDIA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL; do
  if [[ -v "${variable}" ]]; then
    container+=(--env "${variable}=${!variable}")
  fi
done
if (( PRORM_PHASE2_BETA_SOURCE_AGGREGATE_PRESENT )); then
  container+=(
    --bind "${PRORM_PHASE2_BETA_SOURCE_AGGREGATE}:${PRORM_PHASE2_BETA_SOURCE_AGGREGATE}"
  )
fi
if (( PRORM_PHASE2_HORIZON_PARENT_AGGREGATE_PRESENT )) \
  && [[ "${PRORM_PHASE2_HORIZON_PARENT_AGGREGATE}" \
    != "${PRORM_PHASE2_BETA_SOURCE_AGGREGATE:-}" ]]; then
  container+=(
    --bind "${PRORM_PHASE2_HORIZON_PARENT_AGGREGATE}:${PRORM_PHASE2_HORIZON_PARENT_AGGREGATE}"
  )
fi
container+=("${PRORM_IMAGE}")

phase2_config_check="${job_dir}/phase2-config-check.json"
base_config_check="${job_dir}/base-config-check.json"
assert_execution_checkout
"${container[@]}" python -m smart_reward.cli phase2-config-check \
  "${execution_overlay}" | tee "${phase2_config_check}"
"${container[@]}" python -m smart_reward.cli config-check \
  "${execution_base}" | tee "${base_config_check}"
python3 -I -S - \
  "${phase2_config_check}" "${base_config_check}" \
  "${PRORM_PHASE2_DESIGN_SHA256}" "${PRORM_PHASE2_BASE_CONFIG_HASH}" <<'PY'
import json
import sys
from pathlib import Path


phase2 = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
base = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
expected_design, expected_base = sys.argv[3:]
if (
    phase2.get("status") != "ok"
    or phase2.get("design_stage") != "pilot"
    or phase2.get("formal_eligibility") is not False
    or phase2.get("phase2_design_sha256") != expected_design
    or phase2.get("source_config_hash") != expected_base
):
    raise SystemExit("container-computed Phase-2 semantic identities are invalid")
if base.get("status") != "ok" or base.get("config_hash") != expected_base:
    raise SystemExit("container-computed base semantic identity is invalid")
PY

mapfile -t run_info < <(
  "${container[@]}" python - \
    "${execution_overlay}" "${SLURM_ARRAY_TASK_ID}" <<'PY'
import re
import sys
from smart_reward.phase2_config import load_phase2_config_bundle


bundle = load_phase2_config_bundle(sys.argv[1])
index = int(sys.argv[2])
seeds = bundle.config["run"]["seeds"]
if not 0 <= index < len(seeds):
    raise SystemExit(
        f"array index {index} is outside the Phase-2 pilot seed count {len(seeds)}"
    )
name = bundle.config["design"]["name"]
if not isinstance(name, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name) is None:
    raise SystemExit("Phase-2 design.name is unsafe for a project path")
print(name)
print(seeds[index])
print(bundle.design_identity)
PY
)
[[ "${#run_info[@]}" -eq 3 ]] || die "failed to resolve the selected pilot seed"
run_name="${run_info[0]}"
seed="${run_info[1]}"
compute_design_sha="${run_info[2]}"
[[ "${seed}" =~ ^[0-9]+$ ]] || die "resolved pilot seed is invalid"
[[ "${compute_design_sha}" = "${PRORM_PHASE2_DESIGN_SHA256}" ]] \
  || die "compute-resolved design identity differs from submission"

inventory_expected="${PRORM_HF_CACHE}/inventories/${PRORM_PHASE2_BASE_CONFIG_HASH}.json"
inventory_expected="$(realpath -e -- "${inventory_expected}")" \
  || die "base-config inventory disappeared before execution"
[[ "${inventory_expected}" = "${PRORM_HF_INVENTORY}" ]] \
  || die "submitted inventory is not addressed by the base config identity"

project_run="${PRORM_PROJECT_ROOT}/runs/phase2-pilot/${PRORM_PHASE2_DESIGN_SHA256}/seed-${seed}/job-${execution_id}"
mkdir -p "$(dirname "${project_run}")"
[[ ! -e "${project_run}" && ! -L "${project_run}" ]] \
  || die "refusing to reuse Phase-2 project run: ${project_run}"
mkdir "${project_run}"

materialized_this_job=0

atomic_sync_file() {
  local source="$1" filename="$2" destination="${project_run}/${filename}" temporary=""
  [[ -f "${source}" && ! -L "${source}" ]] || return 0
  [[ ! -e "${destination}" && ! -L "${destination}" ]] \
    || { echo "refusing to overwrite run evidence: ${destination}" >&2; return 1; }
  temporary="$(mktemp "${project_run}/.${filename}.tmp.XXXXXX")" || return 1
  cp -p -- "${source}" "${temporary}" || {
    rm -f -- "${temporary}"
    return 1
  }
  mv -T -- "${temporary}" "${destination}" || {
    rm -f -- "${temporary}"
    return 1
  }
}

atomic_write_marker() {
  local marker="$1" workload_exit="$2" final_exit="$3" temporary=""
  [[ ! -e "${project_run}/SUCCESS" && ! -L "${project_run}/SUCCESS" ]] || return 1
  [[ ! -e "${project_run}/FAILED" && ! -L "${project_run}/FAILED" ]] || return 1
  temporary="$(mktemp "${project_run}/.${marker}.tmp.XXXXXX")" || return 1
  {
    printf 'schema_version=prorm-phase2-run-status/v1\n'
    printf 'status=%s\n' "${marker}"
    printf 'workload_exit_code=%s\n' "${workload_exit}"
    printf 'final_exit_code=%s\n' "${final_exit}"
    printf 'array_job_id=%s\n' "${SLURM_ARRAY_JOB_ID}"
    printf 'array_task_id=%s\n' "${SLURM_ARRAY_TASK_ID}"
    printf 'seed=%s\n' "${seed}"
    printf 'phase2_design_sha256=%s\n' "${PRORM_PHASE2_DESIGN_SHA256}"
    printf 'base_config_hash=%s\n' "${PRORM_PHASE2_BASE_CONFIG_HASH}"
    printf 'git_commit=%s\n' "${PRORM_GIT_COMMIT}"
    printf 'beta_source_aggregate_present=%s\n' \
      "${PRORM_PHASE2_BETA_SOURCE_AGGREGATE_PRESENT}"
    printf 'beta_source_aggregate_sha256=%s\n' \
      "${PRORM_PHASE2_BETA_SOURCE_AGGREGATE_SHA256:-none}"
    printf 'horizon_parent_aggregate_present=%s\n' \
      "${PRORM_PHASE2_HORIZON_PARENT_AGGREGATE_PRESENT}"
    printf 'horizon_parent_aggregate_sha256=%s\n' \
      "${PRORM_PHASE2_HORIZON_PARENT_AGGREGATE_SHA256:-none}"
    printf 'created_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "${temporary}" || {
    rm -f -- "${temporary}"
    return 1
  }
  mv -T -- "${temporary}" "${project_run}/${marker}" || {
    rm -f -- "${temporary}"
    return 1
  }
}

finalize_run() {
  local workload_exit=$? sync_exit=0 final_exit marker filename
  trap - EXIT
  set +e
  for filename in \
    phase2-config-check.json base-config-check.json \
    hf-inventory-verification.json gpu-check.log env-report.log \
    run-manifest.json materialize.log memory-materialize.json \
    artifact-materialization.json artifact-verification.json \
    phase2-run.log memory-phase2.json phase2-failure-evidence.json \
    phase2-pilot-diagnostics.json \
    phase2-pilot-diagnostics.diagnostics.jsonl phase2-output-verification.json; do
    atomic_sync_file "${job_dir}/${filename}" "${filename}" || sync_exit=1
  done
  if (( workload_exit == 0 )); then
    for filename in \
      phase2-config-check.json base-config-check.json \
      hf-inventory-verification.json gpu-check.log env-report.log \
      run-manifest.json artifact-materialization.json \
      artifact-verification.json phase2-run.log memory-phase2.json \
      phase2-pilot-diagnostics.json phase2-pilot-diagnostics.diagnostics.jsonl \
      phase2-output-verification.json; do
      [[ -f "${project_run}/${filename}" && ! -L "${project_run}/${filename}" ]] || {
        echo "successful Phase-2 workload is missing core evidence: ${filename}" >&2
        sync_exit=1
      }
    done
    [[ -L "${project_run}/artifact" && -d "${project_run}/artifact" ]] || {
      echo "successful Phase-2 workload is missing its artifact link" >&2
      sync_exit=1
    }
    if (( materialized_this_job == 1 )); then
      for filename in materialize.log memory-materialize.json; do
        [[ -f "${project_run}/${filename}" ]] || {
          echo "artifact-producing workload is missing ${filename}" >&2
          sync_exit=1
        }
      done
    fi
  fi
  final_exit="${workload_exit}"
  if (( sync_exit != 0 && final_exit == 0 )); then final_exit=1; fi
  if (( final_exit == 0 )); then marker="SUCCESS"; else marker="FAILED"; fi
  if ! atomic_write_marker "${marker}" "${workload_exit}" "${final_exit}"; then
    echo "failed to install atomic ${marker} marker in ${project_run}" >&2
    final_exit=1
  fi
  exit "${final_exit}"
}
trap finalize_run EXIT

verification_result="${job_dir}/hf-inventory-verification.json"
assert_execution_checkout
"${container[@]}" python "${execution_repo}/scripts/hpc4/stage_hf_assets.py" \
  "${execution_base}" "${PRORM_HF_CACHE}" \
  --verify-only --inventory "${PRORM_HF_INVENTORY}" \
  | tee "${verification_result}"
python3 -I -S - \
  "${verification_result}" "${PRORM_PHASE2_BASE_CONFIG_HASH}" \
  "${PRORM_HF_INVENTORY_SHA256}" <<'PY'
import json
import sys
from pathlib import Path


value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if (
    value.get("status") != "ok"
    or value.get("verify_only") is not True
    or value.get("config_hash") != sys.argv[2]
    or value.get("inventory_sha256") != sys.argv[3]
):
    raise SystemExit("offline HF inventory verification did not preserve the base identity")
PY
printf '%s  %s\n' "${PRORM_HF_INVENTORY_SHA256}" "${PRORM_HF_INVENTORY}" \
  | sha256sum --check --status || die "HF inventory changed during offline verification"

gpu_check="${job_dir}/gpu-check.log"
assert_execution_checkout
"${container[@]}" python - <<'PY' | tee "${gpu_check}"
import json
import os
import subprocess
import torch


subprocess.run(["git", "--version"], check=True)
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("Phase-2 pilot requires exactly one visible allocated CUDA GPU")
partition = os.environ.get("SLURM_JOB_PARTITION")
name = torch.cuda.get_device_name(0)
if partition != "gpu-l20" or "l20" not in name.lower():
    raise SystemExit(
        "Phase-2 design requires gpu-l20 backed by an NVIDIA L20; "
        f"observed GPU model {name!r} on partition {partition!r}"
    )
print(
    json.dumps(
        {
            "cuda_device_count": 1,
            "gpu_model": name,
            "partition": partition,
            "status": "ok",
        },
        allow_nan=False,
        sort_keys=True,
    )
)
PY

manifest="${job_dir}/run-manifest.json"
env_report_log="${job_dir}/env-report.log"
assert_execution_checkout
"${container[@]}" python -m smart_reward.cli env-report \
  "${execution_base}" --seed "${seed}" --repo-root "${execution_repo}" \
  --output "${manifest}" | tee "${env_report_log}"

artifact_dir="${job_dir}/artifact"
project_artifact="${PRORM_PROJECT_ROOT}/artifacts/${PRORM_PHASE2_BASE_CONFIG_HASH}/${PRORM_IMAGE_SHA256}/${PRORM_HF_INVENTORY_SHA256}/${PRORM_GIT_COMMIT}/seed-${seed}"
artifact_parent="$(dirname "${project_artifact}")"
mkdir -p "${artifact_parent}"
if [[ -e "${project_artifact}" || -L "${project_artifact}" ]]; then
  [[ -d "${project_artifact}" && ! -L "${project_artifact}" \
    && -f "${project_artifact}/metadata.json" \
    && ! -L "${project_artifact}/metadata.json" ]] \
    || die "content-addressed base artifact exists but is incomplete or unsafe"
  cp -a -- "${project_artifact}" "${artifact_dir}"
  artifact_mode="reused"
else
  artifact_lock="${project_artifact}.publish.lock"
  exec {artifact_lock_fd}> "${artifact_lock}"
  flock -n "${artifact_lock_fd}" \
    || die "another job is publishing this exact base artifact"
  [[ ! -e "${project_artifact}" && ! -L "${project_artifact}" ]] \
    || die "base artifact appeared after the initial publication check"
  materialize_log="${job_dir}/materialize.log"
  assert_execution_checkout
  "${container[@]}" env \
    "PRORM_MEMORY_REPORT=${job_dir}/memory-materialize.json" \
    python -m smart_reward.cli controlled-materialize \
    "${execution_base}" "${artifact_dir}" --seed "${seed}" --device cuda \
    2>&1 | tee "${materialize_log}"
  materialized_this_job=1
  [[ -d "${artifact_dir}" && ! -L "${artifact_dir}" \
    && -f "${artifact_dir}/metadata.json" \
    && ! -L "${artifact_dir}/metadata.json" ]] \
    || die "base materializer did not produce a complete artifact"
  artifact_staging="$(
    mktemp -d "${artifact_parent}/.seed-${seed}.phase2-${execution_id}.XXXXXX"
  )"
  cp -a -- "${artifact_dir}/." "${artifact_staging}/"
  [[ ! -e "${project_artifact}" && ! -L "${project_artifact}" ]] \
    || die "base artifact destination appeared before atomic publication"
  mv -T --no-clobber -- "${artifact_staging}" "${project_artifact}" \
    || die "atomic no-overwrite base artifact publication failed"
  [[ ! -e "${artifact_staging}" && ! -L "${artifact_staging}" ]] \
    || die "base artifact publication lost an atomic rename race"
  flock -u "${artifact_lock_fd}"
  artifact_mode="materialized"
fi

artifact_metadata_sha256="$(
  sha256sum -- "${project_artifact}/metadata.json" | awk '{print $1}'
)"
[[ "${artifact_metadata_sha256}" =~ ^[0-9a-f]{64}$ ]] \
  || die "failed to hash base artifact metadata"
job_artifact_metadata_sha256="$(
  sha256sum -- "${artifact_dir}/metadata.json" | awk '{print $1}'
)"
[[ "${job_artifact_metadata_sha256}" = "${artifact_metadata_sha256}" ]] \
  || die "scratch and published base artifact metadata differ"
python3 -I -S - \
  "${job_dir}/artifact-materialization.json" "${artifact_mode}" \
  "${PRORM_PHASE2_BASE_CONFIG_HASH}" "${PRORM_PHASE2_DESIGN_SHA256}" \
  "${seed}" "${artifact_metadata_sha256}" "${PRORM_GIT_COMMIT}" \
  "${PRORM_IMAGE_SHA256}" "${PRORM_HF_INVENTORY_SHA256}" <<'PY'
import json
import sys
from pathlib import Path


(
    output,
    mode,
    base_hash,
    design_hash,
    seed,
    metadata_sha,
    commit,
    image_sha,
    inventory_sha,
) = sys.argv[1:]
payload = {
    "schema_version": "prorm-phase2-artifact-binding/v1",
    "mode": mode,
    "base_config_hash": base_hash,
    "phase2_design_sha256": design_hash,
    "seed": int(seed),
    "artifact_metadata_sha256": metadata_sha,
    "producer": {
        "git_commit": commit,
        "image_sha256": image_sha,
        "hf_inventory_sha256": inventory_sha,
    },
}
with Path(output).open("x", encoding="utf-8", newline="\n") as stream:
    json.dump(payload, stream, allow_nan=False, sort_keys=True, separators=(",", ":"))
    stream.write("\n")
PY

artifact_link="${project_run}/artifact"
[[ ! -e "${artifact_link}" && ! -L "${artifact_link}" ]] \
  || die "refusing to overwrite the Phase-2 run artifact link"
artifact_relative="$(realpath --relative-to="${project_run}" "${project_artifact}")"
[[ "${artifact_relative}" != /* ]] || die "artifact link target must be relative"
ln -s -- "${artifact_relative}" "${artifact_link}"

# This is an explicit pre-run verification of both the base materialization
# artifact and its selected-seed formal manifest.  phase2-run repeats the same
# fail-closed bridge checks before opening either LLM.
artifact_verification="${job_dir}/artifact-verification.json"
assert_execution_checkout
"${container[@]}" python - \
  "${execution_overlay}" "${seed}" "${artifact_dir}" "${manifest}" \
  <<'PY' | tee "${artifact_verification}"
import json
import sys
from smart_reward.phase2_inputs import prepare_phase2_inputs


inputs = prepare_phase2_inputs(
    sys.argv[1],
    seed=int(sys.argv[2]),
    artifact_dir=sys.argv[3],
    run_manifest=sys.argv[4],
)
print(
    json.dumps(
        {
            "artifact_metadata_sha256": inputs.artifact_metadata_sha256,
            "base_config_hash": inputs.source_config_hash,
            "formal_environment": inputs.environment_identity["formal"],
            "phase2_design_sha256": inputs.phase2_config_hash,
            "seed": inputs.seed,
            "status": "ok",
        },
        allow_nan=False,
        sort_keys=True,
    )
)
PY

phase2_result="${job_dir}/phase2-pilot-diagnostics.json"
phase2_diagnostics="${job_dir}/phase2-pilot-diagnostics.diagnostics.jsonl"
phase2_log="${job_dir}/phase2-run.log"
phase2_aggregate_flags=()
if (( PRORM_PHASE2_BETA_SOURCE_AGGREGATE_PRESENT )); then
  printf '%s  %s\n' \
    "${PRORM_PHASE2_BETA_SOURCE_AGGREGATE_SHA256}" \
    "${PRORM_PHASE2_BETA_SOURCE_AGGREGATE}" \
    | sha256sum --check --status \
    || die "beta-source aggregate changed before Phase-2 execution"
  phase2_aggregate_flags+=(
    --beta-source-aggregate "${PRORM_PHASE2_BETA_SOURCE_AGGREGATE}"
  )
fi
if (( PRORM_PHASE2_HORIZON_PARENT_AGGREGATE_PRESENT )); then
  printf '%s  %s\n' \
    "${PRORM_PHASE2_HORIZON_PARENT_AGGREGATE_SHA256}" \
    "${PRORM_PHASE2_HORIZON_PARENT_AGGREGATE}" \
    | sha256sum --check --status \
    || die "horizon-parent aggregate changed before Phase-2 execution"
  phase2_aggregate_flags+=(
    --horizon-parent-aggregate "${PRORM_PHASE2_HORIZON_PARENT_AGGREGATE}"
  )
fi
assert_execution_checkout
"${container[@]}" env \
  "PRORM_MEMORY_REPORT=${job_dir}/memory-phase2.json" \
  "PRORM_FAILURE_EVIDENCE=${job_dir}/phase2-failure-evidence.json" \
  python -m smart_reward.cli phase2-run \
  "${execution_overlay}" "${artifact_dir}" "${manifest}" "${phase2_result}" \
  --seed "${seed}" --device cuda "${phase2_aggregate_flags[@]}" \
  2>&1 | tee "${phase2_log}"
assert_execution_checkout

output_verification="${job_dir}/phase2-output-verification.json"
python3 -I -S - \
  "${phase2_result}" "${phase2_diagnostics}" "${output_verification}" \
  "${seed}" "${PRORM_PHASE2_BASE_CONFIG_HASH}" \
  "${PRORM_PHASE2_DESIGN_SHA256}" "${PRORM_GIT_COMMIT}" \
  "${PRORM_IMAGE_SHA256}" "${PRORM_HF_INVENTORY_SHA256}" \
  "${artifact_metadata_sha256}" \
  "${PRORM_PHASE2_BETA_SOURCE_AGGREGATE_SHA256:-none}" \
  "${PRORM_PHASE2_HORIZON_PARENT_AGGREGATE_SHA256:-none}" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path


(
    result_raw,
    diagnostics_raw,
    output_raw,
    seed_raw,
    base_hash,
    design_hash,
    commit,
    image_sha,
    inventory_sha,
    artifact_metadata_sha,
    submitted_beta_source_sha,
    submitted_horizon_parent_sha,
) = sys.argv[1:]
result_path = Path(result_raw)
diagnostics_path = Path(diagnostics_raw)
result = json.loads(result_path.read_text(encoding="utf-8"))
if (
    result.get("schema_version") != "common-beta-pilot-diagnostics/v2"
    or result.get("design_stage") != "pilot"
    or result.get("pilot_phase") not in {"calibration", "freeze"}
    or result.get("formal_eligibility") is not False
    or result.get("per_seed_supports_formal_claim") is not False
    or result.get("source_config_hash") != base_hash
    or result.get("phase2_design_sha256") != design_hash
    or result.get("seed") != int(seed_raw)
    or result.get("artifact_dir") != "artifact"
    or result.get("artifact_metadata_sha256") != artifact_metadata_sha
    or result.get("run_manifest") != "run-manifest.json"
    or result.get("diagnostics_jsonl")
    != "phase2-pilot-diagnostics.diagnostics.jsonl"
):
    raise SystemExit("Phase-2 pilot diagnostic identity or evidence role is invalid")
pilot_phase = result["pilot_phase"]
boundary = result.get("information_boundary")
if not isinstance(boundary, dict) or any(
    boundary.get(key) is not False
    for key in (
        "final_oracle_session_opened",
        "rollout_responses_oracle_scored",
        "heldout_evaluator_called",
        "oracle_outcomes_serialized",
        "prompt_or_response_text_serialized",
        "token_ids_or_response_masks_serialized",
    )
):
    raise SystemExit("Phase-2 pilot crossed its target-free information boundary")
head_training = result.get("head_training")
if (
    not isinstance(head_training, dict)
    or head_training.get("head_weights_serialized") is not False
):
    raise SystemExit("Phase-2 pilot serialized reward-head weights")

runtime_contract = result.get("phase2_runtime_contract")
if (
    not isinstance(runtime_contract, dict)
    or runtime_contract.get("schema_version") != "common-beta-design/v4"
    or runtime_contract.get("stage") != "pilot"
    or runtime_contract.get("pilot_phase") != pilot_phase
    or runtime_contract.get("formal_eligibility") is not False
    or runtime_contract.get("allowed_horizon_sequence") != [256, 512, 1024]
):
    raise SystemExit("Phase-2 pilot runtime contract is malformed")
horizon_index = runtime_contract.get("horizon_grid_index")
horizon = runtime_contract.get("max_response_tokens")
horizon_parent = runtime_contract.get("parent_pilot_aggregate_sha256")
previous_horizon_failed = runtime_contract.get("previous_horizon_failed_length_gate")


def valid_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


if submitted_beta_source_sha != "none" and not valid_sha256(
    submitted_beta_source_sha
):
    raise SystemExit("submitted Phase-2 beta-source SHA256 is invalid")
if submitted_horizon_parent_sha != "none" and not valid_sha256(
    submitted_horizon_parent_sha
):
    raise SystemExit("submitted Phase-2 horizon-parent SHA256 is invalid")
expected_beta_source_sha = (
    None if submitted_beta_source_sha == "none" else submitted_beta_source_sha
)
expected_horizon_parent_sha = (
    None
    if submitted_horizon_parent_sha == "none"
    else submitted_horizon_parent_sha
)
if runtime_contract.get("beta_source_aggregate_sha256") != expected_beta_source_sha:
    raise SystemExit("Phase-2 runtime beta source differs from submitted evidence")
if horizon_parent != expected_horizon_parent_sha:
    raise SystemExit("Phase-2 runtime horizon parent differs from submitted evidence")


if (
    isinstance(horizon_index, bool)
    or not isinstance(horizon_index, int)
    or horizon_index not in {0, 1, 2}
    or horizon != [256, 512, 1024][horizon_index]
    or not isinstance(previous_horizon_failed, bool)
):
    raise SystemExit("Phase-2 pilot response-horizon identity is invalid")
if horizon_index == 0:
    if previous_horizon_failed:
        raise SystemExit("initial Phase-2 horizon claims an impossible predecessor failure")
    if pilot_phase == "calibration":
        if horizon_parent is not None:
            raise SystemExit("initial calibration horizon unexpectedly names a parent")
    elif not valid_sha256(horizon_parent):
        raise SystemExit("pilot freeze horizon lacks its calibration aggregate parent")
elif not valid_sha256(horizon_parent) or not previous_horizon_failed:
    raise SystemExit("escalated Phase-2 horizon lacks its failed predecessor evidence")

if pilot_phase == "calibration":
    beta_evidence = result.get("train_only_global_beta_calibration_candidate")
    if (
        not isinstance(beta_evidence, dict)
        or beta_evidence.get("schema_version")
        != "global-beta-calibration-candidate/v1"
        or "pilot_fixed_global_beta_rehearsal" in result
        or runtime_contract.get("frozen_global_beta") is not None
        or runtime_contract.get("beta_source_aggregate_sha256") is not None
        or beta_evidence.get("frozen_global_beta") is not None
        or beta_evidence.get("formal_beta_selected") is not False
    ):
        raise SystemExit("Phase-2 calibration beta evidence is malformed")
    expected_beta = beta_evidence.get("candidate_beta")
else:
    beta_evidence = result.get("pilot_fixed_global_beta_rehearsal")
    if (
        not isinstance(beta_evidence, dict)
        or beta_evidence.get("schema_version")
        != "pilot-frozen-global-beta-rehearsal/v1"
        or "train_only_global_beta_calibration_candidate" in result
        or beta_evidence.get("beta_matches_frozen_global_beta") is not True
        or beta_evidence.get("beta_selected_from_current_seed_curvature") is not False
        or beta_evidence.get("post_evaluation_retuning") is not False
        or not valid_sha256(beta_evidence.get("beta_source_aggregate_sha256"))
        or beta_evidence.get("beta_source_aggregate_sha256")
        != expected_beta_source_sha
        or runtime_contract.get("beta_source_aggregate_sha256")
        != expected_beta_source_sha
    ):
        raise SystemExit("Phase-2 freeze beta evidence is malformed")
    expected_beta = beta_evidence.get("beta_common")
    if (
        expected_beta != beta_evidence.get("frozen_global_beta")
        or expected_beta != runtime_contract.get("frozen_global_beta")
    ):
        raise SystemExit("Phase-2 freeze did not deploy its identity-frozen beta")
if (
    not isinstance(expected_beta, (int, float))
    or isinstance(expected_beta, bool)
    or not math.isfinite(float(expected_beta))
    or float(expected_beta) <= 0.0
):
    raise SystemExit("Phase-2 pilot common beta is invalid")

safety = result.get("measured_kl_safety")
expected_safety_keys = {
    "schema_version",
    "cap",
    "passed",
    "measured_by_policy",
    "violations",
    "beta_retuned",
}
if not isinstance(safety, dict) or set(safety) != expected_safety_keys:
    raise SystemExit("Phase-2 pilot measured-KL safety evidence is malformed")
cap = safety.get("cap")
passed = safety.get("passed")
measured = safety.get("measured_by_policy")
arm_names = {"zero_b", "bt_mle", "prorm_plus", "oracle_step"}
arms = result.get("arms")
if (
    safety.get("schema_version") != "measured-kl-safety/v1"
    or not isinstance(cap, (int, float))
    or isinstance(cap, bool)
    or not math.isfinite(float(cap))
    or float(cap) <= 0.0
    or not isinstance(passed, bool)
    or not isinstance(measured, dict)
    or set(measured) != arm_names
    or not isinstance(arms, dict)
    or set(arms) != arm_names
    or safety.get("beta_retuned") is not False
):
    raise SystemExit("Phase-2 pilot measured-KL safety evidence is invalid")
for arm_name in arm_names:
    measured_value = measured.get(arm_name)
    arm = arms.get(arm_name)
    arm_mean = (
        arm.get("mean_on_policy_kl_pi_updated_to_pi0")
        if isinstance(arm, dict)
        else None
    )
    if (
        not isinstance(measured_value, (int, float))
        or isinstance(measured_value, bool)
        or not math.isfinite(float(measured_value))
        or float(measured_value) < 0.0
        or not isinstance(arm_mean, (int, float))
        or isinstance(arm_mean, bool)
        or not math.isfinite(float(arm_mean))
        or not math.isclose(
            float(measured_value),
            float(arm_mean),
            rel_tol=1.0e-6,
            abs_tol=1.0e-8,
        )
    ):
        raise SystemExit("Phase-2 pilot arm and safety KL diagnostics disagree")
expected_violations = [
    arm_name
    for arm_name in sorted(arm_names)
    if float(measured[arm_name]) > float(cap)
]
if (
    safety.get("violations") != expected_violations
    or passed is not (not expected_violations)
):
    raise SystemExit("Phase-2 pilot KL threshold arithmetic is inconsistent")

pilot_kl_gate = result.get("pilot_kl_safety_gate")
expected_pilot_gate_keys = {
    "schema_version",
    "gate_passed",
    "measure_only",
    "supports_formal_claim",
    "violations",
    "on_violation",
}
if (
    not isinstance(pilot_kl_gate, dict)
    or set(pilot_kl_gate) != expected_pilot_gate_keys
    or pilot_kl_gate.get("schema_version") != "pilot-measured-kl-gate/v1"
    or pilot_kl_gate.get("gate_passed") is not passed
    or pilot_kl_gate.get("measure_only") is not True
    or pilot_kl_gate.get("supports_formal_claim") is not False
    or pilot_kl_gate.get("violations") != expected_violations
    or pilot_kl_gate.get("on_violation")
    != "publish_target_free_diagnostics_without_final_oracle"
):
    raise SystemExit("Phase-2 pilot KL gate is not a measure-only diagnostic")

pre_oracle_gate = result.get("pre_oracle_safety_gate")
expected_pre_oracle_keys = {
    "schema_version",
    "design_stage",
    "pilot_phase",
    "measure_only",
    "formal_gate",
    "thresholds",
    "observed_by_arm",
    "violations",
    "passed",
    "beta_retuned",
    "on_violation",
}
expected_thresholds = {
    "mean_policy_to_reference_kl_cap": 0.02,
    "prompt_mean_p95_kl_cap": 0.02,
    "prompt_mean_p99_kl_cap": 0.05,
    "prompt_mean_maximum_kl_cap": 0.10,
    "per_sequence_maximum_kl_cap": 0.20,
    "reached_max_length_rate_cap": 0.05,
}
expected_observed_keys = {
    "mean_policy_to_reference_kl",
    "prompt_mean_p95_kl",
    "prompt_mean_p99_kl",
    "prompt_mean_maximum_kl",
    "per_sequence_maximum_kl",
    "reached_max_length_rate",
}
observed_by_arm = (
    pre_oracle_gate.get("observed_by_arm")
    if isinstance(pre_oracle_gate, dict)
    else None
)
if (
    not isinstance(pre_oracle_gate, dict)
    or set(pre_oracle_gate) != expected_pre_oracle_keys
    or pre_oracle_gate.get("schema_version") != "phase2-pre-oracle-safety-gate/v1"
    or pre_oracle_gate.get("design_stage") != "pilot"
    or pre_oracle_gate.get("pilot_phase") != pilot_phase
    or pre_oracle_gate.get("measure_only") is not True
    or pre_oracle_gate.get("formal_gate") is not False
    or pre_oracle_gate.get("thresholds") != expected_thresholds
    or not isinstance(observed_by_arm, dict)
    or set(observed_by_arm) != arm_names
    or not isinstance(pre_oracle_gate.get("violations"), list)
    or not isinstance(pre_oracle_gate.get("passed"), bool)
    or pre_oracle_gate.get("beta_retuned") is not False
    or pre_oracle_gate.get("on_violation")
    != "publish_target_free_diagnostics_without_final_oracle"
):
    raise SystemExit("Phase-2 pilot pre-oracle safety gate is malformed")
metric_to_cap = {
    metric: expected_thresholds[f"{metric}_cap"]
    for metric in expected_observed_keys
}
arm_order = ("zero_b", "bt_mle", "prorm_plus", "oracle_step")
expected_pre_oracle_violations = []
for arm_name in arm_order:
    observed = observed_by_arm[arm_name]
    arm = arms[arm_name]
    tail = arm.get("on_policy_kl_tail") if isinstance(arm, dict) else None
    rollout_length = arm.get("rollout_length") if isinstance(arm, dict) else None
    expected_observed = {
        "mean_policy_to_reference_kl": arm.get(
            "mean_on_policy_kl_pi_updated_to_pi0"
        ),
        "prompt_mean_p95_kl": (
            tail.get("p95") if isinstance(tail, dict) else None
        ),
        "prompt_mean_p99_kl": (
            tail.get("p99") if isinstance(tail, dict) else None
        ),
        "prompt_mean_maximum_kl": (
            tail.get("maximum") if isinstance(tail, dict) else None
        ),
        "per_sequence_maximum_kl": (
            tail.get("per_sequence_maximum") if isinstance(tail, dict) else None
        ),
        "reached_max_length_rate": (
            rollout_length.get("reached_max_length_rate")
            if isinstance(rollout_length, dict)
            else None
        ),
    }
    if not isinstance(observed, dict) or set(observed) != expected_observed_keys:
        raise SystemExit("Phase-2 pilot per-arm pre-oracle evidence is malformed")
    for metric, value in observed.items():
        expected_value = expected_observed[metric]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) < 0.0
            or not isinstance(expected_value, (int, float))
            or isinstance(expected_value, bool)
            or not math.isfinite(float(expected_value))
            or not math.isclose(
                float(value),
                float(expected_value),
                rel_tol=1.0e-6,
                abs_tol=1.0e-8,
            )
        ):
            raise SystemExit("Phase-2 pilot pre-oracle and arm diagnostics disagree")
        if float(value) > metric_to_cap[metric]:
            expected_pre_oracle_violations.append(f"{arm_name}:{metric}")
if (
    pre_oracle_gate.get("violations") != expected_pre_oracle_violations
    or pre_oracle_gate.get("passed") is not (not expected_pre_oracle_violations)
):
    raise SystemExit("Phase-2 pilot pre-oracle threshold arithmetic is inconsistent")

forbidden_keys = {
    "head_weight",
    "head_weights",
    "prompt",
    "response",
    "token_ids",
    "response_mask",
    "reward",
    "transformed_oracle_reward",
    "target_utility",
    "utility",
    "regret",
    "heldout",
    "heldout_fixed_beta",
    "true_rewards",
}


def nested_keys(value):
    if isinstance(value, dict):
        keys = set(value)
        for item in value.values():
            keys.update(nested_keys(item))
        return keys
    if isinstance(value, list):
        keys = set()
        for item in value:
            keys.update(nested_keys(item))
        return keys
    return set()


if forbidden_keys.intersection(nested_keys(result)):
    raise SystemExit("Phase-2 pilot result contains a forbidden outcome/vector field")
environment = result.get("environment_identity")
if environment != result.get("current_process_identity") or not isinstance(environment, dict):
    raise SystemExit("Phase-2 result process and manifest environments differ")
if (
    environment.get("formal") is not True
    or environment.get("git_commit") != commit
    or environment.get("image_sha256") != image_sha
    or environment.get("hf_inventory_sha256") != inventory_sha
    or environment.get("account") != "sigroup"
):
    raise SystemExit("Phase-2 result lacks its formal producer identity")
digest = hashlib.sha256(diagnostics_path.read_bytes()).hexdigest()
if result.get("diagnostics_sha256") != digest:
    raise SystemExit("Phase-2 diagnostic sidecar SHA256 does not match the result")
count = 0
allowed_row_keys = {
    "schema_version",
    "pilot_phase",
    "arm",
    "beta_common",
    "beta_role",
    "prompt_id",
    "candidate_index",
    "response_token_count",
    "terminated_by_eos",
    "reached_max_length",
    "prompt_rollout_seed",
    "kl_orientation",
    "kl_history_source",
    "on_policy_kl_pi_updated_to_pi0",
    "contains_prompt_text",
    "contains_response_text",
    "contains_token_ids",
    "contains_oracle_outcome",
}
with diagnostics_path.open("r", encoding="utf-8") as stream:
    for line in stream:
        value = json.loads(
            line,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {item}")
            ),
        )
        if (
            value.get("schema_version") != "common-beta-pilot-diagnostic-row/v2"
            or set(value) != allowed_row_keys
            or value.get("pilot_phase") != pilot_phase
            or value.get("arm") not in arm_names
            or value.get("beta_role")
            != (
                "seed_calibration_candidate"
                if pilot_phase == "calibration"
                else "frozen_global_beta_candidate"
            )
            or not isinstance(value.get("beta_common"), (int, float))
            or isinstance(value.get("beta_common"), bool)
            or not math.isfinite(float(value["beta_common"]))
            or float(value["beta_common"]) <= 0.0
            or not math.isclose(
                float(value["beta_common"]),
                float(expected_beta),
                rel_tol=0.0,
                abs_tol=0.0,
            )
            or forbidden_keys.intersection(value)
            or any(
                value.get(key) is not False
                for key in (
                    "contains_prompt_text",
                    "contains_response_text",
                    "contains_token_ids",
                    "contains_oracle_outcome",
                )
            )
        ):
            raise SystemExit("Phase-2 pilot sidecar is not strictly target-free")
        for key, item in value.items():
            if isinstance(item, float) and not math.isfinite(item):
                raise SystemExit(f"non-finite pilot diagnostic value: {key}")
        count += 1
if count < 1:
    raise SystemExit("Phase-2 pilot diagnostic sidecar is empty")
verification = {
    "schema_version": "prorm-phase2-output-verification/v1",
    "status": "passed",
    "seed": int(seed_raw),
    "source_config_hash": base_hash,
    "phase2_design_sha256": design_hash,
    "pilot_phase": pilot_phase,
    "diagnostic_records": count,
    "diagnostics_sha256": digest,
    "kl_gate_passed": passed,
    "kl_measure_only": True,
    "kl_violations": expected_violations,
    "pre_oracle_gate_passed": pre_oracle_gate["passed"],
    "pre_oracle_violations": expected_pre_oracle_violations,
    "environment_identity": environment,
}
with Path(output_raw).open("x", encoding="utf-8", newline="\n") as stream:
    json.dump(
        verification,
        stream,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    stream.write("\n")
PY

printf '%s  %s\n' "${artifact_metadata_sha256}" "${project_artifact}/metadata.json" \
  | sha256sum --check --status \
  || die "published base artifact metadata changed during Phase-2 execution"
printf '%s  %s\n' "${PRORM_HF_INVENTORY_SHA256}" "${PRORM_HF_INVENTORY}" \
  | sha256sum --check --status \
  || die "HF inventory changed during Phase-2 execution"
printf '%s  %s\n' "${PRORM_IMAGE_SHA256}" "${PRORM_IMAGE}" \
  | sha256sum --check --status \
  || die "research image changed during Phase-2 execution"
if (( PRORM_PHASE2_BETA_SOURCE_AGGREGATE_PRESENT )); then
  printf '%s  %s\n' \
    "${PRORM_PHASE2_BETA_SOURCE_AGGREGATE_SHA256}" \
    "${PRORM_PHASE2_BETA_SOURCE_AGGREGATE}" \
    | sha256sum --check --status \
    || die "beta-source aggregate changed during Phase-2 execution"
fi
if (( PRORM_PHASE2_HORIZON_PARENT_AGGREGATE_PRESENT )); then
  printf '%s  %s\n' \
    "${PRORM_PHASE2_HORIZON_PARENT_AGGREGATE_SHA256}" \
    "${PRORM_PHASE2_HORIZON_PARENT_AGGREGATE}" \
    | sha256sum --check --status \
    || die "horizon-parent aggregate changed during Phase-2 execution"
fi
assert_execution_checkout
printf 'Phase-2 pilot complete: design=%s seed=%s run=%s\n' \
  "${run_name}" "${seed}" "${project_run}"
