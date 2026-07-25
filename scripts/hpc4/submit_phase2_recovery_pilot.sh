#!/usr/bin/env bash
set -euo pipefail

die() { echo "error: $*" >&2; exit 2; }

[[ $# -ge 2 && $# -le 3 ]] \
  || die "usage: $0 <gpu-partition> <walltime> [array-concurrency:1-3]"
partition="$1"
walltime="$2"
concurrency="${3:-3}"
[[ "${partition}" = gpu-l20 ]] || die "recovery is locked to gpu-l20"
[[ "${walltime}" =~ ^([0-9]+-)?[0-9]{2}:[0-9]{2}:[0-9]{2}$ ]] \
  || die "walltime must be HH:MM:SS or D-HH:MM:SS"
[[ "${concurrency}" =~ ^[1-3]$ ]] || die "array concurrency must lie in 1..3"

for variable in $(compgen -e); do
  case "${variable}" in
    APPTAINER*|SINGULARITY*) die "unset exported ${variable}; ambient container controls forbidden" ;;
    SBATCH_*) die "unset exported ${variable}; ambient sbatch overrides forbidden" ;;
  esac
done
for name in PRORM_PROJECT_ROOT PRORM_SCRATCH_ROOT PRORM_IMAGE PRORM_IMAGE_SHA256 PRORM_HF_CACHE; do
  [[ -n "${!name:-}" ]] || die "${name} is required"
done
assert_export_safe() {
  local name="$1" value="${!1}"
  [[ "${value}" != *","* && "${value}" != *$'\n'* && "${value}" != *$'\r'* ]] \
    || die "${name} may not contain commas or newlines (unsafe for sbatch --export)"
}
assert_bind_source_safe() {
  local name="$1" value="${!1}"
  [[ "${value}" != *":"* ]] \
    || die "${name} may not contain a colon (unsafe for Apptainer --bind)"
}
ensure_real_subdirectory() {
  local root="$1" relative="$2" current component resolved
  local -a components
  current="${root}"
  IFS=/ read -r -a components <<<"${relative}"
  for component in "${components[@]}"; do
    [[ -n "${component}" && "${component}" != "." && "${component}" != ".." ]] \
      || die "unsafe recovery directory component: ${component}"
    current="${current}/${component}"
    if [[ -e "${current}" || -L "${current}" ]]; then
      [[ -d "${current}" && ! -L "${current}" ]] \
        || die "recovery directory component is not a real directory: ${current}"
    else
      if ! mkdir -- "${current}"; then
        [[ -d "${current}" && ! -L "${current}" ]] \
          || die "could not safely create recovery directory: ${current}"
      fi
    fi
    resolved="$(realpath -e -- "${current}")"
    [[ "${resolved}" = "${current}" ]] \
      || die "recovery directory component is not canonical: ${current}"
  done
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
project_root="$(realpath -e -- "${PRORM_PROJECT_ROOT}")"
scratch_root="$(realpath -e -- "${PRORM_SCRATCH_ROOT}")"
image="$(realpath -e -- "${PRORM_IMAGE}")"
hf_cache="$(realpath -e -- "${PRORM_HF_CACHE}")"
[[ -d "${project_root}" && ! -L "${project_root}" ]] || die "project root must be a real directory"
[[ -d "${scratch_root}" && ! -L "${scratch_root}" ]] || die "scratch root must be a real directory"
[[ -f "${image}" && ! -L "${image}" ]] || die "image must be a regular file"
[[ -d "${hf_cache}" && ! -L "${hf_cache}" ]] || die "HF cache must be a real directory"

overlay_relative="configs/common_beta_recovery_pilot.yaml"
base_relative="configs/common_beta_pilot_base.yaml"
registry_relative="configs/phase2_recovery_parent_failures.json"
identity_relative="configs/identities.json"
job_relative="scripts/hpc4/phase2_recovery_pilot.sbatch"
validator_relative="scripts/hpc4/validate_phase2_recovery_parent.py"
runner_relative="scripts/hpc4/run_phase2_recovery_train.py"
for relative in \
  "${overlay_relative}" "${base_relative}" "${registry_relative}" "${identity_relative}" "${job_relative}" \
  "${validator_relative}" "${runner_relative}" src/smart_reward/phase2_recovery.py; do
  git -C "${repo_root}" ls-files --error-unmatch -- "${relative}" >/dev/null \
    || die "required recovery file is not committed: ${relative}"
done

current_commit="$(git -C "${repo_root}" rev-parse --verify HEAD)"
[[ "${current_commit}" =~ ^[0-9a-f]{40}$ ]] || die "cannot resolve current Git commit"
[[ -z "$(git -C "${repo_root}" status --porcelain --untracked-files=normal)" ]] \
  || die "recovery submission requires a clean committed worktree"
parent_commit="ae28e2a10f0bd5762899be01ce66bc5b423374cf"
diagnostic_commit="791c2daac7f1601f6798d5878bef1770ca9d5ebf"
[[ "${current_commit}" != "${parent_commit}" ]] \
  || die "recovery training commit must differ from the parent artifact producer"
git -C "${repo_root}" merge-base --is-ancestor "${parent_commit}" "${current_commit}" \
  || die "parent artifact producer must be an ancestor of recovery training"
git -C "${repo_root}" merge-base --is-ancestor "${diagnostic_commit}" "${current_commit}" \
  || die "optimizer diagnostic commit must be an ancestor of recovery training"

materialization_paths=(
  configs/common_beta_pilot_base.yaml
  src/smart_reward/annotations.py src/smart_reward/artifacts.py
  src/smart_reward/config.py src/smart_reward/data.py src/smart_reward/hf.py
  src/smart_reward/oracle.py src/smart_reward/phase1.py
  src/smart_reward/phase1_rollout.py src/smart_reward/prompts.py
  src/smart_reward/scores.py src/smart_reward/seeding.py
)
git -C "${repo_root}" diff --quiet "${parent_commit}" "${current_commit}" -- \
  "${materialization_paths[@]}" \
  || die "materialization-relevant blobs changed since the parent artifact producer"

base_hash="81ccbd3bc9d745d3792d1834116ba2480d34a7201d6f4e463a61d8d8ab0baefa"
design_sha="9602b0f00a73880545fd57ce1886ec65d7901385cce2b919fd72f3efec4592d4"
parent_design="0c8820b67b8ca85c23cd5c31b8d25001018b31a7271f6a861d88cfae2f85d7ca"
registry_sha="7be4ee90b1f494d32f96214f407a57cbee54be86a77dacc1206d2acd527857dc"
diagnostic_sha="bd7c3d80c26500ee273b14bb1ea8bc3428f71fdb319a49c792bf4de567e2c6a9"
image_sha="d6fc044b4fa303747908783ea057d5b8946f613bfec6a6ca301e3a02fd7719cb"
inventory_sha="86c7c0fcab9cc0de612c6a5af05778e8b34617822b2e33474df8ed840eef82fd"
overlay_file_sha="a6a924dae429ceb0df11cea128542cae16fb42a2e69a0d2120acb0e4f8f1d80f"
base_file_sha="e32cf5ad2a7bb2f6fa27180aa2fa6e05e2b457cfe032bab1c33f86646af1beb1"
for binding in \
  "${overlay_relative}:${overlay_file_sha}" "${base_relative}:${base_file_sha}" \
  "${registry_relative}:${registry_sha}"; do
  relative="${binding%%:*}"
  expected="${binding#*:}"
  observed_worktree="$(sha256sum -- "${repo_root}/${relative}" | awk '{print $1}')"
  observed_commit="$(
    git -C "${repo_root}" cat-file blob "${current_commit}:${relative}" | sha256sum | awk '{print $1}'
  )"
  [[ "${observed_worktree}" = "${expected}" && "${observed_commit}" = "${expected}" ]] \
    || die "committed/worktree bytes differ from frozen identity: ${relative}"
done
python3 -I -S - \
  "${repo_root}/${identity_relative}" "${design_sha}" "${overlay_file_sha}" \
  "${base_hash}" "${base_file_sha}" <<'PY'
import json, sys
from pathlib import Path
value=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if value.get("schema_version")!="prorm-config-identities/v1":
    raise SystemExit("unsupported config identity registry")
configs=value.get("configs",{})
expected={
 "configs/common_beta_recovery_pilot.yaml":{"config_hash":sys.argv[2],"file_sha256":sys.argv[3],"seed_count":3},
 "configs/common_beta_pilot_base.yaml":{"config_hash":sys.argv[4],"file_sha256":sys.argv[5],"seed_count":3},
}
for path,binding in expected.items():
    if configs.get(path)!=binding:
        raise SystemExit(f"config identity mismatch: {path}")
PY
[[ "${PRORM_IMAGE_SHA256}" = "${image_sha}" ]] || die "image identity differs from the parent"
printf '%s  %s\n' "${image_sha}" "${image}" | sha256sum --check --status \
  || die "image bytes do not match the frozen identity"
inventory="${hf_cache}/inventories/${base_hash}.json"
[[ -f "${inventory}" && ! -L "${inventory}" ]] || die "frozen HF inventory is missing"
printf '%s  %s\n' "${inventory_sha}" "${inventory}" | sha256sum --check --status \
  || die "HF inventory bytes changed"
printf '%s  %s\n' "${registry_sha}" "${repo_root}/${registry_relative}" \
  | sha256sum --check --status || die "tracked parent registry bytes changed"
printf '%s  %s\n' "${diagnostic_sha}" \
  "${project_root}/diagnostics/bt-convergence/seed-20260801-commit-791c2da.json" \
  | sha256sum --check --status || die "optimizer diagnostic bytes changed"

python3 "${repo_root}/${validator_relative}" \
  "${repo_root}/${registry_relative}" --project-root "${project_root}" \
  --expected-registry-sha256 "${registry_sha}" \
  --expected-parent-design-sha256 "${parent_design}" \
  --expected-base-config-hash "${base_hash}" --verify-sources >/dev/null \
  || die "parent failure/artifact/diagnostic verification failed"

for seed in 20260801 20260802 20260803; do
  seed_root="${project_root}/runs/phase2-recovery-pilot/${design_sha}/seed-${seed}"
  [[ ! -e "${seed_root}" && ! -L "${seed_root}" ]] \
    || die "one-shot recovery already has a terminal namespace for seed ${seed}"
done

export PRORM_GIT_COMMIT="${current_commit}"
export PRORM_IMAGE="${image}"
export PRORM_IMAGE_SHA256="${image_sha}"
export PRORM_HF_CACHE="${hf_cache}"
export PRORM_HF_INVENTORY="${inventory}"
export PRORM_HF_INVENTORY_SHA256="${inventory_sha}"
export PRORM_PROJECT_ROOT="${project_root}"
export PRORM_SCRATCH_ROOT="${scratch_root}"
export PRORM_REPO_ROOT="${repo_root}"
export PRORM_PHASE2_RECOVERY_DESIGN_SHA256="${design_sha}"
export PRORM_PHASE2_BASE_CONFIG_HASH="${base_hash}"
export PRORM_PHASE2_PARENT_DESIGN_SHA256="${parent_design}"
export PRORM_PHASE2_PARENT_REGISTRY_SHA256="${registry_sha}"
export PRORM_PHASE2_PARENT_PRODUCER_GIT_COMMIT="${parent_commit}"
export PRORM_PHASE2_DIAGNOSTIC_GIT_COMMIT="${diagnostic_commit}"

for name in \
  PRORM_GIT_COMMIT PRORM_IMAGE PRORM_IMAGE_SHA256 PRORM_HF_CACHE \
  PRORM_HF_INVENTORY PRORM_HF_INVENTORY_SHA256 PRORM_PROJECT_ROOT \
  PRORM_SCRATCH_ROOT PRORM_REPO_ROOT PRORM_PHASE2_RECOVERY_DESIGN_SHA256 \
  PRORM_PHASE2_BASE_CONFIG_HASH PRORM_PHASE2_PARENT_DESIGN_SHA256 \
  PRORM_PHASE2_PARENT_REGISTRY_SHA256 \
  PRORM_PHASE2_PARENT_PRODUCER_GIT_COMMIT PRORM_PHASE2_DIAGNOSTIC_GIT_COMMIT; do
  assert_export_safe "${name}"
done
for name in PRORM_PROJECT_ROOT PRORM_SCRATCH_ROOT PRORM_HF_CACHE; do
  assert_bind_source_safe "${name}"
done
export_spec="PATH=/usr/local/bin:/usr/bin:/bin,PRORM_GIT_COMMIT=${PRORM_GIT_COMMIT},PRORM_IMAGE=${PRORM_IMAGE},PRORM_IMAGE_SHA256=${PRORM_IMAGE_SHA256},PRORM_HF_CACHE=${PRORM_HF_CACHE},PRORM_HF_INVENTORY=${PRORM_HF_INVENTORY},PRORM_HF_INVENTORY_SHA256=${PRORM_HF_INVENTORY_SHA256},PRORM_PROJECT_ROOT=${PRORM_PROJECT_ROOT},PRORM_SCRATCH_ROOT=${PRORM_SCRATCH_ROOT},PRORM_REPO_ROOT=${PRORM_REPO_ROOT},PRORM_PHASE2_RECOVERY_DESIGN_SHA256=${PRORM_PHASE2_RECOVERY_DESIGN_SHA256},PRORM_PHASE2_BASE_CONFIG_HASH=${PRORM_PHASE2_BASE_CONFIG_HASH},PRORM_PHASE2_PARENT_DESIGN_SHA256=${PRORM_PHASE2_PARENT_DESIGN_SHA256},PRORM_PHASE2_PARENT_REGISTRY_SHA256=${PRORM_PHASE2_PARENT_REGISTRY_SHA256},PRORM_PHASE2_PARENT_PRODUCER_GIT_COMMIT=${PRORM_PHASE2_PARENT_PRODUCER_GIT_COMMIT},PRORM_PHASE2_DIAGNOSTIC_GIT_COMMIT=${PRORM_PHASE2_DIAGNOSTIC_GIT_COMMIT}"
ensure_real_subdirectory "${scratch_root}" "phase2-recovery-jobs"
ensure_real_subdirectory \
  "${project_root}" "runs/phase2-recovery-pilot/${design_sha}"
log_dir="${project_root}/slurm-logs/phase2-recovery-pilot/${design_sha}"
ensure_real_subdirectory \
  "${project_root}" "slurm-logs/phase2-recovery-pilot/${design_sha}"
log_dir="$(realpath -e -- "${log_dir}")"
[[ -d "${log_dir}" && ! -L "${log_dir}" ]] || die "recovery log directory is unsafe"
case "${log_dir}" in
  "${project_root}"/slurm-logs/phase2-recovery-pilot/"${design_sha}") ;;
  *) die "recovery log directory escaped the project root" ;;
esac
assert_export_safe log_dir
assert_bind_source_safe log_dir

sbatch \
  --parsable \
  --account=sigroup --partition="${partition}" --time="${walltime}" \
  --chdir="${repo_root}" \
  --array="0-2%${concurrency}" \
  --output="${log_dir}/%x-%A_%a.out" \
  --error="${log_dir}/%x-%A_%a.err" \
  --export="${export_spec}" \
  "${repo_root}/${job_relative}"
