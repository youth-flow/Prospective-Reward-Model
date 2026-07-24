#!/usr/bin/env bash
set -euo pipefail

die() {
  echo "error: $*" >&2
  exit 2
}

if [[ $# -lt 8 ]]; then
  die "usage: $0 <overlay.yaml> <base.yaml> <output.json> <cpu-partition> <walltime> <run-dir-1> <run-dir-2> <run-dir-3> [--producer-commit <full-commit>] [--beta-source-aggregate <json>] [--horizon-parent-aggregate <json>]"
fi

overlay_input="$1"
base_input="$2"
output_input="$3"
partition="$4"
walltime="$5"
run_inputs=("$6" "$7" "$8")
shift 8
beta_source_input=""
horizon_parent_input=""
producer_commit_input=""
while (( $# )); do
  case "$1" in
    --producer-commit)
      [[ $# -ge 2 && -z "${producer_commit_input}" ]] \
        || die "--producer-commit requires exactly one full commit"
      producer_commit_input="$2"
      shift 2
      ;;
    --beta-source-aggregate)
      [[ $# -ge 2 && -z "${beta_source_input}" ]] \
        || die "--beta-source-aggregate requires exactly one path"
      beta_source_input="$2"
      shift 2
      ;;
    --horizon-parent-aggregate)
      [[ $# -ge 2 && -z "${horizon_parent_input}" ]] \
        || die "--horizon-parent-aggregate requires exactly one path"
      horizon_parent_input="$2"
      shift 2
      ;;
    *) die "unknown pilot aggregate option: $1" ;;
  esac
done

case "${partition}" in
  amd|intel) ;;
  *) die "pilot aggregation partition must be amd or intel" ;;
esac
[[ "${walltime}" =~ ^[0-9]+-[0-9]{2}:[0-9]{2}:[0-9]{2}$|^[0-9]{2}:[0-9]{2}:[0-9]{2}$ ]] \
  || die "walltime must be HH:MM:SS or D-HH:MM:SS"

while IFS= read -r variable; do
  case "${variable}" in
    APPTAINER*|SINGULARITY*)
      die "unset exported ${variable}; pilot aggregate submission forbids container controls"
      ;;
    SBATCH_*)
      die "unset exported ${variable}; pilot aggregate submission forbids sbatch overrides"
      ;;
  esac
done < <(compgen -e)
for name in \
  PRORM_PROJECT_ROOT PRORM_SCRATCH_ROOT PRORM_IMAGE PRORM_IMAGE_SHA256 \
  PRORM_HF_CACHE; do
  [[ -n "${!name:-}" ]] || die "${name} is required"
done
for command_name in git python3 realpath sbatch sha256sum awk grep; do
  command -v "${command_name}" >/dev/null 2>&1 \
    || die "required command is unavailable: ${command_name}"
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
aggregator_git_commit="$(git -C "${repo_root}" rev-parse --verify HEAD)"
[[ "${aggregator_git_commit}" =~ ^[0-9a-f]{40,64}$ ]] || die "invalid Git HEAD"
[[ -z "$(git -C "${repo_root}" status --porcelain --untracked-files=normal)" ]] \
  || die "pilot aggregate submission requires a clean committed worktree"
if [[ -n "${producer_commit_input}" ]]; then
  [[ "${producer_commit_input}" =~ ^[0-9a-f]{40,64}$ ]] \
    || die "--producer-commit must be a full lowercase Git object ID"
  producer_git_commit="$(
    git -C "${repo_root}" rev-parse --verify "${producer_commit_input}^{commit}"
  )" || die "producer commit cannot be resolved"
  [[ "${producer_git_commit}" = "${producer_commit_input}" ]] \
    || die "--producer-commit must be the exact full commit ID"
else
  producer_git_commit="${aggregator_git_commit}"
fi
git -C "${repo_root}" merge-base --is-ancestor \
  "${producer_git_commit}" "${aggregator_git_commit}" \
  || die "producer commit must be an ancestor of the aggregation commit"

overlay="$(realpath -e -- "${overlay_input}")" || die "overlay cannot be resolved"
base_config="$(realpath -e -- "${base_input}")" || die "base config cannot be resolved"
for path in "${overlay}" "${base_config}"; do
  [[ -f "${path}" && ! -L "${path}" ]] || die "config must be a regular non-symlink file"
  case "${path}" in
    "${repo_root}"/configs/*.yaml) ;;
    *) die "pilot aggregate configs must be tracked configs/*.yaml files" ;;
  esac
done
overlay_relative="$(realpath --relative-to="${repo_root}" "${overlay}")"
base_relative="$(realpath --relative-to="${repo_root}" "${base_config}")"
[[ "${overlay_relative}" = "configs/common_beta_pilot.yaml" ]] \
  || die "entry accepts only configs/common_beta_pilot.yaml"
[[ "${base_relative}" = "configs/common_beta_pilot_base.yaml" ]] \
  || die "entry accepts only configs/common_beta_pilot_base.yaml"
identity_relative="configs/identities.json"
identity_path="${repo_root}/${identity_relative}"
[[ -f "${identity_path}" && ! -L "${identity_path}" ]] \
  || die "configs/identities.json is missing or unsafe"

overlay_sha="$(sha256sum -- "${overlay}" | awk '{print $1}')"
base_sha="$(sha256sum -- "${base_config}" | awk '{print $1}')"
identity_sha="$(sha256sum -- "${identity_path}" | awk '{print $1}')"
for binding in \
  "${overlay_relative}:${overlay_sha}" \
  "${base_relative}:${base_sha}" \
  "${identity_relative}:${identity_sha}"; do
  relative="${binding%%:*}"
  expected="${binding#*:}"
  observed="$(
    git -C "${repo_root}" cat-file blob \
      "${aggregator_git_commit}:${relative}" | sha256sum | awk '{print $1}'
  )"
  [[ "${observed}" = "${expected}" ]] \
    || die "worktree bytes differ from committed input: ${relative}"
  producer_observed="$(
    git -C "${repo_root}" cat-file blob \
      "${producer_git_commit}:${relative}" | sha256sum | awk '{print $1}'
  )"
  [[ "${producer_observed}" = "${expected}" ]] \
    || die "producer and aggregator commits do not bind identical input: ${relative}"
done
validator_relative="src/smart_reward/phase2_pilot_aggregate.py"
validator_path="${repo_root}/${validator_relative}"
[[ -f "${validator_path}" && ! -L "${validator_path}" ]] \
  || die "pilot aggregate validator source is missing or unsafe"
validator_source_sha="$(sha256sum -- "${validator_path}" | awk '{print $1}')"
validator_committed_sha="$(
  git -C "${repo_root}" cat-file blob \
    "${aggregator_git_commit}:${validator_relative}" | sha256sum | awk '{print $1}'
)"
[[ "${validator_source_sha}" = "${validator_committed_sha}" ]] \
  || die "pilot aggregate validator source differs from the aggregation commit"

mapfile -t identities < <(
  python3 -I -S - \
    "${identity_path}" "${overlay_relative}" "${base_relative}" \
    "${overlay_sha}" "${base_sha}" <<'PY'
import json
import re
import sys
from pathlib import Path


identity_path, overlay, base, overlay_sha, base_sha = sys.argv[1:]
value = json.loads(Path(identity_path).read_text(encoding="utf-8"))
if value.get("schema_version") != "prorm-config-identities/v1":
    raise SystemExit("invalid identity schema")
configs = value.get("configs")
if not isinstance(configs, dict):
    raise SystemExit("identity config map is missing")


def entry(path, file_sha):
    item = configs.get(path)
    if not isinstance(item, dict) or item.get("file_sha256") != file_sha:
        raise SystemExit(f"identity file binding failed: {path}")
    digest = item.get("config_hash")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise SystemExit(f"invalid semantic identity: {path}")
    if item.get("seed_count") != 3:
        raise SystemExit(f"pilot aggregate requires exactly three seeds: {path}")
    return digest


print(entry(overlay, overlay_sha))
print(entry(base, base_sha))
PY
)
[[ "${#identities[@]}" -eq 2 ]] || die "cannot resolve committed pilot identities"
design_sha="${identities[0]}"
base_hash="${identities[1]}"
[[ "${design_sha}" != "${base_hash}" ]] || die "design and base identities must differ"

canonical_root() {
  local name="$1" raw="${!1}" resolved=""
  [[ "${raw}" = /* ]] || die "${name} must be absolute"
  resolved="$(realpath -e -- "${raw}")" || die "${name} cannot be resolved"
  [[ -d "${resolved}" && "${resolved}" != "/" ]] || die "${name} must be a non-root directory"
  printf '%s\n' "${resolved}"
}

resolve_project_path() {
  local raw="$1" kind="$2" candidate="" resolved=""
  if [[ "${raw}" = /* ]]; then candidate="${raw}"; else candidate="${project_root}/${raw}"; fi
  resolved="$(realpath -e -- "${candidate}")" || die "project path cannot be resolved: ${raw}"
  case "${resolved}" in
    "${project_root}"/*) ;;
    *) die "project path escaped PRORM_PROJECT_ROOT: ${raw}" ;;
  esac
  case "${kind}" in
    file) [[ -f "${resolved}" ]] || die "project path is not a file: ${resolved}" ;;
    directory) [[ -d "${resolved}" ]] || die "project path is not a directory: ${resolved}" ;;
    *) die "invalid internal project path kind" ;;
  esac
  printf '%s\n' "${resolved}"
}

reject_delimiters() {
  local value="$1"
  [[ "${value}" != *","* && "${value}" != *":"* && "${value}" != *"="* \
    && "${value}" != *$'\n'* && "${value}" != *$'\r'* ]] \
    || die "path contains an unsafe sbatch/export or bind delimiter"
}

project_root="$(canonical_root PRORM_PROJECT_ROOT)"
scratch_root="$(canonical_root PRORM_SCRATCH_ROOT)"
case "${project_root}" in
  "${scratch_root}"|"${scratch_root}"/*) die "project and scratch roots overlap" ;;
esac
case "${scratch_root}" in
  "${project_root}"|"${project_root}"/*) die "project and scratch roots overlap" ;;
esac
image="$(resolve_project_path "${PRORM_IMAGE}" file)"
hf_cache="$(resolve_project_path "${PRORM_HF_CACHE}" directory)"
[[ "${PRORM_IMAGE_SHA256}" =~ ^[0-9a-f]{64}$ ]] \
  || die "PRORM_IMAGE_SHA256 must be lowercase SHA256"
printf '%s  %s\n' "${PRORM_IMAGE_SHA256}" "${image}" \
  | sha256sum --check --status || die "image SHA256 mismatch"
inventory="$(realpath -e -- "${hf_cache}/inventories/${base_hash}.json")" \
  || die "base-identity HF inventory is missing"
[[ -f "${inventory}" && ! -L "${inventory}" ]] || die "HF inventory is unsafe"
inventory_sha="$(sha256sum -- "${inventory}" | awk '{print $1}')"

output_candidate="${output_input}"
if [[ "${output_candidate}" != /* ]]; then output_candidate="${project_root}/${output_candidate}"; fi
output_parent="$(realpath -e -- "$(dirname "${output_candidate}")")" \
  || die "aggregate output parent does not exist"
output="${output_parent}/$(basename "${output_candidate}")"
case "${output}" in
  "${project_root}"/*) ;;
  *) die "aggregate output escaped PRORM_PROJECT_ROOT" ;;
esac
[[ ! -e "${output}" && ! -L "${output}" ]] || die "refusing to overwrite pilot aggregate"
[[ -w "${output_parent}" ]] || die "aggregate output parent is not writable"

run_dirs=()
result_hashes=()
sidecar_hashes=()
marker_hashes=()
manifest_hashes=()
output_verification_hashes=()
artifact_metadata_hashes=()
for index in 0 1 2; do
  raw="${run_inputs[$index]}"
  if [[ "${raw}" != /* ]]; then raw="${project_root}/${raw}"; fi
  run_dir="$(realpath -e -- "${raw}")" || die "pilot run directory is missing"
  [[ -d "${run_dir}" && ! -L "${run_dir}" ]] || die "pilot run directory is unsafe"
  case "${run_dir}" in
    "${project_root}"/runs/phase2-pilot/"${design_sha}"/*) ;;
    *) die "pilot run directory is outside the bound design identity" ;;
  esac
  marker="${run_dir}/SUCCESS"
  sidecar="${run_dir}/phase2-pilot-diagnostics.diagnostics.jsonl"
  result="${run_dir}/phase2-pilot-diagnostics.json"
  manifest="${run_dir}/run-manifest.json"
  output_verification="${run_dir}/phase2-output-verification.json"
  artifact_link="${run_dir}/artifact"
  [[ -L "${artifact_link}" && -d "${artifact_link}" ]] \
    || die "pilot SUCCESS run lacks its bound artifact symlink"
  artifact_metadata="${artifact_link}/metadata.json"
  for path in \
    "${marker}" "${result}" "${sidecar}" "${manifest}" \
    "${output_verification}" "${artifact_metadata}"; do
    [[ -f "${path}" && ! -L "${path}" ]] || die "pilot SUCCESS run is incomplete"
  done
  grep -Fx 'status=SUCCESS' "${marker}" >/dev/null || die "run marker is not SUCCESS"
  grep -Fx "phase2_design_sha256=${design_sha}" "${marker}" >/dev/null \
    || die "SUCCESS marker design identity mismatch"
  grep -Fx "base_config_hash=${base_hash}" "${marker}" >/dev/null \
    || die "SUCCESS marker base identity mismatch"
  grep -Fx "git_commit=${producer_git_commit}" "${marker}" >/dev/null \
    || die "SUCCESS marker Git identity mismatch"
  run_dirs+=("${run_dir}")
  marker_hashes+=("$(sha256sum -- "${marker}" | awk '{print $1}')")
  result_hashes+=("$(sha256sum -- "${result}" | awk '{print $1}')")
  sidecar_hashes+=("$(sha256sum -- "${sidecar}" | awk '{print $1}')")
  manifest_hashes+=("$(sha256sum -- "${manifest}" | awk '{print $1}')")
  output_verification_hashes+=(
    "$(sha256sum -- "${output_verification}" | awk '{print $1}')"
  )
  artifact_metadata_hashes+=(
    "$(sha256sum -- "${artifact_metadata}" | awk '{print $1}')"
  )
done
[[ "${run_dirs[0]}" != "${run_dirs[1]}" \
  && "${run_dirs[0]}" != "${run_dirs[2]}" \
  && "${run_dirs[1]}" != "${run_dirs[2]}" ]] \
  || die "pilot aggregate run directories must be distinct"

beta_present=0
beta_source=""
beta_sha=""
if [[ -n "${beta_source_input}" ]]; then
  beta_source="$(resolve_project_path "${beta_source_input}" file)"
  beta_sha="$(sha256sum -- "${beta_source}" | awk '{print $1}')"
  beta_present=1
fi
horizon_present=0
horizon_parent=""
horizon_sha=""
if [[ -n "${horizon_parent_input}" ]]; then
  horizon_parent="$(resolve_project_path "${horizon_parent_input}" file)"
  horizon_sha="$(sha256sum -- "${horizon_parent}" | awk '{print $1}')"
  horizon_present=1
fi

for value in \
  "${project_root}" "${scratch_root}" "${repo_root}" "${overlay}" "${base_config}" \
  "${image}" "${hf_cache}" "${inventory}" "${output}" "${run_dirs[@]}" \
  "${beta_source}" "${horizon_parent}"; do
  reject_delimiters "${value}"
done

[[ "$(git -C "${repo_root}" rev-parse --verify HEAD)" = "${aggregator_git_commit}" \
  && -z "$(git -C "${repo_root}" status --porcelain --untracked-files=normal)" ]] \
  || die "submission checkout changed during validation"
printf '%s  %s\n' "${PRORM_IMAGE_SHA256}" "${image}" \
  | sha256sum --check --status || die "image changed before submission"
printf '%s  %s\n' "${inventory_sha}" "${inventory}" \
  | sha256sum --check --status || die "inventory changed before submission"
for index in 0 1 2; do
  run_dir="${run_dirs[$index]}"
  printf '%s  %s\n' \
    "${result_hashes[$index]}" \
    "${run_dir}/phase2-pilot-diagnostics.json" \
    | sha256sum --check --status \
    || die "pilot result changed before submission"
  printf '%s  %s\n' \
    "${sidecar_hashes[$index]}" \
    "${run_dir}/phase2-pilot-diagnostics.diagnostics.jsonl" \
    | sha256sum --check --status \
    || die "pilot sidecar changed before submission"
  printf '%s  %s\n' \
    "${marker_hashes[$index]}" \
    "${run_dir}/SUCCESS" \
    | sha256sum --check --status \
    || die "pilot SUCCESS marker changed before submission"
  printf '%s  %s\n' \
    "${manifest_hashes[$index]}" \
    "${run_dir}/run-manifest.json" \
    | sha256sum --check --status \
    || die "pilot run manifest changed before submission"
  printf '%s  %s\n' \
    "${output_verification_hashes[$index]}" \
    "${run_dir}/phase2-output-verification.json" \
    | sha256sum --check --status \
    || die "pilot output verification changed before submission"
  printf '%s  %s\n' \
    "${artifact_metadata_hashes[$index]}" \
    "${run_dir}/artifact/metadata.json" \
    | sha256sum --check --status \
    || die "pilot artifact metadata changed before submission"
done
if (( beta_present )); then
  printf '%s  %s\n' "${beta_sha}" "${beta_source}" \
    | sha256sum --check --status \
    || die "beta-source aggregate changed before submission"
fi
if (( horizon_present )); then
  printf '%s  %s\n' "${horizon_sha}" "${horizon_parent}" \
    | sha256sum --check --status \
    || die "horizon-parent aggregate changed before submission"
fi
for binding in \
  "${overlay}:${overlay_sha}" \
  "${base_config}:${base_sha}" \
  "${identity_path}:${identity_sha}"; do
  path="${binding%%:*}"
  expected="${binding#*:}"
  printf '%s  %s\n' "${expected}" "${path}" \
    | sha256sum --check --status \
    || die "committed identity input changed before submission"
done

export_spec="PATH=/usr/local/bin:/usr/bin:/bin,PRORM_PROJECT_ROOT=${project_root},PRORM_SCRATCH_ROOT=${scratch_root},PRORM_IMAGE=${image},PRORM_IMAGE_SHA256=${PRORM_IMAGE_SHA256},PRORM_HF_CACHE=${hf_cache},PRORM_HF_INVENTORY=${inventory},PRORM_HF_INVENTORY_SHA256=${inventory_sha},PRORM_REPO_ROOT=${repo_root},PRORM_PHASE2_OVERLAY_REL=${overlay_relative},PRORM_PHASE2_BASE_REL=${base_relative},PRORM_PHASE2_OVERLAY_FILE_SHA256=${overlay_sha},PRORM_PHASE2_BASE_FILE_SHA256=${base_sha},PRORM_IDENTITIES_FILE_SHA256=${identity_sha},PRORM_PHASE2_DESIGN_SHA256=${design_sha},PRORM_PHASE2_BASE_CONFIG_HASH=${base_hash},PRORM_PHASE2_AGGREGATOR_GIT_COMMIT=${aggregator_git_commit},PRORM_PHASE2_PRODUCER_GIT_COMMIT=${producer_git_commit},PRORM_PHASE2_AGGREGATE_VALIDATOR_SOURCE_SHA256=${validator_source_sha},PRORM_PHASE2_AGGREGATE_OUTPUT=${output},PRORM_PHASE2_BETA_SOURCE_AGGREGATE_PRESENT=${beta_present},PRORM_PHASE2_HORIZON_PARENT_AGGREGATE_PRESENT=${horizon_present}"
for index in 0 1 2; do
  export_spec+=",PRORM_PHASE2_RUN_DIR_${index}=${run_dirs[$index]},PRORM_PHASE2_RESULT_SHA256_${index}=${result_hashes[$index]},PRORM_PHASE2_SIDECAR_SHA256_${index}=${sidecar_hashes[$index]},PRORM_PHASE2_SUCCESS_SHA256_${index}=${marker_hashes[$index]},PRORM_PHASE2_MANIFEST_SHA256_${index}=${manifest_hashes[$index]},PRORM_PHASE2_OUTPUT_VERIFICATION_SHA256_${index}=${output_verification_hashes[$index]},PRORM_PHASE2_ARTIFACT_METADATA_SHA256_${index}=${artifact_metadata_hashes[$index]}"
done
if (( beta_present )); then
  export_spec+=",PRORM_PHASE2_BETA_SOURCE_AGGREGATE=${beta_source},PRORM_PHASE2_BETA_SOURCE_AGGREGATE_SHA256=${beta_sha}"
fi
if (( horizon_present )); then
  export_spec+=",PRORM_PHASE2_HORIZON_PARENT_AGGREGATE=${horizon_parent},PRORM_PHASE2_HORIZON_PARENT_AGGREGATE_SHA256=${horizon_sha}"
fi

slurm_log_dir="${project_root}/slurm-logs"
mkdir -p "${slurm_log_dir}" "${scratch_root}/phase2-pilot-aggregate-jobs"
sbatch \
  --parsable \
  --account=sigroup \
  --job-name=prorm-phase2-pilot-aggregate \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=1 \
  --mem=8G \
  --chdir="${repo_root}" \
  --partition="${partition}" \
  --time="${walltime}" \
  --output="${slurm_log_dir}/%x-%j.out" \
  --export="${export_spec}" \
  "${repo_root}/scripts/hpc4/phase2_pilot_aggregate.sbatch"
