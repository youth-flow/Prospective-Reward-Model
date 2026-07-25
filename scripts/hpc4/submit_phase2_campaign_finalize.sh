#!/usr/bin/env bash
set -euo pipefail

die() {
  echo "error: $*" >&2
  exit 2
}

expected_terminal_count=30
expected_argument_count=6
if [[ "$#" -ne "${expected_argument_count}" ]]; then
  die "usage: $0 <confirmatory-overlay.yaml> <base.yaml> <campaign-terminal.json> <primary-aggregate.json> <cpu-partition> <walltime>"
fi

overlay_input="$1"
base_input="$2"
terminal_output_input="$3"
aggregate_output_input="$4"
partition="$5"
walltime="$6"
shift 6
terminal_inputs=()

case "${partition}" in
  amd|intel) ;;
  *) die "campaign finalization partition must be amd or intel" ;;
esac
[[ "${walltime}" =~ ^[0-9]+-[0-9]{2}:[0-9]{2}:[0-9]{2}$|^[0-9]{2}:[0-9]{2}:[0-9]{2}$ ]] \
  || die "walltime must be HH:MM:SS or D-HH:MM:SS"

while IFS= read -r variable; do
  case "${variable}" in
    APPTAINER*|SINGULARITY*)
      die "unset exported ${variable}; campaign finalization submission forbids container controls"
      ;;
    SBATCH_*)
      die "unset exported ${variable}; campaign finalization submission forbids sbatch overrides"
      ;;
  esac
done < <(compgen -e)

for name in \
  PRORM_PROJECT_ROOT PRORM_SCRATCH_ROOT PRORM_IMAGE PRORM_IMAGE_SHA256 \
  PRORM_HF_CACHE; do
  [[ -n "${!name:-}" ]] || die "${name} is required"
done
for command_name in \
  git python3 realpath sbatch sha256sum awk grep dirname basename; do
  command -v "${command_name}" >/dev/null 2>&1 \
    || die "required command is unavailable: ${command_name}"
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
git_commit="$(git -C "${repo_root}" rev-parse --verify HEAD)"
[[ "${git_commit}" =~ ^[0-9a-f]{40,64}$ ]] || die "invalid Git HEAD"
[[ -z "$(git -C "${repo_root}" status --porcelain --untracked-files=normal)" ]] \
  || die "campaign finalization submission requires a clean committed worktree"

canonical_root() {
  local name="$1" raw="${!1}" resolved=""
  [[ "${raw}" = /* ]] || die "${name} must be absolute"
  resolved="$(realpath -e -- "${raw}")" || die "${name} cannot be resolved"
  [[ -d "${resolved}" && "${resolved}" != "/" && "${resolved}" = "${raw}" ]] \
    || die "${name} must be a canonical non-root directory"
  printf '%s\n' "${resolved}"
}

resolve_project_path() {
  local raw="$1" kind="$2" candidate="" resolved=""
  if [[ "${raw}" = /* ]]; then
    candidate="${raw}"
  else
    candidate="${project_root}/${raw}"
  fi
  [[ ! -L "${candidate}" ]] || die "project path must not be a symbolic link: ${raw}"
  resolved="$(realpath -e -- "${candidate}")" \
    || die "project path cannot be resolved: ${raw}"
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
[[ -w "${project_root}" && -w "${scratch_root}" ]] \
  || die "project and scratch roots must be writable"

overlay="$(realpath -e -- "${overlay_input}")" \
  || die "confirmatory overlay cannot be resolved"
base_config="$(realpath -e -- "${base_input}")" \
  || die "confirmatory base config cannot be resolved"
for path in "${overlay}" "${base_config}"; do
  [[ -f "${path}" && ! -L "${path}" ]] \
    || die "config must be a regular non-symlink file"
  case "${path}" in
    "${repo_root}"/configs/*.yaml) ;;
    *) die "campaign finalization configs must be tracked configs/*.yaml files" ;;
  esac
done
overlay_relative="$(realpath --relative-to="${repo_root}" "${overlay}")"
base_relative="$(realpath --relative-to="${repo_root}" "${base_config}")"
for relative in "${overlay_relative}" "${base_relative}"; do
  [[ "${relative}" =~ ^configs/[A-Za-z0-9._-]+\.yaml$ ]] \
    || die "config has an unsafe repository-relative path: ${relative}"
  git -C "${repo_root}" ls-files --error-unmatch -- "${relative}" >/dev/null \
    || die "config is not tracked by Git: ${relative}"
done
[[ "${overlay_relative}" != "${base_relative}" ]] \
  || die "confirmatory overlay and base config must be distinct"

identity_relative="configs/identities.json"
identity_path="${repo_root}/${identity_relative}"
[[ -f "${identity_path}" && ! -L "${identity_path}" ]] \
  || die "configs/identities.json is missing or unsafe"
git -C "${repo_root}" ls-files --error-unmatch -- "${identity_relative}" >/dev/null \
  || die "configs/identities.json is not tracked"

overlay_file_sha256="$(sha256sum -- "${overlay}" | awk '{print $1}')"
base_file_sha256="$(sha256sum -- "${base_config}" | awk '{print $1}')"
identity_file_sha256="$(sha256sum -- "${identity_path}" | awk '{print $1}')"
for binding in \
  "${overlay_relative}:${overlay_file_sha256}" \
  "${base_relative}:${base_file_sha256}" \
  "${identity_relative}:${identity_file_sha256}"; do
  relative="${binding%%:*}"
  expected="${binding#*:}"
  observed="$(
    git -C "${repo_root}" cat-file blob \
      "${git_commit}:${relative}" | sha256sum | awk '{print $1}'
  )"
  [[ "${observed}" = "${expected}" ]] \
    || die "worktree bytes differ from committed exact source: ${relative}"
done

identity_output="$(
  python3 -I -S - \
    "${identity_path}" "${overlay_relative}" "${base_relative}" \
    "${overlay_file_sha256}" "${base_file_sha256}" <<'PY'
import json
import re
import sys
from pathlib import Path


def reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


identity_path, overlay, base, overlay_sha, base_sha = sys.argv[1:]
payload = json.loads(
    Path(identity_path).read_text(encoding="utf-8"),
    object_pairs_hook=reject_duplicates,
    parse_constant=lambda value: (_ for _ in ()).throw(
        ValueError(f"non-finite JSON constant: {value}")
    ),
)
if payload.get("schema_version") != "prorm-config-identities/v1":
    raise SystemExit("invalid config identity schema")
configs = payload.get("configs")
if not isinstance(configs, dict):
    raise SystemExit("identity config map is missing")


def entry(path, file_sha):
    value = configs.get(path)
    if not isinstance(value, dict) or set(value) != {
        "config_hash",
        "file_sha256",
        "seed_count",
    }:
        raise SystemExit(f"invalid identity entry: {path}")
    if value["file_sha256"] != file_sha:
        raise SystemExit(f"identity file binding failed: {path}")
    if value["seed_count"] != 30:
        raise SystemExit(f"formal campaign requires exactly 30 seeds: {path}")
    digest = value["config_hash"]
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise SystemExit(f"invalid semantic identity: {path}")
    return digest


print(entry(overlay, overlay_sha))
print(entry(base, base_sha))
PY
)" || die "cannot resolve committed confirmatory identities"
mapfile -t identities <<< "${identity_output}"
[[ "${#identities[@]}" -eq 2 ]] || die "cannot resolve committed confirmatory identities"
design_sha256="${identities[0]}"
base_config_hash="${identities[1]}"
[[ "${design_sha256}" != "${base_config_hash}" ]] \
  || die "confirmatory design and base identities must differ"

image="$(resolve_project_path "${PRORM_IMAGE}" file)"
hf_cache="$(resolve_project_path "${PRORM_HF_CACHE}" directory)"
[[ "${PRORM_IMAGE_SHA256}" =~ ^[0-9a-f]{64}$ ]] \
  || die "PRORM_IMAGE_SHA256 must be lowercase SHA256"
printf '%s  %s\n' "${PRORM_IMAGE_SHA256}" "${image}" \
  | sha256sum --check --status || die "image SHA256 mismatch"
inventory_expected="${hf_cache}/inventories/${base_config_hash}.json"
inventory="$(realpath -e -- "${inventory_expected}")" \
  || die "base-identity HF inventory is missing"
[[ "${inventory}" = "${inventory_expected}" && -f "${inventory}" && ! -L "${inventory}" ]] \
  || die "HF inventory is unsafe or not addressed by the base identity"
inventory_sha256="$(sha256sum -- "${inventory}" | awk '{print $1}')"
[[ "${inventory_sha256}" =~ ^[0-9a-f]{64}$ ]] \
  || die "HF inventory SHA256 is invalid"

campaign_root_expected="${project_root}/runs/phase2-confirmatory/${design_sha256}"
campaign_root="$(realpath -e -- "${campaign_root_expected}")" \
  || die "confirmatory design run root does not exist"
[[ "${campaign_root}" = "${campaign_root_expected}" \
  && -d "${campaign_root}" && ! -L "${campaign_root_expected}" ]] \
  || die "confirmatory design run root is unsafe"

normalize_output() {
  local raw="$1" candidate=""
  if [[ "${raw}" = /* ]]; then
    candidate="${raw}"
  else
    candidate="${project_root}/${raw}"
  fi
  realpath -m -- "${candidate}"
}

terminal_output="$(normalize_output "${terminal_output_input}")"
aggregate_output="$(normalize_output "${aggregate_output_input}")"
terminal_output_parent="$(dirname -- "${terminal_output}")"
aggregate_output_parent="$(dirname -- "${aggregate_output}")"
[[ "${terminal_output_parent}" = "${aggregate_output_parent}" ]] \
  || die "terminal and aggregate outputs must share one new publication directory"
output_dir="${terminal_output_parent}"
[[ "${output_dir}" = "${campaign_root}/campaign-final" ]] \
  || die "campaign publication is locked to the one-shot campaign-final directory"
terminal_output_name="$(basename -- "${terminal_output}")"
aggregate_output_name="$(basename -- "${aggregate_output}")"
[[ "${terminal_output_name}" = "phase2-campaign-terminal.json" \
  && "${aggregate_output_name}" = "phase2-primary-aggregate.json" ]] \
  || die "campaign output filenames are locked by the formal contract"
[[ ! -e "${output_dir}" && ! -L "${output_dir}" ]] \
  || die "refusing to overwrite campaign publication directory"
[[ -w "${campaign_root}" ]] || die "campaign design run root is not writable"

validate_terminal() {
  local terminal="$1" expected_seed="$2"
  python3 -I -S \
    "${repo_root}/scripts/hpc4/validate_phase2_terminal.py" \
    "${terminal}" "${expected_seed}" "${design_sha256}" "${base_config_hash}" \
    "${git_commit}" "${PRORM_IMAGE_SHA256}" "${inventory_sha256}"
}

read_terminal_seed() {
  local terminal="$1"
  python3 -I -S - "${terminal}" <<'PY'
import json
import sys
from pathlib import Path


def reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


value = json.loads(
    Path(sys.argv[1]).read_text(encoding="utf-8"),
    object_pairs_hook=reject_duplicates,
    parse_constant=lambda item: (_ for _ in ()).throw(
        ValueError(f"non-finite JSON constant: {item}")
    ),
)
seed = value.get("seed") if isinstance(value, dict) else None
if isinstance(seed, bool) or not isinstance(seed, int) or not 20260901 <= seed <= 20260930:
    raise SystemExit("terminal manifest seed is outside 20260901..20260930")
print(seed)
PY
}

mapfile -t terminal_inputs < <(
  python3 -I -S \
    "${repo_root}/scripts/hpc4/resolve_phase2_campaign_registry.py" \
    "${campaign_root}" "${design_sha256}" "${base_config_hash}" \
    "${git_commit}" "${PRORM_IMAGE_SHA256}" "${inventory_sha256}"
)
[[ "${#terminal_inputs[@]}" -eq "${expected_terminal_count}" ]] \
  || die "campaign registry did not resolve exactly 30 terminal heads"
campaign_plan="${campaign_root}/campaign-registry/campaign-plan.json"
campaign_plan="$(resolve_project_path "${campaign_plan}" file)"
campaign_plan_sha256="$(sha256sum -- "${campaign_plan}" | awk '{print $1}')"
[[ "${campaign_plan_sha256}" =~ ^[0-9a-f]{64}$ ]] \
  || die "fixed-wave campaign plan SHA256 is invalid"

terminals=()
terminal_sha256s=()
markers=()
marker_sha256s=()
source_bindings=()
source_bindings+=("${campaign_plan}:${campaign_plan_sha256}")
accepted_freeze_aggregate_sha256=""
declare -A seen_terminal_paths=()
declare -A seen_terminal_seeds=()
for input_index in {0..29}; do
  terminal="$(resolve_project_path "${terminal_inputs[$input_index]}" file)"
  seed="$(read_terminal_seed "${terminal}")" \
    || die "terminal input ${input_index} has an invalid declared seed"
  [[ "${seed}" =~ ^202609(0[1-9]|[12][0-9]|30)$ ]] \
    || die "terminal input ${input_index} is outside the exact-30 seed set"
  [[ ! -v "seen_terminal_seeds[${seed}]" ]] \
    || die "duplicate terminal manifest for seed ${seed}"
  seen_terminal_seeds["${seed}"]=1
  index=$((seed - 20260901))
  case "${terminal}" in
    "${campaign_root}/seed-${seed}/"*) ;;
    *) die "terminal ${seed} is outside its bound confirmatory seed directory" ;;
  esac
  [[ ! -v "seen_terminal_paths[${terminal}]" ]] \
    || die "duplicate terminal manifest path: ${terminal}"
  seen_terminal_paths["${terminal}"]=1
  validator_output="$(validate_terminal "${terminal}" "${seed}")" \
    || die "terminal ${seed} failed manifest/marker/ledger validation"
  mapfile -t terminal_info <<< "${validator_output}"
  [[ "${#terminal_info[@]}" -eq 8 ]] \
    || die "terminal ${seed} validator returned an invalid binding"
  marker="${terminal_info[0]}"
  ledger_path="${terminal_info[1]}"
  ledger_sha="${terminal_info[2]}"
  nested_result="${terminal_info[3]}"
  nested_result_sha="${terminal_info[4]}"
  terminal_freeze_sha="${terminal_info[5]}"
  nested_rollout="${terminal_info[6]}"
  nested_rollout_sha="${terminal_info[7]}"
  if [[ -z "${accepted_freeze_aggregate_sha256}" ]]; then
    accepted_freeze_aggregate_sha256="${terminal_freeze_sha}"
  else
    [[ "${terminal_freeze_sha}" = "${accepted_freeze_aggregate_sha256}" ]] \
      || die "terminal markers disagree on the accepted freeze aggregate"
  fi
  marker="$(realpath -e -- "${marker}")" || die "terminal marker cannot be resolved"
  ledger_path="$(realpath -e -- "${ledger_path}")" \
    || die "terminal attempt ledger cannot be resolved"
  terminal_sha="$(sha256sum -- "${terminal}" | awk '{print $1}')"
  marker_sha="$(sha256sum -- "${marker}" | awk '{print $1}')"
  terminals[$index]="${terminal}"
  terminal_sha256s[$index]="${terminal_sha}"
  markers[$index]="${marker}"
  marker_sha256s[$index]="${marker_sha}"
  source_bindings+=(
    "${terminal}:${terminal_sha}"
    "${marker}:${marker_sha}"
    "${ledger_path}:${ledger_sha}"
  )
  if [[ "${nested_result}" != "-" ]]; then
    reject_delimiters "${nested_result}"
    source_bindings+=("${nested_result}:${nested_result_sha}")
  fi
  if [[ "${nested_rollout}" != "-" ]]; then
    nested_rollout="$(realpath -e -- "${nested_rollout}")" \
      || die "nested success rollout cannot be resolved"
    reject_delimiters "nested success rollout" "${nested_rollout}"
    source_bindings+=("${nested_rollout}:${nested_rollout_sha}")
  fi
done
[[ "${#terminals[@]}" -eq 30 ]] \
  || die "formal campaign requires exactly 30 terminal inputs"
[[ "${accepted_freeze_aggregate_sha256}" =~ ^[0-9a-f]{64}$ ]] \
  || die "accepted freeze aggregate identity is invalid"
for index in {0..29}; do
  seed=$((20260901 + index))
  [[ -n "${terminals[$index]:-}" ]] \
    || die "formal campaign is missing terminal seed ${seed}"
done

for value in \
  "${project_root}" "${scratch_root}" "${repo_root}" "${overlay}" "${base_config}" \
  "${identity_path}" "${image}" "${hf_cache}" "${inventory}" "${output_dir}" \
  "${terminal_output}" "${aggregate_output}" "${campaign_plan}" \
  "${terminals[@]}" "${markers[@]}"; do
  reject_delimiters "${value}"
done

[[ "$(git -C "${repo_root}" rev-parse --verify HEAD)" = "${git_commit}" \
  && -z "$(git -C "${repo_root}" status --porcelain --untracked-files=normal)" ]] \
  || die "submission checkout changed during validation"
printf '%s  %s\n' "${PRORM_IMAGE_SHA256}" "${image}" \
  | sha256sum --check --status || die "image changed before submission"
printf '%s  %s\n' "${inventory_sha256}" "${inventory}" \
  | sha256sum --check --status || die "inventory changed before submission"
for binding in "${source_bindings[@]}"; do
  path="${binding%%:*}"
  expected="${binding#*:}"
  printf '%s  %s\n' "${expected}" "${path}" \
    | sha256sum --check --status \
    || die "terminal input or marker changed before submission"
done
for binding in \
  "${overlay}:${overlay_file_sha256}" \
  "${base_config}:${base_file_sha256}" \
  "${identity_path}:${identity_file_sha256}"; do
  path="${binding%%:*}"
  expected="${binding#*:}"
  printf '%s  %s\n' "${expected}" "${path}" \
    | sha256sum --check --status \
    || die "committed identity input changed before submission"
done
[[ ! -e "${output_dir}" && ! -L "${output_dir}" ]] \
  || die "campaign publication destination appeared before submission"

export_spec="PATH=/usr/local/bin:/usr/bin:/bin,PRORM_PROJECT_ROOT=${project_root},PRORM_SCRATCH_ROOT=${scratch_root},PRORM_IMAGE=${image},PRORM_IMAGE_SHA256=${PRORM_IMAGE_SHA256},PRORM_HF_CACHE=${hf_cache},PRORM_HF_INVENTORY=${inventory},PRORM_HF_INVENTORY_SHA256=${inventory_sha256},PRORM_REPO_ROOT=${repo_root},PRORM_PHASE2_OVERLAY_REL=${overlay_relative},PRORM_PHASE2_BASE_REL=${base_relative},PRORM_PHASE2_OVERLAY_FILE_SHA256=${overlay_file_sha256},PRORM_PHASE2_BASE_FILE_SHA256=${base_file_sha256},PRORM_IDENTITIES_FILE_SHA256=${identity_file_sha256},PRORM_PHASE2_DESIGN_SHA256=${design_sha256},PRORM_PHASE2_BASE_CONFIG_HASH=${base_config_hash},PRORM_PHASE2_ACCEPTED_FREEZE_AGGREGATE_SHA256=${accepted_freeze_aggregate_sha256},PRORM_PHASE2_CAMPAIGN_PLAN_SHA256=${campaign_plan_sha256},PRORM_GIT_COMMIT=${git_commit},PRORM_PHASE2_TERMINAL_COUNT=30,PRORM_PHASE2_CAMPAIGN_OUTPUT_DIR=${output_dir},PRORM_PHASE2_CAMPAIGN_TERMINAL_OUTPUT=${terminal_output},PRORM_PHASE2_PRIMARY_AGGREGATE_OUTPUT=${aggregate_output}"
for index in {0..29}; do
  printf -v slot '%02d' "${index}"
  export_spec+=",PRORM_PHASE2_TERMINAL_${slot}=${terminals[$index]},PRORM_PHASE2_TERMINAL_SHA256_${slot}=${terminal_sha256s[$index]},PRORM_PHASE2_MARKER_${slot}=${markers[$index]},PRORM_PHASE2_MARKER_SHA256_${slot}=${marker_sha256s[$index]}"
done

slurm_log_dir="${project_root}/slurm-logs"
mkdir -p "${slurm_log_dir}" "${scratch_root}/phase2-campaign-finalize-jobs"
sbatch \
  --parsable \
  --account=sigroup \
  --job-name=prorm-phase2-campaign-finalize \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=1 \
  --mem=8G \
  --chdir="${repo_root}" \
  --partition="${partition}" \
  --time="${walltime}" \
  --output="${slurm_log_dir}/%x-%j.out" \
  --export="${export_spec}" \
  "${repo_root}/scripts/hpc4/phase2_campaign_finalize.sbatch"
