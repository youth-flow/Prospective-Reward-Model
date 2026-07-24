#!/usr/bin/env bash
#SBATCH --job-name=prorm-phase2-campaign-finalize
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
      die "unset exported ${variable}; campaign finalization forbids ambient container controls"
      ;;
  esac
done < <(compgen -e)

for name in \
  PRORM_PROJECT_ROOT PRORM_SCRATCH_ROOT PRORM_IMAGE PRORM_IMAGE_SHA256 \
  PRORM_HF_CACHE PRORM_HF_INVENTORY PRORM_HF_INVENTORY_SHA256 \
  PRORM_REPO_ROOT PRORM_PHASE2_OVERLAY_REL PRORM_PHASE2_BASE_REL \
  PRORM_PHASE2_OVERLAY_FILE_SHA256 PRORM_PHASE2_BASE_FILE_SHA256 \
  PRORM_IDENTITIES_FILE_SHA256 PRORM_PHASE2_DESIGN_SHA256 \
  PRORM_PHASE2_BASE_CONFIG_HASH PRORM_PHASE2_ACCEPTED_FREEZE_AGGREGATE_SHA256 \
  PRORM_GIT_COMMIT \
  PRORM_PHASE2_TERMINAL_COUNT PRORM_PHASE2_CAMPAIGN_OUTPUT_DIR \
  PRORM_PHASE2_CAMPAIGN_TERMINAL_OUTPUT PRORM_PHASE2_PRIMARY_AGGREGATE_OUTPUT; do
  [[ -n "${!name:-}" ]] || die "${name} is required"
done
[[ "${PRORM_PHASE2_TERMINAL_COUNT}" = "30" ]] \
  || die "campaign finalization requires exactly 30 terminal inputs"
for index in {0..29}; do
  printf -v slot '%02d' "${index}"
  for prefix in \
    PRORM_PHASE2_TERMINAL PRORM_PHASE2_TERMINAL_SHA256 \
    PRORM_PHASE2_MARKER PRORM_PHASE2_MARKER_SHA256; do
    name="${prefix}_${slot}"
    [[ -n "${!name:-}" ]] || die "${name} is required"
  done
done

: "${SLURM_JOB_ID:?campaign finalization must execute inside Slurm}"
[[ "${SLURM_JOB_ID}" =~ ^[1-9][0-9]*$ ]] || die "SLURM_JOB_ID must be positive"
[[ "${SLURM_JOB_ACCOUNT:-}" = "sigroup" ]] \
  || die "campaign finalization requires account sigroup"
case "${SLURM_JOB_PARTITION:-}" in
  amd|intel) ;;
  *) die "campaign finalization requires CPU partition amd or intel" ;;
esac
[[ -z "${SLURM_ARRAY_JOB_ID:-}" && -z "${SLURM_ARRAY_TASK_ID:-}" ]] \
  || die "campaign finalization must be a single non-array job"
[[ "${SLURM_NNODES:-}" = "1" && "${SLURM_NTASKS:-}" = "1" ]] \
  || die "campaign finalization requires exactly one node and task"
for variable in \
  SLURM_GPUS SLURM_GPUS_ON_NODE SLURM_JOB_GPUS SLURM_STEP_GPUS \
  CUDA_VISIBLE_DEVICES NVIDIA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL; do
  case "${!variable:-}" in
    ""|0) ;;
    *) die "campaign finalization must not receive GPU allocation or visibility" ;;
  esac
done
for variable in \
  SLURM_JOB_GRES SLURM_STEP_GRES SLURM_TRES_PER_JOB \
  SLURM_TRES_PER_NODE SLURM_TRES_PER_TASK; do
  value="${!variable:-}"
  [[ "${value,,}" != *gpu* ]] \
    || die "campaign finalization must not receive GPU GRES/TRES"
done

for command_name in \
  apptainer flock git python3 realpath sha256sum awk grep mktemp rm mv tee \
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
  [[ ! -L "${raw}" ]] || die "${name} must not be a symbolic link"
  resolved="$(realpath -e -- "${raw}")" || die "${name} cannot be resolved"
  [[ -f "${resolved}" && "${resolved}" = "${raw}" ]] \
    || die "${name} must be a canonical regular file"
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

fsync_tree() {
  local root="$1" mode="$2" terminal_name="$3" aggregate_name="$4"
  python3 -I -S - \
    "${root}" "${mode}" "${terminal_name}" "${aggregate_name}" <<'PY'
import os
import stat
import sys
from pathlib import Path


root = Path(sys.argv[1])
mode, terminal_name, aggregate_name = sys.argv[2:]
if root.is_symlink() or not root.is_dir():
    raise SystemExit("campaign publication staging root is unsafe")
expected = {terminal_name, "phase2-publication-receipt.json"}
if mode == "success":
    expected.add(aggregate_name)
elif mode != "failure":
    raise SystemExit("campaign publication mode is invalid")
observed = {path.name for path in root.iterdir()}
if observed != expected:
    raise SystemExit("campaign publication staging tree differs from its declaration")
for name in sorted(expected):
    path = root / name
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise SystemExit(f"campaign publication member is not regular: {path}")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
descriptor = os.open(
    root,
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0),
)
try:
    os.fsync(descriptor)
finally:
    os.close(descriptor)
PY
}

fsync_directory() {
  local directory="$1"
  python3 -I -S - "${directory}" <<'PY'
import os
import sys


descriptor = os.open(
    sys.argv[1],
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0),
)
try:
    os.fsync(descriptor)
finally:
    os.close(descriptor)
PY
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
  "${PRORM_SCRATCH_ROOT}"|"${PRORM_SCRATCH_ROOT}"/*)
    die "project and scratch roots overlap"
    ;;
esac
case "${PRORM_SCRATCH_ROOT}" in
  "${PRORM_PROJECT_ROOT}"|"${PRORM_PROJECT_ROOT}"/*)
    die "project and scratch roots overlap"
    ;;
esac
[[ -w "${PRORM_PROJECT_ROOT}" && -w "${PRORM_SCRATCH_ROOT}" ]] \
  || die "project and scratch roots must be writable"

for name in \
  PRORM_PROJECT_ROOT PRORM_SCRATCH_ROOT PRORM_IMAGE PRORM_HF_CACHE \
  PRORM_HF_INVENTORY PRORM_REPO_ROOT PRORM_PHASE2_CAMPAIGN_OUTPUT_DIR \
  PRORM_PHASE2_CAMPAIGN_TERMINAL_OUTPUT PRORM_PHASE2_PRIMARY_AGGREGATE_OUTPUT; do
  reject_delimiters "${name}" "${!name}"
done
for name in \
  PRORM_IMAGE_SHA256 PRORM_HF_INVENTORY_SHA256 \
  PRORM_PHASE2_OVERLAY_FILE_SHA256 PRORM_PHASE2_BASE_FILE_SHA256 \
  PRORM_IDENTITIES_FILE_SHA256 PRORM_PHASE2_DESIGN_SHA256 \
  PRORM_PHASE2_BASE_CONFIG_HASH \
  PRORM_PHASE2_ACCEPTED_FREEZE_AGGREGATE_SHA256; do
  [[ "${!name}" =~ ^[0-9a-f]{64}$ ]] || die "${name} must be lowercase SHA256"
done
[[ "${PRORM_GIT_COMMIT}" =~ ^[0-9a-f]{40,64}$ ]] || die "invalid Git commit"
for relative_name in PRORM_PHASE2_OVERLAY_REL PRORM_PHASE2_BASE_REL; do
  [[ "${!relative_name}" =~ ^configs/[A-Za-z0-9._-]+\.yaml$ ]] \
    || die "${relative_name} is not a safe tracked config path"
done
[[ "${PRORM_PHASE2_OVERLAY_REL}" != "${PRORM_PHASE2_BASE_REL}" ]] \
  || die "confirmatory overlay and base config must be distinct"
[[ "${PRORM_HF_INVENTORY}" = \
  "${PRORM_HF_CACHE}/inventories/${PRORM_PHASE2_BASE_CONFIG_HASH}.json" ]] \
  || die "HF inventory is not addressed by the base config identity"
printf '%s  %s\n' "${PRORM_IMAGE_SHA256}" "${PRORM_IMAGE}" \
  | sha256sum --check --status || die "image SHA256 mismatch"
printf '%s  %s\n' "${PRORM_HF_INVENTORY_SHA256}" "${PRORM_HF_INVENTORY}" \
  | sha256sum --check --status || die "HF inventory SHA256 mismatch"

repository_top="$(git -C "${PRORM_REPO_ROOT}" rev-parse --show-toplevel)"
[[ "${repository_top}" = "${PRORM_REPO_ROOT}" ]] || die "invalid repository root"
[[ "$(git -C "${PRORM_REPO_ROOT}" rev-parse --verify HEAD)" = "${PRORM_GIT_COMMIT}" ]] \
  || die "repository HEAD changed after submission"
[[ -z "$(git -C "${PRORM_REPO_ROOT}" status --porcelain --untracked-files=normal)" ]] \
  || die "campaign finalization requires a clean committed repository"
git -C "${PRORM_REPO_ROOT}" cat-file -e \
  "${PRORM_GIT_COMMIT}^{commit}" 2>/dev/null \
  || die "submitted Git commit is unavailable"

campaign_root_expected="${PRORM_PROJECT_ROOT}/runs/phase2-confirmatory/${PRORM_PHASE2_DESIGN_SHA256}"
campaign_root="$(realpath -e -- "${campaign_root_expected}")" \
  || die "confirmatory design run root is missing"
[[ "${campaign_root}" = "${campaign_root_expected}" \
  && -d "${campaign_root}" && ! -L "${campaign_root_expected}" ]] \
  || die "confirmatory design run root is unsafe"
[[ "${PRORM_PHASE2_CAMPAIGN_OUTPUT_DIR}" = "${campaign_root}/campaign-final" ]] \
  || die "campaign publication is locked to the one-shot campaign-final directory"
[[ ! -e "${PRORM_PHASE2_CAMPAIGN_OUTPUT_DIR}" \
  && ! -L "${PRORM_PHASE2_CAMPAIGN_OUTPUT_DIR}" ]] \
  || die "refusing to overwrite campaign publication directory"
[[ -w "${campaign_root}" ]] || die "campaign design run root is not writable"
registry_lock="${campaign_root}/campaign-registry/registry.lock"
[[ -f "${registry_lock}" && ! -L "${registry_lock}" ]] \
  || die "campaign registry lock is missing or unsafe"
exec {registry_lock_fd}>> "${registry_lock}"
flock -x "${registry_lock_fd}" \
  || die "failed to acquire exclusive campaign registry finalization lock"
terminal_output_name="$(basename -- "${PRORM_PHASE2_CAMPAIGN_TERMINAL_OUTPUT}")"
aggregate_output_name="$(basename -- "${PRORM_PHASE2_PRIMARY_AGGREGATE_OUTPUT}")"
[[ "${PRORM_PHASE2_CAMPAIGN_TERMINAL_OUTPUT}" = \
  "${PRORM_PHASE2_CAMPAIGN_OUTPUT_DIR}/${terminal_output_name}" ]] \
  || die "campaign terminal output escaped its publication directory"
[[ "${PRORM_PHASE2_PRIMARY_AGGREGATE_OUTPUT}" = \
  "${PRORM_PHASE2_CAMPAIGN_OUTPUT_DIR}/${aggregate_output_name}" ]] \
  || die "primary aggregate output escaped its publication directory"
[[ "${terminal_output_name}" = "phase2-campaign-terminal.json" \
  && "${aggregate_output_name}" = "phase2-primary-aggregate.json" ]] \
  || die "campaign output filenames are locked by the formal contract"

validate_terminal() {
  local terminal="$1" expected_seed="$2"
  python3 -I -S \
    "${PRORM_REPO_ROOT}/scripts/hpc4/validate_phase2_terminal.py" \
    "${terminal}" "${expected_seed}" "${PRORM_PHASE2_DESIGN_SHA256}" \
    "${PRORM_PHASE2_BASE_CONFIG_HASH}" "${PRORM_GIT_COMMIT}" \
    "${PRORM_IMAGE_SHA256}" "${PRORM_HF_INVENTORY_SHA256}"
}

mapfile -t registry_terminals < <(
  python3 -I -S \
    "${PRORM_REPO_ROOT}/scripts/hpc4/resolve_phase2_campaign_registry.py" \
    "${campaign_root}" "${PRORM_PHASE2_DESIGN_SHA256}" \
    "${PRORM_PHASE2_BASE_CONFIG_HASH}" "${PRORM_GIT_COMMIT}" \
    "${PRORM_IMAGE_SHA256}" "${PRORM_HF_INVENTORY_SHA256}"
)
[[ "${#registry_terminals[@]}" -eq 30 ]] \
  || die "campaign registry did not resolve exactly 30 terminal heads"

campaign_inputs_snapshot_sha256() {
  python3 -I -S - "${campaign_root}" <<'PY'
import hashlib
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
selected = [root / "campaign-registry", *sorted(root.glob("seed-*"))]
for parent in selected:
    if parent.is_symlink() or not parent.is_dir():
        raise SystemExit(f"campaign input root is missing or unsafe: {parent}")
    for directory, names, files in os.walk(parent, followlinks=False):
        names.sort()
        files.sort()
        directory_path = Path(directory)
        for name in names:
            path = directory_path / name
            if path.is_symlink():
                relative = path.relative_to(root).as_posix()
                digest = hashlib.sha256(
                    ("symlink:" + os.readlink(path)).encode("utf-8")
                ).hexdigest()
                rows.append(f"{relative}\0{digest}\n")
        for name in files:
            path = directory_path / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                digest = hashlib.sha256(
                    ("symlink:" + os.readlink(path)).encode("utf-8")
                ).hexdigest()
            elif path.is_file():
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
            else:
                raise SystemExit(f"campaign input is not a file or symlink: {path}")
            rows.append(f"{relative}\0{digest}\n")
print(hashlib.sha256("".join(rows).encode("utf-8")).hexdigest())
PY
}
campaign_inputs_snapshot="$(
  campaign_inputs_snapshot_sha256
)" || die "cannot snapshot the exact campaign registry/attempt evidence"
[[ "${campaign_inputs_snapshot}" =~ ^[0-9a-f]{64}$ ]] \
  || die "campaign input snapshot digest is invalid"

terminals=()
source_bindings=()
declare -A seen_terminal_paths=()
for index in {0..29}; do
  printf -v slot '%02d' "${index}"
  seed=$((20260901 + index))
  terminal_name="PRORM_PHASE2_TERMINAL_${slot}"
  terminal_sha_name="PRORM_PHASE2_TERMINAL_SHA256_${slot}"
  marker_name="PRORM_PHASE2_MARKER_${slot}"
  marker_sha_name="PRORM_PHASE2_MARKER_SHA256_${slot}"
  require_file "${terminal_name}"
  require_file "${marker_name}"
  terminal="${!terminal_name}"
  marker="${!marker_name}"
  [[ "${terminal}" = "${registry_terminals[$index]}" ]] \
    || die "submitted terminal is not the current registry head for seed ${seed}"
  reject_delimiters "${terminal_name}" "${terminal}"
  reject_delimiters "${marker_name}" "${marker}"
  [[ "${!terminal_sha_name}" =~ ^[0-9a-f]{64}$ ]] \
    || die "${terminal_sha_name} must be lowercase SHA256"
  [[ "${!marker_sha_name}" =~ ^[0-9a-f]{64}$ ]] \
    || die "${marker_sha_name} must be lowercase SHA256"
  case "${terminal}" in
    "${campaign_root}/seed-${seed}/"*) ;;
    *) die "terminal ${seed} escaped its bound confirmatory seed directory" ;;
  esac
  [[ ! -v "seen_terminal_paths[${terminal}]" ]] \
    || die "duplicate terminal manifest path"
  seen_terminal_paths["${terminal}"]=1
  printf '%s  %s\n' "${!terminal_sha_name}" "${terminal}" \
    | sha256sum --check --status || die "terminal manifest SHA256 mismatch"
  printf '%s  %s\n' "${!marker_sha_name}" "${marker}" \
    | sha256sum --check --status || die "terminal marker SHA256 mismatch"
  validator_output="$(validate_terminal "${terminal}" "${seed}")" \
    || die "terminal ${seed} failed manifest/marker/ledger validation"
  mapfile -t terminal_info <<< "${validator_output}"
  [[ "${#terminal_info[@]}" -eq 8 ]] \
    || die "terminal ${seed} validator returned an invalid binding"
  [[ "${terminal_info[0]}" = "${marker}" ]] \
    || die "terminal ${seed} marker path differs from the submitted binding"
  [[ "${terminal_info[5]}" = \
    "${PRORM_PHASE2_ACCEPTED_FREEZE_AGGREGATE_SHA256}" ]] \
    || die "terminal ${seed} marker disagrees with the accepted freeze aggregate"
  terminals+=("${terminal}")
  source_bindings+=(
    "${terminal}:${!terminal_sha_name}"
    "${marker}:${!marker_sha_name}"
    "${terminal_info[1]}:${terminal_info[2]}"
  )
  if [[ "${terminal_info[3]}" != "-" ]]; then
    reject_delimiters "nested success result" "${terminal_info[3]}"
    source_bindings+=("${terminal_info[3]}:${terminal_info[4]}")
  fi
  if [[ "${terminal_info[6]}" != "-" ]]; then
    reject_delimiters "nested success rollout" "${terminal_info[6]}"
    source_bindings+=("${terminal_info[6]}:${terminal_info[7]}")
  fi
done
[[ "${#terminals[@]}" -eq 30 ]] \
  || die "campaign finalization lost an exact terminal slot"

job_dir="${PRORM_SCRATCH_ROOT}/phase2-campaign-finalize-jobs/${SLURM_JOB_ID}"
[[ ! -e "${job_dir}" && ! -L "${job_dir}" ]] \
  || die "refusing to reuse campaign finalization scratch directory"
mkdir -p "${PRORM_SCRATCH_ROOT}/phase2-campaign-finalize-jobs"
mkdir "${job_dir}"
[[ "$(realpath -e -- "${job_dir}")" = "${job_dir}" && ! -L "${job_dir}" ]] \
  || die "campaign finalization scratch directory is unsafe"

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
    | sha256sum --check --status \
    || die "detached identity input differs from the submitted exact source"
done

assert_execution_checkout() {
  [[ "$(git -C "${execution_repo}" rev-parse --verify HEAD)" = "${PRORM_GIT_COMMIT}" \
    && -z "$(git -C "${execution_repo}" status --porcelain --untracked-files=normal)" ]] \
    || die "detached campaign finalization checkout changed"
}
assert_control_checkout() {
  [[ "$(git -C "${PRORM_REPO_ROOT}" rev-parse --verify HEAD)" = "${PRORM_GIT_COMMIT}" \
    && -z "$(git -C "${PRORM_REPO_ROOT}" status --porcelain --untracked-files=normal)" ]] \
    || die "control-plane checkout changed during campaign finalization"
}
assert_execution_checkout
assert_control_checkout

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
assert_execution_checkout
"${container[@]}" python -m smart_reward.cli phase2-config-check \
  "${execution_overlay}" | tee "${phase2_check}"
"${container[@]}" python -m smart_reward.cli config-check \
  "${execution_base}" | tee "${base_check}"
python3 -I -S - \
  "${phase2_check}" "${base_check}" "${execution_base}" \
  "${PRORM_PHASE2_DESIGN_SHA256}" "${PRORM_PHASE2_BASE_CONFIG_HASH}" <<'PY'
import json
import sys
from pathlib import Path


phase2 = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
base = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
expected_base_path = Path(sys.argv[3]).resolve()
if (
    phase2.get("status") != "ok"
    or phase2.get("design_stage") != "confirmatory"
    or phase2.get("formal_eligibility") is not True
    or phase2.get("phase2_design_sha256") != sys.argv[4]
    or phase2.get("source_config_hash") != sys.argv[5]
    or Path(str(phase2.get("source_config_path"))).resolve() != expected_base_path
    or base.get("status") != "ok"
    or base.get("config_hash") != sys.argv[5]
):
    raise SystemExit("container confirmatory identities differ from submission")
PY

confirmatory_contract_check="${job_dir}/confirmatory-contract-check.json"
"${container[@]}" python - "${execution_overlay}" <<'PY' \
  | tee "${confirmatory_contract_check}"
import json
import sys

from smart_reward.phase2_config import load_phase2_config_bundle


bundle = load_phase2_config_bundle(sys.argv[1])
payload = {
    "accepted_freeze_aggregate_sha256": bundle.config["objective"]["common_beta"][
        "beta_source_aggregate_sha256"
    ],
    "seeds": bundle.config["run"]["seeds"],
    "status": "ok",
}
print(json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":")))
PY
python3 -I -S - \
  "${confirmatory_contract_check}" \
  "${PRORM_PHASE2_ACCEPTED_FREEZE_AGGREGATE_SHA256}" <<'PY'
import json
import sys
from pathlib import Path


value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if (
    value.get("status") != "ok"
    or value.get("accepted_freeze_aggregate_sha256") != sys.argv[2]
    or value.get("seeds") != list(range(20260901, 20260931))
):
    raise SystemExit("detached confirmatory contract differs from terminal markers")
PY

inventory_verification="${job_dir}/hf-inventory-verification.json"
assert_execution_checkout
"${container[@]}" python "${execution_repo}/scripts/hpc4/stage_hf_assets.py" \
  "${execution_base}" "${PRORM_HF_CACHE}" \
  --verify-only --inventory "${PRORM_HF_INVENTORY}" \
  | tee "${inventory_verification}"
python3 -I -S - \
  "${inventory_verification}" "${PRORM_PHASE2_BASE_CONFIG_HASH}" \
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
    raise SystemExit("offline HF inventory verification changed the base identity")
PY
printf '%s  %s\n' "${PRORM_IMAGE_SHA256}" "${PRORM_IMAGE}" \
  | sha256sum --check --status || die "image changed during identity validation"
printf '%s  %s\n' "${PRORM_HF_INVENTORY_SHA256}" "${PRORM_HF_INVENTORY}" \
  | sha256sum --check --status || die "inventory changed during offline verification"

for binding in "${source_bindings[@]}"; do
  path="${binding%%:*}"
  expected="${binding#*:}"
  printf '%s  %s\n' "${expected}" "${path}" \
    | sha256sum --check --status \
    || die "terminal input, marker, or nested result changed before finalization"
done
[[ ! -e "${PRORM_PHASE2_CAMPAIGN_OUTPUT_DIR}" \
  && ! -L "${PRORM_PHASE2_CAMPAIGN_OUTPUT_DIR}" ]] \
  || die "campaign publication destination appeared before finalization"

staging_dir=""
cleanup_staging() {
  local workload_exit=$?
  trap - EXIT
  set +e
  if (( workload_exit != 0 )) && [[ -n "${staging_dir}" ]] \
    && [[ -e "${staging_dir}" || -L "${staging_dir}" ]]; then
    case "${staging_dir}" in
      "${campaign_root}"/.campaign.publish-"${SLURM_JOB_ID}".*) ;;
      *)
        printf 'refusing to clean unexpected campaign staging path: %s\n' \
          "${staging_dir}" >&2
        exit 1
        ;;
    esac
    rm -r --one-file-system -- "${staging_dir}"
  fi
  exit "${workload_exit}"
}
trap cleanup_staging EXIT

staging_dir="$(mktemp -d "${campaign_root}/.campaign.publish-${SLURM_JOB_ID}.XXXXXX")"
[[ "$(dirname -- "${staging_dir}")" = "${campaign_root}" \
  && -d "${staging_dir}" && ! -L "${staging_dir}" ]] \
  || die "campaign staging directory is unsafe"
staged_terminal="${staging_dir}/${terminal_output_name}"
staged_aggregate="${staging_dir}/${aggregate_output_name}"

assert_execution_checkout
"${container[@]}" python -m smart_reward.cli phase2-campaign-finalize \
  "${execution_overlay}" "${staged_terminal}" "${staged_aggregate}" \
  "${terminals[@]}" | tee "${job_dir}/phase2-campaign-finalize.log"

publication_mode="$(
  python3 -I -S - \
    "${staged_terminal}" "${staged_aggregate}" "${aggregate_output_name}" \
    "${PRORM_PHASE2_DESIGN_SHA256}" "${PRORM_PHASE2_BASE_CONFIG_HASH}" <<'PY'
import hashlib
import json
import re
import sys
from pathlib import Path


terminal = Path(sys.argv[1])
aggregate = Path(sys.argv[2])
aggregate_name = sys.argv[3]
design_sha = sys.argv[4]
base_hash = sys.argv[5]
expected_seeds = list(range(20260901, 20260931))


def reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load(path):
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f"expected regular JSON file is missing: {path}")
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicates,
        parse_constant=lambda item: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant: {item}")
        ),
    )


value = load(terminal)
entries = value.get("entries")
failed = value.get("failed_seeds")
if (
    not isinstance(value, dict)
    or value.get("schema_version") != "phase2-campaign-terminal/v2"
    or value.get("terminal") is not True
    or value.get("source_config_hash") != base_hash
    or value.get("phase2_design_sha256") != design_sha
    or value.get("declared_seeds") != expected_seeds
    or value.get("num_declared_seeds") != 30
    or value.get("terminal_seed_set_complete") is not True
    or value.get("seed_replacement_allowed") is not False
    or value.get("retry_policy")
    != "single_predeclared_attempt_no_retry"
    or not isinstance(entries, list)
    or len(entries) != 30
    or [entry.get("seed") for entry in entries if isinstance(entry, dict)]
    != expected_seeds
    or not isinstance(failed, list)
):
    raise SystemExit("staged campaign terminal failed its exact-30 identity contract")

if failed:
    if (
        value.get("status") != "not_passed_due_to_seed_failure"
        or value.get("supports_formal_claim") is not False
        or value.get("primary_ci_computed") is not False
        or value.get("primary_aggregate") is not None
        or aggregate.exists()
        or aggregate.is_symlink()
    ):
        raise SystemExit("failed campaign attempted to publish a primary aggregate")
    print("failure")
else:
    if (
        value.get("status") != "primary_aggregate_completed"
        or value.get("primary_ci_computed") is not True
        or not isinstance(value.get("primary_aggregate"), dict)
    ):
        raise SystemExit("successful campaign did not complete its primary aggregate")
    aggregate_value = load(aggregate)
    aggregate_sha = hashlib.sha256(aggregate.read_bytes()).hexdigest()
    primary = value["primary_aggregate"]
    if (
        primary.get("path") != aggregate_name
        or primary.get("sha256") != aggregate_sha
        or primary.get("schema_version") != "common-beta-seed-aggregate/v3"
        or aggregate_value.get("schema_version") != "common-beta-seed-aggregate/v3"
        or aggregate_value.get("source_config_hash") != base_hash
        or aggregate_value.get("phase2_design_sha256") != design_sha
        or not isinstance(aggregate_value.get("sources"), list)
        or len(aggregate_value["sources"]) != 30
    ):
        raise SystemExit("staged primary aggregate failed its bound output contract")
    print("success")
PY
)" || die "staged campaign publication failed output validation"
case "${publication_mode}" in
  success|failure) ;;
  *) die "campaign output validator returned an invalid publication mode" ;;
esac

for binding in "${source_bindings[@]}"; do
  path="${binding%%:*}"
  expected="${binding#*:}"
  printf '%s  %s\n' "${expected}" "${path}" \
    | sha256sum --check --status \
    || die "terminal input, marker, or nested result changed during finalization"
done
printf '%s  %s\n' "${PRORM_IMAGE_SHA256}" "${PRORM_IMAGE}" \
  | sha256sum --check --status || die "image changed during campaign finalization"
printf '%s  %s\n' "${PRORM_HF_INVENTORY_SHA256}" "${PRORM_HF_INVENTORY}" \
  | sha256sum --check --status || die "inventory changed during campaign finalization"
assert_execution_checkout
assert_control_checkout
[[ ! -e "${PRORM_PHASE2_CAMPAIGN_OUTPUT_DIR}" \
  && ! -L "${PRORM_PHASE2_CAMPAIGN_OUTPUT_DIR}" ]] \
  || die "campaign publication destination appeared before atomic publication"
[[ -f "${staged_terminal}" && ! -L "${staged_terminal}" ]] \
  || die "staged campaign terminal is not a regular file"
if [[ "${publication_mode}" = "success" ]]; then
  [[ -f "${staged_aggregate}" && ! -L "${staged_aggregate}" ]] \
    || die "successful campaign is missing its staged primary aggregate"
else
  [[ ! -e "${staged_aggregate}" && ! -L "${staged_aggregate}" ]] \
    || die "failed campaign produced a staged primary aggregate"
fi
terminal_sha256="$(sha256sum -- "${staged_terminal}" | awk '{print $1}')"
aggregate_sha256="none"
if [[ "${publication_mode}" = "success" ]]; then
  aggregate_sha256="$(sha256sum -- "${staged_aggregate}" | awk '{print $1}')"
fi
mapfile -t final_registry_terminals < <(
  python3 -I -S \
    "${PRORM_REPO_ROOT}/scripts/hpc4/resolve_phase2_campaign_registry.py" \
    "${campaign_root}" "${PRORM_PHASE2_DESIGN_SHA256}" \
    "${PRORM_PHASE2_BASE_CONFIG_HASH}" "${PRORM_GIT_COMMIT}" \
    "${PRORM_IMAGE_SHA256}" "${PRORM_HF_INVENTORY_SHA256}"
)
[[ "${#final_registry_terminals[@]}" -eq 30 ]] \
  || die "campaign registry lost an exact terminal head before publication"
for index in {0..29}; do
  [[ "${final_registry_terminals[$index]}" = "${terminals[$index]}" ]] \
    || die "campaign registry head changed during finalization"
done
final_campaign_inputs_snapshot="$(
  campaign_inputs_snapshot_sha256
)" || die "cannot re-snapshot campaign inputs before publication"
[[ "${final_campaign_inputs_snapshot}" = "${campaign_inputs_snapshot}" ]] \
  || die "campaign registry or attempt evidence changed during finalization"
publication_receipt="${staging_dir}/phase2-publication-receipt.json"
python3 -I -S - \
  "${publication_receipt}" "${publication_mode}" "${terminal_output_name}" \
  "${terminal_sha256}" "${aggregate_output_name}" "${aggregate_sha256}" \
  "${PRORM_GIT_COMMIT}" "${SLURM_JOB_ID}" \
  "${PRORM_PHASE2_DESIGN_SHA256}" "${PRORM_PHASE2_BASE_CONFIG_HASH}" \
  "${PRORM_PHASE2_ACCEPTED_FREEZE_AGGREGATE_SHA256}" \
  "${PRORM_IMAGE_SHA256}" "${PRORM_HF_INVENTORY_SHA256}" \
  "${campaign_inputs_snapshot}" \
  "${source_bindings[@]}" <<'PY'
import json
import os
import re
import sys
from pathlib import Path


(
    receipt_raw,
    mode,
    terminal_name,
    terminal_sha,
    aggregate_name,
    aggregate_sha,
    git_commit,
    slurm_job_id,
    design_sha,
    base_hash,
    freeze_sha,
    image_sha,
    inventory_sha,
    campaign_inputs_sha,
    *bindings,
) = sys.argv[1:]
if mode not in {"success", "failure"}:
    raise SystemExit("invalid publication mode")
digests = []
for binding in bindings:
    if ":" not in binding:
        raise SystemExit("malformed publication input binding")
    _, digest = binding.rsplit(":", 1)
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise SystemExit("invalid publication input SHA256")
    digests.append(digest)
payload = {
    "schema_version": "phase2-campaign-publication-receipt/v1",
    "publication_commit": "atomic_directory_rename",
    "canonical_output_directory": "campaign-final",
    "publication_mode": mode,
    "git_commit": git_commit,
    "finalizer_slurm_job_id": slurm_job_id,
    "source_config_hash": base_hash,
    "phase2_design_sha256": design_sha,
    "accepted_freeze_aggregate_sha256": freeze_sha,
    "image_sha256": image_sha,
    "hf_inventory_sha256": inventory_sha,
    "campaign_inputs_snapshot_sha256": campaign_inputs_sha,
    "terminal": {
        "path": terminal_name,
        "sha256": terminal_sha,
    },
    "primary_aggregate": (
        None
        if mode == "failure"
        else {
            "path": aggregate_name,
            "sha256": aggregate_sha,
        }
    ),
    "input_sha256": digests,
}
with Path(receipt_raw).open("x", encoding="utf-8", newline="\n") as stream:
    json.dump(payload, stream, allow_nan=False, sort_keys=True, separators=(",", ":"))
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
PY
[[ -f "${publication_receipt}" && ! -L "${publication_receipt}" ]] \
  || die "staged publication receipt is missing or unsafe"
fsync_tree \
  "${staging_dir}" "${publication_mode}" \
  "${terminal_output_name}" "${aggregate_output_name}" \
  || die "campaign publication staging durability barrier failed"

if ! mv -T --no-clobber -- \
  "${staging_dir}" "${PRORM_PHASE2_CAMPAIGN_OUTPUT_DIR}"; then
  die "atomic no-overwrite campaign directory publication failed"
fi
fsync_directory "${campaign_root}" \
  || die "campaign publication parent durability barrier failed"
published_terminal="${PRORM_PHASE2_CAMPAIGN_OUTPUT_DIR}/${terminal_output_name}"
published_aggregate="${PRORM_PHASE2_CAMPAIGN_OUTPUT_DIR}/${aggregate_output_name}"
staging_dir=""
trap - EXIT
printf 'Phase-2 campaign published: mode=%s terminal=%s terminal_sha256=%s aggregate=%s aggregate_sha256=%s\n' \
  "${publication_mode}" "${published_terminal}" "${terminal_sha256}" \
  "${published_aggregate}" "${aggregate_sha256}" || true
