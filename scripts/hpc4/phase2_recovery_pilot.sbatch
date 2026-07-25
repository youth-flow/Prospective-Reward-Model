#!/usr/bin/env bash
#SBATCH --job-name=prorm-p2-recovery
#SBATCH --account=sigroup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gpus-per-node=1
#SBATCH --no-requeue
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err

set -euo pipefail
die() { echo "error: $*" >&2; exit 2; }

: "${SLURM_JOB_ID:?recovery must execute inside Slurm}"
: "${SLURM_ARRAY_JOB_ID:?recovery must be a Slurm array}"
: "${SLURM_ARRAY_TASK_ID:?recovery must be a Slurm array}"
[[ "${SLURM_RESTART_COUNT:-0}" = 0 ]] \
  || die "one-shot recovery refuses a scheduler restart/requeue"
[[ "${SLURM_JOB_ACCOUNT:-}" = sigroup ]] || die "recovery requires account sigroup"
[[ "${SLURM_JOB_PARTITION:-}" = gpu-l20 ]] || die "recovery requires gpu-l20"
[[ "${SLURM_CLUSTER_NAME:-}" = hpc4 ]] || die "recovery requires HPC4"
[[ "${SLURM_NNODES:-}" = 1 && "${SLURM_NTASKS:-}" = 1 ]] \
  || die "recovery requires exactly one node/task"
[[ "${SLURM_ARRAY_TASK_ID}" =~ ^[0-2]$ ]] || die "array task must lie in 0..2"
for name in \
  PRORM_GIT_COMMIT PRORM_IMAGE PRORM_IMAGE_SHA256 PRORM_HF_CACHE \
  PRORM_HF_INVENTORY PRORM_HF_INVENTORY_SHA256 PRORM_PROJECT_ROOT \
  PRORM_SCRATCH_ROOT PRORM_REPO_ROOT PRORM_PHASE2_RECOVERY_DESIGN_SHA256 \
  PRORM_PHASE2_BASE_CONFIG_HASH PRORM_PHASE2_PARENT_DESIGN_SHA256 \
  PRORM_PHASE2_PARENT_REGISTRY_SHA256 \
  PRORM_PHASE2_PARENT_PRODUCER_GIT_COMMIT PRORM_PHASE2_DIAGNOSTIC_GIT_COMMIT; do
  [[ -n "${!name:-}" ]] || die "${name} is required"
done
[[ "${PRORM_PHASE2_RECOVERY_DESIGN_SHA256}" = \
  9602b0f00a73880545fd57ce1886ec65d7901385cce2b919fd72f3efec4592d4 ]] \
  || die "recovery design identity mismatch"
[[ "${PRORM_PHASE2_BASE_CONFIG_HASH}" = \
  81ccbd3bc9d745d3792d1834116ba2480d34a7201d6f4e463a61d8d8ab0baefa ]] \
  || die "base identity mismatch"
[[ "${PRORM_IMAGE_SHA256}" = \
  d6fc044b4fa303747908783ea057d5b8946f613bfec6a6ca301e3a02fd7719cb ]] \
  || die "container image identity mismatch"
[[ "${PRORM_HF_INVENTORY_SHA256}" = \
  86c7c0fcab9cc0de612c6a5af05778e8b34617822b2e33474df8ed840eef82fd ]] \
  || die "HF inventory identity mismatch"
[[ "${PRORM_PHASE2_PARENT_DESIGN_SHA256}" = \
  0c8820b67b8ca85c23cd5c31b8d25001018b31a7271f6a861d88cfae2f85d7ca ]] \
  || die "parent design identity mismatch"
[[ "${PRORM_PHASE2_PARENT_REGISTRY_SHA256}" = \
  7be4ee90b1f494d32f96214f407a57cbee54be86a77dacc1206d2acd527857dc ]] \
  || die "parent registry identity mismatch"
[[ "${PRORM_PHASE2_PARENT_PRODUCER_GIT_COMMIT}" = \
  ae28e2a10f0bd5762899be01ce66bc5b423374cf ]] \
  || die "parent producer commit mismatch"
[[ "${PRORM_PHASE2_DIAGNOSTIC_GIT_COMMIT}" = \
  791c2daac7f1601f6798d5878bef1770ca9d5ebf ]] \
  || die "optimizer diagnostic commit mismatch"
for name in PRORM_PROJECT_ROOT PRORM_SCRATCH_ROOT PRORM_REPO_ROOT; do
  path="${!name}"
  [[ -d "${path}" && ! -L "${path}" ]] || die "${name} must be a real directory"
  [[ "$(realpath -e -- "${path}")" = "${path}" ]] || die "${name} must be canonical"
done
for name in PRORM_PROJECT_ROOT PRORM_SCRATCH_ROOT PRORM_HF_CACHE; do
  path="${!name}"
  [[ "${path}" != *":"* && "${path}" != *","* \
    && "${path}" != *$'\n'* && "${path}" != *$'\r'* ]] \
    || die "${name} is unsafe for the locked Apptainer bind syntax"
done
[[ -f "${PRORM_IMAGE}" && ! -L "${PRORM_IMAGE}" ]] \
  || die "container image must be a regular non-symlink file"
[[ "$(realpath -e -- "${PRORM_IMAGE}")" = "${PRORM_IMAGE}" ]] \
  || die "container image path must be canonical"
[[ -d "${PRORM_HF_CACHE}" && ! -L "${PRORM_HF_CACHE}" ]] \
  || die "HF cache must be a real directory"
[[ "$(realpath -e -- "${PRORM_HF_CACHE}")" = "${PRORM_HF_CACHE}" ]] \
  || die "HF cache path must be canonical"
expected_inventory="${PRORM_HF_CACHE}/inventories/${PRORM_PHASE2_BASE_CONFIG_HASH}.json"
[[ "${PRORM_HF_INVENTORY}" = "${expected_inventory}" ]] \
  || die "HF inventory path is not the frozen cache inventory"
[[ -f "${PRORM_HF_INVENTORY}" && ! -L "${PRORM_HF_INVENTORY}" ]] \
  || die "HF inventory must be a regular non-symlink file"
[[ "$(realpath -e -- "${PRORM_HF_INVENTORY}")" = "${PRORM_HF_INVENTORY}" ]] \
  || die "HF inventory path must be canonical"
printf '%s  %s\n' "${PRORM_IMAGE_SHA256}" "${PRORM_IMAGE}" \
  | sha256sum --check --status || die "container image bytes changed before job execution"
printf '%s  %s\n' "${PRORM_HF_INVENTORY_SHA256}" "${PRORM_HF_INVENTORY}" \
  | sha256sum --check --status || die "HF inventory bytes changed before job execution"
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

seeds=(20260801 20260802 20260803)
seed="${seeds[SLURM_ARRAY_TASK_ID]}"
execution_id="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
execution_revision="2"
ensure_real_subdirectory "${PRORM_SCRATCH_ROOT}" "phase2-recovery-jobs"
ensure_real_subdirectory \
  "${PRORM_PROJECT_ROOT}" \
  "runs/phase2-recovery-pilot/${PRORM_PHASE2_RECOVERY_DESIGN_SHA256}/execution-${execution_revision}"
scratch_jobs_root="${PRORM_SCRATCH_ROOT}/phase2-recovery-jobs"
campaign_root="${PRORM_PROJECT_ROOT}/runs/phase2-recovery-pilot/${PRORM_PHASE2_RECOVERY_DESIGN_SHA256}/execution-${execution_revision}"
job_dir="${scratch_jobs_root}/${execution_id}"
seed_root="${campaign_root}/seed-${seed}"
project_run="${seed_root}/job-${execution_id}"
[[ ! -e "${job_dir}" && ! -L "${job_dir}" ]] || die "scratch recovery directory already exists"
[[ ! -e "${seed_root}" && ! -L "${seed_root}" ]] \
  || die "one-shot recovery namespace already exists for seed ${seed}"
mkdir -- "${job_dir}" || die "could not create scratch recovery directory"
[[ -d "${job_dir}" && ! -L "${job_dir}" \
  && "$(realpath -e -- "${job_dir}")" = "${job_dir}" ]] \
  || die "scratch recovery directory is unsafe"
mkdir -- "${seed_root}" || die "could not create recovery seed namespace"
[[ -d "${seed_root}" && ! -L "${seed_root}" \
  && "$(realpath -e -- "${seed_root}")" = "${seed_root}" ]] \
  || die "recovery seed namespace is unsafe"
mkdir -- "${project_run}" || die "could not create recovery job namespace"
[[ -d "${project_run}" && ! -L "${project_run}" \
  && "$(realpath -e -- "${project_run}")" = "${project_run}" ]] \
  || die "recovery job namespace is unsafe"

atomic_copy_required() {
  local source="$1" name="$2" target temporary
  target="${project_run}/${name}"
  [[ -f "${source}" && ! -L "${source}" ]] || return 1
  [[ ! -e "${target}" && ! -L "${target}" ]] || return 1
  temporary="$(mktemp "${project_run}/.${name}.XXXXXX")"
  cp -p -- "${source}" "${temporary}" || return 1
  mv -T -- "${temporary}" "${target}" || return 1
  cmp -s -- "${source}" "${target}"
}

atomic_copy_optional() {
  local source="$1" name="$2"
  [[ ! -e "${source}" && ! -L "${source}" ]] && return 0
  atomic_copy_required "${source}" "${name}"
}

write_marker() {
  local status="$1" workload="$2" final="$3" temporary
  temporary="$(mktemp "${project_run}/.${status}.XXXXXX")"
  {
    printf 'schema_version=prorm-phase2-recovery-run-status/v1\n'
    printf 'status=%s\n' "${status}"
    printf 'workload_exit_code=%s\n' "${workload}"
    printf 'final_exit_code=%s\n' "${final}"
    printf 'array_job_id=%s\n' "${SLURM_ARRAY_JOB_ID}"
    printf 'array_task_id=%s\n' "${SLURM_ARRAY_TASK_ID}"
    printf 'seed=%s\n' "${seed}"
    printf 'execution_revision=%s\n' "${execution_revision}"
    printf 'retry_reason=pretrainer_hf_datasets_runtime_lock\n'
    printf 'recovery_design_sha256=%s\n' "${PRORM_PHASE2_RECOVERY_DESIGN_SHA256}"
    printf 'base_config_hash=%s\n' "${PRORM_PHASE2_BASE_CONFIG_HASH}"
    printf 'recovery_git_commit=%s\n' "${PRORM_GIT_COMMIT}"
    printf 'parent_design_sha256=%s\n' "${PRORM_PHASE2_PARENT_DESIGN_SHA256}"
    printf 'parent_registry_sha256=%s\n' "${PRORM_PHASE2_PARENT_REGISTRY_SHA256}"
    printf 'parent_producer_git_commit=%s\n' "${PRORM_PHASE2_PARENT_PRODUCER_GIT_COMMIT}"
    printf 'one_shot_no_further_adaptation=true\n'
    printf 'created_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } >"${temporary}"
  mv -T -- "${temporary}" "${project_run}/${status}"
}

finalize() {
  local workload=$? sync=0 final marker file
  trap - EXIT
  set +e
  if [[ "${workload}" = 0 ]]; then
    for file in \
      phase2-config-check.json base-config-check.json \
      infrastructure-failure-verification.json parent-verification-before.json \
      parent-verification-after.json artifact-snapshot-before.json \
      artifact-snapshot-after.json parent-run-snapshot-before.json \
      parent-run-snapshot-after.json hf-inventory-verification.json gpu-check.json \
      run-manifest.json env-report.log recovery-train.log recovery-result.json \
      recovery-output-verification.json; do
      atomic_copy_required "${job_dir}/${file}" "${file}" || sync=1
    done
    [[ ! -e "${job_dir}/recovery-failure-evidence.json" \
      && ! -L "${job_dir}/recovery-failure-evidence.json" ]] || sync=1
  else
    for file in \
      phase2-config-check.json base-config-check.json \
      infrastructure-failure-verification.json parent-verification-before.json \
      parent-verification-after.json artifact-snapshot-before.json \
      artifact-snapshot-after.json parent-run-snapshot-before.json \
      parent-run-snapshot-after.json hf-inventory-verification.json gpu-check.json \
      run-manifest.json env-report.log recovery-train.log recovery-result.json \
      recovery-failure-evidence.json recovery-output-verification.json; do
      atomic_copy_optional "${job_dir}/${file}" "${file}" || sync=1
    done
  fi
  final="${workload}"
  (( sync == 0 )) || [[ "${final}" != 0 ]] || final=1
  if [[ "${final}" = 0 ]]; then marker=SUCCESS; else marker=FAILED; fi
  write_marker "${marker}" "${workload}" "${final}" || final=1
  exit "${final}"
}
trap finalize EXIT

execution_repo="${job_dir}/source"
git clone --quiet --no-hardlinks --no-checkout "${PRORM_REPO_ROOT}" "${execution_repo}"
git -C "${execution_repo}" checkout --quiet --detach "${PRORM_GIT_COMMIT}"
[[ -z "$(git -C "${execution_repo}" status --porcelain --untracked-files=normal)" ]] \
  || die "detached recovery checkout is dirty"
git -C "${execution_repo}" merge-base --is-ancestor \
  "${PRORM_PHASE2_PARENT_PRODUCER_GIT_COMMIT}" "${PRORM_GIT_COMMIT}" \
  || die "parent producer is not an ancestor of recovery code"
git -C "${execution_repo}" merge-base --is-ancestor \
  "${PRORM_PHASE2_DIAGNOSTIC_GIT_COMMIT}" "${PRORM_GIT_COMMIT}" \
  || die "optimizer diagnostic is not an ancestor of recovery code"
materialization_paths=(
  configs/common_beta_pilot_base.yaml
  src/smart_reward/annotations.py src/smart_reward/artifacts.py
  src/smart_reward/config.py src/smart_reward/data.py src/smart_reward/hf.py
  src/smart_reward/oracle.py src/smart_reward/phase1.py
  src/smart_reward/phase1_rollout.py src/smart_reward/prompts.py
  src/smart_reward/scores.py src/smart_reward/seeding.py
)
git -C "${execution_repo}" diff --quiet \
  "${PRORM_PHASE2_PARENT_PRODUCER_GIT_COMMIT}" "${PRORM_GIT_COMMIT}" -- \
  "${materialization_paths[@]}" \
  || die "materialization-relevant code changed after parent production"

overlay="${execution_repo}/configs/common_beta_recovery_pilot.yaml"
base="${execution_repo}/configs/common_beta_pilot_base.yaml"
registry="${execution_repo}/configs/phase2_recovery_parent_failures.json"
validator="${execution_repo}/scripts/hpc4/validate_phase2_recovery_parent.py"
infrastructure_registry="${execution_repo}/configs/phase2_recovery_infrastructure_failure.json"
infrastructure_validator="${execution_repo}/scripts/hpc4/validate_phase2_recovery_infrastructure_failure.py"
printf '%s  %s\n' \
  a6a924dae429ceb0df11cea128542cae16fb42a2e69a0d2120acb0e4f8f1d80f "${overlay}" \
  e32cf5ad2a7bb2f6fa27180aa2fa6e05e2b457cfe032bab1c33f86646af1beb1 "${base}" \
  "${PRORM_PHASE2_PARENT_REGISTRY_SHA256}" "${registry}" \
  e09eefa403f72044192c58f19e06b1e89b939c1d35ddba5081b13693e995cafd \
  "${infrastructure_registry}" \
  | sha256sum --check --status || die "detached config/registry bytes changed"
python3 "${infrastructure_validator}" "${infrastructure_registry}" \
  --project-root "${PRORM_PROJECT_ROOT}" \
  --expected-registry-sha256 \
  e09eefa403f72044192c58f19e06b1e89b939c1d35ddba5081b13693e995cafd \
  --output "${job_dir}/infrastructure-failure-verification.json"
python3 "${validator}" "${registry}" --project-root "${PRORM_PROJECT_ROOT}" \
  --expected-registry-sha256 "${PRORM_PHASE2_PARENT_REGISTRY_SHA256}" \
  --expected-parent-design-sha256 "${PRORM_PHASE2_PARENT_DESIGN_SHA256}" \
  --expected-base-config-hash "${PRORM_PHASE2_BASE_CONFIG_HASH}" \
  --seed "${seed}" --verify-sources \
  --output "${job_dir}/parent-verification-before.json"
artifact="$(
  python3 -I -S - "${job_dir}/parent-verification-before.json" <<'PY'
import json, sys
from pathlib import Path
value=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(value["selected_seed"]["source_artifact_resolved"])
PY
)"
source_run="$(
  python3 -I -S - "${job_dir}/parent-verification-before.json" <<'PY'
import json, sys
from pathlib import Path
value=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(value["selected_seed"]["source_run_resolved"])
PY
)"
[[ -d "${artifact}" && ! -L "${artifact}" ]] || die "resolved parent artifact is unsafe"
[[ -d "${source_run}" && ! -L "${source_run}" ]] || die "resolved parent run is unsafe"

snapshot_artifact() {
  python3 -I -S - "$1" "$2" <<'PY'
import hashlib, json, os, stat, sys
from pathlib import Path
root=Path(sys.argv[1])
expected={"metadata.json","tensors.safetensors","candidates.jsonl","prompts.jsonl","training_edges.jsonl","evaluation_edges.jsonl"}
if {p.name for p in root.iterdir()} != expected:
    raise SystemExit("artifact directory schema changed")
records=[]
for name in sorted(expected):
    path=root/name
    info=path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise SystemExit("artifact entry is not a regular file")
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    records.append({"name":name,"sha256":digest.hexdigest(),"size":info.st_size,"mtime_ns":info.st_mtime_ns})
payload={"schema_version":"prorm-read-only-artifact-snapshot/v1","records":records}
with Path(sys.argv[2]).open("x",encoding="utf-8",newline="\n") as stream:
    json.dump(payload,stream,allow_nan=False,sort_keys=True,separators=(",",":")); stream.write("\n")
PY
}
snapshot_artifact "${artifact}" "${job_dir}/artifact-snapshot-before.json"

snapshot_parent_run() {
  python3 -I -S - "$1" "$2" <<'PY'
import hashlib, json, os, stat, sys
from pathlib import Path
root=Path(sys.argv[1])
names={"FAILED","run-manifest.json","artifact-materialization.json","artifact-verification.json","phase2-run.log"}
records=[]
for name in sorted(names):
    path=root/name
    info=path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise SystemExit("parent evidence entry is not a regular file")
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    records.append({"name":name,"sha256":digest,"size":info.st_size,"mtime_ns":info.st_mtime_ns})
link=root/"artifact"
info=link.lstat()
if not stat.S_ISLNK(info.st_mode):
    raise SystemExit("parent artifact binding is not a symlink")
records.append({"name":"artifact","symlink_target":os.readlink(link),"mtime_ns":info.st_mtime_ns})
payload={"schema_version":"prorm-read-only-parent-run-snapshot/v1","records":records}
with Path(sys.argv[2]).open("x",encoding="utf-8",newline="\n") as stream:
    json.dump(payload,stream,allow_nan=False,sort_keys=True,separators=(",",":")); stream.write("\n")
PY
}
snapshot_parent_run "${source_run}" "${job_dir}/parent-run-snapshot-before.json"

verification_datasets_cache="${job_dir}/hf-datasets-cache"
mkdir -m 700 -- "${verification_datasets_cache}" \
  || die "could not create per-job verification Datasets cache"
[[ -d "${verification_datasets_cache}" && ! -L "${verification_datasets_cache}" \
  && "$(realpath -e -- "${verification_datasets_cache}")" = "${verification_datasets_cache}" ]] \
  || die "per-job verification Datasets cache is unsafe"

container=(
  apptainer exec --cleanenv --nv --no-mount home,cwd,bind-paths
  --pwd "${execution_repo}"
  --bind "${job_dir}:${job_dir},${artifact}:/parent-artifact:ro,${PRORM_HF_CACHE}:${PRORM_HF_CACHE}:ro"
  --env "PYTHONPATH=${execution_repo}/src"
  --env "PYTHONNOUSERSITE=1" --env "PYTHONDONTWRITEBYTECODE=1"
  --env "HF_HOME=${PRORM_HF_CACHE}" --env "HF_HUB_CACHE=${PRORM_HF_CACHE}/hub"
  --env "HF_DATASETS_CACHE=${verification_datasets_cache}"
  --env "HF_HUB_OFFLINE=1" --env "TRANSFORMERS_OFFLINE=1"
  --env "HF_DATASETS_OFFLINE=1" --env "HF_HUB_DISABLE_TELEMETRY=1"
  --env "TOKENIZERS_PARALLELISM=false"
  --env "PRORM_GIT_COMMIT=${PRORM_GIT_COMMIT}"
  --env "PRORM_IMAGE_SHA256=${PRORM_IMAGE_SHA256}"
  --env "PRORM_HF_INVENTORY_SHA256=${PRORM_HF_INVENTORY_SHA256}"
)
for variable in \
  SLURM_JOB_ID SLURM_ARRAY_JOB_ID SLURM_ARRAY_TASK_ID SLURM_JOB_NAME \
  SLURM_JOB_ACCOUNT SLURM_CLUSTER_NAME SLURM_JOB_PARTITION SLURM_JOB_NODELIST \
  SLURM_NNODES SLURM_NTASKS SLURM_CPUS_PER_TASK SLURM_GPUS \
  SLURM_GPUS_ON_NODE SLURM_PROCID SLURM_LOCALID SLURM_NODEID \
  CUDA_VISIBLE_DEVICES NVIDIA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL; do
  [[ -v "${variable}" ]] && container+=(--env "${variable}=${!variable}")
done
container+=("${PRORM_IMAGE}")

"${container[@]}" python -m smart_reward.cli phase2-config-check "${overlay}" \
  | tee "${job_dir}/phase2-config-check.json"
"${container[@]}" python -m smart_reward.cli config-check "${base}" \
  | tee "${job_dir}/base-config-check.json"
python3 -I -S - \
  "${job_dir}/phase2-config-check.json" "${job_dir}/base-config-check.json" \
  "${PRORM_PHASE2_RECOVERY_DESIGN_SHA256}" "${PRORM_PHASE2_BASE_CONFIG_HASH}" <<'PY'
import json,sys
from pathlib import Path
phase2=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
base=json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
if (
 phase2.get("status")!="ok" or phase2.get("phase2_design_sha256")!=sys.argv[3]
 or phase2.get("source_config_hash")!=sys.argv[4]
 or phase2.get("design_stage")!="pilot" or phase2.get("formal_eligibility") is not False
 or base.get("status")!="ok" or base.get("config_hash")!=sys.argv[4]
):
 raise SystemExit("container-computed recovery config identities are invalid")
PY
"${container[@]}" python "${execution_repo}/scripts/hpc4/stage_hf_assets.py" \
  "${base}" "${PRORM_HF_CACHE}" --verify-only --inventory "${PRORM_HF_INVENTORY}" \
  --verification-datasets-cache "${verification_datasets_cache}" \
  | tee "${job_dir}/hf-inventory-verification.json"
printf '%s  %s\n' "${PRORM_HF_INVENTORY_SHA256}" "${PRORM_HF_INVENTORY}" \
  | sha256sum --check --status || die "HF inventory changed during in-container verification"
"${container[@]}" python - <<'PY' | tee "${job_dir}/gpu-check.json"
import json, torch
if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
    raise SystemExit("recovery requires exactly one visible CUDA GPU")
name=torch.cuda.get_device_name(0)
if name!="NVIDIA L20":
    raise SystemExit(f"expected NVIDIA L20, observed {name!r}")
print(json.dumps({"status":"ok","gpu_model":name,"cuda_device_count":1},sort_keys=True))
PY

"${container[@]}" python -m smart_reward.cli env-report \
  "${base}" --seed "${seed}" --repo-root "${execution_repo}" \
  --output "${job_dir}/run-manifest.json" | tee "${job_dir}/env-report.log"
"${container[@]}" env \
  "PRORM_FAILURE_EVIDENCE=${job_dir}/recovery-failure-evidence.json" \
  python "${execution_repo}/scripts/hpc4/run_phase2_recovery_train.py" \
  "${overlay}" "${registry}" /parent-artifact "${job_dir}/run-manifest.json" \
  "${job_dir}/recovery-result.json" --seed "${seed}" --device cuda \
  2>&1 | tee "${job_dir}/recovery-train.log"

python3 "${validator}" "${registry}" --project-root "${PRORM_PROJECT_ROOT}" \
  --expected-registry-sha256 "${PRORM_PHASE2_PARENT_REGISTRY_SHA256}" \
  --expected-parent-design-sha256 "${PRORM_PHASE2_PARENT_DESIGN_SHA256}" \
  --expected-base-config-hash "${PRORM_PHASE2_BASE_CONFIG_HASH}" \
  --seed "${seed}" --verify-sources \
  --output "${job_dir}/parent-verification-after.json"
snapshot_artifact "${artifact}" "${job_dir}/artifact-snapshot-after.json"
snapshot_parent_run "${source_run}" "${job_dir}/parent-run-snapshot-after.json"
cmp -s "${job_dir}/artifact-snapshot-before.json" "${job_dir}/artifact-snapshot-after.json" \
  || die "parent artifact content or mtime changed during recovery"
cmp -s "${job_dir}/parent-run-snapshot-before.json" \
  "${job_dir}/parent-run-snapshot-after.json" \
  || die "parent FAILED evidence content or mtime changed during recovery"

python3 -I -S - \
  "${job_dir}/recovery-result.json" "${job_dir}/recovery-output-verification.json" \
  "${seed}" "${PRORM_PHASE2_RECOVERY_DESIGN_SHA256}" \
  "${PRORM_PHASE2_BASE_CONFIG_HASH}" "${PRORM_GIT_COMMIT}" \
  "${PRORM_PHASE2_PARENT_DESIGN_SHA256}" \
  "${PRORM_PHASE2_PARENT_REGISTRY_SHA256}" <<'PY'
import hashlib,json,sys
from pathlib import Path
raw,out,seed,design,base,commit,parent,registry=sys.argv[1:]
value=json.loads(Path(raw).read_text(encoding="utf-8"))
boundary=value.get("information_boundary",{})
training=value.get("head_training",{})
audit=training.get("audit",{}) if isinstance(training,dict) else {}
primary=audit.get("primary_heads",{}) if isinstance(audit,dict) else {}
schedule_sha="46e0b0fdc70c507b0325c445068326ba7bc30326d70b65fa33803cc3c876c216"
tie_break="exact_zero_initialized_deterministic_adamw_lr_decay_path"
hex_digits=set("0123456789abcdef")
def is_digest(item):
    return isinstance(item,str) and len(item)==64 and set(item)<=hex_digits
def as_mapping(item,name):
    if not isinstance(item,dict):
        raise SystemExit(f"{name} is not a mapping")
    return item
def validate_measurement(item,name):
    measurement=as_mapping(item,name)
    if measurement.get("audit_dtype")!="float64":
        raise SystemExit(f"{name} is not a serialized FP64 audit")
def validate_head(name,head):
    head=as_mapping(head,name)
    convergence=head.get("first_order_convergence",{}) if isinstance(head,dict) else {}
    spec=convergence.get("spec",{}) if isinstance(convergence,dict) else {}
    protocol=spec.get("optimizer_protocol",{}) if isinstance(spec,dict) else {}
    schedule=protocol.get("learning_rate_schedule",{}) if isinstance(protocol,dict) else {}
    execution=convergence.get("optimizer_protocol_execution",{}) if isinstance(convergence,dict) else {}
    checks=convergence.get("checks") if isinstance(convergence,dict) else None
    fixed=convergence.get("fixed_step_compute_matched_snapshot",{})
    legacy=convergence.get("legacy_constant_lr_boundary_snapshot",{})
    final=convergence.get("final_gate",{})
    if (
        convergence.get("schema_version")!="objective-first-order-convergence/v2"
        or convergence.get("converged") is not True
        or convergence.get("fail_closed") is not True
        or spec.get("schema_version")!="objective-first-order-convergence-spec/v2"
        or spec.get("max_steps")!=12760
        or schedule.get("schedule_sha256")!=schedule_sha
        or protocol.get("one_time_recovery") is not True
        or protocol.get("tie_break")!=tie_break
        or not isinstance(checks,list) or not checks
        or fixed.get("step")!=720
        or legacy.get("step")!=5760
        or execution.get("schema_version")!="deterministic-adamw-lr-decay-execution/v2"
        or execution.get("selected_primary_optimizer_state_restored_and_verified") is not True
        or execution.get("selected_optimizer_object_identity_preserved") is not True
        or execution.get("selected_optimizer_moments_restored_and_verified") is not True
    ):
        raise SystemExit(f"recovery {name} convergence evidence is invalid")
    equal_hash_pairs=(
        ("selected_head_sha256","restored_head_sha256"),
        ("selected_optimizer_state_sha256","restored_optimizer_state_sha256"),
        (
            "selected_checkpoint_optimizer_state_dict_sha256",
            "restored_optimizer_state_dict_sha256",
        ),
    )
    for left,right in equal_hash_pairs:
        if (
            not is_digest(execution.get(left))
            or not is_digest(execution.get(right))
            or execution[left]!=execution[right]
        ):
            raise SystemExit(f"recovery {name} restored-state hash mismatch: {left}")
    if not is_digest(execution.get("selected_checkpoint_sha256")):
        raise SystemExit(f"recovery {name} selected checkpoint digest is invalid")
    state_checks=as_mapping(execution.get("per_update_state_checks"),f"{name}.state_checks")
    completed=execution.get("completed_updates_observed")
    if (
        isinstance(completed,bool) or not isinstance(completed,int)
        or completed<5760 or completed>12760
        or state_checks.get("before_update_checks")!=completed
        or state_checks.get("after_update_checks")!=completed
        or state_checks.get("completed_updates_covered")!=completed
        or state_checks.get("first_pre_update_state_empty") is not True
        or state_checks.get("all_updates_checked_before_and_after") is not True
        or state_checks.get("all_subsequent_pre_update_scalar_steps_exact") is not True
        or state_checks.get("all_post_update_scalar_steps_exact") is not True
        or state_checks.get("exp_avg_and_exp_avg_sq_shape_dtype_device_valid") is not True
        or not is_digest(state_checks.get("check_sequence_sha256"))
    ):
        raise SystemExit(f"recovery {name} per-update AdamW evidence is invalid")
    validate_measurement(
        convergence.get("initial_zero_head_measurement"),
        f"{name}.initial",
    )
    validate_measurement(final.get("measurement"),f"{name}.final")
    validate_measurement(fixed.get("measurement"),f"{name}.fixed720")
    validate_measurement(legacy.get("measurement"),f"{name}.legacy5760")
    for index,check in enumerate(checks):
        check=as_mapping(check,f"{name}.checks[{index}]")
        if check.get("learning_rate_schedule_sha256")!=schedule_sha:
            raise SystemExit(f"recovery {name} check schedule binding is invalid")
        validate_measurement(check.get("measurement"),f"{name}.checks[{index}].measurement")
    return convergence.get("selected_primary_step")
head_entries={
    "primary_bt_mle": primary.get("bt_mle") if isinstance(primary,dict) else None,
    "primary_prorm_plus": primary.get("prorm_plus") if isinstance(primary,dict) else None,
    "low_dimensional_prorm_plus": (
        audit.get("low_dimensional_control",{}).get("head")
        if isinstance(audit.get("low_dimensional_control"),dict) else None
    ),
    "exact_margin_prorm_plus": (
        audit.get("exact_margin_control",{}).get("head")
        if isinstance(audit.get("exact_margin_control"),dict) else None
    ),
    "exact_soft_label_bt": (
        audit.get("exact_soft_label_bt_control",{}).get("head")
        if isinstance(audit.get("exact_soft_label_bt_control"),dict) else None
    ),
}
selected_steps={name:validate_head(name,head) for name,head in head_entries.items()}
optimization=as_mapping(audit.get("primary_optimization_audit"),"primary_optimization_audit")
reward_rank=as_mapping(optimization.get("reward_head_identifiability"),"reward_head_rank")
moment_rank=as_mapping(optimization.get("prorm_moment_map_identifiability"),"moment_rank")
low_control=as_mapping(audit.get("low_dimensional_control"),"low_dimensional_control")
low_rank=as_mapping(
    low_control.get("projected_prorm_moment_map_identifiability"),
    "projected_moment_rank",
)
if (
    reward_rank.get("schema_version")!="reward-head-identifiability/v2"
    or reward_rank.get("algorithmic_tie_break")!=tie_break
    or reward_rank.get("full_design_rank_proves_finite_bt_minimizer_exists") is not False
    or not isinstance(reward_rank.get("mixed_outcome_edge_coercivity_diagnostic"),dict)
    or moment_rank.get("schema_version")!="prorm-moment-map-identifiability/v2"
    or moment_rank.get("algorithmic_tie_break")!=tie_break
    or low_rank.get("schema_version")!="projected-prorm-moment-map-identifiability/v2"
    or low_rank.get("algorithmic_tie_break")!=tie_break
):
    raise SystemExit("recovery identifiability/tie-break evidence is invalid")
label=as_mapping(audit.get("label_stream"),"label_stream")
rescore=as_mapping(value.get("train_oracle_rescore"),"train_oracle_rescore")
if (
    not is_digest(rescore.get("transformed_rewards_sha256"))
    or rescore.get("raw_oracle_logits_serialized") is not False
    or label.get("oracle_reward_sha256")!=rescore.get("transformed_rewards_sha256")
):
    raise SystemExit("recovery oracle/label reward identity is invalid")
anchor_applicable=int(seed)==20260801
anchor_passed=None
if anchor_applicable:
    expected_anchor={
        "namespace":"prorm-common-beta-r4-labels-v1",
        "base_seed":20260801,
        "derived_seed":2443486425476852717,
        "derivation_sha256":"a1e901f534096bedb757ba978e2ba9838031aeacab4e86265557279953b236ae",
        "initial_state_sha256":"6f8e7260e641e4f52990e6f28c6558f333133bd0286549dbca3de426bf51a3d1",
        "final_state_sha256":"7dbc6a6143c98a995abda1baab73de8867267121c1299d80b6c10f8165f6ce83",
        "mean_h_sha256":"524eb0c9936dffe8d0ef807d4b4181cb3c4bbad556a51c7578a16c06b1e13cf0",
        "replicate_count_sha256":"b905ce98d6ec87a03bbda10405e3ffc766ff913bf9613e58bb50bf1ffa7b63c8",
        "replicate_win_sha256":"de68ae122cfe40a09146fdd00f367b640bdcc03a12aff46e6713fe7e88cafb13",
        "replicate_h_sha256":"92a99a227ce5679049c856b3fbd92005d0d9a8460760bc346a5ef266d7d3350d",
        "realized_total_annotations":61011,
    }
    if (
        rescore.get("transformed_rewards_sha256")
        !="7a7d7b005ec7e377205d6f40743bed950ad38154dec6f54516f7ced8ffca0b1a"
        or any(label.get(key)!=expected for key,expected in expected_anchor.items())
    ):
        raise SystemExit("diagnostic seed oracle/label reproducibility anchor failed")
    anchor_passed=True
execution_identity=as_mapping(
    value.get("recovery_execution_identity"),
    "recovery_execution_identity",
)
parent_binding=as_mapping(value.get("parent_failure_binding"),"parent_failure_binding")
parent_producer=as_mapping(
    parent_binding.get("parent_artifact_producer"),
    "parent_artifact_producer",
)
if (
    value.get("schema_version")!="prorm-phase2-recovery-train-only-result/v1"
    or value.get("status")!="SUCCESS" or value.get("seed")!=int(seed)
    or value.get("recovery_design_sha256")!=design
    or value.get("source_config_hash")!=base
    or execution_identity.get("formal") is not True
    or execution_identity.get("git_commit")!=commit
    or execution_identity.get("image_sha256")
       !="d6fc044b4fa303747908783ea057d5b8946f613bfec6a6ca301e3a02fd7719cb"
    or execution_identity.get("hf_inventory_sha256")
       !="86c7c0fcab9cc0de612c6a5af05778e8b34617822b2e33474df8ed840eef82fd"
    or execution_identity.get("account")!="sigroup"
    or execution_identity.get("partition")!="gpu-l20"
    or execution_identity.get("gpu_models")!=["NVIDIA L20"]
    or parent_binding.get("parent_phase2_design_sha256")!=parent
    or parent_binding.get("registry_sha256")!=registry
    or parent_producer.get("git_commit")!="ae28e2a10f0bd5762899be01ce66bc5b423374cf"
    or parent_producer.get("image_sha256")
       !="d6fc044b4fa303747908783ea057d5b8946f613bfec6a6ca301e3a02fd7719cb"
    or parent_producer.get("hf_inventory_sha256")
       !="86c7c0fcab9cc0de612c6a5af05778e8b34617822b2e33474df8ed840eef82fd"
    or value.get("one_shot_no_further_adaptation") is not True
    or training.get("schema_version")!="phase2-fresh-head-training/v3"
    or audit.get("schema_version")!="phase2-fresh-head-training/v3"
    or audit.get("training_design_sha256")!=design
    or audit.get("training_settings_sha256")
       !="34574e1b1dc22a9503b89249059596d92aa5c3df074022ecfc8ff008dc4bc3af"
    or not is_digest(audit.get("training_instance_sha256"))
    or not is_digest(audit.get("input_training_sha256"))
    or set(training.get("heads",{}))!={"bt_mle","prorm_plus"}
    or boundary.get("validation_tensors_decoded") is not False
    or boundary.get("test_tensors_decoded") is not False
    or boundary.get("validation_or_test_candidates_decoded") is not False
    or boundary.get("policy_session_opened") is not False
    or boundary.get("policy_rollout_performed") is not False
    or boundary.get("heldout_evaluator_called") is not False
    or boundary.get("final_oracle_session_opened") is not False
    or boundary.get("downstream_utility_computed") is not False
):
    raise SystemExit("recovery output identity or train-only boundary is invalid")
payload={
    "status":"ok",
    "result_sha256":hashlib.sha256(Path(raw).read_bytes()).hexdigest(),
    "five_head_recovery_protocol_verified":True,
    "selected_primary_steps":selected_steps,
    "diagnostic_seed_reproduction":{
        "anchor_seed":20260801,
        "applicable":anchor_applicable,
        "passed":anchor_passed,
    },
}
with Path(out).open("x",encoding="utf-8",newline="\n") as stream:
    json.dump(payload,stream,sort_keys=True,separators=(",",":")); stream.write("\n")
PY

artifact_relative="$(realpath --relative-to="${project_run}" "${artifact}")"
ln -s -- "${artifact_relative}" "${project_run}/parent-artifact"
