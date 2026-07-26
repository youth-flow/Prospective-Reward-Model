#!/usr/bin/env bash
# This is an exploratory-only, fixed-three end-to-end array.  It is intentionally
# not a formal-wave entry point and does not consume a formal registry.
#SBATCH --job-name=prorm-p2-budgeted-e2e
#SBATCH --account=sigroup
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gpus-per-node=1
#SBATCH --no-requeue

set -euo pipefail
umask 027

readonly BUDGETED_STAGE="budgeted_end_to_end"
readonly BUDGETED_EVIDENCE_ROLE="budgeted_end_to_end_fixed_three_exploratory_only"
readonly ADOPTED_SCHEDULE_SHA256="46e0b0fdc70c507b0325c445068326ba7bc30326d70b65fa33803cc3c876c216"
readonly SBATCH_RELATIVE="scripts/hpc4/phase2_budgeted_end_to_end.sbatch"
readonly R3_AUTHORIZATION_RELATIVE="runs/phase2-recovery-r3-controls/gate-c-success-authorization.json"
readonly R3_BIND_PLAN_RELATIVE="scripts/hpc4/phase2_budgeted_r3_bind_plan_stdlib.py"
readonly HOST_PYTHON="/opt/shared/spack/local/linux-rocky9-x86_64_v4/gcc-11.4.1/miniconda3-24.3.0-quc3pyudmzikgo2r4qsyqpwnrvzpin63/bin/python3.12"
readonly HOST_PYTHON_SHA256="9c91f9aa231c61c6bf2eabb9b93ebc5a8269a4126a36125e9548d8853e32da9c"
verification_complete=0

die() { echo "error: $*" >&2; exit 2; }
sha256() { sha256sum -- "$1" | awk '{print $1}'; }

for name in \
  PRORM_PROJECT_ROOT PRORM_SCRATCH_ROOT PRORM_REPO_ROOT \
  PRORM_IMAGE PRORM_IMAGE_SHA256 PRORM_HF_CACHE \
  PRORM_HF_INVENTORY PRORM_HF_INVENTORY_SHA256 \
  PRORM_BUDGETED_OVERLAY_REL PRORM_BUDGETED_BASE_REL \
  PRORM_BUDGETED_OVERLAY_SHA256 PRORM_BUDGETED_BASE_SHA256 \
  PRORM_BUDGETED_DESIGN_SHA256 PRORM_BUDGETED_BASE_CONFIG_HASH \
  PRORM_RECOVERY_AUTHORIZATION PRORM_RECOVERY_AUTHORIZATION_SHA256 \
  PRORM_OPTIMIZER_SCHEDULE_SHA256 PRORM_BUDGETED_FROZEN_BETA \
  PRORM_BUDGETED_FREEZE_EVIDENCE PRORM_BUDGETED_FREEZE_EVIDENCE_SHA256 \
  PRORM_GIT_COMMIT; do
  [[ -n "${!name:-}" ]] || die "${name} is required"
done
: "${SLURM_JOB_ID:?budgeted run must execute inside Slurm}"
: "${SLURM_ARRAY_JOB_ID:?budgeted run must execute as a Slurm array}"
: "${SLURM_ARRAY_TASK_ID:?budgeted run requires an array task}"

[[ "${SLURM_CLUSTER_NAME:-}" = hpc4 ]] || die "cluster must be hpc4"
[[ "${SLURM_JOB_ACCOUNT:-}" = sigroup ]] || die "account must be sigroup"
[[ "${SLURM_JOB_PARTITION:-}" = gpu-l20 ]] || die "partition must be gpu-l20"
[[ "${SLURM_RESTART_COUNT:-0}" = 0 ]] || die "requeue/restart is forbidden"
[[ "${SLURM_NNODES:-}" = 1 && "${SLURM_NTASKS:-}" = 1 ]] || die "allocation must be 1 node / 1 task"
case "${SLURM_GPUS_ON_NODE:-}" in 1|gpu:1) ;; *) die "exactly one GPU is required" ;; esac
[[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != *,* ]] || die "exactly one visible GPU is required"
[[ "${SLURM_ARRAY_TASK_COUNT:-}" = 3 ]] || die "array must contain exactly three tasks"
[[ "${SLURM_ARRAY_TASK_MIN:-}" = 0 && "${SLURM_ARRAY_TASK_MAX:-}" = 2 && "${SLURM_ARRAY_TASK_STEP:-}" = 1 ]] || die "array must be exactly 0-2 step 1"
[[ "${SLURM_ARRAY_TASK_ID}" =~ ^[0-2]$ ]] || die "array task must be 0..2"

for command_name in apptainer git realpath sha256sum awk mktemp cp rm ln id; do
  command -v "${command_name}" >/dev/null 2>&1 || die "missing command: ${command_name}"
done
host_python="$(realpath -e -- "${HOST_PYTHON}")" \
  || die "fixed host Python is unavailable"
[[ "${host_python}" == "${HOST_PYTHON}" && -f "${host_python}" \
  && -x "${host_python}" && ! -L "${host_python}" ]] \
  || die "fixed host Python must be a canonical executable"
printf '%s  %s\n' "${HOST_PYTHON_SHA256}" "${host_python}" \
  | sha256sum --check --status || die "fixed host Python SHA-256 mismatch"
[[ "$("${host_python}" --version 2>&1)" == "Python 3.12.2" ]] \
  || die "fixed host Python version mismatch"
while IFS= read -r variable; do
  case "${variable}" in APPTAINER*|SINGULARITY*) die "ambient container control is forbidden: ${variable}" ;; esac
done < <(compgen -e)

require_dir() {
  local name="$1" path="${!1}" resolved
  [[ "${path}" = /* ]] || die "${name} must be absolute"
  resolved="$(realpath -e -- "${path}")" || die "${name} cannot be resolved"
  [[ -d "${resolved}" && ! -L "${path}" && "${resolved}" = "${path}" && "${resolved}" != / ]] || die "${name} must be a canonical real directory"
}
require_file() {
  local name="$1" path="${!1}" resolved
  [[ "${path}" = /* ]] || die "${name} must be absolute"
  resolved="$(realpath -e -- "${path}")" || die "${name} cannot be resolved"
  [[ -f "${resolved}" && ! -L "${path}" && "${resolved}" = "${path}" ]] || die "${name} must be a canonical regular file"
}
contained() { case "$2" in "$3"/*) ;; *) die "$1 leaves its required root" ;; esac; }
for name in PRORM_PROJECT_ROOT PRORM_SCRATCH_ROOT PRORM_REPO_ROOT PRORM_HF_CACHE; do require_dir "${name}"; done
for name in PRORM_IMAGE PRORM_HF_INVENTORY PRORM_RECOVERY_AUTHORIZATION PRORM_BUDGETED_FREEZE_EVIDENCE; do require_file "${name}"; done
contained image "${PRORM_IMAGE}" "${PRORM_PROJECT_ROOT}"
contained inventory "${PRORM_HF_INVENTORY}" "${PRORM_HF_CACHE}"
contained authorization "${PRORM_RECOVERY_AUTHORIZATION}" "${PRORM_PROJECT_ROOT}"
[[ "${PRORM_RECOVERY_AUTHORIZATION}" == \
  "${PRORM_PROJECT_ROOT}/${R3_AUTHORIZATION_RELATIVE}" ]] \
  || die "GateE requires the exact fixed R3 combined authorization"
contained freeze-evidence "${PRORM_BUDGETED_FREEZE_EVIDENCE}" "${PRORM_PROJECT_ROOT}"
case "${PRORM_PROJECT_ROOT}" in "${PRORM_SCRATCH_ROOT}"|"${PRORM_SCRATCH_ROOT}"/*) die "project and scratch roots overlap" ;; esac
case "${PRORM_SCRATCH_ROOT}" in "${PRORM_PROJECT_ROOT}"|"${PRORM_PROJECT_ROOT}"/*) die "project and scratch roots overlap" ;; esac
for name in PRORM_IMAGE_SHA256 PRORM_HF_INVENTORY_SHA256 PRORM_BUDGETED_OVERLAY_SHA256 PRORM_BUDGETED_BASE_SHA256 PRORM_BUDGETED_DESIGN_SHA256 PRORM_BUDGETED_BASE_CONFIG_HASH PRORM_RECOVERY_AUTHORIZATION_SHA256 PRORM_BUDGETED_FREEZE_EVIDENCE_SHA256; do
  [[ "${!name}" =~ ^[0-9a-f]{64}$ ]] || die "${name} must be a lowercase SHA256"
done
[[ "${PRORM_GIT_COMMIT}" =~ ^[0-9a-f]{40,64}$ ]] || die "PRORM_GIT_COMMIT must be a full lowercase object id"
[[ "${PRORM_OPTIMIZER_SCHEDULE_SHA256}" = "${ADOPTED_SCHEDULE_SHA256}" ]] || die "optimizer schedule is not the adopted recovery schedule"
"${host_python}" - "${PRORM_BUDGETED_FROZEN_BETA}" <<'PY'
import math, sys
value = float(sys.argv[1])
if not math.isfinite(value) or value <= 0:
    raise SystemExit("frozen beta must be a finite positive scalar")
PY
[[ "${PRORM_BUDGETED_OVERLAY_REL}" = configs/common_beta_post_recovery_budgeted_end_to_end.yaml ]] || die "unexpected budgeted overlay"
[[ "${PRORM_BUDGETED_BASE_REL}" = configs/common_beta_post_recovery_budgeted_end_to_end_base.yaml ]] || die "unexpected budgeted base"
for pair in \
  "${PRORM_IMAGE}:${PRORM_IMAGE_SHA256}" \
  "${PRORM_HF_INVENTORY}:${PRORM_HF_INVENTORY_SHA256}" \
  "${PRORM_RECOVERY_AUTHORIZATION}:${PRORM_RECOVERY_AUTHORIZATION_SHA256}" \
  "${PRORM_BUDGETED_FREEZE_EVIDENCE}:${PRORM_BUDGETED_FREEZE_EVIDENCE_SHA256}"; do
  path="${pair%%:*}"; digest="${pair#*:}"
  [[ "$(sha256 "${path}")" = "${digest}" ]] || die "immutable input changed: ${path}"
done

submission_head="$(git -C "${PRORM_REPO_ROOT}" rev-parse --verify HEAD)"
[[ "${submission_head}" = "${PRORM_GIT_COMMIT}" && -z "$(git -C "${PRORM_REPO_ROOT}" status --porcelain --untracked-files=normal)" ]] || die "submission checkout must be clean and pinned"
for pair in \
  "${PRORM_BUDGETED_OVERLAY_REL}:${PRORM_BUDGETED_OVERLAY_SHA256}" \
  "${PRORM_BUDGETED_BASE_REL}:${PRORM_BUDGETED_BASE_SHA256}" \
  "${SBATCH_RELATIVE}:$(git -C "${PRORM_REPO_ROOT}" cat-file blob "${PRORM_GIT_COMMIT}:${SBATCH_RELATIVE}" | sha256sum | awk '{print $1}')"; do
  relative="${pair%%:*}"; expected="${pair#*:}"
  observed="$(git -C "${PRORM_REPO_ROOT}" cat-file blob "${PRORM_GIT_COMMIT}:${relative}" | sha256sum | awk '{print $1}')"
  [[ "${observed}" = "${expected}" ]] || die "committed bytes mismatch: ${relative}"
done
[[ "$(sha256 "$0")" = "$(git -C "${PRORM_REPO_ROOT}" cat-file blob "${PRORM_GIT_COMMIT}:${SBATCH_RELATIVE}" | sha256sum | awk '{print $1}')" ]] \
  || die "running sbatch script differs from committed bytes"

seeds=(20261001 20261002 20261003)
seed="${seeds[SLURM_ARRAY_TASK_ID]}"
execution_id="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
campaign_root="${PRORM_PROJECT_ROOT}/runs/phase2-budgeted-end-to-end/${PRORM_BUDGETED_DESIGN_SHA256}"
artifact_root="${PRORM_PROJECT_ROOT}/artifacts/phase2-budgeted-end-to-end/${PRORM_BUDGETED_DESIGN_SHA256}"
ledger="${campaign_root}/submission-ledger"
job_dir="${PRORM_SCRATCH_ROOT}/phase2-budgeted-end-to-end-jobs/${execution_id}"
seed_root="${campaign_root}/seed-${seed}"
project_run="${seed_root}/job-${execution_id}"
project_artifact="${artifact_root}/seed-${seed}/job-${execution_id}"

# Reconstruct the exact submission export specification from the received
# environment.  The ledger stores the SHA256 of this string; carrying that
# digest inside the string itself would create an impossible self-reference.
for name in \
  PRORM_PROJECT_ROOT PRORM_SCRATCH_ROOT PRORM_REPO_ROOT PRORM_IMAGE \
  PRORM_IMAGE_SHA256 PRORM_HF_CACHE PRORM_HF_INVENTORY \
  PRORM_HF_INVENTORY_SHA256 PRORM_BUDGETED_OVERLAY_REL \
  PRORM_BUDGETED_BASE_REL PRORM_BUDGETED_OVERLAY_SHA256 \
  PRORM_BUDGETED_BASE_SHA256 PRORM_BUDGETED_DESIGN_SHA256 \
  PRORM_BUDGETED_BASE_CONFIG_HASH PRORM_RECOVERY_AUTHORIZATION \
  PRORM_RECOVERY_AUTHORIZATION_SHA256 PRORM_OPTIMIZER_SCHEDULE_SHA256 \
  PRORM_BUDGETED_FROZEN_BETA PRORM_BUDGETED_FREEZE_EVIDENCE \
  PRORM_BUDGETED_FREEZE_EVIDENCE_SHA256 PRORM_GIT_COMMIT; do
  case "${!name}" in
    *","*|*"="*|*$'\n'*|*$'\r'*) die "${name} is unsafe for the canonical export specification" ;;
  esac
done
runtime_export_spec="PATH=/usr/local/bin:/usr/bin:/bin,PRORM_PROJECT_ROOT=${PRORM_PROJECT_ROOT},PRORM_SCRATCH_ROOT=${PRORM_SCRATCH_ROOT},PRORM_REPO_ROOT=${PRORM_REPO_ROOT},PRORM_IMAGE=${PRORM_IMAGE},PRORM_IMAGE_SHA256=${PRORM_IMAGE_SHA256},PRORM_HF_CACHE=${PRORM_HF_CACHE},PRORM_HF_INVENTORY=${PRORM_HF_INVENTORY},PRORM_HF_INVENTORY_SHA256=${PRORM_HF_INVENTORY_SHA256},PRORM_BUDGETED_OVERLAY_REL=${PRORM_BUDGETED_OVERLAY_REL},PRORM_BUDGETED_BASE_REL=${PRORM_BUDGETED_BASE_REL},PRORM_BUDGETED_OVERLAY_SHA256=${PRORM_BUDGETED_OVERLAY_SHA256},PRORM_BUDGETED_BASE_SHA256=${PRORM_BUDGETED_BASE_SHA256},PRORM_BUDGETED_DESIGN_SHA256=${PRORM_BUDGETED_DESIGN_SHA256},PRORM_BUDGETED_BASE_CONFIG_HASH=${PRORM_BUDGETED_BASE_CONFIG_HASH},PRORM_RECOVERY_AUTHORIZATION=${PRORM_RECOVERY_AUTHORIZATION},PRORM_RECOVERY_AUTHORIZATION_SHA256=${PRORM_RECOVERY_AUTHORIZATION_SHA256},PRORM_OPTIMIZER_SCHEDULE_SHA256=${PRORM_OPTIMIZER_SCHEDULE_SHA256},PRORM_BUDGETED_FROZEN_BETA=${PRORM_BUDGETED_FROZEN_BETA},PRORM_BUDGETED_FREEZE_EVIDENCE=${PRORM_BUDGETED_FREEZE_EVIDENCE},PRORM_BUDGETED_FREEZE_EVIDENCE_SHA256=${PRORM_BUDGETED_FREEZE_EVIDENCE_SHA256},PRORM_GIT_COMMIT=${PRORM_GIT_COMMIT}"
runtime_export_spec_sha256="$(
  printf '%s' "${runtime_export_spec}" | sha256sum | awk '{print $1}'
)"

# The submitter API is intentionally invoked directly below: it binds this
# task to the exact held-and-ledgered array, rather than trusting a path alone.
mapfile -t submission_hashes < <(PYTHONPATH="${PRORM_REPO_ROOT}/src" "${host_python}" - "${ledger}" "${PRORM_PROJECT_ROOT}" "${PRORM_REPO_ROOT}" "${PRORM_BUDGETED_DESIGN_SHA256}" "${PRORM_BUDGETED_BASE_CONFIG_HASH}" "${PRORM_RECOVERY_AUTHORIZATION_SHA256}" "${PRORM_OPTIMIZER_SCHEDULE_SHA256}" "${PRORM_GIT_COMMIT}" "${PRORM_IMAGE_SHA256}" "${PRORM_HF_INVENTORY_SHA256}" "${PRORM_BUDGETED_OVERLAY_SHA256}" "${PRORM_BUDGETED_BASE_SHA256}" "${runtime_export_spec_sha256}" "${SLURM_ARRAY_JOB_ID}" "$(id -un)" <<'PY'
import importlib.util, json, sys
# Load from the committed submission checkout, not from a package registry.
path = sys.argv[3] + "/scripts/hpc4/submit_phase2_budgeted_end_to_end_once.py"
spec = importlib.util.spec_from_file_location("budgeted_submit", path)
if spec is None or spec.loader is None: raise SystemExit("cannot load budgeted submitter")
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
value = module.verify_submission_ledger(
    __import__("pathlib").Path(sys.argv[1]), project_root=__import__("pathlib").Path(sys.argv[2]), repo_root=__import__("pathlib").Path(sys.argv[3]),
    design_sha256=sys.argv[4], base_config_hash=sys.argv[5], authorization_sha256=sys.argv[6], optimizer_schedule_sha256=sys.argv[7], git_commit=sys.argv[8], image_sha256=sys.argv[9], inventory_sha256=sys.argv[10], overlay_sha256=sys.argv[11], base_file_sha256=sys.argv[12], export_spec_sha256=sys.argv[13], array_job_id=sys.argv[14], submitter_user=sys.argv[15])
if value.get("status") != "verified": raise SystemExit("ledger is not verified")
print(value["intent_sha256"]); print(value["submission_sha256"])
PY
)
[[ "${#submission_hashes[@]}" = 2 ]] || die "submission ledger verification failed"
submission_intent_sha256="${submission_hashes[0]}"; submission_ledger_sha256="${submission_hashes[1]}"

# Ledger verification deliberately precedes every seed/job namespace mutation:
# a malformed or cross-bound array must never leave a permanent occupied slot.
for path in "${job_dir}" "${seed_root}" "${project_run}" "${project_artifact}"; do [[ ! -e "${path}" && ! -L "${path}" ]] || die "fresh budgeted namespace already exists: ${path}"; done
mkdir -p -- "$(dirname -- "${job_dir}")" "${seed_root}" "$(dirname -- "${project_artifact}")"
mkdir -- "${job_dir}" "${project_run}"

fsync_directory() {
  "${host_python}" -I -S - "$1" <<'PY'
import os
import stat
import sys
from pathlib import Path

path = Path(sys.argv[1]).absolute()
metadata = path.lstat()
if (
    not stat.S_ISDIR(metadata.st_mode)
    or path.is_symlink()
    or path.resolve(strict=True) != path
):
    raise SystemExit("fsync target must be a canonical real directory")
descriptor = os.open(
    path,
    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
)
try:
    os.fsync(descriptor)
finally:
    os.close(descriptor)
PY
}
install_file_no_replace() {
  "${host_python}" -I -S - "$1" "$2" <<'PY'
import os
import stat
import sys
from pathlib import Path


def inode_identity(path):
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
        return None
    return (metadata.st_dev, metadata.st_ino)


def unlink_if_same_inode(path, expected):
    if inode_identity(path) != expected:
        return False
    path.unlink()
    return True


def fsync_parent(path):
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


staged = Path(sys.argv[1]).absolute()
target = Path(sys.argv[2]).absolute()
parent = target.parent
staged_metadata = staged.lstat()
parent_metadata = parent.lstat()
if (
    not stat.S_ISREG(staged_metadata.st_mode)
    or staged.is_symlink()
    or staged.resolve(strict=True) != staged
    or staged.parent != parent
    or not stat.S_ISDIR(parent_metadata.st_mode)
    or parent.is_symlink()
    or parent.resolve(strict=True) != parent
):
    raise SystemExit("atomic file publication paths are unsafe")
try:
    target.lstat()
except FileNotFoundError:
    pass
else:
    raise FileExistsError(f"refusing to replace occupied publication target: {target}")

staged_identity = None
link_created = False
descriptor = os.open(
    staged,
    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
)
try:
    opened_metadata = os.fstat(descriptor)
    staged_identity = (opened_metadata.st_dev, opened_metadata.st_ino)
    if (
        not stat.S_ISREG(opened_metadata.st_mode)
        or staged_identity != (staged_metadata.st_dev, staged_metadata.st_ino)
    ):
        raise RuntimeError("staged file changed before publication")
    os.fsync(descriptor)
finally:
    os.close(descriptor)
try:
    # Hard-link creation is the atomic no-replace operation.  EEXIST is a
    # hard failure, unlike legacy no-clobber moves that can report success.
    os.link(staged, target, follow_symlinks=False)
    link_created = True
    target_metadata = target.lstat()
    if (
        not stat.S_ISREG(target_metadata.st_mode)
        or target.is_symlink()
        or (target_metadata.st_dev, target_metadata.st_ino)
        != staged_identity
    ):
        raise RuntimeError("published file is not the staged inode")
    descriptor = os.open(
        target,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if not unlink_if_same_inode(staged, staged_identity):
        raise RuntimeError("staged file changed before cleanup")
    fsync_parent(parent)
except BaseException:
    if link_created and staged_identity is not None:
        try:
            removed_target = unlink_if_same_inode(target, staged_identity)
        except OSError:
            removed_target = False
        if removed_target:
            try:
                fsync_parent(parent)
            except OSError:
                pass
    if staged_identity is not None:
        try:
            unlink_if_same_inode(staged, staged_identity)
        except OSError:
            pass
    raise
finally:
    try:
        if staged_identity is not None:
            unlink_if_same_inode(staged, staged_identity)
    except OSError:
        pass
PY
}
atomic_copy() {
  local source="$1" name="$2" target="${project_run}/${name}" temporary
  if [[ ! -f "${source}" || -L "${source}" || -e "${target}" || -L "${target}" ]]; then
    echo "error: refusing unsafe or occupied evidence publication: ${target}" >&2
    return 1
  fi
  temporary="$(mktemp "${project_run}/.${name}.XXXXXX")" || return 1
  if ! cp -p -- "${source}" "${temporary}"; then
    rm -f -- "${temporary}"
    return 1
  fi
  if [[ "$(sha256 "${source}")" != "$(sha256 "${temporary}")" ]]; then
    echo "error: staged evidence differs from source: ${source}" >&2
    rm -f -- "${temporary}"
    return 1
  fi
  install_file_no_replace "${temporary}" "${target}"
}
atomic_directory_publish_no_replace() {
  "${host_python}" -I -S - "$1" "$2" "$3" <<'PY'
import ctypes
import errno
import hashlib
import os
import stat
import sys
from pathlib import Path

source = Path(sys.argv[1]).absolute()
staged = Path(sys.argv[2]).absolute()
target = Path(sys.argv[3]).absolute()


def canonical_directory(path: Path, *, name: str) -> Path:
    metadata = path.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or path.is_symlink()
        or path.resolve(strict=True) != path
    ):
        raise ValueError(f"{name} must be a canonical real directory")
    return path


def manifest(root: Path, *, fsync: bool) -> tuple[tuple[str, str, int], ...]:
    records: list[tuple[str, str, int]] = []
    directories: list[Path] = []
    for current_raw, directory_names, filenames in os.walk(root, followlinks=False):
        current = Path(current_raw)
        canonical_directory(current, name="artifact tree directory")
        directories.append(current)
        relative_directory = current.relative_to(root).as_posix()
        records.append(
            (
                "." if relative_directory == "." else f"{relative_directory}/",
                "directory",
                0,
            )
        )
        for name in sorted(directory_names):
            child = current / name
            metadata = child.lstat()
            if not stat.S_ISDIR(metadata.st_mode) or child.is_symlink():
                raise ValueError("artifact tree contains an unsafe directory entry")
        for name in sorted(filenames):
            child = current / name
            metadata = child.lstat()
            if not stat.S_ISREG(metadata.st_mode) or child.is_symlink():
                raise ValueError("artifact tree contains a non-regular file")
            digest = hashlib.sha256()
            descriptor = os.open(
                child,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                with os.fdopen(descriptor, "rb", closefd=False) as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                if fsync:
                    os.fsync(descriptor)
            finally:
                os.close(descriptor)
            records.append(
                (
                    child.relative_to(root).as_posix(),
                    digest.hexdigest(),
                    metadata.st_size,
                )
            )
    if fsync:
        for directory in reversed(directories):
            descriptor = os.open(
                directory,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    return tuple(sorted(records))


canonical_directory(source, name="source artifact")
canonical_directory(staged, name="staged artifact")
parent = canonical_directory(target.parent, name="artifact publication parent")
try:
    target.lstat()
except FileNotFoundError:
    pass
else:
    raise FileExistsError(f"refusing to replace occupied artifact target: {target}")
source_manifest = manifest(source, fsync=False)
staged_manifest = manifest(staged, fsync=True)
if source_manifest != staged_manifest:
    raise ValueError("staged artifact tree differs from the verified source artifact")

libc = ctypes.CDLL(None, use_errno=True)
renameat2 = getattr(libc, "renameat2", None)
if renameat2 is None:
    raise RuntimeError("renameat2 is required for atomic no-replace artifact publication")
renameat2.argtypes = [
    ctypes.c_int,
    ctypes.c_char_p,
    ctypes.c_int,
    ctypes.c_char_p,
    ctypes.c_uint,
]
renameat2.restype = ctypes.c_int
at_fdcwd = -100
rename_noreplace = 1
if (
    renameat2(
        at_fdcwd,
        os.fsencode(staged),
        at_fdcwd,
        os.fsencode(target),
        rename_noreplace,
    )
    != 0
):
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(f"artifact target became occupied: {target}")
    raise OSError(error_number, os.strerror(error_number), os.fspath(target))
canonical_directory(target, name="published artifact")
if staged.exists() or staged.is_symlink():
    raise RuntimeError("atomic artifact rename left the staging path occupied")
descriptor = os.open(
    parent,
    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
)
try:
    os.fsync(descriptor)
finally:
    os.close(descriptor)
PY
}
hash_or_none() { [[ -f "$1" && ! -L "$1" ]] && sha256 "$1" || printf 'none\n'; }
write_marker() {
  local status="$1" workload="$2" final="$3" temporary
  if [[ -e "${project_run}/SUCCESS" || -L "${project_run}/SUCCESS" || -e "${project_run}/FAILED" || -L "${project_run}/FAILED" ]]; then
    echo "error: refusing a second or conflicting terminal marker" >&2
    return 1
  fi
  temporary="$(mktemp "${project_run}/.${status}.XXXXXX")" || return 1
  if ! {
    printf 'schema_version=prorm-phase2-budgeted-end-to-end-fixed-three-run-status/v1\nstatus=%s\nformal=false\nevidence_role=%s\nstage=%s\nseed=%s\n' "${status}" "${BUDGETED_EVIDENCE_ROLE}" "${BUDGETED_STAGE}" "${seed}"
    printf 'slurm_job_id=%s\narray_job_id=%s\narray_task_id=%s\ncluster=hpc4\naccount=sigroup\npartition=gpu-l20\nrestart_count=0\n' "${SLURM_JOB_ID}" "${SLURM_ARRAY_JOB_ID}" "${SLURM_ARRAY_TASK_ID}"
    printf 'phase2_design_sha256=%s\nbase_config_hash=%s\ngit_commit=%s\nsubmission_intent_sha256=%s\nsubmission_ledger_sha256=%s\n' "${PRORM_BUDGETED_DESIGN_SHA256}" "${PRORM_BUDGETED_BASE_CONFIG_HASH}" "${PRORM_GIT_COMMIT}" "${submission_intent_sha256}" "${submission_ledger_sha256}"
    printf 'freeze_evidence_sha256=%s\nfrozen_global_beta=%s\noptimizer_schedule_sha256=%s\nartifact_metadata_sha256=%s\nphase2_result_sha256=%s\nrollouts_sha256=%s\nverification_sha256=%s\nmanifest_sha256=%s\n' "${PRORM_BUDGETED_FREEZE_EVIDENCE_SHA256}" "${PRORM_BUDGETED_FROZEN_BETA}" "${PRORM_OPTIMIZER_SCHEDULE_SHA256}" "$(hash_or_none "${project_run}/artifact/metadata.json")" "$(hash_or_none "${project_run}/phase2-result.json")" "$(hash_or_none "${project_run}/phase2-result.rollouts.jsonl")" "$(hash_or_none "${project_run}/phase2-budgeted-output-verification.json")" "$(hash_or_none "${project_run}/run-manifest.json")"
    printf 'workload_exit_code=%s\nfinal_exit_code=%s\ncreated_at_utc=%s\n' "${workload}" "${final}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } >"${temporary}"; then
    rm -f -- "${temporary}"
    return 1
  fi
  install_file_no_replace "${temporary}" "${project_run}/${status}"
}
verify_published_success_closure() {
  "${host_python}" -I -S - \
    "${project_run}" "${project_artifact}" "${verification_complete}" \
    "${seed}" "${PRORM_BUDGETED_DESIGN_SHA256}" \
    "${PRORM_BUDGETED_BASE_CONFIG_HASH}" \
    "${PRORM_BUDGETED_FREEZE_EVIDENCE_SHA256}" \
    "${PRORM_BUDGETED_FROZEN_BETA}" "${PRORM_GIT_COMMIT}" \
    "${PRORM_IMAGE_SHA256}" "${PRORM_HF_INVENTORY_SHA256}" \
    "${SLURM_JOB_ID}" "${SLURM_ARRAY_JOB_ID}" "${SLURM_ARRAY_TASK_ID}" \
    "${PRORM_BUDGETED_OVERLAY_SHA256}" "${artifact_metadata_sha256}" <<'PY'
import hashlib
import json
import math
import os
import stat
import sys
from collections.abc import Sequence
from pathlib import Path

(
    run_raw,
    artifact_raw,
    verification_complete,
    seed_raw,
    design_sha256,
    base_config_hash,
    freeze_sha256,
    frozen_beta_raw,
    git_commit,
    image_sha256,
    inventory_sha256,
    slurm_job_id,
    array_job_id,
    array_task_id_raw,
    expected_overlay_sha256,
    expected_artifact_metadata_sha256,
) = sys.argv[1:]
if verification_complete != "1":
    raise SystemExit("in-job output verification did not complete")
seed = int(seed_raw)
array_task_id = int(array_task_id_raw)
frozen_beta = float(frozen_beta_raw)
if not math.isfinite(frozen_beta) or frozen_beta <= 0.0:
    raise SystemExit("published frozen beta is invalid")


def canonical_directory(path: Path, *, name: str) -> Path:
    absolute = path.absolute()
    metadata = absolute.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or absolute.is_symlink()
        or absolute.resolve(strict=True) != absolute
    ):
        raise ValueError(f"{name} must be a canonical real directory")
    return absolute


def canonical_file(path: Path, *, name: str) -> Path:
    absolute = path.absolute()
    metadata = absolute.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or absolute.is_symlink()
        or absolute.resolve(strict=True) != absolute
    ):
        raise ValueError(f"{name} must be a canonical regular file")
    return absolute


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_canonical_json(path: Path) -> dict[str, object]:
    raw = path.read_bytes()

    def reject_duplicates(
        pairs: Sequence[tuple[str, object]],
    ) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"published verification contains duplicate key {key!r}")
            value[key] = item
        return value

    value = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=reject_duplicates,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"published verification contains non-finite constant {token}")
        ),
    )
    if not isinstance(value, dict):
        raise TypeError("published verification must be a JSON object")
    canonical = (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    if raw != canonical:
        raise ValueError("published verification is not canonical deterministic JSON")
    return value


run = canonical_directory(Path(run_raw), name="published run")
artifact = canonical_directory(Path(artifact_raw), name="published artifact")
artifact_link = run / "artifact"
link_metadata = artifact_link.lstat()
if (
    not stat.S_ISLNK(link_metadata.st_mode)
    or artifact_link.resolve(strict=True) != artifact
):
    raise ValueError("published run artifact link differs from the atomic artifact target")
verification_path = canonical_file(
    run / "phase2-budgeted-output-verification.json",
    name="published verification",
)
result_path = canonical_file(run / "phase2-result.json", name="published result")
rollouts_path = canonical_file(
    run / "phase2-result.rollouts.jsonl",
    name="published rollouts",
)
manifest_path = canonical_file(run / "run-manifest.json", name="published manifest")
materialization_path = canonical_file(
    run / "artifact-materialization.json",
    name="published artifact materialization",
)
metadata_path = canonical_file(
    artifact / "metadata.json",
    name="published artifact metadata",
)
linked_metadata = (artifact_link / "metadata.json").resolve(strict=True)
if linked_metadata != metadata_path:
    raise ValueError("published artifact metadata link resolves to a different file")

verification = strict_canonical_json(verification_path)
relative_files = verification.get("relative_files")
input_sha256 = verification.get("input_sha256")
slurm = verification.get("slurm")
if (
    not isinstance(relative_files, dict)
    or not isinstance(input_sha256, dict)
    or not isinstance(slurm, dict)
):
    raise ValueError("published verification hash/reference maps are invalid")
expected_relative_files = {
    "result": "phase2-result.json",
    "rollouts": "phase2-result.rollouts.jsonl",
    "run_manifest": "run-manifest.json",
    "artifact_metadata": "artifact/metadata.json",
    "artifact_materialization": "artifact-materialization.json",
}
observed_hashes = {
    "result": sha256(result_path),
    "rollouts": sha256(rollouts_path),
    "run_manifest": sha256(manifest_path),
    "artifact_metadata": sha256(metadata_path),
    "artifact_materialization": sha256(materialization_path),
}
if (
    verification.get("schema_version")
    != "prorm-phase2-budgeted-fixed-three-seed-output-verification/v1"
    or verification.get("status") != "verified"
    or verification.get("design_stage") != "budgeted_end_to_end"
    or verification.get("evidence_role")
    != "budgeted_end_to_end_fixed_three_exploratory_only"
    or verification.get("formal_eligibility") is not False
    or verification.get("formal_claim_eligible") is not False
    or verification.get("supports_formal_claim") is not False
    or verification.get("inferential_or_significance_claim_produced") is not False
    or verification.get("seed") != seed
    or verification.get("phase2_design_sha256") != design_sha256
    or verification.get("base_config_hash") != base_config_hash
    or verification.get("accepted_freeze_aggregate_sha256") != freeze_sha256
    or float(verification.get("frozen_global_beta")) != frozen_beta
    or verification.get("git_commit") != git_commit
    or verification.get("image_sha256") != image_sha256
    or verification.get("hf_inventory_sha256") != inventory_sha256
    or verification.get("slurm_job_id_raw") != slurm_job_id
    or verification.get("array_job_id") != array_job_id
    or verification.get("array_task_id") != array_task_id
    or slurm
    != {
        "job_id_raw": slurm_job_id,
        "array_job_id": array_job_id,
        "array_task_id": array_task_id,
        "account": "sigroup",
        "cluster": "hpc4",
        "partition": "gpu-l20",
    }
    or relative_files != expected_relative_files
    or verification.get("result_sha256") != observed_hashes["result"]
    or verification.get("rollouts_sha256") != observed_hashes["rollouts"]
    or verification.get("run_manifest_sha256") != observed_hashes["run_manifest"]
    or verification.get("artifact_metadata_sha256")
    != observed_hashes["artifact_metadata"]
    or verification.get("artifact_materialization_sha256")
    != observed_hashes["artifact_materialization"]
    or observed_hashes["artifact_metadata"] != expected_artifact_metadata_sha256
    or set(input_sha256)
    != {
        "overlay",
        "result",
        "rollouts",
        "run_manifest",
        "artifact_metadata",
        "artifact_materialization",
    }
    or input_sha256.get("overlay") != expected_overlay_sha256
    or input_sha256.get("result") != observed_hashes["result"]
    or input_sha256.get("rollouts") != observed_hashes["rollouts"]
    or input_sha256.get("run_manifest") != observed_hashes["run_manifest"]
    or input_sha256.get("artifact_metadata") != observed_hashes["artifact_metadata"]
    or input_sha256.get("artifact_materialization")
    != observed_hashes["artifact_materialization"]
):
    raise ValueError(
        "published result/rollouts/manifest/artifact metadata do not match verification"
    )
PY
}
finalize() {
  local workload=$? final marker file sync=0
  trap - EXIT; set +e
  for file in authorization-check-container.json accepted-freeze-preflight.json phase2-config-check.json base-config-check.json budgeted-overlay-binding-check.json hf-inventory-verification.json gpu-check.json run-manifest.json materialize.log memory-materialize.json artifact-materialization.json artifact-verification.json phase2-run.log memory-phase2.json phase2-result.json phase2-result.rollouts.jsonl phase2-budgeted-output-verification.json phase2-failure-evidence.json; do
    if [[ -e "${job_dir}/${file}" || -L "${job_dir}/${file}" ]]; then
      atomic_copy "${job_dir}/${file}" "${file}" || sync=1
    fi
  done
  if [[ "${workload}" = 0 ]]; then
    for file in authorization-check-container.json accepted-freeze-preflight.json budgeted-overlay-binding-check.json run-manifest.json artifact-materialization.json phase2-result.json phase2-result.rollouts.jsonl phase2-budgeted-output-verification.json; do [[ -f "${project_run}/${file}" && ! -L "${project_run}/${file}" ]] || sync=1; done
    [[ -L "${project_run}/artifact" && -d "${project_run}/artifact" ]] || sync=1
    [[ "${verification_complete}" = 1 ]] || sync=1
    if (( sync == 0 )); then
      verify_published_success_closure || sync=1
    fi
  fi
  final="${workload}"; (( sync == 0 )) || [[ "${final}" != 0 ]] || final=1
  [[ "${final}" = 0 ]] && marker=SUCCESS || marker=FAILED
  if ! write_marker "${marker}" "${workload}" "${final}"; then
    final=1
    if [[ "${marker}" = SUCCESS && ! -e "${project_run}/SUCCESS" && ! -L "${project_run}/SUCCESS" && ! -e "${project_run}/FAILED" && ! -L "${project_run}/FAILED" ]]; then
      write_marker FAILED "${workload}" 1 || true
    fi
  fi
  exit "${final}"
}
trap finalize EXIT

execution_repo="${job_dir}/source"
git clone --quiet --no-hardlinks --no-checkout "${PRORM_REPO_ROOT}" "${execution_repo}"
git -C "${execution_repo}" checkout --quiet --detach "${PRORM_GIT_COMMIT}"
execution_overlay="${execution_repo}/${PRORM_BUDGETED_OVERLAY_REL}"; execution_base="${execution_repo}/${PRORM_BUDGETED_BASE_REL}"
for pair in "${execution_overlay}:${PRORM_BUDGETED_OVERLAY_SHA256}" "${execution_base}:${PRORM_BUDGETED_BASE_SHA256}"; do path="${pair%%:*}"; digest="${pair#*:}"; [[ "$(sha256 "${path}")" = "${digest}" ]] || die "detached config bytes differ"; done
[[ "$(git -C "${execution_repo}" rev-parse --verify HEAD)" = "${PRORM_GIT_COMMIT}" && -z "$(git -C "${execution_repo}" status --porcelain --untracked-files=normal)" ]] || die "detached checkout changed"

datasets_cache="${job_dir}/hf-datasets-cache"; tmp_dir="${job_dir}/tmp"; xdg_cache="${job_dir}/xdg-cache"; mkdir -m 700 -- "${datasets_cache}" "${tmp_dir}" "${xdg_cache}"
for evidence_root in \
  "${PRORM_PROJECT_ROOT}/aggregates" \
  "${PRORM_PROJECT_ROOT}/runs/phase2-post-recovery-calibration" \
  "${PRORM_PROJECT_ROOT}/artifacts/phase2-post-recovery-calibration" \
  "${PRORM_PROJECT_ROOT}/runs/phase2-post-recovery-freeze" \
  "${PRORM_PROJECT_ROOT}/artifacts/phase2-post-recovery-freeze"; do
  [[ -d "${evidence_root}" && ! -L "${evidence_root}" ]] || die "accepted-freeze evidence root is missing or unsafe: ${evidence_root}"
done
mapfile -t r3_bind_paths < <(
  "${host_python}" -I -S "${execution_repo}/${R3_BIND_PLAN_RELATIVE}" \
    "${PRORM_RECOVERY_AUTHORIZATION}" \
    --expected-sha256 "${PRORM_RECOVERY_AUTHORIZATION_SHA256}" \
    --project-root "${PRORM_PROJECT_ROOT}" \
    --emit paths
) || die "minimal R3 authorization bind-plan construction failed"
[[ "${#r3_bind_paths[@]}" -ge 10 ]] \
  || die "minimal R3 authorization bind plan is incomplete"
r3_bind_arguments=()
for path in "${r3_bind_paths[@]}"; do
  [[ "${path}" == "${PRORM_PROJECT_ROOT}/"* \
    && "${path}" != *","* && "${path}" != *":"* \
    && "${path}" != *$'\n'* && "${path}" != *$'\r'* ]] \
    || die "unsafe R3 verification bind path"
  r3_bind_arguments+=(--bind "${path}:${path}:ro")
done
container=(
  apptainer exec --cleanenv --nv --no-mount home,cwd,bind-paths --pwd "${execution_repo}"
  --bind "${job_dir}:${job_dir},${PRORM_HF_CACHE}:${PRORM_HF_CACHE}:ro,${PRORM_BUDGETED_FREEZE_EVIDENCE}:${PRORM_BUDGETED_FREEZE_EVIDENCE}:ro"
  "${r3_bind_arguments[@]}"
  --bind "${PRORM_PROJECT_ROOT}/aggregates:${PRORM_PROJECT_ROOT}/aggregates:ro"
  --bind "${PRORM_PROJECT_ROOT}/runs/phase2-post-recovery-calibration:${PRORM_PROJECT_ROOT}/runs/phase2-post-recovery-calibration:ro"
  --bind "${PRORM_PROJECT_ROOT}/artifacts/phase2-post-recovery-calibration:${PRORM_PROJECT_ROOT}/artifacts/phase2-post-recovery-calibration:ro"
  --bind "${PRORM_PROJECT_ROOT}/runs/phase2-post-recovery-freeze:${PRORM_PROJECT_ROOT}/runs/phase2-post-recovery-freeze:ro"
  --bind "${PRORM_PROJECT_ROOT}/artifacts/phase2-post-recovery-freeze:${PRORM_PROJECT_ROOT}/artifacts/phase2-post-recovery-freeze:ro"
  --env "PYTHONPATH=${execution_repo}/src" --env "PYTHONNOUSERSITE=1" --env "PYTHONDONTWRITEBYTECODE=1"
  --env "HF_HOME=${PRORM_HF_CACHE}" --env "HF_HUB_CACHE=${PRORM_HF_CACHE}/hub" --env "HF_DATASETS_CACHE=${datasets_cache}" --env "TMPDIR=${tmp_dir}" --env "XDG_CACHE_HOME=${xdg_cache}"
  --env "HF_HUB_OFFLINE=1" --env "TRANSFORMERS_OFFLINE=1" --env "HF_DATASETS_OFFLINE=1" --env "HF_HUB_DISABLE_TELEMETRY=1" --env "TOKENIZERS_PARALLELISM=false"
  --env "PRORM_IMAGE=${PRORM_IMAGE}" --env "PRORM_IMAGE_SHA256=${PRORM_IMAGE_SHA256}" --env "PRORM_GIT_COMMIT=${PRORM_GIT_COMMIT}"
  --env "PRORM_HF_INVENTORY=${PRORM_HF_INVENTORY}" --env "PRORM_HF_INVENTORY_SHA256=${PRORM_HF_INVENTORY_SHA256}"
  --env "PRORM_BUDGETED_DESIGN_SHA256=${PRORM_BUDGETED_DESIGN_SHA256}" --env "PRORM_BUDGETED_BASE_CONFIG_HASH=${PRORM_BUDGETED_BASE_CONFIG_HASH}"
  --env "PRORM_BUDGETED_FROZEN_BETA=${PRORM_BUDGETED_FROZEN_BETA}" --env "PRORM_BUDGETED_FREEZE_EVIDENCE_SHA256=${PRORM_BUDGETED_FREEZE_EVIDENCE_SHA256}"
)
for variable in \
  SLURM_JOB_ID SLURM_ARRAY_JOB_ID SLURM_ARRAY_TASK_ID SLURM_ARRAY_TASK_COUNT \
  SLURM_ARRAY_TASK_MIN SLURM_ARRAY_TASK_MAX SLURM_ARRAY_TASK_STEP SLURM_JOB_NAME \
  SLURM_JOB_ACCOUNT SLURM_CLUSTER_NAME SLURM_JOB_PARTITION SLURM_RESTART_COUNT \
  SLURM_JOB_NODELIST SLURM_NNODES SLURM_NTASKS SLURM_CPUS_PER_TASK SLURM_GPUS \
  SLURM_GPUS_ON_NODE SLURM_PROCID SLURM_LOCALID SLURM_NODEID CUDA_VISIBLE_DEVICES \
  NVIDIA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL; do
  [[ -v "${variable}" ]] && container+=(--env "${variable}=${!variable}")
done
container+=("${PRORM_IMAGE}")
"${container[@]}" python \
  "${execution_repo}/scripts/hpc4/verify_phase2_budgeted_r3_authorization.py" \
  "${PRORM_RECOVERY_AUTHORIZATION}" "${execution_overlay}" \
  --expected-sha256 "${PRORM_RECOVERY_AUTHORIZATION_SHA256}" \
  --project-root "${PRORM_PROJECT_ROOT}" \
  >"${job_dir}/authorization-check-container.json"
"${container[@]}" python -m smart_reward.cli phase2-config-check "${execution_overlay}" | tee "${job_dir}/phase2-config-check.json"
"${container[@]}" python -m smart_reward.cli config-check "${execution_base}" | tee "${job_dir}/base-config-check.json"
"${host_python}" -I -S - "${job_dir}/phase2-config-check.json" "${job_dir}/base-config-check.json" "${PRORM_BUDGETED_DESIGN_SHA256}" "${PRORM_BUDGETED_BASE_CONFIG_HASH}" <<'PY'
import json, sys
from pathlib import Path
phase2=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")); base=json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
if (
    phase2.get("status") != "ok" or phase2.get("design_stage") != "budgeted_end_to_end"
    or phase2.get("formal_eligibility") is not False or phase2.get("phase2_design_sha256") != sys.argv[3]
    or phase2.get("source_config_hash") != sys.argv[4] or base.get("status") != "ok"
    or base.get("config_hash") != sys.argv[4]
): raise SystemExit("container-computed budgeted config identities are invalid")
PY
"${container[@]}" python - "${execution_overlay}" "${PRORM_BUDGETED_BASE_CONFIG_HASH}" "${PRORM_BUDGETED_FREEZE_EVIDENCE_SHA256}" "${PRORM_BUDGETED_FROZEN_BETA}" <<'PY' >"${job_dir}/budgeted-overlay-binding-check.json"
import math, sys
from smart_reward.phase2_config import load_phase2_config_bundle
config = load_phase2_config_bundle(sys.argv[1]).config
common = config["objective"]["common_beta"]
horizon = config["evaluation"]["max_length"]
if (
    config["design"]["stage"] != "budgeted_end_to_end"
    or config["design"].get("formal_eligibility") is not False
    or config["design"].get("evidence_role")
    != "budgeted_end_to_end_fixed_three_exploratory_only"
    or config["design"].get("source_config_hash") != sys.argv[2]
    or tuple(config["run"].get("seeds", ())) != (20261001, 20261002, 20261003)
    or config["run"].get("confirmatory") is not False
    or common.get("beta_source_aggregate_sha256") != sys.argv[3]
    or horizon.get("parent_pilot_aggregate_sha256") != sys.argv[3]
    or not math.isclose(float(common["frozen_global_beta"]), float(sys.argv[4]), rel_tol=0.0, abs_tol=0.0)
):
    raise SystemExit("budgeted overlay does not bind the accepted freeze, beta, and source identity")
print('{"status":"ok","formal":false,"evidence_role":"budgeted_end_to_end_fixed_three_exploratory_only"}')
PY
"${container[@]}" python - "${execution_overlay}" "${PRORM_BUDGETED_FREEZE_EVIDENCE}" "${PRORM_BUDGETED_FREEZE_EVIDENCE_SHA256}" "${PRORM_BUDGETED_FROZEN_BETA}" <<'PY' >"${job_dir}/accepted-freeze-preflight.json"
import json, math, sys
from smart_reward.phase2_config import load_phase2_config_bundle
from smart_reward.phase2_pilot_aggregate import verify_beta_source_aggregate, verify_horizon_parent_aggregate
overlay, path, expected_sha, expected_beta = sys.argv[1:]
config = load_phase2_config_bundle(overlay).config
beta = verify_beta_source_aggregate(config, path)
horizon = verify_horizon_parent_aggregate(config, path)
if (
    beta is None or horizon is None or beta.get("sha256") != expected_sha
    or horizon.get("sha256") != expected_sha or horizon.get("source_pilot_phase") != "freeze"
    or not math.isclose(float(beta.get("accepted_beta")), float(expected_beta), rel_tol=0.0, abs_tol=0.0)
): raise SystemExit("accepted freeze cannot jointly authorize beta and horizon")
print(json.dumps({"schema_version":"prorm-phase2-budgeted-accepted-freeze-preflight/v1","status":"passed","same_freeze_for_beta_and_horizon":True,"freeze_sha256":expected_sha,"frozen_beta":float(expected_beta),"formal":False},allow_nan=False,sort_keys=True,separators=(",",":")))
PY
"${container[@]}" python "${execution_repo}/scripts/hpc4/stage_hf_assets.py" "${execution_base}" "${PRORM_HF_CACHE}" --verify-only --inventory "${PRORM_HF_INVENTORY}" --verification-datasets-cache "${datasets_cache}" | tee "${job_dir}/hf-inventory-verification.json"
"${container[@]}" python - <<'PY' | tee "${job_dir}/gpu-check.json"
import json, torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or torch.cuda.get_device_name(0) != "NVIDIA L20": raise SystemExit("exactly one NVIDIA L20 is required")
print(json.dumps({"status":"ok","gpu_model":"NVIDIA L20"}, sort_keys=True))
PY
"${container[@]}" python -m smart_reward.cli env-report "${execution_base}" --seed "${seed}" --repo-root "${execution_repo}" --output "${job_dir}/run-manifest.json"

artifact_dir="${job_dir}/artifact"
"${container[@]}" env "PRORM_MEMORY_REPORT=${job_dir}/memory-materialize.json" python -m smart_reward.cli controlled-materialize "${execution_base}" "${artifact_dir}" --seed "${seed}" --device cuda 2>&1 | tee "${job_dir}/materialize.log"
artifact_metadata_sha256="$(sha256 "${artifact_dir}/metadata.json")"
"${host_python}" -I -S - "${job_dir}/artifact-materialization.json" "${seed}" "${PRORM_BUDGETED_DESIGN_SHA256}" "${PRORM_BUDGETED_BASE_CONFIG_HASH}" "${artifact_metadata_sha256}" <<'PY'
import json, sys
from pathlib import Path
output, seed, design, base, metadata = sys.argv[1:]
value={"schema_version":"prorm-phase2-budgeted-artifact-binding/v1","mode":"fresh","seed":int(seed),"phase2_design_sha256":design,"base_config_hash":base,"artifact_metadata_sha256":metadata,"recovery_artifact_reused":False,"recovery_reward_heads_reused":False,"recovery_optimizer_state_reused":False}
with Path(output).open("x",encoding="utf-8",newline="\n") as stream:
    json.dump(value,stream,allow_nan=False,sort_keys=True,separators=(",",":")); stream.write("\n")
PY
"${container[@]}" python - "${execution_overlay}" "${seed}" "${artifact_dir}" "${job_dir}/run-manifest.json" <<'PY' | tee "${job_dir}/artifact-verification.json"
import json, sys
from smart_reward.phase2_inputs import prepare_phase2_inputs
value=prepare_phase2_inputs(sys.argv[1], seed=int(sys.argv[2]), artifact_dir=sys.argv[3], run_manifest=sys.argv[4])
print(json.dumps({"status":"ok","seed":value.seed,"phase2_design_sha256":value.phase2_config_hash,"base_config_hash":value.source_config_hash},sort_keys=True))
PY

# One accepted freeze is deliberately passed in both roles; no trained recovery
# artifact, optimizer state, or head directory is mounted or reused.
"${container[@]}" env "PRORM_MEMORY_REPORT=${job_dir}/memory-phase2.json" "PRORM_FAILURE_EVIDENCE=${job_dir}/phase2-failure-evidence.json" python -m smart_reward.cli phase2-run "${execution_overlay}" "${artifact_dir}" "${job_dir}/run-manifest.json" "${job_dir}/phase2-result.json" --seed "${seed}" --device cuda --beta-source-aggregate "${PRORM_BUDGETED_FREEZE_EVIDENCE}" --horizon-parent-aggregate "${PRORM_BUDGETED_FREEZE_EVIDENCE}" 2>&1 | tee "${job_dir}/phase2-run.log"

# The pre-oracle safety/output verifier is a hard gate before publication and
# before SUCCESS can be written.  It is supplied as a separate, budgeted-only
# verifier; it is not a formal finalizer.
# The verifier interface is populated by the dedicated verifier implementation;
# this sidecar name is fixed by phase2-run and must be present before invoking it.
[[ -f "${job_dir}/phase2-result.rollouts.jsonl" && ! -L "${job_dir}/phase2-result.rollouts.jsonl" ]] || die "phase2-run did not produce the required rollout sidecar"
"${container[@]}" python "${execution_repo}/scripts/hpc4/verify_phase2_budgeted_end_to_end_seed_output.py" "${execution_overlay}" "${job_dir}/phase2-result.json" "${job_dir}/phase2-result.rollouts.jsonl" "${job_dir}/phase2-budgeted-output-verification.json" --seed "${seed}" --design-sha256 "${PRORM_BUDGETED_DESIGN_SHA256}" --base-config-hash "${PRORM_BUDGETED_BASE_CONFIG_HASH}" --git-commit "${PRORM_GIT_COMMIT}" --image-sha256 "${PRORM_IMAGE_SHA256}" --hf-inventory-sha256 "${PRORM_HF_INVENTORY_SHA256}" --artifact-metadata-sha256 "${artifact_metadata_sha256}" --freeze-evidence-sha256 "${PRORM_BUDGETED_FREEZE_EVIDENCE_SHA256}" --slurm-job-id-raw "${SLURM_JOB_ID}" --array-job-id "${SLURM_ARRAY_JOB_ID}" --array-task-id "${SLURM_ARRAY_TASK_ID}"
verification_complete=1

mkdir -p -- "$(dirname -- "${project_artifact}")"
artifact_staging="$(mktemp -d "$(dirname -- "${project_artifact}")/.${execution_id}.staged.XXXXXX")"
cp -a -- "${artifact_dir}/." "${artifact_staging}/"
atomic_directory_publish_no_replace \
  "${artifact_dir}" "${artifact_staging}" "${project_artifact}"
ln -s -- "$(realpath --relative-to="${project_run}" "${project_artifact}")" "${project_run}/artifact"
fsync_directory "${project_run}"
printf 'budgeted end-to-end complete: seed=%s run=%s\n' "${seed}" "${project_run}"
