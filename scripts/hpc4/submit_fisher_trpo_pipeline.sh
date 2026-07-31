#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 5 ]] || {
  echo "usage: $0 CONFIG IMAGE HF_CACHE RUN_ROOT SOURCE_RUN_ROOT" >&2
  exit 2
}

repo_root="$(git rev-parse --show-toplevel)"
config="$(realpath -e "$1")"
image="$(realpath -e "$2")"
hf_cache="$(realpath -e "$3")"
run_root="$(realpath -m "$4")"
source_run_root="$(realpath -e "$5")"

protocol="$(PYTHONPATH="${repo_root}/src" python3 - "${config}" <<'PY'
import sys
from smart_reward.config import load_config
print(load_config(sys.argv[1])["protocol"])
PY
)"
[[ "${protocol}" = "prorm_fisher_trpo_v1" ]] || {
  echo "the full Fisher-TRPO DAG requires prorm_fisher_trpo_v1" >&2
  exit 2
}
case "${source_run_root}" in /project/sigroup/*) ;;
  *) echo "source run root must be an immutable project archive" >&2; exit 2 ;;
esac
[[ ! -w "${source_run_root}" ]] || {
  echo "source run root must not be writable" >&2
  exit 2
}
[[ -z "$(git -C "${repo_root}" status --porcelain)" ]] || {
  echo "formal DAG submission requires a clean worktree" >&2
  exit 2
}
mkdir -p "${run_root}"
manifest="${run_root}/submission-dag.tsv"
[[ ! -e "${manifest}" ]] || {
  echo "refusing to overwrite an existing submission DAG: ${manifest}" >&2
  exit 2
}
mapfile -t source_seeds < <(PYTHONPATH="${repo_root}/src" python3 - "${config}" <<'PY'
import sys
from smart_reward.config import load_config
for seed in load_config(sys.argv[1])["run"]["seeds"]:
    print(seed)
PY
)
for seed in "${source_seeds[@]}"; do
  [[ -f "${source_run_root}/seed-${seed}/artifact/metadata.json" ]] || {
    echo "source artifact is missing for seed ${seed}" >&2
    exit 2
  }
  [[ -f "${source_run_root}/seed-${seed}/reward_result.json" ]] || {
    echo "source reward result is missing for seed ${seed}" >&2
    exit 2
  }
done

submit_stage() {
  local stage="$1"
  local dependency="${2:-}"
  local output job_id
  output="$(
    PRORM_SOURCE_RUN_ROOT="${source_run_root}" \
    PRORM_SBATCH_DEPENDENCY="${dependency}" \
      "${repo_root}/scripts/hpc4/submit_pipeline.sh" \
      "${config}" "${image}" "${hf_cache}" "${run_root}" "${stage}"
  )"
  job_id="$(awk -F= '$1 == "job_id" {print $2}' <<< "${output}")"
  job_id="${job_id%%;*}"
  [[ "${job_id}" =~ ^[0-9]+$ ]] || {
    echo "failed to parse job ID for ${stage}" >&2
    exit 2
  }
  printf '%s\t%s\t%s\n' "${stage}" "${job_id}" "${dependency}" >> "${manifest}"
  printf '%s' "${job_id}"
}

materialize_job="$(submit_stage materialize)"
crossfit_job="$(submit_stage fisher-crossfit "${materialize_job}")"
selection_job="$(submit_stage fisher-select "${crossfit_job}")"
reward_job="$(submit_stage reward "${selection_job}")"
adapter_job="$(submit_stage adapters "${reward_job}")"
calibration_job="$(submit_stage kl-calibration "${adapter_job}")"
calibration_aggregate_job="$(
  submit_stage kl-calibration-aggregate "${calibration_job}"
)"
rollout_job="$(submit_stage rollout "${calibration_aggregate_job}")"
rollout_aggregate_job="$(submit_stage rollout-aggregate "${rollout_job}")"
aggregate_job="$(submit_stage aggregate "${rollout_aggregate_job}")"
audit_job="$(submit_stage audit "${aggregate_job}")"

printf 'submission_manifest=%s\n' "${manifest}"
printf 'terminal_job_id=%s\n' "${audit_job}"
