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

mapfile -t config_values < <(PYTHONPATH="${repo_root}/src" python3 - "${config}" <<'PY'
import sys
from smart_reward.config import load_config
config = load_config(sys.argv[1])
print(config["protocol"])
for seed in config["run"]["seeds"]:
    print(seed)
PY
)
[[ "${config_values[0]}" = "prorm_fisher_trpo_v1" ]] || {
  echo "the Fisher-TRPO controller requires prorm_fisher_trpo_v1" >&2
  exit 2
}
seeds=("${config_values[@]:1}")
case "${source_run_root}" in /project/sigroup/*) ;;
  *) echo "source run root must be an immutable project archive" >&2; exit 2 ;;
esac
[[ ! -w "${source_run_root}" ]] || {
  echo "source run root must not be writable" >&2
  exit 2
}
[[ -z "$(git -C "${repo_root}" status --porcelain)" ]] || {
  echo "formal stage submission requires a clean worktree" >&2
  exit 2
}
for seed in "${seeds[@]}"; do
  [[ -f "${source_run_root}/seed-${seed}/artifact/metadata.json" ]] || {
    echo "source artifact is missing for seed ${seed}" >&2
    exit 2
  }
  [[ -f "${source_run_root}/seed-${seed}/reward_result.json" ]] || {
    echo "source reward result is missing for seed ${seed}" >&2
    exit 2
  }
done
mkdir -p "${run_root}"
manifest="${run_root}/submission-stages.tsv"

all_seeds_have() {
  local relative="$1"
  local seed
  for seed in "${seeds[@]}"; do
    [[ -f "${run_root}/seed-${seed}/${relative}" ]] || return 1
  done
}

if ! all_seeds_have "stage_receipts/materialize.json"; then
  next_stage="materialize"
elif ! all_seeds_have "stage_receipts/fisher-crossfit.json"; then
  next_stage="fisher-crossfit"
elif [[ ! -f "${run_root}/fisher_selection.json" ]]; then
  next_stage="fisher-select"
elif ! all_seeds_have "stage_receipts/reward.json"; then
  next_stage="reward"
elif ! all_seeds_have "stage_receipts/adapters.json"; then
  next_stage="adapters"
elif ! all_seeds_have "stage_receipts/kl-calibration.json"; then
  if find "${run_root}" -path '*/calibrated_adapters/.checkpoints/*.json' \
      -type f -print -quit | grep -q .; then
    # Re-running workers reuses every accepted component and fills only missing ones.
    next_stage="kl-calibration"
  else
    next_stage="kl-calibration"
  fi
elif ! all_seeds_have "policy_utility/receipt.json"; then
  if find "${run_root}" -path '*/policy_rollout_parts/*/receipt.json' \
      -type f -print -quit | grep -q .; then
    # Re-running workers resumes/reuses completed policy components.
    next_stage="rollout"
  else
    next_stage="rollout"
  fi
elif [[ ! -f "${run_root}/aggregate.json" ]]; then
  next_stage="aggregate"
elif [[ ! -f "${run_root}/integrity-audit.json" ]]; then
  next_stage="audit"
else
  printf 'status=all-compute-stages-complete\n'
  printf 'integrity_audit=%s\n' "${run_root}/integrity-audit.json"
  exit 0
fi

# Calibration and rollout workers write component receipts first.  Their
# aggregate receipts are separate CPU stages and are selected only after all
# component counts are complete.
if [[ "${next_stage}" = "kl-calibration" ]]; then
  expected=$(( ${#seeds[@]} * 9 ))
  observed="$(
    find "${run_root}" -path '*/calibrated_adapters/.checkpoints/*.json' -type f \
      | wc -l
  )"
  if (( observed == expected )); then
    next_stage="kl-calibration-aggregate"
  fi
fi
if [[ "${next_stage}" = "rollout" ]]; then
  expected=$(( ${#seeds[@]} * 10 ))
  observed="$(
    find "${run_root}" -path '*/policy_rollout_parts/*/receipt.json' -type f \
      | wc -l
  )"
  if (( observed == expected )); then
    next_stage="rollout-aggregate"
  fi
fi

output="$(
  PRORM_SOURCE_RUN_ROOT="${source_run_root}" \
    bash "${repo_root}/scripts/hpc4/submit_pipeline.sh" \
    "${config}" "${image}" "${hf_cache}" "${run_root}" "${next_stage}"
)"
job_id="$(awk -F= '$1 == "job_id" {print $2}' <<< "${output}")"
job_id="${job_id%%;*}"
[[ "${job_id}" =~ ^[0-9]+$ ]] || {
  echo "failed to parse job ID for ${next_stage}" >&2
  exit 2
}
printf '%s\t%s\t%s\n' "$(date -Is)" "${next_stage}" "${job_id}" >> "${manifest}"
printf 'stage=%s\n' "${next_stage}"
printf 'job_id=%s\n' "${job_id}"
printf 'submission_manifest=%s\n' "${manifest}"
