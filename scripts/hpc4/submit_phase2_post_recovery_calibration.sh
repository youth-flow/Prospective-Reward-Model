#!/usr/bin/env bash
set -euo pipefail

die() { echo "error: $*" >&2; exit 2; }

if [[ $# -ne 2 ]]; then
  die "usage: $0 <recovery-success-authorization.json> <walltime>"
fi

authorization="$1"
walltime="$2"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
overlay="${repo_root}/configs/common_beta_post_recovery_calibration.yaml"
generic_submit="${repo_root}/scripts/hpc4/submit_phase2_post_recovery_pilot.sh"

[[ -f "${overlay}" && ! -L "${overlay}" ]] \
  || die "locked calibration overlay is missing or unsafe"
[[ -f "${generic_submit}" && ! -L "${generic_submit}" ]] \
  || die "generic post-recovery submitter is missing or unsafe"

# All authorization, source, and scheduler checks are intentionally owned by
# the generic path.  In particular it invokes/locks:
# - validate_phase2_recovery_authorization.py;
# - submit_phase2_post_recovery_array_once.py;
# - authorized action issue_schedule_frozen_full_common_beta_calibration_pilot;
# - authorization must be the locked recovery receipt.
exec bash "${generic_submit}" "${overlay}" "${authorization}" "${walltime}"
