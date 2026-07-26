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

# The active route delegates in R3-only mode. The generic path inspects the
# fixed combined Gate-R/Gate-C authorization, then performs deep SIF
# verification before submitting exactly one immutable three-seed array.
exec bash "${generic_submit}" "${overlay}" "${authorization}" "${walltime}"
