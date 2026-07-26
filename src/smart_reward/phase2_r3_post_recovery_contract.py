"""Primitive contracts for the R3 Gate-R/Gate-C to Gate-F bridge.

This module intentionally contains constants only.  Keeping the wire schemas
and production paths here lets both the configuration validator and the HPC4
evidence plane share one contract without introducing an import cycle through
the Phase-2 training implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

R3_FINAL_AUTHORIZATION_SCHEMA: Final = "phase2-recovery-r3-gate-c-success-authorization/v1"
R3_FINAL_AUTHORIZATION_ROLE: Final = "head_free_exact_three_by_three_gate_c_success_capability"
R3_FINAL_AUTHORIZATION_REFERENCE_SCHEMA: Final = (
    "phase2-recovery-r3-final-authorization-reference/v1"
)
R3_FINAL_AUTHORIZATION_PROJECTION_SCHEMA: Final = (
    "phase2-recovery-r3-final-authorization-projection/v1"
)

R3_PRODUCTION_PROJECT_ROOT: Final = Path("/project/sigroup/smart-reward-model")
R3_GATE_R_AUTHORIZATION_RELATIVE: Final = Path(
    "runs/phase2-recovery-r3/recovery-success-authorization.json"
)
R3_GATE_C_AGGREGATE_RELATIVE: Final = Path("runs/phase2-recovery-r3-controls/gate-c-aggregate.json")
R3_FINAL_AUTHORIZATION_RELATIVE: Final = Path(
    "runs/phase2-recovery-r3-controls/gate-c-success-authorization.json"
)

R3_EXECUTION_REVISION: Final = 3
R3_ORDERED_RECOVERY_SEEDS: Final = (20260801, 20260802, 20260803)
R3_OPTIMIZER_SCHEDULE_SHA256: Final = (
    "46e0b0fdc70c507b0325c445068326ba7bc30326d70b65fa33803cc3c876c216"
)
R3_AUTHORIZED_INFORMATION: Final = "gate_r_design_optimizer_schedule_and_gate_source_hashes_only"
R3_AUTHORIZED_NEXT_ACTION: Final = "materialize_fresh_common_beta_calibration"

R3_TRANSPORT_BOUNDARY: Final = {
    "parameters": False,
    "optimizer_moments": False,
    "checkpoints": False,
    "labels_or_data": False,
    "gradients_or_directions": False,
    "validation_or_test_values": False,
    "policy_outputs": False,
    "utility_values": False,
    "beta_values": False,
}

R3_FINAL_AUTHORIZATION_PROJECTION_FIELDS: Final = (
    "role",
    "recovery_design_sha256",
    "optimizer_schedule_sha256",
    "optimizer_schedule_is_unique",
    "execution_revision",
    "ordered_seeds",
    "gate_r_authorization_path",
    "gate_r_authorization_file_sha256",
    "gate_r_authorization_sha256",
    "gate_c_aggregate_path",
    "gate_c_aggregate_file_sha256",
    "gate_c_aggregate_sha256",
    "gate_c_source_set_sha256",
    "gate_r_passed",
    "gate_c_passed",
    "fresh_calibration_authorized",
    "authorized_information",
    "authorized_next_action",
    "formal_efficacy_claim_authorized",
    "recovery_or_control_outputs_reusable",
    "validation_or_heldout_access_authorized",
    "policy_or_final_utility_access_authorized",
    "transport_boundary",
    "authorization_sha256",
)


__all__ = [
    "R3_AUTHORIZED_INFORMATION",
    "R3_AUTHORIZED_NEXT_ACTION",
    "R3_EXECUTION_REVISION",
    "R3_FINAL_AUTHORIZATION_PROJECTION_FIELDS",
    "R3_FINAL_AUTHORIZATION_PROJECTION_SCHEMA",
    "R3_FINAL_AUTHORIZATION_REFERENCE_SCHEMA",
    "R3_FINAL_AUTHORIZATION_RELATIVE",
    "R3_FINAL_AUTHORIZATION_ROLE",
    "R3_FINAL_AUTHORIZATION_SCHEMA",
    "R3_GATE_C_AGGREGATE_RELATIVE",
    "R3_GATE_R_AUTHORIZATION_RELATIVE",
    "R3_OPTIMIZER_SCHEDULE_SHA256",
    "R3_ORDERED_RECOVERY_SEEDS",
    "R3_PRODUCTION_PROJECT_ROOT",
    "R3_TRANSPORT_BOUNDARY",
]
