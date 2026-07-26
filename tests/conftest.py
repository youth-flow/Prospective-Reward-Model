"""Shared white-box factories for validator-sealed R3 test capabilities.

Production code can issue these objects only after live validation.  Tests use
the private issuers deliberately so downstream fixtures exercise the same
exact-type and seal checks without weakening the public authorization APIs.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from functools import cache
from pathlib import Path
from tempfile import mkdtemp

import pytest

from smart_reward import phase2_r3_gate0, phase2_r3_gate1, phase2_r3_inputs
from smart_reward.phase2_r3_gate0 import R3Gate0Capability
from smart_reward.phase2_r3_gate1 import R3Gate1Capabilities
from smart_reward.phase2_r3_identity import (
    VERIFIED_CONTINUATION_CHECKPOINT_ROLE,
    VERIFIED_CONTINUATION_CHECKPOINT_SCHEMA,
    ArtifactRef,
    PrimarySegmentAdmission,
)
from smart_reward.phase2_r3_inputs import R3TrainMaterializationCapability
from smart_reward.phase2_r3_materialization import ValidatedR3Materialization
from smart_reward.phase2_r3_profile_artifacts import (
    VerifiedGatePOperationalBundle,
)
from smart_reward.phase2_r3_terminal import (
    ContinuablePrimaryTerminalCapability,
    SuccessfulProfileTerminalCapability,
)

ROOT = Path(__file__).resolve().parents[1]


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def make_sealed_r3_gate0_capability() -> R3Gate0Capability:
    return R3Gate0Capability(
        schema_version=phase2_r3_gate0.GATE0_ARTIFACT_SCHEMA,
        role=phase2_r3_gate0.GATE0_ARTIFACT_ROLE,
        artifact_sha256=_digest("gate0"),
        file_sha256=_digest("gate0-file"),
        production_relative=phase2_r3_gate0._GATE0_RELATIVE.as_posix(),
        _factory_token=phase2_r3_gate0._FACTORY_TOKEN,
    )


def make_sealed_r3_gate1_capabilities() -> R3Gate1Capabilities:
    source_payload = phase2_r3_gate1._source_capability_payload(
        artifact_sha256=_digest("source"),
        commit="1" * 40,
        tree="2" * 40,
        inventory_sha256=_digest("inventory"),
        formal_path_count=12,
    )
    source = phase2_r3_gate1.R3SourceCapability(
        **source_payload,
        capability_sha256=phase2_r3_gate1._canonical_sha256(source_payload),
        _factory_token=phase2_r3_gate1._SOURCE_FACTORY_TOKEN,
    )
    container_payload = phase2_r3_gate1._container_capability_payload(
        artifact_sha256=_digest("container"),
        canonical_sif_path=str((ROOT / "fixture.sif").resolve()),
        sif_sha256=_digest("sif"),
        sif_size_bytes=1,
        definition_git_blob_sha256=_digest("definition"),
        requirements_lock_git_blob_sha256=_digest("requirements"),
        runtime_probe_sha256=_digest("runtime"),
        live_runtime_probe_sha256=_digest("runtime"),
    )
    container = phase2_r3_gate1.R3ContainerCapability(
        **container_payload,
        capability_sha256=phase2_r3_gate1._canonical_sha256(container_payload),
        _factory_token=phase2_r3_gate1._CONTAINER_FACTORY_TOKEN,
    )
    gate1_payload = phase2_r3_gate1._gate1_capability_payload(
        artifact_sha256=_digest("gate1"),
        file_sha256=_digest("gate1-file"),
        source_artifact_sha256=source.artifact_sha256,
        container_artifact_sha256=container.artifact_sha256,
        source_test_receipt_artifact_sha256=_digest("source-test-receipt-artifact"),
        source_test_receipt_file_sha256=_digest("source-test-receipt-file"),
        verification_suite_sha256=_digest("suite"),
        live_reverification_sha256=_digest("live-reverification"),
        production_relative=phase2_r3_gate1._GATE1_RELATIVE.as_posix(),
    )
    gate1 = phase2_r3_gate1.R3Gate1Capability(
        **gate1_payload,
        capability_sha256=phase2_r3_gate1._canonical_sha256(gate1_payload),
        _factory_token=phase2_r3_gate1._GATE1_FACTORY_TOKEN,
    )
    return R3Gate1Capabilities(gate1=gate1, source=source, container=container)


def seal_r3_train_materialization_capability(
    materialization: ValidatedR3Materialization,
) -> R3TrainMaterializationCapability:
    provenance = materialization.provenance
    return phase2_r3_inputs._issue_train_materialization_capability(
        materialization=materialization,
        source_config_hash=(materialization.science_bundle.settings.source_config_hash),
        parent_registry_file_sha256=provenance.parent_artifact_registry_sha256,
        parent_seed_entry_sha256=_digest(f"seed-entry-{materialization.seed}"),
        artifact_metadata_sha256=provenance.artifact_metadata_sha256,
        artifact_tensors_sha256=provenance.artifact_tensors_sha256,
        artifact_candidates_sha256=provenance.artifact_candidates_sha256,
        candidate_train_prefix_sha256=(provenance.candidate_train_prefix_sha256),
        artifact_materialization_sha256=(provenance.artifact_materialization_sha256),
        artifact_verification_sha256=(provenance.artifact_verification_sha256),
        source_run_manifest_sha256=provenance.source_run_manifest_sha256,
        source_producer_identity_sha256=(provenance.source_producer_identity_sha256),
        oracle_chat_template_sha256=_digest("oracle-chat-template"),
        oracle_transform_sha256=_digest("oracle-transform"),
    )


class SharedGatePEvidence:
    __slots__ = (
        "science",
        "gate0_capability",
        "gate1_capabilities",
        "materialization_capability",
        "operational_bundle",
        "successful_terminal",
        "formal_result",
        "resource_plan",
    )

    def __init__(
        self,
        *,
        science: object,
        gate0_capability: R3Gate0Capability,
        gate1_capabilities: R3Gate1Capabilities,
        materialization_capability: R3TrainMaterializationCapability,
        operational_bundle: VerifiedGatePOperationalBundle,
        successful_terminal: SuccessfulProfileTerminalCapability,
        formal_result: object,
        resource_plan: object,
    ) -> None:
        self.science = science
        self.gate0_capability = gate0_capability
        self.gate1_capabilities = gate1_capabilities
        self.materialization_capability = materialization_capability
        self.operational_bundle = operational_bundle
        self.successful_terminal = successful_terminal
        self.formal_result = formal_result
        self.resource_plan = resource_plan


class _PersistentTempPathFactory:
    def mktemp(self, basename: str) -> Path:
        return Path(mkdtemp(prefix=f"r3-{basename}-")).resolve()


@cache
def make_shared_gate_p_evidence() -> SharedGatePEvidence:
    """Build one real sealed Gate-P closure reused by integration-style tests."""

    import importlib.util

    from smart_reward.phase2_r3_config import load_r3_science_config

    profile_path = ROOT / "tests" / "test_phase2_r3_profile.py"
    spec = importlib.util.spec_from_file_location("_r3_profile_test_support", profile_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load local R3 profile test support")
    profile_support = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(profile_support)

    science = load_r3_science_config(ROOT / "configs" / "phase2_recovery_r3_science.yaml")
    gate0 = make_sealed_r3_gate0_capability()
    gate1 = make_sealed_r3_gate1_capabilities()
    run = profile_support._validated_profile_run(
        science,
        gate0_capability=gate0,
        gate1_capabilities=gate1,
        seal_materialization=seal_r3_train_materialization_capability,
    )
    policy = profile_support.safety_policy.__wrapped__(run)
    envelope = profile_support.envelope.__wrapped__(run)
    preparation = profile_support.preparation.__wrapped__(run)
    paths = _PersistentTempPathFactory()
    bundle, formal_result, resource_plan = profile_support.sealed_operational_bundle.__wrapped__(
        run,
        policy,
        envelope,
        preparation,
        paths,
    )
    terminal = profile_support._successful_profile_terminal(
        bundle,
        paths.mktemp("successful-profile-terminal"),
    )
    return SharedGatePEvidence(
        science=science,
        gate0_capability=gate0,
        gate1_capabilities=gate1,
        materialization_capability=run.materialization_capability,
        operational_bundle=bundle,
        successful_terminal=terminal,
        formal_result=formal_result,
        resource_plan=resource_plan,
    )


def make_continuable_primary_terminal(
    *,
    predecessor: PrimarySegmentAdmission,
    operational_bundle: VerifiedGatePOperationalBundle,
    checkpoint_token: str | None = None,
) -> ContinuablePrimaryTerminalCapability:
    """Publish a real sealed continuation closure and scheduler terminal for tests."""

    from unittest.mock import patch

    from smart_reward import phase2_r3_orchestrator as orchestrator
    from smart_reward import phase2_r3_terminal as terminal
    from smart_reward.phase2_r3_primary import capture_slurm_segment_runtime

    predecessor.validate_integrity()
    operational_bundle.validate_integrity()
    checkpoint = ArtifactRef(
        schema_version=VERIFIED_CONTINUATION_CHECKPOINT_SCHEMA,
        artifact_sha256=_digest(
            checkpoint_token or f"continuation-checkpoint:{predecessor.admission_sha256}"
        ),
        role=VERIFIED_CONTINUATION_CHECKPOINT_ROLE,
    )
    array_job_id = str(610000 + 10 * predecessor.segment_index)
    job_id = str(int(array_job_id) + predecessor.task_id + 1)
    environment = {
        "SLURM_CLUSTER_NAME": "hpc4",
        "SLURM_JOB_ID": job_id,
        "SLURM_ARRAY_JOB_ID": array_job_id,
        "SLURM_ARRAY_TASK_ID": str(predecessor.task_id),
        "SLURM_JOB_ACCOUNT": operational_bundle.slurm_account,
        "SLURM_JOB_PARTITION": operational_bundle.partition,
    }
    with patch.dict("os.environ", environment, clear=False):
        runtime = capture_slurm_segment_runtime(
            predecessor,
            requested_walltime_seconds=(operational_bundle.requested_walltime_seconds_per_segment),
        )
    directory = _PersistentTempPathFactory().mktemp("continuable-primary-terminal")
    outcome = orchestrator._materialize_outcome(
        path=(directory / "segment-outcome.json").resolve(),
        admission=predecessor,
        runtime=runtime,
        operational_policy=operational_bundle,
        status="continuation_required_after_safe_checkpoint",
        receipts={},
        active_learner="bt_mle",
        continuation_checkpoint=checkpoint,
        continuation_reason="test sealed scheduler boundary",
    )
    (directory / "runtime-closures").mkdir()
    (directory / "terminal-evidence").mkdir()
    closure = terminal.publish_primary_segment_runtime_closure(
        (directory / "runtime-closures" / f"task-{predecessor.task_id}.json").resolve(),
        admission=predecessor,
        runtime=runtime,
        outcome=outcome,
        operational_bundle=operational_bundle,
    )
    memory_mib = operational_bundle.memory_bytes // 1024**2
    common_tres = (
        f"billing={operational_bundle.cpus_per_task},"
        f"cpu={operational_bundle.cpus_per_task},"
        f"gres/gpu={operational_bundle.gpus_per_task},"
        f"mem={memory_mib}M,node=1"
    )
    gpu_token = operational_bundle.partition.removeprefix("gpu-").lower()
    raw = (
        "|".join(
            (
                closure.job_selector,
                closure.job_id,
                "COMPLETED",
                "0:0",
                "0:0",
                "hpc4",
                operational_bundle.slurm_account,
                operational_bundle.partition,
                "test_qos",
                "1",
                str(operational_bundle.cpus_per_task),
                common_tres,
                (f"{common_tres},gres/gpu:{gpu_token}={operational_bundle.gpus_per_task}"),
                "1",
            )
        )
        + "\n"
    ).encode()
    inspection = terminal.inspect_sacct_terminal_bytes(
        raw,
        expected_raw_sha256=hashlib.sha256(raw).hexdigest(),
    )
    return terminal.produce_continuable_primary_terminal(
        operational_bundle,
        runtime_closure=closure,
        inspection=inspection,
        evidence_directory=(
            directory
            / "terminal-evidence"
            / f"task-{predecessor.task_id}-segment-{predecessor.segment_index}"
        ).resolve(),
    )


@pytest.fixture(scope="session")
def sealed_r3_gate0_capability() -> R3Gate0Capability:
    return make_sealed_r3_gate0_capability()


@pytest.fixture(scope="session")
def sealed_r3_gate1_capabilities() -> R3Gate1Capabilities:
    return make_sealed_r3_gate1_capabilities()


@pytest.fixture(scope="session")
def seal_r3_train_materialization() -> Callable[
    [ValidatedR3Materialization],
    R3TrainMaterializationCapability,
]:
    return seal_r3_train_materialization_capability
