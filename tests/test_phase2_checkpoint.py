from __future__ import annotations

import json
import random
from pathlib import Path

import pytest
import torch

import smart_reward.phase2_checkpoint as phase2_checkpoint
from smart_reward.phase2_checkpoint import (
    CHECKPOINT_MANIFEST_SCHEMA,
    CHECKPOINT_SCHEMA,
    PROGRESS_SCHEMA,
    SIGNAL_RECEIPT_SCHEMA,
    TRAINING_PROGRESS_DETAILS_SCHEMA,
    CheckpointSignal,
    DurableCheckpointStore,
)


def _binding(tag: str = "a") -> dict[str, object]:
    return {
        "schema_version": "phase2-checkpoint-test-binding/v1",
        "design_sha256": tag * 64,
        "input_sha256": "b" * 64,
        "label_stream_sha256": "c" * 64,
        "seed": 20260801,
    }


def test_durable_checkpoint_round_trip_restores_payload_rng_and_progress(
    tmp_path: Path,
) -> None:
    np = pytest.importorskip("numpy")
    store = DurableCheckpointStore(
        tmp_path,
        objective="bt_mle",
        binding=_binding(),
    )
    random.seed(101)
    np.random.seed(151)
    torch.manual_seed(202)
    manifest = store.save(
        {
            "controller_schema": "test-controller/v1",
            "trainer": {"weight": torch.tensor([1.0, 2.0])},
        },
        completed_steps=20,
        reason="interval",
    )
    expected_python = random.random()
    expected_numpy = np.random.random(3)
    expected_torch = torch.rand(3)

    random.seed(303)
    np.random.seed(353)
    torch.manual_seed(404)
    payload = store.load()

    assert payload is not None
    assert payload["controller_schema"] == "test-controller/v1"
    assert torch.equal(payload["trainer"]["weight"], torch.tensor([1.0, 2.0]))
    # Caller restores model/optimizer/controller state here, then consumes the
    # RNG token immediately before the next audit/update.
    store.restore_pending_rng_state()
    assert random.random() == expected_python
    assert (np.random.random(3) == expected_numpy).all()
    assert torch.equal(torch.rand(3), expected_torch)
    assert manifest["schema_version"] == CHECKPOINT_MANIFEST_SCHEMA
    assert manifest["generation"] == 1
    assert manifest["completed_steps"] == 20
    generation = store.generations_path / "generation-00000001"
    assert generation.is_dir()
    envelope = torch.load(generation / "state.pt", weights_only=True)
    assert envelope["schema_version"] == CHECKPOINT_SCHEMA
    progress_files = sorted(store.progress_directory.iterdir())
    assert [path.name for path in progress_files] == [
        "event-00000001.json",
        "event-00000002.json",
    ]
    progress = [json.loads(path.read_text(encoding="utf-8")) for path in progress_files]
    assert all(item["schema_version"] == PROGRESS_SCHEMA for item in progress)
    assert [item["status"] for item in progress] == [
        "checkpointed",
        "checkpoint_loaded",
    ]
    assert progress[0]["previous_progress_sha256"] is None
    assert len(progress[1]["previous_progress_sha256"]) == 64


def test_durable_checkpoint_keeps_immutable_generations_and_uses_newest(
    tmp_path: Path,
) -> None:
    store = DurableCheckpointStore(tmp_path, objective="prorm_plus", binding=_binding())
    first = store.save({"value": 1}, completed_steps=20, reason="interval")
    stale_latest = store.manifest_path.read_bytes()
    second = store.save({"value": 2}, completed_steps=40, reason="stage_boundary")

    assert first["generation"] == 1
    assert second["generation"] == 2
    assert sorted(path.name for path in store.generations_path.iterdir()) == [
        "generation-00000001",
        "generation-00000002",
    ]

    # Simulate a crash after generation commit but before the LATEST pointer
    # and progress updates.  Committed generation directories are primary.
    store.manifest_path.write_bytes(stale_latest)
    (store.progress_directory / "event-00000002.json").unlink()
    payload = store.load()
    assert payload == {"value": 2}
    statuses = [
        json.loads(path.read_text(encoding="utf-8"))["status"]
        for path in sorted(store.progress_directory.iterdir())
    ]
    assert statuses == [
        "checkpointed",
        "checkpoint_discovered_after_crash",
        "checkpoint_loaded",
    ]


def test_checkpoint_binding_and_committed_bytes_fail_closed(tmp_path: Path) -> None:
    store = DurableCheckpointStore(tmp_path, objective="bt_mle", binding=_binding())
    store.save({"value": 1}, completed_steps=20, reason="interval")

    wrong_identity = DurableCheckpointStore(
        tmp_path,
        objective="bt_mle",
        binding=_binding("d"),
    )
    with pytest.raises(ValueError, match="identity"):
        wrong_identity.load()

    checkpoint = store.generations_path / "generation-00000001" / "state.pt"
    with checkpoint.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(RuntimeError, match="checkpoint bytes"):
        store.load()


def test_terminal_audit_can_verify_every_historical_state_file(tmp_path: Path) -> None:
    store = DurableCheckpointStore(tmp_path, objective="bt_mle", binding=_binding())
    store.save({"value": 1}, completed_steps=20, reason="interval")
    store.save({"value": 2}, completed_steps=40, reason="interval")
    assert len(store.audit_generations(verify_all_checkpoint_bytes=True)) == 2

    old_state = store.generations_path / "generation-00000001" / "state.pt"
    with old_state.open("ab") as stream:
        stream.write(b"historical-tamper")
    # Operational resume verifies the complete metadata chain and newest state.
    assert len(store.audit_generations(verify_all_checkpoint_bytes=False)) == 2
    # Terminal publication performs the expensive all-state verification.
    with pytest.raises(RuntimeError, match="checkpoint bytes"):
        store.audit_generations(verify_all_checkpoint_bytes=True)


def test_checkpoint_append_does_not_rehash_historical_state_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = DurableCheckpointStore(tmp_path, objective="bt_mle", binding=_binding())
    store.save({"value": 1}, completed_steps=20, reason="interval")
    store.save({"value": 2}, completed_steps=40, reason="interval")
    original = phase2_checkpoint._sha256_file
    hashed_state_paths: list[Path] = []

    def observe(path: Path) -> str:
        if path.name == "state.pt":
            hashed_state_paths.append(path)
        return original(path)

    monkeypatch.setattr(phase2_checkpoint, "_sha256_file", observe)
    store.save({"value": 3}, completed_steps=60, reason="interval")

    assert len(hashed_state_paths) == 1
    assert ".generation-00000003." in hashed_state_paths[0].parent.name


def test_resume_load_preserves_recorded_tensor_devices_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = DurableCheckpointStore(tmp_path, objective="bt_mle", binding=_binding())
    store.save({"weight": torch.tensor([1.0])}, completed_steps=20, reason="interval")
    original = torch.load
    observed: list[object] = []

    def load_with_observation(*args, **kwargs):
        observed.append(kwargs.get("map_location"))
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "load", load_with_observation)
    payload = store.load()

    assert payload is not None
    assert observed == [None]


def test_checkpoint_rejects_pointer_without_generation(tmp_path: Path) -> None:
    store = DurableCheckpointStore(tmp_path, objective="bt_mle", binding=_binding())
    store.manifest_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="without any generation"):
        store.load()


def test_checkpoint_signal_handler_only_latches_request() -> None:
    latch = CheckpointSignal()
    latch._handle(15, None)
    assert latch.requested is True
    assert latch.signal_name in {"SIGTERM", "15"}
    assert latch.received_at_utc is not None
    assert latch.received_at_utc.endswith("Z")
    first_name = latch.signal_name
    first_time = latch.received_at_utc
    latch._handle(2, None)
    assert latch.signal_name == first_name
    assert latch.received_at_utc == first_time
    assert latch.additional_signal_count == 1


def test_checkpoint_signal_captures_exact_first_in_flight_update() -> None:
    latch = CheckpointSignal()
    assert latch.begin_update(21) is True
    assert latch.active_update == 21

    latch._handle(15, None)
    latch._handle(2, None)
    latch.end_update(21)

    assert latch.requested is True
    assert latch.received_in_flight_update == 21
    assert latch.active_update is None
    assert latch.additional_signal_count == 1
    assert latch.begin_update(22) is False


def test_checkpoint_signal_refuses_update_when_already_requested() -> None:
    latch = CheckpointSignal()
    latch._handle(15, None)

    assert latch.begin_update(1) is False
    assert latch.active_update is None
    assert latch.received_in_flight_update is None


def test_strict_training_progress_is_hash_chained_and_train_only(
    tmp_path: Path,
) -> None:
    store = DurableCheckpointStore(tmp_path, objective="prorm_plus", binding=_binding())
    first = store.record_training_progress(
        status="running",
        completed_steps=20,
        next_update=21,
        learning_rate=1.0e-3,
        gradient_ratio=0.25,
        consecutive_passes=2,
        pcg={
            "iterations": 7,
            "relative_residual": 2.0e-8,
            "reason": "converged",
            "converged": True,
        },
        current_gpu_memory_bytes=100,
        peak_gpu_memory_bytes=120,
        cumulative_training_seconds=10.0,
        cumulative_audit_seconds=2.0,
        cumulative_checkpoint_io_seconds=0.5,
        checkpoint_metadata_sha256="d" * 64,
        signal_state={
            "requested": False,
            "signal_name": None,
            "received_at_utc": None,
            "additional_signal_count": 0,
        },
        scheduler_segment=1,
        remaining_allocation_seconds=300.0,
    )
    first_sha256 = store.latest_progress_sha256()
    second = store.record_training_progress(
        status="completed",
        completed_steps=40,
        next_update=None,
        learning_rate=None,
        gradient_ratio=0.01,
        consecutive_passes=3,
        pcg={
            "iterations": 5,
            "relative_residual": 1.0e-9,
            "reason": "converged",
            "converged": True,
        },
        current_gpu_memory_bytes=90,
        peak_gpu_memory_bytes=120,
        cumulative_training_seconds=20.0,
        cumulative_audit_seconds=4.0,
        cumulative_checkpoint_io_seconds=1.0,
        checkpoint_metadata_sha256="e" * 64,
        signal_state={
            "requested": True,
            "signal_name": "SIGUSR1",
            "received_at_utc": "2026-07-26T08:00:00Z",
            "additional_signal_count": 1,
        },
        scheduler_segment=2,
        remaining_allocation_seconds=None,
    )

    first_details = first["details"]
    second_details = second["details"]
    assert first_details["schema_version"] == TRAINING_PROGRESS_DETAILS_SCHEMA
    assert first_details["information_boundary"] == "train_only"
    assert first_details["last_progress_sha256_before_this_event"] is None
    assert second["previous_progress_sha256"] == first_sha256
    assert second_details["last_progress_sha256_before_this_event"] == first_sha256
    assert second_details["next_update"] is None


def test_strict_training_progress_allows_finalizing_without_next_update(
    tmp_path: Path,
) -> None:
    store = DurableCheckpointStore(tmp_path, objective="bt_mle", binding=_binding())
    event = store.record_training_progress(
        status="finalizing",
        completed_steps=5760,
        next_update=None,
        learning_rate=None,
        gradient_ratio=1.0e-4,
        consecutive_passes=3,
        pcg=None,
        current_gpu_memory_bytes=100,
        peak_gpu_memory_bytes=120,
        cumulative_training_seconds=10.0,
        cumulative_audit_seconds=2.0,
        cumulative_checkpoint_io_seconds=0.5,
        checkpoint_metadata_sha256="d" * 64,
        signal_state={
            "requested": False,
            "signal_name": None,
            "received_at_utc": None,
            "additional_signal_count": 0,
        },
        scheduler_segment=1,
        remaining_allocation_seconds=300.0,
    )
    assert event["status"] == "finalizing"
    assert event["details"]["next_update"] is None
    with pytest.raises(ValueError, match="cannot advertise a next_update"):
        store.record_training_progress(
            status="finalizing",
            completed_steps=5760,
            next_update=5761,
            learning_rate=3.0e-4,
            gradient_ratio=1.0e-4,
            consecutive_passes=3,
            pcg=None,
            current_gpu_memory_bytes=100,
            peak_gpu_memory_bytes=120,
            cumulative_training_seconds=10.0,
            cumulative_audit_seconds=2.0,
            cumulative_checkpoint_io_seconds=0.5,
            checkpoint_metadata_sha256="d" * 64,
            signal_state={
                "requested": False,
                "signal_name": None,
                "received_at_utc": None,
                "additional_signal_count": 0,
            },
            scheduler_segment=1,
            remaining_allocation_seconds=300.0,
        )


@pytest.mark.parametrize(
    "status",
    [
        "initialized",
        "checkpoint_loaded",
        "resumed",
        "checkpointed",
        "signal_checkpointed",
        "checkpoint_discovered_after_crash",
        "not-a-status",
    ],
)
def test_strict_training_progress_rejects_non_training_status_before_writing(
    tmp_path: Path,
    status: str,
) -> None:
    store = DurableCheckpointStore(tmp_path, objective="bt_mle", binding=_binding())

    with pytest.raises(ValueError, match="invalid training progress status"):
        store.record_training_progress(
            status=status,
            completed_steps=20,
            next_update=21,
            learning_rate=1.0e-3,
            gradient_ratio=0.25,
            consecutive_passes=2,
            pcg=None,
            current_gpu_memory_bytes=100,
            peak_gpu_memory_bytes=120,
            cumulative_training_seconds=10.0,
            cumulative_audit_seconds=2.0,
            cumulative_checkpoint_io_seconds=0.5,
            checkpoint_metadata_sha256="d" * 64,
            signal_state={
                "requested": False,
                "signal_name": None,
                "received_at_utc": None,
                "additional_signal_count": 0,
            },
            scheduler_segment=1,
            remaining_allocation_seconds=300.0,
        )

    assert list(store.progress_directory.iterdir()) == []
    assert not store.progress_path.exists()


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"next_update": None, "learning_rate": None}, "requires next_update"),
        ({"next_update": 22}, "completed_steps \\+ 1"),
        (
            {"status": "completed", "next_update": 21},
            "cannot advertise a next_update",
        ),
        (
            {"status": "failed", "next_update": None, "learning_rate": 1.0e-3},
            "requires an active next_update",
        ),
        (
            {"current_gpu_memory_bytes": 121},
            "cannot exceed peak",
        ),
        (
            {
                "signal_state": {
                    "requested": False,
                    "signal_name": "SIGTERM",
                    "received_at_utc": None,
                    "additional_signal_count": 0,
                }
            },
            "cannot contain signal metadata",
        ),
    ],
)
def test_strict_training_progress_rejects_inconsistent_state(
    tmp_path: Path,
    override: dict[str, object],
    message: str,
) -> None:
    store = DurableCheckpointStore(tmp_path, objective="bt_mle", binding=_binding())
    arguments: dict[str, object] = {
        "status": "running",
        "completed_steps": 20,
        "next_update": 21,
        "learning_rate": 1.0e-3,
        "gradient_ratio": 0.25,
        "consecutive_passes": 2,
        "pcg": None,
        "current_gpu_memory_bytes": 100,
        "peak_gpu_memory_bytes": 120,
        "cumulative_training_seconds": 10.0,
        "cumulative_audit_seconds": 2.0,
        "cumulative_checkpoint_io_seconds": 0.5,
        "checkpoint_metadata_sha256": "d" * 64,
        "signal_state": {
            "requested": False,
            "signal_name": None,
            "received_at_utc": None,
            "additional_signal_count": 0,
        },
        "scheduler_segment": 1,
        "remaining_allocation_seconds": 300.0,
    }
    arguments.update(override)

    with pytest.raises(ValueError, match=message):
        store.record_training_progress(**arguments)


def test_signal_receipts_are_immutable_hash_chained_and_not_success(
    tmp_path: Path,
) -> None:
    store = DurableCheckpointStore(tmp_path, objective="bt_mle", binding=_binding())
    manifest = store.save({"value": 1}, completed_steps=20, reason="signal")
    first = store.record_signal_receipt(
        head_name="bt_mle",
        signal_name="SIGUSR1",
        received_at_utc="2026-07-26T08:00:00Z",
        additional_signal_count=0,
        completed_steps=20,
        in_flight_update=20,
        reached_safe_boundary=True,
        checkpoint_metadata_sha256=manifest["metadata_sha256"],
        checkpoint_flush_succeeded=True,
        checkpoint_verified=True,
        last_progress_sha256=store.latest_progress_sha256(),
        scheduler_identity={"job_id": "123", "segment": 1},
        planned_action="continue_same_logical_run",
    )
    second = store.record_signal_receipt(
        head_name="bt_mle",
        signal_name="SIGTERM",
        received_at_utc="2026-07-26T08:01:00Z",
        additional_signal_count=1,
        completed_steps=20,
        in_flight_update=None,
        reached_safe_boundary=False,
        checkpoint_metadata_sha256=None,
        checkpoint_flush_succeeded=False,
        checkpoint_verified=False,
        last_progress_sha256=store.latest_progress_sha256(),
        scheduler_identity={"job_id": "123", "segment": 1},
        planned_action="fail_closed",
    )
    third = store.record_signal_receipt(
        head_name="bt_mle",
        signal_name="SIGTERM",
        received_at_utc="2026-07-26T08:02:00Z",
        additional_signal_count=2,
        completed_steps=21,
        in_flight_update=None,
        reached_safe_boundary=True,
        checkpoint_metadata_sha256=None,
        checkpoint_flush_succeeded=False,
        checkpoint_verified=False,
        last_progress_sha256=store.latest_progress_sha256(),
        scheduler_identity={"job_id": "123", "segment": 1},
        planned_action="fail_closed",
    )

    assert first["schema_version"] == SIGNAL_RECEIPT_SCHEMA
    assert first["terminal_success_claimed"] is False
    assert second["terminal_success_claimed"] is False
    assert third["reached_safe_boundary"] is True
    assert third["checkpoint_flush_succeeded"] is False
    assert third["continuation_checkpoint_usable"] is False
    assert len(second["previous_signal_receipt_sha256"]) == 64
    assert sorted(path.name for path in store.signal_directory.iterdir()) == [
        "event-00000001.json",
        "event-00000002.json",
        "event-00000003.json",
    ]


def test_planned_boundary_receipt_is_separate_and_hash_chained(
    tmp_path: Path,
) -> None:
    store = DurableCheckpointStore(tmp_path, objective="bt_mle", binding=_binding())
    manifest = store.save({"value": 1}, completed_steps=20, reason="stage_boundary")
    first = store.record_planned_boundary_receipt(
        head_name="bt_mle",
        completed_steps=20,
        checkpoint_metadata_sha256=manifest["metadata_sha256"],
        checkpoint_verified=True,
        last_progress_sha256=store.latest_progress_sha256(),
        scheduler_identity={"job_id": "123", "segment": 1},
        execution_slice_sha256="a" * 64,
        update_blocks_consumed=1,
        update_blocks_remaining=0,
        planned_action="continue_same_logical_run",
    )
    second = store.record_planned_boundary_receipt(
        head_name="bt_mle",
        completed_steps=20,
        checkpoint_metadata_sha256=manifest["metadata_sha256"],
        checkpoint_verified=True,
        last_progress_sha256=store.latest_progress_sha256(),
        scheduler_identity={"job_id": "124", "segment": 2},
        execution_slice_sha256="b" * 64,
        update_blocks_consumed=0,
        update_blocks_remaining=0,
        planned_action="fail_closed",
    )

    assert first["terminal_success_claimed"] is False
    assert second["previous_planned_boundary_receipt_sha256"]
    assert list(store.signal_directory.iterdir()) == []
    assert sorted(path.name for path in store.planned_boundary_directory.iterdir()) == [
        "event-00000001.json",
        "event-00000002.json",
    ]
