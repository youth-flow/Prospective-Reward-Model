from __future__ import annotations

import json
from pathlib import Path

import pytest

from smart_reward.checkpoints import validate_stage_receipt, write_stage_receipt
from smart_reward.config import load_config

ROOT = Path(__file__).parents[1]


def _clear_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "PRORM_GIT_COMMIT",
        "PRORM_IMAGE_SHA256",
        "PRORM_HF_INVENTORY_SHA256",
        "SLURM_JOB_ID",
    ):
        monkeypatch.delenv(name, raising=False)


def test_stage_receipt_roundtrip_and_output_binding(tmp_path, monkeypatch) -> None:
    _clear_identity(monkeypatch)
    config = load_config(ROOT / "configs" / "smoke.yaml")
    path = tmp_path / "receipt.json"
    inputs = {"artifact": "a" * 64}
    outputs = {"result": "b" * 64}
    write_stage_receipt(
        path,
        config,
        stage="reward",
        seed=20261001,
        inputs=inputs,
        outputs=outputs,
    )
    receipt = validate_stage_receipt(
        path,
        config,
        stage="reward",
        seed=20261001,
        inputs=inputs,
        outputs=outputs,
    )
    assert receipt["status"] == "complete"
    with pytest.raises(ValueError, match="receipt mismatch"):
        validate_stage_receipt(
            path,
            config,
            stage="reward",
            seed=20261001,
            inputs=inputs,
            outputs={"result": "c" * 64},
        )


def test_stage_receipt_rejects_changed_producer(tmp_path, monkeypatch) -> None:
    _clear_identity(monkeypatch)
    config = load_config(ROOT / "configs" / "smoke.yaml")
    path = tmp_path / "receipt.json"
    write_stage_receipt(
        path,
        config,
        stage="materialize",
        seed=20261001,
        inputs={},
        outputs={"artifact": "d" * 64},
    )
    monkeypatch.setenv("PRORM_GIT_COMMIT", "e" * 40)
    with pytest.raises(ValueError, match="receipt mismatch"):
        validate_stage_receipt(
            path,
            config,
            stage="materialize",
            seed=20261001,
            inputs={},
            outputs={"artifact": "d" * 64},
        )


def test_stage_receipt_is_strict_json(tmp_path, monkeypatch) -> None:
    _clear_identity(monkeypatch)
    config = load_config(ROOT / "configs" / "smoke.yaml")
    path = tmp_path / "receipt.json"
    write_stage_receipt(
        path,
        config,
        stage="adapters",
        seed=20261001,
        inputs={"reward": "f" * 64},
        outputs={"metadata": "1" * 64},
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert set(payload["producer"]) == set()
    assert path.read_bytes().endswith(b"\n")
