from __future__ import annotations

import json

import pytest
import torch

from smart_reward.data import CandidateNode, SchemaError, load_jsonl, save_jsonl
from smart_reward.oracle import fit_affine_oracle_transform


def node(index: int = 0) -> CandidateNode:
    return CandidateNode(
        prompt_id="p",
        candidate_id=f"p::{index}",
        candidate_index=index,
        split="train",
        prompt="question",
        response="answer",
        raw_oracle_score=0.25 + index,
        oracle_reward=-0.5 + index,
        token_ids=(1, 2, 3),
        response_mask=(0, 1, 1),
        terminated_by_eos=True,
        reached_max_length=False,
    )


def test_candidate_jsonl_roundtrip(tmp_path) -> None:
    path = tmp_path / "nodes.jsonl"
    save_jsonl(path, [node(0), node(1)])
    assert load_jsonl(path, CandidateNode) == [node(0), node(1)]


def test_candidate_jsonl_exposes_oracle_scores() -> None:
    value = node().to_dict()
    assert value["raw_oracle_score"] == 0.25
    assert value["oracle_reward"] == -0.5
    assert value["candidate_index"] == 0
    assert value["split"] == "train"


def test_candidate_rejects_unknown_fields() -> None:
    value = node().to_dict()
    value["h"] = 1.0
    with pytest.raises(SchemaError):
        CandidateNode.from_dict(value)


def test_jsonl_rejects_blank_lines(tmp_path) -> None:
    path = tmp_path / "nodes.jsonl"
    path.write_text(json.dumps(node().to_dict()) + "\n\n", encoding="utf-8")
    with pytest.raises(SchemaError, match="blank"):
        load_jsonl(path, CandidateNode)


def test_oracle_transform_uses_median_and_mad_without_tanh() -> None:
    scores = torch.tensor([-2.0, 0.0, 2.0, 4.0])
    transform = fit_affine_oracle_transform(scores)
    transformed = transform(scores)
    assert transform.b == 0.0
    assert torch.allclose(transformed, (scores - transform.b) / transform.tau)
    assert transformed.max() > 1.0
