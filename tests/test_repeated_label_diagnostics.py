from __future__ import annotations

import json
from collections.abc import Mapping

import pytest
import torch

from smart_reward.repeated_label_diagnostics import (
    REPEATED_LABEL_TAIL_DIAGNOSTICS_SCHEMA,
    build_repeated_label_tail_diagnostics,
    nearest_rank_summary,
    validate_repeated_label_tail_diagnostics,
)

COUNT_SHA = "1" * 64
REPLICATE_H_SHA = "2" * 64
MEAN_H_SHA = "3" * 64


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    counts = torch.tensor(
        [
            [1, 2],
            [3, 4],
            [5, 6],
            [7, 8],
        ],
        dtype=torch.int64,
    )
    replicate_h = torch.tensor(
        [
            [-8.0, 1.0],
            [-4.0, 3.0],
            [2.0, -2.0],
            [0.0, 7.0],
        ],
        dtype=torch.float64,
    )
    mean_h = torch.tensor([-1.5, 2.5], dtype=torch.float64)
    return counts, replicate_h, mean_h


def _diagnostics() -> dict[str, object]:
    counts, replicate_h, mean_h = _inputs()
    return build_repeated_label_tail_diagnostics(
        replicate_counts=counts,
        replicate_h=replicate_h,
        mean_h=mean_h,
        replicate_count_sha256=COUNT_SHA,
        replicate_h_sha256=REPLICATE_H_SHA,
        mean_h_sha256=MEAN_H_SHA,
    )


def _validate(value: object) -> dict[str, object]:
    return validate_repeated_label_tail_diagnostics(
        value,
        expected_num_edges=2,
        replicate_count_sha256=COUNT_SHA,
        replicate_h_sha256=REPLICATE_H_SHA,
        mean_h_sha256=MEAN_H_SHA,
    )


def test_nearest_rank_is_exact_one_indexed_ceiling_without_interpolation() -> None:
    assert nearest_rank_summary([4, 1, 3, 2]) == {
        "sample_size": 4,
        "p50": 2,
        "p90": 4,
        "p95": 4,
        "p99": 4,
        "max": 4,
    }
    assert nearest_rank_summary([9.5]) == {
        "sample_size": 1,
        "p50": 9.5,
        "p90": 9.5,
        "p95": 9.5,
        "p99": 9.5,
        "max": 9.5,
    }


def test_tail_diagnostics_have_exact_scalar_statistics_and_locked_semantics() -> None:
    diagnostics = _diagnostics()

    assert diagnostics["schema_version"] == REPEATED_LABEL_TAIL_DIAGNOSTICS_SCHEMA
    assert diagnostics["split"] == "train"
    assert diagnostics["gamma"] == 0.9
    assert diagnostics["num_replicates"] == 4
    assert diagnostics["scalar_only"] is True
    assert diagnostics["descriptive_only"] is True
    assert diagnostics["used_for_clipping"] is False
    assert diagnostics["used_for_selection"] is False
    assert diagnostics["used_for_gating"] is False
    assert diagnostics["metrics"] == {
        "replicate_count": {
            "sample_size": 8,
            "p50": 4,
            "p90": 8,
            "p95": 8,
            "p99": 8,
            "max": 8,
        },
        "abs_replicate_h": {
            "sample_size": 8,
            "p50": 2.0,
            "p90": 8.0,
            "p95": 8.0,
            "p99": 8.0,
            "max": 8.0,
        },
        "abs_mean_h": {
            "sample_size": 2,
            "p50": 1.5,
            "p90": 2.5,
            "p95": 2.5,
            "p99": 2.5,
            "max": 2.5,
        },
    }
    assert _validate(diagnostics) == diagnostics


def test_nearest_rank_summaries_are_permutation_invariant_but_source_binding_is_not() -> None:
    counts, replicate_h, mean_h = _inputs()
    row_permutation = torch.tensor([2, 0, 3, 1])
    column_permutation = torch.tensor([1, 0])
    permuted_counts = counts[row_permutation][:, column_permutation]
    permuted_h = replicate_h[row_permutation][:, column_permutation]
    permuted_mean = mean_h[column_permutation]

    same_sources = build_repeated_label_tail_diagnostics(
        replicate_counts=permuted_counts,
        replicate_h=permuted_h,
        mean_h=permuted_mean,
        replicate_count_sha256=COUNT_SHA,
        replicate_h_sha256=REPLICATE_H_SHA,
        mean_h_sha256=MEAN_H_SHA,
    )
    original = _diagnostics()
    assert same_sources["metrics"] == original["metrics"]
    assert same_sources["diagnostics_sha256"] == original["diagnostics_sha256"]

    changed_source = build_repeated_label_tail_diagnostics(
        replicate_counts=permuted_counts,
        replicate_h=permuted_h,
        mean_h=permuted_mean,
        replicate_count_sha256="4" * 64,
        replicate_h_sha256=REPLICATE_H_SHA,
        mean_h_sha256=MEAN_H_SHA,
    )
    assert changed_source["metrics"] == original["metrics"]
    assert changed_source["diagnostics_sha256"] != original["diagnostics_sha256"]


def test_diagnostics_serialize_no_vectors_or_raw_label_values() -> None:
    diagnostics = _diagnostics()

    def assert_scalar_tree(value: object) -> None:
        if isinstance(value, Mapping):
            for child in value.values():
                assert_scalar_tree(child)
            return
        assert not isinstance(value, (list, tuple, torch.Tensor))

    assert_scalar_tree(diagnostics)
    rendered = json.dumps(diagnostics, allow_nan=False, sort_keys=True)
    assert "replicate_counts" not in rendered
    assert "replicate_h_values" not in rendered
    assert "mean_h_values" not in rendered


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("descriptive_only", False, "descriptive_only"),
        ("used_for_clipping", True, "used_for_clipping"),
        ("used_for_selection", True, "used_for_selection"),
        ("used_for_gating", True, "used_for_gating"),
        ("diagnostics_sha256", "f" * 64, "does not bind"),
    ],
)
def test_validator_rejects_semantic_or_hash_tampering(
    field: str,
    replacement: object,
    message: str,
) -> None:
    diagnostics = _diagnostics()
    diagnostics[field] = replacement
    with pytest.raises(ValueError, match=message):
        _validate(diagnostics)


def test_validator_rejects_source_sample_size_and_order_statistic_tampering() -> None:
    diagnostics = _diagnostics()
    diagnostics["source_tensor_sha256"]["replicate_h_sha256"] = "9" * 64
    with pytest.raises(ValueError, match="does not bind"):
        _validate(diagnostics)

    diagnostics = _diagnostics()
    diagnostics["metrics"]["abs_mean_h"]["sample_size"] = 8
    with pytest.raises(ValueError, match="sample_size is inconsistent"):
        _validate(diagnostics)

    diagnostics = _diagnostics()
    diagnostics["metrics"]["abs_mean_h"]["p95"] = 1.0
    with pytest.raises(ValueError, match="order statistics decrease"):
        _validate(diagnostics)


def test_builder_and_nearest_rank_reject_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        nearest_rank_summary([1.0, float("nan")])

    counts, replicate_h, mean_h = _inputs()
    bad_replicate_h = replicate_h.clone()
    bad_replicate_h[0, 0] = float("inf")
    with pytest.raises(ValueError, match="must be finite"):
        build_repeated_label_tail_diagnostics(
            replicate_counts=counts,
            replicate_h=bad_replicate_h,
            mean_h=mean_h,
            replicate_count_sha256=COUNT_SHA,
            replicate_h_sha256=REPLICATE_H_SHA,
            mean_h_sha256=MEAN_H_SHA,
        )


def test_validator_rejects_extra_fields_and_non_scalar_metric_values() -> None:
    diagnostics = _diagnostics()
    diagnostics["raw_values"] = [1, 2]
    with pytest.raises(ValueError, match="contain exactly"):
        _validate(diagnostics)

    diagnostics = _diagnostics()
    diagnostics["metrics"]["replicate_count"]["max"] = [8]
    with pytest.raises((TypeError, ValueError), match="real scalar|positive integer"):
        _validate(diagnostics)
