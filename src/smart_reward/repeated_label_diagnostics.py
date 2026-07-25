"""Scalar-only descriptive diagnostics for the repeated-label stream.

The randomized-truncation estimator is unbiased with a finite second moment
under the locked Phase-2 design, but it is deliberately not clipped and can
have a heavy upper tail.  This module records deterministic empirical order
statistics without exposing any train-time vector and without creating a
selection, clipping, or acceptance rule.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from numbers import Integral, Real

import torch

REPEATED_LABEL_TAIL_DIAGNOSTICS_SCHEMA = "repeated-label-tail-diagnostics/v1"
NEAREST_RANK_QUANTILES: tuple[tuple[str, float], ...] = (
    ("p50", 0.50),
    ("p90", 0.90),
    ("p95", 0.95),
    ("p99", 0.99),
)
_METRIC_NAMES = ("replicate_count", "abs_replicate_h", "abs_mean_h")
_DIGEST_NAMES = (
    "replicate_count_sha256",
    "replicate_h_sha256",
    "mean_h_sha256",
)
_HEX_DIGITS = frozenset("0123456789abcdef")


def _canonical_sha256(value: Mapping[str, object]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _digest(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX_DIGITS for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _nonnegative_finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def nearest_rank_summary(values: Sequence[int | float]) -> dict[str, int | float]:
    """Return deterministic nearest-rank quantiles and the maximum.

    For a non-empty sample of size ``n``, sort values in ascending order and
    define the quantile at probability ``q`` as the one-indexed order statistic
    ``x_(ceil(q*n))``.  The reported maximum is ``x_(n)``.  No interpolation is
    performed.
    """

    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
        raise ValueError("values must be a non-empty sequence of finite real scalars")
    normalized: list[int | float] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"values[{index}] must be a real scalar")
        scalar = float(value)
        if not math.isfinite(scalar):
            raise ValueError(f"values[{index}] must be finite")
        normalized.append(int(value) if isinstance(value, Integral) else scalar)
    ordered = sorted(normalized)
    sample_size = len(ordered)
    result: dict[str, int | float] = {"sample_size": sample_size}
    for label, probability in NEAREST_RANK_QUANTILES:
        result[label] = ordered[math.ceil(probability * sample_size) - 1]
    result["max"] = ordered[-1]
    return result


def _tensor(value: object, *, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.requires_grad:
        raise ValueError(f"{name} must not require gradients")
    return value.detach()


def build_repeated_label_tail_diagnostics(
    *,
    replicate_counts: torch.Tensor,
    replicate_h: torch.Tensor,
    mean_h: torch.Tensor,
    replicate_count_sha256: str,
    replicate_h_sha256: str,
    mean_h_sha256: str,
) -> dict[str, object]:
    """Build the locked Phase-2 scalar-only repeated-label tail record."""

    counts = _tensor(replicate_counts, name="replicate_counts")
    replicate_estimates = _tensor(replicate_h, name="replicate_h")
    mean_estimates = _tensor(mean_h, name="mean_h")
    if counts.dtype == torch.bool or counts.is_floating_point() or counts.is_complex():
        raise TypeError("replicate_counts must have an integer dtype")
    if not replicate_estimates.is_floating_point() or not mean_estimates.is_floating_point():
        raise TypeError("replicate_h and mean_h must have floating-point dtypes")
    if counts.ndim != 2 or tuple(replicate_estimates.shape) != tuple(counts.shape):
        raise ValueError("replicate_counts and replicate_h must share shape [R, num_edges]")
    if counts.shape[0] != 4 or tuple(mean_estimates.shape) != (counts.shape[1],):
        raise ValueError("tail diagnostics require R=4 and one mean_h scalar per train edge")
    if counts.numel() < 1 or bool((counts < 1).any()):
        raise ValueError("replicate_counts must contain positive annotation counts")
    if not bool(torch.isfinite(replicate_estimates).all()) or not bool(
        torch.isfinite(mean_estimates).all()
    ):
        raise ValueError("replicate_h and mean_h must be finite")

    source_digests = {
        "replicate_count_sha256": _digest(
            replicate_count_sha256,
            name="replicate_count_sha256",
        ),
        "replicate_h_sha256": _digest(replicate_h_sha256, name="replicate_h_sha256"),
        "mean_h_sha256": _digest(mean_h_sha256, name="mean_h_sha256"),
    }
    count_values = [int(value) for value in counts.to(device="cpu").reshape(-1).tolist()]
    replicate_h_values = [
        abs(float(value)) for value in replicate_estimates.to(device="cpu").reshape(-1).tolist()
    ]
    mean_h_values = [
        abs(float(value)) for value in mean_estimates.to(device="cpu").reshape(-1).tolist()
    ]
    payload: dict[str, object] = {
        "schema_version": REPEATED_LABEL_TAIL_DIAGNOSTICS_SCHEMA,
        "split": "train",
        "gamma": 0.9,
        "num_replicates": 4,
        "quantile_estimator": {
            "name": "nearest_rank",
            "sorting": "ascending",
            "rank_formula": "k=ceil(q*n)",
            "indexing": "one_indexed",
            "p50_probability": 0.50,
            "p90_probability": 0.90,
            "p95_probability": 0.95,
            "p99_probability": 0.99,
            "maximum_definition": "x_(n)",
            "interpolation": False,
        },
        "source_tensor_sha256": source_digests,
        "metrics": {
            "replicate_count": nearest_rank_summary(count_values),
            "abs_replicate_h": nearest_rank_summary(replicate_h_values),
            "abs_mean_h": nearest_rank_summary(mean_h_values),
        },
        "scalar_only": True,
        "descriptive_only": True,
        "used_for_clipping": False,
        "used_for_selection": False,
        "used_for_gating": False,
    }
    payload["diagnostics_sha256"] = _canonical_sha256(payload)
    return payload


def validate_repeated_label_tail_diagnostics(
    value: object,
    *,
    expected_num_edges: int,
    replicate_count_sha256: str,
    replicate_h_sha256: str,
    mean_h_sha256: str,
    name: str = "repeated_label_tail_diagnostics",
) -> dict[str, object]:
    """Strictly validate and normalize one scalar-only diagnostics record."""

    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    expected_keys = {
        "schema_version",
        "split",
        "gamma",
        "num_replicates",
        "quantile_estimator",
        "source_tensor_sha256",
        "metrics",
        "scalar_only",
        "descriptive_only",
        "used_for_clipping",
        "used_for_selection",
        "used_for_gating",
        "diagnostics_sha256",
    }
    if set(value) != expected_keys:
        raise ValueError(f"{name} must contain exactly {sorted(expected_keys)!r}")
    locked_strings = {
        "schema_version": REPEATED_LABEL_TAIL_DIAGNOSTICS_SCHEMA,
        "split": "train",
    }
    for field, expected in locked_strings.items():
        if not isinstance(value[field], str) or value[field] != expected:
            raise ValueError(f"{name}.{field} must equal {expected!r}")
    if (
        isinstance(value["gamma"], bool)
        or not isinstance(value["gamma"], Real)
        or float(value["gamma"]) != 0.9
    ):
        raise ValueError(f"{name}.gamma must equal 0.9")
    if (
        isinstance(value["num_replicates"], bool)
        or not isinstance(value["num_replicates"], Integral)
        or int(value["num_replicates"]) != 4
    ):
        raise ValueError(f"{name}.num_replicates must equal 4")
    locked_booleans = {
        "scalar_only": True,
        "descriptive_only": True,
        "used_for_clipping": False,
        "used_for_selection": False,
        "used_for_gating": False,
    }
    for field, expected in locked_booleans.items():
        if type(value[field]) is not bool or value[field] is not expected:
            raise ValueError(f"{name}.{field} must equal {expected!r}")

    quantiles = value["quantile_estimator"]
    if not isinstance(quantiles, Mapping):
        raise TypeError(f"{name}.quantile_estimator must be a mapping")
    expected_quantiles = {
        "name": "nearest_rank",
        "sorting": "ascending",
        "rank_formula": "k=ceil(q*n)",
        "indexing": "one_indexed",
        "p50_probability": 0.50,
        "p90_probability": 0.90,
        "p95_probability": 0.95,
        "p99_probability": 0.99,
        "maximum_definition": "x_(n)",
        "interpolation": False,
    }
    if dict(quantiles) != expected_quantiles or type(quantiles.get("interpolation")) is not bool:
        raise ValueError(f"{name}.quantile_estimator is not the locked nearest-rank rule")

    sources = value["source_tensor_sha256"]
    if not isinstance(sources, Mapping) or set(sources) != set(_DIGEST_NAMES):
        raise ValueError(f"{name}.source_tensor_sha256 has an invalid field set")
    expected_sources = {
        "replicate_count_sha256": _digest(
            replicate_count_sha256,
            name=f"{name}.expected.replicate_count_sha256",
        ),
        "replicate_h_sha256": _digest(
            replicate_h_sha256,
            name=f"{name}.expected.replicate_h_sha256",
        ),
        "mean_h_sha256": _digest(
            mean_h_sha256,
            name=f"{name}.expected.mean_h_sha256",
        ),
    }
    for field, expected in expected_sources.items():
        if _digest(sources[field], name=f"{name}.source_tensor_sha256.{field}") != expected:
            raise ValueError(f"{name} does not bind label_stream.{field}")

    num_edges = _positive_integer(expected_num_edges, name="expected_num_edges")
    metrics = value["metrics"]
    if not isinstance(metrics, Mapping) or set(metrics) != set(_METRIC_NAMES):
        raise ValueError(f"{name}.metrics must contain exactly {list(_METRIC_NAMES)!r}")
    normalized_metrics: dict[str, dict[str, int | float]] = {}
    for metric_name in _METRIC_NAMES:
        metric = metrics[metric_name]
        if not isinstance(metric, Mapping):
            raise TypeError(f"{name}.metrics.{metric_name} must be a mapping")
        metric_keys = {"sample_size", "p50", "p90", "p95", "p99", "max"}
        if set(metric) != metric_keys:
            raise ValueError(f"{name}.metrics.{metric_name} has an invalid field set")
        expected_size = 4 * num_edges if metric_name != "abs_mean_h" else num_edges
        if (
            _positive_integer(
                metric["sample_size"],
                name=f"{name}.metrics.{metric_name}.sample_size",
            )
            != expected_size
        ):
            raise ValueError(f"{name}.metrics.{metric_name}.sample_size is inconsistent")
        normalized: dict[str, int | float] = {"sample_size": expected_size}
        previous = -math.inf
        for statistic in ("p50", "p90", "p95", "p99", "max"):
            raw = metric[statistic]
            if metric_name == "replicate_count":
                scalar: int | float = _positive_integer(
                    raw,
                    name=f"{name}.metrics.{metric_name}.{statistic}",
                )
            else:
                scalar = _nonnegative_finite(
                    raw,
                    name=f"{name}.metrics.{metric_name}.{statistic}",
                )
            if float(scalar) < previous:
                raise ValueError(f"{name}.metrics.{metric_name} order statistics decrease")
            normalized[statistic] = scalar
            previous = float(scalar)
        normalized_metrics[metric_name] = normalized

    recorded_sha = _digest(value["diagnostics_sha256"], name=f"{name}.diagnostics_sha256")
    payload = {
        "schema_version": value["schema_version"],
        "split": value["split"],
        "gamma": value["gamma"],
        "num_replicates": value["num_replicates"],
        "quantile_estimator": dict(quantiles),
        "source_tensor_sha256": dict(sources),
        "metrics": normalized_metrics,
        "scalar_only": value["scalar_only"],
        "descriptive_only": value["descriptive_only"],
        "used_for_clipping": value["used_for_clipping"],
        "used_for_selection": value["used_for_selection"],
        "used_for_gating": value["used_for_gating"],
    }
    if _canonical_sha256(payload) != recorded_sha:
        raise ValueError(f"{name}.diagnostics_sha256 does not bind its canonical payload")
    return {**payload, "diagnostics_sha256": recorded_sha}


__all__ = [
    "NEAREST_RANK_QUANTILES",
    "REPEATED_LABEL_TAIL_DIAGNOSTICS_SCHEMA",
    "build_repeated_label_tail_diagnostics",
    "nearest_rank_summary",
    "validate_repeated_label_tail_diagnostics",
]
