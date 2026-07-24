"""Deterministic paired-seed aggregation for ProRM+/BT experiments.

Every reported effect is defined as ``ProRM+ - BT`` on the *same* seed.  Metric
direction is metadata, not a hypothesis test:

* local regret and error metrics are lower-is-better, so a negative paired
  difference favors ProRM+;
* cosine and improvement metrics are higher-is-better, so a positive paired
  difference favors ProRM+.

The module intentionally reports no p-value and no ``significant`` boolean.
Its percentile interval is descriptive uncertainty over paired seeds.  All
bootstrap randomness comes from an explicitly supplied seed or
``torch.Generator``; the process-global PyTorch random stream is never used.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from enum import Enum
from numbers import Real
from pathlib import Path

import torch

from .contracts import PAIRED_AGGREGATE_SCHEMA_V2

PAIRED_AGGREGATE_SCHEMA_VERSION = PAIRED_AGGREGATE_SCHEMA_V2
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
_MAX_TORCH_SEED = 2**63 - 1


class MetricDirection(str, Enum):
    """Whether a larger or smaller scalar metric is preferable."""

    LOWER_IS_BETTER = "lower_is_better"
    HIGHER_IS_BETTER = "higher_is_better"

    @property
    def favorable_prorm_plus_minus_bt_sign(self) -> str:
        """Return the ProRM+-minus-BT sign that favors ProRM+, without inference."""

        if self is MetricDirection.LOWER_IS_BETTER:
            return "negative"
        return "positive"

    @property
    def favorable_srm_minus_bt_sign(self) -> str:
        """Compatibility alias for the canonical ProRM+ sign property."""

        return self.favorable_prorm_plus_minus_bt_sign


def infer_metric_direction(metric: str) -> MetricDirection:
    """Return the locked direction for a conventionally named metric.

    Names ending in ``local_regret`` or ``error`` are lower-is-better.  Names
    ending in ``cosine`` or ``improvement`` are higher-is-better.  Unknown
    names are rejected so a new metric cannot silently receive the wrong
    interpretation; callers must provide an explicit direction override.
    """

    if not isinstance(metric, str) or not metric.strip():
        raise ValueError("metric names must be non-empty strings")
    normalized = metric.strip().lower()
    if normalized.endswith("local_regret") or normalized.endswith("error"):
        return MetricDirection.LOWER_IS_BETTER
    if normalized.endswith("cosine") or normalized.endswith("improvement"):
        return MetricDirection.HIGHER_IS_BETTER
    raise ValueError(f"metric {metric!r} has no locked direction; provide it explicitly")


@dataclass(frozen=True, slots=True)
class PercentileConfidenceInterval:
    """A two-sided paired-bootstrap percentile confidence interval."""

    lower: float
    upper: float
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL
    method: str = "paired_seed_percentile_bootstrap"

    def __post_init__(self) -> None:
        lower = _finite_float(self.lower, "lower")
        upper = _finite_float(self.upper, "upper")
        confidence = _confidence_level(self.confidence_level)
        if lower > upper:
            raise ValueError("confidence interval lower bound must not exceed upper bound")
        if self.method != "paired_seed_percentile_bootstrap":
            raise ValueError("unsupported confidence interval method")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "confidence_level", confidence)

    def to_dict(self) -> dict[str, object]:
        return {
            "lower": self.lower,
            "upper": self.upper,
            "confidence_level": self.confidence_level,
            "method": self.method,
        }


@dataclass(frozen=True, slots=True)
class PerSeedPairedMetric:
    """One auditable BT/ProRM+ pair and its oriented numerical difference."""

    seed: int
    bt: float
    prorm_plus: float
    prorm_plus_minus_bt: float

    def __post_init__(self) -> None:
        _validate_seed(self.seed, name="seed")
        bt = _finite_float(self.bt, "bt")
        prorm_plus = _finite_float(self.prorm_plus, "prorm_plus")
        difference = _finite_float(self.prorm_plus_minus_bt, "prorm_plus_minus_bt")
        expected = prorm_plus - bt
        if difference != expected:
            raise ValueError("prorm_plus_minus_bt must equal prorm_plus - bt exactly")
        object.__setattr__(self, "bt", bt)
        object.__setattr__(self, "prorm_plus", prorm_plus)
        object.__setattr__(self, "prorm_plus_minus_bt", difference)

    @property
    def srm(self) -> float:
        """Compatibility view of :attr:`prorm_plus`."""

        return self.prorm_plus

    @property
    def srm_minus_bt(self) -> float:
        """Compatibility view of :attr:`prorm_plus_minus_bt`."""

        return self.prorm_plus_minus_bt

    def to_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "bt": self.bt,
            "prorm_plus": self.prorm_plus,
            "prorm_plus_minus_bt": self.prorm_plus_minus_bt,
        }


@dataclass(frozen=True, slots=True)
class PairedMetricSummary:
    """Paired descriptive statistics for one named scalar metric."""

    metric: str
    direction: MetricDirection
    per_seed: tuple[PerSeedPairedMetric, ...]
    paired_mean: float
    sample_std: float
    standard_error: float
    bootstrap_ci: PercentileConfidenceInterval

    def __post_init__(self) -> None:
        if not isinstance(self.metric, str) or not self.metric.strip():
            raise ValueError("metric must be a non-empty string")
        if not isinstance(self.direction, MetricDirection):
            raise TypeError("direction must be a MetricDirection")
        if not isinstance(self.per_seed, tuple) or len(self.per_seed) < 2:
            raise ValueError("per_seed must contain at least two paired seeds")
        if not all(isinstance(item, PerSeedPairedMetric) for item in self.per_seed):
            raise TypeError("per_seed must contain PerSeedPairedMetric values")
        seeds = tuple(item.seed for item in self.per_seed)
        if seeds != tuple(sorted(seeds)) or len(set(seeds)) != len(seeds):
            raise ValueError("per_seed entries must have unique seeds in sorted order")
        for name in ("paired_mean", "sample_std", "standard_error"):
            value = _finite_float(getattr(self, name), name)
            if name != "paired_mean" and value < 0.0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        if not isinstance(self.bootstrap_ci, PercentileConfidenceInterval):
            raise TypeError("bootstrap_ci must be a PercentileConfidenceInterval")

    @property
    def favorable_prorm_plus_minus_bt_sign(self) -> str:
        """Document which sign favors ProRM+; this is not a significance claim."""

        return self.direction.favorable_prorm_plus_minus_bt_sign

    @property
    def favorable_srm_minus_bt_sign(self) -> str:
        """Compatibility alias for the canonical ProRM+ sign property."""

        return self.favorable_prorm_plus_minus_bt_sign

    def to_dict(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "direction": self.direction.value,
            "favorable_prorm_plus_minus_bt_sign": (self.favorable_prorm_plus_minus_bt_sign),
            "num_seeds": len(self.per_seed),
            "per_seed": [item.to_dict() for item in self.per_seed],
            "paired_mean": self.paired_mean,
            "sample_std": self.sample_std,
            "standard_error": self.standard_error,
            "bootstrap_ci": self.bootstrap_ci.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class PairedMetricsAggregate:
    """JSON-compatible aggregate of all same-name paired scalar metrics."""

    seeds: tuple[int, ...]
    metrics: tuple[PairedMetricSummary, ...]
    bootstrap_resamples: int
    bootstrap_seed: int | None
    schema_version: str = PAIRED_AGGREGATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.seeds, tuple) or len(self.seeds) < 2:
            raise ValueError("seeds must contain at least two entries")
        for seed in self.seeds:
            _validate_seed(seed, name="seed")
        if self.seeds != tuple(sorted(self.seeds)) or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique and sorted")
        if not isinstance(self.metrics, tuple) or not self.metrics:
            raise ValueError("metrics must be a non-empty tuple")
        if not all(isinstance(item, PairedMetricSummary) for item in self.metrics):
            raise TypeError("metrics must contain PairedMetricSummary values")
        names = tuple(item.metric for item in self.metrics)
        if names != tuple(sorted(names)) or len(set(names)) != len(names):
            raise ValueError("metric summaries must have unique names in sorted order")
        if any(
            tuple(item.seed for item in metric.per_seed) != self.seeds for metric in self.metrics
        ):
            raise ValueError("every metric summary must use the aggregate seed set")
        _positive_integer(self.bootstrap_resamples, "bootstrap_resamples")
        if self.bootstrap_seed is not None:
            _validate_seed(self.bootstrap_seed, name="bootstrap_seed")
        if self.schema_version != PAIRED_AGGREGATE_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {PAIRED_AGGREGATE_SCHEMA_VERSION!r}")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "seeds": list(self.seeds),
            "num_seeds": len(self.seeds),
            "bootstrap": {
                "resamples": self.bootstrap_resamples,
                "seed": self.bootstrap_seed,
                "method": "paired_seed_percentile_bootstrap",
            },
            "metrics": {item.metric: item.to_dict() for item in self.metrics},
        }


FiveSeedAggregate = PairedMetricsAggregate


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_seed(seed: object, *, name: str) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"{name} must be an integer")
    if not 0 <= seed <= _MAX_TORCH_SEED:
        raise ValueError(f"{name} must lie in [0, {_MAX_TORCH_SEED}]")
    return seed


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _confidence_level(value: object) -> float:
    result = _finite_float(value, "confidence_level")
    if not 0.0 < result < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")
    return result


def _resolve_generator(
    *,
    bootstrap_seed: int | None,
    generator: torch.Generator | None,
) -> torch.Generator:
    if (bootstrap_seed is None) == (generator is None):
        raise ValueError("provide exactly one of bootstrap_seed or generator")
    if generator is not None:
        if not isinstance(generator, torch.Generator):
            raise TypeError("generator must be a torch.Generator")
        if str(generator.device) != "cpu":
            raise ValueError("paired bootstrap requires a CPU torch.Generator")
        return generator
    checked_seed = _validate_seed(bootstrap_seed, name="bootstrap_seed")
    return torch.Generator(device="cpu").manual_seed(checked_seed)


def _metric_bootstrap_seed(base_seed: int, metric: str) -> int:
    payload = f"paired-seed-aggregate/v1\0{base_seed}\0{metric}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & _MAX_TORCH_SEED


def paired_bootstrap_ci(
    paired_differences: Sequence[Real],
    *,
    bootstrap_seed: int | None = None,
    generator: torch.Generator | None = None,
    num_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> PercentileConfidenceInterval:
    """Bootstrap the mean of already paired ``ProRM+ - BT`` differences."""

    if isinstance(paired_differences, (str, bytes)) or not isinstance(paired_differences, Sequence):
        raise TypeError("paired_differences must be a sequence of real scalars")
    values = [
        _finite_float(value, f"paired_differences[{index}]")
        for index, value in enumerate(paired_differences)
    ]
    if len(values) < 2:
        raise ValueError("paired bootstrap requires at least two paired differences")
    resamples = _positive_integer(num_resamples, "num_resamples")
    confidence = _confidence_level(confidence_level)
    random = _resolve_generator(bootstrap_seed=bootstrap_seed, generator=generator)

    differences = torch.tensor(values, dtype=torch.float64)
    indices = torch.randint(
        len(values),
        (resamples, len(values)),
        generator=random,
        device="cpu",
    )
    bootstrap_means = differences[indices].mean(dim=1)
    tail = 0.5 * (1.0 - confidence)
    quantiles = torch.quantile(
        bootstrap_means,
        torch.tensor([tail, 1.0 - tail], dtype=torch.float64),
        interpolation="linear",
    )
    return PercentileConfidenceInterval(
        lower=float(quantiles[0].item()),
        upper=float(quantiles[1].item()),
        confidence_level=confidence,
    )


def _coerce_direction(value: MetricDirection | str, metric: str) -> MetricDirection:
    try:
        direction = value if isinstance(value, MetricDirection) else MetricDirection(value)
    except (TypeError, ValueError) as error:
        choices = [item.value for item in MetricDirection]
        raise ValueError(f"direction for {metric!r} must be one of {choices!r}") from error
    return direction


def _resolve_direction(
    metric: str,
    directions: Mapping[str, MetricDirection | str] | None,
) -> MetricDirection:
    if directions is not None and metric in directions:
        supplied = _coerce_direction(directions[metric], metric)
        try:
            locked = infer_metric_direction(metric)
        except ValueError:
            return supplied
        if supplied is not locked:
            raise ValueError(f"direction for canonical metric {metric!r} must be {locked.value!r}")
        return supplied
    return infer_metric_direction(metric)


def _validate_result_tables(
    bt_by_seed: Mapping[int, Mapping[str, Real]],
    prorm_plus_by_seed: Mapping[int, Mapping[str, Real]],
) -> tuple[
    tuple[int, ...],
    tuple[str, ...],
    dict[int, dict[str, float]],
    dict[int, dict[str, float]],
]:
    for name, value in (
        ("bt_by_seed", bt_by_seed),
        ("prorm_plus_by_seed", prorm_plus_by_seed),
    ):
        if not isinstance(value, Mapping):
            raise TypeError(f"{name} must map seeds to metric mappings")
    bt_seeds = set(bt_by_seed)
    prorm_plus_seeds = set(prorm_plus_by_seed)
    for seed in bt_seeds | prorm_plus_seeds:
        _validate_seed(seed, name="seed")
    if bt_seeds != prorm_plus_seeds:
        raise ValueError(
            "BT and ProRM+ seed sets must match exactly; "
            f"BT-only={sorted(bt_seeds - prorm_plus_seeds)!r}, "
            f"ProRM+-only={sorted(prorm_plus_seeds - bt_seeds)!r}"
        )
    if len(bt_seeds) < 2:
        raise ValueError("paired aggregation requires at least two shared seeds")
    seeds = tuple(sorted(bt_seeds))

    normalized: dict[str, dict[int, dict[str, float]]] = {"bt": {}, "prorm_plus": {}}
    expected_metrics: set[str] | None = None
    for learner, table in (("bt", bt_by_seed), ("prorm_plus", prorm_plus_by_seed)):
        for seed in seeds:
            row = table[seed]
            if not isinstance(row, Mapping):
                raise TypeError(f"{learner} metrics for seed {seed} must be a mapping")
            if not row:
                raise ValueError(f"{learner} metrics for seed {seed} must not be empty")
            metric_names = set(row)
            if any(not isinstance(metric, str) or not metric.strip() for metric in metric_names):
                raise ValueError("metric names must be non-empty strings")
            if expected_metrics is None:
                expected_metrics = metric_names
            elif metric_names != expected_metrics:
                raise ValueError(
                    "every BT/ProRM+ seed row must contain exactly the same metric names"
                )
            normalized[learner][seed] = {
                metric: _finite_float(value, f"{learner}[{seed}][{metric!r}]")
                for metric, value in row.items()
            }
    assert expected_metrics is not None
    return (
        seeds,
        tuple(sorted(expected_metrics)),
        normalized["bt"],
        normalized["prorm_plus"],
    )


def aggregate_paired_metrics(
    bt_by_seed: Mapping[int, Mapping[str, Real]],
    prorm_plus_by_seed: Mapping[int, Mapping[str, Real]],
    *,
    directions: Mapping[str, MetricDirection | str] | None = None,
    bootstrap_seed: int | None = None,
    generator: torch.Generator | None = None,
    num_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> PairedMetricsAggregate:
    """Aggregate same-name BT/ProRM+ scalar metrics over exactly paired seeds.

    The returned paired mean, sample standard deviation, standard error, and
    percentile interval are all computed from per-seed ``ProRM+ - BT`` values.
    No result is labelled statistically significant.
    """

    seeds, metric_names, bt, prorm_plus = _validate_result_tables(bt_by_seed, prorm_plus_by_seed)
    resamples = _positive_integer(num_resamples, "num_resamples")
    confidence = _confidence_level(confidence_level)
    if directions is not None:
        if not isinstance(directions, Mapping):
            raise TypeError("directions must be a metric-to-direction mapping")
        unknown_directions = set(directions) - set(metric_names)
        if unknown_directions:
            raise ValueError(
                "directions contains metrics absent from the results: "
                f"{sorted(unknown_directions)!r}"
            )
    shared_generator = _resolve_generator(
        bootstrap_seed=bootstrap_seed,
        generator=generator,
    )

    summaries: list[PairedMetricSummary] = []
    for metric in metric_names:
        per_seed = tuple(
            PerSeedPairedMetric(
                seed=seed,
                bt=bt[seed][metric],
                prorm_plus=prorm_plus[seed][metric],
                prorm_plus_minus_bt=prorm_plus[seed][metric] - bt[seed][metric],
            )
            for seed in seeds
        )
        differences = torch.tensor(
            [item.prorm_plus_minus_bt for item in per_seed],
            dtype=torch.float64,
        )
        sample_std = float(torch.std(differences, correction=1).item())
        standard_error = sample_std / math.sqrt(len(seeds))
        if bootstrap_seed is None:
            ci = paired_bootstrap_ci(
                differences.tolist(),
                generator=shared_generator,
                num_resamples=resamples,
                confidence_level=confidence,
            )
        else:
            metric_seed = _metric_bootstrap_seed(bootstrap_seed, metric)
            ci = paired_bootstrap_ci(
                differences.tolist(),
                bootstrap_seed=metric_seed,
                num_resamples=resamples,
                confidence_level=confidence,
            )
        summaries.append(
            PairedMetricSummary(
                metric=metric,
                direction=_resolve_direction(metric, directions),
                per_seed=per_seed,
                paired_mean=float(differences.mean().item()),
                sample_std=sample_std,
                standard_error=standard_error,
                bootstrap_ci=ci,
            )
        )
    return PairedMetricsAggregate(
        seeds=seeds,
        metrics=tuple(summaries),
        bootstrap_resamples=resamples,
        bootstrap_seed=bootstrap_seed,
    )


aggregate_five_seed_results = aggregate_paired_metrics


def atomic_write_aggregate_json(
    path: str | os.PathLike[str],
    aggregate: PairedMetricsAggregate,
) -> None:
    """Atomically write a paired aggregate as deterministic UTF-8 JSON."""

    if not isinstance(aggregate, PairedMetricsAggregate):
        raise TypeError("aggregate must be a PairedMetricsAggregate")
    destination = Path(path)
    if not destination.parent.exists():
        raise FileNotFoundError(f"destination directory does not exist: {destination.parent}")

    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary_name = handle.name
            json.dump(
                aggregate.to_dict(),
                handle,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
        temporary_name = None
    finally:
        if temporary_name is not None:
            with suppress(FileNotFoundError):
                os.unlink(temporary_name)


write_paired_aggregate_json = atomic_write_aggregate_json


__all__ = [
    "DEFAULT_BOOTSTRAP_RESAMPLES",
    "DEFAULT_CONFIDENCE_LEVEL",
    "PAIRED_AGGREGATE_SCHEMA_VERSION",
    "FiveSeedAggregate",
    "MetricDirection",
    "PairedMetricSummary",
    "PairedMetricsAggregate",
    "PerSeedPairedMetric",
    "PercentileConfidenceInterval",
    "aggregate_five_seed_results",
    "aggregate_paired_metrics",
    "atomic_write_aggregate_json",
    "infer_metric_direction",
    "paired_bootstrap_ci",
    "write_paired_aggregate_json",
]
