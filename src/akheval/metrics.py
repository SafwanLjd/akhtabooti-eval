"""Confusion counts, precision, recall, F-scores and bootstrap intervals over documents."""

from __future__ import annotations

import functools
import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 20261011
CONFIDENCE = 0.95
# Decimals stored for a rate in summary.json; reports print one decimal of a percent, so 6 avoids double rounding.
DIGITS = 6
PACK_BITS = 32
PACK_MASK = (1 << PACK_BITS) - 1


@dataclass(frozen=True, slots=True)
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def __add__(self, other: Counts) -> Counts:
        return Counts(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn)

    @property
    def support(self) -> int:
        return self.tp + self.fn

    @property
    def predicted(self) -> int:
        return self.tp + self.fp

    @property
    def precision(self) -> float | None:
        return self.tp / self.predicted if self.predicted else None

    @property
    def recall(self) -> float | None:
        return self.tp / self.support if self.support else None

    def fbeta(self, beta: float) -> float | None:
        """F-beta from counts; None only when there is nothing to score (tp = fp = fn = 0)."""
        b2 = beta * beta
        denominator = (1 + b2) * self.tp + b2 * self.fn + self.fp
        return (1 + b2) * self.tp / denominator if denominator else None

    @property
    def f1(self) -> float | None:
        return self.fbeta(1.0)

    @property
    def f2(self) -> float | None:
        return self.fbeta(2.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "support": self.support,
            "predicted": self.predicted,
            "precision": _round(self.precision),
            "recall": _round(self.recall),
            "f1": _round(self.f1),
            "f2": _round(self.f2),
        }


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, DIGITS)


TP = Counts(tp=1)
FP = Counts(fp=1)
FN = Counts(fn=1)


def total(counts: Iterable[Counts]) -> Counts:
    return sum(counts, Counts())


def tally(table: dict[str, Counts], label: str, counts: Counts) -> None:
    """Add ``counts`` to ``table[label]``."""
    table[label] = table.get(label, Counts()) + counts


@dataclass(frozen=True, slots=True)
class Interval:
    low: float
    high: float

    def excludes_zero(self) -> bool:
        return self.low > 0 or self.high < 0

    def to_dict(self) -> dict[str, float]:
        return {"low": round(self.low, DIGITS), "high": round(self.high, DIGITS)}


Metric = Callable[[Counts], float | None]


def precision_of(counts: Counts) -> float | None:
    return counts.precision


def recall_of(counts: Counts) -> float | None:
    return counts.recall


def f1_of(counts: Counts) -> float | None:
    return counts.f1


def f2_of(counts: Counts) -> float | None:
    return counts.f2


PRECISION = "precision"
RECALL = "recall"
F1 = "f1"
F2 = "f2"
METRICS: dict[str, Metric] = {PRECISION: precision_of, RECALL: recall_of, F1: f1_of, F2: f2_of}


@dataclass(frozen=True, slots=True)
class Delta:
    """Candidate minus baseline, observed on the full set, with its paired-bootstrap interval."""

    value: float | None
    interval: Interval | None


@functools.lru_cache(maxsize=2)
def _resamples(n: int, resamples: int, seed: int) -> tuple[tuple[int, ...], ...]:
    rng = random.Random(seed)
    population = range(n)
    return tuple(tuple(rng.choices(population, k=n)) for _ in range(resamples))


class _Packed:
    """Per-item counts packed into one integer each, so a resample needs one sum instead of three.

    Each count takes a 32-bit field; a resample's totals stay far below 2**32.
    """

    def __init__(self, items: Sequence[Counts]) -> None:
        self.values = [c.tp | c.fp << PACK_BITS | c.fn << 2 * PACK_BITS for c in items]

    def resampled(self, indices: Sequence[int]) -> Counts:
        packed = sum(map(self.values.__getitem__, indices))
        return Counts(packed & PACK_MASK, packed >> PACK_BITS & PACK_MASK, packed >> 2 * PACK_BITS)


def _interval(samples: list[float]) -> Interval | None:
    if not samples:
        return None
    samples.sort()
    tail = (1 - CONFIDENCE) / 2
    last = len(samples) - 1
    return Interval(samples[round(tail * last)], samples[round((1 - tail) * last)])


def bootstrap(
    items: Sequence[Counts],
    metrics: Mapping[str, Metric] = METRICS,
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Interval | None]:
    """Percentile interval of each micro-averaged metric, resampling items with replacement."""
    if len(items) < 2:
        return dict.fromkeys(metrics)
    packed = _Packed(items)
    samples: dict[str, list[float]] = {name: [] for name in metrics}
    for indices in _resamples(len(items), resamples, seed):
        counts = packed.resampled(indices)
        for name, metric in metrics.items():
            if (value := metric(counts)) is not None:
                samples[name].append(value)
    return {name: _interval(values) for name, values in samples.items()}


def _delta(base: Counts, candidate: Counts, metric: Metric) -> float | None:
    b, c = metric(base), metric(candidate)
    return None if b is None or c is None else c - b


def paired_bootstrap(
    base: Sequence[Counts],
    candidate: Sequence[Counts],
    metrics: Mapping[str, Metric] = METRICS,
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Delta]:
    """Difference of each metric between two runs scored on the same items in the same order."""
    if len(base) != len(candidate):
        raise ValueError(f"paired bootstrap needs aligned items; got {len(base)} and {len(candidate)}")
    observed = {name: _delta(total(base), total(candidate), metric) for name, metric in metrics.items()}
    if len(base) < 2:
        return {name: Delta(value, None) for name, value in observed.items()}
    base_packed, candidate_packed = _Packed(base), _Packed(candidate)
    samples: dict[str, list[float]] = {name: [] for name in metrics}
    for indices in _resamples(len(base), resamples, seed):
        b, c = base_packed.resampled(indices), candidate_packed.resampled(indices)
        for name, metric in metrics.items():
            if (d := _delta(b, c, metric)) is not None:
                samples[name].append(d)
    return {name: Delta(observed[name], _interval(samples[name])) for name in metrics}
