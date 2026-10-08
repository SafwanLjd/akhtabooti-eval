"""Detector contract. A detector reads plain text and reports PII it found."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar, Self

from ..dataset import Span


@dataclass
class Detection:
    """Findings for one input text.

    ``spans`` carry offsets into the input text and, for models, a score.
    Detectors that cannot give offsets report ``values`` as (label, string)
    pairs. ``labels_present`` holds categories a detector flags for the whole
    text without locating them, such as the Core's identity-document rules.
    ``raw_labels`` counts the detector's own labels before mapping;
    ``unmapped`` counts the ones dropped as out of scope.
    """

    spans: list[Span] = field(default_factory=list)
    values: list[tuple[str, str]] | None = None
    labels_present: set[str] = field(default_factory=set)
    raw_labels: Counter[str] = field(default_factory=Counter)
    unmapped: Counter[str] = field(default_factory=Counter)

    def with_threshold(self, threshold: float) -> Detection:
        """Drop scored spans below ``threshold``. Unscored spans stay."""
        kept = [s for s in self.spans if s.score is None or s.score >= threshold]
        return replace(self, spans=kept)

    def predicted_values(self, text: str) -> list[tuple[str, str]]:
        if self.values is not None:
            return list(self.values)
        return [(s.label, s.text or text[s.start : s.end]) for s in self.spans]

    def predicted_labels(self) -> set[str]:
        labels = {s.label for s in self.spans} | set(self.labels_present)
        labels |= {label for label, _ in self.values or []}
        return labels

    def to_dict(self) -> dict[str, Any]:
        return {
            "spans": [s.to_dict() for s in self.spans],
            "values": None if self.values is None else [list(v) for v in self.values],
            "labels_present": sorted(self.labels_present),
            "raw_labels": dict(self.raw_labels),
            "unmapped": dict(self.unmapped),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Detection:
        values = d.get("values")
        return cls(
            spans=[Span.from_dict(s) for s in d.get("spans", [])],
            values=None if values is None else [(str(a), str(b)) for a, b in values],
            labels_present=set(d.get("labels_present", [])),
            raw_labels=Counter(d.get("raw_labels", {})),
            unmapped=Counter(d.get("unmapped", {})),
        )


DetectorResolver = Callable[[str], "Detector"]


class Detector(ABC):
    """Base class for every detector under evaluation.

    Subclasses are built from a config table by ``from_config`` and must not
    load models in ``__init__``: the runner times ``prepare`` separately as
    initialisation cost.
    """

    kind: ClassVar[str]

    @classmethod
    def from_config(cls, params: Mapping[str, Any], resolve: DetectorResolver) -> Self:
        """Build from a config table. ``resolve`` returns other named detectors."""
        del resolve
        return cls(**params)

    def prepare(self) -> None:  # noqa: B027 - optional hook
        """Load models or start helper processes."""

    def close(self) -> None:  # noqa: B027 - optional hook
        """Stop helper processes and wait for them. Calling it twice is harmless; ``prepare`` starts them again."""

    @abstractmethod
    def info(self) -> dict[str, Any]:
        """Everything needed to reproduce this detector: versions, checksums, settings."""

    @abstractmethod
    def detect(self, text: str) -> Detection: ...
