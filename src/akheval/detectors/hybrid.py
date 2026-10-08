"""Rules first, models for the rest: the finding-resolution policy of the MVP brief.

The hybrid keeps every span of its first part. A span of a later part is added
when it overlaps no span kept so far. Document labels and reported values are
the union over all parts. An optional per-part label filter routes each label
to the parts trusted with it, before the merge.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from dataclasses import replace
from typing import Any, Self

from ..dataset import Span
from ..taxonomy import LABELS
from .base import Detection, Detector, DetectorResolver

MIN_PARTS = 2


def _span_order(span: Span) -> tuple[int, int, str]:
    return span.start, span.end, span.label


def restrict(detection: Detection, labels: Collection[str]) -> Detection:
    """Keep only the spans, values and document labels in ``labels``; raw label counts stay as reported."""
    return replace(
        detection,
        spans=[s for s in detection.spans if s.label in labels],
        values=None if detection.values is None else [v for v in detection.values if v[0] in labels],
        labels_present={label for label in detection.labels_present if label in labels},
    )


def merge(text: str, detections: Sequence[Detection]) -> Detection:
    """Combine the detections of all parts for one text, first part winning on overlap."""
    first, *rest = detections
    kept = list(first.spans)
    values = first.predicted_values(text)
    for detection in rest:
        added = []
        for span in sorted(detection.spans, key=_span_order):
            if not any(span.overlaps(k) for k in kept):
                kept.append(span)
                added.append(span)
        values += detection.values or []
        values += [(s.label, s.text or text[s.start : s.end]) for s in added]
    has_values = any(d.values is not None for d in detections)
    return Detection(
        spans=sorted(kept, key=_span_order),
        values=list(dict.fromkeys(values)) if has_values else None,
        labels_present=set().union(*(d.labels_present for d in detections)),
        raw_labels=sum((d.raw_labels for d in detections), Counter()),
        unmapped=sum((d.unmapped for d in detections), Counter()),
    )


class HybridDetector(Detector):
    """``only`` holds one label filter per part, aligned with ``parts``; None keeps every label of that part."""

    kind = "hybrid"

    def __init__(self, parts: Sequence[Detector], only: Sequence[Collection[str] | None] | None = None) -> None:
        if len(parts) < MIN_PARTS:
            raise ValueError(f"a hybrid needs at least {MIN_PARTS} parts, got {len(parts)}")
        filters = list(only) if only is not None else [None] * len(parts)
        if len(filters) != len(parts):
            raise ValueError(f"only has {len(filters)} filters for {len(parts)} parts")
        unknown = sorted({label for f in filters if f is not None for label in f} - set(LABELS))
        if unknown:
            raise ValueError(f"only names labels outside the taxonomy: {unknown}")
        self.parts = list(parts)
        self.only = [None if f is None else frozenset(f) for f in filters]

    @classmethod
    def from_config(cls, params: Mapping[str, Any], resolve: DetectorResolver) -> Self:
        """``parts`` lists detector names; ``only`` maps a part name to the labels kept from that part."""
        options = dict(params)
        names = options.pop("parts", [])
        only: Mapping[str, Sequence[str]] = options.pop("only", {})
        strangers = sorted(set(only) - set(names))
        if strangers:
            raise ValueError(f"only names {strangers}, which are not parts of this hybrid")
        return cls([resolve(name) for name in names], only=[only.get(name) for name in names], **options)

    def prepare(self) -> None:
        for part in self.parts:
            part.prepare()

    def close(self) -> None:
        for part in self.parts:
            part.close()

    def info(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "parts": [part.info() for part in self.parts],
            "only": [None if f is None else sorted(f) for f in self.only],
        }

    def detect(self, text: str) -> Detection:
        outputs = [part.detect(text) for part in self.parts]
        kept = [d if labels is None else restrict(d, labels) for d, labels in zip(outputs, self.only, strict=True)]
        return merge(text, kept)
