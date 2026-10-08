"""Two standard-library regexes, email and Jordanian mobile: a sanity floor and a test double."""

from __future__ import annotations

import re
from typing import Any

from ..dataset import Span
from .base import Detection, Detector

PATTERNS: dict[str, re.Pattern[str]] = {
    "EMAIL": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "PHONE": re.compile(r"(?<![\d+])(?:(?:\+|00)962[ -]?|0)7[789](?:[ -]?\d){7}(?!\d)"),
}


class RegexBaselineDetector(Detector):
    kind = "regex-baseline"

    def info(self) -> dict[str, Any]:
        return {"kind": self.kind, "patterns": {label: p.pattern for label, p in PATTERNS.items()}}

    def detect(self, text: str) -> Detection:
        spans = [Span(m.start(), m.end(), label, m.group()) for label, p in PATTERNS.items() for m in p.finditer(text)]
        detection = Detection(spans=sorted(spans, key=lambda s: (s.start, s.end)))
        detection.raw_labels.update(s.label for s in spans)
        return detection
