"""akhtabooti-core text rules, run through the shim in ``rust/core-rules``.

The shim compiles the exact ``text_utils.rs`` of a pinned core commit and reads
JSON lines on stdin. Core API v1 (the August keyword engine) reports keyword
categories by name only; API v2 (the rule-branch engine) reports evidence
strings for every category.

The desktop app shows the user e-mail and phone values (``FilePIIs.email_accounts``
and ``phone_numbers``) and every other category as a name for the whole file
(``other_piis``). ``located_categories`` mirrors that: a Core category in it
yields values and spans, every other category only flags the document. Two
categories can share a taxonomy label (a number rule and a keyword rule for
refugee documents), so the choice goes by category name. The Core
returns matched strings without offsets, so the adapter finds every occurrence
of each located string in the text to rebuild spans.
"""

from __future__ import annotations

import contextlib
import json
import re
import subprocess
import weakref
from collections.abc import Collection, Iterable, Mapping, Sequence
from pathlib import Path
from typing import IO, Any

from ..dataset import Span, sha256_file
from ..taxonomy import CORE_LABELS, map_label
from .base import Detection, Detector

BIN_DIR = Path("rust/core-rules/bin")
STOP_TIMEOUT_SECONDS = 10
APP_LOCATED_CATEGORIES = ("Email", "Phone Number")
"""Core categories whose values the desktop app shows; the app names every other category without a value."""


def locate(text: str, needle: str) -> list[tuple[int, int]]:
    """Every non-overlapping occurrence of ``needle``; falls back to any whitespace between its words."""
    found = []
    start = text.find(needle) if needle else -1
    while start >= 0:
        found.append((start, start + len(needle)))
        start = text.find(needle, start + len(needle))
    if found or not needle.strip():
        return found
    pattern = r"\s+".join(re.escape(word) for word in needle.split())
    return [m.span() for m in re.finditer(pattern, text)]


def drop_nested(spans: Iterable[Span]) -> list[Span]:
    """Remove duplicates and spans that lie inside a longer span of the same label."""
    kept: list[Span] = []
    reach: dict[str, int] = {}
    for span in sorted(spans, key=lambda s: (s.start, -s.end, s.label)):
        if span.end > reach.get(span.label, -1):
            kept.append(span)
            reach[span.label] = span.end
    return kept


def to_detection(
    text: str, record: Mapping[str, Any], located_categories: Collection[str] = APP_LOCATED_CATEGORIES
) -> Detection:
    """Turn one shim output line into a Detection over ``text``; only ``located_categories`` yield values and spans."""
    detection = Detection()
    values: list[tuple[str, str]] = []
    spans: list[Span] = []
    for category, evidence in sorted(record.get("detections", {}).items()):
        label = _count(detection, category, len(evidence))
        if label is None or category not in located_categories:
            continue
        for found in evidence:
            values.append((label, found))
            spans += [Span(s, e, label, text[s:e]) for s, e in locate(text, found)]
    for category in record.get("keyword_categories", []):
        _count(detection, category, 1)
    detection.values = values
    detection.spans = drop_nested(spans)
    return detection


def _count(detection: Detection, category: str, hits: int) -> str | None:
    """Record a Core category; returns its taxonomy label or None when out of scope."""
    detection.raw_labels[category] += hits
    label = map_label(CORE_LABELS, category)
    if label is None:
        detection.unmapped[category] += hits
    else:
        detection.labels_present.add(label)
    return label


def _stop(process: subprocess.Popen[str]) -> None:
    if process.stdin is not None:
        with contextlib.suppress(BrokenPipeError):  # a dead shim leaves unflushed input behind
            process.stdin.close()
    try:
        process.wait(timeout=STOP_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    for stream in (process.stdout, process.stderr):
        if stream is not None:
            stream.close()


class CoreRulesDetector(Detector):
    """Core text rules of one commit, optionally with another rule file (API v2 only)."""

    kind = "core-rules"

    def __init__(
        self,
        core_sha: str,
        definitions: str | None = None,
        binary: str | None = None,
        located_categories: Sequence[str] = APP_LOCATED_CATEGORIES,
    ) -> None:
        unknown = sorted(c for c in located_categories if map_label(CORE_LABELS, c) is None)
        if unknown:
            raise ValueError(f"located_categories {unknown} map to no taxonomy label")
        self.core_sha = core_sha
        self.located_categories = frozenset(located_categories)
        self.definitions = Path(definitions) if definitions else None
        self.binary = Path(binary) if binary else BIN_DIR / f"akh-core-rules-{core_sha}"
        self._version: dict[str, Any] | None = None
        self._process: subprocess.Popen[str] | None = None

    def _read_version(self) -> dict[str, Any]:
        if self._version is None:
            if not self.binary.is_file():
                raise FileNotFoundError(
                    f"shim binary {self.binary} not found; build it with rust/core-rules/build.sh {self.core_sha}"
                )
            out = subprocess.run([str(self.binary), "--version"], check=True, capture_output=True, text=True)
            self._version = json.loads(out.stdout)
            if self.definitions is not None and not self._version.get("accepts_definitions"):
                raise ValueError(f"core {self.core_sha} uses API {self._version['core_api']} with built-in rules")
        return self._version

    def prepare(self) -> None:
        self._read_version()
        if self._process is not None and self._process.poll() is None:
            return
        self.close()
        command = [str(self.binary)]
        if self.definitions is not None:
            if not self.definitions.is_file():
                raise FileNotFoundError(f"rule file {self.definitions} not found")
            command += ["--definitions", str(self.definitions)]
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        weakref.finalize(self, _stop, self._process)

    def close(self) -> None:
        """Stop the shim process."""
        if self._process is not None:
            _stop(self._process)
            self._process = None

    def info(self) -> dict[str, Any]:
        version = self._read_version()
        definitions = self.definitions or Path(version["definitions"])
        return {
            "kind": self.kind,
            "core_sha": self.core_sha,
            "core_commit": version["core_commit"],
            "core_api": version["core_api"],
            "located_categories": sorted(self.located_categories),
            "definitions": str(definitions),
            "definitions_sha256": sha256_file(definitions),
            "binary": str(self.binary),
            "binary_sha256": sha256_file(self.binary),
            "shim_version": version["shim_version"],
        }

    def _exchange(self, stdin: IO[str], stdout: IO[str], text: str) -> dict[str, Any]:
        stdin.write(json.dumps({"id": "0", "text": text}, ensure_ascii=False) + "\n")
        stdin.flush()
        line = stdout.readline()
        if not line:
            raise RuntimeError(f"core-rules shim exited: {self._stderr()}")
        result: dict[str, Any] = json.loads(line)
        return result

    def _stderr(self) -> str:
        process = self._process
        if process is None or process.stderr is None:
            return "no process"
        process.wait()
        return process.stderr.read().strip() or f"exit code {process.returncode}"

    def detect(self, text: str) -> Detection:
        self.prepare()
        process = self._process
        if process is None or process.stdin is None or process.stdout is None:
            raise RuntimeError("core-rules shim did not start")
        try:
            return to_detection(text, self._exchange(process.stdin, process.stdout, text), self.located_categories)
        except BrokenPipeError as exc:
            raise RuntimeError(f"core-rules shim exited: {self._stderr()}") from exc
