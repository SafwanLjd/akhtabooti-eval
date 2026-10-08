"""Evaluation config: one TOML file holds the project paths, components, pipelines, runs and comparisons.

::

    [project]
    dataset = "datasets/akhtabooti-v1"
    results = "reports/v1/runs"
    tuning = "reports/v1/tuning"

    [detectors.<name>]    kind = "<detector kind>"; other keys are constructor parameters
    [extractors.<name>]   kind = "<extractor kind>"; other keys are constructor parameters
    [pipelines.<name>]    detector = "<detector>", extractors = { default = "<extractor>", pdf = "<extractor>" }

    [[runs]]              name, mode = "text" | "pipeline" | "extract", split = "dev" | "test", and
                          detector (text), pipeline (pipeline) or extractors = {...} (extract);
                          pipeline and extract runs take an optional channel = "native" | "ocr"
    [[comparisons]]       name, baseline = "<run>", candidate = "<run>"

Extractor maps take format keys from ``dataset.FORMATS`` plus a required
``default``. Paths are relative to the directory you run akheval from, which
is the repository root.
"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .dataset import CHANNELS, FORMATS, SPLITS
from .registry import DETECTOR_KINDS, EXTRACTOR_KINDS, Component, ConfigError

TEXT = "text"
PIPELINE = "pipeline"
EXTRACT = "extract"
MODES = (TEXT, PIPELINE, EXTRACT)
DEFAULT_EXTRACTOR = "default"
NAME = re.compile(r"^[a-z0-9][a-z0-9._-]*$")

_PROJECT_KEYS = ("dataset", "results", "tuning")
_TOP_LEVEL = {"project", "detectors", "extractors", "pipelines", "runs", "comparisons"}
_RUN_KEYS = {
    TEXT: {"name", "mode", "split", "detector"},
    PIPELINE: {"name", "mode", "split", "pipeline", "channel"},
    EXTRACT: {"name", "mode", "split", "extractors", "channel"},
}


@dataclass(frozen=True)
class Project:
    dataset: Path
    results: Path
    tuning: Path


@dataclass(frozen=True)
class Pipeline:
    name: str
    detector: str
    extractors: Mapping[str, str]


@dataclass(frozen=True)
class RunSpec:
    name: str
    mode: str
    split: str
    detector: str | None = None
    pipeline: str | None = None
    extractors: Mapping[str, str] = field(default_factory=dict)
    channel: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mode": self.mode,
            "split": self.split,
            "detector": self.detector,
            "pipeline": self.pipeline,
            "extractors": dict(sorted(self.extractors.items())),
            "channel": self.channel,
        }


@dataclass(frozen=True)
class Comparison:
    name: str
    baseline: str
    candidate: str


@dataclass(frozen=True)
class EvalConfig:
    path: Path
    project: Project
    detectors: Mapping[str, Component]
    extractors: Mapping[str, Component]
    pipelines: Mapping[str, Pipeline]
    runs: Mapping[str, RunSpec]
    comparisons: Mapping[str, Comparison]

    def run(self, name: str) -> RunSpec:
        return _lookup(self.runs, name, "run")

    def comparison(self, name: str) -> Comparison:
        return _lookup(self.comparisons, name, "comparison")

    def run_dir(self, name: str) -> Path:
        return self.project.results / name

    def detector_of(self, run: RunSpec) -> str | None:
        """Name of the detector a run uses, or None for extract runs."""
        if run.mode == PIPELINE and run.pipeline is not None:
            return self.pipelines[run.pipeline].detector
        return run.detector

    def extractors_of(self, run: RunSpec) -> Mapping[str, str]:
        """Format key (or "default") to extractor name for a pipeline or extract run."""
        if run.mode == PIPELINE and run.pipeline is not None:
            return self.pipelines[run.pipeline].extractors
        return run.extractors


def extractor_for(extractors: Mapping[str, str], fmt: str) -> str:
    return extractors.get(fmt, extractors[DEFAULT_EXTRACTOR])


def _lookup(table: Mapping[str, Any], name: str, what: str) -> Any:
    if name not in table:
        raise ConfigError(f"unknown {what} {name!r}; defined: {', '.join(table) or 'none'}")
    return table[name]


def load_config(path: Path | str) -> EvalConfig:
    path = Path(path)
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"config file {path} not found") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: invalid TOML: {exc}") from exc
    return parse_config(data, path)


def parse_config(data: Mapping[str, Any], path: Path) -> EvalConfig:
    """Build and validate a config; raises ConfigError listing every problem found."""
    problems: list[str] = [f"unknown top-level table [{key}]" for key in sorted(data.keys() - _TOP_LEVEL)]
    project = _parse_project(data.get("project", {}), problems)
    detectors = _parse_components(data.get("detectors", {}), DETECTOR_KINDS, "detector", problems)
    extractors = _parse_components(data.get("extractors", {}), EXTRACTOR_KINDS, "extractor", problems)
    pipelines = _parse_pipelines(data.get("pipelines", {}), detectors, extractors, problems)
    runs = _parse_runs(data.get("runs", []), detectors, extractors, pipelines, problems)
    comparisons = _parse_comparisons(data.get("comparisons", []), runs, problems)
    if problems:
        raise ConfigError(f"{path}: invalid config:\n  " + "\n  ".join(problems))
    return EvalConfig(path, project, detectors, extractors, pipelines, runs, comparisons)


def _parse_project(table: Mapping[str, Any], problems: list[str]) -> Project:
    paths = {}
    for key in _PROJECT_KEYS:
        value = table.get(key)
        if not isinstance(value, str) or not value:
            problems.append(f"[project] needs a path string for {key!r}")
        paths[key] = Path(value) if isinstance(value, str) else Path()
    problems += [f"[project] has unknown key {key!r}" for key in sorted(table.keys() - set(_PROJECT_KEYS))]
    return Project(**paths)


def _check_name(name: object, what: str, problems: list[str]) -> bool:
    if isinstance(name, str) and NAME.match(name):
        return True
    problems.append(f"{what} name {name!r} must match {NAME.pattern}")
    return False


def _parse_components(
    tables: Mapping[str, Any], kinds: Mapping[str, str], what: str, problems: list[str]
) -> dict[str, Component]:
    components = {}
    for name, table in tables.items():
        if not _check_name(name, what, problems):
            continue
        params = dict(table)
        kind = params.pop("kind", None)
        if kind not in kinds:
            problems.append(f"{what} {name!r}: kind {kind!r} is not one of {', '.join(sorted(kinds))}")
            continue
        components[name] = Component(name, kind, params)
    return components


def _check_extractor_map(
    owner: str, mapping: object, extractors: Mapping[str, Component], problems: list[str]
) -> dict[str, str]:
    if not isinstance(mapping, Mapping) or DEFAULT_EXTRACTOR not in mapping:
        problems.append(f"{owner}: extractors must be a table with a {DEFAULT_EXTRACTOR!r} key")
        return {}
    for key, name in mapping.items():
        if key != DEFAULT_EXTRACTOR and key not in FORMATS:
            problems.append(f"{owner}: extractor key {key!r} is not 'default' or a format in {FORMATS}")
        if name not in extractors:
            problems.append(f"{owner}: extractor {name!r} is not defined")
    return {str(k): str(v) for k, v in mapping.items()}


def _parse_pipelines(
    tables: Mapping[str, Any],
    detectors: Mapping[str, Component],
    extractors: Mapping[str, Component],
    problems: list[str],
) -> dict[str, Pipeline]:
    pipelines = {}
    for name, table in tables.items():
        if not _check_name(name, "pipeline", problems):
            continue
        owner = f"pipeline {name!r}"
        detector = table.get("detector")
        if detector not in detectors:
            problems.append(f"{owner}: detector {detector!r} is not defined")
        problems += [f"{owner}: unknown key {key!r}" for key in sorted(table.keys() - {"detector", "extractors"})]
        mapping = _check_extractor_map(owner, table.get("extractors"), extractors, problems)
        pipelines[name] = Pipeline(name, str(detector), mapping)
    return pipelines


def _parse_runs(
    entries: list[Mapping[str, Any]],
    detectors: Mapping[str, Component],
    extractors: Mapping[str, Component],
    pipelines: Mapping[str, Pipeline],
    problems: list[str],
) -> dict[str, RunSpec]:
    runs: dict[str, RunSpec] = {}
    for entry in entries:
        name = entry.get("name")
        if not _check_name(name, "run", problems):
            continue
        owner = f"run {name!r}"
        if name in runs:
            problems.append(f"{owner} is defined twice")
        mode, split, channel = entry.get("mode"), entry.get("split"), entry.get("channel")
        if mode not in MODES:
            problems.append(f"{owner}: mode {mode!r} is not one of {MODES}")
            continue
        if split not in SPLITS:
            problems.append(f"{owner}: split {split!r} is not one of {SPLITS}")
        if channel is not None and channel not in CHANNELS:
            problems.append(f"{owner}: channel {channel!r} is not one of {CHANNELS}")
        problems += [f"{owner}: key {key!r} does not apply to {mode} runs" for key in entry.keys() - _RUN_KEYS[mode]]
        _check_reference(owner, mode, entry, detectors, pipelines, problems)
        mapping = _check_extractor_map(owner, entry.get("extractors"), extractors, problems) if mode == EXTRACT else {}
        runs[str(name)] = RunSpec(
            name=str(name),
            mode=mode,
            split=str(split),
            detector=entry.get("detector"),
            pipeline=entry.get("pipeline"),
            extractors=mapping,
            channel=channel,
        )
    return runs


def _check_reference(
    owner: str,
    mode: str,
    entry: Mapping[str, Any],
    detectors: Mapping[str, Component],
    pipelines: Mapping[str, Pipeline],
    problems: list[str],
) -> None:
    checks: dict[str, tuple[str, Mapping[str, Any]]] = {
        TEXT: ("detector", detectors),
        PIPELINE: ("pipeline", pipelines),
    }
    if mode in checks:
        key, table = checks[mode]
        if entry.get(key) not in table:
            problems.append(f"{owner}: {key} {entry.get(key)!r} is not defined")


def _parse_comparisons(
    entries: list[Mapping[str, Any]], runs: Mapping[str, RunSpec], problems: list[str]
) -> dict[str, Comparison]:
    comparisons: dict[str, Comparison] = {}
    for entry in entries:
        name = entry.get("name")
        if not _check_name(name, "comparison", problems):
            continue
        owner = f"comparison {name!r}"
        if name in comparisons:
            problems.append(f"{owner} is defined twice")
        sides = [entry.get("baseline"), entry.get("candidate")]
        missing = [side for side in sides if side not in runs]
        problems += [f"{owner}: run {side!r} is not defined" for side in missing]
        if not missing:
            check_comparable(owner, runs[str(sides[0])], runs[str(sides[1])], problems)
        comparisons[str(name)] = Comparison(str(name), str(sides[0]), str(sides[1]))
    return comparisons


def check_comparable(owner: str, base: RunSpec, candidate: RunSpec, problems: list[str]) -> None:
    """Append a problem for each reason the two runs score different items or have no detection scores."""
    for what in ("mode", "split", "channel"):
        a, b = getattr(base, what), getattr(candidate, what)
        if a != b:
            problems.append(f"{owner}: runs differ in {what} ({a!r} vs {b!r})")
    if base.mode == EXTRACT:
        problems.append(f"{owner}: extract runs have no detection scores to compare")
