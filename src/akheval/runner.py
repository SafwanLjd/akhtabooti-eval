"""Execute one configured run, persist its predictions and metadata, then score it.

A run directory holds ``run.json`` (run config, component info, dataset
identity, environment, timing, memory), ``predictions.jsonl`` (one line per
document or rendering), ``summary.json``, ``errors.jsonl`` and ``report.md``.
A file that fails to extract or detect is recorded with its error; the run
continues with the next one.

Memory: ``peak_rss_mb`` holds this process's peak and the peak of each helper
process (Core shim, akh-ocr, extractous worker) read from /proc while the
helper still runs, just before the run closes it. A helper that starts and
exits per file, such as tesseract, is not measured.
"""

from __future__ import annotations

import datetime as dt
import importlib.metadata
import json
import math
import os
import platform
import resource
import shutil
import subprocess
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import __version__
from .config import TEXT, EvalConfig, RunSpec, extractor_for
from .dataset import Dataset, Document, Rendering
from .detectors.base import Detector
from .extractors.base import Extraction, ExtractionError, Extractor
from .registry import detector_factory, extractor_factory
from .report.run import render_run
from .scoring import ItemScore, Prediction, score_predictions, summarize

RUN_JSON = "run.json"
PREDICTIONS = "predictions.jsonl"
SUMMARY = "summary.json"
ERRORS = "errors.jsonl"
REPORT = "report.md"

KEY_PACKAGES = (
    "akheval",
    "rapidfuzz",
    "numpy",
    "onnxruntime",
    "tokenizers",
    "transformers",
    "torch",
    "pytorch-crf",
    "huggingface_hub",
    "extractous",
    "pytesseract",
    "pillow",
    "python-bidi",
)
KIB_PER_MIB = 1024
BYTES_PER_GIB = 1024**3
P50 = 0.50
P95 = 0.95


@dataclass
class Execution:
    """Raw outcome of a run before scoring."""

    predictions: list[Prediction]
    components: dict[str, Any] = field(default_factory=dict)
    init_seconds: dict[str, float] = field(default_factory=dict)
    wall_seconds: float = 0.0
    helper_rss_mb: dict[str, float] = field(default_factory=dict)


def _timed(action: Callable[[], None]) -> float:
    start = time.perf_counter()
    action()
    return time.perf_counter() - start


def _failure(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def detect_into(detector: Detector, prediction: Prediction, text: str) -> None:
    """Run the detector on one text and store the detection, or the error, and the time taken."""
    start = time.perf_counter()
    try:
        prediction.detection = detector.detect(text)
    except Exception as exc:  # one failing item must not abort the run
        prediction.detection_error = _failure(exc)
    prediction.detect_seconds = time.perf_counter() - start


def extract_one(extractor: Extractor, path: Path) -> Extraction:
    """Extract one file and record its error, if any, with the time taken."""
    start = time.perf_counter()
    try:
        text, error = extractor.extract(path), None
    except ExtractionError as exc:
        text, error = "", str(exc)
    except Exception as exc:  # one failing file must not abort the run
        text, error = "", _failure(exc)
    return Extraction(text, time.perf_counter() - start, error)


def selected_renderings(documents: Sequence[Document], channel: str | None) -> list[tuple[Document, Rendering]]:
    return [(d, r) for d in documents for r in d.renderings if channel is None or r.channel == channel]


def execute(config: EvalConfig, spec: RunSpec, documents: Sequence[Document], root: Path) -> Execution:
    """Build the run's components, then extract and detect over the selected items.

    The detector is built before any file is extracted, so a bad parameter fails the run before OCR starts.
    """
    execution = Execution(predictions=[])
    name = config.detector_of(spec)
    detector = None if name is None else detector_factory(config.detectors).build(name)
    if spec.mode == TEXT:
        execution.predictions = [Prediction(item_id=d.id, doc_id=d.id) for d in documents]
        inputs = [d.text for d in documents]
    else:
        _run_renderings(execution, config, spec, documents, root)
        inputs = [p.extracted or "" for p in execution.predictions]
    if name is not None and detector is not None:
        _run_detection(execution, name, detector, inputs)
    return execution


def _prepare(execution: Execution, name: str, component: Detector | Extractor) -> None:
    execution.init_seconds[name] = _timed(component.prepare)
    execution.components[name] = component.info()


def _run_renderings(
    execution: Execution, config: EvalConfig, spec: RunSpec, documents: Sequence[Document], root: Path
) -> None:
    mapping = config.extractors_of(spec)
    groups: dict[str, list[tuple[Document, Rendering]]] = {}
    for doc, rendering in selected_renderings(documents, spec.channel):
        groups.setdefault(extractor_for(mapping, rendering.format), []).append((doc, rendering))
    factory = extractor_factory(config.extractors)
    built: list[Extractor] = []
    try:
        for name in sorted(groups):
            extractor = factory.build(name)
            built.append(extractor)
            _extract_group(execution, name, extractor, groups[name], root)
    finally:
        _record_helpers(execution)
        # Extractors can share components (pdf-ocr wraps others), so none closes before all groups ran.
        for extractor in built:
            extractor.close()


def _extract_group(
    execution: Execution, name: str, extractor: Extractor, group: Sequence[tuple[Document, Rendering]], root: Path
) -> None:
    _prepare(execution, name, extractor)
    start = time.perf_counter()
    results = [extract_one(extractor, root / r.path) for _, r in group]
    execution.wall_seconds += time.perf_counter() - start
    for (doc, rendering), result in zip(group, results, strict=True):
        execution.predictions.append(
            Prediction(
                item_id=rendering.id,
                doc_id=doc.id,
                rendering_id=rendering.id,
                extractor=name,
                extracted=result.text,
                extraction_error=result.error,
                extract_seconds=result.seconds,
            )
        )


def _run_detection(execution: Execution, name: str, detector: Detector, inputs: Sequence[str]) -> None:
    try:
        _prepare(execution, name, detector)
        start = time.perf_counter()
        for prediction, text in zip(execution.predictions, inputs, strict=True):
            detect_into(detector, prediction, text)
        execution.wall_seconds += time.perf_counter() - start
    finally:
        _record_helpers(execution)
        detector.close()


def _record_helpers(execution: Execution) -> None:
    for program, peak in helper_peaks().items():
        execution.helper_rss_mb[program] = max(peak, execution.helper_rss_mb.get(program, 0.0))


def percentile(values: Sequence[float], q: float) -> float | None:
    """Nearest-rank percentile."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def _stats(values: Sequence[float]) -> dict[str, float | None]:
    mean = sum(values) / len(values) if values else None
    return {key: None if v is None else round(v, 6) for key, v in (("mean", mean), *_quantiles(values))}


def _quantiles(values: Sequence[float]) -> tuple[tuple[str, float | None], ...]:
    return ("p50", percentile(values, P50)), ("p95", percentile(values, P95))


def timing(execution: Execution, chars: Mapping[str, int]) -> dict[str, Any]:
    predictions = execution.predictions
    detect = [p.detect_seconds for p in predictions if p.detect_seconds is not None]
    extract = [p.extract_seconds for p in predictions if p.extract_seconds is not None]
    per_item = [s for p in predictions if (s := p.seconds) is not None]
    busy = sum(per_item)
    return {
        "init_seconds": {k: round(v, 4) for k, v in sorted(execution.init_seconds.items())},
        "total_seconds": round(execution.wall_seconds, 4),
        "items": len(predictions),
        "per_item_seconds": _stats(per_item),
        "extract_seconds": _stats(extract),
        "detect_seconds": _stats(detect),
        "chars_per_second": round(sum(chars.values()) / busy, 1) if busy else None,
    }


def peak_rss_mb(helpers: Mapping[str, float]) -> dict[str, Any]:
    """Peak resident memory of this process and of each helper program, in MiB.

    RUSAGE_CHILDREN is no use here: Linux starts a child's count at the parent's peak, so it reports this process.
    """
    own = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"self": round(own / KIB_PER_MIB, 1), "helpers": dict(sorted(helpers.items()))}


def _descendants(pid: int) -> list[int]:
    """Live descendants of ``pid``; empty where /proc lists no children."""
    children: list[int] = []
    for listing in Path(f"/proc/{pid}/task").glob("*/children"):
        try:
            children += [int(child) for child in listing.read_text(encoding="ascii").split()]
        except OSError:
            continue  # the thread exited
    return [descendant for child in children for descendant in (child, *_descendants(child))]


def vm_hwm_mb(status: str) -> float | None:
    """Peak resident memory from the text of /proc/<pid>/status, in MiB; None without a VmHWM line."""
    for line in status.splitlines():
        if line.startswith("VmHWM:"):
            return round(int(line.split()[1]) / KIB_PER_MIB, 1)
    return None


def _program(pid: int) -> str:
    argv = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    return Path(argv[0].decode(errors="replace")).name


def helper_peaks() -> dict[str, float]:
    """Peak resident memory of each live helper process of this run by program name, in MiB."""
    peaks: dict[str, float] = {}
    for pid in _descendants(os.getpid()):
        try:
            program, peak = _program(pid), vm_hwm_mb(Path(f"/proc/{pid}/status").read_text(encoding="utf-8"))
        except OSError:
            continue  # the process exited
        if program and peak is not None:
            peaks[program] = max(peak, peaks.get(program, 0.0))
    return peaks


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def _ram_gb() -> float | None:
    try:
        pages, page_size = os.sysconf("SC_PHYS_PAGES"), os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return None
    return round(pages * page_size / BYTES_PER_GIB, 1)


def _command_output(command: Sequence[str]) -> str | None:
    if shutil.which(command[0]) is None:
        return None
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_state() -> dict[str, Any] | None:
    commit = _command_output(["git", "rev-parse", "HEAD"])
    if commit is None:
        return None
    status = _command_output(["git", "--no-optional-locks", "status", "--porcelain"])
    return {"commit": commit, "dirty": bool(status)}


def environment() -> dict[str, Any]:
    tesseract = _command_output(["tesseract", "--version"])
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "ram_gb": _ram_gb(),
        "packages": {name: _package_version(name) for name in KEY_PACKAGES},
        "tesseract": tesseract.splitlines()[0] if tesseract else None,
        "git": _git_state(),
    }


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def dataset_identity(dataset: Dataset, split: str | None) -> dict[str, Any]:
    return {
        "name": dataset.name,
        "version": dataset.version,
        "frozen": dataset.frozen,
        "root": str(dataset.root),
        "documents_sha256": dataset.documents_sha256(),
        "split": split,
    }


def _chars(prediction: Prediction, documents: Mapping[str, Document], mode: str) -> int:
    return len(documents[prediction.doc_id].text) if mode == TEXT else len(prediction.extracted or "")


def run(config: EvalConfig, run_name: str) -> Path:
    """Execute a configured run and write its directory under ``[project].results``."""
    spec = config.run(run_name)
    dataset = Dataset.load(config.project.dataset)
    documents = dataset.split(spec.split)
    started = _now()
    execution = execute(config, spec, documents, dataset.root)
    by_id = {d.id: d for d in documents}
    chars = {p.item_id: _chars(p, by_id, spec.mode) for p in execution.predictions}
    meta = {
        "run": spec.to_dict(),
        "akheval": __version__,
        "config": str(config.path),
        "components": execution.components,
        "dataset": dataset_identity(dataset, spec.split),
        "environment": environment(),
        "timing": timing(execution, chars),
        "peak_rss_mb": peak_rss_mb(execution.helper_rss_mb),
        "started": started,
        "finished": _now(),
    }
    run_dir = config.run_dir(spec.name)
    write_run(run_dir, meta, execution.predictions)
    score_run(run_dir, dataset)
    return run_dir


def write_run(run_dir: Path, meta: Mapping[str, Any], predictions: Sequence[Prediction]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / RUN_JSON).write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (run_dir / PREDICTIONS).open("w", encoding="utf-8") as fh:
        for prediction in sorted(predictions, key=lambda p: p.item_id):
            fh.write(json.dumps(prediction.to_dict(), ensure_ascii=False) + "\n")


def load_run(run_dir: Path) -> tuple[dict[str, Any], list[Prediction]]:
    meta: dict[str, Any] = json.loads((run_dir / RUN_JSON).read_text(encoding="utf-8"))
    with (run_dir / PREDICTIONS).open(encoding="utf-8") as fh:
        predictions = [Prediction.from_dict(json.loads(line)) for line in fh if line.strip()]
    return meta, predictions


def check_dataset(meta: Mapping[str, Any], dataset: Dataset) -> None:
    """Refuse to score a run against a dataset other than the one it ran on."""
    recorded = meta["dataset"]["documents_sha256"]
    current = dataset.documents_sha256()
    if recorded != current:
        raise ValueError(
            f"run {meta['run']['name']} used documents.jsonl {recorded[:12]}, the dataset now has {current[:12]}"
        )


def _score(meta: Mapping[str, Any], predictions: list[Prediction], dataset: Dataset) -> list[ItemScore]:
    check_dataset(meta, dataset)
    return score_predictions(dataset.documents, meta["run"]["mode"], predictions)


def scored_items(run_dir: Path, dataset: Dataset) -> tuple[dict[str, Any], list[ItemScore]]:
    meta, predictions = load_run(run_dir)
    return meta, _score(meta, predictions, dataset)


def _counted(counters: Iterable[Counter[str]]) -> dict[str, int]:
    return dict(sorted(sum(counters, Counter()).items()))


def score_run(run_dir: Path, dataset: Dataset) -> Path:
    """Score a stored run and write ``summary.json``, ``errors.jsonl`` and ``report.md``; returns the report path."""
    meta, predictions = load_run(run_dir)
    items = _score(meta, predictions, dataset)
    detections = [p.detection for p in predictions if p.detection is not None]
    summary = summarize(meta["run"]["mode"], items) | {
        "run": meta["run"],
        "dataset": meta["dataset"],
        "timing": meta.get("timing"),
        "peak_rss_mb": meta.get("peak_rss_mb"),
        "raw_labels": _counted(d.raw_labels for d in detections),
        "unmapped_labels": _counted(d.unmapped for d in detections),
    }
    (run_dir / SUMMARY).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    errors = sorted(
        (e.to_dict() for item in items for e in item.errors),
        key=lambda e: (e["kind"], e["doc_id"], e["rendering_id"] or "", e["label"], e["text"]),
    )
    with (run_dir / ERRORS).open("w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(e, ensure_ascii=False) + "\n" for e in errors)
    report = run_dir / REPORT
    report.write_text(render_run(meta, summary, items), encoding="utf-8")
    return report
