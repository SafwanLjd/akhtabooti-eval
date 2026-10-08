"""Build Evaluation Dataset v1: plan the documents, write their text, native files and images, then the manifest."""

from __future__ import annotations

import json
import logging
import random
import shutil
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..dataset import DOCUMENTS, LANGUAGES, MANIFEST, Dataset, Document, Rendering, write_dataset
from .builder import Blocks, block_side, has_back, side_ranges
from .config import SynthConfig, load_config
from .context import Ctx
from .coverage import coverage_gaps, entity_coverage
from .doctypes import DOC_TYPES
from .imaging import ImageRenderer, default_renderer, plan_images
from .native import LIBREOFFICE_FORMATS, libreoffice_version, write_native_files
from .readback import Recovery, extractous_readback, missing_values
from .split import assign_splits

GENERATOR_NAME = "akheval.synth"
GENERATOR_VERSION = "1.2.0"
OUTPUT_ENTRIES = ("files", DOCUMENTS, MANIFEST)
OCR_CHANNEL = "ocr"
MAX_LISTED_PROBLEMS = 50

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DocSpec:
    doc_type: str
    language: str
    index: int

    @property
    def id(self) -> str:
        return f"{self.doc_type}-{self.language}-{self.index:03d}"

    @property
    def native_format(self) -> str:
        """Rotate through the type's formats; the language offset spreads formats across a small sample."""
        formats = DOC_TYPES[self.doc_type].native_formats
        return formats[(self.index - 1 + LANGUAGES.index(self.language)) % len(formats)]


def plan_documents(counts: Mapping[str, int]) -> list[DocSpec]:
    """Every document the config asks for, ``counts[type]`` per language."""
    return [
        DocSpec(doc_type, lang, index)
        for doc_type in DOC_TYPES
        for lang in LANGUAGES
        for index in range(1, counts.get(doc_type, 0) + 1)
    ]


def limit_documents(specs: Sequence[DocSpec], limit: int | None) -> list[DocSpec]:
    """The first ``limit`` documents in round-robin order over (type, language), sorted by id."""
    if limit is None:
        return sorted(specs, key=lambda s: s.id)
    type_order = list(DOC_TYPES)
    ordered = sorted(specs, key=lambda s: (s.index, type_order.index(s.doc_type), LANGUAGES.index(s.language)))
    return sorted(ordered[:limit], key=lambda s: s.id)


def generate_document(spec: DocSpec, *, seed: int, split: str) -> Document:
    """Write the text, gold spans, negatives and layout of one document from its own seeded RNG."""
    ctx = Ctx(spec.language, random.Random(f"{seed}:{spec.id}"))
    style = DOC_TYPES[spec.doc_type].template(ctx)
    layout = ctx.b.build()
    return Document(
        id=spec.id,
        split=split,
        language=spec.language,
        doc_type=spec.doc_type,
        text=layout.text,
        spans=layout.spans,
        negatives=layout.negatives,
        meta={"synthetic": True, "style": style, "indic_digits": ctx.indic, **ctx.meta, "layout": layout.blocks},
    )


def generate_documents(cfg: SynthConfig, limit: int | None = None) -> list[Document]:
    """Documents with text and annotations, without rendering files."""
    specs = plan_documents(cfg.documents)
    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for spec in specs:
        strata[(spec.doc_type, spec.language)].append(spec.id)
    splits = assign_splits(strata.values(), seed=cfg.seed, dev_fraction=cfg.dev_fraction)
    return [generate_document(s, seed=cfg.seed, split=splits[s.id]) for s in limit_documents(specs, limit)]


def _shown(layout: Blocks, face: str | None) -> tuple[Blocks, tuple[tuple[int, int], ...] | None]:
    """Blocks that an image of card side ``face`` draws and their ranges of the text; None draws every block."""
    if face is None:
        return layout, None
    return [b for b in layout if block_side(b) == face], side_ranges(layout, face)


def render_images(
    documents: Sequence[Document], root: Path, cfg: SynthConfig, renderer: ImageRenderer
) -> dict[str, list[Rendering]]:
    """Photos, scans and screenshots planned for each document, drawn by ``renderer``."""
    renderings: dict[str, list[Rendering]] = {}
    for doc in documents:
        layout = doc.meta["layout"]
        rng = random.Random(f"{cfg.seed}:{doc.id}:images")
        jobs = plan_images(DOC_TYPES[doc.doc_type].images, cfg.images, rng, two_sided=has_back(layout))
        renderings[doc.id] = []
        for job in jobs:
            rendering_id = job.rendering_id(doc.id)
            blocks, visible = _shown(layout, job.face)
            params = renderer(
                blocks,
                style=doc.meta["style"],
                language=doc.language,
                capture=job.capture,
                severity=job.severity,
                rng=random.Random(f"{cfg.seed}:{rendering_id}"),
                out=root / job.relpath(doc.id),
            )
            renderings[doc.id].append(
                Rendering(
                    id=rendering_id,
                    path=job.relpath(doc.id),
                    format=job.fmt,
                    channel=OCR_CHANNEL,
                    capture=job.capture,
                    severity=job.severity,
                    params=params,
                    visible=visible,
                )
            )
    return renderings


def _prepare_output(out: Path, *, overwrite: bool) -> None:
    manifest = out / MANIFEST
    if manifest.is_file() and json.loads(manifest.read_text(encoding="utf-8")).get("frozen"):
        raise FileExistsError(f"{out} holds a frozen dataset; build a new version into a new directory")
    existing = [out / name for name in OUTPUT_ENTRIES if (out / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(f"{out} already holds a dataset; pass overwrite=True to replace it")
    for path in existing:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    out.mkdir(parents=True, exist_ok=True)


def _check_native_files(documents: Sequence[Document], root: Path) -> Recovery:
    """Stop when extractous cannot parse a native file or a file lacks a gold value; return extractous recovery."""
    recovery, problems = extractous_readback(documents, root)
    problems += [p for doc in documents for p in missing_values(doc, root)]
    if problems:
        listed = "\n".join(problems[:MAX_LISTED_PROBLEMS])
        raise ValueError(f"{len(problems)} problem(s) in native files:\n{listed}")
    return recovery


def _generator_info(
    cfg: SynthConfig, documents: Sequence[Document], limit: int | None, recovery: Recovery | None
) -> dict[str, Any]:
    coverage = entity_coverage(documents, "test")
    info: dict[str, Any] = {
        "name": GENERATOR_NAME,
        "version": GENERATOR_VERSION,
        "config": dict(cfg.raw),
        "limit": limit,
        "test_coverage": coverage,
        "test_coverage_gaps": coverage_gaps(coverage, cfg.min_test_entities),
    }
    if recovery is not None:
        info["extractous_recovery"] = recovery
    if any(r.format in LIBREOFFICE_FORMATS and r.channel == "native" for d in documents for r in d.renderings):
        info["libreoffice"] = libreoffice_version(cfg.soffice)
    return info


def build_dataset(
    config: Path,
    out: Path,
    *,
    limit: int | None = None,
    renderer: ImageRenderer | None = None,
    overwrite: bool = False,
) -> Dataset:
    """Generate the dataset described by ``config`` into ``out`` and return it, unfrozen.

    ``limit`` caps the number of documents; the first documents of every
    (type, language) pair come first, so 39 documents cover every pair once.
    """
    cfg = load_config(config)
    _prepare_output(out, overwrite=overwrite)
    documents = generate_documents(cfg, limit)
    log.info("generated %d documents", len(documents))
    formats = {spec.id: spec.native_format for spec in plan_documents(cfg.documents)}
    natives = write_native_files(documents, formats, out, soffice=cfg.soffice, timeout=cfg.soffice_timeout)
    images = render_images(documents, out, cfg, renderer or default_renderer())
    for doc in documents:
        doc.renderings = (natives[doc.id], *images[doc.id])
    log.info("wrote %d renderings", sum(len(d.renderings) for d in documents))
    recovery = _check_native_files(documents, out) if cfg.verify_native else None
    generator = _generator_info(cfg, documents, limit, recovery)
    for gap in generator["test_coverage_gaps"] if limit is None else []:
        log.warning("test split below %d gold entities: %s", cfg.min_test_entities, gap)
    manifest = write_dataset(documents, out, name=cfg.name, version=cfg.version, generator=generator, notes=cfg.notes)
    return Dataset(root=out, manifest=manifest, documents=documents)
