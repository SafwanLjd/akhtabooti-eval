"""Command line: ``akheval <command>``. Each command delegates to one module function.

Exit codes: 0 success, 1 validation or run failure, 2 usage or config error.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path

from . import __version__
from .config import load_config
from .dataset import SPLITS, Dataset, counts, freeze, validate
from .registry import ConfigError

OK = 0
FAILURE = 1
USAGE = 2
DEFAULT_CONFIG = Path("configs/eval-v1.toml")
CONTACT_SHEET_LIMIT = 48

Handler = Callable[[argparse.Namespace], int]


def _dataset_path(args: argparse.Namespace) -> Path:
    return Path(args.dataset) if args.dataset else load_config(args.config).project.dataset


def cmd_synth(args: argparse.Namespace) -> int:
    from .synth.build import build_dataset
    from .synth.coverage import entity_coverage, format_coverage

    dataset = build_dataset(Path(args.config), Path(args.out), limit=args.limit, overwrite=args.overwrite)
    print(f"wrote {len(dataset.documents)} documents to {dataset.root}")
    print("gold entities in the test split:")
    print(format_coverage(entity_coverage(dataset.documents, "test")))
    return OK


def cmd_contact_sheet(args: argparse.Namespace) -> int:
    from .synth.images.contact import make_contact_sheet

    dataset = Dataset.load(_dataset_path(args))
    images = sorted(dataset.root / r.path for _, r in dataset.renderings(args.split) if r.channel == "ocr")
    step = max(1, len(images) // args.limit)
    make_contact_sheet(images[::step][: args.limit], args.out)
    print(f"contact sheet of {min(len(images), args.limit)} images written to {args.out}")
    return OK


def cmd_validate(args: argparse.Namespace) -> int:
    dataset = Dataset.load(_dataset_path(args))
    problems = validate(dataset)
    print(json.dumps(counts(dataset.documents), ensure_ascii=False, indent=2, sort_keys=True))
    print(f"dataset {dataset.name} {dataset.version}, frozen: {dataset.frozen}")
    for problem in problems:
        print(f"problem: {problem}")
    print(f"{len(problems)} problem(s)" if problems else "valid")
    return FAILURE if problems else OK


def cmd_freeze(args: argparse.Namespace) -> int:
    dataset = Dataset.load(_dataset_path(args))
    manifest = freeze(dataset)
    print(f"froze {dataset.name} {dataset.version} at {manifest['frozen_at']}")
    return OK


def cmd_fetch(args: argparse.Namespace) -> int:
    from .fetch import fetch

    outcomes = fetch(Path(args.models_dir), args.only or ())
    for outcome in outcomes:
        print(f"{outcome.name}: {outcome.status}" + (f" ({outcome.error})" if outcome.error else ""))
    return FAILURE if any(o.error for o in outcomes) else OK


def cmd_run(args: argparse.Namespace) -> int:
    from .runner import REPORT, run

    run_dir = run(load_config(args.config), args.run)
    print(f"run written to {run_dir}; report: {run_dir / REPORT}")
    return OK


def cmd_suite(args: argparse.Namespace) -> int:
    from .suite import suite

    return suite(Path(args.config), args.only or ())


def cmd_tune(args: argparse.Namespace) -> int:
    from .tuning import tune

    path, threshold = tune(load_config(args.config), args.detector, args.criterion)
    print(f"sweep written to {path}")
    print(f"set threshold = {threshold} in [detectors.{args.detector}]")
    return OK


def cmd_report(args: argparse.Namespace) -> int:
    from .runner import score_run

    config = load_config(args.config)
    config.run(args.run)
    print(score_run(config.run_dir(args.run), Dataset.load(config.project.dataset)))
    return OK


def cmd_compare(args: argparse.Namespace) -> int:
    from .config import Comparison, check_comparable
    from .suite import write_comparison

    config = load_config(args.config)
    if args.comparison:
        comparison = config.comparison(args.comparison)
    elif args.baseline and args.candidate:
        base, candidate = config.run(args.baseline), config.run(args.candidate)
        problems: list[str] = []
        check_comparable("--baseline and --candidate", base, candidate, problems)
        if problems:
            raise ConfigError("; ".join(problems))
        comparison = Comparison(args.name or f"{base.name}-vs-{candidate.name}", base.name, candidate.name)
    else:
        raise ConfigError("give --comparison, or both --baseline and --candidate")
    print(write_comparison(config, comparison, Dataset.load(config.project.dataset)))
    return OK


def cmd_score_app_report(args: argparse.Namespace) -> int:
    from .app_report import score_app_report

    config = load_config(args.config)
    if args.dataset:
        config = replace(config, project=replace(config.project, dataset=Path(args.dataset)))
    run_dir = score_app_report(
        config, Path(args.report), split=args.split, name=args.name, app_version=args.app_version
    )
    print(f"run written to {run_dir}")
    return OK


def _add(
    sub: argparse._SubParsersAction[argparse.ArgumentParser], name: str, handler: Handler, help_text: str
) -> argparse.ArgumentParser:
    parser = sub.add_parser(name, help=help_text, description=help_text)
    parser.set_defaults(handler=handler)
    return parser


def _config_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help=f"evaluation config (default {DEFAULT_CONFIG})"
    )


def _dataset_options(parser: argparse.ArgumentParser) -> None:
    _config_option(parser)
    parser.add_argument("--dataset", help="dataset directory; defaults to [project].dataset of the config")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="akheval", description="Evaluation harness for the Akhtabooti PII scanner")
    parser.add_argument("--version", action="version", version=f"akheval {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = _add(sub, "synth", cmd_synth, "build the synthetic dataset from a generator config")
    p.add_argument("--config", type=Path, required=True, help="generator config")
    p.add_argument("--out", type=Path, required=True, help="dataset directory to write")
    p.add_argument("--limit", type=int, help="build at most this many documents")
    p.add_argument("--overwrite", action="store_true", help="replace an existing dataset directory")

    p = _add(sub, "contact-sheet", cmd_contact_sheet, "tile a sample of a dataset's image renderings into one image")
    _dataset_options(p)
    p.add_argument("--out", type=Path, required=True, help="image file to write")
    p.add_argument("--split", choices=SPLITS, help="only this split")
    p.add_argument(
        "--limit", type=int, default=CONTACT_SHEET_LIMIT, help="images on the sheet, spread over the dataset"
    )

    _dataset_options(_add(sub, "validate", cmd_validate, "check a dataset's annotations, files and checksums"))
    _dataset_options(_add(sub, "freeze", cmd_freeze, "mark a valid dataset as frozen"))

    p = _add(sub, "fetch", cmd_fetch, "download pinned models listed in configs/models/*.toml")
    p.add_argument("--models-dir", default="configs/models")
    p.add_argument("--only", action="append", metavar="NAME", help="fetch only this entry; repeatable")

    p = _add(sub, "run", cmd_run, "execute one configured run and score it")
    _config_option(p)
    p.add_argument("--run", required=True, metavar="NAME")

    p = _add(sub, "suite", cmd_suite, "execute every configured run, the comparisons and the index")
    _config_option(p)
    p.add_argument("--only", action="append", metavar="NAME", help="execute only this run; repeatable")

    p = _add(sub, "tune", cmd_tune, "sweep the score threshold of a detector on the dev split")
    _config_option(p)
    p.add_argument("--detector", required=True)
    p.add_argument(
        "--criterion", choices=("f1", "f2"), default="f2", help="lenient span F-score that picks the threshold"
    )

    p = _add(sub, "report", cmd_report, "re-score a stored run and rewrite its report")
    _config_option(p)
    p.add_argument("--run", required=True, metavar="NAME")

    p = _add(sub, "compare", cmd_compare, "compare two stored runs")
    _config_option(p)
    p.add_argument("--comparison", metavar="NAME", help="a [[comparisons]] entry")
    p.add_argument("--baseline", metavar="RUN")
    p.add_argument("--candidate", metavar="RUN")
    p.add_argument("--name", help="report name for an ad hoc comparison")

    p = _add(sub, "score-app-report", cmd_score_app_report, "score a JSON report exported by the desktop app")
    _dataset_options(p)
    p.add_argument("--report", required=True, type=Path)
    p.add_argument("--split", choices=SPLITS, default="test")
    p.add_argument("--name", default="app-report", help="run name for the results directory")
    p.add_argument("--app-version", default="unknown")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    handler: Handler = args.handler
    try:
        return handler(args)
    except ConfigError as exc:
        print(f"akheval: {exc}", file=sys.stderr)
        return USAGE
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"akheval: {exc}", file=sys.stderr)
        return FAILURE
