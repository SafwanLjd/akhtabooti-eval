"""Download the pinned model files listed in ``configs/models/*.toml``.

``[[file]]`` entries name a URL and the sha256 the download must have.
``[[hf]]`` entries name a Hugging Face repository at a commit; after the
download, ``<path>/FETCHED.json`` records the repository, revision and the
sha256 of every file fetched.
"""

from __future__ import annotations

import json
import shutil
import tomllib
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .dataset import sha256_file
from .registry import ConfigError

MODELS_DIR = Path("configs/models")
FETCHED = "FETCHED.json"
TIMEOUT_SECONDS = 60
CHUNK_BYTES = 1 << 20
PRESENT = "present"
DOWNLOADED = "downloaded"


class FetchError(RuntimeError):
    """A download whose sha256 differs from the pinned value."""


@dataclass(frozen=True)
class FileEntry:
    name: str
    url: str
    path: Path
    sha256: str


@dataclass(frozen=True)
class HfEntry:
    name: str
    repo: str
    revision: str
    path: Path
    allow: tuple[str, ...] = ()


Entry = FileEntry | HfEntry


@dataclass(frozen=True)
class Outcome:
    name: str
    status: str
    error: str | None = None


def _require(table: Mapping[str, Any], keys: Iterable[str], where: str) -> None:
    missing = [k for k in keys if not isinstance(table.get(k), str) or not table[k]]
    if missing:
        raise ConfigError(f"{where}: missing {', '.join(missing)}")


def parse_entries(data: Mapping[str, Any], source: Path) -> list[Entry]:
    entries: list[Entry] = []
    for table in data.get("file", []):
        _require(table, ("name", "url", "path", "sha256"), f"{source} [[file]]")
        entries.append(FileEntry(table["name"], table["url"], Path(table["path"]), table["sha256"].lower()))
    for table in data.get("hf", []):
        _require(table, ("name", "repo", "revision", "path"), f"{source} [[hf]]")
        allow = tuple(str(p) for p in table.get("allow", []))
        entries.append(HfEntry(table["name"], table["repo"], table["revision"], Path(table["path"]), allow))
    unknown = sorted(data.keys() - {"file", "hf"})
    if unknown:
        raise ConfigError(f"{source}: unknown tables {', '.join(unknown)}")
    return entries


def load_entries(directory: Path = MODELS_DIR) -> list[Entry]:
    if not directory.is_dir():
        raise ConfigError(f"model manifest directory {directory} not found")
    entries: list[Entry] = []
    for source in sorted(directory.glob("*.toml")):
        try:
            data = tomllib.loads(source.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"{source}: invalid TOML: {exc}") from exc
        entries += parse_entries(data, source)
    names = [e.name for e in entries]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ConfigError(f"duplicate model entry names: {', '.join(duplicates)}")
    return entries


def fetch_file(entry: FileEntry) -> str:
    if entry.path.is_file() and sha256_file(entry.path) == entry.sha256:
        return PRESENT
    entry.path.parent.mkdir(parents=True, exist_ok=True)
    partial = entry.path.with_name(entry.path.name + ".part")
    with urllib.request.urlopen(entry.url, timeout=TIMEOUT_SECONDS) as response, partial.open("wb") as fh:
        shutil.copyfileobj(response, fh, CHUNK_BYTES)
    digest = sha256_file(partial)
    if digest != entry.sha256:
        partial.unlink()
        raise FetchError(f"{entry.name}: sha256 {digest} does not match the pinned {entry.sha256}")
    partial.replace(entry.path)
    return DOWNLOADED


def _snapshot_files(root: Path) -> dict[str, str]:
    files = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if path.is_file() and relative.parts[0] != ".cache" and relative.as_posix() != FETCHED:
            files[relative.as_posix()] = sha256_file(path)
    return files


def _fetched_record(entry: HfEntry) -> dict[str, Any]:
    return {"repo": entry.repo, "revision": entry.revision, "allow": list(entry.allow)}


def _is_current(entry: HfEntry) -> bool:
    marker = entry.path / FETCHED
    if not marker.is_file():
        return False
    record = json.loads(marker.read_text(encoding="utf-8"))
    files = record.get("files", {})
    same_source = all(record.get(k) == v for k, v in _fetched_record(entry).items())
    return (
        same_source
        and bool(files)
        and all(
            (entry.path / name).is_file() and sha256_file(entry.path / name) == digest for name, digest in files.items()
        )
    )


def fetch_hf(entry: HfEntry) -> str:
    if _is_current(entry):
        return PRESENT
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=entry.repo,
        revision=entry.revision,
        local_dir=entry.path,
        allow_patterns=list(entry.allow) or None,
    )
    record = _fetched_record(entry) | {"files": _snapshot_files(entry.path)}
    (entry.path / FETCHED).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return DOWNLOADED


def fetch(directory: Path = MODELS_DIR, only: Sequence[str] = ()) -> list[Outcome]:
    """Fetch every entry, or only the named ones; failures are reported per entry."""
    entries = load_entries(directory)
    unknown = sorted(set(only) - {e.name for e in entries})
    if unknown:
        raise ConfigError(f"no model entry named {', '.join(unknown)}")
    outcomes = []
    for entry in entries:
        if only and entry.name not in only:
            continue
        try:
            status = fetch_file(entry) if isinstance(entry, FileEntry) else fetch_hf(entry)
            outcomes.append(Outcome(entry.name, status))
        except Exception as exc:  # one failing entry must not stop the others
            outcomes.append(Outcome(entry.name, "failed", f"{type(exc).__name__}: {exc}"))
    return outcomes
