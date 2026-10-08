from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from akheval.fetch import (
    DOWNLOADED,
    FETCHED,
    PRESENT,
    FileEntry,
    HfEntry,
    fetch,
    fetch_hf,
    load_entries,
    parse_entries,
)
from akheval.registry import ConfigError


def _source(tmp_path: Path, content: bytes = b"weights") -> tuple[str, str]:
    path = tmp_path / "source.bin"
    path.write_bytes(content)
    return path.as_uri(), hashlib.sha256(content).hexdigest()


def _write_models(directory: Path, body: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "models.toml").write_text(body, encoding="utf-8")
    return directory


def test_file_entry_downloads_verifies_and_skips(tmp_path: Path) -> None:
    url, digest = _source(tmp_path)
    target = tmp_path / "models" / "w.bin"
    models = _write_models(
        tmp_path / "configs",
        f'[[file]]\nname = "w"\nurl = "{url}"\npath = "{target.as_posix()}"\nsha256 = "{digest}"\n',
    )
    assert [(o.name, o.status) for o in fetch(models)] == [("w", DOWNLOADED)]
    assert target.read_bytes() == b"weights"
    assert fetch(models)[0].status == PRESENT
    target.write_bytes(b"corrupt")
    assert fetch(models)[0].status == DOWNLOADED


def test_checksum_mismatch_fails_and_leaves_no_file(tmp_path: Path) -> None:
    url, _ = _source(tmp_path)
    target = tmp_path / "models" / "w.bin"
    models = _write_models(
        tmp_path / "configs",
        f'[[file]]\nname = "w"\nurl = "{url}"\npath = "{target.as_posix()}"\nsha256 = "{"0" * 64}"\n',
    )
    outcome = fetch(models)[0]
    assert outcome.status == "failed"
    assert outcome.error is not None
    assert "does not match" in outcome.error
    assert not target.exists()
    assert not target.with_name("w.bin.part").exists()


def test_only_selects_entries(tmp_path: Path) -> None:
    url, digest = _source(tmp_path)
    body = "".join(
        f'[[file]]\nname = "{n}"\nurl = "{url}"\npath = "{(tmp_path / n).as_posix()}"\nsha256 = "{digest}"\n'
        for n in ("a", "b")
    )
    models = _write_models(tmp_path / "configs", body)
    assert [o.name for o in fetch(models, ["b"])] == ["b"]
    with pytest.raises(ConfigError, match="no model entry named c"):
        fetch(models, ["c"])


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"file": [{"name": "x", "url": "u", "path": "p"}]}, "missing sha256"),
        ({"hf": [{"name": "x", "repo": "r", "path": "p"}]}, "missing revision"),
        ({"model": []}, "unknown tables model"),
    ],
)
def test_malformed_entries(data: dict[str, Any], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        parse_entries(data, Path("m.toml"))


def test_missing_manifest_directory(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_entries(tmp_path / "nowhere")


def test_invalid_toml_is_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="invalid TOML"):
        load_entries(_write_models(tmp_path / "configs", "[[file]\n"))


def test_duplicate_names_are_rejected(tmp_path: Path) -> None:
    entry = '[[hf]]\nname = "x"\nrepo = "r"\nrevision = "abc"\npath = "p"\n'
    models = _write_models(tmp_path / "configs", entry + entry)
    with pytest.raises(ConfigError, match="duplicate"):
        load_entries(models)


def test_hf_entry_records_files_and_skips_when_current(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import huggingface_hub

    calls: list[dict[str, Any]] = []

    def fake_snapshot(**kwargs: Any) -> str:
        calls.append(kwargs)
        root = Path(kwargs["local_dir"])
        (root / "onnx").mkdir(parents=True, exist_ok=True)
        (root / "onnx" / "model.onnx").write_bytes(b"onnx")
        (root / ".cache").mkdir(exist_ok=True)
        (root / ".cache" / "lock").write_bytes(b"")
        return str(root)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    entry = HfEntry("m", "org/model", "0123abc", tmp_path / "m", ("onnx/*",))
    assert fetch_hf(entry) == DOWNLOADED
    assert calls[0]["revision"] == "0123abc"
    assert calls[0]["allow_patterns"] == ["onnx/*"]
    record = json.loads((tmp_path / "m" / FETCHED).read_text(encoding="utf-8"))
    assert record["repo"] == "org/model"
    assert record["files"] == {"onnx/model.onnx": hashlib.sha256(b"onnx").hexdigest()}
    assert fetch_hf(entry) == PRESENT
    assert len(calls) == 1
    (tmp_path / "m" / "onnx" / "model.onnx").write_bytes(b"changed")
    assert fetch_hf(entry) == DOWNLOADED


def test_file_entry_fields_are_parsed() -> None:
    entries = parse_entries({"file": [{"name": "x", "url": "u", "path": "p/x", "sha256": "ABC"}]}, Path("m.toml"))
    assert entries == [FileEntry("x", "u", Path("p/x"), "abc")]
