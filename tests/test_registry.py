from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from akheval import registry
from akheval.detectors.hybrid import HybridDetector
from akheval.detectors.regex_baseline import RegexBaselineDetector
from akheval.registry import Component, ConfigError, detector_factory, load_class

REPO = Path(__file__).resolve().parents[1]
HEAVY = ("torch", "onnxruntime", "extractous", "transformers", "tokenizers", "huggingface_hub")


def test_building_light_components_imports_no_model_runtime() -> None:
    code = f"""
import json, sys
import akheval.cli, akheval.runner, akheval.suite, akheval.tuning
from akheval.config import load_config
from akheval.registry import detector_factory
config = load_config("configs/eval-v1.toml")
factory = detector_factory(config.detectors)
factory.build("regex-baseline")
factory.build("core-main")
print(json.dumps([m for m in {HEAVY!r} if m in sys.modules]))
"""
    done = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, check=True)
    assert json.loads(done.stdout) == []


def test_kinds_point_at_module_and_class() -> None:
    for target in [*registry.DETECTOR_KINDS.values(), *registry.EXTRACTOR_KINDS.values()]:
        module, _, cls = target.partition(":")
        assert module.startswith("akheval.")
        assert cls[0].isupper()


def test_build_resolves_references_once() -> None:
    specs = {
        "rx": Component("rx", "regex-baseline"),
        "both": Component("both", "hybrid", {"parts": ["rx", "rx"]}),
    }
    factory = detector_factory(specs)
    hybrid = factory.build("both")
    assert isinstance(hybrid, HybridDetector)
    assert hybrid.parts[0] is hybrid.parts[1] is factory.build("rx")
    assert isinstance(hybrid.parts[0], RegexBaselineDetector)


def test_cycles_are_reported_with_the_chain() -> None:
    specs = {
        "a": Component("a", "hybrid", {"parts": ["b", "c"]}),
        "b": Component("b", "hybrid", {"parts": ["a", "c"]}),
        "c": Component("c", "regex-baseline"),
    }
    with pytest.raises(ConfigError, match="cycle: a -> b -> a"):
        detector_factory(specs).build("a")


def test_unknown_name_kind_and_parameters() -> None:
    specs = {
        "odd": Component("odd", "telepathy"),
        "loud": Component("loud", "regex-baseline", {"volume": 11}),
        "ref": Component("ref", "hybrid", {"parts": ["ghost", "loud"]}),
    }
    factory = detector_factory(specs)
    with pytest.raises(ConfigError, match=r"^unknown detector 'ghost'"):
        factory.build("ref")
    with pytest.raises(ConfigError, match="unknown kind 'telepathy'"):
        factory.build("odd")
    with pytest.raises(ConfigError, match="bad parameters"):
        factory.build("loud")


def test_a_rejected_parameter_value_is_a_config_error() -> None:
    params = {"core_sha": "68d1749", "located_categories": ["Email", "Phones"]}
    specs = {"core": Component("core", "core-rules", params)}
    with pytest.raises(ConfigError, match=r"detector 'core': bad parameters .*\['Phones'\]"):
        detector_factory(specs).build("core")


def test_target_must_be_a_detector(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(registry.DETECTOR_KINDS, "not-a-detector", "akheval.registry:Component")
    with pytest.raises(ConfigError, match="is not a Detector"):
        detector_factory({"x": Component("x", "not-a-detector")}).build("x")


@pytest.mark.parametrize(
    ("target", "message"),
    [
        ("akheval.registry", "must look like"),
        ("akheval.no_such_module:Thing", "cannot import"),
        ("akheval.registry:NoSuchClass", "has no class"),
    ],
)
def test_load_class_errors(target: str, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        load_class(target)
