"""Smoke test: every module imports and every config in configs/ loads."""

import dataclasses
import importlib
import importlib.util
from pathlib import Path

import pytest

from nano_agentrl.config import Config, load_config, save_config

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_MODULES = sorted(
    ".".join(p.relative_to(ROOT).with_suffix("").parts).removesuffix(".__init__")
    for p in (ROOT / "nano_agentrl").rglob("*.py")
)
SCRIPTS = sorted((ROOT / "scripts").glob("*.py"))
CONFIGS = sorted((ROOT / "configs").glob("*.yaml"))


@pytest.mark.parametrize("name", PACKAGE_MODULES)
def test_package_module_imports(name: str) -> None:
    importlib.import_module(name)


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_script_imports(path: Path) -> None:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_config_loads(path: Path) -> None:
    config = load_config(path)
    assert isinstance(config, Config)


def test_configs_exist() -> None:
    assert CONFIGS, "expected at least one YAML file in configs/"


def test_unknown_key_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("group_sise: 8\n")
    with pytest.raises(ValueError, match="group_sise"):
        load_config(bad)


def test_save_config_round_trip(tmp_path: Path) -> None:
    config = load_config(CONFIGS[0], overrides={"seed": 7})
    saved = save_config(config, tmp_path / "run")
    assert dataclasses.asdict(load_config(saved)) == dataclasses.asdict(config)
