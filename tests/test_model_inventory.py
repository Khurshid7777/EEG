"""Frozen-model inventory (risk R14): docs/regulatory/software_lifecycle.md lists every working model
with the start of its SHA-256. A model changed without a new pre-registration, or added to a config
without an inventory entry, fails here."""

import glob
import hashlib
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _inventory() -> list[str]:
    text = (ROOT / "docs/regulatory/software_lifecycle.md").read_text(encoding="utf-8")
    table = text[text.index("| Модель | Назначение"):]
    table = table[:table.index("\n\n")]
    return [p for row in re.findall(r"\| ([0-9a-f]{16})(?: / ([0-9a-f]{16}))? \|", table) for p in row if p]


def _files() -> dict[str, str]:
    return {Path(f).relative_to(ROOT).as_posix(): hashlib.sha256(Path(f).read_bytes()).hexdigest()
            for f in glob.glob(str(ROOT / "configs/models/*.json"))}


def _config_models() -> set[str]:
    refs: set[str] = set()

    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                if isinstance(v, str) and v.startswith("configs/models/"):
                    refs.add(v)
                else:
                    walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    for y in glob.glob(str(ROOT / "configs/*.yaml")):
        walk(yaml.safe_load(Path(y).read_text(encoding="utf-8")) or {})
    return refs


def test_every_inventory_entry_matches_exactly_one_model_file():
    files = _files()
    inv = _inventory()
    assert len(inv) >= 10
    for prefix in inv:
        assert len([f for f, h in files.items() if h.startswith(prefix)]) == 1, prefix


def test_every_model_used_by_a_config_is_in_the_inventory():
    files, inv = _files(), _inventory()
    refs = _config_models()
    assert refs, "no model references found in configs/*.yaml"
    missing = sorted(r for r in refs if not any(files[r].startswith(p) for p in inv))
    assert not missing, f"models in use but not in the frozen inventory: {missing}"
