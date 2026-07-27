from __future__ import annotations

import json
from pathlib import Path

from tools.check_experiment_catalog import CATALOG, validate


def test_public_experiment_catalog_is_hash_closed() -> None:
    assert validate() == []


def test_catalog_rejects_stale_evidence_hash(tmp_path: Path) -> None:
    value = json.loads(CATALOG.read_text(encoding="utf-8"))
    value["experiments"][0]["evidence"][0]["sha256"] = "0" * 64
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    assert any("stale evidence hash" in item for item in validate(path))
