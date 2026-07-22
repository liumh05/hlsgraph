from __future__ import annotations

import base64
import gzip
import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from hlsgraph.bundle import BundleError, GraphBundle
from hlsgraph.sdk import Project
from hlsgraph.store import LedgerStore, StoreError
from hlsgraph.version import SCHEMA_VERSION


FIXTURE = Path(__file__).with_name("fixtures") / "v02_minimal_bundle.json"
IMMUTABLE_LEGACY_TABLES = (
    "snapshots", "snapshot_manifests", "project_state", "graph_views",
    "artifacts", "snapshot_artifacts", "entities", "relations",
    "entity_correspondences", "observations", "derivations", "runs",
    "diagnostics", "verifications", "variants", "action_materializations",
    "predictions", "knowledge_rules",
)


def _materialize_fixture(root: Path) -> dict[str, object]:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert fixture["format"] == "hlsgraph.test.v02_bundle.v1"
    assert fixture["generator"] == {
        "command": "hlsgraph init; hlsgraph index --degraded",
        "package_version": "0.2.0",
        "source_commit": "7b26bfb07aa7c4d1a4705d3076cac684c3561e6f",
        "wheel_sha256": "ee7c58bda360e1d6f60a233cb1c3ef3e620b2c1d85e4824361b805f9a8045fbc",
    }
    for relative, text in fixture["source_files"].items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
    (root / "hlsgraph.toml").write_text(
        fixture["source_manifest"], encoding="utf-8", newline="\n",
    )
    bundle = root / ".hlsgraph"
    (bundle / "artifacts").mkdir(parents=True)
    (bundle / "exports").mkdir()
    (bundle / "bundle.json").write_text(
        json.dumps(fixture["bundle_metadata"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )
    (bundle / "manifest.json").write_text(
        json.dumps(fixture["internal_manifest"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )
    database = gzip.decompress(base64.b64decode(
        "".join(fixture["database_gzip_base64"]), validate=True,
    ))
    assert hashlib.sha256(database).hexdigest() == fixture["database_sha256"]
    (bundle / "graph.db").write_bytes(database)
    return fixture


def _file_hashes(root: Path) -> dict[str, str]:
    paths = ("hlsgraph.toml", ".hlsgraph/manifest.json",
             ".hlsgraph/bundle.json", ".hlsgraph/graph.db")
    return {
        path: hashlib.sha256((root / path).read_bytes()).hexdigest()
        for path in paths
    }


def _legacy_rows(database: Path) -> dict[str, list[list[object]]]:
    uri = database.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.execute("PRAGMA query_only=ON")
        return {
            table: [list(row) for row in connection.execute(
                f'SELECT * FROM "{table}" ORDER BY 1'
            ).fetchall()]
            for table in IMMUTABLE_LEGACY_TABLES
        }


def test_real_v02_bundle_and_source_manifest_migrate_once_without_rewriting_truth(
    tmp_path: Path,
) -> None:
    fixture = _materialize_fixture(tmp_path)
    database = tmp_path / ".hlsgraph" / "graph.db"
    before_files = _file_hashes(tmp_path)
    before_rows = _legacy_rows(database)
    source_before = (tmp_path / "hlsgraph.toml").read_text(encoding="utf-8")

    # Before the explicit operation, audit access is read-only and every public
    # open/write path fails closed rather than performing an implicit upgrade.
    readonly_uri = database.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(readonly_uri, uri=True) as connection:
        assert connection.execute(
            "SELECT value FROM schema_info WHERE key='schema_version'"
        ).fetchone()[0] == "0.2.0"
        with pytest.raises(sqlite3.OperationalError, match="readonly|read-only"):
            connection.execute("UPDATE schema_info SET value='tampered'")
    with pytest.raises(BundleError, match="migration|supported"):
        Project.open(tmp_path)
    with pytest.raises(StoreError, match="explicit migration"):
        with LedgerStore(database).write():
            pass

    plan = GraphBundle.migration_plan(tmp_path)
    assert _file_hashes(tmp_path) == before_files
    assert [item["scope"] for item in plan] == [
        "bundle", "manifest_source", "ledger",
    ]
    assert [(item["from_version"], item["to_version"])
            for item in plan if item["scope"] == "ledger"] == [
        ("0.2.0", "0.3.0"),
    ]

    assert GraphBundle.migrate(tmp_path) == plan
    assert GraphBundle.migration_plan(tmp_path) == []
    assert _legacy_rows(database) == before_rows
    assert (tmp_path / "hlsgraph.toml").read_text(encoding="utf-8") == source_before.replace(
        'schema_version = "0.2.0"', 'schema_version = "0.3.0"', 1,
    )

    project = Project.open(tmp_path)
    expected = fixture["expected"]
    graph = project.bundle.store.load_graph(expected["snapshot_id"])
    assert graph.schema_version == "0.2.0"
    assert graph.graph_hash == expected["graph_hash"]
    assert sorted(graph.entities) == expected["entity_ids"]
    assert [item.id for item in project.bundle.store.observations(
        expected["snapshot_id"],
    )] == expected["observation_ids"]
    assert project.bundle.store.snapshot_manifest(
        expected["snapshot_id"],
    ).schema_version == "0.2.0"
    assert json.loads((tmp_path / ".hlsgraph" / "bundle.json").read_text(
        encoding="utf-8",
    ))["bundle_version"] == SCHEMA_VERSION
