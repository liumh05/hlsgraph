"""Explicit ledger migration registry.

HLSGraph never updates a database's schema marker merely because newer code
opened it.  A breaking schema change must add a deterministic, reviewed step to
``MIGRATIONS`` and be invoked explicitly by a user-facing migration command.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import sqlite3
from typing import Callable

from ..model import Derivation, json_ready
from ..version import SCHEMA_VERSION


MigrationApply = Callable[[sqlite3.Connection], None]


@dataclass(frozen=True, slots=True)
class MigrationStep:
    from_version: str
    to_version: str
    description: str
    apply: MigrationApply


_V01 = "0.1.0"
_V02 = "0.2.0"
_V03 = "0.3.0"


def _require_columns(
    connection: sqlite3.Connection, table: str, expected: frozenset[str],
) -> None:
    rows = connection.execute(f"PRAGMA table_info({table})").fetchall()
    actual = {str(row[1]) for row in rows}
    if not rows or not expected.issubset(actual):
        raise ValueError(
            f"ledger table {table!r} is missing required columns: "
            + ", ".join(sorted(expected - actual))
        )


def _migrate_v01_to_v02(connection: sqlite3.Connection) -> None:
    """Add v0.2 contracts without changing any v0.1 fact identity."""
    for table in (
        "schema_info", "snapshots", "entities", "artifacts", "snapshot_artifacts",
        "observations", "derivations", "diagnostics", "variants", "graph_views",
    ):
        if connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,),
        ).fetchone() is None:
            raise ValueError(f"v0.1 ledger is missing required table {table!r}")

    statements = (
        """CREATE TABLE IF NOT EXISTS entity_correspondences (
          id TEXT PRIMARY KEY,
          source_snapshot_id TEXT NOT NULL,
          source_entity_id TEXT NOT NULL,
          target_snapshot_id TEXT NOT NULL,
          target_entity_id TEXT NOT NULL,
          kind TEXT NOT NULL,
          payload_json TEXT NOT NULL,
          FOREIGN KEY(source_snapshot_id, source_entity_id)
            REFERENCES entities(snapshot_id, id),
          FOREIGN KEY(target_snapshot_id, target_entity_id)
            REFERENCES entities(snapshot_id, id)
        )""",
        """CREATE INDEX IF NOT EXISTS idx_correspondences_source
          ON entity_correspondences(source_snapshot_id, source_entity_id, kind)""",
        """CREATE INDEX IF NOT EXISTS idx_correspondences_target
          ON entity_correspondences(target_snapshot_id, target_entity_id, kind)""",
        """CREATE TABLE IF NOT EXISTS action_materializations (
          id TEXT PRIMARY KEY,
          action_id TEXT NOT NULL REFERENCES variants(id),
          parent_snapshot_id TEXT NOT NULL REFERENCES snapshots(id),
          result_snapshot_id TEXT REFERENCES snapshots(id),
          status TEXT NOT NULL,
          attempted_at TEXT NOT NULL,
          payload_json TEXT NOT NULL
        )""",
        """CREATE INDEX IF NOT EXISTS idx_materializations_action
          ON action_materializations(action_id, attempted_at, id)""",
    )
    for statement in statements:
        connection.execute(statement)
    _require_columns(connection, "entity_correspondences", frozenset({
        "id", "source_snapshot_id", "source_entity_id", "target_snapshot_id",
        "target_entity_id", "kind", "payload_json",
    }))
    _require_columns(connection, "action_materializations", frozenset({
        "id", "action_id", "parent_snapshot_id", "result_snapshot_id", "status",
        "attempted_at", "payload_json",
    }))

    # Old derivations contain only input_observation_ids.  Constructing the v0.2
    # model adds typed observation EvidenceRefs while deliberately preserving the
    # legacy stable ID algorithm for this exact case.
    for row in connection.execute(
        "SELECT id,payload_json FROM derivations ORDER BY id"
    ).fetchall():
        try:
            payload = json.loads(row[1])
            if not isinstance(payload, dict):
                raise ValueError("payload is not an object")
            migrated = Derivation.from_dict(payload)
            canonical = json_ready(migrated)
            expected = Derivation.from_dict({**canonical, "id": ""}).id
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid v0.1 derivation {row[0]!r}: {exc}") from exc
        if migrated.id != row[0] or expected != row[0]:
            raise ValueError(
                f"v0.1 derivation {row[0]!r} would change identity during migration"
            )
        if any(
            reference.kind.value != "observation"
            or reference.snapshot_id != migrated.snapshot_id
            or connection.execute(
                "SELECT 1 FROM observations WHERE snapshot_id=? AND id=?",
                (migrated.snapshot_id, reference.target_id),
            ).fetchone() is None
            for reference in migrated.evidence_refs
        ):
            raise ValueError(
                f"v0.1 derivation {row[0]!r} has unresolved observation evidence"
            )
        connection.execute(
            "UPDATE derivations SET payload_json=? WHERE id=?",
            (json.dumps(canonical, ensure_ascii=False, sort_keys=True,
                        separators=(",", ":"), allow_nan=False), row[0]),
        )

    unexpected_views = connection.execute(
        "SELECT DISTINCT schema_version FROM graph_views "
        "WHERE schema_version NOT IN (?,?)", (_V01, _V02),
    ).fetchall()
    if unexpected_views:
        raise ValueError(
            "v0.1 ledger contains graph views with unsupported schema markers: "
            + ", ".join(sorted(str(row[0]) for row in unexpected_views))
        )
    connection.execute(
        "UPDATE graph_views SET schema_version=? WHERE schema_version=?", (_V02, _V01),
    )
    violations = connection.execute("PRAGMA foreign_key_check").fetchall()
    if violations:
        raise ValueError("v0.1 ledger contains foreign-key violations")


def _migrate_v02_to_v03(connection: sqlite3.Connection) -> None:
    """Add knowledge inventory/binding indexes without rewriting old facts.

    In particular, v0.2 ``graph_views.schema_version`` values remain untouched:
    the marker participates in ``CanonicalGraph.graph_hash``.  The v0.3 reader
    treats those projections as immutable, read-only historical graphs.
    """
    for table in ("schema_info", "snapshots", "graph_views"):
        if connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,),
        ).fetchone() is None:
            raise ValueError(f"v0.2 ledger is missing required table {table!r}")

    # Some early v0.1 fixtures legitimately had no knowledge table.  Creating
    # the v0.2 table shape here keeps the chained 0.1 -> 0.2 -> 0.3 migration
    # additive while leaving every existing row byte-for-byte unchanged.
    connection.execute("""CREATE TABLE IF NOT EXISTS knowledge_rules (
      id TEXT PRIMARY KEY,
      document_id TEXT NOT NULL,
      document_version TEXT NOT NULL,
      section TEXT NOT NULL,
      payload_json TEXT NOT NULL
    )""")
    statements = (
        """CREATE TABLE IF NOT EXISTS index_commit_receipts (
          id TEXT PRIMARY KEY,
          run_id TEXT NOT NULL UNIQUE REFERENCES runs(id),
          snapshot_id TEXT NOT NULL UNIQUE REFERENCES snapshots(id),
          payload_json TEXT NOT NULL
        )""",
        """CREATE TABLE IF NOT EXISTS execution_attestations (
          id TEXT PRIMARY KEY,
          run_id TEXT NOT NULL UNIQUE REFERENCES runs(id),
          snapshot_id TEXT NOT NULL REFERENCES snapshots(id),
          payload_json TEXT NOT NULL
        )""",
        """CREATE TABLE IF NOT EXISTS execution_commit_receipts (
          id TEXT PRIMARY KEY,
          attestation_id TEXT NOT NULL UNIQUE REFERENCES execution_attestations(id),
          run_id TEXT NOT NULL UNIQUE REFERENCES runs(id),
          snapshot_id TEXT NOT NULL REFERENCES snapshots(id),
          payload_json TEXT NOT NULL
        )""",
        """CREATE TABLE IF NOT EXISTS knowledge_packs (
          pack_id TEXT PRIMARY KEY,
          pack_schema_version TEXT NOT NULL,
          content_hash TEXT NOT NULL,
          installed_at TEXT NOT NULL,
          payload_json TEXT NOT NULL
        )""",
        """CREATE TABLE IF NOT EXISTS knowledge_bindings (
          id TEXT PRIMARY KEY,
          knowledge_rule_id TEXT NOT NULL REFERENCES knowledge_rules(id),
          target_kind TEXT NOT NULL,
          target TEXT NOT NULL,
          payload_json TEXT NOT NULL
        )""",
        """CREATE INDEX IF NOT EXISTS idx_knowledge_bindings_target
          ON knowledge_bindings(target_kind, target, knowledge_rule_id)""",
        """CREATE TABLE IF NOT EXISTS knowledge_coverage (
          id TEXT PRIMARY KEY,
          pack_id TEXT NOT NULL REFERENCES knowledge_packs(pack_id),
          coverage_scope TEXT NOT NULL,
          payload_json TEXT NOT NULL
        )""",
        """CREATE INDEX IF NOT EXISTS idx_knowledge_coverage_pack
          ON knowledge_coverage(pack_id, coverage_scope)""",
    )
    for statement in statements:
        connection.execute(statement)
    _require_columns(connection, "index_commit_receipts", frozenset({
        "id", "run_id", "snapshot_id", "payload_json",
    }))
    _require_columns(connection, "execution_attestations", frozenset({
        "id", "run_id", "snapshot_id", "payload_json",
    }))
    _require_columns(connection, "execution_commit_receipts", frozenset({
        "id", "attestation_id", "run_id", "snapshot_id", "payload_json",
    }))
    try:
        connection.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_rules_fts USING fts5("
            "knowledge_rule_id UNINDEXED, rule_id, document_id, document_version, "
            "section, title, summary)"
        )
        connection.execute("DELETE FROM knowledge_rules_fts")
        for row in connection.execute(
            "SELECT id,document_id,document_version,section,payload_json "
            "FROM knowledge_rules ORDER BY id"
        ).fetchall():
            payload = json.loads(row[4])
            connection.execute(
                "INSERT INTO knowledge_rules_fts("
                "knowledge_rule_id,rule_id,document_id,document_version,section,title,summary) "
                "VALUES(?,?,?,?,?,?,?)",
                (row[0], payload.get("rule_id", ""), row[1], row[2], row[3],
                 payload.get("title", ""), payload.get("summary") or ""),
            )
        connection.execute(
            "INSERT OR REPLACE INTO schema_info(key,value) "
            "VALUES('knowledge_fts5','1')"
        )
    except (sqlite3.OperationalError, json.JSONDecodeError) as exc:
        if isinstance(exc, json.JSONDecodeError):
            raise ValueError(f"invalid v0.2 knowledge rule payload: {exc}") from exc
        connection.execute(
            "INSERT OR REPLACE INTO schema_info(key,value) "
            "VALUES('knowledge_fts5','0')"
        )

    unexpected_views = connection.execute(
        "SELECT DISTINCT schema_version FROM graph_views "
        "WHERE schema_version NOT IN (?,?)", (_V02, _V03),
    ).fetchall()
    if unexpected_views:
        raise ValueError(
            "v0.2 ledger contains graph views with unsupported schema markers: "
            + ", ".join(sorted(str(row[0]) for row in unexpected_views))
        )
    violations = connection.execute("PRAGMA foreign_key_check").fetchall()
    if violations:
        raise ValueError("v0.2 ledger contains foreign-key violations")


# Migration is explicit and append-only.  It supplements legacy derivation
# payloads but never changes observation meaning, historical manifests, or IDs.
MIGRATIONS: tuple[MigrationStep, ...] = (
    MigrationStep(
        from_version=_V01,
        to_version=_V02,
        description=(
            "add typed feature evidence, entity correspondence, and action "
            "materialization contracts"
        ),
        apply=_migrate_v01_to_v02,
    ),
    MigrationStep(
        from_version=_V02,
        to_version=_V03,
        description=(
            "add knowledge pack inventory, fail-closed bindings, coverage "
            "manifests, and rebuildable knowledge FTS"
        ),
        apply=_migrate_v02_to_v03,
    ),
)


def migration_path(from_version: str, to_version: str = SCHEMA_VERSION) -> list[MigrationStep]:
    if from_version == to_version:
        return []
    by_source = {step.from_version: step for step in MIGRATIONS}
    result: list[MigrationStep] = []
    current = from_version
    visited: set[str] = set()
    while current != to_version:
        if current in visited or current not in by_source:
            raise ValueError(
                f"no explicit HLSGraph ledger migration from {from_version!r} to {to_version!r}"
            )
        visited.add(current)
        step = by_source[current]
        result.append(step)
        current = step.to_version
    return result


def apply_migrations(connection: sqlite3.Connection, from_version: str,
                     to_version: str = SCHEMA_VERSION) -> list[MigrationStep]:
    steps = migration_path(from_version, to_version)
    for step in steps:
        step.apply(connection)
        connection.execute(
            "UPDATE schema_info SET value=? WHERE key='schema_version'", (step.to_version,)
        )
    return steps
