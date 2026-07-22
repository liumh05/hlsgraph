"""Append-only SQLite ledger for artifacts, runs, observations, and graph projections."""
from __future__ import annotations

import json
import math
import sqlite3
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from ..evidence_policy import (
    TOOL_EVIDENCE_POLICY_VERSION,
    execution_attestation_error,
    execution_commit_receipt_error,
    real_tool_run_claim_error,
    run_claims_tool_truth,
    tool_evidence_compatibility_error,
    tool_run_manifest_identity_error,
)
from ..graph import CanonicalGraph
from ..model import (
    ActionMaterialization,
    ArtifactRef,
    AuthorityClass,
    Derivation,
    DesignSnapshot,
    Diagnostic,
    Entity,
    EntityCorrespondence,
    ExecutionAttestation,
    ExecutionCommitReceipt,
    EvidenceKind,
    EvidenceRef,
    CoverageStatus,
    CoverageManifest,
    KnowledgeBinding,
    KnowledgeRule,
    Observation,
    PredictionEnvelope,
    ProjectManifest,
    Relation,
    ToolRun,
    VariantAction,
    VerificationResult,
    artifact_hash_map,
    hash_artifact_bytes,
    json_ready,
    reject_embedded_body_fields,
    stable_hash,
    utc_now,
)
from ..knowledge.supported_targets import canonical_supported_targets
from ..runner.core import _consume_execution_authorization, RunnerProtocolError
from ..runner.staging import StagingError, read_verified_file
from ..version import SCHEMA_VERSION, SUPPORTED_GRAPH_SCHEMA_VERSIONS
from .migrations import apply_migrations, migration_path


class StoreError(RuntimeError):
    pass


_NON_FACT_AUTHORITIES = frozenset({
    AuthorityClass.KNOWLEDGE_RULE,
    AuthorityClass.PREDICTION_HYPOTHESIS,
})


# These are parser-backed report contracts, not prefix heuristics.  Public fixtures
# must use the same kinds and scope metadata as real reports; a convenient alias or
# arbitrary vendor-namespaced binary is deliberately not enough.
_VERIFICATION_REPORT_POLICY: dict[str, dict[str, frozenset[str] | str]] = {
    "csim": {
        "run_stage": "csim",
        "observation_stages": frozenset({"csim"}),
        "artifact_kinds": frozenset({
            "amd.vitis.csim_result",
        }),
    },
    "rtl_cosim": {
        "run_stage": "rtl_cosim",
        "observation_stages": frozenset({"cosim"}),
        "artifact_kinds": frozenset({
            "amd.vitis.cosim_rpt",
            "amd.vitis.cosim_report",
        }),
    },
}

_PHYSICAL_GATE_REPORT_KINDS: dict[str, frozenset[str]] = {
    "gate.resource_fits": frozenset({
        "amd.vivado.post_route_utilization",
        "amd.vivado.utilization",
    }),
    "gate.post_route_timing": frozenset({
        "amd.vivado.post_route_timing",
        "amd.vivado.timing_summary",
    }),
}

_TOOL_EVIDENCE_AUTHORITIES = frozenset({
    AuthorityClass.TOOL_OBSERVATION,
    AuthorityClass.VERIFICATION_EVIDENCE,
    AuthorityClass.PHYSICAL_MEASUREMENT,
})


DDL = """
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS schema_info (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS projects (
  project_id TEXT PRIMARY KEY,
  manifest_hash TEXT NOT NULL,
  manifest_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS snapshots (
  id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL REFERENCES projects(project_id),
  created_at TEXT NOT NULL,
  payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_snapshots_project ON snapshots(project_id, created_at);
CREATE TABLE IF NOT EXISTS snapshot_manifests (
  snapshot_id TEXT PRIMARY KEY REFERENCES snapshots(id),
  manifest_hash TEXT NOT NULL,
  manifest_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS project_state (
  project_id TEXT PRIMARY KEY REFERENCES projects(project_id),
  active_snapshot_id TEXT REFERENCES snapshots(id),
  updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS graph_views (
  snapshot_id TEXT PRIMARY KEY REFERENCES snapshots(id),
  schema_version TEXT NOT NULL,
  metadata_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS artifacts (
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  uri TEXT NOT NULL,
  sha256 TEXT NOT NULL,
  size INTEGER NOT NULL,
  payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS snapshot_artifacts (
  snapshot_id TEXT NOT NULL REFERENCES snapshots(id),
  artifact_id TEXT NOT NULL REFERENCES artifacts(id),
  role TEXT,
  PRIMARY KEY(snapshot_id, artifact_id)
);
CREATE TABLE IF NOT EXISTS entities (
  id TEXT NOT NULL,
  snapshot_id TEXT NOT NULL REFERENCES snapshots(id),
  kind TEXT NOT NULL,
  name TEXT NOT NULL,
  qualified_name TEXT,
  stage TEXT NOT NULL,
  authority TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  PRIMARY KEY(snapshot_id, id)
);
CREATE INDEX IF NOT EXISTS idx_entities_snapshot_kind ON entities(snapshot_id, kind);
CREATE TABLE IF NOT EXISTS relations (
  id TEXT NOT NULL,
  snapshot_id TEXT NOT NULL REFERENCES snapshots(id),
  src TEXT NOT NULL,
  dst TEXT NOT NULL,
  kind TEXT NOT NULL,
  stage TEXT NOT NULL,
  authority TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  PRIMARY KEY(snapshot_id, id),
  FOREIGN KEY(snapshot_id, src) REFERENCES entities(snapshot_id, id),
  FOREIGN KEY(snapshot_id, dst) REFERENCES entities(snapshot_id, id)
);
CREATE INDEX IF NOT EXISTS idx_relations_endpoints ON relations(snapshot_id, src, dst);
CREATE TABLE IF NOT EXISTS entity_correspondences (
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
);
CREATE INDEX IF NOT EXISTS idx_correspondences_source
  ON entity_correspondences(source_snapshot_id, source_entity_id, kind);
CREATE INDEX IF NOT EXISTS idx_correspondences_target
  ON entity_correspondences(target_snapshot_id, target_entity_id, kind);
CREATE TABLE IF NOT EXISTS observations (
  id TEXT PRIMARY KEY,
  snapshot_id TEXT NOT NULL REFERENCES snapshots(id),
  subject_id TEXT NOT NULL,
  predicate TEXT NOT NULL,
  stage TEXT NOT NULL,
  authority TEXT NOT NULL,
  run_id TEXT,
  artifact_id TEXT,
  payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_observations_subject ON observations(snapshot_id, subject_id, predicate);
CREATE TABLE IF NOT EXISTS derivations (
  id TEXT PRIMARY KEY,
  snapshot_id TEXT NOT NULL REFERENCES snapshots(id),
  subject_id TEXT NOT NULL,
  predicate TEXT NOT NULL,
  payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY,
  snapshot_id TEXT NOT NULL REFERENCES snapshots(id),
  stage TEXT NOT NULL,
  status TEXT NOT NULL,
  request_hash TEXT NOT NULL,
  payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_snapshot_stage ON runs(snapshot_id, stage, status);
CREATE TABLE IF NOT EXISTS execution_attestations (
  id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL UNIQUE REFERENCES runs(id),
  snapshot_id TEXT NOT NULL REFERENCES snapshots(id),
  payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS execution_commit_receipts (
  id TEXT PRIMARY KEY,
  attestation_id TEXT NOT NULL UNIQUE REFERENCES execution_attestations(id),
  run_id TEXT NOT NULL UNIQUE REFERENCES runs(id),
  snapshot_id TEXT NOT NULL REFERENCES snapshots(id),
  payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS diagnostics (
  id TEXT PRIMARY KEY,
  snapshot_id TEXT NOT NULL REFERENCES snapshots(id),
  code TEXT NOT NULL,
  severity TEXT NOT NULL,
  stage TEXT NOT NULL,
  payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS verifications (
  id TEXT PRIMARY KEY,
  snapshot_id TEXT NOT NULL REFERENCES snapshots(id),
  kind TEXT NOT NULL,
  status TEXT NOT NULL,
  payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS variants (
  id TEXT PRIMARY KEY,
  parent_snapshot_id TEXT NOT NULL,
  kind TEXT NOT NULL,
  payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS action_materializations (
  id TEXT PRIMARY KEY,
  action_id TEXT NOT NULL REFERENCES variants(id),
  parent_snapshot_id TEXT NOT NULL REFERENCES snapshots(id),
  result_snapshot_id TEXT REFERENCES snapshots(id),
  status TEXT NOT NULL,
  attempted_at TEXT NOT NULL,
  payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_materializations_action
  ON action_materializations(action_id, attempted_at, id);
CREATE TABLE IF NOT EXISTS knowledge_rules (
  id TEXT PRIMARY KEY,
  document_id TEXT NOT NULL,
  document_version TEXT NOT NULL,
  section TEXT NOT NULL,
  payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS knowledge_packs (
  pack_id TEXT PRIMARY KEY,
  pack_schema_version TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  installed_at TEXT NOT NULL,
  payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS knowledge_bindings (
  id TEXT PRIMARY KEY,
  knowledge_rule_id TEXT NOT NULL REFERENCES knowledge_rules(id),
  target_kind TEXT NOT NULL,
  target TEXT NOT NULL,
  payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_knowledge_bindings_target
  ON knowledge_bindings(target_kind, target, knowledge_rule_id);
CREATE TABLE IF NOT EXISTS knowledge_coverage (
  id TEXT PRIMARY KEY,
  pack_id TEXT NOT NULL REFERENCES knowledge_packs(pack_id),
  coverage_scope TEXT NOT NULL,
  payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_knowledge_coverage_pack
  ON knowledge_coverage(pack_id, coverage_scope);
CREATE TABLE IF NOT EXISTS predictions (
  id TEXT PRIMARY KEY,
  snapshot_id TEXT NOT NULL REFERENCES snapshots(id),
  subject_id TEXT NOT NULL,
  predicate TEXT NOT NULL,
  payload_json TEXT NOT NULL
);
"""


class LedgerStore:
    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as connection:
            table = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_info'"
            ).fetchone()
            if table:
                row = connection.execute(
                    "SELECT value FROM schema_info WHERE key='schema_version'"
                ).fetchone()
                if row is None:
                    raise StoreError("ledger schema marker is missing; refusing an implicit migration")
                if row[0] != SCHEMA_VERSION:
                    raise StoreError(
                        f"ledger schema {row[0]!r} is not supported by this build "
                        f"({SCHEMA_VERSION!r}); run an explicit migration"
                    )
            connection.executescript(DDL)
            connection.execute("INSERT OR IGNORE INTO schema_info(key,value) VALUES(?,?)",
                               ("schema_version", SCHEMA_VERSION))
            try:
                connection.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS entities_fts USING fts5("
                    "snapshot_id UNINDEXED, entity_id UNINDEXED, name, qualified_name, aliases, attrs)"
                )
                connection.execute("INSERT OR REPLACE INTO schema_info(key,value) VALUES('fts5','1')")
            except sqlite3.OperationalError:
                connection.execute("INSERT OR REPLACE INTO schema_info(key,value) VALUES('fts5','0')")
            try:
                connection.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_rules_fts USING fts5("
                    "knowledge_rule_id UNINDEXED, rule_id, document_id, document_version, "
                    "section, title, summary)"
                )
                connection.execute(
                    "INSERT OR REPLACE INTO schema_info(key,value) "
                    "VALUES('knowledge_fts5','1')"
                )
            except sqlite3.OperationalError:
                connection.execute(
                    "INSERT OR REPLACE INTO schema_info(key,value) "
                    "VALUES('knowledge_fts5','0')"
                )

    def migration_plan(self, to_version: str = SCHEMA_VERSION) -> list[dict[str, str]]:
        if not self.path.is_file():
            return []
        with sqlite3.connect(self.path) as connection:
            table = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_info'"
            ).fetchone()
            if not table:
                raise StoreError("database is not an HLSGraph ledger")
            row = connection.execute(
                "SELECT value FROM schema_info WHERE key='schema_version'"
            ).fetchone()
            if row is None:
                raise StoreError("ledger schema marker is missing")
            try:
                steps = migration_path(str(row[0]), to_version)
            except (ValueError, sqlite3.DatabaseError) as exc:
                raise StoreError(str(exc)) from exc
            return [{"from_version": step.from_version, "to_version": step.to_version,
                     "description": step.description} for step in steps]

    def migrate(self, to_version: str = SCHEMA_VERSION) -> list[dict[str, str]]:
        """Apply only registered migrations; callers must invoke this explicitly."""
        if not self.path.is_file():
            raise StoreError(f"ledger does not exist: {self.path}")
        with sqlite3.connect(self.path) as connection:
            connection.execute("PRAGMA foreign_keys=ON")
            row = connection.execute(
                "SELECT value FROM schema_info WHERE key='schema_version'"
            ).fetchone()
            if row is None:
                raise StoreError("ledger schema marker is missing")
            try:
                steps = apply_migrations(connection, str(row[0]), to_version)
            except (ValueError, sqlite3.DatabaseError) as exc:
                raise StoreError(str(exc)) from exc
            connection.commit()
            return [{"from_version": step.from_version, "to_version": step.to_version,
                     "description": step.description} for step in steps]

    @contextmanager
    def write(self) -> Iterator[sqlite3.Connection]:
        self.initialize()
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @contextmanager
    def read(self) -> Iterator[sqlite3.Connection]:
        if not self.path.is_file():
            raise StoreError(f"ledger does not exist: {self.path}")
        uri = self.path.as_uri() + "?mode=ro"
        connection = sqlite3.connect(uri, uri=True)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        try:
            version = connection.execute(
                "SELECT value FROM schema_info WHERE key='schema_version'"
            ).fetchone()
            if version is None:
                raise StoreError("database is not an HLSGraph ledger")
            if version[0] != SCHEMA_VERSION:
                raise StoreError(
                    f"ledger schema {version[0]!r} is not supported by this build "
                    f"({SCHEMA_VERSION!r}); run an explicit migration"
                )
            yield connection
        finally:
            connection.close()

    @staticmethod
    def _payload(value: Any) -> str:
        return json.dumps(json_ready(value), ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False)

    @staticmethod
    def _revalidate_model(value: Any, model_type: Any, label: str) -> Any:
        """Re-run model guards and deterministic IDs at the write boundary."""
        data = json_ready(value)

        def build(payload: dict[str, Any]) -> Any:
            factory = getattr(model_type, "from_dict", None)
            return factory(payload) if factory else model_type(**payload)

        try:
            rebuilt = build(dict(data))
            if json_ready(rebuilt) != data:
                raise ValueError("payload is not in canonical normalized form")
            if "id" in data:
                identity = dict(data)
                identity["id"] = ""
                expected = build(identity).id
                if data.get("id") != expected:
                    raise ValueError(
                        f"stable id {data.get('id')!r} does not match {expected!r}"
                    )
            return rebuilt
        except (KeyError, TypeError, ValueError) as exc:
            raise StoreError(f"invalid {label}: {exc}") from exc

    @staticmethod
    def _immutable_payload(connection: sqlite3.Connection, table: str, item_id: str,
                           payload: str) -> bool:
        """Return True when an identical row exists; reject semantic replacement."""
        allowed = {
            "observations", "derivations", "runs", "diagnostics", "verifications",
            "variants", "predictions", "knowledge_rules", "entity_correspondences",
            "action_materializations", "knowledge_bindings", "knowledge_coverage",
            "execution_attestations", "execution_commit_receipts",
        }
        if table not in allowed:
            raise StoreError(f"unsupported immutable table: {table}")
        previous = connection.execute(
            f"SELECT payload_json FROM {table} WHERE id=?", (item_id,)
        ).fetchone()
        if previous and previous[0] != payload and table == "predictions":
            # v0.1.x added an optional action_id without changing legacy
            # prediction identities.  Treat an absent key and an explicit null
            # as the same payload so a pre-upgrade row can be re-recorded
            # idempotently after deserialization.
            old_value = json.loads(previous[0])
            new_value = json.loads(payload)
            if old_value.get("action_id") is None and new_value.get("action_id") is None:
                old_value.pop("action_id", None)
                new_value.pop("action_id", None)
                if old_value == new_value:
                    return True
        if previous and previous[0] != payload:
            raise StoreError(f"immutable {table} row changed: {item_id}")
        return previous is not None

    @staticmethod
    def _require_subject(connection: sqlite3.Connection, snapshot_id: str,
                         subject_id: str) -> None:
        entity = connection.execute(
            "SELECT 1 FROM entities WHERE snapshot_id=? AND id=?", (snapshot_id, subject_id)
        ).fetchone()
        artifact = connection.execute(
            "SELECT 1 FROM snapshot_artifacts WHERE snapshot_id=? AND artifact_id=?",
            (snapshot_id, subject_id),
        ).fetchone()
        if not entity and not artifact:
            raise StoreError(
                f"subject {subject_id!r} does not exist in snapshot {snapshot_id}"
            )

    @staticmethod
    def _require_artifact(connection: sqlite3.Connection, snapshot_id: str,
                          artifact_id: str) -> None:
        if not connection.execute(
            "SELECT 1 FROM snapshot_artifacts WHERE snapshot_id=? AND artifact_id=?",
            (snapshot_id, artifact_id),
        ).fetchone():
            raise StoreError(
                f"artifact {artifact_id!r} is not attached to snapshot {snapshot_id}"
            )

    @staticmethod
    def _require_run(connection: sqlite3.Connection, snapshot_id: str, run_id: str) -> None:
        if not connection.execute(
            "SELECT 1 FROM runs WHERE snapshot_id=? AND id=?", (snapshot_id, run_id)
        ).fetchone():
            raise StoreError(f"run {run_id!r} does not exist in snapshot {snapshot_id}")

    @staticmethod
    def _require_fact_authority(authority: AuthorityClass, subject: str) -> None:
        if str(authority) in {item.value for item in _NON_FACT_AUTHORITIES}:
            raise StoreError(
                f"{subject} cannot use {str(authority)!r} authority; "
                "knowledge rules and predictions belong in dedicated tables"
            )

    @staticmethod
    def _require_artifact_producer(
        connection: sqlite3.Connection,
        snapshot_id: str,
        artifact: ArtifactRef,
        *,
        future_run_ids: frozenset[str] = frozenset(),
    ) -> None:
        """Require producers in the same snapshot; cross-snapshot lineage is not implicit.

        An artifact without ``producer_run_id`` is an input/reference artifact.  Once a
        producer is declared, the run must belong to the snapshot receiving the artifact.
        Reusing produced bytes in another snapshot requires a new ArtifactRef without an
        inherited producer, or an explicit import run in that snapshot.
        """
        producer = artifact.producer_run_id
        if not producer:
            return
        if producer in future_run_ids:
            return
        row = connection.execute(
            "SELECT payload_json FROM runs WHERE snapshot_id=? AND id=?",
            (snapshot_id, producer),
        ).fetchone()
        if row:
            run = ToolRun.from_dict(json.loads(row[0]))
            if artifact.id not in run.output_artifact_ids:
                raise StoreError(
                    f"artifact {artifact.id!r} names producer {producer!r} but the run "
                    "does not declare it as an output; use commit_run_result"
                )
            if artifact.id in run.input_artifact_ids:
                raise StoreError("an artifact cannot be both input and output of its producer run")
            return
        elsewhere = connection.execute(
            "SELECT snapshot_id FROM runs WHERE id=?", (producer,)
        ).fetchone()
        if elsewhere:
            raise StoreError(
                f"artifact producer run {producer!r} belongs to snapshot {elsewhere[0]}; "
                "cross-snapshot producer references are not supported"
            )
        raise StoreError(
            f"artifact producer run {producer!r} does not exist in snapshot {snapshot_id}"
        )

    @staticmethod
    def _evidence_exists(
        connection: sqlite3.Connection,
        snapshot_id: str,
        evidence_id: str,
        *,
        future_evidence_ids: frozenset[str] = frozenset(),
    ) -> bool:
        if evidence_id in future_evidence_ids:
            return True
        for table in ("observations", "derivations", "verifications", "diagnostics", "runs"):
            if connection.execute(
                f"SELECT 1 FROM {table} WHERE snapshot_id=? AND id=?",
                (snapshot_id, evidence_id),
            ).fetchone():
                return True
        return connection.execute(
            "SELECT 1 FROM snapshot_artifacts WHERE snapshot_id=? AND artifact_id=?",
            (snapshot_id, evidence_id),
        ).fetchone() is not None

    @staticmethod
    def _evidence_kind(connection: sqlite3.Connection, snapshot_id: str,
                       evidence_id: str) -> str | None:
        for table, kind in (
            ("observations", "observation"), ("derivations", "derivation"),
            ("verifications", "verification"), ("diagnostics", "diagnostic"),
            ("runs", "run"),
        ):
            if connection.execute(
                f"SELECT 1 FROM {table} WHERE snapshot_id=? AND id=?",
                (snapshot_id, evidence_id),
            ).fetchone():
                return kind
        if connection.execute(
            "SELECT 1 FROM snapshot_artifacts WHERE snapshot_id=? AND artifact_id=?",
            (snapshot_id, evidence_id),
        ).fetchone():
            return "artifact"
        return None

    @classmethod
    def _resolve_evidence_ref(
        cls,
        connection: sqlite3.Connection,
        reference: EvidenceRef,
        *,
        allowed_snapshot_ids: frozenset[str],
        label: str,
    ) -> str:
        """Resolve one typed reference in exactly one permitted snapshot.

        An omitted snapshot qualifier is convenience, not permission to guess:
        the target must resolve in exactly one allowed snapshot.  The kind is
        checked against its own namespace, so a coincidentally equal ID in a
        different ledger table cannot satisfy the reference.
        """
        if not allowed_snapshot_ids:
            raise StoreError("evidence resolution requires at least one allowed snapshot")
        if (reference.snapshot_id is not None
                and reference.snapshot_id not in allowed_snapshot_ids):
            raise StoreError(
                f"{label} evidence belongs to disallowed snapshot "
                f"{reference.snapshot_id!r}"
            )
        candidates = (
            [reference.snapshot_id] if reference.snapshot_id is not None
            else sorted(allowed_snapshot_ids)
        )
        resolved: list[str] = []
        for snapshot_id in candidates:
            if snapshot_id is None:  # only for static type narrowing
                continue
            if reference.kind == EvidenceKind.OBSERVATION:
                exists = connection.execute(
                    "SELECT 1 FROM observations WHERE snapshot_id=? AND id=?",
                    (snapshot_id, reference.target_id),
                ).fetchone() is not None
            elif reference.kind == EvidenceKind.DERIVATION:
                exists = connection.execute(
                    "SELECT 1 FROM derivations WHERE snapshot_id=? AND id=?",
                    (snapshot_id, reference.target_id),
                ).fetchone() is not None
            elif reference.kind == EvidenceKind.ARTIFACT:
                exists = connection.execute(
                    "SELECT 1 FROM snapshot_artifacts WHERE snapshot_id=? AND artifact_id=?",
                    (snapshot_id, reference.target_id),
                ).fetchone() is not None
            elif reference.kind == EvidenceKind.ENTITY_ANCHOR:
                row = connection.execute(
                    "SELECT payload_json FROM entities WHERE snapshot_id=? AND id=?",
                    (snapshot_id, reference.target_id),
                ).fetchone()
                exists = row is not None
                if exists and reference.anchor is not None:
                    entity = Entity.from_dict(json.loads(row[0]))
                    exists = any(
                        cls._payload(anchor) == cls._payload(reference.anchor)
                        for anchor in entity.anchors
                    )
                    if exists:
                        cls._require_artifact(
                            connection, snapshot_id, reference.anchor.artifact_id,
                        )
            elif reference.kind == EvidenceKind.RELATION:
                row = connection.execute(
                    "SELECT payload_json FROM relations WHERE snapshot_id=? AND id=?",
                    (snapshot_id, reference.target_id),
                ).fetchone()
                exists = row is not None
                if exists and reference.anchor is not None:
                    relation = Relation.from_dict(json.loads(row[0]))
                    exists = any(
                        cls._payload(anchor) == cls._payload(reference.anchor)
                        for anchor in relation.anchors
                    )
                    if exists:
                        cls._require_artifact(
                            connection, snapshot_id, reference.anchor.artifact_id,
                        )
            else:  # pragma: no cover - EvidenceKind is closed
                exists = False
            if exists:
                resolved.append(snapshot_id)
        if not resolved:
            raise StoreError(
                f"{label} {reference.kind.value} evidence "
                f"{reference.target_id!r} does not exist in an allowed snapshot"
            )
        if len(resolved) != 1:
            raise StoreError(
                f"{label} evidence {reference.target_id!r} is ambiguous across snapshots; "
                "set EvidenceRef.snapshot_id explicitly"
            )
        return resolved[0]

    @classmethod
    def _evidence_producer_runs(
        cls, connection: sqlite3.Connection, snapshot_id: str, evidence_id: str,
        stack: frozenset[str] = frozenset(),
    ) -> set[str]:
        if evidence_id in stack:
            raise StoreError("cyclic evidence producer lineage")
        nested = stack | {evidence_id}
        row = connection.execute(
            "SELECT payload_json FROM observations WHERE snapshot_id=? AND id=?",
            (snapshot_id, evidence_id),
        ).fetchone()
        if row:
            item = Observation.from_dict(json.loads(row[0]))
            producers = {item.run_id} if item.run_id else set()
            for artifact_id in {value for value in (
                item.artifact_id,
                item.anchor.artifact_id if item.anchor else None,
            ) if value}:
                producers.update(cls._evidence_producer_runs(
                    connection, snapshot_id, artifact_id, nested,
                ))
            return producers
        row = connection.execute(
            "SELECT payload_json FROM derivations WHERE snapshot_id=? AND id=?",
            (snapshot_id, evidence_id),
        ).fetchone()
        if row:
            value = Derivation.from_dict(json.loads(row[0]))
            producers: set[str] = set()
            for reference in value.evidence_refs:
                if reference.kind in {
                    EvidenceKind.OBSERVATION,
                    EvidenceKind.DERIVATION,
                    EvidenceKind.ARTIFACT,
                }:
                    producers.update(cls._evidence_producer_runs(
                        connection, snapshot_id, reference.target_id, nested,
                    ))
                elif reference.anchor is not None:
                    producers.update(cls._evidence_producer_runs(
                        connection, snapshot_id, reference.anchor.artifact_id, nested,
                    ))
            return producers
        row = connection.execute(
            "SELECT payload_json FROM verifications WHERE snapshot_id=? AND id=?",
            (snapshot_id, evidence_id),
        ).fetchone()
        if row:
            value = json.loads(row[0])
            producers = {str(value["run_id"])} if value.get("run_id") else set()
            for nested_id in value.get("evidence_ids", []):
                producers.update(cls._evidence_producer_runs(
                    connection, snapshot_id, str(nested_id), nested,
                ))
            return producers
        row = connection.execute(
            "SELECT a.payload_json FROM artifacts a JOIN snapshot_artifacts sa "
            "ON a.id=sa.artifact_id WHERE sa.snapshot_id=? AND a.id=?",
            (snapshot_id, evidence_id),
        ).fetchone()
        if row:
            producer = ArtifactRef.from_dict(json.loads(row[0])).producer_run_id
            return {producer} if producer else set()
        return set()

    @classmethod
    def _evidence_workloads(
        cls, connection: sqlite3.Connection, snapshot_id: str, evidence_id: str,
        stack: frozenset[str] = frozenset(),
    ) -> set[str]:
        if evidence_id in stack:
            raise StoreError("cyclic evidence workload lineage")
        nested = stack | {evidence_id}
        row = connection.execute(
            "SELECT payload_json FROM observations WHERE snapshot_id=? AND id=?",
            (snapshot_id, evidence_id),
        ).fetchone()
        if row:
            item = Observation.from_dict(json.loads(row[0]))
            return {item.workload_id} if item.workload_id else set()
        row = connection.execute(
            "SELECT payload_json FROM derivations WHERE snapshot_id=? AND id=?",
            (snapshot_id, evidence_id),
        ).fetchone()
        if row:
            values: set[str] = set()
            item = Derivation.from_dict(json.loads(row[0]))
            for reference in item.evidence_refs:
                if reference.kind in {
                    EvidenceKind.OBSERVATION,
                    EvidenceKind.DERIVATION,
                    EvidenceKind.ARTIFACT,
                }:
                    values.update(cls._evidence_workloads(
                        connection, snapshot_id, reference.target_id, nested,
                    ))
                elif reference.anchor is not None:
                    values.update(cls._evidence_workloads(
                        connection, snapshot_id, reference.anchor.artifact_id, nested,
                    ))
            return values
        row = connection.execute(
            "SELECT payload_json FROM verifications WHERE snapshot_id=? AND id=?",
            (snapshot_id, evidence_id),
        ).fetchone()
        if row:
            value = json.loads(row[0])
            values: set[str] = set()
            if value.get("workload_id"):
                values.add(str(value["workload_id"]))
            for nested_id in value.get("evidence_ids", []):
                values.update(cls._evidence_workloads(
                    connection, snapshot_id, str(nested_id), nested,
                ))
            return values
        row = connection.execute(
            "SELECT a.payload_json FROM artifacts a JOIN snapshot_artifacts sa "
            "ON a.id=sa.artifact_id WHERE sa.snapshot_id=? AND a.id=?",
            (snapshot_id, evidence_id),
        ).fetchone()
        if row:
            metadata = ArtifactRef.from_dict(json.loads(row[0])).metadata
            workload = metadata.get("workload_id")
            return {str(workload)} if isinstance(workload, str) and workload else set()
        return set()

    @staticmethod
    def _evidence_artifact(
        connection: sqlite3.Connection, snapshot_id: str, artifact_id: str,
    ) -> ArtifactRef | None:
        row = connection.execute(
            "SELECT a.payload_json FROM artifacts a JOIN snapshot_artifacts sa "
            "ON a.id=sa.artifact_id WHERE sa.snapshot_id=? AND a.id=?",
            (snapshot_id, artifact_id),
        ).fetchone()
        return ArtifactRef.from_dict(json.loads(row[0])) if row else None

    @classmethod
    def _evidence_observation_leaves(
        cls, connection: sqlite3.Connection, snapshot_id: str, evidence_id: str,
        stack: frozenset[str] = frozenset(),
    ) -> list[Observation]:
        """Return observation leaves without treating a report file as an observation.

        Verification and gate decisions need semantic observations, not merely the
        existence of an opaque output file.  Direct typed report artifacts remain valid
        verification evidence, while physical boolean gates require observation leaves.
        """
        if evidence_id in stack:
            raise StoreError("cyclic evidence observation lineage")
        nested = stack | {evidence_id}
        row = connection.execute(
            "SELECT payload_json FROM observations WHERE snapshot_id=? AND id=?",
            (snapshot_id, evidence_id),
        ).fetchone()
        if row:
            return [Observation.from_dict(json.loads(row[0]))]
        row = connection.execute(
            "SELECT payload_json FROM derivations WHERE snapshot_id=? AND id=?",
            (snapshot_id, evidence_id),
        ).fetchone()
        if row:
            result: list[Observation] = []
            item = Derivation.from_dict(json.loads(row[0]))
            for reference in item.evidence_refs:
                if reference.kind in {
                    EvidenceKind.OBSERVATION, EvidenceKind.DERIVATION,
                }:
                    result.extend(cls._evidence_observation_leaves(
                        connection, snapshot_id, reference.target_id, nested,
                    ))
            return result
        return []

    @classmethod
    def _observation_artifacts(
        cls, connection: sqlite3.Connection, snapshot_id: str, item: Observation,
    ) -> list[ArtifactRef]:
        artifact_ids = {value for value in (
            item.artifact_id,
            item.anchor.artifact_id if item.anchor else None,
        ) if value}
        artifacts: list[ArtifactRef] = []
        for artifact_id in sorted(artifact_ids):
            artifact = cls._evidence_artifact(connection, snapshot_id, artifact_id)
            if artifact is None:
                raise StoreError(
                    f"evidence observation {item.id} cites missing artifact {artifact_id!r}"
                )
            artifacts.append(artifact)
        return artifacts

    @classmethod
    def _validate_execution_attestation(
        cls,
        connection: sqlite3.Connection,
        run: ToolRun,
        *,
        attestation: ExecutionAttestation | None = None,
        require_receipt: bool = True,
    ) -> ExecutionAttestation:
        """Validate one capability-authorized execution against live ledger state."""

        if attestation is None:
            row = connection.execute(
                "SELECT payload_json FROM execution_attestations "
                "WHERE snapshot_id=? AND run_id=?",
                (run.snapshot_id, run.id),
            ).fetchone()
            if row is None:
                raise StoreError(
                    f"trusted run {run.id} has no pipeline-issued execution attestation"
                )
            try:
                attestation = ExecutionAttestation.from_dict(json.loads(row[0]))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise StoreError(
                    f"trusted run {run.id} has an invalid execution attestation"
                ) from exc
        snapshot_row = connection.execute(
            "SELECT payload_json FROM snapshots WHERE id=?", (run.snapshot_id,),
        ).fetchone()
        if snapshot_row is None:
            raise StoreError(f"unknown run snapshot: {run.snapshot_id}")
        snapshot = DesignSnapshot.from_dict(json.loads(snapshot_row[0]))
        manifest = cls._snapshot_manifest_for_evidence(connection, run.snapshot_id)
        artifact_rows = connection.execute(
            "SELECT a.payload_json FROM artifacts a JOIN snapshot_artifacts sa "
            "ON a.id=sa.artifact_id WHERE sa.snapshot_id=?",
            (run.snapshot_id,),
        ).fetchall()
        by_id = {
            artifact.id: artifact
            for artifact in (
                ArtifactRef.from_dict(json.loads(row[0])) for row in artifact_rows
            )
        }
        output_artifacts = [
            by_id[artifact_id] for artifact_id in run.output_artifact_ids
            if artifact_id in by_id
        ]
        if len(output_artifacts) != len(run.output_artifact_ids):
            raise StoreError(
                f"trusted run {run.id} has unattached attested output artifacts"
            )
        error = execution_attestation_error(
            attestation, run, snapshot, manifest, output_artifacts,
        )
        if error is not None:
            raise StoreError(f"trusted run {run.id} execution attestation is invalid: {error}")
        if any(not cls._managed_artifact_bytes_valid(connection, artifact)
               for artifact in output_artifacts):
            raise StoreError(
                f"trusted run {run.id} attested output bytes are not live in managed CAS"
            )
        if require_receipt:
            row = connection.execute(
                "SELECT payload_json FROM execution_commit_receipts "
                "WHERE snapshot_id=? AND run_id=? AND attestation_id=?",
                (run.snapshot_id, run.id, attestation.id),
            ).fetchone()
            if row is None:
                raise StoreError(
                    f"trusted run {run.id} has no execution commit receipt"
                )
            try:
                receipt = ExecutionCommitReceipt.from_dict(json.loads(row[0]))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise StoreError(
                    f"trusted run {run.id} has an invalid execution commit receipt"
                ) from exc
            receipt_error = execution_commit_receipt_error(receipt, attestation, run)
            if receipt_error is not None:
                raise StoreError(
                    f"trusted run {run.id} execution receipt is invalid: {receipt_error}"
                )
        return attestation

    @classmethod
    def _require_real_tool_run(
        cls, connection: sqlite3.Connection, run: ToolRun, *, stage: str, label: str,
    ) -> None:
        authority = str(run.metadata.get("authority", ""))
        backend = str(run.backend).casefold()
        if (
            run.stage != stage
            or str(run.status) not in {"succeeded", "cached"}
            or str(run.failure_class) != "none"
            or run.exit_code not in {None, 0}
            or run.metadata.get("tool_truth") is not True
            or run.metadata.get("fresh_execution") is not True
            or run.metadata.get("fresh_tool_truth") is not True
            or authority in {"synthetic", "fake", "replay"}
            or backend in {"runner.fake", "runner.replay"}
        ):
            raise StoreError(f"{label} requires one successful real {stage} tool run")
        cls._validate_execution_attestation(connection, run)

    @staticmethod
    def _snapshot_manifest_for_evidence(
        connection: sqlite3.Connection, snapshot_id: str,
    ) -> ProjectManifest:
        row = connection.execute(
            "SELECT manifest_json FROM snapshot_manifests WHERE snapshot_id=?",
            (snapshot_id,),
        ).fetchone()
        if row is None:
            raise StoreError(f"snapshot {snapshot_id} has no immutable manifest context")
        return ProjectManifest.from_dict(json.loads(row[0]))

    @staticmethod
    def _managed_artifact_bytes_valid(
        connection: sqlite3.Connection, artifact: ArtifactRef,
    ) -> bool:
        if str(artifact.retention) != "managed":
            return False
        database = connection.execute("PRAGMA database_list").fetchone()
        if database is None or not database[2]:
            return False
        project_root = Path(str(database[2])).resolve().parent.parent
        try:
            _data, size, digest, _path = read_verified_file(
                project_root, artifact.uri,
                expected_size=artifact.size,
                expected_sha256=artifact.sha256,
                max_bytes=artifact.size,
            )
        except (OSError, StagingError, ValueError):
            return False
        return size == artifact.size and digest == artifact.sha256

    @staticmethod
    def _physical_artifact_scope_valid(
        artifact: ArtifactRef, manifest: ProjectManifest,
    ) -> bool:
        scope = artifact.metadata.get("scope")
        if not isinstance(scope, dict):
            return False
        top = manifest.build.top
        if (scope.get("kind") != "kernel" or str(scope.get("top") or "") != top
                or str(scope.get("instance") or "") != top):
            return False
        if manifest.target.part and str(scope.get("part") or "") != manifest.target.part:
            return False
        if (manifest.target.platform
                and str(scope.get("platform") or "") != manifest.target.platform):
            return False
        if artifact.kind in {
            "amd.vivado.post_route_timing", "amd.vivado.timing_summary",
        }:
            clock = str(scope.get("clock") or "")
            known = {item.name for item in manifest.target.clocks}
            if clock != "all" and clock not in known:
                return False
        return True

    @staticmethod
    def _require_verification_value_semantics(
        item: VerificationResult, leaves: list[Observation], expected_workload: str | None,
    ) -> None:
        if expected_workload and any(
            observation.workload_id != expected_workload for observation in leaves
        ):
            raise StoreError(
                f"verification {item.id} observations do not share workload "
                f"{expected_workload!r}"
            )
        grouped: dict[str, list[Observation]] = {}
        for observation in leaves:
            grouped.setdefault(observation.predicate, []).append(observation)
        if str(item.kind) == "csim":
            required = {
                "csim.exit_code", "csim.mismatches", "csim.assertions_failed",
            }
            if any(len(grouped.get(predicate, [])) != 1 for predicate in required):
                raise StoreError(
                    f"verification {item.id} requires one observation for each CSim "
                    "exit/mismatch/assertion predicate"
                )
            for predicate in required:
                observation = grouped[predicate][0]
                value = observation.value
                if (isinstance(value, bool) or not isinstance(value, (int, float))
                        or not math.isfinite(float(value)) or float(value) != 0.0
                        or observation.unit != "count"):
                    raise StoreError(
                        f"passing verification {item.id} contradicts {predicate}={value!r}"
                    )
        elif str(item.kind) == "rtl_cosim":
            statuses = grouped.get("cosim.status", [])
            if (len(statuses) != 1 or not isinstance(statuses[0].value, str)
                    or statuses[0].value.casefold() != "pass"
                    or statuses[0].unit is not None):
                raise StoreError(
                    f"verification {item.id} requires one cosim.status='pass' observation"
                )

    @classmethod
    def _require_typed_verification_evidence(
        cls,
        connection: sqlite3.Connection,
        item: VerificationResult,
        run: ToolRun,
    ) -> None:
        policy = _VERIFICATION_REPORT_POLICY.get(str(item.kind))
        if policy is None:
            raise StoreError(
                f"passing verification kind {str(item.kind)!r} has no typed report policy"
            )
        run_stage = str(policy["run_stage"])
        cls._require_real_tool_run(
            connection, run, stage=run_stage, label=f"verification {item.id}",
        )
        allowed_stages = frozenset(policy["observation_stages"])
        allowed_kinds = frozenset(policy["artifact_kinds"])
        saw_typed_evidence = False
        all_leaves: list[Observation] = []
        for evidence_id in item.evidence_ids:
            kind = cls._evidence_kind(connection, item.snapshot_id, evidence_id)
            if kind == "artifact":
                raise StoreError(
                    f"verification {item.id} cannot infer PASS from a bare report artifact"
                )
            leaves = cls._evidence_observation_leaves(
                connection, item.snapshot_id, evidence_id,
            )
            if not leaves:
                raise StoreError(
                    f"verification {item.id} evidence is not a typed report or observation"
                )
            for observation in leaves:
                artifacts = cls._observation_artifacts(
                    connection, item.snapshot_id, observation,
                )
                if (
                    str(observation.completeness) != "complete"
                    or observation.stage not in allowed_stages
                    or observation.authority not in _TOOL_EVIDENCE_AUTHORITIES
                    or observation.run_id != run.id
                    or not artifacts
                    or any(
                        artifact.kind not in allowed_kinds
                        or artifact.producer_run_id != run.id
                        or not cls._managed_artifact_bytes_valid(connection, artifact)
                        for artifact in artifacts
                    )
                ):
                    raise StoreError(
                        f"verification {item.id} evidence is not a complete, stage-aligned "
                        f"{str(item.kind)} tool observation"
                    )
                saw_typed_evidence = True
                all_leaves.append(observation)
        if not saw_typed_evidence:
            raise StoreError(f"verification {item.id} has no typed report evidence")
        run_workload = (str(run.metadata.get("workload_id"))
                        if run.metadata.get("workload_id") else None)
        cls._require_verification_value_semantics(
            item, all_leaves, item.workload_id or run_workload,
        )

    @classmethod
    def _require_physical_derivation_semantics(
        cls,
        connection: sqlite3.Connection,
        item: Derivation,
        leaves: list[Observation],
    ) -> None:
        manifest = cls._snapshot_manifest_for_evidence(connection, item.snapshot_id)
        if item.predicate == "gate.post_route_timing":
            if (item.algorithm != "hlsgraph.gate.wns_nonnegative"
                    or item.algorithm_version != "1"):
                raise StoreError(
                    f"physical gate {item.id} uses an unsupported timing derivation algorithm"
                )
            wns = [value for value in leaves if value.predicate == "timing.wns_ns"]
            if len(wns) != 1:
                raise StoreError(
                    f"physical gate {item.id} requires exactly one timing.wns_ns observation"
                )
            value = wns[0].value
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or item.value is not (float(value) >= 0.0)):
                raise StoreError(
                    f"physical gate {item.id} value contradicts its post-route WNS"
                )
            if any(value.predicate not in {"timing.wns_ns", "timing.tns_ns"}
                   for value in leaves):
                raise StoreError(f"physical gate {item.id} contains unrelated timing evidence")
            if any(value.unit != "ns" for value in leaves):
                raise StoreError(f"physical gate {item.id} timing evidence must use ns")
            return
        if (item.algorithm != "hlsgraph.gate.capacity_compare"
                or item.algorithm_version != "1"):
            raise StoreError(
                f"physical gate {item.id} uses an unsupported resource derivation algorithm"
            )
        capacities = {str(key).lower(): float(value)
                      for key, value in manifest.target.capacities.items()}
        reserved = {str(key).lower(): float(value)
                    for key, value in manifest.target.reserved_resources.items()}
        usage: dict[str, float] = {}
        for observation in leaves:
            if not observation.predicate.startswith("resource."):
                raise StoreError(f"physical gate {item.id} contains unrelated resource evidence")
            name = observation.predicate.split(".", 1)[1].lower()
            value = observation.value
            if (name in usage or isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value)) or float(value) < 0
                    or observation.unit != "count"):
                raise StoreError(f"physical gate {item.id} has invalid resource observation")
            usage[name] = float(value)
        if not capacities or set(usage) != set(capacities) or set(reserved) - set(capacities):
            raise StoreError(
                f"physical gate {item.id} does not cover the complete target capacity set"
            )
        expected = all(
            usage[name] <= capacities[name] - reserved.get(name, 0.0)
            for name in capacities
        )
        if item.value is not expected:
            raise StoreError(
                f"physical gate {item.id} value contradicts target resource capacity"
            )
        if item.metadata.get("target_profile_hash") != stable_hash(manifest.target):
            raise StoreError(
                f"physical gate {item.id} target_profile_hash does not match its snapshot"
            )

    @classmethod
    def _require_physical_gate_evidence(
        cls,
        connection: sqlite3.Connection,
        item: Derivation,
        producer_id: str,
    ) -> None:
        row = connection.execute(
            "SELECT payload_json FROM runs WHERE snapshot_id=? AND id=?",
            (item.snapshot_id, producer_id),
        ).fetchone()
        if row is None:
            raise StoreError(f"physical gate {item.id} cites a missing producer run")
        run = ToolRun.from_dict(json.loads(row[0]))
        cls._require_real_tool_run(
            connection, run, stage="post_route", label=f"physical gate {item.id}",
        )
        if item.stage != "post_route":
            raise StoreError(f"physical gate {item.id} must use the post_route stage")
        allowed_kinds = _PHYSICAL_GATE_REPORT_KINDS[item.predicate]
        if not item.input_observation_ids:
            raise StoreError(f"physical gate {item.id} has no observation evidence")
        manifest = cls._snapshot_manifest_for_evidence(connection, item.snapshot_id)
        all_leaves: list[Observation] = []
        for evidence_id in item.input_observation_ids:
            if cls._evidence_producer_runs(
                connection, item.snapshot_id, evidence_id,
            ) != {producer_id}:
                raise StoreError(
                    f"physical gate {item.id} evidence must close to exactly one producer run"
                )
            leaves = cls._evidence_observation_leaves(
                connection, item.snapshot_id, evidence_id,
            )
            if not leaves:
                raise StoreError(f"physical gate {item.id} has no observation evidence")
            for observation in leaves:
                artifacts = cls._observation_artifacts(
                    connection, item.snapshot_id, observation,
                )
                if (
                    observation.stage != "post_route"
                    or str(observation.completeness) != "complete"
                    or observation.authority not in _TOOL_EVIDENCE_AUTHORITIES
                    or observation.run_id != producer_id
                    or not artifacts
                    or any(
                        artifact.kind not in allowed_kinds
                        or artifact.producer_run_id != producer_id
                        or not cls._managed_artifact_bytes_valid(connection, artifact)
                        or not cls._physical_artifact_scope_valid(artifact, manifest)
                        or (
                            artifact.kind in {
                                "amd.vivado.utilization",
                                "amd.vivado.timing_summary",
                            }
                            and artifact.metadata.get("stage") != "post_route"
                        )
                        for artifact in artifacts
                    )
                ):
                    raise StoreError(
                        f"physical gate {item.id} requires complete post_route observations "
                        "from an allowed typed report"
                    )
                all_leaves.append(observation)
        cls._require_physical_derivation_semantics(
            connection, item, all_leaves,
        )

    @staticmethod
    def _require_evidence(connection: sqlite3.Connection, snapshot_id: str,
                          evidence_id: str, *, tables: tuple[str, ...], label: str) -> None:
        allowed = {"observations", "derivations", "diagnostics", "runs"}
        if not tables or not set(tables).issubset(allowed):
            raise StoreError("invalid evidence table policy")
        for table in tables:
            if connection.execute(
                f"SELECT 1 FROM {table} WHERE snapshot_id=? AND id=?",
                (snapshot_id, evidence_id),
            ).fetchone():
                return
        raise StoreError(
            f"{label} {evidence_id!r} does not exist in snapshot {snapshot_id}"
        )

    def save_project(self, manifest: ProjectManifest) -> None:
        payload = self._payload(manifest)
        with self.write() as connection:
            previous = connection.execute(
                "SELECT manifest_json FROM projects WHERE project_id=?", (manifest.project_id,)
            ).fetchone()
            if previous and previous[0] != payload:
                # The project row is the current public manifest; snapshots retain immutable identity.
                connection.execute(
                    "UPDATE projects SET manifest_hash=?, manifest_json=? WHERE project_id=?",
                    (stable_hash(manifest.identity_payload()), payload, manifest.project_id),
                )
            elif not previous:
                connection.execute(
                    "INSERT INTO projects(project_id,manifest_hash,manifest_json) VALUES(?,?,?)",
                    (manifest.project_id, stable_hash(manifest.identity_payload()), payload),
                )

    def save_snapshot(self, snapshot: DesignSnapshot, artifacts: list[ArtifactRef]) -> None:
        snapshot = self._revalidate_model(snapshot, DesignSnapshot, "design snapshot")
        artifacts = [self._revalidate_model(item, ArtifactRef, "artifact")
                     for item in artifacts]
        if len({item.id for item in artifacts}) != len(artifacts):
            raise StoreError("snapshot artifact identifiers must be unique")
        if len({item.uri for item in artifacts}) != len(artifacts):
            raise StoreError("snapshot artifact URIs must be unique")
        if snapshot.artifact_hashes != artifact_hash_map(artifacts):
            raise StoreError("snapshot artifact hash closure does not match attached artifacts")
        with self.write() as connection:
            project = connection.execute(
                "SELECT manifest_hash,manifest_json FROM projects WHERE project_id=?",
                (snapshot.project_id,),
            ).fetchone()
            if project is None:
                raise StoreError(f"snapshot project is not registered: {snapshot.project_id}")
            try:
                manifest = ProjectManifest.from_dict(json.loads(project[1]))
            except (KeyError, TypeError, ValueError) as exc:
                raise StoreError(f"registered project manifest is invalid: {exc}") from exc
            expected_manifest_hash = stable_hash(manifest.identity_payload())
            expected_context = {
                "manifest_hash": expected_manifest_hash,
                "build_hash": stable_hash(manifest.build),
                "target_hash": stable_hash(manifest.target),
                "constraint_hash": stable_hash(manifest.constraints),
                "toolchain_hash": stable_hash({
                    "toolchains": manifest.toolchains,
                    "stage_toolchains": manifest.stage_toolchains,
                }),
            }
            actual_context = {name: getattr(snapshot, name) for name in expected_context}
            if project[0] != expected_manifest_hash:
                raise StoreError("registered project manifest hash is internally inconsistent")
            if actual_context != expected_context:
                raise StoreError("snapshot manifest hash does not match the registered project context")
            if snapshot.parent_snapshot_id:
                parent = connection.execute(
                    "SELECT project_id FROM snapshots WHERE id=?", (snapshot.parent_snapshot_id,)
                ).fetchone()
                if parent is None or parent[0] != snapshot.project_id:
                    raise StoreError("snapshot parent must exist in the same project")
            if snapshot.action_id:
                action = connection.execute(
                    "SELECT parent_snapshot_id FROM variants WHERE id=?",
                    (snapshot.action_id,),
                ).fetchone()
                # Legacy snapshots may carry an opaque action identifier whose
                # action row was never imported.  Once a row exists, however,
                # the action's recorded input snapshot is authoritative for
                # lineage and must match exactly (including rejecting None).
                if (action is not None
                        and snapshot.parent_snapshot_id != action[0]):
                    raise StoreError(
                        "snapshot parent_snapshot_id does not match its recorded action"
                    )
            snapshot_payload = self._payload(snapshot)
            previous_snapshot = connection.execute(
                "SELECT payload_json FROM snapshots WHERE id=?", (snapshot.id,)
            ).fetchone()
            if previous_snapshot:
                previous = DesignSnapshot.from_dict(json.loads(previous_snapshot[0]))
                if previous.identity_payload() != snapshot.identity_payload():
                    raise StoreError(f"snapshot identity collision: {snapshot.id}")
            else:
                connection.execute(
                    "INSERT INTO snapshots(id,project_id,created_at,payload_json) VALUES(?,?,?,?)",
                    (snapshot.id, snapshot.project_id, snapshot.created_at, snapshot_payload),
                )
            previous_manifest = connection.execute(
                "SELECT manifest_hash,manifest_json FROM snapshot_manifests WHERE snapshot_id=?",
                (snapshot.id,),
            ).fetchone()
            if previous_manifest and (previous_manifest[0] != project[0]
                                      or previous_manifest[1] != project[1]):
                raise StoreError(f"immutable snapshot manifest changed: {snapshot.id}")
            connection.execute(
                "INSERT OR IGNORE INTO snapshot_manifests(snapshot_id,manifest_hash,manifest_json) "
                "VALUES(?,?,?)", (snapshot.id, project[0], project[1]),
            )
            for artifact in artifacts:
                self._require_artifact_producer(connection, snapshot.id, artifact)
                payload = self._payload(artifact)
                existing = connection.execute("SELECT payload_json FROM artifacts WHERE id=?",
                                              (artifact.id,)).fetchone()
                if existing and existing[0] != payload:
                    raise StoreError(f"artifact id collision: {artifact.id}")
                connection.execute(
                    "INSERT OR IGNORE INTO artifacts(id,kind,uri,sha256,size,payload_json) VALUES(?,?,?,?,?,?)",
                    (artifact.id, artifact.kind, artifact.uri, artifact.sha256, artifact.size, payload),
                )
                connection.execute(
                    "INSERT OR IGNORE INTO snapshot_artifacts(snapshot_id,artifact_id,role) VALUES(?,?,?)",
                    (snapshot.id, artifact.id, artifact.role),
                )

    def _validate_graph(self, connection: sqlite3.Connection, graph: CanonicalGraph) -> None:
        if getattr(graph, "schema_version", None) != SCHEMA_VERSION:
            raise StoreError(
                f"canonical graph schema {getattr(graph, 'schema_version', None)!r} is not "
                f"supported by this build ({SCHEMA_VERSION!r})"
            )
        try:
            reject_embedded_body_fields(graph.metadata, "canonical graph metadata")
        except ValueError as exc:
            raise StoreError(str(exc)) from exc
        if not connection.execute(
            "SELECT 1 FROM snapshots WHERE id=?", (graph.snapshot_id,)
        ).fetchone():
            raise StoreError(f"unknown snapshot: {graph.snapshot_id}")
        attached_artifacts = {row[0] for row in connection.execute(
            "SELECT artifact_id FROM snapshot_artifacts WHERE snapshot_id=?",
            (graph.snapshot_id,),
        )}
        for key, entity in graph.entities.items():
            self._revalidate_model(entity, Entity, f"entity {key}")
            if key != entity.id:
                raise StoreError(f"canonical entity dictionary key disagrees with id: {key}")
            if entity.snapshot_id != graph.snapshot_id:
                raise StoreError(f"entity {entity.id} belongs to another snapshot")
            self._require_fact_authority(entity.authority, f"entity {entity.id}")
            for anchor in entity.anchors:
                if anchor.artifact_id not in attached_artifacts:
                    raise StoreError(
                        f"entity {entity.id} anchor artifact {anchor.artifact_id!r} is not "
                        f"attached to snapshot {graph.snapshot_id}"
                    )
        for key, relation in graph.relations.items():
            self._revalidate_model(relation, Relation, f"relation {key}")
            if key != relation.id:
                raise StoreError(f"canonical relation dictionary key disagrees with id: {key}")
            if relation.snapshot_id != graph.snapshot_id:
                raise StoreError(f"relation {relation.id} belongs to another snapshot")
            self._require_fact_authority(relation.authority, f"relation {relation.id}")
            if relation.src not in graph.entities or relation.dst not in graph.entities:
                raise StoreError(
                    f"relation endpoints must exist in the canonical graph: "
                    f"{relation.src} -> {relation.dst}"
                )
            for anchor in relation.anchors:
                if anchor.artifact_id not in attached_artifacts:
                    raise StoreError(
                        f"relation {relation.id} anchor artifact {anchor.artifact_id!r} is not "
                        f"attached to snapshot {graph.snapshot_id}"
                    )

    def _save_graph(self, connection: sqlite3.Connection, graph: CanonicalGraph) -> None:
        self._validate_graph(connection, graph)
        metadata_payload = self._payload(graph.metadata)
        previous_view = connection.execute(
            "SELECT schema_version,metadata_json FROM graph_views WHERE snapshot_id=?",
            (graph.snapshot_id,),
        ).fetchone()
        if previous_view and (
            previous_view[0] != graph.schema_version or previous_view[1] != metadata_payload
        ):
            raise StoreError(f"immutable graph metadata changed: {graph.snapshot_id}")
        if previous_view:
            entity_ids = {row[0] for row in connection.execute(
                "SELECT id FROM entities WHERE snapshot_id=?", (graph.snapshot_id,)
            )}
            relation_ids = {row[0] for row in connection.execute(
                "SELECT id FROM relations WHERE snapshot_id=?", (graph.snapshot_id,)
            )}
            if entity_ids != set(graph.entities):
                raise StoreError(f"immutable graph entity set changed: {graph.snapshot_id}")
            if relation_ids != set(graph.relations):
                raise StoreError(f"immutable graph relation set changed: {graph.snapshot_id}")
        connection.execute(
            "INSERT OR IGNORE INTO graph_views(snapshot_id,schema_version,metadata_json) VALUES(?,?,?)",
            (graph.snapshot_id, graph.schema_version, metadata_payload),
        )
        for entity in sorted(graph.entities.values(), key=lambda item: item.id):
            payload = self._payload(entity)
            previous = connection.execute(
                "SELECT payload_json FROM entities WHERE snapshot_id=? AND id=?",
                (graph.snapshot_id, entity.id),
            ).fetchone()
            if previous and previous[0] != payload:
                raise StoreError(f"immutable entity changed: {entity.id}")
            connection.execute(
                "INSERT OR IGNORE INTO entities(id,snapshot_id,kind,name,qualified_name,stage,authority,payload_json) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (entity.id, graph.snapshot_id, entity.kind, entity.name, entity.qualified_name,
                 entity.stage, str(entity.authority), payload),
            )
        for relation in sorted(graph.relations.values(), key=lambda item: item.id):
            payload = self._payload(relation)
            previous = connection.execute(
                "SELECT payload_json FROM relations WHERE snapshot_id=? AND id=?",
                (graph.snapshot_id, relation.id),
            ).fetchone()
            if previous and previous[0] != payload:
                raise StoreError(f"immutable relation changed: {relation.id}")
            connection.execute(
                "INSERT OR IGNORE INTO relations(id,snapshot_id,src,dst,kind,stage,authority,payload_json) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (relation.id, graph.snapshot_id, relation.src, relation.dst, relation.kind,
                 relation.stage, str(relation.authority), payload),
            )
        fts = connection.execute("SELECT value FROM schema_info WHERE key='fts5'").fetchone()
        if fts and fts[0] == "1":
            connection.execute("DELETE FROM entities_fts WHERE snapshot_id=?", (graph.snapshot_id,))
            for entity in sorted(graph.entities.values(), key=lambda item: item.id):
                connection.execute(
                    "INSERT INTO entities_fts(snapshot_id,entity_id,name,qualified_name,aliases,attrs) "
                    "VALUES(?,?,?,?,?,?)",
                    (graph.snapshot_id, entity.id, entity.name, entity.qualified_name or "",
                     " ".join(entity.aliases), self._payload(entity.attrs)),
                )

    def save_graph(self, graph: CanonicalGraph) -> None:
        with self.write() as connection:
            self._save_graph(connection, graph)

    def load_graph(self, snapshot_id: str) -> CanonicalGraph:
        with self.read() as connection:
            if not connection.execute("SELECT 1 FROM snapshots WHERE id=?", (snapshot_id,)).fetchone():
                raise KeyError(snapshot_id)
            view = connection.execute(
                "SELECT schema_version,metadata_json FROM graph_views WHERE snapshot_id=?", (snapshot_id,)
            ).fetchone()
            if view is None:
                raise StoreError(f"snapshot has no successful canonical graph view: {snapshot_id}")
            if view[0] not in SUPPORTED_GRAPH_SCHEMA_VERSIONS:
                raise StoreError(
                    f"graph schema {view[0]!r} is not supported by this build "
                    f"({sorted(SUPPORTED_GRAPH_SCHEMA_VERSIONS)!r})"
                )
            graph = CanonicalGraph(
                snapshot_id=snapshot_id,
                schema_version=view[0], metadata=json.loads(view[1]),
            )
            for row in connection.execute(
                "SELECT payload_json FROM entities WHERE snapshot_id=? ORDER BY id", (snapshot_id,)
            ):
                graph.add_entity(Entity.from_dict(json.loads(row[0])))
            for row in connection.execute(
                "SELECT payload_json FROM relations WHERE snapshot_id=? ORDER BY id", (snapshot_id,)
            ):
                graph.add_relation(Relation.from_dict(json.loads(row[0])))
            return graph

    @staticmethod
    def _load_graph_from_connection(
        connection: sqlite3.Connection, snapshot_id: str,
    ) -> CanonicalGraph:
        """Load the immutable graph inside the caller's current transaction."""

        view = connection.execute(
            "SELECT schema_version,metadata_json FROM graph_views WHERE snapshot_id=?",
            (snapshot_id,),
        ).fetchone()
        if view is None or view[0] not in SUPPORTED_GRAPH_SCHEMA_VERSIONS:
            raise StoreError(
                f"snapshot {snapshot_id} has no supported canonical graph for parser replay"
            )
        graph = CanonicalGraph(
            snapshot_id=snapshot_id,
            schema_version=view[0], metadata=json.loads(view[1]),
        )
        for row in connection.execute(
            "SELECT payload_json FROM entities WHERE snapshot_id=? ORDER BY id",
            (snapshot_id,),
        ):
            graph.add_entity(Entity.from_dict(json.loads(row[0])))
        for row in connection.execute(
            "SELECT payload_json FROM relations WHERE snapshot_id=? ORDER BY id",
            (snapshot_id,),
        ):
            graph.add_relation(Relation.from_dict(json.loads(row[0])))
        return graph

    def has_graph(self, snapshot_id: str) -> bool:
        with self.read() as connection:
            return connection.execute(
                "SELECT 1 FROM graph_views WHERE snapshot_id=?", (snapshot_id,)
            ).fetchone() is not None

    def latest_snapshot(self, project_id: str | None = None) -> DesignSnapshot | None:
        """Return the active graph, or newest successful graph if no pointer exists.

        The fallback supports low-level SDK users that persist a graph directly;
        failed candidates are never selected because they have no graph view.
        """
        with self.read() as connection:
            if project_id:
                active = connection.execute(
                    "SELECT s.payload_json FROM project_state ps JOIN snapshots s "
                    "ON s.id=ps.active_snapshot_id WHERE ps.project_id=?", (project_id,),
                ).fetchone()
                if active:
                    return DesignSnapshot.from_dict(json.loads(active[0]))
                row = connection.execute(
                    "SELECT s.payload_json FROM snapshots s JOIN graph_views g "
                    "ON g.snapshot_id=s.id WHERE s.project_id=? ORDER BY s.rowid DESC LIMIT 1",
                    (project_id,),
                ).fetchone()
            else:
                active = connection.execute(
                    "SELECT s.payload_json FROM project_state ps JOIN snapshots s "
                    "ON s.id=ps.active_snapshot_id ORDER BY ps.rowid DESC LIMIT 1"
                ).fetchone()
                if active:
                    return DesignSnapshot.from_dict(json.loads(active[0]))
                row = connection.execute(
                    "SELECT s.payload_json FROM snapshots s JOIN graph_views g "
                    "ON g.snapshot_id=s.id ORDER BY s.rowid DESC LIMIT 1"
                ).fetchone()
            return DesignSnapshot.from_dict(json.loads(row[0])) if row else None

    def latest_candidate(self, project_id: str | None = None) -> DesignSnapshot | None:
        """Return the newest candidate event, whether or not extraction succeeded."""
        with self.read() as connection:
            if project_id:
                row = connection.execute(
                    "SELECT payload_json FROM snapshots WHERE project_id=? ORDER BY rowid DESC LIMIT 1",
                    (project_id,),
                ).fetchone()
            else:
                row = connection.execute(
                    "SELECT payload_json FROM snapshots ORDER BY rowid DESC LIMIT 1"
                ).fetchone()
            return DesignSnapshot.from_dict(json.loads(row[0])) if row else None

    def _set_active_snapshot(
        self, connection: sqlite3.Connection, project_id: str, snapshot_id: str
    ) -> None:
        row = connection.execute(
            "SELECT project_id FROM snapshots WHERE id=?", (snapshot_id,)
        ).fetchone()
        if row is None:
            raise KeyError(snapshot_id)
        if row[0] != project_id:
            raise StoreError("active snapshot belongs to another project")
        view = connection.execute(
            "SELECT schema_version FROM graph_views WHERE snapshot_id=?", (snapshot_id,)
        ).fetchone()
        if view is None:
            raise StoreError("only a successful canonical graph can become active")
        if view[0] not in SUPPORTED_GRAPH_SCHEMA_VERSIONS:
            raise StoreError(
                f"canonical graph schema {view[0]!r} is not supported by this build "
                f"({sorted(SUPPORTED_GRAPH_SCHEMA_VERSIONS)!r}); refusing to activate it"
            )
        connection.execute(
            "INSERT INTO project_state(project_id,active_snapshot_id,updated_at) VALUES(?,?,?) "
            "ON CONFLICT(project_id) DO UPDATE SET active_snapshot_id=excluded.active_snapshot_id, "
            "updated_at=excluded.updated_at",
            (project_id, snapshot_id, utc_now()),
        )

    def set_active_snapshot(self, project_id: str, snapshot_id: str) -> None:
        with self.write() as connection:
            self._set_active_snapshot(connection, project_id, snapshot_id)

    def snapshot(self, snapshot_id: str) -> DesignSnapshot:
        with self.read() as connection:
            row = connection.execute(
                "SELECT payload_json FROM snapshots WHERE id=?", (snapshot_id,)
            ).fetchone()
            if row is None:
                raise KeyError(snapshot_id)
            return DesignSnapshot.from_dict(json.loads(row[0]))

    def snapshot_manifest(self, snapshot_id: str) -> ProjectManifest:
        with self.read() as connection:
            row = connection.execute(
                "SELECT manifest_json FROM snapshot_manifests WHERE snapshot_id=?", (snapshot_id,)
            ).fetchone()
            if row is None:
                if not connection.execute(
                    "SELECT 1 FROM snapshots WHERE id=?", (snapshot_id,)
                ).fetchone():
                    raise KeyError(snapshot_id)
                raise StoreError(
                    f"snapshot has no immutable manifest context: {snapshot_id}; "
                    "run an explicit migration"
                )
            return ProjectManifest.from_dict(json.loads(row[0]))

    def artifacts(self, snapshot_id: str) -> list[ArtifactRef]:
        with self.read() as connection:
            rows = connection.execute(
                "SELECT a.payload_json FROM artifacts a JOIN snapshot_artifacts sa ON a.id=sa.artifact_id "
                "WHERE sa.snapshot_id=? ORDER BY a.id", (snapshot_id,),
            ).fetchall()
            return [ArtifactRef.from_dict(json.loads(row[0])) for row in rows]

    def _add_artifact(
        self,
        connection: sqlite3.Connection,
        snapshot_id: str,
        artifact: ArtifactRef,
        *,
        future_run_ids: frozenset[str] = frozenset(),
    ) -> None:
        artifact = self._revalidate_model(artifact, ArtifactRef, "artifact")
        if not connection.execute(
            "SELECT 1 FROM snapshots WHERE id=?", (snapshot_id,)
        ).fetchone():
            raise KeyError(snapshot_id)
        self._require_artifact_producer(
            connection, snapshot_id, artifact, future_run_ids=future_run_ids,
        )
        payload = self._payload(artifact)
        previous = connection.execute(
            "SELECT payload_json FROM artifacts WHERE id=?", (artifact.id,)
        ).fetchone()
        if previous and previous[0] != payload:
            raise StoreError(f"artifact id collision: {artifact.id}")
        if not previous:
            connection.execute(
                "INSERT INTO artifacts(id,kind,uri,sha256,size,payload_json) VALUES(?,?,?,?,?,?)",
                (artifact.id, artifact.kind, artifact.uri, artifact.sha256,
                 artifact.size, payload),
            )
        connection.execute(
            "INSERT OR IGNORE INTO snapshot_artifacts(snapshot_id,artifact_id,role) VALUES(?,?,?)",
            (snapshot_id, artifact.id, artifact.role),
        )

    def add_artifact(self, snapshot_id: str, artifact: ArtifactRef) -> None:
        """Attach an artifact whose producer, when declared, already exists."""
        with self.write() as connection:
            self._add_artifact(connection, snapshot_id, artifact)

    def _add_correspondences(
        self,
        connection: sqlite3.Connection,
        values: list[EntityCorrespondence],
    ) -> None:
        for item in values:
            item = self._revalidate_model(
                item, EntityCorrespondence, "entity correspondence",
            )
            snapshots = connection.execute(
                "SELECT id,project_id FROM snapshots WHERE id IN (?,?)",
                (item.source_snapshot_id, item.target_snapshot_id),
            ).fetchall()
            found = {str(row[0]): str(row[1]) for row in snapshots}
            missing = {
                item.source_snapshot_id, item.target_snapshot_id,
            } - set(found)
            if missing:
                raise StoreError(
                    "correspondence snapshots do not exist: "
                    + ", ".join(sorted(missing))
                )
            if len(set(found.values())) != 1:
                raise StoreError("correspondence endpoints must belong to one project")
            for snapshot_id, entity_id, endpoint in (
                (item.source_snapshot_id, item.source_entity_id, "source"),
                (item.target_snapshot_id, item.target_entity_id, "target"),
            ):
                if not connection.execute(
                    "SELECT 1 FROM entities WHERE snapshot_id=? AND id=?",
                    (snapshot_id, entity_id),
                ).fetchone():
                    raise StoreError(
                        f"correspondence {endpoint} entity {entity_id!r} does not "
                        f"exist in snapshot {snapshot_id}"
                    )
            allowed = frozenset({item.source_snapshot_id, item.target_snapshot_id})
            for reference in item.evidence_refs:
                self._resolve_evidence_ref(
                    connection, reference, allowed_snapshot_ids=allowed,
                    label=f"correspondence {item.id}",
                )
            payload = self._payload(item)
            if self._immutable_payload(
                connection, "entity_correspondences", item.id, payload,
            ):
                continue
            connection.execute(
                "INSERT INTO entity_correspondences("
                "id,source_snapshot_id,source_entity_id,target_snapshot_id,"
                "target_entity_id,kind,payload_json) VALUES(?,?,?,?,?,?,?)",
                (item.id, item.source_snapshot_id, item.source_entity_id,
                 item.target_snapshot_id, item.target_entity_id, item.kind, payload),
            )

    def add_correspondence(self, value: EntityCorrespondence) -> None:
        self.add_correspondences([value])

    def add_correspondences(self, values: list[EntityCorrespondence]) -> None:
        with self.write() as connection:
            self._add_correspondences(connection, values)

    def correspondences(
        self,
        snapshot_id: str | None = None,
        *,
        source_snapshot_id: str | None = None,
        target_snapshot_id: str | None = None,
        kind: str | None = None,
    ) -> list[EntityCorrespondence]:
        clauses: list[str] = []
        params: list[str] = []
        if snapshot_id is not None:
            clauses.append("(source_snapshot_id=? OR target_snapshot_id=?)")
            params.extend([snapshot_id, snapshot_id])
        if source_snapshot_id is not None:
            clauses.append("source_snapshot_id=?")
            params.append(source_snapshot_id)
        if target_snapshot_id is not None:
            clauses.append("target_snapshot_id=?")
            params.append(target_snapshot_id)
        if kind is not None:
            clauses.append("kind=?")
            params.append(kind)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self.read() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM entity_correspondences"
                f"{where} ORDER BY id", params,
            ).fetchall()
            return [
                EntityCorrespondence.from_dict(json.loads(row[0])) for row in rows
            ]

    def _add_observations(
        self,
        connection: sqlite3.Connection,
        values: list[Observation],
        *,
        future_run_ids: frozenset[str] = frozenset(),
    ) -> None:
        parser_replay_cache: dict[
            tuple[str, str, str], tuple[Observation, ...]
        ] = {}
        for item in values:
            item = self._revalidate_model(item, Observation, "observation")
            self._require_fact_authority(item.authority, f"observation {item.id}")
            self._require_subject(connection, item.snapshot_id, item.subject_id)
            if item.artifact_id:
                self._require_artifact(connection, item.snapshot_id, item.artifact_id)
            if item.anchor:
                self._require_artifact(connection, item.snapshot_id, item.anchor.artifact_id)
            if item.run_id and item.run_id not in future_run_ids:
                self._require_run(connection, item.snapshot_id, item.run_id)
            cited_artifacts = {value for value in (
                item.artifact_id,
                item.anchor.artifact_id if item.anchor else None,
            ) if value}
            producer_ids: set[str] = set()
            cited_artifact_values: list[ArtifactRef] = []
            for artifact_id in cited_artifacts:
                row = connection.execute(
                    "SELECT a.payload_json FROM artifacts a JOIN snapshot_artifacts sa "
                    "ON a.id=sa.artifact_id WHERE sa.snapshot_id=? AND a.id=?",
                    (item.snapshot_id, artifact_id),
                ).fetchone()
                if row:
                    artifact = ArtifactRef.from_dict(json.loads(row[0]))
                    cited_artifact_values.append(artifact)
                    producer = artifact.producer_run_id
                    if producer:
                        producer_ids.add(producer)
            if producer_ids and (not item.run_id or producer_ids != {item.run_id}):
                raise StoreError(
                    f"observation {item.id} run does not match cited artifact producer"
                )
            tool_authorities = {
                AuthorityClass.TOOL_OBSERVATION,
                AuthorityClass.VERIFICATION_EVIDENCE,
                AuthorityClass.PHYSICAL_MEASUREMENT,
            }
            if item.run_id and item.authority in tool_authorities and not producer_ids:
                raise StoreError(
                    f"tool-backed observation {item.id} must cite an artifact produced by its run"
                )
            if item.run_id and item.authority in tool_authorities:
                run_row = connection.execute(
                    "SELECT payload_json FROM runs WHERE snapshot_id=? AND id=?",
                    (item.snapshot_id, item.run_id),
                ).fetchone()
                producer_run = (ToolRun.from_dict(json.loads(run_row[0]))
                                if run_row else None)
                if producer_run is not None and run_claims_tool_truth(producer_run):
                    self._validate_execution_attestation(connection, producer_run)
                    compatibility_error = tool_evidence_compatibility_error(
                        item, producer_run, cited_artifact_values,
                    )
                    if compatibility_error is not None:
                        raise StoreError(
                            f"tool-backed observation {item.id} violates "
                            f"{TOOL_EVIDENCE_POLICY_VERSION}: {compatibility_error}"
                        )
                    self._require_single_observation_source(
                        connection, item, producer_run, cited_artifact_values,
                        parser_replay_cache,
                    )
            elif (item.run_id
                  and item.authority == AuthorityClass.COMPILER_DECISION):
                run_row = connection.execute(
                    "SELECT payload_json FROM runs WHERE snapshot_id=? AND id=?",
                    (item.snapshot_id, item.run_id),
                ).fetchone()
                producer_run = (ToolRun.from_dict(json.loads(run_row[0]))
                                if run_row else None)
                if producer_run is not None and run_claims_tool_truth(producer_run):
                    self._validate_execution_attestation(connection, producer_run)
                    compatibility_error = tool_evidence_compatibility_error(
                        item, producer_run, cited_artifact_values,
                    )
                    if compatibility_error is not None:
                        raise StoreError(
                            f"run-backed compiler observation {item.id} violates "
                            f"{TOOL_EVIDENCE_POLICY_VERSION}: {compatibility_error}"
                        )
                    self._require_single_observation_source(
                        connection, item, producer_run, cited_artifact_values,
                        parser_replay_cache,
                    )
            payload = self._payload(item)
            if self._immutable_payload(connection, "observations", item.id, payload):
                continue
            connection.execute(
                "INSERT INTO observations(id,snapshot_id,subject_id,predicate,stage,authority,"
                "run_id,artifact_id,payload_json) VALUES(?,?,?,?,?,?,?,?,?)",
                (item.id, item.snapshot_id, item.subject_id, item.predicate, item.stage,
                 str(item.authority), item.run_id, item.artifact_id, payload),
            )

    def _require_single_observation_source(
        self,
        connection: sqlite3.Connection,
        item: Observation,
        run: ToolRun,
        cited_artifacts: list[ArtifactRef],
        parser_replay_cache: dict[
            tuple[str, str, str], tuple[Observation, ...]
        ],
    ) -> None:
        """Close executable typed evidence to one parser-owned report.

        Historical/runless observations retain their prior identity and remain
        queryable, but a run claiming present tool truth cannot be published
        unless its parser has committed predicate/value/unit to one exact
        declared output.  Multi-report joins require a future predicate-specific
        contract and therefore fail closed here.
        """

        source = item.source
        if source is None:
            raise StoreError(
                f"tool-backed observation {item.id} lacks parser-issued source provenance"
            )
        if (item.artifact_id is None or item.anchor is None
                or item.artifact_id != item.anchor.artifact_id
                or source.artifact_id != item.artifact_id):
            raise StoreError(
                f"tool-backed observation {item.id} does not have one canonical source report"
            )
        if len(cited_artifacts) != 1 or cited_artifacts[0].id != source.artifact_id:
            raise StoreError(
                f"tool-backed observation {item.id} cites multiple or missing source reports"
            )
        artifact = cited_artifacts[0]
        if source.artifact_sha256 != artifact.sha256:
            raise StoreError(
                f"tool-backed observation {item.id} parser source hash does not match artifact"
            )
        source_error = source.validation_error(
            predicate=item.predicate, value=item.value, unit=item.unit,
        )
        if source_error is not None:
            raise StoreError(f"tool-backed observation {item.id}: {source_error}")

        manifest_row = connection.execute(
            "SELECT manifest_json FROM snapshot_manifests WHERE snapshot_id=?",
            (item.snapshot_id,),
        ).fetchone()
        if manifest_row is None:
            raise StoreError(
                f"tool-backed observation {item.id} has no immutable snapshot manifest"
            )
        manifest = ProjectManifest.from_dict(json.loads(manifest_row[0]))
        output_path = artifact.metadata.get("declared_output_path")
        if not isinstance(output_path, str) or not output_path:
            raise StoreError(
                f"tool-backed observation {item.id} source has no declared output path"
            )
        matching_specs = [
            spec for spec in manifest.stage_outputs.get(run.stage, [])
            if spec.path == output_path
        ]
        if len(matching_specs) != 1:
            raise StoreError(
                f"tool-backed observation {item.id} source path lacks unique declaration"
            )
        spec = matching_specs[0]
        if (spec.kind != artifact.kind or spec.role != artifact.role
                or str(spec.access) != str(artifact.access)
                or spec.license != artifact.license):
            raise StoreError(
                f"tool-backed observation {item.id} source disagrees with declared output"
            )

        owned = []
        for row in connection.execute(
            "SELECT a.payload_json FROM artifacts a "
            "JOIN snapshot_artifacts sa ON a.id=sa.artifact_id "
            "WHERE sa.snapshot_id=?",
            (item.snapshot_id,),
        ).fetchall():
            candidate = ArtifactRef.from_dict(json.loads(row[0]))
            if (candidate.producer_run_id == run.id
                    and candidate.metadata.get("declared_output_path") == output_path):
                owned.append(candidate.id)
        if owned != [artifact.id]:
            raise StoreError(
                f"tool-backed observation {item.id} source path has ambiguous run ownership"
            )

        snapshot_row = connection.execute(
            "SELECT payload_json FROM snapshots WHERE id=?", (item.snapshot_id,),
        ).fetchone()
        if snapshot_row is None:
            raise StoreError(f"tool-backed observation {item.id} has no snapshot")
        snapshot = DesignSnapshot.from_dict(json.loads(snapshot_row[0]))
        graph = self._load_graph_from_connection(connection, item.snapshot_id)
        from ..extract.observation_replay import replay_observation_source_error
        replay_error = replay_observation_source_error(
            project_root=self.path.parent.parent,
            manifest=manifest,
            snapshot=snapshot,
            graph=graph,
            artifact=artifact,
            observation=item,
            cache=parser_replay_cache,
        )
        if replay_error is not None:
            raise StoreError(
                f"tool-backed observation {item.id} parser replay failed: {replay_error}"
            )

    def add_observations(self, values: list[Observation]) -> None:
        with self.write() as connection:
            self._add_observations(connection, values)

    def observations(self, snapshot_id: str, *, subject_id: str | None = None,
                     predicate: str | None = None) -> list[Observation]:
        clauses = ["snapshot_id=?"]
        params: list[Any] = [snapshot_id]
        if subject_id:
            clauses.append("subject_id=?")
            params.append(subject_id)
        if predicate:
            clauses.append("predicate=?")
            params.append(predicate)
        sql = "SELECT payload_json FROM observations WHERE " + " AND ".join(clauses) + " ORDER BY id"
        with self.read() as connection:
            return [Observation.from_dict(json.loads(row[0]))
                    for row in connection.execute(sql, params).fetchall()]

    def _add_diagnostics(
        self,
        connection: sqlite3.Connection,
        values: list[Diagnostic],
        *,
        future_run_ids: frozenset[str] = frozenset(),
    ) -> None:
        for item in values:
            item = self._revalidate_model(item, Diagnostic, "diagnostic")
            if item.subject_id:
                self._require_subject(connection, item.snapshot_id, item.subject_id)
            if item.artifact_id:
                self._require_artifact(connection, item.snapshot_id, item.artifact_id)
            if item.anchor:
                self._require_artifact(connection, item.snapshot_id, item.anchor.artifact_id)
            if item.run_id and item.run_id not in future_run_ids:
                self._require_run(connection, item.snapshot_id, item.run_id)
            payload = self._payload(item)
            if self._immutable_payload(connection, "diagnostics", item.id, payload):
                continue
            connection.execute(
                "INSERT INTO diagnostics(id,snapshot_id,code,severity,stage,payload_json) "
                "VALUES(?,?,?,?,?,?)",
                (item.id, item.snapshot_id, item.code, str(item.severity), item.stage,
                 payload),
            )

    def add_diagnostics(self, values: list[Diagnostic]) -> None:
        with self.write() as connection:
            self._add_diagnostics(connection, values)

    def diagnostics(self, snapshot_id: str) -> list[Diagnostic]:
        with self.read() as connection:
            return [Diagnostic.from_dict(json.loads(row[0])) for row in connection.execute(
                "SELECT payload_json FROM diagnostics WHERE snapshot_id=? ORDER BY id", (snapshot_id,)
            ).fetchall()]

    def active_diagnostics(self, snapshot_id: str) -> list[Diagnostic]:
        """Diagnostics for the successful index attempt that produced the active view.

        Unscoped diagnostics are retained for compatibility and non-index tool
        events. Failed-attempt diagnostics remain available through diagnostics().
        """
        values = self.diagnostics(snapshot_id)
        successful = [run for run in self.runs(snapshot_id)
                      if run.stage == "index" and str(run.status) == "succeeded"]
        if not successful:
            return ([item for item in values if item.run_id is None]
                    if self.has_graph(snapshot_id) else [])
        run_id = successful[-1].id
        return [item for item in values if item.run_id in {None, run_id}]

    def _add_run(
        self,
        connection: sqlite3.Connection,
        run: ToolRun,
        *,
        future_evidence_ids: frozenset[str] = frozenset(),
        future_diagnostic_ids: frozenset[str] = frozenset(),
        future_execution_attestation: ExecutionAttestation | None = None,
    ) -> None:
        run = self._revalidate_model(run, ToolRun, "tool run")
        if not connection.execute(
            "SELECT 1 FROM snapshots WHERE id=?", (run.snapshot_id,)
        ).fetchone():
            raise StoreError(f"unknown run snapshot: {run.snapshot_id}")
        if len(set(run.input_artifact_ids)) != len(run.input_artifact_ids):
            raise StoreError("run input_artifact_ids contains duplicates")
        if len(set(run.output_artifact_ids)) != len(run.output_artifact_ids):
            raise StoreError("run output_artifact_ids contains duplicates")
        overlap = set(run.input_artifact_ids) & set(run.output_artifact_ids)
        if overlap:
            raise StoreError(
                "run input and output artifacts must be disjoint: "
                + ", ".join(sorted(overlap))
            )
        claims_tool_truth = run_claims_tool_truth(run)
        if claims_tool_truth:
            trust_error = real_tool_run_claim_error(run)
            if trust_error is not None:
                raise StoreError(
                    f"trusted run {run.id} has an invalid real-tool claim: "
                    f"{trust_error}"
                )
            manifest = self._snapshot_manifest_for_evidence(
                connection, run.snapshot_id,
            )
            identity_error = tool_run_manifest_identity_error(run, manifest)
            if identity_error is not None:
                raise StoreError(
                    f"trusted run {run.id} {identity_error}"
                )
            base_rows = connection.execute(
                "SELECT a.payload_json FROM artifacts a JOIN snapshot_artifacts sa "
                "ON a.id=sa.artifact_id WHERE sa.snapshot_id=?",
                (run.snapshot_id,),
            ).fetchall()
            base_artifact_ids = {
                artifact.id
                for artifact in (
                    ArtifactRef.from_dict(json.loads(row[0])) for row in base_rows
                )
                if artifact.producer_run_id is None
            }
            missing_base = sorted(base_artifact_ids - set(run.input_artifact_ids))
            if missing_base:
                raise StoreError(
                    f"trusted run {run.id} omits immutable snapshot input artifacts: "
                    + ", ".join(missing_base)
                )
        artifact_ids = set(run.input_artifact_ids) | set(run.output_artifact_ids)
        if artifact_ids:
            placeholders = ",".join("?" for _ in artifact_ids)
            known = {row[0] for row in connection.execute(
                f"SELECT artifact_id FROM snapshot_artifacts WHERE snapshot_id=? "
                f"AND artifact_id IN ({placeholders})",
                (run.snapshot_id, *sorted(artifact_ids)),
            )}
            missing = sorted(artifact_ids - known)
            if missing:
                raise StoreError(
                    f"run references artifacts not attached to its snapshot: {', '.join(missing)}"
                )
        for artifact_id in run.output_artifact_ids:
            row = connection.execute(
                "SELECT a.payload_json FROM artifacts a JOIN snapshot_artifacts sa "
                "ON a.id=sa.artifact_id WHERE sa.snapshot_id=? AND a.id=?",
                (run.snapshot_id, artifact_id),
            ).fetchone()
            artifact = ArtifactRef.from_dict(json.loads(row[0])) if row else None
            if artifact is None or artifact.producer_run_id != run.id:
                raise StoreError(
                    f"run output {artifact_id!r} does not name run {run.id!r} as producer; "
                    "use commit_run_result"
                )
        for artifact_id in run.input_artifact_ids:
            row = connection.execute(
                "SELECT a.payload_json FROM artifacts a JOIN snapshot_artifacts sa "
                "ON a.id=sa.artifact_id WHERE sa.snapshot_id=? AND a.id=?",
                (run.snapshot_id, artifact_id),
            ).fetchone()
            if row and ArtifactRef.from_dict(json.loads(row[0])).producer_run_id == run.id:
                raise StoreError("a run cannot consume an artifact it claims to produce")
        if claims_tool_truth:
            self._validate_execution_attestation(
                connection, run,
                attestation=future_execution_attestation,
                require_receipt=future_execution_attestation is None,
            )
        for gate in run.gates:
            if str(gate.status) == "pass" and not gate.evidence_ids:
                raise StoreError("a passing run gate must cite at least one evidence_id")
            for evidence_id in gate.evidence_ids:
                if evidence_id == run.id:
                    raise StoreError(f"run {run.id} cannot cite itself as gate evidence")
                if not self._evidence_exists(
                    connection, run.snapshot_id, evidence_id,
                    future_evidence_ids=future_evidence_ids,
                ):
                    raise StoreError(
                        f"run gate evidence {evidence_id!r} does not exist in snapshot "
                        f"{run.snapshot_id}"
                    )
                if evidence_id not in future_evidence_ids:
                    evidence_kind = self._evidence_kind(
                        connection, run.snapshot_id, evidence_id,
                    )
                    if str(gate.status) == "pass" and evidence_kind not in {
                        "observation", "derivation", "verification", "artifact",
                    }:
                        raise StoreError(
                            "passing run gate evidence must be report-backed non-run evidence"
                        )
                    producers = self._evidence_producer_runs(
                        connection, run.snapshot_id, evidence_id,
                    )
                    if (producers and producers != {run.id}) or (
                        str(gate.status) == "pass" and producers != {run.id}
                    ):
                        raise StoreError("run gate evidence belongs to another producer run")
        for diagnostic_id in run.diagnostics:
            if diagnostic_id in future_diagnostic_ids:
                continue
            if not connection.execute(
                "SELECT 1 FROM diagnostics WHERE snapshot_id=? AND id=?",
                (run.snapshot_id, diagnostic_id),
            ).fetchone():
                raise StoreError(
                    f"run diagnostic {diagnostic_id!r} does not exist in snapshot "
                    f"{run.snapshot_id}"
                )
        payload = self._payload(run)
        if self._immutable_payload(connection, "runs", run.id, payload):
            return
        connection.execute(
            "INSERT INTO runs(id,snapshot_id,stage,status,request_hash,payload_json) "
            "VALUES(?,?,?,?,?,?)",
            (run.id, run.snapshot_id, run.stage, str(run.status), run.request_hash,
             payload),
        )

    def add_run(self, run: ToolRun) -> None:
        with self.write() as connection:
            self._add_run(connection, run)

    def _add_execution_commit(
        self,
        connection: sqlite3.Connection,
        run: ToolRun,
        attestation: ExecutionAttestation,
    ) -> ExecutionCommitReceipt:
        """Persist the public half of a consumed one-shot execution capability."""

        self._validate_execution_attestation(
            connection, run, attestation=attestation, require_receipt=False,
        )
        payload = self._payload(attestation)
        if not self._immutable_payload(
            connection, "execution_attestations", attestation.id, payload,
        ):
            connection.execute(
                "INSERT INTO execution_attestations(id,run_id,snapshot_id,payload_json) "
                "VALUES(?,?,?,?)",
                (attestation.id, run.id, run.snapshot_id, payload),
            )
        receipt = ExecutionCommitReceipt(
            attestation_id=attestation.id,
            run_id=run.id,
            snapshot_id=run.snapshot_id,
            run_payload_hash=stable_hash(json_ready(run)),
            attestation_payload_hash=stable_hash(json_ready(attestation)),
        )
        receipt_payload = self._payload(receipt)
        if not self._immutable_payload(
            connection, "execution_commit_receipts", receipt.id, receipt_payload,
        ):
            connection.execute(
                "INSERT INTO execution_commit_receipts("
                "id,attestation_id,run_id,snapshot_id,payload_json) VALUES(?,?,?,?,?)",
                (receipt.id, attestation.id, run.id, run.snapshot_id, receipt_payload),
            )
        return receipt

    def execution_attestation(self, run_id: str) -> ExecutionAttestation | None:
        with self.read() as connection:
            row = connection.execute(
                "SELECT payload_json FROM execution_attestations WHERE run_id=?",
                (run_id,),
            ).fetchone()
            return (ExecutionAttestation.from_dict(json.loads(row[0]))
                    if row is not None else None)

    def execution_commit_receipt(self, run_id: str) -> ExecutionCommitReceipt | None:
        with self.read() as connection:
            row = connection.execute(
                "SELECT payload_json FROM execution_commit_receipts WHERE run_id=?",
                (run_id,),
            ).fetchone()
            return (ExecutionCommitReceipt.from_dict(json.loads(row[0]))
                    if row is not None else None)

    def has_valid_execution_commit(self, snapshot_id: str, run_id: str) -> bool:
        """Return whether a tool-truth run has a complete, re-verifiable receipt."""

        with self.read() as connection:
            row = connection.execute(
                "SELECT payload_json FROM runs WHERE snapshot_id=? AND id=?",
                (snapshot_id, run_id),
            ).fetchone()
            if row is None:
                return False
            try:
                run = ToolRun.from_dict(json.loads(row[0]))
                if not run_claims_tool_truth(run):
                    return False
                self._validate_execution_attestation(connection, run)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError, StoreError):
                return False
            return True

    def runs(self, snapshot_id: str) -> list[ToolRun]:
        with self.read() as connection:
            return [ToolRun.from_dict(json.loads(row[0])) for row in connection.execute(
                "SELECT payload_json FROM runs WHERE snapshot_id=? ORDER BY rowid", (snapshot_id,)
            ).fetchall()]

    def _add_derivations(
        self,
        connection: sqlite3.Connection,
        values: list[Derivation],
        *,
        future_observation_ids: frozenset[str] = frozenset(),
    ) -> None:
        for item in values:
            item = self._revalidate_model(item, Derivation, "derivation")
            self._require_fact_authority(item.authority, f"derivation {item.id}")
            self._require_subject(connection, item.snapshot_id, item.subject_id)
            for reference in item.evidence_refs:
                if (reference.kind == EvidenceKind.OBSERVATION
                        and reference.target_id in future_observation_ids):
                    continue
                if (reference.kind == EvidenceKind.DERIVATION
                        and reference.target_id == item.id):
                    raise StoreError(f"derivation {item.id} cannot cite itself")
                self._resolve_evidence_ref(
                    connection, reference,
                    allowed_snapshot_ids=frozenset({item.snapshot_id}),
                    label=f"derivation {item.id}",
                )
            if item.predicate in _PHYSICAL_GATE_REPORT_KINDS:
                if any(reference.kind != EvidenceKind.OBSERVATION
                       for reference in item.evidence_refs):
                    raise StoreError(
                        f"physical gate {item.id} accepts observation evidence only"
                    )
                producer_ids: set[str] = set()
                for observation_id in item.input_observation_ids:
                    producer_ids.update(self._evidence_producer_runs(
                        connection, item.snapshot_id, observation_id,
                    ))
                # Imported/synthetic evidence with no tool producer remains visible but
                # can never be trusted by the query layer.  Once a producer is claimed,
                # however, a physical gate is indivisible: every leaf must close to the
                # same real post-route run and an explicitly typed report.
                if producer_ids:
                    if len(producer_ids) != 1:
                        raise StoreError(
                            f"physical gate {item.id} evidence must close to exactly one "
                            "post_route producer run"
                        )
                    self._require_physical_gate_evidence(
                        connection, item, next(iter(producer_ids)),
                    )
            payload = self._payload(item)
            if self._immutable_payload(connection, "derivations", item.id, payload):
                continue
            connection.execute(
                "INSERT INTO derivations(id,snapshot_id,subject_id,predicate,payload_json) "
                "VALUES(?,?,?,?,?)", (item.id, item.snapshot_id, item.subject_id,
                                       item.predicate, payload),
            )

    def add_derivations(self, values: list[Derivation]) -> None:
        with self.write() as connection:
            self._add_derivations(connection, values)

    def derivations(self, snapshot_id: str) -> list[dict[str, Any]]:
        with self.read() as connection:
            return [json.loads(row[0]) for row in connection.execute(
                "SELECT payload_json FROM derivations WHERE snapshot_id=? ORDER BY id", (snapshot_id,)
            ).fetchall()]

    def _add_verifications(
        self,
        connection: sqlite3.Connection,
        values: list[VerificationResult],
        *,
        future_run_ids: frozenset[str] = frozenset(),
        future_evidence_ids: frozenset[str] = frozenset(),
    ) -> None:
        for item in values:
            item = self._revalidate_model(item, VerificationResult, "verification")
            if item.run_id and item.run_id not in future_run_ids:
                self._require_run(connection, item.snapshot_id, item.run_id)
            referenced_run = None
            if item.run_id:
                row = connection.execute(
                    "SELECT payload_json FROM runs WHERE snapshot_id=? AND id=?",
                    (item.snapshot_id, item.run_id),
                ).fetchone()
                referenced_run = ToolRun.from_dict(json.loads(row[0])) if row else None
                expected_stage = {
                    "csim": "csim", "rtl_cosim": "rtl_cosim",
                }.get(str(item.kind))
                if expected_stage and referenced_run and referenced_run.stage != expected_stage:
                    raise StoreError(
                        f"verification {item.id} kind requires a {expected_stage} producer run"
                    )
            evidence_producers: set[str] = set()
            evidence_workloads: set[str] = set()
            for evidence_id in item.evidence_ids:
                if evidence_id == item.id:
                    raise StoreError(f"verification {item.id} cannot cite itself as evidence")
                if not self._evidence_exists(
                    connection, item.snapshot_id, evidence_id,
                    future_evidence_ids=future_evidence_ids,
                ):
                    raise StoreError(
                        f"verification evidence {evidence_id!r} does not exist in snapshot "
                        f"{item.snapshot_id}"
                    )
                evidence_kind = self._evidence_kind(
                    connection, item.snapshot_id, evidence_id,
                )
                if str(item.status) == "pass" and evidence_kind not in {
                    "observation", "derivation", "artifact",
                }:
                    raise StoreError(
                        "passing verification evidence must be an observation, derivation, "
                        "or report artifact; a ToolRun ID is not evidence"
                    )
                evidence_producers.update(self._evidence_producer_runs(
                    connection, item.snapshot_id, evidence_id,
                ))
                evidence_workloads.update(self._evidence_workloads(
                    connection, item.snapshot_id, evidence_id,
                ))
            if item.run_id:
                if (evidence_producers and evidence_producers != {item.run_id}) or (
                    str(item.status) == "pass" and evidence_producers != {item.run_id}
                ):
                    raise StoreError(
                        f"verification {item.id} evidence does not belong to its producer run"
                    )
                run_workload = (str(referenced_run.metadata.get("workload_id"))
                                if referenced_run
                                and referenced_run.metadata.get("workload_id") else None)
                expected_workload = item.workload_id or run_workload
                if (item.workload_id and run_workload
                        and item.workload_id != run_workload):
                    raise StoreError("verification workload does not match its producer run")
                if expected_workload and evidence_workloads - {expected_workload}:
                    raise StoreError("verification evidence belongs to another workload")
                if str(item.status) == "pass" and referenced_run is not None:
                    self._require_typed_verification_evidence(
                        connection, item, referenced_run,
                    )
            payload = self._payload(item)
            if self._immutable_payload(connection, "verifications", item.id, payload):
                continue
            connection.execute(
                "INSERT INTO verifications(id,snapshot_id,kind,status,payload_json) "
                "VALUES(?,?,?,?,?)", (item.id, item.snapshot_id, str(item.kind),
                                       str(item.status), payload),
            )

    def add_verifications(self, values: list[VerificationResult]) -> None:
        with self.write() as connection:
            self._add_verifications(connection, values)

    def verifications(self, snapshot_id: str) -> list[dict[str, Any]]:
        with self.read() as connection:
            return [json.loads(row[0]) for row in connection.execute(
                "SELECT payload_json FROM verifications WHERE snapshot_id=? ORDER BY id", (snapshot_id,)
            ).fetchall()]

    @staticmethod
    def _require_batch_snapshot(snapshot_id: str, values: list[Any], label: str) -> None:
        mismatched = [item.id for item in values if item.snapshot_id != snapshot_id]
        if mismatched:
            raise StoreError(
                f"{label} belong to another snapshot: {', '.join(sorted(mismatched))}"
            )

    def commit_run_result(
        self,
        *,
        run: ToolRun,
        artifacts: list[ArtifactRef] | None = None,
        observations: list[Observation] | None = None,
        derivations: list[Derivation] | None = None,
        verifications: list[VerificationResult] | None = None,
        diagnostics: list[Diagnostic] | None = None,
        execution_authorization: object | None = None,
    ) -> None:
        """Atomically persist a run and any outputs/evidence it produces.

        This is the write path for the otherwise circular run/output contract:
        produced artifacts may cite this future run, the run may list those output
        artifacts, and its gates may cite observations/derivations/verifications in
        the same batch.  Individual ``add_run``/``add_artifact`` calls remain strict
        and require all referenced rows to exist already.
        """
        claims_tool_truth = run_claims_tool_truth(run)
        attestation: ExecutionAttestation | None = None
        if claims_tool_truth:
            if execution_authorization is None:
                raise StoreError(
                    "tool-truth commit requires a StageOrchestrator-issued authorization"
                )
            try:
                attestation = _consume_execution_authorization(
                    execution_authorization,
                )
            except RunnerProtocolError as exc:
                raise StoreError(str(exc)) from exc
        elif execution_authorization is not None:
            raise StoreError(
                "execution authorization cannot be attached to a non-tool-truth run"
            )
        artifacts = list(artifacts or [])
        observations = list(observations or [])
        derivations = list(derivations or [])
        verifications = list(verifications or [])
        diagnostics = list(diagnostics or [])
        snapshot_id = run.snapshot_id
        self._require_batch_snapshot(snapshot_id, observations, "observations")
        self._require_batch_snapshot(snapshot_id, derivations, "derivations")
        self._require_batch_snapshot(snapshot_id, verifications, "verifications")
        self._require_batch_snapshot(snapshot_id, diagnostics, "diagnostics")
        observation_ids = frozenset(item.id for item in observations)
        derivation_ids = frozenset(item.id for item in derivations)
        verification_ids = frozenset(item.id for item in verifications)
        diagnostic_ids = frozenset(item.id for item in diagnostics)
        evidence_ids = observation_ids | derivation_ids | verification_ids | diagnostic_ids
        supplied_artifacts = {item.id: item for item in artifacts}
        if len(supplied_artifacts) != len(artifacts):
            raise StoreError("run batch contains duplicate artifact identifiers")
        declared_outputs = set(run.output_artifact_ids)
        if len(declared_outputs) != len(run.output_artifact_ids):
            raise StoreError("run output_artifact_ids contains duplicates")
        produced_in_batch = {
            item.id for item in artifacts if item.producer_run_id == run.id
        }
        if declared_outputs != produced_in_batch:
            missing = sorted(declared_outputs - produced_in_batch)
            undeclared = sorted(produced_in_batch - declared_outputs)
            details = []
            if missing:
                details.append("declared outputs without matching batch producer: "
                               + ", ".join(missing))
            if undeclared:
                details.append("produced batch artifacts absent from run outputs: "
                               + ", ".join(undeclared))
            raise StoreError("run/output artifact contract mismatch; " + "; ".join(details))
        with self.write() as connection:
            for artifact in artifacts:
                self._add_artifact(
                    connection, snapshot_id, artifact,
                    future_run_ids=frozenset({run.id}),
                )
            self._add_run(
                connection, run, future_evidence_ids=evidence_ids,
                future_diagnostic_ids=diagnostic_ids,
                future_execution_attestation=attestation,
            )
            if attestation is not None:
                self._add_execution_commit(connection, run, attestation)
            self._add_observations(connection, observations)
            self._add_derivations(connection, derivations)
            self._add_verifications(
                connection, verifications,
                future_evidence_ids=verification_ids | diagnostic_ids,
            )
            self._add_diagnostics(connection, diagnostics)
            # Re-run the public run contract now that all future gate evidence
            # and diagnostic rows exist; this closes batch-only lineage gaps.
            self._add_run(connection, run)

    def commit_index_success(
        self,
        *,
        project_id: str,
        graph: CanonicalGraph,
        run: ToolRun,
        observations: list[Observation],
        derivations: list[Derivation],
        verifications: list[VerificationResult],
        diagnostics: list[Diagnostic],
        materialization: ActionMaterialization | None = None,
    ) -> None:
        """Atomically publish one successful canonical index result.

        All evidence identifiers are validated against the union of persisted rows
        and this batch before the active pointer moves.  A failure at any point rolls
        back graph projection, run, evidence, diagnostics, FTS rows, and activation.
        """
        snapshot_id = graph.snapshot_id
        graph_snapshot = self.snapshot(snapshot_id)
        if run.snapshot_id != snapshot_id:
            raise StoreError("index run belongs to another snapshot")
        if run.stage != "index" or str(run.status) not in {"succeeded", "cached"}:
            raise StoreError("successful index commit requires a succeeded/cached index run")
        if materialization is not None:
            if str(materialization.status) != "materialized":
                raise StoreError(
                    "successful index materialization must use materialized status"
                )
            if (materialization.result_snapshot_id != snapshot_id
                    or materialization.action_id != graph_snapshot.action_id
                    or materialization.parent_snapshot_id
                    != graph_snapshot.parent_snapshot_id):
                raise StoreError(
                    "successful index materialization does not match graph snapshot lineage"
                )
        self._require_batch_snapshot(snapshot_id, observations, "observations")
        self._require_batch_snapshot(snapshot_id, derivations, "derivations")
        self._require_batch_snapshot(snapshot_id, verifications, "verifications")
        self._require_batch_snapshot(snapshot_id, diagnostics, "diagnostics")
        observation_ids = frozenset(item.id for item in observations)
        derivation_ids = frozenset(item.id for item in derivations)
        verification_ids = frozenset(item.id for item in verifications)
        diagnostic_ids = frozenset(item.id for item in diagnostics)
        evidence_ids = observation_ids | derivation_ids | verification_ids | diagnostic_ids
        all_ids = [run.id, *(item.id for item in observations),
                   *(item.id for item in derivations), *(item.id for item in verifications),
                   *(item.id for item in diagnostics)]
        if len(all_ids) != len(set(all_ids)):
            raise StoreError("index batch contains duplicate evidence identifiers")
        with self.write() as connection:
            self._save_graph(connection, graph)
            self._add_run(
                connection, run, future_evidence_ids=evidence_ids,
                future_diagnostic_ids=diagnostic_ids,
            )
            self._add_observations(connection, observations)
            self._add_derivations(connection, derivations)
            self._add_verifications(
                connection, verifications,
                future_evidence_ids=verification_ids | diagnostic_ids,
            )
            self._add_diagnostics(connection, diagnostics)
            if materialization is not None:
                self._add_materializations(connection, [materialization])
            self._add_run(connection, run)
            self._set_active_snapshot(connection, project_id, snapshot_id)

    def commit_index_failure(
        self, *, run: ToolRun, diagnostics: list[Diagnostic],
        materialization: ActionMaterialization | None = None,
    ) -> None:
        """Atomically retain a failed index event without publishing a graph view."""
        if run.stage != "index" or str(run.status) in {"succeeded", "cached"}:
            raise StoreError("failed index commit requires a non-successful index run")
        if materialization is not None:
            if str(materialization.status) not in {"no_op", "failed"}:
                raise StoreError(
                    "failed index materialization must use no_op or failed status"
                )
            candidate = self.snapshot(run.snapshot_id)
            if (candidate.action_id != materialization.action_id
                    or candidate.parent_snapshot_id
                    != materialization.parent_snapshot_id):
                raise StoreError(
                    "failed index materialization does not match candidate snapshot lineage"
                )
        self._require_batch_snapshot(run.snapshot_id, diagnostics, "diagnostics")
        diagnostic_ids = frozenset(item.id for item in diagnostics)
        with self.write() as connection:
            self._add_run(
                connection, run, future_diagnostic_ids=diagnostic_ids,
            )
            self._add_diagnostics(connection, diagnostics)
            if materialization is not None:
                self._add_materializations(connection, [materialization])
            self._add_run(connection, run)

    def add_variant(self, value: VariantAction) -> None:
        value = self._revalidate_model(value, VariantAction, "variant action")
        with self.write() as connection:
            if not connection.execute(
                "SELECT 1 FROM snapshots WHERE id=?", (value.parent_snapshot_id,)
            ).fetchone():
                raise KeyError(value.parent_snapshot_id)
            if not connection.execute(
                "SELECT 1 FROM graph_views WHERE snapshot_id=?", (value.parent_snapshot_id,)
            ).fetchone():
                raise StoreError("variant actions require a successful parent graph")
            if value.scope_id:
                self._require_subject(connection, value.parent_snapshot_id, value.scope_id)
            # Preserve compatibility with opaque legacy snapshot lineage, but
            # do not allow a subsequently imported action row to contradict
            # any snapshot that already names that action identifier.
            for row in connection.execute(
                "SELECT payload_json FROM snapshots ORDER BY rowid"
            ).fetchall():
                snapshot = DesignSnapshot.from_dict(json.loads(row[0]))
                if (snapshot.action_id == value.id
                        and snapshot.parent_snapshot_id != value.parent_snapshot_id):
                    raise StoreError(
                        "variant action parent_snapshot_id conflicts with existing "
                        "snapshot lineage"
                    )
            payload = self._payload(value)
            if self._immutable_payload(connection, "variants", value.id, payload):
                return
            connection.execute(
                "INSERT INTO variants(id,parent_snapshot_id,kind,payload_json) VALUES(?,?,?,?)",
                (value.id, value.parent_snapshot_id, value.kind, payload),
            )

    def variants(self, parent_snapshot_id: str | None = None, *,
                 action_id: str | None = None) -> list[dict[str, Any]]:
        with self.read() as connection:
            clauses = []
            values: list[str] = []
            if parent_snapshot_id:
                clauses.append("parent_snapshot_id=?")
                values.append(parent_snapshot_id)
            if action_id:
                clauses.append("id=?")
                values.append(action_id)
            where = " WHERE " + " AND ".join(clauses) if clauses else ""
            rows = connection.execute(
                f"SELECT payload_json FROM variants{where} ORDER BY id", values,
            ).fetchall()
            return [json.loads(row[0]) for row in rows]

    def variant(self, action_id: str) -> dict[str, Any] | None:
        values = self.variants(action_id=action_id)
        return values[0] if values else None

    def _add_materializations(
        self,
        connection: sqlite3.Connection,
        values: list[ActionMaterialization],
        *,
        future_diagnostics: dict[str, str] | None = None,
    ) -> None:
        future_diagnostics = dict(future_diagnostics or {})
        for item in values:
            item = self._revalidate_model(
                item, ActionMaterialization, "action materialization",
            )
            action = connection.execute(
                "SELECT parent_snapshot_id FROM variants WHERE id=?",
                (item.action_id,),
            ).fetchone()
            if action is None:
                raise StoreError(
                    f"materialization action does not exist: {item.action_id}"
                )
            if str(action[0]) != item.parent_snapshot_id:
                raise StoreError(
                    "materialization parent_snapshot_id does not match its action"
                )
            parent = connection.execute(
                "SELECT project_id FROM snapshots WHERE id=?",
                (item.parent_snapshot_id,),
            ).fetchone()
            if parent is None:
                raise StoreError(
                    f"materialization parent snapshot does not exist: "
                    f"{item.parent_snapshot_id}"
                )
            project_id = str(parent[0])
            allowed_snapshots = {item.parent_snapshot_id}
            # Candidate snapshots naming the action are legitimate locations
            # for failed/no-op attempt diagnostics even when no result is
            # published by the materialization record.
            for row in connection.execute(
                "SELECT payload_json FROM snapshots WHERE project_id=?",
                (project_id,),
            ).fetchall():
                snapshot = DesignSnapshot.from_dict(json.loads(row[0]))
                if (snapshot.action_id == item.action_id
                        and snapshot.parent_snapshot_id == item.parent_snapshot_id):
                    allowed_snapshots.add(snapshot.id)
            if item.result_snapshot_id is not None:
                try:
                    result = next(
                        snapshot for snapshot in (
                            DesignSnapshot.from_dict(json.loads(row[0]))
                            for row in connection.execute(
                                "SELECT payload_json FROM snapshots WHERE id=?",
                                (item.result_snapshot_id,),
                            ).fetchall()
                        )
                    )
                except StopIteration as exc:
                    raise StoreError(
                        f"materialization result snapshot does not exist: "
                        f"{item.result_snapshot_id}"
                    ) from exc
                if (result.project_id != project_id
                        or result.action_id != item.action_id
                        or result.parent_snapshot_id != item.parent_snapshot_id):
                    raise StoreError(
                        "materialization result snapshot has inconsistent action lineage"
                    )
                allowed_snapshots.add(result.id)
            for diagnostic_id in item.diagnostic_ids:
                row = connection.execute(
                    "SELECT snapshot_id FROM diagnostics WHERE id=?",
                    (diagnostic_id,),
                ).fetchone()
                diagnostic_snapshot = (
                    str(row[0]) if row is not None
                    else future_diagnostics.get(diagnostic_id)
                )
                if (diagnostic_snapshot is None
                        or diagnostic_snapshot not in allowed_snapshots):
                    raise StoreError(
                        f"materialization diagnostic {diagnostic_id!r} is missing or "
                        "does not belong to this action attempt"
                    )
            for reference in item.evidence_refs:
                self._resolve_evidence_ref(
                    connection, reference,
                    allowed_snapshot_ids=frozenset(allowed_snapshots),
                    label=f"materialization {item.id}",
                )
            payload = self._payload(item)
            if self._immutable_payload(
                connection, "action_materializations", item.id, payload,
            ):
                continue
            connection.execute(
                "INSERT INTO action_materializations("
                "id,action_id,parent_snapshot_id,result_snapshot_id,status,"
                "attempted_at,payload_json) VALUES(?,?,?,?,?,?,?)",
                (item.id, item.action_id, item.parent_snapshot_id,
                 item.result_snapshot_id, str(item.status), item.attempted_at, payload),
            )

    def add_materialization(self, value: ActionMaterialization) -> None:
        self.add_materializations([value])

    def add_materializations(self, values: list[ActionMaterialization]) -> None:
        with self.write() as connection:
            self._add_materializations(connection, values)

    def materializations(
        self,
        action_id: str | None = None,
        *,
        parent_snapshot_id: str | None = None,
        status: str | None = None,
    ) -> list[ActionMaterialization]:
        clauses: list[str] = []
        params: list[str] = []
        if action_id is not None:
            clauses.append("action_id=?")
            params.append(action_id)
        if parent_snapshot_id is not None:
            clauses.append("parent_snapshot_id=?")
            params.append(parent_snapshot_id)
        if status is not None:
            clauses.append("status=?")
            params.append(status)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self.read() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM action_materializations"
                f"{where} ORDER BY attempted_at,id", params,
            ).fetchall()
            return [
                ActionMaterialization.from_dict(json.loads(row[0])) for row in rows
            ]

    def result_snapshots(self, action_id: str, *,
                         parent_snapshot_id: str | None = None) -> list[DesignSnapshot]:
        """Return only snapshots with an explicit, matching action lineage.

        Snapshot payloads are decoded instead of name-matched or inferred from
        graph differences.  Failed index candidates are intentionally retained;
        callers can use ``has_graph`` to distinguish published graph views.
        """
        with self.read() as connection:
            snapshots = [DesignSnapshot.from_dict(json.loads(row[0])) for row in
                         connection.execute(
                             "SELECT payload_json FROM snapshots ORDER BY rowid"
                         ).fetchall()]
        return sorted(
            (item for item in snapshots
             if item.action_id == action_id
             and (parent_snapshot_id is None
                  or item.parent_snapshot_id == parent_snapshot_id)),
            key=lambda item: item.id,
        )

    def add_prediction(self, value: PredictionEnvelope) -> None:
        value = self._revalidate_model(value, PredictionEnvelope, "prediction")
        with self.write() as connection:
            self._require_subject(connection, value.snapshot_id, value.subject_id)
            if value.action_id is not None:
                action = connection.execute(
                    "SELECT parent_snapshot_id FROM variants WHERE id=?",
                    (value.action_id,),
                ).fetchone()
                if action is None:
                    raise KeyError(value.action_id)
                if action[0] != value.snapshot_id:
                    raise StoreError(
                        "prediction action must belong to its input snapshot"
                    )
            payload = self._payload(value)
            if self._immutable_payload(connection, "predictions", value.id, payload):
                return
            connection.execute(
                "INSERT INTO predictions(id,snapshot_id,subject_id,predicate,payload_json) "
                "VALUES(?,?,?,?,?)", (value.id, value.snapshot_id, value.subject_id,
                                       value.predicate, payload),
            )

    def predictions(self, snapshot_id: str) -> list[dict[str, Any]]:
        with self.read() as connection:
            return [json.loads(row[0]) for row in connection.execute(
                "SELECT payload_json FROM predictions WHERE snapshot_id=? ORDER BY id", (snapshot_id,)
            ).fetchall()]

    def add_knowledge_rules(self, values: list[KnowledgeRule]) -> None:
        values = [self._revalidate_model(item, KnowledgeRule, "knowledge rule")
                  for item in values]
        with self.write() as connection:
            for item in values:
                payload = self._payload(item)
                if self._immutable_payload(connection, "knowledge_rules", item.id, payload):
                    continue
                connection.execute(
                    "INSERT INTO knowledge_rules(id,document_id,document_version,section,payload_json) "
                    "VALUES(?,?,?,?,?)", (item.id, item.document_id, item.document_version,
                                           item.section, payload),
                )
                self._index_knowledge_rule(connection, item)

    def _index_knowledge_rule(
        self, connection: sqlite3.Connection, item: KnowledgeRule,
    ) -> None:
        fts = connection.execute(
            "SELECT value FROM schema_info WHERE key='knowledge_fts5'"
        ).fetchone()
        if not fts or fts[0] != "1":
            return
        connection.execute(
            "DELETE FROM knowledge_rules_fts WHERE knowledge_rule_id=?", (item.id,),
        )
        connection.execute(
            "INSERT INTO knowledge_rules_fts("
            "knowledge_rule_id,rule_id,document_id,document_version,section,title,summary) "
            "VALUES(?,?,?,?,?,?,?)",
            (item.id, item.rule_id, item.document_id, item.document_version,
             item.section, item.title, item.summary or ""),
        )

    def install_knowledge_pack(
        self,
        *,
        pack_id: str,
        pack_schema_version: str,
        content_hash: str,
        installed_at: str,
        inventory: dict[str, Any],
        rules: list[KnowledgeRule],
        bindings: list[KnowledgeBinding],
        coverage: CoverageManifest | None,
    ) -> bool:
        """Atomically install one immutable, already-validated knowledge pack.

        Reopening a bundle never invokes this operation.  Callers must select an
        install/sync action explicitly; an existing pack ID with different bytes
        is rejected rather than silently changing guidance in an old bundle.
        """
        if not isinstance(pack_id, str) or not pack_id.strip():
            raise StoreError("knowledge pack_id is required")
        if not isinstance(pack_schema_version, str) or not pack_schema_version.strip():
            raise StoreError("knowledge pack schema version is required")
        if (not isinstance(content_hash, str)
                or not __import__("re").fullmatch(r"[0-9a-f]{64}", content_hash)):
            raise StoreError("knowledge pack content_hash must be lowercase SHA-256")
        try:
            reject_embedded_body_fields(inventory, "knowledge pack inventory")
        except ValueError as exc:
            raise StoreError(str(exc)) from exc
        rules = [self._revalidate_model(item, KnowledgeRule, "knowledge rule")
                 for item in rules]
        bindings = [self._revalidate_model(item, KnowledgeBinding, "knowledge binding")
                    for item in bindings]
        if coverage is not None:
            coverage = self._revalidate_model(
                coverage, CoverageManifest, "knowledge coverage",
            )
            if coverage.pack_id != pack_id:
                raise StoreError("knowledge coverage belongs to another pack")
        supplied_rule_ids = {item.id for item in rules}
        supplied_binding_ids = {item.id for item in bindings}
        if len(supplied_rule_ids) != len(rules) or len(supplied_binding_ids) != len(bindings):
            raise StoreError("knowledge pack contains duplicate rule or binding IDs")
        if any(item.knowledge_rule_id not in supplied_rule_ids for item in bindings):
            raise StoreError("knowledge binding references a rule outside its pack")
        if inventory.get("pack_id") != pack_id:
            raise StoreError("knowledge inventory belongs to another pack")
        if set(inventory.get("rule_ids", [])) != supplied_rule_ids:
            raise StoreError("knowledge inventory rule IDs do not match supplied rules")
        if set(inventory.get("binding_ids", [])) != supplied_binding_ids:
            raise StoreError("knowledge inventory binding IDs do not match supplied bindings")
        if coverage is not None:
            if inventory.get("coverage_id") != coverage.id:
                raise StoreError("knowledge inventory coverage ID does not match the manifest")
            if inventory.get("coverage_scope") != coverage.coverage_scope:
                raise StoreError("knowledge inventory coverage scope does not match the manifest")
            if (inventory.get("target_registry_version")
                    != coverage.target_registry_version):
                raise StoreError(
                    "knowledge inventory target registry version does not match the manifest"
                )
            for entry in coverage.entries:
                if any(item not in supplied_rule_ids for item in entry.rule_ids):
                    raise StoreError("knowledge coverage references a rule outside its pack")
                if any(item not in supplied_binding_ids for item in entry.binding_ids):
                    raise StoreError("knowledge coverage references a binding outside its pack")
            rule_coverage: Counter[str] = Counter()
            binding_coverage: Counter[str] = Counter()
            bindings_by_id = {item.id: item for item in bindings}
            for entry in coverage.entries:
                if entry.status != CoverageStatus.RULE:
                    continue
                rule_coverage.update(entry.rule_ids)
                binding_coverage.update(entry.binding_ids)
                if any(
                    bindings_by_id[item].knowledge_rule_id not in entry.rule_ids
                    for item in entry.binding_ids
                ):
                    raise StoreError(
                        "knowledge coverage places a binding under a different rule"
                    )
            if (set(rule_coverage) != supplied_rule_ids
                    or any(count != 1 for count in rule_coverage.values())):
                raise StoreError(
                    "every knowledge rule must be covered exactly once by a rule entry"
                )
            if (set(binding_coverage) != supplied_binding_ids
                    or any(count != 1 for count in binding_coverage.values())):
                raise StoreError(
                    "every knowledge binding must be covered exactly once by a rule entry"
                )
            if coverage.target_inventory or bindings:
                try:
                    expected_targets = canonical_supported_targets(
                        coverage.coverage_scope,
                        coverage.target_registry_version,
                    )
                except KeyError as exc:
                    raise StoreError(str(exc)) from exc
                actual_targets = {
                    (item.target_kind, item.target)
                    for item in coverage.target_inventory
                }
                if actual_targets != expected_targets:
                    raise StoreError(
                        "knowledge target inventory does not match its canonical registry"
                    )
            covered_target_bindings: set[str] = set()
            for target in coverage.target_inventory:
                for binding_id in target.binding_ids:
                    binding = bindings_by_id.get(binding_id)
                    if binding is None or (
                        binding.target_kind, binding.target,
                    ) != (target.target_kind, target.target):
                        raise StoreError(
                            "knowledge target inventory references a missing or different binding"
                        )
                    covered_target_bindings.add(binding_id)
            if covered_target_bindings != supplied_binding_ids:
                raise StoreError(
                    "knowledge target inventory must cover every binding exactly once"
                )
        elif pack_schema_version != "1.0" and (supplied_rule_ids or supplied_binding_ids):
            raise StoreError("knowledge rules or bindings require coverage")
        if supplied_binding_ids and (
            coverage is None
            or not coverage.review_ready
            or inventory.get("review_ready") is not True
            or inventory.get("review_status") != coverage.review_status
        ):
            raise StoreError(
                "executable knowledge bindings require a review_ready pack"
            )

        inventory_payload = self._payload(inventory)
        with self.write() as connection:
            previous = connection.execute(
                "SELECT content_hash,payload_json FROM knowledge_packs WHERE pack_id=?",
                (pack_id,),
            ).fetchone()
            if previous:
                if previous[0] != content_hash or previous[1] != inventory_payload:
                    raise StoreError(
                        f"installed knowledge pack changed without a new pack_id: {pack_id}"
                    )
                return False
            for item in rules:
                payload = self._payload(item)
                if not self._immutable_payload(
                    connection, "knowledge_rules", item.id, payload,
                ):
                    connection.execute(
                        "INSERT INTO knowledge_rules("
                        "id,document_id,document_version,section,payload_json) "
                        "VALUES(?,?,?,?,?)",
                        (item.id, item.document_id, item.document_version,
                         item.section, payload),
                    )
                self._index_knowledge_rule(connection, item)
            connection.execute(
                "INSERT INTO knowledge_packs("
                "pack_id,pack_schema_version,content_hash,installed_at,payload_json) "
                "VALUES(?,?,?,?,?)",
                (pack_id, pack_schema_version, content_hash, installed_at,
                 inventory_payload),
            )
            for item in bindings:
                payload = self._payload(item)
                if self._immutable_payload(
                    connection, "knowledge_bindings", item.id, payload,
                ):
                    continue
                connection.execute(
                    "INSERT INTO knowledge_bindings("
                    "id,knowledge_rule_id,target_kind,target,payload_json) "
                    "VALUES(?,?,?,?,?)",
                    (item.id, item.knowledge_rule_id, item.target_kind,
                     item.target, payload),
                )
            if coverage is not None:
                payload = self._payload(coverage)
                if not self._immutable_payload(
                    connection, "knowledge_coverage", coverage.id, payload,
                ):
                    connection.execute(
                        "INSERT INTO knowledge_coverage("
                        "id,pack_id,coverage_scope,payload_json) VALUES(?,?,?,?)",
                        (coverage.id, pack_id, coverage.coverage_scope, payload),
                    )
        return True

    def knowledge_rules(self) -> list[KnowledgeRule]:
        with self.read() as connection:
            return [KnowledgeRule(**json.loads(row[0])) for row in connection.execute(
                "SELECT payload_json FROM knowledge_rules ORDER BY id"
            ).fetchall()]

    def installed_knowledge_packs(self) -> list[dict[str, Any]]:
        with self.read() as connection:
            result: list[dict[str, Any]] = []
            for row in connection.execute(
                "SELECT pack_id,pack_schema_version,content_hash,installed_at,payload_json "
                "FROM knowledge_packs ORDER BY pack_id"
            ).fetchall():
                value = json.loads(row[4])
                value.update({
                    "pack_id": row[0],
                    "pack_schema_version": row[1],
                    "content_hash": row[2],
                    "installed_at": row[3],
                })
                result.append(value)
            return result

    def knowledge_bindings(
        self, *, target_kind: str | None = None, target: str | None = None,
    ) -> list[KnowledgeBinding]:
        clauses: list[str] = []
        parameters: list[Any] = []
        if target_kind is not None:
            clauses.append("target_kind=?")
            parameters.append(target_kind)
        if target is not None:
            clauses.append("target=?")
            parameters.append(target)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self.read() as connection:
            return [KnowledgeBinding.from_dict(json.loads(row[0])) for row in connection.execute(
                "SELECT payload_json FROM knowledge_bindings" + where + " ORDER BY id",
                parameters,
            ).fetchall()]

    def knowledge_coverage(
        self, *, pack_id: str | None = None,
    ) -> list[CoverageManifest]:
        where = " WHERE pack_id=?" if pack_id is not None else ""
        parameters: tuple[Any, ...] = (pack_id,) if pack_id is not None else ()
        with self.read() as connection:
            return [CoverageManifest.from_dict(json.loads(row[0])) for row in connection.execute(
                "SELECT payload_json FROM knowledge_coverage" + where + " ORDER BY id",
                parameters,
            ).fetchall()]

    def search_knowledge_rules(
        self, query: str, *, limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Search citation/paraphrase fields only; no local document body is read."""
        if not isinstance(query, str) or "\x00" in query or len(query) > 4_096:
            raise StoreError(
                "knowledge query must be a string of at most 4096 characters without NUL"
            )
        query = query.strip()
        limit = max(1, min(int(limit), 100))
        if not query:
            return []
        with self.read() as connection:
            rules = [KnowledgeRule(**json.loads(row[0])) for row in connection.execute(
                "SELECT payload_json FROM knowledge_rules ORDER BY id"
            ).fetchall()]
            by_id = {item.id: item for item in rules}
            available = connection.execute(
                "SELECT value FROM schema_info WHERE key='knowledge_fts5'"
            ).fetchone()
            if available and available[0] == "1":
                tokens = [token for token in __import__("re").findall(
                    r"[\w:.+-]+", query,
                ) if token]
                if tokens:
                    expression = " AND ".join(
                        '"' + token.replace('"', '""') + '"' for token in tokens
                    )
                    try:
                        rows = connection.execute(
                            "SELECT knowledge_rule_id,bm25(knowledge_rules_fts) AS score "
                            "FROM knowledge_rules_fts WHERE knowledge_rules_fts MATCH ? "
                            "ORDER BY score,knowledge_rule_id LIMIT ?",
                            (expression, limit),
                        ).fetchall()
                    except sqlite3.OperationalError:
                        rows = []
                    if rows:
                        return [{"rule": by_id[row[0]], "score": float(row[1]),
                                 "channel": "fts5"}
                                for row in rows if row[0] in by_id]
            needle = query.casefold()
            matches = [item for item in rules if needle in " ".join((
                item.rule_id, item.document_id, item.section, item.title,
                item.summary or "",
            )).casefold()]
            return [{"rule": item, "score": 0.0, "channel": "substring"}
                    for item in matches[:limit]]

    def search_entities(self, snapshot_id: str, query: str, limit: int = 20) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 100))
        query = query.strip()
        if not query:
            return []
        with self.read() as connection:
            fts = connection.execute("SELECT value FROM schema_info WHERE key='fts5'").fetchone()
            rows: list[sqlite3.Row] = []
            if fts and fts[0] == "1":
                tokens = [token for token in __import__("re").findall(r"[\w:.+-]+", query) if token]
                if tokens:
                    expression = " AND ".join('"' + token.replace('"', '""') + '"' for token in tokens)
                    try:
                        rows = connection.execute(
                            "SELECT entity_id, bm25(entities_fts) AS score FROM entities_fts "
                            "WHERE snapshot_id=? AND entities_fts MATCH ? ORDER BY score, entity_id LIMIT ?",
                            (snapshot_id, expression, limit),
                        ).fetchall()
                    except sqlite3.OperationalError:
                        rows = []
            if rows:
                return [{"entity_id": row[0], "score": float(-row[1]), "match_type": "fts"}
                        for row in rows]
            like = f"%{query.lower()}%"
            rows = connection.execute(
                "SELECT id,name,qualified_name FROM entities WHERE snapshot_id=? AND "
                "(lower(name) LIKE ? OR lower(coalesce(qualified_name,'')) LIKE ?) ORDER BY id LIMIT ?",
                (snapshot_id, like, like, limit),
            ).fetchall()
            return [{"entity_id": row[0], "score": 1.0, "match_type": "substring"}
                    for row in rows]

    def file_hash(self) -> str:
        return hash_artifact_bytes(self.path.read_bytes()) if self.path.is_file() else stable_hash("")
