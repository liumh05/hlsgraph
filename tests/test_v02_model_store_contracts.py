from __future__ import annotations

import json
import sqlite3

import pytest

from hlsgraph.bundle import GraphBundle
from hlsgraph.graph import CanonicalGraph
from hlsgraph.manifest import minimal_manifest
from hlsgraph.model import (
    ActionMaterialization,
    ActionMaterializationStatus,
    AuthorityClass,
    DatasetManifest,
    Derivation,
    Diagnostic,
    DiagnosticSeverity,
    Entity,
    EntityCorrespondence,
    EvidenceKind,
    EvidenceRef,
    FailureClass,
    Observation,
    Relation,
    RunStatus,
    SourceAnchor,
    ToolRun,
    VariantAction,
    json_ready,
    stable_id,
)
from hlsgraph.store import LedgerStore, StoreError


def _two_snapshot_bundle(tmp_path):
    (tmp_path / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    bundle = GraphBundle.create(
        tmp_path,
        minimal_manifest("test.v02.contracts", "v0.2 contracts", "dut", "kernel.cpp"),
    )
    parent = bundle.snapshot()
    artifact = bundle.store.artifacts(parent.id)[0]
    parent_entity = Entity(
        "hls.kernel", "dut", parent.id, stage="ast",
        anchors=[SourceAnchor(artifact.id, start_line=1)],
    )
    parent_graph = CanonicalGraph(parent.id)
    parent_graph.add_entity(parent_entity)
    bundle.store.save_graph(parent_graph)
    bundle.store.set_active_snapshot(bundle.manifest.project_id, parent.id)
    action = VariantAction(
        parent_snapshot_id=parent.id,
        kind="hls.directive.pipeline",
        scope_id=parent_entity.id,
        delta={"requested_ii": 1},
        proposer="consumer.fixture",
    )
    bundle.store.add_variant(action)
    child = bundle.snapshot(
        parent_snapshot_id=parent.id, action_id=action.id,
        extraction_hash="candidate.v02",
    )
    child_entity = Entity(
        "hls.kernel", "dut", child.id, stage="ast",
        anchors=[SourceAnchor(artifact.id, start_line=1)],
    )
    child_graph = CanonicalGraph(child.id)
    child_graph.add_entity(child_entity)
    bundle.store.save_graph(child_graph)
    return bundle, parent, parent_entity, child, child_entity, artifact, action


def test_typed_evidence_and_derivation_preserve_legacy_identity() -> None:
    legacy = Derivation(
        "snapshot_owner", "entity_subject", "feature.trip_count", 8,
        "hlsgraph.feature.constant", "1", ["observation_input"],
    )
    assert legacy.id == stable_id("derivation", {
        "snapshot": "snapshot_owner",
        "subject": "entity_subject",
        "predicate": "feature.trip_count",
        "algorithm": "hlsgraph.feature.constant",
        "version": "1",
        "inputs": ["observation_input"],
    })
    assert [(item.kind.value, item.target_id, item.snapshot_id)
            for item in legacy.evidence_refs] == [
        ("observation", "observation_input", "snapshot_owner"),
    ]
    assert Derivation.from_dict(json_ready(legacy)).id == legacy.id

    generic = Derivation(
        snapshot_id="snapshot_owner", subject_id="entity_subject",
        predicate="feature.bitwidth", value=32,
        algorithm="hlsgraph.feature.lookup", algorithm_version="1",
        evidence_refs=[EvidenceRef(
            EvidenceKind.ARTIFACT, "artifact_ir", "snapshot_owner",
        )],
    )
    assert generic.input_observation_ids == []
    assert generic.id != legacy.id
    with pytest.raises(ValueError, match="snapshot"):
        Derivation(
            snapshot_id="snapshot_owner", subject_id="entity_subject",
            predicate="feature.bad", value=1,
            algorithm="hlsgraph.feature.lookup", algorithm_version="1",
            evidence_refs=[EvidenceRef(
                EvidenceKind.ARTIFACT, "artifact_ir", "snapshot_other",
            )],
        )


def test_evidence_ref_combinations_and_dataset_opt_ins_fail_closed() -> None:
    anchor = SourceAnchor("artifact_a", start_line=1)
    with pytest.raises(ValueError, match="cannot carry"):
        EvidenceRef(EvidenceKind.OBSERVATION, "observation_a", anchor=anchor)
    with pytest.raises(ValueError, match="target artifact"):
        EvidenceRef(EvidenceKind.ARTIFACT, "artifact_b", anchor=anchor)
    valid = EvidenceRef(EvidenceKind.ENTITY_ANCHOR, "entity_a", anchor=anchor)
    relation_ref = EvidenceRef(EvidenceKind.RELATION, "relation_a", anchor=anchor)
    assert relation_ref.kind == EvidenceKind.RELATION
    with pytest.raises(ValueError, match="stable id"):
        EvidenceRef.from_dict({**json_ready(valid), "id": "evidence_ref_tampered"})

    manifest = DatasetManifest(
        "dataset.opt_in", "features.v02", ["snapshot_a"],
        feature_evidence_predicates=["feature.bitwidth"],
        entity_correspondence_kinds=["mapping.ast_to_ir"],
    )
    assert manifest.feature_evidence_predicates == ["feature.bitwidth"]
    assert manifest.entity_correspondence_kinds == ["mapping.ast_to_ir"]
    assert DatasetManifest("dataset.empty", "features.v02", ["snapshot_a"]
                           ).feature_evidence_predicates == []
    with pytest.raises(ValueError, match="unique"):
        DatasetManifest(
            "dataset.duplicate", "features.v02", ["snapshot_a"],
            feature_evidence_predicates=["feature.x", "feature.x"],
        )


def test_relation_evidence_resolves_in_derivation_store(tmp_path) -> None:
    (tmp_path / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    bundle = GraphBundle.create(
        tmp_path,
        minimal_manifest(
            "test.v02.relation_evidence", "relation evidence", "dut", "kernel.cpp",
        ),
    )
    snapshot = bundle.snapshot()
    artifact = bundle.store.artifacts(snapshot.id)[0]
    kernel = Entity(
        "hls.kernel", "dut", snapshot.id, stage="ast",
        anchors=[SourceAnchor(artifact.id, start_line=1)],
    )
    loop = Entity(
        "hls.loop", "loop", snapshot.id, stage="ast",
        anchors=[SourceAnchor(artifact.id, start_line=1)],
    )
    graph = CanonicalGraph(snapshot.id)
    graph.add_entity(kernel)
    graph.add_entity(loop)
    contains = graph.add_relation(Relation(
        kernel.id, loop.id, "hls.contains", snapshot.id, stage="ast",
    ))
    bundle.store.save_graph(graph)
    feature = Derivation(
        snapshot.id, loop.id, "feature.trip_count", None,
        "hlsgraph.static.trip_count", "1", stage="ast",
        completeness="missing",
        evidence_refs=[EvidenceRef(
            EvidenceKind.RELATION, contains.id, snapshot.id,
        )],
    )
    bundle.store.add_derivations([feature])
    assert bundle.store.derivations(snapshot.id)[0]["evidence_refs"][0][
        "target_id"
    ] == contains.id

    missing = Derivation(
        snapshot.id, loop.id, "feature.loop_bounds", None,
        "hlsgraph.static.loop_bounds", "1", stage="ast",
        completeness="missing",
        evidence_refs=[EvidenceRef(
            EvidenceKind.RELATION, "relation_missing", snapshot.id,
        )],
    )
    with pytest.raises(StoreError, match="does not exist"):
        bundle.store.add_derivations([missing])


def test_correspondence_generic_evidence_and_materialization_attempts(tmp_path) -> None:
    (bundle, parent, parent_entity, child, child_entity,
     artifact, action) = _two_snapshot_bundle(tmp_path)
    observation = Observation(
        parent.id, parent_entity.id, "feature.loop_count", 1, "ast",
        AuthorityClass.STATIC_FACT, artifact_id=artifact.id,
    )
    bundle.store.add_observations([observation])
    derivation = Derivation(
        parent.id, parent_entity.id, "feature.bitwidth", 32,
        "hlsgraph.feature.lookup", "1",
        evidence_refs=[EvidenceRef(
            EvidenceKind.OBSERVATION, observation.id, parent.id,
        )],
    )
    bundle.store.add_derivations([derivation])

    correspondence = EntityCorrespondence(
        source_snapshot_id=parent.id, source_entity_id=parent_entity.id,
        target_snapshot_id=child.id, target_entity_id=child_entity.id,
        kind="mapping.ast_to_ast", producer="hlsgraph.correspondence.identity",
        producer_version="1",
        evidence_refs=[
            EvidenceRef(EvidenceKind.DERIVATION, derivation.id, parent.id),
            EvidenceRef(
                EvidenceKind.ENTITY_ANCHOR, child_entity.id, child.id,
                child_entity.anchors[0],
            ),
        ],
    )
    bundle.store.add_correspondence(correspondence)
    assert bundle.store.correspondences(parent.id) == [correspondence]
    assert bundle.store.correspondences(
        source_snapshot_id=parent.id, target_snapshot_id=child.id,
        kind="mapping.ast_to_ast",
    ) == [correspondence]

    missing = EntityCorrespondence(
        source_snapshot_id=parent.id, source_entity_id=parent_entity.id,
        target_snapshot_id=child.id, target_entity_id=child_entity.id,
        kind="mapping.missing", producer="hlsgraph.correspondence.identity",
        producer_version="1",
        evidence_refs=[EvidenceRef(
            EvidenceKind.DERIVATION, "derivation_missing", parent.id,
        )],
    )
    with pytest.raises(StoreError, match="does not exist"):
        bundle.store.add_correspondence(missing)

    materialized = ActionMaterialization(
        action.id, parent.id, ActionMaterializationStatus.MATERIALIZED,
        result_snapshot_id=child.id, attempted_at="2026-01-01T00:00:00+00:00",
    )
    bundle.store.add_materialization(materialized)
    diagnostic = Diagnostic(
        child.id, "action.retry_failed", DiagnosticSeverity.ERROR,
        "candidate could not be indexed",
    )
    bundle.store.add_diagnostics([diagnostic])
    failed = ActionMaterialization(
        action.id, parent.id, ActionMaterializationStatus.FAILED,
        result_snapshot_id=child.id, diagnostic_ids=[diagnostic.id],
        attempted_at="2026-01-02T00:00:00+00:00",
    )
    bundle.store.add_materialization(failed)
    assert bundle.store.materializations(action.id) == [materialized, failed]
    assert bundle.store.materializations(action.id, status="failed") == [failed]

    with pytest.raises(ValueError, match="diagnostic"):
        ActionMaterialization(
            action.id, parent.id, ActionMaterializationStatus.NO_OP,
        )


def test_failed_index_and_no_op_materialization_commit_atomically(tmp_path) -> None:
    (bundle, parent, parent_entity, _child, _child_entity,
     _artifact, _action) = _two_snapshot_bundle(tmp_path)
    action = VariantAction(
        parent.id, "hls.directive.unroll", parent_entity.id, {"factor": 1},
        "consumer.fixture",
    )
    bundle.store.add_variant(action)
    candidate = bundle.snapshot(
        parent_snapshot_id=parent.id, action_id=action.id,
        extraction_hash="candidate.no_op",
    )
    run = ToolRun(
        candidate.id, "index", "extractor.local", "7" * 64,
        status=RunStatus.FAILED, failure_class=FailureClass.INPUT, exit_code=1,
    )
    diagnostic = Diagnostic(
        candidate.id, "action.semantic_no_op", DiagnosticSeverity.WARNING,
        "the requested action does not change the design", run_id=run.id,
    )
    run.diagnostics = [diagnostic.id]
    materialization = ActionMaterialization(
        action.id, parent.id, ActionMaterializationStatus.NO_OP,
        diagnostic_ids=[diagnostic.id],
        attempted_at="2026-01-03T00:00:00+00:00",
    )
    bundle.store.commit_index_failure(
        run=run, diagnostics=[diagnostic], materialization=materialization,
    )
    assert bundle.store.runs(candidate.id) == [run]
    assert bundle.store.diagnostics(candidate.id) == [diagnostic]
    assert bundle.store.materializations(action.id) == [materialization]
    assert bundle.store.has_graph(candidate.id) is False


def test_explicit_v01_to_v02_sqlite_migration_preserves_derivation_id(tmp_path) -> None:
    path = tmp_path / "legacy.db"
    legacy = Derivation(
        "snapshot_old", "entity_old", "feature.legacy", 4,
        "hlsgraph.feature.constant", "1", ["observation_old"],
    )
    payload = json_ready(legacy)
    payload.pop("evidence_refs")
    with sqlite3.connect(path) as connection:
        connection.executescript("""
        PRAGMA foreign_keys=ON;
        CREATE TABLE schema_info (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE snapshots (id TEXT PRIMARY KEY, project_id TEXT, payload_json TEXT);
        CREATE TABLE entities (
          id TEXT NOT NULL, snapshot_id TEXT NOT NULL, payload_json TEXT,
          PRIMARY KEY(snapshot_id,id)
        );
        CREATE TABLE artifacts (id TEXT PRIMARY KEY);
        CREATE TABLE snapshot_artifacts (
          snapshot_id TEXT NOT NULL, artifact_id TEXT NOT NULL,
          PRIMARY KEY(snapshot_id,artifact_id)
        );
        CREATE TABLE observations (
          id TEXT PRIMARY KEY, snapshot_id TEXT NOT NULL, payload_json TEXT
        );
        CREATE TABLE derivations (
          id TEXT PRIMARY KEY, snapshot_id TEXT NOT NULL, payload_json TEXT NOT NULL
        );
        CREATE TABLE diagnostics (
          id TEXT PRIMARY KEY, snapshot_id TEXT NOT NULL, payload_json TEXT
        );
        CREATE TABLE variants (
          id TEXT PRIMARY KEY, parent_snapshot_id TEXT NOT NULL, payload_json TEXT
        );
        CREATE TABLE graph_views (
          snapshot_id TEXT PRIMARY KEY, schema_version TEXT NOT NULL
        );
        """)
        connection.execute(
            "INSERT INTO schema_info(key,value) VALUES('schema_version','0.1.0')"
        )
        connection.execute(
            "INSERT INTO snapshots(id,project_id,payload_json) VALUES(?,?,?)",
            ("snapshot_old", "project_old", "{}"),
        )
        connection.execute(
            "INSERT INTO entities(id,snapshot_id,payload_json) VALUES(?,?,?)",
            ("entity_old", "snapshot_old", "{}"),
        )
        connection.execute(
            "INSERT INTO observations(id,snapshot_id,payload_json) VALUES(?,?,?)",
            ("observation_old", "snapshot_old", "{}"),
        )
        connection.execute(
            "INSERT INTO derivations(id,snapshot_id,payload_json) VALUES(?,?,?)",
            (legacy.id, "snapshot_old", json.dumps(payload)),
        )
        connection.execute(
            "INSERT INTO graph_views(snapshot_id,schema_version) VALUES(?,?)",
            ("snapshot_old", "0.1.0"),
        )

    store = LedgerStore(path)
    with pytest.raises(StoreError, match="explicit migration"):
        store.initialize()
    planned = [{
        "from_version": "0.1.0",
        "to_version": "0.2.0",
        "description": (
            "add typed feature evidence, entity correspondence, and action "
            "materialization contracts"
        ),
    }, {
        "from_version": "0.2.0",
        "to_version": "0.3.0",
        "description": (
            "add knowledge pack inventory, fail-closed bindings, coverage "
            "manifests, and rebuildable knowledge FTS"
        ),
    }]
    assert store.migration_plan() == planned
    assert store.migrate() == planned
    assert store.migration_plan() == []
    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT value FROM schema_info WHERE key='schema_version'"
        ).fetchone()[0] == "0.3.0"
        assert connection.execute(
            "SELECT schema_version FROM graph_views"
        ).fetchone()[0] == "0.2.0"
        migrated = json.loads(connection.execute(
            "SELECT payload_json FROM derivations WHERE id=?", (legacy.id,),
        ).fetchone()[0])
        assert migrated["id"] == legacy.id
        assert migrated["evidence_refs"][0]["kind"] == "observation"
        assert migrated["evidence_refs"][0]["target_id"] == "observation_old"
