from __future__ import annotations

import json
import sqlite3

import pytest

from hlsgraph import (
    AuthorityClass,
    DatasetManifest,
    Derivation,
    EntityCorrespondence,
    EvidenceKind,
    EvidenceRef,
    FEATURE_SCHEMA_VERSION,
    Observation,
    Project,
    ProjectManifest,
    SCHEMA_VERSION,
    VariantAction,
)
from hlsgraph.bundle import BundleError, GraphBundle
from hlsgraph.api import RestApplication, openapi_document
from hlsgraph.cli import main
from hlsgraph.export import export_dataset
from hlsgraph.manifest import minimal_manifest
from hlsgraph.mcp import ReadOnlyMcpService
from hlsgraph.model import stable_hash


def _project(tmp_path) -> tuple[Project, object]:
    source = tmp_path / "kernel.cpp"
    source.write_text("void dut() { int value = 0; }\n", encoding="utf-8")
    manifest = minimal_manifest(
        "test.v02.sdk", "v0.2 SDK contracts", "dut", "kernel.cpp",
    )
    return Project(GraphBundle.create(tmp_path, manifest)), source


def _kernel(project: Project, snapshot_id: str):
    return next(
        item for item in project.service(snapshot_id).graph().entities.values()
        if item.kind == "hls.kernel"
    )


def test_index_variant_records_noop_then_materialized_attempt(tmp_path) -> None:
    project, source = _project(tmp_path)
    parent = project.index(degraded=True)
    assert parent.success
    parent_kernel = _kernel(project, parent.snapshot_id)
    action = VariantAction(
        parent_snapshot_id=parent.snapshot_id,
        kind="source.edit",
        scope_id=parent_kernel.id,
        delta={"change": "constant"},
        proposer="consumer.fixture",
    )
    project.record_variant_action(action)

    no_op = project.index_variant(action.id, degraded=True)
    assert no_op.success is False
    assert no_op.materialization_status == "no_op"
    assert project.bundle.store.has_graph(no_op.snapshot_id) is False
    assert project.bundle.latest_snapshot().id == parent.snapshot_id
    no_op_rows = project.materializations(action.id)
    assert [item["status"] for item in no_op_rows] == ["no_op"]
    assert no_op_rows[0]["diagnostic_ids"]

    source.write_text("void dut() { int value = 1; }\n", encoding="utf-8")
    result = project.index_variant(action.id, degraded=True)
    assert result.success is True
    assert result.materialization_status == "materialized"
    assert result.parent_snapshot_id == parent.snapshot_id
    assert result.snapshot_id != parent.snapshot_id
    assert project.bundle.store.has_graph(result.snapshot_id)
    rows = project.materializations(action.id)
    assert [item["status"] for item in rows] == ["no_op", "materialized"]
    assert rows[-1]["result_snapshot_id"] == result.snapshot_id

    dataset = DatasetManifest(
        dataset_id="dataset.v02.materializations",
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        snapshot_ids=[parent.snapshot_id, result.snapshot_id],
    )
    output = tmp_path / "materialization-export"
    project.export_dataset(output, dataset, snapshot_id=result.snapshot_id)
    variants = [json.loads(line) for line in (
        output / "variants.jsonl"
    ).read_text(encoding="utf-8").splitlines()]
    assert variants[0]["result_snapshot_ids"] == [result.snapshot_id]
    attempts = [json.loads(line) for line in (
        output / "action_materializations.jsonl"
    ).read_text(encoding="utf-8").splitlines()]
    assert [item["status"] for item in attempts] == ["no_op", "materialized"]
    assert attempts[0]["result_snapshot_id"] is None
    assert attempts[0]["result_snapshot_declared"] is False


def test_feature_evidence_and_correspondence_are_opt_in_exports(tmp_path) -> None:
    project, source = _project(tmp_path)
    parent = project.index(degraded=True)
    parent_kernel = _kernel(project, parent.snapshot_id)
    action = VariantAction(
        parent_snapshot_id=parent.snapshot_id,
        kind="source.edit",
        scope_id=parent_kernel.id,
        delta={"change": "constant"},
        proposer="consumer.fixture",
    )
    project.record_variant_action(action)
    source.write_text("void dut() { int value = 2; }\n", encoding="utf-8")
    result = project.index_variant(action.id, degraded=True)
    assert result.success
    result_kernel = _kernel(project, result.snapshot_id)

    feature = Derivation(
        snapshot_id=result.snapshot_id,
        subject_id=result_kernel.id,
        predicate="feature.fixture_operation_histogram",
        value={"add": 1, "compare": 0},
        unit="operations",
        stage="ast",
        algorithm="feature.static_histogram",
        algorithm_version="1",
        evidence_refs=[EvidenceRef(
            kind=EvidenceKind.ENTITY_ANCHOR,
            target_id=result_kernel.id,
            snapshot_id=result.snapshot_id,
        )],
    )
    project.bundle.store.add_derivations([feature])
    correspondence = EntityCorrespondence(
        source_snapshot_id=parent.snapshot_id,
        source_entity_id=parent_kernel.id,
        target_snapshot_id=result.snapshot_id,
        target_entity_id=result_kernel.id,
        kind="mapping.semantic_identity",
        producer="mapping.anchor_match",
        producer_version="1",
        evidence_refs=[
            EvidenceRef(
                kind=EvidenceKind.ENTITY_ANCHOR,
                target_id=parent_kernel.id,
                snapshot_id=parent.snapshot_id,
            ),
            EvidenceRef(
                kind=EvidenceKind.ENTITY_ANCHOR,
                target_id=result_kernel.id,
                snapshot_id=result.snapshot_id,
            ),
        ],
    )
    project.record_entity_correspondence(correspondence)

    queried = project.feature_evidence(
        result_kernel.id, predicates=["feature.fixture_operation_histogram"],
    )
    assert queried["items"][0]["id"] == feature.id
    mapped = project.correspondences(
        result_kernel.id, other_snapshot_id=parent.snapshot_id,
        kinds=["mapping.semantic_identity"], direction="target",
    )
    assert mapped["items"][0]["id"] == correspondence.id

    dataset = DatasetManifest(
        dataset_id="dataset.v02.features",
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        snapshot_ids=[parent.snapshot_id, result.snapshot_id],
        feature_evidence_predicates=["feature.fixture_operation_histogram"],
        entity_correspondence_kinds=["mapping.semantic_identity"],
        splits={parent.snapshot_id: "unassigned", result.snapshot_id: "unassigned"},
    )
    output = tmp_path / "selected-export"
    manifest = export_dataset(project.bundle, result.snapshot_id, output, dataset)
    feature_rows = [json.loads(line) for line in
                    (output / "feature_evidence.jsonl").read_text(
                        encoding="utf-8",
                    ).splitlines()]
    correspondence_rows = [json.loads(line) for line in
                           (output / "entity_correspondence.jsonl").read_text(
                               encoding="utf-8",
                           ).splitlines()]
    assert feature_rows == [{
        "algorithm": "feature.static_histogram",
        "algorithm_version": "1",
        "authority": "derived_fact",
        "completeness": "complete",
        "derivation_id": feature.id,
        "evidence_ids": [result_kernel.id],
        "evidence_refs": [json.loads(json.dumps({
            "anchor": None,
            "id": feature.evidence_refs[0].id,
            "kind": "entity_anchor",
            "snapshot_id": result.snapshot_id,
            "target_id": result_kernel.id,
        }))],
        "mask": True,
        "predicate": "feature.fixture_operation_histogram",
        "selected_as_feature": True,
        "snapshot_id": result.snapshot_id,
        "stage": "ast",
        "subject_id": result_kernel.id,
        "unit": "operations",
        "value": {"add": 1, "compare": 0},
    }]
    assert correspondence_rows[0]["correspondence_id"] == correspondence.id
    assert correspondence_rows[0]["parent_snapshot_id"] == parent.snapshot_id
    assert correspondence_rows[0]["result_snapshot_id"] == result.snapshot_id
    assert correspondence_rows[0]["candidates"] == [result_kernel.id]
    assert manifest["row_counts"]["feature_evidence"] == 1
    assert manifest["row_counts"]["entity_correspondence"] == 1

    default_output = tmp_path / "default-export"
    export_dataset(project.bundle, result.snapshot_id, default_output)
    assert (default_output / "feature_evidence.jsonl").read_text(encoding="utf-8") == ""
    assert (default_output / "entity_correspondence.jsonl").read_text(
        encoding="utf-8",
    ) == ""

    outcome = Derivation(
        snapshot_id=result.snapshot_id,
        subject_id=result_kernel.id,
        predicate="feature.latency",
        value=10,
        stage="ast",
        algorithm="feature.static_value",
        algorithm_version="1",
        evidence_refs=[EvidenceRef(
            kind=EvidenceKind.ENTITY_ANCHOR,
            target_id=result_kernel.id,
            snapshot_id=result.snapshot_id,
        )],
    )
    project.bundle.store.add_derivations([outcome])
    with pytest.raises(ValueError, match="outcome-shaped"):
        project.feature_evidence(predicates=["feature.latency"])
    blocked = DatasetManifest(
        dataset_id="dataset.v02.blocked",
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        snapshot_ids=[result.snapshot_id],
        feature_evidence_predicates=["feature.latency"],
    )
    with pytest.raises(ValueError, match="outcome-shaped"):
        export_dataset(
            project.bundle, result.snapshot_id, tmp_path / "blocked-export", blocked,
        )

    nested_outcome = Derivation(
        snapshot_id=result.snapshot_id,
        subject_id=result_kernel.id,
        predicate="feature.summary",
        value={"latency": 10},
        stage="ast",
        algorithm="feature.static_value",
        algorithm_version="1",
        evidence_refs=[EvidenceRef(
            kind=EvidenceKind.ENTITY_ANCHOR,
            target_id=result_kernel.id,
            snapshot_id=result.snapshot_id,
        )],
    )
    project.bundle.store.add_derivations([nested_outcome])
    nested_query = project.feature_evidence(
        predicates=["feature.summary"],
    )
    assert nested_query["items"] == []
    assert nested_query["rejected_nonstatic_records"] == 1
    closure_root = Derivation(
        snapshot_id=result.snapshot_id,
        subject_id=result_kernel.id,
        predicate="feature.closure",
        value={"count": 1},
        stage="ast",
        algorithm="feature.static_copy",
        algorithm_version="1",
        evidence_refs=[EvidenceRef(
            kind=EvidenceKind.DERIVATION,
            target_id=nested_outcome.id,
            snapshot_id=result.snapshot_id,
        )],
    )
    project.bundle.store.add_derivations([closure_root])
    closure_query = project.feature_evidence(
        predicates=["feature.closure"],
    )
    assert closure_query["items"] == []
    assert closure_query["rejected_nonstatic_records"] == 1
    nested_blocked = DatasetManifest(
        dataset_id="dataset.v02.nested_blocked",
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        snapshot_ids=[result.snapshot_id],
        feature_evidence_predicates=["feature.summary"],
    )
    with pytest.raises(ValueError, match="outcome-shaped key"):
        export_dataset(
            project.bundle, result.snapshot_id,
            tmp_path / "nested-blocked-export", nested_blocked,
        )
    closure_blocked = DatasetManifest(
        dataset_id="dataset.v02.closure_blocked",
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        snapshot_ids=[result.snapshot_id],
        feature_evidence_predicates=["feature.closure"],
    )
    with pytest.raises(ValueError, match="outcome-shaped key"):
        export_dataset(
            project.bundle, result.snapshot_id,
            tmp_path / "closure-blocked-export", closure_blocked,
        )


def test_public_queries_keep_ambiguous_correspondence_unresolved(
    tmp_path, capsys,
) -> None:
    project, source = _project(tmp_path)
    parent = project.index(degraded=True)
    parent_kernel = _kernel(project, parent.snapshot_id)
    action = VariantAction(
        parent_snapshot_id=parent.snapshot_id,
        kind="source.edit",
        scope_id=parent_kernel.id,
        delta={"change": "add helper"},
        proposer="consumer.fixture",
    )
    project.record_variant_action(action)
    source.write_text(
        "int helper(int x) { return x + 1; }\n"
        "void dut() { int value = helper(1); }\n",
        encoding="utf-8",
    )
    result = project.index_variant(action.id, degraded=True)
    assert result.success
    result_graph = project.service(result.snapshot_id).graph()
    result_kernel = _kernel(project, result.snapshot_id)
    result_helper = next(
        item for item in result_graph.entities.values()
        if item.kind == "hls.function" and item.name == "helper"
    )
    private_marker = str(tmp_path / "private-feature-path")
    feature = Derivation(
        snapshot_id=result.snapshot_id,
        subject_id=result_kernel.id,
        predicate="feature.fixture_operation_histogram",
        value={"add": 1},
        unit="operations",
        stage="ast",
        algorithm="feature.static_histogram",
        algorithm_version="1",
        evidence_refs=[EvidenceRef(
            kind=EvidenceKind.ENTITY_ANCHOR,
            target_id=result_kernel.id,
            snapshot_id=result.snapshot_id,
        )],
        metadata={"path": private_marker},
    )
    project.bundle.store.add_derivations([feature])

    mappings = []
    for target in (result_kernel, result_helper):
        mapping = EntityCorrespondence(
            source_snapshot_id=parent.snapshot_id,
            source_entity_id=parent_kernel.id,
            target_snapshot_id=result.snapshot_id,
            target_entity_id=target.id,
            kind="mapping.candidate",
            producer="mapping.explicit_fixture",
            producer_version="1",
            evidence_refs=[
                EvidenceRef(
                    kind=EvidenceKind.ENTITY_ANCHOR,
                    target_id=parent_kernel.id,
                    snapshot_id=parent.snapshot_id,
                ),
                EvidenceRef(
                    kind=EvidenceKind.ENTITY_ANCHOR,
                    target_id=target.id,
                    snapshot_id=result.snapshot_id,
                ),
            ],
            metadata={"path": private_marker},
        )
        project.record_entity_correspondence(mapping)
        mappings.append(mapping)

    expected = project.service(parent.snapshot_id).correspondences(
        parent_kernel.id,
        other_snapshot_id=result.snapshot_id,
        kinds=["mapping.candidate"],
        direction="source",
    )
    assert {item["id"] for item in expected["items"]} == {
        item.id for item in mappings
    }
    assert expected["candidate_groups"] == [{
        "role": "source",
        "from_snapshot_id": parent.snapshot_id,
        "from_entity_id": parent_kernel.id,
        "to_snapshot_id": result.snapshot_id,
        "kind": "mapping.candidate",
        "candidate_entity_ids": sorted([result_helper.id, result_kernel.id]),
        "candidate_count": 2,
        "resolution_status": "ambiguous",
        "resolved_entity_id": None,
    }]
    assert private_marker not in json.dumps(expected)
    assert all("metadata" not in item for item in expected["items"])

    rest = RestApplication(project, snapshot_id=parent.snapshot_id).dispatch(
        "GET",
        "/api/v1/correspondences?"
        f"entity_id={parent_kernel.id}&other_snapshot_id={result.snapshot_id}"
        "&kinds=mapping.candidate&direction=source",
    )
    assert rest.status == 200
    assert rest.body == expected
    mcp = ReadOnlyMcpService(project, snapshot_id=parent.snapshot_id)
    assert mcp.correspondences(
        parent_kernel.id, result.snapshot_id, ["mapping.candidate"], "source",
    ) == expected

    feature_expected = project.service(result.snapshot_id).feature_evidence(
        result_kernel.id, predicates=["feature.fixture_operation_histogram"],
    )
    assert private_marker not in json.dumps(feature_expected)
    assert "metadata" not in feature_expected["items"][0]
    feature_rest = RestApplication(project, snapshot_id=result.snapshot_id).dispatch(
        "GET",
            "/api/v1/feature-evidence?"
            f"entity_id={result_kernel.id}&predicates=feature.fixture_operation_histogram",
    )
    assert feature_rest.status == 200 and feature_rest.body == feature_expected
    assert ReadOnlyMcpService(
        project, snapshot_id=result.snapshot_id,
        ).feature_evidence(
            result_kernel.id, ["feature.fixture_operation_histogram"],
    ) == feature_expected

    code = main([
        "query", "--project", str(tmp_path),
        "--record-class", "correspondence",
        "--snapshot-id", parent.snapshot_id,
        "--entity-id", parent_kernel.id,
        "--other-snapshot-id", result.snapshot_id,
        "--correspondence-kind", "mapping.candidate",
        "--direction", "source",
    ])
    cli_payload = json.loads(capsys.readouterr().out)
    assert code == 0
    cli_payload.pop("command")
    assert cli_payload == expected

    dataset = DatasetManifest(
        dataset_id="dataset.v02.ambiguous",
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        snapshot_ids=[parent.snapshot_id, result.snapshot_id],
        entity_correspondence_kinds=["mapping.candidate"],
    )
    output = tmp_path / "ambiguous-export"
    exported = project.export_dataset(
        output, dataset, snapshot_id=result.snapshot_id,
    )
    rows = [json.loads(line) for line in (
        output / "entity_correspondence.jsonl"
    ).read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert all(item["resolution_status"] == "ambiguous" for item in rows)
    assert all(item["resolved_target_entity_id"] is None for item in rows)
    assert all(item["candidate_count"] == 2 for item in rows)
    materializations = [json.loads(line) for line in (
        output / "action_materializations.jsonl"
    ).read_text(encoding="utf-8").splitlines()]
    stored_materialization = project.materializations(action.id)[0]
    assert materializations == [{
        "action_id": action.id,
        "attempt_index": 1,
        "diagnostic_count": len(stored_materialization["diagnostic_ids"]),
        "evidence_count": 0,
        "materialization_id": stored_materialization["id"],
        "parent_snapshot_id": parent.snapshot_id,
        "result_snapshot_declared": True,
        "result_snapshot_id": result.snapshot_id,
        "status": "materialized",
    }]
    assert exported["row_counts"]["action_materializations"] == 1

    cli_output = tmp_path / "ambiguous-cli-export"
    code = main([
        "export", "--project", str(tmp_path), str(cli_output),
        "--kind", "dataset",
        "--snapshot-id", parent.snapshot_id,
        "--snapshot-id", result.snapshot_id,
        "--feature-evidence-predicate", "feature.fixture_operation_histogram",
        "--entity-correspondence-kind", "mapping.candidate",
    ])
    cli_export = json.loads(capsys.readouterr().out)
    assert code == 0
    assert cli_export["row_counts"]["feature_evidence"] == 1
    assert cli_export["row_counts"]["entity_correspondence"] == 2
    assert "/api/v1/feature-evidence" in openapi_document()["paths"]
    assert "/api/v1/correspondences" in openapi_document()["paths"]


def test_explicit_bundle_and_ledger_migration_from_v01(tmp_path) -> None:
    project, _source = _project(tmp_path)
    indexed = project.index(degraded=True)
    kernel = _kernel(project, indexed.snapshot_id)
    observation = Observation(
        snapshot_id=indexed.snapshot_id,
        subject_id=kernel.id,
        predicate="feature.seed",
        value=1,
        stage="ast",
        authority=AuthorityClass.STATIC_FACT,
    )
    project.bundle.store.add_observations([observation])
    derivation = Derivation(
        snapshot_id=indexed.snapshot_id,
        subject_id=kernel.id,
        predicate="feature.copy",
        value=1,
        stage="ast",
        algorithm="feature.copy",
        algorithm_version="1",
        input_observation_ids=[observation.id],
    )
    project.bundle.store.add_derivations([derivation])

    automatic_feature_ids = [
        item["id"] for item in project.bundle.store.derivations(indexed.snapshot_id)
        if str(item.get("algorithm", "")).startswith("hlsgraph.static.")
    ]
    database = project.bundle.store.path
    with sqlite3.connect(database) as connection:
        # Static feature derivations are a v0.2 facility.  Remove them before
        # constructing the deliberately downgraded v0.1 fixture.
        connection.executemany(
            "DELETE FROM derivations WHERE id=?",
            [(item,) for item in automatic_feature_ids],
        )
        payload = json.loads(connection.execute(
            "SELECT payload_json FROM derivations WHERE id=?", (derivation.id,),
        ).fetchone()[0])
        payload.pop("evidence_refs")
        connection.execute(
            "UPDATE derivations SET payload_json=? WHERE id=?",
            (json.dumps(payload, sort_keys=True, separators=(",", ":")), derivation.id),
        )
        connection.execute("DROP TABLE entity_correspondences")
        connection.execute("DROP TABLE action_materializations")
        connection.execute(
            "UPDATE graph_views SET schema_version='0.1.0'"
        )
        connection.execute(
            "UPDATE schema_info SET value='0.1.0' WHERE key='schema_version'"
        )
        row = connection.execute(
            "SELECT manifest_json FROM projects WHERE project_id=?",
            (project.bundle.manifest.project_id,),
        ).fetchone()
        manifest_value = json.loads(row[0])
        manifest_value["schema_version"] = "0.1.0"
        legacy_manifest = ProjectManifest.from_dict(manifest_value)
        connection.execute(
            "UPDATE projects SET manifest_hash=?,manifest_json=? WHERE project_id=?",
            (stable_hash(legacy_manifest.identity_payload()),
             json.dumps(manifest_value, sort_keys=True, separators=(",", ":")),
             project.bundle.manifest.project_id),
        )
        historical = connection.execute(
            "SELECT manifest_json FROM snapshot_manifests WHERE snapshot_id=?",
            (indexed.snapshot_id,),
        ).fetchone()
        historical_value = json.loads(historical[0])
        historical_value["schema_version"] = "0.1.0"
        connection.execute(
            "UPDATE snapshot_manifests SET manifest_json=? WHERE snapshot_id=?",
            (json.dumps(historical_value, sort_keys=True, separators=(",", ":")),
             indexed.snapshot_id),
        )

    for name, key in (("manifest.json", "schema_version"),
                      ("bundle.json", "schema_version")):
        path = project.bundle.root / name
        value = json.loads(path.read_text(encoding="utf-8"))
        value[key] = "0.1.0"
        if name == "bundle.json":
            value["bundle_version"] = "0.1.0"
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    manifest_source = tmp_path / "hlsgraph.toml"
    # GraphBundle.create was called from an in-memory manifest, so there is no
    # external source manifest to mutate in this fixture.
    assert not manifest_source.exists()

    with pytest.raises(BundleError, match="version|schema"):
        Project.open(tmp_path)
    plan = GraphBundle.migration_plan(tmp_path)
    assert {item["scope"] for item in plan} == {"bundle", "ledger"}
    applied = GraphBundle.migrate(tmp_path)
    assert applied == plan

    reopened = Project.open(tmp_path)
    assert reopened.bundle.store.load_graph(indexed.snapshot_id).snapshot_id == indexed.snapshot_id
    migrated = next(
        item for item in reopened.bundle.store.derivations(indexed.snapshot_id)
        if item["id"] == derivation.id
    )
    assert migrated["id"] == derivation.id
    assert migrated["input_observation_ids"] == [observation.id]
    assert migrated["evidence_refs"][0]["target_id"] == observation.id
    assert reopened.bundle.store.snapshot_manifest(indexed.snapshot_id).schema_version == "0.1.0"
    assert json.loads((reopened.bundle.root / "bundle.json").read_text(
        encoding="utf-8",
    ))["bundle_version"] == SCHEMA_VERSION
