from __future__ import annotations

import json

import pytest

from hlsgraph import (
    DatasetManifest,
    FEATURE_SCHEMA_VERSION,
    PredictionEnvelope,
    Project,
    VariantAction,
)
from hlsgraph.api import RestApplication
from hlsgraph.bundle import GraphBundle
from hlsgraph.export import export_dataset
from hlsgraph.manifest import minimal_manifest
from hlsgraph.mcp import ReadOnlyMcpService
from hlsgraph.model import json_ready, stable_hash, stable_id
from hlsgraph.store import StoreError


PRIVATE_SOURCE_SENTINEL = "PRIVATE_SOURCE_SENTINEL_ACTION_7d39f6"


def _project(tmp_path) -> Project:
    (tmp_path / "kernel.cpp").write_text(
        f"// {PRIVATE_SOURCE_SENTINEL}\nvoid dut() {{\n  int value = 0;\n}}\n",
        encoding="utf-8",
    )
    return Project(GraphBundle.create(
        tmp_path,
        minimal_manifest(
            "test.action_lineage", "action lineage", "dut", "kernel.cpp",
        ),
    ))


def _prediction(snapshot_id: str, subject_id: str, *,
                action_id: str | None = None, model_id: str = "test.model") -> PredictionEnvelope:
    return PredictionEnvelope(
        snapshot_id=snapshot_id,
        subject_id=subject_id,
        predicate="prediction.latency_cycles",
        value=12.5,
        model_id=model_id,
        model_version="1",
        input_schema_version=FEATURE_SCHEMA_VERSION,
        action_id=action_id,
    )


def test_prediction_action_is_optional_without_changing_legacy_identity() -> None:
    prediction = _prediction("snapshot_input", "entity_kernel")
    expected = stable_id("prediction", {
        "snapshot": "snapshot_input",
        "subject": "entity_kernel",
        "predicate": "prediction.latency_cycles",
        "model": "test.model",
        "version": "1",
        "input_schema_version": FEATURE_SCHEMA_VERSION,
        "trainset_hash": None,
        "value": 12.5,
        "unit": None,
        "uncertainty": None,
        "applicability": {},
        "ood": {},
        "metadata": {},
    })
    assert prediction.id == expected
    assert _prediction(
        "snapshot_input", "entity_kernel", action_id="action_candidate",
    ).id != expected
    with pytest.raises(ValueError, match="action_id"):
        _prediction("snapshot_input", "entity_kernel", action_id=" ")


def test_action_prediction_snapshot_lineage_is_closed_across_interfaces(tmp_path) -> None:
    project = _project(tmp_path)
    first = project.index(degraded=True)
    assert first.success
    first_graph = project.service(first.snapshot_id).graph()
    first_kernel = next(
        item for item in first_graph.entities.values() if item.kind == "hls.kernel"
    )

    action = VariantAction(
        parent_snapshot_id=first.snapshot_id,
        kind="hls.directive.pipeline",
        scope_id=first_kernel.id,
        delta={"requested_ii": 1, "private_note": PRIVATE_SOURCE_SENTINEL},
        proposer=f"test.fixture.{PRIVATE_SOURCE_SENTINEL}",
        rationale=f"local-only rationale {PRIVATE_SOURCE_SENTINEL}",
    )
    project.record_variant_action(action)
    prediction = _prediction(
        first.snapshot_id, first_kernel.id, action_id=action.id,
    )
    project.record_prediction(prediction)

    # A consumer materializes the action before asking HLSGraph to index it.
    source_path = tmp_path / "kernel.cpp"
    source_path.write_text(
        source_path.read_text(encoding="utf-8") + "\n// materialized action\n",
        encoding="utf-8",
    )
    second = project.index_variant(action.id, degraded=True)
    assert second.success
    assert second.parent_snapshot_id == first.snapshot_id
    assert second.action_id == action.id
    persisted = project.bundle.store.snapshot(second.snapshot_id)
    assert persisted.parent_snapshot_id == first.snapshot_id
    assert persisted.action_id == action.id

    pending = VariantAction(
        parent_snapshot_id=first.snapshot_id,
        kind="hls.directive.unroll",
        scope_id=first_kernel.id,
        delta={"factor": 2},
        proposer="test.fixture",
    )
    project.record_variant_action(pending)

    sdk = project.variants(action_id=action.id)
    materializations = project.materializations(action.id)
    assert sdk["parent_snapshot_id"] == first.snapshot_id
    assert sdk["record_class"] == "variant_action"
    assert sdk["lineage_semantics"] == (
        "explicit_materialization_records_with_legacy_snapshot_fallback"
    )
    assert sdk["items"] == [{
        **json_ready(action),
        "prediction_ids": [prediction.id],
        "materializations": materializations,
        "result_snapshot_ids": [second.snapshot_id],
        "result_snapshots": [{
            "snapshot_id": second.snapshot_id,
            "parent_snapshot_id": first.snapshot_id,
            "action_id": action.id,
            "created_at": persisted.created_at,
            "graph_available": True,
        }],
    }]
    assert PRIVATE_SOURCE_SENTINEL in json.dumps(sdk, ensure_ascii=False)
    pending_item = project.variants(action_id=pending.id)["items"][0]
    assert pending_item["prediction_ids"] == []
    assert pending_item["materializations"] == []
    assert pending_item["result_snapshot_ids"] == []
    assert pending_item["result_snapshots"] == []

    rest = RestApplication(project).dispatch(
        "GET", f"/api/v1/variants?action_id={action.id}",
    )
    assert rest.status == 200
    assert rest.body["items"] == sdk["items"]
    assert rest.body["parent_snapshot_id"] == sdk["parent_snapshot_id"]
    assert rest.body["lineage_semantics"] == sdk["lineage_semantics"]
    assert PRIVATE_SOURCE_SENTINEL in json.dumps(rest.body, ensure_ascii=False)

    mcp = ReadOnlyMcpService(project).variants(action_id=action.id)
    assert mcp["items"] == sdk["items"]
    assert mcp["parent_snapshot_id"] == sdk["parent_snapshot_id"]
    assert mcp["truncated"] is False
    assert PRIVATE_SOURCE_SENTINEL in json.dumps(mcp, ensure_ascii=False)

    dataset = DatasetManifest(
        dataset_id="test.action_lineage_export",
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        snapshot_ids=[first.snapshot_id, second.snapshot_id],
        labels=[],
        splits={first.snapshot_id: "unassigned", second.snapshot_id: "unassigned"},
    )
    output = tmp_path / "dataset"
    manifest = export_dataset(
        project.bundle, first.snapshot_id, output, dataset,
    )
    variants = [json.loads(line) for line in
                (output / "variants.jsonl").read_text(encoding="utf-8").splitlines()]
    lineage = [json.loads(line) for line in
               (output / "snapshot_lineage.jsonl").read_text(
                   encoding="utf-8",
               ).splitlines()]
    exported_action = next(item for item in variants if item["id"] == action.id)
    assert set(exported_action) == {
        "id", "parent_snapshot_id", "prediction_ids", "result_snapshot_ids",
        "details_exported", "kind", "scope_present", "scope_exported",
        "scope_id", "delta_sha256",
    }
    assert exported_action["parent_snapshot_id"] == first.snapshot_id
    assert exported_action["kind"] == action.kind
    assert exported_action["scope_id"] == first_kernel.id
    assert exported_action["scope_present"] is True
    assert exported_action["scope_exported"] is True
    assert exported_action["delta_sha256"] == stable_hash(action.delta)
    assert exported_action["details_exported"] is True
    assert exported_action["prediction_ids"] == [prediction.id]
    assert exported_action["result_snapshot_ids"] == [second.snapshot_id]
    assert not {"delta", "rationale", "proposer", "created_at"}.intersection(
        exported_action
    )
    assert next(item for item in variants if item["id"] == pending.id)[
        "result_snapshot_ids"
    ] == []
    assert {item["snapshot_id"]: item for item in lineage}[second.snapshot_id] == {
        "snapshot_id": second.snapshot_id,
        "parent_snapshot_id": first.snapshot_id,
        "action_id": action.id,
    }
    assert manifest["row_counts"]["variants"] == 2
    assert manifest["row_counts"]["snapshot_lineage"] == 2
    assert manifest["snapshots"][second.snapshot_id]["action_id"] == action.id
    feature_spec = json.loads(
        (output / "feature_spec.json").read_text(encoding="utf-8")
    )
    projection = feature_spec["variant_public_projection"]
    assert projection["mode"] == "minimal_positive_allowlist"
    assert projection["undeclared_parent_policy"] == "opaque_lineage_stub"
    assert projection["omitted_sensitive_fields"] == [
        "delta", "rationale", "proposer", "created_at",
    ]
    public_export = json.dumps(manifest, ensure_ascii=False) + "".join(
        path.read_text(encoding="utf-8")
        for path in sorted(output.iterdir()) if path.is_file()
    )
    assert PRIVATE_SOURCE_SENTINEL not in public_export

    result_only = tmp_path / "result-only-dataset"
    result_manifest = export_dataset(project.bundle, second.snapshot_id, result_only)
    result_variants = [json.loads(line) for line in
                       (result_only / "variants.jsonl").read_text(
                           encoding="utf-8",
                       ).splitlines()]
    assert result_variants == [{
        "id": action.id,
        "parent_snapshot_id": first.snapshot_id,
        "prediction_ids": [],
        "result_snapshot_ids": [second.snapshot_id],
        "details_exported": False,
    }]
    result_export = json.dumps(result_manifest, ensure_ascii=False) + "".join(
        path.read_text(encoding="utf-8")
        for path in sorted(result_only.iterdir()) if path.is_file()
    )
    assert PRIVATE_SOURCE_SENTINEL not in result_export


def test_action_consistency_checks_fail_before_creating_false_lineage(tmp_path) -> None:
    project = _project(tmp_path)
    first = project.index(degraded=True)
    kernel = next(item for item in project.service().graph().entities.values()
                  if item.kind == "hls.kernel")
    action = VariantAction(
        parent_snapshot_id=first.snapshot_id,
        kind="hls.directive.pipeline",
        scope_id=kernel.id,
        delta={"requested_ii": 1},
        proposer="test.fixture",
    )
    project.record_variant_action(action)
    source_path = tmp_path / "kernel.cpp"
    source_path.write_text(
        source_path.read_text(encoding="utf-8") + "\n// materialized action\n",
        encoding="utf-8",
    )
    second = project.index_variant(action.id, degraded=True)
    second_kernel = next(
        item for item in project.service(second.snapshot_id).graph().entities.values()
        if item.kind == "hls.kernel"
    )

    with pytest.raises(StoreError, match="input snapshot"):
        project.record_prediction(_prediction(
            second.snapshot_id, second_kernel.id, action_id=action.id,
        ))
    with pytest.raises(KeyError):
        project.record_prediction(_prediction(
            second.snapshot_id, second_kernel.id, action_id="action_missing",
            model_id="test.missing_action",
        ))

    # GraphBundle is a public low-level boundary: a known action cannot be
    # attached to a different parent even if callers bypass Project.index().
    with pytest.raises(StoreError, match="recorded action"):
        project.bundle.snapshot(
            parent_snapshot_id=second.snapshot_id,
            action_id=action.id,
        )

    # An unknown opaque action remains compatible for legacy imports.
    opaque = project.bundle.snapshot(
        parent_snapshot_id=second.snapshot_id,
        action_id="legacy.opaque_action",
    )
    assert opaque.parent_snapshot_id == second.snapshot_id
    assert opaque.action_id == "legacy.opaque_action"

    compatible_action = VariantAction(
        parent_snapshot_id=first.snapshot_id,
        kind="hls.directive.array_partition",
        scope_id=kernel.id,
        delta={"factor": 2},
        proposer="test.fixture",
    )
    compatible_snapshot = project.bundle.snapshot(
        parent_snapshot_id=first.snapshot_id,
        action_id=compatible_action.id,
    )
    project.record_variant_action(compatible_action)
    assert compatible_snapshot.parent_snapshot_id == first.snapshot_id
    assert project.bundle.store.variant(compatible_action.id) is not None

    # The inverse write order is also fail-closed: importing an action later
    # cannot contradict an already-retained opaque snapshot reference.
    late_action = VariantAction(
        parent_snapshot_id=first.snapshot_id,
        kind="hls.directive.unroll",
        scope_id=kernel.id,
        delta={"factor": 4},
        proposer="test.fixture",
    )
    late_snapshot = project.bundle.snapshot(
        parent_snapshot_id=second.snapshot_id,
        action_id=late_action.id,
    )
    assert late_snapshot.action_id == late_action.id
    with pytest.raises(StoreError, match="conflicts with existing snapshot lineage"):
        project.record_variant_action(late_action)
    assert project.bundle.store.variant(late_action.id) is None

    latest = project.bundle.store.latest_candidate(project.bundle.manifest.project_id)
    assert latest is not None
    with pytest.raises(ValueError, match="different parent_snapshot_id"):
        project.index(
            degraded=True,
            parent_snapshot_id=second.snapshot_id,
            action_id=action.id,
        )
    with pytest.raises(KeyError):
        project.index(degraded=True, action_id="action_missing")
    assert project.bundle.store.latest_candidate(
        project.bundle.manifest.project_id,
    ).id == latest.id


def test_legacy_prediction_payload_without_action_key_remains_idempotent(tmp_path) -> None:
    project = _project(tmp_path)
    indexed = project.index(degraded=True)
    kernel = next(item for item in project.service().graph().entities.values()
                  if item.kind == "hls.kernel")
    prediction = _prediction(
        indexed.snapshot_id, kernel.id, model_id="test.legacy_model",
    )
    legacy = json_ready(prediction)
    legacy.pop("action_id")
    payload = json.dumps(
        legacy, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )
    with project.bundle.store.write() as connection:
        connection.execute(
            "INSERT INTO predictions(id,snapshot_id,subject_id,predicate,payload_json) "
            "VALUES(?,?,?,?,?)",
            (prediction.id, prediction.snapshot_id, prediction.subject_id,
             prediction.predicate, payload),
        )

    project.record_prediction(prediction)
    values = project.bundle.store.predictions(indexed.snapshot_id)
    assert values == [legacy]
