from __future__ import annotations

import json
from pathlib import Path

import pytest

from hlsgraph.bundle import BundleError, GraphBundle
from hlsgraph.export import export_dataset, export_graph_json
from hlsgraph.extract import ExtractionError, ExtractionResult, LibClangExtractor
from hlsgraph.graph import CanonicalGraph
from hlsgraph.manifest import minimal_manifest
from hlsgraph.model import (
    ArtifactRef,
    AuthorityClass,
    Completeness,
    Derivation,
    Entity,
    GateKind,
    GateResult,
    GateStatus,
    Observation,
    Relation,
    RunStatus,
    SourceAnchor,
    ToolRun,
    ToolchainContext,
    VerificationKind,
    VerificationResult,
    json_ready,
)
from hlsgraph.query import CoreService
from hlsgraph.sdk import Project
from hlsgraph.store import StoreError


def _write_manifest(root: Path, *, project_id: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    manifest = minimal_manifest(
        project_id, "redline fixture", "dut", "kernel.cpp",
        part="part-a", clock_ns=5.0,
    )
    manifest.toolchains = [ToolchainContext(
        id="vendor.tool.a", vendor="vendor", name="hls", version="1.0", build="a",
    )]
    path = root / "hlsgraph.json"
    path.write_text(json.dumps(json_ready(manifest), indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    return path


def _rewrite(path: Path, mutator) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    mutator(value)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _standard_success(_self, context):
    graph = CanonicalGraph(context.snapshot.id)
    graph.add_entity(Entity(
        "hls.kernel", context.manifest.build.top, context.snapshot.id,
        qualified_name=context.manifest.build.top, stage="ast",
    ))
    return ExtractionResult(
        graph=graph, capabilities=["source.ast"],
        coverage={"fidelity": "standard_test_double"},
    )


def test_historical_snapshot_export_uses_its_own_target_and_toolchain(tmp_path):
    manifest_path = _write_manifest(tmp_path, project_id="test.historical_export")
    project = Project.create_from_manifest(manifest_path)
    first = project.index(degraded=True)
    assert first.success

    def change_context(value):
        value["target"]["part"] = "part-b"
        value["target"]["clocks"][0]["period_ns"] = 3.0
        value["toolchains"][0].update({"id": "vendor.tool.b", "version": "2.0", "build": "b"})

    _rewrite(manifest_path, change_context)
    reopened = Project.open(tmp_path)
    second = reopened.index(degraded=True)
    assert second.success and second.snapshot_id != first.snapshot_id

    old_manifest = export_dataset(
        reopened.bundle, first.snapshot_id, tmp_path / "export-old"
    )
    new_manifest = export_dataset(
        reopened.bundle, second.snapshot_id, tmp_path / "export-new"
    )
    assert old_manifest["target_profile"]["part"] == "part-a"
    assert old_manifest["target_profile"]["clocks"][0]["period_ns"] == 5.0
    assert old_manifest["toolchains"][0]["id"] == "vendor.tool.a"
    assert old_manifest["toolchains"][0]["version"] == "1.0"
    assert new_manifest["target_profile"]["part"] == "part-b"
    assert new_manifest["target_profile"]["clocks"][0]["period_ns"] == 3.0
    assert new_manifest["toolchains"][0]["id"] == "vendor.tool.b"
    assert new_manifest["toolchains"][0]["version"] == "2.0"


def test_schedule_unknown_fields_and_private_payload_never_enter_hot_graph_db_or_exports(tmp_path):
    secret = "PRIVATE_SCHEDULE_PAYLOAD_SENTINEL_62bd"
    (tmp_path / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    reports = tmp_path / "reports"
    reports.mkdir()
    schedule = reports / "schedule.json"
    schedule.write_text(json.dumps({
        "schema_version": "hlsgraph.vitis.schedule.v1",
        "top": "dut",
        "operations": [{
            "name": "compute.add",
            "start_cycle": 0,
            "end_cycle": 1,
            "pipeline_stage": 0,
            "latency": 1,
            "binding": "adder",
            "source_text": f"int private_value = 0; // {secret}",
            "private_payload": {"token": secret},
            f"unknown_{secret}": secret,
        }],
    }, indent=2) + "\n", encoding="utf-8")
    manifest = minimal_manifest(
        "test.schedule_sanitize", "schedule sanitize", "dut", "kernel.cpp"
    )
    manifest.artifact_paths.append({
        "path": "reports/schedule.json", "kind": "amd.vitis.schedule_json",
        "role": "schedule_report", "access": "private",
    })
    project = Project(GraphBundle.create(tmp_path, manifest))
    indexed = project.index(degraded=True)
    assert indexed.success

    graph = project.service().graph()
    graph_payload = json.dumps(graph.to_dict(), ensure_ascii=False)
    assert secret not in graph_payload
    scheduled = graph.by_kind("hls.scheduled_operation")
    assert len(scheduled) == 1
    assert not ({"source_text", "private_payload", "unknown_vendor_dump"}
                & set(scheduled[0].attrs))
    assert secret.encode() not in project.bundle.store.path.read_bytes()

    graph_export = export_graph_json(
        project.bundle, indexed.snapshot_id, tmp_path / "graph-export.json"
    )
    export_dataset(project.bundle, indexed.snapshot_id, tmp_path / "dataset-export")
    exported = graph_export.read_text(encoding="utf-8") + "\n" + "\n".join(
        path.read_text(encoding="utf-8")
        for path in (tmp_path / "dataset-export").iterdir() if path.is_file()
    )
    assert secret not in exported


def _two_snapshot_ledger(tmp_path):
    (tmp_path / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    manifest = minimal_manifest(
        "test.evidence_integrity", "evidence", "dut", "kernel.cpp",
    )
    manifest.stage_commands = {"csim": ["vitis_hls", "--csim"]}
    manifest.toolchains = [ToolchainContext(
        id="amd.vitis.2024_2", vendor="amd", name="vitis_hls", version="2024.2",
        environment_hash="e" * 64,
    )]
    bundle = GraphBundle.create(tmp_path, manifest)
    first = bundle.snapshot(action_id="first")
    first_artifact = bundle.store.artifacts(first.id)[0]
    first_entity = Entity("hls.kernel", "dut", first.id, qualified_name="dut")
    first_graph = CanonicalGraph(first.id)
    first_graph.add_entity(first_entity)
    bundle.store.save_graph(first_graph)

    (tmp_path / "kernel.cpp").write_text("void dut() { int changed = 1; }\n", encoding="utf-8")
    second = bundle.snapshot(action_id="second")
    second_artifact = bundle.store.artifacts(second.id)[0]
    second_entity = Entity("hls.kernel", "dut", second.id, qualified_name="dut")
    second_graph = CanonicalGraph(second.id)
    second_graph.add_entity(second_entity)
    bundle.store.save_graph(second_graph)
    return (bundle, first, first_entity, first_artifact,
            second, second_entity, second_artifact)


def test_observation_references_must_exist_in_the_same_snapshot(tmp_path):
    bundle, first, first_entity, first_artifact, second, second_entity, second_artifact = (
        _two_snapshot_ledger(tmp_path)
    )
    second_run = ToolRun(
        snapshot_id=second.id, stage="csim", backend="runner.fake",
        request_hash="2" * 64,
    )
    bundle.store.add_run(second_run)

    valid = Observation(
        snapshot_id=first.id, subject_id=first_entity.id,
        predicate="qor.latency_cycles", value=4, stage="schedule",
        authority=AuthorityClass.TOOL_OBSERVATION, artifact_id=first_artifact.id,
    )
    bundle.store.add_observations([valid])
    assert bundle.store.observations(first.id) == [valid]

    invalid = [
        Observation(
            snapshot_id=first.id, subject_id="entity.missing",
            predicate="test.value", value=1, stage="source",
            authority=AuthorityClass.STATIC_FACT,
        ),
        Observation(
            snapshot_id=first.id, subject_id=second_entity.id,
            predicate="test.value", value=1, stage="source",
            authority=AuthorityClass.STATIC_FACT,
        ),
        Observation(
            snapshot_id=first.id, subject_id=first_entity.id,
            predicate="test.value", value=1, stage="source",
            authority=AuthorityClass.STATIC_FACT, artifact_id=second_artifact.id,
        ),
        Observation(
            snapshot_id=first.id, subject_id=first_entity.id,
            predicate="test.value", value=1, stage="source",
            authority=AuthorityClass.STATIC_FACT, run_id=second_run.id,
        ),
    ]
    for item in invalid:
        with pytest.raises(StoreError, match="snapshot|subject|artifact|run"):
            bundle.store.add_observations([item])
    assert bundle.store.observations(first.id) == [valid]


def test_derivation_and_verification_evidence_must_exist_in_the_same_snapshot(tmp_path):
    bundle, first, first_entity, first_artifact, second, second_entity, _second_artifact = (
        _two_snapshot_ledger(tmp_path)
    )
    first_observation = Observation(
        snapshot_id=first.id, subject_id=first_entity.id,
        predicate="timing.wns_ns", value=0.1, unit="ns", stage="post_route",
        authority=AuthorityClass.TOOL_OBSERVATION, artifact_id=first_artifact.id,
    )
    bundle.store.add_observations([first_observation])
    first_run = ToolRun(
        snapshot_id=first.id, stage="csim", backend="runner.fake",
        request_hash="1" * 64,
    )
    second_run = ToolRun(
        snapshot_id=second.id, stage="csim", backend="runner.fake",
        request_hash="2" * 64,
    )
    bundle.store.add_run(first_run)
    bundle.store.add_run(second_run)

    valid_derivation = Derivation(
        snapshot_id=first.id, subject_id=first_entity.id,
        predicate="gate.post_route_timing", value=True,
        algorithm="test.wns", algorithm_version="1",
        input_observation_ids=[first_observation.id],
    )
    bundle.store.add_derivations([valid_derivation])
    assert bundle.store.derivations(first.id)[0]["id"] == valid_derivation.id

    invalid_derivations = [
        Derivation(
            snapshot_id=first.id, subject_id="entity.missing",
            predicate="test.derived", value=1, algorithm="test", algorithm_version="1",
            input_observation_ids=[first_observation.id],
        ),
        Derivation(
            snapshot_id=second.id, subject_id=second_entity.id,
            predicate="test.derived", value=1, algorithm="test", algorithm_version="1",
            input_observation_ids=[first_observation.id],
        ),
        Derivation(
            snapshot_id=first.id, subject_id=first_entity.id,
            predicate="test.derived", value=1, algorithm="test", algorithm_version="1",
            input_observation_ids=["observation.missing"],
        ),
    ]
    for item in invalid_derivations:
        with pytest.raises(StoreError, match="snapshot|subject|observation|evidence"):
            bundle.store.add_derivations([item])

    historical_manifest = bundle.store.snapshot_manifest(first.id)
    historical_toolchain = historical_manifest.toolchain_for_stage("csim")
    verified_run = ToolRun(
        snapshot_id=first.id, stage="csim", backend="runner.local",
        request_hash="3" * 64, toolchain_id=historical_toolchain.id,
        status=RunStatus.SUCCEEDED, exit_code=0,
        command=list(historical_manifest.stage_commands["csim"]),
        working_directory=".",
        environment_hash=historical_toolchain.environment_hash,
        input_artifact_ids=[
            item.id for item in bundle.store.artifacts(first.id)
            if item.producer_run_id is None
        ],
        metadata={
            "authority": "tool_observation", "tool_truth": True,
            "fresh_execution": True, "fresh_tool_truth": True,
            "campaign_id": "campaign.valid", "workload_id": "tb.valid",
        },
    )
    source = tmp_path / "valid-csim.json"
    source.write_text('{"status":"pass"}\n', encoding="utf-8")
    report, _path, _created = bundle.prepare_managed_artifact(
        source, kind="amd.vitis.csim_result", role="tool_output",
        producer_run_id=verified_run.id, metadata={"workload_id": "tb.valid"},
    )
    verified_observations = [
        Observation(
            snapshot_id=first.id, subject_id=first_entity.id,
            predicate=predicate, value=0, unit="count", stage="csim",
            authority=AuthorityClass.VERIFICATION_EVIDENCE,
            run_id=verified_run.id, artifact_id=report.id, workload_id="tb.valid",
        )
        for predicate in (
            "csim.exit_code", "csim.mismatches", "csim.assertions_failed",
        )
    ]
    valid_verification = VerificationResult(
        snapshot_id=first.id, kind=VerificationKind.CSIM, status=GateStatus.PASS,
        run_id=verified_run.id, workload_id="tb.valid",
        evidence_ids=[item.id for item in verified_observations],
    )
    verified_run.output_artifact_ids = [report.id]
    bundle.store.commit_run_result(
        run=verified_run, artifacts=[report], observations=verified_observations,
        verifications=[valid_verification],
    )
    assert valid_verification.id in {
        item["id"] for item in bundle.store.verifications(first.id)
    }

    invalid_verifications = [
        VerificationResult(
            snapshot_id=second.id, kind=VerificationKind.CSIM, status=GateStatus.PASS,
            run_id=second_run.id, evidence_ids=[verified_observations[0].id],
        ),
        VerificationResult(
            snapshot_id=first.id, kind=VerificationKind.CSIM, status=GateStatus.PASS,
            run_id=second_run.id, evidence_ids=[verified_observations[0].id],
        ),
        VerificationResult(
            snapshot_id=first.id, kind=VerificationKind.CSIM, status=GateStatus.PASS,
            run_id=verified_run.id, evidence_ids=["observation.missing"],
        ),
    ]
    for item in invalid_verifications:
        with pytest.raises(StoreError, match="snapshot|run|evidence|observation"):
            bundle.store.add_verifications([item])


def _comparison_bundle(tmp_path, *, entity_change: str | None = None,
                       relation_change: str | None = None):
    (tmp_path / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    bundle = GraphBundle.create(
        tmp_path, minimal_manifest("test.semantic_compare", "compare", "dut", "kernel.cpp")
    )
    left = bundle.snapshot(action_id="left")
    right = bundle.snapshot(action_id="right")
    artifact = bundle.store.artifacts(left.id)[0]

    def entity(snapshot_id, name, changed=False):
        values = {
            "authority": AuthorityClass.STATIC_FACT,
            "stage": "ast",
            "completeness": Completeness.COMPLETE,
            "anchors": [SourceAnchor(artifact.id, start_line=1, end_line=1)],
        }
        if changed and entity_change == "authority":
            values["authority"] = AuthorityClass.COMPILER_DECISION
        if changed and entity_change == "stage":
            values["stage"] = "mlir"
        if changed and entity_change == "completeness":
            values["completeness"] = Completeness.PARTIAL
        if changed and entity_change == "anchors":
            values["anchors"] = [SourceAnchor(artifact.id, start_line=2, end_line=2)]
        return Entity(
            "hls.process", name, snapshot_id, qualified_name=f"dut::{name}",
            attrs={"operation": name}, **values,
        )

    left_source, left_sink = entity(left.id, "source"), entity(left.id, "sink")
    right_source = entity(right.id, "source", changed=True)
    right_sink = entity(right.id, "sink")
    left_graph, right_graph = CanonicalGraph(left.id), CanonicalGraph(right.id)
    for graph, nodes in ((left_graph, (left_source, left_sink)),
                         (right_graph, (right_source, right_sink))):
        for node in nodes:
            graph.add_entity(node)

    def relation(snapshot_id, source, sink, changed=False):
        values = {
            "authority": AuthorityClass.COMPILER_DECISION,
            "stage": "mlir",
            "completeness": Completeness.COMPLETE,
            "anchors": [SourceAnchor(artifact.id, start_line=1, end_line=1)],
            "mapping_kind": "handshake.explicit",
        }
        if changed and relation_change == "authority":
            values["authority"] = AuthorityClass.DERIVED_FACT
        if changed and relation_change == "stage":
            values["stage"] = "schedule"
        if changed and relation_change == "completeness":
            values["completeness"] = Completeness.PARTIAL
        if changed and relation_change == "anchors":
            values["anchors"] = [SourceAnchor(artifact.id, start_line=2, end_line=2)]
        if changed and relation_change == "mapping_kind":
            values["mapping_kind"] = "schedule.explicit"
        return Relation(
            source.id, sink.id, "hls.streams_to", snapshot_id,
            attrs={"fifo_depth": 8}, **values,
        )

    left_graph.add_relation(relation(left.id, left_source, left_sink))
    right_graph.add_relation(relation(right.id, right_source, right_sink, changed=True))
    bundle.store.save_graph(left_graph)
    bundle.store.save_graph(right_graph)
    return bundle, left.id, right.id


@pytest.mark.parametrize("field", ["authority", "stage", "completeness", "anchors"])
def test_compare_detects_non_attribute_entity_semantic_changes(tmp_path, field):
    bundle, left, right = _comparison_bundle(tmp_path, entity_change=field)
    result = CoreService(bundle, left).compare(right)
    assert ["hls.process", "dut::source"] in result["entities_changed"]


@pytest.mark.parametrize(
    "field", ["authority", "stage", "completeness", "anchors", "mapping_kind"]
)
def test_compare_detects_relation_provenance_and_mapping_changes(tmp_path, field):
    bundle, left, right = _comparison_bundle(tmp_path, relation_change=field)
    result = CoreService(bundle, left).compare(right)
    assert result["relations_changed"], field


def test_csim_pass_alone_never_marks_design_overall_verified(tmp_path):
    (tmp_path / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    bundle = GraphBundle.create(
        tmp_path, minimal_manifest("test.csim_only", "csim only", "dut", "kernel.cpp")
    )
    snapshot = bundle.snapshot()
    kernel = Entity("hls.kernel", "dut", snapshot.id, qualified_name="dut")
    graph = CanonicalGraph(snapshot.id)
    graph.add_entity(kernel)
    bundle.store.save_graph(graph)
    observation = Observation(
        snapshot_id=snapshot.id, subject_id=kernel.id,
        predicate="csim.status", value="pass", stage="csim",
        authority=AuthorityClass.VERIFICATION_EVIDENCE,
    )
    bundle.store.add_observations([observation])
    verification = VerificationResult(
        snapshot_id=snapshot.id, kind=VerificationKind.CSIM,
        status=GateStatus.PASS, evidence_ids=[observation.id],
    )
    bundle.store.add_verifications([verification])
    bundle.store.add_derivations([
        Derivation(
            snapshot.id, kernel.id, "gate.resource_fits", True,
            "fixture.resource", "1", [observation.id], stage="post_route",
            authority=AuthorityClass.SYNTHETIC,
        ),
        Derivation(
            snapshot.id, kernel.id, "gate.post_route_timing", True,
            "fixture.timing", "1", [observation.id], stage="post_route",
            authority=AuthorityClass.SYNTHETIC,
        ),
    ])

    gates = CoreService(bundle, snapshot.id).verification_gates()
    assert gates["correctness"]["status"] == "pass"
    assert gates["correctness"]["checks"]["csim"]["status"] == "pass"
    assert gates["resource_fits"]["status"] == "pass"
    assert gates["post_route_timing"]["status"] == "pass"
    assert gates["correctness"]["required_checks_met"] is False
    assert gates["verified"] is False


def test_source_snippet_is_snapshot_aware_and_fails_if_bytes_changed(tmp_path):
    project = Project.create_from_manifest(
        _write_manifest(tmp_path, project_id="test.snippet_hash")
    )
    indexed = project.index(degraded=True)
    artifact = next(item for item in project.bundle.store.artifacts(indexed.snapshot_id)
                    if item.uri == "kernel.cpp")
    assert "void dut" in project.bundle.source_snippet(
        artifact.id, 1, 1, snapshot_id=indexed.snapshot_id, allow_private=True,
    )

    (tmp_path / "kernel.cpp").write_text(
        "void dut() { int newer = 1; }\n", encoding="utf-8"
    )
    assert project.bundle.is_stale(project.bundle.store.snapshot(indexed.snapshot_id))
    with pytest.raises(BundleError, match="no longer matches"):
        project.bundle.source_snippet(
            artifact.id, 1, 1, snapshot_id=indexed.snapshot_id, allow_private=True,
        )


def test_first_failed_index_is_not_selected_by_default_and_successful_retry_is_clean(
    tmp_path, monkeypatch,
):
    project = Project.create_from_manifest(
        _write_manifest(tmp_path, project_id="test.failed_retry")
    )

    def fail(_self, _context):
        raise ExtractionError("intentional first-attempt failure")

    monkeypatch.setattr(LibClangExtractor, "extract", fail)
    failed = project.index(degraded=False)
    assert failed.success is False
    assert project.bundle.store.has_graph(failed.snapshot_id) is False
    with pytest.raises(ValueError, match="successful|indexed|canonical graph"):
        project.service()

    monkeypatch.setattr(LibClangExtractor, "extract", _standard_success)
    retried = project.index(degraded=False)
    assert retried.success is True
    assert retried.snapshot_id == failed.snapshot_id
    assert project.bundle.store.has_graph(retried.snapshot_id)
    assert len(project.bundle.store.runs(retried.snapshot_id)) == 2
    assert project.bundle.store.diagnostics(retried.snapshot_id)  # retained attempt history

    status = project.status().to_dict()
    assert status["snapshot_id"] == retried.snapshot_id
    assert status["graph_available"] is True
    assert status["completeness"] == "complete"
