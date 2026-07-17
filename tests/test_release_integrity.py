from __future__ import annotations

import json
import sqlite3

import pytest

from hlsgraph.bundle import GraphBundle
from hlsgraph.extract import ExtractionContext, ExtractionPipeline, ExtractionResult
from hlsgraph.graph import CanonicalGraph
from hlsgraph.manifest import minimal_manifest
from hlsgraph.model import (
    ArtifactRef,
    AuthorityClass,
    Completeness,
    Derivation,
    Diagnostic,
    DiagnosticSeverity,
    Entity,
    GateKind,
    GateResult,
    GateStatus,
    Observation,
    PredictionEnvelope,
    Relation,
    RunStatus,
    SourceAnchor,
    ToolRun,
    ToolchainContext,
    VerificationKind,
    VerificationResult,
    json_ready,
    stable_hash,
)
from hlsgraph.query import CoreService, ExploreSpec, QuerySpec
from hlsgraph.sdk import Project
from hlsgraph.store import StoreError
from hlsgraph.version import SCHEMA_VERSION


def _bundle(tmp_path, project_id: str = "test.release.integrity"):
    (tmp_path / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    manifest = minimal_manifest(
        project_id, "release integrity", "dut", "kernel.cpp",
        part="part-a", clock_ns=5.0,
    )
    manifest.target.capacities = {"lut": 100.0}
    manifest.stage_commands = {
        "csim": ["vitis_hls", "--csim"],
        "rtl_cosim": ["vitis_hls", "--cosim"],
        "post_route": ["vivado", "--post-route"],
    }
    manifest.toolchains = [ToolchainContext(
        id="amd.unified.2024_2", vendor="amd", name="unified", version="2024.2",
        environment_hash="e" * 64,
    )]
    bundle = GraphBundle.create(tmp_path, manifest)
    snapshot = bundle.snapshot()
    artifact = bundle.store.artifacts(snapshot.id)[0]
    kernel = Entity("hls.kernel", "dut", snapshot.id, qualified_name="dut", stage="ast")
    graph = CanonicalGraph(snapshot.id)
    graph.add_entity(kernel)
    return bundle, snapshot, artifact, kernel, graph


def test_canonical_graph_schema_and_authority_fail_closed(tmp_path):
    bundle, snapshot, _artifact, kernel, graph = _bundle(tmp_path)
    payload = graph.to_dict()
    payload.pop("schema_version")
    with pytest.raises(ValueError, match="schema_version"):
        CanonicalGraph.from_dict(payload)
    with pytest.raises(ValueError, match="schema"):
        CanonicalGraph(snapshot.id, schema_version="99.0")

    graph.schema_version = "99.0"
    with pytest.raises(StoreError, match="schema"):
        bundle.store.save_graph(graph)
    graph.schema_version = SCHEMA_VERSION
    bundle.store.save_graph(graph)
    with sqlite3.connect(bundle.store.path) as connection:
        connection.execute(
            "UPDATE graph_views SET schema_version='99.0' WHERE snapshot_id=?",
            (snapshot.id,),
        )
    with pytest.raises(StoreError, match="schema"):
        bundle.store.set_active_snapshot(bundle.manifest.project_id, snapshot.id)

    for authority in (
        AuthorityClass.KNOWLEDGE_RULE, AuthorityClass.PREDICTION_HYPOTHESIS,
    ):
        with pytest.raises(ValueError, match="dedicated"):
            Entity("hls.process", "bad", snapshot.id, authority=authority)
        with pytest.raises(ValueError, match="dedicated"):
            Relation(kernel.id, kernel.id, "hls.contains", snapshot.id,
                     authority=authority)
        with pytest.raises(ValueError, match="dedicated"):
            Observation(snapshot.id, kernel.id, "test.value", 1, "source", authority)
        with pytest.raises(ValueError, match="dedicated"):
            Derivation(
                snapshot.id, kernel.id, "test.derived", 1, "test", "1", ["obs"],
                authority=authority,
            )


def test_graph_metadata_and_extractor_failures_cannot_leak_embedded_bodies(tmp_path):
    bundle, snapshot, _artifact, _kernel, graph = _bundle(tmp_path)
    secret = "PRIVATE_EXCEPTION_OR_SOURCE_BODY_SENTINEL_71c9"
    with pytest.raises(ValueError, match="embedded body"):
        CanonicalGraph(snapshot.id, metadata={"raw_source": secret})
    graph.metadata["source_text"] = secret
    with pytest.raises(StoreError, match="embedded body"):
        bundle.store.save_graph(graph)
    assert secret.encode() not in bundle.store.path.read_bytes()
    with pytest.raises(ValueError, match="exceeds"):
        Diagnostic(
            snapshot.id, "test.too_long", DiagnosticSeverity.ERROR, "x" * 2049,
        )

    class LeakyMetadataExtractor:
        name = "test.leaky_metadata"
        version = "1"

        @staticmethod
        def supports(_context):
            return True

        @staticmethod
        def extract(context):
            value = CanonicalGraph(context.snapshot.id)
            value.metadata["raw_source"] = secret
            return ExtractionResult(graph=value)

    class LeakyExceptionExtractor:
        name = "test.leaky_exception"
        version = "1"

        @staticmethod
        def supports(_context):
            return True

        @staticmethod
        def extract(_context):
            raise RuntimeError(secret)

    context = ExtractionContext(
        project_root=bundle.project_root, manifest=bundle.manifest, snapshot=snapshot,
        artifacts={item.id: item for item in bundle.store.artifacts(snapshot.id)},
    )
    result = ExtractionPipeline([
        LeakyMetadataExtractor(), LeakyExceptionExtractor(),
    ]).run(context)
    serialized = json.dumps({
        "graph": result.graph.to_dict(),
        "diagnostics": [json_ready(item) for item in result.diagnostics],
    })
    assert secret not in serialized
    assert len(result.diagnostics) == 2
    assert all("details withheld" in item.message for item in result.diagnostics)
    assert all(len(item.metadata["error_fingerprint"]) == 64
               for item in result.diagnostics)


def test_entity_relation_identity_distinguishes_stage_and_authority(tmp_path):
    _bundle_value, snapshot, _artifact, _kernel, _graph = _bundle(tmp_path)
    first = Entity("hls.process", "compute", snapshot.id, qualified_name="dut::compute",
                   stage="ast", authority=AuthorityClass.STATIC_FACT)
    second = Entity("hls.process", "compute", snapshot.id, qualified_name="dut::compute",
                    stage="hls_ir", authority=AuthorityClass.COMPILER_DECISION)
    assert first.id != second.id
    edge_a = Relation(first.id, second.id, "cross.maps_to", snapshot.id,
                      stage="ast", authority=AuthorityClass.STATIC_FACT)
    edge_b = Relation(first.id, second.id, "cross.maps_to", snapshot.id,
                      stage="hls_ir", authority=AuthorityClass.COMPILER_DECISION)
    assert edge_a.id != edge_b.id


@pytest.mark.parametrize("field,changed", [
    ("trainset_hash", "b" * 64),
    ("input_schema_version", "features.v2"),
    ("unit", "ns"),
    ("uncertainty", {"stddev": 2.0}),
    ("applicability", {"vendor": "amd", "part": "xck26"}),
    ("ood", {"score": 0.9, "flag": True}),
    ("metadata", {"calibration": "isotonic"}),
    ("value", 12.0),
])
def test_prediction_identity_covers_complete_semantics(tmp_path, field, changed):
    _bundle_value, snapshot, _artifact, kernel, _graph = _bundle(tmp_path)
    base = {
        "snapshot_id": snapshot.id,
        "subject_id": kernel.id,
        "predicate": "prediction.latency_cycles",
        "value": 10.0,
        "model_id": "model.test",
        "model_version": "1",
        "input_schema_version": "features.v1",
        "unit": "cycle",
        "trainset_hash": "a" * 64,
        "uncertainty": {"stddev": 1.0},
        "applicability": {"vendor": "amd"},
        "ood": {"score": 0.1, "flag": False},
        "metadata": {"calibration": "none"},
    }
    first = PredictionEnvelope(**base)
    second = PredictionEnvelope(**{**base, field: changed})
    assert first.id != second.id, field


def test_graph_anchors_must_reference_artifacts_attached_to_same_snapshot(tmp_path):
    bundle, snapshot, artifact, kernel, _graph = _bundle(tmp_path)
    bad_entity = Entity(
        "hls.process", "bad", snapshot.id, stage="hls_ir",
        anchors=[SourceAnchor("artifact.missing", start_line=1)],
    )
    graph = CanonicalGraph(snapshot.id)
    graph.add_entity(kernel)
    graph.add_entity(bad_entity)
    with pytest.raises(StoreError, match="anchor artifact"):
        bundle.store.save_graph(graph)

    process = Entity(
        "hls.process", "ok", snapshot.id, stage="hls_ir",
        anchors=[SourceAnchor(artifact.id, start_line=1)],
    )
    graph = CanonicalGraph(snapshot.id)
    graph.add_entity(kernel)
    graph.add_entity(process)
    graph.add_relation(Relation(
        kernel.id, process.id, "hls.contains", snapshot.id, stage="hls_ir",
        anchors=[SourceAnchor("artifact.missing", start_line=1)],
    ))
    with pytest.raises(StoreError, match="anchor artifact"):
        bundle.store.save_graph(graph)


def test_successful_index_commit_is_atomic_and_allows_forward_evidence(tmp_path):
    bundle, snapshot, artifact, kernel, graph = _bundle(tmp_path)
    run = ToolRun(
        snapshot.id, "index", "extractor.local", "1" * 64,
        status=RunStatus.SUCCEEDED,
    )
    observation = Observation(
        snapshot.id, kernel.id, "test.index_observation", 1, "ast",
        AuthorityClass.STATIC_FACT, artifact_id=artifact.id, run_id=run.id,
    )
    run.gates = [GateResult(GateKind.RESOURCE_FITS, GateStatus.PASS,
                            evidence_ids=[observation.id])]
    diagnostic = Diagnostic(
        snapshot.id, "test.index_info", DiagnosticSeverity.INFO, "ok",
        run_id=run.id,
    )
    run.diagnostics = [diagnostic.id]
    bundle.store.commit_index_success(
        project_id=bundle.manifest.project_id, graph=graph, run=run,
        observations=[observation], derivations=[], verifications=[],
        diagnostics=[diagnostic],
    )
    assert bundle.store.latest_snapshot(bundle.manifest.project_id).id == snapshot.id
    assert bundle.store.observations(snapshot.id)[0].run_id == run.id

    second = bundle.snapshot(action_id="rollback")
    second_kernel = Entity("hls.kernel", "dut", second.id, qualified_name="dut", stage="ast")
    second_graph = CanonicalGraph(second.id)
    second_graph.add_entity(second_kernel)
    second_run = ToolRun(
        second.id, "index", "extractor.local", "2" * 64,
        status=RunStatus.SUCCEEDED,
    )
    invalid = Derivation(
        second.id, second_kernel.id, "test.invalid", True, "test", "1",
        ["observation.missing"],
    )
    with pytest.raises(StoreError, match="observation"):
        bundle.store.commit_index_success(
            project_id=bundle.manifest.project_id, graph=second_graph, run=second_run,
            observations=[], derivations=[invalid], verifications=[], diagnostics=[],
        )
    assert bundle.store.has_graph(second.id) is False
    assert bundle.store.runs(second.id) == []
    assert bundle.store.latest_snapshot(bundle.manifest.project_id).id == snapshot.id


def test_run_output_artifact_and_gate_evidence_have_atomic_write_path(tmp_path):
    bundle, snapshot, source_artifact, kernel, graph = _bundle(tmp_path)
    bundle.store.save_graph(graph)
    run = _bound_tool_run(bundle, snapshot, "csim", "3")
    report = _managed_report(
        bundle, run, "atomic-csim.json", "amd.vitis.csim_result",
    )
    observation = Observation(
        snapshot.id, kernel.id, "csim.status", "pass", "csim",
        AuthorityClass.VERIFICATION_EVIDENCE, run_id=run.id, artifact_id=report.id,
    )
    run.output_artifact_ids = [report.id]
    run.gates = [GateResult(
        GateKind.CORRECTNESS, GateStatus.PASS, evidence_ids=[observation.id],
    )]
    bundle.store.commit_run_result(
        run=run, artifacts=[report], observations=[observation],
    )
    assert bundle.store.runs(snapshot.id)[0].output_artifact_ids == [report.id]
    assert next(item for item in bundle.store.artifacts(snapshot.id)
                if item.id == report.id).producer_run_id == run.id

    spoof = _bound_tool_run(bundle, snapshot, "csim", "d")
    spoof.output_artifact_ids = [source_artifact.id]
    with pytest.raises(StoreError, match="contract mismatch"):
        bundle.store.commit_run_result(run=spoof)
    undeclared_run = _bound_tool_run(bundle, snapshot, "csim", "e")
    undeclared = ArtifactRef(
        "tool.report", ".hlsgraph/artifacts/undeclared/report.json", "e" * 64, 1,
        producer_run_id=undeclared_run.id,
    )
    with pytest.raises(StoreError, match="contract mismatch"):
        bundle.store.commit_run_result(run=undeclared_run, artifacts=[undeclared])

    missing = ArtifactRef(
        "tool.report", ".hlsgraph/artifacts/missing/report.json", "b" * 64, 1,
        producer_run_id="run.missing",
    )
    with pytest.raises(StoreError, match="producer run"):
        bundle.store.add_artifact(snapshot.id, missing)
    other = bundle.snapshot(action_id="other")
    with pytest.raises(StoreError, match="cross-snapshot"):
        bundle.store.add_artifact(other.id, report)
    with pytest.raises(StoreError, match="gate evidence"):
        bundle.store.add_run(ToolRun(
            other.id, "csim", "runner.local", "4" * 64,
            gates=[GateResult(GateKind.CORRECTNESS, GateStatus.PASS,
                              evidence_ids=["observation.missing"])],
        ))
    with pytest.raises(ValueError, match="passing gate"):
        GateResult(GateKind.CORRECTNESS, GateStatus.PASS)
    bypassed = GateResult(GateKind.CORRECTNESS, GateStatus.FAIL)
    bypassed.status = GateStatus.PASS
    with pytest.raises(StoreError, match="passing gate"):
        bundle.store.add_run(ToolRun(
            other.id, "csim", "runner.local", "5" * 64, gates=[bypassed],
        ))


def _bound_tool_run(bundle, snapshot, stage: str, request_char: str) -> ToolRun:
    manifest = bundle.store.snapshot_manifest(snapshot.id)
    toolchain = manifest.toolchain_for_stage(stage)
    return ToolRun(
        snapshot.id, stage, "runner.local", request_char * 64,
        toolchain_id=toolchain.id,
        status=RunStatus.SUCCEEDED, exit_code=0,
        command=list(manifest.stage_commands[stage]),
        working_directory=".",
        environment_hash=toolchain.environment_hash,
        input_artifact_ids=[
            item.id for item in bundle.store.artifacts(snapshot.id)
            if item.producer_run_id is None
        ],
        metadata={
            "authority": "tool_observation", "tool_truth": True,
            "fresh_execution": True, "fresh_tool_truth": True,
        },
    )


def _managed_report(bundle, run, name, kind, metadata=None):
    source = bundle.project_root / f"fixture-{name}"
    source.write_text(f"sanitized {name}\n", encoding="utf-8")
    artifact, _path, _created = bundle.prepare_managed_artifact(
        source, kind=kind, role="tool_output", producer_run_id=run.id,
        metadata=metadata or {},
    )
    return artifact


def _commit_verification_stage(
    bundle, snapshot, kernel, stage, char, kind, *,
    campaign="campaign.golden", workload="workload.golden",
):
    run = _bound_tool_run(bundle, snapshot, stage, char)
    run.metadata.update({"campaign_id": campaign, "workload_id": workload})
    artifact = _managed_report(
        bundle, run, f"{stage}-{char}.rpt",
        "amd.vitis.csim_result" if stage == "csim" else "amd.vitis.cosim_rpt",
        {"workload_id": workload},
    )
    run.output_artifact_ids = [artifact.id]
    if stage == "csim":
        observations = [
            Observation(
                snapshot.id, kernel.id, predicate, 0, "csim",
                AuthorityClass.VERIFICATION_EVIDENCE, run_id=run.id,
                artifact_id=artifact.id, workload_id=workload, unit="count",
            )
            for predicate in (
                "csim.exit_code", "csim.mismatches", "csim.assertions_failed",
            )
        ]
    else:
        observations = [Observation(
            snapshot.id, kernel.id, "cosim.status", "pass", "cosim",
            AuthorityClass.VERIFICATION_EVIDENCE, run_id=run.id,
            artifact_id=artifact.id, workload_id=workload,
        )]
    verification = VerificationResult(
        snapshot.id, kind, GateStatus.PASS, run_id=run.id,
        workload_id=workload, evidence_ids=[item.id for item in observations],
    )
    bundle.store.commit_run_result(
        run=run, artifacts=[artifact], observations=observations,
        verifications=[verification],
    )


def test_verified_requires_recursive_real_tool_truth_and_failure_dominates(tmp_path):
    bundle, snapshot, _artifact, kernel, graph = _bundle(tmp_path)
    bundle.store.save_graph(graph)
    _commit_verification_stage(
        bundle, snapshot, kernel, "csim", "a", VerificationKind.CSIM,
    )
    _commit_verification_stage(
        bundle, snapshot, kernel, "rtl_cosim", "b", VerificationKind.RTL_COSIM,
    )

    run = _bound_tool_run(bundle, snapshot, "post_route", "c")
    scope = {"kind": "kernel", "top": "dut", "instance": "dut", "part": "part-a"}
    utilization = _managed_report(
        bundle, run, "post-route-util.rpt", "amd.vivado.post_route_utilization",
        {"scope": scope},
    )
    timing_report = _managed_report(
        bundle, run, "post-route-timing.rpt", "amd.vivado.post_route_timing",
        {"scope": {**scope, "clock": "default"}},
    )
    run.output_artifact_ids = [utilization.id, timing_report.id]
    resource = Observation(
        snapshot.id, kernel.id, "resource.lut", 10, "post_route",
        AuthorityClass.TOOL_OBSERVATION, run_id=run.id, artifact_id=utilization.id,
        unit="count",
    )
    timing = Observation(
        snapshot.id, kernel.id, "timing.wns_ns", 0.1, "post_route",
        AuthorityClass.TOOL_OBSERVATION, run_id=run.id, artifact_id=timing_report.id,
        unit="ns",
    )
    fits = Derivation(
        snapshot.id, kernel.id, "gate.resource_fits", True,
        "hlsgraph.gate.capacity_compare", "1", [resource.id], stage="post_route",
        metadata={"target_profile_hash": stable_hash(bundle.manifest.target)},
    )
    met = Derivation(
        snapshot.id, kernel.id, "gate.post_route_timing", True,
        "hlsgraph.gate.wns_nonnegative", "1", [timing.id], stage="post_route",
    )
    bundle.store.commit_run_result(
        run=run, artifacts=[utilization, timing_report], observations=[resource, timing],
        derivations=[fits, met],
    )
    gates = CoreService(bundle, snapshot.id).verification_gates()
    assert gates["correctness"]["required_checks_met"] is True
    assert gates["resource_fits"]["trusted_pass"] is True
    assert gates["post_route_timing"]["trusted_pass"] is True
    assert gates["verified"] is True

    synthetic_observation = Observation(
        snapshot.id, kernel.id, "fixture.resource", 999, "post_route",
        AuthorityClass.SYNTHETIC,
    )
    synthetic_failure = Derivation(
        snapshot.id, kernel.id, "gate.resource_fits", False,
        "fixture.answer", "1", [synthetic_observation.id], stage="post_route",
        authority=AuthorityClass.SYNTHETIC,
    )
    bundle.store.add_observations([synthetic_observation])
    bundle.store.add_derivations([synthetic_failure])
    gates = CoreService(bundle, snapshot.id).verification_gates()
    assert gates["resource_fits"]["status"] == "fail"
    assert gates["verified"] is False


def test_correctness_checks_do_not_combine_across_campaigns(tmp_path):
    bundle, snapshot, _artifact, kernel, graph = _bundle(tmp_path)
    bundle.store.save_graph(graph)
    _commit_verification_stage(
        bundle, snapshot, kernel, "csim", "a", VerificationKind.CSIM,
        campaign="campaign.a", workload="workload.a",
    )
    _commit_verification_stage(
        bundle, snapshot, kernel, "rtl_cosim", "b", VerificationKind.RTL_COSIM,
        campaign="campaign.b", workload="workload.b",
    )
    correctness = CoreService(bundle, snapshot.id).verification_gates()["correctness"]
    assert correctness["checks"]["csim"]["status"] == "pass"
    assert correctness["checks"]["rtl_cosim"]["status"] == "pass"
    assert correctness["required_checks_met"] is False
    assert correctness["eligible_campaigns"] == []


def test_manifest_typed_report_without_tool_producer_cannot_verify(tmp_path):
    bundle, snapshot, artifact, kernel, graph = _bundle(tmp_path)
    bundle.store.save_graph(graph)
    imported = Observation(
        snapshot.id, kernel.id, "timing.wns_ns", 0.2, "post_route",
        AuthorityClass.TOOL_OBSERVATION, artifact_id=artifact.id,
    )
    derived = Derivation(
        snapshot.id, kernel.id, "gate.post_route_timing", True,
        "test.imported_report", "1", [imported.id], stage="post_route",
    )
    bundle.store.add_observations([imported])
    bundle.store.add_derivations([derived])
    gate = CoreService(bundle, snapshot.id).verification_gates()["post_route_timing"]
    assert gate["status"] == "pass"
    assert gate["trusted_pass"] is False
    assert gate["tool_truth"] is False


def test_cursor_architecture_projection_and_exact_context_ambiguity(tmp_path):
    bundle, snapshot, _artifact, kernel, graph = _bundle(tmp_path)
    first = Entity("hls.process", "compute", snapshot.id,
                   qualified_name="dut::a::compute", stage="hls_ir",
                   authority=AuthorityClass.COMPILER_DECISION)
    second = Entity("hls.process", "compute", snapshot.id,
                    qualified_name="dut::b::compute", stage="hls_ir",
                    authority=AuthorityClass.COMPILER_DECISION)
    llvm = Entity("ir.llvm.operation", "add", snapshot.id,
                  qualified_name="dut::llvm::add", stage="llvm",
                  authority=AuthorityClass.COMPILER_DECISION)
    source_loop = Entity("hls.loop", "loop", snapshot.id,
                         qualified_name="dut::loop", stage="ast")
    for item in (first, second, llvm, source_loop):
        graph.add_entity(item)
    graph.add_relation(Relation(
        kernel.id, first.id, "hls.contains", snapshot.id, stage="hls_ir",
        authority=AuthorityClass.COMPILER_DECISION,
    ))
    graph.add_relation(Relation(
        first.id, second.id, "hls.streams_to", snapshot.id, stage="hls_ir",
        authority=AuthorityClass.COMPILER_DECISION,
    ))
    graph.add_relation(Relation(
        kernel.id, source_loop.id, "hls.contains", snapshot.id, stage="ast",
        authority=AuthorityClass.STATIC_FACT,
    ))
    graph.add_relation(Relation(
        first.id, source_loop.id, "hls.streams_to", snapshot.id, stage="ast",
        authority=AuthorityClass.STATIC_FACT,
    ))
    graph.add_relation(Relation(
        kernel.id, second.id, "software.calls", snapshot.id, stage="ast",
        authority=AuthorityClass.STATIC_FACT,
        attrs={"hardware_instance": False},
    ))
    graph.add_relation(Relation(
        llvm.id, llvm.id, "llvm.cfg", snapshot.id, stage="llvm",
        authority=AuthorityClass.COMPILER_DECISION,
        attrs={"hardware_topology": False},
    ))
    bundle.store.save_graph(graph)
    service = CoreService(bundle, snapshot.id)

    first_page = service.query(QuerySpec("compute", limit=1))
    assert first_page.next_cursor
    second_page = service.query(QuerySpec(
        "compute", limit=1, cursor=first_page.next_cursor,
    ))
    assert second_page.items
    assert second_page.items[0].entity_id != first_page.items[0].entity_id
    with pytest.raises(ValueError, match="ambiguous exact context"):
        service.explore(ExploreSpec(query="compute", view="architecture"))
    with pytest.raises(KeyError, match="architecture view"):
        service.explore(ExploreSpec(query="loop", view="architecture"))

    architecture = service.explore(ExploreSpec(
        scope_id=kernel.id, view="architecture", depth=3,
    ))
    assert llvm.id not in {item["id"] for item in architecture.entities}
    kinds = {item["kind"] for item in architecture.relations}
    assert "hls.streams_to" in kinds
    assert not ({"software.calls", "llvm.cfg"} & kinds)
    assert all(not (item["kind"] == "hls.contains" and item["stage"] == "ast")
               for item in architecture.relations)
    assert all(not (item["kind"] == "hls.streams_to" and item["stage"] == "ast")
               for item in architecture.relations)
    evidence = service.explore(ExploreSpec(
        scope_id=kernel.id, view="evidence", depth=3,
    ))
    assert "software.calls" in {item["kind"] for item in evidence.relations}


def test_compare_preserves_parallel_relations_and_observation_completeness(tmp_path):
    bundle, left, artifact, _kernel, _graph = _bundle(tmp_path)
    right = bundle.snapshot(action_id="right")

    def make_graph(snapshot_id, depths):
        source = Entity("hls.process", "source", snapshot_id,
                        qualified_name="dut::source", stage="hls_ir",
                        authority=AuthorityClass.COMPILER_DECISION)
        sink = Entity("hls.process", "sink", snapshot_id,
                      qualified_name="dut::sink", stage="hls_ir",
                      authority=AuthorityClass.COMPILER_DECISION)
        value = CanonicalGraph(snapshot_id)
        value.add_entity(source)
        value.add_entity(sink)
        for depth in depths:
            value.add_relation(Relation(
                source.id, sink.id, "hls.streams_to", snapshot_id,
                stage="hls_ir", authority=AuthorityClass.COMPILER_DECISION,
                attrs={"fifo_depth": depth},
            ))
        return value, source

    left_graph, left_source = make_graph(left.id, [4, 8])
    right_graph, right_source = make_graph(right.id, [4, 16])
    bundle.store.save_graph(left_graph)
    bundle.store.save_graph(right_graph)
    left_observation = Observation(
        left.id, left_source.id, "qor.latency_cycles", 10, "schedule",
        AuthorityClass.COMPILER_DECISION, artifact_id=artifact.id,
        completeness=Completeness.COMPLETE,
    )
    right_observation = Observation(
        right.id, right_source.id, "qor.latency_cycles", 10, "schedule",
        AuthorityClass.COMPILER_DECISION, artifact_id=artifact.id,
        completeness=Completeness.PARTIAL,
    )
    bundle.store.add_observations([left_observation, right_observation])
    left_policy = ArtifactRef(
        "tool.report", ".hlsgraph/artifacts/policy/report.json", "d" * 64, 1,
        role="report", license="Apache-2.0",
    )
    right_policy = ArtifactRef(
        "tool.report", ".hlsgraph/artifacts/policy/report.json", "d" * 64, 1,
        role="report", license="Proprietary",
    )
    bundle.store.add_artifact(left.id, left_policy)
    bundle.store.add_artifact(right.id, right_policy)
    compared = CoreService(bundle, left.id).compare(right.id)
    assert compared["relations_changed"]
    assert compared["observations_changed"]
    assert ".hlsgraph/artifacts/policy/report.json" in compared["artifacts_changed"]


def test_sdk_rejects_index_outside_single_top_kernel_boundary(tmp_path):
    (tmp_path / "kernel.cpp").write_text("void other() {}\n", encoding="utf-8")
    manifest = minimal_manifest(
        "test.kernel.boundary", "kernel boundary", "dut", "kernel.cpp",
    )
    path = tmp_path / "hlsgraph.json"
    path.write_text(__import__("json").dumps(json_ready(manifest)), encoding="utf-8")
    project = Project.create_from_manifest(path)
    result = project.index(degraded=True)
    assert result.success is False
    assert project.bundle.store.has_graph(result.snapshot_id) is False
    assert any(item.code == "extractor.kernel_boundary"
               for item in project.bundle.store.diagnostics(result.snapshot_id))
