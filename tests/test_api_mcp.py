from __future__ import annotations

import asyncio
import importlib.util
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from hlsgraph.api import RestApplication, make_handler, openapi_document, serve
from hlsgraph.bundle import GraphBundle
from hlsgraph.graph import CanonicalGraph
from hlsgraph.manifest import minimal_manifest
from hlsgraph.mcp import ReadOnlyMcpService, create_mcp
from hlsgraph.model import (
    AuthorityClass,
    Diagnostic,
    DiagnosticSeverity,
    Entity,
    GateKind,
    GateResult,
    GateStatus,
    KnowledgeRule,
    Observation,
    PredictionEnvelope,
    Relation,
    RunStatus,
    SourceAnchor,
    ToolRun,
)
from hlsgraph.query import CoreService, QuerySpec


@pytest.fixture()
def indexed_bundle(tmp_path):
    secret = "PRIVATE_KERNEL_BODY_DO_NOT_EXPOSE_7f43"
    source = tmp_path / "kernel.cpp"
    source.write_text(f"// {secret}\nvoid dut() {{}}\n", encoding="utf-8")
    manifest = minimal_manifest("test.api", "API fixture", "dut", "kernel.cpp")
    manifest.artifact_paths[0]["license"] = "Proprietary"
    bundle = GraphBundle.create(tmp_path, manifest)

    first = bundle.snapshot()
    artifact = bundle.store.artifacts(first.id)[0]
    kernel = Entity(
        kind="hls.kernel", name="dut", qualified_name="dut", snapshot_id=first.id,
        stage="ast", anchors=[SourceAnchor(artifact.id, start_line=2, end_line=2)],
    )
    region = Entity(
        kind="hls.dataflow_region", name="compute", qualified_name="dut::compute",
        snapshot_id=first.id, stage="hls_ir",
    )
    graph = CanonicalGraph(first.id)
    graph.add_entity(kernel)
    graph.add_entity(region)
    graph.add_relation(Relation(kernel.id, region.id, "hls.contains", first.id,
                                stage="hls_ir"))
    bundle.store.save_graph(graph)
    observation = Observation(
        snapshot_id=first.id, subject_id=region.id, predicate="schedule.achieved_ii",
        value=2, unit="cycles", stage="schedule",
        authority=AuthorityClass.TOOL_OBSERVATION, artifact_id=artifact.id,
    )
    bundle.store.add_observations([observation])
    bundle.store.add_diagnostics([Diagnostic(
        snapshot_id=first.id, code="fixture.partial", severity=DiagnosticSeverity.WARNING,
        message="fixture intentionally has partial evidence", subject_id=region.id,
        stage="schedule",
    )])
    bundle.store.add_run(ToolRun(
        snapshot_id=first.id, stage="csynth", backend="fake.runner", request_hash="a" * 64,
        status=RunStatus.SUCCEEDED,
    ))
    bundle.store.add_knowledge_rules([KnowledgeRule(
        document_id="amd.ug1399", document_version="2024.2", section="Dataflow Viewer",
        rule_id="dynamic-workload-scope", title="Dynamic dataflow values are workload scoped",
        applicability={"tool": "vitis_hls", "version": "2024.2"},
        condition={"observation": "dynamic_fifo_stall"},
        effect={"requires": "workload_id"},
        citation_url="https://docs.amd.com/r/en-US/ug1399-vitis-hls/Dataflow-Viewer",
        summary="Do not promote a workload observation to an unconditional static fact.",
    )])

    source.write_text("void dut() {}\nvoid added() {}\n", encoding="utf-8")
    second = bundle.snapshot(parent_snapshot_id=first.id)
    graph2 = CanonicalGraph(second.id)
    graph2.add_entity(Entity(kind="hls.kernel", name="dut", qualified_name="dut",
                             snapshot_id=second.id, stage="ast"))
    graph2.add_entity(Entity(kind="hls.process", name="added", qualified_name="dut::added",
                             snapshot_id=second.id, stage="hls_ir"))
    bundle.store.save_graph(graph2)

    return {
        "bundle": bundle,
        "first": first.id,
        "second": second.id,
        "kernel": kernel.id,
        "region": region.id,
        "secret": secret,
    }


def test_rest_is_get_only_and_openapi_contains_no_mutations(indexed_bundle):
    service = CoreService(indexed_bundle["bundle"], indexed_bundle["first"])
    app = RestApplication(service)

    status = app.dispatch("GET", "/api/v1/status")
    assert status.status == 200
    assert status.body == service.status().to_dict()

    rejected = app.dispatch("POST", "/api/v1/status")
    assert rejected.status == 405
    assert rejected.headers["Allow"] == "GET"

    document = app.dispatch("GET", "/openapi.json")
    assert document.status == 200
    assert document.body == openapi_document()
    assert all(set(operations) == {"get"} for operations in document.body["paths"].values())


def test_rest_search_delegates_to_core_service(indexed_bundle):
    core = CoreService(indexed_bundle["bundle"], indexed_bundle["first"])
    app = RestApplication(core)
    expected = core.query(QuerySpec(query="compute", limit=5)).to_dict()

    response = app.dispatch("GET", "/api/v1/entities?q=compute&limit=5")

    assert response.status == 200
    assert response.body == expected
    assert response.body["items"][0]["entity_id"] == indexed_bundle["region"]


def test_rest_resources_and_evidence_never_include_source_contents(indexed_bundle):
    app = RestApplication(CoreService(indexed_bundle["bundle"], indexed_bundle["first"]))
    targets = [
        "/api/v1/overview", "/api/v1/graph", "/api/v1/entities",
        "/api/v1/observations", "/api/v1/diagnostics", "/api/v1/runs",
        f"/api/v1/evidence?entity_id={indexed_bundle['region']}",
    ]
    for target in targets:
        response = app.dispatch("GET", target)
        assert response.status == 200, target
        serialized = json.dumps(response.body, ensure_ascii=False)
        assert indexed_bundle["secret"] not in serialized

    evidence = app.dispatch(
        "GET", f"/api/v1/evidence/{indexed_bundle['region']}"
    ).body
    assert evidence["observations"][0]["predicate"] == "schedule.achieved_ii"
    assert evidence["artifacts"][0]["access"] == "private"
    assert "content" not in evidence["artifacts"][0]


def test_rest_compare_and_validation_errors(indexed_bundle):
    app = RestApplication(CoreService(indexed_bundle["bundle"], indexed_bundle["first"]))
    compared = app.dispatch(
        "GET", f"/api/v1/compare?other_snapshot_id={indexed_bundle['second']}"
    )
    assert compared.status == 200
    assert ["hls.process", "dut::added"] in compared.body["entities_added"]

    assert app.dispatch("GET", "/api/v1/compare").status == 400
    assert app.dispatch("GET", "/api/v1/evidence/missing").status == 404
    assert app.dispatch("GET", "/api/v2/status").status == 404


def test_rest_defaults_to_loopback(indexed_bundle):
    with pytest.raises(ValueError, match="non-loopback"):
        serve(indexed_bundle["bundle"].project_root, host="0.0.0.0", port=0)


def test_http_adapter_serves_json_and_rejects_post(indexed_bundle):
    app = RestApplication(CoreService(indexed_bundle["bundle"], indexed_bundle["first"]))
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urllib.request.urlopen(base + "/api/v1/graph", timeout=2) as response:
            assert response.status == 200
            assert json.load(response)["snapshot_id"] == indexed_bundle["first"]
        request = urllib.request.Request(base + "/api/v1/status", data=b"{}", method="POST")
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request, timeout=2)
        assert error.value.code == 405
        assert error.value.headers["Allow"] == "GET"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_mcp_facade_reuses_queries_and_keeps_predictions_out_of_impact(indexed_bundle):
    core = CoreService(indexed_bundle["bundle"], indexed_bundle["first"])
    mcp = ReadOnlyMcpService(core)

    assert mcp.search("compute", limit=5) == core.query(
        QuerySpec(query="compute", limit=5)
    ).to_dict()
    assert mcp.overview()["snapshot_id"] == indexed_bundle["first"]
    assert mcp.context(scope_id=indexed_bundle["kernel"])["focus"] == indexed_bundle["kernel"]
    assert mcp.module_or_region("compute")["focus"] == indexed_bundle["region"]

    traversed = mcp.traverse(indexed_bundle["kernel"], direction="out")
    assert traversed["inference_policy"] == "explicit_relations_only"
    assert {item["id"] for item in traversed["entities"]} == {
        indexed_bundle["kernel"], indexed_bundle["region"],
    }
    impact = mcp.impact(indexed_bundle["kernel"])
    assert impact["impact_semantics"] == "dependency_facts_only"
    assert "software.calls" not in impact["relation_kinds"]
    assert "llvm.cfg" not in impact["relation_kinds"]
    assert impact["qor_prediction"] is None


def test_mcp_evidence_health_knowledge_compare_and_render(indexed_bundle):
    mcp = ReadOnlyMcpService(
        CoreService(indexed_bundle["bundle"], indexed_bundle["first"])
    )
    evidence = mcp.evidence(indexed_bundle["region"])
    assert evidence["observations"][0]["authority"] == "tool_observation"
    health = mcp.health()
    assert health["diagnostics"][0]["code"] == "fixture.partial"
    knowledge = mcp.knowledge(document_id="amd.ug1399")
    assert knowledge["authority_class"] == "knowledge_rule"
    assert knowledge["items"][0]["document_version"] == "2024.2"
    assert mcp.compare(indexed_bundle["second"])["right_snapshot_id"] == indexed_bundle["second"]

    rendered = mcp.render(format="mermaid")
    assert rendered["format"] == "mermaid"
    assert "flowchart LR" in rendered["content"]
    assert indexed_bundle["secret"] not in json.dumps(rendered)


def test_mcp_predictions_and_redacted_runs_are_separate_surfaces(indexed_bundle):
    bundle = indexed_bundle["bundle"]
    prediction = PredictionEnvelope(
        snapshot_id=indexed_bundle["first"], subject_id=indexed_bundle["region"],
        predicate="prediction.latency_cycles", value=12.5,
        model_id="test.model", model_version="1", input_schema_version="0.1.0",
    )
    bundle.store.add_prediction(prediction)
    mcp = ReadOnlyMcpService(CoreService(bundle, indexed_bundle["first"]))
    predictions = mcp.predictions(model_id="test.model")
    assert predictions["authority_class"] == "prediction_hypothesis"
    assert predictions["items"][0]["id"] == prediction.id
    runs = mcp.runs()
    assert runs["items"] and all("command" not in item for item in runs["items"])


def test_rest_and_mcp_expose_failed_candidate_health_without_graph(tmp_path):
    (tmp_path / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    bundle = GraphBundle.create(
        tmp_path, minimal_manifest("test.failed_surface", "failed", "dut", "kernel.cpp")
    )
    snapshot = bundle.snapshot()
    run = ToolRun(
        snapshot.id, "index", "extractor.local", "f" * 64,
        status=RunStatus.FAILED,
    )
    bundle.store.add_run(run)
    bundle.store.add_diagnostics([Diagnostic(
        snapshot_id=snapshot.id, code="extractor.failed",
        severity=DiagnosticSeverity.ERROR, message="safe failure summary", run_id=run.id,
    )])

    rest = RestApplication(bundle)
    status = rest.dispatch("GET", "/api/v1/status")
    assert status.status == 200
    assert status.body["latest_candidate_snapshot_id"] == snapshot.id
    assert rest.dispatch("GET", "/api/v1/diagnostics").body["total"] == 1
    assert rest.dispatch("GET", "/api/v1/runs").body["total"] == 1

    mcp = ReadOnlyMcpService(bundle)
    overview = rest.dispatch("GET", "/api/v1/overview")
    assert overview.status == 200
    assert overview.body == mcp.overview()
    assert overview.body["snapshot_id"] == snapshot.id
    assert overview.body["architecture"] is None
    assert overview.body["incomplete"] is True
    assert overview.body["status"]["graph_available"] is False
    assert mcp.health()["diagnostics"][0]["code"] == "extractor.failed"
    assert mcp.runs()["items"][0]["status"] == "failed"
    with pytest.raises(ValueError, match="no successful canonical graph"):
        mcp.search("dut")


def test_mcp_name_resolution_prefers_canonical_hls_entity(tmp_path):
    (tmp_path / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    bundle = GraphBundle.create(
        tmp_path, minimal_manifest("test.mcp_resolution", "resolution", "dut", "kernel.cpp")
    )
    snapshot = bundle.snapshot()
    graph = CanonicalGraph(snapshot.id)
    graph.add_entity(Entity("ir.mlir.function", "dut", snapshot.id,
                            qualified_name="ir/dut.mlir::dut", stage="mlir"))
    kernel = graph.add_entity(Entity("hls.kernel", "dut", snapshot.id,
                                     qualified_name="dut", stage="ast"))
    bundle.store.save_graph(graph)
    result = ReadOnlyMcpService(CoreService(bundle, snapshot.id)).module_or_region("dut")
    assert result["focus"] == kernel.id


def test_mcp_name_resolution_reports_same_priority_ambiguity(tmp_path):
    (tmp_path / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    bundle = GraphBundle.create(
        tmp_path, minimal_manifest("test.mcp_ambiguity", "ambiguity", "dut", "kernel.cpp")
    )
    snapshot = bundle.snapshot()
    graph = CanonicalGraph(snapshot.id)
    graph.add_entity(Entity("hls.process", "compute", snapshot.id,
                            qualified_name="dut::region_a::compute", stage="hls_ir"))
    graph.add_entity(Entity("hls.process", "compute", snapshot.id,
                            qualified_name="dut::region_b::compute", stage="hls_ir"))
    bundle.store.save_graph(graph)
    with pytest.raises(ValueError, match="ambiguous identifier"):
        ReadOnlyMcpService(CoreService(bundle, snapshot.id)).module_or_region("compute")


def test_rest_run_rows_redact_commands_paths_messages_and_backend_details(indexed_bundle):
    secret = "RUN_SECRET_SENTINEL_77dd"
    bundle = indexed_bundle["bundle"]
    bundle.store.add_run(ToolRun(
        snapshot_id=indexed_bundle["first"], stage="post_route",
        backend="runner.ssh", request_hash="e" * 64, status=RunStatus.FAILED,
        command=["vendor-tool", "--token", secret], working_directory="private/work",
        message="failed under " + "C" + f":/private/{secret}",
        gates=[GateResult(GateKind.POST_ROUTE_TIMING, GateStatus.FAIL,
                          reason=f"private timing detail {secret}")],
        metadata={"ssh_host": f"secret-host-{secret}",
                  "remote_project_root": f"/private/{secret}",
                  "stdout_sha256": "f" * 64, "stdout_bytes": 10},
    ))
    response = RestApplication(CoreService(bundle, indexed_bundle["first"])).dispatch(
        "GET", "/api/v1/runs"
    )
    serialized = json.dumps(response.body, ensure_ascii=False)
    assert response.status == 200 and secret not in serialized
    item = next(value for value in response.body["items"] if value["backend"] == "runner.ssh")
    assert "command" not in item and "working_directory" not in item and "message" not in item
    assert item["execution_metadata"]["command_redacted"] is True


def test_mcp_context_and_render_validate_bounds(indexed_bundle):
    mcp = ReadOnlyMcpService(
        CoreService(indexed_bundle["bundle"], indexed_bundle["first"])
    )
    with pytest.raises(ValueError, match="requires"):
        mcp.context()
    with pytest.raises(ValueError, match="format"):
        mcp.render(format="png")
    with pytest.raises(KeyError):
        mcp.traverse("missing")


def test_optional_fastmcp_defaults_to_explore_and_can_opt_in_legacy_tools(
    indexed_bundle, monkeypatch,
):
    if importlib.util.find_spec("mcp") is None:
        pytest.skip("optional mcp dependency is not installed")
    server = create_mcp(indexed_bundle["bundle"].project_root,
                        snapshot_id=indexed_bundle["first"])
    registered = {item.name for item in asyncio.run(server.list_tools())}
    assert registered == {"explore"}

    monkeypatch.setenv("HLSGRAPH_MCP_TOOLS", "all")
    server = create_mcp(indexed_bundle["bundle"].project_root,
                        snapshot_id=indexed_bundle["first"])
    registered = {item.name for item in asyncio.run(server.list_tools())}
    assert registered == {"explore",
        "overview", "search", "context", "module_or_region", "traverse", "impact",
        "evidence", "feature_evidence", "correspondences", "compare", "health",
        "runs", "predictions", "variants", "render", "knowledge",
    }
