from __future__ import annotations

import json

from hlsgraph.api import RestApplication
from hlsgraph.bundle import GraphBundle
from hlsgraph.cli import main
from hlsgraph.diagnostic_projection import public_diagnostic
from hlsgraph.graph import CanonicalGraph
from hlsgraph.manifest import minimal_manifest
from hlsgraph.mcp import ReadOnlyMcpService
from hlsgraph.model import (
    Diagnostic, DiagnosticSeverity, Entity, SourceAnchor,
)
from hlsgraph.query import CoreService, ExploreSpec
from hlsgraph.render import render
from hlsgraph.sdk import Project


_SENTINEL = "PRIVATE_DIAGNOSTIC_SOURCE_91fbb3"
_PRIVATE_LOCATION = chr(47).join((
    "C:", "Users", "customer", "secret", f"{_SENTINEL}.cpp:17:9",
))
_PUBLIC_KEYS = {
    "id", "snapshot_id", "code", "severity", "stage", "run_id",
    "subject_id", "artifact_id", "anchor", "detail_sha256",
    "detail_redacted", "message",
}


def _diagnostic(snapshot_id: str, *, subject_id: str | None = None,
                artifact_id: str | None = None) -> Diagnostic:
    return Diagnostic(
        snapshot_id=snapshot_id,
        code="vendor.extract.failed",
        severity=DiagnosticSeverity.ERROR,
        stage="hls_ir",
        subject_id=subject_id,
        artifact_id=artifact_id,
        anchor=(SourceAnchor(
            artifact_id=artifact_id,
            start_line=17,
            start_column=9,
            ir_location=_PRIVATE_LOCATION,
        ) if artifact_id else None),
        message=f"parser copied {_PRIVATE_LOCATION}: int {_SENTINEL} = 1;",
        guidance=f"inspect private workspace {_SENTINEL}",
        metadata={
            "backend_trace": {"path": _PRIVATE_LOCATION},
            "plugin_detail": _SENTINEL,
        },
    )


def _assert_public(payload: object) -> None:
    serialized = json.dumps(payload, ensure_ascii=False)
    assert _SENTINEL not in serialized
    assert _PRIVATE_LOCATION not in serialized


def _assert_projection(item: dict) -> None:
    assert set(item) == _PUBLIC_KEYS
    assert item["code"] == "vendor.extract.failed"
    assert item["severity"] == "error"
    assert item["detail_redacted"] is True
    assert len(item["detail_sha256"]) == 64
    assert "guidance" not in item and "metadata" not in item
    _assert_public(item)


def test_successful_graph_public_surfaces_redact_diagnostic_details(
    tmp_path, capsys,
) -> None:
    (tmp_path / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    bundle = GraphBundle.create(
        tmp_path,
        minimal_manifest("test.diagnostic_projection", "diagnostic", "dut", "kernel.cpp"),
    )
    snapshot = bundle.snapshot()
    artifact = bundle.store.artifacts(snapshot.id)[0]
    entity = Entity(
        "hls.kernel", "dut", snapshot.id, qualified_name="dut", stage="hls_ir",
    )
    graph = CanonicalGraph(snapshot.id)
    graph.add_entity(entity)
    bundle.store.save_graph(graph)
    diagnostic = _diagnostic(
        snapshot.id, subject_id=entity.id, artifact_id=artifact.id,
    )
    bundle.store.add_diagnostics([diagnostic])

    # The trusted local ledger deliberately retains exact diagnostic details.
    raw = bundle.store.diagnostics(snapshot.id)[0]
    assert _SENTINEL in raw.message
    assert _SENTINEL in (raw.guidance or "")
    assert raw.metadata["plugin_detail"] == _SENTINEL

    core = CoreService(bundle, snapshot.id)
    rest = RestApplication(core)
    mcp = ReadOnlyMcpService(core)
    direct = public_diagnostic(raw)
    explore = core.explore(ExploreSpec(scope_id=entity.id)).to_dict()
    evidence = core.evidence(entity.id)
    rest_diagnostics = rest.dispatch("GET", "/api/v1/diagnostics").body
    rest_overview = rest.dispatch("GET", "/api/v1/overview").body
    mcp_health = mcp.health()
    mcp_evidence = mcp.evidence(entity.id)
    mcp_context = mcp.context(scope_id=entity.id)
    mcp_render_json = mcp.render(format="json")
    mcp_render_html = mcp.render(format="html", max_chars=5_000_000)
    direct_render = render(graph, format="json", diagnostics=[raw])
    sdk_render_path = Project(bundle).render(tmp_path / "sdk-render.html")
    sdk_render = sdk_render_path.read_text(encoding="utf-8")
    assert main([
        "--compact", "render", "--project", str(tmp_path),
        "cli-render.json", "--format", "json",
    ]) == 0
    cli_render_result = json.loads(capsys.readouterr().out)
    cli_render = (tmp_path / "cli-render.json").read_text(encoding="utf-8")
    assert cli_render_result["format"] == "json"

    for payload in (
        direct, explore, evidence, rest_diagnostics, rest_overview,
        mcp_health, mcp_evidence, mcp_context, mcp_render_json, mcp_render_html,
        direct_render, sdk_render, cli_render,
    ):
        _assert_public(payload)
    for item in (
        direct,
        explore["diagnostics"][0],
        evidence["diagnostics"][0],
        rest_diagnostics["items"][0],
        rest_overview["architecture"]["diagnostics"][0],
        mcp_health["diagnostics"][0],
        mcp_evidence["diagnostics"][0],
        mcp_context["diagnostics"][0],
    ):
        _assert_projection(item)
        assert item["detail_sha256"] == direct["detail_sha256"]
    assert direct["anchor"]["ir_location"].startswith("redacted.sha256:")


def test_failed_candidate_rest_mcp_and_cli_status_use_public_projection(
    tmp_path, capsys,
) -> None:
    (tmp_path / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    bundle = GraphBundle.create(
        tmp_path,
        minimal_manifest("test.failed_diagnostic", "failed", "dut", "kernel.cpp"),
    )
    snapshot = bundle.snapshot()
    diagnostic = _diagnostic(snapshot.id)
    bundle.store.add_diagnostics([diagnostic])

    rest = RestApplication(bundle)
    rest_status = rest.dispatch("GET", "/api/v1/status").body
    rest_diagnostics = rest.dispatch("GET", "/api/v1/diagnostics").body
    mcp_health = ReadOnlyMcpService(bundle).health()

    assert main([
        "--compact", "status", "--project", str(tmp_path),
    ]) == 0
    cli_status = json.loads(capsys.readouterr().out)

    for payload in rest_status, rest_diagnostics, mcp_health, cli_status:
        _assert_public(payload)
    for item in (
        rest_status["diagnostics"][0],
        rest_diagnostics["items"][0],
        mcp_health["diagnostics"][0],
        cli_status["diagnostics"][0],
    ):
        _assert_projection(item)

    # Public projection is not destructive: local troubleshooting still sees raw text.
    raw = bundle.store.diagnostics(snapshot.id)[0]
    assert _SENTINEL in raw.message
    assert raw.metadata["plugin_detail"] == _SENTINEL
