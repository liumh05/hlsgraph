"""Synthetic source and 2025.1 XML contract tests for the additive release."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

import pytest

from hlsgraph.bundle import GraphBundle
from hlsgraph.cli import main
from hlsgraph.extract import LibClangExtractor
from hlsgraph.manifest import minimal_manifest
from hlsgraph.mcp.service import ReadOnlyMcpService
from hlsgraph.model import TranslationUnit
from hlsgraph.render import render, to_render_data
from hlsgraph.sdk import Project
from tests.test_source_facts_v031 import STREAM

pytestmark = pytest.mark.skipif(not LibClangExtractor.available(), reason="requires libclang")

DATAFLOW = STREAM + """
void stage_a(hls::stream<int> &x) { x.write(1); }
void stage_b(hls::stream<int> &x, hls::stream<int> &y) { y.write(x.read()); }
void stage_c(hls::stream<int> &y, int out[8]) { out[0] = y.read(); }
void kern(int out[8]) {
#pragma HLS DATAFLOW
  hls::stream<int> x, y;
#pragma HLS STREAM variable=x depth=8
#pragma HLS STREAM variable=y depth=32
  stage_a(x);
  stage_b(x, y);
  stage_c(y, out);
}
"""
PLAIN = """
void stage_a(int out[8]) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=out complete
row_loop: for (int i=0; i<8; ++i) {
#pragma HLS PIPELINE II=1
  out[i] = i;
}
}
void kern(int out[8]) { stage_a(out); }
"""


def full_report():
    """Only generic, authored names; retain the vendor format's XML shape."""
    root = ET.Element("profile")
    ET.SubElement(ET.SubElement(root, "ReportVersion"), "Version").text = "2025.1"
    ET.SubElement(ET.SubElement(root, "UserAssignments"), "TopModelName").text = "kern"
    perf = ET.SubElement(ET.SubElement(root, "PerformanceEstimates"), "SummaryOfOverallLatency")
    ET.SubElement(perf, "Interval-max").text = "101"
    table = ET.SubElement(root, "ModuleInformation")
    for name, latency in [("kern", 100), ("stage_a", 30), ("stage_a_Pipeline_row_loop", 8),
                          ("duplicate", 7), ("unmapped", 2)]:
        module = ET.SubElement(table, "Module")
        ET.SubElement(module, "Name").text = name
        performance = ET.SubElement(module, "PerformanceEstimates")
        overall = ET.SubElement(performance, "SummaryOfOverallLatency")
        for tag, value in [("Worst-caseLatency", latency), ("Best-caseLatency", latency-1),
                           ("PipelineInitiationInterval", 2)]:
            ET.SubElement(overall, tag).text = str(value)
        resources = ET.SubElement(ET.SubElement(module, "AreaEstimates"), "Resources")
        for tag, value in [("LUT", 40), ("FF", 50), ("DSP", 2), ("BRAM_18K", 1), ("URAM", 0)]:
            ET.SubElement(resources, tag).text = str(value)
        if name == "stage_a":
            loop = ET.SubElement(ET.SubElement(performance, "SummaryOfLoopLatency"), "row_loop")
            for tag, value in [("Name", "row_loop"), ("Latency", "8"), ("PipelineII", "1"),
                               ("PipelineDepth", "3")]:
                ET.SubElement(loop, tag).text = value
    top = ET.SubElement(ET.SubElement(root, "RTLDesignHierarchy"), "TopModule", MAX_DEPTH="2")
    ET.SubElement(top, "ModuleName").text = "kern"

    def instance(parent, name, inst):
        children = parent.find("InstancesList")
        if children is None:
            children = ET.SubElement(parent, "InstancesList")
        child = ET.SubElement(children, "Instance")
        ET.SubElement(child, "InstName").text = inst
        ET.SubElement(child, "ModuleName").text = name
        return child

    child = instance(top, "stage_a", "stage_a_0")
    instance(top, "stage_a", "stage_a_1")
    instance(child, "stage_a_Pipeline_row_loop", "row_loop_0")
    instance(top, "duplicate", "duplicate_0")
    instance(top, "unmapped", "unmapped_0")
    return ET.tostring(root, encoding="unicode")


def project_at(tmp_path, source=PLAIN, *, report=False, header=None, other_unit=None):
    (tmp_path / "kernel.cpp").write_text(source)
    manifest = minimal_manifest("test.release.v032", "synthetic release test", "kern", "kernel.cpp")
    if header:
        (tmp_path / "globals.h").write_text(header)
    if other_unit:
        (tmp_path / "other.cpp").write_text(other_unit)
        manifest.build.translation_units.append(TranslationUnit("other.cpp"))
    if report:
        (tmp_path / "csynth_full.xml").write_text(full_report())
        manifest.artifact_paths.append({"path": "csynth_full.xml", "kind": "amd.vitis.csynth_full_xml",
                                        "metadata": {"fixture_authority": "synthetic"}})
    project = Project(GraphBundle.create(tmp_path, manifest))
    indexed = project.index()
    diagnostics = project.bundle.store.diagnostics(indexed.snapshot_id)
    assert indexed.success, [(d.code, d.message) for d in diagnostics]
    graph = project.service().graph()
    return project, graph, diagnostics


def test_source_dataflow_view(tmp_path):
    project, graph, diagnostics = project_at(tmp_path, DATAFLOW)
    data = to_render_data(graph, [], diagnostics, view="source")
    assert data["view"] == data["meta"]["view"] == "source_structure"
    assert data["notice"] == "Source structure evidence; not hardware topology."
    region, = [n for n in data["nodes"] if n["type"] == "source.dataflow_region"]
    processes = [n for n in data["nodes"] if n["type"] == "source.dataflow_process"]
    assert len(processes) == 3 and all(n["parent"] == region["id"] for n in processes)
    channels = [e for e in data["edges"] if e["type"] == "CHANNEL"]
    assert {e["depth"] for e in channels} == {8, 32}
    assert all(not e["depth_assumed"] and e["via"] == "stream" and e["variable"] for e in channels)
    sites = {n["attrs"]["call_site_id"] for n in processes}
    assert len([e for e in data["edges"] if e["type"] == "BINDS"
                and e["attrs"]["call_site_id"] in sites]) == 5
    # Only the three process -> callee calls: the region already shows kern's calls, and
    # hls::stream methods are library helpers, not design structure.
    assert len([e for e in data["edges"] if e["type"] == "CALLS"]) == 3
    assert not {"read", "write"} & {n["name"] for n in data["nodes"]}
    assert any("DATAFLOW" in n["badges"] for n in data["nodes"])
    assert all(n["metrics"]["latency"] is None and not n["is_bottleneck"] for n in data["nodes"])
    assert to_render_data(graph, [], []) == to_render_data(graph, [], [], view="auto")
    assert not to_render_data(graph, [], [], view="hardware")["nodes"]
    html = render(graph, view="source")
    assert "http" not in html.lower() and "<script src=" not in html.lower()
    for marker in ("'elk.direction':'RIGHT'", "'elk.algorithm':'layered'", "byId[parent].children",
                   "line-style':'dotted'", "Math.log2", "DATA.notice", "selector:':parent'"):
        assert marker in html
    for format, marker in [("dot", "subgraph cluster_"), ("mermaid", "subgraph"), ("svg", 'rx="12"')]:
        output = render(graph, view="source", format=format)
        assert marker in output and data["notice"] in output
    assert graph.graph_hash == project.service().graph().graph_hash


def test_plain_source_contains_binds_and_badges(tmp_path):
    _, graph, _ = project_at(tmp_path)
    data = to_render_data(graph, [], [], view="source")
    assert {e["type"] for e in data["edges"]} == {"CONTAINS", "BINDS", "CALLS"}
    names = {n["id"]: n["name"] for n in data["nodes"]}
    assert [(names[e["source"]], names[e["target"]]) for e in data["edges"] if e["type"] == "CALLS"] == [("kern", "stage_a")]
    # The bound argument hangs off each function, so the binding reads kern.out -> stage_a.out.
    owned = {(names[e["source"]], names[e["target"]]) for e in data["edges"] if e["type"] == "CONTAINS"}
    assert {("kern", "out"), ("stage_a", "out")} <= owned
    badges = {b for n in data["nodes"] for b in n["badges"]}
    assert badges >= {"PIPELINE II=1", "ARRAY_PARTITION complete", "INLINE off"} and "INLINE" not in badges
    binding, = [e for e in data["edges"] if e["type"] == "BINDS"]
    assert binding["arg_index"] == 0 and binding["passing"] == "array"
    with pytest.raises(ValueError, match="view"):
        render(graph, view="invalid")


def test_empty_region_and_notice_survive_static_export(tmp_path):
    _, graph, _ = project_at(tmp_path, "void kern() {\n#pragma HLS DATAFLOW\n}\n")
    data = to_render_data(graph, [], [], view="source")
    region, = [node for node in data["nodes"] if node["type"] == "source.dataflow_region"]
    for format in ("dot", "mermaid", "svg"):
        output = render(graph, view="source", format=format)
        assert region["name"] in output and data["notice"] in output
    svg = ET.fromstring(render(graph, view="source", format="svg"))
    assert float(svg.attrib["width"]) >= 480


@pytest.mark.skipif(shutil.which("node") is None, reason="requires Node for offline layout check")
def test_source_compound_layout_runs_offline(tmp_path):
    _, graph, _ = project_at(tmp_path, DATAFLOW)
    html = render(graph, view="source")
    html_path = tmp_path / "source.html"
    html_path.write_text(html)
    # Exercise the shipped layout with the shipped JS libraries. Any ELK
    # fallback is a failure: compound children must receive finite positions.
    script = r"""
(async function () {
    const fs = require('fs');
    const cytoscape = require('./src/hlsgraph/render/vendor/cytoscape.min.js');
    const ELK = require('./src/hlsgraph/render/vendor/elk.bundled.js');
    const html = fs.readFileSync(process.argv[1], 'utf8');
    const DATA = JSON.parse(html.match(/const DATA=(.*);/)[1]);
    const cy = cytoscape({headless:true, elements:{
        nodes:DATA.nodes.map(n=>({data:n})), edges:DATA.edges.map(e=>({data:e}))}});
    const NODE_W=132, NODE_H=48, window={};
    console.warn = error => {throw error};
    const code = html.slice(html.indexOf('async function layout(){'),
                            html.indexOf('const notice=document.getElementById'));
    await eval(code + ';layout()');
    const parents = cy.nodes(':parent');
    if (!window.hlsgraphRenderReady || parents.length !== 1
        || parents.children().length !== 3
        || !cy.nodes().every(n => Number.isFinite(n.position().x)
                                && Number.isFinite(n.position().y))) {
        throw new Error('invalid compound layout');
    }
    cy.destroy();
})().catch(error => {console.error(String(error)); process.exitCode=1});
"""
    completed = subprocess.run([shutil.which("node"), "-e", script, str(html_path)],
                               cwd=Path(__file__).resolve().parents[1],
                               capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr


def test_full_report_hierarchy_metrics_mapping_and_bottleneck(tmp_path):
    source = PLAIN + "\nnamespace left {void duplicate() {}}\nnamespace right {void duplicate() {}}\n"
    project, graph, diagnostics = project_at(tmp_path, source, report=True)
    modules = graph.by_kind("hls.module")
    assert len(modules) == 5 and all(e.stage == "schedule" and str(e.authority) == "synthetic" for e in modules)
    observations = project.bundle.store.observations(graph.snapshot_id)
    data = to_render_data(graph, observations, diagnostics, view="hardware")
    assert data["meta"]["view"] == "architecture_hierarchy"
    assert len(data["edges"]) == 6
    assert all(n["stage"] == "schedule" for n in data["nodes"])
    assert {e["attrs"].get("instance_name") for e in data["edges"]} >= {"stage_a_0", "stage_a_1", "row_loop_0"}
    bottleneck, = [n for n in data["nodes"] if n["is_bottleneck"]]
    # The top's latency contains its children, so the slowest measured child is the bottleneck.
    assert bottleneck["name"] == "stage_a" and bottleneck["metrics"]["latency"] == 30
    assert bottleneck["metrics"]["achieved_II"] == 2
    assert {d.code for d in diagnostics} >= {"mapping.ambiguous_report_scope", "mapping.unresolved_report_scope"}
    mapped = [r for r in graph.relations.values() if r.kind == "cross.maps_to"]
    assert len(mapped) == 4
    top = next(e for e in modules if e.name == "kern")
    values = {o.predicate: o.value for o in observations if o.subject_id == top.id}
    assert values == {"qor.latency_cycles": 100, "qor.latency_best_cycles": 99,
                      "qor.interval_max_cycles": 101, "qor.achieved_ii": 2,
                      "resource.lut": 40, "resource.ff": 50, "resource.dsp": 2,
                      "resource.bram_18k": 1, "resource.uram": 0}
    assert all(o.source.parser_version == "2" for o in observations if o.artifact_id == top.anchors[0].artifact_id)
    source_data = to_render_data(graph, observations, diagnostics, view="source")
    source_top = next(n for n in source_data["nodes"] if n["type"] == "hls.kernel")
    assert source_top["metrics"]["latency"] == 100
    assert source_top["metric_evidence"]["latency"]["stage"] == "schedule"
    source_loop = next(n for n in source_data["nodes"] if n["type"] == "hls.loop")
    # Two equal-rank explicit mappings disagree on II: no guessed display value.
    assert source_loop["metrics"]["achieved_II"] is None
    assert "qor.achieved_ii" in source_loop["display_conflicts"]


def test_cli_and_mcp_view(tmp_path, capsys):
    project, graph, _ = project_at(tmp_path)
    service = ReadOnlyMcpService(project.bundle)
    result = service.render(format="json", view="source")
    assert json.loads(result["content"])["view"] == "source_structure"
    assert main(["render", "--project", str(tmp_path), "--view", "source", "--format", "json", "source.json"]) == 0
    assert json.loads((tmp_path / "source.json").read_text())["view"] == "source_structure"
    capsys.readouterr()
    assert not to_render_data(graph, [], [], view="hardware")["edges"]


@pytest.mark.parametrize("declaration", ["int table[8];", "static int table[8];", "const int table[8] = {};", "static const int table[8] = {};"])
def test_global_partition_binding(tmp_path, declaration):
    _, graph, diagnostics = project_at(tmp_path, declaration + "\nvoid kern() {\n#pragma HLS ARRAY_PARTITION variable=table complete\n}\n")
    memory, = graph.by_kind("hls.memory")
    assert memory.attrs["storage"] == "global"
    assert memory.qualified_name == "kernel.cpp::global::table@1"
    owner, = [r for r in graph.relations.values() if r.kind == "hls.contains" and r.dst == memory.id]
    assert graph.entities[owner.src].kind == "source.file"
    directive, = graph.by_kind("hls.directive")
    assert directive.attrs["variable_id"] == memory.id
    assert not any(d.code == "directive.unresolved_scope" for d in diagnostics)


def test_namespace_header_global_and_unrelated_unit(tmp_path):
    header = "namespace storage { static int table[8]; const int size = 8; }\n"
    source = '#include "globals.h"\nnamespace storage {\nvoid stage_a() {\n#pragma HLS ARRAY_PARTITION variable=table complete\n}\n}\nvoid kern() { storage::stage_a(); }\n'
    _, graph, diagnostics = project_at(tmp_path, source, header=header, other_unit="static int table[8];\n")
    directive, = graph.by_kind("hls.directive")
    target = graph.entities[directive.attrs["variable_id"]]
    assert target.qualified_name.startswith("globals.h::global::storage::table")
    scalar, = graph.by_kind("source.variable")
    assert scalar.attrs["storage"] == "global"
    assert not any(d.code == "directive.unresolved_scope" for d in diagnostics)


@pytest.mark.parametrize("source", [
    "namespace other {int table[8];}\nvoid kern() {\n#pragma HLS ARRAY_PARTITION variable=table complete\n}\n",
    "void kern() {\n#pragma HLS ARRAY_PARTITION variable=table complete\n}\nint table[8];\n",
    "int table[8];\nvoid kern() { int table=1;\n#pragma HLS ARRAY_PARTITION variable=table complete\n}\n",
])
def test_global_invisible_or_ambiguous_stays_unresolved(tmp_path, source):
    _, graph, diagnostics = project_at(tmp_path, source)
    assert not [r for r in graph.relations.values() if r.kind == "hls.annotates"]
    assert any(d.code == "directive.unresolved_scope" for d in diagnostics)
