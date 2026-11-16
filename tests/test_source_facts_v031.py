from __future__ import annotations

import json
from pathlib import Path

import pytest

from hlsgraph.bundle import GraphBundle
from hlsgraph.extract import LibClangExtractor
from hlsgraph.manifest import minimal_manifest
from hlsgraph.model import TranslationUnit
from hlsgraph.sdk import Project

PYTHON_REQUIRED = pytest.mark.skipif(not LibClangExtractor.available(), reason="requires libclang")
pytestmark = PYTHON_REQUIRED

STREAM = """
namespace hls {
template<class T> class stream {
public:
  T read() { return T(); }
  void write(T) {}
  bool empty() { return false; }
  bool full() { return false; }
  bool read_nb(T &) { return true; }
  bool write_nb(T) { return true; }
  stream &operator>>(T &) { return *this; }
  stream &operator<<(T) { return *this; }
};
}
"""


def extract(tmp_path, source, *, testbench=None, defines=None):
    (tmp_path / "kernel.cpp").write_text(source)
    manifest = minimal_manifest("test.source.facts", "source facts", "dut", "kernel.cpp")
    manifest.build.defines = defines or {}
    if testbench:
        (tmp_path / "tb.cpp").write_text(testbench)
        manifest.build.testbench_files = ["tb.cpp"]
        manifest.build.translation_units.append(TranslationUnit("tb.cpp"))
    project = Project(GraphBundle.create(tmp_path, manifest))
    indexed = project.index()
    assert indexed.success, [(d.code, d.message) for d in project.bundle.store.diagnostics(indexed.snapshot_id)]
    graph = project.service().graph()
    diagnostics = project.bundle.store.diagnostics(indexed.snapshot_id)
    return graph, diagnostics, project


def relations(graph, kind):
    return [r for r in graph.relations.values() if r.kind == kind]


def test_binding_partition_chain_and_passing(tmp_path):
    graph, diagnostics, _ = extract(tmp_path, """
void sub(int output[8], int &ref, int *ptr, int value) {
#pragma HLS ARRAY_PARTITION variable=output complete
  output[0] = value;
}
void dut(int top[8], int &r, int *p, int v) {
  sub(top, r, p, v);
  sub(top, r, p, v);
}
""")
    bindings = relations(graph, "source.binds_argument")
    assert len(bindings) == 8
    assert {r.attrs["passing"] for r in bindings} == {"array", "reference", "pointer", "value"}
    assert {r.attrs["call_order"] for r in bindings} == {0, 1}
    assert len({r.attrs["call_site_id"] for r in bindings}) == 2
    partition = next(e for e in graph.entities.values() if e.name == "ARRAY_PARTITION")
    output = next(e for e in graph.entities.values() if e.name == "output")
    assert partition.attrs["port_id"] == output.id
    assert any(graph.entities[r.src].name == "top" and r.dst == output.id for r in bindings)
    assert not any(d.code == "mapping.ambiguous_argument" for d in diagnostics)
    for binding in bindings:
        assert binding.attrs["hardware_instance"] is False
        assert binding.attrs["ml_input_evidence"] is True
        assert binding.stage == "ast" and str(binding.authority) == "static_fact"


def test_elements_expressions_ambiguity_and_shadowing(tmp_path):
    graph, diagnostics, _ = extract(tmp_path, """
void sub(int value) {}
void dut(int a[8], int b, int i) {
  sub(a[i]);
  sub(b + 1);
  sub(b + i);
  { int b = 2; sub(b); }
  sub(b);
}
""")
    bindings = sorted(relations(graph, "source.binds_argument"), key=lambda r: r.attrs["call_order"])
    assert [r.attrs["expression"] for r in bindings] == ["element", "expression", "entity", "entity"]
    assert graph.entities[bindings[0].src].name == "a"
    assert graph.entities[bindings[-2].src].kind == "source.variable"
    assert graph.entities[bindings[-1].src].kind == "hls.port"
    assert any(d.code == "mapping.ambiguous_argument" for d in diagnostics)


CANONICAL = STREAM + """
void first(hls::stream<int> &x, hls::stream<int> &bypass) { x.write(1); bypass << 2; }
void second(hls::stream<int> &x, hls::stream<int> &y) { y.write(x.read()); }
void third(hls::stream<int> &y, hls::stream<int> &bypass) { int v; bypass >> v; v = y.read(); }
void dut() {
#pragma HLS DATAFLOW
  hls::stream<int> x, y, bypass;
#pragma HLS STREAM variable=x depth=7
  first(x, bypass);
  second(x, y);
  third(y, bypass);
}
"""


def test_canonical_channels_bypass_depth_and_determinism(tmp_path):
    graph, diagnostics, project = extract(tmp_path, CANONICAL)
    assert len(graph.by_kind("source.dataflow_region")) == 1
    assert len(graph.by_kind("source.dataflow_process")) == 3
    channels = relations(graph, "source.dataflow_channel")
    assert len(channels) == 3
    assert {(graph.entities[r.src].name, graph.entities[r.dst].name) for r in channels} == {
        ("first", "second"), ("second", "third"), ("first", "third")}
    for channel in channels:
        name = graph.entities[channel.attrs["variable_id"]].name
        assert channel.attrs["depth"] == (7 if name == "x" else None)
        assert channel.attrs["depth_assumed"] == (name != "x")
        assert graph.entities[channel.attrs["producer_param"]].kind == "hls.port"
        assert graph.entities[channel.attrs["consumer_param"]].kind == "hls.port"
    assert not any(d.code.startswith("dataflow.") for d in diagnostics)
    initial = graph.graph_hash
    project.index()
    assert project.service().graph().graph_hash == initial
    assert not {e.kind for e in graph.entities.values()} & {"hls.process", "hls.region", "hls.dataflow_region"}
    assert not relations(graph, "hls.streams_to")


def test_noncanonical_multiple_endpoints_conflict(tmp_path):
    graph, diagnostics, _ = extract(tmp_path, STREAM + """
void put(hls::stream<int> &x) { x.write(1); }
void get(hls::stream<int> &x) { x.read(); }
void value(int x) {}
void dut(int a[8]) {
#pragma HLS DATAFLOW
#pragma HLS PIPELINE
  hls::stream<int> x;
  put(x);
  put(x);
  get(x);
  get(x);
  if (a[0]) get(x);
  value(a[0]);
}
""")
    codes = {d.code for d in diagnostics}
    assert {"dataflow.noncanonical_statement", "dataflow.noncanonical_argument",
            "dataflow.multiple_producers", "dataflow.multiple_consumers",
            "dataflow.pipeline_conflict"} <= codes
    assert len(graph.by_kind("source.dataflow_process")) == 5
    assert len(relations(graph, "source.dataflow_channel")) == 4
    assert all(str(d.severity) == "warning" for d in diagnostics if d.code.startswith("dataflow."))


def test_loop_region_and_array_channel(tmp_path):
    graph, diagnostics, _ = extract(tmp_path, """
void put(int a[4]) { a[0] = 1; }
void get(int a[4]) { int v = a[0]; }
void dut() {
  for (int i = 0; i < 4; ++i) {
#pragma HLS DATAFLOW
    int a[4];
    put(a);
    get(a);
  }
}
""")
    region, = graph.by_kind("source.dataflow_region")
    assert region.attrs["scope_kind"] == "loop"
    channel, = relations(graph, "source.dataflow_channel")
    assert channel.attrs["via"] == "array"
    assert not any(d.code.startswith("dataflow.") for d in diagnostics)


def test_stream_access_nested_loops(tmp_path):
    graph, _, _ = extract(tmp_path, STREAM + """
void dut(hls::stream<int> &input, hls::stream<int> &output) {
  input.empty();
  for (int i = 0; i < 4; ++i) {
    input.read(); input.read();
    for (int j = 0; j < 3; ++j) {
      output.write(input.read()); output << 2;
    }
  }
}
""")
    rows = relations(graph, "source.stream_access")
    assert len(rows) == 4
    assert sorted((graph.entities[r.src].kind, graph.entities[r.dst].name,
                   r.attrs["direction"], r.attrs["count_per_iteration"]) for r in rows) == [
        ("hls.kernel", "input", "read", 1),
        ("hls.loop", "input", "read", 1),
        ("hls.loop", "input", "read", 2),
        ("hls.loop", "output", "write", 2)]


def test_unknown_direction_and_conditional_access(tmp_path):
    graph, diagnostics, _ = extract(tmp_path, STREAM + """
void put(hls::stream<int> &x) { x.write(1); }
void unknown(hls::stream<int> &x) { x.empty(); }
void dut(int enabled) {
#pragma HLS DATAFLOW
  hls::stream<int> x;
  put(x);
  unknown(x);
  if (enabled) x.read();
}
""")
    assert not relations(graph, "source.dataflow_channel")
    assert "dataflow.unknown_direction" in {d.code for d in diagnostics}
    assert any(str(r.completeness) == "partial" for r in relations(graph, "source.stream_access"))


@pytest.mark.parametrize(("code", "declarations", "body"), [
    ("dynamic_allocation", "", "int *p = new int; delete p;"),
    ("dynamic_allocation", "extern void *malloc(unsigned long); extern void free(void*);", "free(malloc(4));"),
    ("dynamic_allocation", "extern void *calloc(unsigned long,unsigned long);", "void *p = calloc(1,4);"),
    ("dynamic_allocation", "extern void *realloc(void*,unsigned long);", "void *p = realloc(0,4);"),
    ("recursion", "void rec() { rec(); }", "rec();"),
    ("virtual_function", "struct A { virtual void f() {} };", ""),
    ("virtual_function", "struct A { virtual void f() = 0; };", ""),
    ("function_pointer", "void f() {}", "void (*p)() = f; p();"),
    ("function_pointer", "void f(void (*p)()) { p(); }", ""),
    ("pointer_to_pointer", "", "int **p = 0;"),
    ("long_double", "", "long double v = 0;"),
    ("system_call", "extern int printf(const char *, ...);", 'printf("x");'),
    ("system_call", "namespace std { int cout; }", "std::cout = 1;"),
])
def test_synthesis_positive(tmp_path, code, declarations, body):
    graph, diagnostics, _ = extract(tmp_path, declarations + "\nvoid dut() { " + body + " }\n")
    matches = [d for d in diagnostics if d.code == f"synth.{code}"]
    assert matches, [(d.code, d.message) for d in diagnostics]
    assert all(d.subject_id in graph.entities and d.anchor and d.anchor.start_line
               and str(d.severity) == "warning" for d in matches)


def test_synthesis_guard_testbench_main_and_safe_types(tmp_path):
    graph, diagnostics, _ = extract(tmp_path, """
extern int printf(const char *, ...);
void safe(int *p, double v) { p[0] = 1; }
void dut(int *p) {
#ifndef __SYNTHESIS__
  printf("simulation");
  int *q = new int; delete q;
#endif
  safe(p, 0.0);
}
int main() { printf("main"); int *p = new int; delete p; }
""", testbench="void tb() { int *p = new int; delete p; }\n")
    assert not [d for d in diagnostics if d.code.startswith("synth.")]


def test_synthesis_active_macro_and_mutual_recursion(tmp_path):
    _, diagnostics, _ = extract(tmp_path, """
extern int printf(const char *, ...);
void b();
void a() { b(); }
void b() { a(); }
void dut() {
#ifdef __SYNTHESIS__
  printf("synthesis");
#endif
  a();
}
""")
    assert len([d for d in diagnostics if d.code == "synth.recursion"]) == 2
    assert len([d for d in diagnostics if d.code == "synth.system_call"]) == 1


def test_no_new_facts_without_calls_or_dataflow(tmp_path):
    graph, diagnostics, _ = extract(tmp_path, "void dut(int a[4]) { a[0] = 1; }\n")
    assert not [r for r in graph.relations.values() if r.kind.startswith("source.")]
    assert not graph.by_kind("source.dataflow_region")
    assert not [d for d in diagnostics if d.code.startswith(("synth.", "dataflow."))]


def test_export_health_and_evidence_queries(tmp_path):
    from hlsgraph.mcp.service import ReadOnlyMcpService
    from hlsgraph.model import DatasetManifest
    from hlsgraph.query import ExploreSpec
    from hlsgraph.version import FEATURE_SCHEMA_VERSION

    graph, _, project = extract(tmp_path, CANONICAL.replace(
        "#pragma HLS DATAFLOW", "#pragma HLS DATAFLOW\n#pragma HLS PIPELINE"))
    service = ReadOnlyMcpService(project)
    assert "dataflow.pipeline_conflict" in {d["code"] for d in service.health()["diagnostics"]}
    region, = graph.by_kind("source.dataflow_region")
    reachable = service.traverse(region.id, depth=2)
    assert "source.dataflow_channel" in {r["kind"] for r in reachable["relations"]}
    evidence = project.service().explore(ExploreSpec(scope_id=region.id, view="evidence", depth=2))
    assert "source.dataflow_process" in {e["kind"] for e in evidence.entities}
    architecture = project.service()._view_graph(graph, "architecture")
    assert not any(r.kind.startswith("source.") for r in architecture.relations.values())
    assert not any(e.kind.startswith("source.") for e in architecture.entities.values())
    dataset = DatasetManifest(
        dataset_id="dataset.source.facts", feature_schema_version=FEATURE_SCHEMA_VERSION,
        snapshot_ids=[graph.snapshot_id], feature_attribute_allowlist=[
            "hardware_instance", "ml_input_evidence", "call_site_id", "arg_index",
            "direction", "count_per_iteration", "via", "depth", "depth_assumed",
            "producer_param", "consumer_param", "expression", "passing",
        ],
    )
    output = tmp_path / "export"
    project.export_dataset(output, dataset)
    nodes = [json.loads(line) for line in (output / "nodes.jsonl").read_text().splitlines()]
    edges = [json.loads(line) for line in (output / "edges.jsonl").read_text().splitlines()]
    assert {"source.dataflow_region", "source.dataflow_process"} <= {n["kind"] for n in nodes}
    assert {"source.binds_argument", "source.dataflow_channel", "source.stream_access", "source.contains"} <= {e["kind"] for e in edges}
    assert all(e["features"]["hardware_instance"] is False for e in edges if e["kind"].startswith("source."))


def test_unresolved_and_expression_region_arguments(tmp_path):
    graph, diagnostics, _ = extract(tmp_path, """
extern void external(int *p);
void value(int v) {}
void overloaded(int v) {}
void overloaded(float v) {}
void dut(int a, int b, int *p) {
#pragma HLS DATAFLOW
  value(a + b);
  external(p);
  overloaded(a);
}
""")
    assert not relations(graph, "source.binds_argument")
    assert len(graph.by_kind("source.dataflow_process")) == 1
    assert {"mapping.ambiguous_argument", "mapping.ambiguous_call",
            "dataflow.noncanonical_argument", "dataflow.unresolved_process"} <= {d.code for d in diagnostics}


def test_escaped_parameter_does_not_guess_direction(tmp_path):
    graph, diagnostics, _ = extract(tmp_path, STREAM + """
extern void opaque(hls::stream<int> &s);
void put(hls::stream<int> &s) { s.write(1); opaque(s); }
void get(hls::stream<int> &s) { s.read(); }
void dut() {
#pragma HLS DATAFLOW
  hls::stream<int> s;
  put(s);
  get(s);
}
""")
    assert not relations(graph, "source.dataflow_channel")
    assert "dataflow.unknown_direction" in {d.code for d in diagnostics}


@pytest.mark.parametrize("symbol,decl,call", [
    ("fprintf", "extern int fprintf(void *, const char *, ...);", 'fprintf(0,"x");'),
    ("puts", "extern int puts(const char *);", 'puts("x");'),
    ("fopen", "extern void *fopen(const char *,const char *);", 'fopen("x","r");'),
    ("time", "extern long time(long *);", 'time(0);'),
    ("sleep", "extern unsigned sleep(unsigned);", 'sleep(1);'),
    ("exit", "extern void exit(int);", 'exit(0);'),
    ("cerr", "namespace std { int cerr; }", 'std::cerr = 0;'),
    ("cin", "namespace std { int cin; }", '(void)std::cin;'),
])
def test_system_calls_and_synthesis_guards(tmp_path, symbol, decl, call):
    _, diagnostics, _ = extract(tmp_path, decl + "\nvoid dut() {\n" + call + "\n#ifndef __SYNTHESIS__\n" + call + "\n#endif\n}\n")
    matches = [d for d in diagnostics if d.code == "synth.system_call"]
    assert len(matches) == 1
    assert matches[0].anchor.start_line == 3


def test_synthesis_negative_all_constructs_in_guard(tmp_path):
    _, diagnostics, _ = extract(tmp_path, """
#ifndef __SYNTHESIS__
extern int printf(const char *, ...);
struct A { virtual void f() {} };
void rec() { rec(); }
void nonsynth(void (*fp)(), int **pp) { long double x; fp(); printf("x"); }
#endif
void dut() {
#if !defined(__SYNTHESIS__)
  int *p = new int; delete p; rec();
#endif
}
""")
    assert not [d for d in diagnostics if d.code.startswith("synth.")]


def test_cross_translation_unit_binding(tmp_path):
    (tmp_path / "kernel.cpp").write_text("void helper(int a[4]);\nvoid dut(int a[4]) { helper(a); }\n")
    (tmp_path / "helper.cpp").write_text("void helper(int output[4]) {\n#pragma HLS ARRAY_PARTITION variable=output complete\n}\n")
    manifest = minimal_manifest("test.source.cross_tu", "cross TU", "dut", "kernel.cpp")
    manifest.build.translation_units.append(TranslationUnit("helper.cpp"))
    project = Project(GraphBundle.create(tmp_path, manifest))
    assert project.index().success
    graph = project.service().graph()
    binding, = relations(graph, "source.binds_argument")
    assert graph.entities[binding.src].name == "a"
    assert graph.entities[binding.dst].name == "output"


def test_macro_call_span_collision_is_not_guessed(tmp_path):
    graph, diagnostics, _ = extract(tmp_path, """
#define BOTH(x) put(x); get(x)
void put(int x) {}
void get(int x) {}
void dut(int a) { BOTH(a); }
""")
    assert len(relations(graph, "software.calls")) == 2
    assert not relations(graph, "source.binds_argument")
    assert "mapping.ambiguous_call_site" in {d.code for d in diagnostics}


def test_short_circuit_and_header_counts_are_partial(tmp_path):
    graph, diagnostics, _ = extract(tmp_path, STREAM + """
void dut(hls::stream<int> &s, int v) {
  v && s.read();
  for (int i = s.read(); i < 4; ++i) { s.write(i); }
}
""")
    reads = [r for r in relations(graph, "source.stream_access") if r.attrs["direction"] == "read"]
    assert len(reads) == 2 and all(str(r.completeness) == "partial" for r in reads)
    assert "mapping.unknown_stream_count" in {d.code for d in diagnostics}


def test_existing_kind_payloads_and_ids_are_unchanged(tmp_path, monkeypatch):
    from hlsgraph.extract import ExtractionContext
    from hlsgraph.extract.source_facts import SourceFacts
    from hlsgraph.model import json_ready

    graph, _, project = extract(tmp_path, CANONICAL)
    snapshot = project.bundle.store.snapshot(graph.snapshot_id)
    context = ExtractionContext(
        project_root=tmp_path, manifest=project.bundle.manifest, snapshot=snapshot,
        artifacts={a.id: a for a in project.bundle.store.artifacts(snapshot.id)},
    )
    with_facts = LibClangExtractor().extract(context).graph
    monkeypatch.setattr(SourceFacts, "emit", lambda *args: None)
    monkeypatch.setattr(SourceFacts, "synthesizability", lambda *args: None)
    without_facts = LibClangExtractor().extract(context).graph
    assert {k: json_ready(e) for k, e in with_facts.entities.items()
            if e.kind not in {"source.dataflow_region", "source.dataflow_process"}} == {
                k: json_ready(e) for k, e in without_facts.entities.items()}
    assert {k: json_ready(r) for k, r in with_facts.relations.items()
            if not r.kind.startswith("source.")} == {
                k: json_ready(r) for k, r in without_facts.relations.items()}


def test_translation_unit_repetition_does_not_double_access_counts(tmp_path):
    (tmp_path / "common.h").write_text(STREAM + "\ninline void helper(hls::stream<int> &s) { s.read(); }\n")
    (tmp_path / "kernel.cpp").write_text('#include "common.h"\nvoid dut(hls::stream<int> &s) { helper(s); }\n')
    (tmp_path / "other.cpp").write_text('#include "common.h"\nvoid other() {}\n')
    manifest = minimal_manifest("test.source.repeat", "repeat TU", "dut", "kernel.cpp")
    manifest.build.translation_units.append(TranslationUnit("other.cpp"))
    project = Project(GraphBundle.create(tmp_path, manifest))
    assert project.index().success
    graph = project.service().graph()
    access, = relations(graph, "source.stream_access")
    assert access.attrs["count_per_iteration"] == 1
    assert not any(d.code == "mapping.ambiguous_call_site"
                   for d in project.bundle.store.diagnostics(graph.snapshot_id))
