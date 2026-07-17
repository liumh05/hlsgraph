from __future__ import annotations

from hlsgraph.bundle import GraphBundle
from hlsgraph.extract import ExtractionContext, VivadoReportExtractor
from hlsgraph.graph import CanonicalGraph
from hlsgraph.manifest import minimal_manifest
from hlsgraph.model import Entity, Stage


def _context(tmp_path, reports, *, capacities=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    manifest = minimal_manifest(
        "test.vivado_contract", "vivado", "dut", "kernel.cpp", part="xck26-test",
    )
    manifest.target.capacities = dict(capacities or {})
    for index, (kind, text, metadata) in enumerate(reports):
        path = f"report-{index}.txt"
        (tmp_path / path).write_text(text, encoding="utf-8")
        manifest.artifact_paths.append({
            "path": path, "kind": kind, "role": "implementation_report",
            "access": "project", "metadata": metadata,
        })
    bundle = GraphBundle.create(tmp_path, manifest)
    snapshot = bundle.snapshot()
    graph = CanonicalGraph(snapshot.id)
    graph.add_entity(Entity(
        "hls.kernel", "dut", snapshot.id, qualified_name="dut", stage="ast",
    ))
    artifacts = bundle.store.artifacts(snapshot.id)
    context = ExtractionContext(
        project_root=tmp_path, manifest=bundle.manifest, snapshot=snapshot,
        artifacts={item.id: item for item in artifacts},
        options={"existing_graph": graph},
    )
    return context


def _scope(**overrides):
    value = {"kind": "kernel", "top": "dut", "instance": "dut",
             "part": "xck26-test", "clock": "all"}
    value.update(overrides)
    return value


def test_generic_vivado_report_requires_stage_and_only_post_route_can_gate(tmp_path):
    missing = VivadoReportExtractor().extract(_context(tmp_path / "missing", [(
        "amd.vivado.timing_summary", "WNS: 0.25\nTNS: 0\n",
        {"scope": _scope()},
    )]))
    assert any(item.code == "vivado.report_parse_error" and "metadata.stage" in item.message
               for item in missing.diagnostics)
    assert not any(item.predicate == "timing.wns_ns" for item in missing.observations)

    post_synth = VivadoReportExtractor().extract(_context(tmp_path / "post-synth", [(
        "amd.vivado.timing_summary", "WNS: 0.25\nTNS: 0\n",
        {"stage": "post_synth", "scope": _scope()},
    )]))
    assert any(item.predicate == "timing.wns_ns" and item.stage == Stage.POST_SYNTH.value
               for item in post_synth.observations)
    assert not any(item.predicate == "gate.post_route_timing"
                   for item in post_synth.derivations)


def test_vivado_scope_must_match_top_and_part_before_design_gate(tmp_path):
    result = VivadoReportExtractor().extract(_context(tmp_path, [(
        "amd.vivado.post_route_timing", "WNS: 0.25\nTNS: 0\n",
        {"scope": _scope(part="wrong-part")},
    )]))
    assert any(item.code == "vivado.report_parse_error" and "does not match" in item.message
               for item in result.diagnostics)
    assert not result.observations
    assert not any(item.predicate == "gate.post_route_timing" for item in result.derivations)


def test_unscoped_report_remains_artifact_evidence_and_cannot_gate(tmp_path):
    result = VivadoReportExtractor().extract(_context(tmp_path, [(
        "amd.vivado.post_route_timing", "WNS: 0.25\nTNS: 0\n", {},
    )]))
    assert any(item.code == "vivado.report_scope_unbound" for item in result.diagnostics)
    assert len(result.observations) == 2
    assert all(item.subject_id == item.artifact_id for item in result.observations)
    assert not any(item.predicate == "gate.post_route_timing" for item in result.derivations)


def test_resource_gate_requires_complete_single_artifact_capacity_set(tmp_path):
    incomplete = VivadoReportExtractor().extract(_context(
        tmp_path / "incomplete",
        [("amd.vivado.post_route_utilization", "LUT: 10\nFF: 20\n",
          {"scope": _scope()})],
        capacities={"lut": 100, "ff": 100, "dsp": 10},
    ))
    assert any(item.code == "gate.resource_capacity_incomplete"
               for item in incomplete.diagnostics)
    assert not any(item.predicate == "gate.resource_fits" for item in incomplete.derivations)

    report = "LUT: 10\nFF: 20\nDSP: 1\n"
    ambiguous = VivadoReportExtractor().extract(_context(
        tmp_path / "ambiguous",
        [
            ("amd.vivado.post_route_utilization", report, {"scope": _scope()}),
            ("amd.vivado.utilization", report,
             {"stage": "post_route", "scope": _scope()}),
        ],
        capacities={"lut": 100, "ff": 100, "dsp": 10},
    ))
    assert any(item.code == "gate.resource_source_ambiguous"
               for item in ambiguous.diagnostics)
    assert not any(item.predicate == "gate.resource_fits" for item in ambiguous.derivations)

    complete = VivadoReportExtractor().extract(_context(
        tmp_path / "complete",
        [("amd.vivado.post_route_utilization", report, {"scope": _scope()})],
        capacities={"lut": 100, "ff": 100, "dsp": 10},
    ))
    gate = [item for item in complete.derivations if item.predicate == "gate.resource_fits"]
    assert len(gate) == 1 and gate[0].value is True
