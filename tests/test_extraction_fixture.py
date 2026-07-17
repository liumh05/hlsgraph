from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pytest

from hlsgraph.bundle import GraphBundle
from hlsgraph.extract import (
    ExtractionContext,
    ExtractionError,
    ExtractionPipeline,
    ExtractionResult,
    LibClangExtractor,
    VitisReportExtractor,
)
from hlsgraph.manifest import minimal_manifest
from hlsgraph.model import ArtifactRef, AuthorityClass, Stage
from hlsgraph.sdk import Project


FIXTURE = Path(__file__).resolve().parents[1] / "examples" / "dataflow_gemm"


@pytest.fixture()
def golden(tmp_path):
    if not LibClangExtractor.available():
        pytest.skip("the full AST-to-report fixture requires the hlsgraph[clang] extra")
    root = tmp_path / "dataflow_gemm"
    shutil.copytree(
        FIXTURE, root,
        ignore=shutil.ignore_patterns(".hlsgraph", "__pycache__", "*.pyc"),
    )
    project = Project.create_from_manifest(root / "hlsgraph.toml")
    result = project.index()
    assert result.success, [
        (item.code, item.message)
        for item in project.bundle.store.diagnostics(result.snapshot_id)
        if str(item.severity) in {"error", "critical"}
    ]
    return project, result


def _by_name(graph, kind: str, name: str):
    items = [item for item in graph.entities.values()
             if item.kind == kind and item.name == name]
    assert len(items) == 1, (kind, name, [(item.kind, item.name) for item in items])
    return items[0]


def test_golden_fixture_preserves_planes_and_only_projects_dialect_defined_topology(golden):
    project, result = golden
    graph = project.service().graph()

    assert {"hls.kernel", "hls.loop", "hls.directive", "hls.buffer", "hls.process",
            "ir.mlir.operation", "ir.llvm.operation", "hls.scheduled_operation"}.issubset(
        graph.stats()["entity_kinds"]
    )
    assert {"source.ast", "ir.mlir.evidence", "ir.llvm.cfg_evidence",
            "amd.vitis.schedule", "amd.vivado.timing"}.issubset(result.capabilities)

    architecture_edges = [item for item in graph.relations.values()
                          if item.kind == "hls.streams_to"]
    assert len(architecture_edges) == 2
    assert {item.attrs["fifo_depth"] for item in architecture_edges} == {8, 16}
    assert all(item.attrs["projection"] == "handshake_semantics"
               for item in architecture_edges)

    software_calls = [item for item in graph.relations.values()
                      if item.kind == "software.calls"]
    llvm_cfg = [item for item in graph.relations.values() if item.kind == "llvm.cfg"]
    assert software_calls and all(item.attrs["hardware_instance"] is False
                                  for item in software_calls)
    assert llvm_cfg and all(item.attrs["hardware_topology"] is False for item in llvm_cfg)
    assert not ({item.id for item in software_calls + llvm_cfg}
                & {item.id for item in architecture_edges})

    schedule_maps = [item for item in graph.relations.values()
                     if item.kind == "cross.maps_to"]
    assert len(schedule_maps) == 1
    assert schedule_maps[0].mapping_kind == "schedule.explicit_architecture_name"
    assert schedule_maps[0].attrs["cardinality"] == "explicit"
    mapped = graph.entities[schedule_maps[0].dst]
    assert mapped.kind == "hls.process" and mapped.name == "handshake.mul@4"


def test_missing_and_ambiguous_cross_layer_mappings_are_diagnostic_not_guessed(golden):
    project, _result = golden
    graph = project.service().graph()
    diagnostics = project.bundle.store.diagnostics(graph.snapshot_id)

    ambiguous = [item for item in diagnostics
                 if item.code == "mapping.ambiguous_mlir_location"]
    assert ambiguous
    assert len(ambiguous[0].metadata["candidate_ids"]) > 1
    ambiguous_subject = ambiguous[0].subject_id
    assert ambiguous_subject
    assert not any(
        item.src == ambiguous_subject and item.kind == "cross.maps_source"
        for item in graph.relations.values()
    )

    llvm_operations = graph.by_kind("ir.llvm.operation")
    assert llvm_operations
    assert all(item.attrs["cfg_is_hls_topology"] is False for item in llvm_operations)
    assert any(anchor.mapping_kind == "llvm.debug"
               for item in llvm_operations for anchor in item.anchors)

    schedule_operations = graph.by_kind("hls.scheduled_operation")
    unmapped = [item for item in schedule_operations if not any(
        relation.src == item.id and relation.kind == "cross.maps_to"
        for relation in graph.relations.values()
    )]
    assert unmapped
    assert all(any(
        observation.subject_id == item.id
        and observation.metadata.get("mapping_status") == "operation_only"
        for observation in project.bundle.store.observations(graph.snapshot_id)
    ) for item in unmapped)


def test_pragma_and_tcl_directives_bind_scope_and_keep_requested_effective_achieved_separate(golden):
    project, _result = golden
    graph = project.service().graph()
    observations = project.bundle.store.observations(graph.snapshot_id)
    diagnostics = project.bundle.store.diagnostics(graph.snapshot_id)
    compute_loop = _by_name(graph, "hls.loop", "compute_loop")

    pipeline_annotations = [relation for relation in graph.relations.values()
                            if relation.kind == "hls.annotates"
                            and relation.dst == compute_loop.id
                            and graph.entities[relation.src].name == "PIPELINE"]
    assert len(pipeline_annotations) == 2
    assert all(item.attrs["scope_node_id"] == compute_loop.id
               for item in pipeline_annotations)
    directives = [graph.entities[item.src] for item in pipeline_annotations]
    inline = next(item for item in directives if item.attrs["origin"] == "source_pragma")
    external = next(item for item in directives if item.attrs["origin"] == "tcl")
    assert inline.attrs["options"]["ii"] == 2
    assert inline.attrs["state"] == "overridden_declared"
    assert external.attrs["options"]["ii"] == 1
    assert external.attrs["state"] == "effective_declared"
    assert any(item.code == "directive.declared_override" and item.subject_id == inline.id
               for item in diagnostics)

    values = {item.predicate: item for item in observations if item.subject_id == external.id}
    assert values["directive.requested"].value == {"ii": 1}
    assert values["directive.effective"].value == {"ii": 1}
    assert values["directive.effective"].metadata["tool_applied"] is False
    assert values["directive.reported_requested"].value == {"ii": 1}
    assert values["directive.tool_effective"].value == {"ii": 1}
    assert values["directive.achieved"].value == {"ii": 2}
    assert values["directive.tool_status"].value == "unmet"
    assert any(item.code == "directive.unmet" and item.subject_id == external.id
               for item in diagnostics)

    # Directives of one kind on different objects are not precedence conflicts.
    stream_directives = [item for item in graph.entities.values()
                         if item.kind == "hls.directive" and item.name == "STREAM"]
    stream_targets = {
        relation.dst for relation in graph.relations.values()
        if relation.kind == "hls.annotates"
        and relation.src in {item.id for item in stream_directives}
    }
    assert len(stream_directives) == 2
    assert len(stream_targets) == 2
    assert all(item.attrs["state"] == "effective_declared" for item in stream_directives)


def test_stage_scoped_observations_and_verification_evidence_coexist(golden):
    project, _result = golden
    graph = project.service().graph()
    kernel = _by_name(graph, "hls.kernel", "dut")
    observations = project.bundle.store.observations(graph.snapshot_id, subject_id=kernel.id)

    lut = [item for item in observations if item.predicate == "resource.lut"]
    assert {(item.stage, float(item.value)) for item in lut} == {
        (Stage.SCHEDULE.value, 1100.0),
        (Stage.POST_ROUTE.value, 1200.0),
    }
    assert all(item.unit == "count" for item in lut)
    assert all(item.authority == AuthorityClass.SYNTHETIC for item in lut)

    wns = next(item for item in observations if item.predicate == "timing.wns_ns")
    assert wns.stage == Stage.POST_ROUTE.value and wns.value == -0.12
    cosim = [item for item in observations if item.stage == Stage.COSIM.value]
    assert cosim and {item.workload_id for item in cosim} == {"tb.default"}
    assert all(item.metadata["dynamic"] is True for item in cosim)

    derivations = {item["predicate"]: item
                   for item in project.bundle.store.derivations(graph.snapshot_id)}
    assert derivations["gate.resource_fits"]["value"] is True
    assert derivations["gate.post_route_timing"]["value"] is False
    verifications = project.bundle.store.verifications(graph.snapshot_id)
    verification_status = {item["kind"]: item["status"] for item in verifications}
    assert verification_status == {"csim": "pass", "rtl_cosim": "pass"}
    assert verifications[0]["workload_id"] == "tb.default"

    gates = project.service().verification_gates()
    assert gates["correctness"]["status"] == "pass"
    assert gates["resource_fits"]["status"] == "pass"
    assert gates["post_route_timing"]["status"] == "fail"
    assert gates["verified"] is False


def test_degraded_mode_is_explicit_and_visible_in_health(tmp_path):
    (tmp_path / "kernel.cpp").write_text(
        "void dut(int *x) { for (int i=0; i<2; ++i) x[i]++; }\n",
        encoding="utf-8",
    )
    project = Project(GraphBundle.create(
        tmp_path, minimal_manifest("test.degraded", "degraded", "dut", "kernel.cpp")
    ))
    result = project.index(degraded=True)
    diagnostics = project.bundle.store.diagnostics(result.snapshot_id)
    assert result.success
    assert "source.degraded" in result.capabilities
    assert any(item.code == "source.degraded_mode" for item in diagnostics)
    assert project.status().to_dict()["completeness"] == "partial"


def test_libclang_fails_closed_when_macro_include_is_not_in_snapshot(tmp_path):
    if not LibClangExtractor.available():
        pytest.skip("libclang extra is unavailable")
    (tmp_path / "kernel.cpp").write_text(
        '#define PROJECT_HEADER "hidden.hpp"\n#include PROJECT_HEADER\nvoid dut() {}\n',
        encoding="utf-8",
    )
    (tmp_path / "hidden.hpp").write_text("#define HIDDEN 1\n", encoding="utf-8")
    bundle = GraphBundle.create(
        tmp_path, minimal_manifest("test.macro_include", "macro include", "dut", "kernel.cpp")
    )
    snapshot = bundle.snapshot()
    artifacts = bundle.store.artifacts(snapshot.id)
    assert "hidden.hpp" not in {item.uri for item in artifacts}
    result = LibClangExtractor().extract(ExtractionContext(
        project_root=tmp_path, manifest=bundle.manifest, snapshot=snapshot,
        artifacts={item.id: item for item in artifacts},
    ))
    assert any(item.code == "source.untracked_project_include"
               and item.metadata["path"] == "hidden.hpp"
               for item in result.diagnostics)


def test_unsupported_mlir_dialect_is_preserved_as_evidence_and_reported(tmp_path):
    (tmp_path / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    (tmp_path / "unknown.mlir").write_text(
        "module {\n  %0 = mystery.compute %arg0 : i32\n}\n", encoding="utf-8"
    )
    manifest = minimal_manifest("test.unknown_dialect", "unknown dialect", "dut", "kernel.cpp")
    manifest.artifact_paths.append({
        "path": "unknown.mlir", "kind": "ir.mlir", "role": "hls_ir",
        "access": "project",
    })
    project = Project(GraphBundle.create(tmp_path, manifest))
    result = project.index(degraded=True)
    graph = project.service().graph()
    unknown = [item for item in graph.entities.values()
               if item.kind == "ir.mlir.operation" and item.name == "mystery.compute"]
    assert len(unknown) == 1
    assert unknown[0].attrs["plane"] == "evidence"
    assert not any(relation.src == unknown[0].id and relation.kind.startswith("hls.")
                   for relation in graph.relations.values())
    assert any(item.code == "mlir.unsupported_dialect"
               for item in project.bundle.store.diagnostics(result.snapshot_id))


def test_missing_report_and_extractor_failure_are_structured_diagnostics(tmp_path):
    (tmp_path / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    bundle = GraphBundle.create(
        tmp_path, minimal_manifest("test.diagnostics", "diagnostics", "dut", "kernel.cpp")
    )
    snapshot = bundle.snapshot()
    missing = ArtifactRef(
        kind="amd.vitis.csynth_xml", uri="reports/missing.xml",
        sha256=hashlib.sha256(b"missing").hexdigest(), size=0, access="project",
    )
    context = ExtractionContext(
        project_root=tmp_path, manifest=bundle.manifest, snapshot=snapshot,
        artifacts={missing.id: missing},
    )
    report_result = VitisReportExtractor().extract(context)
    assert any(item.code == "vitis.report_parse_error"
               and item.artifact_id == missing.id for item in report_result.diagnostics)

    class BrokenExtractor:
        name = "test.broken"
        version = "1"

        def supports(self, _context):
            return True

        def extract(self, _context):
            raise ExtractionError("intentional fixture failure")

    failed = ExtractionPipeline([BrokenExtractor()]).run(context)
    assert len(failed.diagnostics) == 1
    assert failed.diagnostics[0].code == "extractor.failed"
    assert "details withheld" in failed.diagnostics[0].message
    assert failed.diagnostics[0].metadata["error_type"] == "ExtractionError"
    assert len(failed.diagnostics[0].metadata["error_fingerprint"]) == 64
    assert "intentional fixture failure" not in failed.diagnostics[0].message
    assert not failed.graph.entities and not failed.graph.relations
    assert failed.graph.metadata == {"coverage": {}, "capabilities": []}
