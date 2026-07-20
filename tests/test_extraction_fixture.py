from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from hlsgraph.bundle import GraphBundle
from hlsgraph.graph import CanonicalGraph
from hlsgraph.extract import (
    ExtractionContext,
    ExtractionError,
    ExtractionPipeline,
    ExtractionResult,
    LibClangExtractor,
    VitisReportExtractor,
)
from hlsgraph.manifest import load_manifest, minimal_manifest
from hlsgraph.model import (
    ArtifactRef, AuthorityClass, DatasetManifest, Entity, Relation, Stage,
)
from hlsgraph.sdk import Project
from hlsgraph.version import FEATURE_SCHEMA_VERSION


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
    assert all(item.attrs["ml_input_evidence"] is True
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


def test_static_features_are_scope_level_evidence_with_explicit_masks(golden):
    project, result = golden
    graph = project.service().graph()
    derivations = [
        item for item in project.bundle.store.derivations(result.snapshot_id)
        if item["predicate"].startswith("feature.")
    ]
    pairs = [(item["subject_id"], item["predicate"]) for item in derivations]
    assert len(pairs) == len(set(pairs))

    llvm_function = _by_name(graph, "ir.llvm.function", "compute")
    llvm_rows = {
        item["predicate"]: item for item in derivations
        if item["subject_id"] == llvm_function.id
    }
    assert llvm_rows["feature.operation_histogram"]["value"] == {
        "br": 1, "mul": 1, "ret": 1,
    }
    assert llvm_rows["feature.bitwidth"]["value"] == {"32": 5}
    assert llvm_rows["feature.index_histogram"]["value"] == {}
    assert llvm_rows["feature.memory_access"]["value"] == {}
    complete_llvm_predicates = {
        "feature.operation_histogram", "feature.bitwidth",
        "feature.index_histogram", "feature.memory_access",
    }
    assert all(llvm_rows[predicate]["completeness"] == "complete"
               for predicate in complete_llvm_predicates)
    assert all(item["algorithm_version"] == "1" for item in llvm_rows.values())
    assert any(
        reference["kind"] == "relation"
        for item in llvm_rows.values() for reference in item["evidence_refs"]
    )
    queried = project.feature_evidence(
        llvm_function.id,
        predicates=["feature.operation_histogram", "feature.memory_access"],
    )
    assert queried["rejected_nonstatic_records"] == 0
    assert all(item["mask"] is True for item in queried["items"])

    compute_loop = _by_name(graph, "hls.loop", "compute_loop")
    loop_rows = {
        item["predicate"]: item for item in derivations
        if item["subject_id"] == compute_loop.id
    }
    assert loop_rows["feature.trip_count"]["value"] == 16
    assert loop_rows["feature.loop_bounds"]["value"] == {
        "comparison": "lt", "lower": 0, "step": 1, "upper": 16,
        "upper_inclusive": False,
    }
    assert loop_rows["feature.operation_histogram"]["value"] is None
    assert loop_rows["feature.operation_histogram"]["completeness"] == "missing"
    loop_query = project.feature_evidence(
        compute_loop.id,
        predicates=["feature.trip_count", "feature.operation_histogram"],
    )
    masks = {item["predicate"]: item["mask"] for item in loop_query["items"]}
    assert masks == {
        "feature.operation_histogram": False,
        "feature.trip_count": True,
    }

    mlir_function = _by_name(graph, "ir.mlir.function", "dut")
    mlir_operation = project.feature_evidence(
        mlir_function.id, predicates=["feature.operation_histogram"],
    )["items"][0]
    assert mlir_operation["value"]["handshake.mul"] == 1
    assert mlir_operation["completeness"] == "complete"
    assert mlir_operation["mask"] is True

    kernel = _by_name(graph, "hls.kernel", "dut")
    call_targets = project.feature_evidence(
        kernel.id, predicates=["feature.software_call_targets"],
    )["items"][0]
    assert call_targets["value"] == sorted(
        item.id for item in graph.entities.values()
        if item.kind == "hls.function" and item.name in {"load", "compute", "store"}
    )
    assert call_targets["unit"] == "entity_ids"
    assert call_targets["completeness"] == "partial"
    assert call_targets["mask"] is False
    assert all(
        graph.relations[reference["target_id"]].kind == "software.calls"
        for reference in call_targets["evidence_refs"]
        if reference["kind"] == "relation"
    )
    empty_call_scope = _by_name(graph, "hls.function", "read")
    empty_calls = project.feature_evidence(
        empty_call_scope.id, predicates=["feature.software_call_targets"],
    )["items"][0]
    assert empty_calls["value"] == []
    assert empty_calls["mask"] is True
    missing_dependence = [
        item for item in derivations
        if item["predicate"] == "feature.dependence_distance"
    ]
    assert missing_dependence
    assert all(item["value"] is None and item["completeness"] == "missing"
               for item in missing_dependence)

    dataset = DatasetManifest(
        dataset_id="dataset.indexed_static_features",
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        snapshot_ids=[result.snapshot_id],
        feature_evidence_predicates=[
            "feature.operation_histogram",
            "feature.index_histogram",
            "feature.trip_count",
            "feature.loop_bounds",
            "feature.bitwidth",
            "feature.memory_access",
            "feature.dependence_distance",
            "feature.software_call_targets",
        ],
    )
    output = project.bundle.project_root / "static-feature-export"
    manifest = project.export_dataset(
        output, dataset, snapshot_id=result.snapshot_id,
    )
    exported = [json.loads(line) for line in (
        output / "feature_evidence.jsonl"
    ).read_text(encoding="utf-8").splitlines()]
    assert manifest["row_counts"]["feature_evidence"] == len(exported)
    exported_by_key = {
        (item["subject_id"], item["predicate"]): item for item in exported
    }
    assert exported_by_key[
        (llvm_function.id, "feature.operation_histogram")
    ]["mask"] is True
    assert exported_by_key[
        (compute_loop.id, "feature.trip_count")
    ]["value"] == 16
    assert exported_by_key[
        (kernel.id, "feature.software_call_targets")
    ]["unit"] == "entity_ids"
    exported_dependence = [
        item for item in exported
        if item["predicate"] == "feature.dependence_distance"
    ]
    assert exported_dependence
    assert all(item["value"] is None and item["mask"] is False
               for item in exported_dependence)


def test_config_scope_and_cosim_mismatch_trace_through_independent_gates(tmp_path):
    if not LibClangExtractor.available():
        pytest.skip("the full AST-to-report fixture requires the hlsgraph[clang] extra")
    root = tmp_path / "cosim_failure"
    shutil.copytree(
        FIXTURE, root,
        ignore=shutil.ignore_patterns(".hlsgraph", "__pycache__", "*.pyc"),
    )
    manifest = load_manifest(root / "hlsgraph.toml")
    (root / "hlsgraph.cfg").write_text(
        "syn.directive.unroll = compute_loop,factor=2\n", encoding="utf-8",
    )
    manifest.build.config_files = ["hlsgraph.cfg"]
    manifest.artifact_paths = [
        item for item in manifest.artifact_paths
        if item.get("path") != "reports/dut_cosim.rpt"
    ]
    manifest.artifact_paths.append({
        "path": "cases/cosim_fail.rpt",
        "kind": "amd.vitis.cosim_rpt",
        "role": "cosim_report",
        "access": "project",
        "license": "Apache-2.0",
        "metadata": {
            "workload_id": "tb.mismatch",
            "fixture_authority": "synthetic",
        },
    })
    project = Project(GraphBundle.create(root, manifest))
    result = project.index()
    assert result.success is True

    graph = project.service().graph()
    config_directives = [item for item in graph.entities.values()
                         if item.kind == "hls.directive"
                         and item.name == "UNROLL"
                         and item.attrs.get("origin") == "config"]
    assert len(config_directives) == 1
    annotation = next(item for item in graph.relations.values()
                      if item.kind == "hls.annotates"
                      and item.src == config_directives[0].id)
    assert graph.entities[annotation.dst].name == "compute_loop"
    assert annotation.attrs["scope_node_id"] == annotation.dst
    config_values = {
        item.predicate: item for item in
        project.bundle.store.observations(result.snapshot_id)
        if item.subject_id == config_directives[0].id
    }
    assert config_values["directive.requested"].value == {"factor": 2}
    assert config_values["directive.effective"].value == {"factor": 2}
    assert config_values["directive.effective"].metadata["tool_applied"] is False

    verifications = project.bundle.store.verifications(result.snapshot_id)
    cosim = [item for item in verifications if item["kind"] == "rtl_cosim"]
    assert len(cosim) == 1
    assert cosim[0]["status"] == "fail"
    assert cosim[0]["workload_id"] == "tb.mismatch"
    gates = project.service().verification_gates()
    assert gates["correctness"]["status"] == "fail"
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
    assert "feature.static_derivations" in result.capabilities
    assert any(item.code == "source.degraded_mode" for item in diagnostics)
    assert project.status().to_dict()["completeness"] == "partial"

    graph = project.service().graph()
    kernel = _by_name(graph, "hls.kernel", "dut")
    loop = next(item for item in graph.entities.values() if item.kind == "hls.loop")
    feature_rows = [
        item for item in project.bundle.store.derivations(result.snapshot_id)
        if item["predicate"].startswith("feature.")
    ]
    by_pair = {(item["subject_id"], item["predicate"]): item
               for item in feature_rows}
    assert by_pair[(kernel.id, "feature.operation_histogram")]["value"] is None
    assert by_pair[(kernel.id, "feature.operation_histogram")]["completeness"] == "missing"
    assert by_pair[(loop.id, "feature.trip_count")]["value"] is None
    assert by_pair[(loop.id, "feature.loop_bounds")]["value"] is None

    selected = DatasetManifest(
        dataset_id="dataset.degraded_static_features",
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        snapshot_ids=[result.snapshot_id],
        feature_evidence_predicates=[
            "feature.operation_histogram", "feature.trip_count",
            "feature.dependence_distance",
            "feature.software_call_targets",
        ],
    )
    output = tmp_path / "feature-export"
    project.export_dataset(output, selected, snapshot_id=result.snapshot_id)
    exported = [json.loads(line) for line in (
        output / "feature_evidence.jsonl"
    ).read_text(encoding="utf-8").splitlines()]
    assert exported
    assert all(item["value"] is None and item["mask"] is False
               for item in exported)

    first_payloads = {
        item["id"]: item for item in feature_rows
    }
    first_hash = graph.graph_hash
    repeated = project.index(degraded=True)
    assert repeated.snapshot_id == result.snapshot_id
    assert project.service().graph().graph_hash == first_hash
    repeated_payloads = {
        item["id"]: item for item in
        project.bundle.store.derivations(repeated.snapshot_id)
        if item["predicate"].startswith("feature.")
    }
    assert repeated_payloads == first_payloads


def test_llvm_static_features_include_memory_and_index_evidence(tmp_path):
    (tmp_path / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    (tmp_path / "dut.ll").write_text(
        """define i32 @dut(ptr %p, i64 %idx) {
entry:
  %addr = getelementptr i32, ptr %p, i64 %idx
  %value = load i32, ptr %addr
  store i32 %value, ptr %p
  ret i32 %value
}
""",
        encoding="utf-8",
    )
    manifest = minimal_manifest(
        "test.llvm_static_features", "LLVM static features", "dut", "kernel.cpp",
    )
    manifest.artifact_paths.append({
        "path": "dut.ll", "kind": "ir.llvm", "role": "llvm_ir",
        "access": "project", "license": "Apache-2.0",
    })
    project = Project(GraphBundle.create(tmp_path, manifest))
    result = project.index(degraded=True)
    assert result.success
    graph = project.service().graph()
    function = _by_name(graph, "ir.llvm.function", "dut")
    rows = {
        item["predicate"]: item for item in
        project.feature_evidence(function.id)["items"]
    }
    assert rows["feature.operation_histogram"]["value"] == {
        "getelementptr": 1, "load": 1, "ret": 1, "store": 1,
    }
    assert rows["feature.index_histogram"]["value"] == {"dynamic": 1}
    assert rows["feature.memory_access"]["value"] == {
        "address": 1, "load": 1, "store": 1,
    }
    assert rows["feature.bitwidth"]["value"] == {"32": 5, "64": 2}
    complete_predicates = {
        "feature.operation_histogram", "feature.index_histogram",
        "feature.memory_access", "feature.bitwidth",
    }
    assert all(rows[predicate]["mask"] is True
               for predicate in complete_predicates)
    assert all(rows[predicate]["stage"] == "llvm"
               for predicate in complete_predicates)
    assert rows["feature.dependence_distance"]["value"] is None
    assert rows["feature.dependence_distance"]["mask"] is False

    dataset = DatasetManifest(
        dataset_id="dataset.llvm_static_features",
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        snapshot_ids=[result.snapshot_id],
        feature_evidence_predicates=[
            "feature.operation_histogram", "feature.index_histogram",
            "feature.bitwidth", "feature.memory_access",
        ],
    )
    output = tmp_path / "llvm-feature-export"
    project.export_dataset(output, dataset, snapshot_id=result.snapshot_id)
    exported = [json.loads(line) for line in (
        output / "feature_evidence.jsonl"
    ).read_text(encoding="utf-8").splitlines()]
    exported_function = [
        item for item in exported if item["subject_id"] == function.id
    ]
    assert {item["predicate"] for item in exported_function} == {
        "feature.operation_histogram", "feature.index_histogram",
        "feature.bitwidth", "feature.memory_access",
    }
    assert all(item["mask"] is True for item in exported_function)


def test_supported_untruncated_mlir_is_a_complete_static_feature_domain(tmp_path):
    (tmp_path / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    (tmp_path / "dut.mlir").write_text(
        """module {
  func.func @dut() {
    scf.for %i = 0 to 8 step 2 {
      func.return
    }
  }
}
""",
        encoding="utf-8",
    )
    manifest = minimal_manifest(
        "test.mlir_complete_features", "MLIR complete features", "dut", "kernel.cpp",
    )
    manifest.artifact_paths.append({
        "path": "dut.mlir", "kind": "ir.mlir", "role": "hls_ir",
        "access": "project", "license": "Apache-2.0",
    })
    project = Project(GraphBundle.create(tmp_path, manifest))
    result = project.index(degraded=True)
    assert result.success
    assert "ir.mlir.complete_static_feature_domain" in result.capabilities
    graph = project.service().graph()
    function = _by_name(graph, "ir.mlir.function", "dut")
    operation_row = project.feature_evidence(
        function.id, predicates=["feature.operation_histogram"],
    )["items"][0]
    assert operation_row["value"]["scf.for"] == 1
    assert operation_row["completeness"] == "complete"
    assert operation_row["mask"] is True

    loop = next(item for item in graph.entities.values()
                if item.kind == "hls.loop" and item.stage == "mlir")
    loop_rows = {
        item["predicate"]: item for item in project.feature_evidence(
            loop.id,
            predicates=["feature.trip_count", "feature.loop_bounds"],
        )["items"]
    }
    assert loop_rows["feature.trip_count"]["value"] == 4
    assert loop_rows["feature.trip_count"]["mask"] is True
    assert loop_rows["feature.loop_bounds"]["value"] == {
        "comparison": "lt", "lower": 0, "step": 2, "upper": 8,
        "upper_inclusive": False,
    }
    assert loop_rows["feature.loop_bounds"]["mask"] is True

    dataset = DatasetManifest(
        dataset_id="dataset.mlir_complete_features",
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        snapshot_ids=[result.snapshot_id],
        feature_evidence_predicates=[
            "feature.operation_histogram", "feature.trip_count",
            "feature.loop_bounds",
        ],
    )
    output = tmp_path / "mlir-feature-export"
    project.export_dataset(output, dataset, snapshot_id=result.snapshot_id)
    exported = [json.loads(line) for line in (
        output / "feature_evidence.jsonl"
    ).read_text(encoding="utf-8").splitlines()]
    assert any(item["subject_id"] == function.id
               and item["predicate"] == "feature.operation_histogram"
               and item["mask"] is True for item in exported)
    assert any(item["subject_id"] == loop.id
               and item["predicate"] == "feature.trip_count"
               and item["mask"] is True for item in exported)


def test_dependence_distance_is_emitted_only_from_explicit_evidence() -> None:
    from hlsgraph.extract.static_features import derive_static_features

    graph = CanonicalGraph("snapshot.explicit_dependence")
    loop = graph.add_entity(Entity(
        "hls.loop", "loop", graph.snapshot_id, stage="ast",
    ))
    variable = graph.add_entity(Entity(
        "source.variable", "value", graph.snapshot_id, stage="ast",
    ))
    relation = graph.add_relation(Relation(
        loop.id, variable.id, "hls.contains", graph.snapshot_id, stage="ast",
        attrs={"dependence_distance": 2},
    ))
    extraction = ExtractionResult(graph=graph)
    derive_static_features(extraction)
    dependence = [
        item for item in extraction.derivations
        if item.predicate == "feature.dependence_distance"
    ]
    assert len(dependence) == 1
    assert dependence[0].value == {"distances": [2]}
    assert dependence[0].completeness == "complete"
    assert relation.id in {item.target_id for item in dependence[0].evidence_refs}

    graph_without_distance = CanonicalGraph("snapshot.no_dependence")
    plain_loop = graph_without_distance.add_entity(Entity(
        "hls.loop", "loop", graph_without_distance.snapshot_id, stage="ast",
    ))
    plain_variable = graph_without_distance.add_entity(Entity(
        "source.variable", "value", graph_without_distance.snapshot_id, stage="ast",
    ))
    graph_without_distance.add_relation(Relation(
        plain_loop.id, plain_variable.id, "hls.contains",
        graph_without_distance.snapshot_id, stage="ast",
    ))
    empty = ExtractionResult(graph=graph_without_distance)
    derive_static_features(empty)
    missing = [item for item in empty.derivations
               if item.predicate == "feature.dependence_distance"]
    assert len(missing) == 1
    assert missing[0].value is None
    assert missing[0].completeness == "missing"


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
    module = next(item for item in graph.entities.values()
                  if item.kind == "ir.mlir.module")
    feature = project.feature_evidence(
        module.id, predicates=["feature.operation_histogram"],
    )["items"][0]
    assert feature["value"] == {"mystery.compute": 1}
    assert feature["completeness"] == "partial"
    assert feature["mask"] is False


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
