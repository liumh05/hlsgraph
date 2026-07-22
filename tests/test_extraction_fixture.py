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
    ArtifactRef, ArtifactSemanticAttestation, ArtifactSemanticClaim,
    AuthorityClass, DatasetManifest, Entity, LanguageSpecCompatibility,
    Relation, Stage, ToolchainContext, json_ready, stable_hash,
)
from hlsgraph.retrieval import HybridRetriever
from hlsgraph.sdk import Project
from hlsgraph.version import FEATURE_SCHEMA_VERSION
from tests.reviewed_knowledge_support import install_reviewed_builtin_packs


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
    install_reviewed_builtin_packs(project.bundle)
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

    native_handshake = [
        item for item in graph.relations.values()
        if item.kind == "handshake.dataflow"
    ]
    assert native_handshake
    assert all(
        item.attrs["hardware_topology"] is False
        and item.attrs["native_ir_evidence"] is True
        and item.attrs["native_ir_evidence_contract"]
        == "hlsgraph.mlir.ssa_def_use.v1"
        and item.attrs["native_ir_relation_provenance"] == "mlir.ssa_def_use"
        and len(item.anchors) == 1
        and item.anchors[0].artifact_id == item.attrs["native_ir_artifact_id"]
        and graph.entities[item.src].attrs["ssa_result"]
        == item.attrs["ssa_value"]
        and item.attrs["ssa_value"]
        in graph.entities[item.dst].attrs["ssa_operands"]
        for item in native_handshake
    )
    retriever = HybridRetriever(project.bundle, graph.snapshot_id)
    handshake_contexts = retriever._binding_target_contexts(
        graph, set(graph.entities),
    )[("relation_kind", "handshake.dataflow")]
    assert handshake_contexts and all(
        "semantic_artifact_evidence_qualified" not in context
        and "language_spec_family" not in context
        and "semantic_attestation_identity" not in context
        for context in handshake_contexts
    )
    assert not any(
        item.target_kind == "relation_kind"
        and item.target == "handshake.dataflow"
        for item in project.bundle.store.knowledge_bindings()
    )

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
    assert ambiguous[0].metadata["mapping_resolution"] == "ambiguous"
    assert ambiguous[0].metadata["mapping_ambiguous"] is True
    assert ambiguous[0].metadata["mapping_unresolved"] is False
    assert ambiguous[0].metadata["mapping_redacted"] is False
    assert ambiguous[0].metadata["mapping_candidate_count"] > 1
    ambiguous_subject = ambiguous[0].subject_id
    assert ambiguous_subject
    assert not any(
        item.src == ambiguous_subject and item.kind == "cross.maps_to"
        and item.stage == "mlir"
        for item in graph.relations.values()
    )
    diagnostic_contexts = HybridRetriever(
        project.bundle, graph.snapshot_id,
    )._binding_target_contexts(
        graph, set(graph.entities),
    )[("diagnostic_code", "mapping.ambiguous_mlir_location")]
    for context in diagnostic_contexts:
        assert "unique_mlir_location_mapping_resolved" not in context
        assert "typed_source_anchor_identity" not in context
        assert "resolved_target_anchor_identity" not in context

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


def test_pragma_and_tcl_directives_bind_scope_and_keep_declared_tool_and_achieved_separate(golden):
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
    assert external.attrs["state"] == "selected_declared"
    for directive in directives:
        assert directive.attrs["directive_instance_id"] == directive.id
        assert directive.attrs["scope_id"] == compute_loop.id
        assert directive.attrs["scope_kind"] == "hls.loop"
        assert directive.attrs["loop_id"] == compute_loop.id
    assert any(item.code == "directive.declared_override" and item.subject_id == inline.id
               for item in diagnostics)

    values = {item.predicate: item for item in observations if item.subject_id == external.id}
    assert values["directive.requested"].value == {"ii": 1}
    assert values["directive.declared_selected"].value == {"ii": 1}
    assert values["directive.declared_selected"].metadata["tool_applied"] is False
    assert values["directive.reported_requested"].value == {"ii": 1}
    assert values["directive.tool_effective"].value == {"ii": 1}
    assert values["directive.achieved"].value == {"ii": 2}
    assert values["directive.tool_status"].value == "unmet"
    assert all(
        values[predicate].metadata["directive_instance_id"] == external.id
        and values[predicate].metadata["scope_id"] == compute_loop.id
        and values[predicate].metadata["loop_id"] == compute_loop.id
        for predicate in (
            "directive.requested", "directive.declared_selected",
            "directive.reported_requested", "directive.tool_effective",
            "directive.achieved", "directive.tool_status",
        )
    )
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
    assert all(item.attrs["state"] == "selected_declared" for item in stream_directives)
    assert all(item.attrs["directive_instance_id"] == item.id
               and item.attrs["scope_id"] in stream_targets
               and item.attrs["scope_kind"] == "hls.stream"
               and item.attrs["variable_id"] == item.attrs["scope_id"]
               for item in stream_directives)
    retriever = HybridRetriever(project.bundle, graph.snapshot_id)
    declaration_predicates = {
        "directive.requested", "directive.declared_selected",
    }
    report_predicates = {
        "directive.reported_requested", "directive.tool_effective",
        "directive.achieved", "directive.tool_status",
    }
    for predicate in sorted(declaration_predicates | report_predicates):
        contexts = retriever._binding_target_contexts(
            graph, set(graph.entities),
        )[("predicate", predicate)]
        context = next(
            item for item in contexts
            if item.get("directive_instance_id") == {external.id.casefold()}
        )
        assert context["requested_directive_present"] == {"true"}
        binding = next(
            item for item in project.bundle.store.knowledge_bindings()
            if item.target_kind == "predicate" and item.target == predicate
        )
        applicable = HybridRetriever._binding_applicable(
            binding, context, {"predicate": {predicate}},
        )
        if predicate in declaration_predicates:
            assert applicable
        else:
            # The golden report fixture is intentionally synthetic and has no
            # fresh real ToolRun plus immutable declared-output closure.  Its
            # parsed status remains evidence, but must not activate normative
            # tool-observation guidance.
            assert not applicable
            assert "observation_evidence_qualified" not in context
    operand_targets = {
        ("STREAM", item.id) for item in stream_directives
    } | {
        ("ARRAY_PARTITION", item.id) for item in graph.entities.values()
        if item.kind == "hls.directive" and item.name == "ARRAY_PARTITION"
    }
    for kind, directive_id in operand_targets:
        contexts = retriever._binding_target_contexts(
            graph, set(graph.entities),
        )[("directive_kind", kind)]
        context = next(
            item for item in contexts
            if item.get("directive_instance_id") == {directive_id.casefold()}
        )
        assert context["directive_operand_linked"] == {
            "derived_from_current_directive_operand_link_v1"
        }
        assert context["directive_operand_identity"]


@pytest.mark.parametrize("interface_mode", ["m_axi", "s_axilite", "axis"])
def test_interface_binding_uses_the_extracted_concrete_port_instance(
    tmp_path: Path, interface_mode: str,
) -> None:
    if not LibClangExtractor.available():
        pytest.skip("the scoped INTERFACE test requires the standard libclang extractor")
    (tmp_path / "kernel.cpp").write_text(
        "void dut(int *input) {\n"
        f"#pragma HLS INTERFACE mode={interface_mode} port=input\n"
        "  input[0] += 1;\n"
        "}\n",
        encoding="utf-8",
    )
    manifest = minimal_manifest(
        "test.directive_port_scope", "directive port scope", "dut", "kernel.cpp",
    )
    manifest.toolchains = [ToolchainContext(
        id="amd.vitis_hls.2024_2", vendor="amd", name="vitis_hls",
        version="2024.2",
    )]
    project = Project(GraphBundle.create(tmp_path, manifest))
    install_reviewed_builtin_packs(project.bundle)
    indexed = project.index()
    assert indexed.success
    graph = project.service(indexed.snapshot_id).graph()
    directive = next(item for item in graph.entities.values()
                     if item.kind == "hls.directive" and item.name == "INTERFACE")
    port = next(item for item in graph.entities.values()
                if item.kind == "hls.port" and item.name == "input")
    assert directive.attrs["directive_instance_id"] == directive.id
    assert directive.attrs["scope_id"] == port.id
    assert directive.attrs["scope_kind"] == "hls.port"
    assert directive.attrs["scope_resolution"] == "source_ast"
    assert directive.attrs["port_id"] == port.id
    assert port.attrs["direction"] == "unknown"

    retriever = HybridRetriever(project.bundle, indexed.snapshot_id)
    contexts = retriever._binding_target_contexts(graph, set(graph.entities))[
        ("directive_kind", "INTERFACE")
    ]
    context = next(item for item in contexts
                   if item.get("directive_instance_id") == {directive.id.casefold()})
    assert context["interface_mode"] == {interface_mode}
    assert not {
        "direction", "protocol", "spec_version", "endpoint_role", "channel_role",
        "interface_instance_id", "transmitter_id", "receiver_id",
    }.intersection(context)
    amd_binding = next(
        item for item in project.bundle.store.knowledge_bindings()
        if item.knowledge_rule_id.endswith(":directive.interface_is_port_contract")
    )
    axi_binding = next(
        item for item in project.bundle.store.knowledge_bindings()
        if item.knowledge_rule_id.endswith(":axi.interface_mode_is_scoped_request")
    )
    target = {"directive_kind": {"INTERFACE"}}
    assert HybridRetriever._binding_applicable(amd_binding, context, target)
    assert HybridRetriever._binding_applicable(axi_binding, context, target)

    annotation = next(
        item for item in graph.relations.values()
        if item.kind == "hls.annotates" and item.src == directive.id
        and item.dst == port.id
    )
    del graph.relations[annotation.id]
    directive.attrs.update({
        "directive_operand_linked":
            "derived_from_current_directive_operand_link_v1",
        "directive_operand_identity": "f" * 64,
    })
    unlinked = retriever._binding_target_contexts(
        graph, set(graph.entities),
    )[("directive_kind", "INTERFACE")]
    unlinked_context = next(
        item for item in unlinked
        if item.get("directive_instance_id") == {directive.id.casefold()}
    )
    assert "directive_operand_linked" not in unlinked_context
    assert "directive_operand_identity" not in unlinked_context
    assert not HybridRetriever._binding_applicable(
        amd_binding, unlinked_context, target,
    )
    assert not HybridRetriever._binding_applicable(
        axi_binding, unlinked_context, target,
    )


@pytest.mark.parametrize(
    ("scope_case", "expected_scope_kind", "required_role"),
    [
        ("loop", "hls.loop", "loop_id"),
        ("function", "hls.kernel", "function_id"),
    ],
)
def test_dependence_separates_enclosing_scope_from_variable_operand(
    tmp_path: Path, scope_case: str, expected_scope_kind: str, required_role: str,
) -> None:
    if not LibClangExtractor.available():
        pytest.skip("the DEPENDENCE scope test requires the standard libclang extractor")
    if scope_case == "loop":
        body = (
            "  for (int i = 0; i < 4; ++i) {\n"
            "#pragma HLS DEPENDENCE variable=input inter false\n"
            "    input[i] += 1;\n"
            "  }\n"
        )
    else:
        body = (
            "#pragma HLS DEPENDENCE variable=input inter false\n"
            "  input[0] += 1;\n"
        )
    (tmp_path / "kernel.cpp").write_text(
        "void dut(int *input) {\n" + body + "}\n", encoding="utf-8",
    )
    manifest = minimal_manifest(
        f"test.dependence_{scope_case}", "dependence scope", "dut", "kernel.cpp",
    )
    manifest.toolchains = [ToolchainContext(
        id="amd.vitis_hls.2024_2", vendor="amd", name="vitis_hls",
        version="2024.2",
    )]
    project = Project(GraphBundle.create(tmp_path, manifest))
    install_reviewed_builtin_packs(project.bundle)
    indexed = project.index()
    assert indexed.success
    graph = project.service(indexed.snapshot_id).graph()
    directive = next(item for item in graph.entities.values()
                     if item.kind == "hls.directive" and item.name == "DEPENDENCE")
    operand = next(item for item in graph.entities.values()
                   if item.kind == "hls.port" and item.name == "input")
    scope = graph.entities[directive.attrs["scope_id"]]
    assert scope.kind == expected_scope_kind
    assert directive.attrs[required_role] == scope.id
    assert directive.attrs["variable_id"] == operand.id
    assert operand.id != scope.id
    assert "port_id" not in directive.attrs
    annotation = next(
        item for item in graph.relations.values()
        if item.kind == "hls.annotates" and item.src == directive.id
    )
    assert annotation.dst == scope.id

    retriever = HybridRetriever(project.bundle, indexed.snapshot_id)
    contexts = retriever._binding_target_contexts(graph, set(graph.entities))[
        ("directive_kind", "DEPENDENCE")
    ]
    context = next(item for item in contexts
                   if item.get("directive_instance_id") == {directive.id.casefold()})
    assert context["scope_id"] == {scope.id.casefold()}
    assert context["variable_id"] == {operand.id.casefold()}
    assert "directive_operand_linked" not in context
    assert context["dependence_operand_resolved"] == {
        "derived_from_current_dependence_operand_v1"
    }
    assert context["directive_operand_identity"]
    bindings = [
        item for item in project.bundle.store.knowledge_bindings()
        if item.target_kind == "directive_kind" and item.target == "DEPENDENCE"
    ]
    target = {"directive_kind": {"DEPENDENCE"}}
    # UG1399 permits either an optional variable selector or a class selector.
    # The current binding condition language cannot express that exclusive
    # alternative without over-constraining valid declarations, so the public
    # pack deliberately publishes no executable DEPENDENCE binding.
    assert bindings == []
    assert sum(HybridRetriever._binding_applicable(item, context, target)
               for item in bindings) == 0

    missing_scope = {
        key: value for key, value in context.items()
        if key not in {"scope_id", "scope_kind", "scope_resolution",
                       "loop_id", "function_id"}
    }
    assert not any(HybridRetriever._binding_applicable(
        item, missing_scope, target,
    ) for item in bindings)
    legacy_variable_as_scope = {
        **context,
        "scope_id": {operand.id.casefold()},
        "scope_kind": {"hls.port"},
    }
    legacy_variable_as_scope.pop("loop_id", None)
    legacy_variable_as_scope.pop("function_id", None)
    assert not any(HybridRetriever._binding_applicable(
        item, legacy_variable_as_scope, target,
    ) for item in bindings)

    del graph.relations[annotation.id]
    directive.attrs.update({
        "dependence_operand_resolved": (
            "derived_from_current_dependence_operand_v1"
        ),
        "directive_operand_identity": "f" * 64,
    })
    unlinked_context = next(
        item for item in retriever._binding_target_contexts(
            graph, set(graph.entities),
        )[("directive_kind", "DEPENDENCE")]
        if item.get("directive_instance_id") == {directive.id.casefold()}
    )
    assert "dependence_operand_resolved" not in unlinked_context
    assert "directive_operand_identity" not in unlinked_context
    assert not any(HybridRetriever._binding_applicable(
        item, unlinked_context, target,
    ) for item in bindings)

    graph.relations[annotation.id] = annotation
    alternate_owner = graph.add_entity(Entity(
        "hls.function", "alternate_owner", graph.snapshot_id,
        qualified_name="alternate_owner", stage="ast",
        authority=AuthorityClass.STATIC_FACT,
    ))
    ambiguous_owner_relation = graph.add_relation(Relation(
        alternate_owner.id, operand.id, "hls.contains", graph.snapshot_id,
        stage="ast", authority=AuthorityClass.STATIC_FACT,
    ))
    ambiguous_owner_context = next(
        item for item in retriever._binding_target_contexts(
            graph, set(graph.entities),
        )[("directive_kind", "DEPENDENCE")]
        if item.get("directive_instance_id") == {directive.id.casefold()}
    )
    assert "dependence_operand_resolved" not in ambiguous_owner_context
    assert "directive_operand_identity" not in ambiguous_owner_context
    assert not any(HybridRetriever._binding_applicable(
        item, ambiguous_owner_context, target,
    ) for item in bindings)
    del graph.relations[ambiguous_owner_relation.id]
    del graph.entities[alternate_owner.id]

    (tmp_path / "kernel.cpp").write_text(
        "void dut(int *changed) {}\n", encoding="utf-8",
    )
    tampered_context = next(
        item for item in retriever._binding_target_contexts(
            graph, set(graph.entities),
        )[("directive_kind", "DEPENDENCE")]
        if item.get("directive_instance_id") == {directive.id.casefold()}
    )
    assert "dependence_operand_resolved" not in tampered_context
    assert "directive_operand_identity" not in tampered_context
    assert not any(HybridRetriever._binding_applicable(
        item, tampered_context, target,
    ) for item in bindings)


def test_dependence_precedence_is_partitioned_by_exact_variable_operand(
    tmp_path: Path,
) -> None:
    if not LibClangExtractor.available():
        pytest.skip("the DEPENDENCE operand test requires the standard libclang extractor")
    (tmp_path / "kernel.cpp").write_text(
        "void dut(int *left, int *right) {\n"
        "  for (int i = 0; i < 4; ++i) {\n"
        "#pragma HLS DEPENDENCE variable=left inter false\n"
        "#pragma HLS DEPENDENCE variable=right inter false\n"
        "    left[i] += right[i];\n"
        "  }\n"
        "}\n",
        encoding="utf-8",
    )
    manifest = minimal_manifest(
        "test.dependence_two_operands", "dependence operands", "dut",
        "kernel.cpp",
    )
    manifest.toolchains = [ToolchainContext(
        id="amd.vitis_hls.2024_2", vendor="amd", name="vitis_hls",
        version="2024.2",
    )]
    project = Project(GraphBundle.create(tmp_path, manifest))
    install_reviewed_builtin_packs(project.bundle)
    indexed = project.index()
    assert indexed.success
    graph = project.service(indexed.snapshot_id).graph()
    directives = [
        item for item in graph.entities.values()
        if item.kind == "hls.directive" and item.name == "DEPENDENCE"
    ]
    assert len(directives) == 2
    assert {item.attrs["options"]["variable"] for item in directives} == {
        "left", "right",
    }
    assert all(
        item.attrs["state"] == "selected_declared"
        and item.completeness.value == "complete"
        for item in directives
    )
    assert not any(
        item.code == "directive.declared_override"
        and item.subject_id in {directive.id for directive in directives}
        for item in project.bundle.store.active_diagnostics(indexed.snapshot_id)
    )
    selected = {
        item.subject_id: item
        for item in project.bundle.store.observations(indexed.snapshot_id)
        if item.predicate == "directive.declared_selected"
        and item.subject_id in {directive.id for directive in directives}
    }
    assert set(selected) == {item.id for item in directives}
    assert all(item.completeness.value == "complete" for item in selected.values())

    retriever = HybridRetriever(project.bundle, indexed.snapshot_id)
    contexts = retriever._binding_target_contexts(
        graph, set(graph.entities),
    )[("directive_kind", "DEPENDENCE")]
    bindings = [
        item for item in project.bundle.store.knowledge_bindings()
        if item.target_kind == "directive_kind" and item.target == "DEPENDENCE"
    ]
    for directive in directives:
        context = next(
            item for item in contexts
            if item.get("directive_instance_id") == {directive.id.casefold()}
        )
        assert context["dependence_operand_resolved"] == {
            "derived_from_current_dependence_operand_v1"
        }
        assert context["directive_operand_identity"]
        assert bindings == []
        assert sum(HybridRetriever._binding_applicable(
            item, context, {"directive_kind": {"DEPENDENCE"}},
        ) for item in bindings) == 0


def test_dependence_missing_exact_operand_is_incomplete_and_has_no_rule_binding(
    tmp_path: Path,
) -> None:
    if not LibClangExtractor.available():
        pytest.skip("the DEPENDENCE operand test requires the standard libclang extractor")
    (tmp_path / "kernel.cpp").write_text(
        "void dut(int *input) {\n"
        "#pragma HLS DEPENDENCE variable=missing inter false\n"
        "  input[0] += 1;\n"
        "}\n",
        encoding="utf-8",
    )
    manifest = minimal_manifest(
        "test.dependence_missing_operand", "dependence missing operand",
        "dut", "kernel.cpp",
    )
    manifest.toolchains = [ToolchainContext(
        id="amd.vitis_hls.2024_2", vendor="amd", name="vitis_hls",
        version="2024.2",
    )]
    project = Project(GraphBundle.create(tmp_path, manifest))
    install_reviewed_builtin_packs(project.bundle)
    indexed = project.index()
    assert indexed.success
    graph = project.service(indexed.snapshot_id).graph()
    directive = next(item for item in graph.entities.values()
                     if item.kind == "hls.directive" and item.name == "DEPENDENCE")
    assert directive.completeness.value == "ambiguous"
    assert directive.attrs["state"] == "requested"
    assert "variable_id" not in directive.attrs
    assert not any(
        item.subject_id == directive.id
        and item.predicate == "directive.declared_selected"
        for item in project.bundle.store.observations(indexed.snapshot_id)
    )
    assert any(
        item.code == "directive.unresolved_operand" and item.subject_id == directive.id
        for item in project.bundle.store.active_diagnostics(indexed.snapshot_id)
    )
    contexts = HybridRetriever(
        project.bundle, indexed.snapshot_id,
    )._binding_target_contexts(graph, set(graph.entities))[
        ("directive_kind", "DEPENDENCE")
    ]
    bindings = [
        item for item in project.bundle.store.knowledge_bindings()
        if item.target_kind == "directive_kind" and item.target == "DEPENDENCE"
    ]
    target = {"directive_kind": {"DEPENDENCE"}}
    assert not any(
        HybridRetriever._binding_applicable(binding, context, target)
        for binding in bindings for context in contexts
    )


def test_dependence_without_enclosing_scope_does_not_bind_nearby_function(
    tmp_path: Path,
) -> None:
    if not LibClangExtractor.available():
        pytest.skip("the DEPENDENCE scope test requires the standard libclang extractor")
    (tmp_path / "kernel.cpp").write_text(
        "#pragma HLS DEPENDENCE variable=input inter false\n"
        "void dut(int *input) {\n"
        "  for (int i = 0; i < 4; ++i) input[i] += 1;\n"
        "}\n",
        encoding="utf-8",
    )
    manifest = minimal_manifest(
        "test.dependence_no_scope", "dependence without scope", "dut", "kernel.cpp",
    )
    manifest.toolchains = [ToolchainContext(
        id="amd.vitis_hls.2024_2", vendor="amd", name="vitis_hls",
        version="2024.2",
    )]
    project = Project(GraphBundle.create(tmp_path, manifest))
    install_reviewed_builtin_packs(project.bundle)
    indexed = project.index()
    assert indexed.success
    graph = project.service(indexed.snapshot_id).graph()
    directive = next(item for item in graph.entities.values()
                     if item.kind == "hls.directive" and item.name == "DEPENDENCE")
    assert directive.completeness.value == "ambiguous"
    assert not {"scope_id", "scope_kind", "function_id", "loop_id", "variable_id"}.intersection(
        directive.attrs
    )
    assert any(
        item.code == "directive.unresolved_scope" and item.subject_id == directive.id
        for item in project.bundle.store.active_diagnostics(indexed.snapshot_id)
    )
    contexts = HybridRetriever(
        project.bundle, indexed.snapshot_id,
    )._binding_target_contexts(graph, set(graph.entities))[
        ("directive_kind", "DEPENDENCE")
    ]
    bindings = [
        item for item in project.bundle.store.knowledge_bindings()
        if item.target_kind == "directive_kind" and item.target == "DEPENDENCE"
    ]
    target = {"directive_kind": {"DEPENDENCE"}}
    assert not any(
        HybridRetriever._binding_applicable(binding, context, target)
        for binding in bindings for context in contexts
    )


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
    assert llvm_rows["feature.operation_histogram"]["metadata"] == {
        "operation_histogram_domain_complete": True,
        "operation_histogram_provenance": "typed_ir_entity_evidence.v1",
        "operation_histogram_qualification": "opcode_qualified",
        "operation_histogram_schema": "llvm.opcode_histogram.v1",
        "semantic": "histogram_of_explicit_ir_operation_entities",
        "unknown_is_zero": False,
    }
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

    retriever = HybridRetriever(project.bundle, graph.snapshot_id)
    target_contexts = retriever._binding_target_contexts(
        graph, set(graph.entities),
    )
    for predicate in complete_llvm_predicates:
        context = next(
            item for item in target_contexts[("predicate", predicate)]
            if item.get("derivation_instance_id")
            == {llvm_rows[predicate]["id"].casefold()}
        )
        assert "aggregate_evidence_qualified" not in context
        assert "aggregate_evidence_identity" not in context
        assert "semantic_attestation_identity" not in context

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
    mlir_derivation = next(
        item for item in derivations
        if item["subject_id"] == mlir_function.id
        and item["predicate"] == "feature.operation_histogram"
    )
    mlir_context = next(
        item for item in target_contexts[
            ("predicate", "feature.operation_histogram")
        ]
        if item.get("derivation_instance_id")
        == {mlir_derivation["id"].casefold()}
    )
    assert "semantic_attestation_identity" not in mlir_context
    assert "semantic_artifact_evidence_qualified" not in mlir_context
    assert not any(
        item.target_kind == "predicate"
        and item.target == "feature.operation_histogram"
        and item.required_context.get("stage") == "mlir"
        for item in project.bundle.store.knowledge_bindings()
    )

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
    assert config_values["directive.declared_selected"].value == {"factor": 2}
    assert config_values["directive.declared_selected"].metadata["tool_applied"] is False

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
    derivations = {
        item["predicate"]: item for item in
        project.bundle.store.derivations(result.snapshot_id)
        if item["subject_id"] == function.id
    }
    index_contract = {
        "index_histogram_schema": (
            "llvm.explicit_index_operand_kind_histogram.v1"
        ),
        "index_histogram_provenance": "typed_ir_entity_evidence.v1",
        "index_operand_definition": (
            "llvm.gep_extract_insert_explicit_operand.v1"
        ),
        "index_histogram_domain_complete": True,
    }
    assert all(
        derivations["feature.index_histogram"]["metadata"].get(key) == value
        for key, value in index_contract.items()
    )
    bitwidth_contract = {
        "bitwidth_schema": (
            "llvm.explicit_integer_width_occurrence_histogram.v1"
        ),
        "bitwidth_provenance": "typed_ir_entity_evidence.v1",
        "bitwidth_definition": "llvm.explicit_integer_type_occurrence.v1",
        "bitwidth_domain_complete": True,
    }
    assert all(
        derivations["feature.bitwidth"]["metadata"].get(key) == value
        for key, value in bitwidth_contract.items()
    )
    memory_contract = {
        "memory_access_schema": "llvm.memory_access_kind_histogram.v1",
        "memory_access_provenance": "typed_ir_entity_evidence.v1",
        "memory_access_opcode_definition": (
            "llvm.load_store_gep_atomic_fence.v1"
        ),
        "memory_access_domain_complete": True,
    }
    assert all(
        derivations["feature.memory_access"]["metadata"].get(key) == value
        for key, value in memory_contract.items()
    )
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
    operation_derivation = next(
        item for item in project.bundle.store.derivations(result.snapshot_id)
        if item["subject_id"] == function.id
        and item["predicate"] == "feature.operation_histogram"
    )
    assert operation_derivation["metadata"]["operation_histogram_schema"] == (
        "mlir.dialect_qualified_opcode_histogram.v1"
    )
    assert operation_derivation["metadata"][
        "operation_histogram_provenance"
    ] == "typed_ir_entity_evidence.v1"
    assert operation_derivation["metadata"][
        "operation_histogram_domain_complete"
    ] is True

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


def test_mlir_mapping_rules_close_only_to_source_ast_and_nonhardware_projection(
    tmp_path: Path,
) -> None:
    if not LibClangExtractor.available():
        pytest.skip("the source mapping contract requires the standard libclang extractor")
    (tmp_path / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    (tmp_path / "dut.mlir").write_text(
        """module {
  func.func @dut() {
    %0 = arith.constant 0 : i32 loc("kernel.cpp":1:1)
    scf.for %i = 0 to 2 step 1 {
      func.return
    }
  }
}
""",
        encoding="utf-8",
    )
    manifest = minimal_manifest(
        "test.mlir_mapping_contract", "MLIR mapping contract", "dut", "kernel.cpp",
    )
    manifest.artifact_paths.append({
        "path": "dut.mlir", "kind": "ir.mlir", "role": "hls_ir",
        "access": "project", "license": "Apache-2.0",
        "metadata": {
            "artifact_revision": "sha256:fixture-mlir",
            "language_spec_contracts": [{
                "family": "mlir",
                "revision": (
                    "git-429c88d37f1f02e68ebc1fc7b0da4511ce6407e3"
                ),
                "compatibility_contract": (
                    "hlsgraph.mlir.language_spec_compatibility.v1"
                ),
            }],
        },
    })
    project = Project(GraphBundle.create(tmp_path, manifest))
    result = project.index()
    assert result.success
    graph = project.service(result.snapshot_id).graph()

    mapping = next(
        item for item in graph.relations.values()
        if item.kind == "cross.maps_to" and item.stage == "mlir"
    )
    assert graph.entities[mapping.src].kind == "ir.mlir.operation"
    assert graph.entities[mapping.dst].kind == "hls.kernel"
    assert graph.entities[mapping.dst].stage == "ast"
    assert mapping.mapping_kind == "mlir.location"
    assert mapping.attrs["hardware_topology"] is False
    assert mapping.attrs["mapping_provenance"] == "mlir.location_anchor"
    assert mapping.attrs["mapping_resolution"] == "unique_exact"
    assert mapping.attrs["mapping_resolution_contract"] == (
        "hlsgraph.mlir_location_resolution.v1"
    )
    assert mapping.attrs["mapping_candidate_count"] == 1
    assert mapping.attrs["mapping_ambiguous"] is False
    assert mapping.attrs["mapping_unresolved"] is False
    assert mapping.attrs["mapping_redacted"] is False
    assert mapping.attrs["resolved_target_id"] == mapping.dst
    locations = [
        anchor for anchor in mapping.anchors
        if anchor.mapping_kind == "mlir.filelinecol"
    ]
    assert len(locations) == 1
    assert mapping.attrs["typed_source_anchor_identity"] == stable_hash(
        locations[0]
    )
    target_anchor = next(
        anchor for anchor in graph.entities[mapping.dst].anchors
        if stable_hash(anchor)
        == mapping.attrs["resolved_target_anchor_identity"]
    )
    assert any(
        stable_hash(anchor) == stable_hash(target_anchor)
        for anchor in mapping.anchors
    )
    assert mapping.attrs["source_anchor_identity_contract"] == (
        "hlsgraph.source_anchor_identity.v1"
    )

    retriever = HybridRetriever(project.bundle, result.snapshot_id)
    contexts = retriever._binding_target_contexts(
        graph, set(graph.entities),
    )[("relation_kind", "cross.maps_to")]
    context = next(
        item for item in contexts
        if item["relation_instance_id"] == {mapping.id.casefold()}
    )
    assert "artifact_revision" not in context
    assert "semantic_attestation_identity" not in context
    assert "semantic_artifact_evidence_qualified" not in context
    assert "language_spec_compatibility_contract" not in context
    assert context["unique_mlir_location_mapping_resolved"] == {"true"}
    assert context["typed_source_anchor_identity"] == {
        stable_hash(locations[0])
    }
    assert context["resolved_target_anchor_identity"] == {
        stable_hash(target_anchor)
    }
    assert not any(
        item.target_kind == "relation_kind" and item.target == "cross.maps_to"
        for item in project.bundle.store.knowledge_bindings()
    )

    hardware_projection = next(
        item for item in graph.relations.values()
        if item.kind == "cross.projects_to"
    )
    assert graph.entities[hardware_projection.dst].kind == "hls.loop"
    assert hardware_projection.attrs["hardware_projection"] is True
    assert hardware_projection.attrs["hardware_topology"] is False
    assert not any(
        item.target_kind == "relation_kind"
        and item.target == "cross.projects_to"
        for item in project.bundle.store.knowledge_bindings()
    )


def test_mlir_mapping_zero_candidates_and_redacted_location_are_diagnostic_only(
    tmp_path: Path,
) -> None:
    if not LibClangExtractor.available():
        pytest.skip("the source mapping contract requires the standard libclang extractor")
    (tmp_path / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    external_location = f"{chr(67)}:/private/kernel.cpp"
    (tmp_path / "dut.mlir").write_text(
        f"""module {{
  func.func @dut() {{
    %0 = arith.constant 0 : i32 loc("kernel.cpp":9:1)
    %1 = arith.constant 1 : i32 loc("{external_location}":1:1)
    func.return
  }}
}}
""",
        encoding="utf-8",
    )
    manifest = minimal_manifest(
        "test.mlir_mapping_unresolved", "MLIR unresolved mapping", "dut",
        "kernel.cpp",
    )
    manifest.artifact_paths.append({
        "path": "dut.mlir", "kind": "ir.mlir", "role": "hls_ir",
        "access": "project", "license": "Apache-2.0",
        "metadata": {
            "artifact_revision": "sha256:fixture-mlir",
            "language_spec_contracts": [{
                "family": "mlir",
                "revision": (
                    "git-429c88d37f1f02e68ebc1fc7b0da4511ce6407e3"
                ),
                "compatibility_contract": (
                    "hlsgraph.mlir.language_spec_compatibility.v1"
                ),
            }],
        },
    })
    project = Project(GraphBundle.create(tmp_path, manifest))
    result = project.index()
    assert result.success
    graph = project.service(result.snapshot_id).graph()
    mlir_operations = {
        item.id for item in graph.entities.values()
        if item.kind == "ir.mlir.operation"
    }
    assert not any(
        relation.kind == "cross.maps_to" and relation.stage == "mlir"
        and relation.src in mlir_operations
        for relation in graph.relations.values()
    )
    unresolved = [
        item for item in project.bundle.store.diagnostics(result.snapshot_id)
        if item.code == "mapping.unresolved_mlir_location"
    ]
    assert len(unresolved) == 2
    assert all(item.metadata["mapping_resolution"] == "unresolved"
               and item.metadata["mapping_candidate_count"] == 0
               and item.metadata["mapping_ambiguous"] is False
               and item.metadata["mapping_unresolved"] is True
               for item in unresolved)
    assert {item.metadata["mapping_redacted"] for item in unresolved} == {
        False, True,
    }


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


def test_mixed_ir_operation_histogram_has_no_qualified_schema() -> None:
    from hlsgraph.extract.static_features import derive_static_features

    graph = CanonicalGraph("snapshot.mixed_histogram")
    scope = graph.add_entity(Entity(
        "hls.function", "dut", graph.snapshot_id, stage="hls_ir",
        attrs={"static_feature_domain_complete": True},
    ))
    mlir = graph.add_entity(Entity(
        "ir.mlir.operation", "arith.addi", graph.snapshot_id, stage="mlir",
        attrs={"operation": "arith.addi", "dialect": "arith"},
    ))
    llvm = graph.add_entity(Entity(
        "ir.llvm.operation", "add", graph.snapshot_id, stage="llvm",
        attrs={"opcode": "add"},
    ))
    for operation in (mlir, llvm):
        graph.add_relation(Relation(
            operation.id, scope.id, "cross.maps_to", graph.snapshot_id,
            stage=operation.stage,
        ))
    extraction = ExtractionResult(graph=graph)
    derive_static_features(extraction)
    histogram = next(
        item for item in extraction.derivations
        if item.subject_id == scope.id
        and item.predicate == "feature.operation_histogram"
    )
    assert histogram.value == {"add": 1, "arith.addi": 1}
    assert histogram.completeness == "complete"
    assert histogram.metadata["operation_histogram_qualification"] == (
        "unknown_or_mixed"
    )
    assert "operation_histogram_schema" not in histogram.metadata
    context: dict[str, set[str]] = {"stage": {"llvm"}, "ir": {"llvm"}}
    HybridRetriever.__new__(HybridRetriever)._context_derivation_evidence(
        context, json_ready(histogram), graph, {},
    )
    assert "dialect_qualified_operation_histogram_present" not in context
    assert "opcode_qualified_operation_histogram_present" not in context


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


def test_plugin_cannot_self_attest_open_ir_spec_compatibility(tmp_path):
    (tmp_path / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    (tmp_path / "dut.ll").write_text(
        "define void @dut() {\nentry:\n  ret void\n}\n", encoding="utf-8",
    )
    manifest = minimal_manifest(
        "test.untrusted_semantic_claim", "untrusted claim", "dut", "kernel.cpp",
    )
    manifest.artifact_paths.append({
        "path": "dut.ll", "kind": "ir.llvm", "role": "llvm_ir",
        "access": "project",
    })
    bundle = GraphBundle.create(tmp_path, manifest)
    snapshot = bundle.snapshot()
    artifact = next(
        item for item in bundle.store.artifacts(snapshot.id)
        if item.kind == "ir.llvm"
    )
    context = ExtractionContext(
        project_root=tmp_path, manifest=bundle.manifest, snapshot=snapshot,
        artifacts={artifact.id: artifact},
    )

    class SpoofedExtractor:
        name = "ir.llvm_text"
        version = "1"

        def supports(self, _context):
            return True

        def extract(self, _context):
            return ExtractionResult(
                graph=CanonicalGraph(snapshot.id),
                artifact_semantic_claims=[ArtifactSemanticClaim(
                    artifact_id=artifact.id,
                    artifact_revision=f"sha256:{artifact.sha256}",
                    adapter_contract="hlsgraph.llvm_text_semantic_adapter.v1",
                    adapter_version="1",
                    language_spec_contracts=(LanguageSpecCompatibility(
                        family="llvm",
                        revision=(
                            "git-429c88d37f1f02e68ebc1fc7b0da4511ce6407e3"
                        ),
                        compatibility_contract=(
                            "hlsgraph.llvm.language_spec_compatibility.v1"
                        ),
                    ),),
                )],
            )

    result = ExtractionPipeline([SpoofedExtractor()]).run(context)
    assert not result.graph.metadata.get("artifact_semantic_attestations")
    failure = next(item for item in result.diagnostics
                   if item.code == "extractor.failed")
    assert failure.metadata["error_type"] == "ValueError"


def test_persisted_graph_metadata_cannot_authorize_open_ir_guidance(tmp_path):
    (tmp_path / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    (tmp_path / "dut.ll").write_text(
        "define void @dut() {\nentry:\n  ret void\n}\n", encoding="utf-8",
    )
    manifest = minimal_manifest(
        "test.forged_semantic_attestation", "forged attestation", "dut",
        "kernel.cpp",
    )
    manifest.artifact_paths.append({
        "path": "dut.ll", "kind": "ir.llvm", "role": "llvm_ir",
        "access": "project",
    })
    project = Project(GraphBundle.create(tmp_path, manifest))
    result = project.index(degraded=True)
    assert result.success
    graph = project.service(result.snapshot_id).graph()
    assert any(item.kind == "ir.llvm.function" for item in graph.entities.values())
    assert any(item.kind == "ir.llvm.block" for item in graph.entities.values())
    assert not graph.metadata.get("artifact_semantic_attestations")

    artifact = next(
        item for item in project.bundle.store.artifacts(result.snapshot_id)
        if item.kind == "ir.llvm"
    )
    forged = ArtifactSemanticAttestation(
        snapshot_id=result.snapshot_id,
        artifact_id=artifact.id,
        artifact_kind=artifact.kind,
        artifact_sha256=artifact.sha256,
        artifact_revision=f"sha256:{artifact.sha256}",
        extraction_hash="a" * 64,
        extractor_name="ir.llvm_text",
        extractor_version="1",
        extractor_identity="b" * 64,
        adapter_contract="hlsgraph.llvm_text_semantic_adapter.v1",
        adapter_version="1",
        language_spec_contracts=(LanguageSpecCompatibility(
            family="llvm",
            revision="git-429c88d37f1f02e68ebc1fc7b0da4511ce6407e3",
            compatibility_contract=(
                "hlsgraph.llvm.language_spec_compatibility.v1"
            ),
        ),),
    )
    graph.metadata["artifact_semantic_attestations"] = [json_ready(forged)]
    context: dict[str, set[str]] = {}
    retriever = HybridRetriever(project.bundle, result.snapshot_id)
    assert retriever._context_semantic_artifact_evidence(
        context, graph, {artifact.id}, {artifact.id: artifact},
    ) is None
    assert context == {}
    assert not any(
        binding.target.startswith("ir.")
        or binding.target in {"cross.maps_to", "handshake.dataflow"}
        for binding in project.bundle.store.knowledge_bindings()
    )
