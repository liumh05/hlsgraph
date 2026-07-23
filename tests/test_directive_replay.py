from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from hlsgraph.bundle import GraphBundle
from hlsgraph.extract.base import ExtractionContext, ExtractionPipeline
from hlsgraph.extract.directive_identity import (
    DIRECTIVE_IDENTITY_FIELDS,
    bind_directive_identity,
    directive_identity_metadata,
)
from hlsgraph.extract.directives import ExternalDirectiveExtractor
from hlsgraph.extract.source import LibClangExtractor, RegexSourceExtractor
from hlsgraph.graph import CanonicalGraph
from hlsgraph.manifest import minimal_manifest
from hlsgraph.model import (
    AuthorityClass,
    DiagnosticSeverity,
    Entity,
    Observation,
    Relation,
    SourceAnchor,
    ToolchainContext,
)
from hlsgraph.retrieval import HybridRetriever


def _manifest(root: Path, source: str, *, tcl: str | None = None):
    (root / "kernel.cpp").write_text(source, encoding="utf-8")
    manifest = minimal_manifest(
        f"test.directive.replay.{root.name}",
        "directive replay",
        "dut",
        "kernel.cpp",
    )
    manifest.toolchains = [ToolchainContext(
        id="amd.vitis_hls.2024_2",
        vendor="amd",
        name="vitis_hls",
        version="2024.2",
    )]
    if tcl is not None:
        (root / "directives.tcl").write_text(tcl, encoding="utf-8")
        manifest.build.tcl_files = ["directives.tcl"]
        manifest.artifact_paths.append({
            "path": "directives.tcl",
            "kind": "config.tcl",
            "role": "directive",
            "access": "private",
        })
    return manifest


def _fixed_records(root: Path, source: str, *, tcl: str | None = None):
    if not LibClangExtractor.available():
        pytest.skip("directive replay tests require the standard libclang parser")
    manifest = _manifest(root, source, tcl=tcl)
    bundle = GraphBundle.create(root, manifest)
    snapshot = bundle.snapshot()
    artifacts = {
        item.id: item for item in bundle.store.artifacts(snapshot.id)
    }
    result = ExtractionPipeline([
        LibClangExtractor(), ExternalDirectiveExtractor(),
    ]).run(ExtractionContext(
        project_root=root,
        manifest=manifest,
        snapshot=snapshot,
        artifacts=artifacts,
        allow_degraded=False,
    ))
    assert not any(
        item.severity in {DiagnosticSeverity.ERROR, DiagnosticSeverity.CRITICAL}
        for item in result.diagnostics
    )
    return bundle, snapshot, result


def _requested(result, directive_id: str) -> Observation:
    return next(
        item for item in result.observations
        if item.subject_id == directive_id
        and item.predicate == "directive.requested"
    )


def _context(bundle, snapshot, graph, directive: Entity):
    contexts = HybridRetriever(
        bundle, snapshot.id,
    )._binding_target_contexts(graph, set(graph.entities))[
        ("directive_kind", directive.name)
    ]
    return next(
        item for item in contexts
        if item.get("directive_instance_id") == {directive.id.casefold()}
    )


def _requested_context(bundle, snapshot, graph, directive: Entity):
    contexts = HybridRetriever(
        bundle, snapshot.id,
    )._binding_target_contexts(graph, set(graph.entities))[
        ("predicate", "directive.requested")
    ]
    return next(
        item for item in contexts
        if item.get("directive_instance_id") == {directive.id.casefold()}
    )


def _persist(bundle, graph, observations) -> None:
    bundle.store.save_graph(graph)
    bundle.store.add_observations(list(observations))


def _retarget(
    graph: CanonicalGraph,
    directive: Entity,
    target: Entity,
    observation: Observation,
    *,
    operand_target: Entity | None = None,
) -> Observation:
    annotation = next(
        item for item in graph.relations.values()
        if item.kind == "hls.annotates" and item.src == directive.id
    )
    del graph.relations[annotation.id]
    for key in DIRECTIVE_IDENTITY_FIELDS:
        directive.attrs.pop(key, None)
    bind_directive_identity(
        directive,
        target,
        scope_resolution="source_ast",
        operand_target=operand_target,
    )
    graph.add_relation(Relation(
        directive.id,
        target.id,
        "hls.annotates",
        directive.snapshot_id,
        authority=AuthorityClass.DECLARED_CONSTRAINT,
        stage="source",
        attrs={
            "scope_node_id": target.id,
            "scope_resolution": "source_ast",
        },
        anchors=list(annotation.anchors),
    ))
    return replace(
        observation,
        id="",
        metadata={
            "directive_kind": directive.name,
            **directive_identity_metadata(directive),
        },
    )


def test_external_interface_preserves_component_and_port_scope(
    tmp_path: Path,
) -> None:
    source = (
        "void helper(int *input) { input[0] = 0; }\n"
        "void dut(int *input) { input[0] += 1; }\n"
    )
    good_root = tmp_path / "good"
    good_root.mkdir()
    bundle, snapshot, result = _fixed_records(
        good_root, source,
        tcl="set_directive_interface -mode m_axi dut input\n",
    )
    directive = next(
        item for item in result.graph.entities.values()
        if item.kind == "hls.directive"
        and item.name == "INTERFACE"
        and item.attrs.get("origin") == "tcl"
    )
    port = result.graph.entities[directive.attrs["scope_id"]]
    assert port.kind == "hls.port"
    owner = next(
        relation for relation in result.graph.relations.values()
        if relation.kind == "hls.contains" and relation.dst == port.id
    )
    assert result.graph.entities[owner.src].kind == "hls.kernel"
    assert result.graph.entities[owner.src].name == "dut"
    _persist(bundle, result.graph, result.observations)
    context = _context(bundle, snapshot, result.graph, directive)
    assert context["port_owner_id"] == {owner.src.casefold()}
    assert context["configured_component_id"] == {owner.src.casefold()}
    assert context["port_ownership_qualified"] == {
        "derived_from_unique_current_component_port_v1",
    }

    bad_root = tmp_path / "bad"
    bad_root.mkdir()
    _bad_bundle, _bad_snapshot, bad = _fixed_records(
        bad_root, source,
        tcl="set_directive_interface -mode m_axi helper input\n",
    )
    rejected = next(
        item for item in bad.graph.entities.values()
        if item.kind == "hls.directive"
        and item.name == "INTERFACE"
        and item.attrs.get("origin") == "tcl"
    )
    assert str(rejected.completeness) == "ambiguous"
    assert "scope_id" not in rejected.attrs
    assert not any(
        item.subject_id == rejected.id
        and item.predicate == "directive.requested"
        for item in bad.observations
    )


def test_direct_save_cannot_invent_a_source_directive(tmp_path: Path) -> None:
    bundle, snapshot, result = _fixed_records(tmp_path, "void dut() {}\n")
    graph = result.graph
    kernel = next(
        item for item in graph.entities.values() if item.kind == "hls.kernel"
    )
    artifact = next(
        item for item in bundle.store.artifacts(snapshot.id)
        if item.uri == "kernel.cpp"
    )
    anchor = SourceAnchor(
        artifact_id=artifact.id,
        start_line=1,
        start_column=1,
        end_line=1,
        end_column=14,
    )
    directive = Entity(
        "hls.directive",
        "PIPELINE",
        snapshot.id,
        qualified_name="kernel.cpp:1:PIPELINE",
        stage="source",
        authority=AuthorityClass.DECLARED_CONSTRAINT,
        attrs={
            "directive_kind": "PIPELINE",
            "options": {"ii": 1},
            "origin": "source_pragma",
            "precedence": 10,
            "state": "selected_declared",
            "directive_source_declaration_qualified": (
                "derived_from_current_directive_source_declaration_v1"
            ),
        },
        anchors=[anchor],
    )
    bind_directive_identity(directive, kernel, scope_resolution="source_ast")
    graph.add_entity(directive)
    graph.add_relation(Relation(
        directive.id,
        kernel.id,
        "hls.annotates",
        snapshot.id,
        stage="source",
        authority=AuthorityClass.DECLARED_CONSTRAINT,
        attrs={
            "scope_node_id": kernel.id,
            "scope_resolution": "source_ast",
        },
        anchors=[anchor],
    ))
    requested = Observation(
        snapshot.id,
        directive.id,
        "directive.requested",
        {"ii": 1},
        "source",
        AuthorityClass.DECLARED_CONSTRAINT,
        artifact_id=artifact.id,
        anchor=anchor,
        metadata={
            "directive_kind": "PIPELINE",
            **directive_identity_metadata(directive),
            "requested_directive_present": True,
        },
    )
    _persist(bundle, graph, [requested])

    context = _context(bundle, snapshot, graph, directive)
    assert "directive_source_declaration_qualified" not in context
    assert "requested_directive_present" not in _requested_context(
        bundle, snapshot, graph, directive,
    )


def test_replay_rejects_sibling_scope_even_when_records_are_consistent(
    tmp_path: Path,
) -> None:
    source = (
        "void dut() {\n"
        "#pragma HLS pipeline II=1\n"
        "  for (int i = 0; i < 2; ++i) {}\n"
        "  for (int j = 0; j < 2; ++j) {}\n"
        "}\n"
    )
    bundle, snapshot, result = _fixed_records(tmp_path, source)
    graph = result.graph
    directive = next(
        item for item in graph.entities.values()
        if item.kind == "hls.directive"
    )
    original_scope = directive.attrs["scope_id"]
    sibling = next(
        item for item in graph.entities.values()
        if item.kind == "hls.loop" and item.id != original_scope
    )
    forged = _retarget(
        graph, directive, sibling, _requested(result, directive.id),
    )
    _persist(bundle, graph, [forged])

    context = _context(bundle, snapshot, graph, directive)
    assert context["scope_id"] == {sibling.id.casefold()}
    assert "directive_source_declaration_qualified" not in context


def test_replay_rejects_forged_options(tmp_path: Path) -> None:
    bundle, snapshot, result = _fixed_records(
        tmp_path,
        "void dut() {\n#pragma HLS pipeline II=1\n}\n",
    )
    graph = result.graph
    directive = next(
        item for item in graph.entities.values()
        if item.kind == "hls.directive"
    )
    requested = _requested(result, directive.id)
    directive.attrs["options"] = {"ii": 7}
    forged = replace(requested, id="", value={"ii": 7})
    _persist(bundle, graph, [forged])

    context = _context(bundle, snapshot, graph, directive)
    assert "directive_source_declaration_qualified" not in context
    assert "requested_directive_present" not in _requested_context(
        bundle, snapshot, graph, directive,
    )


def test_replay_rejects_forged_array_operand(tmp_path: Path) -> None:
    bundle, snapshot, result = _fixed_records(
        tmp_path,
        "void dut(int *left, int *right) {\n"
        "#pragma HLS ARRAY_PARTITION variable=left complete\n"
        "  left[0] += right[0];\n"
        "}\n",
    )
    graph = result.graph
    directive = next(
        item for item in graph.entities.values()
        if item.kind == "hls.directive"
    )
    right = next(
        item for item in graph.entities.values()
        if item.kind == "hls.port" and item.name == "right"
    )
    directive.attrs["options"] = {"variable": "right", "complete": True}
    forged = _retarget(
        graph, directive, right, _requested(result, directive.id),
    )
    forged = replace(forged, id="", value=dict(directive.attrs["options"]))
    _persist(bundle, graph, [forged])

    context = _context(bundle, snapshot, graph, directive)
    assert "directive_source_declaration_qualified" not in context
    assert "directive_operand_linked" not in context
    assert "directive_operand_identity" not in context


def test_replay_rejects_forged_dependence_operand(tmp_path: Path) -> None:
    bundle, snapshot, result = _fixed_records(
        tmp_path,
        "void dut(int *left, int *right) {\n"
        "#pragma HLS DEPENDENCE variable=left inter=false\n"
        "  left[0] += right[0];\n"
        "}\n",
    )
    graph = result.graph
    directive = next(
        item for item in graph.entities.values()
        if item.kind == "hls.directive"
    )
    scope = graph.entities[directive.attrs["scope_id"]]
    right = next(
        item for item in graph.entities.values()
        if item.kind == "hls.port" and item.name == "right"
    )
    directive.attrs["options"] = {"variable": "right", "inter": False}
    forged = _retarget(
        graph,
        directive,
        scope,
        _requested(result, directive.id),
        operand_target=right,
    )
    forged = replace(forged, id="", value=dict(directive.attrs["options"]))
    _persist(bundle, graph, [forged])

    context = _context(bundle, snapshot, graph, directive)
    assert "dependence_operand_resolved" not in context
    assert "directive_operand_identity" not in context
    assert "requested_directive_present" not in _requested_context(
        bundle, snapshot, graph, directive,
    )


def test_regex_degraded_graph_never_activates_replay_markers(
    tmp_path: Path,
) -> None:
    manifest = _manifest(
        tmp_path,
        "void dut() {\n#pragma HLS pipeline II=1\n"
        "  for (int i = 0; i < 2; ++i) {}\n}\n",
    )
    bundle = GraphBundle.create(tmp_path, manifest)
    snapshot = bundle.snapshot()
    artifacts = {
        item.id: item for item in bundle.store.artifacts(snapshot.id)
    }
    result = ExtractionPipeline([RegexSourceExtractor()]).run(ExtractionContext(
        project_root=tmp_path,
        manifest=manifest,
        snapshot=snapshot,
        artifacts=artifacts,
        allow_degraded=True,
    ))
    directive = next(
        item for item in result.graph.entities.values()
        if item.kind == "hls.directive"
    )
    _persist(bundle, result.graph, [
        item for item in result.observations
        if item.predicate == "directive.requested"
    ])

    context = _context(bundle, snapshot, result.graph, directive)
    assert directive.attrs["scope_resolution"] == "regex_degraded"
    assert "directive_source_declaration_qualified" not in context
    assert "requested_directive_present" not in _requested_context(
        bundle, snapshot, result.graph, directive,
    )


def test_missing_libclang_environment_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, snapshot, result = _fixed_records(
        tmp_path,
        "void dut() {\n#pragma HLS pipeline II=1\n}\n",
    )
    directive = next(
        item for item in result.graph.entities.values()
        if item.kind == "hls.directive"
    )
    _persist(bundle, result.graph, [_requested(result, directive.id)])
    monkeypatch.setattr(
        LibClangExtractor,
        "available",
        staticmethod(lambda: False),
    )

    context = _context(bundle, snapshot, result.graph, directive)
    assert "directive_source_declaration_qualified" not in context
    assert "requested_directive_present" not in _requested_context(
        bundle, snapshot, result.graph, directive,
    )


def test_any_snapshot_compiler_input_drift_invalidates_directive_replay(
    tmp_path: Path,
) -> None:
    if not LibClangExtractor.available():
        pytest.skip("directive replay tests require the standard libclang parser")
    (tmp_path / "defs.hpp").write_text("#define LIMIT 2\n", encoding="utf-8")
    manifest = _manifest(
        tmp_path,
        '#include "defs.hpp"\n'
        "void dut() {\n#pragma HLS pipeline II=1\n"
        "  for (int i = 0; i < LIMIT; ++i) {}\n}\n",
    )
    manifest.artifact_paths.append({
        "path": "defs.hpp",
        "kind": "source.hpp",
        "role": "header",
        "access": "private",
    })
    bundle = GraphBundle.create(tmp_path, manifest)
    snapshot = bundle.snapshot()
    artifacts = {
        item.id: item for item in bundle.store.artifacts(snapshot.id)
    }
    result = ExtractionPipeline([
        LibClangExtractor(), ExternalDirectiveExtractor(),
    ]).run(ExtractionContext(
        project_root=tmp_path,
        manifest=manifest,
        snapshot=snapshot,
        artifacts=artifacts,
        allow_degraded=False,
    ))
    assert not any(
        item.severity in {DiagnosticSeverity.ERROR, DiagnosticSeverity.CRITICAL}
        for item in result.diagnostics
    )
    directive = next(
        item for item in result.graph.entities.values()
        if item.kind == "hls.directive"
    )
    _persist(bundle, result.graph, [_requested(result, directive.id)])
    retriever = HybridRetriever(bundle, snapshot.id)

    def contexts():
        return retriever._binding_target_contexts(
            result.graph, set(result.graph.entities),
        )

    qualified = contexts()
    directive_context = next(
        item for item in qualified[("directive_kind", directive.name)]
        if item.get("directive_instance_id") == {directive.id.casefold()}
    )
    requested_context = next(
        item for item in qualified[("predicate", "directive.requested")]
        if item.get("directive_instance_id") == {directive.id.casefold()}
    )
    assert "directive_source_declaration_qualified" in directive_context
    assert requested_context["requested_directive_present"] == {"true"}

    (tmp_path / "defs.hpp").write_text("#define LIMIT 3\n", encoding="utf-8")
    rejected_contexts = contexts()
    rejected = next(
        item for item in rejected_contexts[("directive_kind", directive.name)]
        if item.get("directive_instance_id") == {directive.id.casefold()}
    )
    rejected_requested = next(
        item for item in rejected_contexts[("predicate", "directive.requested")]
        if item.get("directive_instance_id") == {directive.id.casefold()}
    )
    assert "directive_source_declaration_qualified" not in rejected
    assert "requested_directive_present" not in rejected_requested


def test_external_directive_is_replayed_and_option_tamper_is_rejected(
    tmp_path: Path,
) -> None:
    bundle, snapshot, result = _fixed_records(
        tmp_path,
        "void dut() {}\n",
        tcl="set_directive_pipeline -II 2 dut\n",
    )
    graph = result.graph
    directive = next(
        item for item in graph.entities.values()
        if item.kind == "hls.directive" and item.attrs.get("origin") == "tcl"
    )
    requested = _requested(result, directive.id)
    _persist(bundle, graph, [requested])
    qualified = _context(bundle, snapshot, graph, directive)
    assert qualified["scope_resolution"] == {"external_exact"}
    assert qualified["directive_source_declaration_qualified"] == {
        "derived_from_current_directive_source_declaration_v1"
    }

    directive.attrs["options"] = {"ii": 9}
    # Use a fresh bundle so the immutable ledger does not contain two requests.
    other = tmp_path / "tampered"
    other.mkdir()
    forged_bundle, forged_snapshot, forged_result = _fixed_records(
        other,
        "void dut() {}\n",
        tcl="set_directive_pipeline -II 2 dut\n",
    )
    forged_graph = forged_result.graph
    forged_directive = next(
        item for item in forged_graph.entities.values()
        if item.kind == "hls.directive" and item.attrs.get("origin") == "tcl"
    )
    forged_directive.attrs["options"] = {"ii": 9}
    forged_request = replace(
        _requested(forged_result, forged_directive.id),
        id="",
        value={"ii": 9},
    )
    _persist(forged_bundle, forged_graph, [forged_request])
    rejected = _context(
        forged_bundle, forged_snapshot, forged_graph, forged_directive,
    )
    assert "directive_source_declaration_qualified" not in rejected


def test_replay_rejects_changed_scope_or_operand_ownership_lineage(
    tmp_path: Path,
) -> None:
    bundle, snapshot, result = _fixed_records(
        tmp_path,
        "void dut(int *left) { left[0] += 1; }\n",
        tcl=(
            "set_directive_array_partition -type complete dut left\n"
        ),
    )
    graph = result.graph
    directive = next(
        item for item in graph.entities.values()
        if item.kind == "hls.directive"
        and item.name == "ARRAY_PARTITION"
        and item.attrs.get("origin") == "tcl"
    )
    _persist(bundle, graph, [_requested(result, directive.id)])
    qualified = _context(bundle, snapshot, graph, directive)
    assert "directive_source_declaration_qualified" in qualified

    operand = graph.entities[directive.attrs["scope_id"]]
    alternate = graph.add_entity(Entity(
        kind="hls.function", name="alternate", qualified_name="alternate",
        snapshot_id=snapshot.id, authority=AuthorityClass.STATIC_FACT,
        stage="ast",
    ))
    graph.add_relation(Relation(
        alternate.id, operand.id, "hls.contains", snapshot.id,
        authority=AuthorityClass.STATIC_FACT, stage="ast",
    ))
    rejected = _context(bundle, snapshot, graph, directive)
    assert "directive_source_declaration_qualified" not in rejected
    assert "directive_operand_linked" not in rejected
    assert "directive_operand_identity" not in rejected
