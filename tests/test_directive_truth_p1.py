from __future__ import annotations

from pathlib import Path

from hlsgraph.bundle import GraphBundle
from hlsgraph.extract import ExtractionContext, ExternalDirectiveExtractor
from hlsgraph.extract.directives import resolve_directives
from hlsgraph.graph import CanonicalGraph
from hlsgraph.manifest import minimal_manifest
from hlsgraph.model import (
    AuthorityClass,
    Completeness,
    Entity,
    Relation,
    Stage,
)


def _extract_tcl(tmp_path: Path, text: str):
    (tmp_path / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    (tmp_path / "directives.tcl").write_text(text, encoding="utf-8")
    manifest = minimal_manifest("test.tcl_truth", "tcl truth", "dut", "kernel.cpp")
    manifest.build.tcl_files = ["directives.tcl"]
    manifest.artifact_paths.append({
        "path": "directives.tcl", "kind": "config.tcl", "role": "hls_tcl",
        "access": "private",
    })
    bundle = GraphBundle.create(tmp_path, manifest)
    snapshot = bundle.snapshot()
    artifacts = bundle.store.artifacts(snapshot.id)
    existing = CanonicalGraph(snapshot_id=snapshot.id)
    existing.add_entity(Entity(
        kind="hls.kernel", name="dut", qualified_name="dut",
        snapshot_id=snapshot.id, authority=AuthorityClass.STATIC_FACT,
        stage=Stage.AST.value,
    ))
    context = ExtractionContext(
        project_root=tmp_path, manifest=bundle.manifest, snapshot=snapshot,
        artifacts={item.id: item for item in artifacts},
        options={"existing_graph": existing},
    )
    result = ExternalDirectiveExtractor().extract(context)
    resolve_directives(result)
    return result


def _extract_external(
    tmp_path: Path,
    text: str,
    *,
    origin: str,
    graph_factory=None,
):
    (tmp_path / "kernel.cpp").write_text(
        "void dut(int *input) { input[0] += 1; }\n",
        encoding="utf-8",
    )
    filename = "directives.tcl" if origin == "tcl" else "hlsgraph.cfg"
    (tmp_path / filename).write_text(text, encoding="utf-8")
    manifest = minimal_manifest(
        f"test.external.{origin}.{tmp_path.name}",
        "external directive grammar",
        "dut",
        "kernel.cpp",
    )
    if origin == "tcl":
        manifest.build.tcl_files = [filename]
    else:
        manifest.build.config_files = [filename]
    manifest.artifact_paths.append({
        "path": filename,
        "kind": "config.tcl" if origin == "tcl" else "config.hls",
        "role": "hls_directive",
        "access": "private",
    })
    bundle = GraphBundle.create(tmp_path, manifest)
    snapshot = bundle.snapshot()
    existing = (
        graph_factory(snapshot.id)
        if graph_factory is not None
        else CanonicalGraph(snapshot.id)
    )
    if graph_factory is None:
        existing.add_entity(Entity(
            kind="hls.kernel", name="dut", qualified_name="dut",
            snapshot_id=snapshot.id, authority=AuthorityClass.STATIC_FACT,
            stage=Stage.AST.value,
        ))
    context = ExtractionContext(
        project_root=tmp_path,
        manifest=bundle.manifest,
        snapshot=snapshot,
        artifacts={
            item.id: item for item in bundle.store.artifacts(snapshot.id)
        },
        options={"existing_graph": existing},
    )
    result = ExternalDirectiveExtractor().extract(context)
    resolve_directives(result)
    return result, existing


def _complete_directive_graph(snapshot_id: str) -> CanonicalGraph:
    graph = CanonicalGraph(snapshot_id)
    kernel = graph.add_entity(Entity(
        kind="hls.kernel", name="dut", qualified_name="dut",
        snapshot_id=snapshot_id, authority=AuthorityClass.STATIC_FACT,
        stage=Stage.AST.value,
    ))
    helper = graph.add_entity(Entity(
        kind="hls.function", name="helper", qualified_name="helper",
        snapshot_id=snapshot_id, authority=AuthorityClass.STATIC_FACT,
        stage=Stage.AST.value,
    ))
    loop = graph.add_entity(Entity(
        kind="hls.loop", name="L1", qualified_name="dut::L1",
        snapshot_id=snapshot_id, authority=AuthorityClass.STATIC_FACT,
        stage=Stage.AST.value,
    ))
    port = graph.add_entity(Entity(
        kind="hls.port", name="input", qualified_name="dut::input",
        snapshot_id=snapshot_id, authority=AuthorityClass.STATIC_FACT,
        stage=Stage.AST.value,
    ))
    memory = graph.add_entity(Entity(
        kind="hls.memory", name="buffer", qualified_name="dut::buffer",
        snapshot_id=snapshot_id, authority=AuthorityClass.STATIC_FACT,
        stage=Stage.AST.value,
    ))
    stream = graph.add_entity(Entity(
        kind="hls.stream", name="fifo", qualified_name="dut::fifo",
        snapshot_id=snapshot_id, authority=AuthorityClass.STATIC_FACT,
        stage=Stage.AST.value,
    ))
    for child in (loop, port, memory, stream):
        graph.add_relation(Relation(
            kernel.id, child.id, "hls.contains", snapshot_id,
            authority=AuthorityClass.STATIC_FACT, stage=Stage.AST.value,
        ))
    # Keep the helper in the graph so INLINE is exercised without making it a
    # second owner of a top-level port or variable.
    assert helper.id in graph.entities
    return graph


def test_tcl_only_promotes_complete_top_level_literal_commands(tmp_path: Path) -> None:
    result = _extract_tcl(
        tmp_path,
        """if {0} {
  set_directive_pipeline -II 9 dut
}
proc deferred {} {
  set_directive_unroll -factor 8 dut
}
foreach item {dut} {
  set_directive_dataflow $item
}
if {0} { set_directive_inline dut }
set_directive_stream -depth $depth dut
set_directive_array_partition -factor [expr {1 + 1}] dut
set_directive_latency -min 1 dut; set ignored 1
set_directive_pipeline -II 1 {dut}
""",
    )

    directives = [item for item in result.graph.entities.values()
                  if item.kind == "hls.directive"]
    assert [(item.name, item.attrs["options"]) for item in directives] == [
        ("PIPELINE", {"ii": 1}),
    ]
    assert directives[0].attrs["state"] == "selected_declared"
    assert directives[0].attrs["directive_instance_id"] == directives[0].id
    assert directives[0].attrs["scope_id"]
    assert directives[0].attrs["scope_kind"] == "hls.kernel"
    assert directives[0].attrs["scope_resolution"] == "external_exact"
    assert directives[0].attrs["function_id"] == directives[0].attrs["scope_id"]
    assert directives[0].attrs["parse_policy"] == (
        "hlsgraph.amd_2024_2_tcl_literal_strict_v1"
    )
    assert {item.predicate for item in result.observations} == {
        "directive.requested", "directive.declared_selected",
    }

    rejected = [item for item in result.diagnostics
                if item.code == "directive.tcl_nonliteral_context"]
    assert len(rejected) == 7
    assert {item.metadata["reason"] for item in rejected} == {
        "nested_script_context",
        "embedded_or_constructed_command",
        "dynamic_substitution",
        "multiple_commands",
    }
    assert all(item.metadata["completeness"] == "ambiguous" for item in rejected)
    assert len({item.id for item in rejected}) == len(rejected)
    assert result.coverage["ambiguous_tcl_directives"] == len(rejected)


def test_amd_2024_2_config_whitespace_grammar_covers_all_supported_kinds(
    tmp_path: Path,
) -> None:
    result, graph = _extract_external(
        tmp_path,
        "syn.directive.dataflow=dut\n"
        "syn.directive.pipeline=dut II=2 style=stp\n"
        "syn.directive.unroll=skip_exit_check factor=4 dut/L1\n"
        "syn.directive.array_partition=dut buffer type=cyclic factor=2 dim=1\n"
        "syn.directive.interface=mode=ap_hs interrupt=16 register dut input\n"
        "syn.directive.stream=depth=4 type=fifo dut fifo\n"
        "syn.directive.dependence=variable=buffer dependent=false "
        "type=inter dut/L1\n"
        "syn.directive.loop_tripcount=min=1 avg=2 max=4 dut/L1\n"
        "syn.directive.inline=recursive helper\n",
        origin="config",
        graph_factory=_complete_directive_graph,
    )

    directives = {
        item.name: item for item in result.graph.entities.values()
        if item.kind == "hls.directive"
    }
    assert set(directives) == {
        "DATAFLOW", "PIPELINE", "UNROLL", "ARRAY_PARTITION", "INTERFACE",
        "STREAM", "DEPENDENCE", "LOOP_TRIPCOUNT", "INLINE",
    }
    assert all(str(item.completeness) == "complete"
               for item in directives.values())
    assert directives["UNROLL"].attrs["options"] == {
        "skip_exit_check": True, "factor": 4,
    }
    assert directives["INTERFACE"].attrs["options"] == {
        "mode": "ap_hs", "interrupt": 16, "register": True,
    }
    assert directives["DEPENDENCE"].attrs["options"] == {
        "variable": "buffer", "dependent": False, "type": "inter",
    }
    assert directives["ARRAY_PARTITION"].attrs["scope_kind"] == "hls.memory"
    assert directives["STREAM"].attrs["scope_kind"] == "hls.stream"
    assert directives["INTERFACE"].attrs["scope_kind"] == "hls.port"
    assert directives["DEPENDENCE"].attrs["scope_kind"] == "hls.loop"
    assert directives["DEPENDENCE"].attrs["variable_id"] == (
        directives["ARRAY_PARTITION"].attrs["scope_id"]
    )
    assert len([
        item for item in result.observations
        if item.predicate == "directive.requested"
    ]) == 9
    assert not [
        item for item in result.diagnostics
        if item.code in {
            "directive.invalid_external_syntax",
            "directive.unresolved_scope",
            "directive.unresolved_operand",
        }
    ]
    interface_port = graph.entities[directives["INTERFACE"].attrs["scope_id"]]
    assert interface_port.name == "input"


def test_tcl_interface_interrupt_is_valued_and_register_is_a_flag(
    tmp_path: Path,
) -> None:
    result, _graph = _extract_external(
        tmp_path,
        "set_directive_interface -interrupt 16 -register -mode ap_hs "
        "dut input\n"
        "set_directive_interface -interrupt not_an_int -mode ap_hs "
        "dut input\n",
        origin="tcl",
        graph_factory=_complete_directive_graph,
    )

    directives = [
        item for item in result.graph.entities.values()
        if item.kind == "hls.directive"
    ]
    assert len(directives) == 1
    assert directives[0].attrs["options"] == {
        "interrupt": 16, "register": True, "mode": "ap_hs",
    }
    rejected = [
        item for item in result.diagnostics
        if item.code == "directive.invalid_external_syntax"
    ]
    assert len(rejected) == 1
    assert rejected[0].metadata["reason"] == "invalid_integer"
    assert rejected[0].metadata["option"] == "interrupt"
    assert len([
        item for item in result.observations
        if item.predicate == "directive.requested"
    ]) == 1


def test_tcl_valued_option_cannot_consume_the_next_option(
    tmp_path: Path,
) -> None:
    result, _graph = _extract_external(
        tmp_path,
        "set_directive_interface -bundle -register -mode ap_hs dut input\n",
        origin="tcl",
        graph_factory=_complete_directive_graph,
    )

    assert not [
        item for item in result.graph.entities.values()
        if item.kind == "hls.directive"
    ]
    assert not result.observations
    diagnostic = next(
        item for item in result.diagnostics
        if item.code == "directive.invalid_external_syntax"
    )
    assert diagnostic.metadata["reason"] == "missing_option_value"
    assert diagnostic.metadata["option"] == "bundle"


def test_external_frontends_do_not_rewrite_words_comments_or_other_grammar(
    tmp_path: Path,
) -> None:
    tcl_root = tmp_path / "tcl"
    tcl_root.mkdir()
    tcl, _graph = _extract_external(
        tcl_root,
        "set_directive_pipeline dut # not-a-Tcl-tail-comment\n"
        "set_directive_pipeline \"du{t}\"\n"
        "set_directive_pipeline {du{t}}\n"
        "syn.directive.pipeline=dut\n",
        origin="tcl", graph_factory=_complete_directive_graph,
    )
    assert not tcl.observations

    config_root = tmp_path / "config"
    config_root.mkdir()
    config, _graph = _extract_external(
        config_root,
        "syn.directive.pipeline=du{t}\n"
        "set_directive_pipeline dut\n",
        origin="config", graph_factory=_complete_directive_graph,
    )
    assert not config.observations


def test_tcl_scope_spelling_and_command_case_are_not_normalized(
    tmp_path: Path,
) -> None:
    result, _graph = _extract_external(
        tmp_path,
        'set_directive_pipeline "{dut}"\n'
        'set_directive_pipeline /dut/\n'
        'set_directive_pipeline "/dut/"\n'
        'SET_DIRECTIVE_PIPELINE dut\n'
        'Set_Directive_Pipeline dut\n',
        origin="tcl",
        graph_factory=_complete_directive_graph,
    )

    directives = [
        item for item in result.graph.entities.values()
        if item.kind == "hls.directive"
    ]
    # Literal but unresolved lowercase commands may remain as incomplete
    # declarations for diagnostics; none may become a requested fact.
    assert directives
    assert all(str(item.completeness) == "ambiguous" for item in directives)
    assert not result.observations


def test_dependence_requires_a_resolved_variable_or_an_explicit_class(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dependence"
    root.mkdir()
    result, graph = _extract_external(
        root,
        "syn.directive.dependence=dut/L1 variable=missing type=inter\n"
        "syn.directive.dependence=dut/L1 class=array type=inter\n",
        origin="config", graph_factory=_complete_directive_graph,
    )
    directives = sorted(
        (item for item in result.graph.entities.values()
         if item.kind == "hls.directive"),
        key=lambda item: item.qualified_name or "",
    )
    assert len(directives) == 2
    unresolved, by_class = directives
    assert str(unresolved.completeness) == "ambiguous"
    assert not any(
        item.subject_id == unresolved.id and item.predicate == "directive.requested"
        for item in result.observations
    )
    assert any(
        item.subject_id == unresolved.id and item.code == "directive.unresolved_operand"
        for item in result.diagnostics
    )
    assert str(by_class.completeness) == "complete"
    assert by_class.attrs["scope_id"] == next(
        item.id for item in graph.entities.values()
        if item.kind == "hls.loop" and item.name == "L1"
    )
    assert "variable_id" not in by_class.attrs
    assert any(
        item.subject_id == by_class.id and item.predicate == "directive.requested"
        for item in result.observations
    )


def test_invalid_external_grammar_is_diagnostic_only_and_rejects_legacy_comma(
    tmp_path: Path,
) -> None:
    result, _graph = _extract_external(
        tmp_path,
        "syn.directive.unroll=dut/L1,factor=2\n"
        "syn.directive.pipeline=II=2\n"
        "syn.directive.pipeline=dut II=2 II=3\n"
        "syn.directive.pipeline=dut mystery=1\n"
        "syn.directive.pipeline=dut II=fast\n"
        "syn.directive.pipeline=dut II=1_024\n"
        "syn.directive.pipeline=dut II=0\n"
        "syn.directive.pipeline=\n"
        "syn.directive.interface=max_widen_bitwidth=-1 mode=m_axi dut input\n"
        "syn.directive.latency=dut min=1\n"
        "syn.directive.inline=dut off recursive\n",
        origin="config",
        graph_factory=_complete_directive_graph,
    )

    assert not [
        item for item in result.graph.entities.values()
        if item.kind == "hls.directive"
    ]
    assert not result.observations
    rejected = [
        item for item in result.diagnostics
        if item.code == "directive.invalid_external_syntax"
    ]
    assert len(rejected) == 11
    assert len({item.id for item in rejected}) == 11
    assert {item.metadata["reason"] for item in rejected} == {
        "legacy_comma_syntax",
        "wrong_positional_arity",
        "duplicate_option",
        "unknown_option",
        "invalid_integer",
        "integer_out_of_range",
        "unsupported_directive_kind",
        "inline_off_conflict",
    }
    assert result.coverage["rejected_external_syntax"] == 11


def test_external_interface_requires_source_top_spelling_and_one_direct_owner(
    tmp_path: Path,
) -> None:
    def internal_id_graph(snapshot_id: str) -> CanonicalGraph:
        graph = CanonicalGraph(snapshot_id)
        kernel = graph.add_entity(Entity(
            kind="hls.kernel", name="dut", qualified_name="dut",
            snapshot_id=snapshot_id, id="entity_internal_top",
            authority=AuthorityClass.STATIC_FACT, stage=Stage.AST.value,
        ))
        port = graph.add_entity(Entity(
            kind="hls.port", name="input", qualified_name="dut::input",
            snapshot_id=snapshot_id, authority=AuthorityClass.STATIC_FACT,
            stage=Stage.AST.value,
        ))
        graph.add_relation(Relation(
            kernel.id, port.id, "hls.contains", snapshot_id,
            authority=AuthorityClass.STATIC_FACT, stage=Stage.AST.value,
        ))
        return graph

    id_root = tmp_path / "internal-id"
    id_root.mkdir()
    internal_id, _graph = _extract_external(
        id_root,
        "syn.directive.interface=mode=m_axi entity_internal_top input\n",
        origin="config", graph_factory=internal_id_graph,
    )
    assert not internal_id.observations
    assert any(
        item.code == "directive.unresolved_scope"
        for item in internal_id.diagnostics
    )

    def extra_owner_graph(snapshot_id: str) -> CanonicalGraph:
        graph = _complete_directive_graph(snapshot_id)
        loop = next(
            item for item in graph.entities.values() if item.kind == "hls.loop"
        )
        port = next(
            item for item in graph.entities.values()
            if item.kind == "hls.port" and item.name == "input"
        )
        graph.add_relation(Relation(
            loop.id, port.id, "hls.contains", snapshot_id,
            authority=AuthorityClass.STATIC_FACT, stage=Stage.AST.value,
        ))
        return graph

    owner_root = tmp_path / "extra-owner"
    owner_root.mkdir()
    extra_owner, _graph = _extract_external(
        owner_root,
        "syn.directive.interface=mode=m_axi dut input\n",
        origin="config", graph_factory=extra_owner_graph,
    )
    assert not extra_owner.observations
    assert any(
        item.code == "directive.unresolved_scope"
        for item in extra_owner.diagnostics
    )


def test_block_control_interface_is_preserved_unsupported_without_port_contract(
    tmp_path: Path,
) -> None:
    result, _graph = _extract_external(
        tmp_path,
        "syn.directive.interface=mode=ap_ctrl_none dut\n",
        origin="config",
        graph_factory=_complete_directive_graph,
    )

    directive = next(
        item for item in result.graph.entities.values()
        if item.kind == "hls.directive"
    )
    assert directive.name == "INTERFACE"
    assert directive.attrs["state"] == "unsupported_requested"
    assert str(directive.completeness) == "ambiguous"
    assert not {"scope_id", "scope_kind", "port_id", "variable_id"}.intersection(
        directive.attrs
    )
    assert not result.graph.relations
    assert not result.observations
    diagnostic = next(
        item for item in result.diagnostics
        if item.code == "directive.unsupported_scope_form"
    )
    assert diagnostic.subject_id == directive.id
    assert diagnostic.metadata["reason"] == "block_control_port_not_modeled"


def test_loop_scope_and_operand_require_complete_single_owner_evidence(
    tmp_path: Path,
) -> None:
    def partial_owner_graph(snapshot_id: str) -> CanonicalGraph:
        graph = _complete_directive_graph(snapshot_id)
        loop = next(
            item for item in graph.entities.values()
            if item.kind == "hls.loop" and item.name == "L1"
        )
        owner_relation = next(
            item for item in graph.relations.values()
            if item.kind == "hls.contains" and item.dst == loop.id
        )
        owner_relation.completeness = Completeness.AMBIGUOUS
        return graph

    partial_root = tmp_path / "partial"
    partial_root.mkdir()
    partial, _graph = _extract_external(
        partial_root,
        "syn.directive.unroll=dut/L1 factor=2\n",
        origin="config",
        graph_factory=partial_owner_graph,
    )
    assert not partial.observations
    assert any(
        item.code == "directive.unresolved_scope"
        for item in partial.diagnostics
    )

    def other_owner_graph(snapshot_id: str) -> CanonicalGraph:
        graph = _complete_directive_graph(snapshot_id)
        helper = next(
            item for item in graph.entities.values()
            if item.kind == "hls.function" and item.name == "helper"
        )
        foreign = graph.add_entity(Entity(
            kind="hls.memory", name="foreign", qualified_name="helper::foreign",
            snapshot_id=snapshot_id, authority=AuthorityClass.STATIC_FACT,
            stage=Stage.AST.value,
        ))
        graph.add_relation(Relation(
            helper.id, foreign.id, "hls.contains", snapshot_id,
            authority=AuthorityClass.STATIC_FACT, stage=Stage.AST.value,
        ))
        return graph

    foreign_root = tmp_path / "foreign"
    foreign_root.mkdir()
    foreign, _graph = _extract_external(
        foreign_root,
        "syn.directive.array_partition=dut/L1 foreign type=complete\n",
        origin="config",
        graph_factory=other_owner_graph,
    )
    assert not foreign.observations
    assert any(
        item.code == "directive.unresolved_operand"
        for item in foreign.diagnostics
    )


def test_external_dependence_keeps_function_scope_and_operand_separate(
    tmp_path: Path,
) -> None:
    (tmp_path / "kernel.cpp").write_text(
        "void dut(int *input) {}\n", encoding="utf-8",
    )
    (tmp_path / "directives.tcl").write_text(
        "set_directive_dependence -variable input dut\n"
        "set_directive_dependence -variable input input\n",
        encoding="utf-8",
    )
    manifest = minimal_manifest(
        "test.tcl_dependence", "tcl dependence", "dut", "kernel.cpp",
    )
    manifest.build.tcl_files = ["directives.tcl"]
    manifest.artifact_paths.append({
        "path": "directives.tcl", "kind": "config.tcl", "role": "hls_tcl",
        "access": "private",
    })
    bundle = GraphBundle.create(tmp_path, manifest)
    snapshot = bundle.snapshot()
    existing = CanonicalGraph(snapshot.id)
    kernel = existing.add_entity(Entity(
        kind="hls.kernel", name="dut", qualified_name="dut",
        snapshot_id=snapshot.id, authority=AuthorityClass.STATIC_FACT,
        stage=Stage.AST.value,
    ))
    port = existing.add_entity(Entity(
        kind="hls.port", name="input", qualified_name="dut::input",
        snapshot_id=snapshot.id, authority=AuthorityClass.STATIC_FACT,
        stage=Stage.AST.value,
    ))
    existing.add_relation(Relation(
        kernel.id, port.id, "hls.contains", snapshot.id,
        authority=AuthorityClass.STATIC_FACT, stage=Stage.AST.value,
    ))
    context = ExtractionContext(
        project_root=tmp_path, manifest=bundle.manifest, snapshot=snapshot,
        artifacts={item.id: item for item in bundle.store.artifacts(snapshot.id)},
        options={"existing_graph": existing},
    )
    result = ExternalDirectiveExtractor().extract(context)
    directives = sorted(
        (item for item in result.graph.entities.values()
         if item.kind == "hls.directive"),
        key=lambda item: item.qualified_name or "",
    )
    assert len(directives) == 2
    valid, invalid = directives
    assert valid.attrs["scope_id"] == kernel.id
    assert valid.attrs["scope_kind"] == "hls.kernel"
    assert valid.attrs["function_id"] == kernel.id
    assert valid.attrs["variable_id"] == port.id
    assert port.id != valid.attrs["scope_id"]
    assert "scope_id" not in invalid.attrs
    assert "variable_id" not in invalid.attrs
    assert any(
        item.code == "directive.unresolved_scope" and item.subject_id == invalid.id
        for item in result.diagnostics
    )
    annotation = next(
        item for item in result.graph.relations.values()
        if item.kind == "hls.annotates" and item.src == valid.id
    )
    assert annotation.dst == kernel.id


def test_tcl_continuations_and_uncertain_structure_remain_diagnostic_only(
    tmp_path: Path,
) -> None:
    result = _extract_tcl(
        tmp_path,
        """set note \\
set_directive_pipeline -II 3 dut
set broken \"unterminated
set_directive_unroll -factor 2 dut
""",
    )

    assert not [item for item in result.graph.entities.values()
                if item.kind == "hls.directive"]
    reasons = [item.metadata["reason"] for item in result.diagnostics
               if item.code == "directive.tcl_nonliteral_context"]
    assert reasons == ["continued_command", "nested_script_context"]
    assert not result.observations


def test_tcl_multiline_bracket_and_comment_continuation_are_not_top_level(
    tmp_path: Path,
) -> None:
    result = _extract_tcl(
        tmp_path,
        "set captured [\n"
        "  set_directive_pipeline -II 9 dut\n"
        "]\n"
        "# disabled \\\n"
        "set_directive_unroll -factor 8 dut\n"
        "set_directive_pipeline -II 1 dut\n",
    )

    directives = [item for item in result.graph.entities.values()
                  if item.kind == "hls.directive"]
    assert [(item.name, item.attrs["options"]) for item in directives] == [
        ("PIPELINE", {"ii": 1}),
    ]
    rejected = [item for item in result.diagnostics
                if item.code == "directive.tcl_nonliteral_context"]
    assert [item.metadata["reason"] for item in rejected] == [
        "nested_script_context", "continued_command",
    ]
    assert {item.predicate for item in result.observations
            if item.subject_id == directives[0].id} == {
        "directive.requested", "directive.declared_selected",
    }


def test_tcl_multiline_brace_quote_and_nested_bracket_state_recovers(
    tmp_path: Path,
) -> None:
    result = _extract_tcl(
        tmp_path,
        "set braced {\n"
        "  set_directive_pipeline -II 7 dut\n"
        "}\n"
        "set quoted \"prefix [\n"
        "  set_directive_unroll -factor 4 dut\n"
        "] suffix\"\n"
        "set_directive_dataflow dut\n",
    )

    directives = [item for item in result.graph.entities.values()
                  if item.kind == "hls.directive"]
    assert [item.name for item in directives] == ["DATAFLOW"]
    reasons = [item.metadata["reason"] for item in result.diagnostics
               if item.code == "directive.tcl_nonliteral_context"]
    assert reasons == ["nested_script_context", "nested_script_context"]
