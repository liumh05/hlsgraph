from __future__ import annotations

from pathlib import Path

from hlsgraph.bundle import GraphBundle
from hlsgraph.extract import ExtractionContext, ExternalDirectiveExtractor
from hlsgraph.extract.directives import resolve_directives
from hlsgraph.graph import CanonicalGraph
from hlsgraph.manifest import minimal_manifest
from hlsgraph.model import AuthorityClass, Entity, Relation, Stage


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
    assert directives[0].attrs["parse_policy"] == "hlsgraph.tcl_literal_top_level_v1"
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
