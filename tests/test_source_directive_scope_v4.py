from __future__ import annotations

from pathlib import Path

import pytest

from hlsgraph.bundle import GraphBundle
from hlsgraph.extract.base import ExtractionContext, ExtractionPipeline
from hlsgraph.extract.source import LibClangExtractor, RegexSourceExtractor
from hlsgraph.manifest import minimal_manifest
from hlsgraph.model import DiagnosticSeverity, ToolchainContext


def _extract_source(root: Path, source: str, *, degraded: bool = False):
    if not degraded and not LibClangExtractor.available():
        pytest.skip("source directive v4 tests require the libclang extra")
    (root / "kernel.cpp").write_text(source, encoding="utf-8")
    manifest = minimal_manifest(
        f"test.source.directive.v4.{root.name}",
        "source directive v4",
        "dut",
        "kernel.cpp",
    )
    manifest.toolchains = [ToolchainContext(
        id="amd.vitis_hls.2024_2",
        vendor="amd",
        name="vitis_hls",
        version="2024.2",
    )]
    bundle = GraphBundle.create(root, manifest)
    snapshot = bundle.snapshot()
    artifacts = {
        item.id: item for item in bundle.store.artifacts(snapshot.id)
    }
    extractor = RegexSourceExtractor() if degraded else LibClangExtractor()
    result = ExtractionPipeline([extractor]).run(
        ExtractionContext(
            project_root=root,
            manifest=manifest,
            snapshot=snapshot,
            artifacts=artifacts,
            allow_degraded=degraded,
        )
    )
    errors = [
        item for item in result.diagnostics
        if item.severity in {
            DiagnosticSeverity.ERROR,
            DiagnosticSeverity.CRITICAL,
        }
    ]
    assert not errors, [(item.code, item.message) for item in errors]
    return result


@pytest.mark.parametrize(
    "pseudo_directive",
    [
        (
            "void dut() {\n"
            "/*\n"
            "#pragma HLS PIPELINE II=9\n"
            "*/\n"
            "}\n"
        ),
        (
            "void dut() {\n"
            '  const char *text = R"hls(\n'
            "#pragma HLS UNROLL factor=9\n"
            ')hls";\n'
            "  (void)text;\n"
            "}\n"
        ),
        (
            "void dut() {\n"
            "  // the next physical line is still this comment \\\n"
            "#pragma HLS DATAFLOW\n"
            "}\n"
        ),
    ],
    ids=["block-comment", "raw-string", "line-spliced-comment"],
)
def test_comment_literal_and_spliced_comment_pragmas_are_not_facts(
    tmp_path: Path,
    pseudo_directive: str,
) -> None:
    result = _extract_source(tmp_path, pseudo_directive)
    assert not [
        item for item in result.graph.entities.values()
        if item.kind == "hls.directive"
    ]
    assert not [
        item for item in result.observations
        if item.predicate == "directive.requested"
    ]


@pytest.mark.parametrize(
    "pseudo_directive",
    [
        (
            "void dut() {\n"
            "  // the next physical line is still this comment \\\n"
            "#pragma HLS PIPELINE II=9\n"
            "}\n"
        ),
        (
            "void dut() {\n"
            "/\\\n"
            "*\n"
            "#pragma HLS PIPELINE II=9\n"
            "*\\\n"
            "/\n"
            "}\n"
        ),
    ],
    ids=["continued-line-comment", "continued-block-comment-delimiters"],
)
def test_degraded_spliced_prose_withholds_all_pragma_candidates(
    tmp_path: Path,
    pseudo_directive: str,
) -> None:
    result = _extract_source(tmp_path, pseudo_directive, degraded=True)
    assert not [
        item for item in result.graph.entities.values()
        if item.kind == "hls.directive"
    ]
    assert not [
        item for item in result.observations
        if item.predicate == "directive.requested"
    ]
    assert any(
        item.code == "directive.preprocessor_activity_unknown"
        for item in result.diagnostics
    )


def test_file_scope_loop_pragma_cannot_borrow_a_later_function_loop(
    tmp_path: Path,
) -> None:
    result = _extract_source(
        tmp_path,
        "#pragma HLS UNROLL factor=2\n"
        "void dut() {\n"
        "  for (int i = 0; i < 4; ++i) {}\n"
        "}\n",
    )
    directive = next(
        item for item in result.graph.entities.values()
        if item.kind == "hls.directive"
    )
    assert str(directive.completeness) == "ambiguous"
    assert "scope_id" not in directive.attrs
    assert not [
        item for item in result.observations
        if item.subject_id == directive.id
        and item.predicate == "directive.requested"
    ]


def test_intervening_statement_prevents_following_loop_binding(
    tmp_path: Path,
) -> None:
    result = _extract_source(
        tmp_path,
        "void dut(int *output) {\n"
        "#pragma HLS UNROLL factor=2\n"
        "  output[0] = 0;\n"
        "  for (int i = 0; i < 4; ++i) output[i] += 1;\n"
        "}\n",
    )
    directive = next(
        item for item in result.graph.entities.values()
        if item.kind == "hls.directive"
    )
    assert str(directive.completeness) == "ambiguous"
    assert "scope_id" not in directive.attrs


def test_immediately_following_loop_binds_with_exact_owner(
    tmp_path: Path,
) -> None:
    result = _extract_source(
        tmp_path,
        "void dut() {\n"
        "#pragma HLS UNROLL factor=2\n"
        "  for (int i = 0; i < 4; ++i) {}\n"
        "}\n",
    )
    directive = next(
        item for item in result.graph.entities.values()
        if item.kind == "hls.directive"
    )
    loop = next(
        item for item in result.graph.entities.values()
        if item.kind == "hls.loop"
    )
    assert directive.attrs["scope_id"] == loop.id
    assert directive.attrs["loop_id"] == loop.id
    assert directive.attrs["scope_resolution"] == "source_ast"
    assert str(directive.completeness) == "complete"


def test_dataflow_inside_loop_binds_the_enclosing_loop(
    tmp_path: Path,
) -> None:
    result = _extract_source(
        tmp_path,
        "void dut(int *output) {\n"
        "  for (int i = 0; i < 4; ++i) {\n"
        "#pragma HLS DATAFLOW\n"
        "    output[i] += 1;\n"
        "  }\n"
        "}\n",
    )
    directive = next(
        item for item in result.graph.entities.values()
        if item.kind == "hls.directive"
    )
    loop = next(
        item for item in result.graph.entities.values()
        if item.kind == "hls.loop"
    )
    assert directive.attrs["scope_id"] == loop.id
    assert directive.attrs["loop_id"] == loop.id
    assert directive.attrs["scope_kind"] == "hls.loop"


@pytest.mark.parametrize(
    "body",
    [
        (
            "#pragma HLS DEPENDENCE variable=buffer inter=false\n"
            "  int buffer[4] = {};\n"
        ),
        (
            "  {\n"
            "    int buffer[4] = {};\n"
            "    buffer[0] = 1;\n"
            "  }\n"
            "  {\n"
            "#pragma HLS DEPENDENCE variable=buffer inter=false\n"
            "  }\n"
        ),
    ],
    ids=["declared-after", "sibling-block"],
)
def test_dependence_operand_requires_prior_lexically_visible_declaration(
    tmp_path: Path,
    body: str,
) -> None:
    result = _extract_source(
        tmp_path,
        "void dut() {\n" + body + "}\n",
    )
    directive = next(
        item for item in result.graph.entities.values()
        if item.kind == "hls.directive"
    )
    assert directive.attrs["scope_kind"] == "hls.kernel"
    assert "variable_id" not in directive.attrs
    assert str(directive.completeness) == "ambiguous"
    assert any(
        item.code == "directive.unresolved_operand"
        and item.subject_id == directive.id
        for item in result.diagnostics
    )


def test_dependence_operand_accepts_prior_outer_scope_declaration(
    tmp_path: Path,
) -> None:
    result = _extract_source(
        tmp_path,
        "void dut() {\n"
        "  int buffer[4] = {};\n"
        "  {\n"
        "#pragma HLS DEPENDENCE variable=buffer inter=false\n"
        "    buffer[0] += 1;\n"
        "  }\n"
        "}\n",
    )
    directive = next(
        item for item in result.graph.entities.values()
        if item.kind == "hls.directive"
    )
    operand = next(
        item for item in result.graph.entities.values()
        if item.kind == "hls.memory" and item.name == "buffer"
    )
    assert directive.attrs["variable_id"] == operand.id
    assert str(directive.completeness) == "complete"


def test_for_init_operand_is_not_visible_after_the_loop(
    tmp_path: Path,
) -> None:
    result = _extract_source(
        tmp_path,
        "void dut() {\n"
        "  for (int i = 0; i < 4; ++i) {}\n"
        "#pragma HLS DEPENDENCE variable=i inter=false\n"
        "}\n",
    )
    directive = next(
        item for item in result.graph.entities.values()
        if item.kind == "hls.directive"
    )
    assert directive.attrs["scope_kind"] == "hls.kernel"
    assert "variable_id" not in directive.attrs
    assert str(directive.completeness) == "ambiguous"
    assert any(
        item.code == "directive.unresolved_operand"
        and item.subject_id == directive.id
        for item in result.diagnostics
    )


def test_for_init_operand_is_visible_inside_the_loop(
    tmp_path: Path,
) -> None:
    result = _extract_source(
        tmp_path,
        "void dut() {\n"
        "  for (int i = 0; i < 4; ++i) {\n"
        "#pragma HLS DEPENDENCE variable=i inter=false\n"
        "  }\n"
        "}\n",
    )
    directive = next(
        item for item in result.graph.entities.values()
        if item.kind == "hls.directive"
    )
    loop = next(
        item for item in result.graph.entities.values()
        if item.kind == "hls.loop"
    )
    operand = next(
        item for item in result.graph.entities.values()
        if item.kind == "source.variable" and item.name == "i"
    )
    assert directive.attrs["scope_id"] == loop.id
    assert directive.attrs["variable_id"] == operand.id
    assert str(directive.completeness) == "complete"


@pytest.mark.parametrize(
    "statement",
    [
        (
            "  if (int scoped = output[0]) {\n"
            "#pragma HLS DEPENDENCE variable=scoped inter=false\n"
            "    output[0] += scoped;\n"
            "  }\n"
        ),
        (
            "  int values[2] = {1, 2};\n"
            "  for (int scoped : values) {\n"
            "#pragma HLS DEPENDENCE variable=scoped inter=false\n"
            "    output[0] += scoped;\n"
            "  }\n"
        ),
    ],
    ids=["if-init", "range-for"],
)
def test_statement_local_operand_is_visible_inside_its_scope(
    tmp_path: Path,
    statement: str,
) -> None:
    result = _extract_source(
        tmp_path,
        "void dut(int *output) {\n" + statement + "}\n",
    )
    directive = next(
        item for item in result.graph.entities.values()
        if item.kind == "hls.directive"
    )
    operand = next(
        item for item in result.graph.entities.values()
        if item.kind == "source.variable" and item.name == "scoped"
    )
    assert directive.attrs["variable_id"] == operand.id
    assert str(directive.completeness) == "complete"


@pytest.mark.parametrize(
    "statement",
    [
        "  if (int scoped = output[0]) { output[0] += scoped; }\n",
        (
            "  int values[2] = {1, 2};\n"
            "  for (int scoped : values) { output[0] += scoped; }\n"
        ),
    ],
    ids=["if-init", "range-for"],
)
def test_statement_local_operand_is_not_visible_after_its_scope(
    tmp_path: Path,
    statement: str,
) -> None:
    result = _extract_source(
        tmp_path,
        "void dut(int *output) {\n"
        + statement
        + "#pragma HLS DEPENDENCE variable=scoped inter=false\n"
        + "}\n",
    )
    directive = next(
        item for item in result.graph.entities.values()
        if item.kind == "hls.directive"
    )
    assert "variable_id" not in directive.attrs
    assert str(directive.completeness) == "ambiguous"
    assert any(
        item.code == "directive.unresolved_operand"
        and item.subject_id == directive.id
        for item in result.diagnostics
    )


def test_array_partition_does_not_guess_through_local_shadowing(
    tmp_path: Path,
) -> None:
    result = _extract_source(
        tmp_path,
        "void dut() {\n"
        "  int buffer[4] = {};\n"
        "  {\n"
        "    int buffer = 0;\n"
        "#pragma HLS ARRAY_PARTITION variable=buffer complete\n"
        "    buffer += 1;\n"
        "  }\n"
        "}\n",
    )
    directive = next(
        item for item in result.graph.entities.values()
        if item.kind == "hls.directive"
    )
    assert str(directive.completeness) == "ambiguous"
    assert "scope_id" not in directive.attrs
    assert "variable_id" not in directive.attrs
    assert any(
        item.code == "directive.unresolved_scope"
        and item.subject_id == directive.id
        for item in result.diagnostics
    )


@pytest.mark.parametrize(
    "options",
    [
        "II=1 ii=2",
        "flags=internal rewind",
        "II=1=2",
    ],
    ids=["duplicate-key", "reserved-flags-collision", "extra-equals"],
)
def test_ambiguous_source_options_are_withheld(
    tmp_path: Path,
    options: str,
) -> None:
    result = _extract_source(
        tmp_path,
        "void dut() {\n"
        f"#pragma HLS PIPELINE {options}\n"
        "}\n",
    )
    assert not [
        item for item in result.graph.entities.values()
        if item.kind == "hls.directive"
    ]
    assert any(
        item.code == "directive.ambiguous_source_options"
        for item in result.diagnostics
    )


def test_multiline_source_pragma_is_withheld(
    tmp_path: Path,
) -> None:
    result = _extract_source(
        tmp_path,
        "void dut() {\n"
        "#pragma HLS PIPELINE \\\n"
        "  II=1\n"
        "}\n",
    )
    assert not [
        item for item in result.graph.entities.values()
        if item.kind == "hls.directive"
    ]
    assert any(
        item.code in {
            "source.pragma_token_mismatch",
            "directive.preprocessor_activity_unknown",
        }
        for item in result.diagnostics
    )


def test_unrelated_line_continuation_does_not_hide_valid_pragma(
    tmp_path: Path,
) -> None:
    result = _extract_source(
        tmp_path,
        "#define ADD_ONE(value) \\\n"
        "  ((value) + 1)\n"
        "void dut(int *output) {\n"
        "#pragma HLS PIPELINE II=1\n"
        "  output[0] = ADD_ONE(output[0]);\n"
        "}\n",
    )
    directive = next(
        item for item in result.graph.entities.values()
        if item.kind == "hls.directive"
    )
    assert directive.name == "PIPELINE"
    assert directive.attrs["scope_kind"] == "hls.kernel"
    assert str(directive.completeness) == "complete"
