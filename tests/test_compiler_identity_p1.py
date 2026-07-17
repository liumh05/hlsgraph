from __future__ import annotations

from pathlib import Path

import pytest

from hlsgraph.bundle import GraphBundle
from hlsgraph.extract import LibClangExtractor
from hlsgraph.extract.source import _unsafe_preprocessor_tokens
from hlsgraph.manifest import (
    ManifestError,
    collect_artifacts,
    minimal_manifest,
    resolve_compiler_arguments,
)
from hlsgraph.sdk import Project


@pytest.mark.parametrize(
    ("source", "feature"),
    [
        ('constexpr auto path = __builtin_FILE();\n', "__builtin_FILE"),
        ('constexpr auto loc = __builtin_source_location();\n',
         "__builtin_source_location"),
    ],
)
def test_path_dependent_builtins_are_rejected_before_ast_extraction(
    source: str, feature: str,
) -> None:
    assert _unsafe_preprocessor_tokens(source) == {feature}


def test_explicit_artifact_metadata_cannot_hide_reachable_forced_include(
    tmp_path: Path,
) -> None:
    if not LibClangExtractor.available():
        pytest.skip("libclang is optional")
    (tmp_path / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    (tmp_path / "forced.hpp").write_text(
        "constexpr auto build_path = __builtin_FILE();\n", encoding="utf-8",
    )
    manifest = minimal_manifest(
        "test.compiler_identity.explicit_override", "override", "dut", "kernel.cpp",
    )
    manifest.build.cflags = ["-include", "forced.hpp"]
    manifest.artifact_paths.append({
        "path": "forced.hpp",
        "kind": "opaque.header",
        "role": "unrelated_input",
        "metadata": {"hlsgraph.compiler_reachable_text": False},
    })

    artifacts = collect_artifacts(manifest, tmp_path)
    forced = next(item for item in artifacts if item.uri == "forced.hpp")
    assert forced.kind == "opaque.header"
    assert forced.role == "unrelated_input"
    assert forced.metadata["hlsgraph.compiler_reachable_text"] is True

    project = Project(GraphBundle.create(tmp_path, manifest))
    indexed = project.index()
    assert indexed.success is False
    diagnostic = next(
        item for item in project.bundle.store.diagnostics(indexed.snapshot_id)
        if item.code == "source.unsupported_preprocessor_input"
    )
    assert diagnostic.metadata["features"] == ["__builtin_FILE"]
    assert project.bundle.store.has_graph(indexed.snapshot_id) is False


@pytest.mark.parametrize(
    ("builtin", "feature"),
    [
        ("__builtin_FILE()", "__builtin_FILE"),
        ("__builtin_source_location()", "__builtin_source_location"),
    ],
)
def test_macro_expanded_include_is_scanned_from_libclang_actual_dependencies(
    tmp_path: Path, builtin: str, feature: str,
) -> None:
    if not LibClangExtractor.available():
        pytest.skip("libclang is optional")
    (tmp_path / "kernel.cpp").write_text(
        '#define HEADER_NAME "hidden.hpp"\n'
        "#include HEADER_NAME\n"
        "void dut() {}\n",
        encoding="utf-8",
    )
    (tmp_path / "hidden.hpp").write_text(
        f"#define HIDDEN_PATH_VALUE {builtin}\n", encoding="utf-8",
    )
    manifest = minimal_manifest(
        f"test.compiler_identity.macro_include.{feature}",
        "macro include", "dut", "kernel.cpp",
    )
    manifest.artifact_paths.append({
        "path": "hidden.hpp",
        "kind": "opaque.payload",
        "role": "unrelated_input",
        "metadata": {"declared_by": "fixture"},
    })

    artifacts = collect_artifacts(manifest, tmp_path)
    hidden = next(item for item in artifacts if item.uri == "hidden.hpp")
    assert hidden.kind == "opaque.payload"
    assert hidden.role == "unrelated_input"
    # The static scanner cannot expand HEADER_NAME.  The safety property must
    # therefore come from libclang's actual dependency set, not this marker.
    assert hidden.metadata.get("hlsgraph.compiler_reachable_text") is not True

    project = Project(GraphBundle.create(tmp_path, manifest))
    indexed = project.index()
    assert indexed.success is False
    diagnostic = next(
        item for item in project.bundle.store.diagnostics(indexed.snapshot_id)
        if item.code == "source.unsupported_preprocessor_input"
        and item.artifact_id == hidden.id
    )
    assert diagnostic.metadata["features"] == [feature]
    assert project.bundle.store.has_graph(indexed.snapshot_id) is False


def test_unreferenced_binary_artifact_is_not_treated_as_compiler_text(
    tmp_path: Path,
) -> None:
    if not LibClangExtractor.available():
        pytest.skip("libclang is optional")
    (tmp_path / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    (tmp_path / "payload.bin").write_bytes(b"\x00__builtin_FILE()\xff")
    manifest = minimal_manifest(
        "test.compiler_identity.unreferenced_binary", "binary", "dut", "kernel.cpp",
    )
    manifest.artifact_paths.append({
        "path": "payload.bin",
        "kind": "opaque.payload",
        "role": "unrelated_input",
    })

    project = Project(GraphBundle.create(tmp_path, manifest))
    indexed = project.index()
    assert indexed.success
    assert not any(
        item.code in {
            "source.unsupported_preprocessor_input",
            "source.unsupported_text_encoding",
        }
        for item in project.bundle.store.diagnostics(indexed.snapshot_id)
    )


@pytest.mark.parametrize(
    "flag",
    [
        "-I", "/I", "/i", "-isystem", "-iquote", "-idirafter",
        "--include-directory", "-F", "-iframework", "-imsvc", "/imsvc",
        "/IMsvc", "/external:I", "/external:i", "-cxx-isystem",
        "-stdlib++-isystem", "-isystem-after",
    ],
)
def test_every_supported_separate_include_path_is_anchored_to_its_context(
    tmp_path: Path, flag: str,
) -> None:
    include = tmp_path / "include"
    include.mkdir()
    arguments, _ = resolve_compiler_arguments(
        tmp_path, tmp_path, [flag, "include"],
    )
    assert arguments == [flag, str(include.resolve())]


@pytest.mark.parametrize(
    "prefix",
    [
        "-I", "/I", "/i", "-isystem", "-iquote", "-idirafter", "-F",
        "-iframework", "-imsvc", "/imsvc", "/IMsvc", "/external:I",
        "/external:i",
        "-cxx-isystem", "-stdlib++-isystem", "-isystem-after",
    ],
)
def test_every_supported_joined_include_path_is_anchored_to_its_context(
    tmp_path: Path, prefix: str,
) -> None:
    include = tmp_path / "include"
    include.mkdir()
    arguments, _ = resolve_compiler_arguments(
        tmp_path, tmp_path, [prefix + "include"],
    )
    assert arguments == [prefix + str(include.resolve())]


def test_global_cxx_system_include_uses_project_root_for_collection_and_clang(
    tmp_path: Path,
) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "include").mkdir()
    (tmp_path / "src/kernel.cpp").write_text(
        "#include <dep.hpp>\nvoid dut() {}\n", encoding="utf-8",
    )
    (tmp_path / "include/dep.hpp").write_text("#define DEP 1\n", encoding="utf-8")
    manifest = minimal_manifest(
        "test.compiler_identity.global_path", "global path", "dut", "src/kernel.cpp",
    )
    manifest.build.translation_units[0].directory = "src"
    manifest.build.cflags = ["-cxx-isystem", "include"]

    artifacts = collect_artifacts(manifest, tmp_path)
    assert "include/dep.hpp" in {item.uri for item in artifacts}
    resolved, _ = resolve_compiler_arguments(
        tmp_path, tmp_path, manifest.build.cflags,
    )
    assert resolved == ["-cxx-isystem", str((tmp_path / "include").resolve())]


@pytest.mark.parametrize(
    "arguments",
    [
        ["-DROOT=${PROJECT_ROOT}"],
        ["-D", "ROOT=${PROJECT_ROOT}"],
        ["-std=${PROJECT_ROOT}"],
        ["${PROJECT_ROOT}/compiler"],
    ],
)
def test_project_root_placeholder_is_rejected_outside_path_arguments(
    tmp_path: Path, arguments: list[str],
) -> None:
    with pytest.raises(ManifestError, match="only supported.*path-valued"):
        resolve_compiler_arguments(tmp_path, tmp_path, arguments)


def test_project_root_placeholder_remains_available_for_recognized_paths(
    tmp_path: Path,
) -> None:
    (tmp_path / "include").mkdir()
    (tmp_path / "flags.rsp").write_text("-I${PROJECT_ROOT}/include\n", encoding="utf-8")
    arguments, responses = resolve_compiler_arguments(
        tmp_path, tmp_path, ["@${PROJECT_ROOT}/flags.rsp"],
    )
    assert arguments == ["-I" + str((tmp_path / "include").resolve())]
    assert responses == {"flags.rsp"}


@pytest.mark.parametrize("degraded", [False, True])
def test_inactive_source_pragma_never_becomes_a_directive_fact(
    tmp_path: Path, degraded: bool,
) -> None:
    if not degraded and not LibClangExtractor.available():
        pytest.skip("libclang is optional")
    (tmp_path / "kernel.cpp").write_text(
        "void dut(int *x) {\n"
        "#if 0\n"
        "#pragma HLS PIPELINE II=1\n"
        "#endif\n"
        "  for (int i = 0; i < 2; ++i) x[i]++;\n"
        "}\n",
        encoding="utf-8",
    )
    project = Project(GraphBundle.create(
        tmp_path,
        minimal_manifest(
            f"test.compiler_identity.inactive.{degraded}", "inactive", "dut", "kernel.cpp",
        ),
    ))
    indexed = project.index(degraded=degraded)
    assert indexed.success
    assert not any(
        item.kind == "hls.directive" for item in project.service().graph().entities.values()
    )
    assert any(
        item.code == "directive.inactive_source_pragma"
        for item in project.bundle.store.diagnostics(indexed.snapshot_id)
    )


def test_degraded_macro_conditional_pragma_is_unknown_not_requested(
    tmp_path: Path,
) -> None:
    (tmp_path / "kernel.cpp").write_text(
        "void dut(int *x) {\n"
        "#if ENABLE_PIPELINE\n"
        "#pragma HLS PIPELINE II=1\n"
        "#endif\n"
        "  for (int i = 0; i < 2; ++i) x[i]++;\n"
        "}\n",
        encoding="utf-8",
    )
    project = Project(GraphBundle.create(
        tmp_path,
        minimal_manifest(
            "test.compiler_identity.unknown_degraded", "unknown", "dut", "kernel.cpp",
        ),
    ))
    indexed = project.index(degraded=True)
    assert indexed.success
    assert not any(
        item.kind == "hls.directive" for item in project.service().graph().entities.values()
    )
    assert any(
        item.code == "directive.preprocessor_activity_unknown"
        for item in project.bundle.store.diagnostics(indexed.snapshot_id)
    )
