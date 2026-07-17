from __future__ import annotations

import copy
import json
import random
import re
import string
import subprocess
from dataclasses import replace

import pytest

from hlsgraph.api import RestApplication
from hlsgraph.bundle import GraphBundle
from hlsgraph.extract import LibClangExtractor
from hlsgraph.graph import CanonicalGraph
from hlsgraph.manifest import (
    ManifestError,
    collect_artifacts,
    load_compile_commands,
    make_snapshot,
    minimal_manifest,
    resolve_compiler_arguments,
    split_compilation_command,
)
from hlsgraph.mcp import ReadOnlyMcpService
from hlsgraph.model import (
    ClockConstraint,
    Entity,
    ProjectManifest,
    Relation,
    ToolchainContext,
    json_ready,
)
from hlsgraph.query import CoreService, QuerySpec
from hlsgraph.sdk import Project


def _snapshot_manifest(tmp_path):
    (tmp_path / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    (tmp_path / "directives.tcl").write_text(
        "set_directive_pipeline -II 1 dut\n", encoding="utf-8"
    )
    manifest = minimal_manifest(
        "test.snapshot_identity", "snapshot identity", "dut", "kernel.cpp",
        part="xck26-test", clock_ns=5.0,
    )
    manifest.build.defines = {"MODE": "1"}
    manifest.build.tcl_files = ["directives.tcl"]
    manifest.artifact_paths.append({
        "path": "directives.tcl", "kind": "config.tcl", "role": "hls_tcl",
        "access": "private",
    })
    manifest.toolchains = [ToolchainContext(
        id="amd.vitis.2024_2", vendor="amd", name="vitis_hls", version="2024.2",
        build="5239630",
    )]
    return manifest


def test_snapshot_identity_is_stable_and_covers_every_semantic_input(tmp_path):
    manifest = _snapshot_manifest(tmp_path)
    artifacts = collect_artifacts(manifest, tmp_path)
    baseline = make_snapshot(manifest, artifacts)
    repeated = make_snapshot(copy.deepcopy(manifest), collect_artifacts(manifest, tmp_path))
    assert baseline.id == repeated.id
    assert baseline.identity_payload() == repeated.identity_payload()

    variants = []

    macro = copy.deepcopy(manifest)
    macro.build.defines["MODE"] = "2"
    variants.append(make_snapshot(macro, artifacts))

    top = copy.deepcopy(manifest)
    top.build.top = "dut_variant"
    variants.append(make_snapshot(top, artifacts))

    part = copy.deepcopy(manifest)
    part.target.part = "xczu-test"
    variants.append(make_snapshot(part, artifacts))

    clock = copy.deepcopy(manifest)
    clock.target.clocks = [ClockConstraint("default", 4.0, 0.1)]
    variants.append(make_snapshot(clock, artifacts))

    tool = copy.deepcopy(manifest)
    tool.toolchains[0] = replace(tool.toolchains[0], version="2025.1", build="6000000")
    variants.append(make_snapshot(tool, artifacts))

    (tmp_path / "directives.tcl").write_text(
        "set_directive_pipeline -II 2 dut\n", encoding="utf-8"
    )
    variants.append(make_snapshot(manifest, collect_artifacts(manifest, tmp_path)))

    assert len({item.id for item in variants}) == len(variants)
    assert all(item.id != baseline.id for item in variants)


def test_snapshot_hashes_angle_includes_forced_headers_and_response_files(tmp_path):
    for directory in ("compile/nested", "declared", "global"):
        (tmp_path / directory).mkdir(parents=True, exist_ok=True)
    (tmp_path / "kernel.cpp").write_text(
        "#include <angle.hpp>\n#include <declared.hpp>\nvoid dut() {}\n",
        encoding="utf-8",
    )
    (tmp_path / "compile/angle.hpp").write_text(
        "#include <nested/deep.hpp>\n", encoding="utf-8"
    )
    (tmp_path / "compile/nested/deep.hpp").write_text("#define DEEP 1\n", encoding="utf-8")
    (tmp_path / "compile/cli_forced.hpp").write_text("#define CLI 1\n", encoding="utf-8")
    (tmp_path / "declared/declared.hpp").write_text("#define DECLARED 1\n", encoding="utf-8")
    (tmp_path / "global/global_forced.hpp").write_text("#define GLOBAL 1\n", encoding="utf-8")
    (tmp_path / "compile.rsp").write_text(
        "-I compile -include cli_forced.hpp\n", encoding="utf-8"
    )
    (tmp_path / "global.rsp").write_text(
        "-I global -include global_forced.hpp\n", encoding="utf-8"
    )

    manifest = minimal_manifest("test.include_closure", "includes", "dut", "kernel.cpp")
    manifest.build.include_dirs = ["declared"]
    manifest.build.translation_units[0].arguments = ["-std=c++17", "@compile.rsp"]
    manifest.build.cflags = ["@global.rsp"]
    artifacts = collect_artifacts(manifest, tmp_path)
    uris = {item.uri for item in artifacts}
    assert {
        "kernel.cpp", "compile.rsp", "global.rsp",
        "compile/angle.hpp", "compile/nested/deep.hpp", "compile/cli_forced.hpp",
        "declared/declared.hpp", "global/global_forced.hpp",
    }.issubset(uris)

    before = make_snapshot(manifest, artifacts)
    (tmp_path / "compile/nested/deep.hpp").write_text("#define DEEP 2\n", encoding="utf-8")
    after = make_snapshot(manifest, collect_artifacts(manifest, tmp_path))
    assert after.id != before.id


def test_global_forced_include_resolves_through_translation_unit_include_root(tmp_path):
    (tmp_path / "inc").mkdir()
    (tmp_path / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    (tmp_path / "inc/forced.hpp").write_text("#define FORCED 1\n", encoding="utf-8")
    manifest = minimal_manifest(
        "test.global_forced_include", "global forced include", "dut", "kernel.cpp",
    )
    manifest.build.cflags = ["-include", "forced.hpp"]
    manifest.build.translation_units[0].arguments.extend(["-I", "inc"])

    artifacts = collect_artifacts(manifest, tmp_path)
    assert "inc/forced.hpp" in {item.uri for item in artifacts}


def test_snapshot_recurses_through_common_textual_include_suffixes(tmp_path):
    (tmp_path / "kernel.cpp").write_text(
        '#include "defs.inc"\nvoid dut() {}\n', encoding="utf-8",
    )
    (tmp_path / "defs.inc").write_text(
        '#include "nested.hpp"\n', encoding="utf-8",
    )
    (tmp_path / "nested.hpp").write_text("#define VALUE 1\n", encoding="utf-8")
    manifest = minimal_manifest(
        "test.textual_include_suffix", "text includes", "dut", "kernel.cpp",
    )
    artifacts = collect_artifacts(manifest, tmp_path)
    assert {"kernel.cpp", "defs.inc", "nested.hpp"}.issubset(
        item.uri for item in artifacts
    )


def test_snapshot_hashes_project_local_vfs_overlay_before_standard_rejects_it(tmp_path):
    (tmp_path / "cfg").mkdir()
    (tmp_path / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    overlay = tmp_path / "cfg/overlay.yaml"
    overlay.write_text("{ 'version': 0, 'roots': [] }\n", encoding="utf-8")

    manifest = minimal_manifest(
        "test.vfs_input_identity", "VFS input", "dut", "kernel.cpp",
    )
    manifest.build.translation_units[0].arguments.extend([
        "-ivfsoverlay", "cfg/overlay.yaml",
    ])
    artifacts = collect_artifacts(manifest, tmp_path)
    assert {"cfg/overlay.yaml"}.issubset(
        item.uri for item in artifacts
    )
    baseline = make_snapshot(manifest, artifacts)

    overlay.write_text("{ 'version': 0, 'roots': ['changed'] }\n", encoding="utf-8")
    changed_overlay = make_snapshot(manifest, collect_artifacts(manifest, tmp_path))
    assert changed_overlay.id != baseline.id


def test_response_files_are_bom_aware_and_fail_closed_on_invalid_text(tmp_path):
    utf8 = tmp_path / "utf8.rsp"
    utf16 = tmp_path / "utf16.rsp"
    invalid = tmp_path / "invalid.rsp"
    utf8.write_bytes(b"\xef\xbb\xbf" + '-I"include dir" -DUTF8=1'.encode("utf-8"))
    utf16.write_text('-I"wide include" -DWIDE=1', encoding="utf-16")
    invalid.write_bytes(b"\x80\x81\x82")

    utf8_args, utf8_files = resolve_compiler_arguments(
        tmp_path, tmp_path, ["@utf8.rsp"], platform="nt",
    )
    utf16_args, utf16_files = resolve_compiler_arguments(
        tmp_path, tmp_path, ["@utf16.rsp"], platform="nt",
    )
    assert utf8_args == ["-I" + str((tmp_path / "include dir").resolve()), "-DUTF8=1"]
    assert utf16_args == ["-I" + str((tmp_path / "wide include").resolve()), "-DWIDE=1"]
    assert utf8_files == {"utf8.rsp"}
    assert utf16_files == {"utf16.rsp"}
    with pytest.raises(ManifestError, match="not valid UTF-8/UTF-16"):
        resolve_compiler_arguments(
            tmp_path, tmp_path, ["@invalid.rsp"], platform="nt",
        )


def test_external_compiler_inputs_and_response_files_fail_closed(tmp_path):
    root = tmp_path / "project"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    external_header = outside / "forced.hpp"
    external_header.write_text("#define PRIVATE_VALUE 1\n", encoding="utf-8")
    external_response = outside / "flags.rsp"
    external_response.write_text("-DPRIVATE_VALUE=1\n", encoding="utf-8")

    forced_manifest = minimal_manifest(
        "test.external_forced_include", "external include", "dut", "kernel.cpp",
    )
    forced_manifest.build.translation_units[0].arguments.extend([
        "-include", str(external_header),
    ])
    with pytest.raises(ManifestError, match="forced compiler includes.*project-local"):
        collect_artifacts(forced_manifest, root)

    for flag in ("-I", "-isystem"):
        include_manifest = minimal_manifest(
            "test.external_include_root", "external include root", "dut", "kernel.cpp",
        )
        include_manifest.build.translation_units[0].arguments.extend([
            flag, str(outside),
        ])
        with pytest.raises(ManifestError, match="include search roots.*project-local"):
            collect_artifacts(include_manifest, root)

    declared_include_manifest = minimal_manifest(
        "test.external_declared_include", "external declared include", "dut", "kernel.cpp",
    )
    declared_include_manifest.build.include_dirs = [str(outside)]
    with pytest.raises(ManifestError, match="manifest include directories.*project-local"):
        collect_artifacts(declared_include_manifest, root)

    missing_include_manifest = minimal_manifest(
        "test.missing_include_root", "missing include root", "dut", "kernel.cpp",
    )
    missing_include_manifest.build.translation_units[0].arguments.extend([
        "-I", "missing-include-dir",
    ])
    with pytest.raises(ManifestError, match="include search roots.*project-local"):
        collect_artifacts(missing_include_manifest, root)

    response_manifest = minimal_manifest(
        "test.external_response", "external response", "dut", "kernel.cpp",
    )
    response_manifest.build.translation_units[0].arguments.append(
        "@" + str(external_response)
    )
    with pytest.raises(ManifestError, match="response files must be project-local"):
        collect_artifacts(response_manifest, root)

    missing_manifest = minimal_manifest(
        "test.missing_response", "missing response", "dut", "kernel.cpp",
    )
    missing_manifest.build.translation_units[0].arguments.append("@missing.rsp")
    with pytest.raises(ManifestError, match="response file does not exist"):
        collect_artifacts(missing_manifest, root)


@pytest.mark.parametrize("arguments, option", [
    (["--config=clang.cfg"], "--config"),
    (["--config", "clang.cfg"], "--config"),
    (["-resource-dir", "resource"], "-resource-dir"),
    (["-isysroot", "sysroot"], "-isysroot"),
    (["-iwithsysroot", "/usr/include"], "-iwithsysroot"),
    (["-Xclang", "-load", "-Xclang", "plugin.dll"], "-Xclang"),
    (["-fplugin=plugin.dll"], "-fplugin"),
    (["/clang:-load"], "/clang:"),
    (["-include-pch", "pre.pch"], "-include-pch"),
    (["-fmodule-map-file=module.modulemap"], "-fmodule-map-file"),
    (["-fmodule-file=Local=local.pcm"], "-fmodule-file"),
    (["-fmodules", "-fimplicit-module-maps"], "-fmodules"),
    (["-fmodules-cache-path=cache"], "-fmodules-cache-path"),
    (["/external:env:AUDIT_INCLUDE"], "/external:env:"),
    (["-march=native"], "-march=native"),
])
def test_untracked_compiler_context_escape_hatches_fail_closed(
    tmp_path, arguments, option,
):
    with pytest.raises(ManifestError, match=re.escape(option)):
        resolve_compiler_arguments(tmp_path, tmp_path, arguments)


def test_standard_extractor_reports_vfs_overlay_as_unsupported(tmp_path):
    if not LibClangExtractor.available():
        pytest.skip("libclang is optional")
    (tmp_path / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    (tmp_path / "overlay.yaml").write_text(
        "{ 'version': 0, 'roots': [] }\n", encoding="utf-8",
    )
    manifest = minimal_manifest(
        "test.vfs_overlay", "overlay", "dut", "kernel.cpp",
    )
    manifest.build.translation_units[0].arguments.extend([
        "-ivfsoverlay", "overlay.yaml",
    ])
    project = Project(GraphBundle.create(tmp_path, manifest))
    result = project.index()
    assert result.success is False
    assert any(item.code == "source.unsupported_vfs_overlay"
               for item in project.bundle.store.diagnostics(result.snapshot_id))
    assert not project.bundle.store.has_graph(result.snapshot_id)


def test_windows_compile_command_tokenizer_preserves_quoted_path_arguments(tmp_path):
    command = (
        '"C:\\Program Files\\LLVM\\bin\\clang++.exe" '
        '-std=c++17 -I"include dir" /FI"forced dir\\forced.hpp" '
        '-DNAME=\\"demo\\" -c kernel.cpp'
    )
    assert split_compilation_command(command, platform="nt") == [
        "C:\\Program Files\\LLVM\\bin\\clang++.exe",
        "-std=c++17", "-Iinclude dir", "/FIforced dir\\forced.hpp",
        '-DNAME="demo"', "-c", "kernel.cpp",
    ]

    compile_commands = tmp_path / "compile_commands.json"
    compile_commands.write_text(json.dumps([{
        "directory": str(tmp_path), "file": "kernel.cpp", "command": command,
    }]), encoding="utf-8")
    (tmp_path / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    unit = load_compile_commands(compile_commands, tmp_path, platform="nt")[0]
    assert "-Iinclude dir" in unit.arguments
    assert "/FIforced dir\\forced.hpp" in unit.arguments

    assert split_compilation_command(
        "clang++ -I C:\\include\\ -c kernel.cpp", platform="nt",
    ) == ["clang++", "-I", "C:\\include\\", "-c", "kernel.cpp"]
    assert split_compilation_command(
        "clang++ -IC:\\include\\ -c kernel.cpp", platform="nt",
    ) == ["clang++", "-IC:\\include\\", "-c", "kernel.cpp"]


def test_windows_compile_command_tokenizer_round_trips_python_crt_quoting():
    randomizer = random.Random(0x484C5347)
    alphabet = string.ascii_letters + string.digits + " -_./:\\\"\t"
    for _ in range(2_000):
        argv = [
            "".join(randomizer.choice(alphabet)
                    for _ in range(randomizer.randrange(0, 32)))
            for _ in range(randomizer.randrange(1, 9))
        ]
        command = subprocess.list2cmdline(argv)
        assert split_compilation_command(command, platform="nt") == argv, (
            command, argv,
        )


def test_standard_index_honors_compile_directory_response_file_and_space_include(tmp_path):
    if not LibClangExtractor.available():
        pytest.skip("libclang is optional")
    for directory in ("build", "src", "include dir", "global include"):
        (tmp_path / directory).mkdir()
    (tmp_path / "src/kernel.cpp").write_text(
        "#include <dep.hpp>\n#include <global.hpp>\n"
        "#ifndef FORCED_VALUE\n#error forced include missing\n#endif\n"
        "void dut() { int value = DEP_VALUE + GLOBAL_VALUE + FORCED_VALUE; (void)value; }\n",
        encoding="utf-8",
    )
    (tmp_path / "include dir/dep.hpp").write_text(
        "#define DEP_VALUE 1\n", encoding="utf-8",
    )
    (tmp_path / "include dir/forced.hpp").write_text(
        "#define FORCED_VALUE 2\n", encoding="utf-8",
    )
    (tmp_path / "global include/global.hpp").write_text(
        "#define GLOBAL_VALUE 3\n", encoding="utf-8",
    )
    (tmp_path / "build/flags.rsp").write_text(
        '-I"../include dir" -include forced.hpp\n', encoding="utf-8",
    )
    (tmp_path / "global.rsp").write_text(
        '-I"global include"\n', encoding="utf-8",
    )
    (tmp_path / "compile_commands.json").write_text(json.dumps([{
        "directory": str(tmp_path / "build"),
        "file": "../src/kernel.cpp",
        "arguments": [
            "clang++", "-std=c++17", "@flags.rsp", "-c", "../src/kernel.cpp",
        ],
    }]), encoding="utf-8")

    manifest = minimal_manifest(
        "test.compile_database", "compile database", "dut", "src/kernel.cpp",
    )
    manifest.build.compile_commands = "compile_commands.json"
    manifest.build.translation_units = []
    manifest.build.cflags = ["@global.rsp"]
    project = Project(GraphBundle.create(tmp_path, manifest))
    result = project.index()

    assert result.success is True
    snapshot = project.bundle.store.snapshot(result.snapshot_id)
    assert snapshot is not None
    uris = {item.uri for item in project.bundle.store.artifacts(snapshot.id)}
    assert {"build/flags.rsp", "compile_commands.json", "global.rsp",
            "include dir/dep.hpp", "include dir/forced.hpp",
            "global include/global.hpp", "src/kernel.cpp"}.issubset(uris)
    fatal = [item for item in project.bundle.store.diagnostics(snapshot.id)
             if str(item.severity) in {"error", "critical"}]
    assert fatal == []


def test_stage_toolchain_resolution_is_exact_and_part_of_snapshot_identity(tmp_path):
    (tmp_path / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    first = ToolchainContext(
        id="amd.vitis.2024_2", vendor="amd", name="vitis_hls", version="2024.2",
    )
    second = ToolchainContext(
        id="amd.vivado.2024_2", vendor="amd", name="vivado", version="2024.2",
    )
    manifest = minimal_manifest("test.stage_toolchains", "toolchains", "dut", "kernel.cpp")
    manifest.stage_commands = {
        "csynth": ["vitis_hls", "run.tcl"],
        "post_route": ["vivado", "-source", "route.tcl"],
    }
    manifest.toolchains = [first, second]
    manifest.stage_toolchains = {
        "csynth": first.id,
        "post_route": second.id,
    }
    loaded = ProjectManifest.from_dict(json_ready(manifest))
    assert loaded.toolchain_for_stage("csynth").id == first.id
    assert loaded.toolchain_for_stage("post_route").id == second.id

    artifacts = collect_artifacts(loaded, tmp_path)
    baseline = make_snapshot(loaded, artifacts)
    swapped = copy.deepcopy(loaded)
    swapped.stage_toolchains = {"csynth": second.id, "post_route": first.id}
    changed = make_snapshot(swapped, artifacts)
    assert changed.id != baseline.id
    assert changed.toolchain_hash != baseline.toolchain_hash


def test_stage_toolchain_contract_fails_closed_for_zero_duplicate_or_unknown(tmp_path):
    (tmp_path / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    base = minimal_manifest("test.stage_toolchain_errors", "errors", "dut", "kernel.cpp")
    base.stage_commands = {"csynth": ["tool", "run"]}

    with pytest.raises(ValueError, match="at least one declared toolchain"):
        ProjectManifest.from_dict(json_ready(base))

    one = ToolchainContext("tool.one", "test", "one", "1")
    base.toolchains = [one]
    compatible = ProjectManifest.from_dict(json_ready(base))
    assert compatible.toolchain_for_stage("csynth").id == one.id

    duplicate = copy.deepcopy(base)
    duplicate.toolchains = [one, copy.deepcopy(one)]
    duplicate.stage_toolchains = {"csynth": one.id}
    with pytest.raises(ValueError, match="IDs must be unique"):
        ProjectManifest.from_dict(json_ready(duplicate))

    second = ToolchainContext("tool.two", "test", "two", "1")
    multiple = copy.deepcopy(base)
    multiple.toolchains = [one, second]
    with pytest.raises(ValueError, match="explicit stage_toolchains"):
        ProjectManifest.from_dict(json_ready(multiple))

    unknown_id = copy.deepcopy(multiple)
    unknown_id.stage_toolchains = {"csynth": "tool.missing"}
    with pytest.raises(ValueError, match="unknown toolchain"):
        ProjectManifest.from_dict(json_ready(unknown_id))

    unknown_stage = copy.deepcopy(multiple)
    unknown_stage.stage_toolchains = {"csynth": one.id, "rtl_cosim": second.id}
    with pytest.raises(ValueError, match="no matching stage command"):
        ProjectManifest.from_dict(json_ready(unknown_stage))


def test_previous_snapshot_becomes_stale_for_every_identity_input_change(tmp_path):
    changes = ("macro", "top", "directive", "part", "clock", "tool")
    for change in changes:
        root = tmp_path / change
        root.mkdir()
        manifest = _snapshot_manifest(root)
        bundle = GraphBundle.create(root, manifest)
        previous = bundle.snapshot()
        assert bundle.is_stale(previous) is False

        if change == "macro":
            bundle.manifest.build.defines["MODE"] = "changed"
        elif change == "top":
            bundle.manifest.build.top = "dut_changed"
        elif change == "directive":
            (root / "directives.tcl").write_text(
                "set_directive_pipeline -II 4 dut\n", encoding="utf-8"
            )
        elif change == "part":
            bundle.manifest.target.part = "xczu-changed"
        elif change == "clock":
            bundle.manifest.target.clocks = [ClockConstraint("default", 3.5, 0.2)]
        elif change == "tool":
            bundle.manifest.toolchains[0] = replace(
                bundle.manifest.toolchains[0], version="2025.1", build="changed"
            )

        assert bundle.is_stale(previous) is True, change
        current = bundle.snapshot()
        assert current.id != previous.id, change
        assert bundle.store.snapshot(previous.id).id == previous.id


def test_entity_relation_graph_hash_and_query_are_order_independent(tmp_path):
    (tmp_path / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    bundle = GraphBundle.create(
        tmp_path, minimal_manifest("test.determinism", "determinism", "dut", "kernel.cpp")
    )
    snapshot = bundle.snapshot()

    kernel_a = Entity("hls.kernel", "dut", snapshot.id, qualified_name="dut", stage="ast")
    process_a = Entity(
        "hls.process", "compute", snapshot.id, qualified_name="dut::compute",
        stage="hls_ir", aliases=["multiply"],
    )
    edge_a = Relation(kernel_a.id, process_a.id, "hls.contains", snapshot.id, stage="hls_ir")

    graph_a = CanonicalGraph(snapshot.id, metadata={"top": "dut"})
    graph_a.add_entity(kernel_a)
    graph_a.add_entity(process_a)
    graph_a.add_relation(edge_a)

    kernel_b = Entity("hls.kernel", "dut", snapshot.id, qualified_name="dut", stage="ast")
    process_b = Entity(
        "hls.process", "compute", snapshot.id, qualified_name="dut::compute",
        stage="hls_ir", aliases=["multiply"],
    )
    edge_b = Relation(kernel_b.id, process_b.id, "hls.contains", snapshot.id, stage="hls_ir")
    graph_b = CanonicalGraph(snapshot.id, metadata={"top": "dut"})
    graph_b.add_entity(process_b)
    graph_b.add_entity(kernel_b)
    graph_b.add_relation(edge_b)

    assert (kernel_a.id, process_a.id, edge_a.id) == (kernel_b.id, process_b.id, edge_b.id)
    assert graph_a.to_dict() == graph_b.to_dict()
    assert graph_a.graph_hash == graph_b.graph_hash

    bundle.store.save_graph(graph_a)
    loaded = bundle.store.load_graph(snapshot.id)
    assert loaded.graph_hash == graph_a.graph_hash

    service = CoreService(bundle, snapshot.id)
    first = service.query(QuerySpec("compute", limit=10)).to_dict()
    second = service.query(QuerySpec("compute", limit=10)).to_dict()
    assert first == second
    assert first["items"][0]["entity_id"] == process_a.id


def test_private_source_is_reference_only_and_snippets_require_explicit_authorization(tmp_path):
    secret = "PRIVATE_SOURCE_SENTINEL_5f8d17"
    (tmp_path / "kernel.cpp").write_text(
        f"// {secret}\nvoid dut() {{}}\n", encoding="utf-8"
    )
    manifest = minimal_manifest("test.private", "private source", "dut", "kernel.cpp")
    manifest.artifact_paths[0]["license"] = "Proprietary"
    bundle = GraphBundle.create(tmp_path, manifest)
    snapshot = bundle.snapshot()
    artifact = bundle.store.artifacts(snapshot.id)[0]

    kernel = Entity("hls.kernel", "dut", snapshot.id, qualified_name="dut", stage="ast")
    graph = CanonicalGraph(snapshot.id, metadata={"top": "dut"})
    graph.add_entity(kernel)
    bundle.store.save_graph(graph)

    persisted = b"".join(path.read_bytes() for path in bundle.root.rglob("*") if path.is_file())
    assert secret.encode() not in persisted
    assert artifact.uri == "kernel.cpp"
    assert artifact.size > 0 and len(artifact.sha256) == 64
    assert str(artifact.access) == "private"

    with pytest.raises(PermissionError, match="explicit authorization"):
        bundle.source_snippet(artifact.id, 1, 1)
    assert bundle.source_snippet(artifact.id, 1, 1, allow_private=True) == f"// {secret}"

    core = CoreService(bundle, snapshot.id)
    rest = RestApplication(core)
    mcp = ReadOnlyMcpService(core)
    public_payloads = [
        rest.dispatch("GET", "/api/v1/graph").body,
        rest.dispatch("GET", "/api/v1/overview").body,
        mcp.overview(),
        mcp.search("dut"),
        mcp.evidence(kernel.id),
    ]
    assert all(secret not in json.dumps(value, ensure_ascii=False) for value in public_payloads)
    assert json_ready(artifact).get("content") is None
