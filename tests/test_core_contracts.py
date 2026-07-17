from __future__ import annotations

import copy
import json
from dataclasses import replace

import pytest

from hlsgraph.api import RestApplication
from hlsgraph.bundle import GraphBundle
from hlsgraph.graph import CanonicalGraph
from hlsgraph.manifest import collect_artifacts, make_snapshot, minimal_manifest
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
