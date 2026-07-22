from __future__ import annotations

import copy
import json
import hashlib
from pathlib import Path
import sys
import types

import pytest

from hlsgraph import (
    AccessPolicy,
    ArtifactRef,
    AuthorityClass,
    CanonicalGraph,
    DatasetManifest,
    Entity,
    FEATURE_SCHEMA_VERSION,
    LabelSpec,
    Observation,
    Project,
    Relation,
    FailureClass,
    RunStatus,
    SourceAnchor,
    ToolOutputSpec,
    ToolRun,
    ToolchainContext,
)
from hlsgraph.bundle import GraphBundle
from hlsgraph.evidence_policy import TOOL_EVIDENCE_POLICY_VERSION
from hlsgraph.export import export_dataset, export_graph_json, to_pyg_data
from hlsgraph.export.ml import _public_target, _public_toolchain
from hlsgraph.export.pyg import _feature_graph
from hlsgraph.extract.vitis import _safe_source_location
from hlsgraph.manifest import minimal_manifest
from hlsgraph.render.projection import to_render_data
from hlsgraph.store import StoreError
from tests.attested_run_support import commit_attested
from tests.typed_report_support import (
    parsed_report_observation,
    write_csynth_xml,
    write_vivado_timing,
)


def test_source_pragma_comment_never_enters_graph_db_or_exports(tmp_path: Path) -> None:
    sentinel = "PRIVATE_PRAGMA_COMMENT_SENTINEL_6e2a"
    source = (
        "void dut() {\n"
        f"#pragma HLS PIPELINE II=1 // {sentinel}\n"
        "  for (int i = 0; i < 4; ++i) {}\n"
        "}\n"
    )
    (tmp_path / "kernel.cpp").write_text(source, encoding="utf-8")
    project = Project(GraphBundle.create(
        tmp_path, minimal_manifest("test.pragma_privacy", "pragma", "dut", "kernel.cpp")
    ))
    indexed = project.index(degraded=True)
    assert indexed.success
    graph = project.service().graph()
    directive = next(item for item in graph.entities.values()
                     if item.kind == "hls.directive")
    assert directive.attrs["options"] == {"ii": 1}

    graph_output = export_graph_json(project.bundle, indexed.snapshot_id,
                                     tmp_path / "graph.json")
    dataset_dir = tmp_path / "dataset"
    export_dataset(project.bundle, indexed.snapshot_id, dataset_dir)
    serialized = graph_output.read_bytes() + project.bundle.store.path.read_bytes()
    serialized += b"".join(path.read_bytes() for path in dataset_dir.iterdir())
    assert sentinel.encode() not in serialized


def test_mlir_locations_normalize_project_paths_and_redact_external_paths(
    tmp_path: Path,
) -> None:
    source = tmp_path / "kernel.cpp"
    source.write_text("void dut() {}\n", encoding="utf-8")
    external = (tmp_path.parent / "private-external.cpp").resolve().as_posix()
    mlir = tmp_path / "dut.mlir"
    mlir.write_text(
        "module {\n"
        "  func.func @dut() {\n"
        f'    %0 = arith.constant 0 : i32 loc("{source.resolve().as_posix()}":1:1)\n'
        f'    %1 = arith.constant 1 : i32 loc("{external}":7:3)\n'
        "  }\n"
        "}\n",
        encoding="utf-8",
    )
    manifest = minimal_manifest(
        "test.mlir_location_privacy", "MLIR location privacy", "dut", "kernel.cpp",
    )
    manifest.artifact_paths.append({
        "path": "dut.mlir", "kind": "ir.mlir", "role": "hls_ir",
        "access": "project", "license": "Apache-2.0",
    })
    project = Project(GraphBundle.create(tmp_path, manifest))
    assert project.index(degraded=True).success

    graph = project.service().graph()
    locations = {
        anchor.ir_location
        for entity in graph.entities.values()
        for anchor in entity.anchors
        if anchor.ir_location
    }
    serialized = json.dumps(graph.to_dict(), sort_keys=True)
    assert 'loc("kernel.cpp":1:1)' in locations
    assert 'loc("<external>":7:3)' in locations
    assert source.resolve().as_posix() not in serialized
    assert external not in serialized


def test_source_anchor_keeps_relative_and_symbolic_locations() -> None:
    relative = SourceAnchor(
        "artifact.valid", ir_location='loc("src/kernel.cpp":18:5)',
        ambiguity="many-to-many mapping remains explicit",
    )
    symbolic = SourceAnchor("artifact.valid", ir_location="!dbg !4")

    assert relative.ir_location == 'loc("src/kernel.cpp":18:5)'
    assert relative.ambiguity == "many-to-many mapping remains explicit"
    assert symbolic.ir_location == "!dbg !4"


@pytest.mark.parametrize(("field", "value"), [
    ("ir_location", "C" + ":/private/kernel.cpp:7:2"),
    ("ir_location", 'loc("/var/private/kernel.cpp":7:2)'),
    ("ir_location", "\\\\" + "private-host\\share\\kernel.cpp"),
    ("symbol", "\\" + "Users\\alice\\private_symbol"),
    ("ambiguity", "resolved from /" + "home/alice/private/kernel.cpp"),
])
def test_source_anchor_redacts_host_absolute_paths(field: str, value: str) -> None:
    anchor = SourceAnchor("artifact.valid", **{field: value})
    normalized = getattr(anchor, field)
    assert normalized is not None
    assert normalized.startswith("redacted.sha256:")
    assert value not in json.dumps({field: normalized})


@pytest.mark.parametrize("value", [True, False, 1.5, "1"])
def test_source_anchor_rejects_non_integer_source_positions(value: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        SourceAnchor("artifact.valid", start_line=value)  # type: ignore[arg-type]


def test_source_anchor_requires_namespaced_artifact_id() -> None:
    with pytest.raises(ValueError, match="artifact_id"):
        SourceAnchor("invalid artifact id", start_line=1)


def test_mutated_absolute_anchor_fails_closed_at_sqlite_boundary(tmp_path: Path) -> None:
    (tmp_path / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    bundle = GraphBundle.create(
        tmp_path,
        minimal_manifest("test.anchor_store_privacy", "anchor privacy", "dut", "kernel.cpp"),
    )
    snapshot = bundle.snapshot()
    source_artifact = next(
        item for item in bundle.store.artifacts(snapshot.id)
        if item.uri == "kernel.cpp"
    )
    anchor = SourceAnchor(
        source_artifact.id, start_line=1, ir_location='loc("kernel.cpp":1:1)',
    )
    graph = CanonicalGraph(snapshot.id)
    graph.add_entity(Entity(
        "hls.kernel", "dut", snapshot.id, stage="ast", anchors=[anchor],
    ))

    # Mutable dataclass members are revalidated before any graph row is written.
    anchor.ir_location = "C" + ":/private/kernel.cpp:1:1"
    with pytest.raises(StoreError, match="canonical normalized"):
        bundle.store.save_graph(graph)
    assert bundle.store.has_graph(snapshot.id) is False


@pytest.mark.parametrize("value", [
    "kernel.cpp:12", "src/kernel.cpp:12:7",
])
def test_schedule_source_location_accepts_only_relative_anchors(value: str) -> None:
    assert _safe_source_location(value) == value


@pytest.mark.parametrize("value", [
    "PRIVATE SOURCE FRAGMENT WITHOUT SEMICOLON",
    "C" + ":/private/kernel.cpp:12:2",
    "/var/private-project/kernel.cpp:12:2",
    "../private/kernel.cpp:12:2",
])
def test_schedule_source_location_rejects_prose_and_private_paths(value: str) -> None:
    assert _safe_source_location(value) is None


def test_render_never_promotes_ast_containment_to_hardware_topology() -> None:
    graph = CanonicalGraph("snapshot.render-boundary", metadata={"top": "dut"})
    kernel = graph.add_entity(Entity(
        "hls.kernel", "dut", graph.snapshot_id, stage="ast",
        authority=AuthorityClass.STATIC_FACT,
    ))
    loop = graph.add_entity(Entity(
        "hls.loop", "loop", graph.snapshot_id, stage="ast",
        authority=AuthorityClass.STATIC_FACT,
    ))
    graph.add_relation(Relation(
        kernel.id, loop.id, "hls.contains", graph.snapshot_id, stage="ast",
        authority=AuthorityClass.STATIC_FACT,
    ))
    rendered = to_render_data(graph, [], [])
    assert rendered["edges"] == []
    assert rendered["meta"]["view"] == "architecture_evidence_incomplete"
    assert rendered["meta"]["incomplete"] is True


def test_render_never_promotes_ast_stream_edge_to_hardware_topology() -> None:
    graph = CanonicalGraph("snapshot.render-stream-boundary", metadata={"top": "dut"})
    source = graph.add_entity(Entity("hls.process", "source", graph.snapshot_id,
                                     stage="ast"))
    sink = graph.add_entity(Entity("hls.process", "sink", graph.snapshot_id,
                                   stage="ast"))
    graph.add_relation(Relation(
        source.id, sink.id, "hls.streams_to", graph.snapshot_id,
        stage="ast", authority=AuthorityClass.STATIC_FACT,
    ))
    rendered = to_render_data(graph, [], [])
    assert rendered["edges"] == []
    assert rendered["meta"]["view"] == "architecture_evidence_incomplete"


def _snapshot_with_graph(bundle: GraphBundle, source_text: str) -> tuple[str, str]:
    (bundle.project_root / "kernel.cpp").write_text(source_text, encoding="utf-8")
    snapshot = bundle.snapshot(extraction_hash="release-export-test")
    graph = CanonicalGraph(snapshot.id)
    kernel = graph.add_entity(Entity("hls.kernel", "dut", snapshot.id))
    bundle.store.save_graph(graph)
    return snapshot.id, kernel.id


def test_ml_export_checks_label_semantics_and_split_leakage(tmp_path: Path) -> None:
    (tmp_path / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    bundle = GraphBundle.create(
        tmp_path, minimal_manifest("test.ml_release", "ml", "dut", "kernel.cpp")
    )
    first, first_kernel = _snapshot_with_graph(bundle, "void dut() {}\n")
    observation = Observation(
        snapshot_id=first, subject_id=first_kernel,
        predicate="qor.latency_cycles", value=4, unit="cycle",
        stage="schedule", authority=AuthorityClass.SYNTHETIC,
    )
    bundle.store.add_observations([observation])
    second, _ = _snapshot_with_graph(bundle, "void dut() { int x = 1; (void)x; }\n")

    mismatch = DatasetManifest(
        dataset_id="dataset.mismatch", feature_schema_version=FEATURE_SCHEMA_VERSION,
        snapshot_ids=[first], labels=[LabelSpec(
            "latency", first, observation.id,
            "qor.latency_cycles", "post_route", "cycle"
        )],
    )
    with pytest.raises(ValueError, match="predicate/stage/unit"):
        export_dataset(bundle, first, tmp_path / "mismatch", mismatch)

    leakage = DatasetManifest(
        dataset_id="dataset.leakage", feature_schema_version=FEATURE_SCHEMA_VERSION,
        snapshot_ids=[first, second], splits={first: "train", second: "test"},
        kernel_families={first: "family.same", second: "family.same"},
        dedup_groups={first: "dedup.first", second: "dedup.second"},
        licenses={first: "Apache-2.0", second: "Apache-2.0"},
    )
    with pytest.raises(ValueError, match="kernel family crosses"):
        export_dataset(bundle, first, tmp_path / "leakage", leakage)

    multi = DatasetManifest(
        dataset_id="dataset.multi", feature_schema_version=FEATURE_SCHEMA_VERSION,
        snapshot_ids=[first, second],
    )
    manifest = export_dataset(bundle, first, tmp_path / "multi", multi)
    assert set(manifest["snapshots"]) == {first, second}
    node_snapshots = {json.loads(line)["snapshot_id"] for line in
                      (tmp_path / "multi" / "nodes.jsonl").read_text().splitlines()}
    assert node_snapshots == {first, second}


def test_ml_export_has_explicit_stage_and_attribute_firewalls(tmp_path: Path) -> None:
    (tmp_path / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    bundle = GraphBundle.create(
        tmp_path, minimal_manifest("test.ml_firewall", "ml", "dut", "kernel.cpp")
    )
    snapshot = bundle.snapshot()
    kernel = Entity(
        "hls.directive", "PIPELINE", snapshot.id, stage="ast",
        authority=AuthorityClass.DECLARED_CONSTRAINT,
        attrs={
            "bitwidth": 32,
            "dims": [1, 16, 2_147_483_647],
            "plugin_secret_qor": 99,
            "lut_used": 91, "bram": 92, "dsp": 93,
            "fmax_mhz": 94, "throughput": 95,
            "plugin_container": {
                "operation": "mul", "lut_used": 101, "bram": 102,
                "dsp": 103, "fmax_mhz": 104, "throughput": 105,
            },
            "options": {
                "ii": 1, "factor": 4, "variable": "input", "impl": "dsp",
                "flags": ["rewind", "off", "throughput"],
                "latency": {"lut_used": 106},
                "lut_used": 107, "bram": 108, "dsp": 109,
                "fmax_mhz": 110, "throughput": 111,
                "plugin": {"nested": {"lut_used": 112, "bram": 113}},
            },
        },
    )
    scheduled = Entity(
        "hls.process", "scheduled", snapshot.id, stage="schedule",
        attrs={"operation": "mul", "plugin_secret_qor": 123},
    )
    graph = CanonicalGraph(snapshot.id)
    graph.add_entity(kernel)
    graph.add_entity(scheduled)
    bundle.store.save_graph(graph)

    dataset = DatasetManifest(
        dataset_id="dataset.firewall", feature_schema_version=FEATURE_SCHEMA_VERSION,
        snapshot_ids=[snapshot.id],
    )
    # Even an explicitly top-level-allowlisted plugin container stays excluded
    # until a future feature-schema version defines its child fields.
    dataset.feature_attribute_allowlist.append("plugin_container")
    dataset.feature_attribute_allowlist.append("dims")
    dataset.feature_attribute_allowlist.extend(
        ["lut_used", "bram", "dsp", "fmax_mhz", "throughput"]
    )
    output = tmp_path / "firewall"
    export_dataset(bundle, snapshot.id, output, dataset)
    nodes = [json.loads(line) for line in
             (output / "nodes.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [item["node_id"] for item in nodes] == [kernel.id]
    expected_features = {
        "bitwidth": 32,
        "dims": [1, 16, 2_147_483_647],
        "options": {
            "factor": 4,
            "flags": ["rewind", "off"],
            "ii": 1,
            "impl": "dsp",
            "variable": "input",
        },
    }
    assert nodes[0]["features"] == expected_features
    spec = json.loads((output / "feature_spec.json").read_text(encoding="utf-8"))
    assert "schedule" not in spec["feature_stages"]
    assert "plugin_secret_qor" not in spec["feature_attribute_allowlist"]
    nested_schema = spec["static_feature_schema"]
    assert nested_schema["unknown_container_policy"] == "exclude"
    assert nested_schema["unknown_container_key_policy"] == "exclude"
    assert nested_schema["containers"]["options"]["additionalProperties"] is False
    assert nested_schema["containers"]["dims"] == {
        "type": "positive_integer_list",
        "min_items": 1,
        "max_items": 16,
        "min_value": 1,
        "max_value": 2_147_483_647,
    }
    assert "ii" in nested_schema["containers"]["options"]["properties"]
    assert all(name not in nested_schema["containers"]["options"]["properties"]
               for name in ("lut_used", "bram", "dsp", "fmax_mhz", "throughput"))
    assert all(name in nested_schema["top_level_excluded_outcome_prefixes"]
               for name in ("lut", "bram", "dsp", "fmax", "throughput"))
    pyg_nodes, _pyg_edges, _ = _feature_graph(bundle, snapshot.id)
    assert [item.id for item in pyg_nodes] == [kernel.id]

    with pytest.raises(ValueError, match="embedded body"):
        DatasetManifest(
            dataset_id="dataset.private_body",
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            snapshot_ids=[snapshot.id],
            metadata={"source_text": "PRIVATE_DATASET_SENTINEL"},
        )


@pytest.mark.parametrize(
    "invalid_dims",
    [
        [],
        [0],
        [-1, 2],
        [True, 2],
        [1.0, 2],
        ["1", 2],
        [None, 2],
        [2_147_483_648],
        list(range(1, 18)),
        (1, 2),
        {"rows": 1, "cols": 2},
        "1,2",
    ],
)
def test_dims_feature_schema_rejects_invalid_or_non_list_values(
    invalid_dims: object,
) -> None:
    from hlsgraph.export.ml import _static_features

    assert _static_features({"dims": invalid_dims}, {"dims"}) == {}


def test_pyg_uses_the_same_nested_feature_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    bundle = GraphBundle.create(
        tmp_path, minimal_manifest("test.pyg_firewall", "pyg", "dut", "kernel.cpp")
    )
    snapshot = bundle.snapshot()
    kernel = Entity(
        "hls.directive", "PIPELINE", snapshot.id, stage="ast",
        authority=AuthorityClass.DECLARED_CONSTRAINT,
        attrs={
            "options": {
                "ii": 2, "factor": 8, "impl": "bram",
                "flags": ["rewind", "throughput"],
                "lut_used": 1, "bram": 2, "dsp": 3,
                "fmax_mhz": 4, "throughput": 5,
            },
        },
    )
    graph = CanonicalGraph(snapshot.id)
    graph.add_entity(kernel)
    graph.add_relation(Relation(
        kernel.id, kernel.id, "test.self", snapshot.id, stage="ast",
        attrs={"options": {"ii": 3, "lut_used": 9, "throughput": 10}},
    ))
    bundle.store.save_graph(graph)

    fake_torch = types.ModuleType("torch")
    fake_torch.long = "long"  # type: ignore[attr-defined]
    fake_torch.tensor = lambda value, dtype=None: value  # type: ignore[attr-defined]
    fake_torch.empty = lambda shape, dtype=None: {"shape": shape}  # type: ignore[attr-defined]

    class FakeData:
        def __init__(self, **values):
            self.__dict__.update(values)

    fake_geometric = types.ModuleType("torch_geometric")
    fake_geometric.__path__ = []  # type: ignore[attr-defined]
    fake_data = types.ModuleType("torch_geometric.data")
    fake_data.Data = FakeData  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch_geometric", fake_geometric)
    monkeypatch.setitem(sys.modules, "torch_geometric.data", fake_data)

    data = to_pyg_data(bundle, snapshot.id)
    assert data.node_features == [{
        "options": {
            "factor": 8, "flags": ["rewind"], "ii": 2, "impl": "bram",
        },
    }]
    assert data.edge_features == [{}]
    assert data.static_feature_schema["containers"]["options"][
        "additionalProperties"
    ] is False


def test_ml_export_revalidates_mutated_dataset_and_labels(tmp_path: Path) -> None:
    (tmp_path / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    bundle = GraphBundle.create(
        tmp_path, minimal_manifest("test.dataset_mutation", "mutation", "dut", "kernel.cpp")
    )
    snapshot = bundle.snapshot()
    graph = CanonicalGraph(snapshot.id)
    kernel = Entity("hls.kernel", "dut", snapshot.id, stage="ast")
    graph.add_entity(kernel)
    bundle.store.save_graph(graph)

    sentinel = "PRIVATE_MUTATED_DATASET_SOURCE_SENTINEL_92d7"
    mutated_metadata = DatasetManifest(
        dataset_id="dataset.mutated_metadata", feature_schema_version=FEATURE_SCHEMA_VERSION,
        snapshot_ids=[snapshot.id],
    )
    mutated_metadata.metadata["source_text"] = sentinel
    metadata_output = tmp_path / "mutated-metadata"
    with pytest.raises(ValueError, match="embedded body"):
        export_dataset(bundle, snapshot.id, metadata_output, mutated_metadata)
    assert not metadata_output.exists()

    mutated_label = LabelSpec(
        label_id="latency", snapshot_id=snapshot.id,
        observation_id="observation.placeholder",
        predicate="qor.latency_cycles", stage="schedule",
    )
    mutated_label.observation_id = None
    label_dataset = DatasetManifest(
        dataset_id="dataset.mutated_label", feature_schema_version=FEATURE_SCHEMA_VERSION,
        snapshot_ids=[snapshot.id], labels=[mutated_label],
    )
    label_output = tmp_path / "mutated-label"
    with pytest.raises(ValueError, match="present label"):
        export_dataset(bundle, snapshot.id, label_output, label_dataset)
    assert not label_output.exists()


def test_parquet_receives_the_same_sanitized_nested_features(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    bundle = GraphBundle.create(
        tmp_path, minimal_manifest("test.parquet_firewall", "parquet", "dut", "kernel.cpp")
    )
    snapshot = bundle.snapshot()
    graph = CanonicalGraph(snapshot.id)
    graph.add_entity(Entity(
        "hls.directive", "PIPELINE", snapshot.id, stage="ast",
        authority=AuthorityClass.DECLARED_CONSTRAINT,
        attrs={"options": {
            "ii": 1, "factor": 2, "impl": "dsp",
            "lut_used": 10, "bram": 11, "dsp": 12,
            "fmax_mhz": 13, "throughput": 14,
            "plugin": {"throughput": 15},
        }},
    ))
    bundle.store.save_graph(graph)

    fake_arrow = types.ModuleType("pyarrow")
    fake_arrow.__path__ = []  # type: ignore[attr-defined]

    class FakeTable:
        @staticmethod
        def from_pylist(rows):
            return rows

    fake_arrow.Table = FakeTable  # type: ignore[attr-defined]
    fake_arrow.table = lambda _columns: []  # type: ignore[attr-defined]
    fake_parquet = types.ModuleType("pyarrow.parquet")

    def write_table(rows, path):
        Path(path).write_text(json.dumps(rows, sort_keys=True), encoding="utf-8")

    fake_parquet.write_table = write_table  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyarrow", fake_arrow)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", fake_parquet)

    output = tmp_path / "parquet-dataset"
    export_dataset(bundle, snapshot.id, output, format="parquet")
    rows = json.loads((output / "nodes.parquet").read_text(encoding="utf-8"))
    assert rows[0]["features"] == {
        "options": {"factor": 2, "ii": 1, "impl": "dsp"},
    }


def test_ml_export_closes_redacted_run_constraint_and_toolchain_provenance(
    tmp_path: Path,
) -> None:
    (tmp_path / "constraints").mkdir()
    xdc_text = "create_clock -period 5.0 [get_ports clk]\n"
    (tmp_path / "constraints/timing.xdc").write_text(xdc_text, encoding="utf-8")
    (tmp_path / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    manifest = minimal_manifest(
        "test.ml_provenance", "provenance", "dut", "kernel.cpp"
    )
    manifest.constraints.xdc_files = ["constraints/timing.xdc"]
    manifest.constraints.performance = {"target_ii": 1}
    private_interface = "PRIVATE_SHORT_INTERFACE_SENTINEL_51a4"
    private_memory_kind = "PRIVATE_SHORT_MEMORY_SENTINEL_76c2"
    manifest.constraints.interfaces = {"mode": private_interface}
    manifest.constraints.assumptions = ["private board setup assumption"]
    private_executable = str((tmp_path / "private/tool/vitis_hls.exe").resolve())
    private_working_directory = str((tmp_path / "private/workspace").resolve())
    private_platform = str((tmp_path / "private/platform.xpfm").resolve())
    private_clock_source = str((tmp_path / "private/timing.xdc").resolve())
    manifest.target.platform = private_platform
    manifest.target.clocks[0].source = private_clock_source
    manifest.target.metadata = {"install_root": private_working_directory}
    manifest.target.memory_topology = [{"kind": private_memory_kind, "banks": 4}]
    manifest.toolchains = [ToolchainContext(
        id="amd.vitis.2024_2", vendor="amd", name="vitis_hls", version="2024.2",
        executable=private_executable,
        environment_hash="e" * 64,
    )]
    manifest.stage_commands = {
        "csynth": [private_executable, "-f", "private.tcl"],
    }
    manifest.stage_toolchains = {"csynth": "amd.vitis.2024_2"}
    bundle = GraphBundle.create(tmp_path, manifest)
    snapshot = bundle.snapshot()
    graph = CanonicalGraph(snapshot.id)
    kernel = Entity("hls.kernel", "dut", snapshot.id, stage="ast")
    graph.add_entity(kernel)
    bundle.store.save_graph(graph)

    secret = "PRIVATE_RUN_MESSAGE_SENTINEL_38f1"
    run = ToolRun(
        snapshot_id=snapshot.id, stage="csynth", backend="local",
        request_hash="a" * 64, toolchain_id="amd.vitis.2024_2",
        status=RunStatus.FAILED,
        command=[private_executable, "--secret-argument"],
        working_directory=private_working_directory,
        environment_hash="e" * 64,
        failure_class=FailureClass.LICENSE, exit_code=1, message=secret,
        metadata={
            "authority": "tool_observation", "tool_truth": False,
            "fresh_tool_truth": False, "campaign_id": "campaign.failed",
            "workload_id": "workload.golden",
        },
    )
    report_bytes = b"sanitized partial report"
    report = ArtifactRef(
        kind="amd.vitis.report", uri="reports/partial.json",
        sha256=hashlib.sha256(report_bytes).hexdigest(), size=len(report_bytes),
        role="tool_output", access=AccessPolicy.PROJECT,
        producer_run_id=run.id, license="LicenseRef-AMD-Output",
    )
    run.output_artifact_ids = [report.id]
    observation = Observation(
        snapshot_id=snapshot.id, subject_id=kernel.id,
        predicate="qor.partial_latency", value=10, unit="cycle", stage="csynth",
        authority=AuthorityClass.TOOL_OBSERVATION,
        run_id=run.id, artifact_id=report.id,
    )
    bundle.store.commit_run_result(
        run=run, artifacts=[report], observations=[observation]
    )
    dataset = DatasetManifest(
        dataset_id="dataset.provenance", feature_schema_version=FEATURE_SCHEMA_VERSION,
        snapshot_ids=[snapshot.id], labels=[
            LabelSpec(
                label_id="failed_timing", snapshot_id=snapshot.id,
                observation_id=None,
                predicate="qor.post_route_wns", stage="post_route", unit="ns",
                mask=False, missing_reason="tool_failure", censored=True,
            ),
        ],
    )

    output = tmp_path / "provenance-dataset"
    exported = export_dataset(bundle, snapshot.id, output, dataset)
    runs = [json.loads(line) for line in
            (output / "runs.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(runs) == 1
    assert runs[0]["id"] == run.id
    assert runs[0]["status"] == "failed"
    assert runs[0]["failure_class"] == "license"
    assert runs[0]["campaign_id"] == "campaign.failed"
    assert runs[0]["workload_id"] == "workload.golden"
    assert runs[0]["tool_truth"] is False
    assert all(key not in runs[0] for key in ("command", "working_directory", "message"))
    assert secret not in (output / "runs.jsonl").read_text(encoding="utf-8")

    observations = [json.loads(line) for line in
                    (output / "observations.jsonl").read_text().splitlines()]
    nontruth_observations = [json.loads(line) for line in
                            (output / "nontruth_observations.jsonl").read_text().splitlines()]
    artifacts = [json.loads(line) for line in
                 (output / "artifacts.jsonl").read_text().splitlines()]
    assert observations == []
    assert nontruth_observations[0]["run_id"] == run.id
    assert nontruth_observations[0]["tool_truth"] is False
    assert nontruth_observations[0]["nontruth_reason"] == (
        "producer_does_not_claim_tool_truth"
    )
    assert next(item for item in artifacts if item["artifact_id"] == report.id)[
        "producer_run_id"
    ] == run.id
    assert exported["row_counts"]["runs"] == 1
    assert "runs.jsonl" in exported["file_integrity"]

    snapshot_manifest = exported["snapshots"][snapshot.id]
    constraints = snapshot_manifest["constraints"]
    xdc_artifact = next(
        item for item in bundle.store.artifacts(snapshot.id)
        if item.uri == "constraints/timing.xdc"
    )
    assert constraints["xdc_artifacts"] == [{
        "artifact_id": xdc_artifact.id,
        "sha256": xdc_artifact.sha256,
        "size": xdc_artifact.size,
        "access": str(xdc_artifact.access),
        "license": xdc_artifact.license,
    }]
    serialized_constraints = json.dumps(constraints, sort_keys=True)
    assert "constraints/timing.xdc" not in serialized_constraints
    assert xdc_text.strip() not in serialized_constraints
    assert "private board setup assumption" not in serialized_constraints
    assert private_interface not in serialized_constraints
    assert snapshot_manifest["stage_toolchains"] == {
        "csynth": "amd.vitis.2024_2"
    }
    public_target = snapshot_manifest["target_profile"]
    assert public_target["platform"] is None
    assert public_target["platform_hash"]
    assert public_target["clocks"][0]["source_hash"]
    serialized_target = json.dumps(public_target, sort_keys=True)
    assert private_platform not in serialized_target
    assert private_clock_source not in serialized_target
    assert private_working_directory not in serialized_target
    assert private_memory_kind not in serialized_target
    feature_spec = json.loads(
        (output / "feature_spec.json").read_text(encoding="utf-8")
    )
    assert feature_spec["provenance_tables"] == ["runs", "artifacts"]
    assert feature_spec["provenance_tables_are_input_features"] is False
    assert feature_spec["nontruth_tables"] == ["nontruth_observations"]


def test_ml_export_rejects_dangling_run_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    bundle = GraphBundle.create(
        tmp_path, minimal_manifest("test.dangling_run", "dangling", "dut", "kernel.cpp")
    )
    snapshot = bundle.snapshot()
    graph = CanonicalGraph(snapshot.id)
    kernel = Entity("hls.kernel", "dut", snapshot.id, stage="ast")
    graph.add_entity(kernel)
    bundle.store.save_graph(graph)
    run = ToolRun(
        snapshot_id=snapshot.id, stage="csynth", backend="local",
        request_hash="b" * 64, status=RunStatus.FAILED,
        failure_class=FailureClass.TIMEOUT,
    )
    report_bytes = b"partial"
    report = ArtifactRef(
        kind="amd.vitis.report", uri="reports/partial.json",
        sha256=hashlib.sha256(report_bytes).hexdigest(), size=len(report_bytes),
        producer_run_id=run.id,
    )
    run.output_artifact_ids = [report.id]
    observation = Observation(
        snapshot_id=snapshot.id, subject_id=kernel.id,
        predicate="qor.partial", value=1, stage="csynth",
        authority=AuthorityClass.TOOL_OBSERVATION, run_id=run.id,
        artifact_id=report.id,
    )
    bundle.store.commit_run_result(
        run=run, artifacts=[report], observations=[observation]
    )

    monkeypatch.setattr(type(bundle.store), "runs", lambda _self, _snapshot_id: [])
    output = tmp_path / "dangling-dataset"
    with pytest.raises(ValueError, match="dangling run provenance"):
        export_dataset(bundle, snapshot.id, output)
    assert not output.exists()


def test_ml_export_rejects_run_backed_label_after_report_cas_is_lost(
    tmp_path: Path,
) -> None:
    (tmp_path / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    manifest = minimal_manifest(
        "test.label_report_integrity", "label report", "dut", "kernel.cpp"
    )
    manifest.toolchains = [ToolchainContext(
        id="test.fixture_tool", vendor="test", name="fixture_tool", version="1",
        environment_hash="e" * 64,
    )]
    manifest.stage_commands = {"csynth": ["tool", "--report"]}
    manifest.stage_toolchains = {"csynth": "test.fixture_tool"}
    manifest.stage_outputs = {"csynth": [ToolOutputSpec(
        path="reports/latency.xml", kind="amd.vitis.csynth_xml",
        role="verification_report",
    )]}
    bundle = GraphBundle.create(tmp_path, manifest)
    snapshot = bundle.snapshot()
    graph = CanonicalGraph(snapshot.id)
    kernel = Entity("hls.kernel", "dut", snapshot.id, stage="ast")
    graph.add_entity(kernel)
    bundle.store.save_graph(graph)

    run = ToolRun(
        snapshot_id=snapshot.id, stage="csynth", backend="runner.local",
        request_hash="c" * 64, toolchain_id="test.fixture_tool",
        status=RunStatus.SUCCEEDED,
        command=list(manifest.stage_commands["csynth"]), working_directory=".",
        environment_hash="e" * 64,
        input_artifact_ids=[
            item.id for item in bundle.store.artifacts(snapshot.id)
            if item.producer_run_id is None
        ],
        exit_code=0,
        metadata={
            "authority": "tool_observation", "fresh_execution": True,
            "fresh_tool_truth": True, "tool_truth": True,
        },
    )
    report_source = tmp_path / "latency.xml"
    write_csynth_xml(report_source, latency=42)
    report, retained_path, _created = bundle.prepare_managed_artifact(
        report_source, kind="amd.vitis.csynth_xml", role="verification_report",
        producer_run_id=run.id,
        metadata={"declared_output_path": "reports/latency.xml"},
    )
    run.output_artifact_ids = [report.id]
    observation = parsed_report_observation(
        bundle, report, predicate="qor.latency_best_cycles", value=42,
        subject_id=kernel.id, run_id=run.id,
    )
    commit_attested(bundle,
        run=run, artifacts=[report], observations=[observation],
    )
    dataset = DatasetManifest(
        dataset_id="dataset.label_report_integrity", feature_schema_version=FEATURE_SCHEMA_VERSION,
        snapshot_ids=[snapshot.id], labels=[LabelSpec(
            label_id="latency", snapshot_id=snapshot.id,
            observation_id=observation.id,
            predicate="qor.latency_best_cycles", stage="schedule", unit="cycle", mask=True,
        )],
    )

    valid_output = tmp_path / "valid-csynth-label-dataset"
    export_dataset(bundle, snapshot.id, valid_output, dataset)
    valid_rows = [json.loads(line) for line in
                  (valid_output / "labels.jsonl").read_text(encoding="utf-8").splitlines()]
    assert valid_rows[0]["observation_id"] == observation.id

    retained_path.unlink()
    output = tmp_path / "lost-report-dataset"
    with pytest.raises(ValueError, match="report integrity failed"):
        export_dataset(bundle, snapshot.id, output, dataset)
    assert not output.exists()


def test_trusted_label_stage_and_report_must_match_the_producer_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    manifest = minimal_manifest(
        "test.label_stage_policy", "label stage policy", "dut", "kernel.cpp"
    )
    manifest.toolchains = [ToolchainContext(
        id="test.fixture_tool", vendor="test", name="fixture_tool", version="1",
        environment_hash="e" * 64,
    )]
    manifest.stage_commands = {"csynth": ["tool", "--report"]}
    manifest.stage_toolchains = {"csynth": "test.fixture_tool"}
    manifest.stage_outputs = {"csynth": [ToolOutputSpec(
        path="reports/csynth.xml", kind="amd.vitis.csynth_xml",
        role="verification_report",
    )]}
    bundle = GraphBundle.create(tmp_path, manifest)
    snapshot = bundle.snapshot()
    graph = CanonicalGraph(snapshot.id)
    kernel = graph.add_entity(Entity("hls.kernel", "dut", snapshot.id, stage="ast"))
    bundle.store.save_graph(graph)

    run = ToolRun(
        snapshot_id=snapshot.id, stage="csynth", backend="runner.local",
        request_hash="d" * 64, toolchain_id="test.fixture_tool",
        status=RunStatus.SUCCEEDED,
        command=list(manifest.stage_commands["csynth"]), working_directory=".",
        environment_hash="e" * 64,
        input_artifact_ids=[
            item.id for item in bundle.store.artifacts(snapshot.id)
            if item.producer_run_id is None
        ],
        exit_code=0,
        metadata={
            "authority": "tool_observation", "fresh_execution": True,
            "fresh_tool_truth": True, "tool_truth": True,
        },
    )
    report_source = tmp_path / "csynth.xml"
    write_csynth_xml(report_source, latency=42)
    report, _retained, _created = bundle.prepare_managed_artifact(
        report_source, kind="amd.vitis.csynth_xml", role="verification_report",
        producer_run_id=run.id,
        metadata={"declared_output_path": "reports/csynth.xml"},
    )
    run.output_artifact_ids = [report.id]
    disguised_post_route = Observation(
        snapshot_id=snapshot.id, subject_id=kernel.id,
        predicate="timing.wns_ns", value=0.25, unit="ns", stage="post_route",
        authority=AuthorityClass.PHYSICAL_MEASUREMENT,
        run_id=run.id, artifact_id=report.id,
    )

    with pytest.raises(StoreError, match="incompatible with producer stage 'csynth'"):
        commit_attested(bundle,
            run=run, artifacts=[report], observations=[disguised_post_route],
        )
    assert bundle.store.runs(snapshot.id) == []

    valid_csynth = parsed_report_observation(
        bundle, report, predicate="qor.latency_best_cycles", value=42,
        subject_id=kernel.id, run_id=run.id,
    )
    commit_attested(bundle,
        run=run, artifacts=[report], observations=[valid_csynth],
    )

    # Simulate a pre-policy/foreign ledger row to prove the export boundary
    # independently rejects the same stage laundering attempt.
    monkeypatch.setattr(
        type(bundle.store), "observations",
        lambda _self, _snapshot_id, **_kwargs: [disguised_post_route],
    )
    dataset = DatasetManifest(
        dataset_id="dataset.disguised_post_route", feature_schema_version=FEATURE_SCHEMA_VERSION,
        snapshot_ids=[snapshot.id], labels=[LabelSpec(
            label_id="post_route_wns", snapshot_id=snapshot.id,
            observation_id=disguised_post_route.id,
            predicate=disguised_post_route.predicate,
            stage=disguised_post_route.stage, unit=disguised_post_route.unit,
        )],
    )
    with pytest.raises(ValueError, match="incompatible with producer stage 'csynth'"):
        export_dataset(bundle, snapshot.id, tmp_path / "disguised-label", dataset)


def test_known_report_kind_cannot_be_retyped_by_plugin_extension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    manifest = minimal_manifest(
        "test.known_kind_stage_policy", "known kind stage policy", "dut", "kernel.cpp"
    )
    manifest.toolchains = [ToolchainContext(
        id="test.fixture_tool", vendor="test", name="fixture_tool", version="1",
        environment_hash="e" * 64,
    )]
    manifest.stage_commands = {"post_route": ["tool", "--post-route"]}
    manifest.stage_toolchains = {"post_route": "test.fixture_tool"}
    manifest.stage_outputs = {"post_route": [
        ToolOutputSpec(
            path="reports/disguised_csynth.xml", kind="amd.vitis.csynth_xml",
            role="verification_report",
        ),
        ToolOutputSpec(
            path="reports/post_route_timing.rpt",
            kind="amd.vivado.post_route_timing", role="verification_report",
        ),
    ]}
    bundle = GraphBundle.create(tmp_path, manifest)
    snapshot = bundle.snapshot()
    graph = CanonicalGraph(snapshot.id)
    kernel = graph.add_entity(Entity("hls.kernel", "dut", snapshot.id, stage="ast"))
    bundle.store.save_graph(graph)

    run = ToolRun(
        snapshot_id=snapshot.id, stage="post_route", backend="runner.local",
        request_hash="f" * 64, toolchain_id="test.fixture_tool",
        status=RunStatus.SUCCEEDED,
        command=list(manifest.stage_commands["post_route"]), working_directory=".",
        environment_hash="e" * 64,
        input_artifact_ids=[
            item.id for item in bundle.store.artifacts(snapshot.id)
            if item.producer_run_id is None
        ],
        exit_code=0,
        metadata={
            "authority": "tool_observation", "fresh_execution": True,
            "fresh_tool_truth": True, "tool_truth": True,
        },
    )
    disguised_source = tmp_path / "disguised_csynth.xml"
    disguised_source.write_text("<csynth/>\n", encoding="utf-8")
    disguised_report, _retained, _created = bundle.prepare_managed_artifact(
        disguised_source, kind="amd.vitis.csynth_xml", role="verification_report",
        producer_run_id=run.id,
        metadata={
            "declared_output_path": "reports/disguised_csynth.xml",
            "hlsgraph_evidence": {
                "policy_version": TOOL_EVIDENCE_POLICY_VERSION,
                "observation_stage": "post_route",
                "run_stage": "post_route",
                "semantics": "tool_report",
            }
        },
    )
    valid_source = tmp_path / "post_route_timing.rpt"
    write_vivado_timing(valid_source, wns=0.25)
    valid_report, _retained, _created = bundle.prepare_managed_artifact(
        valid_source, kind="amd.vivado.post_route_timing", role="verification_report",
        producer_run_id=run.id,
        metadata={
            "declared_output_path": "reports/post_route_timing.rpt",
            "scope": {
                "kind": "kernel", "top": "dut", "instance": "dut",
                "clock": "default",
            },
        },
    )
    run.output_artifact_ids = [disguised_report.id, valid_report.id]
    disguised_observation = Observation(
        snapshot_id=snapshot.id, subject_id=kernel.id,
        predicate="timing.wns_ns", value=0.25, unit="ns", stage="post_route",
        authority=AuthorityClass.TOOL_OBSERVATION,
        run_id=run.id, artifact_id=disguised_report.id,
    )

    with pytest.raises(StoreError, match="typed for another evidence stage"):
        commit_attested(bundle,
            run=run, artifacts=[disguised_report, valid_report],
            observations=[disguised_observation],
        )
    assert bundle.store.runs(snapshot.id) == []

    valid_observation = parsed_report_observation(
        bundle, valid_report, predicate="timing.wns_ns", value=0.25,
        subject_id=kernel.id, run_id=run.id,
    )
    commit_attested(bundle,
        run=run, artifacts=[disguised_report, valid_report],
        observations=[valid_observation],
    )

    # Simulate a pre-policy/foreign ledger: export must independently reject
    # the known csynth kind even though it carries a superficially complete
    # post-route plugin contract.
    monkeypatch.setattr(
        type(bundle.store), "observations",
        lambda _self, _snapshot_id, **_kwargs: [disguised_observation],
    )
    with pytest.raises(ValueError, match="typed for another evidence stage"):
        export_dataset(
            bundle, snapshot.id, tmp_path / "known-kind-unlabelled-observation"
        )
    dataset = DatasetManifest(
        dataset_id="dataset.known_kind_stage_policy", feature_schema_version=FEATURE_SCHEMA_VERSION,
        snapshot_ids=[snapshot.id], labels=[LabelSpec(
            label_id="post_route_wns", snapshot_id=snapshot.id,
            observation_id=disguised_observation.id,
            predicate=disguised_observation.predicate,
            stage=disguised_observation.stage, unit=disguised_observation.unit,
        )],
    )
    with pytest.raises(ValueError, match="typed for another evidence stage"):
        export_dataset(bundle, snapshot.id, tmp_path / "known-kind-disguised-label", dataset)


def test_index_run_cannot_claim_tool_truth_at_ledger_or_export_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    bundle = GraphBundle.create(
        tmp_path,
        minimal_manifest("test.index_tool_truth", "index tool truth", "dut", "kernel.cpp"),
    )
    snapshot = bundle.snapshot()
    graph = CanonicalGraph(snapshot.id)
    graph.add_entity(Entity("hls.kernel", "dut", snapshot.id, stage="ast"))
    bundle.store.save_graph(graph)
    invalid = ToolRun(
        snapshot_id=snapshot.id, stage="index", backend="runner.fake",
        request_hash="a" * 64, status=RunStatus.SUCCEEDED, exit_code=0,
        metadata={
            "authority": "tool_observation", "fresh_execution": True,
            "fresh_tool_truth": True, "tool_truth": True,
        },
    )

    with pytest.raises(StoreError, match="index runs cannot claim external tool truth"):
        bundle.store.add_run(invalid)
    assert bundle.store.runs(snapshot.id) == []

    # A foreign/pre-policy row must not regain the claim through ML run
    # projection merely because its metadata booleans say tool_truth.
    monkeypatch.setattr(
        type(bundle.store), "runs", lambda _self, _snapshot_id: [invalid],
    )
    with pytest.raises(ValueError, match="index runs cannot claim external tool truth"):
        export_dataset(bundle, snapshot.id, tmp_path / "invalid-index-truth-dataset")


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("stage", "post_route"),
        ("toolchain_id", "test.other_tool"),
        ("environment_hash", "f" * 64),
        ("command", ["different-tool", "--report"]),
        ("working_directory", "elsewhere"),
    ],
    ids=["stage", "toolchain", "environment", "command", "working-directory"],
)
def test_ml_export_replays_tool_run_immutable_manifest_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    invalid_value: object,
) -> None:
    (tmp_path / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    manifest = minimal_manifest(
        "test.export_run_identity", "export run identity", "dut", "kernel.cpp"
    )
    manifest.toolchains = [ToolchainContext(
        id="test.fixture_tool", vendor="test", name="fixture_tool", version="1",
        environment_hash="e" * 64,
    )]
    manifest.stage_commands = {"csynth": ["tool", "--report"]}
    manifest.stage_toolchains = {"csynth": "test.fixture_tool"}
    manifest.stage_outputs = {"csynth": [ToolOutputSpec(
        path="reports/csynth.xml", kind="amd.vitis.csynth_xml",
        role="verification_report",
    )]}
    bundle = GraphBundle.create(tmp_path, manifest)
    snapshot = bundle.snapshot()
    graph = CanonicalGraph(snapshot.id)
    kernel = graph.add_entity(Entity("hls.kernel", "dut", snapshot.id, stage="ast"))
    bundle.store.save_graph(graph)
    run = ToolRun(
        snapshot_id=snapshot.id, stage="csynth", backend="runner.local",
        request_hash="b" * 64, toolchain_id="test.fixture_tool",
        status=RunStatus.SUCCEEDED,
        command=list(manifest.stage_commands["csynth"]), working_directory=".",
        environment_hash="e" * 64,
        input_artifact_ids=[
            item.id for item in bundle.store.artifacts(snapshot.id)
            if item.producer_run_id is None
        ],
        exit_code=0,
        metadata={
            "authority": "tool_observation", "fresh_execution": True,
            "fresh_tool_truth": True, "tool_truth": True,
        },
    )
    report_source = tmp_path / "csynth.xml"
    write_csynth_xml(report_source, latency=42)
    report, _retained, _created = bundle.prepare_managed_artifact(
        report_source, kind="amd.vitis.csynth_xml", role="verification_report",
        producer_run_id=run.id,
        metadata={"declared_output_path": "reports/csynth.xml"},
    )
    run.output_artifact_ids = [report.id]
    observation = parsed_report_observation(
        bundle, report, predicate="qor.latency_best_cycles", value=42,
        subject_id=kernel.id, run_id=run.id,
    )
    commit_attested(bundle,
        run=run, artifacts=[report], observations=[observation],
    )

    foreign_run = copy.deepcopy(run)
    setattr(foreign_run, field, invalid_value)
    monkeypatch.setattr(
        type(bundle.store), "runs", lambda _self, _snapshot_id: [foreign_run],
    )
    with pytest.raises(ValueError, match="immutable snapshot manifest"):
        export_dataset(
            bundle, snapshot.id,
            tmp_path / f"invalid-run-identity-{field.replace('_', '-')}",
        )


def test_ml_export_rejects_real_authority_observation_from_nontruth_run(
    tmp_path: Path,
) -> None:
    """The ledger may retain failed evidence, but public ML truth downgrades it explicitly."""

    (tmp_path / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    bundle = GraphBundle.create(
        tmp_path,
        minimal_manifest(
            "test.nontruth_observation_export", "nontruth observation export",
            "dut", "kernel.cpp",
        ),
    )
    snapshot = bundle.snapshot()
    graph = CanonicalGraph(snapshot.id)
    kernel = graph.add_entity(Entity("hls.kernel", "dut", snapshot.id, stage="ast"))
    bundle.store.save_graph(graph)
    run = ToolRun(
        snapshot_id=snapshot.id, stage="csynth", backend="runner.fake",
        request_hash="c" * 64, status=RunStatus.FAILED,
        failure_class=FailureClass.INFRASTRUCTURE,
        metadata={
            "authority": "tool_observation", "fresh_execution": False,
            "fresh_tool_truth": False, "tool_truth": False,
        },
    )
    report_source = tmp_path / "partial_csynth.rpt"
    report_source.write_text("partial latency=42\n", encoding="utf-8")
    report, _retained, _created = bundle.prepare_managed_artifact(
        report_source, kind="amd.vitis.csynth_report", role="partial_report",
        producer_run_id=run.id,
    )
    run.output_artifact_ids = [report.id]
    observation = Observation(
        snapshot_id=snapshot.id, subject_id=kernel.id,
        predicate="qor.latency_cycles", value=42, unit="cycle", stage="schedule",
        authority=AuthorityClass.TOOL_OBSERVATION,
        run_id=run.id, artifact_id=report.id,
    )
    bundle.store.commit_run_result(
        run=run, artifacts=[report], observations=[observation],
    )
    assert bundle.store.observations(snapshot.id) == [observation]

    output = tmp_path / "nontruth-observation-dataset"
    exported = export_dataset(bundle, snapshot.id, output)
    assert exported["row_counts"]["observations"] == 0
    assert exported["row_counts"]["nontruth_observations"] == 1
    downgraded = json.loads(
        (output / "nontruth_observations.jsonl").read_text(encoding="utf-8")
    )
    assert downgraded["id"] == observation.id
    assert downgraded["claimed_authority"] == "tool_observation"
    assert downgraded["tool_truth"] is False
    assert downgraded["nontruth_reason"] == "producer_does_not_claim_tool_truth"
    feature_spec = json.loads(
        (output / "feature_spec.json").read_text(encoding="utf-8")
    )
    assert feature_spec["truth_tables"] == ["observations", "labels"]
    assert feature_spec["nontruth_tables"] == ["nontruth_observations"]


def test_ml_labels_are_keyed_by_snapshot_and_preserve_masked_truth(
    tmp_path: Path,
) -> None:
    (tmp_path / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    bundle = GraphBundle.create(
        tmp_path, minimal_manifest("test.multi_label", "multi label", "dut", "kernel.cpp")
    )
    first, _ = _snapshot_with_graph(bundle, "void dut() {}\n")
    second, _ = _snapshot_with_graph(
        bundle, "void dut() { int x = 1; (void)x; }\n"
    )
    labels = [
        LabelSpec(
            label_id="latency", snapshot_id=first, observation_id=None,
            predicate="qor.latency_cycles", stage="csynth", unit="cycle",
            mask=False, missing_reason="tool_timeout", censored=True,
        ),
        LabelSpec(
            label_id="latency", snapshot_id=second, observation_id=None,
            predicate="qor.latency_cycles", stage="csynth", unit="cycle",
            mask=False, missing_reason="unsupported_design", unbounded=True,
        ),
    ]
    dataset = DatasetManifest(
        dataset_id="dataset.multi_label", feature_schema_version=FEATURE_SCHEMA_VERSION,
        snapshot_ids=[first, second], labels=labels,
    )
    output = tmp_path / "multi-label"
    export_dataset(bundle, first, output, dataset)
    rows = [json.loads(line) for line in
            (output / "labels.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [(row["snapshot_id"], row["label_id"]) for row in rows] == sorted([
        (first, "latency"), (second, "latency"),
    ])
    assert {row["missing_reason"] for row in rows} == {
        "tool_timeout", "unsupported_design",
    }

    duplicate = DatasetManifest(
        dataset_id="dataset.duplicate_label", feature_schema_version=FEATURE_SCHEMA_VERSION,
        snapshot_ids=[first, second], labels=[labels[0], labels[0]],
    )
    with pytest.raises(ValueError, match="unique by snapshot_id and label_id"):
        export_dataset(bundle, first, tmp_path / "duplicate-label", duplicate)


def test_ml_present_label_rejects_runless_and_synthetic_tool_observations(
    tmp_path: Path,
) -> None:
    (tmp_path / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    bundle = GraphBundle.create(
        tmp_path, minimal_manifest("test.label_trust", "label trust", "dut", "kernel.cpp")
    )
    snapshot = bundle.snapshot()
    graph = CanonicalGraph(snapshot.id)
    kernel = graph.add_entity(Entity("hls.kernel", "dut", snapshot.id))
    bundle.store.save_graph(graph)

    runless = Observation(
        snapshot_id=snapshot.id, subject_id=kernel.id,
        predicate="qor.runless_latency", value=10, unit="cycle", stage="csynth",
        authority=AuthorityClass.TOOL_OBSERVATION,
    )
    bundle.store.add_observations([runless])
    with pytest.raises(ValueError, match="lacks a producer tool run"):
        export_dataset(bundle, snapshot.id, tmp_path / "runless-unlabelled-observation")
    runless_dataset = DatasetManifest(
        dataset_id="dataset.runless_label", feature_schema_version=FEATURE_SCHEMA_VERSION,
        snapshot_ids=[snapshot.id], labels=[LabelSpec(
            label_id="runless_latency", snapshot_id=snapshot.id,
            observation_id=runless.id, predicate=runless.predicate,
            stage=runless.stage, unit=runless.unit,
        )],
    )
    with pytest.raises(ValueError, match="lacks a producer tool run"):
        export_dataset(bundle, snapshot.id, tmp_path / "runless-label", runless_dataset)

    synthetic_root = tmp_path / "synthetic-project"
    synthetic_root.mkdir()
    (synthetic_root / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    bundle = GraphBundle.create(
        synthetic_root,
        minimal_manifest(
            "test.synthetic_label_trust", "synthetic label trust", "dut", "kernel.cpp"
        ),
    )
    snapshot = bundle.snapshot()
    graph = CanonicalGraph(snapshot.id)
    kernel = graph.add_entity(Entity("hls.kernel", "dut", snapshot.id))
    bundle.store.save_graph(graph)
    synthetic_run = ToolRun(
        snapshot_id=snapshot.id, stage="csynth", backend="runner.fake",
        request_hash="f" * 64, status=RunStatus.SUCCEEDED,
        command=["fixture-tool"], exit_code=0,
        metadata={
            "authority": "synthetic", "fresh_execution": False,
            "fresh_tool_truth": False, "tool_truth": False,
        },
    )
    report_source = tmp_path / "synthetic.rpt"
    report_source.write_text("latency=9\n", encoding="utf-8")
    report, _retained, _created = bundle.prepare_managed_artifact(
        report_source, kind="test.synthetic_report", role="tool_output",
        producer_run_id=synthetic_run.id,
    )
    synthetic_run.output_artifact_ids = [report.id]
    synthetic = Observation(
        snapshot_id=snapshot.id, subject_id=kernel.id,
        predicate="qor.synthetic_latency", value=9, unit="cycle", stage="csynth",
        authority=AuthorityClass.TOOL_OBSERVATION,
        run_id=synthetic_run.id, artifact_id=report.id,
    )
    bundle.store.commit_run_result(
        run=synthetic_run, artifacts=[report], observations=[synthetic],
    )
    synthetic_dataset = DatasetManifest(
        dataset_id="dataset.synthetic_label", feature_schema_version=FEATURE_SCHEMA_VERSION,
        snapshot_ids=[snapshot.id], labels=[LabelSpec(
            label_id="synthetic_latency", snapshot_id=snapshot.id,
            observation_id=synthetic.id, predicate=synthetic.predicate,
            stage=synthetic.stage, unit=synthetic.unit,
        )],
    )
    with pytest.raises(ValueError, match="successful fresh real-tool run"):
        export_dataset(
            bundle, snapshot.id, tmp_path / "synthetic-label", synthetic_dataset
        )


def test_label_spec_rejects_ambiguous_mask_and_malformed_identifiers() -> None:
    with pytest.raises(ValueError, match="mask must be a boolean"):
        LabelSpec(
            label_id="latency", snapshot_id="snapshot.valid",
            observation_id="observation.valid", predicate="qor.latency",
            stage="csynth", mask=1,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="masked label must declare"):
        LabelSpec(
            label_id="latency", snapshot_id="snapshot.valid",
            observation_id=None, predicate="qor.latency", stage="csynth",
            mask=False,
        )
    with pytest.raises(ValueError, match="snapshot_id"):
        LabelSpec(
            label_id="latency", snapshot_id="invalid snapshot",
            observation_id="observation.valid", predicate="qor.latency",
            stage="csynth",
        )


def test_release_metadata_and_sbom_are_consistent() -> None:
    root = Path(__file__).parents[1]
    sbom = json.loads((root / "sbom.spdx.json").read_text(encoding="utf-8"))
    assert sbom["name"] == "hlsgraph-0.3.0"
    packages = {item["name"]: item for item in sbom["packages"]}
    assert {"hlsgraph", "cytoscape", "elkjs", "tomli"}.issubset(packages)
    from tools.audit_release import (
        _audit_sbom,
        _audit_source_tree,
        _package_verification_code,
    )

    assert _audit_source_tree(root) == []
    assert _audit_sbom((root / "sbom.spdx.json").read_bytes(), root) == []
    files = {item["SPDXID"]: item for item in sbom["files"]}
    for package_name in ("cytoscape", "elkjs"):
        package = packages[package_name]
        analyzed = []
        for spdx_id in package["hasFiles"]:
            file_name = files[spdx_id]["fileName"]
            path = root / file_name.removeprefix("./")
            analyzed.append((file_name, path.read_bytes()))
        sha1_values = sorted(hashlib.sha1(data).hexdigest() for _, data in analyzed)
        independently_recomputed = hashlib.sha1(
            "".join(sha1_values).encode("ascii")
        ).hexdigest()
        recorded = package["packageVerificationCode"]["packageVerificationCodeValue"]
        assert recorded == independently_recomputed
        assert recorded == _package_verification_code(analyzed)
    assert (root / "CITATION.cff").is_file()
    assert (root / "docs" / "references.md").is_file()


def test_ml_manifest_projection_rejects_private_identity_and_resource_values() -> None:
    private_path = "C" + ":" + "/Us" + "ers/alice/secret"
    sentinel = "PRIVATE MANIFEST SENTINEL " + private_path
    projected_toolchain = _public_toolchain({
        "id": sentinel,
        "vendor": sentinel,
        "name": sentinel,
        "version": "2024.2",
        "build": sentinel,
        "environment_hash": sentinel,
        "metadata": {"private": sentinel},
    })
    assert projected_toolchain["version"] == "2024.2"
    assert projected_toolchain["id"] is None
    assert projected_toolchain["vendor"] is None
    assert projected_toolchain["name"] is None
    assert projected_toolchain["build"] is None
    assert projected_toolchain["environment_hash"] == hashlib.sha256(
        json.dumps(sentinel, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert _public_toolchain({
        "id": "amd.vitis.2024_2", "vendor": "amd", "name": "vitis_hls",
        "version": "2024.2", "environment_hash": "A" * 64,
    })["environment_hash"] == "a" * 64

    projected_target = _public_target({
        "vendor": "amd",
        "part": "xck26-test",
        "package": None,
        "speed_grade": "-2",
        "board": sentinel,
        "platform": sentinel,
        "platform_hash": sentinel,
        "capacities": {
            "LUT": 100,
            "BRAM_18K": 2.5,
            sentinel: 1,
            "negative": -1,
            "boolean": True,
            "infinite": float("inf"),
            "prose": sentinel,
        },
        "reserved_resources": {
            "LUT": 10,
            sentinel: 1,
            "not-a-number": sentinel,
        },
        "clocks": [{
            "name": sentinel,
            "period_ns": sentinel,
            "uncertainty_ns": -1,
            "source": sentinel,
        }],
        "memory_topology": [{"kind": sentinel, "banks": 4, sentinel: "secret"}],
        "metadata": {"private": sentinel},
    }, [])
    assert projected_target["vendor"] == "amd"
    assert projected_target["part"] == "xck26-test"
    assert projected_target["speed_grade"] == "-2"
    assert projected_target["board"] is None
    assert projected_target["platform"] is None
    assert projected_target["capacities"] == {"BRAM_18K": 2.5, "LUT": 100}
    assert projected_target["reserved_resources"] == {"LUT": 10}
    assert projected_target["clocks"][0]["period_ns"] is None
    assert projected_target["clocks"][0]["uncertainty_ns"] is None
    assert len(projected_target["platform_hash"]) == 64
    assert sentinel not in json.dumps({
        "toolchain": projected_toolchain,
        "target": projected_target,
    }, sort_keys=True)


def test_release_scanner_detects_credentials_and_private_user_paths() -> None:
    from tools.audit_release import _scan

    assigned_token = b"access_" + b"token='" + (b"x" * 24) + b"'"
    windows_home = b"C:" + b"\\Users\\private-user\\project"
    posix_home = b"/" + b"home/private-user/project"
    assert _scan("token.txt", assigned_token)
    assert _scan("windows.txt", windows_home)
    assert _scan("posix.txt", posix_home)


def test_release_audit_accepts_crlf_metadata_and_only_standard_sdist_egg_info() -> None:
    from tools.audit_release import _audit_wheel_metadata, _forbidden

    metadata = (
        b"Metadata-Version: 2.4\r\n"
        b"Name: hlsgraph\r\n"
        b"Version: 0.3.0\r\n"
        b"Project-URL: Source, https://github.com/liumh05/hlsgraph\r\n\r\n"
    )
    assert _audit_wheel_metadata(metadata) == []
    standard = "src/hlsgraph.egg-info/SOURCES.txt"
    assert _forbidden(standard, sdist=True) is None
    assert _forbidden(standard) == ".egg-info/"
    assert _forbidden("src/hlsgraph.egg-info/private.txt", sdist=True) == ".egg-info/"
