from __future__ import annotations

import hashlib
import json
import re

import pytest

from hlsgraph import FEATURE_SCHEMA_VERSION
from hlsgraph.bundle import GraphBundle
from hlsgraph.export import export_dataset
from hlsgraph.graph import CanonicalGraph
from hlsgraph.manifest import minimal_manifest
from hlsgraph.model import (
    AuthorityClass,
    DatasetManifest,
    Entity,
    LabelSpec,
    Observation,
    PredictionEnvelope,
    Relation,
    RunStatus,
    ToolRun,
    ToolchainContext,
)
from hlsgraph.render import render, to_render_data
from hlsgraph.render.html import to_html


def _dataflow(snapshot_id: str):
    graph = CanonicalGraph(snapshot_id, metadata={"top": "dut"})
    source = Entity(
        "hls.buffer", "input_fifo", snapshot_id, qualified_name="dut::input_fifo",
        stage="mlir", attrs={"depth": 8},
    )
    compute = Entity(
        "hls.process", "multiply", snapshot_id, qualified_name="dut::multiply",
        stage="mlir", attrs={"operation": "handshake.mul"},
    )
    sink = Entity(
        "hls.buffer", "output_fifo", snapshot_id, qualified_name="dut::output_fifo",
        stage="mlir", attrs={"depth": 16},
    )
    for entity in (source, compute, sink):
        graph.add_entity(entity)
    graph.add_relation(Relation(
        source.id, compute.id, "hls.streams_to", snapshot_id, stage="mlir",
        authority=AuthorityClass.COMPILER_DECISION,
        attrs={"fifo_depth": 8, "elem_type": "i32", "projection": "handshake_semantics"},
    ))
    graph.add_relation(Relation(
        compute.id, sink.id, "hls.streams_to", snapshot_id, stage="mlir",
        authority=AuthorityClass.COMPILER_DECISION,
        attrs={"fifo_depth": 16, "elem_type": "i32", "projection": "handshake_semantics"},
    ))
    observations = [
        Observation(
            snapshot_id=snapshot_id, subject_id=compute.id,
            predicate="qor.achieved_ii", value=2, unit="cycle", stage="schedule",
            authority=AuthorityClass.TOOL_OBSERVATION,
        ),
        Observation(
            snapshot_id=snapshot_id, subject_id=compute.id,
            predicate="qor.target_ii", value=1, unit="cycle", stage="schedule",
            authority=AuthorityClass.DECLARED_CONSTRAINT,
        ),
        Observation(
            snapshot_id=snapshot_id, subject_id=compute.id,
            predicate="schedule.operation_latency", value=3, unit="cycle", stage="schedule",
            authority=AuthorityClass.COMPILER_DECISION,
        ),
    ]
    return graph, observations, source, compute, sink


def test_render_projection_is_dataflow_specific_and_marks_fifo_and_bottleneck():
    graph, observations, _source, compute, _sink = _dataflow("snapshot.render")
    data = to_render_data(graph, observations, [])

    assert data["meta"]["view"] == "dataflow"
    assert data["meta"]["graph_hash"] == graph.graph_hash
    assert len(data["nodes"]) == 3 and len(data["edges"]) == 2
    assert {edge["fifo_depth"] for edge in data["edges"]} == {8, 16}
    assert all(edge["type"] == "STREAMS_TO" for edge in data["edges"])

    bottlenecks = [node for node in data["nodes"] if node["is_bottleneck"]]
    assert len(bottlenecks) == 1
    assert bottlenecks[0]["id"] == compute.id
    assert bottlenecks[0]["metrics"]["achieved_II"] == 2
    assert bottlenecks[0]["metrics"]["target_II"] == 1
    assert "exceeds target" in bottlenecks[0]["bottleneck_cause"]


def test_html_is_offline_layered_uniform_interactive_and_theme_aware():
    graph, observations, *_ = _dataflow("snapshot.render_html")
    html = to_html(to_render_data(graph, observations, []))
    lower = html.lower()

    assert html.startswith("<!doctype html>")
    assert "<script src=" not in lower
    assert "http" not in lower and "cdn." not in lower
    assert "new Function(atob(" in html
    assert "'elk.algorithm':'layered'" in html
    assert "'elk.direction':'RIGHT'" in html
    assert "'elk.edgeRouting':'ORTHOGONAL'" in html
    assert "curve-style':'taxi'" in html
    assert "NODE_W=132,NODE_H=48" in html
    assert "Math.log2" in html and "fifo_depth" in html
    assert "?css('--bottleneck')" in html
    assert "prefers-color-scheme:dark" in lower
    assert ':root[data-theme="dark"]' in html
    assert ':root[data-theme="light"]' in html

    for control in ("search", "category", "stage", "authority", "theme", "reset"):
        assert f'id="{control}"' in html
    assert "closedNeighborhood" in html
    assert "panelFor(n)" in html
    assert "mouseover','node'" in html
    assert "mouseover','edge'" in html
    assert "toggleClass('dim'" in html

    # The node face is name plus at most achieved II; other metrics stay in the panel.
    node_label = re.search(r"label:e=>\{(.+?)\},'font-size'", html)
    assert node_label and "achieved_II" in node_label.group(1)
    assert all(metric not in node_label.group(1)
               for metric in ("latency", "lut", "dsp", "bitwidth", "replication"))


def test_all_render_formats_are_deterministic_and_do_not_mutate_canonical_graph():
    graph, observations, *_ = _dataflow("snapshot.formats")
    before = graph.to_dict()
    for format in ("json", "mermaid", "dot", "svg", "html"):
        first = render(graph, format=format, observations=observations)
        second = render(graph, format=format, observations=observations)
        assert first == second
        assert first.strip()
    assert graph.to_dict() == before
    assert render(graph, format="mermaid", observations=observations).startswith("flowchart LR")
    with pytest.raises(ValueError, match="format must be"):
        render(graph, format="png", observations=observations)


def test_ml_export_separates_features_truth_labels_predictions_and_private_source(tmp_path):
    secret = "PRIVATE_EXPORT_SENTINEL_814c"
    (tmp_path / "kernel.cpp").write_text(
        f"// {secret}\nvoid dut() {{}}\n", encoding="utf-8"
    )
    project_manifest = minimal_manifest(
        "test.ml_export", "ML export", "dut", "kernel.cpp",
        part="xck26-test", clock_ns=5.0,
    )
    project_manifest.toolchains = [ToolchainContext(
        id="test.fixture_tool", vendor="test", name="fixture_tool", version="1",
        environment_hash="e" * 64,
    )]
    project_manifest.stage_commands = {
        "csynth": ["fixture-tool", "--csynth"],
    }
    project_manifest.stage_toolchains = {"csynth": "test.fixture_tool"}
    bundle = GraphBundle.create(tmp_path, project_manifest)
    snapshot = bundle.snapshot()
    kernel = Entity(
        "hls.kernel", "dut", snapshot.id, qualified_name="dut", stage="ast",
        attrs={"safe_feature": 7, "qor.leaky": 999, "label_target": 888,
               "prediction_hint": 777},
    )
    graph = CanonicalGraph(snapshot.id, metadata={"top": "dut"})
    graph.add_entity(kernel)
    graph.add_relation(Relation(
        kernel.id, kernel.id, "test.self", snapshot.id,
        attrs={"safe_edge_feature": 3, "timing.slack": -1,
               "achieved_ii": 2, "power_w": 9.0},
    ))
    bundle.store.save_graph(graph)

    run = ToolRun(
        snapshot_id=snapshot.id, stage="csynth", backend="runner.local",
        request_hash="a" * 64, toolchain_id="test.fixture_tool",
        status=RunStatus.SUCCEEDED,
        command=list(project_manifest.stage_commands["csynth"]),
        working_directory=".", environment_hash="e" * 64,
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
    report_source = tmp_path / "latency.rpt"
    report_source.write_text("latency_cycles=42\n", encoding="utf-8")
    report, _retained_path, _created = bundle.prepare_managed_artifact(
        report_source, kind="amd.vitis.csynth_report", role="verification_report",
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
    prediction = PredictionEnvelope(
        snapshot_id=snapshot.id, subject_id=kernel.id,
        predicate="prediction.latency_cycles", value=39.5, unit="cycle",
        model_id="test.model", model_version="1.0", input_schema_version="0.1.0",
        uncertainty={"stddev": 2.0}, applicability={"part": "xck26-test"},
    )
    bundle.store.add_prediction(prediction)
    dataset = DatasetManifest(
        dataset_id="test.dataset", feature_schema_version=FEATURE_SCHEMA_VERSION,
        snapshot_ids=[snapshot.id],
        feature_attribute_allowlist=["safe_feature", "safe_edge_feature"],
        labels=[LabelSpec(
            label_id="latency", snapshot_id=snapshot.id,
            observation_id=observation.id,
            predicate="qor.latency_cycles", stage="schedule", unit="cycle", mask=True,
        )],
        splits={snapshot.id: "test"},
        kernel_families={snapshot.id: "synthetic.dataflow"},
        dedup_groups={snapshot.id: "family.hash"},
        licenses={snapshot.id: "Apache-2.0"},
    )

    output = tmp_path / "dataset"
    manifest = export_dataset(bundle, snapshot.id, output, dataset)
    serialized = "\n".join(path.read_text(encoding="utf-8")
                             for path in output.iterdir() if path.is_file())
    assert secret not in serialized
    assert manifest["private_source_embedded"] is False
    assert manifest["target_profile"]["part"] == "xck26-test"
    assert set(manifest["file_integrity"]) == set(manifest["files"])
    for filename, integrity in manifest["file_integrity"].items():
        payload = (output / filename).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == integrity["sha256"]
        assert len(payload) == integrity["size"]

    nodes = [json.loads(line) for line in (output / "nodes.jsonl").read_text().splitlines()]
    assert nodes[0]["features"] == {"safe_feature": 7}
    assert "value" not in nodes[0]
    edges = [json.loads(line) for line in (output / "edges.jsonl").read_text().splitlines()]
    assert edges[0]["features"] == {"safe_edge_feature": 3}
    observations = [json.loads(line) for line in
                    (output / "observations.jsonl").read_text().splitlines()]
    labels = [json.loads(line) for line in (output / "labels.jsonl").read_text().splitlines()]
    predictions = [json.loads(line) for line in
                   (output / "predictions.jsonl").read_text().splitlines()]
    assert observations[0]["id"] == observation.id and observations[0]["value"] == 42
    assert labels == [{
        "censored": False,
        "label_id": "latency",
        "mask": True,
        "missing_reason": None,
        "observation_id": observation.id,
        "predicate": "qor.latency_cycles",
        "snapshot_id": snapshot.id,
        "stage": "schedule",
        "unbounded": False,
        "unit": "cycle",
    }]
    assert predictions[0]["id"] == prediction.id
    assert predictions[0]["value"] == 39.5
    assert predictions[0]["predicate"] == "prediction.latency_cycles"

    feature_spec = json.loads((output / "feature_spec.json").read_text(encoding="utf-8"))
    assert feature_spec["truth_tables"] == ["observations", "labels"]
    assert feature_spec["prediction_table"] == "predictions"
    assert feature_spec["label_contract"] == (
        "present labels reference same-snapshot complete observations from successful "
        "fresh real-tool runs and stage-compatible typed retained reports; values are "
        "not duplicated"
    )
    assert feature_spec["tool_evidence_policy_version"] == "hlsgraph.tool-evidence.v0.1"
    artifact_rows = (output / "artifacts.jsonl").read_text(encoding="utf-8")
    assert '"source_text_embedded":false' in artifact_rows
    assert secret not in artifact_rows

    with pytest.raises(PermissionError, match="never embed source text"):
        export_dataset(bundle, snapshot.id, tmp_path / "unsafe", include_source=True)
    with pytest.raises(FileExistsError, match="must not already exist"):
        export_dataset(bundle, snapshot.id, output, dataset)


def test_label_contract_rejects_unavailable_truth_observation(tmp_path):
    (tmp_path / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    bundle = GraphBundle.create(
        tmp_path, minimal_manifest("test.bad_label", "bad label", "dut", "kernel.cpp")
    )
    snapshot = bundle.snapshot()
    graph = CanonicalGraph(snapshot.id)
    graph.add_entity(Entity("hls.kernel", "dut", snapshot.id))
    bundle.store.save_graph(graph)
    dataset = DatasetManifest(
        dataset_id="test.bad", feature_schema_version=FEATURE_SCHEMA_VERSION,
        snapshot_ids=[snapshot.id],
        labels=[LabelSpec(
            label_id="missing", snapshot_id=snapshot.id,
            observation_id="observation_missing",
            predicate="qor.latency_cycles", stage="schedule", mask=True,
        )],
    )
    with pytest.raises(ValueError, match="unavailable observation"):
        export_dataset(bundle, snapshot.id, tmp_path / "dataset", dataset)
