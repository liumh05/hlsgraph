from __future__ import annotations

import json
from pathlib import Path

import pytest

from hlsgraph.api.rest import RestApplication, _public_run as rest_public_run
from hlsgraph.bundle import GraphBundle
from hlsgraph.export import export_dataset
from hlsgraph.graph import CanonicalGraph
from hlsgraph.manifest import minimal_manifest
from hlsgraph.mcp.service import ReadOnlyMcpService
from hlsgraph.model import Entity, FailureClass, RunStatus, ToolRun
from hlsgraph.query import CoreService
from hlsgraph.run_projection import sanitize_run_metadata
from hlsgraph.store import StoreError


def _public_bundle(tmp_path: Path) -> tuple[GraphBundle, str]:
    (tmp_path / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    bundle = GraphBundle.create(
        tmp_path, minimal_manifest("test.public_run", "public run", "dut", "kernel.cpp")
    )
    snapshot = bundle.snapshot()
    graph = CanonicalGraph(snapshot.id)
    graph.add_entity(Entity("hls.kernel", "dut", snapshot.id, stage="ast"))
    bundle.store.save_graph(graph)
    return bundle, snapshot.id


def test_tool_run_rejects_non_string_argv_at_construction_and_write(tmp_path: Path):
    with pytest.raises(ValueError, match="command.*non-empty strings"):
        ToolRun(
            "snapshot_valid", "index", "extractor.local", "a" * 64,
            command=[123],  # type: ignore[list-item]
        )

    bundle, snapshot_id = _public_bundle(tmp_path)
    run = ToolRun(snapshot_id, "index", "extractor.local", "b" * 64)
    run.command = [123]  # type: ignore[list-item]
    with pytest.raises(StoreError, match="command.*non-empty strings"):
        bundle.store.add_run(run)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("request_hash", "short", "request_hash"),
        ("environment_hash", "not-a-digest", "environment_hash"),
        ("toolchain_id", "has spaces", "toolchain_id"),
        ("attempt", True, "attempt"),
        ("elapsed_s", float("inf"), "elapsed_s"),
    ],
)
def test_tool_run_validates_public_contract_shapes(field: str, value: object, message: str):
    values: dict[str, object] = {
        "snapshot_id": "snapshot_valid", "stage": "index",
        "backend": "extractor.local", "request_hash": "a" * 64,
    }
    values[field] = value
    with pytest.raises(ValueError, match=message):
        ToolRun(**values)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="runner_fingerprint"):
        ToolRun(
            "snapshot_valid", "index", "extractor.local", "a" * 64,
            metadata={"runner_fingerprint": "not-a-fingerprint"},
        )

    with pytest.raises(ValueError, match="without NUL"):
        ToolRun(
            "snapshot_valid", "index", "extractor.local", "a" * 64,
            command=["tool\x00argument"],
        )


def test_shared_run_metadata_projection_is_positive_and_typed():
    secret = "PRIVATE_METADATA_SENTINEL/unsafe"
    projected = sanitize_run_metadata({
        "authority": "tool_observation",
        "tool_truth": True,
        "fresh_tool_truth": "true",
        "stdout_bytes": 12,
        "stderr_bytes": -1,
        "stdout_sha256": "A" * 64,
        "stderr_sha256": secret,
        "bootstrap_environment_hash": "c" * 64,
        "runner_fingerprint": "b" * 64,
        "campaign_id": "campaign.safe",
        "workload_id": secret,
        "input_mismatch_ids": ["artifact.safe", secret, 7],
        "unknown_nested": {"secret": secret},
        "failure_type": secret,
    })
    assert projected == {
        "tool_truth": True,
        "stdout_bytes": 12,
        "bootstrap_environment_hash": "c" * 64,
        "runner_fingerprint": "b" * 64,
        "stdout_sha256": "a" * 64,
        "campaign_id": "campaign.safe",
        "authority": "tool_observation",
        "input_mismatch_ids": ["artifact.safe"],
    }
    assert secret not in json.dumps(projected, sort_keys=True)


def test_run_metadata_sentinel_is_absent_from_rest_mcp_health_status_and_ml(
    tmp_path: Path,
):
    bundle, snapshot_id = _public_bundle(tmp_path)
    secret = "PRIVATE_RUN_METADATA_SENTINEL/41dd"
    run = ToolRun(
        snapshot_id=snapshot_id, stage="csynth", backend="runner.local",
        request_hash="c" * 64, status=RunStatus.FAILED,
        failure_class=FailureClass.INFRASTRUCTURE,
        command=["vendor-tool", "--private-argument"],
        metadata={
            "authority": "tool_observation",
            "tool_truth": secret,
            "fresh_tool_truth": False,
            "stdout_bytes": 9,
            "stdout_sha256": "d" * 64,
            "runner_fingerprint": "e" * 64,
            "campaign_id": "campaign.safe",
            "workload_id": secret,
            "unknown_nested": {"private": secret},
            "failure_type": secret,
        },
    )
    bundle.store.add_run(run)
    core = CoreService(bundle, snapshot_id)

    rest = RestApplication(core).dispatch("GET", "/api/v1/runs")
    assert rest.status == 200
    rest_item = next(item for item in rest.body["items"] if item["id"] == run.id)
    assert rest_item["metadata"]["fresh_tool_truth"] is False
    assert rest_item["metadata"]["stdout_bytes"] == 9
    assert rest_item["metadata"]["campaign_id"] == "campaign.safe"
    assert "tool_truth" not in rest_item["metadata"]

    mcp = ReadOnlyMcpService(core)
    mcp_runs = mcp.runs()["items"]
    mcp_item = next(item for item in mcp_runs if item["id"] == run.id)
    health = mcp.health()
    status = core.status().to_dict()
    assert mcp_item["metadata"]["runner_fingerprint"] == "e" * 64

    output = tmp_path / "ml-export"
    export_dataset(bundle, snapshot_id, output)
    ml_runs = [json.loads(line) for line in
               (output / "runs.jsonl").read_text(encoding="utf-8").splitlines()]
    ml_item = next(item for item in ml_runs if item["id"] == run.id)
    assert ml_item["metadata"]["stdout_sha256"] == "d" * 64
    assert ml_item["campaign_id"] == "campaign.safe"
    assert ml_item["workload_id"] is None

    for value in (rest.body, mcp.runs(), health, status, ml_runs):
        assert secret not in json.dumps(value, ensure_ascii=False, sort_keys=True)


def test_rest_public_run_tolerates_malformed_legacy_argv():
    payload = rest_public_run({
        "id": "run.safe", "snapshot_id": "snapshot.safe", "stage": "index",
        "backend": "extractor.local", "request_hash": "f" * 64,
        "status": "failed", "failure_class": "input", "attempt": 1,
        "command": [123], "started_at": "PRIVATE_TIMESTAMP_SENTINEL",
        "finished_at": "also-not-a-timestamp", "metadata": {},
    })
    assert payload["execution_metadata"]["argv0"] is None
    assert payload["execution_metadata"]["command_redacted"] is True
    assert payload["execution_metadata"]["command_hash"]
    assert payload["started_at"] is None
    assert payload["finished_at"] is None
