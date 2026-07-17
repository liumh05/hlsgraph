from __future__ import annotations

import json

import pytest

from hlsgraph.bundle import GraphBundle
from hlsgraph.graph import CanonicalGraph
from hlsgraph.manifest import minimal_manifest
from hlsgraph.model import (
    ArtifactRef,
    AuthorityClass,
    Derivation,
    Entity,
    GateStatus,
    Observation,
    RunStatus,
    ToolRun,
    ToolchainContext,
    VerificationKind,
    VerificationResult,
    json_ready,
    stable_hash,
)
from hlsgraph.query import CoreService
from hlsgraph.store import StoreError


def _bundle(tmp_path):
    (tmp_path / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    manifest = minimal_manifest(
        "test.evidence.closure", "evidence closure", "dut", "kernel.cpp",
        part="part-a", clock_ns=5.0,
    )
    manifest.target.capacities = {"lut": 100.0, "dsp": 10.0}
    manifest.target.reserved_resources = {"lut": 10.0, "dsp": 0.0}
    manifest.stage_commands = {
        "csim": ["vitis_hls", "--csim"],
        "rtl_cosim": ["vitis_hls", "--cosim"],
        "post_route": ["vivado", "--post-route"],
    }
    manifest.toolchains = [ToolchainContext(
        id="amd.unified.2024_2", vendor="amd", name="unified", version="2024.2",
        environment_hash="e" * 64,
    )]
    bundle = GraphBundle.create(tmp_path, manifest)
    snapshot = bundle.snapshot()
    kernel = Entity(
        "hls.kernel", "dut", snapshot.id, qualified_name="dut", stage="ast",
    )
    graph = CanonicalGraph(snapshot.id)
    graph.add_entity(kernel)
    bundle.store.save_graph(graph)
    return bundle, snapshot, kernel


def _run(bundle, snapshot, stage: str, char: str, *, workload: str | None = None):
    manifest = bundle.store.snapshot_manifest(snapshot.id)
    toolchain = manifest.toolchain_for_stage(stage)
    metadata = {
        "authority": "tool_observation", "tool_truth": True,
        "fresh_execution": True, "fresh_tool_truth": True,
    }
    if workload:
        metadata.update({
            "campaign_id": "campaign.golden", "workload_id": workload,
        })
    return ToolRun(
        snapshot.id, stage, "runner.local", char * 64,
        toolchain_id=toolchain.id, status=RunStatus.SUCCEEDED,
        command=list(manifest.stage_commands[stage]),
        working_directory=".",
        environment_hash=toolchain.environment_hash,
        input_artifact_ids=[
            item.id for item in bundle.store.artifacts(snapshot.id)
            if item.producer_run_id is None
        ],
        exit_code=0, metadata=metadata,
    )


def _scope(*, timing: bool = False):
    value = {
        "kind": "kernel", "top": "dut", "instance": "dut", "part": "part-a",
    }
    if timing:
        value["clock"] = "default"
    return {"scope": value}


def _managed_report(bundle, run, name: str, kind: str, metadata: dict):
    source = bundle.project_root / f"fixture-{name}"
    source.write_text(f"sanitized report {name}\n", encoding="utf-8")
    artifact, _path, _created = bundle.prepare_managed_artifact(
        source, kind=kind, role="tool_output", producer_run_id=run.id,
        metadata=metadata,
    )
    return artifact


def _commit_observations(bundle, run, artifacts, observations):
    run.output_artifact_ids = [item.id for item in artifacts]
    bundle.store.commit_run_result(
        run=run, artifacts=artifacts, observations=observations,
    )


def test_physical_derivation_rejects_cross_run_closure_and_query_defends_pollution(
    tmp_path,
):
    bundle, snapshot, kernel = _bundle(tmp_path)
    first = _run(bundle, snapshot, "post_route", "a")
    first_report = _managed_report(
        bundle, first, "a-util.rpt", "amd.vivado.post_route_utilization", _scope(),
    )
    lut = Observation(
        snapshot.id, kernel.id, "resource.lut", 50, "post_route",
        AuthorityClass.TOOL_OBSERVATION, run_id=first.id,
        artifact_id=first_report.id, unit="count",
    )
    _commit_observations(bundle, first, [first_report], [lut])

    second = _run(bundle, snapshot, "post_route", "b")
    second_report = _managed_report(
        bundle, second, "b-util.rpt", "amd.vivado.post_route_utilization", _scope(),
    )
    dsp = Observation(
        snapshot.id, kernel.id, "resource.dsp", 2, "post_route",
        AuthorityClass.TOOL_OBSERVATION, run_id=second.id,
        artifact_id=second_report.id, unit="count",
    )
    _commit_observations(bundle, second, [second_report], [dsp])

    polluted = Derivation(
        snapshot.id, kernel.id, "gate.resource_fits", True,
        "hlsgraph.gate.capacity_compare", "1", [lut.id, dsp.id],
        stage="post_route",
        metadata={"target_profile_hash": stable_hash(bundle.manifest.target)},
    )
    with pytest.raises(StoreError, match="exactly one"):
        bundle.store.add_derivations([polluted])

    # Simulate a ledger created by an older build: read-time trust must still fail.
    with bundle.store.write() as connection:
        connection.execute(
            "INSERT INTO derivations(id,snapshot_id,subject_id,predicate,payload_json) "
            "VALUES(?,?,?,?,?)",
            (polluted.id, snapshot.id, kernel.id, polluted.predicate,
             json.dumps(json_ready(polluted), sort_keys=True, separators=(",", ":"))),
        )
    gate = CoreService(bundle, snapshot.id).verification_gates()["resource_fits"]
    assert gate["status"] == "pass"
    assert gate["trusted_pass"] is False


def test_physical_gates_must_share_one_eligible_post_route_run(tmp_path):
    bundle, snapshot, kernel = _bundle(tmp_path)

    resource_run = _run(bundle, snapshot, "post_route", "c")
    utilization = _managed_report(
        bundle, resource_run, "util.rpt", "amd.vivado.post_route_utilization", _scope(),
    )
    resources = [
        Observation(
            snapshot.id, kernel.id, f"resource.{name}", value, "post_route",
            AuthorityClass.TOOL_OBSERVATION, run_id=resource_run.id,
            artifact_id=utilization.id, unit="count",
        )
        for name, value in (("lut", 50), ("dsp", 2))
    ]
    fits = Derivation(
        snapshot.id, kernel.id, "gate.resource_fits", True,
        "hlsgraph.gate.capacity_compare", "1", [item.id for item in resources],
        stage="post_route",
        metadata={"target_profile_hash": stable_hash(bundle.manifest.target)},
    )
    resource_run.output_artifact_ids = [utilization.id]
    bundle.store.commit_run_result(
        run=resource_run, artifacts=[utilization], observations=resources,
        derivations=[fits],
    )

    timing_run = _run(bundle, snapshot, "post_route", "d")
    timing_report = _managed_report(
        bundle, timing_run, "timing.rpt", "amd.vivado.post_route_timing",
        _scope(timing=True),
    )
    wns = Observation(
        snapshot.id, kernel.id, "timing.wns_ns", 0.1, "post_route",
        AuthorityClass.TOOL_OBSERVATION, run_id=timing_run.id,
        artifact_id=timing_report.id, unit="ns",
    )
    met = Derivation(
        snapshot.id, kernel.id, "gate.post_route_timing", True,
        "hlsgraph.gate.wns_nonnegative", "1", [wns.id], stage="post_route",
    )
    timing_run.output_artifact_ids = [timing_report.id]
    bundle.store.commit_run_result(
        run=timing_run, artifacts=[timing_report], observations=[wns],
        derivations=[met],
    )

    gates = CoreService(bundle, snapshot.id).verification_gates()
    assert gates["resource_fits"]["trusted_pass"] is True
    assert gates["post_route_timing"]["trusted_pass"] is True
    assert gates["eligible_physical_runs"] == []
    assert gates["verified"] is False


@pytest.mark.parametrize(
    ("algorithm", "wns", "derived"),
    [
        ("vendor.anything", 0.1, True),
        ("hlsgraph.gate.wns_nonnegative", -0.1, True),
    ],
)
def test_timing_gate_rejects_unapproved_or_contradictory_derivation(
    tmp_path, algorithm, wns, derived,
):
    bundle, snapshot, kernel = _bundle(tmp_path)
    run = _run(bundle, snapshot, "post_route", "f")
    report = _managed_report(
        bundle, run, "f-timing.rpt", "amd.vivado.post_route_timing",
        _scope(timing=True),
    )
    observation = Observation(
        snapshot.id, kernel.id, "timing.wns_ns", wns, "post_route",
        AuthorityClass.TOOL_OBSERVATION, run_id=run.id, artifact_id=report.id,
        unit="ns",
    )
    gate = Derivation(
        snapshot.id, kernel.id, "gate.post_route_timing", derived,
        algorithm, "1", [observation.id], stage="post_route",
    )
    run.output_artifact_ids = [report.id]
    with pytest.raises(StoreError, match="algorithm|contradicts"):
        bundle.store.commit_run_result(
            run=run, artifacts=[report], observations=[observation], derivations=[gate],
        )


def test_passing_csim_requires_parser_typed_complete_zero_value_observations(tmp_path):
    bundle, snapshot, kernel = _bundle(tmp_path)
    run = _run(bundle, snapshot, "csim", "1", workload="tb.one")
    report = _managed_report(
        bundle, run, "csim.json", "amd.vitis.csim_result",
        {"workload_id": "tb.one"},
    )
    observations = [
        Observation(
            snapshot.id, kernel.id, predicate, value, "csim",
            AuthorityClass.VERIFICATION_EVIDENCE, run_id=run.id,
            artifact_id=report.id, workload_id="tb.one", unit="count",
        )
        for predicate, value in (
            ("csim.exit_code", 0), ("csim.mismatches", 0),
            ("csim.assertions_failed", 0),
        )
    ]
    verification = VerificationResult(
        snapshot.id, VerificationKind.CSIM, GateStatus.PASS,
        run_id=run.id, workload_id="tb.one",
        evidence_ids=[item.id for item in observations],
    )
    run.output_artifact_ids = [report.id]
    bundle.store.commit_run_result(
        run=run, artifacts=[report], observations=observations,
        verifications=[verification],
    )
    check = CoreService(bundle, snapshot.id).verification_gates()["correctness"]
    assert check["checks"]["csim"]["trusted_pass"] is True

    cosim_run = _run(bundle, snapshot, "rtl_cosim", "3", workload="tb.one")
    cosim_report = _managed_report(
        bundle, cosim_run, "cosim.rpt", "amd.vitis.cosim_report",
        {"workload_id": "tb.one"},
    )
    cosim_status = Observation(
        snapshot.id, kernel.id, "cosim.status", "pass", "cosim",
        AuthorityClass.VERIFICATION_EVIDENCE, run_id=cosim_run.id,
        artifact_id=cosim_report.id, workload_id="tb.one",
    )
    cosim_verification = VerificationResult(
        snapshot.id, VerificationKind.RTL_COSIM, GateStatus.PASS,
        run_id=cosim_run.id, workload_id="tb.one",
        evidence_ids=[cosim_status.id],
    )
    cosim_run.output_artifact_ids = [cosim_report.id]
    bundle.store.commit_run_result(
        run=cosim_run, artifacts=[cosim_report], observations=[cosim_status],
        verifications=[cosim_verification],
    )

    bad_run = _run(bundle, snapshot, "csim", "2", workload="tb.one")
    arbitrary = _managed_report(
        bundle, bad_run, "not-a-report.bin", "vendor.arbitrary_binary",
        {"workload_id": "tb.one"},
    )
    bad_observations = [
        Observation(
            snapshot.id, kernel.id, predicate, value, "csim",
            AuthorityClass.VERIFICATION_EVIDENCE, run_id=bad_run.id,
            artifact_id=arbitrary.id, workload_id="tb.one", unit="count",
        )
        for predicate, value in (
            ("csim.exit_code", 0), ("csim.mismatches", 0),
            ("csim.assertions_failed", 0),
        )
    ]
    bad = VerificationResult(
        snapshot.id, VerificationKind.CSIM, GateStatus.PASS,
        run_id=bad_run.id, workload_id="tb.one",
        evidence_ids=[item.id for item in bad_observations],
    )
    bad_run.output_artifact_ids = [arbitrary.id]
    with pytest.raises(StoreError, match="typed|stage-aligned"):
        bundle.store.commit_run_result(
            run=bad_run, artifacts=[arbitrary], observations=bad_observations,
            verifications=[bad],
        )
    with pytest.raises(StoreError, match="not typed by the policy"):
        bundle.store.commit_run_result(
            run=bad_run, artifacts=[arbitrary], observations=bad_observations,
        )
    bundle.store.commit_run_result(run=bad_run, artifacts=[arbitrary])
    with bundle.store.write() as connection:
        for observation in bad_observations:
            connection.execute(
                "INSERT INTO observations(id,snapshot_id,subject_id,predicate,stage,authority,"
                "run_id,artifact_id,payload_json) VALUES(?,?,?,?,?,?,?,?,?)",
                (observation.id, snapshot.id, observation.subject_id,
                 observation.predicate, observation.stage, str(observation.authority),
                 observation.run_id, observation.artifact_id,
                 json.dumps(json_ready(observation), sort_keys=True, separators=(",", ":"))),
            )
        connection.execute(
            "INSERT INTO verifications(id,snapshot_id,kind,status,payload_json) "
            "VALUES(?,?,?,?,?)",
            (bad.id, snapshot.id, str(bad.kind), str(bad.status),
             json.dumps(json_ready(bad), sort_keys=True, separators=(",", ":"))),
        )
    polluted = CoreService(bundle, snapshot.id).verification_gates()["correctness"]
    bad_cohort = "campaign=campaign.golden;workload=tb.one"
    assert polluted["campaigns"][bad_cohort]["csim"]["trusted_pass"] is True
    assert polluted["eligible_run_ids"][bad_cohort] == {
        "csim": [run.id], "rtl_cosim": [cosim_run.id],
    }
    assert bad_run.id not in polluted["eligible_run_ids"][bad_cohort]["csim"]

    (bundle.project_root / report.uri).write_text("tampered report\n", encoding="utf-8")
    tampered = CoreService(bundle, snapshot.id).verification_gates()["correctness"]
    assert tampered["checks"]["csim"]["trusted_pass"] is False


def test_trusted_run_is_bound_to_snapshot_manifest_and_all_base_inputs(tmp_path):
    bundle, snapshot, _kernel = _bundle(tmp_path)
    valid = _run(bundle, snapshot, "csim", "3")
    bundle.store.add_run(valid)

    bad_command = _run(bundle, snapshot, "csim", "4")
    bad_command.command = ["different-tool"]
    with pytest.raises(StoreError, match="execution identity"):
        bundle.store.add_run(bad_command)

    bad_cwd = _run(bundle, snapshot, "csim", "6")
    bad_cwd.working_directory = "elsewhere"
    with pytest.raises(StoreError, match="execution identity"):
        bundle.store.add_run(bad_cwd)

    missing_input = _run(bundle, snapshot, "csim", "5")
    missing_input.input_artifact_ids = []
    with pytest.raises(StoreError, match="omits immutable snapshot"):
        bundle.store.add_run(missing_input)
