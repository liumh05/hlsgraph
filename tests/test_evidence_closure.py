from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
import sqlite3

import pytest

from hlsgraph.bundle import GraphBundle
from hlsgraph.extract.base import ExtractionContext
from hlsgraph.extract.vitis import VitisReportExtractor
from hlsgraph.extract.vivado import VivadoReportExtractor
from hlsgraph.graph import CanonicalGraph
from hlsgraph.knowledge.core import canonical_context_scalar
from hlsgraph.manifest import minimal_manifest
from hlsgraph.model import (
    ArtifactRef,
    AuthorityClass,
    Derivation,
    Entity,
    GateKind,
    GateResult,
    GateStatus,
    Observation,
    RunStatus,
    SourceAnchor,
    ToolOutputSpec,
    ToolRun,
    ToolchainContext,
    VerificationKind,
    VerificationResult,
    json_ready,
    stable_hash,
)
from hlsgraph.model import _observation_source_commitment
from hlsgraph.query import CoreService
import hlsgraph.retrieval as retrieval_module
from hlsgraph.retrieval import HybridRetriever
from hlsgraph.store import StoreError
from tests.attested_run_support import commit_attested
from tests.reviewed_knowledge_support import install_reviewed_builtin_packs


def _bundle(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    (tmp_path / "timing.xdc").write_text(
        "create_clock -period 5.0 [get_ports ap_clk]\n", encoding="utf-8",
    )
    manifest = minimal_manifest(
        "test.evidence.closure", "evidence closure", "dut", "kernel.cpp",
        part="part-a", clock_ns=5.0,
    )
    manifest.target.capacities = {"lut": 100.0, "dsp": 10.0}
    manifest.target.reserved_resources = {"lut": 10.0, "dsp": 0.0}
    manifest.constraints.xdc_files = ["timing.xdc"]
    manifest.stage_commands = {
        "csim": ["vitis_hls", "--csim"],
        "rtl_cosim": ["vitis_hls", "--cosim"],
        "post_route": ["vivado", "--post-route"],
    }
    manifest.toolchains = [ToolchainContext(
        id="amd.unified.2024_2", vendor="amd", name="unified", version="2024.2",
        environment_hash="e" * 64,
    )]
    manifest.stage_outputs = {
        "csim": [
            ToolOutputSpec(
                path="reports/csim.json", kind="amd.vitis.csim_result",
                required=False,
            ),
            ToolOutputSpec(
                path="reports/not-a-report.bin", kind="vendor.arbitrary_binary",
                required=False,
            ),
        ],
        "rtl_cosim": [
            ToolOutputSpec(
                path="reports/cosim.rpt", kind="amd.vitis.cosim_report",
                required=False,
            ),
        ],
        "post_route": [
            ToolOutputSpec(
                path=f"reports/{name}", kind=kind, required=False,
            )
            for name, kind in (
                ("a-util.rpt", "amd.vivado.post_route_utilization"),
                ("b-util.rpt", "amd.vivado.post_route_utilization"),
                ("util.rpt", "amd.vivado.post_route_utilization"),
                ("timing.rpt", "amd.vivado.post_route_timing"),
                ("routed.dcp", "amd.vivado.routed_checkpoint"),
                ("f-timing.rpt", "amd.vivado.post_route_timing"),
            )
        ],
    }
    bundle = GraphBundle.create(tmp_path, manifest)
    install_reviewed_builtin_packs(bundle)
    snapshot = bundle.snapshot()
    kernel = Entity(
        "hls.kernel", "dut", snapshot.id, qualified_name="dut", stage="ast",
    )
    graph = CanonicalGraph(snapshot.id)
    graph.add_entity(kernel)
    bundle.store.save_graph(graph)
    return bundle, snapshot, kernel


def _run(
    bundle, snapshot, stage: str, char: str, *, workload: str | None = None,
    backend: str = "runner.local",
):
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
        snapshot.id, stage, backend, char * 64,
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
    metadata = dict(metadata)
    fixture_wns = metadata.pop("_fixture_wns", 0.1)
    source = bundle.project_root / f"fixture-{name}"
    if kind == "amd.vitis.csim_result":
        content = json.dumps({
            "schema_version": "hlsgraph.vitis.csim.v1",
            "status": "pass", "exit_code": 0,
            "mismatches": 0, "assertions_failed": 0,
        }, sort_keys=True)
    elif kind in {"amd.vitis.cosim_report", "amd.vitis.cosim_rpt"}:
        content = "| Verilog | Pass | 1 | 1 | 1 | 1 | 1 | 1 |\n"
    elif kind == "amd.vivado.post_route_utilization":
        content = "LUT=50\nDSP=2\n"
    elif kind == "amd.vivado.post_route_timing":
        content = f"WNS: {fixture_wns}\nTNS: 0\n"
    else:
        content = f"sanitized artifact {name}\n"
    source.write_text(content, encoding="utf-8")
    artifact, _path, _created = bundle.prepare_managed_artifact(
        source, kind=kind, role="tool_output", producer_run_id=run.id,
        metadata={**metadata, "declared_output_path": f"reports/{name}"},
    )
    return artifact


def _parsed_observation(
    bundle, artifact, snapshot_id, subject_id, predicate, value, stage, authority,
    **values,
):
    parser_name = (
        "amd.vitis.reports" if artifact.kind.startswith("amd.vitis.")
        else "amd.vivado.reports" if artifact.kind.startswith("amd.vivado.")
        else "test.fixture.reports"
    )
    if parser_name != "test.fixture.reports":
        parser = (VitisReportExtractor()
                  if parser_name == "amd.vitis.reports"
                  else VivadoReportExtractor())
        snapshot = bundle.store.snapshot(snapshot_id)
        graph = bundle.store.load_graph(snapshot_id)
        result = parser.extract(ExtractionContext(
            project_root=bundle.project_root,
            manifest=bundle.store.snapshot_manifest(snapshot_id),
            snapshot=snapshot, artifacts={artifact.id: artifact},
            options={"existing_graph": graph},
        ))
        matches = [
            item for item in result.observations
            if (item.subject_id == subject_id and item.predicate == predicate
                and item.value == value and item.stage == stage
                and item.authority == authority
                and item.unit == values.get("unit")
                and item.workload_id == values.get("workload_id"))
        ]
        assert len(matches) == 1, [
            (item.subject_id, item.predicate, item.value, item.stage,
             item.authority, item.unit, item.workload_id)
            for item in result.observations
        ]
        return replace(matches[0], id="", run_id=values.get("run_id"))

    unit = values.get("unit")
    return Observation(
        snapshot_id, subject_id, predicate, value, stage, authority,
        artifact_id=artifact.id,
        anchor=SourceAnchor(artifact.id, ir_location=artifact.kind),
        source=_observation_source_commitment(
            artifact=artifact, parser_name=parser_name, parser_version="1",
            predicate=predicate, value=value, unit=unit,
        ),
        **values,
    )


def _commit_observations(bundle, run, artifacts, observations):
    run.output_artifact_ids = [item.id for item in artifacts]
    commit_attested(bundle,
        run=run, artifacts=artifacts, observations=observations,
    )


def _gate_contexts(bundle, snapshot, gate_kind: str):
    graph = bundle.store.load_graph(snapshot.id)
    return HybridRetriever(bundle, snapshot.id)._binding_target_contexts(
        graph, set(graph.entities),
    ).get(("gate_kind", gate_kind), [])


def _has_qualified_gate(bundle, snapshot, gate_kind: str) -> bool:
    return any(
        item.get("gate_evidence_qualified") == {
            "derived_from_typed_evidence_v1"
        }
        for item in _gate_contexts(bundle, snapshot, gate_kind)
    )


def _valid_csim_case(tmp_path, *, backend: str = "runner.local"):
    bundle, snapshot, kernel = _bundle(tmp_path)
    run = _run(
        bundle, snapshot, "csim", "e", workload="tb.gate", backend=backend,
    )
    report = _managed_report(
        bundle, run, "csim.json", "amd.vitis.csim_result",
        {"workload_id": "tb.gate"},
    )
    observations = [
        _parsed_observation(bundle, report,
            snapshot.id, kernel.id, predicate, value, "csim",
            AuthorityClass.VERIFICATION_EVIDENCE, run_id=run.id,
            workload_id="tb.gate", unit="count",
        )
        for predicate, value in (
            ("csim.exit_code", 0), ("csim.mismatches", 0),
            ("csim.assertions_failed", 0),
        )
    ]
    verification = VerificationResult(
        snapshot.id, VerificationKind.CSIM, GateStatus.PASS,
        run_id=run.id, workload_id="tb.gate",
        evidence_ids=[item.id for item in observations],
    )
    run.output_artifact_ids = [report.id]
    commit_attested(bundle,
        run=run, artifacts=[report], observations=observations,
        verifications=[verification],
    )
    assert _has_qualified_gate(bundle, snapshot, "correctness")
    return bundle, snapshot, run, report, observations


def _valid_timing_case(tmp_path):
    bundle, snapshot, kernel = _bundle(tmp_path)
    run = _run(bundle, snapshot, "post_route", "f")
    timing = _managed_report(
        bundle, run, "timing.rpt", "amd.vivado.post_route_timing",
        _scope(timing=True),
    )
    routed = _managed_report(
        bundle, run, "routed.dcp", "amd.vivado.routed_checkpoint",
        {**_scope(), "stage": "post_route"},
    )
    observation = _parsed_observation(bundle, timing,
        snapshot.id, kernel.id, "timing.wns_ns", 0.1, "post_route",
        AuthorityClass.TOOL_OBSERVATION, run_id=run.id,
        unit="ns",
    )
    derivation = Derivation(
        snapshot.id, kernel.id, "gate.post_route_timing", True,
        "hlsgraph.gate.wns_nonnegative", "1", [observation.id],
        stage="post_route",
    )
    run.output_artifact_ids = [timing.id, routed.id]
    commit_attested(bundle,
        run=run, artifacts=[timing, routed], observations=[observation],
        derivations=[derivation],
    )
    assert _has_qualified_gate(bundle, snapshot, "post_route_timing")
    return bundle, snapshot, run, timing, routed, observation


def _valid_resource_case(tmp_path):
    bundle, snapshot, kernel = _bundle(tmp_path)
    run = _run(bundle, snapshot, "post_route", "a")
    utilization = _managed_report(
        bundle, run, "util.rpt", "amd.vivado.post_route_utilization",
        _scope(),
    )
    observations = [
        _parsed_observation(bundle, utilization,
            snapshot.id, kernel.id, f"resource.{name}", value, "post_route",
            AuthorityClass.TOOL_OBSERVATION, run_id=run.id,
            unit="count",
        )
        for name, value in (("lut", 50), ("dsp", 2))
    ]
    derivation = Derivation(
        snapshot.id, kernel.id, "gate.resource_fits", True,
        "hlsgraph.gate.capacity_compare", "1",
        [item.id for item in observations], stage="post_route",
        metadata={"target_profile_hash": stable_hash(bundle.manifest.target)},
    )
    run.output_artifact_ids = [utilization.id]
    commit_attested(bundle,
        run=run, artifacts=[utilization], observations=observations,
        derivations=[derivation],
    )
    assert _has_qualified_gate(bundle, snapshot, "resource_fits")
    return bundle, snapshot, run


@pytest.mark.parametrize(
    ("stage", "gate_kind", "request_char", "workload"),
    [
        ("csim", GateKind.CORRECTNESS, "7", "tb.spoof-csim"),
        ("rtl_cosim", GateKind.CORRECTNESS, "8", "tb.spoof-cosim"),
        ("post_route", GateKind.RESOURCE_FITS, "9", None),
        ("post_route", GateKind.POST_ROUTE_TIMING, "0", None),
    ],
)
def test_gate_evidence_marker_cannot_be_injected_by_run_metadata(
    tmp_path, stage, gate_kind, request_char, workload,
):
    bundle, snapshot, _kernel = _bundle(tmp_path)
    run = _run(
        bundle, snapshot, stage, request_char, workload=workload,
    )
    run.metadata["gate_evidence_qualified"] = "derived_from_typed_evidence_v1"
    run.gates = [GateResult(gate_kind, GateStatus.FAIL)]
    commit_attested(bundle, run=run)

    graph = bundle.store.load_graph(snapshot.id)
    contexts = HybridRetriever(bundle, snapshot.id)._binding_target_contexts(
        graph, set(graph.entities),
    )[("gate_kind", str(gate_kind))]
    assert contexts
    assert all("gate_evidence_qualified" not in item for item in contexts)


def test_physical_derivation_rejects_cross_run_closure_and_query_defends_pollution(
    tmp_path,
):
    bundle, snapshot, kernel = _bundle(tmp_path)
    first = _run(bundle, snapshot, "post_route", "a")
    first_report = _managed_report(
        bundle, first, "a-util.rpt", "amd.vivado.post_route_utilization", _scope(),
    )
    lut = _parsed_observation(bundle, first_report,
        snapshot.id, kernel.id, "resource.lut", 50, "post_route",
        AuthorityClass.TOOL_OBSERVATION, run_id=first.id,
        unit="count",
    )
    _commit_observations(bundle, first, [first_report], [lut])

    second = _run(bundle, snapshot, "post_route", "b")
    second_report = _managed_report(
        bundle, second, "b-util.rpt", "amd.vivado.post_route_utilization", _scope(),
    )
    dsp = _parsed_observation(bundle, second_report,
        snapshot.id, kernel.id, "resource.dsp", 2, "post_route",
        AuthorityClass.TOOL_OBSERVATION, run_id=second.id,
        unit="count",
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
        _parsed_observation(bundle, utilization,
            snapshot.id, kernel.id, f"resource.{name}", value, "post_route",
            AuthorityClass.TOOL_OBSERVATION, run_id=resource_run.id,
            unit="count",
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
    commit_attested(bundle,
        run=resource_run, artifacts=[utilization], observations=resources,
        derivations=[fits],
    )

    timing_run = _run(bundle, snapshot, "post_route", "d")
    timing_report = _managed_report(
        bundle, timing_run, "timing.rpt", "amd.vivado.post_route_timing",
        _scope(timing=True),
    )
    routed_checkpoint = _managed_report(
        bundle, timing_run, "routed.dcp", "amd.vivado.routed_checkpoint",
        {**_scope(), "stage": "post_route"},
    )
    wns = _parsed_observation(bundle, timing_report,
        snapshot.id, kernel.id, "timing.wns_ns", 0.1, "post_route",
        AuthorityClass.TOOL_OBSERVATION, run_id=timing_run.id,
        unit="ns",
    )
    met = Derivation(
        snapshot.id, kernel.id, "gate.post_route_timing", True,
        "hlsgraph.gate.wns_nonnegative", "1", [wns.id], stage="post_route",
    )
    timing_run.output_artifact_ids = [timing_report.id, routed_checkpoint.id]
    commit_attested(bundle,
        run=timing_run, artifacts=[timing_report, routed_checkpoint], observations=[wns],
        derivations=[met],
    )

    gates = CoreService(bundle, snapshot.id).verification_gates()
    assert gates["resource_fits"]["trusted_pass"] is True
    assert gates["post_route_timing"]["trusted_pass"] is True
    assert gates["eligible_physical_runs"] == []
    assert gates["verified"] is False

    retriever = HybridRetriever(bundle, snapshot.id)
    graph = bundle.store.load_graph(snapshot.id)
    contexts = retriever._binding_target_contexts(graph, set(graph.entities))
    resource_context = contexts[("gate_kind", "resource_fits")]
    assert any(
        item.get("gate_evidence_qualified") == {
            "derived_from_typed_evidence_v1"
        }
        and item.get("complete_post_route_utilization") == {
            canonical_context_scalar(True),
        }
        and item.get("target_profile_hash")
        and item.get("target_device_identity")
        and item.get("capacity_identity")
        for item in resource_context
    )
    timing_context = contexts[("gate_kind", "post_route_timing")]
    assert any(
        item.get("gate_evidence_qualified") == {
            "derived_from_typed_evidence_v1"
        }
        and item.get("routed_design_identity") == {routed_checkpoint.sha256}
        and item.get("timing_report_identity")
        and item.get("constraint_hash") == {snapshot.constraint_hash}
        for item in timing_context
    )

    (bundle.project_root / utilization.uri).write_text(
        "tampered utilization\n", encoding="utf-8",
    )
    tampered_resource = retriever._binding_target_contexts(
        graph, set(graph.entities),
    ).get(("gate_kind", "resource_fits"), [])
    assert not any(
        item.get("gate_evidence_qualified") == {
            "derived_from_typed_evidence_v1"
        }
        for item in tampered_resource
    )

    (bundle.project_root / timing_report.uri).write_text(
        "tampered timing\n", encoding="utf-8",
    )
    tampered_timing = retriever._binding_target_contexts(
        graph, set(graph.entities),
    ).get(("gate_kind", "post_route_timing"), [])
    assert not any(
        item.get("gate_evidence_qualified") == {
            "derived_from_typed_evidence_v1"
        }
        for item in tampered_timing
    )


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
        {**_scope(timing=True), "_fixture_wns": wns},
    )
    observation = _parsed_observation(bundle, report,
        snapshot.id, kernel.id, "timing.wns_ns", wns, "post_route",
        AuthorityClass.TOOL_OBSERVATION, run_id=run.id,
        unit="ns",
    )
    gate = Derivation(
        snapshot.id, kernel.id, "gate.post_route_timing", derived,
        algorithm, "1", [observation.id], stage="post_route",
    )
    run.output_artifact_ids = [report.id]
    with pytest.raises(StoreError, match="algorithm|contradicts"):
        commit_attested(bundle,
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
        _parsed_observation(bundle, report,
            snapshot.id, kernel.id, predicate, value, "csim",
            AuthorityClass.VERIFICATION_EVIDENCE, run_id=run.id,
            workload_id="tb.one", unit="count",
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
    commit_attested(bundle,
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
    cosim_status = _parsed_observation(bundle, cosim_report,
        snapshot.id, kernel.id, "cosim.status", "pass", "cosim",
        AuthorityClass.VERIFICATION_EVIDENCE, run_id=cosim_run.id,
        workload_id="tb.one",
    )
    cosim_verification = VerificationResult(
        snapshot.id, VerificationKind.RTL_COSIM, GateStatus.PASS,
        run_id=cosim_run.id, workload_id="tb.one",
        evidence_ids=[cosim_status.id],
    )
    cosim_run.output_artifact_ids = [cosim_report.id]
    commit_attested(bundle,
        run=cosim_run, artifacts=[cosim_report], observations=[cosim_status],
        verifications=[cosim_verification],
    )

    bad_run = _run(bundle, snapshot, "csim", "2", workload="tb.one")
    arbitrary = _managed_report(
        bundle, bad_run, "not-a-report.bin", "vendor.arbitrary_binary",
        {"workload_id": "tb.one"},
    )
    bad_observations = [
        _parsed_observation(bundle, arbitrary,
            snapshot.id, kernel.id, predicate, value, "csim",
            AuthorityClass.VERIFICATION_EVIDENCE, run_id=bad_run.id,
            workload_id="tb.one", unit="count",
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
    with pytest.raises(StoreError, match="typed|stage-aligned|fixed built-in"):
        commit_attested(bundle,
            run=bad_run, artifacts=[arbitrary], observations=bad_observations,
            verifications=[bad],
        )
    with pytest.raises(StoreError, match="not typed by the policy|fixed built-in"):
        commit_attested(bundle,
            run=bad_run, artifacts=[arbitrary], observations=bad_observations,
        )
    commit_attested(bundle, run=bad_run, artifacts=[arbitrary])
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

    retriever = HybridRetriever(bundle, snapshot.id)
    graph = bundle.store.load_graph(snapshot.id)
    correctness_contexts = retriever._binding_target_contexts(
        graph, set(graph.entities),
    )[("gate_kind", "correctness")]
    qualified = [
        item for item in correctness_contexts
        if item.get("gate_evidence_qualified") == {
            "derived_from_typed_evidence_v1"
        }
    ]
    assert len(qualified) == 2
    assert all(
        item.get("snapshot_association") == {"verified"}
        and item.get("workload_id") == {"tb.one"}
        and item.get("verification_observation_identity")
        and item.get("verification_report_identity")
        for item in qualified
    )
    assert {next(iter(item["stage"])) for item in qualified} == {"csim", "cosim"}

    (bundle.project_root / report.uri).write_text("tampered report\n", encoding="utf-8")
    tampered = CoreService(bundle, snapshot.id).verification_gates()["correctness"]
    assert tampered["checks"]["csim"]["trusted_pass"] is False
    tampered_contexts = retriever._binding_target_contexts(
        graph, set(graph.entities),
    )[("gate_kind", "correctness")]
    assert sum(
        item.get("gate_evidence_qualified") == {
            "derived_from_typed_evidence_v1"
        }
        for item in tampered_contexts
    ) == 1


def test_trusted_run_is_bound_to_snapshot_manifest_and_all_base_inputs(tmp_path):
    bundle, snapshot, _kernel = _bundle(tmp_path)
    valid = _run(bundle, snapshot, "csim", "3")
    commit_attested(bundle, run=valid)

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


@pytest.mark.parametrize(
    ("case_name", "mutate"),
    [
        ("cached-status", lambda run: setattr(run, "status", RunStatus.CACHED)),
        ("replay-backend", lambda run: setattr(run, "backend", "runner.replay")),
        ("fake-backend", lambda run: setattr(run, "backend", "runner.fake")),
        ("failed-status", lambda run: setattr(run, "status", RunStatus.FAILED)),
        (
            "stale-truth",
            lambda run: run.metadata.__setitem__("fresh_tool_truth", False),
        ),
        (
            "non-tool-authority",
            lambda run: run.metadata.__setitem__("authority", "synthetic"),
        ),
    ],
)
def test_gate_marker_rejects_cached_fake_failed_or_stale_run(
    tmp_path, case_name, mutate,
):
    bundle, snapshot, run, _report, _observations = _valid_csim_case(
        tmp_path / case_name,
    )
    mutate(run)
    with sqlite3.connect(bundle.store.path) as connection:
        connection.execute(
            "UPDATE runs SET payload_json=? WHERE id=?",
            (json.dumps(json_ready(run), sort_keys=True), run.id),
        )
    assert not _has_qualified_gate(bundle, snapshot, "correctness")


def test_gate_marker_accepts_declared_fresh_ssh_run(tmp_path):
    bundle, snapshot, run, _report, _observations = _valid_csim_case(
        tmp_path, backend="runner.ssh",
    )
    assert run.backend == "runner.ssh"
    assert _has_qualified_gate(bundle, snapshot, "correctness")


def test_physical_gate_markers_also_reject_cached_runs(tmp_path):
    for case_name, maker, gate_kind in (
        ("resource", _valid_resource_case, "resource_fits"),
        ("timing", _valid_timing_case, "post_route_timing"),
    ):
        values = maker(tmp_path / case_name)
        bundle, snapshot, run = values[:3]
        run.status = RunStatus.CACHED
        with sqlite3.connect(bundle.store.path) as connection:
            connection.execute(
                "UPDATE runs SET payload_json=? WHERE id=?",
                (json.dumps(json_ready(run), sort_keys=True), run.id),
            )
        assert not _has_qualified_gate(bundle, snapshot, gate_kind)


def test_gate_marker_rejects_undeclared_output_identity(tmp_path):
    bundle, snapshot, _run_value, report, _observations = _valid_csim_case(
        tmp_path,
    )
    report.metadata["declared_output_path"] = "reports/undeclared.json"
    with sqlite3.connect(bundle.store.path) as connection:
        connection.execute(
            "UPDATE artifacts SET payload_json=? WHERE id=?",
            (json.dumps(json_ready(report), sort_keys=True), report.id),
        )
    assert not _has_qualified_gate(bundle, snapshot, "correctness")


def test_gate_marker_rejects_changed_snapshot_manifest(tmp_path):
    bundle, snapshot, _run_value, _report, _observations = _valid_csim_case(
        tmp_path,
    )
    manifest = bundle.store.snapshot_manifest(snapshot.id)
    manifest.stage_commands["csim"] = ["vitis_hls", "--different-csim"]
    with sqlite3.connect(bundle.store.path) as connection:
        connection.execute(
            "UPDATE snapshot_manifests SET manifest_json=? WHERE snapshot_id=?",
            (json.dumps(json_ready(manifest), sort_keys=True), snapshot.id),
        )
    assert not _has_qualified_gate(bundle, snapshot, "correctness")


def test_gate_marker_rejects_linked_report_path(tmp_path):
    bundle, snapshot, _run_value, report, _observations = _valid_csim_case(
        tmp_path,
    )
    report_path = bundle.project_root / report.uri
    replacement = tmp_path / "same-report-bytes.json"
    replacement.write_bytes(report_path.read_bytes())
    report_path.unlink()
    try:
        os.symlink(replacement, report_path)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")
    assert not _has_qualified_gate(bundle, snapshot, "correctness")


def test_gate_marker_rejects_report_path_flagged_as_reparse(
    tmp_path, monkeypatch,
):
    bundle, snapshot, _run_value, report, _observations = _valid_csim_case(
        tmp_path,
    )
    report_path = (bundle.project_root / report.uri).resolve()
    original = retrieval_module._is_link_or_reparse

    def flagged(path):
        return Path(path).resolve() == report_path or original(path)

    monkeypatch.setattr(retrieval_module, "_is_link_or_reparse", flagged)
    assert not _has_qualified_gate(bundle, snapshot, "correctness")


def test_timing_gate_revalidates_xdc_bytes_and_binds_each_input_identity(tmp_path):
    bundle, snapshot, _run_value, _timing, _routed, _observation = (
        _valid_timing_case(tmp_path)
    )
    xdc = next(
        item for item in bundle.store.artifacts(snapshot.id)
        if item.kind == "constraint.xdc"
    )
    context = next(
        item for item in _gate_contexts(bundle, snapshot, "post_route_timing")
        if item.get("gate_evidence_qualified") == {
            "derived_from_typed_evidence_v1"
        }
    )
    assert context["constraint_artifact_identity"] == {stable_hash([{
        "artifact_id": xdc.id, "uri": xdc.uri, "sha256": xdc.sha256,
    }])}

    (bundle.project_root / xdc.uri).write_text(
        "create_clock -period 7.5 [get_ports ap_clk]\n", encoding="utf-8",
    )
    assert not _has_qualified_gate(bundle, snapshot, "post_route_timing")


def test_gate_marker_rejects_leaf_run_mismatch(tmp_path):
    bundle, snapshot, run, _timing, _routed, observation = _valid_timing_case(
        tmp_path,
    )
    other = _run(bundle, snapshot, "post_route", "1")
    commit_attested(bundle, run=other)
    polluted = replace(observation, id="", run_id=other.id)
    with sqlite3.connect(bundle.store.path) as connection:
        connection.execute(
            "UPDATE observations SET run_id=?,payload_json=? WHERE id=?",
            (other.id, json.dumps(json_ready(polluted), sort_keys=True),
             observation.id),
        )
    assert other.id != run.id
    assert not _has_qualified_gate(bundle, snapshot, "post_route_timing")
