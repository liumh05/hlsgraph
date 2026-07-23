from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from hlsgraph.bundle import GraphBundle
from hlsgraph.extract.directive_identity import (
    bind_directive_identity,
    directive_identity_metadata,
)
from hlsgraph.extract.base import ExtractionContext
from hlsgraph.extract.vitis import VitisReportExtractor
from hlsgraph.extract.vivado import VivadoReportExtractor
from hlsgraph.graph import CanonicalGraph
from hlsgraph.manifest import minimal_manifest
from hlsgraph.model import (
    ArtifactRef,
    AuthorityClass,
    Entity,
    Observation,
    ObservationSource,
    Relation,
    RunStatus,
    SourceAnchor,
    ToolOutputSpec,
    ToolRun,
    ToolchainContext,
    json_ready,
    stable_hash,
)
from hlsgraph.model import _observation_source_commitment
from hlsgraph.retrieval import HybridRetriever
from hlsgraph.sdk import Project
from hlsgraph.store import StoreError
from tests.attested_run_support import commit_attested
from tests.reviewed_knowledge_support import install_reviewed_builtin_packs


_TOKEN = "derived_from_typed_observation_evidence_v1"


def _typed_report_bytes(
    artifact_kind: str, *, activity_source: str | None,
) -> bytes:
    """Return a minimal report accepted by the fixed public parser."""

    if artifact_kind == "amd.vitis.csynth_xml":
        return b"""<profile>
  <UserAssignments><TopModelName>dut</TopModelName><TargetClockPeriod>1</TargetClockPeriod></UserAssignments>
  <SummaryOfTimingAnalysis><EstimatedClockPeriod>1</EstimatedClockPeriod></SummaryOfTimingAnalysis>
  <SummaryOfOverallLatency>
    <Best-caseLatency>1</Best-caseLatency><Worst-caseLatency>1</Worst-caseLatency>
    <Interval-min>1</Interval-min><Interval-max>1</Interval-max>
  </SummaryOfOverallLatency>
  <SummaryOfLoopLatency><loop0>
    <TripCount>1</TripCount><Latency>1</Latency><IterationLatency>1</IterationLatency>
    <PipelineII>1</PipelineII><PipelineDepth>1</PipelineDepth>
  </loop0></SummaryOfLoopLatency>
  <AreaEstimates><Resources><LUT>1</LUT></Resources></AreaEstimates>
</profile>\n"""
    if artifact_kind == "amd.vitis.schedule_json":
        return json.dumps({
            "schema_version": "hlsgraph.vitis.schedule.v1",
            "top": "dut",
            "operations": [{
                "name": "op0", "architecture_name": "dut",
                "start_cycle": 1, "end_cycle": 1, "pipeline_stage": 1,
                "latency": 1, "achieved_ii": 1, "target_ii": 1,
            }],
        }, sort_keys=True).encode()
    if artifact_kind == "amd.vitis.csim_result":
        return json.dumps({
            "schema_version": "hlsgraph.vitis.csim.v1",
            "status": "fail", "exit_code": 1,
            "mismatches": 1, "assertions_failed": 1,
        }, sort_keys=True).encode()
    if artifact_kind in {"amd.vitis.cosim_rpt", "amd.vitis.cosim_report"}:
        return b"| Verilog | Pass | 1 | 1 | 1 | 1 | 1 | 1 |\n"
    if artifact_kind == "amd.vitis.dataflow_profile":
        return json.dumps({
            "schema_version": "hlsgraph.vitis.dataflow_profile.v1",
            "channels": [{
                "name": "fifo0", "max_occupancy": 1,
                "read_block_cycles": 1, "write_block_cycles": 1,
                "tokens": 1,
            }],
        }, sort_keys=True).encode()
    if artifact_kind == "amd.vitis.directive_status":
        return json.dumps({
            "schema_version": "hlsgraph.vitis.directive_status.v1",
            "tool": "vitis_hls", "tool_version": "2024.2",
            "directives": [{
                "directive_kind": "PIPELINE", "scope": "dut",
                "status": "applied",
            }],
        }, sort_keys=True).encode()
    if artifact_kind in {
        "amd.vivado.timing_summary", "amd.vivado.post_route_timing",
    }:
        return b"WNS: 1\nTNS: 1\n"
    if artifact_kind in {
        "amd.vivado.utilization", "amd.vivado.post_route_utilization",
    }:
        return b"LUT=1\n"
    if artifact_kind == "amd.vivado.physical_summary":
        return json.dumps({
            "schema_version": "hlsgraph.vivado.physical_summary.v1",
            "congestion_level": 1, "slr_crossings": 1,
            "critical_path_delay_ns": 1, "drc_errors": 1,
            "cdc_critical": 1, "dynamic_power_w": 1, "static_power_w": 1,
            "activity_source": activity_source,
        }, sort_keys=True).encode()
    raise AssertionError(f"no typed report fixture for {artifact_kind}")


def _observation_case(
    root: Path,
    *,
    predicate: str,
    observation_stage: str,
    run_stage: str,
    artifact_kind: str,
    authority: AuthorityClass,
    activity_source: str | None = None,
    workload_id: str | None = None,
    run_workload_id: str | None = None,
    artifact_workload_id: str | None = None,
    testcase_id: str | None = None,
    run_testcase_id: str | None = None,
    artifact_testcase_id: str | None = None,
    declared: bool = True,
    injected_metadata: bool = False,
    duplicate_declared_path: bool = False,
    duplicate_run_ownership: bool = False,
    forge_source_hash: bool = False,
    forge_report_value: bool = False,
):
    root.mkdir()
    (root / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    tool = "vitis_hls" if artifact_kind.startswith("amd.vitis.") else "vivado"
    manifest = minimal_manifest(
        f"test.typed_observation.{stable_hash(root.name)[:12]}",
        "typed observation evidence", "dut", "kernel.cpp",
    )
    manifest.toolchains = [ToolchainContext(
        id=f"amd.{tool}.2024_2", vendor="amd", name=tool,
        version="2024.2", environment_hash="e" * 64,
    )]
    manifest.stage_commands = {run_stage: [tool, f"--{run_stage}"]}
    declared_specs = [ToolOutputSpec(
        path="declared/report.dat", kind=artifact_kind,
    )]
    if duplicate_declared_path:
        declared_specs.append(ToolOutputSpec(
            path="declared/report.dat", kind=artifact_kind,
        ))
    manifest.stage_outputs = ({
        run_stage: declared_specs,
    } if declared else {})
    bundle = GraphBundle.create(root, manifest)
    install_reviewed_builtin_packs(bundle)
    snapshot = bundle.snapshot()
    kernel = Entity(
        "hls.kernel", "dut", snapshot.id, qualified_name="dut", stage="ast",
    )
    loop = Entity(
        "hls.loop", "loop0", snapshot.id, qualified_name="dut::loop0",
        stage="ast",
    )
    stream = Entity(
        "hls.stream", "fifo0", snapshot.id, qualified_name="dut::fifo0",
        stage="ast",
    )
    graph = CanonicalGraph(snapshot.id)
    graph.add_entity(kernel)
    graph.add_entity(loop)
    graph.add_entity(stream)
    bundle.store.save_graph(graph)

    toolchain = manifest.toolchain_for_stage(run_stage)
    effective_run_workload = run_workload_id or workload_id
    effective_artifact_workload = artifact_workload_id or workload_id
    effective_run_testcase = run_testcase_id or testcase_id
    effective_artifact_testcase = artifact_testcase_id or testcase_id
    run_metadata: dict[str, object] = {
        "authority": "tool_observation", "tool_truth": True,
        "fresh_execution": True, "fresh_tool_truth": True,
    }
    if effective_run_workload:
        run_metadata["workload_id"] = effective_run_workload
    if effective_run_testcase:
        run_metadata["testcase_id"] = effective_run_testcase
    run = ToolRun(
        snapshot.id, run_stage, "runner.local", stable_hash({
            "predicate": predicate, "kind": artifact_kind,
        }),
        toolchain_id=toolchain.id, status=RunStatus.SUCCEEDED,
        command=list(manifest.stage_commands[run_stage]), working_directory=".",
        environment_hash=toolchain.environment_hash,
        input_artifact_ids=[
            item.id for item in bundle.store.artifacts(snapshot.id)
            if item.producer_run_id is None
        ],
        exit_code=0,
        metadata=run_metadata,
    )
    source = root / "report.dat"
    source.write_bytes(_typed_report_bytes(
        artifact_kind, activity_source=activity_source,
    ))
    artifact_metadata: dict[str, object] = {
        "declared_output_path": "declared/report.dat",
        "stage": observation_stage,
    }
    if artifact_kind.startswith("amd.vivado."):
        artifact_metadata["scope"] = {
            "kind": "kernel", "top": "dut", "instance": "dut",
            "clock": "default",
        }
    if authority == AuthorityClass.SYNTHETIC:
        artifact_metadata["fixture_authority"] = "synthetic"
    if effective_artifact_workload:
        artifact_metadata["workload_id"] = effective_artifact_workload
    if effective_artifact_testcase:
        artifact_metadata["testcase_id"] = effective_artifact_testcase
    artifact, artifact_path, _created = bundle.prepare_managed_artifact(
        source, kind=artifact_kind, role="tool_output", producer_run_id=run.id,
        metadata=artifact_metadata,
    )
    metadata: dict[str, object] = {}
    if injected_metadata:
        metadata.update({
            "observation_evidence_qualified": _TOKEN,
            "observation_instance_id": "observation.injected",
            "observation_artifact_kind": "amd.vitis.csynth_xml",
            "observation_artifact_identity": "f" * 64,
            "observation_parser_identity": "b" * 64,
            "observation_source_identity": "c" * 64,
            "observation_run_identity": "a" * 64,
        })
    parser = (VitisReportExtractor() if artifact_kind.startswith("amd.vitis.")
              else VivadoReportExtractor())
    parsed = parser.extract(ExtractionContext(
        project_root=root, manifest=manifest, snapshot=snapshot,
        artifacts={artifact.id: artifact}, options={"existing_graph": graph},
    ))
    matching = [item for item in parsed.observations if item.predicate == predicate]
    if len(matching) == 1:
        parsed_observation = matching[0]
        parser_source = parsed_observation.source
        assert parser_source is not None
        observation = replace(
            parsed_observation, id="", run_id=run.id,
            metadata={**parsed_observation.metadata, **metadata},
        )
    else:
        # Deliberately create a self-consistent receipt for a predicate the
        # chosen report parser did not emit.  Store replay must reject it.
        parser_source = _observation_source_commitment(
            artifact=artifact,
            parser_name=("amd.vitis.reports"
                         if artifact_kind.startswith("amd.vitis.")
                         else "amd.vivado.reports"),
            parser_version="1", predicate=predicate, value=1, unit="count",
        )
        observation = Observation(
            snapshot.id, kernel.id, predicate, 1, observation_stage, authority,
            run_id=run.id, artifact_id=artifact.id, unit="count",
            anchor=SourceAnchor(artifact.id, ir_location="fixture.report"),
            source=parser_source,
            workload_id=workload_id,
            metadata=metadata,
        )
    if forge_source_hash:
        forged_hash = "f" * 64
        parser_source = replace(
            parser_source,
            artifact_sha256=forged_hash,
            binding_sha256=stable_hash({
                "contract": parser_source.contract,
                "artifact_id": parser_source.artifact_id,
                "artifact_sha256": forged_hash,
                "parser_name": parser_source.parser_name,
                "parser_version": parser_source.parser_version,
                "payload_sha256": parser_source.payload_sha256,
            }),
        )
        observation = replace(observation, id="", source=parser_source)
    if forge_report_value:
        forged_value = 999
        observation = replace(
            observation, id="", value=forged_value,
            source=_observation_source_commitment(
                artifact=artifact,
                parser_name=parser_source.parser_name,
                parser_version=parser_source.parser_version,
                predicate=observation.predicate,
                value=forged_value,
                unit=observation.unit,
            ),
        )
    artifacts = [artifact]
    if duplicate_run_ownership:
        sibling_source = root / "sibling-report.dat"
        sibling_source.write_bytes(b"sibling report bytes\n")
        sibling, _sibling_path, _created = bundle.prepare_managed_artifact(
            sibling_source, kind=artifact_kind, role="tool_output",
            producer_run_id=run.id,
            metadata={**artifact_metadata,
                      "declared_output_path": "declared/report.dat"},
        )
        artifacts.append(sibling)
    run.output_artifact_ids = [item.id for item in artifacts]
    commit_attested(bundle,
        run=run, artifacts=artifacts, observations=[observation],
    )
    return bundle, snapshot, graph, observation, artifact, artifact_path, tool


@pytest.mark.parametrize(
    ("predicate", "observation_stage", "run_stage", "artifact_kind", "authority", "activity"),
    [
        ("clock.estimated_period_ns", "schedule", "csynth",
         "amd.vitis.csynth_xml", AuthorityClass.TOOL_OBSERVATION, None),
        ("qor.latency_cycles", "schedule", "csynth",
         "amd.vitis.csynth_xml", AuthorityClass.TOOL_OBSERVATION, None),
        ("qor.latency_best_cycles", "schedule", "csynth",
         "amd.vitis.csynth_xml", AuthorityClass.TOOL_OBSERVATION, None),
        ("qor.latency_worst_cycles", "schedule", "csynth",
         "amd.vitis.csynth_xml", AuthorityClass.TOOL_OBSERVATION, None),
        ("qor.interval_min_cycles", "schedule", "csynth",
         "amd.vitis.csynth_xml", AuthorityClass.TOOL_OBSERVATION, None),
        ("qor.interval_max_cycles", "schedule", "csynth",
         "amd.vitis.csynth_xml", AuthorityClass.TOOL_OBSERVATION, None),
        ("qor.iteration_latency_cycles", "schedule", "csynth",
         "amd.vitis.csynth_xml", AuthorityClass.TOOL_OBSERVATION, None),
        ("qor.achieved_ii", "schedule", "csynth",
         "amd.vitis.csynth_xml", AuthorityClass.TOOL_OBSERVATION, None),
        ("qor.achieved_ii", "schedule", "csynth",
         "amd.vitis.schedule_json", AuthorityClass.COMPILER_DECISION, None),
        ("qor.target_ii", "schedule", "csynth",
         "amd.vitis.schedule_json", AuthorityClass.COMPILER_DECISION, None),
        ("schedule.start_cycle", "schedule", "csynth",
         "amd.vitis.schedule_json", AuthorityClass.COMPILER_DECISION, None),
        ("timing.wns_ns", "post_route", "post_route",
         "amd.vivado.timing_summary", AuthorityClass.TOOL_OBSERVATION, None),
        ("resource.lut", "post_route", "post_route",
         "amd.vivado.utilization", AuthorityClass.TOOL_OBSERVATION, None),
        ("physical.congestion_level", "post_place", "post_place",
         "amd.vivado.physical_summary", AuthorityClass.TOOL_OBSERVATION, None),
        ("power.dynamic_w", "post_route", "post_route",
         "amd.vivado.physical_summary", AuthorityClass.TOOL_OBSERVATION,
         "vectorless-defaults"),
    ],
)
def test_amd_observation_binding_requires_current_declared_report_closure(
    tmp_path: Path,
    predicate: str,
    observation_stage: str,
    run_stage: str,
    artifact_kind: str,
    authority: AuthorityClass,
    activity: str | None,
) -> None:
    bundle, snapshot, graph, observation, artifact, _path, tool = _observation_case(
        tmp_path / predicate.replace(".", "-"), predicate=predicate,
        observation_stage=observation_stage, run_stage=run_stage,
        artifact_kind=artifact_kind, authority=authority,
        activity_source=activity,
    )
    contexts = HybridRetriever(bundle, snapshot.id)._binding_target_contexts(
        graph, set(graph.entities),
    )[("predicate", predicate)]
    context = next(
        item for item in contexts
        if item.get("observation_instance_id") == {observation.id.casefold()}
    )
    assert context["observation_evidence_qualified"] == {_TOKEN}
    assert context["snapshot_association"] == {"verified"}
    assert context["observation_artifact_identity"] == {stable_hash([{
        "artifact_id": artifact.id, "sha256": artifact.sha256,
    }])}
    assert context["observation_source_identity"] == {
        stable_hash(observation.source)
    }
    assert context["observation_parser_identity"] == {stable_hash({
        "name": observation.source.parser_name,
        "version": observation.source.parser_version,
        "contract": observation.source.contract,
    })}
    assert context["observation_run_identity"]
    assert context["observation_artifact_kind"] == {artifact_kind}

    bindings = [
        item for item in bundle.store.knowledge_bindings()
        if item.target_kind == "predicate" and item.target == predicate
        and item.required_context.get("tool") == tool
    ]
    if predicate in {
        "clock.estimated_period_ns", "qor.target_ii", "qor.achieved_ii",
    }:
        assert bindings == []
        return
    assert bindings
    assert all(HybridRetriever._binding_constraints_match_values(
        binding, context, {"predicate": {predicate}},
    ) for binding in bindings)
    assert not any(
        binding.knowledge_rule_id.endswith(":qor.csynth_is_estimate")
        for binding in bindings
    )


def test_observation_marker_fails_closed_on_tamper_and_metadata_injection(
    tmp_path: Path,
) -> None:
    case = _observation_case(
        tmp_path / "injection", predicate="timing.wns_ns",
        observation_stage="post_route", run_stage="post_route",
        artifact_kind="amd.vivado.timing_summary",
        authority=AuthorityClass.SYNTHETIC, injected_metadata=True,
    )
    bundle, snapshot, graph, _observation, _artifact, artifact_path, _tool = case
    retriever = HybridRetriever(bundle, snapshot.id)
    contexts = retriever._binding_target_contexts(
        graph, set(graph.entities),
    )[("predicate", "timing.wns_ns")]
    assert contexts
    assert all("observation_evidence_qualified" not in item for item in contexts)
    assert all("observation_instance_id" not in item for item in contexts)
    assert all("observation_artifact_kind" not in item for item in contexts)
    assert all("observation_parser_identity" not in item for item in contexts)
    assert all("observation_source_identity" not in item for item in contexts)

    valid = _observation_case(
        tmp_path / "tamper", predicate="timing.wns_ns",
        observation_stage="post_route", run_stage="post_route",
        artifact_kind="amd.vivado.timing_summary",
        authority=AuthorityClass.TOOL_OBSERVATION,
    )
    bundle, snapshot, graph, observation, _artifact, artifact_path, _tool = valid
    retriever = HybridRetriever(bundle, snapshot.id)
    assert any(
        item.get("observation_instance_id") == {observation.id.casefold()}
        for item in retriever._binding_target_contexts(
            graph, set(graph.entities),
        )[("predicate", "timing.wns_ns")]
    )
    artifact_path.write_text("tampered report\n", encoding="utf-8")
    assert not any(
        item.get("observation_evidence_qualified") == {_TOKEN}
        for item in retriever._binding_target_contexts(
            graph, set(graph.entities),
        )[("predicate", "timing.wns_ns")]
    )


@pytest.mark.parametrize(
    ("artifact_kind", "declared"),
    [
        ("amd.vivado.utilization", True),
        ("amd.vivado.timing_summary", False),
    ],
)
def test_observation_marker_rejects_wrong_report_kind_or_undeclared_output(
    tmp_path: Path, artifact_kind: str, declared: bool,
) -> None:
    if not declared or artifact_kind == "amd.vivado.utilization":
        expected = ("lacks a declared output path" if not declared
                    else "parser replay failed")
        with pytest.raises(StoreError, match=expected):
            _observation_case(
                tmp_path / f"negative-{artifact_kind.rsplit('.', 1)[-1]}-{declared}",
                predicate="timing.wns_ns", observation_stage="post_route",
                run_stage="post_route", artifact_kind=artifact_kind,
                authority=AuthorityClass.TOOL_OBSERVATION, declared=declared,
            )
        return
    bundle, snapshot, graph, _observation, _artifact, _path, _tool = _observation_case(
        tmp_path / f"negative-{artifact_kind.rsplit('.', 1)[-1]}-{declared}",
        predicate="timing.wns_ns", observation_stage="post_route",
        run_stage="post_route", artifact_kind=artifact_kind,
        authority=AuthorityClass.TOOL_OBSERVATION, declared=declared,
    )
    contexts = HybridRetriever(bundle, snapshot.id)._binding_target_contexts(
        graph, set(graph.entities),
    )[("predicate", "timing.wns_ns")]
    assert contexts
    assert all("observation_evidence_qualified" not in item for item in contexts)


def test_typed_observation_model_rejects_two_report_anchors_and_sibling_source() -> None:
    first = ArtifactRef(
        "amd.vitis.csynth_xml", "reports/first.xml", "1" * 64, 1,
    )
    sibling = ArtifactRef(
        "amd.vitis.csynth_xml", "reports/sibling.xml", "2" * 64, 1,
    )
    source = _observation_source_commitment(
        artifact=first, parser_name="amd.vitis.reports", parser_version="1",
        predicate="qor.latency_cycles", value=7, unit="cycle",
    )
    with pytest.raises(ValueError, match="artifact_id must match"):
        Observation(
            "snapshot.test", "entity.test", "qor.latency_cycles", 7,
            "schedule", AuthorityClass.TOOL_OBSERVATION,
            artifact_id=first.id,
            anchor=SourceAnchor(sibling.id, ir_location="sibling"),
            source=source, unit="cycle",
        )
    sibling_source = _observation_source_commitment(
        artifact=sibling, parser_name="amd.vitis.reports", parser_version="1",
        predicate="qor.latency_cycles", value=7, unit="cycle",
    )
    with pytest.raises(ValueError, match="canonical artifact"):
        Observation(
            "snapshot.test", "entity.test", "qor.latency_cycles", 7,
            "schedule", AuthorityClass.TOOL_OBSERVATION,
            artifact_id=first.id,
            anchor=SourceAnchor(first.id, ir_location="first"),
            source=sibling_source, unit="cycle",
        )


def test_parser_source_binds_canonical_value_and_explicit_null_unit() -> None:
    assert not hasattr(ObservationSource, "issue")
    artifact = ArtifactRef(
        "amd.vitis.directive_status", "reports/directives.json", "3" * 64, 1,
    )
    source = _observation_source_commitment(
        artifact=artifact, parser_name="amd.vitis.reports", parser_version="1",
        predicate="directive.tool_status", value="applied", unit=None,
    )
    assert source.payload_sha256 == stable_hash({
        "predicate": "directive.tool_status", "value": "applied", "unit": None,
    })
    forged_payload = stable_hash({
        "predicate": "directive.tool_status", "value": "ignored", "unit": None,
    })
    forged = replace(
        source, payload_sha256=forged_payload,
        binding_sha256=stable_hash({
            "contract": source.contract,
            "artifact_id": source.artifact_id,
            "artifact_sha256": source.artifact_sha256,
            "parser_name": source.parser_name,
            "parser_version": source.parser_version,
            "payload_sha256": forged_payload,
        }),
    )
    with pytest.raises(ValueError, match="predicate/value/unit"):
        Observation(
            "snapshot.test", "entity.test", "directive.tool_status", "applied",
            "schedule", AuthorityClass.TOOL_OBSERVATION,
            artifact_id=artifact.id,
            anchor=SourceAnchor(artifact.id, ir_location="directive_status"),
            source=forged,
        )


def test_typed_source_extends_only_new_observation_identity() -> None:
    artifact = ArtifactRef(
        "amd.vitis.csynth_xml", "reports/csynth.xml", "4" * 64, 1,
    )
    legacy = Observation(
        "snapshot.test", "entity.test", "qor.latency_cycles", 7,
        "schedule", AuthorityClass.TOOL_OBSERVATION,
        artifact_id=artifact.id, unit="cycle",
    )
    assert Observation.from_dict(json_ready(legacy)).id == legacy.id
    typed = Observation(
        "snapshot.test", "entity.test", "qor.latency_cycles", 7,
        "schedule", AuthorityClass.TOOL_OBSERVATION,
        artifact_id=artifact.id,
        anchor=SourceAnchor(artifact.id, ir_location="csynth.xml"),
        source=_observation_source_commitment(
            artifact=artifact, parser_name="amd.vitis.reports",
            parser_version="1", predicate="qor.latency_cycles",
            value=7, unit="cycle",
        ),
        unit="cycle",
    )
    assert typed.id != legacy.id
    assert Observation.from_dict(json_ready(typed)).id == typed.id


def test_store_rejects_forged_source_hash_and_duplicate_output_ownership(
    tmp_path: Path,
) -> None:
    with pytest.raises(StoreError, match="source hash does not match"):
        _observation_case(
            tmp_path / "forged-source-hash",
            predicate="timing.wns_ns", observation_stage="post_route",
            run_stage="post_route", artifact_kind="amd.vivado.timing_summary",
            authority=AuthorityClass.TOOL_OBSERVATION,
            forge_source_hash=True,
        )
    with pytest.raises(ValueError, match="outputs must have unique paths"):
        _observation_case(
            tmp_path / "duplicate-ownership",
            predicate="timing.wns_ns", observation_stage="post_route",
            run_stage="post_route", artifact_kind="amd.vivado.timing_summary",
            authority=AuthorityClass.TOOL_OBSERVATION,
            duplicate_run_ownership=True,
        )


def test_store_replays_parser_and_rejects_self_signed_false_report_value(
    tmp_path: Path,
) -> None:
    with pytest.raises(StoreError, match="parser replay failed"):
        _observation_case(
            tmp_path / "self-signed-value",
            predicate="timing.wns_ns", observation_stage="post_route",
            run_stage="post_route", artifact_kind="amd.vivado.timing_summary",
            authority=AuthorityClass.TOOL_OBSERVATION,
            forge_report_value=True,
        )

    bundle, snapshot, graph, observation, artifact, _path, _tool = (
        _observation_case(
            tmp_path / "retrieval-self-signed-value",
            predicate="timing.wns_ns", observation_stage="post_route",
            run_stage="post_route", artifact_kind="amd.vivado.timing_summary",
            authority=AuthorityClass.TOOL_OBSERVATION,
        )
    )
    forged = replace(
        observation, id="", value=999,
        source=_observation_source_commitment(
            artifact=artifact,
            parser_name=observation.source.parser_name,
            parser_version=observation.source.parser_version,
            predicate=observation.predicate, value=999, unit=observation.unit,
        ),
    )
    with sqlite3.connect(bundle.store.path) as connection:
        connection.execute(
            "UPDATE observations SET payload_json=? WHERE id=?",
            (json.dumps(json_ready(forged), sort_keys=True), observation.id),
        )
    contexts = HybridRetriever(bundle, snapshot.id)._binding_target_contexts(
        graph, set(graph.entities),
    )[("predicate", "timing.wns_ns")]
    assert all("observation_evidence_qualified" not in item for item in contexts)


def test_manifest_rejects_duplicate_declared_output_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicate paths"):
        _observation_case(
            tmp_path / "duplicate-declaration",
            predicate="timing.wns_ns", observation_stage="post_route",
            run_stage="post_route", artifact_kind="amd.vivado.timing_summary",
            authority=AuthorityClass.TOOL_OBSERVATION,
            duplicate_declared_path=True,
        )


@pytest.mark.parametrize(
    ("predicate", "observation_stage", "run_stage", "artifact_kind"),
    [
        ("csim.exit_code", "csim", "csim", "amd.vitis.csim_result"),
        ("cosim.status", "cosim", "rtl_cosim", "amd.vitis.cosim_rpt"),
        (
            "profile.fifo_max_occupancy", "cosim", "rtl_cosim",
            "amd.vitis.dataflow_profile",
        ),
    ],
)
def test_dynamic_observation_binding_closes_exact_workload_and_testcase(
    tmp_path: Path,
    predicate: str,
    observation_stage: str,
    run_stage: str,
    artifact_kind: str,
) -> None:
    bundle, snapshot, graph, observation, artifact, _path, tool = _observation_case(
        tmp_path / predicate.replace(".", "-"), predicate=predicate,
        observation_stage=observation_stage, run_stage=run_stage,
        artifact_kind=artifact_kind,
        authority=AuthorityClass.VERIFICATION_EVIDENCE,
        workload_id="tb.default", testcase_id="case.default",
    )
    contexts = HybridRetriever(bundle, snapshot.id)._binding_target_contexts(
        graph, set(graph.entities),
    )[("predicate", predicate)]
    context = next(
        item for item in contexts
        if item.get("observation_instance_id") == {observation.id.casefold()}
    )
    assert context["observation_evidence_qualified"] == {_TOKEN}
    assert context["snapshot_association"] == {"verified"}
    assert context["workload_id"] == {"tb.default"}
    assert context["testcase_id"] == {"case.default"}
    assert context["observation_artifact_identity"] == {stable_hash([{
        "artifact_id": artifact.id, "sha256": artifact.sha256,
    }])}
    assert context["observation_run_identity"]

    binding = next(
        item for item in bundle.store.knowledge_bindings()
        if item.target_kind == "predicate" and item.target == predicate
        and item.required_context.get("tool") == tool
    )
    assert HybridRetriever._binding_constraints_match_values(
        binding, context, {"predicate": {predicate}},
    )


@pytest.mark.parametrize(
    ("case_name", "authority", "run_workload", "run_testcase", "inject"),
    [
        (
            "metadata-injection", AuthorityClass.SYNTHETIC,
            "tb.default", "case.default", True,
        ),
        (
            "workload-mismatch", AuthorityClass.VERIFICATION_EVIDENCE,
            "tb.other", "case.default", False,
        ),
        (
            "testcase-mismatch", AuthorityClass.VERIFICATION_EVIDENCE,
            "tb.default", "case.other", False,
        ),
    ],
)
def test_dynamic_observation_marker_rejects_injection_or_scope_mismatch(
    tmp_path: Path,
    case_name: str,
    authority: AuthorityClass,
    run_workload: str,
    run_testcase: str,
    inject: bool,
) -> None:
    bundle, snapshot, graph, _observation, _artifact, _path, _tool = (
        _observation_case(
            tmp_path / case_name, predicate="cosim.status",
            observation_stage="cosim", run_stage="rtl_cosim",
            artifact_kind="amd.vitis.cosim_rpt", authority=authority,
            workload_id="tb.default", run_workload_id=run_workload,
            testcase_id="case.default", run_testcase_id=run_testcase,
            injected_metadata=inject,
        )
    )
    contexts = HybridRetriever(bundle, snapshot.id)._binding_target_contexts(
        graph, set(graph.entities),
    )[("predicate", "cosim.status")]
    assert contexts
    assert all("observation_evidence_qualified" not in item for item in contexts)
    assert all("observation_instance_id" not in item for item in contexts)


def test_dynamic_observation_marker_rejects_stale_run_and_tampered_bytes(
    tmp_path: Path,
) -> None:
    stale = _observation_case(
        tmp_path / "stale", predicate="csim.exit_code",
        observation_stage="csim", run_stage="csim",
        artifact_kind="amd.vitis.csim_result",
        authority=AuthorityClass.VERIFICATION_EVIDENCE,
        workload_id="tb.default", testcase_id="case.default",
    )
    bundle, snapshot, graph, observation, _artifact, _path, _tool = stale
    assert any(
        item.get("observation_instance_id") == {observation.id.casefold()}
        for item in HybridRetriever(bundle, snapshot.id)._binding_target_contexts(
            graph, set(graph.entities),
        )[("predicate", "csim.exit_code")]
    )
    run = bundle.store.runs(snapshot.id)[0]
    run.metadata["fresh_tool_truth"] = False
    with sqlite3.connect(bundle.store.path) as connection:
        connection.execute(
            "UPDATE runs SET payload_json=? WHERE id=?",
            (json.dumps(json_ready(run), sort_keys=True), run.id),
        )
    assert not any(
        item.get("observation_evidence_qualified") == {_TOKEN}
        for item in HybridRetriever(bundle, snapshot.id)._binding_target_contexts(
            graph, set(graph.entities),
        )[("predicate", "csim.exit_code")]
    )

    tampered = _observation_case(
        tmp_path / "dynamic-tamper", predicate="profile.fifo_max_occupancy",
        observation_stage="cosim", run_stage="rtl_cosim",
        artifact_kind="amd.vitis.dataflow_profile",
        authority=AuthorityClass.VERIFICATION_EVIDENCE,
        workload_id="tb.default", testcase_id="case.default",
    )
    bundle, snapshot, graph, observation, _artifact, artifact_path, _tool = tampered
    assert any(
        item.get("observation_instance_id") == {observation.id.casefold()}
        for item in HybridRetriever(bundle, snapshot.id)._binding_target_contexts(
            graph, set(graph.entities),
        )[("predicate", "profile.fifo_max_occupancy")]
    )
    artifact_path.write_text("tampered dynamic report\n", encoding="utf-8")
    assert not any(
        item.get("observation_evidence_qualified") == {_TOKEN}
        for item in HybridRetriever(bundle, snapshot.id)._binding_target_contexts(
            graph, set(graph.entities),
        )[("predicate", "profile.fifo_max_occupancy")]
    )


def test_requested_directive_marker_cannot_be_injected_without_source_record(
    tmp_path: Path,
) -> None:
    (tmp_path / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    manifest = minimal_manifest(
        "test.directive.requested_evidence", "directive evidence", "dut",
        "kernel.cpp",
    )
    manifest.toolchains = [ToolchainContext(
        id="amd.vitis_hls.2024_2", vendor="amd", name="vitis_hls",
        version="2024.2",
    )]
    bundle = GraphBundle.create(tmp_path, manifest)
    install_reviewed_builtin_packs(bundle)
    snapshot = bundle.snapshot()
    source = next(
        item for item in bundle.store.artifacts(snapshot.id)
        if item.uri == "kernel.cpp"
    )
    anchor = SourceAnchor(
        artifact_id=source.id, start_line=1, start_column=1,
        end_line=1, end_column=14,
    )
    kernel = Entity(
        "hls.kernel", "dut", snapshot.id, qualified_name="dut", stage="ast",
        anchors=[anchor],
    )
    directive = Entity(
        "hls.directive", "PIPELINE", snapshot.id,
        qualified_name="kernel.cpp:1:PIPELINE", stage="source",
        authority=AuthorityClass.DECLARED_CONSTRAINT,
        attrs={
            "directive_kind": "PIPELINE", "options": {"ii": 1},
            "state": "selected_declared",
        },
        anchors=[anchor],
    )
    bind_directive_identity(
        directive, kernel, scope_resolution="source_ast",
    )
    graph = CanonicalGraph(snapshot.id)
    graph.add_entity(kernel)
    graph.add_entity(directive)
    graph.add_relation(Relation(
        directive.id, kernel.id, "hls.annotates", snapshot.id,
        authority=AuthorityClass.DECLARED_CONSTRAINT, stage="source",
        attrs={
            "scope_node_id": kernel.id, "scope_resolution": "source_ast",
        },
        anchors=[anchor],
    ))
    bundle.store.save_graph(graph)
    observation = Observation(
        snapshot.id, directive.id, "directive.tool_status", "applied",
        "schedule", AuthorityClass.TOOL_OBSERVATION,
        artifact_id=source.id,
        metadata={
            **directive_identity_metadata(directive),
            "directive_kind": "PIPELINE", "tool": "vitis_hls",
            "tool_version": "2024.2", "requested_directive_present": True,
        },
    )
    bundle.store.add_observations([observation])

    contexts = HybridRetriever(bundle, snapshot.id)._binding_target_contexts(
        graph, set(graph.entities),
    )[("predicate", "directive.tool_status")]
    context = next(
        item for item in contexts
        if item.get("directive_instance_id") == {directive.id.casefold()}
    )
    assert "requested_directive_present" not in context
    binding = next(
        item for item in bundle.store.knowledge_bindings()
        if item.target_kind == "predicate"
        and item.target == "directive.tool_status"
    )
    assert not HybridRetriever._binding_constraints_match_values(
        binding, context, {"predicate": {"directive.tool_status"}},
    )


def _source_directive_case(
    root: Path, *, duplicate_request: bool = False,
    with_tool_report: bool = False,
):
    root.mkdir()
    (root / "kernel.cpp").write_text(
        "void dut() {\n#pragma HLS pipeline II=1\n}\n", encoding="utf-8",
    )
    manifest = minimal_manifest(
        f"test.directive.source.{stable_hash(root.name)[:12]}",
        "directive source closure", "dut", "kernel.cpp",
    )
    manifest.toolchains = [ToolchainContext(
        id="amd.vitis_hls.2024_2", vendor="amd", name="vitis_hls",
        version="2024.2", environment_hash="e" * 64,
    )]
    if with_tool_report:
        manifest.stage_commands = {"csynth": ["vitis_hls", "--csynth"]}
        manifest.stage_outputs = {"csynth": [ToolOutputSpec(
            path="declared/directives.json",
            kind="amd.vitis.directive_status",
        )]}
    bundle = GraphBundle.create(root, manifest)
    install_reviewed_builtin_packs(bundle)
    indexed = Project(bundle).index()
    assert indexed.success
    snapshot = bundle.store.snapshot(indexed.snapshot_id)
    source = next(
        item for item in bundle.store.artifacts(snapshot.id)
        if item.uri == "kernel.cpp"
    )
    graph = bundle.store.load_graph(snapshot.id)
    directive = next(
        item for item in graph.entities.values()
        if item.kind == "hls.directive" and item.name == "PIPELINE"
    )
    requested = next(
        item for item in bundle.store.observations(snapshot.id)
        if item.subject_id == directive.id
        and item.predicate == "directive.requested"
    )
    if duplicate_request:
        bundle.store.add_observations([Observation(
            snapshot.id, directive.id, "directive.requested", {"ii": 1},
            "source", AuthorityClass.DECLARED_CONSTRAINT,
            artifact_id=source.id, anchor=requested.anchor,
            metadata={**requested.metadata, "duplicate_fixture": "second"},
        )])
    report_path = None
    if with_tool_report:
        toolchain = manifest.toolchain_for_stage("csynth")
        run = ToolRun(
            snapshot.id, "csynth", "runner.local", stable_hash({
                "fixture": "directive status", "directive": directive.id,
            }),
            toolchain_id=toolchain.id, status=RunStatus.SUCCEEDED,
            command=list(manifest.stage_commands["csynth"]),
            working_directory=".", environment_hash=toolchain.environment_hash,
            input_artifact_ids=[
                item.id for item in bundle.store.artifacts(snapshot.id)
                if item.producer_run_id is None
            ],
            exit_code=0,
            metadata={
                "authority": "tool_observation", "tool_truth": True,
                "fresh_execution": True, "fresh_tool_truth": True,
            },
        )
        raw_report = root / "directive-status.json"
        raw_report.write_bytes(_typed_report_bytes(
            "amd.vitis.directive_status", activity_source=None,
        ))
        report, report_path, _created = bundle.prepare_managed_artifact(
            raw_report, kind="amd.vitis.directive_status",
            role="tool_output", producer_run_id=run.id,
            metadata={
                "declared_output_path": "declared/directives.json",
                "stage": "schedule",
            },
        )
        parsed = VitisReportExtractor().extract(ExtractionContext(
            project_root=root, manifest=manifest, snapshot=snapshot,
            artifacts={report.id: report}, options={"existing_graph": graph},
        ))
        matches = [item for item in parsed.observations
                   if item.predicate == "directive.tool_status"]
        assert len(matches) == 1
        tool_observation = replace(matches[0], id="", run_id=run.id)
        run.output_artifact_ids = [report.id]
        commit_attested(bundle,
            run=run, artifacts=[report], observations=[tool_observation],
        )
    return bundle, snapshot, graph, directive, source, report_path


def test_directive_kind_requires_unique_live_source_declaration(
    tmp_path: Path,
) -> None:
    bundle, snapshot, graph, directive, _source, _report = _source_directive_case(
        tmp_path / "qualified",
    )
    context = HybridRetriever(bundle, snapshot.id)._binding_target_contexts(
        graph, set(graph.entities),
    )[("directive_kind", "PIPELINE")][0]
    assert context["directive_source_declaration_qualified"] == {
        "derived_from_current_directive_source_declaration_v1",
    }
    assert context["directive_source_identity"]
    binding = next(
        item for item in bundle.store.knowledge_bindings()
        if item.target_kind == "directive_kind" and item.target == "PIPELINE"
        and item.required_context.get("function_id") == {"required": True}
    )
    assert HybridRetriever._binding_constraints_match_values(
        binding, context, {"directive_kind": {"PIPELINE"}},
    )

    # Reserved metadata is not an attestation source.
    directive.attrs.update({
        "directive_source_declaration_qualified": (
            "derived_from_current_directive_source_declaration_v1"
        ),
        "directive_source_identity": "f" * 64,
    })
    (tmp_path / "qualified" / "kernel.cpp").write_text(
        "void dut() {}\n", encoding="utf-8",
    )
    changed = HybridRetriever(bundle, snapshot.id)._binding_target_contexts(
        graph, set(graph.entities),
    )[("directive_kind", "PIPELINE")][0]
    assert "directive_source_declaration_qualified" not in changed
    assert "directive_source_identity" not in changed
    assert not HybridRetriever._binding_constraints_match_values(
        binding, changed, {"directive_kind": {"PIPELINE"}},
    )


def test_duplicate_directive_request_fails_closed(tmp_path: Path) -> None:
    bundle, snapshot, graph, _directive, _source, _report = _source_directive_case(
        tmp_path / "duplicate", duplicate_request=True,
    )
    context = HybridRetriever(bundle, snapshot.id)._binding_target_contexts(
        graph, set(graph.entities),
    )[("directive_kind", "PIPELINE")][0]
    assert "directive_source_declaration_qualified" not in context


def test_schedule_directive_status_requires_real_declared_report_closure(
    tmp_path: Path,
) -> None:
    bundle, snapshot, graph, directive, _source, report_path = (
        _source_directive_case(tmp_path / "tool", with_tool_report=True)
    )
    contexts = HybridRetriever(bundle, snapshot.id)._binding_target_contexts(
        graph, set(graph.entities),
    )[("predicate", "directive.tool_status")]
    context = next(
        item for item in contexts
        if item.get("directive_instance_id") == {directive.id.casefold()}
    )
    assert context["requested_directive_present"] == {"true"}
    assert context["observation_evidence_qualified"] == {
        "derived_from_typed_observation_evidence_v1",
    }
    assert context["observation_artifact_kind"] == {
        "amd.vitis.directive_status",
    }
    binding = next(
        item for item in bundle.store.knowledge_bindings()
        if item.target_kind == "predicate" and item.target == "directive.tool_status"
    )
    assert HybridRetriever._binding_constraints_match_values(
        binding, context, {"predicate": {"directive.tool_status"}},
    )

    assert report_path is not None
    report_path.write_bytes(b"tampered\n")
    changed = HybridRetriever(bundle, snapshot.id)._binding_target_contexts(
        graph, set(graph.entities),
    )[("predicate", "directive.tool_status")]
    changed_context = next(
        item for item in changed
        if item.get("directive_instance_id") == {directive.id.casefold()}
    )
    assert "observation_evidence_qualified" not in changed_context
    assert not HybridRetriever._binding_constraints_match_values(
        binding, changed_context, {"predicate": {"directive.tool_status"}},
    )


@pytest.mark.parametrize(
    ("predicate", "observation_stage", "run_stage", "artifact_kind", "authority"),
    [
        ("qor.latency_cycles", "schedule", "csynth",
         "amd.vitis.csynth_xml", AuthorityClass.TOOL_OBSERVATION),
        ("schedule.start_cycle", "schedule", "csynth",
         "amd.vitis.schedule_json", AuthorityClass.COMPILER_DECISION),
        ("timing.wns_ns", "post_route", "post_route",
         "amd.vivado.timing_summary", AuthorityClass.TOOL_OBSERVATION),
        ("timing.wns_ns", "post_route", "post_route",
         "amd.vivado.post_route_timing", AuthorityClass.TOOL_OBSERVATION),
        ("resource.lut", "post_route", "post_route",
         "amd.vivado.utilization", AuthorityClass.TOOL_OBSERVATION),
        ("resource.lut", "post_route", "post_route",
         "amd.vivado.post_route_utilization", AuthorityClass.TOOL_OBSERVATION),
    ],
)
def test_tool_artifact_binding_requires_declared_live_run_output(
    tmp_path: Path, predicate: str, observation_stage: str, run_stage: str,
    artifact_kind: str, authority: AuthorityClass,
) -> None:
    bundle, snapshot, graph, _observation, artifact, path, _tool = (
        _observation_case(
            tmp_path / artifact_kind.rsplit(".", 1)[-1],
            predicate=predicate, observation_stage=observation_stage,
            run_stage=run_stage, artifact_kind=artifact_kind,
            authority=authority,
        )
    )
    contexts = HybridRetriever(bundle, snapshot.id)._binding_target_contexts(
        graph, set(graph.entities),
    )[("artifact_kind", artifact_kind)]
    context = next(
        item for item in contexts
        if "tool_artifact_evidence_qualified" in item
    )
    assert context["tool_artifact_evidence_qualified"] == {
        "derived_from_declared_live_tool_output_v1",
    }
    assert context["tool_artifact_identity"]
    assert context["tool_artifact_run_identity"]
    bindings = [
        item for item in bundle.store.knowledge_bindings()
        if item.target_kind == "artifact_kind" and item.target == artifact_kind
    ]
    if artifact_kind == "amd.vitis.csynth_xml":
        assert bindings == []
        path.write_bytes(b"tampered report bytes\n")
        changed = HybridRetriever(bundle, snapshot.id)._binding_target_contexts(
            graph, set(graph.entities),
        )[("artifact_kind", artifact.kind)]
        assert all(
            "tool_artifact_evidence_qualified" not in item for item in changed
        )
        return
    binding = next(iter(bindings))
    assert HybridRetriever._binding_constraints_match_values(
        binding, context, {"artifact_kind": {artifact_kind}},
    )

    path.write_bytes(b"tampered report bytes\n")
    changed = HybridRetriever(bundle, snapshot.id)._binding_target_contexts(
        graph, set(graph.entities),
    )[("artifact_kind", artifact.kind)]
    assert all("tool_artifact_evidence_qualified" not in item for item in changed)
    assert not any(HybridRetriever._binding_constraints_match_values(
        binding, item, {"artifact_kind": {artifact_kind}},
    ) for item in changed)
