"""Vitis HLS report adapters producing stage-scoped observations."""
from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from ..graph import CanonicalGraph
from ..manifest import project_path
from ..model import (
    AuthorityClass,
    Completeness,
    Diagnostic,
    DiagnosticSeverity,
    Entity,
    GateStatus,
    Observation,
    Relation,
    SourceAnchor,
    Stage,
    VerificationKind,
    VerificationResult,
    safe_relative_path,
)
from .base import ExtractionContext, ExtractionResult


MAX_REPORT_BYTES = 32 * 1024 * 1024
_SCHEDULE_OBSERVATION_FIELDS = frozenset({
    "start_cycle", "end_cycle", "pipeline_stage", "latency", "achieved_ii", "target_ii",
})
_SCHEDULE_NUMERIC_ATTR_FIELDS = frozenset({"bitwidth", "bank", "port"})
_SCHEDULE_NUMERIC_FIELDS = _SCHEDULE_OBSERVATION_FIELDS | _SCHEDULE_NUMERIC_ATTR_FIELDS
_SCHEDULE_IDENTIFIER_FIELDS = frozenset({
    "binding", "binding_kind", "functional_unit", "resource_class",
    "resource_instance", "memory", "implementation", "recurrence_id",
})
_SCHEDULE_CONTROL_FIELDS = frozenset({
    "name", "architecture_name", "source_location", "limiting_factor",
})
_SAFE_SCHEDULE_IDENTIFIER = re.compile(r"^[A-Za-z0-9_.$@#:/<>+\-\[\]]{1,256}$")
_SOURCE_LOCATION = re.compile(
    r"^(?P<path>[^:\r\n]{1,384}):(?P<line>[1-9][0-9]*)(?::(?P<column>[1-9][0-9]*))?$"
)


def _safe_source_location(value: Any) -> str | None:
    """Accept only project-relative ``path:line[:column]`` locations."""
    candidate = str(value)
    match = _SOURCE_LOCATION.fullmatch(candidate)
    if not match:
        return None
    try:
        relative = safe_relative_path(match.group("path"), "schedule source location")
    except ValueError:
        return None
    suffix = f":{match.group('line')}"
    if match.group("column"):
        suffix += f":{match.group('column')}"
    return relative + suffix


def _number(value: str | None) -> int | float | None:
    if value is None:
        return None
    value = value.strip()
    if not value or value.upper() in {"N/A", "NA", "?"}:
        return None
    try:
        return int(value.replace(",", ""))
    except ValueError:
        try:
            return float(value.replace(",", ""))
        except ValueError:
            return None


def _text(root: ET.Element, path: str) -> str | None:
    element = root.find(path)
    return element.text.strip() if element is not None and element.text else None


def _read_limited(path: Path) -> bytes:
    size = path.stat().st_size
    if size > MAX_REPORT_BYTES:
        raise ValueError(f"report exceeds {MAX_REPORT_BYTES} bytes: {path.name}")
    return path.read_bytes()


def _kernel(graph: CanonicalGraph, top: str | None) -> Entity | None:
    candidates = [item for item in graph.entities.values()
                  if item.kind == "hls.kernel" and (not top or item.name == top)]
    return candidates[0] if len(candidates) == 1 else None


class VitisReportExtractor:
    name = "amd.vitis.reports"
    version = "1"

    def supports(self, context: ExtractionContext) -> bool:
        return any(item.kind.startswith("amd.vitis.") for item in context.artifacts.values())

    def extract(self, context: ExtractionContext) -> ExtractionResult:
        graph = CanonicalGraph(snapshot_id=context.snapshot.id)
        result = ExtractionResult(graph=graph,
                                  capabilities=["amd.vitis.csynth", "amd.vitis.cosim",
                                                "amd.vitis.csim",
                                                "amd.vitis.schedule", "amd.vitis.dataflow_profile",
                                                "amd.vitis.directive_status"])
        existing: CanonicalGraph = context.options.get("existing_graph") or graph
        for artifact in sorted(context.artifacts.values(), key=lambda item: item.id):
            try:
                if artifact.kind == "amd.vitis.csynth_xml":
                    self._csynth(context, existing, graph, result, artifact)
                elif artifact.kind == "amd.vitis.csim_result":
                    self._csim(context, existing, result, artifact)
                elif artifact.kind in {"amd.vitis.cosim_rpt", "amd.vitis.cosim_report"}:
                    self._cosim(context, existing, graph, result, artifact)
                elif artifact.kind == "amd.vitis.schedule_json":
                    self._schedule(context, existing, graph, result, artifact)
                elif artifact.kind == "amd.vitis.dataflow_profile":
                    self._dataflow_profile(context, existing, result, artifact)
                elif artifact.kind == "amd.vitis.directive_status":
                    self._directive_status(context, existing, result, artifact)
            except Exception as exc:
                result.diagnostics.append(Diagnostic(
                    snapshot_id=context.snapshot.id, code="vitis.report_parse_error",
                    severity=DiagnosticSeverity.ERROR,
                    message=f"failed to parse {artifact.uri}: {type(exc).__name__}: {exc}",
                    stage=Stage.UNKNOWN.value, artifact_id=artifact.id,
                    metadata={"artifact_kind": artifact.kind},
                ))
        result.coverage = {
            "observations": len(result.observations),
            "verifications": len(result.verifications),
            "diagnostics": len(result.diagnostics),
        }
        return result

    @staticmethod
    def _subject_or_artifact(existing: CanonicalGraph, top: str | None, artifact_id: str) -> str:
        item = _kernel(existing, top)
        return item.id if item else artifact_id

    def _csim(self, context: ExtractionContext, existing: CanonicalGraph,
              result: ExtractionResult, artifact: Any) -> None:
        data = json.loads(_read_limited(project_path(context.project_root, artifact.uri)))
        if data.get("schema_version") != "hlsgraph.vitis.csim.v1":
            raise ValueError("unsupported normalized C simulation schema")
        status = str(data.get("status", "unknown")).lower()
        if status not in {"pass", "fail"}:
            raise ValueError("C simulation status must be pass or fail")
        workload = data.get("workload_id") or artifact.metadata.get("workload_id")
        subject = self._subject_or_artifact(existing, context.manifest.build.top, artifact.id)
        values = {
            "csim.exit_code": int(data.get("exit_code", 0 if status == "pass" else 1)),
            "csim.mismatches": int(data.get("mismatches", 0)),
            "csim.assertions_failed": int(data.get("assertions_failed", 0)),
        }
        evidence: list[str] = []
        for predicate, value in values.items():
            observation = Observation(
                snapshot_id=context.snapshot.id, subject_id=subject,
                predicate=predicate, value=value, unit="count",
                stage=Stage.CSIM.value,
                authority=context.authority_for(artifact, AuthorityClass.VERIFICATION_EVIDENCE),
                artifact_id=artifact.id, workload_id=workload,
                completeness=Completeness.COMPLETE if workload else Completeness.PARTIAL,
                metadata={"dynamic": True, "testbench_scoped": True},
            )
            result.observations.append(observation)
            evidence.append(observation.id)
        passed = status == "pass" and values["csim.exit_code"] == 0 and not (
            values["csim.mismatches"] or values["csim.assertions_failed"]
        )
        result.verifications.append(VerificationResult(
            snapshot_id=context.snapshot.id, kind=VerificationKind.CSIM,
            status=GateStatus.PASS if passed else GateStatus.FAIL,
            workload_id=workload, evidence_ids=evidence,
            details={"reported_status": status, "artifact_id": artifact.id,
                     "fixture_authority": artifact.metadata.get("fixture_authority")},
        ))
        if values["csim.mismatches"]:
            result.verifications.append(VerificationResult(
                snapshot_id=context.snapshot.id, kind=VerificationKind.MISMATCH,
                status=GateStatus.FAIL, workload_id=workload, evidence_ids=evidence,
                details={"count": values["csim.mismatches"], "artifact_id": artifact.id,
                         "fixture_authority": artifact.metadata.get("fixture_authority")},
            ))
        if data.get("deadlock") is True:
            result.verifications.append(VerificationResult(
                snapshot_id=context.snapshot.id, kind=VerificationKind.DEADLOCK,
                status=GateStatus.FAIL, workload_id=workload, evidence_ids=evidence,
                details={"artifact_id": artifact.id,
                         "fixture_authority": artifact.metadata.get("fixture_authority")},
            ))

    def _csynth(self, context: ExtractionContext, existing: CanonicalGraph,
                 graph: CanonicalGraph, result: ExtractionResult, artifact: Any) -> None:
        path = project_path(context.project_root, artifact.uri)
        root = ET.fromstring(_read_limited(path))
        top = _text(root, ".//UserAssignments/TopModelName") or context.manifest.build.top
        subject = self._subject_or_artifact(existing, top, artifact.id)
        anchor = SourceAnchor(artifact_id=artifact.id, ir_location="csynth.xml")
        metrics = [
            ("clock.requested_period_ns", _number(_text(root, ".//UserAssignments/TargetClockPeriod")), "ns"),
            ("clock.estimated_period_ns", _number(_text(root, ".//SummaryOfTimingAnalysis/EstimatedClockPeriod")), "ns"),
            ("qor.latency_best_cycles", _number(_text(root, ".//SummaryOfOverallLatency/Best-caseLatency")), "cycle"),
            ("qor.latency_worst_cycles", _number(_text(root, ".//SummaryOfOverallLatency/Worst-caseLatency")), "cycle"),
            ("qor.interval_min_cycles", _number(_text(root, ".//SummaryOfOverallLatency/Interval-min")), "cycle"),
            ("qor.interval_max_cycles", _number(_text(root, ".//SummaryOfOverallLatency/Interval-max")), "cycle"),
        ]
        resources = root.find(".//AreaEstimates/Resources")
        if resources is not None:
            for child in resources:
                metrics.append((f"resource.{child.tag.lower()}", _number(child.text), "count"))
        available = root.find(".//AreaEstimates/AvailableResources")
        if available is not None:
            for child in available:
                metrics.append((f"resource.available_{child.tag.lower()}", _number(child.text), "count"))
        for predicate, value, unit in metrics:
            if value is None:
                continue
            result.observations.append(Observation(
                snapshot_id=context.snapshot.id, subject_id=subject, predicate=predicate,
                value=value, unit=unit, stage=Stage.SCHEDULE.value,
                authority=context.authority_for(artifact, AuthorityClass.TOOL_OBSERVATION), artifact_id=artifact.id,
                anchor=anchor, metadata={"source": "hls_estimate", "tool": "vitis_hls",
                                         "report_scope": top},
            ))
        loops = root.find(".//SummaryOfLoopLatency")
        if loops is not None:
            for loop_element in loops:
                name = loop_element.tag
                matches = [item for item in existing.entities.values()
                           if item.kind == "hls.loop" and item.name == name]
                loop_subject = matches[0].id if len(matches) == 1 else artifact.id
                if len(matches) != 1:
                    result.diagnostics.append(Diagnostic(
                        snapshot_id=context.snapshot.id,
                        code="mapping.unresolved_report_scope" if not matches else "mapping.ambiguous_report_scope",
                        severity=DiagnosticSeverity.WARNING,
                        message=f"csynth loop scope {name!r} mapped to {len(matches)} AST loops; observations remain on artifact",
                        stage=Stage.SCHEDULE.value, artifact_id=artifact.id,
                        metadata={"report_scope": name, "candidate_ids": [item.id for item in matches]},
                    ))
                loop_metrics = {
                    "qor.trip_count": ("TripCount", "iteration"),
                    "qor.latency_cycles": ("Latency", "cycle"),
                    "qor.iteration_latency_cycles": ("IterationLatency", "cycle"),
                    "qor.achieved_ii": ("PipelineII", "cycle"),
                    "qor.pipeline_depth": ("PipelineDepth", "cycle"),
                }
                for predicate, (tag, unit) in loop_metrics.items():
                    value = _number(_text(loop_element, tag))
                    if value is None:
                        continue
                    result.observations.append(Observation(
                        snapshot_id=context.snapshot.id, subject_id=loop_subject,
                        predicate=predicate, value=value, unit=unit,
                        stage=Stage.SCHEDULE.value,
                        authority=context.authority_for(artifact, AuthorityClass.TOOL_OBSERVATION),
                        artifact_id=artifact.id, anchor=anchor,
                        metadata={"source": "hls_estimate", "report_scope": name,
                                  "mapping_status": "exact" if len(matches) == 1 else "unresolved"},
                    ))

    def _cosim(self, context: ExtractionContext, existing: CanonicalGraph,
               graph: CanonicalGraph, result: ExtractionResult, artifact: Any) -> None:
        path = project_path(context.project_root, artifact.uri)
        text = _read_limited(path).decode("utf-8", errors="replace")
        row = re.search(
            r"\|\s*(?:Verilog|VHDL|SystemC)\s*\|\s*(Pass|Fail)\s*\|\s*"
            r"(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|",
            text, re.I,
        )
        if not row:
            raise ValueError("no cosimulation result row found")
        status = row.group(1).lower()
        workload = artifact.metadata.get("workload_id")
        completeness = Completeness.COMPLETE if workload else Completeness.PARTIAL
        subject = self._subject_or_artifact(existing, context.manifest.build.top, artifact.id)
        result.observations.append(Observation(
            snapshot_id=context.snapshot.id, subject_id=subject,
            predicate="cosim.status", value=status, stage=Stage.COSIM.value,
            authority=context.authority_for(
                artifact, AuthorityClass.VERIFICATION_EVIDENCE,
            ),
            artifact_id=artifact.id, workload_id=workload,
            completeness=completeness,
            metadata={"dynamic": True, "testbench_scoped": True},
        ))
        values = [int(value) for value in row.groups()[1:]]
        for predicate, value in zip((
            "cosim.latency_min_cycles", "cosim.latency_avg_cycles", "cosim.latency_max_cycles",
            "cosim.interval_min_cycles", "cosim.interval_avg_cycles", "cosim.interval_max_cycles",
        ), values):
            result.observations.append(Observation(
                snapshot_id=context.snapshot.id, subject_id=subject, predicate=predicate,
                value=value, unit="cycle", stage=Stage.COSIM.value,
                authority=context.authority_for(artifact, AuthorityClass.VERIFICATION_EVIDENCE),
                artifact_id=artifact.id, workload_id=workload,
                completeness=completeness,
                metadata={"dynamic": True, "testbench_scoped": True},
            ))
        result.verifications.append(VerificationResult(
            snapshot_id=context.snapshot.id, kind=VerificationKind.RTL_COSIM,
            status=GateStatus.PASS if status == "pass" else GateStatus.FAIL,
            workload_id=workload, evidence_ids=[item.id for item in result.observations
                                                if item.artifact_id == artifact.id],
            details={"reported_status": status, "artifact_id": artifact.id,
                     "fixture_authority": artifact.metadata.get("fixture_authority")},
        ))
        if not workload:
            result.diagnostics.append(Diagnostic(
                snapshot_id=context.snapshot.id, code="verification.missing_workload",
                severity=DiagnosticSeverity.WARNING,
                message="cosimulation evidence has no workload/testbench identity",
                stage=Stage.COSIM.value, artifact_id=artifact.id,
            ))

    def _schedule(self, context: ExtractionContext, existing: CanonicalGraph,
                  graph: CanonicalGraph, result: ExtractionResult, artifact: Any) -> None:
        data = json.loads(_read_limited(project_path(context.project_root, artifact.uri)))
        if data.get("schema_version") != "hlsgraph.vitis.schedule.v1":
            raise ValueError("unsupported normalized schedule schema")
        parent = _kernel(existing, data.get("top"))
        for index, item in enumerate(data.get("operations", [])):
            if not isinstance(item, dict):
                raise ValueError(f"schedule operation {index} must be an object")
            name = str(item.get("name") or f"operation-{index}")
            if not _SAFE_SCHEDULE_IDENTIFIER.fullmatch(name):
                raise ValueError(f"schedule operation {index} has an invalid name")
            attrs: dict[str, Any] = {"plane": "evidence", "hot": False}
            for key in sorted(_SCHEDULE_NUMERIC_FIELDS & item.keys()):
                value = item[key]
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise ValueError(f"schedule operation {index} field {key!r} must be numeric")
                if key in _SCHEDULE_NUMERIC_ATTR_FIELDS:
                    attrs[key] = value
            for key in sorted(_SCHEDULE_IDENTIFIER_FIELDS & item.keys()):
                value = str(item[key])
                if not _SAFE_SCHEDULE_IDENTIFIER.fullmatch(value):
                    raise ValueError(
                        f"schedule operation {index} field {key!r} is not a safe identifier"
                    )
                attrs[key] = value
            if "limiting_factor" in item:
                factor = str(item["limiting_factor"]).casefold()
                if factor not in {"recurrence", "resource", "memory", "interface", "none", "unknown"}:
                    raise ValueError(f"schedule operation {index} has an invalid limiting_factor")
                attrs["limiting_factor"] = factor
            unknown = sorted(set(item) - _SCHEDULE_NUMERIC_FIELDS
                             - _SCHEDULE_IDENTIFIER_FIELDS - _SCHEDULE_CONTROL_FIELDS)
            if unknown:
                result.diagnostics.append(Diagnostic(
                    snapshot_id=context.snapshot.id, code="schedule.unknown_fields_ignored",
                    severity=DiagnosticSeverity.INFO,
                    message=(f"schedule operation {index} contains unsupported fields; "
                             "their values were not imported"),
                    stage=Stage.SCHEDULE.value, artifact_id=artifact.id,
                    metadata={"field_count": len(unknown), "operation_index": index},
                ))
            source_location = item.get("source_location")
            safe_location = None
            if source_location is not None:
                safe_location = _safe_source_location(source_location)
                if safe_location is None:
                    result.diagnostics.append(Diagnostic(
                        snapshot_id=context.snapshot.id, code="schedule.invalid_location_ignored",
                        severity=DiagnosticSeverity.WARNING,
                        message=f"schedule operation {index} has an invalid source location",
                        stage=Stage.SCHEDULE.value, artifact_id=artifact.id,
                        metadata={"operation_index": index},
                    ))
            operation = Entity(
                kind="hls.scheduled_operation", name=name,
                qualified_name=f"{artifact.uri}::{name}::{index}",
                snapshot_id=context.snapshot.id,
                authority=context.authority_for(artifact, AuthorityClass.COMPILER_DECISION), stage=Stage.SCHEDULE.value,
                attrs=attrs,
                anchors=[SourceAnchor(artifact_id=artifact.id,
                                      ir_location=safe_location)],
            )
            graph.add_entity(operation)
            architecture_target = None
            architecture_name = item.get("architecture_name")
            if architecture_name:
                architecture_name = str(architecture_name)
                if not _SAFE_SCHEDULE_IDENTIFIER.fullmatch(architecture_name):
                    raise ValueError(
                        f"schedule operation {index} has an invalid architecture_name"
                    )
                mapped = [entity for entity in existing.entities.values()
                          if entity.name == architecture_name and entity.kind.startswith("hls.")]
                if len(mapped) == 1:
                    architecture_target = mapped[0]
                    graph.add_relation(Relation(
                        src=operation.id, dst=architecture_target.id, kind="cross.maps_to",
                        snapshot_id=context.snapshot.id,
                        authority=context.authority_for(artifact, AuthorityClass.COMPILER_DECISION), stage=Stage.SCHEDULE.value,
                        mapping_kind="schedule.explicit_architecture_name",
                        attrs={"cardinality": "explicit"},
                    ), allow_dangling=True)
                else:
                    result.diagnostics.append(Diagnostic(
                        snapshot_id=context.snapshot.id, code="mapping.unresolved_schedule_target",
                        severity=DiagnosticSeverity.WARNING,
                        message=f"schedule architecture target {architecture_name!r} matched {len(mapped)} entities",
                        stage=Stage.SCHEDULE.value, subject_id=operation.id,
                        metadata={"candidate_ids": [entity.id for entity in mapped]},
                    ))
            if parent:
                graph.add_relation(Relation(
                    src=parent.id, dst=operation.id, kind="hls.contains",
                    snapshot_id=context.snapshot.id,
                    authority=context.authority_for(artifact, AuthorityClass.COMPILER_DECISION), stage=Stage.SCHEDULE.value,
                ), allow_dangling=True)
            for predicate, key, unit in (
                ("schedule.start_cycle", "start_cycle", "cycle"),
                ("schedule.end_cycle", "end_cycle", "cycle"),
                ("schedule.pipeline_stage", "pipeline_stage", "stage"),
                ("schedule.operation_latency", "latency", "cycle"),
                ("qor.achieved_ii", "achieved_ii", "cycle"),
                ("qor.target_ii", "target_ii", "cycle"),
            ):
                if key in item:
                    result.observations.append(Observation(
                        snapshot_id=context.snapshot.id,
                        subject_id=architecture_target.id if architecture_target else operation.id,
                        predicate=predicate, value=item[key], unit=unit,
                        stage=Stage.SCHEDULE.value,
                        authority=context.authority_for(artifact, AuthorityClass.COMPILER_DECISION),
                        artifact_id=artifact.id,
                        metadata={"schedule_operation_id": operation.id,
                                  "mapping_status": "explicit" if architecture_target else "operation_only"},
                    ))

    def _dataflow_profile(self, context: ExtractionContext, existing: CanonicalGraph,
                          result: ExtractionResult, artifact: Any) -> None:
        data = json.loads(_read_limited(project_path(context.project_root, artifact.uri)))
        if data.get("schema_version") != "hlsgraph.vitis.dataflow_profile.v1":
            raise ValueError("unsupported normalized dataflow profile schema")
        workload = data.get("workload_id") or artifact.metadata.get("workload_id")
        for channel in data.get("channels", []):
            name = str(channel.get("name", ""))
            matches = [item for item in existing.entities.values()
                       if item.kind in {"hls.stream", "hls.buffer"} and item.name == name]
            subject = matches[0].id if len(matches) == 1 else artifact.id
            for key, predicate, unit in (
                ("max_occupancy", "profile.fifo_max_occupancy", "token"),
                ("read_block_cycles", "profile.read_block_cycles", "cycle"),
                ("write_block_cycles", "profile.write_block_cycles", "cycle"),
                ("tokens", "profile.token_count", "token"),
            ):
                if key in channel:
                    result.observations.append(Observation(
                        snapshot_id=context.snapshot.id, subject_id=subject,
                        predicate=predicate, value=channel[key], unit=unit,
                        stage=Stage.COSIM.value,
                        authority=context.authority_for(artifact, AuthorityClass.VERIFICATION_EVIDENCE),
                        artifact_id=artifact.id, workload_id=workload,
                        completeness=Completeness.COMPLETE if workload else Completeness.PARTIAL,
                        metadata={"channel": name, "dynamic": True,
                                  "mapping_status": "exact" if len(matches) == 1 else "unresolved"},
                    ))

    def _directive_status(self, context: ExtractionContext, existing: CanonicalGraph,
                          result: ExtractionResult, artifact: Any) -> None:
        """Import a normalized, tool-produced directive application summary.

        The adapter deliberately keeps declared precedence separate from tool
        application.  A scope/name match is accepted only when it is unique;
        otherwise the record stays on the report artifact and is incomplete.
        """
        data = json.loads(_read_limited(project_path(context.project_root, artifact.uri)))
        if data.get("schema_version") != "hlsgraph.vitis.directive_status.v1":
            raise ValueError("unsupported normalized directive-status schema")
        allowed = {"applied", "ignored", "unmet", "rejected", "unknown"}
        annotations = [relation for relation in existing.relations.values()
                       if relation.kind == "hls.annotates"]
        for index, row in enumerate(data.get("directives", [])):
            kind = str(row.get("directive_kind") or "").upper()
            scope = str(row.get("scope") or "")
            status = str(row.get("status") or "unknown").lower()
            if not kind or status not in allowed:
                raise ValueError(f"invalid directive status at index {index}")
            scope_matches = [entity for entity in existing.entities.values()
                             if entity.id == scope or entity.name == scope
                             or entity.qualified_name == scope]
            target_ids = {entity.id for entity in scope_matches}
            candidates = [existing.entities[relation.src] for relation in annotations
                          if relation.dst in target_ids and relation.src in existing.entities
                          and existing.entities[relation.src].kind == "hls.directive"
                          and str(existing.entities[relation.src].attrs.get(
                              "directive_kind", existing.entities[relation.src].name)).upper() == kind]
            effective = [item for item in candidates
                         if item.attrs.get("state") == "effective_declared"]
            if len(effective) == 1:
                candidates = effective
            elif candidates:
                highest = max(int(item.attrs.get("precedence", 0)) for item in candidates)
                declared_winners = [item for item in candidates
                                    if int(item.attrs.get("precedence", 0)) == highest]
                if len(declared_winners) == 1:
                    candidates = declared_winners
            subject = candidates[0].id if len(candidates) == 1 else artifact.id
            completeness = Completeness.COMPLETE if len(candidates) == 1 else Completeness.AMBIGUOUS
            common = {
                "scope": scope, "directive_kind": kind,
                "mapping_status": "exact" if len(candidates) == 1 else "unresolved",
                "candidate_ids": [item.id for item in sorted(candidates, key=lambda item: item.id)],
                "tool": data.get("tool", "vitis_hls"),
                "tool_version": data.get("tool_version"),
            }
            result.observations.append(Observation(
                snapshot_id=context.snapshot.id, subject_id=subject,
                predicate="directive.tool_status", value=status,
                stage=Stage.SCHEDULE.value,
                authority=context.authority_for(artifact, AuthorityClass.TOOL_OBSERVATION),
                artifact_id=artifact.id, completeness=completeness, metadata=common,
            ))
            for key, predicate in (
                ("requested", "directive.reported_requested"),
                ("effective", "directive.tool_effective"),
                ("achieved", "directive.achieved"),
            ):
                if key in row:
                    result.observations.append(Observation(
                        snapshot_id=context.snapshot.id, subject_id=subject,
                        predicate=predicate, value=row[key],
                        stage=Stage.SCHEDULE.value,
                        authority=context.authority_for(artifact, AuthorityClass.TOOL_OBSERVATION),
                        artifact_id=artifact.id, completeness=completeness,
                        metadata=common,
                    ))
            if len(candidates) != 1:
                result.diagnostics.append(Diagnostic(
                    snapshot_id=context.snapshot.id,
                    code="mapping.unresolved_directive_status",
                    severity=DiagnosticSeverity.WARNING,
                    message=f"tool directive status for {kind} on {scope!r} did not map uniquely",
                    stage=Stage.SCHEDULE.value, artifact_id=artifact.id,
                    metadata=common,
                ))
            elif status in {"ignored", "unmet", "rejected"}:
                result.diagnostics.append(Diagnostic(
                    snapshot_id=context.snapshot.id,
                    code=f"directive.{status}", severity=DiagnosticSeverity.WARNING,
                    message=f"Vitis reported {kind} on {scope!r} as {status}",
                    stage=Stage.SCHEDULE.value, subject_id=subject,
                    artifact_id=artifact.id,
                    metadata={**common, "reason": row.get("reason")},
                ))
