"""Vivado implementation summary adapters with stage-correct timing/resource facts."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..graph import CanonicalGraph
from ..manifest import project_path
from ..model import (
    AuthorityClass,
    Derivation,
    Diagnostic,
    DiagnosticSeverity,
    Observation,
    SourceAnchor,
    Stage,
    _observation_source_commitment,
    stable_hash,
)
from .base import ExtractionContext, ExtractionResult


MAX_REPORT_BYTES = 64 * 1024 * 1024
_PARSER_NAME = "amd.vivado.reports"
_PARSER_VERSION = "1"
IMPLEMENTATION_STAGES = {
    Stage.POST_SYNTH.value, Stage.POST_PLACE.value, Stage.POST_ROUTE.value,
}
POST_ROUTE_KINDS = {
    "amd.vivado.post_route_timing", "amd.vivado.post_route_utilization",
}
ROUTED_DESIGN_KINDS = {"amd.vivado.routed_checkpoint"}
GENERIC_KINDS = {
    "amd.vivado.timing_summary", "amd.vivado.utilization",
    "amd.vivado.physical_summary", "amd.vivado.qor_summary",
}


def _read(path: Path) -> str:
    if path.stat().st_size > MAX_REPORT_BYTES:
        raise ValueError(f"report exceeds {MAX_REPORT_BYTES} bytes")
    return path.read_text(encoding="utf-8", errors="replace")


def _report_observation(artifact: Any, **values: Any) -> Observation:
    """Create one observation committed to this exact Vivado report."""

    artifact_id = values.setdefault("artifact_id", artifact.id)
    if artifact_id != artifact.id:
        raise ValueError("Vivado observation cannot cite a sibling report")
    values.setdefault(
        "anchor",
        SourceAnchor(artifact_id=artifact.id, ir_location=artifact.kind),
    )
    values["source"] = _observation_source_commitment(
        artifact=artifact,
        parser_name=_PARSER_NAME,
        parser_version=_PARSER_VERSION,
        predicate=values["predicate"],
        value=values["value"],
        unit=values.get("unit"),
    )
    return Observation(**values)


def _report_stage(artifact: Any) -> str:
    declared = artifact.metadata.get("stage")
    if artifact.kind in ROUTED_DESIGN_KINDS:
        if str(declared or "") != Stage.POST_ROUTE.value:
            raise ValueError(
                f"{artifact.kind} requires metadata.stage='post_route'"
            )
        return Stage.POST_ROUTE.value
    if artifact.kind in POST_ROUTE_KINDS:
        if declared is not None and str(declared) != Stage.POST_ROUTE.value:
            raise ValueError(
                f"{artifact.kind} cannot declare contradictory stage {declared!r}"
            )
        return Stage.POST_ROUTE.value
    if artifact.kind in GENERIC_KINDS:
        if declared is None:
            raise ValueError(
                f"generic report {artifact.kind} requires explicit artifact metadata.stage"
            )
        stage = str(declared)
        if stage not in IMPLEMENTATION_STAGES:
            raise ValueError(f"unsupported Vivado implementation stage {stage!r}")
        return stage
    raise ValueError(f"unsupported Vivado report kind {artifact.kind!r}")


def _report_subject(context: ExtractionContext, graph: CanonicalGraph, artifact: Any,
                    result: ExtractionResult, stage: str) -> tuple[str, bool]:
    """Resolve an explicitly scoped report; otherwise keep it artifact-scoped."""
    scope = artifact.metadata.get("scope")
    if scope is None:
        result.diagnostics.append(Diagnostic(
            snapshot_id=context.snapshot.id, code="vivado.report_scope_unbound",
            severity=DiagnosticSeverity.WARNING,
            message=("Vivado report has no explicit kernel/top/part scope; observations remain "
                     "artifact-scoped and cannot produce design verification gates"),
            stage=stage, artifact_id=artifact.id, subject_id=artifact.id,
        ))
        return artifact.id, False
    if not isinstance(scope, dict):
        raise ValueError("artifact metadata.scope must be an object")
    if scope.get("kind") != "kernel":
        raise ValueError("v0.1 Vivado design gates require scope.kind='kernel'")
    top = str(scope.get("top") or "")
    if top != context.manifest.build.top:
        raise ValueError(
            f"report top {top!r} does not match manifest top {context.manifest.build.top!r}"
        )
    instance = str(scope.get("instance") or "")
    if instance != top:
        raise ValueError(
            f"v0.1 kernel report instance {instance!r} must identify top instance {top!r}"
        )
    expected_part = context.manifest.target.part
    report_part = scope.get("part")
    if expected_part and str(report_part or "") != expected_part:
        raise ValueError(
            f"report part {report_part!r} does not match target part {expected_part!r}"
        )
    expected_platform = context.manifest.target.platform
    if expected_platform and str(scope.get("platform") or "") != expected_platform:
        raise ValueError(
            f"report platform {scope.get('platform')!r} does not match target platform "
            f"{expected_platform!r}"
        )
    if artifact.kind in {"amd.vivado.timing_summary", "amd.vivado.post_route_timing"}:
        clock = str(scope.get("clock") or "")
        known_clocks = {item.name for item in context.manifest.target.clocks}
        if clock != "all" and clock not in known_clocks:
            raise ValueError(
                f"timing report clock scope {clock!r} is neither 'all' nor a target clock"
            )
    items = [item for item in graph.entities.values()
             if item.kind == "hls.kernel" and item.name == top]
    if len(items) != 1:
        raise ValueError(f"report scope resolves to {len(items)} matching kernel entities")
    return items[0].id, True


class VivadoReportExtractor:
    name = _PARSER_NAME
    version = _PARSER_VERSION

    def supports(self, context: ExtractionContext) -> bool:
        return any(item.kind.startswith("amd.vivado.") for item in context.artifacts.values())

    def extract(self, context: ExtractionContext) -> ExtractionResult:
        graph = CanonicalGraph(snapshot_id=context.snapshot.id)
        result = ExtractionResult(graph=graph,
                                  capabilities=["amd.vivado.timing", "amd.vivado.utilization",
                                                "amd.vivado.physical_summary"])
        existing: CanonicalGraph = context.options.get("existing_graph") or graph
        resource_gate_artifacts: set[str] = set()
        for artifact in sorted(context.artifacts.values(), key=lambda item: item.id):
            if not artifact.kind.startswith("amd.vivado."):
                continue
            diagnostic_stage = (Stage.POST_ROUTE.value if artifact.kind in
                                (POST_ROUTE_KINDS | ROUTED_DESIGN_KINDS)
                                else str(artifact.metadata.get("stage") or Stage.UNKNOWN.value))
            try:
                stage = _report_stage(artifact)
                artifact_subject, gate_allowed = _report_subject(
                    context, existing, artifact, result, stage,
                )
                if artifact.kind in ROUTED_DESIGN_KINDS:
                    # A routed checkpoint remains a cold, content-addressed
                    # artifact.  Its bytes are not expanded into the canonical
                    # graph, but validated scope/stage identity can qualify
                    # post-route knowledge guidance through the ledger.
                    continue
                if artifact.kind in {"amd.vivado.timing_summary", "amd.vivado.post_route_timing"}:
                    self._timing(context, result, artifact, artifact_subject, stage,
                                 gate_allowed=gate_allowed)
                elif artifact.kind in {"amd.vivado.utilization", "amd.vivado.post_route_utilization"}:
                    self._utilization(context, result, artifact, artifact_subject, stage)
                    if stage == Stage.POST_ROUTE.value and gate_allowed:
                        resource_gate_artifacts.add(artifact.id)
                elif artifact.kind in {"amd.vivado.physical_summary", "amd.vivado.qor_summary"}:
                    self._physical_json(context, result, artifact, artifact_subject, stage)
            except Exception as exc:
                result.diagnostics.append(Diagnostic(
                    snapshot_id=context.snapshot.id, code="vivado.report_parse_error",
                    severity=DiagnosticSeverity.ERROR,
                    message=f"failed to parse {artifact.uri}: {type(exc).__name__}: {exc}",
                    stage=(diagnostic_stage if diagnostic_stage in IMPLEMENTATION_STAGES
                           else Stage.UNKNOWN.value), artifact_id=artifact.id,
                ))
        self._derive_resource_gate(context, result, resource_gate_artifacts)
        result.coverage = {"observations": len(result.observations),
                           "derivations": len(result.derivations)}
        return result

    @staticmethod
    def _timing(context: ExtractionContext, result: ExtractionResult,
                artifact: Any, subject: str, stage: str, *, gate_allowed: bool) -> None:
        text = _read(project_path(context.project_root, artifact.uri))
        # Supports standard timing summary tables and compact sanitized fixtures.
        wns = re.search(r"\bWNS(?:\(ns\))?\s*[:|]?\s*(-?\d+(?:\.\d+)?)", text, re.I)
        tns = re.search(r"\bTNS(?:\(ns\))?\s*[:|]?\s*(-?\d+(?:\.\d+)?)", text, re.I)
        if not wns:
            table = re.search(r"WNS\(ns\).*?\n\s*-+.*?\n\s*(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)",
                              text, re.I | re.S)
            if table:
                wns = table
                if not tns:
                    tns_value = float(table.group(2))
                else:
                    tns_value = float(tns.group(1))
            else:
                raise ValueError("WNS was not found")
        else:
            tns_value = float(tns.group(1)) if tns else None
        wns_value = float(wns.group(1))
        observations: list[Observation] = []
        observations.append(_report_observation(artifact,
            snapshot_id=context.snapshot.id, subject_id=subject,
            predicate="timing.wns_ns", value=wns_value, unit="ns",
            stage=stage,
            authority=context.authority_for(artifact, AuthorityClass.TOOL_OBSERVATION),
            artifact_id=artifact.id, metadata={"path_group": "all", "stage": stage},
        ))
        if tns_value is not None:
            observations.append(_report_observation(artifact,
                snapshot_id=context.snapshot.id, subject_id=subject,
                predicate="timing.tns_ns", value=tns_value, unit="ns",
                stage=stage,
                authority=context.authority_for(artifact, AuthorityClass.TOOL_OBSERVATION),
                artifact_id=artifact.id, metadata={"path_group": "all", "stage": stage},
            ))
        result.observations.extend(observations)
        if stage != Stage.POST_ROUTE.value or not gate_allowed:
            return
        result.derivations.append(Derivation(
            snapshot_id=context.snapshot.id, subject_id=subject,
            predicate="gate.post_route_timing", value=wns_value >= 0,
            algorithm="hlsgraph.gate.wns_nonnegative", algorithm_version="1",
            input_observation_ids=[item.id for item in observations],
            stage=Stage.POST_ROUTE.value,
            authority=context.authority_for(artifact, AuthorityClass.DERIVED_FACT),
            metadata={"gate": "post_route_timing", "status": "pass" if wns_value >= 0 else "fail",
                      "fixture_authority": artifact.metadata.get("fixture_authority")},
        ))

    @staticmethod
    def _utilization(context: ExtractionContext, result: ExtractionResult,
                     artifact: Any, subject: str, stage: str) -> None:
        text = _read(project_path(context.project_root, artifact.uri))
        resources: dict[str, float] = {}
        patterns = {
            "lut": [r"\bCLB LUTs\s*\|\s*([\d,]+)", r"\bLUT\s*[:=]\s*([\d,]+)"],
            "ff": [r"\bCLB Registers\s*\|\s*([\d,]+)", r"\bFF\s*[:=]\s*([\d,]+)"],
            "dsp": [r"\bDSPs?\s*\|\s*([\d,]+)", r"\bDSP\s*[:=]\s*([\d,]+)"],
            "bram_18k": [r"\bBlock RAM Tile\s*\|\s*([\d,.]+)", r"\bBRAM_18K\s*[:=]\s*([\d,]+)"],
            "uram": [r"\bURAM\s*\|\s*([\d,]+)", r"\bURAM\s*[:=]\s*([\d,]+)"],
        }
        for name, alternatives in patterns.items():
            for pattern in alternatives:
                match = re.search(pattern, text, re.I)
                if match:
                    resources[name] = float(match.group(1).replace(",", ""))
                    break
        if not resources:
            raise ValueError("no utilization resources were found")
        for name, value in resources.items():
            result.observations.append(_report_observation(artifact,
                snapshot_id=context.snapshot.id, subject_id=subject,
                predicate=f"resource.{name}", value=value, unit="count",
                stage=stage,
                authority=context.authority_for(artifact, AuthorityClass.TOOL_OBSERVATION),
                artifact_id=artifact.id, metadata={"stage": stage},
            ))

    @staticmethod
    def _physical_json(context: ExtractionContext, result: ExtractionResult,
                       artifact: Any, subject: str, stage: str) -> None:
        data = json.loads(_read(project_path(context.project_root, artifact.uri)))
        if data.get("schema_version") != "hlsgraph.vivado.physical_summary.v1":
            raise ValueError("unsupported normalized physical summary schema")
        mapping = {
            "congestion_level": ("physical.congestion_level", None),
            "slr_crossings": ("physical.slr_crossings", "count"),
            "critical_path_delay_ns": ("timing.critical_path_delay_ns", "ns"),
            "drc_errors": ("physical.drc_errors", "count"),
            "cdc_critical": ("physical.cdc_critical", "count"),
            "dynamic_power_w": ("power.dynamic_w", "W"),
            "static_power_w": ("power.static_w", "W"),
        }
        for key, (predicate, unit) in mapping.items():
            if key in data:
                result.observations.append(_report_observation(artifact,
                    snapshot_id=context.snapshot.id, subject_id=subject,
                    predicate=predicate, value=data[key], unit=unit,
                    stage=stage,
                    authority=context.authority_for(artifact, AuthorityClass.TOOL_OBSERVATION),
                    artifact_id=artifact.id,
                    metadata={"activity_source": data.get("activity_source")
                              if predicate.startswith("power.") else None},
                ))

    @staticmethod
    def _derive_resource_gate(context: ExtractionContext, result: ExtractionResult,
                              allowed_artifact_ids: set[str]) -> None:
        grouped: dict[str, dict[str, Observation]] = {}
        duplicate = False
        for item in result.observations:
            if (item.stage != Stage.POST_ROUTE.value
                    or not item.predicate.startswith("resource.")
                    or item.artifact_id not in allowed_artifact_ids):
                continue
            values = grouped.setdefault(str(item.artifact_id), {})
            name = item.predicate.split(".", 1)[1].lower()
            if name in values:
                duplicate = True
            values[name] = item
        capacities = {str(key).lower(): float(value)
                      for key, value in context.manifest.target.capacities.items()}
        reserved = {str(key).lower(): float(value)
                    for key, value in context.manifest.target.reserved_resources.items()}
        if not grouped or not capacities:
            result.diagnostics.append(Diagnostic(
                snapshot_id=context.snapshot.id, code="gate.resource_capacity_unknown",
                severity=DiagnosticSeverity.INFO,
                message="post-route resource fit is unknown because utilization or effective capacities are missing",
                stage=Stage.POST_ROUTE.value,
            ))
            return
        if duplicate or len(grouped) != 1:
            result.diagnostics.append(Diagnostic(
                snapshot_id=context.snapshot.id, code="gate.resource_source_ambiguous",
                severity=DiagnosticSeverity.WARNING,
                message=("post-route resource fit requires exactly one scoped utilization "
                         "artifact with one value per resource"),
                stage=Stage.POST_ROUTE.value,
                metadata={"source_count": len(grouped), "duplicate_metric": duplicate},
            ))
            return
        _artifact_id, latest = next(iter(grouped.items()))
        missing_usage = sorted(set(capacities) - set(latest))
        missing_capacity = sorted(set(latest) - set(capacities))
        invalid_reserved = sorted(set(reserved) - set(capacities))
        if missing_usage or missing_capacity or invalid_reserved:
            result.diagnostics.append(Diagnostic(
                snapshot_id=context.snapshot.id, code="gate.resource_capacity_incomplete",
                severity=DiagnosticSeverity.WARNING,
                message=("post-route resource fit is unknown because utilization, capacity, "
                         "and reserved-resource keys are not a complete matching set"),
                stage=Stage.POST_ROUTE.value,
                subject_id=next(iter(latest.values())).subject_id,
                metadata={"missing_usage": missing_usage,
                          "missing_capacity": missing_capacity,
                          "reserved_without_capacity": invalid_reserved},
            ))
            return
        subject = next(iter(latest.values())).subject_id
        comparisons: dict[str, dict[str, float | bool]] = {}
        inputs: list[str] = []
        fit = True
        for name in sorted(capacities):
            observation = latest[name]
            available = capacities[name] - reserved.get(name, 0.0)
            if available < 0:
                result.diagnostics.append(Diagnostic(
                    snapshot_id=context.snapshot.id, code="gate.resource_capacity_invalid",
                    severity=DiagnosticSeverity.WARNING,
                    message=f"reserved {name} capacity exceeds the target capacity",
                    stage=Stage.POST_ROUTE.value, subject_id=subject,
                ))
                return
            ok = float(observation.value) <= available
            comparisons[name] = {"used": float(observation.value), "available": available, "fits": ok}
            inputs.append(observation.id)
            fit = fit and ok
        result.derivations.append(Derivation(
            snapshot_id=context.snapshot.id, subject_id=subject,
            predicate="gate.resource_fits", value=fit,
            algorithm="hlsgraph.gate.capacity_compare", algorithm_version="1",
            input_observation_ids=inputs,
            stage=Stage.POST_ROUTE.value,
            authority=(AuthorityClass.SYNTHETIC
                       if set(str(item.authority) for item in latest.values()) == {"synthetic"}
                       else AuthorityClass.DERIVED_FACT),
            metadata={"gate": "resource_fits", "comparisons": comparisons,
                      "target_profile_hash": stable_hash(context.manifest.target),
                      "input_authorities": sorted({str(item.authority) for item in latest.values()})},
        ))
