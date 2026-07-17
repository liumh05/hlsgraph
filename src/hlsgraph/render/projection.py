"""Canonical graph -> deterministic presentation-only dataflow projection."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from ..diagnostic_projection import redacted_diagnostic_record
from ..graph import CanonicalGraph
from ..model import Diagnostic, DiagnosticSeverity, Observation, json_ready, stable_hash


RENDER_SCHEMA_VERSION = "hlsgraph.render.v1"
STAGE_ORDER = {
    "source": 0, "ast": 1, "mlir": 2, "hls_ir": 3, "llvm": 4,
    "schedule": 5, "rtl": 6, "post_synth": 7, "post_place": 8,
    "post_route": 9, "csim": 10, "cosim": 11, "hardware_runtime": 12,
}
_HARDWARE_STRUCTURE_STAGES = {
    "mlir", "hls_ir", "schedule", "rtl", "post_synth", "post_place", "post_route",
}


def _is_hardware_containment(relation: Any) -> bool:
    """AST/source containment is a lexical anchor, never hardware topology."""
    return (relation.kind == "hls.contains"
            and relation.stage in _HARDWARE_STRUCTURE_STAGES
            and str(relation.authority) not in {"static_fact", "prediction_hypothesis"})


def _is_hardware_dataflow(relation: Any) -> bool:
    """Only compiler/tool-backed HLS structure may become a dataflow edge."""
    return (
        relation.kind == "hls.streams_to"
        and relation.stage in _HARDWARE_STRUCTURE_STAGES
        and str(relation.authority) not in {
            "static_fact", "declared_constraint", "knowledge_rule",
            "prediction_hypothesis",
        }
        and relation.attrs.get("hardware_topology") is not False
        and relation.attrs.get("hardware_instance") is not False
    )


def _value(item: Observation) -> Any:
    return item.value


_AUTHORITY_ORDER = {
    "synthetic": 0,
    "declared_constraint": 1,
    "static_fact": 2,
    "derived_fact": 3,
    "compiler_decision": 4,
    "tool_observation": 5,
    "verification_evidence": 6,
    "physical_measurement": 7,
}
_COMPLETENESS_ORDER = {"missing": 0, "ambiguous": 1, "partial": 2, "complete": 3}


def _display_observations(
        values: list[Observation]) -> tuple[dict[str, Observation], list[str]]:
    """Choose display metrics without hiding equal-rank conflicting evidence."""
    grouped: dict[str, list[Observation]] = defaultdict(list)
    for item in values:
        grouped[item.predicate].append(item)
    selected: dict[str, Observation] = {}
    conflicts: list[str] = []
    for predicate, candidates in grouped.items():
        rank = lambda item: (  # noqa: E731 - compact deterministic policy
            _AUTHORITY_ORDER.get(str(item.authority), -1),
            _COMPLETENESS_ORDER.get(str(item.completeness), -1),
            STAGE_ORDER.get(item.stage, -1),
        )
        best_rank = max(rank(item) for item in candidates)
        best = [item for item in candidates if rank(item) == best_rank]
        semantic_values = {stable_hash(item.value) for item in best}
        if len(semantic_values) > 1:
            conflicts.append(predicate)
            continue
        selected[predicate] = sorted(best, key=lambda item: item.id)[-1]
    return selected, sorted(conflicts)


def _scope(graph: CanonicalGraph, scope_id: str | None) -> set[str]:
    if not scope_id:
        return set(graph.entities)
    entities, _relations = graph.traverse(scope_id, depth=8, direction="both",
                                          relation_kinds=["hls.contains", "hls.streams_to"])
    return {item.id for item in entities}


def to_render_data(graph: CanonicalGraph, observations: list[Observation],
                   diagnostics: list[Diagnostic], *, scope_id: str | None = None) -> dict[str, Any]:
    # Rendering is a public output surface even when invoked directly through
    # the SDK.  Normalize at the lowest shared boundary so HTML/JSON/static
    # formats cannot accidentally serialize vendor/plugin diagnostic details.
    diagnostics = [projected for item in diagnostics
                   if (projected := redacted_diagnostic_record(item)) is not None]
    allowed = _scope(graph, scope_id)
    dataflow = [item for item in graph.relations.values()
                if _is_hardware_dataflow(item)
                and item.src in allowed and item.dst in allowed]
    if dataflow:
        node_ids = {item.src for item in dataflow} | {item.dst for item in dataflow}
        edge_values = dataflow
        view = "dataflow"
    else:
        architectural = {item.id for item in graph.entities.values()
                         if item.id in allowed and item.kind in {
                             "hls.kernel", "hls.function", "hls.process", "hls.region",
                             "hls.loop", "hls.memory", "hls.buffer", "hls.stream", "hls.port",
                         }}
        node_ids = architectural
        edge_values = [item for item in graph.relations.values()
                       if _is_hardware_containment(item)
                       and item.src in node_ids and item.dst in node_ids]
        view = ("architecture_hierarchy" if edge_values
                else "architecture_evidence_incomplete")
    by_subject: dict[str, list[Observation]] = defaultdict(list)
    for item in observations:
        by_subject[item.subject_id].append(item)
    by_diagnostic: dict[str | None, list[Diagnostic]] = defaultdict(list)
    for item in diagnostics:
        by_diagnostic[item.subject_id].append(item)
    directives_by_scope: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for relation in sorted(graph.relations.values(), key=lambda item: item.id):
        if relation.kind != "hls.annotates":
            continue
        directive = graph.entities.get(relation.src)
        if not directive or directive.kind != "hls.directive":
            continue
        directive_observations = sorted(by_subject.get(directive.id, []),
                                        key=lambda item: item.id)
        directives_by_scope[relation.dst].append({
            "directive_id": directive.id,
            "kind": directive.attrs.get("directive_kind", directive.name),
            "state": directive.attrs.get("state", "requested"),
            "origin": directive.attrs.get("origin"),
            "options": directive.attrs.get("options", {}),
            "scope_node_id": relation.dst,
            "scope_resolution": relation.attrs.get("scope_resolution"),
            "completeness": str(directive.completeness),
            "observations": [json_ready(item) for item in directive_observations],
            "evidence": [json_ready(anchor) for anchor in directive.anchors],
        })

    prepared: dict[str, dict[str, Any]] = {}
    for entity_id in sorted(node_ids):
        entity = graph.entities.get(entity_id)
        if not entity:
            continue
        selected, observation_conflicts = _display_observations(
            by_subject.get(entity_id, [])
        )
        achieved_ii = selected.get("qor.achieved_ii")
        target_ii = selected.get("qor.target_ii")
        latency = (selected.get("qor.latency_cycles") or
                   selected.get("schedule.operation_latency") or
                   selected.get("qor.latency_worst_cycles"))
        resource_lut = selected.get("resource.lut")
        resource_dsp = selected.get("resource.dsp")
        metric_sources = {
            "latency": latency,
            "achieved_II": achieved_ii,
            "target_II": target_ii,
            "lut": resource_lut,
            "dsp": resource_dsp,
        }
        category = "mem" if entity.kind in {
            "hls.memory", "hls.buffer", "hls.stream", "hls.port"
        } else "compute"
        prepared[entity_id] = {
            "id": entity.id,
            "label": entity.name,
            "name": entity.name,
            "qualified_name": entity.qualified_name,
            "type": entity.kind,
            "category": category,
            "stage": entity.stage,
            "authority": str(entity.authority),
            "is_bottleneck": False,
            "bottleneck_cause": "",
            "metrics": {
                "latency": _value(latency) if latency else None,
                "achieved_II": _value(achieved_ii) if achieved_ii else None,
                "target_II": _value(target_ii) if target_ii else None,
                "replication": entity.attrs.get("replication"),
                "lut": _value(resource_lut) if resource_lut else None,
                "dsp": _value(resource_dsp) if resource_dsp else None,
                "bitwidth": entity.attrs.get("bitwidth"),
                "fifo_depth": entity.attrs.get("depth") or entity.attrs.get("fifo_depth"),
            },
            "metric_evidence": {
                name: {
                    "observation_id": observation.id,
                    "predicate": observation.predicate,
                    "stage": observation.stage,
                    "authority": str(observation.authority),
                    "artifact_id": observation.artifact_id,
                    "run_id": observation.run_id,
                }
                for name, observation in metric_sources.items() if observation is not None
            },
            "attrs": entity.attrs,
            "observations": [json_ready(item) for item in sorted(by_subject.get(entity_id, []),
                                                                  key=lambda value: value.id)],
            "evidence": [json_ready(anchor) for anchor in entity.anchors],
            "diagnostics": [json_ready(item) for item in sorted(by_diagnostic.get(entity_id, []),
                                                                  key=lambda value: value.id)],
            "directives": directives_by_scope.get(entity_id, []),
            "display_conflicts": observation_conflicts,
            "completeness": str(entity.completeness),
        }

    bottleneck_candidates: list[tuple[float, float, str, str]] = []
    for entity_id, item in prepared.items():
        achieved = item["metrics"]["achieved_II"]
        target = item["metrics"]["target_II"]
        latency = item["metrics"]["latency"]
        violation = float(achieved) - float(target) if achieved is not None and target is not None else 0.0
        if violation > 0 or latency is not None:
            evidence_key = "achieved_II" if violation > 0 else "latency"
            authority = item["metric_evidence"].get(evidence_key, {}).get(
                "authority", "unknown"
            )
            reason = (f"achieved II {achieved} exceeds target II {target}" if violation > 0
                      else f"highest observed latency candidate: {latency} cycles")
            if authority == "synthetic":
                reason = "synthetic fixture candidate: " + reason
            bottleneck_candidates.append((violation, float(latency or 0), entity_id, reason))
    if bottleneck_candidates:
        _, _, bottleneck_id, reason = max(bottleneck_candidates,
                                           key=lambda value: (value[0], value[1], value[2]))
        prepared[bottleneck_id]["is_bottleneck"] = True
        prepared[bottleneck_id]["bottleneck_cause"] = reason

    edges = []
    for relation in sorted(edge_values, key=lambda item: item.id):
        edges.append({
            "id": relation.id,
            "source": relation.src,
            "target": relation.dst,
            "type": "STREAMS_TO" if relation.kind == "hls.streams_to" else "CONTAINS",
            "fifo_depth": relation.attrs.get("fifo_depth"),
            "elem_type": relation.attrs.get("elem_type"),
            "stage": relation.stage,
            "authority": str(relation.authority),
            "attrs": relation.attrs,
            "evidence": [json_ready(anchor) for anchor in relation.anchors],
        })
    nodes = [prepared[key] for key in sorted(prepared)]
    return {
        "render_schema_version": RENDER_SCHEMA_VERSION,
        "canonical_schema_version": graph.schema_version,
        "meta": {
            "snapshot_id": graph.snapshot_id,
            "graph_hash": graph.graph_hash,
            "top": graph.metadata.get("top"),
            "scope": scope_id or "whole design",
            "view": view,
            "incomplete": (
                view == "architecture_evidence_incomplete"
                or any(item["completeness"] != "complete"
                       or item["display_conflicts"] for item in nodes)
                or any(str(item.completeness) != "complete" for item in edge_values)
                or any(str(item.completeness) != "complete" for item in observations)
                or any(item.severity in {DiagnosticSeverity.WARNING,
                                         DiagnosticSeverity.ERROR,
                                         DiagnosticSeverity.CRITICAL}
                       for item in diagnostics)
                or any(item.kind == "hls.directive" and not any(
                    relation.kind == "hls.annotates" and relation.src == item.id
                    for relation in graph.relations.values()
                ) for item in graph.entities.values())
            ),
            "topology_policy": (
                "source/AST containment is excluded; only HLS IR, schedule, RTL, or "
                "dialect-derived dataflow may form architecture edges"
            ),
        },
        "nodes": nodes,
        "edges": edges,
    }
