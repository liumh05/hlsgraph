"""One structured query service shared by SDK, CLI, REST, and MCP."""
from __future__ import annotations

import base64
import difflib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from .bundle import GraphBundle
from .graph import CanonicalGraph
from .model import (
    AuthorityClass, DiagnosticSeverity, Entity, GateKind, GateStatus, hash_artifact_bytes,
    json_ready, stable_hash,
)
from .version import SCHEMA_VERSION


DEFAULT_IMPACT_RELATIONS = (
    "hls.streams_to", "handshake.dataflow", "hls.annotates",
    "cross.maps_to", "cross.projects_to",
)

_NON_ARCHITECTURE_RELATIONS = frozenset({
    "software.calls", "llvm.calls", "llvm.cfg", "ir.contains",
})


# Keep this explicit allow-list in lockstep with the parser contracts.  Prefixes
# such as ``vendor.*`` are intentionally insufficient proof of a verification.
_VERIFICATION_REPORT_POLICY: dict[str, dict[str, frozenset[str] | str]] = {
    "csim": {
        "run_stage": "csim",
        "observation_stages": frozenset({"csim"}),
        "artifact_kinds": frozenset({
            "amd.vitis.csim_result",
        }),
    },
    "rtl_cosim": {
        "run_stage": "rtl_cosim",
        "observation_stages": frozenset({"cosim"}),
        "artifact_kinds": frozenset({
            "amd.vitis.cosim_rpt", "amd.vitis.cosim_report",
        }),
    },
}

_PHYSICAL_GATE_REPORT_KINDS: dict[str, frozenset[str]] = {
    "gate.resource_fits": frozenset({
        "amd.vivado.post_route_utilization", "amd.vivado.utilization",
    }),
    "gate.post_route_timing": frozenset({
        "amd.vivado.post_route_timing", "amd.vivado.timing_summary",
    }),
}

_TOOL_EVIDENCE_AUTHORITIES = frozenset({
    "tool_observation", "verification_evidence", "physical_measurement",
})


def managed_artifact_integrity(bundle: GraphBundle, artifact: Any) -> tuple[bool, str]:
    """Revalidate one immutable evidence blob against its live content-addressed bytes.

    This helper is intentionally reusable by ML/export surfaces: a ledger producer link
    is provenance, not proof that a managed report still exists or retains its hash.
    """
    if str(artifact.retention) != "managed":
        return (False, "evidence_artifact_not_managed")
    project_root = bundle.project_root.resolve()
    path = (project_root / artifact.uri).resolve()
    try:
        path.relative_to(project_root)
        data = path.read_bytes()
    except (OSError, ValueError):
        return (False, "evidence_artifact_missing")
    if len(data) != artifact.size or hash_artifact_bytes(data) != artifact.sha256:
        return (False, "evidence_artifact_hash_mismatch")
    return (True, "managed_artifact_verified")


def _tokens(value: str) -> list[str]:
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value)
    return [item.casefold() for item in re.findall(r"[\w:+.-]+", value, re.UNICODE)]


def _cursor(snapshot_id: str, query_hash: str, offset: int) -> str:
    raw = json.dumps({"s": snapshot_id, "q": query_hash, "o": offset},
                     separators=(",", ":"), sort_keys=True).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _decode_cursor(value: str, snapshot_id: str, query_hash: str) -> int:
    try:
        raw = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
        data = json.loads(raw)
        if data["s"] != snapshot_id or data["q"] != query_hash:
            raise ValueError("cursor belongs to another query or snapshot")
        return max(0, int(data["o"]))
    except Exception as exc:
        raise ValueError("invalid query cursor") from exc


@dataclass(slots=True)
class QuerySpec:
    query: str
    kinds: list[str] = field(default_factory=list)
    scope_id: str | None = None
    stages: list[str] = field(default_factory=list)
    authorities: list[str] = field(default_factory=list)
    limit: int = 20
    cursor: str | None = None


@dataclass(slots=True)
class QueryItem:
    entity_id: str
    kind: str
    name: str
    qualified_name: str | None
    score: float
    match_type: str
    matched_fields: list[str]
    source_spans: list[dict[str, Any]]


@dataclass(slots=True)
class QueryResult:
    snapshot_id: str
    items: list[QueryItem]
    next_cursor: str | None = None
    truncated: bool = False
    warnings: list[str] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return json_ready(self)


@dataclass(slots=True)
class ExploreSpec:
    query: str | None = None
    scope_id: str | None = None
    view: str = "architecture"
    depth: int = 1
    top_k: int = 8
    cursor: str | None = None


@dataclass(slots=True)
class ExploreResult:
    snapshot_id: str
    summary: str
    focus: str | None
    entities: list[dict[str, Any]]
    relations: list[dict[str, Any]]
    observations: list[dict[str, Any]]
    diagnostics: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    incomplete: bool
    next_cursor: str | None = None
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return json_ready(self)


@dataclass(slots=True)
class StatusResult:
    data: dict[str, Any]
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, **json_ready(self.data)}


class CoreService:
    def __init__(self, bundle: GraphBundle, snapshot_id: str | None = None):
        self.bundle = bundle
        snapshot = bundle.latest_snapshot() if snapshot_id is None else None
        self.snapshot_id = snapshot_id or (snapshot.id if snapshot else None)
        if not self.snapshot_id:
            raise ValueError("bundle has no indexed snapshot")

    def graph(self) -> CanonicalGraph:
        return self.bundle.store.load_graph(self.snapshot_id)

    def query(self, spec: QuerySpec) -> QueryResult:
        graph = self.graph()
        query = spec.query.strip()
        if not query:
            return QueryResult(snapshot_id=self.snapshot_id, items=[])
        limit = max(1, min(int(spec.limit), 100))
        allowed_ids = self._scope_ids(graph, spec.scope_id) if spec.scope_id else set(graph.entities)
        candidates = [entity for entity in graph.entities.values()
                      if entity.id in allowed_ids
                      and (not spec.kinds or entity.kind in spec.kinds)
                      and (not spec.stages or entity.stage in spec.stages)
                      and (not spec.authorities or str(entity.authority) in spec.authorities)]
        query_folded = query.casefold()
        query_tokens = _tokens(query)
        ranked: dict[str, tuple[int, float, str, list[str]]] = {}

        def offer(entity: Entity, rank: int, score: float, match_type: str,
                  fields: list[str]) -> None:
            previous = ranked.get(entity.id)
            value = (rank, score, match_type, fields)
            if previous is None or (rank, score) > (previous[0], previous[1]):
                ranked[entity.id] = value

        for entity in candidates:
            fields = {
                "id": entity.id,
                "name": entity.name,
                "qualified_name": entity.qualified_name or "",
                "aliases": " ".join(entity.aliases),
            }
            exact_fields = [name for name, value in fields.items() if value.casefold() == query_folded]
            if exact_fields:
                offer(entity, 4, 1.0, "exact", exact_fields)
                continue
            substring_fields = [name for name, value in fields.items() if query_folded in value.casefold()]
            if substring_fields:
                offer(entity, 3, min(1.0, len(query) / max(1, min(len(fields[name]) for name in substring_fields))),
                      "substring", substring_fields)

        for hit in self.bundle.store.search_entities(self.snapshot_id, query, limit=100):
            entity = graph.entities.get(hit["entity_id"])
            if entity and entity in candidates:
                offer(entity, 2, float(hit["score"]), "fts", ["indexed_text"])

        for entity in candidates:
            if entity.id in ranked:
                continue
            target_tokens = _tokens(" ".join([entity.name, entity.qualified_name or "", *entity.aliases]))
            if not target_tokens:
                continue
            left = " ".join(query_tokens)
            right = " ".join(target_tokens)
            score = difflib.SequenceMatcher(None, left, right).ratio()
            token_score = max((difflib.SequenceMatcher(None, token, candidate).ratio()
                               for token in query_tokens for candidate in target_tokens), default=0.0)
            score = max(score, token_score)
            if score >= 0.58:
                offer(entity, 1, score, "fuzzy", ["name"])

        ordered = sorted(ranked.items(), key=lambda item: (-item[1][0], -item[1][1], item[0]))
        query_identity = dict(json_ready(spec))
        query_identity.pop("cursor", None)
        query_hash = stable_hash(query_identity)
        offset = _decode_cursor(spec.cursor, self.snapshot_id, query_hash) if spec.cursor else 0
        page = ordered[offset:offset + limit]
        items = [QueryItem(
            entity_id=entity_id, kind=graph.entities[entity_id].kind,
            name=graph.entities[entity_id].name,
            qualified_name=graph.entities[entity_id].qualified_name,
            score=round(data[1], 8), match_type=data[2], matched_fields=data[3],
            source_spans=[json_ready(anchor) for anchor in graph.entities[entity_id].anchors],
        ) for entity_id, data in page]
        next_offset = offset + len(page)
        truncated = next_offset < len(ordered)
        return QueryResult(snapshot_id=self.snapshot_id, items=items,
                           next_cursor=_cursor(self.snapshot_id, query_hash, next_offset) if truncated else None,
                           truncated=truncated)

    @staticmethod
    def _scope_ids(graph: CanonicalGraph, scope_id: str | None) -> set[str]:
        if not scope_id:
            return set(graph.entities)
        if scope_id not in graph.entities:
            raise KeyError(scope_id)
        keep = {scope_id}
        changed = True
        while changed:
            changed = False
            for relation in graph.relations.values():
                if relation.kind in {"hls.contains", "ir.contains"} and relation.src in keep and relation.dst not in keep:
                    keep.add(relation.dst)
                    changed = True
        return keep

    def explore(self, spec: ExploreSpec) -> ExploreResult:
        graph = self.graph()
        if spec.view not in {"architecture", "evidence"}:
            raise ValueError("view must be architecture or evidence")
        traversal_graph = self._view_graph(graph, spec.view)
        focus = spec.scope_id
        next_cursor = None
        if focus and focus not in traversal_graph.entities:
            raise ValueError(
                f"entity {focus!r} is not present in the {spec.view} view; "
                "select evidence view for source/IR-only entities"
            )
        if not focus and spec.query:
            folded = spec.query.strip().casefold()
            exact = sorted({entity.id for entity in traversal_graph.entities.values()
                            if folded and any(value.casefold() == folded for value in (
                                entity.id, entity.name, entity.qualified_name or "", *entity.aliases,
                            ))})
            if len(exact) > 1:
                raise ValueError(
                    f"ambiguous exact context query {spec.query!r}; use one stable entity ID: "
                    f"{', '.join(exact)}"
                )
            query_result = self.query(QuerySpec(
                query=spec.query, limit=100,
                cursor=spec.cursor,
            ))
            view_items = [item for item in query_result.items
                          if item.entity_id in traversal_graph.entities]
            focus = exact[0] if exact else (
                view_items[0].entity_id if view_items else None
            )
            next_cursor = query_result.next_cursor
            if focus is None:
                raise KeyError(
                    f"no entity matching {spec.query!r} exists in the {spec.view} view; "
                    "select evidence view for source/IR-only entities"
                )
        if focus:
            entities, relations = traversal_graph.traverse(
                focus, depth=max(0, min(spec.depth, 8))
            )
        else:
            entities = sorted(traversal_graph.entities.values(), key=lambda item: item.id)[
                :max(1, min(spec.top_k, 50))
            ]
            keep = {item.id for item in entities}
            relations = [item for item in traversal_graph.relations.values()
                         if item.src in keep and item.dst in keep]
        entity_ids = {item.id for item in entities}
        observations = [item for entity_id in sorted(entity_ids)
                        for item in self.bundle.store.observations(self.snapshot_id, subject_id=entity_id)]
        diagnostics = [item for item in self.bundle.store.active_diagnostics(self.snapshot_id)
                       if item.subject_id is None or item.subject_id in entity_ids]
        artifact_ids = {anchor.artifact_id for entity in entities for anchor in entity.anchors}
        artifact_ids.update(item.artifact_id for item in observations if item.artifact_id)
        artifacts = [item for item in self.bundle.store.artifacts(self.snapshot_id) if item.id in artifact_ids]
        incomplete = any(str(entity.completeness) != "complete" for entity in entities) or any(
            item.severity in {DiagnosticSeverity.ERROR, DiagnosticSeverity.CRITICAL}
            for item in diagnostics
        )
        focus_name = graph.entities[focus].name if focus in graph.entities else None
        summary = (f"focus={focus_name or 'overview'}; entities={len(entities)}; "
                   f"relations={len(relations)}; observations={len(observations)}; "
                   f"diagnostics={len(diagnostics)}; incomplete={str(incomplete).lower()}")
        return ExploreResult(
            snapshot_id=self.snapshot_id, summary=summary, focus=focus,
            entities=[json_ready(item) for item in entities],
            relations=[json_ready(item) for item in sorted(relations, key=lambda value: value.id)],
            observations=[json_ready(item) for item in sorted(observations, key=lambda value: value.id)],
            diagnostics=[json_ready(item) for item in sorted(diagnostics, key=lambda value: value.id)],
            evidence=[json_ready(item) for item in sorted(artifacts, key=lambda value: value.id)],
            incomplete=incomplete, next_cursor=next_cursor,
        )

    @staticmethod
    def _view_graph(graph: CanonicalGraph, view: str) -> CanonicalGraph:
        if view == "evidence":
            return graph
        projected = CanonicalGraph(
            snapshot_id=graph.snapshot_id, metadata=dict(graph.metadata),
            schema_version=graph.schema_version,
        )
        for entity in graph.entities.values():
            if entity.kind.startswith(("ir.", "source.", "software.", "ast.")):
                continue
            projected.add_entity(entity)
        for relation in graph.relations.values():
            if relation.src not in projected.entities or relation.dst not in projected.entities:
                continue
            if relation.kind in _NON_ARCHITECTURE_RELATIONS:
                continue
            if relation.kind == "hls.contains" and relation.stage in {"source", "ast"}:
                continue
            if relation.kind == "hls.streams_to" and (
                relation.stage in {"source", "ast"}
                or str(relation.authority) == AuthorityClass.STATIC_FACT.value
            ):
                continue
            if relation.attrs.get("hardware_topology") is False:
                continue
            if relation.attrs.get("hardware_instance") is False:
                continue
            projected.add_relation(relation)
        connected = {endpoint for relation in projected.relations.values()
                     for endpoint in (relation.src, relation.dst)}
        for entity_id, entity in list(projected.entities.items()):
            if (entity_id not in connected
                    and entity.kind != "hls.kernel"
                    and entity.stage in {"source", "ast"}):
                del projected.entities[entity_id]
        return projected

    def evidence(self, entity_id: str) -> dict[str, Any]:
        graph = self.graph()
        entity = graph.entities.get(entity_id)
        if not entity:
            raise KeyError(entity_id)
        observations = self.bundle.store.observations(self.snapshot_id, subject_id=entity_id)
        artifact_ids = {anchor.artifact_id for anchor in entity.anchors}
        artifact_ids.update(item.artifact_id for item in observations if item.artifact_id)
        artifacts = {item.id: item for item in self.bundle.store.artifacts(self.snapshot_id)}
        observation_ids = {item.id for item in observations}
        derivations = [item for item in self.bundle.store.derivations(self.snapshot_id)
                       if item.get("subject_id") == entity_id
                       or observation_ids.intersection(item.get("input_observation_ids", []))]
        verifications = [item for item in self.bundle.store.verifications(self.snapshot_id)
                         if observation_ids.intersection(item.get("evidence_ids", []))]
        return {
            "schema_version": SCHEMA_VERSION,
            "snapshot_id": self.snapshot_id,
            "entity": json_ready(entity),
            "observations": [json_ready(item) for item in observations],
            "artifacts": [json_ready(artifacts[item]) for item in sorted(artifact_ids) if item in artifacts],
            "derivations": derivations,
            "verifications": verifications,
            "diagnostics": [json_ready(item) for item in self.bundle.store.active_diagnostics(self.snapshot_id)
                            if item.subject_id == entity_id],
        }

    def traverse(self, entity_id: str, *, depth: int = 1, direction: str = "both",
                 relation_kinds: Iterable[str] = ()) -> dict[str, Any]:
        if direction not in {"in", "out", "both"}:
            raise ValueError("direction must be in, out, or both")
        graph = self.graph()
        entities, relations = graph.traverse(
            entity_id, depth=max(0, min(int(depth), 8)), direction=direction,
            relation_kinds=list(relation_kinds),
        )
        return {
            "schema_version": SCHEMA_VERSION, "snapshot_id": self.snapshot_id,
            "start_id": entity_id, "direction": direction,
            "entities": [json_ready(item) for item in entities],
            "relations": [json_ready(item) for item in relations],
            "inference_policy": "explicit_relations_only",
        }

    def impact(self, entity_id: str, *, depth: int = 2,
               relation_kinds: Iterable[str] = ()) -> dict[str, Any]:
        selected = tuple(relation_kinds) or DEFAULT_IMPACT_RELATIONS
        graph = self._view_graph(self.graph(), "architecture")
        entities, relations = graph.traverse(
            entity_id, depth=max(0, min(int(depth), 8)), direction="out",
            relation_kinds=selected,
        )
        traversed = {
            "schema_version": SCHEMA_VERSION, "snapshot_id": self.snapshot_id,
            "start_id": entity_id, "direction": "out",
            "entities": [json_ready(item) for item in entities],
            "relations": [json_ready(item) for item in relations],
            "inference_policy": "explicit_architecture_relations_only",
        }
        traversed.update({
            "impact_semantics": "dependency_facts_only",
            "relation_kinds": list(selected),
            "excluded_from_default": [
                "software.calls", "llvm.calls", "llvm.cfg", "ir.contains",
                "source/ast hls.contains",
            ],
            "qor_prediction": None,
            "warning": ("Affected entities are graph dependencies, not predicted latency, "
                        "resource, correctness, or timing changes."),
        })
        return traversed

    def status(self) -> StatusResult:
        graph = self.graph()
        diagnostics = self.bundle.store.active_diagnostics(self.snapshot_id)
        data = self.bundle.status(self.snapshot_id)
        data.update({
            "graph": graph.stats(),
            "runs": len(self.bundle.store.runs(self.snapshot_id)),
            "observations": len(self.bundle.store.observations(self.snapshot_id)),
            "verification_gates": self.verification_gates(),
            "completeness": "error" if any(item.severity in {
                DiagnosticSeverity.ERROR, DiagnosticSeverity.CRITICAL} for item in diagnostics)
            else "partial" if diagnostics else "complete",
        })
        return StatusResult(data=data)

    def verification_gates(self) -> dict[str, Any]:
        """Report correctness, resource fit, and post-route timing independently."""
        gates: dict[str, Any] = {
            str(kind): {"status": GateStatus.UNKNOWN.value, "evidence_ids": [],
                        "authorities": [], "synthetic_only": False,
                        "trusted_pass": False, "tool_truth": False}
            for kind in (GateKind.CORRECTNESS, GateKind.RESOURCE_FITS,
                         GateKind.POST_ROUTE_TIMING)
        }

        observations = {item.id: item for item in
                        self.bundle.store.observations(self.snapshot_id)}
        derivations = {str(item["id"]): item for item in
                       self.bundle.store.derivations(self.snapshot_id)}
        verifications = {str(item["id"]): item for item in
                         self.bundle.store.verifications(self.snapshot_id)}
        runs = {item.id: item for item in self.bundle.store.runs(self.snapshot_id)}
        diagnostics = {item.id: item for item in
                       self.bundle.store.diagnostics(self.snapshot_id)}
        artifacts = {item.id: item for item in self.bundle.store.artifacts(self.snapshot_id)}
        snapshot_manifest = self.bundle.store.snapshot_manifest(self.snapshot_id)
        trust_cache: dict[tuple[str, str], tuple[bool, bool, str]] = {}

        def run_trust(run_id: str) -> tuple[bool, bool, str]:
            key = ("run", run_id)
            if key in trust_cache:
                return trust_cache[key]
            run = runs.get(run_id)
            if run is None:
                result = (False, False, "missing_run")
            else:
                terminal = str(run.status) in {"succeeded", "cached"}
                clean = (str(run.failure_class) == "none"
                         and run.exit_code in {None, 0})
                authority = str(run.metadata.get("authority", ""))
                backend = str(run.backend).casefold()
                denied = (authority in {"synthetic", "fake", "replay"}
                          or backend in {"runner.fake", "runner.replay"})
                tool_truth = (
                    run.metadata.get("tool_truth") is True
                    and run.metadata.get("fresh_execution") is True
                    and run.metadata.get("fresh_tool_truth") is True
                )
                inputs = list(run.input_artifact_ids)
                outputs = list(run.output_artifact_ids)
                artifact_contract = (
                    len(inputs) == len(set(inputs))
                    and len(outputs) == len(set(outputs))
                    and not (set(inputs) & set(outputs))
                    and all(
                        item in artifacts and artifacts[item].producer_run_id == run.id
                        for item in outputs
                    )
                    and all(
                        item not in artifacts or artifacts[item].producer_run_id != run.id
                        for item in inputs
                    )
                )
                manifest_contract = True
                if run.stage != "index":
                    try:
                        expected_toolchain = snapshot_manifest.toolchain_for_stage(run.stage)
                    except (KeyError, TypeError, ValueError):
                        manifest_contract = False
                    else:
                        base_artifact_ids = {
                            item.id for item in artifacts.values()
                            if item.producer_run_id is None
                        }
                        manifest_contract = (
                            run.stage in snapshot_manifest.stage_commands
                            and run.toolchain_id == expected_toolchain.id
                            and run.environment_hash == expected_toolchain.environment_hash
                            and list(run.command)
                            == list(snapshot_manifest.stage_commands[run.stage])
                            and run.working_directory == "."
                            and base_artifact_ids.issubset(set(inputs))
                        )
                valid = (terminal and clean and tool_truth and not denied
                         and artifact_contract and manifest_contract)
                if not manifest_contract:
                    authority = "run_snapshot_manifest_mismatch"
                result = (valid, valid, authority or "untrusted_run")
            trust_cache[key] = result
            return result

        def artifact_trust(artifact_id: str) -> tuple[bool, bool, str]:
            key = ("artifact", artifact_id)
            if key in trust_cache:
                return trust_cache[key]
            artifact = artifacts.get(artifact_id)
            if artifact is None:
                result = (False, False, "missing_artifact")
            elif artifact.producer_run_id:
                integrity, integrity_reason = managed_artifact_integrity(
                    self.bundle, artifact,
                )
                producer = runs.get(artifact.producer_run_id)
                if not integrity:
                    result = (False, False, integrity_reason)
                elif (producer is None or artifact.id not in producer.output_artifact_ids
                        or artifact.id in producer.input_artifact_ids):
                    result = (False, False, "artifact_producer_contract_mismatch")
                else:
                    result = run_trust(artifact.producer_run_id)
            else:
                # Input/source artifacts are valid static evidence, but cannot by
                # themselves establish that a vendor tool actually ran.
                result = (True, False, "unproduced_artifact")
            trust_cache[key] = result
            return result

        def combine(values: list[tuple[bool, bool, str]], *, require_tool: bool = False
                    ) -> tuple[bool, bool, str]:
            if not values:
                return (False, False, "missing_provenance")
            valid = all(item[0] for item in values)
            tool = any(item[1] for item in values)
            if require_tool:
                valid = valid and tool
            labels = "+".join(sorted(set(item[2] for item in values)))
            return (valid, tool, labels)

        def evidence_kind(evidence_id: str) -> str | None:
            for kind, values in (
                ("observation", observations), ("derivation", derivations),
                ("verification", verifications), ("diagnostic", diagnostics),
                ("artifact", artifacts), ("run", runs),
            ):
                if evidence_id in values:
                    return kind
            return None

        def evidence_run_ids(evidence_id: str,
                             stack: frozenset[str] = frozenset()) -> set[str]:
            if evidence_id in stack:
                return set()
            nested = stack | {evidence_id}
            if evidence_id in runs:
                return {evidence_id}
            artifact = artifacts.get(evidence_id)
            if artifact:
                return {artifact.producer_run_id} if artifact.producer_run_id else set()
            observation = observations.get(evidence_id)
            if observation:
                result = {observation.run_id} if observation.run_id else set()
                for artifact_id in {item for item in (
                    observation.artifact_id,
                    observation.anchor.artifact_id if observation.anchor else None,
                ) if item}:
                    result.update(evidence_run_ids(artifact_id, nested))
                return result
            derivation = derivations.get(evidence_id)
            if derivation:
                result: set[str] = set()
                for item in derivation.get("input_observation_ids", []):
                    result.update(evidence_run_ids(str(item), nested))
                return result
            verification = verifications.get(evidence_id)
            if verification:
                result = ({str(verification["run_id"])}
                          if verification.get("run_id") else set())
                for item in verification.get("evidence_ids", []):
                    result.update(evidence_run_ids(str(item), nested))
                return result
            diagnostic = diagnostics.get(evidence_id)
            if diagnostic:
                result = {diagnostic.run_id} if diagnostic.run_id else set()
                for artifact_id in {item for item in (
                    diagnostic.artifact_id,
                    diagnostic.anchor.artifact_id if diagnostic.anchor else None,
                ) if item}:
                    result.update(evidence_run_ids(artifact_id, nested))
                return result
            return set()

        def evidence_workloads(evidence_id: str,
                               stack: frozenset[str] = frozenset()) -> set[str]:
            if evidence_id in stack:
                return set()
            nested = stack | {evidence_id}
            observation = observations.get(evidence_id)
            if observation:
                return {observation.workload_id} if observation.workload_id else set()
            artifact = artifacts.get(evidence_id)
            if artifact:
                value = artifact.metadata.get("workload_id")
                return {str(value)} if isinstance(value, str) and value else set()
            derivation = derivations.get(evidence_id)
            if derivation:
                result: set[str] = set()
                for item in derivation.get("input_observation_ids", []):
                    result.update(evidence_workloads(str(item), nested))
                return result
            verification = verifications.get(evidence_id)
            if verification:
                result = ({str(verification["workload_id"])}
                          if verification.get("workload_id") else set())
                for item in verification.get("evidence_ids", []):
                    result.update(evidence_workloads(str(item), nested))
                return result
            return set()

        def evidence_observation_leaves(
            evidence_id: str, stack: frozenset[str] = frozenset(),
        ) -> list[Any] | None:
            """Resolve semantic leaves; ``None`` marks missing/cyclic pollution."""
            if evidence_id in stack:
                return None
            nested = stack | {evidence_id}
            observation = observations.get(evidence_id)
            if observation:
                return [observation]
            derivation = derivations.get(evidence_id)
            if derivation:
                result: list[Any] = []
                for item in derivation.get("input_observation_ids", []):
                    leaves = evidence_observation_leaves(str(item), nested)
                    if leaves is None:
                        return None
                    result.extend(leaves)
                return result
            if evidence_id in artifacts:
                return []
            return None

        def observation_artifacts(observation: Any) -> list[Any] | None:
            artifact_ids = {item for item in (
                observation.artifact_id,
                observation.anchor.artifact_id if observation.anchor else None,
            ) if item}
            if not artifact_ids or any(item not in artifacts for item in artifact_ids):
                return None
            return [artifacts[item] for item in sorted(artifact_ids)]

        def typed_verification_evidence(
            value: dict[str, Any], run_id: str,
        ) -> tuple[bool, str]:
            policy = _VERIFICATION_REPORT_POLICY.get(str(value.get("kind")))
            if policy is None:
                return (False, "verification_report_policy_missing")
            run = runs.get(run_id)
            if run is None or run.stage != str(policy["run_stage"]):
                return (False, "verification_stage_mismatch")
            allowed_stages = frozenset(policy["observation_stages"])
            allowed_kinds = frozenset(policy["artifact_kinds"])
            saw_typed_evidence = False
            all_leaves: list[Any] = []
            for evidence_id in (str(item) for item in value.get("evidence_ids", [])):
                kind = evidence_kind(evidence_id)
                if kind == "artifact":
                    return (False, "verification_bare_report_does_not_prove_pass")
                if kind not in {"observation", "derivation"}:
                    return (False, "passing_verification_has_nonreport_evidence")
                leaves = evidence_observation_leaves(evidence_id)
                if not leaves:
                    return (False, "verification_observation_evidence_missing")
                for observation in leaves:
                    cited = observation_artifacts(observation)
                    if (
                        str(observation.completeness) != "complete"
                        or observation.stage not in allowed_stages
                        or str(observation.authority) not in _TOOL_EVIDENCE_AUTHORITIES
                        or observation.run_id != run_id
                        or cited is None
                        or any(
                            artifact.kind not in allowed_kinds
                            or artifact.producer_run_id != run_id
                            for artifact in cited
                        )
                    ):
                        return (False, "verification_typed_observation_mismatch")
                    saw_typed_evidence = True
                    all_leaves.append(observation)
            if not saw_typed_evidence:
                return (False, "verification_report_evidence_missing")
            referenced_run = runs.get(run_id)
            run_workload = (str(referenced_run.metadata.get("workload_id"))
                            if referenced_run
                            and referenced_run.metadata.get("workload_id") else "")
            expected_workload = str(value.get("workload_id") or run_workload)
            if expected_workload and any(
                observation.workload_id != expected_workload
                for observation in all_leaves
            ):
                return (False, "verification_observation_workload_mismatch")
            grouped: dict[str, list[Any]] = {}
            for observation in all_leaves:
                grouped.setdefault(observation.predicate, []).append(observation)
            if str(value.get("kind")) == "csim":
                required = {
                    "csim.exit_code", "csim.mismatches", "csim.assertions_failed",
                }
                if any(len(grouped.get(predicate, [])) != 1 for predicate in required):
                    return (False, "csim_required_observation_missing")
                for predicate in required:
                    observation = grouped[predicate][0]
                    observed = observation.value
                    if (isinstance(observed, bool)
                            or not isinstance(observed, (int, float))
                            or not math.isfinite(float(observed))
                            or float(observed) != 0.0
                            or observation.unit != "count"):
                        return (False, "csim_pass_contradicts_observation")
            elif str(value.get("kind")) == "rtl_cosim":
                statuses = grouped.get("cosim.status", [])
                if (len(statuses) != 1 or not isinstance(statuses[0].value, str)
                        or statuses[0].value.casefold() != "pass"
                        or statuses[0].unit is not None):
                    return (False, "cosim_pass_status_observation_missing")
            return (True, "typed_verification_evidence")

        def typed_physical_gate_evidence(
            value: dict[str, Any], producer_id: str,
        ) -> tuple[bool, str]:
            allowed_kinds = _PHYSICAL_GATE_REPORT_KINDS.get(str(value.get("predicate")))
            if allowed_kinds is None:
                return (False, "physical_gate_policy_missing")
            inputs = [str(item) for item in value.get("input_observation_ids", [])]
            if not inputs:
                return (False, "physical_gate_observation_evidence_missing")
            all_leaves: list[Any] = []

            def scope_valid(artifact: Any) -> bool:
                scope = artifact.metadata.get("scope")
                if not isinstance(scope, dict):
                    return False
                top = snapshot_manifest.build.top
                if (scope.get("kind") != "kernel"
                        or str(scope.get("top") or "") != top
                        or str(scope.get("instance") or "") != top):
                    return False
                if (snapshot_manifest.target.part
                        and str(scope.get("part") or "") != snapshot_manifest.target.part):
                    return False
                if (snapshot_manifest.target.platform
                        and str(scope.get("platform") or "")
                        != snapshot_manifest.target.platform):
                    return False
                if artifact.kind in {
                    "amd.vivado.post_route_timing", "amd.vivado.timing_summary",
                }:
                    clock = str(scope.get("clock") or "")
                    known = {item.name for item in snapshot_manifest.target.clocks}
                    if clock != "all" and clock not in known:
                        return False
                return True

            for evidence_id in inputs:
                if evidence_run_ids(evidence_id) != {producer_id}:
                    return (False, "physical_gate_producer_mismatch")
                leaves = evidence_observation_leaves(evidence_id)
                if not leaves:
                    return (False, "physical_gate_observation_evidence_missing")
                for observation in leaves:
                    cited = observation_artifacts(observation)
                    if (
                        observation.stage != "post_route"
                        or str(observation.completeness) != "complete"
                        or str(observation.authority) not in _TOOL_EVIDENCE_AUTHORITIES
                        or observation.run_id != producer_id
                        or cited is None
                        or any(
                            artifact.kind not in allowed_kinds
                            or artifact.producer_run_id != producer_id
                            or not scope_valid(artifact)
                            or (
                                artifact.kind in {
                                    "amd.vivado.utilization",
                                    "amd.vivado.timing_summary",
                                }
                                and artifact.metadata.get("stage") != "post_route"
                            )
                            for artifact in cited
                        )
                    ):
                        return (False, "physical_gate_typed_observation_mismatch")
                    all_leaves.append(observation)
            predicate = str(value.get("predicate"))
            if predicate == "gate.post_route_timing":
                if (value.get("algorithm") != "hlsgraph.gate.wns_nonnegative"
                        or str(value.get("algorithm_version")) != "1"):
                    return (False, "physical_gate_algorithm_mismatch")
                wns = [item for item in all_leaves
                       if item.predicate == "timing.wns_ns"]
                if len(wns) != 1 or any(
                    item.predicate not in {"timing.wns_ns", "timing.tns_ns"}
                    for item in all_leaves
                ) or any(item.unit != "ns" for item in all_leaves):
                    return (False, "physical_gate_timing_evidence_mismatch")
                observed = wns[0].value
                if (isinstance(observed, bool)
                        or not isinstance(observed, (int, float))
                        or not math.isfinite(float(observed))
                        or value.get("value") is not (float(observed) >= 0.0)):
                    return (False, "physical_gate_timing_value_mismatch")
            else:
                if (value.get("algorithm") != "hlsgraph.gate.capacity_compare"
                        or str(value.get("algorithm_version")) != "1"):
                    return (False, "physical_gate_algorithm_mismatch")
                usage: dict[str, float] = {}
                for observation in all_leaves:
                    if not observation.predicate.startswith("resource."):
                        return (False, "physical_gate_resource_evidence_mismatch")
                    name = observation.predicate.split(".", 1)[1].lower()
                    observed = observation.value
                    if (name in usage or isinstance(observed, bool)
                            or not isinstance(observed, (int, float))
                            or not math.isfinite(float(observed)) or float(observed) < 0
                            or observation.unit != "count"):
                        return (False, "physical_gate_resource_evidence_mismatch")
                    usage[name] = float(observed)
                capacities = {str(key).lower(): float(item)
                              for key, item in snapshot_manifest.target.capacities.items()}
                reserved = {str(key).lower(): float(item)
                            for key, item in
                            snapshot_manifest.target.reserved_resources.items()}
                if (not capacities or set(usage) != set(capacities)
                        or set(reserved) - set(capacities)):
                    return (False, "physical_gate_capacity_coverage_mismatch")
                expected = all(
                    usage[name] <= capacities[name] - reserved.get(name, 0.0)
                    for name in capacities
                )
                if value.get("value") is not expected:
                    return (False, "physical_gate_resource_value_mismatch")
                if (value.get("metadata", {}).get("target_profile_hash")
                        != stable_hash(snapshot_manifest.target)):
                    return (False, "physical_gate_target_profile_mismatch")
            return (True, "typed_physical_gate_evidence")

        def evidence_trust(
            evidence_id: str, stack: frozenset[tuple[str, str]] = frozenset()
        ) -> tuple[bool, bool, str]:
            candidates = [
                ("observation", observations.get(evidence_id)),
                ("derivation", derivations.get(evidence_id)),
                ("verification", verifications.get(evidence_id)),
                ("diagnostic", diagnostics.get(evidence_id)),
                ("artifact", artifacts.get(evidence_id)),
                ("run", runs.get(evidence_id)),
            ]
            kind, value = next(((kind, value) for kind, value in candidates
                                if value is not None), ("missing", None))
            key = (kind, evidence_id)
            if key in trust_cache:
                return trust_cache[key]
            if value is None or key in stack:
                return (False, False, "missing_or_cyclic_evidence")
            nested = stack | {key}
            if kind == "run":
                result = run_trust(evidence_id)
            elif kind == "artifact":
                result = artifact_trust(evidence_id)
            elif kind == "observation":
                authority = str(value.authority)
                if authority in {"synthetic", "prediction_hypothesis", "knowledge_rule"}:
                    result = (False, False, authority)
                elif str(value.completeness) != "complete":
                    result = (False, False, "incomplete_observation")
                else:
                    refs: list[tuple[bool, bool, str]] = []
                    if value.run_id:
                        refs.append(run_trust(value.run_id))
                    cited_artifacts = {item for item in (
                        value.artifact_id,
                        value.anchor.artifact_id if value.anchor else None,
                    ) if item}
                    refs.extend(artifact_trust(item) for item in sorted(cited_artifacts))
                    produced_by = {
                        artifacts[item].producer_run_id for item in cited_artifacts
                        if item in artifacts and artifacts[item].producer_run_id
                    }
                    lineage_mismatch = bool(produced_by) and (
                        not value.run_id or produced_by != {value.run_id}
                    )
                    if authority in {"tool_observation", "verification_evidence",
                                     "physical_measurement"}:
                        # A manifest-typed report without a producer run remains
                        # visible, but cannot satisfy a real verification gate.
                        if (not value.run_id or produced_by != {value.run_id}
                                or lineage_mismatch):
                            result = (False, False, "observation_producer_mismatch")
                        else:
                            result = combine(refs, require_tool=True)
                    elif authority == "derived_fact":
                        result = (False, False, "derived_observation_without_derivation")
                    elif lineage_mismatch:
                        result = (False, False, "observation_producer_mismatch")
                    elif refs:
                        result = combine(refs)
                    else:
                        result = (True, False, authority)
            elif kind == "derivation":
                authority = str(value.get("authority", ""))
                if authority in {"synthetic", "prediction_hypothesis", "knowledge_rule"}:
                    result = (False, False, authority)
                elif str(value.get("completeness", "missing")) != "complete":
                    result = (False, False, "incomplete_derivation")
                else:
                    result = combine([
                        evidence_trust(str(item), nested)
                        for item in value.get("input_observation_ids", [])
                    ], require_tool=True)
            elif kind == "verification":
                details = value.get("details", {})
                if details.get("fixture_authority") in {"synthetic", "fake"}:
                    result = (False, False, "synthetic")
                else:
                    refs = []
                    run_id = str(value.get("run_id") or "")
                    evidence_ids = [str(item) for item in value.get("evidence_ids", [])]
                    if run_id:
                        referenced_run = runs.get(run_id)
                        expected_stage = {
                            "csim": "csim", "rtl_cosim": "rtl_cosim",
                        }.get(str(value.get("kind")))
                        if (expected_stage and referenced_run
                                and referenced_run.stage != expected_stage):
                            refs.append((False, False, "verification_stage_mismatch"))
                        else:
                            refs.append(run_trust(run_id))
                    refs.extend(evidence_trust(item, nested) for item in evidence_ids)
                    result = combine(refs, require_tool=True)
                    status = str(value.get("status", "unknown"))
                    if status == GateStatus.PASS.value and (not run_id or not evidence_ids):
                        result = (False, False, "passing_verification_missing_run_or_evidence")
                    elif status == GateStatus.PASS.value and any(
                        evidence_kind(item) not in {"observation", "derivation", "artifact"}
                        for item in evidence_ids
                    ):
                        result = (False, False, "passing_verification_has_nonreport_evidence")
                    else:
                        producers: set[str] = set()
                        workloads: set[str] = set()
                        for item in evidence_ids:
                            producers.update(evidence_run_ids(item))
                            workloads.update(evidence_workloads(item))
                        if run_id and producers != {run_id}:
                            result = (False, False, "verification_producer_mismatch")
                        referenced_run = runs.get(run_id) if run_id else None
                        run_workload = (str(referenced_run.metadata.get("workload_id"))
                                        if referenced_run
                                        and referenced_run.metadata.get("workload_id") else "")
                        expected_workload = str(value.get("workload_id") or run_workload)
                        if (expected_workload and workloads - {expected_workload}) or (
                            value.get("workload_id") and run_workload
                            and str(value.get("workload_id")) != run_workload
                        ):
                            result = (False, False, "verification_workload_mismatch")
                        if status == GateStatus.PASS.value and run_id:
                            typed, reason = typed_verification_evidence(value, run_id)
                            if not typed:
                                result = (False, False, reason)
            else:  # diagnostic
                refs = []
                if value.run_id:
                    refs.append(run_trust(value.run_id))
                cited_artifacts = {item for item in (
                    value.artifact_id,
                    value.anchor.artifact_id if value.anchor else None,
                ) if item}
                refs.extend(artifact_trust(item) for item in sorted(cited_artifacts))
                result = combine(refs, require_tool=True)
            trust_cache[key] = result
            return result

        def offer(kind: GateKind, status: str, evidence_id: str,
                  authority: str | None = None,
                  trust: tuple[bool, bool, str] = (False, False, "untrusted")) -> None:
            item = gates[str(kind)]
            item["evidence_ids"].append(evidence_id)
            item["authorities"].append(authority or trust[2])
            # A failure dominates pass; pass dominates unknown.  The source and
            # authority remain visible so synthetic fixtures cannot look real.
            order = {GateStatus.UNKNOWN.value: 0, GateStatus.PASS.value: 1,
                     GateStatus.FAIL.value: 2}
            if order.get(status, 0) > order.get(item["status"], 0):
                item["status"] = status
            if status == GateStatus.PASS.value and trust[0] and trust[1]:
                item["trusted_pass"] = True
                item["tool_truth"] = True

        def offer_check(check: dict[str, Any], status: str, evidence_id: str,
                        authority: str, trust: tuple[bool, bool, str]) -> None:
            check["evidence_ids"].append(evidence_id)
            check["authorities"].append(authority)
            order = {GateStatus.UNKNOWN.value: 0, GateStatus.PASS.value: 1,
                     GateStatus.FAIL.value: 2}
            if order.get(status, 0) > order.get(check["status"], 0):
                check["status"] = status
            if status == GateStatus.PASS.value and trust[0] and trust[1]:
                check["trusted_pass"] = True
                check["tool_truth"] = True

        def empty_check() -> dict[str, Any]:
            return {"status": GateStatus.UNKNOWN.value, "evidence_ids": [],
                    "authorities": [], "trusted_pass": False, "tool_truth": False}

        correctness_campaign_runs: dict[str, dict[str, set[str]]] = {}
        for verification in verifications.values():
            kind = str(verification.get("kind"))
            if kind in {"csim", "rtl_cosim", "assertion", "formal", "mismatch", "deadlock"}:
                trust = evidence_trust(str(verification.get("id")))
                authority = trust[2]
                run = runs.get(str(verification.get("run_id")))
                run_campaign = str(run.metadata.get("campaign_id", "")) if run else ""
                run_workload = str(run.metadata.get("workload_id", "")) if run else ""
                verification_workload = str(verification.get("workload_id") or "")
                if (run_workload and verification_workload
                        and run_workload != verification_workload):
                    trust = (False, False, "verification_workload_mismatch")
                    authority = trust[2]
                workload = verification_workload or run_workload
                if run_campaign or workload:
                    cohort = f"campaign={run_campaign or '-'};workload={workload or '-'}"
                else:
                    # Missing workload/campaign never aliases another verification.
                    cohort = f"unbound:{verification.get('id')}"
                correctness = gates[str(GateKind.CORRECTNESS)]
                checks = correctness.setdefault("checks", {})
                check = checks.setdefault(kind, empty_check())
                status = str(verification.get("status", "unknown"))
                evidence_id = str(verification.get("id"))
                offer_check(check, status, evidence_id, authority, trust)
                campaigns = correctness.setdefault("campaigns", {})
                campaign_check = campaigns.setdefault(cohort, {}).setdefault(
                    kind, empty_check(),
                )
                offer_check(campaign_check, status, evidence_id, authority, trust)
                # This map is consumed by Project.run to prove that the current
                # invocation itself supplied both correctness checks.  Keep only
                # run IDs whose own PASS evidence closed successfully; otherwise
                # a polluted/untrusted row in an already-eligible historical
                # cohort could make a partial rerun look current.
                if (run is not None and kind in {"csim", "rtl_cosim"}
                        and status == GateStatus.PASS.value
                        and trust[0] and trust[1]):
                    correctness_campaign_runs.setdefault(cohort, {}).setdefault(
                        kind, set(),
                    ).add(run.id)
                offer(GateKind.CORRECTNESS, str(verification.get("status", "unknown")),
                      str(verification.get("id")), authority, trust)

        predicate_to_gate = {
            "gate.resource_fits": GateKind.RESOURCE_FITS,
            "gate.post_route_timing": GateKind.POST_ROUTE_TIMING,
        }
        physical_runs_by_gate: dict[GateKind, set[str]] = {
            GateKind.RESOURCE_FITS: set(),
            GateKind.POST_ROUTE_TIMING: set(),
        }
        for derivation in derivations.values():
            kind = predicate_to_gate.get(str(derivation.get("predicate")))
            if not kind:
                continue
            value = derivation.get("value")
            status = (GateStatus.PASS.value if value is True else
                      GateStatus.FAIL.value if value is False else GateStatus.UNKNOWN.value)
            trust = evidence_trust(str(derivation.get("id")))
            producer_ids = evidence_run_ids(str(derivation.get("id")))
            producer_stages = {runs[item].stage for item in producer_ids if item in runs}
            if (str(derivation.get("stage")) != "post_route"
                    or len(producer_ids) != 1 or producer_stages != {"post_route"}):
                trust = (False, False, "physical_gate_producer_stage_mismatch")
            else:
                producer_id = next(iter(producer_ids))
                typed, reason = typed_physical_gate_evidence(derivation, producer_id)
                if not typed:
                    trust = (False, False, reason)
                elif status == GateStatus.PASS.value and trust[0] and trust[1]:
                    physical_runs_by_gate[kind].add(producer_id)
            offer(kind, status, str(derivation.get("id")), trust[2], trust)

        for run in runs.values():
            base_trust = run_trust(run.id)
            authority = base_trust[2]
            for gate in run.gates:
                evidence_trusts = [evidence_trust(item) for item in gate.evidence_ids]
                if gate.status == GateStatus.PASS and not evidence_trusts:
                    trust = (False, False, "passing_gate_without_evidence")
                else:
                    trust = (combine([base_trust, *evidence_trusts], require_tool=True)
                             if evidence_trusts else base_trust)
                producers: set[str] = set()
                for evidence_id in gate.evidence_ids:
                    producers.update(evidence_run_ids(evidence_id))
                if gate.status == GateStatus.PASS and (
                    any(evidence_kind(item) in {"run", "diagnostic", None}
                        for item in gate.evidence_ids)
                    or producers != {run.id}
                ):
                    trust = (False, False, "gate_evidence_producer_mismatch")
                valid_stages = {
                    GateKind.CORRECTNESS: {"csim", "rtl_cosim"},
                    GateKind.RESOURCE_FITS: {"post_route"},
                    GateKind.POST_ROUTE_TIMING: {"post_route"},
                }[gate.kind]
                if run.stage not in valid_stages:
                    trust = (False, False, "gate_stage_mismatch")
                if (gate.status == GateStatus.PASS
                        and gate.kind in physical_runs_by_gate):
                    expected_predicate = {
                        GateKind.RESOURCE_FITS: "gate.resource_fits",
                        GateKind.POST_ROUTE_TIMING: "gate.post_route_timing",
                    }[gate.kind]
                    physical_derivations = [
                        derivations.get(str(evidence_id))
                        for evidence_id in gate.evidence_ids
                    ]
                    if (
                        not physical_derivations
                        or any(
                            value is None
                            or value.get("predicate") != expected_predicate
                            or not typed_physical_gate_evidence(value, run.id)[0]
                            for value in physical_derivations
                        )
                    ):
                        trust = (False, False, "gate_typed_physical_evidence_mismatch")
                    elif trust[0] and trust[1]:
                        physical_runs_by_gate[gate.kind].add(run.id)
                offer(gate.kind, str(gate.status), run.id, authority, trust)

        for item in gates.values():
            if not isinstance(item, dict):
                continue
            item["evidence_ids"] = sorted(set(item["evidence_ids"]))
            item["authorities"] = sorted(set(item["authorities"]))
            item["synthetic_only"] = bool(item["authorities"]) and set(item["authorities"]) == {"synthetic"}
            for check in item.get("checks", {}).values():
                check["evidence_ids"] = sorted(set(check["evidence_ids"]))
                check["authorities"] = sorted(set(check["authorities"]))
            for campaign in item.get("campaigns", {}).values():
                for check in campaign.values():
                    check["evidence_ids"] = sorted(set(check["evidence_ids"]))
                    check["authorities"] = sorted(set(check["authorities"]))
        correctness = gates[str(GateKind.CORRECTNESS)]
        required_correctness = ("csim", "rtl_cosim")
        correctness["required_checks"] = list(required_correctness)
        eligible_campaigns = sorted(
            cohort for cohort, checks in correctness.get("campaigns", {}).items()
            if not cohort.startswith("unbound:") and all(
                checks.get(kind, {}).get("status") == GateStatus.PASS.value
                and checks.get(kind, {}).get("trusted_pass") is True
                for kind in required_correctness
            )
        )
        correctness["eligible_campaigns"] = eligible_campaigns
        correctness["eligible_run_ids"] = {
            cohort: {
                kind: sorted(correctness_campaign_runs.get(cohort, {}).get(kind, set()))
                for kind in required_correctness
            }
            for cohort in eligible_campaigns
        }
        correctness["required_checks_met"] = bool(eligible_campaigns)
        eligible_physical_runs = sorted(
            physical_runs_by_gate[GateKind.RESOURCE_FITS]
            & physical_runs_by_gate[GateKind.POST_ROUTE_TIMING]
        )
        gates["eligible_physical_runs"] = eligible_physical_runs
        gates["verified"] = all(
            gates[str(kind)]["status"] == GateStatus.PASS.value
            and gates[str(kind)]["trusted_pass"] is True
            and gates[str(kind)]["tool_truth"] is True
            for kind in (GateKind.CORRECTNESS, GateKind.RESOURCE_FITS,
                         GateKind.POST_ROUTE_TIMING)
        ) and correctness["required_checks_met"] and bool(eligible_physical_runs)
        gates["verification_policy"] = {
            "correctness_requires": list(required_correctness),
            "correctness_campaign_policy": (
                "csim and rtl_cosim must share an explicit campaign_id and/or workload_id"
            ),
            "physical_run_policy": (
                "resource_fits and post_route_timing must come from the same trusted "
                "post_route run"
            ),
            "independent_gates": [str(GateKind.CORRECTNESS), str(GateKind.RESOURCE_FITS),
                                  str(GateKind.POST_ROUTE_TIMING)],
        }
        return gates

    def compare(self, other_snapshot_id: str) -> dict[str, Any]:
        left = self.graph()
        right = self.bundle.store.load_graph(other_snapshot_id)

        def entity_map(graph: CanonicalGraph) -> dict[tuple[str, str], list[dict[str, Any]]]:
            grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
            for item in graph.entities.values():
                key = (item.kind, item.qualified_name or item.name)
                grouped.setdefault(key, []).append(self._semantic_payload(
                    item, {"id", "snapshot_id", "kind", "name", "qualified_name"},
                ))
            return {key: sorted(values, key=lambda value: json.dumps(
                value, ensure_ascii=False, sort_keys=True
            )) for key, values in grouped.items()}

        left_keys = entity_map(left)
        right_keys = entity_map(right)
        left_by_id = {item.id: (item.kind, item.qualified_name or item.name)
                      for item in left.entities.values()}
        right_by_id = {item.id: (item.kind, item.qualified_name or item.name)
                       for item in right.entities.values()}

        def relation_map(
            graph: CanonicalGraph, identities: dict[str, tuple[str, str]]
        ) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
            grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
            for item in graph.relations.values():
                key = (item.kind, identities.get(item.src, ("unmapped", item.src)),
                       identities.get(item.dst, ("unmapped", item.dst)))
                grouped.setdefault(key, []).append(self._semantic_payload(
                    item, {"id", "snapshot_id", "src", "dst", "kind"},
                ))
            return {key: sorted(values, key=lambda value: json.dumps(
                value, ensure_ascii=False, sort_keys=True
            )) for key, values in grouped.items()}

        left_relations = relation_map(left, left_by_id)
        right_relations = relation_map(right, right_by_id)

        def observation_map(
            snapshot_id: str, identities: dict[str, tuple[str, str]]
        ) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
            grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
            for item in self.bundle.store.observations(snapshot_id):
                key = (identities.get(item.subject_id, ("unmapped", item.subject_id)),
                       item.predicate, item.stage, str(item.authority), item.unit,
                       item.workload_id)
                grouped.setdefault(key, []).append(self._semantic_payload(
                    item, {"id", "snapshot_id", "subject_id", "predicate", "stage",
                           "authority", "unit", "workload_id"},
                ))
            return {key: sorted(values, key=lambda value: json.dumps(
                json_ready(value), ensure_ascii=False, sort_keys=True
            )) for key, values in grouped.items()}

        left_observations = observation_map(self.snapshot_id, left_by_id)
        right_observations = observation_map(other_snapshot_id, right_by_id)
        def artifact_map(snapshot_id: str) -> dict[str, list[dict[str, Any]]]:
            grouped: dict[str, list[dict[str, Any]]] = {}
            for item in self.bundle.store.artifacts(snapshot_id):
                grouped.setdefault(item.uri, []).append(self._semantic_payload(
                    item, {"id", "uri"},
                ))
            return {uri: sorted(values, key=lambda value: json.dumps(
                value, ensure_ascii=False, sort_keys=True
            )) for uri, values in grouped.items()}

        left_artifacts = artifact_map(self.snapshot_id)
        right_artifacts = artifact_map(other_snapshot_id)
        return {
            "schema_version": SCHEMA_VERSION,
            "left_snapshot_id": self.snapshot_id,
            "right_snapshot_id": other_snapshot_id,
            "entities_added": [list(key) for key in sorted(right_keys.keys() - left_keys.keys())],
            "entities_removed": [list(key) for key in sorted(left_keys.keys() - right_keys.keys())],
            "entities_changed": [list(key) for key in sorted(left_keys.keys() & right_keys.keys())
                                 if left_keys[key] != right_keys[key]],
            "relations_added": [json_ready(key) for key in sorted(right_relations.keys() - left_relations.keys())],
            "relations_removed": [json_ready(key) for key in sorted(left_relations.keys() - right_relations.keys())],
            "relations_changed": [json_ready(key) for key in sorted(
                left_relations.keys() & right_relations.keys())
                if left_relations[key] != right_relations[key]],
            "observations_added": [json_ready(key) for key in sorted(
                right_observations.keys() - left_observations.keys(), key=str)],
            "observations_removed": [json_ready(key) for key in sorted(
                left_observations.keys() - right_observations.keys(), key=str)],
            "observations_changed": [json_ready(key) for key in sorted(
                left_observations.keys() & right_observations.keys(), key=str)
                if left_observations[key] != right_observations[key]],
            "artifacts_added": sorted(right_artifacts.keys() - left_artifacts.keys()),
            "artifacts_removed": sorted(left_artifacts.keys() - right_artifacts.keys()),
            "artifacts_changed": sorted(
                uri for uri in left_artifacts.keys() & right_artifacts.keys()
                if left_artifacts[uri] != right_artifacts[uri]
            ),
            "left_gates": self.verification_gates(),
            "right_gates": CoreService(self.bundle, other_snapshot_id).verification_gates(),
        }

    @staticmethod
    def _semantic_payload(value: Any, excluded: set[str]) -> dict[str, Any]:
        payload = dict(json_ready(value))
        for key in excluded:
            payload.pop(key, None)
        return payload
