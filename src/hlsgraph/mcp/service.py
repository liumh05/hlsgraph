"""Read-only MCP facade over the canonical HLSGraph query service."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..bundle import GraphBundle
from ..diagnostic_projection import public_diagnostic, redacted_diagnostic_record
from ..model import json_ready
from ..query import DEFAULT_IMPACT_RELATIONS, CoreService, ExploreSpec, QuerySpec
from ..render import render as render_graph
from ..run_projection import (
    PUBLIC_FAILURE_CLASSES, PUBLIC_GATE_KINDS, PUBLIC_GATE_STATUSES,
    PUBLIC_RUN_STATUSES, public_enum, public_identifier,
    public_identifier_list, public_sha256, public_timestamp,
    sanitize_run_metadata,
)
from ..sdk import Project
from ..version import SCHEMA_VERSION


def _service(value: Project | GraphBundle | CoreService | str | Path,
             snapshot_id: str | None = None) -> CoreService:
    if isinstance(value, CoreService):
        if snapshot_id and snapshot_id != value.snapshot_id:
            return CoreService(value.bundle, snapshot_id=snapshot_id)
        return value
    if isinstance(value, Project):
        return value.service(snapshot_id)
    if isinstance(value, GraphBundle):
        return CoreService(value, snapshot_id=snapshot_id)
    return Project.open(value).service(snapshot_id)


def _bundle(value: Project | GraphBundle | CoreService | str | Path) -> GraphBundle:
    if isinstance(value, CoreService):
        return value.bundle
    if isinstance(value, Project):
        return value.bundle
    if isinstance(value, GraphBundle):
        return value
    return Project.open(value).bundle


def _run_summary(run: Any) -> dict[str, Any]:
    """Agent-safe run state without argv, paths, messages, or raw output."""
    ready = json_ready(run)
    value = dict(ready) if isinstance(ready, Mapping) else {}
    elapsed = value.get("elapsed_s")
    result = {
        "id": public_identifier(value.get("id")),
        "snapshot_id": public_identifier(value.get("snapshot_id")),
        "stage": public_identifier(value.get("stage")),
        "backend": public_identifier(value.get("backend")),
        "request_hash": public_sha256(value.get("request_hash")),
        "toolchain_id": public_identifier(value.get("toolchain_id")),
        "status": public_enum(value.get("status"), PUBLIC_RUN_STATUSES),
        "environment_hash": public_sha256(value.get("environment_hash")),
        "input_artifact_ids": public_identifier_list(value.get("input_artifact_ids")),
        "output_artifact_ids": public_identifier_list(value.get("output_artifact_ids")),
        "diagnostics": public_identifier_list(value.get("diagnostics")),
        "failure_class": public_enum(
            value.get("failure_class"), PUBLIC_FAILURE_CLASSES
        ),
        "exit_code": (value.get("exit_code") if isinstance(value.get("exit_code"), int)
                      and not isinstance(value.get("exit_code"), bool) else None),
        "attempt": (value.get("attempt") if isinstance(value.get("attempt"), int)
                    and not isinstance(value.get("attempt"), bool)
                    and value.get("attempt") >= 1 else None),
        "started_at": public_timestamp(value.get("started_at")),
        "finished_at": public_timestamp(value.get("finished_at")),
        "elapsed_s": (elapsed if isinstance(elapsed, (int, float))
                      and not isinstance(elapsed, bool) and math.isfinite(float(elapsed))
                      and elapsed >= 0 else None),
        "metadata": sanitize_run_metadata(value.get("metadata")),
        "command_redacted": True,
        "working_directory_redacted": True,
        "message_present": value.get("message") is not None,
    }
    gates = value.get("gates") if isinstance(value.get("gates"), list) else []
    result["gates"] = [{
        "kind": public_enum(item.get("kind"), PUBLIC_GATE_KINDS),
        "status": public_enum(item.get("status"), PUBLIC_GATE_STATUSES),
        "evidence_ids": public_identifier_list(item.get("evidence_ids")),
        "reason_redacted": item.get("reason") is not None,
    } for item in gates if isinstance(item, Mapping)]
    return result


class ReadOnlyMcpService:
    """Agent-facing tools; all facts remain backed by CoreService or ledger rows."""

    def __init__(self, project: Project | GraphBundle | CoreService | str | Path,
                 *, snapshot_id: str | None = None):
        self.bundle = _bundle(project)
        try:
            self.core: CoreService | None = _service(project, snapshot_id)
        except ValueError:
            self.core = None
        if self.core is not None and not self.bundle.store.has_graph(self.core.snapshot_id):
            self.core = None
        candidate = (self.bundle.store.snapshot(snapshot_id) if snapshot_id else
                     self.bundle.latest_snapshot() or
                     self.bundle.store.latest_candidate(self.bundle.manifest.project_id))
        self.snapshot_id = self.core.snapshot_id if self.core else (
            candidate.id if candidate else None
        )

    def _require_core(self) -> CoreService:
        if self.core is None:
            raise ValueError(
                "bundle has no successful canonical graph; use health/runs for candidate diagnostics"
            )
        return self.core

    def overview(self, depth: int = 1, top_k: int = 12) -> dict[str, Any]:
        """Summarize indexed architecture, health, evidence coverage, and staleness."""
        if self.core is None:
            return {
                "schema_version": SCHEMA_VERSION,
                "snapshot_id": self.snapshot_id,
                "status": self.health()["status"],
                "architecture": None,
                "incomplete": True,
                "message": "no successful canonical graph is available",
            }
        architecture = self.core.explore(ExploreSpec(
            view="architecture", depth=max(0, min(int(depth), 8)),
            top_k=max(1, min(int(top_k), 50)),
        )).to_dict()
        return {
            "schema_version": SCHEMA_VERSION,
            "snapshot_id": self.core.snapshot_id,
            "status": self.core.status().to_dict(),
            "architecture": architecture,
        }

    def search(self, query: str, kinds: Iterable[str] = (), scope_id: str | None = None,
               stages: Iterable[str] = (), authorities: Iterable[str] = (),
               limit: int = 20, cursor: str | None = None) -> dict[str, Any]:
        """Search canonical entities through the shared exact/FTS/fuzzy query chain."""
        return self._require_core().query(QuerySpec(
            query=query, kinds=list(kinds), scope_id=scope_id, stages=list(stages),
            authorities=list(authorities), limit=max(1, min(int(limit), 100)), cursor=cursor,
        )).to_dict()

    def context(self, query: str | None = None, scope_id: str | None = None,
                depth: int = 1, top_k: int = 8, cursor: str | None = None) -> dict[str, Any]:
        """Return a bounded graph neighborhood plus observations and evidence metadata."""
        if not query and not scope_id:
            raise ValueError("context requires query or scope_id")
        return self._require_core().explore(ExploreSpec(
            query=query, scope_id=scope_id, view="architecture",
            depth=max(0, min(int(depth), 8)), top_k=max(1, min(int(top_k), 20)),
            cursor=cursor,
        )).to_dict()

    def module_or_region(self, identifier: str, depth: int = 2) -> dict[str, Any]:
        """Resolve an ID/name through CoreService and return its architecture neighborhood."""
        core = self._require_core()
        graph = core.graph()
        entity_id = identifier if identifier in graph.entities else None
        if entity_id is None:
            hits = core.query(QuerySpec(query=identifier, limit=20)).items
            preferred = (
                "hls.kernel", "hls.component", "hls.module", "hls.region", "hls.process",
            )

            def semantic_priority(item: Any) -> tuple[int, int]:
                kind = item.kind.casefold()
                if kind in preferred:
                    kind_rank = preferred.index(kind)
                elif kind.startswith("hls."):
                    kind_rank = len(preferred)
                elif kind.startswith("ir.") and kind.endswith(".function"):
                    kind_rank = len(preferred) + 1
                else:
                    kind_rank = len(preferred) + 2
                exact = 0 if item.name.casefold() == identifier.casefold() else 1
                return (kind_rank, exact)

            hit = None
            if hits:
                best_priority = min(semantic_priority(item) for item in hits)
                best = [item for item in hits if semantic_priority(item) == best_priority]
                qualified = [item for item in best
                             if (item.qualified_name or "").casefold() == identifier.casefold()]
                if qualified:
                    best = qualified
                if len(best) != 1:
                    candidates = ", ".join(sorted(item.entity_id for item in best))
                    raise ValueError(
                        f"ambiguous identifier {identifier!r}; use one stable entity ID: {candidates}"
                    )
                hit = best[0]
            entity_id = hit.entity_id if hit else None
        if entity_id is None:
            raise KeyError(identifier)
        return core.explore(ExploreSpec(
            scope_id=entity_id, view="architecture", depth=max(0, min(int(depth), 8)),
        )).to_dict()

    def traverse(self, entity_id: str, depth: int = 1, direction: str = "both",
                 relation_kinds: Iterable[str] = ()) -> dict[str, Any]:
        """Traverse explicit canonical relations without inferring missing hardware edges."""
        return self._require_core().traverse(entity_id, depth=depth, direction=direction,
                                  relation_kinds=relation_kinds)

    def impact(self, entity_id: str, depth: int = 2,
               relation_kinds: Iterable[str] = ()) -> dict[str, Any]:
        """Report deterministic downstream dependencies, never fabricated QoR deltas."""
        return self._require_core().impact(entity_id, depth=depth, relation_kinds=relation_kinds)

    def evidence(self, entity_id: str) -> dict[str, Any]:
        """Trace one entity to observations, anchors, artifact metadata, and diagnostics."""
        return self._require_core().evidence(entity_id)

    def feature_evidence(self, entity_id: str | None = None,
                         predicates: Iterable[str] = (),
                         stages: Iterable[str] = (),
                         limit: int = 100) -> dict[str, Any]:
        """Read selected deterministic feature evidence, never labels or predictions."""
        return self._require_core().feature_evidence(
            entity_id, predicates=predicates, stages=stages,
            limit=max(1, min(int(limit), 1000)),
        )

    def correspondences(self, entity_id: str | None = None,
                        other_snapshot_id: str | None = None,
                        kinds: Iterable[str] = (), direction: str = "both",
                        limit: int = 100) -> dict[str, Any]:
        """Read explicit entity mappings and surface ambiguous candidate groups."""
        return self._require_core().correspondences(
            entity_id, other_snapshot_id=other_snapshot_id,
            kinds=kinds, direction=direction,
            limit=max(1, min(int(limit), 1000)),
        )

    def compare(self, other_snapshot_id: str) -> dict[str, Any]:
        """Compare the selected immutable snapshot with another ledger snapshot."""
        return self._require_core().compare(other_snapshot_id)

    def health(self) -> dict[str, Any]:
        """Expose degraded parsing, missing reports, failures, and bundle staleness."""
        if self.snapshot_id is None:
            return {
                "schema_version": SCHEMA_VERSION, "snapshot_id": None,
                "status": self.bundle.status(), "diagnostics": [], "runs": [],
            }
        diagnostics = self.bundle.store.diagnostics(self.snapshot_id)
        status = (self.core.status().to_dict() if self.core
                  else self.bundle.status(self.snapshot_id))
        return {
            "schema_version": SCHEMA_VERSION,
            "snapshot_id": self.snapshot_id,
            "status": status,
            "diagnostics": [public_diagnostic(item) for item in diagnostics],
            "runs": [_run_summary(item) for item in self.bundle.store.runs(self.snapshot_id)],
        }

    def runs(self, stage: str | None = None, status: str | None = None,
             limit: int = 50) -> dict[str, Any]:
        """Read redacted immutable run-ledger entries, including failed candidates."""
        if self.snapshot_id is None:
            return {"schema_version": SCHEMA_VERSION, "snapshot_id": None,
                    "items": [], "truncated": False}
        values = self.bundle.store.runs(self.snapshot_id)
        if stage:
            values = [item for item in values if item.stage == stage]
        if status:
            values = [item for item in values if str(item.status) == status]
        limit = max(1, min(int(limit), 100))
        return {
            "schema_version": SCHEMA_VERSION, "snapshot_id": self.snapshot_id,
            "items": [_run_summary(item) for item in values[:limit]],
            "truncated": len(values) > limit,
        }

    def predictions(self, subject_id: str | None = None,
                    predicate: str | None = None, model_id: str | None = None,
                    limit: int = 50) -> dict[str, Any]:
        """Read prediction envelopes separately from facts and observations."""
        if self.snapshot_id is None:
            return {"schema_version": SCHEMA_VERSION, "snapshot_id": None,
                    "authority_class": "prediction_hypothesis",
                    "items": [], "truncated": False}
        values = self.bundle.store.predictions(self.snapshot_id)
        if subject_id:
            values = [item for item in values if item.get("subject_id") == subject_id]
        if predicate:
            values = [item for item in values if item.get("predicate") == predicate]
        if model_id:
            values = [item for item in values if item.get("model_id") == model_id]
        limit = max(1, min(int(limit), 100))
        return {
            "schema_version": SCHEMA_VERSION, "snapshot_id": self.snapshot_id,
            "authority_class": "prediction_hypothesis",
            "items": values[:limit], "truncated": len(values) > limit,
        }

    def variants(self, parent_snapshot_id: str | None = None,
                 action_id: str | None = None, limit: int = 50) -> dict[str, Any]:
        """Read proposed actions and explicit prediction/result links without mutation."""
        if self.snapshot_id is None:
            return {
                "schema_version": SCHEMA_VERSION,
                "snapshot_id": None,
                "parent_snapshot_id": parent_snapshot_id,
                "record_class": "variant_action",
                "lineage_semantics": "recorded_links_only",
                "items": [],
                "truncated": False,
            }
        core = self.core or CoreService(self.bundle, self.snapshot_id)
        result = core.variants(
            parent_snapshot_id=parent_snapshot_id, action_id=action_id,
        )
        values = result["items"]
        limit = max(1, min(int(limit), 100))
        return {
            **result,
            "items": values[:limit],
            "truncated": len(values) > limit,
        }

    def render(self, scope_id: str | None = None, format: str = "mermaid",
               max_chars: int = 500_000) -> dict[str, Any]:
        """Render in memory; this tool never writes files or mutates the graph."""
        if format not in {"html", "json", "mermaid", "dot", "svg"}:
            raise ValueError("format must be html, json, mermaid, dot, or svg")
        if not 1_000 <= int(max_chars) <= 5_000_000:
            raise ValueError("max_chars must be in 1000..5000000")
        core = self._require_core()
        graph = core.graph()
        diagnostics = [
            projected for item in self.bundle.store.active_diagnostics(core.snapshot_id)
            if (projected := redacted_diagnostic_record(item)) is not None
        ]
        content = render_graph(
            graph, format=format, scope_id=scope_id,
            observations=self.bundle.store.observations(core.snapshot_id),
            diagnostics=diagnostics,
        )
        if len(content) > int(max_chars):
            raise ValueError(
                f"render output is {len(content)} characters; raise max_chars explicitly or use SDK/REST"
            )
        media_types = {
            "html": "text/html", "json": "application/json", "mermaid": "text/plain",
            "dot": "text/vnd.graphviz", "svg": "image/svg+xml",
        }
        return {
            "schema_version": SCHEMA_VERSION,
            "snapshot_id": core.snapshot_id,
            "format": format,
            "media_type": media_types[format],
            "content": content,
        }

    def knowledge(self, query: str | None = None, document_id: str | None = None,
                  document_version: str | None = None, vendor: str | None = None,
                  tool: str | None = None, tool_version: str | None = None,
                  stage: str | None = None, limit: int = 20) -> dict[str, Any]:
        """Read versioned knowledge rules; rules are guidance, not design observations."""
        rules = self.bundle.store.knowledge_rules()
        from ..knowledge import filter_rules
        applicability = {key: value for key, value in {
            "vendor": vendor, "tool": tool, "tool_version": tool_version, "stage": stage,
        }.items() if value is not None}
        rules = filter_rules(
            rules, document_id=document_id, document_version=document_version,
            applicability=applicability or None,
        )
        if query:
            folded = query.casefold()
            rules = [item for item in rules if folded in " ".join(filter(None, [
                item.id, item.title, item.summary, item.section,
            ])).casefold()]
        limit = max(1, min(int(limit), 100))
        return {
            "schema_version": SCHEMA_VERSION,
            "snapshot_id": self.snapshot_id,
            "authority_class": "knowledge_rule",
            "applicability_context": applicability,
            "items": [{"id": item.id, **json_ready(item)} for item in rules[:limit]],
            "truncated": len(rules) > limit,
        }


__all__ = ["DEFAULT_IMPACT_RELATIONS", "ReadOnlyMcpService"]
