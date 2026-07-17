"""External Tcl/config directive extraction and declared-precedence resolution."""
from __future__ import annotations

import re
import shlex
from collections import defaultdict
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
    Observation,
    Relation,
    SourceAnchor,
    Stage,
    stable_hash,
)
from .base import ExtractionContext, ExtractionResult


_TCL_DIRECTIVE = re.compile(r"^\s*set_directive_([A-Za-z0-9_]+)\s+(.+?)\s*$", re.I)
_CONFIG_DIRECTIVE = re.compile(r"^\s*(?:syn\.)?directive\.([A-Za-z0-9_]+)\s*=\s*(.+?)\s*$", re.I)


def _strip_comment(text: str) -> str:
    """Drop external-file comments before storing directive semantics."""
    quote: str | None = None
    brace_depth = 0
    escaped = False
    for index, char in enumerate(text):
        following = text[index + 1] if index + 1 < len(text) else ""
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
        elif char == "{":
            brace_depth += 1
        elif char == "}" and brace_depth:
            brace_depth -= 1
        elif not brace_depth and char == "/" and following == "/":
            return text[:index].rstrip()
        elif not brace_depth and char == "#" and (
                index == 0 or text[index - 1].isspace() or text[index - 1] == ";"):
            return text[:index].rstrip(" ;")
    return text.rstrip()


def _value(value: str) -> Any:
    value = value.strip().strip('{}"')
    try:
        return int(value, 0)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            if value.lower() in {"true", "false"}:
                return value.lower() == "true"
            return value


def _tcl_tokens(text: str) -> list[str]:
    try:
        return shlex.split(text.replace("{", '"').replace("}", '"'), posix=True)
    except ValueError:
        return text.split()


def _parse_options(tokens: list[str]) -> tuple[dict[str, Any], str | None]:
    options: dict[str, Any] = {}
    scope: str | None = None
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.startswith("-"):
            key = token.lstrip("-").lower()
            if index + 1 < len(tokens) and not tokens[index + 1].startswith("-"):
                options[key] = _value(tokens[index + 1])
                index += 2
            else:
                options[key] = True
                index += 1
        else:
            scope = token.strip('{}"')
            index += 1
    return options, scope


def _normalize_scope(value: str) -> str:
    return value.strip().strip('{}"').replace("\\", "/").strip("/")


def _resolve_scope(graph: CanonicalGraph, scope: str | None) -> Entity | None:
    if not scope:
        kernels = [item for item in graph.entities.values() if item.kind == "hls.kernel"]
        return kernels[0] if len(kernels) == 1 else None
    scope = _normalize_scope(scope)
    leaf = scope.split("/")[-1]
    exact = [item for item in graph.entities.values()
             if item.id == scope or item.qualified_name == scope or scope in item.aliases]
    if len(exact) == 1:
        return exact[0]
    matches = [item for item in graph.entities.values()
               if item.name == leaf and (len(scope.split("/")) == 1 or
                                         (item.qualified_name or "").replace("::", "/").endswith(scope))]
    return matches[0] if len(matches) == 1 else None


class ExternalDirectiveExtractor:
    name = "directive.external"
    version = "1"

    def supports(self, context: ExtractionContext) -> bool:
        return bool(context.manifest.build.tcl_files or context.manifest.build.config_files)

    def extract(self, context: ExtractionContext) -> ExtractionResult:
        graph = CanonicalGraph(snapshot_id=context.snapshot.id)
        result = ExtractionResult(graph=graph, capabilities=["directive.external"])
        existing: CanonicalGraph = context.options.get("existing_graph") or graph
        sources = [(path, "tcl", 30) for path in context.manifest.build.tcl_files]
        sources += [(path, "config", 20) for path in context.manifest.build.config_files]
        count = 0
        for relative, origin, precedence in sources:
            artifact = context.artifact_for_uri(relative)
            path = project_path(context.project_root, relative)
            if not artifact or not path.is_file():
                result.diagnostics.append(Diagnostic(
                    snapshot_id=context.snapshot.id, code="directive.input_missing",
                    severity=DiagnosticSeverity.ERROR,
                    message=f"directive input is missing from snapshot: {relative}",
                    stage=Stage.SOURCE.value, artifact_id=artifact.id if artifact else None,
                ))
                continue
            for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                line = _strip_comment(line)
                directive_kind: str | None = None
                options: dict[str, Any] = {}
                scope: str | None = None
                tcl_match = _TCL_DIRECTIVE.match(line)
                config_match = _CONFIG_DIRECTIVE.match(line)
                if tcl_match:
                    directive_kind = tcl_match.group(1).upper()
                    options, scope = _parse_options(_tcl_tokens(tcl_match.group(2)))
                elif config_match:
                    directive_kind = config_match.group(1).upper()
                    fields = [item.strip() for item in config_match.group(2).split(",") if item.strip()]
                    if fields and "=" not in fields[0]:
                        scope = fields.pop(0)
                    for field in fields:
                        if "=" in field:
                            key, raw = field.split("=", 1)
                            options[key.strip().lower()] = _value(raw)
                if not directive_kind:
                    continue
                count += 1
                anchor = SourceAnchor(artifact_id=artifact.id, start_line=line_number,
                                      start_column=1, end_line=line_number,
                                      end_column=len(line) + 1)
                target = _resolve_scope(existing, scope)
                directive = Entity(
                    kind="hls.directive", name=directive_kind,
                    qualified_name=f"{relative}:{line_number}:{directive_kind}",
                    snapshot_id=context.snapshot.id,
                    authority=AuthorityClass.DECLARED_CONSTRAINT,
                    stage=Stage.SOURCE.value,
                    attrs={"directive_kind": directive_kind, "options": options,
                           "scope_text": scope, "origin": origin,
                           "precedence": precedence, "state": "requested"},
                    anchors=[anchor],
                    completeness=(Completeness.COMPLETE if target
                                  else Completeness.AMBIGUOUS),
                )
                graph.add_entity(directive)
                if target:
                    graph.add_relation(Relation(
                        src=directive.id, dst=target.id, kind="hls.annotates",
                        snapshot_id=context.snapshot.id,
                        authority=AuthorityClass.DECLARED_CONSTRAINT,
                        stage=Stage.SOURCE.value,
                        attrs={"scope_node_id": target.id, "scope_text": scope,
                               "scope_resolution": "external_exact"}, anchors=[anchor],
                    ), allow_dangling=True)
                    result.observations.append(Observation(
                        snapshot_id=context.snapshot.id, subject_id=directive.id,
                        predicate="directive.requested", value=options or True,
                        stage=Stage.SOURCE.value,
                        authority=AuthorityClass.DECLARED_CONSTRAINT,
                        artifact_id=artifact.id, anchor=anchor,
                        metadata={"directive_kind": directive_kind, "scope_id": target.id,
                                  "origin": origin, "precedence": precedence},
                    ))
                else:
                    result.diagnostics.append(Diagnostic(
                        snapshot_id=context.snapshot.id, code="directive.unresolved_scope",
                        severity=DiagnosticSeverity.WARNING,
                        message=f"could not deterministically resolve external scope {scope!r}",
                        stage=Stage.SOURCE.value, subject_id=directive.id,
                        artifact_id=artifact.id, anchor=anchor,
                    ))
        result.coverage = {"external_directives": count, "policy": "declared_precedence_v1"}
        return result


def resolve_directives(result: ExtractionResult) -> None:
    """Resolve only declared precedence; this never claims a tool applied a directive."""
    graph = result.graph
    annotations: dict[str, Relation] = {
        relation.src: relation for relation in graph.relations.values()
        if relation.kind == "hls.annotates"
    }
    groups: dict[tuple[str, str], list[Entity]] = defaultdict(list)
    for entity in graph.entities.values():
        if entity.kind != "hls.directive" or entity.id not in annotations:
            continue
        kind = str(entity.attrs.get("directive_kind", entity.name)).upper()
        groups[(annotations[entity.id].dst, kind)].append(entity)
    existing_effective = {(item.subject_id, item.predicate) for item in result.observations}
    for (scope_id, kind), directives in groups.items():
        directives.sort(key=lambda item: (int(item.attrs.get("precedence", 0)), item.id))
        highest = max(int(item.attrs.get("precedence", 0)) for item in directives)
        top = [item for item in directives
               if int(item.attrs.get("precedence", 0)) == highest]
        top_values = {stable_hash(item.attrs.get("options") or True) for item in top}
        if len(top) > 1 and len(top_values) > 1:
            for item in top:
                item.attrs["state"] = "conflicting_declared"
                item.completeness = Completeness.AMBIGUOUS
            for overridden in [item for item in directives if item not in top]:
                overridden.attrs["state"] = "overridden_declared"
            result.diagnostics.append(Diagnostic(
                snapshot_id=top[0].snapshot_id,
                code="directive.ambiguous_same_precedence",
                severity=DiagnosticSeverity.WARNING,
                message=(f"{kind} on {scope_id} has conflicting declarations at the "
                         "same precedence; no effective declaration was inferred"),
                stage=Stage.SOURCE.value, subject_id=scope_id,
                metadata={"candidate_directive_ids": sorted(item.id for item in top),
                          "resolution_policy": "hlsgraph.declared_precedence_v1"},
            ))
            continue
        winner = top[-1]
        winner.attrs["state"] = "effective_declared"
        key = (winner.id, "directive.effective")
        if key not in existing_effective:
            anchor = winner.anchors[0] if winner.anchors else None
            result.observations.append(Observation(
                snapshot_id=winner.snapshot_id, subject_id=winner.id,
                predicate="directive.effective", value=winner.attrs.get("options") or True,
                stage=Stage.SOURCE.value, authority=AuthorityClass.DECLARED_CONSTRAINT,
                artifact_id=anchor.artifact_id if anchor else None, anchor=anchor,
                metadata={"directive_kind": kind, "scope_id": scope_id,
                          "resolution_policy": "hlsgraph.declared_precedence_v1",
                          "tool_applied": False},
            ))
        for overridden in directives[:-1]:
            overridden.attrs["state"] = "overridden_declared"
            result.diagnostics.append(Diagnostic(
                snapshot_id=overridden.snapshot_id, code="directive.declared_override",
                severity=DiagnosticSeverity.INFO,
                message=f"{kind} on {scope_id} is superseded by a higher-precedence declaration; tool application remains unverified",
                stage=Stage.SOURCE.value, subject_id=overridden.id,
                metadata={"winner_directive_id": winner.id,
                          "resolution_policy": "hlsgraph.declared_precedence_v1"},
            ))
