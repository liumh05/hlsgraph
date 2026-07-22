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
from .directive_identity import (
    bind_directive_identity,
    directive_identity_metadata,
    resolve_directive_variable_operand,
)


_TCL_DIRECTIVE = re.compile(r"^\s*set_directive_([A-Za-z0-9_]+)\s+(.+?)\s*$", re.I)
_TCL_DIRECTIVE_MARKER = re.compile(r"\bset_directive_[A-Za-z0-9_]*", re.I)
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


def _advance_tcl_lexical_state(
    text: str, state: tuple[str, ...] = (),
) -> tuple[tuple[str, ...], bool]:
    """Advance a conservative Tcl ``brace/quote/bracket`` context stack.

    The stack is intentionally lexical rather than evaluative.  An open frame
    means the next physical line cannot be proven to start a top-level command.
    ``malformed`` records an unmatched closing delimiter and permanently moves
    the file into fail-closed mode.
    """
    frames = list(state)
    escaped = False
    malformed = False
    for char in text:
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        current = frames[-1] if frames else None
        if current == "brace":
            if char == "{":
                frames.append("brace")
            elif char == "}":
                frames.pop()
            continue
        if current == "quote":
            if char == '"':
                frames.pop()
            elif char == "[":
                # Command substitution is parsed as a nested Tcl script and
                # returns to the surrounding quote after its closing bracket.
                frames.append("bracket")
            continue

        # Top-level script text and bracket command substitutions share the
        # same word-level delimiter rules.
        if char == "{":
            frames.append("brace")
        elif char == '"':
            frames.append("quote")
        elif char == "[":
            frames.append("bracket")
        elif char == "]":
            if current == "bracket":
                frames.pop()
            else:
                malformed = True
        elif char == "}":
            malformed = True
    return tuple(frames), malformed


def _tcl_line_profile(text: str) -> dict[str, Any]:
    """Return the small lexical profile needed by the literal Tcl policy.

    This is deliberately not a Tcl interpreter.  Its only purpose is to prove
    that a candidate is one complete, top-level command made from literal
    words.  Anything requiring Tcl evaluation remains diagnostic-only.
    """
    brace_depth = 0
    minimum_brace_depth = 0
    quote = False
    escaped = False
    semicolon = False
    substitution = False
    unsupported_quote = False
    backslash = False
    for char in text:
        if escaped:
            escaped = False
            continue
        if char == "\\":
            backslash = True
            escaped = True
            continue
        if quote:
            if char == '"':
                quote = False
            elif char in {"$", "["}:
                substitution = True
            continue
        if char == '"' and brace_depth == 0:
            quote = True
        elif char == "{":
            brace_depth += 1
        elif char == "}":
            brace_depth -= 1
            minimum_brace_depth = min(minimum_brace_depth, brace_depth)
        elif brace_depth == 0 and char == ";":
            semicolon = True
        elif brace_depth == 0 and char in {"$", "["}:
            substitution = True
        elif brace_depth == 0 and char == "'":
            # A single quote has no grouping semantics in Tcl, while shlex
            # would interpret it as a quote.  Reject it rather than parse a
            # different command from the one Tcl would execute.
            unsupported_quote = True
    lexical_state, lexical_malformed = _advance_tcl_lexical_state(text)
    return {
        "brace_delta": brace_depth,
        "minimum_brace_depth": minimum_brace_depth,
        "quote_unclosed": quote,
        "semicolon": semicolon,
        "substitution": substitution,
        "unsupported_quote": unsupported_quote,
        "backslash": backslash,
        "lexical_open": bool(lexical_state),
        "lexical_malformed": lexical_malformed,
    }


def _tcl_literal_rejection(
    text: str, *, lexical_context_open: bool, continued_from_previous: bool,
    structure_uncertain: bool,
) -> str | None:
    """Explain why a Tcl directive cannot be asserted as a declaration."""
    if structure_uncertain:
        return "uncertain_script_structure"
    if continued_from_previous:
        return "continued_command"
    if lexical_context_open:
        return "nested_script_context"
    if not _TCL_DIRECTIVE.match(text):
        return "embedded_or_constructed_command"
    profile = _tcl_line_profile(text)
    if (profile["lexical_open"] or profile["lexical_malformed"]
            or profile["brace_delta"] != 0
            or profile["minimum_brace_depth"] < 0
            or profile["quote_unclosed"]):
        return "incomplete_command"
    if profile["semicolon"]:
        return "multiple_commands"
    if profile["substitution"] or re.search(r"(?:^|\s)\{\*\}", text):
        return "dynamic_substitution"
    if profile["backslash"]:
        return "escape_or_continuation"
    if profile["unsupported_quote"]:
        return "unsupported_quoting"
    return None


def _tcl_continues(text: str) -> bool:
    stripped = text.rstrip()
    trailing = len(stripped) - len(stripped.rstrip("\\"))
    return bool(trailing % 2)


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
        ambiguous_tcl = 0
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
            tcl_lexical_state: tuple[str, ...] = ()
            tcl_continued = False
            tcl_structure_uncertain = False
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            for line_number, raw_line in enumerate(lines, 1):
                line = _strip_comment(raw_line)
                if origin == "tcl":
                    lexical_state_before = tcl_lexical_state
                    continued_from_previous = tcl_continued
                    uncertain_before = tcl_structure_uncertain
                    tcl_lexical_state, malformed = _advance_tcl_lexical_state(
                        line, tcl_lexical_state,
                    )
                    if malformed:
                        tcl_structure_uncertain = True
                    # Continuation is a physical-line property.  Computing it
                    # after stripping a comment would incorrectly promote the
                    # next line of ``# disabled \\`` as a fresh command.
                    tcl_continued = _tcl_continues(raw_line)
                    if _TCL_DIRECTIVE_MARKER.search(line):
                        rejection = _tcl_literal_rejection(
                            line, lexical_context_open=bool(lexical_state_before),
                            continued_from_previous=continued_from_previous,
                            structure_uncertain=uncertain_before,
                        )
                        if rejection:
                            ambiguous_tcl += 1
                            anchor = SourceAnchor(
                                artifact_id=artifact.id, start_line=line_number,
                                start_column=1, end_line=line_number,
                                end_column=len(line) + 1,
                            )
                            result.diagnostics.append(Diagnostic(
                                snapshot_id=context.snapshot.id,
                                code="directive.tcl_nonliteral_context",
                                severity=DiagnosticSeverity.WARNING,
                                message=("a possible Tcl directive was not imported because "
                                         "v0.1 only accepts complete, top-level literal commands"),
                                stage=Stage.SOURCE.value, artifact_id=artifact.id,
                                anchor=anchor,
                                id=("diagnostic_" + stable_hash({
                                    "snapshot": context.snapshot.id,
                                    "code": "directive.tcl_nonliteral_context",
                                    "artifact": artifact.id,
                                    "line": line_number,
                                    "reason": rejection,
                                })[:24]),
                                metadata={
                                    "reason": rejection,
                                    "completeness": Completeness.AMBIGUOUS.value,
                                    "parse_policy": "hlsgraph.tcl_literal_top_level_v1",
                                },
                            ))
                            continue
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
                if (directive_kind == "DEPENDENCE" and target is not None
                        and target.kind not in {
                            "hls.kernel", "hls.function", "hls.loop",
                        }):
                    target = None
                operand_target = (
                    resolve_directive_variable_operand(
                        existing, target, options.get("variable"),
                    )
                    if directive_kind == "DEPENDENCE" else None
                )
                identity_complete = bool(
                    target is not None
                    and (directive_kind != "DEPENDENCE" or operand_target is not None)
                )
                directive = Entity(
                    kind="hls.directive", name=directive_kind,
                    qualified_name=f"{relative}:{line_number}:{directive_kind}",
                    snapshot_id=context.snapshot.id,
                    authority=AuthorityClass.DECLARED_CONSTRAINT,
                    stage=Stage.SOURCE.value,
                    attrs={"directive_kind": directive_kind, "options": options,
                           "scope_text": scope, "origin": origin,
                           "precedence": precedence, "state": "requested",
                           "parse_policy": (
                               "hlsgraph.tcl_literal_top_level_v1"
                               if origin == "tcl" else "hlsgraph.config_literal_v1"
                           )},
                    anchors=[anchor],
                    completeness=(Completeness.COMPLETE if identity_complete
                                  else Completeness.AMBIGUOUS),
                )
                bind_directive_identity(
                    directive, target, scope_resolution="external_exact" if target else None,
                    operand_target=operand_target,
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
                        completeness=directive.completeness,
                        metadata={"directive_kind": directive_kind,
                                  **directive_identity_metadata(directive),
                                  "origin": origin, "precedence": precedence,
                                  "parse_policy": directive.attrs["parse_policy"]},
                    ))
                else:
                    result.diagnostics.append(Diagnostic(
                        snapshot_id=context.snapshot.id, code="directive.unresolved_scope",
                        severity=DiagnosticSeverity.WARNING,
                        message=f"could not deterministically resolve external scope {scope!r}",
                        stage=Stage.SOURCE.value, subject_id=directive.id,
                        artifact_id=artifact.id, anchor=anchor,
                    ))
                if (target is not None and directive_kind == "DEPENDENCE"
                        and operand_target is None):
                    result.diagnostics.append(Diagnostic(
                        snapshot_id=context.snapshot.id,
                        code="directive.unresolved_operand",
                        severity=DiagnosticSeverity.WARNING,
                        message=("could not deterministically resolve DEPENDENCE operand "
                                 f"{options.get('variable')!r} inside scope {scope!r}"),
                        stage=Stage.SOURCE.value, subject_id=directive.id,
                        artifact_id=artifact.id, anchor=anchor,
                    ))
        result.coverage = {
            "external_directives": count,
            "ambiguous_tcl_directives": ambiguous_tcl,
            "policy": "declared_precedence_v1",
            "tcl_parse_policy": "hlsgraph.tcl_literal_top_level_v1",
        }
        return result


def resolve_directives(result: ExtractionResult) -> None:
    """Resolve only declared precedence; this never claims a tool applied a directive."""
    graph = result.graph
    annotation_candidates: dict[str, list[Relation]] = defaultdict(list)
    for relation in graph.relations.values():
        if relation.kind == "hls.annotates":
            annotation_candidates[relation.src].append(relation)
    annotations = {
        directive_id: values[0]
        for directive_id, values in annotation_candidates.items()
        if len(values) == 1
    }
    groups: dict[tuple[str, str, str], list[Entity]] = defaultdict(list)
    for entity in graph.entities.values():
        if entity.kind != "hls.directive" or entity.id not in annotations:
            continue
        kind = str(entity.attrs.get("directive_kind", entity.name)).upper()
        operand_id = ""
        if kind == "DEPENDENCE":
            raw_operand = entity.attrs.get("variable_id")
            if (str(entity.completeness) != "complete"
                    or not isinstance(raw_operand, str) or not raw_operand):
                # An unresolved operand cannot participate in precedence or
                # become a selected declaration for another variable.
                continue
            operand_id = raw_operand
        groups[(annotations[entity.id].dst, kind, operand_id)].append(entity)
    existing_selected = {(item.subject_id, item.predicate) for item in result.observations}
    for (scope_id, kind, _operand_id), directives in groups.items():
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
                         "same precedence; no selected declaration was inferred"),
                stage=Stage.SOURCE.value, subject_id=scope_id,
                metadata={"candidate_directive_ids": sorted(item.id for item in top),
                          "resolution_policy": "hlsgraph.declared_precedence_v1"},
            ))
            continue
        winner = top[-1]
        winner.attrs["state"] = "selected_declared"
        key = (winner.id, "directive.declared_selected")
        if key not in existing_selected:
            anchor = winner.anchors[0] if winner.anchors else None
            result.observations.append(Observation(
                snapshot_id=winner.snapshot_id, subject_id=winner.id,
                predicate="directive.declared_selected",
                value=winner.attrs.get("options") or True,
                stage=Stage.SOURCE.value, authority=AuthorityClass.DECLARED_CONSTRAINT,
                artifact_id=anchor.artifact_id if anchor else None, anchor=anchor,
                completeness=winner.completeness,
                metadata={"directive_kind": kind,
                          **directive_identity_metadata(winner),
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
