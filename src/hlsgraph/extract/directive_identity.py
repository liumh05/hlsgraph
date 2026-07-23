"""Instance-local identity fields for deterministically scoped directives.

The fields produced here are deliberately redundant with the ``hls.annotates``
relation.  Knowledge retrieval must decide applicability from one directive (or
one directive observation) at a time; it must not join a directive to a
similarly named loop, variable, or port while answering a query.
"""
from __future__ import annotations

from typing import Any, Mapping

from ..graph import CanonicalGraph
from ..model import Entity


DIRECTIVE_IDENTITY_FIELDS = (
    "directive_instance_id",
    "scope_id",
    "scope_kind",
    "scope_resolution",
    "function_id",
    "loop_id",
    "variable_id",
    "port_id",
)


def bind_directive_identity(
    directive: Entity,
    target: Entity | None,
    *,
    scope_resolution: str | None,
    operand_target: Entity | None = None,
) -> None:
    """Attach only extractor-proven instance and scope identities.

    No symbol lookup is performed here. ``target`` must already be the unique
    entity selected by the source/Tcl/config extractor. For DEPENDENCE,
    ``target`` is the enclosing loop/function and ``operand_target`` is the
    separately resolved variable. Missing or ambiguous identities intentionally
    remain absent so downstream knowledge bindings fail closed.
    """
    directive.attrs["directive_instance_id"] = directive.id
    directive_kind = str(
        directive.attrs.get("directive_kind", directive.name)
    ).upper()
    if (directive_kind == "DEPENDENCE"
            and target is not None
            and target.kind not in {"hls.kernel", "hls.function", "hls.loop"}):
        # DEPENDENCE constrains scheduling in an enclosing function/loop.  Its
        # named variable is an operand, never the directive scope itself.
        target = None
    if target is None:
        return
    directive.attrs["scope_id"] = target.id
    directive.attrs["scope_kind"] = target.kind
    if scope_resolution:
        directive.attrs["scope_resolution"] = scope_resolution
    if target.kind in {"hls.kernel", "hls.function"}:
        directive.attrs["function_id"] = target.id
    elif target.kind == "hls.loop":
        directive.attrs["loop_id"] = target.id
    elif target.kind == "hls.port":
        directive.attrs["port_id"] = target.id
        # An array-valued port is also the concrete variable named by storage
        # directives such as ARRAY_PARTITION, STREAM, or DEPENDENCE.
        directive.attrs["variable_id"] = target.id
    elif target.kind in {"hls.memory", "hls.stream"}:
        directive.attrs["variable_id"] = target.id
    if (directive_kind == "DEPENDENCE" and operand_target is not None
            and operand_target.kind in {
                "hls.memory", "hls.stream", "hls.port", "source.variable",
            }
            and operand_target.id != target.id):
        directive.attrs["variable_id"] = operand_target.id


def resolve_directive_variable_operand(
    graph: CanonicalGraph,
    scope: Entity | None,
    variable_name: Any,
    *,
    artifact_id: str,
    pragma_line: int,
    pragma_scope_path: tuple[str, ...] | None,
    entity_lexical_paths: Mapping[str, tuple[str, ...]],
) -> Entity | None:
    """Resolve one DEPENDENCE operand inside the exact enclosing function.

    This uses explicit ``hls.contains`` ownership and an exact source symbol
    spelling.  It never falls back to a repository-global same-name match.
    Shadowing or missing containment therefore remains ambiguous.
    """
    def static_entity(entity: Entity) -> bool:
        return bool(
            entity.snapshot_id == graph.snapshot_id
            and entity.stage == "ast"
            and str(entity.authority) == "static_fact"
            and str(entity.completeness) == "complete"
        )

    def parent(entity_id: str) -> Entity | None:
        relations = [
            relation for relation in graph.relations.values()
            if relation.kind == "hls.contains" and relation.dst == entity_id
        ]
        if len(relations) != 1:
            return None
        relation = relations[0]
        value = graph.entities.get(relation.src)
        if (
            value is None
            or relation.snapshot_id != graph.snapshot_id
            or relation.stage != "ast"
            or str(relation.authority) != "static_fact"
            or str(relation.completeness) != "complete"
            or not static_entity(value)
        ):
            return None
        return value

    if (
        scope is None
        or not static_entity(scope)
        or not isinstance(variable_name, str)
        or not variable_name
        or pragma_scope_path is None
    ):
        return None
    if scope.kind in {"hls.kernel", "hls.function"}:
        owners = [scope]
    elif scope.kind == "hls.loop":
        current = scope
        visited: set[str] = set()
        while current.kind not in {"hls.kernel", "hls.function"}:
            if current.id in visited:
                return None
            visited.add(current.id)
            value = parent(current.id)
            if value is None:
                return None
            current = value
        owners = [current]
    else:
        return None
    if len(owners) != 1:
        return None
    owner = owners[0]
    descendants = {owner.id}
    frontier = {owner.id}
    while frontier:
        children: set[str] = set()
        for relation in graph.relations.values():
            if (
                relation.kind != "hls.contains"
                or relation.src not in frontier
                or relation.dst not in graph.entities
                or relation.snapshot_id != graph.snapshot_id
                or relation.stage != "ast"
                or str(relation.authority) != "static_fact"
                or str(relation.completeness) != "complete"
                or parent(relation.dst) is None
            ):
                continue
            children.add(relation.dst)
        children -= descendants
        descendants.update(children)
        frontier = children
    allowed_kinds = {"hls.memory", "hls.stream", "hls.port", "source.variable"}
    matches: list[Entity] = []
    for item in sorted(descendants):
        candidate = graph.entities[item]
        if (
            candidate.kind not in allowed_kinds
            or candidate.name != variable_name
            or not static_entity(candidate)
        ):
            continue
        if candidate.kind == "hls.port":
            if parent(candidate.id) == owner:
                matches.append(candidate)
            continue
        starts = [
            anchor.start_line for anchor in candidate.anchors
            if anchor.artifact_id == artifact_id
            and anchor.start_line is not None
        ]
        declaration_path = entity_lexical_paths.get(candidate.id)
        if (
            starts
            and min(starts) < pragma_line
            and declaration_path is not None
            and len(declaration_path) <= len(pragma_scope_path)
            and pragma_scope_path[:len(declaration_path)] == declaration_path
        ):
            matches.append(candidate)
    return matches[0] if len(matches) == 1 else None


def directive_identity_metadata(directive: Entity) -> dict[str, Any]:
    """Project a directive's explicit identity into one observation record."""
    result: dict[str, Any] = {}
    for key in DIRECTIVE_IDENTITY_FIELDS:
        value = directive.attrs.get(key)
        if isinstance(value, str) and value:
            result[key] = value
    return result
