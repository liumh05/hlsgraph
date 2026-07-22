"""Instance-local identity fields for deterministically scoped directives.

The fields produced here are deliberately redundant with the ``hls.annotates``
relation.  Knowledge retrieval must decide applicability from one directive (or
one directive observation) at a time; it must not join a directive to a
similarly named loop, variable, or port while answering a query.
"""
from __future__ import annotations

from typing import Any

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
) -> Entity | None:
    """Resolve one DEPENDENCE operand inside the exact enclosing function.

    This uses explicit ``hls.contains`` ownership and an exact source symbol
    spelling.  It never falls back to a repository-global same-name match.
    Shadowing or missing containment therefore remains ambiguous.
    """
    if scope is None or not isinstance(variable_name, str) or not variable_name:
        return None
    if scope.kind in {"hls.kernel", "hls.function"}:
        owners = [scope]
    elif scope.kind == "hls.loop":
        frontier = {scope.id}
        visited = set(frontier)
        owner_ids: set[str] = set()
        while frontier:
            parents = {
                relation.src for relation in graph.relations.values()
                if relation.kind == "hls.contains" and relation.dst in frontier
                and relation.src in graph.entities
            } - visited
            visited.update(parents)
            found = {
                item for item in parents
                if graph.entities[item].kind in {"hls.kernel", "hls.function"}
            }
            owner_ids.update(found)
            # Stop each ancestry path at its nearest function-like owner, but
            # continue unresolved sibling paths so a second owner cannot be
            # hidden behind a different containment depth.
            frontier = parents - found
        owners = [graph.entities[item] for item in sorted(owner_ids)]
    else:
        return None
    if len(owners) != 1:
        return None
    owner = owners[0]
    descendants = {owner.id}
    frontier = {owner.id}
    while frontier:
        children = {
            relation.dst for relation in graph.relations.values()
            if relation.kind == "hls.contains" and relation.src in frontier
            and relation.dst in graph.entities
        } - descendants
        descendants.update(children)
        frontier = children
    allowed_kinds = {"hls.memory", "hls.stream", "hls.port", "source.variable"}
    matches = [
        graph.entities[item] for item in sorted(descendants)
        if graph.entities[item].kind in allowed_kinds
        and graph.entities[item].name == variable_name
    ]
    return matches[0] if len(matches) == 1 else None


def directive_identity_metadata(directive: Entity) -> dict[str, Any]:
    """Project a directive's explicit identity into one observation record."""
    result: dict[str, Any] = {}
    for key in DIRECTIVE_IDENTITY_FIELDS:
        value = directive.attrs.get(key)
        if isinstance(value, str) and value:
            result[key] = value
    return result
