"""Human-facing render projection; never used as the canonical SDK/ML graph."""
from __future__ import annotations

import json
from typing import Iterable

from ..graph import CanonicalGraph
from ..model import Diagnostic, Observation
from .projection import to_render_data
from .html import to_html
from .static import to_dot, to_mermaid, to_svg


def render(graph: CanonicalGraph, *, format: str = "html", scope_id: str | None = None,
           observations: Iterable[Observation] = (), diagnostics: Iterable[Diagnostic] = ()) -> str:
    data = to_render_data(graph, list(observations), list(diagnostics), scope_id=scope_id)
    if format == "html":
        return to_html(data)
    if format == "json":
        return json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if format == "mermaid":
        return to_mermaid(data)
    if format == "dot":
        return to_dot(data)
    if format == "svg":
        return to_svg(data)
    raise ValueError("format must be html, json, mermaid, dot, or svg")


__all__ = ["render", "to_render_data"]

