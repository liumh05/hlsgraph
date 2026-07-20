"""Optional FastMCP registration for the read-only HLSGraph facade."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .service import ReadOnlyMcpService


def create_mcp(project_root: str | Path, *, snapshot_id: str | None = None) -> Any:
    """Create a FastMCP server, importing the optional dependency only on demand."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise RuntimeError("MCP support requires `pip install hlsgraph[mcp]`") from exc

    tools = ReadOnlyMcpService(project_root, snapshot_id=snapshot_id)
    mcp = FastMCP(
        "hlsgraph",
        instructions=("Read-only deterministic HLS architecture facts. Never treat software calls "
                      "as hardware topology or predictions as synthesis observations."),
    )

    @mcp.tool()
    def overview(depth: int = 1, top_k: int = 12) -> dict[str, Any]:
        """Summarize architecture, evidence coverage, health, and staleness."""
        return tools.overview(depth=depth, top_k=top_k)

    @mcp.tool()
    def search(query: str, kinds: list[str] | None = None, scope_id: str | None = None,
               stages: list[str] | None = None, authorities: list[str] | None = None,
               limit: int = 20, cursor: str | None = None) -> dict[str, Any]:
        """Search canonical entities with stable pagination."""
        return tools.search(query, kinds or (), scope_id, stages or (), authorities or (), limit, cursor)

    @mcp.tool()
    def context(query: str | None = None, scope_id: str | None = None,
                depth: int = 1, top_k: int = 8, cursor: str | None = None) -> dict[str, Any]:
        """Read bounded graph context, observations, diagnostics, and evidence metadata."""
        return tools.context(query, scope_id, depth, top_k, cursor)

    @mcp.tool()
    def module_or_region(identifier: str, depth: int = 2) -> dict[str, Any]:
        """Resolve and inspect one kernel, module, process, function, or region."""
        return tools.module_or_region(identifier, depth)

    @mcp.tool()
    def traverse(entity_id: str, depth: int = 1, direction: str = "both",
                 relation_kinds: list[str] | None = None) -> dict[str, Any]:
        """Traverse explicit graph relations only."""
        return tools.traverse(entity_id, depth, direction, relation_kinds or ())

    @mcp.tool()
    def impact(entity_id: str, depth: int = 2,
               relation_kinds: list[str] | None = None) -> dict[str, Any]:
        """Report downstream dependency facts without inventing QoR changes."""
        return tools.impact(entity_id, depth, relation_kinds or ())

    @mcp.tool()
    def evidence(entity_id: str) -> dict[str, Any]:
        """Trace an entity to observations and artifact metadata."""
        return tools.evidence(entity_id)

    @mcp.tool()
    def feature_evidence(entity_id: str | None = None,
                         predicates: list[str] | None = None,
                         stages: list[str] | None = None,
                         limit: int = 100) -> dict[str, Any]:
        """Read opt-in deterministic feature evidence without outcome data."""
        return tools.feature_evidence(
            entity_id, predicates or (), stages or (), limit,
        )

    @mcp.tool()
    def correspondences(entity_id: str | None = None,
                        other_snapshot_id: str | None = None,
                        kinds: list[str] | None = None,
                        direction: str = "both",
                        limit: int = 100) -> dict[str, Any]:
        """Read explicit entity mappings; ambiguous candidates remain unresolved."""
        return tools.correspondences(
            entity_id, other_snapshot_id, kinds or (), direction, limit,
        )

    @mcp.tool()
    def compare(other_snapshot_id: str) -> dict[str, Any]:
        """Compare the active snapshot with another immutable snapshot."""
        return tools.compare(other_snapshot_id)

    @mcp.tool()
    def health() -> dict[str, Any]:
        """Read parser/report/tool diagnostics and stale state."""
        return tools.health()

    @mcp.tool()
    def runs(stage: str | None = None, status: str | None = None,
             limit: int = 50) -> dict[str, Any]:
        """Read redacted immutable tool-run records, including failures."""
        return tools.runs(stage, status, limit)

    @mcp.tool()
    def predictions(subject_id: str | None = None, predicate: str | None = None,
                    model_id: str | None = None, limit: int = 50) -> dict[str, Any]:
        """Read model predictions as hypotheses, never as tool observations."""
        return tools.predictions(subject_id, predicate, model_id, limit)

    @mcp.tool()
    def variants(parent_snapshot_id: str | None = None,
                 action_id: str | None = None, limit: int = 50) -> dict[str, Any]:
        """Read proposed actions and their explicitly recorded lineage only."""
        return tools.variants(parent_snapshot_id, action_id, limit)

    @mcp.tool()
    def render(scope_id: str | None = None, format: str = "mermaid",
               max_chars: int = 500_000) -> dict[str, Any]:
        """Render the graph in memory without changing files or facts."""
        return tools.render(scope_id, format, max_chars)

    @mcp.tool()
    def knowledge(query: str | None = None, document_id: str | None = None,
                  document_version: str | None = None, vendor: str | None = None,
                  tool: str | None = None, tool_version: str | None = None,
                  stage: str | None = None, limit: int = 20) -> dict[str, Any]:
        """Read versioned rules as guidance, separately from design observations."""
        return tools.knowledge(query, document_id, document_version, vendor, tool,
                               tool_version, stage, limit)

    return mcp


def run_stdio(project_root: str | Path, *, snapshot_id: str | None = None) -> None:
    """Run the optional MCP stdio transport until the client disconnects."""
    create_mcp(project_root, snapshot_id=snapshot_id).run(transport="stdio")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HLSGraph read-only MCP server")
    parser.add_argument("project_root", nargs="?", default=".")
    parser.add_argument("--snapshot-id")
    args = parser.parse_args(argv)
    run_stdio(args.project_root, snapshot_id=args.snapshot_id)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["create_mcp", "main", "run_stdio"]
