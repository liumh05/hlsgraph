"""Optional MCP SDK 2 registration for the read-only HLSGraph facade."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Literal, cast

from ..retrieval import RetrievalResult
from ..version import __version__
from .service import ReadOnlyMcpService


LEGACY_TOOL_NAMES = (
    "overview", "search", "context", "module_or_region", "traverse",
    "impact", "evidence", "feature_evidence", "correspondences", "compare",
    "health", "runs", "predictions", "variants", "render", "knowledge",
)


def _register_legacy_tools(
    mcp: Any, tools: ReadOnlyMcpService, annotations: Any,
) -> None:
    """Register the v0.2 narrow tools only under explicit operator opt-in."""

    @mcp.tool(annotations=annotations)
    def overview(depth: int = 1, top_k: int = 12) -> dict[str, Any]:
        """Summarize architecture, evidence coverage, health, and staleness."""
        return tools.overview(depth=depth, top_k=top_k)

    @mcp.tool(annotations=annotations)
    def search(query: str, kinds: list[str] | None = None, scope_id: str | None = None,
               stages: list[str] | None = None, authorities: list[str] | None = None,
               limit: int = 20, cursor: str | None = None) -> dict[str, Any]:
        """Search canonical entities with stable pagination."""
        return tools.search(query, kinds or (), scope_id, stages or (), authorities or (), limit, cursor)

    @mcp.tool(annotations=annotations)
    def context(query: str | None = None, scope_id: str | None = None,
                depth: int = 1, top_k: int = 8, cursor: str | None = None) -> dict[str, Any]:
        """Read bounded graph context, observations, diagnostics, and evidence metadata."""
        return tools.context(query, scope_id, depth, top_k, cursor)

    @mcp.tool(annotations=annotations)
    def module_or_region(identifier: str, depth: int = 2) -> dict[str, Any]:
        """Resolve and inspect one kernel, module, process, function, or region."""
        return tools.module_or_region(identifier, depth)

    @mcp.tool(annotations=annotations)
    def traverse(entity_id: str, depth: int = 1, direction: str = "both",
                 relation_kinds: list[str] | None = None) -> dict[str, Any]:
        """Traverse explicit graph relations only."""
        return tools.traverse(entity_id, depth, direction, relation_kinds or ())

    @mcp.tool(annotations=annotations)
    def impact(entity_id: str, depth: int = 2,
               relation_kinds: list[str] | None = None) -> dict[str, Any]:
        """Report downstream dependency facts without inventing QoR changes."""
        return tools.impact(entity_id, depth, relation_kinds or ())

    @mcp.tool(annotations=annotations)
    def evidence(entity_id: str) -> dict[str, Any]:
        """Trace an entity to observations and artifact metadata."""
        return tools.evidence(entity_id)

    @mcp.tool(annotations=annotations)
    def feature_evidence(entity_id: str | None = None,
                         predicates: list[str] | None = None,
                         stages: list[str] | None = None,
                         limit: int = 100) -> dict[str, Any]:
        """Read opt-in deterministic feature evidence without outcome data."""
        return tools.feature_evidence(entity_id, predicates or (), stages or (), limit)

    @mcp.tool(annotations=annotations)
    def correspondences(entity_id: str | None = None,
                        other_snapshot_id: str | None = None,
                        kinds: list[str] | None = None,
                        direction: str = "both", limit: int = 100) -> dict[str, Any]:
        """Read explicit entity mappings; ambiguous candidates remain unresolved."""
        return tools.correspondences(entity_id, other_snapshot_id, kinds or (), direction, limit)

    @mcp.tool(annotations=annotations)
    def compare(other_snapshot_id: str) -> dict[str, Any]:
        """Compare the active snapshot with another immutable snapshot."""
        return tools.compare(other_snapshot_id)

    @mcp.tool(annotations=annotations)
    def health() -> dict[str, Any]:
        """Read parser/report/tool diagnostics and stale state."""
        return tools.health()

    @mcp.tool(annotations=annotations)
    def runs(stage: str | None = None, status: str | None = None,
             limit: int = 50) -> dict[str, Any]:
        """Read redacted immutable tool-run records, including failures."""
        return tools.runs(stage, status, limit)

    @mcp.tool(annotations=annotations)
    def predictions(subject_id: str | None = None, predicate: str | None = None,
                    model_id: str | None = None, limit: int = 50) -> dict[str, Any]:
        """Read model predictions as hypotheses, never as tool observations."""
        return tools.predictions(subject_id, predicate, model_id, limit)

    @mcp.tool(annotations=annotations)
    def variants(parent_snapshot_id: str | None = None,
                 action_id: str | None = None, limit: int = 50) -> dict[str, Any]:
        """Read proposed actions and their explicitly recorded lineage only."""
        return tools.variants(parent_snapshot_id, action_id, limit)

    @mcp.tool(annotations=annotations)
    def render(scope_id: str | None = None, format: str = "mermaid",
               max_chars: int = 500_000) -> dict[str, Any]:
        """Render the graph in memory without changing files or facts."""
        return tools.render(scope_id, format, max_chars)

    @mcp.tool(annotations=annotations)
    def knowledge(query: str | None = None, document_id: str | None = None,
                  document_version: str | None = None, vendor: str | None = None,
                  tool: str | None = None, tool_version: str | None = None,
                  stage: str | None = None, limit: int = 20) -> dict[str, Any]:
        """Read versioned rules as guidance, separately from design observations."""
        return tools.knowledge(query, document_id, document_version, vendor, tool,
                               tool_version, stage, limit)


def create_mcp(project_root: str | Path, *, snapshot_id: str | None = None) -> Any:
    """Create an MCP SDK 2 server with one strong retrieval tool by default."""
    try:
        from mcp.server import MCPServer
        from mcp.types import CallToolResult, TextContent, ToolAnnotations
    except ImportError as exc:  # pragma: no cover - optional environment
        raise RuntimeError("MCP support requires `pip install hlsgraph[mcp]`") from exc

    mode = os.environ.get("HLSGRAPH_MCP_TOOLS", "explore").strip().casefold() or "explore"
    if mode not in {"explore", "all"}:
        raise ValueError("HLSGRAPH_MCP_TOOLS must be 'explore' or 'all'")
    tools = ReadOnlyMcpService(project_root, snapshot_id=snapshot_id)
    annotations = ToolAnnotations(readOnlyHint=True, openWorldHint=False)
    mcp = MCPServer(
        "hlsgraph",
        version=__version__,
        instructions=(
            "Use explore first for one bounded, evidence-backed HLS answer. Facts, guidance, "
            "and predictions are separate truth planes. Software calls and LLVM CFG are not "
            "hardware topology. Re-read source only when explore reports stale, incomplete, "
            "ambiguous, or low-confidence context."
        ),
    )

    @mcp.tool(annotations=annotations)
    def explore(query: str, snapshot_id: str | None = None,
                scope_id: str | None = None,
                view: Literal["architecture", "evidence"] = "architecture",
                max_chars: int | None = None,
                include_private_snippets: bool = False,
                include_predictions: bool = False) -> RetrievalResult:
        """Retrieve ranked HLS facts, evidence, flow, and applicable guidance in one call."""
        result = tools.explore_result(
            query, snapshot_id, scope_id, view, max_chars,
            include_private_snippets, include_predictions,
        )
        payload = result.to_dict()
        content = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        )
        if len(content) > result.trace.output_budget_chars:  # pragma: no cover
            raise RuntimeError("retrieval output exceeded its declared character budget")
        # The return annotation supplies the SDK's strong output schema.  The
        # explicit protocol result keeps the LLM-facing text compact while the
        # same value remains available to hosts as structuredContent.
        return cast(RetrievalResult, CallToolResult(
            content=[TextContent(type="text", text=content)],
            structuredContent=payload,
        ))

    if mode == "all":
        _register_legacy_tools(mcp, tools, annotations)
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


__all__ = ["LEGACY_TOOL_NAMES", "create_mcp", "main", "run_stdio"]
