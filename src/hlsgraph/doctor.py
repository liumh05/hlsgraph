"""Low-cost, read-only environment diagnostics used by the CLI.

The doctor deliberately only discovers modules, executables, and bundle metadata.  It never
invokes vendor tools, opens an SSH connection, or mutates a project.
"""
from __future__ import annotations

import importlib.util
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

from .bundle import BundleError, GraphBundle
from .version import __version__


def _check(name: str, status: str, message: str, **details: Any) -> dict[str, Any]:
    return {"name": name, "status": status, "message": message, **details}


def diagnose(project_root: str | Path | None = None) -> dict[str, Any]:
    """Return deterministic-enough diagnostics without executing discovered programs."""
    checks: list[dict[str, Any]] = []
    supported = sys.version_info >= (3, 10)
    checks.append(_check(
        "python", "pass" if supported else "fail",
        f"Python {platform.python_version()}", executable=sys.executable,
    ))

    for module, purpose in (
        ("clang.cindex", "standard source/AST extraction"),
        ("mcp", "MCP server"),
        ("pyarrow", "Parquet/Arrow ML export"),
    ):
        try:
            available = importlib.util.find_spec(module) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            available = False
        checks.append(_check(
            f"module:{module}", "pass" if available else "warn",
            f"available for {purpose}" if available else f"optional dependency missing: {purpose}",
        ))

    for executable, purpose in (
        ("clang", "C/C++ parsing diagnostics"),
        ("vitis_hls", "AMD HLS execution"),
        ("vivado", "AMD implementation execution"),
        ("ssh", "SSH runner"),
    ):
        path = shutil.which(executable)
        checks.append(_check(
            f"executable:{executable}", "pass" if path else "warn",
            f"found for {purpose}" if path else f"not found (only needed for {purpose})",
            path=path,
        ))

    if project_root is not None:
        root = Path(project_root).resolve()
        try:
            bundle = GraphBundle.open(root)
        except (BundleError, OSError, ValueError) as exc:
            checks.append(_check("bundle", "fail", str(exc), project_root=str(root)))
        else:
            status = bundle.status()
            checks.append(_check(
                "bundle", "pass", "HLSGraph bundle is readable", project_root=str(root),
                project_id=status["project_id"], snapshot_id=status["snapshot_id"],
                stale=status["stale"],
            ))
            if status["snapshot_id"] is None:
                checks.append(_check("snapshot", "warn", "project has not been indexed"))
            elif status["stale"]:
                checks.append(_check("snapshot", "warn", "indexed snapshot is stale"))
            else:
                checks.append(_check("snapshot", "pass", "indexed snapshot matches current inputs"))

    counts = {level: sum(item["status"] == level for item in checks)
              for level in ("pass", "warn", "fail")}
    return {
        "hlsgraph_version": __version__,
        "platform": platform.platform(),
        "checks": checks,
        "summary": counts,
        "healthy": counts["fail"] == 0,
        "notes": [
            "Executable checks use PATH discovery only; no vendor tool or SSH command was run.",
            "Missing optional tools are warnings because indexing from existing artifacts remains valid.",
        ],
    }


__all__ = ["diagnose"]
