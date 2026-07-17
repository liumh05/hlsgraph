from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = ROOT / "tools" / "audit_release.py"
SPEC = importlib.util.spec_from_file_location("hlsgraph_release_audit", AUDIT_PATH)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


@pytest.mark.parametrize("payload", [
    b"hlsgraph" + b"-research",
    b"research" + b"-integration.md",
    b"HLS" + b"Pilot",
    b"Timely" + b"HLS",
    b"G" + b"NN",
    b"R" + b"CD",
    b"control" + b"ler",
    b"agent" + b"ic",
    b"196472" + b"2203@qq.com",
])
def test_non_public_roadmap_markers_are_rejected(payload: bytes) -> None:
    assert AUDIT._scan("docs/public.md", payload)
    assert AUDIT._scan(
        "docs/" + payload.decode("ascii") + ".md",
        b"public content",
    )


def test_audited_vendor_symbol_does_not_trigger_short_marker() -> None:
    issues = AUDIT._scan(
        "src/hlsgraph/render/vendor/elk.bundled.js",
        b"function r" + b"Cd(){}",
    )
    assert not any("roadmap marker" in issue for issue in issues)
    assert AUDIT._scan(
        "src/hlsgraph/render/vendor/elk.bundled.js",
        b"HLS" + b"Pilot",
    )


def test_current_public_source_tree_passes_boundary_audit() -> None:
    assert AUDIT._audit_source_tree(ROOT) == []
