#!/usr/bin/env python3
"""Hash the semantic surface of public knowledge packs for review evidence.

Review attestations are deliberately excluded so adding an attestation after a
successful review does not change the bytes that the review claims to cover.
All rules, citations, bindings, coverage classifications, and other metadata
remain in the hashed surface.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
PACK_ROOT = ROOT / "src" / "hlsgraph" / "knowledge" / "packs"
IMPLEMENTATION_ROOT = ROOT / "src" / "hlsgraph"
_REVIEW_KEYS = frozenset({
    "review_status", "reviewers", "source_hashes", "review_evidence",
})


def semantic_surface(value: dict[str, Any]) -> dict[str, Any]:
    surface = json.loads(json.dumps(value))
    metadata = surface.get("metadata")
    if isinstance(metadata, dict):
        for key in _REVIEW_KEYS:
            metadata.pop(key, None)
    coverage = surface.get("coverage")
    if isinstance(coverage, dict):
        for key in _REVIEW_KEYS:
            coverage.pop(key, None)
    return surface


def surface_sha256(path: Path) -> str:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"knowledge pack root must be an object: {path}")
    encoded = json.dumps(
        semantic_surface(value), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def implementation_surface_sha256() -> str:
    """Hash public Python implementation bytes inspected by the reviewers."""
    digest = hashlib.sha256()
    for path in sorted(IMPLEMENTATION_ROOT.rglob("*.py")):
        relative = path.relative_to(ROOT).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
        digest.update(b"\0")
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = args.paths or sorted(PACK_ROOT.glob("*.json"))
    rows = []
    for path in paths:
        value = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "pack_id": value.get("pack_id"),
            "path": path.resolve().relative_to(ROOT).as_posix(),
            "review_surface_sha256": surface_sha256(path),
        })
    print(json.dumps({
        "implementation_surface_sha256": implementation_surface_sha256(),
        "packs": rows,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
