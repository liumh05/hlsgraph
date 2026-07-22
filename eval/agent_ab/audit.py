"""Fail-closed audit for frozen inputs and sanitized public A/B artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

try:  # pragma: no cover - Python 3.10 compatibility
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

from .common import (
    asset_digest, load_corpus_lock, load_manifest, load_questions, load_static_cases,
)
from .sanitize import TOOL_ITEM_TYPES, TOOL_PAYLOAD_FIELDS, TOOL_PAYLOAD_REDACTION


SENSITIVE_PATTERNS = {
    "credential": re.compile(r"(?i)\b(?:sk|ghp|github_pat)-?[A-Za-z0-9_-]{12,}\b"),
    "bearer": re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{12,}"),
    "windows-absolute-path": re.compile(r"(?i)\b[A-Z]:[\\/]"),
    "unc-absolute-path": re.compile(r"\\\\[^\\\s\"']+\\[^\s\"']+"),
    "user-home-path": re.compile(r"/(?:home|Users)/[^/\s\"']+"),
    "posix-absolute-path": re.compile(r"(?<![A-Za-z0-9:/])/(?!/)[^\s\"']+"),
    "ipv4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
}

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_MANIFEST_PATH = ROOT / "examples" / "dataflow_gemm" / "hlsgraph.toml"
EXPECTED_LICENSE_SHA256 = {
    "dataflow_gemm": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
    "stream_blocks": "ee37cb403c7b162482ef62ca0a7f087bfdaab4cf219bfff5dd2a347678bade36",
    "bitonic": "5d3016bfaa895975fa5e815e4a18911c4a3f73e2773cedb4e4382cfe9e2613fd",
    "cordic": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
}


def _raw_tool_payloads(value: Any, location: str = "$") -> list[str]:
    findings: list[str] = []
    if isinstance(value, list):
        for index, item in enumerate(value):
            findings.extend(_raw_tool_payloads(item, f"{location}[{index}]"))
        return findings
    if not isinstance(value, dict):
        return findings
    item = value.get("item")
    event_type = str(value.get("type", "")).casefold()
    if isinstance(item, dict) and str(item.get("type", "")).casefold() in TOOL_ITEM_TYPES:
        for key in TOOL_PAYLOAD_FIELDS & set(item):
            if item[key] is not None and item[key] != TOOL_PAYLOAD_REDACTION:
                findings.append(f"{location}.item.{key}")
    elif (event_type in TOOL_ITEM_TYPES or "tool" in event_type or "command" in event_type) \
            and not event_type.endswith(".started"):
        for key in TOOL_PAYLOAD_FIELDS & set(value):
            if value[key] is not None and value[key] != TOOL_PAYLOAD_REDACTION:
                findings.append(f"{location}.{key}")
    for key, item_value in value.items():
        findings.extend(_raw_tool_payloads(item_value, f"{location}.{key}"))
    return findings


def _structured_values(path: Path, text: str) -> list[Any]:
    if path.suffix.casefold() == ".json":
        return [json.loads(text)]
    if path.suffix.casefold() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return []


def _text_views(data: bytes) -> tuple[list[str], bool]:
    """Normalize BOM-declared wide text while retaining a raw scan view."""
    raw = data.decode("utf-8", errors="replace")
    views = [raw]
    malformed = False
    bom_encodings = (
        (b"\xff\xfe\x00\x00", "utf-32-le"),
        (b"\x00\x00\xfe\xff", "utf-32-be"),
        (b"\xff\xfe", "utf-16-le"),
        (b"\xfe\xff", "utf-16-be"),
        (b"\xef\xbb\xbf", "utf-8-sig"),
    )
    declared = next(
        (encoding for marker, encoding in bom_encodings if data.startswith(marker)),
        None,
    )
    if declared is not None:
        try:
            normalized = data.decode(declared).lstrip("\ufeff")
        except UnicodeError:
            malformed = True
        else:
            views.insert(0, normalized)
    if b"\x00" in data:
        nul_free = data.replace(b"\x00", b"").decode("utf-8", errors="replace")
        if nul_free not in views:
            views.append(nul_free)
    return list(dict.fromkeys(views)), malformed


def _license_errors(lock: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for corpus in lock["corpora"]:
        corpus_id = corpus["id"]
        expected = EXPECTED_LICENSE_SHA256.get(corpus_id)
        if expected is None:
            errors.append(f"{corpus_id}: license bytes are not frozen by the audit")
            continue
        if corpus_id == "dataflow_gemm":
            try:
                license_bytes = (ROOT / "LICENSE").read_bytes()
            except OSError as exc:
                errors.append(f"dataflow_gemm: cannot read repository license: {exc}")
                continue
            actual = hashlib.sha256(license_bytes).hexdigest()
            if actual != expected:
                errors.append("dataflow_gemm: repository license byte hash drift")
            continue
        license_entries = [
            entry for entry in corpus["files"]
            if entry.get("destination") == "UPSTREAM_LICENSE.txt"
        ]
        if len(license_entries) != 1:
            errors.append(f"{corpus_id}: expected one pinned UPSTREAM_LICENSE.txt")
            continue
        if license_entries[0].get("sha256") != expected:
            errors.append(f"{corpus_id}: pinned license byte hash drift")
    return errors


def _fixture_report_errors(
    lock: dict[str, Any], fixture_manifest: dict[str, Any],
) -> list[str]:
    """Require every report-like public fixture mapping to be synthetic."""
    errors: list[str] = []
    artifacts = {
        str(item.get("path", "")).replace("\\", "/"): item
        for item in fixture_manifest.get("artifact_paths", [])
        if isinstance(item, dict)
    }
    for path, artifact in artifacts.items():
        kind = str(artifact.get("kind", "")).casefold()
        role = str(artifact.get("role", "")).casefold()
        report_like = (
            path.startswith("reports/")
            or kind.startswith(("amd.vitis.", "amd.vivado."))
            or any(
                token in role
                for token in (
                    "report", "result", "profile", "schedule", "timing",
                    "utilization", "physical_summary", "directive_status",
                )
            )
        )
        if not report_like:
            continue
        metadata = artifact.get("metadata")
        if not isinstance(metadata, dict) or metadata.get("fixture_authority") != "synthetic":
            errors.append(f"dataflow_gemm: report {path} is not explicitly synthetic")

    corpus = next(
        (item for item in lock["corpora"] if item["id"] == "dataflow_gemm"),
        None,
    )
    if corpus is None:
        return [*errors, "dataflow_gemm: corpus is missing"]
    prefix = "examples/dataflow_gemm/"
    for entry in corpus["files"]:
        source_path = str(entry.get("source_path", "")).replace("\\", "/")
        if not source_path.startswith(prefix + "reports/"):
            continue
        fixture_path = source_path.removeprefix(prefix)
        artifact = artifacts.get(fixture_path)
        metadata = artifact.get("metadata") if isinstance(artifact, dict) else None
        if not isinstance(metadata, dict) or metadata.get("fixture_authority") != "synthetic":
            errors.append(
                f"dataflow_gemm: locked report {fixture_path} lacks a synthetic manifest mapping"
            )
    return errors


def audit_frozen_assets() -> dict[str, Any]:
    manifest = load_manifest()
    lock = load_corpus_lock()
    questions = load_questions()
    static_cases = load_static_cases()
    errors: list[str] = []
    errors.extend(_license_errors(lock))
    try:
        fixture_manifest = tomllib.loads(FIXTURE_MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        errors.append(f"dataflow_gemm: cannot parse fixture manifest: {exc}")
    else:
        errors.extend(_fixture_report_errors(lock, fixture_manifest))
    corpus_ids = {item["id"] for item in lock["corpora"]}
    for question in questions:
        if question["corpus_id"] not in corpus_ids:
            errors.append(f"{question['id']}: unknown corpus")
        serialized = json.dumps(question, ensure_ascii=False)
        if re.search(r"(?i)[A-Z]:\\|/(?:home|Users)/", serialized):
            errors.append(f"{question['id']}: absolute path in ground truth")
    for case in static_cases:
        if case["corpus_id"] not in corpus_ids:
            errors.append(f"{case['id']}: unknown corpus")
        serialized = json.dumps(case, ensure_ascii=False)
        if re.search(r"(?i)[A-Z]:\\|/(?:home|Users)/", serialized):
            errors.append(f"{case['id']}: absolute path in static gold")
    for corpus in lock["corpora"]:
        revision = corpus.get("revision")
        if revision:
            for entry in corpus["files"]:
                url = entry.get("url")
                if url and revision not in url:
                    errors.append(f"{corpus['id']}: URL is not pinned to corpus revision")
        if corpus["id"] != "dataflow_gemm" and corpus["tool_evidence"] != "none":
            errors.append(f"{corpus['id']}: external corpus must remain source-only")
    if manifest["arms"][1]["revision"] != "286e9ccc2dad45336d4fd67052930322054d64b5":
        errors.append("CodeGraph revision drift")
    codegraph = manifest["arms"][1]
    build = codegraph.get("build_identity", {})
    if build.get("runtime_tree_algorithm") != "hlsgraph.runtime_tree.v1":
        errors.append("CodeGraph runtime-tree algorithm drift")
    return {
        "schema_version": "hlsgraph.agent_eval.asset_audit.v1",
        "asset_sha256": asset_digest(),
        "questions": len(questions),
        "static_cases": len(static_cases),
        "corpora": len(lock["corpora"]),
        "arms": len(manifest["arms"]),
        "codegraph_entrypoint_sha256": codegraph["entrypoint_sha256"],
        "codegraph_dist_tree_sha256": build.get("dist_tree_sha256"),
        "codegraph_dependency_tree_sha256": build.get("dependency_tree_sha256"),
        "errors": errors,
        "passed": not errors,
    }


def _files(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        if path.is_dir():
            yield from (item for item in sorted(path.rglob("*")) if item.is_file())
        elif path.is_file():
            yield path


def audit_public_artifacts(paths: Iterable[Path]) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    count = 0
    for path in _files(paths):
        count += 1
        try:
            data = path.read_bytes()
        except OSError as exc:
            findings.append({"file": str(path), "kind": "unreadable", "error": str(exc)})
            continue
        text_views, malformed = _text_views(data)
        text = text_views[0]
        if malformed:
            findings.append({"file": str(path), "kind": "malformed-unicode"})
        for name, pattern in SENSITIVE_PATTERNS.items():
            if any(pattern.search(view) for view in text_views):
                findings.append({"file": str(path), "kind": name})
        try:
            values = _structured_values(path, text)
        except json.JSONDecodeError:
            findings.append({"file": str(path), "kind": "invalid-json"})
            continue
        for index, value in enumerate(values):
            for location in _raw_tool_payloads(value):
                findings.append({
                    "file": str(path), "kind": "raw-tool-payload",
                    "record": index, "location": location,
                })
    return {
        "schema_version": "hlsgraph.agent_eval.public_artifact_audit.v1",
        "files": count,
        "findings": findings,
        "passed": not findings,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = {"frozen_assets": audit_frozen_assets()}
    if args.paths:
        report["public_artifacts"] = audit_public_artifacts(args.paths)
    passed = all(item["passed"] for item in report.values())
    report["passed"] = passed
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0 if passed else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
