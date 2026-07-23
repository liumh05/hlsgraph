#!/usr/bin/env python3
"""Fail-closed hygiene audit for the public tree, wheel, and sdist."""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import ipaddress
import io
import json
import os
import re
import stat
import sys
import tarfile
import zipfile
from collections import Counter
from email import policy
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from typing import Any, Iterable
from urllib.parse import parse_qsl, urlsplit

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if SOURCE_ROOT.is_dir() and str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from hlsgraph.knowledge import KnowledgeCatalog, load_pack

try:
    from tools import knowledge_review_surface as _knowledge_review_surface
except ModuleNotFoundError:  # ``python tools/audit_release.py``
    import knowledge_review_surface as _knowledge_review_surface  # type: ignore[no-redef]
try:
    from tools import audit_knowledge_citations as _knowledge_citation_audit
except ModuleNotFoundError:  # ``python tools/audit_release.py``
    import audit_knowledge_citations as _knowledge_citation_audit  # type: ignore[no-redef]
try:
    from tools import run_knowledge_review as _knowledge_review_runner
except ModuleNotFoundError:  # ``python tools/audit_release.py``
    import run_knowledge_review as _knowledge_review_runner  # type: ignore[no-redef]
try:
    from tools import knowledge_review_shards as _knowledge_review_shards
    from tools import knowledge_review_suite_cache as _knowledge_review_suite_cache
    from tools import knowledge_review_suite_replay as _knowledge_review_suite_replay
    from tools import run_knowledge_review_suite as _knowledge_review_suite
    from tools import seal_knowledge_review_suite as _knowledge_review_suite_seal
    from tools import execute_knowledge_review_suite as _knowledge_review_suite_executor
except ModuleNotFoundError:  # ``python tools/audit_release.py``
    import knowledge_review_shards as _knowledge_review_shards  # type: ignore[no-redef]
    import knowledge_review_suite_cache as _knowledge_review_suite_cache  # type: ignore[no-redef]
    import knowledge_review_suite_replay as _knowledge_review_suite_replay  # type: ignore[no-redef]
    import run_knowledge_review_suite as _knowledge_review_suite  # type: ignore[no-redef]
    import seal_knowledge_review_suite as _knowledge_review_suite_seal  # type: ignore[no-redef]
    import execute_knowledge_review_suite as _knowledge_review_suite_executor  # type: ignore[no-redef]


FORBIDDEN_NAMES = (
    "/.hlsgraph/", "__pycache__", ".pytest_cache", ".wheel-test",
    ".packaging-test", "/build/", ".egg-info/", ".db", ".sqlite",
    ".pyc", ".pyo", "/.env", ".pem", ".key",
)
FORBIDDEN_KNOWLEDGE_BODY_KEYS = frozenset({
    "body", "chunk", "chunks", "chunk_text", "content", "document_body",
    "document_text", "embedding", "embeddings", "extracted_text",
    "full_text", "ocr", "ocr_text", "page_text", "pages", "pdf_bytes",
    "raw_text", "screenshot", "screenshots", "text",
})
FORBIDDEN_KNOWLEDGE_BODY_PARTS = frozenset({
    "chunks", "extracted", "knowledge_chunks", "ocr", "pages",
    "references_md",
})
MAX_KNOWLEDGE_PACK_STRING_CHARS = 512
RFC1918_NETWORKS = (
    ipaddress.IPv4Network((0x0A000000, 8)),
    ipaddress.IPv4Network((0xAC100000, 12)),
    ipaddress.IPv4Network((0xC0A80000, 16)),
)
ALLOWED_SDIST_EGG_INFO = frozenset({
    "PKG-INFO",
    "SOURCES.txt",
    "dependency_links.txt",
    "entry_points.txt",
    "requires.txt",
    "top_level.txt",
})
SECRET_PATTERNS = (
    ("private key", re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("assigned credential", re.compile(
        rb"(?i)(?:api[_-]?key|(?:access|auth|refresh)[_-]?token|"
        rb"client[_-]?secret|password|passwd|credential)\s*[:=]\s*"
        rb"['\"]?[A-Za-z0-9_./+=:@-]{8,}"
    )),
    ("license server", re.compile(
        rb"(?i)license[_-]?server\s*[:=]\s*[^\s,;]+"
    )),
    ("bearer credential", re.compile(
        rb"(?i)authorization\s*:\s*bearer\s+[A-Za-z0-9._~+/=-]{12,}"
    )),
    ("credential in URL", re.compile(
        rb"(?i)https?://[^/@\s:]+:[^/@\s]+@"
    )),
    ("GitHub token", re.compile(
        rb"(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{40,})"
    )),
    ("cloud/API token", re.compile(
        rb"(?:AKIA[0-9A-Z]{16}|sk-[A-Za-z0-9_-]{20,})"
    )),
    ("Windows absolute path", re.compile(
        rb"(?i)(?<![A-Za-z0-9+.-])[A-Za-z]:[\\/](?![\\/])"
    )),
    ("POSIX user-home path", re.compile(rb"/(?:home|Users)/[^/\s]+/")),
)


def _public_boundary_pattern(*parts: bytes, word: bool = False) -> re.Pattern[bytes]:
    escaped = re.escape(b"".join(parts))
    if word:
        escaped = rb"(?<![A-Za-z0-9_])" + escaped + rb"(?![A-Za-z0-9_])"
    return re.compile(escaped, re.IGNORECASE | re.ASCII)


# The final boolean is true only for short identifiers known to occur as
# unrelated symbols in audited third-party minified files.
PUBLIC_BOUNDARY_PATTERNS = (
    ("non-public repository identifier", _public_boundary_pattern(
        b"hlsgraph", b"-", b"research",
    ), False),
    ("non-public roadmap document", _public_boundary_pattern(
        b"research", b"-", b"integration",
    ), False),
    ("non-public roadmap marker 1", _public_boundary_pattern(
        b"HLS", b"Pilot", word=True,
    ), False),
    ("non-public roadmap marker 2", _public_boundary_pattern(
        b"Timely", b"HLS", word=True,
    ), False),
    ("non-public roadmap marker 3", _public_boundary_pattern(
        b"G", b"NN", word=True,
    ), True),
    ("non-public roadmap marker 4", _public_boundary_pattern(
        b"R", b"CD", word=True,
    ), True),
    ("non-public roadmap marker 5", _public_boundary_pattern(
        b"control", b"ler", word=True,
    ), False),
    ("non-public roadmap marker 6", _public_boundary_pattern(
        b"agent", b"ic", word=True,
    ), False),
    ("historical personal address", _public_boundary_pattern(
        b"1964722203", b"@", b"qq", b".", b"com",
    ), False),
    ("non-public laboratory host", _public_boundary_pattern(
        b"fpga", b"5090", word=True,
    ), False),
    ("non-public laboratory user", _public_boundary_pattern(
        b"srtp", b"-", b"agent", word=True,
    ), False),
    ("non-public laboratory SSH alias", re.compile(
        rb"(?i)(?<![A-Za-z0-9_])s" + rb"sh(?:\.exe)?\s+"
        rb"(?:-[^\s]+\s+)*h" + rb"ls(?![A-Za-z0-9_])"
    ), False),
)
PUBLIC_BOUNDARY_SHORT_EXCLUSIONS = frozenset({
    "src/hlsgraph/render/vendor/elk.bundled.js",
    "src/hlsgraph/render/vendor/cytoscape.min.js",
    "hlsgraph/render/vendor/elk.bundled.js",
    "hlsgraph/render/vendor/cytoscape.min.js",
})
REQUIRED_SDIST = {
    "LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md", "SECURITY.md",
    "CONTRIBUTING.md", "CODE_OF_CONDUCT.md", "DCO", "CITATION.cff",
    "sbom.spdx.json", "docs/references.md",
    "docs/privacy-and-security.md",
    "docs/knowledge-review-runbook.md",
    "tests/attested_run_support.py",
    "tests/typed_report_support.py",
    "tests/fixtures/v02_minimal_bundle.json",
    "tools/knowledge_review.schema.json",
    "tools/knowledge_review_receipt.schema.json",
    "tools/knowledge_review_evidence.schema.json",
    "tools/knowledge_review_shard.schema.json",
    "tools/knowledge_review_suite_evidence.schema.json",
    "tools/knowledge_review_suite_receipt.schema.json",
    "tools/knowledge_review_suite_trace.schema.json",
    "tools/knowledge_review_prompts/adversarial.md",
    "tools/knowledge_review_prompts/adversarial_shard.md",
    "tools/knowledge_review_prompts/semantic.md",
    "tools/knowledge_review_prompts/semantic_shard.md",
    "tools/knowledge_review_shards.py",
    "tools/knowledge_review_suite_cache.py",
    "tools/knowledge_review_suite_replay.py",
    "tools/apply_knowledge_review_suite_attestation.py",
    "tools/execute_knowledge_review_suite.py",
    "tools/run_knowledge_review.py",
    "tools/run_knowledge_review_suite.py",
    "tools/seal_knowledge_review_suite.py",
    "tools/audit_release.py",
    "docs/knowledge-citation-audit-v0.3.json",
    "docs/knowledge-review-evidence-v0.3.json",
    "docs/knowledge-review-v0.3.adversarial.json",
    "docs/knowledge-review-v0.3.adversarial.receipt.json",
    "docs/knowledge-review-v0.3.adversarial.trace.jsonl",
    "docs/knowledge-review-v0.3.semantic.json",
    "docs/knowledge-review-v0.3.semantic.receipt.json",
    "docs/knowledge-review-v0.3.semantic.trace.jsonl",
}
RELEASE_VERSION = "0.3.0"
SEMANTIC_REVIEW_PROTOCOL = "hlsgraph.knowledge-review.semantic.v1"
ADVERSARIAL_REVIEW_PROTOCOL = "hlsgraph.knowledge-review.adversarial.v1"
SEMANTIC_REVIEW_PATH = "docs/knowledge-review-v0.3.semantic.json"
ADVERSARIAL_REVIEW_PATH = "docs/knowledge-review-v0.3.adversarial.json"
REVIEW_SCHEMA_PATH = "tools/knowledge_review.schema.json"
REVIEW_RECEIPT_SCHEMA_PATH = "tools/knowledge_review_receipt.schema.json"
SEMANTIC_REVIEW_PROMPT_PATH = "tools/knowledge_review_prompts/semantic.md"
ADVERSARIAL_REVIEW_PROMPT_PATH = "tools/knowledge_review_prompts/adversarial.md"
SEMANTIC_REVIEW_RECEIPT_PATH = (
    "docs/knowledge-review-v0.3.semantic.receipt.json"
)
ADVERSARIAL_REVIEW_RECEIPT_PATH = (
    "docs/knowledge-review-v0.3.adversarial.receipt.json"
)
SEMANTIC_REVIEW_TRACE_PATH = "docs/knowledge-review-v0.3.semantic.trace.jsonl"
ADVERSARIAL_REVIEW_TRACE_PATH = "docs/knowledge-review-v0.3.adversarial.trace.jsonl"
CITATION_AUDIT_PATH = "docs/knowledge-citation-audit-v0.3.json"
CITATION_EVIDENCE_PATH = "docs/knowledge-review-evidence-v0.3.json"
CITATION_EVIDENCE_SCHEMA_PATH = "tools/knowledge_review_evidence.schema.json"
CITATION_EVIDENCE_SCHEMA_VERSION = (
    "hlsgraph.knowledge-review.evidence-map.v2"
)
CITATION_EVIDENCE_SCHEMA_SHA256 = (
    "0caa895fbf0e2622a85e61df1c4eab38e9b4e0727f19facbe688bb8bf9d7bcef"
)
REVIEW_RECEIPT_SCHEMA_VERSION = "hlsgraph.knowledge-review.cli-receipt.v4"
REVIEW_SUITE_RECEIPT_SCHEMA_VERSION = (
    "hlsgraph.knowledge-review.cli-receipt.v6"
)
REVIEW_SUITE_EVIDENCE_SCHEMA_VERSION = (
    "hlsgraph.knowledge-review.suite-evidence.v1"
)
REVIEW_SUITE_EVIDENCE_MANIFEST = "suite-evidence.json"
REVIEW_SHARD_SCHEMA_PATH = "tools/knowledge_review_shard.schema.json"
REVIEW_SUITE_EVIDENCE_SCHEMA_PATH = (
    "tools/knowledge_review_suite_evidence.schema.json"
)
REVIEW_SUITE_RECEIPT_SCHEMA_PATH = (
    "tools/knowledge_review_suite_receipt.schema.json"
)
REVIEW_SUITE_TRACE_SCHEMA_PATH = (
    "tools/knowledge_review_suite_trace.schema.json"
)
SEMANTIC_SHARD_PROMPT_PATH = (
    "tools/knowledge_review_prompts/semantic_shard.md"
)
ADVERSARIAL_SHARD_PROMPT_PATH = (
    "tools/knowledge_review_prompts/adversarial_shard.md"
)
REVIEW_BOUNDARY_CONTRACT_SCHEMA_VERSION = (
    "hlsgraph.knowledge-review.boundary-contract.v3"
)
REVIEW_RUNTIME_MANIFEST_SCHEMA_VERSION = (
    "hlsgraph.knowledge-review.runtime-manifest.v3"
)
REVIEW_BOUNDARY_POLICY = "default_deny_minimal_allowlist_v1"
REVIEW_CACHE_PARENT_POLICY = "caller_owned_0700_single_cache_v1"
REVIEW_EVIDENCE_PARENT_POLICY = "caller_owned_0700_dedicated_evidence_v1"
REVIEW_RUNTIME_OWNERSHIP_POLICY = (
    "caller_owned_frozen_0500_no_links_exact_codex_bwrap_v2"
)
REVIEW_TRACE_SCHEMA_VERSION = "hlsgraph.knowledge-review.tool-trace.v3"
REVIEW_MODEL = "gpt-5.6-sol"
REVIEW_REASONING_EFFORT = "medium"
REVIEW_CODEX_CLI_VERSION = "codex-cli 0.144.0"
REVIEW_TOOL_OUTPUT_TOKEN_LIMIT = 50_000
REVIEW_INITIAL_PROMPT_MAX_BYTES = 512 * 1024
REVIEW_MAX_RAW_BYTES = 64 * 1024 * 1024
REVIEW_OFFICIAL_CODEX_ELF_SHA256 = (
    "901923c1808a151f6926d41d703c17ad48815662cefb1c8d832a052c44271429"
)
REVIEW_OFFICIAL_CODEX_BWRAP_SHA256 = (
    "77360cb751ccedc5971391444ac86a8a33c15b04d6b4a6fe45f5d25496e62c4c"
)
REVIEW_CODEX_EXECUTABLE_RELATIVE_PATH = "codex"
REVIEW_CODEX_BWRAP_RELATIVE_PATH = "codex-resources/bwrap"
REVIEW_INITIAL_PROCESS_PATH = [
    "$CODEX_RUNTIME/codex-resources", "/usr/bin", "/bin",
]
REVIEW_INVOCATIONS_KEY = "review_invocations"
IMPLEMENTATION_SURFACE_HASH_KEY = (
    "src/hlsgraph/**/*.py#implementation-surface"
)
PACK_SURFACE_HASH_PREFIX = "src/hlsgraph/knowledge/packs/"
PACK_SURFACE_HASH_SUFFIX = "#semantic-surface"
SURFACE_HELPER_HASH_KEY = "tools/knowledge_review_surface.py#sha256"
SDIST_BUILD_BOUND_PATHS = frozenset({
    "MANIFEST.in", "build_backend.py", "pyproject.toml",
})
ELK_SOURCE_REVISIONS = frozenset({
    "a8304cf79fde75bc2ab1a89d28320f53f8637436",
    "62d5909f96fad541bc101ad52dabaece6b7eab7e",
    "7ca51784e42a24201f29bc13e458728b6fc61cdc",
})


def _formal_host_is_windows() -> bool:
    """Test seam; formal production audits use the interpreter host."""

    return os.name == "nt"
SOURCE_SKIP_DIRS = frozenset({
    ".git", ".hlsgraph", ".mypy_cache", ".nox", ".packaging-test",
    ".pytest_cache", ".ruff_cache", ".tox", ".venv", ".wheel-test",
    "__pycache__", "build", "dist", "htmlcov",
})
SOURCE_SCAN_EXCLUSIONS = frozenset({
    # This file necessarily contains the credential-detection expressions.
    "tools/audit_release.py",
})
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_ADVANTAGE_CLAIM_PATTERNS = (
    re.compile(
        r"(?i)\b(?:outperform(?:s|ed|ing)?|beats?|superior\s+to|faster\s+than|"
        r"more\s+accurate\s+than)\b"
    ),
    re.compile(
        r"(?i)\bperformance\s+advantage\s+(?:over|against|versus|is\s+supported|"
        r"is\s+established|was\s+demonstrated)\b"
    ),
    re.compile(r"(?:优于|胜过|领先于|显著提升|性能优势(?:已|得到|获得|相对))"),
)


def _allowed_sdist_egg_info(name: str) -> bool:
    parts = PurePosixPath(name.replace("\\", "/")).parts
    return (
        len(parts) == 3
        and parts[0] == "src"
        and parts[1] == "hlsgraph.egg-info"
        and parts[2] in ALLOWED_SDIST_EGG_INFO
    )


def _forbidden(name: str, *, sdist: bool = False) -> str | None:
    relative = name.replace("\\", "/").lstrip("/")
    if PurePosixPath(relative).suffix.casefold() == ".pdf":
        return ".pdf"
    lowered_parts = tuple(part.casefold() for part in PurePosixPath(relative).parts)
    for index in range(len(lowered_parts) - 2):
        if (
            lowered_parts[index:index + 3] == ("hlsgraph", "knowledge", "packs")
            and PurePosixPath(relative).suffix.casefold() != ".json"
        ):
            return "non-JSON knowledge-pack payload"
    if any(part in FORBIDDEN_KNOWLEDGE_BODY_PARTS for part in lowered_parts):
        return "knowledge document body"
    if (
        PurePosixPath(relative).name.casefold()
        in {"full.md", "full.txt", "full.json"}
    ):
        return "knowledge document body"
    normalized = "/" + relative
    lowered = normalized.casefold()
    for item in FORBIDDEN_NAMES:
        if item.casefold() not in lowered:
            continue
        if item == ".egg-info/" and sdist and _allowed_sdist_egg_info(relative):
            continue
        return item
    return None


def _scan_views(data: bytes) -> tuple[list[bytes], bool]:
    """Return raw and normalized views plus malformed-BOM status."""
    views = [data]
    malformed = False
    bom_encodings = (
        (b"\xff\xfe\x00\x00", "utf-32-le"),
        (b"\x00\x00\xfe\xff", "utf-32-be"),
        (b"\xff\xfe", "utf-16-le"),
        (b"\xfe\xff", "utf-16-be"),
        (b"\xef\xbb\xbf", "utf-8-sig"),
    )
    declared_encoding = next(
        (encoding for marker, encoding in bom_encodings if data.startswith(marker)),
        None,
    )
    if declared_encoding is not None:
        try:
            normalized = data.decode(declared_encoding).lstrip("\ufeff").encode("utf-8")
        except UnicodeError:
            malformed = True
        else:
            if normalized not in views:
                views.append(normalized)
    if b"\x00" in data:
        nul_free = data.replace(b"\x00", b"")
        if nul_free not in views:
            views.append(nul_free)
    return views, malformed


def _private_ipv4(view: bytes) -> str | None:
    for match in re.finditer(rb"(?<![0-9])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![0-9])", view):
        try:
            address = ipaddress.IPv4Address(match.group().decode("ascii"))
        except ipaddress.AddressValueError:
            continue
        if any(address in network for network in RFC1918_NETWORKS):
            return str(address)
    return None


def _knowledge_payload_issues(name: str, data: bytes) -> list[str]:
    """Reject document bodies while allowing short authored pack paraphrases."""
    normalized = name.replace("\\", "/").lstrip("/").casefold()
    if not (
        normalized.startswith("src/hlsgraph/knowledge/packs/")
        or normalized.startswith("hlsgraph/knowledge/packs/")
    ) or not normalized.endswith(".json"):
        return []
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return [f"invalid knowledge-pack JSON in {name}"]
    issues: list[str] = []

    def visit(item: object, location: str) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                normalized_key = str(key).casefold().replace("-", "_")
                child_location = f"{location}.{key}"
                if normalized_key in FORBIDDEN_KNOWLEDGE_BODY_KEYS:
                    issues.append(
                        f"knowledge document body field {child_location} in {name}"
                    )
                visit(child, child_location)
        elif isinstance(item, list):
            for index, child in enumerate(item):
                visit(child, f"{location}[{index}]")
        elif isinstance(item, str) and len(item) > MAX_KNOWLEDGE_PACK_STRING_CHARS:
            issues.append(f"oversized knowledge-pack text at {location} in {name}")

    visit(value, "$")
    return issues


def _unsafe_archive_name(name: str) -> str | None:
    """Return why an archive member name is unsafe, without extracting it."""
    if not name or "\x00" in name:
        return "empty or NUL-containing path"
    if "\\" in name:
        return "backslash path"
    if name.startswith("/") or re.match(r"^[A-Za-z]:", name):
        return "absolute path"
    trimmed = name[:-1] if name.endswith("/") else name
    if not trimmed:
        return "empty path"
    parts = trimmed.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        return "non-canonical or traversing path"
    return None


def _duplicate_archive_names(names: Iterable[str]) -> list[str]:
    counts = Counter(names)
    duplicates = {name for name, count in counts.items() if count > 1}
    folded: dict[str, set[str]] = {}
    for name in names:
        folded.setdefault(name.casefold(), set()).add(name)
    duplicates.update(
        " / ".join(sorted(group)) for group in folded.values() if len(group) > 1
    )
    return sorted(duplicates)


def _scan(name: str, data: bytes) -> list[str]:
    issues: list[str] = []
    # Run ASCII-oriented boundary patterns over both the original bytes and a
    # canonical UTF-8 view.  Scanning raw bytes alone misses every ASCII token
    # in UTF-16/UTF-32 text (including Windows paths and credentials).  BOM
    # decoding is strict and malformed declared Unicode is itself a release
    # blocker.  The NUL-free view covers BOM-less wide-character text without
    # attempting to classify arbitrary binary payloads as prose.
    scan_views, malformed_unicode = _scan_views(data)
    if malformed_unicode:
        issues.append(f"malformed declared Unicode text in {name}")
    if b"%PDF-" in data[:1024]:
        issues.append(f"PDF document magic in {name}")
    if len(data) <= 8 * 1024 * 1024:
        for label, pattern in SECRET_PATTERNS:
            if any(pattern.search(view) for view in scan_views):
                issues.append(f"sensitive {label} pattern in {name}")
        private_endpoint = next(
            (address for view in scan_views if (address := _private_ipv4(view))),
            None,
        )
        if private_endpoint is not None:
            issues.append(f"non-public RFC1918 endpoint in {name}")
    normalized_name = name.replace("\\", "/").lstrip("/")
    encoded_name = normalized_name.encode("utf-8", errors="surrogateescape")
    for label, pattern, allow_short_vendor_symbol in PUBLIC_BOUNDARY_PATTERNS:
        if pattern.search(encoded_name):
            issues.append(f"{label} in member name {name}")
        if (
            allow_short_vendor_symbol
            and normalized_name in PUBLIC_BOUNDARY_SHORT_EXCLUSIONS
        ):
            continue
        if any(pattern.search(view) for view in scan_views):
            issues.append(f"{label} in {name}")
    issues.extend(_knowledge_payload_issues(name, data))
    return list(dict.fromkeys(issues))


def _audit_source_tree(root: Path) -> list[str]:
    """Scan files intended for the public repository, excluding build state."""
    issues: list[str] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        parts = PurePosixPath(relative).parts
        if any(part in SOURCE_SKIP_DIRS or part.endswith(".egg-info") for part in parts):
            continue
        if relative in SOURCE_SCAN_EXCLUSIONS or not path.is_file():
            continue
        if marker := _forbidden(relative):
            issues.append(f"forbidden public-tree member {relative} ({marker})")
            continue
        try:
            issues.extend(_scan(relative, path.read_bytes()))
        except OSError as exc:
            issues.append(f"cannot read public-tree member {relative}: {exc}")
    return issues


def _package_verification_code(files: Iterable[tuple[str, bytes]]) -> str:
    """Compute the SPDX package verification code from analyzed files."""
    hashes = sorted(
        hashlib.sha1(data).hexdigest()  # noqa: S324 - SPDX 2.3 mandates SHA-1
        for _name, data in files
    )
    concatenated = "".join(hashes)
    # SHA-1 is required by SPDX 2.3 packageVerificationCode.
    return hashlib.sha1(concatenated.encode("ascii")).hexdigest()  # noqa: S324


def _audit_sbom(sbom_data: bytes, root: Path) -> list[str]:
    issues: list[str] = []
    try:
        sbom = json.loads(sbom_data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return [f"invalid root SBOM: {exc}"]
    if sbom.get("spdxVersion") != "SPDX-2.3":
        issues.append("root SBOM is not SPDX-2.3")
    if sbom.get("name") != f"hlsgraph-{RELEASE_VERSION}":
        issues.append("root SBOM release name does not match the package version")

    packages = {
        item.get("SPDXID"): item for item in sbom.get("packages", [])
        if isinstance(item, dict) and isinstance(item.get("SPDXID"), str)
    }
    hlsgraph = packages.get("SPDXRef-Package-HLSGraph", {})
    if hlsgraph.get("versionInfo") != RELEASE_VERSION:
        issues.append("root SBOM HLSGraph package version is stale")
    elkjs = packages.get("SPDXRef-Package-ELKJS", {})
    source_info = elkjs.get("sourceInfo")
    external_refs = elkjs.get("externalRefs")
    source_text = source_info if isinstance(source_info, str) else ""
    ref_text = " ".join(
        item.get("referenceLocator", "") for item in external_refs or []
        if isinstance(item, dict) and isinstance(item.get("referenceLocator"), str)
    )
    missing_revisions = sorted(
        revision for revision in ELK_SOURCE_REVISIONS
        if revision not in source_text or revision not in ref_text
    )
    if missing_revisions:
        issues.append(
            "SBOM elkjs corresponding-source lineage is incomplete: "
            + ", ".join(missing_revisions)
        )
    try:
        notice_text = (root / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    except OSError as exc:
        issues.append(f"cannot read THIRD_PARTY_NOTICES.md: {exc}")
    else:
        if any(revision not in notice_text for revision in ELK_SOURCE_REVISIONS):
            issues.append("THIRD_PARTY_NOTICES.md lacks exact elkjs/ELK source revisions")
        if "corresponding source availability" not in notice_text.casefold():
            issues.append("THIRD_PARTY_NOTICES.md lacks an EPL source-availability section")

    file_data: dict[str, tuple[str, bytes]] = {}
    for item in sbom.get("files", []):
        spdx_id = item.get("SPDXID")
        file_name = item.get("fileName")
        if not isinstance(spdx_id, str) or not isinstance(file_name, str):
            issues.append("SBOM file entry lacks SPDXID or fileName")
            continue
        if spdx_id in file_data:
            issues.append(f"duplicate SBOM file SPDXID: {spdx_id}")
            continue
        pure_name = PurePosixPath(file_name)
        if not file_name.startswith("./") or ".." in pure_name.parts:
            issues.append(f"unsafe SBOM fileName: {file_name}")
            continue
        candidate = root.joinpath(*pure_name.parts)
        if not candidate.is_file():
            issues.append(f"SBOM file is missing: {file_name}")
            continue
        data = candidate.read_bytes()
        checksums = {
            value.get("algorithm", "").upper(): value.get("checksumValue", "").lower()
            for value in item.get("checksums", []) if isinstance(value, dict)
        }
        actual_sha256 = hashlib.sha256(data).hexdigest()
        if checksums.get("SHA256") != actual_sha256:
            issues.append(f"invalid SBOM SHA256 for {file_name}")
        file_data[spdx_id] = (file_name, data)

    for package in sbom.get("packages", []):
        if package.get("filesAnalyzed") is not True:
            continue
        name = package.get("name", "<unnamed>")
        file_ids = package.get("hasFiles")
        if not isinstance(file_ids, list) or not file_ids:
            issues.append(f"analyzed SBOM package has no files: {name}")
            continue
        missing = [spdx_id for spdx_id in file_ids if spdx_id not in file_data]
        if missing:
            issues.append(f"SBOM package {name} has unknown files: {missing}")
            continue
        expected = package.get("packageVerificationCode", {}).get(
            "packageVerificationCodeValue"
        )
        actual = _package_verification_code(file_data[spdx_id] for spdx_id in file_ids)
        if expected != actual:
            issues.append(f"invalid SPDX packageVerificationCode for {name}")
    return issues


def _audit_wheel_metadata(data: bytes) -> list[str]:
    """Validate core metadata using RFC-aware parsing (LF and CRLF safe)."""
    issues: list[str] = []
    metadata = BytesParser(policy=policy.compat32).parsebytes(data)
    if metadata.get("Version") != RELEASE_VERSION:
        issues.append(f"wheel metadata is not final v{RELEASE_VERSION}")
    urls = metadata.get_all("Project-URL", [])
    if not any("https://github.com/liumh05/hlsgraph" in url for url in urls):
        issues.append("wheel metadata has stale repository URLs")
    return issues


def _expected_wheel_package(root: Path) -> dict[str, bytes]:
    """Return every release-intended byte below ``src/hlsgraph``."""
    source_root = root / "src" / "hlsgraph"
    expected: dict[str, bytes] = {}
    for path in sorted(source_root.rglob("*")):
        relative = path.relative_to(source_root)
        if (
            not path.is_file()
            or any(part in SOURCE_SKIP_DIRS for part in relative.parts)
        ):
            continue
        expected[(PurePosixPath("hlsgraph") / PurePosixPath(*relative.parts)).as_posix()] = (
            _strict_file_bytes(
                path, f"installable source {relative.as_posix()}", root=root,
            )
        )
    return expected


def _expected_sdist_installable(root: Path) -> dict[str, bytes]:
    """Return every installable package byte and build-control byte in source."""

    result = {
        "src/" + name: data for name, data in _expected_wheel_package(root).items()
    }
    for relative in sorted(SDIST_BUILD_BOUND_PATHS):
        result[relative] = _strict_file_bytes(
            root / relative, f"sdist build input {relative}", root=root,
        )
    return result


def _payload_digest(payload: dict[str, bytes]) -> str:
    """Use the evaluation wheel-identity digest over package paths and bytes."""
    digest = hashlib.sha256()
    for name, data in sorted(payload.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(data).digest())
        digest.update(b"\0")
    return digest.hexdigest()


def _release_wheel_package_digest(data: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        infos = archive.infolist()
        names = [item.filename.replace("\\", "/") for item in infos]
        if len(names) != len(set(names)):
            raise ValueError("release wheel has duplicate members")
        payload: dict[str, bytes] = {}
        for info, name in zip(infos, names):
            member = PurePosixPath(name)
            mode = (info.external_attr >> 16) & 0xFFFF
            if (_unsafe_archive_name(name) is not None
                    or info.create_system == 3 and stat.S_ISLNK(mode)):
                raise ValueError(f"release wheel has unsafe package member: {name}")
            if (member.parts and member.parts[0].casefold() == "hlsgraph"
                    and not info.is_dir()):
                payload[name] = archive.read(info)
    if "hlsgraph/__init__.py" not in payload:
        raise ValueError("release wheel lacks the HLSGraph package")
    return _payload_digest(payload)


def _release_sdist_package_digest(data: bytes) -> str:
    """Hash the package payload a normal sdist build is expected to consume."""

    expected_prefix = f"hlsgraph-{RELEASE_VERSION}/src/"
    payload: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        members = archive.getmembers()
        names = [item.name for item in members]
        if _duplicate_archive_names(names):
            raise ValueError("release sdist has duplicate members")
        for member in members:
            if not member.isfile() or not member.name.startswith(
                expected_prefix + "hlsgraph/"
            ):
                continue
            relative = member.name.removeprefix(expected_prefix)
            if _unsafe_archive_name(relative) is not None:
                raise ValueError(f"release sdist has unsafe package member: {relative}")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"release sdist package member is unreadable: {relative}")
            payload[relative] = stream.read()
    if "hlsgraph/__init__.py" not in payload:
        raise ValueError("release sdist lacks the HLSGraph package")
    return _payload_digest(payload)


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns,
        value.st_ctime_ns, value.st_mode, value.st_nlink,
        int(getattr(value, "st_uid", -1)),
    )


def _is_reparse_or_link(path: Path) -> bool:
    """Return whether one lexical path component redirects resolution."""

    try:
        if path.is_symlink():
            return True
        is_junction = getattr(path, "is_junction", None)
        if bool(is_junction and is_junction()):
            return True
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
    except OSError:
        return False
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(reparse_flag and attributes & reparse_flag)


def _lexical_absolute(path: Path) -> Path:
    """Make a path absolute without hiding a symlink/junction component."""

    return path if path.is_absolute() else Path.cwd() / path


def _strict_path_components(
    path: Path, label: str, *, root: Path | None = None,
) -> tuple[Path, Path]:
    """Validate containment and every ancestor before opening a release input."""

    lexical = _lexical_absolute(path)
    boundary = _lexical_absolute(root) if root is not None else Path(lexical.anchor)
    try:
        relative = lexical.relative_to(boundary)
    except ValueError as exc:
        raise ValueError(f"{label} escapes its required root") from exc
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"{label} uses a non-canonical or traversing path")
    current = boundary
    if _is_reparse_or_link(current):
        raise ValueError(f"{label} has a linked or reparse root")
    for part in relative.parts:
        current /= part
        if _is_reparse_or_link(current):
            raise ValueError(
                f"{label} has a linked or reparse path component: {current}"
            )
    try:
        resolved_boundary = boundary.resolve(strict=True)
        resolved = lexical.resolve(strict=True)
        resolved.relative_to(resolved_boundary)
    except (OSError, ValueError) as exc:
        raise ValueError(f"{label} is missing or escapes its required root") from exc
    return lexical, boundary


def _strict_file_bytes(
    path: Path, label: str, *, root: Path | None = None,
    required_file_mode: int | None = None,
    required_parent_mode: int | None = None,
    require_current_owner: bool = False,
    require_single_link: bool = False,
    max_bytes: int | None = None,
) -> bytes:
    """Read one regular file through a stable handle without following links."""

    lexical, boundary = _strict_path_components(path, label, root=root)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    parent = lexical.parent
    try:
        parent_before = os.stat(parent, follow_symlinks=False)
        before_path = os.stat(lexical, follow_symlinks=False)
        descriptor = os.open(lexical, flags)
    except OSError:
        raise ValueError(f"{label} is missing, linked, or unreadable") from None
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} is not a regular file")
        if (before_path.st_dev, before_path.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError(f"{label} changed before it was opened")
        if require_single_link and (before_path.st_nlink != 1 or before.st_nlink != 1):
            raise ValueError(f"{label} has hard-link aliases")
        if max_bytes is not None and before.st_size > max_bytes:
            raise ValueError(f"{label} exceeds its fixed byte limit")
        if os.name != "nt" and require_current_owner:
            uid = os.geteuid()
            if before.st_uid != uid or parent_before.st_uid != uid:
                raise ValueError(f"{label} violates its owner contract")
        if (os.name != "nt" and required_file_mode is not None
                and stat.S_IMODE(before.st_mode) != required_file_mode):
            raise ValueError(f"{label} violates its file-mode contract")
        if (os.name != "nt" and required_parent_mode is not None
                and stat.S_IMODE(parent_before.st_mode) != required_parent_mode):
            raise ValueError(f"{label} violates its parent-mode contract")
        chunks: list[bytes] = []
        size = 0
        while True:
            count = 1024 * 1024
            if max_bytes is not None:
                count = min(count, max_bytes + 1 - size)
            chunk = os.read(descriptor, count)
            if not chunk:
                break
            size += len(chunk)
            if max_bytes is not None and size > max_bytes:
                raise ValueError(f"{label} exceeds its fixed byte limit")
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        after_path = os.stat(lexical, follow_symlinks=False)
        parent_after = os.stat(parent, follow_symlinks=False)
    except OSError:
        raise ValueError(f"{label} disappeared while it was read") from None
    if (
        _stat_identity(before) != _stat_identity(after)
        or _stat_identity(before_path) != _stat_identity(after_path)
        or _stat_identity(parent_before) != _stat_identity(parent_after)
        or (after_path.st_dev, after_path.st_ino) != (after.st_dev, after.st_ino)
    ):
        raise ValueError(f"{label} changed while it was read")
    # Recheck ancestors after the read so a directory swap cannot silently
    # turn a fixed public path into an external path during verification.
    _strict_path_components(lexical, label, root=boundary)
    data = b"".join(chunks)
    if len(data) != before.st_size:
        raise ValueError(f"{label} size changed while it was read")
    return data


def _strict_json_object(
    path: Path, label: str, *, root: Path | None = None,
) -> tuple[dict[str, Any], bytes]:
    data = _strict_file_bytes(path, label, root=root)

    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            data.decode("utf-8", errors="strict"),
            object_pairs_hook=no_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"{label} contains non-finite JSON number {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"cannot read strict {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value, data


def _review_result_contract_issues(
    value: dict[str, Any], *, label: str, expected_protocol: str,
    expected_surfaces: dict[str, str], expected_implementation: str,
    expected_citation_audit_sha256: str,
    expected_citations: dict[str, dict[str, Any]],
) -> list[str]:
    """Validate the deliberately dependency-free review-result schema subset."""

    issues: list[str] = []
    expected_keys = {
        "protocol_id", "review_surface_sha256",
        "implementation_surface_sha256", "citation_audit_sha256",
        "citation_results", "approved", "issues", "summary",
    }
    if set(value) != expected_keys:
        issues.append(
            f"{label} does not match the closed knowledge-review result schema"
        )
    if value.get("protocol_id") != expected_protocol:
        issues.append(f"{label} has the wrong review protocol")
    if value.get("approved") is not True:
        issues.append(f"{label} is not approved")
    if value.get("issues") != []:
        issues.append(f"{label} contains unresolved issues")
    if value.get("summary") != "approved_no_issues":
        issues.append(f"{label} summary is not the fixed approved value")
    surfaces = value.get("review_surface_sha256")
    if surfaces != expected_surfaces:
        issues.append(f"{label} does not bind the exact current pack surfaces")
    if value.get("implementation_surface_sha256") != expected_implementation:
        issues.append(f"{label} does not bind the exact current implementation surface")
    if value.get("citation_audit_sha256") != expected_citation_audit_sha256:
        issues.append(f"{label} does not bind the exact citation-audit artifact")
    rows = value.get("citation_results")
    by_reference: dict[str, dict[str, Any]] = {}
    if not isinstance(rows, list):
        issues.append(f"{label} citation_results is not an array")
        rows = []
    for row in rows:
        if not isinstance(row, dict):
            issues.append(f"{label} contains a malformed citation result")
            continue
        expected_row_keys = {
            "reference_id", "reference_surface_sha256", "verdict",
            "exact_locator_inspected", "declared_version_matched",
            "declared_section_matched", "paraphrase_supported",
            "applicability_not_broader", "issues",
        }
        if set(row) != expected_row_keys:
            issues.append(f"{label} contains a non-canonical citation result")
            continue
        reference_id = row.get("reference_id")
        if not isinstance(reference_id, str) or reference_id in by_reference:
            issues.append(f"{label} contains a duplicate or invalid citation reference")
            continue
        by_reference[reference_id] = row
    if set(by_reference) != set(expected_citations):
        issues.append(f"{label} citation inventory differs from the exact current manifest")
    for reference_id, expected in expected_citations.items():
        row = by_reference.get(reference_id)
        if row is None:
            continue
        if row.get("reference_surface_sha256") != expected["reference_surface_sha256"]:
            issues.append(f"{label} citation {reference_id} has a stale surface hash")
        if (row.get("verdict") != "verified"
                or row.get("declared_version_matched") is not True
                or row.get("issues") != []):
            issues.append(f"{label} citation {reference_id} is not verified")
        if expected.get("reference_kind") == "rule":
            if row.get("exact_locator_inspected") is not True:
                issues.append(
                    f"{label} rule citation {reference_id} lacks section inspection"
                )
            for field in (
                "declared_section_matched", "paraphrase_supported",
                "applicability_not_broader",
            ):
                if row.get(field) is not True:
                    issues.append(
                        f"{label} citation {reference_id} lacks rule check {field}"
                    )
        elif (row.get("exact_locator_inspected") is not False
                or any(row.get(field) is not None for field in (
                    "declared_section_matched", "paraphrase_supported",
                    "applicability_not_broader",
                ))):
            issues.append(
                f"{label} document citation {reference_id} must use null rule checks"
            )
    return issues


_RECEIPT_FIELDS = frozenset({
    "schema_version", "protocol_id", "invocation_id", "thread_id", "model",
    "reasoning_effort", "prompt_sha256", "output_schema_sha256",
    "review_snapshot_sha256", "cache_manifest_sha256",
    "citation_evidence_sha256",
    "chunk_contract_sha256", "model_inspection_contract_sha256",
    "parser_contract_sha256s", "tool_output_token_limit",
    "initial_prompt_bytes", "initial_prompt_token_upper_bound",
    "initial_prompt_max_bytes",
    "official_codex_elf_sha256",
    "boundary_contract",
    "result_sha256", "command_sha256", "raw_event_stream_sha256",
    "event_stream_path",
    "event_stream_sha256",
    "codex_cli_version", "completed", "exit_code",
})
_BOUNDARY_CONTRACT_FIELDS = frozenset({
    "schema_version", "policy", "filesystem_allowlist", "network_enabled",
    "initial_process_path",
    "runtime_manifest", "cache_manifest_sha256", "cache_parent_policy",
    "evidence_parent_policy", "canary_results", "contract_sha256",
})
_RUNTIME_MANIFEST_FIELDS = frozenset({
    "schema_version", "ownership_policy", "executable_relative_path",
    "executable_sha256", "bubblewrap_relative_path", "bubblewrap_sha256",
    "entries", "sha256",
})
_RUNTIME_ENTRY_FIELDS = frozenset({
    "relative_path", "kind", "size", "mode", "sha256",
})
_REVIEW_CANARY_FIELDS = frozenset({
    "cache_read", "runtime_read", "checkout_denied", "auth_denied",
    "external_denied", "peer_sibling_denied", "evidence_denied",
    "cache_write_denied",
})
_REVIEW_FILESYSTEM_ALLOWLIST = [
    {"token": ":minimal", "access": "read"},
    {"token": "$CACHE", "access": "read"},
    {"token": "$CODEX_RUNTIME", "access": "read"},
]
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
_INVOCATION_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{7,127}")


def _expected_boundary_schema_contract() -> dict[str, Any]:
    sha256 = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
    allowlist_rows = [
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["token", "access"],
            "properties": {
                "token": {"const": token},
                "access": {"const": "read"},
            },
        }
        for token in (":minimal", "$CACHE", "$CODEX_RUNTIME")
    ]
    runtime_entry = {
        "type": "object",
        "additionalProperties": False,
        "required": ["relative_path", "kind", "size", "mode", "sha256"],
        "properties": {
            "relative_path": {"type": "string", "minLength": 1},
            "kind": {"enum": ["file", "dir"]},
            "size": {"type": "integer", "minimum": 0},
            "mode": {"type": "string", "pattern": "^[0-7]{4}$"},
            "sha256": sha256,
        },
    }
    runtime_manifest = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version", "ownership_policy", "executable_relative_path",
            "executable_sha256", "bubblewrap_relative_path",
            "bubblewrap_sha256", "entries", "sha256",
        ],
        "properties": {
            "schema_version": {
                "const": REVIEW_RUNTIME_MANIFEST_SCHEMA_VERSION,
            },
            "ownership_policy": {
                "const": REVIEW_RUNTIME_OWNERSHIP_POLICY,
            },
            "executable_relative_path": {
                "const": REVIEW_CODEX_EXECUTABLE_RELATIVE_PATH,
            },
            "executable_sha256": {
                "const": REVIEW_OFFICIAL_CODEX_ELF_SHA256,
            },
            "bubblewrap_relative_path": {
                "const": REVIEW_CODEX_BWRAP_RELATIVE_PATH,
            },
            "bubblewrap_sha256": {
                "const": REVIEW_OFFICIAL_CODEX_BWRAP_SHA256,
            },
            "entries": {
                "type": "array",
                "minItems": 4,
                "maxItems": 4,
                "uniqueItems": True,
                "items": runtime_entry,
            },
            "sha256": sha256,
        },
    }
    canary_results = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "cache_read", "runtime_read", "checkout_denied", "auth_denied",
            "external_denied", "peer_sibling_denied", "evidence_denied",
            "cache_write_denied",
        ],
        "properties": {
            key: {"const": True} for key in (
                "cache_read", "runtime_read", "checkout_denied", "auth_denied",
                "external_denied", "peer_sibling_denied", "evidence_denied",
                "cache_write_denied",
            )
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version", "policy", "filesystem_allowlist",
            "network_enabled", "initial_process_path", "runtime_manifest",
            "cache_manifest_sha256",
            "cache_parent_policy", "evidence_parent_policy", "canary_results",
            "contract_sha256",
        ],
        "properties": {
            "schema_version": {
                "const": REVIEW_BOUNDARY_CONTRACT_SCHEMA_VERSION,
            },
            "policy": {"const": REVIEW_BOUNDARY_POLICY},
            "filesystem_allowlist": {
                "type": "array",
                "minItems": 3,
                "maxItems": 3,
                "prefixItems": allowlist_rows,
                "items": False,
            },
            "network_enabled": {"const": False},
            "initial_process_path": {
                "type": "array",
                "minItems": 3,
                "maxItems": 3,
                "prefixItems": [
                    {"const": item} for item in REVIEW_INITIAL_PROCESS_PATH
                ],
                "items": False,
            },
            "runtime_manifest": runtime_manifest,
            "cache_manifest_sha256": sha256,
            "cache_parent_policy": {"const": REVIEW_CACHE_PARENT_POLICY},
            "evidence_parent_policy": {
                "const": REVIEW_EVIDENCE_PARENT_POLICY,
            },
            "canary_results": canary_results,
            "contract_sha256": sha256,
        },
    }


def _receipt_schema_contract_issues(value: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    properties = value.get("properties")
    if (
        value.get("$schema") != "https://json-schema.org/draft/2020-12/schema"
        or value.get("type") != "object"
        or value.get("additionalProperties") is not False
        or set(value.get("required", [])) != set(_RECEIPT_FIELDS)
        or not isinstance(properties, dict)
        or set(properties) != set(_RECEIPT_FIELDS)
    ):
        return ["knowledge-review receipt schema is not the closed v4 contract"]
    expected_constants = {
        "schema_version": {"const": REVIEW_RECEIPT_SCHEMA_VERSION},
        "model": {"const": REVIEW_MODEL},
        "reasoning_effort": {"const": REVIEW_REASONING_EFFORT},
        "codex_cli_version": {"const": REVIEW_CODEX_CLI_VERSION},
        "tool_output_token_limit": {"const": REVIEW_TOOL_OUTPUT_TOKEN_LIMIT},
        "initial_prompt_max_bytes": {"const": REVIEW_INITIAL_PROMPT_MAX_BYTES},
        "official_codex_elf_sha256": {
            "const": REVIEW_OFFICIAL_CODEX_ELF_SHA256,
        },
        "completed": {"const": True},
        "exit_code": {"const": 0},
    }
    for key, expected in expected_constants.items():
        if properties.get(key) != expected:
            issues.append(f"knowledge-review receipt schema weakens {key}")
    if properties.get("protocol_id") != {"enum": sorted({
        SEMANTIC_REVIEW_PROTOCOL, ADVERSARIAL_REVIEW_PROTOCOL,
    })}:
        # JSON source order is immaterial; enum list order is not semantic.
        protocol = properties.get("protocol_id")
        if not isinstance(protocol, dict) or set(protocol.get("enum", [])) != {
            SEMANTIC_REVIEW_PROTOCOL, ADVERSARIAL_REVIEW_PROTOCOL,
        } or set(protocol) != {"enum"}:
            issues.append("knowledge-review receipt schema weakens protocol_id")
    for key in ("invocation_id", "thread_id"):
        if properties.get(key) != {
            "type": "string", "pattern": "^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$",
        }:
            issues.append(f"knowledge-review receipt schema weakens {key}")
    for key in (
        "prompt_sha256", "output_schema_sha256", "result_sha256",
        "review_snapshot_sha256", "citation_evidence_sha256",
        "cache_manifest_sha256", "command_sha256",
        "raw_event_stream_sha256", "event_stream_sha256",
        "chunk_contract_sha256", "model_inspection_contract_sha256",
    ):
        if properties.get(key) != {
            "type": "string", "pattern": "^[0-9a-f]{64}$",
        }:
            issues.append(f"knowledge-review receipt schema weakens {key}")
    trace_property = properties.get("event_stream_path")
    if (not isinstance(trace_property, dict)
            or set(trace_property) != {"enum"}
            or set(trace_property.get("enum", [])) != {
                SEMANTIC_REVIEW_TRACE_PATH, ADVERSARIAL_REVIEW_TRACE_PATH,
            }):
        issues.append("knowledge-review receipt schema weakens event_stream_path")
    if properties.get("boundary_contract") != _expected_boundary_schema_contract():
        issues.append("knowledge-review receipt schema weakens boundary_contract")
    parser_contracts = properties.get("parser_contract_sha256s")
    if parser_contracts != {
        "type": "array", "uniqueItems": True,
        "items": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
    }:
        issues.append("knowledge-review receipt schema weakens parser contracts")
    bounded_integer = {
        "type": "integer", "minimum": 1,
        "maximum": REVIEW_INITIAL_PROMPT_MAX_BYTES,
    }
    for key in ("initial_prompt_bytes", "initial_prompt_token_upper_bound"):
        if properties.get(key) != bounded_integer:
            issues.append(f"knowledge-review receipt schema weakens {key}")
    return issues


def _review_contract_sha256(value: dict[str, Any], hash_field: str) -> str:
    payload = dict(value)
    payload.pop(hash_field, None)
    data = (json.dumps(
        payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False,
    ) + "\n").encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _runtime_manifest_contract_issues(
    value: object, *, label: str,
) -> list[str]:
    if not isinstance(value, dict):
        return [f"{label} runtime_manifest is not an object"]
    issues: list[str] = []
    if set(value) != _RUNTIME_MANIFEST_FIELDS:
        issues.append(f"{label} runtime_manifest is not a closed contract")
    if value.get("schema_version") != REVIEW_RUNTIME_MANIFEST_SCHEMA_VERSION:
        issues.append(f"{label} runtime_manifest has an invalid schema_version")
    if value.get("ownership_policy") != REVIEW_RUNTIME_OWNERSHIP_POLICY:
        issues.append(f"{label} runtime_manifest has an invalid ownership_policy")
    entries = value.get("entries")
    if not isinstance(entries, list) or not entries:
        issues.append(f"{label} runtime_manifest has no entries")
        entries = []
    elif len(entries) != 4:
        issues.append(
            f"{label} runtime_manifest is not the exact Codex+bwrap runtime"
        )
    paths: list[str] = []
    for index, entry in enumerate(entries):
        entry_label = f"{label} runtime_manifest entry {index}"
        if not isinstance(entry, dict):
            issues.append(f"{entry_label} is not an object")
            continue
        if set(entry) != _RUNTIME_ENTRY_FIELDS:
            issues.append(f"{entry_label} is not a closed contract")
        relative = entry.get("relative_path")
        if not isinstance(relative, str) or not relative:
            issues.append(f"{entry_label} has an invalid relative_path")
        else:
            paths.append(relative)
            path = PurePosixPath(relative)
            if relative != "." and (
                "\\" in relative
                or path.is_absolute()
                or path.as_posix() != relative
                or any(part in {"", ".", ".."} for part in path.parts)
            ):
                issues.append(f"{entry_label} has an unsafe relative_path")
        kind = entry.get("kind")
        if kind not in {"file", "dir"}:
            issues.append(f"{entry_label} has an invalid kind")
        size = entry.get("size")
        if type(size) is not int or size < 0:
            issues.append(f"{entry_label} has an invalid size")
        mode = entry.get("mode")
        if not isinstance(mode, str) or re.fullmatch(r"[0-7]{4}", mode) is None:
            issues.append(f"{entry_label} has an invalid mode")
            parsed_mode = None
        else:
            parsed_mode = int(mode, 8)
            if parsed_mode & 0o222:
                issues.append(f"{entry_label} is writable")
        digest = entry.get("sha256")
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            issues.append(f"{entry_label} has an invalid sha256")
        if kind == "dir" and (size != 0 or digest != _EMPTY_SHA256):
            issues.append(f"{entry_label} has invalid directory content metadata")
        if kind == "dir" and parsed_mode is not None and parsed_mode & 0o500 != 0o500:
            issues.append(f"{entry_label} is not owner-readable/traversable")
        if kind == "file" and parsed_mode is not None and parsed_mode & 0o400 == 0:
            issues.append(f"{entry_label} is not owner-readable")
        if relative == "." and (index != 0 or kind != "dir"):
            issues.append(f"{entry_label} is not the canonical runtime root")
    if not paths or paths[0] != "." or paths.count(".") != 1:
        issues.append(f"{label} runtime_manifest lacks one canonical root entry")
    entries_by_path = {
        str(entry.get("relative_path")): entry
        for entry in entries if isinstance(entry, dict)
        and isinstance(entry.get("relative_path"), str)
    }
    root_entry = entries_by_path.get(".")
    if not isinstance(root_entry, dict) or root_entry.get("mode") != "0500":
        issues.append(f"{label} runtime_manifest root is not frozen mode 0500")
    for relative, entry in entries_by_path.items():
        if relative == ".":
            continue
        parent = PurePosixPath(relative).parent.as_posix()
        parent_entry = entries_by_path.get(parent)
        if not isinstance(parent_entry, dict) or parent_entry.get("kind") != "dir":
            issues.append(f"{label} runtime_manifest has an incomplete directory tree")
            break
    executable_relative = value.get("executable_relative_path")
    executable_sha256 = value.get("executable_sha256")
    executable_entry = entries_by_path.get(str(executable_relative))
    bubblewrap_relative = value.get("bubblewrap_relative_path")
    bubblewrap_sha256 = value.get("bubblewrap_sha256")
    bubblewrap_entry = entries_by_path.get(str(bubblewrap_relative))
    resources_relative = PurePosixPath(
        REVIEW_CODEX_BWRAP_RELATIVE_PATH
    ).parent.as_posix()
    if (executable_relative != REVIEW_CODEX_EXECUTABLE_RELATIVE_PATH
            or set(entries_by_path) != {
                ".", REVIEW_CODEX_EXECUTABLE_RELATIVE_PATH,
                resources_relative, REVIEW_CODEX_BWRAP_RELATIVE_PATH,
            }
            or not isinstance(executable_entry, dict)
            or executable_entry.get("kind") != "file"
            or executable_entry.get("mode") != "0500"
            or not isinstance(executable_sha256, str)
            or _SHA256_RE.fullmatch(executable_sha256) is None
            or executable_entry.get("sha256") != executable_sha256):
        issues.append(f"{label} runtime_manifest lacks its exact executable identity")
    if executable_sha256 != REVIEW_OFFICIAL_CODEX_ELF_SHA256:
        issues.append(f"{label} runtime_manifest does not use the fixed official Codex ELF")
    if (bubblewrap_relative != REVIEW_CODEX_BWRAP_RELATIVE_PATH
            or not isinstance(bubblewrap_entry, dict)
            or bubblewrap_entry.get("kind") != "file"
            or bubblewrap_entry.get("mode") != "0500"
            or not isinstance(bubblewrap_sha256, str)
            or _SHA256_RE.fullmatch(bubblewrap_sha256) is None
            or bubblewrap_entry.get("sha256") != bubblewrap_sha256):
        issues.append(f"{label} runtime_manifest lacks its exact bundled bwrap identity")
    if bubblewrap_sha256 != REVIEW_OFFICIAL_CODEX_BWRAP_SHA256:
        issues.append(f"{label} runtime_manifest does not use the fixed official bwrap")
    resources_entry = entries_by_path.get(resources_relative)
    if (not isinstance(resources_entry, dict)
            or resources_entry.get("kind") != "dir"
            or resources_entry.get("mode") != "0500"):
        issues.append(f"{label} runtime_manifest has an invalid resources directory")
    if len(paths) != len(set(paths)):
        issues.append(f"{label} runtime_manifest has duplicate relative_path values")
    if paths != sorted(paths):
        issues.append(f"{label} runtime_manifest entries are not canonically ordered")
    manifest_sha256 = value.get("sha256")
    if (
        not isinstance(manifest_sha256, str)
        or _SHA256_RE.fullmatch(manifest_sha256) is None
        or manifest_sha256 != _review_contract_sha256(value, "sha256")
    ):
        issues.append(f"{label} runtime_manifest has an invalid sha256")
    return issues


def _boundary_contract_issues(
    value: object, *, label: str, expected_cache_manifest_sha256: str,
) -> list[str]:
    if not isinstance(value, dict):
        return [f"{label} boundary_contract is not an object"]
    issues: list[str] = []
    if set(value) != _BOUNDARY_CONTRACT_FIELDS:
        issues.append(f"{label} boundary_contract is not a closed contract")
    expected_scalars = {
        "schema_version": REVIEW_BOUNDARY_CONTRACT_SCHEMA_VERSION,
        "policy": REVIEW_BOUNDARY_POLICY,
        "filesystem_allowlist": _REVIEW_FILESYSTEM_ALLOWLIST,
        "network_enabled": False,
        "initial_process_path": REVIEW_INITIAL_PROCESS_PATH,
        "cache_manifest_sha256": expected_cache_manifest_sha256,
        "cache_parent_policy": REVIEW_CACHE_PARENT_POLICY,
        "evidence_parent_policy": REVIEW_EVIDENCE_PARENT_POLICY,
        "canary_results": {key: True for key in _REVIEW_CANARY_FIELDS},
    }
    for key, expected in expected_scalars.items():
        if value.get(key) != expected:
            issues.append(f"{label} boundary_contract has invalid {key}")
    canaries = value.get("canary_results")
    if not isinstance(canaries, dict) or set(canaries) != _REVIEW_CANARY_FIELDS:
        issues.append(f"{label} boundary_contract canary_results is not closed")
    issues.extend(_runtime_manifest_contract_issues(value.get("runtime_manifest"), label=label))
    contract_sha256 = value.get("contract_sha256")
    if (
        not isinstance(contract_sha256, str)
        or _SHA256_RE.fullmatch(contract_sha256) is None
        or contract_sha256 != _review_contract_sha256(value, "contract_sha256")
    ):
        issues.append(f"{label} boundary_contract has an invalid contract_sha256")
    return list(dict.fromkeys(issues))


def _review_receipt_contract_issues(
    value: dict[str, Any], *, label: str, expected_protocol: str,
    expected_prompt_sha256: str, expected_schema_sha256: str,
    expected_result_sha256: str, expected_event_stream_path: str,
    expected_event_stream_sha256: str, expected_raw_event_stream_sha256: str,
    expected_invocation_id: str, expected_thread_id: str,
    expected_command_sha256: str, expected_snapshot_sha256: str,
    expected_citation_evidence_sha256: str,
    expected_cache_manifest_sha256: str,
    expected_chunk_contract_sha256: str,
    expected_model_inspection_contract_sha256: str,
    expected_parser_contract_sha256s: list[str],
    expected_prompt_bytes: int,
) -> list[str]:
    issues: list[str] = []
    if set(value) != _RECEIPT_FIELDS:
        issues.append(f"{label} does not match the closed CLI receipt contract")
    expected_scalars = {
        "schema_version": REVIEW_RECEIPT_SCHEMA_VERSION,
        "protocol_id": expected_protocol,
        "model": REVIEW_MODEL,
        "reasoning_effort": REVIEW_REASONING_EFFORT,
        "prompt_sha256": expected_prompt_sha256,
        "output_schema_sha256": expected_schema_sha256,
        "review_snapshot_sha256": expected_snapshot_sha256,
        "citation_evidence_sha256": expected_citation_evidence_sha256,
        "cache_manifest_sha256": expected_cache_manifest_sha256,
        "chunk_contract_sha256": expected_chunk_contract_sha256,
        "model_inspection_contract_sha256": (
            expected_model_inspection_contract_sha256
        ),
        "parser_contract_sha256s": expected_parser_contract_sha256s,
        "tool_output_token_limit": REVIEW_TOOL_OUTPUT_TOKEN_LIMIT,
        "initial_prompt_bytes": expected_prompt_bytes,
        "initial_prompt_token_upper_bound": expected_prompt_bytes,
        "initial_prompt_max_bytes": REVIEW_INITIAL_PROMPT_MAX_BYTES,
        "official_codex_elf_sha256": REVIEW_OFFICIAL_CODEX_ELF_SHA256,
        "result_sha256": expected_result_sha256,
        "command_sha256": expected_command_sha256,
        "raw_event_stream_sha256": expected_raw_event_stream_sha256,
        "event_stream_path": expected_event_stream_path,
        "event_stream_sha256": expected_event_stream_sha256,
        "invocation_id": expected_invocation_id,
        "thread_id": expected_thread_id,
        "codex_cli_version": REVIEW_CODEX_CLI_VERSION,
        "completed": True,
        "exit_code": 0,
    }
    for key, expected in expected_scalars.items():
        if value.get(key) != expected:
            issues.append(f"{label} has invalid {key}")
    for key in ("invocation_id", "thread_id"):
        token = value.get(key)
        if not isinstance(token, str) or _INVOCATION_TOKEN_RE.fullmatch(token) is None:
            issues.append(f"{label} has invalid {key}")
    for key in (
        "review_snapshot_sha256", "citation_evidence_sha256",
        "cache_manifest_sha256", "command_sha256",
        "raw_event_stream_sha256", "event_stream_sha256",
        "chunk_contract_sha256", "model_inspection_contract_sha256",
    ):
        if _SHA256_RE.fullmatch(str(value.get(key, ""))) is None:
            issues.append(f"{label} has invalid {key}")
    issues.extend(_boundary_contract_issues(
        value.get("boundary_contract"), label=label,
        expected_cache_manifest_sha256=expected_cache_manifest_sha256,
    ))
    parser_contracts = value.get("parser_contract_sha256s")
    if (not isinstance(parser_contracts, list)
            or parser_contracts != sorted(set(parser_contracts))
            or any(_SHA256_RE.fullmatch(str(item)) is None
                   for item in parser_contracts)):
        issues.append(f"{label} has invalid parser_contract_sha256s")
    return issues


def _receipt_invocation_projection(
    receipt: dict[str, Any], receipt_bytes: bytes,
) -> dict[str, Any]:
    keys = (
        "protocol_id", "invocation_id", "thread_id", "model",
        "reasoning_effort", "prompt_sha256", "output_schema_sha256",
        "review_snapshot_sha256", "citation_evidence_sha256",
        "cache_manifest_sha256", "chunk_contract_sha256",
        "model_inspection_contract_sha256", "parser_contract_sha256s",
        "tool_output_token_limit", "initial_prompt_bytes",
        "initial_prompt_token_upper_bound", "initial_prompt_max_bytes",
        "official_codex_elf_sha256",
        "result_sha256", "command_sha256", "event_stream_path",
        "raw_event_stream_sha256", "event_stream_sha256",
        "codex_cli_version",
    )
    result = {key: receipt.get(key) for key in keys}
    boundary = receipt.get("boundary_contract")
    result["boundary_contract_sha256"] = (
        boundary.get("contract_sha256") if isinstance(boundary, dict) else None
    )
    runtime = boundary.get("runtime_manifest") if isinstance(boundary, dict) else None
    result["runtime_manifest_sha256"] = (
        runtime.get("sha256") if isinstance(runtime, dict) else None
    )
    result["cli_receipt_sha256"] = hashlib.sha256(receipt_bytes).hexdigest()
    return result


def _runtime_pack_contract_issues(
    pack_rows: list[tuple[Path, dict[str, Any]]],
) -> list[str]:
    """Run the same public loader and review-ready contract used at runtime."""

    issues: list[str] = []
    runtime_packs = []
    for path, value in pack_rows:
        try:
            runtime_pack = load_pack(value)
        except (TypeError, KeyError, ValueError) as exc:
            issues.append(f"knowledge pack {path.name} fails the runtime contract: {exc}")
            continue
        runtime_packs.append(runtime_pack)
    if len(runtime_packs) != len(pack_rows):
        return issues
    ids = [pack.pack_id for pack in runtime_packs]
    if len(set(ids)) != len(ids):
        issues.append("runtime knowledge catalog contains duplicate pack IDs")
        return issues
    catalog = KnowledgeCatalog(runtime_packs)
    if {pack.pack_id for pack in catalog.packs} != {
        str(value.get("pack_id")) for _path, value in pack_rows
    }:
        issues.append("runtime knowledge catalog inventory differs from reviewed packs")
    for pack in catalog.packs:
        coverage = pack.coverage
        if not pack.review_ready:
            issues.append(f"knowledge pack {pack.pack_id} is not runtime review_ready")
        if coverage is None or not coverage.complete:
            issues.append(f"knowledge pack {pack.pack_id} has incomplete runtime coverage")
            continue
        if any(str(entry.status) == "deferred" for entry in coverage.entries):
            issues.append(f"knowledge pack {pack.pack_id} retains deferred coverage")
    return issues


def _audit_citation_release_artifact(
    root: Path, *, expected_surfaces: dict[str, str],
) -> tuple[list[str], bytes, dict[str, dict[str, Any]]]:
    """Validate the fixed online citation inventory without performing network I/O."""

    issues: list[str] = []
    try:
        citation, data = _strict_json_object(
            root / CITATION_AUDIT_PATH, "knowledge citation-audit artifact", root=root,
        )
        generator_bytes = _strict_file_bytes(
            root / "tools/audit_knowledge_citations.py",
            "knowledge citation-audit generator", root=root,
        )
        citation_root = Path(_knowledge_citation_audit.ROOT).resolve()
    except (AttributeError, OSError, ValueError) as exc:
        return [str(exc)], b"", {}
    if citation_root != root:
        return ["knowledge citation-audit generator is not bound to the audited root"], data, {}
    if (citation.get("schema_version") != "hlsgraph.knowledge-citation-audit.v2"
            or citation.get("mode") != "online"
            or citation.get("passed") is not True):
        issues.append("fixed citation-audit artifact is not a passed online v2 audit")
    generator = citation.get("generator")
    expected_generator = {
        "path": "tools/audit_knowledge_citations.py",
        "sha256": hashlib.sha256(generator_bytes).hexdigest(),
    }
    if generator != expected_generator:
        issues.append("citation-audit artifact does not bind its exact generator bytes")
    unhashed = dict(citation)
    manifest_sha256 = unhashed.pop("manifest_sha256", None)
    try:
        expected_manifest_sha256 = _knowledge_citation_audit._manifest_hash(unhashed)
    except (TypeError, ValueError) as exc:
        issues.append(f"cannot recompute citation-audit manifest hash: {exc}")
    else:
        if manifest_sha256 != expected_manifest_sha256:
            issues.append("citation-audit manifest_sha256 is inconsistent")
    try:
        offline = _knowledge_citation_audit.audit_builtin_citations(online=False)
    except (OSError, TypeError, ValueError) as exc:
        issues.append(f"cannot recompute current citation inventory: {exc}")
        offline = {}
    if citation.get("packs") != offline.get("packs"):
        issues.append("citation-audit pack surfaces differ from the current packs")
    if citation.get("references") != offline.get("references"):
        issues.append("citation-audit references differ from the current pack inventory")
    if citation.get("surface_policy") != offline.get("surface_policy"):
        issues.append("citation-audit surface hashing policy differs from the generator")
    if citation.get("policy") != offline.get("policy"):
        issues.append("citation-audit locator policy differs from the generator defaults")
    artifact_surfaces = {
        item.get("pack_id"): item.get("review_surface_sha256")
        for item in citation.get("packs", []) if isinstance(item, dict)
    } if isinstance(citation.get("packs"), list) else {}
    if artifact_surfaces != expected_surfaces:
        issues.append("citation-audit pack surface mapping is stale or incomplete")
    references = citation.get("references")
    expected_citations: dict[str, dict[str, Any]] = {}
    if not isinstance(references, list):
        issues.append("citation-audit references is not an array")
        references = []
    for row in references:
        if not isinstance(row, dict):
            issues.append("citation-audit contains a malformed reference")
            continue
        reference_id = row.get("reference_id")
        surface = row.get("reference_surface_sha256")
        if (_SHA256_RE.fullmatch(str(reference_id or "")) is None
                or _SHA256_RE.fullmatch(str(surface or "")) is None
                or reference_id in expected_citations):
            issues.append("citation-audit contains a duplicate or invalid reference ID")
            continue
        if row.get("offline_status") != "pass" or row.get("issues") != []:
            issues.append(f"citation-audit reference {reference_id} failed locator policy")
        expected_citations[str(reference_id)] = row
    fetches = citation.get("fetches")
    if not isinstance(fetches, list) or not fetches:
        issues.append("online citation-audit artifact has no fetch records")
        fetches = []
    expected_fetch_urls = {
        str(row.get("fetch_url")) for row in expected_citations.values()
    }
    observed_fetch_urls: set[str] = set()
    for row in fetches:
        if (not isinstance(row, dict) or row.get("body_stored") is not False
                or row.get("issues") != [] or row.get("verification_level") == "failed"
                or not isinstance(row.get("status"), int)
                or not 200 <= row["status"] < 300):
            issues.append("online citation-audit contains a failed or body-retaining fetch")
            break
        fetch_url = row.get("fetch_url")
        final_url = row.get("final_url")
        if not isinstance(fetch_url, str) or fetch_url in observed_fetch_urls:
            issues.append("online citation-audit has a duplicate or invalid fetch URL")
            continue
        observed_fetch_urls.add(fetch_url)
        try:
            fetch_parts = urlsplit(fetch_url)
            final_parts = urlsplit(str(final_url))
        except ValueError:
            issues.append("online citation-audit has an invalid redirect locator")
            continue
        if (fetch_url not in expected_fetch_urls
                or fetch_parts.scheme.casefold() != "https"
                or final_parts.scheme.casefold() != "https"
                or not fetch_parts.hostname
                or fetch_parts.hostname.casefold()
                != (final_parts.hostname or "").casefold()):
            issues.append("online citation-audit leaves an exact same-host locator")
    if observed_fetch_urls != expected_fetch_urls:
        issues.append("online citation-audit fetch inventory differs from exact references")
    evidence = citation.get("document_evidence")
    if not isinstance(evidence, list) or not evidence:
        issues.append("citation-audit has no document evidence records")
        evidence = []
    for row in evidence:
        if (not isinstance(row, dict) or row.get("body_stored") is not False
                or row.get("evidence_sha256_is_document_body_hash") is not False
                or "failed" in row.get("verification_levels", [])):
            issues.append("citation-audit document evidence violates metadata-only policy")
            break
    expected_document_keys = {
        str(row.get("document_key")) for row in offline.get("document_evidence", [])
        if isinstance(row, dict)
    }
    observed_document_keys = {
        str(row.get("document_key")) for row in evidence if isinstance(row, dict)
    }
    if (len(evidence) != len(expected_document_keys)
            or observed_document_keys != expected_document_keys):
        issues.append("citation-audit document evidence inventory is incomplete")
    for policy_key in ("policy", "document_evidence_policy"):
        policy = citation.get(policy_key)
        if not isinstance(policy, dict) or policy.get("response_bodies_stored") is not False:
            issues.append(f"citation-audit {policy_key} does not prohibit body storage")
    summary = citation.get("summary")
    if not isinstance(summary, dict):
        issues.append("citation-audit summary is missing")
    else:
        expected_summary = offline.get("summary", {})
        for key in (
            "pack_count", "document_references", "document_evidence_records",
            "offline_failures", "rule_references", "reference_count",
        ):
            if summary.get(key) != expected_summary.get(key):
                issues.append(f"citation-audit summary has invalid {key}")
        if (summary.get("fetch_failures") != 0
                or summary.get("unique_fetch_urls") != len(fetches)):
            issues.append("citation-audit online fetch summary is inconsistent")
    return list(dict.fromkeys(issues)), data, expected_citations


def _required_review_read_paths(root: Path, prompt_path: str) -> set[str]:
    protocol_id = (
        SEMANTIC_REVIEW_PROTOCOL
        if prompt_path == SEMANTIC_REVIEW_PROMPT_PATH
        else ADVERSARIAL_REVIEW_PROTOCOL
    )
    return _knowledge_review_runner.required_read_paths(root, protocol_id)


def _audit_citation_evidence_mapping(
    root: Path, *, citation_bytes: bytes,
    expected_citations: dict[str, dict[str, Any]],
) -> tuple[list[str], bytes, dict[str, dict[str, Any]]]:
    """Independently validate the closed public citation/evidence map."""

    issues: list[str] = []
    try:
        schema, schema_bytes = _strict_json_object(
            root / CITATION_EVIDENCE_SCHEMA_PATH,
            "citation evidence mapping schema", root=root,
        )
        value, data = _strict_json_object(
            root / CITATION_EVIDENCE_PATH,
            "citation evidence mapping", root=root,
        )
    except (OSError, ValueError) as exc:
        return [str(exc)], b"", {}
    if hashlib.sha256(schema_bytes).hexdigest() != CITATION_EVIDENCE_SCHEMA_SHA256:
        issues.append("citation evidence schema bytes differ from the closed v1 contract")
    properties = schema.get("properties")
    if (schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema"
            or schema.get("type") != "object"
            or schema.get("additionalProperties") is not False
            or set(schema.get("required", [])) != {
                "schema_version", "citation_audit_sha256", "entries",
            }
            or not isinstance(properties, dict)
            or set(properties) != {
                "schema_version", "citation_audit_sha256", "entries",
            }
            or properties.get("schema_version") != {
                "const": CITATION_EVIDENCE_SCHEMA_VERSION,
            }
            or properties.get("citation_audit_sha256") != {
                "pattern": "^[0-9a-f]{64}$", "type": "string",
            }):
        issues.append("citation evidence schema is not the closed v1 contract")
    if set(value) != {"schema_version", "citation_audit_sha256", "entries"}:
        issues.append("citation evidence mapping is not a closed contract")
    if value.get("schema_version") != CITATION_EVIDENCE_SCHEMA_VERSION:
        issues.append("citation evidence mapping has the wrong schema version")
    if value.get("citation_audit_sha256") != hashlib.sha256(citation_bytes).hexdigest():
        issues.append("citation evidence mapping does not bind the citation audit")
    expected_urls = {
        str(row.get("citation_url")) for row in expected_citations.values()
    }
    references_by_url: dict[str, list[dict[str, Any]]] = {}
    for row in expected_citations.values():
        references_by_url.setdefault(str(row.get("citation_url")), []).append(row)

    def reference_binding(row: dict[str, Any]) -> dict[str, Any]:
        section = row.get("section")
        section_bytes = (
            json.dumps(
                section, ensure_ascii=False, sort_keys=True, indent=2,
                allow_nan=False,
            ) + "\n"
        ).encode("utf-8")
        return {
            "reference_id": row.get("reference_id"),
            "reference_kind": row.get("reference_kind"),
            "reference_surface_sha256": row.get("reference_surface_sha256"),
            "document_id": row.get("document_id"),
            "document_version": row.get("document_version"),
            "rule_id": row.get("rule_id"),
            "rule_surface_sha256": row.get("rule_surface_sha256"),
            "section": section,
            "section_sha256": hashlib.sha256(section_bytes).hexdigest(),
        }

    expected_bindings = {
        url: sorted(
            (reference_binding(row) for row in rows),
            key=lambda row: str(row["reference_id"]),
        )
        for url, rows in references_by_url.items()
    }
    try:
        citation_value = json.loads(citation_bytes)
    except (TypeError, ValueError):
        citation_value = {}
    fetches_by_url = {
        str(row.get("fetch_url")): row
        for row in citation_value.get("fetches", [])
        if isinstance(row, dict) and isinstance(row.get("fetch_url"), str)
    } if isinstance(citation_value, dict) else {}
    entries = value.get("entries")
    if not isinstance(entries, list):
        issues.append("citation evidence mapping entries is not an array")
        entries = []
    observed: dict[str, dict[str, Any]] = {}
    amd_citation = re.compile(
        r"^/r/(?P<version>[A-Za-z0-9._-]+)-English/"
        r"(?P<slug>[A-Za-z0-9._-]+)(?P<topic>/.*)?$"
    )
    amd_map = re.compile(r"^/api/khub/maps/(?P<publication>[A-Za-z0-9_~-]+)$")
    amd_topic = re.compile(
        r"^/api/khub/maps/(?P<publication>[A-Za-z0-9_~-]+)/topics/"
        r"(?P<content>[A-Za-z0-9_~-]+)/content$"
    )
    amd_opaque_id = re.compile(r"^[A-Za-z0-9_~-]+$")
    identity_control = re.compile(r"[\x00-\x1f\x7f]")
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {
            "citation_url", "evidence_url", "resolver_id", "identity",
            "reference_bindings",
        }:
            issues.append("citation evidence mapping contains a malformed entry")
            continue
        citation_url = entry.get("citation_url")
        evidence_url = entry.get("evidence_url")
        if (not isinstance(citation_url, str) or citation_url in observed
                or not isinstance(evidence_url, str)):
            issues.append("citation evidence mapping has duplicate or invalid locators")
            continue
        observed[citation_url] = entry
        if entry.get("reference_bindings") != expected_bindings.get(
            citation_url, [],
        ):
            issues.append("citation evidence mapping reference bindings are stale")
            continue
        try:
            citation_parts = urlsplit(citation_url)
            evidence_parts = urlsplit(evidence_url)
            valid_https = (
                citation_parts.scheme.casefold() == "https"
                and evidence_parts.scheme.casefold() == "https"
                and bool(citation_parts.hostname)
                and bool(evidence_parts.hostname)
                and citation_parts.username is None
                and citation_parts.password is None
                and evidence_parts.username is None
                and evidence_parts.password is None
                and citation_parts.port is None and evidence_parts.port is None
            )
        except ValueError:
            valid_https = False
        if not valid_https:
            issues.append("citation evidence mapping leaves same-host HTTPS")
            continue
        citation_host = (citation_parts.hostname or "").casefold()
        evidence_host = (evidence_parts.hostname or "").casefold()
        resolver_id = entry.get("resolver_id")
        if resolver_id == "github.raw.lines.v1":
            github_blob = re.fullmatch(
                r"/(?P<owner>[A-Za-z0-9_.-]+)/(?P<repository>[A-Za-z0-9_.-]+)/"
                r"blob/(?P<commit>[0-9a-f]{40})/(?P<path>[^?#]+)",
                citation_parts.path,
            )
            github_raw = re.fullmatch(
                r"/(?P<owner>[A-Za-z0-9_.-]+)/(?P<repository>[A-Za-z0-9_.-]+)/"
                r"(?P<commit>[0-9a-f]{40})/(?P<path>[^?#]+)",
                evidence_parts.path,
            )
            identity = entry.get("identity")
            expected_identity_fields = {
                "repository", "commit", "path", "source_sha256",
                "start_line", "end_line", "slice_sha256",
            }
            if (citation_host != "github.com"
                    or evidence_host != "raw.githubusercontent.com"
                    or github_blob is None or github_raw is None
                    or citation_parts.query or evidence_parts.query
                    or evidence_parts.fragment
                    or not isinstance(identity, dict)
                    or set(identity) != expected_identity_fields
                    or identity.get("repository") != (
                        github_blob.group("owner") + "/"
                        + github_blob.group("repository")
                    )
                    or identity.get("commit") != github_blob.group("commit")
                    or identity.get("path") != github_blob.group("path")
                    or any(
                        github_raw.group(key) != github_blob.group(key)
                        for key in ("owner", "repository", "commit", "path")
                    )
                    or _SHA256_RE.fullmatch(
                        str(identity.get("source_sha256", "")),
                    ) is None
                    or _SHA256_RE.fullmatch(
                        str(identity.get("slice_sha256", "")),
                    ) is None
                    or type(identity.get("start_line")) is not int
                    or type(identity.get("end_line")) is not int
                    or not 1 <= identity["start_line"] <= identity["end_line"]
                    or identity["end_line"] - identity["start_line"] + 1 > 1024
                    or PurePosixPath(str(identity.get("path", ""))).is_absolute()
                    or any(
                        part in {"", ".", ".."}
                        for part in PurePosixPath(
                            str(identity.get("path", "")),
                        ).parts
                    )
                    or evidence_url != (
                        "https://raw.githubusercontent.com/"
                        f"{identity.get('repository')}/{identity.get('commit')}/"
                        f"{identity.get('path')}"
                    )):
                issues.append("GitHub raw line evidence identity does not close")
            rule_sections = [
                str(row["section"]) for row in references_by_url.get(citation_url, [])
                if row.get("reference_kind") == "rule"
                and isinstance(row.get("section"), str)
            ]
            aliases = {
                "Blocks and Regions": "blocks",
            }
            anchors = {
                re.sub(r"[-\s]+", "-", re.sub(
                    r"[^a-z0-9\s-]", "", section.strip().casefold(),
                )).strip("-")
                for section in rule_sections
            } | {
                aliases[section] for section in rule_sections if section in aliases
            }
            if not rule_sections or (
                citation_parts.fragment and citation_parts.fragment not in anchors
            ):
                issues.append(
                    "GitHub raw line evidence does not bind its declared rule section"
                )
            continue
        if resolver_id == "github.raw.document.v1":
            github_raw = re.fullmatch(
                r"/(?P<owner>[A-Za-z0-9_.-]+)/(?P<repository>[A-Za-z0-9_.-]+)/"
                r"(?P<commit>[0-9a-f]{40})/(?P<path>[^?#]+)",
                evidence_parts.path,
            )
            identity = entry.get("identity")
            canonical_sources = {
                "https://github.com/EPFL-LAP/dynamatic/blob/"
                "4dd0bbc86aa55d01854b93fbbc8b818cc318ea80/"
                "docs/DeveloperGuide/CompilerIntrinsics/MLIRPrimer.md": (
                    "EPFL-LAP/dynamatic",
                    "docs/DeveloperGuide/CompilerIntrinsics/MLIRPrimer.md",
                ),
                "https://llvm.org/docs/LangRef.html": (
                    "llvm/llvm-project", "llvm/docs/LangRef.md",
                ),
                "https://llvm.org/docs/SourceLevelDebugging.html": (
                    "llvm/llvm-project", "llvm/docs/SourceLevelDebugging.md",
                ),
                "https://mlir.llvm.org/docs/LangRef/": (
                    "llvm/llvm-project", "mlir/docs/LangRef.md",
                ),
            }
            document_rows = [
                row for row in references_by_url.get(citation_url, [])
                if row.get("reference_kind") == "document"
            ]
            if (evidence_host != "raw.githubusercontent.com"
                    or github_raw is None
                    or citation_parts.query or citation_parts.fragment
                    or evidence_parts.query or evidence_parts.fragment
                    or not isinstance(identity, dict)
                    or set(identity) != {
                        "repository", "commit", "path", "source_sha256",
                        "source_size", "document_id", "document_version",
                    }
                    or canonical_sources.get(citation_url) != (
                        identity.get("repository"), identity.get("path"),
                    )
                    or len(document_rows) != 1
                    or identity.get("document_id")
                    != document_rows[0].get("document_id")
                    or identity.get("document_version")
                    != document_rows[0].get("document_version")
                    or identity.get("document_version")
                    != f"git-{identity.get('commit')}"
                    or github_raw.group("owner") + "/"
                    + github_raw.group("repository")
                    != identity.get("repository")
                    or github_raw.group("commit") != identity.get("commit")
                    or github_raw.group("path") != identity.get("path")
                    or _SHA256_RE.fullmatch(
                        str(identity.get("source_sha256", "")),
                    ) is None
                    or type(identity.get("source_size")) is not int
                    or identity["source_size"] <= 0
                    or PurePosixPath(str(identity.get("path", ""))).is_absolute()
                    or any(
                        part in {"", ".", ".."}
                        for part in PurePosixPath(
                            str(identity.get("path", "")),
                        ).parts
                    )
                    or evidence_url != (
                        "https://raw.githubusercontent.com/"
                        f"{identity.get('repository')}/{identity.get('commit')}/"
                        f"{identity.get('path')}"
                    )):
                issues.append("GitHub raw document identity does not close")
            continue
        if resolver_id == "direct.sha256.v1":
            identity = entry.get("identity")
            document_rows = [
                row for row in references_by_url.get(citation_url, [])
                if row.get("reference_kind") == "document"
            ]
            fetched = fetches_by_url.get(citation_url)
            if (citation_host != evidence_host or evidence_url != citation_url
                    or citation_parts.query or citation_parts.fragment
                    or not isinstance(identity, dict)
                    or set(identity) != {
                        "document_id", "document_version", "body_sha256",
                        "body_size", "content_type",
                    }
                    or len(document_rows) != 1
                    or identity.get("document_id")
                    != document_rows[0].get("document_id")
                    or identity.get("document_version")
                    != document_rows[0].get("document_version")
                    or _SHA256_RE.fullmatch(
                        str(identity.get("body_sha256", "")),
                    ) is None
                    or type(identity.get("body_size")) is not int
                    or identity["body_size"] <= 0
                    or identity.get("content_type") not in {
                        "application/pdf", "text/html",
                    }
                    or not isinstance(fetched, dict)
                    or fetched.get("status") != 200
                    or fetched.get("final_url") != citation_url
                    or fetched.get("sha256") != identity.get("body_sha256")
                    or fetched.get("byte_count") != identity.get("body_size")
                    or fetched.get("content_type")
                    != identity.get("content_type")):
                issues.append("direct SHA-256 document identity does not close")
            continue
        if citation_host != evidence_host:
            issues.append("citation evidence mapping leaves same-host HTTPS")
            continue
        if citation_host != "docs.amd.com":
            issues.append("non-AMD citation does not use an approved resolver")
            continue
        citation_match = amd_citation.fullmatch(citation_parts.path)
        identity = entry.get("identity")
        if citation_match is None or not isinstance(identity, dict):
            issues.append("AMD citation evidence identity is malformed")
            continue
        version = citation_match.group("version")
        slug = citation_match.group("slug")
        is_root = citation_match.group("topic") in (None, "", "/")
        evidence_match = (
            amd_map.fullmatch(evidence_parts.path)
            if is_root else amd_topic.fullmatch(evidence_parts.path)
        )
        base_fields = {
            "publication_id", "document_id", "document_slug", "version", "title",
        }
        expected_fields = base_fields if is_root else base_fields | {
            "toc_id", "content_id", "topic_title",
        }
        if (set(identity) != expected_fields or evidence_match is None
                or identity.get("version") != version
                or identity.get("document_slug") != slug
                or identity.get("publication_id")
                != evidence_match.group("publication")
                or any(not isinstance(identity.get(key), str) or not identity.get(key)
                       for key in expected_fields)
                or any(
                    not isinstance(identity.get(key), str)
                    or len(identity[key]) > 256
                    or identity_control.search(identity[key]) is not None
                    for key in expected_fields
                )
                or any(
                    amd_opaque_id.fullmatch(str(identity.get(key, ""))) is None
                    for key in ({"publication_id", "document_id"}
                                if is_root else {
                                    "publication_id", "document_id", "toc_id",
                                    "content_id",
                                })
                )
                or citation_parts.query or citation_parts.fragment
                or evidence_parts.fragment):
            issues.append("AMD citation evidence identity does not close")
            continue
        if is_root:
            if (entry.get("resolver_id") != "amd.docs.khub.map.v1"
                    or evidence_parts.query):
                issues.append("AMD document root does not use exact map evidence")
        elif (entry.get("resolver_id") != "amd.docs.khub.topic.v1"
                or identity.get("content_id") != evidence_match.group("content")
                or parse_qsl(evidence_parts.query, keep_blank_values=True)
                != [("target", "DESIGNED_READER")]):
            issues.append("AMD topic does not use exact KHUB content evidence")
        elif {
            row.get("section") for row in references_by_url.get(citation_url, [])
            if row.get("reference_kind") == "rule"
        } not in (set(), {identity.get("topic_title")}):
            issues.append("AMD topic evidence does not bind its declared rule section")
    if set(observed) != expected_urls or len(entries) != len(expected_urls):
        issues.append("citation evidence mapping inventory is incomplete or has extras")
    if entries != [observed[url] for url in sorted(observed)]:
        issues.append("citation evidence mapping is not uniquely sorted")
    return list(dict.fromkeys(issues)), data, observed


def _audit_cache_evidence_projection(
    manifest: dict[str, Any], mappings: dict[str, dict[str, Any]], *, label: str,
) -> list[str]:
    citations = manifest.get("citations")
    if not isinstance(citations, list):
        return [f"{label} cache has no citation inventory"]
    observed: dict[str, dict[str, Any]] = {}
    for row in citations:
        if not isinstance(row, dict):
            return [f"{label} cache has a malformed citation entry"]
        requested = row.get("requested_url")
        if not isinstance(requested, str) or requested in observed:
            return [f"{label} cache has duplicate citation entries"]
        observed[requested] = row
        mapping = mappings.get(requested)
        if (mapping is None
                or row.get("evidence_url") != mapping.get("evidence_url")
                or row.get("resolver_id") != mapping.get("resolver_id")):
            return [f"{label} cache citation/evidence projection is stale"]
    if set(observed) != set(mappings) or len(citations) != len(mappings):
        return [f"{label} cache citation/evidence inventory has missing or extra rows"]
    return []


def _audit_review_tool_trace(
    root: Path, *, trace_path: str, prompt_path: str, result_bytes: bytes,
    expected_citations: dict[str, dict[str, Any]], snapshot: Any,
    cache: Any,
) -> tuple[list[str], bytes]:
    """Validate a content-free normalized CLI trace against an exact allowlist."""

    issues: list[str] = []
    try:
        rows, data = _strict_json_lines(
            root / trace_path, f"knowledge-review tool trace {trace_path}",
        )
    except (OSError, ValueError) as exc:
        return [str(exc)], b""
    rendered = "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
        for row in rows
    ).encode("utf-8")
    if rendered != data:
        issues.append(f"knowledge-review tool trace {trace_path} is not canonical JSONL")
    required_file_chunks = {
        str(item["path"]): {str(chunk["path"]) for chunk in item["chunks"]}
        for item in cache.manifest.get("files", [])
        if isinstance(item, dict) and item.get("model_inspection_required") is True
    }
    observed_file_chunks: dict[str, set[str]] = {}
    expected_urls = {
        str(row.get("citation_url")) for row in expected_citations.values()
    }
    required_citation_chunks = {
        str(item["requested_url"]): {
            str(chunk["path"]) for chunk in item.get("inspection_chunks", [])
        }
        for item in cache.manifest.get("citations", [])
        if (isinstance(item, dict) and item.get("available") is True
            and item.get("inspection_required") is True)
    }
    observed_citation_chunks: dict[str, set[str]] = {}
    snapshot_files = snapshot.file_map
    cache_citations = {
        str(item.get("requested_url")): item
        for item in cache.manifest.get("citations", []) if isinstance(item, dict)
    }
    result_rows = 0
    for index, row in enumerate(rows, start=1):
        prefix = f"knowledge-review tool trace {trace_path} line {index}"
        if row.get("schema_version") != REVIEW_TRACE_SCHEMA_VERSION:
            issues.append(f"{prefix} has the wrong schema version")
        if row.get("sequence") != index:
            issues.append(f"{prefix} has a non-canonical sequence")
        kind = row.get("kind")
        if kind == "file_chunk_read":
            if set(row) != {
                "schema_version", "sequence", "kind", "path", "hash_kind",
                "sha256", "cache_sha256", "chunk_contract_sha256",
                "chunk_index", "chunk_path", "chunk_sha256", "chunk_size",
                "byte_start", "byte_end",
            }:
                issues.append(f"{prefix} has a malformed frozen-file record")
                continue
            relative = row.get("path")
            if (not isinstance(relative, str) or not relative
                    or "\\" in relative or relative.startswith("/")
                    or re.match(r"^[A-Za-z]:", relative)
                    or any(part in {"", ".", ".."} for part in PurePosixPath(relative).parts)
                    or any(part in SOURCE_SKIP_DIRS for part in PurePosixPath(relative).parts)
                    or _forbidden(relative) is not None):
                issues.append(f"{prefix} reads a non-public or non-canonical path")
                continue
            expected_file = snapshot_files.get(relative)
            if expected_file is None or (
                row.get("hash_kind"), row.get("sha256"), row.get("cache_sha256")
            ) != (
                expected_file.hash_kind, expected_file.sha256,
                expected_file.cache_sha256,
            ):
                issues.append(f"{prefix} has a stale frozen-file hash")
            expected_manifest = next((
                item for item in cache.manifest.get("files", [])
                if isinstance(item, dict) and item.get("path") == relative
            ), None)
            chunks = (
                expected_manifest.get("chunks", [])
                if isinstance(expected_manifest, dict) else []
            )
            expected_chunk = next((
                item for item in chunks
                if isinstance(item, dict) and item.get("path") == row.get("chunk_path")
            ), None)
            expected_chunk_projection = {
                "chunk_index": expected_chunk.get("index") if expected_chunk else None,
                "chunk_sha256": expected_chunk.get("sha256") if expected_chunk else None,
                "chunk_size": expected_chunk.get("size") if expected_chunk else None,
                "byte_start": expected_chunk.get("byte_start") if expected_chunk else None,
                "byte_end": expected_chunk.get("byte_end") if expected_chunk else None,
                "chunk_contract_sha256": cache.manifest["chunk_contract"]["sha256"],
            }
            if (expected_chunk is None or any(
                row.get(key) != value
                for key, value in expected_chunk_projection.items()
            )):
                issues.append(f"{prefix} has stale frozen-file chunk metadata")
            else:
                observed_file_chunks.setdefault(relative, set()).add(
                    str(row["chunk_path"])
                )
        elif kind == "citation_chunk_read":
            if set(row) != {
                "schema_version", "sequence", "kind", "requested_url",
                "evidence_url", "resolver_id",
                "reference_ids", "body_sha256", "inspection_sha256",
                "parser_id", "parser_contract_sha256",
                "chunk_contract_sha256", "chunk_index", "chunk_path",
                "chunk_sha256", "chunk_size", "byte_start", "byte_end",
                "body_stored",
            }:
                issues.append(f"{prefix} has a malformed citation-inspection record")
                continue
            requested = row.get("requested_url")
            if not isinstance(requested, str) or requested not in expected_urls:
                issues.append(f"{prefix} fetches an unapproved locator")
                continue
            expected_cache = cache_citations.get(requested)
            if expected_cache is None or expected_cache.get("available") is not True:
                issues.append(f"{prefix} inspects an unavailable citation")
                continue
            expected_projection = {
                "reference_ids": expected_cache.get("reference_ids"),
                "evidence_url": expected_cache.get("evidence_url"),
                "resolver_id": expected_cache.get("resolver_id"),
                "body_sha256": expected_cache.get("body_sha256"),
                "inspection_sha256": expected_cache.get("inspection_sha256"),
                "parser_id": expected_cache.get("parser_id"),
                "parser_contract_sha256": expected_cache.get("parser_command_sha256"),
                "chunk_contract_sha256": cache.manifest["chunk_contract"]["sha256"],
                "body_stored": False,
            }
            if any(row.get(key) != value for key, value in expected_projection.items()):
                issues.append(f"{prefix} differs from the frozen citation cache")
                continue
            expected_chunk = next((
                item for item in expected_cache.get("inspection_chunks", [])
                if isinstance(item, dict) and item.get("path") == row.get("chunk_path")
            ), None)
            if expected_chunk is None or any(row.get(key) != expected_chunk.get(source) for key, source in (
                ("chunk_index", "index"), ("chunk_sha256", "sha256"),
                ("chunk_size", "size"), ("byte_start", "byte_start"),
                ("byte_end", "byte_end"),
            )):
                issues.append(f"{prefix} has stale citation chunk metadata")
            else:
                observed_citation_chunks.setdefault(requested, set()).add(
                    str(row["chunk_path"])
                )
        elif kind in {"file_chunk_hash", "citation_chunk_hash"}:
            # Hash rows are content-addressing evidence only and never satisfy
            # the model-inspection completeness sets.
            if kind == "file_chunk_hash" and set(row) != {
                "schema_version", "sequence", "kind", "path", "chunk_index",
                "chunk_path", "chunk_sha256",
            }:
                issues.append(f"{prefix} has a malformed source-chunk hash record")
            elif kind == "citation_chunk_hash" and set(row) != {
                "schema_version", "sequence", "kind", "requested_url",
                "evidence_url", "resolver_id", "inspection_sha256",
                "chunk_index", "chunk_path", "chunk_sha256",
            }:
                issues.append(f"{prefix} has a malformed citation-chunk hash record")
        elif kind == "result_emit":
            result_rows += 1
            if set(row) != {
                "schema_version", "sequence", "kind", "result_sha256",
            } or row.get("result_sha256") != hashlib.sha256(result_bytes).hexdigest():
                issues.append(f"{prefix} does not emit the exact review result")
            if index != len(rows):
                issues.append(f"{prefix} is not the final trace record")
        else:
            issues.append(
                f"{prefix} uses a forbidden search, write, command, or unknown operation"
            )
    missing_reads = sorted(
        path for path, required in required_file_chunks.items()
        if observed_file_chunks.get(path, set()) != required
    )
    if missing_reads:
        issues.append(
            f"knowledge-review tool trace {trace_path} omits reviewed files: {missing_reads!r}"
        )
    incomplete_citations = sorted(
        url for url, required in required_citation_chunks.items()
        if observed_citation_chunks.get(url, set()) != required
    )
    expected_inspection_urls = {
        str(item["requested_url"])
        for item in cache.manifest.get("citations", [])
        if (isinstance(item, dict)
            and item.get("inspection_required") is True)
    }
    if (incomplete_citations
            or set(required_citation_chunks) != expected_inspection_urls):
        issues.append(
            f"knowledge-review tool trace {trace_path} does not inspect every exact locator"
        )
    if result_rows != 1:
        issues.append(
            f"knowledge-review tool trace {trace_path} must emit exactly one result"
        )
    return list(dict.fromkeys(issues)), data


_SUITE_EVIDENCE_FIELDS = frozenset({
    "schema_version", "runtime_manifest", "protocols",
})
_SUITE_EVIDENCE_PROTOCOL_FIELDS = frozenset({
    "protocol_id", "acquisition_mode", "replay_source_manifest_sha256",
    "review_snapshot_sha256", "full_cache_manifest_sha256",
    "full_citation_evidence_surface_sha256", "shards",
})
_SUITE_EVIDENCE_SHARD_FIELDS = frozenset({
    "shard_id", "projected_cache_manifest_sha256", "raw_output_sha256",
    "sanitized_output_sha256", "process_evidence_sha256",
    "boundary_contract",
})


def _suite_protocol_label(protocol_id: str) -> str:
    if protocol_id == SEMANTIC_REVIEW_PROTOCOL:
        return "semantic"
    if protocol_id == ADVERSARIAL_REVIEW_PROTOCOL:
        return "adversarial"
    raise ValueError("suite evidence uses an unknown protocol")


def _validate_full_cache_offline_replay_equivalence(
    semantic: Any, adversarial: Any,
) -> None:
    """Replay every acquired URL and require identical identity and bytes."""

    def urls(cache: Any) -> set[str]:
        result: set[str] = set()
        for citation in cache.manifest["citations"]:
            evidence_url = citation.get("evidence_url")
            if not isinstance(evidence_url, str) or not evidence_url:
                raise ValueError("full cache has a malformed evidence URL")
            result.add(evidence_url)
            for artifact in citation.get("resolver_artifacts", []):
                requested_url = (
                    artifact.get("requested_url")
                    if isinstance(artifact, dict) else None
                )
                if not isinstance(requested_url, str) or not requested_url:
                    raise ValueError("full cache has a malformed resolver URL")
                result.add(requested_url)
        return result

    semantic_urls = urls(semantic)
    adversarial_urls = urls(adversarial)
    if semantic_urls != adversarial_urls:
        raise ValueError(
            "adversarial full cache does not replay the semantic URL inventory"
        )
    semantic_fetch = _knowledge_review_suite_cache.frozen_cache_fetcher(
        semantic,
    )
    adversarial_fetch = _knowledge_review_suite_cache.frozen_cache_fetcher(
        adversarial,
    )
    for url in sorted(semantic_urls):
        left = semantic_fetch(
            url, 1.0, _knowledge_review_runner.MAX_CITATION_BYTES,
        )
        right = adversarial_fetch(
            url, 1.0, _knowledge_review_runner.MAX_CITATION_BYTES,
        )
        identity = lambda item: (
            item.status, item.final_url, item.redirect_chain,
            item.content_type, item.charset, item.content_length,
        )
        if identity(left) != identity(right) or left.body != right.body:
            raise ValueError(
                "adversarial full cache response differs from semantic replay"
            )


def _suite_evidence_paths(
    evidence_root: Path, protocol_id: str, shard_id: str | None = None,
) -> dict[str, Path]:
    """Derive the only accepted external suite layout.

    Paths are never accepted from JSON.  That keeps a syntactically valid
    manifest from redirecting the release auditor to another cache or raw
    stream after the formal review.
    """

    label = _suite_protocol_label(protocol_id)
    root = evidence_root.absolute()
    if shard_id is None:
        return {
            "cache": root / "full" / label / "cache-parent" / "cache",
        }
    if shard_id not in _knowledge_review_shards.SHARD_ORDER:
        raise ValueError("suite evidence uses an unknown shard")
    invocation = root / "invocations" / label / shard_id
    return {
        "cache": invocation / "cache-parent" / "cache",
        "raw": invocation / "evidence" / "raw.jsonl",
        "sanitized": invocation / "evidence" / "sanitized.jsonl",
        "raw_stderr": invocation / "evidence" / "stderr.raw.log",
        "stderr": invocation / "evidence" / "stderr.log",
        "process": invocation / "evidence" / "process.json",
        "invocation": invocation / "evidence" / "invocation.json",
    }


def _validate_suite_evidence_tree(evidence_root: Path) -> None:
    """Require the executor's complete fixed container layout, without extras."""

    root = evidence_root.resolve(strict=True)

    def directory(path: Path, children: set[str], *, mode: int = 0o700) -> None:
        _strict_path_components(path, "knowledge-review suite directory", root=root)
        metadata = path.lstat()
        if (not stat.S_ISDIR(metadata.st_mode) or _is_reparse_or_link(path)):
            raise ValueError("knowledge-review suite layout contains a linked directory")
        if os.name != "nt" and (
            metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != mode
        ):
            raise ValueError("knowledge-review suite directory mode or owner is invalid")
        observed = {item.name for item in path.iterdir()}
        if observed != children:
            raise ValueError(
                "knowledge-review suite layout has missing or unmanifested entries"
            )

    directory(root, {
        REVIEW_SUITE_EVIDENCE_MANIFEST,
        _knowledge_review_suite_executor.PAIR_SEAL_PATH,
        "full", "invocations",
    })
    directory(root / "full", {"semantic", "adversarial"})
    directory(root / "invocations", {"semantic", "adversarial"})
    for protocol_id in (
        SEMANTIC_REVIEW_PROTOCOL, ADVERSARIAL_REVIEW_PROTOCOL,
    ):
        label = _suite_protocol_label(protocol_id)
        directory(root / "full" / label, {"cache-parent"})
        directory(root / "full" / label / "cache-parent", {"cache"})
        _strict_path_components(
            root / "full" / label / "cache-parent" / "cache",
            f"{label} full cache", root=root,
        )
        directory(
            root / "invocations" / label,
            set(_knowledge_review_shards.SHARD_ORDER),
        )
        for shard_id in _knowledge_review_shards.SHARD_ORDER:
            invocation = root / "invocations" / label / shard_id
            directory(invocation, {"cache-parent", "evidence"})
            directory(invocation / "cache-parent", {"cache"})
            _strict_path_components(
                invocation / "cache-parent" / "cache",
                f"{label} {shard_id} projected cache", root=root,
            )
            directory(invocation / "evidence", {
                _knowledge_review_suite_executor.RAW_STREAM_PATH,
                _knowledge_review_suite_executor.SANITIZED_STREAM_PATH,
                _knowledge_review_suite_executor.RAW_STDERR_PATH,
                _knowledge_review_suite_executor.STDERR_PATH,
                _knowledge_review_suite_executor.PROCESS_EVIDENCE_PATH,
                _knowledge_review_suite_executor.INVOCATION_ENVELOPE_PATH,
            })


def _load_suite_evidence_manifest(
    evidence_root: Path,
) -> tuple[dict[str, Any], bytes, dict[str, dict[str, Any]]]:
    """Load the closed, path-free suite evidence inventory.

    This function validates only the external envelope.  Full/projected cache
    bytes and all six raw streams are independently replayed by the v6 gate.
    """

    lexical_root = evidence_root.absolute()
    if _is_reparse_or_link(lexical_root):
        raise ValueError("knowledge-review suite evidence root is linked")
    try:
        root_stat = lexical_root.lstat()
        resolved_root = lexical_root.resolve(strict=True)
    except OSError:
        raise ValueError("knowledge-review suite evidence root is unavailable") from None
    if not stat.S_ISDIR(root_stat.st_mode):
        raise ValueError("knowledge-review suite evidence root is not a directory")
    if os.name != "nt":
        if root_stat.st_uid != os.geteuid():
            raise ValueError("knowledge-review suite evidence root has another owner")
        if stat.S_IMODE(root_stat.st_mode) != 0o700:
            raise ValueError("knowledge-review suite evidence root is not mode 0700")
    _validate_suite_evidence_tree(resolved_root)
    manifest_path = resolved_root / REVIEW_SUITE_EVIDENCE_MANIFEST
    manifest_bytes = _strict_file_bytes(
        manifest_path, "knowledge-review suite evidence manifest",
        root=resolved_root, required_file_mode=0o600,
        required_parent_mode=0o700, require_current_owner=True,
        require_single_link=True, max_bytes=1024 * 1024,
    )
    manifest = _knowledge_review_runner._strict_json_bytes(
        manifest_bytes, label="knowledge-review suite evidence manifest",
    )
    if not isinstance(manifest, dict):
        raise ValueError("knowledge-review suite evidence manifest is not an object")
    if _knowledge_review_runner._canonical_json(manifest) != manifest_bytes:
        raise ValueError("knowledge-review suite evidence manifest is not canonical")
    if (set(manifest) != _SUITE_EVIDENCE_FIELDS
            or manifest.get("schema_version")
            != REVIEW_SUITE_EVIDENCE_SCHEMA_VERSION):
        raise ValueError("knowledge-review suite evidence manifest is not closed")
    runtime = manifest.get("runtime_manifest")
    runtime_issues = _runtime_manifest_contract_issues(
        runtime, label="knowledge-review suite evidence",
    )
    if runtime_issues:
        raise ValueError(runtime_issues[0])
    protocols = manifest.get("protocols")
    expected_protocols = [
        SEMANTIC_REVIEW_PROTOCOL, ADVERSARIAL_REVIEW_PROTOCOL,
    ]
    if (not isinstance(protocols, list) or len(protocols) != 2
            or [row.get("protocol_id") if isinstance(row, dict) else None
                for row in protocols] != expected_protocols):
        raise ValueError("knowledge-review suite evidence protocols are not canonical")
    by_protocol: dict[str, dict[str, Any]] = {}
    for protocol in protocols:
        if (not isinstance(protocol, dict)
                or set(protocol) != _SUITE_EVIDENCE_PROTOCOL_FIELDS
                or any(_SHA256_RE.fullmatch(str(protocol.get(key, ""))) is None
                       for key in (
                           "review_snapshot_sha256",
                           "full_cache_manifest_sha256",
                           "full_citation_evidence_surface_sha256",
                       ))):
            raise ValueError("knowledge-review suite protocol evidence is malformed")
        protocol_id = str(protocol["protocol_id"])
        expected_acquisition = (
            "online_pinned_identity"
            if protocol_id == SEMANTIC_REVIEW_PROTOCOL
            else "offline_replay_from_semantic"
        )
        semantic_manifest_sha256 = str(
            protocols[0].get("full_cache_manifest_sha256", "")
        )
        expected_replay_source = (
            None
            if protocol_id == SEMANTIC_REVIEW_PROTOCOL
            else semantic_manifest_sha256
        )
        if (protocol.get("acquisition_mode") != expected_acquisition
                or protocol.get("replay_source_manifest_sha256")
                != expected_replay_source):
            raise ValueError(
                "knowledge-review suite acquisition provenance is invalid"
            )
        shards = protocol.get("shards")
        if (not isinstance(shards, list) or len(shards) != 3
                or [row.get("shard_id") if isinstance(row, dict) else None
                    for row in shards]
                != list(_knowledge_review_shards.SHARD_ORDER)):
            raise ValueError("knowledge-review suite shard evidence is not canonical")
        for shard in shards:
            if (not isinstance(shard, dict)
                    or set(shard) != _SUITE_EVIDENCE_SHARD_FIELDS
                    or any(_SHA256_RE.fullmatch(str(shard.get(key, ""))) is None
                           for key in (
                               "projected_cache_manifest_sha256",
                               "raw_output_sha256",
                               "sanitized_output_sha256",
                               "process_evidence_sha256",
                           ))):
                raise ValueError("knowledge-review suite shard evidence is malformed")
            boundary_issues = _boundary_contract_issues(
                shard.get("boundary_contract"),
                label=(
                    f"{_suite_protocol_label(protocol_id)} "
                    f"{shard.get('shard_id')} suite evidence"
                ),
                expected_cache_manifest_sha256=str(
                    shard["projected_cache_manifest_sha256"]
                ),
            )
            if boundary_issues:
                raise ValueError(boundary_issues[0])
            boundary = shard["boundary_contract"]
            if boundary.get("runtime_manifest") != runtime:
                raise ValueError(
                    "knowledge-review suite shards did not use one runtime"
                )
        by_protocol[protocol_id] = protocol
    return manifest, manifest_bytes, by_protocol


def _audit_knowledge_review_release_gate_v4(
    root: Path, *, semantic_review: Path, adversarial_review: Path,
    semantic_raw: Path, adversarial_raw: Path,
    semantic_cache: Path, adversarial_cache: Path,
) -> list[str]:
    """Require two approved reviews bound to every final knowledge byte.

    The review-only fields are excluded by ``knowledge_review_surface.py``;
    all rules, citations, coverage, and binding semantics remain in the hashed
    pack surface.  Python implementation bytes are independently hashed by the
    same review helper.  Prompt and result *file bytes* are then bound from
    every pack's review metadata so a post-review edit fails closed.
    """

    if _formal_host_is_windows():
        return ["formal knowledge-review release audit is Linux/WSL2-only; Windows is NO-GO"]

    issues: list[str] = []
    root = root.resolve()
    try:
        # ``absolute()`` normalizes the spelling without dereferencing the
        # final component.  Dereferencing here would hide a review-result
        # symlink before ``_strict_json_object`` can reject it.
        semantic_path = semantic_review.absolute()
        adversarial_path = adversarial_review.absolute()
    except OSError as exc:
        return [f"cannot resolve knowledge-review result paths: {exc}"]
    if semantic_path != (root / SEMANTIC_REVIEW_PATH).absolute():
        issues.append("semantic review is not the fixed public release artifact")
    if adversarial_path != (root / ADVERSARIAL_REVIEW_PATH).absolute():
        issues.append("adversarial review is not the fixed public release artifact")
    raw_roots: list[tuple[str, Path]] = []
    for label, raw_path in (
        ("semantic", semantic_raw.absolute()),
        ("adversarial", adversarial_raw.absolute()),
    ):
        try:
            raw_path.relative_to(root)
        except ValueError:
            pass
        else:
            issues.append(
                f"raw {label} Codex review stream must be a restricted external artifact"
            )
        try:
            raw_roots.append((
                f"raw {label} evidence", raw_path.parent.resolve(strict=True),
            ))
        except OSError:
            issues.append(f"raw {label} evidence parent is unavailable")
    cache_roots: list[tuple[str, Path]] = []
    for label, cache_path in (
        ("semantic", semantic_cache.absolute()),
        ("adversarial", adversarial_cache.absolute()),
    ):
        try:
            cache_path.relative_to(root)
        except ValueError:
            pass
        else:
            issues.append(f"{label} review cache must be an external artifact")
        try:
            cache_roots.append((
                f"{label} review cache", cache_path.resolve(strict=True),
            ))
        except OSError:
            issues.append(f"{label} review cache is unavailable")
    protected_roots: list[tuple[str, Path]] = [
        ("review checkout", root), *raw_roots, *cache_roots,
    ]
    if os.name != "nt":
        codex_home_text = os.environ.get("CODEX_HOME", "")
        if not codex_home_text:
            issues.append("formal release audit requires the dedicated CODEX_HOME")
        else:
            try:
                protected_roots.append((
                    "dedicated CODEX_HOME", Path(codex_home_text).resolve(strict=True),
                ))
            except OSError:
                issues.append("dedicated CODEX_HOME is unavailable")
    for index, (left_label, left_path) in enumerate(protected_roots):
        for right_label, right_path in protected_roots[index + 1:]:
            if (left_path == right_path
                    or left_path.is_relative_to(right_path)
                    or right_path.is_relative_to(left_path)):
                issues.append(f"{left_label} must be disjoint from {right_label}")
    try:
        helper_root = Path(_knowledge_review_surface.ROOT).resolve()
    except (AttributeError, OSError) as exc:
        return [f"cannot resolve knowledge-review surface helper: {exc}"]
    if helper_root != root:
        return [
            "knowledge-review surface helper is not bound to the audited source root"
        ]

    try:
        schema, schema_bytes = _strict_json_object(
            root / REVIEW_SCHEMA_PATH, "knowledge-review schema", root=root,
        )
        receipt_schema, receipt_schema_bytes = _strict_json_object(
            root / REVIEW_RECEIPT_SCHEMA_PATH,
            "knowledge-review receipt schema", root=root,
        )
        semantic, semantic_bytes = _strict_json_object(
            semantic_path, "semantic knowledge review", root=root,
        )
        adversarial, adversarial_bytes = _strict_json_object(
            adversarial_path, "adversarial knowledge review", root=root,
        )
        semantic_prompt = _strict_file_bytes(
            root / SEMANTIC_REVIEW_PROMPT_PATH, "semantic knowledge-review prompt",
            root=root,
        )
        adversarial_prompt = _strict_file_bytes(
            root / ADVERSARIAL_REVIEW_PROMPT_PATH,
            "adversarial knowledge-review prompt", root=root,
        )
        surface_helper_bytes = _strict_file_bytes(
            root / "tools/knowledge_review_surface.py",
            "knowledge-review surface helper", root=root,
        )
        citation_generator_bytes = _strict_file_bytes(
            root / "tools/audit_knowledge_citations.py",
            "knowledge citation-audit generator", root=root,
        )
        review_runner_bytes = _strict_file_bytes(
            root / "tools/run_knowledge_review.py",
            "knowledge-review restricted runner", root=root,
        )
        semantic_receipt, semantic_receipt_bytes = _strict_json_object(
            root / SEMANTIC_REVIEW_RECEIPT_PATH,
            "semantic knowledge-review CLI receipt", root=root,
        )
        adversarial_receipt, adversarial_receipt_bytes = _strict_json_object(
            root / ADVERSARIAL_REVIEW_RECEIPT_PATH,
            "adversarial knowledge-review CLI receipt", root=root,
        )
        semantic_raw_bytes = _strict_file_bytes(
            semantic_raw, "raw semantic Codex review stream",
            required_file_mode=0o600, required_parent_mode=0o700,
            require_current_owner=True, require_single_link=True,
            max_bytes=REVIEW_MAX_RAW_BYTES,
        )
        adversarial_raw_bytes = _strict_file_bytes(
            adversarial_raw, "raw adversarial Codex review stream",
            required_file_mode=0o600, required_parent_mode=0o700,
            require_current_owner=True, require_single_link=True,
            max_bytes=REVIEW_MAX_RAW_BYTES,
        )
    except (OSError, ValueError) as exc:
        return [str(exc)]

    semantic_snapshot = adversarial_snapshot = None
    try:
        semantic_snapshot = _knowledge_review_runner.freeze_review_snapshot(
            root, SEMANTIC_REVIEW_PROTOCOL,
        )
        adversarial_snapshot = _knowledge_review_runner.freeze_review_snapshot(
            root, ADVERSARIAL_REVIEW_PROTOCOL,
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        issues.append(f"cannot freeze current knowledge-review inputs: {exc}")
    semantic_cache_value = adversarial_cache_value = None
    semantic_replay = adversarial_replay = None
    if semantic_snapshot is not None:
        try:
            semantic_cache_value = _knowledge_review_runner.load_review_cache(
                semantic_cache, semantic_snapshot,
            )
            semantic_replay = _knowledge_review_runner.replay_raw_review(
                root, SEMANTIC_REVIEW_PROTOCOL, semantic_raw_bytes,
                snapshot=semantic_snapshot, cache=semantic_cache_value,
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            issues.append(f"cannot replay raw semantic knowledge-review stream: {exc}")
    if adversarial_snapshot is not None:
        try:
            adversarial_cache_value = _knowledge_review_runner.load_review_cache(
                adversarial_cache, adversarial_snapshot,
            )
            adversarial_replay = _knowledge_review_runner.replay_raw_review(
                root, ADVERSARIAL_REVIEW_PROTOCOL, adversarial_raw_bytes,
                snapshot=adversarial_snapshot, cache=adversarial_cache_value,
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            issues.append(f"cannot replay raw adversarial knowledge-review stream: {exc}")
    if semantic_replay is not None and semantic_replay.result_bytes != semantic_bytes:
        issues.append("semantic review result was not derived from its raw Codex stream")
    if (adversarial_replay is not None
            and adversarial_replay.result_bytes != adversarial_bytes):
        issues.append("adversarial review result was not derived from its raw Codex stream")

    required_result_fields = {
        "protocol_id", "review_surface_sha256",
        "implementation_surface_sha256", "citation_audit_sha256",
        "citation_results", "approved", "issues", "summary",
    }
    if (
        schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema"
        or schema.get("type") != "object"
        or schema.get("additionalProperties") is not False
        or set(schema.get("required", [])) != required_result_fields
        or set(schema.get("properties", {})) != required_result_fields
    ):
        issues.append("knowledge-review schema is not the closed v0.3 review contract")
    issues.extend(_receipt_schema_contract_issues(receipt_schema))
    schema_sha256 = hashlib.sha256(schema_bytes).hexdigest()
    semantic_prompt_sha256 = hashlib.sha256(semantic_prompt).hexdigest()
    adversarial_prompt_sha256 = hashlib.sha256(adversarial_prompt).hexdigest()
    if semantic_receipt.get("invocation_id") == adversarial_receipt.get("invocation_id"):
        issues.append("semantic and adversarial reviews reuse an invocation ID")
    if semantic_receipt.get("thread_id") == adversarial_receipt.get("thread_id"):
        issues.append("semantic and adversarial reviews reuse a Codex thread ID")
    if semantic_receipt.get("event_stream_sha256") == adversarial_receipt.get(
        "event_stream_sha256"
    ):
        issues.append("semantic and adversarial reviews reuse one CLI event stream")

    pack_root = root / "src" / "hlsgraph" / "knowledge" / "packs"
    pack_rows: list[tuple[Path, dict[str, Any]]] = []
    pack_ids: set[str] = set()
    for path in sorted(pack_root.glob("*.json")):
        try:
            value, _data = _strict_json_object(
                path, f"knowledge pack {path.name}", root=root,
            )
        except (OSError, ValueError) as exc:
            issues.append(str(exc))
            continue
        pack_id = value.get("pack_id")
        if not isinstance(pack_id, str) or not pack_id:
            issues.append(f"knowledge pack {path.name} lacks a pack_id")
            continue
        if pack_id in pack_ids:
            issues.append(f"duplicate reviewed knowledge pack ID: {pack_id}")
            continue
        pack_ids.add(pack_id)
        pack_rows.append((path, value))
    if not pack_rows:
        issues.append("no public knowledge packs were found for final review")
    issues.extend(_runtime_pack_contract_issues(pack_rows))

    try:
        expected_surfaces = {
            value["pack_id"]: _knowledge_review_surface.surface_sha256(path)
            for path, value in pack_rows
        }
        expected_implementation = (
            _knowledge_review_surface.implementation_surface_sha256()
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        issues.append(f"cannot recompute knowledge-review surfaces: {exc}")
        return issues
    if (semantic_snapshot is not None and adversarial_snapshot is not None
            and (semantic_snapshot.surfaces != expected_surfaces
                 or adversarial_snapshot.surfaces != expected_surfaces)):
        issues.append("retained review snapshots do not bind the current pack surfaces")
    if (semantic_snapshot is not None and adversarial_snapshot is not None
            and (semantic_snapshot.implementation_surface_sha256
                 != expected_implementation
                 or adversarial_snapshot.implementation_surface_sha256
                 != expected_implementation)):
        issues.append("retained review snapshots do not bind the current implementation")
    citation_issues, citation_bytes, expected_citations = (
        _audit_citation_release_artifact(
            root, expected_surfaces=expected_surfaces,
        )
    )
    issues.extend(citation_issues)
    citation_audit_sha256 = hashlib.sha256(citation_bytes).hexdigest()
    evidence_issues, evidence_bytes, evidence_mappings = (
        _audit_citation_evidence_mapping(
            root, citation_bytes=citation_bytes,
            expected_citations=expected_citations,
        )
    )
    issues.extend(evidence_issues)
    citation_evidence_sha256 = hashlib.sha256(evidence_bytes).hexdigest()
    if semantic_snapshot is None or adversarial_snapshot is None:
        return list(dict.fromkeys(issues))
    if (semantic_snapshot.citation_evidence_sha256
            != citation_evidence_sha256
            or adversarial_snapshot.citation_evidence_sha256
            != citation_evidence_sha256):
        issues.append("retained review snapshots do not bind the evidence mapping")
    if semantic_cache_value is not None:
        issues.extend(_audit_cache_evidence_projection(
            semantic_cache_value.manifest, evidence_mappings,
            label="semantic knowledge-review",
        ))
    if adversarial_cache_value is not None:
        issues.extend(_audit_cache_evidence_projection(
            adversarial_cache_value.manifest, evidence_mappings,
            label="adversarial knowledge-review",
        ))
    semantic_trace_bytes: bytes | None = None
    if semantic_cache_value is not None:
        semantic_trace_issues, semantic_trace_bytes = _audit_review_tool_trace(
            root, trace_path=SEMANTIC_REVIEW_TRACE_PATH,
            prompt_path=SEMANTIC_REVIEW_PROMPT_PATH,
            result_bytes=semantic_bytes, expected_citations=expected_citations,
            snapshot=semantic_snapshot, cache=semantic_cache_value,
        )
        issues.extend(semantic_trace_issues)
        if (semantic_replay is not None
                and semantic_replay.trace_bytes != semantic_trace_bytes):
            issues.append(
                "semantic normalized trace was not replayed from raw Codex JSONL"
            )
    adversarial_trace_bytes: bytes | None = None
    if adversarial_cache_value is not None:
        adversarial_trace_issues, adversarial_trace_bytes = _audit_review_tool_trace(
            root, trace_path=ADVERSARIAL_REVIEW_TRACE_PATH,
            prompt_path=ADVERSARIAL_REVIEW_PROMPT_PATH,
            result_bytes=adversarial_bytes, expected_citations=expected_citations,
            snapshot=adversarial_snapshot, cache=adversarial_cache_value,
        )
        issues.extend(adversarial_trace_issues)
        if (adversarial_replay is not None
                and adversarial_replay.trace_bytes != adversarial_trace_bytes):
            issues.append(
                "adversarial normalized trace was not replayed from raw Codex JSONL"
            )
    if (semantic_cache_value is not None and semantic_replay is not None
            and semantic_trace_bytes is not None):
        semantic_effective_prompt = _knowledge_review_runner.build_review_prompt(
            root, SEMANTIC_REVIEW_PROTOCOL, snapshot=semantic_snapshot,
            cache=semantic_cache_value,
        )
        issues.extend(_review_receipt_contract_issues(
            semantic_receipt, label="semantic knowledge-review CLI receipt",
            expected_protocol=SEMANTIC_REVIEW_PROTOCOL,
            expected_prompt_sha256=hashlib.sha256(
                semantic_effective_prompt
            ).hexdigest(),
            expected_schema_sha256=schema_sha256,
            expected_result_sha256=hashlib.sha256(semantic_bytes).hexdigest(),
            expected_event_stream_path=SEMANTIC_REVIEW_TRACE_PATH,
            expected_event_stream_sha256=hashlib.sha256(
                semantic_trace_bytes
            ).hexdigest(),
            expected_raw_event_stream_sha256=semantic_replay.raw_sha256,
            expected_invocation_id=semantic_replay.invocation_id,
            expected_thread_id=semantic_replay.thread_id,
            expected_command_sha256=(
                _knowledge_review_runner.command_contract_sha256(
                    SEMANTIC_REVIEW_PROTOCOL
                )
            ),
            expected_snapshot_sha256=semantic_snapshot.sha256,
            expected_citation_evidence_sha256=citation_evidence_sha256,
            expected_cache_manifest_sha256=semantic_cache_value.sha256,
            expected_chunk_contract_sha256=(
                semantic_cache_value.manifest["chunk_contract"]["sha256"]
            ),
            expected_model_inspection_contract_sha256=(
                semantic_cache_value.manifest["inspection_contract"]["sha256"]
            ),
            expected_parser_contract_sha256s=(
                semantic_cache_value.manifest["parser_contract_sha256s"]
            ),
            expected_prompt_bytes=len(semantic_effective_prompt),
        ))
    if (adversarial_cache_value is not None and adversarial_replay is not None
            and adversarial_trace_bytes is not None):
        adversarial_effective_prompt = _knowledge_review_runner.build_review_prompt(
            root, ADVERSARIAL_REVIEW_PROTOCOL, snapshot=adversarial_snapshot,
            cache=adversarial_cache_value,
        )
        issues.extend(_review_receipt_contract_issues(
            adversarial_receipt, label="adversarial knowledge-review CLI receipt",
            expected_protocol=ADVERSARIAL_REVIEW_PROTOCOL,
            expected_prompt_sha256=hashlib.sha256(
                adversarial_effective_prompt
            ).hexdigest(),
            expected_schema_sha256=schema_sha256,
            expected_result_sha256=hashlib.sha256(adversarial_bytes).hexdigest(),
            expected_event_stream_path=ADVERSARIAL_REVIEW_TRACE_PATH,
            expected_event_stream_sha256=hashlib.sha256(
                adversarial_trace_bytes
            ).hexdigest(),
            expected_raw_event_stream_sha256=adversarial_replay.raw_sha256,
            expected_invocation_id=adversarial_replay.invocation_id,
            expected_thread_id=adversarial_replay.thread_id,
            expected_command_sha256=(
                _knowledge_review_runner.command_contract_sha256(
                    ADVERSARIAL_REVIEW_PROTOCOL
                )
            ),
            expected_snapshot_sha256=adversarial_snapshot.sha256,
            expected_citation_evidence_sha256=citation_evidence_sha256,
            expected_cache_manifest_sha256=adversarial_cache_value.sha256,
            expected_chunk_contract_sha256=(
                adversarial_cache_value.manifest["chunk_contract"]["sha256"]
            ),
            expected_model_inspection_contract_sha256=(
                adversarial_cache_value.manifest["inspection_contract"]["sha256"]
            ),
            expected_parser_contract_sha256s=(
                adversarial_cache_value.manifest["parser_contract_sha256s"]
            ),
            expected_prompt_bytes=len(adversarial_effective_prompt),
        ))

    semantic_boundary = semantic_receipt.get("boundary_contract")
    adversarial_boundary = adversarial_receipt.get("boundary_contract")
    semantic_runtime = (
        semantic_boundary.get("runtime_manifest")
        if isinstance(semantic_boundary, dict) else None
    )
    adversarial_runtime = (
        adversarial_boundary.get("runtime_manifest")
        if isinstance(adversarial_boundary, dict) else None
    )
    if (not isinstance(semantic_runtime, dict)
            or semantic_runtime != adversarial_runtime):
        issues.append(
            "semantic and adversarial reviews did not use one identical Codex runtime"
        )

    schema_surface = (
        schema.get("properties", {})
        .get("review_surface_sha256", {})
        .get("required", [])
    )
    if set(schema_surface) != set(expected_surfaces):
        issues.append(
            "knowledge-review schema pack inventory differs from the current packs"
        )
    schema_properties = schema.get("properties", {})
    surface_contract = schema_properties.get("review_surface_sha256", {})
    surface_properties = surface_contract.get("properties", {})
    if (
        surface_contract.get("type") != "object"
        or surface_contract.get("additionalProperties") is not False
        or set(surface_properties) != set(expected_surfaces)
        or any(
            not isinstance(surface_properties.get(pack_id), dict)
            or surface_properties[pack_id].get("type") != "string"
            or surface_properties[pack_id].get("pattern") != "^[0-9a-f]{64}$"
            for pack_id in expected_surfaces
        )
        or schema_properties.get("protocol_id") != {
            "enum": [SEMANTIC_REVIEW_PROTOCOL, ADVERSARIAL_REVIEW_PROTOCOL],
        }
        or schema_properties.get("implementation_surface_sha256") != {
            "type": "string", "pattern": "^[0-9a-f]{64}$",
        }
        or schema_properties.get("citation_audit_sha256") != {
            "type": "string", "pattern": "^[0-9a-f]{64}$",
        }
        or not isinstance(schema_properties.get("citation_results"), dict)
        or schema_properties.get("approved") != {"type": "boolean"}
        or schema_properties.get("summary") != {
            "enum": ["approved_no_issues", "rejected_with_controlled_issues"],
        }
    ):
        issues.append("knowledge-review schema weakens a required result field")
    issue_contract = schema_properties.get("issues")
    expected_issue_contract = {
        "type": "array",
        "items": {
            "type": "object", "additionalProperties": False,
            "required": ["severity", "code"],
            "properties": {
                "severity": {"enum": ["critical", "high", "medium", "low"]},
                "code": {"enum": [
                    "semantic_gap", "activation_bypass", "citation_unavailable",
                    "citation_rejected", "contract_violation",
                ]},
            },
        },
    }
    if issue_contract != expected_issue_contract:
        issues.append("knowledge-review schema permits uncontrolled public issue prose")
    citation_items = schema_properties.get("citation_results", {}).get("items", {})
    citation_issue_contract = citation_items.get("properties", {}).get("issues")
    if citation_issue_contract != {
        "type": "array",
        "items": {"enum": [
            "locator_unavailable", "resolver_mismatch", "version_mismatch",
            "section_mismatch", "paraphrase_unsupported",
            "applicability_too_broad", "inspection_incomplete",
        ]},
    }:
        issues.append("knowledge-review schema permits uncontrolled citation issue prose")
    issues.extend(_review_result_contract_issues(
        semantic, label="semantic knowledge review",
        expected_protocol=SEMANTIC_REVIEW_PROTOCOL,
        expected_surfaces=expected_surfaces,
        expected_implementation=expected_implementation,
        expected_citation_audit_sha256=citation_audit_sha256,
        expected_citations=expected_citations,
    ))
    issues.extend(_review_result_contract_issues(
        adversarial, label="adversarial knowledge review",
        expected_protocol=ADVERSARIAL_REVIEW_PROTOCOL,
        expected_surfaces=expected_surfaces,
        expected_implementation=expected_implementation,
        expected_citation_audit_sha256=citation_audit_sha256,
        expected_citations=expected_citations,
    ))
    if semantic.get("protocol_id") == adversarial.get("protocol_id"):
        issues.append("semantic and adversarial reviews must use different protocols")
    semantic_citations = semantic.get("citation_results")
    adversarial_citations = adversarial.get("citation_results")
    if (not isinstance(semantic_citations, list)
            or not isinstance(adversarial_citations, list)
            or sorted(
                semantic_citations, key=lambda item: str(item.get("reference_id"))
                if isinstance(item, dict) else "",
            ) != sorted(
                adversarial_citations, key=lambda item: str(item.get("reference_id"))
                if isinstance(item, dict) else "",
            )):
        issues.append("semantic and adversarial citation verdicts do not agree exactly")

    try:
        required_source_hashes = _knowledge_review_runner.review_source_hashes(
            root, expected_surfaces, expected_implementation,
        )
    except (OSError, TypeError, ValueError) as exc:
        issues.append(f"cannot compute deterministic review source hashes: {exc}")
        return issues

    expected_evidence = {
        "independent_invocations": True,
        "same_model_repeated_review": True,
        "distinct_model_families": False,
        "citation_verified": True,
        "review_agreement": True,
        "unresolved_conflicts": False,
    }
    expected_invocations = sorted([
        _receipt_invocation_projection(semantic_receipt, semantic_receipt_bytes),
        _receipt_invocation_projection(adversarial_receipt, adversarial_receipt_bytes),
    ], key=lambda item: (str(item["protocol_id"]), str(item["invocation_id"])))
    expected_reviewers = sorted(
        f"{item['model']}@{item['reasoning_effort']}#{item['invocation_id']}"
        for item in expected_invocations
    )
    for _path, value in pack_rows:
        pack_id = value["pack_id"]
        metadata = value.get("metadata")
        coverage = value.get("coverage")
        if not isinstance(metadata, dict) or not isinstance(coverage, dict):
            issues.append(f"knowledge pack {pack_id} lacks review metadata or coverage")
            continue
        if (metadata.get("review_status") != "machine_repeated_reviewed"
                or coverage.get("review_status") != "machine_repeated_reviewed"):
            issues.append(f"knowledge pack {pack_id} is not machine-repeated reviewed")
        reviewers = coverage.get("reviewers")
        if (not isinstance(reviewers, list) or len(reviewers) != 2
                or any(not isinstance(item, str) or not item for item in reviewers)
                or len(set(reviewers)) != 2):
            issues.append(
                f"knowledge pack {pack_id} must name exactly two unique review invocations"
            )
            reviewers = []
        evidence = coverage.get("review_evidence")
        if not isinstance(evidence, dict):
            issues.append(f"knowledge pack {pack_id} lacks review_evidence")
            continue
        for key, expected in expected_evidence.items():
            if evidence.get(key) is not expected:
                issues.append(
                    f"knowledge pack {pack_id} has invalid review evidence {key}"
                )
        invocations = evidence.get(REVIEW_INVOCATIONS_KEY)
        if not isinstance(invocations, list) or len(invocations) != 2:
            issues.append(
                f"knowledge pack {pack_id} must bind exactly two review invocations"
            )
        else:
            if any(not isinstance(item, dict) for item in invocations):
                issues.append(
                    f"knowledge pack {pack_id} has malformed review invocation evidence"
                )
            else:
                normalized_invocations = sorted(
                    invocations,
                    key=lambda item: (
                        str(item.get("protocol_id")), str(item.get("invocation_id")),
                    ),
                )
                if normalized_invocations != expected_invocations:
                    issues.append(
                        f"knowledge pack {pack_id} invocation evidence does not "
                        "exactly match the two CLI receipt envelopes"
                    )
        if reviewers and sorted(reviewers) != expected_reviewers:
            issues.append(
                f"knowledge pack {pack_id} reviewers do not match CLI receipt envelopes"
            )
        source_hashes = coverage.get("source_hashes")
        if not isinstance(source_hashes, dict):
            issues.append(f"knowledge pack {pack_id} lacks review source hashes")
        else:
            if any(
                not isinstance(key, str) or not key
                or not isinstance(digest, str)
                or _SHA256_RE.fullmatch(digest) is None
                for key, digest in source_hashes.items()
            ):
                issues.append(f"knowledge pack {pack_id} has malformed source hashes")
            if source_hashes != required_source_hashes:
                missing = sorted(set(required_source_hashes) - set(source_hashes))
                extra = sorted(set(source_hashes) - set(required_source_hashes))
                changed = sorted(
                    key for key in set(source_hashes) & set(required_source_hashes)
                    if source_hashes.get(key) != required_source_hashes[key]
                )
                issues.append(
                    f"knowledge pack {pack_id} source hashes differ from the exact "
                    f"review closure: missing={missing!r}, extra={extra!r}, "
                    f"changed={changed!r}"
                )
    return list(dict.fromkeys(issues))


class KnowledgeReviewSuiteReplayAudit:
    """Fail-closed result of replaying the complete private v6 evidence tree.

    Validated products are exposed only when ``issues`` is empty.  This lets
    the attestation finalizer consume exactly the same raw/cache/process
    replay as the release gate without allow-listing audit failures or
    trusting already-published pack metadata.
    """

    __slots__ = (
        "issues", "semantic_snapshot", "adversarial_snapshot",
        "semantic_receipt", "adversarial_receipt", "semantic_result",
        "adversarial_result", "pair_seal", "plan", "citation_audit",
        "attestation_material", "pack_rows",
    )

    def __init__(
        self, issues: Iterable[str], *, semantic_snapshot: Any | None = None,
        adversarial_snapshot: Any | None = None,
        semantic_receipt: dict[str, Any] | None = None,
        adversarial_receipt: dict[str, Any] | None = None,
        semantic_result: dict[str, Any] | None = None,
        adversarial_result: dict[str, Any] | None = None,
        pair_seal: dict[str, Any] | None = None,
        plan: dict[str, Any] | None = None,
        citation_audit: dict[str, Any] | None = None,
        attestation_material: dict[str, Any] | None = None,
        pack_rows: tuple[tuple[Path, dict[str, Any]], ...] = (),
    ) -> None:
        self.issues = tuple(issues)
        self.semantic_snapshot = semantic_snapshot
        self.adversarial_snapshot = adversarial_snapshot
        self.semantic_receipt = semantic_receipt
        self.adversarial_receipt = adversarial_receipt
        self.semantic_result = semantic_result
        self.adversarial_result = adversarial_result
        self.pair_seal = pair_seal
        self.plan = plan
        self.citation_audit = citation_audit
        self.attestation_material = attestation_material
        self.pack_rows = tuple(pack_rows)

    def __repr__(self) -> str:
        return (
            "KnowledgeReviewSuiteReplayAudit("
            f"issues={self.issues!r}, verified={self.verified!r})"
        )

    @property
    def verified(self) -> bool:
        return not self.issues and self.attestation_material is not None


def _replay_knowledge_review_suite_evidence_impl(
    root: Path, *, semantic_review: Path, adversarial_review: Path,
    suite_evidence: Path, capture: dict[str, Any],
) -> list[str]:
    """Replay and deterministically rebuild the closed v6 three-shard suite."""

    issues: list[str] = []
    root = root.resolve()
    semantic_path = semantic_review.absolute()
    adversarial_path = adversarial_review.absolute()
    if semantic_path != (root / SEMANTIC_REVIEW_PATH).absolute():
        issues.append("semantic review is not the fixed public release artifact")
    if adversarial_path != (root / ADVERSARIAL_REVIEW_PATH).absolute():
        issues.append("adversarial review is not the fixed public release artifact")
    evidence_root = suite_evidence.absolute()
    try:
        evidence_root.relative_to(root)
    except ValueError:
        pass
    else:
        issues.append("knowledge-review suite evidence must be external")
    try:
        resolved_evidence = evidence_root.resolve(strict=True)
    except OSError:
        return [*issues, "knowledge-review suite evidence root is unavailable"]
    protected_roots: list[tuple[str, Path]] = [
        ("review checkout", root),
        ("knowledge-review suite evidence", resolved_evidence),
    ]
    codex_home_text = os.environ.get("CODEX_HOME", "")
    if not codex_home_text:
        issues.append("formal release audit requires the dedicated CODEX_HOME")
    else:
        try:
            protected_roots.append((
                "dedicated CODEX_HOME", Path(codex_home_text).resolve(strict=True),
            ))
        except OSError:
            issues.append("dedicated CODEX_HOME is unavailable")
    for index, (left_label, left_path) in enumerate(protected_roots):
        for right_label, right_path in protected_roots[index + 1:]:
            if (left_path == right_path
                    or left_path.is_relative_to(right_path)
                    or right_path.is_relative_to(left_path)):
                issues.append(f"{left_label} must be disjoint from {right_label}")
    try:
        helper_root = Path(_knowledge_review_surface.ROOT).resolve()
    except (AttributeError, OSError) as exc:
        return [f"cannot resolve knowledge-review surface helper: {exc}"]
    if helper_root != root:
        return [
            "knowledge-review surface helper is not bound to the audited source root"
        ]

    try:
        schema, schema_bytes = _strict_json_object(
            root / REVIEW_SCHEMA_PATH, "knowledge-review schema", root=root,
        )
        shard_schema, shard_schema_bytes = _strict_json_object(
            root / REVIEW_SHARD_SCHEMA_PATH,
            "knowledge-review shard schema", root=root,
        )
        suite_evidence_schema, _suite_evidence_schema_bytes = _strict_json_object(
            root / REVIEW_SUITE_EVIDENCE_SCHEMA_PATH,
            "knowledge-review suite evidence schema", root=root,
        )
        suite_receipt_schema, suite_receipt_schema_bytes = _strict_json_object(
            root / REVIEW_SUITE_RECEIPT_SCHEMA_PATH,
            "knowledge-review suite receipt schema", root=root,
        )
        suite_trace_schema, suite_trace_schema_bytes = _strict_json_object(
            root / REVIEW_SUITE_TRACE_SCHEMA_PATH,
            "knowledge-review suite trace schema", root=root,
        )
        semantic, semantic_bytes = _strict_json_object(
            semantic_path, "semantic knowledge review", root=root,
        )
        adversarial, adversarial_bytes = _strict_json_object(
            adversarial_path, "adversarial knowledge review", root=root,
        )
        semantic_receipt, semantic_receipt_bytes = _strict_json_object(
            root / SEMANTIC_REVIEW_RECEIPT_PATH,
            "semantic knowledge-review suite receipt", root=root,
        )
        adversarial_receipt, adversarial_receipt_bytes = _strict_json_object(
            root / ADVERSARIAL_REVIEW_RECEIPT_PATH,
            "adversarial knowledge-review suite receipt", root=root,
        )
        semantic_trace_bytes = _strict_file_bytes(
            root / SEMANTIC_REVIEW_TRACE_PATH,
            "semantic knowledge-review suite trace", root=root,
        )
        adversarial_trace_bytes = _strict_file_bytes(
            root / ADVERSARIAL_REVIEW_TRACE_PATH,
            "adversarial knowledge-review suite trace", root=root,
        )
        semantic_shard_prompt = _strict_file_bytes(
            root / SEMANTIC_SHARD_PROMPT_PATH,
            "semantic knowledge-review shard prompt", root=root,
        )
        adversarial_shard_prompt = _strict_file_bytes(
            root / ADVERSARIAL_SHARD_PROMPT_PATH,
            "adversarial knowledge-review shard prompt", root=root,
        )
        suite_manifest, _suite_manifest_bytes, suite_protocols = (
            _load_suite_evidence_manifest(resolved_evidence)
        )
        pair_seal_bytes = _strict_file_bytes(
            resolved_evidence / _knowledge_review_suite_executor.PAIR_SEAL_PATH,
            "knowledge-review suite pair seal", root=resolved_evidence,
            required_file_mode=0o600, required_parent_mode=0o700,
            require_current_owner=True, require_single_link=True,
            max_bytes=1024 * 1024,
        )
        stored_pair_seal = _knowledge_review_runner._strict_json_bytes(
            pair_seal_bytes, label="knowledge-review suite pair seal",
        )
        if (not isinstance(stored_pair_seal, dict)
                or _knowledge_review_runner._canonical_json(stored_pair_seal)
                != pair_seal_bytes):
            raise ValueError("knowledge-review suite pair seal is not canonical")
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return [*issues, str(exc)]

    required_result_fields = {
        "protocol_id", "review_surface_sha256",
        "implementation_surface_sha256", "citation_audit_sha256",
        "citation_results", "approved", "issues", "summary",
    }
    if (
        schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema"
        or schema.get("type") != "object"
        or schema.get("additionalProperties") is not False
        or set(schema.get("required", [])) != required_result_fields
        or set(schema.get("properties", {})) != required_result_fields
    ):
        issues.append("knowledge-review schema is not the closed v0.3 review contract")
    if (suite_evidence_schema.get("type") != "object"
            or suite_evidence_schema.get("additionalProperties") is not False
            or set(suite_evidence_schema.get("required", []))
            != _SUITE_EVIDENCE_FIELDS
            or set(suite_evidence_schema.get("properties", {}))
            != _SUITE_EVIDENCE_FIELDS
            or suite_evidence_schema.get("properties", {}).get(
                "schema_version"
            ) != {"const": REVIEW_SUITE_EVIDENCE_SCHEMA_VERSION}):
        issues.append("knowledge-review suite evidence schema is not closed")
    if (shard_schema.get("type") != "object"
            or shard_schema.get("additionalProperties") is not False):
        issues.append("knowledge-review shard schema is not closed")
    if (suite_receipt_schema.get("type") != "object"
            or suite_receipt_schema.get("additionalProperties") is not False
            or suite_receipt_schema.get("properties", {}).get(
                "schema_version"
            ) != {"const": REVIEW_SUITE_RECEIPT_SCHEMA_VERSION}):
        issues.append("knowledge-review suite receipt schema is not closed")
    if (suite_trace_schema.get("$schema")
            != "https://json-schema.org/draft/2020-12/schema"):
        issues.append("knowledge-review suite trace schema is not the pinned contract")

    pack_root = root / "src" / "hlsgraph" / "knowledge" / "packs"
    pack_rows: list[tuple[Path, dict[str, Any]]] = []
    pack_ids: set[str] = set()
    for path in sorted(pack_root.glob("*.json")):
        try:
            value, _data = _strict_json_object(
                path, f"knowledge pack {path.name}", root=root,
            )
        except (OSError, ValueError) as exc:
            issues.append(str(exc))
            continue
        pack_id = value.get("pack_id")
        if not isinstance(pack_id, str) or not pack_id:
            issues.append(f"knowledge pack {path.name} lacks a pack_id")
            continue
        if pack_id in pack_ids:
            issues.append(f"duplicate reviewed knowledge pack ID: {pack_id}")
            continue
        pack_ids.add(pack_id)
        pack_rows.append((path, value))
    if not pack_rows:
        return [*issues, "no public knowledge packs were found for final review"]
    try:
        expected_surfaces = {
            value["pack_id"]: _knowledge_review_surface.surface_sha256(path)
            for path, value in pack_rows
        }
        expected_implementation = (
            _knowledge_review_surface.implementation_surface_sha256()
        )
        citation_issues, citation_bytes, expected_citations = (
            _audit_citation_release_artifact(
                root, expected_surfaces=expected_surfaces,
            )
        )
        citation_audit = _knowledge_review_runner._strict_json_bytes(
            citation_bytes, label="knowledge citation audit",
        )
        if not isinstance(citation_audit, dict):
            raise ValueError("knowledge citation audit is not an object")
        plan = _knowledge_review_shards.build_shard_plan(citation_audit)
        plan_sha256 = _knowledge_review_shards.shard_plan_sha256(plan)
        snapshots = {
            protocol_id: _knowledge_review_runner.freeze_review_snapshot(
                root, protocol_id,
            )
            for protocol_id in (
                SEMANTIC_REVIEW_PROTOCOL, ADVERSARIAL_REVIEW_PROTOCOL,
            )
        }
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return [*issues, f"cannot freeze v6 knowledge-review inputs: {exc}"]
    issues.extend(citation_issues)
    citation_audit_sha256 = hashlib.sha256(citation_bytes).hexdigest()
    evidence_issues, evidence_bytes, evidence_mappings = (
        _audit_citation_evidence_mapping(
            root, citation_bytes=citation_bytes,
            expected_citations=expected_citations,
        )
    )
    issues.extend(evidence_issues)
    citation_evidence_sha256 = hashlib.sha256(evidence_bytes).hexdigest()
    for protocol_id, snapshot in snapshots.items():
        if (snapshot.surfaces != expected_surfaces
                or snapshot.implementation_surface_sha256
                != expected_implementation
                or snapshot.citation_evidence_sha256
                != citation_evidence_sha256):
            issues.append(
                f"{_suite_protocol_label(protocol_id)} v6 snapshot is stale"
            )

    try:
        tokenizer = _knowledge_review_shards.load_verified_tokenizer()
    except (ImportError, TypeError, ValueError) as exc:
        return [*issues, f"cannot load pinned knowledge-review tokenizer: {exc}"]
    runtime = suite_manifest["runtime_manifest"]
    runtime_sha256 = str(runtime["sha256"])
    protocol_products: dict[str, dict[str, Any]] = {}
    loaded_full_caches: dict[str, Any] = {}
    full_surface_hashes: set[str] = set()
    for protocol_id in (
        SEMANTIC_REVIEW_PROTOCOL, ADVERSARIAL_REVIEW_PROTOCOL,
    ):
        label = _suite_protocol_label(protocol_id)
        snapshot = snapshots[protocol_id]
        protocol_manifest = suite_protocols[protocol_id]
        if protocol_manifest.get("review_snapshot_sha256") != snapshot.sha256:
            issues.append(f"{label} suite evidence uses a stale snapshot")
        try:
            full_cache_path = _suite_evidence_paths(
                resolved_evidence, protocol_id,
            )["cache"]
            _strict_path_components(
                full_cache_path, f"{label} full review cache",
                root=resolved_evidence,
            )
            full_cache = _knowledge_review_runner.load_review_cache(
                full_cache_path,
                snapshot,
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            issues.append(f"cannot load {label} full review cache: {exc}")
            continue
        if full_cache.sha256 != protocol_manifest.get(
            "full_cache_manifest_sha256"
        ):
            issues.append(f"{label} full review cache hash differs from manifest")
        loaded_full_caches[protocol_id] = full_cache
        issues.extend(_audit_cache_evidence_projection(
            full_cache.manifest, evidence_mappings,
            label=f"{label} v6 knowledge-review",
        ))
        full_surface = _knowledge_review_suite.citation_evidence_surface_sha256(
            full_cache.manifest["citations"],
        )
        full_surface_hashes.add(full_surface)
        if full_surface != protocol_manifest.get(
            "full_citation_evidence_surface_sha256"
        ):
            issues.append(f"{label} full citation evidence surface is stale")
        base_prompt = (
            semantic_shard_prompt
            if protocol_id == SEMANTIC_REVIEW_PROTOCOL
            else adversarial_shard_prompt
        )
        shard_products: list[dict[str, Any]] = []
        shard_rows = protocol_manifest["shards"]
        for shard_id, shard_evidence in zip(
            _knowledge_review_shards.SHARD_ORDER, shard_rows, strict=True,
        ):
            paths = _suite_evidence_paths(
                resolved_evidence, protocol_id, shard_id,
            )
            try:
                _strict_path_components(
                    paths["cache"], f"{label} {shard_id} projected cache",
                    root=resolved_evidence,
                )
                projection = _knowledge_review_suite.project_shard_manifest(
                    protocol_id=protocol_id,
                    snapshot_inventory=snapshot.inventory(),
                    cache_manifest=full_cache.manifest,
                    plan=plan, shard_id=shard_id,
                )
                projected_cache = (
                    _knowledge_review_suite_cache.validate_shard_cache(
                        paths["cache"], full_cache, projection,
                    )
                )
                if projected_cache.sha256 != shard_evidence.get(
                    "projected_cache_manifest_sha256"
                ):
                    raise ValueError("projected cache hash differs from manifest")
                raw_bytes = _strict_file_bytes(
                    paths["raw"], f"{label} {shard_id} raw Codex stream",
                    root=resolved_evidence, required_file_mode=0o600,
                    required_parent_mode=0o700, require_current_owner=True,
                    require_single_link=True, max_bytes=REVIEW_MAX_RAW_BYTES,
                )
                if hashlib.sha256(raw_bytes).hexdigest() != shard_evidence.get(
                    "raw_output_sha256"
                ):
                    raise ValueError("raw output hash differs from manifest")
                sanitized_bytes = _strict_file_bytes(
                    paths["sanitized"],
                    f"{label} {shard_id} sanitized Codex stream",
                    root=resolved_evidence, required_file_mode=0o600,
                    required_parent_mode=0o700, require_current_owner=True,
                    require_single_link=True, max_bytes=REVIEW_MAX_RAW_BYTES,
                )
                derived_sanitized = (
                    _knowledge_review_suite_executor.sanitize_shard_raw_stream(
                        raw_bytes, projected_cache,
                    )
                )
                sanitized_sha256 = hashlib.sha256(
                    sanitized_bytes
                ).hexdigest()
                if (sanitized_bytes != derived_sanitized
                        or sanitized_sha256 != shard_evidence.get(
                            "sanitized_output_sha256"
                        )):
                    raise ValueError(
                        "sanitized stream is not the exact raw-stream derivative"
                    )
                raw_stderr_bytes = _strict_file_bytes(
                    paths["raw_stderr"],
                    f"{label} {shard_id} raw Codex stderr",
                    root=resolved_evidence, required_file_mode=0o600,
                    required_parent_mode=0o700, require_current_owner=True,
                    require_single_link=True, max_bytes=REVIEW_MAX_RAW_BYTES,
                )
                stderr_bytes = _strict_file_bytes(
                    paths["stderr"],
                    f"{label} {shard_id} sanitized Codex stderr",
                    root=resolved_evidence, required_file_mode=0o600,
                    required_parent_mode=0o700, require_current_owner=True,
                    require_single_link=True, max_bytes=REVIEW_MAX_RAW_BYTES,
                )
                if stderr_bytes != (
                    _knowledge_review_suite_executor._redact_shard_payloads(
                        raw_stderr_bytes, projected_cache,
                    )
                ):
                    raise ValueError(
                        "sanitized stderr is not the exact raw-stderr derivative"
                    )
                process_bytes = _strict_file_bytes(
                    paths["process"],
                    f"{label} {shard_id} process evidence",
                    root=resolved_evidence, required_file_mode=0o600,
                    required_parent_mode=0o700, require_current_owner=True,
                    require_single_link=True, max_bytes=1024 * 1024,
                )
                if hashlib.sha256(process_bytes).hexdigest() != shard_evidence.get(
                    "process_evidence_sha256"
                ):
                    raise ValueError("process evidence hash differs from manifest")
                process_evidence = _knowledge_review_runner._strict_json_bytes(
                    process_bytes, label=f"{label} {shard_id} process evidence",
                )
                process_fields = {
                    "schema_version", "actual_argv", "cwd", "stdin_sha256",
                    "stdout_sha256", "stderr_sha256", "returncode",
                    "command_contract_sha256",
                }
                if (not isinstance(process_evidence, dict)
                        or set(process_evidence) != process_fields
                        or _knowledge_review_runner._canonical_json(
                            process_evidence
                        ) != process_bytes
                        or process_evidence.get("schema_version")
                        != "hlsgraph.knowledge-review.process-evidence.v1"
                        or process_evidence.get("cwd")
                        != str(projected_cache.root.resolve(strict=True))
                        or process_evidence.get("stdin_sha256")
                        != hashlib.sha256(prompt := (
                            _knowledge_review_suite.build_shard_prompt(
                                base_protocol_text=base_prompt,
                                snapshot_inventory=snapshot.inventory(),
                                plan_sha256=plan_sha256,
                                shard_projection=projection,
                            )
                        )).hexdigest()
                        or process_evidence.get("stdout_sha256")
                        != hashlib.sha256(raw_bytes).hexdigest()
                        or process_evidence.get("stderr_sha256")
                        != hashlib.sha256(raw_stderr_bytes).hexdigest()
                        or process_evidence.get("returncode") != 0):
                    raise ValueError("process evidence does not bind the invocation")
                actual_argv = process_evidence.get("actual_argv")
                if (not isinstance(actual_argv, list) or not actual_argv
                        or any(not isinstance(item, str) or not item
                               for item in actual_argv)):
                    raise ValueError("process evidence has malformed actual argv")
                codex_executable = Path(actual_argv[0])
                command_sha256 = (
                    _knowledge_review_suite_executor.validate_actual_shard_command(
                        actual_argv, root=root,
                        cache_root=projected_cache.root,
                        codex=codex_executable,
                    )
                )
                if (process_evidence.get("command_contract_sha256")
                        != command_sha256
                        or _knowledge_review_runner._freeze_runtime_manifest(
                            codex_executable,
                        ) != runtime):
                    raise ValueError(
                        "actual Codex argv does not bind the frozen runtime"
                    )
                replayed = (
                    _knowledge_review_suite_replay.replay_shard_raw_review(
                        raw_bytes, cache=projected_cache,
                        shard_manifest=projection,
                    )
                )
                material = (
                    _knowledge_review_suite_executor.assigned_chunk_material(
                        projected_cache,
                    )
                )
                _knowledge_review_suite_executor.require_exact_command_inventory(
                    raw_bytes, material.commands,
                )
                budget = (
                    _knowledge_review_suite_executor.enforce_shard_token_budget(
                        prompt=prompt, material=material, tokenizer=tokenizer,
                    )
                )
                envelope = (
                    _knowledge_review_suite_executor.build_invocation_envelope(
                        replayed=replayed, shard_manifest=projection,
                        cache=projected_cache, prompt=prompt,
                        boundary_contract=shard_evidence["boundary_contract"],
                        runtime_manifest_sha256=runtime_sha256,
                        token_budget=budget,
                        assigned_chunk_inventory_sha256=(
                            material.inventory_sha256
                        ),
                        replay_contract_digest=(
                            _knowledge_review_suite_executor.replay_contract_sha256(
                                root,
                            )
                        ),
                        sanitized_output_sha256=sanitized_sha256,
                        command_sha256=command_sha256,
                    )
                )
                invocation_bytes = _strict_file_bytes(
                    paths["invocation"],
                    f"{label} {shard_id} invocation envelope",
                    root=resolved_evidence, required_file_mode=0o600,
                    required_parent_mode=0o700, require_current_owner=True,
                    require_single_link=True, max_bytes=16 * 1024 * 1024,
                )
                invocation_value = _knowledge_review_runner._strict_json_bytes(
                    invocation_bytes,
                    label=f"{label} {shard_id} invocation envelope",
                )
                if (invocation_value != envelope
                        or invocation_bytes
                        != _knowledge_review_runner._canonical_json(envelope)):
                    raise ValueError(
                        "stored invocation envelope differs from deterministic replay"
                    )
            except (
                OSError, RuntimeError, TypeError, ValueError,
                json.JSONDecodeError,
            ) as exc:
                issues.append(f"cannot replay {label} {shard_id} review: {exc}")
                continue
            shard_products.append(envelope)
        if len(shard_products) != 3:
            continue
        try:
            aggregate = _knowledge_review_suite.aggregate_shard_results(
                protocol_id=protocol_id,
                snapshot_inventory=snapshot.inventory(),
                cache_manifest=full_cache.manifest,
                citation_audit=citation_audit,
                plan=plan,
                shard_results=[row["shard_result"] for row in shard_products],
            )
        except (TypeError, ValueError) as exc:
            issues.append(f"cannot aggregate {label} suite: {exc}")
            continue
        public_result = semantic if protocol_id == SEMANTIC_REVIEW_PROTOCOL else adversarial
        public_result_bytes = (
            semantic_bytes if protocol_id == SEMANTIC_REVIEW_PROTOCOL
            else adversarial_bytes
        )
        public_trace = (
            semantic_trace_bytes if protocol_id == SEMANTIC_REVIEW_PROTOCOL
            else adversarial_trace_bytes
        )
        public_receipt = (
            semantic_receipt if protocol_id == SEMANTIC_REVIEW_PROTOCOL
            else adversarial_receipt
        )
        public_receipt_bytes = (
            semantic_receipt_bytes if protocol_id == SEMANTIC_REVIEW_PROTOCOL
            else adversarial_receipt_bytes
        )
        if (public_result != aggregate
                or public_result_bytes
                != _knowledge_review_runner._canonical_json(aggregate)):
            issues.append(f"{label} public review differs from suite aggregate")
        if public_receipt_bytes != _knowledge_review_runner._canonical_json(
            public_receipt
        ):
            issues.append(f"{label} suite receipt is not canonical")
        try:
            validated_receipt = (
                _knowledge_review_suite_seal.validate_protocol_receipt(
                    public_receipt, trace_bytes=public_trace,
                    protocol_id=protocol_id, plan=plan,
                    citation_audit=citation_audit,
                    review_snapshot_sha256=snapshot.sha256,
                    citation_evidence_sha256=citation_evidence_sha256,
                    full_evidence_surface_sha256=full_surface,
                    runtime_manifest_sha256=runtime_sha256,
                    invocations=shard_products,
                    aggregate_result=aggregate,
                    output_schema_sha256=hashlib.sha256(schema_bytes).hexdigest(),
                    shard_output_schema_sha256=hashlib.sha256(
                        shard_schema_bytes
                    ).hexdigest(),
                    suite_receipt_schema_sha256=hashlib.sha256(
                        suite_receipt_schema_bytes
                    ).hexdigest(),
                    suite_trace_schema_sha256=hashlib.sha256(
                        suite_trace_schema_bytes
                    ).hexdigest(),
                )
            )
        except (TypeError, ValueError) as exc:
            issues.append(f"cannot validate {label} v6 receipt and trace: {exc}")
            continue
        protocol_products[protocol_id] = {
            "snapshot": snapshot,
            "full_cache": full_cache,
            "invocations": shard_products,
            "aggregate": aggregate,
            "receipt": validated_receipt,
            "receipt_bytes": public_receipt_bytes,
        }
    if set(loaded_full_caches) == {
        SEMANTIC_REVIEW_PROTOCOL, ADVERSARIAL_REVIEW_PROTOCOL,
    }:
        try:
            _validate_full_cache_offline_replay_equivalence(
                loaded_full_caches[SEMANTIC_REVIEW_PROTOCOL],
                loaded_full_caches[ADVERSARIAL_REVIEW_PROTOCOL],
            )
        except (OSError, TypeError, ValueError) as exc:
            issues.append(f"cannot verify offline full-cache replay: {exc}")
    if len(full_surface_hashes) != 1:
        issues.append("semantic and adversarial suites use different evidence surfaces")
    if len(protocol_products) != 2:
        return list(dict.fromkeys(issues))
    semantic_product = protocol_products[SEMANTIC_REVIEW_PROTOCOL]
    adversarial_product = protocol_products[ADVERSARIAL_REVIEW_PROTOCOL]
    try:
        pair_seal = _knowledge_review_suite_seal.validate_suite_pair(
            semantic_receipt=semantic_product["receipt"],
            adversarial_receipt=adversarial_product["receipt"],
            semantic_result=semantic_product["aggregate"],
            adversarial_result=adversarial_product["aggregate"],
            plan=plan, citation_audit=citation_audit,
        )
    except (TypeError, ValueError) as exc:
        issues.append(f"cannot validate six-way knowledge-review suite pair: {exc}")
        return list(dict.fromkeys(issues))
    if stored_pair_seal != pair_seal:
        issues.append("stored knowledge-review suite pair seal differs from replay")
    issues.extend(_review_result_contract_issues(
        semantic_product["aggregate"], label="semantic knowledge review",
        expected_protocol=SEMANTIC_REVIEW_PROTOCOL,
        expected_surfaces=expected_surfaces,
        expected_implementation=expected_implementation,
        expected_citation_audit_sha256=citation_audit_sha256,
        expected_citations=expected_citations,
    ))
    issues.extend(_review_result_contract_issues(
        adversarial_product["aggregate"], label="adversarial knowledge review",
        expected_protocol=ADVERSARIAL_REVIEW_PROTOCOL,
        expected_surfaces=expected_surfaces,
        expected_implementation=expected_implementation,
        expected_citation_audit_sha256=citation_audit_sha256,
        expected_citations=expected_citations,
    ))
    if (semantic_product["aggregate"].get("citation_results")
            != adversarial_product["aggregate"].get("citation_results")):
        issues.append("semantic and adversarial citation verdicts do not agree exactly")

    try:
        from tools import apply_knowledge_review_suite_attestation as suite_attestation
    except (ImportError, ModuleNotFoundError):
        issues.append("knowledge-review suite attestation helper is unavailable")
        return list(dict.fromkeys(issues))
    try:
        material = suite_attestation.build_attestation_material(
            root,
            semantic_snapshot=semantic_product["snapshot"],
            adversarial_snapshot=adversarial_product["snapshot"],
            semantic_receipt=semantic_product["receipt"],
            adversarial_receipt=adversarial_product["receipt"],
            semantic_result=semantic_product["aggregate"],
            adversarial_result=adversarial_product["aggregate"],
            plan=plan, citation_audit=citation_audit,
            suite_pair_seal=pair_seal,
        )
    except (OSError, TypeError, ValueError) as exc:
        issues.append(f"cannot rebuild suite pack attestation material: {exc}")
        return list(dict.fromkeys(issues))
    normalized_issues = list(dict.fromkeys(issues))
    if not normalized_issues:
        capture.update({
            "semantic_snapshot": semantic_product["snapshot"],
            "adversarial_snapshot": adversarial_product["snapshot"],
            "semantic_receipt": semantic_product["receipt"],
            "adversarial_receipt": adversarial_product["receipt"],
            "semantic_result": semantic_product["aggregate"],
            "adversarial_result": adversarial_product["aggregate"],
            "pair_seal": pair_seal,
            "plan": plan,
            "citation_audit": citation_audit,
            "attestation_material": material,
            "pack_rows": tuple(pack_rows),
        })
    return normalized_issues


def replay_knowledge_review_suite_evidence(
    root: Path, *, semantic_review: Path, adversarial_review: Path,
    suite_evidence: Path,
) -> KnowledgeReviewSuiteReplayAudit:
    """Replay v6 evidence without trusting or modifying pack attestations."""

    capture: dict[str, Any] = {}
    issues = _replay_knowledge_review_suite_evidence_impl(
        root, semantic_review=semantic_review,
        adversarial_review=adversarial_review,
        suite_evidence=suite_evidence, capture=capture,
    )
    if issues:
        return KnowledgeReviewSuiteReplayAudit(tuple(issues))
    required = {
        "semantic_snapshot", "adversarial_snapshot", "semantic_receipt",
        "adversarial_receipt", "semantic_result", "adversarial_result",
        "pair_seal", "attestation_material", "pack_rows",
        "plan", "citation_audit",
    }
    if set(capture) != required:  # pragma: no cover - defensive closure
        return KnowledgeReviewSuiteReplayAudit((
            "knowledge-review suite replay did not return its closed products",
        ))
    return KnowledgeReviewSuiteReplayAudit(issues=(), **capture)


def _suite_pack_attestation_issues(
    replay: KnowledgeReviewSuiteReplayAudit,
) -> list[str]:
    """Compare current pack metadata with one independently replayed suite."""

    if not replay.verified or replay.attestation_material is None:
        return list(replay.issues) or [
            "knowledge-review suite replay is incomplete",
        ]
    material = replay.attestation_material
    issues = _runtime_pack_contract_issues(list(replay.pack_rows))
    for _path, pack in replay.pack_rows:
        pack_id = str(pack["pack_id"])
        metadata = pack.get("metadata")
        coverage = pack.get("coverage")
        if not isinstance(metadata, dict) or not isinstance(coverage, dict):
            issues.append(f"knowledge pack {pack_id} lacks review metadata or coverage")
            continue
        if (metadata.get("review_status") != "machine_repeated_reviewed"
                or coverage.get("review_status") != "machine_repeated_reviewed"):
            issues.append(f"knowledge pack {pack_id} is not machine-repeated reviewed")
        if coverage.get("reviewers") != material.get("reviewers"):
            issues.append(f"knowledge pack {pack_id} has stale suite reviewers")
        if coverage.get("source_hashes") != material.get("source_hashes"):
            issues.append(f"knowledge pack {pack_id} has stale suite source hashes")
        if coverage.get("review_evidence") != material.get("review_evidence"):
            issues.append(f"knowledge pack {pack_id} has stale suite review evidence")
    return list(dict.fromkeys(issues))


def _audit_knowledge_review_suite_release_gate(
    root: Path, *, semantic_review: Path, adversarial_review: Path,
    suite_evidence: Path,
) -> list[str]:
    replay = replay_knowledge_review_suite_evidence(
        root, semantic_review=semantic_review,
        adversarial_review=adversarial_review,
        suite_evidence=suite_evidence,
    )
    if replay.issues:
        return list(replay.issues)
    return _suite_pack_attestation_issues(replay)


def _audit_knowledge_review_release_gate(
    root: Path, *, semantic_review: Path, adversarial_review: Path,
    semantic_raw: Path | None = None, adversarial_raw: Path | None = None,
    semantic_cache: Path | None = None, adversarial_cache: Path | None = None,
    suite_evidence: Path | None = None,
) -> list[str]:
    """Audit the only formal v0.3 release evidence: the v6 sharded suite."""

    if _formal_host_is_windows():
        return ["formal knowledge-review release audit is Linux/WSL2-only; Windows is NO-GO"]
    try:
        semantic_receipt, _ = _strict_json_object(
            root / SEMANTIC_REVIEW_RECEIPT_PATH,
            "semantic knowledge-review receipt", root=root,
        )
        adversarial_receipt, _ = _strict_json_object(
            root / ADVERSARIAL_REVIEW_RECEIPT_PATH,
            "adversarial knowledge-review receipt", root=root,
        )
    except (OSError, ValueError) as exc:
        return [str(exc)]
    versions = {
        semantic_receipt.get("schema_version"),
        adversarial_receipt.get("schema_version"),
    }
    if versions != {REVIEW_SUITE_RECEIPT_SCHEMA_VERSION}:
        return [
            "formal v0.3 release requires v6 six-invocation knowledge-review "
            "receipts; legacy v4 evidence is historical verification only"
        ]
    if suite_evidence is None:
        return ["v6 knowledge review requires --knowledge-review-suite-evidence"]
    if any(value is not None for value in (
        semantic_raw, adversarial_raw, semantic_cache, adversarial_cache,
    )):
        return ["v6 knowledge review rejects legacy raw/cache arguments"]
    return _audit_knowledge_review_suite_release_gate(
        root, semantic_review=semantic_review,
        adversarial_review=adversarial_review,
        suite_evidence=suite_evidence,
    )


def verify_legacy_v4_review_evidence(
    root: Path, *, semantic_review: Path, adversarial_review: Path,
    semantic_raw: Path, adversarial_raw: Path,
    semantic_cache: Path, adversarial_cache: Path,
) -> list[str]:
    """Replay frozen v4 artifacts without granting v0.3 release approval."""

    if _formal_host_is_windows():
        return [
            "legacy knowledge-review evidence verification is Linux/WSL2-only; "
            "Windows is NO-GO"
        ]
    return _audit_knowledge_review_release_gate_v4(
        root, semantic_review=semantic_review,
        adversarial_review=adversarial_review,
        semantic_raw=semantic_raw, adversarial_raw=adversarial_raw,
        semantic_cache=semantic_cache, adversarial_cache=adversarial_cache,
    )


def _candidate_identity_from_environment(environment: dict[str, Any]) -> dict[str, str]:
    from eval.agent_ab.common import (
        ENVIRONMENT_SCHEMA_VERSION, _validate_runtime_identity,
    )

    if environment.get("schema_version") != ENVIRONMENT_SCHEMA_VERSION:
        raise ValueError("evaluation environment is not the required v2 schema")
    _validate_runtime_identity(environment.get("runtime_identity"))
    declared = environment.get("hlsgraph_v03")
    checks = environment.get("identity_checks")
    if not isinstance(declared, dict) or not isinstance(checks, list):
        raise ValueError("evaluation environment lacks the v0.3 identity")
    matching = [
        item.get("identity") for item in checks
        if isinstance(item, dict)
        and item.get("kind") == "verify-hlsgraph-wheel-installation"
        and item.get("arm") == "hlsgraph-v03"
    ]
    if len(matching) != 1 or not isinstance(matching[0], dict):
        raise ValueError("evaluation environment lacks one v0.3 wheel identity check")
    identity = matching[0]
    candidate = {
        "arm": "hlsgraph-v03",
        "version": str(declared.get("version", "")),
        "wheel_sha256": str(declared.get("wheel_sha256", "")),
        "installed_payload_sha256": str(identity.get("installed_payload_sha256", "")),
        "revision": str(declared.get("revision", "")),
        "source_revision": str(identity.get("source_revision", "")),
        "source_package_sha256": str(identity.get("source_package_sha256", "")),
        "wheel_package_sha256": str(identity.get("wheel_package_sha256", "")),
    }
    hashes = (
        candidate["wheel_sha256"], candidate["installed_payload_sha256"],
        candidate["source_package_sha256"], candidate["wheel_package_sha256"],
    )
    if (environment.get("official_profile") is not True
            or identity.get("schema_version") != "hlsgraph.agent_eval.wheel_identity.v1"
            or identity.get("verified") is not True
            or identity.get("version") != RELEASE_VERSION
            or candidate["version"] != RELEASE_VERSION
            or any(_SHA256_RE.fullmatch(value) is None for value in hashes)
            or re.fullmatch(r"[0-9a-f]{40}", candidate["revision"]) is None
            or candidate["source_revision"] != candidate["revision"]
            or candidate["wheel_package_sha256"] != candidate["source_package_sha256"]
            or identity.get("wheel_sha256") != candidate["wheel_sha256"]
            or identity.get("wheel_payload_sha256")
            != candidate["installed_payload_sha256"]
            or identity.get("installed_payload_sha256")
            != candidate["installed_payload_sha256"]):
        raise ValueError("evaluation environment has an inconsistent v0.3 identity")
    expected_source_checks = {
        "verify-v03-repo-clean": "",
        "record-v03-revision": candidate["revision"],
        "verify-v03-repo-clean-after": "",
        "record-v03-revision-after": candidate["revision"],
    }
    for kind, stdout in expected_source_checks.items():
        matching_checks = [
            item for item in checks
            if isinstance(item, dict) and item.get("kind") == kind
        ]
        if len(matching_checks) != 1 or matching_checks[0].get("stdout") != stdout:
            raise ValueError(f"evaluation environment lacks exact source check {kind}")
    return candidate


def _strict_json_lines(path: Path, label: str) -> tuple[list[dict[str, Any]], bytes]:
    data = _strict_file_bytes(path, label)
    if not data or not data.endswith(b"\n"):
        raise ValueError(f"{label} must be non-empty canonical JSONL ending in LF")

    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(data.splitlines(), start=1):
        if not raw:
            raise ValueError(f"{label} contains a blank line at {index}")
        try:
            value = json.loads(
                raw.decode("utf-8", errors="strict"),
                object_pairs_hook=no_duplicates,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ValueError(f"{label} contains non-finite JSON number {token}")
                ),
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"cannot read strict {label} line {index}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{label} line {index} is not an object")
        rows.append(value)
    return rows, data


def _verify_evaluation_raw_closure(
    *, environment: dict[str, Any], environment_bytes: bytes,
    eval_identity: Path, run_set_path: Path,
    frozen_run_set: dict[str, Any], run_set_bytes: bytes,
    scores_bytes: bytes,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Reload the exact 192-cell run set and deterministically rescore raw traces."""
    try:
        boundary = environment["runtime_identity"]["sandbox_boundary"]
        declared_work_root = Path(boundary["work_root"])
        declared_runs_root = Path(frozen_run_set["runs_root"])
    except (KeyError, TypeError) as exc:
        raise ValueError("evaluation does not declare exact work/runs roots") from exc
    if not declared_work_root.is_absolute() or not declared_runs_root.is_absolute():
        raise ValueError("evaluation work/runs roots must be absolute")
    identity_lexical, _ = _strict_path_components(
        eval_identity, "evaluation environment lock", root=declared_work_root,
    )
    run_set_lexical, _ = _strict_path_components(
        run_set_path, "evaluation run set", root=declared_runs_root,
    )
    expected_identity = declared_work_root / "environment.lock.json"
    expected_run_set = declared_runs_root / "run-set.json"
    if identity_lexical.resolve(strict=True) != expected_identity.resolve(strict=True):
        raise ValueError(
            "evaluation identity must be the declared work_root/environment.lock.json"
        )
    if run_set_lexical.resolve(strict=True) != expected_run_set.resolve(strict=True):
        raise ValueError("run-set input must be the declared runs_root/run-set.json")

    from eval.agent_ab.score import load_run_set, render_score_rows, score_runs

    environment_sha256 = hashlib.sha256(environment_bytes).hexdigest()
    loaded = load_run_set(
        declared_runs_root, declared_work_root,
        environment_lock_sha256=environment_sha256, environment=environment,
    )
    if loaded != frozen_run_set:
        raise ValueError("supplied run set differs from the fully validated 192-cell run set")
    rescored = score_runs(declared_runs_root, declared_work_root)
    if render_score_rows(rescored) != scores_bytes:
        raise ValueError(
            "evaluation scores differ byte-for-byte from deterministic raw-trace rescoring"
        )
    if (_strict_file_bytes(
        expected_identity, "evaluation environment lock after raw rescoring",
        root=declared_work_root,
    ) != environment_bytes or _strict_file_bytes(
        expected_run_set, "evaluation run set after raw rescoring",
        root=declared_runs_root,
    ) != run_set_bytes):
        raise ValueError("evaluation identity or run set changed during raw rescoring")
    return loaded, rescored


def _audit_evaluation_release_gate(
    wheel: Path, *, eval_identity: Path, static_report: Path,
    bootstrap_report: Path, scores: Path, run_set: Path, release_notes: Path,
) -> list[str]:
    """Bind release bytes and claims to one complete, frozen evaluation."""
    issues: list[str] = []
    try:
        environment, environment_bytes = _strict_json_object(
            eval_identity, "evaluation environment lock",
        )
        static, static_bytes = _strict_json_object(static_report, "static report")
        bootstrap, _bootstrap_bytes = _strict_json_object(
            bootstrap_report, "bootstrap report",
        )
        frozen_run_set, run_set_bytes = _strict_json_object(
            run_set, "evaluation run set",
        )
        score_rows, scores_bytes = _strict_json_lines(scores, "evaluation scores")
        from eval.agent_ab.common import load_environment_lock
        validated_environment = load_environment_lock(eval_identity)
        environment_after, environment_after_bytes = _strict_json_object(
            eval_identity, "evaluation environment lock after validation",
        )
        if (validated_environment != environment
                or environment_after != environment
                or environment_after_bytes != environment_bytes):
            raise ValueError(
                "evaluation environment changed during or disagrees with full v2 validation"
            )
        candidate = _candidate_identity_from_environment(environment)
    except (OSError, ValueError) as exc:
        return [str(exc)]

    environment_sha256 = hashlib.sha256(environment_bytes).hexdigest()
    static_sha256 = hashlib.sha256(static_bytes).hexdigest()
    scores_sha256 = hashlib.sha256(scores_bytes).hexdigest()
    suite_sha256 = environment.get("suite_asset_sha256")
    harness_sha256 = environment.get("evaluation_harness_sha256")
    if (_SHA256_RE.fullmatch(str(suite_sha256 or "")) is None
            or _SHA256_RE.fullmatch(str(harness_sha256 or "")) is None):
        issues.append("evaluation environment lacks frozen suite/harness digests")
    for label, report in (("static report", static), ("bootstrap report", bootstrap)):
        report_candidate = report.get("candidate_identity")
        if report_candidate != candidate:
            issues.append(f"{label} does not bind the evaluated v0.3 candidate identity")
        if (report.get("environment_lock_sha256") != environment_sha256
                or report.get("suite_asset_sha256") != suite_sha256
                or report.get("evaluation_harness_sha256") != harness_sha256):
            issues.append(f"{label} does not bind the exact evaluation environment")
    if (static.get("schema_version") != "hlsgraph.agent_eval.static_report.v1"
            or static.get("passed") is not True):
        issues.append("static retrieval report is absent, stale, or unpassed")
    if (bootstrap.get("schema_version") != "hlsgraph.agent_eval.bootstrap_report.v1"
            or bootstrap.get("static_report_sha256") != static_sha256):
        issues.append("bootstrap report does not bind the exact static report bytes")
    for field in ("scores_sha256", "run_set_sha256", "run_batch_sha256"):
        if _SHA256_RE.fullmatch(str(bootstrap.get(field, ""))) is None:
            issues.append(f"bootstrap report lacks the frozen {field}")
    if bootstrap.get("scores_sha256") != scores_sha256:
        issues.append("bootstrap report does not bind the exact score rows")
    if bootstrap.get("run_set_sha256") != frozen_run_set.get("run_set_sha256"):
        issues.append("bootstrap report does not bind the exact frozen run set")
    try:
        from eval.agent_ab.bootstrap import analyze as analyze_evaluation
        from eval.agent_ab.score import render_score_rows
        from eval.agent_ab.static_eval import render_static_json

        if render_score_rows(score_rows) != scores_bytes:
            raise ValueError("evaluation scores are not canonical JSONL")
        verified_run_set, verified_score_rows = _verify_evaluation_raw_closure(
            environment=environment, environment_bytes=environment_bytes,
            eval_identity=eval_identity, run_set_path=run_set,
            frozen_run_set=frozen_run_set, run_set_bytes=run_set_bytes,
            scores_bytes=scores_bytes,
        )
        recomputed = analyze_evaluation(
            verified_score_rows, static,
            environment_lock_sha256=environment_sha256,
            candidate_identity=candidate,
            scores_sha256=scores_sha256,
            static_report_sha256=static_sha256,
            run_set=verified_run_set,
        )
        if render_static_json(bootstrap) != _bootstrap_bytes:
            raise ValueError("bootstrap report is not canonical JSON")
        if recomputed != bootstrap:
            raise ValueError("bootstrap report differs from deterministic recomputation")
    except Exception as exc:  # deterministic verifier failures are release blockers
        issues.append(f"cannot independently recompute final evaluation: {exc}")

    try:
        wheel_bytes = _strict_file_bytes(wheel, "release wheel")
        wheel_sha256 = hashlib.sha256(wheel_bytes).hexdigest()
        package_sha256 = _release_wheel_package_digest(wheel_bytes)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        issues.append(f"cannot bind release wheel bytes: {exc}")
    else:
        if wheel_sha256 != candidate["wheel_sha256"]:
            issues.append("release wheel SHA-256 differs from the evaluated v0.3 wheel")
        if (package_sha256 != candidate["wheel_package_sha256"]
                or package_sha256 != candidate["source_package_sha256"]):
            issues.append("release wheel package bytes differ from the evaluated source package")

    gates = bootstrap.get("gates")
    supported = gates.get("performance_advantage_supported") if isinstance(gates, dict) else None
    if not isinstance(supported, bool):
        issues.append("bootstrap report lacks a boolean performance advantage decision")
        return issues
    try:
        notes_bytes = _strict_file_bytes(release_notes, "release notes")
        notes = notes_bytes.decode("utf-8", errors="strict")
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        issues.append(f"cannot read UTF-8 release notes: {exc}")
        return issues
    if not supported:
        if re.search(r"(?i)\bTechnical\s+Preview\b", notes) is None:
            issues.append(
                "release notes must explicitly say Technical Preview when advantage is unsupported"
            )
        if any(pattern.search(notes) for pattern in _ADVANTAGE_CLAIM_PATTERNS):
            issues.append(
                "release notes claim an advantage without performance_advantage_supported=true"
            )
    return issues


def _audit_technical_preview_release_notes(release_notes: Path) -> list[str]:
    """Allow an unevaluated preview only when its claims are explicitly bounded."""

    try:
        notes_bytes = _strict_file_bytes(release_notes, "release notes")
        notes = notes_bytes.decode("utf-8", errors="strict")
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        return [f"cannot read UTF-8 release notes: {exc}"]
    if re.search(r"(?i)\b(?:Technical|Developer)\s+Preview\b", notes) is None:
        return [
            "release notes must explicitly say Technical Preview or Developer "
            "Preview when the Agent A/B evaluation is omitted"
        ]
    if any(pattern.search(notes) for pattern in _ADVANTAGE_CLAIM_PATTERNS):
        return [
            "release notes claim an advantage without a completed Agent A/B "
            "evaluation"
        ]
    return []


def _audit_wheel(path: Path, root: Path, root_sbom: bytes) -> list[str]:
    issues: list[str] = []
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = [item.filename for item in infos]
        duplicates = _duplicate_archive_names(names)
        if duplicates:
            issues.append(f"wheel has duplicate member names: {duplicates}")
        for info in infos:
            name = info.filename
            if reason := _unsafe_archive_name(name):
                issues.append(f"unsafe wheel member {name!r}: {reason}")
            mode = (info.external_attr >> 16) & 0xFFFF
            if info.create_system == 3 and stat.S_ISLNK(mode):
                issues.append(f"wheel contains linked member: {name}")
        if duplicates or any(
            item.startswith("unsafe wheel member")
            or item.startswith("wheel contains linked member")
            for item in issues
        ):
            return issues

        info_by_name = {item.filename: item for item in infos}
        for name, info in info_by_name.items():
            if marker := _forbidden(name):
                issues.append(f"forbidden wheel member {name} ({marker})")
            if not info.is_dir():
                issues.extend(_scan(name, archive.read(info)))

        expected_package = _expected_wheel_package(root)
        wheel_package = {
            name: archive.read(info) for name, info in info_by_name.items()
            if name.casefold().startswith("hlsgraph/") and not info.is_dir()
        }
        missing_package = sorted(set(expected_package) - set(wheel_package))
        extra_package = sorted(set(wheel_package) - set(expected_package))
        if missing_package:
            issues.append(f"wheel is missing source package files: {missing_package}")
        if extra_package:
            issues.append(f"wheel has extra source package files: {extra_package}")
        for name in sorted(set(expected_package) & set(wheel_package)):
            if wheel_package[name] != expected_package[name]:
                issues.append(f"wheel package bytes differ from source: {name}")

        roots = {
            name.split("/", 1)[0] for name in names
            if name.split("/", 1)[0].endswith(".dist-info")
        }
        if len(roots) != 1:
            issues.append("wheel must have exactly one dist-info root")
            return issues
        dist_info = roots.pop()
        required = {
            f"{dist_info}/METADATA", f"{dist_info}/RECORD",
            f"{dist_info}/sboms/sbom.spdx.json",
        }
        missing = required - set(info_by_name)
        if missing:
            issues.append(f"wheel is missing: {sorted(missing)}")
            return issues
        license_members = [
            name for name in info_by_name if name.startswith(f"{dist_info}/licenses/")
        ]
        if not any(name.endswith("/LICENSE") for name in license_members):
            issues.append("wheel has no Apache-2.0 LICENSE in dist-info/licenses")

        issues.extend(_audit_wheel_metadata(
            archive.read(f"{dist_info}/METADATA")
        ))

        try:
            record_rows = list(csv.reader(io.StringIO(
                archive.read(f"{dist_info}/RECORD").decode("utf-8")
            )))
        except (UnicodeDecodeError, csv.Error) as exc:
            issues.append(f"invalid wheel RECORD: {exc}")
            return issues
        record_names = [row[0] for row in record_rows if row]
        duplicate_records = _duplicate_archive_names(record_names)
        if duplicate_records:
            issues.append(f"wheel RECORD has duplicate paths: {duplicate_records}")
        recorded = {row[0]: row for row in record_rows if row}
        extra_records = sorted(set(recorded) - set(info_by_name))
        if extra_records:
            issues.append(f"wheel RECORD names absent archive members: {extra_records}")
        for name in info_by_name:
            if name == f"{dist_info}/RECORD":
                continue
            row = recorded.get(name)
            if not row or len(row) < 3 or not row[1].startswith("sha256="):
                issues.append(f"missing RECORD hash: {name}")
                continue
            data = archive.read(info_by_name[name])
            digest = base64.urlsafe_b64encode(
                hashlib.sha256(data).digest()
            ).rstrip(b"=").decode("ascii")
            if row[1] != "sha256=" + digest or row[2] != str(len(data)):
                issues.append(f"invalid RECORD entry: {name}")

        wheel_sbom = archive.read(f"{dist_info}/sboms/sbom.spdx.json")
        if wheel_sbom != root_sbom:
            issues.append("wheel SBOM does not exactly match root sbom.spdx.json")
        try:
            sbom = json.loads(root_sbom)
            for item in sbom.get("files", []):
                file_name = item.get("fileName", "")
                if not file_name.startswith("./src/"):
                    continue
                member_name = file_name[len("./src/"):]
                source_path = root / file_name[len("./"):]
                if member_name not in info_by_name:
                    issues.append(f"wheel is missing SBOM vendor file: {member_name}")
                elif source_path.is_file() and archive.read(member_name) != source_path.read_bytes():
                    issues.append(f"wheel vendor bytes differ from source: {member_name}")
        except json.JSONDecodeError:
            # The root-SBOM audit reports the more specific parse failure.
            pass
    return issues


def _review_bound_sdist_paths(root: Path) -> set[str]:
    """Return source paths whose exact bytes participate in knowledge review."""

    paths = {
        REVIEW_SCHEMA_PATH,
        REVIEW_RECEIPT_SCHEMA_PATH,
        SEMANTIC_REVIEW_PROMPT_PATH,
        ADVERSARIAL_REVIEW_PROMPT_PATH,
        SEMANTIC_REVIEW_PATH,
        ADVERSARIAL_REVIEW_PATH,
        SEMANTIC_REVIEW_RECEIPT_PATH,
        ADVERSARIAL_REVIEW_RECEIPT_PATH,
        SEMANTIC_REVIEW_TRACE_PATH,
        ADVERSARIAL_REVIEW_TRACE_PATH,
        CITATION_AUDIT_PATH,
        "tools/audit_knowledge_citations.py",
        "tools/knowledge_review_surface.py",
        "tools/run_knowledge_review.py",
        "tools/audit_release.py",
    }
    try:
        from tools import apply_knowledge_review_suite_attestation as suite_attestation
    except (ImportError, ModuleNotFoundError):
        suite_paths = {
            REVIEW_SHARD_SCHEMA_PATH,
            REVIEW_SUITE_EVIDENCE_SCHEMA_PATH,
            REVIEW_SUITE_RECEIPT_SCHEMA_PATH,
            REVIEW_SUITE_TRACE_SCHEMA_PATH,
            SEMANTIC_SHARD_PROMPT_PATH,
            ADVERSARIAL_SHARD_PROMPT_PATH,
            "tools/knowledge_review_shards.py",
            "tools/knowledge_review_suite_cache.py",
            "tools/knowledge_review_suite_replay.py",
            "tools/run_knowledge_review_suite.py",
            "tools/seal_knowledge_review_suite.py",
            "tools/execute_knowledge_review_suite.py",
            "tools/apply_knowledge_review_suite_attestation.py",
        }
    else:
        suite_paths = set(suite_attestation.SUITE_REVIEW_SOURCE_PATHS)
    paths.update(suite_paths)
    implementation_root = root / "src" / "hlsgraph"
    for path in implementation_root.rglob("*.py"):
        relative = path.relative_to(root)
        if path.is_file() and not any(
            part in SOURCE_SKIP_DIRS for part in relative.parts
        ):
            paths.add(relative.as_posix())
    for path in (implementation_root / "knowledge" / "packs").glob("*.json"):
        if path.is_file():
            paths.add(path.relative_to(root).as_posix())
    return paths


def _audit_sdist(
    path: Path, root_sbom: bytes, *, root: Path | None = None,
) -> list[str]:
    issues: list[str] = []
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        names = [item.name for item in members]
        duplicates = _duplicate_archive_names(names)
        if duplicates:
            issues.append(f"sdist has duplicate member names: {duplicates}")
        safe_members: list[tarfile.TarInfo] = []
        for member in members:
            if reason := _unsafe_archive_name(member.name):
                issues.append(f"unsafe sdist member {member.name!r}: {reason}")
                continue
            if not (member.isfile() or member.isdir()):
                issues.append(
                    f"sdist contains linked or special member: {member.name}"
                )
                continue
            safe_members.append(member)
        roots = {member.name.split("/", 1)[0] for member in safe_members}
        expected_root = f"hlsgraph-{RELEASE_VERSION}"
        if roots != {expected_root}:
            issues.append(
                f"sdist must have exactly the root {expected_root!r}, found {sorted(roots)}"
            )
        files = [item for item in safe_members if item.isfile()]
        stripped = {
            item.name.removeprefix(expected_root + "/"): item
            for item in files if item.name.startswith(expected_root + "/")
        }
        stripped_bytes: dict[str, bytes] = {}
        for name, member in stripped.items():
            if marker := _forbidden(name, sdist=True):
                issues.append(f"forbidden sdist member {name} ({marker})")
            stream = archive.extractfile(member)
            if stream:
                data = stream.read()
                stripped_bytes[name] = data
                issues.extend(_scan(name, data))
        missing = REQUIRED_SDIST - set(stripped)
        if missing:
            issues.append(f"sdist is missing: {sorted(missing)}")
        if root is not None:
            try:
                expected_installable = _expected_sdist_installable(root)
            except (OSError, ValueError) as exc:
                issues.append(f"cannot enumerate exact sdist source inputs: {exc}")
                expected_installable = {}
            actual_installable = {
                name: data for name, data in stripped_bytes.items()
                if name.casefold().startswith("src/hlsgraph/")
                or name in SDIST_BUILD_BOUND_PATHS
            }
            missing_installable = sorted(
                set(expected_installable) - set(actual_installable)
            )
            extra_installable = sorted(
                set(actual_installable) - set(expected_installable)
            )
            if missing_installable:
                issues.append(
                    "sdist is missing exact installable source/build inputs: "
                    + repr(missing_installable)
                )
            if extra_installable:
                issues.append(
                    "sdist has extra installable source/build inputs: "
                    + repr(extra_installable)
                )
            for relative in sorted(
                set(expected_installable) & set(actual_installable)
            ):
                if actual_installable[relative] != expected_installable[relative]:
                    issues.append(
                        f"sdist installable source/build bytes differ from source: {relative}"
                    )
            for relative in sorted(_review_bound_sdist_paths(root)):
                source_path = root / relative
                member = stripped.get(relative)
                if not source_path.is_file():
                    issues.append(
                        f"review-bound source file is missing: {relative}"
                    )
                    continue
                if member is None:
                    issues.append(
                        f"sdist is missing review-bound source bytes: {relative}"
                    )
                    continue
                try:
                    source_bytes = _strict_file_bytes(
                        source_path, f"review-bound source {relative}", root=root,
                    )
                except (OSError, ValueError) as exc:
                    issues.append(str(exc))
                    continue
                if stripped_bytes.get(relative) != source_bytes:
                    issues.append(
                        f"sdist review-bound bytes differ from source: {relative}"
                    )
        sbom_member = stripped.get("sbom.spdx.json")
        if sbom_member:
            if stripped_bytes.get("sbom.spdx.json") != root_sbom:
                issues.append("sdist SBOM does not exactly match root sbom.spdx.json")
    return issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dist", type=Path)
    parser.add_argument(
        "--preflight-only", action="store_true",
        help=(
            "run archive/privacy hygiene only; this mode deliberately skips final "
            "knowledge-review and evaluation approval and cannot approve a release"
        ),
    )
    parser.add_argument(
        "--technical-preview-without-agent-eval", action="store_true",
        help=(
            "approve only a Technical/Developer Preview without Agent A/B evidence; "
            "all evaluation evidence options must be omitted, --release-notes is "
            "required, and performance-advantage claims are forbidden"
        ),
    )
    parser.add_argument(
        "--eval-identity", "--environment-lock", dest="eval_identity", type=Path,
        help="explicit prepared environment.lock.json used by the final evaluation",
    )
    parser.add_argument("--static-report", type=Path)
    parser.add_argument("--bootstrap-report", type=Path)
    parser.add_argument("--scores", type=Path)
    parser.add_argument("--run-set", type=Path)
    parser.add_argument("--release-notes", type=Path)
    parser.add_argument(
        "--semantic-review", type=Path,
        help=(
            "final semantic review JSON; it must resolve to "
            f"{SEMANTIC_REVIEW_PATH} in the audited checkout"
        ),
    )
    parser.add_argument(
        "--adversarial-review", type=Path,
        help=(
            "final adversarial review JSON; it must resolve to "
            f"{ADVERSARIAL_REVIEW_PATH} in the audited checkout"
        ),
    )
    parser.add_argument(
        "--semantic-review-raw", type=Path,
        help=(
            "legacy v4 compatibility input; the formal v0.3 release CLI "
            "rejects it"
        ),
    )
    parser.add_argument(
        "--adversarial-review-raw", type=Path,
        help=(
            "legacy v4 compatibility input; the formal v0.3 release CLI "
            "rejects it"
        ),
    )
    parser.add_argument(
        "--semantic-review-cache", type=Path,
        help="legacy v4 compatibility input; rejected for v0.3 release approval",
    )
    parser.add_argument(
        "--adversarial-review-cache", type=Path,
        help="legacy v4 compatibility input; rejected for v0.3 release approval",
    )
    parser.add_argument(
        "--knowledge-review-suite-evidence", type=Path,
        help=(
            "restricted external v6 suite work root containing six raw streams, "
            "six projected caches, two full caches, and suite-evidence.json"
        ),
    )
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    root_sbom = (root / "sbom.spdx.json").read_bytes()
    wheels = sorted(args.dist.glob(f"hlsgraph-{RELEASE_VERSION}-*.whl"))
    sdists = sorted(args.dist.glob(f"hlsgraph-{RELEASE_VERSION}.tar.gz"))
    issues = _audit_source_tree(root) + _audit_sbom(root_sbom, root)
    if len(wheels) != 1:
        issues.append(f"expected one v{RELEASE_VERSION} wheel, found {len(wheels)}")
    else:
        issues.extend(_audit_wheel(wheels[0], root, root_sbom))
    if len(sdists) != 1:
        issues.append(f"expected one v{RELEASE_VERSION} sdist, found {len(sdists)}")
    else:
        issues.extend(_audit_sdist(sdists[0], root_sbom, root=root))
    if len(wheels) == 1 and len(sdists) == 1:
        try:
            wheel_data = _strict_file_bytes(
                wheels[0], "release wheel", root=args.dist.absolute(),
            )
            sdist_data = _strict_file_bytes(
                sdists[0], "release sdist", root=args.dist.absolute(),
            )
            wheel_package_digest = _release_wheel_package_digest(wheel_data)
            sdist_package_digest = _release_sdist_package_digest(sdist_data)
        except (OSError, ValueError, tarfile.TarError, zipfile.BadZipFile) as exc:
            issues.append(f"cannot compare wheel and sdist package payloads: {exc}")
        else:
            if wheel_package_digest != sdist_package_digest:
                issues.append(
                    "release wheel and sdist installable package payloads differ"
                )
    evaluation_evidence_inputs = (
        args.eval_identity, args.static_report, args.bootstrap_report,
        args.scores, args.run_set,
    )
    final_claim_inputs = (
        *evaluation_evidence_inputs, args.release_notes,
    )
    if args.preflight_only:
        if args.technical_preview_without_agent_eval or any(
            value is not None for value in final_claim_inputs
        ) or any(
            value is not None for value in (
                args.semantic_review, args.adversarial_review,
                args.semantic_review_raw, args.adversarial_review_raw,
                args.semantic_review_cache, args.adversarial_review_cache,
                args.knowledge_review_suite_evidence,
            )
        ):
            issues.append(
                "--preflight-only cannot consume or imply final review/evaluation approval"
            )
    else:
        semantic_review = args.semantic_review or root / SEMANTIC_REVIEW_PATH
        adversarial_review = args.adversarial_review or root / ADVERSARIAL_REVIEW_PATH
        legacy_review_inputs = (
            args.semantic_review_raw, args.adversarial_review_raw,
            args.semantic_review_cache, args.adversarial_review_cache,
        )
        if args.knowledge_review_suite_evidence is None:
            issues.append(
                "formal v0.3 release audit requires "
                "--knowledge-review-suite-evidence for the v6 six-invocation suite"
            )
        elif any(value is not None for value in legacy_review_inputs):
            issues.append(
                "formal v0.3 release audit rejects legacy v4 raw/cache inputs"
            )
        else:
            issues.extend(_audit_knowledge_review_release_gate(
                root, semantic_review=semantic_review,
                adversarial_review=adversarial_review,
                suite_evidence=args.knowledge_review_suite_evidence,
            ))
        if args.technical_preview_without_agent_eval:
            if any(value is not None for value in evaluation_evidence_inputs):
                issues.append(
                    "--technical-preview-without-agent-eval requires every Agent "
                    "A/B evaluation evidence input to be omitted"
                )
            if args.release_notes is None:
                issues.append(
                    "--technical-preview-without-agent-eval requires --release-notes"
                )
            else:
                issues.extend(
                    _audit_technical_preview_release_notes(args.release_notes)
                )
        elif not all(value is not None for value in final_claim_inputs):
            issues.append(
                "formal release audit requires --eval-identity, --static-report, "
                "--bootstrap-report, --scores, --run-set, and --release-notes; "
                "use --preflight-only "
                "only for non-release hygiene"
            )
        elif len(wheels) == 1:
            issues.extend(_audit_evaluation_release_gate(
                wheels[0], eval_identity=args.eval_identity,
                static_report=args.static_report,
                bootstrap_report=args.bootstrap_report,
                scores=args.scores, run_set=args.run_set,
                release_notes=args.release_notes,
            ))
    if issues:
        for issue in issues:
            print(f"ERROR: {issue}", file=sys.stderr)
        return 1
    if args.preflight_only:
        print(
            "PRE-FLIGHT ONLY: source and archives passed hygiene checks; "
            "knowledge review and evaluation were not approved, so this is not "
            "a release approval"
        )
    elif args.technical_preview_without_agent_eval:
        print(
            "Technical/Developer Preview release source and archives passed privacy, "
            "RECORD, SPDX, final knowledge-review, and no-advantage-claim checks; "
            "Agent A/B evaluation and performance-advantage approval were omitted"
        )
    else:
        print(
            "release source and archives passed privacy, RECORD, SPDX, final "
            "knowledge-review, and frozen evaluation-byte/claim checks"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
