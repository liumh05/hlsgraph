"""Run and replay the legacy v4 monolithic knowledge-review contract.

This module remains integrity-bound because v5 reuses its cache, snapshot, and
replay primitives and because historical v4 artifacts must remain verifiable.
Its ``review`` and ``seal`` CLI operations cannot approve a v0.3 release; the
formal v0.3 workflow is ``execute_knowledge_review_suite.py`` followed by
``apply_knowledge_review_suite_attestation.py`` and the v5 release audit.

V4 review execution is Linux/WSL2-only.  The model runs in a default-deny named
permission profile which exposes only Codex's minimal system runtime, one
frozen evidence cache and one frozen Codex runtime directory.  The public
checkout, ``CODEX_HOME`` and every unrelated host path remain outside that
allowlist.
The raw ``codex exec --json`` stream is retained outside the checkout and is
the authority for the normalized public trace, result and receipt.

This module deliberately does not offer a generic shell.  A completed command
event is accepted only when it is one of the small read-only grammars below.
Unknown events or tools fail the review instead of being ignored.
"""
from __future__ import annotations

import argparse
import hashlib
from http.client import HTTPException
import json
import os
import re
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


# ``python3 tools/run_knowledge_review.py`` puts ``tools/`` rather than the
# checkout root on sys.path.  The documented script entry point nevertheless
# imports the sibling ``eval`` package and the ``tools`` namespace package.
SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))


SEMANTIC_PROTOCOL = "hlsgraph.knowledge-review.semantic.v1"
ADVERSARIAL_PROTOCOL = "hlsgraph.knowledge-review.adversarial.v1"
PROTOCOLS = frozenset({SEMANTIC_PROTOCOL, ADVERSARIAL_PROTOCOL})
MODEL = "gpt-5.6-sol"
REASONING_EFFORT = "medium"
CODEX_CLI_VERSION = "codex-cli 0.144.0"
TRACE_SCHEMA_VERSION = "hlsgraph.knowledge-review.tool-trace.v3"
RECEIPT_SCHEMA_VERSION = "hlsgraph.knowledge-review.cli-receipt.v4"
BOUNDARY_CONTRACT_SCHEMA_VERSION = (
    "hlsgraph.knowledge-review.boundary-contract.v3"
)
RUNTIME_MANIFEST_SCHEMA_VERSION = (
    "hlsgraph.knowledge-review.runtime-manifest.v3"
)
RUNTIME_OWNERSHIP_POLICY = (
    "caller_owned_frozen_0500_no_links_exact_codex_bwrap_v2"
)
BOUNDARY_POLICY = "default_deny_minimal_allowlist_v1"
CACHE_PARENT_POLICY = "caller_owned_0700_single_cache_v1"
EVIDENCE_PARENT_POLICY = "caller_owned_0700_dedicated_evidence_v1"
REVIEW_SCHEMA_PATH = "tools/knowledge_review.schema.json"
REVIEW_RECEIPT_SCHEMA_PATH = "tools/knowledge_review_receipt.schema.json"
CITATION_AUDIT_PATH = "docs/knowledge-citation-audit-v0.3.json"
CITATION_EVIDENCE_PATH = "docs/knowledge-review-evidence-v0.3.json"
CITATION_EVIDENCE_SCHEMA_PATH = "tools/knowledge_review_evidence.schema.json"
CITATION_EVIDENCE_SCHEMA_VERSION = (
    "hlsgraph.knowledge-review.evidence-map.v2"
)
RUNNER_PATH = "tools/run_knowledge_review.py"
CITATION_GENERATOR_PATH = "tools/audit_knowledge_citations.py"
SURFACE_HELPER_PATH = "tools/knowledge_review_surface.py"
RELEASE_AUDITOR_PATH = "tools/audit_release.py"
# Integrity-only inputs for the formal six-invocation v5 suite.  They are
# frozen into each ReviewSnapshot, but remain outside
# MODEL_INSPECTION_EXACT_PATHS: the reviewer is expected to inspect the HLS
# implementation and assigned evidence, not its own orchestration code.
SUITE_REVIEW_SOURCE_PATHS = frozenset({
    "pyproject.toml",
    "tools/apply_knowledge_review_suite_attestation.py",
    "tools/execute_knowledge_review_suite.py",
    "tools/knowledge_review_shards.py",
    "tools/run_knowledge_review_suite.py",
    "tools/knowledge_review_suite_cache.py",
    "tools/knowledge_review_suite_replay.py",
    "tools/seal_knowledge_review_suite.py",
    "tools/knowledge_review_shard.schema.json",
    "tools/knowledge_review_suite_evidence.schema.json",
    "tools/knowledge_review_suite_receipt.schema.json",
    "tools/knowledge_review_suite_trace.schema.json",
    "tools/knowledge_review_prompts/semantic_shard.md",
    "tools/knowledge_review_prompts/adversarial_shard.md",
})
CACHE_SCHEMA_VERSION = "hlsgraph.knowledge-review.cache.v3"
CACHE_MANIFEST_NAME = "manifest.json"
CHUNK_CONTRACT_SCHEMA_VERSION = "hlsgraph.knowledge-review.chunks.v1"
MAX_REVIEW_CHUNK_BYTES = 24_000
TOOL_OUTPUT_TOKEN_LIMIT = 50_000
MAX_INITIAL_PROMPT_BYTES = 512 * 1024
CACHE_DIRECTORY_MODE = 0o500
CACHE_FILE_MODE = 0o400
PDFTOTEXT_ALLOWED_PATH = "/usr/bin/pdftotext"
OFFICIAL_CODEX_ELF_SHA256 = (
    "901923c1808a151f6926d41d703c17ad48815662cefb1c8d832a052c44271429"
)
OFFICIAL_CODEX_BWRAP_SHA256 = (
    "77360cb751ccedc5971391444ac86a8a33c15b04d6b4a6fe45f5d25496e62c4c"
)
CODEX_EXECUTABLE_RELATIVE_PATH = "codex"
CODEX_BWRAP_RELATIVE_PATH = "codex-resources/bwrap"
RUNTIME_INITIAL_PATH_TOKENS = (
    "$CODEX_RUNTIME/codex-resources", "/usr/bin", "/bin",
)
OFFICIAL_CODEX_RELEASE_ASSET_SHA256 = (
    "725883fc20ab4af3072829aaa0edf6d12c216238f9f7315a6656b950fb05c8bb"
)
MAX_CITATION_BYTES = 32 * 1024 * 1024
MAX_REDIRECTS = 5
MAX_FETCH_ATTEMPTS = 3
MAX_EVIDENCE_LINE_RANGE = 1024
MAX_PDF_TEXT_BYTES = 32 * 1024 * 1024
MAX_PARSER_STDERR_BYTES = 64 * 1024
MAX_PARSER_VERSION_BYTES = 16 * 1024
MAX_RAW_REVIEW_BYTES = 64 * 1024 * 1024
MAX_EVIDENCE_IDENTITY_CHARS = 256
IMPLEMENTATION_SURFACE_HASH_KEY = "src/hlsgraph/**/*.py#implementation-surface"
PACK_SURFACE_HASH_PREFIX = "src/hlsgraph/knowledge/packs/"
PACK_SURFACE_HASH_SUFFIX = "#semantic-surface"
SURFACE_HELPER_HASH_KEY = SURFACE_HELPER_PATH + "#sha256"

PROTOCOL_FILES = {
    SEMANTIC_PROTOCOL: {
        "prompt": "tools/knowledge_review_prompts/semantic.md",
        "result": "docs/knowledge-review-v0.3.semantic.json",
        "trace": "docs/knowledge-review-v0.3.semantic.trace.jsonl",
        "receipt": "docs/knowledge-review-v0.3.semantic.receipt.json",
    },
    ADVERSARIAL_PROTOCOL: {
        "prompt": "tools/knowledge_review_prompts/adversarial.md",
        "result": "docs/knowledge-review-v0.3.adversarial.json",
        "trace": "docs/knowledge-review-v0.3.adversarial.trace.jsonl",
        "receipt": "docs/knowledge-review-v0.3.adversarial.receipt.json",
    },
}

DISABLED_CODEX_FEATURES = (
    "browser_use", "browser_use_external", "browser_use_full_cdp_access",
    "in_app_browser", "standalone_web_search", "computer_use",
    "image_generation", "apps", "enable_mcp_apps", "multi_agent",
    "multi_agent_v2", "plugins", "plugin_sharing", "remote_plugin",
    "hooks", "workspace_dependencies", "code_mode", "code_mode_host",
    "code_mode_only",
)
PERMISSION_PROFILE = "hlsgraph_knowledge_review"
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{7,127}")
_CALL_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{2,127}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_FORBIDDEN_SHELL = re.compile(
    r"[;&|><`\r\n]|\$\(|\$\{|\$env:|%[A-Za-z_][A-Za-z0-9_]*%",
    re.IGNORECASE,
)
_ALLOWED_EVENT_TYPES = frozenset({
    "thread.started", "turn.started", "item.started", "item.completed",
    "turn.completed",
})
_ALLOWED_NONCOMMAND_ITEMS = frozenset({"reasoning", "agent_message"})
CONTROLLED_REVIEW_ISSUE_CODES = frozenset({
    "semantic_gap", "activation_bypass", "citation_unavailable",
    "citation_rejected", "contract_violation",
})
CONTROLLED_CITATION_ISSUE_CODES = frozenset({
    "locator_unavailable", "resolver_mismatch", "version_mismatch",
    "section_mismatch", "paraphrase_unsupported",
    "applicability_too_broad", "inspection_incomplete",
})
MODEL_INSPECTION_EXACT_PATHS = frozenset({
    CITATION_AUDIT_PATH,
    CITATION_EVIDENCE_PATH,
    "src/hlsgraph/bundle.py",
    "src/hlsgraph/graph.py",
    "src/hlsgraph/manifest.py",
    "src/hlsgraph/knowledge/activation.py",
    "src/hlsgraph/knowledge/core.py",
    "src/hlsgraph/knowledge/supported_targets.py",
    "src/hlsgraph/retrieval.py",
    "src/hlsgraph/model.py",
    "src/hlsgraph/evidence_policy.py",
    "src/hlsgraph/runner/core.py",
    "src/hlsgraph/runner/staging.py",
    "src/hlsgraph/store/migrations.py",
    "src/hlsgraph/store/sqlite.py",
    "src/hlsgraph/extract/base.py",
    "src/hlsgraph/extract/index_authorization.py",
    "src/hlsgraph/extract/directives.py",
    "src/hlsgraph/extract/directive_identity.py",
    "src/hlsgraph/extract/directive_replay.py",
    "src/hlsgraph/extract/llvm.py",
    "src/hlsgraph/extract/mlir.py",
    "src/hlsgraph/extract/observation_replay.py",
    "src/hlsgraph/extract/source.py",
    "src/hlsgraph/extract/static_features.py",
    "src/hlsgraph/extract/vitis.py",
    "src/hlsgraph/extract/vivado.py",
    "src/hlsgraph/static_aggregate.py",
})
# The result/evidence schemas remain mandatory, immutable snapshot inputs and
# are validated by the trusted runner.  The model sees the corresponding
# shard contract and projected evidence values, so duplicating these generic
# validation schemas as citation chunks spends context without adding HLS
# implementation evidence.


def _model_inspection_required(path: str) -> bool:
    return (
        path in MODEL_INSPECTION_EXACT_PATHS
        or path.startswith("review-projections/v1/")
        or path.startswith(PACK_SURFACE_HASH_PREFIX)
        and path.endswith(".json")
    )


@dataclass(frozen=True)
class ReviewReplay:
    protocol_id: str
    invocation_id: str
    thread_id: str
    raw_sha256: str
    result: dict[str, Any]
    result_bytes: bytes
    trace_bytes: bytes


@dataclass(frozen=True)
class ReviewFileSnapshot:
    """One immutable logical review input and its cache projection."""

    path: str
    hash_kind: str
    sha256: str
    cache_sha256: str
    payload: bytes = field(repr=False, compare=True)

    def inventory(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "hash_kind": self.hash_kind,
            "sha256": self.sha256,
            "cache_path": f"files/{self.path}",
            "cache_sha256": self.cache_sha256,
            "cache_size": len(self.payload),
        }


@dataclass(frozen=True)
class ReviewSnapshot:
    """All checkout bytes and semantic projections promised to one review."""

    protocol_id: str
    files: tuple[ReviewFileSnapshot, ...]
    review_surface_sha256: tuple[tuple[str, str], ...]
    implementation_surface_sha256: str
    citation_audit_sha256: str
    citation_evidence_sha256: str
    output_schema_sha256: str
    receipt_schema_sha256: str
    exact_citation_urls: tuple[str, ...]

    @property
    def file_map(self) -> dict[str, ReviewFileSnapshot]:
        return {item.path: item for item in self.files}

    @property
    def surfaces(self) -> dict[str, str]:
        return dict(self.review_surface_sha256)

    def inventory(self) -> dict[str, Any]:
        return {
            "protocol_id": self.protocol_id,
            "review_surface_sha256": self.surfaces,
            "implementation_surface_sha256": self.implementation_surface_sha256,
            "citation_audit_sha256": self.citation_audit_sha256,
            "citation_evidence_sha256": self.citation_evidence_sha256,
            "output_schema_sha256": self.output_schema_sha256,
            "receipt_schema_sha256": self.receipt_schema_sha256,
            "exact_citation_urls": list(self.exact_citation_urls),
            "required_files": [item.inventory() for item in self.files],
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.inventory())).hexdigest()


@dataclass(frozen=True)
class TrustedFetch:
    status: int
    final_url: str
    redirect_chain: tuple[str, ...]
    content_type: str
    body: bytes = field(repr=False)
    charset: str | None = None
    content_length: int | None = None


@dataclass(frozen=True)
class TextDerivation:
    text: bytes = field(repr=False)
    parser_id: str
    parser_version: str
    command_sha256: str
    executable_sha256: str | None = None
    version_output_sha256: str | None = None


def _chunk_contract() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": CHUNK_CONTRACT_SCHEMA_VERSION,
        "encoding": "utf-8-strict",
        "max_chunk_bytes": MAX_REVIEW_CHUNK_BYTES,
        "read_command": "head -n 100000000 $CHUNK_PATH",
        "tool_output_token_limit": TOOL_OUTPUT_TOKEN_LIMIT,
        "complete_read_policy": "every_manifested_chunk_exact_output_v1",
        "reconstruction_policy": "contiguous_byte_ranges_exact_sha256_v1",
    }
    payload["sha256"] = hashlib.sha256(_canonical_json(payload)).hexdigest()
    return payload


def _inspection_contract(
    files: Sequence[dict[str, Any]],
    citations: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    required = sorted(
        str(item["path"]) for item in files
        if item.get("model_inspection_required") is True
    )
    integrity_only = sorted(
        str(item["path"]) for item in files
        if item.get("model_inspection_required") is False
    )
    citation_required = sorted(
        str(item["requested_url"]) for item in citations
        if item.get("inspection_required") is True
    )
    citation_identity_only = sorted(
        str(item["requested_url"]) for item in citations
        if item.get("inspection_required") is False
    )
    payload: dict[str, Any] = {
        "schema_version": "hlsgraph.knowledge-review.inspection-scope.v2",
        "model_inspection_required": required,
        "integrity_bound_only": integrity_only,
        "citation_section_inspection_required": citation_required,
        "citation_identity_bound_only": citation_identity_only,
        "policy": "explicit_activation_tcb_plus_rule_sections_v2",
        "integrity_statement": (
            "integrity-bound files and document-only locators invalidate the "
            "snapshot but are not claimed as model content inspection"
        ),
    }
    payload["sha256"] = hashlib.sha256(_canonical_json(payload)).hexdigest()
    return payload


@dataclass(frozen=True)
class ReviewCache:
    root: Path
    manifest: dict[str, Any]
    manifest_bytes: bytes = field(repr=False)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.manifest_bytes).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False,
    ) + "\n").encode("utf-8")


def _canonical_jsonl(rows: Iterable[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
        for row in rows
    ).encode("utf-8")


def _strict_json_bytes(data: bytes, *, label: str) -> Any:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(
            data.decode("utf-8", errors="strict"),
            object_pairs_hook=no_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"{label} contains non-finite number {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"cannot parse strict {label}: {exc}") from exc


def _strict_jsonl(data: bytes, *, label: str) -> list[dict[str, Any]]:
    if not data or not data.endswith(b"\n"):
        raise ValueError(f"{label} must be non-empty JSONL ending in LF")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(data.splitlines(), 1):
        if not line:
            raise ValueError(f"{label} contains a blank line at {line_number}")
        value = _strict_json_bytes(line, label=f"{label}:{line_number}")
        if not isinstance(value, dict):
            raise ValueError(f"{label}:{line_number} is not an object")
        rows.append(value)
    return rows


def _content_text(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        pieces = [
            text for item in value
            if (text := _content_text(item)) is not None
        ]
        return "\n".join(pieces) if pieces else None
    if isinstance(value, dict):
        for key in ("text", "output_text", "content"):
            text = _content_text(value.get(key))
            if text is not None:
                return text
    return None


def required_read_paths(root: Path, protocol_id: str) -> set[str]:
    files = PROTOCOL_FILES.get(protocol_id)
    if files is None:
        raise ValueError(f"unknown knowledge-review protocol: {protocol_id}")
    paths = {
        REVIEW_SCHEMA_PATH, REVIEW_RECEIPT_SCHEMA_PATH, CITATION_AUDIT_PATH,
        CITATION_EVIDENCE_PATH, CITATION_EVIDENCE_SCHEMA_PATH,
        RUNNER_PATH, CITATION_GENERATOR_PATH, SURFACE_HELPER_PATH,
        RELEASE_AUDITOR_PATH,
        *(item["prompt"] for item in PROTOCOL_FILES.values()),
        *SUITE_REVIEW_SOURCE_PATHS,
    }
    implementation = root / "src" / "hlsgraph"
    paths.update(
        path.relative_to(root).as_posix()
        for path in implementation.rglob("*.py") if path.is_file()
    )
    paths.update(
        path.relative_to(root).as_posix()
        for path in (implementation / "knowledge" / "packs").glob("*.json")
        if path.is_file()
    )
    return paths


def _citation_rows(root: Path) -> list[dict[str, Any]]:
    value = _strict_json_bytes(
        (root / CITATION_AUDIT_PATH).read_bytes(), label="citation audit",
    )
    if not isinstance(value, dict) or not isinstance(value.get("references"), list):
        raise ValueError("citation audit has no reference inventory")
    rows = value["references"]
    if any(not isinstance(item, dict) for item in rows):
        raise ValueError("citation audit has a malformed reference inventory")
    return rows


def exact_citation_urls(root: Path) -> set[str]:
    urls = {str(item.get("citation_url", "")) for item in _citation_rows(root)}
    for url in urls:
        parts = urlsplit(url)
        if parts.scheme.casefold() != "https" or not parts.hostname:
            raise ValueError("citation inventory contains a non-HTTPS locator")
    if "" in urls:
        raise ValueError("citation inventory contains an empty locator")
    return urls


_AMD_CITATION_PATH_RE = re.compile(
    r"^/r/(?P<version>[A-Za-z0-9._-]+)-English/"
    r"(?P<document_slug>[A-Za-z0-9._-]+)(?:/[^?#]*)?/$|"
    r"^/r/(?P<version_topic>[A-Za-z0-9._-]+)-English/"
    r"(?P<document_slug_topic>[A-Za-z0-9._-]+)(?:/[^?#]+)?$"
)
_AMD_EVIDENCE_PATH_RE = re.compile(
    r"^/api/khub/maps/(?P<document_id>[A-Za-z0-9_~-]+)/topics/"
    r"(?P<content_id>[A-Za-z0-9_~-]+)/content$"
)
_AMD_MAP_EVIDENCE_PATH_RE = re.compile(
    r"^/api/khub/maps/(?P<document_id>[A-Za-z0-9_~-]+)$"
)
_GITHUB_BLOB_PATH_RE = re.compile(
    r"^/(?P<owner>[A-Za-z0-9_.-]+)/(?P<repository>[A-Za-z0-9_.-]+)/blob/"
    r"(?P<commit>[0-9a-f]{40})/(?P<path>[^?#]+)$"
)
_GITHUB_RAW_PATH_RE = re.compile(
    r"^/(?P<owner>[A-Za-z0-9_.-]+)/(?P<repository>[A-Za-z0-9_.-]+)/"
    r"(?P<commit>[0-9a-f]{40})/(?P<path>[^?#]+)$"
)
_GITHUB_DOCUMENT_SOURCES = {
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
_GITHUB_SECTION_ANCHOR_ALIASES = {
    "Blocks and Regions": "blocks",
}
_AMD_ID_RE = re.compile(r"[A-Za-z0-9_~-]+")
_IDENTITY_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")


def _valid_identity_string(
    value: Any, *, pattern: re.Pattern[str] | None = None,
) -> bool:
    return (
        isinstance(value, str)
        and 1 <= len(value) <= MAX_EVIDENCE_IDENTITY_CHARS
        and _IDENTITY_CONTROL_RE.search(value) is None
        and (pattern is None or pattern.fullmatch(value) is not None)
    )


def _reference_binding(row: dict[str, Any]) -> dict[str, Any]:
    section = row.get("section")
    if section is not None and not isinstance(section, str):
        raise ValueError("citation reference section is not text or null")
    return {
        "reference_id": row.get("reference_id"),
        "reference_kind": row.get("reference_kind"),
        "reference_surface_sha256": row.get("reference_surface_sha256"),
        "document_id": row.get("document_id"),
        "document_version": row.get("document_version"),
        "rule_id": row.get("rule_id"),
        "rule_surface_sha256": row.get("rule_surface_sha256"),
        "section": section,
        "section_sha256": hashlib.sha256(_canonical_json(section)).hexdigest(),
    }


def _github_section_slug(section: str) -> str:
    lowered = section.strip().casefold()
    lowered = re.sub(r"[^a-z0-9\s-]", "", lowered)
    return re.sub(r"[-\s]+", "-", lowered).strip("-")


def _validate_citation_evidence_mapping(
    value: Any, *, citation_audit_sha256: str,
    exact_urls: Iterable[str], expected_references: Iterable[dict[str, Any]],
    expected_fetches: Iterable[dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Validate the closed human-citation to fetched-evidence mapping."""

    if not isinstance(value, dict) or set(value) != {
        "schema_version", "citation_audit_sha256", "entries",
    }:
        raise ValueError("citation evidence mapping is not a closed contract")
    if (value.get("schema_version") != CITATION_EVIDENCE_SCHEMA_VERSION
            or value.get("citation_audit_sha256") != citation_audit_sha256):
        raise ValueError("citation evidence mapping is stale")
    entries = value.get("entries")
    if not isinstance(entries, list):
        raise ValueError("citation evidence mapping has no entry inventory")
    expected_urls = set(exact_urls)
    expected_bindings_by_url: dict[str, list[dict[str, Any]]] = {}
    expected_reference_rows_by_url: dict[str, list[dict[str, Any]]] = {}
    for reference in expected_references:
        if not isinstance(reference, dict):
            raise ValueError("citation evidence mapping reference inventory is malformed")
        reference_url = str(reference.get("citation_url", ""))
        expected_reference_rows_by_url.setdefault(reference_url, []).append(reference)
        expected_bindings_by_url.setdefault(reference_url, []).append(
            _reference_binding(reference),
        )
    for rows in expected_bindings_by_url.values():
        rows.sort(key=lambda row: str(row["reference_id"]))
    fetches_by_url: dict[str, dict[str, Any]] = {}
    for fetched in expected_fetches:
        if (not isinstance(fetched, dict)
                or not isinstance(fetched.get("fetch_url"), str)
                or fetched["fetch_url"] in fetches_by_url):
            raise ValueError("citation audit fetch inventory is malformed")
        fetches_by_url[str(fetched["fetch_url"])] = fetched
    observed: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {
            "citation_url", "evidence_url", "resolver_id", "identity",
            "reference_bindings",
        }:
            raise ValueError("citation evidence mapping has a malformed entry")
        citation_url = entry.get("citation_url")
        evidence_url = entry.get("evidence_url")
        resolver_id = entry.get("resolver_id")
        if (not isinstance(citation_url, str)
                or not isinstance(evidence_url, str)
                or not isinstance(resolver_id, str)
                or citation_url in observed):
            raise ValueError("citation evidence mapping has duplicate or invalid locators")
        if entry.get("reference_bindings") != expected_bindings_by_url.get(
            citation_url, [],
        ):
            raise ValueError("citation evidence mapping reference bindings are stale")
        reference_rows = expected_reference_rows_by_url.get(citation_url, [])
        citation_parts = urlsplit(citation_url)
        evidence_parts = urlsplit(evidence_url)
        citation_host = (citation_parts.hostname or "").casefold()
        evidence_host = (evidence_parts.hostname or "").casefold()
        if (citation_parts.scheme.casefold() != "https" or not citation_host
                or evidence_parts.scheme.casefold() != "https" or not evidence_host
                or citation_parts.port is not None
                or evidence_parts.port is not None
                or citation_parts.username is not None
                or citation_parts.password is not None
                or evidence_parts.username is not None
                or evidence_parts.password is not None):
            raise ValueError("citation evidence mapping leaves same-host HTTPS")
        if citation_host == "docs.amd.com":
            if evidence_host != citation_host:
                raise ValueError("AMD citation evidence leaves same-host HTTPS")
            identity = entry.get("identity")
            citation_match = _AMD_CITATION_PATH_RE.fullmatch(citation_parts.path)
            if citation_match is None:
                raise ValueError("AMD citation URL has an unsupported shape")
            version = citation_match.group("version") or citation_match.group(
                "version_topic"
            )
            document_slug = citation_match.group(
                "document_slug"
            ) or citation_match.group("document_slug_topic")
            is_document_root = citation_parts.path == (
                f"/r/{version}-English/{document_slug}/"
            )
            if is_document_root:
                evidence_match = _AMD_MAP_EVIDENCE_PATH_RE.fullmatch(
                    evidence_parts.path
                )
                if (resolver_id != "amd.docs.khub.map.v1"
                        or not isinstance(identity, dict)
                        or set(identity) != {
                            "publication_id", "document_id", "document_slug",
                            "version", "title",
                        }
                        or evidence_match is None
                        or evidence_parts.query or evidence_parts.fragment
                        or identity.get("version") != version
                        or identity.get("document_slug") != document_slug
                        or identity.get("publication_id")
                        != evidence_match.group("document_id")
                        or any(
                            not _valid_identity_string(
                                identity.get(key), pattern=_AMD_ID_RE,
                            )
                            for key in (
                                "publication_id", "document_id",
                            )
                        )
                        or any(
                            not _valid_identity_string(identity.get(key))
                            for key in ("document_slug", "version", "title")
                        )):
                    raise ValueError("AMD map evidence identity does not close")
            else:
                evidence_match = _AMD_EVIDENCE_PATH_RE.fullmatch(
                    evidence_parts.path
                )
                if (resolver_id != "amd.docs.khub.topic.v1"
                        or not isinstance(identity, dict)
                        or set(identity) != {
                            "publication_id", "document_id", "document_slug",
                            "version", "title", "toc_id", "content_id",
                            "topic_title",
                        }
                        or evidence_match is None
                        or evidence_parts.fragment
                        or parse_qsl(
                            evidence_parts.query, keep_blank_values=True,
                        ) != [("target", "DESIGNED_READER")]
                        or any(
                            not _valid_identity_string(
                                identity.get(key), pattern=_AMD_ID_RE,
                            )
                            for key in (
                                "publication_id", "document_id", "toc_id", "content_id",
                            )
                        )
                        or any(
                            not _valid_identity_string(identity.get(key))
                            for key in (
                                "document_slug", "version", "title", "topic_title",
                            )
                        )
                        or identity["version"] != version
                        or identity["document_slug"] != document_slug
                        or identity["publication_id"]
                        != evidence_match.group("document_id")
                        or identity["content_id"]
                        != evidence_match.group("content_id")):
                    raise ValueError("AMD topic evidence identity does not close")
                rule_sections = {
                    row.get("section") for row in reference_rows
                    if row.get("reference_kind") == "rule"
                }
                if rule_sections and rule_sections != {identity["topic_title"]}:
                    raise ValueError(
                        "AMD topic evidence does not bind the declared rule section",
                    )
            if citation_parts.query or citation_parts.fragment:
                raise ValueError("AMD human citation must be an exact clean locator")
        elif resolver_id == "github.raw.document.v1":
            identity = entry.get("identity")
            evidence_match = _GITHUB_RAW_PATH_RE.fullmatch(evidence_parts.path)
            canonical_source = _GITHUB_DOCUMENT_SOURCES.get(citation_url)
            document_rows = [
                row for row in reference_rows
                if row.get("reference_kind") == "document"
            ]
            if (evidence_host != "raw.githubusercontent.com"
                    or evidence_match is None
                    or citation_parts.query or citation_parts.fragment
                    or evidence_parts.query or evidence_parts.fragment
                    or not isinstance(identity, dict)
                    or set(identity) != {
                        "repository", "commit", "path", "source_sha256",
                        "source_size", "document_id", "document_version",
                    }
                    or canonical_source != (
                        identity.get("repository"), identity.get("path"),
                    )
                    or len(document_rows) != 1
                    or identity.get("document_id")
                    != document_rows[0].get("document_id")
                    or identity.get("document_version")
                    != document_rows[0].get("document_version")
                    or identity.get("document_version")
                    != f"git-{identity.get('commit')}"
                    or evidence_match.group("owner") + "/"
                    + evidence_match.group("repository")
                    != identity.get("repository")
                    or evidence_match.group("commit") != identity.get("commit")
                    or evidence_match.group("path") != identity.get("path")
                    or _SHA256_RE.fullmatch(
                        str(identity.get("source_sha256", "")),
                    ) is None
                    or type(identity.get("source_size")) is not int
                    or identity["source_size"] <= 0
                    or not _valid_identity_string(identity.get("repository"))
                    or not _valid_identity_string(identity.get("commit"))
                    or not _valid_identity_string(identity.get("path"))):
                raise ValueError("GitHub raw document identity does not close")
            path = PurePosixPath(str(identity["path"]))
            if (path.is_absolute()
                    or any(part in {"", ".", ".."} for part in path.parts)
                    or evidence_url != (
                        "https://raw.githubusercontent.com/"
                        f"{identity['repository']}/{identity['commit']}/"
                        f"{identity['path']}"
                    )):
                raise ValueError("GitHub raw document path is non-canonical")
        elif resolver_id == "github.raw.lines.v1":
            identity = entry.get("identity")
            citation_match = _GITHUB_BLOB_PATH_RE.fullmatch(citation_parts.path)
            evidence_match = _GITHUB_RAW_PATH_RE.fullmatch(evidence_parts.path)
            if (citation_host != "github.com"
                    or evidence_host != "raw.githubusercontent.com"
                    or citation_match is None or evidence_match is None
                    or citation_parts.query or evidence_parts.query
                    or evidence_parts.fragment
                    or not isinstance(identity, dict)
                    or set(identity) != {
                        "repository", "commit", "path", "source_sha256",
                        "start_line", "end_line", "slice_sha256",
                    }
                    or identity.get("repository") != (
                        citation_match.group("owner") + "/"
                        + citation_match.group("repository")
                    )
                    or identity.get("commit") != citation_match.group("commit")
                    or identity.get("path") != citation_match.group("path")
                    or evidence_match.group("owner")
                    != citation_match.group("owner")
                    or evidence_match.group("repository")
                    != citation_match.group("repository")
                    or evidence_match.group("commit")
                    != citation_match.group("commit")
                    or evidence_match.group("path")
                    != citation_match.group("path")
                    or not _valid_identity_string(identity.get("repository"))
                    or not _valid_identity_string(identity.get("commit"))
                    or not _valid_identity_string(identity.get("path"))
                    or _SHA256_RE.fullmatch(
                        str(identity.get("source_sha256", "")),
                    ) is None
                    or _SHA256_RE.fullmatch(
                        str(identity.get("slice_sha256", "")),
                    ) is None
                    or type(identity.get("start_line")) is not int
                    or type(identity.get("end_line")) is not int
                    or not 1 <= identity["start_line"] <= identity["end_line"]
                    or identity["end_line"] - identity["start_line"] + 1
                    > MAX_EVIDENCE_LINE_RANGE):
                raise ValueError("GitHub raw line evidence identity does not close")
            path = PurePosixPath(str(identity["path"]))
            if (path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts)
                    or evidence_url != (
                        "https://raw.githubusercontent.com/"
                        f"{identity['repository']}/{identity['commit']}/"
                        f"{identity['path']}"
                    )):
                raise ValueError("GitHub raw line evidence path is non-canonical")
            rule_sections = [
                str(row["section"]) for row in reference_rows
                if row.get("reference_kind") == "rule"
                and isinstance(row.get("section"), str)
            ]
            if not rule_sections:
                raise ValueError("GitHub raw line evidence has no bound rule reference")
            if (citation_parts.fragment
                    and citation_parts.fragment not in {
                        _github_section_slug(section) for section in rule_sections
                    } | {
                        _GITHUB_SECTION_ANCHOR_ALIASES[section]
                        for section in rule_sections
                        if section in _GITHUB_SECTION_ANCHOR_ALIASES
                    }):
                raise ValueError(
                    "GitHub raw line evidence fragment does not bind a rule section",
                )
        elif resolver_id == "direct.sha256.v1":
            identity = entry.get("identity")
            audited_fetch = fetches_by_url.get(citation_url)
            document_rows = [
                row for row in reference_rows
                if row.get("reference_kind") == "document"
            ]
            if (citation_host != evidence_host
                    or evidence_url != citation_url
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
                    or not isinstance(audited_fetch, dict)
                    or audited_fetch.get("status") != 200
                    or audited_fetch.get("final_url") != citation_url
                    or audited_fetch.get("sha256") != identity.get("body_sha256")
                    or audited_fetch.get("byte_count") != identity.get("body_size")
                    or audited_fetch.get("content_type")
                    != identity.get("content_type")):
                raise ValueError("direct SHA-256 document identity does not close")
        else:
            raise ValueError("citation must use an approved exact resolver")
        observed[citation_url] = entry
    if set(observed) != expected_urls or len(entries) != len(expected_urls):
        raise ValueError("citation evidence mapping inventory is incomplete or has extras")
    if entries != [observed[url] for url in sorted(observed)]:
        raise ValueError("citation evidence mapping is not uniquely sorted")
    return tuple(observed[url] for url in sorted(observed))


def _citation_evidence_rows(snapshot: ReviewSnapshot) -> tuple[dict[str, Any], ...]:
    mapping_item = snapshot.file_map[CITATION_EVIDENCE_PATH]
    value = _strict_json_bytes(mapping_item.payload, label="citation evidence mapping")
    citation_item = snapshot.file_map[CITATION_AUDIT_PATH]
    citation = _strict_json_bytes(citation_item.payload, label="citation audit")
    if not isinstance(citation, dict):
        raise ValueError("citation audit root is not an object")
    return _validate_citation_evidence_mapping(
        value, citation_audit_sha256=snapshot.citation_audit_sha256,
        exact_urls=snapshot.exact_citation_urls,
        expected_references=_citation_reference_rows(snapshot),
        expected_fetches=citation.get("fetches", []),
    )


def _semantic_pack_projection(data: bytes, *, label: str) -> tuple[str, str, bytes]:
    value = _strict_json_bytes(data, label=label)
    if not isinstance(value, dict):
        raise ValueError(f"{label} root must be an object")
    try:
        from tools import knowledge_review_surface
        semantic = knowledge_review_surface.semantic_surface(value)
    except (ImportError, TypeError, ValueError) as exc:
        raise ValueError(f"cannot derive {label} semantic surface: {exc}") from exc
    pack_id = semantic.get("pack_id")
    if not isinstance(pack_id, str) or not pack_id:
        raise ValueError(f"{label} has no pack_id")
    compact = json.dumps(
        semantic, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return pack_id, hashlib.sha256(compact).hexdigest(), _canonical_json(semantic)


def _implementation_hash(files: Iterable[ReviewFileSnapshot]) -> str:
    digest = hashlib.sha256()
    implementation = sorted(
        (item for item in files
         if item.path.startswith("src/hlsgraph/") and item.path.endswith(".py")),
        key=lambda item: item.path,
    )
    for item in implementation:
        digest.update(item.path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(item.payload).digest())
        digest.update(b"\0")
    return digest.hexdigest()


def freeze_review_snapshot(root: Path, protocol_id: str) -> ReviewSnapshot:
    """Read the complete review surface once into an immutable snapshot."""

    if protocol_id not in PROTOCOLS:
        raise ValueError(f"unknown knowledge-review protocol: {protocol_id}")
    root = root.resolve(strict=True)
    snapshots: list[ReviewFileSnapshot] = []
    surfaces: dict[str, str] = {}
    for relative in sorted(required_read_paths(root, protocol_id)):
        path = root / PurePosixPath(relative)
        data = path.read_bytes()
        if relative.startswith("src/hlsgraph/knowledge/packs/"):
            pack_id, digest, cache_payload = _semantic_pack_projection(
                data, label=f"knowledge pack {relative}",
            )
            if pack_id in surfaces:
                raise ValueError(f"duplicate reviewed pack ID: {pack_id}")
            surfaces[pack_id] = digest
            hash_kind = "review_surface_sha256"
        else:
            digest = hashlib.sha256(data).hexdigest()
            cache_payload = data
            hash_kind = "raw_sha256"
        snapshots.append(ReviewFileSnapshot(
            path=relative, hash_kind=hash_kind, sha256=digest,
            cache_sha256=hashlib.sha256(cache_payload).hexdigest(),
            payload=cache_payload,
        ))
    try:
        from tools import knowledge_review_shards
        source_payloads = {
            item.path: item.payload for item in snapshots
            if item.path in {
                knowledge_review_shards.CITATION_AUDIT_SOURCE_PATH,
                knowledge_review_shards.CITATION_EVIDENCE_SOURCE_PATH,
                knowledge_review_shards.AMD_PACK_SOURCE_PATH,
                knowledge_review_shards.AXI_PACK_SOURCE_PATH,
                knowledge_review_shards.OPEN_IR_PACK_SOURCE_PATH,
            }
        }
        virtual_sources = knowledge_review_shards.build_model_source_projections(
            source_payloads,
        )
    except (ImportError, TypeError, ValueError) as exc:
        raise ValueError(f"cannot derive shard-local model sources: {exc}") from exc
    existing_paths = {item.path for item in snapshots}
    if existing_paths & set(virtual_sources):
        raise ValueError("shard-local model source collides with a checkout path")
    for relative, payload in virtual_sources.items():
        digest = hashlib.sha256(payload).hexdigest()
        snapshots.append(ReviewFileSnapshot(
            path=relative, hash_kind="raw_sha256", sha256=digest,
            cache_sha256=digest, payload=payload,
        ))
    snapshots.sort(key=lambda item: item.path)
    by_path = {item.path: item for item in snapshots}
    citation = by_path.get(CITATION_AUDIT_PATH)
    citation_evidence = by_path.get(CITATION_EVIDENCE_PATH)
    schema = by_path.get(REVIEW_SCHEMA_PATH)
    receipt_schema = by_path.get(REVIEW_RECEIPT_SCHEMA_PATH)
    if (citation is None or citation_evidence is None or schema is None
            or receipt_schema is None):
        raise ValueError("review snapshot omits a required schema or citation manifest")
    citation_value = _strict_json_bytes(citation.payload, label="citation audit")
    if not isinstance(citation_value, dict) or not isinstance(
        citation_value.get("references"), list
    ):
        raise ValueError("citation audit has no reference inventory")
    urls: set[str] = set()
    for row in citation_value["references"]:
        if not isinstance(row, dict):
            raise ValueError("citation audit has a malformed reference inventory")
        url = row.get("citation_url")
        if not isinstance(url, str) or not url:
            raise ValueError("citation audit contains an empty locator")
        parts = urlsplit(url)
        if parts.scheme.casefold() != "https" or not parts.hostname:
            raise ValueError("citation audit contains a non-HTTPS locator")
        urls.add(url)
    snapshot = ReviewSnapshot(
        protocol_id=protocol_id, files=tuple(snapshots),
        review_surface_sha256=tuple(sorted(surfaces.items())),
        implementation_surface_sha256=_implementation_hash(snapshots),
        citation_audit_sha256=citation.sha256,
        citation_evidence_sha256=citation_evidence.sha256,
        output_schema_sha256=schema.sha256,
        receipt_schema_sha256=receipt_schema.sha256,
        exact_citation_urls=tuple(sorted(urls)),
    )
    if not snapshots or not surfaces:
        raise ValueError("review snapshot has an empty implementation or pack inventory")
    _citation_evidence_rows(snapshot)
    return snapshot


class _SameHostRedirectHandler(HTTPRedirectHandler):
    def __init__(self, requested_url: str, max_redirects: int) -> None:
        super().__init__()
        self.requested_url = requested_url
        self.expected_host = (urlsplit(requested_url).hostname or "").casefold()
        self.max_redirects = max_redirects
        self.chain: list[str] = [requested_url]

    def redirect_request(  # type: ignore[override]
        self, req: Request, fp: Any, code: int, msg: str,
        headers: Any, newurl: str,
    ) -> Request | None:
        resolved = urljoin(req.full_url, newurl)
        parts = urlsplit(resolved)
        if (parts.scheme.casefold() != "https" or not parts.hostname
                or parts.hostname.casefold() != self.expected_host):
            raise ValueError("citation redirect leaves the exact same-host HTTPS boundary")
        if resolved != self.requested_url:
            raise ValueError("citation redirect changes the exact evidence locator")
        if len(self.chain) > self.max_redirects:
            raise ValueError("citation redirect chain exceeds the fixed maximum")
        self.chain.append(resolved)
        return super().redirect_request(req, fp, code, msg, headers, resolved)


def _default_fetch(
    url: str, timeout_seconds: float, max_bytes: int,
) -> TrustedFetch:
    last_error: BaseException | None = None
    for _attempt in range(MAX_FETCH_ATTEMPTS):
        handler = _SameHostRedirectHandler(url, MAX_REDIRECTS)
        opener = build_opener(handler)
        request = Request(
            url, headers={"User-Agent": "hlsgraph-knowledge-review/1"},
        )
        try:
            with opener.open(request, timeout=timeout_seconds) as response:
                status_code = int(
                    getattr(response, "status", response.getcode()),
                )
                final_url = str(response.geturl())
                body = response.read(max_bytes + 1)
                content_type = str(response.headers.get_content_type())
                charset = response.headers.get_content_charset()
                get_header = getattr(response.headers, "get", None)
                declared_length_text = (
                    get_header("Content-Length") if callable(get_header) else None
                )
        except (HTTPError, URLError, HTTPException, OSError) as exc:
            # Transient TLS/socket/chunk failures are common on large public
            # manuals.  Retry only the exact same locator with a fresh opener;
            # redirect, host, byte-limit and content checks remain unchanged.
            last_error = exc
            continue
        declared_length: int | None = None
        if declared_length_text is not None:
            try:
                declared_length = int(str(declared_length_text), 10)
            except ValueError:
                raise ValueError("exact citation has an invalid Content-Length")
            if declared_length < 0:
                raise ValueError("exact citation has an invalid Content-Length")
            if declared_length > max_bytes:
                raise ValueError("exact citation response exceeds the fixed byte limit")
            if len(body) != declared_length:
                last_error = HTTPException("incomplete exact citation response")
                continue
        if len(body) > max_bytes:
            raise ValueError("exact citation response exceeds the fixed byte limit")
        if not body:
            last_error = HTTPException("empty exact citation response")
            continue
        if (body.startswith(b"%PDF-")
                or content_type.casefold() == "application/pdf") and (
            not body.startswith(b"%PDF-")
            or body.rfind(b"%%EOF") < max(0, len(body) - 4096)
        ):
            last_error = HTTPException("incomplete exact PDF response")
            continue
        final_parts = urlsplit(final_url)
        expected_host = (urlsplit(url).hostname or "").casefold()
        if (status_code != 200
                or final_parts.scheme.casefold() != "https"
                or (final_parts.hostname or "").casefold() != expected_host):
            raise ValueError("exact citation fetch did not finish at same-host HTTPS")
        chain = list(handler.chain)
        if not chain or chain[-1] != final_url:
            chain.append(final_url)
        for item in chain:
            parts = urlsplit(item)
            if (parts.scheme.casefold() != "https"
                    or (parts.hostname or "").casefold() != expected_host):
                raise ValueError("citation redirect chain is not same-host HTTPS")
        return TrustedFetch(
            status=status_code, final_url=final_url,
            redirect_chain=tuple(chain), content_type=content_type,
            charset=charset, body=body, content_length=declared_length,
        )
    assert last_error is not None
    raise ValueError(
        f"exact citation fetch failed after {MAX_FETCH_ATTEMPTS} attempts: "
        f"{type(last_error).__name__}"
    ) from last_error


def _mkdir_private(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=False, mode=0o700)
    if os.name != "nt":
        path.chmod(0o700)


def _ensure_private_parent(path: Path) -> None:
    if not path.exists():
        _mkdir_private(path)
    if not path.is_dir() or path.is_symlink():
        raise RuntimeError(f"restricted evidence parent is not a plain directory: {path}")
    if os.name != "nt":
        mode = stat.S_IMODE(path.stat().st_mode)
        if mode != 0o700:
            raise RuntimeError(f"restricted evidence directory must be mode 0700: {path}")


def _write_private(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name != "nt":
        path.parent.chmod(0o700)
    flags = (
        os.O_WRONLY | os.O_CREAT | os.O_EXCL
        | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short private evidence write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if os.name != "nt":
        path.chmod(0o600)


def _harden_private_tree(root: Path) -> None:
    """Set the final exact modes before a cache becomes review evidence."""

    if os.name == "nt":
        return
    for current, directories, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        current_path.chmod(CACHE_DIRECTORY_MODE)
        for name in directories:
            path = current_path / name
            if path.is_symlink():
                raise RuntimeError(f"private evidence tree contains a symlink: {path}")
            path.chmod(CACHE_DIRECTORY_MODE)
        for name in filenames:
            path = current_path / name
            if path.is_symlink():
                raise RuntimeError(f"private evidence tree contains a symlink: {path}")
            path.chmod(CACHE_FILE_MODE)


def _utf8_chunks(payload: bytes) -> tuple[tuple[int, int, bytes], ...]:
    """Split UTF-8 bytes without cutting a code point or exceeding the cap."""

    payload.decode("utf-8", errors="strict")
    if not payload:
        return ((0, 0, b""),)
    chunks: list[tuple[int, int, bytes]] = []
    start = 0
    while start < len(payload):
        end = min(start + MAX_REVIEW_CHUNK_BYTES, len(payload))
        while end < len(payload) and end > start and payload[end] & 0xC0 == 0x80:
            end -= 1
        if end == start:
            raise ValueError("UTF-8 review chunk limit cannot contain one code point")
        chunk = payload[start:end]
        chunk.decode("utf-8", errors="strict")
        chunks.append((start, end, chunk))
        start = end
    return tuple(chunks)


def _chunk_rows(
    *, origin_kind: str, origin_id: str, original_sha256: str, payload: bytes,
) -> list[dict[str, Any]]:
    if origin_kind not in {"source", "citation"}:
        raise ValueError("unsupported review chunk origin")
    identity = hashlib.sha256(origin_id.encode("utf-8")).hexdigest()
    rows: list[dict[str, Any]] = []
    for index, (start, end, chunk) in enumerate(_utf8_chunks(payload)):
        digest = hashlib.sha256(chunk).hexdigest()
        relative = (
            f"chunks/{origin_kind}/{identity}/"
            f"{index:06d}-{digest}.utf8"
        )
        rows.append({
            "index": index,
            "path": relative,
            "sha256": digest,
            "size": len(chunk),
            "byte_start": start,
            "byte_end": end,
            "original_sha256": original_sha256,
            "original_size": len(payload),
        })
    return rows


def _chunk_inventory(
    cache_root: Path, *, origin_kind: str, origin_id: str,
    original_sha256: str, payload: bytes,
) -> list[dict[str, Any]]:
    """Write deterministic content-addressed chunks and return reconstruction rows."""

    rows = _chunk_rows(
        origin_kind=origin_kind, origin_id=origin_id,
        original_sha256=original_sha256, payload=payload,
    )
    chunks = _utf8_chunks(payload)
    for row, (_start, _end, chunk) in zip(rows, chunks, strict=True):
        _write_private(cache_root / PurePosixPath(row["path"]), chunk)
    return rows


def _is_link_like(path: Path) -> bool:
    if path.is_symlink():
        return True
    isjunction = getattr(os.path, "isjunction", None)
    return bool(isjunction is not None and isjunction(path))


def _formal_host_is_windows() -> bool:
    """Test seam; production truth is the interpreter host."""

    return os.name == "nt"


def _resolved_unlinked_path(path: Path, *, label: str) -> Path:
    """Resolve a prospective path while rejecting every existing link alias."""
    lexical = path.absolute()
    for component in (lexical, *lexical.parents):
        if _is_link_like(component):
            raise RuntimeError(f"{label} has a linked path component: {component}")
    return lexical.resolve(strict=False)


def _assert_private_mode(path: Path, expected: int, *, label: str) -> None:
    if os.name == "nt":
        return
    actual = stat.S_IMODE(path.stat(follow_symlinks=False).st_mode)
    if actual != expected:
        raise ValueError(
            f"{label} must be mode {expected:04o}, found {actual:04o}: {path}"
        )


def _restricted_identity(value: os.stat_result) -> tuple[int, ...]:
    """Return every metadata field that must remain stable while reading."""

    return (
        value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns,
        value.st_ctime_ns, value.st_mode, value.st_nlink,
        int(getattr(value, "st_uid", -1)),
    )


def _read_stable_restricted_file(
    path: Path, *, label: str, file_mode: int, parent_mode: int,
    max_bytes: int,
) -> bytes:
    """Read one caller-owned file while detecting aliases and path replacement.

    Every lexical ancestor is checked for symlink/junction redirection before
    and after the descriptor read.  The immediate parent and file identities
    are also compared across the read, including ctime, so replacing a parent
    directory or relinking a retained evidence file fails closed.
    """

    lexical = path.absolute()
    try:
        resolved_before = _resolved_unlinked_path(
            lexical, label=label,
        ).resolve(strict=True)
        parent = lexical.parent
        parent_before = parent.lstat()
        path_before = lexical.lstat()
    except OSError:
        raise ValueError(f"{label} is missing, linked, or unreadable") from None
    if (not stat.S_ISDIR(parent_before.st_mode)
            or not stat.S_ISREG(path_before.st_mode)):
        raise ValueError(f"{label} is not a plain file in a plain directory")
    if path_before.st_nlink != 1:
        raise ValueError(f"{label} has hard-link aliases")
    if path_before.st_size > max_bytes:
        raise ValueError(f"{label} exceeds its fixed byte limit")
    if os.name != "nt":
        uid = os.geteuid()
        if (parent_before.st_uid != uid or path_before.st_uid != uid
                or stat.S_IMODE(parent_before.st_mode) != parent_mode
                or stat.S_IMODE(path_before.st_mode) != file_mode):
            raise ValueError(f"{label} violates its owner or mode contract")

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(lexical, flags)
    except OSError:
        raise ValueError(f"{label} could not be opened without following links") from None
    try:
        opened_before = os.fstat(descriptor)
        if (not stat.S_ISREG(opened_before.st_mode)
                or opened_before.st_nlink != 1
                or (opened_before.st_dev, opened_before.st_ino)
                != (path_before.st_dev, path_before.st_ino)):
            raise ValueError(f"{label} changed before its stable read")
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, max_bytes + 1 - size))
            if not chunk:
                break
            size += len(chunk)
            if size > max_bytes:
                raise ValueError(f"{label} exceeds its fixed byte limit")
            chunks.append(chunk)
        opened_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)

    try:
        path_after = lexical.lstat()
        parent_after = parent.lstat()
        resolved_after = _resolved_unlinked_path(
            lexical, label=label,
        ).resolve(strict=True)
    except OSError:
        raise ValueError(f"{label} path changed during its stable read") from None
    if (resolved_before != resolved_after
            or _restricted_identity(path_before)
            != _restricted_identity(path_after)
            or _restricted_identity(opened_before)
            != _restricted_identity(opened_after)
            or _restricted_identity(parent_before)
            != _restricted_identity(parent_after)
            or (path_after.st_dev, path_after.st_ino)
            != (opened_after.st_dev, opened_after.st_ino)
            or size != opened_before.st_size):
        raise ValueError(f"{label} or its parent changed during its stable read")
    return b"".join(chunks)


def _read_private_cache_file(root: Path, relative: str) -> bytes:
    relative_path = _safe_relative(relative)
    current = root
    for index, part in enumerate(relative_path.parts):
        current = current / part
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise ValueError(f"review cache path is missing: {relative}") from exc
        if _is_link_like(current):
            raise ValueError(f"review cache contains a linked path: {relative}")
        if index < len(relative_path.parts) - 1:
            if not stat.S_ISDIR(metadata.st_mode):
                raise ValueError(f"review cache parent is not a directory: {relative}")
            _assert_private_mode(
                current, CACHE_DIRECTORY_MODE, label="review cache directory",
            )
        else:
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError(f"review cache entry is not a plain file: {relative}")
            if metadata.st_nlink != 1:
                raise ValueError(f"review cache entry has aliases: {relative}")
            _assert_private_mode(
                current, CACHE_FILE_MODE, label="review cache file",
            )
    return _read_stable_restricted_file(
        current, label="review cache entry",
        file_mode=CACHE_FILE_MODE, parent_mode=CACHE_DIRECTORY_MODE,
        max_bytes=MAX_CITATION_BYTES,
    )


def _text_derivation(fetch: TrustedFetch) -> TextDerivation | None:
    if fetch.body.startswith(b"%PDF-") or fetch.content_type.casefold() == "application/pdf":
        return None
    charset = (fetch.charset or "utf-8").casefold()
    try:
        text = fetch.body.decode(charset, errors="strict")
    except (LookupError, UnicodeDecodeError):
        return None
    encoded = text.encode("utf-8")
    if not text.strip():
        return None
    if _is_portal_javascript_shell(text, fetch.content_type):
        return None
    contract = {
        "parser_id": "hlsgraph.review.utf8-text.v1",
        "parser_version": "identity-v1",
        "charset": charset,
    }
    return TextDerivation(
        text=encoded, parser_id=contract["parser_id"],
        parser_version=contract["parser_version"],
        command_sha256=hashlib.sha256(_canonical_json(contract)).hexdigest(),
    )


def _github_line_range_derivation(
    mapping: dict[str, Any], fetch: TrustedFetch,
) -> TextDerivation:
    """Select one audited line range from an exact commit-pinned raw file."""

    identity = mapping.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("GitHub raw evidence lacks its closed identity")
    source_sha256 = hashlib.sha256(fetch.body).hexdigest()
    if source_sha256 != identity.get("source_sha256"):
        raise ValueError("GitHub raw evidence source SHA-256 differs from the mapping")
    try:
        fetch.body.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("GitHub raw evidence is not strict UTF-8") from exc
    lines = fetch.body.splitlines(keepends=True)
    start = int(identity["start_line"])
    end = int(identity["end_line"])
    if end > len(lines):
        raise ValueError("GitHub raw evidence line range exceeds the pinned source")
    selected = b"".join(lines[start - 1:end])
    if (not selected.strip()
            or hashlib.sha256(selected).hexdigest()
            != identity.get("slice_sha256")):
        raise ValueError("GitHub raw evidence selected range differs from the mapping")
    contract = {
        "parser_id": "hlsgraph.review.github-raw-lines.v1",
        "parser_version": "exact-commit-line-range-v1",
        "repository": identity["repository"],
        "commit": identity["commit"],
        "path": identity["path"],
        "source_sha256": identity["source_sha256"],
        "start_line": start,
        "end_line": end,
        "slice_sha256": identity["slice_sha256"],
    }
    return TextDerivation(
        text=selected, parser_id=contract["parser_id"],
        parser_version=contract["parser_version"],
        command_sha256=hashlib.sha256(_canonical_json(contract)).hexdigest(),
    )


def _validate_document_identity_body(
    mapping: dict[str, Any], fetch: TrustedFetch,
) -> None:
    """Verify document-only bytes against their explicit immutable identity."""

    identity = mapping.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("document evidence lacks its closed identity")
    resolver_id = mapping.get("resolver_id")
    body_sha256 = hashlib.sha256(fetch.body).hexdigest()
    if resolver_id == "direct.sha256.v1":
        if (body_sha256 != identity.get("body_sha256")
                or len(fetch.body) != identity.get("body_size")
                or fetch.content_type.casefold()
                != str(identity.get("content_type", "")).casefold()):
            raise ValueError("direct document bytes differ from the pinned identity")
        return
    if resolver_id == "github.raw.document.v1":
        if (body_sha256 != identity.get("source_sha256")
                or len(fetch.body) != identity.get("source_size")):
            raise ValueError("GitHub document bytes differ from the pinned source")
        try:
            fetch.body.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise ValueError("GitHub document source is not strict UTF-8") from exc
        return
    raise ValueError("document evidence uses a non-document resolver")


def _is_portal_javascript_shell(text: str, content_type: str) -> bool:
    """Reject deterministic app shells that contain no inspectable document body."""

    lowered = text.casefold()
    if "html" not in content_type.casefold() and "<html" not in lowered:
        return False
    has_script = "<script" in lowered
    empty_mount = re.search(
        r"<(?:div|main)[^>]+id=[\"'](?:root|app)[\"'][^>]*>\s*</(?:div|main)>",
        lowered,
    ) is not None or "<app-root" in lowered
    scripts_removed = re.sub(r"(?is)<script\b[^>]*>.*?</script>", " ", text)
    styles_removed = re.sub(r"(?is)<style\b[^>]*>.*?</style>", " ", scripts_removed)
    visible = re.sub(r"(?s)<[^>]+>", " ", styles_removed)
    visible = re.sub(r"\s+", " ", visible).strip().casefold()
    javascript_notice = (
        "enable javascript" in visible or "javascript is required" in visible
    )
    return has_script and empty_mount and (javascript_notice or len(visible) < 256)


def _stop_bounded_process(process: subprocess.Popen[bytes]) -> None:
    """Terminate and then kill a parser process without surfacing OS text."""

    try:
        process.terminate()
    except BaseException:
        pass
    try:
        process.wait(timeout=1.0)
        return
    except BaseException:
        pass
    try:
        process.kill()
    except BaseException:
        pass
    try:
        process.wait(timeout=2.0)
    except BaseException:
        pass


def _bounded_process_output(
    argv: Sequence[str], *, env: dict[str, str], timeout: float,
    stdout_limit: int, stderr_limit: int,
) -> tuple[int, bytes, bytes]:
    """Drain both child streams concurrently under strict in-memory limits."""

    if (stdout_limit < 1 or stderr_limit < 1 or timeout <= 0
            or not argv or any(not isinstance(item, str) or not item for item in argv)):
        raise ValueError("controlled parser process contract is invalid")
    try:
        process = subprocess.Popen(
            list(argv), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL, env=env,
        )
    except BaseException:
        raise ValueError("controlled parser process could not start") from None
    if process.stdout is None or process.stderr is None:
        _stop_bounded_process(process)
        raise ValueError("controlled parser pipes could not be established")

    overflow = threading.Event()
    read_failure = threading.Event()
    stdout = bytearray()
    stderr = bytearray()

    def drain(stream: Any, destination: bytearray, limit: int) -> None:
        try:
            while True:
                chunk = stream.read(64 * 1024)
                if not chunk:
                    return
                remaining = limit - len(destination)
                if remaining <= 0 or len(chunk) > remaining:
                    if remaining > 0:
                        destination.extend(chunk[:remaining])
                    overflow.set()
                    return
                destination.extend(chunk)
        except BaseException:
            read_failure.set()

    threads = [
        threading.Thread(
            target=drain, args=(process.stdout, stdout, stdout_limit), daemon=True,
        ),
        threading.Thread(
            target=drain, args=(process.stderr, stderr, stderr_limit), daemon=True,
        ),
    ]
    for thread in threads:
        thread.start()
    deadline = time.monotonic() + timeout
    timed_out = False
    try:
        while process.poll() is None:
            if overflow.is_set() or read_failure.is_set():
                _stop_bounded_process(process)
                break
            if time.monotonic() >= deadline:
                timed_out = True
                _stop_bounded_process(process)
                break
            overflow.wait(0.02)
        for thread in threads:
            thread.join(timeout=2.0)
        if any(thread.is_alive() for thread in threads):
            _stop_bounded_process(process)
            raise ValueError("controlled parser streams did not close")
        if timed_out:
            raise ValueError("controlled parser process timed out")
        if overflow.is_set():
            raise ValueError("controlled parser output exceeded its fixed byte limit")
        if read_failure.is_set():
            raise ValueError("controlled parser stream read failed")
        return int(process.returncode), bytes(stdout), bytes(stderr)
    finally:
        try:
            process.stdout.close()
            process.stderr.close()
        except BaseException:
            pass
        if process.poll() is None:
            _stop_bounded_process(process)


def _pdftotext_derivation(
    body_path: Path, command: str | None, expected_sha256: str | None,
) -> TextDerivation | None:
    if not command:
        return None
    if (not Path(command).is_absolute()
            or Path(command).as_posix() != PDFTOTEXT_ALLOWED_PATH):
        raise ValueError("pdftotext must use the fixed absolute /usr/bin path")
    if expected_sha256 is None or _SHA256_RE.fullmatch(expected_sha256) is None:
        raise ValueError("pdftotext requires an explicit expected SHA-256")
    executable = Path(command).resolve(strict=True)
    if executable.as_posix() != PDFTOTEXT_ALLOWED_PATH or _is_link_like(executable):
        raise ValueError("pdftotext executable path is not the fixed plain binary")
    binary_sha256 = hashlib.sha256(executable.read_bytes()).hexdigest()
    if binary_sha256 != expected_sha256:
        raise ValueError("pdftotext executable SHA-256 differs from the declared contract")
    minimal_env = {
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/bin:/bin",
    }
    version_code, version_stdout, version_stderr = _bounded_process_output(
        [str(executable), "-v"], env=minimal_env, timeout=15,
        stdout_limit=MAX_PARSER_VERSION_BYTES,
        stderr_limit=MAX_PARSER_VERSION_BYTES,
    )
    version_bytes = version_stdout + version_stderr
    version_text = version_bytes.decode("utf-8", errors="strict").strip()
    if version_code != 0 or not version_text:
        return None
    returncode, parser_stdout, _parser_stderr = _bounded_process_output(
        [str(executable), "-layout", str(body_path), "-"],
        env=minimal_env, timeout=120,
        stdout_limit=MAX_PDF_TEXT_BYTES,
        stderr_limit=MAX_PARSER_STDERR_BYTES,
    )
    if returncode != 0:
        return None
    try:
        text = parser_stdout.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return None
    if not text.strip():
        return None
    contract = {
        "parser_id": "hlsgraph.review.pdftotext.v1",
        "parser_version": "poppler-version-output-sha256-v1",
        "binary_path": PDFTOTEXT_ALLOWED_PATH,
        "binary_sha256": binary_sha256,
        "version_output_sha256": hashlib.sha256(version_bytes).hexdigest(),
        "argv": [PDFTOTEXT_ALLOWED_PATH, "-layout", "$INPUT", "-"],
        "environment": minimal_env,
    }
    return TextDerivation(
        text=text.encode("utf-8"), parser_id=contract["parser_id"],
        parser_version=contract["parser_version"],
        command_sha256=hashlib.sha256(_canonical_json(contract)).hexdigest(),
        executable_sha256=binary_sha256,
        version_output_sha256=contract["version_output_sha256"],
    )


def _citation_reference_rows(snapshot: ReviewSnapshot) -> list[dict[str, Any]]:
    item = snapshot.file_map[CITATION_AUDIT_PATH]
    value = _strict_json_bytes(item.payload, label="citation audit")
    rows = value.get("references") if isinstance(value, dict) else None
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("citation audit has a malformed reference inventory")
    return rows


def _checked_fetch(
    fetcher: Callable[[str, float, int], TrustedFetch], url: str,
    timeout_seconds: float, max_bytes: int,
) -> TrustedFetch:
    fetched = fetcher(url, timeout_seconds, max_bytes)
    if not isinstance(fetched, TrustedFetch):
        raise TypeError("fetcher did not return a TrustedFetch")
    if (not isinstance(fetched.body, bytes) or not fetched.body
            or len(fetched.body) > max_bytes):
        raise ValueError("fetcher returned an invalid or oversized body")
    if not isinstance(fetched.content_type, str):
        raise TypeError("fetcher returned an invalid content type")
    if type(fetched.status) is not int or fetched.status != 200:
        raise ValueError("non-success citation status")
    if (fetched.content_length is not None
            and (type(fetched.content_length) is not int
                 or fetched.content_length != len(fetched.body))):
        raise ValueError("fetcher returned an incomplete declared body")
    if (fetched.body.startswith(b"%PDF-")
            or fetched.content_type.casefold() == "application/pdf"):
        if (not fetched.body.startswith(b"%PDF-")
                or fetched.body.rfind(b"%%EOF")
                < max(0, len(fetched.body) - 4096)):
            raise ValueError("fetcher returned an incomplete PDF body")
    expected_host = (urlsplit(url).hostname or "").casefold()
    if not fetched.redirect_chain or fetched.redirect_chain[0] != url:
        raise ValueError("fetcher omitted the exact evidence locator")
    if len(fetched.redirect_chain) > MAX_REDIRECTS + 1:
        raise ValueError("fetcher exceeded the fixed redirect limit")
    for redirected in fetched.redirect_chain:
        parts = urlsplit(redirected)
        if (parts.scheme.casefold() != "https"
                or (parts.hostname or "").casefold() != expected_host):
            raise ValueError("fetcher left the exact same-host HTTPS boundary")
    final_parts = urlsplit(fetched.final_url)
    if (fetched.redirect_chain[-1] != fetched.final_url
            or final_parts.scheme.casefold() != "https"
            or (final_parts.hostname or "").casefold() != expected_host):
        raise ValueError("fetcher final URL differs from its same-host redirect chain")
    if (fetched.final_url != url
            or any(redirected != url for redirected in fetched.redirect_chain)):
        raise ValueError("fetcher changed the exact evidence path or query")
    return fetched


def _metadata_values(value: dict[str, Any]) -> dict[str, list[str]]:
    metadata = value.get("metadata")
    if not isinstance(metadata, list):
        raise ValueError("AMD map metadata inventory is absent")
    result: dict[str, list[str]] = {}
    for row in metadata:
        if not isinstance(row, dict):
            raise ValueError("AMD map metadata row is malformed")
        key = row.get("key")
        values = row.get("values")
        if (not isinstance(key, str) or key in result
                or not isinstance(values, list)
                or any(not isinstance(item, str) for item in values)):
            raise ValueError("AMD map metadata row is not unique text")
        result[key] = values
    return result


def _walk_json_objects(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_json_objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_json_objects(child)


def _validate_amd_map_body(mapping: dict[str, Any], body: bytes) -> None:
    identity = mapping["identity"]
    value = _strict_json_bytes(body, label="AMD KHUB map metadata")
    if not isinstance(value, dict):
        raise ValueError("AMD KHUB map metadata is not an object")
    version = str(identity["version"])
    publication_id = str(identity["publication_id"])
    citation_parts = urlsplit(str(mapping["citation_url"]))
    match = _AMD_CITATION_PATH_RE.fullmatch(citation_parts.path)
    if match is None:
        raise ValueError("AMD citation path is malformed")
    slug = match.group("document_slug") or match.group("document_slug_topic")
    reader_root = f"/r/{version}-English/{slug}"
    metadata = _metadata_values(value)
    if (identity["document_slug"] != slug
            or value.get("id") != publication_id
            or value.get("readerUrl") != reader_root
            or value.get("prettyUrl") != f"/go/{version}-English/{slug}"
            or value.get("title") != identity["title"]
            or metadata.get("Doc_Version") != [f"{version} English"]
            or metadata.get("Access_Level") != ["Public"]
            or metadata.get("ft:publicationId") != [publication_id]
            or metadata.get("ft:prettyUrl") != [f"{version}-English/{slug}"]
            or metadata.get("Document_ID") != [identity["document_id"]]
            or value.get("clusterId") != identity["document_id"]):
        raise ValueError("AMD KHUB map identity or public version does not match")


def _validate_amd_pages_body(mapping: dict[str, Any], body: bytes) -> None:
    identity = mapping["identity"]
    value = _strict_json_bytes(body, label="AMD KHUB pages")
    if (not isinstance(value, dict)
            or set(value) != {"configuration", "paginatedToc", "translationError"}):
        raise ValueError("AMD KHUB pages response has an unsupported root")
    citation_path = urlsplit(str(mapping["citation_url"])).path.rstrip("/")
    matches = [
        row for row in _walk_json_objects(value)
        if row.get("prettyUrl") == citation_path
    ]
    if len(matches) != 1:
        raise ValueError("AMD KHUB pages does not uniquely map the human citation")
    row = matches[0]
    if (row.get("tocId") != identity["toc_id"]
            or row.get("contentId") != identity["content_id"]
            or row.get("title") != identity["topic_title"]):
        raise ValueError("AMD KHUB topic TOC/content/title identity does not match")


def _resolver_artifact(
    root: Path, kind: str, requested_url: str, fetched: TrustedFetch,
) -> dict[str, Any]:
    body_hash = hashlib.sha256(fetched.body).hexdigest()
    relative = f"citations/resolver/{body_hash}.body"
    path = root / PurePosixPath(relative)
    if not path.exists():
        _write_private(path, fetched.body)
    elif path.read_bytes() != fetched.body:
        raise ValueError("resolver body hash collision")
    return {
        "kind": kind, "requested_url": requested_url,
        "status": fetched.status, "final_url": fetched.final_url,
        "redirect_chain": list(fetched.redirect_chain),
        "content_type": fetched.content_type, "body_path": relative,
        "body_sha256": body_hash, "body_size": len(fetched.body),
    }


def create_review_cache(
    root: Path, snapshot: ReviewSnapshot, cache_root: Path, *,
    fetcher: Callable[[str, float, int], TrustedFetch] = _default_fetch,
    timeout_seconds: float = 60.0, max_bytes: int = MAX_CITATION_BYTES,
    pdf_text_extractor: Callable[[bytes], TextDerivation | None] | None = None,
    pdftotext_command: str | None = None,
    pdftotext_sha256: str | None = None,
) -> ReviewCache:
    """Create one private, immutable review cache outside the checkout."""

    if (pdftotext_command is None) != (pdftotext_sha256 is None):
        raise ValueError("pdftotext path and expected SHA-256 must be supplied together")
    if pdftotext_command is not None and (
        not Path(pdftotext_command).is_absolute()
        or Path(pdftotext_command).as_posix() != PDFTOTEXT_ALLOWED_PATH
        or _SHA256_RE.fullmatch(str(pdftotext_sha256)) is None
    ):
        raise ValueError("pdftotext contract requires /usr/bin/pdftotext and SHA-256")

    root = root.resolve(strict=True)
    lexical_cache = cache_root.absolute()
    try:
        lexical_cache.resolve(strict=False).relative_to(root)
    except ValueError:
        pass
    else:
        raise RuntimeError("review cache must stay outside the public checkout")
    if lexical_cache.exists():
        raise RuntimeError("review cache path already exists")
    if not lexical_cache.parent.is_dir() or lexical_cache.parent.is_symlink():
        raise RuntimeError("review cache parent must be an existing plain directory")
    _mkdir_private(lexical_cache)
    files_inventory: list[dict[str, Any]] = []
    for item in snapshot.files:
        relative = PurePosixPath("files") / PurePosixPath(item.path)
        target = lexical_cache / relative
        _write_private(target, item.payload)
        inventory = item.inventory()
        inventory["model_inspection_required"] = _model_inspection_required(item.path)
        inventory["chunks"] = (
            _chunk_inventory(
                lexical_cache, origin_kind="source", origin_id=item.path,
                original_sha256=item.cache_sha256, payload=item.payload,
            )
            if inventory["model_inspection_required"] else []
        )
        files_inventory.append(inventory)

    references_by_url: dict[str, list[str]] = {}
    rule_references_by_url: dict[str, list[str]] = {}
    for row in _citation_reference_rows(snapshot):
        url = str(row.get("citation_url", ""))
        reference_id = str(row.get("reference_id", ""))
        if url not in snapshot.exact_citation_urls or _SHA256_RE.fullmatch(reference_id) is None:
            raise ValueError("citation reference is not bound to the frozen inventory")
        references_by_url.setdefault(url, []).append(reference_id)
        if row.get("reference_kind") == "rule":
            rule_references_by_url.setdefault(url, []).append(reference_id)

    evidence_rows = _citation_evidence_rows(snapshot)
    citations: list[dict[str, Any]] = []
    # A review cache may cite many topics from one AMD publication.  Fetch the
    # immutable map identity and pages inventory once per publication, then
    # attach the same content-addressed supporting artifacts to every topic.
    # This both reduces rate-limit exposure and prevents one review snapshot
    # from silently mixing multiple server responses for the same map.
    amd_map_fetches: dict[str, TrustedFetch] = {}
    amd_pages_fetches: dict[str, TrustedFetch] = {}
    primary_fetches: dict[str, TrustedFetch] = {}
    primary_fetch_failures: set[str] = set()
    for mapping in evidence_rows:
        url = str(mapping["citation_url"])
        evidence_url = str(mapping["evidence_url"])
        base: dict[str, Any] = {
            "requested_url": url,
            "evidence_url": evidence_url,
            "resolver_id": mapping["resolver_id"],
            "reference_ids": sorted(references_by_url.get(url, [])),
            "inspection_required": bool(rule_references_by_url.get(url)),
            "identity_verified": False,
            "available": False,
            "status": None,
            "final_url": None,
            "redirect_chain": [],
            "content_type": None,
            "body_path": None,
            "body_sha256": None,
            "body_size": None,
            "inspection_path": None,
            "inspection_sha256": None,
            "inspection_size": None,
            "inspection_chunks": [],
            "parser_id": None,
            "parser_version": None,
            "parser_command_sha256": None,
            "parser_executable_sha256": None,
            "parser_version_output_sha256": None,
            "resolver_artifacts": [],
            "error_code": None,
        }
        try:
            if evidence_url in primary_fetch_failures:
                raise ValueError("shared exact evidence fetch previously failed")
            fetched = primary_fetches.get(evidence_url)
            if fetched is None:
                try:
                    fetched = _checked_fetch(
                        fetcher, evidence_url, timeout_seconds, max_bytes,
                    )
                except (AttributeError, OSError, TypeError, ValueError):
                    primary_fetch_failures.add(evidence_url)
                    raise
                primary_fetches[evidence_url] = fetched
            body_hash = hashlib.sha256(fetched.body).hexdigest()
            body_relative = f"citations/bodies/{body_hash}.body"
            body_path = lexical_cache / PurePosixPath(body_relative)
            if not body_path.exists():
                _write_private(body_path, fetched.body)
            elif body_path.read_bytes() != fetched.body:
                raise ValueError("citation body hash collision")
            base.update({
                "status": fetched.status, "final_url": fetched.final_url,
                "redirect_chain": list(fetched.redirect_chain),
                "content_type": fetched.content_type,
                "body_path": body_relative, "body_sha256": body_hash,
                "body_size": len(fetched.body),
            })
            derivation: TextDerivation | None = None
            if base["inspection_required"]:
                if mapping["resolver_id"] == "github.raw.lines.v1":
                    derivation = _github_line_range_derivation(mapping, fetched)
                else:
                    derivation = _text_derivation(fetched)
                    if derivation is None and (
                        fetched.body.startswith(b"%PDF-")
                        or fetched.content_type.casefold() == "application/pdf"
                    ):
                        derivation = (
                            pdf_text_extractor(fetched.body)
                            if pdf_text_extractor is not None
                            else _pdftotext_derivation(
                                body_path, pdftotext_command, pdftotext_sha256,
                            )
                        )
            resolver_artifacts: list[dict[str, Any]] = []
            if mapping["resolver_id"] == "amd.docs.khub.map.v1":
                _validate_amd_map_body(mapping, fetched.body)
                base["identity_verified"] = True
                publication_id = str(mapping["identity"]["publication_id"])
                amd_map_fetches[publication_id] = fetched
            elif mapping["resolver_id"] == "amd.docs.khub.topic.v1":
                publication_id = str(mapping["identity"]["publication_id"])
                map_url = f"https://docs.amd.com/api/khub/maps/{publication_id}"
                pages_url = map_url + "/pages"
                map_fetch = amd_map_fetches.get(publication_id)
                if map_fetch is None:
                    map_fetch = _checked_fetch(
                        fetcher, map_url, timeout_seconds, max_bytes,
                    )
                    amd_map_fetches[publication_id] = map_fetch
                pages_fetch = amd_pages_fetches.get(publication_id)
                if pages_fetch is None:
                    pages_fetch = _checked_fetch(
                        fetcher, pages_url, timeout_seconds, max_bytes,
                    )
                    amd_pages_fetches[publication_id] = pages_fetch
                _validate_amd_map_body(mapping, map_fetch.body)
                _validate_amd_pages_body(mapping, pages_fetch.body)
                resolver_artifacts = [
                    _resolver_artifact(
                        lexical_cache, "amd_map_metadata", map_url, map_fetch,
                    ),
                    _resolver_artifact(
                        lexical_cache, "amd_pages", pages_url, pages_fetch,
                    ),
                ]
                base["identity_verified"] = True
            elif mapping["resolver_id"] in {
                "direct.sha256.v1", "github.raw.document.v1",
            }:
                _validate_document_identity_body(mapping, fetched)
                base["identity_verified"] = True
            elif mapping["resolver_id"] == "github.raw.lines.v1":
                if derivation is None:
                    raise ValueError(
                        "GitHub rule evidence lacks its exact line derivation",
                    )
                base["identity_verified"] = True
            base["resolver_artifacts"] = resolver_artifacts
            if not base["identity_verified"]:
                raise ValueError("citation resolver did not verify its identity")
            if not base["inspection_required"]:
                base["available"] = True
            elif derivation is None or not derivation.text.strip():
                base["error_code"] = "citation_text_unavailable"
            else:
                if (not isinstance(derivation, TextDerivation)
                        or not isinstance(derivation.text, bytes)
                        or not isinstance(derivation.parser_id, str)
                        or not derivation.parser_id
                        or not isinstance(derivation.parser_version, str)
                        or not derivation.parser_version
                        or _SHA256_RE.fullmatch(
                            str(derivation.command_sha256)
                        ) is None):
                    raise ValueError("citation text derivation lacks a bound parser contract")
                text_hash = hashlib.sha256(derivation.text).hexdigest()
                text_relative = f"citations/text/{text_hash}.txt"
                text_path = lexical_cache / PurePosixPath(text_relative)
                if not text_path.exists():
                    _write_private(text_path, derivation.text)
                elif text_path.read_bytes() != derivation.text:
                    raise ValueError("citation text hash collision")
                base.update({
                    "available": True,
                    "inspection_path": text_relative,
                    "inspection_sha256": text_hash,
                    "inspection_size": len(derivation.text),
                    "parser_id": derivation.parser_id,
                    "parser_version": derivation.parser_version,
                    "parser_command_sha256": derivation.command_sha256,
                    "parser_executable_sha256": derivation.executable_sha256,
                    "parser_version_output_sha256": (
                        derivation.version_output_sha256
                    ),
                    "inspection_chunks": _chunk_inventory(
                        lexical_cache, origin_kind="citation", origin_id=url,
                        original_sha256=text_hash, payload=derivation.text,
                    ),
                })
        except (
            AttributeError, OSError, TypeError, ValueError,
            subprocess.SubprocessError,
        ) as exc:
            base["error_code"] = type(exc).__name__
        citations.append(base)

    parser_contract_sha256s = sorted({
        str(entry["parser_command_sha256"])
        for entry in citations
        if entry.get("available") is True
        and entry.get("inspection_required") is True
    })
    manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "chunk_contract": _chunk_contract(),
        "inspection_contract": _inspection_contract(files_inventory, citations),
        "parser_contract_sha256s": parser_contract_sha256s,
        "protocol_id": snapshot.protocol_id,
        "review_snapshot_sha256": snapshot.sha256,
        "citation_evidence_sha256": snapshot.citation_evidence_sha256,
        "review_snapshot": snapshot.inventory(),
        "files": files_inventory,
        "citations": citations,
    }
    manifest_bytes = _canonical_json(manifest)
    _write_private(lexical_cache / CACHE_MANIFEST_NAME, manifest_bytes)
    _harden_private_tree(lexical_cache)
    return ReviewCache(lexical_cache.resolve(), manifest, manifest_bytes)


def build_review_prompt(
    root: Path, protocol_id: str, *, snapshot: ReviewSnapshot | None = None,
    cache: ReviewCache | None = None,
) -> bytes:
    files = PROTOCOL_FILES.get(protocol_id)
    if files is None:
        raise ValueError(f"unknown knowledge-review protocol: {protocol_id}")
    snapshot = snapshot or freeze_review_snapshot(root, protocol_id)
    if snapshot.protocol_id != protocol_id:
        raise ValueError("review snapshot belongs to another protocol")
    prompt_file = snapshot.file_map.get(files["prompt"])
    if prompt_file is None:
        raise ValueError("review snapshot omits its protocol prompt")
    protocol = prompt_file.payload.decode("utf-8", errors="strict")
    inventory = {
        **snapshot.inventory(),
        "review_snapshot_sha256": snapshot.sha256,
        "cache_manifest_sha256": cache.sha256 if cache is not None else None,
        "cache_manifest": cache.manifest if cache is not None else None,
    }
    command_contract = """
The model has no network and the checkout itself is not readable. The current
working directory is the private frozen cache. The only permitted shell
commands are these exact read-only forms, using forward-slash relative paths
listed only in cache_manifest.files[*].chunks[*].path or an available,
`inspection_required=true`
cache_manifest.citations[*].inspection_chunks[*].path:

  head -n COUNT PATH
  sha256sum PATH [PATH ...]

Only source rows with `model_inspection_required=true` are readable and must
be inspected. Other snapshot rows are `integrity_bound_only`: their hashes
still invalidate the review when changed, but neither the prompt nor receipt
claims the model inspected their content. The explicit split and its digest
are in `cache_manifest.inspection_contract`.

Document-only citation locators have `inspection_required=false`. Their exact
fetch, identity, version and body hash are deterministically bound by the
cache, but no claim is made that the model read their full content. For those
document citation-result rows set `exact_locator_inspected=false`; rule rows
must inspect every normalized section chunk and set it true.

Use `head -n 100000000 PATH` exactly once or more for every required source
chunk and every required citation-section chunk.
Every chunk is at most 24000 UTF-8 bytes; all contiguous ranges must be seen
with exact, untruncated tool output before the parent file or citation counts
as inspected. `sha256sum` alone is hash evidence, not content-inspection
evidence. Do not access the unchunked source, derived text, or cached raw
response bodies directly. Do not use any other command, interpreter, pipe,
redirection, environment expansion, native web/search tool, MCP tool, network
operation, or file-changing operation. Every unknown event or tool makes the
review unusable. If a citation entry is unavailable or cannot be read in full,
its rule verdict must not be verified and approved must be false. An unavailable
document-only locator also makes approval false.
""".strip()
    payload = json.dumps(inventory, indent=2, sort_keys=True, ensure_ascii=False)
    prompt = (protocol.rstrip() + "\n\n" + command_contract +
              "\n\n# Frozen review inventory\n\n```json\n" + payload +
              "\n```\n").encode("utf-8")
    if cache is not None and len(prompt) > MAX_INITIAL_PROMPT_BYTES:
        raise RuntimeError(
            "formal review initial prompt exceeds the fixed visibility budget; NO-GO"
        )
    return prompt


def _safe_relative(token: str) -> PurePosixPath:
    if (not token or "\\" in token or token.startswith("/")
            or re.match(r"^[A-Za-z]:", token)):
        raise ValueError(f"command uses a non-project-relative path: {token!r}")
    path = PurePosixPath(token)
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"command uses a non-canonical path: {token!r}")
    return path


def _split_command(command: str) -> list[str]:
    if not command or _FORBIDDEN_SHELL.search(command):
        raise ValueError("command contains chaining, expansion, redirection, or control text")
    try:
        parts = shlex.split(command, posix=True)
    except ValueError as exc:
        raise ValueError(f"command has invalid quoting: {exc}") from exc
    if not parts:
        raise ValueError("empty command")
    return parts


def _codex_shell_event_command(inner_command: str) -> str:
    """Return the one canonical wrapper emitted by the pinned Codex CLI."""

    if (not isinstance(inner_command, str) or not inner_command
            or "'" in inner_command):
        raise ValueError("inner review command cannot use shell quoting")
    _split_command(inner_command)
    return f"/bin/bash -lc '{inner_command}'"


def _unwrap_codex_shell_event_command(event_command: str) -> str:
    """Accept only the pinned CLI's exact ``/bin/bash -lc`` event wrapper."""

    prefix = "/bin/bash -lc '"
    if (not isinstance(event_command, str)
            or not event_command.startswith(prefix)
            or not event_command.endswith("'")):
        raise ValueError("command does not use the canonical Codex shell wrapper")
    inner = event_command[len(prefix):-1]
    if _codex_shell_event_command(inner) != event_command:
        raise ValueError("Codex shell wrapper is not canonical")
    return inner


def _validate_chunk_rows(
    root: Path, rows: Any, *, origin_kind: str, origin_id: str,
    original_sha256: str, original: bytes, expected_paths: set[str],
) -> None:
    expected = _chunk_rows(
        origin_kind=origin_kind, origin_id=origin_id,
        original_sha256=original_sha256, payload=original,
    )
    if rows != expected:
        raise ValueError(f"review cache {origin_kind} chunk inventory is stale")
    rebuilt = bytearray()
    for row in expected:
        relative = str(row["path"])
        expected_paths.add(relative)
        chunk = _read_private_cache_file(root, relative)
        if (len(chunk) > MAX_REVIEW_CHUNK_BYTES
                or len(chunk) != row["size"]
                or hashlib.sha256(chunk).hexdigest() != row["sha256"]
                or row["byte_start"] != len(rebuilt)
                or row["byte_end"] != len(rebuilt) + len(chunk)):
            raise ValueError(f"review cache {origin_kind} chunk is stale")
        chunk.decode("utf-8", errors="strict")
        rebuilt.extend(chunk)
    if bytes(rebuilt) != original:
        raise ValueError(f"review cache {origin_kind} chunks do not reconstruct input")


def load_review_cache(cache_root: Path, snapshot: ReviewSnapshot) -> ReviewCache:
    lexical_root = cache_root.absolute()
    if _is_link_like(lexical_root):
        raise ValueError("review cache root must be a plain directory")
    root = lexical_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("review cache root is not a directory")
    _assert_private_mode(root, CACHE_DIRECTORY_MODE, label="review cache root")
    manifest_bytes = _read_private_cache_file(root, CACHE_MANIFEST_NAME)
    manifest = _strict_json_bytes(manifest_bytes, label="review cache manifest")
    if not isinstance(manifest, dict) or _canonical_json(manifest) != manifest_bytes:
        raise ValueError("review cache manifest is not canonical JSON")
    if set(manifest) != {
        "schema_version", "protocol_id", "review_snapshot_sha256",
        "citation_evidence_sha256", "review_snapshot", "files", "citations",
        "chunk_contract", "inspection_contract", "parser_contract_sha256s",
    }:
        raise ValueError("review cache manifest is not a closed contract")
    if (manifest.get("schema_version") != CACHE_SCHEMA_VERSION
            or manifest.get("protocol_id") != snapshot.protocol_id
            or manifest.get("review_snapshot_sha256") != snapshot.sha256
            or manifest.get("citation_evidence_sha256")
            != snapshot.citation_evidence_sha256
            or manifest.get("review_snapshot") != snapshot.inventory()):
        raise ValueError("review cache manifest does not bind the exact snapshot")
    if manifest.get("chunk_contract") != _chunk_contract():
        raise ValueError("review cache uses a stale or weakened chunk contract")
    parser_contracts = manifest.get("parser_contract_sha256s")
    if (not isinstance(parser_contracts, list)
            or parser_contracts != sorted(set(parser_contracts))
            or any(_SHA256_RE.fullmatch(str(item)) is None for item in parser_contracts)):
        raise ValueError("review cache parser contract inventory is malformed")
    expected_files = [item.inventory() for item in snapshot.files]
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != len(expected_files):
        raise ValueError("review cache file inventory differs from the snapshot")
    expected_paths = {CACHE_MANIFEST_NAME}
    for item, expected, frozen in zip(files, expected_files, snapshot.files, strict=True):
        if (not isinstance(item, dict)
                or set(item) != set(expected) | {
                    "chunks", "model_inspection_required",
                }
                or {key: item.get(key) for key in expected} != expected):
            raise ValueError("review cache file inventory differs from the snapshot")
        if item.get("model_inspection_required") is not _model_inspection_required(
            frozen.path
        ):
            raise ValueError("review cache inspection scope differs from the TCB contract")
        relative = str(expected["cache_path"])
        expected_paths.add(relative)
        data = _read_private_cache_file(root, relative)
        if (hashlib.sha256(data).hexdigest() != expected["cache_sha256"]
                or len(data) != expected["cache_size"]):
            raise ValueError(f"review cache source is stale: {expected['path']}")
        if item["model_inspection_required"]:
            _validate_chunk_rows(
                root, item.get("chunks"), origin_kind="source",
                origin_id=frozen.path, original_sha256=frozen.cache_sha256,
                original=data, expected_paths=expected_paths,
            )
        elif item.get("chunks") != []:
            raise ValueError("integrity-only source must not expose model chunks")
    citations = manifest.get("citations")
    if not isinstance(citations, list):
        raise ValueError("review cache has no citation inventory")
    references_by_url: dict[str, list[str]] = {}
    rule_references_by_url: dict[str, list[str]] = {}
    for row in _citation_reference_rows(snapshot):
        url = str(row["citation_url"])
        references_by_url.setdefault(url, []).append(str(row["reference_id"]))
        if row.get("reference_kind") == "rule":
            rule_references_by_url.setdefault(url, []).append(
                str(row["reference_id"]),
            )
    mappings_by_url = {
        str(row["citation_url"]): row for row in _citation_evidence_rows(snapshot)
    }
    observed_urls: set[str] = set()
    for entry in citations:
        if not isinstance(entry, dict) or set(entry) != {
            "requested_url", "evidence_url", "resolver_id", "reference_ids",
            "inspection_required", "identity_verified", "available", "status",
            "final_url", "redirect_chain", "content_type", "body_path",
            "body_sha256", "body_size", "inspection_path",
            "inspection_sha256", "inspection_size", "parser_id",
            "parser_version", "parser_command_sha256",
            "parser_executable_sha256", "parser_version_output_sha256",
            "resolver_artifacts",
            "inspection_chunks", "error_code",
        }:
            raise ValueError("review cache contains a malformed citation entry")
        url = entry.get("requested_url")
        if not isinstance(url, str) or url in observed_urls:
            raise ValueError("review cache contains a duplicate citation locator")
        observed_urls.add(url)
        mapping = mappings_by_url.get(url)
        if (mapping is None
                or entry.get("evidence_url") != mapping["evidence_url"]
                or entry.get("resolver_id") != mapping["resolver_id"]):
            raise ValueError("review cache citation evidence mapping is stale")
        if entry.get("reference_ids") != sorted(references_by_url.get(url, [])):
            raise ValueError("review cache citation reference inventory is stale")
        if entry.get("inspection_required") is not bool(
            rule_references_by_url.get(url),
        ):
            raise ValueError("review cache citation inspection scope is stale")
        requested_parts = urlsplit(str(entry["evidence_url"]))
        expected_host = (requested_parts.hostname or "").casefold()
        chain = entry.get("redirect_chain")
        if entry.get("status") is not None:
            if (not isinstance(chain, list) or not chain
                    or chain[0] != entry["evidence_url"]
                    or len(chain) > MAX_REDIRECTS + 1
                    or chain[-1] != entry.get("final_url")):
                raise ValueError("review cache has an invalid redirect chain")
            for redirected in chain:
                parts = urlsplit(str(redirected))
                if (parts.scheme.casefold() != "https" or not parts.hostname
                        or parts.hostname.casefold() != expected_host):
                    raise ValueError("review cache redirect leaves same-host HTTPS")
                if redirected != entry["evidence_url"]:
                    raise ValueError("review cache redirect changes exact evidence locator")
        elif chain != [] or entry.get("final_url") is not None:
            raise ValueError("failed citation cache entry claims redirect evidence")
        for path_key, hash_key, size_key in (
            ("body_path", "body_sha256", "body_size"),
            ("inspection_path", "inspection_sha256", "inspection_size"),
        ):
            relative = entry.get(path_key)
            if relative is None:
                continue
            relative = _safe_relative(str(relative)).as_posix()
            expected_prefix = (
                "citations/bodies/" if path_key == "body_path"
                else "citations/text/"
            )
            expected_suffix = ".body" if path_key == "body_path" else ".txt"
            digest = entry.get(hash_key)
            if (not relative.startswith(expected_prefix)
                    or relative != f"{expected_prefix}{digest}{expected_suffix}"
                    or _SHA256_RE.fullmatch(str(digest)) is None
                    or not isinstance(entry.get(size_key), int)
                    or entry[size_key] < 0):
                raise ValueError("review cache citation path is not content addressed")
            expected_paths.add(relative)
            data = _read_private_cache_file(root, relative)
            if (hashlib.sha256(data).hexdigest() != entry.get(hash_key)
                    or len(data) != entry.get(size_key)):
                raise ValueError(f"review cache citation data is stale: {url}")
        resolver_artifacts = entry.get("resolver_artifacts")
        if not isinstance(resolver_artifacts, list):
            raise ValueError("review cache resolver artifacts are not an array")
        resolver_payloads: dict[str, bytes] = {}
        for artifact in resolver_artifacts:
            if not isinstance(artifact, dict) or set(artifact) != {
                "kind", "requested_url", "status", "final_url",
                "redirect_chain", "content_type", "body_path", "body_sha256",
                "body_size",
            }:
                raise ValueError("review cache has a malformed resolver artifact")
            kind = artifact.get("kind")
            supporting_url = artifact.get("requested_url")
            supporting_chain = artifact.get("redirect_chain")
            supporting_parts = urlsplit(str(supporting_url))
            supporting_host = (supporting_parts.hostname or "").casefold()
            if (kind not in {"amd_map_metadata", "amd_pages"}
                    or not isinstance(supporting_url, str)
                    or supporting_parts.scheme.casefold() != "https"
                    or supporting_host != "docs.amd.com"
                    or type(artifact.get("status")) is not int
                    or not 200 <= artifact["status"] < 300
                    or not isinstance(supporting_chain, list)
                    or not supporting_chain
                    or supporting_chain[0] != supporting_url
                    or supporting_chain[-1] != artifact.get("final_url")
                    or len(supporting_chain) > MAX_REDIRECTS + 1
                    or any(
                        urlsplit(str(item)).scheme.casefold() != "https"
                        or (urlsplit(str(item)).hostname or "").casefold()
                        != supporting_host
                        or item != supporting_url
                        for item in supporting_chain
                    )):
                raise ValueError("review cache resolver fetch chain is invalid")
            relative = _safe_relative(str(artifact.get("body_path"))).as_posix()
            digest = artifact.get("body_sha256")
            if (not relative.startswith("citations/resolver/")
                    or relative != f"citations/resolver/{digest}.body"
                    or _SHA256_RE.fullmatch(str(digest)) is None
                    or type(artifact.get("body_size")) is not int
                    or artifact["body_size"] < 0):
                raise ValueError("review cache resolver body is not content addressed")
            expected_paths.add(relative)
            payload = _read_private_cache_file(root, relative)
            if (hashlib.sha256(payload).hexdigest() != digest
                    or len(payload) != artifact["body_size"]):
                raise ValueError("review cache resolver body is stale")
            if kind in resolver_payloads:
                raise ValueError("review cache duplicates a resolver artifact kind")
            resolver_payloads[str(kind)] = payload
        available = entry.get("available")
        if type(entry.get("identity_verified")) is not bool:
            raise ValueError("review cache citation identity state is not boolean")
        resolver_id = entry["resolver_id"]
        if resolver_id in {
            "direct.sha256.v1", "github.raw.document.v1",
            "github.raw.lines.v1",
        }:
            if resolver_artifacts:
                raise ValueError("direct citation has unexpected resolver artifacts")
            if resolver_id == "github.raw.lines.v1" and available is True:
                primary_body = _read_private_cache_file(
                    root, str(entry["body_path"]),
                )
                derived = _github_line_range_derivation(
                    mapping,
                    TrustedFetch(
                        status=int(entry["status"]),
                        final_url=str(entry["final_url"]),
                        redirect_chain=tuple(entry["redirect_chain"]),
                        content_type=str(entry["content_type"]),
                        body=primary_body,
                        charset="utf-8",
                    ),
                )
                if entry.get("inspection_required") is True:
                    inspection = _read_private_cache_file(
                        root, str(entry["inspection_path"]),
                    )
                    if inspection != derived.text:
                        raise ValueError(
                            "GitHub inspection bytes differ from the bound raw range",
                        )
            elif resolver_id in {
                "direct.sha256.v1", "github.raw.document.v1",
            } and available is True:
                primary_body = _read_private_cache_file(
                    root, str(entry["body_path"]),
                )
                _validate_document_identity_body(
                    mapping,
                    TrustedFetch(
                        status=int(entry["status"]),
                        final_url=str(entry["final_url"]),
                        redirect_chain=tuple(entry["redirect_chain"]),
                        content_type=str(entry["content_type"]),
                        body=primary_body,
                    ),
                )
        elif resolver_id == "amd.docs.khub.map.v1":
            if resolver_artifacts:
                raise ValueError("AMD map citation has unexpected resolver artifacts")
            if available is True:
                primary_body = _read_private_cache_file(
                    root, str(entry["body_path"]),
                )
                _validate_amd_map_body(mapping, primary_body)
        elif resolver_id == "amd.docs.khub.topic.v1":
            publication_id = str(mapping["identity"]["publication_id"])
            expected_support = [
                ("amd_map_metadata", f"https://docs.amd.com/api/khub/maps/{publication_id}"),
                ("amd_pages", f"https://docs.amd.com/api/khub/maps/{publication_id}/pages"),
            ]
            observed_support = [
                (item.get("kind"), item.get("requested_url"))
                for item in resolver_artifacts
            ]
            if observed_support not in ([], expected_support):
                raise ValueError("AMD topic resolver artifact inventory is stale")
            if observed_support == expected_support:
                _validate_amd_map_body(mapping, resolver_payloads["amd_map_metadata"])
                _validate_amd_pages_body(mapping, resolver_payloads["amd_pages"])
            elif available is True:
                raise ValueError("available AMD topic lacks resolver evidence")
        else:
            raise ValueError("review cache uses an unknown resolver")
        if available is True:
            if (not isinstance(entry.get("status"), int)
                    or not 200 <= entry["status"] < 300
                    or not entry.get("body_path")
                    or entry.get("identity_verified") is not True
                    or entry.get("error_code") is not None):
                raise ValueError("available citation lacks fetched locator evidence")
            if entry["inspection_required"] is False:
                if any(entry.get(key) is not None for key in (
                    "inspection_path", "inspection_sha256", "inspection_size",
                    "parser_id", "parser_version", "parser_command_sha256",
                    "parser_executable_sha256", "parser_version_output_sha256",
                )) or entry.get("inspection_chunks") != []:
                    raise ValueError(
                        "document-only citation claims model content inspection",
                    )
                continue
            if (not entry.get("inspection_path")
                    or not isinstance(entry.get("parser_id"), str)
                    or not entry["parser_id"]
                    or not isinstance(entry.get("parser_version"), str)
                    or not entry["parser_version"]
                    or _SHA256_RE.fullmatch(
                        str(entry.get("parser_command_sha256", ""))
                    ) is None):
                raise ValueError("available citation lacks fetched parser-bound text")
            if entry.get("parser_id") == "hlsgraph.review.pdftotext.v1":
                if (_SHA256_RE.fullmatch(str(entry.get("parser_executable_sha256"))) is None
                        or _SHA256_RE.fullmatch(
                            str(entry.get("parser_version_output_sha256"))
                        ) is None):
                    raise ValueError("PDF citation lacks executable/version hash contract")
            elif (entry.get("parser_executable_sha256") is not None
                  or entry.get("parser_version_output_sha256") is not None):
                raise ValueError("non-PDF citation claims an executable parser contract")
            inspection = _read_private_cache_file(
                root, str(entry["inspection_path"]),
            )
            _validate_chunk_rows(
                root, entry.get("inspection_chunks"), origin_kind="citation",
                origin_id=url, original_sha256=str(entry["inspection_sha256"]),
                original=inspection, expected_paths=expected_paths,
            )
        elif available is not False:
            raise ValueError("review cache citation availability is not boolean")
        elif entry.get("inspection_chunks") != []:
            raise ValueError("unavailable citation claims inspection chunks")
    if observed_urls != set(snapshot.exact_citation_urls):
        raise ValueError("review cache citation inventory differs from the snapshot")
    if manifest.get("inspection_contract") != _inspection_contract(files, citations):
        raise ValueError("review cache inspection-scope contract is stale")
    observed_parser_contracts = sorted({
        str(entry["parser_command_sha256"])
        for entry in citations
        if entry.get("available") is True
        and entry.get("inspection_required") is True
    })
    if observed_parser_contracts != parser_contracts:
        raise ValueError("review cache parser contract inventory is stale")

    observed_paths: set[str] = set()
    observed_directories: set[str] = {"."}
    for current, directories, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        relative_parent = current_path.relative_to(root)
        _assert_private_mode(
            current_path, CACHE_DIRECTORY_MODE, label="review cache directory",
        )
        for name in directories:
            path = current_path / name
            if _is_link_like(path):
                raise ValueError("review cache contains a linked directory")
            observed_directories.add((relative_parent / name).as_posix())
        for name in filenames:
            path = current_path / name
            if (_is_link_like(path) or not path.is_file()
                    or path.stat(follow_symlinks=False).st_nlink != 1):
                raise ValueError("review cache contains a non-plain file")
            observed_paths.add((relative_parent / name).as_posix())
            _assert_private_mode(
                path, CACHE_FILE_MODE, label="review cache file",
            )
    expected_directories = {"."}
    for relative in expected_paths:
        parent = PurePosixPath(relative).parent
        while parent.as_posix() != ".":
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    if observed_paths != expected_paths or observed_directories != expected_directories:
        raise ValueError("review cache contains missing or unmanifested filesystem entries")
    return ReviewCache(root, manifest, manifest_bytes)


def _cache_targets(cache: ReviewCache) -> dict[str, dict[str, Any]]:
    targets: dict[str, dict[str, Any]] = {}
    for item in cache.manifest["files"]:
        if item.get("model_inspection_required") is not True:
            continue
        for chunk in item["chunks"]:
            path = str(chunk["path"])
            if path in targets:
                raise ValueError("review cache duplicates one source chunk path")
            targets[path] = {
                "target_kind": "source_chunk", "source": item, "chunk": chunk,
            }
    for entry in cache.manifest["citations"]:
        for chunk in entry.get("inspection_chunks", []):
            path = str(chunk["path"])
            if path in targets:
                raise ValueError("review cache duplicates one citation chunk path")
            targets[path] = {
                "target_kind": "citation_chunk", "entry": entry, "chunk": chunk,
            }
    return targets


def _command_output(item: dict[str, Any]) -> tuple[str, str]:
    present = [key for key in ("aggregated_output", "output", "result") if key in item]
    if present != ["aggregated_output"] or not isinstance(
        item.get("aggregated_output"), str,
    ):
        raise ValueError("completed command lacks one literal aggregated_output")
    return "aggregated_output", item["aggregated_output"]


def _expected_command(
    cache: ReviewCache, command: str,
) -> tuple[str, list[dict[str, Any]], bool]:
    parts = _split_command(command)
    executable = parts[0]
    targets = _cache_targets(cache)
    rows: list[dict[str, Any]] = []
    citation_content = False
    if executable == "head":
        if (len(parts) != 4 or parts[1] != "-n"
                or re.fullmatch(r"[0-9]+", parts[2]) is None
                or int(parts[2]) <= 0):
            raise ValueError("head command is outside the approved complete-read grammar")
        token = _safe_relative(parts[3]).as_posix()
        target = targets.get(token)
        if target is None:
            raise ValueError("head command reads a non-review cache file")
        data = _read_private_cache_file(cache.root, token)
        text = data.decode("utf-8", errors="strict")
        lines = text.splitlines(keepends=True)
        count = int(parts[2])
        expected = "".join(lines[:count])
        if count < len(lines):
            raise ValueError("head command does not inspect the complete cached file")
        chunk = target["chunk"]
        contract_sha256 = cache.manifest["chunk_contract"]["sha256"]
        if target["target_kind"] == "source_chunk":
            source = target["source"]
            rows.append({
                "kind": "file_chunk_read", "path": source["path"],
                "hash_kind": source["hash_kind"], "sha256": source["sha256"],
                "cache_sha256": source["cache_sha256"],
                "chunk_contract_sha256": contract_sha256,
                "chunk_index": chunk["index"], "chunk_path": chunk["path"],
                "chunk_sha256": chunk["sha256"], "chunk_size": chunk["size"],
                "byte_start": chunk["byte_start"], "byte_end": chunk["byte_end"],
            })
        else:
            citation_content = True
            entry = target["entry"]
            rows.append({
                "kind": "citation_chunk_read",
                "requested_url": entry["requested_url"],
                "evidence_url": entry["evidence_url"],
                "resolver_id": entry["resolver_id"],
                "reference_ids": entry["reference_ids"],
                "body_sha256": entry["body_sha256"],
                "inspection_sha256": entry["inspection_sha256"],
                "parser_id": entry["parser_id"],
                "parser_contract_sha256": entry["parser_command_sha256"],
                "chunk_contract_sha256": contract_sha256,
                "chunk_index": chunk["index"], "chunk_path": chunk["path"],
                "chunk_sha256": chunk["sha256"], "chunk_size": chunk["size"],
                "byte_start": chunk["byte_start"], "byte_end": chunk["byte_end"],
                "body_stored": False,
            })
        return expected, rows, citation_content
    if executable == "sha256sum":
        if len(parts) < 2:
            raise ValueError("sha256sum command has no path")
        output: list[str] = []
        for raw in parts[1:]:
            token = _safe_relative(raw).as_posix()
            target = targets.get(token)
            if target is None:
                raise ValueError("sha256sum reads a non-review cache file")
            data = _read_private_cache_file(cache.root, token)
            digest = hashlib.sha256(data).hexdigest()
            output.append(f"{digest}  {raw}\n")
            chunk = target["chunk"]
            if target["target_kind"] == "source_chunk":
                source = target["source"]
                rows.append({
                    "kind": "file_chunk_hash", "path": source["path"],
                    "chunk_index": chunk["index"],
                    "chunk_path": chunk["path"],
                    "chunk_sha256": chunk["sha256"],
                })
            else:
                entry = target["entry"]
                rows.append({
                    "kind": "citation_chunk_hash",
                    "requested_url": entry["requested_url"],
                    "evidence_url": entry["evidence_url"],
                    "resolver_id": entry["resolver_id"],
                    "inspection_sha256": entry["inspection_sha256"],
                    "chunk_index": chunk["index"],
                    "chunk_path": chunk["path"],
                    "chunk_sha256": chunk["sha256"],
                })
        return "".join(output), rows, False
    raise ValueError(f"unapproved command executable: {parts[0]!r}")


def _citation_marker(output: str) -> str:
    encoded = output.encode("utf-8")
    return (
        "HLSGRAPH_REVIEW_CACHE_OUTPUT:"
        + hashlib.sha256(encoded).hexdigest() + f":{len(encoded)}\n"
    )


def sanitize_raw_review_stream(
    raw_bytes: bytes, cache: ReviewCache,
) -> bytes:
    """Validate CLI command output and redact cached citation text in memory."""

    events = _strict_jsonl(raw_bytes, label="raw Codex review stream")
    for event in events:
        if event.get("type") != "item.completed":
            continue
        item = event.get("item")
        if not isinstance(item, dict) or item.get("type") != "command_execution":
            continue
        if (item.get("status") != "completed"
                or type(item.get("exit_code")) is not int
                or item["exit_code"] != 0):
            raise ValueError("review command did not complete successfully")
        key, output = _command_output(item)
        expected, _rows, citation_content = _expected_command(
            cache, str(item.get("command", "")),
        )
        if output != expected:
            raise ValueError("review command output differs from deterministic cache replay")
        if citation_content:
            item[key] = _citation_marker(expected)
    sanitized = _canonical_jsonl(events)
    for entry in cache.manifest["citations"]:
        private_paths = [entry.get("body_path"), entry.get("inspection_path")]
        private_paths.extend(
            chunk.get("path") for chunk in entry.get("inspection_chunks", [])
            if isinstance(chunk, dict)
        )
        for relative in private_paths:
            if not relative:
                continue
            payload = _read_private_cache_file(cache.root, str(relative))
            needles = {payload}
            try:
                text = payload.decode("utf-8", errors="strict")
            except UnicodeDecodeError:
                pass
            else:
                needles.add(json.dumps(
                    text, ensure_ascii=False,
                )[1:-1].encode("utf-8"))
            if payload and any(needle and needle in sanitized for needle in needles):
                raise ValueError("sanitized raw review stream retains citation content")
        for artifact in entry.get("resolver_artifacts", []):
            payload = _read_private_cache_file(
                cache.root, str(artifact["body_path"]),
            )
            if payload and payload in sanitized:
                raise ValueError("sanitized raw review stream retains resolver content")
    return sanitized


def _command_operations(
    cache: ReviewCache, command: str, output: str,
) -> list[dict[str, Any]]:
    expected, rows, citation_content = _expected_command(cache, command)
    required_output = _citation_marker(expected) if citation_content else expected
    if output != required_output:
        raise ValueError("stored command output differs from deterministic cache replay")
    return rows


def _review_result_issues(
    snapshot: ReviewSnapshot, cache: ReviewCache, result: dict[str, Any],
    operations: list[dict[str, Any]],
) -> list[str]:
    issues: list[str] = []
    expected_keys = {
        "protocol_id", "review_surface_sha256", "implementation_surface_sha256",
        "citation_audit_sha256", "citation_results", "approved", "issues", "summary",
    }
    if set(result) != expected_keys:
        issues.append("review result does not match the closed result contract")
    if result.get("protocol_id") != snapshot.protocol_id:
        issues.append("review result has the wrong protocol")
    if result.get("review_surface_sha256") != snapshot.surfaces:
        issues.append("review result has stale pack surfaces")
    if result.get("implementation_surface_sha256") != snapshot.implementation_surface_sha256:
        issues.append("review result has a stale implementation surface")
    if result.get("citation_audit_sha256") != snapshot.citation_audit_sha256:
        issues.append("review result has a stale citation audit")
    rows = result.get("citation_results")
    by_reference: dict[str, dict[str, Any]] = {}
    if not isinstance(rows, list):
        issues.append("review result citation_results is not an array")
        rows = []
    for row in rows:
        reference_id = row.get("reference_id") if isinstance(row, dict) else None
        if not isinstance(reference_id, str) or reference_id in by_reference:
            issues.append("review result has a duplicate or malformed citation row")
            continue
        if set(row) != {
            "reference_id", "reference_surface_sha256", "verdict",
            "exact_locator_inspected", "declared_version_matched",
            "declared_section_matched", "paraphrase_supported",
            "applicability_not_broader", "issues",
        }:
            issues.append("review result has a non-canonical citation row")
            continue
        citation_issues = row.get("issues")
        if (not isinstance(citation_issues, list)
                or any(item not in CONTROLLED_CITATION_ISSUE_CODES
                       for item in citation_issues)):
            issues.append("review result has uncontrolled citation issue content")
        by_reference[reference_id] = row
    references = {str(row["reference_id"]): row for row in _citation_reference_rows(snapshot)}
    if set(by_reference) != set(references):
        issues.append("review result citation inventory differs from the snapshot")
    file_chunk_reads: dict[str, set[str]] = {}
    citation_chunk_reads: dict[str, set[str]] = {}
    for operation in operations:
        if operation.get("kind") == "file_chunk_read":
            file_chunk_reads.setdefault(str(operation["path"]), set()).add(
                str(operation["chunk_path"])
            )
        elif operation.get("kind") == "citation_chunk_read":
            citation_chunk_reads.setdefault(
                str(operation["requested_url"]), set(),
            ).add(str(operation["chunk_path"]))
    required_file_chunks = {
        str(item["path"]): {str(chunk["path"]) for chunk in item["chunks"]}
        for item in cache.manifest["files"]
        if item.get("model_inspection_required") is True
    }
    required_citation_chunks = {
        str(item["requested_url"]): {
            str(chunk["path"]) for chunk in item.get("inspection_chunks", [])
        }
        for item in cache.manifest["citations"]
        if item.get("available") is True
        and item.get("inspection_required") is True
    }
    inspected_files = {
        path for path, required in required_file_chunks.items()
        if file_chunk_reads.get(path, set()) == required
    }
    inspected_urls = {
        url for url, required in required_citation_chunks.items()
        if citation_chunk_reads.get(url, set()) == required
    }
    cache_by_url = {
        str(row["requested_url"]): row for row in cache.manifest["citations"]
    }
    all_verified = True
    for reference_id, expected in references.items():
        row = by_reference.get(reference_id)
        if row is None:
            all_verified = False
            continue
        url = str(expected["citation_url"])
        verified = (
            row.get("reference_surface_sha256") == expected.get("reference_surface_sha256")
            and row.get("verdict") == "verified"
            and row.get("declared_version_matched") is True
            and row.get("issues") == []
            and cache_by_url[url].get("available") is True
            and cache_by_url[url].get("identity_verified") is True
        )
        if expected.get("reference_kind") == "rule":
            verified = (
                verified
                and row.get("exact_locator_inspected") is True
                and url in inspected_urls
                and cache_by_url[url].get("inspection_required") is True
                and all(
                row.get(key) is True for key in (
                    "declared_section_matched", "paraphrase_supported",
                    "applicability_not_broader",
                )
                )
            )
        else:
            verified = (
                verified
                and row.get("exact_locator_inspected") is False
                and all(
                    row.get(key) is None for key in (
                        "declared_section_matched", "paraphrase_supported",
                        "applicability_not_broader",
                    )
                )
            )
        all_verified = all_verified and verified
        if row.get("verdict") == "verified" and not verified:
            issues.append(f"verified citation lacks inspection evidence: {reference_id}")
    if result.get("approved") is True:
        missing_files = sorted(set(required_file_chunks) - inspected_files)
        if result.get("issues") != [] or not all_verified or missing_files:
            issues.append("approved review has unresolved or uninspected evidence")
    elif result.get("approved") is not False:
        issues.append("review approved field is not boolean")
    result_issue_rows = result.get("issues")
    if isinstance(result_issue_rows, list):
        for item in result_issue_rows:
            if (not isinstance(item, dict)
                    or set(item) != {"severity", "code"}
                    or item.get("severity") not in {"critical", "high", "medium", "low"}
                    or item.get("code") not in CONTROLLED_REVIEW_ISSUE_CODES):
                issues.append("review result contains uncontrolled issue content")
                break
    controlled_summary = (
        "approved_no_issues" if result.get("approved") is True
        else "rejected_with_controlled_issues"
    )
    if (not isinstance(result.get("issues"), list)
            or result.get("summary") != controlled_summary):
        issues.append("review result has malformed issues or summary")
    return issues


def replay_raw_review(
    root: Path, protocol_id: str, raw_bytes: bytes, *,
    snapshot: ReviewSnapshot | None = None, cache: ReviewCache | None = None,
) -> ReviewReplay:
    """Replay raw Codex JSONL and deterministically derive all public artifacts."""

    if protocol_id not in PROTOCOLS:
        raise ValueError(f"unknown knowledge-review protocol: {protocol_id}")
    snapshot = snapshot or freeze_review_snapshot(root, protocol_id)
    if cache is None:
        raise ValueError("raw review replay requires its retained frozen cache")
    if snapshot.protocol_id != protocol_id:
        raise ValueError("raw review replay uses a snapshot for another protocol")
    events = _strict_jsonl(raw_bytes, label="raw Codex review stream")
    event_types = [event.get("type") for event in events]
    if (len(events) < 4 or event_types[0] != "thread.started"
            or event_types[1] != "turn.started"
            or event_types[-1] != "turn.completed"):
        raise ValueError(
            "raw review stream must be one ordered thread/turn ending at turn.completed"
        )
    thread_ids: list[str] = []
    referenced_thread_ids = {
        str(event["thread_id"]) for event in events
        if isinstance(event.get("thread_id"), str) and event["thread_id"]
    }
    started_commands: dict[str, str] = {}
    completed_commands: set[str] = set()
    operations: list[dict[str, Any]] = []
    messages: list[str] = []
    turn_started = 0
    turn_completed = 0
    final_message_seen = False
    for index, event in enumerate(events, 1):
        event_type = event.get("type")
        if event_type not in _ALLOWED_EVENT_TYPES:
            raise ValueError(f"raw event {index} has forbidden or unknown type {event_type!r}")
        if event_type == "thread.started":
            thread_id = event.get("thread_id")
            if not isinstance(thread_id, str) or _TOKEN_RE.fullmatch(thread_id) is None:
                raise ValueError("raw review stream has an invalid thread ID")
            thread_ids.append(thread_id)
            continue
        if event_type == "turn.started":
            turn_started += 1
            continue
        if event_type == "turn.completed":
            turn_completed += 1
            continue
        item = event.get("item")
        if not isinstance(item, dict):
            raise ValueError(f"raw item event {index} has no object item")
        item_type = item.get("type")
        if item_type in _ALLOWED_NONCOMMAND_ITEMS:
            if event_type == "item.completed" and item_type == "agent_message":
                if final_message_seen:
                    raise ValueError("raw review stream has multiple final messages")
                text = _content_text(item)
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("completed agent message has no text")
                messages.append(text.strip())
                final_message_seen = True
            elif item_type == "agent_message":
                raise ValueError("agent message must be one completed final item")
            elif final_message_seen:
                raise ValueError("raw review stream has an item after the final message")
            continue
        if item_type != "command_execution":
            raise ValueError(f"raw item event {index} uses forbidden or unknown tool {item_type!r}")
        call_id = item.get("id") or item.get("call_id")
        if final_message_seen:
            raise ValueError("raw review stream executes a command after the final message")
        if not isinstance(call_id, str) or _CALL_ID_RE.fullmatch(call_id) is None:
            raise ValueError("command event has an invalid call ID")
        command = item.get("command")
        if not isinstance(command, str):
            raise ValueError("command event does not contain one literal command")
        if event_type == "item.started":
            if call_id in started_commands or call_id in completed_commands:
                raise ValueError("command call ID is reused")
            _split_command(command)
            started_commands[call_id] = command
            continue
        if call_id in completed_commands:
            raise ValueError("completed command call ID is reused")
        if call_id not in started_commands:
            raise ValueError("completed command has no matching start event")
        if started_commands.get(call_id) != command:
            raise ValueError("completed command differs from its start event")
        if (item.get("status") != "completed"
                or type(item.get("exit_code")) is not int
                or item["exit_code"] != 0):
            raise ValueError("review command did not complete successfully")
        _key, output = _command_output(item)
        operations.extend(_command_operations(cache, command, output))
        completed_commands.add(call_id)
    if len(thread_ids) != 1 or len(set(thread_ids)) != 1:
        raise ValueError("raw review stream must contain exactly one unique thread")
    if referenced_thread_ids and referenced_thread_ids != {thread_ids[0]}:
        raise ValueError("raw review stream mixes multiple thread identities")
    if turn_started != 1 or turn_completed != 1:
        raise ValueError("raw review stream must contain one completed turn")
    if set(started_commands) != completed_commands:
        raise ValueError("raw review stream has an incomplete command event")
    if len(messages) != 1:
        raise ValueError("raw review stream must contain exactly one final agent JSON")
    result = _strict_json_bytes(messages[0].encode("utf-8"), label="final review result")
    if not isinstance(result, dict):
        raise ValueError("final review result is not an object")
    result_issues = _review_result_issues(snapshot, cache, result, operations)
    if result_issues:
        raise ValueError("invalid final review result: " + "; ".join(result_issues))
    result_bytes = _canonical_json(result)
    raw_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    invocation_id = f"review-{protocol_id.rsplit('.', 2)[-2]}-{raw_sha256[:32]}"
    rows: list[dict[str, Any]] = []
    for sequence, operation in enumerate(operations, 1):
        rows.append({
            "schema_version": TRACE_SCHEMA_VERSION, "sequence": sequence,
            **operation,
        })
    rows.append({
        "schema_version": TRACE_SCHEMA_VERSION, "sequence": len(rows) + 1,
        "kind": "result_emit",
        "result_sha256": hashlib.sha256(result_bytes).hexdigest(),
    })
    return ReviewReplay(
        protocol_id=protocol_id, invocation_id=invocation_id,
        thread_id=thread_ids[0], raw_sha256=raw_sha256, result=result,
        result_bytes=result_bytes, trace_bytes=_canonical_jsonl(rows),
    )


def canonical_command_argv(protocol_id: str) -> list[str]:
    if protocol_id not in PROTOCOLS:
        raise ValueError(f"unknown knowledge-review protocol: {protocol_id}")
    values = ["$CODEX", "--strict-config", "-a", "never"]
    for feature in DISABLED_CODEX_FEATURES:
        values.extend(["--disable", feature])
    values.extend([
        "exec",
        "-c", f'default_permissions="{PERMISSION_PROFILE}"',
        "-c", f"permissions.{PERMISSION_PROFILE}.network.enabled=false",
        "-c", (
            f"permissions.{PERMISSION_PROFILE}.filesystem="
            '{":minimal"="read","$CACHE"="read",'
            '"$CODEX_RUNTIME"="read"}'
        ),
        "-c", 'web_search="disabled"',
        "--ignore-user-config", "--ignore-rules", "--ephemeral",
        "--json", "--color", "never", "--skip-git-repo-check", "--model",
        MODEL, "-c", f'model_reasoning_effort="{REASONING_EFFORT}"',
        "-c", f"tool_output_token_limit={TOOL_OUTPUT_TOKEN_LIMIT}",
        "--output-schema", "$ROOT/" + REVIEW_SCHEMA_PATH,
        "--cd", "$CACHE", "-",
    ])
    return values


def command_contract_sha256(protocol_id: str) -> str:
    return hashlib.sha256(_canonical_json(canonical_command_argv(protocol_id))).hexdigest()


def build_receipt(
    root: Path, replay: ReviewReplay, *, snapshot: ReviewSnapshot,
    cache: ReviewCache, prompt: bytes, boundary_contract: dict[str, Any],
) -> dict[str, Any]:
    files = PROTOCOL_FILES[replay.protocol_id]
    normalized_boundary = _validate_boundary_contract(
        boundary_contract, expected_cache_sha256=cache.sha256,
    )
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "protocol_id": replay.protocol_id,
        "invocation_id": replay.invocation_id,
        "thread_id": replay.thread_id,
        "model": MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "prompt_sha256": hashlib.sha256(prompt).hexdigest(),
        "output_schema_sha256": snapshot.output_schema_sha256,
        "review_snapshot_sha256": snapshot.sha256,
        "citation_evidence_sha256": snapshot.citation_evidence_sha256,
        "cache_manifest_sha256": cache.sha256,
        "chunk_contract_sha256": cache.manifest["chunk_contract"]["sha256"],
        "model_inspection_contract_sha256": (
            cache.manifest["inspection_contract"]["sha256"]
        ),
        "parser_contract_sha256s": cache.manifest["parser_contract_sha256s"],
        "tool_output_token_limit": TOOL_OUTPUT_TOKEN_LIMIT,
        "initial_prompt_bytes": len(prompt),
        "initial_prompt_token_upper_bound": len(prompt),
        "initial_prompt_max_bytes": MAX_INITIAL_PROMPT_BYTES,
        "official_codex_elf_sha256": OFFICIAL_CODEX_ELF_SHA256,
        "boundary_contract": normalized_boundary,
        "result_sha256": hashlib.sha256(replay.result_bytes).hexdigest(),
        "command_sha256": command_contract_sha256(replay.protocol_id),
        "raw_event_stream_sha256": replay.raw_sha256,
        "event_stream_path": files["trace"],
        "event_stream_sha256": hashlib.sha256(replay.trace_bytes).hexdigest(),
        "codex_cli_version": CODEX_CLI_VERSION,
        "completed": True,
        "exit_code": 0,
    }


def _toml(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _toml_inline_table(value: dict[str, str]) -> str:
    return "{" + ",".join(
        f"{_toml(key)}={_toml(item)}" for key, item in sorted(value.items())
    ) + "}"


def _mount_fstype(path: Path) -> str:
    candidates: list[tuple[int, str]] = []
    for line in Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines():
        left, separator, right = line.partition(" - ")
        if not separator:
            continue
        fields = left.split()
        trailing = right.split()
        if len(fields) < 5 or not trailing:
            continue
        mount = Path(fields[4].replace("\\040", " "))
        try:
            path.relative_to(mount)
        except ValueError:
            continue
        candidates.append((len(mount.parts), trailing[0]))
    if not candidates:
        raise RuntimeError("cannot identify the review checkout filesystem")
    return max(candidates)[1]


def _validate_review_cache_parent(cache_root: Path) -> str:
    """Require a private parent containing only the frozen review cache.

    The cache lives in a caller-created 0700 directory with no siblings.  This
    makes the exact cache mount independently auditable and avoids treating a
    broad parent directory as an implicit capability.
    """

    parent = cache_root.parent
    if parent == Path("/"):
        raise RuntimeError("formal review cache cannot be a direct child of root")
    _resolved_unlinked_path(parent, label="formal review cache parent")
    if not parent.is_dir():
        raise RuntimeError("formal review cache parent must be an unlinked directory")
    info = parent.stat()
    if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o700:
        raise RuntimeError("formal review cache parent must be owned by the caller with mode 0700")
    try:
        children = sorted(parent.iterdir(), key=lambda path: path.name)
    except OSError as exc:
        raise RuntimeError("formal review cache parent cannot be enumerated") from exc
    if children != [cache_root]:
        raise RuntimeError("formal review cache parent must contain only the frozen cache")
    return CACHE_PARENT_POLICY


def _validate_review_evidence_parent(raw_output: Path, stderr_path: Path) -> Path:
    """Require an empty, private directory dedicated to one invocation."""

    parent = _resolved_unlinked_path(
        raw_output.parent, label="formal review evidence parent",
    ).resolve(strict=True)
    if (parent == Path("/") or raw_output.parent.resolve(strict=True) != parent
            or stderr_path.parent.resolve(strict=True) != parent
            or raw_output.parent != stderr_path.parent):
        raise RuntimeError("formal review evidence must share one dedicated parent")
    if _mount_fstype(parent) != "ext4":
        raise RuntimeError("formal review evidence parent must live on WSL ext4")
    info = parent.lstat()
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) != 0o700):
        raise RuntimeError(
            "formal review evidence parent must be caller-owned mode 0700"
        )
    if any(parent.iterdir()):
        raise RuntimeError("formal review evidence parent must start empty")
    return parent


def _paths_overlap(left: Path, right: Path) -> bool:
    return (
        left == right or left.is_relative_to(right) or right.is_relative_to(left)
    )


def _assert_private_evidence_disjoint(
    evidence_root: Path, protected: tuple[tuple[str, Path], ...],
) -> None:
    for label, path in protected:
        if _paths_overlap(evidence_root, path):
            raise RuntimeError(
                f"private-evidence directory must be disjoint from {label}"
            )


def _assert_mutually_disjoint_roots(
    roots: Sequence[tuple[str, Path]],
) -> None:
    for index, (left_label, left) in enumerate(roots):
        for right_label, right in roots[index + 1:]:
            if _paths_overlap(left, right):
                raise RuntimeError(
                    f"{left_label} must be disjoint from {right_label}"
                )


def _freeze_runtime_manifest(codex_executable: Path) -> dict[str, Any]:
    """Hash the exact ext4 Codex+bwrap runtime without recording its host path."""

    executable = _resolved_unlinked_path(
        codex_executable, label="Codex executable",
    ).resolve(strict=True)
    executable_info = executable.lstat()
    if not stat.S_ISREG(executable_info.st_mode):
        raise RuntimeError("formal review Codex executable must be a plain file")
    runtime_root = _resolved_unlinked_path(
        executable.parent, label="Codex runtime directory",
    ).resolve(strict=True)
    if _mount_fstype(runtime_root) != "ext4":
        raise RuntimeError("formal review Codex runtime must live on WSL ext4")
    root_info = runtime_root.lstat()
    if (root_info.st_uid != os.geteuid()
            or stat.S_IMODE(root_info.st_mode) != 0o500):
        raise RuntimeError(
            "formal review Codex runtime root must be caller-owned frozen mode 0500"
        )
    try:
        runtime_children = sorted(runtime_root.iterdir(), key=lambda item: item.name)
    except OSError:
        raise RuntimeError("formal review Codex runtime cannot be enumerated") from None
    resources = runtime_root / PurePosixPath(CODEX_BWRAP_RELATIVE_PATH).parent
    bwrap = runtime_root / PurePosixPath(CODEX_BWRAP_RELATIVE_PATH)
    if (executable.name != CODEX_EXECUTABLE_RELATIVE_PATH
            or runtime_children != sorted(
                [executable, resources], key=lambda item: item.name,
            )):
        raise RuntimeError(
            "formal review Codex runtime must contain exactly codex and codex-resources"
        )
    try:
        resource_children = sorted(resources.iterdir(), key=lambda item: item.name)
    except OSError:
        raise RuntimeError("formal review Codex resources cannot be enumerated") from None
    if resource_children != [bwrap]:
        raise RuntimeError(
            "formal review Codex resources must contain exactly bundled bwrap"
        )

    entries: list[dict[str, Any]] = []
    for current, directory_names, file_names in os.walk(
        runtime_root, topdown=True, followlinks=False,
    ):
        directory_names.sort()
        file_names.sort()
        current_path = Path(current)
        current_info = current_path.lstat()
        if not stat.S_ISDIR(current_info.st_mode) or _is_link_like(current_path):
            raise RuntimeError("formal review Codex runtime contains a linked directory")
        current_mode = stat.S_IMODE(current_info.st_mode)
        if (current_info.st_uid != os.geteuid() or current_mode & 0o222
                or current_mode & 0o500 != 0o500):
            raise RuntimeError(
                "formal review Codex runtime contains an unsafe directory"
            )
        relative = current_path.relative_to(runtime_root).as_posix() or "."
        entries.append({
            "relative_path": relative,
            "kind": "dir",
            "size": 0,
            "mode": f"{current_mode:04o}",
            "sha256": hashlib.sha256(b"").hexdigest(),
        })
        for name in [*directory_names, *file_names]:
            if _is_link_like(current_path / name):
                raise RuntimeError("formal review Codex runtime contains a linked entry")
        for name in file_names:
            path = current_path / name
            before = path.lstat()
            if not stat.S_ISREG(before.st_mode):
                raise RuntimeError("formal review Codex runtime contains a special file")
            file_mode = stat.S_IMODE(before.st_mode)
            if (before.st_uid != os.geteuid() or before.st_nlink != 1
                    or file_mode & 0o222
                    or file_mode & 0o400 != 0o400):
                raise RuntimeError(
                    "formal review Codex runtime contains an unsafe file"
                )
            payload = path.read_bytes()
            after = path.lstat()
            identity_before = (
                before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
                before.st_ctime_ns, before.st_mode, before.st_nlink,
            )
            identity_after = (
                after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns,
                after.st_ctime_ns, after.st_mode, after.st_nlink,
            )
            if identity_before != identity_after or len(payload) != before.st_size:
                raise RuntimeError("formal review Codex runtime changed while hashing")
            entries.append({
                "relative_path": path.relative_to(runtime_root).as_posix(),
                "kind": "file",
                "size": len(payload),
                "mode": f"{file_mode:04o}",
                "sha256": hashlib.sha256(payload).hexdigest(),
            })
    entries.sort(key=lambda item: item["relative_path"])
    executable_relative = executable.relative_to(runtime_root).as_posix()
    if (executable_relative != CODEX_EXECUTABLE_RELATIVE_PATH
            or len(entries) != 4):
        raise RuntimeError(
            "formal review Codex runtime is not the exact codex+bwrap contract"
        )
    executable_entry = next((
        item for item in entries
        if item["kind"] == "file" and item["relative_path"] == executable_relative
    ), None)
    if executable_entry is None:
        raise RuntimeError("Codex executable is absent from its frozen runtime manifest")
    if int(str(executable_entry["mode"]), 8) & 0o100 == 0:
        raise RuntimeError("formal review Codex runtime entry is not owner-executable")
    if executable_entry["sha256"] != OFFICIAL_CODEX_ELF_SHA256:
        raise RuntimeError(
            "formal review requires the fixed official rust-v0.144.0 Linux musl ELF"
        )
    bwrap_entry = next((
        item for item in entries
        if item["kind"] == "file"
        and item["relative_path"] == CODEX_BWRAP_RELATIVE_PATH
    ), None)
    if (bwrap_entry is None
            or bwrap_entry["mode"] != "0500"
            or bwrap_entry["sha256"] != OFFICIAL_CODEX_BWRAP_SHA256):
        raise RuntimeError(
            "formal review requires the fixed official rust-v0.144.0 bundled bwrap"
        )
    resources_entry = next((
        item for item in entries
        if item["kind"] == "dir"
        and item["relative_path"] == PurePosixPath(
            CODEX_BWRAP_RELATIVE_PATH
        ).parent.as_posix()
    ), None)
    if resources_entry is None or resources_entry["mode"] != "0500":
        raise RuntimeError("formal review Codex resources directory is not frozen")
    payload: dict[str, Any] = {
        "schema_version": RUNTIME_MANIFEST_SCHEMA_VERSION,
        "ownership_policy": RUNTIME_OWNERSHIP_POLICY,
        "executable_relative_path": executable_relative,
        "executable_sha256": executable_entry["sha256"],
        "bubblewrap_relative_path": CODEX_BWRAP_RELATIVE_PATH,
        "bubblewrap_sha256": bwrap_entry["sha256"],
        "entries": entries,
    }
    payload["sha256"] = hashlib.sha256(_canonical_json(payload)).hexdigest()
    return payload


def _validate_runtime_manifest(manifest: Any) -> dict[str, Any]:
    if not isinstance(manifest, dict) or set(manifest) != {
        "schema_version", "ownership_policy", "executable_relative_path",
        "executable_sha256", "bubblewrap_relative_path",
        "bubblewrap_sha256", "entries", "sha256",
    }:
        raise ValueError("boundary contract has a malformed runtime manifest")
    entries = manifest.get("entries")
    if (manifest.get("schema_version") != RUNTIME_MANIFEST_SCHEMA_VERSION
            or not isinstance(entries, list) or not entries):
        raise ValueError("boundary contract has a malformed runtime manifest")
    normalized_entries: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {
            "relative_path", "kind", "size", "mode", "sha256",
        }:
            raise ValueError("boundary contract has a malformed runtime entry")
        relative = entry.get("relative_path")
        kind = entry.get("kind")
        size = entry.get("size")
        mode = entry.get("mode")
        digest = entry.get("sha256")
        if not isinstance(relative, str) or not relative:
            raise ValueError("boundary runtime entry lacks a relative path")
        if relative != ".":
            path = PurePosixPath(relative)
            if (path.is_absolute() or "\\" in relative
                    or path.as_posix() != relative
                    or any(part in {"", ".", ".."} for part in path.parts)):
                raise ValueError("boundary runtime entry is not path-relative")
        if (kind not in {"dir", "file"}
                or not isinstance(size, int) or isinstance(size, bool) or size < 0
                or not isinstance(mode, str) or re.fullmatch(r"[0-7]{4}", mode) is None
                or not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None):
            raise ValueError("boundary contract has a malformed runtime entry")
        parsed_mode = int(mode, 8)
        if parsed_mode & 0o222:
            raise ValueError("boundary runtime entry is writable")
        if kind == "dir" and (
            size != 0 or digest != hashlib.sha256(b"").hexdigest()
        ):
            raise ValueError("boundary runtime directory entry is not canonical")
        if kind == "dir" and parsed_mode & 0o500 != 0o500:
            raise ValueError("boundary runtime directory is not owner-readable/traversable")
        if kind == "file" and parsed_mode & 0o400 == 0:
            raise ValueError("boundary runtime file is not owner-readable")
        normalized_entries.append(dict(entry))
    expected_order = sorted(normalized_entries, key=lambda item: item["relative_path"])
    if normalized_entries != expected_order or len({
        item["relative_path"] for item in normalized_entries
    }) != len(normalized_entries):
        raise ValueError("boundary runtime manifest is not uniquely sorted")
    root_entries = [item for item in normalized_entries if item["relative_path"] == "."]
    if (len(root_entries) != 1 or root_entries[0]["kind"] != "dir"
            or root_entries[0]["mode"] != "0500"):
        raise ValueError("boundary runtime manifest lacks its canonical root")
    entries_by_path = {item["relative_path"]: item for item in normalized_entries}
    for relative, entry in entries_by_path.items():
        if relative == ".":
            continue
        parent = PurePosixPath(relative).parent.as_posix()
        parent_entry = entries_by_path.get(parent)
        if parent_entry is None or parent_entry["kind"] != "dir":
            raise ValueError("boundary runtime manifest has an incomplete directory tree")
    executable_relative = manifest.get("executable_relative_path")
    executable_sha256 = manifest.get("executable_sha256")
    executable_entry = entries_by_path.get(str(executable_relative))
    bubblewrap_relative = manifest.get("bubblewrap_relative_path")
    bubblewrap_sha256 = manifest.get("bubblewrap_sha256")
    bubblewrap_entry = entries_by_path.get(str(bubblewrap_relative))
    resources_relative = PurePosixPath(CODEX_BWRAP_RELATIVE_PATH).parent.as_posix()
    if (manifest.get("ownership_policy") != RUNTIME_OWNERSHIP_POLICY
            or executable_relative != CODEX_EXECUTABLE_RELATIVE_PATH
            or len(normalized_entries) != 4
            or set(entries_by_path) != {
                ".", CODEX_EXECUTABLE_RELATIVE_PATH,
                resources_relative, CODEX_BWRAP_RELATIVE_PATH,
            }
            or executable_entry is None or executable_entry["kind"] != "file"
            or executable_entry["mode"] != "0500"
            or not isinstance(executable_sha256, str)
            or _SHA256_RE.fullmatch(executable_sha256) is None
            or executable_entry["sha256"] != executable_sha256
            or executable_sha256 != OFFICIAL_CODEX_ELF_SHA256
            or bubblewrap_relative != CODEX_BWRAP_RELATIVE_PATH
            or not isinstance(bubblewrap_sha256, str)
            or _SHA256_RE.fullmatch(bubblewrap_sha256) is None
            or bubblewrap_entry is None or bubblewrap_entry["kind"] != "file"
            or bubblewrap_entry["mode"] != "0500"
            or bubblewrap_entry["sha256"] != bubblewrap_sha256
            or bubblewrap_sha256 != OFFICIAL_CODEX_BWRAP_SHA256
            or entries_by_path[resources_relative]["kind"] != "dir"
            or entries_by_path[resources_relative]["mode"] != "0500"):
        raise ValueError("boundary runtime manifest lacks its exact Codex+bwrap identity")
    payload = {
        "schema_version": RUNTIME_MANIFEST_SCHEMA_VERSION,
        "ownership_policy": RUNTIME_OWNERSHIP_POLICY,
        "executable_relative_path": executable_relative,
        "executable_sha256": executable_sha256,
        "bubblewrap_relative_path": bubblewrap_relative,
        "bubblewrap_sha256": bubblewrap_sha256,
        "entries": normalized_entries,
    }
    digest = hashlib.sha256(_canonical_json(payload)).hexdigest()
    if manifest.get("sha256") != digest:
        raise ValueError("boundary runtime manifest hash is invalid")
    return {**payload, "sha256": digest}


def _build_boundary_contract(
    *, runtime_manifest: dict[str, Any], cache_manifest_sha256: str,
    cache_parent_policy: str, evidence_parent_policy: str,
    canary_results: dict[str, bool],
) -> dict[str, Any]:
    manifest = _validate_runtime_manifest(runtime_manifest)
    expected_canaries = {
        "cache_read", "runtime_read", "checkout_denied", "auth_denied",
        "external_denied", "peer_sibling_denied", "evidence_denied",
        "cache_write_denied",
    }
    if (not isinstance(cache_manifest_sha256, str)
            or _SHA256_RE.fullmatch(cache_manifest_sha256) is None
            or cache_parent_policy != CACHE_PARENT_POLICY
            or evidence_parent_policy != EVIDENCE_PARENT_POLICY
            or set(canary_results) != expected_canaries
            or any(value is not True for value in canary_results.values())):
        raise ValueError("cannot build an incomplete review boundary contract")
    payload: dict[str, Any] = {
        "schema_version": BOUNDARY_CONTRACT_SCHEMA_VERSION,
        "policy": BOUNDARY_POLICY,
        "filesystem_allowlist": [
            {"token": ":minimal", "access": "read"},
            {"token": "$CACHE", "access": "read"},
            {"token": "$CODEX_RUNTIME", "access": "read"},
        ],
        "network_enabled": False,
        "initial_process_path": list(RUNTIME_INITIAL_PATH_TOKENS),
        "runtime_manifest": manifest,
        "cache_manifest_sha256": cache_manifest_sha256,
        "cache_parent_policy": CACHE_PARENT_POLICY,
        "evidence_parent_policy": EVIDENCE_PARENT_POLICY,
        "canary_results": dict(sorted(canary_results.items())),
    }
    payload["contract_sha256"] = hashlib.sha256(_canonical_json(payload)).hexdigest()
    return payload


def _validate_boundary_contract(
    contract: Any, *, expected_cache_sha256: str,
) -> dict[str, Any]:
    if not isinstance(contract, dict) or set(contract) != {
        "schema_version", "policy", "filesystem_allowlist", "network_enabled",
        "initial_process_path",
        "runtime_manifest", "cache_manifest_sha256", "cache_parent_policy",
        "evidence_parent_policy", "canary_results", "contract_sha256",
    }:
        raise ValueError("receipt has a malformed boundary contract")
    rebuilt = _build_boundary_contract(
        runtime_manifest=contract.get("runtime_manifest"),
        cache_manifest_sha256=str(contract.get("cache_manifest_sha256", "")),
        cache_parent_policy=str(contract.get("cache_parent_policy", "")),
        evidence_parent_policy=str(contract.get("evidence_parent_policy", "")),
        canary_results=contract.get("canary_results", {}),
    )
    if (contract.get("schema_version") != BOUNDARY_CONTRACT_SCHEMA_VERSION
            or contract.get("policy") != BOUNDARY_POLICY
            or contract.get("filesystem_allowlist")
            != rebuilt["filesystem_allowlist"]
            or contract.get("network_enabled") is not False
            or contract.get("initial_process_path")
            != rebuilt["initial_process_path"]
            or contract.get("cache_manifest_sha256") != expected_cache_sha256
            or contract != rebuilt):
        raise ValueError("receipt boundary contract is not canonical or is stale")
    return rebuilt


def _runtime_initial_process_path(runtime_root: Path) -> str:
    """Expand the pathless receipt contract for the Linux-only subprocess."""

    root = runtime_root.as_posix()
    return ":".join(
        token.replace("$CODEX_RUNTIME", root)
        for token in RUNTIME_INITIAL_PATH_TOKENS
    )


def _official_boundary(
    root: Path, cache_root: Path | None = None,
    external_canary_root: Path | None = None,
    *, codex_executable: Path | None = None,
    cache_manifest_sha256: str | None = None,
    peer_sibling_canary: Path | None = None,
    evidence_canary: Path | None = None,
) -> tuple[dict[str, str], list[str], dict[str, Any]]:
    if _formal_host_is_windows() or sys.platform != "linux":
        raise RuntimeError("formal knowledge review is Linux/WSL2-only; Windows is NO-GO")
    from eval.agent_ab.common import (
        official_process_environment, require_official_linux_wsl2,
    )
    require_official_linux_wsl2()
    resolved = root.resolve(strict=True)
    if str(resolved).startswith("/mnt/") or _mount_fstype(resolved) != "ext4":
        raise RuntimeError("formal review checkout must be on WSL ext4, not drvfs/overlay")
    codex_home_text = os.environ.get("CODEX_HOME", "")
    if not codex_home_text:
        raise RuntimeError("formal review requires a dedicated CODEX_HOME")
    codex_home = _resolved_unlinked_path(
        Path(codex_home_text), label="dedicated CODEX_HOME",
    ).resolve(strict=True)
    if _paths_overlap(resolved, codex_home):
        raise RuntimeError("review checkout must be disjoint from CODEX_HOME")
    auth_path = codex_home / "auth.json"
    if not auth_path.is_file() or _is_link_like(auth_path):
        raise RuntimeError("dedicated CODEX_HOME lacks a plain auth.json")
    if cache_root is None:
        raise RuntimeError("formal review requires an external frozen cache")
    resolved_cache = cache_root.resolve(strict=True)
    try:
        resolved_cache.relative_to(resolved)
    except ValueError:
        pass
    else:
        raise RuntimeError("formal review cache must be outside the checkout")
    if _mount_fstype(resolved_cache) != "ext4":
        raise RuntimeError("formal review cache must live on WSL ext4")
    cache_parent_policy = _validate_review_cache_parent(resolved_cache)
    if codex_executable is None:
        raise RuntimeError("formal review requires a resolved Codex executable")
    runtime_manifest = _freeze_runtime_manifest(codex_executable)
    if runtime_manifest != _freeze_runtime_manifest(codex_executable):
        raise RuntimeError("formal review Codex runtime is not stable")
    resolved_codex = _resolved_unlinked_path(
        codex_executable, label="Codex executable",
    ).resolve(strict=True)
    runtime_root = resolved_codex.parent
    for label, path in (
        ("checkout", resolved), ("cache", resolved_cache),
        ("CODEX_HOME", codex_home),
    ):
        if _paths_overlap(runtime_root, path):
            raise RuntimeError(f"Codex runtime must be disjoint from {label}")
    if _paths_overlap(resolved_cache, codex_home):
        raise RuntimeError("formal review cache must be disjoint from CODEX_HOME")
    if cache_manifest_sha256 is None or _SHA256_RE.fullmatch(
        cache_manifest_sha256,
    ) is None:
        raise RuntimeError("formal review cache manifest hash is missing")
    if external_canary_root is None:
        raise RuntimeError("formal review requires an external privacy canary")
    external_path = external_canary_root.resolve(strict=True) / "canary.txt"
    if not external_path.is_file() or _is_link_like(external_path):
        raise RuntimeError("external review privacy canary is not a plain file")
    if peer_sibling_canary is None:
        raise RuntimeError("formal review requires a peer-sibling privacy canary")
    peer_path = peer_sibling_canary.resolve(strict=True)
    if not peer_path.is_file() or _is_link_like(peer_path):
        raise RuntimeError("peer-sibling review privacy canary is not a plain file")
    if evidence_canary is None:
        raise RuntimeError("formal review requires a private-evidence privacy canary")
    evidence_path = evidence_canary.resolve(strict=True)
    if not evidence_path.is_file() or _is_link_like(evidence_path):
        raise RuntimeError("private-evidence review canary is not a plain file")
    evidence_root = evidence_path.parent
    _assert_private_evidence_disjoint(evidence_root, (
        ("checkout", resolved), ("cache", resolved_cache),
        ("Codex runtime", runtime_root), ("CODEX_HOME", codex_home),
        ("external canary", external_path.parent),
        ("peer-sibling canary", peer_path.parent),
    ))
    for label, path in (
        ("external", external_path), ("peer sibling", peer_path),
        ("private evidence", evidence_path),
    ):
        if any(_paths_overlap(path, allowed) for allowed in (resolved_cache, runtime_root)):
            raise RuntimeError(f"{label} privacy canary overlaps the allowlist")
    filesystem = {
        ":minimal": "read",
        str(resolved_cache): "read",
        str(runtime_root): "read",
    }
    profile_values = [
        f"default_permissions={_toml(PERMISSION_PROFILE)}",
        f"permissions.{PERMISSION_PROFILE}.network.enabled=false",
        f"permissions.{PERMISSION_PROFILE}.filesystem=" + _toml_inline_table(filesystem),
        'web_search="disabled"',
    ]
    environment = official_process_environment()
    environment["PATH"] = _runtime_initial_process_path(runtime_root)
    return environment, profile_values, {
        "codex_home": str(codex_home),
        "checkout_root": str(resolved),
        "cache_root": str(resolved_cache),
        "cache_parent_policy": cache_parent_policy,
        "cache_manifest_sha256": cache_manifest_sha256,
        "runtime_root": str(runtime_root),
        "runtime_probe": str(resolved_codex),
        "runtime_manifest": runtime_manifest,
        "auth_probe": str(auth_path),
        "external_probe": str(external_path),
        "peer_sibling_probe": str(peer_path),
        "evidence_probe": str(evidence_path),
    }


def _sandbox_prefix(
    codex: str, root: Path, profile_values: list[str],
) -> list[str]:
    command = [codex]
    for value in profile_values:
        command.extend(["-c", value])
    command.extend([
        "sandbox", "-P", PERMISSION_PROFILE,
        "--sandbox-state-disable-network", "-C", str(root.resolve()),
    ])
    return command


def _run_canary(command: list[str], environment: dict[str, str]) -> int:
    completed = subprocess.run(
        command, env=environment, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, timeout=60, check=False,
    )
    return completed.returncode


def _verify_boundary_canaries(
    *, codex: str, root: Path, cache_root: Path, profile_values: list[str],
    boundary: dict[str, Any], environment: dict[str, str],
) -> dict[str, bool]:
    """Prove all allowlist reads and default-deny failures in one sandbox."""

    prefix = _sandbox_prefix(codex, cache_root, profile_values)
    python = shutil.which("python3", path=environment.get("PATH"))
    if not python:
        raise RuntimeError("official review boundary canary requires python3")
    allowed = cache_root / CACHE_MANIFEST_NAME
    runtime_probe = Path(str(boundary["runtime_probe"]))
    probes = [
        ("cache_read", allowed),
        ("runtime_read", runtime_probe),
        ("checkout_denied", root / "README.md"),
        ("auth_denied", Path(str(boundary["auth_probe"]))),
        ("external_denied", Path(str(boundary["external_probe"]))),
        ("peer_sibling_denied", Path(str(boundary["peer_sibling_probe"]))),
        ("evidence_denied", Path(str(boundary["evidence_probe"]))),
    ]
    for result_name, target in probes:
        if not target.is_file():
            raise RuntimeError(f"review boundary probe is not a plain file: {result_name}")
    write_target = cache_root / ".hlsgraph-review-write-canary"
    # Exit bits identify a failed expectation without printing any host path or
    # file bytes.  Bits 0-1 mean an allowed read failed; bits 2-6 mean a
    # default-denied read succeeded; bit 7 means the read-only cache was
    # writable.  One sandbox instance avoids repeated startup cost and closes
    # the boundary drift window between independent probes.
    probe_script = (
        "import pathlib,sys;"
        "p=[pathlib.Path(x) for x in sys.argv[1:]];"
        "r=lambda x:(x.open('rb').read(1),True)[1];"
        "bits=0;"
        "\nfor i,x in enumerate(p[:7]):"
        "\n try: ok=r(x)"
        "\n except OSError: ok=False"
        "\n if (i<2 and not ok) or (i>=2 and ok): bits|=1<<i"
        "\ntry: p[7].write_bytes(b'x'); bits|=1<<7"
        "\nexcept OSError: pass"
        "\nraise SystemExit(bits)"
    )
    try:
        try:
            failed_bits = _run_canary(
                [
                    *prefix, python, "-I", "-c", probe_script,
                    *(str(target) for _name, target in probes),
                    str(write_target),
                ],
                environment,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("review sandbox boundary canary timed out") from exc
    finally:
        if write_target.exists():
            write_target.unlink()
    if failed_bits:
        names = [name for index, (name, _target) in enumerate(probes)
                 if failed_bits & (1 << index)]
        if failed_bits & (1 << 7):
            names.append("cache_write_denied")
        if failed_bits & ~0xFF:
            names.append("unexpected_exit_bits")
        raise RuntimeError(
            "review sandbox boundary canary failed: " + ", ".join(names)
        )
    return {
        "cache_read": True,
        "runtime_read": True,
        "checkout_denied": True,
        "auth_denied": True,
        "external_denied": True,
        "peer_sibling_denied": True,
        "evidence_denied": True,
        "cache_write_denied": True,
    }


def _actual_command(
    root: Path, cache_root: Path, protocol_id: str, codex_command: str,
    profile_values: list[str],
) -> list[str]:
    command = [codex_command, "--strict-config", "-a", "never"]
    for feature in DISABLED_CODEX_FEATURES:
        command.extend(["--disable", feature])
    command.append("exec")
    for value in profile_values:
        command.extend(["-c", value])
    command.extend([
        "--ignore-user-config", "--ignore-rules", "--ephemeral", "--json",
        "--color", "never", "--skip-git-repo-check", "--model", MODEL,
        "-c", f'model_reasoning_effort="{REASONING_EFFORT}"',
        "-c", f"tool_output_token_limit={TOOL_OUTPUT_TOKEN_LIMIT}",
        "--output-schema", str((root / REVIEW_SCHEMA_PATH).resolve()),
        "--cd", str(cache_root.resolve()), "-",
    ])
    return command


def _terminate(process: subprocess.Popen[bytes]) -> None:
    try:
        if os.name == "nt":
            process.terminate()
        else:
            os.killpg(process.pid, signal.SIGTERM)
    except (OSError, ProcessLookupError):
        return


def _redact_cache_payloads(data: bytes, cache: ReviewCache) -> bytes:
    result = data
    for entry in cache.manifest["citations"]:
        relatives = [entry.get("body_path"), entry.get("inspection_path")]
        relatives.extend(
            chunk.get("path") for chunk in entry.get("inspection_chunks", [])
            if isinstance(chunk, dict)
        )
        relatives.extend(
            artifact.get("body_path")
            for artifact in entry.get("resolver_artifacts", [])
            if isinstance(artifact, dict)
        )
        for relative in relatives:
            if not relative:
                continue
            payload = (cache.root / PurePosixPath(str(relative))).read_bytes()
            if payload and payload in result:
                marker = (
                    f"[redacted-cache-{hashlib.sha256(payload).hexdigest()}]"
                ).encode("ascii")
                result = result.replace(payload, marker)
    return result


def _publish_artifacts(root: Path, artifacts: dict[str, bytes]) -> None:
    targets = {relative: root / PurePosixPath(relative) for relative in artifacts}
    existing = [relative for relative, path in targets.items() if path.exists()]
    if existing:
        raise RuntimeError(f"refusing to overwrite review artifacts: {sorted(existing)!r}")
    docs = root / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    published: list[Path] = []
    with tempfile.TemporaryDirectory(prefix=".hlsgraph-review-stage-", dir=docs) as temp:
        staging = Path(temp).resolve(strict=True)
        staging.relative_to(root.resolve(strict=True))
        staged: dict[str, Path] = {}
        for index, (relative, data) in enumerate(sorted(artifacts.items())):
            path = staging / f"{index:02d}.artifact"
            path.write_bytes(data)
            staged[relative] = path
        try:
            for relative in sorted(staged):
                target = targets[relative]
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(staged[relative], target)
                published.append(target)
        except BaseException:
            for path in published:
                try:
                    path.unlink()
                except OSError:
                    pass
            raise


def preflight_review(root: Path, protocol_id: str) -> dict[str, Any]:
    """Build the real frozen input contract without network or model execution."""

    resolved = root.resolve(strict=True)
    if resolved != SCRIPT_ROOT:
        raise RuntimeError("review runner must execute from its own checkout root")
    snapshot = freeze_review_snapshot(resolved, protocol_id)
    prompt = build_review_prompt(resolved, protocol_id, snapshot=snapshot)
    return {
        "protocol_id": protocol_id,
        "review_snapshot_sha256": snapshot.sha256,
        "prompt_contract_sha256": hashlib.sha256(prompt).hexdigest(),
        "review_surface_sha256": snapshot.surfaces,
        "implementation_surface_sha256": snapshot.implementation_surface_sha256,
        "citation_audit_sha256": snapshot.citation_audit_sha256,
        "output_schema_sha256": snapshot.output_schema_sha256,
        "required_file_count": len(snapshot.files),
        "exact_citation_url_count": len(snapshot.exact_citation_urls),
        "network_used": False,
        "model_used": False,
    }


def run_review(
    root: Path, protocol_id: str, raw_output: Path, cache_root: Path, *,
    codex_command: str, timeout_seconds: int,
    fetcher: Callable[[str, float, int], TrustedFetch] = _default_fetch,
    fetch_timeout_seconds: float = 60.0,
    pdf_text_extractor: Callable[[bytes], TextDerivation | None] | None = None,
    pdftotext_command: str | None = None,
    pdftotext_sha256: str | None = None,
) -> ReviewReplay:
    """Execute one review and atomically derive its three public artifacts."""

    if _formal_host_is_windows():
        raise RuntimeError("formal knowledge review is Linux/WSL2-only; Windows is NO-GO")

    lexical_root = root.absolute()
    if lexical_root.is_symlink():
        raise RuntimeError("formal review checkout must not be a symlink")
    root = lexical_root.resolve(strict=True)
    if root != SCRIPT_ROOT:
        raise RuntimeError("formal review runner must belong to the reviewed checkout")
    raw_output = raw_output.absolute()
    cache_root = cache_root.absolute()
    stderr_path = raw_output.with_suffix(raw_output.suffix + ".stderr.log")
    resolved_raw = _resolved_unlinked_path(raw_output, label="raw Codex stream")
    resolved_stderr = _resolved_unlinked_path(stderr_path, label="review stderr")
    resolved_cache = _resolved_unlinked_path(cache_root, label="review cache")
    try:
        resolved_raw.relative_to(root)
    except ValueError:
        pass
    else:
        raise RuntimeError("raw Codex stream must stay outside the public checkout")
    try:
        resolved_cache.relative_to(root)
    except ValueError:
        pass
    else:
        raise RuntimeError("review cache must stay outside the public checkout")
    for evidence_path, label in (
        (resolved_raw, "raw Codex stream"),
        (resolved_stderr, "review stderr"),
    ):
        try:
            evidence_path.relative_to(resolved_cache)
        except ValueError:
            pass
        else:
            raise RuntimeError(f"{label} must stay outside the review cache")
        try:
            resolved_cache.relative_to(evidence_path)
        except ValueError:
            pass
        else:
            raise RuntimeError(f"review cache must stay outside the {label} path")
    if raw_output.exists():
        raise RuntimeError("raw Codex stream path already exists")
    if cache_root.exists():
        raise RuntimeError("review cache path already exists")
    if stderr_path.exists():
        raise RuntimeError("review stderr path already exists")
    evidence_parent = _validate_review_evidence_parent(raw_output, stderr_path)
    canary_handle = tempfile.TemporaryDirectory(
        prefix="hlsgraph-knowledge-review-boundary-", dir="/tmp",
    )
    canary_root = Path(canary_handle.name)
    canary_path = canary_root / "canary.txt"
    canary_bytes = os.urandom(48)
    canary_path.write_bytes(canary_bytes)
    peer_handle = tempfile.TemporaryDirectory(
        prefix="hlsgraph-knowledge-review-peer-",
        dir=str(cache_root.parent.parent),
    )
    peer_path = Path(peer_handle.name) / "canary.txt"
    peer_bytes = os.urandom(48)
    peer_path.write_bytes(peer_bytes)
    evidence_path = evidence_parent / ".hlsgraph-review-evidence-canary"
    evidence_bytes = os.urandom(48)
    try:
        _write_private(evidence_path, evidence_bytes)
        from eval.agent_ab.common import (
            _resolve_executable, official_process_environment,
            require_official_linux_wsl2,
        )
        require_official_linux_wsl2()
        environment = official_process_environment()
        resolved_codex = str(_resolve_executable(codex_command, "Codex CLI"))
        version = subprocess.run(
            [resolved_codex, "--version"], env=environment, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15,
        ).stdout.decode("utf-8", errors="strict").strip()
        if version != CODEX_CLI_VERSION:
            raise RuntimeError(
                f"formal review requires {CODEX_CLI_VERSION!r}, found {version!r}"
            )
        clean = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=str(root), env=environment, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, timeout=30, check=False,
        )
        if clean.returncode != 0 or clean.stdout:
            raise RuntimeError("formal review checkout must be a clean committed candidate")
        snapshot = freeze_review_snapshot(root, protocol_id)
        cache = create_review_cache(
            root, snapshot, cache_root, fetcher=fetcher,
            timeout_seconds=fetch_timeout_seconds,
            pdf_text_extractor=pdf_text_extractor,
            pdftotext_command=pdftotext_command,
            pdftotext_sha256=pdftotext_sha256,
        )
        cache = load_review_cache(cache.root, snapshot)
        environment, profile, boundary = _official_boundary(
            root, cache.root, canary_root,
            codex_executable=Path(resolved_codex),
            cache_manifest_sha256=cache.sha256,
            peer_sibling_canary=peer_path,
            evidence_canary=evidence_path,
        )
        canary_results = _verify_boundary_canaries(
            codex=resolved_codex, root=root, cache_root=cache.root,
            profile_values=profile,
            boundary=boundary, environment=environment,
        )
        boundary_contract = _build_boundary_contract(
            runtime_manifest=boundary["runtime_manifest"],
            cache_manifest_sha256=cache.sha256,
            cache_parent_policy=boundary["cache_parent_policy"],
            evidence_parent_policy=EVIDENCE_PARENT_POLICY,
            canary_results=canary_results,
        )
        command = _actual_command(
            root, cache.root, protocol_id, resolved_codex, profile,
        )
        prompt = build_review_prompt(
            root, protocol_id, snapshot=snapshot, cache=cache,
        )
        process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, cwd=str(cache.root), env=environment,
            start_new_session=True,
        )
        try:
            stdout_bytes, stderr_bytes = process.communicate(
                prompt, timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            _terminate(process)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
            raise RuntimeError("knowledge review timed out") from exc
        stderr_bytes = _redact_cache_payloads(stderr_bytes, cache)
        _ensure_private_parent(raw_output.parent)
        if process.returncode != 0:
            _write_private(stderr_path, stderr_bytes)
            raise RuntimeError(
                f"Codex review failed with exit code {process.returncode}; see {stderr_path}"
            )
        sanitized_raw = sanitize_raw_review_stream(stdout_bytes, cache)
        post_snapshot = freeze_review_snapshot(root, protocol_id)
        post_cache = load_review_cache(cache.root, post_snapshot)
        post_prompt = build_review_prompt(
            root, protocol_id, snapshot=post_snapshot, cache=post_cache,
        )
        if (canary_path.read_bytes() != canary_bytes
                or peer_path.read_bytes() != peer_bytes
                or evidence_path.read_bytes() != evidence_bytes
                or post_snapshot != snapshot or post_cache.manifest_bytes != cache.manifest_bytes
                or post_prompt != prompt
                or _freeze_runtime_manifest(Path(resolved_codex))
                != boundary["runtime_manifest"]
                or _validate_review_cache_parent(cache.root)
                != boundary["cache_parent_policy"]):
            raise RuntimeError("review boundary or source bytes changed during invocation")
        replay = replay_raw_review(
            root, protocol_id, sanitized_raw, snapshot=snapshot, cache=cache,
        )
        _write_private(raw_output, sanitized_raw)
        _write_private(stderr_path, stderr_bytes)
        files = PROTOCOL_FILES[protocol_id]
        artifacts = {
            files["result"]: replay.result_bytes,
            files["trace"]: replay.trace_bytes,
        }
        receipt = build_receipt(
            root, replay, snapshot=snapshot, cache=cache, prompt=prompt,
            boundary_contract=boundary_contract,
        )
        artifacts[files["receipt"]] = _canonical_json(receipt)
        _publish_artifacts(root, artifacts)
        return replay
    finally:
        evidence_path.unlink(missing_ok=True)
        peer_handle.cleanup()
        canary_handle.cleanup()


def _receipt_projection(receipt: dict[str, Any], receipt_bytes: bytes) -> dict[str, Any]:
    keys = (
        "protocol_id", "invocation_id", "thread_id", "model",
        "reasoning_effort", "prompt_sha256", "output_schema_sha256",
        "review_snapshot_sha256", "citation_evidence_sha256",
        "cache_manifest_sha256", "chunk_contract_sha256",
        "model_inspection_contract_sha256",
        "parser_contract_sha256s", "tool_output_token_limit",
        "initial_prompt_bytes", "initial_prompt_token_upper_bound",
        "initial_prompt_max_bytes",
        "official_codex_elf_sha256",
        "result_sha256", "command_sha256", "event_stream_path",
        "raw_event_stream_sha256", "event_stream_sha256", "codex_cli_version",
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


def review_source_hashes(
    root: Path, surfaces: dict[str, str], implementation_sha256: str,
) -> dict[str, str]:
    required = {
        REVIEW_SCHEMA_PATH, REVIEW_RECEIPT_SCHEMA_PATH,
        CITATION_AUDIT_PATH, CITATION_EVIDENCE_PATH,
        CITATION_EVIDENCE_SCHEMA_PATH, CITATION_GENERATOR_PATH, RUNNER_PATH,
        SURFACE_HELPER_PATH, RELEASE_AUDITOR_PATH,
        *(item[key] for item in PROTOCOL_FILES.values()
          for key in ("prompt", "result", "trace", "receipt")),
        *SUITE_REVIEW_SOURCE_PATHS,
    }
    hashes = {
        relative: hashlib.sha256(
            (root / PurePosixPath(relative)).read_bytes()
        ).hexdigest()
        for relative in sorted(required)
    }
    helper_hash = hashes.pop(SURFACE_HELPER_PATH)
    hashes[SURFACE_HELPER_HASH_KEY] = helper_hash
    hashes[IMPLEMENTATION_SURFACE_HASH_KEY] = implementation_sha256
    pack_root = root / "src" / "hlsgraph" / "knowledge" / "packs"
    for path in sorted(pack_root.glob("*.json")):
        data = path.read_bytes()
        pack_id, surface, _payload = _semantic_pack_projection(
            data, label=f"knowledge pack {path.name}",
        )
        if surfaces.get(pack_id) != surface:
            raise ValueError(f"knowledge pack surface changed before sealing: {pack_id}")
        hashes[
            PACK_SURFACE_HASH_PREFIX + path.name + PACK_SURFACE_HASH_SUFFIX
        ] = surface
    return hashes


def _atomic_replace_pack_bytes(root: Path, updates: dict[Path, bytes]) -> None:
    pack_root = root / "src" / "hlsgraph" / "knowledge" / "packs"
    originals = {path: path.read_bytes() for path in updates}
    with tempfile.TemporaryDirectory(prefix=".hlsgraph-seal-stage-", dir=pack_root) as temp:
        stage_root = Path(temp).resolve(strict=True)
        stage_root.relative_to(root.resolve(strict=True))
        staged: dict[Path, Path] = {}
        for index, (target, data) in enumerate(sorted(
            updates.items(), key=lambda item: item[0].name,
        )):
            stage = stage_root / f"{index:02d}.json"
            stage.write_bytes(data)
            staged[target] = stage
        replaced: list[Path] = []
        try:
            for target in sorted(staged, key=lambda item: item.name):
                os.replace(staged[target], target)
                replaced.append(target)
        except BaseException:
            for target in replaced:
                recovery = stage_root / (target.name + ".recovery")
                recovery.write_bytes(originals[target])
                os.replace(recovery, target)
            raise


def seal_review_attestations(
    root: Path, *, semantic_raw: Path, adversarial_raw: Path,
    semantic_cache: Path, adversarial_cache: Path,
) -> None:
    """Verify two retained invocations and deterministically seal all three packs."""

    if _formal_host_is_windows():
        raise RuntimeError("formal knowledge review sealing is Linux/WSL2-only; Windows is NO-GO")

    root = root.resolve(strict=True)
    if root != SCRIPT_ROOT:
        raise RuntimeError("review sealer must belong to the reviewed checkout")
    semantic_raw_parent = _resolved_unlinked_path(
        semantic_raw.absolute().parent, label="semantic raw evidence parent",
    ).resolve(strict=True)
    adversarial_raw_parent = _resolved_unlinked_path(
        adversarial_raw.absolute().parent, label="adversarial raw evidence parent",
    ).resolve(strict=True)
    semantic_cache_root = _resolved_unlinked_path(
        semantic_cache.absolute(), label="semantic review cache",
    ).resolve(strict=True)
    adversarial_cache_root = _resolved_unlinked_path(
        adversarial_cache.absolute(), label="adversarial review cache",
    ).resolve(strict=True)
    protected_roots: list[tuple[str, Path]] = [
        ("review checkout", root),
        ("semantic raw evidence", semantic_raw_parent),
        ("adversarial raw evidence", adversarial_raw_parent),
        ("semantic review cache", semantic_cache_root),
        ("adversarial review cache", adversarial_cache_root),
    ]
    if os.name != "nt":
        codex_home_text = os.environ.get("CODEX_HOME", "")
        if not codex_home_text:
            raise RuntimeError("formal review sealing requires the dedicated CODEX_HOME")
        codex_home = _resolved_unlinked_path(
            Path(codex_home_text), label="dedicated CODEX_HOME",
        ).resolve(strict=True)
        protected_roots.append(("dedicated CODEX_HOME", codex_home))
    _assert_mutually_disjoint_roots(protected_roots)

    invocations: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    snapshots: list[ReviewSnapshot] = []
    runtime_manifests: list[dict[str, Any]] = []
    inputs = (
        (SEMANTIC_PROTOCOL, semantic_raw, semantic_cache),
        (ADVERSARIAL_PROTOCOL, adversarial_raw, adversarial_cache),
    )
    for protocol_id, raw_path, cache_path in inputs:
        snapshot = freeze_review_snapshot(root, protocol_id)
        cache = load_review_cache(cache_path, snapshot)
        prompt = build_review_prompt(
            root, protocol_id, snapshot=snapshot, cache=cache,
        )
        raw_bytes = _read_stable_restricted_file(
            raw_path, label=f"{protocol_id} raw review evidence",
            file_mode=0o600, parent_mode=0o700,
            max_bytes=MAX_RAW_REVIEW_BYTES,
        )
        replay = replay_raw_review(
            root, protocol_id, raw_bytes, snapshot=snapshot, cache=cache,
        )
        files = PROTOCOL_FILES[protocol_id]
        if (root / files["result"]).read_bytes() != replay.result_bytes:
            raise ValueError(f"{protocol_id} result differs from retained raw replay")
        if (root / files["trace"]).read_bytes() != replay.trace_bytes:
            raise ValueError(f"{protocol_id} trace differs from retained raw replay")
        receipt_path = root / files["receipt"]
        receipt_bytes = receipt_path.read_bytes()
        receipt = _strict_json_bytes(receipt_bytes, label=f"{protocol_id} receipt")
        if not isinstance(receipt, dict):
            raise ValueError(f"{protocol_id} receipt is not an object")
        boundary_contract = _validate_boundary_contract(
            receipt.get("boundary_contract"),
            expected_cache_sha256=cache.sha256,
        )
        runtime_manifests.append(boundary_contract["runtime_manifest"])
        expected_receipt = build_receipt(
            root, replay, snapshot=snapshot, cache=cache, prompt=prompt,
            boundary_contract=boundary_contract,
        )
        if receipt != expected_receipt:
            raise ValueError(f"{protocol_id} receipt differs from deterministic replay")
        if replay.result.get("approved") is not True or replay.result.get("issues") != []:
            raise ValueError(f"{protocol_id} is not an approved issue-free review")
        invocations.append(_receipt_projection(receipt, receipt_bytes))
        results.append(replay.result)
        snapshots.append(snapshot)
    if (invocations[0]["invocation_id"] == invocations[1]["invocation_id"]
            or invocations[0]["thread_id"] == invocations[1]["thread_id"]
            or invocations[0]["raw_event_stream_sha256"]
            == invocations[1]["raw_event_stream_sha256"]):
        raise ValueError("semantic and adversarial reviews are not independent")
    if runtime_manifests[0] != runtime_manifests[1]:
        raise ValueError(
            "semantic and adversarial reviews did not use one identical Codex runtime"
        )
    if (snapshots[0].surfaces != snapshots[1].surfaces
            or snapshots[0].implementation_surface_sha256
            != snapshots[1].implementation_surface_sha256):
        raise ValueError("review invocations do not bind one common semantic surface")
    if sorted(
        results[0]["citation_results"], key=lambda row: row["reference_id"],
    ) != sorted(
        results[1]["citation_results"], key=lambda row: row["reference_id"],
    ):
        raise ValueError("semantic and adversarial citation verdicts disagree")
    source_hashes = review_source_hashes(
        root, snapshots[0].surfaces,
        snapshots[0].implementation_surface_sha256,
    )
    invocations = sorted(
        invocations, key=lambda item: (item["protocol_id"], item["invocation_id"]),
    )
    reviewers = sorted(
        f"{item['model']}@{item['reasoning_effort']}#{item['invocation_id']}"
        for item in invocations
    )
    evidence = {
        "independent_invocations": True,
        "same_model_repeated_review": True,
        "distinct_model_families": False,
        "citation_verified": True,
        "review_agreement": True,
        "unresolved_conflicts": False,
        "review_invocations": invocations,
    }
    updates: dict[Path, bytes] = {}
    pack_root = root / "src" / "hlsgraph" / "knowledge" / "packs"
    for path in sorted(pack_root.glob("*.json")):
        original = path.read_bytes()
        value = _strict_json_bytes(original, label=f"knowledge pack {path.name}")
        if not isinstance(value, dict):
            raise ValueError(f"knowledge pack is not an object: {path.name}")
        before_id, before_surface, _payload = _semantic_pack_projection(
            original, label=f"knowledge pack {path.name}",
        )
        metadata = value.get("metadata")
        coverage = value.get("coverage")
        if not isinstance(metadata, dict) or not isinstance(coverage, dict):
            raise ValueError(f"knowledge pack lacks review fields: {before_id}")
        metadata["review_status"] = "machine_repeated_reviewed"
        coverage.update({
            "review_status": "machine_repeated_reviewed",
            "reviewers": reviewers,
            "source_hashes": source_hashes,
            "review_evidence": evidence,
        })
        encoded = _canonical_json(value)
        after_id, after_surface, _payload = _semantic_pack_projection(
            encoded, label=f"sealed knowledge pack {path.name}",
        )
        if (after_id, after_surface) != (before_id, before_surface):
            raise RuntimeError(f"sealing changed semantic pack surface: {before_id}")
        updates[path] = encoded
    _atomic_replace_pack_bytes(root, updates)
    for snapshot in snapshots:
        sealed = freeze_review_snapshot(root, snapshot.protocol_id)
        if sealed != snapshot:
            raise RuntimeError("sealed pack attestations changed the frozen review snapshot")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    preflight = commands.add_parser("preflight", help="freeze inputs without network/model")
    preflight.add_argument("--root", type=Path, default=Path.cwd())
    preflight.add_argument("--protocol", choices=sorted(PROTOCOLS), required=True)
    review = commands.add_parser(
        "review",
        help="run one legacy v4 cached review (not v0.3 release approval)",
    )
    review.add_argument("--root", type=Path, default=Path.cwd())
    review.add_argument("--protocol", choices=sorted(PROTOCOLS), required=True)
    review.add_argument("--raw-output", type=Path, required=True)
    review.add_argument("--cache-root", type=Path, required=True)
    review.add_argument("--codex-command", default="codex")
    review.add_argument("--timeout-seconds", type=int, default=3600)
    review.add_argument("--fetch-timeout-seconds", type=float, default=60.0)
    review.add_argument("--pdftotext-command")
    review.add_argument("--pdftotext-sha256")
    seal = commands.add_parser(
        "seal",
        help=(
            "rebuild a legacy v4 compatibility seal; the v0.3 release gate "
            "rejects it"
        ),
    )
    seal.add_argument("--root", type=Path, default=Path.cwd())
    seal.add_argument("--semantic-raw", type=Path, required=True)
    seal.add_argument("--adversarial-raw", type=Path, required=True)
    seal.add_argument("--semantic-cache", type=Path, required=True)
    seal.add_argument("--adversarial-cache", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.operation == "preflight":
        print(json.dumps(
            preflight_review(args.root, args.protocol), indent=2,
            sort_keys=True, ensure_ascii=False,
        ))
    elif args.operation == "review":
        run_review(
            args.root, args.protocol, args.raw_output, args.cache_root,
            codex_command=args.codex_command, timeout_seconds=args.timeout_seconds,
            fetch_timeout_seconds=args.fetch_timeout_seconds,
            pdftotext_command=args.pdftotext_command,
            pdftotext_sha256=args.pdftotext_sha256,
        )
    elif args.operation == "seal":
        seal_review_attestations(
            args.root, semantic_raw=args.semantic_raw,
            adversarial_raw=args.adversarial_raw,
            semantic_cache=args.semantic_cache,
            adversarial_cache=args.adversarial_cache,
        )
    else:  # pragma: no cover - argparse owns the closed operation set.
        raise ValueError(f"unknown operation: {args.operation}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
