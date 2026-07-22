#!/usr/bin/env python3
"""Audit public knowledge-pack citations without retaining referenced content.

The default audit is offline and deterministic.  ``--online`` adds one bounded
GET per unique fragment-free locator and records only response metadata and a
SHA-256 digest. Both modes emit a reproducible metadata-only evidence SHA-256
for every document identity. It never writes a downloaded response body.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from http.client import HTTPException
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urldefrag, urlsplit, urlunsplit
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if SOURCE_ROOT.is_dir() and str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from hlsgraph.knowledge import KnowledgeCatalog  # noqa: E402
from hlsgraph.model import json_ready  # noqa: E402

try:  # noqa: E402 - this file must also work as ``python tools/...``
    from tools import knowledge_review_surface as _knowledge_review_surface
except ModuleNotFoundError:  # pragma: no cover - direct-script import path
    import knowledge_review_surface as _knowledge_review_surface  # type: ignore[no-redef]


AUDIT_SCHEMA_VERSION = "hlsgraph.knowledge-citation-audit.v2"
LOCATOR_POLICY_ID = "hlsgraph.citation-locator-policy.v1"
DOCUMENT_EVIDENCE_HASH_METHOD = "canonical-json-sha256"
OFFLINE_DOCUMENT_EVIDENCE_VERSION = (
    "hlsgraph.document-citation-evidence.offline-metadata.v2"
)
ONLINE_DOCUMENT_EVIDENCE_VERSION = (
    "hlsgraph.document-citation-evidence.online-fetch-metadata.v2"
)
DOCUMENT_SURFACE_VERSION = "hlsgraph.knowledge-document-surface.v1"
RULE_SURFACE_VERSION = "hlsgraph.knowledge-rule-surface.v1"
REFERENCE_ID_VERSION = "hlsgraph.knowledge-citation-reference-id.v1"
REFERENCE_SURFACE_VERSION = "hlsgraph.knowledge-citation-reference-surface.v1"
GENERATOR_PATH = "tools/audit_knowledge_citations.py"
RELEASE_ARTIFACT_PATH = "docs/knowledge-citation-audit-v0.3.json"
DEFAULT_TIMEOUT_SECONDS = 12.0
DEFAULT_MAX_BYTES = 8 * 1024 * 1024
DEFAULT_MAX_ATTEMPTS = 2
ALLOWED_OFFICIAL_HOSTS = frozenset({
    "arxiv.org",
    "docs.amd.com",
    "documentation-service.arm.com",
    "github.com",
    "llvm.org",
    "mlir.llvm.org",
})

_GITHUB_REPOSITORIES = {
    "circt.handshake": ("llvm", "circt"),
    "dynamatic.mlir_primer": ("epfl-lap", "dynamatic"),
    "llvm.ir.debug": ("llvm", "llvm-project"),
    "llvm.ir.langref": ("llvm", "llvm-project"),
    "llvm.mlir.builtin": ("llvm", "llvm-project"),
    "llvm.mlir.langref": ("llvm", "llvm-project"),
}


@dataclass(frozen=True, slots=True)
class OnlineFetch:
    """Bounded online response metadata; response bytes are intentionally absent."""

    status: int | None
    content_type: str | None
    byte_count: int
    sha256: str | None
    final_url: str
    pdf_magic: bool = False
    error_code: str | None = None
    attempt_count: int = 1


Fetcher = Callable[[str, float, int], OnlineFetch]


def _clean_content_type(value: str | None) -> str | None:
    if not value:
        return None
    return value.partition(";")[0].strip().casefold() or None


def _safe_url_parts(url: str) -> tuple[Any, list[str]]:
    issues: list[str] = []
    try:
        parsed = urlsplit(url)
        _ = parsed.port
    except (TypeError, ValueError):
        return urlsplit(""), ["malformed_url"]
    if parsed.scheme != "https" or not parsed.netloc:
        issues.append("not_absolute_https")
    if parsed.username is not None or parsed.password is not None:
        issues.append("url_credentials_forbidden")
    if parsed.port not in (None, 443):
        issues.append("nonstandard_port_forbidden")
    if parsed.query:
        issues.append("query_parameters_forbidden")
    host = (parsed.hostname or "").casefold()
    if host not in ALLOWED_OFFICIAL_HOSTS:
        issues.append("host_not_allowlisted")
    return parsed, issues


def _locator_kind(document_id: str, url: str) -> str:
    parsed = urlsplit(url)
    host = (parsed.hostname or "").casefold()
    if host == "docs.amd.com":
        return "amd_fluidtopics_locator"
    if host == "documentation-service.arm.com":
        return "arm_static_pdf"
    if host == "github.com":
        return "github_immutable_blob"
    if host == "arxiv.org":
        return "arxiv_official_record"
    if host in {"llvm.org", "mlir.llvm.org"}:
        return "official_current_locator"
    return "unknown"


def _document_host_issues(document_id: str, host: str) -> list[str]:
    if document_id.startswith("amd.") and host != "docs.amd.com":
        return ["document_publisher_host_mismatch"]
    if document_id.startswith("arm.") and host != "documentation-service.arm.com":
        return ["document_publisher_host_mismatch"]
    if document_id == "scalehls.paper" and host != "arxiv.org":
        return ["document_publisher_host_mismatch"]
    if document_id in _GITHUB_REPOSITORIES and host not in {
        "github.com", "llvm.org", "mlir.llvm.org",
    }:
        return ["document_publisher_host_mismatch"]
    return []


def _locator_policy_issues(
    *, document_id: str, document_version: str, url: str, reference_kind: str,
) -> tuple[str, list[str]]:
    parsed, issues = _safe_url_parts(url)
    host = (parsed.hostname or "").casefold()
    issues.extend(_document_host_issues(document_id, host))
    kind = _locator_kind(document_id, url)
    path_parts = [item for item in parsed.path.split("/") if item]

    if kind == "amd_fluidtopics_locator":
        expected_prefix = f"{document_version}-English"
        if len(path_parts) < 3 or path_parts[0] != "r" or path_parts[1] != expected_prefix:
            issues.append("amd_document_version_not_bound_in_url")
    elif kind == "arm_static_pdf":
        if (len(path_parts) != 2 or path_parts[0] != "static"
                or len(path_parts[1]) < 16
                or any(char not in "0123456789abcdefABCDEF" for char in path_parts[1])):
            issues.append("arm_locator_is_not_static_document")
    elif kind == "github_immutable_blob":
        if len(path_parts) < 5 or path_parts[2] != "blob":
            issues.append("github_url_is_not_blob_locator")
        else:
            revision = path_parts[3]
            if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
                issues.append("github_revision_is_not_full_commit")
            if document_version.startswith("git-") and revision != document_version[4:]:
                issues.append("github_revision_document_version_mismatch")
            expected_repo = _GITHUB_REPOSITORIES.get(document_id)
            actual_repo = (path_parts[0].casefold(), path_parts[1].casefold())
            if expected_repo is not None and actual_repo != expected_repo:
                issues.append("github_repository_document_mismatch")
    elif kind == "arxiv_official_record":
        identifier = document_version.removeprefix("arxiv-")
        if len(path_parts) != 2 or path_parts[0] != "abs" or path_parts[1] != identifier:
            issues.append("arxiv_identifier_document_version_mismatch")
    elif kind == "official_current_locator":
        if reference_kind == "rule" and document_version.startswith("git-"):
            issues.append("git_versioned_rule_requires_immutable_blob")

    return kind, sorted(set(issues))


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _typed_surface_sha256(version: str, value: Any) -> str:
    """Hash one public surface with an explicit domain/version separator."""

    return _canonical_sha256({"surface_version": version, "value": value})


def _reference_record(
    *, pack_id: str, review_surface_sha256: str, reference_kind: str,
    document_id: str, document_version: str, url: str,
    document_surface_sha256: str,
    section: str | None = None, rule_id: str | None = None,
    rule_surface_sha256: str | None = None,
) -> dict[str, Any]:
    completeness: list[str] = []
    if not document_id.strip():
        completeness.append("missing_document_id")
    if not document_version.strip():
        completeness.append("missing_document_version")
    if reference_kind == "rule" and (section is None or not section.strip()):
        completeness.append("missing_section")
    if not url.strip():
        completeness.append("missing_citation_url")
    locator_kind, locator_issues = _locator_policy_issues(
        document_id=document_id,
        document_version=document_version,
        url=url,
        reference_kind=reference_kind,
    )
    parsed = urlsplit(url)
    issues = sorted(set(completeness + locator_issues))
    identity = {
        "citation_url": url,
        "document_id": document_id,
        "document_version": document_version,
        "pack_id": pack_id,
        "reference_kind": reference_kind,
        "rule_id": rule_id,
        "section": section,
    }
    reference_id = _typed_surface_sha256(REFERENCE_ID_VERSION, identity)
    record = {
        **identity,
        "document_surface_sha256": document_surface_sha256,
        "fetch_url": urldefrag(url).url,
        "host": (parsed.hostname or "").casefold(),
        "issues": issues,
        "locator_kind": locator_kind,
        "offline_status": "pass" if not issues else "fail",
        "pack_review_surface_sha256": review_surface_sha256,
        "reference_id": reference_id,
        "rule_surface_sha256": rule_surface_sha256,
    }
    record["reference_surface_sha256"] = _typed_surface_sha256(
        REFERENCE_SURFACE_VERSION, record,
    )
    return record


def _urllib_fetch(url: str, timeout_seconds: float, max_bytes: int) -> OnlineFetch:
    request = Request(
        url,
        method="GET",
        headers={
            "Accept": "application/pdf,text/html,text/plain;q=0.9,*/*;q=0.1",
            "Accept-Encoding": "identity",
            "User-Agent": "hlsgraph-citation-audit/0.3 (+https://github.com/liumh05/hlsgraph)",
        },
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
            status = int(response.status)
            content_type = _clean_content_type(response.headers.get("Content-Type"))
            final_url = response.geturl()
            length_value = response.headers.get("Content-Length")
            if length_value:
                try:
                    if int(length_value) > max_bytes:
                        return OnlineFetch(
                            status, content_type, 0, None, final_url,
                            error_code="response_too_large",
                        )
                except ValueError:
                    pass
            digest = hashlib.sha256()
            count = 0
            prefix = b""
            while True:
                chunk = response.read(min(64 * 1024, max_bytes + 1 - count))
                if not chunk:
                    break
                if len(prefix) < 5:
                    prefix += chunk[:5 - len(prefix)]
                count += len(chunk)
                if count > max_bytes:
                    return OnlineFetch(
                        status, content_type, count, None, final_url,
                        pdf_magic=prefix.startswith(b"%PDF-"),
                        error_code="response_too_large",
                    )
                digest.update(chunk)
            return OnlineFetch(
                status=status,
                content_type=content_type,
                byte_count=count,
                sha256=digest.hexdigest(),
                final_url=final_url,
                pdf_magic=prefix.startswith(b"%PDF-"),
            )
    except HTTPError as exc:
        headers = exc.headers
        return OnlineFetch(
            status=int(exc.code),
            content_type=_clean_content_type(
                headers.get("Content-Type") if headers is not None else None
            ),
            byte_count=0,
            sha256=None,
            final_url=exc.geturl(),
            error_code="http_error",
        )
    except (TimeoutError, URLError, OSError):
        return OnlineFetch(
            status=None,
            content_type=None,
            byte_count=0,
            sha256=None,
            final_url=url,
            error_code="network_error",
        )
    except HTTPException:
        return OnlineFetch(
            status=None,
            content_type=None,
            byte_count=0,
            sha256=None,
            final_url=url,
            error_code="protocol_error",
        )


_CURL_META_MARKER = "HLSGRAPH_CITATION_AUDIT_META_V1"


def _curl_fetch(
    executable: str, url: str, timeout_seconds: float, max_bytes: int,
) -> OnlineFetch:
    """Stream one curl response into a digest; never create an output file."""
    command = [
        executable,
        "--silent",
        "--show-error",
        "--fail",
        "--max-time", f"{timeout_seconds:g}",
        "--max-filesize", str(max_bytes),
        "--proto", "=https",
        "--proto-redir", "=https",
        "--max-redirs", "0",
        "--header", "Accept-Encoding: identity",
        "--header", "Accept: application/pdf,text/html,text/plain;q=0.9,*/*;q=0.1",
        "--user-agent", "hlsgraph-citation-audit/0.3 (+https://github.com/liumh05/hlsgraph)",
        "--write-out",
        f"%{{stderr}}\n{_CURL_META_MARKER}\t%{{http_code}}\t%{{content_type}}\t%{{url_effective}}\n",
        url,
    ]
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError:
        return OnlineFetch(None, None, 0, None, url, error_code="transport_unavailable")
    assert process.stdout is not None and process.stderr is not None
    digest = hashlib.sha256()
    count = 0
    prefix = b""
    too_large = False
    try:
        while True:
            chunk = process.stdout.read(64 * 1024)
            if not chunk:
                break
            if len(prefix) < 5:
                prefix += chunk[:5 - len(prefix)]
            count += len(chunk)
            if count > max_bytes:
                too_large = True
                process.kill()
                break
            digest.update(chunk)
        stderr = process.stderr.read(64 * 1024)
        try:
            return_code = process.wait(timeout=timeout_seconds + 3.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            return OnlineFetch(
                None, None, count, None, url,
                pdf_magic=prefix.startswith(b"%PDF-"),
                error_code="network_error",
            )
    finally:
        process.stdout.close()
        process.stderr.close()

    status: int | None = None
    content_type: str | None = None
    final_url = url
    decoded = stderr.decode("utf-8", errors="replace")
    marker_index = decoded.rfind("\n" + _CURL_META_MARKER + "\t")
    if marker_index >= 0:
        fields = decoded[marker_index + 1:].strip().split("\t", 3)
        if len(fields) == 4:
            try:
                candidate_status = int(fields[1])
                status = candidate_status if candidate_status else None
            except ValueError:
                status = None
            content_type = _clean_content_type(fields[2])
            final_url = fields[3] or url

    error_code: str | None = None
    if too_large or return_code == 63:
        error_code = "response_too_large"
    elif return_code == 22:
        error_code = "http_error"
    elif return_code != 0:
        error_code = "network_error"
    complete = error_code is None
    return OnlineFetch(
        status=status,
        content_type=content_type,
        byte_count=count,
        sha256=digest.hexdigest() if complete else None,
        final_url=final_url,
        pdf_magic=prefix.startswith(b"%PDF-"),
        error_code=error_code,
    )


def _default_fetch(url: str, timeout_seconds: float, max_bytes: int) -> OnlineFetch:
    """Prefer curl, then retry transient transport failures with urllib.

    A few HTTPS endpoints terminate an otherwise successful HTTP/2 curl stream
    after returning only a prefix.  That prefix is never accepted as evidence.
    The independent standard-library path re-reads the complete, still-bounded
    response and likewise never writes response bytes to disk.
    """
    executable = shutil.which("curl")
    if executable:
        result = _curl_fetch(executable, url, timeout_seconds, max_bytes)
        if result.error_code not in {
            "network_error", "protocol_error", "transport_unavailable",
        }:
            return result
    return _urllib_fetch(url, timeout_seconds, max_bytes)


def _online_record(
    url: str, locator_kind: str, result: OnlineFetch, max_bytes: int,
) -> dict[str, Any]:
    issues: list[str] = []
    _, final_url_issues = _safe_url_parts(result.final_url)
    if final_url_issues:
        issues.append("unsafe_or_unofficial_redirect_target")
    if ((urlsplit(url).hostname or "").casefold()
            != (urlsplit(result.final_url).hostname or "").casefold()):
        issues.append("redirected_to_different_host")
    if result.error_code:
        issues.append(result.error_code)
    if result.status is None or not 200 <= result.status < 300:
        issues.append("locator_not_reachable")
    if result.byte_count < 0 or result.byte_count > max_bytes:
        # Fetchers are injectable in tests, so reject nonsensical metadata even
        # though the built-in fetcher is bounded by its caller's exact limit.
        issues.append("invalid_byte_count")
    if result.sha256 is not None and (
        len(result.sha256) != 64
        or any(char not in "0123456789abcdef" for char in result.sha256)
    ):
        issues.append("invalid_sha256")
    if result.status is not None and 200 <= result.status < 300:
        if result.byte_count == 0:
            issues.append("empty_response")
        if result.sha256 is None:
            issues.append("missing_sha256")

    if locator_kind == "arm_static_pdf":
        if result.content_type != "application/pdf":
            issues.append("arm_static_document_not_pdf_content_type")
        if not result.pdf_magic:
            issues.append("arm_static_document_not_pdf_bytes")
        verification_level = "document_bytes_verified" if not issues else "failed"
    elif locator_kind == "github_immutable_blob":
        verification_level = "immutable_locator_reachable" if not issues else "failed"
    elif locator_kind == "arxiv_official_record":
        verification_level = "official_record_reachable" if not issues else "failed"
    else:
        # AMD FluidTopics returns a dynamic application shell.  A 2xx response
        # proves only that the locator resolves, never that cited prose was read.
        verification_level = "reachable_locator_only" if not issues else "failed"

    return {
        "attempt_count": result.attempt_count,
        "body_stored": False,
        "byte_count": result.byte_count,
        "content_type": result.content_type,
        "error_code": result.error_code,
        "fetch_url": url,
        "final_url": result.final_url,
        "issues": sorted(set(issues)),
        "locator_kind": locator_kind,
        "pdf_magic": result.pdf_magic,
        "sha256": result.sha256,
        "status": result.status,
        "verification_level": verification_level,
    }


def _manifest_hash(value: dict[str, Any]) -> str:
    return _canonical_sha256(value)


def _queryless_locator(value: Any, *, keep_fragment: bool) -> str | None:
    """Return only public locator components; query strings are never evidence."""
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = urlsplit(value)
    except (TypeError, ValueError):
        return None
    return urlunsplit((
        parsed.scheme.casefold(), parsed.netloc.casefold(), parsed.path, "",
        parsed.fragment if keep_fragment else "",
    ))


def _canonical_distinct(values: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    by_json = {
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")): value
        for value in values
    }
    return [by_json[key] for key in sorted(by_json)]


def _document_reference_evidence(value: dict[str, Any]) -> dict[str, Any]:
    """Whitelist public document metadata used by a document evidence hash."""
    return {
        "document_id": str(value.get("document_id") or ""),
        "document_version": str(value.get("document_version") or ""),
        "kind": value.get("kind"),
        "license_note": value.get("license_note"),
        "official_locator": _queryless_locator(
            value.get("official_url"), keep_fragment=True,
        ),
        "publisher": value.get("publisher"),
        "title": value.get("title"),
    }


def _rule_locator_evidence(value: dict[str, Any]) -> dict[str, Any]:
    """Whitelist one rule's public citation locator metadata."""
    return {
        "citation_locator": _queryless_locator(
            value.get("citation_url"), keep_fragment=True,
        ),
        "fetch_locator": _queryless_locator(
            value.get("fetch_url"), keep_fragment=False,
        ),
        "host": value.get("host"),
        "issues": sorted(str(item) for item in value.get("issues", [])),
        "locator_kind": value.get("locator_kind"),
        "offline_status": value.get("offline_status"),
        "rule_id": value.get("rule_id"),
        "section": value.get("section"),
    }


def _fetch_evidence(value: dict[str, Any]) -> dict[str, Any]:
    """Whitelist response metadata; operational retries and bodies are excluded."""
    return {
        "byte_count": value.get("byte_count"),
        "content_type": value.get("content_type"),
        "error_code": value.get("error_code"),
        "fetch_locator": _queryless_locator(
            value.get("fetch_url"), keep_fragment=False,
        ),
        "final_locator": _queryless_locator(
            value.get("final_url"), keep_fragment=False,
        ),
        "issues": sorted(str(item) for item in value.get("issues", [])),
        "locator_kind": value.get("locator_kind"),
        "pdf_magic": bool(value.get("pdf_magic", False)),
        "response_sha256": value.get("sha256"),
        "status": value.get("status"),
        "verification_level": value.get("verification_level"),
    }


def _document_evidence_records(
    *, document_metadata: Iterable[dict[str, Any]],
    references: Iterable[dict[str, Any]], fetches: Iterable[dict[str, Any]],
    online: bool,
) -> list[dict[str, Any]]:
    """Build deterministic per-document citation evidence hashes.

    The hash payload is deliberately reconstructed from allowlisted public
    fields. Unknown keys (including ``body``, query text, timestamps, and
    transport stderr) cannot influence it.
    """
    documents_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for raw in document_metadata:
        canonical = _document_reference_evidence(raw)
        key = (canonical["document_id"], canonical["document_version"])
        documents_by_key.setdefault(key, []).append(canonical)

    references_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for item in references:
        key = (str(item.get("document_id") or ""),
               str(item.get("document_version") or ""))
        references_by_key.setdefault(key, []).append(item)

    fetch_by_locator = {
        str(item.get("fetch_url") or ""): item for item in fetches
        if item.get("fetch_url")
    }
    method_version = (
        ONLINE_DOCUMENT_EVIDENCE_VERSION
        if online else OFFLINE_DOCUMENT_EVIDENCE_VERSION
    )
    result: list[dict[str, Any]] = []
    for document_id, document_version in sorted(
        set(documents_by_key) | set(references_by_key)
    ):
        raw_references = references_by_key.get((document_id, document_version), [])
        document_locators = _canonical_distinct(
            {
                **_rule_locator_evidence(item),
                "rule_id": None,
                "section": None,
            }
            for item in raw_references if item.get("reference_kind") == "document"
        )
        rule_locators = _canonical_distinct(
            _rule_locator_evidence(item)
            for item in raw_references if item.get("reference_kind") == "rule"
        )
        referenced_fetch_urls = {
            str(item.get("fetch_url") or "") for item in raw_references
            if item.get("fetch_url")
        }
        fetch_metadata = _canonical_distinct(
            _fetch_evidence(fetch_by_locator[url])
            for url in sorted(referenced_fetch_urls)
            if online and url in fetch_by_locator
        )
        payload = {
            "document_locators": document_locators,
            "document_metadata": _canonical_distinct(
                documents_by_key.get((document_id, document_version), [])
            ),
            "hash_method": DOCUMENT_EVIDENCE_HASH_METHOD,
            "hash_method_version": method_version,
            "online_fetch_metadata": fetch_metadata,
            "rule_citation_locators": rule_locators,
        }
        locator_kinds = sorted({
            str(item.get("locator_kind")) for item in raw_references
            if item.get("locator_kind")
        })
        verification_levels = sorted({
            str(item.get("verification_level")) for item in fetch_metadata
            if item.get("verification_level")
        })
        amd_fluidtopics = locator_kinds == ["amd_fluidtopics_locator"]
        if online and (not fetch_metadata or "failed" in verification_levels):
            evidence_scope = "failed_locator_metadata_only"
        elif online and amd_fluidtopics:
            evidence_scope = "reachable_locator_only"
        elif online:
            evidence_scope = "citation_and_fetch_metadata_only"
        else:
            evidence_scope = "citation_locator_metadata_only"
        result.append({
            "body_stored": False,
            "document_id": document_id,
            "document_key": f"{document_id}@{document_version}",
            "document_version": document_version,
            "evidence_scope": evidence_scope,
            "evidence_sha256": _manifest_hash(payload),
            "evidence_sha256_is_document_body_hash": False,
            "fetch_count": len(fetch_metadata),
            "hash_method": DOCUMENT_EVIDENCE_HASH_METHOD,
            "hash_method_version": method_version,
            "locator_kinds": locator_kinds,
            "reference_count": len(document_locators) + len(rule_locators),
            "verification_levels": verification_levels,
        })
    return result


def _builtin_pack_review_surfaces(
    pack_ids: Iterable[str],
) -> dict[str, dict[str, str]]:
    """Bind built-ins to review-excluded pack surfaces without a hash cycle."""

    try:
        pack_root = Path(_knowledge_review_surface.PACK_ROOT).resolve()
        helper_root = Path(_knowledge_review_surface.ROOT).resolve()
    except (AttributeError, OSError) as exc:
        raise ValueError(f"cannot resolve knowledge review surfaces: {exc}") from exc
    if helper_root != ROOT.resolve():
        raise ValueError("knowledge review surface helper belongs to another source root")

    rows: dict[str, dict[str, str]] = {}
    for path in sorted(pack_root.glob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read knowledge pack surface {path}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"knowledge pack surface is not an object: {path}")
        pack_id = value.get("pack_id")
        if not isinstance(pack_id, str) or not pack_id:
            raise ValueError(f"knowledge pack surface lacks pack_id: {path}")
        if pack_id in rows:
            raise ValueError(f"duplicate knowledge pack surface: {pack_id}")
        try:
            relative = path.resolve().relative_to(ROOT.resolve()).as_posix()
            digest = _knowledge_review_surface.surface_sha256(path)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot hash knowledge pack surface {path}: {exc}") from exc
        rows[pack_id] = {
            "path": relative,
            "review_surface_sha256": digest,
        }

    expected = set(pack_ids)
    if set(rows) != expected:
        raise ValueError(
            "knowledge catalog and review-surface pack inventories differ: "
            f"missing={sorted(expected - set(rows))!r}, "
            f"extra={sorted(set(rows) - expected)!r}"
        )
    return rows


def audit_builtin_citations(
    *, online: bool = False, timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_bytes: int = DEFAULT_MAX_BYTES, max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    fetcher: Fetcher | None = None,
) -> dict[str, Any]:
    """Return a deterministic metadata-only audit manifest for built-in packs."""
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    if not 1 <= max_attempts <= 3:
        raise ValueError("max_attempts must be between 1 and 3")

    references: list[dict[str, Any]] = []
    document_metadata: list[dict[str, Any]] = []
    packs: list[dict[str, Any]] = []
    catalog_packs = sorted(
        KnowledgeCatalog.builtin().packs, key=lambda item: item.pack_id,
    )
    pack_surfaces = _builtin_pack_review_surfaces(
        pack.pack_id for pack in catalog_packs
    )
    for pack in catalog_packs:
        pack_surface = pack_surfaces[pack.pack_id]
        packs.append({
            "document_count": len(pack.documents),
            "pack_id": pack.pack_id,
            "path": pack_surface["path"],
            "review_surface_sha256": pack_surface["review_surface_sha256"],
            "rule_count": len(pack.rules),
        })
        document_surfaces: dict[tuple[str, str], str] = {}
        for document in pack.documents:
            metadata = {
                "document_id": document.document_id,
                "document_version": document.document_version,
                "kind": document.kind,
                "license_note": document.license_note,
                "official_url": document.official_url,
                "publisher": document.publisher,
                "title": document.title,
            }
            document_metadata.append(metadata)
            document_surface = _typed_surface_sha256(
                DOCUMENT_SURFACE_VERSION, json_ready(document),
            )
            document_surfaces[
                (document.document_id, document.document_version)
            ] = document_surface
            references.append(_reference_record(
                pack_id=pack.pack_id,
                review_surface_sha256=pack_surface["review_surface_sha256"],
                reference_kind="document",
                document_id=document.document_id,
                document_version=document.document_version,
                document_surface_sha256=document_surface,
                url=document.official_url,
            ))
        for rule in pack.rules:
            document_surface = document_surfaces.get(
                (rule.document_id, rule.document_version)
            )
            if document_surface is None:
                # KnowledgePack validation should already reject this, but the
                # audit remains independently fail-closed if that changes.
                raise ValueError(
                    f"rule {rule.id!r} cites a document outside pack {pack.pack_id!r}"
                )
            references.append(_reference_record(
                pack_id=pack.pack_id,
                review_surface_sha256=pack_surface["review_surface_sha256"],
                reference_kind="rule",
                document_id=rule.document_id,
                document_version=rule.document_version,
                document_surface_sha256=document_surface,
                section=rule.section,
                rule_id=rule.id,
                rule_surface_sha256=_typed_surface_sha256(
                    RULE_SURFACE_VERSION, json_ready(rule),
                ),
                url=rule.citation_url,
            ))
    references.sort(key=lambda item: (
        item["pack_id"], item["document_id"], item["document_version"],
        item["reference_kind"], item["rule_id"] or "", item["citation_url"],
    ))

    offline_failures = sum(item["offline_status"] != "pass" for item in references)
    fetches: list[dict[str, Any]] = []
    if online:
        fetch_impl = fetcher or _default_fetch
        by_url: dict[str, set[str]] = {}
        for item in references:
            if item["offline_status"] == "pass":
                by_url.setdefault(item["fetch_url"], set()).add(item["locator_kind"])
        for url in sorted(by_url):
            kinds = sorted(by_url[url])
            if len(kinds) != 1:
                fetches.append({
                    "attempt_count": 0,
                    "body_stored": False,
                    "byte_count": 0,
                    "content_type": None,
                    "error_code": "conflicting_locator_policies",
                    "fetch_url": url,
                    "final_url": url,
                    "issues": ["conflicting_locator_policies"],
                    "locator_kind": "+".join(kinds),
                    "pdf_magic": False,
                    "sha256": None,
                    "status": None,
                    "verification_level": "failed",
                })
                continue
            result = fetch_impl(url, timeout_seconds, max_bytes)
            attempts_used = 1
            while (result.error_code in {"network_error", "protocol_error"}
                   and attempts_used < max_attempts):
                attempts_used += 1
                result = fetch_impl(url, timeout_seconds, max_bytes)
            result = OnlineFetch(
                status=result.status,
                content_type=result.content_type,
                byte_count=result.byte_count,
                sha256=result.sha256,
                final_url=result.final_url,
                pdf_magic=result.pdf_magic,
                error_code=result.error_code,
                attempt_count=attempts_used,
            )
            record = _online_record(url, kinds[0], result, max_bytes)
            if record["byte_count"] > max_bytes:
                record["issues"] = sorted(set(record["issues"] + ["response_too_large"]))
                record["verification_level"] = "failed"
            fetches.append(record)

    online_failures = sum(bool(item["issues"]) for item in fetches)
    document_evidence = _document_evidence_records(
        document_metadata=document_metadata,
        references=references,
        fetches=fetches,
        online=online,
    )
    generator_path = ROOT / GENERATOR_PATH
    manifest: dict[str, Any] = {
        "document_evidence": document_evidence,
        "document_evidence_policy": {
            "amd_fluidtopics_online_scope": "reachable_locator_only",
            "canonicalization": "utf8-json-sort-keys-no-whitespace",
            "evidence_sha256_is_document_body_hash": False,
            "hash_method": DOCUMENT_EVIDENCE_HASH_METHOD,
            "offline_hash_method_version": OFFLINE_DOCUMENT_EVIDENCE_VERSION,
            "online_hash_method_version": ONLINE_DOCUMENT_EVIDENCE_VERSION,
            "query_parameters_in_hash": False,
            "response_bodies_stored": False,
        },
        "fetches": fetches,
        "generator": {
            "path": GENERATOR_PATH,
            "sha256": hashlib.sha256(generator_path.read_bytes()).hexdigest(),
        },
        "mode": "online" if online else "offline",
        "packs": packs,
        "passed": offline_failures == 0 and online_failures == 0,
        "policy": {
            "allowed_official_hosts": sorted(ALLOWED_OFFICIAL_HOSTS),
            "locator_policy_id": LOCATOR_POLICY_ID,
            "max_bytes": max_bytes,
            "max_attempts": max_attempts,
            "response_bodies_stored": False,
            "timeout_seconds": timeout_seconds,
        },
        "references": references,
        "schema_version": AUDIT_SCHEMA_VERSION,
        "summary": {
            "pack_count": len(packs),
            "document_references": sum(
                item["reference_kind"] == "document" for item in references
            ),
            "document_evidence_records": len(document_evidence),
            "fetch_failures": online_failures,
            "offline_failures": offline_failures,
            "rule_references": sum(
                item["reference_kind"] == "rule" for item in references
            ),
            "reference_count": len(references),
            "unique_fetch_urls": len(fetches),
        },
        "surface_policy": {
            "canonicalization": "utf8-json-sort-keys-no-whitespace-no-nan",
            "document_surface_version": DOCUMENT_SURFACE_VERSION,
            "pack_surface_method": "knowledge-review-semantic-surface-sha256",
            "reference_id_version": REFERENCE_ID_VERSION,
            "reference_surface_version": REFERENCE_SURFACE_VERSION,
            "rule_surface_version": RULE_SURFACE_VERSION,
        },
    }
    manifest["manifest_sha256"] = _manifest_hash(manifest)
    return manifest


def dump_manifest(manifest: dict[str, Any]) -> str:
    return json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--online", action="store_true",
                        help="perform bounded GETs after the offline audit")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS,
                        help="per-request timeout in seconds")
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES,
                        help="maximum response bytes read per unique locator")
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS,
                        help="bounded attempts for transient network/protocol failures (1-3)")
    parser.add_argument("--output", type=Path,
                        help="write the JSON manifest here instead of stdout")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = audit_builtin_citations(
            online=args.online,
            timeout_seconds=args.timeout,
            max_bytes=args.max_bytes,
            max_attempts=args.max_attempts,
        )
    except (OSError, ValueError) as exc:
        print(f"citation audit configuration error: {exc}", file=sys.stderr)
        return 2
    rendered = dump_manifest(manifest)
    if args.output is None:
        sys.stdout.write(rendered)
    else:
        try:
            args.output.write_text(rendered, encoding="utf-8", newline="\n")
        except OSError as exc:
            print(f"cannot write citation audit manifest: {exc}", file=sys.stderr)
            return 2
    return 0 if manifest["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
