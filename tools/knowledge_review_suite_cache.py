#!/usr/bin/env python3
"""Private cache projection and offline replay for sharded reviews.

The full review cache is the citation acquisition boundary.  This module can
replay its exact successful fetches without network access and can project one
body-free shard cache containing only the source and citation chunks assigned
by a deterministic shard manifest.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping

from tools import knowledge_review_shards as shard_plan
from tools import run_knowledge_review_suite as suite
from tools.run_knowledge_review import (
    CACHE_DIRECTORY_MODE,
    CACHE_FILE_MODE,
    CACHE_MANIFEST_NAME,
    CACHE_SCHEMA_VERSION,
    MAX_CITATION_BYTES,
    ReviewCache,
    TrustedFetch,
    _assert_private_mode,
    _canonical_json,
    _harden_private_tree,
    _is_link_like,
    _mkdir_private,
    _read_private_cache_file,
    _resolved_unlinked_path,
    _safe_relative,
    _strict_json_bytes,
    _write_private,
)


_SHA256_LENGTH = 64
_POSIX_DESCRIPTOR_MATERIALIZATION = (
    os.name != "nt"
    and hasattr(os, "O_DIRECTORY")
    and hasattr(os, "O_NOFOLLOW")
    and os.mkdir in os.supports_dir_fd
    and os.open in os.supports_dir_fd
)
_FULL_MANIFEST_KEYS = frozenset({
    "schema_version", "protocol_id", "review_snapshot_sha256",
    "citation_evidence_sha256", "review_snapshot", "files", "citations",
    "chunk_contract", "inspection_contract", "parser_contract_sha256s",
})
_FULL_FILE_KEYS = frozenset({
    "path", "hash_kind", "sha256", "cache_path", "cache_sha256",
    "cache_size", "model_inspection_required", "chunks",
})
_FULL_CITATION_KEYS = frozenset({
    "requested_url", "evidence_url", "resolver_id", "reference_ids",
    "inspection_required", "identity_verified", "available", "status",
    "final_url", "redirect_chain", "content_type", "body_path",
    "body_sha256", "body_size", "inspection_path", "inspection_sha256",
    "inspection_size", "parser_id", "parser_version",
    "parser_command_sha256", "parser_executable_sha256",
    "parser_version_output_sha256", "resolver_artifacts",
    "inspection_chunks", "error_code",
})
_FULL_RESOLVER_KEYS = frozenset({
    "kind", "requested_url", "status", "final_url", "redirect_chain",
    "content_type", "body_path", "body_sha256", "body_size",
})
_CHUNK_KEYS = frozenset({
    "index", "path", "sha256", "size", "byte_start", "byte_end",
    "original_sha256", "original_size",
})
_SHARD_MANIFEST_KEYS = frozenset({
    "schema_version", "protocol_id", "review_snapshot_sha256",
    "shard_plan_sha256", "shard_id", "citation_evidence_surface_sha256",
    "full_citation_evidence_surface_sha256", "source_paths", "assertion_ids",
    "rule_references", "files", "citations", "chunk_contract",
    "token_budget_contract",
})
_SHARD_FILE_KEYS = frozenset({
    "path", "hash_kind", "sha256", "cache_sha256", "cache_size",
    "model_inspection_required", "chunks",
})
_SHARD_CITATION_KEYS = frozenset({
    "requested_url", "evidence_url", "final_url", "redirect_chain",
    "resolver_id", "status", "content_type", "body_sha256", "body_size",
    "inspection_required", "identity_verified", "available",
    "inspection_sha256", "inspection_size", "parser_id", "parser_version",
    "parser_command_sha256", "parser_executable_sha256",
    "parser_version_output_sha256", "resolver_artifacts",
    "inspection_chunks", "error_code", "reference_ids",
})
_SHARD_RESOLVER_KEYS = frozenset({
    "kind", "requested_url", "status", "final_url", "redirect_chain",
    "content_type", "body_sha256", "body_size",
})
_RULE_REFERENCE_KEYS = frozenset({
    "reference_id", "reference_surface_sha256", "rule_id", "citation_url",
    "section",
})


class SuiteCacheError(ValueError):
    """A full or projected review cache violates the closed contract."""


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _plain_frozen_root(root: Path, *, label: str) -> Path:
    lexical = root.absolute()
    if _is_link_like(lexical):
        raise SuiteCacheError(f"{label} root must not be a link")
    try:
        resolved = lexical.resolve(strict=True)
        metadata = lexical.lstat()
    except OSError as exc:
        raise SuiteCacheError(f"{label} root is missing or unreadable") from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise SuiteCacheError(f"{label} root is not a directory")
    _assert_private_mode(resolved, CACHE_DIRECTORY_MODE, label=f"{label} root")
    return resolved


def _closed_tree(root: Path, expected_paths: set[str], *, label: str) -> None:
    observed_paths: set[str] = set()
    observed_directories: set[str] = {"."}
    for current, directories, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        relative_parent = current_path.relative_to(root)
        if _is_link_like(current_path):
            raise SuiteCacheError(f"{label} contains a linked directory")
        _assert_private_mode(
            current_path, CACHE_DIRECTORY_MODE, label=f"{label} directory",
        )
        for name in directories:
            path = current_path / name
            if _is_link_like(path):
                raise SuiteCacheError(f"{label} contains a linked directory")
            metadata = path.lstat()
            if not stat.S_ISDIR(metadata.st_mode):
                raise SuiteCacheError(f"{label} contains a non-directory parent")
            observed_directories.add((relative_parent / name).as_posix())
        for name in filenames:
            path = current_path / name
            metadata = path.lstat()
            if (_is_link_like(path) or not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_nlink != 1):
                raise SuiteCacheError(f"{label} contains a linked or aliased file")
            _assert_private_mode(path, CACHE_FILE_MODE, label=f"{label} file")
            observed_paths.add((relative_parent / name).as_posix())

    expected_directories = {"."}
    for relative in expected_paths:
        parent = PurePosixPath(relative).parent
        while parent.as_posix() != ".":
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    if observed_paths != expected_paths or observed_directories != expected_directories:
        raise SuiteCacheError(
            f"{label} contains missing or unmanifested filesystem entries"
        )


def _chunk_rows(value: Any, *, prefix: str, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise SuiteCacheError(f"{label} chunks must be an array")
    result: list[dict[str, Any]] = []
    previous_end = 0
    for expected_index, row in enumerate(value):
        if not isinstance(row, dict) or set(row) != _CHUNK_KEYS:
            raise SuiteCacheError(f"{label} contains a malformed chunk")
        path = row.get("path")
        if (not isinstance(path, str)
                or _safe_relative(path).as_posix() != path
                or not path.startswith(prefix)):
            raise SuiteCacheError(f"{label} has a non-canonical chunk path")
        integers = (
            row.get("index"), row.get("size"), row.get("byte_start"),
            row.get("byte_end"), row.get("original_size"),
        )
        if any(type(item) is not int or item < 0 for item in integers):
            raise SuiteCacheError(f"{label} chunk has malformed integer metadata")
        if (row["index"] != expected_index
                or row["byte_start"] != previous_end
                or row["byte_end"] - row["byte_start"] != row["size"]
                or not _is_sha256(row.get("sha256"))
                or not _is_sha256(row.get("original_sha256"))):
            raise SuiteCacheError(f"{label} chunk ranges or hashes are malformed")
        previous_end = row["byte_end"]
        result.append(dict(row))
    if result and previous_end != result[-1]["original_size"]:
        raise SuiteCacheError(f"{label} chunks do not cover the original bytes")
    return result


def _validated_full_payloads(cache: ReviewCache) -> dict[str, bytes]:
    """Revalidate a full cache and return every declared immutable payload."""

    if not isinstance(cache, ReviewCache):
        raise TypeError("full cache must be a ReviewCache")
    root = _plain_frozen_root(cache.root, label="full review cache")
    manifest_bytes = _read_private_cache_file(root, CACHE_MANIFEST_NAME)
    parsed = _strict_json_bytes(manifest_bytes, label="full review cache manifest")
    if (not isinstance(parsed, dict) or set(parsed) != _FULL_MANIFEST_KEYS
            or parsed.get("schema_version") != CACHE_SCHEMA_VERSION
            or _canonical_json(parsed) != manifest_bytes
            or manifest_bytes != cache.manifest_bytes
            or parsed != cache.manifest):
        raise SuiteCacheError("full review cache manifest is stale or not closed")
    snapshot = parsed.get("review_snapshot")
    if (not isinstance(snapshot, dict)
            or hashlib.sha256(_canonical_json(snapshot)).hexdigest()
            != parsed.get("review_snapshot_sha256")):
        raise SuiteCacheError("full review cache snapshot binding is invalid")

    payloads: dict[str, bytes] = {CACHE_MANIFEST_NAME: manifest_bytes}
    expected_paths = {CACHE_MANIFEST_NAME}

    def declared_payload(
        relative: Any, digest: Any, size: Any, *, prefix: str, suffix: str,
        label: str,
    ) -> bytes:
        if (not isinstance(relative, str)
                or _safe_relative(relative).as_posix() != relative
                or not relative.startswith(prefix) or not relative.endswith(suffix)
                or not _is_sha256(digest)
                or type(size) is not int or size < 0):
            raise SuiteCacheError(f"{label} path or identity is malformed")
        expected_paths.add(relative)
        payload = payloads.get(relative)
        if payload is None:
            payload = _read_private_cache_file(root, relative)
            payloads[relative] = payload
        if len(payload) != size or hashlib.sha256(payload).hexdigest() != digest:
            raise SuiteCacheError(f"{label} bytes differ from their manifest")
        return payload

    seen_sources: set[str] = set()
    files = parsed.get("files")
    if not isinstance(files, list):
        raise SuiteCacheError("full review cache files are not an array")
    for row in files:
        if not isinstance(row, dict) or set(row) != _FULL_FILE_KEYS:
            raise SuiteCacheError("full review cache has a malformed source row")
        source_path = row.get("path")
        if not isinstance(source_path, str) or source_path in seen_sources:
            raise SuiteCacheError("full review cache duplicates a source path")
        seen_sources.add(source_path)
        source = declared_payload(
            row.get("cache_path"), row.get("cache_sha256"), row.get("cache_size"),
            prefix="files/", suffix=source_path, label=f"source {source_path}",
        )
        chunks = _chunk_rows(
            row.get("chunks"), prefix="chunks/source/",
            label=f"source {source_path}",
        )
        if row.get("model_inspection_required") is True:
            if not chunks:
                raise SuiteCacheError(f"source {source_path} has no review chunks")
        elif row.get("model_inspection_required") is False:
            if chunks:
                raise SuiteCacheError("integrity-only source exposes review chunks")
        else:
            raise SuiteCacheError("source inspection scope is not boolean")
        rebuilt = bytearray()
        for chunk in chunks:
            payload = declared_payload(
                chunk["path"], chunk["sha256"], chunk["size"],
                prefix="chunks/source/", suffix=".utf8",
                label=f"source chunk {source_path}",
            )
            payload.decode("utf-8", errors="strict")
            rebuilt.extend(payload)
        if chunks and (
            bytes(rebuilt) != source
            or chunks[0]["original_sha256"] != row.get("cache_sha256")
            or chunks[0]["original_size"] != row.get("cache_size")
            or any(
                chunk["original_sha256"] != row.get("cache_sha256")
                or chunk["original_size"] != row.get("cache_size")
                for chunk in chunks
            )
        ):
            raise SuiteCacheError(f"source {source_path} chunks do not reconstruct it")

    seen_citations: set[str] = set()
    citations = parsed.get("citations")
    if not isinstance(citations, list):
        raise SuiteCacheError("full review cache citations are not an array")
    for row in citations:
        if not isinstance(row, dict) or set(row) != _FULL_CITATION_KEYS:
            raise SuiteCacheError("full review cache has a malformed citation row")
        requested = row.get("requested_url")
        if not isinstance(requested, str) or requested in seen_citations:
            raise SuiteCacheError("full review cache duplicates a citation URL")
        seen_citations.add(requested)
        body: bytes | None = None
        if row.get("body_path") is not None:
            body = declared_payload(
                row["body_path"], row.get("body_sha256"), row.get("body_size"),
                prefix="citations/bodies/", suffix=".body",
                label=f"citation body {requested}",
            )
        inspection: bytes | None = None
        if row.get("inspection_path") is not None:
            inspection = declared_payload(
                row["inspection_path"], row.get("inspection_sha256"),
                row.get("inspection_size"), prefix="citations/text/",
                suffix=".txt", label=f"citation text {requested}",
            )
        artifacts = row.get("resolver_artifacts")
        if not isinstance(artifacts, list):
            raise SuiteCacheError("citation resolver artifacts are not an array")
        for artifact in artifacts:
            if not isinstance(artifact, dict) or set(artifact) != _FULL_RESOLVER_KEYS:
                raise SuiteCacheError("full review cache has a malformed resolver artifact")
            declared_payload(
                artifact.get("body_path"), artifact.get("body_sha256"),
                artifact.get("body_size"), prefix="citations/resolver/",
                suffix=".body", label="citation resolver body",
            )
        chunks = _chunk_rows(
            row.get("inspection_chunks"), prefix="chunks/citation/",
            label=f"citation {requested}",
        )
        rebuilt = bytearray()
        for chunk in chunks:
            payload = declared_payload(
                chunk["path"], chunk["sha256"], chunk["size"],
                prefix="chunks/citation/", suffix=".utf8",
                label=f"citation chunk {requested}",
            )
            payload.decode("utf-8", errors="strict")
            rebuilt.extend(payload)
        if chunks and (
            inspection is None or bytes(rebuilt) != inspection
            or any(
                chunk["original_sha256"] != row.get("inspection_sha256")
                or chunk["original_size"] != row.get("inspection_size")
                for chunk in chunks
            )
        ):
            raise SuiteCacheError(f"citation {requested} chunks do not reconstruct it")
        if row.get("inspection_required") is False and chunks:
            raise SuiteCacheError("document-only citation exposes inspection chunks")
        if row.get("available") is True and body is None:
            raise SuiteCacheError("available citation has no frozen response body")

    _closed_tree(root, expected_paths, label="full review cache")
    return payloads


def _fetch_identity(fetch: TrustedFetch) -> tuple[Any, ...]:
    return (
        fetch.status, fetch.final_url, fetch.redirect_chain,
        fetch.content_type, fetch.body,
    )


def frozen_cache_fetcher(
    full_cache: ReviewCache,
) -> Callable[[str, float, int], TrustedFetch]:
    """Return an offline fetcher for exact primary and resolver responses.

    All declared response bytes are loaded only after the full cache passes a
    closed inventory/hash/mode/link validation.  The returned closure performs
    no filesystem or network I/O and rejects every URL absent from that frozen
    acquisition inventory.
    """

    payloads = _validated_full_payloads(full_cache)
    replay: dict[str, TrustedFetch] = {}

    def register(row: Mapping[str, Any], *, url_key: str = "requested_url") -> None:
        url = row.get(url_key)
        body_path = row.get("body_path")
        if (not isinstance(url, str) or not url
                or not isinstance(body_path, str)
                or type(row.get("status")) is not int
                or not isinstance(row.get("final_url"), str)
                or not isinstance(row.get("redirect_chain"), list)
                or any(not isinstance(item, str) for item in row["redirect_chain"])
                or not isinstance(row.get("content_type"), str)):
            raise SuiteCacheError("full cache cannot replay incomplete fetch metadata")
        body = payloads[body_path]
        content_type = str(row["content_type"])
        fetch = TrustedFetch(
            status=int(row["status"]), final_url=str(row["final_url"]),
            redirect_chain=tuple(row["redirect_chain"]),
            content_type=content_type, body=body,
            charset=None if content_type.casefold() == "application/pdf" else "utf-8",
            content_length=len(body),
        )
        previous = replay.get(url)
        if previous is not None and _fetch_identity(previous) != _fetch_identity(fetch):
            raise SuiteCacheError(f"full cache has conflicting fetch evidence for {url}")
        replay[url] = fetch

    for citation in full_cache.manifest["citations"]:
        # create_review_cache fetches evidence_url, not the human requested URL.
        register(citation, url_key="evidence_url")
        for artifact in citation["resolver_artifacts"]:
            register(artifact)

    def fetch(url: str, timeout_seconds: float, max_bytes: int) -> TrustedFetch:
        if (not isinstance(url, str) or url not in replay):
            raise SuiteCacheError("offline review fetch URL was not declared by the full cache")
        if (not isinstance(timeout_seconds, (int, float))
                or isinstance(timeout_seconds, bool) or timeout_seconds <= 0):
            raise SuiteCacheError("offline review fetch timeout must be positive")
        if type(max_bytes) is not int or max_bytes <= 0:
            raise SuiteCacheError("offline review fetch byte limit must be positive")
        frozen = replay[url]
        if len(frozen.body) > max_bytes:
            raise SuiteCacheError("offline review fetch exceeds the caller byte limit")
        return TrustedFetch(
            status=frozen.status, final_url=frozen.final_url,
            redirect_chain=frozen.redirect_chain,
            content_type=frozen.content_type, body=frozen.body,
            charset=frozen.charset, content_length=frozen.content_length,
        )

    return fetch


def _projected_citation(full: Mapping[str, Any], reference_ids: list[str]) -> dict[str, Any]:
    result = {
        key: full.get(key) for key in _SHARD_CITATION_KEYS
        if key not in {"reference_ids", "resolver_artifacts", "inspection_chunks"}
    }
    result["reference_ids"] = reference_ids
    result["resolver_artifacts"] = [
        {key: artifact.get(key) for key in _SHARD_RESOLVER_KEYS}
        for artifact in full["resolver_artifacts"]
    ]
    result["inspection_chunks"] = [dict(row) for row in full["inspection_chunks"]]
    return result


def _validate_shard_manifest(
    full_cache: ReviewCache, shard_manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not isinstance(shard_manifest, Mapping):
        raise TypeError("shard manifest must be a mapping")
    manifest = json.loads(json.dumps(
        dict(shard_manifest), ensure_ascii=False, allow_nan=False,
    ))
    if (set(manifest) != _SHARD_MANIFEST_KEYS
            or manifest.get("schema_version") != suite.SHARD_MANIFEST_SCHEMA_VERSION
            or manifest.get("protocol_id") != full_cache.manifest.get("protocol_id")
            or manifest.get("review_snapshot_sha256")
            != full_cache.manifest.get("review_snapshot_sha256")
            or not _is_sha256(manifest.get("shard_plan_sha256"))
            or not _is_sha256(manifest.get("citation_evidence_surface_sha256"))
            or not _is_sha256(
                manifest.get("full_citation_evidence_surface_sha256")
            )
            or manifest.get("chunk_contract") != full_cache.manifest.get("chunk_contract")
            or manifest.get("token_budget_contract")
            != shard_plan.DEFAULT_TOKEN_BUDGET_CONTRACT.to_dict()):
        raise SuiteCacheError("shard manifest is stale, open, or not bound to full cache")

    source_paths = manifest.get("source_paths")
    assertion_ids = manifest.get("assertion_ids")
    if (not isinstance(source_paths, list)
            or source_paths != sorted(set(source_paths))
            or any(not isinstance(path, str) or not path for path in source_paths)
            or not isinstance(assertion_ids, list)
            or assertion_ids != sorted(set(assertion_ids))
            or any(not isinstance(item, str) or not item for item in assertion_ids)):
        raise SuiteCacheError("shard source or assertion inventory is malformed")

    references = manifest.get("rule_references")
    if not isinstance(references, list):
        raise SuiteCacheError("shard rule references are not an array")
    references_by_url: dict[str, list[str]] = {}
    previous_reference = ""
    for row in references:
        if (not isinstance(row, dict) or set(row) != _RULE_REFERENCE_KEYS
                or not _is_sha256(row.get("reference_id"))
                or not _is_sha256(row.get("reference_surface_sha256"))
                or not isinstance(row.get("citation_url"), str)
                or not isinstance(row.get("rule_id"), str)
                or not isinstance(row.get("section"), str)
                or row["reference_id"] <= previous_reference):
            raise SuiteCacheError("shard rule reference inventory is malformed")
        previous_reference = row["reference_id"]
        references_by_url.setdefault(row["citation_url"], []).append(
            row["reference_id"],
        )

    full_files = {row["path"]: row for row in full_cache.manifest["files"]}
    files = manifest.get("files")
    if not isinstance(files, list) or [row.get("path") for row in files] != source_paths:
        raise SuiteCacheError("shard file inventory differs from source paths")
    assigned_chunks: list[dict[str, Any]] = []
    for row in files:
        if (not isinstance(row, dict) or set(row) != _SHARD_FILE_KEYS
                or row.get("model_inspection_required") is not True):
            raise SuiteCacheError("shard contains an open or unreadable source row")
        full = full_files.get(row["path"])
        if full is None:
            raise SuiteCacheError("shard source is absent from the full cache")
        expected = {
            key: full.get(key) for key in _SHARD_FILE_KEYS if key != "chunks"
        }
        expected["chunks"] = [dict(chunk) for chunk in full["chunks"]]
        if row != expected:
            raise SuiteCacheError("shard source metadata differs from the full cache")
        chunks = _chunk_rows(
            row["chunks"], prefix="chunks/source/", label=f"shard source {row['path']}",
        )
        if not chunks:
            raise SuiteCacheError("assigned shard source has no chunks")
        assigned_chunks.extend(chunks)

    full_citations = {
        row["requested_url"]: row for row in full_cache.manifest["citations"]
    }
    full_surface = [
        _projected_citation(row, sorted(row["reference_ids"]))
        for row in sorted(
            full_cache.manifest["citations"],
            key=lambda item: item["requested_url"],
        )
    ]
    if suite.citation_evidence_surface_sha256(full_surface) != manifest[
        "full_citation_evidence_surface_sha256"
    ]:
        raise SuiteCacheError("shard full citation evidence surface hash is stale")
    citations = manifest.get("citations")
    if (not isinstance(citations, list)
            or [row.get("requested_url") for row in citations]
            != sorted(references_by_url)):
        raise SuiteCacheError("shard citation inventory differs from rule references")
    for row in citations:
        if (not isinstance(row, dict) or set(row) != _SHARD_CITATION_KEYS
                or row.get("inspection_required") is not True
                or row.get("available") is not True):
            raise SuiteCacheError("shard contains non-rule or unavailable citation evidence")
        full = full_citations.get(row["requested_url"])
        reference_ids = sorted(references_by_url[row["requested_url"]])
        if (full is None or not set(reference_ids).issubset(full["reference_ids"])
                or row != _projected_citation(full, reference_ids)):
            raise SuiteCacheError("shard citation metadata differs from the full cache")
        chunks = _chunk_rows(
            row["inspection_chunks"], prefix="chunks/citation/",
            label=f"shard citation {row['requested_url']}",
        )
        if not chunks:
            raise SuiteCacheError("assigned shard citation has no chunks")
        assigned_chunks.extend(chunks)

    paths = [row["path"] for row in assigned_chunks]
    if len(paths) != len(set(paths)):
        raise SuiteCacheError("shard assigns one chunk path more than once")
    if suite.citation_evidence_surface_sha256(citations) != manifest[
        "citation_evidence_surface_sha256"
    ]:
        raise SuiteCacheError("shard citation evidence surface hash is stale")
    return manifest, assigned_chunks


def validate_shard_cache(
    cache: ReviewCache | Path, full_cache: ReviewCache,
    shard_manifest: Mapping[str, Any],
) -> ReviewCache:
    """Validate one projected cache's closed inventory, bytes, modes and links."""

    full_payloads = _validated_full_payloads(full_cache)
    expected_manifest, chunks = _validate_shard_manifest(full_cache, shard_manifest)
    cache_object = cache if isinstance(cache, ReviewCache) else None
    root = _plain_frozen_root(
        cache.root if cache_object is not None else Path(cache),
        label="shard review cache",
    )
    manifest_bytes = _read_private_cache_file(root, CACHE_MANIFEST_NAME)
    parsed = _strict_json_bytes(manifest_bytes, label="shard review cache manifest")
    if (not isinstance(parsed, dict) or parsed != expected_manifest
            or _canonical_json(parsed) != manifest_bytes
            or (cache_object is not None and (
                cache_object.manifest != parsed
                or cache_object.manifest_bytes != manifest_bytes
            ))):
        raise SuiteCacheError("shard cache manifest is stale or non-canonical")

    expected_paths = {CACHE_MANIFEST_NAME}
    for chunk in chunks:
        relative = chunk["path"]
        expected_paths.add(relative)
        payload = _read_private_cache_file(root, relative)
        if (payload != full_payloads.get(relative)
                or len(payload) != chunk["size"]
                or hashlib.sha256(payload).hexdigest() != chunk["sha256"]):
            raise SuiteCacheError("shard chunk differs from frozen full-cache evidence")
    _closed_tree(root, expected_paths, label="shard review cache")
    return ReviewCache(root, parsed, manifest_bytes)


def _directory_binding(metadata: os.stat_result) -> tuple[int, int, int, int]:
    """Return the path fields that must not change while materializing."""

    return (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
        int(getattr(metadata, "st_uid", -1)),
    )


def _plain_directory_metadata(path: Path, *, label: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise SuiteCacheError(f"{label} is missing or unreadable") from exc
    if _is_link_like(path) or not stat.S_ISDIR(metadata.st_mode):
        raise SuiteCacheError(f"{label} must be a plain directory")
    return metadata


def _open_posix_directory_at(parent_fd: int, name: str) -> int:
    """Open one child directory without following a substituted link.

    Kept as a small helper so the replacement window can be exercised by a
    deterministic regression test.  It is only called on platforms that
    expose the POSIX dir-fd contract.
    """

    flags = (
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )
    return os.open(name, flags, dir_fd=parent_fd)


def _assert_posix_directory_fd(
    descriptor: int, *, expected: os.stat_result | None, label: str,
) -> os.stat_result:
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode):
        raise SuiteCacheError(f"{label} descriptor is not a directory")
    if expected is not None and _directory_binding(metadata) != _directory_binding(
        expected,
    ):
        raise SuiteCacheError(f"{label} changed before its descriptor was bound")
    if metadata.st_uid != os.geteuid():
        raise SuiteCacheError(f"{label} is not owned by the current user")
    return metadata


def _write_posix_file_at(parent_fd: int, name: str, payload: bytes) -> None:
    flags = (
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_BINARY", 0)
    )
    try:
        descriptor = os.open(name, flags, 0o600, dir_fd=parent_fd)
    except OSError as exc:
        raise SuiteCacheError(
            "shard cache file could not be created without following links"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        if (not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1
                or opened.st_uid != os.geteuid()):
            raise SuiteCacheError("shard cache file is linked, aliased, or foreign")
        os.fchmod(descriptor, 0o600)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise SuiteCacheError("short shard cache write")
            view = view[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, CACHE_FILE_MODE)
        completed = os.fstat(descriptor)
        linked = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if (not stat.S_ISREG(completed.st_mode)
                or completed.st_nlink != 1
                or completed.st_size != len(payload)
                or stat.S_IMODE(completed.st_mode) != CACHE_FILE_MODE
                or (completed.st_dev, completed.st_ino)
                != (linked.st_dev, linked.st_ino)):
            raise SuiteCacheError("shard cache file changed during materialization")
    finally:
        os.close(descriptor)


def _materialize_posix_tree(
    *, lexical_target: Path, resolved_parent: Path,
    payloads: Mapping[str, bytes],
) -> Path:
    """Create a shard tree relative to stable directory descriptors.

    No attacker-controlled path is traversed after the parent descriptor is
    bound.  All directories remain open until every file has been written and
    the tree has been hardened, and the lexical path is rebound to those same
    inodes before the descriptors are released.
    """

    parent_before = _plain_directory_metadata(
        lexical_target.parent, label="shard cache parent",
    )
    parent_flags = (
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        parent_fd = os.open(lexical_target.parent, parent_flags)
    except OSError as exc:
        raise SuiteCacheError(
            "shard cache parent could not be opened without following links"
        ) from exc
    directory_fds: dict[tuple[str, ...], int] = {}
    try:
        parent_opened = _assert_posix_directory_fd(
            parent_fd, expected=parent_before, label="shard cache parent",
        )
        try:
            os.mkdir(lexical_target.name, 0o700, dir_fd=parent_fd)
        except OSError as exc:
            raise SuiteCacheError(
                "shard cache target could not be created exclusively"
            ) from exc
        try:
            root_fd = _open_posix_directory_at(parent_fd, lexical_target.name)
        except OSError as exc:
            raise SuiteCacheError(
                "shard cache target was replaced before it could be bound"
            ) from exc
        directory_fds[()] = root_fd
        _assert_posix_directory_fd(
            root_fd,
            expected=os.stat(
                lexical_target.name, dir_fd=parent_fd, follow_symlinks=False,
            ),
            label="shard cache target",
        )
        os.fchmod(root_fd, 0o700)

        for relative, payload in sorted(payloads.items()):
            relative_path = _safe_relative(relative)
            parts = relative_path.parts
            if not parts:
                raise SuiteCacheError("shard cache payload path is empty")
            key: tuple[str, ...] = ()
            current_fd = root_fd
            for part in parts[:-1]:
                child_key = (*key, part)
                child_fd = directory_fds.get(child_key)
                if child_fd is None:
                    try:
                        os.mkdir(part, 0o700, dir_fd=current_fd)
                        child_fd = _open_posix_directory_at(current_fd, part)
                    except OSError as exc:
                        raise SuiteCacheError(
                            "shard cache directory was linked or replaced"
                        ) from exc
                    child_path_metadata = os.stat(
                        part, dir_fd=current_fd, follow_symlinks=False,
                    )
                    _assert_posix_directory_fd(
                        child_fd, expected=child_path_metadata,
                        label="shard cache directory",
                    )
                    os.fchmod(child_fd, 0o700)
                    directory_fds[child_key] = child_fd
                current_fd = child_fd
                key = child_key
            _write_posix_file_at(current_fd, parts[-1], payload)

        # Freeze children first and the root last.  The directory descriptors
        # stay live, so pathname replacement cannot redirect this operation.
        for key, descriptor in sorted(
            directory_fds.items(), key=lambda item: len(item[0]), reverse=True,
        ):
            os.fsync(descriptor)
            os.fchmod(descriptor, CACHE_DIRECTORY_MODE)
            metadata = os.fstat(descriptor)
            if (not stat.S_ISDIR(metadata.st_mode)
                    or metadata.st_uid != os.geteuid()
                    or stat.S_IMODE(metadata.st_mode) != CACHE_DIRECTORY_MODE):
                raise SuiteCacheError("shard cache directory did not freeze")

        root_after = os.fstat(root_fd)
        target_via_parent = os.stat(
            lexical_target.name, dir_fd=parent_fd, follow_symlinks=False,
        )
        parent_after = _plain_directory_metadata(
            lexical_target.parent, label="shard cache parent",
        )
        target_after = lexical_target.lstat()
        try:
            resolved_after = _resolved_unlinked_path(
                lexical_target, label="shard cache target",
            ).resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise SuiteCacheError(
                "shard cache path changed during materialization"
            ) from exc
        if (_directory_binding(parent_opened) != _directory_binding(parent_after)
                or _directory_binding(root_after)
                != _directory_binding(target_via_parent)
                or _directory_binding(root_after) != _directory_binding(target_after)
                or resolved_after != resolved_parent / lexical_target.name):
            raise SuiteCacheError(
                "shard cache target or parent changed during materialization"
            )
        return resolved_after
    finally:
        for descriptor in reversed(tuple(directory_fds.values())):
            try:
                os.close(descriptor)
            except OSError:
                pass
        os.close(parent_fd)


def _path_binding(path: Path, *, label: str) -> tuple[int, int, int, int]:
    return _directory_binding(_plain_directory_metadata(path, label=label))


def _materialize_checked_path_tree(
    *, lexical_target: Path, resolved_parent: Path,
    payloads: Mapping[str, bytes],
) -> Path:
    """Development-platform fallback with fail-closed identity checks.

    Formal review is Linux/ext4-only and therefore always uses the descriptor
    implementation above.  This fallback keeps Windows development and CI
    usable while rejecting every detectable symlink, junction, or replacement
    before and after each exclusive write.
    """

    parent_binding = _path_binding(
        lexical_target.parent, label="shard cache parent",
    )
    _mkdir_private(lexical_target)
    root_binding = _path_binding(lexical_target, label="shard cache target")
    directory_bindings: dict[Path, tuple[int, int, int, int]] = {
        lexical_target: root_binding,
    }

    def assert_bound() -> None:
        if (_path_binding(lexical_target.parent, label="shard cache parent")
                != parent_binding):
            raise SuiteCacheError("shard cache parent changed during materialization")
        for directory, expected in directory_bindings.items():
            if _path_binding(directory, label="shard cache directory") != expected:
                raise SuiteCacheError(
                    "shard cache directory changed during materialization"
                )

    for relative, payload in sorted(payloads.items()):
        relative_path = _safe_relative(relative)
        current = lexical_target
        for part in relative_path.parts[:-1]:
            current = current / part
            if current not in directory_bindings:
                assert_bound()
                try:
                    current.mkdir(mode=0o700, exist_ok=False)
                except OSError as exc:
                    raise SuiteCacheError(
                        "shard cache directory could not be created exclusively"
                    ) from exc
                directory_bindings[current] = _path_binding(
                    current, label="shard cache directory",
                )
        assert_bound()
        _write_private(lexical_target / relative_path, payload)
        assert_bound()

    _harden_private_tree(lexical_target)
    assert_bound()
    try:
        resolved_after = _resolved_unlinked_path(
            lexical_target, label="shard cache target",
        ).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise SuiteCacheError("shard cache path changed during materialization") from exc
    if resolved_after != resolved_parent / lexical_target.name:
        raise SuiteCacheError("shard cache target escaped its bound parent")
    return resolved_after


def materialize_shard_cache(
    full_cache: ReviewCache, shard_manifest: Mapping[str, Any], target: Path,
) -> ReviewCache:
    """Create a private shard cache containing only its assigned chunk files."""

    full_payloads = _validated_full_payloads(full_cache)
    manifest, chunks = _validate_shard_manifest(full_cache, shard_manifest)
    lexical_target = Path(target).absolute()
    if lexical_target.exists() or _is_link_like(lexical_target):
        raise SuiteCacheError("shard cache target already exists or is linked")
    parent = lexical_target.parent
    if (not parent.is_dir() or _is_link_like(parent)):
        raise SuiteCacheError("shard cache parent must be an existing plain directory")
    resolved_parent = _resolved_unlinked_path(
        parent, label="shard cache parent",
    ).resolve(strict=True)
    _resolved_unlinked_path(lexical_target, label="shard cache target")
    full_root = full_cache.root.resolve(strict=True)
    prospective = resolved_parent / lexical_target.name
    try:
        prospective.relative_to(full_root)
    except ValueError:
        pass
    else:
        raise SuiteCacheError("shard cache must not be created inside the full cache")

    materialized_payloads: dict[str, bytes] = {}
    for chunk in chunks:
        relative = chunk["path"]
        payload = full_payloads.get(relative)
        if payload is None:
            raise SuiteCacheError("assigned chunk is absent from validated full cache")
        materialized_payloads[relative] = payload
    manifest_bytes = _canonical_json(manifest)
    materialized_payloads[CACHE_MANIFEST_NAME] = manifest_bytes
    if _POSIX_DESCRIPTOR_MATERIALIZATION:
        materialized_root = _materialize_posix_tree(
            lexical_target=lexical_target, resolved_parent=resolved_parent,
            payloads=materialized_payloads,
        )
    else:
        materialized_root = _materialize_checked_path_tree(
            lexical_target=lexical_target, resolved_parent=resolved_parent,
            payloads=materialized_payloads,
        )
    projected = ReviewCache(
        materialized_root, manifest, manifest_bytes,
    )
    return validate_shard_cache(projected, full_cache, manifest)


__all__ = [
    "SuiteCacheError", "frozen_cache_fetcher", "materialize_shard_cache",
    "validate_shard_cache",
]
