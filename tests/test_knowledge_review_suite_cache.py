from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

import pytest

from tools import knowledge_review_shards as shard_plan
from tools import knowledge_review_suite_cache as cache_tools
from tools import run_knowledge_review as review
from tools import run_knowledge_review_suite as suite


def _digest(payload: bytes | str) -> str:
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write(root: Path, relative: str, payload: bytes) -> str:
    review._write_private(root / relative, payload)
    return relative


def _citation(
    root: Path, *, requested_url: str, evidence_url: str, body: bytes,
    reference_ids: list[str], inspection: bytes | None,
    resolver_url: str | None = None,
) -> dict[str, object]:
    body_hash = _digest(body)
    body_path = _write(root, f"citations/bodies/{body_hash}.body", body)
    resolver_artifacts: list[dict[str, object]] = []
    if resolver_url is not None:
        resolver_body = b'{"publication":"frozen"}'
        resolver_hash = _digest(resolver_body)
        resolver_path = f"citations/resolver/{resolver_hash}.body"
        if not (root / resolver_path).exists():
            _write(root, resolver_path, resolver_body)
        resolver_artifacts.append({
            "kind": "amd_map_metadata",
            "requested_url": resolver_url,
            "status": 200,
            "final_url": resolver_url,
            "redirect_chain": [resolver_url],
            "content_type": "application/json",
            "body_path": resolver_path,
            "body_sha256": resolver_hash,
            "body_size": len(resolver_body),
        })

    if inspection is None:
        inspection_path = None
        inspection_hash = None
        chunks: list[dict[str, object]] = []
    else:
        inspection_hash = _digest(inspection)
        inspection_path = _write(
            root, f"citations/text/{inspection_hash}.txt", inspection,
        )
        chunks = review._chunk_inventory(
            root, origin_kind="citation", origin_id=requested_url,
            original_sha256=inspection_hash, payload=inspection,
        )
    return {
        "requested_url": requested_url,
        "evidence_url": evidence_url,
        "resolver_id": (
            "amd.docs.khub.topic.v1" if resolver_url is not None
            else "direct.sha256.v1"
        ),
        "reference_ids": sorted(reference_ids),
        "inspection_required": inspection is not None,
        "identity_verified": True,
        "available": True,
        "status": 200,
        "final_url": evidence_url,
        "redirect_chain": [evidence_url],
        "content_type": "text/plain",
        "body_path": body_path,
        "body_sha256": body_hash,
        "body_size": len(body),
        "inspection_path": inspection_path,
        "inspection_sha256": inspection_hash,
        "inspection_size": len(inspection) if inspection is not None else None,
        "parser_id": "fixture.utf8.v1" if inspection is not None else None,
        "parser_version": "1" if inspection is not None else None,
        "parser_command_sha256": _digest("fixture parser") if inspection is not None else None,
        "parser_executable_sha256": None,
        "parser_version_output_sha256": None,
        "resolver_artifacts": resolver_artifacts,
        "inspection_chunks": chunks,
        "error_code": None,
    }


@pytest.fixture
def full_cache(tmp_path: Path) -> tuple[review.ReviewCache, dict[str, object]]:
    root = tmp_path / "full-cache"
    review._mkdir_private(root)

    assigned_source = b"void kernel() { /* assigned source */ }\n"
    private_source = b"private implementation that must never enter a shard\n"
    assigned_path = "src/assigned.cpp"
    private_path = "src/private.cpp"
    assigned_cache_path = _write(root, f"files/{assigned_path}", assigned_source)
    private_cache_path = _write(root, f"files/{private_path}", private_source)
    assigned_chunks = review._chunk_inventory(
        root, origin_kind="source", origin_id=assigned_path,
        original_sha256=_digest(assigned_source), payload=assigned_source,
    )
    files = [
        {
            "path": assigned_path,
            "hash_kind": "raw_sha256",
            "sha256": _digest(assigned_source),
            "cache_path": assigned_cache_path,
            "cache_sha256": _digest(assigned_source),
            "cache_size": len(assigned_source),
            "model_inspection_required": True,
            "chunks": assigned_chunks,
        },
        {
            "path": private_path,
            "hash_kind": "raw_sha256",
            "sha256": _digest(private_source),
            "cache_path": private_cache_path,
            "cache_sha256": _digest(private_source),
            "cache_size": len(private_source),
            "model_inspection_required": False,
            "chunks": [],
        },
    ]

    rule_reference = "1" * 64
    document_reference = "2" * 64
    rule_url = "https://docs.example.test/rule"
    evidence_url = "https://docs.example.test/evidence/rule"
    resolver_url = "https://docs.example.test/api/map"
    document_url = "https://docs.example.test/document-only"
    citations = [
        _citation(
            root, requested_url=rule_url, evidence_url=evidence_url,
            body=b"frozen primary response", reference_ids=[rule_reference],
            inspection=b"Only this normalized rule section is readable.\n",
            resolver_url=resolver_url,
        ),
        _citation(
            root, requested_url=document_url, evidence_url=document_url,
            body=b"full document-only body is private", reference_ids=[document_reference],
            inspection=None,
        ),
    ]
    snapshot = {
        "protocol_id": "semantic-v1",
        "required_files": [],
        "exact_citation_urls": [document_url, rule_url],
    }
    manifest = {
        "schema_version": review.CACHE_SCHEMA_VERSION,
        "protocol_id": snapshot["protocol_id"],
        "review_snapshot_sha256": _digest(review._canonical_json(snapshot)),
        "citation_evidence_sha256": "3" * 64,
        "review_snapshot": snapshot,
        "files": files,
        "citations": citations,
        "chunk_contract": {"schema_version": "fixture.chunk.v1", "sha256": "4" * 64},
        "inspection_contract": {"schema_version": "fixture.inspection.v1"},
        "parser_contract_sha256s": [_digest("fixture parser")],
    }
    manifest_bytes = review._canonical_json(manifest)
    review._write_private(root / review.CACHE_MANIFEST_NAME, manifest_bytes)
    review._harden_private_tree(root)
    cache = review.ReviewCache(root.resolve(), manifest, manifest_bytes)

    projected_file = {
        key: value for key, value in files[0].items() if key != "cache_path"
    }
    full_rule = citations[0]
    projected_citation_keys = {
        "requested_url", "evidence_url", "final_url", "redirect_chain",
        "resolver_id", "status", "content_type", "body_sha256", "body_size",
        "inspection_required", "identity_verified", "available",
        "inspection_sha256", "inspection_size", "parser_id", "parser_version",
        "parser_command_sha256", "parser_executable_sha256",
        "parser_version_output_sha256", "inspection_chunks", "error_code",
        "reference_ids",
    }
    projected_citation = {
        key: full_rule[key] for key in projected_citation_keys
    }
    projected_citation["resolver_artifacts"] = [{
        key: value for key, value in full_rule["resolver_artifacts"][0].items()
        if key != "body_path"
    }]
    citation_surface = suite.citation_evidence_surface_sha256(
        [projected_citation],
    )
    full_surface = []
    for full_citation in citations:
        row = {
            key: full_citation[key] for key in projected_citation_keys
        }
        row["resolver_artifacts"] = [
            {
                key: value for key, value in artifact.items()
                if key != "body_path"
            }
            for artifact in full_citation["resolver_artifacts"]
        ]
        full_surface.append(row)
    full_citation_surface = suite.citation_evidence_surface_sha256(full_surface)
    shard_manifest: dict[str, object] = {
        "schema_version": suite.SHARD_MANIFEST_SCHEMA_VERSION,
        "protocol_id": manifest["protocol_id"],
        "review_snapshot_sha256": manifest["review_snapshot_sha256"],
        "shard_plan_sha256": "5" * 64,
        "shard_id": "knowledge_activation",
        "citation_evidence_surface_sha256": citation_surface,
        "full_citation_evidence_surface_sha256": full_citation_surface,
        "source_paths": [assigned_path],
        "assertion_ids": ["S01.fixture"],
        "rule_references": [{
            "reference_id": rule_reference,
            "reference_surface_sha256": "6" * 64,
            "rule_id": "fixture:1:rule",
            "citation_url": rule_url,
            "section": "Fixture section",
        }],
        "files": [projected_file],
        "citations": [projected_citation],
        "chunk_contract": manifest["chunk_contract"],
        "token_budget_contract": shard_plan.DEFAULT_TOKEN_BUDGET_CONTRACT.to_dict(),
    }
    return cache, shard_manifest


def _tree_files(root: Path) -> set[str]:
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*") if path.is_file()
    }


def _rewrite_frozen(path: Path, payload: bytes) -> None:
    if os.name != "nt":
        path.chmod(0o600)
    path.write_bytes(payload)
    if os.name != "nt":
        path.chmod(review.CACHE_FILE_MODE)


def test_frozen_fetcher_replays_primary_and_resolver_without_network(
    full_cache: tuple[review.ReviewCache, dict[str, object]],
) -> None:
    cache, _manifest = full_cache
    fetch = cache_tools.frozen_cache_fetcher(cache)

    primary = fetch("https://docs.example.test/evidence/rule", 1.0, 1024)
    resolver = fetch("https://docs.example.test/api/map", 1.0, 1024)
    assert primary.body == b"frozen primary response"
    assert primary.content_length == len(primary.body)
    assert resolver.body == b'{"publication":"frozen"}'
    assert resolver.redirect_chain == ("https://docs.example.test/api/map",)
    with pytest.raises(cache_tools.SuiteCacheError, match="not declared"):
        fetch("https://docs.example.test/not-declared", 1.0, 1024)
    with pytest.raises(cache_tools.SuiteCacheError, match="byte limit"):
        fetch("https://docs.example.test/evidence/rule", 1.0, 2)

    # Replay is a memory snapshot: cache mutation after construction cannot
    # substitute response bytes into the second protocol.
    body_path = cache.root / cache.manifest["citations"][0]["body_path"]
    _rewrite_frozen(body_path, b"tampered after fetcher construction")
    assert fetch("https://docs.example.test/evidence/rule", 1.0, 1024).body == (
        b"frozen primary response"
    )


def test_frozen_fetcher_rejects_tampered_full_cache(
    full_cache: tuple[review.ReviewCache, dict[str, object]],
) -> None:
    cache, _manifest = full_cache
    body_path = cache.root / cache.manifest["citations"][0]["body_path"]
    _rewrite_frozen(body_path, b"same-size-would-still-have-wrong-hash")
    with pytest.raises(cache_tools.SuiteCacheError, match="manifest"):
        cache_tools.frozen_cache_fetcher(cache)


def test_materialized_shard_contains_only_manifest_and_assigned_chunks(
    tmp_path: Path,
    full_cache: tuple[review.ReviewCache, dict[str, object]],
) -> None:
    cache, manifest = full_cache
    target = tmp_path / "projected-cache"
    projected = cache_tools.materialize_shard_cache(cache, manifest, target)
    assigned = {
        chunk["path"]
        for row in manifest["files"] for chunk in row["chunks"]
    } | {
        chunk["path"]
        for row in manifest["citations"] for chunk in row["inspection_chunks"]
    }
    assert _tree_files(target) == {review.CACHE_MANIFEST_NAME, *assigned}
    serialized = projected.manifest_bytes.decode("utf-8")
    assert "body_path" not in serialized
    assert "cache_path" not in serialized
    assert "document-only" not in serialized
    assert "private implementation" not in serialized
    assert not (target / "files").exists()
    assert not (target / "citations").exists()
    assert cache_tools.validate_shard_cache(projected, cache, manifest) == projected
    if os.name != "nt":
        assert stat.S_IMODE(target.stat().st_mode) == 0o500
        assert all(
            stat.S_IMODE(path.stat().st_mode) == 0o400
            for path in target.rglob("*") if path.is_file()
        )


def test_shard_validation_rejects_chunk_tampering_and_extra_files(
    tmp_path: Path,
    full_cache: tuple[review.ReviewCache, dict[str, object]],
) -> None:
    cache, manifest = full_cache
    target = tmp_path / "projected-cache"
    projected = cache_tools.materialize_shard_cache(cache, manifest, target)
    chunk_path = next(
        target / chunk["path"]
        for row in manifest["files"] for chunk in row["chunks"]
    )
    original = chunk_path.read_bytes()
    _rewrite_frozen(chunk_path, b"X" + original[1:])
    with pytest.raises(cache_tools.SuiteCacheError, match="differs"):
        cache_tools.validate_shard_cache(projected, cache, manifest)

    # Restore the chunk, then add one correctly permissioned but undeclared
    # file: the closed inventory must still reject it.
    _rewrite_frozen(chunk_path, original)
    if os.name != "nt":
        target.chmod(0o700)
    review._write_private(target / "unmanifested.bin", b"secret")
    review._harden_private_tree(target)
    with pytest.raises(cache_tools.SuiteCacheError, match="unmanifested"):
        cache_tools.validate_shard_cache(target, cache, manifest)


def test_materialization_rejects_body_paths_and_document_only_citations(
    tmp_path: Path,
    full_cache: tuple[review.ReviewCache, dict[str, object]],
) -> None:
    cache, manifest = full_cache
    opened = json.loads(json.dumps(manifest))
    opened["citations"][0]["body_path"] = "citations/bodies/private.body"
    with pytest.raises(cache_tools.SuiteCacheError, match="citation evidence"):
        cache_tools.materialize_shard_cache(cache, opened, tmp_path / "opened")

    document = json.loads(json.dumps(manifest))
    full_document = cache.manifest["citations"][1]
    document["citations"] = [{
        key: value for key, value in full_document.items()
        if key not in {"body_path", "inspection_path"}
    }]
    with pytest.raises(cache_tools.SuiteCacheError):
        cache_tools.materialize_shard_cache(cache, document, tmp_path / "document")


@pytest.mark.skipif(
    os.name == "nt" or not hasattr(os, "O_NOFOLLOW"),
    reason="descriptor-relative replacement defense is the Linux formal path",
)
def test_materialization_rejects_target_replaced_by_symlink_before_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    full_cache: tuple[review.ReviewCache, dict[str, object]],
) -> None:
    cache, manifest = full_cache
    target = tmp_path / "projected-cache"
    outside = tmp_path / "outside"
    outside.mkdir()
    original_mkdir = cache_tools.os.mkdir
    swapped = False

    def replace_new_target(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        mode: int = 0o777, *, dir_fd: int | None = None,
    ) -> None:
        nonlocal swapped
        original_mkdir(path, mode, dir_fd=dir_fd)
        if not swapped and dir_fd is not None and os.fspath(path) == target.name:
            swapped = True
            target.rmdir()
            target.symlink_to(outside, target_is_directory=True)

    monkeypatch.setattr(cache_tools.os, "mkdir", replace_new_target)
    with pytest.raises(cache_tools.SuiteCacheError, match="replaced before"):
        cache_tools.materialize_shard_cache(cache, manifest, target)
    assert swapped is True
    assert list(outside.iterdir()) == []


@pytest.mark.skipif(
    os.name == "nt" or not hasattr(os, "O_NOFOLLOW"),
    reason="descriptor-relative replacement defense is the Linux formal path",
)
def test_materialization_rejects_nested_directory_symlink_before_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    full_cache: tuple[review.ReviewCache, dict[str, object]],
) -> None:
    cache, manifest = full_cache
    target = tmp_path / "projected-cache"
    outside = tmp_path / "outside"
    outside.mkdir()
    original_mkdir = cache_tools.os.mkdir
    swapped = False

    def replace_new_chunk_parent(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        mode: int = 0o777, *, dir_fd: int | None = None,
    ) -> None:
        nonlocal swapped
        original_mkdir(path, mode, dir_fd=dir_fd)
        if not swapped and dir_fd is not None and os.fspath(path) == "chunks":
            swapped = True
            chunk_parent = target / "chunks"
            chunk_parent.rmdir()
            chunk_parent.symlink_to(outside, target_is_directory=True)

    monkeypatch.setattr(cache_tools.os, "mkdir", replace_new_chunk_parent)
    with pytest.raises(cache_tools.SuiteCacheError, match="linked or replaced"):
        cache_tools.materialize_shard_cache(cache, manifest, target)
    assert swapped is True
    assert list(outside.iterdir()) == []


@pytest.mark.skipif(
    os.name == "nt" or not hasattr(os, "O_NOFOLLOW"),
    reason="descriptor-relative replacement defense is the Linux formal path",
)
def test_materialization_parent_replacement_cannot_redirect_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    full_cache: tuple[review.ReviewCache, dict[str, object]],
) -> None:
    cache, manifest = full_cache
    parent = tmp_path / "cache-parent"
    parent.mkdir(mode=0o700)
    target = parent / "cache"
    displaced = tmp_path / "cache-parent-original"
    outside = tmp_path / "outside"
    outside.mkdir()
    original_mkdir = cache_tools.os.mkdir
    swapped = False

    def replace_bound_parent(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        mode: int = 0o777, *, dir_fd: int | None = None,
    ) -> None:
        nonlocal swapped
        original_mkdir(path, mode, dir_fd=dir_fd)
        if not swapped and dir_fd is not None and os.fspath(path) == target.name:
            swapped = True
            parent.rename(displaced)
            parent.symlink_to(outside, target_is_directory=True)

    monkeypatch.setattr(cache_tools.os, "mkdir", replace_bound_parent)
    try:
        with pytest.raises(cache_tools.SuiteCacheError, match="parent"):
            cache_tools.materialize_shard_cache(cache, manifest, target)
        assert swapped is True
        assert list(outside.iterdir()) == []
        assert (displaced / target.name / review.CACHE_MANIFEST_NAME).is_file()
    finally:
        if parent.is_symlink():
            parent.unlink()
        if displaced.exists():
            displaced.rename(parent)
