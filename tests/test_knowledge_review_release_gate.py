from __future__ import annotations

import copy
import hashlib
from http.client import IncompleteRead
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from urllib.parse import urlsplit

import pytest

from tools import knowledge_review_surface
from tools import audit_knowledge_citations
from tools import run_knowledge_review
from tools import audit_release
from tools.audit_release import (
    ADVERSARIAL_REVIEW_PATH,
    ADVERSARIAL_REVIEW_PROMPT_PATH,
    ADVERSARIAL_REVIEW_PROTOCOL,
    ADVERSARIAL_REVIEW_RECEIPT_PATH,
    ADVERSARIAL_REVIEW_TRACE_PATH,
    CITATION_AUDIT_PATH,
    IMPLEMENTATION_SURFACE_HASH_KEY,
    PACK_SURFACE_HASH_PREFIX,
    PACK_SURFACE_HASH_SUFFIX,
    REVIEW_MODEL,
    REVIEW_REASONING_EFFORT,
    REVIEW_RECEIPT_SCHEMA_PATH,
    REVIEW_RECEIPT_SCHEMA_VERSION,
    REVIEW_TRACE_SCHEMA_VERSION,
    REVIEW_SCHEMA_PATH,
    SEMANTIC_REVIEW_PATH,
    SEMANTIC_REVIEW_PROMPT_PATH,
    SEMANTIC_REVIEW_PROTOCOL,
    SEMANTIC_REVIEW_RECEIPT_PATH,
    SEMANTIC_REVIEW_TRACE_PATH,
    SURFACE_HELPER_HASH_KEY,
    verify_legacy_v4_review_evidence,
)


ROOT = Path(__file__).parents[1]
PACKS = {
    "amd_public_guidance_2024_2.json": "hlsgraph.amd.public_guidance.2024_2",
    "axi_public_guidance.json": "hlsgraph.axi.public_guidance.v1",
    "open_ir_public_guidance.json": (
        "hlsgraph.open_ir.public_guidance.2026_07_21"
    ),
}


class _FetchHeaders:
    def get_content_type(self) -> str:
        return "text/plain"

    def get_content_charset(self) -> str:
        return "utf-8"


class _FetchResponse:
    status = 200
    headers = _FetchHeaders()

    def __init__(self, url: str, body: bytes | BaseException) -> None:
        self._url = url
        self._body = body

    def __enter__(self) -> "_FetchResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def getcode(self) -> int:
        return self.status

    def geturl(self) -> str:
        return self._url

    def read(self, _limit: int) -> bytes:
        if isinstance(self._body, BaseException):
            raise self._body
        return self._body


class _LengthFetchHeaders(_FetchHeaders):
    def __init__(self, length: int, content_type: str = "text/plain") -> None:
        self._length = length
        self._content_type = content_type

    def get(self, name: str) -> str | None:
        return str(self._length) if name.casefold() == "content-length" else None

    def get_content_type(self) -> str:
        return self._content_type


class _LengthFetchResponse(_FetchResponse):
    def __init__(
        self, url: str, body: bytes | BaseException, *, declared_length: int,
        content_type: str = "text/plain",
    ) -> None:
        super().__init__(url, body)
        self.headers = _LengthFetchHeaders(declared_length, content_type)


class _FetchOpener:
    def __init__(self, response: _FetchResponse) -> None:
        self._response = response

    def open(self, _request: object, *, timeout: float) -> _FetchResponse:
        assert timeout > 0
        return self._response


def _raw_evidence_path(root: Path, raw_name: str) -> Path:
    protocol = raw_name.split(".", 1)[0]
    return root.parent / f"{protocol}.evidence" / raw_name


def _write_json(path: Path, value: object) -> bytes:
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> bytes:
    data = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


def _trusted_fetch(
    url: str, _timeout: float, _max_bytes: int,
) -> run_knowledge_review.TrustedFetch:
    mapping_path = ROOT / run_knowledge_review.CITATION_EVIDENCE_PATH
    if mapping_path.is_file():
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        return _evidence_fetcher(mapping)(url, _timeout, _max_bytes)
    locator_hash = hashlib.sha256(url.encode("utf-8")).hexdigest().encode("ascii")
    if "documentation-service.arm.com/static/" in url:
        body = b"%PDF-1.7\nHLSGRAPH-CACHE-PDF-" + locator_hash + b"\n%%EOF\n"
        content_type = "application/pdf"
    else:
        body = b"HLSGRAPH-CACHE-TEXT-" + locator_hash
        content_type = "text/plain"
    return run_knowledge_review.TrustedFetch(
        status=200, final_url=url, redirect_chain=(url,),
        content_type=content_type, body=body, charset="utf-8",
    )


def _fixture_primary_body(entry: dict[str, object]) -> bytes:
    url = str(entry["evidence_url"])
    locator_hash = hashlib.sha256(url.encode("utf-8")).hexdigest().encode("ascii")
    if "documentation-service.arm.com/static/" in url:
        return b"%PDF-1.7\nHLSGRAPH-CACHE-PDF-" + locator_hash + b"\n%%EOF\n"
    if str(entry["resolver_id"]).startswith("github.raw."):
        return b"HLSGRAPH-PINNED-RAW-" + locator_hash + b"\n"
    return b"HLSGRAPH-MAPPED-TOPIC-TEXT-" + locator_hash


def _fixture_reference_binding(row: dict[str, object]) -> dict[str, object]:
    section = row.get("section")
    return {
        "reference_id": row.get("reference_id"),
        "reference_kind": row.get("reference_kind"),
        "reference_surface_sha256": row.get("reference_surface_sha256"),
        "document_id": row.get("document_id"),
        "document_version": row.get("document_version"),
        "rule_id": row.get("rule_id"),
        "rule_surface_sha256": row.get("rule_surface_sha256"),
        "section": section,
        "section_sha256": hashlib.sha256(
            run_knowledge_review._canonical_json(section),
        ).hexdigest(),
    }


def _fixture_evidence_mapping(citation_bytes: bytes) -> dict[str, object]:
    citation = json.loads(citation_bytes)
    public = json.loads(
        (ROOT / run_knowledge_review.CITATION_EVIDENCE_PATH).read_text(
            encoding="utf-8",
        )
    )
    references_by_url: dict[str, list[dict[str, object]]] = {}
    for row in citation["references"]:
        references_by_url.setdefault(str(row["citation_url"]), []).append(row)
    entries = copy.deepcopy(public["entries"])
    for entry in entries:
        url = str(entry["citation_url"])
        entry["reference_bindings"] = sorted(
            (
                _fixture_reference_binding(row)
                for row in references_by_url[url]
            ),
            key=lambda row: str(row["reference_id"]),
        )
        body = _fixture_primary_body(entry)
        resolver_id = str(entry["resolver_id"])
        identity = entry["identity"]
        if resolver_id == "direct.sha256.v1":
            assert isinstance(identity, dict)
            identity["body_sha256"] = hashlib.sha256(body).hexdigest()
            identity["body_size"] = len(body)
            identity["content_type"] = (
                "application/pdf" if body.startswith(b"%PDF-") else "text/html"
            )
        elif resolver_id == "github.raw.document.v1":
            assert isinstance(identity, dict)
            identity["source_sha256"] = hashlib.sha256(body).hexdigest()
            identity["source_size"] = len(body)
        elif resolver_id == "github.raw.lines.v1":
            assert isinstance(identity, dict)
            identity.update({
                "source_sha256": hashlib.sha256(body).hexdigest(),
                "start_line": 1, "end_line": 1,
                "slice_sha256": hashlib.sha256(body).hexdigest(),
            })
    return {
        "schema_version": run_knowledge_review.CITATION_EVIDENCE_SCHEMA_VERSION,
        "citation_audit_sha256": hashlib.sha256(citation_bytes).hexdigest(),
        "entries": entries,
    }


def _evidence_fetcher(mapping: dict[str, object]):
    entries = [row for row in mapping["entries"] if isinstance(row, dict)]
    by_evidence = {str(row["evidence_url"]): row for row in entries}
    by_publication: dict[str, list[dict[str, object]]] = {}
    for row in entries:
        identity = row.get("identity")
        if isinstance(identity, dict) and "publication_id" in identity:
            by_publication.setdefault(str(identity["publication_id"]), []).append(row)

    def fetch(url: str, _timeout: float, _max_bytes: int):
        parts = urlsplit(url)
        path_parts = parts.path.strip("/").split("/")
        body: bytes
        content_type = "text/plain"
        if (parts.hostname == "docs.amd.com" and len(path_parts) == 4
                and path_parts[:3] == ["api", "khub", "maps"]):
            publication_id = path_parts[3]
            rows = by_publication[publication_id]
            identity = rows[0]["identity"]
            assert isinstance(identity, dict)
            body = run_knowledge_review._canonical_json({
                "id": publication_id, "title": identity["title"],
                "baseId": f"{identity['document_id']}-en-us-{identity['version']}.ditamap",
                "clusterId": identity["document_id"],
                "prettyUrl": (
                    f"/go/{identity['version']}-English/{identity['document_slug']}"
                ),
                "readerUrl": (
                    f"/r/{identity['version']}-English/{identity['document_slug']}"
                ),
                "fingerprint": "fixture-fingerprint",
                "metadata": [
                    {"key": "Doc_Version", "values": [f"{identity['version']} English"]},
                    {"key": "Document_ID", "values": [identity["document_id"]]},
                    {"key": "Access_Level", "values": ["Public"]},
                    {"key": "ft:publicationId", "values": [publication_id]},
                    {"key": "ft:prettyUrl", "values": [
                        f"{identity['version']}-English/{identity['document_slug']}"
                    ]},
                ],
            })
            content_type = "application/json"
        elif (parts.hostname == "docs.amd.com" and len(path_parts) == 5
                and path_parts[:3] == ["api", "khub", "maps"]
                and path_parts[4] == "pages"):
            publication_id = path_parts[3]
            topics = []
            for row in by_publication[publication_id]:
                identity = row["identity"]
                if row["resolver_id"] != "amd.docs.khub.topic.v1":
                    continue
                assert isinstance(identity, dict)
                topics.append({
                    "tocId": identity["toc_id"],
                    "contentId": identity["content_id"],
                    "title": identity["topic_title"],
                    "prettyUrl": urlsplit(str(row["citation_url"])).path,
                    "children": [],
                })
            body = run_knowledge_review._canonical_json({
                "configuration": {}, "paginatedToc": topics,
                "translationError": False,
            })
            content_type = "application/json"
        elif url in by_evidence:
            row = by_evidence[url]
            body = _fixture_primary_body(row)
            identity = row.get("identity")
            if (row.get("resolver_id") == "direct.sha256.v1"
                    and isinstance(identity, dict)):
                content_type = str(identity["content_type"])
        elif "documentation-service.arm.com/static/" in url:
            body = b"%PDF-1.7\nHLSGRAPH-CACHE-PDF-" + hashlib.sha256(
                url.encode("utf-8")
            ).hexdigest().encode("ascii") + b"\n%%EOF\n"
            content_type = "application/pdf"
        else:
            body = b"HLSGRAPH-CACHE-TEXT-" + hashlib.sha256(
                url.encode("utf-8")
            ).hexdigest().encode("ascii")
        return run_knowledge_review.TrustedFetch(
            status=200, final_url=url, redirect_chain=(url,),
            content_type=content_type, body=body, charset="utf-8",
        )

    return fetch


def _fixture_pdf_text(body: bytes) -> run_knowledge_review.TextDerivation:
    digest = hashlib.sha256(body).hexdigest().encode("ascii")
    return run_knowledge_review.TextDerivation(
        text=b"HLSGRAPH-CONTROLLED-PDF-TEXT-" + digest,
        parser_id="fixture-pdf-parser", parser_version="fixture-pdf-parser/1",
        command_sha256=hashlib.sha256(
            b"fixture-pdf-parser/1:$INPUT:$STDOUT"
        ).hexdigest(),
    )


def _approved_result(
    snapshot: run_knowledge_review.ReviewSnapshot, *, summary: str,
) -> dict[str, object]:
    citation_results: list[dict[str, object]] = []
    for reference in run_knowledge_review._citation_reference_rows(snapshot):
        is_rule = reference["reference_kind"] == "rule"
        citation_results.append({
            "reference_id": reference["reference_id"],
            "reference_surface_sha256": reference["reference_surface_sha256"],
            "verdict": "verified",
            "exact_locator_inspected": is_rule,
            "declared_version_matched": True,
            "declared_section_matched": True if is_rule else None,
            "paraphrase_supported": True if is_rule else None,
            "applicability_not_broader": True if is_rule else None,
            "issues": [],
        })
    return {
        "protocol_id": snapshot.protocol_id,
        "review_surface_sha256": snapshot.surfaces,
        "implementation_surface_sha256": (
            snapshot.implementation_surface_sha256
        ),
        "citation_audit_sha256": snapshot.citation_audit_sha256,
        "citation_results": citation_results,
        "approved": True,
        "issues": [],
        "summary": "approved_no_issues",
    }


def _raw_stream(
    *, snapshot: run_knowledge_review.ReviewSnapshot,
    cache: run_knowledge_review.ReviewCache,
    result: dict[str, object], thread_id: str,
) -> bytes:
    rows: list[dict[str, object]] = [
        {"type": "thread.started", "thread_id": thread_id},
        {"type": "turn.started"},
    ]
    sequence = 0

    def command(value: str) -> None:
        nonlocal sequence
        sequence += 1
        call_id = f"command-{sequence:04d}"
        output, _operations, _citation = run_knowledge_review._expected_command(
            cache, value,
        )
        item = {"id": call_id, "type": "command_execution", "command": value}
        rows.append({"type": "item.started", "item": item})
        rows.append({
            "type": "item.completed",
            "item": {
                **item, "status": "completed", "exit_code": 0,
                "aggregated_output": output,
            },
        })

    for item in cache.manifest["files"]:
        if item.get("model_inspection_required") is not True:
            continue
        for chunk in item["chunks"]:
            command(f"head -n 100000000 {chunk['path']}")
    for entry in cache.manifest["citations"]:
        if entry.get("available") is True:
            for chunk in entry["inspection_chunks"]:
                command(f"head -n 100000000 {chunk['path']}")
    rows.extend([
        {
            "type": "item.completed",
            "item": {
                "id": "agent-message-final", "type": "agent_message",
                "text": json.dumps(result, sort_keys=True),
            },
        },
        {"type": "turn.completed", "usage": {"total_tokens": 1}},
    ])
    raw = run_knowledge_review._canonical_jsonl(rows)
    return run_knowledge_review.sanitize_raw_review_stream(raw, cache)


def _snapshot_cache(
    root: Path, protocol_id: str = SEMANTIC_REVIEW_PROTOCOL,
) -> tuple[
    run_knowledge_review.ReviewSnapshot, run_knowledge_review.ReviewCache,
]:
    snapshot = run_knowledge_review.freeze_review_snapshot(root, protocol_id)
    cache_name = (
        "semantic.cache" if protocol_id == SEMANTIC_REVIEW_PROTOCOL
        else "adversarial.cache"
    )
    cache = run_knowledge_review.load_review_cache(
        root.parent / cache_name, snapshot,
    )
    return snapshot, cache


def _fixture_boundary_contract(
    cache: run_knowledge_review.ReviewCache,
) -> dict[str, object]:
    empty_sha256 = hashlib.sha256(b"").hexdigest()
    runtime_payload: dict[str, object] = {
        "schema_version": run_knowledge_review.RUNTIME_MANIFEST_SCHEMA_VERSION,
        "ownership_policy": run_knowledge_review.RUNTIME_OWNERSHIP_POLICY,
        "executable_relative_path": "codex",
        "executable_sha256": run_knowledge_review.OFFICIAL_CODEX_ELF_SHA256,
        "bubblewrap_relative_path": "codex-resources/bwrap",
        "bubblewrap_sha256": run_knowledge_review.OFFICIAL_CODEX_BWRAP_SHA256,
        "entries": [
            {
                "relative_path": ".", "kind": "dir", "size": 0,
                "mode": "0500", "sha256": empty_sha256,
            },
            {
                "relative_path": "codex", "kind": "file", "size": 5,
                "mode": "0500", "sha256": run_knowledge_review.OFFICIAL_CODEX_ELF_SHA256,
            },
            {
                "relative_path": "codex-resources", "kind": "dir", "size": 0,
                "mode": "0500", "sha256": empty_sha256,
            },
            {
                "relative_path": "codex-resources/bwrap", "kind": "file", "size": 5,
                "mode": "0500", "sha256": run_knowledge_review.OFFICIAL_CODEX_BWRAP_SHA256,
            },
        ],
    }
    runtime_payload["sha256"] = hashlib.sha256(
        run_knowledge_review._canonical_json(runtime_payload)
    ).hexdigest()
    return run_knowledge_review._build_boundary_contract(
        runtime_manifest=runtime_payload,
        cache_manifest_sha256=cache.sha256,
        cache_parent_policy=run_knowledge_review.CACHE_PARENT_POLICY,
        evidence_parent_policy=run_knowledge_review.EVIDENCE_PARENT_POLICY,
        canary_results={
            "cache_read": True,
            "runtime_read": True,
            "checkout_denied": True,
            "auth_denied": True,
            "external_denied": True,
            "peer_sibling_denied": True,
            "evidence_denied": True,
            "cache_write_denied": True,
        },
    )


@pytest.fixture
def reviewed_release_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "public"
    pack_root = root / "src" / "hlsgraph" / "knowledge" / "packs"
    shutil.copytree(ROOT / "src" / "hlsgraph", root / "src" / "hlsgraph")
    for relative in (
        REVIEW_SCHEMA_PATH,
        REVIEW_RECEIPT_SCHEMA_PATH,
        run_knowledge_review.CITATION_EVIDENCE_SCHEMA_PATH,
        SEMANTIC_REVIEW_PROMPT_PATH,
        ADVERSARIAL_REVIEW_PROMPT_PATH,
        "tools/knowledge_review_surface.py",
        "tools/audit_knowledge_citations.py",
        "tools/run_knowledge_review.py",
        "tools/audit_release.py",
        *sorted(run_knowledge_review.SUITE_REVIEW_SOURCE_PATHS),
    ):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)

    monkeypatch.setattr(knowledge_review_surface, "ROOT", root)
    monkeypatch.setattr(knowledge_review_surface, "PACK_ROOT", pack_root)
    monkeypatch.setattr(
        knowledge_review_surface, "IMPLEMENTATION_ROOT", root / "src" / "hlsgraph",
    )
    monkeypatch.setattr(audit_knowledge_citations, "ROOT", root)
    monkeypatch.setattr(audit_knowledge_citations, "SOURCE_ROOT", root / "src")
    monkeypatch.setattr(run_knowledge_review, "SCRIPT_ROOT", root)
    monkeypatch.setattr(run_knowledge_review, "_formal_host_is_windows", lambda: False)
    monkeypatch.setattr(audit_release, "_formal_host_is_windows", lambda: False)
    codex_home = root.parent / "codex-home"
    codex_home.mkdir(mode=0o700)
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    def citation_fetch(url: str, _timeout: float, _max_bytes: int):
        locator_hash = hashlib.sha256(url.encode("utf-8")).hexdigest().encode("ascii")
        payload = (
            b"%PDF-1.7\nHLSGRAPH-CACHE-PDF-" + locator_hash + b"\n%%EOF\n"
            if "documentation-service.arm.com/static/" in url
            else b"HLSGRAPH-MAPPED-TOPIC-TEXT-" + locator_hash
        )
        return audit_knowledge_citations.OnlineFetch(
            200,
            "application/pdf" if payload.startswith(b"%PDF") else "text/html",
            len(payload), hashlib.sha256(payload).hexdigest(), url,
            pdf_magic=payload.startswith(b"%PDF"),
        )

    citation = audit_knowledge_citations.audit_builtin_citations(
        online=True, fetcher=citation_fetch,
    )
    citation_bytes = _write_json(root / CITATION_AUDIT_PATH, citation)
    evidence_mapping = _fixture_evidence_mapping(citation_bytes)
    _write_json(
        root / run_knowledge_review.CITATION_EVIDENCE_PATH,
        evidence_mapping,
    )

    invocations: list[
        tuple[
            str, run_knowledge_review.ReviewSnapshot,
            run_knowledge_review.ReviewCache, run_knowledge_review.ReviewReplay,
        ]
    ] = []
    for protocol_id, thread_id, cache_name, raw_name, summary in (
        (
            SEMANTIC_REVIEW_PROTOCOL, "thread-semantic-0001",
            "semantic.cache", "semantic.raw.jsonl",
            "The frozen semantic review closes its declared knowledge surface.",
        ),
        (
            ADVERSARIAL_REVIEW_PROTOCOL, "thread-adversarial-0001",
            "adversarial.cache", "adversarial.raw.jsonl",
            "The independent activation review found no bypass.",
        ),
    ):
        snapshot = run_knowledge_review.freeze_review_snapshot(root, protocol_id)
        cache = run_knowledge_review.create_review_cache(
            root, snapshot, root.parent / cache_name,
            fetcher=_evidence_fetcher(evidence_mapping),
            pdf_text_extractor=_fixture_pdf_text,
        )
        cache = run_knowledge_review.load_review_cache(cache.root, snapshot)
        result_value = _approved_result(snapshot, summary=summary)
        raw_bytes = _raw_stream(
            snapshot=snapshot, cache=cache, result=result_value,
            thread_id=thread_id,
        )
        raw_path = _raw_evidence_path(root, raw_name)
        run_knowledge_review._write_private(raw_path, raw_bytes)
        replay = run_knowledge_review.replay_raw_review(
            root, protocol_id, raw_bytes, snapshot=snapshot, cache=cache,
        )
        files = run_knowledge_review.PROTOCOL_FILES[protocol_id]
        (root / files["result"]).parent.mkdir(parents=True, exist_ok=True)
        (root / files["result"]).write_bytes(replay.result_bytes)
        (root / files["trace"]).write_bytes(replay.trace_bytes)
        prompt = run_knowledge_review.build_review_prompt(
            root, protocol_id, snapshot=snapshot, cache=cache,
        )
        receipt = run_knowledge_review.build_receipt(
            root, replay, snapshot=snapshot, cache=cache, prompt=prompt,
            boundary_contract=_fixture_boundary_contract(cache),
        )
        (root / files["receipt"]).write_bytes(
            run_knowledge_review._canonical_json(receipt)
        )
        invocations.append((protocol_id, snapshot, cache, replay))

    run_knowledge_review.seal_review_attestations(
        root,
        semantic_raw=_raw_evidence_path(root, "semantic.raw.jsonl"),
        adversarial_raw=_raw_evidence_path(root, "adversarial.raw.jsonl"),
        semantic_cache=root.parent / "semantic.cache",
        adversarial_cache=root.parent / "adversarial.cache",
    )
    return root
def _audit(root: Path) -> list[str]:
    return verify_legacy_v4_review_evidence(
        root,
        semantic_review=root / SEMANTIC_REVIEW_PATH,
        adversarial_review=root / ADVERSARIAL_REVIEW_PATH,
        semantic_raw=_raw_evidence_path(root, "semantic.raw.jsonl"),
        adversarial_raw=_raw_evidence_path(root, "adversarial.raw.jsonl"),
        semantic_cache=root.parent / "semantic.cache",
        adversarial_cache=root.parent / "adversarial.cache",
    )


def test_default_review_fetch_retries_incomplete_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    url = "https://example.com/manual.txt"
    responses = iter((
        _FetchResponse(url, IncompleteRead(b"partial", 8)),
        _FetchResponse(url, b"complete"),
    ))
    attempts = 0

    def opener(_handler: object) -> _FetchOpener:
        nonlocal attempts
        attempts += 1
        return _FetchOpener(next(responses))

    monkeypatch.setattr(run_knowledge_review, "build_opener", opener)
    fetched = run_knowledge_review._default_fetch(url, 1.0, 1024)

    assert attempts == 2
    assert fetched.body == b"complete"
    assert fetched.final_url == url


def test_default_review_fetch_fails_closed_after_fixed_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    url = "https://example.com/manual.txt"
    attempts = 0

    def opener(_handler: object) -> _FetchOpener:
        nonlocal attempts
        attempts += 1
        return _FetchOpener(
            _FetchResponse(url, IncompleteRead(b"private-partial", 8)),
        )

    monkeypatch.setattr(run_knowledge_review, "build_opener", opener)
    with pytest.raises(ValueError, match="failed after 3 attempts") as failure:
        run_knowledge_review._default_fetch(url, 1.0, 1024)

    assert attempts == run_knowledge_review.MAX_FETCH_ATTEMPTS
    assert "private-partial" not in str(failure.value)


def test_default_review_fetch_retries_declared_partial_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    url = "https://example.com/manual.txt"
    complete = b"complete-document"
    responses = iter((
        _LengthFetchResponse(
            url, complete[:5], declared_length=len(complete),
        ),
        _LengthFetchResponse(
            url, complete, declared_length=len(complete),
        ),
    ))
    attempts = 0

    def opener(_handler: object) -> _FetchOpener:
        nonlocal attempts
        attempts += 1
        return _FetchOpener(next(responses))

    monkeypatch.setattr(run_knowledge_review, "build_opener", opener)
    fetched = run_knowledge_review._default_fetch(url, 1.0, 1024)

    assert attempts == 2
    assert fetched.body == complete
    assert fetched.content_length == len(complete)


def test_default_review_fetch_retries_incomplete_pdf_without_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    url = "https://example.com/manual.pdf"
    complete = b"%PDF-1.7\npublic-document\n%%EOF\n"
    responses = iter((
        _LengthFetchResponse(
            url, complete[:-7], declared_length=len(complete) - 7,
            content_type="application/pdf",
        ),
        _LengthFetchResponse(
            url, complete, declared_length=len(complete),
            content_type="application/pdf",
        ),
    ))
    attempts = 0

    def opener(_handler: object) -> _FetchOpener:
        nonlocal attempts
        attempts += 1
        return _FetchOpener(next(responses))

    monkeypatch.setattr(run_knowledge_review, "build_opener", opener)
    fetched = run_knowledge_review._default_fetch(url, 1.0, 1024)

    assert attempts == 2
    assert fetched.body == complete


def test_final_review_gate_accepts_exact_repeated_review_bytes(
    reviewed_release_root: Path,
) -> None:
    assert _audit(reviewed_release_root) == []


def test_formal_v03_release_gate_rejects_valid_legacy_v4_evidence(
    reviewed_release_root: Path,
) -> None:
    issues = audit_release._audit_knowledge_review_release_gate(
        reviewed_release_root,
        semantic_review=reviewed_release_root / SEMANTIC_REVIEW_PATH,
        adversarial_review=reviewed_release_root / ADVERSARIAL_REVIEW_PATH,
        semantic_raw=_raw_evidence_path(
            reviewed_release_root, "semantic.raw.jsonl",
        ),
        adversarial_raw=_raw_evidence_path(
            reviewed_release_root, "adversarial.raw.jsonl",
        ),
        semantic_cache=reviewed_release_root.parent / "semantic.cache",
        adversarial_cache=reviewed_release_root.parent / "adversarial.cache",
    )
    assert issues == [
        "formal v0.3 release requires v6 six-invocation knowledge-review "
        "receipts; legacy v4 evidence is historical verification only"
    ]


def test_seal_rejects_boundary_contract_tamper(
    reviewed_release_root: Path,
) -> None:
    path = reviewed_release_root / SEMANTIC_REVIEW_RECEIPT_PATH
    receipt = json.loads(path.read_text(encoding="utf-8"))
    receipt["boundary_contract"]["canary_results"]["peer_sibling_denied"] = False
    _write_json(path, receipt)
    with pytest.raises(ValueError, match="boundary contract"):
        run_knowledge_review.seal_review_attestations(
            reviewed_release_root,
            semantic_raw=_raw_evidence_path(
                reviewed_release_root, "semantic.raw.jsonl",
            ),
            adversarial_raw=_raw_evidence_path(
                reviewed_release_root, "adversarial.raw.jsonl",
            ),
            semantic_cache=reviewed_release_root.parent / "semantic.cache",
            adversarial_cache=reviewed_release_root.parent / "adversarial.cache",
        )


def test_seal_and_release_audit_reject_different_review_runtimes(
    reviewed_release_root: Path,
) -> None:
    path = reviewed_release_root / ADVERSARIAL_REVIEW_RECEIPT_PATH
    receipt = json.loads(path.read_text(encoding="utf-8"))
    contract = receipt["boundary_contract"]
    runtime = contract["runtime_manifest"]
    executable = next(
        item for item in runtime["entries"]
        if item["relative_path"] == runtime["executable_relative_path"]
    )
    executable["sha256"] = "b" * 64
    runtime["executable_sha256"] = "b" * 64
    runtime["sha256"] = hashlib.sha256(
        run_knowledge_review._canonical_json({
            key: value for key, value in runtime.items() if key != "sha256"
        })
    ).hexdigest()
    contract["contract_sha256"] = hashlib.sha256(
        run_knowledge_review._canonical_json({
            key: value for key, value in contract.items()
            if key != "contract_sha256"
        })
    ).hexdigest()
    _write_json(path, receipt)
    with pytest.raises(ValueError, match=r"exact Codex\+bwrap identity"):
        run_knowledge_review.seal_review_attestations(
            reviewed_release_root,
            semantic_raw=_raw_evidence_path(
                reviewed_release_root, "semantic.raw.jsonl",
            ),
            adversarial_raw=_raw_evidence_path(
                reviewed_release_root, "adversarial.raw.jsonl",
            ),
            semantic_cache=reviewed_release_root.parent / "semantic.cache",
            adversarial_cache=reviewed_release_root.parent / "adversarial.cache",
        )
    assert any("identical Codex runtime" in item for item in _audit(reviewed_release_root))


def test_final_review_gate_rejects_post_review_semantic_change(
    reviewed_release_root: Path,
) -> None:
    implementation = reviewed_release_root / "src" / "hlsgraph" / "model.py"
    implementation.write_text("# changed after review\n", encoding="utf-8")
    issues = _audit(reviewed_release_root)
    assert any("implementation surface" in item for item in issues)


def test_final_review_gate_rejects_post_review_pack_semantic_change(
    reviewed_release_root: Path,
) -> None:
    path = (
        reviewed_release_root / "src" / "hlsgraph" / "knowledge" / "packs"
        / "amd_public_guidance_2024_2.json"
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    value["rules"][0]["rule_id"] += ".changed"
    _write_json(path, value)
    issues = _audit(reviewed_release_root)
    assert any("pack surfaces" in item for item in issues)


def test_final_review_gate_rejects_review_result_byte_tamper(
    reviewed_release_root: Path,
) -> None:
    semantic_path = reviewed_release_root / SEMANTIC_REVIEW_PATH
    value = json.loads(semantic_path.read_text(encoding="utf-8"))
    value["summary"] = "rejected_with_controlled_issues"
    _write_json(semantic_path, value)
    issues = _audit(reviewed_release_root)
    assert any(
        "semantic knowledge-review CLI receipt has invalid result_sha256" in item
        or SEMANTIC_REVIEW_PATH in item and "source hashes differ" in item
        for item in issues
    )


def test_final_review_gate_rejects_reused_invocation(
    reviewed_release_root: Path,
) -> None:
    pack_root = reviewed_release_root / "src" / "hlsgraph" / "knowledge" / "packs"
    for filename in PACKS:
        path = pack_root / filename
        value = json.loads(path.read_text(encoding="utf-8"))
        invocations = value["coverage"]["review_evidence"]["review_invocations"]
        invocations[1]["invocation_id"] = invocations[0]["invocation_id"]
        _write_json(path, value)
    issues = _audit(reviewed_release_root)
    assert any(
        "does not exactly match the two CLI receipt envelopes" in item
        for item in issues
    )


def test_final_review_gate_rejects_unreviewed_pack(
    reviewed_release_root: Path,
) -> None:
    path = (
        reviewed_release_root / "src" / "hlsgraph" / "knowledge" / "packs"
        / "axi_public_guidance.json"
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    value["metadata"]["review_status"] = "unreviewed"
    value["coverage"]["review_status"] = "unreviewed"
    _write_json(path, value)
    issues = _audit(reviewed_release_root)
    assert any("is not machine-repeated reviewed" in item for item in issues)


def test_final_review_gate_requires_exact_model_and_medium_effort(
    reviewed_release_root: Path,
) -> None:
    path = reviewed_release_root / SEMANTIC_REVIEW_RECEIPT_PATH
    value = json.loads(path.read_text(encoding="utf-8"))
    value["model"] = "not-the-approved-model"
    value["reasoning_effort"] = "low"
    _write_json(path, value)
    issues = _audit(reviewed_release_root)
    assert any("invalid model" in item for item in issues)
    assert any("invalid reasoning_effort" in item for item in issues)


def test_final_review_gate_requires_distinct_thread_and_event_stream(
    reviewed_release_root: Path,
) -> None:
    semantic = json.loads(
        (reviewed_release_root / SEMANTIC_REVIEW_RECEIPT_PATH).read_text(
            encoding="utf-8"
        )
    )
    path = reviewed_release_root / ADVERSARIAL_REVIEW_RECEIPT_PATH
    adversarial = json.loads(path.read_text(encoding="utf-8"))
    adversarial["thread_id"] = semantic["thread_id"]
    adversarial["event_stream_sha256"] = semantic["event_stream_sha256"]
    _write_json(path, adversarial)
    issues = _audit(reviewed_release_root)
    assert any("reuse a Codex thread ID" in item for item in issues)
    assert any("reuse one CLI event stream" in item for item in issues)


def test_final_review_gate_requires_every_exact_citation_verdict(
    reviewed_release_root: Path,
) -> None:
    path = reviewed_release_root / SEMANTIC_REVIEW_PATH
    value = json.loads(path.read_text(encoding="utf-8"))
    value["citation_results"].pop()
    _write_json(path, value)
    issues = _audit(reviewed_release_root)
    assert any("citation inventory differs" in item for item in issues)


def test_final_review_gate_rejects_stale_citation_artifact(
    reviewed_release_root: Path,
) -> None:
    path = reviewed_release_root / CITATION_AUDIT_PATH
    value = json.loads(path.read_text(encoding="utf-8"))
    value["references"][0]["reference_surface_sha256"] = "0" * 64
    _write_json(path, value)
    issues = _audit(reviewed_release_root)
    assert any(
        "manifest_sha256 is inconsistent" in item
        or "citation evidence mapping is stale" in item
        for item in issues
    )
    assert any("references differ from the current pack inventory" in item for item in issues)


def test_final_review_gate_rejects_weakened_evidence_schema(
    reviewed_release_root: Path,
) -> None:
    path = reviewed_release_root / run_knowledge_review.CITATION_EVIDENCE_SCHEMA_PATH
    value = json.loads(path.read_text(encoding="utf-8"))
    value["properties"]["entries"]["minItems"] = 0
    _write_json(path, value)
    issues = _audit(reviewed_release_root)
    assert any(
        "citation evidence schema bytes differ from the closed v1 contract" in item
        for item in issues
    )


def test_final_review_gate_binds_surface_helper_raw_bytes(
    reviewed_release_root: Path,
) -> None:
    path = reviewed_release_root / "tools" / "knowledge_review_surface.py"
    path.write_text(path.read_text(encoding="utf-8") + "\n# post-review change\n",
                    encoding="utf-8")
    issues = _audit(reviewed_release_root)
    assert any(SURFACE_HELPER_HASH_KEY in item for item in issues)


def test_final_review_gate_binds_release_auditor_raw_bytes(
    reviewed_release_root: Path,
) -> None:
    path = reviewed_release_root / "tools" / "audit_release.py"
    path.write_text(
        path.read_text(encoding="utf-8") + "\n# post-review change\n",
        encoding="utf-8",
    )
    issues = _audit(reviewed_release_root)
    assert any("tools/audit_release.py" in item for item in issues)


def test_final_review_gate_runs_runtime_coverage_validation(
    reviewed_release_root: Path,
) -> None:
    path = (
        reviewed_release_root / "src" / "hlsgraph" / "knowledge" / "packs"
        / "axi_public_guidance.json"
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    value["coverage"]["entries"] = []
    _write_json(path, value)
    issues = _audit(reviewed_release_root)
    assert any("fails the runtime contract" in item for item in issues)


def test_final_review_gate_rejects_extra_claimed_source_hash(
    reviewed_release_root: Path,
) -> None:
    pack_root = reviewed_release_root / "src" / "hlsgraph" / "knowledge" / "packs"
    for filename in PACKS:
        path = pack_root / filename
        value = json.loads(path.read_text(encoding="utf-8"))
        value["coverage"]["source_hashes"]["unverified-extra"] = "f" * 64
        _write_json(path, value)
    issues = _audit(reviewed_release_root)
    assert any("extra=['unverified-extra']" in item for item in issues)


@pytest.mark.skipif(os.name != "nt", reason="Windows junction test")
def test_final_review_gate_rejects_parent_junction(
    reviewed_release_root: Path,
) -> None:
    docs = reviewed_release_root / "docs"
    external = reviewed_release_root.parent / "external-review-docs"
    shutil.move(str(docs), str(external))
    completed = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(docs), str(external)],
        capture_output=True, text=True, check=False,
    )
    if completed.returncode != 0:
        pytest.skip("Windows junction creation is unavailable")
    issues = _audit(reviewed_release_root)
    assert any("linked or reparse path component" in item for item in issues)


@pytest.mark.skipif(os.name == "nt", reason="POSIX directory symlink test")
def test_final_review_gate_rejects_parent_symlink(
    reviewed_release_root: Path,
) -> None:
    docs = reviewed_release_root / "docs"
    external = reviewed_release_root.parent / "external-review-docs"
    shutil.move(str(docs), str(external))
    docs.symlink_to(external, target_is_directory=True)
    issues = _audit(reviewed_release_root)
    assert any("linked or reparse path component" in item for item in issues)


def test_review_rejects_raw_stream_inside_cache(tmp_path: Path) -> None:
    cache = tmp_path / "semantic.cache"
    raw = cache / "semantic.codex.jsonl"
    expected = "Windows is NO-GO" if os.name == "nt" else "outside the review cache"
    with pytest.raises(RuntimeError, match=expected):
        run_knowledge_review.run_review(
            ROOT, run_knowledge_review.SEMANTIC_PROTOCOL, raw, cache,
            codex_command="codex", timeout_seconds=1,
        )


def _review_boundary_fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    root = tmp_path / "checkout"
    cache = tmp_path / "cache"
    codex_home = tmp_path / "codex-home"
    external = tmp_path / "external"
    runtime = tmp_path / "runtime"
    peer = tmp_path / "peer-sibling"
    evidence = tmp_path / "evidence"
    for directory in (root, cache, codex_home, external, runtime, peer, evidence):
        directory.mkdir()
    (root / "README.md").write_text("public checkout\n", encoding="utf-8")
    (cache / run_knowledge_review.CACHE_MANIFEST_NAME).write_text(
        "{}\n", encoding="utf-8",
    )
    (codex_home / "auth.json").write_text("{}\n", encoding="utf-8")
    (external / "canary.txt").write_text("private\n", encoding="utf-8")
    (runtime / "codex").write_text("runtime\n", encoding="utf-8")
    (peer / "canary.txt").write_text("private\n", encoding="utf-8")
    (evidence / "canary.txt").write_text("private\n", encoding="utf-8")
    boundary: dict[str, object] = {
        "runtime_probe": str(runtime / "codex"),
        "auth_probe": str(codex_home / "auth.json"),
        "external_probe": str(external / "canary.txt"),
        "peer_sibling_probe": str(peer / "canary.txt"),
        "evidence_probe": str(evidence / "canary.txt"),
    }
    return root, cache, boundary


def test_review_boundary_uses_direct_allowlist_and_default_deny_probes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, cache, boundary = _review_boundary_fixture(tmp_path)
    commands: list[list[str]] = []

    monkeypatch.setattr(run_knowledge_review.shutil, "which", lambda *_args, **_kwargs: "/usr/bin/python3")

    def fake_canary(command: list[str], _environment: dict[str, str]) -> int:
        commands.append(command)
        return 0

    monkeypatch.setattr(run_knowledge_review, "_run_canary", fake_canary)
    results = run_knowledge_review._verify_boundary_canaries(
        codex="codex",
        root=root,
        cache_root=cache,
        profile_values=[],
        boundary=boundary,
        environment={"PATH": "/usr/bin:/bin"},
    )
    assert all(value is True for value in results.values())
    assert set(results) == {
        "cache_read", "runtime_read", "checkout_denied", "auth_denied",
        "external_denied", "peer_sibling_denied", "evidence_denied",
        "cache_write_denied",
    }
    assert len(commands) == 1
    assert str(boundary["peer_sibling_probe"]) in commands[0]
    assert str(cache / run_knowledge_review.CACHE_MANIFEST_NAME) in commands[0]
    assert str(boundary["runtime_probe"]) in commands[0]
    assert str(boundary["evidence_probe"]) in commands[0]
    assert not any("scandir" in value for item in commands for value in item)


def test_review_boundary_rejects_readable_peer_sibling_under_default_deny(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, cache, boundary = _review_boundary_fixture(tmp_path)
    monkeypatch.setattr(run_knowledge_review.shutil, "which", lambda *_args, **_kwargs: "/usr/bin/python3")

    def fake_canary(command: list[str], _environment: dict[str, str]) -> int:
        assert str(boundary["peer_sibling_probe"]) in command
        return 1 << 5

    monkeypatch.setattr(run_knowledge_review, "_run_canary", fake_canary)
    with pytest.raises(RuntimeError, match="peer_sibling_denied"):
        run_knowledge_review._verify_boundary_canaries(
            codex="codex",
            root=root,
            cache_root=cache,
            profile_values=[],
            boundary=boundary,
            environment={"PATH": "/usr/bin:/bin"},
        )


def test_review_private_evidence_must_not_overlap_runtime_or_cache(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "evidence"
    runtime = evidence / "runtime"
    cache = tmp_path / "cache"
    for path in (evidence, runtime, cache):
        path.mkdir()
    with pytest.raises(RuntimeError, match="Codex runtime"):
        run_knowledge_review._assert_private_evidence_disjoint(
            evidence, (("Codex runtime", runtime), ("cache", cache)),
        )
    run_knowledge_review._assert_private_evidence_disjoint(
        evidence, (("cache", cache),),
    )


def test_runtime_manifest_requires_exact_executable_and_closed_tree() -> None:
    empty_sha256 = hashlib.sha256(b"").hexdigest()
    root_only: dict[str, object] = {
        "schema_version": run_knowledge_review.RUNTIME_MANIFEST_SCHEMA_VERSION,
        "ownership_policy": run_knowledge_review.RUNTIME_OWNERSHIP_POLICY,
        "executable_relative_path": "codex",
        "executable_sha256": "a" * 64,
        "bubblewrap_relative_path": "codex-resources/bwrap",
        "bubblewrap_sha256": run_knowledge_review.OFFICIAL_CODEX_BWRAP_SHA256,
        "entries": [{
            "relative_path": ".", "kind": "dir", "size": 0,
            "mode": "0500", "sha256": empty_sha256,
        }],
    }
    root_only["sha256"] = hashlib.sha256(
        run_knowledge_review._canonical_json(root_only)
    ).hexdigest()
    with pytest.raises(ValueError, match=r"exact Codex\+bwrap identity"):
        run_knowledge_review._validate_runtime_manifest(root_only)

    orphan = copy.deepcopy(root_only)
    orphan["entries"].extend([{
        "relative_path": "missing/codex", "kind": "file", "size": 5,
        "mode": "0500", "sha256": run_knowledge_review.OFFICIAL_CODEX_ELF_SHA256,
    }, {
        "relative_path": "codex-resources", "kind": "dir", "size": 0,
        "mode": "0500", "sha256": empty_sha256,
    }, {
        "relative_path": "codex-resources/bwrap", "kind": "file", "size": 5,
        "mode": "0500", "sha256": run_knowledge_review.OFFICIAL_CODEX_BWRAP_SHA256,
    }])
    orphan["executable_relative_path"] = "missing/codex"
    orphan["executable_sha256"] = run_knowledge_review.OFFICIAL_CODEX_ELF_SHA256
    orphan["entries"].sort(key=lambda item: item["relative_path"])
    orphan.pop("sha256")
    orphan["sha256"] = hashlib.sha256(
        run_knowledge_review._canonical_json(orphan)
    ).hexdigest()
    with pytest.raises(ValueError, match="incomplete directory tree"):
        run_knowledge_review._validate_runtime_manifest(orphan)


def test_review_command_contract_is_default_deny_minimal_allowlist() -> None:
    argv = run_knowledge_review.canonical_command_argv(
        run_knowledge_review.SEMANTIC_PROTOCOL,
    )
    assert not any(".extends=" in value for value in argv)
    filesystem = next(
        value for value in argv
        if value.startswith(
            f"permissions.{run_knowledge_review.PERMISSION_PROFILE}.filesystem="
        )
    )
    assert filesystem.endswith(
        '{":minimal"="read","$CACHE"="read","$CODEX_RUNTIME"="read"}'
    )
    assert "deny" not in filesystem


def test_review_runtime_path_pins_bundled_bwrap_before_system_tools() -> None:
    assert run_knowledge_review._runtime_initial_process_path(Path("/frozen/runtime")) == (
        "/frozen/runtime/codex-resources:/usr/bin:/bin"
    )


@pytest.mark.skipif(os.name == "nt", reason="POSIX ownership and mode contract")
def test_review_cache_parent_must_be_private_and_single_child(tmp_path: Path) -> None:
    parent = tmp_path / "semantic-cache-parent"
    parent.mkdir(mode=0o700)
    cache = parent / "cache"
    cache.mkdir(mode=0o700)
    assert run_knowledge_review._validate_review_cache_parent(cache) == (
        run_knowledge_review.CACHE_PARENT_POLICY
    )

    sibling = parent / "raw-output.jsonl"
    sibling.write_text("private\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="contain only the frozen cache"):
        run_knowledge_review._validate_review_cache_parent(cache)
    sibling.unlink()

    parent.chmod(0o755)
    with pytest.raises(RuntimeError, match="mode 0700"):
        run_knowledge_review._validate_review_cache_parent(cache)


@pytest.mark.skipif(os.name == "nt", reason="POSIX runtime tree contract")
def test_review_runtime_manifest_is_pathless_stable_and_link_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir(mode=0o700)
    codex = runtime / "codex"
    codex.write_bytes(b"frozen-codex-runtime")
    codex.chmod(0o500)
    resources = runtime / "codex-resources"
    resources.mkdir(mode=0o700)
    bwrap = resources / "bwrap"
    bwrap.write_bytes(b"frozen-codex-bwrap")
    bwrap.chmod(0o500)
    resources.chmod(0o500)
    runtime.chmod(0o500)
    monkeypatch.setattr(run_knowledge_review, "_mount_fstype", lambda _path: "ext4")
    monkeypatch.setattr(
        run_knowledge_review, "OFFICIAL_CODEX_ELF_SHA256",
        hashlib.sha256(b"frozen-codex-runtime").hexdigest(),
    )
    monkeypatch.setattr(
        run_knowledge_review, "OFFICIAL_CODEX_BWRAP_SHA256",
        hashlib.sha256(b"frozen-codex-bwrap").hexdigest(),
    )
    first = run_knowledge_review._freeze_runtime_manifest(codex)
    second = run_knowledge_review._freeze_runtime_manifest(codex)
    assert first == second
    assert first["entries"][0]["relative_path"] == "."
    assert all(not entry["relative_path"].startswith("/") for entry in first["entries"])
    assert str(tmp_path) not in json.dumps(first, sort_keys=True)

    runtime.chmod(0o700)
    linked = runtime / "linked"
    linked.symlink_to(codex)
    runtime.chmod(0o500)
    with pytest.raises(RuntimeError, match="codex and codex-resources|linked entry"):
        run_knowledge_review._freeze_runtime_manifest(codex)

    runtime.chmod(0o700)
    linked.unlink()
    extra = runtime / "NOTICE"
    extra.write_text("unexpected", encoding="utf-8")
    extra.chmod(0o400)
    runtime.chmod(0o500)
    with pytest.raises(RuntimeError, match="exactly codex and codex-resources"):
        run_knowledge_review._freeze_runtime_manifest(codex)


@pytest.mark.skipif(os.name == "nt", reason="POSIX runtime alias contract")
def test_review_runtime_rejects_bundled_bwrap_tamper_and_aliases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir(mode=0o700)
    codex = runtime / "codex"
    codex.write_bytes(b"frozen-codex-runtime")
    codex.chmod(0o500)
    resources = runtime / "codex-resources"
    resources.mkdir(mode=0o700)
    bwrap = resources / "bwrap"
    bwrap.write_bytes(b"frozen-codex-bwrap")
    bwrap.chmod(0o500)
    resources.chmod(0o500)
    runtime.chmod(0o500)
    monkeypatch.setattr(run_knowledge_review, "_mount_fstype", lambda _path: "ext4")
    monkeypatch.setattr(
        run_knowledge_review, "OFFICIAL_CODEX_ELF_SHA256",
        hashlib.sha256(b"frozen-codex-runtime").hexdigest(),
    )
    monkeypatch.setattr(
        run_knowledge_review, "OFFICIAL_CODEX_BWRAP_SHA256",
        hashlib.sha256(b"frozen-codex-bwrap").hexdigest(),
    )
    run_knowledge_review._freeze_runtime_manifest(codex)

    bwrap.chmod(0o700)
    bwrap.write_bytes(b"tampered-codex-bwrap")
    bwrap.chmod(0o500)
    with pytest.raises(RuntimeError, match="fixed official .* bundled bwrap"):
        run_knowledge_review._freeze_runtime_manifest(codex)

    resources.chmod(0o700)
    bwrap.unlink()
    external = tmp_path / "external-bwrap"
    external.write_bytes(b"frozen-codex-bwrap")
    external.chmod(0o500)
    bwrap.symlink_to(external)
    resources.chmod(0o500)
    with pytest.raises(RuntimeError, match="linked entry"):
        run_knowledge_review._freeze_runtime_manifest(codex)

    resources.chmod(0o700)
    bwrap.unlink()
    os.link(external, bwrap)
    resources.chmod(0o500)
    with pytest.raises(RuntimeError, match="unsafe file"):
        run_knowledge_review._freeze_runtime_manifest(codex)


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlink alias test")
def test_review_rejects_raw_cache_symlink_alias(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    cache = real / "semantic.cache"
    raw = alias / "semantic.cache" / "semantic.codex.jsonl"
    with pytest.raises(RuntimeError, match="linked path component"):
        run_knowledge_review.run_review(
            ROOT, run_knowledge_review.SEMANTIC_PROTOCOL, raw, cache,
            codex_command="codex", timeout_seconds=1,
        )


@pytest.mark.skipif(os.name != "nt", reason="Windows junction alias test")
def test_review_rejects_raw_cache_junction_alias(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    completed = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(alias), str(real)],
        capture_output=True, text=True, check=False,
    )
    if completed.returncode != 0:
        pytest.skip("Windows junction creation is unavailable")
    cache = real / "semantic.cache"
    raw = alias / "semantic.cache" / "semantic.codex.jsonl"
    with pytest.raises(RuntimeError, match="Windows is NO-GO"):
        run_knowledge_review.run_review(
            ROOT, run_knowledge_review.SEMANTIC_PROTOCOL, raw, cache,
            codex_command="codex", timeout_seconds=1,
        )


def test_review_snapshot_binds_release_auditor() -> None:
    snapshot = run_knowledge_review.freeze_review_snapshot(
        ROOT, run_knowledge_review.SEMANTIC_PROTOCOL,
    )
    assert run_knowledge_review.RELEASE_AUDITOR_PATH in snapshot.file_map


def test_public_amd_mapping_accepts_real_khub_opaque_ids() -> None:
    value = json.loads(
        (ROOT / run_knowledge_review.CITATION_EVIDENCE_PATH).read_text(
            encoding="utf-8",
        )
    )
    citation_bytes = (ROOT / run_knowledge_review.CITATION_AUDIT_PATH).read_bytes()
    assert value["citation_audit_sha256"] == hashlib.sha256(
        citation_bytes,
    ).hexdigest()
    assert [row["citation_url"] for row in value["entries"]] == sorted({
        row["citation_url"] for row in value["entries"]
    })
    amd = [
        row for row in value["entries"]
        if row["resolver_id"].startswith("amd.docs.khub.")
    ]
    assert amd
    assert any(
        "~" in str(row["identity"].get(key, ""))
        for row in amd
        for key in ("publication_id", "toc_id", "content_id")
    )
    snapshot = run_knowledge_review.freeze_review_snapshot(
        ROOT, run_knowledge_review.SEMANTIC_PROTOCOL,
    )
    assert len(run_knowledge_review._citation_evidence_rows(snapshot)) == 46


def test_public_github_rule_evidence_uses_exact_raw_line_ranges() -> None:
    value = json.loads(
        (ROOT / run_knowledge_review.CITATION_EVIDENCE_PATH).read_text(
            encoding="utf-8",
        )
    )
    rows = [
        row for row in value["entries"]
        if row["resolver_id"] == "github.raw.lines.v1"
    ]
    assert len(rows) == 13
    assert len({row["evidence_url"] for row in rows}) == 5
    for row in rows:
        identity = row["identity"]
        assert row["citation_url"].startswith("https://github.com/")
        assert row["evidence_url"] == (
            "https://raw.githubusercontent.com/"
            f"{identity['repository']}/{identity['commit']}/{identity['path']}"
        )
        assert len(identity["commit"]) == 40
        assert 1 <= identity["start_line"] <= identity["end_line"]
        assert (
            identity["end_line"] - identity["start_line"] + 1
            <= run_knowledge_review.MAX_EVIDENCE_LINE_RANGE
        )
        assert len(identity["source_sha256"]) == 64
        assert len(identity["slice_sha256"]) == 64


def test_public_document_evidence_is_version_and_body_bound() -> None:
    mapping = json.loads(
        (ROOT / run_knowledge_review.CITATION_EVIDENCE_PATH).read_text(
            encoding="utf-8",
        )
    )
    citation = json.loads(
        (ROOT / run_knowledge_review.CITATION_AUDIT_PATH).read_text(
            encoding="utf-8",
        )
    )
    fetches = {row["fetch_url"]: row for row in citation["fetches"]}
    direct = [
        row for row in mapping["entries"]
        if row["resolver_id"] == "direct.sha256.v1"
    ]
    immutable = [
        row for row in mapping["entries"]
        if row["resolver_id"] == "github.raw.document.v1"
    ]
    assert len(direct) == 3
    assert len(immutable) == 4
    for row in direct:
        identity = row["identity"]
        fetched = fetches[row["citation_url"]]
        assert identity["body_sha256"] == fetched["sha256"]
        assert identity["body_size"] == fetched["byte_count"]
        assert identity["content_type"] == fetched["content_type"]
        assert row["evidence_url"] == row["citation_url"]
    for row in immutable:
        identity = row["identity"]
        assert identity["document_version"] == f"git-{identity['commit']}"
        assert row["evidence_url"] == (
            "https://raw.githubusercontent.com/"
            f"{identity['repository']}/{identity['commit']}/{identity['path']}"
        )

    mutated = copy.deepcopy(mapping)
    next(
        row for row in mutated["entries"]
        if row["resolver_id"] == "direct.sha256.v1"
    )["identity"]["body_sha256"] = "0" * 64
    snapshot = run_knowledge_review.freeze_review_snapshot(
        ROOT, run_knowledge_review.SEMANTIC_PROTOCOL,
    )
    with pytest.raises(ValueError, match="direct SHA-256 document identity"):
        run_knowledge_review._validate_citation_evidence_mapping(
            mutated,
            citation_audit_sha256=mapping["citation_audit_sha256"],
            exact_urls=snapshot.exact_citation_urls,
            expected_references=citation["references"],
            expected_fetches=citation["fetches"],
        )


def test_github_raw_line_derivation_checks_source_and_slice_hashes() -> None:
    body = b"heading\nfirst\nsecond\nfooter\n"
    selected = b"first\nsecond\n"
    mapping = {
        "identity": {
            "repository": "owner/repository", "commit": "a" * 40,
            "path": "docs/spec.md", "source_sha256": hashlib.sha256(body).hexdigest(),
            "start_line": 2, "end_line": 3,
            "slice_sha256": hashlib.sha256(selected).hexdigest(),
        },
    }
    fetched = run_knowledge_review.TrustedFetch(
        200, "https://raw.githubusercontent.com/owner/repository/" + "a" * 40
        + "/docs/spec.md", (), "text/plain", body, charset="utf-8",
    )
    derived = run_knowledge_review._github_line_range_derivation(mapping, fetched)
    assert derived.text == selected
    assert derived.parser_id == "hlsgraph.review.github-raw-lines.v1"

    mapping["identity"]["slice_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="selected range differs"):
        run_knowledge_review._github_line_range_derivation(mapping, fetched)


def test_document_only_pdfs_do_not_invoke_text_parser(
    reviewed_release_root: Path,
) -> None:
    mapping = json.loads(
        (reviewed_release_root / run_knowledge_review.CITATION_EVIDENCE_PATH)
        .read_text(encoding="utf-8")
    )
    delegated = _evidence_fetcher(mapping)

    def forbidden_parser(_body: bytes) -> run_knowledge_review.TextDerivation:
        raise AssertionError("document-only PDF parser must not run")

    snapshot = run_knowledge_review.freeze_review_snapshot(
        reviewed_release_root, run_knowledge_review.SEMANTIC_PROTOCOL,
    )
    cache = run_knowledge_review.create_review_cache(
        reviewed_release_root, snapshot,
        reviewed_release_root.parent / "document-only-pdf.cache",
        fetcher=delegated, pdf_text_extractor=forbidden_parser,
    )
    arm = [
        entry for entry in cache.manifest["citations"]
        if "documentation-service.arm.com/static/" in entry["requested_url"]
    ]
    assert len(arm) == 2
    assert all(entry["available"] is True for entry in arm)
    assert all(entry["inspection_required"] is False for entry in arm)
    assert all(entry["inspection_chunks"] == [] for entry in arm)
    assert all(entry["parser_id"] is None for entry in arm)


def test_amd_map_and_pages_are_fetched_once_per_publication(
    reviewed_release_root: Path,
) -> None:
    mapping = json.loads(
        (reviewed_release_root / run_knowledge_review.CITATION_EVIDENCE_PATH)
        .read_text(encoding="utf-8")
    )
    delegated = _evidence_fetcher(mapping)
    calls: dict[str, int] = {}

    def counted(url: str, timeout: float, max_bytes: int):
        calls[url] = calls.get(url, 0) + 1
        return delegated(url, timeout, max_bytes)

    snapshot = run_knowledge_review.freeze_review_snapshot(
        reviewed_release_root, run_knowledge_review.SEMANTIC_PROTOCOL,
    )
    run_knowledge_review.create_review_cache(
        reviewed_release_root, snapshot,
        reviewed_release_root.parent / "resolver-reuse.cache",
        fetcher=counted, pdf_text_extractor=_fixture_pdf_text,
    )
    publications = {
        row["identity"]["publication_id"]
        for row in mapping["entries"]
        if row["resolver_id"].startswith("amd.docs.khub.")
    }
    for publication_id in publications:
        map_url = f"https://docs.amd.com/api/khub/maps/{publication_id}"
        assert calls[map_url] == 1
        assert calls[map_url + "/pages"] == 1
    assert snapshot.file_map[
        run_knowledge_review.RELEASE_AUDITOR_PATH
    ].sha256 == hashlib.sha256(
        (ROOT / run_knowledge_review.RELEASE_AUDITOR_PATH).read_bytes()
    ).hexdigest()


def test_shared_evidence_fetch_failure_is_reused_fail_closed(
    reviewed_release_root: Path,
) -> None:
    mapping = json.loads(
        (reviewed_release_root / run_knowledge_review.CITATION_EVIDENCE_PATH)
        .read_text(encoding="utf-8")
    )
    delegated = _evidence_fetcher(mapping)
    shared_url = next(
        row["evidence_url"] for row in mapping["entries"]
        if row["resolver_id"] == "github.raw.lines.v1"
        and sum(
            candidate["evidence_url"] == row["evidence_url"]
            for candidate in mapping["entries"]
        ) > 1
    )
    calls = 0

    def fail_shared(url: str, timeout: float, max_bytes: int):
        nonlocal calls
        if url == shared_url:
            calls += 1
            raise ValueError("fixture transport failure")
        return delegated(url, timeout, max_bytes)

    snapshot = run_knowledge_review.freeze_review_snapshot(
        reviewed_release_root, run_knowledge_review.SEMANTIC_PROTOCOL,
    )
    cache = run_knowledge_review.create_review_cache(
        reviewed_release_root, snapshot,
        reviewed_release_root.parent / "shared-fetch-failure.cache",
        fetcher=fail_shared,
    )
    affected = [
        row for row in cache.manifest["citations"]
        if row["evidence_url"] == shared_url
    ]
    assert len(affected) > 1
    assert calls == 1
    assert all(row["available"] is False for row in affected)


def test_final_review_gate_rejects_search_or_write_trace_operations(
    reviewed_release_root: Path,
) -> None:
    path = reviewed_release_root / SEMANTIC_REVIEW_TRACE_PATH
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0] = {
        "schema_version": REVIEW_TRACE_SCHEMA_VERSION,
        "sequence": 1,
        "kind": "web_search",
        "query": "substitute source",
    }
    _write_jsonl(path, rows)
    issues = _audit(reviewed_release_root)
    assert any("forbidden search, write, command" in item for item in issues)


def test_final_review_gate_rejects_substitute_locator_in_trace(
    reviewed_release_root: Path,
) -> None:
    path = reviewed_release_root / ADVERSARIAL_REVIEW_TRACE_PATH
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    fetch = next(item for item in rows if item["kind"] == "citation_chunk_read")
    fetch["requested_url"] = "https://example.invalid/search?q=substitute"
    _write_jsonl(path, rows)
    issues = _audit(reviewed_release_root)
    assert any("unapproved locator" in item for item in issues)
    assert any("does not inspect every exact locator" in item for item in issues)


def test_final_review_gate_replays_and_rejects_unknown_raw_tool(
    reviewed_release_root: Path,
) -> None:
    path = _raw_evidence_path(reviewed_release_root, "semantic.raw.jsonl")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[2] = {
        "type": "item.completed",
        "item": {"id": "web-tool-0001", "type": "web_search", "query": "substitute"},
    }
    _write_jsonl(path, rows)
    issues = _audit(reviewed_release_root)
    assert any("forbidden or unknown tool" in item for item in issues)


def test_final_review_gate_replays_and_rejects_executable_raw_command(
    reviewed_release_root: Path,
) -> None:
    path = _raw_evidence_path(reviewed_release_root, "adversarial.raw.jsonl")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    command = next(
        item for item in rows
        if item.get("type") == "item.completed"
        and item.get("item", {}).get("type") == "command_execution"
    )
    call_id = command["item"]["id"]
    for item in rows:
        if item.get("item", {}).get("id") == call_id:
            item["item"]["command"] = "python -c pass"
    _write_jsonl(path, rows)
    issues = _audit(reviewed_release_root)
    assert any("unapproved command executable" in item for item in issues)


def test_final_review_gate_rejects_raw_result_not_equal_to_committed_result(
    reviewed_release_root: Path,
) -> None:
    path = _raw_evidence_path(reviewed_release_root, "semantic.raw.jsonl")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    message = next(
        item for item in reversed(rows)
        if item.get("type") == "item.completed"
        and item.get("item", {}).get("type") == "agent_message"
    )
    value = json.loads(message["item"]["text"])
    value["approved"] = False
    value["summary"] = "rejected_with_controlled_issues"
    message["item"]["text"] = json.dumps(value, sort_keys=True)
    _write_jsonl(path, rows)
    issues = _audit(reviewed_release_root)
    assert any("was not derived from its raw Codex stream" in item for item in issues)


def test_documented_script_path_preflight_imports_without_network_or_model() -> None:
    completed = subprocess.run(
        [
            sys.executable, "tools/run_knowledge_review.py", "preflight",
            "--root", str(ROOT), "--protocol", SEMANTIC_REVIEW_PROTOCOL,
        ],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, check=False, timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["network_used"] is False
    assert result["model_used"] is False
    assert result["required_file_count"] > 0


def test_prompt_injects_exact_snapshot_and_cache_hashes(
    reviewed_release_root: Path,
) -> None:
    snapshot, cache = _snapshot_cache(reviewed_release_root)
    prompt = run_knowledge_review.build_review_prompt(
        reviewed_release_root, SEMANTIC_REVIEW_PROTOCOL,
        snapshot=snapshot, cache=cache,
    ).decode("utf-8")
    for digest in (
        snapshot.sha256, snapshot.implementation_surface_sha256,
        snapshot.citation_audit_sha256, snapshot.output_schema_sha256,
        snapshot.receipt_schema_sha256, cache.sha256,
        *snapshot.surfaces.values(),
    ):
        assert digest in prompt
    assert "The model has no network" in prompt
    assert "head -n COUNT PATH" in prompt
    assert "sed -n" not in prompt


def test_snapshot_and_cache_fail_after_source_mutation(
    reviewed_release_root: Path,
) -> None:
    snapshot, cache = _snapshot_cache(reviewed_release_root)
    path = reviewed_release_root / "src" / "hlsgraph" / "model.py"
    path.write_bytes(path.read_bytes() + b"\n# snapshot mutation\n")
    changed = run_knowledge_review.freeze_review_snapshot(
        reviewed_release_root, SEMANTIC_REVIEW_PROTOCOL,
    )
    assert changed != snapshot
    with pytest.raises(ValueError, match="exact snapshot"):
        run_knowledge_review.load_review_cache(cache.root, changed)


def test_model_command_contract_disables_network_and_uses_cache_cwd() -> None:
    argv = run_knowledge_review.canonical_command_argv(SEMANTIC_REVIEW_PROTOCOL)
    assert f"permissions.{run_knowledge_review.PERMISSION_PROFILE}.network.enabled=false" in argv
    assert "--cd" in argv
    assert argv[argv.index("--cd") + 1] == "$CACHE"
    assert "$ROOT/tools/knowledge_review.schema.json" in argv
    assert not any(value.endswith("network.enabled=true") for value in argv)


def test_cache_rejects_cross_host_redirect_chain(
    reviewed_release_root: Path,
) -> None:
    snapshot = run_knowledge_review.freeze_review_snapshot(
        reviewed_release_root, SEMANTIC_REVIEW_PROTOCOL,
    )

    def cross_host(
        url: str, _timeout: float, _max_bytes: int,
    ) -> run_knowledge_review.TrustedFetch:
        final = "https://example.invalid/substitute"
        return run_knowledge_review.TrustedFetch(
            200, final, (url, final), "text/plain", b"substitute",
            charset="utf-8",
        )

    cache = run_knowledge_review.create_review_cache(
        reviewed_release_root, snapshot,
        reviewed_release_root.parent / "cross-host.cache", fetcher=cross_host,
    )
    cache = run_knowledge_review.load_review_cache(cache.root, snapshot)
    assert all(entry["available"] is False for entry in cache.manifest["citations"])
    assert all(entry["error_code"] == "ValueError" for entry in cache.manifest["citations"])


def test_cache_rejects_same_host_redirect_that_changes_locator(
    reviewed_release_root: Path,
) -> None:
    snapshot = run_knowledge_review.freeze_review_snapshot(
        reviewed_release_root, SEMANTIC_REVIEW_PROTOCOL,
    )

    def same_host(
        url: str, _timeout: float, _max_bytes: int,
    ) -> run_knowledge_review.TrustedFetch:
        parts = run_knowledge_review.urlsplit(url)
        final = f"https://{parts.netloc}/hlsgraph-final"
        return run_knowledge_review.TrustedFetch(
            200, final, (url, final), "text/plain",
            hashlib.sha256(url.encode("utf-8")).hexdigest().encode("ascii"),
            charset="utf-8",
        )

    cache = run_knowledge_review.create_review_cache(
        reviewed_release_root, snapshot,
        reviewed_release_root.parent / "same-host.cache", fetcher=same_host,
    )
    cache = run_knowledge_review.load_review_cache(cache.root, snapshot)
    assert all(entry["available"] is False for entry in cache.manifest["citations"])
    assert all(entry["error_code"] == "ValueError" for entry in cache.manifest["citations"])


def test_cache_allows_only_identical_same_host_redirect_chain(
    reviewed_release_root: Path,
) -> None:
    snapshot = run_knowledge_review.freeze_review_snapshot(
        reviewed_release_root, SEMANTIC_REVIEW_PROTOCOL,
    )

    mapping = json.loads(
        (reviewed_release_root / run_knowledge_review.CITATION_EVIDENCE_PATH)
        .read_text(encoding="utf-8")
    )
    delegated = _evidence_fetcher(mapping)

    def identical(url: str, timeout: float, max_bytes: int):
        fetched = delegated(url, timeout, max_bytes)
        return run_knowledge_review.TrustedFetch(
            fetched.status, url, (url, url), fetched.content_type, fetched.body,
            charset=fetched.charset,
        )

    cache = run_knowledge_review.create_review_cache(
        reviewed_release_root, snapshot,
        reviewed_release_root.parent / "identical-redirect.cache", fetcher=identical,
        pdf_text_extractor=_fixture_pdf_text,
    )
    cache = run_knowledge_review.load_review_cache(cache.root, snapshot)
    assert all(entry["available"] is True for entry in cache.manifest["citations"])


def test_pdf_without_controlled_parser_is_unavailable_and_cannot_approve(
    reviewed_release_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = run_knowledge_review.freeze_review_snapshot(
        reviewed_release_root, SEMANTIC_REVIEW_PROTOCOL,
    )
    mapping = json.loads(
        (reviewed_release_root / run_knowledge_review.CITATION_EVIDENCE_PATH)
        .read_text(encoding="utf-8")
    )
    delegated = _evidence_fetcher(mapping)
    validated_evidence = run_knowledge_review._citation_evidence_rows(snapshot)
    references = run_knowledge_review._citation_reference_rows(snapshot)
    promoted = False
    patched_references: list[dict[str, object]] = []
    for reference in references:
        row = dict(reference)
        if (not promoted
                and "documentation-service.arm.com/static/"
                in str(row["citation_url"])):
            row["reference_kind"] = "rule"
            promoted = True
        patched_references.append(row)
    assert promoted
    monkeypatch.setattr(
        run_knowledge_review, "_citation_reference_rows",
        lambda _snapshot: patched_references,
    )
    monkeypatch.setattr(
        run_knowledge_review, "_citation_evidence_rows",
        lambda _snapshot: validated_evidence,
    )

    def with_unparsed_pdf(url: str, timeout: float, max_bytes: int):
        if "documentation-service.arm.com/static/" not in url:
            return delegated(url, timeout, max_bytes)
        body = b"%PDF-1.7\nHLSGRAPH-CACHE-PDF-" + hashlib.sha256(
            url.encode("utf-8")
        ).hexdigest().encode("ascii") + b"\n%%EOF\n"
        return run_knowledge_review.TrustedFetch(
            200, url, (url,), "application/pdf", body, charset=None,
        )

    cache = run_knowledge_review.create_review_cache(
        reviewed_release_root, snapshot,
        reviewed_release_root.parent / "no-pdf-parser.cache",
        fetcher=with_unparsed_pdf,
    )
    cache = run_knowledge_review.load_review_cache(cache.root, snapshot)
    pdf_entries = [
        entry for entry in cache.manifest["citations"]
        if (entry["content_type"] == "application/pdf"
            and entry["inspection_required"] is True)
    ]
    assert len(pdf_entries) == 1
    assert all(entry["available"] is False for entry in pdf_entries)
    assert all(
        entry["error_code"] == "citation_text_unavailable"
        for entry in pdf_entries
    )
    raw = _raw_stream(
        snapshot=snapshot, cache=cache,
        result=_approved_result(snapshot, summary="Must fail closed."),
        thread_id="thread-no-pdf-parser-0001",
    )
    with pytest.raises(ValueError, match="uninspected evidence"):
        run_knowledge_review.replay_raw_review(
            reviewed_release_root, SEMANTIC_REVIEW_PROTOCOL, raw,
            snapshot=snapshot, cache=cache,
        )


def test_portal_javascript_shell_is_not_inspectable_evidence() -> None:
    body = b"""<!doctype html><html><body><div id=\"root\"></div>
    <noscript>JavaScript is required.</noscript><script src=\"app.js\"></script>
    </body></html>"""
    fetched = run_knowledge_review.TrustedFetch(
        200, "https://docs.amd.com/r/example",
        ("https://docs.amd.com/r/example",), "text/html", body,
        charset="utf-8",
    )
    assert run_knowledge_review._text_derivation(fetched) is None


def test_sanitized_raw_contains_no_cached_citation_body_or_text(
    reviewed_release_root: Path,
) -> None:
    _snapshot, cache = _snapshot_cache(reviewed_release_root)
    raw = _raw_evidence_path(
        reviewed_release_root, "semantic.raw.jsonl",
    ).read_bytes()
    for entry in cache.manifest["citations"]:
        for key in ("body_path", "inspection_path"):
            relative = entry.get(key)
            if not relative:
                continue
            payload = (cache.root / relative).read_bytes()
            assert payload not in raw
            try:
                escaped = json.dumps(
                    payload.decode("utf-8"), ensure_ascii=False,
                )[1:-1].encode("utf-8")
            except UnicodeDecodeError:
                continue
            assert escaped not in raw


def test_cache_hash_tamper_and_unmanifested_file_fail_closed(
    reviewed_release_root: Path,
) -> None:
    snapshot, cache = _snapshot_cache(reviewed_release_root)
    entry = next(
        item for item in cache.manifest["citations"]
        if item.get("inspection_path")
    )
    target = cache.root / entry["inspection_path"]
    target.write_bytes(target.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="stale"):
        run_knowledge_review.load_review_cache(cache.root, snapshot)


def test_cache_rejects_unmanifested_extra_file(
    reviewed_release_root: Path,
) -> None:
    snapshot, cache = _snapshot_cache(reviewed_release_root)
    # Inject an unmanifested file without weakening the independently checked
    # frozen-tree mode.  The ordinary private writer intentionally prepares
    # parent directories as 0700, which would make this test exercise the mode
    # guard before it reaches the closed filesystem inventory.
    cache.root.chmod(0o700)
    extra = cache.root / "extra.txt"
    extra.write_bytes(b"extra")
    if os.name != "nt":
        extra.chmod(run_knowledge_review.CACHE_FILE_MODE)
        cache.root.chmod(run_knowledge_review.CACHE_DIRECTORY_MODE)
    with pytest.raises(ValueError, match="unmanifested"):
        run_knowledge_review.load_review_cache(cache.root, snapshot)


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits are required")
def test_cache_tree_is_frozen_0500_directories_and_0400_files(
    reviewed_release_root: Path,
) -> None:
    _snapshot, cache = _snapshot_cache(reviewed_release_root)
    for current, _directories, files in os.walk(cache.root):
        assert Path(current).stat().st_mode & 0o777 == 0o500
        for name in files:
            assert (Path(current) / name).stat().st_mode & 0o777 == 0o400
    raw = _raw_evidence_path(reviewed_release_root, "semantic.raw.jsonl")
    assert raw.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("mutation", ["empty", "missing-start", "missing-status", "bad-output"])
def test_raw_replay_rejects_incomplete_lifecycle_and_output(
    reviewed_release_root: Path, mutation: str,
) -> None:
    snapshot, cache = _snapshot_cache(reviewed_release_root)
    raw_path = _raw_evidence_path(reviewed_release_root, "semantic.raw.jsonl")
    rows = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()]
    if mutation == "empty":
        raw = b""
    elif mutation == "missing-start":
        rows.pop(next(
            index for index, row in enumerate(rows)
            if row.get("type") == "item.started"
        ))
        raw = run_knowledge_review._canonical_jsonl(rows)
    elif mutation == "missing-status":
        completed = next(
            row for row in rows
            if row.get("type") == "item.completed"
            and row.get("item", {}).get("type") == "command_execution"
        )
        completed["item"].pop("status")
        raw = run_knowledge_review._canonical_jsonl(rows)
    else:
        completed = next(
            row for row in rows
            if row.get("type") == "item.completed"
            and row.get("item", {}).get("type") == "command_execution"
        )
        completed["item"]["aggregated_output"] += "tamper"
        raw = run_knowledge_review._canonical_jsonl(rows)
    with pytest.raises(ValueError):
        run_knowledge_review.replay_raw_review(
            reviewed_release_root, SEMANTIC_REVIEW_PROTOCOL, raw,
            snapshot=snapshot, cache=cache,
        )


def test_seal_is_deterministic_and_preserves_semantic_surfaces(
    reviewed_release_root: Path,
) -> None:
    pack_root = reviewed_release_root / "src" / "hlsgraph" / "knowledge" / "packs"
    before_bytes = {path.name: path.read_bytes() for path in pack_root.glob("*.json")}
    before_surfaces = {
        path.name: knowledge_review_surface.surface_sha256(path)
        for path in pack_root.glob("*.json")
    }
    run_knowledge_review.seal_review_attestations(
        reviewed_release_root,
        semantic_raw=_raw_evidence_path(
            reviewed_release_root, "semantic.raw.jsonl",
        ),
        adversarial_raw=_raw_evidence_path(
            reviewed_release_root, "adversarial.raw.jsonl",
        ),
        semantic_cache=reviewed_release_root.parent / "semantic.cache",
        adversarial_cache=reviewed_release_root.parent / "adversarial.cache",
    )
    assert {path.name: path.read_bytes() for path in pack_root.glob("*.json")} == before_bytes
    assert {
        path.name: knowledge_review_surface.surface_sha256(path)
        for path in pack_root.glob("*.json")
    } == before_surfaces


def test_seal_tamper_does_not_partially_update_packs(
    reviewed_release_root: Path,
) -> None:
    pack_root = reviewed_release_root / "src" / "hlsgraph" / "knowledge" / "packs"
    before = {path.name: path.read_bytes() for path in pack_root.glob("*.json")}
    raw = _raw_evidence_path(reviewed_release_root, "semantic.raw.jsonl")
    raw.write_bytes(raw.read_bytes() + b"\n")
    with pytest.raises(ValueError):
        run_knowledge_review.seal_review_attestations(
            reviewed_release_root,
            semantic_raw=raw,
            adversarial_raw=_raw_evidence_path(
                reviewed_release_root, "adversarial.raw.jsonl",
            ),
            semantic_cache=reviewed_release_root.parent / "semantic.cache",
            adversarial_cache=reviewed_release_root.parent / "adversarial.cache",
        )
    assert {path.name: path.read_bytes() for path in pack_root.glob("*.json")} == before


def test_sealer_rejects_hardlinked_raw_evidence(
    reviewed_release_root: Path,
) -> None:
    semantic = _raw_evidence_path(
        reviewed_release_root, "semantic.raw.jsonl",
    )
    alias = semantic.parent / "semantic.alias.jsonl"
    os.link(semantic, alias)
    with pytest.raises(ValueError, match="hard-link aliases"):
        run_knowledge_review.seal_review_attestations(
            reviewed_release_root,
            semantic_raw=semantic,
            adversarial_raw=_raw_evidence_path(
                reviewed_release_root, "adversarial.raw.jsonl",
            ),
            semantic_cache=reviewed_release_root.parent / "semantic.cache",
            adversarial_cache=reviewed_release_root.parent / "adversarial.cache",
        )


def test_staged_publish_rolls_back_on_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "public"
    (root / "docs").mkdir(parents=True)
    original_replace = os.replace
    replacements = 0

    def fail_second(source: str | os.PathLike[str], target: str | os.PathLike[str]) -> None:
        nonlocal replacements
        replacements += 1
        if replacements == 2:
            raise OSError("injected publish failure")
        original_replace(source, target)

    monkeypatch.setattr(run_knowledge_review.os, "replace", fail_second)
    with pytest.raises(OSError, match="injected publish failure"):
        run_knowledge_review._publish_artifacts(root, {
            "docs/one.json": b"one", "docs/two.json": b"two",
            "docs/three.json": b"three",
        })
    assert not any((root / "docs" / name).exists() for name in (
        "one.json", "two.json", "three.json",
    ))


def test_result_schema_protocol_is_closed_enum() -> None:
    schema = json.loads((ROOT / REVIEW_SCHEMA_PATH).read_text(encoding="utf-8"))
    assert set(schema["properties"]["protocol_id"]["enum"]) == {
        SEMANTIC_REVIEW_PROTOCOL, ADVERSARIAL_REVIEW_PROTOCOL,
    }
    assert set(schema["properties"]["protocol_id"]) == {"enum"}


def test_utf8_chunk_contract_reconstructs_large_unicode_without_splitting() -> None:
    payload = (("数据流-PIPELINE-🙂\n" * 900) + "尾").encode("utf-8")
    assert len(payload) > 10_000
    chunks = run_knowledge_review._utf8_chunks(payload)
    assert b"".join(chunk for _start, _end, chunk in chunks) == payload
    assert all(len(chunk) <= run_knowledge_review.MAX_REVIEW_CHUNK_BYTES for _, _, chunk in chunks)
    assert all(chunk.decode("utf-8") for _, _, chunk in chunks)
    assert [start for start, _end, _chunk in chunks] == [
        0, *[end for _start, end, _chunk in chunks[:-1]],
    ]


def test_integrity_only_sources_have_no_model_visible_chunks(
    reviewed_release_root: Path,
) -> None:
    _snapshot, cache = _snapshot_cache(reviewed_release_root)
    integrity_only = [
        item for item in cache.manifest["files"]
        if item["model_inspection_required"] is False
    ]
    assert integrity_only
    assert all(item["chunks"] == [] for item in integrity_only)
    assert cache.manifest["inspection_contract"]["integrity_bound_only"] == sorted(
        item["path"] for item in integrity_only
    )


def test_model_inspection_scope_covers_activation_tcb() -> None:
    required_activation_tcb = {
        "src/hlsgraph/bundle.py",
        "src/hlsgraph/graph.py",
        "src/hlsgraph/manifest.py",
        "src/hlsgraph/knowledge/activation.py",
        "src/hlsgraph/knowledge/core.py",
        "src/hlsgraph/retrieval.py",
        "src/hlsgraph/runner/core.py",
        "src/hlsgraph/runner/staging.py",
        "src/hlsgraph/store/migrations.py",
        "src/hlsgraph/store/sqlite.py",
        "src/hlsgraph/extract/base.py",
        "src/hlsgraph/extract/directives.py",
        "src/hlsgraph/extract/llvm.py",
        "src/hlsgraph/extract/mlir.py",
        "src/hlsgraph/extract/source.py",
        "src/hlsgraph/extract/static_features.py",
    }
    assert required_activation_tcb <= run_knowledge_review.MODEL_INSPECTION_EXACT_PATHS


def test_approved_replay_rejects_one_skipped_required_chunk(
    reviewed_release_root: Path,
) -> None:
    snapshot, cache = _snapshot_cache(reviewed_release_root)
    raw_path = _raw_evidence_path(reviewed_release_root, "semantic.raw.jsonl")
    rows = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()]
    required = next(
        item for item in cache.manifest["files"]
        if item["model_inspection_required"] and len(item["chunks"]) > 1
    )
    command = f"head -n 100000000 {required['chunks'][1]['path']}"
    rows = [
        row for row in rows
        if not (isinstance(row.get("item"), dict)
                and row["item"].get("command") == command)
    ]
    with pytest.raises(ValueError, match="uninspected evidence"):
        run_knowledge_review.replay_raw_review(
            reviewed_release_root, SEMANTIC_REVIEW_PROTOCOL,
            run_knowledge_review._canonical_jsonl(rows),
            snapshot=snapshot, cache=cache,
        )


def test_public_result_schema_contains_no_model_authored_prose_fields() -> None:
    schema = json.loads((ROOT / REVIEW_SCHEMA_PATH).read_text(encoding="utf-8"))
    encoded = json.dumps(schema, sort_keys=True)
    for forbidden in ('"finding"', '"evidence"', '"required_fix"'):
        assert forbidden not in encoded
    assert schema["properties"]["summary"] == {
        "enum": ["approved_no_issues", "rejected_with_controlled_issues"],
    }


def test_prompt_visibility_budget_fails_closed(
    reviewed_release_root: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot, cache = _snapshot_cache(reviewed_release_root)
    monkeypatch.setattr(run_knowledge_review, "MAX_INITIAL_PROMPT_BYTES", 128)
    with pytest.raises(RuntimeError, match="visibility budget"):
        run_knowledge_review.build_review_prompt(
            reviewed_release_root, SEMANTIC_REVIEW_PROTOCOL,
            snapshot=snapshot, cache=cache,
        )


def test_pdftotext_contract_rejects_nonabsolute_or_unhashed_binary(
    tmp_path: Path,
) -> None:
    snapshot = run_knowledge_review.freeze_review_snapshot(
        ROOT, SEMANTIC_REVIEW_PROTOCOL,
    )
    with pytest.raises(ValueError, match="/usr/bin/pdftotext"):
        run_knowledge_review.create_review_cache(
            ROOT, snapshot, tmp_path / "bad-pdf.cache",
            fetcher=_trusted_fetch, pdftotext_command="pdftotext",
            pdftotext_sha256="a" * 64,
        )


def test_bounded_parser_output_kills_compressed_bomb_style_stream() -> None:
    script = (
        "import os\n"
        "while True:\n"
        " os.write(1, b'PRIVATE-PDF-BODY-' * 4096)\n"
    )
    with pytest.raises(
        ValueError, match=r"^controlled parser output exceeded its fixed byte limit$",
    ) as caught:
        run_knowledge_review._bounded_process_output(
            [sys.executable, "-c", script], env=dict(os.environ), timeout=10,
            stdout_limit=1024, stderr_limit=1024,
        )
    assert "PRIVATE-PDF-BODY" not in str(caught.value)


@pytest.mark.parametrize("bad_title", ["line one\nline two", "x" * 257])
def test_evidence_identity_strings_are_bounded_and_control_free(
    bad_title: str,
) -> None:
    value = json.loads(
        (ROOT / run_knowledge_review.CITATION_EVIDENCE_PATH).read_text(
            encoding="utf-8",
        )
    )
    amd = next(
        row for row in value["entries"]
        if str(row["resolver_id"]).startswith("amd.docs.khub.")
    )
    amd["identity"]["title"] = bad_title
    citation = json.loads(
        (ROOT / run_knowledge_review.CITATION_AUDIT_PATH).read_text(
            encoding="utf-8",
        )
    )
    with pytest.raises(ValueError, match="identity does not close"):
        run_knowledge_review._validate_citation_evidence_mapping(
            value,
            citation_audit_sha256=value["citation_audit_sha256"],
            exact_urls=[row["citation_url"] for row in value["entries"]],
            expected_references=citation["references"],
            expected_fetches=citation["fetches"],
        )


@pytest.mark.skipif(os.name == "nt", reason="POSIX hard-link count contract")
def test_cache_load_rejects_hard_linked_input(
    reviewed_release_root: Path, tmp_path: Path,
) -> None:
    snapshot, cache = _snapshot_cache(reviewed_release_root)
    alias = tmp_path / "manifest-alias.json"
    os.link(cache.root / run_knowledge_review.CACHE_MANIFEST_NAME, alias)
    with pytest.raises(ValueError, match="aliases"):
        run_knowledge_review.load_review_cache(cache.root, snapshot)


def test_review_runner_refuses_formal_windows_execution(
    reviewed_release_root: Path,
) -> None:
    if os.name != "nt":
        pytest.skip("Windows-only NO-GO assertion")
    with pytest.raises(RuntimeError, match="Windows is NO-GO"):
        run_knowledge_review._official_boundary(reviewed_release_root)


def test_release_auditor_refuses_formal_windows_execution(tmp_path: Path) -> None:
    if os.name != "nt":
        pytest.skip("Windows-only NO-GO assertion")
    issues = audit_release._audit_knowledge_review_release_gate(
        tmp_path, semantic_review=tmp_path / "semantic.json",
        adversarial_review=tmp_path / "adversarial.json",
        semantic_raw=tmp_path / "semantic.raw",
        adversarial_raw=tmp_path / "adversarial.raw",
        semantic_cache=tmp_path / "semantic.cache",
        adversarial_cache=tmp_path / "adversarial.cache",
    )
    assert issues == [
        "formal knowledge-review release audit is Linux/WSL2-only; Windows is NO-GO"
    ]
