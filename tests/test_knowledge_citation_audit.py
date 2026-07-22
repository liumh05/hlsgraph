from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from http.client import IncompleteRead
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = ROOT / "tools" / "audit_knowledge_citations.py"
SPEC = importlib.util.spec_from_file_location("hlsgraph_knowledge_citation_audit", AUDIT_PATH)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_builtin_citation_audit_is_offline_deterministic_and_fail_closed() -> None:
    first = AUDIT.audit_builtin_citations()
    second = AUDIT.audit_builtin_citations()

    assert first == second
    assert first["schema_version"] == "hlsgraph.knowledge-citation-audit.v2"
    assert first["passed"] is True
    assert first["mode"] == "offline"
    assert first["fetches"] == []
    assert first["summary"]["offline_failures"] == 0
    assert first["summary"]["document_references"] > 0
    assert first["summary"]["rule_references"] > 0
    assert first["summary"]["document_evidence_records"] == len(
        first["document_evidence"]
    )
    assert first["policy"]["response_bodies_stored"] is False
    assert first["document_evidence"] == sorted(
        first["document_evidence"], key=lambda item: item["document_key"],
    )
    assert len({item["document_key"] for item in first["document_evidence"]}) == len(
        first["document_evidence"]
    )
    assert all(
        item["hash_method"] == "canonical-json-sha256"
        and item["hash_method_version"]
        == "hlsgraph.document-citation-evidence.offline-metadata.v2"
        and item["body_stored"] is False
        and item["evidence_sha256_is_document_body_hash"] is False
        and len(item["evidence_sha256"]) == 64
        for item in first["document_evidence"]
    )
    assert first["document_evidence_policy"]["query_parameters_in_hash"] is False
    assert all(item["document_id"] and item["document_version"]
               for item in first["references"])
    assert all(item["section"] for item in first["references"]
               if item["reference_kind"] == "rule")
    assert all(item["citation_url"].startswith("https://")
               for item in first["references"])
    assert first["generator"] == {
        "path": "tools/audit_knowledge_citations.py",
        "sha256": _digest(AUDIT_PATH.read_bytes()),
    }
    assert first["policy"]["locator_policy_id"] == (
        "hlsgraph.citation-locator-policy.v1"
    )
    assert first["summary"]["pack_count"] == len(first["packs"])
    assert first["summary"]["reference_count"] == len(first["references"])
    assert all(
        "content_hash" not in item
        and len(item["review_surface_sha256"]) == 64
        and item["path"].startswith("src/hlsgraph/knowledge/packs/")
        for item in first["packs"]
    )
    assert len({item["reference_id"] for item in first["references"]}) == len(
        first["references"]
    )
    for item in first["references"]:
        assert "pack_content_hash" not in item
        assert len(item["document_surface_sha256"]) == 64
        assert len(item["pack_review_surface_sha256"]) == 64
        assert len(item["reference_id"]) == 64
        assert len(item["reference_surface_sha256"]) == 64
        if item["reference_kind"] == "rule":
            assert len(item["rule_surface_sha256"]) == 64
            assert item["rule_id"].startswith(
                f'{item["document_id"]}:{item["document_version"]}:'
            )
        else:
            assert item["rule_surface_sha256"] is None
            assert item["rule_id"] is None
        surface = dict(item)
        digest = surface.pop("reference_surface_sha256")
        assert digest == AUDIT._typed_surface_sha256(
            AUDIT.REFERENCE_SURFACE_VERSION, surface,
        )
        identity = {
            key: item[key]
            for key in (
                "citation_url", "document_id", "document_version", "pack_id",
                "reference_kind", "rule_id", "section",
            )
        }
        assert item["reference_id"] == AUDIT._typed_surface_sha256(
            AUDIT.REFERENCE_ID_VERSION, identity,
        )
    unhashed = dict(first)
    manifest_sha256 = unhashed.pop("manifest_sha256")
    assert manifest_sha256 == AUDIT._manifest_hash(unhashed)
    assert json.loads(AUDIT.dump_manifest(first))["manifest_sha256"] == first["manifest_sha256"]


def test_v2_reference_surfaces_bind_exact_documents_rules_and_review_surfaces() -> None:
    manifest = AUDIT.audit_builtin_citations()
    catalog = {
        pack.pack_id: pack for pack in AUDIT.KnowledgeCatalog.builtin().packs
    }
    pack_rows = {item["pack_id"]: item for item in manifest["packs"]}

    for pack_id, pack in catalog.items():
        pack_row = pack_rows[pack_id]
        pack_path = ROOT / pack_row["path"]
        assert pack_row["review_surface_sha256"] == (
            AUDIT._knowledge_review_surface.surface_sha256(pack_path)
        )
        documents = {
            (item.document_id, item.document_version): item
            for item in pack.documents
        }
        rules = {item.id: item for item in pack.rules}
        rows = [item for item in manifest["references"] if item["pack_id"] == pack_id]
        assert len(rows) == len(pack.documents) + len(pack.rules)
        for row in rows:
            document = documents[(row["document_id"], row["document_version"])]
            assert row["document_surface_sha256"] == AUDIT._typed_surface_sha256(
                AUDIT.DOCUMENT_SURFACE_VERSION, AUDIT.json_ready(document),
            )
            assert row["pack_review_surface_sha256"] == (
                pack_row["review_surface_sha256"]
            )
            if row["reference_kind"] == "rule":
                rule = rules[row["rule_id"]]
                assert row["rule_surface_sha256"] == AUDIT._typed_surface_sha256(
                    AUDIT.RULE_SURFACE_VERSION, AUDIT.json_ready(rule),
                )


def test_reference_identity_and_surface_have_separate_tamper_domains() -> None:
    kwargs = {
        "pack_id": "hlsgraph.test.pack",
        "review_surface_sha256": "a" * 64,
        "reference_kind": "rule",
        "document_id": "test.document",
        "document_version": "1",
        "document_surface_sha256": "b" * 64,
        "section": "Section One",
        "rule_id": "test.document:1:test.rule",
        "rule_surface_sha256": "c" * 64,
        "url": "https://llvm.org/docs/LangRef.html#functions",
    }
    baseline = AUDIT._reference_record(**kwargs)
    changed_rule = AUDIT._reference_record(
        **{**kwargs, "rule_surface_sha256": "d" * 64},
    )
    changed_locator = AUDIT._reference_record(
        **{**kwargs, "url": "https://llvm.org/docs/LangRef.html#blocks"},
    )

    assert changed_rule["reference_id"] == baseline["reference_id"]
    assert changed_rule["reference_surface_sha256"] != (
        baseline["reference_surface_sha256"]
    )
    assert changed_locator["reference_id"] != baseline["reference_id"]
    assert changed_locator["reference_surface_sha256"] != (
        baseline["reference_surface_sha256"]
    )


def test_offline_policy_rejects_unofficial_unpinned_and_mismatched_locators() -> None:
    _, unofficial = AUDIT._locator_policy_issues(
        document_id="amd.ug1399", document_version="2024.2",
        url="https://example.invalid/ug1399", reference_kind="rule",
    )
    assert "host_not_allowlisted" in unofficial
    assert "document_publisher_host_mismatch" in unofficial

    _, unpinned = AUDIT._locator_policy_issues(
        document_id="llvm.ir.langref", document_version="git-" + "a" * 40,
        url="https://github.com/llvm/llvm-project/blob/main/llvm/docs/LangRef.md",
        reference_kind="rule",
    )
    assert "github_revision_is_not_full_commit" in unpinned
    assert "github_revision_document_version_mismatch" in unpinned

    _, wrong_amd_version = AUDIT._locator_policy_issues(
        document_id="amd.ug1399", document_version="2024.2",
        url="https://docs.amd.com/r/2024.1-English/ug1399-vitis-hls/pragma-HLS-pipeline",
        reference_kind="rule",
    )
    assert "amd_document_version_not_bound_in_url" in wrong_amd_version


def test_mocked_online_audit_deduplicates_gets_and_never_stores_bodies() -> None:
    calls: list[tuple[str, float, int]] = []

    def fetch(url: str, timeout: float, max_bytes: int):
        calls.append((url, timeout, max_bytes))
        if "documentation-service.arm.com/static/" in url:
            payload = b"%PDF-fixture"
            return AUDIT.OnlineFetch(
                200, "application/pdf", len(payload), _digest(payload), url,
                pdf_magic=True,
            )
        payload = b"public-locator-fixture"
        return AUDIT.OnlineFetch(
            200, "text/html", len(payload), _digest(payload), url,
        )

    manifest = AUDIT.audit_builtin_citations(
        online=True, timeout_seconds=3.5, max_bytes=4096, fetcher=fetch,
    )
    assert manifest["passed"] is True
    assert len(calls) == len({item[0] for item in calls})
    assert len(calls) == manifest["summary"]["unique_fetch_urls"]
    assert all(item[1:] == (3.5, 4096) for item in calls)
    assert all(item["body_stored"] is False for item in manifest["fetches"])
    assert all(item["attempt_count"] == 1 for item in manifest["fetches"])
    assert all("body" not in item and "content" not in item
               for item in manifest["fetches"])
    assert all(item["verification_level"] == "reachable_locator_only"
               for item in manifest["fetches"]
               if item["locator_kind"] == "amd_fluidtopics_locator")
    assert all(item["verification_level"] == "document_bytes_verified"
               for item in manifest["fetches"]
               if item["locator_kind"] == "arm_static_pdf")
    assert all(
        item["hash_method_version"]
        == "hlsgraph.document-citation-evidence.online-fetch-metadata.v2"
        and item["body_stored"] is False
        and item["evidence_sha256_is_document_body_hash"] is False
        for item in manifest["document_evidence"]
    )
    amd_evidence = [
        item for item in manifest["document_evidence"]
        if item["document_id"].startswith("amd.")
    ]
    assert amd_evidence
    assert all(item["evidence_scope"] == "reachable_locator_only"
               for item in amd_evidence)


def test_mocked_online_audit_requires_arm_pdf_and_rejects_unsafe_redirect() -> None:
    def fetch(url: str, timeout: float, max_bytes: int):
        del timeout, max_bytes
        if "documentation-service.arm.com/static/" in url:
            payload = b"not a pdf"
            return AUDIT.OnlineFetch(
                200, "text/html", len(payload), _digest(payload), url,
                pdf_magic=False,
            )
        if "arxiv.org/" in url:
            payload = b"ok"
            return AUDIT.OnlineFetch(
                200, "text/html", len(payload), _digest(payload),
                "https://example.invalid/redirected",
            )
        payload = b"ok"
        return AUDIT.OnlineFetch(
            200, "text/html", len(payload), _digest(payload), url,
        )

    manifest = AUDIT.audit_builtin_citations(online=True, fetcher=fetch)
    assert manifest["passed"] is False
    arm = [item for item in manifest["fetches"]
           if item["locator_kind"] == "arm_static_pdf"]
    assert arm
    assert all("arm_static_document_not_pdf_content_type" in item["issues"] for item in arm)
    assert all("arm_static_document_not_pdf_bytes" in item["issues"] for item in arm)
    arxiv = [item for item in manifest["fetches"]
             if item["locator_kind"] == "arxiv_official_record"]
    assert any("unsafe_or_unofficial_redirect_target" in item["issues"] for item in arxiv)


def test_cli_writes_only_requested_manifest(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "citation-audit.json"
    assert AUDIT.main(["--output", str(output)]) == 0
    value = json.loads(output.read_text(encoding="utf-8"))
    assert value["mode"] == "offline"
    assert value["passed"] is True
    assert sorted(tmp_path.iterdir()) == [output]


def test_urllib_fetch_turns_incomplete_chunked_response_into_metadata(monkeypatch) -> None:
    class Response:
        status = 200
        headers = {"Content-Type": "text/html"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def geturl(self):
            return "https://docs.amd.com/r/2024.2-English/ug1399-vitis-hls/"

        def read(self, amount: int):
            del amount
            raise IncompleteRead(b"partial", 5)

    monkeypatch.setattr(AUDIT, "urlopen", lambda *args, **kwargs: Response())
    result = AUDIT._urllib_fetch(
        "https://docs.amd.com/r/2024.2-English/ug1399-vitis-hls/", 1.0, 4096,
    )
    assert result.error_code == "protocol_error"
    assert result.sha256 is None
    assert result.byte_count == 0


def test_default_fetch_uses_stdlib_when_curl_is_unavailable(monkeypatch) -> None:
    expected = AUDIT.OnlineFetch(
        200, "text/html", 2, _digest(b"ok"), "https://llvm.org/docs/LangRef.html",
    )
    monkeypatch.setattr(AUDIT.shutil, "which", lambda executable: None)
    monkeypatch.setattr(AUDIT, "_urllib_fetch", lambda *args: expected)
    assert AUDIT._default_fetch("https://llvm.org/docs/LangRef.html", 1.0, 1024) == expected


def test_default_fetch_retries_transient_curl_failure_with_stdlib(monkeypatch) -> None:
    url = "https://llvm.org/docs/LangRef.html"
    partial = AUDIT.OnlineFetch(
        200, "text/html", 7, None, url, error_code="network_error",
    )
    complete = AUDIT.OnlineFetch(
        200, "text/html", 2, _digest(b"ok"), url,
    )
    calls: list[str] = []
    monkeypatch.setattr(AUDIT.shutil, "which", lambda executable: "curl")
    monkeypatch.setattr(
        AUDIT, "_curl_fetch",
        lambda *args: calls.append("curl") or partial,
    )
    monkeypatch.setattr(
        AUDIT, "_urllib_fetch",
        lambda *args: calls.append("urllib") or complete,
    )

    assert AUDIT._default_fetch(url, 1.0, 1024) == complete
    assert calls == ["curl", "urllib"]


def test_online_audit_retries_only_bounded_transient_failures() -> None:
    counts: dict[str, int] = {}

    def fetch(url: str, timeout: float, max_bytes: int):
        del timeout, max_bytes
        counts[url] = counts.get(url, 0) + 1
        if counts[url] == 1:
            return AUDIT.OnlineFetch(
                None, None, 0, None, url, error_code="network_error",
            )
        payload = b"%PDF-ok" if "documentation-service.arm.com" in url else b"ok"
        return AUDIT.OnlineFetch(
            200,
            "application/pdf" if payload.startswith(b"%PDF-") else "text/html",
            len(payload), _digest(payload), url,
            pdf_magic=payload.startswith(b"%PDF-"),
        )

    manifest = AUDIT.audit_builtin_citations(
        online=True, max_attempts=2, fetcher=fetch,
    )
    assert manifest["passed"] is True
    assert set(counts.values()) == {2}
    assert all(item["attempt_count"] == 2 for item in manifest["fetches"])


def test_document_evidence_hash_is_stable_and_ignores_query_and_body_fields() -> None:
    documents = [{
        "document_id": "amd.test",
        "document_version": "1",
        "title": "Public title",
        "official_url": "https://docs.amd.com/r/1-English/test/?secret=one",
        "publisher": "AMD",
        "kind": "guide",
        "license_note": "metadata only",
        "body": "must never enter evidence",
    }]
    references = [
        {
            "reference_kind": "document", "document_id": "amd.test",
            "document_version": "1", "citation_url": documents[0]["official_url"],
            "fetch_url": documents[0]["official_url"], "host": "docs.amd.com",
            "issues": [], "locator_kind": "amd_fluidtopics_locator",
            "offline_status": "pass", "rule_id": None, "section": None,
            "body": "ignored document body",
        },
        {
            "reference_kind": "rule", "document_id": "amd.test",
            "document_version": "1",
            "citation_url": "https://docs.amd.com/r/1-English/test/section?secret=one#part",
            "fetch_url": "https://docs.amd.com/r/1-English/test/section?secret=one",
            "host": "docs.amd.com", "issues": [],
            "locator_kind": "amd_fluidtopics_locator", "offline_status": "pass",
            "rule_id": "test.rule", "section": "Section", "body": "ignored rule body",
        },
    ]
    fetches = [{
        "fetch_url": references[1]["fetch_url"],
        "final_url": references[1]["fetch_url"],
        "status": 200, "content_type": "text/html", "byte_count": 10,
        "sha256": "a" * 64, "pdf_magic": False, "error_code": None,
        "issues": [], "locator_kind": "amd_fluidtopics_locator",
        "verification_level": "reachable_locator_only",
        "attempt_count": 2, "body": "ignored response body",
    }]
    first = AUDIT._document_evidence_records(
        document_metadata=documents, references=references,
        fetches=fetches, online=True,
    )
    reordered = AUDIT._document_evidence_records(
        document_metadata=list(reversed(documents + documents)),
        references=list(reversed(references + references)),
        fetches=list(reversed(fetches)), online=True,
    )
    assert first == reordered
    assert first[0]["evidence_scope"] == "reachable_locator_only"
    assert "secret" not in json.dumps(first, sort_keys=True)
    assert "ignored" not in json.dumps(first, sort_keys=True)

    query_and_body_changed = json.loads(json.dumps({
        "documents": documents, "references": references, "fetches": fetches,
    }))
    query_and_body_changed["documents"][0]["official_url"] = (
        "https://docs.amd.com/r/1-English/test/?secret=two"
    )
    query_and_body_changed["documents"][0]["body"] = "different"
    for item in query_and_body_changed["references"]:
        item["citation_url"] = item["citation_url"].replace("secret=one", "secret=two")
        item["fetch_url"] = item["fetch_url"].replace("secret=one", "secret=two")
        item["body"] = "different"
    query_and_body_changed["fetches"][0]["fetch_url"] = (
        query_and_body_changed["fetches"][0]["fetch_url"].replace(
            "secret=one", "secret=two",
        )
    )
    query_and_body_changed["fetches"][0]["final_url"] = (
        query_and_body_changed["fetches"][0]["final_url"].replace(
            "secret=one", "secret=two",
        )
    )
    query_and_body_changed["fetches"][0]["body"] = "different"
    assert AUDIT._document_evidence_records(
        document_metadata=query_and_body_changed["documents"],
        references=query_and_body_changed["references"],
        fetches=query_and_body_changed["fetches"], online=True,
    ) == first


def test_document_evidence_hash_changes_for_public_locator_or_fetch_evidence() -> None:
    document = {
        "document_id": "arm.test", "document_version": "B",
        "title": "Title", "official_url": "https://documentation-service.arm.com/static/abc123456789def0",
        "publisher": "Arm", "kind": "standard", "license_note": "metadata only",
    }
    reference = {
        "reference_kind": "rule", "document_id": "arm.test",
        "document_version": "B", "citation_url": document["official_url"],
        "fetch_url": document["official_url"],
        "host": "documentation-service.arm.com", "issues": [],
        "locator_kind": "arm_static_pdf", "offline_status": "pass",
        "rule_id": "test.rule", "section": "2.2",
    }
    fetch = {
        "fetch_url": document["official_url"], "final_url": document["official_url"],
        "status": 200, "content_type": "application/pdf", "byte_count": 10,
        "sha256": "a" * 64, "pdf_magic": True, "error_code": None,
        "issues": [], "locator_kind": "arm_static_pdf",
        "verification_level": "document_bytes_verified",
    }

    def evidence(*, section: str = "2.2", digest: str = "a" * 64, online=True):
        selected_reference = {**reference, "section": section}
        selected_fetch = {**fetch, "sha256": digest}
        return AUDIT._document_evidence_records(
            document_metadata=[document], references=[selected_reference],
            fetches=[selected_fetch], online=online,
        )[0]

    baseline = evidence()
    assert evidence(section="2.3")["evidence_sha256"] != baseline["evidence_sha256"]
    assert evidence(digest="b" * 64)["evidence_sha256"] != baseline["evidence_sha256"]
    offline = evidence(online=False)
    assert offline["hash_method_version"].endswith("offline-metadata.v2")
    assert baseline["hash_method_version"].endswith("online-fetch-metadata.v2")
    assert offline["evidence_sha256"] != baseline["evidence_sha256"]
