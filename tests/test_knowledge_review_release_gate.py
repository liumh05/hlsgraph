from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from tools import knowledge_review_surface
from tools import audit_knowledge_citations
from tools import run_knowledge_review
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
    _audit_knowledge_review_release_gate,
)


ROOT = Path(__file__).parents[1]
PACKS = {
    "amd_public_guidance_2024_2.json": "hlsgraph.amd.public_guidance.2024_2",
    "axi_public_guidance.json": "hlsgraph.axi.public_guidance.v1",
    "open_ir_public_guidance.json": (
        "hlsgraph.open_ir.public_guidance.2026_07_21"
    ),
}


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
    locator_hash = hashlib.sha256(url.encode("utf-8")).hexdigest().encode("ascii")
    if "documentation-service.arm.com/static/" in url:
        body = b"%PDF-1.7\nHLSGRAPH-CACHE-PDF-" + locator_hash
        content_type = "application/pdf"
    else:
        body = b"HLSGRAPH-CACHE-TEXT-" + locator_hash
        content_type = "text/plain"
    return run_knowledge_review.TrustedFetch(
        status=200, final_url=url, redirect_chain=(url,),
        content_type=content_type, body=body, charset="utf-8",
    )


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
            "exact_locator_inspected": True,
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
        "summary": summary,
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

    for item in snapshot.files:
        command(f"head -n 100000000 files/{item.path}")
    inspected_paths: set[str] = set()
    for entry in cache.manifest["citations"]:
        path = entry.get("inspection_path")
        if entry.get("available") is True and path not in inspected_paths:
            inspected_paths.add(str(path))
            command(f"head -n 100000000 {path}")
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


@pytest.fixture
def reviewed_release_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "public"
    pack_root = root / "src" / "hlsgraph" / "knowledge" / "packs"
    shutil.copytree(ROOT / "src" / "hlsgraph", root / "src" / "hlsgraph")
    for relative in (
        REVIEW_SCHEMA_PATH,
        REVIEW_RECEIPT_SCHEMA_PATH,
        SEMANTIC_REVIEW_PROMPT_PATH,
        ADVERSARIAL_REVIEW_PROMPT_PATH,
        "tools/knowledge_review_surface.py",
        "tools/audit_knowledge_citations.py",
        "tools/run_knowledge_review.py",
        "tools/audit_release.py",
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

    def citation_fetch(url: str, _timeout: float, _max_bytes: int):
        payload = (
            b"%PDF-review-fixture" if "documentation-service.arm.com/static/" in url
            else b"public-review-locator-fixture"
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
    _write_json(root / CITATION_AUDIT_PATH, citation)

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
            fetcher=_trusted_fetch, pdf_text_extractor=_fixture_pdf_text,
        )
        cache = run_knowledge_review.load_review_cache(cache.root, snapshot)
        result_value = _approved_result(snapshot, summary=summary)
        raw_bytes = _raw_stream(
            snapshot=snapshot, cache=cache, result=result_value,
            thread_id=thread_id,
        )
        raw_path = root.parent / raw_name
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
        )
        (root / files["receipt"]).write_bytes(
            run_knowledge_review._canonical_json(receipt)
        )
        invocations.append((protocol_id, snapshot, cache, replay))

    run_knowledge_review.seal_review_attestations(
        root,
        semantic_raw=root.parent / "semantic.raw.jsonl",
        adversarial_raw=root.parent / "adversarial.raw.jsonl",
        semantic_cache=root.parent / "semantic.cache",
        adversarial_cache=root.parent / "adversarial.cache",
    )
    return root
def _audit(root: Path) -> list[str]:
    return _audit_knowledge_review_release_gate(
        root,
        semantic_review=root / SEMANTIC_REVIEW_PATH,
        adversarial_review=root / ADVERSARIAL_REVIEW_PATH,
        semantic_raw=root.parent / "semantic.raw.jsonl",
        adversarial_raw=root.parent / "adversarial.raw.jsonl",
        semantic_cache=root.parent / "semantic.cache",
        adversarial_cache=root.parent / "adversarial.cache",
    )


def test_final_review_gate_accepts_exact_repeated_review_bytes(
    reviewed_release_root: Path,
) -> None:
    assert _audit(reviewed_release_root) == []


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
    value["summary"] += " Changed after approval."
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
    assert any("manifest_sha256 is inconsistent" in item for item in issues)
    assert any("references differ from the current pack inventory" in item for item in issues)


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
    with pytest.raises(RuntimeError, match="outside the review cache"):
        run_knowledge_review.run_review(
            ROOT, run_knowledge_review.SEMANTIC_PROTOCOL, raw, cache,
            codex_command="codex", timeout_seconds=1,
        )


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
    with pytest.raises(RuntimeError, match="linked path component"):
        run_knowledge_review.run_review(
            ROOT, run_knowledge_review.SEMANTIC_PROTOCOL, raw, cache,
            codex_command="codex", timeout_seconds=1,
        )


def test_review_snapshot_binds_release_auditor() -> None:
    snapshot = run_knowledge_review.freeze_review_snapshot(
        ROOT, run_knowledge_review.SEMANTIC_PROTOCOL,
    )
    assert run_knowledge_review.RELEASE_AUDITOR_PATH in snapshot.file_map
    assert snapshot.file_map[
        run_knowledge_review.RELEASE_AUDITOR_PATH
    ].sha256 == hashlib.sha256(
        (ROOT / run_knowledge_review.RELEASE_AUDITOR_PATH).read_bytes()
    ).hexdigest()


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
    fetch = next(item for item in rows if item["kind"] == "citation_inspect")
    fetch["requested_url"] = "https://example.invalid/search?q=substitute"
    _write_jsonl(path, rows)
    issues = _audit(reviewed_release_root)
    assert any("unapproved locator" in item for item in issues)
    assert any("does not inspect every exact locator" in item for item in issues)


def test_final_review_gate_replays_and_rejects_unknown_raw_tool(
    reviewed_release_root: Path,
) -> None:
    path = reviewed_release_root.parent / "semantic.raw.jsonl"
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
    path = reviewed_release_root.parent / "adversarial.raw.jsonl"
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
    path = reviewed_release_root.parent / "semantic.raw.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    message = next(
        item for item in reversed(rows)
        if item.get("type") == "item.completed"
        and item.get("item", {}).get("type") == "agent_message"
    )
    value = json.loads(message["item"]["text"])
    value["approved"] = False
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


def test_cache_retains_full_valid_same_host_redirect_chain(
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
    assert all(len(entry["redirect_chain"]) == 2 for entry in cache.manifest["citations"])
    assert all(entry["redirect_chain"][-1] == entry["final_url"] for entry in cache.manifest["citations"])


def test_pdf_without_controlled_parser_is_unavailable_and_cannot_approve(
    reviewed_release_root: Path,
) -> None:
    snapshot = run_knowledge_review.freeze_review_snapshot(
        reviewed_release_root, SEMANTIC_REVIEW_PROTOCOL,
    )
    cache = run_knowledge_review.create_review_cache(
        reviewed_release_root, snapshot,
        reviewed_release_root.parent / "no-pdf-parser.cache",
        fetcher=_trusted_fetch,
    )
    cache = run_knowledge_review.load_review_cache(cache.root, snapshot)
    pdf_entries = [
        entry for entry in cache.manifest["citations"]
        if entry["content_type"] == "application/pdf"
    ]
    assert pdf_entries
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


def test_sanitized_raw_contains_no_cached_citation_body_or_text(
    reviewed_release_root: Path,
) -> None:
    _snapshot, cache = _snapshot_cache(reviewed_release_root)
    raw = (reviewed_release_root.parent / "semantic.raw.jsonl").read_bytes()
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
    run_knowledge_review._write_private(cache.root / "extra.txt", b"extra")
    with pytest.raises(ValueError, match="unmanifested"):
        run_knowledge_review.load_review_cache(cache.root, snapshot)


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits are required")
def test_cache_tree_uses_0700_directories_and_0600_files(
    reviewed_release_root: Path,
) -> None:
    _snapshot, cache = _snapshot_cache(reviewed_release_root)
    for current, _directories, files in os.walk(cache.root):
        assert Path(current).stat().st_mode & 0o777 == 0o700
        for name in files:
            assert (Path(current) / name).stat().st_mode & 0o777 == 0o600
    raw = reviewed_release_root.parent / "semantic.raw.jsonl"
    assert raw.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("mutation", ["empty", "missing-start", "missing-status", "bad-output"])
def test_raw_replay_rejects_incomplete_lifecycle_and_output(
    reviewed_release_root: Path, mutation: str,
) -> None:
    snapshot, cache = _snapshot_cache(reviewed_release_root)
    raw_path = reviewed_release_root.parent / "semantic.raw.jsonl"
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
        semantic_raw=reviewed_release_root.parent / "semantic.raw.jsonl",
        adversarial_raw=reviewed_release_root.parent / "adversarial.raw.jsonl",
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
    raw = reviewed_release_root.parent / "semantic.raw.jsonl"
    raw.write_bytes(raw.read_bytes() + b"\n")
    with pytest.raises(ValueError):
        run_knowledge_review.seal_review_attestations(
            reviewed_release_root,
            semantic_raw=raw,
            adversarial_raw=reviewed_release_root.parent / "adversarial.raw.jsonl",
            semantic_cache=reviewed_release_root.parent / "semantic.cache",
            adversarial_cache=reviewed_release_root.parent / "adversarial.cache",
        )
    assert {path.name: path.read_bytes() for path in pack_root.glob("*.json")} == before


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


def test_review_runner_refuses_formal_windows_execution(
    reviewed_release_root: Path,
) -> None:
    if os.name != "nt":
        pytest.skip("Windows-only NO-GO assertion")
    with pytest.raises(RuntimeError, match="Windows is NO-GO"):
        run_knowledge_review._official_boundary(reviewed_release_root)
