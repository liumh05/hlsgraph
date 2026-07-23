from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Callable

import pytest

from tools import execute_knowledge_review_suite as executor
from tools import knowledge_review_shards as shards
from tools import knowledge_review_suite_replay as replay
from tools import run_knowledge_review as review
from tools import run_knowledge_review_suite as suite


ROOT = Path(__file__).resolve().parents[1]


def _json(path: str) -> dict[str, object]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _digest(value: str | bytes) -> str:
    payload = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(payload).hexdigest()


def _write_chunk(
    cache_root: Path, *, origin: str, kind: str, payload: bytes,
) -> dict[str, object]:
    digest = _digest(payload)
    path = (
        f"chunks/{kind}/{_digest(origin)}/"
        f"000000-{digest}.utf8"
    )
    target = cache_root / Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    return {
        "index": 0,
        "path": path,
        "sha256": digest,
        "size": len(payload),
        "byte_start": 0,
        "byte_end": len(payload),
        "original_sha256": digest,
        "original_size": len(payload),
    }


def _fixture(
    tmp_path: Path, *,
    protocol_id: str = shards.SEMANTIC_PROTOCOL_ID,
    shard_id: str = "knowledge_activation",
) -> tuple[review.ReviewCache, dict[str, object], dict[str, object]]:
    audit = _json("docs/knowledge-citation-audit-v0.3.json")
    plan = shards.build_shard_plan(audit)
    plan_row = next(
        row for row in plan["shards"] if row["shard_id"] == shard_id
    )
    cache_root = tmp_path / "projected-cache"
    cache_root.mkdir()

    files: list[dict[str, object]] = []
    for path in sorted(plan_row["source_paths"]):
        payload = ("source:" + path + "\n").encode("utf-8")
        chunk = _write_chunk(
            cache_root, origin=path, kind="source", payload=payload,
        )
        digest = _digest(payload)
        files.append({
            "path": path,
            "hash_kind": "raw_sha256",
            "sha256": digest,
            "cache_sha256": digest,
            "cache_size": len(payload),
            "model_inspection_required": True,
            "chunks": [chunk],
        })

    references = plan_row["rule_references"]
    by_url: dict[str, list[dict[str, object]]] = {}
    for row in references:
        by_url.setdefault(str(row["citation_url"]), []).append(row)
    citations: list[dict[str, object]] = []
    for url in sorted(by_url):
        payload = ("citation section:" + url + "\n").encode("utf-8")
        chunk = _write_chunk(
            cache_root, origin=url, kind="citation", payload=payload,
        )
        body_sha256 = _digest("body:" + url)
        inspection_sha256 = _digest(payload)
        citations.append({
            "requested_url": url,
            "evidence_url": url,
            "final_url": url,
            "redirect_chain": [url],
            "resolver_id": "fixture.exact-section.v1",
            "status": 200,
            "content_type": "text/plain",
            "body_sha256": body_sha256,
            "body_size": 128,
            "inspection_required": True,
            "identity_verified": True,
            "available": True,
            "inspection_sha256": inspection_sha256,
            "inspection_size": len(payload),
            "parser_id": "fixture.parser.v1",
            "parser_version": "fixture/1",
            "parser_command_sha256": _digest("fixture parser command"),
            "parser_executable_sha256": None,
            "parser_version_output_sha256": None,
            "resolver_artifacts": [],
            "inspection_chunks": [chunk],
            "error_code": None,
            "reference_ids": sorted(
                str(row["reference_id"]) for row in by_url[url]
            ),
        })

    assertion_owners = shards.assertion_owners(protocol_id)
    assertion_ids = sorted(
        assertion_id for assertion_id, owner in assertion_owners.items()
        if owner == shard_id
    )
    evidence_hash = suite.citation_evidence_surface_sha256(citations)
    manifest: dict[str, object] = {
        "schema_version": suite.SHARD_MANIFEST_SCHEMA_VERSION,
        "protocol_id": protocol_id,
        "review_snapshot_sha256": "3" * 64,
        "shard_plan_sha256": shards.shard_plan_sha256(plan),
        "shard_id": shard_id,
        "citation_evidence_surface_sha256": evidence_hash,
        "full_citation_evidence_surface_sha256": evidence_hash,
        "source_paths": sorted(plan_row["source_paths"]),
        "assertion_ids": assertion_ids,
        "rule_references": copy.deepcopy(references),
        "files": files,
        "citations": citations,
        "chunk_contract": {
            "schema_version": "fixture.chunk.v1",
            "sha256": "2" * 64,
        },
        "token_budget_contract": (
            shards.DEFAULT_TOKEN_BUDGET_CONTRACT.to_dict()
        ),
    }
    cache = review.ReviewCache(
        root=cache_root,
        manifest=copy.deepcopy(manifest),
        manifest_bytes=review._canonical_json(manifest),
    )
    review._harden_private_tree(cache_root)
    return cache, manifest, _approved_result(manifest)


def _approved_result(manifest: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": suite.SHARD_RESULT_SCHEMA_VERSION,
        "protocol_id": manifest["protocol_id"],
        "review_snapshot_sha256": manifest["review_snapshot_sha256"],
        "shard_plan_sha256": manifest["shard_plan_sha256"],
        "shard_id": manifest["shard_id"],
        "citation_evidence_surface_sha256": manifest[
            "citation_evidence_surface_sha256"
        ],
        "assertion_results": [{
            "assertion_id": assertion_id,
            "verdict": "verified",
            "issues": [],
        } for assertion_id in manifest["assertion_ids"]],
        "citation_results": [{
            "reference_id": row["reference_id"],
            "reference_surface_sha256": row["reference_surface_sha256"],
            "verdict": "verified",
            "exact_locator_inspected": True,
            "declared_version_matched": True,
            "declared_section_matched": True,
            "paraphrase_supported": True,
            "applicability_not_broader": True,
            "issues": [],
        } for row in manifest["rule_references"]],
        "approved": True,
        "issues": [],
        "summary": "approved_no_issues",
    }


def _read_commands(manifest: dict[str, object]) -> list[str]:
    paths = [
        chunk["path"]
        for row in manifest["files"]
        for chunk in row["chunks"]
    ]
    paths.extend(
        chunk["path"]
        for row in manifest["citations"]
        for chunk in row["inspection_chunks"]
    )
    return [
        review._codex_shell_event_command(f"head -n 100000000 {path}")
        for path in paths
    ]


def _raw_stream(
    cache: review.ReviewCache,
    manifest: dict[str, object],
    result: dict[str, object],
    *,
    commands: list[str] | None = None,
    thread_id: str = "thread-shard-0001",
) -> bytes:
    rows: list[dict[str, object]] = [
        {"type": "thread.started", "thread_id": thread_id},
        {"type": "turn.started"},
    ]
    for index, command in enumerate(
        commands if commands is not None else _read_commands(manifest), 1,
    ):
        call_id = f"command-{index:04d}"
        item = {
            "id": call_id,
            "type": "command_execution",
            "command": command,
        }
        try:
            inner_command = review._unwrap_codex_shell_event_command(command)
            output, _operations, _citation = review._expected_command(
                cache, inner_command,
            )
        except ValueError:
            output = ""
        rows.append({"type": "item.started", "item": item})
        rows.append({
            "type": "item.completed",
            "item": {
                **item,
                "status": "completed",
                "exit_code": 0,
                "aggregated_output": output,
            },
        })
    rows.extend([
        {
            "type": "item.completed",
            "item": {
                "id": "agent-message-final",
                "type": "agent_message",
                "text": json.dumps(result, sort_keys=True),
            },
        },
        {"type": "turn.completed", "usage": {
            "input_tokens": 1000,
            "cached_input_tokens": 250,
            "output_tokens": 100,
            "reasoning_output_tokens": 25,
        }},
    ])
    return review._canonical_jsonl(rows)


def _events(raw: bytes) -> list[dict[str, object]]:
    return review._strict_jsonl(raw, label="fixture raw stream")


def test_replay_accepts_exact_raw_stream_and_emits_shard_trace(
    tmp_path: Path,
) -> None:
    cache, manifest, result = _fixture(tmp_path)
    raw = _raw_stream(cache, manifest, result)

    actual = replay.replay_shard_raw_review(
        raw, cache=cache, shard_manifest=manifest,
    )
    trace = review._strict_jsonl(actual.trace_bytes, label="shard trace")

    expected_reads = len(_read_commands(manifest))
    assert actual.protocol_id == shards.SEMANTIC_PROTOCOL_ID
    assert actual.shard_id == "knowledge_activation"
    assert actual.thread_id == "thread-shard-0001"
    assert actual.raw_sha256 == hashlib.sha256(raw).hexdigest()
    assert actual.reported_input_tokens == 1000
    assert actual.reported_cached_input_tokens == 250
    assert actual.reported_output_tokens == 100
    assert actual.reported_reasoning_output_tokens == 25
    assert actual.derived_input_plus_output_tokens == 1100
    assert actual.result == result
    assert json.loads(actual.result_bytes) == result
    assert len(trace) == expected_reads + 1
    assert [row["shard_sequence"] for row in trace] == list(
        range(1, len(trace) + 1)
    )
    assert all(row["shard_id"] == "knowledge_activation" for row in trace)
    assert all(row["invocation_id"] == actual.invocation_id for row in trace)
    assert trace[-1]["kind"] == "shard_result_emit"
    assert trace[-1]["result_sha256"] == hashlib.sha256(
        actual.result_bytes
    ).hexdigest()


def test_replay_accepts_existing_citation_redaction_markers(tmp_path: Path) -> None:
    cache, manifest, result = _fixture(tmp_path)
    raw = _raw_stream(cache, manifest, result)
    sanitized = executor.sanitize_shard_raw_stream(raw, cache)

    actual = replay.replay_shard_raw_review(
        sanitized, cache=cache, shard_manifest=manifest,
    )

    assert actual.result == result
    assert actual.raw_sha256 == hashlib.sha256(sanitized).hexdigest()


@pytest.mark.parametrize("mode", ["missing", "duplicate", "hash_only"])
def test_replay_requires_each_assigned_chunk_exactly_once(
    tmp_path: Path, mode: str,
) -> None:
    cache, manifest, result = _fixture(tmp_path)
    commands = _read_commands(manifest)
    if mode == "missing":
        commands = commands[:-1]
    elif mode == "duplicate":
        commands.append(commands[0])
    else:
        inner = review._unwrap_codex_shell_event_command(commands[0])
        path = inner.rsplit(" ", 1)[-1]
        commands[0] = review._codex_shell_event_command(f"sha256sum {path}")
    raw = _raw_stream(cache, manifest, result, commands=commands)

    with pytest.raises(replay.ShardReplayError, match="read exactly once"):
        replay.replay_shard_raw_review(
            raw, cache=cache, shard_manifest=manifest,
        )


@pytest.mark.parametrize(
    "command, message",
    [
        ("head -n 100000000 chunks/source/unassigned.utf8", "non-review cache"),
        ("python -c pass", "unapproved command executable"),
        ("echo x > output.txt", "chaining, expansion, redirection"),
    ],
)
def test_replay_rejects_unassigned_reads_and_write_capable_commands(
    tmp_path: Path, command: str, message: str,
) -> None:
    cache, manifest, result = _fixture(tmp_path)
    commands = _read_commands(manifest) + [f"/bin/bash -lc '{command}'"]
    raw = _raw_stream(cache, manifest, result, commands=commands)

    with pytest.raises(replay.ShardReplayError, match=message):
        replay.replay_shard_raw_review(
            raw, cache=cache, shard_manifest=manifest,
        )


def _insert_before_final(
    rows: list[dict[str, object]], value: dict[str, object],
) -> None:
    rows.insert(-2, value)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (
            lambda rows: _insert_before_final(
                rows, {"type": "turn.compacted"},
            ),
            "forbidden or unknown type",
        ),
        (
            lambda rows: _insert_before_final(
                rows,
                {
                    "type": "item.completed",
                    "item": {"id": "web-001", "type": "web_search"},
                },
            ),
            "forbidden or unknown tool",
        ),
        (
            lambda rows: rows.insert(
                2,
                {"type": "thread.started", "thread_id": "thread-other-0002"},
            ),
            "one unique thread",
        ),
        (
            lambda rows: rows.insert(-1, copy.deepcopy(rows[-2])),
            "multiple final messages",
        ),
    ],
)
def test_replay_rejects_noncanonical_lifecycle_and_tools(
    tmp_path: Path,
    mutate: Callable[[list[dict[str, object]]], None],
    message: str,
) -> None:
    cache, manifest, result = _fixture(tmp_path)
    rows = _events(_raw_stream(cache, manifest, result))
    mutate(rows)
    raw = review._canonical_jsonl(rows)

    with pytest.raises(replay.ShardReplayError, match=message):
        replay.replay_shard_raw_review(
            raw, cache=cache, shard_manifest=manifest,
        )


def test_replay_rejects_nested_compaction_and_open_usage_schema(
    tmp_path: Path,
) -> None:
    cache, manifest, result = _fixture(tmp_path)
    rows = _events(_raw_stream(cache, manifest, result))
    usage = rows[-1]["usage"]
    assert isinstance(usage, dict)
    usage["compaction_state"] = {"count": 1}
    with pytest.raises(replay.ShardReplayError, match="compaction metadata"):
        replay.replay_shard_raw_review(
            review._canonical_jsonl(rows), cache=cache,
            shard_manifest=manifest,
        )

    rows = _events(_raw_stream(cache, manifest, result))
    usage = rows[-1]["usage"]
    assert isinstance(usage, dict)
    usage["unknown_usage_counter"] = 1100
    with pytest.raises(replay.ShardReplayError, match="closed schema"):
        replay.replay_shard_raw_review(
            review._canonical_jsonl(rows), cache=cache,
            shard_manifest=manifest,
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda usage: usage.__setitem__(
                "total_tokens", usage.pop("reasoning_output_tokens")
            ),
            "closed schema",
        ),
        (
            lambda usage: usage.pop("reasoning_output_tokens"),
            "closed schema",
        ),
        (
            lambda usage: usage.__setitem__("input_tokens", True),
            "invalid input_tokens",
        ),
        (
            lambda usage: usage.__setitem__("output_tokens", -1),
            "invalid output_tokens",
        ),
        (
            lambda usage: usage.__setitem__("cached_input_tokens", 1001),
            "cached input tokens exceed",
        ),
        (
            lambda usage: usage.__setitem__("reasoning_output_tokens", 101),
            "reasoning output tokens exceed",
        ),
    ],
)
def test_replay_rejects_noncanonical_usage_counters(
    tmp_path: Path,
    mutate: Callable[[dict[str, object]], object],
    message: str,
) -> None:
    cache, manifest, result = _fixture(tmp_path)
    rows = _events(_raw_stream(cache, manifest, result))
    usage = rows[-1]["usage"]
    assert isinstance(usage, dict)
    mutate(usage)

    with pytest.raises(replay.ShardReplayError, match=message):
        replay.replay_shard_raw_review(
            review._canonical_jsonl(rows), cache=cache,
            shard_manifest=manifest,
        )


def test_replay_rejects_multiple_invocation_identities(tmp_path: Path) -> None:
    cache, manifest, result = _fixture(tmp_path)
    rows = _events(_raw_stream(cache, manifest, result))
    rows[0]["invocation_id"] = "invocation-one-0001"
    rows[1]["invocation_id"] = "invocation-two-0002"

    with pytest.raises(replay.ShardReplayError, match="invocation identities"):
        replay.replay_shard_raw_review(
            review._canonical_jsonl(rows),
            cache=cache,
            shard_manifest=manifest,
        )


def test_replay_rejects_nonclosed_or_cross_shard_final_result(
    tmp_path: Path,
) -> None:
    cache, manifest, result = _fixture(tmp_path)
    result["unexpected"] = True
    raw = _raw_stream(cache, manifest, result)

    with pytest.raises(replay.ShardReplayError, match="closed contract"):
        replay.replay_shard_raw_review(
            raw, cache=cache, shard_manifest=manifest,
        )


def test_replay_rejects_cache_projection_with_extra_assigned_surface(
    tmp_path: Path,
) -> None:
    cache, manifest, result = _fixture(tmp_path)
    changed = copy.deepcopy(cache.manifest)
    changed["files"].append(copy.deepcopy(changed["files"][0]))
    changed_cache = review.ReviewCache(
        root=cache.root,
        manifest=changed,
        manifest_bytes=review._canonical_json(changed),
    )
    raw = _raw_stream(cache, manifest, result)

    with pytest.raises(replay.ShardReplayError, match="different source files"):
        replay.replay_shard_raw_review(
            raw, cache=changed_cache, shard_manifest=manifest,
        )


def test_invocation_id_is_deterministic_and_raw_stream_unique(
    tmp_path: Path,
) -> None:
    cache, manifest, result = _fixture(tmp_path)
    first_raw = _raw_stream(
        cache, manifest, result, thread_id="thread-shard-0001",
    )
    second_raw = _raw_stream(
        cache, manifest, result, thread_id="thread-shard-0002",
    )

    first = replay.replay_shard_raw_review(
        first_raw, cache=cache, shard_manifest=manifest,
    )
    repeated = replay.replay_shard_raw_review(
        first_raw, cache=cache, shard_manifest=manifest,
    )
    second = replay.replay_shard_raw_review(
        second_raw, cache=cache, shard_manifest=manifest,
    )

    assert repeated.invocation_id == first.invocation_id
    assert repeated.raw_sha256 == first.raw_sha256
    assert second.invocation_id != first.invocation_id
    assert second.raw_sha256 != first.raw_sha256
