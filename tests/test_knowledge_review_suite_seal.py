from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import jsonschema
import pytest

from tools import knowledge_review_shards as shards
from tools import run_knowledge_review_suite as suite
from tools import seal_knowledge_review_suite as seal
from tests.test_knowledge_review_suite import _approved_shard, _projection, _state


ROOT = Path(__file__).resolve().parents[1]


def _file_sha256(path: str) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _budget() -> dict[str, object]:
    prompt = 1_000
    chunks = 100_000
    command = 1_000
    events = 20
    overhead = events * shards.TOOL_EVENT_OVERHEAD_TOKENS
    runtime_allowance = shards.RUNTIME_ENVELOPE_ALLOWANCE_TOKENS
    visible = prompt + chunks + command + overhead + runtime_allowance
    return {
        "contract": shards.DEFAULT_TOKEN_BUDGET_CONTRACT.to_dict(),
        "prompt_tokens": prompt,
        "chunk_tokens": chunks,
        "command_tokens": command,
        "tool_event_count": events,
        "tool_event_overhead_tokens": overhead,
        "runtime_envelope_allowance_tokens": runtime_allowance,
        "visible_input_tokens": visible,
        "context_reserve_tokens": (
            shards.MODEL_CONTEXT_WINDOW_TOKENS - visible
        ),
        "within_budget": True,
    }


def _protocol_artifacts(protocol_id: str, *, namespace: str):
    audit, plan, snapshot, cache = _state(protocol_id)
    runtime_hash = _digest("one-official-runtime")
    projections = {
        shard_id: _projection((audit, plan, snapshot, cache), shard_id)
        for shard_id in shards.SHARD_ORDER
    }
    results = {
        shard_id: _approved_shard(projections[shard_id])
        for shard_id in shards.SHARD_ORDER
    }
    aggregate = suite.aggregate_shard_results(
        protocol_id=protocol_id,
        snapshot_inventory=snapshot,
        cache_manifest=cache,
        citation_audit=audit,
        plan=plan,
        shard_results=[results[shard_id] for shard_id in shards.SHARD_ORDER],
    )
    invocations = []
    for index, shard_id in enumerate(shards.SHARD_ORDER):
        prefix = f"{namespace}:{shard_id}:{index}"
        invocations.append({
            "shard_manifest": projections[shard_id],
            "shard_result": results[shard_id],
            "invocation_id": f"invoke-{namespace}-{index:02d}",
            "thread_id": f"thread-{namespace}-{index:02d}",
            "raw_output_sha256": _digest(prefix + ":raw"),
            "sanitized_output_sha256": _digest(prefix + ":sanitized"),
            "reported_input_tokens": 100_000,
            "reported_cached_input_tokens": 50_000,
            "reported_output_tokens": 1_000,
            "reported_reasoning_output_tokens": 250,
            "derived_input_plus_output_tokens": 101_000,
            "normalized_shard_trace_sha256": _digest(prefix + ":trace"),
            "cache_manifest_sha256": _digest(prefix + ":cache"),
            "prompt_sha256": _digest(prefix + ":prompt"),
            "command_sha256": _digest(prefix + ":command"),
            "boundary_contract_sha256": _digest(prefix + ":boundary"),
            "runtime_manifest_sha256": runtime_hash,
            "replay_contract_sha256": _digest("one-replay-contract"),
            "assigned_chunk_inventory_sha256": _digest(prefix + ":chunks"),
            "token_budget": _budget(),
            "completed": True,
            "exit_code": 0,
            "compaction_event_count": 0,
            "unknown_event_count": 0,
            "assigned_chunks_complete": True,
        })
    full_evidence = suite.citation_evidence_surface_sha256(cache["citations"])
    trace = seal.build_protocol_trace(
        protocol_id=protocol_id,
        plan=plan,
        citation_audit=audit,
        review_snapshot_sha256=suite.review_snapshot_sha256(snapshot),
        citation_evidence_sha256=snapshot["citation_evidence_sha256"],
        full_evidence_surface_sha256=full_evidence,
        runtime_manifest_sha256=runtime_hash,
        invocations=invocations,
        aggregate_result=aggregate,
    )
    receipt_schema_sha = _file_sha256(
        "tools/knowledge_review_suite_receipt.schema.json"
    )
    trace_schema_sha = _file_sha256(
        "tools/knowledge_review_suite_trace.schema.json"
    )
    receipt = seal.build_protocol_receipt(
        trace_bytes=trace,
        protocol_id=protocol_id,
        plan=plan,
        citation_audit=audit,
        review_snapshot_sha256=suite.review_snapshot_sha256(snapshot),
        citation_evidence_sha256=snapshot["citation_evidence_sha256"],
        full_evidence_surface_sha256=full_evidence,
        runtime_manifest_sha256=runtime_hash,
        invocations=invocations,
        aggregate_result=aggregate,
        output_schema_sha256=_file_sha256("tools/knowledge_review.schema.json"),
        shard_output_schema_sha256=_file_sha256(
            "tools/knowledge_review_shard.schema.json"
        ),
        suite_receipt_schema_sha256=receipt_schema_sha,
        suite_trace_schema_sha256=trace_schema_sha,
    )
    return {
        "audit": audit,
        "plan": plan,
        "snapshot": snapshot,
        "cache": cache,
        "runtime": runtime_hash,
        "full_evidence": full_evidence,
        "invocations": invocations,
        "aggregate": aggregate,
        "trace": trace,
        "receipt": receipt,
        "receipt_schema_sha": receipt_schema_sha,
        "trace_schema_sha": trace_schema_sha,
    }


def _pair():
    semantic = _protocol_artifacts(
        shards.SEMANTIC_PROTOCOL_ID, namespace="semantic",
    )
    adversarial = _protocol_artifacts(
        shards.ADVERSARIAL_PROTOCOL_ID, namespace="adversarial",
    )
    assert semantic["plan"] == adversarial["plan"]
    assert semantic["audit"] == adversarial["audit"]
    assert semantic["runtime"] == adversarial["runtime"]
    assert semantic["full_evidence"] == adversarial["full_evidence"]
    return semantic, adversarial


def test_receipt_hashes_the_exact_public_artifact_bytes() -> None:
    semantic, adversarial = _pair()
    receipt = semantic["receipt"]
    invocation = semantic["invocations"][0]
    public = receipt["shard_invocations"][0]
    assert public["shard_manifest_sha256"] == hashlib.sha256(
        seal.artifact_json_bytes(invocation["shard_manifest"])
    ).hexdigest()
    assert public["shard_result_sha256"] == hashlib.sha256(
        seal.artifact_json_bytes(invocation["shard_result"])
    ).hexdigest()
    assert receipt["result_sha256"] == hashlib.sha256(
        seal.artifact_json_bytes(semantic["aggregate"])
    ).hexdigest()
    pair = seal.validate_suite_pair(
        semantic_receipt=semantic["receipt"],
        adversarial_receipt=adversarial["receipt"],
        semantic_result=semantic["aggregate"],
        adversarial_result=adversarial["aggregate"],
        plan=semantic["plan"], citation_audit=semantic["audit"],
    )
    assert pair["receipt_sha256s"] == sorted([
        seal.artifact_sha256(semantic["receipt"]),
        seal.artifact_sha256(adversarial["receipt"]),
    ])


def test_v6_receipt_and_trace_are_closed_and_schema_valid() -> None:
    semantic, _adversarial = _pair()
    receipt_schema = json.loads(
        (ROOT / "tools/knowledge_review_suite_receipt.schema.json").read_text()
    )
    trace_schema = json.loads(
        (ROOT / "tools/knowledge_review_suite_trace.schema.json").read_text()
    )
    jsonschema.Draft202012Validator.check_schema(receipt_schema)
    jsonschema.Draft202012Validator.check_schema(trace_schema)
    jsonschema.Draft202012Validator(receipt_schema).validate(
        semantic["receipt"]
    )
    rows = seal.parse_protocol_trace(semantic["trace"])
    for row in rows:
        jsonschema.Draft202012Validator(trace_schema).validate(row)
    assert [row["kind"] for row in rows] == [
        "suite_start", "shard_invocation", "shard_invocation",
        "shard_invocation", "aggregate_emit",
    ]
    assert semantic["receipt"]["schema_version"] == (
        seal.SUITE_RECEIPT_SCHEMA_VERSION
    )


def test_protocol_receipt_rebuilds_exactly() -> None:
    semantic, _adversarial = _pair()
    validated = seal.validate_protocol_receipt(
        semantic["receipt"],
        trace_bytes=semantic["trace"],
        protocol_id=shards.SEMANTIC_PROTOCOL_ID,
        plan=semantic["plan"],
        citation_audit=semantic["audit"],
        review_snapshot_sha256=suite.review_snapshot_sha256(
            semantic["snapshot"]
        ),
        citation_evidence_sha256=semantic["snapshot"][
            "citation_evidence_sha256"
        ],
        full_evidence_surface_sha256=semantic["full_evidence"],
        runtime_manifest_sha256=semantic["runtime"],
        invocations=semantic["invocations"],
        aggregate_result=semantic["aggregate"],
        output_schema_sha256=_file_sha256("tools/knowledge_review.schema.json"),
        shard_output_schema_sha256=_file_sha256(
            "tools/knowledge_review_shard.schema.json"
        ),
        suite_receipt_schema_sha256=semantic["receipt_schema_sha"],
        suite_trace_schema_sha256=semantic["trace_schema_sha"],
    )
    assert validated == semantic["receipt"]
    tampered = semantic["trace"].replace(b'"sequence":2', b'"sequence":9', 1)
    with pytest.raises(seal.SuiteSealError, match="differs|closed"):
        seal.validate_protocol_trace(
            tampered,
            protocol_id=shards.SEMANTIC_PROTOCOL_ID,
            plan=semantic["plan"],
            citation_audit=semantic["audit"],
            review_snapshot_sha256=suite.review_snapshot_sha256(
                semantic["snapshot"]
            ),
            citation_evidence_sha256=semantic["snapshot"][
                "citation_evidence_sha256"
            ],
            full_evidence_surface_sha256=semantic["full_evidence"],
            runtime_manifest_sha256=semantic["runtime"],
            invocations=semantic["invocations"],
            aggregate_result=semantic["aggregate"],
        )


def test_pair_seal_requires_six_unique_invocation_thread_and_raw_hashes() -> None:
    semantic, adversarial = _pair()
    pair = seal.validate_suite_pair(
        semantic_receipt=semantic["receipt"],
        adversarial_receipt=adversarial["receipt"],
        semantic_result=semantic["aggregate"],
        adversarial_result=adversarial["aggregate"],
        plan=semantic["plan"],
        citation_audit=semantic["audit"],
    )
    assert pair["invocation_count"] == 6
    assert len(pair["invocation_ids"]) == len(set(pair["invocation_ids"])) == 6
    assert len(pair["thread_ids"]) == len(set(pair["thread_ids"])) == 6
    assert len(pair["raw_output_sha256s"]) == len(
        set(pair["raw_output_sha256s"])
    ) == 6
    assert len(pair["sanitized_output_sha256s"]) == len(
        set(pair["sanitized_output_sha256s"])
    ) == 6

    for field in (
        "invocation_id", "thread_id", "raw_output_sha256",
        "sanitized_output_sha256",
    ):
        changed = copy.deepcopy(adversarial["receipt"])
        changed["shard_invocations"][0][field] = semantic[
            "receipt"
        ]["shard_invocations"][0][field]
        with pytest.raises(seal.SuiteSealError, match=f"unique {field}"):
            seal.validate_suite_pair(
                semantic_receipt=semantic["receipt"],
                adversarial_receipt=changed,
                semantic_result=semantic["aggregate"],
                adversarial_result=adversarial["aggregate"],
                plan=semantic["plan"],
                citation_audit=semantic["audit"],
            )


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("runtime_manifest_sha256", "runtime_manifest_sha256"),
        ("full_evidence_surface_sha256", "full_evidence_surface_sha256"),
    ],
)
def test_pair_seal_requires_same_runtime_and_full_evidence(
    field: str, message: str,
) -> None:
    semantic, adversarial = _pair()
    changed = copy.deepcopy(adversarial["receipt"])
    changed[field] = "f" * 64
    with pytest.raises(seal.SuiteSealError, match=message):
        seal.validate_suite_pair(
            semantic_receipt=semantic["receipt"],
            adversarial_receipt=changed,
            semantic_result=semantic["aggregate"],
            adversarial_result=adversarial["aggregate"],
            plan=semantic["plan"],
            citation_audit=semantic["audit"],
        )


def test_invocation_rejects_compaction_unknown_and_budget_breach() -> None:
    semantic, _adversarial = _pair()
    base = semantic["invocations"]
    cases = [
        ("compaction_event_count", 1, "compaction"),
        ("unknown_event_count", 1, "unknown"),
        ("assigned_chunks_complete", False, "complete assigned"),
    ]
    for field, value, message in cases:
        changed = copy.deepcopy(base)
        changed[0][field] = value
        with pytest.raises(seal.SuiteSealError, match=message):
            seal.build_protocol_trace(
                protocol_id=shards.SEMANTIC_PROTOCOL_ID,
                plan=semantic["plan"],
                citation_audit=semantic["audit"],
                review_snapshot_sha256=suite.review_snapshot_sha256(
                    semantic["snapshot"]
                ),
                citation_evidence_sha256=semantic["snapshot"][
                    "citation_evidence_sha256"
                ],
                full_evidence_surface_sha256=semantic["full_evidence"],
                runtime_manifest_sha256=semantic["runtime"],
                invocations=changed,
                aggregate_result=semantic["aggregate"],
            )

    changed = copy.deepcopy(base)
    changed[0]["token_budget"]["chunk_tokens"] = 300_000
    with pytest.raises(seal.SuiteSealError, match="inconsistent|breaches"):
        seal.build_protocol_trace(
            protocol_id=shards.SEMANTIC_PROTOCOL_ID,
            plan=semantic["plan"],
            citation_audit=semantic["audit"],
            review_snapshot_sha256=suite.review_snapshot_sha256(
                semantic["snapshot"]
            ),
            citation_evidence_sha256=semantic["snapshot"][
                "citation_evidence_sha256"
            ],
            full_evidence_surface_sha256=semantic["full_evidence"],
            runtime_manifest_sha256=semantic["runtime"],
            invocations=changed,
            aggregate_result=semantic["aggregate"],
        )

    changed = copy.deepcopy(base)
    changed[0]["derived_input_plus_output_tokens"] += 1
    with pytest.raises(
        seal.SuiteSealError, match="reported or derived token usage",
    ):
        seal.build_protocol_trace(
            protocol_id=shards.SEMANTIC_PROTOCOL_ID,
            plan=semantic["plan"], citation_audit=semantic["audit"],
            review_snapshot_sha256=suite.review_snapshot_sha256(
                semantic["snapshot"]
            ),
            citation_evidence_sha256=semantic["snapshot"][
                "citation_evidence_sha256"
            ],
            full_evidence_surface_sha256=semantic["full_evidence"],
            runtime_manifest_sha256=semantic["runtime"],
            invocations=changed, aggregate_result=semantic["aggregate"],
        )

    changed = copy.deepcopy(base)
    changed[0]["reported_reasoning_output_tokens"] = (
        changed[0]["reported_output_tokens"] + 1
    )
    with pytest.raises(
        seal.SuiteSealError, match="reported or derived token usage",
    ):
        seal.build_protocol_trace(
            protocol_id=shards.SEMANTIC_PROTOCOL_ID,
            plan=semantic["plan"], citation_audit=semantic["audit"],
            review_snapshot_sha256=suite.review_snapshot_sha256(
                semantic["snapshot"]
            ),
            citation_evidence_sha256=semantic["snapshot"][
                "citation_evidence_sha256"
            ],
            full_evidence_surface_sha256=semantic["full_evidence"],
            runtime_manifest_sha256=semantic["runtime"],
            invocations=changed, aggregate_result=semantic["aggregate"],
        )


def test_cumulative_usage_is_not_compared_to_static_context_budget() -> None:
    semantic, _adversarial = _pair()
    changed = copy.deepcopy(semantic["invocations"])
    changed[0]["reported_input_tokens"] = 3_958_642
    changed[0]["reported_cached_input_tokens"] = 3_774_976
    changed[0]["reported_output_tokens"] = 8_339
    changed[0]["reported_reasoning_output_tokens"] = 2_353
    changed[0]["derived_input_plus_output_tokens"] = 3_966_981

    trace_bytes = seal.build_protocol_trace(
        protocol_id=shards.SEMANTIC_PROTOCOL_ID,
        plan=semantic["plan"], citation_audit=semantic["audit"],
        review_snapshot_sha256=suite.review_snapshot_sha256(
            semantic["snapshot"]
        ),
        citation_evidence_sha256=semantic["snapshot"][
            "citation_evidence_sha256"
        ],
        full_evidence_surface_sha256=semantic["full_evidence"],
        runtime_manifest_sha256=semantic["runtime"],
        invocations=changed, aggregate_result=semantic["aggregate"],
    )

    trace = seal.parse_protocol_trace(trace_bytes)
    assert trace[1]["reported_input_tokens"] == 3_958_642
    assert trace[1]["token_budget"]["visible_input_tokens"] < 250_001


def test_pair_rejects_non_exact_assertion_and_reference_unions() -> None:
    semantic, adversarial = _pair()
    changed = copy.deepcopy(semantic["receipt"])
    changed["assertion_ids"] = changed["assertion_ids"][:-1]
    with pytest.raises(seal.SuiteSealError, match="assertion union"):
        seal.validate_suite_pair(
            semantic_receipt=changed,
            adversarial_receipt=adversarial["receipt"],
            semantic_result=semantic["aggregate"],
            adversarial_result=adversarial["aggregate"],
            plan=semantic["plan"],
            citation_audit=semantic["audit"],
        )

    for field, message in (
        ("rule_reference_ids", "rule-reference union"),
        ("result_reference_ids", "result-reference union"),
    ):
        left = copy.deepcopy(semantic["receipt"])
        right = copy.deepcopy(adversarial["receipt"])
        left[field] = left[field][:-1]
        right[field] = right[field][:-1]
        with pytest.raises(seal.SuiteSealError, match=message):
            seal.validate_suite_pair(
                semantic_receipt=left,
                adversarial_receipt=right,
                semantic_result=semantic["aggregate"],
                adversarial_result=adversarial["aggregate"],
                plan=semantic["plan"],
                citation_audit=semantic["audit"],
            )


def test_pair_rejects_protocol_citation_disagreement() -> None:
    semantic, adversarial = _pair()
    changed_result = copy.deepcopy(adversarial["aggregate"])
    changed_result["citation_results"][0]["verdict"] = "rejected"
    with pytest.raises(seal.SuiteSealError, match="verdicts disagree"):
        seal.validate_suite_pair(
            semantic_receipt=semantic["receipt"],
            adversarial_receipt=adversarial["receipt"],
            semantic_result=semantic["aggregate"],
            adversarial_result=changed_result,
            plan=semantic["plan"],
            citation_audit=semantic["audit"],
        )
