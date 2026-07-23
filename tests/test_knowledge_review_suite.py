from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import jsonschema
import pytest

from tools import knowledge_review_shards as shards
from tools import run_knowledge_review as review
from tools import run_knowledge_review_suite as suite


ROOT = Path(__file__).resolve().parents[1]


def _bytes(path: str) -> bytes:
    return (ROOT / path).read_bytes()


def _json(path: str) -> dict[str, object]:
    return json.loads(_bytes(path))


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _snapshot(protocol_id: str, audit: dict[str, object]) -> dict[str, object]:
    surfaces = {
        str(row["pack_id"]): str(row["review_surface_sha256"])
        for row in audit["packs"]  # type: ignore[index]
    }
    return {
        "protocol_id": protocol_id,
        "review_surface_sha256": surfaces,
        "implementation_surface_sha256": "1" * 64,
        "citation_audit_sha256": hashlib.sha256(
            _bytes("docs/knowledge-citation-audit-v0.3.json")
        ).hexdigest(),
        "citation_evidence_sha256": hashlib.sha256(
            _bytes("docs/knowledge-review-evidence-v0.3.json")
        ).hexdigest(),
        "output_schema_sha256": hashlib.sha256(
            _bytes("tools/knowledge_review.schema.json")
        ).hexdigest(),
        "receipt_schema_sha256": hashlib.sha256(
            _bytes("tools/knowledge_review_receipt.schema.json")
        ).hexdigest(),
        "exact_citation_urls": sorted({
            str(row["citation_url"])
            for row in audit["references"]  # type: ignore[index]
        }),
        "required_files": [],
    }


def _chunk(origin: str, *, kind: str) -> dict[str, object]:
    digest = _digest(origin + ":payload")
    return {
        "index": 0,
        "path": f"chunks/{kind}/{_digest(origin)}/000000-{digest}.utf8",
        "sha256": digest,
        "size": 16,
        "byte_start": 0,
        "byte_end": 16,
        "original_sha256": digest,
        "original_size": 16,
    }


def _cache(
    protocol_id: str, snapshot: dict[str, object],
    audit: dict[str, object], plan: dict[str, object],
) -> dict[str, object]:
    source_paths = sorted({
        str(path)
        for shard in plan["shards"]  # type: ignore[index]
        for path in shard["source_paths"]
    })
    files = []
    for path in source_paths:
        digest = _digest(path + ":source")
        files.append({
            "path": path,
            "hash_kind": "raw_sha256",
            "sha256": digest,
            "cache_path": f"files/{path}",
            "cache_sha256": digest,
            "cache_size": 16,
            "model_inspection_required": True,
            "chunks": [_chunk(path, kind="source")],
        })

    evidence = {
        str(row["citation_url"]): row
        for row in _json("docs/knowledge-review-evidence-v0.3.json")["entries"]
    }
    references_by_url: dict[str, list[dict[str, object]]] = {}
    for row in audit["references"]:  # type: ignore[index]
        references_by_url.setdefault(str(row["citation_url"]), []).append(row)
    citations = []
    for url in sorted(references_by_url):
        references = references_by_url[url]
        is_rule = any(row["reference_kind"] == "rule" for row in references)
        mapping = evidence[url]
        body_sha256 = _digest(url + ":body")
        inspection_sha256 = _digest(url + ":inspection") if is_rule else None
        citations.append({
            "requested_url": url,
            "evidence_url": mapping["evidence_url"],
            "resolver_id": mapping["resolver_id"],
            "reference_ids": sorted(str(row["reference_id"]) for row in references),
            "inspection_required": is_rule,
            "identity_verified": True,
            "available": True,
            "status": 200,
            "final_url": mapping["evidence_url"],
            "redirect_chain": [mapping["evidence_url"]],
            "content_type": "text/plain" if is_rule else "text/html",
            "body_path": f"citations/bodies/{body_sha256}.body",
            "body_sha256": body_sha256,
            "body_size": 128,
            "inspection_path": (
                f"citations/text/{inspection_sha256}.txt" if is_rule else None
            ),
            "inspection_sha256": inspection_sha256,
            "inspection_size": 64 if is_rule else None,
            "parser_id": "fixture.parser.v1" if is_rule else None,
            "parser_version": "fixture/1" if is_rule else None,
            "parser_command_sha256": _digest("fixture parser") if is_rule else None,
            "parser_executable_sha256": None,
            "parser_version_output_sha256": None,
            "resolver_artifacts": [],
            "inspection_chunks": [_chunk(url, kind="citation")] if is_rule else [],
            "error_code": None,
        })
    return {
        "schema_version": "fixture.review-cache.v1",
        "protocol_id": protocol_id,
        "review_snapshot_sha256": suite.review_snapshot_sha256(snapshot),
        "review_snapshot": copy.deepcopy(snapshot),
        "files": files,
        "citations": citations,
        "chunk_contract": {"schema_version": "fixture.chunk.v1", "sha256": "2" * 64},
    }


def _state(protocol_id: str = shards.SEMANTIC_PROTOCOL_ID):
    audit = _json("docs/knowledge-citation-audit-v0.3.json")
    plan = shards.build_shard_plan(audit)
    snapshot = _snapshot(protocol_id, audit)
    cache = _cache(protocol_id, snapshot, audit, plan)
    return audit, plan, snapshot, cache


def test_snapshot_hash_matches_the_frozen_runner_contract() -> None:
    snapshot = review.freeze_review_snapshot(
        ROOT, shards.SEMANTIC_PROTOCOL_ID,
    )
    assert suite.review_snapshot_sha256(snapshot.inventory()) == snapshot.sha256


def _projection(state, shard_id: str):
    _audit, plan, snapshot, cache = state
    return suite.project_shard_manifest(
        protocol_id=snapshot["protocol_id"],
        snapshot_inventory=snapshot,
        cache_manifest=cache,
        plan=plan,
        shard_id=shard_id,
    )


def _approved_shard(projection: dict[str, object]) -> dict[str, object]:
    assertions = [{
        "assertion_id": assertion_id,
        "verdict": "verified",
        "issues": [],
    } for assertion_id in projection["assertion_ids"]]
    citations = [{
        "reference_id": row["reference_id"],
        "reference_surface_sha256": row["reference_surface_sha256"],
        "verdict": "verified",
        "exact_locator_inspected": True,
        "declared_version_matched": True,
        "declared_section_matched": True,
        "paraphrase_supported": True,
        "applicability_not_broader": True,
        "issues": [],
    } for row in projection["rule_references"]]
    return {
        "schema_version": suite.SHARD_RESULT_SCHEMA_VERSION,
        "protocol_id": projection["protocol_id"],
        "review_snapshot_sha256": projection["review_snapshot_sha256"],
        "shard_plan_sha256": projection["shard_plan_sha256"],
        "shard_id": projection["shard_id"],
        "citation_evidence_surface_sha256": projection[
            "citation_evidence_surface_sha256"
        ],
        "assertion_results": assertions,
        "citation_results": citations,
        "approved": True,
        "issues": [],
        "summary": "approved_no_issues",
    }


def test_projection_is_rule_only_protocol_neutral_and_body_path_free() -> None:
    semantic = _state()
    adversarial = _state(shards.ADVERSARIAL_PROTOCOL_ID)
    audit = semantic[0]
    document_ids = {
        row["reference_id"] for row in audit["references"]
        if row["reference_kind"] == "document"
    }
    for shard_id in shards.SHARD_ORDER:
        projected = _projection(semantic, shard_id)
        adversarial_projection = _projection(adversarial, shard_id)
        plan_row = next(
            row for row in semantic[1]["shards"] if row["shard_id"] == shard_id
        )
        assert projected["source_paths"] == sorted(plan_row["source_paths"])
        assert {row["path"] for row in projected["files"]} == set(
            plan_row["source_paths"]
        )
        assigned = {
            row["reference_id"] for row in projected["rule_references"]
        }
        visible = {
            reference_id
            for row in projected["citations"]
            for reference_id in row["reference_ids"]
        }
        assert visible == assigned
        assert visible.isdisjoint(document_ids)
        assert all(row["inspection_chunks"] for row in projected["citations"])
        assert "body_path" not in json.dumps(projected, sort_keys=True)
        assert projected["citation_evidence_surface_sha256"] == (
            adversarial_projection["citation_evidence_surface_sha256"]
        )
        assert projected["full_citation_evidence_surface_sha256"] == (
            suite.citation_evidence_surface_sha256(semantic[3]["citations"])
        )
        assert projected["full_citation_evidence_surface_sha256"] == (
            adversarial_projection["full_citation_evidence_surface_sha256"]
        )


def test_prompt_requires_only_assigned_assertions_and_rule_references() -> None:
    state = _state()
    projection = _projection(state, "knowledge_activation")
    prompt = suite.build_shard_prompt(
        base_protocol_text="BASE PROTOCOL",
        snapshot_inventory=state[2],
        plan_sha256=shards.shard_plan_sha256(state[1]),
        shard_projection=projection,
    )
    assert prompt == suite.build_shard_prompt(
        base_protocol_text=b"BASE PROTOCOL",
        snapshot_inventory=state[2],
        plan_sha256=shards.shard_plan_sha256(state[1]),
        shard_projection=projection,
    )
    text = prompt.decode("utf-8")
    for assertion_id in projection["assertion_ids"]:
        assert assertion_id in text
    all_assertions = set(shards.assertion_owners(
        shards.SEMANTIC_PROTOCOL_ID,
    ))
    assert all(
        assertion_id not in text
        for assertion_id in all_assertions - set(projection["assertion_ids"])
    )
    for row in projection["rule_references"]:
        assert row["reference_id"] in text
    document_ids = {
        row["reference_id"] for row in state[0]["references"]
        if row["reference_kind"] == "document"
    }
    assert all(reference_id not in text for reference_id in document_ids)
    assert "Document references are suite-generated" in text
    assert "assertion_results sorted by assertion_id" in text
    assert "citation_results sorted by reference_id" in text
    assert "top-level issues sorted by severity then code" in text
    assert "never duplicate an issue" in text


@pytest.mark.parametrize(
    "protocol_id,prompt_path",
    [
        (
            shards.SEMANTIC_PROTOCOL_ID,
            "tools/knowledge_review_prompts/semantic_shard.md",
        ),
        (
            shards.ADVERSARIAL_PROTOCOL_ID,
            "tools/knowledge_review_prompts/adversarial_shard.md",
        ),
    ],
)
@pytest.mark.parametrize("shard_id", shards.SHARD_ORDER)
def test_real_shard_prompt_does_not_name_another_shards_assertions(
    protocol_id: str, prompt_path: str, shard_id: str,
) -> None:
    state = _state(protocol_id)
    projection = _projection(state, shard_id)
    prompt = suite.build_shard_prompt(
        base_protocol_text=_bytes(prompt_path),
        snapshot_inventory=state[2],
        plan_sha256=shards.shard_plan_sha256(state[1]),
        shard_projection=projection,
    ).decode("utf-8")
    own = set(projection["assertion_ids"])
    all_assertions = set(shards.assertion_owners(protocol_id))
    assert all(assertion_id in prompt for assertion_id in own)
    assert all(assertion_id not in prompt for assertion_id in all_assertions - own)
    assert "The Codex shell tool is the only permitted tool" in prompt
    assert "MUST invoke one separate shell call for every" in prompt
    assert "head -n 100000000 PATH" in prompt
    assert "Do not emit preliminary JSON or a self-correction" in prompt
    assert "A rejecting result still requires every assigned chunk read" in prompt
    assert "generic shell" not in prompt


def test_evidence_surface_is_order_stable_and_changes_with_evidence() -> None:
    projection = _projection(_state(), "ir_semantics")
    citations = projection["citations"]
    baseline = suite.citation_evidence_surface_sha256(citations)
    assert baseline == suite.citation_evidence_surface_sha256(
        list(reversed(citations))
    )
    changed = copy.deepcopy(citations)
    changed[0]["body_sha256"] = "f" * 64
    assert suite.citation_evidence_surface_sha256(changed) != baseline
    changed = copy.deepcopy(citations)
    changed[0]["redirect_chain"].append(changed[0]["final_url"])
    assert suite.citation_evidence_surface_sha256(changed) != baseline


def test_shard_schema_and_closed_result_validate() -> None:
    projection = _projection(_state(), "tool_evidence")
    result = _approved_shard(projection)
    schema = _json("tools/knowledge_review_shard.schema.json")
    assert schema.get("type") == "object"
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(result)
    assert suite.validate_shard_result(result, projection) == result


def test_shard_schema_uses_openai_structured_outputs_subset() -> None:
    """Keep the model-facing schema inside OpenAI's documented strict subset."""

    schema = _json("tools/knowledge_review_shard.schema.json")
    allowed_keywords = {
        "type", "properties", "required", "additionalProperties",
        "items", "enum", "pattern",
    }

    def check(node: object, path: str = "$") -> None:
        assert isinstance(node, dict), f"{path} must be a schema object"
        assert set(node) <= allowed_keywords, (
            f"{path} uses unsupported Structured Outputs keywords: "
            f"{sorted(set(node) - allowed_keywords)}"
        )
        if node.get("type") == "object":
            properties = node.get("properties")
            assert isinstance(properties, dict), f"{path} has no properties"
            assert node.get("additionalProperties") is False
            assert set(node.get("required", [])) == set(properties)
            for name, child in properties.items():
                check(child, f"{path}.properties.{name}")
        if node.get("type") == "array":
            check(node.get("items"), f"{path}.items")
        if "enum" in node:
            assert node.get("type") == "string"
            assert isinstance(node["enum"], list) and node["enum"]
            assert all(isinstance(value, str) for value in node["enum"])
            assert len(node["enum"]) == len(set(node["enum"]))
        if "pattern" in node:
            assert node.get("type") == "string"

    check(schema)


def test_validate_rejects_missing_duplicate_and_cross_shard_assertions() -> None:
    state = _state()
    projection = _projection(state, "knowledge_activation")
    other = _projection(state, "ir_semantics")

    missing = _approved_shard(projection)
    missing["assertion_results"].pop()
    with pytest.raises(suite.SuiteContractError, match="assertion inventory"):
        suite.validate_shard_result(missing, projection)

    duplicate = _approved_shard(projection)
    duplicate["assertion_results"].append(
        copy.deepcopy(duplicate["assertion_results"][0])
    )
    with pytest.raises(suite.SuiteContractError, match="duplicate"):
        suite.validate_shard_result(duplicate, projection)

    cross = _approved_shard(projection)
    cross["assertion_results"].append({
        "assertion_id": other["assertion_ids"][0],
        "verdict": "verified",
        "issues": [],
    })
    with pytest.raises(suite.SuiteContractError, match="another shard"):
        suite.validate_shard_result(cross, projection)


def test_validate_rejects_missing_duplicate_and_cross_shard_references() -> None:
    state = _state()
    projection = _projection(state, "knowledge_activation")
    other = _projection(state, "ir_semantics")

    missing = _approved_shard(projection)
    missing["citation_results"].pop()
    with pytest.raises(suite.SuiteContractError, match="citation inventory"):
        suite.validate_shard_result(missing, projection)

    duplicate = _approved_shard(projection)
    duplicate["citation_results"].append(
        copy.deepcopy(duplicate["citation_results"][0])
    )
    with pytest.raises(suite.SuiteContractError, match="duplicate"):
        suite.validate_shard_result(duplicate, projection)

    cross = _approved_shard(projection)
    foreign = _approved_shard(other)["citation_results"][0]
    cross["citation_results"].append(copy.deepcopy(foreign))
    with pytest.raises(suite.SuiteContractError, match="another shard"):
        suite.validate_shard_result(cross, projection)


def test_validate_rejects_schema_issue_and_manifest_boundary_bypasses() -> None:
    projection = _projection(_state(), "knowledge_activation")

    stale = _approved_shard(projection)
    stale["schema_version"] = "hlsgraph.knowledge-review.shard-result.invalid"
    with pytest.raises(suite.SuiteContractError, match="header is stale"):
        suite.validate_shard_result(stale, projection)

    duplicate_assertion_issue = _approved_shard(projection)
    duplicate_assertion_issue["assertion_results"][0].update({
        "verdict": "rejected",
        "issues": ["assertion_rejected", "assertion_rejected"],
    })
    with pytest.raises(suite.SuiteContractError, match="uncontrolled issues"):
        suite.validate_shard_result(duplicate_assertion_issue, projection)

    non_string_assertion_issue = _approved_shard(projection)
    non_string_assertion_issue["assertion_results"][0].update({
        "verdict": "rejected",
        "issues": [[]],
    })
    with pytest.raises(suite.SuiteContractError, match="uncontrolled issues"):
        suite.validate_shard_result(non_string_assertion_issue, projection)

    duplicate_citation_issue = _approved_shard(projection)
    duplicate_citation_issue["citation_results"][0].update({
        "verdict": "rejected",
        "issues": ["inspection_incomplete", "inspection_incomplete"],
    })
    with pytest.raises(suite.SuiteContractError, match="uncontrolled issues"):
        suite.validate_shard_result(duplicate_citation_issue, projection)

    duplicated_top_level_issue = _approved_shard(projection)
    duplicated_top_level_issue["issues"] = [
        {"severity": "high", "code": "semantic_gap"},
        {"severity": "high", "code": "semantic_gap"},
    ]
    with pytest.raises(suite.SuiteContractError, match="duplicated or not sorted"):
        suite.validate_shard_result(duplicated_top_level_issue, projection)

    unsorted_top_level_issues = _approved_shard(projection)
    unsorted_top_level_issues["issues"] = [
        {"severity": "low", "code": "semantic_gap"},
        {"severity": "critical", "code": "activation_bypass"},
    ]
    with pytest.raises(suite.SuiteContractError, match="duplicated or not sorted"):
        suite.validate_shard_result(unsorted_top_level_issues, projection)

    forged_projection = copy.deepcopy(projection)
    forged_projection["assertion_ids"] = [""]
    with pytest.raises(suite.SuiteContractError, match="malformed assertion IDs"):
        suite.validate_shard_result(
            _approved_shard(forged_projection), forged_projection,
        )


def test_aggregate_is_deterministic_and_generates_document_rows() -> None:
    audit, plan, snapshot, cache = _state()
    projections = {
        shard_id: _projection((audit, plan, snapshot, cache), shard_id)
        for shard_id in shards.SHARD_ORDER
    }
    results = [_approved_shard(projections[shard_id]) for shard_id in shards.SHARD_ORDER]
    first = suite.aggregate_shard_results(
        protocol_id=snapshot["protocol_id"], snapshot_inventory=snapshot,
        cache_manifest=cache, citation_audit=audit, plan=plan,
        shard_results=list(reversed(results)),
    )
    second = suite.aggregate_shard_results(
        protocol_id=snapshot["protocol_id"], snapshot_inventory=snapshot,
        cache_manifest=cache, citation_audit=audit, plan=plan,
        shard_results=results,
    )
    assert first == second
    assert first["approved"] is True
    assert len(first["citation_results"]) == 53
    document_ids = {
        row["reference_id"] for row in audit["references"]
        if row["reference_kind"] == "document"
    }
    document_rows = [
        row for row in first["citation_results"]
        if row["reference_id"] in document_ids
    ]
    assert len(document_rows) == 15
    assert all(row["verdict"] == "verified" for row in document_rows)
    assert all(row["exact_locator_inspected"] is False for row in document_rows)
    jsonschema.Draft202012Validator(
        _json("tools/knowledge_review.schema.json")
    ).validate(first)

    unavailable = copy.deepcopy(cache)
    document_only_url = next(
        row["citation_url"] for row in audit["references"]
        if row["reference_kind"] == "document"
        and not any(
            candidate["reference_kind"] == "rule"
            and candidate["citation_url"] == row["citation_url"]
            for candidate in audit["references"]
        )
    )
    entry = next(
        row for row in unavailable["citations"]
        if row["requested_url"] == document_only_url
    )
    entry["available"] = False
    entry["identity_verified"] = False
    rejected = suite.aggregate_shard_results(
        protocol_id=snapshot["protocol_id"], snapshot_inventory=snapshot,
        cache_manifest=unavailable, citation_audit=audit, plan=plan,
        shard_results=results,
    )
    assert rejected["approved"] is False
    assert rejected["summary"] == "rejected_with_controlled_issues"
    assert {row["code"] for row in rejected["issues"]} == {
        "citation_unavailable"
    }


def test_aggregate_requires_exactly_one_result_per_shard() -> None:
    audit, plan, snapshot, cache = _state()
    results = [
        _approved_shard(_projection((audit, plan, snapshot, cache), shard_id))
        for shard_id in shards.SHARD_ORDER
    ]
    with pytest.raises(suite.SuiteContractError, match="fixed three shards"):
        suite.aggregate_shard_results(
            protocol_id=snapshot["protocol_id"], snapshot_inventory=snapshot,
            cache_manifest=cache, citation_audit=audit, plan=plan,
            shard_results=results[:-1],
        )
    with pytest.raises(suite.SuiteContractError, match="duplicate"):
        suite.aggregate_shard_results(
            protocol_id=snapshot["protocol_id"], snapshot_inventory=snapshot,
            cache_manifest=cache, citation_audit=audit, plan=plan,
            shard_results=[results[0], results[0], results[2]],
        )
