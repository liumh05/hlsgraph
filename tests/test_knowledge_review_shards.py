from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tools import knowledge_review_shards as shards
from tools import run_knowledge_review as review


ROOT = Path(__file__).parents[1]
CITATION_AUDIT = ROOT / "docs" / "knowledge-citation-audit-v0.3.json"
EXPECTED_PLAN_SHA256 = (
    "99b94e5bb48f83244eccd79d11b5d00e7d6eb39f97c5569e81b27156b2fbfa2d"
)


def _audit() -> dict[str, object]:
    return json.loads(CITATION_AUDIT.read_text(encoding="utf-8"))


def _rule_row(audit: dict[str, object], rule_id: str) -> dict[str, object]:
    rows = audit["references"]
    assert isinstance(rows, list)
    return next(
        row for row in rows
        if isinstance(row, dict) and row.get("rule_id") == rule_id
    )


class _CharacterTokenizer:
    def encode(self, text: str) -> range:
        return range(len(text))


class _ContractTokenizer:
    name = "o200k_base"
    n_vocab = 4
    max_token_value = 3
    _pat_str = "fixture-pattern"
    _mergeable_ranks = {b"a": 0, b"bc": 1}
    _special_tokens = {"<fixture>": 3}


def test_fixed_three_shard_plan_matches_current_closed_audit() -> None:
    plan = shards.build_shard_plan(_audit())

    assert plan["schema_version"] == shards.PLAN_SCHEMA_VERSION
    assert plan["shard_order"] == list(shards.SHARD_ORDER)
    assert [row["shard_id"] for row in plan["shards"]] == list(
        shards.SHARD_ORDER
    )
    assert [len(row["source_paths"]) for row in plan["shards"]] == [17, 10, 18]
    assert [len(row["rule_references"]) for row in plan["shards"]] == [10, 15, 13]
    assert sum(len(row["rule_references"]) for row in plan["shards"]) == 38
    assert plan["token_budget_contract"] == (
        shards.DEFAULT_TOKEN_BUDGET_CONTRACT.to_dict()
    )
    assert shards.shard_plan_sha256(plan) == EXPECTED_PLAN_SHA256


def test_shard_union_covers_the_complete_model_inspection_surface() -> None:
    assigned = set().union(*(
        set(shard.source_paths) for shard in shards.SHARD_DEFINITIONS
    ))
    original_sensitive = {
        shards.CITATION_AUDIT_SOURCE_PATH,
        shards.CITATION_EVIDENCE_SOURCE_PATH,
        shards.AMD_PACK_SOURCE_PATH,
        shards.AXI_PACK_SOURCE_PATH,
        shards.OPEN_IR_PACK_SOURCE_PATH,
    }
    original_expected = set(review.MODEL_INSPECTION_EXACT_PATHS) | {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "src" / "hlsgraph" / "knowledge" / "packs").glob(
            "*.json"
        )
    }
    virtual = {
        path for shard_id in shards.SHARD_ORDER
        for path in shards.projected_model_source_paths(shard_id)
    }
    expected = (original_expected - original_sensitive) | virtual
    assert assigned == expected
    assert assigned.isdisjoint(original_sensitive)


def test_knowledge_shard_sees_the_complete_binding_activation_boundary() -> None:
    knowledge = next(
        item for item in shards.SHARD_DEFINITIONS
        if item.shard_id == "knowledge_activation"
    )
    assert {
        "src/hlsgraph/retrieval.py",
        "src/hlsgraph/knowledge/activation.py",
        "src/hlsgraph/knowledge/core.py",
        "src/hlsgraph/extract/directive_replay.py",
        "src/hlsgraph/extract/directive_identity.py",
        "src/hlsgraph/extract/directives.py",
        "src/hlsgraph/extract/observation_replay.py",
        "src/hlsgraph/extract/source.py",
        "src/hlsgraph/bundle.py",
        "src/hlsgraph/graph.py",
    } <= set(knowledge.source_paths)


def test_shard_local_json_projections_do_not_expose_other_rule_ids() -> None:
    snapshot = review.freeze_review_snapshot(ROOT, shards.SEMANTIC_PROTOCOL_ID)
    all_rules = {item.rule_id for item in shards.RULE_REFERENCE_SPECS}
    observed_union: set[str] = set()
    for shard_id in shards.SHARD_ORDER:
        own = {
            item.rule_id for item in shards.RULE_REFERENCE_SPECS
            if item.shard_id == shard_id
        }
        other = all_rules - own
        audit_path = shards.model_source_projection_path(
            shard_id, shards.CITATION_AUDIT_SOURCE_PATH,
        )
        audit_projection = json.loads(snapshot.file_map[audit_path].payload)
        assert set(audit_projection["assigned_rule_ids"]) == own
        observed_union.update(audit_projection["assigned_rule_ids"])
        for virtual_path in shards.projected_model_source_paths(shard_id):
            payload = snapshot.file_map[virtual_path].payload
            value = json.loads(payload)
            assert value["shard_id"] == shard_id
            assert set(value["assigned_rule_ids"]) <= own
            text = payload.decode("utf-8")
            assert all(rule_id not in text for rule_id in other)
    assert observed_union == all_rules


def test_projection_builder_is_deterministic_and_rule_union_is_exact() -> None:
    snapshot = review.freeze_review_snapshot(ROOT, shards.SEMANTIC_PROTOCOL_ID)
    source_payloads = {
        path: snapshot.file_map[path].payload for path in (
            shards.CITATION_AUDIT_SOURCE_PATH,
            shards.CITATION_EVIDENCE_SOURCE_PATH,
            shards.AMD_PACK_SOURCE_PATH,
            shards.AXI_PACK_SOURCE_PATH,
            shards.OPEN_IR_PACK_SOURCE_PATH,
        )
    }
    first = shards.build_model_source_projections(source_payloads)
    second = shards.build_model_source_projections(dict(reversed(
        list(source_payloads.items())
    )))
    assert first == second
    assert set(first) == {
        path for shard_id in shards.SHARD_ORDER
        for path in shards.projected_model_source_paths(shard_id)
    }


def test_suite_tcb_is_frozen_but_not_exposed_as_model_content() -> None:
    expected = {
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
    }
    assert set(review.SUITE_REVIEW_SOURCE_PATHS) == expected
    assert expected.isdisjoint(review.MODEL_INSPECTION_EXACT_PATHS)
    for protocol_id in (
        shards.SEMANTIC_PROTOCOL_ID, shards.ADVERSARIAL_PROTOCOL_ID,
    ):
        assert expected <= review.required_read_paths(ROOT, protocol_id)


def test_validation_schemas_are_integrity_bound_but_not_model_visible() -> None:
    validation_schemas = {
        review.REVIEW_SCHEMA_PATH,
        review.CITATION_EVIDENCE_SCHEMA_PATH,
    }
    assigned = set().union(*(
        set(shard.source_paths) for shard in shards.SHARD_DEFINITIONS
    ))

    assert validation_schemas.isdisjoint(review.MODEL_INSPECTION_EXACT_PATHS)
    assert validation_schemas.isdisjoint(assigned)
    for protocol_id in (
        shards.SEMANTIC_PROTOCOL_ID, shards.ADVERSARIAL_PROTOCOL_ID,
    ):
        assert validation_schemas <= review.required_read_paths(ROOT, protocol_id)


def test_tokenizer_contract_fingerprints_tables_and_pinned_version() -> None:
    payload = shards.tokenizer_contract_payload(
        _ContractTokenizer(), package_version="0.13.0",
    )
    assert payload["mergeable_rank_count"] == 2
    assert payload["special_tokens"] == {"<fixture>": 3}
    changed = _ContractTokenizer()
    changed._mergeable_ranks = {b"a": 0, b"bd": 1}
    assert shards.tokenizer_contract_payload(
        changed, package_version="0.13.0",
    )["mergeable_ranks_sha256"] != payload["mergeable_ranks_sha256"]
    with pytest.raises(shards.TokenBudgetError, match="requires tiktoken"):
        shards.tokenizer_contract_payload(
            _ContractTokenizer(), package_version="0.12.0",
        )


def test_installed_formal_tokenizer_matches_pinned_tables() -> None:
    pytest.importorskip("tiktoken")
    encoding = shards.load_verified_tokenizer()
    assert encoding.name == shards.DEFAULT_TOKENIZER_ID


def test_plan_and_hash_are_independent_of_audit_reference_order() -> None:
    audit = _audit()
    reversed_audit = copy.deepcopy(audit)
    references = reversed_audit["references"]
    assert isinstance(references, list)
    references.reverse()

    first = shards.build_shard_plan(audit)
    second = shards.build_shard_plan(reversed_audit)

    assert second == first
    assert shards.shard_plan_sha256(second) == EXPECTED_PLAN_SHA256


def test_rule_references_sharing_a_url_have_one_owner() -> None:
    allocated = shards.allocate_rule_references(_audit())
    owners: dict[str, set[str]] = {}
    for shard_id, rows in allocated.items():
        for row in rows:
            owners.setdefault(row["citation_url"], set()).add(shard_id)

    assert all(len(owner_set) == 1 for owner_set in owners.values())
    interface_url = (
        "https://docs.amd.com/r/2024.2-English/ug1399-vitis-hls/"
        "pragma-HLS-interface"
    )
    interface_rows = [
        row for row in allocated["knowledge_activation"]
        if row["citation_url"] == interface_url
    ]
    assert len(interface_rows) == 2
    assert {row["section"] for row in interface_rows} == {
        "pragma HLS interface"
    }
    signoff_rows = [
        row for row in allocated["tool_evidence"]
        if row["section"] == "Verifying Timing Signoff"
    ]
    assert len(signoff_rows) == 2


@pytest.mark.parametrize("field", ["citation_url", "section"])
def test_rule_allocation_rejects_url_or_section_drift(field: str) -> None:
    audit = _audit()
    row = _rule_row(
        audit,
        "amd.ug1399:2024.2:directive.interface_is_port_contract",
    )
    row[field] = str(row[field]) + "-drift"

    expected = "citation URL" if field == "citation_url" else "section"
    with pytest.raises(shards.ShardPlanError, match=expected):
        shards.allocate_rule_references(audit)


def test_rule_allocation_rejects_missing_extra_and_duplicate_rules() -> None:
    audit = _audit()
    rows = audit["references"]
    assert isinstance(rows, list)
    target = _rule_row(
        audit,
        "amd.ug1399:2024.2:directive.array_partition_requests_banking",
    )

    missing = copy.deepcopy(audit)
    missing_rows = missing["references"]
    assert isinstance(missing_rows, list)
    missing_rows.remove(_rule_row(
        missing,
        "amd.ug1399:2024.2:directive.array_partition_requests_banking",
    ))
    with pytest.raises(shards.ShardPlanError, match="missing rule references"):
        shards.allocate_rule_references(missing)

    extra = copy.deepcopy(audit)
    extra_row = copy.deepcopy(target)
    extra_row["rule_id"] = "future.vendor:1.0:unexpected_rule"
    extra_rows = extra["references"]
    assert isinstance(extra_rows, list)
    extra_rows.append(extra_row)
    with pytest.raises(shards.ShardPlanError, match="unexpected rule references"):
        shards.allocate_rule_references(extra)

    duplicate = copy.deepcopy(audit)
    duplicate_rows = duplicate["references"]
    assert isinstance(duplicate_rows, list)
    duplicate_row = copy.deepcopy(_rule_row(
        duplicate,
        "amd.ug1399:2024.2:directive.array_partition_requests_banking",
    ))
    duplicate_row["reference_id"] = "f" * 64
    duplicate_rows.append(duplicate_row)
    with pytest.raises(shards.ShardPlanError, match="duplicate rule reference"):
        shards.allocate_rule_references(duplicate)


def test_document_references_are_not_assigned_to_model_shards() -> None:
    audit = _audit()
    rows = audit["references"]
    assert isinstance(rows, list)
    assert sum(
        isinstance(row, dict) and row.get("reference_kind") == "document"
        for row in rows
    ) == 15

    allocated = shards.allocate_rule_references(audit)

    assert sum(map(len, allocated.values())) == 38
    assert all(
        row["rule_id"] is not None
        for shard_rows in allocated.values() for row in shard_rows
    )


def test_assertion_owner_sets_are_closed_disjoint_and_complete() -> None:
    semantic = shards.assertion_owners(shards.SEMANTIC_PROTOCOL_ID)
    adversarial = shards.assertion_owners("adversarial")

    assert len(semantic) == 11
    assert len(adversarial) == 19
    assert set(semantic.values()) == set(shards.SHARD_ORDER)
    assert set(adversarial.values()) == set(shards.SHARD_ORDER)
    assert semantic["S03.directive_exact_scope_source_operand_proof"] == (
        "knowledge_activation"
    )
    assert semantic["S10.retrieval_plane_isolation"] == "ir_semantics"
    assert semantic[
        "S09.aggregate_static_feature_recomputation"
    ] == "tool_evidence"
    assert semantic[
        "S06.requested_effective_achieved_stage_and_three_gate_separation"
    ] == "tool_evidence"
    assert adversarial[
        "A19.requested_achieved_estimate_postroute_confusion"
    ] == "tool_evidence"
    assert adversarial["A11.aggregate_feature_spoof"] == "tool_evidence"
    with pytest.raises(shards.ShardPlanError, match="unknown review protocol"):
        shards.assertion_owners("unknown")


def test_token_budget_accepts_exact_250k_boundary() -> None:
    tokenizer = _CharacterTokenizer()
    prompt = "p" * (
        shards.MAX_VISIBLE_INPUT_TOKENS
        - shards.TOOL_EVENT_OVERHEAD_TOKENS
        - shards.RUNTIME_ENVELOPE_ALLOWANCE_TOKENS
        - 2
    )

    budget = shards.enforce_token_budget(
        prompt=prompt,
        chunks=["c"],
        commands=["h"],
        tokenizer=tokenizer,
    )

    assert budget.prompt_tokens == len(prompt)
    assert budget.chunk_tokens == 1
    assert budget.command_tokens == 1
    assert budget.tool_event_count == 1
    assert budget.tool_event_overhead_tokens == 192
    assert budget.runtime_envelope_allowance_tokens == 32_000
    assert budget.visible_input_tokens == 250_000
    assert budget.context_reserve_tokens == 122_000
    assert budget.within_budget is True


def test_token_budget_rejects_one_token_over_boundary() -> None:
    tokenizer = _CharacterTokenizer()
    prompt = "p" * (
        shards.MAX_VISIBLE_INPUT_TOKENS
        - shards.TOOL_EVENT_OVERHEAD_TOKENS
        - shards.RUNTIME_ENVELOPE_ALLOWANCE_TOKENS
        - 1
    )

    calculated = shards.calculate_token_budget(
        prompt=prompt,
        chunks=["c"],
        commands=["h"],
        tokenizer=tokenizer,
    )
    assert calculated.visible_input_tokens == 250_001
    assert calculated.within_budget is False
    with pytest.raises(shards.TokenBudgetExceeded, match="visible=250001"):
        shards.enforce_token_budget(
            prompt=prompt,
            chunks=["c"],
            commands=["h"],
            tokenizer=tokenizer,
        )


def test_token_budget_fails_closed_on_unaccounted_or_invalid_input() -> None:
    tokenizer = _CharacterTokenizer()
    with pytest.raises(shards.TokenBudgetError, match="exactly one read command"):
        shards.calculate_token_budget(
            prompt="p", chunks=["c"], commands=[], tokenizer=tokenizer,
        )
    with pytest.raises(shards.TokenBudgetError, match="strict UTF-8"):
        shards.calculate_token_budget(
            prompt=b"\xff", chunks=[], commands=[], tokenizer=tokenizer,
        )
    with pytest.raises(shards.TokenBudgetError, match="cannot tokenize prompt"):
        shards.calculate_token_budget(
            prompt="p", chunks=[], commands=[],
            tokenizer=lambda _text: (_ for _ in ()).throw(RuntimeError("boom")),
        )
    with pytest.raises(shards.TokenBudgetError, match="limits are fixed"):
        shards.TokenBudgetContract(max_visible_input_tokens=249_999)


def test_injected_tokenizer_identity_is_recorded_in_budget() -> None:
    contract = shards.TokenBudgetContract(
        tokenizer_id="fixture.character.v1",
        tokenizer_contract_sha256="a" * 64,
    )

    budget = shards.enforce_token_budget(
        prompt="prompt", chunks=[], commands=[],
        tokenizer=_CharacterTokenizer(), contract=contract,
    )

    assert budget.contract.tokenizer_id == "fixture.character.v1"
    assert budget.to_dict()["contract"]["tokenizer_contract_sha256"] == "a" * 64
    assert budget.to_dict()["within_budget"] is True
