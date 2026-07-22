from __future__ import annotations

import hashlib
import json

import pytest

from hlsgraph.knowledge import (
    LEGACY_PACK_SCHEMA_VERSIONS,
    PACK_SCHEMA_VERSION,
    KnowledgeCatalog,
    KnowledgePackError,
    filter_rules,
    index_local_document,
    load_builtin_packs,
    load_local_index,
    load_pack,
    matches_applicability,
    save_local_index,
)


def test_builtin_packs_are_citation_only_and_versioned():
    packs = load_builtin_packs()
    assert {pack.pack_id for pack in packs} == {
        "hlsgraph.amd.public_guidance.2024_2",
        "hlsgraph.axi.public_guidance.v1",
        "hlsgraph.open_ir.public_guidance.2026_07_21",
    }
    rule_ids: set[str] = set()
    for pack in packs:
        assert pack.schema_version in LEGACY_PACK_SCHEMA_VERSIONS | {PACK_SCHEMA_VERSION}
        assert pack.documents
        for document in pack.documents:
            assert document.official_url.startswith("https://")
        for rule in pack.rules:
            assert rule.id not in rule_ids
            rule_ids.add(rule.id)
            assert rule.section
            assert rule.citation_url.startswith("https://")
            assert rule.summary and len(rule.summary) <= 500
            assert all(
                not isinstance(constraint, list)
                for constraints in (rule.applicability, rule.condition)
                for constraint in constraints.values()
            )


@pytest.mark.parametrize("field", ["applicability", "condition"])
def test_rule_constraint_alternatives_require_explicit_one_of(field: str) -> None:
    rule = {
        "document_id": "test.spec", "document_version": "1",
        "section": "Section", "rule_id": "test.rule", "title": "Test rule",
        "applicability": {"stage": "source"},
        "condition": {"mode": "one"},
        # Lists in effect are ordinary result data, not match constraints.
        "effect": {"ordered_roles": ["producer", "consumer"]},
        "citation_url": "https://example.com/spec#section",
        "summary": "A short project-authored paraphrase.",
    }
    rule[field] = {"mode": ["one", "two"]}
    payload = {
        "schema_version": "1.0", "pack_id": "test.constraint.pack",
        "title": "Constraint syntax", "license": "Apache-2.0",
        "documents": [{
            "document_id": "test.spec", "document_version": "1",
            "title": "Test specification", "official_url": "https://example.com/spec",
            "publisher": "Example",
        }],
        "rules": [rule],
    }
    with pytest.raises(KnowledgePackError, match="explicit one_of"):
        load_pack(payload)

    rule[field] = {"mode": {"one_of": ["one", "two"]}}
    loaded = load_pack(payload)
    assert loaded.rules[0].effect["ordered_roles"] == ["producer", "consumer"]


def test_mutated_rule_bare_applicability_list_fails_closed() -> None:
    rule = KnowledgeCatalog.builtin().packs[0].rules[0]
    rule.applicability["stage"] = ["source", "schedule"]
    assert not matches_applicability(rule, {
        "vendor": "amd", "tool": "vitis_hls",
        "tool_version": "2024.2", "stage": "source",
    })


def test_version_and_applicability_filters_fail_closed():
    catalog = KnowledgeCatalog.builtin()
    rules = catalog.filter(document_id="amd.ug1399", document_version="2024.2")
    assert rules
    assert catalog.filter(document_id="amd.ug1399", document_version="2023.1") == []

    cosim = catalog.filter(
        applicability={
            "vendor": "amd", "tool": "vitis_hls",
            "tool_version": "2024.2", "stage": "cosim",
        }
    )
    assert {rule.rule_id for rule in cosim} == {
        "dataflow.dynamic_results_are_workload_scoped",
        "verification.cosim_is_workload_scoped",
    }
    dataflow_rule = next(rule for rule in rules
                         if rule.rule_id == "dataflow.dynamic_results_are_workload_scoped")
    assert matches_applicability(
        dataflow_rule, {
            "vendor": "AMD", "tool": "VITIS_HLS",
            "tool_version": "2024.2", "stage": "COSIM",
        }
    )
    assert not matches_applicability(dataflow_rule, {"vendor": "amd", "tool": "vitis_hls"})
    assert filter_rules([dataflow_rule], applicability={"vendor": "amd", "tool": "vivado"}) == []


def test_local_document_index_contains_metadata_not_document_bytes(tmp_path):
    secret = b"%PDF-1.7\nPRIVATE-UG-CONTENT-SENTINEL\n"
    document = tmp_path / "UG1399.pdf"
    document.write_bytes(secret)
    metadata = index_local_document(
        document,
        document_id="amd.ug1399",
        document_version="2024.2",
        title="User-owned UG1399 copy",
        official_url="https://docs.amd.com/r/2024.2-English/ug1399-vitis-hls/",
    )
    assert metadata.sha256 == hashlib.sha256(secret).hexdigest()
    assert metadata.size == len(secret)

    index_path = save_local_index([metadata], tmp_path / "index.json")
    serialized = index_path.read_text(encoding="utf-8")
    assert "PRIVATE-UG-CONTENT-SENTINEL" not in serialized
    assert "full_text" not in serialized
    assert load_local_index(index_path) == [metadata]


def test_pack_rejects_embedded_text_and_undeclared_documents():
    with pytest.raises(KnowledgePackError, match="metadata/citation-only"):
        load_pack({
            "schema_version": "1.0",
            "pack_id": "test.pack",
            "title": "bad",
            "license": "Apache-2.0",
            "documents": [],
            "rules": [],
            "full_text": "copied guide",
        })

    bad = {
        "schema_version": "1.0",
        "pack_id": "test.pack",
        "title": "bad",
        "license": "Apache-2.0",
        "documents": [],
        "rules": [{
            "document_id": "amd.ug1399",
            "document_version": "2024.2",
            "section": "Dataflow Viewer",
            "rule_id": "test.rule",
            "title": "test",
            "applicability": {},
            "condition": {},
            "effect": {},
            "citation_url": "https://docs.amd.com/example",
            "summary": "A short paraphrase.",
        }],
    }
    with pytest.raises(KnowledgePackError, match="undeclared document"):
        load_pack(bad)


def test_local_index_rejects_content_bearing_fields(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({
        "schema_version": "1.0",
        "documents": [{"content": "copied document"}],
    }), encoding="utf-8")
    with pytest.raises(KnowledgePackError, match="metadata/citation-only"):
        load_local_index(path)
