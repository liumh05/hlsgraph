from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

import hlsgraph.knowledge.sidecar as sidecar_module

from hlsgraph import (
    CanonicalGraph,
    CoverageEntry,
    CoverageManifest,
    CoverageStatus,
    KnowledgeTargetCoverage,
    KnowledgeBinding,
    LocalKnowledgeIndexManifest,
    Entity,
    TargetCoverageStatus,
)
from hlsgraph.bundle import GraphBundle
from hlsgraph.knowledge import (
    binding_entails_rule_condition,
    KnowledgeCatalog,
    KnowledgePackError,
    LocalDocumentMetadata,
    LocalKnowledgeSidecar,
    index_local_document,
    knowledge_activation_hash,
    load_pack,
    matches_binding_constraints,
    migrate_pack,
    pack_migration_plan,
)
from hlsgraph.manifest import minimal_manifest
from hlsgraph.model import json_ready
from hlsgraph.retrieval import HybridRetriever
from hlsgraph.store import LedgerStore, StoreError
from hlsgraph.version import SCHEMA_VERSION
from tools.knowledge_review_surface import surface_sha256


class _PdfParser:
    name = "test.pdf_parser"
    version = "1"
    fingerprint = hashlib.sha256(b"test.pdf_parser.v1").hexdigest()

    @staticmethod
    def capabilities():
        return {"protocol_version": "hlsgraph.knowledge_parser.v1",
                "local_only": True, "network_access": False,
                "media_types": ["application/pdf"]}

    @staticmethod
    def parse(data, metadata):
        return "PDF pipeline schedule evidence"


class _SlowPdfParser(_PdfParser):
    fingerprint = hashlib.sha256(b"test.slow_pdf_parser.v1").hexdigest()

    @staticmethod
    def parse(data, metadata):
        import time
        time.sleep(10)
        return "late"


class _NonTextPdfParser(_PdfParser):
    fingerprint = hashlib.sha256(b"test.nontext_pdf_parser.v1").hexdigest()

    @staticmethod
    def parse(data, metadata):
        return {"not": "text"}


class _HugePdfParser(_PdfParser):
    fingerprint = hashlib.sha256(b"test.huge_pdf_parser.v1").hexdigest()

    @staticmethod
    def parse(data, metadata):
        return "x" * 1000


class _ExitPdfParser(_PdfParser):
    fingerprint = hashlib.sha256(b"test.exit_pdf_parser.v1").hexdigest()

    @staticmethod
    def parse(data, metadata):
        import os
        os._exit(17)


def _bundle(tmp_path: Path) -> GraphBundle:
    (tmp_path / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    return GraphBundle.create(
        tmp_path,
        minimal_manifest("test.knowledge.v03", "knowledge v0.3", "dut", "kernel.cpp"),
    )


def _reviewed_pack(pack):
    """Create deterministic, explicitly test-only review evidence."""

    value = json_ready(pack)
    value["metadata"]["review_status"] = "machine_repeated_reviewed"
    value["coverage"].update({
        "review_status": "machine_repeated_reviewed",
        "reviewers": [
            "test.model@pinned#invocation-1",
            "test.model@pinned#invocation-2",
        ],
        "source_hashes": {
            f"{item.document_id}@{item.document_version}": hashlib.sha256(
                f"{item.document_id}@{item.document_version}".encode()
            ).hexdigest()
            for item in pack.documents
        },
        "review_evidence": {
            "independent_invocations": True,
            "citation_verified": True,
            "review_agreement": True,
            "unresolved_conflicts": False,
            "same_model_repeated_review": True,
            "distinct_model_families": False,
        },
    })
    value["coverage"].pop("id", None)
    return load_pack(value)


def test_binding_and_coverage_contracts_fail_closed() -> None:
    binding = KnowledgeBinding(
        knowledge_rule_id=(
            "amd.ug1399:2024.2:dataflow.dynamic_results_are_workload_scoped"
        ),
        target_kind="predicate",
        target="profile.fifo_max_occupancy",
        required_context={
            "vendor": "amd", "stage": "cosim",
            "workload_id": {"required": True},
        },
        producer="hlsgraph.knowledge.binding",
        producer_version="1",
    )
    assert not matches_binding_constraints(
        binding, target_kind="predicate", target="profile.fifo_max_occupancy",
        context={"vendor": "amd", "stage": "cosim"},
    )
    assert matches_binding_constraints(
        binding, target_kind="predicate", target="profile.fifo_max_occupancy",
        context={"vendor": "amd", "stage": "cosim", "workload_id": "tb.sha256"},
    )
    with pytest.raises(ValueError, match="two review invocations"):
        CoverageManifest(
            "test.pack", "test.scope",
            [CoverageEntry(
                "test.doc", "1", "section", "citation_only",
                rationale="A public citation exists without a normative rule.",
            )],
            review_status="machine_cross_reviewed", reviewers=["model.a"],
        )
    with pytest.raises(ValueError, match="verified source hashes"):
        CoverageManifest(
            "test.pack", "test.scope",
            [CoverageEntry(
                "test.doc", "1", "section", "citation_only",
                rationale="A public citation exists without a normative rule.",
            )],
            review_status="machine_cross_reviewed",
            reviewers=["model.a@pinned", "model.b@pinned"],
        )
    with pytest.raises(ValueError, match="truthful independent review provenance"):
        CoverageManifest(
            "test.pack", "test.scope",
            [CoverageEntry(
                "test.doc", "1", "section", "citation_only",
                rationale="A public citation exists without a normative rule.",
            )],
            review_status="machine_cross_reviewed",
            reviewers=["model.a@pinned", "model.b@pinned"],
            source_hashes={"test.doc@1": "a" * 64},
        )
    reviewed = CoverageManifest(
        "test.pack", "test.scope",
        [CoverageEntry(
            "test.doc", "1", "section", "citation_only",
            rationale="A public citation exists without a normative rule.",
        )],
        review_status="machine_cross_reviewed",
        reviewers=["model.a@pinned", "model.b@pinned"],
        source_hashes={"test.doc@1": "a" * 64},
        review_evidence={
            "independent_invocations": True,
            "citation_verified": True,
            "review_agreement": True,
            "unresolved_conflicts": False,
            "distinct_model_families": True,
        },
    )
    assert reviewed.complete
    assert reviewed.review_ready

    repeated = CoverageManifest(
        "test.pack", "test.scope",
        [CoverageEntry(
            "test.doc", "1", "section", "citation_only",
            rationale="A public citation exists without a normative rule.",
        )],
        review_status="machine_repeated_reviewed",
        reviewers=["model.a@pinned#invocation-1", "model.a@pinned#invocation-2"],
        source_hashes={"test.doc@1": "b" * 64},
        review_evidence={
            "independent_invocations": True,
            "same_model_repeated_review": True,
            "distinct_model_families": False,
            "citation_verified": True,
            "review_agreement": True,
            "unresolved_conflicts": False,
        },
    )
    assert repeated.complete
    assert repeated.review_ready
    assert not CoverageManifest(
        "test.pack", "test.scope",
        [CoverageEntry(
            "test.doc", "1", "section", "citation_only",
            rationale="A public citation exists without a normative rule.",
        )],
    ).review_ready
    with pytest.raises(ValueError, match="truthful independent review provenance"):
        CoverageManifest(
            "test.pack", "test.scope",
            [CoverageEntry(
                "test.doc", "1", "section", "citation_only",
                rationale="A public citation exists without a normative rule.",
            )],
            review_status="machine_repeated_reviewed",
            reviewers=["model.a@pinned#invocation-1", "model.a@pinned#invocation-2"],
            source_hashes={"test.doc@1": "b" * 64},
            review_evidence={
                "independent_invocations": True,
                "same_model_repeated_review": True,
                "distinct_model_families": True,
                "citation_verified": True,
                "review_agreement": True,
                "unresolved_conflicts": False,
            },
        )

    with pytest.raises(ValueError, match="requires a binding"):
        KnowledgeTargetCoverage(
            target_kind="predicate", target="test.value",
            status=TargetCoverageStatus.BOUND,
        )
    with pytest.raises(ValueError, match="requires a rationale"):
        KnowledgeTargetCoverage(
            target_kind="predicate", target="test.value",
            status=TargetCoverageStatus.NO_NORMATIVE,
        )
    builtin_binding_id = next(
        binding.id
        for pack in KnowledgeCatalog.builtin().packs
        for binding in pack.bindings
    )
    with pytest.raises(
        ValueError, match="only rule coverage may reference knowledge bindings",
    ):
        CoverageEntry(
            "test.doc", "1", "citation", "citation_only",
            binding_ids=[builtin_binding_id],
            rationale="A citation-only row cannot activate executable guidance.",
        )


def test_pack_coverage_ids_are_explicit_exact_and_not_synthesized() -> None:
    amd = next(
        pack for pack in KnowledgeCatalog.builtin().packs
        if pack.pack_id == "hlsgraph.amd.public_guidance.2024_2"
    )

    missing_binding = json_ready(amd)
    binding_entry = next(
        entry for entry in missing_binding["coverage"]["entries"]
        if entry["binding_ids"]
    )
    binding_entry["binding_ids"].pop()
    binding_entry.pop("id", None)
    missing_binding["coverage"].pop("id", None)
    with pytest.raises(
        KnowledgePackError,
        match="every knowledge binding must be covered exactly once",
    ):
        load_pack(missing_binding)

    wrong_rule = json_ready(amd)
    source_entry = next(
        entry for entry in wrong_rule["coverage"]["entries"]
        if entry["binding_ids"]
    )
    moved_binding_id = source_entry["binding_ids"].pop()
    moved_binding = next(
        item for item in amd.bindings if item.id == moved_binding_id
    )
    destination_entry = next(
        entry for entry in wrong_rule["coverage"]["entries"]
        if entry["status"] == "rule"
        and moved_binding.knowledge_rule_id not in entry["rule_ids"]
    )
    destination_entry["binding_ids"].append(moved_binding_id)
    source_entry.pop("id", None)
    destination_entry.pop("id", None)
    wrong_rule["coverage"].pop("id", None)
    with pytest.raises(KnowledgePackError, match="different knowledge rule"):
        load_pack(wrong_rule)

    missing_rule = json_ready(amd)
    removed_entry = next(
        entry for entry in missing_rule["coverage"]["entries"]
        if entry["status"] == "rule"
    )
    missing_rule["coverage"]["entries"].remove(removed_entry)
    missing_rule["coverage"].pop("id", None)
    with pytest.raises(
        KnowledgePackError,
        match="every knowledge rule must be covered exactly once",
    ):
        load_pack(missing_rule)


def test_pack_target_inventory_must_exactly_match_versioned_registry() -> None:
    amd = next(
        pack for pack in KnowledgeCatalog.builtin().packs
        if pack.pack_id == "hlsgraph.amd.public_guidance.2024_2"
    )

    missing_registry = json_ready(amd)
    del missing_registry["coverage"]["target_registry_version"]
    missing_registry["coverage"].pop("id", None)
    with pytest.raises(
        KnowledgePackError, match="explicitly declare target_registry_version",
    ):
        load_pack(missing_registry)

    missing_target = json_ready(amd)
    no_normative = next(
        item for item in missing_target["coverage"]["target_inventory"]
        if item["status"] == "no_normative"
    )
    missing_target["coverage"]["target_inventory"].remove(no_normative)
    missing_target["coverage"].pop("id", None)
    with pytest.raises(KnowledgePackError, match="exactly match the canonical"):
        load_pack(missing_target)

    extra_target = json_ready(amd)
    extra_target["coverage"]["target_inventory"].append({
        "target_kind": "predicate",
        "target": "test.unsupported_extra",
        "status": "no_normative",
        "binding_ids": [],
        "rationale": "An unsupported target must not extend the canonical inventory.",
    })
    extra_target["coverage"].pop("id", None)
    with pytest.raises(KnowledgePackError, match="exactly match the canonical"):
        load_pack(extra_target)


def test_retrieval_regates_injected_unreviewed_bindings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle(tmp_path)
    pack = next(
        item for item in KnowledgeCatalog.builtin().packs
        if item.bindings and not item.review_ready
    )
    retriever = HybridRetriever(bundle, bundle.snapshot().id)
    monkeypatch.setattr(
        bundle.store, "installed_knowledge_packs", lambda: [pack.inventory()],
    )
    monkeypatch.setattr(
        bundle.store, "knowledge_coverage", lambda: [pack.coverage],
    )
    monkeypatch.setattr(
        bundle.store, "knowledge_bindings", lambda: list(pack.bindings),
    )
    monkeypatch.setattr(
        bundle.store, "knowledge_rules", lambda: list(pack.rules),
    )
    assert retriever._review_ready_binding_ids() == set()

    reviewed = _reviewed_pack(pack)
    monkeypatch.setattr(
        bundle.store, "installed_knowledge_packs", lambda: [reviewed.inventory()],
    )
    monkeypatch.setattr(
        bundle.store, "knowledge_coverage", lambda: [reviewed.coverage],
    )
    monkeypatch.setattr(
        bundle.store, "knowledge_bindings", lambda: list(reviewed.bindings),
    )
    monkeypatch.setattr(
        bundle.store, "knowledge_rules", lambda: list(reviewed.rules),
    )
    assert retriever._review_ready_binding_ids() == {
        item.id for item in reviewed.bindings
    }

    altered_rules = list(reviewed.rules)
    altered = type(altered_rules[0])(**json_ready(altered_rules[0]))
    altered.title = altered.title + " (tampered)"
    altered_rules[0] = altered
    monkeypatch.setattr(
        bundle.store, "knowledge_rules", lambda: list(altered_rules),
    )
    assert retriever._review_ready_binding_ids() == set()

    # Even a recomputed inventory activation hash cannot make a changed rule
    # executable when its condition is no longer entailed by the reviewed
    # binding evidence contract. This exercises the independent read-side
    # entailment replay rather than relying on the inventory hash alone.
    condition_changed_rules = list(reviewed.rules)
    condition_changed = type(condition_changed_rules[0])(
        **json_ready(condition_changed_rules[0])
    )
    condition_changed.condition = {
        **condition_changed.condition,
        "protocol": "axis",
    }
    condition_changed_rules[0] = condition_changed
    forged_inventory = dict(reviewed.inventory())
    forged_inventory["activation_hash"] = knowledge_activation_hash(
        condition_changed_rules, reviewed.bindings, reviewed.coverage,
    )
    monkeypatch.setattr(
        bundle.store, "installed_knowledge_packs",
        lambda: [forged_inventory],
    )
    monkeypatch.setattr(
        bundle.store, "knowledge_rules",
        lambda: list(condition_changed_rules),
    )
    assert retriever._review_ready_binding_ids() == set()


def test_builtin_binding_boolean_discriminators_use_json_booleans() -> None:
    def scalar_values(value: object):
        if isinstance(value, dict):
            for item in value.values():
                yield from scalar_values(item)
        elif isinstance(value, list):
            for item in value:
                yield from scalar_values(item)
        else:
            yield value

    for pack in KnowledgeCatalog.builtin().packs:
        for binding in pack.bindings:
            assert not any(
                isinstance(item, str)
                and item.casefold() in {"true", "false"}
                for item in scalar_values(binding.required_context)
            ), binding.id


def test_binding_alternatives_require_explicit_one_of_in_both_matchers() -> None:
    common = {
        "knowledge_rule_id": "test.document:1:test.rule",
        "target_kind": "test.target",
        "target": "test.value",
        "producer": "test.binding",
        "producer_version": "1",
        "metadata": {"dynamic_scope": "static"},
    }
    explicit = KnowledgeBinding(
        **common,
        required_context={"stage": {"one_of": ["post_place", "post_route"]}},
    )
    scalar_context = {"stage": "post_route"}
    retrieval_context = {"stage": {"post_route"}}
    targets = {"test.target": {"test.value"}}
    assert matches_binding_constraints(
        explicit, target_kind="test.target", target="test.value",
        context=scalar_context,
    )
    assert HybridRetriever._constraint_matches(
        explicit.required_context["stage"], retrieval_context["stage"],
    )
    # A generic target remains lexical-only even when its scalar constraints
    # happen to match. Executable activation is closed to reviewed target
    # evidence contracts.
    assert not HybridRetriever._binding_constraints_match_values(
        explicit, retrieval_context, targets,
    )

    naked = KnowledgeBinding(
        **common,
        required_context={"stage": ["post_place", "post_route"]},
    )
    assert not matches_binding_constraints(
        naked, target_kind="test.target", target="test.value",
        context=scalar_context,
    )
    assert not HybridRetriever._binding_constraints_match_values(
        naked, retrieval_context, targets,
    )

    assert all(
        not isinstance(constraint, list)
        for pack in KnowledgeCatalog.builtin().packs
        for binding in pack.bindings
        for constraint in binding.required_context.values()
    )


def test_local_document_metadata_bounds_publicly_projected_fields() -> None:
    base = {
        "document_id": "test.local.doc", "document_version": "1",
        "uri": "file:///private/doc.md", "sha256": "a" * 64,
        "size": 1, "modified_ns": 1,
        "indexed_at": "2026-01-01T00:00:00+00:00",
    }
    with pytest.raises(KnowledgePackError, match="title"):
        LocalDocumentMetadata(**base, title="x" * 513)
    with pytest.raises(KnowledgePackError, match="version"):
        LocalDocumentMetadata(**{**base, "document_version": "v\x00bad"})
    with pytest.raises(KnowledgePackError, match="media_type"):
        LocalDocumentMetadata(**base, media_type="text/plain\x00private")


def test_local_document_metadata_index_is_stable_and_rejects_links(
    tmp_path: Path,
) -> None:
    source = tmp_path / "guide.md"
    source.write_text("pipeline evidence\n", encoding="utf-8")
    metadata = index_local_document(
        source, document_id="test.local.index", document_version="1",
    )
    target = tmp_path / "private" / "documents.json"
    from hlsgraph.knowledge import load_local_index, save_local_index

    save_local_index([metadata], target)
    assert load_local_index(target) == [metadata]
    if hasattr(target, "symlink_to"):
        link = tmp_path / "documents-link.json"
        try:
            link.symlink_to(target)
        except OSError:
            return
        with pytest.raises(KnowledgePackError, match="links/reparse"):
            load_local_index(link)


def test_builtin_pack_install_is_explicit_idempotent_and_separate_from_facts(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    catalog = KnowledgeCatalog.builtin()
    initially_installable = [
        pack for pack in catalog.packs
        if not pack.bindings or pack.review_ready
    ]
    inventory = bundle.store.installed_knowledge_packs()
    assert {item["pack_id"] for item in inventory} == {
        item.pack_id for item in initially_installable
    }
    unreviewed_executable = [
        pack for pack in catalog.packs if pack.bindings and not pack.review_ready
    ]
    for pack in unreviewed_executable:
        with pytest.raises(KnowledgePackError, match="not review_ready"):
            catalog.install(bundle.store, pack_ids=[pack.pack_id])
        with pytest.raises(
            StoreError, match="executable knowledge bindings require a review_ready pack",
        ):
            bundle.store.install_knowledge_pack(
                pack_id=pack.pack_id,
                pack_schema_version=pack.schema_version,
                content_hash=pack.content_hash,
                installed_at="2026-07-21T00:00:00+00:00",
                inventory=pack.inventory(),
                rules=pack.rules,
                bindings=pack.bindings,
                coverage=pack.coverage,
            )

    reviewed_catalog = KnowledgeCatalog([
        _reviewed_pack(pack) if pack.bindings and not pack.review_ready else pack
        for pack in catalog.packs
    ])
    reviewed_catalog.install(bundle.store)
    expected_counts = {
        pack.pack_id: (
            len(pack.rules), len(pack.bindings), len(pack.coverage.target_inventory),
        )
        for pack in reviewed_catalog.packs
    }
    inventory = bundle.store.installed_knowledge_packs()
    assert {item["pack_id"] for item in inventory} == set(expected_counts)
    for item in inventory:
        assert item["contains_document_body"] is False
        assert item["pack_schema_version"] == "2.0"
        assert item["installed_at"].endswith("+00:00")
        rule_count, binding_count, _target_count = expected_counts[item["pack_id"]]
        assert len(item["rule_ids"]) == rule_count
        assert len(item["binding_ids"]) == binding_count
    assert len(bundle.store.knowledge_bindings()) == sum(
        item[1] for item in expected_counts.values()
    )
    coverage = bundle.store.knowledge_coverage()
    assert {item.pack_id for item in coverage} == set(expected_counts)
    assert all(item.complete for item in coverage)
    assert all(
        not item.target_inventory or item.target_registry_version
        == "hlsgraph.knowledge_supported_targets.v1"
        for item in coverage
    )
    assert {
        item.pack_id: len(item.target_inventory) for item in coverage
    } == {
        pack_id: counts[2] for pack_id, counts in expected_counts.items()
    }
    assert {item["status"] for item in reviewed_catalog.sync(bundle.store)} == {
        "unchanged"
    }
    assert bundle.store.search_knowledge_rules("WNS timing")[0]["rule"].document_id == "amd.ug835"
    with sqlite3.connect(bundle.store.path) as connection:
        tables = {row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
    assert "knowledge_bindings" in tables
    assert "knowledge_coverage" in tables
    assert not {"entities", "relations"} & {
        "knowledge_bindings", "knowledge_coverage"
    }


def test_builtin_coverage_inventory_matches_the_supported_public_surface() -> None:
    packs = {pack.pack_id: pack for pack in KnowledgeCatalog.builtin().packs}
    amd = packs["hlsgraph.amd.public_guidance.2024_2"]
    axi = packs["hlsgraph.axi.public_guidance.v1"]
    open_ir = packs["hlsgraph.open_ir.public_guidance.2026_07_21"]

    assert {item.document_id for item in amd.documents} == {
        "amd.ug1399", "amd.ug903", "amd.ug835", "amd.ug906", "amd.ug907",
    }
    assert {
        item.target for item in amd.coverage.target_inventory
        if item.target_kind == "directive_kind"
    } == {
        "DATAFLOW", "PIPELINE", "UNROLL", "ARRAY_PARTITION", "INTERFACE",
        "STREAM", "DEPENDENCE", "LOOP_TRIPCOUNT", "INLINE",
    }
    assert {
        item.target for item in amd.coverage.target_inventory
        if item.target_kind == "gate_kind"
    } == {"correctness", "resource_fits", "post_route_timing"}
    for binding in amd.bindings:
        assert binding.required_context["vendor"] == "amd"
        assert binding.required_context["tool_version"] == "2024.2"
        assert "stage" in binding.required_context
    directive_bindings = [
        item for item in amd.bindings if item.target_kind == "directive_kind"
    ]
    assert directive_bindings
    for binding in directive_bindings:
        required = binding.required_context
        assert required["directive_instance_id"] == {"required": True}
        assert required["scope_id"] == {"required": True}
        assert required["scope_resolution"] == {
            "one_of": ["source_ast", "external_exact"],
        }
        assert "scope_kind" in required
    required_roles = {
        "DATAFLOW": {"function_id", "loop_id"},
        "PIPELINE": {"function_id", "loop_id"},
        "UNROLL": {"loop_id"},
        "ARRAY_PARTITION": {"variable_id"},
        "INTERFACE": {"port_id"},
        "STREAM": {"variable_id"},
        "LOOP_TRIPCOUNT": {"loop_id"},
        "INLINE": {"function_id"},
    }
    for binding in directive_bindings:
        present_roles = {
            key for key, value in binding.required_context.items()
            if key in required_roles[binding.target]
            and value == {"required": True}
        }
        assert present_roles
        if binding.target in {"ARRAY_PARTITION", "INTERFACE", "STREAM"}:
            assert binding.required_context["directive_operand_linked"] == (
                "derived_from_current_directive_operand_link_v1"
            )
            assert binding.required_context["directive_operand_identity"] == {
                "required": True,
            }
            if binding.target == "INTERFACE":
                assert binding.required_context["port_owner_id"] == {
                    "required": True,
                }
                assert binding.required_context["configured_component_id"] == {
                    "required": True,
                }
                assert binding.required_context["port_ownership_qualified"] == (
                    "derived_from_unique_current_component_port_v1"
                )
                assert binding.required_context["port_ownership_identity"] == {
                    "required": True,
                }
        else:
            assert "directive_operand_linked" not in binding.required_context
    dependence_bindings = [
        item for item in directive_bindings if item.target == "DEPENDENCE"
    ]
    assert dependence_bindings == []
    dependence_rule = next(
        item for item in amd.rules
        if item.rule_id == "directive.dependence_is_user_assertion"
    )
    assert dependence_rule.condition == {"directive_kind": "DEPENDENCE"}
    dependence_target = next(
        item for item in amd.coverage.target_inventory
        if item.target_kind == "directive_kind" and item.target == "DEPENDENCE"
    )
    assert dependence_target.status == TargetCoverageStatus.NO_NORMATIVE
    assert not dependence_target.binding_ids
    assert "mutually exclusive" in dependence_target.rationale
    directive_predicates = [
        item for item in amd.bindings
        if item.target_kind == "predicate" and item.target.startswith("directive.")
    ]
    assert {item.target for item in directive_predicates} == {
        "directive.requested", "directive.declared_selected",
        "directive.tool_status", "directive.reported_requested",
        "directive.tool_effective", "directive.achieved",
    }
    assert all(
        item.required_context["directive_instance_id"] == {"required": True}
        and item.required_context["scope_id"] == {"required": True}
        and item.required_context["scope_resolution"] == {
            "one_of": ["source_ast", "external_exact"],
        }
        and item.required_context["scope_kind"] == {"required": True}
        and item.required_context["requested_directive_present"] is True
        for item in directive_predicates
    )

    assert {item.document_id for item in axi.documents} == {
        "amd.ug1399", "arm.ihi0022", "arm.ihi0051",
    }
    assert {item.document_id for item in axi.rules} == {"amd.ug1399"}
    arm_coverage = {
        item.document_id: item for item in axi.coverage.entries
        if item.document_id.startswith("arm.")
    }
    assert set(arm_coverage) == {"arm.ihi0022", "arm.ihi0051"}
    assert all(
        item.status == CoverageStatus.CITATION_ONLY
        and not item.rule_ids and not item.binding_ids and item.rationale
        for item in arm_coverage.values()
    )
    assert not any(
        item.knowledge_rule_id.startswith("arm.") for item in axi.bindings
    )
    assert [(item.target_kind, item.target) for item in axi.coverage.target_inventory
            if item.status == TargetCoverageStatus.BOUND] == [
        ("directive_kind", "INTERFACE"),
    ]
    axi_interface = next(item for item in axi.bindings
                         if item.target_kind == "directive_kind"
                         and item.target == "INTERFACE")
    assert axi_interface.required_context["directive_instance_id"] == {"required": True}
    assert axi_interface.required_context["scope_id"] == {"required": True}
    assert axi_interface.required_context["scope_resolution"] == {
        "one_of": ["source_ast", "external_exact"],
    }
    assert axi_interface.required_context["scope_kind"] == "hls.port"
    assert axi_interface.required_context["port_id"] == {"required": True}
    assert axi_interface.required_context["directive_operand_linked"] == (
        "derived_from_current_directive_operand_link_v1"
    )
    assert axi_interface.required_context["directive_operand_identity"] == {
        "required": True,
    }
    assert axi_interface.required_context["port_owner_id"] == {"required": True}
    assert axi_interface.required_context["configured_component_id"] == {
        "required": True,
    }
    assert axi_interface.required_context["port_ownership_qualified"] == (
        "derived_from_unique_current_component_port_v1"
    )
    assert axi_interface.required_context["port_ownership_identity"] == {
        "required": True,
    }
    axi_rule = next(
        item for item in axi.rules
        if item.id.endswith(":axi.interface_mode_is_scoped_request")
    )
    assert axi_rule.condition == {
        "directive_kind": "INTERFACE",
        "interface_mode": {"one_of": ["m_axi", "s_axilite", "axis"]},
        "port_ownership_qualified": (
            "derived_from_unique_current_component_port_v1"
        ),
    }
    axi_no_normative = {
        item.target: item for item in axi.coverage.target_inventory
        if item.status == TargetCoverageStatus.NO_NORMATIVE
    }
    assert set(axi_no_normative) == {"hls.port", "hls.stream", "hls.streams_to"}
    assert all(
        item.rationale and "specification revision" in item.rationale
        and not item.binding_ids
        for item in axi_no_normative.values()
    )
    assert "direction" in axi_no_normative["hls.port"].rationale
    assert "transmitter" in axi_no_normative["hls.stream"].rationale
    assert "internal design evidence" in axi_no_normative["hls.streams_to"].rationale

    assert {item.document_id for item in open_ir.documents} == {
        "llvm.mlir.langref", "llvm.mlir.builtin", "llvm.ir.langref",
        "llvm.ir.debug", "circt.handshake", "scalehls.paper",
        "dynamatic.mlir_primer",
    }
    llvm_revision = "git-429c88d37f1f02e68ebc1fc7b0da4511ce6407e3"
    assert {
        item.document_version for item in open_ir.documents
        if item.document_id.startswith("llvm.")
    } == {llvm_revision}
    assert open_ir.bindings == []
    assert "supported_language_spec_contracts" not in open_ir.metadata
    assert "no executable language-spec" in open_ir.metadata["coverage_boundary"]
    assert "not a language-spec attestation" in open_ir.metadata["truth_boundary"]
    llvm_rules = [rule for rule in open_ir.rules
                  if rule.document_id.startswith("llvm.")]
    assert llvm_rules
    assert all(
        rule.effect.get("specification_revision") == rule.document_version
        and rule.effect.get("artifact_revision_required_for_binding") is True
        and rule.effect.get(
            "specification_revision_does_not_imply_artifact_revision"
        ) is True
        for rule in llvm_rules
    )
    llvm_call_rule = next(
        rule for rule in llvm_rules
        if rule.rule_id == "llvm.calls_are_low_level_ir_evidence"
    )
    assert llvm_call_rule.effect["authority"] == "compiler_decision"
    assert llvm_call_rule.effect["hardware_topology"] is False
    assert llvm_call_rule.effect["hardware_instance"] is False
    assert llvm_call_rule.effect["callee_resolution_must_be_explicit"] is True
    assert all(
        item.status == TargetCoverageStatus.NO_NORMATIVE
        and not item.binding_ids and item.rationale
        for item in open_ir.coverage.target_inventory
    )
    assert all(not entry.binding_ids for entry in open_ir.coverage.entries)
    handshake_rule = next(
        rule for rule in open_ir.rules
        if rule.document_id == "circt.handshake"
    )
    assert handshake_rule.effect["native_ir_evidence_only"] is True
    assert handshake_rule.effect["hardware_topology"] is False
    assert not any(
        rule.document_id in {"scalehls.paper", "dynamatic.mlir_primer"}
        for rule in open_ir.rules
    )
    adapter_citations = {
        entry.document_id: entry for entry in open_ir.coverage.entries
        if entry.document_id in {"scalehls.paper", "dynamatic.mlir_primer"}
    }
    assert set(adapter_citations) == {"scalehls.paper", "dynamatic.mlir_primer"}
    assert all(
        entry.status.value == "citation_only"
        and not entry.rule_ids and entry.rationale
        for entry in adapter_citations.values()
    )

    amd_bindings = {(item.target_kind, item.target): item for item in amd.bindings}
    bindings_for = lambda target: [
        item for item in amd.bindings
        if item.target_kind == "predicate" and item.target == target
    ]
    assert bindings_for("qor.target_ii") == []
    assert bindings_for("qor.achieved_ii") == []
    assert not any(
        item.knowledge_rule_id.endswith(":qor.csynth_is_estimate")
        for item in amd.bindings
    )
    assert not any(
        item.rule_id == "qor.csynth_is_estimate" for item in amd.rules
    )
    for target in {
        "qor.latency_best_cycles", "qor.latency_worst_cycles",
        "qor.interval_min_cycles", "qor.interval_max_cycles",
        "qor.latency_cycles", "qor.iteration_latency_cycles",
    }:
        assert {
            item.knowledge_rule_id.rsplit(":", 1)[-1]
            for item in bindings_for(target)
        } == {"qor.latency_and_ii_are_distinct"}
    critical_path = amd_bindings[("predicate", "timing.critical_path_delay_ns")]
    assert critical_path.required_context["stage"] == "post_route"
    assert critical_path.knowledge_rule_id.endswith(
        ":timing.post_route_observations_are_routed_evidence"
    )
    post_route_report = amd_bindings[
        ("artifact_kind", "amd.vivado.post_route_timing")
    ]
    assert post_route_report.knowledge_rule_id.endswith(
        ":timing.post_route_observations_are_routed_evidence"
    )
    assert not any(
        item.knowledge_rule_id.endswith(
            ":timing.post_route_signoff_requires_routed_design"
        )
        for item in amd.bindings
        if item.target_kind != "gate_kind"
    )

    typed_observation_targets = {
        (item.target_kind, item.target) for item in amd.bindings
        if item.target_kind == "predicate"
        and "observation_evidence_qualified" in item.required_context
    }
    assert typed_observation_targets
    assert all(
        item.required_context.get("snapshot_association") == "verified"
        and item.required_context.get("observation_evidence_qualified")
        == "derived_from_typed_observation_evidence_v1"
        and all(item.required_context.get(key) == {"required": True}
                for key in (
                    "observation_instance_id", "observation_artifact_identity",
                    "observation_run_identity",
                ))
        for item in amd.bindings
        if (item.target_kind, item.target) in typed_observation_targets
    )

    no_normative = {
        (item.target_kind, item.target)
        for item in amd.coverage.target_inventory
        if item.status == TargetCoverageStatus.NO_NORMATIVE
    }
    assert {
        ("predicate", "clock.requested_period_ns"),
        ("predicate", "qor.trip_count"),
        ("predicate", "qor.pipeline_depth"),
        ("predicate", "physical.slr_crossings"),
        ("predicate", "physical.drc_errors"),
        ("predicate", "physical.cdc_critical"),
        ("predicate", "resource.available_lut"),
        ("predicate", "resource.available_ff"),
        ("predicate", "resource.available_dsp"),
        ("predicate", "resource.available_bram_18k"),
        ("predicate", "resource.available_uram"),
        ("artifact_kind", "amd.vitis.csim_result"),
        ("artifact_kind", "amd.vitis.cosim_rpt"),
        ("artifact_kind", "amd.vitis.cosim_report"),
        ("artifact_kind", "amd.vitis.dataflow_profile"),
        ("artifact_kind", "amd.vitis.directive_status"),
        ("artifact_kind", "amd.vivado.physical_summary"),
        ("artifact_kind", "amd.vivado.qor_summary"),
        ("artifact_kind", "amd.vivado.routed_checkpoint"),
        ("diagnostic_code", "gate.resource_capacity_unknown"),
        ("diagnostic_code", "gate.resource_capacity_incomplete"),
    }.issubset(no_normative)
    assert not {
        (item.target_kind, item.target) for item in amd.bindings
    }.intersection(no_normative)
    waiver_targets = {
        item.target: item for item in amd.coverage.target_inventory
        if item.target in {"physical.drc_errors", "physical.cdc_critical"}
    }
    assert set(waiver_targets) == {"physical.drc_errors", "physical.cdc_critical"}
    assert all(
        item.status == TargetCoverageStatus.NO_NORMATIVE
        and item.rationale and "waiver-set identity" in item.rationale
        for item in waiver_targets.values()
    )
    waiver_coverage = next(
        item for item in amd.coverage.entries
        if item.document_id == "amd.ug906"
        and item.section == "Reporting the Waivers"
    )
    assert waiver_coverage.status == CoverageStatus.CITATION_ONLY
    assert not waiver_coverage.rule_ids
    assert not waiver_coverage.binding_ids
    assert waiver_coverage.rationale

    # A container kind may bind only when its condition is container-level and
    # the current bytes close to a fresh run's declared output. Field-specific
    # conditions stay on emitted predicates.
    artifact_conditions = {
        "amd.vitis.schedule_json": "schedule_artifact_present",
        "constraint.xdc": "constraint_artifact_present",
        "amd.vivado.timing_summary": "timing_summary_present",
        "amd.vivado.post_route_timing": "post_route_timing_result_present",
        "amd.vivado.utilization": "utilization_report_present",
        "amd.vivado.post_route_utilization": "utilization_report_present",
    }
    artifact_bindings = [
        item for item in amd.bindings if item.target_kind == "artifact_kind"
    ]
    assert {item.target for item in artifact_bindings} == set(artifact_conditions)
    rules_by_id = {item.id: item for item in amd.rules}
    for binding in artifact_bindings:
        assert rules_by_id[binding.knowledge_rule_id].condition == {
            artifact_conditions[binding.target]: True,
        }

    xdc = amd_bindings[("artifact_kind", "constraint.xdc")]
    assert xdc.required_context["snapshot_association"] == "verified"
    assert xdc.required_context["artifact_sha256"] == {"required": True}
    assert xdc.required_context["constraint_hash"] == {"required": True}
    assert xdc.required_context["constraint_input_evidence_qualified"] == (
        "derived_from_unique_live_snapshot_input_v1"
    )
    assert xdc.required_context["constraint_artifact_identity"] == {
        "required": True,
    }
    xdc_rule = next(
        item for item in amd.rules
        if item.id.endswith(":xdc.constraints_are_design_inputs")
    )
    assert xdc_rule.condition == {"constraint_artifact_present": True}
    source_directives = {
        "DATAFLOW", "PIPELINE", "UNROLL", "ARRAY_PARTITION", "INTERFACE",
        "STREAM", "LOOP_TRIPCOUNT", "INLINE",
    }
    for pack in (amd, axi):
        for binding in pack.bindings:
            if (binding.target_kind != "directive_kind"
                    or binding.target not in source_directives):
                continue
            assert binding.required_context[
                "directive_source_declaration_qualified"
            ] == "derived_from_current_directive_source_declaration_v1"
            assert binding.required_context["directive_source_identity"] == {
                "required": True,
            }
    for binding in artifact_bindings:
        if binding.target == "constraint.xdc":
            continue
        assert binding.required_context["snapshot_association"] == "verified"
        assert binding.required_context["tool_artifact_evidence_qualified"] == (
            "derived_from_declared_live_tool_output_v1"
        )
        assert binding.required_context["tool_artifact_identity"] == {
            "required": True,
        }
        assert binding.required_context["tool_artifact_run_identity"] == {
            "required": True,
        }
    tool_directive_targets = {
        "directive.tool_status", "directive.reported_requested",
        "directive.tool_effective", "directive.achieved",
    }
    for binding in amd.bindings:
        if binding.target_kind != "predicate" or binding.target not in tool_directive_targets:
            continue
        assert binding.required_context["snapshot_association"] == "verified"
        assert binding.required_context["observation_evidence_qualified"] == (
            "derived_from_typed_observation_evidence_v1"
        )
        assert binding.required_context["observation_artifact_kind"] == (
            "amd.vitis.directive_status"
        )
    gate_bindings = [item for item in amd.bindings
                     if item.target_kind == "gate_kind"]
    assert gate_bindings
    assert all(
        item.required_context["snapshot_association"] == "verified"
        and item.required_context["gate_evidence_qualified"]
        == "derived_from_typed_evidence_v1"
        for item in gate_bindings
    )
    assert {item.target for item in gate_bindings} == {"correctness"}
    assert all({
        "verification_observation_identity", "verification_report_identity",
    }.issubset(item.required_context) for item in gate_bindings)
    for target in {"resource_fits", "post_route_timing"}:
        coverage = next(
            item for item in amd.coverage.target_inventory
            if item.target_kind == "gate_kind" and item.target == target
        )
        assert coverage.status == TargetCoverageStatus.NO_NORMATIVE
        assert coverage.rationale and not coverage.binding_ids

    for pack in packs.values():
        assert pack.coverage is not None and pack.coverage.complete
        assert pack.coverage.review_status == "unreviewed"
        assert all(entry.status.value != "deferred" for entry in pack.coverage.entries)
        assert all(
            not entry.rule_ids and not entry.binding_ids
            for entry in pack.coverage.entries
            if entry.status != CoverageStatus.RULE
        )
        assert Counter(
            rule_id
            for entry in pack.coverage.entries
            for rule_id in entry.rule_ids
        ) == Counter({item.id: 1 for item in pack.rules})
        assert Counter(
            binding_id
            for entry in pack.coverage.entries
            for binding_id in entry.binding_ids
        ) == Counter({item.id: 1 for item in pack.bindings})
        bindings = {item.id: item for item in pack.bindings}
        covered: set[str] = set()
        for target in pack.coverage.target_inventory:
            if target.status == TargetCoverageStatus.BOUND:
                assert target.binding_ids
            else:
                assert target.rationale and not target.binding_ids
            for binding_id in target.binding_ids:
                binding = bindings[binding_id]
                assert (binding.target_kind, binding.target) == (
                    target.target_kind, target.target,
                )
                covered.add(binding_id)
        assert covered == set(bindings)


def test_pack_rejects_target_inventory_binding_for_a_different_target() -> None:
    binding = KnowledgeBinding(
        knowledge_rule_id="test.doc:1:test.rule",
        target_kind="predicate", target="test.value",
        required_context={"stage": "ast"},
        producer="test.binding", producer_version="1",
        metadata={"dynamic_scope": "static"},
    )
    with pytest.raises(KnowledgePackError, match="different target"):
        load_pack({
            "schema_version": "2.0",
            "pack_id": "test.coverage.pack",
            "title": "test coverage",
            "license": "Apache-2.0",
            "documents": [{
                "document_id": "test.doc", "document_version": "1",
                "title": "Test", "official_url": "https://example.com/test",
                "publisher": "Example",
            }],
            "rules": [{
                "document_id": "test.doc", "document_version": "1",
                "section": "Section", "rule_id": "test.rule", "title": "Rule",
                "applicability": {"stage": "ast"}, "condition": {}, "effect": {},
                "citation_url": "https://example.com/test#section",
                "summary": "A short project-authored paraphrase.",
            }],
            "bindings": [json_ready(binding)],
            "coverage": {
                "pack_id": "test.coverage.pack",
                "coverage_scope": "test.supported_surface",
                "target_registry_version": (
                    "hlsgraph.knowledge_supported_targets.v1"
                ),
                "entries": [{
                    "document_id": "test.doc", "document_version": "1",
                    "section": "Section", "status": "rule",
                    "rule_ids": ["test.doc:1:test.rule"],
                    "binding_ids": [binding.id],
                }],
                "target_inventory": [{
                    "target_kind": "predicate", "target": "test.other",
                    "status": "bound", "binding_ids": [binding.id],
                }],
            },
        })


def test_v1_pack_remains_lexical_only() -> None:
    pack = load_pack({
        "schema_version": "1.0",
        "pack_id": "test.legacy.pack",
        "title": "legacy",
        "license": "Apache-2.0",
        "documents": [{
            "document_id": "test.doc", "document_version": "1",
            "title": "Test", "official_url": "https://example.com/test",
            "publisher": "Example",
        }],
        "rules": [{
            "document_id": "test.doc", "document_version": "1",
            "section": "Section", "rule_id": "test.rule", "title": "Rule",
            "applicability": {}, "condition": {}, "effect": {},
            "citation_url": "https://example.com/test#section",
            "summary": "A short project-authored paraphrase.",
        }],
    })
    assert pack.bindings == [] and pack.coverage is None
    assert pack_migration_plan(pack)[0]["to_version"] == "2.0"
    migrated = migrate_pack(pack)
    assert migrated.schema_version == "2.0"
    assert [item.id for item in migrated.rules] == [item.id for item in pack.rules]
    assert migrated.bindings == []
    assert migrated.coverage is not None
    assert migrated.coverage.review_status == "unreviewed"
    assert migrated.coverage.target_inventory == []
    assert {
        rule_id
        for entry in migrated.coverage.entries
        for rule_id in entry.rule_ids
    } == {item.id for item in pack.rules}


def _condition_pack(binding_context: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": "2.0",
        "pack_id": "test.condition.pack",
        "title": "condition closure",
        "license": "Apache-2.0",
        "documents": [{
            "document_id": "test.condition.doc", "document_version": "1",
            "title": "Condition test", "official_url": "https://example.com/condition",
            "publisher": "Example",
        }],
        "rules": [{
            "document_id": "test.condition.doc", "document_version": "1",
            "section": "Condition", "rule_id": "test.rule", "title": "Rule",
            "applicability": {"stage": "csim"},
            "condition": {"csim_result_present": True},
            "effect": {"scope": "workload"},
            "citation_url": "https://example.com/condition#rule",
            "summary": "A synthetic rule used to test premise closure.",
        }],
        "bindings": [{
            "knowledge_rule_id": "test.condition.doc:1:test.rule",
            "target_kind": "predicate", "target": "csim.exit_code",
            "required_context": binding_context,
            "producer": "test.binding", "producer_version": "1",
            "metadata": {},
        }],
    }


def test_pack_load_rejects_unproved_or_weakened_rule_condition() -> None:
    with pytest.raises(KnowledgePackError, match="does not entail its rule condition"):
        load_pack(_condition_pack({"stage": "csim"}))

    weakened = _condition_pack({
        "stage": "csim", "csim_result_present": {"required": True},
    })
    weakened["rules"][0]["condition"] = {"csim_result_present": True}
    with pytest.raises(KnowledgePackError, match="weakened or contradicted"):
        load_pack(weakened)


def test_builtin_bindings_have_audited_condition_entailment() -> None:
    for pack in KnowledgeCatalog.builtin().packs:
        assert pack.metadata["binding_condition_contract"] == (
            "hlsgraph.binding_condition_entailment.v1"
        )
        rules = {rule.id: rule for rule in pack.rules}
        for binding in pack.bindings:
            entailed, errors = binding_entails_rule_condition(
                rules[binding.knowledge_rule_id], binding,
            )
            assert entailed, (binding.id, errors)


def test_v02_to_v03_migration_is_additive_and_keeps_graph_marker(tmp_path: Path) -> None:
    path = tmp_path / "graph.db"
    store = LedgerStore(path)
    store.initialize()
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE schema_info SET value='0.2.0' WHERE key='schema_version'"
        )
        connection.execute(
            "INSERT INTO projects(project_id,manifest_hash,manifest_json) VALUES(?,?,?)",
            ("test.project", "0" * 64, "{}"),
        )
        connection.execute(
            "INSERT INTO snapshots(id,project_id,created_at,payload_json) VALUES(?,?,?,?)",
            ("snapshot_old", "test.project", "2026-01-01T00:00:00+00:00", "{}"),
        )
        connection.execute(
            "INSERT INTO graph_views(snapshot_id,schema_version,metadata_json) VALUES(?,?,?)",
            ("snapshot_old", "0.2.0", "{}"),
        )
    original = path.read_bytes()
    plan = store.migration_plan()
    assert [(item["from_version"], item["to_version"]) for item in plan] == [
        ("0.2.0", "0.3.0")
    ]
    store.migrate()
    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT value FROM schema_info WHERE key='schema_version'"
        ).fetchone()[0] == SCHEMA_VERSION
        assert connection.execute(
            "SELECT schema_version FROM graph_views WHERE snapshot_id='snapshot_old'"
        ).fetchone()[0] == "0.2.0"
        assert connection.execute(
            "SELECT COUNT(*) FROM knowledge_bindings"
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM index_commit_receipts"
        ).fetchone()[0] == 0
    assert path.read_bytes() != original


def test_v02_graph_hash_is_unchanged_by_v03_migration(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    snapshot = bundle.snapshot()
    entity = Entity("hls.kernel", "dut", snapshot.id, stage="ast")
    graph = CanonicalGraph(snapshot.id)
    graph.add_entity(entity)
    bundle.store.save_graph(graph)
    legacy_graph = CanonicalGraph(snapshot.id, schema_version="0.2.0")
    legacy_graph.add_entity(entity)
    expected_hash = legacy_graph.graph_hash
    with sqlite3.connect(bundle.store.path) as connection:
        connection.execute(
            "UPDATE schema_info SET value='0.2.0' WHERE key='schema_version'"
        )
        connection.execute(
            "UPDATE graph_views SET schema_version='0.2.0' WHERE snapshot_id=?",
            (snapshot.id,),
        )
    bundle.store.migrate()
    loaded = bundle.store.load_graph(snapshot.id)
    assert loaded.schema_version == "0.2.0"
    assert loaded.graph_hash == expected_hash


def test_private_sidecar_default_is_metadata_only_and_detects_stale_source(
    tmp_path: Path,
) -> None:
    sentinel = "PRIVATE-KNOWLEDGE-SENTINEL"
    bundle = _bundle(tmp_path)
    document = tmp_path / "guide.md"
    document.write_text(f"# Pipeline\n{sentinel} achieved II evidence.\n", encoding="utf-8")
    metadata = index_local_document(
        document, document_id="test.local.guide", document_version="1",
    )
    sidecar = LocalKnowledgeSidecar(tmp_path)
    manifest = sidecar.build(bundle.manifest.project_id, [metadata])
    assert isinstance(manifest, LocalKnowledgeIndexManifest)
    assert manifest.content_embedded_in_canonical is False
    assert sentinel.encode() not in bundle.store.path.read_bytes()
    assert sentinel not in sidecar.manifest_path.read_text(encoding="utf-8")
    assert sentinel.encode() in sidecar.database_path.read_bytes()
    rebuilt = sidecar.sync(bundle.manifest.project_id, [metadata])
    assert rebuilt.id == manifest.id
    assert rebuilt.index_sha256 == manifest.index_sha256
    metadata_hit = sidecar.search(sentinel)[0]
    assert metadata_hit.excerpt is None
    assert sidecar.search("\\") == []
    assert sentinel in sidecar.search(sentinel, include_text=True)[0].excerpt
    document.write_text("changed\n", encoding="utf-8")
    with pytest.raises(KnowledgePackError, match="changed"):
        sidecar.search(sentinel, include_text=True)


def test_sidecar_manifest_and_database_tampering_fail_closed(tmp_path: Path) -> None:
    _bundle(tmp_path)
    document = tmp_path / "guide.txt"
    document.write_text("pipeline schedule evidence", encoding="utf-8")
    metadata = index_local_document(
        document, document_id="test.local.tamper", document_version="1",
    )
    sidecar = LocalKnowledgeSidecar(tmp_path)
    manifest = sidecar.build("test.knowledge.v03", [metadata])
    value = json.loads(sidecar.manifest_path.read_text(encoding="utf-8"))
    value["index_sha256"] = "0" * 64
    sidecar.manifest_path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(KnowledgePackError, match="local_sidecar_manifest.contract_invalid"):
        sidecar.search("pipeline")
    sidecar.manifest_path.write_text(
        json.dumps(json_ready(manifest), sort_keys=True), encoding="utf-8",
    )
    with sidecar.database_path.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(KnowledgePackError, match="hash"):
        sidecar.search("pipeline")


def test_sidecar_search_is_bound_to_verified_bytes_across_swap_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    trusted_root = tmp_path / "trusted"
    trusted_root.mkdir()
    trusted_document = trusted_root / "guide.md"
    trusted_document.write_text(
        "# Trusted\nverified pipeline guidance\n", encoding="utf-8",
    )
    trusted_metadata = index_local_document(
        trusted_document,
        document_id="test.local.swap_trusted",
        document_version="1",
    )
    sidecar = LocalKnowledgeSidecar(trusted_root)
    sidecar.build("test.knowledge.swap_trusted", [trusted_metadata])
    trusted_database = sidecar.database_path.read_bytes()

    attacker_root = tmp_path / "attacker"
    attacker_root.mkdir()
    attacker_document = attacker_root / "guide.md"
    attacker_document.write_text(
        "# Attacker\nSWAP-ONLY-SECRET\n", encoding="utf-8",
    )
    attacker_metadata = index_local_document(
        attacker_document,
        document_id="test.local.swap_attacker",
        document_version="1",
    )
    attacker_sidecar = LocalKnowledgeSidecar(attacker_root)
    attacker_sidecar.build("test.knowledge.swap_attacker", [attacker_metadata])
    attacker_database = attacker_sidecar.database_path.read_bytes()

    original_snapshot = sidecar._verified_database_snapshot
    real_connect = sqlite3.connect
    state = {"malicious_path_live": False, "restored": False}

    def swap_after_verification():
        manifest, verified_bytes = original_snapshot()
        assert verified_bytes == trusted_database
        # Simulate replacement in the exact former verify/open window.
        sidecar.database_path.write_bytes(attacker_database)
        state["malicious_path_live"] = True
        return manifest, verified_bytes

    def restore_before_query(database, *args, **kwargs):
        # The query may open only in-memory storage or a private staging file
        # containing the already-verified bytes (Python 3.10 fallback).  It
        # must never reopen the mutable sidecar database path.
        assert str(sidecar.database_path) not in str(database)
        if database == ":memory:":
            assert state["malicious_path_live"] is True
            sidecar.database_path.write_bytes(trusted_database)
            state["malicious_path_live"] = False
            state["restored"] = True
        else:
            assert str(database).startswith((
                "file:/proc/self/fd/", "file:/dev/fd/", "file:///",
            ))
        return real_connect(database, *args, **kwargs)

    monkeypatch.setattr(sidecar, "_verified_database_snapshot", swap_after_verification)
    monkeypatch.setattr(sidecar_module.sqlite3, "connect", restore_before_query)

    assert sidecar.search("SWAP-ONLY-SECRET") == []
    assert state == {"malicious_path_live": False, "restored": True}
    assert sidecar.database_path.read_bytes() == trusted_database


def test_sidecar_search_uses_verified_snapshot_without_sqlite_deserialize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = tmp_path / "guide.md"
    document.write_text("# Pipeline\nverified evidence\n", encoding="utf-8")
    metadata = index_local_document(
        document,
        document_id="test.local.no_deserialize",
        document_version="1",
    )
    sidecar = LocalKnowledgeSidecar(tmp_path)
    sidecar.build("test.knowledge.no_deserialize", [metadata])

    monkeypatch.setattr(
        LocalKnowledgeSidecar, "_deserialize_into", staticmethod(
            lambda _connection, _database_bytes: False
        ),
    )
    hits = sidecar.search("pipeline")
    assert hits and hits[0].document_id == "test.local.no_deserialize"


@pytest.mark.parametrize("tamper", ["bytes", "identity"])
def test_sidecar_fallback_revalidates_staged_snapshot_after_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper: str,
) -> None:
    document = tmp_path / "guide.md"
    document.write_text("# Pipeline\nverified evidence\n", encoding="utf-8")
    metadata = index_local_document(
        document,
        document_id="test.local.staging_revalidation",
        document_version="1",
    )
    sidecar = LocalKnowledgeSidecar(tmp_path)
    sidecar.build("test.knowledge.staging_revalidation", [metadata])

    monkeypatch.setattr(
        LocalKnowledgeSidecar, "_deserialize_into", staticmethod(
            lambda _connection, _database_bytes: False
        ),
    )
    real_read = sidecar_module._read_stable_local_file
    staged_reads = 0

    def changed_after_backup(path, *, max_bytes):
        nonlocal staged_reads
        data, info = real_read(path, max_bytes=max_bytes)
        if Path(path).name != "snapshot.sqlite3":
            return data, info
        staged_reads += 1
        if staged_reads != 2:
            return data, info
        if tamper == "bytes":
            return bytes([data[0] ^ 0xFF]) + data[1:], info
        return data, SimpleNamespace(
            st_dev=info.st_dev,
            st_ino=info.st_ino + 1,
            st_mode=info.st_mode,
        )

    monkeypatch.setattr(
        sidecar_module, "_read_stable_local_file", changed_after_backup,
    )
    with pytest.raises(KnowledgePackError, match="changed during open"):
        sidecar.search("pipeline")
    assert staged_reads == 2


def test_sidecar_fallback_rejects_staged_reparse_point(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = tmp_path / "guide.md"
    document.write_text("# Pipeline\nverified evidence\n", encoding="utf-8")
    metadata = index_local_document(
        document,
        document_id="test.local.staging_reparse",
        document_version="1",
    )
    sidecar = LocalKnowledgeSidecar(tmp_path)
    sidecar.build("test.knowledge.staging_reparse", [metadata])

    monkeypatch.setattr(
        LocalKnowledgeSidecar, "_deserialize_into", staticmethod(
            lambda _connection, _database_bytes: False
        ),
    )
    real_is_link = sidecar_module._is_link_or_reparse

    def staged_is_link(path):
        return Path(path).name == "snapshot.sqlite3" or real_is_link(path)

    monkeypatch.setattr(sidecar_module, "_is_link_or_reparse", staged_is_link)
    with pytest.raises(KnowledgePackError, match="links/reparse"):
        sidecar.search("pipeline")


def test_sidecar_embedder_must_be_local_and_vectors_stay_private(tmp_path: Path) -> None:
    _bundle(tmp_path)
    document = tmp_path / "guide.md"
    document.write_text("# II\nPipeline evidence", encoding="utf-8")
    metadata = index_local_document(
        document, document_id="test.local.embed", document_version="1",
    )

    class LocalEmbedder:
        name = "test.local"
        version = "1"
        fingerprint = hashlib.sha256(b"test.local.v1").hexdigest()

        @staticmethod
        def capabilities():
            return {"protocol_version": "hlsgraph.embedder.v1",
                    "local_only": True, "network_access": False}

        @staticmethod
        def embed(texts):
            return [[float(len(item)), 1.0] for item in texts]

    sidecar = LocalKnowledgeSidecar(tmp_path)
    manifest = sidecar.build("test.knowledge.v03", [metadata], embedder=LocalEmbedder())
    assert manifest.embedder_id == "test.local"
    with sqlite3.connect(sidecar.database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM vectors").fetchone()[0] == 1
    assert LocalEmbedder.fingerprint.encode() not in (
        tmp_path / ".hlsgraph/graph.db"
    ).read_bytes()

    class RemoteEmbedder(LocalEmbedder):
        @staticmethod
        def capabilities():
            return {"protocol_version": "hlsgraph.embedder.v1",
                    "local_only": False, "network_access": True}

    with pytest.raises(KnowledgePackError, match="local-only"):
        sidecar.build("test.knowledge.v03", [metadata], embedder=RemoteEmbedder())


def test_embedder_stdio_and_exception_body_are_suppressed(
    tmp_path: Path, capfd,
) -> None:
    _bundle(tmp_path)
    sentinel = "PRIVATE_EMBEDDER_SENTINEL_7cce0f"
    document = tmp_path / "private-guide.md"
    document.write_text(f"# II\n{sentinel}\n", encoding="utf-8")
    metadata = index_local_document(
        document, document_id="test.local.noisy_embed", document_version="1",
    )

    class NoisyFailingEmbedder:
        name = "test.noisy"
        version = "1"
        fingerprint = hashlib.sha256(b"test.noisy.v1").hexdigest()

        @staticmethod
        def capabilities():
            return {"protocol_version": "hlsgraph.embedder.v1",
                    "local_only": True, "network_access": False}

        @staticmethod
        def embed(texts):
            assert sentinel in texts[0]
            os.write(1, sentinel.encode("ascii"))
            os.write(2, sentinel.encode("ascii"))
            raise RuntimeError(f"embedding failed for {sentinel}")

    with pytest.raises(
        KnowledgePackError, match=r"^local embedder raised RuntimeError$",
    ) as caught:
        LocalKnowledgeSidecar(tmp_path).build(
            "test.knowledge.v03", [metadata], embedder=NoisyFailingEmbedder(),
        )

    captured = capfd.readouterr()
    assert sentinel not in captured.out
    assert sentinel not in captured.err
    assert sentinel not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_noisy_successful_embedder_still_publishes_vectors(
    tmp_path: Path, capfd,
) -> None:
    _bundle(tmp_path)
    sentinel = "PRIVATE_SUCCESS_EMBEDDER_SENTINEL_408b4a"
    document = tmp_path / "private-guide.md"
    document.write_text(f"# Pipeline\n{sentinel}\n", encoding="utf-8")
    metadata = index_local_document(
        document, document_id="test.local.success_embed", document_version="1",
    )

    class NoisySuccessfulEmbedder:
        name = "test.noisy_success"
        version = "1"
        fingerprint = hashlib.sha256(b"test.noisy_success.v1").hexdigest()

        @staticmethod
        def capabilities():
            return {"protocol_version": "hlsgraph.embedder.v1",
                    "local_only": True, "network_access": False}

        @staticmethod
        def embed(texts):
            assert sentinel in texts[0]
            os.write(1, sentinel.encode("ascii"))
            os.write(2, sentinel.encode("ascii"))
            return [[float(len(text)), 1.0] for text in texts]

    sidecar = LocalKnowledgeSidecar(tmp_path)
    manifest = sidecar.build(
        "test.knowledge.v03", [metadata], embedder=NoisySuccessfulEmbedder(),
    )
    captured = capfd.readouterr()
    assert sentinel not in captured.out
    assert sentinel not in captured.err
    assert manifest.embedder_id == "test.noisy_success"
    with sqlite3.connect(sidecar.database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM vectors").fetchone()[0] == 1


def test_embedder_identity_is_frozen_across_every_call(tmp_path: Path) -> None:
    document = tmp_path / "guide.md"
    document.write_text("# Pipeline\nprivate evidence\n", encoding="utf-8")
    metadata = index_local_document(
        document, document_id="test.local.mutable_embedder", document_version="1",
    )

    class MutableEmbedder:
        name = "test.mutable"
        version = "1"
        fingerprint = hashlib.sha256(b"test.mutable.v1").hexdigest()

        @staticmethod
        def capabilities():
            return {"protocol_version": "hlsgraph.embedder.v1",
                    "local_only": True, "network_access": False}

        def embed(self, texts):
            self.fingerprint = hashlib.sha256(b"changed").hexdigest()
            return [[1.0] for _text in texts]

    sidecar = LocalKnowledgeSidecar(tmp_path)
    with pytest.raises(
        KnowledgePackError, match=r"^embedder_identity.changed_after_call$",
    ):
        sidecar.build(
            "test.knowledge.mutable_embedder", [metadata],
            embedder=MutableEmbedder(),
        )
    assert not sidecar.manifest_path.exists()


def test_sidecar_decode_error_is_fixed_and_body_free(tmp_path: Path) -> None:
    sidecar = LocalKnowledgeSidecar(tmp_path)
    sidecar.prepare()
    sidecar.manifest_path.write_bytes(b"\xffPRIVATE-MANIFEST-BODY")
    with pytest.raises(
        KnowledgePackError,
        match=r"^local_sidecar_manifest.utf8_decode_failed$",
    ) as caught:
        sidecar.manifest()
    assert "PRIVATE-MANIFEST-BODY" not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_parser_exception_type_name_is_strictly_whitelisted() -> None:
    class UnsafeNameError(Exception):
        pass

    UnsafeNameError.__name__ = "Unsafe\nPRIVATE"
    assert sidecar_module._safe_external_type_name(UnsafeNameError()) == (
        "external_error"
    )


def test_pdf_is_metadata_only_without_explicit_local_parser(tmp_path: Path) -> None:
    _bundle(tmp_path)
    document = tmp_path / "guide.pdf"
    document.write_bytes(b"%PDF-1.7\nsynthetic test fixture\n")
    metadata = index_local_document(
        document, document_id="test.local.pdf", document_version="1",
    )
    sidecar = LocalKnowledgeSidecar(tmp_path)
    manifest = sidecar.build("test.knowledge.v03", [metadata])
    assert manifest.chunk_count == 0
    assert manifest.metadata["metadata_only_documents"] == ["test.local.pdf@1"]
    assert sidecar.search("synthetic") == []

    parsed = sidecar.build(
        "test.knowledge.v03", [metadata], parser=_PdfParser(),
    )
    assert parsed.chunk_count == 1
    assert parsed.parser_id == "test.pdf_parser"
    assert parsed.parser_fingerprint == _PdfParser.fingerprint
    assert sidecar.search("pipeline", include_text=True)[0].excerpt.startswith("PDF")


@pytest.mark.parametrize(
    ("parser", "kwargs", "message"),
    [
        (_SlowPdfParser(), {"parser_timeout_s": 0.1}, "timed out"),
        (_NonTextPdfParser(), {}, "non-text"),
        (_HugePdfParser(), {"max_parsed_chars": 16}, "exceeded"),
        (_ExitPdfParser(), {}, "without a bounded result"),
    ],
)
def test_parser_timeout_and_invalid_output_do_not_publish_partial_index(
    tmp_path: Path, parser, kwargs, message: str,
) -> None:
    _bundle(tmp_path)
    document = tmp_path / "guide.pdf"
    document.write_bytes(b"%PDF-1.7\nfixture\n")
    metadata = index_local_document(
        document, document_id="test.local.parser_guard", document_version="1",
    )
    sidecar = LocalKnowledgeSidecar(tmp_path)
    baseline = sidecar.build("test.knowledge.v03", [metadata])
    baseline_db = sidecar.database_path.read_bytes()
    with pytest.raises(KnowledgePackError, match=message):
        sidecar.build(
            "test.knowledge.v03", [metadata], parser=parser, **kwargs,
        )
    assert sidecar.manifest() == baseline
    assert sidecar.database_path.read_bytes() == baseline_db


def test_parser_input_limit_fails_before_plugin_execution(tmp_path: Path) -> None:
    _bundle(tmp_path)
    document = tmp_path / "guide.pdf"
    document.write_bytes(b"%PDF-1.7\n" + b"x" * 100)
    metadata = index_local_document(
        document, document_id="test.local.parser_input", document_version="1",
    )
    sidecar = LocalKnowledgeSidecar(tmp_path)
    with pytest.raises(KnowledgePackError, match="exceeds 16 bytes"):
        sidecar.build(
            "test.knowledge.v03", [metadata], parser=_PdfParser(),
            max_document_bytes=16,
        )
    assert not sidecar.manifest_path.exists()
    assert not sidecar.database_path.exists()


def test_review_surface_hash_excludes_only_attestation_metadata(
    tmp_path: Path,
) -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "src/hlsgraph/knowledge/packs/amd_public_guidance_2024_2.json"
    )
    value = json.loads(source.read_text(encoding="utf-8"))
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(value), encoding="utf-8")
    expected = surface_sha256(baseline)

    reviewed = json.loads(json.dumps(value))
    reviewed["metadata"].update({
        "review_status": "machine_repeated_reviewed",
        "reviewers": ["review.invocation.one", "review.invocation.two"],
        "source_hashes": {"document": "a" * 64},
        "review_evidence": {"review_agreement": True},
    })
    reviewed["coverage"].update({
        "review_status": "machine_repeated_reviewed",
        "reviewers": ["review.invocation.one", "review.invocation.two"],
        "review_evidence": {"review_agreement": True},
    })
    reviewed_path = tmp_path / "reviewed.json"
    reviewed_path.write_text(json.dumps(reviewed), encoding="utf-8")
    assert surface_sha256(reviewed_path) == expected

    semantic_change = json.loads(json.dumps(reviewed))
    semantic_change["rules"][0]["summary"] += " changed"
    changed_path = tmp_path / "changed.json"
    changed_path.write_text(json.dumps(semantic_change), encoding="utf-8")
    assert surface_sha256(changed_path) != expected
