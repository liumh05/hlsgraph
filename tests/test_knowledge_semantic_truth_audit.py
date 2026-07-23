from __future__ import annotations

from collections import Counter

from hlsgraph.knowledge import KnowledgeCatalog


def _packs():
    return {pack.pack_id: pack for pack in KnowledgeCatalog.builtin().packs}


def _rule(pack, suffix: str):
    return next(rule for rule in pack.rules if rule.id.endswith(f":{suffix}"))


def _target(pack, target_kind: str, target: str):
    return next(
        item for item in pack.coverage.target_inventory
        if (item.target_kind, item.target) == (target_kind, target)
    )


def test_truth_first_downgrades_remove_unsupported_amd_bindings() -> None:
    amd = _packs()["hlsgraph.amd.public_guidance.2024_2"]

    dependence = _rule(amd, "directive.dependence_is_user_assertion")
    assert dependence.condition == {"directive_kind": "DEPENDENCE"}
    assert "variable" not in dependence.summary.casefold().split("must", 1)[-1]
    assert not [
        binding for binding in amd.bindings
        if binding.knowledge_rule_id == dependence.id
    ]
    dependence_target = _target(amd, "directive_kind", "DEPENDENCE")
    assert dependence_target.status.value == "no_normative"
    assert not dependence_target.binding_ids
    assert "mutually exclusive" in dependence_target.rationale

    assert not any(
        rule.rule_id == "qor.csynth_is_estimate" for rule in amd.rules
    )
    assert not any(
        binding.knowledge_rule_id.endswith(":qor.csynth_is_estimate")
        for binding in amd.bindings
    )
    csynth_section = next(
        entry for entry in amd.coverage.entries
        if entry.document_id == "amd.ug1399"
        and entry.section == "Output of C Synthesis"
    )
    assert csynth_section.status.value == "citation_only"
    assert not csynth_section.rule_ids and not csynth_section.binding_ids
    assert _target(
        amd, "artifact_kind", "amd.vitis.csynth_xml"
    ).status.value == "no_normative"

    latency = _rule(amd, "qor.latency_and_ii_are_distinct")
    assert "target II" not in latency.summary
    assert "achieved II" not in latency.summary
    assert {
        binding.target for binding in amd.bindings
        if binding.knowledge_rule_id == latency.id
    } == {
        "qor.latency_best_cycles",
        "qor.latency_worst_cycles",
        "qor.interval_min_cycles",
        "qor.interval_max_cycles",
        "qor.latency_cycles",
        "qor.iteration_latency_cycles",
    }
    for target in ("qor.target_ii", "qor.achieved_ii"):
        assert _target(amd, "predicate", target).status.value == "no_normative"


def test_metric_rules_do_not_implicitly_decide_complete_gates() -> None:
    amd = _packs()["hlsgraph.amd.public_guidance.2024_2"]

    timing = _rule(amd, "timing.summary_keeps_wns_tns_distinct")
    assert timing.effect == {
        "predicates": ["timing.wns_ns", "timing.tns_ns"],
        "preserve_report_stage": True,
    }
    assert {
        binding.target for binding in amd.bindings
        if binding.knowledge_rule_id == timing.id
    } == {
        "amd.vivado.timing_summary", "timing.wns_ns", "timing.tns_ns",
    }
    timing_gate = _target(amd, "gate_kind", "post_route_timing")
    assert timing_gate.status.value == "no_normative"
    assert "bus-skew" in timing_gate.rationale

    utilization = _rule(amd, "resource.utilization_is_stage_scoped")
    assert utilization.effect == {
        "preserve_stage": True,
        "preserve_report_scope": True,
    }
    assert "post-route-only" in utilization.metadata[
        "hlsgraph_local_derivation_policy"
    ]
    resource_gate = _target(amd, "gate_kind", "resource_fits")
    assert resource_gate.status.value == "no_normative"
    assert "local derivation" in resource_gate.rationale
    assert {
        binding.target for binding in amd.bindings
        if binding.target_kind == "gate_kind"
    } == {"correctness"}


def test_open_ir_uses_pinned_semantic_sources_and_labels_local_policy() -> None:
    open_ir = _packs()["hlsgraph.open_ir.public_guidance.2026_07_21"]
    documents = {item.document_id: item for item in open_ir.documents}

    location_source = (
        "https://github.com/llvm/llvm-project/blob/"
        "429c88d37f1f02e68ebc1fc7b0da4511ce6407e3/"
        "mlir/include/mlir/IR/BuiltinLocationAttributes.td"
    )
    assert documents["llvm.mlir.builtin"].official_url == location_source
    location = _rule(open_ir, "mlir.locations_are_mapping_evidence")
    assert location.citation_url == location_source
    assert "hlsgraph_local_derivation_policy" in location.metadata

    gep = _rule(open_ir, "llvm.index_histograms_require_explicit_operand_schema")
    assert gep.section == "GetElementPtr Instruction"
    assert gep.condition == {"typed_gep_index_histogram_present": True}
    assert "aggregate index" not in gep.summary.casefold()
    assert gep.citation_url.endswith("/llvm/docs/LangRef.md#getelementptr-instruction")
    assert "hlsgraph_local_derivation_policy" in gep.metadata

    handshake_source = (
        "https://github.com/llvm/circt/blob/"
        "ef03d45c960607315a8b62903b92d072d8542e30/"
        "include/circt/Dialect/Handshake/HandshakeOps.td"
    )
    assert documents["circt.handshake"].official_url == handshake_source
    handshake = _rule(open_ir, "circt.handshake_has_dataflow_semantics")
    assert handshake.citation_url == handshake_source
    assert "fine-grained dataflow operations" in handshake.summary
    assert "Handshake function operation" in handshake.summary
    assert "dataflow graph regions" not in handshake.summary
    assert "hlsgraph_local_derivation_policy" in handshake.metadata

    scalehls = documents["scalehls.paper"]
    assert scalehls.document_version == "arxiv-2107.11673v4"
    assert scalehls.official_url == "https://arxiv.org/abs/2107.11673v4"


def test_reference_metadata_and_coverage_remain_exact_once_and_unreviewed() -> None:
    packs = _packs()
    axi = packs["hlsgraph.axi.public_guidance.v1"]
    assert next(
        item for item in axi.documents if item.document_id == "arm.ihi0022"
    ).title == "AMBA AXI and ACE Protocol Specification"

    for pack in packs.values():
        assert pack.metadata["review_status"] == "unreviewed"
        assert "hlsgraph_local_derivation_policy" in pack.metadata
        assert pack.coverage.review_status == "unreviewed"
        assert pack.coverage.reviewers == []
        assert pack.coverage.source_hashes == {}
        assert Counter(
            rule_id
            for entry in pack.coverage.entries
            if entry.status.value == "rule"
            for rule_id in entry.rule_ids
        ) == Counter({rule.id: 1 for rule in pack.rules})
        assert Counter(
            binding_id
            for entry in pack.coverage.entries
            if entry.status.value == "rule"
            for binding_id in entry.binding_ids
        ) == Counter({binding.id: 1 for binding in pack.bindings})
        assert Counter(
            binding_id
            for target in pack.coverage.target_inventory
            for binding_id in target.binding_ids
        ) == Counter({binding.id: 1 for binding in pack.bindings})
