from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
import stat
from types import MappingProxyType, SimpleNamespace
from urllib.parse import quote

import pytest

from hlsgraph import RETRIEVAL_PROFILE_SCHEMA_VERSION
from hlsgraph.api import RestApplication
from hlsgraph.bundle import GraphBundle
from hlsgraph.cli import main as cli_main
from hlsgraph.graph import CanonicalGraph
from hlsgraph.knowledge import (
    KnowledgeCatalog, LocalKnowledgeSidecar, index_local_document, matches_binding_constraints,
)
from hlsgraph.knowledge.activation import BindingActivationSession
from hlsgraph.manifest import minimal_manifest
from hlsgraph.mcp import ReadOnlyMcpService
from hlsgraph.model import (
    ArtifactRef, AuthorityClass, Entity, GateKind, GateResult, GateStatus, KnowledgeBinding,
    KnowledgeRule, Observation, PredictionEnvelope, Relation, RunStatus,
    SourceAnchor, ToolchainContext, ToolRun, stable_hash,
)
from hlsgraph.query import CoreService
from hlsgraph.retrieval import (
    DEFAULT_PLANES, HybridRetriever, RetrievalItem, RetrievalSpec, normalize_terms,
)
from hlsgraph.sdk import Project
from tests.reviewed_knowledge_support import install_reviewed_builtin_packs


@pytest.fixture()
def retrieval_project(tmp_path: Path) -> dict[str, object]:
    secret = "PRIVATE_SOURCE_BODY_MUST_NOT_ENTER_RETRIEVAL"
    (tmp_path / "kernel.cpp").write_text(
        f"// {secret}\nvoid dut() {{}}\n", encoding="utf-8",
    )
    manifest = minimal_manifest("test.retrieval", "retrieval", "dut", "kernel.cpp")
    manifest.toolchains = [ToolchainContext(
        id="amd.vitis_hls.2024_2", vendor="amd", name="vitis_hls", version="2024.2",
    )]
    bundle = GraphBundle.create(tmp_path, manifest)
    install_reviewed_builtin_packs(bundle)
    snapshot = bundle.snapshot()
    artifact = bundle.store.artifacts(snapshot.id)[0]
    kernel = Entity(
        kind="hls.kernel", name="dut", qualified_name="dut", snapshot_id=snapshot.id,
        stage="hls_ir", anchors=[SourceAnchor(
            artifact.id, start_line=2, end_line=2, symbol="dut",
        )],
    )
    region = Entity(
        kind="hls.dataflow_region", name="compute", qualified_name="dut::compute",
        snapshot_id=snapshot.id, stage="hls_ir",
    )
    sink = Entity(
        kind="hls.process", name="store", qualified_name="dut::store",
        snapshot_id=snapshot.id, stage="hls_ir",
    )
    helper = Entity(
        kind="software.function", name="helper", qualified_name="helper",
        snapshot_id=snapshot.id, stage="ast",
    )
    llvm_a = Entity(
        kind="ir.llvm.block", name="entry", qualified_name="dut::entry",
        snapshot_id=snapshot.id, stage="llvm",
    )
    llvm_b = Entity(
        kind="ir.llvm.block", name="exit", qualified_name="dut::exit",
        snapshot_id=snapshot.id, stage="llvm",
    )
    graph = CanonicalGraph(snapshot.id)
    for entity in (kernel, region, sink, helper, llvm_a, llvm_b):
        graph.add_entity(entity)
    graph.add_relation(Relation(
        kernel.id, region.id, "hls.contains", snapshot.id, stage="hls_ir",
        authority=AuthorityClass.COMPILER_DECISION,
    ))
    stream_relation = Relation(
        region.id, sink.id, "hls.streams_to", snapshot.id, stage="hls_ir",
        authority=AuthorityClass.COMPILER_DECISION,
        attrs={"fifo_depth": 4},
    )
    graph.add_relation(stream_relation)
    graph.add_relation(Relation(
        helper.id, kernel.id, "software.calls", snapshot.id, stage="ast",
    ))
    graph.add_relation(Relation(
        llvm_a.id, llvm_b.id, "llvm.cfg", snapshot.id, stage="llvm",
        authority=AuthorityClass.COMPILER_DECISION,
    ))
    bundle.store.save_graph(graph)
    observation = Observation(
        snapshot_id=snapshot.id, subject_id=region.id,
        predicate="schedule.achieved_ii", value=2, unit="cycles",
        stage="schedule", authority=AuthorityClass.TOOL_OBSERVATION,
        artifact_id=artifact.id,
    )
    bundle.store.add_observations([observation])
    bundle.store.add_knowledge_rules([KnowledgeRule(
        document_id="amd.ug1399", document_version="2024.2",
        section="Dataflow Viewer", rule_id="test.workload_scope",
        title="FIFO stalls are workload scoped",
        applicability={"vendor": "amd", "stage": "cosim"},
        condition={"predicate": "fifo.stall_cycles"},
        effect={"requires": "workload_id"},
        citation_url="https://docs.amd.com/r/2024.2-English/ug1399-vitis-hls/Dataflow-Viewer",
        summary="Treat dynamic FIFO stalls as observations of one cosimulation workload.",
    )])
    prediction = PredictionEnvelope(
        snapshot_id=snapshot.id, subject_id=region.id,
        predicate="prediction.latency_cycles", value=42,
        model_id="model.fixture", model_version="1",
        input_schema_version="fixture.v1",
    )
    bundle.store.add_prediction(prediction)
    return {
        "root": tmp_path, "bundle": bundle, "snapshot": snapshot.id,
        "kernel": kernel.id, "region": region.id, "sink": sink.id,
        "helper": helper.id, "prediction": prediction.id,
        "observation": observation.id, "artifact": artifact.id, "secret": secret,
        "stream_relation": stream_relation.id,
    }


def test_flow_spine_excludes_evidence_only_relations(retrieval_project) -> None:
    bundle = retrieval_project["bundle"]
    snapshot_id = retrieval_project["snapshot"]
    assert isinstance(bundle, GraphBundle)
    assert isinstance(snapshot_id, str)
    graph = bundle.store.load_graph(snapshot_id)
    evidence_only = Relation(
        retrieval_project["region"], retrieval_project["sink"],
        "handshake.dataflow", snapshot_id, stage="mlir",
        authority=AuthorityClass.COMPILER_DECISION,
        attrs={"hardware_topology": False, "native_ir_evidence": True},
    )
    graph.add_relation(evidence_only)
    fact = RetrievalItem(
        record_id=str(retrieval_project["region"]), plane="facts",
        record_kind="entity", title="compute",
        summary="dataflow region", entity_id=str(retrieval_project["region"]),
    )

    flow = HybridRetriever(bundle, snapshot_id)._flow_spine(
        graph,
        {
            str(retrieval_project["region"]): 1.0,
            str(retrieval_project["sink"]): 1.0,
        },
        {evidence_only.id},
        [fact],
        max_edges=8,
    )

    assert flow == []


@pytest.mark.parametrize(
    ("constraint", "generic_actual", "runtime_actual", "expected"),
    [
        (True, True, {"true"}, True),
        (True, "true", {"true"}, True),
        (False, False, {"false"}, True),
        (False, "false", {"false"}, True),
        (False, {"false"}, {"false"}, True),
        (True, False, {"false"}, False),
    ],
)
def test_json_boolean_binding_discriminators_match_identically(
    constraint: bool, generic_actual: object,
    runtime_actual: set[str], expected: bool,
) -> None:
    binding = KnowledgeBinding(
        knowledge_rule_id="test.document:1:test.boolean_rule",
        target_kind="relation_kind", target="cross.maps_to",
        required_context={"hardware_topology": constraint},
        producer="hlsgraph.knowledge.binding", producer_version="1",
    )
    assert matches_binding_constraints(
        binding, target_kind="relation_kind", target="cross.maps_to",
        context={"hardware_topology": generic_actual},
    ) is expected
    assert HybridRetriever._constraint_matches(
        constraint, runtime_actual,
    ) is expected
    assert HybridRetriever._constraint_mentions(
        constraint, "true" if constraint else "false",
    )


def test_bilingual_identifier_normalization_is_stable() -> None:
    terms = normalize_terms("数据流 achievedII 与启动间隔")
    assert terms[:3] == ["数据流", "dataflow", "stream"]
    assert {"achieved", "ii", "initiation", "interval"}.issubset(terms)
    assert terms == normalize_terms("数据流 achievedII 与启动间隔")
    with pytest.raises(ValueError, match="4096"):
        RetrievalSpec(query="x" * 4_097)
    with pytest.raises(ValueError, match="NUL"):
        RetrievalSpec(query="pipeline\x00ii")
    with pytest.raises(ValueError, match="unsupported retrieval profile"):
        RetrievalSpec(query="pipeline II", profile="unregistered.weights.v9")
    with pytest.raises(ValueError, match="include_private_snippets must be a boolean"):
        RetrievalSpec(query="pipeline II", include_private_snippets="true")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="include_predictions must be a boolean"):
        RetrievalSpec(query="pipeline II", include_predictions=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="requires include_predictions=True"):
        RetrievalSpec(query="pipeline II", planes=("predictions",))

    prediction_spec = RetrievalSpec(query="pipeline II", include_predictions=True)
    assert prediction_spec.planes == (*DEFAULT_PLANES, "predictions")


def test_hybrid_retrieval_separates_truth_planes_and_hides_query_and_source(
    retrieval_project: dict[str, object],
) -> None:
    core = CoreService(retrieval_project["bundle"], retrieval_project["snapshot"])
    result = core.retrieve(RetrievalSpec(query="compute achieved II", top_k=8))
    payload = result.to_dict()
    serialized = json.dumps(payload, ensure_ascii=False)

    assert retrieval_project["region"] in {item.record_id for item in result.facts}
    assert retrieval_project["observation"] in {item.record_id for item in result.facts}
    assert result.predictions == []
    assert all(item.authority_class != "prediction_hypothesis" for item in result.facts)
    assert "compute achieved II" not in serialized
    assert retrieval_project["secret"] not in serialized
    assert result.trace.query_sha256 and len(result.trace.query_sha256) == 64
    assert result.trace.profile_schema_version == RETRIEVAL_PROFILE_SCHEMA_VERSION == "0.3.0"
    assert payload["trace"]["profile_schema_version"] == "0.3.0"
    assert result.trace.output_chars <= result.trace.output_budget_chars
    assert len(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"))) <= result.trace.output_budget_chars


def test_rule_conditions_are_evaluated_on_one_instance_context() -> None:
    pack = next(item for item in KnowledgeCatalog.builtin().packs
                if item.pack_id == "hlsgraph.amd.public_guidance.2024_2")
    rule = next(item for item in pack.rules
                if item.rule_id == "verification.csim_is_workload_scoped")
    binding = next(item for item in pack.bindings
                   if item.knowledge_rule_id == rule.id
                   and item.target == "csim.exit_code")
    context = {
        "vendor": {"amd"}, "tool": {"vitis_hls"},
        "tool_version": {"2024.2"}, "stage": {"csim"},
        "workload_id": {"workload.fixture"},
        "snapshot_association": {"verified"},
        "observation_evidence_qualified": {
            "derived_from_typed_observation_evidence_v1",
        },
        "observation_instance_id": {"observation.fixture"},
        "observation_artifact_kind": {"amd.vitis.csim_result"},
        "observation_artifact_identity": {"artifact.fixture"},
        "observation_run_identity": {"run.fixture"},
    }
    targets = {"predicate": {"csim.exit_code"}}
    assert HybridRetriever._binding_constraints_match_values(
        binding, context, targets, condition=rule.condition,
    )

    for changed in (
        {**context, "observation_artifact_kind": set()},
        {**context, "observation_artifact_kind": {"amd.vitis.cosim_rpt"}},
        {**context, "observation_artifact_kind": {
            "amd.vitis.csim_result", "amd.vitis.cosim_rpt",
        }},
        {**context, "observation_instance_id": {
            "observation.fixture", "observation.sibling",
        }},
    ):
        assert not HybridRetriever._binding_constraints_match_values(
            binding, changed, targets, condition=rule.condition,
        )


def test_executable_binding_requires_current_retriever_session(
    retrieval_project: dict[str, object],
) -> None:
    pack = next(item for item in KnowledgeCatalog.builtin().packs
                if item.pack_id == "hlsgraph.amd.public_guidance.2024_2")
    rule = next(item for item in pack.rules
                if item.rule_id == "verification.csim_is_workload_scoped")
    binding = next(item for item in pack.bindings
                   if item.knowledge_rule_id == rule.id
                   and item.target == "csim.exit_code")
    raw_context = {
        "vendor": {"amd"}, "tool": {"vitis_hls"},
        "tool_version": {"2024.2"}, "stage": {"csim"},
        "workload_id": {"workload.fixture"},
        "snapshot_association": {"verified"},
        "observation_evidence_qualified": {
            "derived_from_typed_observation_evidence_v1",
        },
        "observation_instance_id": {"observation.fixture"},
        "observation_artifact_kind": {"amd.vitis.csim_result"},
        "observation_artifact_identity": {"artifact.fixture"},
        "observation_run_identity": {"run.fixture"},
    }
    bundle = retrieval_project["bundle"]
    snapshot_id = str(retrieval_project["snapshot"])
    graph = bundle.store.load_graph(snapshot_id)
    retriever = HybridRetriever(bundle, snapshot_id)
    session = BindingActivationSession(
        snapshot_id=snapshot_id, graph_hash=graph.graph_hash,
        allowed_ids=sorted(graph.entities), bindings=[binding], rules=[rule],
        raw_contexts={("predicate", "csim.exit_code"): [raw_context]},
    )
    attested = session.issue(binding, raw_context)
    assert attested is not None
    spec = RetrievalSpec(query="csim", applicability={"stage": "csim"})

    def binding_applies(candidate, candidate_context) -> bool:
        evaluation = retriever._binding_evaluation(
            session, candidate, candidate_context, spec,
        )
        return bool(
            evaluation is not None
            and evaluation.request_matches
            and evaluation.binding_matches
            and evaluation.rule_applicable
        )

    try:
        assert binding_applies(binding, attested)
        assert not binding_applies(binding, raw_context)
        assert not binding_applies(binding, dict(raw_context))
        assert not binding_applies(binding, replace(attested))
        detached_binding = session.binding_snapshot_for(binding, attested)
        detached_binding_again = session.binding_snapshot_for(binding, attested)
        assert detached_binding is not None and detached_binding is not binding
        assert detached_binding_again is not None
        assert detached_binding_again is not detached_binding
        original_target = detached_binding.target
        original_stage = detached_binding.required_context["stage"]
        object.__setattr__(detached_binding, "target", "forged.target")
        object.__setattr__(detached_binding, "required_context", {"stage": "ast"})
        assert detached_binding_again.target == original_target
        assert detached_binding_again.required_context["stage"] == original_stage
        assert not binding_applies(detached_binding, attested)
        assert binding_applies(binding, attested)

        detached_rule = session.rule_snapshot_for(rule)
        detached_rule_again = session.rule_snapshot_for(rule)
        assert detached_rule is not None and detached_rule_again is not None
        assert detached_rule is not detached_rule_again
        original_title = detached_rule.title
        original_effect = dict(detached_rule.effect)
        object.__setattr__(detached_rule, "title", "forged title")
        object.__setattr__(detached_rule, "effect", {"forged": True})
        assert detached_rule_again.title == original_title
        assert detached_rule_again.effect == original_effect
        assert not hasattr(session, "authorize_and_snapshot")
        assert not hasattr(retriever, "_authorized_binding_applicable")
        original_values = attested.values
        forged_values = dict(original_values)
        forged_values["tool_version"] = frozenset({"2023.2"})
        object.__setattr__(
            attested, "values", MappingProxyType(forged_values),
        )
        assert not session.validate(binding, attested)
        assert not binding_applies(binding, attested)
        object.__setattr__(attested, "values", original_values)
        assert session.validate(binding, attested)
        original_stage = binding.required_context["stage"]
        binding.required_context["stage"] = "cosim"
        assert detached_binding_again.required_context["stage"] == original_stage
        assert not binding_applies(binding, attested)
        binding.required_context["stage"] = original_stage
        original_title = rule.title
        rule.title = "mutated after issuance"
        assert not session.validate(binding, attested)
        rule.title = original_title

        object.__setattr__(rule, "title", "\ud800")
        assert not session.validate(binding, attested)
        assert session.evaluate_atomically(
            binding, attested, lambda _binding, _rule, _values: True,
        ) is None
        object.__setattr__(rule, "title", original_title)
        object.__delattr__(rule, "title")
        assert not session.validate(binding, attested)
        object.__setattr__(rule, "title", original_title)
        assert session.validate(binding, attested)

        original_raw_stage = raw_context["stage"]
        raw_context["stage"] = {"ast"}
        assert not session.validate(binding, attested)
        assert not binding_applies(binding, attested)
        raw_context["stage"] = original_raw_stage
        assert session.validate(binding, attested)

        def mutate_live_rule_midflight(_binding, _rule, _values):
            rule.title = "mutated inside evaluator"
            return "must be discarded"

        assert session.evaluate_atomically(
            binding, attested, mutate_live_rule_midflight,
        ) is None
        rule.title = original_title
        assert binding_applies(binding, attested)
    finally:
        session.close()
    assert not binding_applies(binding, attested)


def test_binding_activation_duplicate_id_fails_without_partial_session_dos(
    retrieval_project: dict[str, object],
) -> None:
    pack = next(item for item in KnowledgeCatalog.builtin().packs
                if item.pack_id == "hlsgraph.amd.public_guidance.2024_2")
    binding = pack.bindings[0]
    rule = next(
        item for item in pack.rules if item.id == binding.knowledge_rule_id
    )
    duplicate = KnowledgeBinding.from_dict({
        "knowledge_rule_id": binding.knowledge_rule_id,
        "target_kind": binding.target_kind,
        "target": binding.target,
        "required_context": dict(binding.required_context),
        "producer": binding.producer,
        "producer_version": binding.producer_version,
        "id": binding.id,
        "metadata": dict(binding.metadata),
    })
    bundle = retrieval_project["bundle"]
    snapshot_id = str(retrieval_project["snapshot"])
    graph = bundle.store.load_graph(snapshot_id)
    with pytest.raises(ValueError, match="duplicate IDs"):
        BindingActivationSession(
            snapshot_id=snapshot_id, graph_hash=graph.graph_hash,
            allowed_ids=sorted(graph.entities),
            bindings=[binding, duplicate], rules=[rule], raw_contexts={},
        )


def test_issued_binding_claims_cannot_be_retargeted_or_rescoped(
    retrieval_project: dict[str, object],
) -> None:
    pack = next(item for item in KnowledgeCatalog.builtin().packs
                if item.pack_id == "hlsgraph.amd.public_guidance.2024_2")
    rule_a = next(item for item in pack.rules
                  if item.rule_id == "verification.csim_is_workload_scoped")
    binding_a = next(item for item in pack.bindings
                     if item.knowledge_rule_id == rule_a.id
                     and item.target == "csim.exit_code")
    rule_b = KnowledgeRule(
        document_id="test.activation", document_version="1",
        section="Retarget", rule_id="test.retarget_b",
        title="Retarget B", applicability=dict(rule_a.applicability),
        condition=dict(rule_a.condition), effect={"forged": True},
        citation_url="https://example.com/retarget-b",
    )
    binding_b = KnowledgeBinding(
        knowledge_rule_id=rule_b.id,
        target_kind=binding_a.target_kind,
        target=binding_a.target,
        required_context=dict(binding_a.required_context),
        producer="hlsgraph.test.binding", producer_version="1",
    )
    binding_c = KnowledgeBinding(
        knowledge_rule_id=rule_a.id,
        target_kind=binding_a.target_kind,
        target="csim.return_code",
        required_context=dict(binding_a.required_context),
        producer="hlsgraph.test.binding", producer_version="1",
    )
    raw_context = {
        "vendor": {"amd"}, "tool": {"vitis_hls"},
        "tool_version": {"2024.2"}, "stage": {"csim"},
        "workload_id": {"workload.fixture"},
        "snapshot_association": {"verified"},
        "observation_evidence_qualified": {
            "derived_from_typed_observation_evidence_v1",
        },
        "observation_instance_id": {"observation.fixture"},
        "observation_artifact_kind": {"amd.vitis.csim_result"},
        "observation_artifact_identity": {"artifact.fixture"},
        "observation_run_identity": {"run.fixture"},
    }
    bundle = retrieval_project["bundle"]
    snapshot_id = str(retrieval_project["snapshot"])
    graph = bundle.store.load_graph(snapshot_id)
    session = BindingActivationSession(
        snapshot_id=snapshot_id, graph_hash=graph.graph_hash,
        allowed_ids=sorted(graph.entities),
        bindings=[binding_a, binding_b, binding_c], rules=[rule_a, rule_b],
        raw_contexts={
            (binding_a.target_kind, binding_a.target): [raw_context],
            (binding_c.target_kind, binding_c.target): [raw_context],
        },
    )
    handle = session.issue(binding_a, raw_context)
    assert handle is not None and session.validate(binding_a, handle)

    original_claims = {
        name: getattr(handle, name) for name in (
            "binding_id", "binding_fingerprint", "rule_id",
            "rule_fingerprint", "snapshot_id", "graph_hash", "scope_hash",
        )
    }
    object.__setattr__(handle, "binding_id", binding_b.id)
    object.__setattr__(handle, "binding_fingerprint", stable_hash(binding_b))
    object.__setattr__(handle, "rule_id", rule_b.id)
    object.__setattr__(handle, "rule_fingerprint", stable_hash(rule_b))
    assert not session.validate(binding_a, handle)
    assert not session.validate(binding_b, handle)
    assert session.evaluate_atomically(
        binding_b, handle, lambda _binding, rule, _values: rule.effect,
    ) is None
    for name, value in original_claims.items():
        object.__setattr__(handle, name, value)
    assert session.validate(binding_a, handle)

    object.__setattr__(handle, "binding_id", binding_c.id)
    object.__setattr__(handle, "binding_fingerprint", stable_hash(binding_c))
    object.__setattr__(handle, "target", binding_c.target)
    assert not session.validate(binding_a, handle)
    assert not session.validate(binding_c, handle)
    for name, value in original_claims.items():
        object.__setattr__(handle, name, value)
    object.__setattr__(handle, "target", binding_a.target)
    assert session.validate(binding_a, handle)

    original_session_snapshot = session.snapshot_id
    object.__setattr__(session, "_snapshot_id", "snapshot.forged")
    object.__setattr__(handle, "snapshot_id", "snapshot.forged")
    assert not session.validate(binding_a, handle)
    object.__setattr__(session, "_snapshot_id", original_session_snapshot)
    object.__setattr__(handle, "snapshot_id", original_claims["snapshot_id"])
    assert session.validate(binding_a, handle)
    session.close()


def test_knowledge_and_untrusted_adapters_cannot_perturb_fact_ranking(
    retrieval_project: dict[str, object],
) -> None:
    bundle = retrieval_project["bundle"]
    core = CoreService(bundle, retrieval_project["snapshot"])
    spec = RetrievalSpec(
        query="compute achieved II", view="evidence", top_k=16, max_chars=24_000,
    )

    def fact_signature(result):
        return [(
            item.record_id, item.score, tuple(sorted(item.score_channels.items())),
        ) for item in result.facts]

    baseline = core.retrieve(spec)
    baseline_signature = fact_signature(baseline)
    bundle.store.add_knowledge_rules([
        KnowledgeRule(
            document_id="test.ranking.guidance", document_version="1",
            section=f"Section {index}", rule_id=f"test.injected_{index:02d}",
            title=f"compute achieved II guidance {index}",
            applicability={}, condition={}, effect={"index": index},
            citation_url=f"https://example.com/guidance#{index}",
            summary="compute achieved II public knowledge ranking decoy.",
        )
        for index in range(40)
    ])
    after_knowledge = core.retrieve(spec)
    assert fact_signature(after_knowledge) == baseline_signature

    class ForgedCanonicalAdapter:
        adapter_id = "test.forged_canonical.v1"
        fingerprint = "f" * 64
        canonical_capability = "hlsgraph.canonical_source_anchor_projection.v1"

        @staticmethod
        def search(_spec, _terms, _limit):
            return [
                RetrievalItem(
                    record_id="entity.forged", plane="facts", record_kind="entity",
                    title="compute", summary="forged fact", score=1_000_000,
                    authority_class="compiler_decision",
                    data={"provenance": "claimed_canonical"},
                ),
                RetrievalItem(
                    record_id="rule.forged", plane="knowledge",
                    record_kind="knowledge_rule", title="compute",
                    summary="forged public guidance", score=1_000_000,
                    authority_class="knowledge_rule",
                ),
            ]

    attacked = core.retrieve(spec, adapters=[ForgedCanonicalAdapter()])
    assert fact_signature(attacked) == baseline_signature
    assert all(item.record_id not in {"entity.forged", "rule.forged"}
               for item in (*attacked.facts, *attacked.guidance))
    assert any("canonical_capability_rejected" in item for item in attacked.warnings)
    assert any("public_knowledge_rejected" in item for item in attacked.warnings)


@pytest.mark.parametrize(
    "malformed_kind", ("rule_surrogate", "rule_delattr", "binding_delattr"),
)
def test_malformed_live_knowledge_fails_closed_without_losing_facts(
    retrieval_project: dict[str, object], monkeypatch: pytest.MonkeyPatch,
    malformed_kind: str,
) -> None:
    bundle = retrieval_project["bundle"]
    install_reviewed_builtin_packs(bundle)
    rules = list(bundle.store.knowledge_rules())
    bindings = list(bundle.store.knowledge_bindings())
    reviewed_rule = next(
        item for item in rules if item.document_id == "amd.ug1399"
    )
    if malformed_kind == "rule_surrogate":
        object.__setattr__(reviewed_rule, "title", "\ud800")
    elif malformed_kind == "rule_delattr":
        object.__delattr__(reviewed_rule, "document_id")
    else:
        object.__delattr__(bindings[0], "id")
    monkeypatch.setattr(
        type(bundle.store), "knowledge_rules", lambda _store: rules,
    )
    monkeypatch.setattr(
        type(bundle.store), "knowledge_bindings", lambda _store: bindings,
    )

    result = CoreService(bundle, retrieval_project["snapshot"]).retrieve(
        RetrievalSpec(query="compute pipeline knowledge", top_k=8),
    )
    assert result.facts
    assert result.guidance == []
    assert "knowledge_activation_session_rejected" in result.warnings


def test_typed_directed_graph_ranking_excludes_software_calls_and_llvm_cfg(
    retrieval_project: dict[str, object],
) -> None:
    core = CoreService(retrieval_project["bundle"], retrieval_project["snapshot"])
    architecture = core.retrieve(RetrievalSpec(query="dut", view="architecture"))
    assert retrieval_project["region"] in {item.record_id for item in architecture.facts}
    assert retrieval_project["helper"] not in {item.record_id for item in architecture.facts}
    assert all(item["kind"] not in {"software.calls", "llvm.cfg"}
               for item in architecture.flow)

    evidence = core.retrieve(RetrievalSpec(query="helper", view="evidence"))
    assert retrieval_project["helper"] in {item.record_id for item in evidence.facts}
    assert all(item["kind"] not in {"software.calls", "llvm.cfg"}
               for item in evidence.flow)


def test_explicit_hardware_relations_are_searchable_facts(
    retrieval_project: dict[str, object],
) -> None:
    result = CoreService(
        retrieval_project["bundle"], retrieval_project["snapshot"],
    ).retrieve(RetrievalSpec(query="hls.streams_to fifo depth", top_k=8))
    relation = next(item for item in result.facts
                    if item.record_id == retrieval_project["stream_relation"])
    assert relation.record_kind == "relation"
    assert relation.data["kind"] == "hls.streams_to"
    assert relation.data["attrs"]["fifo_depth"] == 4
    assert relation.authority_class == "compiler_decision"


def test_knowledge_binding_semantic_conditions_fail_closed(
    retrieval_project: dict[str, object],
) -> None:
    rule_id = retrieval_project["bundle"].store.knowledge_rules()[0].id
    targets = {"predicate": {"schedule.achieved_ii", "profile.fifo_max_occupancy"}}
    context = {
        "vendor": {"amd"}, "tool": {"vitis_hls"},
        "tool_version": {"2024.2"}, "stage": {"schedule", "cosim"},
        "workload_id": {"tb.default"},
    }

    missing_version = KnowledgeBinding(
        knowledge_rule_id=rule_id, target_kind="predicate",
        target="schedule.achieved_ii",
        required_context={"vendor": "amd", "tool": "vitis_hls", "stage": "schedule"},
        producer="hlsgraph.builtin", producer_version="0.3",
        metadata={"dynamic_scope": "static"},
    )
    assert not HybridRetriever._binding_constraints_match_values(missing_version, context, targets)

    static = KnowledgeBinding(
        knowledge_rule_id=rule_id, target_kind="predicate",
        target="schedule.achieved_ii",
        required_context={
            "vendor": "amd", "tool": "vitis_hls",
            "tool_version": "2024.2", "stage": "schedule",
        },
        producer="hlsgraph.builtin", producer_version="0.3",
        metadata={"dynamic_scope": "static"},
    )
    assert HybridRetriever._binding_constraints_match_values(static, context, targets)

    dynamic_missing_workload = KnowledgeBinding(
        knowledge_rule_id=rule_id, target_kind="predicate",
        target="profile.fifo_max_occupancy",
        required_context={
            "vendor": "amd", "tool": "vitis_hls",
            "tool_version": "2024.2", "stage": "cosim",
        },
        producer="hlsgraph.builtin", producer_version="0.3",
    )
    assert not HybridRetriever._binding_constraints_match_values(
        dynamic_missing_workload, context, targets,
    )

    dynamic = KnowledgeBinding(
        knowledge_rule_id=rule_id, target_kind="predicate",
        target="profile.fifo_max_occupancy",
        required_context={
            "vendor": "amd", "tool": "vitis_hls", "tool_version": "2024.2",
            "stage": "cosim", "workload_id": {"required": True},
            "snapshot_association": "verified",
            "observation_evidence_qualified": (
                "derived_from_typed_observation_evidence_v1"
            ),
            "observation_instance_id": {"required": True},
            "observation_artifact_identity": {"required": True},
            "observation_run_identity": {"required": True},
        },
        producer="hlsgraph.builtin", producer_version="0.3",
    )
    closed_context = {
        **context,
        "snapshot_association": {"verified"},
        "observation_evidence_qualified": {
            "derived_from_typed_observation_evidence_v1",
        },
        "observation_instance_id": {"observation.current"},
        "observation_artifact_identity": {"artifact.current"},
        "observation_run_identity": {"run.current"},
    }
    assert HybridRetriever._binding_constraints_match_values(dynamic, closed_context, targets)

    unknown_operator = KnowledgeBinding(
        knowledge_rule_id=rule_id, target_kind="predicate",
        target="schedule.achieved_ii",
        required_context={
            "vendor": "amd", "tool": "vitis_hls",
            "tool_version": "2024.2", "stage": {"typo_equals": "schedule"},
        },
        producer="hlsgraph.builtin", producer_version="0.3",
        metadata={"dynamic_scope": "static"},
    )
    assert not HybridRetriever._binding_constraints_match_values(
        unknown_operator, context, targets,
    )


def test_source_tool_context_never_cross_pairs_tool_and_version(
    tmp_path: Path,
) -> None:
    (tmp_path / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    manifest = minimal_manifest(
        "test.tool_pairing", "tool pairing", "dut", "kernel.cpp",
    )
    manifest.toolchains = [
        ToolchainContext(
            "amd.vitis_hls.2023_2", "amd", "vitis_hls", "2023.2",
        ),
        ToolchainContext(
            "amd.vivado.2024_2", "amd", "vivado", "2024.2",
        ),
    ]
    bundle = GraphBundle.create(tmp_path, manifest)
    snapshot = bundle.snapshot()
    retriever = HybridRetriever(bundle, snapshot.id)
    base, toolchains = retriever._manifest_context()
    assert "tool" not in base and "tool_version" not in base
    source = retriever._source_tool_context(base, toolchains)
    assert source["tool"] == {
        "vitis_hls", "amd.vitis_hls.2023_2",
    }
    assert source["tool_version"] == {"2023.2"}
    assert "2024.2" not in source["tool_version"]

    manifest.toolchains.append(ToolchainContext(
        "amd.vitis_hls.2024_2", "amd", "vitis_hls", "2024.2",
    ))
    ambiguous = retriever._source_tool_context(
        {"vendor": {"amd"}}, {item.id: item for item in manifest.toolchains},
    )
    assert "tool" not in ambiguous
    assert "tool_version" not in ambiguous


def test_xdc_binding_uses_ledger_hash_and_snapshot_stage_association(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    (tmp_path / "timing.xdc").write_text(
        "create_clock -period 5 [get_ports ap_clk]\n", encoding="utf-8",
    )
    manifest = minimal_manifest("test.xdc.binding", "xdc binding", "dut", "kernel.cpp")
    manifest.constraints.xdc_files = ["timing.xdc"]
    manifest.stage_commands = {"post_route": ["vivado", "-mode", "batch"]}
    manifest.toolchains = [ToolchainContext(
        id="amd.vivado.2024_2", vendor="amd", name="vivado", version="2024.2",
    )]
    bundle = GraphBundle.create(tmp_path, manifest)
    install_reviewed_builtin_packs(bundle)
    snapshot = bundle.snapshot()
    graph = CanonicalGraph(snapshot.id)
    kernel = Entity("hls.kernel", "dut", snapshot.id, stage="ast")
    graph.add_entity(kernel)
    bundle.store.save_graph(graph)
    retriever = HybridRetriever(bundle, snapshot.id)
    contexts = retriever._binding_target_contexts(graph, {kernel.id})[
        ("artifact_kind", "constraint.xdc")
    ]
    assert len(contexts) == 1
    context = contexts[0]
    assert context["snapshot_association"] == {"verified"}
    assert context["snapshot_id"] == {snapshot.id.casefold()}
    assert context["artifact_sha256"]
    assert context["constraint_hash"] == {snapshot.constraint_hash}
    assert context["stage"] == {"post_route"}
    assert context["constraint_input_evidence_qualified"] == {
        "derived_from_unique_live_snapshot_input_v1",
    }
    assert context["constraint_artifact_identity"]

    binding = next(
        item for item in bundle.store.knowledge_bindings()
        if item.target_kind == "artifact_kind" and item.target == "constraint.xdc"
    )
    assert HybridRetriever._binding_constraints_match_values(
        binding, context, {"artifact_kind": {"constraint.xdc"}},
    )
    xdc_artifact = next(
        item for item in bundle.store.artifacts(snapshot.id)
        if item.kind == "constraint.xdc"
    )
    base, toolchains = retriever._manifest_context()
    altered_manifest = bundle.store.snapshot_manifest(snapshot.id)
    altered_manifest.toolchains[0].version = "2023.2"
    with monkeypatch.context() as scoped:
        scoped.setattr(
            bundle.store, "snapshot_manifest", lambda _snapshot_id: altered_manifest,
        )
        altered = retriever._artifact_context(xdc_artifact, base, toolchains)
    assert "constraint_input_evidence_qualified" not in altered
    assert "constraint_artifact_identity" not in altered
    incomplete = {key: set(value) for key, value in context.items()}
    incomplete.pop("artifact_sha256")
    assert not HybridRetriever._binding_constraints_match_values(
        binding, incomplete, {"artifact_kind": {"constraint.xdc"}},
    )
    (tmp_path / "timing.xdc").write_text(
        "create_clock -period 7 [get_ports ap_clk]\n", encoding="utf-8",
    )
    changed = retriever._binding_target_contexts(graph, {kernel.id})[
        ("artifact_kind", "constraint.xdc")
    ][0]
    assert "constraint_input_evidence_qualified" not in changed
    assert not HybridRetriever._binding_constraints_match_values(
        binding, changed, {"artifact_kind": {"constraint.xdc"}},
    )


def test_xdc_binding_rejects_duplicate_manifest_declaration(tmp_path: Path) -> None:
    (tmp_path / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    (tmp_path / "timing.xdc").write_text(
        "create_clock -period 5 [get_ports ap_clk]\n", encoding="utf-8",
    )
    manifest = minimal_manifest(
        "test.xdc.duplicate", "duplicate xdc", "dut", "kernel.cpp",
    )
    manifest.constraints.xdc_files = ["timing.xdc", "timing.xdc"]
    manifest.stage_commands = {"post_route": ["vivado", "-mode", "batch"]}
    manifest.toolchains = [ToolchainContext(
        id="amd.vivado.2024_2", vendor="amd", name="vivado", version="2024.2",
    )]
    bundle = GraphBundle.create(tmp_path, manifest)
    install_reviewed_builtin_packs(bundle)
    snapshot = bundle.snapshot()
    graph = CanonicalGraph(snapshot.id)
    kernel = graph.add_entity(Entity(
        "hls.kernel", "dut", snapshot.id, stage="ast",
    ))
    bundle.store.save_graph(graph)
    context = HybridRetriever(bundle, snapshot.id)._binding_target_contexts(
        graph, {kernel.id},
    )[("artifact_kind", "constraint.xdc")][0]
    assert "constraint_input_evidence_qualified" not in context
    binding = next(
        item for item in bundle.store.knowledge_bindings()
        if item.target_kind == "artifact_kind" and item.target == "constraint.xdc"
    )
    assert not HybridRetriever._binding_constraints_match_values(
        binding, context, {"artifact_kind": {"constraint.xdc"}},
    )


def test_xdc_context_never_cross_pairs_stage_and_self_described_tool(
    tmp_path: Path,
) -> None:
    (tmp_path / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    (tmp_path / "timing.xdc").write_text(
        "create_clock -period 5 [get_ports ap_clk]\n", encoding="utf-8",
    )
    manifest = minimal_manifest(
        "test.xdc.tool_pair", "xdc tool pair", "dut", "kernel.cpp",
    )
    manifest.constraints.xdc_files = ["timing.xdc"]
    manifest.stage_commands = {"post_route": ["vivado", "-mode", "batch"]}
    vivado_2023 = ToolchainContext(
        "amd.vivado.2023_2", "amd", "vivado", "2023.2",
    )
    vivado_2024 = ToolchainContext(
        "amd.vivado.2024_2", "amd", "vivado", "2024.2",
    )
    manifest.toolchains = [vivado_2023, vivado_2024]
    manifest.stage_toolchains = {"post_route": vivado_2023.id}
    bundle = GraphBundle.create(tmp_path, manifest)
    install_reviewed_builtin_packs(bundle)
    snapshot = bundle.snapshot()
    retriever = HybridRetriever(bundle, snapshot.id)
    base, toolchains = retriever._manifest_context()
    artifact = next(
        item for item in bundle.store.artifacts(snapshot.id)
        if item.kind == "constraint.xdc"
    )
    artifact.metadata.update({
        "vendor": "amd", "tool": "vivado",
        "tool_version": "2024.2", "version": "2024.2",
        "stage": "post_place",
    })
    context = retriever._artifact_context(artifact, base, toolchains)
    assert context["stage"] == {"post_route"}
    assert context["tool_version"] == {"2023.2"}
    assert "2024.2" not in context["tool_version"]
    binding = next(
        item for item in bundle.store.knowledge_bindings()
        if item.target_kind == "artifact_kind" and item.target == "constraint.xdc"
    )
    assert not HybridRetriever._binding_constraints_match_values(
        binding, context, {"artifact_kind": {"constraint.xdc"}},
    )

    manifest.stage_commands["post_place"] = ["vivado", "-mode", "batch"]
    manifest.stage_toolchains["post_place"] = vivado_2024.id
    split_root = tmp_path / "split"
    split_root.mkdir()
    (split_root / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    (split_root / "timing.xdc").write_text(
        "create_clock -period 5 [get_ports ap_clk]\n", encoding="utf-8",
    )
    split_bundle = GraphBundle.create(split_root, manifest)
    split_snapshot = split_bundle.snapshot()
    split_retriever = HybridRetriever(split_bundle, split_snapshot.id)
    split_base, split_toolchains = split_retriever._manifest_context()
    split_artifact = next(
        item for item in split_bundle.store.artifacts(split_snapshot.id)
        if item.kind == "constraint.xdc"
    )
    split_context = split_retriever._artifact_context(
        split_artifact, split_base, split_toolchains,
    )
    assert "tool" not in split_context
    assert "tool_version" not in split_context
    assert "stage" not in split_context


def test_amd_gate_binding_cannot_be_qualified_by_stage_only() -> None:
    loose = KnowledgeBinding(
        knowledge_rule_id="amd.ug906:2024.2:test.gate",
        target_kind="gate_kind", target="post_route_timing",
        required_context={
            "vendor": "amd", "tool": "vivado", "tool_version": "2024.2",
            "stage": "post_route",
        },
        producer="test.binding", producer_version="1",
        metadata={"dynamic_scope": "static"},
    )
    context = {
        "vendor": {"amd"}, "tool": {"vivado"},
        "tool_version": {"2024.2"}, "stage": {"post_route"},
    }
    assert not HybridRetriever._binding_constraints_match_values(
        loose, context, {"gate_kind": {"post_route_timing"}},
    )


def test_directive_binding_uses_only_one_explicit_instance_scope(
    retrieval_project: dict[str, object],
) -> None:
    bundle = retrieval_project["bundle"]
    snapshot_id = retrieval_project["snapshot"]
    graph = bundle.store.load_graph(snapshot_id)
    loop = Entity(
        kind="hls.loop", name="same_name", qualified_name="dut::same_name",
        snapshot_id=snapshot_id, stage="source",
    )
    good = Entity(
        kind="hls.directive", name="PIPELINE",
        qualified_name="kernel.cpp:10:PIPELINE", snapshot_id=snapshot_id,
        stage="source", authority=AuthorityClass.DECLARED_CONSTRAINT,
        attrs={"directive_kind": "PIPELINE", "options": {"ii": 1}},
    )
    good.attrs.update({
        "directive_instance_id": good.id,
        "scope_id": loop.id,
        "scope_kind": "hls.loop",
        "scope_resolution": "source_ast",
        "loop_id": loop.id,
    })
    copied = Entity(
        kind="hls.directive", name="PIPELINE",
        qualified_name="kernel.cpp:20:PIPELINE", snapshot_id=snapshot_id,
        stage="source", authority=AuthorityClass.DECLARED_CONSTRAINT,
        attrs={
            "directive_kind": "PIPELINE", "options": {"ii": 2},
            # Deliberately copied from another record.  Neither the matching
            # name nor the ANNOTATES relation below may repair this self-ID.
            "directive_instance_id": good.id,
            "scope_id": loop.id,
            "scope_kind": "hls.loop",
            "scope_resolution": "source_ast",
            "loop_id": loop.id,
        },
    )
    degraded = Entity(
        kind="hls.directive", name="PIPELINE",
        qualified_name="kernel.cpp:30:PIPELINE", snapshot_id=snapshot_id,
        stage="source", authority=AuthorityClass.DECLARED_CONSTRAINT,
        attrs={"directive_kind": "PIPELINE", "options": {"ii": 3}},
    )
    degraded.attrs.update({
        "directive_instance_id": degraded.id,
        "scope_id": loop.id,
        "scope_kind": "hls.loop",
        "scope_resolution": "regex_degraded",
        "loop_id": loop.id,
    })
    inconsistent = Entity(
        kind="hls.directive", name="PIPELINE",
        qualified_name="kernel.cpp:40:PIPELINE", snapshot_id=snapshot_id,
        stage="source", authority=AuthorityClass.DECLARED_CONSTRAINT,
        attrs={"directive_kind": "PIPELINE", "options": {"ii": 4}},
    )
    inconsistent.attrs.update({
        "directive_instance_id": inconsistent.id,
        "scope_id": loop.id,
        "scope_kind": "hls.loop",
        "scope_resolution": "source_ast",
        "loop_id": retrieval_project["kernel"],
    })
    for entity in (loop, good, copied, degraded, inconsistent):
        graph.add_entity(entity)
    graph.add_relation(Relation(
        copied.id, loop.id, "hls.annotates", snapshot_id, stage="source",
        authority=AuthorityClass.DECLARED_CONSTRAINT,
        attrs={"scope_node_id": loop.id, "scope_resolution": "source_ast"},
    ))

    retriever = HybridRetriever(bundle, snapshot_id)
    contexts = retriever._binding_target_contexts(graph, set(graph.entities))[
        ("directive_kind", "PIPELINE")
    ]
    good_context = next(item for item in contexts
                        if item.get("directive_instance_id") == {good.id.casefold()})
    copied_context = next(item for item in contexts
                          if "directive_instance_id" not in item)
    degraded_context = next(item for item in contexts
                            if item.get("directive_instance_id") == {degraded.id.casefold()})
    inconsistent_context = next(
        item for item in contexts
        if item.get("directive_instance_id") == {inconsistent.id.casefold()}
    )
    assert good_context["scope_id"] == {loop.id.casefold()}
    assert good_context["loop_id"] == {loop.id.casefold()}
    assert "scope_id" not in copied_context and "loop_id" not in copied_context

    binding = next(
        item for item in bundle.store.knowledge_bindings()
        if item.target_kind == "directive_kind" and item.target == "PIPELINE"
        and item.required_context.get("loop_id") == {"required": True}
    )
    targets = {"directive_kind": {"PIPELINE"}}
    # Exact scope identity is necessary but no longer sufficient: this manual
    # fixture has no uniquely anchored, live directive.requested record.
    assert "directive_source_declaration_qualified" not in good_context
    assert not HybridRetriever._binding_constraints_match_values(binding, good_context, targets)
    assert not HybridRetriever._binding_constraints_match_values(binding, copied_context, targets)
    assert not HybridRetriever._binding_constraints_match_values(binding, degraded_context, targets)
    assert not HybridRetriever._binding_constraints_match_values(binding, inconsistent_context, targets)


def test_binding_context_does_not_mix_unrelated_stage_or_workload(
    retrieval_project: dict[str, object],
) -> None:
    bundle = retrieval_project["bundle"]
    snapshot = retrieval_project["snapshot"]
    target = Observation(
        snapshot_id=snapshot, subject_id=retrieval_project["region"],
        predicate="profile.fifo_max_occupancy", value=4,
        stage="post_synth", authority=AuthorityClass.TOOL_OBSERVATION,
    )
    unrelated = Observation(
        snapshot_id=snapshot, subject_id=retrieval_project["sink"],
        predicate="profile.unrelated_cosim", value=1,
        stage="cosim", authority=AuthorityClass.TOOL_OBSERVATION,
        workload_id="tb.default",
    )
    bundle.store.add_observations([target, unrelated])
    graph = bundle.store.load_graph(snapshot)
    retriever = HybridRetriever(bundle, snapshot)
    contexts = retriever._binding_target_contexts(graph, set(graph.entities))[
        ("predicate", "profile.fifo_max_occupancy")
    ]
    assert len(contexts) == 1
    assert contexts[0]["stage"] == {"post_synth"}
    assert "workload_id" not in contexts[0]

    rule_id = bundle.store.knowledge_rules()[0].id
    binding = KnowledgeBinding(
        knowledge_rule_id=rule_id, target_kind="predicate",
        target="profile.fifo_max_occupancy",
        required_context={
            "vendor": "amd", "tool": "vitis_hls", "tool_version": "2024.2",
            "stage": "cosim", "workload_id": {"required": True},
        },
        producer="hlsgraph.builtin", producer_version="0.3",
    )
    targets = {"predicate": {"profile.fifo_max_occupancy"}}
    assert not any(
        HybridRetriever._binding_constraints_match_values(binding, context, targets)
        for context in contexts
    )

    ir_contexts = retriever._binding_target_contexts(graph, set(graph.entities))[
        ("entity_kind", "ir.llvm.block")
    ]
    assert ir_contexts and all(
        item["stage"] == {"llvm"} and item["ir"] == {"llvm"}
        for item in ir_contexts
    )

    pinned_toolchain = ToolchainContext(
        id="amd.vitis_hls.pinned", vendor="amd", name="vitis_hls",
        version="2023.1",
    )
    conflicting_run = ToolRun(
        snapshot_id=snapshot, stage="csynth", backend="runner.local",
        request_hash="b" * 64, toolchain_id=pinned_toolchain.id,
        metadata={
            "tool": "vitis_hls", "tool_version": "2024.2",
            "version": "2024.2", "authority": "tool_observation",
        },
    )
    pinned = retriever._run_context(
        conflicting_run, {"vendor": {"amd"}},
        {pinned_toolchain.id: pinned_toolchain},
    )
    assert pinned["tool_version"] == {"2023.1"}
    assert pinned["version"] == {"2023.1"}


def test_open_ir_guidance_does_not_infer_language_or_artifact_revision(
    retrieval_project: dict[str, object],
) -> None:
    result = CoreService(
        retrieval_project["bundle"], retrieval_project["snapshot"],
    ).retrieve(RetrievalSpec(
        query="LLVM CFG low level control evidence", view="evidence",
    ))
    assert not any(
        item.citation and item.citation["document_id"] == "llvm.ir.langref"
        for item in result.guidance
    )
    assert "knowledge_binding_artifact_revision_unbound" not in result.warnings


def test_open_ir_context_cannot_mint_language_or_artifact_revision(
    retrieval_project: dict[str, object],
) -> None:
    assert not any(
        item for item in retrieval_project["bundle"].store.knowledge_bindings()
        if (item.target_kind, item.target) == ("entity_kind", "ir.llvm.block")
    )

    generic: dict[str, set[str]] = {}
    HybridRetriever._context_metadata(generic, {
        "artifact_revision": "must-not-flow-through-generic-context",
        "language_spec_revision": (
            "git-429c88d37f1f02e68ebc1fc7b0da4511ce6407e3"
        ),
        "language_spec_revision_source": "immutable_extractor_attestation.v1",
        "gate_evidence_qualified": "derived_from_typed_evidence_v1",
    })
    assert "artifact_revision" not in generic
    assert "language_spec_revision" not in generic
    assert "language_spec_revision_source" not in generic
    assert "gate_evidence_qualified" not in generic
    instance_local: dict[str, set[str]] = {}
    HybridRetriever._context_projection_metadata(instance_local, {
        "artifact_revision": "sha256:fixture-llvm-ir",
        "adapter_version": "adapter.v3",
        "projection_mapping": "mapping.fixture.v1",
        "operation": "llvm.add",
        "language_spec_revision": (
            "git-429c88d37f1f02e68ebc1fc7b0da4511ce6407e3"
        ),
    })
    assert instance_local == {
        "projection_mapping": {"mapping.fixture.v1"},
        "operation": {"llvm.add"},
    }

    graph = CanonicalGraph("snapshot_projection_context")
    block = graph.add_entity(Entity(
        kind="ir.llvm.block", name="entry",
        snapshot_id=graph.snapshot_id, stage="llvm",
        anchors=[SourceAnchor("artifact_ir"), SourceAnchor("artifact_source")],
    ))
    anchored: dict[str, set[str]] = {}
    HybridRetriever(
        retrieval_project["bundle"], retrieval_project["snapshot"],
    )._context_unique_anchor_artifact(
        anchored, (block.id,), graph, {
            "artifact_ir": SimpleNamespace(kind="ir.llvm", metadata={
                "artifact_revision": "sha256:fixture-llvm-ir",
                "adapter_version": "adapter.v3",
                "language_spec_contracts": [{
                    "family": "llvm",
                    "revision": (
                        "git-429c88d37f1f02e68ebc1fc7b0da4511ce6407e3"
                    ),
                    "compatibility_contract": (
                        "hlsgraph.llvm.language_spec_compatibility.v1"
                    ),
                }],
            }),
            "artifact_source": SimpleNamespace(kind="source.cpp", metadata={}),
        },
    )
    assert anchored == {}

    conflicting: dict[str, set[str]] = {}
    HybridRetriever._context_metadata(
        conflicting, {
            "language_spec_contracts": [{
                "family": "llvm",
                "revision": (
                    "git-429c88d37f1f02e68ebc1fc7b0da4511ce6407e3"
                ),
                "compatibility_contract": (
                    "hlsgraph.llvm.language_spec_compatibility.v1"
                ),
            }, {
                "family": "llvm",
                "revision": "git-different",
                "compatibility_contract": (
                    "hlsgraph.llvm.language_spec_compatibility.v1"
                ),
            }],
        },
    )
    assert not conflicting

    artifact_revision_only: dict[str, set[str]] = {}
    HybridRetriever._context_metadata(
        artifact_revision_only,
        {
            "artifact_revision": "sha256:not-a-language-revision",
        },
    )
    assert not artifact_revision_only

    wrong_artifact_kind: dict[str, set[str]] = {}
    HybridRetriever._context_metadata(
        wrong_artifact_kind,
        {
            "language_spec_contracts": [{
                "family": "llvm",
                "revision": (
                    "git-429c88d37f1f02e68ebc1fc7b0da4511ce6407e3"
                ),
                "compatibility_contract": (
                    "hlsgraph.llvm.language_spec_compatibility.v1"
                ),
            }],
        },
    )
    assert not wrong_artifact_kind


def test_open_ir_condition_tokens_are_typed_and_current_instance_only(
    retrieval_project: dict[str, object],
) -> None:
    graph = CanonicalGraph("snapshot_open_ir_conditions")
    module = graph.add_entity(Entity(
        kind="ir.llvm.module", name="unit.ll",
        snapshot_id=graph.snapshot_id, stage="llvm",
    ))
    block = graph.add_entity(Entity(
        kind="ir.llvm.block", name="entry",
        snapshot_id=graph.snapshot_id, stage="llvm",
    ))
    add = graph.add_entity(Entity(
        kind="ir.llvm.operation", name="add",
        snapshot_id=graph.snapshot_id, stage="llvm",
        attrs={"opcode": "add", "bitwidths": [32], "memory_access": False},
    ))
    mlir_op = graph.add_entity(Entity(
        kind="ir.mlir.operation", name="arith.addi",
        snapshot_id=graph.snapshot_id, stage="mlir",
        attrs={"operation": "arith.addi", "dialect": "arith"},
    ))
    target_anchor = SourceAnchor(
        "artifact_source", start_line=1, start_column=1,
        end_line=20, end_column=1, symbol="dut",
    )
    source = graph.add_entity(Entity(
        kind="hls.function", name="dut",
        snapshot_id=graph.snapshot_id, stage="ast",
        anchors=[target_anchor],
    ))

    module_context: dict[str, set[str]] = {}
    HybridRetriever._context_entity_evidence(
        module_context, module, current=True,
    )
    assert module_context["llvm_container_present"] == {"true"}
    assert "basic_blocks_or_branches_present" not in module_context

    block_context: dict[str, set[str]] = {}
    HybridRetriever._context_entity_evidence(
        block_context, block, current=True,
    )
    assert block_context["basic_blocks_or_branches_present"] == {"true"}

    operation_context: dict[str, set[str]] = {}
    HybridRetriever._context_entity_evidence(
        operation_context, add, current=True,
    )
    assert operation_context["llvm_instruction_present"] == {"true"}
    assert operation_context["explicit_integer_width_present"] == {"true"}
    assert "memory_instruction_present" not in operation_context

    untyped = Relation(
        mlir_op.id, source.id, "cross.maps_to", graph.snapshot_id,
        stage="mlir", mapping_kind="mlir.location",
        attrs={"hardware_topology": False},
    )
    graph.add_relation(untyped)
    untyped_context: dict[str, set[str]] = {}
    HybridRetriever._context_relation_evidence(
        untyped_context, untyped, graph, current=True,
    )
    assert "typed_mlir_location_present" not in untyped_context
    assert "mapping_provenance" not in untyped_context

    location = SourceAnchor(
        "artifact_source", start_line=7, start_column=3,
        ir_location='loc("kernel.cpp":7:3)',
        mapping_kind="mlir.filelinecol",
    )
    mapping_attrs = {
        "cardinality": "many_to_many",
        "hardware_topology": False,
        "mapping_ambiguous": False,
        "mapping_candidate_count": 1,
        "mapping_provenance": "mlir.location_anchor",
        "mapping_redacted": False,
        "mapping_resolution": "unique_exact",
        "mapping_resolution_contract": "hlsgraph.mlir_location_resolution.v1",
        "mapping_unresolved": False,
        "resolved_target_anchor_identity": stable_hash(target_anchor),
        "resolved_target_id": source.id,
        "source_anchor_identity_contract": "hlsgraph.source_anchor_identity.v1",
        "target_layer": "source_ast",
        "typed_source_anchor_identity": stable_hash(location),
    }
    typed = Relation(
        mlir_op.id, source.id, "cross.maps_to", graph.snapshot_id,
        stage="mlir", mapping_kind="mlir.location",
        attrs=mapping_attrs,
        anchors=[location, target_anchor],
    )
    graph.add_relation(typed)
    typed_context: dict[str, set[str]] = {"ir": {"mlir"}}
    HybridRetriever._context_relation_evidence(
        typed_context, typed, graph, current=True,
    )
    assert typed_context["typed_mlir_location_present"] == {"true"}
    assert typed_context["mapping_provenance"] == {"mlir.location_anchor"}
    assert typed_context["location_kind"] == {"mlir.filelinecol"}
    assert typed_context["mapping_resolution"] == {"unique_exact"}
    assert typed_context["mapping_resolution_contract"] == {
        "hlsgraph.mlir_location_resolution.v1"
    }
    assert typed_context["unique_mlir_location_mapping_resolved"] == {"true"}
    assert typed_context["typed_source_anchor_identity"] == {
        stable_hash(location)
    }
    assert typed_context["resolved_target_anchor_identity"] == {
        stable_hash(target_anchor)
    }
    assert typed_context["resolved_target_id"] == {source.id.casefold()}
    assert typed_context["source_anchor_identity_contract"] == {
        "hlsgraph.source_anchor_identity.v1"
    }

    cited_context: dict[str, set[str]] = {}
    HybridRetriever._context_relation_evidence(
        cited_context, typed, graph, current=False,
    )
    assert "unique_mlir_location_mapping_resolved" not in cited_context
    assert "typed_source_anchor_identity" not in cited_context

    assert not any(
        item.target_kind == "relation_kind" and item.target == "cross.maps_to"
        for item in retrieval_project["bundle"].store.knowledge_bindings()
    )

    spoofed = Relation(
        mlir_op.id, source.id, "cross.maps_to", graph.snapshot_id,
        stage="mlir", mapping_kind="mlir.location",
        attrs={
            **mapping_attrs,
            "typed_mlir_location_present": True,
            "unique_mlir_location_mapping_resolved": True,
        },
    )
    spoofed_context: dict[str, set[str]] = {"ir": {"mlir"}}
    HybridRetriever._context_relation_evidence(
        spoofed_context, spoofed, graph, current=True,
    )
    HybridRetriever._context_metadata(
        spoofed_context, spoofed.attrs,
    )
    assert "unique_mlir_location_mapping_resolved" not in spoofed_context
    assert "typed_mlir_location_present" not in spoofed_context
    assert "typed_source_anchor_identity" not in spoofed_context

    tampered_identity = Relation(
        mlir_op.id, source.id, "cross.maps_to", graph.snapshot_id,
        stage="mlir", mapping_kind="mlir.location",
        attrs={**mapping_attrs, "typed_source_anchor_identity": "f" * 64},
        anchors=[location, target_anchor],
    )
    tampered_context: dict[str, set[str]] = {"ir": {"mlir"}}
    HybridRetriever._context_relation_evidence(
        tampered_context, tampered_identity, graph, current=True,
    )
    assert "unique_mlir_location_mapping_resolved" not in tampered_context
    assert "typed_source_anchor_identity" not in tampered_context

    missing_target_anchor = Relation(
        mlir_op.id, source.id, "cross.maps_to", graph.snapshot_id,
        stage="mlir", mapping_kind="mlir.location",
        attrs=mapping_attrs, anchors=[location],
    )
    missing_target_context: dict[str, set[str]] = {"ir": {"mlir"}}
    HybridRetriever._context_relation_evidence(
        missing_target_context, missing_target_anchor, graph, current=True,
    )
    assert "unique_mlir_location_mapping_resolved" not in missing_target_context

    overlapping = graph.add_entity(Entity(
        kind="hls.function", name="also_dut",
        snapshot_id=graph.snapshot_id, stage="ast",
        anchors=[SourceAnchor(
            "artifact_source", start_line=1, start_column=1,
            end_line=20, end_column=1, symbol="also_dut",
        )],
    ))
    assert overlapping.id != source.id
    ambiguous_context: dict[str, set[str]] = {"ir": {"mlir"}}
    HybridRetriever._context_relation_evidence(
        ambiguous_context, typed, graph, current=True,
    )
    assert "unique_mlir_location_mapping_resolved" not in ambiguous_context
    assert "typed_source_anchor_identity" not in ambiguous_context

    redacted = Relation(
        mlir_op.id, source.id, "cross.maps_to", graph.snapshot_id,
        stage="mlir", mapping_kind="mlir.location",
        attrs=mapping_attrs,
        anchors=[SourceAnchor(
            "artifact_source", start_line=7, start_column=3,
            ir_location='loc("<external>":7:3)',
            mapping_kind="mlir.filelinecol.redacted",
            ambiguity="external source path was redacted",
        ), target_anchor],
    )
    redacted_context: dict[str, set[str]] = {"ir": {"mlir"}}
    HybridRetriever._context_relation_evidence(
        redacted_context, redacted, graph, current=True,
    )
    HybridRetriever._context_metadata(
        redacted_context, redacted.attrs,
    )
    assert "unique_mlir_location_mapping_resolved" not in redacted_context
    assert "typed_source_anchor_identity" not in redacted_context

    unknown = Relation(
        mlir_op.id, source.id, "cross.maps_to", graph.snapshot_id,
        stage="mlir", mapping_kind="mlir.location",
        attrs={"hardware_topology": False},
        anchors=[SourceAnchor(
            "artifact_source", ir_location="loc(unknown)",
            mapping_kind="mlir.unknown",
        )],
    )
    graph.add_relation(unknown)
    unknown_context: dict[str, set[str]] = {"ir": {"mlir"}}
    HybridRetriever._context_relation_evidence(
        unknown_context, unknown, graph, current=True,
    )
    assert "typed_mlir_location_present" not in unknown_context
    assert "mapping_provenance" not in unknown_context

    hardware_target = graph.add_entity(Entity(
        kind="hls.process", name="projected",
        snapshot_id=graph.snapshot_id, stage="mlir",
    ))
    hardware_projection = Relation(
        mlir_op.id, hardware_target.id, "cross.projects_to", graph.snapshot_id,
        stage="mlir", attrs={
            "projection_mapping": "dialect_semantics",
            "hardware_topology": False,
            "hardware_projection": True,
        },
    )
    graph.add_relation(hardware_projection)
    hardware_context: dict[str, set[str]] = {"ir": {"mlir"}}
    HybridRetriever._context_relation_evidence(
        hardware_context, hardware_projection, graph, current=True,
    )
    assert "mlir_operation_present" not in hardware_context

    ir_target = add
    evidence_projection = Relation(
        mlir_op.id, ir_target.id, "cross.projects_to", graph.snapshot_id,
        stage="mlir", attrs={
            "projection_mapping": "dialect_semantics",
            "hardware_topology": False,
        },
    )
    graph.add_relation(evidence_projection)
    evidence_context: dict[str, set[str]] = {"ir": {"mlir"}}
    HybridRetriever._context_relation_evidence(
        evidence_context, evidence_projection, graph, current=True,
    )
    assert "mlir_operation_present" not in evidence_context
    assert "projection_mapping" not in evidence_context
    assert evidence_context["target_entity_kind"] == {"ir.llvm.operation"}


def test_operation_histogram_binding_requires_recomputed_qualified_aggregate(
    retrieval_project: dict[str, object],
) -> None:
    graph = CanonicalGraph("snapshot_histogram_contract")
    artifact = ArtifactRef(
        kind="ir.mlir", uri="ir/dut.mlir", sha256="a" * 64, size=16,
        access="project", metadata={
            "artifact_revision": "sha256:fixture-mlir",
            "language_spec_contracts": [{
                "family": "mlir",
                "revision": (
                    "git-429c88d37f1f02e68ebc1fc7b0da4511ce6407e3"
                ),
                "compatibility_contract": (
                    "hlsgraph.mlir.language_spec_compatibility.v1"
                ),
            }],
        },
    )
    function = graph.add_entity(Entity(
        kind="ir.mlir.function", name="dut",
        snapshot_id=graph.snapshot_id, stage="mlir",
        anchors=[SourceAnchor(artifact.id)],
    ))
    add = graph.add_entity(Entity(
        kind="ir.mlir.operation", name="arith.addi",
        snapshot_id=graph.snapshot_id, stage="mlir",
        attrs={"operation": "arith.addi", "dialect": "arith"},
        anchors=[SourceAnchor(artifact.id)],
    ))
    relation = graph.add_relation(Relation(
        function.id, add.id, "ir.contains", graph.snapshot_id, stage="mlir",
    ))
    derivation = {
        "id": "derivation_histogram_valid",
        "snapshot_id": graph.snapshot_id,
        "subject_id": function.id,
        "predicate": "feature.operation_histogram",
        "value": {"arith.addi": 1},
        "algorithm": "hlsgraph.static.operation_histogram",
        "algorithm_version": "1",
        "stage": "mlir",
        "completeness": "complete",
        "metadata": {
            "operation_histogram_schema": (
                "mlir.dialect_qualified_opcode_histogram.v1"
            ),
            "operation_histogram_provenance": "typed_ir_entity_evidence.v1",
            "operation_histogram_domain_complete": True,
        },
        "evidence_refs": [
            {"kind": "entity_anchor", "target_id": function.id,
             "snapshot_id": graph.snapshot_id},
            {"kind": "entity_anchor", "target_id": add.id,
             "snapshot_id": graph.snapshot_id},
            {"kind": "relation", "target_id": relation.id,
             "snapshot_id": graph.snapshot_id},
            {"kind": "artifact", "target_id": artifact.id,
             "snapshot_id": graph.snapshot_id},
        ],
    }
    context: dict[str, set[str]] = {"ir": {"mlir"}, "stage": {"mlir"}}
    retriever = HybridRetriever(
        retrieval_project["bundle"], retrieval_project["snapshot"],
    )
    retriever._context_derivation_evidence(
        context, derivation, graph, {artifact.id: artifact},
    )
    assert "dialect_qualified_operation_histogram_present" not in context
    assert "operation_histogram_schema" not in context
    assert "artifact_identity" not in context
    assert "evidence_origin_identity" not in context

    assert not any(
        item.target_kind == "predicate"
        and item.target == "feature.operation_histogram"
        and item.required_context.get("stage") == "mlir"
        for item in retrieval_project["bundle"].store.knowledge_bindings()
    )

    for mutation in (
        {"value": {"addi": 1}},
        {"completeness": "partial"},
        {"metadata": {
            **derivation["metadata"],
            "operation_histogram_schema": "flattened.v1",
        }},
    ):
        invalid = {**derivation, **mutation}
        invalid_context: dict[str, set[str]] = {
            "ir": {"mlir"}, "stage": {"mlir"},
        }
        retriever._context_derivation_evidence(
            invalid_context, invalid, graph, {artifact.id: artifact},
        )
        assert "dialect_qualified_operation_histogram_present" not in invalid_context


@pytest.mark.parametrize(
    ("predicate", "value", "metadata", "condition"),
    [
        (
            "feature.index_histogram", {"dynamic": 1}, {
                "index_histogram_schema": (
                    "llvm.explicit_index_operand_kind_histogram.v1"
                ),
                "index_histogram_provenance": "typed_ir_entity_evidence.v1",
                "index_operand_definition": (
                    "llvm.gep_extract_insert_explicit_operand.v1"
                ),
                "index_histogram_domain_complete": True,
            }, "typed_index_histogram_present",
        ),
        (
            "feature.bitwidth", {"32": 2, "64": 1}, {
                "bitwidth_schema": (
                    "llvm.explicit_integer_width_occurrence_histogram.v1"
                ),
                "bitwidth_provenance": "typed_ir_entity_evidence.v1",
                "bitwidth_definition": (
                    "llvm.explicit_integer_type_occurrence.v1"
                ),
                "bitwidth_domain_complete": True,
            }, "typed_bitwidth_histogram_present",
        ),
        (
            "feature.memory_access", {"address": 1, "load": 1}, {
                "memory_access_schema": "llvm.memory_access_kind_histogram.v1",
                "memory_access_provenance": "typed_ir_entity_evidence.v1",
                "memory_access_opcode_definition": (
                    "llvm.load_store_gep_atomic_fence.v1"
                ),
                "memory_access_domain_complete": True,
            }, "typed_memory_access_histogram_present",
        ),
    ],
)
def test_llvm_feature_bindings_require_recomputed_schema_and_artifact_identity(
    retrieval_project: dict[str, object], predicate: str,
    value: dict[str, int], metadata: dict[str, object], condition: str,
) -> None:
    graph = CanonicalGraph("snapshot_llvm_feature_contract")
    artifact = ArtifactRef(
        kind="ir.llvm", uri="ir/dut.ll", sha256="b" * 64, size=32,
        access="project", producer_run_id="run.fixture.llvm",
        metadata={
            "artifact_revision": "sha256:fixture-llvm",
            "language_spec_contracts": [{
                "family": "llvm",
                "revision": (
                    "git-429c88d37f1f02e68ebc1fc7b0da4511ce6407e3"
                ),
                "compatibility_contract": (
                    "hlsgraph.llvm.language_spec_compatibility.v1"
                ),
            }],
        },
    )
    function = graph.add_entity(Entity(
        kind="ir.llvm.function", name="dut",
        snapshot_id=graph.snapshot_id, stage="llvm",
        anchors=[SourceAnchor(artifact.id)],
    ))
    gep = graph.add_entity(Entity(
        kind="ir.llvm.operation", name="getelementptr",
        snapshot_id=graph.snapshot_id, stage="llvm", attrs={
            "opcode": "getelementptr", "index_kinds": ["dynamic"],
            "bitwidths": [32, 64], "memory_access_kind": "address",
        },
        anchors=[SourceAnchor(artifact.id)],
    ))
    load = graph.add_entity(Entity(
        kind="ir.llvm.operation", name="load",
        snapshot_id=graph.snapshot_id, stage="llvm", attrs={
            "opcode": "load", "bitwidths": [32],
            "memory_access_kind": "load",
        },
        anchors=[SourceAnchor(artifact.id)],
    ))
    relations = [
        graph.add_relation(Relation(
            function.id, operation.id, "ir.contains", graph.snapshot_id,
            stage="llvm",
        )) for operation in (gep, load)
    ]
    evidence_refs = [
        *(
            {"kind": "entity_anchor", "target_id": entity.id,
             "snapshot_id": graph.snapshot_id}
            for entity in (function, gep, load)
        ),
        *(
            {"kind": "relation", "target_id": relation.id,
             "snapshot_id": graph.snapshot_id}
            for relation in relations
        ),
        {"kind": "artifact", "target_id": artifact.id,
         "snapshot_id": graph.snapshot_id},
    ]
    derivation = {
        "id": f"derivation_{predicate.removeprefix('feature.')}",
        "snapshot_id": graph.snapshot_id,
        "subject_id": function.id,
        "predicate": predicate,
        "value": value,
        "algorithm": f"hlsgraph.static.{predicate.removeprefix('feature.')}",
        "algorithm_version": "1",
        "stage": "llvm",
        "completeness": "complete",
        "metadata": metadata,
        "evidence_refs": evidence_refs,
    }
    context: dict[str, set[str]] = {"ir": {"llvm"}, "stage": {"llvm"}}
    retriever = HybridRetriever(
        retrieval_project["bundle"], retrieval_project["snapshot"],
    )
    retriever._context_derivation_evidence(
        context, derivation, graph, {artifact.id: artifact},
    )
    assert condition not in context
    assert "artifact_identity" not in context
    assert "evidence_origin_identity" not in context

    assert not any(
        item.target_kind == "predicate" and item.target == predicate
        for item in retrieval_project["bundle"].store.knowledge_bindings()
    )

    wrong_value = {**derivation, "value": {"tampered": 1}}
    wrong_context: dict[str, set[str]] = {
        "ir": {"llvm"}, "stage": {"llvm"},
    }
    retriever._context_derivation_evidence(
        wrong_context, wrong_value, graph, {artifact.id: artifact},
    )
    assert condition not in wrong_context

    missing_contract = {
        **derivation,
        "metadata": {
            key: item for key, item in metadata.items()
            if not key.endswith("_domain_complete")
        },
    }
    missing_context: dict[str, set[str]] = {
        "ir": {"llvm"}, "stage": {"llvm"},
    }
    retriever._context_derivation_evidence(
        missing_context, missing_contract, graph, {artifact.id: artifact},
    )
    assert condition not in missing_context

    no_artifact = {
        **derivation,
        "evidence_refs": [
            item for item in evidence_refs if item["kind"] != "artifact"
        ],
    }
    no_artifact_context: dict[str, set[str]] = {
        "ir": {"llvm"}, "stage": {"llvm"},
    }
    retriever._context_derivation_evidence(
        no_artifact_context, no_artifact, graph, {artifact.id: artifact},
    )
    assert condition not in no_artifact_context
    assert "artifact_identity" not in no_artifact_context


def test_handshake_ir_evidence_is_structural_and_hls_projection_is_not_normative(
    retrieval_project: dict[str, object],
) -> None:
    bindings = retrieval_project["bundle"].store.knowledge_bindings()
    assert not any(
        item.target_kind == "relation_kind" and item.target == "handshake.dataflow"
        for item in bindings
    )
    assert not any(
        (item.target_kind, item.target) in {
            ("entity_kind", "hls.process"),
            ("entity_kind", "hls.buffer"),
            ("relation_kind", "hls.streams_to"),
        }
        for item in bindings
    )
    graph = CanonicalGraph("snapshot_handshake_contract")
    artifact = ArtifactRef(
        kind="ir.mlir", uri="ir/handshake.mlir", sha256="c" * 64, size=32,
        access="project", metadata={
            "artifact_revision": "sha256:fixture-handshake-ir",
            "language_spec_contracts": [{
                "family": "circt.handshake",
                "revision": (
                    "git-ef03d45c960607315a8b62903b92d072d8542e30"
                ),
                "compatibility_contract": (
                    "hlsgraph.circt.handshake_spec_compatibility.v1"
                ),
            }],
        },
    )
    source = graph.add_entity(Entity(
        kind="ir.mlir.operation", name="handshake.buffer",
        snapshot_id=graph.snapshot_id, stage="mlir",
        attrs={
            "operation": "handshake.buffer", "dialect": "handshake",
            "ssa_result": "%0", "ssa_operands": [],
        },
        anchors=[SourceAnchor(artifact.id)],
    ))
    target = graph.add_entity(Entity(
        kind="ir.mlir.operation", name="handshake.return",
        snapshot_id=graph.snapshot_id, stage="mlir",
        attrs={
            "operation": "handshake.return", "dialect": "handshake",
            "ssa_result": None, "ssa_operands": ["%0"],
        },
        anchors=[SourceAnchor(artifact.id)],
    ))
    relation = graph.add_relation(Relation(
        source.id, target.id, "handshake.dataflow", graph.snapshot_id,
        stage="mlir", attrs={
            "hardware_topology": False,
            "native_ir_artifact_id": artifact.id,
            "native_ir_evidence": True,
            "native_ir_evidence_contract": "hlsgraph.mlir.ssa_def_use.v1",
            "native_ir_relation_provenance": "mlir.ssa_def_use",
            "ssa_value": "%0",
        },
        anchors=[SourceAnchor(artifact.id)],
    ))

    complete: dict[str, set[str]] = {"ir": {"mlir"}, "stage": {"mlir"}}
    HybridRetriever._context_relation_evidence(
        complete, relation, graph, current=True,
    )
    for metadata in (relation.attrs, source.attrs, target.attrs):
        HybridRetriever._context_metadata(complete, metadata)
        HybridRetriever._context_projection_metadata(complete, metadata)
    retriever = HybridRetriever(
        retrieval_project["bundle"], retrieval_project["snapshot"],
    )
    retriever._context_unique_anchor_artifact(
        complete, (source.id, target.id), graph, {artifact.id: artifact},
        relation.anchors,
    )
    assert complete["hardware_topology"] == {"false"}
    assert complete["native_ir_evidence"] == {"true"}
    assert complete["native_ir_evidence_contract"] == {
        "hlsgraph.mlir.ssa_def_use.v1"
    }
    assert complete["native_ir_relation_provenance"] == {"mlir.ssa_def_use"}
    assert complete["native_ir_artifact_identity"] == {artifact.id.casefold()}
    assert "language_spec_family" not in complete
    assert "language_spec_revision" not in complete

    cited: dict[str, set[str]] = {"ir": {"mlir"}, "stage": {"mlir"}}
    HybridRetriever._context_relation_evidence(
        cited, relation, graph, current=False,
    )
    assert "native_ir_evidence" not in cited
    assert "handshake_operation_present" not in cited

    spoofed = Relation(
        source.id, target.id, "handshake.dataflow", graph.snapshot_id,
        stage="mlir", attrs={
            **relation.attrs,
            "language_spec_revision": (
                "git-ef03d45c960607315a8b62903b92d072d8542e30"
            ),
        },
    )
    spoofed_context: dict[str, set[str]] = {
        "ir": {"mlir"}, "stage": {"mlir"},
    }
    HybridRetriever._context_relation_evidence(
        spoofed_context, spoofed, graph, current=True,
    )
    HybridRetriever._context_metadata(spoofed_context, spoofed.attrs)
    assert "native_ir_evidence" not in spoofed_context
    assert "language_spec_revision" not in spoofed_context

    wrong_revision_artifact = SimpleNamespace(
        id=artifact.id, kind="ir.mlir", metadata={
            "artifact_revision": "sha256:fixture-handshake-ir",
            "language_spec_contracts": [{
                "family": "circt.handshake",
                "revision": "git-different",
                "compatibility_contract": (
                    "hlsgraph.circt.handshake_spec_compatibility.v1"
                ),
            }],
        },
    )
    wrong_revision: dict[str, set[str]] = {
        key: set(value) for key, value in complete.items()
        if not key.startswith("language_spec_")
    }
    retriever._context_unique_anchor_artifact(
        wrong_revision, (source.id, target.id), graph,
        {artifact.id: wrong_revision_artifact}, relation.anchors,
    )
    assert "language_spec_revision" not in wrong_revision


def test_unreviewed_knowledge_is_inert_and_predictions_remain_separate(
    retrieval_project: dict[str, object],
) -> None:
    core = CoreService(retrieval_project["bundle"], retrieval_project["snapshot"])
    knowledge = core.retrieve(RetrievalSpec(
        query="FIFO stall workload", applicability={"stage": "cosim"},
    ))
    # Directly injected rows have no reviewed coverage/inventory activation
    # surface and therefore cannot become executable or displayed guidance.
    assert knowledge.guidance == []
    assert not knowledge.predictions

    without_predictions = core.retrieve(RetrievalSpec(
        query="prediction latency cycles", planes=("facts", "evidence"),
    ))
    predicted_spec = RetrievalSpec(
        query="prediction latency cycles", planes=("facts", "evidence"),
        include_predictions=True,
    )
    assert predicted_spec.planes == ("facts", "evidence", "predictions")
    predicted = core.retrieve(predicted_spec)
    assert retrieval_project["prediction"] in {item.record_id for item in predicted.predictions}
    assert all(item.plane == "predictions" for item in predicted.predictions)
    assert all(item.record_id != retrieval_project["prediction"] for item in predicted.facts)
    assert [(item.record_id, item.score) for item in predicted.facts] == [
        (item.record_id, item.score) for item in without_predictions.facts
    ]
    assert [(item.record_id, item.score) for item in predicted.guidance] == [
        (item.record_id, item.score) for item in without_predictions.guidance
    ]

    prediction_only = core.retrieve(RetrievalSpec(
        query="prediction latency cycles", planes=("predictions",),
        include_predictions=True,
    ))
    assert prediction_only.facts == []
    assert prediction_only.guidance == []
    assert {item.record_id for item in prediction_only.predictions} == {
        retrieval_project["prediction"],
    }


def test_fake_gate_stays_synthetic_and_free_form_reason_is_redacted(
    retrieval_project: dict[str, object],
) -> None:
    reason = "PRIVATE gate failure at C" + ":/secret/project/kernel.cpp"
    run = ToolRun(
        snapshot_id=retrieval_project["snapshot"], stage="csim",
        backend="runner.fake", request_hash="a" * 64,
        status=RunStatus.SUCCEEDED, exit_code=0,
        gates=[GateResult(
            GateKind.CORRECTNESS, GateStatus.FAIL, reason=reason,
        )],
        metadata={"authority": "synthetic", "tool_truth": False},
    )
    retrieval_project["bundle"].store.add_run(run)
    result = CoreService(
        retrieval_project["bundle"], retrieval_project["snapshot"],
    ).retrieve(RetrievalSpec(query="correctness fail", view="evidence"))
    gate = next(item for item in result.facts
                if item.record_kind == "verification_gate")
    assert gate.authority_class == "synthetic"
    assert gate.data["tool_truth"] is False
    assert gate.data["reason_redacted"] is True
    assert reason not in json.dumps(result.to_dict(), ensure_ascii=False)


def test_sdk_cli_rest_and_mcp_share_retrieval_semantics(
    retrieval_project: dict[str, object], capsys: pytest.CaptureFixture[str],
) -> None:
    root = retrieval_project["root"]
    snapshot = retrieval_project["snapshot"]
    project = Project.open(root)
    sdk = project.retrieve(RetrievalSpec(query="compute achieved II", snapshot_id=snapshot))

    rest = RestApplication(CoreService(retrieval_project["bundle"], snapshot)).dispatch(
        "GET", f"/api/v1/retrieve?q={quote('compute achieved II')}&top_k=8",
    )
    assert rest.status == 200
    mcp = ReadOnlyMcpService(CoreService(retrieval_project["bundle"], snapshot)).explore(
        "compute achieved II",
    )
    code = cli_main([
        "retrieve", "--project", str(root), "--snapshot-id", str(snapshot),
        "compute achieved II",
    ])
    cli = json.loads(capsys.readouterr().out)
    assert code == 0

    expected = [item.record_id for item in sdk.facts]
    assert [item["record_id"] for item in rest.body["facts"]] == expected
    assert [item["record_id"] for item in mcp["facts"]] == expected
    assert [item["record_id"] for item in cli["facts"]] == expected
    assert rest.body["trace"]["query_sha256"] == sdk.trace.query_sha256
    assert mcp["trace"]["query_sha256"] == sdk.trace.query_sha256
    assert cli["trace"]["query_sha256"] == sdk.trace.query_sha256

    def semantic_projection(payload: dict[str, object]) -> dict[str, object]:
        return {
            "facts": [item["record_id"] for item in payload["facts"]],
            "guidance": [item["record_id"] for item in payload["guidance"]],
            "predictions": [item["record_id"] for item in payload["predictions"]],
            "flow": payload["flow"],
            "citations": payload["citations"],
            "ambiguities": payload["ambiguities"],
            "confidence": payload["confidence"],
            "incomplete": payload["incomplete"],
            "stale": payload["stale"],
            "warnings": payload["warnings"],
        }

    expected_semantics = semantic_projection(sdk.to_dict())
    assert semantic_projection(rest.body) == expected_semantics
    assert semantic_projection(mcp) == expected_semantics
    assert semantic_projection(cli) == expected_semantics


def test_rest_never_honors_private_snippet_requests(
    retrieval_project: dict[str, object],
) -> None:
    app = RestApplication(CoreService(
        retrieval_project["bundle"], retrieval_project["snapshot"],
    ))
    response = app.dispatch(
        "GET", "/api/v1/retrieve?q=compute&include_private_snippets=true",
    )
    assert response.status == 200
    assert response.body["trace"]["private_snippets_requested"] is False
    assert response.body["trace"]["private_snippets_returned"] is False

    bounded = CoreService(
        retrieval_project["bundle"], retrieval_project["snapshot"],
    ).retrieve(RetrievalSpec(query="compute", max_chars=1_000))
    encoded = json.dumps(bounded.to_dict(), ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"))
    assert len(encoded) <= 1_000


def test_sdk_cli_rest_and_mcp_share_prediction_opt_in(
    retrieval_project: dict[str, object], capsys: pytest.CaptureFixture[str],
) -> None:
    root = retrieval_project["root"]
    snapshot = retrieval_project["snapshot"]
    query = "prediction latency cycles"
    project = Project.open(root)
    sdk = project.retrieve(RetrievalSpec(
        query=query, snapshot_id=snapshot, include_predictions=True,
    )).to_dict()
    rest = RestApplication(CoreService(
        retrieval_project["bundle"], snapshot,
    )).dispatch(
        "GET", f"/api/v1/retrieve?q={quote(query)}&include_predictions=true",
    )
    assert rest.status == 200
    mcp = ReadOnlyMcpService(CoreService(
        retrieval_project["bundle"], snapshot,
    )).explore(query, include_predictions=True)
    code = cli_main([
        "retrieve", "--project", str(root), "--snapshot-id", str(snapshot),
        "--include-predictions", query,
    ])
    cli = json.loads(capsys.readouterr().out)
    assert code == 0

    expected_prediction_ids = [item["record_id"] for item in sdk["predictions"]]
    assert expected_prediction_ids == [retrieval_project["prediction"]]
    for payload in (rest.body, mcp, cli):
        for section in ("facts", "guidance", "predictions"):
            assert [item["record_id"] for item in payload[section]] == [
                item["record_id"] for item in sdk[section]
            ]
        for field in (
            "flow", "citations", "ambiguities", "confidence", "incomplete",
            "stale", "warnings",
        ):
            assert payload[field] == sdk[field]


def test_private_adapter_excerpt_requires_request_and_project_authorization(
    retrieval_project: dict[str, object],
) -> None:
    class FixtureAdapter:
        adapter_id = "fixture.local"

        def search(self, spec, terms, limit):
            del spec, terms, limit
            return [RetrievalItem(
                record_id="local.chunk.fixture", plane="local",
                record_kind="local_document_chunk", title="Authorized local section",
                summary="Bounded local excerpt", authority_class="local_document_excerpt",
                data={
                    "private_excerpt": "one private, bounded line",
                    "authorization": "project_bounded",
                    "excerpt_sha256": "a" * 64,
                    "review_status": "local_unreviewed",
                },
            )]

    core = CoreService(retrieval_project["bundle"], retrieval_project["snapshot"])
    hidden = core.retrieve(
        RetrievalSpec(query="local section"), adapters=[FixtureAdapter()],
    )
    assert hidden.guidance == []
    assert any("private_excerpt_rejected" in item for item in hidden.warnings)

    visible = core.retrieve(RetrievalSpec(
        query="local section", include_private_snippets=True,
    ), adapters=[FixtureAdapter()])
    assert visible.guidance[0].data["private_excerpt"] == "one private, bounded line"
    assert visible.trace.private_snippets_returned is True

    local_plane_filtered = core.retrieve(RetrievalSpec(
        query="local section", planes=("facts",),
        include_private_snippets=True,
    ), adapters=[FixtureAdapter()])
    assert local_plane_filtered.guidance == []
    assert local_plane_filtered.trace.private_snippets_returned is False


def test_local_adapter_cannot_label_unreviewed_excerpt_as_public_rule(
    retrieval_project: dict[str, object],
) -> None:
    class MislabelledLocalAdapter:
        adapter_id = "fixture.mislabelled_local"

        @staticmethod
        def search(spec, terms, limit):
            del spec, terms, limit
            return [RetrievalItem(
                record_id="local.chunk.mislabelled", plane="local",
                record_kind="local_document_chunk", title="Local text",
                summary="Not reviewed", authority_class="knowledge_rule",
                data={"review_status": "human_reviewed"},
            )]

    result = CoreService(
        retrieval_project["bundle"], retrieval_project["snapshot"],
    ).retrieve(
        RetrievalSpec(query="local text"), adapters=[MislabelledLocalAdapter()],
    )
    assert result.guidance == []
    assert any("invalid_local_authority" in item for item in result.warnings)


def test_source_snippet_adapter_cannot_bypass_request_or_body_projection(
    retrieval_project: dict[str, object],
) -> None:
    class UnsafeSnippetAdapter:
        adapter_id = "test.unsafe_source_snippet.v1"

        @staticmethod
        def search(spec, terms, limit):
            del spec, terms, limit
            return []

        @staticmethod
        def source_snippets(spec, facts, *, snapshot_id, limit):
            del spec, facts, snapshot_id, limit
            return [RetrievalItem(
                record_id="unsafe", plane="evidence", record_kind="source_snippet",
                title="unsafe", summary="unsafe", score=1.0,
                data={
                    "authorization": "project_bounded",
                    "private_excerpt": "private source body",
                },
            )]

    core = CoreService(retrieval_project["bundle"], retrieval_project["snapshot"])
    hidden = core.retrieve(
        RetrievalSpec(query="dut"), adapters=[UnsafeSnippetAdapter()],
    )
    assert all(item.record_id != "unsafe" for item in hidden.facts)
    assert hidden.trace.private_snippets_returned is False

    class EmbeddedBodyAdapter(UnsafeSnippetAdapter):
        adapter_id = "test.embedded_body_source_snippet.v1"

        @staticmethod
        def source_snippets(spec, facts, *, snapshot_id, limit):
            item = UnsafeSnippetAdapter.source_snippets(
                spec, facts, snapshot_id=snapshot_id, limit=limit,
            )[0]
            item.data["content"] = "second private body"
            return [item]

    rejected = core.retrieve(
        RetrievalSpec(query="dut", include_private_snippets=True),
        adapters=[EmbeddedBodyAdapter()],
    )
    assert all(item.record_id != "unsafe" for item in rejected.facts)
    assert rejected.trace.private_snippets_returned is False

    from hlsgraph.retrieval import SourceSnippetRetrievalAdapter

    retrieval_project["bundle"].manifest.metadata["privacy"] = {
        "mcp_source_snippets": "bounded",
    }
    monkeypatched = SourceSnippetRetrievalAdapter(
        retrieval_project["bundle"], allow_private_snippets=True,
    )
    monkeypatched.source_snippets = lambda *_args, **_kwargs: [RetrievalItem(
        record_id="source_snippet_forged", plane="evidence",
        record_kind="source_snippet", title="forged", summary="forged",
        authority_class="static_fact", stage="source",
        entity_id=retrieval_project["kernel"],
        evidence_ids=[retrieval_project["kernel"], retrieval_project["artifact"]],
        data={
            "artifact_id": retrieval_project["artifact"],
            "artifact_sha256": "0" * 64,
            "anchor": {"start_line": 1, "end_line": 1, "symbol": "dut"},
            "projection_provenance": "hlsgraph.source_snippets.v1",
            "canonical_adapter_capability": (
                "hlsgraph.canonical_source_anchor_projection.v1"
            ),
            "private_excerpt": "forged private source",
            "authorization": "project_bounded",
            "excerpt_sha256": "1" * 64,
        },
    )]
    forged = core.retrieve(
        RetrievalSpec(query="dut", include_private_snippets=True),
        adapters=[monkeypatched],
    )
    assert all(item.record_id != "source_snippet_forged" for item in forged.facts)
    assert forged.trace.private_snippets_returned is False


def test_adapter_warning_cannot_leak_text_and_small_budget_remains_hard(
    retrieval_project: dict[str, object],
) -> None:
    class NoisyAdapter:
        adapter_id = "fixture.noisy"
        warnings = ["PRIVATE WARNING BODY with spaces"] * 50

        @staticmethod
        def search(spec, terms, limit):
            del spec, terms, limit
            return []

    result = CoreService(
        retrieval_project["bundle"], retrieval_project["snapshot"],
    ).retrieve(RetrievalSpec(query="dut", max_chars=1_000), adapters=[NoisyAdapter()])
    serialized = json.dumps(
        result.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    assert len(serialized) <= 1_000
    assert "PRIVATE WARNING BODY" not in serialized
    assert any("warning_rejected" in item for item in result.warnings)


def test_project_sidecar_is_metadata_only_until_bounded_policy_is_enabled(
    retrieval_project: dict[str, object],
) -> None:
    root = retrieval_project["root"]
    bundle = retrieval_project["bundle"]
    document = root / "private-guide.md"
    sentinel = "LOCAL_PRIVATE_SCHEDULE_SENTINEL"
    document.write_text(f"# Schedule\n{sentinel} evidence.\n", encoding="utf-8")
    metadata = index_local_document(
        document, document_id="test.local.schedule", document_version="1",
    )
    LocalKnowledgeSidecar(root).build(bundle.manifest.project_id, [metadata])
    core = CoreService(bundle, retrieval_project["snapshot"])

    metadata_only = core.retrieve(RetrievalSpec(query=sentinel))
    assert metadata_only.guidance and metadata_only.guidance[0].plane == "local"
    assert metadata_only.guidance[0].authority_class == "local_document_excerpt"
    assert metadata_only.guidance[0].data["review_status"] == "local_unreviewed"
    assert "private_excerpt" not in metadata_only.guidance[0].data
    assert metadata_only.trace.semantic_channel == "optional_not_enabled"

    still_hidden = core.retrieve(RetrievalSpec(
        query=sentinel, include_private_snippets=True,
    ))
    assert "private_excerpt" not in still_hidden.guidance[0].data
    assert still_hidden.trace.private_snippets_returned is False

    bundle.manifest.metadata["privacy"] = {"mcp_source_snippets": "bounded"}
    authorized = core.retrieve(RetrievalSpec(
        query=sentinel, include_private_snippets=True,
    ))
    assert sentinel in authorized.guidance[0].data["private_excerpt"]
    assert authorized.trace.private_snippets_returned is True

    access_log = root / ".hlsgraph/private/retrieval-access.jsonl"
    records = [json.loads(line) for line in access_log.read_text(
        encoding="ascii",
    ).splitlines()]
    local_records = [item for item in records
                     if item["anchor"]["kind"] == "knowledge_chunk"]
    assert {item["result"] for item in local_records} >= {"denied_policy", "returned"}
    assert all(set(item) == {"content_sha256", "anchor", "result", "byte_count"}
               for item in local_records)
    log_text = access_log.read_text(encoding="ascii")
    assert sentinel not in log_text
    assert "private-guide.md" not in log_text

    rest = RestApplication(core).dispatch("GET", f"/api/v1/retrieve?q={sentinel}")
    assert rest.status == 200
    assert "private_excerpt" not in rest.body["guidance"][0]["data"]


def test_private_sidecar_heading_is_body_and_requires_bounded_authorization(
    retrieval_project: dict[str, object],
) -> None:
    root = retrieval_project["root"]
    bundle = retrieval_project["bundle"]
    heading = "PRIVATE_HEADING_SENTINEL"
    document = root / "private-heading.md"
    document.write_text(
        f"# {heading}\nPipeline schedule evidence.\n", encoding="utf-8",
    )
    metadata = index_local_document(
        document, document_id="test.local.heading", document_version="1",
        title="Declared public-safe title",
    )
    sidecar = LocalKnowledgeSidecar(root)
    sidecar.build(bundle.manifest.project_id, [metadata])

    metadata_hit = sidecar.search("pipeline", include_text=False)[0]
    assert metadata_hit.heading is None
    assert metadata_hit.title == "Declared public-safe title"
    hidden = CoreService(bundle, retrieval_project["snapshot"]).retrieve(
        RetrievalSpec(query="pipeline"),
    )
    serialized = json.dumps(hidden.to_dict(), ensure_ascii=False)
    assert heading not in serialized

    bundle.manifest.metadata["privacy"] = {"mcp_source_snippets": "bounded"}
    visible = CoreService(bundle, retrieval_project["snapshot"]).retrieve(
        RetrievalSpec(query="pipeline", include_private_snippets=True),
    )
    assert heading in json.dumps(visible.to_dict(), ensure_ascii=False)


def test_source_snippet_requires_bounded_policy_and_revalidates_artifact(
    retrieval_project: dict[str, object],
) -> None:
    bundle = retrieval_project["bundle"]
    core = CoreService(bundle, retrieval_project["snapshot"])
    denied = core.retrieve(RetrievalSpec(
        query="dut", include_private_snippets=True,
    ))
    assert all(item.record_kind != "source_snippet" for item in denied.facts)
    assert denied.trace.private_snippets_returned is False

    bundle.manifest.metadata["privacy"] = {"mcp_source_snippets": "bounded"}
    allowed = core.retrieve(RetrievalSpec(
        query="dut", include_private_snippets=True,
    ))
    snippets = [item for item in allowed.facts if item.record_kind == "source_snippet"]
    assert len(snippets) == 1
    assert retrieval_project["secret"] in snippets[0].data["private_excerpt"]
    assert len(snippets[0].data["private_excerpt"].splitlines()) <= 80
    assert len(snippets[0].data["private_excerpt"]) <= 4_000
    assert allowed.trace.private_snippets_returned is True

    bounded = core.retrieve(RetrievalSpec(
        query="dut", include_private_snippets=True, max_chars=1_000,
    ))
    assert len(json.dumps(
        bounded.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )) <= 1_000

    mcp = ReadOnlyMcpService(core).explore(
        "dut", include_private_snippets=True,
    )
    assert any(item["record_kind"] == "source_snippet" for item in mcp["facts"])

    rest = RestApplication(core).dispatch(
        "GET", "/api/v1/retrieve?q=dut&include_private_snippets=true",
    )
    assert rest.status == 200
    assert all(item["record_kind"] != "source_snippet" for item in rest.body["facts"])

    (retrieval_project["root"] / "kernel.cpp").write_text(
        "void changed() {}\n", encoding="utf-8",
    )
    stale = core.retrieve(RetrievalSpec(
        query="dut", include_private_snippets=True,
    ))
    assert all(item.record_kind != "source_snippet" for item in stale.facts)
    assert any("source_snippet_" in item and (
        "hash_mismatch" in item or "size_mismatch" in item
    ) for item in stale.warnings)
    access_log = retrieval_project["root"] / ".hlsgraph/private/retrieval-access.jsonl"
    records = [json.loads(line) for line in access_log.read_text(
        encoding="ascii",
    ).splitlines()]
    source_records = [item for item in records
                      if item["anchor"]["kind"] == "source_line"]
    assert {item["result"] for item in source_records} >= {"denied_policy", "returned"}
    assert all(set(item) == {"content_sha256", "anchor", "result", "byte_count"}
               for item in source_records)
    log_text = access_log.read_text(encoding="ascii")
    assert retrieval_project["secret"] not in log_text
    assert '"query"' not in log_text and "dut" not in log_text


def test_source_snippet_adapter_rejects_parent_traversal(
    retrieval_project: dict[str, object],
) -> None:
    from hlsgraph.retrieval import SourceSnippetRetrievalAdapter

    artifact = next(item for item in retrieval_project["bundle"].store.artifacts(
        retrieval_project["snapshot"],
    ) if item.id == retrieval_project["artifact"])
    artifact.uri = "../outside.cpp"
    data, reason = SourceSnippetRetrievalAdapter._verified_bytes(
        retrieval_project["root"], artifact,
    )
    assert data is None and reason == "unsafe_path"


def test_private_access_log_fails_closed_on_unsafe_private_path(tmp_path: Path) -> None:
    from hlsgraph.retrieval import _append_private_access

    (tmp_path / ".hlsgraph").mkdir()
    (tmp_path / ".hlsgraph/private").write_text("not a directory", encoding="ascii")
    assert not _append_private_access(
        tmp_path, content_sha256="a" * 64,
        anchor={"kind": "source_line", "start_line": 1, "end_line": 1},
        result="denied_policy", byte_count=0,
    )


@pytest.mark.skipif(os.name == "nt", reason="POSIX directory mode contract")
def test_private_access_log_does_not_rechmod_hardened_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hlsgraph.retrieval as retrieval

    ledger_root = tmp_path / ".hlsgraph"
    private_root = ledger_root / "private"
    ledger_root.mkdir()
    private_root.mkdir(mode=0o700)
    private_root.chmod(0o700)

    def reject_redundant_chmod(_path: object, _mode: int) -> None:
        raise AssertionError("an already-hardened directory must not be rewritten")

    monkeypatch.setattr(retrieval.os, "chmod", reject_redundant_chmod)
    assert retrieval._append_private_access(
        tmp_path, content_sha256="a" * 64,
        anchor={"kind": "source_line", "start_line": 1, "end_line": 1},
        result="returned", byte_count=1,
    )


@pytest.mark.skipif(os.name == "nt", reason="POSIX directory mode contract")
def test_private_access_log_hardens_weaker_directory_mode(tmp_path: Path) -> None:
    import hlsgraph.retrieval as retrieval

    ledger_root = tmp_path / ".hlsgraph"
    private_root = ledger_root / "private"
    ledger_root.mkdir()
    private_root.mkdir(mode=0o755)
    private_root.chmod(0o755)
    assert retrieval._append_private_access(
        tmp_path, content_sha256="a" * 64,
        anchor={"kind": "source_line", "start_line": 1, "end_line": 1},
        result="returned", byte_count=1,
    )
    assert stat.S_IMODE(private_root.stat().st_mode) == 0o700


def test_private_access_log_verifies_opened_file_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hlsgraph.retrieval as retrieval

    (tmp_path / ".hlsgraph").mkdir()
    original_fstat = retrieval.os.fstat

    class ReplacedIdentity:
        def __init__(self, value):
            self.st_mode = value.st_mode
            self.st_dev = value.st_dev
            self.st_ino = value.st_ino + 1

    monkeypatch.setattr(
        retrieval.os, "fstat", lambda descriptor: ReplacedIdentity(
            original_fstat(descriptor)
        ),
    )
    assert not retrieval._append_private_access(
        tmp_path, content_sha256="a" * 64,
        anchor={"kind": "source_line", "start_line": 1, "end_line": 1},
        result="denied_policy", byte_count=0,
    )
