from __future__ import annotations

import sqlite3
from pathlib import Path

from hlsgraph.model import (
    AuthorityClass,
    GateKind,
    GateResult,
    GateStatus,
    VerificationKind,
    VerificationResult,
)
from hlsgraph.retrieval import HybridRetriever, RetrievalSpec
from tests.attested_run_support import commit_attested
from tests.test_evidence_closure import (
    _bundle,
    _managed_report,
    _parsed_observation,
    _run,
)


def _attested_csim_case(root: Path):
    bundle, snapshot, kernel = _bundle(root)
    run = _run(bundle, snapshot, "csim", "9", workload="tb.receipt")
    report = _managed_report(
        bundle,
        run,
        "csim.json",
        "amd.vitis.csim_result",
        {"workload_id": "tb.receipt"},
    )
    observations = [
        _parsed_observation(
            bundle,
            report,
            snapshot.id,
            kernel.id,
            predicate,
            value,
            "csim",
            AuthorityClass.VERIFICATION_EVIDENCE,
            run_id=run.id,
            workload_id="tb.receipt",
            unit="count",
        )
        for predicate, value in (
            ("csim.exit_code", 0),
            ("csim.mismatches", 0),
            ("csim.assertions_failed", 0),
        )
    ]
    evidence_ids = [item.id for item in observations]
    verification = VerificationResult(
        snapshot.id,
        VerificationKind.CSIM,
        GateStatus.PASS,
        run_id=run.id,
        workload_id="tb.receipt",
        evidence_ids=evidence_ids,
    )
    run.output_artifact_ids = [report.id]
    run.gates = [GateResult(
        GateKind.CORRECTNESS,
        GateStatus.PASS,
        evidence_ids=evidence_ids,
    )]
    commit_attested(
        bundle,
        run=run,
        artifacts=[report],
        observations=observations,
        verifications=[verification],
    )
    return bundle, snapshot, run, report, observations, verification


def _contexts(retriever: HybridRetriever):
    graph = retriever.bundle.store.load_graph(retriever.snapshot_id)
    return graph, retriever._binding_target_contexts(graph, set(graph.entities))


def _evidence_documents(retriever: HybridRetriever, graph):
    return retriever._documents(
        RetrievalSpec(
            query="CSim correctness",
            snapshot_id=retriever.snapshot_id,
            view="evidence",
        ),
        graph,
        set(graph.entities),
        [],
    )


def test_missing_execution_receipt_disables_all_retrieval_truth_capabilities(
    tmp_path: Path,
) -> None:
    bundle, snapshot, run, _report, observations, verification = _attested_csim_case(
        tmp_path / "receipt-case",
    )
    retriever = HybridRetriever(bundle, snapshot.id)

    assert bundle.store.has_valid_execution_commit(snapshot.id, run.id)
    graph, before = _contexts(retriever)
    assert any(
        item.get("observation_evidence_qualified")
        == {"derived_from_typed_observation_evidence_v1"}
        for item in before[("predicate", "csim.exit_code")]
    )
    assert any(
        item.get("gate_evidence_qualified")
        == {"derived_from_typed_evidence_v1"}
        for item in before[("gate_kind", "correctness")]
    )
    before_documents = _evidence_documents(retriever, graph)
    before_verification = next(
        item.item for item in before_documents
        if item.item.record_id == verification.id
    )
    before_gate = next(
        item.item for item in before_documents
        if item.item.record_kind == "verification_gate"
        and item.item.data.get("run_id") == run.id
    )
    assert before_verification.data["tool_truth"] is True
    assert before_gate.data["tool_truth"] is True

    # Model a legacy migration or direct database injection that preserved all
    # serializable run/report metadata but never received the pipeline-issued
    # atomic commit receipt.
    with sqlite3.connect(bundle.store.path) as connection:
        connection.execute(
            "DELETE FROM execution_commit_receipts WHERE run_id=?",
            (run.id,),
        )

    assert not bundle.store.has_valid_execution_commit(snapshot.id, run.id)
    graph, after = _contexts(retriever)
    assert all(
        "observation_evidence_qualified" not in item
        for item in after[("predicate", "csim.exit_code")]
    )
    assert all(
        "gate_evidence_qualified" not in item
        for item in after[("gate_kind", "correctness")]
    )
    after_documents = _evidence_documents(retriever, graph)
    after_verification = next(
        item.item for item in after_documents
        if item.item.record_id == verification.id
    )
    after_gate = next(
        item.item for item in after_documents
        if item.item.record_kind == "verification_gate"
        and item.item.data.get("run_id") == run.id
    )
    assert after_verification.data["tool_truth"] is False
    assert after_gate.data["tool_truth"] is False
    assert all(item.run_id == run.id for item in observations)
