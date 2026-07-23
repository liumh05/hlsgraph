from __future__ import annotations

import json

from hlsgraph.bundle import GraphBundle
from hlsgraph.export.ml import export_dataset
from hlsgraph.graph import CanonicalGraph
from hlsgraph.manifest import minimal_manifest
from hlsgraph.model import (
    AuthorityClass,
    DatasetManifest,
    Derivation,
    Entity,
    EvidenceKind,
    EvidenceRef,
    json_ready,
)
from hlsgraph.query import CoreService, static_aggregate_receipt_valid
from hlsgraph.retrieval import HybridRetriever, RetrievalSpec
from hlsgraph.version import FEATURE_SCHEMA_VERSION


def _aggregate_bundle(tmp_path):
    (tmp_path / "kernel.cpp").write_text(
        "void dut() {}\n", encoding="utf-8",
    )
    bundle = GraphBundle.create(
        tmp_path,
        minimal_manifest(
            "test.aggregate_consumers", "aggregate consumers",
            "dut", "kernel.cpp",
        ),
    )
    snapshot = bundle.snapshot()
    function = Entity(
        kind="ir.llvm.function",
        name="dut",
        qualified_name="kernel.ll::dut",
        snapshot_id=snapshot.id,
        stage="llvm",
        authority=AuthorityClass.COMPILER_DECISION,
    )
    graph = CanonicalGraph(snapshot.id)
    graph.add_entity(function)
    bundle.store.save_graph(graph)
    aggregate = Derivation(
        snapshot_id=snapshot.id,
        subject_id=function.id,
        predicate="feature.operation_histogram",
        value={"add": 1},
        stage="llvm",
        algorithm="hlsgraph.static.operation_histogram",
        algorithm_version="2",
        evidence_refs=[EvidenceRef(
            kind=EvidenceKind.ENTITY_ANCHOR,
            target_id=function.id,
            snapshot_id=snapshot.id,
        )],
    )
    # Simulate a pre-receipt/partially migrated database. The current write
    # boundary correctly refuses this row; consumers must still fail closed
    # when such legacy bytes are encountered at read time.
    with bundle.store.write() as connection:
        connection.execute(
            "INSERT INTO derivations("
            "id,snapshot_id,subject_id,predicate,payload_json"
            ") VALUES(?,?,?,?,?)",
            (
                aggregate.id,
                aggregate.snapshot_id,
                aggregate.subject_id,
                aggregate.predicate,
                json.dumps(
                    json_ready(aggregate),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            ),
        )
    return bundle, snapshot.id, function, aggregate


def _feature_row(output):
    return json.loads(
        (output / "feature_evidence.jsonl").read_text(
            encoding="utf-8",
        ).strip()
    )


def test_standard_aggregate_masks_fail_closed_then_accepts_store_receipt(
    tmp_path, monkeypatch,
) -> None:
    bundle, snapshot_id, function, aggregate = _aggregate_bundle(tmp_path)
    service = CoreService(bundle, snapshot_id)

    queried = service.feature_evidence(
        function.id, predicates=[aggregate.predicate],
    )
    assert queried["items"][0]["mask"] is False
    assert queried["items"][0]["value"] is None
    assert queried["items"][0]["aggregate_receipt_valid"] is False

    dataset = DatasetManifest(
        dataset_id="dataset.aggregate_consumers",
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        snapshot_ids=[snapshot_id],
        feature_evidence_predicates=[aggregate.predicate],
    )
    unverified_output = tmp_path / "unverified-export"
    export_dataset(
        bundle, snapshot_id, unverified_output, dataset,
    )
    unverified = _feature_row(unverified_output)
    assert unverified["mask"] is False
    assert unverified["value"] is None
    assert unverified["aggregate_receipt_valid"] is False

    monkeypatch.setattr(
        type(bundle.store),
        "valid_static_aggregate_ids",
        lambda _store, selected_snapshot: (
            frozenset({aggregate.id})
            if selected_snapshot == snapshot_id else frozenset()
        ),
    )
    queried = service.feature_evidence(
        function.id, predicates=[aggregate.predicate],
    )
    assert queried["items"][0]["mask"] is True
    assert queried["items"][0]["value"] == {"add": 1}
    assert queried["items"][0]["aggregate_receipt_valid"] is True

    verified_output = tmp_path / "verified-export"
    export_dataset(
        bundle, snapshot_id, verified_output, dataset,
    )
    verified = _feature_row(verified_output)
    assert verified["mask"] is True
    assert verified["value"] == {"add": 1}
    assert verified["aggregate_receipt_valid"] is True


def test_retrieval_downgrades_and_withholds_unreceipted_aggregate_context(
    tmp_path, monkeypatch,
) -> None:
    bundle, snapshot_id, function, aggregate = _aggregate_bundle(tmp_path)
    retriever = HybridRetriever(bundle, snapshot_id)
    graph = bundle.store.load_graph(snapshot_id)
    spec = RetrievalSpec(
        query="operation histogram",
        planes=("evidence",),
    )

    documents = retriever._documents(
        spec, graph, {function.id}, [],
    )
    aggregate_document = next(
        item for item in documents
        if item.item.record_id == aggregate.id
    )
    assert aggregate_document.item.completeness == "incomplete"
    assert aggregate_document.item.data["value"] is None
    assert aggregate_document.item.data["aggregate_receipt_valid"] is False
    contexts = retriever._binding_target_contexts(
        graph, {function.id},
    )
    assert not contexts.get(("predicate", aggregate.predicate))

    monkeypatch.setattr(
        type(bundle.store),
        "valid_static_aggregate_ids",
        lambda _store, _snapshot: frozenset({aggregate.id}),
    )
    retriever = HybridRetriever(bundle, snapshot_id)
    documents = retriever._documents(
        spec, graph, {function.id}, [],
    )
    aggregate_document = next(
        item for item in documents
        if item.item.record_id == aggregate.id
    )
    assert aggregate_document.item.completeness == "complete"
    assert aggregate_document.item.data["value"] == {"add": 1}
    assert aggregate_document.item.data["aggregate_receipt_valid"] is True
    contexts = retriever._binding_target_contexts(
        graph, {function.id},
    )
    assert contexts[("predicate", aggregate.predicate)]


def test_custom_derivation_cannot_mask_true_through_unreceipted_aggregate(
    tmp_path, monkeypatch,
) -> None:
    bundle, snapshot_id, function, aggregate = _aggregate_bundle(tmp_path)
    wrapper = Derivation(
        snapshot_id=snapshot_id,
        subject_id=function.id,
        predicate="feature.fixture_wrapper",
        value={"present": 1},
        stage="llvm",
        algorithm="test.wrapper",
        algorithm_version="1",
        evidence_refs=[EvidenceRef(
            kind=EvidenceKind.DERIVATION,
            target_id=aggregate.id,
            snapshot_id=snapshot_id,
        )],
    )
    bundle.store.add_derivations([wrapper])
    service = CoreService(bundle, snapshot_id)

    queried = service.feature_evidence(
        function.id, predicates=[wrapper.predicate],
    )
    assert queried["items"] == []
    retriever = HybridRetriever(bundle, snapshot_id)
    graph = bundle.store.load_graph(snapshot_id)
    wrapper_document = next(
        item for item in retriever._documents(
            RetrievalSpec(query="fixture wrapper", planes=("evidence",)),
            graph, {function.id}, [],
        )
        if item.item.record_id == wrapper.id
    )
    assert wrapper_document.item.completeness == "incomplete"
    assert wrapper_document.item.data["value"] is None
    assert wrapper_document.item.data["evidence_chain_valid"] is False
    assert not retriever._binding_target_contexts(
        graph, {function.id},
    ).get(("predicate", wrapper.predicate))
    dataset = DatasetManifest(
        dataset_id="dataset.aggregate_wrapper",
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        snapshot_ids=[snapshot_id],
        feature_evidence_predicates=[wrapper.predicate],
    )
    unverified_output = tmp_path / "unverified-wrapper-export"
    export_dataset(
        bundle, snapshot_id, unverified_output, dataset,
    )
    unverified_rows = [
        json.loads(line)
        for line in (
            unverified_output / "feature_evidence.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert all(row["mask"] is False for row in unverified_rows)
    assert all(row["value"] is None for row in unverified_rows)

    monkeypatch.setattr(
        type(bundle.store),
        "valid_static_aggregate_ids",
        lambda _store, _snapshot: frozenset({aggregate.id}),
    )
    queried = service.feature_evidence(
        function.id, predicates=[wrapper.predicate],
    )
    assert queried["items"][0]["mask"] is True
    retriever = HybridRetriever(bundle, snapshot_id)
    wrapper_document = next(
        item for item in retriever._documents(
            RetrievalSpec(query="fixture wrapper", planes=("evidence",)),
            graph, {function.id}, [],
        )
        if item.item.record_id == wrapper.id
    )
    assert wrapper_document.item.completeness == "complete"
    assert wrapper_document.item.data["value"] == {"present": 1}
    assert wrapper_document.item.data["evidence_chain_valid"] is True
    assert retriever._binding_target_contexts(
        graph, {function.id},
    )[("predicate", wrapper.predicate)]
    verified_output = tmp_path / "verified-wrapper-export"
    export_dataset(
        bundle, snapshot_id, verified_output, dataset,
    )
    verified_rows = [
        json.loads(line)
        for line in (
            verified_output / "feature_evidence.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert all(row["mask"] is True for row in verified_rows)
    assert all(row["value"] is not None for row in verified_rows)


def test_missing_or_failing_store_receipt_capability_is_false() -> None:
    aggregate = {
        "id": "derivation.fixture",
        "snapshot_id": "snapshot.fixture",
        "predicate": "feature.operation_histogram",
        "completeness": "complete",
        "value": {"add": 1},
    }

    class OldStore:
        pass

    class FailingStore:
        @staticmethod
        def static_aggregate_receipt_valid(_snapshot, _item):
            raise RuntimeError("legacy table is unavailable")

    assert static_aggregate_receipt_valid(
        OldStore(), "snapshot.fixture", aggregate,
    ) is False
    assert static_aggregate_receipt_valid(
        FailingStore(), "snapshot.fixture", aggregate,
    ) is False
    assert static_aggregate_receipt_valid(
        FailingStore(), "snapshot.other", aggregate,
    ) is False
    assert static_aggregate_receipt_valid(
        OldStore(), "snapshot.fixture",
        {"predicate": "feature.custom"},
    ) is True


def test_partial_standard_aggregate_does_not_require_a_receipt() -> None:
    partial = {
        "id": "derivation.partial",
        "snapshot_id": "snapshot.fixture",
        "predicate": "feature.operation_histogram",
        "completeness": "partial",
        "value": {"mystery.compute": 1},
    }

    class OldStore:
        pass

    assert static_aggregate_receipt_valid(
        OldStore(), "snapshot.fixture", partial,
    ) is True


def test_retriever_does_not_cache_bare_valid_aggregate_ids(
    tmp_path, monkeypatch,
) -> None:
    bundle, snapshot_id, function, aggregate = _aggregate_bundle(tmp_path)
    state = {"valid": True}
    monkeypatch.setattr(
        type(bundle.store),
        "valid_static_aggregate_ids",
        lambda _store, _snapshot: (
            frozenset({aggregate.id}) if state["valid"] else frozenset()
        ),
    )
    retriever = HybridRetriever(bundle, snapshot_id)
    graph = bundle.store.load_graph(snapshot_id)
    spec = RetrievalSpec(query="operation histogram", planes=("evidence",))

    first = next(
        item for item in retriever._documents(
            spec, graph, {function.id}, [],
        )
        if item.item.record_id == aggregate.id
    )
    assert first.item.data["value"] == {"add": 1}
    assert first.item.completeness == "complete"

    state["valid"] = False
    second = next(
        item for item in retriever._documents(
            spec, graph, {function.id}, [],
        )
        if item.item.record_id == aggregate.id
    )
    assert second.item.data["value"] is None
    assert second.item.completeness == "incomplete"
