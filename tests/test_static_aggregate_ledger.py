from __future__ import annotations

import hashlib
import json
import sqlite3

import pytest

from hlsgraph.bundle import GraphBundle
from hlsgraph.extract import (
    ExtractionContext,
    ExtractionPipeline,
    ExtractionResult,
)
from hlsgraph.extract.index_authorization import (
    _finalize_index_authorization,
)
from hlsgraph.graph import CanonicalGraph
from hlsgraph.manifest import minimal_manifest
from hlsgraph.model import (
    ArtifactRef,
    Completeness,
    Derivation,
    Entity,
    EvidenceKind,
    EvidenceRef,
    Relation,
    SourceAnchor,
    json_ready,
)
from hlsgraph.sdk import Project
from hlsgraph.static_aggregate import STANDARD_STATIC_AGGREGATE_PREDICATES
from hlsgraph.store import StoreError


def _bare_bundle(tmp_path):
    (tmp_path / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    bundle = GraphBundle.create(
        tmp_path,
        minimal_manifest(
            "test.static.receipt", "static receipt", "dut", "kernel.cpp",
        ),
    )
    snapshot = bundle.snapshot()
    kernel = Entity(
        "hls.kernel", "dut", snapshot.id, qualified_name="dut", stage="ast",
    )
    graph = CanonicalGraph(snapshot.id)
    graph.add_entity(kernel)
    bundle.store.save_graph(graph)
    return bundle, snapshot, kernel


def _forged_complete(snapshot_id: str, subject_id: str) -> Derivation:
    return Derivation(
        snapshot_id=snapshot_id,
        subject_id=subject_id,
        predicate="feature.operation_histogram",
        value={"add": 1},
        algorithm="hlsgraph.static_features",
        algorithm_version="2",
        stage="llvm",
        completeness=Completeness.COMPLETE,
        evidence_refs=[EvidenceRef(
            kind=EvidenceKind.ENTITY_ANCHOR,
            target_id=subject_id,
            snapshot_id=snapshot_id,
        )],
    )


def test_direct_complete_standard_aggregate_write_is_rejected(tmp_path) -> None:
    bundle, snapshot, kernel = _bare_bundle(tmp_path)
    forged = _forged_complete(snapshot.id, kernel.id)

    with pytest.raises(StoreError, match="pipeline index authorization"):
        bundle.store.add_derivations([forged])

    assert bundle.store.derivations(snapshot.id) == []


@pytest.mark.parametrize(
    ("predicate", "value"),
    [
        ("feature.trip_count", 8),
        (
            "feature.loop_bounds",
            {
                "lower": 0, "upper": 8, "step": 1,
                "comparison": "lt", "upper_inclusive": False,
            },
        ),
        ("feature.dependence_distance", {"distances": [1]}),
    ],
)
def test_direct_complete_non_histogram_static_claim_is_rejected(
    tmp_path, predicate, value,
) -> None:
    bundle, snapshot, kernel = _bare_bundle(tmp_path)
    forged = Derivation(
        snapshot_id=snapshot.id,
        subject_id=kernel.id,
        predicate=predicate,
        value=value,
        algorithm=f"hlsgraph.static.{predicate.removeprefix('feature.')}",
        algorithm_version="2",
        stage="ast",
        completeness=Completeness.COMPLETE,
        evidence_refs=[EvidenceRef(
            kind=EvidenceKind.ENTITY_ANCHOR,
            target_id=kernel.id,
            snapshot_id=snapshot.id,
        )],
    )

    with pytest.raises(StoreError, match="pipeline index authorization"):
        bundle.store.add_derivations([forged])


def test_complete_null_standard_claim_is_rejected(tmp_path) -> None:
    bundle, snapshot, kernel = _bare_bundle(tmp_path)
    forged = Derivation(
        snapshot_id=snapshot.id,
        subject_id=kernel.id,
        predicate="feature.operation_histogram",
        value=None,
        algorithm="hlsgraph.static.operation_histogram",
        algorithm_version="2",
        stage="llvm",
        completeness=Completeness.COMPLETE,
        evidence_refs=[EvidenceRef(
            kind=EvidenceKind.ENTITY_ANCHOR,
            target_id=kernel.id,
            snapshot_id=snapshot.id,
        )],
    )

    with pytest.raises(StoreError, match="known value"):
        bundle.store.add_derivations([forged])


def test_legacy_complete_aggregate_without_index_receipt_is_invalid(tmp_path) -> None:
    bundle, snapshot, kernel = _bare_bundle(tmp_path)
    forged = _forged_complete(snapshot.id, kernel.id)
    with sqlite3.connect(bundle.store.path) as connection:
        connection.execute(
            "INSERT INTO derivations("
            "id,snapshot_id,subject_id,predicate,payload_json) VALUES(?,?,?,?,?)",
            (
                forged.id,
                forged.snapshot_id,
                forged.subject_id,
                forged.predicate,
                json.dumps(
                    json_ready(forged), ensure_ascii=False, sort_keys=True,
                    separators=(",", ":"), allow_nan=False,
                ),
            ),
        )

    assert bundle.store.index_commit_receipt(snapshot.id) is None
    assert bundle.store.valid_static_aggregate_ids(snapshot.id) == frozenset()
    assert not bundle.store.static_aggregate_receipt_valid(snapshot.id, forged)

    # Candidate-era v0.3 databases may predate the additive receipt table.
    # Read-side trust must degrade to invalid rather than making the graph
    # unreadable or performing an implicit migration.
    with sqlite3.connect(bundle.store.path) as connection:
        connection.execute("DROP TABLE index_commit_receipts")
    assert bundle.store.index_commit_receipt(snapshot.id) is None
    assert bundle.store.valid_static_aggregate_ids(snapshot.id) == frozenset()
    assert not bundle.store.static_aggregate_receipt_valid(snapshot.id, forged)


def test_builtin_llvm_index_persists_and_revalidates_receipt(tmp_path) -> None:
    (tmp_path / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    (tmp_path / "dut.ll").write_text(
        "define i32 @dut(i32 %arg) {\n"
        "entry:\n"
        "  %sum = add i32 %arg, 1\n"
        "  ret i32 %sum\n"
        "}\n",
        encoding="utf-8",
    )
    manifest = minimal_manifest(
        "test.static.receipt.llvm", "LLVM static receipt", "dut", "kernel.cpp",
    )
    manifest.artifact_paths.append({
        "path": "dut.ll",
        "kind": "ir.llvm",
        "role": "llvm_ir",
        "access": "project",
    })
    project = Project(GraphBundle.create(tmp_path, manifest))

    result = project.index(degraded=True)

    assert result.success
    complete = [
        item for item in project.bundle.store.derivations(result.snapshot_id)
        if item["predicate"] in STANDARD_STATIC_AGGREGATE_PREDICATES
        and item["completeness"] == "complete"
        and item["value"] is not None
    ]
    assert complete
    assert any(item["predicate"] == "feature.operation_histogram"
               for item in complete)
    receipt = project.bundle.store.index_commit_receipt(result.snapshot_id)
    assert receipt is not None
    assert receipt["snapshot_id"] == result.snapshot_id
    valid_ids = project.bundle.store.valid_static_aggregate_ids(
        result.snapshot_id,
    )
    assert valid_ids == {item["id"] for item in complete}
    assert all(
        project.bundle.store.static_aggregate_receipt_valid(
            result.snapshot_id, item,
        )
        for item in complete
    )

    # Later run outputs may be attached to the same immutable design snapshot.
    # They must not invalidate a receipt whose original input artifacts remain
    # present and byte-identical.
    (tmp_path / "aux.bin").write_bytes(b"")
    project.bundle.store.add_artifact(
        result.snapshot_id,
        ArtifactRef(
            kind="test.aux_output",
            uri="aux.bin",
            sha256=hashlib.sha256(b"").hexdigest(),
            size=0,
            role="test_output",
            access="project",
        ),
    )
    assert project.bundle.store.valid_static_aggregate_ids(
        result.snapshot_id,
    ) == valid_ids

    repeated = project.index(degraded=True)
    assert repeated.success
    assert repeated.snapshot_id == result.snapshot_id
    assert project.bundle.store.index_commit_receipt(
        repeated.snapshot_id,
    ) == receipt
    assert project.bundle.store.valid_static_aggregate_ids(
        repeated.snapshot_id,
    ) == valid_ids

    forged_payload = dict(complete[0])
    forged_payload["value"] = {"forged": 999}
    with sqlite3.connect(project.bundle.store.path) as connection:
        connection.execute(
            "UPDATE derivations SET payload_json=? WHERE id=?",
            (
                json.dumps(
                    forged_payload, ensure_ascii=False, sort_keys=True,
                    separators=(",", ":"), allow_nan=False,
                ),
                complete[0]["id"],
            ),
        )
    assert project.bundle.store.valid_static_aggregate_ids(
        result.snapshot_id,
    ) == frozenset()
    assert not project.bundle.store.static_aggregate_receipt_valid(
        result.snapshot_id, complete[0],
    )


def test_same_named_plugin_cannot_authorize_complete_static_aggregates(
    tmp_path,
) -> None:
    """Serialized parser names never substitute for the built-in class."""

    bundle, snapshot, _kernel = _bare_bundle(tmp_path)
    artifact = bundle.store.artifacts(snapshot.id)[0]
    parser_attrs = {
        "static_feature_domain_complete": True,
        "static_feature_domain_contract": (
            "hlsgraph.ir.llvm_text.static_feature_domain.v1"
        ),
        "static_feature_parser": "ir.llvm_text",
        "static_feature_parser_version": "2",
        "static_feature_unparsed_construct_count": 0,
        "static_feature_artifact_id": artifact.id,
        "static_feature_artifact_sha256": artifact.sha256,
    }

    class SameNamedPlugin:
        name = "ir.llvm_text"
        version = "2"

        @staticmethod
        def supports(_context):
            return True

        @staticmethod
        def extract(_context):
            graph = CanonicalGraph(snapshot.id)
            function = graph.add_entity(Entity(
                kind="ir.llvm.function",
                name="dut",
                qualified_name="kernel.ll::dut",
                snapshot_id=snapshot.id,
                stage="llvm",
                attrs=dict(parser_attrs),
                anchors=[SourceAnchor(artifact.id)],
            ))
            operation = graph.add_entity(Entity(
                kind="ir.llvm.operation",
                name="add",
                qualified_name="kernel.ll:1:add",
                snapshot_id=snapshot.id,
                stage="llvm",
                attrs={
                    **parser_attrs,
                    "opcode": "add",
                    "index_kinds": [],
                    "bitwidths": [32],
                },
                anchors=[SourceAnchor(artifact.id)],
            ))
            graph.add_relation(Relation(
                src=function.id,
                dst=operation.id,
                kind="ir.contains",
                snapshot_id=snapshot.id,
                stage="llvm",
            ))
            return ExtractionResult(graph=graph)

    context = ExtractionContext(
        project_root=tmp_path,
        manifest=bundle.manifest,
        snapshot=snapshot,
        artifacts={artifact.id: artifact},
        allow_degraded=True,
    )
    extracted = ExtractionPipeline([SameNamedPlugin()]).run(context)
    assert extracted._index_origin_capability is not None
    assert any(
        item.predicate in STANDARD_STATIC_AGGREGATE_PREDICATES
        and item.completeness == Completeness.COMPLETE
        for item in extracted.derivations
    )

    finalized, authorization, withheld = _finalize_index_authorization(
        extracted._index_origin_capability,
        extracted.graph,
        extracted.derivations,
        context.artifacts,
    )
    assert authorization is None
    assert withheld
    assert not [
        item for item in finalized
        if item.predicate in STANDARD_STATIC_AGGREGATE_PREDICATES
        and item.completeness == Completeness.COMPLETE
    ]


def test_source_plugin_cannot_upgrade_syntactic_loop_facts(
    tmp_path,
) -> None:
    bundle, snapshot, _kernel = _bare_bundle(tmp_path)
    artifact = bundle.store.artifacts(snapshot.id)[0]
    parser_attrs = {
        "loop_kind": "for",
        "trip_count": 8,
        "loop_bounds": {
            "lower": 0, "upper": 8, "step": 1,
            "comparison": "lt", "upper_inclusive": False,
        },
        "static_feature_domain_complete": True,
        "static_feature_domain_contract": (
            "hlsgraph.source.libclang.loop_fact_domain.v1"
        ),
        "static_feature_parser": "source.libclang",
        "static_feature_parser_version": "4",
        "static_feature_unparsed_construct_count": 0,
        "static_feature_artifact_id": artifact.id,
        "static_feature_artifact_sha256": artifact.sha256,
    }

    class SameNamedSourcePlugin:
        name = "source.libclang"
        version = "4"

        @staticmethod
        def supports(_context):
            return True

        @staticmethod
        def extract(_context):
            graph = CanonicalGraph(snapshot.id)
            graph.add_entity(Entity(
                kind="hls.loop",
                name="loop",
                qualified_name="dut::loop",
                snapshot_id=snapshot.id,
                stage="ast",
                attrs=dict(parser_attrs),
                anchors=[SourceAnchor(artifact.id)],
            ))
            return ExtractionResult(graph=graph)

    context = ExtractionContext(
        project_root=tmp_path,
        manifest=bundle.manifest,
        snapshot=snapshot,
        artifacts={artifact.id: artifact},
        allow_degraded=True,
    )
    extracted = ExtractionPipeline([SameNamedSourcePlugin()]).run(context)
    loop_facts = [
        item for item in extracted.derivations
        if item.predicate in {
            "feature.trip_count", "feature.loop_bounds",
        }
    ]
    assert len(loop_facts) == 2
    assert all(
        item.completeness == Completeness.PARTIAL
        and item.value is not None
        for item in loop_facts
    )
    assert extracted._index_origin_capability is None
