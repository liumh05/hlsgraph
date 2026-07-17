from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

import pytest

from hlsgraph.bundle import GraphBundle
from hlsgraph.extract import ExtractionResult, LibClangExtractor
from hlsgraph.graph import CanonicalGraph
from hlsgraph.manifest import minimal_manifest
from hlsgraph.model import (
    ArtifactRef,
    AuthorityClass,
    DesignSnapshot,
    Entity,
    Observation,
    PredictionEnvelope,
    Relation,
)
from hlsgraph.sdk import Project
from hlsgraph.store import StoreError


def _bundle(root: Path, *, project_id: str) -> tuple[
    GraphBundle, DesignSnapshot, ArtifactRef, Entity, CanonicalGraph,
]:
    root.mkdir(parents=True)
    (root / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    bundle = GraphBundle.create(
        root,
        minimal_manifest(project_id, "release boundary fixture", "dut", "kernel.cpp"),
    )
    snapshot = bundle.snapshot()
    source = bundle.store.artifacts(snapshot.id)[0]
    kernel = Entity(
        "hls.kernel", "dut", snapshot.id, qualified_name="dut", stage="ast",
    )
    graph = CanonicalGraph(snapshot.id)
    graph.add_entity(kernel)
    return bundle, snapshot, source, kernel, graph


def _assert_sqlite_does_not_contain(bundle: GraphBundle, secret: str) -> None:
    needle = secret.encode("utf-8")
    for path in bundle.store.path.parent.glob(f"{bundle.store.path.name}*"):
        assert needle not in path.read_bytes(), f"private sentinel leaked into {path.name}"


def _standard_extract(_self, context) -> ExtractionResult:
    graph = CanonicalGraph(context.snapshot.id)
    graph.add_entity(Entity(
        "hls.kernel",
        context.manifest.build.top,
        context.snapshot.id,
        qualified_name=context.manifest.build.top,
        stage="ast",
    ))
    return ExtractionResult(
        graph=graph,
        capabilities=["source.ast"],
        coverage={"fidelity": "libclang_test_double"},
    )


def test_libclang_runtime_identity_describes_available_native_runtime() -> None:
    identity = LibClangExtractor.runtime_identity()

    assert isinstance(identity.get("available"), bool)
    if not identity["available"]:
        assert identity.get("reason") in {
            "python_binding_unavailable", "native_runtime_unavailable",
        }
        return

    assert isinstance(identity.get("python_distribution_version"), str)
    assert identity["python_distribution_version"]
    assert isinstance(identity.get("native_version"), str)
    assert identity["native_version"].strip()
    assert re.fullmatch(r"[0-9a-f]{64}", identity.get("native_sha256", ""))
    assert re.fullmatch(r"[0-9a-f]{64}", identity.get("binding_sha256", ""))
    assert isinstance(identity.get("native_size"), int)
    assert identity["native_size"] > 0
    assert "path" not in " ".join(identity).casefold()


def test_libclang_runtime_identity_changes_standard_snapshot_and_profile(
    tmp_path, monkeypatch,
) -> None:
    root = tmp_path / "runtime-profile"
    root.mkdir()
    (root / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    project = Project(GraphBundle.create(
        root,
        minimal_manifest(
            "test.release.runtime_profile", "runtime profile", "dut", "kernel.cpp",
        ),
    ))
    monkeypatch.setattr(LibClangExtractor, "extract", _standard_extract)

    runtime_a = {
        "available": True,
        "python_distribution_version": "test-a",
        "native_version": "clang test-a",
        "native_sha256": "a" * 64,
        "native_size": 1,
        "binding_sha256": "b" * 64,
    }
    runtime_b = {
        **runtime_a,
        "native_version": "clang test-b",
        "native_sha256": "c" * 64,
    }
    monkeypatch.setattr(
        LibClangExtractor, "runtime_identity", staticmethod(lambda: dict(runtime_a)),
    )
    first = project.index(degraded=False)
    assert first.success
    first_snapshot = project.bundle.store.snapshot(first.snapshot_id)
    first_graph = project.service(first.snapshot_id).graph()

    monkeypatch.setattr(
        LibClangExtractor, "runtime_identity", staticmethod(lambda: dict(runtime_b)),
    )
    second = project.index(degraded=False)
    assert second.success
    second_snapshot = project.bundle.store.snapshot(second.snapshot_id)
    second_graph = project.service(second.snapshot_id).graph()

    assert first.snapshot_id != second.snapshot_id
    assert first_snapshot.extraction_hash != second_snapshot.extraction_hash
    first_source = next(
        item for item in first_graph.metadata["extractor_identities"]
        if item["name"] == "source.libclang"
    )
    second_source = next(
        item for item in second_graph.metadata["extractor_identities"]
        if item["name"] == "source.libclang"
    )
    assert first_source["runtime"] == runtime_a
    assert second_source["runtime"] == runtime_b


def test_mutated_entity_attrs_privacy_is_revalidated_before_graph_write(tmp_path) -> None:
    bundle, _snapshot, _source, kernel, graph = _bundle(
        tmp_path / "private-entity", project_id="test.release.private_entity",
    )
    secret = "PRIVATE_ENTITY_SOURCE_SENTINEL_9f63"
    kernel.attrs["source_text"] = secret

    with pytest.raises(StoreError, match="embedded body"):
        bundle.store.save_graph(graph)
    _assert_sqlite_does_not_contain(bundle, secret)


def test_mutated_relation_attrs_privacy_is_revalidated_before_graph_write(tmp_path) -> None:
    bundle, snapshot, _source, kernel, graph = _bundle(
        tmp_path / "private-relation", project_id="test.release.private_relation",
    )
    process = graph.add_entity(Entity(
        "hls.process", "compute", snapshot.id, qualified_name="dut::compute", stage="ast",
    ))
    relation = graph.add_relation(Relation(
        kernel.id, process.id, "hls.contains", snapshot.id, stage="ast",
    ))
    secret = "PRIVATE_RELATION_SOURCE_SENTINEL_44a1"
    relation.attrs["raw_source"] = secret

    with pytest.raises(StoreError, match="embedded body"):
        bundle.store.save_graph(graph)
    _assert_sqlite_does_not_contain(bundle, secret)


def test_mutated_artifact_metadata_privacy_is_revalidated_before_write(tmp_path) -> None:
    bundle, snapshot, _source, _kernel, _graph = _bundle(
        tmp_path / "private-artifact", project_id="test.release.private_artifact",
    )
    artifact = ArtifactRef(
        "test.private_artifact", "evidence/private.bin", "a" * 64, 1,
    )
    secret = "PRIVATE_ARTIFACT_SOURCE_SENTINEL_1d72"
    artifact.metadata["source_text"] = secret

    with pytest.raises(StoreError, match="embedded body"):
        bundle.store.add_artifact(snapshot.id, artifact)
    _assert_sqlite_does_not_contain(bundle, secret)


def test_mutated_observation_metadata_privacy_is_revalidated_before_write(tmp_path) -> None:
    bundle, snapshot, _source, kernel, graph = _bundle(
        tmp_path / "private-observation", project_id="test.release.private_observation",
    )
    bundle.store.save_graph(graph)
    observation = Observation(
        snapshot.id,
        kernel.id,
        "test.static_value",
        1,
        "ast",
        AuthorityClass.STATIC_FACT,
    )
    secret = "PRIVATE_OBSERVATION_SOURCE_SENTINEL_628e"
    observation.metadata["raw_source"] = secret

    with pytest.raises(StoreError, match="embedded body"):
        bundle.store.add_observations([observation])
    _assert_sqlite_does_not_contain(bundle, secret)


def test_mutated_prediction_metadata_privacy_is_revalidated_before_write(tmp_path) -> None:
    bundle, snapshot, _source, kernel, graph = _bundle(
        tmp_path / "private-prediction", project_id="test.release.private_prediction",
    )
    bundle.store.save_graph(graph)
    prediction = PredictionEnvelope(
        snapshot.id,
        kernel.id,
        "prediction.latency_cycles",
        10,
        "test.model",
        "1",
        "features.v1",
    )
    secret = "PRIVATE_PREDICTION_SOURCE_SENTINEL_b307"
    prediction.metadata["source_text"] = secret

    with pytest.raises(StoreError, match="embedded body"):
        bundle.store.add_prediction(prediction)
    _assert_sqlite_does_not_contain(bundle, secret)


@pytest.mark.parametrize("uri", [
    "/PRIVATE_ABSOLUTE_ARTIFACT_PATH_SENTINEL_021e.bin",
    "C" + ":/PRIVATE_DRIVE_ARTIFACT_PATH_SENTINEL_725b.bin",
    "evidence/../PRIVATE_TRAVERSAL_ARTIFACT_PATH_SENTINEL_c125.bin",
])
def test_artifact_uri_must_be_safe_project_relative_and_never_reach_sqlite(
    tmp_path, uri,
) -> None:
    bundle, snapshot, _source, _kernel, _graph = _bundle(
        tmp_path / "unsafe-uri", project_id="test.release.unsafe_artifact_uri",
    )
    rejected = False
    try:
        artifact = ArtifactRef("test.unsafe_path", uri, "d" * 64, 1)
        bundle.store.add_artifact(snapshot.id, artifact)
    except (ValueError, StoreError):
        rejected = True

    leaked = any(
        uri.encode("utf-8") in path.read_bytes()
        for path in bundle.store.path.parent.glob(f"{bundle.store.path.name}*")
    )
    assert rejected and not leaked, (
        f"unsafe ArtifactRef.uri was accepted={not rejected}, leaked_to_sqlite={leaked}: {uri}"
    )


@pytest.mark.parametrize("kind", [
    "entity", "relation", "artifact", "observation", "prediction",
])
def test_store_rejects_post_construction_mutation_of_stable_identity(
    tmp_path, kind,
) -> None:
    bundle, snapshot, _source, kernel, graph = _bundle(
        tmp_path / kind, project_id=f"test.release.stable_identity_{kind}",
    )

    if kind == "entity":
        kernel.qualified_name = "renamed::dut"
        write = lambda: bundle.store.save_graph(graph)
    elif kind == "relation":
        process = graph.add_entity(Entity(
            "hls.process", "compute", snapshot.id,
            qualified_name="dut::compute", stage="ast",
        ))
        relation = graph.add_relation(Relation(
            kernel.id, process.id, "hls.contains", snapshot.id,
            stage="ast", attrs={"ordinal": 1},
        ))
        relation.attrs["ordinal"] = 2
        write = lambda: bundle.store.save_graph(graph)
    elif kind == "artifact":
        artifact = ArtifactRef(
            "test.identity_artifact", "evidence/identity.bin", "e" * 64, 1,
            role="before",
        )
        artifact.role = "after"
        write = lambda: bundle.store.add_artifact(snapshot.id, artifact)
    elif kind == "observation":
        bundle.store.save_graph(graph)
        observation = Observation(
            snapshot.id, kernel.id, "test.identity_value", 1, "ast",
            AuthorityClass.STATIC_FACT,
        )
        observation.value = 2
        write = lambda: bundle.store.add_observations([observation])
    else:
        bundle.store.save_graph(graph)
        prediction = PredictionEnvelope(
            snapshot.id, kernel.id, "prediction.identity_value", 1,
            "test.model", "1", "features.v1",
        )
        prediction.value = 2
        write = lambda: bundle.store.add_prediction(prediction)

    with pytest.raises(StoreError, match="stable id"):
        write()


def test_snapshot_artifact_hash_closure_must_match_attached_artifacts(tmp_path) -> None:
    bundle, snapshot, source, _kernel, _graph = _bundle(
        tmp_path / "snapshot-closure", project_id="test.release.snapshot_closure",
    )
    bad_snapshot = replace(
        snapshot,
        artifact_hashes={source.uri: "f" * 64},
        action_id="test.closure_mismatch",
        id="",
    )

    with pytest.raises(StoreError, match="artifact hash closure"):
        bundle.store.save_snapshot(bad_snapshot, [source])
    with pytest.raises(KeyError):
        bundle.store.snapshot(bad_snapshot.id)
