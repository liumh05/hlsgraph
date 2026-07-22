from __future__ import annotations

import hashlib

import hlsgraph.sdk as sdk_module
import pytest

from hlsgraph.bundle import BundleError, GraphBundle
from hlsgraph.extract import ExtractionResult
from hlsgraph.graph import CanonicalGraph
from hlsgraph.manifest import minimal_manifest
from hlsgraph.model import (
    AuthorityClass,
    Derivation,
    EvidenceKind,
    EvidenceRef,
    Observation,
    RunStatus,
    ToolOutputSpec,
    ToolRun,
)
from hlsgraph.runner import RunnerExecution, StagedOutput
from hlsgraph.sdk import Project


def test_declared_run_output_rebinds_typed_evidence_before_atomic_commit(
    tmp_path, monkeypatch,
) -> None:
    source = tmp_path / "kernel.cpp"
    source.write_text("void dut() {}\n", encoding="utf-8")
    manifest = minimal_manifest(
        "test.run_output_evidence", "run output evidence", "dut", "kernel.cpp",
    )
    project = Project(GraphBundle.create(tmp_path, manifest))
    indexed = project.index(degraded=True)
    assert indexed.success
    snapshot = project.bundle.latest_snapshot()
    assert snapshot is not None

    captured: dict[str, str] = {}

    class FixtureExtractor:
        name = "test.run_output_evidence"
        version = "1"

        @staticmethod
        def supports(_context) -> bool:
            return True

        @staticmethod
        def extract(context) -> ExtractionResult:
            graph = context.options["existing_graph"]
            subject = next(
                item.id for item in graph.entities.values() if item.kind == "hls.kernel"
            )
            artifact = next(iter(context.artifacts.values()))
            observation = Observation(
                snapshot_id=context.snapshot.id,
                subject_id=subject,
                predicate="qor.fixture_cycles",
                value=7,
                unit="cycle",
                stage="csynth",
                authority=AuthorityClass.TOOL_OBSERVATION,
                artifact_id=artifact.id,
            )
            direct = Derivation(
                snapshot_id=context.snapshot.id,
                subject_id=subject,
                predicate="derived.fixture_double",
                value=14,
                algorithm="fixture.double",
                algorithm_version="1",
                evidence_refs=[EvidenceRef(
                    kind=EvidenceKind.OBSERVATION,
                    target_id=observation.id,
                    snapshot_id=context.snapshot.id,
                )],
            )
            chained = Derivation(
                snapshot_id=context.snapshot.id,
                subject_id=subject,
                predicate="derived.fixture_plus_one",
                value=15,
                algorithm="fixture.plus_one",
                algorithm_version="1",
                evidence_refs=[EvidenceRef(
                    kind=EvidenceKind.DERIVATION,
                    target_id=direct.id,
                    snapshot_id=context.snapshot.id,
                )],
            )
            captured.update(
                observation=observation.id, direct=direct.id, chained=chained.id,
            )
            # Deliberately return the dependency after its consumer.  The SDK must
            # rebuild stable IDs and persist the same-batch chain topologically.
            return ExtractionResult(
                graph=CanonicalGraph(context.snapshot.id),
                observations=[observation],
                derivations=[chained, direct],
            )

    class UnsupportedExtractor:
        @staticmethod
        def supports(_context) -> bool:
            return False

    monkeypatch.setattr(sdk_module, "VitisReportExtractor", FixtureExtractor)
    monkeypatch.setattr(sdk_module, "VivadoReportExtractor", UnsupportedExtractor)

    staging = tmp_path / "staging"
    report = staging / "reports" / "fixture.rpt"
    report.parent.mkdir(parents=True)
    report.write_bytes(b"fixture report\n")
    digest = hashlib.sha256(report.read_bytes()).hexdigest()
    run = ToolRun(
        snapshot_id=snapshot.id,
        stage="csynth",
        backend="runner.fixture",
        request_hash="a" * 64,
        status=RunStatus.SUCCEEDED,
        metadata={"workload_id": "tb.default", "testcase_id": "case.default"},
    )
    output = ToolOutputSpec(
        path="reports/fixture.rpt",
        kind="fixture.report",
    )
    execution = RunnerExecution(
        run=run,
        staged_outputs=[StagedOutput(
            path=output.path,
            local_path=report,
            size=report.stat().st_size,
            sha256=digest,
        )],
        staging_directory=staging,
    )

    conflicting_output = ToolOutputSpec(
        path=output.path, kind=output.kind,
        metadata={"workload_id": "tb.other"},
    )
    with pytest.raises(BundleError, match="workload_id conflicts"):
        project._commit_declared_run_outputs(
            snapshot, manifest, run, [conflicting_output], execution,
        )

    result = project._commit_declared_run_outputs(
        snapshot, manifest, run, [output], execution,
    )

    managed = next(
        item for item in project.bundle.store.artifacts(snapshot.id)
        if item.producer_run_id == run.id
    )
    assert managed.metadata["workload_id"] == "tb.default"
    assert managed.metadata["testcase_id"] == "case.default"
    observation = result.observations[0]
    direct, chained = result.derivations
    assert observation.id != captured["observation"]
    assert observation.run_id == run.id
    assert direct.id != captured["direct"]
    assert direct.input_observation_ids == [observation.id]
    assert direct.evidence_refs[0].target_id == observation.id
    assert direct.evidence_refs[0].snapshot_id == snapshot.id
    assert chained.id != captured["chained"]
    assert chained.evidence_refs[0].target_id == direct.id
    assert chained.evidence_refs[0].snapshot_id == snapshot.id

    stored = project.bundle.store.derivations(snapshot.id)
    by_predicate = {item["predicate"]: item for item in stored}
    assert by_predicate["derived.fixture_double"]["evidence_refs"][0][
        "target_id"
    ] == observation.id
    assert by_predicate["derived.fixture_plus_one"]["evidence_refs"][0][
        "target_id"
    ] == direct.id
