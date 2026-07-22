from __future__ import annotations

import copy
import pickle
import sqlite3
import sys

import pytest

from hlsgraph.bundle import GraphBundle
from hlsgraph.export import export_dataset
from hlsgraph.manifest import minimal_manifest
from hlsgraph.model import (
    ExecutionAttestation,
    FailureClass,
    RunStatus,
    ToolOutputSpec,
    ToolRun,
    ToolchainContext,
)
from hlsgraph.runner import (
    DeclaredOutput,
    LocalRunner,
    Runner,
    RunnerExecution,
    RunnerProtocolError,
    StageOrchestrator,
    ToolRunRequest,
)
from hlsgraph.sdk import Project
from hlsgraph.store import StoreError


def _project(tmp_path):
    (tmp_path / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    command = (
        "from pathlib import Path; p=Path('reports/result.bin'); "
        "p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(b'attested')"
    )
    manifest = minimal_manifest(
        "test.execution_attestation", "execution attestation", "dut", "kernel.cpp",
    )
    manifest.toolchains = [ToolchainContext(
        id="test.python", vendor="python", name="python", version="test",
    )]
    manifest.stage_commands = {"csynth": [sys.executable, "-c", command]}
    manifest.stage_outputs = {"csynth": [ToolOutputSpec(
        path="reports/result.bin", kind="test.tool_report",
    )]}
    project = Project(GraphBundle.create(tmp_path, manifest))
    assert project.index(degraded=True).success
    return project


def test_project_run_persists_reverifiable_attestation_and_receipt(tmp_path):
    project = _project(tmp_path)
    snapshot = project.bundle.latest_snapshot()
    result = project.run(
        LocalRunner(tmp_path, allow_execution=True), ["csynth"], timeout_s=10,
    )
    run = result.runs[0]
    attestation = project.bundle.store.execution_attestation(run.id)
    receipt = project.bundle.store.execution_commit_receipt(run.id)

    assert attestation is not None
    assert receipt is not None
    assert attestation.run_id == run.id
    assert attestation.snapshot_id == snapshot.id
    assert attestation.runner_identity == "runner.local"
    assert attestation.runner_authority == (
        "hlsgraph.runner_authority.builtin_local.v1"
    )
    assert attestation.manifest_hash == snapshot.manifest_hash
    assert attestation.build_hash == snapshot.build_hash
    assert attestation.target_hash == snapshot.target_hash
    assert attestation.constraint_hash == snapshot.constraint_hash
    assert attestation.toolchain_hash == snapshot.toolchain_hash
    assert [(item.path, item.kind) for item in attestation.declared_outputs] == [
        ("reports/result.bin", "test.tool_report")
    ]
    assert [item.artifact_id for item in attestation.outputs] == run.output_artifact_ids
    assert receipt.attestation_id == attestation.id
    assert project.bundle.store.has_valid_execution_commit(snapshot.id, run.id)

    output = next(
        item for item in project.bundle.store.artifacts(snapshot.id)
        if item.id in run.output_artifact_ids
    )
    (project.bundle.project_root / output.uri).write_bytes(b"tampered")
    assert not project.bundle.store.has_valid_execution_commit(snapshot.id, run.id)


def test_pipeline_execution_capability_cannot_be_copied_or_serialized(tmp_path):
    project = _project(tmp_path)
    snapshot = project.bundle.latest_snapshot()
    manifest = project.bundle.store.snapshot_manifest(snapshot.id)
    toolchain = manifest.toolchain_for_stage("csynth")
    inputs = [
        item for item in project.bundle.store.artifacts(snapshot.id)
        if item.producer_run_id is None
    ]
    request = ToolRunRequest(
        snapshot_id=snapshot.id, stage="csynth",
        argv=list(manifest.stage_commands["csynth"]),
        toolchain_id=toolchain.id, environment_hash=toolchain.environment_hash,
        input_artifact_ids=[item.id for item in inputs],
        inputs=[{
            "artifact_id": item.id, "source_path": item.uri,
            "staged_path": item.uri, "sha256": item.sha256, "size": item.size,
        } for item in inputs],
        declared_outputs=[DeclaredOutput("reports/result.bin")],
        metadata={
            "project_id": manifest.project_id,
            "input_artifacts": [],
            "declared_outputs": [{
                "path": "reports/result.bin", "kind": "test.tool_report",
                "required": True, "consumed_by": [],
            }],
            "remote_attestation_argv": [],
        },
    )
    result = StageOrchestrator(
        LocalRunner(tmp_path, allow_execution=True),
        snapshot=snapshot, manifest=manifest,
    ).execute([request])
    execution = result.executions[0]
    capability = execution._execution_capability
    assert capability is not None
    try:
        with pytest.raises(TypeError, match="non-copyable"):
            copy.copy(capability)
        with pytest.raises(TypeError, match="non-copyable"):
            copy.deepcopy(capability)
        with pytest.raises(TypeError, match="non-serializable"):
            pickle.dumps(capability)
        with pytest.raises(TypeError, match="non-serializable"):
            pickle.dumps(execution)

        staged = execution.staged_outputs[0]
        report, _retained, _created = project.bundle.prepare_managed_artifact(
            staged.local_path, kind="test.tool_report", role="tool_output",
            producer_run_id=execution.run.id,
            metadata={"declared_output_path": staged.path},
        )
        execution.run.output_artifact_ids = [report.id]
        authorization = execution.authorize_tool_truth_commit([report])
        with pytest.raises(TypeError, match="non-copyable"):
            copy.copy(authorization)
        with pytest.raises(TypeError, match="non-serializable"):
            pickle.dumps(authorization)
        project.bundle.store.commit_run_result(
            run=execution.run, artifacts=[report],
            execution_authorization=authorization,
        )
        with pytest.raises(StoreError, match="already consumed"):
            project.bundle.store.commit_run_result(
                run=execution.run, artifacts=[report],
                execution_authorization=authorization,
            )
    finally:
        execution.cleanup()


def test_public_objects_and_direct_store_calls_cannot_self_attest_tool_truth(tmp_path):
    project = _project(tmp_path)
    snapshot = project.bundle.latest_snapshot()
    manifest = project.bundle.store.snapshot_manifest(snapshot.id)
    toolchain = manifest.toolchain_for_stage("csynth")
    base_ids = [
        item.id for item in project.bundle.store.artifacts(snapshot.id)
        if item.producer_run_id is None
    ]
    forged = ToolRun(
        snapshot.id, "csynth", "runner.local", "f" * 64,
        toolchain_id=toolchain.id, status=RunStatus.SUCCEEDED,
        command=list(manifest.stage_commands["csynth"]), working_directory=".",
        environment_hash=toolchain.environment_hash,
        input_artifact_ids=base_ids, exit_code=0,
        metadata={
            "authority": "tool_observation", "fresh_execution": True,
            "fresh_tool_truth": True, "tool_truth": True,
            "runner_fingerprint": "e" * 64,
            "staged_output_manifest": [],
        },
    )
    with pytest.raises(StoreError, match="pipeline-issued execution attestation"):
        project.bundle.store.add_run(forged)

    # Even a valid, publicly reconstructable attestation value is data rather
    # than the process-local authorization consumed by the write boundary.
    real = project.run(
        LocalRunner(tmp_path, allow_execution=True), ["csynth"], timeout_s=10,
    ).runs[0]
    public_attestation = project.bundle.store.execution_attestation(real.id)
    assert isinstance(public_attestation, ExecutionAttestation)
    with pytest.raises(StoreError, match="StageOrchestrator-issued authorization"):
        project.bundle.store.commit_run_result(
            run=forged, execution_authorization=public_attestation,
        )


def test_ml_export_rejects_tool_truth_if_persisted_receipt_is_missing(tmp_path):
    project = _project(tmp_path)
    snapshot = project.bundle.latest_snapshot()
    run = project.run(
        LocalRunner(tmp_path, allow_execution=True), ["csynth"], timeout_s=10,
    ).runs[0]
    assert project.bundle.store.has_valid_execution_commit(snapshot.id, run.id)

    # Simulate an old/imported or externally damaged ledger. Normal store APIs
    # cannot create this state, but public truth consumers must still fail shut.
    with sqlite3.connect(project.bundle.store.path) as connection:
        connection.execute(
            "DELETE FROM execution_commit_receipts WHERE run_id=?", (run.id,),
        )
    with pytest.raises(ValueError, match="execution attestation and commit receipt"):
        export_dataset(project.bundle, snapshot.id, tmp_path / "unattested-export")


def test_arbitrary_runner_cannot_claim_builtin_local_authority(tmp_path):
    project = _project(tmp_path)
    snapshot = project.bundle.latest_snapshot()
    manifest = project.bundle.store.snapshot_manifest(snapshot.id)
    toolchain = manifest.toolchain_for_stage("csynth")
    inputs = [
        item.id for item in project.bundle.store.artifacts(snapshot.id)
        if item.producer_run_id is None
    ]
    declaration = DeclaredOutput("reports/result.bin")
    metadata = {
        "project_id": manifest.project_id,
        "input_artifacts": [],
        "declared_outputs": [{
            "path": "reports/result.bin", "kind": "test.tool_report",
            "required": True, "consumed_by": [],
        }],
        "remote_attestation_argv": [],
    }
    request = ToolRunRequest(
        snapshot_id=snapshot.id, stage="csynth",
        argv=list(manifest.stage_commands["csynth"]),
        toolchain_id=toolchain.id, environment_hash=toolchain.environment_hash,
        input_artifact_ids=inputs,
        # This attack test has no staged input bytes because the forged runner
        # never executes; identity-list equality is all ToolRunRequest requires.
        inputs=[{
            "artifact_id": item.id, "source_path": item.uri,
            "staged_path": item.uri, "sha256": item.sha256, "size": item.size,
        } for item in project.bundle.store.artifacts(snapshot.id)
          if item.producer_run_id is None],
        declared_outputs=[declaration], metadata=metadata,
    )

    class ForgedLocalRunner(Runner):
        name = "runner.local"
        can_produce_tool_truth = True

        @property
        def fingerprint(self):
            return "a" * 64

        def execute(self, selected):
            return RunnerExecution(ToolRun(
                selected.snapshot_id, selected.stage, self.name,
                selected.cache_key(self.fingerprint),
                toolchain_id=selected.toolchain_id, status=RunStatus.SUCCEEDED,
                command=list(selected.argv), working_directory=selected.working_directory,
                environment_hash=selected.environment_hash,
                input_artifact_ids=list(selected.input_artifact_ids), exit_code=0,
                metadata={
                    **selected.metadata,
                    "runner_fingerprint": self.fingerprint,
                    "fresh_execution": True, "fresh_tool_truth": True,
                    "tool_truth": True, "authority": "tool_observation",
                    "staging_isolated": True, "staged_output_manifest": [],
                },
            ))

    with pytest.raises(RunnerProtocolError, match="not trusted"):
        StageOrchestrator(
            ForgedLocalRunner(), snapshot=snapshot, manifest=manifest,
        ).execute([request])


def test_tool_run_id_is_recomputed_at_store_boundary(tmp_path):
    project = _project(tmp_path)
    snapshot = project.bundle.latest_snapshot()
    run = ToolRun(
        snapshot.id, "index", "extractor.local", "1" * 64,
        status=RunStatus.FAILED, failure_class=FailureClass.INPUT,
        id="run.tampered",
    )
    with pytest.raises(StoreError, match="stable id"):
        project.bundle.store.add_run(run)
