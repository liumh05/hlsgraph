from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import Callable

import pytest

from hlsgraph.bundle import BundleError, GraphBundle
from hlsgraph.extract import RegexSourceExtractor
from hlsgraph.manifest import minimal_manifest
from hlsgraph.model import (
    FailureClass,
    RunStatus,
    ToolOutputSpec,
    ToolRun,
    ToolchainContext,
    hash_artifact_bytes,
    stable_hash,
    utc_now,
)
from hlsgraph.runner import (
    FakeRunner,
    LocalRunner,
    ReplayRunner,
    Runner,
    RunnerProtocolError,
    StageOrchestrator,
    ToolRunRequest,
)
from hlsgraph.sdk import Project


def _request(**overrides: object) -> ToolRunRequest:
    values: dict[str, object] = {
        "snapshot_id": "snapshot_fixture",
        "stage": "csynth",
        "argv": ["fixture-tool", "--run"],
        "working_directory": ".",
        "environment": {"MODE": "test"},
        "environment_hash": "a" * 64,
        "toolchain_id": "test.toolchain",
        "input_artifact_ids": ["artifact_fixture"],
        "timeout_s": 10,
        "metadata": {"fixture": "expected"},
    }
    values.update(overrides)
    return ToolRunRequest(**values)  # type: ignore[arg-type]


def _successful_run(request: ToolRunRequest, runner: Runner) -> ToolRun:
    event_time = utc_now()
    return ToolRun(
        snapshot_id=request.snapshot_id,
        stage=request.stage,
        backend=runner.name,
        request_hash=request.cache_key(runner.fingerprint),
        toolchain_id=request.toolchain_id,
        status=RunStatus.SUCCEEDED,
        command=list(request.argv),
        working_directory=request.working_directory,
        environment_hash=request.environment_hash,
        input_artifact_ids=list(request.input_artifact_ids),
        failure_class=FailureClass.NONE,
        exit_code=0,
        started_at=event_time,
        finished_at=event_time,
        elapsed_s=0.0,
        metadata={
            **request.metadata,
            "runner_fingerprint": runner.fingerprint,
            "fresh_execution": True,
            "fresh_tool_truth": True,
            "authority": "tool_observation",
            "tool_truth": True,
        },
    )


class _CallbackRunner(Runner):
    # This callback is a deterministic stand-in for the approved local backend;
    # using a made-up backend must not grant a run real-tool truth authority.
    name = "runner.local"
    provides_local_output_bytes = True

    def __init__(self, callback: Callable[[ToolRunRequest], ToolRun]):
        self.callback = callback
        self.calls: list[ToolRunRequest] = []

    @property
    def fingerprint(self) -> str:
        return stable_hash({"runner": self.name, "version": 1})

    def execute(self, request: ToolRunRequest) -> ToolRun:
        self.calls.append(request)
        return self.callback(request)


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda run: replace(run, id="", snapshot_id="snapshot_wrong"),
        lambda run: replace(run, id="", stage="rtl_cosim"),
        lambda run: replace(run, id="", request_hash="b" * 64),
        lambda run: replace(
            run,
            id="",
            metadata={**run.metadata, "fixture": "wrong"},
        ),
    ],
    ids=["snapshot", "stage", "request_hash", "request_metadata"],
)
def test_orchestrator_rejects_runner_identity_and_metadata_spoofing(corrupt):
    request = _request()
    runner: _CallbackRunner

    def respond(received: ToolRunRequest) -> ToolRun:
        return corrupt(_successful_run(received, runner))

    runner = _CallbackRunner(respond)
    with pytest.raises(RunnerProtocolError, match="identity mismatch"):
        StageOrchestrator(runner).execute([request])


def test_orchestrator_isolates_request_from_in_place_runner_mutation():
    request = _request()
    original_snapshot = request.snapshot_id
    original_stage = request.stage
    original_metadata = dict(request.metadata)
    runner: _CallbackRunner

    def mutate_and_match(received: ToolRunRequest) -> ToolRun:
        received.snapshot_id = "snapshot_mutated_by_runner"
        received.stage = "post_route"
        received.metadata["fixture"] = "mutated_by_runner"
        return _successful_run(received, runner)

    runner = _CallbackRunner(mutate_and_match)
    with pytest.raises(RunnerProtocolError, match="identity mismatch"):
        StageOrchestrator(runner).execute([request])

    assert runner.calls[0] is not request
    assert request.snapshot_id == original_snapshot
    assert request.stage == original_stage
    assert request.metadata == original_metadata


def test_runner_freshness_flags_distinguish_execution_fake_and_replay(tmp_path):
    request = _request(
        input_artifact_ids=[],
        argv=[sys.executable, "-c", "print('fresh')"],
        environment={},
        environment_hash=None,
    )
    local = LocalRunner(
        tmp_path, allow_execution=True, inherit_environment=False,
    ).execute(request)
    assert local.status == RunStatus.SUCCEEDED
    assert local.metadata["fresh_execution"] is True
    assert local.metadata["fresh_tool_truth"] is True

    fake_runner = FakeRunner()
    fake = fake_runner.execute(request)
    assert fake.metadata["fresh_execution"] is False
    assert fake.metadata["fresh_tool_truth"] is False

    source_key = request.cache_key(fake_runner.fingerprint)
    replay = ReplayRunner(
        {source_key: fake}, source_runner_fingerprint=fake_runner.fingerprint,
    ).execute(request)
    assert replay.status == RunStatus.CACHED
    assert replay.metadata["fresh_execution"] is False
    assert replay.metadata["fresh_tool_truth"] is False


def _project(
    root: Path,
    *,
    project_id: str,
    stages: list[str] | None = None,
    outputs: dict[str, list[ToolOutputSpec]] | None = None,
) -> Project:
    root.mkdir(parents=True)
    (root / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    manifest = minimal_manifest(project_id, "freshness fixture", "dut", "kernel.cpp")
    if stages:
        manifest.stage_commands = {stage: ["fixture-tool", stage] for stage in stages}
        manifest.toolchains = [ToolchainContext(
            id="test.fixture", vendor="test", name="fixture", version="1",
        )]
        manifest.stage_outputs = dict(outputs or {})
    project = Project(GraphBundle.create(root, manifest))
    return project


def test_index_source_change_during_extractor_commits_only_failed_candidate(
    tmp_path, monkeypatch,
):
    project = _project(
        tmp_path / "index-drift", project_id="test.index_drift",
    )
    source = project.bundle.project_root / "kernel.cpp"
    original_extract = RegexSourceExtractor.extract

    def extract_then_edit(self, context):
        result = original_extract(self, context)
        source.write_text("void dut() { int changed = 1; }\n", encoding="utf-8")
        return result

    monkeypatch.setattr(RegexSourceExtractor, "extract", extract_then_edit)
    result = project.index(degraded=True)

    assert result.success is False
    assert project.bundle.latest_snapshot() is None
    candidate = project.bundle.store.latest_candidate("test.index_drift")
    assert candidate is not None and candidate.id == result.snapshot_id
    assert project.bundle.store.has_graph(candidate.id) is False
    diagnostics = project.bundle.store.diagnostics(candidate.id)
    assert any(item.code == "extractor.snapshot_changed" for item in diagnostics)
    runs = project.bundle.store.runs(candidate.id)
    assert len(runs) == 1
    assert runs[0].stage == "index"
    assert runs[0].status == RunStatus.FAILED
    assert runs[0].failure_class == FailureClass.INPUT


def _assert_input_failure(run: ToolRun, stage: str) -> None:
    assert run.stage == stage
    assert run.status == RunStatus.FAILED
    assert run.failure_class == FailureClass.INPUT
    assert run.gates == []
    assert run.output_artifact_ids == []
    assert run.metadata["tool_truth"] is False
    assert run.metadata["fresh_tool_truth"] is False
    assert run.metadata["input_validation_failed"] is True


def test_run_detects_base_input_drift_before_next_stage_and_stops(
    tmp_path, monkeypatch,
):
    stages = ["csim", "csynth", "post_route"]
    project = _project(
        tmp_path / "base-pre", project_id="test.base_pre", stages=stages,
    )
    assert project.index(degraded=True).success
    source = project.bundle.project_root / "kernel.cpp"
    runner: _CallbackRunner
    runner = _CallbackRunner(lambda request: _successful_run(request, runner))
    original_add_run = project.bundle.store.add_run
    changed = False

    def add_then_change_source(run: ToolRun) -> None:
        nonlocal changed
        original_add_run(run)
        if run.stage == "csim" and not changed:
            source.write_text("void dut() { int changed = 1; }\n", encoding="utf-8")
            changed = True

    monkeypatch.setattr(project.bundle.store, "add_run", add_then_change_source)
    result = project.run(runner, stages)

    assert [item.stage for item in runner.calls] == ["csim"]
    assert result.stopped_after_stage == "csynth"
    assert len(result.runs) == 2
    _assert_input_failure(result.runs[-1], "csynth")
    persisted = project.bundle.store.runs(result.runs[-1].snapshot_id)
    _assert_input_failure(persisted[-1], "csynth")


def test_run_detects_base_input_drift_after_runner_and_stops(tmp_path):
    stages = ["csim", "csynth"]
    project = _project(
        tmp_path / "base-post", project_id="test.base_post", stages=stages,
    )
    assert project.index(degraded=True).success
    source = project.bundle.project_root / "kernel.cpp"
    runner: _CallbackRunner

    def change_source(request: ToolRunRequest) -> ToolRun:
        source.write_text("void dut() { int changed = 1; }\n", encoding="utf-8")
        return _successful_run(request, runner)

    runner = _CallbackRunner(change_source)
    result = project.run(runner, stages)

    assert [item.stage for item in runner.calls] == ["csim"]
    assert result.stopped_after_stage == "csim"
    assert len(result.runs) == 1
    _assert_input_failure(result.runs[0], "csim")
    persisted = project.bundle.store.runs(result.runs[0].snapshot_id)
    _assert_input_failure(persisted[-1], "csim")


def _chained_project(root: Path, project_id: str) -> Project:
    stages = ["csynth", "post_route", "hardware_runtime"]
    return _project(
        root,
        project_id=project_id,
        stages=stages,
        outputs={
            "csynth": [ToolOutputSpec(
                path="reports/chained.bin",
                kind="test.chained_output",
                consumed_by=["post_route"],
            )],
        },
    )


@pytest.mark.parametrize("location", ["artifact", "declared_output"])
def test_run_detects_chained_output_drift_before_consumer(
    tmp_path, monkeypatch, location,
):
    project = _chained_project(
        tmp_path / f"chain-pre-{location}", f"test.chain_pre_{location}",
    )
    assert project.index(degraded=True).success
    report = project.bundle.project_root / "reports" / "chained.bin"
    runner: _CallbackRunner

    def produce(request: ToolRunRequest) -> ToolRun:
        if request.stage == "csynth":
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_bytes(b"original-chain")
        return _successful_run(request, runner)

    runner = _CallbackRunner(produce)
    original_commit = project._commit_declared_run_outputs

    def commit_then_corrupt(snapshot, manifest, run, output_specs):
        extraction = original_commit(snapshot, manifest, run, output_specs)
        artifact = next(
            item for item in project.bundle.store.artifacts(snapshot.id)
            if item.id in run.output_artifact_ids
        )
        relative = (
            artifact.uri
            if location == "artifact"
            else artifact.metadata["declared_output_path"]
        )
        (project.bundle.project_root / relative).write_bytes(b"corrupt-chain!")
        return extraction

    monkeypatch.setattr(project, "_commit_declared_run_outputs", commit_then_corrupt)
    result = project.run(runner, ["csynth", "post_route", "hardware_runtime"])

    assert [item.stage for item in runner.calls] == ["csynth"]
    assert result.stopped_after_stage == "post_route"
    assert len(result.runs) == 2
    _assert_input_failure(result.runs[-1], "post_route")
    assert any(location in item for item in
               result.runs[-1].metadata["input_mismatch_ids"])


def test_run_detects_chained_cas_drift_after_consumer_and_stops(tmp_path):
    project = _chained_project(
        tmp_path / "chain-post", "test.chain_post",
    )
    assert project.index(degraded=True).success
    report = project.bundle.project_root / "reports" / "chained.bin"
    runner: _CallbackRunner

    def produce_then_consume(request: ToolRunRequest) -> ToolRun:
        if request.stage == "csynth":
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_bytes(b"original-chain")
        elif request.stage == "post_route":
            chained = next(
                item for item in request.metadata["input_artifacts"]
                if item["id"] not in {
                    artifact.id for artifact in project.bundle.store.artifacts(
                        request.snapshot_id,
                    ) if not artifact.producer_run_id
                }
            )
            (project.bundle.project_root / chained["uri"]).write_bytes(b"corrupt-chain!")
        return _successful_run(request, runner)

    runner = _CallbackRunner(produce_then_consume)
    result = project.run(runner, ["csynth", "post_route", "hardware_runtime"])

    assert [item.stage for item in runner.calls] == ["csynth", "post_route"]
    assert result.stopped_after_stage == "post_route"
    assert len(result.runs) == 2
    _assert_input_failure(result.runs[-1], "post_route")
    assert any(":artifact" in item for item in
               result.runs[-1].metadata["input_mismatch_ids"])


def test_add_managed_artifact_rejects_non_atomic_producer_link(tmp_path):
    project = _project(
        tmp_path / "producer-link", project_id="test.producer_link",
    )
    assert project.index(degraded=True).success
    source = project.bundle.project_root / "run-output.rpt"
    source.write_text("report\n", encoding="utf-8")

    with pytest.raises(BundleError, match="commit_run_result"):
        project.bundle.add_managed_artifact(
            source,
            kind="test.report",
            role="tool_output",
            producer_run_id="run_fixture",
        )


def test_published_cas_survives_ledger_commit_failure(tmp_path, monkeypatch):
    project = _project(
        tmp_path / "cas-orphan", project_id="test.cas_orphan",
    )
    index = project.index(degraded=True)
    assert index.success
    source = project.bundle.project_root / "orphan.rpt"
    data = b"immutable report bytes\n"
    source.write_bytes(data)
    digest = hash_artifact_bytes(data)
    target = project.bundle.root / "artifacts" / digest / source.name

    def fail_ledger_commit(*_args, **_kwargs):
        raise RuntimeError("synthetic ledger failure")

    monkeypatch.setattr(project.bundle.store, "add_artifact", fail_ledger_commit)
    with pytest.raises(RuntimeError, match="synthetic ledger failure"):
        project.bundle.add_managed_artifact(
            source, kind="test.report", role="tool_output",
        )

    assert target.read_bytes() == data
    assert all(item.uri != target.relative_to(project.bundle.project_root).as_posix()
               for item in project.bundle.store.artifacts(index.snapshot_id))
