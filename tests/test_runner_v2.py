from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hlsgraph.bundle import BundleError, GraphBundle
from hlsgraph.manifest import minimal_manifest
from hlsgraph.model import FailureClass, RunStatus, ToolchainContext
from hlsgraph.runner import (
    PROTOCOL_VERSION,
    PROCESS_GROUP_PID_TOKEN,
    CacheMiss,
    DeclaredOutput,
    FakeOutcome,
    FakeRunner,
    LocalRunner,
    ResourceGuard,
    ResourceGuardResult,
    RuntimeResourceMonitor,
    RuntimeResourceMonitorResult,
    ReplayRunner,
    Runner,
    RunnerExecution,
    RunnerInput,
    RunnerProtocolError,
    SSHRunner,
    StageOrchestrator,
    ToolRunRequest,
)
from hlsgraph.runner.staging import StagingError, read_verified_file
from hlsgraph.run_projection import sanitize_run_metadata
from hlsgraph.sdk import Project
from hlsgraph.model import ToolRun, stable_hash, utc_now


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _request(root: Path, command: str, *, max_bytes: int = 1024) -> ToolRunRequest:
    data = (root / "kernel.cpp").read_bytes()
    return ToolRunRequest(
        snapshot_id="snapshot.v2", stage="csynth",
        argv=[sys.executable, "-c", command],
        input_artifact_ids=["artifact.kernel"],
        inputs=[RunnerInput(
            "artifact.kernel", "kernel.cpp", "src/kernel.cpp", _digest(data), len(data),
        )],
        declared_outputs=[DeclaredOutput("reports/result.txt", max_bytes=max_bytes)],
        timeout_s=10,
    )


def test_runner_v2_protocol_and_request_paths_are_fail_closed() -> None:
    assert PROTOCOL_VERSION == "hlsgraph.runner.v2"
    with pytest.raises(RunnerProtocolError, match="protocol mismatch"):
        ToolRunRequest(
            snapshot_id="s", stage="x", argv=["tool"],
            protocol_version="hlsgraph.runner.v1",
        )
    with pytest.raises(RunnerProtocolError, match="project-relative|unsafe|non-portable"):
        RunnerInput("a", "../private.cpp", "private.cpp", "0" * 64, 0)
    with pytest.raises(RunnerProtocolError, match="exactly match"):
        ToolRunRequest(
            snapshot_id="s", stage="x", argv=["tool"], input_artifact_ids=["a"],
        )
    with pytest.raises(RunnerProtocolError, match="cannot contain one another"):
        ToolRunRequest(
            snapshot_id="s", stage="x", argv=["tool"],
            input_artifact_ids=["a"],
            inputs=[RunnerInput("a", "a", "tree", "0" * 64, 0)],
            declared_outputs=[DeclaredOutput("tree/result")],
        )


def test_local_runner_uses_one_run_scope_and_returns_only_declared_bytes(tmp_path) -> None:
    (tmp_path / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    (tmp_path / "ambient.txt").write_text("must not be staged", encoding="utf-8")
    existing = tmp_path / "reports" / "result.txt"
    existing.parent.mkdir()
    existing.write_text("pre-existing project bytes", encoding="utf-8")
    command = (
        "from pathlib import Path; "
        "assert Path('src/kernel.cpp').read_text(encoding='utf-8').startswith('void'); "
        "assert not Path('ambient.txt').exists(); "
        "p=Path('reports/result.txt'); p.parent.mkdir(parents=True,exist_ok=True); "
        "p.write_text('fresh staged bytes',encoding='utf-8'); "
        "Path('undeclared.txt').write_text('not returned',encoding='utf-8')"
    )
    execution = LocalRunner(tmp_path, allow_execution=True).execute(
        _request(tmp_path, command),
    )
    assert execution.run.status == RunStatus.SUCCEEDED
    assert execution.run.metadata["staging_isolated"] is True
    assert len(execution.staged_outputs) == 1
    assert execution.staged_outputs[0].local_path.read_text(encoding="utf-8") == (
        "fresh staged bytes"
    )
    assert existing.read_text(encoding="utf-8") == "pre-existing project bytes"
    assert not (tmp_path / "undeclared.txt").exists()
    staging = execution.staging_directory
    execution.cleanup()
    execution.cleanup()
    assert staging is not None and not staging.exists()


def test_local_runner_rejects_output_over_size_limit(tmp_path) -> None:
    (tmp_path / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    command = (
        "from pathlib import Path; p=Path('reports/result.txt'); "
        "p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(b'12345')"
    )
    execution = LocalRunner(tmp_path, allow_execution=True).execute(
        _request(tmp_path, command, max_bytes=4),
    )
    assert execution.run.status == RunStatus.FAILED
    assert execution.run.failure_class == FailureClass.INPUT
    assert execution.staged_outputs == []
    execution.cleanup()


def test_staged_reader_does_not_follow_symbolic_links(tmp_path) -> None:
    root = tmp_path / "stage"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("private", encoding="utf-8")
    link = root / "result.txt"
    try:
        os.symlink(outside, link)
    except OSError:
        pytest.skip("symbolic-link creation is not available to this test user")
    with pytest.raises(StagingError, match="link or reparse"):
        read_verified_file(root, "result.txt")


def test_orchestrator_rechecks_staged_hash_at_the_spi_boundary(tmp_path) -> None:
    (tmp_path / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    command = (
        "from pathlib import Path; p=Path('reports/result.txt'); "
        "p.parent.mkdir(parents=True,exist_ok=True); p.write_text('good')"
    )

    class TamperingRunner(LocalRunner):
        def execute(self, request):
            execution = super().execute(request)
            execution.staged_outputs[0].local_path.write_text("evil", encoding="utf-8")
            return execution

    with pytest.raises(RunnerProtocolError, match="hash mismatch|changed"):
        StageOrchestrator(TamperingRunner(tmp_path, allow_execution=True)).execute([
            _request(tmp_path, command),
        ])


def test_ssh_runner_returns_only_explicit_verified_outputs(tmp_path, monkeypatch) -> None:
    (tmp_path / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_transport(argv, **kwargs):
        captured["argv"] = argv
        payload = json.loads(kwargs["input"])
        captured["payload"] = payload
        data = b"remote result"
        response = {
            "kind": "tool", "exit_code": 0,
            "stdout": base64.b64encode(b"tool stdout").decode(),
            "stderr": base64.b64encode(b"").decode(),
            "outputs": [{
                "path": "reports/result.txt", "size": len(data),
                "sha256": _digest(data), "data": base64.b64encode(data).decode(),
            }],
        }
        return subprocess.CompletedProcess(
            argv, 0, stdout="HLSGRAPH_RUNNER_V2:" + json.dumps(response) + "\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_transport)
    request = _request(tmp_path, "unused")
    request.environment_hash = _digest(b"environment")
    request.metadata["remote_attestation_argv"] = ["environment-probe"]
    runner = SSHRunner(
        "fixture-host", "/tmp/hlsgraph-staging", project_root=tmp_path,
        allow_execution=True, ssh_options=(),
    )
    execution = runner.execute(request)
    assert execution.run.status == RunStatus.SUCCEEDED
    assert execution.run.metadata["remote_inputs_verified"] is True
    assert execution.run.metadata["remote_environment_verified"] is True
    assert execution.staged_outputs[0].local_path.read_bytes() == b"remote result"
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["input_manifest_sha256"] == _digest(json.dumps(
        [{key: payload["inputs"][0][key]
          for key in ("artifact_id", "path", "sha256", "size")}],
        sort_keys=True, separators=(",", ":"),
    ).encode())
    assert str(tmp_path) not in json.dumps(payload)
    assert "hlsgraph-run-" in captured["argv"][-1]
    execution.cleanup()


def test_ssh_resource_guard_is_structured_and_tool_output_cannot_spoof_it(
    tmp_path, monkeypatch,
) -> None:
    (tmp_path / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    responses = [
        {"kind": "resource_guard", "exit_code": 31,
         "message": "resource guard rejected execution"},
        {"kind": "tool", "exit_code": 7,
         "stdout": base64.b64encode(b"infra_resource_guard").decode(),
         "stderr": "", "outputs": []},
    ]
    payloads = []

    def fake_transport(argv, **kwargs):
        payloads.append(json.loads(kwargs["input"]))
        response = responses.pop(0)
        return subprocess.CompletedProcess(
            argv, 0, stdout="HLSGRAPH_RUNNER_V2:" + json.dumps(response) + "\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_transport)
    request = _request(tmp_path, "unused")
    request.environment_hash = _digest(b"environment")
    request.metadata["remote_attestation_argv"] = ["environment-probe"]
    request.nonzero_failure = FailureClass.DESIGN_COMPILE
    runner = SSHRunner(
        "fixture-host", "/tmp/hlsgraph-staging", project_root=tmp_path,
        allow_execution=True, ssh_options=(),
        resource_guard=ResourceGuard(("resource-probe",)),
    )
    rejected = runner.execute(request)
    assert rejected.run.failure_class == FailureClass.INFRA_RESOURCE_GUARD
    assert rejected.run.metadata["resource_guard_checked"] is True
    assert rejected.run.metadata["fresh_execution"] is False
    StageOrchestrator(runner)._validate_response(
        request, rejected, runner.fingerprint, runner.name, runner.capabilities(),
    )
    rejected.cleanup()

    tool_failed = runner.execute(request)
    assert tool_failed.run.failure_class == FailureClass.DESIGN_COMPILE
    assert tool_failed.run.metadata["resource_guard_passed"] is True
    assert tool_failed.run.metadata["stdout_sha256"] == _digest(b"infra_resource_guard")
    assert payloads[0]["resource_guard"] == {
        "argv": ["resource-probe"], "timeout_s": 30.0,
    }
    tool_failed.cleanup()


def test_fake_runner_never_claims_staging_or_tool_truth() -> None:
    request = ToolRunRequest(snapshot_id="s", stage="csim", argv=["fixture"])
    execution = FakeRunner().execute(request)
    assert execution.staged_outputs == []
    assert execution.run.metadata["staged_output_manifest"] == []
    assert execution.run.metadata["staging_isolated"] is False
    assert execution.run.metadata["tool_truth"] is False


def test_project_run_can_pin_an_earlier_materialized_snapshot(tmp_path) -> None:
    source = tmp_path / "kernel.cpp"
    first_bytes = "void dut() { int version = 1; }\n"
    source.write_text(first_bytes, encoding="utf-8")
    manifest = minimal_manifest("test.pinned_run", "pinned run", "dut", "kernel.cpp")
    manifest.stage_commands = {"csim": ["fixture-tool", "csim"]}
    manifest.toolchains = [ToolchainContext(
        id="test.fixture", vendor="test", name="fixture", version="1",
    )]
    project = Project(GraphBundle.create(tmp_path, manifest))
    first = project.index(degraded=True)
    assert first.success

    source.write_text("void dut() { int version = 2; }\n", encoding="utf-8")
    second = project.index(degraded=True)
    assert second.success and second.snapshot_id != first.snapshot_id
    source.write_text(first_bytes, encoding="utf-8")

    with pytest.raises(BundleError, match="selected snapshot is stale"):
        project.run(FakeRunner(), ["csim"])
    pinned = project.run(
        FakeRunner(), ["csim"], snapshot_id=first.snapshot_id,
    )
    assert len(pinned.runs) == 1
    assert pinned.runs[0].snapshot_id == first.snapshot_id
    assert project.bundle.latest_snapshot().id == second.snapshot_id


def test_local_resource_guard_failure_preempts_stage_classification(tmp_path) -> None:
    request = ToolRunRequest(
        snapshot_id="s", stage="csim",
        argv=[sys.executable, "-c", "raise SystemExit(9)"],
        nonzero_failure=FailureClass.CORRECTNESS,
    )
    execution = LocalRunner(
        tmp_path, allow_execution=True,
        resource_guard=ResourceGuard((
            sys.executable, "-c",
            "print('untrusted output is ignored'); raise SystemExit(23)",
        )),
    ).execute(request)
    assert execution.run.status == RunStatus.FAILED
    assert execution.run.failure_class == FailureClass.INFRA_RESOURCE_GUARD
    assert execution.run.exit_code == 23
    assert execution.run.metadata["resource_guard_configured"] is True
    assert execution.run.metadata["resource_guard_checked"] is True
    assert execution.run.metadata["resource_guard_passed"] is False
    assert execution.run.metadata["fresh_execution"] is False
    assert execution.run.metadata["tool_truth"] is False
    assert execution.run.metadata["stdout_bytes"] == 0
    StageOrchestrator(LocalRunner(
        tmp_path, allow_execution=True,
        resource_guard=ResourceGuard((sys.executable, "-c", "raise SystemExit(2)")),
    )).execute([request]).cleanup()
    execution.cleanup()


def test_tool_stdout_and_request_metadata_cannot_forge_resource_guard(tmp_path) -> None:
    request = ToolRunRequest(
        snapshot_id="s", stage="csim",
        argv=[sys.executable, "-c", "print('infra_resource_guard'); raise SystemExit(3)"],
        nonzero_failure=FailureClass.CORRECTNESS,
    )
    execution = LocalRunner(tmp_path, allow_execution=True).execute(request)
    assert execution.run.failure_class == FailureClass.CORRECTNESS
    assert execution.resource_guard is None
    execution.cleanup()
    with pytest.raises(RunnerProtocolError, match="runner-measured keys"):
        ToolRunRequest(
            snapshot_id="s", stage="csim", argv=["tool"],
            metadata={
                "resource_guard_configured": True,
                "resource_guard_checked": True,
                "resource_guard_passed": False,
            },
        )
    with pytest.raises(RunnerProtocolError, match="runner-measured keys"):
        ToolRunRequest(
            snapshot_id="s", stage="csim", argv=["tool"],
            metadata={"failure_type": "infra_resource_guard"},
        )
    with pytest.raises(RunnerProtocolError, match="runner-owned"):
        ToolRunRequest(
            snapshot_id="s", stage="csim", argv=["tool"],
            nonzero_failure=FailureClass.INFRA_RESOURCE_GUARD,
        )
    with pytest.raises(RunnerProtocolError, match="runner-owned"):
        ToolRunRequest(
            snapshot_id="s", stage="csim", argv=["tool"],
            nonzero_failure=FailureClass.RESOURCE,
        )
    assert "failure_type" not in sanitize_run_metadata({
        "failure_type": "infra_resource_guard",
    })
    with pytest.raises(ValueError, match="structured resource-guard provenance"):
        ToolRun(
            snapshot_id="s", stage="csim", backend="runner.untrusted",
            request_hash="a" * 64, status=RunStatus.FAILED,
            failure_class=FailureClass.INFRA_RESOURCE_GUARD,
            metadata={"failure_type": "infra_resource_guard"},
        )


def _guard_run(request: ToolRunRequest, runner: Runner) -> ToolRun:
    event_time = utc_now()
    return ToolRun(
        snapshot_id=request.snapshot_id, stage=request.stage, backend=runner.name,
        request_hash=request.cache_key(runner.fingerprint),
        toolchain_id=request.toolchain_id, status=RunStatus.FAILED,
        command=list(request.argv), working_directory=request.working_directory,
        environment_hash=request.environment_hash,
        input_artifact_ids=list(request.input_artifact_ids),
        failure_class=FailureClass.INFRA_RESOURCE_GUARD, exit_code=17,
        started_at=event_time, finished_at=event_time, elapsed_s=0,
        metadata={
            **request.metadata, "runner_fingerprint": runner.fingerprint,
            "resource_guard_configured": True, "resource_guard_checked": True,
            "resource_guard_passed": False, "fresh_execution": False,
            "fresh_tool_truth": False, "authority": "infrastructure",
            "tool_truth": False,
        },
    )


def test_plugin_resource_guard_requires_explicit_runner_capability() -> None:
    class GuardPlugin(Runner):
        name = "runner.plugin_fixture"
        can_report_resource_guard = True

        @property
        def fingerprint(self):
            return stable_hash({"runner": self.name})

        def execute(self, request):
            result = ResourceGuardResult(True, False, 17)
            return RunnerExecution(
                _guard_run(request, self), resource_guard=result,
            )

    request = ToolRunRequest(snapshot_id="s", stage="csim", argv=["tool"])
    accepted = StageOrchestrator(GuardPlugin()).execute([request])
    assert accepted.runs[0].failure_class == FailureClass.INFRA_RESOURCE_GUARD

    class UntrustedGuardPlugin(GuardPlugin):
        can_report_resource_guard = False

    with pytest.raises(RunnerProtocolError, match="not trusted"):
        StageOrchestrator(UntrustedGuardPlugin()).execute([request])


def test_fake_and_replay_cannot_upgrade_resource_guard_failure() -> None:
    request = ToolRunRequest(snapshot_id="s", stage="csim", argv=["tool"])
    fake_runner = FakeRunner({"csim": [FakeOutcome(
        status=RunStatus.FAILED,
        failure_class=FailureClass.INFRA_RESOURCE_GUARD,
        exit_code=5,
    )]})
    fake = fake_runner.execute(request)
    assert fake.run.failure_class == FailureClass.INFRASTRUCTURE
    assert fake.resource_guard is None
    assert fake.run.metadata["resource_guard_configured"] is False
    key = request.cache_key(fake_runner.fingerprint)
    with pytest.raises(CacheMiss):
        ReplayRunner(
            {key: fake}, source_runner_fingerprint=fake_runner.fingerprint,
        ).execute(request)


def test_sdk_persists_and_exports_resource_guard_failure(tmp_path) -> None:
    source = tmp_path / "kernel.cpp"
    source.write_text("void dut() {}\n", encoding="utf-8")
    manifest = minimal_manifest("test.guard_export", "guard export", "dut", "kernel.cpp")
    manifest.stage_commands = {
        "csim": [sys.executable, "-c", "raise SystemExit(9)"],
    }
    manifest.toolchains = [ToolchainContext(
        id="test.fixture", vendor="test", name="fixture", version="1",
    )]
    project = Project(GraphBundle.create(tmp_path, manifest))
    indexed = project.index(degraded=True)
    result = project.run(LocalRunner(
        tmp_path, allow_execution=True,
        resource_guard=ResourceGuard((sys.executable, "-c", "raise SystemExit(11)")),
    ), ["csim"])
    assert result.runs[0].failure_class == FailureClass.INFRA_RESOURCE_GUARD
    persisted = project.bundle.store.runs(indexed.snapshot_id)
    assert persisted[-1].failure_class == FailureClass.INFRA_RESOURCE_GUARD

    output = tmp_path / "dataset"
    project.export_dataset(output, snapshot_id=indexed.snapshot_id)
    rows = [json.loads(line) for line in
            (output / "runs.jsonl").read_text(encoding="utf-8").splitlines()]
    exported = next(item for item in rows if item["id"] == result.runs[0].id)
    assert exported["failure_class"] == "infra_resource_guard"
    assert exported["metadata"]["resource_guard_checked"] is True
    assert exported["metadata"]["resource_guard_passed"] is False


def test_sdk_persists_and_exports_runtime_monitor_failure(tmp_path) -> None:
    source = tmp_path / "kernel.cpp"
    source.write_text("void dut() {}\n", encoding="utf-8")
    manifest = minimal_manifest(
        "test.runtime_guard_export", "runtime guard export", "dut", "kernel.cpp",
    )
    manifest.stage_commands = {
        "csim": [sys.executable, "-c", "import time; time.sleep(5)"],
    }
    manifest.toolchains = [ToolchainContext(
        id="test.fixture", vendor="test", name="fixture", version="1",
    )]
    project = Project(GraphBundle.create(tmp_path, manifest))
    indexed = project.index(degraded=True)
    monitor = RuntimeResourceMonitor((
        sys.executable, "-c", "raise SystemExit(37)",
        PROCESS_GROUP_PID_TOKEN,
    ), interval_s=0.05, timeout_s=1, resource_exit_codes=(37,))
    result = project.run(LocalRunner(
        tmp_path, allow_execution=True, runtime_resource_monitor=monitor,
    ), ["csim"])
    assert result.runs[0].failure_class == FailureClass.RESOURCE
    persisted = project.bundle.store.runs(indexed.snapshot_id)
    assert persisted[-1].metadata["runtime_guard_triggered"] is True

    output = tmp_path / "dataset"
    project.export_dataset(output, snapshot_id=indexed.snapshot_id)
    rows = [json.loads(line) for line in
            (output / "runs.jsonl").read_text(encoding="utf-8").splitlines()]
    exported = next(item for item in rows if item["id"] == result.runs[0].id)
    assert exported["failure_class"] == "resource"
    assert exported["metadata"]["runtime_guard_checked"] is True
    assert exported["metadata"]["runtime_guard_triggered"] is True


def _runtime_monitor(exit_code: int = 0, *, timeout_s: float = 1.0,
                     resource_exit_codes: tuple[int, ...] = ()) -> RuntimeResourceMonitor:
    return RuntimeResourceMonitor(
        (sys.executable, "-c", f"raise SystemExit({exit_code})",
         PROCESS_GROUP_PID_TOKEN),
        interval_s=0.05, timeout_s=timeout_s,
        resource_exit_codes=resource_exit_codes,
    )


def test_runtime_monitor_configuration_is_runner_owned_and_fingerprinted(tmp_path) -> None:
    monitor = _runtime_monitor()
    first = LocalRunner(
        tmp_path, allow_execution=True, inherit_environment=False,
        runtime_resource_monitor=monitor,
    )
    second = LocalRunner(
        tmp_path, allow_execution=True, inherit_environment=False,
        runtime_resource_monitor=RuntimeResourceMonitor(**monitor.identity_payload()),
    )
    mapped = LocalRunner(
        tmp_path, allow_execution=True, inherit_environment=False,
        runtime_resource_monitor=_runtime_monitor(resource_exit_codes=(19,)),
    )
    assert first.fingerprint == second.fingerprint
    assert first.fingerprint != mapped.fingerprint
    assert first.capabilities()["can_report_runtime_resource_guard"] is True
    assert LocalRunner(tmp_path).capabilities()[
        "can_report_runtime_resource_guard"
    ] is False
    with pytest.raises(RunnerProtocolError, match="exactly one"):
        RuntimeResourceMonitor(("monitor",))
    for metadata in (
        {"runtime_guard_triggered": True},
        {"runtime_resource_monitor": {"argv": ["untrusted"]}},
        {"runtime_guard_failure_class": "resource"},
    ):
        with pytest.raises(RunnerProtocolError, match="runner-measured keys"):
            ToolRunRequest(
                snapshot_id="s", stage="csim", argv=["tool"], metadata=metadata,
            )


def test_local_runtime_monitor_passes_pid_and_discards_probe_output(tmp_path) -> None:
    monitor = RuntimeResourceMonitor((
        sys.executable, "-c",
        "import sys; assert int(sys.argv[1]) > 0; "
        "print('guard-secret'); print('guard-error', file=sys.stderr)",
        PROCESS_GROUP_PID_TOKEN,
    ), interval_s=0.05, timeout_s=1)
    request = ToolRunRequest(
        snapshot_id="s", stage="csim",
        argv=[sys.executable, "-c",
              "import sys,time; sys.stdout.buffer.write(b'tool-only'); "
              "sys.stdout.flush(); time.sleep(0.15)"],
        timeout_s=2,
    )
    runner = LocalRunner(
        tmp_path, allow_execution=True, runtime_resource_monitor=monitor,
    )
    result = StageOrchestrator(runner).execute([request])
    run = result.runs[0]
    assert run.status == RunStatus.SUCCEEDED
    assert run.metadata["runtime_guard_configured"] is True
    assert run.metadata["runtime_guard_checked"] is True
    assert run.metadata["runtime_guard_passed"] is True
    assert run.metadata["runtime_guard_triggered"] is False
    assert run.metadata["stdout_sha256"] == _digest(b"tool-only")
    assert run.metadata["stderr_bytes"] == 0
    result.cleanup()


def test_local_runtime_monitor_trigger_has_no_tool_truth_or_outputs(tmp_path) -> None:
    request = ToolRunRequest(
        snapshot_id="s", stage="csim",
        argv=[sys.executable, "-c",
              "from pathlib import Path; import time; "
              "p=Path('reports/result.txt'); p.parent.mkdir(parents=True); "
              "p.write_text('must-not-return'); time.sleep(5)"],
        declared_outputs=[DeclaredOutput("reports/result.txt")], timeout_s=10,
        nonzero_failure=FailureClass.CORRECTNESS,
    )
    runner = LocalRunner(
        tmp_path, allow_execution=True,
        # This case validates the monitor's non-zero exit mapping, not its
        # timeout path.  A fresh Python probe can take over one second to
        # start on a loaded Windows CI host; the timeout path has a separate
        # explicit test below.
        runtime_resource_monitor=_runtime_monitor(19, timeout_s=5.0),
    )
    result = StageOrchestrator(runner).execute([request])
    execution = result.executions[0]
    assert execution.run.failure_class == FailureClass.INFRA_RESOURCE_GUARD
    assert execution.run.exit_code == 19
    assert execution.run.metadata["runtime_guard_triggered"] is True
    assert execution.run.metadata["fresh_execution"] is True
    assert execution.run.metadata["fresh_tool_truth"] is False
    assert execution.run.metadata["tool_truth"] is False
    assert execution.staged_outputs == []
    assert execution.run.output_artifact_ids == []
    result.cleanup()


def test_local_runtime_monitor_has_trusted_resource_exit_mapping_and_timeout(tmp_path) -> None:
    request = ToolRunRequest(
        snapshot_id="s", stage="csynth",
        argv=[sys.executable, "-c", "import time; time.sleep(5)"], timeout_s=10,
    )
    resource_runner = LocalRunner(
        tmp_path, allow_execution=True,
        runtime_resource_monitor=_runtime_monitor(
            42, resource_exit_codes=(42,),
        ),
    )
    resource = StageOrchestrator(resource_runner).execute([request])
    assert resource.runs[0].failure_class == FailureClass.RESOURCE
    assert resource.executions[0].runtime_resource_monitor == (
        RuntimeResourceMonitorResult(
            True, False, True, 42, FailureClass.RESOURCE,
        )
    )
    resource.cleanup()

    timeout_monitor = RuntimeResourceMonitor((
        sys.executable, "-c", "import time; time.sleep(1)",
        PROCESS_GROUP_PID_TOKEN,
    ), interval_s=0.05, timeout_s=0.05)
    timeout_runner = LocalRunner(
        tmp_path, allow_execution=True,
        runtime_resource_monitor=timeout_monitor,
    )
    timed_out = StageOrchestrator(timeout_runner).execute([request])
    assert timed_out.runs[0].failure_class == FailureClass.INFRA_RESOURCE_GUARD
    assert timed_out.executions[0].runtime_resource_monitor == (
        RuntimeResourceMonitorResult(
            True, False, True, None, FailureClass.INFRA_RESOURCE_GUARD,
        )
    )
    timed_out.cleanup()


def test_tool_exit_cannot_select_runtime_monitors_resource_mapping(tmp_path) -> None:
    request = ToolRunRequest(
        snapshot_id="s", stage="csim",
        argv=[sys.executable, "-c",
              "print('runtime_guard_triggered'); raise SystemExit(42)"],
        timeout_s=2, nonzero_failure=FailureClass.CORRECTNESS,
    )
    runner = LocalRunner(
        tmp_path, allow_execution=True,
        runtime_resource_monitor=_runtime_monitor(
            0, resource_exit_codes=(42,),
        ),
    )
    result = StageOrchestrator(runner).execute([request])
    run = result.runs[0]
    assert run.failure_class == FailureClass.CORRECTNESS
    assert run.metadata["runtime_guard_passed"] is True
    assert run.metadata["runtime_guard_triggered"] is False
    result.cleanup()


def test_request_environment_cannot_control_runtime_monitor(tmp_path) -> None:
    variable = "HLSGRAPH_TEST_UNTRUSTED_RUNTIME_GUARD_EXIT"
    os.environ.pop(variable, None)
    monitor = RuntimeResourceMonitor((
        sys.executable, "-c",
        f"import os; raise SystemExit(int(os.environ.get({variable!r}, '0')))",
        PROCESS_GROUP_PID_TOKEN,
    ), interval_s=0.05, timeout_s=1, resource_exit_codes=(42,))
    request = ToolRunRequest(
        snapshot_id="s", stage="csim",
        argv=[sys.executable, "-c", "import time; time.sleep(0.15)"],
        environment={variable: "42"}, timeout_s=2,
    )
    runner = LocalRunner(
        tmp_path, allow_execution=True, runtime_resource_monitor=monitor,
    )
    result = StageOrchestrator(runner).execute([request])
    assert result.runs[0].status == RunStatus.SUCCEEDED
    assert result.runs[0].metadata["runtime_guard_passed"] is True
    result.cleanup()


@pytest.mark.skipif(os.name == "nt", reason="process-group assertion is POSIX-specific")
def test_runtime_monitor_terminates_descendant_process_group(tmp_path) -> None:
    marker = tmp_path / "descendant-survived.txt"
    child = (
        "import time; from pathlib import Path; time.sleep(0.8); "
        f"Path({str(marker)!r}).write_text('alive')"
    )
    tool = (
        "import subprocess,sys,time; "
        f"subprocess.Popen([sys.executable,'-c',{child!r}]); time.sleep(10)"
    )
    # Pass the first check so the tool has time to create its descendant, then
    # trigger on the next bounded interval.
    probe = (
        "from pathlib import Path; p=Path('.monitor-count'); "
        "n=int(p.read_text()) if p.exists() else 0; p.write_text(str(n+1)); "
        "raise SystemExit(0 if n == 0 else 23)"
    )
    runner = LocalRunner(
        tmp_path, allow_execution=True,
        runtime_resource_monitor=RuntimeResourceMonitor((
            sys.executable, "-c", probe, PROCESS_GROUP_PID_TOKEN,
        ), interval_s=0.25, timeout_s=1),
    )
    result = StageOrchestrator(runner).execute([ToolRunRequest(
        snapshot_id="s", stage="csynth",
        argv=[sys.executable, "-c", tool], timeout_s=10,
    )])
    assert result.runs[0].failure_class == FailureClass.INFRA_RESOURCE_GUARD
    time.sleep(1.0)
    assert not marker.exists()
    result.cleanup()


def test_ssh_runtime_monitor_is_structured_and_exit_mapping_is_local(
    tmp_path, monkeypatch,
) -> None:
    (tmp_path / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    captured = {}

    def fake_transport(argv, **kwargs):
        captured["payload"] = json.loads(kwargs["input"])
        response = {
            "kind": "runtime_guard", "exit_code": 41,
            "runtime_guard_checked": True,
            "stdout": base64.b64encode(b"tool partial output").decode(),
            "stderr": "", "outputs": [{"path": "reports/result.txt"}],
        }
        return subprocess.CompletedProcess(
            argv, 0, stdout="HLSGRAPH_RUNNER_V2:" + json.dumps(response) + "\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_transport)
    request = _request(tmp_path, "unused")
    request.environment_hash = _digest(b"environment")
    request.metadata["remote_attestation_argv"] = ["environment-probe"]
    monitor = RuntimeResourceMonitor((
        "trusted-monitor", PROCESS_GROUP_PID_TOKEN,
    ), interval_s=0.25, timeout_s=2, resource_exit_codes=(41,))
    runner = SSHRunner(
        "fixture-host", "/tmp/hlsgraph-staging", project_root=tmp_path,
        allow_execution=True, ssh_options=(), runtime_resource_monitor=monitor,
    )
    result = StageOrchestrator(runner).execute([request])
    execution = result.executions[0]
    assert execution.run.failure_class == FailureClass.RESOURCE
    assert execution.run.metadata["runtime_guard_triggered"] is True
    assert execution.run.metadata["fresh_tool_truth"] is False
    assert execution.staged_outputs == []
    assert captured["payload"]["runtime_resource_monitor"] == monitor.identity_payload()
    assert PROCESS_GROUP_PID_TOKEN in captured["payload"][
        "runtime_resource_monitor"
    ]["argv"]
    result.cleanup()


def test_ssh_runtime_monitor_timeout_is_guard_failure(tmp_path, monkeypatch) -> None:
    (tmp_path / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")

    def fake_transport(argv, **kwargs):
        response = {
            "kind": "runtime_guard", "exit_code": None,
            "runtime_guard_checked": True,
            "message": "runtime resource monitor timed out",
            "stdout": "", "stderr": "", "outputs": [],
        }
        return subprocess.CompletedProcess(
            argv, 0, stdout="HLSGRAPH_RUNNER_V2:" + json.dumps(response) + "\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_transport)
    request = _request(tmp_path, "unused")
    request.environment_hash = _digest(b"environment")
    request.metadata["remote_attestation_argv"] = ["environment-probe"]
    runner = SSHRunner(
        "fixture-host", "/tmp/hlsgraph-staging", project_root=tmp_path,
        allow_execution=True, ssh_options=(),
        runtime_resource_monitor=RuntimeResourceMonitor((
            "trusted-monitor", PROCESS_GROUP_PID_TOKEN,
        )),
    )
    result = StageOrchestrator(runner).execute([request])
    execution = result.executions[0]
    assert execution.run.failure_class == FailureClass.INFRA_RESOURCE_GUARD
    assert execution.runtime_resource_monitor == RuntimeResourceMonitorResult(
        True, False, True, None, FailureClass.INFRA_RESOURCE_GUARD,
    )
    result.cleanup()


def test_fake_runtime_metadata_is_downgraded_and_replay_clears_monitor_state() -> None:
    request = ToolRunRequest(snapshot_id="s", stage="csim", argv=["tool"])
    fake = FakeRunner({"csim": [FakeOutcome(metadata={
        "runtime_guard_configured": True,
        "runtime_guard_checked": True,
        "runtime_guard_passed": False,
        "runtime_guard_triggered": True,
        "runtime_resource_monitor": {"argv": ["private"]},
    })]}).execute(request)
    assert fake.runtime_resource_monitor is None
    assert fake.run.metadata["runtime_guard_configured"] is False
    assert "runtime_resource_monitor" not in fake.run.metadata

    fake_resource = FakeRunner({"csim": [FakeOutcome(
        status=RunStatus.FAILED, failure_class=FailureClass.RESOURCE,
        exit_code=42, metadata={"runtime_guard_triggered": True},
    )]}).execute(request)
    assert fake_resource.run.failure_class == FailureClass.INFRASTRUCTURE
    assert fake_resource.run.metadata["runtime_guard_triggered"] is False

    source = FakeRunner()
    original = source.execute(request)
    key = request.cache_key(source.fingerprint)
    replay = ReplayRunner(
        {key: original}, source_runner_fingerprint=source.fingerprint,
    ).execute(request)
    assert replay.runtime_resource_monitor is None
    assert replay.run.metadata["runtime_guard_configured"] is False
