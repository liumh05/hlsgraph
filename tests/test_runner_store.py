from __future__ import annotations

import hashlib
import base64
import json
import shlex
import sqlite3
import subprocess
import sys

import pytest

from hlsgraph.bundle import GraphBundle
from hlsgraph.graph import CanonicalGraph
from hlsgraph.manifest import minimal_manifest
from hlsgraph.model import (
    AuthorityClass,
    Entity,
    FailureClass,
    GateKind,
    GateResult,
    GateStatus,
    Observation,
    RunStatus,
)
from hlsgraph.runner import (
    CacheMiss,
    DeclaredOutput,
    FakeOutcome,
    FakeRunner,
    LocalRunner,
    ReplayRunner,
    RunnerInput,
    RunnerProtocolError,
    SSHRunner,
    StageOrchestrator,
    ToolRunRequest,
)
from hlsgraph.runner.core import _local_bootstrap_environment
from hlsgraph.store import StoreError


def _request(stage: str = "csynth", **overrides) -> ToolRunRequest:
    values = {
        "snapshot_id": "snapshot_test",
        "stage": stage,
        "argv": ["fixture-tool", "--stage", stage],
        "working_directory": ".",
        "environment": {"MODE": "test"},
        "environment_hash": "e" * 64,
        "toolchain_id": "amd.vitis.2024_2",
        "input_artifact_ids": [],
        "timeout_s": 10.0,
        "nonzero_failure": FailureClass.DESIGN_COMPILE,
        "metadata": {"fixture": True},
    }
    values.update(overrides)
    return ToolRunRequest(**values)


def test_runner_cache_key_is_deterministic_and_complete():
    request = _request()
    fingerprint = "runner-fingerprint"
    assert request.cache_key(fingerprint) == _request(
        environment={"MODE": "test"}, input_artifact_ids=[]
    ).cache_key(fingerprint)

    variants = [
        _request(stage="rtl_cosim"),
        _request(argv=["fixture-tool", "--different"]),
        _request(working_directory="work"),
        _request(environment={"MODE": "release"}),
        _request(environment_hash="f" * 64),
        _request(toolchain_id="amd.vitis.2025_1"),
        _request(input_artifact_ids=["artifact_c"], inputs=[RunnerInput(
            "artifact_c", "input.cpp", "input.cpp", "a" * 64, 1,
        )]),
        _request(timeout_s=20),
        _request(nonzero_failure=FailureClass.CORRECTNESS),
        _request(metadata={"fixture": False}),
    ]
    keys = {request.cache_key(fingerprint), *(item.cache_key(fingerprint) for item in variants)}
    assert len(keys) == len(variants) + 1


def test_local_and_ssh_execution_are_disabled_by_default(tmp_path):
    request = _request()
    local = LocalRunner(tmp_path).execute(request)
    ssh = SSHRunner("fixture-host", "/tmp/fixture").execute(request)
    for run, backend in ((local, "runner.local"), (ssh, "runner.ssh")):
        assert run.backend == backend
        assert run.status == RunStatus.SKIPPED
        assert run.failure_class == FailureClass.UNSUPPORTED
        assert run.exit_code is None
        assert run.metadata["execution_enabled"] is False
        assert "enable it explicitly" in (run.message or "")

    assert LocalRunner(tmp_path).capabilities()["provides_local_output_bytes"] is True
    assert SSHRunner("fixture-host", "/tmp/fixture").capabilities()[
        "provides_local_output_bytes"
    ] is True


def test_local_runner_classifies_spawn_failure_as_infrastructure(tmp_path):
    run = LocalRunner(tmp_path, allow_execution=True).execute(
        _request(argv=["hlsgraph-command-that-does-not-exist-9f13"])
    )
    assert run.status == RunStatus.FAILED
    assert run.failure_class == FailureClass.INFRASTRUCTURE
    assert run.exit_code is None
    assert run.metadata["execution_enabled"] is True


def test_local_runner_hashes_environment_and_reserves_measured_metadata(tmp_path,
                                                                        monkeypatch):
    runner = LocalRunner(tmp_path, allow_execution=True)
    monkeypatch.setenv("HLSGRAPH_RUNNER_CACHE_PROBE", "one")
    first = runner.fingerprint
    monkeypatch.setenv("HLSGRAPH_RUNNER_CACHE_PROBE", "two")
    assert runner.fingerprint != first
    fixed = LocalRunner(tmp_path, inherit_environment=False)
    fixed_first = fixed.fingerprint
    monkeypatch.setenv("HLSGRAPH_RUNNER_CACHE_PROBE", "three")
    assert fixed.fingerprint == fixed_first

    with pytest.raises(RunnerProtocolError, match="runner-measured keys"):
        _request(metadata={
            "stdout_bytes": 999, "stdout_sha256": "0" * 64,
            "bootstrap_environment_hash": "1" * 64,
            "output_embedded": True, "authority": "synthetic", "tool_truth": False,
        })

    run = runner.execute(_request(
        argv=[sys.executable, "-c", "print('ok')"], metadata={"fixture": "preserved"},
    ))
    assert run.status == RunStatus.SUCCEEDED
    assert run.metadata["stdout_bytes"] == 3
    assert run.metadata["stdout_sha256"] == hashlib.sha256(b"ok\n").hexdigest()
    assert run.metadata["output_embedded"] is False
    assert run.metadata["authority"] == "tool_observation"
    assert run.metadata["tool_truth"] is True
    assert run.metadata["fresh_execution"] is True
    assert run.metadata["fresh_tool_truth"] is True
    assert run.metadata["fixture"] == "preserved"


def test_isolated_local_environment_uses_only_windows_system_root():
    source = {
        "PATH": "private-path",
        "SYSTEMROOT": "system-root-sentinel",
        "PRIVATE_TOKEN": "must-not-inherit",
    }
    assert _local_bootstrap_environment(source, platform="nt") == {
        "SystemRoot": "system-root-sentinel",
    }
    assert _local_bootstrap_environment(source, platform="posix") == {}


def test_ssh_runner_uses_explicit_manifest_and_transfer(tmp_path, monkeypatch):
    captured = {}

    def fake_subprocess_run(argv, **kwargs):
        captured["argv"] = argv
        captured["payload"] = json.loads(kwargs["input"])
        response = {
            "kind": "tool", "exit_code": 0,
            "stdout": base64.b64encode(b"").decode(),
            "stderr": base64.b64encode(b"").decode(), "outputs": [],
        }
        return subprocess.CompletedProcess(
            argv, 0, stdout="HLSGRAPH_RUNNER_V2:" + json.dumps(response) + "\n", stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)
    artifact_id = "artifact_remote"
    source = tmp_path / "inputs" / "file.cpp"
    source.parent.mkdir()
    source.write_bytes(b"fixture-data")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    probe_output = b"fixture-remote-environment"
    environment_hash = hashlib.sha256(probe_output).hexdigest()
    request = _request(
        argv=["fixture-tool", "argument with spaces", "x; touch injected"],
        environment_hash=environment_hash,
        input_artifact_ids=[artifact_id],
        inputs=[RunnerInput(
            artifact_id, "inputs/file.cpp", "inputs/file.cpp", digest, 12,
        )],
        metadata={
            "remote_attestation_argv": [
                "printf", "%s", probe_output.decode("ascii"),
            ],
        },
    )
    run = SSHRunner("fixture-host", "/tmp/project root", project_root=tmp_path,
                    allow_execution=True, ssh_options=()).execute(request)
    argv = captured["argv"]
    assert argv[:2] == ["ssh", "fixture-host"]
    assert "python3 -c" in argv[-1]
    payload = captured["payload"]
    assert payload["protocol"] == "hlsgraph.runner.v2"
    assert payload["inputs"][0]["path"] == "inputs/file.cpp"
    assert base64.b64decode(payload["inputs"][0]["data"]) == b"fixture-data"
    assert payload["argv"] == ["fixture-tool", "argument with spaces", "x; touch injected"]
    assert run.status == RunStatus.SUCCEEDED
    assert run.metadata["remote_inputs_verified"] is True
    assert run.metadata["remote_environment_verified"] is True
    assert run.metadata["tool_truth"] is True


def test_ssh_runner_requires_project_binding_and_pinned_environment(tmp_path):
    runner = SSHRunner("fixture-host", "/tmp/project", allow_execution=True)
    with pytest.raises(RunnerProtocolError, match="local project root"):
        runner.execute(_request(metadata={"remote_attestation_argv": ["probe"]}))
    runner.bind_project_root(tmp_path)
    with pytest.raises(RunnerProtocolError, match="environment_hash"):
        runner.execute(_request(environment_hash=None))
    with pytest.raises(RunnerProtocolError, match="remote_attestation_argv"):
        runner.execute(_request(metadata={}))


def test_ssh_attestation_mismatch_is_infrastructure_not_tool_truth(tmp_path, monkeypatch):
    def attestation_mismatch(argv, **_kwargs):
        response = {"kind": "attestation", "message": "attestation mismatch", "exit_code": 1}
        return subprocess.CompletedProcess(
            argv, 0, stdout="HLSGRAPH_RUNNER_V2:" + json.dumps(response) + "\n", stderr="",
        )

    monkeypatch.setattr(subprocess, "run", attestation_mismatch)
    request = _request(
        metadata={"remote_attestation_argv": ["fixture-env-probe"]},
    )
    run = SSHRunner("fixture-host", "/tmp/project", project_root=tmp_path,
                    allow_execution=True, ssh_options=()).execute(request)
    assert run.status == RunStatus.FAILED
    assert run.failure_class == FailureClass.INFRASTRUCTURE
    assert run.metadata["remote_inputs_verified"] is True
    assert run.metadata["remote_environment_verified"] is False
    assert run.metadata["tool_truth"] is False


@pytest.mark.parametrize("failure", [
    FailureClass.LICENSE,
    FailureClass.SSH,
    FailureClass.TIMEOUT,
    FailureClass.DESIGN_COMPILE,
    FailureClass.BENCHMARK,
    FailureClass.CORRECTNESS,
])
def test_fake_runner_keeps_failure_classes_distinct(failure):
    runner = FakeRunner({"csynth": [FakeOutcome(
        status=RunStatus.FAILED, failure_class=failure, exit_code=1,
        message=f"synthetic {failure.value}",
    )]})
    run = runner.execute(_request())
    assert run.failure_class == failure
    assert run.status == RunStatus.FAILED
    assert run.metadata["tool_truth"] is False
    assert run.metadata["authority"] == "synthetic"


def test_fake_runner_cannot_override_its_synthetic_authority():
    run = FakeRunner({"csim": [FakeOutcome(metadata={
        "authority": "tool_observation", "tool_truth": True,
    })]}).execute(_request(stage="csim"))
    assert run.metadata["authority"] == "synthetic"
    assert run.metadata["tool_truth"] is False


def test_replay_is_cache_only_and_cannot_be_presented_as_fresh_tool_truth():
    request = _request()
    source = FakeRunner()
    original = source.execute(request)
    cache_key = request.cache_key(source.fingerprint)
    replay = ReplayRunner(
        {cache_key: original}, source_runner_fingerprint=source.fingerprint
    )

    cached = replay.execute(request)
    assert cached.status == RunStatus.CACHED
    assert cached.metadata["replayed_from_run_id"] == original.id
    assert cached.metadata["fresh_tool_truth"] is False
    assert cached.output_artifact_ids == original.output_artifact_ids

    with pytest.raises(CacheMiss):
        replay.execute(_request(argv=["fixture-tool", "--cache-miss"]))


def test_replay_rejects_failed_or_identity_inconsistent_source_runs():
    request = _request()
    source = FakeRunner()
    failed = source.execute(request)
    failed.run.status = RunStatus.FAILED
    failed.run.failure_class = FailureClass.DESIGN_COMPILE
    key = request.cache_key(source.fingerprint)
    with pytest.raises(CacheMiss):
        ReplayRunner({key: failed}, source_runner_fingerprint=source.fingerprint).execute(request)

    original = source.execute(request)
    original.run.stage = "rtl_cosim"
    with pytest.raises(CacheMiss):
        ReplayRunner({key: original}, source_runner_fingerprint=source.fingerprint).execute(request)


def test_orchestrator_requires_three_independent_passing_gates():
    script = {
        "csim": [FakeOutcome(gates=[
            GateResult(GateKind.CORRECTNESS, GateStatus.PASS, ["verification.csim"])
        ], metadata={"campaign_id": "campaign.default"})],
        "rtl_cosim": [FakeOutcome(gates=[
            GateResult(GateKind.CORRECTNESS, GateStatus.PASS, ["verification.cosim"])
        ], metadata={"campaign_id": "campaign.default"})],
        "vivado_synth": [FakeOutcome(gates=[
            GateResult(GateKind.RESOURCE_FITS, GateStatus.PASS, ["derivation.resources"])
        ])],
        "post_route": [FakeOutcome(gates=[
            GateResult(GateKind.POST_ROUTE_TIMING, GateStatus.PASS, ["derivation.timing"])
        ])],
    }
    requests = [_request(stage=stage) for stage in
                ("csim", "rtl_cosim", "vivado_synth", "post_route")]
    result = StageOrchestrator(FakeRunner(script)).execute(requests)
    assert result.gates_complete is True
    assert result.tool_truth is False
    assert result.verified is False
    assert result.stopped_after_stage is None
    assert result.gates == {
        GateKind.CORRECTNESS: GateStatus.PASS,
        GateKind.RESOURCE_FITS: GateStatus.PASS,
        GateKind.POST_ROUTE_TIMING: GateStatus.PASS,
    }
    assert result.correctness_checks == {
        "csim": GateStatus.PASS, "rtl_cosim": GateStatus.PASS,
    }

    only_correct = StageOrchestrator(FakeRunner({
        "rtl_cosim": [FakeOutcome(gates=[
            GateResult(GateKind.CORRECTNESS, GateStatus.PASS, ["verification.cosim"])
        ], metadata={"campaign_id": "campaign.default"})]
    })).execute([_request(stage="rtl_cosim")])
    assert only_correct.verified is False
    assert GateKind.RESOURCE_FITS not in only_correct.gates
    assert GateKind.POST_ROUTE_TIMING not in only_correct.gates

    timing_failure = StageOrchestrator(FakeRunner({
        "csim": [FakeOutcome(gates=[
            GateResult(GateKind.CORRECTNESS, GateStatus.PASS, ["verification.csim"])
        ], metadata={"campaign_id": "campaign.default"})],
        "rtl_cosim": [FakeOutcome(gates=[
            GateResult(GateKind.CORRECTNESS, GateStatus.PASS, ["verification.cosim"])
        ], metadata={"campaign_id": "campaign.default"})],
        "vivado_synth": [FakeOutcome(gates=[
            GateResult(GateKind.RESOURCE_FITS, GateStatus.PASS, ["derivation.resources"])
        ])],
        "post_route": [FakeOutcome(gates=[
            GateResult(GateKind.POST_ROUTE_TIMING, GateStatus.FAIL)
        ])],
    })).execute(requests)
    assert timing_failure.verified is False
    assert timing_failure.stopped_after_stage == "post_route"
    assert timing_failure.gates[GateKind.CORRECTNESS] == GateStatus.PASS
    assert timing_failure.gates[GateKind.RESOURCE_FITS] == GateStatus.PASS
    assert timing_failure.gates[GateKind.POST_ROUTE_TIMING] == GateStatus.FAIL


def test_orchestrator_failure_dominates_duplicate_gates_and_ignores_failed_run_gates():
    duplicate = StageOrchestrator(FakeRunner({"csim": [FakeOutcome(gates=[
        GateResult(GateKind.CORRECTNESS, GateStatus.FAIL),
        GateResult(GateKind.CORRECTNESS, GateStatus.PASS, ["verification.csim"]),
    ])]})).execute([_request(stage="csim")])
    assert duplicate.gates[GateKind.CORRECTNESS] == GateStatus.FAIL
    assert duplicate.correctness_checks["csim"] == GateStatus.FAIL
    assert duplicate.verified is False

    failed = StageOrchestrator(FakeRunner({"csim": [FakeOutcome(
        status=RunStatus.FAILED, failure_class=FailureClass.CORRECTNESS, exit_code=1,
        gates=[GateResult(GateKind.CORRECTNESS, GateStatus.PASS, ["verification.csim"])],
    )]})).execute([_request(stage="csim")])
    assert failed.gates == {}
    assert failed.correctness_checks == {}
    assert failed.stopped_after_stage == "csim"
    assert failed.verified is False


def test_orchestrator_requires_one_explicit_correctness_campaign_and_evidence_validator():
    def script(csim_campaign, cosim_campaign):
        return {
            "csim": [FakeOutcome(
                gates=[GateResult(GateKind.CORRECTNESS, GateStatus.PASS,
                                  ["verification.csim"])],
                metadata={"campaign_id": csim_campaign} if csim_campaign else {},
            )],
            "rtl_cosim": [FakeOutcome(
                gates=[GateResult(GateKind.CORRECTNESS, GateStatus.PASS,
                                  ["verification.cosim"])],
                metadata={"campaign_id": cosim_campaign} if cosim_campaign else {},
            )],
            "vivado_synth": [FakeOutcome(gates=[
                GateResult(GateKind.RESOURCE_FITS, GateStatus.PASS,
                           ["derivation.resources"]),
            ])],
            "post_route": [FakeOutcome(gates=[
                GateResult(GateKind.POST_ROUTE_TIMING, GateStatus.PASS,
                           ["derivation.timing"]),
            ])],
        }

    requests = [_request(stage=stage) for stage in
                ("csim", "rtl_cosim", "vivado_synth", "post_route")]
    assert StageOrchestrator(FakeRunner(script("a", "b"))).execute(
        requests
    ).gates_complete is False
    assert StageOrchestrator(FakeRunner(script(None, None))).execute(
        requests
    ).gates_complete is False

    class ToolTruthRunner(FakeRunner):
        def execute(self, request):
            run = super().execute(request)
            run.metadata["authority"] = "tool_observation"
            run.metadata["tool_truth"] = True
            return run

    without_validator = StageOrchestrator(
        ToolTruthRunner(script("campaign.default", "campaign.default"))
    ).execute(requests)
    assert without_validator.gates_complete is True
    assert without_validator.tool_truth is False
    assert without_validator.verified is False

    validated = StageOrchestrator(
        ToolTruthRunner(script("campaign.default", "campaign.default")),
        evidence_validator=lambda _run, evidence_id: evidence_id.startswith(
            ("verification.", "derivation.")
        ),
    ).execute(requests)
    assert validated.gates_complete is True
    assert validated.tool_truth is True
    assert validated.verified is True


def test_sqlite_ledger_is_append_only_and_read_connections_are_query_only(tmp_path):
    (tmp_path / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    bundle = GraphBundle.create(
        tmp_path, minimal_manifest("test.ledger", "ledger", "dut", "kernel.cpp")
    )
    snapshot = bundle.snapshot()
    kernel = Entity("hls.kernel", "dut", snapshot.id, qualified_name="dut")
    graph = CanonicalGraph(snapshot.id)
    graph.add_entity(kernel)
    bundle.store.save_graph(graph)

    observation = Observation(
        snapshot_id=snapshot.id, subject_id=kernel.id, predicate="qor.latency_cycles",
        value=12, unit="cycle", stage="schedule",
        authority=AuthorityClass.TOOL_OBSERVATION,
    )
    bundle.store.add_observations([observation])
    bundle.store.add_observations([observation])
    assert bundle.store.observations(snapshot.id) == [observation]

    changed = Observation(
        snapshot_id=snapshot.id, subject_id=kernel.id, predicate="qor.latency_cycles",
        value=13, unit="cycle", stage="schedule",
        authority=AuthorityClass.TOOL_OBSERVATION,
    )
    changed.id = observation.id
    with pytest.raises(StoreError, match="stable id"):
        bundle.store.add_observations([changed])

    with bundle.store.read() as connection:
        assert connection.execute("PRAGMA query_only").fetchone()[0] == 1
        with pytest.raises(sqlite3.OperationalError, match="readonly|read-only"):
            connection.execute("DELETE FROM observations")

    loaded = bundle.store.load_graph(snapshot.id)
    assert loaded.graph_hash == graph.graph_hash
