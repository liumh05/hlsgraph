from __future__ import annotations

import json
import hashlib
import sqlite3
import sys
from pathlib import Path

import pytest

from hlsgraph.api import RestApplication
from hlsgraph.bundle import BundleError, GraphBundle
from hlsgraph.export import export_dataset
from hlsgraph.extract import ExtractionError, ExtractionResult, LibClangExtractor
from hlsgraph.graph import CanonicalGraph
from hlsgraph.manifest import ManifestError, minimal_manifest
from hlsgraph.mcp import ReadOnlyMcpService
from hlsgraph.model import (
    AccessPolicy,
    AuthorityClass,
    Entity,
    Observation,
    ToolOutputSpec,
    ToolRun,
    ToolchainContext,
    FailureClass,
    RunStatus,
    json_ready,
)
from hlsgraph.query import CoreService
from hlsgraph.runner import FakeOutcome, FakeRunner, LocalRunner, SSHRunner
from hlsgraph.sdk import Project
from hlsgraph.store import StoreError


def _manifest_file(root: Path, *, project_id: str = "test.lifecycle") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "kernel.cpp").write_text(
        "void dut(int *x) { for (int i=0; i<2; ++i) x[i]++; }\n",
        encoding="utf-8",
    )
    manifest = minimal_manifest(project_id, "lifecycle", "dut", "kernel.cpp")
    path = root / "hlsgraph.json"
    path.write_text(json.dumps(json_ready(manifest), indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    return path


def _runnable_project(root: Path, *, project_id: str) -> Project:
    root.mkdir(parents=True, exist_ok=True)
    (root / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    manifest = minimal_manifest(project_id, "runnable", "dut", "kernel.cpp")
    manifest.stage_commands = {"csim": ["historical-csim", "--run"]}
    manifest.toolchains = [ToolchainContext(
        id="amd.vitis.2024_2", vendor="amd", name="vitis_hls", version="2024.2",
        environment_hash="a" * 64,
        metadata={"remote_attestation_argv": ["vitis_hls", "-version"]},
    )]
    project = Project(GraphBundle.create(root, manifest))
    assert project.index(degraded=True).success
    return project


def test_project_run_rejects_stale_source_before_runner_is_called(tmp_path):
    project = _runnable_project(tmp_path / "stale-run", project_id="test.stale_run")
    (project.bundle.project_root / "kernel.cpp").write_text(
        "void dut() { int changed = 1; }\n", encoding="utf-8"
    )
    runner = FakeRunner()
    with pytest.raises(BundleError, match="stale"):
        project.run(runner, ["csim"])
    assert runner.calls == []


def test_project_run_is_serialized_by_bundle_execution_lock(tmp_path):
    project = _runnable_project(tmp_path / "locked-run", project_id="test.locked_run")
    runner = FakeRunner()
    with project.bundle.execution_lock():
        with pytest.raises(BundleError, match="another stage execution"):
            project.run(runner, ["csim"])
    assert runner.calls == []
    assert not (project.bundle.root / "execution.lock").exists()


def test_project_run_reverifies_all_attached_artifact_bytes(tmp_path):
    project = _runnable_project(tmp_path / "byte-run", project_id="test.byte_run")
    output = project.bundle.project_root / "generated.rpt"
    output.write_text("original\n", encoding="utf-8")
    artifact = project.bundle.add_managed_artifact(
        output, kind="amd.vitis.report", role="tool_output",
    )
    (project.bundle.project_root / artifact.uri).write_text("tampered\n", encoding="utf-8")
    runner = FakeRunner()
    with pytest.raises(BundleError, match="recorded bytes"):
        project.run(runner, ["csim"])
    assert runner.calls == []


def test_project_run_uses_snapshot_manifest_command_and_toolchain(tmp_path, monkeypatch):
    project = _runnable_project(tmp_path / "historical-run",
                                project_id="test.historical_run")
    project.bundle.manifest.stage_commands["csim"] = ["mutated-current-command"]
    project.bundle.manifest.toolchains[0] = ToolchainContext(
        id="amd.vitis.2025_1", vendor="amd", name="vitis_hls", version="2025.1",
        environment_hash="b" * 64,
        metadata={"remote_attestation_argv": ["vitis-run", "--version"]},
    )
    # Simulate a long-lived process whose in-memory current manifest advanced
    # after the active snapshot passed the normal stale check. Project.run must
    # still read execution identity from the immutable snapshot-manifest row.
    monkeypatch.setattr(project.bundle, "is_stale", lambda _snapshot: False)
    runner = FakeRunner()
    project.run(runner, ["csim"])
    assert len(runner.calls) == 1
    request = runner.calls[0]
    assert request.argv == ["historical-csim", "--run"]
    assert request.toolchain_id == "amd.vitis.2024_2"
    assert request.environment_hash == "a" * 64
    assert request.metadata["remote_attestation_argv"] == ["vitis_hls", "-version"]
    assert request.metadata["input_artifacts"]


def test_project_run_resolves_each_stage_to_its_snapshot_toolchain(tmp_path):
    root = tmp_path / "multi-toolchain-run"
    root.mkdir()
    (root / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    manifest = minimal_manifest("test.multi_toolchain_run", "multi", "dut", "kernel.cpp")
    manifest.stage_commands = {
        "csynth": ["run-hls"],
        "post_route": ["run-vivado"],
    }
    manifest.toolchains = [
        ToolchainContext("amd.vitis.2024_2", "amd", "vitis_hls", "2024.2",
                         environment_hash="a" * 64),
        ToolchainContext("amd.vivado.2024_2", "amd", "vivado", "2024.2",
                         environment_hash="b" * 64),
    ]
    manifest.stage_toolchains = {
        "csynth": "amd.vitis.2024_2",
        "post_route": "amd.vivado.2024_2",
    }
    project = Project(GraphBundle.create(root, manifest))
    assert project.index(degraded=True).success
    runner = FakeRunner()
    project.run(runner, ["csynth", "post_route"])
    assert [(item.stage, item.toolchain_id, item.environment_hash)
            for item in runner.calls] == [
        ("csynth", "amd.vitis.2024_2", "a" * 64),
        ("post_route", "amd.vivado.2024_2", "b" * 64),
    ]


def test_project_run_ingests_declared_report_with_run_provenance(tmp_path):
    root = tmp_path / "declared-output"
    root.mkdir()
    (root / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    report_text = json.dumps({
        "schema_version": "hlsgraph.vitis.csim.v1",
        "status": "pass", "exit_code": 0, "mismatches": 0,
        "assertions_failed": 0, "workload_id": "tb.default",
    })
    command = (
        "from pathlib import Path; "
        "p=Path('reports/csim.json'); p.parent.mkdir(parents=True,exist_ok=True); "
        f"p.write_text({report_text!r},encoding='utf-8')"
    )
    manifest = minimal_manifest("test.declared_output", "outputs", "dut", "kernel.cpp")
    manifest.stage_commands = {"csim": [sys.executable, "-c", command]}
    manifest.toolchains = [ToolchainContext(
        id="test.python", vendor="python", name="python", version="test",
    )]
    manifest.stage_outputs = {"csim": [ToolOutputSpec(
        path="reports/csim.json", kind="amd.vitis.csim_result",
        metadata={"workload_id": "tb.default", "campaign_id": "campaign.default"},
    )]}
    project = Project(GraphBundle.create(root, manifest))
    assert project.index(degraded=True).success

    result = project.run(LocalRunner(root, allow_execution=True), ["csim"], timeout_s=10)
    assert result.verified is False
    assert result.correctness_checks["csim"].value == "pass"
    run = result.runs[0]
    outputs = [item for item in project.bundle.store.artifacts(run.snapshot_id)
               if item.producer_run_id == run.id]
    assert len(outputs) == 1 and outputs[0].id in run.output_artifact_ids
    observations = project.bundle.store.observations(run.snapshot_id)
    assert observations and all(item.run_id == run.id for item in observations)
    assert all(item.artifact_id == outputs[0].id for item in observations)
    verifications = project.bundle.store.verifications(run.snapshot_id)
    assert len(verifications) == 1
    assert verifications[0]["run_id"] == run.id
    assert verifications[0]["workload_id"] == "tb.default"


def test_project_run_ignores_preexisting_project_output_and_uses_staging(tmp_path):
    root = tmp_path / "preexisting-output"
    root.mkdir()
    (root / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    (root / "report.json").write_text("{}\n", encoding="utf-8")
    manifest = minimal_manifest("test.preexisting_output", "outputs", "dut", "kernel.cpp")
    manifest.stage_commands = {"csim": ["unused"]}
    manifest.toolchains = [ToolchainContext(
        id="test.fake", vendor="test", name="fake", version="test",
    )]
    manifest.stage_outputs = {"csim": [ToolOutputSpec(
        path="report.json", kind="amd.vitis.csim_result",
    )]}
    project = Project(GraphBundle.create(root, manifest))
    assert project.index(degraded=True).success
    runner = FakeRunner()
    result = project.run(runner, ["csim"])
    assert result.runs[0].status == RunStatus.FAILED
    assert result.runs[0].failure_class == FailureClass.INPUT
    assert (root / "report.json").read_text(encoding="utf-8") == "{}\n"
    assert len(runner.calls) == 1


def test_project_run_empty_and_duplicate_stage_selection_fail_closed(tmp_path):
    project = Project.create_from_manifest(_manifest_file(tmp_path))
    assert project.index(degraded=True).success
    runner = FakeRunner()

    result = project.run(runner, [])
    assert result.runs == []
    assert runner.calls == []
    with pytest.raises(ValueError, match="must be unique"):
        project.run(runner, ["csynth", "csynth"])
    with pytest.raises(ValueError, match="not a string"):
        project.run(runner, "csynth")
    assert runner.calls == []


def test_project_run_failure_cannot_inherit_historical_verified_status(tmp_path,
                                                                       monkeypatch):
    project = _runnable_project(
        tmp_path / "historical-gates", project_id="test.historical_gates",
    )

    class HistoricallyVerifiedService:
        @staticmethod
        def verification_gates():
            return {
                "correctness": {
                    "status": "pass", "tool_truth": True,
                    "eligible_campaigns": ["campaign.golden"],
                    "checks": {
                        "csim": {"status": "pass"},
                        "rtl_cosim": {"status": "pass"},
                    },
                },
                "resource_fits": {"status": "pass", "tool_truth": True},
                "post_route_timing": {"status": "pass", "tool_truth": True},
                "verified": True,
            }

    monkeypatch.setattr(
        project, "service", lambda _snapshot_id=None: HistoricallyVerifiedService(),
    )
    runner = FakeRunner({"csim": [FakeOutcome(
        status=RunStatus.FAILED, failure_class=FailureClass.CORRECTNESS, exit_code=1,
    )]})
    result = project.run(runner, ["csim"])

    assert result.gates["correctness"].value == "pass"
    assert result.stopped_after_stage == "csim"
    assert result.gates_complete is False
    assert result.tool_truth is False
    assert result.verified is False

    assert len(runner.calls) == 1


def test_project_run_partial_success_cannot_inherit_historical_verified_status(
    tmp_path, monkeypatch,
):
    root = tmp_path / "historical-success-gates"
    root.mkdir()
    (root / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    manifest = minimal_manifest(
        "test.historical_success_gates", "historical success", "dut", "kernel.cpp",
    )
    manifest.stage_commands = {"csim": [sys.executable, "-c", "pass"]}
    manifest.toolchains = [ToolchainContext(
        id="test.python", vendor="python", name="python", version="test",
    )]
    project = Project(GraphBundle.create(root, manifest))
    assert project.index(degraded=True).success

    class HistoricallyVerifiedService:
        @staticmethod
        def verification_gates():
            current_run = project.bundle.store.runs(project.bundle.latest_snapshot().id)[-1]
            return {
                "correctness": {
                    "status": "pass", "tool_truth": True,
                    "eligible_campaigns": ["campaign.golden"],
                    "eligible_run_ids": {
                        "campaign.golden": {
                            "csim": [current_run.id],
                            "rtl_cosim": ["run.historical_cosim"],
                        },
                    },
                    "checks": {
                        "csim": {"status": "pass"},
                        "rtl_cosim": {"status": "pass"},
                    },
                },
                "resource_fits": {"status": "pass", "tool_truth": True},
                "post_route_timing": {"status": "pass", "tool_truth": True},
                "eligible_physical_runs": ["run.historical_post_route"],
                "verified": True,
            }

    monkeypatch.setattr(
        project, "service", lambda _snapshot_id=None: HistoricallyVerifiedService(),
    )
    result = project.run(
        LocalRunner(root, allow_execution=True), ["csim"], timeout_s=10,
    )

    assert len(result.runs) == 1
    assert result.runs[0].status == RunStatus.SUCCEEDED
    assert result.gates_complete is False
    assert result.verified is False


def test_declared_output_dependencies_are_ordered_and_enter_consumer_identity(tmp_path):
    root = tmp_path / "output-dependency"
    root.mkdir()
    (root / "kernel.cpp").write_text("void dut() {}\n", encoding="utf-8")
    manifest = minimal_manifest("test.output_dependency", "outputs", "dut", "kernel.cpp")
    manifest.stage_commands = {"csynth": ["produce"], "post_route": ["consume"]}
    manifest.toolchains = [ToolchainContext(
        id="test.fake", vendor="test", name="fake", version="test",
    )]
    manifest.stage_outputs = {"csynth": [ToolOutputSpec(
        path="run/intermediate.bin", kind="tool.intermediate",
        consumed_by=["post_route"],
    )]}
    project = Project(GraphBundle.create(root, manifest))
    assert project.index(degraded=True).success

    with pytest.raises(ValueError, match="requires producer"):
        project.run(FakeRunner(), ["post_route"])
    with pytest.raises(ValueError, match="dependency order"):
        project.run(FakeRunner(), ["post_route", "csynth"])
    skipped = project.run(SSHRunner("example.invalid", "/tmp/project"), ["csynth"])
    assert skipped.runs[0].status == RunStatus.SKIPPED

    missing = project.run(FakeRunner(), ["csynth"])
    assert missing.runs[0].status == RunStatus.FAILED
    assert missing.runs[0].failure_class == FailureClass.INPUT

    produce = (
        "from pathlib import Path; p=Path('run/intermediate.bin'); "
        "p.parent.mkdir(parents=True,exist_ok=True); "
        "p.write_bytes(b'deterministic intermediate')"
    )
    consume = (
        "from pathlib import Path; "
        "assert Path('run/intermediate.bin').read_bytes()==b'deterministic intermediate'"
    )
    manifest.stage_commands = {
        "csynth": [sys.executable, "-c", produce],
        "post_route": [sys.executable, "-c", consume],
    }
    project.bundle.store.save_project(manifest)
    # The active snapshot stores its own manifest; re-index after changing the
    # command identity so execution cannot silently use mutable live config.
    project.bundle.manifest = manifest
    assert project.index(degraded=True).success
    result = project.run(LocalRunner(root, allow_execution=True), ["csynth", "post_route"])
    produced_id = result.runs[0].output_artifact_ids[0]
    assert produced_id in result.runs[1].input_artifact_ids


def test_tool_output_paths_are_unique_across_all_stages() -> None:
    manifest = minimal_manifest("test.output_unique", "outputs", "dut", "kernel.cpp")
    manifest.stage_commands = {"csim": ["a"], "rtl_cosim": ["b"]}
    manifest.toolchains = [ToolchainContext(
        id="test.fake", vendor="test", name="fake", version="test",
    )]
    manifest.stage_outputs = {
        "csim": [ToolOutputSpec("run/result.rpt", "tool.report")],
        "rtl_cosim": [ToolOutputSpec("run/result.rpt", "tool.report")],
    }
    with pytest.raises(ValueError, match="globally unique"):
        manifest.identity_payload()


def _rewrite_manifest(path: Path, mutator) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    mutator(value)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return value


def _successful_standard_extract(_self, context):
    """Dependency-free test double for the standard extraction lifecycle.

    Libclang parsing itself is covered by the golden extraction test.  This test
    isolates active-snapshot behavior so it also runs when the optional clang
    wheel is absent.
    """
    graph = CanonicalGraph(
        context.snapshot.id,
        metadata={"source_backend": "source.libclang", "fidelity": "ast"},
    )
    graph.add_entity(Entity(
        "hls.kernel", context.manifest.build.top, context.snapshot.id,
        qualified_name=context.manifest.build.top, stage="ast",
        attrs={"fidelity": "standard_test_double"},
    ))
    return ExtractionResult(
        graph=graph, capabilities=["source.ast", "directive.source_scope"],
        coverage={"fidelity": "libclang", "entities": 1, "relations": 0},
    )


def test_external_manifest_change_is_stale_then_reindex_atomically_switches_active_snapshot(tmp_path):
    manifest_path = _manifest_file(tmp_path)
    project = Project.create_from_manifest(manifest_path)
    first = project.index(degraded=True)
    assert first.success
    assert project.bundle.latest_snapshot().id == first.snapshot_id
    assert project.status().to_dict()["stale"] is False

    _rewrite_manifest(
        manifest_path,
        lambda value: value["build"]["defines"].update({"MODE": "2"}),
    )
    reopened = Project.open(tmp_path)
    before = reopened.status().to_dict()
    assert before["snapshot_id"] == first.snapshot_id
    assert before["stale"] is True
    assert reopened.bundle.manifest.build.defines == {}
    assert reopened.bundle.source_manifest().build.defines == {"MODE": "2"}

    second = reopened.index(degraded=True)
    assert second.success and second.snapshot_id != first.snapshot_id
    after = reopened.status().to_dict()
    assert after["snapshot_id"] == second.snapshot_id
    assert after["stale"] is False
    assert after["graph_available"] is True
    assert reopened.bundle.manifest.build.defines == {"MODE": "2"}
    assert json.loads((reopened.bundle.root / "manifest.json").read_text(
        encoding="utf-8"))["build"]["defines"] == {"MODE": "2"}

    # Switching active does not erase the previous immutable graph.
    assert reopened.bundle.store.has_graph(first.snapshot_id)
    assert reopened.bundle.store.has_graph(second.snapshot_id)
    assert reopened.service(first.snapshot_id).graph().snapshot_id == first.snapshot_id
    assert reopened.service().graph().snapshot_id == second.snapshot_id


def test_standard_and_degraded_snapshots_coexist_and_can_be_reactivated(tmp_path, monkeypatch):
    manifest_path = _manifest_file(tmp_path, project_id="test.parser_profiles")
    project = Project.create_from_manifest(manifest_path)
    degraded = project.index(degraded=True)
    assert degraded.success
    degraded_graph = project.service().graph()
    assert degraded_graph.metadata["degraded"] is True

    monkeypatch.setattr(LibClangExtractor, "extract", _successful_standard_extract)
    standard = project.index(degraded=False)
    assert standard.success
    assert standard.snapshot_id != degraded.snapshot_id
    assert project.bundle.latest_snapshot().id == standard.snapshot_id
    assert project.service().graph().metadata["degraded"] is False

    assert project.bundle.store.has_graph(degraded.snapshot_id)
    assert project.bundle.store.has_graph(standard.snapshot_id)
    assert project.bundle.store.snapshot(degraded.snapshot_id).extraction_hash
    assert project.bundle.store.snapshot(standard.snapshot_id).extraction_hash
    assert (project.bundle.store.snapshot(degraded.snapshot_id).extraction_hash
            != project.bundle.store.snapshot(standard.snapshot_id).extraction_hash)
    assert project.service(degraded.snapshot_id).graph().metadata["degraded"] is True
    assert project.service(standard.snapshot_id).graph().metadata["degraded"] is False

    # Re-indexing an existing deterministic profile reuses that snapshot and
    # intentionally moves the active pointer back to it.
    degraded_again = project.index(degraded=True)
    assert degraded_again.snapshot_id == degraded.snapshot_id
    assert project.bundle.latest_snapshot().id == degraded.snapshot_id
    assert project.service().graph().metadata["degraded"] is True
    assert project.service(standard.snapshot_id).graph().snapshot_id == standard.snapshot_id


def test_new_bundle_rejects_unsupported_manifest_schema_without_side_effects(tmp_path):
    manifest_path = _manifest_file(tmp_path)
    _rewrite_manifest(manifest_path, lambda value: value.update({"schema_version": "9.9.0"}))
    with pytest.raises(BundleError, match="schema"):
        Project.create_from_manifest(manifest_path)
    assert not (tmp_path / ".hlsgraph").exists()


def test_manifest_schema_version_is_required_at_external_and_internal_boundaries(tmp_path):
    manifest_path = _manifest_file(tmp_path)
    _rewrite_manifest(manifest_path, lambda value: value.pop("schema_version"))
    with pytest.raises(ManifestError, match="schema_version"):
        Project.create_from_manifest(manifest_path)
    assert not (tmp_path / ".hlsgraph").exists()

    manifest_path = _manifest_file(tmp_path)
    project = Project.create_from_manifest(manifest_path)
    internal = project.bundle.root / "manifest.json"
    value = json.loads(internal.read_text(encoding="utf-8"))
    value.pop("schema_version")
    internal.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(BundleError, match="schema|manifest"):
        Project.open(tmp_path)


def test_refresh_rejects_external_manifest_schema_mismatch_and_preserves_active(tmp_path):
    manifest_path = _manifest_file(tmp_path)
    project = Project.create_from_manifest(manifest_path)
    active = project.index(degraded=True)
    assert active.success

    # An edited external manifest is checked again at the write/index boundary.
    _rewrite_manifest(manifest_path, lambda value: value.update({"schema_version": "9.9.0"}))
    reopened = Project.open(tmp_path)
    with pytest.raises(BundleError, match="schema"):
        reopened.index(degraded=True)
    still_active = Project.open(tmp_path)
    assert still_active.bundle.latest_snapshot().id == active.snapshot_id
    assert still_active.bundle.store.has_graph(active.snapshot_id)


@pytest.mark.parametrize("metadata_file", ["manifest.json", "bundle.json"])
def test_open_rejects_internal_bundle_schema_mismatch(tmp_path, metadata_file):
    project = Project.create_from_manifest(_manifest_file(tmp_path))
    indexed = project.index(degraded=True)
    assert indexed.success
    path = project.bundle.root / metadata_file
    value = json.loads(path.read_text(encoding="utf-8"))
    value["schema_version"] = "8.8.0"
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(BundleError, match="schema"):
        Project.open(tmp_path)


def test_ledger_schema_mismatch_fails_closed_without_implicit_marker_update(tmp_path):
    project = Project.create_from_manifest(_manifest_file(tmp_path))
    active = project.index(degraded=True)
    database = project.bundle.store.path
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE schema_info SET value='9.9.0' WHERE key='schema_version'"
        )

    reopened = Project.open(tmp_path)
    with pytest.raises(StoreError, match="schema"):
        reopened.bundle.store.initialize()
    with pytest.raises(StoreError, match="schema"):
        reopened.bundle.store.latest_snapshot(reopened.bundle.manifest.project_id)
    with sqlite3.connect(database) as connection:
        marker = connection.execute(
            "SELECT value FROM schema_info WHERE key='schema_version'"
        ).fetchone()[0]
    assert marker == "9.9.0"
    assert active.snapshot_id


def test_canonical_graph_schema_mismatch_fails_closed(tmp_path):
    project = Project.create_from_manifest(_manifest_file(tmp_path))
    indexed = project.index(degraded=True)
    with sqlite3.connect(project.bundle.store.path) as connection:
        connection.execute(
            "UPDATE graph_views SET schema_version='9.9.0' WHERE snapshot_id=?",
            (indexed.snapshot_id,),
        )
    with pytest.raises(StoreError, match="graph schema|canonical graph"):
        project.bundle.store.load_graph(indexed.snapshot_id)


def test_failed_extraction_persists_run_and_diagnostics_but_no_partial_graph(
    tmp_path, monkeypatch,
):
    project = Project.create_from_manifest(_manifest_file(tmp_path))
    active = project.index(degraded=True)
    assert active.success

    def fail_extract(_self, _context):
        raise ExtractionError("intentional lifecycle extraction failure")

    monkeypatch.setattr(LibClangExtractor, "extract", fail_extract)
    failed = project.index(degraded=False)
    assert failed.success is False
    assert failed.snapshot_id != active.snapshot_id
    assert project.bundle.store.has_graph(failed.snapshot_id) is False
    assert project.bundle.store.observations(failed.snapshot_id) == []
    assert project.bundle.store.derivations(failed.snapshot_id) == []
    assert project.bundle.store.verifications(failed.snapshot_id) == []

    diagnostics = project.bundle.store.diagnostics(failed.snapshot_id)
    assert any(item.code == "extractor.failed"
               and "details withheld" in item.message
               for item in diagnostics)
    assert all("intentional lifecycle extraction failure" not in item.message
               for item in diagnostics)
    runs = project.bundle.store.runs(failed.snapshot_id)
    assert len(runs) == 1 and str(runs[0].status) == "failed"
    assert runs[0].metadata["partial_graph_persisted"] is False

    # Default readers remain on the last successful active graph.  Explicitly
    # selecting the failed candidate also fails closed instead of returning an
    # empty or partially populated canonical graph.
    assert project.bundle.latest_snapshot().id == active.snapshot_id
    assert project.service().graph().snapshot_id == active.snapshot_id
    with pytest.raises(StoreError, match="canonical graph"):
        project.bundle.store.load_graph(failed.snapshot_id)
    with pytest.raises(StoreError, match="canonical graph"):
        project.service(failed.snapshot_id).graph()


def test_managed_private_artifact_is_attached_but_not_exposed_by_services_or_exports(tmp_path):
    project = Project.create_from_manifest(_manifest_file(tmp_path))
    indexed = project.index(degraded=True)
    graph = project.service().graph()
    kernel = next(item for item in graph.entities.values() if item.kind == "hls.kernel")
    producer_run = ToolRun(
        snapshot_id=indexed.snapshot_id, stage="unknown", backend="runner.fixture",
        request_hash="f" * 64, status=RunStatus.FAILED,
        failure_class=FailureClass.INFRASTRUCTURE, exit_code=1,
    )
    secret = "PRIVATE_MANAGED_LOG_SENTINEL_297a"
    external = tmp_path.parent / f"{tmp_path.name}-private-run.log"
    external.write_text(f"{secret}\nstatus=failed\n", encoding="utf-8")
    artifact, managed_path, _created = project.bundle.prepare_managed_artifact(
        external, kind="tool.run_log", role="run_log", access=AccessPolicy.PRIVATE,
        producer_run_id=producer_run.id,
        license="Proprietary",
    )
    producer_run.output_artifact_ids = [artifact.id]
    project.bundle.store.commit_run_result(run=producer_run, artifacts=[artifact])
    assert str(artifact.retention) == "managed"
    assert str(artifact.access) == "private"
    assert artifact.producer_run_id == producer_run.id
    assert managed_path.is_file()
    assert managed_path.read_text(encoding="utf-8").startswith(secret)
    assert artifact in project.bundle.store.artifacts(indexed.snapshot_id)
    assert secret.encode() not in project.bundle.store.path.read_bytes()

    observation = Observation(
        snapshot_id=indexed.snapshot_id, subject_id=kernel.id,
        predicate="run.exit_code", value=1, stage="unknown",
        authority=AuthorityClass.TOOL_OBSERVATION, artifact_id=artifact.id,
        run_id=producer_run.id,
    )
    project.bundle.store.add_observations([observation])

    with pytest.raises(PermissionError, match="explicit authorization"):
        project.bundle.source_snippet(artifact.id, 1, 1)
    assert project.bundle.source_snippet(
        artifact.id, 1, 1, allow_private=True
    ) == secret

    core = CoreService(project.bundle, indexed.snapshot_id)
    rest = RestApplication(core)
    mcp = ReadOnlyMcpService(core)
    public_values = [
        rest.dispatch("GET", "/api/v1/artifacts").body,
        rest.dispatch("GET", f"/api/v1/evidence/{kernel.id}").body,
        mcp.evidence(kernel.id),
        mcp.overview(),
    ]
    assert all(secret not in json.dumps(value, ensure_ascii=False)
               for value in public_values)

    output = tmp_path / "ml-export"
    manifest = export_dataset(project.bundle, indexed.snapshot_id, output)
    serialized = "\n".join(path.read_text(encoding="utf-8")
                             for path in output.iterdir() if path.is_file())
    assert secret not in serialized
    assert manifest["private_source_embedded"] is False
    artifact_rows = [json.loads(line) for line in
                     (output / "artifacts.jsonl").read_text(encoding="utf-8").splitlines()]
    managed_row = next(item for item in artifact_rows if item["artifact_id"] == artifact.id)
    assert managed_row["access"] == "private"
    assert managed_row["source_text_embedded"] is False


def test_managed_artifact_requires_snapshot_before_copying_into_bundle(tmp_path):
    project = Project.create_from_manifest(_manifest_file(tmp_path))
    external = tmp_path.parent / f"{tmp_path.name}-orphan.log"
    external.write_text("must-not-be-orphaned\n", encoding="utf-8")
    with pytest.raises(BundleError, match="create a snapshot"):
        project.bundle.add_managed_artifact(
            external, kind="tool.run_log", role="run_log", access=AccessPolicy.PRIVATE,
        )
    assert list((project.bundle.root / "artifacts").rglob(external.name)) == []


def test_managed_artifact_identity_and_cas_use_the_same_source_read(tmp_path, monkeypatch):
    project = Project.create_from_manifest(_manifest_file(tmp_path))
    source = tmp_path / "report.bin"
    original = b"first complete report bytes"
    source.write_bytes(original)
    real_read_bytes = Path.read_bytes
    changed = False

    def mutate_after_read(path: Path) -> bytes:
        nonlocal changed
        data = real_read_bytes(path)
        if path.resolve() == source.resolve() and not changed:
            changed = True
            source.write_bytes(b"different bytes after the evidence read")
        return data

    monkeypatch.setattr(Path, "read_bytes", mutate_after_read)
    artifact, target, created = project.bundle.prepare_managed_artifact(
        source, kind="tool.report", role="report",
    )

    stored = real_read_bytes(target)
    assert changed is True and created is True
    assert stored == original
    assert artifact.size == len(stored)
    assert artifact.sha256 == hashlib.sha256(stored).hexdigest()
    assert list(target.parent.glob("*.tmp")) == []
