from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

from eval.agent_ab.common import (
    ARM_IDS, CODEGRAPH_OFFLINE_ENV, asset_digest, canonical_json, harness_digest, load_corpus_lock,
    load_environment_lock, load_manifest, prepared_hlsgraph_identity,
    sha256_bytes,
)
from eval.agent_ab.prepare import build_plan, execute_plan
from eval.agent_ab.setup_corpus import materialize
from eval.agent_ab.wheel_identity import main as wheel_identity_main
from eval.agent_ab.score import render_score_rows
from tests.agent_eval_runtime_support import (
    synthetic_cold_start_input_sha256, synthetic_cold_start_matrix,
    synthetic_runtime_identity,
)
from tools.audit_release import _audit_evaluation_release_gate, _payload_digest
from tools import audit_release as release_audit


def _cold_start_matrix() -> list[dict[str, object]]:
    return synthetic_cold_start_matrix()[0]


def _environment(
    wheel_sha256: str, package_sha256: str, *, work_root: Path,
) -> dict[str, object]:
    manifest = load_manifest()
    v02_revision = next(
        item["revision"] for item in manifest["arms"] if item["id"] == "hlsgraph-v02"
    )
    revisions = {"hlsgraph-v02": v02_revision, "hlsgraph-v03": "c" * 40}
    identities = []
    declared: dict[str, object] = {}
    for arm, key, version in (
        ("hlsgraph-v02", "hlsgraph_v02", "0.2.0"),
        ("hlsgraph-v03", "hlsgraph_v03", "0.3.0"),
    ):
        revision = revisions[arm]
        arm_wheel_sha256 = "2" * 64 if arm == "hlsgraph-v02" else wheel_sha256
        installed_sha256 = "a" * 64 if arm == "hlsgraph-v02" else "b" * 64
        source_sha256 = "d" * 64 if arm == "hlsgraph-v02" else package_sha256
        identity = {
            "schema_version": "hlsgraph.agent_eval.wheel_identity.v1",
            "verified": True, "version": version,
            "wheel_sha256": arm_wheel_sha256,
            "wheel_payload_sha256": installed_sha256,
            "installed_payload_sha256": installed_sha256,
            "source_revision": revision,
            "source_package_sha256": source_sha256,
            "wheel_package_sha256": source_sha256,
        }
        identities.append({
            "kind": "verify-hlsgraph-wheel-installation", "arm": arm,
            "identity": identity,
        })
        label = "v02" if arm == "hlsgraph-v02" else "v03"
        identities.extend([
            {"kind": f"verify-{label}-repo-clean", "stdout": ""},
            {"kind": f"record-{label}-revision", "stdout": revision},
            {"kind": f"verify-{label}-repo-clean-after", "stdout": ""},
            {"kind": f"record-{label}-revision-after", "stdout": revision},
        ])
        declared[key] = {
            "version": version, "wheel": f"hlsgraph-{version}.whl",
            "wheel_sha256": arm_wheel_sha256, "revision": revision,
            "source_revision": revision,
            "source_package_sha256": source_sha256,
            "wheel_package_sha256": source_sha256,
        }
    cold_start, cold_checks = synthetic_cold_start_matrix()
    identities.extend(cold_checks)
    workspaces = {}
    for arm in ARM_IDS:
        for corpus in load_corpus_lock()["corpora"]:
            identity = {
                "schema_version": "hlsgraph.agent_eval.workspace_identity.v1",
                "arm": arm, "corpus_id": corpus["id"], "snapshot_id": None,
                "index_sha256": hashlib.sha256(
                    f"{arm}/{corpus['id']}".encode()
                ).hexdigest(),
                "cold_start_input_sha256": synthetic_cold_start_input_sha256(
                    arm, corpus["id"],
                ),
            }
            identity["workspace_identity_sha256"] = sha256_bytes(canonical_json(identity))
            workspaces[f"{arm}/{corpus['id']}"] = identity
    runtime_identity = synthetic_runtime_identity(
        public_repository=Path(__file__).resolve().parents[1],
        work_root=work_root,
    )
    return {
        "schema_version": "hlsgraph.agent_eval.environment.v3",
        "suite_asset_sha256": asset_digest(),
        "evaluation_harness_sha256": harness_digest(),
        "codegraph_revision": manifest["arms"][1]["revision"],
        "codegraph_entrypoint": dict(runtime_identity["codegraph_entrypoint"]),
        "codegraph_build": dict(runtime_identity["codegraph_build"]),
        "source_backend": "libclang", "official_profile": True,
        "runtime_identity": runtime_identity,
        **declared, "identity_checks": identities,
        "cold_start_indexing": cold_start, "workspaces": workspaces,
    }


def _write_json(path: Path, value: dict[str, object]) -> bytes:
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.write_bytes(data)
    return data


def _release_fixture(tmp_path: Path, *, supported: bool = False) -> tuple[Path, ...]:
    wheel = tmp_path / "hlsgraph-0.3.0-py3-none-any.whl"
    package = {"hlsgraph/__init__.py": b'__version__ = "0.3.0"\n'}
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("hlsgraph/__init__.py", package["hlsgraph/__init__.py"])
    wheel_sha256 = hashlib.sha256(wheel.read_bytes()).hexdigest()
    environment = _environment(
        wheel_sha256, _payload_digest(package),
        work_root=tmp_path / "synthetic-eval-work",
    )
    environment_path = tmp_path / "environment.lock.json"
    environment_bytes = _write_json(environment_path, environment)
    candidate = prepared_hlsgraph_identity(environment, "hlsgraph-v03")
    static = {
        "schema_version": "hlsgraph.agent_eval.static_report.v1",
        "suite_asset_sha256": environment["suite_asset_sha256"],
        "evaluation_harness_sha256": environment["evaluation_harness_sha256"],
        "environment_lock_sha256": hashlib.sha256(environment_bytes).hexdigest(),
        "candidate_identity": candidate, "passed": True,
    }
    static_path = tmp_path / "static-report.json"
    static_bytes = _write_json(static_path, static)
    scores_path = tmp_path / "scores.jsonl"
    scores_bytes = render_score_rows([{"fixture_score": True}])
    scores_path.write_bytes(scores_bytes)
    run_set = {
        "schema_version": "hlsgraph.agent_eval.run_set.v1",
        "run_set_sha256": "4" * 64,
    }
    run_set_path = tmp_path / "run-set.json"
    _write_json(run_set_path, run_set)
    bootstrap = {
        "schema_version": "hlsgraph.agent_eval.bootstrap_report.v1",
        "suite_asset_sha256": environment["suite_asset_sha256"],
        "evaluation_harness_sha256": environment["evaluation_harness_sha256"],
        "environment_lock_sha256": hashlib.sha256(environment_bytes).hexdigest(),
        "candidate_identity": candidate,
        "static_report_sha256": hashlib.sha256(static_bytes).hexdigest(),
        "scores_sha256": hashlib.sha256(scores_bytes).hexdigest(),
        "run_set_sha256": run_set["run_set_sha256"],
        "run_batch_sha256": "3" * 64,
        "gates": {"performance_advantage_supported": supported},
    }
    bootstrap_path = tmp_path / "bootstrap-report.json"
    _write_json(bootstrap_path, bootstrap)
    notes = tmp_path / "release-notes.md"
    notes.write_text(
        "# HLSGraph 0.3.0\n\nTechnical Preview; no comparative advantage is claimed.\n",
        encoding="utf-8",
    )
    return (
        wheel, environment_path, static_path, bootstrap_path,
        scores_path, run_set_path, notes,
    )


def test_v02_wheel_verification_requires_a_source_checkout() -> None:
    with pytest.raises(SystemExit, match="v0.2.0 evaluation wheel verification"):
        wheel_identity_main([
            "--wheel", "missing.whl", "--expected-version", "0.2.0",
        ])


def test_setup_corpus_refuses_nonempty_output_without_force(tmp_path: Path) -> None:
    output = tmp_path / "existing-output"
    output.mkdir()
    sentinel = output / "peer-owned.txt"
    sentinel.write_text("keep\n", encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty corpus output"):
        materialize(output, repo_root=Path(__file__).resolve().parents[1])
    assert sentinel.read_text(encoding="utf-8") == "keep\n"


def test_prepare_rejects_any_preexisting_index_before_first_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus_ids = [item["id"] for item in load_corpus_lock()["corpora"]]
    clean_id, stale_id = corpus_ids[:2]
    clean = tmp_path / "codegraph" / clean_id
    stale = tmp_path / "hlsgraph-v03" / stale_id
    clean.mkdir(parents=True)
    stale.mkdir(parents=True)
    (clean / "source.cpp").write_text("void clean() {}\n", encoding="utf-8")
    (stale / "source.cpp").write_text("void stale() {}\n", encoding="utf-8")
    (stale / ".hlsgraph").mkdir()
    calls = 0

    def fake_run(*_args: object, **_kwargs: object) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("eval.agent_ab.prepare.subprocess.run", fake_run)
    plan = [
        {
            "kind": "codegraph-index", "arm": "codegraph", "corpus_id": clean_id,
            "cwd": str(clean), "command": ["codegraph", "init"],
        },
        {
            "kind": "hlsgraph-index", "arm": "hlsgraph-v03", "corpus_id": stale_id,
            "cwd": str(stale), "command": ["python", "-m", "hlsgraph.cli", "index"],
        },
    ]
    with pytest.raises(RuntimeError, match="already exists before pre_execution"):
        execute_plan(plan)
    assert calls == 0


def test_prepare_rechecks_absence_immediately_before_each_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus_ids = [item["id"] for item in load_corpus_lock()["corpora"]]
    first_id, second_id = corpus_ids[:2]
    first = tmp_path / "codegraph" / first_id
    second = tmp_path / "hlsgraph-v03" / second_id
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "source.cpp").write_text("void first() {}\n", encoding="utf-8")
    (second / "source.cpp").write_text("void second() {}\n", encoding="utf-8")
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    calls = 0

    def fake_run(*_args: object, **_kwargs: object) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        if calls == 1:
            (second / ".hlsgraph").mkdir()
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("eval.agent_ab.prepare.subprocess.run", fake_run)
    plan = [
        {
            "kind": "codegraph-index", "arm": "codegraph", "corpus_id": first_id,
            "cwd": str(first), "command": ["codegraph", "init"],
        },
        {
            "kind": "hlsgraph-index", "arm": "hlsgraph-v03", "corpus_id": second_id,
            "cwd": str(second), "command": ["python", "-m", "hlsgraph.cli", "index"],
        },
    ]
    with pytest.raises(RuntimeError, match="already exists before pre_index"):
        execute_plan(plan)
    assert calls == 1


def test_prepare_binds_both_source_repositories_and_records_cold_indexing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = build_plan(
        tmp_path, v02_python="py02", v03_python="py03",
        codegraph_command="node codegraph.js", codegraph_repo=tmp_path / "codegraph",
        v02_wheel=tmp_path / "v02.whl", v03_wheel=tmp_path / "v03.whl",
        invocation_root=tmp_path, v02_repo=tmp_path / "v02", v03_repo=tmp_path / "v03",
    )
    v02_revision = next(
        item["revision"] for item in load_manifest()["arms"]
        if item["id"] == "hlsgraph-v02"
    )
    assert next(
        item for item in plan if item["kind"] == "record-v02-revision"
    )["expected_stdout"] == v02_revision
    identity_steps = {
        item["arm"]: item for item in plan
        if item["kind"] == "verify-hlsgraph-wheel-installation"
    }
    assert all("--source-repo" in item["command"] for item in identity_steps.values())
    index = next(item for item in plan if item["kind"] == "codegraph-index")
    workspace = Path(index["cwd"])
    workspace.mkdir(parents=True)
    (workspace / "source.cpp").write_text("void dut() {}\n", encoding="utf-8")
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    captured: dict[str, object] = {}

    def fake_run(*_args: object, **kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        "eval.agent_ab.prepare.subprocess.run",
        fake_run,
    )
    ticks = iter((10.0, 10.375))
    monkeypatch.setattr("eval.agent_ab.prepare.time.perf_counter", lambda: next(ticks))
    observations = execute_plan([index])
    assert [item["kind"] for item in observations] == [
        "cold-start-index-absence", "cold-start-index-absence",
        "cold-start-index-phase", "cold-start-index",
    ]
    initial, immediate, phase, aggregate = observations
    assert initial["checkpoint"] == "pre_execution"
    assert immediate["checkpoint"] == "pre_index"
    assert initial["input_tree_sha256"] == immediate["input_tree_sha256"]
    assert phase["pre_execution_absence_proof_sha256"] == initial["proof_sha256"]
    assert phase["pre_index_absence_proof_sha256"] == immediate["proof_sha256"]
    assert phase["wall_time_seconds"] == 0.375
    assert phase["command_sha256"] == sha256_bytes(canonical_json(index["command"]))
    assert aggregate["phases"][0]["pre_index_absence_proof_sha256"] == (
        immediate["proof_sha256"]
    )
    assert all(captured["env"][key] == value for key, value in CODEGRAPH_OFFLINE_ENV.items())


def test_prepare_plan_requires_frozen_v02_source_repository(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="v0.2 preparation requires"):
        build_plan(
            tmp_path, v02_python="py02", v03_python="py03",
            codegraph_command="node codegraph.js", codegraph_repo=tmp_path / "codegraph",
            v02_wheel=tmp_path / "v02.whl", v03_wheel=tmp_path / "v03.whl",
            invocation_root=tmp_path, v03_repo=tmp_path / "v03",
        )


def test_environment_lock_requires_exact_v02_source_and_complete_cold_matrix(
    tmp_path: Path,
) -> None:
    environment = _environment(
        "1" * 64, "f" * 64, work_root=tmp_path / "synthetic-eval-work",
    )
    path = tmp_path / "environment.lock.json"
    _write_json(path, environment)
    loaded = load_environment_lock(path)
    v02 = prepared_hlsgraph_identity(loaded, "hlsgraph-v02")
    assert v02["source_revision"] == load_manifest()["arms"][2]["revision"]
    environment["hlsgraph_v02"]["source_revision"] = "0" * 40
    _write_json(path, environment)
    with pytest.raises(ValueError, match="exact source revision"):
        load_environment_lock(path)


def test_environment_lock_requires_v03_knowledge_sync_cold_phase(tmp_path: Path) -> None:
    environment = _environment(
        "1" * 64, "f" * 64, work_root=tmp_path / "synthetic-eval-work",
    )
    record = next(
        item for item in environment["cold_start_indexing"]
        if item["arm"] == "hlsgraph-v03"
    )
    record["phases"] = [
        phase for phase in record["phases"] if phase["phase"] != "knowledge_sync"
    ]
    path = tmp_path / "environment.lock.json"
    _write_json(path, environment)
    with pytest.raises(ValueError, match="phases are incomplete"):
        load_environment_lock(path)


def test_environment_lock_verifies_cold_start_absence_proofs(tmp_path: Path) -> None:
    environment = _environment(
        "1" * 64, "f" * 64, work_root=tmp_path / "synthetic-eval-work",
    )
    proof = next(
        item for item in environment["identity_checks"]
        if item.get("kind") == "cold-start-index-absence"
    )
    proof["status"] = "claimed-absent"
    path = tmp_path / "environment.lock.json"
    _write_json(path, environment)
    with pytest.raises(ValueError, match="absence proof is invalid"):
        load_environment_lock(path)


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda value: value.pop("runtime_identity"), "runtime identity"),
        (
            lambda value: value["runtime_identity"]["codex"].update({"sha256": "0" * 64}),
            "runtime identity hash",
        ),
        (
            lambda value: value["runtime_identity"]["sandbox_boundary"]["deny_roots"].pop(),
            "runtime identity hash",
        ),
    ],
)
def test_environment_lock_rejects_missing_or_mutated_runtime_identity(
    tmp_path: Path, mutation: object, message: str,
) -> None:
    environment = _environment(
        "1" * 64, "f" * 64, work_root=tmp_path / "synthetic-eval-work",
    )
    mutation(environment)
    path = tmp_path / "environment.lock.json"
    _write_json(path, environment)
    with pytest.raises(ValueError, match=message):
        load_environment_lock(path)


def test_release_gate_binds_evaluated_wheel_and_enforces_claim_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    wheel, environment, static, bootstrap, scores, run_set, notes = _release_fixture(tmp_path)
    expected = json.loads(bootstrap.read_text(encoding="utf-8"))
    monkeypatch.setattr("eval.agent_ab.bootstrap.analyze", lambda *_args, **_kwargs: expected)
    monkeypatch.setattr(
        release_audit, "_verify_evaluation_raw_closure",
        lambda **_kwargs: (
            json.loads(run_set.read_text(encoding="utf-8")),
            [json.loads(line) for line in scores.read_text(encoding="utf-8").splitlines()],
        ),
    )
    assert _audit_evaluation_release_gate(
        wheel, eval_identity=environment, static_report=static,
        bootstrap_report=bootstrap, scores=scores, run_set=run_set,
        release_notes=notes,
    ) == []
    notes.write_text(
        "# HLSGraph 0.3.0\n\nTechnical Preview that outperforms CodeGraph.\n",
        encoding="utf-8",
    )
    issues = _audit_evaluation_release_gate(
        wheel, eval_identity=environment, static_report=static,
        bootstrap_report=bootstrap, scores=scores, run_set=run_set,
        release_notes=notes,
    )
    assert any("claim an advantage" in item for item in issues)


def test_release_gate_rejects_a_different_wheel_even_with_same_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    wheel, environment, static, bootstrap, scores, run_set, notes = _release_fixture(tmp_path)
    expected = json.loads(bootstrap.read_text(encoding="utf-8"))
    monkeypatch.setattr("eval.agent_ab.bootstrap.analyze", lambda *_args, **_kwargs: expected)
    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr("hlsgraph/new.py", b"different release bytes\n")
    issues = _audit_evaluation_release_gate(
        wheel, eval_identity=environment, static_report=static,
        bootstrap_report=bootstrap, scores=scores, run_set=run_set,
        release_notes=notes,
    )
    assert any("wheel SHA-256 differs" in item for item in issues)
    assert any("package bytes differ" in item for item in issues)


def test_release_gate_does_not_trust_free_form_score_and_bootstrap_hashes(
    tmp_path: Path,
) -> None:
    wheel, environment, static, bootstrap, scores, run_set, notes = _release_fixture(tmp_path)
    issues = _audit_evaluation_release_gate(
        wheel, eval_identity=environment, static_report=static,
        bootstrap_report=bootstrap, scores=scores, run_set=run_set,
        release_notes=notes,
    )
    assert any("cannot independently recompute final evaluation" in item for item in issues)


def test_raw_closure_requires_declared_paths_and_rescores_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    work_root = tmp_path / "work"
    runs_root = tmp_path / "runs"
    work_root.mkdir()
    runs_root.mkdir()
    environment = {
        "runtime_identity": {"sandbox_boundary": {"work_root": str(work_root)}}
    }
    environment_bytes = b"{}\n"
    environment_path = work_root / "environment.lock.json"
    environment_path.write_bytes(environment_bytes)
    frozen = {"runs_root": str(runs_root), "cells": []}
    run_set_bytes = _write_json(runs_root / "run-set.json", frozen)
    supplied_scores = render_score_rows([{"metric": "supplied"}])

    from eval.agent_ab import score as score_module

    monkeypatch.setattr(score_module, "load_run_set", lambda *_args, **_kwargs: frozen)
    monkeypatch.setattr(score_module, "score_runs", lambda *_args, **_kwargs: [
        {"metric": "recomputed"},
    ])
    with pytest.raises(ValueError, match="raw-trace rescoring"):
        release_audit._verify_evaluation_raw_closure(
            environment=environment, environment_bytes=environment_bytes,
            eval_identity=environment_path,
            run_set_path=runs_root / "run-set.json", frozen_run_set=frozen,
            run_set_bytes=run_set_bytes, scores_bytes=supplied_scores,
        )

    copied = tmp_path / "copied-run-set.json"
    copied.write_bytes(run_set_bytes)
    with pytest.raises(ValueError, match="escapes its required root"):
        release_audit._verify_evaluation_raw_closure(
            environment=environment, environment_bytes=environment_bytes,
            eval_identity=environment_path, run_set_path=copied,
            frozen_run_set=frozen, run_set_bytes=run_set_bytes,
            scores_bytes=supplied_scores,
        )


def test_raw_closure_invokes_full_run_set_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    work_root = tmp_path / "work"
    runs_root = tmp_path / "runs"
    work_root.mkdir()
    runs_root.mkdir()
    environment = {
        "runtime_identity": {"sandbox_boundary": {"work_root": str(work_root)}}
    }
    environment_bytes = b"{}\n"
    environment_path = work_root / "environment.lock.json"
    environment_path.write_bytes(environment_bytes)
    frozen = {"runs_root": str(runs_root), "cells": []}
    run_set_bytes = _write_json(runs_root / "run-set.json", frozen)

    from eval.agent_ab import score as score_module

    def reject_matrix(*_args, **_kwargs):
        raise ValueError("run set is not the exact frozen 192-cell matrix")

    monkeypatch.setattr(score_module, "load_run_set", reject_matrix)
    with pytest.raises(ValueError, match="exact frozen 192-cell matrix"):
        release_audit._verify_evaluation_raw_closure(
            environment=environment, environment_bytes=environment_bytes,
            eval_identity=environment_path,
            run_set_path=runs_root / "run-set.json", frozen_run_set=frozen,
            run_set_bytes=run_set_bytes, scores_bytes=b"",
        )


def _stub_release_archive_audits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path]:
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "hlsgraph-0.3.0-py3-none-any.whl").write_bytes(b"wheel")
    (dist / "hlsgraph-0.3.0.tar.gz").write_bytes(b"sdist")
    suite_evidence = tmp_path / "suite-evidence"
    monkeypatch.setattr(release_audit, "_audit_source_tree", lambda _root: [])
    monkeypatch.setattr(release_audit, "_audit_sbom", lambda _data, _root: [])
    monkeypatch.setattr(release_audit, "_audit_wheel", lambda *_args: [])
    monkeypatch.setattr(release_audit, "_audit_sdist", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        release_audit, "_audit_knowledge_review_release_gate",
        lambda *_args, **_kwargs: [],
    )
    strict_file_bytes = release_audit._strict_file_bytes

    def read_release_input(path: Path, label: str, **kwargs: object) -> bytes:
        if label in {"release wheel", "release sdist"}:
            return b"archive"
        return strict_file_bytes(path, label, **kwargs)

    monkeypatch.setattr(release_audit, "_strict_file_bytes", read_release_input)
    monkeypatch.setattr(release_audit, "_release_wheel_package_digest", lambda _data: "d")
    monkeypatch.setattr(release_audit, "_release_sdist_package_digest", lambda _data: "d")
    return dist, suite_evidence


def test_formal_release_audit_cannot_omit_evaluation_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "hlsgraph-0.3.0-py3-none-any.whl").write_bytes(b"wheel")
    (dist / "hlsgraph-0.3.0.tar.gz").write_bytes(b"sdist")
    monkeypatch.setattr(release_audit, "_audit_source_tree", lambda _root: [])
    monkeypatch.setattr(release_audit, "_audit_sbom", lambda _data, _root: [])
    monkeypatch.setattr(release_audit, "_audit_wheel", lambda *_args: [])
    monkeypatch.setattr(release_audit, "_audit_sdist", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        release_audit, "_audit_knowledge_review_release_gate", lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(release_audit, "_strict_file_bytes", lambda *_args, **_kwargs: b"x")
    monkeypatch.setattr(release_audit, "_release_wheel_package_digest", lambda _data: "d")
    monkeypatch.setattr(release_audit, "_release_sdist_package_digest", lambda _data: "d")
    assert release_audit.main([str(dist)]) == 1
    assert "formal release audit requires" in capsys.readouterr().err


@pytest.mark.parametrize("preview_label", ["Technical Preview", "Developer Preview"])
def test_explicit_technical_preview_can_omit_all_agent_evaluation_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], preview_label: str,
) -> None:
    dist, suite_evidence = _stub_release_archive_audits(tmp_path, monkeypatch)
    notes = tmp_path / "release-notes.md"
    notes.write_text(
        f"# HLSGraph 0.3.0\n\n{preview_label}. Comparative performance is not claimed.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        release_audit, "_audit_evaluation_release_gate",
        lambda *_args, **_kwargs: pytest.fail(
            "the preview path must not consume Agent A/B evaluation evidence"
        ),
    )
    assert release_audit.main([
        str(dist),
        "--technical-preview-without-agent-eval",
        "--knowledge-review-suite-evidence", str(suite_evidence),
        "--release-notes", str(notes),
    ]) == 0
    output = capsys.readouterr().out
    assert "Preview release" in output
    assert "performance-advantage approval were omitted" in output


@pytest.mark.parametrize(
    "option",
    [
        "--eval-identity", "--static-report", "--bootstrap-report",
        "--scores", "--run-set",
    ],
)
def test_technical_preview_rejects_every_agent_evaluation_evidence_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], option: str,
) -> None:
    dist, suite_evidence = _stub_release_archive_audits(tmp_path, monkeypatch)
    notes = tmp_path / "release-notes.md"
    notes.write_text("# HLSGraph 0.3.0\n\nTechnical Preview.\n", encoding="utf-8")
    supplied_evidence = tmp_path / "partial-evidence.json"
    supplied_evidence.write_text("{}\n", encoding="utf-8")
    assert release_audit.main([
        str(dist),
        "--technical-preview-without-agent-eval",
        "--knowledge-review-suite-evidence", str(suite_evidence),
        "--release-notes", str(notes),
        option, str(supplied_evidence),
    ]) == 1
    assert (
        "requires every Agent A/B evaluation evidence input to be omitted"
        in capsys.readouterr().err
    )


@pytest.mark.parametrize(
    ("notes_text", "expected"),
    [
        (
            "# HLSGraph 0.3.0\n\nPreview release.\n",
            "must explicitly say Technical Preview or Developer Preview",
        ),
        (
            "# HLSGraph 0.3.0\n\nTechnical Preview that outperforms other tools.\n",
            "claim an advantage without a completed Agent A/B evaluation",
        ),
    ],
)
def test_technical_preview_requires_preview_label_and_forbids_advantage_claims(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], notes_text: str, expected: str,
) -> None:
    dist, suite_evidence = _stub_release_archive_audits(tmp_path, monkeypatch)
    notes = tmp_path / "release-notes.md"
    notes.write_text(notes_text, encoding="utf-8")
    assert release_audit.main([
        str(dist),
        "--technical-preview-without-agent-eval",
        "--knowledge-review-suite-evidence", str(suite_evidence),
        "--release-notes", str(notes),
    ]) == 1
    assert expected in capsys.readouterr().err


def test_technical_preview_requires_release_notes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dist, suite_evidence = _stub_release_archive_audits(tmp_path, monkeypatch)
    assert release_audit.main([
        str(dist),
        "--technical-preview-without-agent-eval",
        "--knowledge-review-suite-evidence", str(suite_evidence),
    ]) == 1
    assert (
        "--technical-preview-without-agent-eval requires --release-notes"
        in capsys.readouterr().err
    )


def test_preflight_is_explicit_and_never_claims_release_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "hlsgraph-0.3.0-py3-none-any.whl").write_bytes(b"wheel")
    (dist / "hlsgraph-0.3.0.tar.gz").write_bytes(b"sdist")
    monkeypatch.setattr(release_audit, "_audit_source_tree", lambda _root: [])
    monkeypatch.setattr(release_audit, "_audit_sbom", lambda _data, _root: [])
    monkeypatch.setattr(release_audit, "_audit_wheel", lambda *_args: [])
    monkeypatch.setattr(release_audit, "_audit_sdist", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(release_audit, "_strict_file_bytes", lambda *_args, **_kwargs: b"x")
    monkeypatch.setattr(release_audit, "_release_wheel_package_digest", lambda _data: "d")
    monkeypatch.setattr(release_audit, "_release_sdist_package_digest", lambda _data: "d")
    assert release_audit.main([str(dist), "--preflight-only"]) == 0
    output = capsys.readouterr().out
    assert "PRE-FLIGHT ONLY" in output
    assert "not a release approval" in output
