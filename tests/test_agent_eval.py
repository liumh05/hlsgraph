from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
from pathlib import Path
from types import SimpleNamespace
from pathlib import PurePosixPath
import zipfile

from setuptools._distutils.filelist import FileList

import pytest

from eval.agent_ab.audit import audit_frozen_assets, audit_public_artifacts
from eval.agent_ab.bootstrap import (
    _score_identity_issues, _validate_static_identity, evaluate_gates,
    stratified_paired_bootstrap, verify_scores_against_raw,
    verify_static_against_candidate,
)
from eval.agent_ab.common import (
    ARM_IDS, CODEGRAPH_OFFLINE_ENV, asset_digest, canonical_json, harness_digest, load_corpus_lock,
    load_manifest, load_questions,
    load_static_cases, prepared_hlsgraph_identity, resolve_command_argv,
    resolve_local_executable, sha256_bytes, sha256_file,
    verify_evaluation_checkout, workspace_identity,
)
from eval.agent_ab.parse_trace import normalize_trace
from eval.agent_ab.prepare import build_plan as build_prepare_plan
from eval.agent_ab.runner import build_codex_command, build_prompt, build_run_plan
from eval.agent_ab.sanitize import sanitize_file, sanitize_text
from eval.agent_ab.score import (
    _validate_run_metadata, _validate_trace_challenge, render_score_rows,
    score_answer, verify_workspace_corpus,
)
from eval.agent_ab.setup_corpus import _hlsgraph_manifest, _provenance
from eval.agent_ab.static_eval import render_static_json, score as score_static
from eval.agent_ab.wheel_identity import inspect_installation
from tests.agent_eval_runtime_support import (
    synthetic_cold_start_input_sha256, synthetic_cold_start_matrix,
    synthetic_runtime_identity,
)


ROOT = Path(__file__).resolve().parents[1]


def _write_environment_lock(work_root: Path) -> tuple[str, dict[str, str]]:
    manifest = load_manifest()
    identities = []
    declared: dict[str, dict[str, str]] = {}
    for arm, key, version, marker in (
        ("hlsgraph-v02", "hlsgraph_v02", "0.2.0", "2"),
        ("hlsgraph-v03", "hlsgraph_v03", "0.3.0", "3"),
    ):
        wheel_hash = marker * 64
        payload_hash = ("a" if marker == "2" else "b") * 64
        source_revision = (
            manifest["arms"][2]["revision"] if arm == "hlsgraph-v02" else "c" * 40
        )
        source_package_hash = ("e" if marker == "2" else "d") * 64
        declared[key] = {
            "version": version, "wheel": f"hlsgraph-{version}.whl",
            "wheel_sha256": wheel_hash,
            "revision": source_revision,
            "source_revision": source_revision,
            "source_package_sha256": source_package_hash,
            "wheel_package_sha256": source_package_hash,
        }
        identities.append({
            "kind": "verify-hlsgraph-wheel-installation", "arm": arm,
            "identity": {
                "schema_version": "hlsgraph.agent_eval.wheel_identity.v1",
                "verified": True, "version": version,
                "wheel_sha256": wheel_hash,
                "wheel_payload_sha256": payload_hash,
                "installed_payload_sha256": payload_hash,
                "source_revision": source_revision,
                "source_package_sha256": source_package_hash,
                "wheel_package_sha256": source_package_hash,
            },
        })
        label = "v02" if arm == "hlsgraph-v02" else "v03"
        identities.extend([
            {"kind": f"verify-{label}-repo-clean", "stdout": ""},
            {"kind": f"record-{label}-revision", "stdout": source_revision},
            {"kind": f"verify-{label}-repo-clean-after", "stdout": ""},
            {"kind": f"record-{label}-revision-after", "stdout": source_revision},
        ])
    runtime_identity = synthetic_runtime_identity(
        public_repository=ROOT, work_root=work_root,
    )
    environment = {
        "schema_version": "hlsgraph.agent_eval.environment.v3",
        "suite_asset_sha256": asset_digest(),
        "evaluation_harness_sha256": harness_digest(),
        "codegraph_revision": manifest["arms"][1]["revision"],
        "codegraph_entrypoint": dict(runtime_identity["codegraph_entrypoint"]),
        "codegraph_build": dict(runtime_identity["codegraph_build"]),
        "source_backend": "libclang", "official_profile": True,
        "runtime_identity": runtime_identity,
        **declared, "identity_checks": identities,
    }
    for arm in ARM_IDS:
        for corpus in load_corpus_lock()["corpora"]:
            identity = {
                "schema_version": "hlsgraph.agent_eval.workspace_identity.v1",
                "arm": arm, "corpus_id": corpus["id"],
                "snapshot_id": "snapshot" if arm == "hlsgraph-v03" else None,
                "index_sha256": hashlib.sha256(
                    f"{arm}/{corpus['id']}".encode("utf-8")
                ).hexdigest(),
                "cold_start_input_sha256": synthetic_cold_start_input_sha256(
                    arm, corpus["id"],
                ),
            }
            identity["workspace_identity_sha256"] = sha256_bytes(canonical_json(identity))
            environment.setdefault("workspaces", {})[f"{arm}/{corpus['id']}"] = identity
    cold_start, cold_checks = synthetic_cold_start_matrix()
    identities.extend(cold_checks)
    environment["cold_start_indexing"] = cold_start
    work_root.mkdir(parents=True, exist_ok=True)
    path = work_root / "environment.lock.json"
    path.write_text(json.dumps(environment, sort_keys=True) + "\n", encoding="utf-8")
    return sha256_file(path), prepared_hlsgraph_identity(environment, "hlsgraph-v03")


def test_sdist_manifest_includes_frozen_suite_but_excludes_runtime_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(ROOT)
    file_list = FileList()
    file_list.findall()
    for raw in (ROOT / "MANIFEST.in").read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            file_list.process_template_line(line)
    names = {item.replace("\\", "/") for item in file_list.files}
    required = {
        "eval/__init__.py", "eval/agent_ab/README.md",
        "eval/agent_ab/manifest.json", "eval/agent_ab/questions.jsonl",
        "eval/agent_ab/static_cases.jsonl", "eval/agent_ab/score.py",
        "eval/agent_ab/bootstrap.py", "eval/agent_ab/static_eval.py",
        "eval/agent_ab/stubs/hls_stream.h", "eval/agent_ab/stubs/algorithm",
    }
    assert required.issubset(names)
    assert not any(
        name.startswith((
            "eval/agent_ab/work/", "eval/agent_ab/runs/", "eval/agent_ab/results/",
        )) or "/__pycache__/" in name
        for name in names
    )


def test_frozen_public_eval_assets_are_complete_and_licensed() -> None:
    manifest = load_manifest()
    lock = load_corpus_lock()
    questions = load_questions()

    assert manifest["model"] == {"id": "gpt-5.6-sol", "reasoning_effort": "medium"}
    assert manifest["repetitions"] == 4
    assert manifest["indexing"] == {"source_backend": "libclang", "allow_degraded": False}
    assert tuple(item["id"] for item in manifest["arms"]) == ARM_IDS
    assert manifest["arms"][1]["revision"] == "286e9ccc2dad45336d4fd67052930322054d64b5"
    assert manifest["arms"][1]["entrypoint_sha256"] == (
        "03e4c791cc0dd91ed264278461bf9a56c0278aa0670d5942fc4732311c66de03"
    )
    build = manifest["arms"][1]["build_identity"]
    assert build["runtime_tree_algorithm"] == "hlsgraph.runtime_tree.v1"
    assert build["dist_tree_sha256"] == (
        "cc0cefe48514fa34a8c3b488efb4377bec2f62ad84e32c57f495e2cd2cb2e61b"
    )
    assert build["dependency_tree_sha256"] == (
        "20088cced4df7332c2787bf7d281e301a67d8fd831dad53a564a8d50d723a284"
    )
    assert len(questions) == 12
    assert len(load_static_cases()) == 12
    assert all(
        "\x08" not in pattern
        for question in questions
        for rule in question.get("forbidden_claims", [])
        for pattern in rule.get("patterns", [])
    )
    assert {item["license"] for item in lock["corpora"]} == {"Apache-2.0"}
    assert {item["tool_evidence"] for item in lock["corpora"] if item["id"] != "dataflow_gemm"} == {"none"}
    audit = audit_frozen_assets()
    assert audit["passed"] is True
    assert audit["codegraph_entrypoint_sha256"] == manifest["arms"][1][
        "entrypoint_sha256"
    ]
    assert audit["codegraph_dist_tree_sha256"] == build["dist_tree_sha256"]
    assert audit["codegraph_dependency_tree_sha256"] == build[
        "dependency_tree_sha256"
    ]


def test_run_plan_is_seeded_and_has_192_cells() -> None:
    first = build_run_plan()
    second = build_run_plan()
    assert first == second
    assert len(first) == 12 * 4 * 4
    assert len({row["run_id"] for row in first}) == len(first)
    assert {row["arm"] for row in first} == set(ARM_IDS)
    assert {row["timeout_seconds"] for row in first} == {900}
    assert build_run_plan(seed=1) != first


def test_codex_command_is_read_only_ephemeral_and_has_one_arm_server(tmp_path: Path) -> None:
    work_root = tmp_path / "work"
    runs_root = tmp_path / "runs"
    record = {
        "run_id": "x", "question_id": "dg-architecture-flow",
        "corpus_id": "dataflow_gemm", "category": "architecture",
        "arm": "hlsgraph-v03", "repetition": 1,
    }
    command = build_codex_command(
        record, work_root=work_root, runs_root=runs_root,
        codex_command="codex", v02_python="py02",
        v03_python="py03", codegraph_command="codegraph",
    )
    assert "--ignore-user-config" in command
    assert "--ignore-rules" in command
    assert "--ephemeral" in command
    assert command[command.index("-a") + 1] == "never"
    assert "browser_use" in command
    assert "multi_agent" in command
    assert "workspace_dependencies" in command
    assert "--strict-config" in command
    assert "--sandbox" not in command
    assert any(item == 'default_permissions="hlsgraph_eval"' for item in command)
    assert any(item == 'permissions.hlsgraph_eval.network.enabled=false' for item in command)
    assert any(item.startswith('permissions.hlsgraph_eval.filesystem={') for item in command)
    assert any(
        '":minimal"="read"' in item
        for item in command
    )
    assert not any(runs_root.resolve().as_posix() in item for item in command)
    assert command[command.index("--model") + 1] == "gpt-5.6-sol"
    assert any("HLSGRAPH_MCP_TOOLS=\"explore\"" in item for item in command)
    assert not any("codegraph" in item.casefold() for item in command)


def test_prepare_is_dry_plan_with_pinned_identity_checks(tmp_path: Path) -> None:
    v02_wheel = tmp_path / "hlsgraph-0.2.0-py3-none-any.whl"
    v03_wheel = tmp_path / "hlsgraph-0.3.0-py3-none-any.whl"
    plan = build_prepare_plan(
        tmp_path, v02_python="py02", v03_python="py03",
        codegraph_command="node codegraph.js", codegraph_repo=tmp_path / "codegraph-src",
        v02_wheel=v02_wheel, v03_wheel=v03_wheel, invocation_root=tmp_path,
        v02_repo=tmp_path / "v02-src",
        v03_repo=tmp_path / "v03-src",
    )
    assert plan[0]["kind"] == "record-codex-version"
    revision_step = next(step for step in plan if step["kind"] == "verify-codegraph-revision")
    assert revision_step["expected_stdout"] == "286e9ccc2dad45336d4fd67052930322054d64b5"
    assert sum(step["kind"] == "codegraph-index" for step in plan) == 4
    assert sum(step["kind"] == "hlsgraph-index" for step in plan) == 8
    assert sum(step["kind"] == "verify-hlsgraph-wheel-installation" for step in plan) == 2
    assert sum(step["kind"] == "hlsgraph-knowledge-sync" for step in plan) == 4
    assert any(step["kind"] == "record-v03-revision" for step in plan)
    assert any(step["kind"] == "verify-v03-repo-clean" for step in plan)
    assert any(step["kind"] == "record-node-version" for step in plan)
    codegraph_step = next(step for step in plan if step["kind"] == "codegraph-index")
    assert Path(codegraph_step["command"][1]).is_absolute()
    assert codegraph_step["expected_entrypoint_sha256"] == load_manifest()["arms"][1][
        "entrypoint_sha256"
    ]
    assert codegraph_step["environment"] == CODEGRAPH_OFFLINE_ENV
    assert all("--degraded" not in step["command"] for step in plan if step["kind"] == "hlsgraph-index")
    assert {step.get("expected_stdout") for step in plan if step["kind"] == "verify-hlsgraph-version"} == {
        "0.2.0", "0.3.0",
    }
    identity_steps = {
        step["arm"]: step for step in plan
        if step["kind"] == "verify-hlsgraph-wheel-installation"
    }
    v02_command = identity_steps["hlsgraph-v02"]["command"]
    assert Path(v02_command[v02_command.index("--source-repo") + 1]) == (
        tmp_path / "v02-src"
    ).resolve()
    v03_command = identity_steps["hlsgraph-v03"]["command"]
    assert Path(v03_command[v03_command.index("--source-repo") + 1]) == (
        tmp_path / "v03-src"
    ).resolve()


def test_path_like_python_is_resolved_before_per_corpus_cwd(tmp_path: Path) -> None:
    executable = tmp_path / ".venv-v03" / "Scripts" / "python.exe"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"fixture")
    v02_executable = tmp_path / ".venv-v02" / "Scripts" / "python.exe"
    v02_executable.parent.mkdir(parents=True)
    v02_executable.write_bytes(b"fixture")
    expected = str(executable.resolve())
    assert resolve_local_executable(
        ".venv-v03/Scripts/python.exe", tmp_path,
    ) == expected
    assert resolve_local_executable("python", tmp_path) == "python"
    plan = build_prepare_plan(
        tmp_path, v02_python=".venv-v02/Scripts/python.exe",
        v03_python=".venv-v03/Scripts/python.exe",
        codegraph_command="node codegraph.js", invocation_root=tmp_path,
        codegraph_repo=tmp_path / "codegraph-src",
        v02_wheel=tmp_path / "hlsgraph-0.2.0-py3-none-any.whl",
        v03_wheel=tmp_path / "hlsgraph-0.3.0-py3-none-any.whl",
        v02_repo=tmp_path / "v02-src",
        v03_repo=tmp_path / "v03-src",
    )
    commands = [step["command"] for step in plan
                if step["kind"] == "verify-hlsgraph-version"]
    assert all(Path(command[0]).is_absolute() for command in commands)


def test_bare_codex_command_resolves_windows_cmd_shim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    shim = tmp_path / "codex.CMD"
    shim.write_text("@exit /b 0\n", encoding="utf-8")
    import eval.agent_ab.common as common
    monkeypatch.setattr(common.shutil, "which", lambda _value: str(shim))
    assert resolve_command_argv("codex") == [str(shim.resolve())]
    if os.name == "nt" and shutil.which("codex.cmd"):
        assert Path(resolve_command_argv("codex")[0]).suffix.casefold() == ".cmd"


def _perfect_architecture_answer() -> dict[str, object]:
    return {
        "answer": "load feeds compute and then store through values and results.",
        "claims": [
            {
                "id": "flow", "statement": "load sends values to compute, which sends results to store.",
                "truth_plane": "design_fact", "stage": "source", "authority": "static source",
                "evidence": [{"path": "kernel.cpp", "line_start": 31, "line_end": 39}],
            },
            {
                "id": "values", "statement": "values has declared depth 8.",
                "truth_plane": "design_fact", "stage": "source", "authority": "static source",
                "evidence": [{"path": "kernel.cpp", "line_start": 33, "line_end": 35}],
            },
            {
                "id": "results", "statement": "results has declared depth 16.",
                "truth_plane": "design_fact", "stage": "source", "authority": "static source",
                "evidence": [{"path": "kernel.cpp", "line_start": 34, "line_end": 36}],
            },
            {
                "id": "boundary", "statement": "A function call graph is insufficient and cannot alone prove hardware topology.",
                "truth_plane": "knowledge_guidance", "stage": None, "authority": "evaluation invariant",
                "evidence": [{"path": "kernel.cpp", "line_start": 31, "line_end": 39}],
            },
        ],
        "uncertainties": [],
    }


def test_trace_parser_and_evidence_scorer_are_deterministic(tmp_path: Path) -> None:
    answer = _perfect_architecture_answer()
    events = [
        {"type": "item.completed", "item": {"id": "tool-1", "type": "command_execution", "command": "rg DATAFLOW kernel.cpp"}},
        {"type": "item.completed", "item": {"id": "msg-1", "type": "agent_message", "text": json.dumps(answer)}},
        {"type": "turn.completed", "usage": {
            "input_tokens": 100, "cached_input_tokens": 25,
            "output_tokens": 50, "total_tokens": 150,
        }},
    ]
    normalized = normalize_trace(events)
    assert normalized["answer"] == answer
    assert normalized["tool_calls"] == 1
    assert normalized["file_reads"] == 1
    assert normalized["usage"]["cached_input_tokens"] == 25
    assert normalized["usage"]["total_tokens"] == 150

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    shutil.copyfile(ROOT / "examples/dataflow_gemm/kernel.cpp", workspace / "kernel.cpp")
    question = next(item for item in load_questions() if item["id"] == "dg-architecture-flow")
    scored = score_answer(question, answer, workspace)
    assert scored["evidence_grounded_correctness"] == pytest.approx(1.0)
    assert scored["citation_precision"] == pytest.approx(1.0)
    assert scored["fabricated_truth_count"] == 0

    wrong_stage = _perfect_architecture_answer()
    wrong_stage["claims"][0]["stage"] = "post_route"
    wrong = score_answer(question, wrong_stage, workspace)
    assert wrong["criteria"][0]["contract_match"] is False
    assert wrong["fabricated_truth_count"] >= 1

    answer["claims"].append({
        "id": "fake", "statement": "A real Vitis run proves this flow.",
        "truth_plane": "tool_observation", "stage": "source", "authority": "invented",
        "evidence": [{"path": "kernel.cpp", "line_start": 31, "line_end": 39}],
    })
    assert score_answer(question, answer, workspace)["fabricated_truth_count"] >= 1

    irrelevant = _perfect_architecture_answer()
    irrelevant["claims"].append({
        "id": "irrelevant", "statement": "An unrelated source line proves extra hardware.",
        "truth_plane": "design_fact", "stage": "source", "authority": "static source",
        "evidence": [{"path": "kernel.cpp", "line_start": 1, "line_end": 1}],
    })
    irrelevant_score = score_answer(question, irrelevant, workspace)
    assert irrelevant_score["unsupported_claims"] == 1
    assert irrelevant_score["fabricated_truth_count"] >= 1


def test_generated_manifests_enable_bounded_public_context_and_declare_tool_context() -> None:
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10 support declared by the package.
        import tomli as tomllib

    corpus = load_corpus_lock()["corpora"][0]
    value = tomllib.loads(_hlsgraph_manifest(corpus, schema_version="0.3.0"))
    assert value["metadata"]["privacy"]["mcp_source_snippets"] == "bounded"
    assert {(item["name"], item["version"]) for item in value["toolchains"]} == {
        ("vitis_hls", "2024.2"), ("vivado", "2024.2"),
    }
    assert value["build"]["translation_units"][0]["arguments"][:2] == [
        "-std=c++17", "-Isupport/include",
    ]


def _matching_text(pattern: str) -> str:
    return pattern.split("|")[0].replace(r"\b", "").replace("\\", "")


def test_static_retrieval_gate_scores_normalized_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    work_root = tmp_path / "work"
    environment_sha256, candidate_identity = _write_environment_lock(work_root)
    environment = json.loads((work_root / "environment.lock.json").read_text(encoding="utf-8"))
    import eval.agent_ab.static_eval as static_eval
    monkeypatch.setattr(static_eval, "verify_evaluation_checkout", lambda _environment: None)
    monkeypatch.setattr(
        static_eval, "verify_prepared_workspace",
        lambda _environment, _work_root, arm, corpus_id:
            environment["workspaces"][f"{arm}/{corpus_id}"],
    )
    rows = []
    for case in load_static_cases():
        items = []
        for matcher in case["gold"]:
            data = {}
            if matcher.get("data_rule_id"):
                data["rule_id"] = matcher["data_rule_id"]
            if matcher.get("data_kind"):
                data["kind"] = _matching_text(matcher["data_kind"])
            citation = None
            if case["citation_gold"]:
                selector = case["citation_gold"][0]
                citation = {
                    "document_id": selector["document_id"],
                    "section": _matching_text(selector.get("section_pattern", "section")),
                }
            record_kind = matcher.get("record_kind", "entity")
            authority, stage = {
                "entity": ("static_fact", "ast"),
                "derivation": ("derived_fact", "ast"),
                "observation": ("synthetic", "cosim"),
                "relation": ("declared_constraint", "source"),
                "diagnostic": ("tool_observation", "ast"),
                "knowledge_rule": ("knowledge_rule", "source"),
            }[record_kind]
            items.append({
                "record_id": matcher["id"], "plane": (
                    "knowledge" if case["result_section"] == "guidance" else "facts"
                ),
                "record_kind": record_kind,
                "title": _matching_text(matcher.get("pattern", matcher["id"])),
                "summary": "", "authority_class": authority, "stage": stage,
                "data": data, "citation": citation,
            })
        result = {
            "snapshot_id": "snapshot", "facts": [], "guidance": [],
            "predictions": [], "flow": [],
            "trace": {
                "snapshot_id": "snapshot",
                "query_sha256": hashlib.sha256(case["query"].encode("utf-8")).hexdigest(),
                "profile": "hls.default.v1", "profile_schema_version": "0.3.0",
                "algorithm_version": "hlsgraph.hybrid_retrieval.v1",
                "profile_hash": "a" * 64, "graph_hash": "b" * 64,
                "elapsed_ms": {}, "private_snippets_requested": False,
                "private_snippets_returned": False,
            },
        }
        result[case["result_section"]] = items
        rows.append({"case_id": case["id"], "corpus_id": case["corpus_id"],
                     "snapshot_id": "snapshot",
                     "workspace_identity_sha256": environment["workspaces"][
                         f"hlsgraph-v03/{case['corpus_id']}"
                     ]["workspace_identity_sha256"],
                     "result": result})
    payload = {
        "schema_version": "hlsgraph.agent_eval.static_results.v1",
        "suite_asset_sha256": asset_digest(),
        "evaluation_harness_sha256": harness_digest(),
        "environment_lock_sha256": environment_sha256,
        "candidate_identity": candidate_identity,
        "top_k": 8, "cases": rows,
    }
    payload["raw_results_sha256"] = sha256_bytes(canonical_json(payload))
    report = score_static(payload, work_root=work_root)
    assert report["metrics"]["recall_at_8"] == pytest.approx(1.0)
    assert report["metrics"]["ndcg_at_8"] == pytest.approx(1.0)
    assert report["metrics"]["citation_precision"] == pytest.approx(1.0)
    assert report["metrics"]["fabricated_truth_count"] == 0
    assert report["passed"] is True
    stale = json.loads(json.dumps(payload))
    stale["cases"][0]["result"]["snapshot_id"] = "another-snapshot"
    stale["raw_results_sha256"] = sha256_bytes(canonical_json({
        key: value for key, value in stale.items() if key != "raw_results_sha256"
    }))
    with pytest.raises(ValueError, match="stale retrieval trace"):
        score_static(stale, work_root=work_root)
    stale_index = json.loads(json.dumps(payload))
    stale_index["cases"][0]["workspace_identity_sha256"] = "0" * 64
    stale_index["raw_results_sha256"] = sha256_bytes(canonical_json({
        key: value for key, value in stale_index.items() if key != "raw_results_sha256"
    }))
    with pytest.raises(ValueError, match="stale or relabelled identity"):
        score_static(stale_index, work_root=work_root)


def test_static_report_identity_rejects_stale_suite_and_candidate() -> None:
    environment_sha256 = "e" * 64
    candidate_identity = {
        "arm": "hlsgraph-v03", "version": "0.3.0",
        "wheel_sha256": "3" * 64,
        "installed_payload_sha256": "b" * 64,
        "revision": "c" * 40,
    }
    report = {
        "schema_version": "hlsgraph.agent_eval.static_report.v1",
        "suite_asset_sha256": asset_digest(),
        "evaluation_harness_sha256": harness_digest(),
        "environment_lock_sha256": environment_sha256,
        "candidate_identity": candidate_identity,
        "raw_results_sha256": "a" * 64,
        "passed": True, "metrics": {
            "recall_at_8": 1.0, "ndcg_at_8": 1.0,
            "citation_precision": 1.0, "citation_count": 1,
            "fabricated_truth_count": 0,
        },
    }
    _validate_static_identity(
        report, environment_lock_sha256=environment_sha256,
        candidate_identity=candidate_identity,
    )
    for key, bad_value in (
        ("suite_asset_sha256", "0" * 64),
        ("environment_lock_sha256", "1" * 64),
        ("candidate_identity", {**candidate_identity, "revision": "d" * 40}),
        ("passed", False),
    ):
        changed = {**report, key: bad_value}
        with pytest.raises(ValueError, match="stale, unpassed"):
            _validate_static_identity(
                changed, environment_lock_sha256=environment_sha256,
                candidate_identity=candidate_identity,
            )


def test_run_metadata_cannot_be_relabelled_or_detached_from_environment(
    tmp_path: Path,
) -> None:
    question = load_questions()[0]
    arm = "hlsgraph-v03"
    repetition = 1
    run_id = f"{question['id']}__r{repetition:02d}__{arm}"
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    challenge = "a" * 64
    prompt = build_prompt(question, arm, trace_challenge=challenge)
    (run_dir / "prompt.txt").write_text(prompt, encoding="utf-8", newline="")
    environment_sha256 = "e" * 64
    expected_cell = {
        "run_id": run_id, "question_id": question["id"],
        "corpus_id": question["corpus_id"], "category": question["category"],
        "arm": arm, "repetition": repetition,
        "run_contract_sha256": "c" * 64,
        "workspace_identity_sha256": "w" * 64,
        "trace_challenge": challenge,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "command_argv": ["codex", "exec"],
        "timeout_seconds": 900,
    }
    run_set = {"batch_id": "b" * 32, "run_set_sha256": "d" * 64}
    run = {
        "schema_version": "hlsgraph.agent_eval.run.v1",
        "run_id": run_id, "question_id": question["id"],
        "corpus_id": question["corpus_id"], "category": question["category"],
        "arm": arm, "repetition": repetition,
        "suite_asset_sha256": asset_digest(),
        "evaluation_harness_sha256": harness_digest(),
        "environment_lock_sha256": environment_sha256,
        "batch_id": run_set["batch_id"], "run_set_sha256": run_set["run_set_sha256"],
        "run_contract_sha256": expected_cell["run_contract_sha256"],
        "workspace_identity_sha256": expected_cell["workspace_identity_sha256"],
        "trace_challenge": challenge,
        "workspace": f"$WORK_ROOT/{arm}/{question['corpus_id']}",
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "command_argv": ["codex", "exec"],
        "timeout_seconds": 900,
        "wall_time_seconds": 1.0, "timed_out": False, "returncode": 0,
    }
    assert _validate_run_metadata(
        run, run_dir, expected_cell, run_set, environment_sha256,
    ) == question
    mutations = (
        {"arm": "native"},
        {"run_id": "relabelled"},
        {"suite_asset_sha256": "0" * 64},
        {"environment_lock_sha256": "1" * 64},
        {"corpus_id": "another-corpus"},
        {"timeout_seconds": 901},
    )
    for mutation in mutations:
        with pytest.raises(ValueError, match="relabelled, stale"):
            _validate_run_metadata(
                {**run, **mutation}, run_dir, expected_cell, run_set,
                environment_sha256,
            )
    answer = {"uncertainties": [f"eval-context:{challenge}"]}
    _validate_trace_challenge(answer, expected_cell)
    with pytest.raises(ValueError, match="not bound"):
        _validate_trace_challenge(
            {"uncertainties": ["eval-context:" + "e" * 64]}, expected_cell,
        )


def _gate_fixture() -> tuple[
    str, dict[str, str], list[dict[str, object]],
    list[dict[str, object]], dict[str, object],
]:
    environment_sha256 = "e" * 64
    candidate_identity = {
        "arm": "hlsgraph-v03", "version": "0.3.0",
        "wheel_sha256": "3" * 64,
        "installed_payload_sha256": "b" * 64,
        "revision": "c" * 40,
    }
    rows: list[dict[str, object]] = []
    batch_id = "a" * 32
    run_set_sha256 = "b" * 64
    for question in load_questions():
        for repetition in range(1, 5):
            for arm in ARM_IDS:
                run_id = f"{question['id']}__r{repetition:02d}__{arm}"
                source_hashes = {
                    name: hashlib.sha256(f"{run_id}:{name}".encode("utf-8")).hexdigest()
                    for name in (
                        "run.json", "prompt.txt", "codex.jsonl", "codex.stderr.log",
                        "retrieval-access.jsonl",
                    )
                }
                rows.append({
                    "schema_version": "hlsgraph.agent_eval.score.v1",
                    "suite_asset_sha256": asset_digest(),
                    "evaluation_harness_sha256": harness_digest(),
                    "environment_lock_sha256": environment_sha256,
                    "batch_id": batch_id, "run_set_sha256": run_set_sha256,
                    "run_contract_sha256": hashlib.sha256(
                        f"contract:{run_id}".encode("utf-8")
                    ).hexdigest(),
                    "workspace_identity_sha256": hashlib.sha256(
                        f"workspace:{arm}:{question['corpus_id']}".encode("utf-8")
                    ).hexdigest(),
                    "source_hashes": source_hashes,
                    "trace_sha256": source_hashes["codex.jsonl"],
                    "run_source_sha256": sha256_bytes(canonical_json(source_hashes)),
                    "thread_id": f"thread-{run_id}",
                    "run_id": run_id,
                    "question_id": question["id"], "repetition": repetition, "arm": arm,
                    "corpus_id": question["corpus_id"], "category": question["category"],
                    "returncode": 0, "timed_out": False,
                    "timeout_seconds": 900,
                    "evidence_grounded_correctness": 1.0,
                    "citation_precision": 1.0, "file_reads": 1,
                    "tool_calls": 1, "total_tokens": 10,
                    "wall_time_seconds": 1.0,
                    "fabricated_truth_count": 0,
                    "retrieval_audit": {
                        "schema_version": "hlsgraph.agent_eval.retrieval_audit.v1",
                        "status": (
                            "verified" if arm == "hlsgraph-v03" else "not_applicable"
                        ),
                        "sha256": hashlib.sha256(b"").hexdigest(),
                        "record_count": 0, "returned_count": 0,
                        "returned_bytes": 0, "source_access_calls": 0,
                        "receipt_sha256": hashlib.sha256(
                            f"audit:{run_id}".encode("utf-8")
                        ).hexdigest(),
                    },
                    "trace_policy": {
                        "schema_version": "hlsgraph.agent_eval.trace_policy.v1",
                        "passed": True, "arm": arm,
                        "workspace": "$CORPUS_WORKSPACE",
                        "completed_tools": 1,
                        "treatment_mcp_required": arm != "native",
                        "treatment_mcp_calls": 0 if arm == "native" else 1,
                        "first_call_treatment_mcp": arm != "native",
                        "treatment_mcp_first_outcome": (
                            "not_applicable" if arm == "native" else "completed"
                        ),
                        "treatment_mcp_outcomes": (
                            [] if arm == "native" else ["completed"]
                        ),
                    },
                })
    batch_sources = [
        {"run_id": row["run_id"], "run_source_sha256": row["run_source_sha256"]}
        for row in sorted(rows, key=lambda item: str(item["run_id"]))
    ]
    run_batch_sha256 = sha256_bytes(canonical_json(batch_sources))
    for row in rows:
        row["run_batch_sha256"] = run_batch_sha256
    manifest = load_manifest()
    comparisons = [
        {
            "candidate": "hlsgraph-v03", "baseline": baseline, "metric": metric,
            "orientation": "positive_favors_candidate", "observed_delta": 1.0,
            "ci_lower": 1.0, "ci_upper": 1.0,
            "confidence": manifest["bootstrap"]["confidence"],
            "samples": manifest["bootstrap"]["samples"],
            "seed": manifest["bootstrap"]["seed"],
            "question_strata": 12, "paired_cells": 48,
        }
        for baseline in ("native", "codegraph", "hlsgraph-v02")
        for metric in (
            "citation_precision", "evidence_grounded_correctness", "file_reads",
            "tool_calls", "total_tokens", "wall_time_seconds",
        )
    ]
    static_report = {
        "schema_version": "hlsgraph.agent_eval.static_report.v1",
        "suite_asset_sha256": asset_digest(),
        "evaluation_harness_sha256": harness_digest(),
        "environment_lock_sha256": environment_sha256,
        "candidate_identity": candidate_identity,
        "raw_results_sha256": "f" * 64,
        "passed": True,
        "metrics": {
            "recall_at_8": 1.0, "ndcg_at_8": 1.0,
            "citation_precision": 1.0, "citation_count": 1,
            "fabricated_truth_count": 0,
        },
    }
    return environment_sha256, candidate_identity, rows, comparisons, static_report


def test_exact_matrix_gate_rejects_a_duplicate_cell() -> None:
    environment_sha256, candidate_identity, rows, comparisons, static_report = _gate_fixture()
    complete = evaluate_gates(
        rows, comparisons, static_report,
        environment_lock_sha256=environment_sha256,
        candidate_identity=candidate_identity,
    )
    assert complete["complete_matrix"] is True
    duplicate = evaluate_gates(
        [*rows, rows[0]], comparisons, static_report,
        environment_lock_sha256=environment_sha256,
        candidate_identity=candidate_identity,
    )
    assert duplicate["complete_matrix"] is False
    assert duplicate["duplicate_cells"]
    relabelled_row = {**rows[0], "arm": (
        "native" if rows[0]["arm"] != "native" else "codegraph"
    )}
    relabelled = evaluate_gates(
        [relabelled_row, *rows[1:]], comparisons, static_report,
        environment_lock_sha256=environment_sha256,
        candidate_identity=candidate_identity,
    )
    assert relabelled["complete_matrix"] is False
    assert any(issue.endswith(":run_id") for issue in relabelled["identity_issues"])


def test_score_identity_rejects_duplicate_trace_and_invalid_metrics() -> None:
    environment_sha256, candidate, rows, comparisons, static = _gate_fixture()
    duplicate = json.loads(json.dumps(rows))
    duplicate[1]["source_hashes"]["codex.jsonl"] = duplicate[0]["trace_sha256"]
    duplicate[1]["trace_sha256"] = duplicate[0]["trace_sha256"]
    duplicate[1]["run_source_sha256"] = sha256_bytes(canonical_json(
        duplicate[1]["source_hashes"]
    ))
    batch_sources = [
        {"run_id": row["run_id"], "run_source_sha256": row["run_source_sha256"]}
        for row in sorted(duplicate, key=lambda item: item["run_id"])
    ]
    batch_hash = sha256_bytes(canonical_json(batch_sources))
    for row in duplicate:
        row["run_batch_sha256"] = batch_hash
    issues = _score_identity_issues(
        duplicate, environment_lock_sha256=environment_sha256,
    )
    assert "matrix:trace_sha256:duplicate" in issues

    for metric, invalid in (
        ("evidence_grounded_correctness", float("nan")),
        ("citation_precision", 1.01), ("tool_calls", -1),
        ("wall_time_seconds", float("inf")),
    ):
        changed = json.loads(json.dumps(rows))
        changed[0][metric] = invalid
        assert any(
            issue.endswith(f":metric:{metric}") for issue in _score_identity_issues(
                changed, environment_lock_sha256=environment_sha256,
            )
        )
    invalid_comparisons = json.loads(json.dumps(comparisons))
    invalid_comparisons[0]["ci_lower"] = float("nan")
    with pytest.raises(ValueError, match="invalid ci_lower"):
        evaluate_gates(
            rows, invalid_comparisons, static,
            environment_lock_sha256=environment_sha256,
            candidate_identity=candidate,
        )
    invalid_static = json.loads(json.dumps(static))
    invalid_static["metrics"]["recall_at_8"] = float("inf")
    with pytest.raises(ValueError, match="stale, unpassed"):
        _validate_static_identity(
            invalid_static, environment_lock_sha256=environment_sha256,
            candidate_identity=candidate,
        )


def test_bootstrap_recomputes_scores_and_static_report_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import eval.agent_ab.bootstrap as bootstrap

    _environment, _candidate, rows, _comparisons, static_report = _gate_fixture()
    scores_path = tmp_path / "scores.jsonl"
    score_bytes = render_score_rows(rows)
    scores_path.write_bytes(score_bytes)
    monkeypatch.setattr(bootstrap, "score_runs", lambda _runs, _work: rows)
    verified_rows, verified_bytes = verify_scores_against_raw(
        scores_path, tmp_path / "runs", tmp_path / "work",
    )
    assert verified_rows == rows
    assert verified_bytes == score_bytes

    forged = json.loads(json.dumps(rows))
    forged[0]["question_id"], forged[16]["question_id"] = (
        forged[16]["question_id"], forged[0]["question_id"],
    )
    scores_path.write_bytes(render_score_rows(forged))
    with pytest.raises(ValueError, match="raw-trace rescoring"):
        verify_scores_against_raw(scores_path, tmp_path / "runs", tmp_path / "work")

    static_path = tmp_path / "static-report.json"
    static_bytes = render_static_json(static_report)
    static_path.write_bytes(static_bytes)
    monkeypatch.setattr(
        bootstrap, "recompute_static_report",
        lambda _python, _work: static_bytes,
    )
    verified_static, verified_static_bytes = verify_static_against_candidate(
        static_path, "candidate-python", tmp_path / "work",
    )
    assert verified_static == static_report
    assert verified_static_bytes == static_bytes
    static_path.write_bytes(static_bytes.replace(b'"passed": true', b'"passed": false'))
    with pytest.raises(ValueError, match="candidate interpreter recheck"):
        verify_static_against_candidate(
            static_path, "candidate-python", tmp_path / "work",
        )


def test_evaluation_checkout_rejects_untracked_and_tracked_links(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(["git", "config", "user.name", "Eval Test"], cwd=repository, check=True)
    subprocess.run(
        ["git", "config", "user.email", "eval@example.invalid"],
        cwd=repository, check=True,
    )
    (repository / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repository, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "base"], cwd=repository, check=True)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository,
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    environment = {"hlsgraph_v03": {"revision": revision}}
    verify_evaluation_checkout(environment, repository)
    (repository / "injected.txt").write_text("untracked\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dirty"):
        verify_evaluation_checkout(environment, repository)
    (repository / "injected.txt").unlink()

    blob = subprocess.run(
        ["git", "hash-object", "-w", "--stdin"], cwd=repository,
        input=b"tracked.txt", capture_output=True, check=True,
    ).stdout.decode("ascii").strip()
    subprocess.run(
        ["git", "update-index", "--add", "--cacheinfo", f"120000,{blob},linked.txt"],
        cwd=repository, check=True,
    )
    (repository / "linked.txt").write_bytes(b"tracked.txt")
    subprocess.run(["git", "commit", "-q", "-m", "link"], cwd=repository, check=True)
    environment["hlsgraph_v03"]["revision"] = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository,
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    with pytest.raises(ValueError, match="symlink or submodule"):
        verify_evaluation_checkout(environment, repository)


def test_workspace_identity_covers_unexpected_readable_files(tmp_path: Path) -> None:
    corpus = next(
        item for item in load_corpus_lock()["corpora"] if item["id"] == "dataflow_gemm"
    )
    workspace = tmp_path / "native" / corpus["id"]
    for entry in corpus["files"]:
        destination = workspace / entry["destination"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / entry["source_path"], destination)
    (workspace / "EVAL_PROVENANCE.json").write_text(
        json.dumps(_provenance(corpus), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    before = workspace_identity(tmp_path, "native", corpus["id"])
    (workspace / "injected-answer.txt").write_text("hidden answer\n", encoding="utf-8")
    after = workspace_identity(tmp_path, "native", corpus["id"])
    assert after["workspace_tree_sha256"] != before["workspace_tree_sha256"]
    assert after["workspace_identity_sha256"] != before["workspace_identity_sha256"]


def test_wheel_identity_binds_installed_payload_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    wheel = tmp_path / "hlsgraph-0.3.0-py3-none-any.whl"
    package_bytes = b'__version__ = "0.3.0"\n'
    metadata_bytes = b"Name: hlsgraph\nVersion: 0.3.0\n"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("hlsgraph/__init__.py", package_bytes)
        archive.writestr("hlsgraph-0.3.0.dist-info/METADATA", metadata_bytes)
        archive.writestr("hlsgraph-0.3.0.dist-info/RECORD", "")
    site = tmp_path / "site"
    (site / "hlsgraph").mkdir(parents=True)
    (site / "hlsgraph-0.3.0.dist-info").mkdir()
    (site / "hlsgraph/__init__.py").write_bytes(package_bytes)
    (site / "hlsgraph-0.3.0.dist-info/METADATA").write_bytes(metadata_bytes)
    record = "hlsgraph/__init__.py,,\nhlsgraph-0.3.0.dist-info/METADATA,,\n"

    class FakeDistribution:
        version = "0.3.0"
        files = [
            PurePosixPath("hlsgraph/__init__.py"),
            PurePosixPath("hlsgraph-0.3.0.dist-info/METADATA"),
            PurePosixPath("../../Scripts/hlsgraph-mcp.exe"),
        ]

        def locate_file(self, item: object) -> Path:
            return site / str(item)

        def read_text(self, name: str) -> str | None:
            if name == "RECORD":
                return record
            return None

    import eval.agent_ab.wheel_identity as identity
    monkeypatch.setattr(identity.importlib.metadata, "distribution", lambda _name: FakeDistribution())
    monkeypatch.setattr(identity.importlib, "import_module", lambda _name: SimpleNamespace(
        __version__="0.3.0", __file__=str(site / "hlsgraph/__init__.py"),
    ))
    source_repo = tmp_path / "source"
    (source_repo / "src/hlsgraph").mkdir(parents=True)
    (source_repo / "src/hlsgraph/__init__.py").write_bytes(package_bytes)

    def git(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args], cwd=source_repo, capture_output=True, text=True, check=True,
        )
        return completed.stdout.strip()

    git("init")
    git("config", "user.email", "eval-fixture@example.invalid")
    git("config", "user.name", "Eval Fixture")
    git("add", "src/hlsgraph/__init__.py")
    git("commit", "-m", "fixture")

    report = inspect_installation(wheel, "0.3.0", source_repo)
    assert report["verified"] is True
    assert report["installed_payload_sha256"] == report["wheel_payload_sha256"]
    assert report["source_package_sha256"] == report["wheel_package_sha256"]
    assert report["source_revision"] == git("rev-parse", "HEAD")
    wheel_with_generated_extra = tmp_path / "hlsgraph-0.3.0-extra-py3-none-any.whl"
    with zipfile.ZipFile(wheel_with_generated_extra, "w") as archive:
        archive.writestr("hlsgraph/__init__.py", package_bytes)
        archive.writestr("hlsgraph/__pycache__/injected.pyc", b"not-source")
        archive.writestr("hlsgraph-0.3.0.dist-info/METADATA", metadata_bytes)
        archive.writestr("hlsgraph-0.3.0.dist-info/RECORD", "")
    with pytest.raises(RuntimeError, match="wheel/source package set mismatch"):
        inspect_installation(wheel_with_generated_extra, "0.3.0", source_repo)
    (site / "hlsgraph/injected.py").write_text("SOURCE_TREE_POLLUTION = True\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="payload set mismatch"):
        inspect_installation(wheel, "0.3.0")
    (site / "hlsgraph/injected.py").unlink()
    (site / "hlsgraph/__init__.py").write_bytes(b"changed\n")
    with pytest.raises(RuntimeError, match="bytes mismatch"):
        inspect_installation(wheel, "0.3.0")
    (site / "hlsgraph/__init__.py").write_bytes(package_bytes)

    (source_repo / "src/hlsgraph/__init__.py").write_bytes(b"dirty\n")
    with pytest.raises(RuntimeError, match="must be clean"):
        inspect_installation(wheel, "0.3.0", source_repo)
    git("add", "src/hlsgraph/__init__.py")
    git("commit", "-m", "mismatch")
    with pytest.raises(RuntimeError, match="wheel/source package bytes mismatch"):
        inspect_installation(wheel, "0.3.0", source_repo)

    (source_repo / "src/hlsgraph/__init__.py").write_bytes(package_bytes)
    (source_repo / "src/hlsgraph/extra.py").write_text("EXTRA = True\n", encoding="utf-8")
    git("add", "src/hlsgraph")
    git("commit", "-m", "extra")
    with pytest.raises(RuntimeError, match="wheel/source package set mismatch"):
        inspect_installation(wheel, "0.3.0", source_repo)


def test_wheel_identity_rejects_link_members(tmp_path: Path) -> None:
    wheel = tmp_path / "hlsgraph-0.3.0-py3-none-any.whl"
    link = zipfile.ZipInfo("hlsgraph/__init__.py")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(link, "target.py")
        archive.writestr(
            "hlsgraph-0.3.0.dist-info/METADATA",
            "Name: hlsgraph\nVersion: 0.3.0\n",
        )
    with pytest.raises(RuntimeError, match="linked member"):
        inspect_installation(wheel, "0.3.0")


def test_question_stratified_paired_bootstrap_is_seeded() -> None:
    rows = []
    for question in ("q1", "q2", "q3"):
        for repetition in range(1, 5):
            rows.extend([
                {"question_id": question, "repetition": repetition, "arm": "native", "tool_calls": 6},
                {"question_id": question, "repetition": repetition, "arm": "hlsgraph-v03", "tool_calls": 2},
            ])
    first = stratified_paired_bootstrap(
        rows, candidate="hlsgraph-v03", baseline="native", metric="tool_calls",
        samples=250, seed=7,
    )
    second = stratified_paired_bootstrap(
        rows, candidate="hlsgraph-v03", baseline="native", metric="tool_calls",
        samples=250, seed=7,
    )
    assert first == second
    assert first["observed_delta"] == pytest.approx(4.0)
    assert first["ci_lower"] == pytest.approx(4.0)
    assert first["paired_cells"] == 12


def test_sanitizer_and_public_artifact_audit(tmp_path: Path) -> None:
    drive_path = "C:" + "\\" + "Users\\person\\work"
    token = "sk-" + "supersecretvalue123"
    raw = f"path={drive_path} ip=192.0.2.1 token={token}"
    clean = sanitize_text(raw)
    assert "person" not in clean
    assert "192.0.2.1" not in clean
    assert "sk-supersecret" not in clean
    path = tmp_path / "public.jsonl"
    path.write_text(json.dumps({"text": clean}) + "\n", encoding="utf-8")
    assert audit_public_artifacts([path])["passed"] is True


def test_sanitizer_removes_tool_payloads_and_all_absolute_path_forms(tmp_path: Path) -> None:
    source = tmp_path / "raw.jsonl"
    destination = tmp_path / "public.jsonl"
    forward_drive = "C" + ":" + "/" + "private/kernel.cpp"
    unc_path = "\\" + "\\" + "private-host" + "\\" + "share\\kernel.cpp"
    posix_path = "/" + "mnt/private/kernel.cpp"
    event = {
        "type": "item.completed",
        "item": {
            "id": "call-1", "type": "command_execution",
            "command": f"Get-Content {forward_drive}",
            "aggregated_output": f"int secret_source; // {posix_path} {unc_path}",
        },
    }
    assert all(
        raw_path not in sanitize_text(raw_path)
        for raw_path in (forward_drive, unc_path, posix_path)
    )
    source.write_text(json.dumps(event) + "\n", encoding="utf-8")
    assert audit_public_artifacts([source])["passed"] is False
    sanitize_file(source, destination)
    rendered = destination.read_text(encoding="utf-8")
    assert "secret_source" not in rendered
    assert forward_drive not in rendered
    assert unc_path not in rendered
    assert posix_path not in rendered
    assert audit_public_artifacts([destination])["passed"] is True


def test_scoring_rejects_changed_corpus_bytes_and_provenance(tmp_path: Path) -> None:
    corpus = next(item for item in load_corpus_lock()["corpora"] if item["id"] == "dataflow_gemm")
    workspace = tmp_path / "workspace"
    for entry in corpus["files"]:
        destination = workspace / entry["destination"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / entry["source_path"], destination)
    (workspace / "EVAL_PROVENANCE.json").write_text(
        json.dumps(_provenance(corpus), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    verify_workspace_corpus(workspace, "dataflow_gemm")
    with (workspace / "kernel.cpp").open("a", encoding="utf-8") as handle:
        handle.write("// changed after the run\n")
    with pytest.raises(ValueError, match="byte mismatch"):
        verify_workspace_corpus(workspace, "dataflow_gemm")
