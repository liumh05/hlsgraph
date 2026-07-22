from __future__ import annotations

import copy
import inspect
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pytest

from eval.agent_ab import runner as eval_runner
from eval.agent_ab import prepare as eval_prepare
from eval.agent_ab import common as eval_common
from eval.agent_ab.bootstrap import simultaneous_quality_noninferiority
from eval.agent_ab.common import (
    ARM_IDS, EvalManifestError, load_corpus_lock, load_manifest,
    load_questions, official_process_environment, require_official_ext4_directory,
    resolve_local_executable, runtime_tree_identity, sha256_file,
)
from eval.agent_ab.parse_trace import normalize_trace, validate_trace_policy
from eval.agent_ab.runner import (
    DISABLED_CODEX_FEATURES,
    _require_isolated_runs_root,
    _require_isolated_work_root,
    _runtime_identity_preflight,
    build_codex_command,
    build_run_plan,
    run_permission_canaries,
)
from eval.agent_ab.score import (
    _score_retrieval_audit, _validate_execution_contract,
    _validate_terminal_usage, canonical_answer, public_criterion_ids, score_answer,
)
from tests.agent_eval_runtime_support import (
    synthetic_retrieval_audit_placeholder, synthetic_runtime_identity,
)


ROOT = Path(__file__).resolve().parents[1]


def test_execute_record_uses_the_live_post_cell_workspace_validator() -> None:
    """Keep the success path from calling a removed boundary helper."""

    source = inspect.getsource(eval_runner.execute_record)
    assert "_work_root_directory_denies" not in source
    assert "_validate_workspace_inventory(" in source


def _message_event(answer: dict[str, object]) -> dict[str, object]:
    return {
        "type": "item.completed",
        "item": {
            "id": "message-1",
            "type": "agent_message",
            "text": json.dumps(answer),
        },
    }


def _minimal_answer() -> dict[str, object]:
    return {
        "answer": "c01: bounded fact",
        "claims": [{
            "id": "claim-1",
            "criterion_id": "c01",
            "statement": "bounded fact",
            "truth_plane": "design_fact",
            "stage": "source",
            "authority": "static source",
            "evidence": [{"path": "kernel.cpp", "line_start": 1, "line_end": 1}],
        }],
        "uncertainties": [],
    }


def test_codex_command_uses_strict_named_permissions_and_no_sandbox_mode(
    tmp_path: Path,
) -> None:
    work_root = tmp_path / "work"
    runs_root = tmp_path / "runs"
    record = {
        "run_id": "x",
        "question_id": "dg-architecture-flow",
        "corpus_id": "dataflow_gemm",
        "category": "architecture",
        "arm": "hlsgraph-v03",
        "repetition": 1,
        "execution_index": 1,
    }
    command = build_codex_command(
        record,
        work_root=work_root,
        runs_root=runs_root,
        codex_command="codex",
        v02_python="py02",
        v03_python="py03",
        codegraph_command="codegraph",
    )
    assert "--strict-config" in command
    assert "--sandbox" not in command
    assert "standalone_web_search" in DISABLED_CODEX_FEATURES
    assert "in_app_browser" in DISABLED_CODEX_FEATURES
    joined = "\n".join(command)
    assert 'web_search="disabled"' in joined
    assert 'default_permissions="hlsgraph_eval"' in joined
    assert 'permissions.hlsgraph_eval.extends=' not in joined
    assert "permissions.hlsgraph_eval.network.enabled=false" in joined
    assert "permissions.hlsgraph_eval.filesystem={" in joined
    assert '":minimal"="read"' in joined
    assert runs_root.resolve().as_posix() not in joined
    assert "--sandbox" not in joined
    with pytest.raises(RuntimeError, match="runs root must be disjoint"):
        build_codex_command(
            record, work_root=work_root, runs_root=work_root / "runs",
            codex_command="codex", v02_python="py02", v03_python="py03",
            codegraph_command="codegraph",
        )


def test_run_plan_is_blocked_counterbalanced_and_binds_execution_order() -> None:
    plan = build_run_plan()
    assert [item["execution_index"] for item in plan] == list(range(1, 193))
    blocks = [plan[index:index + len(ARM_IDS)] for index in range(0, len(plan), len(ARM_IDS))]
    positions: dict[str, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    for block in blocks:
        assert len({item["question_id"] for item in block}) == 1
        assert len({item["repetition"] for item in block}) == 1
        assert {item["arm"] for item in block} == set(ARM_IDS)
        question_id = str(block[0]["question_id"])
        for position, item in enumerate(block):
            positions[question_id][str(item["arm"])].add(position)
    assert all(
        arm_positions == set(range(len(ARM_IDS)))
        for question in positions.values()
        for arm_positions in question.values()
    )


def test_permission_profile_is_default_deny_and_rejects_unknown_inventory(
    tmp_path: Path,
) -> None:
    work_root = tmp_path / "isolated-work"
    for arm in ARM_IDS:
        for corpus in load_corpus_lock()["corpora"]:
            (work_root / arm / corpus["id"]).mkdir(parents=True)
    (work_root / "_cache").mkdir()
    (work_root / "environment.lock.json").write_text("{}\n", encoding="utf-8")
    (work_root / "materialization.json").write_text("{}\n", encoding="utf-8")
    runtime = synthetic_runtime_identity(public_repository=ROOT, work_root=work_root)
    runs_root = tmp_path / "isolated-runs"
    record = {
        "arm": "native", "corpus_id": "dataflow_gemm",
        "question_id": "dg-architecture-flow", "run_id": "fixture",
        "category": "architecture", "repetition": 1, "execution_index": 1,
        "timeout_seconds": 900,
    }
    command = build_codex_command(
        record, work_root=work_root, codex_command=runtime["codex"]["path"],
        runs_root=runs_root,
        v02_python="py02", v03_python="py03", codegraph_command="codegraph",
        sandbox_boundary=runtime["sandbox_boundary"],
    )
    filesystem = next(
        item for item in command
        if item.startswith("permissions.hlsgraph_eval.filesystem=")
    )
    assert '":minimal"="read"' in filesystem
    assert (work_root / "codegraph").as_posix() not in filesystem
    assert (work_root / "native" / "cordic").as_posix() not in filesystem
    assert f'"{(work_root / "native" / "dataflow_gemm").as_posix()}"="read"' in filesystem
    assert f'"{runtime["codex"]["path"]}"="read"' in filesystem
    assert runs_root.resolve().as_posix() not in filesystem
    assert '="deny"' not in filesystem
    assert ROOT.as_posix() not in filesystem
    assert runtime["sandbox_boundary"]["codex_home"] not in filesystem
    codegraph_record = {**record, "arm": "codegraph"}
    codegraph_command = build_codex_command(
        codegraph_record, work_root=work_root,
        runs_root=runs_root,
        codex_command=runtime["codex"]["path"],
        v02_python="py02", v03_python="py03",
        codegraph_command=(
            f'"{runtime["node"]["path"]}" '
            f'"{runtime["codegraph_entrypoint"]["path"]}"'
        ),
        sandbox_boundary=runtime["sandbox_boundary"],
    )
    codegraph_filesystem = next(
        item for item in codegraph_command
        if item.startswith("permissions.hlsgraph_eval.filesystem=")
    )
    for entry in runtime["sandbox_boundary"]["runtime_allow_roots"]["codegraph"]:
        assert f'"{entry["path"]}"="read"' in codegraph_filesystem
    assert runtime["python"]["hlsgraph_v03"]["path"] not in codegraph_filesystem
    cell = {
        **codegraph_record, "command_argv": codegraph_command,
        "retrieval_audit": {
            "schema_version": "hlsgraph.agent_eval.retrieval_audit.v1",
            "status": "not_applicable",
        },
    }
    cell["run_contract_sha256"] = eval_runner.sha256_bytes(
        eval_runner.canonical_json(cell)
    )
    environment = {"runtime_identity": runtime}
    _validate_execution_contract(
        cell, work_root, environment, runs_root=runs_root,
    )
    _validate_execution_contract(
        cell, work_root, environment, runs_root=tmp_path / "other-runs",
    )

    def rehash(changed_cell: dict[str, object]) -> None:
        changed_cell["run_contract_sha256"] = eval_runner.sha256_bytes(
            eval_runner.canonical_json({
                key: value for key, value in changed_cell.items()
                if key != "run_contract_sha256"
            })
        )

    wrong_codex = copy.deepcopy(cell)
    wrong_codex["command_argv"][0] = str(tmp_path / "other-codex")
    rehash(wrong_codex)
    with pytest.raises(ValueError, match="prepared Codex executable"):
        _validate_execution_contract(
            wrong_codex, work_root, environment, runs_root=runs_root,
        )

    wrong_args = copy.deepcopy(cell)
    args_index = next(
        index for index, value in enumerate(wrong_args["command_argv"])
        if value.startswith("mcp_servers.codegraph.args=")
    )
    wrong_args["command_argv"][args_index] = (
        'mcp_servers.codegraph.args=["different.js","serve","--mcp"]'
    )
    rehash(wrong_args)
    with pytest.raises(ValueError, match="runtime or args"):
        _validate_execution_contract(
            wrong_args, work_root, environment, runs_root=runs_root,
        )

    wrong_timeout = copy.deepcopy(cell)
    wrong_timeout["timeout_seconds"] = 901
    rehash(wrong_timeout)
    with pytest.raises(ValueError, match="frozen timeout"):
        _validate_execution_contract(
            wrong_timeout, work_root, environment, runs_root=runs_root,
        )

    changed = copy.deepcopy(cell)
    telemetry_index = next(
        index for index, value in enumerate(changed["command_argv"])
        if value.startswith("mcp_servers.codegraph.args=")
    )
    telemetry_args = json.loads(
        changed["command_argv"][telemetry_index].split("=", 1)[1]
    )
    value_index = telemetry_args.index("CODEGRAPH_TELEMETRY") + 1
    telemetry_args[value_index] = "1"
    changed["command_argv"][telemetry_index] = (
        "mcp_servers.codegraph.args=" + json.dumps(telemetry_args)
    )
    rehash(changed)
    with pytest.raises(ValueError, match="runtime or args"):
        _validate_execution_contract(
            changed, work_root, environment, runs_root=runs_root,
        )

    changed_minimal_rule = copy.deepcopy(cell)
    filesystem_index = next(
        index for index, value in enumerate(changed_minimal_rule["command_argv"])
        if value.startswith("permissions.hlsgraph_eval.filesystem=")
    )
    changed_minimal_rule["command_argv"][filesystem_index] = (
        changed_minimal_rule["command_argv"][filesystem_index].replace(
            '":minimal"="read"', '":minimal"="deny"',
        )
    )
    rehash(changed_minimal_rule)
    with pytest.raises(ValueError, match="permission profile"):
        _validate_execution_contract(
            changed_minimal_rule, work_root, environment, runs_root=runs_root,
        )

    v03_record = {**record, "arm": "hlsgraph-v03"}
    v03_workspace = work_root / "hlsgraph-v03" / "dataflow_gemm"
    synthetic_retrieval_audit_placeholder(v03_workspace)
    audit_batch = "a" * 32
    audit_descriptor = eval_runner._materialize_retrieval_audit_overlay(
        work_root, batch_id=audit_batch, run_id=v03_record["run_id"],
    )
    audit_overlay = eval_runner._audit_overlay_from_descriptor(
        work_root, v03_workspace, audit_descriptor,
        batch_id=audit_batch, run_id=v03_record["run_id"], require_empty=True,
    )
    v03_command = build_codex_command(
        v03_record, work_root=work_root,
        runs_root=runs_root,
        codex_command=runtime["codex"]["path"],
        v02_python=runtime["python"]["hlsgraph_v02"]["path"],
        v03_python=runtime["python"]["hlsgraph_v03"]["path"],
        codegraph_command="codegraph",
        sandbox_boundary=runtime["sandbox_boundary"],
        audit_overlay=audit_overlay,
    )
    v03_cell = {
        **v03_record, "command_argv": v03_command,
        "retrieval_audit": audit_descriptor,
    }
    v03_filesystem = next(
        item for item in v03_command
        if item.startswith("permissions.hlsgraph_eval.filesystem=")
    )
    for entry in runtime["sandbox_boundary"]["runtime_allow_roots"]["hlsgraph-v03"]:
        assert f'"{entry["path"]}"="read"' in v03_filesystem
    for entry in runtime["sandbox_boundary"]["runtime_allow_roots"]["hlsgraph-v02"]:
        if entry["path"] != runtime["codex"]["path"]:
            assert entry["path"] not in v03_filesystem
    rehash(v03_cell)
    _validate_execution_contract(
        v03_cell, work_root, environment, runs_root=runs_root,
    )
    wrong_hls_args = copy.deepcopy(v03_cell)
    hls_args_index = next(
        index for index, value in enumerate(wrong_hls_args["command_argv"])
        if value.startswith("mcp_servers.hlsgraph.args=")
    )
    wrong_hls_args["command_argv"][hls_args_index] = (
        'mcp_servers.hlsgraph.args=["-m","hlsgraph.mcp.server","other-workspace"]'
    )
    rehash(wrong_hls_args)
    with pytest.raises(ValueError, match="HLSGraph runtime or args"):
        _validate_execution_contract(
            wrong_hls_args, work_root, environment, runs_root=runs_root,
        )

    (work_root / "unexpected-private").mkdir()
    with pytest.raises(RuntimeError, match="unexpected top-level"):
        build_codex_command(
            record, work_root=work_root, codex_command="codex",
            runs_root=runs_root,
            v02_python="py02", v03_python="py03", codegraph_command="codegraph",
            sandbox_boundary=runtime["sandbox_boundary"],
        )


def test_trace_policy_rejects_web_network_escape_gold_and_wrong_mcp(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    answer = _minimal_answer()
    allowed = [
        {"type": "thread.started", "thread_id": "thread-1"},
        {
            "type": "item.completed",
            "item": {
                "id": "tool-1",
                "type": "command_execution",
                "command": "rg DATAFLOW kernel.cpp",
            },
        },
        _message_event(answer),
    ]
    report = validate_trace_policy(allowed, arm="native", workspace=workspace)
    assert report["passed"] is True

    attacks = [
        {
            "type": "item.completed",
            "item": {"id": "web-1", "type": "web_search", "name": "web_search"},
        },
        {
            "type": "item.completed",
            "item": {
                "id": "net-1",
                "type": "command_execution",
                "command": "curl https://example.com",
            },
        },
        {
            "type": "item.completed",
            "item": {
                "id": "escape-1",
                "type": "command_execution",
                "command": "Get-Content ../../questions.jsonl",
            },
        },
        {
            "type": "item.completed",
            "item": {
                "id": "mcp-1",
                "type": "mcp_tool_call",
                "name": "hlsgraph.explore",
                "arguments": {"query": "flow"},
            },
        },
        {
            "type": "item.completed",
            "item": {
                "id": "gold-1",
                "type": "command_execution",
                "command": "rg criterion kernel.cpp",
                "aggregated_output": '"evidence_selectors": [{"path": "kernel.cpp"}]',
            },
        },
    ]
    for attack in attacks:
        with pytest.raises(ValueError, match="trace policy"):
            validate_trace_policy(
                [attack, _message_event(answer)], arm="native", workspace=workspace,
            )

    with pytest.raises(ValueError, match="boundary canary"):
        validate_trace_policy(
            [{
                "type": "item.completed",
                "item": {
                    "id": "canary-1",
                    "type": "command_execution",
                    "command": "rg fact kernel.cpp",
                    "aggregated_output": "secret-boundary-token",
                },
            }, _message_event(answer)],
            arm="native",
            workspace=workspace,
            boundary_canary=b"secret-boundary-token",
        )


def test_graph_trace_requires_exact_treatment_mcp_as_first_call(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with pytest.raises(ValueError, match="missing-treatment-mcp"):
        validate_trace_policy([], arm="hlsgraph-v03", workspace=workspace)

    shell_first = [{
        "type": "item.completed",
        "item": {"id": "shell", "type": "command_execution", "command": "rg flow kernel.cpp"},
    }, {
        "type": "item.completed",
        "item": {
            "id": "mcp", "type": "mcp_tool_call", "server": "hlsgraph",
            "name": "explore", "arguments": {"query": "flow"},
        },
    }]
    with pytest.raises(ValueError, match="first-call-not-treatment-mcp"):
        validate_trace_policy(shell_first, arm="hlsgraph-v03", workspace=workspace)

    failed_first = [{
        "type": "item.completed",
        "item": {
            "id": "mcp", "type": "mcp_tool_call", "server": "hlsgraph",
            "name": "explore", "status": "failed", "error": "index unavailable",
            "arguments": {
                "query": "flow", "include_private_snippets": True,
                "include_predictions": False,
            },
        },
    }]
    report = validate_trace_policy(
        failed_first, arm="hlsgraph-v03", workspace=workspace,
    )
    assert report["treatment_mcp_calls"] == 1
    assert report["first_call_treatment_mcp"] is True
    assert report["treatment_mcp_first_outcome"] == "failed"

    incomplete_first = [{
        "type": "item.started",
        "item": {
            "id": "mcp", "type": "mcp_tool_call", "server": "codegraph",
            "name": "codegraph_explore",
        },
    }]
    report = validate_trace_policy(
        incomplete_first, arm="codegraph", workspace=workspace,
    )
    assert report["treatment_mcp_first_outcome"] == "incomplete"


def test_terminal_usage_is_required_and_positive() -> None:
    answer = _minimal_answer()
    nonterminal = normalize_trace([{
        "type": "item.completed", "usage": {
            "input_tokens": 2, "cached_input_tokens": 1,
            "output_tokens": 1, "total_tokens": 3,
        }, "item": {"id": "msg", "type": "agent_message", "text": json.dumps(answer)},
    }])
    assert nonterminal["usage"] == {}
    with pytest.raises(ValueError, match="lacks input_tokens"):
        _validate_terminal_usage(nonterminal["usage"])
    with pytest.raises(ValueError, match="must be positive"):
        _validate_terminal_usage({
            "input_tokens": 0, "cached_input_tokens": 0,
            "output_tokens": 0, "total_tokens": 0,
        })
    with pytest.raises(ValueError, match="lacks cached_input_tokens"):
        _validate_terminal_usage({
            "input_tokens": 2, "output_tokens": 1, "total_tokens": 3,
        })
    with pytest.raises(ValueError, match="exceeds input tokens"):
        _validate_terminal_usage({
            "input_tokens": 2, "cached_input_tokens": 3,
            "output_tokens": 1, "total_tokens": 3,
        })
    assert _validate_terminal_usage({
        "input_tokens": 2, "cached_input_tokens": 1,
        "output_tokens": 1, "total_tokens": 3,
    }) == {
        "input_tokens": 2, "cached_input_tokens": 1,
        "output_tokens": 1, "total_tokens": 3,
    }


def test_terminal_usage_is_never_synthesized_across_events() -> None:
    events = [
        _message_event(_minimal_answer()),
        {
            "type": "turn.completed",
            "usage": {"input_tokens": 2, "cached_input_tokens": 1},
        },
        {
            "type": "response.completed",
            "response": {"usage": {"output_tokens": 1, "total_tokens": 3}},
        },
    ]
    usage = normalize_trace(events)["usage"]
    assert usage == {"output_tokens": 1, "total_tokens": 3}
    with pytest.raises(ValueError, match="lacks input_tokens"):
        _validate_terminal_usage(usage)


def test_terminal_usage_rejects_conflicting_objects_in_one_event() -> None:
    complete = {
        "input_tokens": 2, "cached_input_tokens": 1,
        "output_tokens": 1, "total_tokens": 3,
    }
    with pytest.raises(ValueError, match="conflicting usage objects"):
        normalize_trace([
            _message_event(_minimal_answer()),
            {
                "type": "response.completed",
                "usage": complete,
                "response": {"usage": {**complete, "output_tokens": 2}},
            },
        ])


def test_mcp_source_request_without_bound_output_does_not_count_as_file_read() -> None:
    events = [
        {
            "type": "item.completed",
            "item": {
                "id": "mcp-1",
                "type": "mcp_tool_call",
                "name": "hlsgraph.explore",
                "arguments": {
                    "query": "flow", "include_private_snippets": True,
                    "include_predictions": False,
                },
            },
        },
        _message_event(_minimal_answer()),
    ]
    normalized = normalize_trace(events)
    assert normalized["tool_calls"] == 1
    assert normalized["file_reads"] == 0
    assert normalized["private_snippet_calls"] == []
    assert normalized["file_read_semantics"] == "source_access_tool_calls"


def test_private_source_read_requires_matching_tool_output_and_audit_record() -> None:
    workspace = ROOT / "examples/dataflow_gemm"
    source = (workspace / "kernel.cpp").read_text(encoding="utf-8").splitlines()
    excerpt = source[6]
    excerpt_sha256 = eval_runner.sha256_bytes(excerpt.encode("utf-8"))
    content_sha256 = next(
        item["sha256"] for corpus in load_corpus_lock()["corpora"]
        if corpus["id"] == "dataflow_gemm" for item in corpus["files"]
        if item["destination"] == "kernel.cpp"
    )
    event = {
        "type": "item.completed",
        "item": {
            "id": "mcp-1", "type": "mcp_tool_call", "server": "hlsgraph",
            "name": "explore",
            "arguments": {
                "query": "values", "include_private_snippets": True,
                "include_predictions": False,
            },
            "result": {"structuredContent": {
                "trace": {
                    "private_snippets_requested": True,
                    "private_snippets_returned": True,
                },
                "facts": [{
                    "record_kind": "source_snippet",
                    "data": {
                        "artifact_sha256": content_sha256,
                        "anchor": {"start_line": 7, "end_line": 7},
                        "private_excerpt": excerpt,
                        "excerpt_sha256": excerpt_sha256,
                        "authorization": "project_bounded",
                    },
                }],
            }},
        },
    }
    normalized = normalize_trace([event, _message_event(_minimal_answer())])
    assert normalized["file_reads"] == 0
    assert len(normalized["private_snippet_calls"]) == 1
    audit_record = {
        "content_sha256": content_sha256,
        "anchor": {"kind": "source_line", "start_line": 7, "end_line": 7},
        "result": "returned", "byte_count": len(excerpt.encode("utf-8")),
    }
    audit = (
        json.dumps(audit_record, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")
    receipt = eval_runner._retrieval_audit_receipt(audit)
    scored = _score_retrieval_audit(
        audit, {"retrieval_audit": receipt}, normalized, arm="hlsgraph-v03",
        workspace=workspace, corpus_id="dataflow_gemm",
    )
    assert scored["source_access_calls"] == 1
    mismatched = audit.replace(content_sha256.encode("ascii"), b"b" * 64)
    mismatched_receipt = eval_runner._retrieval_audit_receipt(mismatched)
    with pytest.raises(ValueError, match="not one-to-one"):
        _score_retrieval_audit(
            mismatched, {"retrieval_audit": mismatched_receipt}, normalized,
            arm="hlsgraph-v03", workspace=workspace, corpus_id="dataflow_gemm",
        )
    no_output = copy.deepcopy(normalized)
    no_output["private_snippet_calls"] = []
    with pytest.raises(ValueError, match="not one-to-one"):
        _score_retrieval_audit(
            audit, {"retrieval_audit": receipt}, no_output, arm="hlsgraph-v03",
            workspace=workspace, corpus_id="dataflow_gemm",
        )
    empty_receipt = eval_runner._retrieval_audit_receipt(b"")
    with pytest.raises(ValueError, match="not one-to-one"):
        _score_retrieval_audit(
            b"", {"retrieval_audit": empty_receipt}, normalized,
            arm="hlsgraph-v03", workspace=workspace, corpus_id="dataflow_gemm",
        )
    duplicate = audit + audit
    with pytest.raises(ValueError, match="duplicate access identities"):
        _score_retrieval_audit(
            duplicate, {"retrieval_audit": eval_runner._retrieval_audit_receipt(duplicate)},
            normalized, arm="hlsgraph-v03", workspace=workspace,
            corpus_id="dataflow_gemm",
        )
    forged = copy.deepcopy(normalized)
    forged["private_snippet_calls"][0]["receipts"][0]["excerpt_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="frozen corpus bytes"):
        _score_retrieval_audit(
            audit, {"retrieval_audit": receipt}, forged, arm="hlsgraph-v03",
            workspace=workspace, corpus_id="dataflow_gemm",
        )


def _bound_architecture_answer(question: dict[str, object]) -> dict[str, object]:
    criterion_ids = public_criterion_ids(question)
    claims = [
        {
            "id": "flow",
            "criterion_id": criterion_ids["flow"],
            "statement": "load sends values to compute, which sends results to store.",
            "truth_plane": "design_fact",
            "stage": "source",
            "authority": "static source",
            "evidence": [{"path": "kernel.cpp", "line_start": 31, "line_end": 39}],
        },
        {
            "id": "values",
            "criterion_id": criterion_ids["values-depth"],
            "statement": "values has declared depth 8.",
            "truth_plane": "design_fact",
            "stage": "source",
            "authority": "static source",
            "evidence": [{"path": "kernel.cpp", "line_start": 33, "line_end": 36}],
        },
        {
            "id": "results",
            "criterion_id": criterion_ids["results-depth"],
            "statement": "results has declared depth 16.",
            "truth_plane": "design_fact",
            "stage": "source",
            "authority": "static source",
            "evidence": [{"path": "kernel.cpp", "line_start": 33, "line_end": 36}],
        },
        {
            "id": "boundary",
            "criterion_id": criterion_ids["topology-boundary"],
            "statement": "A function call graph is insufficient and cannot alone prove hardware topology.",
            "truth_plane": "knowledge_guidance",
            "stage": None,
            "authority": "evaluation invariant",
            "evidence": [{"path": "kernel.cpp", "line_start": 31, "line_end": 39}],
        },
    ]
    return {"answer": canonical_answer(question, claims), "claims": claims, "uncertainties": []}


def test_strict_answer_is_canonical_criterion_bound_and_closed_world(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    shutil.copyfile(ROOT / "examples/dataflow_gemm/kernel.cpp", workspace / "kernel.cpp")
    question = next(item for item in load_questions() if item["id"] == "dg-architecture-flow")
    answer = _bound_architecture_answer(question)
    score = score_answer(question, answer, workspace, require_bound=True)
    assert score["evidence_grounded_correctness"] == pytest.approx(1.0)
    assert score["fabricated_truth_count"] == 0

    fabricated_answer = copy.deepcopy(answer)
    fabricated_answer["answer"] += "\nPost-route WNS is +999 ns and every real-tool gate passed."
    with pytest.raises(ValueError, match="canonical"):
        score_answer(question, fabricated_answer, workspace, require_bound=True)

    extra = copy.deepcopy(answer)
    extra["claims"].append({
        "id": "cheese",
        "criterion_id": "c99",
        "statement": "The Moon is made of cheese.",
        "truth_plane": "design_fact",
        "stage": "source",
        "authority": "static source",
        "evidence": [{"path": "kernel.cpp", "line_start": 31, "line_end": 39}],
    })
    extra["answer"] = canonical_answer(question, extra["claims"], allow_unknown=True)
    with pytest.raises(ValueError, match="criterion"):
        score_answer(question, extra, workspace, require_bound=True)

    wrong_selector = copy.deepcopy(answer)
    wrong_selector["claims"][1]["evidence"] = [
        {"path": "kernel.cpp", "line_start": 1, "line_end": 1},
    ]
    wrong_selector["answer"] = canonical_answer(question, wrong_selector["claims"])
    wrong = score_answer(question, wrong_selector, workspace, require_bound=True)
    assert wrong["unsupported_claims"] == 1
    assert wrong["fabricated_truth_count"] >= 1


def test_forbidden_patterns_cover_canonical_answer_claims_and_uncertainties(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    shutil.copyfile(ROOT / "examples/dataflow_gemm/kernel.cpp", workspace / "kernel.cpp")
    question = next(item for item in load_questions() if item["id"] == "dg-architecture-flow")
    answer = _bound_architecture_answer(question)
    answer["uncertainties"] = ["real Vitis evidence"]
    scored = score_answer(question, answer, workspace, require_bound=True)
    assert any(item["id"].startswith("forbidden-free-text") for item in scored["violations"])


def test_work_root_must_be_disjoint_from_public_repository(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="outside"):
        _require_isolated_work_root(ROOT / "eval/agent_ab/work")
    assert _require_isolated_work_root(tmp_path / "isolated-work") == (
        tmp_path / "isolated-work"
    ).resolve()


@pytest.mark.skipif(os.name != "posix", reason="POSIX audit parent-chain contract")
def test_retrieval_audit_descriptor_rejects_parent_replacement_and_symlink(
    tmp_path: Path,
) -> None:
    work_root = tmp_path / "isolated-work"
    workspace = work_root / "hlsgraph-v03" / "dataflow_gemm"
    workspace.mkdir(parents=True)
    synthetic_retrieval_audit_placeholder(workspace)
    batch_id = "b" * 32
    descriptor = eval_runner._materialize_retrieval_audit_overlay(
        work_root, batch_id=batch_id, run_id="fixture",
    )
    original = work_root / descriptor["path"]
    parent = original.parent
    moved = parent.with_name("retrieval-audit-original")
    parent.rename(moved)
    parent.mkdir(mode=0o700)
    replacement = parent / original.name
    replacement.write_bytes(b"")
    replacement.chmod(0o600)
    with pytest.raises(RuntimeError, match="parent chain changed"):
        eval_runner._audit_overlay_from_descriptor(
            work_root, workspace, descriptor, batch_id=batch_id,
            run_id="fixture", require_empty=True,
        )
    shutil.rmtree(parent)
    outside = tmp_path / "outside-audit"
    outside.mkdir(mode=0o700)
    outside_file = outside / original.name
    outside_file.write_bytes(b"")
    outside_file.chmod(0o600)
    parent.symlink_to(outside, target_is_directory=True)
    with pytest.raises(RuntimeError, match="missing or linked"):
        eval_runner._audit_overlay_from_descriptor(
            work_root, workspace, descriptor, batch_id=batch_id,
            run_id="fixture", require_empty=True,
        )


@pytest.mark.skipif(os.name != "posix", reason="POSIX workspace-mode identity contract")
def test_workspace_identity_binds_private_directory_and_placeholder_modes(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    placeholder = synthetic_retrieval_audit_placeholder(workspace)
    baseline = eval_common._tree_identity(workspace)
    placeholder.parent.chmod(0o755)
    assert eval_common._tree_identity(workspace) != baseline
    placeholder.parent.chmod(0o700)
    assert eval_common._tree_identity(workspace) == baseline
    placeholder.chmod(0o644)
    assert eval_common._tree_identity(workspace) != baseline
    with pytest.raises(RuntimeError, match="placeholder must have mode 0600"):
        eval_runner._verify_audit_placeholder(workspace)


@pytest.mark.skipif(os.name != "posix", reason="POSIX venv symlink contract")
def test_posix_venv_python_symlink_keeps_lexical_path(tmp_path: Path) -> None:
    target = tmp_path / "python-target"
    target.write_bytes(b"fixture")
    launcher = tmp_path / "venv" / "bin" / "python"
    launcher.parent.mkdir(parents=True)
    launcher.symlink_to(target)
    assert resolve_local_executable(str(launcher)) == str(launcher.absolute())
    launcher.unlink()
    launcher.symlink_to(tmp_path / "missing-target")
    with pytest.raises(EvalManifestError, match="existing file"):
        resolve_local_executable(str(launcher))


@pytest.mark.skipif(
    os.name != "posix" or shutil.which("bwrap") is None,
    reason="POSIX locked launcher symlink-hop contract",
)
def test_contained_mcp_rejects_absolute_intermediate_venv_symlink(
    tmp_path: Path,
) -> None:
    work_root = tmp_path / "isolated-work"
    workspace = work_root / "hlsgraph-v03" / "dataflow_gemm"
    workspace.mkdir(parents=True)
    placeholder = synthetic_retrieval_audit_placeholder(workspace)
    (work_root / "_cache").mkdir()
    (work_root / "environment.lock.json").write_text("{}\n", encoding="utf-8")
    (work_root / "materialization.json").write_text("{}\n", encoding="utf-8")
    runtime = synthetic_runtime_identity(public_repository=ROOT, work_root=work_root)
    boundary = runtime["sandbox_boundary"]
    outside = tmp_path / "outer-venv" / "bin"
    outside.mkdir(parents=True)
    (outside / "python").symlink_to("/usr/bin/python3")
    locked_bin = tmp_path / "locked-runtime" / "bin"
    locked_bin.mkdir(parents=True)
    launcher_path = locked_bin / "python"
    launcher_path.symlink_to(outside / "python")
    boundary["runtime_allow_roots"]["hlsgraph-v03"] = [{
        "path": locked_bin.as_posix(), "kind": "tree",
        "algorithm": eval_common.SANDBOX_ALLOW_TREE_ALGORITHM,
        "sha256": eval_common.sandbox_allow_tree_identity(locked_bin),
    }]
    bwrap = Path(shutil.which("bwrap") or "")
    boundary["mcp_containment"]["launcher"] = {
        "path": bwrap.as_posix(), "filename": bwrap.name,
        "sha256": sha256_file(bwrap),
    }
    audit = tmp_path / "audit.jsonl"
    audit.write_bytes(b"")
    audit.chmod(0o600)
    with pytest.raises(RuntimeError, match="symlink hop escapes"):
        eval_runner._contained_mcp_command(
            arm="hlsgraph-v03", workspace=workspace,
            server_command=launcher_path.as_posix(), server_args=[], server_env={},
            sandbox_boundary=boundary,
            audit_overlay=(
                audit, placeholder, tuple(eval_runner._audit_parent_chain(audit)),
            ),
        )


def test_runs_root_requires_ext4_and_disjoint_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import eval.agent_ab.common as common

    candidate = tmp_path / "candidate-runs"
    candidate.mkdir()
    monkeypatch.setattr(common, "_filesystem_type", lambda _path: "ntfs")
    with pytest.raises(EvalManifestError, match="WSL ext4"):
        require_official_ext4_directory(candidate, "runs root")

    work_root = tmp_path / "work"
    work_root.mkdir()
    environment = {
        "runtime_identity": {
            "sandbox_boundary": {"deny_roots": [work_root.as_posix()]},
        },
    }
    monkeypatch.setattr(
        eval_runner, "require_official_ext4_directory",
        lambda path, _label, allow_missing=False: Path(os.path.abspath(path)),
    )
    with pytest.raises(RuntimeError, match="disjoint"):
        _require_isolated_runs_root(
            work_root / "runs", work_root=work_root,
            environment=environment, allow_missing=True,
        )
    outside = tmp_path / "outside-runs"
    assert _require_isolated_runs_root(
        outside, work_root=work_root, environment=environment, allow_missing=True,
    ) == outside.absolute()


def test_runtime_tree_identity_binds_regular_bytes_and_tree_membership(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    for root in (left, right):
        (root / "sub").mkdir(parents=True)
        (root / "sub" / "payload.bin").write_bytes(b"same bytes")
    baseline = runtime_tree_identity(left)
    assert runtime_tree_identity(right) == baseline
    (right / "sub" / "payload.bin").write_bytes(b"changed bytes")
    assert runtime_tree_identity(right) != baseline
    (right / "sub" / "payload.bin").write_bytes(b"same bytes")
    (right / "extra.bin").write_bytes(b"extra")
    assert runtime_tree_identity(right) != baseline


@pytest.mark.skipif(os.name != "posix", reason="POSIX mode identity contract")
def test_runtime_tree_identity_binds_file_and_directory_modes(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    subdirectory = root / "sub"
    subdirectory.mkdir(parents=True)
    payload = subdirectory / "payload.bin"
    payload.write_bytes(b"fixture")
    root.chmod(0o755)
    subdirectory.chmod(0o755)
    payload.chmod(0o644)
    baseline = runtime_tree_identity(root)
    payload.chmod(0o664)
    assert runtime_tree_identity(root) != baseline
    payload.chmod(0o644)
    subdirectory.chmod(0o775)
    assert runtime_tree_identity(root) != baseline
    subdirectory.chmod(0o755)
    root.chmod(0o775)
    assert runtime_tree_identity(root) != baseline


@pytest.mark.skipif(os.name != "posix", reason="POSIX symlink identity contract")
def test_runtime_tree_identity_rejects_unsafe_symlink_chains(tmp_path: Path) -> None:
    container = tmp_path / "container"
    nested_real = container / "real" / "nested"
    nested_real.mkdir(parents=True)
    (nested_real / "payload").write_bytes(b"fixture")
    (container / "linked-parent").symlink_to("real", target_is_directory=True)
    with pytest.raises(EvalManifestError, match="contains a link"):
        runtime_tree_identity(container / "linked-parent" / "nested")

    root = tmp_path / "tree"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    payload = root / "payload.bin"
    payload.write_bytes(b"fixture")
    (root / "internal-link").symlink_to("payload.bin")
    assert len(runtime_tree_identity(root)) == 64

    absolute = root / "absolute-link"
    absolute.symlink_to(payload)
    with pytest.raises(EvalManifestError, match="absolute"):
        runtime_tree_identity(root)
    absolute.unlink()

    dangling = root / "dangling-link"
    dangling.symlink_to("missing.bin")
    with pytest.raises(EvalManifestError, match="dangling or escapes"):
        runtime_tree_identity(root)
    dangling.unlink()

    (outside / "link-back").symlink_to(root, target_is_directory=True)
    lexical_escape = root / "lexical-escape"
    lexical_escape.symlink_to("../outside/link-back/payload.bin")
    with pytest.raises(EvalManifestError, match="dangling or escapes"):
        runtime_tree_identity(root)
    lexical_escape.unlink()

    gateway = root / "gateway"
    gateway.symlink_to("../outside/link-back", target_is_directory=True)
    (root / "via-gateway").symlink_to("gateway/payload.bin")
    with pytest.raises(EvalManifestError, match="dangling or escapes"):
        runtime_tree_identity(root)
    (root / "via-gateway").unlink()
    gateway.unlink()

    first = root / "cycle-a"
    second = root / "cycle-b"
    first.symlink_to("cycle-b")
    second.symlink_to("cycle-a")
    with pytest.raises(EvalManifestError, match="dangling or escapes"):
        runtime_tree_identity(root)


@pytest.mark.skipif(os.name != "posix", reason="POSIX special-file identity contract")
def test_runtime_tree_identity_rejects_special_files(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    root.mkdir()
    os.mkfifo(root / "fifo")
    with pytest.raises(EvalManifestError, match="special file"):
        runtime_tree_identity(root)


def test_runtime_identity_validator_rejects_overlapping_and_external_runtime_roots(
    tmp_path: Path,
) -> None:
    runtime = synthetic_runtime_identity(
        public_repository=ROOT, work_root=tmp_path / "work",
    )
    overlapping = copy.deepcopy(runtime)
    boundary = overlapping["sandbox_boundary"]
    boundary["runtime_root"] = (tmp_path / "work" / "runtime").absolute().as_posix()
    boundary["identity_sha256"] = eval_runner.sha256_bytes(eval_runner.canonical_json({
        key: value for key, value in boundary.items() if key != "identity_sha256"
    }))
    overlapping["identity_sha256"] = eval_runner.sha256_bytes(eval_runner.canonical_json({
        key: value for key, value in overlapping.items() if key != "identity_sha256"
    }))
    with pytest.raises(EvalManifestError, match="roots overlap"):
        eval_common._validate_runtime_identity(overlapping)

    external = copy.deepcopy(runtime)
    external_build = external["codegraph_build"]
    external_build["npm"]["path"] = (tmp_path / "external" / "npm-cli.js").as_posix()
    external_build["identity_sha256"] = eval_runner.sha256_bytes(
        eval_runner.canonical_json({
            key: value for key, value in external_build.items()
            if key != "identity_sha256"
        })
    )
    external["identity_sha256"] = eval_runner.sha256_bytes(eval_runner.canonical_json({
        key: value for key, value in external.items() if key != "identity_sha256"
    }))
    with pytest.raises(EvalManifestError, match="inconsistent with runtime"):
        eval_common._validate_runtime_identity(external)


def test_runtime_identity_rejects_broadened_or_relabelled_sandbox_allowlist(
    tmp_path: Path,
) -> None:
    runtime = synthetic_runtime_identity(
        public_repository=ROOT, work_root=tmp_path / "work",
    )

    def rehash(value: dict[str, object]) -> None:
        boundary = value["sandbox_boundary"]
        assert isinstance(boundary, dict)
        boundary["runtime_allow_roots_sha256"] = eval_runner.sha256_bytes(
            eval_runner.canonical_json(boundary["runtime_allow_roots"])
        )
        boundary["identity_sha256"] = eval_runner.sha256_bytes(
            eval_runner.canonical_json({
                key: item for key, item in boundary.items()
                if key != "identity_sha256"
            })
        )
        value["identity_sha256"] = eval_runner.sha256_bytes(
            eval_runner.canonical_json({
                key: item for key, item in value.items()
                if key != "identity_sha256"
            })
        )

    broadened = copy.deepcopy(runtime)
    boundary = broadened["sandbox_boundary"]
    boundary["runtime_allow_roots"]["native"].append({
        "path": boundary["runtime_root"], "kind": "tree",
        "algorithm": eval_common.SANDBOX_ALLOW_TREE_ALGORITHM,
        "sha256": "a" * 64,
    })
    boundary["runtime_allow_roots"]["native"].sort(key=lambda item: item["path"])
    rehash(broadened)
    with pytest.raises(EvalManifestError, match="broader than the exact locked runtimes"):
        eval_common._validate_runtime_identity(broadened)

    relabelled = copy.deepcopy(runtime)
    relabelled_boundary = relabelled["sandbox_boundary"]
    relabelled_boundary["runtime_allow_roots"]["native"][0]["sha256"] = "b" * 64
    rehash(relabelled)
    with pytest.raises(EvalManifestError, match="broader than the exact locked runtimes"):
        eval_common._validate_runtime_identity(relabelled)


@pytest.mark.skipif(os.name != "posix", reason="POSIX venv-link allowlist contract")
def test_sandbox_allow_tree_hashes_venv_links_but_rejects_external_links(
    tmp_path: Path,
) -> None:
    root = tmp_path / "venv"
    (root / "bin").mkdir(parents=True)
    (root / "lib").mkdir()
    (root / "pyvenv.cfg").write_text("version = 3.10\n", encoding="utf-8")
    (root / "bin" / "python").symlink_to("/usr/bin/python3")
    (root / "lib64").symlink_to("lib", target_is_directory=True)
    first = eval_common.sandbox_allow_tree_identity(root)
    (root / "pyvenv.cfg").write_text("version = 3.10.1\n", encoding="utf-8")
    assert eval_common.sandbox_allow_tree_identity(root) != first

    outside = tmp_path / "outside.txt"
    outside.write_text("secret\n", encoding="utf-8")
    (root / "external").symlink_to(outside)
    with pytest.raises(EvalManifestError, match="escapes the declared minimal roots"):
        eval_common.sandbox_allow_tree_identity(root)


def test_codegraph_capture_rejects_untracked_and_unhashed_ignored_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_root = tmp_path / "runtime"
    repository = runtime_root / "codegraph"
    entrypoint = repository / "dist" / "bin" / "codegraph.js"
    dependencies = repository / "node_modules" / "fixture" / "index.js"
    node = runtime_root / "node"
    npm_cli = runtime_root / "npm-cli.js"
    package_lock = repository / "package-lock.json"
    for path in (entrypoint, dependencies, node, npm_cli, package_lock):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    monkeypatch.setattr(
        eval_common, "_require_regular_ext4_path",
        lambda path, _label, **_kwargs: Path(path).absolute(),
    )
    status = {"value": "?? rogue.env"}
    ignored = {"value": ""}
    calls: list[tuple[str, ...]] = []

    def fake_git(_repository: Path, *args: str) -> str:
        calls.append(args)
        if args == ("rev-parse", "HEAD"):
            return load_manifest()["arms"][1]["revision"]
        if args == ("rev-parse", "HEAD^{tree}"):
            return load_manifest()["arms"][1]["build_identity"]["repository_tree"]
        if args[0] == "status":
            return status["value"]
        if args[0] == "ls-files":
            return ignored["value"]
        raise AssertionError(args)

    monkeypatch.setattr(eval_common, "_git_output", fake_git)
    kwargs = {
        "repository": repository, "runtime_root": runtime_root,
        "node": node, "npm_cli": npm_cli, "entrypoint": entrypoint,
    }
    with pytest.raises(EvalManifestError, match="untracked"):
        eval_common.capture_codegraph_build_identity(**kwargs)
    assert (
        "status", "--porcelain=v1", "--untracked-files=all",
    ) in calls

    status["value"] = ""
    ignored["value"] = ".env"
    with pytest.raises(EvalManifestError, match="outside the hashed build closure"):
        eval_common.capture_codegraph_build_identity(**kwargs)


def test_runtime_preflight_detects_codegraph_and_installed_payload_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    python = tmp_path / ("python.exe" if os.name == "nt" else "python")
    python.write_bytes(b"python-runtime")
    runtime = synthetic_runtime_identity(public_repository=ROOT, work_root=tmp_path / "work")
    node = Path(runtime["node"]["path"])
    entrypoint = Path(runtime["codegraph_entrypoint"]["path"])
    package_lock = Path(runtime["codegraph_build"]["package_lock"]["path"])
    npm_cli = Path(runtime["codegraph_build"]["npm"]["path"])
    runtime["node"].update({
        "path": node.as_posix(), "filename": node.name, "sha256": sha256_file(node),
    })
    runtime["codegraph_entrypoint"] = {
        "path": entrypoint.as_posix(), "filename": entrypoint.name,
        "sha256": sha256_file(entrypoint),
    }
    runtime["codegraph_build"]["node"] = dict(runtime["node"])
    runtime["codegraph_build"]["entrypoint"] = dict(runtime["codegraph_entrypoint"])
    runtime["codegraph_build"]["package_lock"]["sha256"] = sha256_file(package_lock)
    runtime["codegraph_build"]["npm"]["sha256"] = sha256_file(npm_cli)
    runtime["codegraph_build"]["identity_sha256"] = eval_runner.sha256_bytes(
        eval_runner.canonical_json({
            key: value for key, value in runtime["codegraph_build"].items()
            if key != "identity_sha256"
        })
    )
    runtime["identity_sha256"] = eval_runner.sha256_bytes(
        eval_runner.canonical_json({
            key: value for key, value in runtime.items()
            if key != "identity_sha256"
        })
    )
    environment = {
        "suite_asset_sha256": eval_runner.asset_digest(),
        "codegraph_revision": load_manifest()["arms"][1]["revision"],
        "codegraph_entrypoint": dict(runtime["codegraph_entrypoint"]),
        "codegraph_build": dict(runtime["codegraph_build"]),
        "runtime_identity": runtime,
        "identity_checks": [],
    }
    # The public manifest is frozen, so use its hash for the initial fixture
    # bytes and monkeypatch only the manifest loader used by this narrow test.
    manifest = load_manifest()
    manifest = copy.deepcopy(manifest)
    manifest["arms"][1]["entrypoint_sha256"] = sha256_file(entrypoint)
    monkeypatch.setattr(eval_runner, "load_manifest", lambda: manifest)
    command = f'"{node.as_posix()}" "{entrypoint.as_posix()}"'
    _runtime_identity_preflight(
        {"arm": "codegraph"}, environment, v02_python=python.as_posix(),
        v03_python=python.as_posix(), codegraph_command=command,
    )
    entrypoint.write_bytes(b"mutated-codegraph-runtime")
    with pytest.raises(RuntimeError, match="lightweight closure"):
        _runtime_identity_preflight(
            {"arm": "codegraph"}, environment, v02_python=python.as_posix(),
            v03_python=python.as_posix(), codegraph_command=command,
        )
    entrypoint.write_bytes(b"synthetic codegraph entrypoint\n")
    package_lock.write_bytes(b"mutated-package-lock")
    with pytest.raises(RuntimeError, match="lightweight closure"):
        _runtime_identity_preflight(
            {"arm": "codegraph"}, environment, v02_python=python.as_posix(),
            v03_python=python.as_posix(), codegraph_command=command,
        )
    package_lock.write_bytes(b"synthetic package lock\n")
    npm_cli.write_bytes(b"mutated-npm-cli")
    with pytest.raises(RuntimeError, match="lightweight closure"):
        _runtime_identity_preflight(
            {"arm": "codegraph"}, environment, v02_python=python.as_posix(),
            v03_python=python.as_posix(), codegraph_command=command,
        )
    npm_cli.write_bytes(b"synthetic non-executable unit-test fixture\n")

    payload_hash = "a" * 64
    runtime["python"]["hlsgraph_v02"].update({
        "path": python.as_posix(), "filename": python.name,
        "sha256": sha256_file(python),
    })
    environment["identity_checks"] = [{
        "kind": "verify-hlsgraph-wheel-installation", "arm": "hlsgraph-v02",
        "identity": {"version": "0.2.0", "installed_payload_sha256": payload_hash},
    }]
    monkeypatch.setattr(eval_runner, "official_process_environment", lambda: {})
    current_payload = {"value": payload_hash}

    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            [], 0, json.dumps({
                "verified": True,
                "installed_payload_sha256": current_payload["value"],
            }), "",
        )

    monkeypatch.setattr(eval_runner.subprocess, "run", fake_run)
    _runtime_identity_preflight(
        {"arm": "hlsgraph-v02"}, environment, v02_python=python.as_posix(),
        v03_python=python.as_posix(), codegraph_command=command,
    )
    current_payload["value"] = "b" * 64
    with pytest.raises(RuntimeError, match="prepared wheel"):
        _runtime_identity_preflight(
            {"arm": "hlsgraph-v02"}, environment, v02_python=python.as_posix(),
            v03_python=python.as_posix(), codegraph_command=command,
        )


def test_full_codegraph_preflight_binds_dist_and_dependencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = synthetic_runtime_identity(
        public_repository=ROOT, work_root=tmp_path / "work",
    )
    build = runtime["codegraph_build"]
    node = Path(runtime["node"]["path"])
    entrypoint = Path(runtime["codegraph_entrypoint"]["path"])
    package_lock = Path(build["package_lock"]["path"])
    npm_cli = Path(build["npm"]["path"])
    runtime["node"]["sha256"] = sha256_file(node)
    runtime["codegraph_entrypoint"]["sha256"] = sha256_file(entrypoint)
    build["node"] = dict(runtime["node"])
    build["entrypoint"] = dict(runtime["codegraph_entrypoint"])
    build["package_lock"]["sha256"] = sha256_file(package_lock)
    build["npm"]["sha256"] = sha256_file(npm_cli)
    build["identity_sha256"] = eval_runner.sha256_bytes(eval_runner.canonical_json({
        key: value for key, value in build.items() if key != "identity_sha256"
    }))
    runtime["identity_sha256"] = eval_runner.sha256_bytes(eval_runner.canonical_json({
        key: value for key, value in runtime.items() if key != "identity_sha256"
    }))
    environment = {
        "suite_asset_sha256": eval_runner.asset_digest(),
        "codegraph_revision": load_manifest()["arms"][1]["revision"],
        "codegraph_entrypoint": dict(runtime["codegraph_entrypoint"]),
        "codegraph_build": dict(build), "runtime_identity": runtime,
        "identity_checks": [],
    }
    manifest = copy.deepcopy(load_manifest())
    manifest["arms"][1]["entrypoint_sha256"] = sha256_file(entrypoint)
    monkeypatch.setattr(eval_runner, "load_manifest", lambda: manifest)
    dist = Path(build["dist"]["path"])
    dependencies = Path(build["dependencies"]["path"])
    baseline = (runtime_tree_identity(dist), runtime_tree_identity(dependencies))

    def capture(**_kwargs: object) -> dict[str, object]:
        if (runtime_tree_identity(dist), runtime_tree_identity(dependencies)) == baseline:
            return build
        changed = copy.deepcopy(build)
        changed["dist"]["tree_sha256"] = runtime_tree_identity(dist)
        changed["dependencies"]["tree_sha256"] = runtime_tree_identity(dependencies)
        return changed

    monkeypatch.setattr(eval_runner, "capture_codegraph_build_identity", capture)
    command = f'"{node.as_posix()}" "{entrypoint.as_posix()}"'
    _runtime_identity_preflight(
        {"arm": "codegraph"}, environment, v02_python="unused",
        v03_python="unused", codegraph_command=command, full_codegraph=True,
    )
    added_dist_file = dist / "extra.js"
    added_dist_file.write_bytes(b"mutated dist")
    with pytest.raises(RuntimeError, match="full runtime closure"):
        _runtime_identity_preflight(
            {"arm": "codegraph"}, environment, v02_python="unused",
            v03_python="unused", codegraph_command=command, full_codegraph=True,
        )
    added_dist_file.unlink()
    (dependencies / "fixture" / "index.js").write_bytes(b"mutated dependency")
    with pytest.raises(RuntimeError, match="full runtime closure"):
        _runtime_identity_preflight(
            {"arm": "codegraph"}, environment, v02_python="unused",
            v03_python="unused", codegraph_command=command, full_codegraph=True,
        )


def test_official_process_environment_uses_codex_home_but_never_auth_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("OPENAI_API_KEY", "CODEX_API_KEY", "CODEX_ACCESS_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "dedicated-codex-home"))
    monkeypatch.setenv("HLSGRAPH_TEST_SECRET", "must-not-pass")
    environment = official_process_environment()
    assert environment["CODEX_HOME"] == str(tmp_path / "dedicated-codex-home")
    assert "HLSGRAPH_TEST_SECRET" not in environment
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-test-token")
    with pytest.raises(EvalManifestError, match="forbids auth"):
        official_process_environment()
    monkeypatch.delenv("OPENAI_API_KEY")
    # Assemble the deliberately hostile credential form at runtime so the
    # public source tree itself never contains a credential-bearing URL.
    credential_proxy = "http://" + "user" + ":" + "secret" + "@proxy.invalid:8080"
    monkeypatch.setenv("HTTPS_PROXY", credential_proxy)
    with pytest.raises(EvalManifestError, match="credentials in forwarded proxy"):
        official_process_environment()


@pytest.mark.parametrize("module", [eval_prepare, eval_runner])
def test_execute_platform_no_go_precedes_any_output_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, module: object,
) -> None:
    output = tmp_path / "must-not-exist"

    def no_go() -> dict[str, str]:
        raise EvalManifestError("official agent evaluation is NO-GO outside WSL2")

    monkeypatch.setattr(module, "require_official_linux_wsl2", no_go)
    if module is eval_prepare:
        argv = [
            "--work-root", str(output), "--v02-python", "py02",
            "--v03-python", "py03", "--v02-wheel", str(tmp_path / "v02.whl"),
            "--v03-wheel", str(tmp_path / "v03.whl"), "--execute",
        ]
    else:
        argv = [
            "--work-root", str(output), "--runs-root", str(output / "runs"),
            "--v02-python", "py02", "--v03-python", "py03", "--execute",
        ]
    with pytest.raises(EvalManifestError, match="NO-GO"):
        module.main(argv)
    assert not output.exists()


def test_permission_canaries_pass_only_when_gold_and_socket_are_denied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    work_root = tmp_path / "isolated-work"
    corpus_ids = tuple(item["id"] for item in load_corpus_lock()["corpora"])
    for arm in ARM_IDS:
        for corpus_id in corpus_ids:
            workspace = work_root / arm / corpus_id
            workspace.mkdir(parents=True)
            (workspace / "EVAL_PROVENANCE.json").write_text("{}\n", encoding="utf-8")
    (work_root / "_cache").mkdir()
    (work_root / "environment.lock.json").write_text("{}\n", encoding="utf-8")
    (work_root / "materialization.json").write_text("{}\n", encoding="utf-8")
    runtime = synthetic_runtime_identity(
        public_repository=ROOT, work_root=work_root,
    )
    boundary = runtime["sandbox_boundary"]
    runs_root = tmp_path / "isolated-runs"
    runs_root.mkdir()
    Path(boundary["codex_home"]).mkdir()
    Path(boundary["drvfs_roots"][0]).mkdir()
    Path(boundary["home_canary_root"]).mkdir()
    environment = {"runtime_identity": runtime}
    monkeypatch.setattr(
        eval_runner, "_sandbox_canary_prefix",
        lambda _codex, workspace, **_kwargs: ["canary", str(workspace)],
    )
    escaped_kind = {"value": ""}

    def fake_canary(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        workspace = Path(command[1])
        arm = workspace.parent.name
        target_text = command[-1]
        if target_text.isdigit():
            return subprocess.CompletedProcess(command, 1, "", "network denied")
        target = Path(target_text)
        allowed = {
            workspace / "EVAL_PROVENANCE.json",
            *(Path(entry["path"]) for entry in boundary["runtime_allow_roots"][arm]),
        }
        escaped = (
            escaped_kind["value"] == "runs" and target.parent == runs_root
        ) or (
            escaped_kind["value"] == "gold" and target == eval_runner.HERE / "questions.jsonl"
        )
        return subprocess.CompletedProcess(
            command, 0 if target in allowed or escaped else 1,
            "allowed" if target in allowed or escaped else "",
            "" if target in allowed or escaped else "denied",
        )

    monkeypatch.setattr(eval_runner, "_run_canary_command", fake_canary)
    report = run_permission_canaries(
        codex_command="codex", work_root=work_root, runs_root=runs_root,
        environment=environment,
    )
    assert report["workspace_read"] == "pass"
    assert report["sibling_workspace_read"] == "denied"
    assert report["same_arm_sibling_read"] == "denied"
    assert report["other_arm_sibling_read"] == "denied"
    assert report["control_roots_read"] == "denied"
    assert report["runtime_allow_roots_read"] == "pass"
    assert report["undeclared_runtime_read"] == "denied"
    assert report["other_arm_runtime_read"] == "denied"
    assert report["runs_root_read"] == "denied"
    assert report["codex_home_read"] == "denied"
    assert report["user_home_read"] == "denied"
    assert report["external_private_read"] == "denied"
    assert report["drvfs_mount_reads"] == "denied"
    assert report["public_gold_read"] == "denied"
    assert report["network_socket"] == "denied"
    eval_runner._validate_permission_canary(
        report, boundary, runs_root=runs_root,
    )
    with pytest.raises(RuntimeError, match="stale, incomplete, or relabelled"):
        eval_runner._validate_permission_canary(
            report, boundary, runs_root=tmp_path / "other-runs",
        )
    relabelled = copy.deepcopy(report)
    relabelled["runs_root_sha256"] = "0" * 64
    relabelled["canary_sha256"] = eval_runner.sha256_bytes(
        eval_runner.canonical_json({
            key: value for key, value in relabelled.items()
            if key != "canary_sha256"
        })
    )
    with pytest.raises(RuntimeError, match="stale, incomplete, or relabelled"):
        eval_runner._validate_permission_canary(
            relabelled, boundary, runs_root=runs_root,
        )

    escaped_kind["value"] = "runs"
    with pytest.raises(RuntimeError, match="runs root"):
        run_permission_canaries(
            codex_command="codex", work_root=work_root, runs_root=runs_root,
            environment=environment,
        )

    escaped_kind["value"] = "gold"
    with pytest.raises(RuntimeError, match="public gold repository"):
        run_permission_canaries(
            codex_command="codex", work_root=work_root, runs_root=runs_root,
            environment=environment,
        )


def test_treatment_mcp_uses_locked_bwrap_and_minimal_environment(tmp_path: Path) -> None:
    work_root = tmp_path / "isolated-work"
    for arm in ARM_IDS:
        for corpus in load_corpus_lock()["corpora"]:
            (work_root / arm / corpus["id"]).mkdir(parents=True)
    (work_root / "_cache").mkdir()
    (work_root / "environment.lock.json").write_text("{}\n", encoding="utf-8")
    (work_root / "materialization.json").write_text("{}\n", encoding="utf-8")
    runtime = synthetic_runtime_identity(public_repository=ROOT, work_root=work_root)
    boundary = runtime["sandbox_boundary"]
    workspace = work_root / "hlsgraph-v03" / "dataflow_gemm"
    launcher, args = eval_runner._contained_mcp_command(
        arm="hlsgraph-v03", workspace=workspace,
        server_command=runtime["python"]["hlsgraph_v03"]["path"],
        server_args=["-m", "hlsgraph.mcp.server", str(workspace.resolve())],
        server_env={"HLSGRAPH_MCP_TOOLS": "explore"},
        sandbox_boundary=boundary,
    )
    assert launcher == str(Path(runtime["bubblewrap"]["path"]).resolve())
    for flag in ("--unshare-all", "--clearenv", "--cap-drop", "--proc", "--tmpfs"):
        assert flag in args
    home_index = args.index("HOME")
    assert args[home_index - 1:home_index + 2] == ["--setenv", "HOME", "/tmp/home"]
    assert not any("PROXY" in item.upper() or "CODEX_HOME" in item for item in args)
    assert str(workspace.resolve()) in args
    assert runtime["python"]["hlsgraph_v02"]["path"] not in args
    assert runtime["codex"]["path"] not in args
    assert args[args.index("--") + 1] == str(
        Path(runtime["python"]["hlsgraph_v03"]["path"]).resolve()
    )


def test_boundary_probe_is_a_real_stdio_mcp_tool(tmp_path: Path) -> None:
    probe = ROOT / "eval" / "agent_ab" / "mcp_boundary_probe.py"
    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {
            "name": "boundary_probe", "arguments": {
                "allowed": [str(probe)], "denied": [str(tmp_path / "missing")],
                "port": 1,
            },
        }},
    ]
    replies = eval_runner._run_stdio_mcp_exchange(
        [sys.executable, str(probe)], requests, cwd=tmp_path,
    )
    assert replies[2]["result"]["tools"][0]["name"] == "boundary_probe"
    assert replies[3]["result"]["structuredContent"]["allowed"] == [True]
    assert replies[3]["result"]["structuredContent"]["denied"] == [False]


@pytest.mark.skipif(
    os.name != "posix" or shutil.which("bwrap") is None,
    reason="real bubblewrap MCP containment probe",
)
def test_real_bwrap_mcp_probe_denies_host_and_network(tmp_path: Path) -> None:
    work_root = tmp_path / "isolated-work"
    for arm in ARM_IDS:
        for corpus in load_corpus_lock()["corpora"]:
            workspace = work_root / arm / corpus["id"]
            workspace.mkdir(parents=True)
            (workspace / "EVAL_PROVENANCE.json").write_text("{}\n", encoding="utf-8")
    (work_root / "_cache").mkdir()
    (work_root / "environment.lock.json").write_text("{}\n", encoding="utf-8")
    (work_root / "materialization.json").write_text("{}\n", encoding="utf-8")
    runtime = synthetic_runtime_identity(public_repository=ROOT, work_root=work_root)
    boundary = runtime["sandbox_boundary"]
    bwrap = Path(shutil.which("bwrap") or "").resolve()
    boundary["mcp_containment"]["launcher"] = {
        "path": bwrap.as_posix(), "filename": bwrap.name,
        "sha256": eval_runner.sha256_file(bwrap),
    }
    workspace = work_root / "hlsgraph-v03" / "dataflow_gemm"
    listener = __import__("socket").socket()
    try:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        result = eval_runner.probe_contained_mcp_boundary(
            arm="hlsgraph-v03", workspace=workspace,
            allowed=[workspace / "EVAL_PROVENANCE.json"],
            denied=[Path("/etc/passwd"), work_root / "environment.lock.json"],
            port=listener.getsockname()[1], sandbox_boundary=boundary,
        )
    finally:
        listener.close()
    assert result["network"] is False
    assert result["denied"] == [False, False]
    assert result["home"] == "/tmp/home"


@pytest.mark.skipif(
    os.name != "posix" or shutil.which("bwrap") is None,
    reason="real POSIX venv and bubblewrap contract",
)
def test_contained_mcp_executes_lexical_venv_python_and_imports_hlsgraph(
    tmp_path: Path,
) -> None:
    work_root = tmp_path / "isolated-work"
    workspace = work_root / "hlsgraph-v03" / "dataflow_gemm"
    workspace.mkdir(parents=True)
    (workspace / "EVAL_PROVENANCE.json").write_text("{}\n", encoding="utf-8")
    synthetic_retrieval_audit_placeholder(workspace)
    (work_root / "_cache").mkdir()
    (work_root / "environment.lock.json").write_text("{}\n", encoding="utf-8")
    (work_root / "materialization.json").write_text("{}\n", encoding="utf-8")
    runtime = synthetic_runtime_identity(public_repository=ROOT, work_root=work_root)
    boundary = runtime["sandbox_boundary"]

    venv = tmp_path / "locked-runtime" / "v03"
    completed = subprocess.run(
        [str(Path(sys.executable).resolve()), "-m", "venv", "--symlinks", str(venv)],
        capture_output=True, text=True, check=False, timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    python = venv / "bin" / "python"
    assert python.is_symlink(), "the POSIX regression requires a lexical venv symlink"
    purelib = subprocess.run(
        [str(python), "-I", "-c", "import sysconfig;print(sysconfig.get_path('purelib'))"],
        capture_output=True, text=True, check=True, timeout=30,
    ).stdout.strip()
    package = Path(purelib) / "hlsgraph"
    package.mkdir()
    (package / "__init__.py").write_text(
        "CONTAINMENT_SENTINEL = 'venv-hlsgraph-imported'\n", encoding="utf-8",
    )
    entries = [
        {
            "path": (venv / "bin").as_posix(), "kind": "tree",
            "algorithm": eval_common.SANDBOX_ALLOW_TREE_ALGORITHM,
            "sha256": eval_common.sandbox_allow_tree_identity(venv / "bin"),
        },
        {
            "path": (venv / "pyvenv.cfg").as_posix(), "kind": "file",
            "algorithm": "sha256", "sha256": sha256_file(venv / "pyvenv.cfg"),
        },
        {
            "path": Path(purelib).as_posix(), "kind": "tree",
            "algorithm": eval_common.SANDBOX_ALLOW_TREE_ALGORITHM,
            "sha256": eval_common.sandbox_allow_tree_identity(Path(purelib)),
        },
    ]
    boundary["runtime_allow_roots"]["hlsgraph-v03"] = sorted(
        [boundary["runtime_allow_roots"]["native"][0], *entries],
        key=lambda item: item["path"],
    )
    bwrap = Path(shutil.which("bwrap") or "")
    boundary["mcp_containment"]["launcher"] = {
        "path": bwrap.as_posix(), "filename": bwrap.name,
        "sha256": sha256_file(bwrap),
    }
    audit = tmp_path / "audit.jsonl"
    audit.write_bytes(b"")
    audit.chmod(0o600)
    launcher, args = eval_runner._contained_mcp_command(
        arm="hlsgraph-v03", workspace=workspace,
        server_command=python.as_posix(),
        server_args=[
            "-I", "-c",
            "import hlsgraph;print(hlsgraph.CONTAINMENT_SENTINEL)",
        ],
        server_env={}, sandbox_boundary=boundary,
        audit_overlay=(
            audit, workspace / ".hlsgraph/private/retrieval-access.jsonl",
            tuple(eval_runner._audit_parent_chain(audit)),
        ),
    )
    assert args[args.index("--") + 1] == python.as_posix()
    assert args[args.index("--") + 1] != python.resolve().as_posix()
    contained = subprocess.run(
        [launcher, *args], cwd=workspace, env={},
        capture_output=True, text=True, check=False, timeout=30,
    )
    assert contained.returncode == 0, contained.stderr
    assert contained.stdout.strip() == "venv-hlsgraph-imported"


@pytest.mark.parametrize(("private_mode", "expected"), [(0o700, True), (0o755, False)])
@pytest.mark.skipif(
    os.name != "posix" or shutil.which("bwrap") is None,
    reason="real read-only workspace audit overlay contract",
)
def test_read_only_workspace_allows_only_exact_audit_file_overlay(
    tmp_path: Path, private_mode: int, expected: bool,
) -> None:
    work_root = tmp_path / f"isolated-work-{private_mode:o}"
    workspace = work_root / "hlsgraph-v03" / "dataflow_gemm"
    workspace.mkdir(parents=True)
    placeholder = synthetic_retrieval_audit_placeholder(workspace)
    placeholder.parent.chmod(private_mode)
    (work_root / "_cache").mkdir()
    (work_root / "environment.lock.json").write_text("{}\n", encoding="utf-8")
    (work_root / "materialization.json").write_text("{}\n", encoding="utf-8")
    runtime = synthetic_runtime_identity(public_repository=ROOT, work_root=work_root)
    boundary = runtime["sandbox_boundary"]
    source_root = ROOT / "src"
    runtime_entries = [{
        "path": source_root.as_posix(), "kind": "tree",
        "algorithm": eval_common.SANDBOX_ALLOW_TREE_ALGORITHM,
        "sha256": eval_common.sandbox_allow_tree_identity(source_root),
    }]
    python_paths = [source_root.as_posix()]
    if sys.version_info < (3, 11):
        import tomli  # type: ignore[import-not-found]
        tomli_root = Path(tomli.__file__).parent
        runtime_entries.append({
            "path": tomli_root.as_posix(), "kind": "tree",
            "algorithm": eval_common.SANDBOX_ALLOW_TREE_ALGORITHM,
            "sha256": eval_common.sandbox_allow_tree_identity(tomli_root),
        })
        python_paths.append(tomli_root.parent.as_posix())
    boundary["runtime_allow_roots"]["hlsgraph-v03"].extend(runtime_entries)
    bwrap = Path(shutil.which("bwrap") or "")
    boundary["mcp_containment"]["launcher"] = {
        "path": bwrap.as_posix(), "filename": bwrap.name,
        "sha256": sha256_file(bwrap),
    }
    audit = tmp_path / f"audit-{private_mode:o}.jsonl"
    audit.write_bytes(b"")
    audit.chmod(0o600)
    script = (
        "from hlsgraph.retrieval import _append_private_access;"
        "import pathlib,sys;"
        "print(_append_private_access(pathlib.Path(sys.argv[1]),"
        "content_sha256='a'*64,anchor={'kind':'source_line','start_line':1,"
        "'end_line':1},result='returned',byte_count=7))"
    )
    kwargs = dict(
        arm="hlsgraph-v03", workspace=workspace,
        server_command="/usr/bin/python3",
        server_args=["-c", script, workspace.as_posix()],
        server_env={"PYTHONPATH": os.pathsep.join(python_paths)},
        sandbox_boundary=boundary,
        audit_overlay=(audit, placeholder, tuple(eval_runner._audit_parent_chain(audit))),
        allow_system_command=True,
    )
    if not expected:
        with pytest.raises(RuntimeError, match="private directory must have mode 0700"):
            eval_runner._contained_mcp_command(**kwargs)
        assert audit.read_bytes() == b""
        return
    launcher, args = eval_runner._contained_mcp_command(**kwargs)
    completed = subprocess.run(
        [launcher, *args], cwd=workspace, env={}, capture_output=True,
        text=True, check=False, timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == str(expected)
    if expected:
        records = eval_runner._parse_retrieval_audit(audit.read_bytes())
        assert len(records) == 1 and records[0]["result"] == "returned"


def test_quality_noninferiority_requires_every_baseline() -> None:
    comparisons = {
        ("native", "evidence_grounded_correctness"): {"ci_lower": 0.01},
        ("codegraph", "evidence_grounded_correctness"): {"ci_lower": -0.03},
        ("hlsgraph-v02", "evidence_grounded_correctness"): {"ci_lower": 0.0},
    }
    passed, details = simultaneous_quality_noninferiority(comparisons, margin=-0.02)
    assert passed is False
    assert details == {"native": True, "codegraph": False, "hlsgraph-v02": True}
