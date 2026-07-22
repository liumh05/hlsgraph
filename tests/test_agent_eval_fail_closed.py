from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
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
    _validate_execution_contract, _validate_terminal_usage, canonical_answer,
    public_criterion_ids, score_answer,
)
from tests.agent_eval_runtime_support import synthetic_runtime_identity


ROOT = Path(__file__).resolve().parents[1]


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
    assert 'permissions.hlsgraph_eval.extends=":read-only"' in joined
    assert "permissions.hlsgraph_eval.network.enabled=false" in joined
    assert "permissions.hlsgraph_eval.filesystem={" in joined
    assert f'"{runs_root.resolve().as_posix()}"="deny"' in joined
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


def test_permission_profile_denies_sibling_directories_and_rejects_unknown_inventory(
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
    assert f'"{(work_root / "codegraph").as_posix()}"="deny"' in filesystem
    assert (
        f'"{(work_root / "native" / "cordic").as_posix()}"="deny"'
        in filesystem
    )
    assert f'"{(work_root / "native" / "dataflow_gemm").as_posix()}"="read"' in filesystem
    assert f'"{runs_root.resolve().as_posix()}"="deny"' in filesystem
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
    cell = {**codegraph_record, "command_argv": codegraph_command}
    cell["run_contract_sha256"] = eval_runner.sha256_bytes(
        eval_runner.canonical_json(cell)
    )
    environment = {"runtime_identity": runtime}
    _validate_execution_contract(
        cell, work_root, environment, runs_root=runs_root,
    )
    with pytest.raises(ValueError, match="permission profile"):
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
        if value.startswith("mcp_servers.codegraph.env.CODEGRAPH_TELEMETRY=")
    )
    changed["command_argv"][telemetry_index] = (
        "mcp_servers.codegraph.env.CODEGRAPH_TELEMETRY=\"1\""
    )
    rehash(changed)
    with pytest.raises(ValueError, match="offline environment"):
        _validate_execution_contract(
            changed, work_root, environment, runs_root=runs_root,
        )

    changed_runs_rule = copy.deepcopy(cell)
    filesystem_index = next(
        index for index, value in enumerate(changed_runs_rule["command_argv"])
        if value.startswith("permissions.hlsgraph_eval.filesystem=")
    )
    exact_deny = f'"{runs_root.resolve().as_posix()}"="deny"'
    changed_runs_rule["command_argv"][filesystem_index] = (
        changed_runs_rule["command_argv"][filesystem_index].replace(
            exact_deny, f'"{runs_root.resolve().as_posix()}"="read"',
        )
    )
    rehash(changed_runs_rule)
    with pytest.raises(ValueError, match="permission profile"):
        _validate_execution_contract(
            changed_runs_rule, work_root, environment, runs_root=runs_root,
        )

    v03_record = {**record, "arm": "hlsgraph-v03"}
    v03_command = build_codex_command(
        v03_record, work_root=work_root,
        runs_root=runs_root,
        codex_command=runtime["codex"]["path"],
        v02_python=runtime["python"]["hlsgraph_v02"]["path"],
        v03_python=runtime["python"]["hlsgraph_v03"]["path"],
        codegraph_command="codegraph",
        sandbox_boundary=runtime["sandbox_boundary"],
    )
    v03_cell = {**v03_record, "command_argv": v03_command}
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


def test_mcp_source_retrieval_counts_as_file_read() -> None:
    events = [
        {
            "type": "item.completed",
            "item": {
                "id": "mcp-1",
                "type": "mcp_tool_call",
                "name": "hlsgraph.explore",
                "arguments": {"query": "flow", "include_private_snippets": True},
            },
        },
        _message_event(_minimal_answer()),
    ]
    normalized = normalize_trace(events)
    assert normalized["tool_calls"] == 1
    assert normalized["file_reads"] == 1
    assert normalized["file_read_semantics"] == "source_access_tool_calls"


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
    monkeypatch.setattr(eval_runner, "_sandbox_canary_prefix", lambda *_args, **_kwargs: ["canary"])

    results = iter([
        subprocess.CompletedProcess([], 0, "", ""),
        subprocess.CompletedProcess([], 1, "", "denied"),
        subprocess.CompletedProcess([], 1, "", "denied"),
        subprocess.CompletedProcess([], 1, "", "denied"),
        subprocess.CompletedProcess([], 1, "", "denied"),
        subprocess.CompletedProcess([], 1, "", "denied"),
        subprocess.CompletedProcess([], 1, "", "denied"),
        subprocess.CompletedProcess([], 1, "", "denied"),
        subprocess.CompletedProcess([], 1, "", "denied"),
        subprocess.CompletedProcess([], 1, "", "denied"),
        subprocess.CompletedProcess([], 1, "", "denied"),
    ])
    monkeypatch.setattr(eval_runner, "_run_canary_command", lambda *_args, **_kwargs: next(results))
    report = run_permission_canaries(
        codex_command="codex", work_root=work_root, runs_root=runs_root,
        environment=environment,
    )
    assert report["workspace_read"] == "pass"
    assert report["sibling_workspace_read"] == "denied"
    assert report["same_arm_sibling_read"] == "denied"
    assert report["other_arm_sibling_read"] == "denied"
    assert report["boundary_control_read"] == "denied"
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

    runs_escaped = iter([
        subprocess.CompletedProcess([], 0, "", ""),
        subprocess.CompletedProcess([], 1, "", "denied"),
        subprocess.CompletedProcess([], 1, "", "denied"),
        subprocess.CompletedProcess([], 1, "", "denied"),
        subprocess.CompletedProcess([], 0, "runs", ""),
    ])
    monkeypatch.setattr(
        eval_runner, "_run_canary_command", lambda *_args, **_kwargs: next(runs_escaped),
    )
    with pytest.raises(RuntimeError, match="runs root"):
        run_permission_canaries(
            codex_command="codex", work_root=work_root, runs_root=runs_root,
            environment=environment,
        )

    escaped = iter([
        subprocess.CompletedProcess([], 0, "", ""),
        subprocess.CompletedProcess([], 1, "", "denied"),
        subprocess.CompletedProcess([], 1, "", "denied"),
        subprocess.CompletedProcess([], 1, "", "denied"),
        subprocess.CompletedProcess([], 1, "", "denied"),
        subprocess.CompletedProcess([], 0, "gold", ""),
    ])
    monkeypatch.setattr(eval_runner, "_run_canary_command", lambda *_args, **_kwargs: next(escaped))
    with pytest.raises(RuntimeError, match="public gold-file"):
        run_permission_canaries(
            codex_command="codex", work_root=work_root, runs_root=runs_root,
            environment=environment,
        )


def test_quality_noninferiority_requires_every_baseline() -> None:
    comparisons = {
        ("native", "evidence_grounded_correctness"): {"ci_lower": 0.01},
        ("codegraph", "evidence_grounded_correctness"): {"ci_lower": -0.03},
        ("hlsgraph-v02", "evidence_grounded_correctness"): {"ci_lower": 0.0},
    }
    passed, details = simultaneous_quality_noninferiority(comparisons, margin=-0.02)
    assert passed is False
    assert details == {"native": True, "codegraph": False, "hlsgraph-v02": True}
