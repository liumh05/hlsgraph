"""Question-stratified paired bootstrap and public performance-gate report."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import statistics
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from .common import (
    ARM_IDS, HERE, asset_digest, canonical_json, harness_digest,
    load_environment_lock, load_manifest, load_questions,
    prepared_hlsgraph_identity, resolve_command_argv, sha256_bytes, sha256_file,
    verify_evaluation_checkout,
)
from .score import render_score_rows, score_runs
from .runner import build_run_plan
from .static_eval import render_static_json


HIGHER_IS_BETTER = {"evidence_grounded_correctness", "citation_precision"}
LOWER_IS_BETTER = {"tool_calls", "file_reads", "total_tokens", "wall_time_seconds"}
METRICS = tuple(sorted(HIGHER_IS_BETTER | LOWER_IS_BETTER))


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = _strict_json_loads(line, context=f"{path}:{line_number}")
        if not isinstance(value, dict):
            raise ValueError(f"expected object at {path}:{line_number}")
        rows.append(value)
    return rows


def _strict_json_loads(data: str | bytes, *, context: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value} in {context}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {context}")
            result[key] = value
        return result

    try:
        return json.loads(
            data, parse_constant=reject_constant, object_pairs_hook=reject_duplicates,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {context}: {exc}") from exc


def _metric_value(row: dict[str, Any], metric: str) -> float:
    value = row.get(metric)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{metric} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{metric} must be finite")
    if metric in HIGHER_IS_BETTER and not 0.0 <= number <= 1.0:
        raise ValueError(f"{metric} must be in [0, 1]")
    if metric in LOWER_IS_BETTER and number < 0.0:
        raise ValueError(f"{metric} must be non-negative")
    if metric in {"tool_calls", "file_reads", "total_tokens"} and not isinstance(value, int):
        raise ValueError(f"{metric} must be an integer")
    return number


def _quantile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot take a quantile of an empty sample")
    position = (len(sorted_values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def paired_differences(
    rows: Iterable[dict[str, Any]], *, candidate: str, baseline: str, metric: str,
) -> dict[str, list[float]]:
    if metric not in METRICS:
        raise ValueError(f"unsupported metric: {metric}")
    cells: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
    for row in rows:
        arm = row.get("arm")
        if arm not in {candidate, baseline}:
            continue
        key = (str(row["question_id"]), int(row["repetition"]))
        cells[key][str(arm)] = _metric_value(row, metric)
    grouped: dict[str, list[float]] = defaultdict(list)
    for (question_id, _repetition), values in cells.items():
        if candidate not in values or baseline not in values:
            continue
        if metric in HIGHER_IS_BETTER:
            delta = values[candidate] - values[baseline]
        else:
            delta = values[baseline] - values[candidate]
        grouped[question_id].append(delta)
    if not grouped:
        raise ValueError(f"no paired cells for {candidate} vs {baseline}")
    return dict(grouped)


def stratified_paired_bootstrap(
    rows: Iterable[dict[str, Any]], *, candidate: str, baseline: str, metric: str,
    samples: int = 10_000, seed: int = 20260721, confidence: float = 0.95,
) -> dict[str, Any]:
    if samples < 1:
        raise ValueError("samples must be positive")
    grouped = paired_differences(rows, candidate=candidate, baseline=baseline, metric=metric)
    questions = sorted(grouped)
    observed = statistics.fmean(
        statistics.fmean(grouped[question_id]) for question_id in questions
    )
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(samples):
        per_question: list[float] = []
        for question_id in questions:
            values = grouped[question_id]
            resampled = [values[rng.randrange(len(values))] for _ in values]
            per_question.append(statistics.fmean(resampled))
        draws.append(statistics.fmean(per_question))
    draws.sort()
    alpha = 1.0 - confidence
    return {
        "candidate": candidate,
        "baseline": baseline,
        "metric": metric,
        "orientation": "positive_favors_candidate",
        "observed_delta": observed,
        "ci_lower": _quantile(draws, alpha / 2.0),
        "ci_upper": _quantile(draws, 1.0 - alpha / 2.0),
        "confidence": confidence,
        "samples": samples,
        "seed": seed,
        "question_strata": len(questions),
        "paired_cells": sum(len(values) for values in grouped.values()),
    }


def _mean(rows: list[dict[str, Any]], arm: str, metric: str) -> float:
    values = [_metric_value(row, metric) for row in rows if row.get("arm") == arm]
    if not values:
        raise ValueError(f"missing rows for arm {arm}")
    return statistics.fmean(values)


def simultaneous_quality_noninferiority(
    comparisons: dict[tuple[str, str], dict[str, Any]], *, margin: float,
) -> tuple[bool, dict[str, bool]]:
    """Apply a prespecified intersection-union gate across every baseline.

    Selecting the empirically best baseline and then consulting only its
    interval is a post-selection rule.  Non-inferiority to the best baseline is
    instead accepted only when it holds for *all* three frozen baselines.  This
    intersection-union test does not choose a comparator after seeing results.
    """

    details: dict[str, bool] = {}
    for baseline in ("native", "codegraph", "hlsgraph-v02"):
        comparison = comparisons.get((baseline, "evidence_grounded_correctness"))
        if not isinstance(comparison, dict):
            raise ValueError(f"missing quality comparison for {baseline}")
        lower = comparison.get("ci_lower")
        if (isinstance(lower, bool) or not isinstance(lower, (int, float))
                or not math.isfinite(float(lower))):
            raise ValueError(f"invalid quality interval for {baseline}")
        details[baseline] = float(lower) >= margin
    return all(details.values()), details


def _score_identity_issues(
    rows: Iterable[dict[str, Any]], *, environment_lock_sha256: str,
    expected_batch_id: str | None = None,
    expected_run_set_sha256: str | None = None,
) -> list[str]:
    questions = {item["id"]: item for item in load_questions()}
    row_list = list(rows)
    execution_indices_present = any("execution_index" in row for row in row_list)
    file_semantics_present = any("file_read_semantics" in row for row in row_list)
    frozen_timeout = load_manifest()["codex_cli"]["timeout_seconds"]
    expected_execution = {
        (item["question_id"], item["repetition"], item["arm"]): item["execution_index"]
        for item in build_run_plan()
    } if execution_indices_present else {}
    issues: list[str] = []
    successful: list[dict[str, Any]] = []
    for index, row in enumerate(row_list):
        prefix = f"row[{index}]"
        question_id = row.get("question_id")
        question = questions.get(str(question_id))
        arm = row.get("arm")
        repetition = row.get("repetition")
        if row.get("schema_version") != "hlsgraph.agent_eval.score.v1":
            issues.append(f"{prefix}:schema")
        if row.get("suite_asset_sha256") != asset_digest():
            issues.append(f"{prefix}:suite")
        if row.get("evaluation_harness_sha256") != harness_digest():
            issues.append(f"{prefix}:harness")
        if row.get("environment_lock_sha256") != environment_lock_sha256:
            issues.append(f"{prefix}:environment")
        if row.get("timeout_seconds") != frozen_timeout:
            issues.append(f"{prefix}:timeout_seconds")
        if (re.fullmatch(r"[0-9a-f]{32}", str(row.get("batch_id", ""))) is None
                or expected_batch_id is not None
                and row.get("batch_id") != expected_batch_id):
            issues.append(f"{prefix}:batch")
        if (re.fullmatch(r"[0-9a-f]{64}", str(row.get("run_set_sha256", ""))) is None
                or expected_run_set_sha256 is not None
                and row.get("run_set_sha256") != expected_run_set_sha256):
            issues.append(f"{prefix}:run_set")
        for key in (
            "run_contract_sha256", "workspace_identity_sha256", "run_batch_sha256",
        ):
            if re.fullmatch(r"[0-9a-f]{64}", str(row.get(key, ""))) is None:
                issues.append(f"{prefix}:{key}")
        for metric in METRICS:
            try:
                _metric_value(row, metric)
            except ValueError:
                issues.append(f"{prefix}:metric:{metric}")
        fabricated = row.get("fabricated_truth_count")
        if isinstance(fabricated, bool) or not isinstance(fabricated, int) or fabricated < 0:
            issues.append(f"{prefix}:fabricated_truth_count")
        if question is None:
            issues.append(f"{prefix}:question")
            continue
        if arm not in ARM_IDS or not isinstance(repetition, int) or not 1 <= repetition <= 4:
            issues.append(f"{prefix}:cell")
            continue
        expected_run_id = f"{question_id}__r{repetition:02d}__{arm}"
        if row.get("run_id") != expected_run_id:
            issues.append(f"{prefix}:run_id")
        if (row.get("corpus_id") != question["corpus_id"]
                or row.get("category") != question["category"]):
            issues.append(f"{prefix}:question_metadata")
        if execution_indices_present:
            expected_index = expected_execution.get((str(question_id), repetition, str(arm)))
            if (isinstance(row.get("execution_index"), bool)
                    or row.get("execution_index") != expected_index):
                issues.append(f"{prefix}:execution_index")
        if (file_semantics_present
                and row.get("file_read_semantics") != "source_access_tool_calls"):
            issues.append(f"{prefix}:file_read_semantics")
        if row.get("parse_error"):
            continue
        if row.get("total_tokens", 0) <= 0:
            issues.append(f"{prefix}:terminal_usage")
        successful.append(row)
        source_hashes = row.get("source_hashes")
        expected_source_names = {
            "run.json", "prompt.txt", "codex.jsonl", "codex.stderr.log",
            "retrieval-access.jsonl",
        }
        if (not isinstance(source_hashes, dict)
                or set(source_hashes) != expected_source_names
                or any(re.fullmatch(r"[0-9a-f]{64}", str(value)) is None
                       for value in source_hashes.values())):
            issues.append(f"{prefix}:source_hashes")
        else:
            if row.get("trace_sha256") != source_hashes["codex.jsonl"]:
                issues.append(f"{prefix}:trace_sha256")
            if row.get("run_source_sha256") != sha256_bytes(canonical_json(source_hashes)):
                issues.append(f"{prefix}:run_source_sha256")
        if not isinstance(row.get("thread_id"), str) or not row["thread_id"]:
            issues.append(f"{prefix}:thread_id")
        retrieval_audit = row.get("retrieval_audit")
        expected_audit_status = "verified" if arm == "hlsgraph-v03" else "not_applicable"
        if (not isinstance(retrieval_audit, dict)
                or retrieval_audit.get("status") != expected_audit_status
                or re.fullmatch(r"[0-9a-f]{64}", str(
                    retrieval_audit.get("sha256", "")
                )) is None
                or re.fullmatch(r"[0-9a-f]{64}", str(
                    retrieval_audit.get("receipt_sha256", "")
                )) is None
                or any(
                    isinstance(retrieval_audit.get(key), bool)
                    or not isinstance(retrieval_audit.get(key), int)
                    or retrieval_audit[key] < 0
                    for key in (
                        "record_count", "returned_count", "returned_bytes",
                        "source_access_calls",
                    )
                )
                or retrieval_audit.get("returned_count", 0)
                > retrieval_audit.get("record_count", 0)):
            issues.append(f"{prefix}:retrieval_audit")
        trace_policy = row.get("trace_policy")
        if (
            not isinstance(trace_policy, dict) or trace_policy.get("passed") is not True
            or trace_policy.get("arm") != arm
            or trace_policy.get("workspace") != "$CORPUS_WORKSPACE"
        ):
            issues.append(f"{prefix}:trace_policy")
        elif arm == "native":
            if (trace_policy.get("treatment_mcp_required") is not False
                    or trace_policy.get("treatment_mcp_calls") != 0
                    or trace_policy.get("first_call_treatment_mcp") is not False
                    or trace_policy.get("treatment_mcp_first_outcome")
                    != "not_applicable"):
                issues.append(f"{prefix}:trace_policy:treatment")
        elif (trace_policy.get("treatment_mcp_required") is not True
                or isinstance(trace_policy.get("treatment_mcp_calls"), bool)
                or not isinstance(trace_policy.get("treatment_mcp_calls"), int)
                or trace_policy.get("treatment_mcp_calls", 0) < 1
                or trace_policy.get("first_call_treatment_mcp") is not True
                or trace_policy.get("treatment_mcp_first_outcome")
                not in {"completed", "failed", "incomplete"}):
            issues.append(f"{prefix}:trace_policy:treatment")
    for key in ("batch_id", "run_set_sha256", "run_batch_sha256"):
        if len({row.get(key) for row in row_list}) != 1:
            issues.append(f"matrix:{key}:inconsistent")
    expected_batch_sources = [
        {"run_id": row.get("run_id"), "run_source_sha256": row.get("run_source_sha256")}
        for row in sorted(row_list, key=lambda item: str(item.get("run_id", "")))
    ]
    expected_run_batch = sha256_bytes(canonical_json(expected_batch_sources))
    if any(row.get("run_batch_sha256") != expected_run_batch for row in row_list):
        issues.append("matrix:run_batch_sha256:invalid")
    for key in ("thread_id", "trace_sha256", "run_source_sha256"):
        values = [row.get(key) for row in successful]
        if len(values) != len(set(values)):
            issues.append(f"matrix:{key}:duplicate")
    return issues


def _validate_static_identity(
    report: dict[str, Any], *, environment_lock_sha256: str,
    candidate_identity: dict[str, str],
) -> None:
    metrics = report.get("metrics")
    metrics_valid = isinstance(metrics, dict)
    if metrics_valid:
        for key in ("recall_at_8", "ndcg_at_8", "citation_precision"):
            value = metrics.get(key)
            metrics_valid = bool(
                metrics_valid and not isinstance(value, bool)
                and isinstance(value, (int, float)) and math.isfinite(float(value))
                and 0.0 <= float(value) <= 1.0
            )
        for key in ("citation_count", "fabricated_truth_count"):
            value = metrics.get(key)
            metrics_valid = bool(
                metrics_valid and isinstance(value, int)
                and not isinstance(value, bool) and value >= 0
            )
    if (report.get("schema_version") != "hlsgraph.agent_eval.static_report.v1"
            or report.get("suite_asset_sha256") != asset_digest()
            or report.get("evaluation_harness_sha256") != harness_digest()
            or report.get("environment_lock_sha256") != environment_lock_sha256
            or report.get("candidate_identity") != candidate_identity
            or re.fullmatch(r"[0-9a-f]{64}", str(report.get("raw_results_sha256", ""))) is None
            or not metrics_valid
            or report.get("passed") is not True):
        raise ValueError(
            "static retrieval report is stale, unpassed, or belongs to another candidate"
        )


def evaluate_gates(
    rows: list[dict[str, Any]], comparisons: list[dict[str, Any]],
    static_report: dict[str, Any] | None = None, *,
    environment_lock_sha256: str, candidate_identity: dict[str, str],
) -> dict[str, Any]:
    manifest = load_manifest()
    gates = manifest["release_gates"]
    candidate = "hlsgraph-v03"
    expected_keys = {
        (question["id"], repetition, arm)
        for question in load_questions()
        for repetition in range(1, manifest["repetitions"] + 1)
        for arm in ARM_IDS
    }
    observed_counter = Counter(
        (str(row.get("question_id")), int(row.get("repetition", -1)), str(row.get("arm")))
        for row in rows
    )
    observed_keys = set(observed_counter)
    missing_keys = sorted(expected_keys - observed_keys)
    unexpected_keys = sorted(observed_keys - expected_keys)
    duplicate_keys = sorted(key for key, count in observed_counter.items() if count != 1)
    expected_cells = len(expected_keys)
    identity_issues = _score_identity_issues(
        rows, environment_lock_sha256=environment_lock_sha256,
    )
    complete = (
        not missing_keys and not unexpected_keys and not duplicate_keys
        and not identity_issues
        and not any(row.get("parse_error") or row.get("timed_out") or row.get("returncode") != 0
                    for row in rows)
    )
    fabricated = sum(
        int(row.get("fabricated_truth_count", 0))
        for row in rows if row.get("arm") == candidate
    )
    baseline_quality = {
        arm: _mean(rows, arm, "evidence_grounded_correctness")
        for arm in ARM_IDS if arm != candidate
    }
    best_baseline = max(baseline_quality, key=baseline_quality.get)
    expected_comparisons = {
        (baseline, metric)
        for baseline in ("native", "codegraph", "hlsgraph-v02")
        for metric in METRICS
    }
    by_key = {
        (item.get("baseline"), item.get("metric")): item for item in comparisons
        if isinstance(item, dict)
    }
    if len(by_key) != len(comparisons) or set(by_key) != expected_comparisons:
        raise ValueError("bootstrap comparisons are missing, duplicated, or unexpected")
    for (baseline, metric), item in by_key.items():
        if (item.get("candidate") != candidate or item.get("baseline") != baseline
                or item.get("metric") != metric
                or item.get("orientation") != "positive_favors_candidate"):
            raise ValueError("bootstrap comparison identity is invalid")
        for name in ("observed_delta", "ci_lower", "ci_upper", "confidence"):
            value = item.get(name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))):
                raise ValueError(f"bootstrap comparison has invalid {name}")
        if not 0.0 < float(item["confidence"]) < 1.0:
            raise ValueError("bootstrap confidence must be in (0, 1)")
        if float(item["ci_lower"]) > float(item["ci_upper"]):
            raise ValueError("bootstrap confidence interval is inverted")
        for name, expected in (
            ("samples", manifest["bootstrap"]["samples"]),
            ("seed", manifest["bootstrap"]["seed"]),
            ("question_strata", len(load_questions())),
            ("paired_cells", len(load_questions()) * manifest["repetitions"]),
        ):
            if item.get(name) != expected:
                raise ValueError(f"bootstrap comparison has invalid {name}")
    quality_ni, quality_ni_by_baseline = simultaneous_quality_noninferiority(
        by_key, margin=gates["quality_noninferiority_margin"],
    )
    codegraph_superior = by_key[("codegraph", "evidence_grounded_correctness")][
        "ci_lower"
    ] > gates["codegraph_quality_delta_ci_lower"]
    positive_efficiency: list[str] = []
    significantly_worse: list[str] = []
    for metric in sorted(LOWER_IS_BETTER):
        native = by_key[("native", metric)]
        v02 = by_key[("hlsgraph-v02", metric)]
        if native["ci_lower"] > 0.0 and v02["ci_lower"] > 0.0:
            positive_efficiency.append(metric)
        if native["ci_upper"] < 0.0 or v02["ci_upper"] < 0.0:
            significantly_worse.append(metric)
    efficiency = (
        len(positive_efficiency) >= gates["minimum_positive_efficiency_metrics"]
        and not significantly_worse
    )
    static_metrics = static_report.get("metrics", {}) if isinstance(static_report, dict) else {}
    try:
        if static_report is None:
            raise ValueError("static report is missing")
        _validate_static_identity(
            static_report, environment_lock_sha256=environment_lock_sha256,
            candidate_identity=candidate_identity,
        )
        static_identity = True
    except ValueError:
        static_identity = False
    static_gate = static_identity and (
        float(static_metrics.get("recall_at_8", -1.0)) >= gates["static_recall_at_8"]
        and float(static_metrics.get("ndcg_at_8", -1.0)) >= gates["static_ndcg_at_8"]
        and float(static_metrics.get("citation_precision", -1.0))
        >= gates["static_citation_precision"]
        and int(static_metrics.get("fabricated_truth_count", -1))
        == gates["fabricated_truth_count"]
    )
    supported = (
        complete and static_gate and fabricated == 0 and quality_ni
        and codegraph_superior and efficiency
    )
    return {
        "complete_matrix": complete,
        "expected_cells": expected_cells,
        "observed_cells": len(rows),
        "unique_observed_cells": len(observed_keys),
        "missing_cells": [list(item) for item in missing_keys],
        "unexpected_cells": [list(item) for item in unexpected_keys],
        "duplicate_cells": [list(item) for item in duplicate_keys],
        "identity_issues": identity_issues,
        "static_retrieval_report_present": static_report is not None,
        "static_retrieval_gate": static_gate,
        "static_retrieval_metrics": static_metrics,
        "fabricated_truth_count": fabricated,
        "best_quality_baseline": best_baseline,
        "quality_noninferiority": quality_ni,
        "quality_noninferiority_by_baseline": quality_ni_by_baseline,
        "codegraph_hls_quality_superiority": codegraph_superior,
        "positive_efficiency_metrics_vs_native_and_v02": positive_efficiency,
        "significantly_worse_efficiency_metrics": significantly_worse,
        "performance_advantage_supported": supported,
        "claim_policy": (
            "Only performance_advantage_supported=true permits a public advantage claim; "
            "otherwise report a technical preview without such a claim."
        ),
    }


def analyze(
    rows: list[dict[str, Any]], static_report: dict[str, Any], *,
    environment_lock_sha256: str, candidate_identity: dict[str, str],
    scores_sha256: str | None = None, static_report_sha256: str | None = None,
    run_set: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if (run_set is None
            or run_set.get("schema_version") != "hlsgraph.agent_eval.run_set.v1"
            or run_set.get("suite_asset_sha256") != asset_digest()
            or run_set.get("evaluation_harness_sha256") != harness_digest()
            or run_set.get("environment_lock_sha256") != environment_lock_sha256
            or run_set.get("timeout_seconds")
            != load_manifest()["codex_cli"]["timeout_seconds"]
            or not isinstance(run_set.get("runs_root"), str)
            or not Path(run_set["runs_root"]).is_absolute()
            or run_set.get("run_set_sha256") != sha256_bytes(canonical_json({
                key: value for key, value in run_set.items() if key != "run_set_sha256"
            }))):
        raise ValueError("bootstrap analysis requires the verified frozen run set")
    expected_scores_hash = sha256_bytes(render_score_rows(rows))
    expected_static_hash = sha256_bytes(render_static_json(static_report))
    if scores_sha256 != expected_scores_hash or static_report_sha256 != expected_static_hash:
        raise ValueError("bootstrap analysis artifact hashes are missing or inconsistent")
    identity_issues = _score_identity_issues(
        rows, environment_lock_sha256=environment_lock_sha256,
        expected_batch_id=(run_set or {}).get("batch_id"),
        expected_run_set_sha256=(run_set or {}).get("run_set_sha256"),
    )
    if identity_issues:
        raise ValueError("score rows fail frozen identity checks: " + ", ".join(identity_issues))
    _validate_static_identity(
        static_report, environment_lock_sha256=environment_lock_sha256,
        candidate_identity=candidate_identity,
    )
    bootstrap = load_manifest()["bootstrap"]
    comparisons = [
        stratified_paired_bootstrap(
            rows, candidate="hlsgraph-v03", baseline=baseline, metric=metric,
            samples=bootstrap["samples"], seed=bootstrap["seed"],
            confidence=bootstrap["confidence"],
        )
        for baseline in ("native", "codegraph", "hlsgraph-v02")
        for metric in METRICS
    ]
    return {
        "schema_version": "hlsgraph.agent_eval.bootstrap_report.v1",
        "method": bootstrap["method"],
        "comparisons": comparisons,
        "suite_asset_sha256": asset_digest(),
        "evaluation_harness_sha256": harness_digest(),
        "environment_lock_sha256": environment_lock_sha256,
        "candidate_identity": candidate_identity,
        "scores_sha256": scores_sha256,
        "static_report_sha256": static_report_sha256,
        "batch_id": (run_set or {}).get("batch_id"),
        "run_set_sha256": (run_set or {}).get("run_set_sha256"),
        "run_batch_sha256": rows[0].get("run_batch_sha256") if rows else None,
        "gates": evaluate_gates(
            rows, comparisons, static_report,
            environment_lock_sha256=environment_lock_sha256,
            candidate_identity=candidate_identity,
        ),
    }


def _run_static_subprocess(
    python: str, arguments: list[str], *, timeout_seconds: int = 600,
) -> None:
    repository = HERE.parents[1].resolve()
    launcher = (
        "import runpy,sys;"
        "sys.path.insert(0,sys.argv.pop(1));"
        "runpy.run_module('eval.agent_ab.static_eval',run_name='__main__')"
    )
    environment = os.environ.copy()
    for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"):
        environment.pop(name, None)
    completed = subprocess.run(
        [python, "-I", "-c", launcher, str(repository), *arguments],
        cwd=repository, env=environment, capture_output=True, text=True,
        check=False, timeout=timeout_seconds,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "independent candidate static recheck failed: "
            + (completed.stderr.strip() or completed.stdout.strip())
        )


def recompute_static_report(v03_python: str, work_root: Path) -> bytes:
    """Collect and score static retrieval in the candidate wheel interpreter."""
    parts = resolve_command_argv(v03_python)
    if len(parts) != 1:
        raise ValueError("--v03-python must be one direct interpreter")
    python = parts[0]
    with tempfile.TemporaryDirectory(prefix="hlsgraph-static-recheck-") as directory:
        temporary = Path(directory)
        raw_path = temporary / "static-results.json"
        report_path = temporary / "static-report.json"
        _run_static_subprocess(python, [
            "collect", "--work-root", str(work_root.resolve()),
            "--output", str(raw_path), "--execute",
        ])
        _run_static_subprocess(python, [
            "score", str(raw_path), "--work-root", str(work_root.resolve()),
            "--output", str(report_path),
        ])
        try:
            report_bytes = report_path.read_bytes()
        except OSError as exc:
            raise RuntimeError("candidate static recheck produced no report") from exc
        report = _strict_json_loads(report_bytes, context=str(report_path))
        if not isinstance(report, dict) or report_bytes != render_static_json(report):
            raise ValueError("candidate static report is not canonical JSON")
        return report_bytes


def verify_scores_against_raw(
    scores_path: Path, runs_root: Path, work_root: Path,
) -> tuple[list[dict[str, Any]], bytes]:
    rows = score_runs(runs_root, work_root)
    canonical = render_score_rows(rows)
    if scores_path.read_bytes() != canonical:
        raise ValueError(
            "scores artifact differs byte-for-byte from deterministic raw-trace rescoring"
        )
    return rows, canonical


def verify_static_against_candidate(
    static_report_path: Path, v03_python: str, work_root: Path,
) -> tuple[dict[str, Any], bytes]:
    supplied = static_report_path.read_bytes()
    recomputed = recompute_static_report(v03_python, work_root)
    if supplied != recomputed:
        raise ValueError(
            "static report differs byte-for-byte from the candidate interpreter recheck"
        )
    report = _strict_json_loads(supplied, context=str(static_report_path))
    if not isinstance(report, dict):
        raise ValueError("static report must contain a JSON object")
    return report, supplied


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", type=Path)
    parser.add_argument("--static-report", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--v03-python", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    environment_lock = args.work_root / "environment.lock.json"
    environment = load_environment_lock(environment_lock)
    v03_parts = resolve_command_argv(args.v03_python)
    expected_python = environment["runtime_identity"]["python"]["hlsgraph_v03"]
    if (len(v03_parts) != 1 or v03_parts[0] != expected_python.get("path")
            or sha256_file(Path(v03_parts[0])) != expected_python.get("sha256")):
        raise ValueError("--v03-python differs from the prepared runtime identity")
    args.v03_python = v03_parts[0]
    verify_evaluation_checkout(environment)
    recomputed_rows, recomputed_scores = verify_scores_against_raw(
        args.scores, args.runs_root, args.work_root,
    )
    static_report, static_bytes = verify_static_against_candidate(
        args.static_report, args.v03_python, args.work_root,
    )
    run_set_path = args.runs_root / "run-set.json"
    run_set = _strict_json_loads(run_set_path.read_bytes(), context=str(run_set_path))
    if not isinstance(run_set, dict):
        raise ValueError("run-set must contain a JSON object")
    report = analyze(
        recomputed_rows, static_report,
        environment_lock_sha256=sha256_file(environment_lock),
        candidate_identity=prepared_hlsgraph_identity(environment, "hlsgraph-v03"),
        scores_sha256=sha256_bytes(recomputed_scores),
        static_report_sha256=sha256_bytes(static_bytes),
        run_set=run_set,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(render_static_json(report))
    print(json.dumps(report["gates"], sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
