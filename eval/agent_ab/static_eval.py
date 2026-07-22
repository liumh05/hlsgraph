"""Collect and score the deterministic HLSGraph v0.3 retrieval gate.

Collection is inert unless ``--execute`` is supplied.  Scoring uses only frozen
matchers, returned record metadata, and citations; it never calls a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Sequence

from .common import (
    HERE, asset_digest, canonical_json, harness_digest, load_environment_lock,
    load_manifest, load_static_cases, prepared_hlsgraph_identity, sha256_bytes,
    sha256_file, verify_evaluation_checkout, verify_prepared_workspace,
)
from .wheel_identity import inspect_installed


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


_RECORD_CONTRACTS = {
    "entity": (r"static_fact|declared_constraint|synthetic", r"ast|source|schedule"),
    "derivation": (r"derived_fact", r"ast"),
    "observation": (r"synthetic", r"cosim|rtl_cosim"),
    "relation": (r"declared_constraint", r"source"),
    "diagnostic": (r"tool_observation|static_fact", r"ast"),
    "knowledge_rule": (r"knowledge_rule", r"source|schedule"),
}


def _matches(
    item: dict[str, Any], matcher: dict[str, Any], *, check_contract: bool = True,
) -> bool:
    if matcher.get("record_kind") and item.get("record_kind") != matcher["record_kind"]:
        return False
    data = item.get("data") if isinstance(item.get("data"), dict) else {}
    if matcher.get("data_rule_id") and data.get("rule_id") != matcher["data_rule_id"]:
        return False
    if matcher.get("data_kind") and re.search(
        matcher["data_kind"], str(data.get("kind", "")), re.IGNORECASE,
    ) is None:
        return False
    pattern = matcher.get("pattern")
    if pattern and re.search(pattern, _canonical(item), re.IGNORECASE) is None:
        return False
    if check_contract:
        authority_pattern, stage_pattern = _RECORD_CONTRACTS.get(
            str(item.get("record_kind")), (r".+", r".+"),
        )
        authority_pattern = matcher.get("authority_pattern", authority_pattern)
        stage_pattern = matcher.get("stage_pattern", stage_pattern)
        if re.fullmatch(authority_pattern, str(item.get("authority_class", "")), re.IGNORECASE) is None:
            return False
        if re.fullmatch(stage_pattern, str(item.get("stage", "")), re.IGNORECASE) is None:
            return False
    return True


def _citation_matches(citation: dict[str, Any], selector: dict[str, Any]) -> bool:
    if selector.get("document_id") != citation.get("document_id"):
        return False
    section_pattern = selector.get("section_pattern")
    if section_pattern and re.search(
        section_pattern, str(citation.get("section", "")), re.IGNORECASE,
    ) is None:
        return False
    return True


def _dcg(grades: Iterable[int]) -> float:
    return sum((2 ** grade - 1) / math.log2(rank + 2)
               for rank, grade in enumerate(grades))


def _environment_identity(work_root: Path) -> tuple[dict[str, Any], str, dict[str, str]]:
    lock_path = work_root / "environment.lock.json"
    environment = load_environment_lock(lock_path)
    verify_evaluation_checkout(environment)
    lock_sha256 = sha256_file(lock_path)
    candidate = prepared_hlsgraph_identity(environment, "hlsgraph-v03")
    runtime = inspect_installed(
        "0.3.0", candidate["installed_payload_sha256"],
    )
    if (runtime.get("verified") is not True
            or runtime.get("installed_payload_sha256")
            != candidate["installed_payload_sha256"]):
        raise RuntimeError("static collector runtime differs from the prepared v0.3 wheel")
    return environment, lock_sha256, candidate


def collect(work_root: Path) -> dict[str, Any]:
    # Import lazily so dry plans and deterministic scoring do not depend on the
    # candidate package being installed in the orchestration interpreter.
    from hlsgraph.retrieval import RetrievalSpec
    from hlsgraph.sdk import Project

    environment, lock_sha256, candidate = _environment_identity(work_root)
    records: list[dict[str, Any]] = []
    for case in load_static_cases():
        workspace = (work_root / "hlsgraph-v03" / case["corpus_id"]).resolve()
        prepared = verify_prepared_workspace(
            environment, work_root, "hlsgraph-v03", case["corpus_id"],
        )
        snapshot_id = prepared.get("snapshot_id")
        if not isinstance(snapshot_id, str) or not snapshot_id:
            raise RuntimeError(f"missing pinned v0.3 snapshot for static case {case['id']}")
        result = Project.open(workspace).retrieve(RetrievalSpec(
            query=case["query"], view=case["view"], top_k=8,
            snapshot_id=snapshot_id,
            include_private_snippets=False, include_predictions=False,
        )).to_dict()
        # Wall-clock channel timings are useful operational telemetry but are
        # intentionally outside the deterministic release gate.  Removing
        # them makes an independent re-collection byte-reproducible while the
        # profile, candidate counts, graph hash, truncation, and output budget
        # remain bound in the raw result.
        trace = result.get("trace")
        if isinstance(trace, dict):
            trace["elapsed_ms"] = {}
            trace["output_chars"] = 0
            for _ in range(4):
                size = len(json.dumps(
                    result, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                ))
                if trace["output_chars"] == size:
                    break
                trace["output_chars"] = size
        if result.get("snapshot_id") != snapshot_id:
            raise RuntimeError(f"static case {case['id']} returned an unpinned snapshot")
        verify_prepared_workspace(
            environment, work_root, "hlsgraph-v03", case["corpus_id"],
        )
        records.append({
            "case_id": case["id"], "corpus_id": case["corpus_id"],
            "snapshot_id": snapshot_id,
            "workspace_identity_sha256": prepared["workspace_identity_sha256"],
            "result": result,
        })
    payload = {
        "schema_version": "hlsgraph.agent_eval.static_results.v1",
        "suite_asset_sha256": asset_digest(),
        "evaluation_harness_sha256": harness_digest(),
        "environment_lock_sha256": lock_sha256,
        "candidate_identity": candidate,
        "top_k": 8,
        "cases": records,
    }
    payload["raw_results_sha256"] = sha256_bytes(canonical_json(payload))
    return payload


def _fabricated(case: dict[str, Any], result: dict[str, Any]) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    for item in [*result.get("facts", []), *result.get("guidance", [])]:
        authority = str(item.get("authority_class", ""))
        if authority == "prediction_hypothesis" or item.get("plane") == "predictions":
            findings.append({"case_id": case["id"], "record_id": item.get("record_id", ""),
                             "reason": "prediction-entered-fact-or-guidance"})
        for matcher in case.get("forbidden", []):
            if _matches(item, matcher, check_contract=False):
                findings.append({"case_id": case["id"], "record_id": item.get("record_id", ""),
                                 "reason": "forbidden-record-matcher"})
        for matcher in case.get("gold", []):
            if (_matches(item, matcher, check_contract=False)
                    and not _matches(item, matcher, check_contract=True)):
                findings.append({"case_id": case["id"], "record_id": item.get("record_id", ""),
                                 "reason": "gold-record-authority-stage-mismatch"})
    for edge in result.get("flow", []):
        if edge.get("kind") in {"software.calls", "llvm.cfg"}:
            findings.append({"case_id": case["id"], "record_id": edge.get("id", ""),
                             "reason": "non-hardware-edge-in-flow"})
    if result.get("predictions"):
        findings.append({"case_id": case["id"], "record_id": "predictions",
                         "reason": "predictions-returned-while-disabled"})
    return findings


def score(payload: dict[str, Any], *, work_root: Path) -> dict[str, Any]:
    if payload.get("schema_version") != "hlsgraph.agent_eval.static_results.v1":
        raise ValueError("unsupported static result schema")
    if payload.get("suite_asset_sha256") != asset_digest() or payload.get("top_k") != 8:
        raise ValueError("static results do not match the frozen suite identity")
    if payload.get("evaluation_harness_sha256") != harness_digest():
        raise ValueError("static results were produced by different evaluation code")
    unhashed = {key: value for key, value in payload.items() if key != "raw_results_sha256"}
    if payload.get("raw_results_sha256") != sha256_bytes(canonical_json(unhashed)):
        raise ValueError("static results content hash is invalid")
    environment = load_environment_lock(work_root / "environment.lock.json")
    verify_evaluation_checkout(environment)
    lock_sha256 = sha256_file(work_root / "environment.lock.json")
    candidate = prepared_hlsgraph_identity(environment, "hlsgraph-v03")
    if (payload.get("environment_lock_sha256") != lock_sha256
            or payload.get("candidate_identity") != candidate):
        raise ValueError("static results do not match the prepared candidate environment")
    cases = {item["id"]: item for item in load_static_cases()}
    rows = payload.get("cases")
    if not isinstance(rows, list):
        raise ValueError("static results must contain a cases list")
    row_ids = [item.get("case_id") for item in rows]
    if len(row_ids) != len(set(row_ids)) or set(row_ids) != set(cases):
        raise ValueError("static result case matrix is missing, duplicated, or unexpected")

    found_gold = 0
    total_gold = 0
    ndcgs: list[float] = []
    citation_total = 0
    citation_correct = 0
    fabricated: list[dict[str, str]] = []
    case_reports: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: item["case_id"]):
        case = cases[row["case_id"]]
        prepared = verify_prepared_workspace(
            environment, work_root, "hlsgraph-v03", case["corpus_id"],
        )
        if (row.get("corpus_id") != case["corpus_id"]
                or row.get("snapshot_id") != prepared.get("snapshot_id")
                or row.get("workspace_identity_sha256")
                != prepared.get("workspace_identity_sha256")):
            raise ValueError(f"static case {case['id']} has a stale or relabelled identity")
        result = row.get("result")
        if not isinstance(result, dict):
            raise ValueError(f"static case {case['id']} lacks a retrieval result")
        trace = result.get("trace")
        if (result.get("snapshot_id") != prepared.get("snapshot_id")
                or not isinstance(trace, dict)
                or trace.get("snapshot_id") != prepared.get("snapshot_id")
                or trace.get("query_sha256")
                != hashlib.sha256(case["query"].encode("utf-8")).hexdigest()
                or trace.get("profile") != "hls.default.v1"
                or trace.get("profile_schema_version") != "0.3.0"
                or trace.get("algorithm_version") != "hlsgraph.hybrid_retrieval.v1"
                or re.fullmatch(r"[0-9a-f]{64}", str(trace.get("profile_hash", ""))) is None
                or re.fullmatch(r"[0-9a-f]{64}", str(trace.get("graph_hash", ""))) is None
                or trace.get("elapsed_ms") != {}
                or trace.get("private_snippets_requested") is not False
                or trace.get("private_snippets_returned") is not False):
            raise ValueError(f"static case {case['id']} has a stale retrieval trace")
        section = case["result_section"]
        items = result.get(section)
        if not isinstance(items, list):
            raise ValueError(f"static case {case['id']} lacks result section {section}")
        items = items[:8]
        gold = case["gold"]
        matched_ids = {
            matcher["id"] for matcher in gold
            if any(_matches(item, matcher) for item in items)
        }
        found_gold += len(matched_ids)
        total_gold += len(gold)

        unused = {matcher["id"]: matcher for matcher in gold}
        grades: list[int] = []
        for item in items:
            candidates = [matcher for matcher in unused.values() if _matches(item, matcher)]
            if candidates:
                selected = max(candidates, key=lambda matcher: (matcher["grade"], matcher["id"]))
                grades.append(int(selected["grade"]))
                unused.pop(selected["id"])
            else:
                grades.append(0)
            citation = item.get("citation")
            if isinstance(citation, dict):
                citation_total += 1
                if any(_citation_matches(citation, selector)
                       for selector in case.get("citation_gold", [])):
                    citation_correct += 1
        ideal = sorted((int(matcher["grade"]) for matcher in gold), reverse=True)[:8]
        ideal_dcg = _dcg(ideal)
        ndcg = _dcg(grades) / ideal_dcg if ideal_dcg else 0.0
        ndcgs.append(ndcg)
        case_fabricated = _fabricated(case, result)
        fabricated.extend(case_fabricated)
        case_reports.append({
            "case_id": case["id"], "gold": len(gold),
            "found": len(matched_ids), "matched_ids": sorted(matched_ids),
            "ndcg_at_8": ndcg, "returned": len(items),
            "fabricated_truth_count": len(case_fabricated),
        })

    metrics = {
        "recall_at_8": found_gold / total_gold if total_gold else 0.0,
        "ndcg_at_8": sum(ndcgs) / len(ndcgs) if ndcgs else 0.0,
        "citation_precision": citation_correct / citation_total if citation_total else 0.0,
        "citation_count": citation_total,
        "fabricated_truth_count": len(fabricated),
    }
    gates = load_manifest()["release_gates"]
    passed = (
        metrics["recall_at_8"] >= gates["static_recall_at_8"]
        and metrics["ndcg_at_8"] >= gates["static_ndcg_at_8"]
        and metrics["citation_precision"] >= gates["static_citation_precision"]
        and metrics["fabricated_truth_count"] == gates["fabricated_truth_count"]
    )
    return {
        "schema_version": "hlsgraph.agent_eval.static_report.v1",
        "suite_asset_sha256": asset_digest(),
        "evaluation_harness_sha256": harness_digest(),
        "environment_lock_sha256": lock_sha256,
        "candidate_identity": candidate,
        "raw_results_sha256": payload["raw_results_sha256"],
        "metrics": metrics,
        "passed": passed, "cases": case_reports, "fabrications": fabricated,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    collect_parser = sub.add_parser("collect")
    collect_parser.add_argument("--work-root", type=Path, default=HERE / "work")
    collect_parser.add_argument("--output", type=Path, required=True)
    collect_parser.add_argument("--execute", action="store_true")
    score_parser = sub.add_parser("score")
    score_parser.add_argument("input", type=Path)
    score_parser.add_argument("--work-root", type=Path, default=HERE / "work")
    score_parser.add_argument("--output", type=Path, required=True)
    return parser


def render_static_json(value: dict[str, Any]) -> bytes:
    return (json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False,
    ) + "\n").encode("utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "collect":
        if not args.execute:
            print(json.dumps({
                "executed": False, "top_k": 8,
                "cases": [{"id": item["id"], "corpus_id": item["corpus_id"]}
                          for item in load_static_cases()],
            }, indent=2, sort_keys=True))
            return 0
        payload = collect(args.work_root)
    else:
        payload = score(
            json.loads(args.input.read_text(encoding="utf-8")),
            work_root=args.work_root,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(render_static_json(payload))
    print(json.dumps({"output": str(args.output), "passed": payload.get("passed")}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
