"""Deterministically score structured answers against public evidence selectors."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
from pathlib import Path
from typing import Any, Iterable, Sequence

from .common import (
    ARM_IDS, EvalManifestError, asset_digest, canonical_json, harness_digest,
    load_corpus_lock, load_environment_lock, load_manifest, load_questions,
    safe_relative_path, sha256_bytes, sha256_file, verify_evaluation_checkout,
    verify_prepared_workspace,
)
from .parse_trace import normalize_trace, validate_trace_policy
from .runner import (
    CODEGRAPH_ENV, DISABLED_CODEX_FEATURES, PERMISSION_PROFILE, _permission_overrides,
    _require_isolated_work_root, _validate_permission_canary,
    _verify_boundary_canary, build_prompt, build_run_plan,
)
from .setup_corpus import _provenance


def verify_workspace_corpus(workspace: Path, corpus_id: str) -> None:
    """Fail closed if bytes used for scoring differ from the frozen corpus."""
    corpus = next(
        (item for item in load_corpus_lock()["corpora"] if item["id"] == corpus_id),
        None,
    )
    if corpus is None:
        raise EvalManifestError(f"unknown corpus: {corpus_id}")
    root = workspace.resolve()
    for entry in corpus["files"]:
        relative = safe_relative_path(entry["destination"])
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise EvalManifestError(f"corpus file escapes workspace: {relative}") from exc
        if not path.is_file() or sha256_file(path) != entry["sha256"]:
            raise EvalManifestError(f"frozen corpus byte mismatch: {relative.as_posix()}")
    provenance_path = root / "EVAL_PROVENANCE.json"
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvalManifestError("missing or malformed EVAL_PROVENANCE.json") from exc
    if provenance != _provenance(corpus):
        raise EvalManifestError("EVAL_PROVENANCE.json differs from the frozen corpus lock")


def public_criterion_ids(question: dict[str, Any]) -> dict[str, str]:
    """Expose opaque stable-in-question IDs without leaking rubric semantics."""

    return {
        str(criterion["id"]): f"c{index:02d}"
        for index, criterion in enumerate(question["criteria"], 1)
    }


def canonical_answer(
    question: dict[str, Any], claims: Iterable[dict[str, Any]], *,
    allow_unknown: bool = False,
) -> str:
    expected = public_criterion_ids(question)
    order = {public: index for index, public in enumerate(expected.values())}
    claim_list = list(claims)
    if not allow_unknown and any(claim.get("criterion_id") not in order for claim in claim_list):
        raise ValueError("claim has an unknown criterion_id")
    ordered = sorted(
        claim_list,
        key=lambda claim: (
            order.get(str(claim.get("criterion_id")), len(order)),
            str(claim.get("criterion_id", "")), str(claim.get("id", "")),
        ),
    )
    return "\n".join(
        f"{claim['criterion_id']}: {str(claim['statement']).strip()}" for claim in ordered
    )


def validate_answer(answer: dict[str, Any], *, require_bound: bool = False) -> None:
    if require_bound and set(answer) != {"answer", "claims", "uncertainties"}:
        raise ValueError("strict answer contains missing or additional top-level fields")
    if not isinstance(answer.get("answer"), str) or not answer["answer"].strip():
        raise ValueError("answer.answer must be a non-empty string")
    if len(answer["answer"]) > 12_000:
        raise ValueError("answer.answer exceeds the frozen schema limit")
    if not isinstance(answer.get("claims"), list):
        raise ValueError("answer.claims must be a list")
    if require_bound and not 1 <= len(answer["claims"]) <= 32:
        raise ValueError("strict answer has an invalid claim count")
    if not isinstance(answer.get("uncertainties"), list):
        raise ValueError("answer.uncertainties must be a list")
    if require_bound and len(answer["uncertainties"]) > 16:
        raise ValueError("strict answer has too many uncertainties")
    for claim in answer["claims"]:
        if not isinstance(claim, dict):
            raise ValueError("every claim must be an object")
        required = {"id", "statement", "truth_plane", "stage", "authority", "evidence"}
        if require_bound:
            required.add("criterion_id")
            if set(claim) != required:
                raise ValueError("strict claim contains missing or additional fields")
        if not required.issubset(claim):
            raise ValueError(f"claim lacks fields: {sorted(required - set(claim))}")
        if (not isinstance(claim.get("id"), str)
                or re.fullmatch(r"[a-zA-Z0-9_.-]{1,64}", claim["id"]) is None):
            raise ValueError("claim id does not match the frozen schema")
        if (not isinstance(claim["statement"], str) or not claim["statement"].strip()
                or "\n" in claim["statement"] or "\r" in claim["statement"]
                or len(claim["statement"]) > 2_000
                or not isinstance(claim["evidence"], list)):
            raise ValueError("claim statement/evidence has the wrong type")
        if claim.get("truth_plane") not in {
            "design_fact", "synthetic_observation", "tool_observation",
            "knowledge_guidance", "prediction", "unknown",
        }:
            raise ValueError("claim truth_plane is unsupported")
        if (claim.get("stage") is not None and (
            not isinstance(claim.get("stage"), str) or len(claim["stage"]) > 64
        )):
            raise ValueError("claim stage has the wrong type")
        if (not isinstance(claim.get("authority"), str) or not claim["authority"]
                or len(claim["authority"]) > 128):
            raise ValueError("claim authority has the wrong type")
        criterion_id = claim.get("criterion_id")
        if criterion_id is not None and re.fullmatch(r"c[0-9]{2}", str(criterion_id)) is None:
            raise ValueError("claim criterion_id must use the opaque cNN form")
        if not claim["evidence"]:
            raise ValueError("every claim must contain evidence")
        if require_bound and len(claim["evidence"]) > 12:
            raise ValueError("claim has too many citations")
        for citation in claim["evidence"]:
            if not isinstance(citation, dict):
                raise ValueError("every citation must be an object")
            if require_bound and (
                not {"path", "line_start", "line_end"}.issubset(citation)
                or set(citation) - {"path", "line_start", "line_end", "evidence_id"}
            ):
                raise ValueError("strict citation contains missing or additional fields")
            if (not isinstance(citation.get("path"), str) or not citation["path"]
                    or len(citation["path"]) > 512):
                raise ValueError("citation path has the wrong type")
            evidence_id = citation.get("evidence_id")
            if (evidence_id is not None
                    and (not isinstance(evidence_id, str) or len(evidence_id) > 256)):
                raise ValueError("citation evidence_id has the wrong type")
            start, end = citation.get("line_start"), citation.get("line_end")
            if (not isinstance(start, int) or not isinstance(end, int)
                    or start < 1 or end < start or end - start + 1 > 80):
                raise ValueError("citation ranges must contain between 1 and 80 lines")
    ids = [str(claim.get("id")) for claim in answer["claims"]]
    if len(ids) != len(set(ids)):
        raise ValueError("claim IDs must be unique")
    if any(not isinstance(item, str) for item in answer["uncertainties"]):
        raise ValueError("answer.uncertainties entries must be strings")
    if any(len(item) > 1_000 for item in answer["uncertainties"]):
        raise ValueError("answer uncertainty exceeds the frozen schema limit")


def _matches_all(text: str, patterns: Iterable[str]) -> bool:
    return all(re.search(pattern, text, re.IGNORECASE | re.DOTALL) for pattern in patterns)


def _citation_lines(citation: dict[str, Any], workspace: Path) -> tuple[Path, list[str]] | None:
    try:
        relative = safe_relative_path(str(citation.get("path", "")))
    except EvalManifestError:
        return None
    root = Path(os.path.abspath(os.fspath(workspace)))
    path = root / relative
    try:
        path.relative_to(root)
    except ValueError:
        return None
    start, end = citation.get("line_start"), citation.get("line_end")
    if not isinstance(start, int) or not isinstance(end, int) or start < 1 or end < start:
        return None
    try:
        data = _stable_run_bytes(path, max_bytes=64 * 1024 * 1024)
    except ValueError:
        return None
    lines = data.decode("utf-8", errors="replace").splitlines()
    if end > len(lines):
        return None
    return relative, lines[start - 1:end]


def citation_valid(citation: dict[str, Any], workspace: Path) -> bool:
    return _citation_lines(citation, workspace) is not None


def citation_matches_selector(
    citation: dict[str, Any], selector: dict[str, Any], workspace: Path,
) -> bool:
    resolved = _citation_lines(citation, workspace)
    if resolved is None:
        return False
    relative, lines = resolved
    if relative.as_posix() != safe_relative_path(selector["path"]).as_posix():
        return False
    start, end = citation["line_start"], citation["line_end"]
    selector_start = selector.get("line_start")
    selector_end = selector.get("line_end")
    if selector_start is not None and selector_end is not None:
        if end < selector_start or start > selector_end:
            return False
    contains = selector.get("contains")
    if contains is not None and contains not in "\n".join(lines):
        return False
    return True


def _normalize_stage(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return "__invalid__"
    return re.sub(r"[-\s]+", "_", value.strip().casefold())


def _claim_contract_match(
    claim: dict[str, Any], criterion: dict[str, Any], contracts: dict[str, Any],
) -> bool:
    """Require the claimed plane, authority, and stage to agree with frozen gold.

    Synthetic fixtures use the cited artifact class to narrow the broad stage
    contract.  This prevents, for example, a post-route WNS fixture from being
    credited as a source or synthesis-stage statement.
    """
    contract = contracts.get(str(claim.get("truth_plane")))
    if not isinstance(contract, dict):
        return False
    authority = claim.get("authority")
    pattern = contract.get("authority_pattern")
    if not isinstance(authority, str) or not isinstance(pattern, str):
        return False
    if re.search(pattern, authority, re.IGNORECASE) is None:
        return False
    allowed = list(contract.get("allowed_stages", []))
    selector_rules = contract.get("selector_stage_rules", [])
    narrowed: list[Any] = []
    for selector in criterion.get("evidence_selectors", []):
        path = str(selector.get("path", ""))
        for rule in selector_rules:
            if re.search(str(rule.get("path_pattern", r"(?!)")), path, re.IGNORECASE):
                narrowed.extend(rule.get("allowed_stages", []))
    if narrowed:
        specific = [item for item in narrowed if _normalize_stage(item) not in {
            None, "source", "unknown",
        }]
        if specific:
            narrowed = specific
        allowed = narrowed
    normalized_allowed = {_normalize_stage(item) for item in allowed}
    return _normalize_stage(claim.get("stage")) in normalized_allowed


def _bind_claims(
    question: dict[str, Any], claims: list[dict[str, Any]], *, require_bound: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Bind exactly one atomic claim to each frozen criterion.

    ``require_bound=False`` is a narrow compatibility path for direct unit
    callers created before the public answer schema acquired criterion_id.  It
    deterministically infers only a unique rubric match; official raw traces
    always use ``require_bound=True`` and cannot enter this path.
    """

    public = public_criterion_ids(question)
    reverse = {value: key for key, value in public.items()}
    criteria = {str(item["id"]): item for item in question["criteria"]}
    bound: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    for claim in claims:
        copied = dict(claim)
        criterion_id = copied.get("criterion_id")
        internal_id = reverse.get(str(criterion_id)) if criterion_id is not None else None
        if internal_id is None and not require_bound and criterion_id is None:
            candidates = [
                criterion for criterion in question["criteria"]
                if copied.get("truth_plane") in criterion["truth_planes"]
                and _matches_all(str(copied.get("statement", "")), criterion["claim_patterns"])
            ]
            if len(candidates) == 1:
                internal_id = str(candidates[0]["id"])
                copied["criterion_id"] = public[internal_id]
        if internal_id is None:
            if require_bound:
                raise ValueError("every claim must bind one known criterion_id")
            copied["criterion_id"] = "__unbound__"
            violations.append({
                "id": "unbound-claim", "claim_id": copied.get("id"),
                "detail": "claim does not uniquely bind a frozen criterion",
            })
        else:
            copied["_criterion_internal_id"] = internal_id
            criterion = criteria[internal_id]
            if not _matches_all(str(copied.get("statement", "")), criterion["claim_patterns"]):
                violations.append({
                    "id": "criterion-pattern-mismatch", "claim_id": copied.get("id"),
                    "detail": copied.get("criterion_id"),
                })
            if copied.get("truth_plane") not in criterion["truth_planes"]:
                violations.append({
                    "id": "criterion-truth-plane-mismatch", "claim_id": copied.get("id"),
                    "detail": copied.get("criterion_id"),
                })
        bound.append(copied)

    counts = {
        public_id: sum(claim.get("criterion_id") == public_id for claim in bound)
        for public_id in public.values()
    }
    invalid_counts = {key: value for key, value in counts.items() if value != 1}
    if require_bound and invalid_counts:
        raise ValueError(
            "strict answer requires exactly one claim per criterion: "
            + ", ".join(f"{key}={value}" for key, value in sorted(invalid_counts.items()))
        )
    for criterion_id, count in sorted(invalid_counts.items()):
        violations.append({
            "id": "criterion-cardinality", "claim_id": None,
            "detail": f"{criterion_id} has {count} claims",
        })
    return bound, violations


def _criterion_result(
    criterion: dict[str, Any], claims: list[dict[str, Any]], workspace: Path,
    contracts: dict[str, Any], public_criterion_id: str,
) -> dict[str, Any]:
    matching_claims = [
        claim for claim in claims
        if claim.get("criterion_id") == public_criterion_id
        and claim.get("truth_plane") in criterion["truth_planes"]
        and _matches_all(claim.get("statement", ""), criterion["claim_patterns"])
    ]
    contracted_claims = [
        claim for claim in matching_claims
        if _claim_contract_match(claim, criterion, contracts)
    ]
    selectors = criterion.get("evidence_selectors", [])
    mode = criterion.get("evidence_mode", "any")
    grounded_claim: str | None = None
    for claim in contracted_claims:
        citations = claim.get("evidence", [])
        selector_matches = [
            any(citation_matches_selector(citation, selector, workspace) for citation in citations)
            for selector in selectors
        ]
        evidence_ok = all(selector_matches) if mode == "all" else any(selector_matches)
        citation_scope_ok = all(
            any(citation_matches_selector(citation, selector, workspace)
                for selector in selectors)
            for citation in citations
        )
        if not selectors:
            evidence_ok = citation_scope_ok = True
        if evidence_ok and citation_scope_ok:
            grounded_claim = str(claim.get("id"))
            break
    return {
        "criterion_id": criterion["id"],
        "weight": criterion["weight"],
        "claim_match": bool(matching_claims),
        "contract_match": bool(contracted_claims),
        "evidence_match": grounded_claim is not None,
        "grounded_claim_id": grounded_claim,
    }


def score_answer(
    question: dict[str, Any], answer: dict[str, Any], workspace: Path,
    *, require_bound: bool = False,
) -> dict[str, Any]:
    validate_answer(answer, require_bound=require_bound)
    claims, binding_violations = _bind_claims(
        question, answer["claims"], require_bound=require_bound,
    )
    if require_bound and answer["answer"] != canonical_answer(question, answer["claims"]):
        raise ValueError("answer must equal the canonical criterion-ordered claim rendering")
    contracts = load_manifest()["claim_contracts"]
    public_ids = public_criterion_ids(question)
    criterion_results = [
        _criterion_result(
            criterion, claims, workspace, contracts, public_ids[str(criterion["id"])],
        )
        for criterion in question["criteria"]
    ]
    total_weight = sum(item["weight"] for item in criterion_results)
    earned_weight = sum(
        item["weight"] for item in criterion_results if item["evidence_match"]
    )

    citations = [citation for claim in claims for citation in claim.get("evidence", [])]
    valid_citations = [citation for citation in citations if citation_valid(citation, workspace)]
    criteria_by_id = {str(item["id"]): item for item in question["criteria"]}
    relevant_citations: list[dict[str, Any]] = []
    for claim in claims:
        criterion = criteria_by_id.get(str(claim.get("_criterion_internal_id", "")))
        if criterion is None:
            continue
        selectors = criterion.get("evidence_selectors", [])
        relevant_citations.extend(
            citation for citation in claim.get("evidence", [])
            if any(citation_matches_selector(citation, selector, workspace) for selector in selectors)
        )
    citation_precision = len(relevant_citations) / len(citations) if citations else 0.0

    violations: list[dict[str, Any]] = list(binding_violations)
    contract_mismatches: set[str] = set()
    for criterion in question["criteria"]:
        public_id = public_ids[str(criterion["id"])]
        for claim in claims:
            if (claim.get("criterion_id") == public_id
                    and claim.get("truth_plane") in criterion["truth_planes"]
                    and _matches_all(claim.get("statement", ""), criterion["claim_patterns"])
                    and not _claim_contract_match(claim, criterion, contracts)):
                contract_mismatches.add(str(claim.get("id")))
    for claim_id in sorted(contract_mismatches):
        violations.append({
            "id": "authority-stage-contract", "claim_id": claim_id,
            "detail": "claim authority or stage does not match its frozen truth-plane contract",
        })
    forbidden_planes = set(question.get("forbidden_truth_planes", []))
    for claim in claims:
        if claim.get("truth_plane") in forbidden_planes:
            violations.append({
                "id": "forbidden-truth-plane", "claim_id": claim.get("id"),
                "detail": claim.get("truth_plane"),
            })
    free_text = [("answer", None, answer["answer"])]
    free_text.extend(
        ("claim", claim.get("id"), str(claim.get("statement", ""))) for claim in claims
    )
    free_text.extend(
        ("uncertainty", index, str(value))
        for index, value in enumerate(answer.get("uncertainties", []))
    )
    for rule in question.get("forbidden_claims", []):
        for source, identity, text in free_text:
            if any(re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                   for pattern in rule.get("patterns", [])):
                violations.append({
                    "id": f"forbidden-free-text:{rule['id']}",
                    "claim_id": identity if source == "claim" else None,
                    "detail": f"{source}:{identity}",
                })
    unsupported: list[Any] = []
    for claim in claims:
        criterion = criteria_by_id.get(str(claim.get("_criterion_internal_id", "")))
        if criterion is None:
            unsupported.append(claim.get("id"))
            continue
        selectors = criterion.get("evidence_selectors", [])
        matches = [
            any(citation_matches_selector(citation, selector, workspace)
                for citation in claim.get("evidence", []))
            for selector in selectors
        ]
        evidence_ok = (
            all(matches) if criterion.get("evidence_mode", "any") == "all"
            else any(matches)
        ) if selectors else True
        citation_scope_ok = all(
            any(citation_matches_selector(citation, selector, workspace)
                for selector in selectors)
            for citation in claim.get("evidence", [])
        ) if selectors else True
        semantic_ok = (
            claim.get("truth_plane") in criterion["truth_planes"]
            and _matches_all(str(claim.get("statement", "")), criterion["claim_patterns"])
            and _claim_contract_match(claim, criterion, contracts)
        )
        if not evidence_ok or not citation_scope_ok or not semantic_ok:
            unsupported.append(claim.get("id"))
    for claim_id in unsupported:
        violations.append({
            "id": "unsupported-evidence", "claim_id": claim_id,
            "detail": "claim evidence does not satisfy its bound criterion selectors",
        })
    fabricated = len(violations)
    return {
        "question_id": question["id"],
        "evidence_grounded_correctness": earned_weight / total_weight if total_weight else 0.0,
        "citation_precision": citation_precision,
        "citation_count": len(citations),
        "valid_citation_count": len(valid_citations),
        "relevant_citation_count": len(relevant_citations),
        "unsupported_claims": len(unsupported),
        "unsupported_claim_ids": unsupported,
        "fabricated_truth_count": fabricated,
        "violations": violations,
        "criteria": criterion_results,
    }


def _stable_run_bytes(path: Path, *, max_bytes: int) -> bytes:
    is_junction = getattr(path, "is_junction", None)
    if path.is_symlink() or bool(callable(is_junction) and is_junction()):
        raise ValueError(f"run evidence path is linked: {path.name}")
    descriptor = -1
    try:
        before = path.lstat()
        flags = os.O_RDONLY | int(getattr(os, "O_BINARY", 0)) | int(
            getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        closed = os.fstat(descriptor)
        current = path.lstat()
    except OSError as exc:
        raise ValueError(f"cannot snapshot run evidence {path.name}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    identity = lambda value: (
        int(value.st_dev), int(value.st_ino), int(value.st_size),
        int(value.st_mtime_ns),
    )
    data = b"".join(chunks)
    if (not stat.S_ISREG(before.st_mode) or not stat.S_ISREG(opened.st_mode)
            or len(data) > max_bytes or identity(before) != identity(opened)
            or identity(opened) != identity(closed)
            or identity(closed) != identity(current)):
        raise ValueError(f"run evidence changed while snapshotted: {path.name}")
    return data


def _strict_json_bytes(data: bytes, *, context: str) -> Any:
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
            data.decode("utf-8"), parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid UTF-8 JSON in {context}") from exc


def _snapshot_run(
    run_dir: Path,
) -> tuple[dict[str, bytes], dict[str, Any], list[dict[str, Any]]]:
    is_junction = getattr(run_dir, "is_junction", None)
    if (run_dir.is_symlink() or bool(callable(is_junction) and is_junction())
            or not run_dir.is_dir()):
        raise ValueError("run directory is missing or linked")
    limits = {
        "run.json": 1024 * 1024,
        "prompt.txt": 128 * 1024,
        "codex.jsonl": 256 * 1024 * 1024,
        "codex.stderr.log": 64 * 1024 * 1024,
    }
    source = {
        name: _stable_run_bytes(run_dir / name, max_bytes=limit)
        for name, limit in limits.items()
    }
    run = _strict_json_bytes(source["run.json"], context="run.json")
    if not isinstance(run, dict):
        raise ValueError("run.json must contain an object")
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(source["codex.jsonl"].splitlines(), 1):
        if not line.strip():
            continue
        event = _strict_json_bytes(line, context=f"codex.jsonl:{line_number}")
        if not isinstance(event, dict):
            raise ValueError(f"codex.jsonl:{line_number} must contain an object")
        events.append(event)
    normalized = normalize_trace(events)
    normalized["run"] = run
    return source, normalized, events


def _validate_run_metadata(
    run: dict[str, Any], run_dir: Path, expected_cell: dict[str, Any],
    run_set: dict[str, Any], environment_lock_sha256: str,
    *, prompt_bytes: bytes | None = None,
) -> dict[str, Any]:
    questions = {item["id"]: item for item in load_questions()}
    if run.get("schema_version") != "hlsgraph.agent_eval.run.v1":
        raise ValueError("run metadata has an unsupported schema")
    question_id = expected_cell["question_id"]
    question = questions[question_id]
    arm = expected_cell["arm"]
    expected = {
        key: expected_cell[key] for key in (
            "run_id", "question_id", "corpus_id", "category", "arm", "repetition",
        )
    }
    if "execution_index" in expected_cell:
        expected["execution_index"] = expected_cell["execution_index"]
    expected.update({
        "suite_asset_sha256": asset_digest(),
        "evaluation_harness_sha256": harness_digest(),
        "environment_lock_sha256": environment_lock_sha256,
        "batch_id": run_set["batch_id"],
        "run_set_sha256": run_set["run_set_sha256"],
        "run_contract_sha256": expected_cell["run_contract_sha256"],
        "trace_challenge": expected_cell["trace_challenge"],
        "workspace_identity_sha256": expected_cell["workspace_identity_sha256"],
        "workspace": f"$WORK_ROOT/{arm}/{question['corpus_id']}",
        "prompt_sha256": expected_cell["prompt_sha256"],
        "command_argv": expected_cell["command_argv"],
        "timeout_seconds": expected_cell["timeout_seconds"],
    })
    if "boundary_canary" in expected_cell:
        expected["boundary_canary"] = expected_cell["boundary_canary"]
    permission_canary = run_set.get("permission_canary")
    if isinstance(permission_canary, dict) and "canary_sha256" in permission_canary:
        expected["permission_canary_sha256"] = permission_canary["canary_sha256"]
    mismatched = sorted(key for key, value in expected.items() if run.get(key) != value)
    if mismatched or run_dir.name != expected_cell["run_id"]:
        raise ValueError(
            "run metadata is relabelled, stale, or stored under the wrong run ID: "
            + ", ".join(mismatched or ["run_directory"])
        )
    expected_prompt = build_prompt(
        question, str(arm), trace_challenge=expected_cell["trace_challenge"],
    )
    prompt_path = run_dir / "prompt.txt"
    try:
        if prompt_bytes is None:
            prompt_bytes = prompt_path.read_bytes()
        prompt = prompt_bytes.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError("run is missing its exact UTF-8 prompt") from exc
    if prompt != expected_prompt:
        raise ValueError("run prompt differs from the frozen question and arm")
    if run.get("prompt_sha256") != hashlib.sha256(prompt_bytes).hexdigest():
        raise ValueError("run prompt hash does not match prompt.txt")
    wall_time = run.get("wall_time_seconds")
    if (isinstance(wall_time, bool) or not isinstance(wall_time, (int, float))
            or not math.isfinite(float(wall_time)) or float(wall_time) < 0.0):
        raise ValueError("run wall time must be finite and non-negative")
    if not isinstance(run.get("timed_out"), bool):
        raise ValueError("run timeout status must be boolean")
    if (isinstance(run.get("returncode"), bool)
            or not isinstance(run.get("returncode"), int)):
        raise ValueError("run return code must be an integer")
    return question


def _validate_execution_contract(
    cell: dict[str, Any], work_root: Path, environment: dict[str, Any], *,
    runs_root: Path,
) -> None:
    command = cell.get("command_argv")
    if not isinstance(command, list) or not command or any(
        not isinstance(item, str) for item in command
    ):
        raise ValueError("run-set cell lacks command argv")
    contract = {key: value for key, value in cell.items() if key != "run_contract_sha256"}
    if cell.get("run_contract_sha256") != sha256_bytes(canonical_json(contract)):
        raise ValueError("run-set cell contract hash is invalid")
    if cell.get("timeout_seconds") != load_manifest()["codex_cli"]["timeout_seconds"]:
        raise ValueError("run-set cell changes the frozen timeout")
    def same_lexical_path(left: Any, right: Any) -> bool:
        return isinstance(left, str) and isinstance(right, str) and (
            os.path.normcase(os.path.abspath(left))
            == os.path.normcase(os.path.abspath(right))
        )

    runtime = environment.get("runtime_identity")
    codex_identity = runtime.get("codex") if isinstance(runtime, dict) else None
    if (not isinstance(codex_identity, dict)
            or not same_lexical_path(command[0], codex_identity.get("path"))):
        raise ValueError("run-set command does not use the prepared Codex executable")
    for flag in (
        "--strict-config", "exec", "--ignore-user-config", "--ignore-rules",
        "--ephemeral", "--json",
        "--skip-git-repo-check",
    ):
        if command.count(flag) != 1:
            raise ValueError(f"run-set command must contain exactly one {flag}")
    if (command[-1] != "-" or "--enable" in command or "--sandbox" in command
            or "--search" in command):
        raise ValueError("run-set command has an unsafe input or feature override")

    def one_option(flag: str) -> str:
        positions = [index for index, value in enumerate(command) if value == flag]
        if len(positions) != 1 or positions[0] + 1 >= len(command):
            raise ValueError(f"run-set command must contain exactly one {flag} value")
        return command[positions[0] + 1]

    manifest = load_manifest()
    expected_options = {
        "-a": "never",
        "--model": manifest["model"]["id"],
        "--color": "never",
        "--output-schema": str((
            Path(__file__).resolve().parent / manifest["codex_cli"]["output_schema"]
        ).resolve()),
    }
    for flag, expected_value in expected_options.items():
        if one_option(flag) != expected_value:
            raise ValueError(f"run-set command has the wrong {flag} value")
    for feature in DISABLED_CODEX_FEATURES:
        if sum(
            command[index:index + 2] == ["--disable", feature]
            for index in range(len(command) - 1)
        ) != 1:
            raise ValueError(f"run-set command must disable {feature} exactly once")
    disabled_values = [
        command[index + 1] for index, value in enumerate(command[:-1])
        if value == "--disable"
    ]
    if set(disabled_values) != set(DISABLED_CODEX_FEATURES):
        raise ValueError("run-set command changes the frozen disabled-feature set")
    expected_workspace = str((
        work_root / cell["arm"] / cell["corpus_id"]
    ).resolve())
    try:
        workspace = one_option("--cd")
    except ValueError as exc:
        raise ValueError("run-set command lacks a workspace") from exc
    if Path(workspace).resolve() != Path(expected_workspace):
        raise ValueError("run-set command points at another workspace")
    joined = "\n".join(command)
    has_codegraph = "mcp_servers.codegraph." in joined
    has_hlsgraph = "mcp_servers.hlsgraph." in joined
    arm = cell["arm"]
    if arm == "native" and (has_codegraph or has_hlsgraph):
        raise ValueError("native arm unexpectedly enables an MCP server")
    if arm == "codegraph" and (not has_codegraph or has_hlsgraph):
        raise ValueError("CodeGraph arm has the wrong MCP server")
    if arm.startswith("hlsgraph-") and (not has_hlsgraph or has_codegraph):
        raise ValueError("HLSGraph arm has the wrong MCP server")
    if arm == "hlsgraph-v02" and "HLSGRAPH_MCP_TOOLS=\"all\"" not in joined:
        raise ValueError("v0.2 arm does not use the frozen all-tools mode")
    if arm == "hlsgraph-v03" and "HLSGRAPH_MCP_TOOLS=\"explore\"" not in joined:
        raise ValueError("v0.3 arm does not use the frozen explore-only mode")
    config_values = [
        command[index + 1] for index, value in enumerate(command[:-1]) if value == "-c"
    ]
    boundary = environment["runtime_identity"]["sandbox_boundary"]
    expected_permission_values = _permission_overrides(
        workspace=Path(expected_workspace), work_root=work_root,
        runs_root=runs_root,
        sandbox_boundary=boundary,
    )[1::2]
    expected_permission_config = {
        value.split("=", 1)[0]: value.split("=", 1)[1]
        for value in expected_permission_values
    }
    expected_config_keys = {"model_reasoning_effort", *expected_permission_config}
    if arm == "codegraph":
        expected_config_keys.update({
            "mcp_servers.codegraph.command", "mcp_servers.codegraph.args",
            *{
                f"mcp_servers.codegraph.env.{key}" for key in CODEGRAPH_ENV
            },
        })
    elif arm.startswith("hlsgraph-"):
        expected_config_keys.update({
            "mcp_servers.hlsgraph.command", "mcp_servers.hlsgraph.args",
            "mcp_servers.hlsgraph.env.HLSGRAPH_MCP_TOOLS",
        })
    actual_config_keys = [value.split("=", 1)[0] for value in config_values if "=" in value]
    if (len(actual_config_keys) != len(config_values)
            or set(actual_config_keys) != expected_config_keys
            or len(actual_config_keys) != len(expected_config_keys)):
        raise ValueError("run-set command has extra, missing, or duplicate config overrides")
    actual_config = {
        value.split("=", 1)[0]: value.split("=", 1)[1] for value in config_values
    }
    if any(actual_config.get(key) != expected_value
           for key, expected_value in expected_permission_config.items()):
        raise ValueError("run-set command changes the frozen permission profile")
    if arm == "codegraph" and any(
        actual_config.get(f"mcp_servers.codegraph.env.{key}")
        != json.dumps(value, ensure_ascii=False)
        for key, value in CODEGRAPH_ENV.items()
    ):
        raise ValueError("run-set command changes the frozen CodeGraph offline environment")

    def parsed_config(key: str) -> Any:
        try:
            return json.loads(actual_config[key])
        except (KeyError, json.JSONDecodeError) as exc:
            raise ValueError(f"run-set command has invalid {key}") from exc

    if arm == "codegraph":
        node = runtime.get("node") if isinstance(runtime, dict) else None
        entrypoint = runtime.get("codegraph_entrypoint") if isinstance(runtime, dict) else None
        codegraph_args = parsed_config("mcp_servers.codegraph.args")
        if (not isinstance(node, dict) or not isinstance(entrypoint, dict)
                or not same_lexical_path(
                    parsed_config("mcp_servers.codegraph.command"), node.get("path")
                )
                or not isinstance(codegraph_args, list) or len(codegraph_args) != 3
                or not same_lexical_path(codegraph_args[0], entrypoint.get("path"))
                or codegraph_args[1:] != ["serve", "--mcp"]):
            raise ValueError("run-set command changes the prepared CodeGraph runtime or args")
    elif arm.startswith("hlsgraph-"):
        python_key = "hlsgraph_v02" if arm == "hlsgraph-v02" else "hlsgraph_v03"
        python_identity = (
            runtime.get("python", {}).get(python_key)
            if isinstance(runtime, dict) else None
        )
        expected_mode = "all" if arm == "hlsgraph-v02" else "explore"
        if (not isinstance(python_identity, dict)
                or not same_lexical_path(
                    parsed_config("mcp_servers.hlsgraph.command"),
                    python_identity.get("path"),
                )
                or parsed_config("mcp_servers.hlsgraph.args") != [
                    "-m", "hlsgraph.mcp.server", expected_workspace,
                ]
                or parsed_config("mcp_servers.hlsgraph.env.HLSGRAPH_MCP_TOOLS")
                != expected_mode):
            raise ValueError("run-set command changes the prepared HLSGraph runtime or args")
    reasoning = next(
        value.split("=", 1)[1] for value in config_values
        if value.startswith("model_reasoning_effort=")
    )
    if json.loads(reasoning) != manifest["model"]["reasoning_effort"]:
        raise ValueError("run-set command has the wrong reasoning effort")


def load_run_set(
    runs_root: Path, work_root: Path, *, environment_lock_sha256: str,
    environment: dict[str, Any],
) -> dict[str, Any]:
    _require_isolated_work_root(work_root)
    path = runs_root / "run-set.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvalManifestError("missing or malformed run-set.json") from exc
    if not isinstance(value, dict):
        raise EvalManifestError("run set must be an object")
    unhashed = {key: item for key, item in value.items() if key != "run_set_sha256"}
    if (value.get("schema_version") != "hlsgraph.agent_eval.run_set.v1"
            or value.get("suite_asset_sha256") != asset_digest()
            or value.get("evaluation_harness_sha256") != harness_digest()
            or value.get("environment_lock_sha256") != environment_lock_sha256
            or value.get("runs_root")
            != Path(os.path.abspath(os.fspath(runs_root))).as_posix()
            or value.get("timeout_seconds")
            != load_manifest()["codex_cli"]["timeout_seconds"]
            or re.fullmatch(r"[0-9a-f]{32}", str(value.get("batch_id", ""))) is None
            or value.get("run_set_sha256") != sha256_bytes(canonical_json(unhashed))):
        raise EvalManifestError("run set has a stale or invalid identity")
    try:
        _validate_permission_canary(
            value.get("permission_canary"),
            environment["runtime_identity"]["sandbox_boundary"],
            runs_root=runs_root,
        )
    except RuntimeError as exc:
        raise EvalManifestError(str(exc)) from exc
    cells = value.get("cells")
    if not isinstance(cells, list):
        raise EvalManifestError("run set lacks cells")
    expected_records = {
        item["run_id"]: item for item in build_run_plan()
    }
    actual = {item.get("run_id"): item for item in cells if isinstance(item, dict)}
    if len(actual) != len(cells) or set(actual) != set(expected_records):
        raise EvalManifestError("run set is not the exact frozen 192-cell matrix")
    if [item.get("run_id") for item in cells] != sorted(expected_records):
        raise EvalManifestError("run-set cells are not in their canonical order")
    for run_id, record in expected_records.items():
        cell = actual[run_id]
        if any(cell.get(key) != value for key, value in record.items()):
            raise EvalManifestError(f"run-set cell is relabelled: {run_id}")
        workspace = environment.get("workspaces", {}).get(
            f"{record['arm']}/{record['corpus_id']}"
        )
        if (not isinstance(workspace, dict)
                or cell.get("workspace_identity_sha256")
                != workspace.get("workspace_identity_sha256")):
            raise EvalManifestError(f"run-set cell has a stale workspace: {run_id}")
        expected_challenge = sha256_bytes(canonical_json({
            "domain": "hlsgraph.agent_eval.trace_challenge.v1",
            "batch_id": value["batch_id"],
            "environment_lock_sha256": environment_lock_sha256,
            "run_id": run_id,
        }))
        if cell.get("trace_challenge") != expected_challenge:
            raise EvalManifestError(f"run-set cell has an invalid trace challenge: {run_id}")
        try:
            _verify_boundary_canary(
                work_root, cell.get("boundary_canary"),
                batch_id=value["batch_id"], run_id=run_id,
            )
        except (RuntimeError, EvalManifestError) as exc:
            raise EvalManifestError(f"run-set boundary canary is invalid: {run_id}") from exc
        question = next(item for item in load_questions() if item["id"] == record["question_id"])
        expected_prompt = build_prompt(
            question, record["arm"], trace_challenge=expected_challenge,
        )
        if cell.get("prompt_sha256") != hashlib.sha256(
            expected_prompt.encode("utf-8")
        ).hexdigest():
            raise EvalManifestError(f"run-set cell has a relabelled prompt: {run_id}")
        _validate_execution_contract(
            cell, work_root, environment, runs_root=runs_root,
        )
    expected_dirs = set(expected_records)
    actual_dirs = {item.name for item in runs_root.iterdir() if item.is_dir()}
    if actual_dirs != expected_dirs:
        raise EvalManifestError("runs root is missing cells or contains unexpected directories")
    return value


def score_run(
    run_dir: Path, work_root: Path, *, environment: dict[str, Any],
    environment_lock_sha256: str, expected_cell: dict[str, Any],
    run_set: dict[str, Any],
) -> dict[str, Any]:
    source_bytes, normalized, events = _snapshot_run(run_dir)
    run = normalized["run"]
    question = _validate_run_metadata(
        run, run_dir, expected_cell, run_set, environment_lock_sha256,
        prompt_bytes=source_bytes["prompt.txt"],
    )
    _validate_trace_challenge(normalized.get("answer"), expected_cell)
    workspace = work_root / run["arm"] / run["corpus_id"]
    boundary_canary = _verify_boundary_canary(
        work_root, expected_cell.get("boundary_canary"),
        batch_id=run_set["batch_id"], run_id=run["run_id"],
    )
    verify_prepared_workspace(environment, work_root, run["arm"], run["corpus_id"])
    verify_workspace_corpus(workspace, run["corpus_id"])
    trace_policy = validate_trace_policy(
        events, arm=run["arm"], workspace=workspace,
        boundary_canary=boundary_canary,
    )
    score = score_answer(question, normalized["answer"], workspace, require_bound=True)
    verify_workspace_corpus(workspace, run["corpus_id"])
    verify_prepared_workspace(environment, work_root, run["arm"], run["corpus_id"])
    if _verify_boundary_canary(
        work_root, expected_cell.get("boundary_canary"),
        batch_id=run_set["batch_id"], run_id=run["run_id"],
    ) != boundary_canary:
        raise ValueError("boundary canary changed during scoring")
    usage = normalized.get("usage", {})
    thread_ids = normalized.get("thread_ids")
    if not isinstance(thread_ids, list) or len(thread_ids) != 1:
        raise ValueError("Codex trace must contain exactly one thread.started identity")
    source_hashes = {
        name: sha256_bytes(data) for name, data in sorted(source_bytes.items())
    }
    run_source_sha256 = sha256_bytes(canonical_json(source_hashes))
    usage = _validate_terminal_usage(usage)
    for name, expected_hash in source_hashes.items():
        current = _stable_run_bytes(
            run_dir / name,
            max_bytes=(256 * 1024 * 1024 if name == "codex.jsonl"
                       else 64 * 1024 * 1024),
        )
        if sha256_bytes(current) != expected_hash:
            raise ValueError(f"run evidence changed during scoring: {name}")
    return {
        "schema_version": "hlsgraph.agent_eval.score.v1",
        "suite_asset_sha256": asset_digest(),
        "evaluation_harness_sha256": harness_digest(),
        "environment_lock_sha256": environment_lock_sha256,
        "batch_id": run_set["batch_id"],
        "run_set_sha256": run_set["run_set_sha256"],
        "run_contract_sha256": expected_cell["run_contract_sha256"],
        "workspace_identity_sha256": expected_cell["workspace_identity_sha256"],
        "thread_id": thread_ids[0],
        "source_hashes": source_hashes,
        "trace_sha256": source_hashes["codex.jsonl"],
        "run_source_sha256": run_source_sha256,
        "run_id": run["run_id"],
        "question_id": run["question_id"],
        "corpus_id": run["corpus_id"],
        "category": run["category"],
        "arm": run["arm"],
        "repetition": run["repetition"],
        "execution_index": run["execution_index"],
        **score,
        "trace_policy": trace_policy,
        "tool_calls": normalized["tool_calls"],
        "file_reads": normalized["file_reads"],
        "file_read_semantics": normalized["file_read_semantics"],
        "input_tokens": usage["input_tokens"],
        "cached_input_tokens": usage["cached_input_tokens"],
        "output_tokens": usage["output_tokens"],
        "total_tokens": usage["total_tokens"],
        "wall_time_seconds": run.get("wall_time_seconds", 0.0),
        "timeout_seconds": run["timeout_seconds"],
        "timed_out": bool(run.get("timed_out")),
        "returncode": run.get("returncode"),
    }


def _validate_terminal_usage(usage: Any) -> dict[str, int]:
    if not isinstance(usage, dict):
        raise ValueError("Codex terminal trace lacks usage")
    keys = (
        "input_tokens", "cached_input_tokens", "output_tokens", "total_tokens",
    )
    for key in keys:
        if key not in usage:
            raise ValueError(f"Codex terminal trace lacks {key}")
        value = usage[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Codex trace has an invalid {key} value")
    if usage["total_tokens"] <= 0:
        raise ValueError("Codex terminal total token count must be positive")
    if usage["cached_input_tokens"] > usage["input_tokens"]:
        raise ValueError("Codex cached input token count exceeds input tokens")
    if usage["total_tokens"] < usage["input_tokens"] + usage["output_tokens"]:
        raise ValueError("Codex total token count is smaller than its components")
    return {key: usage[key] for key in keys}


def _validate_trace_challenge(answer: Any, expected_cell: dict[str, Any]) -> None:
    challenge = expected_cell.get("trace_challenge")
    if re.fullmatch(r"[0-9a-f]{64}", str(challenge or "")) is None:
        raise ValueError("run-set cell has no valid trace challenge")
    marker = f"eval-context:{challenge}"
    uncertainties = answer.get("uncertainties") if isinstance(answer, dict) else None
    if (not isinstance(uncertainties, list) or not uncertainties
            or uncertainties[-1] != marker or uncertainties.count(marker) != 1):
        raise ValueError("Codex trace is not bound to this frozen run-set cell")


def score_runs(runs_root: Path, work_root: Path) -> list[dict[str, Any]]:
    environment_lock = work_root / "environment.lock.json"
    environment = load_environment_lock(environment_lock)
    verify_evaluation_checkout(environment)
    environment_lock_sha256 = sha256_file(environment_lock)
    run_set = load_run_set(
        runs_root, work_root, environment_lock_sha256=environment_lock_sha256,
        environment=environment,
    )
    cells = {item["run_id"]: item for item in run_set["cells"]}
    output: list[dict[str, Any]] = []
    for run_id in sorted(cells):
        run_dir = runs_root / run_id
        run_json = run_dir / "run.json"
        try:
            output.append(score_run(
                run_dir, work_root,
                environment=environment,
                environment_lock_sha256=environment_lock_sha256,
                expected_cell=cells[run_id], run_set=run_set,
            ))
        except Exception as exc:  # preserve failed cells in the public result table
            try:
                metadata = json.loads(run_json.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                metadata = {}
            expected = cells[run_id]
            output.append({
                "schema_version": "hlsgraph.agent_eval.score.v1",
                "suite_asset_sha256": asset_digest(),
                "evaluation_harness_sha256": harness_digest(),
                "environment_lock_sha256": environment_lock_sha256,
                "batch_id": run_set["batch_id"],
                "run_set_sha256": run_set["run_set_sha256"],
                "run_contract_sha256": expected["run_contract_sha256"],
                "workspace_identity_sha256": expected["workspace_identity_sha256"],
                "run_id": expected["run_id"],
                "question_id": expected["question_id"],
                "corpus_id": expected["corpus_id"],
                "category": expected["category"],
                "arm": expected["arm"],
                "repetition": expected["repetition"],
                "execution_index": expected["execution_index"],
                "evidence_grounded_correctness": 0.0,
                "citation_precision": 0.0,
                "unsupported_claims": 1,
                "fabricated_truth_count": 0,
                "tool_calls": 0,
                "file_reads": 0,
                "file_read_semantics": "source_access_tool_calls",
                "timeout_seconds": expected["timeout_seconds"],
                "total_tokens": 0,
                "wall_time_seconds": metadata.get("wall_time_seconds", 0.0),
                "timed_out": bool(metadata.get("timed_out")),
                "returncode": metadata.get("returncode"),
                "parse_error": f"{type(exc).__name__}: {exc}",
            })
    successful = [item for item in output if not item.get("parse_error")]
    for key in ("thread_id", "trace_sha256", "run_source_sha256"):
        values = [item[key] for item in successful]
        if len(values) != len(set(values)):
            raise EvalManifestError(f"multiple cells reuse the same raw {key}")
    batch_sources = [
        {"run_id": item["run_id"], "run_source_sha256": item.get("run_source_sha256")}
        for item in output
    ]
    batch_sha256 = sha256_bytes(canonical_json(batch_sources))
    for item in output:
        item["run_batch_sha256"] = batch_sha256
    return output


def render_score_rows(rows: Iterable[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
        for row in rows
    ).encode("utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = score_runs(args.runs_root, args.work_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data = render_score_rows(rows)
    args.output.write_bytes(data)
    print(json.dumps({
        "rows": len(rows), "output": str(args.output),
        "scores_sha256": sha256_bytes(data),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
