from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil

import pytest

from tools import apply_knowledge_review_suite_attestation as attestation
from tools import audit_release
from tools import execute_knowledge_review_suite as executor
from tools import knowledge_review_shards as shards
from tools import knowledge_review_suite_cache as suite_cache
from tools import knowledge_review_suite_replay as suite_replay
from tools import run_knowledge_review_suite as suite
from tools import run_knowledge_review as review
from tools import seal_knowledge_review_suite as seal
from tests.test_knowledge_review_release_gate import (
    _fixture_boundary_contract,
    reviewed_release_root,
)


class _FixtureTokenizer:
    """Deterministic stand-in for the separately pinned production tokenizer."""

    @staticmethod
    def encode(text: str) -> list[int]:
        return [0] * max(1, (len(text) + 31) // 32)


def _private_dir(path: Path) -> None:
    missing: list[Path] = []
    current = path
    while not current.exists():
        missing.append(current)
        current = current.parent
    for directory in reversed(missing):
        directory.mkdir(mode=0o700)
        if os.name != "nt":
            directory.chmod(0o700)
    if os.name != "nt":
        path.chmod(0o700)


def _approved_shard(projection: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": suite.SHARD_RESULT_SCHEMA_VERSION,
        "protocol_id": projection["protocol_id"],
        "review_snapshot_sha256": projection["review_snapshot_sha256"],
        "shard_plan_sha256": projection["shard_plan_sha256"],
        "shard_id": projection["shard_id"],
        "citation_evidence_surface_sha256": projection[
            "citation_evidence_surface_sha256"
        ],
        "assertion_results": [{
            "assertion_id": assertion_id,
            "verdict": "verified",
            "issues": [],
        } for assertion_id in projection["assertion_ids"]],
        "citation_results": [{
            "reference_id": row["reference_id"],
            "reference_surface_sha256": row["reference_surface_sha256"],
            "verdict": "verified",
            "exact_locator_inspected": True,
            "declared_version_matched": True,
            "declared_section_matched": True,
            "paraphrase_supported": True,
            "applicability_not_broader": True,
            "issues": [],
        } for row in projection["rule_references"]],
        "approved": True,
        "issues": [],
        "summary": "approved_no_issues",
    }


def _raw_stream(
    cache: review.ReviewCache,
    projection: dict[str, object],
    result: dict[str, object],
    *,
    thread_id: str,
) -> bytes:
    rows: list[dict[str, object]] = [
        {"type": "thread.started", "thread_id": thread_id},
        {"type": "turn.started"},
    ]
    commands = executor.assigned_chunk_material(cache).commands
    for index, command in enumerate(commands, 1):
        inner = review._unwrap_codex_shell_event_command(command)
        output, _operations, _citation = review._expected_command(cache, inner)
        item = {
            "id": f"command-{index:04d}",
            "type": "command_execution",
            "command": command,
        }
        rows.append({"type": "item.started", "item": item})
        rows.append({
            "type": "item.completed",
            "item": {
                **item,
                "status": "completed",
                "exit_code": 0,
                "aggregated_output": output,
            },
        })
    rows.extend([
        {
            "type": "item.completed",
            "item": {
                "id": "agent-message-final",
                "type": "agent_message",
                "text": json.dumps(result, sort_keys=True),
            },
        },
        {
            "type": "turn.completed",
            "usage": {
                "input_tokens": 1000,
                "cached_input_tokens": 250,
                "output_tokens": 100,
                "reasoning_output_tokens": 25,
            },
        },
    ])
    return review._canonical_jsonl(rows)


def _actual_command(
    root: Path, cache: review.ReviewCache, codex: Path,
) -> tuple[list[str], str]:
    filesystem = {
        ":minimal": "read",
        str(cache.root.resolve()): "read",
        str(codex.parent.resolve()): "read",
    }
    profile = [
        f"default_permissions={review._toml(review.PERMISSION_PROFILE)}",
        f"permissions.{review.PERMISSION_PROFILE}.network.enabled=false",
        f"permissions.{review.PERMISSION_PROFILE}.filesystem="
        + review._toml_inline_table(filesystem),
        'web_search="disabled"',
    ]
    command = executor._actual_shard_command(
        root=root,
        cache_root=cache.root,
        codex=str(codex.resolve()),
        profile_values=profile,
    )
    digest = executor.validate_actual_shard_command(
        command,
        root=root,
        cache_root=cache.root,
        codex=codex,
    )
    return command, digest


def _build_v6_suite(
    root: Path,
) -> tuple[Path, dict[str, object], dict[str, object]]:
    evidence_root = root.parent / "suite-evidence"
    _private_dir(evidence_root)
    citation_audit = json.loads(
        (root / review.CITATION_AUDIT_PATH).read_text(encoding="utf-8")
    )
    plan = shards.build_shard_plan(citation_audit)
    plan_sha256 = shards.shard_plan_sha256(plan)
    snapshots = {
        protocol_id: review.freeze_review_snapshot(root, protocol_id)
        for protocol_id in executor.PROTOCOL_ORDER
    }

    runtime_parent = root.parent / "codex-runtime"
    _private_dir(runtime_parent)
    codex = runtime_parent / "codex"
    codex.write_bytes(b"codex")
    if os.name != "nt":
        codex.chmod(0o500)

    full_caches: dict[str, review.ReviewCache] = {}
    for protocol_id in executor.PROTOCOL_ORDER:
        label = executor.PROTOCOL_LABELS[protocol_id]
        parent = evidence_root / "full" / label / "cache-parent"
        _private_dir(parent)
        source = root.parent / f"{label}.cache"
        target = parent / "cache"
        shutil.copytree(source, target, copy_function=shutil.copy2)
        full_caches[protocol_id] = review.load_review_cache(
            target, snapshots[protocol_id],
        )

    protocols: dict[str, executor.ProtocolExecution] = {}
    boundaries: dict[str, list[dict[str, object]]] = {}
    process_hashes: dict[str, list[str]] = {}
    runtime_manifest: dict[str, object] | None = None
    for protocol_index, protocol_id in enumerate(executor.PROTOCOL_ORDER):
        label = executor.PROTOCOL_LABELS[protocol_id]
        snapshot = snapshots[protocol_id]
        full_cache = full_caches[protocol_id]
        base_prompt = (root / executor.SHARD_PROMPT_PATHS[protocol_id]).read_bytes()
        envelopes: list[dict[str, object]] = []
        protocol_boundaries: list[dict[str, object]] = []
        protocol_process_hashes: list[str] = []
        for shard_index, shard_id in enumerate(shards.SHARD_ORDER):
            projection = suite.project_shard_manifest(
                protocol_id=protocol_id,
                snapshot_inventory=snapshot.inventory(),
                cache_manifest=full_cache.manifest,
                plan=plan,
                shard_id=shard_id,
            )
            invocation = evidence_root / "invocations" / label / shard_id
            cache_parent = invocation / "cache-parent"
            evidence_parent = invocation / "evidence"
            _private_dir(cache_parent)
            projected_cache = suite_cache.materialize_shard_cache(
                full_cache, projection, cache_parent / "cache",
            )
            _private_dir(evidence_parent)
            result = _approved_shard(projection)
            raw = _raw_stream(
                projected_cache,
                projection,
                result,
                thread_id=(
                    f"thread-suite-{protocol_index + 1}-{shard_index + 1}"
                ),
            )
            sanitized = executor.sanitize_shard_raw_stream(raw, projected_cache)
            material = executor.assigned_chunk_material(projected_cache)
            executor.require_exact_command_inventory(raw, material.commands)
            replayed = suite_replay.replay_shard_raw_review(
                raw,
                cache=projected_cache,
                shard_manifest=projection,
            )
            prompt = suite.build_shard_prompt(
                base_protocol_text=base_prompt,
                snapshot_inventory=snapshot.inventory(),
                plan_sha256=plan_sha256,
                shard_projection=projection,
            )
            budget = executor.enforce_shard_token_budget(
                prompt=prompt,
                material=material,
                tokenizer=_FixtureTokenizer(),
            )
            boundary = _fixture_boundary_contract(projected_cache)
            if runtime_manifest is None:
                runtime_manifest = boundary["runtime_manifest"]
            else:
                assert boundary["runtime_manifest"] == runtime_manifest
            command, command_sha256 = _actual_command(
                root, projected_cache, codex,
            )
            outcome = executor.ProcessOutcome(0, raw, b"")
            process = executor.build_process_evidence(
                command=command,
                cwd=projected_cache.root,
                prompt=prompt,
                outcome=outcome,
                command_sha256=command_sha256,
            )
            envelope = executor.build_invocation_envelope(
                replayed=replayed,
                shard_manifest=projection,
                cache=projected_cache,
                prompt=prompt,
                boundary_contract=boundary,
                runtime_manifest_sha256=str(runtime_manifest["sha256"]),
                token_budget=budget,
                assigned_chunk_inventory_sha256=material.inventory_sha256,
                replay_contract_digest=executor.replay_contract_sha256(root),
                sanitized_output_sha256=hashlib.sha256(sanitized).hexdigest(),
                command_sha256=command_sha256,
            )
            process_bytes = review._canonical_json(process)
            for name, payload in (
                (executor.RAW_STREAM_PATH, raw),
                (executor.SANITIZED_STREAM_PATH, sanitized),
                (executor.RAW_STDERR_PATH, b""),
                (executor.STDERR_PATH, b""),
                (executor.PROCESS_EVIDENCE_PATH, process_bytes),
                (
                    executor.INVOCATION_ENVELOPE_PATH,
                    review._canonical_json(envelope),
                ),
            ):
                review._write_private(evidence_parent / name, payload)
            envelopes.append(envelope)
            protocol_boundaries.append(boundary)
            protocol_process_hashes.append(
                hashlib.sha256(process_bytes).hexdigest()
            )
        assert runtime_manifest is not None
        boundaries[protocol_id] = protocol_boundaries
        process_hashes[protocol_id] = protocol_process_hashes
        protocols[protocol_id] = executor._build_protocol_execution(
            root=root,
            protocol_id=protocol_id,
            snapshot=snapshot,
            full_cache=full_cache,
            plan=plan,
            citation_audit=citation_audit,
            invocations=envelopes,
            runtime_manifest_sha256=str(runtime_manifest["sha256"]),
        )

    semantic = protocols[shards.SEMANTIC_PROTOCOL_ID]
    adversarial = protocols[shards.ADVERSARIAL_PROTOCOL_ID]
    pair_seal = seal.validate_suite_pair(
        semantic_receipt=semantic.receipt,
        adversarial_receipt=adversarial.receipt,
        semantic_result=semantic.aggregate_result,
        adversarial_result=adversarial.aggregate_result,
        plan=plan,
        citation_audit=citation_audit,
    )
    assert runtime_manifest is not None
    manifest = executor.build_suite_evidence_manifest(
        runtime_manifest=runtime_manifest,
        protocols=protocols,
        boundary_contracts=boundaries,
        process_evidence_sha256s=process_hashes,
    )
    review._write_private(
        evidence_root / executor.SUITE_EVIDENCE_PATH,
        review._canonical_json(manifest),
    )
    review._write_private(
        evidence_root / executor.PAIR_SEAL_PATH,
        review._canonical_json(pair_seal),
    )
    for execution in (semantic, adversarial):
        files = review.PROTOCOL_FILES[execution.protocol_id]
        (root / files["result"]).write_bytes(
            review._canonical_json(execution.aggregate_result)
        )
        (root / files["trace"]).write_bytes(execution.trace_bytes)
        (root / files["receipt"]).write_bytes(
            review._canonical_json(execution.receipt)
        )
    return evidence_root, runtime_manifest, {
        "semantic_receipt": semantic.receipt,
        "adversarial_receipt": adversarial.receipt,
        "semantic_result": semantic.aggregate_result,
        "adversarial_result": adversarial.aggregate_result,
        "suite_pair_seal": pair_seal,
        "plan": plan,
        "citation_audit": citation_audit,
        "semantic_snapshot": semantic.snapshot,
        "adversarial_snapshot": adversarial.snapshot,
    }


def _audit(root: Path, evidence_root: Path) -> list[str]:
    return audit_release._audit_knowledge_review_release_gate(
        root,
        semantic_review=root / audit_release.SEMANTIC_REVIEW_PATH,
        adversarial_review=root / audit_release.ADVERSARIAL_REVIEW_PATH,
        suite_evidence=evidence_root,
    )


def test_v6_release_gate_replays_six_raw_streams_and_rejects_tamper(
    reviewed_release_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = reviewed_release_root
    evidence_root, runtime_manifest, attestation_inputs = _build_v6_suite(root)
    monkeypatch.setattr(
        shards,
        "load_verified_tokenizer",
        lambda: _FixtureTokenizer(),
    )
    monkeypatch.setattr(
        review,
        "_freeze_runtime_manifest",
        lambda _codex: runtime_manifest,
    )

    replayed = audit_release.replay_knowledge_review_suite_evidence(
        root,
        semantic_review=root / audit_release.SEMANTIC_REVIEW_PATH,
        adversarial_review=root / audit_release.ADVERSARIAL_REVIEW_PATH,
        suite_evidence=evidence_root,
    )
    assert replayed.verified is True
    assert replayed.plan == attestation_inputs["plan"]
    assert replayed.citation_audit == attestation_inputs["citation_audit"]
    assert replayed.attestation_material is not None
    assert len(replayed.attestation_material["reviewers"]) == 6
    pre_attestation_issues = _audit(root, evidence_root)
    assert any("stale suite" in issue for issue in pre_attestation_issues)
    attestation.finalize_attestation(root, suite_evidence=evidence_root)
    assert _audit(root, evidence_root) == []

    evidence_parent = (
        evidence_root
        / "invocations"
        / "semantic"
        / shards.SHARD_ORDER[0]
        / "evidence"
    )
    stderr = evidence_parent / executor.STDERR_PATH
    stderr_original = stderr.read_bytes()
    stderr.write_bytes(b"not-the-redacted-derivative")
    issues = _audit(root, evidence_root)
    assert any("exact raw-stderr derivative" in issue for issue in issues)
    stderr.write_bytes(stderr_original)

    raw = evidence_parent / executor.RAW_STREAM_PATH
    original = raw.read_bytes()
    raw.write_bytes(original + b"{}\n")
    issues = _audit(root, evidence_root)
    assert any("raw output hash differs" in issue for issue in issues)
