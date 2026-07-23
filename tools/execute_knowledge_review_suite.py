#!/usr/bin/env python3
"""Execute the fixed three-shard knowledge-review suite, fail closed.

This is the only operational layer for the sharded review contract.  The
lower-level modules remain deliberately pure: they define the shard plan,
project physically isolated caches, replay raw Codex JSONL, and seal a pair of
protocol receipts.  This module composes those pieces without weakening the
single-review process boundary.

Formal publication is intentionally narrow: Linux/WSL2, an ext4 clean
checkout, the pinned Codex+bwrap runtime tree, ``gpt-5.6-sol`` at ``medium``,
the pinned tokenizer, and the default process runner are mandatory.  A custom
process runner is accepted only by the testable invocation helper; it can
never publish release artifacts.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import tempfile
from typing import Any, Callable, Mapping, Sequence

from tools import knowledge_review_shards as shard_plan
from tools import knowledge_review_suite_cache as suite_cache
from tools import knowledge_review_suite_replay as suite_replay
from tools import run_knowledge_review as review
from tools import run_knowledge_review_suite as suite
from tools import seal_knowledge_review_suite as suite_seal


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
SHARD_OUTPUT_SCHEMA_PATH = "tools/knowledge_review_shard.schema.json"
SUITE_RECEIPT_SCHEMA_PATH = "tools/knowledge_review_suite_receipt.schema.json"
SUITE_TRACE_SCHEMA_PATH = "tools/knowledge_review_suite_trace.schema.json"
PAIR_SEAL_PATH = "pair-seal.json"
SUITE_EVIDENCE_PATH = "suite-evidence.json"
INVOCATION_ENVELOPE_PATH = "invocation.json"
RAW_STREAM_PATH = "raw.jsonl"
SANITIZED_STREAM_PATH = "sanitized.jsonl"
STDERR_PATH = "stderr.log"
RAW_STDERR_PATH = "stderr.raw.log"
PROCESS_EVIDENCE_PATH = "process.json"

SHARD_PROMPT_PATHS = {
    shard_plan.SEMANTIC_PROTOCOL_ID:
        "tools/knowledge_review_prompts/semantic_shard.md",
    shard_plan.ADVERSARIAL_PROTOCOL_ID:
        "tools/knowledge_review_prompts/adversarial_shard.md",
}
PROTOCOL_LABELS = {
    shard_plan.SEMANTIC_PROTOCOL_ID: "semantic",
    shard_plan.ADVERSARIAL_PROTOCOL_ID: "adversarial",
}
PROTOCOL_ORDER = (
    shard_plan.SEMANTIC_PROTOCOL_ID,
    shard_plan.ADVERSARIAL_PROTOCOL_ID,
)

EXECUTOR_CONTRACT_VERSION = "hlsgraph.knowledge-review.suite-executor.v1"
COMMAND_CONTRACT_VERSION = "hlsgraph.knowledge-review.shard-command.v1"
REPLAY_CONTRACT_VERSION = "hlsgraph.knowledge-review.shard-replay-binding.v1"
CHUNK_INVENTORY_VERSION = "hlsgraph.knowledge-review.shard-chunks.v1"
SUITE_EVIDENCE_SCHEMA_VERSION = "hlsgraph.knowledge-review.suite-evidence.v1"
MAX_CONTROL_FILE_BYTES = 4 * 1024 * 1024
TOKENIZER_CACHE_ENV = "TIKTOKEN_CACHE_DIR"
TOKENIZER_BPE_CACHE_KEY = "fb374d419588a4632f3f557e76b4b70aebbca790"
TOKENIZER_BPE_SHA256 = (
    "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"
)
TOKENIZER_BPE_SIZE = 3_613_922

CONTROL_SURFACE_PATHS = (
    "tools/execute_knowledge_review_suite.py",
    "tools/knowledge_review_shards.py",
    "tools/run_knowledge_review_suite.py",
    "tools/knowledge_review_suite_cache.py",
    "tools/knowledge_review_suite_replay.py",
    "tools/seal_knowledge_review_suite.py",
    "tools/knowledge_review_shard.schema.json",
    "tools/knowledge_review_suite_receipt.schema.json",
    "tools/knowledge_review_suite_trace.schema.json",
    "tools/knowledge_review_prompts/semantic_shard.md",
    "tools/knowledge_review_prompts/adversarial_shard.md",
)


class SuiteExecutionError(RuntimeError):
    """The formal suite cannot continue without weakening its evidence."""


@dataclass(frozen=True)
class ProcessOutcome:
    returncode: int
    stdout: bytes = field(repr=False)
    stderr: bytes = field(repr=False)


@dataclass(frozen=True)
class ChunkMaterial:
    paths: tuple[str, ...]
    commands: tuple[str, ...]
    payloads: tuple[bytes, ...] = field(repr=False)
    inventory: tuple[dict[str, Any], ...]
    inventory_sha256: str


@dataclass(frozen=True)
class ProtocolExecution:
    protocol_id: str
    snapshot: review.ReviewSnapshot = field(repr=False)
    full_cache: review.ReviewCache = field(repr=False)
    invocations: tuple[dict[str, Any], ...] = field(repr=False)
    aggregate_result: dict[str, Any] = field(repr=False)
    trace_bytes: bytes = field(repr=False)
    receipt: dict[str, Any] = field(repr=False)


@dataclass(frozen=True)
class SuiteExecution:
    semantic: ProtocolExecution = field(repr=False)
    adversarial: ProtocolExecution = field(repr=False)
    pair_seal: dict[str, Any]
    evidence_manifest: dict[str, Any] = field(repr=False)


ProcessRunner = Callable[
    [Sequence[str], bytes, Path, Mapping[str, str], int], ProcessOutcome
]


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ) + "\n").encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_object(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value))


def _protocol_label(protocol_id: str) -> str:
    try:
        return PROTOCOL_LABELS[protocol_id]
    except KeyError as exc:
        raise SuiteExecutionError(f"unknown review protocol: {protocol_id!r}") from exc


def _fixed_shard(shard_id: str) -> None:
    if shard_id not in shard_plan.SHARD_ORDER:
        raise SuiteExecutionError(f"unknown review shard: {shard_id!r}")


def canonical_shard_command_argv() -> list[str]:
    """Return the path-neutral exact Codex argv contract.

    Host paths are represented by capability tokens.  Consequently the hash
    proves command semantics and is stable across otherwise identical ext4
    review roots.
    """

    values = ["$CODEX", "--strict-config", "-a", "never"]
    for feature in review.DISABLED_CODEX_FEATURES:
        values.extend(["--disable", feature])
    values.extend([
        "exec",
        "-c", f'default_permissions="{review.PERMISSION_PROFILE}"',
        "-c", f"permissions.{review.PERMISSION_PROFILE}.network.enabled=false",
        "-c", (
            f"permissions.{review.PERMISSION_PROFILE}.filesystem="
            '{":minimal"="read","$CACHE"="read",'
            '"$CODEX_RUNTIME"="read"}'
        ),
        "-c", 'web_search="disabled"',
        "--ignore-user-config", "--ignore-rules", "--ephemeral", "--json",
        "--color", "never", "--skip-git-repo-check", "--model",
        suite_seal.MODEL,
        "-c", f'model_reasoning_effort="{suite_seal.REASONING_EFFORT}"',
        "-c", f"model_context_window={shard_plan.MODEL_CONTEXT_WINDOW_TOKENS}",
        "-c", (
            "model_auto_compact_token_limit="
            f"{shard_plan.AUTO_COMPACT_TOKEN_LIMIT_TOKENS}"
        ),
        "-c", (
            'model_auto_compact_token_limit_scope="'
            f'{shard_plan.AUTO_COMPACT_TOKEN_LIMIT_SCOPE}"'
        ),
        "-c", f"tool_output_token_limit={review.TOOL_OUTPUT_TOKEN_LIMIT}",
        "--output-schema", "$ROOT/" + SHARD_OUTPUT_SCHEMA_PATH,
        "--cd", "$CACHE", "-",
    ])
    return values


def canonical_shard_command_sha256() -> str:
    payload = {
        "schema_version": COMMAND_CONTRACT_VERSION,
        "model": suite_seal.MODEL,
        "reasoning_effort": suite_seal.REASONING_EFFORT,
        "argv": canonical_shard_command_argv(),
    }
    return _sha256_object(payload)


def normalize_actual_shard_command(
    command: Sequence[str], *, root: Path, cache_root: Path,
    codex: Path,
) -> list[str]:
    """Replace the three runtime capabilities in an actual argv.

    Only exact, independently reconstructed values are replaced.  A flag,
    profile value, ordering, path, or quoting drift therefore survives the
    normalization and fails the byte-semantic comparison with
    :func:`canonical_shard_command_argv`.
    """

    resolved_root = root.resolve(strict=True)
    resolved_cache = cache_root.resolve(strict=True)
    resolved_codex = codex.resolve(strict=True)
    runtime_root = resolved_codex.parent.resolve(strict=True)
    actual_filesystem = (
        f"permissions.{review.PERMISSION_PROFILE}.filesystem="
        + review._toml_inline_table({
            ":minimal": "read",
            str(resolved_cache): "read",
            str(runtime_root): "read",
        })
    )
    canonical_filesystem = (
        f"permissions.{review.PERMISSION_PROFILE}.filesystem="
        '{":minimal"="read","$CACHE"="read",'
        '"$CODEX_RUNTIME"="read"}'
    )
    replacements = {
        str(resolved_codex): "$CODEX",
        str((resolved_root / SHARD_OUTPUT_SCHEMA_PATH).resolve(strict=True)):
            "$ROOT/" + SHARD_OUTPUT_SCHEMA_PATH,
        str(resolved_cache): "$CACHE",
        actual_filesystem: canonical_filesystem,
    }
    return [replacements.get(str(value), str(value)) for value in command]


def validate_actual_shard_command(
    command: Sequence[str], *, root: Path, cache_root: Path,
    codex: Path,
) -> str:
    normalized = normalize_actual_shard_command(
        command, root=root, cache_root=cache_root, codex=codex,
    )
    expected = canonical_shard_command_argv()
    if normalized != expected:
        raise SuiteExecutionError(
            "actual Codex argv differs from the closed shard command contract"
        )
    payload = {
        "schema_version": COMMAND_CONTRACT_VERSION,
        "model": suite_seal.MODEL,
        "reasoning_effort": suite_seal.REASONING_EFFORT,
        "argv": normalized,
    }
    digest = _sha256_object(payload)
    if digest != canonical_shard_command_sha256():  # defensive consistency
        raise SuiteExecutionError("normalized Codex argv hash is inconsistent")
    return digest


def _actual_shard_command(
    *, root: Path, cache_root: Path, codex: str, profile_values: Sequence[str],
) -> list[str]:
    command = [codex, "--strict-config", "-a", "never"]
    for feature in review.DISABLED_CODEX_FEATURES:
        command.extend(["--disable", feature])
    command.append("exec")
    for value in profile_values:
        command.extend(["-c", value])
    command.extend([
        "--ignore-user-config", "--ignore-rules", "--ephemeral", "--json",
        "--color", "never", "--skip-git-repo-check", "--model",
        suite_seal.MODEL,
        "-c", f'model_reasoning_effort="{suite_seal.REASONING_EFFORT}"',
        "-c", f"model_context_window={shard_plan.MODEL_CONTEXT_WINDOW_TOKENS}",
        "-c", (
            "model_auto_compact_token_limit="
            f"{shard_plan.AUTO_COMPACT_TOKEN_LIMIT_TOKENS}"
        ),
        "-c", (
            'model_auto_compact_token_limit_scope="'
            f'{shard_plan.AUTO_COMPACT_TOKEN_LIMIT_SCOPE}"'
        ),
        "-c", f"tool_output_token_limit={review.TOOL_OUTPUT_TOKEN_LIMIT}",
        "--output-schema", str((root / SHARD_OUTPUT_SCHEMA_PATH).resolve()),
        "--cd", str(cache_root.resolve()), "-",
    ])
    return command


def _read_stable_checkout_file(root: Path, relative: str) -> bytes:
    """Read one checkout file through a no-follow directory-fd walk."""

    path = PurePosixPath(relative)
    if (path.is_absolute() or not path.parts
            or any(part in {"", ".", ".."} for part in path.parts)):
        raise SuiteExecutionError("suite control path is not repository-relative")
    resolved_root = root.resolve(strict=True)
    if os.name == "nt":  # formal execution is rejected on Windows
        target = resolved_root / path
        before = target.lstat()
        if (review._is_link_like(target) or not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1 or before.st_size > MAX_CONTROL_FILE_BYTES):
            raise SuiteExecutionError(f"suite control file is unsafe: {relative}")
        payload = target.read_bytes()
        after = target.lstat()
        identity = lambda value: (
            value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns,
            value.st_ctime_ns, value.st_mode, value.st_nlink,
        )
        if identity(before) != identity(after) or len(payload) != before.st_size:
            raise SuiteExecutionError(f"suite control file changed: {relative}")
        return payload

    nofollow = getattr(os, "O_NOFOLLOW", 0)
    cloexec = getattr(os, "O_CLOEXEC", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    descriptors: list[int] = []
    try:
        current = os.open(resolved_root, os.O_RDONLY | directory | nofollow | cloexec)
        descriptors.append(current)
        for component in path.parts[:-1]:
            current = os.open(
                component, os.O_RDONLY | directory | nofollow | cloexec,
                dir_fd=current,
            )
            descriptors.append(current)
        fd = os.open(
            path.parts[-1], os.O_RDONLY | nofollow | cloexec,
            dir_fd=current,
        )
        descriptors.append(fd)
        before = os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                or before.st_size < 0 or before.st_size > MAX_CONTROL_FILE_BYTES):
            raise SuiteExecutionError(f"suite control file is unsafe: {relative}")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(fd, min(remaining, 128 * 1024))
            if not chunk:
                raise SuiteExecutionError(f"suite control file truncated: {relative}")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(fd, 1):
            raise SuiteExecutionError(f"suite control file grew: {relative}")
        after = os.fstat(fd)
        current_entry = os.stat(
            path.parts[-1], dir_fd=descriptors[-2], follow_symlinks=False,
        )
        identity = lambda value: (
            value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns,
            value.st_ctime_ns, value.st_mode, value.st_nlink,
        )
        if identity(before) != identity(after) or identity(after) != identity(
            current_entry
        ):
            raise SuiteExecutionError(f"suite control file changed: {relative}")
        return b"".join(chunks)
    except OSError as exc:
        raise SuiteExecutionError(
            f"cannot read stable suite control file: {relative}"
        ) from exc
    finally:
        for fd in reversed(descriptors):
            try:
                os.close(fd)
            except OSError:
                pass


def freeze_control_surface(root: Path) -> dict[str, str]:
    """Hash the executor-only TCB which is outside the legacy snapshot."""

    resolved = root.resolve(strict=True)
    result: dict[str, str] = {}
    for relative in CONTROL_SURFACE_PATHS:
        result[relative] = _sha256_bytes(
            _read_stable_checkout_file(resolved, relative)
        )
    return result


def assigned_chunk_material(cache: review.ReviewCache) -> ChunkMaterial:
    """Return the exact one-command-per-chunk budget and replay inventory."""

    targets = review._cache_targets(cache)
    if not targets:
        raise SuiteExecutionError("review shard has no assigned chunks")
    paths = tuple(sorted(targets))
    commands = tuple(
        review._codex_shell_event_command(f"head -n 100000000 {path}")
        for path in paths
    )
    payloads: list[bytes] = []
    inventory: list[dict[str, Any]] = []
    for path in paths:
        target = targets[path]
        chunk = target.get("chunk")
        if not isinstance(chunk, dict):
            raise SuiteExecutionError(f"assigned chunk metadata is malformed: {path}")
        payload = review._read_private_cache_file(cache.root, path)
        if (_sha256_bytes(payload) != chunk.get("sha256")
                or len(payload) != chunk.get("size")):
            raise SuiteExecutionError(f"assigned chunk bytes are stale: {path}")
        payload.decode("utf-8", errors="strict")
        payloads.append(payload)
        inventory.append({
            "kind": target.get("target_kind"),
            "path": path,
            "sha256": chunk.get("sha256"),
            "size": chunk.get("size"),
            "byte_start": chunk.get("byte_start"),
            "byte_end": chunk.get("byte_end"),
            "original_sha256": chunk.get("original_sha256"),
            "original_size": chunk.get("original_size"),
        })
    payload = {
        "schema_version": CHUNK_INVENTORY_VERSION,
        "chunks": inventory,
    }
    return ChunkMaterial(
        paths=paths, commands=commands, payloads=tuple(payloads),
        inventory=tuple(inventory), inventory_sha256=_sha256_object(payload),
    )


def enforce_shard_token_budget(
    *, prompt: bytes, material: ChunkMaterial, tokenizer: Any,
) -> shard_plan.TokenBudget:
    return shard_plan.enforce_token_budget(
        prompt=prompt, chunks=material.payloads, commands=material.commands,
        tokenizer=tokenizer,
    )


def _completed_command_inventory(raw_bytes: bytes) -> list[str]:
    events = review._strict_jsonl(raw_bytes, label="raw sharded Codex stream")
    commands: list[str] = []
    for event in events:
        if event.get("type") != "item.completed":
            continue
        item = event.get("item")
        if not isinstance(item, dict) or item.get("type") != "command_execution":
            continue
        command = item.get("command")
        if not isinstance(command, str):
            raise SuiteExecutionError("completed shard command has no literal command")
        commands.append(command)
    return commands


def require_exact_command_inventory(
    raw_bytes: bytes, expected_commands: Sequence[str],
) -> None:
    """Bind the token count to the commands actually replayed by Codex."""

    observed = _completed_command_inventory(raw_bytes)
    expected = list(expected_commands)
    if (len(observed) != len(expected)
            or sorted(observed) != sorted(expected)
            or len(observed) != len(set(observed))):
        raise SuiteExecutionError(
            "Codex command inventory differs from the one-command-per-chunk budget"
        )


def sanitize_shard_raw_stream(
    raw_bytes: bytes, cache: review.ReviewCache,
) -> bytes:
    """Verify command output and remove citation text from retained JSONL."""

    events = review._strict_jsonl(raw_bytes, label="raw sharded Codex review stream")
    for event in events:
        if event.get("type") != "item.completed":
            continue
        item = event.get("item")
        if not isinstance(item, dict) or item.get("type") != "command_execution":
            continue
        if (item.get("status") != "completed"
                or type(item.get("exit_code")) is not int
                or item["exit_code"] != 0):
            raise SuiteExecutionError("review command did not complete successfully")
        key, output = review._command_output(item)
        inner_command = review._unwrap_codex_shell_event_command(
            str(item.get("command", "")),
        )
        expected, _operations, citation_content = review._expected_command(
            cache, inner_command,
        )
        if output != expected:
            raise SuiteExecutionError(
                "review command output differs from frozen shard evidence"
            )
        if citation_content:
            item[key] = review._citation_marker(expected)
    sanitized = review._canonical_jsonl(events)
    for citation in cache.manifest.get("citations", []):
        for chunk in citation.get("inspection_chunks", []):
            relative = chunk.get("path") if isinstance(chunk, dict) else None
            if not isinstance(relative, str):
                raise SuiteExecutionError("shard citation chunk path is malformed")
            payload = review._read_private_cache_file(cache.root, relative)
            needles = {payload}
            try:
                text = payload.decode("utf-8", errors="strict")
            except UnicodeDecodeError:
                pass
            else:
                needles.add(json.dumps(text, ensure_ascii=False)[1:-1].encode("utf-8"))
            if payload and any(needle and needle in sanitized for needle in needles):
                raise SuiteExecutionError(
                    "sanitized shard stream retains private citation content"
                )
    return sanitized


def _redact_shard_payloads(data: bytes, cache: review.ReviewCache) -> bytes:
    result = data
    for path in sorted(review._cache_targets(cache)):
        payload = review._read_private_cache_file(cache.root, path)
        if not payload:
            continue
        marker = f"[redacted-cache-{_sha256_bytes(payload)}]".encode("ascii")
        text = payload.decode("utf-8", errors="strict")
        representations = {payload}
        for ensure_ascii in (False, True):
            escaped = json.dumps(text, ensure_ascii=ensure_ascii)[1:-1].encode(
                "utf-8",
            )
            if escaped:
                representations.add(escaped)
        for representation in sorted(
            representations, key=lambda value: (-len(value), value),
        ):
            result = result.replace(representation, marker)
    return result


def replay_contract_sha256(root: Path) -> str:
    relative = "tools/knowledge_review_suite_replay.py"
    source_hash = _sha256_bytes(_read_stable_checkout_file(root, relative))
    return _sha256_object({
        "schema_version": REPLAY_CONTRACT_VERSION,
        "trace_schema_version": suite_replay.SHARD_TRACE_SCHEMA_VERSION,
        "source_sha256": source_hash,
    })


def build_invocation_envelope(
    *, replayed: suite_replay.ShardReviewReplay,
    shard_manifest: Mapping[str, Any], cache: review.ReviewCache,
    prompt: bytes, boundary_contract: Mapping[str, Any],
    runtime_manifest_sha256: str, token_budget: shard_plan.TokenBudget,
    assigned_chunk_inventory_sha256: str, replay_contract_digest: str,
    sanitized_output_sha256: str, command_sha256: str,
) -> dict[str, Any]:
    """Build the closed input accepted by the deterministic suite sealer."""

    if replayed.result.get("approved") is not True:
        raise SuiteExecutionError("an unapproved shard cannot enter a suite receipt")
    if replayed.shard_id != shard_manifest.get("shard_id"):
        raise SuiteExecutionError("replay and shard manifest identities differ")
    boundary = review._validate_boundary_contract(
        dict(boundary_contract), expected_cache_sha256=cache.sha256,
    )
    if boundary["runtime_manifest"].get("sha256") != runtime_manifest_sha256:
        raise SuiteExecutionError("boundary and invocation runtime identities differ")
    for digest, label in (
        (runtime_manifest_sha256, "runtime manifest"),
        (assigned_chunk_inventory_sha256, "assigned chunk inventory"),
        (replay_contract_digest, "replay contract"),
        (sanitized_output_sha256, "sanitized output"),
        (command_sha256, "actual command"),
    ):
        if not isinstance(digest, str) or review._SHA256_RE.fullmatch(digest) is None:
            raise SuiteExecutionError(f"{label} hash is malformed")
    budget = token_budget.to_dict()
    if budget.get("within_budget") is not True:
        raise SuiteExecutionError("over-budget shard cannot enter a suite receipt")
    reported = (
        replayed.reported_input_tokens,
        replayed.reported_cached_input_tokens,
        replayed.reported_output_tokens,
        replayed.reported_reasoning_output_tokens,
        replayed.derived_input_plus_output_tokens,
    )
    if (any(type(value) is not int or value < 0 for value in reported)
            or replayed.reported_cached_input_tokens
            > replayed.reported_input_tokens
            or replayed.reported_reasoning_output_tokens
            > replayed.reported_output_tokens
            or replayed.derived_input_plus_output_tokens
            != replayed.reported_input_tokens + replayed.reported_output_tokens):
        raise SuiteExecutionError(
            "reported cumulative Codex usage is internally inconsistent"
        )
    return {
        "shard_manifest": json.loads(json.dumps(
            dict(shard_manifest), ensure_ascii=False, allow_nan=False,
        )),
        "shard_result": replayed.result,
        "invocation_id": replayed.invocation_id,
        "thread_id": replayed.thread_id,
        "raw_output_sha256": replayed.raw_sha256,
        "sanitized_output_sha256": sanitized_output_sha256,
        "reported_input_tokens": replayed.reported_input_tokens,
        "reported_cached_input_tokens": replayed.reported_cached_input_tokens,
        "reported_output_tokens": replayed.reported_output_tokens,
        "reported_reasoning_output_tokens": (
            replayed.reported_reasoning_output_tokens
        ),
        "derived_input_plus_output_tokens": (
            replayed.derived_input_plus_output_tokens
        ),
        "normalized_shard_trace_sha256": _sha256_bytes(replayed.trace_bytes),
        "cache_manifest_sha256": cache.sha256,
        "prompt_sha256": _sha256_bytes(prompt),
        "command_sha256": command_sha256,
        "boundary_contract_sha256": boundary["contract_sha256"],
        "runtime_manifest_sha256": runtime_manifest_sha256,
        "replay_contract_sha256": replay_contract_digest,
        "assigned_chunk_inventory_sha256": assigned_chunk_inventory_sha256,
        "token_budget": budget,
        "completed": True,
        "exit_code": 0,
        "compaction_event_count": 0,
        "unknown_event_count": 0,
        "assigned_chunks_complete": True,
    }


def build_suite_evidence_manifest(
    *, runtime_manifest: Mapping[str, Any],
    protocols: Mapping[str, ProtocolExecution],
    boundary_contracts: Mapping[str, Sequence[Mapping[str, Any]]],
    process_evidence_sha256s: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """Build the fixed private manifest consumed by independent auditors."""

    runtime = review._validate_runtime_manifest(dict(runtime_manifest))
    if (set(protocols) != set(PROTOCOL_ORDER)
            or set(boundary_contracts) != set(PROTOCOL_ORDER)
            or set(process_evidence_sha256s) != set(PROTOCOL_ORDER)):
        raise SuiteExecutionError("suite evidence must cover exactly both protocols")
    protocol_rows: list[dict[str, Any]] = []
    for protocol_id in PROTOCOL_ORDER:
        execution = protocols[protocol_id]
        boundaries = list(boundary_contracts[protocol_id])
        process_hashes = list(process_evidence_sha256s[protocol_id])
        if (execution.protocol_id != protocol_id
                or len(execution.invocations) != len(shard_plan.SHARD_ORDER)
                or len(boundaries) != len(shard_plan.SHARD_ORDER)
                or len(process_hashes) != len(shard_plan.SHARD_ORDER)):
            raise SuiteExecutionError("suite evidence has an incomplete protocol")
        shard_rows: list[dict[str, Any]] = []
        for expected_shard, envelope, boundary, process_hash in zip(
            shard_plan.SHARD_ORDER, execution.invocations, boundaries,
            process_hashes,
            strict=True,
        ):
            manifest = envelope.get("shard_manifest")
            if (not isinstance(manifest, Mapping)
                    or manifest.get("shard_id") != expected_shard):
                raise SuiteExecutionError("suite evidence shard order is not canonical")
            cache_hash = envelope.get("cache_manifest_sha256")
            raw_hash = envelope.get("raw_output_sha256")
            sanitized_hash = envelope.get("sanitized_output_sha256")
            if (not isinstance(cache_hash, str)
                    or review._SHA256_RE.fullmatch(cache_hash) is None
                    or not isinstance(raw_hash, str)
                    or review._SHA256_RE.fullmatch(raw_hash) is None
                    or not isinstance(sanitized_hash, str)
                    or review._SHA256_RE.fullmatch(sanitized_hash) is None):
                raise SuiteExecutionError("suite evidence shard hash is malformed")
            normalized_boundary = review._validate_boundary_contract(
                dict(boundary), expected_cache_sha256=cache_hash,
            )
            if (normalized_boundary["runtime_manifest"] != runtime
                    or envelope.get("boundary_contract_sha256")
                    != normalized_boundary["contract_sha256"]):
                raise SuiteExecutionError("suite evidence boundary identity differs")
            if (not isinstance(process_hash, str)
                    or review._SHA256_RE.fullmatch(process_hash) is None):
                raise SuiteExecutionError("suite process-evidence hash is malformed")
            shard_rows.append({
                "shard_id": expected_shard,
                "projected_cache_manifest_sha256": cache_hash,
                "raw_output_sha256": raw_hash,
                "sanitized_output_sha256": sanitized_hash,
                "process_evidence_sha256": process_hash,
                "boundary_contract": normalized_boundary,
            })
        protocol_rows.append({
            "protocol_id": protocol_id,
            "acquisition_mode": (
                "online_pinned_identity"
                if protocol_id == shard_plan.SEMANTIC_PROTOCOL_ID
                else "offline_replay_from_semantic"
            ),
            "replay_source_manifest_sha256": (
                None
                if protocol_id == shard_plan.SEMANTIC_PROTOCOL_ID
                else protocols[
                    shard_plan.SEMANTIC_PROTOCOL_ID
                ].full_cache.sha256
            ),
            "review_snapshot_sha256": execution.snapshot.sha256,
            "full_cache_manifest_sha256": execution.full_cache.sha256,
            "full_citation_evidence_surface_sha256": (
                execution.receipt["full_evidence_surface_sha256"]
            ),
            "shards": shard_rows,
        })
    return {
        "schema_version": SUITE_EVIDENCE_SCHEMA_VERSION,
        "runtime_manifest": runtime,
        "protocols": protocol_rows,
    }


def default_process_runner(
    command: Sequence[str], prompt: bytes, cwd: Path,
    environment: Mapping[str, str], timeout_seconds: int,
) -> ProcessOutcome:
    """Run one fresh Codex process; callers receive bytes, never file handles."""

    process = subprocess.Popen(
        list(command), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, cwd=str(cwd), env=dict(environment),
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(prompt, timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        review._terminate(process)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
        raise SuiteExecutionError("knowledge-review shard timed out") from exc
    return ProcessOutcome(process.returncode, stdout, stderr)


def invoke_process(
    *, runner: ProcessRunner, command: Sequence[str], prompt: bytes,
    cwd: Path, environment: Mapping[str, str], timeout_seconds: int,
) -> ProcessOutcome:
    """Small injected process boundary used by focused, no-Codex tests."""

    if type(timeout_seconds) is not int or timeout_seconds <= 0:
        raise SuiteExecutionError("shard timeout must be a positive integer")
    outcome = runner(command, prompt, cwd, environment, timeout_seconds)
    if (not isinstance(outcome, ProcessOutcome)
            or type(outcome.returncode) is not int
            or not isinstance(outcome.stdout, bytes)
            or not isinstance(outcome.stderr, bytes)):
        raise SuiteExecutionError("process runner returned a malformed outcome")
    if (outcome.returncode == 0
            and len(outcome.stdout) > review.MAX_RAW_REVIEW_BYTES):
        raise SuiteExecutionError("raw shard event stream exceeds the fixed limit")
    return outcome


def build_process_evidence(
    *, command: Sequence[str], cwd: Path, prompt: bytes,
    outcome: ProcessOutcome, command_sha256: str,
) -> dict[str, Any]:
    """Describe the exact process boundary in the private evidence tree."""

    if (not command or any(not isinstance(item, str) or not item for item in command)
            or not isinstance(prompt, bytes)
            or not isinstance(outcome, ProcessOutcome)
            or type(outcome.returncode) is not int
            or review._SHA256_RE.fullmatch(command_sha256) is None):
        raise SuiteExecutionError("process evidence input is malformed")
    return {
        "schema_version": "hlsgraph.knowledge-review.process-evidence.v1",
        "actual_argv": list(command),
        "cwd": str(cwd.resolve(strict=True)),
        "stdin_sha256": _sha256_bytes(prompt),
        "stdout_sha256": _sha256_bytes(outcome.stdout),
        "stderr_sha256": _sha256_bytes(outcome.stderr),
        "returncode": outcome.returncode,
        "command_contract_sha256": command_sha256,
    }


def _persist_process_outcome(
    *, outcome: ProcessOutcome, command: Sequence[str], cwd: Path,
    prompt: bytes, command_sha256: str, cache: review.ReviewCache,
    raw_path: Path, sanitized_path: Path, raw_stderr_path: Path,
    stderr_path: Path, process_path: Path,
) -> bytes:
    """Persist the exact process boundary before interpreting its result."""

    process_bytes = review._canonical_json(build_process_evidence(
        command=command, cwd=cwd, prompt=prompt, outcome=outcome,
        command_sha256=command_sha256,
    ))
    review._write_private(raw_path, outcome.stdout)
    review._write_private(raw_stderr_path, outcome.stderr)
    review._write_private(process_path, process_bytes)
    if outcome.returncode != 0:
        review._write_private(
            sanitized_path, _redact_shard_payloads(outcome.stdout, cache),
        )
        review._write_private(
            stderr_path, _redact_shard_payloads(outcome.stderr, cache),
        )
        raise SuiteExecutionError(
            f"Codex shard failed with exit code {outcome.returncode}; "
            f"see {sanitized_path} and {stderr_path}"
        )
    return process_bytes


def _persist_success_derivatives(
    *, outcome: ProcessOutcome, cache: review.ReviewCache,
    sanitized_path: Path, stderr_path: Path,
) -> tuple[bytes, bytes]:
    """Write exactly one safe derivative for a zero-exit process."""

    if outcome.returncode != 0:
        raise SuiteExecutionError("success derivatives require a zero exit status")
    redacted_stdout = _redact_shard_payloads(outcome.stdout, cache)
    redacted_stderr = _redact_shard_payloads(outcome.stderr, cache)
    try:
        sanitized = sanitize_shard_raw_stream(outcome.stdout, cache)
    except Exception:
        # Preserve a safe diagnostic even when the stricter semantic sanitizer
        # rejects the returned event stream.  Exclusive creation ensures this
        # path can never overwrite earlier evidence.
        review._write_private(sanitized_path, redacted_stdout)
        review._write_private(stderr_path, redacted_stderr)
        raise
    review._write_private(sanitized_path, sanitized)
    review._write_private(stderr_path, redacted_stderr)
    return sanitized, redacted_stderr


def _mkdir_private_chain(path: Path, *, stop: Path) -> None:
    stop = stop.resolve(strict=True)
    missing: list[Path] = []
    current = path.absolute()
    while not current.exists():
        missing.append(current)
        current = current.parent
    if current.resolve(strict=True) != stop and not current.resolve(strict=True).is_relative_to(stop):
        raise SuiteExecutionError("private suite path escapes its work root")
    for item in reversed(missing):
        review._mkdir_private(item)
    if os.name != "nt":
        for item in [path, *path.parents]:
            if item == stop.parent:
                break
            if item.exists() and item.is_relative_to(stop):
                item.chmod(0o700)


def _strict_private_empty_root(path: Path) -> Path:
    lexical = path.absolute()
    if review._is_link_like(lexical):
        raise SuiteExecutionError("suite work root must not be linked")
    resolved = lexical.resolve(strict=True)
    info = resolved.lstat()
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) != 0o700):
        raise SuiteExecutionError("suite work root must be caller-owned mode 0700")
    if any(resolved.iterdir()):
        raise SuiteExecutionError("suite work root must start empty")
    if review._mount_fstype(resolved) != "ext4":
        raise SuiteExecutionError("suite work root must live on WSL ext4")
    return resolved


def _load_offline_formal_tokenizer() -> tuple[Any, Path]:
    """Load the pinned tokenizer only from one frozen external cache file."""

    cache_text = os.environ.get(TOKENIZER_CACHE_ENV, "")
    cache_lexical = Path(cache_text)
    if not cache_text or not cache_lexical.is_absolute():
        raise SuiteExecutionError(
            f"formal suite requires an absolute {TOKENIZER_CACHE_ENV}"
        )
    if review._is_link_like(cache_lexical):
        raise SuiteExecutionError("formal tokenizer cache must not be linked")
    try:
        cache_root = cache_lexical.resolve(strict=True)
        entries = sorted(path.name for path in cache_root.iterdir())
    except OSError:
        raise SuiteExecutionError(
            "formal tokenizer cache is missing or unreadable"
        ) from None
    if entries != [TOKENIZER_BPE_CACHE_KEY]:
        raise SuiteExecutionError(
            "formal tokenizer cache must contain exactly the pinned BPE table"
        )
    if os.name != "nt" and review._mount_fstype(cache_root) != "ext4":
        raise SuiteExecutionError("formal tokenizer cache must live on WSL ext4")
    table = cache_root / TOKENIZER_BPE_CACHE_KEY
    try:
        before = review._read_stable_restricted_file(
            table, label="formal tokenizer BPE table",
            file_mode=0o400, parent_mode=0o500,
            max_bytes=TOKENIZER_BPE_SIZE,
        )
    except (OSError, ValueError) as exc:
        raise SuiteExecutionError(str(exc)) from exc
    if (len(before) != TOKENIZER_BPE_SIZE
            or _sha256_bytes(before) != TOKENIZER_BPE_SHA256):
        raise SuiteExecutionError("formal tokenizer BPE table identity differs")
    tokenizer = shard_plan.load_verified_tokenizer()
    try:
        after = review._read_stable_restricted_file(
            table, label="formal tokenizer BPE table",
            file_mode=0o400, parent_mode=0o500,
            max_bytes=TOKENIZER_BPE_SIZE,
        )
    except (OSError, ValueError) as exc:
        raise SuiteExecutionError(str(exc)) from exc
    if after != before:
        raise SuiteExecutionError("formal tokenizer BPE table changed while loading")
    return tokenizer, cache_root


def _ensure_clean_checkout(root: Path, environment: Mapping[str, str]) -> None:
    completed = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=str(root), env=dict(environment), stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, timeout=30, check=False,
    )
    if completed.returncode != 0 or completed.stdout:
        raise SuiteExecutionError(
            "formal suite requires a clean committed candidate checkout"
        )


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    value = review._strict_json_bytes(path.read_bytes(), label=label)
    if not isinstance(value, dict):
        raise SuiteExecutionError(f"{label} must be a JSON object")
    return value


def _make_full_cache(
    *, root: Path, snapshot: review.ReviewSnapshot, target: Path,
    fetcher: Callable[[str, float, int], review.TrustedFetch],
    fetch_timeout_seconds: float,
    pdf_text_extractor: Callable[[bytes], review.TextDerivation | None] | None,
    pdftotext_command: str | None, pdftotext_sha256: str | None,
) -> review.ReviewCache:
    cache = review.create_review_cache(
        root, snapshot, target, fetcher=fetcher,
        timeout_seconds=fetch_timeout_seconds,
        pdf_text_extractor=pdf_text_extractor,
        pdftotext_command=pdftotext_command,
        pdftotext_sha256=pdftotext_sha256,
    )
    return review.load_review_cache(cache.root, snapshot)


def _schema_hash(root: Path, relative: str) -> str:
    return _sha256_bytes(_read_stable_checkout_file(root, relative))


def _invocation_paths(
    work_root: Path, protocol_id: str, shard_id: str,
) -> tuple[Path, Path, Path]:
    label = _protocol_label(protocol_id)
    base = work_root / "invocations" / label / shard_id
    return base / "cache-parent" / "cache", base / "evidence", base


def _validate_static_invocation_state(
    *, root: Path, control_surface: Mapping[str, str],
    snapshot: review.ReviewSnapshot, full_cache: review.ReviewCache,
    shard_cache: review.ReviewCache, shard_manifest: Mapping[str, Any],
    prompt: bytes, base_prompt: bytes, plan_hash: str,
    runtime_manifest: Mapping[str, Any], resolved_codex: Path,
    cache_parent_policy: str,
) -> None:
    if freeze_control_surface(root) != dict(control_surface):
        raise SuiteExecutionError("suite control bytes changed during invocation")
    post_snapshot = review.freeze_review_snapshot(root, snapshot.protocol_id)
    if post_snapshot != snapshot:
        raise SuiteExecutionError("review snapshot changed during invocation")
    suite_cache.frozen_cache_fetcher(full_cache)
    validated = suite_cache.validate_shard_cache(
        shard_cache, full_cache, shard_manifest,
    )
    rebuilt = suite.build_shard_prompt(
        base_protocol_text=base_prompt,
        snapshot_inventory=post_snapshot.inventory(), plan_sha256=plan_hash,
        shard_projection=shard_manifest,
    )
    if (validated.manifest_bytes != shard_cache.manifest_bytes
            or rebuilt != prompt
            or review._freeze_runtime_manifest(resolved_codex)
            != dict(runtime_manifest)
            or review._validate_review_cache_parent(shard_cache.root)
            != cache_parent_policy):
        raise SuiteExecutionError(
            "runtime, cache, prompt, or boundary bytes changed during invocation"
        )


def _execute_one_formal_shard(
    *, root: Path, work_root: Path, protocol_id: str, shard_id: str,
    snapshot: review.ReviewSnapshot, full_cache: review.ReviewCache,
    plan: Mapping[str, Any], plan_hash: str, tokenizer: Any,
    control_surface: Mapping[str, str], resolved_codex: Path,
    runtime_manifest: Mapping[str, Any], process_runner: ProcessRunner,
    timeout_seconds: int,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    projection = suite.project_shard_manifest(
        protocol_id=protocol_id, snapshot_inventory=snapshot.inventory(),
        cache_manifest=full_cache.manifest, plan=plan, shard_id=shard_id,
    )
    cache_target, evidence_parent, invocation_root = _invocation_paths(
        work_root, protocol_id, shard_id,
    )
    _mkdir_private_chain(cache_target.parent, stop=work_root)
    shard_cache = suite_cache.materialize_shard_cache(
        full_cache, projection, cache_target,
    )
    _mkdir_private_chain(evidence_parent, stop=work_root)
    raw_path = evidence_parent / RAW_STREAM_PATH
    sanitized_path = evidence_parent / SANITIZED_STREAM_PATH
    stderr_path = evidence_parent / STDERR_PATH
    raw_stderr_path = evidence_parent / RAW_STDERR_PATH
    process_path = evidence_parent / PROCESS_EVIDENCE_PATH
    review._validate_review_evidence_parent(raw_path, stderr_path)

    prompt_relative = SHARD_PROMPT_PATHS[protocol_id]
    base_prompt = _read_stable_checkout_file(root, prompt_relative)
    if _sha256_bytes(base_prompt) != control_surface.get(prompt_relative):
        raise SuiteExecutionError("frozen shard prompt differs from control surface")
    prompt = suite.build_shard_prompt(
        base_protocol_text=base_prompt,
        snapshot_inventory=snapshot.inventory(), plan_sha256=plan_hash,
        shard_projection=projection,
    )
    material = assigned_chunk_material(shard_cache)
    budget = enforce_shard_token_budget(
        prompt=prompt, material=material, tokenizer=tokenizer,
    )

    external_handle = tempfile.TemporaryDirectory(
        prefix="hlsgraph-suite-external-", dir="/tmp",
    )
    peer_handle = tempfile.TemporaryDirectory(
        prefix="hlsgraph-suite-peer-", dir=str(work_root),
    )
    external_canary = Path(external_handle.name) / "canary.txt"
    peer_canary = Path(peer_handle.name) / "canary.txt"
    evidence_canary = evidence_parent / ".boundary-canary"
    external_bytes = os.urandom(48)
    peer_bytes = os.urandom(48)
    evidence_bytes = os.urandom(48)
    external_canary.write_bytes(external_bytes)
    peer_canary.write_bytes(peer_bytes)
    review._write_private(evidence_canary, evidence_bytes)
    try:
        environment, profile, boundary = review._official_boundary(
            root, shard_cache.root, Path(external_handle.name),
            codex_executable=resolved_codex,
            cache_manifest_sha256=shard_cache.sha256,
            peer_sibling_canary=peer_canary,
            evidence_canary=evidence_canary,
        )
        if boundary["runtime_manifest"] != dict(runtime_manifest):
            raise SuiteExecutionError("suite invocations do not share one runtime")
        canary_results = review._verify_boundary_canaries(
            codex=str(resolved_codex), root=root, cache_root=shard_cache.root,
            profile_values=profile, boundary=boundary, environment=environment,
        )
        boundary_contract = review._build_boundary_contract(
            runtime_manifest=boundary["runtime_manifest"],
            cache_manifest_sha256=shard_cache.sha256,
            cache_parent_policy=boundary["cache_parent_policy"],
            evidence_parent_policy=review.EVIDENCE_PARENT_POLICY,
            canary_results=canary_results,
        )
        command = _actual_shard_command(
            root=root, cache_root=shard_cache.root,
            codex=str(resolved_codex), profile_values=profile,
        )
        command_digest = validate_actual_shard_command(
            command, root=root, cache_root=shard_cache.root,
            codex=resolved_codex,
        )
        outcome = invoke_process(
            runner=process_runner, command=command, prompt=prompt,
            cwd=shard_cache.root, environment=environment,
            timeout_seconds=timeout_seconds,
        )
        process_bytes = _persist_process_outcome(
            outcome=outcome, command=command, cwd=shard_cache.root,
            prompt=prompt, command_sha256=command_digest, cache=shard_cache,
            raw_path=raw_path, sanitized_path=sanitized_path,
            raw_stderr_path=raw_stderr_path, stderr_path=stderr_path,
            process_path=process_path,
        )
        # The sanitized derivative is separate and can never stand in for the
        # authority stream in receipts or independent replay.
        sanitized, _stderr_bytes = _persist_success_derivatives(
            outcome=outcome, cache=shard_cache,
            sanitized_path=sanitized_path, stderr_path=stderr_path,
        )
        require_exact_command_inventory(outcome.stdout, material.commands)
        _validate_static_invocation_state(
            root=root, control_surface=control_surface, snapshot=snapshot,
            full_cache=full_cache, shard_cache=shard_cache,
            shard_manifest=projection, prompt=prompt, base_prompt=base_prompt,
            plan_hash=plan_hash, runtime_manifest=runtime_manifest,
            resolved_codex=resolved_codex,
            cache_parent_policy=boundary["cache_parent_policy"],
        )
        if (external_canary.read_bytes() != external_bytes
                or peer_canary.read_bytes() != peer_bytes
                or evidence_canary.read_bytes() != evidence_bytes):
            raise SuiteExecutionError("suite privacy canary changed during invocation")
        replayed = suite_replay.replay_shard_raw_review(
            outcome.stdout, cache=shard_cache, shard_manifest=projection,
        )
        envelope = build_invocation_envelope(
            replayed=replayed, shard_manifest=projection, cache=shard_cache,
            prompt=prompt, boundary_contract=boundary_contract,
            runtime_manifest_sha256=str(runtime_manifest["sha256"]),
            token_budget=budget,
            assigned_chunk_inventory_sha256=material.inventory_sha256,
            replay_contract_digest=replay_contract_sha256(root),
            sanitized_output_sha256=_sha256_bytes(sanitized),
            command_sha256=command_digest,
        )
        review._write_private(
            evidence_parent / INVOCATION_ENVELOPE_PATH,
            review._canonical_json(envelope),
        )
        # Keep the deterministic layout honest; no model-writable location was
        # ever mounted and every retained evidence file is caller-created.
        if invocation_root != evidence_parent.parent:
            raise AssertionError("internal invocation layout drift")
        return envelope, boundary_contract, _sha256_bytes(process_bytes)
    finally:
        evidence_canary.unlink(missing_ok=True)
        peer_handle.cleanup()
        external_handle.cleanup()


def _build_protocol_execution(
    *, root: Path, protocol_id: str, snapshot: review.ReviewSnapshot,
    full_cache: review.ReviewCache, plan: Mapping[str, Any],
    citation_audit: Mapping[str, Any], invocations: Sequence[Mapping[str, Any]],
    runtime_manifest_sha256: str,
) -> ProtocolExecution:
    shard_results = [
        value["shard_result"] for value in invocations
        if isinstance(value, Mapping)
    ]
    aggregate = suite.aggregate_shard_results(
        protocol_id=protocol_id, snapshot_inventory=snapshot.inventory(),
        cache_manifest=full_cache.manifest, citation_audit=citation_audit,
        plan=plan, shard_results=shard_results,
    )
    if aggregate.get("approved") is not True:
        raise SuiteExecutionError("protocol aggregate is not approved")
    full_surface = suite.citation_evidence_surface_sha256(
        full_cache.manifest["citations"],
    )
    trace = suite_seal.build_protocol_trace(
        protocol_id=protocol_id, plan=plan, citation_audit=citation_audit,
        review_snapshot_sha256=snapshot.sha256,
        citation_evidence_sha256=snapshot.citation_evidence_sha256,
        full_evidence_surface_sha256=full_surface,
        runtime_manifest_sha256=runtime_manifest_sha256,
        invocations=invocations, aggregate_result=aggregate,
    )
    receipt = suite_seal.build_protocol_receipt(
        trace_bytes=trace, protocol_id=protocol_id, plan=plan,
        citation_audit=citation_audit,
        review_snapshot_sha256=snapshot.sha256,
        citation_evidence_sha256=snapshot.citation_evidence_sha256,
        full_evidence_surface_sha256=full_surface,
        runtime_manifest_sha256=runtime_manifest_sha256,
        invocations=invocations, aggregate_result=aggregate,
        output_schema_sha256=snapshot.output_schema_sha256,
        shard_output_schema_sha256=_schema_hash(root, SHARD_OUTPUT_SCHEMA_PATH),
        suite_receipt_schema_sha256=_schema_hash(root, SUITE_RECEIPT_SCHEMA_PATH),
        suite_trace_schema_sha256=_schema_hash(root, SUITE_TRACE_SCHEMA_PATH),
    )
    suite_seal.validate_protocol_receipt(
        receipt, trace_bytes=trace, protocol_id=protocol_id, plan=plan,
        citation_audit=citation_audit,
        review_snapshot_sha256=snapshot.sha256,
        citation_evidence_sha256=snapshot.citation_evidence_sha256,
        full_evidence_surface_sha256=full_surface,
        runtime_manifest_sha256=runtime_manifest_sha256,
        invocations=invocations, aggregate_result=aggregate,
        output_schema_sha256=snapshot.output_schema_sha256,
        shard_output_schema_sha256=_schema_hash(root, SHARD_OUTPUT_SCHEMA_PATH),
        suite_receipt_schema_sha256=_schema_hash(root, SUITE_RECEIPT_SCHEMA_PATH),
        suite_trace_schema_sha256=_schema_hash(root, SUITE_TRACE_SCHEMA_PATH),
    )
    return ProtocolExecution(
        protocol_id=protocol_id, snapshot=snapshot, full_cache=full_cache,
        invocations=tuple(dict(value) for value in invocations),
        aggregate_result=aggregate, trace_bytes=trace, receipt=receipt,
    )


def execute_knowledge_review_suite(
    root: Path, work_root: Path, *, codex_command: str,
    timeout_seconds: int,
    fetcher: Callable[[str, float, int], review.TrustedFetch] = review._default_fetch,
    fetch_timeout_seconds: float = 60.0,
    pdf_text_extractor: Callable[[bytes], review.TextDerivation | None] | None = None,
    pdftotext_command: str | None = None,
    pdftotext_sha256: str | None = None,
    process_runner: ProcessRunner = default_process_runner,
    publish: bool = True,
) -> SuiteExecution:
    """Run six isolated Codex reviews and publish only a sealed pair.

    ``process_runner`` is injectable so process plumbing can be tested without
    Codex.  Formal publication rejects every non-default runner by identity.
    Citation fetching remains injectable because exact resolver/body hashes in
    the frozen evidence map independently constrain the accepted bytes.
    """

    if publish and process_runner is not default_process_runner:
        raise SuiteExecutionError(
            "formal publication requires the built-in direct process runner"
        )
    if publish and (
        fetcher is not review._default_fetch or pdf_text_extractor is not None
    ):
        raise SuiteExecutionError(
            "formal publication requires the built-in online citation fetch path"
        )
    if review._formal_host_is_windows():
        raise SuiteExecutionError("formal suite is Linux/WSL2-only; Windows is NO-GO")
    root = root.absolute().resolve(strict=True)
    if root != SCRIPT_ROOT:
        raise SuiteExecutionError("suite executor must belong to the reviewed checkout")
    from eval.agent_ab.common import (
        _resolve_executable, official_process_environment,
        require_official_linux_wsl2,
    )
    require_official_linux_wsl2()
    if str(root).startswith("/mnt/") or review._mount_fstype(root) != "ext4":
        raise SuiteExecutionError("formal suite checkout must live on WSL ext4")
    environment = official_process_environment()
    _ensure_clean_checkout(root, environment)
    resolved_codex = _resolve_executable(codex_command, "Codex CLI")
    version = subprocess.run(
        [str(resolved_codex), "--version"], env=environment, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15,
    ).stdout.decode("utf-8", errors="strict").strip()
    if version != suite_seal.CODEX_CLI_VERSION:
        raise SuiteExecutionError(
            f"formal suite requires {suite_seal.CODEX_CLI_VERSION!r}, found {version!r}"
        )
    runtime_manifest = review._freeze_runtime_manifest(resolved_codex)
    if runtime_manifest != review._freeze_runtime_manifest(resolved_codex):
        raise SuiteExecutionError("Codex runtime changed during preflight")
    work_root = _strict_private_empty_root(work_root)
    codex_home = Path(os.environ["CODEX_HOME"]).resolve(strict=True)
    runtime_root = resolved_codex.parent.resolve(strict=True)
    tokenizer, tokenizer_cache = _load_offline_formal_tokenizer()
    for label, path in (
        ("checkout", root), ("CODEX_HOME", codex_home),
        ("Codex runtime", runtime_root), ("tokenizer cache", tokenizer_cache),
    ):
        if review._paths_overlap(work_root, path):
            raise SuiteExecutionError(f"suite work root overlaps {label}")
    for label, path in (
        ("checkout", root), ("CODEX_HOME", codex_home),
        ("Codex runtime", runtime_root),
    ):
        if review._paths_overlap(tokenizer_cache, path):
            raise SuiteExecutionError(f"tokenizer cache overlaps {label}")

    control = freeze_control_surface(root)
    citation_audit = _read_json(
        root / review.CITATION_AUDIT_PATH, label="citation audit",
    )
    plan = shard_plan.build_shard_plan(citation_audit)
    plan_hash = shard_plan.shard_plan_sha256(plan)

    full_caches: dict[str, review.ReviewCache] = {}
    snapshots: dict[str, review.ReviewSnapshot] = {}
    for protocol_id in PROTOCOL_ORDER:
        label = _protocol_label(protocol_id)
        snapshot = review.freeze_review_snapshot(root, protocol_id)
        snapshots[protocol_id] = snapshot
        parent = work_root / "full" / label / "cache-parent"
        _mkdir_private_chain(parent, stop=work_root)
        selected_fetcher = (
            fetcher if protocol_id == shard_plan.SEMANTIC_PROTOCOL_ID
            else suite_cache.frozen_cache_fetcher(
                full_caches[shard_plan.SEMANTIC_PROTOCOL_ID]
            )
        )
        full_caches[protocol_id] = _make_full_cache(
            root=root, snapshot=snapshot, target=parent / "cache",
            fetcher=selected_fetcher,
            fetch_timeout_seconds=fetch_timeout_seconds,
            pdf_text_extractor=pdf_text_extractor,
            pdftotext_command=pdftotext_command,
            pdftotext_sha256=pdftotext_sha256,
        )
    semantic_surface = suite.citation_evidence_surface_sha256(
        full_caches[shard_plan.SEMANTIC_PROTOCOL_ID].manifest["citations"],
    )
    adversarial_surface = suite.citation_evidence_surface_sha256(
        full_caches[shard_plan.ADVERSARIAL_PROTOCOL_ID].manifest["citations"],
    )
    if (semantic_surface != adversarial_surface
            or snapshots[PROTOCOL_ORDER[0]].citation_evidence_sha256
            != snapshots[PROTOCOL_ORDER[1]].citation_evidence_sha256):
        raise SuiteExecutionError(
            "semantic and adversarial protocols do not share one evidence surface"
        )

    protocols: dict[str, ProtocolExecution] = {}
    all_boundaries: dict[str, list[dict[str, Any]]] = {}
    all_process_hashes: dict[str, list[str]] = {}
    for protocol_id in PROTOCOL_ORDER:
        envelopes: list[dict[str, Any]] = []
        protocol_boundaries: list[dict[str, Any]] = []
        protocol_process_hashes: list[str] = []
        for shard_id in shard_plan.SHARD_ORDER:
            envelope, boundary_contract, process_hash = _execute_one_formal_shard(
                root=root, work_root=work_root, protocol_id=protocol_id,
                shard_id=shard_id, snapshot=snapshots[protocol_id],
                full_cache=full_caches[protocol_id], plan=plan,
                plan_hash=plan_hash, tokenizer=tokenizer,
                control_surface=control, resolved_codex=resolved_codex,
                runtime_manifest=runtime_manifest,
                process_runner=process_runner,
                timeout_seconds=timeout_seconds,
            )
            envelopes.append(envelope)
            protocol_boundaries.append(boundary_contract)
            protocol_process_hashes.append(process_hash)
        all_boundaries[protocol_id] = protocol_boundaries
        all_process_hashes[protocol_id] = protocol_process_hashes
        protocols[protocol_id] = _build_protocol_execution(
            root=root, protocol_id=protocol_id,
            snapshot=snapshots[protocol_id],
            full_cache=full_caches[protocol_id], plan=plan,
            citation_audit=citation_audit, invocations=envelopes,
            runtime_manifest_sha256=str(runtime_manifest["sha256"]),
        )

    semantic = protocols[shard_plan.SEMANTIC_PROTOCOL_ID]
    adversarial = protocols[shard_plan.ADVERSARIAL_PROTOCOL_ID]
    pair_seal = suite_seal.validate_suite_pair(
        semantic_receipt=semantic.receipt,
        adversarial_receipt=adversarial.receipt,
        semantic_result=semantic.aggregate_result,
        adversarial_result=adversarial.aggregate_result,
        plan=plan, citation_audit=citation_audit,
    )
    if freeze_control_surface(root) != control:
        raise SuiteExecutionError("suite control bytes changed before publication")
    for protocol_id in PROTOCOL_ORDER:
        if review.freeze_review_snapshot(root, protocol_id) != snapshots[protocol_id]:
            raise SuiteExecutionError("review snapshot changed before publication")
        suite_cache.frozen_cache_fetcher(full_caches[protocol_id])

    evidence_manifest = build_suite_evidence_manifest(
        runtime_manifest=runtime_manifest, protocols=protocols,
        boundary_contracts=all_boundaries,
        process_evidence_sha256s=all_process_hashes,
    )
    review._write_private(
        work_root / SUITE_EVIDENCE_PATH,
        review._canonical_json(evidence_manifest),
    )
    review._write_private(work_root / PAIR_SEAL_PATH, review._canonical_json(pair_seal))
    if publish:
        artifacts: dict[str, bytes] = {}
        for execution in (semantic, adversarial):
            files = review.PROTOCOL_FILES[execution.protocol_id]
            artifacts[files["result"]] = review._canonical_json(
                execution.aggregate_result,
            )
            artifacts[files["trace"]] = execution.trace_bytes
            artifacts[files["receipt"]] = review._canonical_json(execution.receipt)
        review._publish_artifacts(root, artifacts)
    return SuiteExecution(
        semantic=semantic, adversarial=adversarial, pair_seal=pair_seal,
        evidence_manifest=evidence_manifest,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute the pinned HLSGraph three-shard knowledge review",
    )
    parser.add_argument("--root", type=Path, default=SCRIPT_ROOT)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--codex", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--fetch-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--pdftotext-command")
    parser.add_argument("--pdftotext-sha256")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    execute_knowledge_review_suite(
        args.root, args.work_root, codex_command=args.codex,
        timeout_seconds=args.timeout_seconds,
        fetch_timeout_seconds=args.fetch_timeout_seconds,
        pdftotext_command=args.pdftotext_command,
        pdftotext_sha256=args.pdftotext_sha256,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - formal CLI only
    raise SystemExit(main())


__all__ = [
    "CHUNK_INVENTORY_VERSION", "COMMAND_CONTRACT_VERSION",
    "ChunkMaterial", "EXECUTOR_CONTRACT_VERSION", "ProcessOutcome",
    "ProtocolExecution", "SuiteExecution", "SuiteExecutionError",
    "assigned_chunk_material", "build_invocation_envelope",
    "build_process_evidence",
    "build_suite_evidence_manifest",
    "canonical_shard_command_argv", "canonical_shard_command_sha256",
    "default_process_runner", "enforce_shard_token_budget",
    "execute_knowledge_review_suite", "freeze_control_surface",
    "invoke_process", "replay_contract_sha256",
    "normalize_actual_shard_command", "validate_actual_shard_command",
    "require_exact_command_inventory", "sanitize_shard_raw_stream",
]
