"""Run or dry-run the frozen 192-cell Codex A/B comparison.

The command is intentionally inert unless ``--execute`` is present.  It never
invokes an HLS tool, remote server, or model training job.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import secrets
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
from queue import Empty, Queue
from pathlib import Path
from typing import Any, Iterable, Sequence

from .common import (
    ARM_IDS, CODEGRAPH_OFFLINE_ENV, HERE, SANDBOX_FILESYSTEM_POLICY,
    SANDBOX_MINIMAL_TOKEN, SANDBOX_ALLOW_TREE_ALGORITHM,
    RETRIEVAL_ACCESS_RELATIVE, RETRIEVAL_AUDIT_SCHEMA_VERSION, asset_digest,
    capture_codegraph_build_identity, canonical_json, harness_digest,
    load_corpus_lock, load_environment_lock, load_manifest, load_questions,
    resolve_command_argv,
    official_process_environment, require_official_linux_wsl2,
    require_official_ext4_directory, safe_relative_path,
    sandbox_allow_tree_identity, sha256_bytes, sha256_file,
    verify_evaluation_checkout, verify_official_runtime_identity,
    verify_prepared_workspace,
)
from .parse_trace import _private_snippet_access_id, _private_snippet_output_receipts


DISABLED_CODEX_FEATURES = (
    "browser_use",
    "browser_use_external",
    "browser_use_full_cdp_access",
    "in_app_browser",
    "standalone_web_search",
    "computer_use",
    "image_generation",
    "apps",
    "enable_mcp_apps",
    "multi_agent",
    "multi_agent_v2",
    "plugins",
    "plugin_sharing",
    "remote_plugin",
    "hooks",
    "workspace_dependencies",
)

PERMISSION_PROFILE = "hlsgraph_eval"
PUBLIC_REPOSITORY = HERE.parents[1].resolve()
CODEGRAPH_ENV = CODEGRAPH_OFFLINE_ENV
AuditOverlay = tuple[Path, Path, tuple[dict[str, Any], ...]]


def build_run_plan(
    *, seed: int | None = None, arms: Iterable[str] = ARM_IDS,
    question_ids: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    manifest = load_manifest()
    questions = load_questions()
    selected_arms = tuple(arms)
    invalid_arms = set(selected_arms) - set(ARM_IDS)
    if invalid_arms:
        raise ValueError(f"unknown arm(s): {sorted(invalid_arms)}")
    selected_questions = set(question_ids or (item["id"] for item in questions))
    known_questions = {item["id"] for item in questions}
    if selected_questions - known_questions:
        raise ValueError(f"unknown question(s): {sorted(selected_questions - known_questions)}")
    blocks: list[list[dict[str, Any]]] = []
    rng = random.Random(seed if seed is not None else manifest["randomization_seed"])
    for question in questions:
        if question["id"] not in selected_questions:
            continue
        base_order = list(selected_arms)
        rng.shuffle(base_order)
        for repetition in range(1, manifest["repetitions"] + 1):
            offset = (repetition - 1) % max(1, len(base_order))
            arm_order = base_order[offset:] + base_order[:offset]
            block: list[dict[str, Any]] = []
            for arm in arm_order:
                run_id = f"{question['id']}__r{repetition:02d}__{arm}"
                block.append({
                    "run_id": run_id,
                    "question_id": question["id"],
                    "corpus_id": question["corpus_id"],
                    "category": question["category"],
                    "arm": arm,
                    "repetition": repetition,
                    "timeout_seconds": manifest["codex_cli"]["timeout_seconds"],
                })
            blocks.append(block)
    rng.shuffle(blocks)
    records = [record for block in blocks for record in block]
    for execution_index, record in enumerate(records, 1):
        record["execution_index"] = execution_index
    return records


def _question_map() -> dict[str, dict[str, Any]]:
    return {item["id"]: item for item in load_questions()}


def build_prompt(
    question: dict[str, Any], arm: str, *, trace_challenge: str | None = None,
) -> str:
    arm_guidance = {
        "native": "Use the built-in read-only file and search tools.",
        "codegraph": (
            "Use codegraph_explore first. Treat returned public source as already read; "
            "open files only when its response is incomplete or stale."
        ),
        "hlsgraph-v02": (
            "Use the HLSGraph v0.2 read-only MCP tools first. Keep graph facts, source "
            "declarations, synthetic fixtures, and real tool observations separate."
        ),
        "hlsgraph-v03": (
            "Use the single HLSGraph explore tool first with include_private_snippets=true. "
            "The corpus is public, and its project policy permits only anchor-bounded snippets. "
            "Treat that bounded source context as "
            "already read unless it reports stale, incomplete, ambiguous, or low-confidence data."
        ),
    }[arm]
    criterion_ids = [f"c{index:02d}" for index, _ in enumerate(question["criteria"], 1)]
    lines = [
        "You are answering one frozen public HLS evidence question.",
        arm_guidance,
        "Do not use the web or any network access. Do not delegate to subagents.",
        "Do not edit files or run synthesis, implementation, simulation, or training.",
        "Use only the current public corpus workspace. Shell reads must use project-relative paths; "
        "absolute paths, parent traversal, interpreters, and network commands invalidate the cell.",
        "Cite project-relative paths and exact line ranges.",
        "A source pragma is a declaration, not achieved QoR. A synthetic fixture is never real tool evidence.",
        "Software calls and LLVM CFG are not, by themselves, HLS hardware topology.",
        "Return only JSON matching the supplied output schema. Write the answer in English.",
        f"Question ID: {question['id']}",
        "Question:",
        question["prompt"],
        "Return exactly one atomic claim for each opaque criterion ID, in this order: "
        + ", ".join(criterion_ids) + ". Do not add any other claim.",
        "Each claim must include criterion_id, truth plane, stage, authority, and evidence lines.",
        "Set answer exactly to one line per claim in criterion order, formatted "
        "'<criterion_id>: <statement>'; the answer may contain no other prose.",
        "Use truth_plane=unknown for a requested conclusion that the available evidence cannot establish.",
    ]
    if trace_challenge is not None:
        if re.fullmatch(r"[0-9a-f]{64}", trace_challenge) is None:
            raise ValueError("trace challenge must be a SHA-256 token")
        lines.extend([
            "To bind this answer to its frozen evaluation cell, include this exact string as "
            "the final uncertainties entry:",
            f"eval-context:{trace_challenge}",
        ])
    return "\n".join(lines) + "\n"


def _toml(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _linked_directory(path: Path) -> bool:
    is_junction = getattr(path, "is_junction", None)
    return path.is_symlink() or bool(callable(is_junction) and is_junction())


def _validate_workspace_inventory(
    *, root: Path, workspace: Path, sandbox_boundary: dict[str, Any],
) -> str:
    """Validate the frozen matrix inventory and return the current arm."""

    if sandbox_boundary.get("filesystem_policy") != SANDBOX_FILESYSTEM_POLICY:
        raise RuntimeError("official sandbox boundary has the wrong filesystem policy")
    catalog = sandbox_boundary.get("workspace_catalog")
    if not isinstance(catalog, dict):
        raise RuntimeError("official sandbox boundary lacks the workspace catalog")
    arm_roots = catalog.get("arm_roots")
    workspace_roots = catalog.get("workspace_roots")
    control_roots = catalog.get("control_roots")
    if (not isinstance(arm_roots, dict) or set(arm_roots) != set(ARM_IDS)
            or not isinstance(workspace_roots, dict)
            or not isinstance(control_roots, list) or len(control_roots) != 2):
        raise RuntimeError("official sandbox workspace catalog is malformed")
    expected_corpora = {item["id"] for item in load_corpus_lock()["corpora"]}
    expected_workspace_keys = {
        f"{arm}/{corpus_id}" for arm in ARM_IDS for corpus_id in expected_corpora
    }
    if set(workspace_roots) != expected_workspace_keys:
        raise RuntimeError("official sandbox workspace catalog has an incomplete matrix")
    expected_arms = {arm: root / arm for arm in ARM_IDS}
    expected_workspaces = {
        f"{arm}/{corpus_id}": root / arm / corpus_id
        for arm in ARM_IDS for corpus_id in expected_corpora
    }
    expected_controls = {
        root / ".hlsgraph-eval-boundary", root / "_cache",
    }
    if (any(Path(str(arm_roots[key])).resolve() != value.resolve()
            for key, value in expected_arms.items())
            or any(Path(str(workspace_roots[key])).resolve() != value.resolve()
                   for key, value in expected_workspaces.items())
            or {Path(str(item)).resolve() for item in control_roots}
            != {item.resolve() for item in expected_controls}):
        raise RuntimeError("official sandbox workspace catalog points outside the work root")

    allowed_top = {
        *ARM_IDS, "_cache", ".hlsgraph-eval-boundary",
        "environment.lock.json", "materialization.json",
    }
    required_top = {
        *ARM_IDS, "_cache", "environment.lock.json", "materialization.json",
    }
    actual_top = {item.name for item in root.iterdir()}
    if actual_top - allowed_top or not required_top.issubset(actual_top):
        raise RuntimeError("official work root has a missing or unexpected top-level entry")
    for arm, arm_root in expected_arms.items():
        if not arm_root.is_dir() or _linked_directory(arm_root):
            raise RuntimeError(f"official work-root arm is not a plain directory: {arm}")
        children = {item.name: item for item in arm_root.iterdir()}
        if set(children) != expected_corpora or any(
            not item.is_dir() or _linked_directory(item) for item in children.values()
        ):
            raise RuntimeError(f"official work-root arm has an invalid corpus inventory: {arm}")
    for name in ("_cache",):
        path = root / name
        if not path.is_dir() or _linked_directory(path):
            raise RuntimeError(f"official work-root control directory is invalid: {name}")
    boundary_directory = root / ".hlsgraph-eval-boundary"
    if boundary_directory.exists() and (
        not boundary_directory.is_dir() or _linked_directory(boundary_directory)
    ):
        raise RuntimeError("official boundary-canary directory is invalid")
    for name in ("environment.lock.json", "materialization.json"):
        path = root / name
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"official work-root control file is invalid: {name}")

    try:
        relative = workspace.resolve().relative_to(root)
    except ValueError as exc:
        raise RuntimeError("current workspace escapes the official work root") from exc
    if len(relative.parts) != 2 or relative.parts[0] not in ARM_IDS:
        raise RuntimeError("current workspace is not a frozen arm/corpus directory")
    arm, corpus_id = relative.parts
    if corpus_id not in expected_corpora:
        raise RuntimeError("current workspace is not in the frozen corpus matrix")
    return arm


def _permission_overrides(
    *, workspace: Path, work_root: Path, runs_root: Path,
    sandbox_boundary: dict[str, Any],
) -> list[str]:
    """Return the one fail-closed permission profile used by every arm."""

    root = work_root.resolve()
    current = workspace.resolve()
    result_root = Path(os.path.abspath(os.fspath(runs_root))).resolve()
    try:
        current.relative_to(root)
    except ValueError as exc:
        raise RuntimeError("evaluation workspace escapes the frozen work root") from exc
    locked_root = Path(str(sandbox_boundary.get("work_root", ""))).resolve()
    if locked_root != root:
        raise RuntimeError("sandbox boundary belongs to another work root")
    deny_roots = sandbox_boundary.get("deny_roots")
    if not isinstance(deny_roots, list) or not deny_roots:
        raise RuntimeError("sandbox boundary lacks deny roots")
    protected_roots = {root, *(
        Path(str(path)).resolve() for path in deny_roots
    )}
    if any(
        result_root == protected
        or result_root.is_relative_to(protected)
        or protected.is_relative_to(result_root)
        for protected in protected_roots
    ):
        raise RuntimeError(
            "runs root must be disjoint from every sandbox boundary root"
        )
    official_matrix = sandbox_boundary.get("filesystem_policy") is not None
    arm = (
        _validate_workspace_inventory(
            root=root, workspace=current, sandbox_boundary=sandbox_boundary,
        ) if official_matrix else current.relative_to(root).parts[0]
    )
    if official_matrix and (
        sandbox_boundary.get("filesystem_base_token") != SANDBOX_MINIMAL_TOKEN
        or sandbox_boundary.get("filesystem_base_mode") != "read"
        or sandbox_boundary.get("workspace_mode") != "read"
    ):
        raise RuntimeError("official sandbox boundary changes the minimal allowlist")
    runtime_allowlists = sandbox_boundary.get("runtime_allow_roots", {})
    entries = runtime_allowlists.get(arm, []) if isinstance(runtime_allowlists, dict) else []
    if official_matrix and (not isinstance(entries, list) or not entries):
        raise RuntimeError("official sandbox boundary lacks the arm runtime allowlist")
    rules = {SANDBOX_MINIMAL_TOKEN: "read"}
    runtime_root = Path(str(sandbox_boundary.get("runtime_root", ""))).resolve()
    for entry in entries:
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("path"), str)
            or entry.get("kind") not in {"file", "tree"}
            or entry.get("algorithm") not in {
                "sha256", "hlsgraph.runtime_tree.v1",
                "hlsgraph.sandbox_allow_tree.v1",
            }
            or re.fullmatch(r"[0-9a-f]{64}", str(entry.get("sha256", ""))) is None
        ):
            raise RuntimeError("official sandbox runtime allowlist is malformed")
        path = Path(entry["path"]).resolve()
        if path == runtime_root or not path.is_relative_to(runtime_root):
            raise RuntimeError("official sandbox runtime allowlist is broader than one runtime")
        rules[path.as_posix()] = "read"
    rules[current.as_posix()] = "read"
    filesystem = "{" + ",".join(
        f"{_toml(path)}={_toml(mode)}" for path, mode in rules.items()
    ) + "}"
    values = [
        f"default_permissions={_toml(PERMISSION_PROFILE)}",
        f"permissions.{PERMISSION_PROFILE}.network.enabled=false",
        f"permissions.{PERMISSION_PROFILE}.filesystem={filesystem}",
        'web_search="disabled"',
    ]
    output: list[str] = []
    for value in values:
        output.extend(["-c", value])
    return output


def _codegraph_parts(command_text: str) -> list[str]:
    parts = resolve_command_argv(command_text)
    if (len(parts) >= 2
            and Path(parts[0]).name.casefold() in {"node", "node.exe"}):
        script = Path(parts[1])
        if not script.is_absolute():
            parts[1] = str(Path(os.path.abspath(os.fspath(Path.cwd() / script))))
    return parts


def _linked(path: Path) -> bool:
    is_junction = getattr(path, "is_junction", None)
    return path.is_symlink() or bool(
        callable(is_junction) and is_junction()
    )


def _verify_audit_placeholder(workspace: Path) -> Path:
    root = Path(os.path.abspath(os.fspath(workspace)))
    target = root / RETRIEVAL_ACCESS_RELATIVE
    for path in (root, root / ".hlsgraph", target.parent):
        if _linked(path) or not path.is_dir():
            raise RuntimeError("retrieval audit placeholder parent is missing or linked")
    if _linked(target) or not target.is_file():
        raise RuntimeError("retrieval audit placeholder is missing or linked")
    info = target.lstat()
    if (not stat.S_ISREG(info.st_mode) or info.st_size != 0
            or int(getattr(info, "st_nlink", 1)) != 1):
        raise RuntimeError("retrieval audit placeholder must be one zero-byte regular file")
    if os.name != "nt":
        private_mode = stat.S_IMODE(target.parent.lstat().st_mode)
        if private_mode != 0o700:
            raise RuntimeError("retrieval audit private directory must have mode 0700")
        if stat.S_IMODE(info.st_mode) != 0o600:
            raise RuntimeError("retrieval audit placeholder must have mode 0600")
    return target


def _audit_parent_chain(path: Path) -> list[dict[str, Any]]:
    """Capture every lexical parent from the filesystem anchor to the file."""

    lexical = Path(os.path.abspath(os.fspath(path)))
    parent = lexical.parent
    anchor = Path(lexical.anchor)
    paths = [anchor]
    current = anchor
    for part in parent.parts[1:]:
        current /= part
        paths.append(current)
    records: list[dict[str, Any]] = []
    for item in paths:
        if _linked(item) or not item.is_dir():
            raise RuntimeError("retrieval audit parent chain is missing or linked")
        info = item.lstat()
        if not stat.S_ISDIR(info.st_mode):
            raise RuntimeError("retrieval audit parent chain contains a non-directory")
        records.append({
            "path": item.as_posix(),
            "dev": int(info.st_dev), "ino": int(info.st_ino),
            "mode": int(info.st_mode),
            "uid": int(getattr(info, "st_uid", 0)),
            "gid": int(getattr(info, "st_gid", 0)),
        })
    return records


def _verify_audit_parent_chain(
    path: Path, expected: Sequence[dict[str, Any]],
) -> None:
    if not isinstance(expected, (list, tuple)) or not expected:
        raise RuntimeError("retrieval audit parent identity is missing")
    current = _audit_parent_chain(path)
    if current != list(expected):
        raise RuntimeError("retrieval audit parent chain changed after preparation")


def _stable_audit_bytes(
    path: Path, *, parent_chain: Sequence[dict[str, Any]] | None = None,
    max_bytes: int = 4 * 1024 * 1024,
) -> bytes:
    """Read one unlinked, single-link 0600 audit file without a TOCTOU gap."""

    lexical = Path(os.path.abspath(os.fspath(path)))
    expected_parents = list(parent_chain or _audit_parent_chain(lexical))
    _verify_audit_parent_chain(lexical, expected_parents)
    if _linked(lexical) or not lexical.is_file():
        raise RuntimeError("retrieval audit overlay is missing or linked")
    descriptor = -1
    parent_descriptor = -1
    try:
        before = lexical.lstat()
        flags = (os.O_RDONLY | int(getattr(os, "O_BINARY", 0))
                 | int(getattr(os, "O_NOFOLLOW", 0)))
        parent_flags = (os.O_RDONLY | int(getattr(os, "O_DIRECTORY", 0))
                        | int(getattr(os, "O_NOFOLLOW", 0)))
        parent_descriptor = (
            os.open(lexical.parent, parent_flags) if os.name != "nt" else -1
        )
        parent_opened = (
            os.fstat(parent_descriptor) if parent_descriptor >= 0
            else lexical.parent.lstat()
        )
        expected_parent = expected_parents[-1]
        if (int(parent_opened.st_dev), int(parent_opened.st_ino)) != (
            expected_parent["dev"], expected_parent["ino"],
        ):
            raise RuntimeError("retrieval audit parent changed while opened")
        if parent_descriptor >= 0 and os.open in os.supports_dir_fd:
            descriptor = os.open(lexical.name, flags, dir_fd=parent_descriptor)
        else:
            descriptor = os.open(lexical, flags)
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
        current = (
            os.stat(lexical.name, dir_fd=parent_descriptor, follow_symlinks=False)
            if parent_descriptor >= 0 and os.stat in os.supports_dir_fd
            else lexical.lstat()
        )
        parent_closed = (
            os.fstat(parent_descriptor) if parent_descriptor >= 0
            else lexical.parent.lstat()
        )
    except OSError as exc:
        raise RuntimeError("cannot read retrieval audit overlay safely") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if parent_descriptor >= 0:
            os.close(parent_descriptor)
    identity = lambda item: (
        int(item.st_dev), int(item.st_ino), int(item.st_mode),
        int(item.st_size), int(item.st_mtime_ns), int(getattr(item, "st_nlink", 1)),
    )
    data = b"".join(chunks)
    if (len(data) > max_bytes or not stat.S_ISREG(before.st_mode)
            or identity(before) != identity(opened)
            or identity(opened) != identity(closed)
            or identity(closed) != identity(current)
            or (int(parent_opened.st_dev), int(parent_opened.st_ino),
                int(parent_opened.st_mode), int(parent_opened.st_ctime_ns))
            != (int(parent_closed.st_dev), int(parent_closed.st_ino),
                int(parent_closed.st_mode), int(parent_closed.st_ctime_ns))
            or int(getattr(current, "st_nlink", 1)) != 1
            or (os.name != "nt" and stat.S_IMODE(current.st_mode) != 0o600)):
        raise RuntimeError("retrieval audit overlay changed or has unsafe metadata")
    _verify_audit_parent_chain(lexical, expected_parents)
    return data


def _verify_audit_overlay_file(
    path: Path, *, require_empty: bool,
    parent_chain: Sequence[dict[str, Any]] | None = None,
) -> bytes:
    data = _stable_audit_bytes(path, parent_chain=parent_chain)
    if require_empty and data:
        raise RuntimeError("retrieval audit overlay is not empty before execution")
    return data


def _parse_retrieval_audit(data: bytes) -> list[dict[str, Any]]:
    """Validate the product's body-free retrieval audit schema."""

    try:
        text = data.decode("ascii")
    except UnicodeDecodeError as exc:
        raise RuntimeError("retrieval audit must be ASCII JSONL") from exc
    if text and not text.endswith("\n"):
        raise RuntimeError("retrieval audit must end with a newline")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line:
            raise RuntimeError("retrieval audit contains a blank record")

        def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            value: dict[str, Any] = {}
            for key, item in pairs:
                if key in value:
                    raise RuntimeError(
                        f"retrieval audit record {line_number} has duplicate keys"
                    )
                value[key] = item
            return value

        try:
            record = json.loads(line, object_pairs_hook=reject_duplicates)
        except (json.JSONDecodeError, RuntimeError) as exc:
            raise RuntimeError(
                f"retrieval audit record {line_number} is malformed"
            ) from exc
        if not isinstance(record, dict) or set(record) != {
            "content_sha256", "anchor", "result", "byte_count",
        }:
            raise RuntimeError("retrieval audit exposes fields outside the allowed metadata")
        anchor = record.get("anchor")
        if (not isinstance(record.get("content_sha256"), str)
                or re.fullmatch(r"[0-9a-f]{64}", record["content_sha256"]) is None
                or not isinstance(record.get("result"), str)
                or re.fullmatch(r"[a-z][a-z0-9_]{0,63}", record["result"]) is None
                or isinstance(record.get("byte_count"), bool)
                or not isinstance(record.get("byte_count"), int)
                or not 0 <= record["byte_count"] <= 16_000
                or not isinstance(anchor, dict)
                or set(anchor) - {"kind", "start_line", "end_line", "chunk_id"}):
            raise RuntimeError("retrieval audit record violates its metadata schema")
        for key, value in anchor.items():
            if key in {"start_line", "end_line"}:
                if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                    raise RuntimeError("retrieval audit has an invalid line anchor")
            elif (not isinstance(value, str) or not value or len(value) > 128
                  or "\x00" in value):
                raise RuntimeError("retrieval audit has an invalid string anchor")
        if len(records) >= 10_000:
            raise RuntimeError("retrieval audit exceeds the record limit")
        records.append(record)
    return records


def _retrieval_audit_receipt(data: bytes) -> dict[str, Any]:
    records = _parse_retrieval_audit(data)
    returned = [item for item in records if item["result"] == "returned"]
    value = {
        "schema_version": RETRIEVAL_AUDIT_SCHEMA_VERSION,
        "status": "verified",
        "sha256": sha256_bytes(data),
        "record_count": len(records),
        "returned_count": len(returned),
        "returned_bytes": sum(int(item["byte_count"]) for item in returned),
    }
    value["receipt_sha256"] = sha256_bytes(canonical_json(value))
    return value


def _match_private_receipts_to_audit(
    receipts: Sequence[dict[str, Any]], records: Sequence[dict[str, Any]],
) -> None:
    returned = [item for item in records if item.get("result") == "returned"]
    if not receipts or not returned:
        raise RuntimeError("v0.3 explore did not prove an authorized source snippet")
    audit_ids: list[str] = []
    for item in returned:
        anchor = item.get("anchor")
        if not isinstance(anchor, dict):
            raise RuntimeError("v0.3 returned audit has no line anchor")
        audit_ids.append(_private_snippet_access_id(
            str(item.get("content_sha256")), anchor.get("start_line"),
            anchor.get("end_line"), item.get("byte_count"),
        ))
    receipt_ids = [str(item.get("access_id", "")) for item in receipts]
    if (any(re.fullmatch(r"[0-9a-f]{64}", item) is None for item in receipt_ids)
            or len(audit_ids) != len(set(audit_ids))
            or len(receipt_ids) != len(set(receipt_ids))
            or set(audit_ids) != set(receipt_ids)):
        raise RuntimeError(
            "v0.3 source snippets and returned audit records are not one-to-one"
        )


def _mcp_overrides(
    arm: str, *, workspace: Path, v02_python: str, v03_python: str,
    codegraph_command: str, sandbox_boundary: dict[str, Any] | None = None,
    audit_overlay: AuditOverlay | None = None,
) -> list[str]:
    if arm == "native":
        return []
    if (arm == "hlsgraph-v03" and sandbox_boundary is not None
            and sandbox_boundary.get("filesystem_policy") == SANDBOX_FILESYSTEM_POLICY
            and audit_overlay is None):
        raise RuntimeError("v0.3 MCP requires its per-cell retrieval audit overlay")
    if arm != "hlsgraph-v03" and audit_overlay is not None:
        raise RuntimeError("only the v0.3 MCP may receive a retrieval audit overlay")
    if arm == "codegraph":
        parts = _codegraph_parts(codegraph_command)
        server_command, server_args = parts[0], [*parts[1:], "serve", "--mcp"]
        server_env = dict(CODEGRAPH_ENV)
        values = {
            "mcp_servers.codegraph.command": server_command,
            "mcp_servers.codegraph.args": server_args,
            **{
                f"mcp_servers.codegraph.env.{key}": value
                for key, value in CODEGRAPH_ENV.items()
            },
        }
    else:
        python = v02_python if arm == "hlsgraph-v02" else v03_python
        mode = "all" if arm == "hlsgraph-v02" else "explore"
        server_command = python
        server_args = ["-m", "hlsgraph.mcp.server", str(workspace.resolve())]
        server_env = {"HLSGRAPH_MCP_TOOLS": mode}
        values = {
            "mcp_servers.hlsgraph.command": server_command,
            "mcp_servers.hlsgraph.args": server_args,
            "mcp_servers.hlsgraph.env.HLSGRAPH_MCP_TOOLS": mode,
        }
    if sandbox_boundary is not None and sandbox_boundary.get("filesystem_policy") is not None:
        command, args = _contained_mcp_command(
            arm=arm, workspace=workspace, server_command=server_command,
            server_args=server_args, server_env=server_env,
            sandbox_boundary=sandbox_boundary,
            audit_overlay=audit_overlay,
        )
        server_name = "codegraph" if arm == "codegraph" else "hlsgraph"
        values[f"mcp_servers.{server_name}.command"] = command
        values[f"mcp_servers.{server_name}.args"] = args
        # Bubblewrap receives a deliberately empty environment.  Keeping the
        # server-specific variables here would expose them to the unsandboxed
        # launcher before the namespace exists, so they are set only by bwrap.
        for key in tuple(values):
            if key.startswith(f"mcp_servers.{server_name}.env."):
                del values[key]
    output: list[str] = []
    for key, value in values.items():
        output.extend(["-c", f"{key}={_toml(value)}"])
    return output


def _contained_mcp_command(
    *, arm: str, workspace: Path, server_command: str,
    server_args: list[str], server_env: dict[str, str],
    sandbox_boundary: dict[str, Any],
    extra_bindings: Sequence[tuple[Path, Path]] = (),
    audit_overlay: AuditOverlay | None = None,
    audit_overlay_require_empty: bool = True,
    allow_system_command: bool = False,
) -> tuple[str, list[str]]:
    """Build the sole OS boundary for one local treatment MCP server.

    Codex 0.144 launches local stdio MCP servers directly from its orchestrator;
    the model shell permission profile does not contain those children.  The
    configured MCP command is therefore bubblewrap itself, not the product
    server.  No host root, HOME, proxy, CODEX_HOME, or network namespace is
    inherited into the server.
    """

    if arm not in {"codegraph", "hlsgraph-v02", "hlsgraph-v03"}:
        raise RuntimeError("native arm has no MCP containment command")
    contract = sandbox_boundary.get("mcp_containment")
    if not isinstance(contract, dict):
        raise RuntimeError("official sandbox boundary lacks MCP containment")
    launcher = contract.get("launcher")
    if not isinstance(launcher, dict) or not isinstance(launcher.get("path"), str):
        raise RuntimeError("official MCP containment lacks its launcher")
    if contract.get("network_mode") != "unshare_all_no_share_net":
        raise RuntimeError("official MCP containment changes the network policy")
    runtime_entries = sandbox_boundary.get("runtime_allow_roots", {}).get(arm)
    exclusions = contract.get("shared_runtime_exclusions")
    if not isinstance(runtime_entries, list) or not isinstance(exclusions, list):
        raise RuntimeError("official MCP containment lacks exact runtime roots")
    excluded = {Path(os.path.abspath(str(item))) for item in exclusions}
    mounted_entries = [
        entry for entry in runtime_entries
        if Path(os.path.abspath(str(entry["path"]))) not in excluded
    ]
    mounts = [Path(os.path.abspath(str(entry["path"]))) for entry in mounted_entries]
    server_path = Path(os.path.abspath(server_command))
    owner = next((
        entry for entry in mounted_entries
        if server_path == Path(os.path.abspath(str(entry["path"])))
        or server_path.is_relative_to(Path(os.path.abspath(str(entry["path"]))))
    ), None)
    if allow_system_command:
        try:
            server_path.relative_to(Path("/usr"))
        except ValueError as exc:
            raise RuntimeError(
                "treatment MCP system executable is outside the fixed /usr root"
            ) from exc
    elif owner is None:
        raise RuntimeError("treatment MCP executable is outside its exact runtime roots")
    if not server_path.is_file():
        raise RuntimeError("treatment MCP executable is not a regular-file target")
    if server_path.is_symlink():
        owner_root = Path(os.path.abspath(str(owner["path"]))) if owner else None
        usr_root = Path("/usr").resolve(strict=True)
        resolved_in_usr = server_path.resolve(strict=True).is_relative_to(usr_root)
        locked_venv_link = (
            owner_root is not None and owner.get("kind") == "tree"
            and owner.get("algorithm") == SANDBOX_ALLOW_TREE_ALGORITHM
            and sandbox_allow_tree_identity(owner_root) == owner.get("sha256")
        )
        if (server_path.parent.resolve(strict=True) != server_path.parent
                or not resolved_in_usr
                or (not allow_system_command and not locked_venv_link)):
            raise RuntimeError(
                "treatment MCP launcher symlink is not locked to the fixed /usr system root"
            )
        current = server_path
        for _hop in range(32):
            if not current.is_symlink():
                break
            target = Path(os.readlink(current))
            current = Path(os.path.abspath(os.fspath(
                target if target.is_absolute() else current.parent / target
            )))
            in_owner = owner_root is not None and current.is_relative_to(owner_root)
            in_usr = current.is_relative_to(usr_root)
            if not (in_owner or in_usr):
                raise RuntimeError(
                    "treatment MCP launcher symlink hop escapes its locked tree and /usr"
                )
        else:
            raise RuntimeError("treatment MCP launcher symlink chain is too deep")
        if current.is_symlink() or not current.is_file() or not current.is_relative_to(usr_root):
            raise RuntimeError(
                "treatment MCP launcher symlink chain does not terminate in /usr"
            )
    elif not stat.S_ISREG(server_path.lstat().st_mode):
        raise RuntimeError("treatment MCP executable is not a regular file")
    workspace = workspace.resolve()
    destination_parents: set[Path] = set()
    bindings = [(path, path) for path in mounts]
    bindings.extend((Path(source).resolve(), Path(destination))
                    for source, destination in extra_bindings)
    writable_bindings: list[tuple[Path, Path]] = []
    if audit_overlay is not None:
        source, destination, audit_parent_chain = audit_overlay
        source = Path(os.path.abspath(os.fspath(source)))
        destination = Path(os.path.abspath(os.fspath(destination)))
        expected_destination = workspace / RETRIEVAL_ACCESS_RELATIVE
        if arm != "hlsgraph-v03" or destination != expected_destination:
            raise RuntimeError("retrieval audit overlay has the wrong arm or destination")
        _verify_audit_overlay_file(
            source, require_empty=audit_overlay_require_empty,
            parent_chain=audit_parent_chain,
        )
        _verify_audit_placeholder(workspace)
        writable_bindings.append((source, destination))
    for path in [
        workspace,
        *(destination for _source, destination in bindings),
        *(destination for _source, destination in writable_bindings),
    ]:
        current = path.parent
        while current != Path(current.anchor):
            if current.as_posix() not in {"/usr", "/proc", "/dev", "/tmp"}:
                destination_parents.add(current)
            current = current.parent
    args = [
        "--die-with-parent", "--new-session", "--unshare-all",
        "--clearenv", "--cap-drop", "ALL",
        "--ro-bind", "/usr", "/usr",
        "--symlink", "usr/bin", "/bin",
        "--symlink", "usr/lib", "/lib",
        "--symlink", "usr/lib64", "/lib64",
        "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
        "--dir", "/tmp/home",
    ]
    for parent in sorted(destination_parents, key=lambda item: (len(item.parts), item.as_posix())):
        args.extend(["--dir", parent.as_posix()])
    args.extend(["--ro-bind", workspace.as_posix(), workspace.as_posix()])
    for source, destination in sorted(
        set(bindings), key=lambda item: (item[1].as_posix(), item[0].as_posix())
    ):
        args.extend(["--ro-bind", source.as_posix(), destination.as_posix()])
    for source, destination in writable_bindings:
        args.extend(["--bind", source.as_posix(), destination.as_posix()])
    safe_environment = {
        "HOME": "/tmp/home", "PATH": "/usr/bin:/bin", "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8", "TMPDIR": "/tmp", "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1", **server_env,
    }
    for key, value in sorted(safe_environment.items()):
        args.extend(["--setenv", key, value])
    args.extend(["--chdir", workspace.as_posix(), "--", str(server_path), *server_args])
    return str(Path(launcher["path"]).resolve()), args


def build_codex_command(
    record: dict[str, Any], *, work_root: Path, runs_root: Path,
    codex_command: str,
    v02_python: str, v03_python: str, codegraph_command: str,
    sandbox_boundary: dict[str, Any] | None = None,
    audit_overlay: AuditOverlay | None = None,
) -> list[str]:
    manifest = load_manifest()
    workspace = work_root / record["arm"] / record["corpus_id"]
    command = [
        *resolve_command_argv(codex_command), "--strict-config", "-a", "never",
    ]
    for feature in DISABLED_CODEX_FEATURES:
        command.extend(["--disable", feature])
    command.extend([
        "exec", "--ignore-user-config", "--ignore-rules",
        "--ephemeral", "--json", "--color", "never", "--skip-git-repo-check",
        "--model", manifest["model"]["id"],
        "-c", f"model_reasoning_effort={_toml(manifest['model']['reasoning_effort'])}",
        "--output-schema", str((HERE / manifest["codex_cli"]["output_schema"]).resolve()),
        "--cd", str(workspace.resolve()),
    ])
    if sandbox_boundary is None:
        # Command-construction tests and dry-run inspection do not execute this
        # fallback.  Every official run supplies the signed environment lock.
        fallback_codex_home = Path(
            os.environ.get("CODEX_HOME", work_root.parent / ".codex-eval-home")
        ).resolve()
        sandbox_boundary = {
            "work_root": work_root.resolve().as_posix(),
            "deny_roots": [
                PUBLIC_REPOSITORY.as_posix(), work_root.resolve().as_posix(),
                fallback_codex_home.as_posix(),
            ],
        }
    command.extend(_permission_overrides(
        workspace=workspace, work_root=work_root, runs_root=runs_root,
        sandbox_boundary=sandbox_boundary,
    ))
    command.extend(_mcp_overrides(
        record["arm"], workspace=workspace, v02_python=v02_python,
        v03_python=v03_python, codegraph_command=codegraph_command,
        sandbox_boundary=sandbox_boundary,
        audit_overlay=audit_overlay,
    ))
    command.append("-")
    return command


def _preflight(record: dict[str, Any], work_root: Path) -> Path:
    workspace = (work_root / record["arm"] / record["corpus_id"]).resolve()
    if not (workspace / "EVAL_PROVENANCE.json").is_file():
        raise RuntimeError(f"corpus is not materialized: {workspace}")
    if record["arm"] == "codegraph" and not (workspace / ".codegraph").exists():
        raise RuntimeError(f"CodeGraph index is missing: {workspace / '.codegraph'}")
    if record["arm"].startswith("hlsgraph-") and not (workspace / ".hlsgraph").exists():
        raise RuntimeError(f"HLSGraph index is missing: {workspace / '.hlsgraph'}")
    return workspace


def _require_isolated_work_root(work_root: Path) -> Path:
    """Require model corpora to live outside and not contain the public repo."""

    root = work_root.resolve()
    repository = PUBLIC_REPOSITORY
    is_junction = getattr(root, "is_junction", None)
    if root.is_symlink() or bool(callable(is_junction) and is_junction()):
        raise RuntimeError("official evaluation work root must not be linked")
    overlaps = False
    for child, parent in ((root, repository), (repository, root)):
        try:
            child.relative_to(parent)
            overlaps = True
        except ValueError:
            pass
    if overlaps:
        raise RuntimeError(
            "official evaluation work root must be outside and disjoint from the public repository"
        )
    return root


def _require_isolated_runs_root(
    runs_root: Path, *, work_root: Path, environment: dict[str, Any],
    allow_missing: bool,
) -> Path:
    """Require result bytes to remain on ext4 and outside every denied root."""

    root = require_official_ext4_directory(
        runs_root, "runs root", allow_missing=allow_missing,
    )
    boundary = environment.get("runtime_identity", {}).get("sandbox_boundary", {})
    deny_roots = boundary.get("deny_roots")
    if not isinstance(deny_roots, list) or not deny_roots:
        raise RuntimeError("official sandbox boundary lacks deny roots")
    protected = {Path(str(item)).resolve() for item in deny_roots}
    protected.add(work_root.resolve())
    resolved = root.resolve()
    for other in protected:
        overlaps = False
        for child, parent in ((resolved, other), (other, resolved)):
            try:
                child.relative_to(parent)
                overlaps = True
            except ValueError:
                pass
        if overlaps:
            raise RuntimeError(
                "official runs root must be disjoint from every sandbox boundary root"
            )
    return root


def _sandbox_canary_prefix(
    codex_command: str, workspace: Path, *, work_root: Path,
    runs_root: Path, sandbox_boundary: dict[str, Any],
) -> list[str]:
    # ``codex sandbox`` intentionally does not accept --strict-config.  The
    # same profile is still parsed and enforced here, while the real exec argv
    # carries --strict-config and is byte-bound into run-set.json.
    config = _permission_overrides(
        workspace=workspace, work_root=work_root, runs_root=runs_root,
        sandbox_boundary=sandbox_boundary,
    )
    # default_permissions is an exec selection; sandbox requires -P explicitly.
    without_default: list[str] = []
    for index in range(0, len(config), 2):
        if not config[index + 1].startswith("default_permissions="):
            without_default.extend(config[index:index + 2])
    return [
        *resolve_command_argv(codex_command), *without_default,
        "sandbox", "-P", PERMISSION_PROFILE, "--sandbox-state-disable-network",
        "-C", str(workspace.resolve()),
    ]


def _run_canary_command(
    command: list[str], *, timeout_seconds: int = 15,
) -> subprocess.CompletedProcess[str]:
    environment = official_process_environment()
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env=environment,
        creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0),
        start_new_session=(os.name != "nt"),
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        _terminate_process_tree(process)
        raise RuntimeError("Codex permission canary timed out; official suite is NO-GO") from exc
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def _run_stdio_mcp_exchange(
    command: list[str], requests: list[dict[str, Any]], *, cwd: Path,
    timeout_seconds: int = 30, raw_replies: dict[int, bytes] | None = None,
) -> dict[int, dict[str, Any]]:
    """Run one isolated stdio MCP handshake and return its identified replies."""
    process = subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", cwd=str(cwd), env={}, start_new_session=True,
        bufsize=1,
    )
    if process.stdin is None or process.stdout is None or process.stderr is None:
        _terminate_process_tree(process)
        raise RuntimeError("contained MCP did not expose stdio pipes")
    stdout_lines: Queue[str | None] = Queue()
    stderr_lines: list[str] = []

    def read_stdout() -> None:
        try:
            for line in process.stdout:
                stdout_lines.put(line)
        finally:
            stdout_lines.put(None)

    def read_stderr() -> None:
        stderr_lines.extend(process.stderr.readlines())

    threading.Thread(target=read_stdout, daemon=True).start()
    threading.Thread(target=read_stderr, daemon=True).start()
    replies: dict[int, dict[str, Any]] = {}
    deadline = time.monotonic() + timeout_seconds
    try:
        for request in requests:
            process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
            process.stdin.flush()
            identifier = request.get("id")
            if not isinstance(identifier, int):
                continue
            while identifier not in replies:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError
                try:
                    raw = stdout_lines.get(timeout=remaining)
                except Empty as exc:
                    raise TimeoutError from exc
                if raw is None:
                    raise RuntimeError("contained MCP closed stdout before replying")
                try:
                    value = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict) and isinstance(value.get("id"), int):
                    replies[value["id"]] = value
                    if raw_replies is not None:
                        raw_replies[value["id"]] = raw.encode("utf-8")
        process.stdin.close()
        process.wait(timeout=max(0.1, deadline - time.monotonic()))
    except (TimeoutError, subprocess.TimeoutExpired) as exc:
        _terminate_process_tree(process)
        raise RuntimeError("contained MCP canary timed out") from exc
    except Exception:
        _terminate_process_tree(process)
        raise
    stderr = "".join(stderr_lines)
    if process.returncode not in {0, None} or any(
        identifier not in replies or "result" not in replies[identifier]
        for identifier in (1, 2, 3)
    ):
        raise RuntimeError(
            "contained MCP failed initialize/tools-list/tool-call: " + stderr.strip()
        )
    return replies


def probe_contained_mcp_boundary(
    *, arm: str, workspace: Path, allowed: Sequence[Path], denied: Sequence[Path],
    port: int, sandbox_boundary: dict[str, Any],
) -> dict[str, Any]:
    """Exercise file, environment, and network denial from inside an MCP tool."""

    probe = (HERE / "mcp_boundary_probe.py").resolve()
    contained_probe = Path("/hlsgraph-eval-mcp-boundary-probe.py")
    launcher, args = _contained_mcp_command(
        arm=arm, workspace=workspace, server_command="/usr/bin/python3",
        server_args=[str(contained_probe)], server_env={},
        sandbox_boundary=sandbox_boundary,
        extra_bindings=[(probe, contained_probe)], allow_system_command=True,
    )
    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2024-11-05", "capabilities": {},
            "clientInfo": {"name": "hlsgraph-eval-canary", "version": "1"},
        }},
        {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {
            "name": "boundary_probe", "arguments": {
                "allowed": [str(item) for item in allowed],
                "denied": [str(item) for item in denied], "port": port,
            },
        }},
    ]
    replies = _run_stdio_mcp_exchange([launcher, *args], requests, cwd=workspace)
    result = replies[3]["result"].get("structuredContent")
    if not isinstance(result, dict):
        raise RuntimeError("contained MCP boundary probe returned no structured result")
    if (result.get("allowed") != [True] * len(allowed)
            or result.get("denied") != [False] * len(denied)
            or result.get("network") is not False
            or result.get("home") != "/tmp/home"
            or result.get("proxy_present") is not False
            or result.get("codex_home_present") is not False):
        raise RuntimeError("contained MCP escaped its exact filesystem/environment/network boundary")
    return result


def probe_real_treatment_mcp(
    *, arm: str, workspace: Path, v02_python: str, v03_python: str,
    codegraph_command: str, sandbox_boundary: dict[str, Any],
    audit_overlay: AuditOverlay | None = None,
    audit_descriptor: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Start the real treatment server inside bwrap and make one fixed read call."""

    if arm == "codegraph":
        parts = _codegraph_parts(codegraph_command)
        server_command, server_args = parts[0], [*parts[1:], "serve", "--mcp"]
        server_env = dict(CODEGRAPH_ENV)
        tool_name, tool_arguments = "codegraph_explore", {"query": "dataflow"}
    else:
        server_command = v02_python if arm == "hlsgraph-v02" else v03_python
        server_args = ["-m", "hlsgraph.mcp.server", str(workspace.resolve())]
        mode = "all" if arm == "hlsgraph-v02" else "explore"
        server_env = {"HLSGRAPH_MCP_TOOLS": mode}
        tool_name = "overview" if arm == "hlsgraph-v02" else "explore"
        tool_arguments = {} if arm == "hlsgraph-v02" else {
            "query": "load",
            "include_private_snippets": True,
            "include_predictions": False,
        }
    launcher, args = _contained_mcp_command(
        arm=arm, workspace=workspace, server_command=server_command,
        server_args=server_args, server_env=server_env,
        sandbox_boundary=sandbox_boundary, audit_overlay=audit_overlay,
    )
    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2024-11-05", "capabilities": {},
            "clientInfo": {"name": "hlsgraph-eval-real-mcp-canary", "version": "1"},
        }},
        {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {
            "name": tool_name, "arguments": tool_arguments,
        }},
    ]
    raw_replies: dict[int, bytes] = {}
    replies = _run_stdio_mcp_exchange(
        [launcher, *args], requests, cwd=workspace, raw_replies=raw_replies,
    )
    tools = replies[2]["result"].get("tools", [])
    names = [item.get("name") for item in tools if isinstance(item, dict)]
    call = replies[3]["result"]
    if tool_name not in names or call.get("isError") is True:
        raise RuntimeError(f"contained real {arm} MCP did not complete its fixed read call")
    request_bytes = json.dumps(
        requests[-1], separators=(",", ":"), ensure_ascii=True,
    ).encode("ascii")
    result = {
        "arm": arm, "tool": tool_name, "tool_count": len(names), "status": "pass",
        "tool_call_request_sha256": sha256_bytes(request_bytes),
        "raw_tool_response_sha256": sha256_bytes(raw_replies[3]),
    }
    if arm == "hlsgraph-v03":
        if audit_overlay is None:
            raise RuntimeError("v0.3 real MCP canary lacks its retrieval audit overlay")
        if not isinstance(audit_descriptor, dict):
            raise RuntimeError("v0.3 real MCP canary lacks its audit descriptor")
        receipts = _private_snippet_output_receipts({
            "type": "item.completed",
            "item": {"type": "mcp_tool_call", "result": call},
        })
        audit_data = _verify_audit_overlay_file(
            audit_overlay[0], require_empty=False,
            parent_chain=audit_overlay[2],
        )
        records = _parse_retrieval_audit(audit_data)
        _match_private_receipts_to_audit(receipts, records)
        audit_receipt = _retrieval_audit_receipt(audit_data)
        response_descriptor = _materialize_canary_response_artifact(
            Path(str(sandbox_boundary["work_root"])), audit_overlay, raw_replies[3],
        )
        result.update({
            "private_snippets_returned": True,
            "source_snippet_receipt_count": len(receipts),
            "source_access_ids": sorted(item["access_id"] for item in receipts),
            "retrieval_audit_sha256": audit_receipt["sha256"],
            "retrieval_audit_record_count": audit_receipt["record_count"],
            "retrieval_audit_returned_count": audit_receipt["returned_count"],
            "retrieval_audit_descriptor": audit_descriptor,
            "retrieval_audit_descriptor_sha256": sha256_bytes(
                canonical_json(audit_descriptor)
            ),
            "raw_tool_response_descriptor": response_descriptor,
            "raw_tool_response_descriptor_sha256": sha256_bytes(
                canonical_json(response_descriptor)
            ),
        })
    result["response_receipt_sha256"] = sha256_bytes(canonical_json(result))
    return result


def run_treatment_mcp_canaries(
    *, work_root: Path, runs_root: Path, environment: dict[str, Any],
    v02_python: str, v03_python: str, codegraph_command: str,
) -> dict[str, Any]:
    """Prove containment and usability for each real treatment MCP server."""

    root = _require_isolated_work_root(work_root)
    boundary = environment["runtime_identity"]["sandbox_boundary"]
    runtime_paths = {
        arm: [Path(str(entry["path"])) for entry in boundary["runtime_allow_roots"][arm]]
        for arm in ARM_IDS
    }
    runtime_sentinel = Path(boundary["runtime_root"]) / (
        ".hlsgraph-mcp-containment-" + secrets.token_hex(16)
    )
    runtime_sentinel.write_bytes(secrets.token_bytes(32))
    results: list[dict[str, Any]] = []
    canary_batch_id = secrets.token_hex(16)
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = int(listener.getsockname()[1])
        for arm, other in (
            ("codegraph", "hlsgraph-v02"),
            ("hlsgraph-v02", "hlsgraph-v03"),
            ("hlsgraph-v03", "hlsgraph-v02"),
        ):
            workspace = root / arm / "dataflow_gemm"
            allowed = [
                workspace / "EVAL_PROVENANCE.json",
                *[
                    path for path in runtime_paths[arm]
                    if path.resolve() != Path(boundary["mcp_containment"]
                                               ["shared_runtime_exclusions"][0]).resolve()
                ],
            ]
            denied = [
                root / arm / "cordic" / "EVAL_PROVENANCE.json",
                root / "native" / "dataflow_gemm" / "EVAL_PROVENANCE.json",
                root / "environment.lock.json", PUBLIC_REPOSITORY,
                Path(runs_root), Path(boundary["codex_home"]),
                Path(boundary["home_canary_root"]), runtime_sentinel,
                next(path for path in runtime_paths[other]
                     if path not in runtime_paths[arm]),
            ]
            boundary_result = probe_contained_mcp_boundary(
                arm=arm, workspace=workspace, allowed=allowed, denied=denied,
                port=port, sandbox_boundary=boundary,
            )
            audit_overlay = None
            descriptor = None
            if arm == "hlsgraph-v03":
                descriptor = _materialize_retrieval_audit_overlay(
                    root, batch_id=canary_batch_id,
                    run_id="real-mcp-canary-hlsgraph-v03",
                )
                audit_overlay = _audit_overlay_from_descriptor(
                    root, workspace, descriptor, batch_id=canary_batch_id,
                    run_id="real-mcp-canary-hlsgraph-v03", require_empty=True,
                )
            real_result = probe_real_treatment_mcp(
                arm=arm, workspace=workspace, v02_python=v02_python,
                v03_python=v03_python, codegraph_command=codegraph_command,
                sandbox_boundary=boundary, audit_overlay=audit_overlay,
                audit_descriptor=descriptor,
            )
            results.append({
                "arm": arm, "boundary": "pass", "real_server": real_result,
                "safe_home": boundary_result["home"],
            })
    finally:
        listener.close()
        runtime_sentinel.unlink(missing_ok=True)
    value = {
        "schema_version": "hlsgraph.agent_eval.mcp_containment_canary.v1",
        "native": "not_applicable_no_mcp",
        "treatments": results,
        "sandbox_boundary_sha256": boundary["identity_sha256"],
    }
    value["canary_sha256"] = sha256_bytes(canonical_json(value))
    return value


def _validate_treatment_mcp_canary(
    value: Any, sandbox_boundary: dict[str, Any],
) -> dict[str, Any]:
    message = "treatment MCP canary is stale, incomplete, or relabelled"
    if not isinstance(value, dict):
        raise RuntimeError("official run set lacks the treatment MCP canary")
    unhashed = {key: item for key, item in value.items() if key != "canary_sha256"}
    treatments = value.get("treatments")
    if (value.get("schema_version")
            != "hlsgraph.agent_eval.mcp_containment_canary.v1"
            or value.get("native") != "not_applicable_no_mcp"
            or value.get("sandbox_boundary_sha256") != sandbox_boundary["identity_sha256"]
            or not isinstance(treatments, list)
            or [item.get("arm") for item in treatments] != [
                "codegraph", "hlsgraph-v02", "hlsgraph-v03",
            ]
            or value.get("canary_sha256") != sha256_bytes(canonical_json(unhashed))):
        raise RuntimeError(message)
    expected_calls = {
        "codegraph": ("codegraph_explore", {"query": "dataflow"}),
        "hlsgraph-v02": ("overview", {}),
        "hlsgraph-v03": ("explore", {
            "query": "load", "include_private_snippets": True,
            "include_predictions": False,
        }),
    }
    by_arm: dict[str, dict[str, Any]] = {}
    for treatment in treatments:
        if (not isinstance(treatment, dict) or treatment.get("boundary") != "pass"
                or treatment.get("safe_home") != "/tmp/home"
                or not isinstance(treatment.get("real_server"), dict)):
            raise RuntimeError(message)
        arm = treatment.get("arm")
        result = treatment["real_server"]
        if arm not in expected_calls or result.get("arm") != arm or result.get("status") != "pass":
            raise RuntimeError(message)
        tool_name, arguments = expected_calls[arm]
        request = {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {
            "name": tool_name, "arguments": arguments,
        }}
        request_hash = sha256_bytes(json.dumps(
            request, separators=(",", ":"), ensure_ascii=True,
        ).encode("ascii"))
        receipt_body = {
            key: item for key, item in result.items()
            if key != "response_receipt_sha256"
        }
        if (result.get("tool") != tool_name
                or result.get("tool_call_request_sha256") != request_hash
                or re.fullmatch(r"[0-9a-f]{64}", str(
                    result.get("raw_tool_response_sha256", "")
                )) is None
                or result.get("response_receipt_sha256")
                != sha256_bytes(canonical_json(receipt_body))):
            raise RuntimeError(message)
        by_arm[str(arm)] = result
    v03 = by_arm["hlsgraph-v03"]
    descriptor = v03.get("retrieval_audit_descriptor")
    if (not isinstance(descriptor, dict)
            or v03.get("retrieval_audit_descriptor_sha256")
            != sha256_bytes(canonical_json(descriptor))):
        raise RuntimeError(message)
    relative = safe_relative_path(str(descriptor.get("path", "")))
    parts = relative.parts
    if (len(parts) != 4 or parts[0] != ".hlsgraph-eval-boundary"
            or re.fullmatch(r"[0-9a-f]{32}", parts[1]) is None
            or parts[2] != "retrieval-audit"
            or parts[3] != "real-mcp-canary-hlsgraph-v03.jsonl"):
        raise RuntimeError(message)
    root = Path(str(sandbox_boundary.get("work_root", "")))
    overlay = _audit_overlay_from_descriptor(
        root, root / "hlsgraph-v03" / "dataflow_gemm", descriptor,
        batch_id=parts[1], run_id="real-mcp-canary-hlsgraph-v03",
        require_empty=False,
    )
    if overlay is None:
        raise RuntimeError(message)
    audit_data = _verify_audit_overlay_file(
        overlay[0], require_empty=False, parent_chain=overlay[2],
    )
    records = _parse_retrieval_audit(audit_data)
    response_descriptor = v03.get("raw_tool_response_descriptor")
    expected_response_relative = (
        Path(".hlsgraph-eval-boundary") / parts[1] / "retrieval-audit"
        / "real-mcp-canary-hlsgraph-v03.response.json"
    )
    if (not isinstance(response_descriptor, dict)
            or set(response_descriptor) != {
                "schema_version", "path", "size", "sha256", "parent_chain",
                "parent_chain_sha256",
            }
            or response_descriptor.get("schema_version")
            != "hlsgraph.agent_eval.raw_mcp_response.v1"
            or response_descriptor.get("path") != expected_response_relative.as_posix()
            or not isinstance(response_descriptor.get("parent_chain"), list)
            or response_descriptor.get("parent_chain_sha256")
            != sha256_bytes(canonical_json(response_descriptor.get("parent_chain")))
            or v03.get("raw_tool_response_descriptor_sha256")
            != sha256_bytes(canonical_json(response_descriptor))):
        raise RuntimeError(message)
    response_path = root / expected_response_relative
    raw_response = _stable_audit_bytes(
        response_path, parent_chain=response_descriptor["parent_chain"],
        max_bytes=8 * 1024 * 1024,
    )
    if (response_descriptor.get("size") != len(raw_response)
            or response_descriptor.get("sha256") != sha256_bytes(raw_response)
            or v03.get("raw_tool_response_sha256") != sha256_bytes(raw_response)):
        raise RuntimeError(message)
    try:
        raw_reply = json.loads(raw_response)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(message) from exc
    if (not isinstance(raw_reply, dict) or raw_reply.get("id") != 3
            or not isinstance(raw_reply.get("result"), dict)):
        raise RuntimeError(message)
    raw_receipts = _private_snippet_output_receipts({
        "type": "item.completed",
        "item": {"type": "mcp_tool_call", "result": raw_reply["result"]},
    })
    _match_private_receipts_to_audit(raw_receipts, records)
    returned = [item for item in records if item["result"] == "returned"]
    audit_ids = [
        _private_snippet_access_id(
            item["content_sha256"], item["anchor"].get("start_line"),
            item["anchor"].get("end_line"), item["byte_count"],
        )
        for item in returned
    ]
    reported_ids = v03.get("source_access_ids")
    raw_ids = sorted(item["access_id"] for item in raw_receipts)
    receipt = _retrieval_audit_receipt(audit_data)
    if (v03.get("private_snippets_returned") is not True
            or not isinstance(reported_ids, list) or not reported_ids
            or len(reported_ids) != len(set(reported_ids))
            or len(audit_ids) != len(set(audit_ids))
            or sorted(reported_ids) != sorted(audit_ids)
            or sorted(reported_ids) != raw_ids
            or v03.get("source_snippet_receipt_count") != len(reported_ids)
            or v03.get("retrieval_audit_sha256") != receipt["sha256"]
            or v03.get("retrieval_audit_record_count") != receipt["record_count"]
            or v03.get("retrieval_audit_returned_count") != receipt["returned_count"]):
        raise RuntimeError(message)
    return value


def run_permission_canaries(
    *, codex_command: str, work_root: Path, runs_root: Path,
    environment: dict[str, Any],
) -> dict[str, Any]:
    """Prove each arm's minimal exact allowlist before any model call."""

    root = _require_isolated_work_root(work_root)
    runtime = environment.get("runtime_identity")
    if not isinstance(runtime, dict) or not isinstance(
        runtime.get("sandbox_boundary"), dict
    ):
        raise RuntimeError("environment lock lacks a sandbox boundary")
    boundary = runtime["sandbox_boundary"]
    result_root = Path(os.path.abspath(os.fspath(runs_root)))
    if Path(str(boundary.get("work_root", ""))).resolve() != root:
        raise RuntimeError("permission canary work root differs from the environment lock")
    if Path(str(boundary.get("public_repository", ""))).resolve() != PUBLIC_REPOSITORY:
        raise RuntimeError("permission canary public repository differs from the lock")
    same_arm_corpus = next(
        item["id"] for item in load_corpus_lock()["corpora"]
        if item["id"] != "dataflow_gemm"
    )
    public_gold = HERE / "questions.jsonl"
    if not public_gold.is_file():
        raise RuntimeError("permission canary inputs are missing")
    python = "/usr/bin/python3" if os.name == "posix" else str(Path(sys.executable).resolve())
    if not Path(python).is_file():
        raise RuntimeError("official permission canary requires /usr/bin/python3")
    access_script = (
        "import os,pathlib,sys;"
        "p=pathlib.Path(sys.argv[1]);"
        "next(os.scandir(p),None) if p.is_dir() else p.open('rb').read(1);"
        "sys.exit(0)"
    )

    token = secrets.token_hex(16)
    codex_home = Path(str(boundary["codex_home"]))
    external_root = Path(str(boundary["external_canary_root"]))
    home_root = Path(str(boundary["home_canary_root"])).resolve()
    if not any(
        home_root == Path(str(item)).resolve()
        or home_root.is_relative_to(Path(str(item)).resolve())
        for item in boundary["home_roots"]
    ):
        raise RuntimeError("current user home is absent from the sandbox boundary")
    boundary_root = root / ".hlsgraph-eval-boundary"
    boundary_root_created = not boundary_root.exists()
    cache_root = root / "_cache"
    codex_sentinel = codex_home / f".hlsgraph-eval-canary-{token}"
    home_sentinel = home_root / f".hlsgraph-eval-home-canary-{token}"
    boundary_sentinel = boundary_root / f"permission-canary-{token}.bin"
    cache_sentinel = cache_root / f"permission-canary-{token}.bin"
    runs_sentinel = result_root / f".hlsgraph-eval-runs-canary-{token}.bin"
    external_sentinel = external_root / "private-like-sentinel.bin"
    runtime_root = Path(str(boundary["runtime_root"]))
    runtime_sentinel = runtime_root / f".hlsgraph-eval-runtime-canary-{token}.bin"
    drvfs_directories: list[Path] = []
    drvfs_sentinels: list[Path] = []
    try:
        codex_sentinel.write_bytes(secrets.token_bytes(32))
        home_sentinel.write_bytes(secrets.token_bytes(32))
        boundary_root.mkdir(parents=False, exist_ok=True)
        boundary_sentinel.write_bytes(secrets.token_bytes(32))
        cache_sentinel.write_bytes(secrets.token_bytes(32))
        runs_sentinel.write_bytes(secrets.token_bytes(32))
        runtime_sentinel.write_bytes(secrets.token_bytes(32))
        external_root.mkdir(parents=False, exist_ok=False)
        external_sentinel.write_bytes(secrets.token_bytes(32))
        for drvfs_root_text in boundary["drvfs_roots"]:
            drvfs_root = Path(str(drvfs_root_text))
            directory = Path(tempfile.mkdtemp(
                prefix=".hlsgraph-eval-boundary-", dir=str(drvfs_root),
            ))
            sentinel = directory / "private-like-sentinel.bin"
            sentinel.write_bytes(secrets.token_bytes(32))
            drvfs_directories.append(directory)
            drvfs_sentinels.append(sentinel)

        all_runtime_paths = {
            arm: [Path(str(entry["path"])) for entry in boundary["runtime_allow_roots"][arm]]
            for arm in ARM_IDS
        }
        for arm_index, arm in enumerate(ARM_IDS):
            workspace = root / arm / "dataflow_gemm"
            allowed = workspace / "EVAL_PROVENANCE.json"
            same_arm_sibling = (
                root / arm / same_arm_corpus / "EVAL_PROVENANCE.json"
            )
            other_arm = ARM_IDS[(arm_index + 1) % len(ARM_IDS)]
            other_arm_sibling = (
                root / other_arm / "dataflow_gemm" / "EVAL_PROVENANCE.json"
            )
            if not all(
                path.is_file()
                for path in (allowed, same_arm_sibling, other_arm_sibling)
            ):
                raise RuntimeError("permission canary workspace inputs are missing")
            prefix = _sandbox_canary_prefix(
                codex_command, workspace, work_root=root, runs_root=result_root,
                sandbox_boundary=boundary,
            )

            def require_allowed(path: Path, label: str) -> None:
                result = _run_canary_command([
                    *prefix, python, "-c", access_script, str(path),
                ])
                if result.returncode != 0:
                    raise RuntimeError(
                        f"Codex permission profile cannot read {label}: "
                        + (result.stderr.strip() or result.stdout.strip())
                    )

            def require_denied(path: Path, label: str) -> None:
                result = _run_canary_command([
                    *prefix, python, "-c", access_script, str(path),
                ])
                if result.returncode == 0:
                    raise RuntimeError(
                        f"Codex permission profile allowed {label}; official suite is NO-GO"
                    )

            require_allowed(allowed, f"the {arm} isolated corpus workspace")
            for runtime_path in all_runtime_paths[arm]:
                require_allowed(runtime_path, f"the exact {arm} runtime allow root")
            other_runtime = next(
                path for candidate_arm in ARM_IDS if candidate_arm != arm
                for path in all_runtime_paths[candidate_arm]
                if path not in all_runtime_paths[arm]
            )
            require_denied(same_arm_sibling, f"a {arm} same-arm sibling workspace")
            require_denied(other_arm_sibling, f"a {arm} cross-arm workspace")
            require_denied(boundary_sentinel, f"the {arm} boundary control")
            require_denied(cache_sentinel, f"the {arm} cache control")
            require_denied(root / "environment.lock.json", f"the {arm} lock control")
            require_denied(runs_sentinel, f"the {arm} runs root")
            require_denied(public_gold, f"the {arm} public gold repository")
            require_denied(codex_sentinel, f"the {arm} dedicated CODEX_HOME")
            require_denied(home_sentinel, f"the {arm} user home")
            require_denied(external_sentinel, f"the {arm} external private-like root")
            require_denied(runtime_sentinel, f"the {arm} undeclared runtime content")
            require_denied(other_runtime, f"the {arm} other-arm runtime")
            for sentinel in drvfs_sentinels:
                require_denied(sentinel, f"a {arm} drvfs read at {sentinel.parent.parent}")
    finally:
        try:
            codex_sentinel.unlink(missing_ok=True)
        except OSError:
            pass
        try:
            home_sentinel.unlink(missing_ok=True)
        except OSError:
            pass
        try:
            boundary_sentinel.unlink(missing_ok=True)
            if boundary_root_created:
                boundary_root.rmdir()
        except OSError:
            pass
        try:
            cache_sentinel.unlink(missing_ok=True)
        except OSError:
            pass
        try:
            runs_sentinel.unlink(missing_ok=True)
        except OSError:
            pass
        try:
            runtime_sentinel.unlink(missing_ok=True)
        except OSError:
            pass
        for directory in drvfs_directories:
            shutil.rmtree(directory, ignore_errors=True)
        shutil.rmtree(external_root, ignore_errors=True)

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        network_script = (
            "import socket,sys;"
            "s=socket.create_connection(('127.0.0.1',int(sys.argv[1])),1);"
            "s.close();sys.exit(0)"
        )
        network_results = []
        for arm in ARM_IDS:
            workspace = root / arm / "dataflow_gemm"
            prefix = _sandbox_canary_prefix(
                codex_command, workspace, work_root=root, runs_root=result_root,
                sandbox_boundary=boundary,
            )
            network_results.append(_run_canary_command([
                *prefix, python, "-c", network_script, str(port),
            ]))
    finally:
        listener.close()
    if any(result.returncode == 0 for result in network_results):
        raise RuntimeError(
            "Codex permission profile allowed a TCP connection; official suite is NO-GO"
        )
    value = {
        "schema_version": "hlsgraph.agent_eval.permission_canary.v4",
        "profile": PERMISSION_PROFILE,
        "filesystem_policy": SANDBOX_FILESYSTEM_POLICY,
        "filesystem_base_token": SANDBOX_MINIMAL_TOKEN,
        "arms_tested": list(ARM_IDS),
        "workspace_read": "pass",
        "runtime_allow_roots_read": "pass",
        "sibling_workspace_read": "denied",
        "same_arm_sibling_read": "denied",
        "other_arm_sibling_read": "denied",
        "control_roots_read": "denied",
        "runs_root_read": "denied",
        "runs_root_sha256": hashlib.sha256(
            result_root.as_posix().encode("utf-8")
        ).hexdigest(),
        "public_gold_read": "denied",
        "codex_home_read": "denied",
        "user_home_read": "denied",
        "external_private_read": "denied",
        "undeclared_runtime_read": "denied",
        "other_arm_runtime_read": "denied",
        "drvfs_mount_reads": "denied",
        "drvfs_mount_count": len(boundary["drvfs_roots"]),
        "drvfs_roots_sha256": sha256_bytes(canonical_json(boundary["drvfs_roots"])),
        "network_socket": "denied",
        "sandbox_boundary_sha256": boundary["identity_sha256"],
        "runtime_allow_roots_sha256": boundary["runtime_allow_roots_sha256"],
        "public_repository_sha256": hashlib.sha256(
            str(PUBLIC_REPOSITORY).encode("utf-8")
        ).hexdigest(),
    }
    value["canary_sha256"] = sha256_bytes(canonical_json(value))
    return value


def _validate_permission_canary(
    value: Any, sandbox_boundary: dict[str, Any], *, runs_root: Path,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError("official run set lacks a permission-canary result")
    unhashed = {key: item for key, item in value.items() if key != "canary_sha256"}
    expected = {
        "schema_version": "hlsgraph.agent_eval.permission_canary.v4",
        "profile": PERMISSION_PROFILE,
        "filesystem_policy": SANDBOX_FILESYSTEM_POLICY,
        "filesystem_base_token": SANDBOX_MINIMAL_TOKEN,
        "arms_tested": list(ARM_IDS),
        "workspace_read": "pass",
        "runtime_allow_roots_read": "pass",
        "sibling_workspace_read": "denied",
        "same_arm_sibling_read": "denied",
        "other_arm_sibling_read": "denied",
        "control_roots_read": "denied",
        "runs_root_read": "denied",
        "runs_root_sha256": hashlib.sha256(
            Path(os.path.abspath(os.fspath(runs_root))).as_posix().encode("utf-8")
        ).hexdigest(),
        "public_gold_read": "denied",
        "codex_home_read": "denied",
        "user_home_read": "denied",
        "external_private_read": "denied",
        "undeclared_runtime_read": "denied",
        "other_arm_runtime_read": "denied",
        "drvfs_mount_reads": "denied",
        "drvfs_mount_count": len(sandbox_boundary["drvfs_roots"]),
        "drvfs_roots_sha256": sha256_bytes(canonical_json(
            sandbox_boundary["drvfs_roots"]
        )),
        "network_socket": "denied",
        "sandbox_boundary_sha256": sandbox_boundary["identity_sha256"],
        "runtime_allow_roots_sha256": sandbox_boundary["runtime_allow_roots_sha256"],
        "public_repository_sha256": hashlib.sha256(
            str(PUBLIC_REPOSITORY).encode("utf-8")
        ).hexdigest(),
    }
    if (unhashed != expected
            or value.get("canary_sha256") != sha256_bytes(canonical_json(unhashed))):
        raise RuntimeError("permission-canary result is stale, incomplete, or relabelled")
    return value


def _stable_boundary_bytes(path: Path, *, max_bytes: int = 1024) -> bytes:
    """Read a small canary without following a final-component link."""

    descriptor = -1
    try:
        before = path.lstat()
        flags = os.O_RDONLY | int(getattr(os, "O_BINARY", 0)) | int(
            getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        data = os.read(descriptor, max_bytes + 1)
        closed = os.fstat(descriptor)
        current = path.lstat()
    except OSError as exc:
        raise RuntimeError("boundary canary cannot be read safely") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    identity = lambda value: (
        int(value.st_dev), int(value.st_ino), int(value.st_size),
        int(value.st_mtime_ns),
    )
    if (not stat.S_ISREG(before.st_mode) or not stat.S_ISREG(opened.st_mode)
            or len(data) > max_bytes or identity(before) != identity(opened)
            or identity(opened) != identity(closed)
            or identity(closed) != identity(current)):
        raise RuntimeError("boundary canary changed while it was read")
    return data


def _verify_boundary_canary(
    work_root: Path, descriptor: Any, *, batch_id: str, run_id: str,
) -> bytes:
    root = _require_isolated_work_root(work_root)
    if not isinstance(descriptor, dict) or set(descriptor) != {"path", "sha256"}:
        raise RuntimeError("run-set cell has an invalid boundary-canary descriptor")
    relative = safe_relative_path(str(descriptor["path"]))
    expected = Path(".hlsgraph-eval-boundary") / batch_id / f"{run_id}.canary"
    if relative != expected:
        raise RuntimeError("run-set cell has a relabelled boundary canary")
    path = root / relative
    current = root
    for part in relative.parts:
        current = current / part
        is_junction = getattr(current, "is_junction", None)
        if current.is_symlink() or bool(callable(is_junction) and is_junction()):
            raise RuntimeError("boundary-canary path contains a link or junction")
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise RuntimeError("boundary canary escapes the isolated work root") from exc
    data = _stable_boundary_bytes(path)
    if (re.fullmatch(r"[0-9a-f]{64}", str(descriptor["sha256"])) is None
            or hashlib.sha256(data).hexdigest() != descriptor["sha256"]):
        raise RuntimeError("boundary-canary hash mismatch")
    return data


def _materialize_boundary_canary(
    work_root: Path, *, batch_id: str, run_id: str,
) -> dict[str, str]:
    root = _require_isolated_work_root(work_root)
    directory = root / ".hlsgraph-eval-boundary" / batch_id
    directory.mkdir(parents=True, exist_ok=True)
    current = root
    for part in directory.relative_to(root).parts:
        current = current / part
        is_junction = getattr(current, "is_junction", None)
        if current.is_symlink() or bool(callable(is_junction) and is_junction()):
            raise RuntimeError("boundary-canary directory contains a link or junction")
    path = directory / f"{run_id}.canary"
    token = ("hlsgraph-eval-boundary-" + secrets.token_hex(32)).encode("ascii")
    with path.open("xb") as stream:
        stream.write(token)
    relative = path.relative_to(root).as_posix()
    return {"path": relative, "sha256": hashlib.sha256(token).hexdigest()}


def _materialize_retrieval_audit_overlay(
    work_root: Path, *, batch_id: str, run_id: str,
) -> dict[str, Any]:
    root = _require_isolated_work_root(work_root)
    if (re.fullmatch(r"[0-9a-f]{32}", batch_id) is None
            or re.fullmatch(r"[A-Za-z0-9_.-]+", run_id) is None):
        raise RuntimeError("retrieval audit overlay has an invalid cell identity")
    directory = root / ".hlsgraph-eval-boundary" / batch_id / "retrieval-audit"
    directory.mkdir(parents=True, mode=0o700, exist_ok=True)
    current = root
    for part in directory.relative_to(root).parts:
        current /= part
        if _linked(current) or not current.is_dir():
            raise RuntimeError("retrieval audit overlay directory is linked")
    if os.name != "nt":
        os.chmod(directory, 0o700)
    path = directory / f"{run_id}.jsonl"
    flags = (os.O_WRONLY | os.O_CREAT | os.O_EXCL
             | int(getattr(os, "O_BINARY", 0))
             | int(getattr(os, "O_NOFOLLOW", 0)))
    parent_flags = (os.O_RDONLY | int(getattr(os, "O_DIRECTORY", 0))
                    | int(getattr(os, "O_NOFOLLOW", 0)))
    parent_descriptor = os.open(directory, parent_flags) if os.name != "nt" else -1
    try:
        parent_before = (
            os.fstat(parent_descriptor) if parent_descriptor >= 0 else directory.lstat()
        )
        descriptor = (os.open(path.name, flags, 0o600, dir_fd=parent_descriptor)
                      if parent_descriptor >= 0 and os.open in os.supports_dir_fd
                      else os.open(path, flags, 0o600))
        try:
            opened = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        parent_after = (
            os.fstat(parent_descriptor) if parent_descriptor >= 0 else directory.lstat()
        )
    finally:
        if parent_descriptor >= 0:
            os.close(parent_descriptor)
    if ((int(parent_before.st_dev), int(parent_before.st_ino), int(parent_before.st_mode))
            != (int(parent_after.st_dev), int(parent_after.st_ino),
                int(parent_after.st_mode))):
        raise RuntimeError("retrieval audit parent changed during creation")
    if os.name != "nt":
        os.chmod(path, 0o600)
    parent_chain = _audit_parent_chain(path)
    _verify_audit_overlay_file(
        path, require_empty=True, parent_chain=parent_chain,
    )
    if not stat.S_ISREG(opened.st_mode) or int(getattr(opened, "st_nlink", 1)) != 1:
        raise RuntimeError("retrieval audit overlay was not created as a regular file")
    return {
        "schema_version": RETRIEVAL_AUDIT_SCHEMA_VERSION,
        "status": "required",
        "path": path.relative_to(root).as_posix(),
        "destination": RETRIEVAL_ACCESS_RELATIVE.as_posix(),
        "initial_sha256": sha256_bytes(b""),
        "parent_chain": parent_chain,
        "parent_chain_sha256": sha256_bytes(canonical_json(parent_chain)),
    }


def _materialize_canary_response_artifact(
    work_root: Path, audit_overlay: AuditOverlay, data: bytes,
) -> dict[str, Any]:
    """Freeze the exact v0.3 canary reply outside the read-only workspace."""

    root = _require_isolated_work_root(work_root)
    audit_path, _destination, audit_parents = audit_overlay
    _verify_audit_parent_chain(audit_path, audit_parents)
    path = audit_path.with_name("real-mcp-canary-hlsgraph-v03.response.json")
    flags = (os.O_WRONLY | os.O_CREAT | os.O_EXCL
             | int(getattr(os, "O_BINARY", 0))
             | int(getattr(os, "O_NOFOLLOW", 0)))
    parent_flags = (os.O_RDONLY | int(getattr(os, "O_DIRECTORY", 0))
                    | int(getattr(os, "O_NOFOLLOW", 0)))
    parent_descriptor = os.open(path.parent, parent_flags) if os.name != "nt" else -1
    descriptor = -1
    try:
        descriptor = (os.open(path.name, flags, 0o600, dir_fd=parent_descriptor)
                      if parent_descriptor >= 0 and os.open in os.supports_dir_fd
                      else os.open(path, flags, 0o600))
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise RuntimeError("cannot freeze the raw treatment MCP response")
            view = view[written:]
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if parent_descriptor >= 0:
            os.close(parent_descriptor)
    if os.name != "nt":
        os.chmod(path, 0o600)
    parent_chain = _audit_parent_chain(path)
    frozen = _stable_audit_bytes(
        path, parent_chain=parent_chain, max_bytes=8 * 1024 * 1024,
    )
    if frozen != data:
        raise RuntimeError("raw treatment MCP response changed while frozen")
    return {
        "schema_version": "hlsgraph.agent_eval.raw_mcp_response.v1",
        "path": path.relative_to(root).as_posix(),
        "size": len(data), "sha256": sha256_bytes(data),
        "parent_chain": parent_chain,
        "parent_chain_sha256": sha256_bytes(canonical_json(parent_chain)),
    }


def _audit_overlay_from_descriptor(
    work_root: Path, workspace: Path, descriptor: Any, *,
    batch_id: str, run_id: str, require_empty: bool,
) -> AuditOverlay | None:
    if not isinstance(descriptor, dict):
        raise RuntimeError("run-set cell lacks its retrieval audit contract")
    if descriptor.get("status") == "not_applicable":
        if set(descriptor) != {"schema_version", "status"}:
            raise RuntimeError("not-applicable retrieval audit contract has extra fields")
        return None
    expected_relative = (
        Path(".hlsgraph-eval-boundary") / batch_id / "retrieval-audit"
        / f"{run_id}.jsonl"
    )
    if (set(descriptor) != {
            "schema_version", "status", "path", "destination", "initial_sha256",
            "parent_chain", "parent_chain_sha256",
        }
            or descriptor.get("schema_version") != RETRIEVAL_AUDIT_SCHEMA_VERSION
            or descriptor.get("status") != "required"
            or descriptor.get("path") != expected_relative.as_posix()
            or descriptor.get("destination") != RETRIEVAL_ACCESS_RELATIVE.as_posix()
            or descriptor.get("initial_sha256") != sha256_bytes(b"")
            or not isinstance(descriptor.get("parent_chain"), list)
            or descriptor.get("parent_chain_sha256")
            != sha256_bytes(canonical_json(descriptor.get("parent_chain")))):
        raise RuntimeError("run-set retrieval audit contract is invalid or relabelled")
    root = _require_isolated_work_root(work_root)
    source = root / expected_relative
    destination = Path(os.path.abspath(os.fspath(workspace))) / RETRIEVAL_ACCESS_RELATIVE
    parent_chain = tuple(descriptor["parent_chain"])
    _verify_audit_overlay_file(
        source, require_empty=require_empty, parent_chain=parent_chain,
    )
    _verify_audit_placeholder(workspace)
    return source, destination, parent_chain


def build_run_set(
    plan: list[dict[str, Any]], *, work_root: Path, runs_root: Path,
    environment: dict[str, Any],
    environment_lock_sha256: str, codex_command: str, v02_python: str,
    v03_python: str, codegraph_command: str,
    permission_canary: dict[str, Any] | None = None,
    mcp_containment_canary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Freeze each expected cell independently of its eventual run metadata."""
    timeout_seconds = load_manifest()["codex_cli"]["timeout_seconds"]
    if any(record.get("timeout_seconds") != timeout_seconds for record in plan):
        raise RuntimeError("run plan changes the frozen timeout")
    runs_root = _require_isolated_runs_root(
        runs_root, work_root=work_root, environment=environment,
        allow_missing=False,
    )
    boundary = environment["runtime_identity"]["sandbox_boundary"]
    permission_canary = _validate_permission_canary(
        permission_canary, boundary, runs_root=runs_root,
    )
    mcp_containment_canary = _validate_treatment_mcp_canary(
        mcp_containment_canary, boundary,
    )
    cells: list[dict[str, Any]] = []
    questions = _question_map()
    batch_id = secrets.token_hex(16)
    for record in sorted(plan, key=lambda item: item["run_id"]):
        workspace = verify_prepared_workspace(
            environment, work_root, record["arm"], record["corpus_id"],
        )
        retrieval_audit = (
            _materialize_retrieval_audit_overlay(
                work_root, batch_id=batch_id, run_id=record["run_id"],
            )
            if record["arm"] == "hlsgraph-v03"
            else {
                "schema_version": RETRIEVAL_AUDIT_SCHEMA_VERSION,
                "status": "not_applicable",
            }
        )
        audit_overlay = _audit_overlay_from_descriptor(
            work_root, work_root / record["arm"] / record["corpus_id"],
            retrieval_audit, batch_id=batch_id, run_id=record["run_id"],
            require_empty=True,
        )
        trace_challenge = sha256_bytes(canonical_json({
            "domain": "hlsgraph.agent_eval.trace_challenge.v1",
            "batch_id": batch_id,
            "environment_lock_sha256": environment_lock_sha256,
            "run_id": record["run_id"],
        }))
        prompt = build_prompt(
            questions[record["question_id"]], record["arm"],
            trace_challenge=trace_challenge,
        )
        command = build_codex_command(
            record, work_root=work_root, runs_root=runs_root,
            codex_command=codex_command,
            v02_python=v02_python, v03_python=v03_python,
            codegraph_command=codegraph_command, sandbox_boundary=boundary,
            audit_overlay=audit_overlay,
        )
        boundary_canary = _materialize_boundary_canary(
            work_root, batch_id=batch_id, run_id=record["run_id"],
        )
        contract = {
            **record,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "trace_challenge": trace_challenge,
            "command_argv": command,
            "workspace_identity_sha256": workspace["workspace_identity_sha256"],
            "boundary_canary": boundary_canary,
            "retrieval_audit": retrieval_audit,
        }
        contract["run_contract_sha256"] = sha256_bytes(canonical_json(contract))
        cells.append(contract)
    value = {
        "schema_version": "hlsgraph.agent_eval.run_set.v1",
        "batch_id": batch_id,
        "suite_asset_sha256": asset_digest(),
        "evaluation_harness_sha256": harness_digest(),
        "environment_lock_sha256": environment_lock_sha256,
        "runs_root": runs_root.as_posix(),
        "timeout_seconds": timeout_seconds,
        "permission_canary": permission_canary,
        "mcp_containment_canary": mcp_containment_canary,
        "cells": cells,
    }
    value["run_set_sha256"] = sha256_bytes(canonical_json(value))
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            check=False, timeout=10,
        )
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _runtime_identity_preflight(
    record: dict[str, Any], environment: dict[str, Any], *,
    v02_python: str, v03_python: str, codegraph_command: str,
    full_codegraph: bool = False,
) -> None:
    if environment.get("suite_asset_sha256") != asset_digest():
        raise RuntimeError("environment lock belongs to different frozen suite assets")
    manifest = load_manifest()
    if environment.get("codegraph_revision") != manifest["arms"][1]["revision"]:
        raise RuntimeError("environment lock has the wrong CodeGraph revision")
    runtime = environment.get("runtime_identity")
    if not isinstance(runtime, dict):
        raise RuntimeError("environment lock lacks runtime identity")
    arm = record["arm"]
    if arm == "codegraph":
        expected = environment.get("codegraph_entrypoint")
        expected_runtime_entrypoint = runtime.get("codegraph_entrypoint")
        expected_node = runtime.get("node")
        expected_build = runtime.get("codegraph_build")
        frozen_sha256 = manifest["arms"][1]["entrypoint_sha256"]
        parts = _codegraph_parts(codegraph_command)
        if (not isinstance(expected, dict) or expected != expected_runtime_entrypoint
                or not isinstance(expected_node, dict)
                or not isinstance(expected_build, dict)
                or environment.get("codegraph_build") != expected_build
                or len(parts) != 2
                or Path(parts[0]).name.casefold() not in {"node", "node.exe"}):
            raise RuntimeError("CodeGraph runtime identity is incomplete")
        node = Path(parts[0])
        entrypoint = Path(parts[1])
        package_lock = Path(str(expected_build.get("package_lock", {}).get("path", "")))
        npm_cli = Path(str(expected_build.get("npm", {}).get("path", "")))
        if (expected.get("sha256") != frozen_sha256
                or node.as_posix() != expected_node.get("path")
                or not node.is_absolute() or not node.is_file() or node.is_symlink()
                or sha256_file(node) != expected_node.get("sha256")
                or not entrypoint.is_absolute() or not entrypoint.is_file()
                or entrypoint.is_symlink()
                or entrypoint.as_posix() != expected.get("path")
                or entrypoint.name != expected.get("filename")
                or sha256_file(entrypoint) != frozen_sha256
                or not package_lock.is_absolute() or not package_lock.is_file()
                or package_lock.is_symlink()
                or sha256_file(package_lock)
                != expected_build.get("package_lock", {}).get("sha256")
                or not npm_cli.is_absolute() or not npm_cli.is_file()
                or npm_cli.is_symlink()
                or sha256_file(npm_cli) != expected_build.get("npm", {}).get("sha256")):
            raise RuntimeError("CodeGraph lightweight closure differs from preparation")
        if full_codegraph:
            try:
                current_build = capture_codegraph_build_identity(
                    repository=Path(expected_build["repository"]["path"]),
                    runtime_root=Path(runtime["sandbox_boundary"]["runtime_root"]),
                    node=node, npm_cli=npm_cli, entrypoint=entrypoint, full=True,
                )
            except (KeyError, ValueError, OSError) as exc:
                raise RuntimeError("CodeGraph full runtime closure cannot be verified") from exc
            if current_build != expected_build:
                raise RuntimeError("CodeGraph full runtime closure differs from preparation")
        return
    if not arm.startswith("hlsgraph-"):
        return
    prepared = next((
        item.get("identity") for item in environment.get("identity_checks", [])
        if isinstance(item, dict)
        and item.get("kind") == "verify-hlsgraph-wheel-installation"
        and item.get("arm") == arm
    ), None)
    if not isinstance(prepared, dict):
        raise RuntimeError(f"environment lock lacks wheel identity for {arm}")
    python_value = v02_python if arm == "hlsgraph-v02" else v03_python
    python_parts = resolve_command_argv(python_value)
    if len(python_parts) != 1:
        raise RuntimeError(f"{arm} runtime must be one direct Python executable")
    python = python_parts[0]
    python_key = "hlsgraph_v02" if arm == "hlsgraph-v02" else "hlsgraph_v03"
    expected_python = runtime.get("python", {}).get(python_key)
    if (not isinstance(expected_python, dict)
            or Path(python).as_posix() != expected_python.get("path")
            or not Path(python).is_file()
            or sha256_file(Path(python)) != expected_python.get("sha256")):
        raise RuntimeError(f"{arm} Python runtime differs from the prepared executable")
    command = [
        python, str((HERE / "wheel_identity.py").resolve()),
        "--expected-version", str(prepared.get("version", "")),
        "--expected-payload-sha256", str(prepared.get("installed_payload_sha256", "")),
    ]
    completed = subprocess.run(
        command, capture_output=True, text=True, check=False, timeout=30,
        env=official_process_environment(),
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{arm} runtime identity check failed: {completed.stderr.strip()}"
        )
    try:
        current = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{arm} runtime identity check returned invalid JSON") from exc
    if (not current.get("verified")
            or current.get("installed_payload_sha256")
            != prepared.get("installed_payload_sha256")):
        raise RuntimeError(f"{arm} runtime differs from the prepared wheel")


def _verify_all_arm_runtime_payloads(
    environment: dict[str, Any], *, v02_python: str, v03_python: str,
    codegraph_command: str, full_codegraph: bool = False,
) -> None:
    """Re-hash every treatment runtime before the first and after the last cell."""

    for arm in ("codegraph", "hlsgraph-v02", "hlsgraph-v03"):
        _runtime_identity_preflight(
            {"arm": arm}, environment, v02_python=v02_python,
            v03_python=v03_python, codegraph_command=codegraph_command,
            full_codegraph=full_codegraph,
        )


def execute_record(
    record: dict[str, Any], *, work_root: Path, runs_root: Path, codex_command: str,
    v02_python: str, v03_python: str, codegraph_command: str, timeout_seconds: int,
    run_set: dict[str, Any],
) -> dict[str, Any]:
    _preflight(record, work_root)
    environment_lock = work_root / "environment.lock.json"
    if not environment_lock.is_file():
        raise RuntimeError("missing environment.lock.json; run prepare --execute first")
    environment_identity = load_environment_lock(environment_lock)
    runs_root = _require_isolated_runs_root(
        runs_root, work_root=work_root, environment=environment_identity,
        allow_missing=False,
    )
    verify_evaluation_checkout(environment_identity)
    if not environment_identity.get("official_profile"):
        raise RuntimeError("the 192-cell suite requires the official libclang index profile")
    _runtime_identity_preflight(
        record, environment_identity, v02_python=v02_python,
        v03_python=v03_python, codegraph_command=codegraph_command,
    )
    unhashed_run_set = {
        key: value for key, value in run_set.items() if key != "run_set_sha256"
    }
    if (run_set.get("schema_version") != "hlsgraph.agent_eval.run_set.v1"
            or run_set.get("suite_asset_sha256") != asset_digest()
            or run_set.get("evaluation_harness_sha256") != harness_digest()
            or run_set.get("environment_lock_sha256") != sha256_file(environment_lock)
            or run_set.get("runs_root")
            != Path(os.path.abspath(os.fspath(runs_root))).as_posix()
            or run_set.get("timeout_seconds") != timeout_seconds
            or record.get("timeout_seconds") != timeout_seconds
            or re.fullmatch(r"[0-9a-f]{32}", str(run_set.get("batch_id", ""))) is None
            or run_set.get("run_set_sha256")
            != sha256_bytes(canonical_json(unhashed_run_set))):
        raise RuntimeError("run set has a stale or invalid identity")
    boundary = environment_identity["runtime_identity"]["sandbox_boundary"]
    _validate_permission_canary(
        run_set.get("permission_canary"), boundary, runs_root=runs_root,
    )
    _validate_treatment_mcp_canary(
        run_set.get("mcp_containment_canary"), boundary,
    )
    expected = next(
        (item for item in run_set["cells"] if item["run_id"] == record["run_id"]),
        None,
    )
    if not isinstance(expected, dict) or any(
        expected.get(key) != value for key, value in record.items()
    ):
        raise RuntimeError("run record differs from the frozen run set")
    boundary_canary = _verify_boundary_canary(
        work_root, expected.get("boundary_canary"),
        batch_id=run_set["batch_id"], run_id=record["run_id"],
    )
    workspace_path = _preflight(record, work_root)
    prepared_workspace = verify_prepared_workspace(
        environment_identity, work_root, record["arm"], record["corpus_id"],
    )
    if expected.get("workspace_identity_sha256") != prepared_workspace.get(
        "workspace_identity_sha256"
    ):
        raise RuntimeError("run-set workspace differs from the prepared environment")
    audit_overlay = _audit_overlay_from_descriptor(
        work_root, workspace_path, expected.get("retrieval_audit"),
        batch_id=run_set["batch_id"], run_id=record["run_id"],
        require_empty=True,
    )
    question = _question_map()[record["question_id"]]
    prompt = build_prompt(
        question, record["arm"], trace_challenge=expected.get("trace_challenge"),
    )
    command = build_codex_command(
        record, work_root=work_root, runs_root=runs_root,
        codex_command=codex_command,
        v02_python=v02_python, v03_python=v03_python,
        codegraph_command=codegraph_command, sandbox_boundary=boundary,
        audit_overlay=audit_overlay,
    )
    if (command != expected.get("command_argv")
            or hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            != expected.get("prompt_sha256")):
        raise RuntimeError("runtime command or prompt differs from the frozen run set")
    run_dir = runs_root / record["run_id"]
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    stdout_path = run_dir / "codex.jsonl"
    stderr_path = run_dir / "codex.stderr.log"
    started = time.perf_counter()
    timed_out = False
    with stdout_path.open("w", encoding="utf-8", newline="\n") as stdout, stderr_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as stderr:
        process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=stdout, stderr=stderr,
            text=True, encoding="utf-8", cwd=str(Path.cwd()),
            env=official_process_environment(),
            creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0),
            start_new_session=(os.name != "nt"),
        )
        try:
            process.communicate(prompt, timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate_process_tree(process)
    elapsed = time.perf_counter() - started
    if audit_overlay is None:
        audit_data = b""
        audit_receipt = {
            "schema_version": RETRIEVAL_AUDIT_SCHEMA_VERSION,
            "status": "not_applicable",
            "sha256": sha256_bytes(audit_data),
            "record_count": 0,
            "returned_count": 0,
            "returned_bytes": 0,
        }
        audit_receipt["receipt_sha256"] = sha256_bytes(canonical_json(audit_receipt))
    else:
        audit_data = _verify_audit_overlay_file(
            audit_overlay[0], require_empty=False,
            parent_chain=audit_overlay[2],
        )
        audit_receipt = _retrieval_audit_receipt(audit_data)
    (run_dir / "retrieval-access.jsonl").write_bytes(audit_data)
    _verify_all_arm_runtime_payloads(
        environment_identity, v02_python=v02_python,
        v03_python=v03_python, codegraph_command=codegraph_command,
    )
    if _verify_boundary_canary(
        work_root, expected.get("boundary_canary"),
        batch_id=run_set["batch_id"], run_id=record["run_id"],
    ) != boundary_canary:
        raise RuntimeError("boundary canary changed during the model cell")
    _validate_workspace_inventory(
        root=work_root.resolve(), workspace=_preflight(record, work_root),
        sandbox_boundary=boundary,
    )
    verify_prepared_workspace(
        environment_identity, work_root, record["arm"], record["corpus_id"],
    )
    verify_evaluation_checkout(environment_identity)
    metadata = {
        "schema_version": "hlsgraph.agent_eval.run.v1",
        **record,
        "suite_asset_sha256": asset_digest(),
        "evaluation_harness_sha256": harness_digest(),
        "environment_lock_sha256": sha256_file(environment_lock),
        "batch_id": run_set["batch_id"],
        "run_set_sha256": run_set["run_set_sha256"],
        "run_contract_sha256": expected["run_contract_sha256"],
        "trace_challenge": expected["trace_challenge"],
        "boundary_canary": expected["boundary_canary"],
        "permission_canary_sha256": run_set["permission_canary"]["canary_sha256"],
        "mcp_containment_canary_sha256": run_set["mcp_containment_canary"][
            "canary_sha256"
        ],
        "workspace_identity_sha256": prepared_workspace["workspace_identity_sha256"],
        "retrieval_audit": audit_receipt,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "workspace": f"$WORK_ROOT/{record['arm']}/{record['corpus_id']}",
        "command_argv": command,
        "wall_time_seconds": elapsed,
        "timeout_seconds": timeout_seconds,
        "timed_out": timed_out,
        "returncode": process.returncode,
    }
    _write_json(run_dir / "run.json", metadata)
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, default=HERE / "work")
    parser.add_argument("--runs-root", type=Path, default=HERE / "runs")
    parser.add_argument("--codex-command", default="codex")
    parser.add_argument("--v02-python", default="python")
    parser.add_argument("--v03-python", default="python")
    parser.add_argument("--codegraph-command", default="codegraph")
    parser.add_argument("--arm", action="append", choices=ARM_IDS)
    parser.add_argument("--question", action="append")
    frozen_timeout = load_manifest()["codex_cli"]["timeout_seconds"]
    parser.add_argument(
        "--timeout-seconds", type=int, default=frozen_timeout,
        choices=(frozen_timeout,),
    )
    parser.add_argument("--execute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    v02_parts = resolve_command_argv(args.v02_python)
    v03_parts = resolve_command_argv(args.v03_python)
    if len(v02_parts) != 1 or len(v03_parts) != 1:
        raise SystemExit("each HLSGraph Python must be one direct executable")
    args.v02_python = v02_parts[0]
    args.v03_python = v03_parts[0]
    plan = build_run_plan(arms=args.arm or ARM_IDS, question_ids=args.question)
    if not args.execute:
        print(json.dumps({
            "executed": False,
            "cells": len(plan),
            "suite_asset_sha256": asset_digest(),
            "plan": plan,
        }, indent=2, sort_keys=True))
        return 0
    # This guard and the complete runtime comparison deliberately precede the
    # first mkdir, canary, run-set, or model-side mutation.
    require_official_linux_wsl2()
    environment_lock = args.work_root / "environment.lock.json"
    _require_isolated_work_root(args.work_root)
    environment = load_environment_lock(environment_lock)
    verify_official_runtime_identity(
        environment, public_repository=PUBLIC_REPOSITORY,
        work_root=args.work_root, codex_command=args.codex_command,
        codegraph_command=args.codegraph_command,
        v02_python=args.v02_python, v03_python=args.v03_python,
    )
    _verify_all_arm_runtime_payloads(
        environment, v02_python=args.v02_python, v03_python=args.v03_python,
        codegraph_command=args.codegraph_command, full_codegraph=True,
    )
    verify_evaluation_checkout(environment)
    args.runs_root = _require_isolated_runs_root(
        args.runs_root, work_root=args.work_root, environment=environment,
        allow_missing=True,
    )
    args.runs_root.mkdir(parents=True, exist_ok=True)
    args.runs_root = _require_isolated_runs_root(
        args.runs_root, work_root=args.work_root, environment=environment,
        allow_missing=False,
    )
    if any(args.runs_root.iterdir()):
        raise RuntimeError("runs root must be empty before an official batch")
    permission_canary = run_permission_canaries(
        codex_command=args.codex_command, work_root=args.work_root,
        runs_root=args.runs_root, environment=environment,
    )
    mcp_containment_canary = run_treatment_mcp_canaries(
        work_root=args.work_root, runs_root=args.runs_root,
        environment=environment, v02_python=args.v02_python,
        v03_python=args.v03_python, codegraph_command=args.codegraph_command,
    )
    run_set = build_run_set(
        plan, work_root=args.work_root, runs_root=args.runs_root,
        environment=environment,
        environment_lock_sha256=sha256_file(environment_lock),
        codex_command=args.codex_command, v02_python=args.v02_python,
        v03_python=args.v03_python, codegraph_command=args.codegraph_command,
        permission_canary=permission_canary,
        mcp_containment_canary=mcp_containment_canary,
    )
    _write_json(args.runs_root / "run-set.json", run_set)
    for record in plan:
        execute_record(
            record, work_root=args.work_root, runs_root=args.runs_root,
            codex_command=args.codex_command, v02_python=args.v02_python,
            v03_python=args.v03_python, codegraph_command=args.codegraph_command,
            timeout_seconds=args.timeout_seconds, run_set=run_set,
        )
    _verify_all_arm_runtime_payloads(
        environment, v02_python=args.v02_python, v03_python=args.v03_python,
        codegraph_command=args.codegraph_command, full_codegraph=True,
    )
    verify_official_runtime_identity(
        environment, public_repository=PUBLIC_REPOSITORY,
        work_root=args.work_root, codex_command=args.codex_command,
        codegraph_command=args.codegraph_command,
        v02_python=args.v02_python, v03_python=args.v03_python,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
