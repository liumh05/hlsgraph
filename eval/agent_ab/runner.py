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
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

from .common import (
    ARM_IDS, CODEGRAPH_OFFLINE_ENV, HERE, asset_digest,
    capture_codegraph_build_identity, canonical_json, harness_digest,
    load_corpus_lock, load_environment_lock, load_manifest, load_questions,
    resolve_command_argv,
    official_process_environment, require_official_linux_wsl2,
    require_official_ext4_directory, safe_relative_path,
    sha256_bytes, sha256_file,
    verify_evaluation_checkout, verify_official_runtime_identity,
    verify_prepared_workspace,
)


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


def _work_root_directory_denies(
    *, root: Path, workspace: Path, sandbox_boundary: dict[str, Any],
) -> list[Path]:
    """Validate the frozen matrix inventory and return directory-only denies."""

    if sandbox_boundary.get("work_root_policy") != "explicit_sibling_directory_deny_v1":
        raise RuntimeError("official sandbox boundary has the wrong matrix policy")
    catalog = sandbox_boundary.get("work_root_deny_catalog")
    if not isinstance(catalog, dict):
        raise RuntimeError("official sandbox boundary lacks the deny catalog")
    arm_roots = catalog.get("arm_roots")
    workspace_roots = catalog.get("workspace_roots")
    control_roots = catalog.get("control_roots")
    if (not isinstance(arm_roots, dict) or set(arm_roots) != set(ARM_IDS)
            or not isinstance(workspace_roots, dict)
            or not isinstance(control_roots, list) or len(control_roots) != 2):
        raise RuntimeError("official sandbox deny catalog is malformed")
    expected_corpora = {item["id"] for item in load_corpus_lock()["corpora"]}
    expected_workspace_keys = {
        f"{arm}/{corpus_id}" for arm in ARM_IDS for corpus_id in expected_corpora
    }
    if set(workspace_roots) != expected_workspace_keys:
        raise RuntimeError("official sandbox deny catalog has an incomplete matrix")
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
        raise RuntimeError("official sandbox deny catalog points outside the work root")

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
    return [
        *[expected_arms[item] for item in ARM_IDS if item != arm],
        *[
            expected_workspaces[f"{arm}/{item}"]
            for item in sorted(expected_corpora) if item != corpus_id
        ],
        *sorted(expected_controls, key=lambda item: item.as_posix()),
    ]


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
    official_matrix = sandbox_boundary.get("work_root_policy") is not None
    matrix_denies = (
        _work_root_directory_denies(
            root=root, workspace=current, sandbox_boundary=sandbox_boundary,
        ) if official_matrix else []
    )
    deny_paths = sorted(
        {
            Path(str(path)).resolve() for path in deny_roots
            if Path(str(path)).resolve() != root
        } | {path.resolve() for path in matrix_denies},
        key=lambda path: (len(path.parts), path.as_posix()),
    )
    effective_denies: list[Path] = []
    for candidate in deny_paths:
        if any(
            candidate == parent or candidate.is_relative_to(parent)
            for parent in effective_denies
        ):
            continue
        effective_denies.append(candidate)
    # Avoid redundant nested bind mounts: once /mnt/d is denied, a second
    # mount for /mnt/d/<public-repo> can fail while constructing the sandbox.
    # The broader rule already enforces the narrower declared boundary.
    rules = {path.as_posix(): "deny" for path in effective_denies}
    # Results are written by the harness outside the model sandbox.  Keep this
    # exact rule even when another deny would appear sufficient: the frozen
    # per-cell argv and the permission canary both bind this specific root.
    rules[result_root.as_posix()] = "deny"
    # The base profile is read-only; explicit directory rules remove every
    # sibling/control subtree while retaining the current workspace.
    rules[current.as_posix()] = "read"
    filesystem = "{" + ",".join(
        f"{_toml(path)}={_toml(mode)}" for path, mode in rules.items()
    ) + "}"
    values = [
        f"default_permissions={_toml(PERMISSION_PROFILE)}",
        f"permissions.{PERMISSION_PROFILE}.extends={_toml(':read-only')}",
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


def _mcp_overrides(
    arm: str, *, workspace: Path, v02_python: str, v03_python: str,
    codegraph_command: str,
) -> list[str]:
    if arm == "native":
        return []
    if arm == "codegraph":
        parts = _codegraph_parts(codegraph_command)
        command, args = parts[0], [*parts[1:], "serve", "--mcp"]
        values = {
            "mcp_servers.codegraph.command": command,
            "mcp_servers.codegraph.args": args,
            **{
                f"mcp_servers.codegraph.env.{key}": value
                for key, value in CODEGRAPH_ENV.items()
            },
        }
    else:
        python = v02_python if arm == "hlsgraph-v02" else v03_python
        mode = "all" if arm == "hlsgraph-v02" else "explore"
        values = {
            "mcp_servers.hlsgraph.command": python,
            "mcp_servers.hlsgraph.args": [
                "-m", "hlsgraph.mcp.server", str(workspace.resolve()),
            ],
            "mcp_servers.hlsgraph.env.HLSGRAPH_MCP_TOOLS": mode,
        }
    output: list[str] = []
    for key, value in values.items():
        output.extend(["-c", f"{key}={_toml(value)}"])
    return output


def build_codex_command(
    record: dict[str, Any], *, work_root: Path, runs_root: Path,
    codex_command: str,
    v02_python: str, v03_python: str, codegraph_command: str,
    sandbox_boundary: dict[str, Any] | None = None,
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


def _run_canary_command(command: list[str], *, timeout_seconds: int = 15) -> subprocess.CompletedProcess[str]:
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


def run_permission_canaries(
    *, codex_command: str, work_root: Path, runs_root: Path,
    environment: dict[str, Any],
) -> dict[str, Any]:
    """Exercise every frozen filesystem boundary and socket deny before calls.

    A platform that accepts the profile syntax but cannot enforce deny-read is
    not an official evaluation platform.  This deliberately makes the suite a
    NO-GO instead of silently falling back to prompt-only isolation.
    """

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
    workspace = root / "native" / "dataflow_gemm"
    allowed = workspace / "EVAL_PROVENANCE.json"
    same_arm_corpus = next(
        item["id"] for item in load_corpus_lock()["corpora"]
        if item["id"] != "dataflow_gemm"
    )
    same_arm_sibling = root / "native" / same_arm_corpus / "EVAL_PROVENANCE.json"
    other_arm_sibling = (
        root / "codegraph" / "dataflow_gemm" / "EVAL_PROVENANCE.json"
    )
    public_gold = HERE / "questions.jsonl"
    if (not allowed.is_file() or not same_arm_sibling.is_file()
            or not other_arm_sibling.is_file() or not public_gold.is_file()):
        raise RuntimeError("permission canary inputs are missing")
    prefix = _sandbox_canary_prefix(
        codex_command, workspace, work_root=root, runs_root=result_root,
        sandbox_boundary=boundary,
    )
    python = str(Path(sys.executable).resolve())
    read_script = "import pathlib,sys;pathlib.Path(sys.argv[1]).read_bytes();sys.exit(0)"

    def require_denied(path: Path, label: str) -> None:
        result = _run_canary_command([
            *prefix, python, "-c", read_script, str(path),
        ])
        if result.returncode == 0:
            raise RuntimeError(
                f"Codex permission profile allowed {label}; official suite is NO-GO"
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
    codex_sentinel = codex_home / f".hlsgraph-eval-canary-{token}"
    home_sentinel = home_root / f".hlsgraph-eval-home-canary-{token}"
    boundary_sentinel = boundary_root / f"permission-canary-{token}.bin"
    runs_sentinel = result_root / f".hlsgraph-eval-runs-canary-{token}.bin"
    external_sentinel = external_root / "private-like-sentinel.bin"
    drvfs_directories: list[Path] = []
    drvfs_sentinels: list[Path] = []
    try:
        codex_sentinel.write_bytes(secrets.token_bytes(32))
        home_sentinel.write_bytes(secrets.token_bytes(32))
        boundary_root.mkdir(parents=False, exist_ok=True)
        boundary_sentinel.write_bytes(secrets.token_bytes(32))
        runs_sentinel.write_bytes(secrets.token_bytes(32))
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

        allowed_result = _run_canary_command([
            *prefix, python, "-c", read_script, str(allowed),
        ])
        if allowed_result.returncode != 0:
            raise RuntimeError(
                "Codex permission profile cannot read the isolated corpus workspace: "
                + (allowed_result.stderr.strip() or allowed_result.stdout.strip())
            )
        require_denied(same_arm_sibling, "a same-arm sibling workspace read")
        require_denied(other_arm_sibling, "an other-arm sibling workspace read")
        require_denied(boundary_sentinel, "the boundary-control directory")
        require_denied(runs_sentinel, "the runs root")
        require_denied(public_gold, "a public gold-file read")
        require_denied(codex_sentinel, "a dedicated CODEX_HOME read")
        require_denied(home_sentinel, "a user-home read")
        require_denied(external_sentinel, "an external private-like read")
        for sentinel in drvfs_sentinels:
            require_denied(sentinel, f"a drvfs read at {sentinel.parent.parent}")
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
            runs_sentinel.unlink(missing_ok=True)
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
        network_result = _run_canary_command([
            *prefix, python, "-c", network_script, str(port),
        ])
    finally:
        listener.close()
    if network_result.returncode == 0:
        raise RuntimeError(
            "Codex permission profile allowed a TCP connection; official suite is NO-GO"
        )
    value = {
        "schema_version": "hlsgraph.agent_eval.permission_canary.v3",
        "profile": PERMISSION_PROFILE,
        "workspace_read": "pass",
        "sibling_workspace_read": "denied",
        "same_arm_sibling_read": "denied",
        "other_arm_sibling_read": "denied",
        "boundary_control_read": "denied",
        "runs_root_read": "denied",
        "runs_root_sha256": hashlib.sha256(
            result_root.as_posix().encode("utf-8")
        ).hexdigest(),
        "public_gold_read": "denied",
        "codex_home_read": "denied",
        "user_home_read": "denied",
        "external_private_read": "denied",
        "drvfs_mount_reads": "denied",
        "drvfs_mount_count": len(boundary["drvfs_roots"]),
        "drvfs_roots_sha256": sha256_bytes(canonical_json(boundary["drvfs_roots"])),
        "network_socket": "denied",
        "sandbox_boundary_sha256": boundary["identity_sha256"],
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
        "schema_version": "hlsgraph.agent_eval.permission_canary.v3",
        "profile": PERMISSION_PROFILE,
        "workspace_read": "pass",
        "sibling_workspace_read": "denied",
        "same_arm_sibling_read": "denied",
        "other_arm_sibling_read": "denied",
        "boundary_control_read": "denied",
        "runs_root_read": "denied",
        "runs_root_sha256": hashlib.sha256(
            Path(os.path.abspath(os.fspath(runs_root))).as_posix().encode("utf-8")
        ).hexdigest(),
        "public_gold_read": "denied",
        "codex_home_read": "denied",
        "user_home_read": "denied",
        "external_private_read": "denied",
        "drvfs_mount_reads": "denied",
        "drvfs_mount_count": len(sandbox_boundary["drvfs_roots"]),
        "drvfs_roots_sha256": sha256_bytes(canonical_json(
            sandbox_boundary["drvfs_roots"]
        )),
        "network_socket": "denied",
        "sandbox_boundary_sha256": sandbox_boundary["identity_sha256"],
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


def _clear_mutable_access_log(workspace: Path) -> None:
    """Remove the body-free retrieval audit log before each isolated cell.

    That log is the sole intentionally mutable workspace byte.  Clearing it
    prevents an earlier cell (or a pre-populated file) from becoming readable
    context for a later arm while preserving the product's normal append-only
    audit behavior during the cell itself.
    """
    ledger = workspace / ".hlsgraph"
    private = ledger / "private"
    path = private / "retrieval-access.jsonl"
    is_link = lambda item: item.is_symlink() or bool(
        callable(getattr(item, "is_junction", None)) and item.is_junction()
    )
    if any(item.exists() and is_link(item) for item in (workspace, ledger, private, path)):
        raise RuntimeError("evaluation access-log path contains a link or junction")
    if path.exists():
        if not path.is_file():
            raise RuntimeError("evaluation access log is not a regular file")
        path.unlink()


def build_run_set(
    plan: list[dict[str, Any]], *, work_root: Path, runs_root: Path,
    environment: dict[str, Any],
    environment_lock_sha256: str, codex_command: str, v02_python: str,
    v03_python: str, codegraph_command: str,
    permission_canary: dict[str, Any] | None = None,
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
    cells: list[dict[str, Any]] = []
    questions = _question_map()
    batch_id = secrets.token_hex(16)
    for record in sorted(plan, key=lambda item: item["run_id"]):
        workspace = verify_prepared_workspace(
            environment, work_root, record["arm"], record["corpus_id"],
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
    prepared_workspace = verify_prepared_workspace(
        environment_identity, work_root, record["arm"], record["corpus_id"],
    )
    if expected.get("workspace_identity_sha256") != prepared_workspace.get(
        "workspace_identity_sha256"
    ):
        raise RuntimeError("run-set workspace differs from the prepared environment")
    _clear_mutable_access_log(_preflight(record, work_root))
    verify_prepared_workspace(
        environment_identity, work_root, record["arm"], record["corpus_id"],
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
    _verify_all_arm_runtime_payloads(
        environment_identity, v02_python=v02_python,
        v03_python=v03_python, codegraph_command=codegraph_command,
    )
    if _verify_boundary_canary(
        work_root, expected.get("boundary_canary"),
        batch_id=run_set["batch_id"], run_id=record["run_id"],
    ) != boundary_canary:
        raise RuntimeError("boundary canary changed during the model cell")
    _work_root_directory_denies(
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
        "workspace_identity_sha256": prepared_workspace["workspace_identity_sha256"],
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
    run_set = build_run_set(
        plan, work_root=args.work_root, runs_root=args.runs_root,
        environment=environment,
        environment_lock_sha256=sha256_file(environment_lock),
        codex_command=args.codex_command, v02_python=args.v02_python,
        v03_python=args.v03_python, codegraph_command=args.codegraph_command,
        permission_canary=permission_canary,
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
