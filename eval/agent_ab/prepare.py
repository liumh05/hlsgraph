"""Build reproducible indexes for a materialized public A/B corpus.

The default is a dry run.  ``--execute`` is required because indexing invokes
third-party commands and writes only beneath the ignored corpus work directory.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import re
import time
from pathlib import Path
from typing import Any, Sequence

from .common import (
    ARM_IDS, CODEGRAPH_OFFLINE_ENV, ENVIRONMENT_SCHEMA_VERSION, HERE,
    asset_digest, capture_official_runtime_identity, harness_digest,
    cold_start_input_identity,
    load_corpus_lock, load_manifest, canonical_json, official_process_environment,
    require_official_linux_wsl2, resolve_command_argv,
    sha256_bytes, sha256_file, workspace_identity,
)


def _index_root_name(arm: str) -> str:
    if arm == "codegraph":
        return ".codegraph"
    if arm.startswith("hlsgraph-"):
        return ".hlsgraph"
    raise RuntimeError(f"unsupported cold-start index arm: {arm}")


def _redirects_resolution(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(callable(is_junction) and is_junction())


def _cold_start_absence_proof(
    step: dict[str, Any], *, checkpoint: str,
) -> dict[str, Any]:
    """Fail closed unless one planned index root is lexically absent."""
    arm = str(step["arm"])
    corpus_id = str(step["corpus_id"])
    workspace = Path(str(step["cwd"])).absolute()
    if len(workspace.parents) < 2:
        raise RuntimeError("cold-start workspace has no evaluation root")
    work_root = workspace.parents[1]
    expected_workspace = (work_root / arm / corpus_id).absolute()
    if workspace != expected_workspace:
        raise RuntimeError("cold-start index step has an inconsistent workspace")
    index_name = _index_root_name(arm)
    index_root = workspace / index_name
    if index_root.exists() or _redirects_resolution(index_root):
        raise RuntimeError(
            f"cold-start index already exists before {checkpoint}: {arm}/{corpus_id}/{index_name}"
        )
    input_identity = cold_start_input_identity(work_root, arm, corpus_id)
    if input_identity["index_relative_path"] != index_name:
        raise RuntimeError("cold-start input identity selected the wrong index root")
    proof: dict[str, Any] = {
        "schema_version": "hlsgraph.agent_eval.cold_start_absence.v1",
        "kind": "cold-start-index-absence",
        "checkpoint": checkpoint,
        "arm": arm,
        "corpus_id": corpus_id,
        "phase": "index",
        "status": "absent",
        "index_relative_path": index_name,
        "input_tree_sha256": input_identity["input_tree_sha256"],
    }
    proof["proof_sha256"] = sha256_bytes(canonical_json(proof))
    return proof


def build_plan(
    work_root: Path, *, v02_python: str, v03_python: str, codegraph_command: str,
    codegraph_repo: Path | None, v02_wheel: Path, v03_wheel: Path,
    codex_command: str = "codex", degraded: bool = False,
    invocation_root: Path | None = None, v02_repo: Path | None = None,
    v03_repo: Path | None = None,
) -> list[dict[str, Any]]:
    if v02_repo is None:
        raise ValueError("v0.2 preparation requires its frozen clean source repository")
    if v03_repo is None:
        raise ValueError("v0.3 candidate preparation requires its clean source repository")
    manifest = load_manifest()
    corpora = load_corpus_lock()["corpora"]
    expected_codegraph = manifest["arms"][1]["revision"]
    invocation_root = (invocation_root or Path.cwd()).resolve()
    v02_parts = resolve_command_argv(v02_python, invocation_root)
    v03_parts = resolve_command_argv(v03_python, invocation_root)
    if len(v02_parts) != 1 or len(v03_parts) != 1:
        raise ValueError("each HLSGraph Python must be one direct executable")
    v02_python = v02_parts[0]
    v03_python = v03_parts[0]
    codegraph_parts = resolve_command_argv(codegraph_command, invocation_root)
    if (len(codegraph_parts) >= 2
            and Path(codegraph_parts[0]).name.casefold() in {"node", "node.exe"}):
        script = Path(codegraph_parts[1])
        if not script.is_absolute():
            codegraph_parts[1] = str(Path(
                os.path.abspath(os.fspath(invocation_root / script))
            ))
    plan: list[dict[str, Any]] = [{
        "kind": "record-codex-version",
        "cwd": str(work_root.resolve()),
        "command": [*resolve_command_argv(codex_command, invocation_root), "--version"],
        "minimum_version": manifest["codex_cli"]["minimum_version"],
    }]
    if codegraph_repo is not None:
        plan.append({
            "kind": "verify-codegraph-revision",
            "cwd": str(codegraph_repo.resolve()),
            "command": ["git", "rev-parse", "HEAD"],
            "expected_stdout": expected_codegraph,
        })
    if (codegraph_parts
            and Path(codegraph_parts[0]).name.casefold() in {"node", "node.exe"}):
        plan.append({
            "kind": "record-node-version", "cwd": str(invocation_root),
            "command": [codegraph_parts[0], "--version"],
        })
    v02_manifest = next(item for item in manifest["arms"] if item["id"] == "hlsgraph-v02")
    source_repositories = [
        ("v02", v02_repo, v02_manifest["revision"]),
        ("v03", v03_repo, None),
    ]
    for label, repository, expected_revision in source_repositories:
        plan.extend([
            {
                "kind": f"verify-{label}-repo-clean", "cwd": str(repository.resolve()),
                "command": ["git", "status", "--porcelain=v1", "--untracked-files=all"],
                "expected_stdout": "",
            },
            {
                "kind": f"record-{label}-revision", "cwd": str(repository.resolve()),
                "command": ["git", "rev-parse", "HEAD"],
                **({"expected_stdout": expected_revision} if expected_revision is not None else {}),
            },
        ])
    for arm, python, expected, wheel, source_repo in (
        ("hlsgraph-v02", v02_python, "0.2.0", v02_wheel, v02_repo),
        ("hlsgraph-v03", v03_python, "0.3.0", v03_wheel, v03_repo),
    ):
        identity_command = [
            python, str((Path(__file__).with_name("wheel_identity.py")).resolve()),
            "--wheel", str(wheel.resolve()), "--expected-version", expected,
        ]
        identity_command.extend(["--source-repo", str(source_repo.resolve())])
        plan.append({
            "kind": "verify-hlsgraph-wheel-installation",
            "arm": arm,
            "cwd": str(invocation_root),
            "command": identity_command,
            "identity_json": True,
            "expected_version": expected,
            "expected_wheel_sha256": sha256_file(wheel) if wheel.is_file() else None,
        })
    for corpus in corpora:
        corpus_id = corpus["id"]
        plan.append({
            "kind": "codegraph-index",
            "arm": "codegraph",
            "corpus_id": corpus_id,
            "cwd": str((work_root / "codegraph" / corpus_id).resolve()),
            "command": [*codegraph_parts, "init"],
            "environment": dict(CODEGRAPH_OFFLINE_ENV),
            "expected_entrypoint_sha256": manifest["arms"][1]["entrypoint_sha256"],
        })
        for arm, python, expected in (
            ("hlsgraph-v02", v02_python, "0.2.0"),
            ("hlsgraph-v03", v03_python, "0.3.0"),
        ):
            cwd = (work_root / arm / corpus_id).resolve()
            plan.extend([
                {
                    "kind": "verify-hlsgraph-version",
                    "arm": arm,
                    "cwd": str(cwd),
                    "command": [python, "-c", "import hlsgraph; print(hlsgraph.__version__)"],
                    "expected_stdout": expected,
                },
                {
                    "kind": "hlsgraph-index",
                    "arm": arm,
                    "corpus_id": corpus_id,
                    "cwd": str(cwd),
                    "command": [
                        python, "-m", "hlsgraph.cli", "index", "--project", str(cwd),
                    ] + (["--degraded"] if degraded else []),
                    "environment": {"HLSGRAPH_MCP_TOOLS": "all" if arm == "hlsgraph-v02" else "explore"},
                },
            ])
            if arm == "hlsgraph-v03":
                plan.append({
                    "kind": "hlsgraph-knowledge-sync",
                    "arm": arm,
                    "corpus_id": corpus_id,
                    "cwd": str(cwd),
                    "command": [
                        python, "-m", "hlsgraph.cli", "knowledge", "sync",
                        "--project", str(cwd),
                    ],
                })
    for label, repository, expected_revision in source_repositories:
        plan.extend([
            {
                "kind": f"verify-{label}-repo-clean-after", "cwd": str(repository.resolve()),
                "command": ["git", "status", "--porcelain=v1", "--untracked-files=all"],
                "expected_stdout": "",
            },
            {
                "kind": f"record-{label}-revision-after", "cwd": str(repository.resolve()),
                "command": ["git", "rev-parse", "HEAD"],
                **({"expected_stdout": expected_revision} if expected_revision is not None else {}),
            },
        ])
    return plan


def _version_tuple(value: str) -> tuple[int, ...]:
    match = re.search(r"\d+(?:\.\d+)+", value)
    if not match:
        raise RuntimeError(f"cannot parse version from {value!r}")
    return tuple(int(part) for part in match.group(0).split("."))


def execute_plan(plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index_steps = [
        step for step in plan
        if step.get("kind") in {"codegraph-index", "hlsgraph-index"}
    ]
    # This complete sweep occurs before the first subprocess, so a stale index
    # in a later corpus cannot allow any earlier timing measurement to begin.
    initial_absence = {
        (str(step["arm"]), str(step["corpus_id"])):
        _cold_start_absence_proof(step, checkpoint="pre_execution")
        for step in index_steps
    }
    observations: list[dict[str, Any]] = list(initial_absence.values())
    for step in plan:
        immediate_absence: dict[str, Any] | None = None
        if step.get("kind") in {"codegraph-index", "hlsgraph-index"}:
            immediate_absence = _cold_start_absence_proof(
                step, checkpoint="pre_index",
            )
            observations.append(immediate_absence)
        environment = official_process_environment()
        environment.update(step.get("environment", {}))
        started = time.perf_counter()
        completed = subprocess.run(
            step["command"], cwd=step["cwd"], env=environment,
            capture_output=True, text=True, check=False,
        )
        elapsed = time.perf_counter() - started
        if completed.returncode != 0:
            raise RuntimeError(
                f"{step['kind']} failed ({completed.returncode}): {completed.stderr.strip()}"
            )
        expected = step.get("expected_stdout")
        if expected is not None and completed.stdout.strip() != expected:
            raise RuntimeError(
                f"{step['kind']} identity mismatch: expected {expected!r}, "
                f"got {completed.stdout.strip()!r}"
            )
        minimum = step.get("minimum_version")
        if minimum is not None and _version_tuple(completed.stdout) < _version_tuple(minimum):
            raise RuntimeError(
                f"{step['kind']} requires >= {minimum}, got {completed.stdout.strip()!r}"
            )
        if step.get("identity_json"):
            try:
                identity = json.loads(completed.stdout)
            except json.JSONDecodeError as exc:
                raise RuntimeError("wheel identity helper returned invalid JSON") from exc
            if (not isinstance(identity, dict) or not identity.get("verified")
                    or identity.get("version") != step.get("expected_version")
                    or identity.get("wheel_sha256") != step.get("expected_wheel_sha256")
                    or identity.get("installed_payload_sha256")
                    != identity.get("wheel_payload_sha256")):
                raise RuntimeError(f"{step['kind']} returned an inconsistent identity")
            source_hash = identity.get("source_package_sha256")
            package_hash = identity.get("wheel_package_sha256")
            source_revision = identity.get("source_revision")
            if (not isinstance(source_hash, str)
                    or not re.fullmatch(r"[0-9a-f]{64}", source_hash)
                    or package_hash != source_hash
                    or not isinstance(source_revision, str)
                    or not re.fullmatch(r"[0-9a-f]{40}", source_revision)):
                raise RuntimeError(
                    f"{step['arm']} wheel identity is not bound to a clean source revision"
                )
            observations.append({
                "kind": step["kind"], "arm": step["arm"], "identity": identity,
            })
        if step["kind"].startswith(("record-", "verify-")):
            if not step.get("identity_json"):
                observations.append({"kind": step["kind"], "stdout": completed.stdout.strip()})
        phase = {
            "codegraph-index": "index",
            "hlsgraph-index": "index",
            "hlsgraph-knowledge-sync": "knowledge_sync",
        }.get(step["kind"])
        if phase is not None:
            phase_observation = {
                "schema_version": "hlsgraph.agent_eval.cold_start_phase.v1",
                "kind": "cold-start-index-phase",
                "arm": step["arm"],
                "corpus_id": step["corpus_id"],
                "phase": phase,
                "status": "measured",
                "wall_time_seconds": round(elapsed, 9),
                "command_sha256": sha256_bytes(canonical_json(step["command"])),
            }
            if phase == "index":
                if immediate_absence is None:
                    raise RuntimeError("cold-start index lacks an immediate absence proof")
                initial = initial_absence[(str(step["arm"]), str(step["corpus_id"]))]
                phase_observation.update({
                    "input_tree_sha256": immediate_absence["input_tree_sha256"],
                    "pre_execution_absence_proof_sha256": initial["proof_sha256"],
                    "pre_index_absence_proof_sha256": immediate_absence["proof_sha256"],
                })
            observations.append(phase_observation)
    phase_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for item in observations:
        if item.get("kind") == "cold-start-index-phase":
            phase_groups.setdefault((item["arm"], item["corpus_id"]), []).append(item)
    expected_phases = {
        "codegraph": ["index"],
        "hlsgraph-v02": ["index"],
        "hlsgraph-v03": ["index", "knowledge_sync"],
    }
    for (arm, corpus_id), phases in phase_groups.items():
        phase_names = [item["phase"] for item in phases]
        if phase_names != expected_phases[arm]:
            raise RuntimeError(
                f"{arm}/{corpus_id} cold-start phases are incomplete: {phase_names!r}"
            )
        public_phases = [
            {
                "schema_version": item["schema_version"],
                "kind": item["kind"],
                "phase": item["phase"],
                "status": item["status"],
                "wall_time_seconds": item["wall_time_seconds"],
                "command_sha256": item["command_sha256"],
                **({
                    "input_tree_sha256": item["input_tree_sha256"],
                    "pre_execution_absence_proof_sha256":
                        item["pre_execution_absence_proof_sha256"],
                    "pre_index_absence_proof_sha256":
                        item["pre_index_absence_proof_sha256"],
                } if item["phase"] == "index" else {}),
            }
            for item in phases
        ]
        phase_commands = [
            {"phase": item["phase"], "command_sha256": item["command_sha256"]}
            for item in phases
        ]
        observations.append({
            "schema_version": "hlsgraph.agent_eval.cold_start_index.v1",
            "kind": "cold-start-index",
            "arm": arm,
            "corpus_id": corpus_id,
            "status": "measured",
            "phases": public_phases,
            "wall_time_seconds": round(sum(
                float(item["wall_time_seconds"]) for item in phases
            ), 9),
            "command_sha256": sha256_bytes(canonical_json(phase_commands)),
        })
    return observations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, default=HERE / "work")
    parser.add_argument("--v02-python", required=True)
    parser.add_argument("--v03-python", required=True)
    parser.add_argument("--codegraph-command", default="codegraph")
    parser.add_argument("--codegraph-repo", type=Path)
    parser.add_argument(
        "--runtime-root", type=Path,
        help="ext4 root containing Node, npm, CodeGraph, Codex, and both venvs",
    )
    parser.add_argument(
        "--npm-cli", type=Path,
        help="direct npm-cli.js path from the frozen Node distribution",
    )
    parser.add_argument(
        "--v02-repo", type=Path,
        help="clean checkout of the exact frozen v0.2 source revision",
    )
    parser.add_argument("--v03-repo", type=Path, default=HERE.parents[1])
    parser.add_argument("--codex-command", default="codex")
    parser.add_argument("--v02-wheel", type=Path, required=True)
    parser.add_argument("--v03-wheel", type=Path, required=True)
    parser.add_argument(
        "--degraded", action="store_true",
        help="non-official diagnostic mode; release evaluation uses libclang",
    )
    parser.add_argument("--execute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runtime_identity: dict[str, Any] | None = None
    if args.execute:
        # Official execution is Linux/WSL2-only and the guard runs before any
        # indexing, directory creation, or environment-lock write.
        require_official_linux_wsl2()
        if args.codegraph_repo is None:
            raise SystemExit("--codegraph-repo is required with --execute")
        if args.runtime_root is None:
            raise SystemExit("--runtime-root is required with --execute")
        if args.npm_cli is None:
            raise SystemExit("--npm-cli is required with --execute")
        if args.v02_repo is None:
            raise SystemExit("--v02-repo is required with --execute")
        for wheel in (args.v02_wheel, args.v03_wheel):
            if not wheel.is_file():
                raise SystemExit(f"wheel does not exist: {wheel}")
        runtime_identity = capture_official_runtime_identity(
            public_repository=HERE.parents[1], work_root=args.work_root,
            codex_command=args.codex_command,
            codegraph_command=args.codegraph_command,
            codegraph_repository=args.codegraph_repo,
            runtime_root=args.runtime_root, npm_cli=args.npm_cli,
            v02_python=args.v02_python, v03_python=args.v03_python,
        )
    plan = build_plan(
        args.work_root, v02_python=args.v02_python, v03_python=args.v03_python,
        codegraph_command=args.codegraph_command, codegraph_repo=args.codegraph_repo,
        v02_wheel=args.v02_wheel, v03_wheel=args.v03_wheel,
        codex_command=args.codex_command, degraded=args.degraded,
        v02_repo=args.v02_repo, v03_repo=args.v03_repo,
    )
    if args.execute:
        codegraph_step = next(item for item in plan if item["kind"] == "codegraph-index")
        codegraph_argv = codegraph_step["command"]
        if (len(codegraph_argv) != 2
                or Path(str(codegraph_argv[0])).name.casefold() not in {"node", "node.exe"}):
            raise SystemExit("official execution requires node plus the frozen CodeGraph JS entrypoint")
        codegraph_entrypoint = Path(codegraph_argv[1])
        if not codegraph_entrypoint.is_absolute() or not codegraph_entrypoint.is_file():
            raise SystemExit("CodeGraph JS entrypoint must be an existing absolute path")
        try:
            codegraph_entrypoint.resolve().relative_to(args.codegraph_repo.resolve())
        except ValueError as exc:
            raise SystemExit("CodeGraph JS entrypoint must be inside --codegraph-repo") from exc
        expected_codegraph_sha256 = load_manifest()["arms"][1]["entrypoint_sha256"]
        if sha256_file(codegraph_entrypoint) != expected_codegraph_sha256:
            raise SystemExit(
                "CodeGraph JS entrypoint bytes do not match the frozen manifest hash"
            )
        observations = execute_plan(plan)
        current_runtime_identity = capture_official_runtime_identity(
            public_repository=HERE.parents[1], work_root=args.work_root,
            codex_command=args.codex_command,
            codegraph_command=args.codegraph_command,
            codegraph_repository=args.codegraph_repo,
            runtime_root=args.runtime_root, npm_cli=args.npm_cli,
            v02_python=args.v02_python, v03_python=args.v03_python,
        )
        if current_runtime_identity != runtime_identity:
            raise RuntimeError("official runtime changed during preparation")
        source_identities: dict[str, tuple[str, dict[str, Any]]] = {}
        for arm, label in (("hlsgraph-v02", "v02"), ("hlsgraph-v03", "v03")):
            recorded_revision = next(
                item["stdout"] for item in observations
                if item["kind"] == f"record-{label}-revision"
            )
            final_revision = next(
                item["stdout"] for item in observations
                if item["kind"] == f"record-{label}-revision-after"
            )
            identity = next(
                item["identity"] for item in observations
                if item["kind"] == "verify-hlsgraph-wheel-installation"
                and item["arm"] == arm
            )
            if recorded_revision != identity["source_revision"] or final_revision != recorded_revision:
                raise RuntimeError(
                    f"{arm} source revision changed during preparation or differs from its wheel"
                )
            source_identities[arm] = (recorded_revision, identity)
        recorded_v02_revision, v02_identity = source_identities["hlsgraph-v02"]
        recorded_v03_revision, v03_identity = source_identities["hlsgraph-v03"]
        workspaces = {
            f"{arm}/{corpus['id']}": workspace_identity(
                args.work_root, arm, corpus["id"],
            )
            for arm in ARM_IDS for corpus in load_corpus_lock()["corpora"]
        }
        environment = {
            "schema_version": ENVIRONMENT_SCHEMA_VERSION,
            "suite_asset_sha256": asset_digest(),
            "evaluation_harness_sha256": harness_digest(),
            "codegraph_revision": load_manifest()["arms"][1]["revision"],
            "codegraph_entrypoint": {
                **runtime_identity["codegraph_entrypoint"],
            },
            "codegraph_build": {
                **runtime_identity["codegraph_build"],
            },
            "source_backend": "regex_degraded" if args.degraded else "libclang",
            "official_profile": not args.degraded,
            "runtime_identity": runtime_identity,
            "hlsgraph_v02": {
                "version": "0.2.0", "wheel": args.v02_wheel.name,
                "wheel_sha256": sha256_file(args.v02_wheel),
                "revision": recorded_v02_revision,
                "source_revision": v02_identity["source_revision"],
                "source_package_sha256": v02_identity["source_package_sha256"],
                "wheel_package_sha256": v02_identity["wheel_package_sha256"],
            },
            "hlsgraph_v03": {
                "version": "0.3.0", "wheel": args.v03_wheel.name,
                "wheel_sha256": sha256_file(args.v03_wheel),
                "revision": recorded_v03_revision,
                "source_revision": v03_identity["source_revision"],
                "source_package_sha256": v03_identity["source_package_sha256"],
                "wheel_package_sha256": v03_identity["wheel_package_sha256"],
            },
            "identity_checks": observations,
            "cold_start_indexing": [
                *[
                    {
                        "schema_version": "hlsgraph.agent_eval.cold_start_index.v1",
                        "kind": "cold-start-index", "arm": "native",
                        "corpus_id": corpus["id"], "status": "not_applicable",
                        "phases": [], "wall_time_seconds": None,
                        "reason": "no_index_required",
                    }
                    for corpus in load_corpus_lock()["corpora"]
                ],
                *[item for item in observations if item.get("kind") == "cold-start-index"],
            ],
            "workspaces": workspaces,
        }
        args.work_root.mkdir(parents=True, exist_ok=True)
        (args.work_root / "environment.lock.json").write_text(
            json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
    print(json.dumps({"executed": args.execute, "steps": plan}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
