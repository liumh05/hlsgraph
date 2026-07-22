from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from eval.agent_ab.common import (
    ARM_IDS, MCP_CONTAINMENT_SCHEMA_VERSION, MCP_SYSTEM_PROFILE,
    RUNTIME_TREE_ALGORITHM, SANDBOX_ALLOW_TREE_ALGORITHM,
    SANDBOX_BOUNDARY_SCHEMA_VERSION, SANDBOX_FILESYSTEM_POLICY,
    SANDBOX_MINIMAL_TOKEN, canonical_json, load_corpus_lock, load_manifest,
    sandbox_allow_tree_identity, sha256_bytes, sha256_file,
)


def synthetic_cold_start_input_sha256(arm: str, corpus_id: str) -> str:
    return sha256_bytes(canonical_json({
        "arm": arm, "corpus_id": corpus_id, "kind": "synthetic-cold-input",
    }))


def synthetic_retrieval_audit_placeholder(workspace: Path) -> Path:
    target = workspace / ".hlsgraph" / "private" / "retrieval-access.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"")
    if os.name != "nt":
        target.parent.chmod(0o700)
        target.chmod(0o600)
    return target


def _synthetic_absence_proof(
    arm: str, corpus_id: str, checkpoint: str,
) -> dict[str, Any]:
    proof: dict[str, Any] = {
        "schema_version": "hlsgraph.agent_eval.cold_start_absence.v1",
        "kind": "cold-start-index-absence", "checkpoint": checkpoint,
        "arm": arm, "corpus_id": corpus_id, "phase": "index",
        "status": "absent",
        "index_relative_path": ".codegraph" if arm == "codegraph" else ".hlsgraph",
        "input_tree_sha256": synthetic_cold_start_input_sha256(arm, corpus_id),
    }
    proof["proof_sha256"] = sha256_bytes(canonical_json(proof))
    return proof


def synthetic_cold_start_matrix() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    phase_names = {
        "codegraph": ["index"],
        "hlsgraph-v02": ["index"],
        "hlsgraph-v03": ["index", "knowledge_sync"],
    }
    for arm in ARM_IDS:
        for corpus in load_corpus_lock()["corpora"]:
            record: dict[str, Any] = {
                "schema_version": "hlsgraph.agent_eval.cold_start_index.v1",
                "kind": "cold-start-index", "arm": arm,
                "corpus_id": corpus["id"],
            }
            if arm == "native":
                record.update({
                    "status": "not_applicable", "phases": [],
                    "wall_time_seconds": None, "reason": "no_index_required",
                })
            else:
                phases = []
                commands = []
                absence = {
                    checkpoint: _synthetic_absence_proof(
                        arm, corpus["id"], checkpoint,
                    )
                    for checkpoint in ("pre_execution", "pre_index")
                }
                checks.extend(absence.values())
                for phase_name in phase_names[arm]:
                    command_sha256 = sha256_bytes(canonical_json({
                        "arm": arm, "corpus_id": corpus["id"], "phase": phase_name,
                    }))
                    phase = {
                        "schema_version": "hlsgraph.agent_eval.cold_start_phase.v1",
                        "kind": "cold-start-index-phase", "phase": phase_name,
                        "status": "measured", "wall_time_seconds": 0.125,
                        "command_sha256": command_sha256,
                    }
                    if phase_name == "index":
                        phase.update({
                            "input_tree_sha256": synthetic_cold_start_input_sha256(
                                arm, corpus["id"],
                            ),
                            "pre_execution_absence_proof_sha256":
                                absence["pre_execution"]["proof_sha256"],
                            "pre_index_absence_proof_sha256":
                                absence["pre_index"]["proof_sha256"],
                        })
                    phases.append(phase)
                    commands.append({
                        "phase": phase_name, "command_sha256": command_sha256,
                    })
                    checks.append({
                        **phase, "arm": arm, "corpus_id": corpus["id"],
                    })
                record.update({
                    "status": "measured", "phases": phases,
                    "wall_time_seconds": round(0.125 * len(phases), 9),
                    "command_sha256": sha256_bytes(canonical_json(commands)),
                })
                checks.append(record)
            records.append(record)
    return records, checks


def synthetic_runtime_identity(
    *, public_repository: Path, work_root: Path,
) -> dict[str, Any]:
    """Build a structurally valid, non-executable runtime lock for unit tests."""

    public_repository = public_repository.resolve()
    work_root = work_root.resolve()
    parent = work_root.parent
    codex_home = parent / "codex-home"
    external = parent / ".hlsgraph-eval-private-canary"
    drvfs = parent / "synthetic-drvfs-c"
    home = parent / "synthetic-home"
    runtime_root = parent / "synthetic-runtime"
    denies = sorted({
        public_repository.as_posix(), work_root.as_posix(), codex_home.as_posix(),
        runtime_root.as_posix(), external.as_posix(), drvfs.as_posix(), home.as_posix(),
    })
    corpus_ids = [item["id"] for item in load_corpus_lock()["corpora"]]
    workspace_catalog: dict[str, Any] = {
        "arm_roots": {arm: (work_root / arm).as_posix() for arm in ARM_IDS},
        "workspace_roots": {
            f"{arm}/{corpus_id}": (work_root / arm / corpus_id).as_posix()
            for arm in ARM_IDS for corpus_id in corpus_ids
        },
        "control_roots": [
            (work_root / ".hlsgraph-eval-boundary").as_posix(),
            (work_root / "_cache").as_posix(),
        ],
    }
    runtime_root.mkdir(parents=True, exist_ok=True)
    for name in (
        "fixture", "codex", "codex-linux-sandbox", "bwrap", "node",
        "npm-cli.js", "harness-python",
    ):
        path = runtime_root / name
        path.write_bytes(b"synthetic non-executable unit-test fixture\n")
        if os.name == "posix":
            path.chmod(0o700)
    for name in ("v02", "v03"):
        environment_root = runtime_root / name
        (environment_root / "bin").mkdir(parents=True)
        (environment_root / "lib" / "python3.10" / "site-packages").mkdir(
            parents=True,
        )
        (environment_root / "pyvenv.cfg").write_text(
            "home = /usr/bin\nversion = 3.10.12\n", encoding="utf-8",
        )
        python_path = environment_root / "bin" / "python"
        python_path.write_bytes(b"synthetic python launcher\n")
        if os.name == "posix":
            python_path.chmod(0o700)
    codegraph_repo = runtime_root / "codegraph"
    entrypoint = codegraph_repo / "dist" / "bin" / "codegraph.js"
    package_lock = codegraph_repo / "package-lock.json"
    package_json = codegraph_repo / "package.json"
    dependency = codegraph_repo / "node_modules" / "fixture" / "index.js"
    for path, data in (
        (entrypoint, b"synthetic codegraph entrypoint\n"),
        (package_lock, b"synthetic package lock\n"),
        (package_json, b'{"type":"module"}\n'),
        (dependency, b"synthetic dependency\n"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        if os.name == "posix":
            path.chmod(0o644)
    binary = {"filename": "fixture", "path": (runtime_root / "fixture").as_posix(),
              "version": "fixture-1", "sha256": "1" * 64}
    python = {
        "filename": "python", "path": (runtime_root / "python").as_posix(),
        "version": "3.10.12", "implementation": "CPython",
        "cache_tag": "cpython-310", "platform": "Linux-fixture",
        "sha256": "2" * 64,
    }
    runtime: dict[str, Any] = {
        "schema_version": "hlsgraph.agent_eval.runtime_identity.v2",
        "host": {
            "os_name": "posix", "system": "Linux",
            "release": "6.18.0-microsoft-standard-WSL2", "machine": "x86_64",
            "distribution_id": "ubuntu", "distribution_version": "22.04",
            "wsl": "2", "wsl_distribution": "Ubuntu-22.04",
        },
        "codex": {**binary, "path": (runtime_root / "codex").as_posix(),
                  "filename": "codex", "version": "codex-cli 0.144.0", "sha256": "3" * 64},
        "codex_linux_sandbox": {
            **binary, "path": (runtime_root / "codex-linux-sandbox").as_posix(),
            "filename": "codex-linux-sandbox", "sha256": "4" * 64,
        },
        "bubblewrap": {**binary, "path": (runtime_root / "bwrap").as_posix(),
                       "filename": "bwrap", "version": "bubblewrap 0.6.1", "sha256": "5" * 64},
        "node": {**binary, "path": (runtime_root / "node").as_posix(),
                 "filename": "node", "version": "v22.17.0",
                 "sha256": load_manifest()["arms"][1]["build_identity"]["node"]["sha256"]},
        "codegraph_entrypoint": {
            "path": entrypoint.as_posix(),
            "filename": "codegraph.js",
            "sha256": load_manifest()["arms"][1]["entrypoint_sha256"],
        },
        "python": {
            "harness": {**python, "path": (runtime_root / "harness-python").as_posix(),
                        "filename": "harness-python", "sha256": "7" * 64},
            "hlsgraph_v02": {**python, "path": (runtime_root / "v02" / "bin" / "python").as_posix(),
                             "filename": "python", "sha256": "8" * 64},
            "hlsgraph_v03": {**python, "path": (runtime_root / "v03" / "bin" / "python").as_posix(),
                             "filename": "python", "sha256": "9" * 64},
        },
    }
    frozen_build = load_manifest()["arms"][1]["build_identity"]
    runtime["codegraph_build"] = {
        "schema_version": "hlsgraph.agent_eval.codegraph_build.v1",
        "runtime_tree_algorithm": frozen_build["runtime_tree_algorithm"],
        "repository": {
            "path": codegraph_repo.as_posix(),
            "revision": load_manifest()["arms"][1]["revision"],
            "tree": frozen_build["repository_tree"],
        },
        "package_lock": {
            "path": package_lock.as_posix(), "filename": package_lock.name,
            "sha256": frozen_build["package_lock_sha256"],
        },
        "node": dict(runtime["node"]),
        "npm": {
            "path": (runtime_root / "npm-cli.js").as_posix(),
            "filename": "npm-cli.js", "version": frozen_build["npm"]["version"],
            "sha256": frozen_build["npm"]["cli_sha256"],
        },
        "entrypoint": dict(runtime["codegraph_entrypoint"]),
        "dist": {
            "path": (codegraph_repo / "dist").as_posix(),
            "algorithm": frozen_build["runtime_tree_algorithm"],
            "tree_sha256": frozen_build["dist_tree_sha256"],
        },
        "dependencies": {
            "path": (codegraph_repo / "node_modules").as_posix(),
            "algorithm": frozen_build["runtime_tree_algorithm"],
            "tree_sha256": frozen_build["dependency_tree_sha256"],
        },
        "reproduction_contract": frozen_build["reproduction_contract"],
    }
    runtime["codegraph_build"]["identity_sha256"] = sha256_bytes(canonical_json(
        runtime["codegraph_build"]
    ))
    codex_allow = {
        "path": runtime["codex"]["path"], "kind": "file",
        "algorithm": "sha256", "sha256": runtime["codex"]["sha256"],
    }
    runtime_allow_roots = {
        "native": [codex_allow],
        "codegraph": [
            codex_allow,
            {
                "path": runtime["node"]["path"], "kind": "file",
                "algorithm": "sha256", "sha256": runtime["node"]["sha256"],
            },
            {
                "path": package_json.as_posix(), "kind": "file",
                "algorithm": "sha256", "sha256": sha256_file(package_json),
            },
            {
                "path": runtime["codegraph_build"]["dist"]["path"],
                "kind": "tree", "algorithm": RUNTIME_TREE_ALGORITHM,
                "sha256": runtime["codegraph_build"]["dist"]["tree_sha256"],
            },
            {
                "path": runtime["codegraph_build"]["dependencies"]["path"],
                "kind": "tree", "algorithm": RUNTIME_TREE_ALGORITHM,
                "sha256": runtime["codegraph_build"]["dependencies"]["tree_sha256"],
            },
        ],
        "hlsgraph-v02": [codex_allow],
        "hlsgraph-v03": [codex_allow],
    }
    for arm, name in (("hlsgraph-v02", "v02"), ("hlsgraph-v03", "v03")):
        environment_root = runtime_root / name
        bin_root = environment_root / "bin"
        purelib = environment_root / "lib" / "python3.10" / "site-packages"
        config = environment_root / "pyvenv.cfg"
        runtime_allow_roots[arm].extend([
            {
                "path": bin_root.as_posix(), "kind": "tree",
                "algorithm": SANDBOX_ALLOW_TREE_ALGORITHM,
                "sha256": sandbox_allow_tree_identity(bin_root),
            },
            {
                "path": config.as_posix(), "kind": "file",
                "algorithm": "sha256", "sha256": sha256_file(config),
            },
            {
                "path": purelib.as_posix(), "kind": "tree",
                "algorithm": SANDBOX_ALLOW_TREE_ALGORITHM,
                "sha256": sandbox_allow_tree_identity(purelib),
            },
        ])
    runtime_allow_roots = {
        arm: sorted(entries, key=lambda item: item["path"])
        for arm, entries in runtime_allow_roots.items()
    }
    boundary: dict[str, Any] = {
        "schema_version": SANDBOX_BOUNDARY_SCHEMA_VERSION,
        "public_repository": public_repository.as_posix(),
        "work_root": work_root.as_posix(),
        "runtime_root": runtime_root.as_posix(),
        "filesystem_policy": SANDBOX_FILESYSTEM_POLICY,
        "filesystem_base_token": SANDBOX_MINIMAL_TOKEN,
        "filesystem_base_mode": "read", "workspace_mode": "read",
        "workspace_catalog": workspace_catalog,
        "workspace_catalog_sha256": sha256_bytes(canonical_json(workspace_catalog)),
        "runtime_allow_roots": runtime_allow_roots,
        "runtime_allow_roots_sha256": sha256_bytes(canonical_json(runtime_allow_roots)),
        "codex_home": codex_home.as_posix(),
        "home_canary_root": home.as_posix(),
        "external_canary_root": external.as_posix(),
        "drvfs_roots": [drvfs.as_posix()],
        "home_roots": [home.as_posix()],
        "deny_roots": denies,
        "process_environment_contract": {
            "keys": [], "sha256": sha256_bytes(canonical_json({})),
        },
        "mcp_containment": {
            "schema_version": MCP_CONTAINMENT_SCHEMA_VERSION,
            "launcher": {
                "path": runtime["bubblewrap"]["path"],
                "filename": runtime["bubblewrap"]["filename"],
                "sha256": runtime["bubblewrap"]["sha256"],
            },
            "system_profile": MCP_SYSTEM_PROFILE,
            "network_mode": "unshare_all_no_share_net",
            "root_mode": "empty_exact_ro_bind",
            "shared_runtime_exclusions": [runtime["codex"]["path"]],
        },
    }
    boundary["identity_sha256"] = sha256_bytes(canonical_json(boundary))
    runtime["sandbox_boundary"] = boundary
    runtime["identity_sha256"] = sha256_bytes(canonical_json(runtime))
    return runtime
