from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from eval.agent_ab.common import (
    ARM_IDS, canonical_json, load_corpus_lock, load_manifest, sha256_bytes,
)


def synthetic_cold_start_input_sha256(arm: str, corpus_id: str) -> str:
    return sha256_bytes(canonical_json({
        "arm": arm, "corpus_id": corpus_id, "kind": "synthetic-cold-input",
    }))


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
    denies = sorted({
        public_repository.as_posix(), work_root.as_posix(), codex_home.as_posix(),
        external.as_posix(), drvfs.as_posix(), home.as_posix(),
    })
    corpus_ids = [item["id"] for item in load_corpus_lock()["corpora"]]
    deny_catalog: dict[str, Any] = {
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
    boundary: dict[str, Any] = {
        "schema_version": "hlsgraph.agent_eval.sandbox_boundary.v1",
        "public_repository": public_repository.as_posix(),
        "work_root": work_root.as_posix(),
        "work_root_policy": "explicit_sibling_directory_deny_v1",
        "work_root_deny_catalog": deny_catalog,
        "work_root_deny_catalog_sha256": sha256_bytes(canonical_json(deny_catalog)),
        "codex_home": codex_home.as_posix(),
        "home_canary_root": home.as_posix(),
        "external_canary_root": external.as_posix(),
        "drvfs_roots": [drvfs.as_posix()],
        "home_roots": [home.as_posix()],
        "deny_roots": denies,
    }
    boundary["identity_sha256"] = sha256_bytes(canonical_json(boundary))
    runtime_root = parent / "synthetic-runtime"
    runtime_root.mkdir(parents=True, exist_ok=True)
    for name in (
        "fixture", "codex", "codex-linux-sandbox", "bwrap", "node",
        "codegraph.js", "harness-python", "v02-python", "v03-python",
    ):
        path = runtime_root / name
        path.write_bytes(b"synthetic non-executable unit-test fixture\n")
        if os.name == "posix":
            path.chmod(0o700)
    binary = {"filename": "fixture", "path": (runtime_root / "fixture").as_posix(),
              "version": "fixture-1", "sha256": "1" * 64}
    python = {
        "filename": "python", "path": (runtime_root / "python").as_posix(),
        "version": "3.10.12", "implementation": "CPython",
        "cache_tag": "cpython-310", "platform": "Linux-fixture",
        "sha256": "2" * 64,
    }
    runtime: dict[str, Any] = {
        "schema_version": "hlsgraph.agent_eval.runtime_identity.v1",
        "host": {
            "os_name": "posix", "system": "Linux",
            "release": "6.18.0-microsoft-standard-WSL2", "machine": "x86_64",
            "distribution_id": "ubuntu", "distribution_version": "22.04",
            "wsl": "2", "wsl_distribution": "Ubuntu-22.04",
        },
        "sandbox_boundary": boundary,
        "codex": {**binary, "path": (runtime_root / "codex").as_posix(),
                  "filename": "codex", "version": "codex-cli 0.144.0", "sha256": "3" * 64},
        "codex_linux_sandbox": {
            **binary, "path": (runtime_root / "codex-linux-sandbox").as_posix(),
            "filename": "codex-linux-sandbox", "sha256": "4" * 64,
        },
        "bubblewrap": {**binary, "path": (runtime_root / "bwrap").as_posix(),
                       "filename": "bwrap", "version": "bubblewrap 0.6.1", "sha256": "5" * 64},
        "node": {**binary, "path": (runtime_root / "node").as_posix(),
                 "filename": "node", "version": "v22.16.0", "sha256": "6" * 64},
        "codegraph_entrypoint": {
            "path": (runtime_root / "codegraph.js").as_posix(),
            "filename": "codegraph.js",
            "sha256": load_manifest()["arms"][1]["entrypoint_sha256"],
        },
        "python": {
            "harness": {**python, "path": (runtime_root / "harness-python").as_posix(),
                        "filename": "harness-python", "sha256": "7" * 64},
            "hlsgraph_v02": {**python, "path": (runtime_root / "v02-python").as_posix(),
                             "filename": "v02-python", "sha256": "8" * 64},
            "hlsgraph_v03": {**python, "path": (runtime_root / "v03-python").as_posix(),
                             "filename": "v03-python", "sha256": "9" * 64},
        },
    }
    runtime["identity_sha256"] = sha256_bytes(canonical_json(runtime))
    return runtime
