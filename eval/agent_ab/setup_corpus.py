"""Materialize the four-arm public A/B corpus from a hash-locked manifest.

Network access is opt-in through ``--fetch``.  No HLS/Vivado tools are run.
"""
from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
from pathlib import Path
from typing import Any, Sequence

from .common import (
    ARM_IDS, HERE, EvalManifestError, load_corpus_lock, safe_relative_path,
    sha256_bytes, sha256_file,
)


MAX_DOWNLOAD_BYTES = 4 * 1024 * 1024
USER_AGENT = "hlsgraph-public-agent-eval/1"


def _download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 - allowlisted by manifest
        length = response.headers.get("Content-Length")
        if length is not None and int(length) > MAX_DOWNLOAD_BYTES:
            raise EvalManifestError(f"remote corpus file exceeds {MAX_DOWNLOAD_BYTES} bytes")
        data = response.read(MAX_DOWNLOAD_BYTES + 1)
    if len(data) > MAX_DOWNLOAD_BYTES:
        raise EvalManifestError(f"remote corpus file exceeds {MAX_DOWNLOAD_BYTES} bytes")
    return data


def _locked_bytes(
    entry: dict[str, Any], *, repo_root: Path, cache_root: Path, fetch: bool,
    downloads: list[str],
) -> bytes:
    expected = entry["sha256"]
    cached = cache_root / expected
    if cached.is_file():
        data = cached.read_bytes()
    elif "source_path" in entry:
        source = repo_root / safe_relative_path(entry["source_path"])
        if not source.is_file():
            raise EvalManifestError(f"missing repository-local corpus file: {source}")
        data = source.read_bytes()
    elif fetch:
        data = _download(entry["url"])
        cache_root.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(data)
        downloads.append(expected)
    else:
        raise EvalManifestError(
            f"remote corpus object {expected} is not cached; rerun with --fetch"
        )
    actual = sha256_bytes(data)
    if actual != expected:
        if cached.is_file():
            cached.unlink()
        raise EvalManifestError(f"corpus hash mismatch: expected {expected}, got {actual}")
    return data


def _provenance(corpus: dict[str, Any]) -> dict[str, Any]:
    authority = "sanitized_synthetic" if corpus["tool_evidence"] == "synthetic_only" else "source_only"
    value: dict[str, Any] = {
        "schema_version": "hlsgraph.agent_eval.provenance.v1",
        "corpus_id": corpus["id"],
        "repository": corpus["repository"],
        "revision": corpus.get("revision", "current-public-repository-fixture"),
        "license": corpus["license"],
        "license_url": corpus["license_url"],
        "authority": authority,
        "tool_evidence": corpus["tool_evidence"],
        "files": [
            {
                "path": item["destination"], "sha256": item["sha256"],
                "role": "parser_support" if item.get("parser_support") else "corpus",
            }
            for item in corpus["files"]
        ],
    }
    if corpus["id"] == "dataflow_gemm":
        value["resource_constraints"] = {
            "lut": 116000, "ff": 232000, "dsp": 1248,
            "bram_18k": 288, "uram": 64,
        }
        value["warning"] = (
            "All report-like inputs in this corpus are synthetic fixtures; they are not "
            "evidence of an AMD tool invocation."
        )
    else:
        value["warning"] = (
            "This corpus contains source only and has no executed tool, QoR, or verification evidence."
        )
    return value


def _artifact_kind(destination: str) -> tuple[str, str, dict[str, str]]:
    normalized = destination.replace("\\", "/")
    metadata: dict[str, str] = {}
    if normalized.startswith("support/include/"):
        return "source.hpp", "dependency", {"fixture_authority": "parser_support"}
    if normalized.endswith((".cpp", ".cc", ".c")):
        if normalized.endswith(("_test.cpp", "tb.cpp")):
            return "testbench.cpp", "testbench", metadata
        return "source.cpp", "design_source", metadata
    if normalized.endswith((".h", ".hpp")):
        return "source.hpp", "header", metadata
    if normalized.endswith("directives.tcl"):
        return "config.tcl", "hls_tcl", metadata
    if normalized.endswith("schedule.json"):
        return "amd.vitis.schedule_json", "schedule_report", {"fixture_authority": "synthetic"}
    if normalized.endswith("directive_status.json"):
        return "amd.vitis.directive_status", "directive_status", {"fixture_authority": "synthetic"}
    if normalized.endswith("dataflow_profile.json"):
        return "amd.vitis.dataflow_profile", "cosim_profile", {
            "fixture_authority": "synthetic", "workload_id": "tb.default",
        }
    if normalized.endswith("csim_result.json"):
        return "amd.vitis.csim_result", "csim_result", {
            "fixture_authority": "synthetic", "workload_id": "tb.default",
        }
    if normalized.endswith("dut_cosim.rpt"):
        return "amd.vitis.cosim_rpt", "cosim_report", {
            "fixture_authority": "synthetic", "workload_id": "tb.default",
        }
    if normalized.endswith("post_route_timing.rpt"):
        return "amd.vivado.post_route_timing", "post_route_timing", {
            "fixture_authority": "synthetic",
        }
    if normalized.endswith("post_route_utilization.rpt"):
        return "amd.vivado.post_route_utilization", "post_route_utilization", {
            "fixture_authority": "synthetic",
        }
    return "document.text", "documentation", metadata


def _toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _hlsgraph_manifest(corpus: dict[str, Any], *, schema_version: str) -> str:
    translation_units = corpus.get("translation_units") or [corpus["source"]]
    testbenches = [
        entry["destination"] for entry in corpus["files"]
        if entry["destination"].endswith(("_test.cpp", "tb.cpp"))
    ]
    lines = [
        f'schema_version = {_toml_string(schema_version)}',
        f'project_id = "eval.{corpus["id"]}"',
        f'name = {_toml_string(corpus["name"])}',
        "",
        "[build]",
        f'top = {_toml_string(corpus["top"])}',
        'language = "c++"',
        'flow_target = "vitis"',
        ("include_dirs = [\"support/include\", \"upstream\"]"
         if corpus["id"] == "bitonic" else
         "include_dirs = [\"support/include\"]"),
        "config_files = []",
        "tcl_files = [\"directives.tcl\"]" if corpus["id"] == "dataflow_gemm" else "tcl_files = []",
        "testbench_files = [" + ", ".join(_toml_string(item) for item in testbenches) + "]",
        "golden_files = []",
        "",
        "[target]",
        'vendor = "amd"',
        'part = "unspecified-public-eval"',
        "",
        "[[toolchains]]",
        'id = "amd.vitis.2024_2"',
        'vendor = "amd"',
        'name = "vitis_hls"',
        'version = "2024.2"',
        'build = "declared-evaluation-context-only"',
        "",
        "[[toolchains]]",
        'id = "amd.vivado.2024_2"',
        'vendor = "amd"',
        'name = "vivado"',
        'version = "2024.2"',
        'build = "declared-evaluation-context-only"',
        "",
        "[metadata.privacy]",
        'mcp_source_snippets = "bounded"',
        "",
    ]
    for unit in translation_units:
        arguments = ["-std=c++17", "-Isupport/include"]
        if corpus["id"] == "bitonic":
            arguments.append("-Iupstream")
        arguments.append("-I.")
        if unit.endswith((".h", ".hh", ".hpp", ".hxx")):
            arguments.extend(["-x", "c++"])
        lines.extend([
            "[[build.translation_units]]",
            f"file = {_toml_string(unit)}",
            'directory = "."',
            "arguments = [" + ", ".join(_toml_string(item) for item in arguments) + "]",
            "",
        ])
    for entry in corpus["files"]:
        destination = entry["destination"]
        kind, role, metadata = _artifact_kind(destination)
        if kind == "document.text":
            continue
        lines.extend([
            "[[artifact_paths]]",
            f'path = {_toml_string(destination)}',
            f'kind = {_toml_string(kind)}',
            f'role = {_toml_string(role)}',
            'access = "project"',
            'license = "Apache-2.0"',
        ])
        if metadata:
            encoded = ", ".join(
                f"{key} = {_toml_string(value)}" for key, value in sorted(metadata.items())
            )
            lines.append(f"metadata = {{ {encoded} }}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def materialize(
    output_root: Path, *, repo_root: Path, fetch: bool = False, force: bool = False,
) -> dict[str, Any]:
    lock = load_corpus_lock()
    output_root = output_root.resolve()
    repo_root = repo_root.resolve()
    cache_root = output_root / "_cache"
    if force and output_root.exists():
        controlled_default = (HERE / "work").resolve()
        marker = output_root / "materialization.json"
        if output_root != controlled_default and not marker.is_file():
            raise EvalManifestError(
                "refusing --force for an unmarked non-default directory"
            )
        if output_root == repo_root or output_root.parent == output_root:
            raise EvalManifestError("refusing to replace a repository or filesystem root")
        shutil.rmtree(output_root)
    elif output_root.exists():
        if not output_root.is_dir() or any(output_root.iterdir()):
            raise EvalManifestError(
                "refusing to reuse a non-empty corpus output without --force"
            )
    output_root.mkdir(parents=True, exist_ok=True)

    materialized: list[dict[str, Any]] = []
    downloads: list[str] = []
    for corpus in lock["corpora"]:
        blobs = {
            entry["destination"]: _locked_bytes(
                entry, repo_root=repo_root, cache_root=cache_root, fetch=fetch,
                downloads=downloads,
            )
            for entry in corpus["files"]
        }
        for arm in ARM_IDS:
            target = output_root / arm / corpus["id"]
            target.mkdir(parents=True, exist_ok=True)
            for destination, data in blobs.items():
                path = target / safe_relative_path(destination)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            provenance = json.dumps(
                _provenance(corpus), indent=2, sort_keys=True, ensure_ascii=False,
            ) + "\n"
            (target / "EVAL_PROVENANCE.json").write_text(provenance, encoding="utf-8")
            if arm.startswith("hlsgraph-"):
                version = "0.2.0" if arm == "hlsgraph-v02" else "0.3.0"
                (target / "hlsgraph.toml").write_text(
                    _hlsgraph_manifest(corpus, schema_version=version), encoding="utf-8",
                )
            materialized.append({
                "arm": arm, "corpus_id": corpus["id"], "path": str(target),
            })
    summary = {
        "schema_version": "hlsgraph.agent_eval.materialization.v1",
        "corpus_lock_sha256": sha256_file(Path(__file__).with_name("corpus.lock.json")),
        "network_used": bool(downloads),
        "downloaded_objects": sorted(downloads),
        "workspaces": materialized,
    }
    (output_root / "materialization.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "work")
    parser.add_argument("--repo-root", type=Path, default=HERE.parents[1])
    parser.add_argument("--fetch", action="store_true", help="download missing pinned public files")
    parser.add_argument("--force", action="store_true", help="replace the ignored work directory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = materialize(
        args.output, repo_root=args.repo_root, fetch=args.fetch, force=args.force,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
