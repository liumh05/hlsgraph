"""Command line interface for the public HLSGraph infrastructure package.

Commands are intentionally thin adapters over :class:`hlsgraph.sdk.Project` and the shared
query service.  Machine-readable JSON is the stable default output.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Sequence

from .bundle import GraphBundle
from .doctor import diagnose
from .manifest import manifest_template
from .model import DatasetManifest, json_ready
from .query import ExploreSpec, QuerySpec
from .runner import FakeRunner, LocalRunner, SSHRunner
from .sdk import Project
from .version import FEATURE_SCHEMA_VERSION, SCHEMA_VERSION, __version__


class CliError(RuntimeError):
    """Expected command failure suitable for a concise user-facing error."""


def _json(value: Any, *, compact: bool = False) -> str:
    return json.dumps(
        json_ready(value), ensure_ascii=False, sort_keys=True,
        separators=(",", ":") if compact else None,
        indent=None if compact else 2,
    )


def _emit(value: Any, args: argparse.Namespace) -> None:
    print(_json(value, compact=bool(getattr(args, "compact", False))))


def _project_root(args: argparse.Namespace) -> Path:
    return Path(getattr(args, "project", ".")).resolve()


def _open_project(args: argparse.Namespace) -> Project:
    return Project.open(_project_root(args))


def _cmd_init(args: argparse.Namespace) -> int:
    root = _project_root(args)
    root.mkdir(parents=True, exist_ok=True)
    project_id = args.project_id
    if not project_id:
        slug = re.sub(r"[^a-z0-9_]+", "_", root.name.casefold()).strip("_") or "project"
        project_id = f"local.{slug}"
    manifest_path = root / args.manifest
    if manifest_path.exists() and not args.force:
        raise CliError(f"manifest already exists: {manifest_path}; use --force to replace it")
    text = manifest_template(project_id, args.name or root.name, args.top, args.source)
    manifest_path.write_text(text, encoding="utf-8", newline="\n")
    project = Project.create_from_manifest(manifest_path, force=args.force)
    _emit({
        "command": "init", "project_id": project.bundle.manifest.project_id,
        "project_root": str(root), "manifest": str(manifest_path),
        "bundle": str(project.bundle.root),
        "private_source_embedded": False,
    }, args)
    return 0


def _cmd_index(args: argparse.Namespace) -> int:
    root = _project_root(args)
    if args.manifest:
        manifest = Path(args.manifest)
        if not manifest.is_absolute():
            manifest = root / manifest
        project = Project.create_from_manifest(manifest, force=args.force)
    elif (root / GraphBundle.DIRECTORY / "manifest.json").is_file():
        project = Project.open(root)
    else:
        manifest = root / "hlsgraph.toml"
        if not manifest.is_file():
            raise CliError("no bundle or hlsgraph.toml found; run `hlsgraph init` first")
        project = Project.create_from_manifest(manifest, force=args.force)
    result = project.index(degraded=args.degraded,
                           options={"extractor_plugins": args.extractor_plugin})
    payload = {"command": "index", **json_ready(result)}
    _emit(payload, args)
    return 0 if result.success else 1


def _cmd_status(args: argparse.Namespace) -> int:
    project = _open_project(args)
    snapshot = project.bundle.latest_snapshot()
    if snapshot is None or not project.bundle.store.has_graph(snapshot.id):
        payload = project.bundle.status()
        if snapshot is not None:
            payload["runs"] = len(project.bundle.store.runs(snapshot.id))
            payload["diagnostics"] = [json_ready(item)
                                      for item in project.bundle.store.diagnostics(snapshot.id)]
    else:
        payload = project.status().to_dict()
    _emit({"command": "status", **payload}, args)
    return 0


def _cmd_query(args: argparse.Namespace) -> int:
    result = _open_project(args).query(QuerySpec(
        query=args.query, kinds=args.kind, scope_id=args.scope,
        stages=args.stage, authorities=args.authority,
        limit=args.limit, cursor=args.cursor,
    ))
    _emit({"command": "query", **result.to_dict()}, args)
    return 0


def _cmd_explore(args: argparse.Namespace) -> int:
    result = _open_project(args).explore(ExploreSpec(
        query=args.query, scope_id=args.scope, view=args.view,
        depth=args.depth, top_k=args.top_k, cursor=args.cursor,
    ))
    _emit({"command": "explore", **result.to_dict()}, args)
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    project = _open_project(args)
    if args.backend == "fake":
        runner = FakeRunner()
    elif args.backend == "local":
        if not args.allow_execution:
            raise CliError("local execution requires the explicit --allow-execution flag")
        runner = LocalRunner(project.bundle.project_root, allow_execution=True)
    elif args.backend == "ssh":
        if not args.allow_execution:
            raise CliError("SSH execution requires the explicit --allow-execution flag")
        if not args.host or not args.remote_project_root:
            raise CliError("SSH execution requires --host and --remote-project-root")
        runner = SSHRunner(args.host, args.remote_project_root, allow_execution=True)
    else:  # defensive for callers constructing Namespace directly
        raise CliError(f"unsupported runner backend: {args.backend}")
    result = project.run(runner, stages=args.stage or None, timeout_s=args.timeout)
    payload = {
        "command": "run", "backend": args.backend,
        "runs": json_ready(result.runs), "gates": json_ready(result.gates),
        "correctness_checks": json_ready(result.correctness_checks),
        "gates_complete": result.gates_complete,
        "verified": result.verified, "stopped_after_stage": result.stopped_after_stage,
        "tool_truth": result.tool_truth,
        "backend_can_produce_tool_truth": args.backend in {"local", "ssh"},
    }
    _emit(payload, args)
    failed = (
        any(str(run.status) in {"failed", "cancelled", "skipped"}
            for run in result.runs)
        or any(str(status) == "fail" for status in result.gates.values())
        or any(str(status) == "fail" for status in result.correctness_checks.values())
    )
    return 1 if failed else 0


def _cmd_render(args: argparse.Namespace) -> int:
    project = _open_project(args)
    output = Path(args.output)
    if not output.is_absolute():
        output = project.bundle.project_root / output
    path = project.render(output, format=args.format, scope_id=args.scope)
    _emit({"command": "render", "format": args.format, "output": str(path)}, args)
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    project = _open_project(args)
    output = Path(args.output)
    if not output.is_absolute():
        output = project.bundle.project_root / output
    if args.kind == "graph":
        path = project.export_graph(output)
        payload: dict[str, Any] = {"command": "export", "kind": "graph", "output": str(path)}
    else:
        snapshot = project.bundle.latest_snapshot()
        if snapshot is None:
            raise CliError("project has no indexed snapshot")
        dataset = DatasetManifest(
            dataset_id=args.dataset_id or f"dataset.{project.bundle.manifest.project_id}",
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            snapshot_ids=[snapshot.id],
        )
        result = project.export_dataset(output, dataset, format=args.format)
        payload = {"command": "export", "kind": "dataset", **json_ready(result)}
    _emit(payload, args)
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    root = _project_root(args) if args.project is not None else None
    result = diagnose(root)
    _emit({"command": "doctor", **result}, args)
    if not result["healthy"]:
        return 1
    return 1 if args.strict and result["summary"]["warn"] else 0


def _cmd_migrate(args: argparse.Namespace) -> int:
    bundle = GraphBundle.open(_project_root(args))
    plan = bundle.store.migration_plan(args.to_version)
    applied: list[dict[str, str]] = []
    if args.apply:
        applied = bundle.store.migrate(args.to_version)
    _emit({
        "command": "migrate", "target_version": args.to_version,
        "apply_requested": bool(args.apply), "plan": plan, "applied": applied,
        "implicit_migration": False,
    }, args)
    return 0


def _cmd_knowledge(args: argparse.Namespace) -> int:
    # Keep the knowledge framework out of ordinary CLI startup while still delegating all
    # validation/filter semantics to its public API.
    from .knowledge import (
        KnowledgeCatalog, index_local_document, load_local_index, save_local_index,
    )

    if args.action == "index":
        if not args.path or not args.document_id or not args.document_version:
            raise CliError(
                "knowledge index requires --path, --document-id, and --document-version"
            )
        root = _project_root(args)
        source = Path(args.path).resolve()
        output = Path(args.output) if args.output else root / ".hlsgraph" / "knowledge-index.json"
        if not output.is_absolute():
            output = root / output
        existing = load_local_index(output) if output.is_file() else []
        entry = index_local_document(
            source, document_id=args.document_id, document_version=args.document_version,
            title=args.title, official_url=args.official_url,
        )
        entries = [item for item in existing if not (
            item.document_id == entry.document_id
            and item.document_version == entry.document_version
        )]
        entries.append(entry)
        save_local_index(entries, output)
        _emit({
            "command": "knowledge", "action": "index", "output": str(output),
            "document": json_ready(entry), "content_copied": False,
        }, args)
        return 0

    catalog = KnowledgeCatalog.builtin()
    applicability = {key: value for key, value in {
        "vendor": args.vendor, "tool": args.tool, "stage": args.stage,
    }.items() if value is not None}
    rules = catalog.filter(
        document_id=args.document_id, document_version=args.document_version,
        applicability=applicability or None,
    )
    if args.rule_id:
        rules = [rule for rule in rules if rule.rule_id == args.rule_id]
    packs = [{
        "pack_id": pack.pack_id, "title": pack.title,
        "schema_version": pack.schema_version, "license": pack.license,
        "documents": json_ready(pack.documents), "rule_count": len(pack.rules),
    } for pack in catalog.packs]
    _emit({
        "command": "knowledge", "action": "list", "packs": packs,
        "rules": json_ready(rules), "count": len(rules),
        "metadata_only": True,
    }, args)
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    # Optional networking code is intentionally absent from CLI import/startup paths.
    try:
        from .api import serve
    except ImportError as exc:
        raise CliError(f"REST server is unavailable: {exc}") from exc
    serve(_project_root(args), host=args.host, port=args.port,
          snapshot_id=args.snapshot, allow_remote=args.allow_remote)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hlsgraph",
        description="Deterministic, evidence-backed HLS architecture graph infrastructure.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--compact", action="store_true", help="emit compact JSON")
    sub = parser.add_subparsers(dest="command", required=True)

    def project_option(command: argparse.ArgumentParser, *, nullable: bool = False) -> None:
        command.add_argument("--project", default=None if nullable else ".",
                             help="project root (default: current directory)")

    init = sub.add_parser("init", help="create a manifest and local GraphBundle")
    project_option(init)
    init.add_argument("--project-id", help="namespaced stable project ID")
    init.add_argument("--name")
    init.add_argument("--top", required=True)
    init.add_argument("--source", required=True, help="project-relative HLS source path")
    init.add_argument("--manifest", default="hlsgraph.toml")
    init.add_argument("--force", action="store_true")
    init.set_defaults(func=_cmd_init)

    index = sub.add_parser("index", help="build a new deterministic graph snapshot")
    project_option(index)
    index.add_argument("--manifest", help="manifest path, relative to project root")
    index.add_argument("--force", action="store_true")
    index.add_argument("--degraded", action="store_true",
                       help="explicitly use the regex source scanner instead of libclang")
    index.add_argument("--extractor-plugin", action="append", default=[],
                       help="explicitly enable a named hlsgraph.extractors.v1 entry point")
    index.set_defaults(func=_cmd_index)

    status = sub.add_parser("status", help="show bundle, graph, run, and staleness status")
    project_option(status)
    status.set_defaults(func=_cmd_status)

    query = sub.add_parser("query", help="search entities through the shared query service")
    project_option(query)
    query.add_argument("query")
    query.add_argument("--kind", action="append", default=[])
    query.add_argument("--scope")
    query.add_argument("--stage", action="append", default=[])
    query.add_argument("--authority", action="append", default=[])
    query.add_argument("--limit", type=int, default=20)
    query.add_argument("--cursor")
    query.set_defaults(func=_cmd_query)

    explore = sub.add_parser("explore", help="retrieve an evidence-backed graph neighborhood")
    project_option(explore)
    explore.add_argument("query", nargs="?")
    explore.add_argument("--scope")
    explore.add_argument("--view", default="architecture")
    explore.add_argument("--depth", type=int, default=1)
    explore.add_argument("--top-k", type=int, default=8)
    explore.add_argument("--cursor")
    explore.set_defaults(func=_cmd_explore)

    run = sub.add_parser("run", help="execute manifest stages through an explicit runner")
    project_option(run)
    run.add_argument("--backend", choices=("local", "ssh", "fake"), required=True)
    run.add_argument("--allow-execution", action="store_true",
                     help="required safety acknowledgement for local or SSH execution")
    run.add_argument("--stage", action="append", default=[])
    run.add_argument("--timeout", type=float, default=7200.0)
    run.add_argument("--host", help="SSH destination")
    run.add_argument("--remote-project-root", help="absolute project root on SSH host")
    run.set_defaults(func=_cmd_run)

    render = sub.add_parser("render", help="render the shared canonical graph projection")
    project_option(render)
    render.add_argument("output")
    render.add_argument("--format", choices=("html", "json", "mermaid", "dot", "svg"),
                        default="html")
    render.add_argument("--scope")
    render.set_defaults(func=_cmd_render)

    export = sub.add_parser("export", help="export a graph or leakage-aware ML dataset")
    project_option(export)
    export.add_argument("output")
    export.add_argument("--kind", choices=("graph", "dataset"), default="graph")
    export.add_argument("--format", choices=("jsonl", "parquet"), default="jsonl")
    export.add_argument("--dataset-id")
    export.set_defaults(func=_cmd_export)

    doctor = sub.add_parser("doctor", help="perform read-only environment and bundle checks")
    project_option(doctor, nullable=True)
    doctor.add_argument("--strict", action="store_true", help="treat optional warnings as failure")
    doctor.set_defaults(func=_cmd_doctor)

    migrate = sub.add_parser("migrate", help="inspect or explicitly apply registered ledger migrations")
    project_option(migrate)
    migrate.add_argument("--to-version", default=SCHEMA_VERSION)
    migrate.add_argument("--apply", action="store_true",
                         help="apply the displayed, registered migration path")
    migrate.set_defaults(func=_cmd_migrate)

    knowledge = sub.add_parser("knowledge", help="list and inspect packaged knowledge rules")
    project_option(knowledge)
    knowledge.add_argument("action", nargs="?", choices=("list", "index"), default="list")
    knowledge.add_argument("--document-id")
    knowledge.add_argument("--document-version")
    knowledge.add_argument("--rule-id")
    knowledge.add_argument("--vendor")
    knowledge.add_argument("--tool")
    knowledge.add_argument("--stage")
    knowledge.add_argument("--path", help="user-owned local document to hash (index action)")
    knowledge.add_argument("--title")
    knowledge.add_argument("--official-url")
    knowledge.add_argument("--output", help="metadata-only local index path")
    knowledge.set_defaults(func=_cmd_knowledge)

    serve = sub.add_parser("serve", help="serve the versioned read-only REST API")
    project_option(serve)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--snapshot")
    serve.add_argument("--allow-remote", action="store_true",
                       help="explicitly permit a non-loopback bind")
    serve.set_defaults(func=_cmd_serve)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print(_json({"error": "interrupted", "command": args.command},
                    compact=bool(args.compact)), file=sys.stderr)
        return 130
    except (CliError, OSError, ValueError, KeyError, RuntimeError) as exc:
        print(_json({"error": str(exc), "type": type(exc).__name__,
                     "command": args.command}, compact=bool(args.compact)), file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
