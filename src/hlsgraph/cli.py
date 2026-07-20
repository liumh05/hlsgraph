"""Command line interface for the public HLSGraph infrastructure package.

Commands are intentionally thin adapters over :class:`hlsgraph.sdk.Project` and the shared
query service.  Machine-readable JSON is the stable default output.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import stat
import sys
import uuid
from pathlib import Path
from typing import Any, Sequence

from .bundle import GraphBundle
from .diagnostic_projection import public_diagnostic
from .doctor import diagnose
from .manifest import manifest_template, parse_manifest_text, safe_relative_path
from .model import DatasetManifest, json_ready
from .plugins import load_runners
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


def _path_identity(path: Path) -> tuple[int, int]:
    value = path.stat(follow_symlinks=False)
    return int(value.st_dev), int(value.st_ino)


def _is_reparse_path(path: Path) -> bool:
    """Return True for symlinks and Windows junction/reparse entries."""
    try:
        value = path.lstat()
    except OSError:
        return False
    if stat.S_ISLNK(value.st_mode):
        return True
    attributes = int(getattr(value, "st_file_attributes", 0))
    reparse_flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    if attributes & reparse_flag:
        return True
    is_junction = getattr(path, "is_junction", None)
    try:
        return bool(callable(is_junction) and is_junction())
    except OSError:
        # A path that cannot be classified stably is unsafe for publication.
        return True


DirectoryChain = tuple[tuple[Path, tuple[int, int]], ...]


def _prepare_directory_chain(root: Path, parent: Path) -> DirectoryChain:
    """Create ordinary parents and capture every no-follow directory identity."""
    try:
        relative = parent.relative_to(root)
    except ValueError as exc:
        raise CliError("manifest parent must remain inside the project root") from exc
    if (_is_reparse_path(root) or not os.path.lexists(root)
            or not stat.S_ISDIR(root.lstat().st_mode)):
        raise CliError("project root must be an ordinary directory")
    chain: list[tuple[Path, tuple[int, int]]] = [(root, _path_identity(root))]
    current = root
    for component in relative.parts:
        current = current / component
        if not os.path.lexists(current):
            try:
                current.mkdir()
            except FileExistsError:
                pass
        # Validate before descending to the next component so we never
        # knowingly create a child through a symlink/junction won by a peer.
        if (_is_reparse_path(current) or not os.path.lexists(current)
                or not stat.S_ISDIR(current.lstat().st_mode)):
            raise CliError("init manifest parent must contain only ordinary directories")
        chain.append((current, _path_identity(current)))
    _validate_directory_chain(tuple(chain))
    return tuple(chain)


def _validate_directory_chain(chain: DirectoryChain) -> None:
    """Fail if any named parent was replaced or became a reparse point."""
    for path, identity in chain:
        try:
            value = path.lstat()
        except OSError as exc:
            raise FileExistsError(
                "manifest parent changed concurrently; refusing publication"
            ) from exc
        if (_is_reparse_path(path) or not stat.S_ISDIR(value.st_mode)
                or _path_identity(path) != identity):
            raise FileExistsError(
                "manifest parent changed concurrently; refusing publication"
            )


def _exclusive_flags() -> int:
    return (os.O_CREAT | os.O_EXCL | os.O_WRONLY
            | int(getattr(os, "O_NOFOLLOW", 0))
            | int(getattr(os, "O_NOINHERIT", 0))
            | int(getattr(os, "O_BINARY", 0)))


def _write_all(descriptor: int, data: bytes) -> None:
    offset = 0
    while offset < len(data):
        written = os.write(descriptor, data[offset:])
        if written <= 0:
            raise OSError("short write while publishing initialization metadata")
        offset += written


def _open_owner_file(path: Path, token: str) -> tuple[int, tuple[int, int]]:
    """Open an O_EXCL/no-follow token file and keep its descriptor owned."""
    descriptor = os.open(path, _exclusive_flags(), 0o600)
    identity: tuple[int, int] | None = None
    try:
        value = os.fstat(descriptor)
        if not stat.S_ISREG(value.st_mode):
            raise OSError("owner claim is not a regular file")
        identity = int(value.st_dev), int(value.st_ino)
        _write_all(descriptor, (token + "\n").encode("ascii"))
        os.fsync(descriptor)
        if (_is_reparse_path(path) or _path_identity(path) != identity
                or not stat.S_ISREG(path.lstat().st_mode)):
            raise FileExistsError("owner claim path changed concurrently")
        return descriptor, identity
    except BaseException:
        os.close(descriptor)
        try:
            if (identity is not None and not _is_reparse_path(path)
                    and _path_identity(path) == identity):
                path.unlink()
        except OSError:
            pass
        raise


def _open_verified_directory(path: Path, identity: tuple[int, int]) -> int | None:
    """Hold a stable directory fd where the platform supports dir_fd APIs."""
    if os.name == "nt":
        return None
    flags = (os.O_RDONLY | int(getattr(os, "O_DIRECTORY", 0))
             | int(getattr(os, "O_NOFOLLOW", 0)))
    descriptor = os.open(path, flags)
    value = os.fstat(descriptor)
    if ((int(value.st_dev), int(value.st_ino)) != identity
            or not stat.S_ISDIR(value.st_mode)):
        os.close(descriptor)
        raise FileExistsError("manifest parent changed concurrently")
    return descriptor


def _read_descriptor(descriptor: int) -> bytes:
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, 65536)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _atomic_write_text(
    path: Path, text: str, *, replace: bool = True,
    expected_identity: tuple[int, int] | None = None,
    expected_bytes: bytes | None = None,
    directory_chain: DirectoryChain | None = None,
) -> None:
    """Publish text atomically, optionally with fail-closed replacement.

    The no-replace path uses a hard-link publish so an existing file or
    dangling symlink wins the race.  The replace path is reserved for an
    explicit ``--force`` and verifies that the file observed by the caller has
    not been replaced or edited in place before publishing.
    """
    if replace and expected_identity is None:
        raise ValueError("atomic replacement requires an expected manifest identity")
    if directory_chain is None:
        if (_is_reparse_path(path.parent) or not path.parent.is_dir()):
            raise FileExistsError("manifest parent is not an ordinary directory")
        directory_chain = ((path.parent, _path_identity(path.parent)),)
    _validate_directory_chain(directory_chain)

    parent_identity = directory_chain[-1][1]
    directory_fd = _open_verified_directory(path.parent, parent_identity)
    guard_token = uuid.uuid4().hex
    guard_path = path.parent / f".hlsgraph-init-parent.{guard_token}.lock"
    temporary_name = f".{path.name}.{uuid.uuid4().hex}.tmp"
    temporary = path.parent / temporary_name
    guard_descriptor = -1
    guard_identity: tuple[int, int] | None = None
    descriptor = -1
    temporary_identity: tuple[int, int] | None = None
    try:
        guard_descriptor, guard_identity = _open_owner_file(guard_path, guard_token)
        _validate_directory_chain(directory_chain)
        if directory_fd is None:
            descriptor = os.open(temporary, _exclusive_flags(), 0o600)
        else:
            descriptor = os.open(
                temporary_name, _exclusive_flags(), 0o600, dir_fd=directory_fd,
            )
        temporary_value = os.fstat(descriptor)
        temporary_identity = int(temporary_value.st_dev), int(temporary_value.st_ino)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        if guard_descriptor >= 0:
            os.close(guard_descriptor)
        if guard_identity is not None:
            _remove_owned_file(guard_path, guard_identity, guard_token)
        if directory_fd is not None:
            os.close(directory_fd)
        raise
    published = False
    try:
        _write_all(descriptor, text.encode("utf-8"))
        os.fsync(descriptor)
        _validate_directory_chain(directory_chain)
        if replace:
            if expected_identity is not None:
                target_flags = (os.O_RDONLY | int(getattr(os, "O_NOFOLLOW", 0))
                                | int(getattr(os, "O_BINARY", 0)))
                if _is_reparse_path(path):
                    raise FileExistsError(
                        "manifest changed concurrently; refusing to follow a reparse point"
                    )
                if directory_fd is None:
                    target_descriptor = os.open(path, target_flags)
                else:
                    target_descriptor = os.open(
                        path.name, target_flags, dir_fd=directory_fd,
                    )
                try:
                    target_value = os.fstat(target_descriptor)
                    target_identity = int(target_value.st_dev), int(target_value.st_ino)
                    target_bytes = _read_descriptor(target_descriptor)
                finally:
                    os.close(target_descriptor)
                if (_is_reparse_path(path)
                        or target_identity != expected_identity
                        or (expected_bytes is not None
                            and target_bytes != expected_bytes)):
                    raise FileExistsError(
                        "manifest changed concurrently; refusing to overwrite peer data"
                    )
            # Windows cannot replace an open source file.  The still-open
            # parent guard prevents directory replacement while this one
            # descriptor is briefly closed.  POSIX publishes relative to the
            # already-verified directory fd.
            os.close(descriptor)
            descriptor = -1
            _validate_directory_chain(directory_chain)
            if directory_fd is None:
                os.replace(temporary, path)
            else:
                os.replace(
                    temporary_name, path.name,
                    src_dir_fd=directory_fd, dst_dir_fd=directory_fd,
                )
        else:
            # link(2)/CreateHardLink publishes the fully-fsynced inode without
            # replacing any directory entry that appeared after our precheck.
            if directory_fd is None:
                os.link(temporary, path, follow_symlinks=False)
            else:
                os.link(
                    temporary_name, path.name,
                    src_dir_fd=directory_fd, dst_dir_fd=directory_fd,
                    follow_symlinks=False,
                )
        published = True
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
            descriptor = -1
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            if directory_fd is None:
                if (temporary_identity is not None
                        and not _is_reparse_path(temporary)
                        and os.path.lexists(temporary)
                        and _path_identity(temporary) == temporary_identity):
                    temporary.unlink()
            else:
                try:
                    os.unlink(temporary_name, dir_fd=directory_fd)
                except FileNotFoundError:
                    pass
        except OSError:
            # Once the target is published, an unremovable temp hard link is
            # a harmless orphan and must not trigger bundle rollback.  Before
            # publication, preserve the original exception rather than mask it.
            if published:
                pass
        os.close(guard_descriptor)
        if guard_identity is not None:
            _remove_owned_file(guard_path, guard_identity, guard_token)
        if directory_fd is not None:
            os.close(directory_fd)


def _bundle_directory_identity(path: Path) -> tuple[int, int]:
    return _path_identity(path)


def _claim_owner_file(path: Path, token: str) -> tuple[int, int]:
    """Atomically create a token file and return the identity we own."""
    descriptor, identity = _open_owner_file(path, token)
    os.close(descriptor)
    return identity


def _remove_owned_file(path: Path, identity: tuple[int, int], token: str) -> bool:
    """Remove a coordination file only while identity and token still match."""
    try:
        if (_is_reparse_path(path) or _path_identity(path) != identity
                or path.read_text(encoding="ascii").strip() != token):
            return False
        path.unlink()
        return True
    except OSError:
        return False


def _cleanup_owned_bundle(path: Path, identity: tuple[int, int] | None, *,
                          owner_token: str, marker_required: bool,
                          marker_identity: tuple[int, int] | None = None) -> bool:
    """Best-effort rollback without deleting a directory created by a peer."""
    if identity is None:
        return True
    try:
        if _is_reparse_path(path) or _bundle_directory_identity(path) != identity:
            return False
        marker = path / ".init-owner"
        if marker_required:
            if (marker_identity is None or _is_reparse_path(marker)
                    or not marker.is_file()
                    or _path_identity(marker) != marker_identity
                    or marker.read_text(encoding="ascii").strip() != owner_token):
                return False
        else:
            # Without a successfully-published token we only own the empty
            # directory entry.  Never recursively remove files that could
            # have appeared after the marker creation failed.
            path.rmdir()
            return not os.path.lexists(path)
        # sqlite3's connection context manager commits but does not close the
        # connection object.  CPython 3.13 can therefore retain an unreachable
        # initialization connection until cyclic GC, which keeps graph.db
        # undeletable on Windows.  Collect before the first recursive attempt;
        # a failed rmtree could otherwise remove the owner marker and make a
        # token-verified retry impossible.
        import gc
        gc.collect()
        shutil.rmtree(path)
        return not os.path.lexists(path)
    except OSError:
        return False


def _open_project(args: argparse.Namespace) -> Project:
    return Project.open(_project_root(args))


def _cmd_init(args: argparse.Namespace) -> int:
    root = _project_root(args)
    root.mkdir(parents=True, exist_ok=True)
    project_id = args.project_id
    if not project_id:
        slug = re.sub(r"[^a-z0-9_]+", "_", root.name.casefold()).strip("_") or "project"
        project_id = f"local.{slug}"
    manifest_path = root / Path(safe_relative_path(args.manifest))
    if manifest_path.suffix.casefold() != ".toml":
        raise CliError("init manifest must use the .toml extension")
    text = manifest_template(project_id, args.name or root.name, args.top, args.source)
    manifest = parse_manifest_text(text, format="toml")

    bundle_path = root / GraphBundle.DIRECTORY
    lock_path = root / ".hlsgraph-init.lock"
    lock_token = uuid.uuid4().hex
    try:
        lock_identity = _claim_owner_file(lock_path, lock_token)
    except FileExistsError as exc:
        raise CliError(
            "another project initialization owns .hlsgraph-init.lock; "
            "inspect it before removing a stale lock"
        ) from exc

    owned_identity: tuple[int, int] | None = None
    owner_marker = bundle_path / ".init-owner"
    owner_token = uuid.uuid4().hex
    marker_created = False
    marker_identity: tuple[int, int] | None = None
    manifest_published = False
    try:
        if os.path.lexists(bundle_path):
            # ``init`` is a creation transaction, not a ledger migration.
            raise CliError(
                "an HLSGraph bundle already exists; init will not replace an existing ledger"
            )

        # Capture the complete ordinary-directory chain.  Publication later
        # revalidates every identity and, where available, uses a stable dir_fd
        # so a nested parent cannot be swapped for a symlink or junction after
        # this precheck.
        manifest_directory_chain = _prepare_directory_chain(
            root, manifest_path.parent,
        )
        manifest_preexisting = os.path.lexists(manifest_path)
        if (manifest_preexisting and (_is_reparse_path(manifest_path)
                or not stat.S_ISREG(manifest_path.lstat().st_mode))):
            raise CliError(
                "init manifest target must be a regular file, not a symlink or reparse point"
            )
        if manifest_preexisting and not args.force:
            raise CliError(
                f"manifest already exists: {manifest_path}; use --force to replace it"
            )
        expected_manifest_identity = (
            _path_identity(manifest_path) if manifest_preexisting else None
        )
        expected_manifest_bytes = (
            manifest_path.read_bytes() if manifest_preexisting else None
        )

        try:
            # Atomically claim this exact directory.  A pre-check alone is
            # racy and must never authorize cleanup of a peer ledger.
            bundle_path.mkdir(parents=False, exist_ok=False)
            owned_identity = _bundle_directory_identity(bundle_path)
            marker_identity = _claim_owner_file(owner_marker, owner_token)
            marker_created = True
            if _bundle_directory_identity(bundle_path) != owned_identity:
                raise FileExistsError("bundle directory changed during owner claim")
            project = Project(GraphBundle.create(
                root, manifest, force=False, manifest_source=manifest_path,
            ))
            # Keep the ownership marker live until the manifest has been
            # published.  With no pre-existing manifest this is an atomic
            # no-clobber operation; --force may replace only the exact bytes
            # and inode observed under the init lock.
            _atomic_write_text(
                manifest_path, text, replace=manifest_preexisting,
                expected_identity=expected_manifest_identity,
                expected_bytes=expected_manifest_bytes,
                directory_chain=manifest_directory_chain,
            )
            manifest_published = True
            if marker_identity is not None:
                _remove_owned_file(owner_marker, marker_identity, owner_token)
        except BaseException as exc:
            cleaned = False
            if not manifest_published:
                cleaned = _cleanup_owned_bundle(
                    bundle_path, owned_identity, owner_token=owner_token,
                    marker_required=marker_created,
                    marker_identity=marker_identity,
                )
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            suffix = "" if cleaned else "; owned partial state could not be removed"
            raise CliError(
                f"project initialization failed ({type(exc).__name__}){suffix}"
            ) from exc
    finally:
        _remove_owned_file(lock_path, lock_identity, lock_token)
    _emit({
        "command": "init", "project_id": project.bundle.manifest.project_id,
        "project_root": str(root), "manifest": str(manifest_path),
        "bundle": str(project.bundle.root),
        "private_source_embedded": False,
    }, args)
    return 0


def _cmd_index(args: argparse.Namespace) -> int:
    root = _project_root(args)
    if args.action_id:
        if args.manifest:
            raise CliError("--action-id cannot be combined with --manifest")
        if not (root / GraphBundle.DIRECTORY / "manifest.json").is_file():
            raise CliError("variant indexing requires an existing bundle")
        project = Project.open(root)
    elif args.manifest:
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
    options = {"extractor_plugins": args.extractor_plugin}
    result = (
        project.index_variant(args.action_id, degraded=args.degraded, options=options)
        if args.action_id else project.index(degraded=args.degraded, options=options)
    )
    payload = {"command": "index", **json_ready(result)}
    _emit(payload, args)
    return 0 if result.success else 1


def _cmd_status(args: argparse.Namespace) -> int:
    project = _open_project(args)
    snapshot = (project.bundle.latest_snapshot()
                or project.bundle.store.latest_candidate(
                    project.bundle.manifest.project_id
                ))
    if snapshot is None or not project.bundle.store.has_graph(snapshot.id):
        payload = project.bundle.status(snapshot.id if snapshot else None)
        if snapshot is not None:
            payload["runs"] = len(project.bundle.store.runs(snapshot.id))
            payload["diagnostics"] = [public_diagnostic(item)
                                      for item in project.bundle.store.diagnostics(snapshot.id)]
    else:
        payload = project.status().to_dict()
    _emit({"command": "status", **payload}, args)
    return 0


def _cmd_query(args: argparse.Namespace) -> int:
    project = _open_project(args)
    service = project.service(args.snapshot_id)
    if args.record_class == "entity":
        if not args.query:
            raise CliError("entity query requires search text")
        payload = service.query(QuerySpec(
            query=args.query, kinds=args.kind, scope_id=args.scope,
            stages=args.stage, authorities=args.authority,
            limit=args.limit if args.limit is not None else 20,
            cursor=args.cursor,
        )).to_dict()
    elif args.record_class == "feature-evidence":
        if args.query is not None:
            raise CliError("feature-evidence query uses --entity-id and --predicate filters")
        payload = service.feature_evidence(
            args.entity_id, predicates=args.predicate,
            stages=args.stage,
            limit=args.limit if args.limit is not None else 100,
        )
    else:
        if args.query is not None:
            raise CliError("correspondence query uses explicit endpoint filters")
        payload = service.correspondences(
            args.entity_id, other_snapshot_id=args.other_snapshot_id,
            kinds=args.correspondence_kind, direction=args.direction,
            limit=args.limit if args.limit is not None else 100,
        )
    _emit({"command": "query", **payload}, args)
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
        runner = SSHRunner(
            args.host, args.remote_project_root,
            project_root=project.bundle.project_root, allow_execution=True,
        )
    elif args.backend == "plugin":
        if not args.allow_execution:
            raise CliError("runner plugin execution requires the explicit --allow-execution flag")
        if not args.runner_plugin:
            raise CliError("plugin execution requires --runner-plugin")
        try:
            config = json.loads(args.runner_config)
        except json.JSONDecodeError as exc:
            raise CliError("--runner-config must be a JSON object") from exc
        if not isinstance(config, dict):
            raise CliError("--runner-config must be a JSON object")
        runner = load_runners(
            [args.runner_plugin], {args.runner_plugin: config},
        )[0]
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
        "backend_can_produce_tool_truth": (
            str(getattr(runner, "name", "")).casefold()
            not in {"runner.fake", "runner.replay"}
        ),
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
        snapshot_ids = list(dict.fromkeys(args.snapshot_id or [snapshot.id]))
        dataset = DatasetManifest(
            dataset_id=args.dataset_id or f"dataset.{project.bundle.manifest.project_id}",
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            snapshot_ids=snapshot_ids,
            feature_evidence_predicates=args.feature_evidence_predicate,
            entity_correspondence_kinds=args.entity_correspondence_kind,
        )
        result = project.export_dataset(
            output, dataset, format=args.format, snapshot_id=snapshot_ids[0],
        )
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
    root = _project_root(args)
    plan = GraphBundle.migration_plan(root, args.to_version)
    applied: list[dict[str, str]] = []
    if args.apply:
        applied = GraphBundle.migrate(root, args.to_version)
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
    init.add_argument(
        "--force", action="store_true",
        help="replace an existing manifest only when no .hlsgraph ledger exists",
    )
    init.set_defaults(func=_cmd_init)

    index = sub.add_parser("index", help="build a new deterministic graph snapshot")
    project_option(index)
    index.add_argument("--manifest", help="manifest path, relative to project root")
    index.add_argument("--force", action="store_true")
    index.add_argument("--degraded", action="store_true",
                       help="explicitly use the regex source scanner instead of libclang")
    index.add_argument("--extractor-plugin", action="append", default=[],
                       help="explicitly enable a named hlsgraph.extractors.v1 entry point")
    index.add_argument("--action-id",
                       help="materialize one previously recorded variant action")
    index.set_defaults(func=_cmd_index)

    status = sub.add_parser("status", help="show bundle, graph, run, and staleness status")
    project_option(status)
    status.set_defaults(func=_cmd_status)

    query = sub.add_parser("query", help="search entities through the shared query service")
    project_option(query)
    query.add_argument("query", nargs="?")
    query.add_argument(
        "--record-class",
        choices=("entity", "feature-evidence", "correspondence"),
        default="entity",
    )
    query.add_argument("--snapshot-id", help="immutable snapshot to query")
    query.add_argument("--kind", action="append", default=[])
    query.add_argument("--scope")
    query.add_argument("--stage", action="append", default=[])
    query.add_argument("--authority", action="append", default=[])
    query.add_argument("--entity-id")
    query.add_argument("--predicate", action="append", default=[])
    query.add_argument("--other-snapshot-id")
    query.add_argument("--correspondence-kind", action="append", default=[])
    query.add_argument("--direction", choices=("source", "target", "both"),
                       default="both")
    query.add_argument("--limit", type=int)
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
    run.add_argument("--backend", choices=("local", "ssh", "fake", "plugin"), required=True)
    run.add_argument("--allow-execution", action="store_true",
                     help="required safety acknowledgement for local or SSH execution")
    run.add_argument("--stage", action="append", default=[])
    run.add_argument("--timeout", type=float, default=7200.0)
    run.add_argument("--host", help="SSH destination")
    run.add_argument("--remote-project-root", help="absolute project root on SSH host")
    run.add_argument("--runner-plugin", help="explicit hlsgraph.runners.v2 entry-point name")
    run.add_argument("--runner-config", default="{}",
                     help="JSON object passed as keyword arguments to the selected runner plugin")
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
    export.add_argument("--snapshot-id", action="append", default=[],
                        help="snapshot to include; repeat for a multi-snapshot dataset")
    export.add_argument("--feature-evidence-predicate", action="append", default=[],
                        help="opt in one deterministic feature-evidence predicate")
    export.add_argument("--entity-correspondence-kind", action="append", default=[],
                        help="opt in one explicit entity-correspondence kind")
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
