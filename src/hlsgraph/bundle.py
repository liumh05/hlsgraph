"""Local-first GraphBundle with private-source-safe artifact references."""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from .manifest import (
    ManifestError,
    collect_artifacts,
    hydrate_compilation_database,
    load_manifest,
    make_snapshot,
    parse_manifest_text,
    project_path,
    write_internal_manifest,
)
from .model import (
    AccessPolicy,
    ArtifactRef,
    DesignSnapshot,
    ProjectManifest,
    hash_artifact_bytes,
    json_ready,
    stable_hash,
)
from .store import LedgerStore
from .version import BUNDLE_VERSION
from .version import SCHEMA_VERSION


class BundleError(RuntimeError):
    pass


_LEGACY_PUBLIC_VERSIONS = frozenset({"0.1.0", "0.2.0"})


def _atomic_write_text(path: Path, text: str) -> None:
    """Publish one metadata file atomically without silently editing on open."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            descriptor = -1
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _migrated_manifest_source(path: Path, from_version: str) -> str:
    """Return a source manifest upgraded only at its schema marker."""
    text = path.read_text(encoding="utf-8")
    if path.suffix.casefold() == ".json":
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise BundleError(f"manifest source is invalid JSON: {exc}") from exc
        current = value.get("schema_version")
        if current not in {from_version, SCHEMA_VERSION}:
            raise BundleError(
                f"manifest source schema {current!r} cannot be migrated to {SCHEMA_VERSION!r}"
            )
        value["schema_version"] = SCHEMA_VERSION
        migrated = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        parse_manifest_text(migrated, format="json")
        return migrated
    if path.suffix.casefold() != ".toml":
        raise BundleError("manifest source must be .toml or .json")
    pattern = re.compile(
        r'(?m)^(?P<prefix>\s*schema_version\s*=\s*)'
        + r'(?P<quote>[\"\'])'
        + re.escape(from_version)
        + r'(?P=quote)(?P<suffix>\s*(?:#.*)?)$'
    )
    migrated, replacements = pattern.subn(
        lambda match: (
            f"{match.group('prefix')}{match.group('quote')}{SCHEMA_VERSION}"
            f"{match.group('quote')}{match.group('suffix')}"
        ),
        text,
    )
    if replacements == 0:
        # A partially completed migration may already have published the
        # external manifest while the bundle or ledger still needs work.
        current = re.compile(
            r'(?m)^\s*schema_version\s*=\s*[\"\']'
            + re.escape(SCHEMA_VERSION)
            + r'[\"\']\s*(?:#.*)?$'
        )
        if not current.search(text):
            raise BundleError(
                "manifest source does not contain the expected explicit schema marker"
            )
        migrated = text
    elif replacements != 1:
        raise BundleError("manifest source contains multiple schema markers")
    parse_manifest_text(migrated, format="toml")
    return migrated


@contextmanager
def _bundle_migration_lock(bundle_root: Path):
    """Serialize an explicit metadata/ledger migration with indexing and runs."""
    path = bundle_root / "execution.lock"
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise BundleError(
            "another stage execution or migration owns .hlsgraph/execution.lock"
        ) from exc
    try:
        os.write(descriptor, json.dumps({"pid": os.getpid(), "operation": "migration"},
                                        sort_keys=True).encode("ascii") + b"\n")
        os.close(descriptor)
        descriptor = -1
        yield
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        path.unlink(missing_ok=True)


def _require_schema_version(value: object, *, subject: str) -> None:
    if value != SCHEMA_VERSION:
        raise BundleError(
            f"{subject} schema {value!r} is not supported by this build "
            f"({SCHEMA_VERSION!r}); run an explicit migration"
        )


class GraphBundle:
    DIRECTORY = ".hlsgraph"

    def __init__(self, project_root: str | Path, manifest: ProjectManifest,
                 manifest_source: str | Path | None = None):
        self.project_root = Path(project_root).resolve()
        self.root = self.project_root / self.DIRECTORY
        self.manifest = manifest
        self.manifest_source = Path(manifest_source).resolve() if manifest_source else None
        self.store = LedgerStore(self.root / "graph.db")

    @classmethod
    def create(cls, project_root: str | Path, manifest: ProjectManifest, *, force: bool = False,
               manifest_source: str | Path | None = None) -> "GraphBundle":
        _require_schema_version(manifest.schema_version, subject="manifest")
        bundle = cls(project_root, hydrate_compilation_database(manifest, project_root),
                     manifest_source=manifest_source)
        bundle.root.mkdir(parents=True, exist_ok=True)
        internal = bundle.root / "manifest.json"
        if internal.exists() and not force:
            previous = json.loads(internal.read_text(encoding="utf-8"))
            if previous != json_ready(bundle.manifest):
                raise BundleError("bundle already exists with a different manifest; use force explicitly")
        (bundle.root / "artifacts").mkdir(exist_ok=True)
        (bundle.root / "exports").mkdir(exist_ok=True)
        write_internal_manifest(internal, bundle.manifest)
        source_relative = None
        if bundle.manifest_source:
            try:
                source_relative = bundle.manifest_source.relative_to(bundle.project_root).as_posix()
            except ValueError as exc:
                raise BundleError("manifest source must be inside the project root") from exc
        (bundle.root / "bundle.json").write_text(json.dumps({
            "bundle_version": BUNDLE_VERSION,
            "schema_version": bundle.manifest.schema_version,
            "project_id": bundle.manifest.project_id,
            "manifest_source": source_relative,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        bundle.store.initialize()
        bundle.store.save_project(bundle.manifest)
        # Packaged rules are versioned guidance in their own table, never design
        # observations.  Loading them into the local ledger makes SDK/REST/MCP
        # consumers agree without redistributing source documents.
        from .knowledge import KnowledgeCatalog
        catalog = KnowledgeCatalog.builtin()
        # Unreviewed citation-only packs remain useful lexical metadata, but a
        # pack carrying executable bindings is not installed until its complete
        # review attestation is present.  Explicit ``knowledge sync`` applies
        # the same rule and reports the rejected pack instead of silently
        # selecting it.
        KnowledgeCatalog([
            pack for pack in catalog.packs
            if not pack.bindings or pack.review_ready
        ]).install(bundle.store)
        return bundle

    @classmethod
    def _migration_state(cls, project_root: str | Path) -> dict[str, Any]:
        root = Path(project_root).resolve()
        bundle_root = root / cls.DIRECTORY
        internal = bundle_root / "manifest.json"
        metadata_path = bundle_root / "bundle.json"
        database = bundle_root / "graph.db"
        if not internal.is_file() or not metadata_path.is_file() or not database.is_file():
            raise BundleError(f"no complete HLSGraph bundle at {root}")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            manifest = json.loads(internal.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BundleError(f"bundle migration metadata is invalid: {exc}") from exc
        if metadata.get("project_id") != manifest.get("project_id"):
            raise BundleError("bundle and manifest project identities disagree")
        allowed_schema_versions = set(_LEGACY_PUBLIC_VERSIONS) | {SCHEMA_VERSION}
        if metadata.get("schema_version") not in allowed_schema_versions:
            raise BundleError(
                f"bundle schema {metadata.get('schema_version')!r} has no explicit "
                f"migration to {SCHEMA_VERSION!r}"
            )
        if manifest.get("schema_version") not in allowed_schema_versions:
            raise BundleError(
                f"manifest schema {manifest.get('schema_version')!r} has no explicit "
                f"migration to {SCHEMA_VERSION!r}"
            )
        if metadata.get("bundle_version") not in (
            set(_LEGACY_PUBLIC_VERSIONS) | {BUNDLE_VERSION}
        ):
            raise BundleError(
                f"bundle version {metadata.get('bundle_version')!r} has no explicit "
                f"migration to {BUNDLE_VERSION!r}"
            )
        source_path = None
        source_relative = metadata.get("manifest_source")
        if source_relative:
            source_path = project_path(root, str(source_relative))
        elif (root / "hlsgraph.toml").is_file():
            source_path = root / "hlsgraph.toml"
        return {
            "project_root": root,
            "bundle_root": bundle_root,
            "internal_path": internal,
            "metadata_path": metadata_path,
            "database_path": database,
            "metadata": metadata,
            "manifest": manifest,
            "source_path": source_path,
        }

    @classmethod
    def migration_plan(cls, project_root: str | Path,
                       to_version: str = SCHEMA_VERSION) -> list[dict[str, str]]:
        """Inspect an old bundle without opening or mutating it."""
        if to_version != SCHEMA_VERSION or to_version != BUNDLE_VERSION:
            raise BundleError(
                f"this build can migrate only to the unified version {SCHEMA_VERSION!r}"
            )
        state = cls._migration_state(project_root)
        steps: list[dict[str, str]] = []
        metadata = state["metadata"]
        manifest = state["manifest"]
        if (metadata.get("bundle_version") != BUNDLE_VERSION
                or metadata.get("schema_version") != SCHEMA_VERSION
                or manifest.get("schema_version") != SCHEMA_VERSION):
            steps.append({
                "scope": "bundle",
                "from_version": str(metadata.get("bundle_version")),
                "to_version": BUNDLE_VERSION,
                "description": (
                    "upgrade bundle and manifest schema markers after the registered "
                    "ledger migration"
                ),
            })
        source_path = state["source_path"]
        if source_path is not None:
            source_text = source_path.read_text(encoding="utf-8")
            source_format = source_path.suffix.casefold().lstrip(".")
            source_manifest = parse_manifest_text(source_text, format=source_format)
            if source_manifest.schema_version != SCHEMA_VERSION:
                if source_manifest.schema_version not in _LEGACY_PUBLIC_VERSIONS:
                    raise BundleError(
                        f"manifest source schema {source_manifest.schema_version!r} has no "
                        f"explicit migration to {SCHEMA_VERSION!r}"
                    )
                steps.append({
                    "scope": "manifest_source",
                    "from_version": source_manifest.schema_version,
                    "to_version": SCHEMA_VERSION,
                    "description": "upgrade the external manifest schema marker",
                })
        try:
            ledger_steps = LedgerStore(state["database_path"]).migration_plan(SCHEMA_VERSION)
        except Exception as exc:
            raise BundleError(f"cannot plan ledger migration: {exc}") from exc
        steps.extend({"scope": "ledger", **step} for step in ledger_steps)
        return steps

    @classmethod
    def migrate(cls, project_root: str | Path,
                to_version: str = SCHEMA_VERSION) -> list[dict[str, str]]:
        """Explicitly migrate bundle metadata, source manifest, and SQLite ledger.

        Opening a bundle never calls this method.  Each file is atomically
        replaced, and a partially completed multi-file migration can be safely
        resumed because every step accepts either the old or target marker.
        """
        if to_version != SCHEMA_VERSION or to_version != BUNDLE_VERSION:
            raise BundleError(
                f"this build can migrate only to the unified version {SCHEMA_VERSION!r}"
            )
        initial = cls._migration_state(project_root)
        with _bundle_migration_lock(initial["bundle_root"]):
            state = cls._migration_state(project_root)
            planned = cls.migration_plan(project_root, to_version)
            source_path: Path | None = state["source_path"]
            source_text = None
            if source_path is not None:
                source_format = source_path.suffix.casefold().lstrip(".")
                source_version = parse_manifest_text(
                    source_path.read_text(encoding="utf-8"), format=source_format,
                ).schema_version
                source_text = _migrated_manifest_source(
                    source_path, source_version,
                )

            store = LedgerStore(state["database_path"])
            try:
                store.migrate(SCHEMA_VERSION)
            except Exception as exc:
                raise BundleError(f"ledger migration failed: {exc}") from exc

            manifest_value = dict(state["manifest"])
            manifest_value["schema_version"] = SCHEMA_VERSION
            try:
                manifest = ProjectManifest.from_dict(manifest_value)
            except (KeyError, TypeError, ValueError) as exc:
                raise BundleError(f"migrated internal manifest is invalid: {exc}") from exc
            # The projects row is the mutable current manifest. Historical
            # snapshot_manifests remain byte-for-byte 0.1 evidence.
            store.save_project(manifest)

            metadata_value = dict(state["metadata"])
            metadata_value["bundle_version"] = BUNDLE_VERSION
            metadata_value["schema_version"] = SCHEMA_VERSION
            if source_path is not None and source_text is not None:
                _atomic_write_text(source_path, source_text)
            _atomic_write_text(
                state["internal_path"],
                json.dumps(manifest_value, ensure_ascii=False, indent=2,
                           sort_keys=True) + "\n",
            )
            _atomic_write_text(
                state["metadata_path"],
                json.dumps(metadata_value, ensure_ascii=False, indent=2,
                           sort_keys=True) + "\n",
            )
            return planned

    @classmethod
    def open(cls, project_root: str | Path) -> "GraphBundle":
        project_root = Path(project_root).resolve()
        internal = project_root / cls.DIRECTORY / "manifest.json"
        if not internal.is_file():
            raise BundleError(f"no HLSGraph bundle at {project_root}")
        bundle_metadata = project_root / cls.DIRECTORY / "bundle.json"
        if not bundle_metadata.is_file():
            raise BundleError("bundle metadata is missing; run an explicit migration or re-index")
        source = None
        try:
            value = json.loads(bundle_metadata.read_text(encoding="utf-8"))
            internal_value = json.loads(internal.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BundleError(f"bundle metadata is invalid: {exc}") from exc
        if value.get("bundle_version") != BUNDLE_VERSION:
            raise BundleError(
                f"bundle version {value.get('bundle_version')!r} is not supported; "
                "run an explicit migration"
            )
        _require_schema_version(value.get("schema_version"), subject="bundle")
        _require_schema_version(internal_value.get("schema_version"), subject="manifest")
        if value.get("schema_version") != internal_value.get("schema_version"):
            raise BundleError("bundle and manifest schema versions disagree")
        if value.get("project_id") != internal_value.get("project_id"):
            raise BundleError("bundle and manifest project identities disagree")
        if value.get("manifest_source"):
            source = project_path(project_root, str(value["manifest_source"]))
        if source is None and (project_root / "hlsgraph.toml").is_file():
            source = project_root / "hlsgraph.toml"
        try:
            manifest = ProjectManifest.from_dict(internal_value)
        except (KeyError, TypeError, ValueError) as exc:
            raise BundleError(f"internal manifest is invalid: {exc}") from exc
        return cls(project_root, manifest, manifest_source=source)

    @classmethod
    def from_manifest(cls, path: str | Path, *, force: bool = False) -> "GraphBundle":
        path = Path(path).resolve()
        return cls.create(path.parent, load_manifest(path), force=force,
                          manifest_source=path)

    def source_manifest(self) -> ProjectManifest | None:
        if not self.manifest_source or not self.manifest_source.is_file():
            return None
        manifest = load_manifest(self.manifest_source)
        _require_schema_version(manifest.schema_version, subject="manifest")
        return hydrate_compilation_database(manifest, self.project_root)

    @contextmanager
    def execution_lock(self):
        """Serialize stage execution and output attribution across processes."""
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / "execution.lock"
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise BundleError(
                "another stage execution owns .hlsgraph/execution.lock; "
                "inspect the owner before removing a stale lock"
            ) from exc
        try:
            payload = json.dumps({"pid": os.getpid()}, sort_keys=True).encode("ascii") + b"\n"
            os.write(descriptor, payload)
            os.close(descriptor)
            descriptor = -1
            yield
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            path.unlink(missing_ok=True)

    def refresh_manifest(self) -> bool:
        """Refresh the internal copy during an explicit write/index operation."""
        current = self.source_manifest()
        if current is None or json_ready(current) == json_ready(self.manifest):
            return False
        self.manifest = current
        write_internal_manifest(self.root / "manifest.json", self.manifest)
        self.store.save_project(self.manifest)
        return True

    def snapshot(self, *, parent_snapshot_id: str | None = None,
                 action_id: str | None = None, extraction_hash: str = "") -> DesignSnapshot:
        artifacts = collect_artifacts(self.manifest, self.project_root)
        snapshot = make_snapshot(self.manifest, artifacts, parent_snapshot_id=parent_snapshot_id,
                                 action_id=action_id, extraction_hash=extraction_hash)
        active = self.latest_snapshot()
        if parent_snapshot_id is None and active is not None and active.id != snapshot.id:
            # Lineage is recorded but is not part of design identity. The same
            # source/build/target/tool/extraction inputs always recover the same
            # snapshot ID, even when a parser profile is reactivated later.
            snapshot = make_snapshot(
                self.manifest, artifacts, parent_snapshot_id=active.id,
                action_id=action_id, extraction_hash=extraction_hash,
            )
        self.store.save_project(self.manifest)
        self.store.save_snapshot(snapshot, artifacts)
        return self.store.snapshot(snapshot.id)

    def latest_snapshot(self) -> DesignSnapshot | None:
        return self.store.latest_snapshot(self.manifest.project_id)

    def is_stale(self, snapshot: DesignSnapshot | None = None) -> bool:
        snapshot = snapshot or self.latest_snapshot()
        if snapshot is None:
            return True
        try:
            current_manifest = self.source_manifest() or self.manifest
            current = make_snapshot(current_manifest,
                                    collect_artifacts(current_manifest, self.project_root),
                                    parent_snapshot_id=snapshot.parent_snapshot_id,
                                    action_id=snapshot.action_id,
                                    extraction_hash=snapshot.extraction_hash)
        except ManifestError:
            return True
        return current.id != snapshot.id

    def source_snippet(self, artifact_id: str, start_line: int, end_line: int,
                       *, snapshot_id: str | None = None, allow_private: bool = False,
                       max_lines: int = 200) -> str:
        snapshot = self.store.snapshot(snapshot_id) if snapshot_id else self.latest_snapshot()
        if snapshot is None:
            raise BundleError("bundle has no successful active snapshot")
        artifacts = {item.id: item for item in self.store.artifacts(snapshot.id)}
        artifact = artifacts.get(artifact_id)
        if artifact is None:
            raise KeyError(artifact_id)
        if artifact.access == AccessPolicy.PRIVATE and not allow_private:
            raise PermissionError("private source requires explicit authorization")
        if start_line < 1 or end_line < start_line or end_line - start_line + 1 > max_lines:
            raise ValueError(f"line range must contain 1..{max_lines} lines")
        path = project_path(self.project_root, artifact.uri)
        data = path.read_bytes()
        actual_hash = hash_artifact_bytes(data)
        if actual_hash != artifact.sha256:
            raise BundleError(
                "artifact content no longer matches the selected snapshot; "
                "re-index or read a retained content-addressed artifact"
            )
        lines = data.decode("utf-8", errors="replace").splitlines()
        return "\n".join(lines[start_line - 1:end_line])

    def prepare_managed_artifact(
        self, source: str | Path, *, kind: str, role: str,
        access: AccessPolicy = AccessPolicy.PROJECT,
        producer_run_id: str | None = None,
        license: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[ArtifactRef, Path, bool]:
        """Copy bytes into the CAS without writing ledger rows.

        Callers use the returned object with ``commit_run_result`` so the run,
        producer link, parsed evidence, and artifact reference become visible in
        one SQLite transaction. ``created`` reports whether this invocation
        published the CAS entry; callers must not unlink it on ledger failure,
        because another process may already reference the same immutable bytes.
        Unreferenced entries are harmless orphans for an explicit future GC.
        """
        source = Path(source).resolve()
        if not source.is_file():
            raise BundleError(f"managed artifact does not exist: {source}")
        data = source.read_bytes()
        digest = hash_artifact_bytes(data)
        target_dir = self.root / "artifacts" / digest
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / source.name
        created = False
        if not target.exists():
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{source.name}.", suffix=".tmp", dir=target_dir,
            )
            temporary = Path(temporary_name)
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    descriptor = -1
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
                try:
                    # A hard-link publish is atomic and refuses to replace a
                    # concurrently-created CAS entry on both Windows and POSIX.
                    os.link(temporary, target)
                    created = True
                except FileExistsError:
                    if target.read_bytes() != data:
                        raise BundleError(
                            f"content-addressed artifact collision: {target}"
                        )
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
                temporary.unlink(missing_ok=True)
        elif target.read_bytes() != data:
            raise BundleError(f"content-addressed artifact collision: {target}")
        artifact = ArtifactRef(
            kind=kind, uri=target.relative_to(self.project_root).as_posix(),
            sha256=digest, size=len(data), role=role, access=access,
            retention="managed", producer_run_id=producer_run_id,
            license=license, metadata=dict(metadata or {}),
        )
        return artifact, target, created

    def add_managed_artifact(self, source: str | Path, *, kind: str, role: str,
                             access: AccessPolicy = AccessPolicy.PROJECT,
                             producer_run_id: str | None = None,
                             snapshot_id: str | None = None,
                             license: str | None = None,
                             metadata: dict[str, Any] | None = None) -> ArtifactRef:
        if producer_run_id is not None:
            raise BundleError(
                "run-produced artifacts require prepare_managed_artifact followed by "
                "LedgerStore.commit_run_result so producer and output links commit atomically"
            )
        latest = self.latest_snapshot() if snapshot_id is None else None
        selected = snapshot_id or (latest.id if latest else None)
        if selected is None:
            raise BundleError("create a snapshot before attaching a managed artifact")
        try:
            self.store.snapshot(selected)
        except KeyError as exc:
            raise BundleError(f"snapshot does not exist: {selected}") from exc
        artifact, _target, _created = self.prepare_managed_artifact(
            source, kind=kind, role=role, access=access,
            producer_run_id=None, license=license, metadata=metadata,
        )
        # Never delete a published CAS path on a ledger error.  A concurrent
        # bundle/process can already have committed a reference to the same
        # content-addressed file; an orphan is safer than broken provenance.
        self.store.add_artifact(selected, artifact)
        return artifact

    def status(self, snapshot_id: str | None = None) -> dict[str, Any]:
        active = self.latest_snapshot()
        snapshot = self.store.snapshot(snapshot_id) if snapshot_id else active
        candidate = self.store.latest_candidate(self.manifest.project_id)
        diagnostics = self.store.active_diagnostics(snapshot.id) if snapshot else []
        historical_manifest = self.store.snapshot_manifest(snapshot.id) if snapshot else self.manifest
        return {
            "bundle_version": BUNDLE_VERSION,
            "schema_version": self.manifest.schema_version,
            "project_id": self.manifest.project_id,
            "manifest_source": (self.manifest_source.relative_to(self.project_root).as_posix()
                                if self.manifest_source else None),
            "snapshot_id": snapshot.id if snapshot else None,
            "active_snapshot_id": active.id if active else None,
            "latest_candidate_snapshot_id": candidate.id if candidate else None,
            "graph_available": self.store.has_graph(snapshot.id) if snapshot else False,
            "stale": self.is_stale(snapshot) if snapshot else True,
            "artifacts": len(self.store.artifacts(snapshot.id)) if snapshot else 0,
            "diagnostics": len(diagnostics),
            "capabilities": {
                "private_source_embedded": False,
                "runner": ["local", "ssh", "fake"],
                "query": ["sqlite", "fts5"],
            },
            "bundle_hash": stable_hash({
                "manifest": historical_manifest.identity_payload(),
                "snapshot": snapshot.identity_payload() if snapshot else None,
            }),
        }
