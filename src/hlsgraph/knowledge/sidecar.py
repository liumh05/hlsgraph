"""Private, rebuildable full-text sidecar for user-owned knowledge documents.

The canonical ledger stores none of these chunks.  Building/synchronizing this
index is always an explicit SDK/CLI action and the safe search default returns
metadata only.
"""
from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import os
import re
import sqlite3
import stat
import tempfile
from contextlib import closing
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from queue import Empty
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

from hlsgraph.model import LocalKnowledgeIndexManifest, json_ready, stable_hash

from .core import (
    KnowledgePackError,
    LocalDocumentMetadata,
    _is_link_or_reparse,
    _read_stable_local_file,
)


SIDECAR_SCHEMA_VERSION = "1.0"
SIDECAR_RELATIVE_ROOT = Path(".hlsgraph/private/knowledge")
DEFAULT_MAX_DOCUMENT_BYTES = 32 * 1024 * 1024
DEFAULT_MAX_DOCUMENTS = 128
DEFAULT_MAX_CHUNKS = 20_000
DEFAULT_CHUNK_CHARS = 2_000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_PARSER_TIMEOUT_S = 10.0
DEFAULT_MAX_PARSED_CHARS = 8 * 1024 * 1024
DEFAULT_MAX_MANIFEST_BYTES = 1024 * 1024


class _VisibleHtml(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._suppressed = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.casefold() in {"script", "style", "svg"}:
            self._suppressed += 1
        elif not self._suppressed and tag.casefold() in {
            "p", "br", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6",
        }:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() in {"script", "style", "svg"} and self._suppressed:
            self._suppressed -= 1
        elif not self._suppressed and tag.casefold() in {
            "p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6",
        }:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._suppressed:
            self.parts.append(data)


def _parser_worker(
    parser: Any,
    data: bytes,
    metadata: dict[str, Any],
    max_chars: int,
    output: Any,
) -> None:
    """Isolated parser process target; never returns document bytes on failure."""
    try:
        value = parser.parse(data, metadata)
        if value is None:
            output.put(("metadata_only", None))
        elif not isinstance(value, str):
            output.put(("error", "parser returned a non-text value"))
        elif "\x00" in value:
            output.put(("error", "parser returned text containing NUL"))
        elif len(value) > max_chars:
            output.put(("error", "parser output exceeded the declared character limit"))
        else:
            output.put(("ok", value))
    except BaseException as exc:  # isolated untrusted plugin boundary
        output.put(("error", f"parser raised {type(exc).__name__}"))


@dataclass(frozen=True, slots=True)
class LocalKnowledgeHit:
    chunk_id: str
    document_id: str
    document_version: str
    title: str | None
    heading: str | None
    chunk_sha256: str
    score: float
    channel: str
    excerpt: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return json_ready(self)


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            descriptor = -1
            stream.write(json.dumps(
                value, ensure_ascii=False, indent=2, sort_keys=True,
                allow_nan=False,
            ) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _ensure_private_root(project_root: str | Path) -> tuple[Path, Path]:
    project = Path(project_root).resolve()
    if not project.is_dir():
        raise KnowledgePackError(f"project root is not a directory: {project}")
    current = project
    private_subtree = False
    for component in SIDECAR_RELATIVE_ROOT.parts:
        current = current / component
        private_subtree = private_subtree or component == "private"
        if current.exists() and _is_link_or_reparse(current):
            raise KnowledgePackError(
                f"private knowledge path cannot traverse a link/reparse point: {current}"
            )
        current.mkdir(mode=0o700 if private_subtree else 0o755, exist_ok=True)
        if private_subtree and os.name != "nt":
            os.chmod(current, 0o700)
    root = current.resolve()
    try:
        root.relative_to(project)
    except ValueError as exc:
        raise KnowledgePackError("private knowledge sidecar escaped the project root") from exc
    return project, root


def _file_uri_path(uri: str) -> Path | None:
    parsed = urlparse(uri)
    if parsed.scheme.casefold() != "file" or parsed.netloc not in ("", "localhost"):
        return None
    value = url2pathname(unquote(parsed.path))
    if os.name == "nt" and re.match(r"^[/\\][A-Za-z]:", value):
        value = value[1:]
    return Path(value)


def _has_link_ancestor(path: Path) -> bool:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if current.exists() and _is_link_or_reparse(current):
            return True
    return False


def _document_bytes(
    metadata: LocalDocumentMetadata, *, max_bytes: int,
) -> tuple[bytes, Path] | None:
    path = _file_uri_path(metadata.uri)
    if path is None:
        return None
    if _has_link_ancestor(path):
        raise KnowledgePackError(
            f"local knowledge document path traverses a link/reparse point: {path}"
        )
    data, info = _read_stable_local_file(path, max_bytes=max_bytes)
    digest = hashlib.sha256(data).hexdigest()
    if digest != metadata.sha256 or len(data) != metadata.size:
        raise KnowledgePackError(
            f"local knowledge document hash/size changed; re-index metadata first: {path}"
        )
    if info.st_mtime_ns != metadata.modified_ns:
        raise KnowledgePackError(
            f"local knowledge document timestamp changed; re-index metadata first: {path}"
        )
    return data, path


def _parse_text(metadata: LocalDocumentMetadata, data: bytes, path: Path) -> str | None:
    extension = path.suffix.casefold()
    text_like = (
        (metadata.media_type or "").startswith("text/")
        or extension in {".txt", ".md", ".markdown", ".rst", ".html", ".htm"}
    )
    if not text_like:
        return None
    try:
        decoded = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise KnowledgePackError(
            f"local knowledge text must be UTF-8 or use an explicit parser plugin: {path}"
        ) from exc
    if extension in {".html", ".htm"} or metadata.media_type == "text/html":
        parser = _VisibleHtml()
        parser.feed(decoded)
        decoded = "".join(parser.parts)
    return decoded.replace("\r\n", "\n").replace("\r", "\n")


def _validate_parser(parser: Any | None) -> frozenset[str]:
    if parser is None:
        return frozenset()
    for attribute in ("name", "version", "fingerprint", "capabilities", "parse"):
        if not hasattr(parser, attribute):
            raise KnowledgePackError(f"local knowledge parser is missing {attribute!r}")
    capabilities = parser.capabilities()
    if (not isinstance(capabilities, Mapping)
            or capabilities.get("protocol_version") != "hlsgraph.knowledge_parser.v1"
            or capabilities.get("local_only") is not True
            or capabilities.get("network_access") is not False):
        raise KnowledgePackError(
            "knowledge parser must implement the local-only, no-network v1 contract"
        )
    media_types = capabilities.get("media_types")
    if (not isinstance(media_types, (list, tuple)) or not media_types
            or any(not isinstance(item, str) or not item.strip()
                   for item in media_types)):
        raise KnowledgePackError(
            "knowledge parser must declare non-empty media_types"
        )
    if (not isinstance(parser.name, str) or not parser.name.strip()
            or not isinstance(parser.version, str) or not parser.version.strip()
            or not isinstance(parser.fingerprint, str)
            or not re.fullmatch(r"[0-9a-f]{64}", parser.fingerprint)):
        raise KnowledgePackError("knowledge parser has invalid immutable identity")
    return frozenset(str(item).casefold() for item in media_types)


def _parse_with_plugin(
    parser: Any,
    data: bytes,
    metadata: LocalDocumentMetadata,
    *,
    timeout_s: float,
    max_chars: int,
) -> str | None:
    if (not isinstance(timeout_s, (int, float)) or isinstance(timeout_s, bool)
            or not math.isfinite(float(timeout_s)) or not 0.1 <= float(timeout_s) <= 60.0):
        raise KnowledgePackError("knowledge parser timeout must be in 0.1..60 seconds")
    if (not isinstance(max_chars, int) or isinstance(max_chars, bool)
            or not 1 <= max_chars <= DEFAULT_MAX_PARSED_CHARS):
        raise KnowledgePackError(
            f"knowledge parser max_chars must be in 1..{DEFAULT_MAX_PARSED_CHARS}"
        )
    # The plugin sees document bytes and bounded metadata, but never the host
    # path/URI.  A fresh spawn process makes timeout/exit handling deterministic
    # on Windows and POSIX and prevents a failed parser from publishing a DB.
    context = multiprocessing.get_context("spawn")
    output = context.Queue(maxsize=1)
    process = context.Process(
        target=_parser_worker,
        args=(parser, data, {
            "document_id": metadata.document_id,
            "document_version": metadata.document_version,
            "title": metadata.title,
            "media_type": metadata.media_type,
            "sha256": metadata.sha256,
            "size": metadata.size,
        }, max_chars, output),
        daemon=True,
    )
    try:
        try:
            process.start()
        except Exception as exc:
            raise KnowledgePackError(
                "knowledge parser could not start in the isolated process"
            ) from exc
        process.join(float(timeout_s))
        if process.is_alive():
            process.terminate()
            process.join(2.0)
            if process.is_alive() and hasattr(process, "kill"):
                process.kill()
                process.join(2.0)
            raise KnowledgePackError("knowledge parser timed out")
        try:
            status, value = output.get(timeout=0.5)
        except Empty as exc:
            raise KnowledgePackError(
                "knowledge parser exited without a bounded result"
            ) from exc
        if status == "metadata_only":
            return None
        if status != "ok":
            raise KnowledgePackError(str(value))
        return str(value).replace("\r\n", "\n").replace("\r", "\n")
    finally:
        if process.is_alive():
            process.terminate()
            process.join(2.0)
        output.cancel_join_thread()
        output.close()


def _sections(text: str) -> list[tuple[str | None, str]]:
    result: list[tuple[str | None, str]] = []
    heading: str | None = None
    lines: list[str] = []
    for line in text.splitlines(keepends=True):
        match = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", line.rstrip("\n"))
        if match:
            if "".join(lines).strip():
                result.append((heading, "".join(lines)))
            heading = match.group(1).strip()[:512]
            lines = []
        else:
            lines.append(line)
    if "".join(lines).strip():
        result.append((heading, "".join(lines)))
    return result or [(None, text)]


def _chunk_text(
    text: str, *, chunk_chars: int, overlap: int,
) -> list[tuple[str | None, int, int, str]]:
    if chunk_chars < 256 or overlap < 0 or overlap >= chunk_chars:
        raise KnowledgePackError("invalid local knowledge chunk size/overlap")
    chunks: list[tuple[str | None, int, int, str]] = []
    global_offset = 0
    for heading, section in _sections(text):
        start = 0
        while start < len(section):
            limit = min(start + chunk_chars, len(section))
            if limit < len(section):
                boundary = max(
                    section.rfind("\n\n", start + chunk_chars // 2, limit),
                    section.rfind("\n", start + chunk_chars // 2, limit),
                    section.rfind(" ", start + chunk_chars // 2, limit),
                )
                if boundary > start:
                    limit = boundary + 1
            value = section[start:limit].strip()
            if value:
                chunks.append((heading, global_offset + start,
                               global_offset + limit, value))
            if limit >= len(section):
                break
            start = max(start + 1, limit - overlap)
        global_offset += len(section)
    return chunks


def _bounded_excerpt(value: str, *, max_lines: int = 80, max_chars: int = 4_000) -> str:
    lines = value.splitlines(keepends=True)[:max_lines]
    return "".join(lines)[:max_chars]


def _validate_embedder(embedder: Any | None) -> None:
    if embedder is None:
        return
    for attribute in ("name", "version", "fingerprint", "capabilities", "embed"):
        if not hasattr(embedder, attribute):
            raise KnowledgePackError(f"local embedder is missing {attribute!r}")
    capabilities = embedder.capabilities()
    if (not isinstance(capabilities, Mapping)
            or capabilities.get("protocol_version") != "hlsgraph.embedder.v1"
            or capabilities.get("local_only") is not True
            or capabilities.get("network_access") is not False):
        raise KnowledgePackError("embedder must implement the local-only v1 contract")
    if not re.fullmatch(r"[0-9a-f]{64}", str(embedder.fingerprint)):
        raise KnowledgePackError("embedder fingerprint must be lowercase SHA-256")


def _embed_chunks(connection: sqlite3.Connection, chunks: list[tuple[str, str]], embedder: Any) -> None:
    dimension: int | None = None
    for offset in range(0, len(chunks), 64):
        batch = chunks[offset:offset + 64]
        try:
            vectors = embedder.embed([text for _chunk_id, text in batch])
        except Exception as exc:
            raise KnowledgePackError(f"local embedder failed: {exc}") from exc
        if not isinstance(vectors, Sequence) or len(vectors) != len(batch):
            raise KnowledgePackError("local embedder returned the wrong vector count")
        for (chunk_id, _text), vector in zip(batch, vectors):
            if (not isinstance(vector, Sequence) or isinstance(vector, (str, bytes))
                    or not vector or len(vector) > 65_536):
                raise KnowledgePackError("local embedder returned an invalid vector")
            values: list[float] = []
            for value in vector:
                if (not isinstance(value, (int, float)) or isinstance(value, bool)
                        or not math.isfinite(float(value))):
                    raise KnowledgePackError("local embedder returned a non-finite vector")
                values.append(float(value))
            if dimension is None:
                dimension = len(values)
            if len(values) != dimension:
                raise KnowledgePackError("local embedder returned inconsistent dimensions")
            connection.execute(
                "INSERT INTO vectors(chunk_id,vector_json) VALUES(?,?)",
                (chunk_id, json.dumps(values, separators=(",", ":"), allow_nan=False)),
            )


class LocalKnowledgeSidecar:
    """Explicit builder and read-only accessor for one project's private index."""

    def __init__(self, project_root: str | Path):
        self.project_root = Path(project_root).resolve()
        self.root = self.project_root / SIDECAR_RELATIVE_ROOT
        self.database_path = self.root / "chunks.sqlite"
        self.manifest_path = self.root / "manifest.json"

    def prepare(self) -> Path:
        """Create and permission the private sidecar root for an explicit write."""
        _project, root = _ensure_private_root(self.project_root)
        self.root = root
        self.database_path = root / "chunks.sqlite"
        self.manifest_path = root / "manifest.json"
        return root

    def build(
        self,
        project_id: str,
        documents: Iterable[LocalDocumentMetadata],
        *,
        parser: Any | None = None,
        parser_timeout_s: float = DEFAULT_PARSER_TIMEOUT_S,
        max_parsed_chars: int = DEFAULT_MAX_PARSED_CHARS,
        embedder: Any | None = None,
        max_document_bytes: int = DEFAULT_MAX_DOCUMENT_BYTES,
        chunk_chars: int = DEFAULT_CHUNK_CHARS,
        overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> LocalKnowledgeIndexManifest:
        """Explicitly replace the rebuildable sidecar from a complete document set."""
        root = self.prepare()
        entries = sorted(
            list(documents), key=lambda item: (item.document_id, item.document_version),
        )
        if len(entries) > DEFAULT_MAX_DOCUMENTS:
            raise KnowledgePackError("too many local knowledge documents")
        if (not isinstance(max_document_bytes, int) or isinstance(max_document_bytes, bool)
                or not 1 <= max_document_bytes <= DEFAULT_MAX_DOCUMENT_BYTES):
            raise KnowledgePackError(
                f"max_document_bytes must be in 1..{DEFAULT_MAX_DOCUMENT_BYTES}"
            )
        if (not isinstance(chunk_chars, int) or isinstance(chunk_chars, bool)
                or not 256 <= chunk_chars <= 8_000
                or not isinstance(overlap, int) or isinstance(overlap, bool)
                or not 0 <= overlap < chunk_chars):
            raise KnowledgePackError(
                "chunk_chars must be in 256..8000 and overlap in 0..chunk_chars-1"
            )
        keys = [f"{item.document_id}@{item.document_version}" for item in entries]
        if len(set(keys)) != len(keys):
            raise KnowledgePackError("local knowledge document versions must be unique")
        parser_media_types = _validate_parser(parser)
        if (not isinstance(max_parsed_chars, int) or isinstance(max_parsed_chars, bool)
                or not 1 <= max_parsed_chars <= DEFAULT_MAX_PARSED_CHARS):
            raise KnowledgePackError(
                f"max_parsed_chars must be in 1..{DEFAULT_MAX_PARSED_CHARS}"
            )
        if parser is not None and (
            not isinstance(parser_timeout_s, (int, float))
            or isinstance(parser_timeout_s, bool)
            or not math.isfinite(float(parser_timeout_s))
            or not 0.1 <= float(parser_timeout_s) <= 60.0
        ):
            raise KnowledgePackError("knowledge parser timeout must be in 0.1..60 seconds")
        _validate_embedder(embedder)

        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".chunks.", suffix=".sqlite.tmp", dir=root,
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        temporary.unlink(missing_ok=True)
        metadata_only: list[str] = []
        all_chunks: list[tuple[str, str]] = []
        fts_enabled = True
        try:
            with closing(sqlite3.connect(temporary)) as connection:
                connection.executescript("""
                PRAGMA journal_mode=DELETE;
                PRAGMA foreign_keys=ON;
                CREATE TABLE sidecar_info (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE documents (
                  document_key TEXT PRIMARY KEY,
                  document_id TEXT NOT NULL,
                  document_version TEXT NOT NULL,
                  uri TEXT NOT NULL,
                  sha256 TEXT NOT NULL,
                  size INTEGER NOT NULL,
                  modified_ns INTEGER NOT NULL,
                  title TEXT,
                  media_type TEXT,
                  official_url TEXT,
                  status TEXT NOT NULL
                );
                CREATE TABLE chunks (
                  id TEXT PRIMARY KEY,
                  document_key TEXT NOT NULL REFERENCES documents(document_key),
                  ordinal INTEGER NOT NULL,
                  heading TEXT,
                  start_char INTEGER NOT NULL,
                  end_char INTEGER NOT NULL,
                  chunk_sha256 TEXT NOT NULL,
                  text TEXT NOT NULL,
                  UNIQUE(document_key,ordinal)
                );
                CREATE TABLE vectors (
                  chunk_id TEXT PRIMARY KEY REFERENCES chunks(id),
                  vector_json TEXT NOT NULL
                );
                """)
                try:
                    connection.execute(
                        "CREATE VIRTUAL TABLE chunks_fts USING fts5("
                        "chunk_id UNINDEXED,document_id,document_version,heading,text)"
                    )
                except sqlite3.OperationalError:
                    fts_enabled = False
                connection.execute(
                    "INSERT INTO sidecar_info(key,value) VALUES('schema_version',?)",
                    (SIDECAR_SCHEMA_VERSION,),
                )
                for metadata, key in zip(entries, keys):
                    source = _document_bytes(metadata, max_bytes=max_document_bytes)
                    text: str | None = None
                    if source is not None:
                        data, path = source
                        text = _parse_text(metadata, data, path)
                        if text is not None and len(text) > max_parsed_chars:
                            raise KnowledgePackError(
                                "built-in parser output exceeded the declared character limit"
                            )
                        if (text is None and parser is not None
                                and (metadata.media_type or "").casefold()
                                in parser_media_types):
                            text = _parse_with_plugin(
                                parser, data, metadata,
                                timeout_s=parser_timeout_s,
                                max_chars=max_parsed_chars,
                            )
                    status = "indexed" if text is not None else "metadata_only"
                    if text is None:
                        metadata_only.append(key)
                    connection.execute(
                        "INSERT INTO documents VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                        (key, metadata.document_id, metadata.document_version,
                         metadata.uri, metadata.sha256, metadata.size,
                         metadata.modified_ns, metadata.title, metadata.media_type,
                         metadata.official_url, status),
                    )
                    if text is None:
                        continue
                    for ordinal, (heading, start, end, value) in enumerate(_chunk_text(
                        text, chunk_chars=chunk_chars, overlap=overlap,
                    )):
                        if len(all_chunks) >= DEFAULT_MAX_CHUNKS:
                            raise KnowledgePackError("local knowledge index exceeds chunk limit")
                        chunk_hash = hashlib.sha256(value.encode("utf-8")).hexdigest()
                        chunk_id = "local_chunk_" + stable_hash({
                            "document": key, "ordinal": ordinal,
                            "heading": heading, "sha256": chunk_hash,
                        })[:24]
                        connection.execute(
                            "INSERT INTO chunks VALUES(?,?,?,?,?,?,?,?)",
                            (chunk_id, key, ordinal, heading, start, end,
                             chunk_hash, value),
                        )
                        if fts_enabled:
                            connection.execute(
                                "INSERT INTO chunks_fts VALUES(?,?,?,?,?)",
                                (chunk_id, metadata.document_id,
                                 metadata.document_version, heading or "", value),
                            )
                        all_chunks.append((chunk_id, value))
                if embedder is not None:
                    _embed_chunks(connection, all_chunks, embedder)
                connection.commit()
            index_hash = hashlib.sha256(temporary.read_bytes()).hexdigest()
            os.replace(temporary, self.database_path)
            manifest = LocalKnowledgeIndexManifest(
                project_id=project_id,
                document_hashes={key: item.sha256 for key, item in zip(keys, entries)},
                chunk_count=len(all_chunks),
                index_sha256=index_hash,
                parser_id=(str(parser.name) if parser is not None
                           else "hlsgraph.local_text"),
                parser_version=(str(parser.version) if parser is not None else "1"),
                parser_fingerprint=(str(parser.fingerprint) if parser is not None
                                    else hashlib.sha256(
                                        b"hlsgraph.local_text:1"
                                    ).hexdigest()),
                chunker_id="hlsgraph.section_window",
                chunker_version="1",
                fts_enabled=fts_enabled,
                embedder_id=str(embedder.name) if embedder is not None else None,
                embedder_version=str(embedder.version) if embedder is not None else None,
                embedder_fingerprint=(str(embedder.fingerprint)
                                      if embedder is not None else None),
                metadata={
                    "metadata_only_documents": metadata_only,
                    "chunk_chars": chunk_chars,
                    "chunk_overlap": overlap,
                    "private_sidecar": True,
                    "builtin_text_parser": "utf8.text-markdown-html.v1",
                    "plugin_media_types": sorted(parser_media_types),
                },
            )
            _atomic_write_json(self.manifest_path, json_ready(manifest))
            return manifest
        finally:
            temporary.unlink(missing_ok=True)

    def sync(self, project_id: str, documents: Iterable[LocalDocumentMetadata],
             **kwargs: Any) -> LocalKnowledgeIndexManifest:
        """Explicitly rebuild from current declared inputs; never runs on open."""
        return self.build(project_id, documents, **kwargs)

    def manifest(self) -> LocalKnowledgeIndexManifest:
        try:
            data, _info = _read_stable_local_file(
                self.manifest_path, max_bytes=DEFAULT_MAX_MANIFEST_BYTES,
            )
            payload = json.loads(data.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError,
                KnowledgePackError) as exc:
            raise KnowledgePackError(f"cannot load local sidecar manifest: {exc}") from exc
        try:
            return LocalKnowledgeIndexManifest.from_dict(payload)
        except (KeyError, TypeError, ValueError) as exc:
            raise KnowledgePackError(f"invalid local sidecar manifest: {exc}") from exc

    def _verified_database_snapshot(
        self,
    ) -> tuple[LocalKnowledgeIndexManifest, bytes]:
        """Return manifest-bound database bytes from one stable file read.

        Query code must never reopen ``database_path`` after this method
        returns.  The immutable bytes are the object whose SHA-256 was checked;
        deserializing those same bytes into an in-memory connection closes the
        verify/open TOCTOU window, including a swap followed by path restore.
        """
        for path in (self.root, self.manifest_path, self.database_path):
            if _is_link_or_reparse(path):
                raise KnowledgePackError(
                    "local knowledge sidecar cannot use links/reparse points"
                )
        manifest = self.manifest()
        try:
            # Preserve the existing accepted index-size envelope: the builder
            # already bounds documents/chunks/parser output, while optional
            # vectors can make a valid database larger than one document.  The
            # observed size is only a read ceiling; _read_stable_local_file
            # binds the bytes to one descriptor and rejects replacement.
            observed_size = self.database_path.lstat().st_size
            database_bytes, _info = _read_stable_local_file(
                self.database_path, max_bytes=max(1, observed_size),
            )
        except (OSError, KnowledgePackError) as exc:
            raise KnowledgePackError(f"cannot read local knowledge database: {exc}") from exc
        current_hash = hashlib.sha256(database_bytes).hexdigest()
        if current_hash != manifest.index_sha256:
            raise KnowledgePackError("local knowledge database hash does not match its manifest")
        return manifest, database_bytes

    @staticmethod
    def _deserialize_into(
        connection: sqlite3.Connection, database_bytes: bytes,
    ) -> bool:
        """Load bytes with CPython's optional sqlite3_deserialize binding."""
        deserialize = getattr(connection, "deserialize", None)
        if not callable(deserialize):
            return False
        deserialize(database_bytes)
        return True

    @staticmethod
    def _open_database_snapshot(database_bytes: bytes) -> sqlite3.Connection:
        """Open verified SQLite bytes without consulting the sidecar path."""
        connection = sqlite3.connect(":memory:")
        try:
            if not LocalKnowledgeSidecar._deserialize_into(connection, database_bytes):
                # CPython 3.10 does not expose sqlite3_deserialize().  SQLite
                # cannot portably open an anonymous descriptor (notably, the
                # Ubuntu 22.04 VFS rejects /proc/self/fd URIs), so stage only
                # the already verified bytes in a fresh private directory.
                # This never reopens the user-controlled sidecar path.
                expected_digest = hashlib.sha256(database_bytes).digest()

                def identity(info: os.stat_result) -> tuple[int, int, int]:
                    return (info.st_dev, info.st_ino, stat.S_IFMT(info.st_mode))

                with tempfile.TemporaryDirectory(
                    prefix="hlsgraph-verified-sidecar-",
                ) as staging_name:
                    staging = Path(staging_name).absolute()
                    os.chmod(staging, 0o700)
                    if (_has_link_ancestor(staging)
                            or _is_link_or_reparse(staging)
                            or not staging.is_dir()):
                        raise KnowledgePackError(
                            "verified sidecar staging root is not a plain directory"
                        )
                    staging_info = staging.lstat()
                    staging_identity = identity(staging_info)
                    if (not stat.S_ISDIR(staging_info.st_mode)
                            or (os.name != "nt"
                                and stat.S_IMODE(staging_info.st_mode) & 0o077)):
                        raise KnowledgePackError(
                            "verified sidecar staging root is not private"
                        )

                    staged = staging / "snapshot.sqlite3"
                    flags = (
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL
                        | getattr(os, "O_BINARY", 0)
                        | getattr(os, "O_NOFOLLOW", 0)
                    )
                    descriptor = os.open(staged, flags, 0o600)
                    try:
                        opened_info = os.fstat(descriptor)
                        if not stat.S_ISREG(opened_info.st_mode):
                            raise KnowledgePackError(
                                "verified sidecar snapshot is not a regular file"
                            )
                        view = memoryview(database_bytes)
                        while view:
                            written = os.write(descriptor, view)
                            if written <= 0:
                                raise OSError("short verified sidecar snapshot write")
                            view = view[written:]
                        os.fsync(descriptor)
                        written_info = os.fstat(descriptor)
                    finally:
                        os.close(descriptor)
                    os.chmod(staged, 0o600)

                    if (_is_link_or_reparse(staging)
                            or _is_link_or_reparse(staged)):
                        raise KnowledgePackError(
                            "verified sidecar staging cannot use links/reparse points"
                        )
                    before_bytes, before_info = _read_stable_local_file(
                        staged, max_bytes=max(1, len(database_bytes)),
                    )
                    if (identity(opened_info) != identity(written_info)
                            or identity(written_info) != identity(before_info)
                            or before_bytes != database_bytes
                            or hashlib.sha256(before_bytes).digest()
                            != expected_digest):
                        raise KnowledgePackError(
                            "verified local knowledge snapshot changed before open"
                        )

                    source = sqlite3.connect(
                        staged.as_uri() + "?mode=ro&immutable=1", uri=True,
                    )
                    try:
                        source.execute("PRAGMA query_only=ON")
                        source.backup(connection)
                    finally:
                        source.close()

                    after_bytes, after_info = _read_stable_local_file(
                        staged, max_bytes=max(1, len(database_bytes)),
                    )
                    current_staging_info = staging.lstat()
                    if (_is_link_or_reparse(staging)
                            or _is_link_or_reparse(staged)
                            or identity(current_staging_info) != staging_identity
                            or identity(after_info) != identity(before_info)
                            or after_bytes != database_bytes
                            or hashlib.sha256(after_bytes).digest()
                            != expected_digest):
                        raise KnowledgePackError(
                            "verified local knowledge snapshot changed during open"
                        )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA query_only=ON")
        except KnowledgePackError:
            connection.close()
            raise
        except (sqlite3.Error, OSError, TypeError, ValueError) as exc:
            connection.close()
            raise KnowledgePackError(
                f"cannot open verified local knowledge database: {exc}"
            ) from exc
        return connection

    @staticmethod
    def _verify_document_row(row: sqlite3.Row, expected_hash: str) -> None:
        metadata = LocalDocumentMetadata(
            document_id=row[1], document_version=row[2], uri=row[3],
            sha256=row[4], size=int(row[5]), modified_ns=int(row[6]),
            indexed_at="1970-01-01T00:00:00+00:00", title=row[7],
            media_type=row[8], official_url=row[9],
        )
        if metadata.sha256 != expected_hash:
            raise KnowledgePackError("local knowledge manifest/document hash disagreement")
        _document_bytes(metadata, max_bytes=DEFAULT_MAX_DOCUMENT_BYTES)

    def search(
        self, query: str, *, limit: int = 8, include_text: bool = False,
    ) -> list[LocalKnowledgeHit]:
        """Search privately; excerpts require an explicit authorization flag."""
        if not isinstance(query, str) or "\x00" in query or len(query) > 4_096:
            raise KnowledgePackError(
                "local knowledge query must be a string of at most 4096 characters without NUL"
            )
        query = query.strip()
        if not query:
            return []
        limit = max(1, min(int(limit), 100))
        manifest, database_bytes = self._verified_database_snapshot()
        channel = "fts5"
        result: list[LocalKnowledgeHit] = []
        with closing(self._open_database_snapshot(database_bytes)) as connection:
            rows: list[sqlite3.Row] = []
            tokens = [item for item in re.findall(r"[\w:.+-]+", query) if item]
            if manifest.fts_enabled and tokens:
                expression = " AND ".join(
                    '"' + item.replace('"', '""') + '"' for item in tokens
                )
                try:
                    rows = connection.execute(
                        "SELECT c.id,c.document_key,d.document_id,d.document_version,"
                        "d.title,c.heading,c.chunk_sha256,c.text,bm25(chunks_fts) AS score,"
                        "d.uri,d.sha256,d.size,d.modified_ns,d.media_type,d.official_url "
                        "FROM chunks_fts f JOIN chunks c ON c.id=f.chunk_id "
                        "JOIN documents d ON d.document_key=c.document_key "
                        "WHERE chunks_fts MATCH ? ORDER BY score,c.id LIMIT ?",
                        (expression, limit),
                    ).fetchall()
                except sqlite3.OperationalError:
                    rows = []
            if not rows:
                channel = "substring"
                pattern = "%" + query.replace("\\", "\\\\").replace(
                    "%", "\\%",
                ).replace("_", "\\_") + "%"
                rows = connection.execute(
                    "SELECT c.id,c.document_key,d.document_id,d.document_version,"
                    "d.title,c.heading,c.chunk_sha256,c.text,0.0 AS score,"
                    "d.uri,d.sha256,d.size,d.modified_ns,d.media_type,d.official_url "
                    "FROM chunks c JOIN documents d ON d.document_key=c.document_key "
                    "WHERE c.text LIKE ? ESCAPE '\\' ORDER BY c.id LIMIT ?",
                    (pattern, limit),
                ).fetchall()
            verified: set[str] = set()
            for row in rows:
                key = str(row[1])
                if include_text and key not in verified:
                    document_row = connection.execute(
                        "SELECT document_key,document_id,document_version,uri,sha256,size,"
                        "modified_ns,title,media_type,official_url FROM documents "
                        "WHERE document_key=?", (key,),
                    ).fetchone()
                    if document_row is None or key not in manifest.document_hashes:
                        raise KnowledgePackError("local knowledge document is not in the manifest")
                    self._verify_document_row(document_row, manifest.document_hashes[key])
                    verified.add(key)
                result.append(LocalKnowledgeHit(
                    chunk_id=row[0], document_id=row[2], document_version=row[3],
                    # ``heading`` is parsed from the private document body, not
                    # declared index metadata.  Returning it on the default
                    # metadata-only path would therefore disclose document
                    # text without the bounded-snippet authorization gate.
                    title=row[4], heading=row[5] if include_text else None,
                    chunk_sha256=row[6],
                    excerpt=_bounded_excerpt(row[7]) if include_text else None,
                    score=float(row[8]),
                    channel=channel,
                ))
        return result


__all__ = [
    "DEFAULT_CHUNK_CHARS", "DEFAULT_CHUNK_OVERLAP", "DEFAULT_MAX_CHUNKS",
    "DEFAULT_MAX_DOCUMENT_BYTES", "DEFAULT_MAX_PARSED_CHARS",
    "DEFAULT_PARSER_TIMEOUT_S", "LocalKnowledgeHit", "LocalKnowledgeSidecar",
    "SIDECAR_RELATIVE_ROOT", "SIDECAR_SCHEMA_VERSION",
]
