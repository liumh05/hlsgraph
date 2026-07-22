"""Run and replay one fail-closed public knowledge-pack review.

Formal reviews are Linux/WSL2-only.  The model runs in a read-only named
permission profile over an ext4 public checkout.  Every home directory,
``CODEX_HOME`` and Windows/drvfs mount is denied to model-issued commands.
The raw ``codex exec --json`` stream is retained outside the checkout and is
the authority for the normalized public trace, result and receipt.

This module deliberately does not offer a generic shell.  A completed command
event is accepted only when it is one of the small read-only grammars below.
Unknown events or tools fail the review instead of being ignored.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


# ``python3 tools/run_knowledge_review.py`` puts ``tools/`` rather than the
# checkout root on sys.path.  The documented script entry point nevertheless
# imports the sibling ``eval`` package and the ``tools`` namespace package.
SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))


SEMANTIC_PROTOCOL = "hlsgraph.knowledge-review.semantic.v1"
ADVERSARIAL_PROTOCOL = "hlsgraph.knowledge-review.adversarial.v1"
PROTOCOLS = frozenset({SEMANTIC_PROTOCOL, ADVERSARIAL_PROTOCOL})
MODEL = "gpt-5.6-sol"
REASONING_EFFORT = "medium"
CODEX_CLI_VERSION = "codex-cli 0.144.0"
TRACE_SCHEMA_VERSION = "hlsgraph.knowledge-review.tool-trace.v2"
RECEIPT_SCHEMA_VERSION = "hlsgraph.knowledge-review.cli-receipt.v2"
REVIEW_SCHEMA_PATH = "tools/knowledge_review.schema.json"
REVIEW_RECEIPT_SCHEMA_PATH = "tools/knowledge_review_receipt.schema.json"
CITATION_AUDIT_PATH = "docs/knowledge-citation-audit-v0.3.json"
RUNNER_PATH = "tools/run_knowledge_review.py"
CITATION_GENERATOR_PATH = "tools/audit_knowledge_citations.py"
SURFACE_HELPER_PATH = "tools/knowledge_review_surface.py"
RELEASE_AUDITOR_PATH = "tools/audit_release.py"
CACHE_SCHEMA_VERSION = "hlsgraph.knowledge-review.cache.v1"
CACHE_MANIFEST_NAME = "manifest.json"
MAX_CITATION_BYTES = 32 * 1024 * 1024
MAX_REDIRECTS = 5
IMPLEMENTATION_SURFACE_HASH_KEY = "src/hlsgraph/**/*.py#implementation-surface"
PACK_SURFACE_HASH_PREFIX = "src/hlsgraph/knowledge/packs/"
PACK_SURFACE_HASH_SUFFIX = "#semantic-surface"
SURFACE_HELPER_HASH_KEY = SURFACE_HELPER_PATH + "#sha256"

PROTOCOL_FILES = {
    SEMANTIC_PROTOCOL: {
        "prompt": "tools/knowledge_review_prompts/semantic.md",
        "result": "docs/knowledge-review-v0.3.semantic.json",
        "trace": "docs/knowledge-review-v0.3.semantic.trace.jsonl",
        "receipt": "docs/knowledge-review-v0.3.semantic.receipt.json",
    },
    ADVERSARIAL_PROTOCOL: {
        "prompt": "tools/knowledge_review_prompts/adversarial.md",
        "result": "docs/knowledge-review-v0.3.adversarial.json",
        "trace": "docs/knowledge-review-v0.3.adversarial.trace.jsonl",
        "receipt": "docs/knowledge-review-v0.3.adversarial.receipt.json",
    },
}

DISABLED_CODEX_FEATURES = (
    "browser_use", "browser_use_external", "browser_use_full_cdp_access",
    "in_app_browser", "standalone_web_search", "computer_use",
    "image_generation", "apps", "enable_mcp_apps", "multi_agent",
    "multi_agent_v2", "plugins", "plugin_sharing", "remote_plugin",
    "hooks", "workspace_dependencies",
)
PERMISSION_PROFILE = "hlsgraph_knowledge_review"
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{7,127}")
_CALL_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{2,127}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_FORBIDDEN_SHELL = re.compile(
    r"[;&|><`\r\n]|\$\(|\$\{|\$env:|%[A-Za-z_][A-Za-z0-9_]*%",
    re.IGNORECASE,
)
_ALLOWED_EVENT_TYPES = frozenset({
    "thread.started", "turn.started", "item.started", "item.completed",
    "turn.completed",
})
_ALLOWED_NONCOMMAND_ITEMS = frozenset({"reasoning", "agent_message"})


@dataclass(frozen=True)
class ReviewReplay:
    protocol_id: str
    invocation_id: str
    thread_id: str
    raw_sha256: str
    result: dict[str, Any]
    result_bytes: bytes
    trace_bytes: bytes


@dataclass(frozen=True)
class ReviewFileSnapshot:
    """One immutable logical review input and its cache projection."""

    path: str
    hash_kind: str
    sha256: str
    cache_sha256: str
    payload: bytes = field(repr=False, compare=True)

    def inventory(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "hash_kind": self.hash_kind,
            "sha256": self.sha256,
            "cache_path": f"files/{self.path}",
            "cache_sha256": self.cache_sha256,
            "cache_size": len(self.payload),
        }


@dataclass(frozen=True)
class ReviewSnapshot:
    """All checkout bytes and semantic projections promised to one review."""

    protocol_id: str
    files: tuple[ReviewFileSnapshot, ...]
    review_surface_sha256: tuple[tuple[str, str], ...]
    implementation_surface_sha256: str
    citation_audit_sha256: str
    output_schema_sha256: str
    receipt_schema_sha256: str
    exact_citation_urls: tuple[str, ...]

    @property
    def file_map(self) -> dict[str, ReviewFileSnapshot]:
        return {item.path: item for item in self.files}

    @property
    def surfaces(self) -> dict[str, str]:
        return dict(self.review_surface_sha256)

    def inventory(self) -> dict[str, Any]:
        return {
            "protocol_id": self.protocol_id,
            "review_surface_sha256": self.surfaces,
            "implementation_surface_sha256": self.implementation_surface_sha256,
            "citation_audit_sha256": self.citation_audit_sha256,
            "output_schema_sha256": self.output_schema_sha256,
            "receipt_schema_sha256": self.receipt_schema_sha256,
            "exact_citation_urls": list(self.exact_citation_urls),
            "required_files": [item.inventory() for item in self.files],
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.inventory())).hexdigest()


@dataclass(frozen=True)
class TrustedFetch:
    status: int
    final_url: str
    redirect_chain: tuple[str, ...]
    content_type: str
    body: bytes = field(repr=False)
    charset: str | None = None


@dataclass(frozen=True)
class TextDerivation:
    text: bytes = field(repr=False)
    parser_id: str
    parser_version: str
    command_sha256: str


@dataclass(frozen=True)
class ReviewCache:
    root: Path
    manifest: dict[str, Any]
    manifest_bytes: bytes = field(repr=False)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.manifest_bytes).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False,
    ) + "\n").encode("utf-8")


def _canonical_jsonl(rows: Iterable[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
        for row in rows
    ).encode("utf-8")


def _strict_json_bytes(data: bytes, *, label: str) -> Any:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(
            data.decode("utf-8", errors="strict"),
            object_pairs_hook=no_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"{label} contains non-finite number {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"cannot parse strict {label}: {exc}") from exc


def _strict_jsonl(data: bytes, *, label: str) -> list[dict[str, Any]]:
    if not data or not data.endswith(b"\n"):
        raise ValueError(f"{label} must be non-empty JSONL ending in LF")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(data.splitlines(), 1):
        if not line:
            raise ValueError(f"{label} contains a blank line at {line_number}")
        value = _strict_json_bytes(line, label=f"{label}:{line_number}")
        if not isinstance(value, dict):
            raise ValueError(f"{label}:{line_number} is not an object")
        rows.append(value)
    return rows


def _content_text(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        pieces = [
            text for item in value
            if (text := _content_text(item)) is not None
        ]
        return "\n".join(pieces) if pieces else None
    if isinstance(value, dict):
        for key in ("text", "output_text", "content"):
            text = _content_text(value.get(key))
            if text is not None:
                return text
    return None


def required_read_paths(root: Path, protocol_id: str) -> set[str]:
    files = PROTOCOL_FILES.get(protocol_id)
    if files is None:
        raise ValueError(f"unknown knowledge-review protocol: {protocol_id}")
    paths = {
        REVIEW_SCHEMA_PATH, REVIEW_RECEIPT_SCHEMA_PATH, CITATION_AUDIT_PATH,
        RUNNER_PATH, CITATION_GENERATOR_PATH, SURFACE_HELPER_PATH,
        RELEASE_AUDITOR_PATH,
        *(item["prompt"] for item in PROTOCOL_FILES.values()),
    }
    implementation = root / "src" / "hlsgraph"
    paths.update(
        path.relative_to(root).as_posix()
        for path in implementation.rglob("*.py") if path.is_file()
    )
    paths.update(
        path.relative_to(root).as_posix()
        for path in (implementation / "knowledge" / "packs").glob("*.json")
        if path.is_file()
    )
    return paths


def _citation_rows(root: Path) -> list[dict[str, Any]]:
    value = _strict_json_bytes(
        (root / CITATION_AUDIT_PATH).read_bytes(), label="citation audit",
    )
    if not isinstance(value, dict) or not isinstance(value.get("references"), list):
        raise ValueError("citation audit has no reference inventory")
    rows = value["references"]
    if any(not isinstance(item, dict) for item in rows):
        raise ValueError("citation audit has a malformed reference inventory")
    return rows


def exact_citation_urls(root: Path) -> set[str]:
    urls = {str(item.get("citation_url", "")) for item in _citation_rows(root)}
    for url in urls:
        parts = urlsplit(url)
        if parts.scheme.casefold() != "https" or not parts.hostname:
            raise ValueError("citation inventory contains a non-HTTPS locator")
    if "" in urls:
        raise ValueError("citation inventory contains an empty locator")
    return urls


def _semantic_pack_projection(data: bytes, *, label: str) -> tuple[str, str, bytes]:
    value = _strict_json_bytes(data, label=label)
    if not isinstance(value, dict):
        raise ValueError(f"{label} root must be an object")
    try:
        from tools import knowledge_review_surface
        semantic = knowledge_review_surface.semantic_surface(value)
    except (ImportError, TypeError, ValueError) as exc:
        raise ValueError(f"cannot derive {label} semantic surface: {exc}") from exc
    pack_id = semantic.get("pack_id")
    if not isinstance(pack_id, str) or not pack_id:
        raise ValueError(f"{label} has no pack_id")
    compact = json.dumps(
        semantic, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return pack_id, hashlib.sha256(compact).hexdigest(), _canonical_json(semantic)


def _implementation_hash(files: Iterable[ReviewFileSnapshot]) -> str:
    digest = hashlib.sha256()
    implementation = sorted(
        (item for item in files
         if item.path.startswith("src/hlsgraph/") and item.path.endswith(".py")),
        key=lambda item: item.path,
    )
    for item in implementation:
        digest.update(item.path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(item.payload).digest())
        digest.update(b"\0")
    return digest.hexdigest()


def freeze_review_snapshot(root: Path, protocol_id: str) -> ReviewSnapshot:
    """Read the complete review surface once into an immutable snapshot."""

    if protocol_id not in PROTOCOLS:
        raise ValueError(f"unknown knowledge-review protocol: {protocol_id}")
    root = root.resolve(strict=True)
    snapshots: list[ReviewFileSnapshot] = []
    surfaces: dict[str, str] = {}
    for relative in sorted(required_read_paths(root, protocol_id)):
        path = root / PurePosixPath(relative)
        data = path.read_bytes()
        if relative.startswith("src/hlsgraph/knowledge/packs/"):
            pack_id, digest, cache_payload = _semantic_pack_projection(
                data, label=f"knowledge pack {relative}",
            )
            if pack_id in surfaces:
                raise ValueError(f"duplicate reviewed pack ID: {pack_id}")
            surfaces[pack_id] = digest
            hash_kind = "review_surface_sha256"
        else:
            digest = hashlib.sha256(data).hexdigest()
            cache_payload = data
            hash_kind = "raw_sha256"
        snapshots.append(ReviewFileSnapshot(
            path=relative, hash_kind=hash_kind, sha256=digest,
            cache_sha256=hashlib.sha256(cache_payload).hexdigest(),
            payload=cache_payload,
        ))
    by_path = {item.path: item for item in snapshots}
    citation = by_path.get(CITATION_AUDIT_PATH)
    schema = by_path.get(REVIEW_SCHEMA_PATH)
    receipt_schema = by_path.get(REVIEW_RECEIPT_SCHEMA_PATH)
    if citation is None or schema is None or receipt_schema is None:
        raise ValueError("review snapshot omits a required schema or citation manifest")
    citation_value = _strict_json_bytes(citation.payload, label="citation audit")
    if not isinstance(citation_value, dict) or not isinstance(
        citation_value.get("references"), list
    ):
        raise ValueError("citation audit has no reference inventory")
    urls: set[str] = set()
    for row in citation_value["references"]:
        if not isinstance(row, dict):
            raise ValueError("citation audit has a malformed reference inventory")
        url = row.get("citation_url")
        if not isinstance(url, str) or not url:
            raise ValueError("citation audit contains an empty locator")
        parts = urlsplit(url)
        if parts.scheme.casefold() != "https" or not parts.hostname:
            raise ValueError("citation audit contains a non-HTTPS locator")
        urls.add(url)
    snapshot = ReviewSnapshot(
        protocol_id=protocol_id, files=tuple(snapshots),
        review_surface_sha256=tuple(sorted(surfaces.items())),
        implementation_surface_sha256=_implementation_hash(snapshots),
        citation_audit_sha256=citation.sha256,
        output_schema_sha256=schema.sha256,
        receipt_schema_sha256=receipt_schema.sha256,
        exact_citation_urls=tuple(sorted(urls)),
    )
    if not snapshots or not surfaces:
        raise ValueError("review snapshot has an empty implementation or pack inventory")
    return snapshot


class _SameHostRedirectHandler(HTTPRedirectHandler):
    def __init__(self, requested_url: str, max_redirects: int) -> None:
        super().__init__()
        self.requested_url = requested_url
        self.expected_host = (urlsplit(requested_url).hostname or "").casefold()
        self.max_redirects = max_redirects
        self.chain: list[str] = [requested_url]

    def redirect_request(  # type: ignore[override]
        self, req: Request, fp: Any, code: int, msg: str,
        headers: Any, newurl: str,
    ) -> Request | None:
        resolved = urljoin(req.full_url, newurl)
        parts = urlsplit(resolved)
        if (parts.scheme.casefold() != "https" or not parts.hostname
                or parts.hostname.casefold() != self.expected_host):
            raise ValueError("citation redirect leaves the exact same-host HTTPS boundary")
        if len(self.chain) > self.max_redirects:
            raise ValueError("citation redirect chain exceeds the fixed maximum")
        self.chain.append(resolved)
        return super().redirect_request(req, fp, code, msg, headers, resolved)


def _default_fetch(
    url: str, timeout_seconds: float, max_bytes: int,
) -> TrustedFetch:
    handler = _SameHostRedirectHandler(url, MAX_REDIRECTS)
    opener = build_opener(handler)
    request = Request(url, headers={"User-Agent": "hlsgraph-knowledge-review/1"})
    try:
        with opener.open(request, timeout=timeout_seconds) as response:
            status_code = int(getattr(response, "status", response.getcode()))
            final_url = str(response.geturl())
            body = response.read(max_bytes + 1)
            content_type = str(response.headers.get_content_type())
            charset = response.headers.get_content_charset()
    except (HTTPError, URLError, OSError) as exc:
        raise ValueError(f"exact citation fetch failed: {type(exc).__name__}") from exc
    if len(body) > max_bytes:
        raise ValueError("exact citation response exceeds the fixed byte limit")
    final_parts = urlsplit(final_url)
    expected_host = (urlsplit(url).hostname or "").casefold()
    if (not 200 <= status_code < 300 or final_parts.scheme.casefold() != "https"
            or (final_parts.hostname or "").casefold() != expected_host):
        raise ValueError("exact citation fetch did not finish at same-host HTTPS")
    chain = list(handler.chain)
    if not chain or chain[-1] != final_url:
        chain.append(final_url)
    for item in chain:
        parts = urlsplit(item)
        if (parts.scheme.casefold() != "https"
                or (parts.hostname or "").casefold() != expected_host):
            raise ValueError("citation redirect chain is not same-host HTTPS")
    return TrustedFetch(
        status=status_code, final_url=final_url,
        redirect_chain=tuple(chain), content_type=content_type,
        charset=charset, body=body,
    )


def _mkdir_private(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=False, mode=0o700)
    if os.name != "nt":
        path.chmod(0o700)


def _ensure_private_parent(path: Path) -> None:
    if not path.exists():
        _mkdir_private(path)
    if not path.is_dir() or path.is_symlink():
        raise RuntimeError(f"restricted evidence parent is not a plain directory: {path}")
    if os.name != "nt":
        mode = stat.S_IMODE(path.stat().st_mode)
        if mode != 0o700:
            raise RuntimeError(f"restricted evidence directory must be mode 0700: {path}")


def _write_private(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name != "nt":
        path.parent.chmod(0o700)
    flags = (
        os.O_WRONLY | os.O_CREAT | os.O_EXCL
        | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short private evidence write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if os.name != "nt":
        path.chmod(0o600)


def _harden_private_tree(root: Path) -> None:
    """Set the final exact modes before a cache becomes review evidence."""

    if os.name == "nt":
        return
    for current, directories, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        current_path.chmod(0o700)
        for name in directories:
            path = current_path / name
            if path.is_symlink():
                raise RuntimeError(f"private evidence tree contains a symlink: {path}")
            path.chmod(0o700)
        for name in filenames:
            path = current_path / name
            if path.is_symlink():
                raise RuntimeError(f"private evidence tree contains a symlink: {path}")
            path.chmod(0o600)


def _is_link_like(path: Path) -> bool:
    if path.is_symlink():
        return True
    isjunction = getattr(os.path, "isjunction", None)
    return bool(isjunction is not None and isjunction(path))


def _resolved_unlinked_path(path: Path, *, label: str) -> Path:
    """Resolve a prospective path while rejecting every existing link alias."""
    lexical = path.absolute()
    for component in (lexical, *lexical.parents):
        if _is_link_like(component):
            raise RuntimeError(f"{label} has a linked path component: {component}")
    return lexical.resolve(strict=False)


def _assert_private_mode(path: Path, expected: int, *, label: str) -> None:
    if os.name == "nt":
        return
    actual = stat.S_IMODE(path.stat(follow_symlinks=False).st_mode)
    if actual != expected:
        raise ValueError(
            f"{label} must be mode {expected:04o}, found {actual:04o}: {path}"
        )


def _read_private_cache_file(root: Path, relative: str) -> bytes:
    relative_path = _safe_relative(relative)
    current = root
    for index, part in enumerate(relative_path.parts):
        current = current / part
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise ValueError(f"review cache path is missing: {relative}") from exc
        if _is_link_like(current):
            raise ValueError(f"review cache contains a linked path: {relative}")
        if index < len(relative_path.parts) - 1:
            if not stat.S_ISDIR(metadata.st_mode):
                raise ValueError(f"review cache parent is not a directory: {relative}")
            _assert_private_mode(current, 0o700, label="review cache directory")
        else:
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError(f"review cache entry is not a plain file: {relative}")
            _assert_private_mode(current, 0o600, label="review cache file")
    return current.read_bytes()


def _text_derivation(fetch: TrustedFetch) -> TextDerivation | None:
    if fetch.body.startswith(b"%PDF-") or fetch.content_type.casefold() == "application/pdf":
        return None
    charset = (fetch.charset or "utf-8").casefold()
    try:
        text = fetch.body.decode(charset, errors="strict")
    except (LookupError, UnicodeDecodeError):
        return None
    encoded = text.encode("utf-8")
    if not text.strip():
        return None
    contract = {
        "parser_id": "python-text-decode",
        "parser_version": "hlsgraph.citation-text.identity.v1",
        "charset": charset,
    }
    return TextDerivation(
        text=encoded, parser_id=contract["parser_id"],
        parser_version=contract["parser_version"],
        command_sha256=hashlib.sha256(_canonical_json(contract)).hexdigest(),
    )


def _pdftotext_derivation(
    body_path: Path, command: str | None,
) -> TextDerivation | None:
    if not command:
        return None
    resolved = shutil.which(command) if Path(command).parent == Path(".") else command
    if not resolved:
        return None
    executable = Path(resolved).resolve(strict=True)
    version = subprocess.run(
        [str(executable), "-v"], check=False, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, timeout=15,
    )
    version_text = (version.stdout + version.stderr).decode(
        "utf-8", errors="replace",
    ).strip()
    if version.returncode != 0 or not version_text:
        return None
    completed = subprocess.run(
        [str(executable), "-layout", str(body_path), "-"], check=False,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120,
    )
    if completed.returncode != 0:
        return None
    try:
        text = completed.stdout.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return None
    if not text.strip():
        return None
    contract = {
        "parser_id": "pdftotext",
        "parser_version": version_text,
        "binary_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "argv": ["pdftotext", "-layout", "$INPUT", "-"],
    }
    return TextDerivation(
        text=text.encode("utf-8"), parser_id="pdftotext",
        parser_version=version_text,
        command_sha256=hashlib.sha256(_canonical_json(contract)).hexdigest(),
    )


def _citation_reference_rows(snapshot: ReviewSnapshot) -> list[dict[str, Any]]:
    item = snapshot.file_map[CITATION_AUDIT_PATH]
    value = _strict_json_bytes(item.payload, label="citation audit")
    rows = value.get("references") if isinstance(value, dict) else None
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("citation audit has a malformed reference inventory")
    return rows


def create_review_cache(
    root: Path, snapshot: ReviewSnapshot, cache_root: Path, *,
    fetcher: Callable[[str, float, int], TrustedFetch] = _default_fetch,
    timeout_seconds: float = 60.0, max_bytes: int = MAX_CITATION_BYTES,
    pdf_text_extractor: Callable[[bytes], TextDerivation | None] | None = None,
    pdftotext_command: str | None = None,
) -> ReviewCache:
    """Create one private, immutable review cache outside the checkout."""

    root = root.resolve(strict=True)
    lexical_cache = cache_root.absolute()
    try:
        lexical_cache.resolve(strict=False).relative_to(root)
    except ValueError:
        pass
    else:
        raise RuntimeError("review cache must stay outside the public checkout")
    if lexical_cache.exists():
        raise RuntimeError("review cache path already exists")
    if not lexical_cache.parent.is_dir() or lexical_cache.parent.is_symlink():
        raise RuntimeError("review cache parent must be an existing plain directory")
    _mkdir_private(lexical_cache)
    files_inventory: list[dict[str, Any]] = []
    for item in snapshot.files:
        relative = PurePosixPath("files") / PurePosixPath(item.path)
        target = lexical_cache / relative
        _write_private(target, item.payload)
        files_inventory.append(item.inventory())

    references_by_url: dict[str, list[str]] = {}
    for row in _citation_reference_rows(snapshot):
        url = str(row.get("citation_url", ""))
        reference_id = str(row.get("reference_id", ""))
        if url not in snapshot.exact_citation_urls or _SHA256_RE.fullmatch(reference_id) is None:
            raise ValueError("citation reference is not bound to the frozen inventory")
        references_by_url.setdefault(url, []).append(reference_id)

    citations: list[dict[str, Any]] = []
    for url in snapshot.exact_citation_urls:
        base: dict[str, Any] = {
            "requested_url": url,
            "reference_ids": sorted(references_by_url.get(url, [])),
            "available": False,
            "status": None,
            "final_url": None,
            "redirect_chain": [],
            "content_type": None,
            "body_path": None,
            "body_sha256": None,
            "body_size": None,
            "inspection_path": None,
            "inspection_sha256": None,
            "inspection_size": None,
            "parser_id": None,
            "parser_version": None,
            "parser_command_sha256": None,
            "error_code": None,
        }
        try:
            fetched = fetcher(url, timeout_seconds, max_bytes)
            if not isinstance(fetched, TrustedFetch):
                raise TypeError("fetcher did not return a TrustedFetch")
            if not isinstance(fetched.body, bytes) or len(fetched.body) > max_bytes:
                raise ValueError("fetcher returned an invalid or oversized body")
            if not isinstance(fetched.content_type, str):
                raise TypeError("fetcher returned an invalid content type")
            if (type(fetched.status) is not int
                    or fetched.status < 200 or fetched.status >= 300):
                raise ValueError("non-success citation status")
            expected_host = (urlsplit(url).hostname or "").casefold()
            if not fetched.redirect_chain or fetched.redirect_chain[0] != url:
                raise ValueError("fetcher omitted the exact requested locator")
            if len(fetched.redirect_chain) > MAX_REDIRECTS + 1:
                raise ValueError("fetcher exceeded the fixed redirect limit")
            for redirected in fetched.redirect_chain:
                parts = urlsplit(redirected)
                if (parts.scheme.casefold() != "https"
                        or (parts.hostname or "").casefold() != expected_host):
                    raise ValueError("fetcher left the exact same-host HTTPS boundary")
            final_parts = urlsplit(fetched.final_url)
            if (fetched.redirect_chain[-1] != fetched.final_url
                    or final_parts.scheme.casefold() != "https"
                    or (final_parts.hostname or "").casefold() != expected_host):
                raise ValueError("fetcher final URL differs from its same-host redirect chain")
            body_hash = hashlib.sha256(fetched.body).hexdigest()
            body_relative = f"citations/bodies/{body_hash}.body"
            body_path = lexical_cache / PurePosixPath(body_relative)
            if not body_path.exists():
                _write_private(body_path, fetched.body)
            elif body_path.read_bytes() != fetched.body:
                raise ValueError("citation body hash collision")
            derivation = _text_derivation(fetched)
            if derivation is None and (
                fetched.body.startswith(b"%PDF-")
                or fetched.content_type.casefold() == "application/pdf"
            ):
                derivation = (
                    pdf_text_extractor(fetched.body)
                    if pdf_text_extractor is not None
                    else _pdftotext_derivation(body_path, pdftotext_command)
                )
            base.update({
                "status": fetched.status, "final_url": fetched.final_url,
                "redirect_chain": list(fetched.redirect_chain),
                "content_type": fetched.content_type,
                "body_path": body_relative, "body_sha256": body_hash,
                "body_size": len(fetched.body),
            })
            if derivation is None or not derivation.text.strip():
                base["error_code"] = "citation_text_unavailable"
            else:
                if (not isinstance(derivation, TextDerivation)
                        or not isinstance(derivation.text, bytes)
                        or not isinstance(derivation.parser_id, str)
                        or not derivation.parser_id
                        or not isinstance(derivation.parser_version, str)
                        or not derivation.parser_version
                        or _SHA256_RE.fullmatch(
                            str(derivation.command_sha256)
                        ) is None):
                    raise ValueError("citation text derivation lacks a bound parser contract")
                text_hash = hashlib.sha256(derivation.text).hexdigest()
                text_relative = f"citations/text/{text_hash}.txt"
                text_path = lexical_cache / PurePosixPath(text_relative)
                if not text_path.exists():
                    _write_private(text_path, derivation.text)
                elif text_path.read_bytes() != derivation.text:
                    raise ValueError("citation text hash collision")
                base.update({
                    "available": True,
                    "inspection_path": text_relative,
                    "inspection_sha256": text_hash,
                    "inspection_size": len(derivation.text),
                    "parser_id": derivation.parser_id,
                    "parser_version": derivation.parser_version,
                    "parser_command_sha256": derivation.command_sha256,
                })
        except (
            AttributeError, OSError, TypeError, ValueError,
            subprocess.SubprocessError,
        ) as exc:
            base["error_code"] = type(exc).__name__
        citations.append(base)

    manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "protocol_id": snapshot.protocol_id,
        "review_snapshot_sha256": snapshot.sha256,
        "review_snapshot": snapshot.inventory(),
        "files": files_inventory,
        "citations": citations,
    }
    manifest_bytes = _canonical_json(manifest)
    _write_private(lexical_cache / CACHE_MANIFEST_NAME, manifest_bytes)
    _harden_private_tree(lexical_cache)
    return ReviewCache(lexical_cache.resolve(), manifest, manifest_bytes)


def build_review_prompt(
    root: Path, protocol_id: str, *, snapshot: ReviewSnapshot | None = None,
    cache: ReviewCache | None = None,
) -> bytes:
    files = PROTOCOL_FILES.get(protocol_id)
    if files is None:
        raise ValueError(f"unknown knowledge-review protocol: {protocol_id}")
    snapshot = snapshot or freeze_review_snapshot(root, protocol_id)
    if snapshot.protocol_id != protocol_id:
        raise ValueError("review snapshot belongs to another protocol")
    prompt_file = snapshot.file_map.get(files["prompt"])
    if prompt_file is None:
        raise ValueError("review snapshot omits its protocol prompt")
    protocol = prompt_file.payload.decode("utf-8", errors="strict")
    inventory = {
        **snapshot.inventory(),
        "review_snapshot_sha256": snapshot.sha256,
        "cache_manifest_sha256": cache.sha256 if cache is not None else None,
        "cache_manifest": cache.manifest if cache is not None else None,
    }
    command_contract = """
The model has no network and the checkout itself is not readable. The current
working directory is the private frozen cache. The only permitted shell
commands are these exact read-only forms, using forward-slash relative paths
listed in cache_manifest.files[*].cache_path or an available
cache_manifest.citations[*].inspection_path:

  head -n COUNT PATH
  sha256sum PATH [PATH ...]

Use `head -n 100000000 PATH` to inspect
each entire required file and every available citation text. `sha256sum` alone
is hash evidence, not content-inspection evidence. Do not access cached raw
response bodies directly. Do not use any other command, interpreter, pipe,
redirection, environment expansion, native web/search tool, MCP tool, network
operation, or file-changing operation. Every unknown event or tool makes the
review unusable. If a citation entry is unavailable or cannot be read in full,
its verdict must not be verified and approved must be false.
""".strip()
    payload = json.dumps(inventory, indent=2, sort_keys=True, ensure_ascii=False)
    return (protocol.rstrip() + "\n\n" + command_contract +
            "\n\n# Frozen review inventory\n\n```json\n" + payload +
            "\n```\n").encode("utf-8")


def _safe_relative(token: str) -> PurePosixPath:
    if (not token or "\\" in token or token.startswith("/")
            or re.match(r"^[A-Za-z]:", token)):
        raise ValueError(f"command uses a non-project-relative path: {token!r}")
    path = PurePosixPath(token)
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"command uses a non-canonical path: {token!r}")
    return path


def _split_command(command: str) -> list[str]:
    if not command or _FORBIDDEN_SHELL.search(command):
        raise ValueError("command contains chaining, expansion, redirection, or control text")
    try:
        parts = shlex.split(command, posix=True)
    except ValueError as exc:
        raise ValueError(f"command has invalid quoting: {exc}") from exc
    if not parts:
        raise ValueError("empty command")
    return parts


def load_review_cache(cache_root: Path, snapshot: ReviewSnapshot) -> ReviewCache:
    lexical_root = cache_root.absolute()
    if _is_link_like(lexical_root):
        raise ValueError("review cache root must be a plain directory")
    root = lexical_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("review cache root is not a directory")
    _assert_private_mode(root, 0o700, label="review cache root")
    manifest_bytes = _read_private_cache_file(root, CACHE_MANIFEST_NAME)
    manifest = _strict_json_bytes(manifest_bytes, label="review cache manifest")
    if not isinstance(manifest, dict) or _canonical_json(manifest) != manifest_bytes:
        raise ValueError("review cache manifest is not canonical JSON")
    if set(manifest) != {
        "schema_version", "protocol_id", "review_snapshot_sha256",
        "review_snapshot", "files", "citations",
    }:
        raise ValueError("review cache manifest is not a closed contract")
    if (manifest.get("schema_version") != CACHE_SCHEMA_VERSION
            or manifest.get("protocol_id") != snapshot.protocol_id
            or manifest.get("review_snapshot_sha256") != snapshot.sha256
            or manifest.get("review_snapshot") != snapshot.inventory()):
        raise ValueError("review cache manifest does not bind the exact snapshot")
    expected_files = [item.inventory() for item in snapshot.files]
    if manifest.get("files") != expected_files:
        raise ValueError("review cache file inventory differs from the snapshot")
    expected_paths = {CACHE_MANIFEST_NAME}
    for item in expected_files:
        relative = str(item["cache_path"])
        expected_paths.add(relative)
        data = _read_private_cache_file(root, relative)
        if (hashlib.sha256(data).hexdigest() != item["cache_sha256"]
                or len(data) != item["cache_size"]):
            raise ValueError(f"review cache source is stale: {item['path']}")
    citations = manifest.get("citations")
    if not isinstance(citations, list):
        raise ValueError("review cache has no citation inventory")
    references_by_url: dict[str, list[str]] = {}
    for row in _citation_reference_rows(snapshot):
        references_by_url.setdefault(str(row["citation_url"]), []).append(
            str(row["reference_id"])
        )
    observed_urls: set[str] = set()
    for entry in citations:
        if not isinstance(entry, dict) or set(entry) != {
            "requested_url", "reference_ids", "available", "status",
            "final_url", "redirect_chain", "content_type", "body_path",
            "body_sha256", "body_size", "inspection_path",
            "inspection_sha256", "inspection_size", "parser_id",
            "parser_version", "parser_command_sha256", "error_code",
        }:
            raise ValueError("review cache contains a malformed citation entry")
        url = entry.get("requested_url")
        if not isinstance(url, str) or url in observed_urls:
            raise ValueError("review cache contains a duplicate citation locator")
        observed_urls.add(url)
        if entry.get("reference_ids") != sorted(references_by_url.get(url, [])):
            raise ValueError("review cache citation reference inventory is stale")
        requested_parts = urlsplit(url)
        expected_host = (requested_parts.hostname or "").casefold()
        chain = entry.get("redirect_chain")
        if entry.get("status") is not None:
            if (not isinstance(chain, list) or not chain or chain[0] != url
                    or len(chain) > MAX_REDIRECTS + 1
                    or chain[-1] != entry.get("final_url")):
                raise ValueError("review cache has an invalid redirect chain")
            for redirected in chain:
                parts = urlsplit(str(redirected))
                if (parts.scheme.casefold() != "https" or not parts.hostname
                        or parts.hostname.casefold() != expected_host):
                    raise ValueError("review cache redirect leaves same-host HTTPS")
        elif chain != [] or entry.get("final_url") is not None:
            raise ValueError("failed citation cache entry claims redirect evidence")
        for path_key, hash_key, size_key in (
            ("body_path", "body_sha256", "body_size"),
            ("inspection_path", "inspection_sha256", "inspection_size"),
        ):
            relative = entry.get(path_key)
            if relative is None:
                continue
            relative = _safe_relative(str(relative)).as_posix()
            expected_prefix = (
                "citations/bodies/" if path_key == "body_path"
                else "citations/text/"
            )
            expected_suffix = ".body" if path_key == "body_path" else ".txt"
            digest = entry.get(hash_key)
            if (not relative.startswith(expected_prefix)
                    or relative != f"{expected_prefix}{digest}{expected_suffix}"
                    or _SHA256_RE.fullmatch(str(digest)) is None
                    or not isinstance(entry.get(size_key), int)
                    or entry[size_key] < 0):
                raise ValueError("review cache citation path is not content addressed")
            expected_paths.add(relative)
            data = _read_private_cache_file(root, relative)
            if (hashlib.sha256(data).hexdigest() != entry.get(hash_key)
                    or len(data) != entry.get(size_key)):
                raise ValueError(f"review cache citation data is stale: {url}")
        available = entry.get("available")
        if available is True:
            if (not isinstance(entry.get("status"), int)
                    or not 200 <= entry["status"] < 300
                    or not entry.get("body_path")
                    or not entry.get("inspection_path")
                    or not isinstance(entry.get("parser_id"), str)
                    or not entry["parser_id"]
                    or not isinstance(entry.get("parser_version"), str)
                    or not entry["parser_version"]
                    or _SHA256_RE.fullmatch(
                        str(entry.get("parser_command_sha256", ""))
                    ) is None
                    or entry.get("error_code") is not None):
                raise ValueError("available citation lacks fetched parser-bound text")
        elif available is not False:
            raise ValueError("review cache citation availability is not boolean")
    if observed_urls != set(snapshot.exact_citation_urls):
        raise ValueError("review cache citation inventory differs from the snapshot")

    observed_paths: set[str] = set()
    observed_directories: set[str] = {"."}
    for current, directories, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        relative_parent = current_path.relative_to(root)
        _assert_private_mode(current_path, 0o700, label="review cache directory")
        for name in directories:
            path = current_path / name
            if _is_link_like(path):
                raise ValueError("review cache contains a linked directory")
            observed_directories.add((relative_parent / name).as_posix())
        for name in filenames:
            path = current_path / name
            if _is_link_like(path) or not path.is_file():
                raise ValueError("review cache contains a non-plain file")
            observed_paths.add((relative_parent / name).as_posix())
            _assert_private_mode(path, 0o600, label="review cache file")
    expected_directories = {"."}
    for relative in expected_paths:
        parent = PurePosixPath(relative).parent
        while parent.as_posix() != ".":
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    if observed_paths != expected_paths or observed_directories != expected_directories:
        raise ValueError("review cache contains missing or unmanifested filesystem entries")
    return ReviewCache(root, manifest, manifest_bytes)


def _cache_targets(cache: ReviewCache) -> dict[str, dict[str, Any]]:
    targets: dict[str, dict[str, Any]] = {}
    for item in cache.manifest["files"]:
        targets[item["cache_path"]] = {"target_kind": "source", **item}
    for entry in cache.manifest["citations"]:
        path = entry.get("inspection_path")
        if not path:
            continue
        target = targets.setdefault(str(path), {
            "target_kind": "citation", "entries": [],
            "cache_sha256": entry["inspection_sha256"],
            "cache_size": entry["inspection_size"],
        })
        if target.get("target_kind") != "citation" or (
            target["cache_sha256"], target["cache_size"]
        ) != (entry["inspection_sha256"], entry["inspection_size"]):
            raise ValueError("review cache reuses one path for different content")
        target["entries"].append(entry)
    return targets


def _command_output(item: dict[str, Any]) -> tuple[str, str]:
    present = [key for key in ("aggregated_output", "output", "result") if key in item]
    if present != ["aggregated_output"] or not isinstance(
        item.get("aggregated_output"), str,
    ):
        raise ValueError("completed command lacks one literal aggregated_output")
    return "aggregated_output", item["aggregated_output"]


def _expected_command(
    cache: ReviewCache, command: str,
) -> tuple[str, list[dict[str, Any]], bool]:
    parts = _split_command(command)
    executable = parts[0]
    targets = _cache_targets(cache)
    rows: list[dict[str, Any]] = []
    citation_content = False
    if executable == "head":
        if (len(parts) != 4 or parts[1] != "-n"
                or re.fullmatch(r"[0-9]+", parts[2]) is None
                or int(parts[2]) <= 0):
            raise ValueError("head command is outside the approved complete-read grammar")
        token = _safe_relative(parts[3]).as_posix()
        target = targets.get(token)
        if target is None:
            raise ValueError("head command reads a non-review cache file")
        data = (cache.root / PurePosixPath(token)).read_bytes()
        text = data.decode("utf-8", errors="strict")
        lines = text.splitlines(keepends=True)
        count = int(parts[2])
        expected = "".join(lines[:count])
        if count < len(lines):
            raise ValueError("head command does not inspect the complete cached file")
        if target["target_kind"] == "source":
            rows.append({
                "kind": "file_read", "path": target["path"],
                "hash_kind": target["hash_kind"], "sha256": target["sha256"],
                "cache_sha256": target["cache_sha256"],
            })
        else:
            citation_content = True
            for entry in target["entries"]:
                rows.append({
                    "kind": "citation_inspect",
                    "requested_url": entry["requested_url"],
                    "reference_ids": entry["reference_ids"],
                    "body_sha256": entry["body_sha256"],
                    "inspection_sha256": entry["inspection_sha256"],
                    "parser_id": entry["parser_id"],
                    "parser_version": entry["parser_version"],
                    "parser_command_sha256": entry["parser_command_sha256"],
                    "body_stored": False,
                })
        return expected, rows, citation_content
    if executable == "sha256sum":
        if len(parts) < 2:
            raise ValueError("sha256sum command has no path")
        output: list[str] = []
        for raw in parts[1:]:
            token = _safe_relative(raw).as_posix()
            target = targets.get(token)
            if target is None:
                raise ValueError("sha256sum reads a non-review cache file")
            data = (cache.root / PurePosixPath(token)).read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            output.append(f"{digest}  {raw}\n")
            if target["target_kind"] == "source":
                rows.append({
                    "kind": "file_hash", "path": target["path"],
                    "hash_kind": target["hash_kind"], "sha256": target["sha256"],
                    "cache_sha256": target["cache_sha256"],
                })
            else:
                for entry in target["entries"]:
                    rows.append({
                        "kind": "citation_hash", "requested_url": entry["requested_url"],
                        "inspection_sha256": entry["inspection_sha256"],
                    })
        return "".join(output), rows, False
    raise ValueError(f"unapproved command executable: {parts[0]!r}")


def _citation_marker(output: str) -> str:
    encoded = output.encode("utf-8")
    return (
        "HLSGRAPH_REVIEW_CACHE_OUTPUT:"
        + hashlib.sha256(encoded).hexdigest() + f":{len(encoded)}\n"
    )


def sanitize_raw_review_stream(
    raw_bytes: bytes, cache: ReviewCache,
) -> bytes:
    """Validate CLI command output and redact cached citation text in memory."""

    events = _strict_jsonl(raw_bytes, label="raw Codex review stream")
    for event in events:
        if event.get("type") != "item.completed":
            continue
        item = event.get("item")
        if not isinstance(item, dict) or item.get("type") != "command_execution":
            continue
        if (item.get("status") != "completed"
                or type(item.get("exit_code")) is not int
                or item["exit_code"] != 0):
            raise ValueError("review command did not complete successfully")
        key, output = _command_output(item)
        expected, _rows, citation_content = _expected_command(
            cache, str(item.get("command", "")),
        )
        if output != expected:
            raise ValueError("review command output differs from deterministic cache replay")
        if citation_content:
            item[key] = _citation_marker(expected)
    sanitized = _canonical_jsonl(events)
    for entry in cache.manifest["citations"]:
        for path_key in ("body_path", "inspection_path"):
            relative = entry.get(path_key)
            if not relative:
                continue
            payload = _read_private_cache_file(cache.root, str(relative))
            needles = {payload}
            try:
                text = payload.decode("utf-8", errors="strict")
            except UnicodeDecodeError:
                pass
            else:
                needles.add(json.dumps(
                    text, ensure_ascii=False,
                )[1:-1].encode("utf-8"))
            if payload and any(needle and needle in sanitized for needle in needles):
                raise ValueError("sanitized raw review stream retains citation content")
    return sanitized


def _command_operations(
    cache: ReviewCache, command: str, output: str,
) -> list[dict[str, Any]]:
    expected, rows, citation_content = _expected_command(cache, command)
    required_output = _citation_marker(expected) if citation_content else expected
    if output != required_output:
        raise ValueError("stored command output differs from deterministic cache replay")
    return rows


def _review_result_issues(
    snapshot: ReviewSnapshot, cache: ReviewCache, result: dict[str, Any],
    operations: list[dict[str, Any]],
) -> list[str]:
    issues: list[str] = []
    expected_keys = {
        "protocol_id", "review_surface_sha256", "implementation_surface_sha256",
        "citation_audit_sha256", "citation_results", "approved", "issues", "summary",
    }
    if set(result) != expected_keys:
        issues.append("review result does not match the closed result contract")
    if result.get("protocol_id") != snapshot.protocol_id:
        issues.append("review result has the wrong protocol")
    if result.get("review_surface_sha256") != snapshot.surfaces:
        issues.append("review result has stale pack surfaces")
    if result.get("implementation_surface_sha256") != snapshot.implementation_surface_sha256:
        issues.append("review result has a stale implementation surface")
    if result.get("citation_audit_sha256") != snapshot.citation_audit_sha256:
        issues.append("review result has a stale citation audit")
    rows = result.get("citation_results")
    by_reference: dict[str, dict[str, Any]] = {}
    if not isinstance(rows, list):
        issues.append("review result citation_results is not an array")
        rows = []
    for row in rows:
        reference_id = row.get("reference_id") if isinstance(row, dict) else None
        if not isinstance(reference_id, str) or reference_id in by_reference:
            issues.append("review result has a duplicate or malformed citation row")
            continue
        by_reference[reference_id] = row
    references = {str(row["reference_id"]): row for row in _citation_reference_rows(snapshot)}
    if set(by_reference) != set(references):
        issues.append("review result citation inventory differs from the snapshot")
    inspected_urls = {
        str(row["requested_url"]) for row in operations
        if row.get("kind") == "citation_inspect"
    }
    inspected_files = {
        str(row["path"]) for row in operations if row.get("kind") == "file_read"
    }
    cache_by_url = {
        str(row["requested_url"]): row for row in cache.manifest["citations"]
    }
    all_verified = True
    for reference_id, expected in references.items():
        row = by_reference.get(reference_id)
        if row is None:
            all_verified = False
            continue
        url = str(expected["citation_url"])
        verified = (
            row.get("reference_surface_sha256") == expected.get("reference_surface_sha256")
            and row.get("verdict") == "verified"
            and row.get("exact_locator_inspected") is True
            and row.get("declared_version_matched") is True
            and row.get("issues") == []
            and url in inspected_urls
            and cache_by_url[url].get("available") is True
        )
        if expected.get("reference_kind") == "rule":
            verified = verified and all(
                row.get(key) is True for key in (
                    "declared_section_matched", "paraphrase_supported",
                    "applicability_not_broader",
                )
            )
        else:
            verified = verified and all(
                row.get(key) is None for key in (
                    "declared_section_matched", "paraphrase_supported",
                    "applicability_not_broader",
                )
            )
        all_verified = all_verified and verified
        if row.get("verdict") == "verified" and not verified:
            issues.append(f"verified citation lacks inspection evidence: {reference_id}")
    if result.get("approved") is True:
        missing_files = sorted(set(snapshot.file_map) - inspected_files)
        if result.get("issues") != [] or not all_verified or missing_files:
            issues.append("approved review has unresolved or uninspected evidence")
    elif result.get("approved") is not False:
        issues.append("review approved field is not boolean")
    if not isinstance(result.get("issues"), list) or not isinstance(result.get("summary"), str):
        issues.append("review result has malformed issues or summary")
    return issues


def replay_raw_review(
    root: Path, protocol_id: str, raw_bytes: bytes, *,
    snapshot: ReviewSnapshot | None = None, cache: ReviewCache | None = None,
) -> ReviewReplay:
    """Replay raw Codex JSONL and deterministically derive all public artifacts."""

    if protocol_id not in PROTOCOLS:
        raise ValueError(f"unknown knowledge-review protocol: {protocol_id}")
    snapshot = snapshot or freeze_review_snapshot(root, protocol_id)
    if cache is None:
        raise ValueError("raw review replay requires its retained frozen cache")
    if snapshot.protocol_id != protocol_id:
        raise ValueError("raw review replay uses a snapshot for another protocol")
    events = _strict_jsonl(raw_bytes, label="raw Codex review stream")
    event_types = [event.get("type") for event in events]
    if (len(events) < 4 or event_types[0] != "thread.started"
            or event_types[1] != "turn.started"
            or event_types[-1] != "turn.completed"):
        raise ValueError(
            "raw review stream must be one ordered thread/turn ending at turn.completed"
        )
    thread_ids: list[str] = []
    referenced_thread_ids = {
        str(event["thread_id"]) for event in events
        if isinstance(event.get("thread_id"), str) and event["thread_id"]
    }
    started_commands: dict[str, str] = {}
    completed_commands: set[str] = set()
    operations: list[dict[str, Any]] = []
    messages: list[str] = []
    turn_started = 0
    turn_completed = 0
    final_message_seen = False
    for index, event in enumerate(events, 1):
        event_type = event.get("type")
        if event_type not in _ALLOWED_EVENT_TYPES:
            raise ValueError(f"raw event {index} has forbidden or unknown type {event_type!r}")
        if event_type == "thread.started":
            thread_id = event.get("thread_id")
            if not isinstance(thread_id, str) or _TOKEN_RE.fullmatch(thread_id) is None:
                raise ValueError("raw review stream has an invalid thread ID")
            thread_ids.append(thread_id)
            continue
        if event_type == "turn.started":
            turn_started += 1
            continue
        if event_type == "turn.completed":
            turn_completed += 1
            continue
        item = event.get("item")
        if not isinstance(item, dict):
            raise ValueError(f"raw item event {index} has no object item")
        item_type = item.get("type")
        if item_type in _ALLOWED_NONCOMMAND_ITEMS:
            if event_type == "item.completed" and item_type == "agent_message":
                if final_message_seen:
                    raise ValueError("raw review stream has multiple final messages")
                text = _content_text(item)
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("completed agent message has no text")
                messages.append(text.strip())
                final_message_seen = True
            elif item_type == "agent_message":
                raise ValueError("agent message must be one completed final item")
            elif final_message_seen:
                raise ValueError("raw review stream has an item after the final message")
            continue
        if item_type != "command_execution":
            raise ValueError(f"raw item event {index} uses forbidden or unknown tool {item_type!r}")
        call_id = item.get("id") or item.get("call_id")
        if final_message_seen:
            raise ValueError("raw review stream executes a command after the final message")
        if not isinstance(call_id, str) or _CALL_ID_RE.fullmatch(call_id) is None:
            raise ValueError("command event has an invalid call ID")
        command = item.get("command")
        if not isinstance(command, str):
            raise ValueError("command event does not contain one literal command")
        if event_type == "item.started":
            if call_id in started_commands or call_id in completed_commands:
                raise ValueError("command call ID is reused")
            _split_command(command)
            started_commands[call_id] = command
            continue
        if call_id in completed_commands:
            raise ValueError("completed command call ID is reused")
        if call_id not in started_commands:
            raise ValueError("completed command has no matching start event")
        if started_commands.get(call_id) != command:
            raise ValueError("completed command differs from its start event")
        if (item.get("status") != "completed"
                or type(item.get("exit_code")) is not int
                or item["exit_code"] != 0):
            raise ValueError("review command did not complete successfully")
        _key, output = _command_output(item)
        operations.extend(_command_operations(cache, command, output))
        completed_commands.add(call_id)
    if len(thread_ids) != 1 or len(set(thread_ids)) != 1:
        raise ValueError("raw review stream must contain exactly one unique thread")
    if referenced_thread_ids and referenced_thread_ids != {thread_ids[0]}:
        raise ValueError("raw review stream mixes multiple thread identities")
    if turn_started != 1 or turn_completed != 1:
        raise ValueError("raw review stream must contain one completed turn")
    if set(started_commands) != completed_commands:
        raise ValueError("raw review stream has an incomplete command event")
    if len(messages) != 1:
        raise ValueError("raw review stream must contain exactly one final agent JSON")
    result = _strict_json_bytes(messages[0].encode("utf-8"), label="final review result")
    if not isinstance(result, dict):
        raise ValueError("final review result is not an object")
    result_issues = _review_result_issues(snapshot, cache, result, operations)
    if result_issues:
        raise ValueError("invalid final review result: " + "; ".join(result_issues))
    result_bytes = _canonical_json(result)
    raw_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    invocation_id = f"review-{protocol_id.rsplit('.', 2)[-2]}-{raw_sha256[:32]}"
    rows: list[dict[str, Any]] = []
    for sequence, operation in enumerate(operations, 1):
        rows.append({
            "schema_version": TRACE_SCHEMA_VERSION, "sequence": sequence,
            **operation,
        })
    rows.append({
        "schema_version": TRACE_SCHEMA_VERSION, "sequence": len(rows) + 1,
        "kind": "result_emit",
        "result_sha256": hashlib.sha256(result_bytes).hexdigest(),
    })
    return ReviewReplay(
        protocol_id=protocol_id, invocation_id=invocation_id,
        thread_id=thread_ids[0], raw_sha256=raw_sha256, result=result,
        result_bytes=result_bytes, trace_bytes=_canonical_jsonl(rows),
    )


def canonical_command_argv(protocol_id: str) -> list[str]:
    if protocol_id not in PROTOCOLS:
        raise ValueError(f"unknown knowledge-review protocol: {protocol_id}")
    values = ["$CODEX", "--strict-config", "-a", "never"]
    for feature in DISABLED_CODEX_FEATURES:
        values.extend(["--disable", feature])
    values.extend([
        "exec",
        "-c", f'default_permissions="{PERMISSION_PROFILE}"',
        "-c", f'permissions.{PERMISSION_PROFILE}.extends=":read-only"',
        "-c", f"permissions.{PERMISSION_PROFILE}.network.enabled=false",
        "-c", f"permissions.{PERMISSION_PROFILE}.filesystem=$OFFICIAL_BOUNDARY",
        "-c", 'web_search="disabled"',
        "--ignore-user-config", "--ignore-rules", "--ephemeral",
        "--json", "--color", "never", "--skip-git-repo-check", "--model",
        MODEL, "-c", f'model_reasoning_effort="{REASONING_EFFORT}"',
        "--output-schema", "$ROOT/" + REVIEW_SCHEMA_PATH,
        "--cd", "$CACHE", "-",
    ])
    return values


def command_contract_sha256(protocol_id: str) -> str:
    return hashlib.sha256(_canonical_json(canonical_command_argv(protocol_id))).hexdigest()


def build_receipt(
    root: Path, replay: ReviewReplay, *, snapshot: ReviewSnapshot,
    cache: ReviewCache, prompt: bytes,
) -> dict[str, Any]:
    files = PROTOCOL_FILES[replay.protocol_id]
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "protocol_id": replay.protocol_id,
        "invocation_id": replay.invocation_id,
        "thread_id": replay.thread_id,
        "model": MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "prompt_sha256": hashlib.sha256(prompt).hexdigest(),
        "output_schema_sha256": snapshot.output_schema_sha256,
        "review_snapshot_sha256": snapshot.sha256,
        "cache_manifest_sha256": cache.sha256,
        "result_sha256": hashlib.sha256(replay.result_bytes).hexdigest(),
        "command_sha256": command_contract_sha256(replay.protocol_id),
        "raw_event_stream_sha256": replay.raw_sha256,
        "event_stream_path": files["trace"],
        "event_stream_sha256": hashlib.sha256(replay.trace_bytes).hexdigest(),
        "codex_cli_version": CODEX_CLI_VERSION,
        "completed": True,
        "exit_code": 0,
    }


def _toml(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _toml_inline_table(value: dict[str, str]) -> str:
    return "{" + ",".join(
        f"{_toml(key)}={_toml(item)}" for key, item in sorted(value.items())
    ) + "}"


def _mount_fstype(path: Path) -> str:
    candidates: list[tuple[int, str]] = []
    for line in Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines():
        left, separator, right = line.partition(" - ")
        if not separator:
            continue
        fields = left.split()
        trailing = right.split()
        if len(fields) < 5 or not trailing:
            continue
        mount = Path(fields[4].replace("\\040", " "))
        try:
            path.relative_to(mount)
        except ValueError:
            continue
        candidates.append((len(mount.parts), trailing[0]))
    if not candidates:
        raise RuntimeError("cannot identify the review checkout filesystem")
    return max(candidates)[1]


def _official_boundary(
    root: Path, cache_root: Path | None = None,
    external_canary_root: Path | None = None,
) -> tuple[dict[str, str], list[str], dict[str, Any]]:
    if os.name == "nt" or sys.platform != "linux":
        raise RuntimeError("formal knowledge review is Linux/WSL2-only; Windows is NO-GO")
    from eval.agent_ab.common import (
        discover_windows_mount_roots, official_process_environment,
        require_official_linux_wsl2,
    )
    require_official_linux_wsl2()
    resolved = root.resolve(strict=True)
    if str(resolved).startswith("/mnt/") or _mount_fstype(resolved) != "ext4":
        raise RuntimeError("formal review checkout must be on WSL ext4, not drvfs/overlay")
    codex_home_text = os.environ.get("CODEX_HOME", "")
    if not codex_home_text:
        raise RuntimeError("formal review requires a dedicated CODEX_HOME")
    codex_home = Path(codex_home_text).resolve(strict=True)
    if resolved == codex_home or resolved.is_relative_to(codex_home):
        raise RuntimeError("review checkout must be disjoint from CODEX_HOME")
    home_roots = {"/home", "/root"}
    if Path("/home").is_dir():
        home_roots.update(str(path.resolve()) for path in Path("/home").iterdir())
    if any(
        resolved == Path(path) or resolved.is_relative_to(Path(path))
        for path in home_roots
    ):
        raise RuntimeError("formal review checkout must be outside every home root")
    if cache_root is None:
        raise RuntimeError("formal review requires an external frozen cache")
    resolved_cache = cache_root.resolve(strict=True)
    try:
        resolved_cache.relative_to(resolved)
    except ValueError:
        pass
    else:
        raise RuntimeError("formal review cache must be outside the checkout")
    if _mount_fstype(resolved_cache) != "ext4":
        raise RuntimeError("formal review cache must live on WSL ext4")
    deny_roots = {
        str(codex_home), *home_roots, "/media", "/run/media",
        *(str(path) for path in discover_windows_mount_roots()),
    }
    deny_roots.add(str(resolved))
    if resolved_cache.parent != Path("/"):
        deny_roots.add(str(resolved_cache.parent))
    if external_canary_root is not None:
        canary_root = external_canary_root.resolve(strict=True)
        if canary_root == resolved or canary_root.is_relative_to(resolved):
            raise RuntimeError("external review canary must be outside the checkout")
        deny_roots.add(str(canary_root))
    filesystem = {path: "deny" for path in sorted(deny_roots)}
    filesystem[str(resolved_cache)] = "read"
    profile_values = [
        f"default_permissions={_toml(PERMISSION_PROFILE)}",
        f"permissions.{PERMISSION_PROFILE}.extends={_toml(':read-only')}",
        f"permissions.{PERMISSION_PROFILE}.network.enabled=false",
        f"permissions.{PERMISSION_PROFILE}.filesystem=" + _toml_inline_table(filesystem),
        'web_search="disabled"',
    ]
    return official_process_environment(), profile_values, {
        "codex_home": str(codex_home),
        "checkout_root": str(resolved),
        "cache_root": str(resolved_cache),
        "deny_roots": sorted(deny_roots),
        "drvfs_roots": [
            str(path.resolve()) for path in discover_windows_mount_roots()
        ],
        "home_roots": sorted(home_roots),
        "external_canary_root": (
            str(external_canary_root.resolve())
            if external_canary_root is not None else None
        ),
    }


def _sandbox_prefix(
    codex: str, root: Path, profile_values: list[str],
) -> list[str]:
    command = [codex]
    for value in profile_values:
        command.extend(["-c", value])
    command.extend([
        "sandbox", "-P", PERMISSION_PROFILE,
        "--sandbox-state-disable-network", "-C", str(root.resolve()),
    ])
    return command


def _run_canary(command: list[str], environment: dict[str, str]) -> int:
    completed = subprocess.run(
        command, env=environment, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, timeout=20, check=False,
    )
    return completed.returncode


def _verify_boundary_canaries(
    *, codex: str, root: Path, cache_root: Path, profile_values: list[str],
    boundary: dict[str, Any], environment: dict[str, str],
) -> None:
    """Prove allowed-read plus denied-home/drvfs/write before model execution."""

    prefix = _sandbox_prefix(codex, cache_root, profile_values)
    python = shutil.which("python3", path=environment.get("PATH"))
    if not python:
        raise RuntimeError("official review boundary canary requires python3")
    read_script = "import pathlib,sys;pathlib.Path(sys.argv[1]).read_bytes()"
    list_script = "import os,sys;list(os.scandir(sys.argv[1]))"
    allowed = cache_root / CACHE_MANIFEST_NAME
    if not allowed.is_file() or _run_canary(
        [*prefix, python, "-I", "-c", read_script, str(allowed)], environment,
    ) != 0:
        raise RuntimeError("review sandbox cannot read its frozen review cache")
    denied_targets: list[tuple[str, str]] = []
    codex_auth = Path(boundary["codex_home"]) / "auth.json"
    if not codex_auth.is_file():
        raise RuntimeError("dedicated CODEX_HOME lacks auth.json for denial canary")
    denied_targets.append(("file", str(codex_auth)))
    denied_targets.append(("file", str(root / "README.md")))
    external = Path(str(boundary["external_canary_root"])) / "canary.txt"
    denied_targets.append(("file", str(external)))
    denied_targets.extend(
        ("directory", path)
        for path in [*boundary["drvfs_roots"], *boundary["home_roots"]]
        if Path(path).is_dir()
    )
    for kind, target in denied_targets:
        script = read_script if kind == "file" else list_script
        if _run_canary(
            [*prefix, python, "-I", "-c", script, target], environment,
        ) == 0:
            raise RuntimeError(f"review sandbox failed to deny {kind}: {target}")
    write_target = cache_root / ".hlsgraph-review-write-canary"
    write_script = (
        "import pathlib,sys;pathlib.Path(sys.argv[1]).write_bytes(b'x')"
    )
    try:
        if _run_canary(
            [*prefix, python, "-I", "-c", write_script, str(write_target)],
            environment,
        ) == 0:
            raise RuntimeError("review sandbox permits checkout writes")
    finally:
        if write_target.exists():
            write_target.unlink()


def _actual_command(
    root: Path, cache_root: Path, protocol_id: str, codex_command: str,
    profile_values: list[str],
) -> list[str]:
    command = [codex_command, "--strict-config", "-a", "never"]
    for feature in DISABLED_CODEX_FEATURES:
        command.extend(["--disable", feature])
    command.append("exec")
    for value in profile_values:
        command.extend(["-c", value])
    command.extend([
        "--ignore-user-config", "--ignore-rules", "--ephemeral", "--json",
        "--color", "never", "--skip-git-repo-check", "--model", MODEL,
        "-c", f'model_reasoning_effort="{REASONING_EFFORT}"',
        "--output-schema", str((root / REVIEW_SCHEMA_PATH).resolve()),
        "--cd", str(cache_root.resolve()), "-",
    ])
    return command


def _terminate(process: subprocess.Popen[bytes]) -> None:
    try:
        if os.name == "nt":
            process.terminate()
        else:
            os.killpg(process.pid, signal.SIGTERM)
    except (OSError, ProcessLookupError):
        return


def _redact_cache_payloads(data: bytes, cache: ReviewCache) -> bytes:
    result = data
    for entry in cache.manifest["citations"]:
        for key in ("body_path", "inspection_path"):
            relative = entry.get(key)
            if not relative:
                continue
            payload = (cache.root / PurePosixPath(str(relative))).read_bytes()
            if payload and payload in result:
                marker = (
                    f"[redacted-cache-{hashlib.sha256(payload).hexdigest()}]"
                ).encode("ascii")
                result = result.replace(payload, marker)
    return result


def _publish_artifacts(root: Path, artifacts: dict[str, bytes]) -> None:
    targets = {relative: root / PurePosixPath(relative) for relative in artifacts}
    existing = [relative for relative, path in targets.items() if path.exists()]
    if existing:
        raise RuntimeError(f"refusing to overwrite review artifacts: {sorted(existing)!r}")
    docs = root / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    published: list[Path] = []
    with tempfile.TemporaryDirectory(prefix=".hlsgraph-review-stage-", dir=docs) as temp:
        staging = Path(temp).resolve(strict=True)
        staging.relative_to(root.resolve(strict=True))
        staged: dict[str, Path] = {}
        for index, (relative, data) in enumerate(sorted(artifacts.items())):
            path = staging / f"{index:02d}.artifact"
            path.write_bytes(data)
            staged[relative] = path
        try:
            for relative in sorted(staged):
                target = targets[relative]
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(staged[relative], target)
                published.append(target)
        except BaseException:
            for path in published:
                try:
                    path.unlink()
                except OSError:
                    pass
            raise


def preflight_review(root: Path, protocol_id: str) -> dict[str, Any]:
    """Build the real frozen input contract without network or model execution."""

    resolved = root.resolve(strict=True)
    if resolved != SCRIPT_ROOT:
        raise RuntimeError("review runner must execute from its own checkout root")
    snapshot = freeze_review_snapshot(resolved, protocol_id)
    prompt = build_review_prompt(resolved, protocol_id, snapshot=snapshot)
    return {
        "protocol_id": protocol_id,
        "review_snapshot_sha256": snapshot.sha256,
        "prompt_contract_sha256": hashlib.sha256(prompt).hexdigest(),
        "review_surface_sha256": snapshot.surfaces,
        "implementation_surface_sha256": snapshot.implementation_surface_sha256,
        "citation_audit_sha256": snapshot.citation_audit_sha256,
        "output_schema_sha256": snapshot.output_schema_sha256,
        "required_file_count": len(snapshot.files),
        "exact_citation_url_count": len(snapshot.exact_citation_urls),
        "network_used": False,
        "model_used": False,
    }


def run_review(
    root: Path, protocol_id: str, raw_output: Path, cache_root: Path, *,
    codex_command: str, timeout_seconds: int,
    fetcher: Callable[[str, float, int], TrustedFetch] = _default_fetch,
    fetch_timeout_seconds: float = 60.0,
    pdf_text_extractor: Callable[[bytes], TextDerivation | None] | None = None,
    pdftotext_command: str | None = None,
) -> ReviewReplay:
    """Execute one review and atomically derive its three public artifacts."""

    lexical_root = root.absolute()
    if lexical_root.is_symlink():
        raise RuntimeError("formal review checkout must not be a symlink")
    root = lexical_root.resolve(strict=True)
    if root != SCRIPT_ROOT:
        raise RuntimeError("formal review runner must belong to the reviewed checkout")
    raw_output = raw_output.absolute()
    cache_root = cache_root.absolute()
    stderr_path = raw_output.with_suffix(raw_output.suffix + ".stderr.log")
    resolved_raw = _resolved_unlinked_path(raw_output, label="raw Codex stream")
    resolved_stderr = _resolved_unlinked_path(stderr_path, label="review stderr")
    resolved_cache = _resolved_unlinked_path(cache_root, label="review cache")
    try:
        resolved_raw.relative_to(root)
    except ValueError:
        pass
    else:
        raise RuntimeError("raw Codex stream must stay outside the public checkout")
    try:
        resolved_cache.relative_to(root)
    except ValueError:
        pass
    else:
        raise RuntimeError("review cache must stay outside the public checkout")
    for evidence_path, label in (
        (resolved_raw, "raw Codex stream"),
        (resolved_stderr, "review stderr"),
    ):
        try:
            evidence_path.relative_to(resolved_cache)
        except ValueError:
            pass
        else:
            raise RuntimeError(f"{label} must stay outside the review cache")
        try:
            resolved_cache.relative_to(evidence_path)
        except ValueError:
            pass
        else:
            raise RuntimeError(f"review cache must stay outside the {label} path")
    if raw_output.exists():
        raise RuntimeError("raw Codex stream path already exists")
    if cache_root.exists():
        raise RuntimeError("review cache path already exists")
    if stderr_path.exists():
        raise RuntimeError("review stderr path already exists")
    canary_handle = tempfile.TemporaryDirectory(
        prefix="hlsgraph-knowledge-review-boundary-", dir="/tmp",
    )
    canary_root = Path(canary_handle.name)
    canary_path = canary_root / "canary.txt"
    canary_bytes = os.urandom(48)
    canary_path.write_bytes(canary_bytes)
    try:
        from eval.agent_ab.common import (
            _resolve_executable, official_process_environment,
            require_official_linux_wsl2,
        )
        require_official_linux_wsl2()
        environment = official_process_environment()
        resolved_codex = str(_resolve_executable(codex_command, "Codex CLI"))
        version = subprocess.run(
            [resolved_codex, "--version"], env=environment, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15,
        ).stdout.decode("utf-8", errors="strict").strip()
        if version != CODEX_CLI_VERSION:
            raise RuntimeError(
                f"formal review requires {CODEX_CLI_VERSION!r}, found {version!r}"
            )
        clean = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=str(root), env=environment, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, timeout=30, check=False,
        )
        if clean.returncode != 0 or clean.stdout:
            raise RuntimeError("formal review checkout must be a clean committed candidate")
        snapshot = freeze_review_snapshot(root, protocol_id)
        cache = create_review_cache(
            root, snapshot, cache_root, fetcher=fetcher,
            timeout_seconds=fetch_timeout_seconds,
            pdf_text_extractor=pdf_text_extractor,
            pdftotext_command=pdftotext_command,
        )
        cache = load_review_cache(cache.root, snapshot)
        environment, profile, boundary = _official_boundary(
            root, cache.root, canary_root,
        )
        _verify_boundary_canaries(
            codex=resolved_codex, root=root, cache_root=cache.root,
            profile_values=profile,
            boundary=boundary, environment=environment,
        )
        command = _actual_command(
            root, cache.root, protocol_id, resolved_codex, profile,
        )
        prompt = build_review_prompt(
            root, protocol_id, snapshot=snapshot, cache=cache,
        )
        process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, cwd=str(cache.root), env=environment,
            start_new_session=True,
        )
        try:
            stdout_bytes, stderr_bytes = process.communicate(
                prompt, timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            _terminate(process)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
            raise RuntimeError("knowledge review timed out") from exc
        stderr_bytes = _redact_cache_payloads(stderr_bytes, cache)
        _ensure_private_parent(raw_output.parent)
        if process.returncode != 0:
            _write_private(stderr_path, stderr_bytes)
            raise RuntimeError(
                f"Codex review failed with exit code {process.returncode}; see {stderr_path}"
            )
        sanitized_raw = sanitize_raw_review_stream(stdout_bytes, cache)
        post_snapshot = freeze_review_snapshot(root, protocol_id)
        post_cache = load_review_cache(cache.root, post_snapshot)
        post_prompt = build_review_prompt(
            root, protocol_id, snapshot=post_snapshot, cache=post_cache,
        )
        if (canary_path.read_bytes() != canary_bytes
                or post_snapshot != snapshot or post_cache.manifest_bytes != cache.manifest_bytes
                or post_prompt != prompt):
            raise RuntimeError("review boundary or source bytes changed during invocation")
        replay = replay_raw_review(
            root, protocol_id, sanitized_raw, snapshot=snapshot, cache=cache,
        )
        _write_private(raw_output, sanitized_raw)
        _write_private(stderr_path, stderr_bytes)
        files = PROTOCOL_FILES[protocol_id]
        artifacts = {
            files["result"]: replay.result_bytes,
            files["trace"]: replay.trace_bytes,
        }
        receipt = build_receipt(
            root, replay, snapshot=snapshot, cache=cache, prompt=prompt,
        )
        artifacts[files["receipt"]] = _canonical_json(receipt)
        _publish_artifacts(root, artifacts)
        return replay
    finally:
        canary_handle.cleanup()


def _receipt_projection(receipt: dict[str, Any], receipt_bytes: bytes) -> dict[str, Any]:
    keys = (
        "protocol_id", "invocation_id", "thread_id", "model",
        "reasoning_effort", "prompt_sha256", "output_schema_sha256",
        "review_snapshot_sha256", "cache_manifest_sha256",
        "result_sha256", "command_sha256", "event_stream_path",
        "raw_event_stream_sha256", "event_stream_sha256", "codex_cli_version",
    )
    result = {key: receipt.get(key) for key in keys}
    result["cli_receipt_sha256"] = hashlib.sha256(receipt_bytes).hexdigest()
    return result


def review_source_hashes(
    root: Path, surfaces: dict[str, str], implementation_sha256: str,
) -> dict[str, str]:
    required = {
        REVIEW_SCHEMA_PATH, REVIEW_RECEIPT_SCHEMA_PATH,
        CITATION_AUDIT_PATH, CITATION_GENERATOR_PATH, RUNNER_PATH,
        SURFACE_HELPER_PATH, RELEASE_AUDITOR_PATH,
        *(item[key] for item in PROTOCOL_FILES.values()
          for key in ("prompt", "result", "trace", "receipt")),
    }
    hashes = {
        relative: hashlib.sha256(
            (root / PurePosixPath(relative)).read_bytes()
        ).hexdigest()
        for relative in sorted(required)
    }
    helper_hash = hashes.pop(SURFACE_HELPER_PATH)
    hashes[SURFACE_HELPER_HASH_KEY] = helper_hash
    hashes[IMPLEMENTATION_SURFACE_HASH_KEY] = implementation_sha256
    pack_root = root / "src" / "hlsgraph" / "knowledge" / "packs"
    for path in sorted(pack_root.glob("*.json")):
        data = path.read_bytes()
        pack_id, surface, _payload = _semantic_pack_projection(
            data, label=f"knowledge pack {path.name}",
        )
        if surfaces.get(pack_id) != surface:
            raise ValueError(f"knowledge pack surface changed before sealing: {pack_id}")
        hashes[
            PACK_SURFACE_HASH_PREFIX + path.name + PACK_SURFACE_HASH_SUFFIX
        ] = surface
    return hashes


def _atomic_replace_pack_bytes(root: Path, updates: dict[Path, bytes]) -> None:
    pack_root = root / "src" / "hlsgraph" / "knowledge" / "packs"
    originals = {path: path.read_bytes() for path in updates}
    with tempfile.TemporaryDirectory(prefix=".hlsgraph-seal-stage-", dir=pack_root) as temp:
        stage_root = Path(temp).resolve(strict=True)
        stage_root.relative_to(root.resolve(strict=True))
        staged: dict[Path, Path] = {}
        for index, (target, data) in enumerate(sorted(
            updates.items(), key=lambda item: item[0].name,
        )):
            stage = stage_root / f"{index:02d}.json"
            stage.write_bytes(data)
            staged[target] = stage
        replaced: list[Path] = []
        try:
            for target in sorted(staged, key=lambda item: item.name):
                os.replace(staged[target], target)
                replaced.append(target)
        except BaseException:
            for target in replaced:
                recovery = stage_root / (target.name + ".recovery")
                recovery.write_bytes(originals[target])
                os.replace(recovery, target)
            raise


def seal_review_attestations(
    root: Path, *, semantic_raw: Path, adversarial_raw: Path,
    semantic_cache: Path, adversarial_cache: Path,
) -> None:
    """Verify two retained invocations and deterministically seal all three packs."""

    root = root.resolve(strict=True)
    if root != SCRIPT_ROOT:
        raise RuntimeError("review sealer must belong to the reviewed checkout")
    invocations: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    snapshots: list[ReviewSnapshot] = []
    inputs = (
        (SEMANTIC_PROTOCOL, semantic_raw, semantic_cache),
        (ADVERSARIAL_PROTOCOL, adversarial_raw, adversarial_cache),
    )
    for protocol_id, raw_path, cache_path in inputs:
        snapshot = freeze_review_snapshot(root, protocol_id)
        cache = load_review_cache(cache_path, snapshot)
        prompt = build_review_prompt(
            root, protocol_id, snapshot=snapshot, cache=cache,
        )
        raw_bytes = raw_path.read_bytes()
        replay = replay_raw_review(
            root, protocol_id, raw_bytes, snapshot=snapshot, cache=cache,
        )
        files = PROTOCOL_FILES[protocol_id]
        if (root / files["result"]).read_bytes() != replay.result_bytes:
            raise ValueError(f"{protocol_id} result differs from retained raw replay")
        if (root / files["trace"]).read_bytes() != replay.trace_bytes:
            raise ValueError(f"{protocol_id} trace differs from retained raw replay")
        receipt_path = root / files["receipt"]
        receipt_bytes = receipt_path.read_bytes()
        receipt = _strict_json_bytes(receipt_bytes, label=f"{protocol_id} receipt")
        expected_receipt = build_receipt(
            root, replay, snapshot=snapshot, cache=cache, prompt=prompt,
        )
        if not isinstance(receipt, dict) or receipt != expected_receipt:
            raise ValueError(f"{protocol_id} receipt differs from deterministic replay")
        if replay.result.get("approved") is not True or replay.result.get("issues") != []:
            raise ValueError(f"{protocol_id} is not an approved issue-free review")
        invocations.append(_receipt_projection(receipt, receipt_bytes))
        results.append(replay.result)
        snapshots.append(snapshot)
    if (invocations[0]["invocation_id"] == invocations[1]["invocation_id"]
            or invocations[0]["thread_id"] == invocations[1]["thread_id"]
            or invocations[0]["raw_event_stream_sha256"]
            == invocations[1]["raw_event_stream_sha256"]):
        raise ValueError("semantic and adversarial reviews are not independent")
    if (snapshots[0].surfaces != snapshots[1].surfaces
            or snapshots[0].implementation_surface_sha256
            != snapshots[1].implementation_surface_sha256):
        raise ValueError("review invocations do not bind one common semantic surface")
    if sorted(
        results[0]["citation_results"], key=lambda row: row["reference_id"],
    ) != sorted(
        results[1]["citation_results"], key=lambda row: row["reference_id"],
    ):
        raise ValueError("semantic and adversarial citation verdicts disagree")
    source_hashes = review_source_hashes(
        root, snapshots[0].surfaces,
        snapshots[0].implementation_surface_sha256,
    )
    invocations = sorted(
        invocations, key=lambda item: (item["protocol_id"], item["invocation_id"]),
    )
    reviewers = sorted(
        f"{item['model']}@{item['reasoning_effort']}#{item['invocation_id']}"
        for item in invocations
    )
    evidence = {
        "independent_invocations": True,
        "same_model_repeated_review": True,
        "distinct_model_families": False,
        "citation_verified": True,
        "review_agreement": True,
        "unresolved_conflicts": False,
        "review_invocations": invocations,
    }
    updates: dict[Path, bytes] = {}
    pack_root = root / "src" / "hlsgraph" / "knowledge" / "packs"
    for path in sorted(pack_root.glob("*.json")):
        original = path.read_bytes()
        value = _strict_json_bytes(original, label=f"knowledge pack {path.name}")
        if not isinstance(value, dict):
            raise ValueError(f"knowledge pack is not an object: {path.name}")
        before_id, before_surface, _payload = _semantic_pack_projection(
            original, label=f"knowledge pack {path.name}",
        )
        metadata = value.get("metadata")
        coverage = value.get("coverage")
        if not isinstance(metadata, dict) or not isinstance(coverage, dict):
            raise ValueError(f"knowledge pack lacks review fields: {before_id}")
        metadata["review_status"] = "machine_repeated_reviewed"
        coverage.update({
            "review_status": "machine_repeated_reviewed",
            "reviewers": reviewers,
            "source_hashes": source_hashes,
            "review_evidence": evidence,
        })
        encoded = _canonical_json(value)
        after_id, after_surface, _payload = _semantic_pack_projection(
            encoded, label=f"sealed knowledge pack {path.name}",
        )
        if (after_id, after_surface) != (before_id, before_surface):
            raise RuntimeError(f"sealing changed semantic pack surface: {before_id}")
        updates[path] = encoded
    _atomic_replace_pack_bytes(root, updates)
    for snapshot in snapshots:
        sealed = freeze_review_snapshot(root, snapshot.protocol_id)
        if sealed != snapshot:
            raise RuntimeError("sealed pack attestations changed the frozen review snapshot")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    preflight = commands.add_parser("preflight", help="freeze inputs without network/model")
    preflight.add_argument("--root", type=Path, default=Path.cwd())
    preflight.add_argument("--protocol", choices=sorted(PROTOCOLS), required=True)
    review = commands.add_parser("review", help="run one formal cached review")
    review.add_argument("--root", type=Path, default=Path.cwd())
    review.add_argument("--protocol", choices=sorted(PROTOCOLS), required=True)
    review.add_argument("--raw-output", type=Path, required=True)
    review.add_argument("--cache-root", type=Path, required=True)
    review.add_argument("--codex-command", default="codex")
    review.add_argument("--timeout-seconds", type=int, default=3600)
    review.add_argument("--fetch-timeout-seconds", type=float, default=60.0)
    review.add_argument("--pdftotext-command")
    seal = commands.add_parser("seal", help="verify two reviews and seal pack attestations")
    seal.add_argument("--root", type=Path, default=Path.cwd())
    seal.add_argument("--semantic-raw", type=Path, required=True)
    seal.add_argument("--adversarial-raw", type=Path, required=True)
    seal.add_argument("--semantic-cache", type=Path, required=True)
    seal.add_argument("--adversarial-cache", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.operation == "preflight":
        print(json.dumps(
            preflight_review(args.root, args.protocol), indent=2,
            sort_keys=True, ensure_ascii=False,
        ))
    elif args.operation == "review":
        run_review(
            args.root, args.protocol, args.raw_output, args.cache_root,
            codex_command=args.codex_command, timeout_seconds=args.timeout_seconds,
            fetch_timeout_seconds=args.fetch_timeout_seconds,
            pdftotext_command=args.pdftotext_command,
        )
    elif args.operation == "seal":
        seal_review_attestations(
            args.root, semantic_raw=args.semantic_raw,
            adversarial_raw=args.adversarial_raw,
            semantic_cache=args.semantic_cache,
            adversarial_cache=args.adversarial_cache,
        )
    else:  # pragma: no cover - argparse owns the closed operation set.
        raise ValueError(f"unknown operation: {args.operation}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
