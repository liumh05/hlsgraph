"""Loading, filtering, and metadata-only indexing for knowledge packs."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import re
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import urlparse

from hlsgraph.model import KnowledgeRule, require_namespaced


PACK_SCHEMA_VERSION = "1.0"
LOCAL_INDEX_SCHEMA_VERSION = "1.0"

# These names indicate copied or extracted document material.  Knowledge packs
# and local indexes are intentionally citation/metadata-only.
_FORBIDDEN_CONTENT_FIELDS = frozenset({
    "body",
    "chunks",
    "content",
    "document_text",
    "embedding",
    "extracted_text",
    "full_text",
    "page_text",
    "pages",
    "pdf_base64",
    "raw_text",
    "text",
})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class KnowledgePackError(ValueError):
    """Raised when a pack or local metadata index violates its contract."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _require_https(value: str, field_name: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.netloc:
        raise KnowledgePackError(f"{field_name} must be an absolute HTTPS URL")
    return value


def _reject_embedded_content(value: Any, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().casefold().replace("-", "_")
            if normalized in _FORBIDDEN_CONTENT_FIELDS:
                raise KnowledgePackError(
                    f"{path}.{key} is not allowed: indexes and packs are metadata/citation-only"
                )
            _reject_embedded_content(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_embedded_content(item, f"{path}[{index}]")


@dataclass(frozen=True, slots=True)
class DocumentReference:
    """Public metadata for a referenced document; no document body is stored."""

    document_id: str
    document_version: str
    title: str
    official_url: str
    publisher: str
    kind: str = "guide"
    license_note: str | None = None

    def __post_init__(self) -> None:
        require_namespaced(self.document_id, "document_id")
        if not self.document_version.strip():
            raise KnowledgePackError("document_version is required")
        if not self.title.strip() or not self.publisher.strip():
            raise KnowledgePackError("document title and publisher are required")
        _require_https(self.official_url, "official_url")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DocumentReference":
        allowed = {item.name for item in fields(cls)}
        unknown = set(value) - allowed
        if unknown:
            raise KnowledgePackError(f"unknown document metadata fields: {sorted(unknown)}")
        return cls(**dict(value))


@dataclass(slots=True)
class KnowledgePack:
    """A validated collection of short rules tied to declared references."""

    schema_version: str
    pack_id: str
    title: str
    license: str
    documents: list[DocumentReference] = field(default_factory=list)
    rules: list[KnowledgeRule] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != PACK_SCHEMA_VERSION:
            raise KnowledgePackError(
                f"unsupported knowledge pack schema {self.schema_version!r}; "
                f"expected {PACK_SCHEMA_VERSION!r}"
            )
        require_namespaced(self.pack_id, "pack_id")
        if not self.title.strip() or not self.license.strip():
            raise KnowledgePackError("pack title and license are required")
        declared: dict[tuple[str, str], DocumentReference] = {}
        for document in self.documents:
            key = (document.document_id, document.document_version)
            if key in declared:
                raise KnowledgePackError(f"duplicate document reference: {key}")
            declared[key] = document
        seen_rules: set[str] = set()
        for rule in self.rules:
            require_namespaced(rule.document_id, "rule document_id")
            require_namespaced(rule.rule_id, "rule_id")
            if (rule.document_id, rule.document_version) not in declared:
                raise KnowledgePackError(
                    f"rule {rule.rule_id!r} cites an undeclared document version"
                )
            if rule.id in seen_rules:
                raise KnowledgePackError(f"duplicate knowledge rule: {rule.id}")
            seen_rules.add(rule.id)
            if not rule.section.strip():
                raise KnowledgePackError(f"rule {rule.rule_id!r} requires a section reference")
            _require_https(rule.citation_url, "citation_url")
            if rule.summary is None or not rule.summary.strip():
                raise KnowledgePackError(f"rule {rule.rule_id!r} requires a short paraphrase")
            if len(rule.summary) > 500:
                raise KnowledgePackError(
                    f"rule {rule.rule_id!r} summary exceeds the 500-character paraphrase limit"
                )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "KnowledgePack":
        _reject_embedded_content(value)
        allowed = {item.name for item in fields(cls)}
        unknown = set(value) - allowed
        if unknown:
            raise KnowledgePackError(f"unknown pack fields: {sorted(unknown)}")
        data = dict(value)
        data["documents"] = [DocumentReference.from_dict(item)
                             for item in data.get("documents", [])]
        try:
            data["rules"] = [KnowledgeRule(**dict(item)) for item in data.get("rules", [])]
        except TypeError as exc:
            raise KnowledgePackError(f"invalid knowledge rule: {exc}") from exc
        return cls(**data)


def _load_pack_mapping(value: Mapping[str, Any]) -> KnowledgePack:
    try:
        return KnowledgePack.from_dict(value)
    except (TypeError, KeyError, ValueError) as exc:
        if isinstance(exc, KnowledgePackError):
            raise
        raise KnowledgePackError(f"invalid knowledge pack: {exc}") from exc


def load_pack(source: str | Path | Mapping[str, Any]) -> KnowledgePack:
    """Load and validate one JSON pack or an already-decoded mapping."""

    if isinstance(source, Mapping):
        return _load_pack_mapping(source)
    path = Path(source)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise KnowledgePackError(f"cannot load knowledge pack {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise KnowledgePackError("knowledge pack root must be a JSON object")
    return _load_pack_mapping(value)


def load_builtin_packs() -> list[KnowledgePack]:
    """Load all packs distributed with HLSGraph in deterministic filename order."""

    root = resources.files("hlsgraph.knowledge").joinpath("packs")
    packs: list[KnowledgePack] = []
    for item in sorted(root.iterdir(), key=lambda entry: entry.name):
        if not item.name.endswith(".json"):
            continue
        try:
            value = json.loads(item.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise KnowledgePackError(f"invalid built-in pack {item.name}: {exc}") from exc
        if not isinstance(value, Mapping):
            raise KnowledgePackError(f"built-in pack {item.name} must contain an object")
        packs.append(_load_pack_mapping(value))
    return packs


def _version_key(value: Any) -> tuple[tuple[int, Any], ...]:
    parts = re.findall(r"\d+|[A-Za-z]+", str(value))
    return tuple((0, int(part)) if part.isdigit() else (1, part.casefold()) for part in parts)


def _same(left: Any, right: Any) -> bool:
    if isinstance(left, str) and isinstance(right, str):
        return left.casefold() == right.casefold()
    return left == right


def _matches_constraint(constraint: Any, actual: Any) -> bool:
    if constraint in (None, "*"):
        return True
    if actual is None:
        return False
    if isinstance(constraint, list):
        return any(_same(item, actual) for item in constraint)
    if isinstance(constraint, Mapping):
        allowed = {"equals", "one_of", "min_version", "max_version"}
        unknown = set(constraint) - allowed
        if unknown:
            raise KnowledgePackError(f"unsupported applicability operators: {sorted(unknown)}")
        if "equals" in constraint and not _same(constraint["equals"], actual):
            return False
        if "one_of" in constraint:
            choices = constraint["one_of"]
            if not isinstance(choices, list) or not any(_same(item, actual) for item in choices):
                return False
        actual_version = _version_key(actual)
        if "min_version" in constraint and actual_version < _version_key(constraint["min_version"]):
            return False
        if "max_version" in constraint and actual_version > _version_key(constraint["max_version"]):
            return False
        return True
    return _same(constraint, actual)


def matches_applicability(rule: KnowledgeRule, context: Mapping[str, Any]) -> bool:
    """Return whether every constraint declared by ``rule`` is met.

    A missing context value does not satisfy a restrictive rule.  This fail-closed
    behavior prevents a tool- or stage-specific rule from being presented as
    generally applicable.  ``"*"`` is the only unrestricted value.
    """

    return all(_matches_constraint(constraint, context.get(key))
               for key, constraint in rule.applicability.items())


def filter_rules(
    rules: Iterable[KnowledgeRule],
    *,
    document_id: str | None = None,
    document_version: str | None = None,
    applicability: Mapping[str, Any] | None = None,
) -> list[KnowledgeRule]:
    """Filter rules by exact document identity and optional applicability context."""

    result = [rule for rule in rules
              if (document_id is None or rule.document_id == document_id)
              and (document_version is None or rule.document_version == document_version)
              and (applicability is None or matches_applicability(rule, applicability))]
    return sorted(result, key=lambda rule: (rule.document_id, rule.document_version, rule.rule_id))


@dataclass(slots=True)
class KnowledgeCatalog:
    packs: list[KnowledgePack]

    @classmethod
    def builtin(cls) -> "KnowledgeCatalog":
        return cls(load_builtin_packs())

    def all_rules(self) -> list[KnowledgeRule]:
        return [rule for pack in self.packs for rule in pack.rules]

    def filter(
        self,
        *,
        document_id: str | None = None,
        document_version: str | None = None,
        applicability: Mapping[str, Any] | None = None,
    ) -> list[KnowledgeRule]:
        return filter_rules(self.all_rules(), document_id=document_id,
                            document_version=document_version, applicability=applicability)


@dataclass(frozen=True, slots=True)
class LocalDocumentMetadata:
    """Metadata for a user-owned local document; never contains extracted text."""

    document_id: str
    document_version: str
    uri: str
    sha256: str
    size: int
    modified_ns: int
    indexed_at: str
    title: str | None = None
    media_type: str | None = None
    official_url: str | None = None

    def __post_init__(self) -> None:
        require_namespaced(self.document_id, "document_id")
        if not self.document_version.strip() or not self.uri.strip():
            raise KnowledgePackError("local document version and URI are required")
        if not _SHA256.fullmatch(self.sha256):
            raise KnowledgePackError("local document sha256 must be lowercase hexadecimal")
        if self.size < 0 or self.modified_ns < 0:
            raise KnowledgePackError("local document size and modified timestamp must be non-negative")
        if self.official_url:
            _require_https(self.official_url, "official_url")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LocalDocumentMetadata":
        _reject_embedded_content(value)
        allowed = {item.name for item in fields(cls)}
        unknown = set(value) - allowed
        if unknown:
            raise KnowledgePackError(f"unknown local document metadata fields: {sorted(unknown)}")
        return cls(**dict(value))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def index_local_document(
    path: str | Path,
    *,
    document_id: str,
    document_version: str,
    title: str | None = None,
    official_url: str | None = None,
    uri: str | None = None,
) -> LocalDocumentMetadata:
    """Hash a local document and return metadata without parsing or copying it."""

    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise KnowledgePackError(f"local document changed while it was indexed: {path}")
    return LocalDocumentMetadata(
        document_id=document_id,
        document_version=document_version,
        uri=uri or path.as_uri(),
        sha256=digest.hexdigest(),
        size=after.st_size,
        modified_ns=after.st_mtime_ns,
        indexed_at=_utc_now(),
        title=title,
        media_type=mimetypes.guess_type(path.name)[0] or "application/octet-stream",
        official_url=official_url,
    )


def save_local_index(entries: Iterable[LocalDocumentMetadata], path: str | Path) -> Path:
    """Write a deterministic metadata-only index; document bytes are not copied."""

    ordered = sorted(entries, key=lambda item: (item.document_id, item.document_version, item.uri))
    payload = {
        "schema_version": LOCAL_INDEX_SCHEMA_VERSION,
        "documents": [item.to_dict() for item in ordered],
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8", newline="\n")
    return target


def load_local_index(path: str | Path) -> list[LocalDocumentMetadata]:
    """Load a metadata-only local index and reject content-bearing fields."""

    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise KnowledgePackError(f"cannot load local document index {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise KnowledgePackError("local document index root must be an object")
    _reject_embedded_content(payload)
    if set(payload) != {"schema_version", "documents"}:
        raise KnowledgePackError("local document index contains unknown top-level fields")
    if payload.get("schema_version") != LOCAL_INDEX_SCHEMA_VERSION:
        raise KnowledgePackError(
            f"unsupported local index schema {payload.get('schema_version')!r}"
        )
    documents = payload.get("documents")
    if not isinstance(documents, list):
        raise KnowledgePackError("local document index documents must be an array")
    if not all(isinstance(item, Mapping) for item in documents):
        raise KnowledgePackError("local document index entries must be objects")
    return [LocalDocumentMetadata.from_dict(item) for item in documents]
