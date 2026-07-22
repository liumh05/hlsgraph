"""Versioned public data contracts for HLSGraph.

The model deliberately separates inputs, observations, deterministic derivations,
knowledge rules, and predictions.  A value cannot become more authoritative merely
because it is attached to a graph node.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterable, Mapping

from .version import SCHEMA_VERSION


class ValueEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class AuthorityClass(ValueEnum):
    DECLARED_CONSTRAINT = "declared_constraint"
    STATIC_FACT = "static_fact"
    COMPILER_DECISION = "compiler_decision"
    TOOL_OBSERVATION = "tool_observation"
    VERIFICATION_EVIDENCE = "verification_evidence"
    PHYSICAL_MEASUREMENT = "physical_measurement"
    DERIVED_FACT = "derived_fact"
    KNOWLEDGE_RULE = "knowledge_rule"
    PREDICTION_HYPOTHESIS = "prediction_hypothesis"
    SYNTHETIC = "synthetic"


_NON_FACT_AUTHORITIES = frozenset({
    AuthorityClass.KNOWLEDGE_RULE,
    AuthorityClass.PREDICTION_HYPOTHESIS,
})


def require_fact_authority(authority: AuthorityClass, field_name: str) -> None:
    if authority in _NON_FACT_AUTHORITIES:
        raise ValueError(
            f"{field_name} cannot use {authority.value!r} authority; "
            "knowledge rules and predictions require dedicated envelope types"
        )


class Stage(ValueEnum):
    SOURCE = "source"
    AST = "ast"
    MLIR = "mlir"
    HLS_IR = "hls_ir"
    LLVM = "llvm"
    SCHEDULE = "schedule"
    RTL = "rtl"
    POST_SYNTH = "post_synth"
    POST_PLACE = "post_place"
    POST_ROUTE = "post_route"
    CSIM = "csim"
    COSIM = "cosim"
    HARDWARE_RUNTIME = "hardware_runtime"
    UNKNOWN = "unknown"


class AccessPolicy(ValueEnum):
    PRIVATE = "private"
    PROJECT = "project"
    PUBLIC = "public"


class RetentionPolicy(ValueEnum):
    EXTERNAL = "external"
    MANAGED = "managed"
    EPHEMERAL = "ephemeral"


class Completeness(ValueEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    MISSING = "missing"
    AMBIGUOUS = "ambiguous"


class EvidenceKind(ValueEnum):
    """Closed set of deterministic evidence targets used by public contracts."""

    OBSERVATION = "observation"
    DERIVATION = "derivation"
    ARTIFACT = "artifact"
    ENTITY_ANCHOR = "entity_anchor"
    RELATION = "relation"


class ActionMaterializationStatus(ValueEnum):
    MATERIALIZED = "materialized"
    NO_OP = "no_op"
    FAILED = "failed"


class RunStatus(ValueEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CACHED = "cached"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class FailureClass(ValueEnum):
    NONE = "none"
    INPUT = "input"
    DESIGN_COMPILE = "design_compile"
    CORRECTNESS = "correctness"
    RESOURCE = "resource"
    TIMING = "timing"
    LICENSE = "license"
    INFRASTRUCTURE = "infrastructure"
    INFRA_RESOURCE_GUARD = "infra_resource_guard"
    SSH = "ssh"
    TIMEOUT = "timeout"
    BENCHMARK = "benchmark"
    UNSUPPORTED = "unsupported"
    CANCELLED = "cancelled"


class GateKind(ValueEnum):
    CORRECTNESS = "correctness"
    RESOURCE_FITS = "resource_fits"
    POST_ROUTE_TIMING = "post_route_timing"


class GateStatus(ValueEnum):
    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


class VerificationKind(ValueEnum):
    CSIM = "csim"
    RTL_COSIM = "rtl_cosim"
    ASSERTION = "assertion"
    FORMAL = "formal"
    MISMATCH = "mismatch"
    DEADLOCK = "deadlock"


class DiagnosticSeverity(ValueEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


_NAMESPACED = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
_EMBEDDED_BODY_FIELDS = frozenset({
    "body", "chunks", "content", "document_text", "extracted_text", "file_content",
    "full_text", "page_text", "pages", "pdf_base64", "private_payload", "raw_content",
    "raw_dump", "raw_source", "raw_text", "snippet", "source_code", "source_text",
    "vendor_dump",
})
MAX_DIAGNOSTIC_MESSAGE_CHARS = 2048
_WINDOWS_ABSOLUTE_PATH = re.compile(
    r"(?i)(?<![A-Za-z0-9+._-])[A-Za-z]:[\\/]"
)
_UNC_ABSOLUTE_PATH = re.compile(
    r"(?<![A-Za-z0-9+._:/\\-])(?:\\\\|//)(?![\\/])"
)
_POSIX_ABSOLUTE_PATH = re.compile(
    r"(?<![A-Za-z0-9_:/])/(?![/\s])"
)
_ROOTED_WINDOWS_PATH = re.compile(
    r"(?<![A-Za-z0-9_:/\\])\\(?![\\\s])"
)


def utc_now() -> str:
    # Tool runs are immutable events.  Microsecond precision avoids assigning the
    # same event identity to two fast, otherwise identical local/fake executions.
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def json_ready(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {key: json_ready(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (set, frozenset)):
        # Sets have no JSON ordering.  Sorting by each element's own canonical
        # JSON makes generic serialization deterministic across processes and
        # PYTHONHASHSEED values; API boundaries that require ordered semantics
        # (for example extractor options) reject sets before reaching here.
        items = [json_ready(item) for item in value]
        return sorted(items, key=lambda item: json.dumps(
            item, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            allow_nan=False,
        ))
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(json_ready(value), ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def stable_id(prefix: str, value: Any, length: int = 24) -> str:
    return f"{prefix}_{stable_hash(value)[:length]}"


def require_namespaced(value: str, field_name: str) -> str:
    if not value or not _NAMESPACED.fullmatch(value):
        raise ValueError(f"{field_name} must be a non-empty namespaced identifier: {value!r}")
    return value


def reject_embedded_body_fields(value: Any, field_name: str = "metadata") -> None:
    """Reject obvious document/source bodies from public graph metadata.

    Structural identifiers and short semantic attributes are allowed. Raw
    source/report bodies belong in Artifact Store and must be referenced by
    hash/anchor rather than copied into SQLite, REST, MCP, or ML features.
    """
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().casefold().replace("-", "_")
            if normalized in _EMBEDDED_BODY_FIELDS:
                raise ValueError(
                    f"{field_name}.{key} may contain an embedded body; store an ArtifactRef instead"
                )
            reject_embedded_body_fields(item, f"{field_name}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            reject_embedded_body_fields(item, f"{field_name}[{index}]")


def require_safe_anchor_text(
    value: str | None, field_name: str, *, max_length: int,
) -> str | None:
    """Normalize one public anchor string without exposing host-absolute paths.

    Source and IR anchors are serialized through SQLite, REST, MCP, rendering,
    and ML exports.  Relative locations such as ``loc("src/kernel.cpp":18:5)``
    and symbolic locations such as ``!dbg !4`` remain valid; host-absolute
    Windows, UNC, rooted-Windows, and POSIX paths are replaced by a stable,
    non-reversible digest marker.  A value mutated after construction still
    fails closed as a non-canonical payload at the persistence boundary.
    """
    if value is None:
        return None
    if (not isinstance(value, str) or not value.strip() or "\x00" in value
            or len(value) > max_length):
        raise ValueError(
            f"{field_name} must be a bounded non-empty string without NUL or None"
        )
    if any(pattern.search(value) for pattern in (
        _WINDOWS_ABSOLUTE_PATH,
        _UNC_ABSOLUTE_PATH,
        _POSIX_ABSOLUTE_PATH,
        _ROOTED_WINDOWS_PATH,
    )):
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
        return f"redacted.sha256:{digest}"
    return value


def safe_relative_path(value: str, field_name: str = "path") -> str:
    normalized = value.replace("\\", "/")
    if (not normalized or "\x00" in normalized or normalized.startswith("/")
            or re.match(r"^[A-Za-z]:", normalized)):
        raise ValueError(f"{field_name} must be project-relative: {value!r}")
    parts = normalized.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"{field_name} contains an unsafe path component: {value!r}")
    return normalized


def require_nonnegative_finite_map(value: Mapping[str, Any], field_name: str) -> None:
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{field_name} keys must be non-empty strings")
        if (not isinstance(item, (int, float)) or isinstance(item, bool)
                or not math.isfinite(float(item)) or float(item) < 0):
            raise ValueError(f"{field_name}.{key} must be a finite non-negative number")


def enum_value(enum_type: type[ValueEnum], value: ValueEnum | str) -> ValueEnum:
    return value if isinstance(value, enum_type) else enum_type(value)


@dataclass(slots=True)
class SourceAnchor:
    artifact_id: str
    start_line: int | None = None
    start_column: int | None = None
    end_line: int | None = None
    end_column: int | None = None
    symbol: str | None = None
    ir_location: str | None = None
    mapping_kind: str | None = None
    ambiguity: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_id, str):
            raise ValueError("anchor artifact_id must be a string")
        require_namespaced(self.artifact_id, "anchor artifact_id")
        for name in ("start_line", "start_column", "end_line", "end_column"):
            value = getattr(self, name)
            if (value is not None
                    and (not isinstance(value, int) or isinstance(value, bool) or value < 1)):
                raise ValueError(f"{name} must be a positive integer or None")
        for name, limit in (
            ("symbol", 512),
            ("ir_location", 1024),
            ("mapping_kind", 256),
            ("ambiguity", 1024),
        ):
            setattr(
                self,
                name,
                require_safe_anchor_text(
                    getattr(self, name), f"anchor {name}", max_length=limit,
                ),
            )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SourceAnchor":
        return cls(**dict(value))


# Public-contract spelling.  ``SourceAnchor`` remains as a descriptive and
# backwards-compatible name, while callers may use the vendor-neutral ``Anchor``.
Anchor = SourceAnchor


@dataclass(slots=True)
class EvidenceRef:
    """A typed, optionally snapshot-qualified reference to deterministic evidence.

    The reference is deliberately not a generic string.  Its kind determines
    which ledger namespace must contain ``target_id`` and the store resolves it
    fail-closed at the boundary that owns the reference.
    """

    kind: EvidenceKind
    target_id: str
    snapshot_id: str | None = None
    anchor: SourceAnchor | None = None
    id: str = ""

    def __post_init__(self) -> None:
        self.kind = enum_value(EvidenceKind, self.kind)  # type: ignore[assignment]
        require_namespaced(self.target_id, "evidence target_id")
        if self.snapshot_id is not None:
            require_namespaced(self.snapshot_id, "evidence snapshot_id")
        if self.anchor is not None and not isinstance(self.anchor, SourceAnchor):
            self.anchor = SourceAnchor.from_dict(self.anchor)  # type: ignore[arg-type]
        if self.kind in {EvidenceKind.OBSERVATION, EvidenceKind.DERIVATION}:
            if self.anchor is not None:
                raise ValueError(
                    f"{self.kind.value} evidence cannot carry an artifact/entity anchor"
                )
        elif self.kind == EvidenceKind.ARTIFACT:
            if self.anchor is not None and self.anchor.artifact_id != self.target_id:
                raise ValueError(
                    "artifact evidence anchor must name the target artifact"
                )
        elif self.kind not in {
            EvidenceKind.ENTITY_ANCHOR, EvidenceKind.RELATION,
        }:  # pragma: no cover - enum is closed
            raise ValueError(f"unsupported evidence kind: {self.kind!r}")
        expected_id = stable_id("evidence_ref", {
            "kind": self.kind.value,
            "target": self.target_id,
            "snapshot": self.snapshot_id,
            "anchor": self.anchor,
        })
        if self.id and self.id != expected_id:
            raise ValueError(
                f"evidence reference stable id {self.id!r} does not match {expected_id!r}"
            )
        self.id = expected_id

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvidenceRef":
        data = dict(value)
        if data.get("anchor"):
            data["anchor"] = SourceAnchor.from_dict(data["anchor"])
        return cls(**data)


@dataclass(slots=True)
class ArtifactRef:
    kind: str
    uri: str
    sha256: str
    size: int
    id: str = ""
    media_type: str | None = None
    role: str | None = None
    license: str | None = None
    producer_run_id: str | None = None
    retention: RetentionPolicy = RetentionPolicy.EXTERNAL
    access: AccessPolicy = AccessPolicy.PRIVATE
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_namespaced(self.kind, "artifact kind")
        # Public bundles expose ArtifactRef records through SQLite, REST, MCP,
        # and ML provenance tables.  Persist only normalized project-relative
        # locations; absolute host paths and parent traversal would disclose
        # workstation layout even when the referenced bytes remain private.
        self.uri = safe_relative_path(self.uri, "artifact uri")
        if not re.fullmatch(r"[0-9a-fA-F]{64}", self.sha256):
            raise ValueError("artifact sha256 must contain 64 hexadecimal characters")
        if self.size < 0:
            raise ValueError("artifact size must be non-negative")
        self.sha256 = self.sha256.lower()
        self.retention = enum_value(RetentionPolicy, self.retention)  # type: ignore[assignment]
        self.access = enum_value(AccessPolicy, self.access)  # type: ignore[assignment]
        reject_embedded_body_fields(self.metadata, "artifact metadata")
        if not self.id:
            # SHA-256 identifies bytes; ArtifactRef identifies those bytes plus
            # their project-local evidence and policy contract.  Two snapshots
            # may legally attach different licensing/access metadata to the same
            # bytes without mutating an older reference.
            self.id = stable_id("artifact", {
                "kind": self.kind, "uri": self.uri, "sha256": self.sha256,
                "size": self.size, "media_type": self.media_type, "role": self.role,
                "license": self.license, "producer_run_id": self.producer_run_id,
                "retention": str(self.retention), "access": str(self.access),
                "metadata": self.metadata,
            })

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ArtifactRef":
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class LanguageSpecCompatibility:
    """One exact language-specification compatibility claim.

    This is an adapter claim, not a fact inferred from an IR filename, dialect
    spelling, or arbitrary artifact metadata.  The public v0.3 pipeline does
    not authorize any producer to make the claim; this type reserves a future
    persisted capability contract and is not itself a trust token.
    """

    family: str
    revision: str
    compatibility_contract: str

    def __post_init__(self) -> None:
        require_namespaced(self.family, "language specification family")
        require_namespaced(self.revision, "language specification revision")
        require_namespaced(
            self.compatibility_contract,
            "language specification compatibility contract",
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LanguageSpecCompatibility":
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class ArtifactSemanticClaim:
    """Semantic compatibility claimed by the extractor handling one artifact.

    The shape is reserved for a future authorized adapter protocol.  Public
    v0.3 extraction rejects every such claim, including one returned by a
    plugin with a built-in-looking name.
    """

    artifact_id: str
    artifact_revision: str
    adapter_contract: str
    adapter_version: str
    language_spec_contracts: tuple[LanguageSpecCompatibility, ...]

    def __post_init__(self) -> None:
        require_namespaced(self.artifact_id, "semantic claim artifact_id")
        require_namespaced(
            self.artifact_revision, "semantic claim artifact_revision",
        )
        require_namespaced(
            self.adapter_contract, "semantic claim adapter_contract",
        )
        if not isinstance(self.adapter_version, str) or not self.adapter_version.strip():
            raise ValueError("semantic claim adapter_version is required")
        values = tuple(
            item if isinstance(item, LanguageSpecCompatibility)
            else LanguageSpecCompatibility.from_dict(item)
            for item in self.language_spec_contracts
        )
        if not values:
            raise ValueError("semantic claim requires a language specification contract")
        families = [item.family for item in values]
        if len(set(families)) != len(families):
            raise ValueError("semantic claim language specification families must be unique")
        object.__setattr__(
            self, "language_spec_contracts",
            tuple(sorted(values, key=lambda item: item.family)),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ArtifactSemanticClaim":
        data = dict(value)
        data["language_spec_contracts"] = tuple(
            LanguageSpecCompatibility.from_dict(item)
            for item in data.get("language_spec_contracts", ())
        )
        return cls(**data)


@dataclass(frozen=True, slots=True)
class ArtifactSemanticAttestation:
    """Pipeline-issued, content-closed semantic evidence for one IR artifact.

    The value is deterministic and deliberately has no free-form metadata, but
    constructing or storing it does not confer authority.  Public v0.3 has no
    persisted adapter-authorization ledger and retrieval ignores these values.
    """

    snapshot_id: str
    artifact_id: str
    artifact_kind: str
    artifact_sha256: str
    artifact_revision: str
    extraction_hash: str
    extractor_name: str
    extractor_version: str
    extractor_identity: str
    adapter_contract: str
    adapter_version: str
    language_spec_contracts: tuple[LanguageSpecCompatibility, ...]
    attestation_contract: str = "hlsgraph.artifact_semantic_attestation.v1"
    origin_kind: str = "immutable_extraction_manifest"
    id: str = ""

    def __post_init__(self) -> None:
        require_namespaced(self.snapshot_id, "semantic attestation snapshot_id")
        require_namespaced(self.artifact_id, "semantic attestation artifact_id")
        require_namespaced(self.artifact_kind, "semantic attestation artifact_kind")
        if not re.fullmatch(r"[0-9a-fA-F]{64}", self.artifact_sha256):
            raise ValueError("semantic attestation artifact_sha256 must be SHA-256")
        object.__setattr__(self, "artifact_sha256", self.artifact_sha256.lower())
        require_namespaced(
            self.artifact_revision, "semantic attestation artifact_revision",
        )
        for value, label in (
            (self.extraction_hash, "semantic attestation extraction_hash"),
            (self.extractor_identity, "semantic attestation extractor_identity"),
        ):
            if not re.fullmatch(r"[0-9a-fA-F]{64}", value):
                raise ValueError(f"{label} must be SHA-256")
        object.__setattr__(self, "extraction_hash", self.extraction_hash.lower())
        object.__setattr__(self, "extractor_identity", self.extractor_identity.lower())
        require_namespaced(self.extractor_name, "semantic attestation extractor_name")
        if not isinstance(self.extractor_version, str) or not self.extractor_version.strip():
            raise ValueError("semantic attestation extractor_version is required")
        require_namespaced(self.adapter_contract, "semantic attestation adapter_contract")
        if not isinstance(self.adapter_version, str) or not self.adapter_version.strip():
            raise ValueError("semantic attestation adapter_version is required")
        require_namespaced(
            self.attestation_contract,
            "semantic attestation contract",
        )
        if self.origin_kind != "immutable_extraction_manifest":
            raise ValueError("unsupported semantic attestation origin_kind")
        values = tuple(
            item if isinstance(item, LanguageSpecCompatibility)
            else LanguageSpecCompatibility.from_dict(item)
            for item in self.language_spec_contracts
        )
        if not values:
            raise ValueError("semantic attestation requires a language specification contract")
        families = [item.family for item in values]
        if len(set(families)) != len(families):
            raise ValueError(
                "semantic attestation language specification families must be unique"
            )
        values = tuple(sorted(values, key=lambda item: item.family))
        object.__setattr__(self, "language_spec_contracts", values)
        expected = stable_id("semantic_attestation", {
            "snapshot_id": self.snapshot_id,
            "artifact_id": self.artifact_id,
            "artifact_kind": self.artifact_kind,
            "artifact_sha256": self.artifact_sha256,
            "artifact_revision": self.artifact_revision,
            "extraction_hash": self.extraction_hash,
            "extractor_name": self.extractor_name,
            "extractor_version": self.extractor_version,
            "extractor_identity": self.extractor_identity,
            "adapter_contract": self.adapter_contract,
            "adapter_version": self.adapter_version,
            "language_spec_contracts": values,
            "attestation_contract": self.attestation_contract,
            "origin_kind": self.origin_kind,
        })
        if self.id and self.id != expected:
            raise ValueError(
                f"semantic attestation stable id {self.id!r} does not match {expected!r}"
            )
        object.__setattr__(self, "id", expected)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ArtifactSemanticAttestation":
        data = dict(value)
        data["language_spec_contracts"] = tuple(
            LanguageSpecCompatibility.from_dict(item)
            for item in data.get("language_spec_contracts", ())
        )
        return cls(**data)


@dataclass(slots=True)
class TranslationUnit:
    file: str
    directory: str = "."
    arguments: list[str] = field(default_factory=list)
    output: str | None = None

    def __post_init__(self) -> None:
        self.file = safe_relative_path(self.file, "translation unit file")
        if self.directory not in ("", "."):
            self.directory = safe_relative_path(self.directory, "translation unit directory")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TranslationUnit":
        return cls(file=str(value["file"]), directory=str(value.get("directory", ".")),
                   arguments=[str(x) for x in value.get("arguments", [])],
                   output=value.get("output"))


@dataclass(slots=True)
class BuildContext:
    top: str
    language: str = "c++"
    translation_units: list[TranslationUnit] = field(default_factory=list)
    include_dirs: list[str] = field(default_factory=list)
    defines: dict[str, str] = field(default_factory=dict)
    cflags: list[str] = field(default_factory=list)
    config_files: list[str] = field(default_factory=list)
    tcl_files: list[str] = field(default_factory=list)
    testbench_files: list[str] = field(default_factory=list)
    golden_files: list[str] = field(default_factory=list)
    dependencies: list[dict[str, Any]] = field(default_factory=list)
    flow_target: str = "vitis"
    compile_commands: str | None = None

    def __post_init__(self) -> None:
        if not self.top:
            raise ValueError("build.top is required")
        for field_name in ("include_dirs", "config_files", "tcl_files", "testbench_files", "golden_files"):
            values = []
            for path in getattr(self, field_name):
                values.append(safe_relative_path(path, field_name))
            setattr(self, field_name, values)
        if self.compile_commands:
            self.compile_commands = safe_relative_path(self.compile_commands, "compile_commands")
        reject_embedded_body_fields(self.dependencies, "build dependencies")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BuildContext":
        data = dict(value)
        data["translation_units"] = [TranslationUnit.from_dict(x) for x in data.get("translation_units", [])]
        return cls(**data)


@dataclass(slots=True)
class ClockConstraint:
    name: str
    period_ns: float
    uncertainty_ns: float = 0.0
    source: str | None = None

    def __post_init__(self) -> None:
        if self.period_ns <= 0 or self.uncertainty_ns < 0:
            raise ValueError("clock period must be positive and uncertainty non-negative")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ClockConstraint":
        return cls(**dict(value))


@dataclass(slots=True)
class TargetProfile:
    vendor: str = "amd"
    part: str | None = None
    package: str | None = None
    speed_grade: str | None = None
    board: str | None = None
    platform: str | None = None
    platform_hash: str | None = None
    capacities: dict[str, float] = field(default_factory=dict)
    reserved_resources: dict[str, float] = field(default_factory=dict)
    clocks: list[ClockConstraint] = field(default_factory=list)
    memory_topology: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_nonnegative_finite_map(self.capacities, "target capacities")
        require_nonnegative_finite_map(
            self.reserved_resources, "target reserved resources",
        )
        for key, reserved in self.reserved_resources.items():
            if key in self.capacities and float(reserved) > float(self.capacities[key]):
                raise ValueError(f"target reserved resource {key!r} exceeds capacity")
        reject_embedded_body_fields(self.memory_topology, "target memory topology")
        reject_embedded_body_fields(self.metadata, "target metadata")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TargetProfile":
        data = dict(value)
        data["clocks"] = [ClockConstraint.from_dict(x) for x in data.get("clocks", [])]
        return cls(**data)


@dataclass(slots=True)
class ConstraintSet:
    performance: dict[str, Any] = field(default_factory=dict)
    resources: dict[str, float] = field(default_factory=dict)
    power: dict[str, Any] = field(default_factory=dict)
    numerical: dict[str, Any] = field(default_factory=dict)
    interfaces: dict[str, Any] = field(default_factory=dict)
    xdc_files: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.xdc_files = [safe_relative_path(x, "xdc file") for x in self.xdc_files]
        require_nonnegative_finite_map(self.resources, "constraint resources")
        for name in ("performance", "power", "numerical", "interfaces"):
            reject_embedded_body_fields(getattr(self, name), f"constraints {name}")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ConstraintSet":
        return cls(**dict(value))


@dataclass(slots=True)
class ToolchainContext:
    id: str
    vendor: str
    name: str
    version: str
    build: str | None = None
    executable: str | None = None
    environment_hash: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_namespaced(self.id, "toolchain id")
        reject_embedded_body_fields(self.metadata, "toolchain metadata")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ToolchainContext":
        return cls(**dict(value))


@dataclass(slots=True)
class ToolOutputSpec:
    """One explicitly declared, project-relative output of a tool stage."""

    path: str
    kind: str
    role: str = "tool_output"
    access: AccessPolicy = AccessPolicy.PROJECT
    license: str | None = None
    required: bool = True
    consumed_by: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.path = safe_relative_path(self.path, "tool output path")
        require_namespaced(self.kind, "tool output kind")
        require_namespaced(self.role, "tool output role")
        self.access = enum_value(AccessPolicy, self.access)  # type: ignore[assignment]
        for stage in self.consumed_by:
            require_namespaced(stage, "tool output consumer stage")
        if len(set(self.consumed_by)) != len(self.consumed_by):
            raise ValueError("tool output consumed_by stages must be unique")
        reject_embedded_body_fields(self.metadata, "tool output metadata")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ToolOutputSpec":
        return cls(**dict(value))


@dataclass(slots=True)
class ProjectManifest:
    project_id: str
    name: str
    build: BuildContext
    target: TargetProfile = field(default_factory=TargetProfile)
    constraints: ConstraintSet = field(default_factory=ConstraintSet)
    toolchains: list[ToolchainContext] = field(default_factory=list)
    artifact_paths: list[dict[str, Any]] = field(default_factory=list)
    stage_commands: dict[str, list[str]] = field(default_factory=dict)
    stage_outputs: dict[str, list[ToolOutputSpec]] = field(default_factory=dict)
    stage_toolchains: dict[str, str] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_namespaced(self.project_id, "project_id")
        if not self.name:
            raise ValueError("manifest name is required")
        reject_embedded_body_fields(self.metadata, "project metadata")
        for stage, command in self.stage_commands.items():
            require_namespaced(stage, "stage command key")
            if not command or not all(isinstance(x, str) and x for x in command):
                raise ValueError(f"stage command {stage!r} must be a non-empty argv list")
        for stage, outputs in self.stage_outputs.items():
            require_namespaced(stage, "stage output key")
            if stage not in self.stage_commands:
                raise ValueError(f"stage outputs {stage!r} have no matching stage command")
            paths = [item.path for item in outputs]
            if len(set(paths)) != len(paths):
                raise ValueError(f"stage outputs {stage!r} contain duplicate paths")
            unknown_consumers = sorted({consumer for item in outputs
                                        for consumer in item.consumed_by}
                                       - set(self.stage_commands))
            if unknown_consumers:
                raise ValueError(
                    f"stage outputs {stage!r} reference unknown consumer stages: "
                    + ", ".join(unknown_consumers)
                )
            if any(stage in item.consumed_by for item in outputs):
                raise ValueError(f"stage outputs {stage!r} cannot be consumed by the same stage")
        output_owners: dict[str, list[str]] = {}
        for stage, outputs in self.stage_outputs.items():
            for item in outputs:
                output_owners.setdefault(item.path, []).append(stage)
        duplicate_outputs = {path: sorted(owners) for path, owners in output_owners.items()
                             if len(owners) > 1}
        if duplicate_outputs:
            raise ValueError(
                f"tool output paths must be globally unique across stages: {duplicate_outputs}"
            )
        self._validate_stage_toolchains()

    def _validate_stage_toolchains(self) -> None:
        toolchain_ids = [item.id for item in self.toolchains]
        for identifier in toolchain_ids:
            require_namespaced(identifier, "toolchain id")
        if len(set(toolchain_ids)) != len(toolchain_ids):
            raise ValueError("toolchain IDs must be unique")
        for stage, identifier in self.stage_toolchains.items():
            require_namespaced(stage, "stage toolchain key")
            require_namespaced(identifier, "stage toolchain id")
            if stage not in self.stage_commands:
                raise ValueError(f"stage toolchain {stage!r} has no matching stage command")
            if identifier not in toolchain_ids:
                raise ValueError(
                    f"stage toolchain {stage!r} references unknown toolchain {identifier!r}"
                )
        if self.stage_commands and not toolchain_ids:
            raise ValueError("stage commands require at least one declared toolchain")
        if len(toolchain_ids) > 1:
            missing = sorted(set(self.stage_commands) - set(self.stage_toolchains))
            if missing:
                raise ValueError(
                    "multiple toolchains require an explicit stage_toolchains mapping for: "
                    + ", ".join(missing)
                )

    def toolchain_for_stage(self, stage: str) -> ToolchainContext:
        """Resolve one immutable toolchain identity for an executable stage."""
        require_namespaced(stage, "stage")
        if stage not in self.stage_commands:
            raise ValueError(f"manifest has no command for stage {stage!r}")
        # Revalidate because callers may construct and then mutate dataclasses.
        self._validate_stage_toolchains()
        selected = self.stage_toolchains.get(stage)
        if selected is not None:
            matches = [item for item in self.toolchains if item.id == selected]
            if len(matches) != 1:
                raise ValueError(
                    f"stage {stage!r} resolves to {len(matches)} toolchains for ID {selected!r}"
                )
            return matches[0]
        if len(self.toolchains) == 1:
            return self.toolchains[0]
        if not self.toolchains:
            raise ValueError(f"stage {stage!r} has no declared toolchain")
        raise ValueError(f"stage {stage!r} requires an explicit stage_toolchains mapping")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProjectManifest":
        data = dict(value)
        if "schema_version" not in data:
            raise KeyError("schema_version")
        data["build"] = BuildContext.from_dict(data["build"])
        data["target"] = TargetProfile.from_dict(data.get("target", {}))
        data["constraints"] = ConstraintSet.from_dict(data.get("constraints", {}))
        data["toolchains"] = [ToolchainContext.from_dict(x) for x in data.get("toolchains", [])]
        data["stage_outputs"] = {
            str(stage): [ToolOutputSpec.from_dict(item) for item in outputs]
            for stage, outputs in data.get("stage_outputs", {}).items()
        }
        data["stage_toolchains"] = {
            str(stage): str(identifier)
            for stage, identifier in data.get("stage_toolchains", {}).items()
        }
        return cls(**data)

    def identity_payload(self) -> dict[str, Any]:
        # Dataclasses remain intentionally mutable while a caller prepares a
        # variant. Revalidate execution identity before it can be hashed into a
        # snapshot so post-construction mutations cannot bypass any fail-closed
        # stage/toolchain/output contract.
        data = json_ready(self)
        ProjectManifest.from_dict(data)
        data.pop("metadata", None)
        return data


@dataclass(slots=True)
class DesignSnapshot:
    project_id: str
    manifest_hash: str
    artifact_hashes: dict[str, str]
    build_hash: str
    target_hash: str
    constraint_hash: str
    toolchain_hash: str
    extraction_hash: str = ""
    id: str = ""
    parent_snapshot_id: str | None = None
    action_id: str | None = None
    created_at: str = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        reject_embedded_body_fields(self.metadata, "snapshot metadata")
        if not self.id:
            self.id = stable_id("snapshot", self.identity_payload(), 32)

    def identity_payload(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "manifest_hash": self.manifest_hash,
            "artifact_hashes": dict(sorted(self.artifact_hashes.items())),
            "build_hash": self.build_hash,
            "target_hash": self.target_hash,
            "constraint_hash": self.constraint_hash,
            "toolchain_hash": self.toolchain_hash,
            "extraction_hash": self.extraction_hash,
            "action_id": self.action_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DesignSnapshot":
        return cls(**dict(value))


@dataclass(slots=True)
class VariantAction:
    parent_snapshot_id: str
    kind: str
    scope_id: str | None
    delta: dict[str, Any]
    proposer: str
    id: str = ""
    rationale: str | None = None
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        require_namespaced(self.kind, "variant action kind")
        reject_embedded_body_fields(self.delta, "variant delta")
        if not self.id:
            self.id = stable_id("action", {"parent": self.parent_snapshot_id, "kind": self.kind,
                                            "scope": self.scope_id, "delta": self.delta,
                                            "proposer": self.proposer})


@dataclass(slots=True)
class ActionMaterialization:
    """One immutable attempt to materialize a proposed :class:`VariantAction`.

    Multiple attempts may be recorded for one action.  ``materialized`` means a
    distinct child snapshot exists, ``no_op`` records an explicitly diagnosed
    semantic no-op, and ``failed`` retains diagnostics plus an optional failed
    candidate snapshot.
    """

    action_id: str
    parent_snapshot_id: str
    status: ActionMaterializationStatus
    id: str = ""
    result_snapshot_id: str | None = None
    diagnostic_ids: list[str] = field(default_factory=list)
    evidence_refs: list[EvidenceRef] = field(default_factory=list)
    attempted_at: str = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_namespaced(self.action_id, "materialization action_id")
        require_namespaced(self.parent_snapshot_id, "materialization parent_snapshot_id")
        self.status = enum_value(ActionMaterializationStatus, self.status)  # type: ignore[assignment]
        if self.result_snapshot_id is not None:
            require_namespaced(
                self.result_snapshot_id, "materialization result_snapshot_id"
            )
            if self.result_snapshot_id == self.parent_snapshot_id:
                raise ValueError("a materialization result must differ from its parent snapshot")
        if not isinstance(self.diagnostic_ids, list):
            raise ValueError("materialization diagnostic_ids must be a list")
        for diagnostic_id in self.diagnostic_ids:
            require_namespaced(diagnostic_id, "materialization diagnostic_id")
        if len(set(self.diagnostic_ids)) != len(self.diagnostic_ids):
            raise ValueError("materialization diagnostic_ids must be unique")
        self.diagnostic_ids = sorted(self.diagnostic_ids)
        if not isinstance(self.evidence_refs, list):
            raise ValueError("materialization evidence_refs must be a list")
        self.evidence_refs = sorted(
            (item if isinstance(item, EvidenceRef) else EvidenceRef.from_dict(item)
             for item in self.evidence_refs),
            key=lambda item: item.id,
        )
        if len({item.id for item in self.evidence_refs}) != len(self.evidence_refs):
            raise ValueError("materialization evidence_refs must be unique")
        if self.status == ActionMaterializationStatus.MATERIALIZED:
            if self.result_snapshot_id is None:
                raise ValueError("materialized action requires a result_snapshot_id")
        elif self.status == ActionMaterializationStatus.NO_OP:
            if self.result_snapshot_id is not None:
                raise ValueError("no-op action cannot have a result snapshot")
            if not self.diagnostic_ids:
                raise ValueError("no-op action requires at least one diagnostic")
        elif self.status == ActionMaterializationStatus.FAILED:
            if not self.diagnostic_ids:
                raise ValueError("failed action requires at least one diagnostic")
        if (not isinstance(self.attempted_at, str) or not self.attempted_at
                or len(self.attempted_at) > 64 or "\x00" in self.attempted_at):
            raise ValueError("materialization attempted_at must be a bounded timestamp")
        try:
            attempted = datetime.fromisoformat(
                self.attempted_at[:-1] + "+00:00"
                if self.attempted_at.endswith("Z") else self.attempted_at
            )
        except ValueError as exc:
            raise ValueError("materialization attempted_at must be ISO-8601") from exc
        if attempted.tzinfo is None:
            raise ValueError("materialization attempted_at must include a timezone")
        reject_embedded_body_fields(self.metadata, "materialization metadata")
        if not self.id:
            self.id = stable_id("materialization", {
                "action": self.action_id,
                "parent": self.parent_snapshot_id,
                "status": self.status.value,
                "result": self.result_snapshot_id,
                "diagnostics": self.diagnostic_ids,
                "evidence": self.evidence_refs,
                "attempted_at": self.attempted_at,
                "metadata": self.metadata,
            })

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ActionMaterialization":
        data = dict(value)
        data["evidence_refs"] = [
            EvidenceRef.from_dict(item) for item in data.get("evidence_refs", [])
        ]
        return cls(**data)


@dataclass(slots=True)
class EntityCorrespondence:
    """A versioned, evidence-backed mapping between entities in two snapshots."""

    source_snapshot_id: str
    source_entity_id: str
    target_snapshot_id: str
    target_entity_id: str
    kind: str
    producer: str
    producer_version: str
    evidence_refs: list[EvidenceRef]
    id: str = ""
    authority: AuthorityClass = AuthorityClass.DERIVED_FACT
    completeness: Completeness = Completeness.COMPLETE
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("source_snapshot_id", "source_entity_id", "target_snapshot_id",
                     "target_entity_id"):
            require_namespaced(getattr(self, name), f"correspondence {name}")
        require_namespaced(self.kind, "correspondence kind")
        require_namespaced(self.producer, "correspondence producer")
        require_namespaced(self.producer_version, "correspondence producer_version")
        if (self.source_snapshot_id == self.target_snapshot_id
                and self.source_entity_id == self.target_entity_id):
            raise ValueError("a correspondence must connect two distinct entity records")
        self.authority = enum_value(AuthorityClass, self.authority)  # type: ignore[assignment]
        require_fact_authority(self.authority, "entity correspondence")
        self.completeness = enum_value(Completeness, self.completeness)  # type: ignore[assignment]
        if self.completeness == Completeness.MISSING:
            raise ValueError("a missing correspondence must be represented by a diagnostic")
        if not isinstance(self.evidence_refs, list) or not self.evidence_refs:
            raise ValueError("a correspondence must cite at least one evidence_ref")
        values: list[EvidenceRef] = []
        endpoint_snapshots = {self.source_snapshot_id, self.target_snapshot_id}
        for raw in self.evidence_refs:
            item = raw if isinstance(raw, EvidenceRef) else EvidenceRef.from_dict(raw)
            if item.snapshot_id is not None and item.snapshot_id not in endpoint_snapshots:
                raise ValueError(
                    "correspondence evidence must belong to a source or target snapshot"
                )
            if item.snapshot_id is None and len(endpoint_snapshots) > 1:
                raise ValueError(
                    "cross-snapshot correspondence evidence must set snapshot_id explicitly"
                )
            if item.snapshot_id is None and len(endpoint_snapshots) == 1:
                item = EvidenceRef(
                    kind=item.kind, target_id=item.target_id,
                    snapshot_id=self.source_snapshot_id, anchor=item.anchor,
                )
            values.append(item)
        self.evidence_refs = sorted(values, key=lambda item: item.id)
        if len({item.id for item in self.evidence_refs}) != len(self.evidence_refs):
            raise ValueError("correspondence evidence_refs must be unique")
        reject_embedded_body_fields(self.metadata, "correspondence metadata")
        if not self.id:
            self.id = stable_id("correspondence", {
                "source_snapshot": self.source_snapshot_id,
                "source_entity": self.source_entity_id,
                "target_snapshot": self.target_snapshot_id,
                "target_entity": self.target_entity_id,
                "kind": self.kind,
                "producer": self.producer,
                "producer_version": self.producer_version,
                "evidence": self.evidence_refs,
                "authority": self.authority.value,
                "completeness": self.completeness.value,
                "metadata": self.metadata,
            })

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EntityCorrespondence":
        data = dict(value)
        data["evidence_refs"] = [
            EvidenceRef.from_dict(item) for item in data.get("evidence_refs", [])
        ]
        return cls(**data)


@dataclass(slots=True)
class Entity:
    kind: str
    name: str
    snapshot_id: str
    id: str = ""
    qualified_name: str | None = None
    authority: AuthorityClass = AuthorityClass.STATIC_FACT
    stage: str = Stage.SOURCE.value
    attrs: dict[str, Any] = field(default_factory=dict)
    anchors: list[SourceAnchor] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    completeness: Completeness = Completeness.COMPLETE

    def __post_init__(self) -> None:
        require_namespaced(self.kind, "entity kind")
        require_namespaced(self.stage, "entity stage")
        self.authority = enum_value(AuthorityClass, self.authority)  # type: ignore[assignment]
        require_fact_authority(self.authority, "entity")
        self.completeness = enum_value(Completeness, self.completeness)  # type: ignore[assignment]
        reject_embedded_body_fields(self.attrs, "entity attrs")
        if not self.name:
            raise ValueError("entity name is required")
        if not self.id:
            self.id = stable_id("entity", {"snapshot": self.snapshot_id, "kind": self.kind,
                                            "name": self.qualified_name or self.name,
                                            "stage": self.stage,
                                            "authority": str(self.authority)})

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Entity":
        data = dict(value)
        data["anchors"] = [SourceAnchor.from_dict(x) for x in data.get("anchors", [])]
        return cls(**data)


@dataclass(slots=True)
class Relation:
    src: str
    dst: str
    kind: str
    snapshot_id: str
    id: str = ""
    authority: AuthorityClass = AuthorityClass.STATIC_FACT
    stage: str = Stage.SOURCE.value
    attrs: dict[str, Any] = field(default_factory=dict)
    anchors: list[SourceAnchor] = field(default_factory=list)
    mapping_kind: str | None = None
    completeness: Completeness = Completeness.COMPLETE

    def __post_init__(self) -> None:
        require_namespaced(self.kind, "relation kind")
        require_namespaced(self.stage, "relation stage")
        self.authority = enum_value(AuthorityClass, self.authority)  # type: ignore[assignment]
        require_fact_authority(self.authority, "relation")
        self.completeness = enum_value(Completeness, self.completeness)  # type: ignore[assignment]
        reject_embedded_body_fields(self.attrs, "relation attrs")
        if not self.id:
            self.id = stable_id("relation", {"snapshot": self.snapshot_id, "src": self.src,
                                              "dst": self.dst, "kind": self.kind,
                                              "attrs": self.attrs, "mapping": self.mapping_kind,
                                              "stage": self.stage,
                                              "authority": str(self.authority),
                                              "anchors": self.anchors,
                                              "completeness": str(self.completeness)})

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Relation":
        data = dict(value)
        data["anchors"] = [SourceAnchor.from_dict(x) for x in data.get("anchors", [])]
        return cls(**data)


@dataclass(frozen=True, slots=True)
class ObservationSource:
    """Parser-issued commitment for one canonical source report.

    This record is deliberately singular.  It commits the observation
    predicate/value/unit tuple to one exact content-addressed artifact, but is
    not by itself an authorization or signature.  The ledger/retriever accept
    it as tool evidence only after fixed-parser replay, and never assemble an
    observation from sibling reports.  Predicate-specific multi-report joins
    require a separate, future contract and are not represented by this type.
    """

    artifact_id: str
    artifact_sha256: str
    parser_name: str
    parser_version: str
    payload_sha256: str
    binding_sha256: str
    contract: str = "hlsgraph.observation_source.v1"

    def __post_init__(self) -> None:
        require_namespaced(self.artifact_id, "observation source artifact_id")
        require_namespaced(self.parser_name, "observation source parser_name")
        require_namespaced(self.contract, "observation source contract")
        if self.contract != "hlsgraph.observation_source.v1":
            raise ValueError("unsupported observation source contract")
        if not isinstance(self.parser_version, str) or not self.parser_version.strip():
            raise ValueError("observation source parser_version must be non-empty")
        object.__setattr__(self, "parser_version", self.parser_version.strip())
        for field_name in ("artifact_sha256", "payload_sha256", "binding_sha256"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value):
                raise ValueError(f"observation source {field_name} must be SHA-256")
            object.__setattr__(self, field_name, value.lower())

    def validation_error(
        self, *, predicate: str, value: Any, unit: str | None,
    ) -> str | None:
        expected_payload = stable_hash({
            "predicate": predicate, "value": value, "unit": unit,
        })
        if self.payload_sha256 != expected_payload:
            return "parser source payload does not match predicate/value/unit"
        expected_binding = stable_hash({
            "contract": self.contract,
            "artifact_id": self.artifact_id,
            "artifact_sha256": self.artifact_sha256,
            "parser_name": self.parser_name,
            "parser_version": self.parser_version,
            "payload_sha256": self.payload_sha256,
        })
        if self.binding_sha256 != expected_binding:
            return "parser source binding does not match its artifact and payload"
        return None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ObservationSource":
        return cls(**dict(value))


def _observation_source_commitment(
    *, artifact: ArtifactRef, parser_name: str, parser_version: str,
    predicate: str, value: Any, unit: str | None,
) -> ObservationSource:
    """Build an untrusted parser-output content commitment.

    This private helper does not authorize an observation.  Tool truth is
    admitted only after the ledger/retriever reruns the corresponding fixed
    parser over the managed artifact and finds exactly one matching output.
    """

    normalized_parser_version = str(parser_version).strip()
    if not normalized_parser_version:
        raise ValueError("observation source parser_version must be non-empty")
    payload_sha256 = stable_hash({
        "predicate": predicate,
        "value": value,
        # ``None`` is intentional and distinct from an omitted field or an
        # empty unit string in canonical JSON.
        "unit": unit,
    })
    binding_sha256 = stable_hash({
        "contract": "hlsgraph.observation_source.v1",
        "artifact_id": artifact.id,
        "artifact_sha256": artifact.sha256,
        "parser_name": parser_name,
        "parser_version": normalized_parser_version,
        "payload_sha256": payload_sha256,
    })
    return ObservationSource(
        artifact_id=artifact.id,
        artifact_sha256=artifact.sha256,
        parser_name=parser_name,
        parser_version=normalized_parser_version,
        payload_sha256=payload_sha256,
        binding_sha256=binding_sha256,
    )


@dataclass(slots=True)
class Observation:
    snapshot_id: str
    subject_id: str
    predicate: str
    value: Any
    stage: str
    authority: AuthorityClass
    id: str = ""
    unit: str | None = None
    run_id: str | None = None
    artifact_id: str | None = None
    anchor: SourceAnchor | None = None
    completeness: Completeness = Completeness.COMPLETE
    workload_id: str | None = None
    observed_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    source: ObservationSource | None = None

    def __post_init__(self) -> None:
        require_namespaced(self.predicate, "observation predicate")
        require_namespaced(self.stage, "observation stage")
        self.authority = enum_value(AuthorityClass, self.authority)  # type: ignore[assignment]
        require_fact_authority(self.authority, "observation")
        self.completeness = enum_value(Completeness, self.completeness)  # type: ignore[assignment]
        reject_embedded_body_fields(self.metadata, "observation metadata")
        if self.anchor is not None and not isinstance(self.anchor, SourceAnchor):
            self.anchor = SourceAnchor.from_dict(self.anchor)  # type: ignore[arg-type]
        if self.source is not None and not isinstance(self.source, ObservationSource):
            self.source = ObservationSource.from_dict(self.source)  # type: ignore[arg-type]
        if (self.artifact_id is not None and self.anchor is not None
                and self.artifact_id != self.anchor.artifact_id):
            raise ValueError("observation artifact_id must match its anchor artifact_id")
        if self.source is not None:
            if self.artifact_id is None or self.anchor is None:
                raise ValueError(
                    "parser-issued observation source requires one artifact_id and anchor"
                )
            if (self.source.artifact_id != self.artifact_id
                    or self.source.artifact_id != self.anchor.artifact_id):
                raise ValueError(
                    "parser-issued observation source must name the canonical artifact"
                )
            source_error = self.source.validation_error(
                predicate=self.predicate, value=self.value, unit=self.unit,
            )
            if source_error is not None:
                raise ValueError(source_error)
        identity = {
            "snapshot": self.snapshot_id, "subject": self.subject_id,
            "predicate": self.predicate, "value": self.value, "unit": self.unit,
            "stage": self.stage, "authority": str(self.authority), "run": self.run_id,
            "artifact": self.artifact_id, "workload": self.workload_id,
            "anchor": self.anchor, "completeness": str(self.completeness),
            "observed_at": self.observed_at, "metadata": self.metadata,
        }
        # Preserve v0.1/v0.2 observation identity byte-for-byte.  Only the new
        # v0.3 typed-source contract extends the identity payload.
        if self.source is not None:
            identity["source"] = self.source
        expected_id = stable_id("observation", identity)
        if self.source is not None and self.id and self.id != expected_id:
            raise ValueError(
                f"typed observation stable id {self.id!r} does not match {expected_id!r}"
            )
        if not self.id:
            self.id = expected_id

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Observation":
        data = dict(value)
        if data.get("anchor"):
            data["anchor"] = SourceAnchor.from_dict(data["anchor"])
        if data.get("source"):
            data["source"] = ObservationSource.from_dict(data["source"])
        return cls(**data)


@dataclass(slots=True)
class Derivation:
    snapshot_id: str
    subject_id: str
    predicate: str
    value: Any
    algorithm: str
    algorithm_version: str
    input_observation_ids: list[str] = field(default_factory=list)
    id: str = ""
    unit: str | None = None
    stage: str = Stage.UNKNOWN.value
    authority: AuthorityClass = AuthorityClass.DERIVED_FACT
    completeness: Completeness = Completeness.COMPLETE
    metadata: dict[str, Any] = field(default_factory=dict)
    evidence_refs: list[EvidenceRef] = field(default_factory=list)

    def __post_init__(self) -> None:
        require_namespaced(self.predicate, "derivation predicate")
        require_namespaced(self.stage, "derivation stage")
        self.authority = enum_value(AuthorityClass, self.authority)  # type: ignore[assignment]
        require_fact_authority(self.authority, "derivation")
        self.completeness = enum_value(Completeness, self.completeness)  # type: ignore[assignment]
        reject_embedded_body_fields(self.metadata, "derivation metadata")
        if not isinstance(self.input_observation_ids, list):
            raise ValueError("derivation input_observation_ids must be a list")
        for observation_id in self.input_observation_ids:
            require_namespaced(observation_id, "derivation input observation")
        if len(set(self.input_observation_ids)) != len(self.input_observation_ids):
            raise ValueError("derivation input_observation_ids must be unique")
        legacy_ids = sorted(self.input_observation_ids)
        if not isinstance(self.evidence_refs, list):
            raise ValueError("derivation evidence_refs must be a list")
        values = [
            item if isinstance(item, EvidenceRef) else EvidenceRef.from_dict(item)
            for item in self.evidence_refs
        ]
        for item in values:
            if item.snapshot_id not in {None, self.snapshot_id}:
                raise ValueError("derivation evidence must belong to its snapshot")
        # A v0.1 payload cites observations only.  Normalize those IDs into the
        # v0.2 typed contract while retaining the legacy list as a compatibility
        # projection for readers that have not yet adopted EvidenceRef.
        normalized: list[EvidenceRef] = []
        for item in values:
            normalized.append(item if item.snapshot_id is not None else EvidenceRef(
                kind=item.kind, target_id=item.target_id,
                snapshot_id=self.snapshot_id, anchor=item.anchor,
            ))
        observation_targets = {
            item.target_id for item in normalized
            if item.kind == EvidenceKind.OBSERVATION
            and item.snapshot_id == self.snapshot_id
        }
        normalized.extend(EvidenceRef(
            kind=EvidenceKind.OBSERVATION,
            target_id=observation_id,
            snapshot_id=self.snapshot_id,
        ) for observation_id in legacy_ids if observation_id not in observation_targets)
        self.evidence_refs = sorted(normalized, key=lambda item: item.id)
        if len({item.id for item in self.evidence_refs}) != len(self.evidence_refs):
            raise ValueError("derivation evidence_refs must be unique")
        projected_ids = sorted({
            item.target_id for item in self.evidence_refs
            if item.kind == EvidenceKind.OBSERVATION
            and item.snapshot_id == self.snapshot_id and item.anchor is None
        })
        if legacy_ids and not set(legacy_ids).issubset(projected_ids):
            raise ValueError("legacy derivation observations conflict with evidence_refs")
        self.input_observation_ids = projected_ids
        if not self.evidence_refs:
            raise ValueError("a derivation must cite at least one evidence_ref")
        if not self.id:
            identity = {"snapshot": self.snapshot_id,
                        "subject": self.subject_id,
                        "predicate": self.predicate,
                        "algorithm": self.algorithm,
                        "version": self.algorithm_version}
            legacy_only = (
                bool(self.input_observation_ids)
                and len(self.input_observation_ids) == len(self.evidence_refs)
                and all(
                    item.kind == EvidenceKind.OBSERVATION
                    and item.snapshot_id == self.snapshot_id
                    and item.anchor is None
                    for item in self.evidence_refs
                )
            )
            if legacy_only:
                # Preserve v0.1 IDs so migration never cascades into gate and
                # verification references.
                identity["inputs"] = sorted(self.input_observation_ids)
            else:
                identity["evidence_refs"] = self.evidence_refs
            self.id = stable_id("derivation", identity)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Derivation":
        data = dict(value)
        data["evidence_refs"] = [
            EvidenceRef.from_dict(item) for item in data.get("evidence_refs", [])
        ]
        return cls(**data)


@dataclass(slots=True)
class Diagnostic:
    snapshot_id: str
    code: str
    severity: DiagnosticSeverity
    message: str
    id: str = ""
    stage: str = Stage.UNKNOWN.value
    run_id: str | None = None
    subject_id: str | None = None
    artifact_id: str | None = None
    anchor: SourceAnchor | None = None
    guidance: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_namespaced(self.code, "diagnostic code")
        require_namespaced(self.stage, "diagnostic stage")
        if not isinstance(self.message, str) or not self.message.strip():
            raise ValueError("diagnostic message must be a non-empty string")
        if len(self.message) > MAX_DIAGNOSTIC_MESSAGE_CHARS:
            raise ValueError(
                f"diagnostic message exceeds {MAX_DIAGNOSTIC_MESSAGE_CHARS} characters"
            )
        self.severity = enum_value(DiagnosticSeverity, self.severity)  # type: ignore[assignment]
        reject_embedded_body_fields(self.metadata, "diagnostic metadata")
        if not self.id:
            self.id = stable_id("diagnostic", {"snapshot": self.snapshot_id, "code": self.code,
                                                "message": self.message, "stage": self.stage,
                                                "run": self.run_id, "subject": self.subject_id})

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Diagnostic":
        data = dict(value)
        if data.get("anchor"):
            data["anchor"] = SourceAnchor.from_dict(data["anchor"])
        return cls(**data)


@dataclass(slots=True)
class GateResult:
    kind: GateKind
    status: GateStatus
    evidence_ids: list[str] = field(default_factory=list)
    reason: str | None = None

    def __post_init__(self) -> None:
        self.kind = enum_value(GateKind, self.kind)  # type: ignore[assignment]
        self.status = enum_value(GateStatus, self.status)  # type: ignore[assignment]
        if self.status == GateStatus.PASS and not self.evidence_ids:
            raise ValueError("a passing gate must cite at least one evidence_id")


@dataclass(slots=True)
class VerificationResult:
    snapshot_id: str
    kind: VerificationKind
    status: GateStatus
    id: str = ""
    run_id: str | None = None
    workload_id: str | None = None
    evidence_ids: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.kind = enum_value(VerificationKind, self.kind)  # type: ignore[assignment]
        self.status = enum_value(GateStatus, self.status)  # type: ignore[assignment]
        reject_embedded_body_fields(self.details, "verification details")
        if not self.run_id and not self.evidence_ids:
            raise ValueError("a verification result must cite a run or evidence")
        if self.status == GateStatus.PASS and not self.evidence_ids:
            raise ValueError("a passing verification must cite non-run report evidence")
        if not self.id:
            self.id = stable_id("verification", {"snapshot": self.snapshot_id,
                                                  "kind": str(self.kind), "run": self.run_id,
                                                  "workload": self.workload_id,
                                                  "evidence": sorted(self.evidence_ids)})


@dataclass(slots=True)
class ToolRun:
    snapshot_id: str
    stage: str
    backend: str
    request_hash: str
    id: str = ""
    toolchain_id: str | None = None
    status: RunStatus = RunStatus.QUEUED
    command: list[str] = field(default_factory=list)
    working_directory: str | None = None
    environment_hash: str | None = None
    input_artifact_ids: list[str] = field(default_factory=list)
    output_artifact_ids: list[str] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)
    gates: list[GateResult] = field(default_factory=list)
    failure_class: FailureClass = FailureClass.NONE
    exit_code: int | None = None
    attempt: int = 1
    started_at: str | None = None
    finished_at: str | None = None
    elapsed_s: float | None = None
    message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("snapshot_id", "stage", "backend"):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise ValueError(f"{name} must be a string")
        require_namespaced(self.snapshot_id, "run snapshot_id")
        require_namespaced(self.backend, "runner backend")
        require_namespaced(self.stage, "runner stage")
        if (not isinstance(self.request_hash, str)
                or not re.fullmatch(r"[0-9a-fA-F]{64}", self.request_hash)):
            raise ValueError("run request_hash must be a 64-character SHA-256 digest")
        self.request_hash = self.request_hash.lower()
        if self.id:
            if not isinstance(self.id, str):
                raise ValueError("run id must be a string")
            require_namespaced(self.id, "run id")
        if self.toolchain_id is not None:
            if not isinstance(self.toolchain_id, str):
                raise ValueError("run toolchain_id must be a string or None")
            require_namespaced(self.toolchain_id, "run toolchain_id")
        if self.working_directory is not None:
            if (not isinstance(self.working_directory, str)
                    or not self.working_directory.strip()
                    or "\x00" in self.working_directory):
                raise ValueError("run working_directory must be a non-empty string without NUL")
        if self.environment_hash is not None:
            if (not isinstance(self.environment_hash, str)
                    or not re.fullmatch(r"[0-9a-fA-F]{64}", self.environment_hash)):
                raise ValueError("run environment_hash must be a 64-character SHA-256 digest")
            self.environment_hash = self.environment_hash.lower()
        if (not isinstance(self.command, list)
                or any(not isinstance(item, str) or not item.strip() or "\x00" in item
                       for item in self.command)):
            raise ValueError(
                "run command must be a list of non-empty strings without NUL"
            )
        for name in ("input_artifact_ids", "output_artifact_ids", "diagnostics"):
            values = getattr(self, name)
            if not isinstance(values, list):
                raise ValueError(f"run {name} must be a list")
            if any(not isinstance(item, str) or not item for item in values):
                raise ValueError(f"run {name} must contain only non-empty string IDs")
            for item in values:
                require_namespaced(item, f"run {name} item")
        if not isinstance(self.gates, list) or any(
                not isinstance(item, GateResult) for item in self.gates):
            raise ValueError("run gates must be a list of GateResult values")
        if (not isinstance(self.attempt, int) or isinstance(self.attempt, bool)
                or self.attempt < 1 or self.attempt > 2_147_483_647):
            raise ValueError("run attempt must be a positive 32-bit integer")
        if (self.exit_code is not None
                and (not isinstance(self.exit_code, int) or isinstance(self.exit_code, bool))):
            raise ValueError("run exit_code must be an integer or None")
        if self.elapsed_s is not None:
            if (not isinstance(self.elapsed_s, (int, float))
                    or isinstance(self.elapsed_s, bool)
                    or not math.isfinite(float(self.elapsed_s))
                    or float(self.elapsed_s) < 0):
                raise ValueError("run elapsed_s must be a finite non-negative number or None")
        parsed_times: dict[str, datetime] = {}
        for name in ("started_at", "finished_at"):
            value = getattr(self, name)
            if value is None:
                continue
            if not isinstance(value, str) or len(value) > 64 or "\x00" in value:
                raise ValueError(f"run {name} must be a bounded ISO-8601 timestamp or None")
            try:
                parsed = datetime.fromisoformat(
                    value[:-1] + "+00:00" if value.endswith("Z") else value
                )
            except ValueError as exc:
                raise ValueError(
                    f"run {name} must be a bounded ISO-8601 timestamp or None"
                ) from exc
            if parsed.tzinfo is None:
                raise ValueError(f"run {name} must include a timezone")
            parsed_times[name] = parsed
        if ("started_at" in parsed_times and "finished_at" in parsed_times
                and parsed_times["finished_at"] < parsed_times["started_at"]):
            raise ValueError("run finished_at cannot precede started_at")
        if (self.message is not None
                and (not isinstance(self.message, str) or "\x00" in self.message)):
            raise ValueError("run message must be a string without NUL or None")
        if not isinstance(self.metadata, dict):
            raise ValueError("run metadata must be an object")
        if "runner_fingerprint" in self.metadata:
            fingerprint = self.metadata["runner_fingerprint"]
            if (not isinstance(fingerprint, str)
                    or not re.fullmatch(r"[0-9a-fA-F]{64}", fingerprint)):
                raise ValueError(
                    "run metadata.runner_fingerprint must be a 64-character SHA-256 digest"
                )
            self.metadata["runner_fingerprint"] = fingerprint.lower()
        self.status = enum_value(RunStatus, self.status)  # type: ignore[assignment]
        self.failure_class = enum_value(FailureClass, self.failure_class)  # type: ignore[assignment]
        if self.failure_class == FailureClass.INFRA_RESOURCE_GUARD:
            preflight_rejection = (
                self.metadata.get("resource_guard_configured") is True
                and self.metadata.get("resource_guard_checked") is True
                and self.metadata.get("resource_guard_passed") is False
                and self.metadata.get("fresh_execution") is False
            )
            runtime_rejection = (
                self.metadata.get("runtime_guard_configured") is True
                and self.metadata.get("runtime_guard_checked") is True
                and self.metadata.get("runtime_guard_passed") is False
                and self.metadata.get("runtime_guard_triggered") is True
                and self.metadata.get("fresh_execution") is True
            )
            if (self.status != RunStatus.FAILED
                    or not (preflight_rejection or runtime_rejection)
                    or self.metadata.get("fresh_tool_truth") is not False
                    or self.metadata.get("tool_truth") is not False
                    or self.metadata.get("authority") != "infrastructure"
                    or self.output_artifact_ids):
                raise ValueError(
                    "infra_resource_guard requires a failed, non-tool, "
                    "structured resource-guard provenance event"
                )
        reject_embedded_body_fields(self.metadata, "run metadata")
        if not self.id:
            self.id = stable_id("run", {"snapshot": self.snapshot_id, "stage": self.stage,
                                        "backend": self.backend, "request": self.request_hash,
                                        "attempt": self.attempt, "started_at": self.started_at})

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ToolRun":
        data = dict(value)
        data["gates"] = [GateResult(**x) for x in data.get("gates", [])]
        return cls(**data)


@dataclass(frozen=True, slots=True)
class ExecutionDeclaredOutput:
    """One immutable output declaration covered by an execution attestation."""

    path: str
    kind: str
    required: bool
    max_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", safe_relative_path(
            self.path, "execution declared output path",
        ))
        require_namespaced(self.kind, "execution declared output kind")
        if not isinstance(self.required, bool):
            raise ValueError("execution declared output required must be boolean")
        if (not isinstance(self.max_bytes, int) or isinstance(self.max_bytes, bool)
                or self.max_bytes < 0):
            raise ValueError("execution declared output max_bytes must be non-negative")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExecutionDeclaredOutput":
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class ExecutionOutputAttestation:
    """Identity of one declared output whose bytes reached the local CAS."""

    artifact_id: str
    path: str
    kind: str
    sha256: str
    size: int

    def __post_init__(self) -> None:
        require_namespaced(self.artifact_id, "execution output artifact_id")
        object.__setattr__(self, "path", safe_relative_path(
            self.path, "execution output path",
        ))
        require_namespaced(self.kind, "execution output kind")
        digest = str(self.sha256).casefold()
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("execution output sha256 must be SHA-256")
        object.__setattr__(self, "sha256", digest)
        if not isinstance(self.size, int) or isinstance(self.size, bool) or self.size < 0:
            raise ValueError("execution output size must be non-negative")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExecutionOutputAttestation":
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class ExecutionAttestation:
    """Publicly re-verifiable identity of one pipeline-validated tool execution.

    Constructing this serializable value is deliberately *not* sufficient to
    commit tool truth.  The write path additionally consumes a process-local,
    one-shot capability issued inside :class:`StageOrchestrator`.  Once that
    succeeds, this value and an :class:`ExecutionCommitReceipt` are retained so
    later readers can re-check every deterministic binding without possessing
    the private capability.
    """

    run_id: str
    snapshot_id: str
    stage: str
    runner_identity: str
    runner_authority: str
    runner_fingerprint: str
    request_hash: str
    run_payload_hash: str
    manifest_hash: str
    build_hash: str
    target_hash: str
    constraint_hash: str
    toolchain_hash: str
    toolchain_id: str
    declared_outputs: tuple[ExecutionDeclaredOutput, ...] = ()
    outputs: tuple[ExecutionOutputAttestation, ...] = ()
    protocol_version: str = "hlsgraph.execution_attestation.v1"
    validator: str = "hlsgraph.stage_orchestrator.v1"
    id: str = ""

    def __post_init__(self) -> None:
        for value, label in (
            (self.run_id, "execution attestation run_id"),
            (self.snapshot_id, "execution attestation snapshot_id"),
            (self.stage, "execution attestation stage"),
            (self.runner_identity, "execution attestation runner_identity"),
            (self.runner_authority, "execution attestation runner_authority"),
            (self.toolchain_id, "execution attestation toolchain_id"),
            (self.protocol_version, "execution attestation protocol_version"),
            (self.validator, "execution attestation validator"),
        ):
            require_namespaced(value, label)
        for name in (
            "runner_fingerprint", "request_hash", "run_payload_hash",
            "manifest_hash", "build_hash", "target_hash", "constraint_hash",
            "toolchain_hash",
        ):
            digest = str(getattr(self, name)).casefold()
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError(f"execution attestation {name} must be SHA-256")
            object.__setattr__(self, name, digest)
        declarations = tuple(
            item if isinstance(item, ExecutionDeclaredOutput)
            else ExecutionDeclaredOutput.from_dict(item)
            for item in self.declared_outputs
        )
        outputs = tuple(
            item if isinstance(item, ExecutionOutputAttestation)
            else ExecutionOutputAttestation.from_dict(item)
            for item in self.outputs
        )
        if len({item.path for item in declarations}) != len(declarations):
            raise ValueError("execution attestation declarations must have unique paths")
        if len({item.path for item in outputs}) != len(outputs):
            raise ValueError("execution attestation outputs must have unique paths")
        if len({item.artifact_id for item in outputs}) != len(outputs):
            raise ValueError("execution attestation outputs must have unique artifact IDs")
        object.__setattr__(
            self, "declared_outputs", tuple(sorted(declarations, key=lambda item: item.path)),
        )
        object.__setattr__(
            self, "outputs", tuple(sorted(outputs, key=lambda item: item.path)),
        )
        expected = stable_id("execution_attestation", {
            "run_id": self.run_id,
            "snapshot_id": self.snapshot_id,
            "stage": self.stage,
            "runner_identity": self.runner_identity,
            "runner_authority": self.runner_authority,
            "runner_fingerprint": self.runner_fingerprint,
            "request_hash": self.request_hash,
            "run_payload_hash": self.run_payload_hash,
            "manifest_hash": self.manifest_hash,
            "build_hash": self.build_hash,
            "target_hash": self.target_hash,
            "constraint_hash": self.constraint_hash,
            "toolchain_hash": self.toolchain_hash,
            "toolchain_id": self.toolchain_id,
            "declared_outputs": self.declared_outputs,
            "outputs": self.outputs,
            "protocol_version": self.protocol_version,
            "validator": self.validator,
        }, 32)
        if self.id and self.id != expected:
            raise ValueError(
                f"execution attestation stable id {self.id!r} does not match {expected!r}"
            )
        object.__setattr__(self, "id", expected)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExecutionAttestation":
        data = dict(value)
        data["declared_outputs"] = tuple(
            ExecutionDeclaredOutput.from_dict(item)
            for item in data.get("declared_outputs", ())
        )
        data["outputs"] = tuple(
            ExecutionOutputAttestation.from_dict(item)
            for item in data.get("outputs", ())
        )
        return cls(**data)


@dataclass(frozen=True, slots=True)
class ExecutionCommitReceipt:
    """Store-issued public receipt for a capability-authorized attestation."""

    attestation_id: str
    run_id: str
    snapshot_id: str
    run_payload_hash: str
    attestation_payload_hash: str
    receipt_contract: str = "hlsgraph.execution_commit_receipt.v1"
    validator: str = "hlsgraph.ledger.execution_attestation.v1"
    id: str = ""

    def __post_init__(self) -> None:
        for value, label in (
            (self.attestation_id, "execution receipt attestation_id"),
            (self.run_id, "execution receipt run_id"),
            (self.snapshot_id, "execution receipt snapshot_id"),
            (self.receipt_contract, "execution receipt contract"),
            (self.validator, "execution receipt validator"),
        ):
            require_namespaced(value, label)
        for name in ("run_payload_hash", "attestation_payload_hash"):
            digest = str(getattr(self, name)).casefold()
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError(f"execution receipt {name} must be SHA-256")
            object.__setattr__(self, name, digest)
        expected = stable_id("execution_receipt", {
            "attestation_id": self.attestation_id,
            "run_id": self.run_id,
            "snapshot_id": self.snapshot_id,
            "run_payload_hash": self.run_payload_hash,
            "attestation_payload_hash": self.attestation_payload_hash,
            "receipt_contract": self.receipt_contract,
            "validator": self.validator,
        }, 32)
        if self.id and self.id != expected:
            raise ValueError(
                f"execution receipt stable id {self.id!r} does not match {expected!r}"
            )
        object.__setattr__(self, "id", expected)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExecutionCommitReceipt":
        return cls(**dict(value))


@dataclass(slots=True)
class KnowledgeRule:
    document_id: str
    document_version: str
    section: str
    rule_id: str
    title: str
    applicability: dict[str, Any]
    condition: dict[str, Any]
    effect: dict[str, Any]
    citation_url: str
    summary: str | None = None
    license_note: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_namespaced(self.document_id, "knowledge document_id")
        require_namespaced(self.rule_id, "knowledge rule_id")
        for constraints, label in (
            (self.applicability, "knowledge applicability"),
            (self.condition, "knowledge condition"),
        ):
            if not isinstance(constraints, dict):
                raise ValueError(f"{label} must be an object")
            for key, constraint in constraints.items():
                if isinstance(constraint, list):
                    raise ValueError(
                        f"{label} constraint {key!r} alternatives require explicit one_of"
                    )
        for value, label in ((self.applicability, "knowledge applicability"),
                             (self.condition, "knowledge condition"),
                             (self.effect, "knowledge effect"),
                             (self.metadata, "knowledge metadata")):
            reject_embedded_body_fields(value, label)
        if self.summary is not None and len(self.summary) > 500:
            raise ValueError("knowledge rule summary must be at most 500 characters")

    @property
    def id(self) -> str:
        return f"{self.document_id}:{self.document_version}:{self.rule_id}"


_BUILTIN_KNOWLEDGE_TARGET_KINDS = frozenset({
    "predicate", "directive_kind", "artifact_kind", "gate_kind",
    "diagnostic_code", "entity_kind", "relation_kind",
})


@dataclass(slots=True)
class KnowledgeBinding:
    """Fail-closed applicability link from guidance to a public contract term.

    A binding is retrieval metadata, never a relation in the canonical design
    graph.  It may therefore help select a rule without upgrading that rule to
    a design fact.
    """

    knowledge_rule_id: str
    target_kind: str
    target: str
    required_context: dict[str, Any]
    producer: str
    producer_version: str
    id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_namespaced(self.knowledge_rule_id, "knowledge binding rule_id")
        if self.target_kind not in _BUILTIN_KNOWLEDGE_TARGET_KINDS:
            require_namespaced(self.target_kind, "knowledge binding target_kind")
        require_namespaced(self.target, "knowledge binding target")
        require_namespaced(self.producer, "knowledge binding producer")
        if not isinstance(self.producer_version, str) or not self.producer_version.strip():
            raise ValueError("knowledge binding producer_version is required")
        if not isinstance(self.required_context, dict):
            raise ValueError("knowledge binding required_context must be an object")
        reject_embedded_body_fields(
            self.required_context, "knowledge binding required_context",
        )
        reject_embedded_body_fields(self.metadata, "knowledge binding metadata")
        expected = stable_id("knowledge_binding", {
            "rule": self.knowledge_rule_id,
            "target_kind": self.target_kind,
            "target": self.target,
            "required_context": self.required_context,
            "producer": self.producer,
            "producer_version": self.producer_version,
        })
        if self.id and self.id != expected:
            raise ValueError(
                f"knowledge binding stable id {self.id!r} does not match {expected!r}"
            )
        self.id = expected

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "KnowledgeBinding":
        return cls(**dict(value))


class CoverageStatus(ValueEnum):
    RULE = "rule"
    CITATION_ONLY = "citation_only"
    NOT_APPLICABLE = "not_applicable"
    DEFERRED = "deferred"


class TargetCoverageStatus(ValueEnum):
    BOUND = "bound"
    NO_NORMATIVE = "no_normative"


@dataclass(slots=True)
class KnowledgeTargetCoverage:
    """Machine-checkable coverage for one supported public contract target."""

    target_kind: str
    target: str
    status: TargetCoverageStatus
    binding_ids: list[str] = field(default_factory=list)
    rationale: str | None = None
    id: str = ""

    def __post_init__(self) -> None:
        if self.target_kind not in _BUILTIN_KNOWLEDGE_TARGET_KINDS:
            require_namespaced(self.target_kind, "knowledge target coverage kind")
        require_namespaced(self.target, "knowledge target coverage target")
        self.status = enum_value(TargetCoverageStatus, self.status)  # type: ignore[assignment]
        if (not isinstance(self.binding_ids, list)
                or any(not isinstance(item, str) for item in self.binding_ids)):
            raise ValueError("knowledge target coverage binding_ids must be a list of strings")
        for item in self.binding_ids:
            require_namespaced(item, "knowledge target coverage binding_id")
        if len(set(self.binding_ids)) != len(self.binding_ids):
            raise ValueError("knowledge target coverage binding_ids must be unique")
        self.binding_ids = sorted(self.binding_ids)
        if self.status == TargetCoverageStatus.BOUND and not self.binding_ids:
            raise ValueError("bound knowledge target coverage requires a binding")
        if self.status == TargetCoverageStatus.NO_NORMATIVE and self.binding_ids:
            raise ValueError("no_normative knowledge target coverage cannot reference bindings")
        if self.status == TargetCoverageStatus.NO_NORMATIVE and self.rationale is None:
            raise ValueError("no_normative knowledge target coverage requires a rationale")
        if self.rationale is not None and (
            not isinstance(self.rationale, str) or not self.rationale.strip()
            or len(self.rationale) > 500 or "\x00" in self.rationale
        ):
            raise ValueError(
                "knowledge target coverage rationale must be a bounded non-empty string"
            )
        expected = stable_id("knowledge_target_coverage", {
            "target_kind": self.target_kind,
            "target": self.target,
            "status": self.status.value,
            "bindings": self.binding_ids,
            "rationale": self.rationale,
        })
        if self.id and self.id != expected:
            raise ValueError(
                f"knowledge target coverage stable id {self.id!r} does not match {expected!r}"
            )
        self.id = expected

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "KnowledgeTargetCoverage":
        return cls(**dict(value))


@dataclass(slots=True)
class CoverageEntry:
    document_id: str
    document_version: str
    section: str
    status: CoverageStatus
    rule_ids: list[str] = field(default_factory=list)
    binding_ids: list[str] = field(default_factory=list)
    rationale: str | None = None
    id: str = ""

    def __post_init__(self) -> None:
        require_namespaced(self.document_id, "coverage document_id")
        if not isinstance(self.document_version, str) or not self.document_version.strip():
            raise ValueError("coverage document_version is required")
        if (not isinstance(self.section, str) or not self.section.strip()
                or len(self.section) > 512 or "\x00" in self.section):
            raise ValueError("coverage section must be a bounded non-empty string")
        self.status = enum_value(CoverageStatus, self.status)  # type: ignore[assignment]
        for field_name in ("rule_ids", "binding_ids"):
            values = getattr(self, field_name)
            if not isinstance(values, list):
                raise ValueError(f"coverage {field_name} must be a list")
            for value in values:
                if not isinstance(value, str):
                    raise ValueError(f"coverage {field_name} entries must be strings")
                require_namespaced(value, f"coverage {field_name} entry")
            if len(set(values)) != len(values):
                raise ValueError(f"coverage {field_name} must be unique")
            setattr(self, field_name, sorted(values))
        if self.status == CoverageStatus.RULE and not self.rule_ids:
            raise ValueError("rule coverage requires at least one knowledge rule")
        if self.status != CoverageStatus.RULE and self.rule_ids:
            raise ValueError("only rule coverage may reference knowledge rules")
        if self.status != CoverageStatus.RULE and self.binding_ids:
            raise ValueError("only rule coverage may reference knowledge bindings")
        if self.status != CoverageStatus.RULE and self.rationale is None:
            raise ValueError("non-rule coverage requires an explicit rationale")
        if self.rationale is not None and (
            not isinstance(self.rationale, str) or not self.rationale.strip()
            or len(self.rationale) > 500 or "\x00" in self.rationale
        ):
            raise ValueError("coverage rationale must be a bounded paraphrase or None")
        expected = stable_id("coverage_entry", {
            "document": self.document_id,
            "version": self.document_version,
            "section": self.section,
            "status": self.status.value,
            "rules": self.rule_ids,
            "bindings": self.binding_ids,
            "rationale": self.rationale,
        })
        if self.id and self.id != expected:
            raise ValueError(
                f"coverage entry stable id {self.id!r} does not match {expected!r}"
            )
        self.id = expected

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CoverageEntry":
        return cls(**dict(value))


@dataclass(slots=True)
class CoverageManifest:
    """Auditable section inventory for one versioned knowledge-pack scope."""

    pack_id: str
    coverage_scope: str
    entries: list[CoverageEntry]
    target_inventory: list[KnowledgeTargetCoverage] = field(default_factory=list)
    target_registry_version: str = "hlsgraph.knowledge_supported_targets.v1"
    review_status: str = "unreviewed"
    reviewers: list[str] = field(default_factory=list)
    source_hashes: dict[str, str] = field(default_factory=dict)
    review_evidence: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "1.0"
    id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_namespaced(self.pack_id, "coverage pack_id")
        require_namespaced(self.coverage_scope, "coverage scope")
        require_namespaced(
            self.target_registry_version, "coverage target registry version",
        )
        if self.schema_version != "1.0":
            raise ValueError("unsupported coverage manifest schema")
        self.entries = sorted([
            entry if isinstance(entry, CoverageEntry) else CoverageEntry.from_dict(entry)
            for entry in self.entries
        ], key=lambda entry: (
            entry.document_id, entry.document_version, entry.section.casefold(), entry.id,
        ))
        section_keys = [
            (entry.document_id, entry.document_version, entry.section.casefold())
            for entry in self.entries
        ]
        if len(set(section_keys)) != len(section_keys):
            raise ValueError("coverage manifest contains duplicate document sections")
        self.target_inventory = sorted([
            item if isinstance(item, KnowledgeTargetCoverage)
            else KnowledgeTargetCoverage.from_dict(item)
            for item in self.target_inventory
        ], key=lambda item: (item.target_kind, item.target, item.id))
        target_keys = [
            (item.target_kind, item.target) for item in self.target_inventory
        ]
        if len(set(target_keys)) != len(target_keys):
            raise ValueError("coverage manifest contains duplicate supported targets")
        allowed_review_statuses = {
            "unreviewed", "maintainer_reviewed", "human_reviewed",
            "machine_repeated_reviewed", "machine_cross_reviewed",
        }
        if self.review_status not in allowed_review_statuses:
            raise ValueError(
                "coverage review_status must be unreviewed, maintainer_reviewed, "
                "human_reviewed, machine_repeated_reviewed, or "
                "machine_cross_reviewed"
            )
        if (not isinstance(self.reviewers, list) or any(
            not isinstance(item, str) or not item.strip() or "\x00" in item
            for item in self.reviewers
        ) or len(set(self.reviewers)) != len(self.reviewers)):
            raise ValueError("coverage reviewers must be unique non-empty strings")
        self.reviewers = sorted(self.reviewers)
        for key, digest in self.source_hashes.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("coverage source hash keys must be non-empty strings")
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError("coverage source hashes must be lowercase SHA-256 values")
        if not isinstance(self.review_evidence, dict):
            raise ValueError("coverage review_evidence must be an object")
        reject_embedded_body_fields(self.review_evidence, "coverage review_evidence")
        if self.review_status in {
            "machine_repeated_reviewed", "machine_cross_reviewed",
        }:
            if len(self.reviewers) < 2:
                raise ValueError(
                    f"{self.review_status} coverage requires two review invocations"
                )
            if not self.source_hashes:
                raise ValueError(
                    f"{self.review_status} coverage requires verified source hashes"
                )
            required_review_evidence = {
                "independent_invocations": True,
                "citation_verified": True,
                "review_agreement": True,
                "unresolved_conflicts": False,
            }
            if self.review_status == "machine_repeated_reviewed":
                required_review_evidence.update({
                    "same_model_repeated_review": True,
                    "distinct_model_families": False,
                })
            else:
                required_review_evidence["distinct_model_families"] = True
            if any(
                self.review_evidence.get(key) is not expected
                for key, expected in required_review_evidence.items()
            ):
                raise ValueError(
                    f"{self.review_status} coverage requires truthful independent "
                    "review provenance, verified citations, agreement, and no "
                    "unresolved conflicts"
                )
        reject_embedded_body_fields(self.metadata, "coverage metadata")
        expected = stable_id("coverage", {
            "pack": self.pack_id,
            "scope": self.coverage_scope,
            "entries": self.entries,
            "target_inventory": self.target_inventory,
            "target_registry_version": self.target_registry_version,
            "review_status": self.review_status,
            "reviewers": self.reviewers,
            "source_hashes": self.source_hashes,
            "review_evidence": self.review_evidence,
            "schema_version": self.schema_version,
        })
        if self.id and self.id != expected:
            raise ValueError(
                f"coverage manifest stable id {self.id!r} does not match {expected!r}"
            )
        self.id = expected

    @property
    def complete(self) -> bool:
        return bool(self.entries) and all(
            entry.status != CoverageStatus.DEFERRED for entry in self.entries
        )

    @property
    def review_ready(self) -> bool:
        """Whether classification is complete and an explicit review was recorded.

        ``complete`` deliberately answers only the coverage-classification
        question.  Keeping review readiness separate prevents an ``unreviewed``
        catalog with no deferred entries from being presented as release-ready.
        Machine-review evidence has already been validated in ``__post_init__``.
        """

        return self.complete and self.review_status != "unreviewed"

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CoverageManifest":
        data = dict(value)
        if "target_registry_version" not in data:
            raise ValueError(
                "coverage manifest must explicitly declare target_registry_version"
            )
        data["entries"] = [CoverageEntry.from_dict(item) for item in data.get("entries", [])]
        data["target_inventory"] = [
            KnowledgeTargetCoverage.from_dict(item)
            for item in data.get("target_inventory", [])
        ]
        return cls(**data)


@dataclass(slots=True)
class LocalKnowledgeIndexManifest:
    """Metadata-only identity for a private, rebuildable local sidecar index."""

    project_id: str
    document_hashes: dict[str, str]
    chunk_count: int
    index_sha256: str
    parser_id: str
    parser_version: str
    chunker_id: str
    chunker_version: str
    storage_uri: str = ".hlsgraph/private/knowledge/chunks.sqlite"
    fts_enabled: bool = True
    parser_fingerprint: str | None = None
    embedder_id: str | None = None
    embedder_version: str | None = None
    embedder_fingerprint: str | None = None
    schema_version: str = "1.0"
    created_at: str = field(default_factory=utc_now)
    content_embedded_in_canonical: bool = False
    id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_namespaced(self.project_id, "local knowledge project_id")
        if self.schema_version != "1.0":
            raise ValueError("unsupported local knowledge manifest schema")
        self.storage_uri = safe_relative_path(
            self.storage_uri, "local knowledge storage_uri",
        )
        for key, digest in self.document_hashes.items():
            if not isinstance(key, str) or not key.strip() or "\x00" in key:
                raise ValueError("local knowledge document keys must be non-empty strings")
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError("local knowledge document hashes must be lowercase SHA-256")
        if (not isinstance(self.chunk_count, int) or isinstance(self.chunk_count, bool)
                or self.chunk_count < 0):
            raise ValueError("local knowledge chunk_count must be non-negative")
        if not re.fullmatch(r"[0-9a-f]{64}", self.index_sha256):
            raise ValueError("local knowledge index_sha256 must be lowercase SHA-256")
        for field_name in ("parser_id", "chunker_id"):
            require_namespaced(getattr(self, field_name), f"local knowledge {field_name}")
        for field_name in ("parser_version", "chunker_version"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"local knowledge {field_name} is required")
        if self.parser_fingerprint is not None and not re.fullmatch(
            r"[0-9a-f]{64}", self.parser_fingerprint,
        ):
            raise ValueError("local knowledge parser_fingerprint must be SHA-256")
        if (self.embedder_id is None) != (self.embedder_version is None):
            raise ValueError("local knowledge embedder id and version must be set together")
        if self.embedder_id is not None:
            require_namespaced(self.embedder_id, "local knowledge embedder_id")
        if self.embedder_fingerprint is not None and not re.fullmatch(
            r"[0-9a-f]{64}", self.embedder_fingerprint,
        ):
            raise ValueError("local knowledge embedder fingerprint must be SHA-256")
        if self.content_embedded_in_canonical is not False:
            raise ValueError("private knowledge content cannot be embedded in the canonical store")
        reject_embedded_body_fields(self.metadata, "local knowledge metadata")
        expected = stable_id("local_knowledge_index", {
            "project": self.project_id,
            "documents": self.document_hashes,
            "chunks": self.chunk_count,
            "index_sha256": self.index_sha256,
            "parser": [self.parser_id, self.parser_version, self.parser_fingerprint],
            "chunker": [self.chunker_id, self.chunker_version],
            "embedder": [self.embedder_id, self.embedder_version,
                         self.embedder_fingerprint],
            "schema_version": self.schema_version,
        })
        if self.id and self.id != expected:
            raise ValueError(
                f"local knowledge index stable id {self.id!r} does not match {expected!r}"
            )
        self.id = expected

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LocalKnowledgeIndexManifest":
        return cls(**dict(value))


@dataclass(slots=True)
class PredictionEnvelope:
    snapshot_id: str
    subject_id: str
    predicate: str
    value: Any
    model_id: str
    model_version: str
    input_schema_version: str
    id: str = ""
    unit: str | None = None
    trainset_hash: str | None = None
    uncertainty: Any = None
    applicability: dict[str, Any] = field(default_factory=dict)
    ood: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)
    action_id: str | None = None

    def __post_init__(self) -> None:
        require_namespaced(self.predicate, "prediction predicate")
        reject_embedded_body_fields(self.uncertainty, "prediction uncertainty")
        reject_embedded_body_fields(self.applicability, "prediction applicability")
        reject_embedded_body_fields(self.ood, "prediction ood")
        reject_embedded_body_fields(self.metadata, "prediction metadata")
        if (self.action_id is not None
                and (not isinstance(self.action_id, str) or not self.action_id.strip())):
            raise ValueError("prediction action_id must be a non-empty string or None")
        if not self.id:
            identity = {"snapshot": self.snapshot_id,
                        "subject": self.subject_id,
                        "predicate": self.predicate,
                        "model": self.model_id,
                        "version": self.model_version,
                        "input_schema_version": self.input_schema_version,
                        "trainset_hash": self.trainset_hash,
                        "value": self.value,
                        "unit": self.unit,
                        "uncertainty": self.uncertainty,
                        "applicability": self.applicability,
                        "ood": self.ood,
                        "metadata": self.metadata}
            # Omitting the optional key preserves v0.1 prediction identifiers
            # exactly.  A concrete action is semantic input and therefore gets
            # its own prediction identity.
            if self.action_id is not None:
                identity["action_id"] = self.action_id
            self.id = stable_id("prediction", identity)


@dataclass(slots=True)
class LabelSpec:
    label_id: str
    snapshot_id: str
    observation_id: str | None
    predicate: str
    stage: str
    unit: str | None = None
    mask: bool = True
    missing_reason: str | None = None
    censored: bool = False
    unbounded: bool = False

    def __post_init__(self) -> None:
        for name in ("label_id", "snapshot_id"):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise ValueError(f"label {name} must be a string")
            require_namespaced(value, f"label {name}")
        if self.observation_id is not None:
            if not isinstance(self.observation_id, str):
                raise ValueError("label observation_id must be a string or None")
            require_namespaced(self.observation_id, "label observation_id")
        if not isinstance(self.predicate, str):
            raise ValueError("label predicate must be a string")
        require_namespaced(self.predicate, "label predicate")
        if not isinstance(self.stage, str):
            raise ValueError("label stage must be a string")
        require_namespaced(self.stage, "label stage")
        if (self.unit is not None
                and (not isinstance(self.unit, str) or not self.unit.strip()
                     or len(self.unit) > 64 or "\x00" in self.unit)):
            raise ValueError("label unit must be a bounded non-empty string or None")
        for name in ("mask", "censored", "unbounded"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"label {name} must be a boolean")
        if self.missing_reason is not None:
            if not isinstance(self.missing_reason, str):
                raise ValueError("label missing_reason must be a string or None")
            require_namespaced(self.missing_reason, "label missing_reason")
        if self.mask and not self.observation_id:
            raise ValueError("a present label must reference an observation")
        if self.mask and self.missing_reason is not None:
            raise ValueError("a present label cannot declare a missing_reason")
        if not self.mask and self.observation_id is not None:
            raise ValueError("a masked label cannot reference an observation")
        if not self.mask and self.missing_reason is None:
            raise ValueError("a masked label must declare a namespaced missing_reason")


@dataclass(slots=True)
class DatasetManifest:
    dataset_id: str
    feature_schema_version: str
    snapshot_ids: list[str]
    # ML features are an explicit, versioned contract. The safe default stops
    # before scheduling/tool outcomes; research datasets may opt into later
    # stages deliberately and that choice is recorded in feature_spec.json.
    feature_stages: list[str] = field(default_factory=lambda: [
        Stage.SOURCE.value, Stage.AST.value, Stage.MLIR.value,
        Stage.HLS_IR.value, Stage.LLVM.value,
    ])
    # Unknown plugin attributes remain excluded until a dataset author reviews
    # and declares them. This prevents blacklist bypasses and silent drift.
    feature_attribute_allowlist: list[str] = field(default_factory=lambda: [
        "array_size", "bitwidth", "depth", "dialect", "directive_kind",
        "element_type", "elem_type", "fifo_depth", "loop_kind", "operation",
        "options", "origin", "projection", "replication", "signed",
        "trip_count", "type", "width",
    ])
    # Static derivations and cross-snapshot mappings are opt-in feature inputs.
    # Empty lists intentionally export zero rows rather than silently widening
    # an existing feature contract.
    feature_evidence_predicates: list[str] = field(default_factory=list)
    entity_correspondence_kinds: list[str] = field(default_factory=list)
    labels: list[LabelSpec] = field(default_factory=list)
    splits: dict[str, str] = field(default_factory=dict)
    kernel_families: dict[str, str] = field(default_factory=dict)
    dedup_groups: dict[str, str] = field(default_factory=dict)
    licenses: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_namespaced(self.dataset_id, "dataset_id")
        for field_name in (
            "feature_evidence_predicates", "entity_correspondence_kinds",
        ):
            values = getattr(self, field_name)
            if not isinstance(values, list):
                raise ValueError(f"dataset {field_name} must be a list")
            for value in values:
                if not isinstance(value, str):
                    raise ValueError(f"dataset {field_name} values must be strings")
                require_namespaced(value, f"dataset {field_name}")
            if len(set(values)) != len(values):
                raise ValueError(f"dataset {field_name} values must be unique")
        reject_embedded_body_fields(self.metadata, "dataset metadata")


def hash_artifact_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def artifact_hash_map(artifacts: Iterable[ArtifactRef]) -> dict[str, str]:
    return {artifact.uri: artifact.sha256 for artifact in sorted(artifacts, key=lambda x: x.uri)}
