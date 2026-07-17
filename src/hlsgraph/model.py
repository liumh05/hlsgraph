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
    if isinstance(value, (list, tuple, set, frozenset)):
        return [json_ready(item) for item in value]
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

    def __post_init__(self) -> None:
        require_namespaced(self.predicate, "observation predicate")
        require_namespaced(self.stage, "observation stage")
        self.authority = enum_value(AuthorityClass, self.authority)  # type: ignore[assignment]
        require_fact_authority(self.authority, "observation")
        self.completeness = enum_value(Completeness, self.completeness)  # type: ignore[assignment]
        reject_embedded_body_fields(self.metadata, "observation metadata")
        if not self.id:
            self.id = stable_id("observation", {
                "snapshot": self.snapshot_id, "subject": self.subject_id,
                "predicate": self.predicate, "value": self.value, "unit": self.unit,
                "stage": self.stage, "authority": str(self.authority), "run": self.run_id,
                "artifact": self.artifact_id, "workload": self.workload_id,
                "anchor": self.anchor, "completeness": str(self.completeness),
                "observed_at": self.observed_at, "metadata": self.metadata,
            })

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Observation":
        data = dict(value)
        if data.get("anchor"):
            data["anchor"] = SourceAnchor.from_dict(data["anchor"])
        return cls(**data)


@dataclass(slots=True)
class Derivation:
    snapshot_id: str
    subject_id: str
    predicate: str
    value: Any
    algorithm: str
    algorithm_version: str
    input_observation_ids: list[str]
    id: str = ""
    unit: str | None = None
    stage: str = Stage.UNKNOWN.value
    authority: AuthorityClass = AuthorityClass.DERIVED_FACT
    completeness: Completeness = Completeness.COMPLETE
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_namespaced(self.predicate, "derivation predicate")
        require_namespaced(self.stage, "derivation stage")
        self.authority = enum_value(AuthorityClass, self.authority)  # type: ignore[assignment]
        require_fact_authority(self.authority, "derivation")
        self.completeness = enum_value(Completeness, self.completeness)  # type: ignore[assignment]
        reject_embedded_body_fields(self.metadata, "derivation metadata")
        if not self.input_observation_ids:
            raise ValueError("a derivation must cite at least one input observation")
        if not self.id:
            self.id = stable_id("derivation", {"snapshot": self.snapshot_id,
                                                "subject": self.subject_id,
                                                "predicate": self.predicate,
                                                "algorithm": self.algorithm,
                                                "version": self.algorithm_version,
                                                "inputs": sorted(self.input_observation_ids)})


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

    def __post_init__(self) -> None:
        require_namespaced(self.predicate, "prediction predicate")
        reject_embedded_body_fields(self.uncertainty, "prediction uncertainty")
        reject_embedded_body_fields(self.applicability, "prediction applicability")
        reject_embedded_body_fields(self.ood, "prediction ood")
        reject_embedded_body_fields(self.metadata, "prediction metadata")
        if not self.id:
            self.id = stable_id("prediction", {"snapshot": self.snapshot_id,
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
                                                "metadata": self.metadata})


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
    labels: list[LabelSpec] = field(default_factory=list)
    splits: dict[str, str] = field(default_factory=dict)
    kernel_families: dict[str, str] = field(default_factory=dict)
    dedup_groups: dict[str, str] = field(default_factory=dict)
    licenses: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_namespaced(self.dataset_id, "dataset_id")
        reject_embedded_body_fields(self.metadata, "dataset metadata")


def hash_artifact_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def artifact_hash_map(artifacts: Iterable[ArtifactRef]) -> dict[str, str]:
    return {artifact.uri: artifact.sha256 for artifact in sorted(artifacts, key=lambda x: x.uri)}
