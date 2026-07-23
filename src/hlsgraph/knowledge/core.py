"""Loading, filtering, and metadata-only indexing for knowledge packs."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import re
import stat
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import urlparse

from hlsgraph.model import (
    CoverageStatus,
    CoverageManifest,
    KnowledgeBinding,
    KnowledgeRule,
    json_ready,
    require_namespaced,
    stable_hash,
)
from .supported_targets import (
    SUPPORTED_TARGET_REGISTRY_VERSION,
    canonical_supported_targets,
)


PACK_SCHEMA_VERSION = "2.0"
LEGACY_PACK_SCHEMA_VERSIONS = frozenset({"1.0"})
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


def knowledge_activation_hash(
    rules: Iterable[KnowledgeRule],
    bindings: Iterable[KnowledgeBinding],
    coverage: CoverageManifest | None,
) -> str:
    """Hash the complete executable knowledge surface stored in a bundle.

    Rule and binding IDs alone do not commit mutable fields such as rule prose,
    applicability, effects, or binding metadata.  This digest is recorded in
    the immutable pack inventory and recomputed at every read-side activation
    gate so altered ledger payloads become lexical-only.
    """

    rule_values = sorted(rules, key=lambda item: item.id)
    binding_values = sorted(bindings, key=lambda item: item.id)
    return stable_hash({
        "contract": "hlsgraph.knowledge_activation_surface.v1",
        "rules": [json_ready(item) for item in rule_values],
        "bindings": [json_ready(item) for item in binding_values],
        "coverage": json_ready(coverage) if coverage is not None else None,
    })

# A rule condition may be supplied directly by a binding's required context,
# or it may be a premise that follows from the *current binding target
# instance*.  The latter cases are deliberately enumerated here.  Pack authors
# cannot invent a new ``*_present`` spelling and have it become executable by
# convention: adding a premise requires an auditable core change plus a
# retrieval-side witness check.
_TARGET_CONDITION_SOURCE_DIRECTIVE = "hlsgraph.target.directive_instance.v1"
_TARGET_CONDITION_SOURCE_ARTIFACT = "hlsgraph.target.qualified_artifact.v1"
_TARGET_CONDITION_SOURCE_OBSERVATION = "hlsgraph.target.qualified_observation.v1"
_TARGET_CONDITION_SOURCE_GATE = "hlsgraph.target.qualified_gate.v1"
_DIRECT_CONDITION_SOURCE_INTERFACE_MODE = (
    "hlsgraph.condition.directive_interface_mode.v1"
)
_DIRECT_CONDITION_SOURCE_PORT_OWNERSHIP = (
    "hlsgraph.condition.directive_port_ownership.v1"
)
_DIRECT_CONDITION_SOURCE_REQUESTED_DIRECTIVE = (
    "hlsgraph.condition.requested_directive.v1"
)
_DIRECT_CONDITION_SOURCE_DIRECTIVE_OPTIONS = (
    "hlsgraph.condition.directive_options.v1"
)

_TARGET_DIRECTIVE_KINDS = frozenset({
    "DATAFLOW", "PIPELINE", "UNROLL", "ARRAY_PARTITION", "INTERFACE",
    "STREAM", "DEPENDENCE", "LOOP_TRIPCOUNT", "INLINE",
})
_TARGET_ARTIFACT_CONDITIONS: dict[str, str] = {
    "amd.vitis.csynth_xml": "csynth_report_present",
    "amd.vitis.schedule_json": "schedule_artifact_present",
    "constraint.xdc": "constraint_artifact_present",
    "amd.vivado.timing_summary": "timing_summary_present",
    "amd.vivado.post_route_timing": "post_route_timing_result_present",
    "amd.vivado.utilization": "utilization_report_present",
    "amd.vivado.post_route_utilization": "utilization_report_present",
}
_TARGET_OBSERVATION_CONDITIONS: dict[str, frozenset[str]] = {
    "csim_result_present": frozenset({
        "csim.exit_code", "csim.mismatches", "csim.assertions_failed",
    }),
    "cosim_result_present": frozenset({
        "cosim.status", "cosim.latency_min_cycles", "cosim.latency_avg_cycles",
        "cosim.latency_max_cycles", "cosim.interval_min_cycles",
        "cosim.interval_avg_cycles", "cosim.interval_max_cycles",
    }),
    "csynth_report_present": frozenset({
        "clock.estimated_period_ns", "qor.latency_best_cycles",
        "qor.latency_worst_cycles", "qor.interval_min_cycles",
        "qor.interval_max_cycles", "qor.latency_cycles",
        "qor.iteration_latency_cycles", "qor.achieved_ii",
        "resource.lut", "resource.ff", "resource.dsp", "resource.bram_18k",
        "resource.uram",
    }),
    "performance_report_present": frozenset({
        "qor.latency_best_cycles", "qor.latency_worst_cycles",
        "qor.interval_min_cycles", "qor.interval_max_cycles",
        "qor.latency_cycles", "qor.iteration_latency_cycles", "qor.target_ii",
        "qor.achieved_ii",
    }),
    "schedule_artifact_present": frozenset({
        "schedule.start_cycle", "schedule.end_cycle", "schedule.pipeline_stage",
        "schedule.operation_latency",
    }),
    "dataflow_profiling_enabled": frozenset({
        "profile.fifo_max_occupancy", "profile.read_block_cycles",
        "profile.write_block_cycles", "profile.token_count",
    }),
    "timing_summary_present": frozenset({"timing.wns_ns", "timing.tns_ns"}),
    "congestion_report_present": frozenset({"physical.congestion_level"}),
    "post_route_timing_result_present": frozenset({
        "timing.critical_path_delay_ns",
    }),
    "power_report_present": frozenset({"power.dynamic_w", "power.static_w"}),
    "utilization_report_present": frozenset({
        "resource.lut", "resource.ff", "resource.dsp", "resource.bram_18k",
        "resource.uram",
    }),
}
_TARGET_GATE_CONDITIONS: dict[str, frozenset[str]] = {
    "correctness": frozenset({"csim_result_present", "cosim_result_present"}),
    "resource_fits": frozenset({"utilization_report_present"}),
    "post_route_timing": frozenset({"timing_gate_requested"}),
}
_TARGET_GATE_CONDITION_STAGES: dict[tuple[str, str], str] = {
    ("correctness", "csim_result_present"): "csim",
    ("correctness", "cosim_result_present"): "cosim",
    ("resource_fits", "utilization_report_present"): "post_route",
    ("post_route_timing", "timing_gate_requested"): "post_route",
}


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
    bindings: list[KnowledgeBinding] = field(default_factory=list)
    coverage: CoverageManifest | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version not in LEGACY_PACK_SCHEMA_VERSIONS | {PACK_SCHEMA_VERSION}:
            raise KnowledgePackError(
                f"unsupported knowledge pack schema {self.schema_version!r}; "
                f"expected one of {sorted(LEGACY_PACK_SCHEMA_VERSIONS | {PACK_SCHEMA_VERSION})!r}"
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
        rules_by_id = {rule.id: rule for rule in self.rules}
        rule_ids = set(rules_by_id)
        binding_ids: set[str] = set()
        for binding in self.bindings:
            if binding.knowledge_rule_id not in rule_ids:
                raise KnowledgePackError(
                    f"binding {binding.id!r} references a rule outside its pack"
                )
            if binding.id in binding_ids:
                raise KnowledgePackError(f"duplicate knowledge binding: {binding.id}")
            entails, errors = binding_entails_rule_condition(
                rules_by_id[binding.knowledge_rule_id], binding,
            )
            if not entails:
                raise KnowledgePackError(
                    f"binding {binding.id!r} does not entail its rule condition: "
                    + "; ".join(errors)
                )
            binding_ids.add(binding.id)
        if self.schema_version in LEGACY_PACK_SCHEMA_VERSIONS and (
            self.bindings or self.coverage is not None
        ):
            raise KnowledgePackError("v1 knowledge packs cannot contain v2 binding or coverage data")
        if self.coverage is not None:
            if self.coverage.pack_id != self.pack_id:
                raise KnowledgePackError("coverage manifest belongs to another pack")
            metadata_review_status = self.metadata.get("review_status")
            if (metadata_review_status is not None
                    and metadata_review_status != self.coverage.review_status):
                raise KnowledgePackError(
                    "pack metadata and coverage review_status must agree"
                )
            for entry in self.coverage.entries:
                if (entry.document_id, entry.document_version) not in declared:
                    raise KnowledgePackError(
                        f"coverage entry {entry.id!r} references an undeclared document"
                    )
                if any(item not in rule_ids for item in entry.rule_ids):
                    raise KnowledgePackError(
                        f"coverage entry {entry.id!r} references a rule outside its pack"
                    )
                if any(item not in binding_ids for item in entry.binding_ids):
                    raise KnowledgePackError(
                        f"coverage entry {entry.id!r} references a binding outside its pack"
                    )
            rule_coverage: Counter[str] = Counter()
            binding_coverage: Counter[str] = Counter()
            bindings_by_id = {item.id: item for item in self.bindings}
            for entry in self.coverage.entries:
                if entry.status != CoverageStatus.RULE:
                    continue
                rule_coverage.update(entry.rule_ids)
                binding_coverage.update(entry.binding_ids)
                for binding_id in entry.binding_ids:
                    if bindings_by_id[binding_id].knowledge_rule_id not in entry.rule_ids:
                        raise KnowledgePackError(
                            f"coverage entry {entry.id!r} places binding {binding_id!r} "
                            "under a different knowledge rule"
                        )
            missing_rules = sorted(rule_ids - set(rule_coverage))
            duplicate_rules = sorted(
                item for item, count in rule_coverage.items() if count != 1
            )
            missing_bindings = sorted(binding_ids - set(binding_coverage))
            duplicate_bindings = sorted(
                item for item, count in binding_coverage.items() if count != 1
            )
            if missing_rules or duplicate_rules:
                raise KnowledgePackError(
                    "every knowledge rule must be covered exactly once by a rule entry; "
                    f"missing={missing_rules!r}, non_unique={duplicate_rules!r}"
                )
            if missing_bindings or duplicate_bindings:
                raise KnowledgePackError(
                    "every knowledge binding must be covered exactly once by a rule entry; "
                    f"missing={missing_bindings!r}, non_unique={duplicate_bindings!r}"
                )
            covered_binding_ids: set[str] = set()
            for target in self.coverage.target_inventory:
                for binding_id in target.binding_ids:
                    binding = bindings_by_id.get(binding_id)
                    if binding is None:
                        raise KnowledgePackError(
                            f"target coverage {target.id!r} references a binding outside its pack"
                        )
                    if (binding.target_kind, binding.target) != (
                        target.target_kind, target.target,
                    ):
                        raise KnowledgePackError(
                            f"target coverage {target.id!r} references a binding for a "
                            "different target"
                        )
                    covered_binding_ids.add(binding_id)
            if self.coverage.target_inventory and covered_binding_ids != binding_ids:
                missing = sorted(binding_ids - covered_binding_ids)
                raise KnowledgePackError(
                    "coverage target inventory omits binding IDs: " + ", ".join(missing)
                )
            if self.coverage.target_inventory or self.bindings:
                try:
                    expected_targets = canonical_supported_targets(
                        self.coverage.coverage_scope,
                        self.coverage.target_registry_version,
                    )
                except KeyError as exc:
                    raise KnowledgePackError(str(exc)) from exc
                actual_targets = {
                    (item.target_kind, item.target)
                    for item in self.coverage.target_inventory
                }
                if actual_targets != expected_targets:
                    raise KnowledgePackError(
                        "coverage target inventory must exactly match the canonical "
                        f"{SUPPORTED_TARGET_REGISTRY_VERSION} registry for "
                        f"{self.coverage.coverage_scope!r}; "
                        f"missing={sorted(expected_targets - actual_targets)!r}, "
                        f"extra={sorted(actual_targets - expected_targets)!r}"
                    )
        elif self.schema_version == PACK_SCHEMA_VERSION and (self.rules or self.bindings):
            raise KnowledgePackError(
                "v2 knowledge packs with rules or bindings require an explicit coverage manifest"
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
            data["rules"] = [KnowledgeRule.from_dict(item)
                             for item in data.get("rules", [])]
            data["bindings"] = [KnowledgeBinding.from_dict(item)
                                for item in data.get("bindings", [])]
            if data.get("coverage") is not None:
                data["coverage"] = CoverageManifest.from_dict(data["coverage"])
        except TypeError as exc:
            raise KnowledgePackError(f"invalid knowledge rule: {exc}") from exc
        return cls(**data)

    @property
    def content_hash(self) -> str:
        return stable_hash(json_ready(self))

    @property
    def review_ready(self) -> bool:
        """Return whether this pack is classification-complete and reviewed."""

        return bool(
            self.coverage is not None
            and self.coverage.review_ready
            and self.metadata.get("review_status") == self.coverage.review_status
        )

    def inventory(self) -> dict[str, Any]:
        """Return public metadata needed to audit an explicit installation."""
        return {
            "pack_id": self.pack_id,
            "pack_schema_version": self.schema_version,
            "content_hash": self.content_hash,
            "activation_hash": knowledge_activation_hash(
                self.rules, self.bindings, self.coverage,
            ),
            "documents": [
                {
                    "document_id": item.document_id,
                    "document_version": item.document_version,
                    "official_url": item.official_url,
                }
                for item in sorted(
                    self.documents,
                    key=lambda item: (item.document_id, item.document_version),
                )
            ],
            "rule_ids": sorted(item.id for item in self.rules),
            "binding_ids": sorted(item.id for item in self.bindings),
            "coverage_id": self.coverage.id if self.coverage else None,
            "coverage_scope": self.coverage.coverage_scope if self.coverage else None,
            "target_registry_version": (
                self.coverage.target_registry_version if self.coverage else None
            ),
            "review_status": (
                self.coverage.review_status if self.coverage else
                self.metadata.get("review_status", "unreviewed")
            ),
            "review_ready": self.review_ready,
            "contains_document_body": False,
        }


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


def pack_migration_plan(pack: KnowledgePack) -> list[dict[str, str]]:
    """Return the explicit, non-mutating pack migration path for one pack."""
    if pack.schema_version == PACK_SCHEMA_VERSION:
        return []
    if pack.schema_version == "1.0":
        return [{
            "from_version": "1.0",
            "to_version": PACK_SCHEMA_VERSION,
            "description": (
                "adopt optional binding and coverage fields without changing legacy rules"
            ),
        }]
    raise KnowledgePackError(
        f"no explicit knowledge pack migration from {pack.schema_version!r}"
    )


def migrate_pack(pack: KnowledgePack, *, to_version: str = PACK_SCHEMA_VERSION) -> KnowledgePack:
    """Explicitly upgrade a v1 pack while preserving every rule identifier."""
    if to_version != PACK_SCHEMA_VERSION:
        raise KnowledgePackError(
            f"this build can migrate packs only to {PACK_SCHEMA_VERSION!r}"
        )
    plan = pack_migration_plan(pack)
    if not plan:
        return pack
    value = json_ready(pack)
    value["schema_version"] = PACK_SCHEMA_VERSION
    entries: dict[tuple[str, str, str], list[str]] = {}
    for rule in pack.rules:
        key = (rule.document_id, rule.document_version, rule.section)
        entries.setdefault(key, []).append(rule.id)
    value["coverage"] = {
        "pack_id": pack.pack_id,
        "coverage_scope": f"{pack.pack_id}.legacy_lexical_surface",
        "target_registry_version": SUPPORTED_TARGET_REGISTRY_VERSION,
        "review_status": "unreviewed",
        "reviewers": [],
        "source_hashes": {},
        "entries": [
            {
                "document_id": document_id,
                "document_version": document_version,
                "section": section,
                "status": "rule",
                "rule_ids": sorted(rule_ids),
                "binding_ids": [],
            }
            for (document_id, document_version, section), rule_ids
            in sorted(entries.items())
        ],
        "target_inventory": [],
        "metadata": {"migration": "v1_lexical_rules_to_explicit_coverage"},
    }
    migrated = KnowledgePack.from_dict(value)
    before = [(item.id, json_ready(item)) for item in pack.rules]
    after = [(item.id, json_ready(item)) for item in migrated.rules]
    if before != after:
        raise KnowledgePackError("knowledge pack migration would change legacy rules")
    return migrated


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


def canonical_context_scalar(value: Any) -> Any:
    """Return the canonical scalar representation used by every matcher.

    JSON booleans use private serialized tokens so they cannot compare equal
    to integers or boolean-like strings.  Other retrieval context strings
    remain case-insensitive.
    """
    if isinstance(value, bool):
        return (
            "hlsgraph.__context_bool__.true.v1"
            if value else "hlsgraph.__context_bool__.false.v1"
        )
    if isinstance(value, str):
        return value.casefold()
    return value


def _same(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left is right
    return canonical_context_scalar(left) == canonical_context_scalar(right)


def _constraint_entails(required: Any, condition: Any) -> bool:
    """Return whether a binding constraint is at least as strong as a condition.

    This is intentionally conservative.  Unknown operators, bare arrays, and
    incomparable version ranges do not count as implication.  The run-time
    matcher still evaluates the condition independently on one target-local
    context; this helper is the pack-load proof that the binding cannot omit or
    weaken the rule premise.
    """
    if condition in (None, "*"):
        return True
    if isinstance(required, (list, tuple)) or isinstance(condition, (list, tuple)):
        return False
    if isinstance(condition, Mapping):
        allowed = {"equals", "one_of", "min_version", "max_version", "required"}
        if set(condition) - allowed:
            return False
        if "required" in condition and condition["required"] is not True:
            return False
        # Any non-wildcard scalar is a present, singleton value.
        if not isinstance(required, Mapping):
            if required in (None, "*"):
                return False
            return _matches_constraint(condition, required)
        if (set(required) - allowed
                or ("required" in required and required["required"] is not True)):
            return False
        if condition.get("required") is True and not required:
            return False

        def candidates(value: Mapping[str, Any]) -> set[Any] | None:
            if "equals" in value:
                return {canonical_context_scalar(value["equals"])}
            if "one_of" in value:
                choices = value["one_of"]
                if not isinstance(choices, list) or not choices:
                    return None
                return {canonical_context_scalar(item) for item in choices}
            return None

        required_values = candidates(required)
        condition_values = candidates(condition)
        if condition_values is not None:
            if required_values is None or not required_values <= condition_values:
                return False
        elif condition.get("required") is True:
            # ``required: true`` alone merely asks for a present value.  Any
            # binding mapping with the same operator syntax is at least as
            # strong because the matcher rejects an absent actual value.
            pass
        if "min_version" in condition:
            if "min_version" not in required:
                return False
            if _version_key(required["min_version"]) < _version_key(condition["min_version"]):
                return False
        if "max_version" in condition:
            if "max_version" not in required:
                return False
            if _version_key(required["max_version"]) > _version_key(condition["max_version"]):
                return False
        return True
    if isinstance(required, Mapping):
        allowed = {"equals", "one_of", "min_version", "max_version", "required"}
        if set(required) - allowed:
            return False
        if "equals" in required:
            return _same(required["equals"], condition)
        if "one_of" in required:
            values = required["one_of"]
            return (isinstance(values, list) and bool(values)
                    and all(_same(item, condition) for item in values))
        return False
    return _same(required, condition)


def target_derived_condition_source(
    binding: KnowledgeBinding, key: str, condition: Any,
) -> str | None:
    """Return the audited source contract for one target-derived premise.

    The mapping is closed over exact public targets.  It is shared by pack
    validation and retrieval, preventing the loader and executor from
    disagreeing about a condition inferred from a binding target.
    """
    target_kind = str(binding.target_kind)
    target = str(binding.target)
    if (target_kind == "directive_kind" and key == "directive_kind"
            and target in _TARGET_DIRECTIVE_KINDS and _same(condition, target)):
        return _TARGET_CONDITION_SOURCE_DIRECTIVE
    if (target_kind == "artifact_kind"
            and _TARGET_ARTIFACT_CONDITIONS.get(target) == key
            and _same(condition, True)):
        return _TARGET_CONDITION_SOURCE_ARTIFACT
    if (target_kind == "predicate"
            and target in _TARGET_OBSERVATION_CONDITIONS.get(key, frozenset())
            and _same(condition, True)):
        return _TARGET_CONDITION_SOURCE_OBSERVATION
    if (target_kind == "gate_kind"
            and key in _TARGET_GATE_CONDITIONS.get(target, frozenset())
            and _same(condition, True)):
        return _TARGET_CONDITION_SOURCE_GATE
    return None


def target_derived_condition_stage(
    binding: KnowledgeBinding, key: str, source: str,
) -> str | None:
    """Return the sole stage that can witness one target-derived condition."""

    if source != _TARGET_CONDITION_SOURCE_GATE:
        return None
    return _TARGET_GATE_CONDITION_STAGES.get((
        str(binding.target), str(key),
    ))


def direct_condition_source(
    binding: KnowledgeBinding, key: str, condition: Any,
) -> str | None:
    """Return the closed evidence source for one explicitly bound premise.

    An arbitrary metadata field is not an executable condition capability.
    v0.3 registers only the direct premises used by the reviewed AMD/AXI
    surface; extending this table requires a matching runtime witness.
    """

    target_kind = str(binding.target_kind)
    target = str(binding.target)
    if (
        target_kind == "directive_kind"
        and target == "INTERFACE"
        and key == "interface_mode"
    ):
        return _DIRECT_CONDITION_SOURCE_INTERFACE_MODE
    if (
        target_kind == "directive_kind"
        and target == "INTERFACE"
        and key == "port_ownership_qualified"
        and _same(
            condition, "derived_from_unique_current_component_port_v1",
        )
    ):
        return _DIRECT_CONDITION_SOURCE_PORT_OWNERSHIP
    if (
        target_kind == "predicate"
        and target.startswith("directive.")
        and key == "requested_directive_present"
        and _same(condition, True)
    ):
        return _DIRECT_CONDITION_SOURCE_REQUESTED_DIRECTIVE
    if (
        target_kind == "directive_kind"
        and target in {"PIPELINE", "UNROLL", "ARRAY_PARTITION", "INLINE"}
        and key == "directive_semantic_mode"
    ):
        return _DIRECT_CONDITION_SOURCE_DIRECTIVE_OPTIONS
    return None


def _constraint_mentions_exact(constraint: Any, expected: Any) -> bool:
    if isinstance(constraint, Mapping):
        if set(constraint) - {"equals", "one_of", "min_version", "max_version", "required"}:
            return False
        if "equals" in constraint:
            return _same(constraint["equals"], expected)
        values = constraint.get("one_of")
        return isinstance(values, list) and bool(values) and all(
            _same(item, expected) for item in values
        )
    return not isinstance(constraint, (list, tuple)) and _same(constraint, expected)


def _requires_value(constraint: Any) -> bool:
    return isinstance(constraint, Mapping) and constraint.get("required") is True


def _target_condition_source_closed(
    binding: KnowledgeBinding, key: str, source: str,
) -> bool:
    required = binding.required_context
    if source == _TARGET_CONDITION_SOURCE_DIRECTIVE:
        declaration_closed = _constraint_mentions_exact(
            required.get("directive_source_declaration_qualified"),
            "derived_from_current_directive_source_declaration_v1",
        ) or _constraint_mentions_exact(
            required.get("dependence_operand_resolved"),
            "derived_from_current_dependence_operand_v1",
        )
        return (
            _requires_value(required.get("directive_instance_id"))
            and _requires_value(required.get("scope_id"))
            and declaration_closed
        )
    if source == _TARGET_CONDITION_SOURCE_ARTIFACT:
        if binding.target == "constraint.xdc":
            return _constraint_mentions_exact(
                required.get("constraint_input_evidence_qualified"),
                "derived_from_unique_live_snapshot_input_v1",
            ) and all(_requires_value(required.get(key)) for key in (
                "artifact_sha256", "constraint_hash", "constraint_artifact_identity",
            ))
        return _constraint_mentions_exact(
            required.get("tool_artifact_evidence_qualified"),
            "derived_from_declared_live_tool_output_v1",
        ) and all(_requires_value(required.get(key)) for key in (
            "tool_artifact_identity", "tool_artifact_run_identity",
        ))
    if source == _TARGET_CONDITION_SOURCE_OBSERVATION:
        return (
            _constraint_mentions_exact(
                required.get("observation_evidence_qualified"),
                "derived_from_typed_observation_evidence_v1",
            )
            and _constraint_mentions_exact(
                required.get("snapshot_association"), "verified",
            )
            and all(_requires_value(required.get(key)) for key in (
                "observation_instance_id", "observation_artifact_identity",
                "observation_run_identity",
            ))
        )
    if source == _TARGET_CONDITION_SOURCE_GATE:
        expected_stage = target_derived_condition_stage(binding, key, source)
        return (
            expected_stage is not None
            and _constraint_mentions_exact(
                required.get("stage"), expected_stage,
            )
            and _constraint_mentions_exact(
                required.get("gate_evidence_qualified"),
                "derived_from_typed_evidence_v1",
            )
            and _constraint_mentions_exact(
                required.get("snapshot_association"), "verified",
            )
        )
    return False


def _direct_condition_source_closed(
    binding: KnowledgeBinding, source: str,
) -> bool:
    required = binding.required_context
    if source == _DIRECT_CONDITION_SOURCE_INTERFACE_MODE:
        return (
            _constraint_mentions_exact(
                required.get("directive_source_declaration_qualified"),
                "derived_from_current_directive_source_declaration_v1",
            )
            and _requires_value(required.get("directive_source_identity"))
            and _requires_value(required.get("directive_instance_id"))
            and _requires_value(required.get("scope_id"))
            and _requires_value(required.get("port_id"))
        )
    if source == _DIRECT_CONDITION_SOURCE_PORT_OWNERSHIP:
        return (
            _constraint_mentions_exact(
                required.get("port_ownership_qualified"),
                "derived_from_unique_current_component_port_v1",
            )
            and _constraint_mentions_exact(
                required.get("directive_source_declaration_qualified"),
                "derived_from_current_directive_source_declaration_v1",
            )
            and all(_requires_value(required.get(key)) for key in (
                "port_owner_id", "configured_component_id",
                "port_ownership_identity", "directive_source_identity",
            ))
        )
    if source == _DIRECT_CONDITION_SOURCE_REQUESTED_DIRECTIVE:
        return (
            _constraint_mentions_exact(
                required.get("requested_directive_present"), True,
            )
            and _requires_value(required.get("directive_instance_id"))
            and _requires_value(required.get("scope_id"))
            and "scope_kind" in required
        )
    if source == _DIRECT_CONDITION_SOURCE_DIRECTIVE_OPTIONS:
        return (
            _constraint_mentions_exact(
                required.get("directive_options_qualified"),
                "derived_from_current_directive_options_v1",
            )
            and _constraint_mentions_exact(
                required.get("directive_source_declaration_qualified"),
                "derived_from_current_directive_source_declaration_v1",
            )
            and all(_requires_value(required.get(key)) for key in (
                "directive_options_identity", "directive_source_identity",
                "directive_instance_id", "scope_id",
            ))
        )
    return False


def binding_entails_rule_condition(
    rule: KnowledgeRule, binding: KnowledgeBinding,
) -> tuple[bool, tuple[str, ...]]:
    """Audit condition implication for one executable binding.

    Direct context premises must be constrained at least as strongly by the
    binding.  A premise absent from ``required_context`` is accepted only when
    its exact target mapping and evidence-closure contract are registered
    above.  No name-pattern or truthy fallback exists.
    """
    errors: list[str] = []
    for key, condition in rule.condition.items():
        if key in binding.required_context:
            if not _constraint_entails(binding.required_context[key], condition):
                errors.append(f"condition {key!r} is weakened or contradicted")
                continue
            source = direct_condition_source(binding, key, condition)
            if source is None:
                errors.append(
                    f"condition {key!r} has no registered direct evidence source"
                )
            elif not _direct_condition_source_closed(binding, source):
                errors.append(
                    f"condition {key!r} direct premise lacks {source} evidence closure"
                )
            continue
        source = target_derived_condition_source(binding, key, condition)
        if source is None:
            errors.append(f"condition {key!r} has no explicit binding premise")
        elif not _target_condition_source_closed(binding, key, source):
            errors.append(
                f"condition {key!r} target premise lacks {source} evidence closure"
            )
    return (not errors, tuple(errors))


def _matches_constraint(constraint: Any, actual: Any) -> bool:
    if constraint in (None, "*"):
        return True
    if isinstance(actual, (list, tuple, set, frozenset)):
        return bool(actual) and any(
            _matches_constraint(constraint, item) for item in actual
        )
    if isinstance(constraint, (list, tuple)):
        # KnowledgeRule and KnowledgeBinding loaders reject this ambiguous
        # shorthand. Keep the matcher fail-closed for mutated in-memory model
        # instances instead of silently treating a JSON array as ``one_of``.
        return False
    if isinstance(constraint, Mapping):
        allowed = {"equals", "one_of", "min_version", "max_version", "required"}
        unknown = set(constraint) - allowed
        if unknown:
            raise KnowledgePackError(f"unsupported applicability operators: {sorted(unknown)}")
        if "required" in constraint and constraint["required"] is not True:
            raise KnowledgePackError("applicability required operator only accepts true")
        if actual is None:
            return False
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
    if actual is None:
        return False
    return _same(constraint, actual)


def matches_applicability(rule: KnowledgeRule, context: Mapping[str, Any]) -> bool:
    """Return whether every constraint declared by ``rule`` is met.

    A missing context value does not satisfy a restrictive rule.  This fail-closed
    behavior prevents a tool- or stage-specific rule from being presented as
    generally applicable. ``"*"`` is the only unrestricted value, and
    alternatives must use ``{"one_of": [...]}`` rather than a bare array.
    """

    return all(_matches_constraint(constraint, context.get(key))
               for key, constraint in rule.applicability.items())


def matches_binding_constraints(
    binding: KnowledgeBinding,
    *,
    target_kind: str,
    target: str,
    context: Mapping[str, Any],
) -> bool:
    """Inspect whether author-supplied scalar values satisfy a binding shape.

    This helper has no design-instance authority.  It is intended for pack
    authoring and tests only; executable retrieval requires a process-local
    ``AttestedBindingContext`` issued from current graph and ledger evidence.
    A bare JSON array is deliberately not an alternatives operator here.
    """
    return (
        binding.target_kind == target_kind
        and binding.target == target
        and all(not isinstance(constraint, (list, tuple))
                and _matches_constraint(constraint, context.get(key))
                for key, constraint in binding.required_context.items())
    )


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

    def all_bindings(self) -> list[KnowledgeBinding]:
        return [binding for pack in self.packs for binding in pack.bindings]

    def coverage_manifests(self) -> list[CoverageManifest]:
        return [pack.coverage for pack in self.packs if pack.coverage is not None]

    def install(
        self, store: Any, *, pack_ids: Iterable[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Explicitly install selected immutable packs into a ledger.

        ``GraphBundle.open`` never calls this method.  It is safe to use as an
        explicit sync operation: byte-identical packs report ``unchanged``;
        reused pack IDs with different content fail in the store.
        """
        if isinstance(pack_ids, (str, bytes)):
            raise KnowledgePackError("pack_ids must be an iterable of complete pack IDs")
        raw_requested = list(pack_ids) if pack_ids is not None else None
        if raw_requested is not None and (
            any(not isinstance(item, str) or not item.strip()
                for item in raw_requested)
            or len(set(raw_requested)) != len(raw_requested)
        ):
            raise KnowledgePackError("pack_ids must be unique non-empty strings")
        requested = set(raw_requested) if raw_requested is not None else None
        available = {pack.pack_id: pack for pack in self.packs}
        if requested is not None:
            unknown = sorted(requested - set(available))
            if unknown:
                raise KnowledgePackError(
                    "unknown knowledge pack IDs: " + ", ".join(unknown)
                )
        result: list[dict[str, Any]] = []
        for pack_id in sorted(available):
            if requested is not None and pack_id not in requested:
                continue
            pack = available[pack_id]
            if pack.bindings and not pack.review_ready:
                raise KnowledgePackError(
                    f"knowledge pack {pack.pack_id!r} contains executable bindings "
                    "but is not review_ready"
                )
            installed = store.install_knowledge_pack(
                pack_id=pack.pack_id,
                pack_schema_version=pack.schema_version,
                content_hash=pack.content_hash,
                installed_at=_utc_now(),
                inventory=pack.inventory(),
                rules=pack.rules,
                bindings=pack.bindings,
                coverage=pack.coverage,
            )
            result.append({
                "pack_id": pack.pack_id,
                "content_hash": pack.content_hash,
                "status": "installed" if installed else "unchanged",
            })
        return result

    def sync(self, store: Any) -> list[dict[str, Any]]:
        """Explicit alias for installing every pack in this catalog."""
        return self.install(store)

    def filter(
        self,
        *,
        document_id: str | None = None,
        document_version: str | None = None,
        applicability: Mapping[str, Any] | None = None,
    ) -> list[KnowledgeRule]:
        return filter_rules(self.all_rules(), document_id=document_id,
                            document_version=document_version, applicability=applicability)

    def binding_candidates_for(
        self, *, target_kind: str, target: str,
    ) -> list[KnowledgeBinding]:
        """Return metadata candidates by exact public target, never activate."""
        return sorted(
            (item for item in self.all_bindings()
             if item.target_kind == target_kind and item.target == target),
            key=lambda item: item.id,
        )


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
        if (not isinstance(self.document_version, str)
                or not self.document_version.strip()
                or len(self.document_version) > 128
                or "\x00" in self.document_version):
            raise KnowledgePackError(
                "local document version must be a bounded non-empty string"
            )
        if (not isinstance(self.uri, str) or not self.uri.strip()
                or len(self.uri) > 4_096 or "\x00" in self.uri):
            raise KnowledgePackError(
                "local document URI must be a bounded non-empty string"
            )
        if self.title is not None and (
            not isinstance(self.title, str) or not self.title.strip()
            or len(self.title) > 512 or "\x00" in self.title
        ):
            raise KnowledgePackError(
                "local document title must be a bounded non-empty string or None"
            )
        if self.media_type is not None and (
            not isinstance(self.media_type, str) or not self.media_type.strip()
            or len(self.media_type) > 256 or "\x00" in self.media_type
        ):
            raise KnowledgePackError(
                "local document media_type must be a bounded non-empty string or None"
            )
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


def _is_link_or_reparse(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError:
        return False
    if stat.S_ISLNK(info.st_mode):
        return True
    attributes = getattr(info, "st_file_attributes", 0)
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & reparse)


def _has_link_or_reparse_ancestor(path: Path) -> bool:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if current.exists() and _is_link_or_reparse(current):
            return True
    return False


def _read_stable_local_file(path: Path, *, max_bytes: int = 32 * 1024 * 1024) -> tuple[bytes, os.stat_result]:
    """Read a regular file once, rejecting links, replacement, and size bombs."""
    if max_bytes < 1:
        raise ValueError("max_bytes must be positive")
    if _is_link_or_reparse(path):
        raise KnowledgePackError(f"local knowledge document cannot be a link/reparse point: {path}")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise KnowledgePackError(f"cannot open local knowledge document {path}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise KnowledgePackError(f"local knowledge document is not a regular file: {path}")
        if before.st_size > max_bytes:
            raise KnowledgePackError(
                f"local knowledge document exceeds {max_bytes} bytes: {path}"
            )
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        if len(data) > max_bytes:
            raise KnowledgePackError(
                f"local knowledge document exceeds {max_bytes} bytes: {path}"
            )
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        current = path.lstat()
    except OSError as exc:
        raise KnowledgePackError(f"local knowledge document changed while indexed: {path}") from exc
    identity_before = (
        before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns,
    )
    identity_current = (
        current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns,
    )
    if identity_before != identity_after or identity_after != identity_current:
        raise KnowledgePackError(f"local knowledge document changed while indexed: {path}")
    if _is_link_or_reparse(path):
        raise KnowledgePackError(f"local knowledge document became a link/reparse point: {path}")
    return data, after


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

    original = Path(path)
    if not original.is_file():
        raise FileNotFoundError(path)
    if _has_link_or_reparse_ancestor(original):
        raise KnowledgePackError(
            "local document paths containing links/reparse points are not indexable"
        )
    path = original.resolve()
    data, after = _read_stable_local_file(path)
    return LocalDocumentMetadata(
        document_id=document_id,
        document_version=document_version,
        uri=uri or path.as_uri(),
        sha256=hashlib.sha256(data).hexdigest(),
        size=after.st_size,
        modified_ns=after.st_mtime_ns,
        indexed_at=_utc_now(),
        title=title,
        media_type=mimetypes.guess_type(path.name)[0] or "application/octet-stream",
        official_url=official_url,
    )


def save_local_index(entries: Iterable[LocalDocumentMetadata], path: str | Path) -> Path:
    """Write a deterministic metadata-only index; document bytes are not copied."""

    ordered = sorted(
        (LocalDocumentMetadata.from_dict(item.to_dict()) for item in entries),
        key=lambda item: (item.document_id, item.document_version, item.uri),
    )
    payload = {
        "schema_version": LOCAL_INDEX_SCHEMA_VERSION,
        "documents": [item.to_dict() for item in ordered],
    }
    target = Path(path)
    if _has_link_or_reparse_ancestor(target.parent):
        raise KnowledgePackError(
            "local document index path cannot traverse links/reparse points"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    if (_has_link_or_reparse_ancestor(target.parent)
            or (target.exists() and _is_link_or_reparse(target))):
        raise KnowledgePackError(
            "local document index path cannot traverse links/reparse points"
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            descriptor = -1
            stream.write(json.dumps(
                payload, ensure_ascii=False, indent=2, sort_keys=True,
                allow_nan=False,
            ) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        if target.exists() and _is_link_or_reparse(target):
            raise KnowledgePackError(
                "local document index became a link/reparse point"
            )
        os.replace(temporary, target)
        if os.name != "nt":
            os.chmod(target, 0o600)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
    return target


def load_local_index(path: str | Path) -> list[LocalDocumentMetadata]:
    """Load a metadata-only local index and reject content-bearing fields."""

    path = Path(path)
    if _has_link_or_reparse_ancestor(path):
        raise KnowledgePackError(
            "local document index path cannot traverse links/reparse points"
        )
    try:
        data, _info = _read_stable_local_file(path, max_bytes=8 * 1024 * 1024)
        payload = json.loads(data.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
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
