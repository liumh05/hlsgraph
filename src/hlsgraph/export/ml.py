"""Deterministic, torch-free ML interchange exports."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Mapping

from ..bundle import GraphBundle
from ..evidence_policy import (
    TOOL_EVIDENCE_POLICY_VERSION,
    real_tool_run_claim_error,
    run_claims_tool_truth,
    successful_fresh_tool_run_error,
    tool_evidence_compatibility_error,
    tool_run_manifest_identity_error,
)
from ..model import (
    DatasetManifest, LabelSpec, Stage, hash_artifact_bytes, json_ready,
    reject_embedded_body_fields, stable_hash,
)
from ..run_projection import (
    PUBLIC_FAILURE_CLASSES, PUBLIC_GATE_KINDS, PUBLIC_GATE_STATUSES,
    PUBLIC_RUN_STATUSES, public_enum, public_identifier,
    public_identifier_list, public_sha256, public_timestamp,
    sanitize_run_metadata,
)
from ..query import managed_artifact_integrity, static_feature_predicate_allowed
from ..version import FEATURE_SCHEMA_VERSION, SCHEMA_VERSION


_OUTCOME_FEATURE_PREFIXES = (
    "qor", "label", "prediction", "achieved", "measured", "observed",
    "timing", "latency", "cosim", "csim", "profile", "power", "utilization",
    "wns", "tns", "slack", "gate", "lut", "ff", "bram", "uram", "dsp",
    "fmax", "throughput", "resource_used", "resource_usage",
)

_TOOL_EVIDENCE_AUTHORITIES = frozenset({
    "tool_observation", "verification_evidence", "physical_measurement",
})

_STATIC_FEATURE_EVIDENCE_AUTHORITIES = frozenset({
    "declared_constraint", "static_fact", "compiler_decision", "derived_fact",
    "synthetic",
})

# A container in graph attrs is never a feature merely because its top-level
# name was allowlisted.  Each container needs a positive, versioned schema.
# The directive ``options`` and shape ``dims`` containers are the only reviewed
# container contracts.  The option names below are requested directive
# parameters, not achieved results.
_DIRECTIVE_SCALAR_OPTION_NAMES = (
    "avg", "bundle", "class", "compact", "core", "cycle", "depth",
    "dependent", "dim", "direct_io", "direction",
    "disable_start_propagation", "distance", "enable_flush", "factor",
    "flushable", "force", "function", "ii", "impl", "instances",
    "latency", "limit", "max", "max_read_burst_length", "max_widen_bitwidth",
    "max_write_burst_length", "min", "mode", "name", "num_read_outstanding",
    "num_write_outstanding", "off", "offset", "op", "operation", "port",
    "recursive", "region", "register", "register_mode", "rewind",
    "skip_exit_check", "storage_type", "strict_mode", "style", "target_ti",
    "type", "variable",
)
_DIRECTIVE_FLAG_VALUES = (
    "block", "complete", "cyclic", "disable_start_propagation",
    "enable_flush", "flushable", "force", "off", "recursive", "region",
    "rewind", "skip_exit_check", "strict_mode",
)

_SCALAR_SCHEMA = {"type": "scalar"}
_DIRECTIVE_OPTION_PROPERTIES: dict[str, dict[str, Any]] = {
    name: _SCALAR_SCHEMA for name in _DIRECTIVE_SCALAR_OPTION_NAMES
}
_DIRECTIVE_OPTION_PROPERTIES["flags"] = {
    "type": "array",
    "items": {"type": "string", "enum": list(_DIRECTIVE_FLAG_VALUES)},
}

# This JSON-schema-like document is both executable and emitted verbatim in
# feature_spec.json/PyG metadata.  There is deliberately no wildcard or
# ``additionalProperties`` escape hatch for plugin-provided containers.
NESTED_STATIC_FEATURE_SCHEMA: dict[str, dict[str, Any]] = {
    "dims": {
        "type": "positive_integer_list",
        "min_items": 1,
        "max_items": 16,
        "min_value": 1,
        "max_value": 2_147_483_647,
    },
    "options": {
        "type": "object",
        "applicability": {
            "record": "node",
            "entity_kind": "hls.directive",
            "authority": "declared_constraint",
        },
        "properties": _DIRECTIVE_OPTION_PROPERTIES,
        "additionalProperties": False,
    },
}

_OMIT = object()


def _safe_scalar(value: Any) -> bool:
    """Return whether *value* is a finite JSON scalar, never a container."""
    if value is None or isinstance(value, (str, bool, int)):
        return True
    return isinstance(value, float) and math.isfinite(value)


def _static_evidence_value(value: Any, path: str = "value") -> Any:
    """Validate a selected derivation value without lossy field dropping."""
    reject_embedded_body_fields(value, f"feature evidence {path}")
    if _safe_scalar(value):
        return value
    if isinstance(value, (list, tuple)):
        return [
            _static_evidence_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, dict):
        if any(not isinstance(key, str) or not key.strip() for key in value):
            raise ValueError(f"feature evidence {path} requires non-empty string keys")
        result: dict[str, Any] = {}
        for key in sorted(value):
            if not static_feature_predicate_allowed(f"feature.{key}"):
                raise ValueError(
                    f"feature evidence {path} contains an outcome-shaped key {key!r}"
                )
            result[key] = _static_evidence_value(value[key], f"{path}.{key}")
        return result
    raise ValueError(
        f"feature evidence {path} contains unsupported {type(value).__name__} data"
    )


def _apply_feature_schema(value: Any, schema: dict[str, Any]) -> Any:
    """Project a value through a positive schema, returning ``_OMIT`` on mismatch."""
    kind = schema.get("type")
    if kind == "positive_integer_list":
        # This is deliberately stricter than the generic array projection:
        # tuples and partially-valid lists are not silently normalized.  A
        # malformed dimension vector is unknown, not a shorter valid shape.
        if not isinstance(value, list):
            return _OMIT
        min_items = schema.get("min_items")
        max_items = schema.get("max_items")
        min_value = schema.get("min_value")
        max_value = schema.get("max_value")
        limits = (min_items, max_items, min_value, max_value)
        if any(type(item) is not int for item in limits):
            return _OMIT
        if not min_items <= len(value) <= max_items:
            return _OMIT
        if any(type(item) is not int or not min_value <= item <= max_value
               for item in value):
            return _OMIT
        return list(value)
    if kind == "scalar":
        return value if _safe_scalar(value) else _OMIT
    if kind == "string":
        if not isinstance(value, str):
            return _OMIT
        choices = schema.get("enum")
        return value if choices is None or value in choices else _OMIT
    if kind == "array":
        if not isinstance(value, (list, tuple)):
            return _OMIT
        item_schema = schema.get("items")
        if not isinstance(item_schema, dict):
            return _OMIT
        projected = []
        for item in value:
            safe = _apply_feature_schema(item, item_schema)
            if safe is not _OMIT:
                projected.append(safe)
        return projected
    if kind == "object":
        if not isinstance(value, dict):
            return _OMIT
        properties = schema.get("properties")
        if not isinstance(properties, dict):
            return _OMIT
        projected: dict[str, Any] = {}
        for key in sorted(value, key=str):
            child_schema = properties.get(str(key))
            if not isinstance(child_schema, dict):
                continue
            safe = _apply_feature_schema(value[key], child_schema)
            if safe is not _OMIT:
                projected[str(key)] = safe
        return projected
    return _OMIT


def _static_features(
    value: dict[str, Any], allowlist: set[str], *,
    entity_kind: str | None = None, authority: Any = None,
) -> dict[str, Any]:
    """Apply the top-level allowlist and the positive nested feature schema."""
    result: dict[str, Any] = {}
    for key in sorted(value, key=str):
        name = str(key)
        if name not in allowlist:
            continue
        normalized = name.casefold().replace("-", "_")
        if normalized.startswith(_OUTCOME_FEATURE_PREFIXES):
            continue
        item = value[key]
        schema = NESTED_STATIC_FEATURE_SCHEMA.get(name)
        if schema is not None:
            applicability = schema.get("applicability", {})
            if (applicability.get("record") == "node"
                    and (entity_kind != applicability.get("entity_kind")
                         or str(authority) != applicability.get("authority"))):
                continue
            projected = _apply_feature_schema(item, schema)
        else:
            # Unknown plugin containers are not recursively copied.  Dataset
            # authors may allowlist reviewed scalar fields, but a new container
            # requires a future feature-schema revision and explicit adapter.
            projected = item if _safe_scalar(item) else _OMIT
        if projected is not _OMIT:
            result[name] = projected
    return result


def _feature_schema_document(allowlist: set[str]) -> dict[str, Any]:
    """Return the exact positive schema applied to one dataset export."""
    containers = {
        key: json.loads(json.dumps(value, sort_keys=True))
        for key, value in sorted(NESTED_STATIC_FEATURE_SCHEMA.items())
        if key in allowlist
    }
    return {
        "top_level_additional_properties": False,
        "top_level_allowlist": sorted(allowlist),
        "top_level_excluded_outcome_prefixes": list(_OUTCOME_FEATURE_PREFIXES),
        "top_level_scalar_types": ["null", "boolean", "integer", "finite_number", "string"],
        "containers": containers,
        "unknown_container_policy": "exclude",
        "unknown_container_key_policy": "exclude",
    }


def _validated_dataset_manifest(dataset: DatasetManifest) -> DatasetManifest:
    """Round-trip mutable dataclasses so every construction-time guard runs again."""
    raw = json_ready(dataset)
    raw["labels"] = [LabelSpec(**item) for item in raw.get("labels", [])]
    return DatasetManifest(**raw)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(json_ready(value), ensure_ascii=False, indent=2,
                               sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: Iterable[Any]) -> list[dict[str, Any]]:
    rows = [json_ready(value) for value in values]
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False) + "\n")
    return rows


def _public_digest(value: Any) -> str | None:
    """Return an exact SHA-256, or re-hash a malformed legacy identity."""
    if value is None:
        return None
    digest = public_sha256(value)
    if digest is not None:
        return digest
    return stable_hash(value)


def _public_identity_fields(
    raw: Mapping[str, Any], names: tuple[str, ...],
) -> dict[str, Any]:
    """Project bounded identifiers and hash invalid non-null legacy values."""
    result: dict[str, Any] = {}
    for name in names:
        value = raw.get(name)
        safe = public_identifier(value)
        result[name] = safe
        if value is not None and safe is None:
            result[f"{name}_hash"] = stable_hash(value)
    return result


def _public_toolchain(value: Any) -> dict[str, Any]:
    """Dataset identity without executable paths or arbitrary tool metadata."""
    raw = json_ready(value)
    if not isinstance(raw, Mapping):
        return {}
    return _public_identity_fields(
        raw, ("id", "vendor", "name", "version", "build"),
    ) | {
        "environment_hash": _public_digest(raw.get("environment_hash")),
        "metadata_hash": stable_hash(raw.get("metadata", {})),
    }


def _looks_like_path(value: str) -> bool:
    return (
        value.startswith(("/", "\\\\"))
        or re.match(r"^[A-Za-z]:[\\/]", value) is not None
    )


def _public_config_value(value: Any) -> Any:
    """Retain configuration shape while hashing free-form string values."""
    if isinstance(value, dict):
        # Extension dictionaries are untrusted public-surface inputs.  A
        # private path or prose fragment in a key is just as sensitive as one
        # in a value, so only bounded identifier keys retain their spelling.
        return {
            safe_key: _public_config_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if (safe_key := public_identifier(key)) is not None
        }
    if isinstance(value, (list, tuple)):
        return [_public_config_value(item) for item in value]
    if isinstance(value, str):
        # Constraint/memory-topology dictionaries are extension points without
        # a closed string vocabulary.  Hash all string values so a plugin cannot
        # smuggle private prose through a plausible-looking short token.
        return {"redacted": True, "value_hash": stable_hash(value)}
    return value


def _public_resource_map(value: Any) -> dict[str, int | float]:
    """Keep only named, finite, non-negative resource quantities."""
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, int | float] = {}
    for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
        safe_key = public_identifier(key)
        if (safe_key is None or not isinstance(item, (int, float))
                or isinstance(item, bool) or not math.isfinite(float(item))
                or float(item) < 0):
            continue
        result[safe_key] = item
    return result


def _public_nonnegative_number(value: Any, *, positive: bool = False) -> int | float | None:
    if (not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(float(value))):
        return None
    if float(value) < 0 or (positive and float(value) <= 0):
        return None
    return value


def _public_target(value: Any, artifacts: list[Any]) -> dict[str, Any]:
    """Target identity without clock-source paths or arbitrary metadata."""
    raw = json_ready(value)
    if not isinstance(raw, Mapping):
        return {}
    artifacts_by_uri = {item.uri: item for item in artifacts}
    clocks = []
    raw_clocks = raw.get("clocks")
    if not isinstance(raw_clocks, list):
        raw_clocks = []
    for clock in raw_clocks:
        if not isinstance(clock, Mapping):
            continue
        clock_name = clock.get("name")
        public_clock = {
            "name": public_identifier(clock_name),
            "period_ns": _public_nonnegative_number(
                clock.get("period_ns"), positive=True,
            ),
            "uncertainty_ns": _public_nonnegative_number(
                clock.get("uncertainty_ns"),
            ),
        }
        if clock_name and public_clock["name"] is None:
            public_clock["name_hash"] = stable_hash(clock_name)
        source = clock.get("source")
        if source:
            artifact = artifacts_by_uri.get(str(source))
            if artifact is not None:
                public_clock["source_artifact"] = {
                    "artifact_id": artifact.id,
                    "sha256": artifact.sha256,
                }
            else:
                public_clock["source_hash"] = stable_hash(source)
        clocks.append(public_clock)
    platform = raw.get("platform")
    public_platform = public_identifier(platform)
    platform_hash = public_sha256(raw.get("platform_hash"))
    if platform_hash is None:
        digest_source = platform if platform is not None else raw.get("platform_hash")
        platform_hash = _public_digest(digest_source)
    result = _public_identity_fields(
        raw, ("vendor", "part", "package", "board"),
    )
    speed_grade = raw.get("speed_grade")
    public_speed_grade = public_identifier(speed_grade)
    if (public_speed_grade is None and isinstance(speed_grade, str)
            and re.fullmatch(r"[+-][A-Za-z0-9][A-Za-z0-9_.:+-]{0,254}", speed_grade)):
        public_speed_grade = speed_grade
    result["speed_grade"] = public_speed_grade
    if speed_grade is not None and public_speed_grade is None:
        result["speed_grade_hash"] = stable_hash(speed_grade)
    result |= {
        "platform": public_platform,
        "platform_hash": platform_hash,
        "capacities": _public_resource_map(raw.get("capacities")),
        "reserved_resources": _public_resource_map(raw.get("reserved_resources")),
        "clocks": clocks,
        "memory_topology": _public_config_value(raw.get("memory_topology", [])),
        "metadata_hash": stable_hash(raw.get("metadata", {})),
    }
    if platform is not None and public_platform is None:
        result["platform_identity_hash"] = stable_hash(platform)
    return result


def _public_run_scope(value: Any) -> str | None:
    if value is None:
        return None
    if (isinstance(value, str) and len(value) <= 256
            and re.fullmatch(r"[A-Za-z0-9_.:+-]+", value)):
        return value
    return f"sha256:{stable_hash(value)}"


def _public_run(value: Any) -> dict[str, Any]:
    """Run provenance for datasets, without argv, paths, messages, or backend details."""
    ready = json_ready(value)
    raw = dict(ready) if isinstance(ready, Mapping) else {}
    safe_metadata = sanitize_run_metadata(raw.get("metadata"))
    elapsed = raw.get("elapsed_s")
    gates = raw.get("gates") if isinstance(raw.get("gates"), list) else []
    return {
        "id": _public_run_scope(raw.get("id")),
        "snapshot_id": _public_run_scope(raw.get("snapshot_id")),
        "stage": _public_run_scope(raw.get("stage")),
        "backend": _public_run_scope(raw.get("backend")),
        "request_hash": public_sha256(raw.get("request_hash")),
        "toolchain_id": _public_run_scope(raw.get("toolchain_id")),
        "status": public_enum(raw.get("status"), PUBLIC_RUN_STATUSES),
        "environment_hash": public_sha256(raw.get("environment_hash")),
        "input_artifact_ids": [_public_run_scope(item)
                               for item in public_identifier_list(
                                   raw.get("input_artifact_ids"))],
        "output_artifact_ids": [_public_run_scope(item)
                                for item in public_identifier_list(
                                    raw.get("output_artifact_ids"))],
        "diagnostics": [_public_run_scope(item) for item in
                        public_identifier_list(raw.get("diagnostics"))],
        "failure_class": public_enum(
            raw.get("failure_class"), PUBLIC_FAILURE_CLASSES
        ),
        "exit_code": (raw.get("exit_code") if isinstance(raw.get("exit_code"), int)
                      and not isinstance(raw.get("exit_code"), bool) else None),
        "attempt": (raw.get("attempt") if isinstance(raw.get("attempt"), int)
                    and not isinstance(raw.get("attempt"), bool)
                    and raw.get("attempt") >= 1 else None),
        "started_at": public_timestamp(raw.get("started_at")),
        "finished_at": public_timestamp(raw.get("finished_at")),
        "elapsed_s": (elapsed if isinstance(elapsed, (int, float))
                      and not isinstance(elapsed, bool) and math.isfinite(elapsed) else None),
        "campaign_id": safe_metadata.get("campaign_id"),
        "workload_id": safe_metadata.get("workload_id"),
        "tool_truth": safe_metadata.get("tool_truth") is True,
        "fresh_tool_truth": safe_metadata.get("fresh_tool_truth") is True,
        "metadata": safe_metadata,
        "gates": [{
            "kind": public_enum(gate.get("kind"), PUBLIC_GATE_KINDS),
            "status": public_enum(gate.get("status"), PUBLIC_GATE_STATUSES),
            "evidence_ids": [_public_run_scope(item) for item in
                             public_identifier_list(gate.get("evidence_ids"))],
            "reason_present": gate.get("reason") is not None,
        } for gate in gates if isinstance(gate, Mapping)],
    }


def _public_constraints(manifest: Any, artifacts: list[Any]) -> dict[str, Any]:
    """Preserve declared constraints while replacing XDC paths/bodies with CAS refs."""
    raw = json_ready(manifest.constraints)
    artifacts_by_uri = {item.uri: item for item in artifacts}
    xdc_artifacts = []
    for relative in manifest.constraints.xdc_files:
        artifact = artifacts_by_uri.get(relative)
        if artifact is None:
            raise ValueError(
                f"constraint XDC is not represented by a snapshot artifact: {relative!r}"
            )
        xdc_artifacts.append({
            "artifact_id": artifact.id,
            "sha256": artifact.sha256,
            "size": artifact.size,
            "access": str(artifact.access),
            "license": artifact.license,
        })
    assumptions = list(raw.get("assumptions", []))
    return {
        "performance": _public_config_value(raw.get("performance", {})),
        "resources": _public_config_value(raw.get("resources", {})),
        "power": _public_config_value(raw.get("power", {})),
        "numerical": _public_config_value(raw.get("numerical", {})),
        "interfaces": _public_config_value(raw.get("interfaces", {})),
        "xdc_artifacts": xdc_artifacts,
        "assumption_count": len(assumptions),
        "assumptions_hash": stable_hash(assumptions),
        "constraints_hash": stable_hash(raw),
    }


def _effective_stage_toolchains(manifest: Any) -> dict[str, str]:
    """Expose the exact toolchain chosen for each executable stage."""
    result = dict(manifest.stage_toolchains)
    if len(manifest.toolchains) == 1:
        for stage in manifest.stage_commands:
            result.setdefault(stage, manifest.toolchains[0].id)
    public: dict[str, str] = {}
    for stage, toolchain_id in sorted(result.items(), key=lambda pair: str(pair[0])):
        safe_stage = public_identifier(stage)
        safe_toolchain = public_identifier(toolchain_id)
        if safe_stage is not None and safe_toolchain is not None:
            public[safe_stage] = safe_toolchain
    return public


def _tool_observation_evidence_error(
    bundle: GraphBundle,
    observation: Any,
    producer: Any,
    artifact_map: Mapping[str, Any],
) -> str | None:
    """Revalidate retained report lineage for one tool-truth observation."""

    cited_ids = {item for item in (
        observation.artifact_id,
        observation.anchor.artifact_id if observation.anchor else None,
    ) if item}
    cited = [artifact_map.get(item) for item in sorted(cited_ids)]
    if not cited or any(item is None for item in cited):
        return "lacks retained report provenance"
    typed_cited = [item for item in cited if item is not None]
    compatibility_error = tool_evidence_compatibility_error(
        observation, producer, typed_cited,
    )
    if compatibility_error is not None:
        return f"violates {TOOL_EVIDENCE_POLICY_VERSION}: {compatibility_error}"
    for artifact in typed_cited:
        if artifact.producer_run_id != observation.run_id:
            return "report producer does not match the observation run"
        if artifact.id not in producer.output_artifact_ids:
            return "report is not declared by the producer run"
        integrity, reason = managed_artifact_integrity(bundle, artifact)
        if not integrity:
            return f"report integrity failed: {reason}"
    return None


def _validate_static_evidence_ref(
    ref: Mapping[str, Any], owner_snapshot_id: str, *,
    feature_stages: set[str], graphs: Mapping[str, Any],
    exported_node_ids: Mapping[str, set[str]],
    observations: Mapping[str, Mapping[str, Any]],
    derivations: Mapping[str, Mapping[str, Mapping[str, Any]]],
    artifacts: Mapping[str, Mapping[str, Any]],
    runs: Mapping[str, Mapping[str, Any]],
    visiting: set[tuple[str, str]] | None = None,
) -> None:
    """Validate that one evidence reference cannot smuggle outcome data."""
    kind = str(ref.get("kind", ""))
    target_id = str(ref.get("target_id", ""))
    snapshot_id = str(ref.get("snapshot_id") or owner_snapshot_id)
    if snapshot_id not in graphs:
        raise ValueError(
            f"feature evidence reference targets undeclared snapshot {snapshot_id!r}"
        )
    if kind == "observation":
        item = observations.get(snapshot_id, {}).get(target_id)
        if item is None:
            raise ValueError(f"feature evidence references missing observation {target_id!r}")
        if (str(item.authority) not in _STATIC_FEATURE_EVIDENCE_AUTHORITIES
                or item.stage not in feature_stages
                or item.workload_id is not None
                or not static_feature_predicate_allowed(item.predicate)):
            raise ValueError(
                f"observation {target_id!r} is not eligible static feature evidence"
            )
        _static_evidence_value(item.value, f"observation[{target_id}].value")
        if item.run_id is not None:
            producer = runs.get(snapshot_id, {}).get(item.run_id)
            if (producer is None
                    or str(producer.status) not in {"succeeded", "cached"}
                    or str(producer.failure_class) != "none"
                    or producer.stage not in feature_stages | {"index"}):
                raise ValueError(
                    f"observation {target_id!r} lacks eligible compiler provenance"
                )
        return
    if kind == "artifact":
        item = artifacts.get(snapshot_id, {}).get(target_id)
        if item is None:
            raise ValueError(f"feature evidence references missing artifact {target_id!r}")
        if not static_feature_predicate_allowed(f"artifact.{item.kind}"):
            raise ValueError(f"artifact {target_id!r} is outcome-shaped evidence")
        if item.producer_run_id is not None:
            producer = runs.get(snapshot_id, {}).get(item.producer_run_id)
            if (producer is None
                    or str(producer.status) not in {"succeeded", "cached"}
                    or str(producer.failure_class) != "none"
                    or producer.stage not in feature_stages | {"index"}):
                raise ValueError(
                    f"artifact {target_id!r} lacks eligible compiler provenance"
                )
        return
    if kind == "entity_anchor":
        entity = graphs[snapshot_id].entities.get(target_id)
        if entity is None:
            raise ValueError(f"feature evidence references missing entity {target_id!r}")
        if target_id not in exported_node_ids.get(snapshot_id, set()):
            raise ValueError(
                f"feature evidence entity {target_id!r} is outside selected feature stages"
            )
        if (str(entity.authority) not in _STATIC_FEATURE_EVIDENCE_AUTHORITIES
                or entity.stage not in feature_stages
                or not static_feature_predicate_allowed(f"entity.{entity.kind}")):
            raise ValueError(
                f"entity {target_id!r} is not eligible static feature evidence"
            )
        _static_evidence_value(entity.attrs, f"entity[{target_id}].attrs")
        return
    if kind == "relation":
        relation = graphs[snapshot_id].relations.get(target_id)
        if relation is None:
            raise ValueError(f"feature evidence references missing relation {target_id!r}")
        exported = exported_node_ids.get(snapshot_id, set())
        if relation.src not in exported or relation.dst not in exported:
            raise ValueError(
                f"feature evidence relation {target_id!r} has an excluded endpoint"
            )
        if (str(relation.authority) not in _STATIC_FEATURE_EVIDENCE_AUTHORITIES
                or relation.stage not in feature_stages
                or not static_feature_predicate_allowed(
                    f"relation.{relation.kind}"
                )):
            raise ValueError(
                f"relation {target_id!r} is not eligible static feature evidence"
            )
        _static_evidence_value(relation.attrs, f"relation[{target_id}].attrs")
        return
    if kind != "derivation":
        raise ValueError(f"unsupported feature evidence reference kind {kind!r}")
    item = derivations.get(snapshot_id, {}).get(target_id)
    if item is None:
        raise ValueError(f"feature evidence references missing derivation {target_id!r}")
    if (str(item.get("authority")) not in _STATIC_FEATURE_EVIDENCE_AUTHORITIES
            or item.get("stage") not in feature_stages
            or not static_feature_predicate_allowed(str(item.get("predicate", "")))):
        raise ValueError(f"derivation {target_id!r} is not eligible static feature evidence")
    _static_evidence_value(item.get("value"), f"derivation[{target_id}].value")
    trail = set(visiting or set())
    key = (snapshot_id, target_id)
    if key in trail:
        raise ValueError(f"feature evidence derivation cycle includes {target_id!r}")
    trail.add(key)
    for child in item.get("evidence_refs", []):
        _validate_static_evidence_ref(
            child, snapshot_id, feature_stages=feature_stages, graphs=graphs,
            exported_node_ids=exported_node_ids, observations=observations,
            derivations=derivations, artifacts=artifacts, runs=runs,
            visiting=trail,
        )


def _feature_evidence_rows(
    dataset: DatasetManifest, *, feature_stages: set[str],
    graphs: Mapping[str, Any], exported_node_ids: Mapping[str, set[str]],
    observations: Mapping[str, Mapping[str, Any]],
    derivations: Mapping[str, Mapping[str, Mapping[str, Any]]],
    artifacts: Mapping[str, Mapping[str, Any]],
    runs: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    selected_predicates = set(dataset.feature_evidence_predicates)
    if not selected_predicates:
        return []
    disallowed = sorted(
        item for item in selected_predicates
        if not static_feature_predicate_allowed(item)
    )
    if disallowed:
        raise ValueError(
            f"outcome-shaped predicates cannot be feature evidence: {disallowed}"
        )
    selected_keys: set[tuple[str, str]] = set()
    included_keys: set[tuple[str, str]] = set()

    def include(snapshot_id: str, derivation_id: str, *, selected: bool) -> None:
        key = (snapshot_id, derivation_id)
        if selected:
            selected_keys.add(key)
        if key in included_keys:
            return
        item = derivations.get(snapshot_id, {}).get(derivation_id)
        if item is None:
            raise ValueError(f"selected feature evidence {derivation_id!r} is unavailable")
        _validate_static_evidence_ref(
            {"kind": "derivation", "target_id": derivation_id,
             "snapshot_id": snapshot_id},
            snapshot_id, feature_stages=feature_stages, graphs=graphs,
            exported_node_ids=exported_node_ids, observations=observations,
            derivations=derivations, artifacts=artifacts, runs=runs,
        )
        included_keys.add(key)
        for ref in item.get("evidence_refs", []):
            if str(ref.get("kind")) == "derivation":
                include(
                    str(ref.get("snapshot_id") or snapshot_id),
                    str(ref.get("target_id")), selected=False,
                )

    for snapshot_id in sorted(graphs):
        for item in derivations.get(snapshot_id, {}).values():
            if item.get("predicate") not in selected_predicates:
                continue
            if item.get("subject_id") not in exported_node_ids.get(snapshot_id, set()):
                continue
            include(snapshot_id, str(item["id"]), selected=True)

    rows = []
    for snapshot_id, derivation_id in sorted(included_keys):
        item = derivations[snapshot_id][derivation_id]
        refs = [dict(value) for value in item.get("evidence_refs", [])]
        completeness = str(item.get("completeness"))
        rows.append({
            "snapshot_id": snapshot_id,
            "subject_id": item.get("subject_id"),
            "predicate": item.get("predicate"),
            "value": _static_evidence_value(item.get("value")),
            "unit": item.get("unit"),
            "stage": item.get("stage"),
            "authority": str(item.get("authority")),
            "completeness": completeness,
            "mask": completeness == "complete" and item.get("value") is not None,
            "derivation_id": derivation_id,
            "algorithm": item.get("algorithm"),
            "algorithm_version": item.get("algorithm_version"),
            "selected_as_feature": (snapshot_id, derivation_id) in selected_keys,
            "evidence_ids": sorted({str(value.get("target_id")) for value in refs}),
            "evidence_refs": refs,
        })
    return rows


def _correspondence_rows(
    bundle: GraphBundle, dataset: DatasetManifest, *,
    feature_stages: set[str], graphs: Mapping[str, Any],
    exported_node_ids: Mapping[str, set[str]],
    observations: Mapping[str, Mapping[str, Any]],
    derivations: Mapping[str, Mapping[str, Mapping[str, Any]]],
    artifacts: Mapping[str, Mapping[str, Any]],
    runs: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    selected_kinds = set(dataset.entity_correspondence_kinds)
    if not selected_kinds:
        return []
    values = []
    for raw_item in bundle.store.correspondences():
        item = json_ready(raw_item)
        if item.get("kind") not in selected_kinds:
            continue
        source_snapshot = str(item.get("source_snapshot_id"))
        target_snapshot = str(item.get("target_snapshot_id"))
        if source_snapshot not in graphs or target_snapshot not in graphs:
            continue
        if (item.get("source_entity_id") not in exported_node_ids[source_snapshot]
                or item.get("target_entity_id") not in exported_node_ids[target_snapshot]):
            continue
        if str(item.get("authority")) not in _STATIC_FEATURE_EVIDENCE_AUTHORITIES:
            raise ValueError(
                f"correspondence {item.get('id')!r} has non-static authority"
            )
        for ref in item.get("evidence_refs", []):
            _validate_static_evidence_ref(
                ref, source_snapshot, feature_stages=feature_stages, graphs=graphs,
                exported_node_ids=exported_node_ids, observations=observations,
                derivations=derivations, artifacts=artifacts, runs=runs,
            )
        values.append(dict(item))
    candidate_groups: dict[tuple[str, str, str, str], list[str]] = {}
    for item in values:
        key = (
            str(item.get("source_snapshot_id")), str(item.get("source_entity_id")),
            str(item.get("target_snapshot_id")), str(item.get("kind")),
        )
        candidate_groups.setdefault(key, []).append(str(item.get("target_entity_id")))
    rows = []
    for item in values:
        source_snapshot = str(item.get("source_snapshot_id"))
        target_snapshot = str(item.get("target_snapshot_id"))
        target = bundle.store.snapshot(target_snapshot)
        is_parent_child = target.parent_snapshot_id == source_snapshot
        refs = [dict(value) for value in item.get("evidence_refs", [])]
        key = (
            source_snapshot, str(item.get("source_entity_id")),
            target_snapshot, str(item.get("kind")),
        )
        candidates = sorted(set(candidate_groups[key]))
        rows.append({
            "correspondence_id": item.get("id"),
            "source_snapshot_id": source_snapshot,
            "source_entity_id": item.get("source_entity_id"),
            "target_snapshot_id": target_snapshot,
            "target_entity_id": item.get("target_entity_id"),
            "parent_snapshot_id": source_snapshot if is_parent_child else None,
            "result_snapshot_id": target_snapshot if is_parent_child else None,
            "kind": item.get("kind"),
            "producer": item.get("producer"),
            "producer_version": item.get("producer_version"),
            "authority": str(item.get("authority")),
            "completeness": str(item.get("completeness")),
            "candidates": candidates,
            "candidate_count": len(candidates),
            "resolution_status": "unique" if len(candidates) == 1 else "ambiguous",
            "resolved_target_entity_id": (
                candidates[0] if len(candidates) == 1 else None
            ),
            "evidence_ids": sorted({str(value.get("target_id")) for value in refs}),
            "evidence_refs": refs,
        })
    return sorted(rows, key=lambda item: str(item.get("correspondence_id", "")))


def export_graph_json(bundle: GraphBundle, snapshot_id: str, output: str | Path) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output, bundle.store.load_graph(snapshot_id).to_dict())
    return output


def export_dataset(bundle: GraphBundle, snapshot_id: str, output_dir: str | Path,
                   dataset: DatasetManifest | None = None, *, format: str = "jsonl",
                   include_source: bool = False) -> dict[str, Any]:
    if include_source:
        raise PermissionError("dataset exports never embed source text; use authorized snippet APIs instead")
    if format not in {"jsonl", "parquet"}:
        raise ValueError("format must be jsonl or parquet")
    if format == "parquet":
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("Parquet export requires hlsgraph[parquet]") from exc
    if dataset is None:
        dataset = DatasetManifest(
            dataset_id=f"dataset.{snapshot_id}",
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            snapshot_ids=[snapshot_id], labels=[],
            splits={snapshot_id: "unassigned"},
        )
    else:
        # DatasetManifest and LabelSpec contain mutable members.  A caller may
        # have changed metadata/labels after construction, so validation must
        # happen at the export trust boundary rather than relying on __post_init__.
        dataset = _validated_dataset_manifest(dataset)
    if snapshot_id not in dataset.snapshot_ids:
        raise ValueError("dataset manifest does not include the requested snapshot")
    if dataset.feature_schema_version != FEATURE_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported feature schema {dataset.feature_schema_version!r}; "
            f"expected {FEATURE_SCHEMA_VERSION!r}"
        )
    if len(set(dataset.snapshot_ids)) != len(dataset.snapshot_ids):
        raise ValueError("dataset snapshot_ids must be unique")
    valid_stages = {item.value for item in Stage}
    if not dataset.feature_stages:
        raise ValueError("dataset feature_stages must not be empty")
    if len(set(dataset.feature_stages)) != len(dataset.feature_stages):
        raise ValueError("dataset feature_stages must be unique")
    invalid_stages = set(dataset.feature_stages) - valid_stages
    if invalid_stages:
        raise ValueError(f"unsupported feature stages: {sorted(invalid_stages)}")
    if len(set(dataset.feature_attribute_allowlist)) != len(
            dataset.feature_attribute_allowlist):
        raise ValueError("dataset feature_attribute_allowlist must be unique")
    if any(not isinstance(item, str) or not item.strip()
           for item in dataset.feature_attribute_allowlist):
        raise ValueError("feature attribute names must be non-empty strings")
    for name, values in (
        ("feature_evidence_predicates", dataset.feature_evidence_predicates),
        ("entity_correspondence_kinds", dataset.entity_correspondence_kinds),
    ):
        if len(set(values)) != len(values):
            raise ValueError(f"dataset {name} must be unique")
        if any(not isinstance(item, str) or not item.strip() for item in values):
            raise ValueError(f"dataset {name} must contain non-empty strings")
    feature_stages = set(dataset.feature_stages)
    feature_attributes = set(dataset.feature_attribute_allowlist)
    declared_snapshots = set(dataset.snapshot_ids)
    for name, mapping in (("splits", dataset.splits),
                          ("kernel_families", dataset.kernel_families),
                          ("dedup_groups", dataset.dedup_groups)):
        extra = set(mapping) - declared_snapshots
        if extra:
            raise ValueError(f"dataset {name} reference undeclared snapshots: {sorted(extra)}")
    graphs = {key: bundle.store.load_graph(key) for key in sorted(declared_snapshots)}
    snapshot_manifests = {
        key: bundle.store.snapshot_manifest(key) for key in sorted(declared_snapshots)
    }
    observations_by_snapshot = {
        key: bundle.store.observations(key) for key in sorted(declared_snapshots)
    }
    derivations_by_snapshot = {
        key: bundle.store.derivations(key) for key in sorted(declared_snapshots)
    }
    artifacts_by_snapshot = {
        key: bundle.store.artifacts(key) for key in sorted(declared_snapshots)
    }
    predictions_by_snapshot = {
        key: bundle.store.predictions(key) for key in sorted(declared_snapshots)
    }
    snapshots_by_id = {
        key: bundle.store.snapshot(key) for key in sorted(declared_snapshots)
    }
    variants_by_id = {
        item["id"]: item
        for key in sorted(declared_snapshots)
        for item in bundle.store.variants(key)
    }
    # A result-only export still retains the recorded action that produced its
    # snapshot.  Missing legacy action rows remain missing; an action is never
    # reconstructed from a diff or guessed from its opaque identifier.
    for snapshot in snapshots_by_id.values():
        if snapshot.action_id and snapshot.action_id not in variants_by_id:
            action = bundle.store.variant(snapshot.action_id)
            if action is not None:
                variants_by_id[action["id"]] = action
    runs_by_snapshot = {
        key: bundle.store.runs(key) for key in sorted(declared_snapshots)
    }
    unattested_tooltruth_run_ids: set[str] = set()
    for key in sorted(declared_snapshots):
        for run in runs_by_snapshot[key]:
            if not run_claims_tool_truth(run):
                continue
            claim_error = real_tool_run_claim_error(run)
            if claim_error is not None:
                raise ValueError(
                    f"dataset snapshot {key} contains an invalid tool-truth run "
                    f"{run.id}: {claim_error}"
                )
            identity_error = tool_run_manifest_identity_error(
                run, snapshot_manifests[key],
            )
            if identity_error is not None:
                raise ValueError(
                    f"dataset snapshot {key} contains a tool-truth run {run.id} whose "
                    f"{identity_error}"
                )
            if not bundle.store.has_valid_execution_commit(key, run.id):
                unattested_tooltruth_run_ids.add(run.id)
        run_ids = {item.id for item in runs_by_snapshot[key]}
        dangling_observations = sorted(
            item.id for item in observations_by_snapshot[key]
            if item.run_id and item.run_id not in run_ids
        )
        dangling_artifacts = sorted(
            item.id for item in artifacts_by_snapshot[key]
            if item.producer_run_id and item.producer_run_id not in run_ids
        )
        if dangling_observations or dangling_artifacts:
            details = []
            if dangling_observations:
                details.append("observations=" + ",".join(dangling_observations))
            if dangling_artifacts:
                details.append("artifacts=" + ",".join(dangling_artifacts))
            raise ValueError(
                f"dataset snapshot {key} has dangling run provenance: " + "; ".join(details)
            )
    observations_by_id = {
        item.id: item for values in observations_by_snapshot.values() for item in values
    }
    artifact_maps = {
        key: {item.id: item for item in values}
        for key, values in artifacts_by_snapshot.items()
    }
    observation_maps = {
        key: {item.id: item for item in values}
        for key, values in observations_by_snapshot.items()
    }
    derivation_maps = {
        key: {str(item["id"]): item for item in values}
        for key, values in derivations_by_snapshot.items()
    }
    run_maps = {
        key: {item.id: item for item in values}
        for key, values in runs_by_snapshot.items()
    }
    nontruth_observation_ids: set[str] = set()
    nontruth_observation_rows: list[dict[str, Any]] = []
    for key in sorted(declared_snapshots):
        for observation in observations_by_snapshot[key]:
            authority = str(observation.authority)
            if authority not in _TOOL_EVIDENCE_AUTHORITIES:
                continue
            if not observation.run_id:
                raise ValueError(
                    f"dataset snapshot {key} contains real-tool authority observation "
                    f"{observation.id} that lacks a producer tool run"
                )
            producer = run_maps[key].get(observation.run_id)
            if producer is None:
                continue
            if not run_claims_tool_truth(producer):
                nontruth_observation_ids.add(observation.id)
                nontruth_observation_rows.append({
                    **json_ready(observation),
                    "claimed_authority": authority,
                    "tool_truth": False,
                    "nontruth_reason": "producer_does_not_claim_tool_truth",
                })
                continue
            evidence_error = _tool_observation_evidence_error(
                bundle, observation, producer, artifact_maps[key],
            )
            if evidence_error is not None:
                raise ValueError(
                    f"dataset snapshot {key} contains invalid tool-truth observation "
                    f"{observation.id}: {evidence_error}"
                )
    if unattested_tooltruth_run_ids:
        raise ValueError(
            "dataset contains tool-truth runs without a valid pipeline-issued "
            "execution attestation and commit receipt: "
            + ", ".join(sorted(unattested_tooltruth_run_ids))
        )
    label_keys = [(item.snapshot_id, item.label_id) for item in dataset.labels]
    if len(set(label_keys)) != len(label_keys):
        raise ValueError("dataset labels must be unique by snapshot_id and label_id")
    for label in dataset.labels:
        if label.snapshot_id not in declared_snapshots:
            raise ValueError(
                f"label {label.label_id} references undeclared snapshot {label.snapshot_id}"
            )
        if not label.mask:
            continue
        if label.observation_id not in observations_by_id:
            raise ValueError(f"label {label.label_id} references an unavailable observation")
        assert label.observation_id is not None
        truth = observations_by_id[label.observation_id]
        if truth.snapshot_id != label.snapshot_id:
            raise ValueError(
                f"label {label.label_id} observation belongs to another snapshot"
            )
        if (truth.predicate, truth.stage, truth.unit) != (
                label.predicate, label.stage, label.unit):
            raise ValueError(
                f"label {label.label_id} predicate/stage/unit do not match its observation"
            )
        if str(truth.completeness) != "complete":
            raise ValueError(f"label {label.label_id} observation is not complete")
        if str(truth.authority) not in {
            "tool_observation", "verification_evidence", "physical_measurement",
        }:
            raise ValueError(
                f"label {label.label_id} is not backed by real tool/verification authority"
            )
        if not truth.run_id:
            raise ValueError(
                f"label {label.label_id} lacks a producer tool run"
            )
        producer = run_maps[label.snapshot_id].get(truth.run_id)
        if producer is None:
            raise ValueError(f"label {label.label_id} producer run is unavailable")
        trust_error = successful_fresh_tool_run_error(producer)
        if trust_error is not None:
            raise ValueError(
                f"label {label.label_id} producer is not a successful fresh real-tool run: "
                f"{trust_error}"
            )
        # A present tool label remains usable only while its exact retained
        # report bytes are independently re-checkable.  The same helper also
        # audits unlabelled tool-truth observations above.
        evidence_error = _tool_observation_evidence_error(
            bundle, truth, producer, artifact_maps[label.snapshot_id],
        )
        if evidence_error is not None:
            raise ValueError(f"label {label.label_id} {evidence_error}")

    effective_splits = {key: dataset.splits.get(key, "unassigned")
                        for key in sorted(dataset.snapshot_ids)}
    for snapshot_key, split in effective_splits.items():
        if split != "unassigned":
            if snapshot_key not in dataset.kernel_families:
                raise ValueError(f"assigned snapshot {snapshot_key} has no kernel family")
            if snapshot_key not in dataset.dedup_groups:
                raise ValueError(f"assigned snapshot {snapshot_key} has no dedup group")
            if snapshot_key not in dataset.licenses:
                raise ValueError(f"assigned snapshot {snapshot_key} has no dataset license")
    for group_name, mapping in (("kernel family", dataset.kernel_families),
                                ("dedup group", dataset.dedup_groups)):
        grouped: dict[str, set[str]] = {}
        for snapshot_key, group in mapping.items():
            split = effective_splits[snapshot_key]
            if split != "unassigned":
                grouped.setdefault(group, set()).add(split)
        conflicts = {group: sorted(splits) for group, splits in grouped.items()
                     if len(splits) > 1}
        if conflicts:
            raise ValueError(f"{group_name} crosses dataset splits: {conflicts}")

    node_rows = []
    edge_rows = []
    for snapshot_key, graph in graphs.items():
        for entity in sorted(graph.entities.values(), key=lambda item: item.id):
            if entity.stage not in feature_stages:
                continue
            # QoR and labels live only in observation/label tables, never in static features.
            node_rows.append({
                "snapshot_id": snapshot_key, "node_id": entity.id, "kind": entity.kind,
                "name": entity.name, "qualified_name": entity.qualified_name,
                "stage": entity.stage, "authority": str(entity.authority),
                "features": _static_features(
                    entity.attrs, feature_attributes,
                    entity_kind=entity.kind, authority=entity.authority,
                ),
                "source_spans": [json_ready(anchor) for anchor in entity.anchors],
                "completeness": str(entity.completeness),
            })
        exported_node_ids = {item["node_id"] for item in node_rows
                             if item["snapshot_id"] == snapshot_key}
        edge_rows.extend({
            "snapshot_id": snapshot_key, "edge_id": item.id, "src": item.src,
            "dst": item.dst, "kind": item.kind, "stage": item.stage,
            "authority": str(item.authority),
            "features": _static_features(item.attrs, feature_attributes),
            "completeness": str(item.completeness),
        } for item in sorted(graph.relations.values(), key=lambda item: item.id)
          if item.stage in feature_stages
          and item.src in exported_node_ids and item.dst in exported_node_ids)
    observation_rows = [json_ready(item) for key in sorted(observations_by_snapshot)
                        for item in sorted(observations_by_snapshot[key], key=lambda item: item.id)
                        if item.id not in nontruth_observation_ids]
    exported_node_ids_by_snapshot = {
        key: {item["node_id"] for item in node_rows if item["snapshot_id"] == key}
        for key in sorted(declared_snapshots)
    }
    feature_evidence_rows = _feature_evidence_rows(
        dataset, feature_stages=feature_stages, graphs=graphs,
        exported_node_ids=exported_node_ids_by_snapshot,
        observations=observation_maps, derivations=derivation_maps,
        artifacts=artifact_maps, runs=run_maps,
    )
    correspondence_rows = _correspondence_rows(
        bundle, dataset, feature_stages=feature_stages, graphs=graphs,
        exported_node_ids=exported_node_ids_by_snapshot,
        observations=observation_maps, derivations=derivation_maps,
        artifacts=artifact_maps, runs=run_maps,
    )
    label_rows = [json_ready(item) for item in sorted(
        dataset.labels, key=lambda item: (item.snapshot_id, item.label_id)
    )]
    split_rows = [{"snapshot_id": key, "split": value,
                   "kernel_family": dataset.kernel_families.get(key),
                   "dedup_group": dataset.dedup_groups.get(key)}
                  for key, value in effective_splits.items()]
    artifact_rows = [{"snapshot_id": key, "artifact_id": item.id, "kind": item.kind,
                      "uri": (None if _looks_like_path(item.uri) else item.uri),
                      "uri_hash": stable_hash(item.uri),
                      "sha256": item.sha256, "size": item.size,
                      "role": item.role, "license": item.license,
                      "producer_run_id": item.producer_run_id,
                      "access": str(item.access), "source_text_embedded": False}
                     for key in sorted(artifacts_by_snapshot)
                     for item in sorted(artifacts_by_snapshot[key], key=lambda item: item.id)]
    prediction_rows = [item for key in sorted(predictions_by_snapshot)
                       for item in predictions_by_snapshot[key]]
    exported_prediction_ids = {item["id"] for item in prediction_rows}
    variant_rows = []
    materialization_rows = []
    for action in sorted(variants_by_id.values(), key=lambda item: item["id"]):
        parent_id = action["parent_snapshot_id"]
        parent_exported = parent_id in declared_snapshots
        prediction_ids = sorted(
            item["id"] for item in predictions_by_snapshot.get(parent_id, [])
            if item.get("action_id") == action["id"]
            and item["id"] in exported_prediction_ids
        )
        materializations = bundle.store.materializations(
            action["id"], parent_snapshot_id=parent_id,
        )
        materialized_result_ids = {
            item.result_snapshot_id for item in materializations
            if str(item.status) == "materialized" and item.result_snapshot_id
        }
        result_snapshots = bundle.store.result_snapshots(
            action["id"], parent_snapshot_id=parent_id,
        )
        if materializations:
            result_snapshots = [
                item for item in result_snapshots
                if item.id in materialized_result_ids
            ]
        result_ids = [
            item.id for item in result_snapshots if item.id in declared_snapshots
        ]
        for attempt_index, attempt in enumerate(materializations, start=1):
            declared_result = (
                attempt.result_snapshot_id
                if (str(attempt.status) == "materialized"
                    and attempt.result_snapshot_id in declared_snapshots)
                else None
            )
            materialization_rows.append({
                "materialization_id": attempt.id,
                "action_id": attempt.action_id,
                "attempt_index": attempt_index,
                "parent_snapshot_id": attempt.parent_snapshot_id,
                "status": str(attempt.status),
                "result_snapshot_id": declared_result,
                "result_snapshot_declared": declared_result is not None,
                "diagnostic_count": len(attempt.diagnostic_ids),
                "evidence_count": len(attempt.evidence_refs),
            })
        public_action = {
            "id": action["id"],
            "parent_snapshot_id": parent_id,
            "prediction_ids": prediction_ids,
            "result_snapshot_ids": result_ids,
            "details_exported": parent_exported,
        }
        if parent_exported:
            scope_id = action.get("scope_id")
            scope_exported = bool(
                scope_id
                and scope_id in exported_node_ids_by_snapshot.get(parent_id, set())
            )
            public_action.update({
                "kind": action["kind"],
                "scope_present": scope_id is not None,
                "scope_exported": scope_exported,
                "delta_sha256": stable_hash(action.get("delta", {})),
            })
            if scope_exported:
                public_action["scope_id"] = scope_id
            elif scope_id is not None:
                public_action["scope_id_sha256"] = stable_hash(scope_id)
        variant_rows.append(public_action)
    snapshot_lineage_rows = [{
        "snapshot_id": key,
        "parent_snapshot_id": snapshots_by_id[key].parent_snapshot_id,
        "action_id": snapshots_by_id[key].action_id,
    } for key in sorted(snapshots_by_id)]
    run_rows = [_public_run(item) for key in sorted(runs_by_snapshot)
                for item in sorted(runs_by_snapshot[key], key=lambda value: value.id)]

    publish_dir = Path(output_dir)
    if publish_dir.exists():
        raise FileExistsError(
            f"dataset output must not already exist: {publish_dir}; "
            "exports are atomically published and never merged"
        )
    publish_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.TemporaryDirectory(
        prefix=f".{publish_dir.name}.tmp-", dir=publish_dir.parent,
    )
    output_dir = Path(temporary.name)

    rows_by_name = {
        "nodes": _write_jsonl(output_dir / "nodes.jsonl", node_rows),
        "edges": _write_jsonl(output_dir / "edges.jsonl", edge_rows),
        "feature_evidence": _write_jsonl(
            output_dir / "feature_evidence.jsonl", feature_evidence_rows,
        ),
        "entity_correspondence": _write_jsonl(
            output_dir / "entity_correspondence.jsonl", correspondence_rows,
        ),
        "observations": _write_jsonl(output_dir / "observations.jsonl", observation_rows),
        "nontruth_observations": _write_jsonl(
            output_dir / "nontruth_observations.jsonl",
            sorted(nontruth_observation_rows, key=lambda item: (
                str(item.get("snapshot_id", "")), str(item.get("id", "")),
            )),
        ),
        "labels": _write_jsonl(output_dir / "labels.jsonl", label_rows),
        "splits": _write_jsonl(output_dir / "splits.jsonl", split_rows),
        "artifacts": _write_jsonl(output_dir / "artifacts.jsonl", artifact_rows),
        "runs": _write_jsonl(output_dir / "runs.jsonl", run_rows),
        # Predictions remain a physically separate table and can never satisfy a
        # label or observation lookup.
        "predictions": _write_jsonl(output_dir / "predictions.jsonl", prediction_rows),
        # Proposed actions and snapshot lineage are separate from both static
        # input features and real-tool truth tables.
        "variants": _write_jsonl(output_dir / "variants.jsonl", variant_rows),
        "action_materializations": _write_jsonl(
            output_dir / "action_materializations.jsonl",
            sorted(materialization_rows, key=lambda item: (
                item["action_id"], item["attempt_index"], item["materialization_id"],
            )),
        ),
        "snapshot_lineage": _write_jsonl(
            output_dir / "snapshot_lineage.jsonl", snapshot_lineage_rows,
        ),
    }
    feature_spec = {
        "schema_version": SCHEMA_VERSION,
        "feature_schema_version": dataset.feature_schema_version,
        "static_features_table": "nodes",
        "input_feature_tables": [
            "nodes", "edges", "feature_evidence", "entity_correspondence",
        ],
        "feature_evidence_table": "feature_evidence",
        "entity_correspondence_table": "entity_correspondence",
        "feature_evidence_predicates": list(dataset.feature_evidence_predicates),
        "entity_correspondence_kinds": list(dataset.entity_correspondence_kinds),
        "feature_evidence_policy": (
            "default empty opt-in; selected root derivations carry selected_as_feature=true; "
            "supporting derivations are provenance only; tool, workload, prediction, label, "
            "and outcome-shaped evidence is rejected"
        ),
        "entity_correspondence_policy": (
            "default empty opt-in; only explicit evidence-backed records whose endpoints "
            "and snapshots are exported are retained; no name-based mapping is inferred"
        ),
        "truth_tables": ["observations", "labels"],
        "nontruth_tables": ["nontruth_observations"],
        "nontruth_observation_policy": (
            "real-tool authority records without a tool-truth producer are removed from "
            "the observations truth table and retained here with an explicit reason"
        ),
        "provenance_tables": ["runs", "artifacts"],
        "provenance_tables_are_input_features": False,
        "prediction_table": "predictions",
        "variant_table": "variants",
        "action_materialization_table": "action_materializations",
        "snapshot_lineage_table": "snapshot_lineage",
        "action_lineage_contract": (
            "variant and prediction links use stored action_id values; explicit materialization "
            "attempts distinguish materialized, no-op, and failed outcomes; only declared "
            "materialized result snapshots and exported prediction rows are linked; legacy "
            "snapshot-only lineage is used only when no materialization records exist"
        ),
        "variant_public_projection": {
            "mode": "minimal_positive_allowlist",
            "always_fields": [
                "id", "parent_snapshot_id", "prediction_ids",
                "result_snapshot_ids", "details_exported",
            ],
            "declared_parent_fields": [
                "kind", "scope_present", "scope_exported", "scope_id",
                "scope_id_sha256", "delta_sha256",
            ],
            "omitted_sensitive_fields": [
                "delta", "rationale", "proposer", "created_at",
            ],
            "undeclared_parent_policy": "opaque_lineage_stub",
            "scope_policy": (
                "raw scope_id is emitted only when it names an exported node; otherwise "
                "only presence and a one-way hash are retained"
            ),
            "action_payload_policy": (
                "delta is represented only by delta_sha256; rationale and proposer are omitted"
            ),
        },
        "action_lineage_tables_are_input_features": False,
        "label_contract": (
            "present labels reference same-snapshot complete observations from successful "
            "fresh real-tool runs and stage-compatible typed retained reports; values are "
            "not duplicated"
        ),
        "tool_evidence_policy_version": TOOL_EVIDENCE_POLICY_VERSION,
        "run_provenance_contract": (
            "observation.run_id and artifact.producer_run_id reference redacted runs; "
            "argv, working directories, messages, and backend details are excluded"
        ),
        "feature_stages": list(dataset.feature_stages),
        "feature_attribute_allowlist": list(dataset.feature_attribute_allowlist),
        "static_feature_schema": _feature_schema_document(feature_attributes),
        "static_feature_policy": (
            "only explicitly declared stages and top-level scalar attributes are exported; "
            "containers require the positive nested schema; unknown containers and keys are "
            "excluded; observations are separate"
        ),
        "private_source_embedded": False,
    }
    _write_json(output_dir / "feature_spec.json", feature_spec)
    if format == "parquet":
        for name, rows in rows_by_name.items():
            if rows:
                pq.write_table(pa.Table.from_pylist(rows), output_dir / f"{name}.parquet")
            else:
                # Preserve an explicit empty table without inventing columns.
                pq.write_table(pa.table({}), output_dir / f"{name}.parquet")

    dataset_payload = json_ready(dataset)
    dataset_payload["splits"] = effective_splits
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset_payload,
        "snapshot_id": snapshot_id,
        "graph_hash": graphs[snapshot_id].graph_hash,
        # Singular fields remain for one-snapshot consumers; the snapshots
        # map is authoritative for multi-snapshot datasets.
        "target_profile": _public_target(
            snapshot_manifests[snapshot_id].target, artifacts_by_snapshot[snapshot_id]
        ),
        "constraints": _public_constraints(
            snapshot_manifests[snapshot_id], artifacts_by_snapshot[snapshot_id]
        ),
        "toolchains": [_public_toolchain(item)
                       for item in snapshot_manifests[snapshot_id].toolchains],
        "stage_toolchains": _effective_stage_toolchains(
            snapshot_manifests[snapshot_id]
        ),
        "snapshots": {
            key: {
                "graph_hash": graphs[key].graph_hash,
                "parent_snapshot_id": snapshots_by_id[key].parent_snapshot_id,
                "action_id": snapshots_by_id[key].action_id,
                "target_profile": _public_target(
                    snapshot_manifests[key].target, artifacts_by_snapshot[key]
                ),
                "constraints": _public_constraints(
                    snapshot_manifests[key], artifacts_by_snapshot[key]
                ),
                "toolchains": [_public_toolchain(item)
                               for item in snapshot_manifests[key].toolchains],
                "stage_toolchains": _effective_stage_toolchains(snapshot_manifests[key]),
                "license": dataset.licenses.get(key),
            } for key in sorted(graphs)
        },
        "artifact_licenses": {
            f"{key}:{item.id}": item.license
            for key in sorted(artifacts_by_snapshot)
            for item in sorted(artifacts_by_snapshot[key], key=lambda x: x.id)
        },
        "format": format,
        "row_counts": {name: len(rows) for name, rows in sorted(rows_by_name.items())},
        "private_source_embedded": False,
    }
    generated_files = sorted(path for path in output_dir.iterdir()
                             if path.is_file() and path.name != "manifest.json")
    manifest["files"] = [path.name for path in generated_files]
    manifest["file_integrity"] = {
        path.name: {
            "sha256": hash_artifact_bytes(path.read_bytes()),
            "size": path.stat().st_size,
        }
        for path in generated_files
    }
    manifest["export_hash"] = stable_hash(manifest)
    _write_json(output_dir / "manifest.json", manifest)
    os.replace(output_dir, publish_dir)
    temporary.cleanup()
    return manifest
