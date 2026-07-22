"""Deterministic replay of built-in report parsers for typed observations.

An :class:`ObservationSource` is only a content commitment.  It is not trusted
merely because its hashes are self-consistent: the ledger and retriever replay
the fixed built-in parser over the exact managed report bytes and require one
matching parser output before treating the observation as executable evidence.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, MutableMapping

from ..graph import CanonicalGraph
from ..model import (
    ArtifactRef,
    DesignSnapshot,
    Observation,
    ProjectManifest,
    stable_hash,
)
from .base import ExtractionContext
from .vitis import VitisReportExtractor
from .vivado import VivadoReportExtractor


_BUILTIN_PARSERS = {
    (VitisReportExtractor.name, VitisReportExtractor.version): VitisReportExtractor,
    (VivadoReportExtractor.name, VivadoReportExtractor.version): VivadoReportExtractor,
}


def _semantic_identity(item: Observation) -> str:
    """Identity of parser output before SDK run provenance is rebound."""

    return stable_hash({
        "snapshot_id": item.snapshot_id,
        "subject_id": item.subject_id,
        "predicate": item.predicate,
        "value": item.value,
        "unit": item.unit,
        "stage": item.stage,
        "authority": str(item.authority),
        "artifact_id": item.artifact_id,
        "anchor": item.anchor,
        "source": item.source,
        "completeness": str(item.completeness),
        "workload_id": item.workload_id,
        "observed_at": item.observed_at,
        "metadata": item.metadata,
    })


def replay_observation_source_error(
    *,
    project_root: Path,
    manifest: ProjectManifest,
    snapshot: DesignSnapshot,
    graph: CanonicalGraph,
    artifact: ArtifactRef,
    observation: Observation,
    cache: MutableMapping[tuple[str, str, str], tuple[Observation, ...]] | None = None,
) -> str | None:
    """Return why a typed observation is not an exact built-in parser output.

    The cache contains parser outputs only, never an authorization decision, so
    every observation is still compared independently.  Exactly one match is
    required; ambiguous duplicate parser rows fail closed.
    """

    source = observation.source
    if source is None:
        return "observation has no parser source commitment"
    if (observation.artifact_id != artifact.id
            or observation.anchor is None
            or observation.anchor.artifact_id != artifact.id
            or source.artifact_id != artifact.id
            or source.artifact_sha256 != artifact.sha256):
        return "observation source does not name the exact replay artifact"
    source_error = source.validation_error(
        predicate=observation.predicate,
        value=observation.value,
        unit=observation.unit,
    )
    if source_error is not None:
        return source_error
    parser_key = (source.parser_name, source.parser_version)
    parser_type = _BUILTIN_PARSERS.get(parser_key)
    if parser_type is None:
        return "observation source is not issued by a fixed built-in report parser"
    cache_key = (artifact.id, source.parser_name, source.parser_version)
    parsed: tuple[Observation, ...] | None = cache.get(cache_key) if cache is not None else None
    if parsed is None:
        context = ExtractionContext(
            project_root=Path(project_root).resolve(),
            manifest=manifest,
            snapshot=snapshot,
            artifacts={artifact.id: artifact},
            options={"existing_graph": graph},
        )
        result = parser_type().extract(context)
        parse_errors = [
            item for item in result.diagnostics
            if item.severity.value in {"error", "critical"}
        ]
        if parse_errors:
            parsed = ()
        else:
            parsed = tuple(result.observations)
        if cache is not None:
            cache[cache_key] = parsed
    wanted = _semantic_identity(observation)
    matches = [item for item in parsed if _semantic_identity(item) == wanted]
    if len(matches) != 1:
        return (
            "observation is not exactly one deterministic output of the fixed parser"
        )
    return None


__all__ = ["replay_observation_source_error"]
