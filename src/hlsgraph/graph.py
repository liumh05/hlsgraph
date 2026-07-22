"""Canonical, evidence-backed HLS architecture graph."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable

from .model import (
    AuthorityClass, Entity, Relation, canonical_json, json_ready,
    reject_embedded_body_fields, stable_hash,
)
from .version import SCHEMA_VERSION, SUPPORTED_GRAPH_SCHEMA_VERSIONS


_NON_FACT_AUTHORITIES = frozenset({
    AuthorityClass.KNOWLEDGE_RULE,
    AuthorityClass.PREDICTION_HYPOTHESIS,
})


def _require_fact_authority(authority: AuthorityClass, subject: str) -> None:
    if str(authority) in {item.value for item in _NON_FACT_AUTHORITIES}:
        raise ValueError(
            f"{subject} cannot use {str(authority)!r} authority; "
            "knowledge and predictions must remain in their dedicated envelopes"
        )


@dataclass(slots=True)
class CanonicalGraph:
    snapshot_id: str
    entities: dict[str, Entity] = field(default_factory=dict)
    relations: dict[str, Relation] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version not in SUPPORTED_GRAPH_SCHEMA_VERSIONS:
            raise ValueError(
                f"canonical graph schema {self.schema_version!r} is not supported by "
                f"this build ({sorted(SUPPORTED_GRAPH_SCHEMA_VERSIONS)!r})"
            )
        if not self.snapshot_id:
            raise ValueError("canonical graph snapshot_id is required")
        reject_embedded_body_fields(self.metadata, "canonical graph metadata")

    def add_entity(self, entity: Entity) -> Entity:
        if entity.snapshot_id != self.snapshot_id:
            raise ValueError("entity belongs to a different snapshot")
        _require_fact_authority(entity.authority, "canonical entity")
        previous = self.entities.get(entity.id)
        if previous is not None and canonical_json(previous) != canonical_json(entity):
            raise ValueError(f"conflicting entity with stable id {entity.id}")
        self.entities[entity.id] = entity
        return entity

    def add_relation(self, relation: Relation, *, allow_dangling: bool = False) -> Relation:
        if relation.snapshot_id != self.snapshot_id:
            raise ValueError("relation belongs to a different snapshot")
        _require_fact_authority(relation.authority, "canonical relation")
        if not allow_dangling and (relation.src not in self.entities or relation.dst not in self.entities):
            raise ValueError(f"relation endpoints must exist: {relation.src} -> {relation.dst}")
        previous = self.relations.get(relation.id)
        if previous is not None and canonical_json(previous) != canonical_json(relation):
            raise ValueError(f"conflicting relation with stable id {relation.id}")
        self.relations[relation.id] = relation
        return relation

    def by_kind(self, kind: str) -> list[Entity]:
        return sorted((entity for entity in self.entities.values() if entity.kind == kind),
                      key=lambda entity: entity.id)

    def neighbors(self, entity_id: str, *, direction: str = "both",
                  relation_kinds: Iterable[str] | None = None) -> list[tuple[Relation, Entity]]:
        if direction not in {"in", "out", "both"}:
            raise ValueError("direction must be in, out, or both")
        allowed = set(relation_kinds or [])
        result: list[tuple[Relation, Entity]] = []
        for relation in sorted(self.relations.values(), key=lambda item: item.id):
            if allowed and relation.kind not in allowed:
                continue
            if direction in {"out", "both"} and relation.src == entity_id and relation.dst in self.entities:
                result.append((relation, self.entities[relation.dst]))
            if direction in {"in", "both"} and relation.dst == entity_id and relation.src in self.entities:
                result.append((relation, self.entities[relation.src]))
        return result

    def traverse(self, start_id: str, *, depth: int = 1, direction: str = "both",
                 relation_kinds: Iterable[str] | None = None) -> tuple[list[Entity], list[Relation]]:
        if start_id not in self.entities:
            raise KeyError(start_id)
        if depth < 0 or depth > 32:
            raise ValueError("depth must be in 0..32")
        seen = {start_id}
        used_relations: dict[str, Relation] = {}
        queue: deque[tuple[str, int]] = deque([(start_id, 0)])
        while queue:
            current, level = queue.popleft()
            if level >= depth:
                continue
            for relation, entity in self.neighbors(current, direction=direction,
                                                     relation_kinds=relation_kinds):
                used_relations[relation.id] = relation
                if entity.id not in seen:
                    seen.add(entity.id)
                    queue.append((entity.id, level + 1))
        return ([self.entities[item] for item in sorted(seen)],
                [used_relations[item] for item in sorted(used_relations)])

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "snapshot_id": self.snapshot_id,
            "metadata": json_ready(self.metadata),
            "entities": [json_ready(self.entities[key]) for key in sorted(self.entities)],
            "relations": [json_ready(self.relations[key]) for key in sorted(self.relations)],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CanonicalGraph":
        if "schema_version" not in value:
            raise ValueError("canonical graph schema_version is required")
        graph = cls(snapshot_id=value["snapshot_id"], metadata=dict(value.get("metadata", {})),
                    schema_version=value["schema_version"])
        for item in value.get("entities", []):
            graph.add_entity(Entity.from_dict(item))
        for item in value.get("relations", []):
            graph.add_relation(Relation.from_dict(item), allow_dangling=False)
        return graph

    @property
    def graph_hash(self) -> str:
        return stable_hash(self.to_dict())

    def stats(self) -> dict[str, Any]:
        entity_kinds: dict[str, int] = {}
        relation_kinds: dict[str, int] = {}
        for entity in self.entities.values():
            entity_kinds[entity.kind] = entity_kinds.get(entity.kind, 0) + 1
        for relation in self.relations.values():
            relation_kinds[relation.kind] = relation_kinds.get(relation.kind, 0) + 1
        return {
            "entities": len(self.entities),
            "relations": len(self.relations),
            "entity_kinds": dict(sorted(entity_kinds.items())),
            "relation_kinds": dict(sorted(relation_kinds.items())),
            "graph_hash": self.graph_hash,
        }
