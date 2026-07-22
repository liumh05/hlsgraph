"""Deterministic, evidence-backed scope feature derivations.

The pass consumes only canonical entities, relations, and extractor diagnostics.
It never guesses cross-layer correspondence from names or source locations.  A
missing value is materialized as ``None`` with an incomplete mask downstream;
an empty histogram is emitted only when a complete evidence plane proves it.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
import re
from typing import Any, Iterable

from ..model import (
    AuthorityClass,
    Completeness,
    Derivation,
    Entity,
    EvidenceKind,
    EvidenceRef,
    Relation,
    Stage,
    canonical_json,
)
from .base import ExtractionResult


ALGORITHM_VERSION = "1"

_SCOPE_KINDS = frozenset({
    "hls.kernel", "hls.function", "hls.loop", "hls.process", "hls.region",
    "ir.mlir.module", "ir.mlir.function",
    "ir.llvm.module", "ir.llvm.function", "ir.llvm.block",
})
_OPERATION_KINDS = frozenset({"ir.mlir.operation", "ir.llvm.operation"})
_CONTAINS_KINDS = frozenset({"hls.contains", "ir.contains"})
_EXPLICIT_MAPPING_KINDS = frozenset({"cross.maps_to", "cross.projects_to"})
_BASE_PREDICATES = (
    "feature.operation_histogram",
    "feature.index_histogram",
    "feature.bitwidth",
    "feature.memory_access",
)
_LOOP_PREDICATES = ("feature.trip_count", "feature.loop_bounds")
_SOFTWARE_CALL_PREDICATE = "feature.software_call_targets"
_STAGE_RANK = {
    Stage.SOURCE.value: 0,
    Stage.AST.value: 1,
    Stage.MLIR.value: 2,
    Stage.HLS_IR.value: 3,
    Stage.LLVM.value: 4,
    Stage.SCHEDULE.value: 5,
    Stage.RTL.value: 6,
}
_EXPLICIT_WIDTH = re.compile(
    r"(?:\b(?:ap_|ac_)?(?:u?int|u?fixed)\s*<\s*([1-9]\d*)|"
    r"\b_?BitInt\s*\(\s*([1-9]\d*)\s*\)|"
    r"\b(?:u?int)([1-9]\d*)_t\b)",
    re.I,
)
_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_LLVM_INDEX_OPCODES = frozenset({
    "getelementptr", "extractelement", "insertelement",
    "extractvalue", "insertvalue",
})
_LLVM_MEMORY_ACCESS_KINDS = {
    "load": "load",
    "store": "store",
    "getelementptr": "address",
    "atomicrmw": "atomic",
    "cmpxchg": "atomic",
    "fence": "ordering",
}


def _entity_ref(entity: Entity) -> EvidenceRef:
    return EvidenceRef(
        kind=EvidenceKind.ENTITY_ANCHOR,
        target_id=entity.id,
        snapshot_id=entity.snapshot_id,
    )


def _relation_ref(relation: Relation) -> EvidenceRef:
    return EvidenceRef(
        kind=EvidenceKind.RELATION,
        target_id=relation.id,
        snapshot_id=relation.snapshot_id,
    )


def _artifact_refs(entities: Iterable[Entity], relations: Iterable[Relation],
                   snapshot_id: str) -> list[EvidenceRef]:
    artifact_ids = {
        anchor.artifact_id
        for item in [*entities, *relations]
        for anchor in item.anchors
    }
    return [EvidenceRef(
        kind=EvidenceKind.ARTIFACT,
        target_id=artifact_id,
        snapshot_id=snapshot_id,
    ) for artifact_id in sorted(artifact_ids)]


def _evidence_refs(entities: Iterable[Entity], relations: Iterable[Relation],
                   snapshot_id: str) -> list[EvidenceRef]:
    entity_values = {item.id: item for item in entities}
    relation_values = {item.id: item for item in relations}
    values = (
        [_entity_ref(entity_values[key]) for key in sorted(entity_values)]
        + [_relation_ref(relation_values[key]) for key in sorted(relation_values)]
        + _artifact_refs(entity_values.values(), relation_values.values(), snapshot_id)
    )
    return sorted({item.id: item for item in values}.values(), key=lambda item: item.id)


def _stage(entities: Iterable[Entity], relations: Iterable[Relation],
           fallback: str) -> str:
    stages = [item.stage for item in [*entities, *relations]]
    return max(stages or [fallback], key=lambda item: (_STAGE_RANK.get(item, -1), item))


def _authority(entities: Iterable[Entity], relations: Iterable[Relation]) -> AuthorityClass:
    if any(item.authority == AuthorityClass.SYNTHETIC
           for item in [*entities, *relations]):
        return AuthorityClass.SYNTHETIC
    return AuthorityClass.DERIVED_FACT


def _path_evidence(
    target_id: str,
    scope_id: str,
    graph_entities: dict[str, Entity],
    parent: dict[str, tuple[str, Relation]],
) -> tuple[list[Entity], list[Relation]]:
    entities: dict[str, Entity] = {}
    relations: dict[str, Relation] = {}
    current = target_id
    if current in graph_entities:
        entities[current] = graph_entities[current]
    while current != scope_id and current in parent:
        owner, relation = parent[current]
        relations[relation.id] = relation
        if owner in graph_entities:
            entities[owner] = graph_entities[owner]
        current = owner
    return list(entities.values()), list(relations.values())


def _closure(
    scope_id: str,
    graph_entities: dict[str, Entity],
    children: dict[str, list[tuple[Relation, str]]],
) -> tuple[set[str], dict[str, tuple[str, Relation]]]:
    seen = {scope_id}
    parent: dict[str, tuple[str, Relation]] = {}
    queue = deque([scope_id])
    while queue:
        current = queue.popleft()
        for relation, child in children.get(current, []):
            if child in seen or child not in graph_entities:
                continue
            seen.add(child)
            parent[child] = (current, relation)
            queue.append(child)
    return seen, parent


def _scope_operations(
    scope_id: str,
    closure_ids: set[str],
    parent: dict[str, tuple[str, Relation]],
    entities: dict[str, Entity],
    mappings: list[Relation],
) -> tuple[list[Entity], list[Entity], list[Relation]]:
    operations: dict[str, Entity] = {}
    evidence_entities: dict[str, Entity] = {scope_id: entities[scope_id]}
    evidence_relations: dict[str, Relation] = {}

    def include_path(target_id: str) -> None:
        path_entities, path_relations = _path_evidence(
            target_id, scope_id, entities, parent,
        )
        evidence_entities.update({item.id: item for item in path_entities})
        evidence_relations.update({item.id: item for item in path_relations})

    for entity_id in sorted(closure_ids):
        entity = entities[entity_id]
        if entity.kind in _OPERATION_KINDS:
            operations[entity.id] = entity
            include_path(entity.id)
    for relation in mappings:
        mapped_operation: Entity | None = None
        scope_endpoint: str | None = None
        source = entities.get(relation.src)
        target = entities.get(relation.dst)
        if source and source.kind in _OPERATION_KINDS and relation.dst in closure_ids:
            mapped_operation, scope_endpoint = source, relation.dst
        elif target and target.kind in _OPERATION_KINDS and relation.src in closure_ids:
            mapped_operation, scope_endpoint = target, relation.src
        if mapped_operation is None or scope_endpoint is None:
            continue
        operations[mapped_operation.id] = mapped_operation
        evidence_entities[mapped_operation.id] = mapped_operation
        evidence_relations[relation.id] = relation
        include_path(scope_endpoint)
    return (
        [operations[key] for key in sorted(operations)],
        [evidence_entities[key] for key in sorted(evidence_entities)],
        [evidence_relations[key] for key in sorted(evidence_relations)],
    )


def _domain_complete(scope: Entity, truncated_artifacts: set[str]) -> bool:
    if any(anchor.artifact_id in truncated_artifacts for anchor in scope.anchors):
        return False
    if scope.kind.startswith("ir.llvm."):
        return scope.completeness == Completeness.COMPLETE
    if scope.stage in {Stage.MLIR.value, Stage.HLS_IR.value}:
        return scope.attrs.get("static_feature_domain_complete") is True
    return False


def _histogram(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _known_or_missing(value: dict[str, int], *, complete_domain: bool,
                      observed_items: bool) -> tuple[dict[str, int] | None, Completeness]:
    if value:
        return value, (Completeness.COMPLETE if complete_domain
                       else Completeness.PARTIAL)
    if complete_domain and not observed_items:
        return {}, Completeness.COMPLETE
    if complete_domain:
        # Evidence objects existed but did not satisfy this feature adapter's
        # explicit contract, so the scope is not certified empty.
        return None, Completeness.PARTIAL
    return None, Completeness.MISSING


def _operation_name(operation: Entity) -> str | None:
    value = operation.attrs.get("opcode", operation.attrs.get("operation"))
    return value if isinstance(value, str) and _SAFE_TOKEN.fullmatch(value) else None


def _index_values(operation: Entity) -> list[str]:
    raw = operation.attrs.get("index_kinds")
    if not isinstance(raw, list):
        return []
    return [item for item in raw if item in {"constant", "dynamic"}]


def _memory_value(operation: Entity) -> str | None:
    value = operation.attrs.get("memory_access_kind")
    return value if isinstance(value, str) and _SAFE_TOKEN.fullmatch(value) else None


def _explicit_widths(entity: Entity) -> list[int]:
    result: list[int] = []
    raw = entity.attrs.get("bitwidths")
    if isinstance(raw, list):
        result.extend(
            item for item in raw
            if isinstance(item, int) and not isinstance(item, bool)
            and 0 < item <= 1_048_576
        )
    for key in ("type", "element_type", "return_type"):
        value = entity.attrs.get(key)
        if not isinstance(value, str):
            continue
        for match in _EXPLICIT_WIDTH.finditer(value):
            width = next((int(item) for item in match.groups() if item), 0)
            if 0 < width <= 1_048_576:
                result.append(width)
    return result


def _append_derivation(
    result: ExtractionResult,
    scope: Entity,
    predicate: str,
    value: Any,
    completeness: Completeness,
    evidence_entities: Iterable[Entity],
    evidence_relations: Iterable[Relation],
    *,
    semantic: str,
    unit: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    entity_values = list({item.id: item for item in evidence_entities}.values())
    relation_values = list({item.id: item for item in evidence_relations}.values())
    result.derivations.append(Derivation(
        snapshot_id=scope.snapshot_id,
        subject_id=scope.id,
        predicate=predicate,
        value=value,
        unit=unit,
        algorithm=f"hlsgraph.static.{predicate.removeprefix('feature.')}",
        algorithm_version=ALGORITHM_VERSION,
        stage=_stage(entity_values, relation_values, scope.stage),
        authority=_authority(entity_values, relation_values),
        completeness=completeness,
        evidence_refs=_evidence_refs(
            entity_values, relation_values, scope.snapshot_id,
        ),
        metadata={
            "semantic": semantic,
            "unknown_is_zero": False,
            **(metadata or {}),
        },
    ))


def _operation_histogram_metadata(
    operations: list[Entity], completeness: Completeness,
) -> dict[str, Any]:
    """Declare an aggregate schema only for one complete, typed IR layer."""
    incomplete = {"operation_histogram_qualification": "unknown_or_mixed"}
    if completeness != Completeness.COMPLETE or not operations:
        return incomplete
    kinds = {item.kind for item in operations}
    if kinds == {"ir.mlir.operation"}:
        for item in operations:
            operation = item.attrs.get("operation")
            dialect = item.attrs.get("dialect")
            if (not isinstance(operation, str) or not _SAFE_TOKEN.fullmatch(operation)
                    or not isinstance(dialect, str) or "." not in operation
                    or operation.split(".", 1)[0].casefold() != dialect.casefold()):
                return incomplete
        schema = "mlir.dialect_qualified_opcode_histogram.v1"
        qualification = "dialect_qualified"
    elif kinds == {"ir.llvm.operation"}:
        if any(
            not isinstance(item.attrs.get("opcode"), str)
            or not _SAFE_TOKEN.fullmatch(str(item.attrs.get("opcode")))
            for item in operations
        ):
            return incomplete
        schema = "llvm.opcode_histogram.v1"
        qualification = "opcode_qualified"
    else:
        return incomplete
    return {
        "operation_histogram_qualification": qualification,
        "operation_histogram_schema": schema,
        "operation_histogram_provenance": "typed_ir_entity_evidence.v1",
        "operation_histogram_domain_complete": True,
    }


def _llvm_feature_histogram_metadata(
    predicate: str,
    value: dict[str, int] | None,
    operations: list[Entity],
    evidence_entities: Iterable[Entity],
    completeness: Completeness,
) -> dict[str, Any]:
    """Certify one LLVM aggregate only after recomputing its typed domain.

    These metadata fields are a versioned adapter contract, not a claim from
    the cited LLVM specification.  Retrieval independently repeats the same
    checks before a knowledge binding may use the aggregate.
    """
    prefix = {
        "feature.index_histogram": "index_histogram",
        "feature.bitwidth": "bitwidth",
        "feature.memory_access": "memory_access",
    }[predicate]
    incomplete = {f"{prefix}_qualification": "unknown_or_mixed"}
    entities = list({item.id: item for item in evidence_entities}.values())
    if (completeness != Completeness.COMPLETE or value is None
            or not operations or not entities
            or any(not item.kind.startswith("ir.llvm.") for item in entities)
            or any(item.kind != "ir.llvm.operation" for item in operations)):
        return incomplete
    opcodes = [item.attrs.get("opcode") for item in operations]
    if any(not isinstance(item, str) or not _SAFE_TOKEN.fullmatch(item)
           for item in opcodes):
        return incomplete

    if predicate == "feature.index_histogram":
        index_values: list[str] = []
        for operation, opcode in zip(operations, opcodes, strict=True):
            raw = operation.attrs.get("index_kinds")
            if raw is None:
                raw = []
            if (not isinstance(raw, list)
                    or any(item not in {"constant", "dynamic"} for item in raw)
                    or (opcode in _LLVM_INDEX_OPCODES) != bool(raw)):
                return incomplete
            index_values.extend(raw)
        if value != _histogram(index_values):
            return incomplete
        return {
            "index_histogram_qualification": "llvm_explicit_operand_kind",
            "index_histogram_schema": (
                "llvm.explicit_index_operand_kind_histogram.v1"
            ),
            "index_histogram_provenance": "typed_ir_entity_evidence.v1",
            "index_operand_definition": (
                "llvm.gep_extract_insert_explicit_operand.v1"
            ),
            "index_histogram_domain_complete": True,
        }

    if predicate == "feature.bitwidth":
        for entity in entities:
            raw = entity.attrs.get("bitwidths")
            if raw is not None and (
                not isinstance(raw, list) or any(
                    not isinstance(item, int) or isinstance(item, bool)
                    or not 0 < item <= 1_048_576 for item in raw
                )
            ):
                return incomplete
            if any(
                entity.attrs.get(key) is not None
                and not isinstance(entity.attrs.get(key), str)
                for key in ("type", "element_type", "return_type")
            ):
                return incomplete
        widths = [
            width for entity in entities for width in _explicit_widths(entity)
        ]
        if value != _histogram(str(width) for width in widths):
            return incomplete
        return {
            "bitwidth_qualification": "llvm_explicit_integer_occurrence",
            "bitwidth_schema": (
                "llvm.explicit_integer_width_occurrence_histogram.v1"
            ),
            "bitwidth_provenance": "typed_ir_entity_evidence.v1",
            "bitwidth_definition": (
                "llvm.explicit_integer_type_occurrence.v1"
            ),
            "bitwidth_domain_complete": True,
        }

    memory_values: list[str] = []
    for operation, opcode in zip(operations, opcodes, strict=True):
        expected = _LLVM_MEMORY_ACCESS_KINDS.get(opcode)
        actual = operation.attrs.get("memory_access_kind")
        if actual != expected:
            return incomplete
        if expected is not None:
            memory_values.append(expected)
    if value != _histogram(memory_values):
        return incomplete
    return {
        "memory_access_qualification": "llvm_opcode_defined_kind",
        "memory_access_schema": "llvm.memory_access_kind_histogram.v1",
        "memory_access_provenance": "typed_ir_entity_evidence.v1",
        "memory_access_opcode_definition": (
            "llvm.load_store_gep_atomic_fence.v1"
        ),
        "memory_access_domain_complete": True,
    }


def _loop_fact(
    scope: Entity,
    key: str,
    operations: list[Entity],
) -> tuple[Any, Completeness, list[Entity]]:
    owners = [item for item in [scope, *operations] if key in item.attrs]
    values = {canonical_json(item.attrs[key]): item.attrs[key] for item in owners}
    if not values:
        return None, Completeness.MISSING, [scope]
    if len(values) != 1:
        return None, Completeness.AMBIGUOUS, owners
    value = next(iter(values.values()))
    if key == "trip_count" and (
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
    ):
        return None, Completeness.MISSING, owners
    if key == "loop_bounds" and not isinstance(value, dict):
        return None, Completeness.MISSING, owners
    def complete_owner(item: Entity) -> bool:
        return (
            item.completeness == Completeness.COMPLETE
            or item.attrs.get("static_feature_domain_complete") is True
        )

    complete = all(complete_owner(item) for item in owners) and complete_owner(scope)
    return value, (Completeness.COMPLETE if complete else Completeness.PARTIAL), owners


def _dependence_values(entities: Iterable[Entity],
                       relations: Iterable[Relation]) -> tuple[list[int], list[Entity], list[Relation]]:
    distances: set[int] = set()
    used_entities: list[Entity] = []
    used_relations: list[Relation] = []

    def consume(item: Entity | Relation) -> bool:
        found = False
        for key in ("dependence_distance", "dependence_distances"):
            raw = item.attrs.get(key)
            values = raw if isinstance(raw, list) else [raw]
            if raw is None or not all(
                isinstance(value, int) and not isinstance(value, bool)
                and abs(value) <= 2_147_483_647 for value in values
            ):
                continue
            distances.update(values)
            found = True
        return found

    for entity in entities:
        if consume(entity):
            used_entities.append(entity)
    for relation in relations:
        if consume(relation):
            used_relations.append(relation)
    return sorted(distances), used_entities, used_relations


def derive_static_features(result: ExtractionResult) -> None:
    """Append one stable static-feature derivation per supported scope/predicate."""
    graph = result.graph
    initial_derivation_count = len(result.derivations)
    existing = {
        (item.subject_id, item.predicate)
        for item in result.derivations
        if item.predicate.startswith("feature.")
    }
    children: dict[str, list[tuple[Relation, str]]] = defaultdict(list)
    for relation in sorted(graph.relations.values(), key=lambda item: item.id):
        if relation.kind in _CONTAINS_KINDS:
            children[relation.src].append((relation, relation.dst))
    mappings = sorted(
        (item for item in graph.relations.values()
         if item.kind in _EXPLICIT_MAPPING_KINDS),
        key=lambda item: item.id,
    )
    truncated_artifacts = {
        item.artifact_id for item in result.diagnostics
        if item.code in {"llvm.instruction_limit", "mlir.operation_limit"}
        and item.artifact_id is not None
    }
    incomplete_call_scopes = {
        item.subject_id for item in result.diagnostics
        if item.code in {"mapping.ambiguous_call", "mapping.unresolved_call"}
        and item.subject_id is not None
    }
    software_calls = sorted(
        (item for item in graph.relations.values()
         if item.kind == "software.calls"),
        key=lambda item: item.id,
    )

    for scope in sorted(
        (item for item in graph.entities.values() if item.kind in _SCOPE_KINDS),
        key=lambda item: item.id,
    ):
        closure_ids, parent = _closure(scope.id, graph.entities, children)
        operations, operation_entities, operation_relations = _scope_operations(
            scope.id, closure_ids, parent, graph.entities, mappings,
        )
        complete_domain = _domain_complete(scope, truncated_artifacts)

        operation_names = [
            value for item in operations
            if (value := _operation_name(item)) is not None
        ]
        operation_histogram = _histogram(operation_names)
        operation_value, operation_completeness = _known_or_missing(
            operation_histogram,
            complete_domain=(complete_domain
                             and len(operation_names) == len(operations)),
            observed_items=bool(operations),
        )
        if (scope.id, _BASE_PREDICATES[0]) not in existing:
            _append_derivation(
                result, scope, _BASE_PREDICATES[0], operation_value,
                operation_completeness, operation_entities, operation_relations,
                semantic="histogram_of_explicit_ir_operation_entities",
                metadata=_operation_histogram_metadata(
                    operations, operation_completeness,
                ),
            )

        index_histogram = _histogram(
            value for item in operations for value in _index_values(item)
        )
        index_value, index_completeness = _known_or_missing(
            index_histogram, complete_domain=complete_domain,
            observed_items=False,
        )
        if (scope.id, _BASE_PREDICATES[1]) not in existing:
            _append_derivation(
                result, scope, _BASE_PREDICATES[1], index_value,
                index_completeness, operation_entities, operation_relations,
                semantic="histogram_of_explicit_constant_and_dynamic_index_operands",
                metadata=_llvm_feature_histogram_metadata(
                    _BASE_PREDICATES[1], index_value, operations,
                    operation_entities, index_completeness,
                ),
            )

        bitwidth_entities: dict[str, Entity] = {
            item.id: item for item in operation_entities
        }
        bitwidth_relations: dict[str, Relation] = {
            item.id: item for item in operation_relations
        }
        width_values: list[int] = []
        for entity_id in sorted(closure_ids):
            entity = graph.entities[entity_id]
            widths = _explicit_widths(entity)
            if not widths:
                continue
            width_values.extend(widths)
            path_entities, path_relations = _path_evidence(
                entity_id, scope.id, graph.entities, parent,
            )
            bitwidth_entities.update({item.id: item for item in path_entities})
            bitwidth_relations.update({item.id: item for item in path_relations})
        bitwidth_histogram = _histogram(str(value) for value in width_values)
        bitwidth_value, bitwidth_completeness = _known_or_missing(
            bitwidth_histogram, complete_domain=complete_domain,
            observed_items=False,
        )
        if (scope.id, _BASE_PREDICATES[2]) not in existing:
            _append_derivation(
                result, scope, _BASE_PREDICATES[2], bitwidth_value,
                bitwidth_completeness, bitwidth_entities.values(),
                bitwidth_relations.values(),
                semantic="histogram_of_explicit_integer_type_width_occurrences",
                metadata=_llvm_feature_histogram_metadata(
                    _BASE_PREDICATES[2], bitwidth_value, operations,
                    bitwidth_entities.values(), bitwidth_completeness,
                ),
            )

        memory_histogram = _histogram(
            value for item in operations
            if (value := _memory_value(item)) is not None
        )
        memory_value, memory_completeness = _known_or_missing(
            memory_histogram, complete_domain=complete_domain,
            observed_items=False,
        )
        if (scope.id, _BASE_PREDICATES[3]) not in existing:
            _append_derivation(
                result, scope, _BASE_PREDICATES[3], memory_value,
                memory_completeness, operation_entities, operation_relations,
                semantic="histogram_of_explicit_ir_memory_access_kinds",
                metadata=_llvm_feature_histogram_metadata(
                    _BASE_PREDICATES[3], memory_value, operations,
                    operation_entities, memory_completeness,
                ),
            )

        if (scope.kind in {"hls.kernel", "hls.function"}
                and (scope.id, _SOFTWARE_CALL_PREDICATE) not in existing):
            call_relations = [
                relation for relation in software_calls
                if relation.src == scope.id and relation.dst in graph.entities
            ]
            targets = sorted({relation.dst for relation in call_relations})
            call_domain_complete = (
                scope.stage == Stage.AST.value
                and scope.completeness == Completeness.COMPLETE
                and scope.id not in incomplete_call_scopes
            )
            call_value: list[str] | None = (
                targets if targets else ([] if call_domain_complete else None)
            )
            call_completeness = (
                Completeness.COMPLETE if call_domain_complete
                else (Completeness.PARTIAL if targets else Completeness.MISSING)
            )
            call_entities = [
                scope, *(graph.entities[target] for target in targets)
            ]
            _append_derivation(
                result, scope, _SOFTWARE_CALL_PREDICATE, call_value,
                call_completeness, call_entities, call_relations,
                semantic=("deduplicated_sorted_project_local_ast_call_target_entity_ids;"
                          "not_hardware_topology"),
                unit="entity_ids",
            )

        if scope.kind == "hls.loop":
            for predicate, key in zip(
                _LOOP_PREDICATES, ("trip_count", "loop_bounds"), strict=True,
            ):
                if (scope.id, predicate) in existing:
                    continue
                value, completeness, owners = _loop_fact(scope, key, operations)
                loop_evidence_entities = {scope.id: scope}
                loop_evidence_relations: dict[str, Relation] = {}
                for owner in owners:
                    loop_evidence_entities[owner.id] = owner
                    if owner.id in closure_ids:
                        path_entities, path_relations = _path_evidence(
                            owner.id, scope.id, graph.entities, parent,
                        )
                        loop_evidence_entities.update(
                            {item.id: item for item in path_entities}
                        )
                        loop_evidence_relations.update(
                            {item.id: item for item in path_relations}
                        )
                    else:
                        for relation in mappings:
                            if {relation.src, relation.dst} == {scope.id, owner.id}:
                                loop_evidence_relations[relation.id] = relation
                _append_derivation(
                    result, scope, predicate, value, completeness,
                    loop_evidence_entities.values(), loop_evidence_relations.values(),
                    semantic=("positive_exact_iteration_count" if key == "trip_count"
                              else "exact_constant_loop_bound_fields"),
                )

        dependence_entities = {
            entity_id: graph.entities[entity_id] for entity_id in closure_ids
        }
        dependence_entities.update({
            item.id: item for item in operation_entities
        })
        dependence_relations = {
            relation.id: relation for relation in operation_relations
        }
        for entity_id in closure_ids:
            if entity_id == scope.id:
                continue
            _path_entities, path_relations = _path_evidence(
                entity_id, scope.id, graph.entities, parent,
            )
            dependence_relations.update({item.id: item for item in path_relations})
        distances, used_entities, used_relations = _dependence_values(
            dependence_entities.values(), dependence_relations.values(),
        )
        dependence_predicate = "feature.dependence_distance"
        if (scope.id, dependence_predicate) not in existing:
            if not distances:
                _append_derivation(
                    result, scope, dependence_predicate, None,
                    Completeness.MISSING, [scope], [],
                    semantic="no_explicit_dependence_distance_evidence",
                )
                continue
            dependence_completeness = (
                Completeness.COMPLETE
                if (scope.completeness == Completeness.COMPLETE
                    or scope.attrs.get("static_feature_domain_complete") is True)
                and all(
                    item.completeness == Completeness.COMPLETE
                    or item.attrs.get("static_feature_domain_complete") is True
                    for item in used_entities
                )
                and all(item.completeness == Completeness.COMPLETE
                        for item in used_relations)
                else Completeness.PARTIAL
            )
            _append_derivation(
                result, scope, dependence_predicate,
                {"distances": distances}, dependence_completeness,
                [scope, *used_entities], used_relations,
                semantic="explicitly_recorded_dependence_distances_only",
            )

    if len(result.derivations) > initial_derivation_count:
        result.capabilities.append("feature.static_derivations")
