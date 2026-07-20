"""Dialect-aware MLIR text evidence adapter.

This adapter is explicitly experimental.  It preserves native operations, SSA links,
locations, and a few dialect-defined hardware entities; it does not pretend that a
generic MLIR SSA graph is an HLS architecture graph.
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path, PureWindowsPath
from typing import Any

from ..graph import CanonicalGraph
from ..manifest import project_path
from ..model import (
    AuthorityClass,
    Completeness,
    Diagnostic,
    DiagnosticSeverity,
    Entity,
    Relation,
    SourceAnchor,
    Stage,
    safe_relative_path,
)
from .base import ExtractionContext, ExtractionResult


_FUNC = re.compile(r"(?:func\.func|handshake\.func|hls\.func)\s+(?:public\s+|private\s+)?@([\w.$-]+)")
_OP = re.compile(r"^\s*(?:(%[\w.$#-]+)(?:\s*:\s*\d+)?\s*=\s*)?([A-Za-z_][\w-]*(?:\.[A-Za-z_][\w-]*)+)\b(.*)$")
_SSA = re.compile(r"%[\w.$#-]+")
_LOCATION = re.compile(r'loc\("([^"]+)":(\d+):(\d+)\)')
_SLOTS = re.compile(r"(?:numSlots|num_slots|slots)\s*=\s*(\d+)", re.I)
_INTEGER_WIDTH = re.compile(r"(?<![\w!])(?:[su]?i)([1-9]\d*)\b", re.I)
_CONSTANT_LOOP = re.compile(
    r"(?<![\w.$-])(%?[A-Za-z_$][\w.$-]*)\s*=\s*(-?\d+)\s+to\s+(-?\d+)"
    r"(?:\s+step\s+(-?\d+))?\b"
)
_INDEX_LIST = re.compile(r"\[([^\]]*)\]")
_KNOWN_DIALECTS = frozenset({
    "affine", "arith", "builtin", "cf", "func", "handshake", "hls",
    "llvm", "memref", "scf",
})


def _integer_widths(text: str) -> list[int]:
    return [int(value) for value in _INTEGER_WIDTH.findall(text)]


def _memory_access_kind(op_name: str) -> str | None:
    folded = op_name.casefold()
    if folded.endswith((".load", ".read", ".transfer_read")):
        return "load"
    if folded.endswith((".store", ".write", ".transfer_write")):
        return "store"
    if folded.startswith("memref.") and folded.endswith((".alloc", ".alloca")):
        return "allocate"
    if folded == "memref.dealloc":
        return "deallocate"
    if folded in {"memref.copy", "memref.dma_start", "memref.dma_wait"}:
        return "transfer"
    return None


def _index_kinds(op_name: str, tail: str) -> list[str]:
    if _memory_access_kind(op_name) not in {"load", "store"}:
        return []
    values: list[str] = []
    for match in _INDEX_LIST.finditer(tail):
        values.extend(item.strip() for item in match.group(1).split(",") if item.strip())
    return ["constant" if re.fullmatch(r"-?\d+", value) else "dynamic"
            for value in values]


def _positive_trip_count(lower: int, upper: int, step: int) -> int | None:
    if step <= 0 or upper <= lower:
        return None
    count = (upper - lower + step - 1) // step
    return count if count > 0 else None


def _constant_loop_facts(op_name: str, tail: str) -> dict[str, Any]:
    if op_name not in {"affine.for", "scf.for", "hls.loop"}:
        return {}
    match = _CONSTANT_LOOP.search(tail)
    if not match:
        return {}
    _induction, lower_text, upper_text, step_text = match.groups()
    lower, upper = int(lower_text), int(upper_text)
    step = int(step_text) if step_text is not None else 1
    bounds = {
        "lower": lower, "upper": upper, "step": step,
        "comparison": "lt", "upper_inclusive": False,
    }
    result: dict[str, Any] = {"loop_bounds": bounds}
    trip_count = _positive_trip_count(lower, upper, step)
    if trip_count is not None:
        result["trip_count"] = trip_count
    return result


class MlirTextExtractor:
    name = "ir.mlir_text"
    version = "1"

    def supports(self, context: ExtractionContext) -> bool:
        return any(item.uri.lower().endswith(".mlir") for item in context.artifacts.values())

    def extract(self, context: ExtractionContext) -> ExtractionResult:
        graph = CanonicalGraph(snapshot_id=context.snapshot.id,
                               metadata={"mlir_backend": self.name,
                                         "mlir_fidelity": "experimental_text"})
        result = ExtractionResult(graph=graph,
                                  capabilities=["ir.mlir.evidence", "ir.mlir.locations"])
        dialect_counts: dict[str, int] = defaultdict(int)
        operation_count = 0
        complete_feature_artifacts = 0
        max_operations = int(context.options.get("max_ir_operations", 100_000))
        for artifact in sorted(context.artifacts.values(), key=lambda item: item.uri):
            if not artifact.uri.lower().endswith(".mlir"):
                continue
            text = project_path(context.project_root, artifact.uri).read_text(
                encoding="utf-8", errors="replace")
            unit = Entity(kind="ir.mlir.module", name=Path(artifact.uri).name,
                          qualified_name=artifact.uri, snapshot_id=context.snapshot.id,
                          authority=context.authority_for(artifact, AuthorityClass.COMPILER_DECISION), stage=Stage.MLIR.value,
                          attrs={"plane": "evidence", "hot": False,
                                 "parser": "experimental_text"},
                          anchors=[SourceAnchor(artifact_id=artifact.id)])
            graph.add_entity(unit)
            artifact_entity_ids = [unit.id]
            artifact_dialects: set[str] = set()
            artifact_dialect_counts: dict[str, int] = defaultdict(int)
            artifact_truncated = False
            current_parent = unit.id
            function_entities: dict[str, str] = {}
            definitions: dict[str, str] = {}
            projections: dict[str, str] = {}
            pending_uses: list[tuple[str, str, str]] = []
            brace_depth = 0
            parent_stack: list[tuple[int, str]] = [(0, unit.id)]
            for line_number, line in enumerate(text.splitlines(), 1):
                func_match = _FUNC.search(line)
                if func_match:
                    name = func_match.group(1)
                    function_attrs = {
                        "plane": "evidence", "hot": False,
                        "dialect": line.strip().split()[0].split(".")[0],
                    }
                    artifact_dialects.add(str(function_attrs["dialect"]))
                    widths = _integer_widths(line)
                    if widths:
                        function_attrs["bitwidths"] = widths
                    function = Entity(
                        kind="ir.mlir.function", name=name,
                        qualified_name=f"{artifact.uri}::{name}", snapshot_id=context.snapshot.id,
                        authority=context.authority_for(artifact, AuthorityClass.COMPILER_DECISION), stage=Stage.MLIR.value,
                        attrs=function_attrs,
                        anchors=[SourceAnchor(artifact_id=artifact.id, start_line=line_number,
                                              start_column=1, end_line=line_number,
                                              end_column=len(line) + 1)],
                    )
                    graph.add_entity(function)
                    artifact_entity_ids.append(function.id)
                    graph.add_relation(Relation(src=unit.id, dst=function.id, kind="ir.contains",
                                                snapshot_id=context.snapshot.id,
                                                authority=context.authority_for(artifact, AuthorityClass.COMPILER_DECISION),
                                                stage=Stage.MLIR.value))
                    function_entities[name] = function.id
                    current_parent = function.id
                    parent_stack.append((brace_depth + line.count("{") - line.count("}"), function.id))

                op_match = _OP.match(line)
                if op_match and not line.lstrip().startswith(("//", "#", "!")):
                    operation_count += 1
                    if operation_count > max_operations:
                        artifact_truncated = True
                        result.diagnostics.append(Diagnostic(
                            snapshot_id=context.snapshot.id, code="mlir.operation_limit",
                            severity=DiagnosticSeverity.WARNING,
                            message=f"MLIR evidence truncated at {max_operations} operations",
                            stage=Stage.MLIR.value, artifact_id=artifact.id,
                        ))
                        break
                    result_name, op_name, tail = op_match.groups()
                    dialect = op_name.split(".", 1)[0]
                    artifact_dialects.add(dialect)
                    dialect_counts[dialect] += 1
                    artifact_dialect_counts[dialect] += 1
                    location = self._source_location(context, artifact.id, line)
                    bitwidths = _integer_widths(tail)
                    memory_kind = _memory_access_kind(op_name)
                    index_kinds = _index_kinds(op_name, tail)
                    loop_facts = _constant_loop_facts(op_name, tail)
                    op = Entity(
                        kind="ir.mlir.operation", name=op_name,
                        qualified_name=f"{artifact.uri}:{line_number}:{op_name}:{result_name or '-'}",
                        snapshot_id=context.snapshot.id,
                        authority=context.authority_for(artifact, AuthorityClass.COMPILER_DECISION), stage=Stage.MLIR.value,
                        attrs={
                            "plane": "evidence", "hot": False, "dialect": dialect,
                            "operation": op_name, "ssa_result": result_name,
                            "pass_stage": artifact.metadata.get("pass_stage"),
                            **({"bitwidths": bitwidths} if bitwidths else {}),
                            **({"memory_access_kind": memory_kind}
                               if memory_kind else {}),
                            **({"index_kinds": index_kinds} if index_kinds else {}),
                            **loop_facts,
                        },
                        anchors=[SourceAnchor(artifact_id=artifact.id, start_line=line_number,
                                              start_column=1, end_line=line_number,
                                              end_column=len(line) + 1)] + ([location] if location else []),
                    )
                    graph.add_entity(op)
                    artifact_entity_ids.append(op.id)
                    graph.add_relation(Relation(src=current_parent, dst=op.id, kind="ir.contains",
                                                snapshot_id=context.snapshot.id,
                                                authority=context.authority_for(artifact, AuthorityClass.COMPILER_DECISION),
                                                stage=Stage.MLIR.value))
                    if result_name:
                        definitions[result_name] = op.id
                    for operand in _SSA.findall(tail):
                        if operand != result_name:
                            pending_uses.append((operand, op.id, op_name))

                    projected = self._project_hardware_entity(context, artifact, line_number,
                                                               op_name, tail, location,
                                                               op.attrs)
                    if projected:
                        graph.add_entity(projected)
                        artifact_entity_ids.append(projected.id)
                        projections[op.id] = projected.id
                        graph.add_relation(Relation(
                            src=op.id, dst=projected.id, kind="cross.projects_to",
                            snapshot_id=context.snapshot.id,
                            authority=context.authority_for(artifact, AuthorityClass.COMPILER_DECISION), stage=Stage.MLIR.value,
                            attrs={"projection": "dialect_semantics", "parser": "experimental_text"},
                        ))

                    if location:
                        self._cross_map_source(context, graph, op, location, result, artifact)

                brace_depth += line.count("{") - line.count("}")
                while len(parent_stack) > 1 and brace_depth < parent_stack[-1][0]:
                    parent_stack.pop()
                current_parent = parent_stack[-1][1]

            for operand, consumer, consumer_op in pending_uses:
                producer = definitions.get(operand)
                if not producer:
                    continue
                relation_kind = "ir.ssa_use"
                attrs: dict[str, Any] = {"ssa_value": operand, "hardware_topology": False}
                producer_op = graph.entities[producer].attrs.get("operation", "")
                if str(producer_op).startswith("handshake.") and consumer_op.startswith("handshake."):
                    relation_kind = "handshake.dataflow"
                    attrs["hardware_topology"] = True
                graph.add_relation(Relation(
                    src=producer, dst=consumer, kind=relation_kind,
                    snapshot_id=context.snapshot.id,
                    authority=context.authority_for(artifact, AuthorityClass.COMPILER_DECISION), stage=Stage.MLIR.value,
                    attrs=attrs,
                ))
                if relation_kind == "handshake.dataflow" and producer in projections and consumer in projections:
                    source_projection = graph.entities[projections[producer]]
                    target_projection = graph.entities[projections[consumer]]
                    depth = source_projection.attrs.get("depth") or target_projection.attrs.get("depth")
                    graph.add_relation(Relation(
                        src=source_projection.id, dst=target_projection.id, kind="hls.streams_to",
                        snapshot_id=context.snapshot.id,
                        authority=context.authority_for(artifact, AuthorityClass.COMPILER_DECISION), stage=Stage.MLIR.value,
                        attrs={"fifo_depth": depth, "via_ssa": operand,
                               "projection": "handshake_semantics"},
                    ))

            feature_domain_complete = (
                not artifact_truncated
                and artifact_dialects.issubset(_KNOWN_DIALECTS)
            )
            for entity_id in artifact_entity_ids:
                graph.entities[entity_id].attrs[
                    "static_feature_domain_complete"
                ] = feature_domain_complete
            if feature_domain_complete:
                complete_feature_artifacts += 1
            for dialect in sorted(artifact_dialects - _KNOWN_DIALECTS):
                result.diagnostics.append(Diagnostic(
                    snapshot_id=context.snapshot.id, code="mlir.unsupported_dialect",
                    severity=DiagnosticSeverity.WARNING,
                    message=(f"MLIR dialect {dialect!r} has no registered semantic "
                             "adapter; operations remain evidence-only and produce "
                             "no hardware projection"),
                    stage=Stage.MLIR.value, subject_id=unit.id,
                    artifact_id=artifact.id,
                    metadata={"dialect": dialect,
                              "operations": artifact_dialect_counts[dialect],
                              "hardware_projection": False},
                ))

        result.coverage = {"operations": operation_count,
                           "dialects": dict(sorted(dialect_counts.items())),
                           "complete_static_feature_artifacts": complete_feature_artifacts,
                           "fidelity": "experimental_text"}
        if complete_feature_artifacts:
            result.capabilities.append("ir.mlir.complete_static_feature_domain")
        result.diagnostics.append(Diagnostic(
            snapshot_id=context.snapshot.id, code="mlir.experimental_text_parser",
            severity=DiagnosticSeverity.INFO,
            message="MLIR was parsed by the versioned text adapter; native MLIR plugins may provide higher fidelity",
            stage=Stage.MLIR.value,
        ))
        return result

    @staticmethod
    def _source_location(context: ExtractionContext, ir_artifact_id: str,
                         line: str) -> SourceAnchor | None:
        match = _LOCATION.search(line)
        if not match:
            return None
        filename, source_line, column = match.groups()
        candidate = Path(filename)
        absolute = candidate.is_absolute() or PureWindowsPath(filename).is_absolute()
        if absolute:
            try:
                relative = candidate.resolve().relative_to(context.project_root.resolve()).as_posix()
                relative = safe_relative_path(relative, "MLIR source location")
            except ValueError:
                return SourceAnchor(artifact_id=ir_artifact_id,
                                    ir_location=(f'loc("<external>":{source_line}:'
                                                 f'{column})'),
                                    mapping_kind="mlir.filelinecol.redacted",
                                    ambiguity=("location path is outside the project snapshot "
                                               "and was redacted"))
        else:
            try:
                relative = safe_relative_path(filename, "MLIR source location")
            except ValueError:
                return SourceAnchor(artifact_id=ir_artifact_id,
                                    ir_location=(f'loc("<external>":{source_line}:'
                                                 f'{column})'),
                                    mapping_kind="mlir.filelinecol.redacted",
                                    ambiguity=("location path is not a safe project-relative "
                                               "path and was redacted"))
        artifact = context.artifact_for_uri(relative)
        return SourceAnchor(artifact_id=artifact.id if artifact else ir_artifact_id,
                            start_line=int(source_line), start_column=int(column),
                            ir_location=(f'loc("{relative}":{source_line}:{column})'),
                            mapping_kind="mlir.filelinecol",
                            ambiguity=None if artifact else "source artifact is not in snapshot")

    @staticmethod
    def _project_hardware_entity(context: ExtractionContext, artifact: Any, line: int,
                                 op_name: str, tail: str,
                                 source_location: SourceAnchor | None,
                                 operation_attrs: dict[str, Any]) -> Entity | None:
        kind: str | None = None
        attrs: dict[str, Any] = {"source_operation": op_name, "projection_fidelity": "experimental"}
        if op_name in {"scf.for", "affine.for", "hls.loop"}:
            kind = "hls.loop"
            for key in ("loop_bounds", "trip_count"):
                if key in operation_attrs:
                    attrs[key] = operation_attrs[key]
        elif op_name == "handshake.buffer":
            kind = "hls.buffer"
            slots = _SLOTS.search(tail)
            if slots:
                attrs["depth"] = int(slots.group(1))
        elif any(token in op_name for token in ("stream", "fifo", "channel")) and op_name.startswith(("hls.", "handshake.")):
            kind = "hls.stream"
        elif any(token in op_name for token in ("memref", "memory", "alloc")) and op_name.startswith(("hls.", "memref.")):
            kind = "hls.memory"
        elif op_name.startswith("handshake.") and op_name not in {"handshake.func", "handshake.return"}:
            kind = "hls.process"
        if not kind:
            return None
        anchors = [SourceAnchor(artifact_id=artifact.id, start_line=line, start_column=1)]
        if source_location:
            anchors.append(source_location)
        return Entity(kind=kind, name=f"{op_name}@{line}",
                      qualified_name=f"{artifact.id}:{line}:{op_name}",
                      snapshot_id=context.snapshot.id,
                      authority=context.authority_for(artifact, AuthorityClass.COMPILER_DECISION), stage=Stage.MLIR.value,
                      attrs=attrs, anchors=anchors, completeness=Completeness.PARTIAL)

    @staticmethod
    def _cross_map_source(context: ExtractionContext, graph: CanonicalGraph, op: Entity,
                          location: SourceAnchor, result: ExtractionResult, artifact: Any) -> None:
        if location.artifact_id not in context.artifacts:
            return
        existing: CanonicalGraph | None = context.options.get("existing_graph")
        if not existing:
            return
        candidates = [entity for entity in existing.entities.values() for anchor in entity.anchors
                      if anchor.artifact_id == location.artifact_id and anchor.start_line and anchor.end_line
                      and location.start_line and anchor.start_line <= location.start_line <= anchor.end_line]
        candidates = sorted({item.id: item for item in candidates}.values(), key=lambda item: item.id)
        if len(candidates) == 1:
            graph.add_relation(Relation(
                src=op.id, dst=candidates[0].id, kind="cross.maps_to",
                snapshot_id=context.snapshot.id,
                authority=context.authority_for(artifact, AuthorityClass.COMPILER_DECISION), stage=Stage.MLIR.value,
                mapping_kind="mlir.location", attrs={"cardinality": "many_to_many"},
                anchors=[location],
            ), allow_dangling=True)
        elif len(candidates) > 1:
            result.diagnostics.append(Diagnostic(
                snapshot_id=context.snapshot.id, code="mapping.ambiguous_mlir_location",
                severity=DiagnosticSeverity.INFO,
                message=f"MLIR location maps to {len(candidates)} source entities; no single edge was guessed",
                stage=Stage.MLIR.value, subject_id=op.id,
                metadata={"candidate_ids": [item.id for item in candidates]},
            ))
