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
    stable_hash,
)
from .base import ExtractionContext, ExtractionResult


_FUNC = re.compile(
    r"^\s*(?:func\.func|handshake\.func|hls\.func)\s+"
    r"(?:public\s+|private\s+)?@([\w.$-]+)"
)
_OP = re.compile(r"^\s*(?:(%[\w.$#-]+)(?:\s*:\s*\d+)?\s*=\s*)?([A-Za-z_][\w-]*(?:\.[A-Za-z_][\w-]*)+)\b(.*)$")
_QUOTED_OP = re.compile(
    r'^\s*(?:(%[\w.$#-]+)(?:\s*:\s*\d+)?\s*=\s*)?'
    r'"([A-Za-z_][\w-]*(?:\.[A-Za-z_][\w-]*)+)"(.*)$'
)
_SSA = re.compile(r"%[\w.$#-]+")
_LOCATION = re.compile(r'loc\("([^"]+)":(\d+):(\d+)\)')
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
_SOURCE_MAPPING_TARGET_KINDS = frozenset({
    "hls.kernel", "hls.function", "hls.loop", "hls.memory", "hls.port",
    "hls.stream", "source.variable",
})
_CONCRETE_MLIR_MAPPING_LOCATION_KINDS = frozenset({
    "mlir.callsite", "mlir.filelinecol", "mlir.fused", "mlir.name",
    "mlir.opaque",
})
_MLIR_LOCATION_RESOLUTION_CONTRACT = "hlsgraph.mlir_location_resolution.v1"
_SOURCE_ANCHOR_IDENTITY_CONTRACT = "hlsgraph.source_anchor_identity.v1"
_STATIC_FEATURE_DOMAIN_CONTRACT = (
    "hlsgraph.ir.mlir_text.static_feature_domain.v1"
)


def _semantic_line(line: str) -> tuple[str, bool]:
    """Strip an MLIR ``//`` comment and reject unsupported lexical forms."""

    quoted = False
    escaped = False
    position = 0
    while position < len(line):
        character = line[position]
        if quoted:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
            position += 1
            continue
        if character == '"':
            quoted = True
            position += 1
            continue
        pair = line[position:position + 2]
        if pair == "//":
            return line[:position], True
        if pair in {"/*", "*/"}:
            return line, False
        position += 1
    return line, not quoted


def _mask_quoted_strings(text: str) -> str:
    """Preserve offsets while excluding string attributes from IR semantics."""

    masked = list(text)
    quoted = False
    escaped = False
    for position, character in enumerate(text):
        if quoted:
            masked[position] = " "
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
        elif character == '"':
            quoted = True
            masked[position] = " "
    return "".join(masked)


def _outside_quoted_string(text: str, position: int) -> bool:
    """Return whether ``position`` starts outside a quoted string."""

    quoted = False
    escaped = False
    for character in text[:position]:
        if quoted:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
        elif character == '"':
            quoted = True
    return not quoted


def _structural_brace_delta(line: str) -> int:
    """Count braces outside quoted strings after lexical validation."""

    quoted = False
    escaped = False
    delta = 0
    for character in line:
        if quoted:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
            continue
        if character == '"':
            quoted = True
        elif character == "{":
            delta += 1
        elif character == "}":
            delta -= 1
    return delta


def _has_unparsed_inline_region_body(tail: str) -> bool:
    """Reject region-body tokens the one-operation-per-line adapter omits."""

    stack: list[int] = []
    for position, character in enumerate(tail):
        if character == "{":
            stack.append(position)
            continue
        if character != "}":
            continue
        if not stack:
            return True
        start = stack.pop()
        if stack:
            continue
        content = tail[start + 1:position].strip()
        if not content:
            continue
        # A narrow key=value attribute dictionary is covered. Operation-like
        # content is an inline region body and must make the domain incomplete.
        if (
            "%" in content
            or re.search(
                r"(?:^|\s)[A-Za-z_][\w-]*\.[A-Za-z_][\w-]*",
                content,
            )
            or "=" not in content
        ):
            return True
    return any(tail[start + 1:].strip() for start in stack)


def _operation_match(line: str) -> tuple[str | None, str, str] | None:
    match = _QUOTED_OP.match(line) or _OP.match(line)
    if match is None:
        return None
    result_name, op_name, tail = match.groups()
    return result_name, op_name, tail


def _covered_non_operation_line(line: str) -> bool:
    """Return whether a line is known not to contain an MLIR operation.

    This is intentionally narrow.  Multi-line operations, aliases, block
    arguments, and other unmodelled syntax make aggregate coverage incomplete
    rather than being silently treated as an empty operation domain.
    """
    stripped = line.strip()
    if not stripped or stripped.startswith("//"):
        return True
    if re.fullmatch(r"(?:builtin\.)?module(?:\s+attributes\s+\{[^{}]*\})?\s*\{", stripped):
        return True
    return bool(re.fullmatch(r"[{}]+", stripped))


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
    version = "3"

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
            unparsed_construct_lines: list[int] = []
            current_parent = unit.id
            definitions: dict[str, dict[str, str]] = defaultdict(dict)
            brace_depth = 0
            parent_stack: list[tuple[int, str]] = [(0, unit.id)]
            for line_number, raw_line in enumerate(text.splitlines(), 1):
                line, lexical_complete = _semantic_line(raw_line)
                if not lexical_complete:
                    unparsed_construct_lines.append(line_number)
                    break
                if not line.strip() or line.lstrip().startswith("//"):
                    continue
                brace_delta = _structural_brace_delta(line)
                func_match = _FUNC.search(line)
                if func_match:
                    name = func_match.group(1)
                    function_attrs = {
                        "plane": "evidence", "hot": False,
                        "dialect": line.strip().split()[0].split(".")[0],
                    }
                    artifact_dialects.add(str(function_attrs["dialect"]))
                    widths = _integer_widths(_mask_quoted_strings(line))
                    if widths:
                        function_attrs["bitwidths"] = widths
                    function = Entity(
                        kind="ir.mlir.function", name=name,
                        qualified_name=f"{artifact.uri}::{name}", snapshot_id=context.snapshot.id,
                        authority=context.authority_for(artifact, AuthorityClass.COMPILER_DECISION), stage=Stage.MLIR.value,
                        attrs=function_attrs,
                        anchors=[SourceAnchor(artifact_id=artifact.id, start_line=line_number,
                                              start_column=1, end_line=line_number,
                                              end_column=len(raw_line) + 1)],
                    )
                    graph.add_entity(function)
                    artifact_entity_ids.append(function.id)
                    graph.add_relation(Relation(src=unit.id, dst=function.id, kind="ir.contains",
                                                snapshot_id=context.snapshot.id,
                                                authority=context.authority_for(artifact, AuthorityClass.COMPILER_DECISION),
                                                stage=Stage.MLIR.value))
                    if brace_delta > 0:
                        current_parent = function.id
                        parent_stack.append(
                            (brace_depth + brace_delta, function.id)
                        )
                    else:
                        current_parent = parent_stack[-1][1]

                op_match = _operation_match(line)
                if op_match is not None:
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
                    result_name, op_name, tail = op_match
                    semantic_tail = _mask_quoted_strings(tail)
                    if _has_unparsed_inline_region_body(semantic_tail):
                        unparsed_construct_lines.append(line_number)
                    dialect = op_name.split(".", 1)[0]
                    artifact_dialects.add(dialect)
                    dialect_counts[dialect] += 1
                    artifact_dialect_counts[dialect] += 1
                    location = self._source_location(context, artifact.id, line)
                    bitwidths = _integer_widths(semantic_tail)
                    memory_kind = _memory_access_kind(op_name)
                    index_kinds = _index_kinds(op_name, semantic_tail)
                    loop_facts = _constant_loop_facts(
                        op_name, semantic_tail,
                    )
                    ssa_operands = sorted({
                        operand for operand in _SSA.findall(semantic_tail)
                        if operand != result_name
                    })
                    op = Entity(
                        kind="ir.mlir.operation", name=op_name,
                        qualified_name=f"{artifact.uri}:{line_number}:{op_name}:{result_name or '-'}",
                        snapshot_id=context.snapshot.id,
                        authority=context.authority_for(artifact, AuthorityClass.COMPILER_DECISION), stage=Stage.MLIR.value,
                        attrs={
                            "plane": "evidence", "hot": False, "dialect": dialect,
                            "operation": op_name, "ssa_result": result_name,
                            "ssa_operands": ssa_operands,
                            "pass_stage": artifact.metadata.get("pass_stage"),
                            **({"bitwidths": bitwidths} if bitwidths else {}),
                            **({"memory_access_kind": memory_kind}
                               if memory_kind else {}),
                            **({"index_kinds": index_kinds} if index_kinds else {}),
                            **loop_facts,
                        },
                        anchors=[SourceAnchor(artifact_id=artifact.id, start_line=line_number,
                                              start_column=1, end_line=line_number,
                                              end_column=len(raw_line) + 1)] + ([location] if location else []),
                    )
                    graph.add_entity(op)
                    artifact_entity_ids.append(op.id)
                    graph.add_relation(Relation(src=current_parent, dst=op.id, kind="ir.contains",
                                                snapshot_id=context.snapshot.id,
                                                authority=context.authority_for(artifact, AuthorityClass.COMPILER_DECISION),
                                                stage=Stage.MLIR.value))
                    definition_scope = current_parent
                    if result_name:
                        if result_name in definitions[definition_scope]:
                            # Reusing an SSA spelling in one lexical region is
                            # outside this narrow adapter's proven domain.
                            unparsed_construct_lines.append(line_number)
                        else:
                            definitions[definition_scope][result_name] = op.id
                    for operand in ssa_operands:
                        producer = next((
                            definitions[scope_id][operand]
                            for _depth, scope_id in reversed(parent_stack)
                            if operand in definitions.get(scope_id, {})
                        ), None)
                        if producer is None:
                            continue
                        relation_kind = "ir.ssa_use"
                        attrs: dict[str, Any] = {
                            "ssa_value": operand,
                            "hardware_topology": False,
                        }
                        producer_op = graph.entities[producer].attrs.get(
                            "operation", "",
                        )
                        if (
                            str(producer_op).startswith("handshake.")
                            and op_name.startswith("handshake.")
                        ):
                            relation_kind = "handshake.dataflow"
                            attrs.update({
                                "hardware_topology": False,
                                "native_ir_artifact_id": artifact.id,
                                "native_ir_evidence": True,
                                "native_ir_evidence_contract": (
                                    "hlsgraph.mlir.ssa_def_use.v1"
                                ),
                                "native_ir_relation_provenance": (
                                    "mlir.ssa_def_use"
                                ),
                            })
                        graph.add_relation(Relation(
                            src=producer, dst=op.id, kind=relation_kind,
                            snapshot_id=context.snapshot.id,
                            authority=context.authority_for(
                                artifact, AuthorityClass.COMPILER_DECISION,
                            ),
                            stage=Stage.MLIR.value,
                            attrs=attrs,
                            anchors=(
                                [SourceAnchor(artifact_id=artifact.id)]
                                if relation_kind == "handshake.dataflow"
                                else []
                            ),
                        ))

                    if location:
                        self._cross_map_source(context, graph, op, location, result, artifact)
                    if brace_delta > 0 and func_match is None:
                        parent_stack.append(
                            (brace_depth + brace_delta, op.id)
                        )
                elif not func_match and not _covered_non_operation_line(line):
                    unparsed_construct_lines.append(line_number)

                brace_depth += brace_delta
                if brace_depth < 0:
                    unparsed_construct_lines.append(line_number)
                while len(parent_stack) > 1 and brace_depth < parent_stack[-1][0]:
                    parent_stack.pop()
                current_parent = parent_stack[-1][1]
            if brace_depth != 0 or len(parent_stack) != 1:
                unparsed_construct_lines.append(max(1, len(text.splitlines())))
            unparsed_construct_lines = sorted(set(unparsed_construct_lines))
            feature_domain_complete = (
                not artifact_truncated
                and artifact_dialects.issubset(_KNOWN_DIALECTS)
                and not unparsed_construct_lines
            )
            domain_attrs = {
                "static_feature_domain_complete": feature_domain_complete,
                "static_feature_domain_contract": (
                    _STATIC_FEATURE_DOMAIN_CONTRACT
                ),
                "static_feature_parser": self.name,
                "static_feature_parser_version": self.version,
                "static_feature_unparsed_construct_count": (
                    len(unparsed_construct_lines)
                ),
                "static_feature_artifact_id": artifact.id,
                "static_feature_artifact_sha256": artifact.sha256,
            }
            for entity_id in artifact_entity_ids:
                graph.entities[entity_id].attrs.update(domain_attrs)
            if feature_domain_complete:
                complete_feature_artifacts += 1
            elif unparsed_construct_lines:
                result.diagnostics.append(Diagnostic(
                    snapshot_id=context.snapshot.id,
                    code="mlir.static_feature_domain_incomplete",
                    severity=DiagnosticSeverity.WARNING,
                    message=(
                        "MLIR text contains constructs outside the fixed "
                        "static-feature parser; aggregate completeness was withheld"
                    ),
                    stage=Stage.MLIR.value,
                    subject_id=unit.id,
                    artifact_id=artifact.id,
                    metadata={
                        "unparsed_construct_count": len(
                            unparsed_construct_lines
                        ),
                        "sample_lines": unparsed_construct_lines[:16],
                        "parser": self.name,
                        "parser_version": self.version,
                    },
                ))
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
                           "unparsed_static_feature_constructs": sum(
                               int(item.metadata.get(
                                   "unparsed_construct_count", 0,
                               ))
                               for item in result.diagnostics
                               if item.code
                               == "mlir.static_feature_domain_incomplete"
                           ),
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
        match = next((
            candidate for candidate in _LOCATION.finditer(line)
            if _outside_quoted_string(line, candidate.start())
        ), None)
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
    def _cross_map_source(context: ExtractionContext, graph: CanonicalGraph, op: Entity,
                          location: SourceAnchor, result: ExtractionResult, artifact: Any) -> None:
        existing: CanonicalGraph | None = context.options.get("existing_graph")
        concrete_location = (
            location.artifact_id in context.artifacts
            and location.mapping_kind in _CONCRETE_MLIR_MAPPING_LOCATION_KINDS
            and isinstance(location.ir_location, str)
            and location.ir_location.startswith("loc(")
            and location.start_line is not None
            and location.start_column is not None
            and location.ambiguity is None
        )
        candidate_pairs: dict[tuple[str, str], tuple[Entity, SourceAnchor]] = {}
        if existing is not None and concrete_location:
            for entity in existing.entities.values():
                if (entity.kind not in _SOURCE_MAPPING_TARGET_KINDS
                        or entity.stage != Stage.AST.value
                        or entity.completeness != Completeness.COMPLETE):
                    continue
                for target_anchor in entity.anchors:
                    if (target_anchor.artifact_id == location.artifact_id
                            and target_anchor.start_line is not None
                            and target_anchor.end_line is not None
                            and target_anchor.ambiguity is None
                            and target_anchor.start_line <= location.start_line
                            <= target_anchor.end_line):
                        identity = stable_hash(target_anchor)
                        candidate_pairs[(entity.id, identity)] = (
                            entity, target_anchor,
                        )
        candidates = [
            candidate_pairs[key] for key in sorted(candidate_pairs)
        ]
        if len(candidates) == 1:
            target, target_anchor = candidates[0]
            location_identity = stable_hash(location)
            target_anchor_identity = stable_hash(target_anchor)
            graph.add_relation(Relation(
                src=op.id, dst=target.id, kind="cross.maps_to",
                snapshot_id=context.snapshot.id,
                authority=context.authority_for(artifact, AuthorityClass.COMPILER_DECISION), stage=Stage.MLIR.value,
                mapping_kind="mlir.location",
                attrs={
                    "cardinality": "many_to_many",
                    "hardware_topology": False,
                    "mapping_ambiguous": False,
                    "mapping_candidate_count": 1,
                    "mapping_provenance": "mlir.location_anchor",
                    "mapping_redacted": False,
                    "mapping_resolution": "unique_exact",
                    "mapping_resolution_contract": _MLIR_LOCATION_RESOLUTION_CONTRACT,
                    "mapping_unresolved": False,
                    "resolved_target_anchor_identity": target_anchor_identity,
                    "resolved_target_id": target.id,
                    "source_anchor_identity_contract": _SOURCE_ANCHOR_IDENTITY_CONTRACT,
                    "target_layer": "source_ast",
                    "typed_source_anchor_identity": location_identity,
                },
                anchors=[location, target_anchor],
            ), allow_dangling=True)
        elif len(candidates) > 1:
            candidate_ids = sorted({item.id for item, _anchor in candidates})
            result.diagnostics.append(Diagnostic(
                snapshot_id=context.snapshot.id, code="mapping.ambiguous_mlir_location",
                severity=DiagnosticSeverity.INFO,
                message=(f"MLIR location maps to {len(candidates)} source anchors; "
                         "no single edge was guessed"),
                stage=Stage.MLIR.value, subject_id=op.id,
                artifact_id=artifact.id, anchor=location,
                metadata={
                    "candidate_ids": candidate_ids,
                    "candidate_anchor_identities": [
                        stable_hash(anchor) for _entity, anchor in candidates
                    ],
                    "mapping_ambiguous": True,
                    "mapping_candidate_count": len(candidates),
                    "mapping_kind": "mlir.location",
                    "location_kind": location.mapping_kind,
                    "mapping_provenance": "mlir.location_anchor",
                    "mapping_redacted": False,
                    "mapping_resolution": "ambiguous",
                    "mapping_resolution_contract": _MLIR_LOCATION_RESOLUTION_CONTRACT,
                    "mapping_unresolved": False,
                },
            ))
        else:
            redacted = location.mapping_kind == "mlir.filelinecol.redacted"
            result.diagnostics.append(Diagnostic(
                snapshot_id=context.snapshot.id,
                code="mapping.unresolved_mlir_location",
                severity=DiagnosticSeverity.INFO,
                message=("MLIR location did not resolve to one complete, supported "
                         "source AST entity; no mapping edge was created"),
                stage=Stage.MLIR.value, subject_id=op.id,
                artifact_id=artifact.id, anchor=location,
                metadata={
                    "mapping_ambiguous": False,
                    "mapping_candidate_count": 0,
                    "mapping_kind": "mlir.location",
                    "location_kind": location.mapping_kind,
                    "mapping_provenance": "mlir.location_anchor",
                    "mapping_redacted": redacted,
                    "mapping_resolution": "unresolved",
                    "mapping_resolution_contract": _MLIR_LOCATION_RESOLUTION_CONTRACT,
                    "mapping_unresolved": True,
                    "allowed_target_kinds": sorted(_SOURCE_MAPPING_TARGET_KINDS),
                },
            ))
