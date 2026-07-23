"""LLVM IR evidence extraction; CFG is never promoted to HLS architecture topology."""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from ..graph import CanonicalGraph
from ..manifest import project_path
from ..model import (
    AuthorityClass,
    Diagnostic,
    DiagnosticSeverity,
    Entity,
    Relation,
    SourceAnchor,
    Stage,
)
from .base import ExtractionContext, ExtractionResult


_DEFINE = re.compile(r"^\s*define\b.*?@([\w.$-]+)\s*\(")
_LABEL = re.compile(r"^([\w.$-]+):")
_SSA_NAME = r'%(?:"(?:[^"\\]|\\[0-9A-Fa-f]{2})+"|[\w.$-]+)'
_INSTRUCTION = re.compile(
    rf"^\s*(?:({_SSA_NAME})\s*=\s*)?([a-z][a-z0-9_.]*)\b(.*)$"
)
_CALL = re.compile(r"@([\w.$-]+)\s*\(")
_BRANCH = re.compile(r"label\s+%([\w.$-]+)")
_DBG = re.compile(r"!dbg\s+!(\d+)")
_DILOC = re.compile(r"(?m)^\s*!(\d+)\s*=\s*!DILocation\(line:\s*(\d+),\s*column:\s*(\d+),\s*scope:\s*!(\d+)")
_DIFILE = re.compile(r'(?m)^\s*!(\d+)\s*=\s*!DIFile\(filename:\s*"([^"]+)",\s*directory:\s*"([^"]*)"')
_INTEGER_WIDTH = re.compile(r"(?<![\w!])i([1-9]\d*)\b")
_TYPED_INDEX = re.compile(r"\bi[1-9]\d*\s+(-?\d+|%[\w.$-]+)\b")
_LLVM_INSTRUCTION_OPCODES = frozenset({
    "ret", "br", "switch", "indirectbr", "invoke", "callbr", "resume",
    "catchswitch", "catchret", "cleanupret", "unreachable",
    "fneg",
    "add", "fadd", "sub", "fsub", "mul", "fmul", "udiv", "sdiv",
    "fdiv", "urem", "srem", "frem",
    "shl", "lshr", "ashr", "and", "or", "xor",
    "extractelement", "insertelement", "shufflevector",
    "extractvalue", "insertvalue",
    "alloca", "load", "store", "fence", "cmpxchg", "atomicrmw",
    "getelementptr",
    "trunc", "zext", "sext", "fptrunc", "fpext", "fptoui", "fptosi",
    "uitofp", "sitofp", "ptrtoint", "inttoptr", "bitcast",
    "addrspacecast",
    "icmp", "fcmp", "phi", "select", "freeze", "call", "va_arg",
    "landingpad", "catchpad", "cleanuppad",
})
_STATIC_FEATURE_DOMAIN_CONTRACT = (
    "hlsgraph.ir.llvm_text.static_feature_domain.v1"
)


def _semantic_line(line: str) -> tuple[str, bool]:
    """Remove an LLVM ``;`` comment without treating quoted bytes as syntax."""

    quoted = False
    escaped = False
    for position, character in enumerate(line):
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
        elif character == ";":
            return line[:position], True
    return line, not quoted


def _mask_quoted_strings(text: str) -> str:
    """Preserve offsets while excluding quoted payload bytes from semantics."""

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


def _covered_module_line(line: str) -> bool:
    """Recognize module syntax that cannot add executable IR operations.

    Declarations, globals, type aliases, and other type-bearing constructs are
    deliberately excluded because ignoring them would make bitwidth aggregates
    appear complete.  Unknown syntax withholds aggregate completeness.
    """
    stripped = line.strip()
    return bool(
        not stripped
        or stripped.startswith(";")
        or stripped.startswith("source_filename =")
        or stripped.startswith("target datalayout =")
        or stripped.startswith("target triple =")
        or stripped.startswith("module asm ")
        or stripped.startswith("attributes #")
        or stripped.startswith("!")
        or stripped.startswith("uselistorder")
    )


def _integer_widths(text: str) -> list[int]:
    return [int(value) for value in _INTEGER_WIDTH.findall(text)]


def _index_kinds(opcode: str, tail: str) -> list[str]:
    values: list[str] = []
    if opcode == "getelementptr":
        values = _TYPED_INDEX.findall(tail)
    elif opcode in {"extractelement", "insertelement"}:
        typed = _TYPED_INDEX.findall(tail)
        values = typed[-1:] if typed else []
    elif opcode in {"extractvalue", "insertvalue"}:
        values = re.findall(r",\s*(-?\d+)\b", tail)
    return ["constant" if re.fullmatch(r"-?\d+", value) else "dynamic"
            for value in values]


def _memory_access_kind(opcode: str) -> str | None:
    return {
        "load": "load",
        "store": "store",
        "getelementptr": "address",
        "atomicrmw": "atomic",
        "cmpxchg": "atomic",
        "fence": "ordering",
    }.get(opcode)


class LlvmIrExtractor:
    name = "ir.llvm_text"
    version = "2"

    def supports(self, context: ExtractionContext) -> bool:
        return any(Path(item.uri).suffix.lower() in {".ll", ".bc"}
                   for item in context.artifacts.values())

    def extract(self, context: ExtractionContext) -> ExtractionResult:
        graph = CanonicalGraph(snapshot_id=context.snapshot.id,
                               metadata={"llvm_cfg_is_hls_topology": False})
        result = ExtractionResult(graph=graph,
                                  capabilities=["ir.llvm.cfg_evidence", "ir.llvm.memory_evidence"])
        instruction_count = 0
        complete_feature_artifacts = 0
        total_unparsed_constructs = 0
        max_instructions = int(context.options.get("max_ir_operations", 100_000))
        for artifact in sorted(context.artifacts.values(), key=lambda item: item.uri):
            suffix = Path(artifact.uri).suffix.lower()
            if suffix == ".bc":
                result.diagnostics.append(Diagnostic(
                    snapshot_id=context.snapshot.id, code="llvm.bitcode_requires_plugin",
                    severity=DiagnosticSeverity.WARNING,
                    message=f"binary LLVM bitcode requires a native disassembler plugin: {artifact.uri}",
                    stage=Stage.LLVM.value, artifact_id=artifact.id,
                ))
                continue
            if suffix != ".ll":
                continue
            text = project_path(context.project_root, artifact.uri).read_text(
                encoding="utf-8", errors="replace")
            source_lines = text.splitlines()
            semantic_lines = [_semantic_line(line) for line in source_lines]
            semantic_text = "\n".join(
                line if complete else ""
                for line, complete in semantic_lines
            )
            debug_locations = {match.group(1): (int(match.group(2)), int(match.group(3)), match.group(4))
                               for match in _DILOC.finditer(semantic_text)}
            debug_files = {match.group(1): (match.group(2), match.group(3))
                           for match in _DIFILE.finditer(semantic_text)}
            unit = Entity(kind="ir.llvm.module", name=Path(artifact.uri).name,
                          qualified_name=artifact.uri, snapshot_id=context.snapshot.id,
                          authority=context.authority_for(artifact, AuthorityClass.COMPILER_DECISION), stage=Stage.LLVM.value,
                          attrs={"plane": "evidence", "hot": False,
                                 "cfg_is_hls_topology": False},
                          anchors=[SourceAnchor(artifact_id=artifact.id)])
            graph.add_entity(unit)
            artifact_entity_ids = [unit.id]
            artifact_truncated = False
            unparsed_construct_lines: list[int] = []
            current_function: Entity | None = None
            current_block: Entity | None = None
            blocks: dict[tuple[str, str], Entity] = {}
            pending_branches: list[tuple[str, str, str]] = []
            functions: dict[str, Entity] = {}
            pending_calls: list[tuple[str, str]] = []
            for line_number, (raw_line, (line, lexical_complete)) in enumerate(
                zip(source_lines, semantic_lines, strict=True), 1,
            ):
                if not lexical_complete:
                    unparsed_construct_lines.append(line_number)
                    break
                define = _DEFINE.match(line)
                if define:
                    if current_function is not None:
                        unparsed_construct_lines.append(line_number)
                        break
                    name = define.group(1)
                    if re.search(
                        r"\{\s*$", _mask_quoted_strings(line),
                    ) is None:
                        unparsed_construct_lines.append(line_number)
                    function_attrs = {"plane": "evidence", "hot": False}
                    widths = _integer_widths(_mask_quoted_strings(line))
                    if widths:
                        function_attrs["bitwidths"] = widths
                    current_function = Entity(
                        kind="ir.llvm.function", name=name,
                        qualified_name=f"{artifact.uri}::{name}", snapshot_id=context.snapshot.id,
                        authority=context.authority_for(artifact, AuthorityClass.COMPILER_DECISION), stage=Stage.LLVM.value,
                        attrs=function_attrs,
                        anchors=[SourceAnchor(artifact_id=artifact.id, start_line=line_number,
                                              start_column=1, end_line=line_number,
                                              end_column=len(raw_line) + 1)],
                    )
                    graph.add_entity(current_function)
                    artifact_entity_ids.append(current_function.id)
                    graph.add_relation(Relation(src=unit.id, dst=current_function.id, kind="ir.contains",
                                                snapshot_id=context.snapshot.id,
                                                authority=context.authority_for(artifact, AuthorityClass.COMPILER_DECISION),
                                                stage=Stage.LLVM.value))
                    functions[name] = current_function
                    current_block = None
                    continue
                if current_function and line.strip() == "}":
                    current_function = None
                    current_block = None
                    continue
                if not current_function:
                    if not _covered_module_line(line):
                        unparsed_construct_lines.append(line_number)
                    continue
                label = _LABEL.match(line)
                if label:
                    name = label.group(1)
                    current_block = Entity(
                        kind="ir.llvm.block", name=name,
                        qualified_name=f"{current_function.qualified_name}::{name}",
                        snapshot_id=context.snapshot.id,
                        authority=context.authority_for(artifact, AuthorityClass.COMPILER_DECISION), stage=Stage.LLVM.value,
                        attrs={"plane": "evidence", "hot": False,
                               "cfg_is_hls_topology": False},
                        anchors=[SourceAnchor(artifact_id=artifact.id, start_line=line_number,
                                              start_column=1, end_line=line_number,
                                              end_column=len(raw_line) + 1)],
                    )
                    graph.add_entity(current_block)
                    artifact_entity_ids.append(current_block.id)
                    graph.add_relation(Relation(src=current_function.id, dst=current_block.id,
                                                kind="ir.contains", snapshot_id=context.snapshot.id,
                                                authority=context.authority_for(artifact, AuthorityClass.COMPILER_DECISION),
                                                stage=Stage.LLVM.value))
                    blocks[(current_function.name, name)] = current_block
                    continue
                instruction = _INSTRUCTION.match(line)
                if not instruction or line.lstrip().startswith(";"):
                    if line.strip() and not line.lstrip().startswith(";"):
                        unparsed_construct_lines.append(line_number)
                    continue
                instruction_count += 1
                if instruction_count > max_instructions:
                    artifact_truncated = True
                    result.diagnostics.append(Diagnostic(
                        snapshot_id=context.snapshot.id, code="llvm.instruction_limit",
                        severity=DiagnosticSeverity.WARNING,
                        message=f"LLVM evidence truncated at {max_instructions} instructions",
                        stage=Stage.LLVM.value, artifact_id=artifact.id,
                    ))
                    break
                result_name, opcode, tail = instruction.groups()
                if opcode not in _LLVM_INSTRUCTION_OPCODES:
                    unparsed_construct_lines.append(line_number)
                    continue
                semantic_tail = _mask_quoted_strings(tail)
                dbg = _DBG.search(semantic_tail)
                anchors = [SourceAnchor(artifact_id=artifact.id, start_line=line_number,
                                        start_column=1, end_line=line_number,
                                        end_column=len(raw_line) + 1)]
                if dbg and dbg.group(1) in debug_locations:
                    source_line, source_column, _scope = debug_locations[dbg.group(1)]
                    anchors.append(SourceAnchor(artifact_id=artifact.id, start_line=source_line,
                                                start_column=max(1, source_column),
                                                ir_location=f"!dbg !{dbg.group(1)}",
                                                mapping_kind="llvm.debug",
                                                ambiguity="debug file scope requires DI scope traversal"))
                memory_kind = _memory_access_kind(opcode)
                bitwidths = _integer_widths(semantic_tail)
                index_kinds = _index_kinds(opcode, semantic_tail)
                operation = Entity(
                    kind="ir.llvm.operation", name=opcode,
                    qualified_name=f"{artifact.uri}:{line_number}:{opcode}:{result_name or '-'}",
                    snapshot_id=context.snapshot.id,
                    authority=context.authority_for(artifact, AuthorityClass.COMPILER_DECISION), stage=Stage.LLVM.value,
                    attrs={
                        "plane": "evidence", "hot": False, "opcode": opcode,
                        "ssa_result": result_name, "cfg_is_hls_topology": False,
                        "memory_access": memory_kind is not None,
                        **({"memory_access_kind": memory_kind} if memory_kind else {}),
                        **({"bitwidths": bitwidths} if bitwidths else {}),
                        **({"index_kinds": index_kinds} if index_kinds else {}),
                    },
                    anchors=anchors,
                )
                graph.add_entity(operation)
                artifact_entity_ids.append(operation.id)
                graph.add_relation(Relation(src=(current_block or current_function).id,
                                            dst=operation.id, kind="ir.contains",
                                            snapshot_id=context.snapshot.id,
                                            authority=context.authority_for(artifact, AuthorityClass.COMPILER_DECISION),
                                            stage=Stage.LLVM.value))
                if opcode in {"br", "switch", "indirectbr"} and current_block:
                    for target in _BRANCH.findall(semantic_tail):
                        pending_branches.append((current_function.name, current_block.id, target))
                if opcode in {"call", "invoke", "callbr"}:
                    match = _CALL.search(semantic_tail)
                    if match:
                        pending_calls.append((operation.id, match.group(1)))
            if current_function is not None:
                unparsed_construct_lines.append(max(1, len(source_lines)))
            unparsed_construct_lines = sorted(set(unparsed_construct_lines))
            for function_name, source_id, label_name in pending_branches:
                target = blocks.get((function_name, label_name))
                if target:
                    graph.add_relation(Relation(
                        src=source_id, dst=target.id, kind="llvm.cfg",
                        snapshot_id=context.snapshot.id,
                        authority=context.authority_for(artifact, AuthorityClass.COMPILER_DECISION), stage=Stage.LLVM.value,
                        attrs={"hardware_topology": False},
                    ))
            for operation_id, callee in pending_calls:
                target = functions.get(callee)
                if target:
                    graph.add_relation(Relation(
                        src=operation_id, dst=target.id, kind="llvm.calls",
                        snapshot_id=context.snapshot.id,
                        authority=context.authority_for(artifact, AuthorityClass.COMPILER_DECISION), stage=Stage.LLVM.value,
                        attrs={
                            "hardware_instance": False,
                            "hardware_topology": False,
                        },
                    ))
            feature_domain_complete = (
                not artifact_truncated and not unparsed_construct_lines
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
                total_unparsed_constructs += len(unparsed_construct_lines)
                result.diagnostics.append(Diagnostic(
                    snapshot_id=context.snapshot.id,
                    code="llvm.static_feature_domain_incomplete",
                    severity=DiagnosticSeverity.WARNING,
                    message=(
                        "LLVM IR contains constructs outside the fixed "
                        "static-feature parser; aggregate completeness was withheld"
                    ),
                    stage=Stage.LLVM.value,
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
        result.coverage = {"instructions": instruction_count,
                           "cfg_is_hls_topology": False,
                           "complete_static_feature_artifacts": (
                               complete_feature_artifacts
                           ),
                           "unparsed_static_feature_constructs": (
                               total_unparsed_constructs
                           )}
        if complete_feature_artifacts:
            result.capabilities.append(
                "ir.llvm.complete_static_feature_domain"
            )
        return result
