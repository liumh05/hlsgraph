"""LLVM IR evidence extraction; CFG is never promoted to HLS architecture topology."""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from ..graph import CanonicalGraph
from ..manifest import project_path
from ..model import AuthorityClass, Diagnostic, DiagnosticSeverity, Entity, Relation, SourceAnchor, Stage
from .base import ExtractionContext, ExtractionResult


_DEFINE = re.compile(r"^\s*define\b.*?@([\w.$-]+)\s*\(")
_LABEL = re.compile(r"^([\w.$-]+):")
_INSTRUCTION = re.compile(r"^\s*(?:(%[\w.$-]+)\s*=\s*)?([a-z][a-z0-9_.]*)\b(.*)$")
_CALL = re.compile(r"@([\w.$-]+)\s*\(")
_BRANCH = re.compile(r"label\s+%([\w.$-]+)")
_DBG = re.compile(r"!dbg\s+!(\d+)")
_DILOC = re.compile(r"!(\d+)\s*=\s*!DILocation\(line:\s*(\d+),\s*column:\s*(\d+),\s*scope:\s*!(\d+)")
_DIFILE = re.compile(r'!(\d+)\s*=\s*!DIFile\(filename:\s*"([^"]+)",\s*directory:\s*"([^"]*)"')
_INTEGER_WIDTH = re.compile(r"(?<![\w!])i([1-9]\d*)\b")
_TYPED_INDEX = re.compile(r"\bi[1-9]\d*\s+(-?\d+|%[\w.$-]+)\b")


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
    version = "1"

    def supports(self, context: ExtractionContext) -> bool:
        return any(Path(item.uri).suffix.lower() in {".ll", ".bc"}
                   for item in context.artifacts.values())

    def extract(self, context: ExtractionContext) -> ExtractionResult:
        graph = CanonicalGraph(snapshot_id=context.snapshot.id,
                               metadata={"llvm_cfg_is_hls_topology": False})
        result = ExtractionResult(graph=graph,
                                  capabilities=["ir.llvm.cfg_evidence", "ir.llvm.memory_evidence"])
        instruction_count = 0
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
            debug_locations = {match.group(1): (int(match.group(2)), int(match.group(3)), match.group(4))
                               for match in _DILOC.finditer(text)}
            debug_files = {match.group(1): (match.group(2), match.group(3))
                           for match in _DIFILE.finditer(text)}
            unit = Entity(kind="ir.llvm.module", name=Path(artifact.uri).name,
                          qualified_name=artifact.uri, snapshot_id=context.snapshot.id,
                          authority=context.authority_for(artifact, AuthorityClass.COMPILER_DECISION), stage=Stage.LLVM.value,
                          attrs={"plane": "evidence", "hot": False,
                                 "cfg_is_hls_topology": False},
                          anchors=[SourceAnchor(artifact_id=artifact.id)])
            graph.add_entity(unit)
            current_function: Entity | None = None
            current_block: Entity | None = None
            blocks: dict[tuple[str, str], Entity] = {}
            pending_branches: list[tuple[str, str, str]] = []
            functions: dict[str, Entity] = {}
            pending_calls: list[tuple[str, str]] = []
            for line_number, line in enumerate(text.splitlines(), 1):
                define = _DEFINE.match(line)
                if define:
                    name = define.group(1)
                    function_attrs = {"plane": "evidence", "hot": False}
                    widths = _integer_widths(line)
                    if widths:
                        function_attrs["bitwidths"] = widths
                    current_function = Entity(
                        kind="ir.llvm.function", name=name,
                        qualified_name=f"{artifact.uri}::{name}", snapshot_id=context.snapshot.id,
                        authority=context.authority_for(artifact, AuthorityClass.COMPILER_DECISION), stage=Stage.LLVM.value,
                        attrs=function_attrs,
                        anchors=[SourceAnchor(artifact_id=artifact.id, start_line=line_number,
                                              start_column=1, end_line=line_number,
                                              end_column=len(line) + 1)],
                    )
                    graph.add_entity(current_function)
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
                                              end_column=len(line) + 1)],
                    )
                    graph.add_entity(current_block)
                    graph.add_relation(Relation(src=current_function.id, dst=current_block.id,
                                                kind="ir.contains", snapshot_id=context.snapshot.id,
                                                authority=context.authority_for(artifact, AuthorityClass.COMPILER_DECISION),
                                                stage=Stage.LLVM.value))
                    blocks[(current_function.name, name)] = current_block
                    continue
                instruction = _INSTRUCTION.match(line)
                if not instruction or line.lstrip().startswith(";"):
                    continue
                instruction_count += 1
                if instruction_count > max_instructions:
                    result.diagnostics.append(Diagnostic(
                        snapshot_id=context.snapshot.id, code="llvm.instruction_limit",
                        severity=DiagnosticSeverity.WARNING,
                        message=f"LLVM evidence truncated at {max_instructions} instructions",
                        stage=Stage.LLVM.value, artifact_id=artifact.id,
                    ))
                    break
                result_name, opcode, tail = instruction.groups()
                dbg = _DBG.search(tail)
                anchors = [SourceAnchor(artifact_id=artifact.id, start_line=line_number,
                                        start_column=1, end_line=line_number,
                                        end_column=len(line) + 1)]
                if dbg and dbg.group(1) in debug_locations:
                    source_line, source_column, _scope = debug_locations[dbg.group(1)]
                    anchors.append(SourceAnchor(artifact_id=artifact.id, start_line=source_line,
                                                start_column=max(1, source_column),
                                                ir_location=f"!dbg !{dbg.group(1)}",
                                                mapping_kind="llvm.debug",
                                                ambiguity="debug file scope requires DI scope traversal"))
                memory_kind = _memory_access_kind(opcode)
                bitwidths = _integer_widths(tail)
                index_kinds = _index_kinds(opcode, tail)
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
                graph.add_relation(Relation(src=(current_block or current_function).id,
                                            dst=operation.id, kind="ir.contains",
                                            snapshot_id=context.snapshot.id,
                                            authority=context.authority_for(artifact, AuthorityClass.COMPILER_DECISION),
                                            stage=Stage.LLVM.value))
                if opcode in {"br", "switch", "indirectbr"} and current_block:
                    for target in _BRANCH.findall(tail):
                        pending_branches.append((current_function.name, current_block.id, target))
                if opcode in {"call", "invoke", "callbr"}:
                    match = _CALL.search(tail)
                    if match:
                        pending_calls.append((operation.id, match.group(1)))
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
                        attrs={"hardware_instance": False},
                    ))
        result.coverage = {"instructions": instruction_count,
                           "cfg_is_hls_topology": False}
        return result
