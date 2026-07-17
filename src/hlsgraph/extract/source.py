"""Compilation-database-driven libclang extraction.

The regex scanner is intentionally a separate explicit degraded extractor.  It is
never used as an implicit fallback.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from ..graph import CanonicalGraph
from ..manifest import project_path
from ..model import (
    AuthorityClass,
    Completeness,
    Diagnostic,
    DiagnosticSeverity,
    Entity,
    Observation,
    Relation,
    SourceAnchor,
    Stage,
    TranslationUnit,
)
from .base import ExtractionContext, ExtractionError, ExtractionResult


_PRAGMA = re.compile(r"^\s*#\s*pragma\s+HLS\s+([A-Za-z_][A-Za-z0-9_]*)\s*(.*)$", re.I)


def _strip_inline_comments(text: str) -> str:
    """Remove C/C++ comments without treating comment text as directive data.

    Pragmas are line-oriented.  Quoted strings are retained, while ``//`` and
    ``/* ... */`` start comments only outside a quote.  An unterminated block
    comment therefore safely discards the remainder of the pragma line.
    """
    result: list[str] = []
    quote: str | None = None
    escaped = False
    index = 0
    while index < len(text):
        char = text[index]
        following = text[index + 1] if index + 1 < len(text) else ""
        if quote:
            result.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            index += 1
            continue
        if char in {"'", '"'}:
            quote = char
            result.append(char)
            index += 1
            continue
        if char == "/" and following == "/":
            break
        if char == "/" and following == "*":
            end = text.find("*/", index + 2)
            if end < 0:
                break
            index = end + 2
            continue
        result.append(char)
        index += 1
    return "".join(result).strip()


def _qualified(cursor: Any) -> str:
    parts: list[str] = []
    current = cursor
    while current is not None:
        spelling = getattr(current, "spelling", "")
        if spelling:
            parts.append(spelling)
        current = getattr(current, "semantic_parent", None)
        if current is not None and str(getattr(current, "kind", "")).endswith("TRANSLATION_UNIT"):
            break
    parts.reverse()
    display = getattr(cursor, "displayname", "")
    if display and parts:
        parts[-1] = display
    return "::".join(parts) or getattr(cursor, "spelling", "") or "anonymous"


def _relative_file(context: ExtractionContext, filename: str | None) -> str | None:
    if not filename:
        return None
    root = context.project_root.resolve()
    try:
        return Path(filename).resolve().relative_to(root).as_posix()
    except ValueError:
        return None


def _anchor(context: ExtractionContext, cursor: Any) -> SourceAnchor | None:
    location = getattr(cursor, "location", None)
    extent = getattr(cursor, "extent", None)
    filename = str(location.file) if location and location.file else None
    relative = _relative_file(context, filename)
    artifact = context.artifact_for_uri(relative) if relative else None
    if artifact is None:
        return None
    return SourceAnchor(
        artifact_id=artifact.id,
        start_line=int(extent.start.line) if extent else int(location.line),
        start_column=int(extent.start.column) if extent else int(location.column),
        end_line=int(extent.end.line) if extent else int(location.line),
        end_column=int(extent.end.column) if extent else int(location.column),
        symbol=getattr(cursor, "spelling", None) or None,
    )


def _unit_args(context: ExtractionContext, unit: TranslationUnit) -> list[str]:
    root = str(context.project_root.resolve())
    source = str(project_path(context.project_root, unit.file))
    raw = [arg.replace("${PROJECT_ROOT}", root) for arg in unit.arguments]
    if raw and not raw[0].startswith("-"):
        raw = raw[1:]
    result: list[str] = []
    skip = False
    for arg in raw:
        if skip:
            skip = False
            continue
        if arg in {"-c", "/c"}:
            continue
        if arg in {"-o", "/Fo"}:
            skip = True
            continue
        if not arg.startswith("-"):
            candidate = Path(arg)
            if not candidate.is_absolute():
                candidate = context.project_root / unit.directory / candidate
            if candidate.resolve() == Path(source).resolve():
                continue
        result.append(arg)
    for include in context.manifest.build.include_dirs:
        result.extend(["-I", str(project_path(context.project_root, include))])
    for key, value in sorted(context.manifest.build.defines.items()):
        result.append(f"-D{key}={value}" if value else f"-D{key}")
    result.extend(context.manifest.build.cflags)
    return result


def _tokens_to_options(text: str) -> dict[str, Any]:
    options: dict[str, Any] = {}
    positional: list[str] = []
    for token in re.findall(r'"[^"]*"|\S+', text):
        token = token.strip('"')
        if "=" in token:
            key, value = token.split("=", 1)
            options[key.lower()] = _number(value)
        else:
            positional.append(token)
    if positional:
        options["flags"] = positional
    return options


def _number(value: str) -> Any:
    try:
        return int(value, 0)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


class LibClangExtractor:
    name = "source.libclang"
    version = "1"

    def supports(self, context: ExtractionContext) -> bool:
        # Being unsupported would silently omit the source plane.  The standard
        # backend instead runs and emits a fatal diagnostic for missing context.
        return True

    @staticmethod
    def available() -> bool:
        try:
            from clang import cindex  # noqa: F401
            return True
        except Exception:
            return False

    @staticmethod
    def runtime_identity() -> dict[str, Any]:
        """Fingerprint the exact Python binding and native libclang bytes."""
        try:
            from clang import cindex
        except Exception:
            return {"available": False, "reason": "python_binding_unavailable"}
        try:
            package_version = importlib.metadata.version("libclang")
        except importlib.metadata.PackageNotFoundError:
            package_version = "unregistered"
        try:
            library = cindex.conf.lib
            library_path = Path(cindex.conf.get_filename()).resolve()
            binding_path = Path(cindex.__file__).resolve()
            if not library_path.is_file() or not binding_path.is_file():
                raise OSError("libclang runtime files are unavailable")
            library.clang_getClangVersion.restype = cindex._CXString
            native = cindex._CXString.from_result(library.clang_getClangVersion())
            native_version = (native.decode("utf-8", errors="replace")
                              if isinstance(native, bytes) else str(native))
            library_bytes = library_path.read_bytes()
            binding_bytes = binding_path.read_bytes()
        except Exception as exc:
            return {
                "available": False,
                "reason": "native_runtime_unavailable",
                "error_type": type(exc).__name__,
                "python_distribution_version": package_version,
            }
        return {
            "available": True,
            "python_distribution_version": package_version,
            "native_version": native_version,
            "native_sha256": hashlib.sha256(library_bytes).hexdigest(),
            "native_size": len(library_bytes),
            "binding_sha256": hashlib.sha256(binding_bytes).hexdigest(),
        }

    def extract(self, context: ExtractionContext) -> ExtractionResult:
        build = context.manifest.build
        if not build.translation_units:
            raise ExtractionError(
                "no translation units are configured; provide compile_commands.json or explicit build.translation_units"
            )
        has_context = bool(build.compile_commands or build.include_dirs or build.defines or
                           build.cflags or any(unit.arguments for unit in build.translation_units))
        if not has_context:
            raise ExtractionError(
                "compilation context is incomplete; provide compile_commands.json or explicit arguments/includes/defines"
            )
        if not self.available():
            raise ExtractionError("libclang is unavailable; install hlsgraph[clang] or explicitly select degraded mode")
        from clang import cindex

        graph = CanonicalGraph(snapshot_id=context.snapshot.id,
                               metadata={"source_backend": self.name, "fidelity": "ast"})
        result = ExtractionResult(graph=graph, capabilities=["source.ast", "directive.source_scope"])
        function_by_name: dict[str, list[str]] = defaultdict(list)
        pending_calls: list[tuple[str, str, SourceAnchor | None]] = []
        cursor_entity: dict[int, str] = {}
        untracked_project_inputs: set[str] = set()
        index = cindex.Index.create()

        for unit in context.manifest.build.translation_units:
            source = project_path(context.project_root, unit.file)
            if not source.is_file():
                result.diagnostics.append(Diagnostic(
                    snapshot_id=context.snapshot.id, code="source.missing",
                    severity=DiagnosticSeverity.ERROR,
                    message=f"translation unit is missing: {unit.file}", stage=Stage.SOURCE.value,
                ))
                continue
            args = _unit_args(context, unit)
            try:
                tu = index.parse(str(source), args=args,
                                 options=cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD)
            except Exception as exc:
                raise ExtractionError(f"failed to parse {unit.file}: {exc}") from exc
            # The manifest scanner intentionally remains compiler-independent,
            # but macro includes, response files, or compiler-specific flags can
            # expand the real dependency set.  Never accept an AST whose
            # project-local input bytes were absent from the snapshot.
            for inclusion in tu.get_includes():
                relative = _relative_file(context, str(inclusion.include))
                if relative and context.artifact_for_uri(relative) is None:
                    untracked_project_inputs.add(relative)
            for diagnostic in tu.diagnostics:
                severity = {
                    0: DiagnosticSeverity.INFO,
                    1: DiagnosticSeverity.INFO,
                    2: DiagnosticSeverity.WARNING,
                    3: DiagnosticSeverity.ERROR,
                    4: DiagnosticSeverity.CRITICAL,
                }.get(int(diagnostic.severity), DiagnosticSeverity.WARNING)
                location = diagnostic.location
                relative = _relative_file(context, str(location.file) if location.file else None)
                artifact = context.artifact_for_uri(relative) if relative else None
                anchor = SourceAnchor(artifact_id=artifact.id, start_line=location.line,
                                      start_column=location.column) if artifact and location.line else None
                result.diagnostics.append(Diagnostic(
                    snapshot_id=context.snapshot.id, code="clang.diagnostic",
                    severity=severity, message=str(diagnostic.spelling), stage=Stage.AST.value,
                    artifact_id=artifact.id if artifact else None, anchor=anchor,
                    metadata={"category": diagnostic.category_name or None, "tu": unit.file},
                ))

            def visit(cursor: Any, parent_entity: str | None = None,
                      current_function_id: str | None = None,
                      current_function_qname: str | None = None) -> None:
                relative = _relative_file(context, str(cursor.location.file) if cursor.location.file else None)
                kind_name = cursor.kind.name
                if relative is None and kind_name != "TRANSLATION_UNIT":
                    return
                if (relative is not None
                        and context.artifact_for_uri(relative) is None):
                    untracked_project_inputs.add(relative)
                    # Do not materialize unanchored entities from bytes that
                    # were absent from the immutable snapshot.
                    return
                entity: Entity | None = None
                anchor = _anchor(context, cursor)
                qname = _qualified(cursor)
                attrs: dict[str, Any] = {}
                entity_kind: str | None = None
                display_name = cursor.spelling or cursor.displayname or kind_name.lower()

                if kind_name in {"FUNCTION_DECL", "CXX_METHOD", "FUNCTION_TEMPLATE"} and cursor.is_definition():
                    entity_kind = "hls.kernel" if cursor.spelling == context.manifest.build.top else "hls.function"
                    attrs = {"return_type": getattr(cursor.result_type, "spelling", None),
                             "display_name": cursor.displayname}
                elif kind_name in {"FOR_STMT", "WHILE_STMT", "DO_STMT"}:
                    entity_kind = "hls.loop"
                    line = anchor.start_line if anchor else 0
                    display_name = self._loop_label(context, relative, line) or f"loop@{line}"
                    qname = f"{current_function_qname or relative}::{display_name}@{line}"
                    attrs = {"loop_kind": kind_name.lower().replace("_stmt", "")}
                elif kind_name == "PARM_DECL" and current_function_id:
                    entity_kind = "hls.port"
                    qname = f"{current_function_qname}::{cursor.spelling}"
                    attrs = {"type": cursor.type.spelling, "direction": "unknown"}
                elif kind_name == "VAR_DECL" and current_function_id:
                    spelling = cursor.type.spelling
                    entity_kind = "hls.stream" if "stream<" in spelling.replace(" ", "") else (
                        "hls.memory" if cursor.type.kind.name in {"CONSTANTARRAY", "INCOMPLETEARRAY", "VARIABLEARRAY"}
                        else "source.variable"
                    )
                    qname = f"{current_function_qname}::{cursor.spelling}@{anchor.start_line if anchor else 0}"
                    attrs = {"type": spelling}
                    if cursor.type.kind.name == "CONSTANTARRAY":
                        attrs["array_size"] = int(cursor.type.element_count)
                        attrs["element_type"] = cursor.type.element_type.spelling

                if entity_kind:
                    entity = Entity(kind=entity_kind, name=display_name, qualified_name=qname,
                                    snapshot_id=context.snapshot.id, authority=AuthorityClass.STATIC_FACT,
                                    stage=Stage.AST.value, attrs=attrs,
                                    anchors=[anchor] if anchor else [])
                    graph.add_entity(entity)
                    cursor_entity[cursor.hash] = entity.id
                    if parent_entity:
                        graph.add_relation(Relation(
                            src=parent_entity, dst=entity.id, kind="hls.contains",
                            snapshot_id=context.snapshot.id, authority=AuthorityClass.STATIC_FACT,
                            stage=Stage.AST.value,
                        ))
                    if entity_kind in {"hls.kernel", "hls.function"}:
                        current_function_id = entity.id
                        current_function_qname = qname
                        function_by_name[cursor.spelling].append(entity.id)
                    parent_entity = entity.id

                if kind_name == "CALL_EXPR" and current_function_id:
                    owner_id = current_function_id
                    callee = cursor.spelling or cursor.displayname.split("(")[0]
                    if owner_id and callee:
                        pending_calls.append((owner_id, callee, anchor))

                for child in cursor.get_children():
                    visit(child, parent_entity, current_function_id, current_function_qname)

            visit(tu.cursor)

        for relative in sorted(untracked_project_inputs):
            result.diagnostics.append(Diagnostic(
                snapshot_id=context.snapshot.id, code="source.untracked_project_include",
                severity=DiagnosticSeverity.ERROR,
                message=(f"libclang read project-local input {relative!r} that is not hashed in "
                         "the snapshot; add it to artifact_paths or compilation include context"),
                stage=Stage.AST.value, metadata={"path": relative},
            ))

        for owner, callee, anchor in pending_calls:
            targets = sorted(set(function_by_name.get(callee, [])))
            if len(targets) == 1:
                graph.add_relation(Relation(
                    src=owner, dst=targets[0], kind="software.calls", snapshot_id=context.snapshot.id,
                    authority=AuthorityClass.STATIC_FACT, stage=Stage.AST.value,
                    anchors=[anchor] if anchor else [],
                    attrs={"hardware_instance": False},
                ))
            elif len(targets) > 1:
                result.diagnostics.append(Diagnostic(
                    snapshot_id=context.snapshot.id, code="mapping.ambiguous_call",
                    severity=DiagnosticSeverity.WARNING,
                    message=f"call to {callee!r} has {len(targets)} project-local candidates; no edge was guessed",
                    stage=Stage.AST.value, subject_id=owner,
                ))

        self._attach_source_pragmas(context, result)
        result.coverage = {
            "translation_units": len(context.manifest.build.translation_units),
            "entities": len(graph.entities), "relations": len(graph.relations),
            "errors": sum(d.severity in {DiagnosticSeverity.ERROR, DiagnosticSeverity.CRITICAL}
                          for d in result.diagnostics),
            "fidelity": "libclang",
        }
        if not any(entity.kind == "hls.kernel" for entity in graph.entities.values()):
            result.diagnostics.append(Diagnostic(
                snapshot_id=context.snapshot.id, code="source.top_not_found",
                severity=DiagnosticSeverity.ERROR,
                message=f"configured top {context.manifest.build.top!r} was not found by libclang",
                stage=Stage.AST.value,
            ))
        return result

    @staticmethod
    def _loop_label(context: ExtractionContext, relative: str, line: int) -> str | None:
        if line <= 1:
            return None
        lines = project_path(context.project_root, relative).read_text(
            encoding="utf-8", errors="replace").splitlines()
        for index in range(max(0, line - 3), min(len(lines), line)):
            match = re.match(r"\s*([A-Za-z_]\w*)\s*:\s*(?:for|while)?", lines[index])
            if match:
                return match.group(1)
        return None

    @staticmethod
    def _attach_source_pragmas(context: ExtractionContext, result: ExtractionResult) -> None:
        graph = result.graph
        for artifact in sorted(context.artifacts.values(), key=lambda item: item.uri):
            if not artifact.kind.startswith("source."):
                continue
            path = project_path(context.project_root, artifact.uri)
            if not path.is_file():
                continue
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            for line_number, line in enumerate(lines, 1):
                match = _PRAGMA.match(line)
                if not match:
                    continue
                directive_kind = match.group(1).upper()
                # Comments are source prose, not pragma semantics.  Keeping
                # them here would leak private source through graph/REST/MCP/ML
                # exports and could also invent bogus directive flags.
                options = _tokens_to_options(_strip_inline_comments(match.group(2)))
                anchor = SourceAnchor(artifact_id=artifact.id, start_line=line_number,
                                      start_column=max(1, line.find("#") + 1),
                                      end_line=line_number, end_column=len(line) + 1)
                target = _scope_for_pragma(
                    graph, artifact.id, line_number, directive_kind, options
                )
                directive = Entity(
                    kind="hls.directive", name=directive_kind,
                    qualified_name=f"{artifact.uri}:{line_number}:{directive_kind}",
                    snapshot_id=context.snapshot.id,
                    authority=AuthorityClass.DECLARED_CONSTRAINT, stage=Stage.SOURCE.value,
                    attrs={"directive_kind": directive_kind, "options": options,
                           "origin": "source_pragma", "precedence": 10,
                           "state": "requested"}, anchors=[anchor],
                    completeness=(Completeness.COMPLETE if target
                                  else Completeness.AMBIGUOUS),
                )
                graph.add_entity(directive)
                if target:
                    graph.add_relation(Relation(
                        src=directive.id, dst=target.id, kind="hls.annotates",
                        snapshot_id=context.snapshot.id,
                        authority=AuthorityClass.DECLARED_CONSTRAINT, stage=Stage.SOURCE.value,
                        attrs={"scope_node_id": target.id, "scope_resolution": "source_ast"},
                        anchors=[anchor],
                    ))
                    result.observations.append(Observation(
                        snapshot_id=context.snapshot.id, subject_id=directive.id,
                        predicate="directive.requested", value=options or True,
                        stage=Stage.SOURCE.value, authority=AuthorityClass.DECLARED_CONSTRAINT,
                        artifact_id=artifact.id, anchor=anchor,
                        metadata={"directive_kind": directive_kind, "scope_id": target.id},
                    ))
                else:
                    result.diagnostics.append(Diagnostic(
                        snapshot_id=context.snapshot.id, code="directive.unresolved_scope",
                        severity=DiagnosticSeverity.WARNING,
                        message=f"could not deterministically bind {directive_kind} at {artifact.uri}:{line_number}",
                        stage=Stage.SOURCE.value, subject_id=directive.id,
                        artifact_id=artifact.id, anchor=anchor,
                    ))


def _scope_for_pragma(graph: CanonicalGraph, artifact_id: str, line: int,
                      kind: str, options: dict[str, Any]) -> Entity | None:
    candidates: list[Entity] = []
    for entity in graph.entities.values():
        for anchor in entity.anchors:
            if anchor.artifact_id == artifact_id and anchor.start_line and anchor.end_line:
                candidates.append(entity)
                break
    containing = [entity for entity in candidates if any(
        anchor.artifact_id == artifact_id and anchor.start_line is not None and anchor.end_line is not None
        and anchor.start_line <= line <= anchor.end_line for anchor in entity.anchors
    )]
    variable = str(options.get("variable") or options.get("port") or "")
    if variable:
        matches = [entity for entity in candidates if entity.name == variable]
        owners = [entity for entity in containing
                  if entity.kind in {"hls.kernel", "hls.function"}]
        if owners:
            owner = min(owners, key=lambda entity: min(
                (anchor.end_line or line) - (anchor.start_line or line)
                for anchor in entity.anchors
            ))
            owner_name = owner.qualified_name or owner.name
            scoped = [entity for entity in matches
                      if (entity.qualified_name or "").startswith(owner_name + "::")]
            preferred = [entity for entity in scoped
                         if entity.kind in {"hls.stream", "hls.memory", "hls.port"}]
            if len(preferred) == 1:
                return preferred[0]
            if len(scoped) == 1:
                return scoped[0]
        if len(matches) == 1:
            return matches[0]
        # A directive that explicitly names storage/port scope must never fall
        # back to a loop or function merely because that name is ambiguous.
        return None
    loop_directives = {"PIPELINE", "UNROLL", "LOOP_FLATTEN", "LOOP_TRIPCOUNT", "DEPENDENCE"}
    if kind in loop_directives:
        # HLS loop pragmas normally precede the loop they annotate.  Prefer the
        # nearest unique following loop before considering an enclosing loop;
        # otherwise a pragma before an inner loop is incorrectly attached to
        # its outer loop.
        following = [entity for entity in candidates if entity.kind == "hls.loop" and any(
            anchor.start_line and 0 < anchor.start_line - line <= 3 for anchor in entity.anchors
        )]
        if following:
            distances = {entity.id: min(
                (anchor.start_line or 10**9) - line for anchor in entity.anchors
                if anchor.start_line and anchor.start_line > line
            ) for entity in following}
            nearest = min(distances.values())
            winners = [entity for entity in following if distances[entity.id] == nearest]
            if len(winners) == 1:
                return winners[0]
            return None
        loops = [entity for entity in containing if entity.kind == "hls.loop"]
        if loops:
            spans = {entity.id: min(
                (anchor.end_line or line) - (anchor.start_line or line) for anchor in entity.anchors
            ) for entity in loops}
            smallest = min(spans.values())
            winners = [entity for entity in loops if spans[entity.id] == smallest]
            return winners[0] if len(winners) == 1 else None
    functions = [entity for entity in containing if entity.kind in {"hls.kernel", "hls.function"}]
    if functions:
        spans = {entity.id: min(
            (anchor.end_line or line) - (anchor.start_line or line) for anchor in entity.anchors
        ) for entity in functions}
        smallest = min(spans.values())
        winners = [entity for entity in functions if spans[entity.id] == smallest]
        return winners[0] if len(winners) == 1 else None
    return None


class RegexSourceExtractor:
    """Explicit degraded source scanner; never selected automatically."""

    name = "source.regex_degraded"
    version = "1"

    def supports(self, context: ExtractionContext) -> bool:
        return context.allow_degraded

    def extract(self, context: ExtractionContext) -> ExtractionResult:
        if not context.allow_degraded:
            raise ExtractionError("regex extraction requires allow_degraded=True")
        if not context.manifest.build.translation_units:
            raise ExtractionError("degraded extraction still requires explicit translation units")
        graph = CanonicalGraph(snapshot_id=context.snapshot.id,
                               metadata={"source_backend": self.name, "fidelity": "degraded"})
        result = ExtractionResult(graph=graph, capabilities=["source.degraded"])
        function_pattern = re.compile(
            r"(?:^|\n)\s*(?:[A-Za-z_]\w*(?:\s*<[^;{}]+>)?[\s*&]+)+([A-Za-z_]\w*)\s*\([^;{}]*\)\s*\{",
            re.MULTILINE,
        )
        loop_pattern = re.compile(r"\b(for|while)\s*\(")
        for unit in context.manifest.build.translation_units:
            artifact = context.artifact_for_uri(unit.file)
            if artifact is None:
                continue
            text = project_path(context.project_root, unit.file).read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines()
            functions: list[Entity] = []
            for match in function_pattern.finditer(text):
                name = match.group(1)
                line = text.count("\n", 0, match.start(1)) + 1
                anchor = SourceAnchor(artifact_id=artifact.id, start_line=line, start_column=1,
                                      end_line=max(line, len(lines)), end_column=1,
                                      mapping_kind="regex", ambiguity="function extent is approximate")
                entity = Entity(
                    kind="hls.kernel" if name == context.manifest.build.top else "hls.function",
                    name=name, qualified_name=name, snapshot_id=context.snapshot.id,
                    authority=AuthorityClass.STATIC_FACT, stage=Stage.SOURCE.value,
                    attrs={"fidelity": "degraded"}, anchors=[anchor],
                    completeness=Completeness.PARTIAL,
                )
                graph.add_entity(entity)
                functions.append(entity)
            for index, match in enumerate(loop_pattern.finditer(text), 1):
                line = text.count("\n", 0, match.start()) + 1
                parent = next((entity for entity in functions if any(
                    anchor.start_line and anchor.end_line and anchor.start_line <= line <= anchor.end_line
                    for anchor in entity.anchors)), None)
                loop = Entity(kind="hls.loop", name=f"loop@{line}",
                              qualified_name=f"{parent.name if parent else unit.file}::loop@{line}",
                              snapshot_id=context.snapshot.id, authority=AuthorityClass.STATIC_FACT,
                              stage=Stage.SOURCE.value, attrs={"loop_kind": match.group(1),
                                                               "fidelity": "degraded"},
                              anchors=[SourceAnchor(artifact_id=artifact.id, start_line=line,
                                                    start_column=1, end_line=line, end_column=1,
                                                    mapping_kind="regex")],
                              completeness=Completeness.PARTIAL)
                graph.add_entity(loop)
                if parent:
                    graph.add_relation(Relation(src=parent.id, dst=loop.id, kind="hls.contains",
                                                snapshot_id=context.snapshot.id,
                                                authority=AuthorityClass.STATIC_FACT,
                                                stage=Stage.SOURCE.value,
                                                completeness=Completeness.PARTIAL))
        result.diagnostics.append(Diagnostic(
            snapshot_id=context.snapshot.id, code="source.degraded_mode",
            severity=DiagnosticSeverity.WARNING,
            message="regex source scanning was explicitly enabled; hardware topology and precise scope are incomplete",
            stage=Stage.SOURCE.value,
        ))
        LibClangExtractor._attach_source_pragmas(context, result)
        result.coverage = {"fidelity": "regex_degraded", "entities": len(graph.entities),
                           "relations": len(graph.relations)}
        return result
