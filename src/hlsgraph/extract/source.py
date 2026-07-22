"""Compilation-database-driven libclang extraction.

The regex scanner is intentionally a separate explicit degraded extractor.  It is
never used as an implicit fallback.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from ..graph import CanonicalGraph
from ..manifest import project_path, resolve_compiler_arguments
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
from .directive_identity import (
    bind_directive_identity,
    directive_identity_metadata,
    resolve_directive_variable_operand,
)


_PRAGMA = re.compile(r"^\s*#\s*pragma\s+HLS\s+([A-Za-z_][A-Za-z0-9_]*)\s*(.*)$", re.I)
_RAW_STRING = re.compile(
    r'(?:u8|u|U|L)?R"(?P<delimiter>[^ ()\\\t\r\n]{0,16})\(.*?\)(?P=delimiter)"',
    re.DOTALL,
)
_COMMENT_OR_LITERAL = re.compile(
    r'//[^\r\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'',
    re.DOTALL,
)
_UNTRACKED_PREPROCESSOR_TOKEN = re.compile(
    r"\b(?:__has_include|__has_include_next|__has_embed|__DATE__|__TIME__|"
    r"__TIMESTAMP__|__FILE__|__FILE_NAME__|__BASE_FILE__|__builtin_FILE|"
    r"__builtin_source_location)\b|"
    r"^\s*#\s*embed\b",
    re.MULTILINE,
)
_AMBIENT_COMPILER_ENVIRONMENT = frozenset({
    "CPATH", "CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH", "OBJC_INCLUDE_PATH",
    "OBJCPLUS_INCLUDE_PATH", "INCLUDE", "SDKROOT", "GCC_EXEC_PREFIX",
    "COMPILER_PATH", "CCC_OVERRIDE_OPTIONS", "CL", "_CL_",
    "MACOSX_DEPLOYMENT_TARGET", "IPHONEOS_DEPLOYMENT_TARGET",
    "TVOS_DEPLOYMENT_TARGET", "WATCHOS_DEPLOYMENT_TARGET",
    "XROS_DEPLOYMENT_TARGET",
})


def _blank_match(match: re.Match[str]) -> str:
    """Remove prose/literals while preserving line boundaries for directives."""
    return "".join("\n" if character == "\n" else " " for character in match.group(0))


def _unsafe_preprocessor_tokens(text: str) -> set[str]:
    # Translation phases replace trigraphs and splice physical lines before
    # comments, literals, and preprocessing tokens are recognized.  Mirror
    # those two transformations so a split reserved builtin cannot bypass the
    # conservative v0.1 gate.
    for source, replacement in (
        ("??=", "#"), ("??/", "\\"), ("??'", "^"), ("??(", "["),
        ("??)", "]"), ("??!", "|"), ("??<", "{"), ("??>", "}"),
        ("??-", "~"), ("%:%:", "##"), ("%:", "#"),
    ):
        text = text.replace(source, replacement)
    text = re.sub(r"\\\r?\n", "", text)
    code = _RAW_STRING.sub(_blank_match, text)
    code = _COMMENT_OR_LITERAL.sub(_blank_match, code)
    result = {match.group(0).strip().split()[0]
              for match in _UNTRACKED_PREPROCESSOR_TOKEN.finditer(code)}
    if "##" in code:
        result.add("preprocessor_token_paste")
    return result


def _ambient_compiler_environment() -> list[str]:
    # Windows environment names are case-insensitive; case-fold everywhere so
    # a mixed-case alias cannot bypass the deterministic-input boundary.
    present = {str(key).upper() for key, value in os.environ.items() if value}
    return sorted(present & _AMBIENT_COMPILER_ENVIRONMENT)


def _skipped_preprocessor_line_ranges(cindex: Any, translation_unit: Any,
                                      filename: Path) -> list[tuple[int, int]]:
    """Read libclang's exact inactive conditional-compilation ranges.

    The Python bindings do not currently wrap ``clang_getSkippedRanges`` even
    though it is part of the stable libclang C API.  Binding it locally keeps
    pragma activity tied to the same TranslationUnit that produced the AST.
    Failure is propagated so callers can fail closed instead of upgrading raw
    source text in an inactive ``#if`` branch to a requested directive.
    """
    from ctypes import POINTER, Structure, c_uint

    class _SourceRangeList(Structure):
        _fields_ = [
            ("count", c_uint),
            ("ranges", POINTER(cindex.SourceRange)),
        ]

    library = cindex.conf.lib
    getter = library.clang_getSkippedRanges
    disposer = library.clang_disposeSourceRangeList
    getter.argtypes = [cindex.TranslationUnit, cindex.File]
    getter.restype = POINTER(_SourceRangeList)
    disposer.argtypes = [POINTER(_SourceRangeList)]
    disposer.restype = None
    source_file = cindex.File.from_name(translation_unit, str(filename))
    ranges = getter(translation_unit, source_file)
    if not ranges:
        return []
    try:
        return [
            (int(ranges.contents.ranges[index].start.line),
             int(ranges.contents.ranges[index].end.line))
            for index in range(int(ranges.contents.count))
        ]
    finally:
        disposer(ranges)


def _tri_and(left: bool | None, right: bool | None) -> bool | None:
    if left is False or right is False:
        return False
    if left is True and right is True:
        return True
    return None


def _tri_or(left: bool | None, right: bool | None) -> bool | None:
    if left is True or right is True:
        return True
    if left is False and right is False:
        return False
    return None


def _tri_not(value: bool | None) -> bool | None:
    return None if value is None else not value


def _literal_preprocessor_condition(expression: str) -> bool | None:
    """Evaluate only the deliberately tiny literal subset used in degraded mode."""
    value = expression.strip()
    while value.startswith("(") and value.endswith(")"):
        value = value[1:-1].strip()
    negate = False
    while value.startswith("!"):
        negate = not negate
        value = value[1:].strip()
    if value not in {"0", "1"}:
        return None
    result = value == "1"
    return not result if negate else result


def _degraded_pragma_activity(artifact_id: str, text: str) -> dict[tuple[str, int], bool | None]:
    """Conservatively classify pragma activity without pretending to preprocess.

    Literal ``#if 0/1`` nesting is deterministic.  Macro-dependent or otherwise
    non-literal branches remain unknown, and their pragmas are withheld by the
    caller with an explicit diagnostic.
    """
    code = _RAW_STRING.sub(_blank_match, text)
    code = _COMMENT_OR_LITERAL.sub(_blank_match, code)
    raw_lines = text.splitlines()
    code_lines = code.splitlines()
    if len(code_lines) < len(raw_lines):
        code_lines.extend([""] * (len(raw_lines) - len(code_lines)))
    current: bool | None = True
    stack: list[dict[str, bool | None]] = []
    result: dict[tuple[str, int], bool | None] = {}
    conditional = re.compile(r"^\s*#\s*(if|ifdef|ifndef|elif|else|endif)\b(.*)$", re.I)
    for line_number, (raw_line, code_line) in enumerate(zip(raw_lines, code_lines), 1):
        directive = conditional.match(code_line)
        if directive:
            name = directive.group(1).casefold()
            expression = directive.group(2)
            if name in {"if", "ifdef", "ifndef"}:
                condition = (_literal_preprocessor_condition(expression)
                             if name == "if" else None)
                stack.append({"parent": current, "seen": condition})
                current = _tri_and(current, condition)
            elif name == "elif" and stack:
                condition = _literal_preprocessor_condition(expression)
                entry = stack[-1]
                current = _tri_and(
                    entry["parent"], _tri_and(_tri_not(entry["seen"]), condition),
                )
                entry["seen"] = _tri_or(entry["seen"], condition)
            elif name == "else" and stack:
                entry = stack[-1]
                current = _tri_and(entry["parent"], _tri_not(entry["seen"]))
                entry["seen"] = True
            elif name == "endif" and stack:
                current = stack.pop()["parent"]
            else:
                # Malformed conditional structure cannot safely certify any
                # subsequent pragma in the degraded scanner.
                current = None
            continue
        if _PRAGMA.match(raw_line):
            result[(artifact_id, line_number)] = current
    return result


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
    root_path = context.project_root.resolve()
    root = str(root_path)
    source = str(project_path(context.project_root, unit.file))
    working_directory = (root_path / unit.directory).resolve()
    raw, _ = resolve_compiler_arguments(
        root_path, working_directory, unit.arguments,
    )
    clang_cl = False
    gxx_driver = False
    if raw:
        executable = re.split(r"[/\\]", raw[0])[-1].casefold()
        stem = executable[:-4] if executable.endswith(".exe") else executable
        if stem in {"ccache", "sccache", "distcc", "env"} or re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*=.*", raw[0]
        ):
            raise ExtractionError(
                "compiler wrapper/environment prefixes are unsupported; store the "
                "fully expanded compiler arguments in the compilation database"
            )
        clang_cl = bool(re.fullmatch(r"(?:.+-)?clang-cl(?:-\d+)?", stem))
        gxx_driver = bool(re.fullmatch(
            r"(?:.+-)?(?:clang\+\+|g\+\+|c\+\+)(?:-\d+)?", stem,
        ))
        msvc_cl = stem == "cl"
        known_driver = bool(
            clang_cl or msvc_cl
            or re.fullmatch(
                r"(?:(?:.+-)?(?:clang\+\+|clang|gcc|g\+\+|cc|c\+\+))(?:-\d+)?",
                stem,
            )
        )
        windows_option = (
            raw[0].casefold().startswith((
                "/i", "/fi", "/d", "/u", "/c", "/fo", "/std:", "/eh",
                "/zc", "/w", "/o", "/md", "/mt", "/gr", "/tp", "/tc",
                "/permissive", "/external:", "/imsvc", "/clang:",
            ))
            and not (os.name != "nt" and Path(raw[0]).is_absolute())
        )
        if msvc_cl:
            raise ExtractionError(
                "cl.exe compilation databases are unsupported; use clang-cl with an "
                "explicit, self-contained project-local compilation context"
            )
        if known_driver:
            raw = raw[1:]
        elif not raw[0].startswith("-") and not windows_option:
            raise ExtractionError(
                "unsupported compiler driver in translation-unit arguments"
            )
        elif windows_option:
            # Explicit manifest translation units may omit argv[0].  A joined
            # /I, /FI, /D, ... first option is context, not an executable path,
            # and selects clang-cl argument semantics.
            clang_cl = True
    result: list[str] = [f"-working-directory={working_directory}"]
    if clang_cl:
        result.append("--driver-mode=cl")
    elif gxx_driver:
        result.append("--driver-mode=g++")
    skip = False
    for arg in raw:
        if skip:
            skip = False
            continue
        if arg in {"-c", "/c"}:
            continue
        if arg in {"-o", "/Fo", "-working-directory"}:
            skip = True
            continue
        if ((arg.startswith("-o") and len(arg) > 2)
                or (arg.startswith("/Fo") and len(arg) > 3)
                or arg.startswith("-working-directory=")):
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
    global_flags, _ = resolve_compiler_arguments(
        root_path, root_path, context.manifest.build.cflags,
    )
    result.extend(global_flags)
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


_DECIMAL_INTEGER = re.compile(r"^[+-]?\d+$")
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _decimal_integer(tokens: list[str]) -> int | None:
    """Parse only an unambiguous decimal integer token sequence.

    Constant folding belongs to the compiler/IR layer.  The AST adapter therefore
    accepts only a literal (with an optional separate sign token) and leaves every
    expression, macro, enum, or symbolic bound unknown.
    """
    text = "".join(tokens)
    return int(text, 10) if _DECIMAL_INTEGER.fullmatch(text) else None


def _positive_trip_count(lower: int, upper: int, step: int,
                         comparison: str) -> int | None:
    if step > 0 and comparison in {"lt", "le"}:
        distance = upper - lower
        if comparison == "le":
            distance += 1
        if distance <= 0:
            return None
        count = (distance + step - 1) // step
        return count if count > 0 else None
    if step < 0 and comparison in {"gt", "ge"}:
        stride = -step
        distance = lower - upper
        if comparison == "ge":
            distance += 1
        if distance <= 0:
            return None
        count = (distance + stride - 1) // stride
        return count if count > 0 else None
    return None


def _constant_for_loop_facts(cursor: Any) -> dict[str, Any]:
    """Extract facts from one exact canonical constant ``for`` header.

    This intentionally recognizes a very small token-level grammar.  It does not
    evaluate names or infer bounds from a variable spelling, which keeps symbolic
    loops explicitly incomplete until higher-fidelity compiler evidence exists.
    """
    try:
        tokens = [str(item.spelling) for item in cursor.get_tokens()]
    except Exception:
        return {}
    try:
        start = tokens.index("(")
    except ValueError:
        return {}
    depth = 0
    header: list[str] = []
    for token in tokens[start + 1:]:
        if token == "(":
            depth += 1
        elif token == ")":
            if depth == 0:
                break
            depth -= 1
        header.append(token)
    else:
        return {}

    sections: list[list[str]] = [[]]
    depth = 0
    for token in header:
        if token in {"(", "[", "{"}:
            depth += 1
        elif token in {")", "]", "}"}:
            depth -= 1
        if token == ";" and depth == 0:
            sections.append([])
        else:
            sections[-1].append(token)
    if len(sections) != 3 or any(not section for section in sections):
        return {}

    initializer, condition, increment = sections
    if initializer.count("=") != 1:
        return {}
    equals = initializer.index("=")
    lhs, rhs = initializer[:equals], initializer[equals + 1:]
    if not lhs or not _IDENTIFIER.fullmatch(lhs[-1]):
        return {}
    induction = lhs[-1]
    lower = _decimal_integer(rhs)
    if lower is None or len(condition) < 3:
        return {}

    operator_index = next(
        (index for index, token in enumerate(condition)
         if token in {"<", "<=", ">", ">="}),
        -1,
    )
    if operator_index != 1 or condition[0] != induction:
        return {}
    if any(token in {"<", "<=", ">", ">="}
           for token in condition[operator_index + 1:-1]):
        return {}
    upper = _decimal_integer(condition[operator_index + 1:])
    if upper is None:
        return {}
    comparison = {"<": "lt", "<=": "le", ">": "gt", ">=": "ge"}[
        condition[operator_index]
    ]

    step: int | None = None
    compact_increment = "".join(increment)
    if compact_increment in {f"++{induction}", f"{induction}++"}:
        step = 1
    elif compact_increment in {f"--{induction}", f"{induction}--"}:
        step = -1
    elif len(increment) >= 3 and increment[0] == induction \
            and increment[1] in {"+=", "-="}:
        magnitude = _decimal_integer(increment[2:])
        if magnitude is not None and magnitude > 0:
            step = magnitude if increment[1] == "+=" else -magnitude
    if step is None:
        return {}

    bounds = {
        "lower": lower,
        "upper": upper,
        "step": step,
        "comparison": comparison,
        "upper_inclusive": comparison in {"le", "ge"},
    }
    result: dict[str, Any] = {
        "loop_bounds": bounds,
        "loop_bounds_exact": True,
        "loop_fact_source": "libclang.tokens.v1",
    }
    trip_count = _positive_trip_count(lower, upper, step, comparison)
    if trip_count is not None:
        result["trip_count"] = trip_count
    return result


class LibClangExtractor:
    name = "source.libclang"
    version = "2"

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
        ambient = _ambient_compiler_environment()
        if ambient:
            result.diagnostics.append(Diagnostic(
                snapshot_id=context.snapshot.id,
                code="source.ambient_compiler_environment",
                severity=DiagnosticSeverity.ERROR,
                message=("ambient compiler environment inputs are unsupported; express "
                         "all preprocessing context in the project manifest"),
                stage=Stage.AST.value,
                metadata={"variable_names": ambient},
            ))
            return result

        unsafe_tokens: set[str] = set()
        compiler_text: dict[str, str] = {}
        for artifact in sorted(context.artifacts.values(), key=lambda item: item.uri):
            path = project_path(context.project_root, artifact.uri)
            if (not artifact.metadata.get("hlsgraph.compiler_reachable_text")
                    and not artifact.kind.startswith("source.")
                    and artifact.role not in {"design_source", "header", "dependency"}):
                continue
            try:
                source_bytes = path.read_bytes()
            except OSError:
                result.diagnostics.append(Diagnostic(
                    snapshot_id=context.snapshot.id,
                    code="source.unreadable_preprocessor_input",
                    severity=DiagnosticSeverity.ERROR,
                    message="a snapshotted textual compiler input could not be read",
                    stage=Stage.SOURCE.value,
                    artifact_id=artifact.id,
                ))
                return result
            try:
                if b"\x00" in source_bytes:
                    raise UnicodeError("NUL byte")
                source_text = source_bytes.decode("utf-8-sig")
            except UnicodeError:
                result.diagnostics.append(Diagnostic(
                    snapshot_id=context.snapshot.id,
                    code="source.unsupported_text_encoding",
                    severity=DiagnosticSeverity.ERROR,
                    message=("a snapshotted compiler-reachable source input is not "
                             "NUL-free UTF-8"),
                    stage=Stage.SOURCE.value,
                    artifact_id=artifact.id,
                ))
                return result
            compiler_text[artifact.id] = source_text
            unsafe_tokens.update(_unsafe_preprocessor_tokens(source_text))
        if unsafe_tokens:
            result.diagnostics.append(Diagnostic(
                snapshot_id=context.snapshot.id,
                code="source.unsupported_preprocessor_input",
                severity=DiagnosticSeverity.ERROR,
                message=("source uses preprocessing features whose implicit file/time/path "
                         "inputs are not represented by the v0.1 snapshot"),
                stage=Stage.SOURCE.value,
                metadata={"features": sorted(unsafe_tokens)},
            ))
            return result

        function_by_name: dict[str, list[str]] = defaultdict(list)
        pending_calls: list[tuple[str, str, SourceAnchor | None]] = []
        cursor_entity: dict[int, str] = {}
        untracked_project_inputs: set[str] = set()
        untracked_external_inputs = 0
        pragma_activity: dict[tuple[str, int], set[bool | None]] = defaultdict(set)
        pragma_activity_available = True
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
            unsafe_arguments: set[str] = set()
            for argument in args:
                # Scan argv independently: joining permits one value ending in
                # ``//`` or an unmatched literal to lexically hide the next.
                unsafe_arguments.update(_unsafe_preprocessor_tokens(argument))
            if unsafe_arguments:
                result.diagnostics.append(Diagnostic(
                    snapshot_id=context.snapshot.id,
                    code="source.unsupported_preprocessor_input",
                    severity=DiagnosticSeverity.ERROR,
                    message=("compiler arguments use preprocessing features whose implicit "
                             "inputs are not represented by the v0.1 snapshot"),
                    stage=Stage.AST.value,
                    metadata={"features": sorted(unsafe_arguments)},
                ))
                continue
            if any(arg == "-ivfsoverlay" or arg.startswith("-ivfsoverlay=")
                   for arg in args):
                result.diagnostics.append(Diagnostic(
                    snapshot_id=context.snapshot.id,
                    code="source.unsupported_vfs_overlay",
                    severity=DiagnosticSeverity.ERROR,
                    message=("VFS overlay compilation contexts are not supported by the "
                             "standard extractor because backing-file attribution is incomplete"),
                    stage=Stage.AST.value,
                ))
                continue
            try:
                tu = index.parse(str(source), args=args,
                                 options=cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD)
            except Exception as exc:
                raise ExtractionError(f"failed to parse {unit.file}: {exc}") from exc
            # The manifest scanner intentionally remains compiler-independent,
            # but macro includes, response files, or compiler-specific flags can
            # expand the real dependency set.  Never accept an AST whose
            # project-local input bytes were absent from the snapshot.
            inclusions = list(tu.get_includes())
            for inclusion in inclusions:
                inclusion_path = str(inclusion.include)
                relative = _relative_file(context, inclusion_path)
                if relative and context.artifact_for_uri(relative) is None:
                    untracked_project_inputs.add(relative)
                elif relative is None and Path(inclusion_path).is_absolute():
                    untracked_external_inputs += 1
            processed_paths = {source.resolve()}
            processed_paths.update(
                Path(str(inclusion.include)).resolve() for inclusion in inclusions
            )
            for processed_path in sorted(processed_paths, key=str):
                relative = _relative_file(context, str(processed_path))
                artifact = context.artifact_for_uri(relative) if relative else None
                if artifact is None:
                    continue
                # Static include discovery intentionally does not try to expand
                # macros.  Libclang's actual inclusion set is authoritative for
                # reachability, so a tracked macro-expanded include must be
                # scanned even if an explicit arbitrary kind/role kept it out
                # of the conservative pre-scan set.
                if artifact.id not in compiler_text:
                    path = project_path(context.project_root, artifact.uri)
                    try:
                        source_bytes = path.read_bytes()
                    except OSError:
                        graph.entities.clear()
                        graph.relations.clear()
                        result.observations.clear()
                        result.diagnostics.append(Diagnostic(
                            snapshot_id=context.snapshot.id,
                            code="source.unreadable_preprocessor_input",
                            severity=DiagnosticSeverity.ERROR,
                            message="a compiler-processed textual input could not be read",
                            stage=Stage.SOURCE.value,
                            artifact_id=artifact.id,
                        ))
                        return result
                    try:
                        if b"\x00" in source_bytes:
                            raise UnicodeError("NUL byte")
                        source_text = source_bytes.decode("utf-8-sig")
                    except UnicodeError:
                        graph.entities.clear()
                        graph.relations.clear()
                        result.observations.clear()
                        result.diagnostics.append(Diagnostic(
                            snapshot_id=context.snapshot.id,
                            code="source.unsupported_text_encoding",
                            severity=DiagnosticSeverity.ERROR,
                            message=("a compiler-processed source input is not "
                                     "NUL-free UTF-8"),
                            stage=Stage.SOURCE.value,
                            artifact_id=artifact.id,
                        ))
                        return result
                    actual_unsafe_tokens = _unsafe_preprocessor_tokens(source_text)
                    if actual_unsafe_tokens:
                        graph.entities.clear()
                        graph.relations.clear()
                        result.observations.clear()
                        result.diagnostics.append(Diagnostic(
                            snapshot_id=context.snapshot.id,
                            code="source.unsupported_preprocessor_input",
                            severity=DiagnosticSeverity.ERROR,
                            message=("compiler-processed source uses preprocessing features "
                                     "whose implicit file/time/path inputs are not represented "
                                     "by the v0.1 snapshot"),
                            stage=Stage.SOURCE.value,
                            artifact_id=artifact.id,
                            metadata={"features": sorted(actual_unsafe_tokens)},
                        ))
                        return result
                    compiler_text[artifact.id] = source_text
                try:
                    skipped_ranges = _skipped_preprocessor_line_ranges(
                        cindex, tu, processed_path,
                    )
                except Exception as exc:
                    pragma_activity_available = False
                    result.diagnostics.append(Diagnostic(
                        snapshot_id=context.snapshot.id,
                        code="source.preprocessor_activity_unavailable",
                        severity=DiagnosticSeverity.ERROR,
                        message=("libclang could not provide inactive preprocessor ranges; "
                                 "source pragmas were not promoted to directive facts"),
                        stage=Stage.AST.value,
                        artifact_id=artifact.id,
                        metadata={"error_type": type(exc).__name__, "tu": unit.file},
                    ))
                    continue
                for line_number, line in enumerate(
                    compiler_text[artifact.id].splitlines(), 1,
                ):
                    if not _PRAGMA.match(line):
                        continue
                    inactive = any(
                        start <= line_number <= end for start, end in skipped_ranges
                    )
                    pragma_activity[(artifact.id, line_number)].add(not inactive)
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
                nonlocal untracked_external_inputs
                location_file = str(cursor.location.file) if cursor.location.file else None
                relative = _relative_file(context, location_file)
                kind_name = cursor.kind.name
                if relative is None and kind_name != "TRANSLATION_UNIT":
                    if location_file and Path(location_file).is_absolute():
                        untracked_external_inputs += 1
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
                    if kind_name == "FOR_STMT":
                        attrs.update(_constant_for_loop_facts(cursor))
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

        if untracked_external_inputs:
            result.diagnostics.append(Diagnostic(
                snapshot_id=context.snapshot.id,
                code="source.untracked_external_include",
                severity=DiagnosticSeverity.ERROR,
                message=("libclang read source outside the project-local snapshot; mirror "
                         "licensed dependencies into the project or use project-local stubs"),
                stage=Stage.AST.value,
                metadata={"occurrences": untracked_external_inputs},
            ))

        call_diagnostics: set[tuple[str, str, str]] = set()
        for owner, callee, anchor in pending_calls:
            targets = sorted(set(function_by_name.get(callee, [])))
            if len(targets) == 1:
                graph.add_relation(Relation(
                    src=owner, dst=targets[0], kind="software.calls", snapshot_id=context.snapshot.id,
                    authority=AuthorityClass.STATIC_FACT, stage=Stage.AST.value,
                    anchors=[anchor] if anchor else [],
                    attrs={"hardware_instance": False, "ml_input_evidence": True},
                ))
            elif len(targets) > 1:
                diagnostic_key = (owner, callee, "ambiguous")
                if diagnostic_key in call_diagnostics:
                    continue
                call_diagnostics.add(diagnostic_key)
                result.diagnostics.append(Diagnostic(
                    snapshot_id=context.snapshot.id, code="mapping.ambiguous_call",
                    severity=DiagnosticSeverity.WARNING,
                    message=f"call to {callee!r} has {len(targets)} project-local candidates; no edge was guessed",
                    stage=Stage.AST.value, subject_id=owner,
                ))
            else:
                diagnostic_key = (owner, callee, "unresolved")
                if diagnostic_key in call_diagnostics:
                    continue
                call_diagnostics.add(diagnostic_key)
                result.diagnostics.append(Diagnostic(
                    snapshot_id=context.snapshot.id, code="mapping.unresolved_call",
                    severity=DiagnosticSeverity.INFO,
                    message=(f"call to {callee!r} has no project-local function target; "
                             "no software call edge was created"),
                    stage=Stage.AST.value, subject_id=owner,
                    artifact_id=anchor.artifact_id if anchor else None,
                    anchor=anchor,
                ))

        self._attach_source_pragmas(
            context, result,
            pragma_activity if pragma_activity_available else None,
        )
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
    def _attach_source_pragmas(
        context: ExtractionContext,
        result: ExtractionResult,
        activity: dict[tuple[str, int], set[bool | None]] | None,
        *,
        activity_mode: str = "libclang",
    ) -> None:
        graph = result.graph
        for artifact in sorted(context.artifacts.values(), key=lambda item: item.uri):
            if (not artifact.metadata.get("hlsgraph.compiler_reachable_text")
                    and not artifact.kind.startswith("source.")):
                continue
            path = project_path(context.project_root, artifact.uri)
            if not path.is_file():
                continue
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            for line_number, line in enumerate(lines, 1):
                match = _PRAGMA.match(line)
                if not match:
                    continue
                anchor = SourceAnchor(artifact_id=artifact.id, start_line=line_number,
                                      start_column=max(1, line.find("#") + 1),
                                      end_line=line_number, end_column=len(line) + 1)
                if activity is None:
                    # A standard extraction already emitted an error explaining
                    # why activity was unavailable.  Withhold every pragma fact.
                    continue
                states = activity.get((artifact.id, line_number), set())
                if states == {False}:
                    result.diagnostics.append(Diagnostic(
                        snapshot_id=context.snapshot.id,
                        code="directive.inactive_source_pragma",
                        severity=DiagnosticSeverity.INFO,
                        message=("source pragma is in an inactive preprocessing branch "
                                 "and was not promoted to a directive fact"),
                        stage=Stage.SOURCE.value,
                        artifact_id=artifact.id,
                        anchor=anchor,
                        metadata={"activity_mode": activity_mode},
                    ))
                    continue
                if None in states:
                    result.diagnostics.append(Diagnostic(
                        snapshot_id=context.snapshot.id,
                        code="directive.preprocessor_activity_unknown",
                        severity=DiagnosticSeverity.WARNING,
                        message=("source pragma activity could not be proven by the "
                                 f"{activity_mode} extractor and was withheld"),
                        stage=Stage.SOURCE.value,
                        artifact_id=artifact.id,
                        anchor=anchor,
                        metadata={"activity_mode": activity_mode},
                    ))
                    continue
                if states == {False, True}:
                    result.diagnostics.append(Diagnostic(
                        snapshot_id=context.snapshot.id,
                        code="directive.translation_unit_dependent_activity",
                        severity=DiagnosticSeverity.ERROR,
                        message=("source pragma is active in only part of the compilation "
                                 "context; v0.1 cannot represent it as one unconditional fact"),
                        stage=Stage.SOURCE.value,
                        artifact_id=artifact.id,
                        anchor=anchor,
                    ))
                    continue
                if states != {True}:
                    # The artifact closure deliberately over-approximates
                    # same-name include candidates.  An unprocessed candidate
                    # contributes bytes to identity but no directive fact.
                    continue
                directive_kind = match.group(1).upper()
                # Comments are source prose, not pragma semantics.  Keeping
                # them here would leak private source through graph/REST/MCP/ML
                # exports and could also invent bogus directive flags.
                options = _tokens_to_options(_strip_inline_comments(match.group(2)))
                target = _scope_for_pragma(
                    graph, artifact.id, line_number, directive_kind, options
                )
                operand_target = (
                    resolve_directive_variable_operand(
                        graph, target, options.get("variable"),
                    )
                    if directive_kind == "DEPENDENCE" else None
                )
                identity_complete = bool(
                    target is not None
                    and (directive_kind != "DEPENDENCE" or operand_target is not None)
                )
                directive = Entity(
                    kind="hls.directive", name=directive_kind,
                    qualified_name=f"{artifact.uri}:{line_number}:{directive_kind}",
                    snapshot_id=context.snapshot.id,
                    authority=AuthorityClass.DECLARED_CONSTRAINT, stage=Stage.SOURCE.value,
                    attrs={"directive_kind": directive_kind, "options": options,
                           "origin": "source_pragma", "precedence": 10,
                           "state": "requested"}, anchors=[anchor],
                    completeness=(Completeness.COMPLETE if identity_complete
                                  else Completeness.AMBIGUOUS),
                )
                scope_resolution = (
                    "source_ast" if activity_mode == "libclang" else "regex_degraded"
                )
                bind_directive_identity(
                    directive, target, scope_resolution=scope_resolution if target else None,
                    operand_target=operand_target,
                )
                graph.add_entity(directive)
                if target:
                    graph.add_relation(Relation(
                        src=directive.id, dst=target.id, kind="hls.annotates",
                        snapshot_id=context.snapshot.id,
                        authority=AuthorityClass.DECLARED_CONSTRAINT, stage=Stage.SOURCE.value,
                        attrs={"scope_node_id": target.id,
                               "scope_resolution": scope_resolution},
                        anchors=[anchor],
                    ))
                    result.observations.append(Observation(
                        snapshot_id=context.snapshot.id, subject_id=directive.id,
                        predicate="directive.requested", value=options or True,
                        stage=Stage.SOURCE.value, authority=AuthorityClass.DECLARED_CONSTRAINT,
                        artifact_id=artifact.id, anchor=anchor,
                        completeness=directive.completeness,
                        metadata={
                            "directive_kind": directive_kind,
                            **directive_identity_metadata(directive),
                        },
                    ))
                else:
                    result.diagnostics.append(Diagnostic(
                        snapshot_id=context.snapshot.id, code="directive.unresolved_scope",
                        severity=DiagnosticSeverity.WARNING,
                        message=f"could not deterministically bind {directive_kind} at {artifact.uri}:{line_number}",
                        stage=Stage.SOURCE.value, subject_id=directive.id,
                        artifact_id=artifact.id, anchor=anchor,
                    ))
                if (target is not None and directive_kind == "DEPENDENCE"
                        and operand_target is None):
                    result.diagnostics.append(Diagnostic(
                        snapshot_id=context.snapshot.id,
                        code="directive.unresolved_operand",
                        severity=DiagnosticSeverity.WARNING,
                        message=("could not deterministically bind DEPENDENCE operand "
                                 f"{options.get('variable')!r} at "
                                 f"{artifact.uri}:{line_number}"),
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
    if variable and kind != "DEPENDENCE":
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
    if kind == "DEPENDENCE":
        # DEPENDENCE names an operand but applies to the loop/function that
        # lexically encloses the declaration.  Unlike a loop pragma placed
        # immediately before a loop, a nearby following loop is not sufficient
        # evidence for this two-identity contract.
        loops = [entity for entity in containing if entity.kind == "hls.loop"]
        if loops:
            spans = {entity.id: min(
                (anchor.end_line or line) - (anchor.start_line or line)
                for anchor in entity.anchors
            ) for entity in loops}
            smallest = min(spans.values())
            winners = [entity for entity in loops if spans[entity.id] == smallest]
            return winners[0] if len(winners) == 1 else None
    loop_directives = {"PIPELINE", "UNROLL", "LOOP_FLATTEN", "LOOP_TRIPCOUNT"}
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
        degraded_activity: dict[tuple[str, int], set[bool | None]] = defaultdict(set)
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
            for location, state in _degraded_pragma_activity(artifact.id, text).items():
                degraded_activity[location].add(state)
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
        LibClangExtractor._attach_source_pragmas(
            context, result, degraded_activity, activity_mode="regex_degraded",
        )
        result.coverage = {"fidelity": "regex_degraded", "entities": len(graph.entities),
                           "relations": len(graph.relations)}
        return result
