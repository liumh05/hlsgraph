"""External Tcl/config directive extraction and declared-precedence resolution."""
from __future__ import annotations

import re
import shlex
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

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
    stable_hash,
)
from .base import ExtractionContext, ExtractionResult
from .directive_identity import (
    bind_directive_identity,
    directive_identity_metadata,
)


_TCL_DIRECTIVE = re.compile(r"^\s*set_directive_([A-Za-z0-9_]+)\s+(.+?)\s*$")
_TCL_DIRECTIVE_MARKER = re.compile(r"\bset_directive_[A-Za-z0-9_]*")
_CONFIG_DIRECTIVE = re.compile(r"^\s*(?:syn\.)?directive\.([A-Za-z0-9_]+)\s*=\s*(.*?)\s*$", re.I)


def _strip_comment(text: str) -> str:
    """Drop external-file comments before storing directive semantics."""
    quote: str | None = None
    brace_depth = 0
    escaped = False
    for index, char in enumerate(text):
        following = text[index + 1] if index + 1 < len(text) else ""
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
        elif char == "{":
            brace_depth += 1
        elif char == "}" and brace_depth:
            brace_depth -= 1
        elif not brace_depth and char == "/" and following == "/":
            return text[:index].rstrip()
        elif not brace_depth and char == "#" and (
                index == 0 or text[index - 1].isspace() or text[index - 1] == ";"):
            return text[:index].rstrip(" ;")
    return text.rstrip()


def _config_literal_tokens(text: str) -> tuple[list[str] | None, str | None]:
    """Tokenize one config directive without importing Tcl word semantics."""

    # Config values use whitespace words and optional double quotes.  Braces,
    # Tcl substitutions, escapes, semicolons, and single quotes are not
    # normalized into a different spelling: unsupported syntax is withheld.
    if any(char in text for char in "{}\\$[];'"):
        return None, "unsupported_config_literal_syntax"
    try:
        return shlex.split(text, comments=False, posix=True), None
    except ValueError:
        return None, "malformed_literal_words"


def _tcl_literal_tokens(text: str) -> tuple[list[str] | None, str | None]:
    """Parse a conservative subset of Tcl literal words without evaluation.

    Only bare, double-quoted, or one-level braced words are accepted.  The
    lexical top-level guard already rejects substitution and escapes; those
    characters are rejected again here so this parser is safe when called in
    isolation.  Crucially, braces are never globally rewritten or stripped.
    """

    if any(char in text for char in "\\$[];"):
        return None, "dynamic_or_escaped_tcl_word"
    tokens: list[str] = []
    index = 0
    while index < len(text):
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text):
            break
        start = text[index]
        if start == '"':
            end = text.find('"', index + 1)
            if end < 0:
                return None, "malformed_literal_words"
            token = text[index + 1:end]
            index = end + 1
            if index < len(text) and not text[index].isspace():
                return None, "concatenated_tcl_word"
        elif start == "{":
            end = text.find("}", index + 1)
            if end < 0 or "{" in text[index + 1:end]:
                return None, "nested_or_malformed_tcl_brace_word"
            token = text[index + 1:end]
            index = end + 1
            if index < len(text) and not text[index].isspace():
                return None, "concatenated_tcl_word"
        else:
            end = index
            while end < len(text) and not text[end].isspace():
                end += 1
            token = text[index:end]
            index = end
            if any(char in token for char in "{}\"'"):
                return None, "unsupported_tcl_literal_word"
        tokens.append(token)
    return tokens, None


def _advance_tcl_lexical_state(
    text: str, state: tuple[str, ...] = (),
) -> tuple[tuple[str, ...], bool]:
    """Advance a conservative Tcl ``brace/quote/bracket`` context stack.

    The stack is intentionally lexical rather than evaluative.  An open frame
    means the next physical line cannot be proven to start a top-level command.
    ``malformed`` records an unmatched closing delimiter and permanently moves
    the file into fail-closed mode.
    """
    frames = list(state)
    escaped = False
    malformed = False
    for char in text:
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        current = frames[-1] if frames else None
        if current == "brace":
            if char == "{":
                frames.append("brace")
            elif char == "}":
                frames.pop()
            continue
        if current == "quote":
            if char == '"':
                frames.pop()
            elif char == "[":
                # Command substitution is parsed as a nested Tcl script and
                # returns to the surrounding quote after its closing bracket.
                frames.append("bracket")
            continue

        # Top-level script text and bracket command substitutions share the
        # same word-level delimiter rules.
        if char == "{":
            frames.append("brace")
        elif char == '"':
            frames.append("quote")
        elif char == "[":
            frames.append("bracket")
        elif char == "]":
            if current == "bracket":
                frames.pop()
            else:
                malformed = True
        elif char == "}":
            malformed = True
    return tuple(frames), malformed


def _tcl_line_profile(text: str) -> dict[str, Any]:
    """Return the small lexical profile needed by the literal Tcl policy.

    This is deliberately not a Tcl interpreter.  Its only purpose is to prove
    that a candidate is one complete, top-level command made from literal
    words.  Anything requiring Tcl evaluation remains diagnostic-only.
    """
    brace_depth = 0
    minimum_brace_depth = 0
    quote = False
    escaped = False
    semicolon = False
    substitution = False
    unsupported_quote = False
    backslash = False
    for char in text:
        if escaped:
            escaped = False
            continue
        if char == "\\":
            backslash = True
            escaped = True
            continue
        if quote:
            if char == '"':
                quote = False
            elif char in {"$", "["}:
                substitution = True
            continue
        if char == '"' and brace_depth == 0:
            quote = True
        elif char == "{":
            brace_depth += 1
        elif char == "}":
            brace_depth -= 1
            minimum_brace_depth = min(minimum_brace_depth, brace_depth)
        elif brace_depth == 0 and char == ";":
            semicolon = True
        elif brace_depth == 0 and char in {"$", "["}:
            substitution = True
        elif brace_depth == 0 and char == "'":
            # A single quote has no grouping semantics in Tcl, while shlex
            # would interpret it as a quote.  Reject it rather than parse a
            # different command from the one Tcl would execute.
            unsupported_quote = True
    lexical_state, lexical_malformed = _advance_tcl_lexical_state(text)
    return {
        "brace_delta": brace_depth,
        "minimum_brace_depth": minimum_brace_depth,
        "quote_unclosed": quote,
        "semicolon": semicolon,
        "substitution": substitution,
        "unsupported_quote": unsupported_quote,
        "backslash": backslash,
        "lexical_open": bool(lexical_state),
        "lexical_malformed": lexical_malformed,
    }


def _tcl_literal_rejection(
    text: str, *, lexical_context_open: bool, continued_from_previous: bool,
    structure_uncertain: bool,
) -> str | None:
    """Explain why a Tcl directive cannot be asserted as a declaration."""
    if structure_uncertain:
        return "uncertain_script_structure"
    if continued_from_previous:
        return "continued_command"
    if lexical_context_open:
        return "nested_script_context"
    if not _TCL_DIRECTIVE.match(text):
        return "embedded_or_constructed_command"
    profile = _tcl_line_profile(text)
    if (profile["lexical_open"] or profile["lexical_malformed"]
            or profile["brace_delta"] != 0
            or profile["minimum_brace_depth"] < 0
            or profile["quote_unclosed"]):
        return "incomplete_command"
    if profile["semicolon"]:
        return "multiple_commands"
    if profile["substitution"] or re.search(r"(?:^|\s)\{\*\}", text):
        return "dynamic_substitution"
    if profile["backslash"]:
        return "escape_or_continuation"
    if profile["unsupported_quote"]:
        return "unsupported_quoting"
    return None


def _tcl_continues(text: str) -> bool:
    stripped = text.rstrip()
    trailing = len(stripped) - len(stripped.rstrip("\\"))
    return bool(trailing % 2)


@dataclass(frozen=True, slots=True)
class _OptionGrammar:
    """One AMD 2024.2 option with syntax specific to each external frontend."""

    value_kind: str = "string"
    tcl_arity: int = 1
    config_arity: int = 1
    choices: tuple[str, ...] = ()
    minimum: int | None = None
    maximum: int | None = None
    true_only: bool = False


@dataclass(frozen=True, slots=True)
class _DirectiveGrammar:
    positional_roles: tuple[str, ...]
    options: Mapping[str, _OptionGrammar]
    scope_kinds: frozenset[str]


@dataclass(frozen=True, slots=True)
class _ParsedDirective:
    kind: str
    options: dict[str, Any]
    positionals: tuple[str, ...]
    parse_policy: str

    @property
    def location(self) -> str:
        return self.positionals[0]

    @property
    def operand(self) -> str | None:
        return self.positionals[1] if len(self.positionals) > 1 else None

    @property
    def scope_text(self) -> str:
        return " ".join(self.positionals)


@dataclass(frozen=True, slots=True)
class _DirectiveParseFailure:
    reason: str
    option: str | None = None
    expected: int | None = None
    actual: int | None = None


def _flag(*, config_value: bool = False) -> _OptionGrammar:
    return _OptionGrammar(
        value_kind="bool" if config_value else "flag",
        tcl_arity=0,
        config_arity=1 if config_value else 0,
        true_only=config_value,
    )


def _integer(
    *, minimum: int | None = None, maximum: int | None = None,
) -> _OptionGrammar:
    return _OptionGrammar(
        value_kind="int", minimum=minimum, maximum=maximum,
    )


def _enum(*values: str) -> _OptionGrammar:
    return _OptionGrammar(value_kind="enum", choices=tuple(values))


_INTERFACE_MODES = (
    "ap_none", "ap_stable", "ap_vld", "ap_ack", "ap_hs", "ap_ovld",
    "ap_memory", "bram", "ap_fifo", "s_axilite", "m_axi", "axis",
    "ap_ctrl_chain", "ap_ctrl_hs", "ap_ctrl_none",
)
_INTERFACE_BLOCK_CONTROL_MODES = frozenset({
    "ap_ctrl_chain", "ap_ctrl_hs", "ap_ctrl_none",
})


# This is intentionally the complete set that the public v0.3 extractor claims
# for AMD Vitis HLS 2024.2.  A similarly named command outside this table is a
# diagnostic, never a best-effort directive declaration.
_DIRECTIVE_GRAMMARS: dict[str, _DirectiveGrammar] = {
    "DATAFLOW": _DirectiveGrammar(
        ("location",),
        {"disable_start_propagation": _flag()},
        frozenset({"hls.kernel", "hls.function", "hls.loop"}),
    ),
    "PIPELINE": _DirectiveGrammar(
        ("location",),
        {
            "ii": _integer(minimum=1),
            "off": _flag(),
            "rewind": _flag(),
            "style": _enum("stp", "flp", "frp"),
        },
        frozenset({"hls.kernel", "hls.function", "hls.loop"}),
    ),
    "UNROLL": _DirectiveGrammar(
        ("location",),
        {
            "factor": _integer(minimum=1),
            "off": _flag(config_value=True),
            "skip_exit_check": _flag(),
        },
        frozenset({"hls.loop"}),
    ),
    "ARRAY_PARTITION": _DirectiveGrammar(
        ("location", "array"),
        {
            "dim": _integer(minimum=0),
            "factor": _integer(minimum=1),
            "off": _flag(config_value=True),
            "type": _enum("block", "cyclic", "complete"),
        },
        frozenset({"hls.kernel", "hls.function", "hls.loop"}),
    ),
    "INTERFACE": _DirectiveGrammar(
        ("location", "port"),
        {
            "bundle": _OptionGrammar(),
            "channel": _OptionGrammar(),
            "clock": _OptionGrammar(),
            "depth": _integer(minimum=1),
            "interrupt": _integer(minimum=16, maximum=31),
            "latency": _integer(minimum=0),
            "max_read_burst_length": _integer(minimum=1),
            "max_widen_bitwidth": _integer(minimum=0),
            "max_write_burst_length": _integer(minimum=1),
            "mode": _enum(*_INTERFACE_MODES),
            "name": _OptionGrammar(),
            "num_read_outstanding": _integer(minimum=1),
            "num_write_outstanding": _integer(minimum=1),
            "offset": _OptionGrammar(),
            "register": _flag(),
            "register_mode": _enum("both", "forward", "reverse", "off"),
            "storage_impl": _enum("auto", "bram", "uram"),
            "storage_type": _enum(
                "ram_1p", "ram_1wnr", "ram_2p", "ram_s2p", "ram_t2p",
                "rom_1p", "rom_2p", "rom_np",
            ),
        },
        frozenset({"hls.kernel"}),
    ),
    "STREAM": _DirectiveGrammar(
        ("location", "variable"),
        {
            "depth": _integer(minimum=1),
            "type": _enum("fifo", "pipo", "shared", "unsync"),
        },
        frozenset({"hls.kernel", "hls.function", "hls.loop"}),
    ),
    "DEPENDENCE": _DirectiveGrammar(
        ("location",),
        {
            "class": _enum("array", "pointer"),
            "dependent": _OptionGrammar(value_kind="bool"),
            "direction": _enum("raw", "war", "waw"),
            "distance": _integer(minimum=1),
            "type": _enum("intra", "inter"),
            "variable": _OptionGrammar(),
        },
        frozenset({"hls.kernel", "hls.function", "hls.loop"}),
    ),
    "LOOP_TRIPCOUNT": _DirectiveGrammar(
        ("location",),
        {
            "avg": _integer(minimum=0),
            "max": _integer(minimum=0),
            "min": _integer(minimum=0),
        },
        frozenset({"hls.loop"}),
    ),
    "INLINE": _DirectiveGrammar(
        ("location",),
        {"off": _flag(), "recursive": _flag()},
        frozenset({"hls.kernel", "hls.function"}),
    ),
}


def _coerce_option_value(
    raw_value: str, grammar: _OptionGrammar,
) -> tuple[Any | None, str | None]:
    raw_value = raw_value.strip()
    if not raw_value:
        return None, "empty_option_value"
    if grammar.value_kind == "int":
        # The supported AMD directive options use non-negative decimal
        # integers.  Python-only spellings such as ``1_024`` and implicit
        # base prefixes must not be accepted as vendor syntax.
        if re.fullmatch(r"[0-9]+", raw_value) is None:
            return None, "invalid_integer"
        value = int(raw_value, 10)
        if grammar.minimum is not None and value < grammar.minimum:
            return None, "integer_out_of_range"
        if grammar.maximum is not None and value > grammar.maximum:
            return None, "integer_out_of_range"
        return value, None
    if grammar.value_kind == "bool":
        lowered = raw_value.casefold()
        if lowered not in {"true", "false"}:
            return None, "invalid_boolean"
        value = lowered == "true"
        if grammar.true_only and not value:
            return None, "unsupported_false_disable"
        return value, None
    if grammar.value_kind == "enum":
        lowered = raw_value.casefold()
        if lowered not in grammar.choices:
            return None, "invalid_enum"
        return lowered, None
    return raw_value, None


def _parse_tcl_directive(
    kind: str, tokens: list[str], grammar: _DirectiveGrammar,
) -> tuple[dict[str, Any] | None, list[str] | None, _DirectiveParseFailure | None]:
    options: dict[str, Any] = {}
    positional: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("-"):
            # The literal Tcl tokenizer has already removed exactly one outer
            # grouping layer.  Rewriting the remaining spelling would turn a
            # genuinely different location such as ``"{dut}"`` into ``dut``.
            positional.append(token)
            index += 1
            continue
        key = token[1:].casefold()
        option_grammar = grammar.options.get(key)
        if not key or option_grammar is None:
            return None, None, _DirectiveParseFailure("unknown_option", key or None)
        if key in options:
            return None, None, _DirectiveParseFailure("duplicate_option", key)
        if option_grammar.tcl_arity == 0:
            options[key] = True
            index += 1
            continue
        if index + 1 >= len(tokens):
            return None, None, _DirectiveParseFailure("missing_option_value", key)
        raw_value = tokens[index + 1]
        # Tcl options are separate words.  A following option token is never
        # a string value for the current option: accepting ``-bundle
        # -register`` as bundle="-register" would promote an invalid command
        # into a requested directive.
        if raw_value.startswith("-"):
            return None, None, _DirectiveParseFailure(
                "missing_option_value", key,
            )
        value, error = _coerce_option_value(raw_value, option_grammar)
        if error:
            return None, None, _DirectiveParseFailure(error, key)
        options[key] = value
        index += 2
    return options, positional, None


def _parse_config_directive(
    kind: str, text: str, tokens: list[str], grammar: _DirectiveGrammar,
) -> tuple[dict[str, Any] | None, list[str] | None, _DirectiveParseFailure | None]:
    if "," in text:
        return None, None, _DirectiveParseFailure("legacy_comma_syntax")
    options: dict[str, Any] = {}
    positional: list[str] = []
    for token in tokens:
        if token.startswith("-"):
            return None, None, _DirectiveParseFailure("tcl_option_in_config")
        if "=" in token:
            key, raw_value = token.split("=", 1)
            key = key.casefold()
            option_grammar = grammar.options.get(key)
            if not key or option_grammar is None:
                return None, None, _DirectiveParseFailure("unknown_option", key or None)
            if key in options:
                return None, None, _DirectiveParseFailure("duplicate_option", key)
            if option_grammar.config_arity != 1:
                return None, None, _DirectiveParseFailure("flag_has_value", key)
            value, error = _coerce_option_value(raw_value, option_grammar)
            if error:
                return None, None, _DirectiveParseFailure(error, key)
            options[key] = value
            continue
        key = token.casefold()
        option_grammar = grammar.options.get(key)
        if option_grammar is not None:
            if key in options:
                return None, None, _DirectiveParseFailure("duplicate_option", key)
            if option_grammar.config_arity != 0:
                return None, None, _DirectiveParseFailure("missing_option_value", key)
            options[key] = True
            continue
        # Config tokenization owns quoting semantics.  Do not perform a second
        # normalization pass that could change the declared scope spelling.
        positional.append(token)
    return options, positional, None


def _validate_directive_shape(
    kind: str, options: Mapping[str, Any], positional: list[str],
    grammar: _DirectiveGrammar,
) -> _DirectiveParseFailure | None:
    expected = len(grammar.positional_roles)
    mode = str(options.get("mode", "")).casefold()
    block_control = kind == "INTERFACE" and mode in _INTERFACE_BLOCK_CONTROL_MODES
    allowed_counts = {expected}
    if block_control:
        # AMD permits omitting the return/control port spelling.  The current
        # graph has no deterministic return/control-port entity, so this shape
        # is parsed but deliberately kept unsupported below.
        allowed_counts.add(1)
    if len(positional) not in allowed_counts or any(not item for item in positional):
        return _DirectiveParseFailure(
            "wrong_positional_arity", expected=expected, actual=len(positional),
        )
    if kind == "DEPENDENCE":
        has_class = "class" in options
        has_variable = "variable" in options
        if has_class == has_variable:
            return _DirectiveParseFailure("dependence_operand_not_exclusive")
    if kind == "ARRAY_PARTITION" and options.get("off") is True:
        if {"dim", "factor", "type"}.intersection(options):
            return _DirectiveParseFailure("array_partition_off_conflict", "off")
    if kind == "PIPELINE" and options.get("off") is True:
        if {"ii", "rewind", "style"}.intersection(options):
            return _DirectiveParseFailure("pipeline_off_conflict", "off")
    if kind == "UNROLL" and options.get("off") is True:
        if {"factor", "skip_exit_check"}.intersection(options):
            return _DirectiveParseFailure("unroll_off_conflict", "off")
    if kind == "INLINE" and options.get("off") is True:
        if options.get("recursive") is True:
            return _DirectiveParseFailure("inline_off_conflict", "off")
    if kind == "LOOP_TRIPCOUNT" and not options:
        return _DirectiveParseFailure("missing_required_option")
    if kind == "LOOP_TRIPCOUNT":
        minimum = options.get("min")
        average = options.get("avg")
        maximum = options.get("max")
        if (minimum is not None and maximum is not None and minimum > maximum) or (
            minimum is not None and average is not None and minimum > average
        ) or (
            average is not None and maximum is not None and average > maximum
        ):
            return _DirectiveParseFailure("tripcount_order_invalid")
    if kind == "INTERFACE":
        bundle = options.get("bundle")
        if isinstance(bundle, str) and bundle != bundle.casefold():
            return _DirectiveParseFailure("interface_bundle_not_lowercase", "bundle")
    return None


def _parse_directive(
    *, kind: str, text: str, origin: str,
) -> tuple[_ParsedDirective | None, _DirectiveParseFailure | None]:
    grammar = _DIRECTIVE_GRAMMARS.get(kind)
    if grammar is None:
        return None, _DirectiveParseFailure("unsupported_directive_kind")
    tokens, token_error = (
        _tcl_literal_tokens(text)
        if origin == "tcl" else _config_literal_tokens(text)
    )
    if token_error or tokens is None:
        return None, _DirectiveParseFailure(token_error or "tokenization_failed")
    if origin == "tcl":
        options, positional, failure = _parse_tcl_directive(kind, tokens, grammar)
        parse_policy = "hlsgraph.amd_2024_2_tcl_literal_strict_v1"
    else:
        options, positional, failure = _parse_config_directive(
            kind, text, tokens, grammar,
        )
        parse_policy = "hlsgraph.amd_2024_2_config_whitespace_strict_v1"
    if failure or options is None or positional is None:
        return None, failure or _DirectiveParseFailure("parse_failed")
    failure = _validate_directive_shape(kind, options, positional, grammar)
    if failure:
        return None, failure
    return _ParsedDirective(
        kind=kind,
        options=options,
        positionals=tuple(positional),
        parse_policy=parse_policy,
    ), None


def _normalize_scope(value: str) -> str:
    """Return an exact external scope spelling or a fail-closed sentinel.

    Separators and surrounding characters are part of the AMD location.  In
    particular, ``/dut/`` must not be silently rewritten to ``dut``.  Tcl and
    config tokenizers reject escape syntax before this point, so no path-style
    backslash normalization is appropriate here either.
    """

    if not value or value != value.strip() or value.startswith("/") or value.endswith("/"):
        return ""
    return value


def _static_ast_entity(entity: Entity, snapshot_id: str) -> bool:
    return bool(
        entity.snapshot_id == snapshot_id
        and entity.stage == Stage.AST.value
        and str(entity.authority) == "static_fact"
        and str(entity.completeness) == "complete"
    )


def _static_ast_containment(relation: Relation, snapshot_id: str) -> bool:
    return bool(
        relation.snapshot_id == snapshot_id
        and relation.stage == Stage.AST.value
        and str(relation.authority) == "static_fact"
        and str(relation.completeness) == "complete"
    )


def _ownership_lineage(
    graph: CanonicalGraph, entity: Entity,
) -> tuple[Entity, tuple[str, ...]] | None:
    """Return one proven lexical lineage ending at its nearest function owner."""
    if not _static_ast_entity(entity, graph.snapshot_id):
        return None
    current = entity
    lineage = [current.id]
    visited = {current.id}
    while current.kind not in {"hls.kernel", "hls.function"}:
        incoming = [
            relation for relation in graph.relations.values()
            if relation.kind == "hls.contains"
            and relation.dst == current.id
        ]
        # Ownership is identity evidence.  Missing, multiple, partial, stale,
        # or non-AST containment must therefore fail closed.
        if len(incoming) != 1:
            return None
        relation = incoming[0]
        parent = graph.entities.get(relation.src)
        if (
            parent is None
            or not _static_ast_containment(relation, graph.snapshot_id)
            or not _static_ast_entity(parent, graph.snapshot_id)
            or parent.id in visited
        ):
            return None
        current = parent
        visited.add(current.id)
        lineage.append(current.id)
    return current, tuple(lineage)


def _function_like_owners(
    graph: CanonicalGraph, entity: Entity,
) -> list[Entity]:
    resolved = _ownership_lineage(graph, entity)
    return [resolved[0]] if resolved is not None else []


def _scope_spellings(graph: CanonicalGraph, entity: Entity) -> set[str]:
    if entity.kind == "hls.loop":
        owners = _function_like_owners(graph, entity)
        if len(owners) != 1:
            return set()
        # AMD external locations use function[/label].  A bare label or a
        # libclang qualified-name suffix is not an external scope identity.
        values = {f"{owners[0].name}/{entity.name}"}
    else:
        values = {entity.name, entity.qualified_name or "", *entity.aliases}
    return {
        _normalize_scope(value.replace("::", "/"))
        for value in values if value
    }


def _resolve_location(
    graph: CanonicalGraph, location: str, *, allowed_kinds: frozenset[str],
) -> Entity | None:
    normalized = _normalize_scope(location)
    if not normalized:
        return None
    matches = [
        entity for entity in graph.entities.values()
        if entity.kind in allowed_kinds
        and _static_ast_entity(entity, graph.snapshot_id)
        and (
            entity.kind != "hls.loop"
            or _ownership_lineage(graph, entity) is not None
        )
        and normalized in _scope_spellings(graph, entity)
    ]
    return matches[0] if len(matches) == 1 else None


def _resolve_external_variable_operand(
    graph: CanonicalGraph,
    scope: Entity | None,
    variable_name: Any,
) -> Entity | None:
    """Resolve an exact operand using one complete static ownership chain."""
    if scope is None or not isinstance(variable_name, str) or not variable_name:
        return None
    scope_lineage = _ownership_lineage(graph, scope)
    if scope_lineage is None:
        return None
    scope_owner, scope_ids = scope_lineage
    allowed_kinds = {"hls.memory", "hls.stream", "hls.port", "source.variable"}
    matches: list[Entity] = []
    for candidate in graph.entities.values():
        if (
            candidate.kind not in allowed_kinds
            or candidate.name != variable_name
            or not _static_ast_entity(candidate, graph.snapshot_id)
        ):
            continue
        candidate_lineage = _ownership_lineage(graph, candidate)
        if candidate_lineage is None:
            continue
        candidate_owner, candidate_ids = candidate_lineage
        if candidate_owner.id != scope_owner.id:
            continue
        if scope.kind in {"hls.kernel", "hls.function"}:
            matches.append(candidate)
            continue
        candidate_parent = candidate_ids[1] if len(candidate_ids) > 1 else None
        visible_from_scope = bool(
            candidate_parent == scope_owner.id
            or candidate_parent in set(scope_ids[:-1])
            or scope.id in candidate_ids[1:]
        )
        if visible_from_scope:
            matches.append(candidate)
    return matches[0] if len(matches) == 1 else None


def _resolve_interface_scope(
    graph: CanonicalGraph, location: str, port_name: str, *,
    component_name: str,
) -> Entity | None:
    normalized_location = _normalize_scope(location)
    kernels = [
        item for item in graph.entities.values()
        if item.kind == "hls.kernel"
        and _static_ast_entity(item, graph.snapshot_id)
        and item.name == component_name
    ]
    if len(kernels) != 1:
        return None
    kernel = kernels[0]
    # The external grammar names the configured top function.  A canonical
    # entity ID is an internal storage identity, not an AMD location spelling,
    # and accepting it would let a caller bypass the source-level scope proof.
    if normalized_location != _normalize_scope(component_name):
        return None
    candidates: list[Entity] = []
    for port in graph.entities.values():
        if (port.kind != "hls.port" or port.name != port_name
                or not _static_ast_entity(port, graph.snapshot_id)):
            continue
        owners = [
            relation for relation in graph.relations.values()
            if relation.kind == "hls.contains"
            and relation.dst == port.id
        ]
        if (
            len(owners) == 1
            and owners[0].src == kernel.id
            and owners[0].snapshot_id == graph.snapshot_id
            and owners[0].stage == Stage.AST.value
            and str(owners[0].authority) == "static_fact"
            and str(owners[0].completeness) == "complete"
        ):
            candidates.append(port)
    return candidates[0] if len(candidates) == 1 else None


def _resolve_parsed_scope(
    graph: CanonicalGraph, parsed: _ParsedDirective, *, component_name: str,
) -> tuple[Entity | None, Entity | None]:
    grammar = _DIRECTIVE_GRAMMARS[parsed.kind]
    if parsed.kind == "INTERFACE":
        if str(parsed.options.get("mode", "")).casefold() in (
            _INTERFACE_BLOCK_CONTROL_MODES
        ):
            return None, None
        assert parsed.operand is not None
        port = _resolve_interface_scope(
            graph, parsed.location, parsed.operand,
            component_name=component_name,
        )
        return port, port
    location = _resolve_location(
        graph, parsed.location, allowed_kinds=grammar.scope_kinds,
    )
    if parsed.kind in {"ARRAY_PARTITION", "STREAM"}:
        assert parsed.operand is not None
        operand = _resolve_external_variable_operand(
            graph, location, parsed.operand,
        )
        return operand, operand
    if parsed.kind == "DEPENDENCE":
        operand = _resolve_external_variable_operand(
            graph, location, parsed.options.get("variable"),
        )
        return location, operand
    return location, None


class ExternalDirectiveExtractor:
    name = "directive.external"
    version = "3"

    def supports(self, context: ExtractionContext) -> bool:
        return bool(context.manifest.build.tcl_files or context.manifest.build.config_files)

    def extract(self, context: ExtractionContext) -> ExtractionResult:
        graph = CanonicalGraph(snapshot_id=context.snapshot.id)
        result = ExtractionResult(graph=graph, capabilities=["directive.external"])
        existing: CanonicalGraph = context.options.get("existing_graph") or graph
        sources = [(path, "tcl", 30) for path in context.manifest.build.tcl_files]
        sources += [(path, "config", 20) for path in context.manifest.build.config_files]
        count = 0
        ambiguous_tcl = 0
        rejected_syntax = 0
        for relative, origin, precedence in sources:
            artifact = context.artifact_for_uri(relative)
            path = project_path(context.project_root, relative)
            if not artifact or not path.is_file():
                result.diagnostics.append(Diagnostic(
                    snapshot_id=context.snapshot.id, code="directive.input_missing",
                    severity=DiagnosticSeverity.ERROR,
                    message=f"directive input is missing from snapshot: {relative}",
                    stage=Stage.SOURCE.value, artifact_id=artifact.id if artifact else None,
                ))
                continue
            tcl_lexical_state: tuple[str, ...] = ()
            tcl_continued = False
            tcl_structure_uncertain = False
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            for line_number, raw_line in enumerate(lines, 1):
                if origin == "tcl":
                    # Tcl comments are commands beginning with ``#``.  An
                    # inline ``#`` or C++ ``//`` is data unless a Tcl parser
                    # proves otherwise, so this conservative importer does not
                    # erase it into a different valid directive.
                    line = raw_line.rstrip()
                    if (not tcl_lexical_state and not tcl_continued
                            and line.lstrip().startswith("#")):
                        line = ""
                else:
                    line = _strip_comment(raw_line)
                if origin == "tcl":
                    lexical_state_before = tcl_lexical_state
                    continued_from_previous = tcl_continued
                    uncertain_before = tcl_structure_uncertain
                    tcl_lexical_state, malformed = _advance_tcl_lexical_state(
                        line, tcl_lexical_state,
                    )
                    if malformed:
                        tcl_structure_uncertain = True
                    # Continuation is a physical-line property.  Computing it
                    # after stripping a comment would incorrectly promote the
                    # next line of ``# disabled \\`` as a fresh command.
                    tcl_continued = _tcl_continues(raw_line)
                    if _TCL_DIRECTIVE_MARKER.search(line):
                        rejection = _tcl_literal_rejection(
                            line, lexical_context_open=bool(lexical_state_before),
                            continued_from_previous=continued_from_previous,
                            structure_uncertain=uncertain_before,
                        )
                        if rejection:
                            ambiguous_tcl += 1
                            anchor = SourceAnchor(
                                artifact_id=artifact.id, start_line=line_number,
                                start_column=1, end_line=line_number,
                                end_column=len(line) + 1,
                            )
                            result.diagnostics.append(Diagnostic(
                                snapshot_id=context.snapshot.id,
                                code="directive.tcl_nonliteral_context",
                                severity=DiagnosticSeverity.WARNING,
                                message=("a possible Tcl directive was not imported because "
                                         "only complete, top-level literal commands are accepted"),
                                stage=Stage.SOURCE.value, artifact_id=artifact.id,
                                anchor=anchor,
                                id=("diagnostic_" + stable_hash({
                                    "snapshot": context.snapshot.id,
                                    "code": "directive.tcl_nonliteral_context",
                                    "artifact": artifact.id,
                                    "line": line_number,
                                    "reason": rejection,
                                })[:24]),
                                metadata={
                                    "reason": rejection,
                                    "completeness": Completeness.AMBIGUOUS.value,
                                    "parse_policy": "hlsgraph.tcl_literal_top_level_v1",
                                },
                            ))
                            continue
                tcl_match = _TCL_DIRECTIVE.match(line) if origin == "tcl" else None
                config_match = (
                    _CONFIG_DIRECTIVE.match(line) if origin == "config" else None
                )
                directive_kind: str | None = None
                directive_text: str | None = None
                if tcl_match:
                    directive_kind = tcl_match.group(1).upper()
                    directive_text = tcl_match.group(2)
                elif config_match:
                    directive_kind = config_match.group(1).upper()
                    directive_text = config_match.group(2)
                if not directive_kind or directive_text is None:
                    continue
                anchor = SourceAnchor(artifact_id=artifact.id, start_line=line_number,
                                      start_column=1, end_line=line_number,
                                      end_column=len(line) + 1)
                parsed, failure = _parse_directive(
                    kind=directive_kind, text=directive_text, origin=origin,
                )
                if failure is not None or parsed is None:
                    rejected_syntax += 1
                    metadata: dict[str, Any] = {
                        "reason": (
                            failure.reason if failure is not None else "parse_failed"
                        ),
                        "directive_kind": directive_kind,
                        "origin": origin,
                        "completeness": Completeness.AMBIGUOUS.value,
                        "grammar": "amd.vitis_hls.2024_2.external_directives.v1",
                    }
                    if failure is not None:
                        if failure.option:
                            metadata["option"] = failure.option
                        if failure.expected is not None:
                            metadata["expected_positionals"] = failure.expected
                        if failure.actual is not None:
                            metadata["actual_positionals"] = failure.actual
                    result.diagnostics.append(Diagnostic(
                        snapshot_id=context.snapshot.id,
                        code="directive.invalid_external_syntax",
                        severity=DiagnosticSeverity.WARNING,
                        message=(
                            f"{directive_kind} was not imported because it does not "
                            "match the strict AMD Vitis HLS 2024.2 external grammar"
                        ),
                        stage=Stage.SOURCE.value,
                        artifact_id=artifact.id,
                        anchor=anchor,
                        id=("diagnostic_" + stable_hash({
                            "snapshot": context.snapshot.id,
                            "code": "directive.invalid_external_syntax",
                            "artifact": artifact.id,
                            "line": line_number,
                            "kind": directive_kind,
                            "reason": metadata["reason"],
                        })[:24]),
                        metadata=metadata,
                    ))
                    continue
                count += 1
                options = parsed.options
                scope = parsed.scope_text
                block_control_unsupported = bool(
                    directive_kind == "INTERFACE"
                    and str(options.get("mode", "")).casefold()
                    in _INTERFACE_BLOCK_CONTROL_MODES
                )
                target, operand_target = _resolve_parsed_scope(
                    existing, parsed,
                    component_name=context.manifest.build.top,
                )
                identity_complete = bool(
                    target is not None
                    and (
                        directive_kind != "DEPENDENCE"
                        or "class" in options
                        or operand_target is not None
                    )
                )
                directive = Entity(
                    kind="hls.directive", name=directive_kind,
                    qualified_name=f"{relative}:{line_number}:{directive_kind}",
                    snapshot_id=context.snapshot.id,
                    authority=AuthorityClass.DECLARED_CONSTRAINT,
                    stage=Stage.SOURCE.value,
                    attrs={"directive_kind": directive_kind, "options": options,
                           "scope_text": scope, "origin": origin,
                           "precedence": precedence, "state": "requested",
                           "parse_policy": parsed.parse_policy},
                    anchors=[anchor],
                    completeness=(Completeness.COMPLETE if identity_complete
                                  else Completeness.AMBIGUOUS),
                )
                bind_directive_identity(
                    directive, target, scope_resolution="external_exact" if target else None,
                    operand_target=operand_target,
                )
                graph.add_entity(directive)
                if target:
                    graph.add_relation(Relation(
                        src=directive.id, dst=target.id, kind="hls.annotates",
                        snapshot_id=context.snapshot.id,
                        authority=AuthorityClass.DECLARED_CONSTRAINT,
                        stage=Stage.SOURCE.value,
                        attrs={"scope_node_id": target.id, "scope_text": scope,
                               "scope_resolution": "external_exact"}, anchors=[anchor],
                    ), allow_dangling=True)
                if identity_complete:
                    result.observations.append(Observation(
                        snapshot_id=context.snapshot.id, subject_id=directive.id,
                        predicate="directive.requested", value=options or True,
                        stage=Stage.SOURCE.value,
                        authority=AuthorityClass.DECLARED_CONSTRAINT,
                        artifact_id=artifact.id, anchor=anchor,
                        completeness=directive.completeness,
                        metadata={"directive_kind": directive_kind,
                                  **directive_identity_metadata(directive),
                                  "origin": origin, "precedence": precedence,
                                  "parse_policy": directive.attrs["parse_policy"]},
                    ))
                elif block_control_unsupported:
                    directive.attrs["state"] = "unsupported_requested"
                    result.diagnostics.append(Diagnostic(
                        snapshot_id=context.snapshot.id,
                        code="directive.unsupported_scope_form",
                        severity=DiagnosticSeverity.WARNING,
                        message=(
                            "block-control INTERFACE declarations are preserved as "
                            "unsupported because the current graph has no deterministic "
                            "return/control-port entity"
                        ),
                        stage=Stage.SOURCE.value, subject_id=directive.id,
                        artifact_id=artifact.id, anchor=anchor,
                        metadata={
                            "reason": "block_control_port_not_modeled",
                            "directive_kind": directive_kind,
                            "parse_policy": parsed.parse_policy,
                            "completeness": Completeness.AMBIGUOUS.value,
                        },
                    ))
                else:
                    unresolved_code = "directive.unresolved_scope"
                    unresolved_message = (
                        f"could not deterministically resolve external scope {scope!r}"
                    )
                    if (directive_kind == "DEPENDENCE"
                            and target is not None and "variable" in options):
                        unresolved_code = "directive.unresolved_operand"
                        unresolved_message = (
                            "could not deterministically resolve DEPENDENCE operand "
                            f"{options.get('variable')!r} inside scope {scope!r}"
                        )
                    elif directive_kind in {"ARRAY_PARTITION", "STREAM"}:
                        location = _resolve_location(
                            existing, parsed.location,
                            allowed_kinds=_DIRECTIVE_GRAMMARS[
                                directive_kind
                            ].scope_kinds,
                        )
                        if location is not None:
                            unresolved_code = "directive.unresolved_operand"
                            unresolved_message = (
                                f"could not deterministically resolve {directive_kind} "
                                f"operand {parsed.operand!r} inside "
                                f"{parsed.location!r}"
                            )
                    result.diagnostics.append(Diagnostic(
                        snapshot_id=context.snapshot.id, code=unresolved_code,
                        severity=DiagnosticSeverity.WARNING,
                        message=unresolved_message,
                        stage=Stage.SOURCE.value, subject_id=directive.id,
                        artifact_id=artifact.id, anchor=anchor,
                    ))
        result.coverage = {
            "external_directives": count,
            "ambiguous_tcl_directives": ambiguous_tcl,
            "rejected_external_syntax": rejected_syntax,
            "policy": "declared_precedence_v1",
            "tcl_context_policy": "hlsgraph.tcl_literal_top_level_v1",
            "tcl_parse_policy": "hlsgraph.amd_2024_2_tcl_literal_strict_v1",
            "config_parse_policy": (
                "hlsgraph.amd_2024_2_config_whitespace_strict_v1"
            ),
            "directive_grammar": "amd.vitis_hls.2024.2.external_directives.v1",
        }
        return result


def resolve_directives(result: ExtractionResult) -> None:
    """Resolve only declared precedence; this never claims a tool applied a directive."""
    graph = result.graph
    annotation_candidates: dict[str, list[Relation]] = defaultdict(list)
    for relation in graph.relations.values():
        if relation.kind == "hls.annotates":
            annotation_candidates[relation.src].append(relation)
    annotations = {
        directive_id: values[0]
        for directive_id, values in annotation_candidates.items()
        if len(values) == 1
    }
    groups: dict[tuple[str, str, str], list[Entity]] = defaultdict(list)
    for entity in graph.entities.values():
        if entity.kind != "hls.directive" or entity.id not in annotations:
            continue
        kind = str(entity.attrs.get("directive_kind", entity.name)).upper()
        operand_id = ""
        if kind == "DEPENDENCE":
            raw_operand = entity.attrs.get("variable_id")
            if (str(entity.completeness) != "complete"
                    or not isinstance(raw_operand, str) or not raw_operand):
                # An unresolved operand cannot participate in precedence or
                # become a selected declaration for another variable.
                continue
            operand_id = raw_operand
        groups[(annotations[entity.id].dst, kind, operand_id)].append(entity)
    existing_selected = {(item.subject_id, item.predicate) for item in result.observations}
    for (scope_id, kind, _operand_id), directives in groups.items():
        directives.sort(key=lambda item: (int(item.attrs.get("precedence", 0)), item.id))
        highest = max(int(item.attrs.get("precedence", 0)) for item in directives)
        top = [item for item in directives
               if int(item.attrs.get("precedence", 0)) == highest]
        top_values = {stable_hash(item.attrs.get("options") or True) for item in top}
        if len(top) > 1 and len(top_values) > 1:
            for item in top:
                item.attrs["state"] = "conflicting_declared"
                item.completeness = Completeness.AMBIGUOUS
            for overridden in [item for item in directives if item not in top]:
                overridden.attrs["state"] = "overridden_declared"
            result.diagnostics.append(Diagnostic(
                snapshot_id=top[0].snapshot_id,
                code="directive.ambiguous_same_precedence",
                severity=DiagnosticSeverity.WARNING,
                message=(f"{kind} on {scope_id} has conflicting declarations at the "
                         "same precedence; no selected declaration was inferred"),
                stage=Stage.SOURCE.value, subject_id=scope_id,
                metadata={"candidate_directive_ids": sorted(item.id for item in top),
                          "resolution_policy": "hlsgraph.declared_precedence_v1"},
            ))
            continue
        winner = top[-1]
        winner.attrs["state"] = "selected_declared"
        key = (winner.id, "directive.declared_selected")
        if key not in existing_selected:
            anchor = winner.anchors[0] if winner.anchors else None
            result.observations.append(Observation(
                snapshot_id=winner.snapshot_id, subject_id=winner.id,
                predicate="directive.declared_selected",
                value=winner.attrs.get("options") or True,
                stage=Stage.SOURCE.value, authority=AuthorityClass.DECLARED_CONSTRAINT,
                artifact_id=anchor.artifact_id if anchor else None, anchor=anchor,
                completeness=winner.completeness,
                metadata={"directive_kind": kind,
                          **directive_identity_metadata(winner),
                          "resolution_policy": "hlsgraph.declared_precedence_v1",
                          "tool_applied": False},
            ))
        for overridden in directives[:-1]:
            overridden.attrs["state"] = "overridden_declared"
            result.diagnostics.append(Diagnostic(
                snapshot_id=overridden.snapshot_id, code="directive.declared_override",
                severity=DiagnosticSeverity.INFO,
                message=f"{kind} on {scope_id} is superseded by a higher-precedence declaration; tool application remains unverified",
                stage=Stage.SOURCE.value, subject_id=overridden.id,
                metadata={"winner_directive_id": winner.id,
                          "resolution_policy": "hlsgraph.declared_precedence_v1"},
            ))
