"""Additive libclang source evidence; no hardware-instance inference."""
from __future__ import annotations

from collections import defaultdict
import re
from typing import Any

from ..model import (
    AuthorityClass, Completeness, Diagnostic, DiagnosticSeverity, Entity,
    Relation, stable_id,
)

_LABELS = {"hardware_instance": False, "ml_input_evidence": True}
_LOOPS = {"FOR_STMT", "CXX_FOR_RANGE_STMT", "WHILE_STMT", "DO_STMT"}
_VARIABLES = {"hls.port", "hls.memory", "hls.stream", "source.variable"}
_WRAPPERS = {"UNEXPOSED_EXPR", "PAREN_EXPR"}


def _key(cursor: Any) -> tuple:
    extent = cursor.extent
    return (str(extent.start.file), extent.start.offset, extent.end.offset,
            cursor.kind.name)


def _node_key(cursor: Any) -> tuple:
    # Distinct implicit casts can have identical source spans.
    return (*_key(cursor), cursor.hash)


def _decl_key(cursor: Any) -> tuple:
    # Referenced multi-declarator extents can start at the shared type token.
    return (str(cursor.location.file), cursor.location.offset, cursor.kind.name)


def _unwrap(cursor: Any) -> Any:
    while cursor.kind.name in _WRAPPERS:
        children = list(cursor.get_children())
        if len(children) != 1:
            break
        cursor = children[0]
    return cursor


def _refs(cursor: Any) -> list[Any]:
    """A subscript's base identifies its storage; index operands do not."""
    cursor = _unwrap(cursor)
    if cursor.kind.name == "DECL_REF_EXPR":
        return [cursor.referenced] if cursor.referenced else []
    children = list(cursor.get_children())
    if cursor.kind.name == "ARRAY_SUBSCRIPT_EXPR":
        children = children[:1]
    return [ref for child in children for ref in _refs(child)]


def _stream_type(cursor: Any) -> bool:
    return bool(re.match(r"^(?:(?:const|volatile)\s+)*hls::stream\s*<",
                         cursor.type.get_canonical().spelling))


class SourceFacts:
    def __init__(self, context: Any, result: Any, lexical_paths: dict) -> None:
        self.context = context
        self.result = result
        self.graph = result.graph
        self.lexical_paths = lexical_paths
        self.nodes: dict[tuple, tuple] = {}
        self.declarations: dict[tuple, str] = {}
        self.scopes: dict[str, Any] = {}
        self.design_units: list[tuple] = []
        self.bindings: dict[str, list[Relation]] = defaultdict(list)
        self.calls: dict[tuple, tuple] = {}
        self.effects: dict[str, set[str]] = defaultdict(set)
        self.uncertain_effects: set[str] = set()
        self.partial_effects: set[str] = set()
        self.ambiguous_sites: set[tuple] = set()

    def record(self, cursor: Any, entity: Entity | None, owner: str | None,
               path: tuple, parent: Any) -> None:
        if entity:
            self.declarations[_decl_key(cursor)] = entity.id
            if entity.kind in {"hls.kernel", "hls.function", "hls.loop"}:
                self.scopes[entity.id] = cursor
        if owner:
            self.nodes[_node_key(cursor)] = (cursor, owner, path, parent)

    def anchor(self, cursor: Any) -> Any:
        from .source import _anchor
        return _anchor(self.context, cursor)

    def warn(self, code: str, subject: str, cursor: Any, message: str) -> None:
        anchor = self.anchor(cursor)
        diagnostic = Diagnostic(
            snapshot_id=self.context.snapshot.id, code=code,
            severity=DiagnosticSeverity.WARNING, message=message, stage="ast",
            subject_id=subject, anchor=anchor,
            artifact_id=anchor.artifact_id if anchor else None,
        )
        if not any(item.id == diagnostic.id for item in self.result.diagnostics):
            self.result.diagnostics.append(diagnostic)

    def relation(self, kind: str, src: str, dst: str, attrs: dict,
                 cursors: list, completeness: str = "complete") -> Relation:
        return self.graph.add_relation(Relation(
            kind=kind, src=src, dst=dst, snapshot_id=self.context.snapshot.id,
            stage="ast", authority=AuthorityClass.STATIC_FACT,
            attrs={**_LABELS, **attrs, "completeness": completeness},
            anchors=[anchor for cursor in cursors if (anchor := self.anchor(cursor))],
            completeness=Completeness(completeness),
        ))

    def entity(self, kind: str, name: str, identity: str, attrs: dict,
               cursor: Any) -> Entity:
        anchor = self.anchor(cursor)
        return self.graph.add_entity(Entity(
            kind=kind, name=name, qualified_name=identity,
            snapshot_id=self.context.snapshot.id, stage="ast",
            authority=AuthorityClass.STATIC_FACT,
            attrs={**_LABELS, **attrs}, anchors=[anchor] if anchor else [],
        ))

    def operands(self, expression: Any, owner: str, path: tuple) -> set[str]:
        from .source import _lexically_visible_operand, _unique_function_owner
        anchor = self.anchor(expression)
        if not anchor:
            return set()
        found = set()
        for ref in _refs(expression):
            entity_id = self.declarations.get(_decl_key(ref))
            entity = self.graph.entities.get(entity_id)
            if entity is None or entity.kind not in _VARIABLES:
                continue
            visible = _lexically_visible_operand(
                self.graph, entity, owner=self.graph.entities[owner],
                artifact_id=anchor.artifact_id, line=anchor.start_line,
                pragma_scope_path=path, entity_lexical_paths=self.lexical_paths,
            )
            # The pragma resolver is line-oriented. AST references also prove
            # visibility of a declaration earlier on the very same line.
            if not visible and entity.kind != "hls.port":
                declaration_path = self.lexical_paths.get(entity.id)
                visible = bool(
                    declaration_path is not None
                    and _unique_function_owner(self.graph, entity) == self.graph.entities[owner]
                    and path[:len(declaration_path)] == declaration_path
                    and any(a.artifact_id == anchor.artifact_id
                            and a.start_line == anchor.start_line
                            and (a.start_column or 0) < (anchor.start_column or 0)
                            for a in entity.anchors)
                )
            if visible:
                found.add(entity.id)
        return found

    def emit(self, function_by_name: dict) -> None:
        orders: dict[str, int] = defaultdict(int)
        seen_calls: set[tuple] = set()
        sites: dict[tuple, list[tuple]] = defaultdict(list)
        for record in self.nodes.values():
            if record[0].kind.name == "CALL_EXPR":
                sites[_key(record[0])].append(record)
        for site, records in sites.items():
            per_unit: dict[int, int] = defaultdict(int)
            meanings = set()
            for cursor, owner, path, _ in records:
                per_unit[id(cursor.translation_unit)] += 1
                meanings.add(cursor.referenced.get_usr() if cursor.referenced else None)
            if max(per_unit.values()) > 1 or len(meanings) > 1:
                self.ambiguous_sites.add(site)
                cursor, owner, path, _ = records[0]
                self.warn("mapping.ambiguous_call_site", owner, cursor,
                          "multiple call expressions share one source span; call facts were withheld")
                for cursor, owner, path, _ in records:
                    self.uncertain_effects.update(self.operands(cursor, owner, path))
        # Source offsets, not AST visitation or dictionary order, define order.
        for key, (cursor, owner, path, parent) in sorted(self.nodes.items()):
            if cursor.kind.name != "CALL_EXPR":
                continue
            if cursor.referenced and cursor.referenced.kind.name in {"CONSTRUCTOR", "DESTRUCTOR"}:
                continue
            call_key = _key(cursor)
            if call_key in self.ambiguous_sites:
                continue
            if (owner, call_key) in seen_calls:
                continue
            seen_calls.add((owner, call_key))
            order = orders[owner]
            orders[owner] += 1
            targets = set(function_by_name.get(cursor.spelling, []))
            if len(targets) != 1:
                continue
            target = next(iter(targets))
            # Match the existing software.calls resolution, but reject indirect
            # calls and same-spelling external/member overloads.
            referenced = cursor.referenced
            if referenced is None or referenced.kind.name not in {
                "FUNCTION_DECL", "CXX_METHOD", "FUNCTION_TEMPLATE",
            }:
                self.warn("mapping.ambiguous_argument", owner, cursor,
                          "indirect callee cannot establish a parameter binding")
                continue
            definition = referenced.get_definition()
            definition_id = self.declarations.get(_decl_key(definition)) if definition else None
            same_symbol = bool(referenced.get_usr() and
                               referenced.get_usr() == self.scopes[target].get_usr())
            if definition_id != target and not same_symbol:
                self.warn("mapping.ambiguous_argument", owner, cursor,
                          "callee declaration does not uniquely match the resolved local call")
                continue
            site = stable_id("call_site", self.anchor(cursor))
            self.calls[call_key] = (cursor, owner, target, site, order, parent)
            parameters = [child for child in self.scopes[target].get_children()
                          if child.kind.name == "PARM_DECL"]
            arguments = list(cursor.get_arguments())
            if referenced.kind.name == "CXX_METHOD" and cursor.spelling.startswith("operator"):
                arguments = arguments[1:]
            for index, argument in enumerate(arguments):
                if index >= len(parameters):
                    self.warn("mapping.ambiguous_argument", owner, argument,
                              "argument has no unique callee parameter")
                    continue
                param = parameters[index]
                param_id = self.declarations.get(_decl_key(param))
                candidates = self.operands(argument, owner, path)
                if len(candidates) != 1 or param_id is None:
                    self.warn("mapping.ambiguous_argument", owner, argument,
                              "argument has no unique visible base entity and parameter")
                    continue
                base = next(iter(candidates))
                expression = _unwrap(argument).kind.name
                expression = ("entity" if expression == "DECL_REF_EXPR" else
                              "element" if expression == "ARRAY_SUBSCRIPT_EXPR" else "expression")
                param_type = param.type.get_canonical().kind.name
                passing = ("array" if "ARRAY" in param_type else
                           "reference" if "REFERENCE" in param_type else
                           "pointer" if param_type == "POINTER" else "value")
                relation = self.relation(
                    "source.binds_argument", base, param_id,
                    {"call_site_id": site, "arg_index": index, "call_order": order,
                     "passing": passing, "expression": expression}, [argument],
                )
                self.bindings[site].append(relation)
        self.accesses()
        self.regions()

    def accesses(self) -> None:
        counts: dict[tuple, dict[tuple, Any]] = defaultdict(dict)
        partial: set[tuple] = set()
        for key, (cursor, owner, path, parent) in sorted(self.nodes.items()):
            if cursor.kind.name == "CALL_EXPR":
                if _key(cursor) in self.ambiguous_sites:
                    continue
                children = list(cursor.get_children())
                name = cursor.spelling
                receiver = None
                # Member call's first child contains the receiver, whereas
                # operator calls expose it as the first argument.
                if children and name in {"read", "read_nb", "empty", "write", "write_nb", "full"}:
                    receiver = children[0]
                elif name in {"operator>>", "operator<<"}:
                    arguments = list(cursor.get_arguments())
                    receiver = arguments[0] if arguments else None
                else:
                    # Passing storage onward does not prove its transitive
                    # direction. Alias/escape analysis is outside this pass.
                    for argument in cursor.get_arguments():
                        typ = argument.type.get_canonical().kind.name
                        if typ in {"POINTER", "LVALUEREFERENCE", "RVALUEREFERENCE"} or _stream_type(argument):
                            self.uncertain_effects.update(self.operands(argument, owner, path))
                    if children and children[0].kind.name == "MEMBER_REF_EXPR":
                        candidates = self.operands(children[0], owner, path)
                        if any(_stream_type(ref) for ref in _refs(children[0])):
                            self.uncertain_effects.update(candidates)
                            self.warn("mapping.unknown_stream_access", owner, cursor,
                                      "stream method has no supported access direction")
                if receiver is not None:
                    references = _refs(receiver)
                    if not any(_stream_type(ref) for ref in references):
                        continue
                    candidates = self.operands(receiver, owner, path)
                    if len(candidates) != 1:
                        self.warn("mapping.ambiguous_stream_access", owner, cursor,
                                  "stream access has no unique visible receiver")
                        continue
                    variable = next(iter(candidates))
                    if self.graph.entities[variable].kind not in {"hls.port", "hls.stream"}:
                        self.warn("mapping.ambiguous_stream_access", owner, cursor,
                                  "receiver has no existing stream or port entity")
                        continue
                    direction = "write" if name in {"write", "write_nb", "operator<<"} else "read"
                    scope = owner
                    ancestor = parent
                    child = cursor
                    conditional = False
                    while ancestor is not None:
                        if ancestor.kind.name in _LOOPS:
                            scope = self.declarations.get(_decl_key(ancestor), owner)
                            children = list(ancestor.get_children())
                            body = children[0] if ancestor.kind.name == "DO_STMT" else children[-1]
                            if _node_key(child) != _node_key(body):
                                conditional = True
                                self.warn("mapping.unknown_stream_count", scope, cursor,
                                          "loop-header access has no proven per-iteration execution count")
                            break
                        if ancestor.kind.name in {"IF_STMT", "SWITCH_STMT", "CONDITIONAL_OPERATOR"}:
                            conditional = True
                        if ancestor.kind.name == "LAMBDA_EXPR" or (
                            ancestor.kind.name == "BINARY_OPERATOR"
                            and {"&&", "||"} & {token.spelling for token in ancestor.get_tokens()}
                        ):
                            conditional = True
                        if ancestor.kind.name == "LAMBDA_EXPR" or (
                            ancestor.kind.name in {"CXX_UNARY_EXPR", "UNARY_EXPR", "CXX_TYPEID_EXPR"}
                        ):
                            conditional = True
                            self.uncertain_effects.add(variable)
                        record = self.nodes.get(_node_key(ancestor))
                        child = ancestor
                        ancestor = record[3] if record else None
                    group = (scope, variable, direction)
                    counts[group][_key(cursor)] = cursor
                    if conditional or name in {"read_nb", "write_nb"}:
                        partial.add(group)
                        self.partial_effects.add(variable)
                    # Availability queries are source accesses but do not prove
                    # a channel transfer direction by themselves.
                    if name not in {"empty", "full"}:
                        self.effects[variable].add(direction)
            elif cursor.kind.name == "ARRAY_SUBSCRIPT_EXPR":
                candidates = self.operands(cursor, owner, path)
                if len(candidates) != 1:
                    continue
                variable = next(iter(candidates))
                ancestor = parent
                current = cursor
                while ancestor is not None and ancestor.kind.name in _WRAPPERS:
                    current = ancestor
                    record = self.nodes.get(_node_key(ancestor))
                    ancestor = record[3] if record else None
                if ancestor is not None and ancestor.kind.name == "ARRAY_SUBSCRIPT_EXPR":
                    continue
                direction = {"read"}
                if ancestor is not None:
                    children = list(ancestor.get_children())
                    tokens = [token.spelling for token in ancestor.get_tokens()]
                    if ancestor.kind.name in {"BINARY_OPERATOR", "COMPOUND_ASSIGNMENT_OPERATOR"} and children:
                        between = [token.spelling for token in ancestor.get_tokens()
                                   if children[0].extent.end.offset <= token.extent.start.offset
                                   < children[1].extent.start.offset] if len(children) > 1 else []
                        if _key(children[0]) == _key(current) and "=" in between:
                            direction = {"write"}
                        elif _key(children[0]) == _key(current) and ancestor.kind.name == "COMPOUND_ASSIGNMENT_OPERATOR":
                            direction = {"read", "write"}
                    elif ancestor.kind.name == "UNARY_OPERATOR":
                        if "++" in tokens or "--" in tokens:
                            direction = {"read", "write"}
                        elif "&" in tokens:
                            self.uncertain_effects.add(variable)
                            continue
                    elif ancestor.kind.name == "CALL_EXPR":
                        self.uncertain_effects.add(variable)
                        continue
                self.effects[variable].update(direction)
            elif cursor.kind.name == "VAR_DECL":
                typ = cursor.type.get_canonical().kind.name
                if typ in {"POINTER", "LVALUEREFERENCE", "RVALUEREFERENCE"}:
                    self.uncertain_effects.update(self.operands(cursor, owner, path))
        for (scope, variable, direction), cursors in sorted(counts.items()):
            self.relation("source.stream_access", scope, variable,
                          {"direction": direction, "count_per_iteration": len(cursors)},
                          list(cursors.values()),
                          "partial" if (scope, variable, direction) in partial else "complete")

    def regions(self) -> None:
        directives: dict[str, list[Entity]] = defaultdict(list)
        for entity in list(self.graph.entities.values()):
            if entity.kind == "hls.directive" and entity.attrs.get("origin") == "source_pragma":
                directives[entity.attrs.get("scope_id")].append(entity)
        for scope, items in sorted(directives.items(), key=lambda item: str(item[0])):
            dataflow = [item for item in items if item.name == "DATAFLOW"]
            if not dataflow or scope not in self.scopes:
                continue
            cursor = self.scopes[scope]
            if any(item.name == "PIPELINE" for item in items):
                self.warn("dataflow.pipeline_conflict", scope, cursor,
                          "DATAFLOW and PIPELINE are declared on the same source scope")
            directive = sorted(dataflow, key=lambda item: item.id)[0]
            region = self.entity(
                "source.dataflow_region", "dataflow", f"{scope}:dataflow",
                {"scope_id": scope, "scope_kind": "loop" if cursor.kind.name in _LOOPS else "function",
                 "directive_id": directive.id}, cursor,
            )
            self.relation("source.contains", scope, region.id, {}, [cursor])
            bodies = [child for child in cursor.get_children() if child.kind.name == "COMPOUND_STMT"]
            body = bodies[-1] if bodies else None
            statements = list(body.get_children()) if body else []
            if body is None and cursor.kind.name in _LOOPS:
                children = list(cursor.get_children())
                statements = children[-1:] if cursor.kind.name != "DO_STMT" else children[:1]
            channels: dict[str, dict[str, list[tuple]]] = defaultdict(lambda: defaultdict(list))
            direct_locals: set[str] = set()
            for statement in statements:
                if statement.kind.name == "DECL_STMT":
                    direct_locals.update(self.declarations[_decl_key(child)] for child in statement.get_children()
                                         if _decl_key(child) in self.declarations)
            for statement in statements:
                if statement.kind.name == "DECL_STMT":
                    continue
                call = self.calls.get(_key(_unwrap(statement)))
                if statement.kind.name != "CALL_EXPR" and call is None:
                    self.warn("dataflow.noncanonical_statement", region.id, statement,
                              "direct source region statement is not a declaration or call")
                    continue
                if call is None:
                    self.warn("dataflow.unresolved_process", region.id, statement,
                              "direct call has no uniquely resolved project-local callee")
                    for argument in _unwrap(statement).get_arguments():
                        if _unwrap(argument).kind.name != "DECL_REF_EXPR":
                            self.warn("dataflow.noncanonical_argument", region.id, argument,
                                      "call argument is an element or expression")
                    continue
                call_cursor, owner, target, site, order, _ = call
                process = self.entity(
                    "source.dataflow_process", self.graph.entities[target].name,
                    f"{region.id}:{site}",
                    {"region_id": region.id, "callee_id": target,
                     "call_site_id": site, "order": order}, call_cursor,
                )
                self.relation("source.contains", region.id, process.id, {}, [call_cursor])
                for argument in call_cursor.get_arguments():
                    if _unwrap(argument).kind.name != "DECL_REF_EXPR":
                        self.warn("dataflow.noncanonical_argument", process.id, argument,
                                  "call argument is an element or expression")
                for binding in self.bindings[site]:
                    variable = self.graph.entities[binding.src]
                    if variable.id not in direct_locals or variable.kind not in {"hls.stream", "hls.memory"}:
                        continue
                    directions = self.effects[binding.dst]
                    if not directions or binding.dst in self.uncertain_effects:
                        self.warn("dataflow.unknown_direction", process.id, call_cursor,
                                  f"bound parameter {binding.dst} has no proven access direction")
                        continue
                    if binding.attrs["expression"] != "entity":
                        continue
                    for direction in directions:
                        channels[variable.id][direction].append((process.id, binding.dst, call_cursor))
            for variable_id, directions in sorted(channels.items()):
                producers, consumers = directions["write"], directions["read"]
                for role, endpoints in (("producers", producers), ("consumers", consumers)):
                    if len({endpoint[0] for endpoint in endpoints}) > 1:
                        self.warn(f"dataflow.multiple_{role}", variable_id, cursor,
                                  f"local source channel has multiple {role}")
                depths = [item.attrs.get("options", {}).get("depth")
                          for item in directives.get(variable_id, []) if item.name == "STREAM"]
                depth = None
                if len(depths) == 1:
                    try:
                        depth = int(str(depths[0]), 10)
                        if depth <= 0:
                            depth = None
                    except (TypeError, ValueError):
                        pass
                if depths and depth is None:
                    self.warn("dataflow.unknown_depth", variable_id, cursor,
                              "STREAM depth is not a unique positive integer literal")
                for producer, producer_param, producer_cursor in producers:
                    for consumer, consumer_param, consumer_cursor in consumers:
                        if producer == consumer:
                            continue
                        self.relation("source.dataflow_channel", producer, consumer,
                                      {"via": "stream" if self.graph.entities[variable_id].kind == "hls.stream" else "array",
                                       "variable_id": variable_id, "producer_param": producer_param,
                                       "consumer_param": consumer_param, "depth": depth,
                                       "depth_assumed": depth is None},
                                      [producer_cursor, consumer_cursor],
                                      "partial" if {producer_param, consumer_param} & self.partial_effects
                                      else "complete")

    def synthesizability(self, index: Any, cindex: Any) -> None:
        from .source import _relative_file
        context = self.context
        testbenches = set(context.manifest.build.testbench_files)
        adjacency: dict[str, set[str]] = defaultdict(set)
        for unit, original_tu, args in self.design_units:
            if unit.file in testbenches:
                continue
            # Use the same compiler/preprocessor as source extraction. The
            # synthesis diagnostic view defines the standard HLS macro; graph
            # entities and calls still describe the configured compilation.
            try:
                tu = index.parse(str(context.project_root / unit.file),
                                 args=[*args, "-U__SYNTHESIS__", "-D__SYNTHESIS__=1"],
                                 options=cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD)
            except Exception:
                kernel = next(iter(self.graph.by_kind("hls.kernel")), None)
                if kernel:
                    self.warn("synth.analysis_incomplete", kernel.id, self.scopes[kernel.id],
                              "synthesis preprocessing could not be parsed")
                continue
            untracked = [inc for inc in tu.get_includes()
                         if not (relative := _relative_file(context, str(inc.include)))
                         or context.artifact_for_uri(relative) is None]
            if untracked or any(d.severity >= cindex.Diagnostic.Error for d in tu.diagnostics):
                kernel = next(iter(self.graph.by_kind("hls.kernel")), None)
                if kernel:
                    self.warn("synth.analysis_incomplete", kernel.id, self.scopes[kernel.id],
                              "synthesis view has parse errors or inputs outside the snapshot")
                continue
            kernel = next(iter(self.graph.by_kind("hls.kernel")), None)

            resolved_files: dict[str, str | None] = {}

            def scan(cursor: Any, owner: str | None = None) -> None:
                filename = str(cursor.location.file) if cursor.location.file else None
                if filename is not None and filename not in resolved_files:
                    resolved_files[filename] = _relative_file(context, filename)
                relative = resolved_files[filename] if filename is not None else None
                if cursor.kind.name != "TRANSLATION_UNIT" and (
                    relative is None or relative in testbenches
                    or context.artifact_for_uri(relative) is None
                ):
                    return
                if cursor.kind.name in {"FUNCTION_DECL", "CXX_METHOD", "FUNCTION_TEMPLATE"}:
                    if cursor.spelling == "main":
                        return
                    owner = self.declarations.get(_decl_key(cursor)) or owner
                    if owner is None:
                        # Synthesis-only statements may change the function's
                        # extent, but not its declaration start and symbol.
                        matches = [eid for eid, value in self.scopes.items()
                                   if value.kind.name == cursor.kind.name
                                   and value.get_usr() == cursor.get_usr()
                                   and str(value.location.file) == str(cursor.location.file)]
                        owner = matches[0] if len(matches) == 1 else None
                subject = self.declarations.get(_decl_key(cursor)) or owner or (kernel.id if kernel else None)
                if subject is None:
                    return
                kind = cursor.kind.name
                name = cursor.spelling
                codes = set()
                if kind in {"CXX_NEW_EXPR", "CXX_DELETE_EXPR"}:
                    codes.add("dynamic_allocation")
                if kind == "CXX_METHOD" and cursor.is_virtual_method():
                    codes.add("virtual_function")
                if cursor.type.get_canonical().kind.name == "LONGDOUBLE":
                    codes.add("long_double")
                if kind in {"VAR_DECL", "PARM_DECL", "FIELD_DECL", "FUNCTION_DECL", "CXX_METHOD"}:
                    typ = cursor.result_type if kind in {"FUNCTION_DECL", "CXX_METHOD"} else cursor.type
                    typ = typ.get_canonical()
                    types = []
                    while typ.kind.name in {"POINTER", "LVALUEREFERENCE", "RVALUEREFERENCE", "CONSTANTARRAY", "INCOMPLETEARRAY", "VARIABLEARRAY"}:
                        types.append(typ.kind.name)
                        typ = typ.element_type if "ARRAY" in typ.kind.name else typ.get_pointee()
                    if any(types[i:i + 2] == ["POINTER", "POINTER"] for i in range(len(types))):
                        codes.add("pointer_to_pointer")
                    if "POINTER" in types and typ.kind.name in {"FUNCTIONPROTO", "FUNCTIONNOPROTO"}:
                        codes.add("function_pointer")
                    if typ.kind.name == "LONGDOUBLE":
                        codes.add("long_double")
                if kind == "CALL_EXPR":
                    ref = cursor.referenced
                    if ref and ref.kind.name in {"VAR_DECL", "PARM_DECL", "FIELD_DECL"}:
                        codes.add("function_pointer")
                    if ref is None and any(
                        item.type.get_canonical().kind.name == "POINTER"
                        and item.type.get_canonical().get_pointee().kind.name in {"FUNCTIONPROTO", "FUNCTIONNOPROTO"}
                        for child in list(cursor.get_children())[:1] for item in _refs(child)
                    ):
                        codes.add("function_pointer")
                    if name in {"malloc", "calloc", "realloc", "free"}:
                        codes.add("dynamic_allocation")
                    if name in {"printf", "fprintf", "puts", "fopen", "time", "sleep", "exit"}:
                        codes.add("system_call")
                    if ref and owner:
                        targets = [eid for eid, declaration in self.scopes.items()
                                   if declaration.kind.name not in _LOOPS
                                   and ref.get_usr() and declaration.get_usr() == ref.get_usr()]
                        target = targets[0] if len(targets) == 1 else None
                        # Only edges already resolved by software.calls qualify.
                        if target and any(r.kind == "software.calls" and r.src == owner and r.dst == target
                                          for r in self.graph.relations.values()):
                            adjacency[owner].add(target)
                if kind == "DECL_REF_EXPR" and name in {"cout", "cerr", "cin"}:
                    ref = cursor.referenced
                    if ref and ref.semantic_parent and ref.semantic_parent.spelling == "std":
                        codes.add("system_call")
                for code in sorted(codes):
                    self.warn(f"synth.{code}", subject, cursor,
                              f"synthesis source contains {code.replace('_', ' ')}: {name or kind.lower()}")
                for child in cursor.get_children():
                    scan(child, owner)

            scan(tu.cursor)
        # Reachability proves membership in a directed cycle; acyclic callers
        # of a recursive function are not themselves diagnosed as recursive.
        for owner in sorted(adjacency):
            pending = list(adjacency[owner])
            seen = set()
            while pending:
                target = pending.pop()
                if target == owner:
                    cursor = self.scopes[owner]
                    self.warn("synth.recursion", owner, cursor,
                              "function participates in a resolved project-local call cycle")
                    break
                if target not in seen:
                    seen.add(target)
                    pending.extend(adjacency.get(target, ()))
