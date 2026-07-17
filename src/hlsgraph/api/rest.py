"""Read-only, dependency-free REST/OpenAPI adapter for :class:`CoreService`.

The HTTP layer deliberately contains no HLS query semantics.  It validates wire
parameters, delegates to ``CoreService``/the immutable ledger, and serializes the
result.  In particular, it never exposes source file contents.
"""
from __future__ import annotations

import ipaddress
import json
import math
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import parse_qs, unquote, urlsplit

from ..bundle import GraphBundle
from ..model import json_ready, stable_hash
from ..query import CoreService, ExploreSpec, QuerySpec
from ..run_projection import (
    PUBLIC_FAILURE_CLASSES, PUBLIC_GATE_KINDS, PUBLIC_GATE_STATUSES,
    PUBLIC_RUN_STATUSES, public_enum, public_identifier,
    public_identifier_list, public_sha256, public_timestamp,
    sanitize_run_metadata,
)
from ..sdk import Project
from ..store import StoreError
from ..version import SCHEMA_VERSION, __version__


API_PREFIX = "/api/v1"


@dataclass(slots=True)
class ApiResponse:
    status: int
    body: dict[str, Any]
    headers: dict[str, str] = field(default_factory=dict)

    def encoded(self) -> bytes:
        return (json.dumps(json_ready(self.body), ensure_ascii=False, sort_keys=True,
                           separators=(",", ":")) + "\n").encode("utf-8")


def _one(params: Mapping[str, list[str]], name: str, default: str | None = None) -> str | None:
    values = params.get(name)
    if not values:
        return default
    if len(values) != 1:
        raise ValueError(f"query parameter {name!r} must occur once")
    return values[0]


def _integer(params: Mapping[str, list[str]], name: str, default: int,
             minimum: int, maximum: int) -> int:
    raw = _one(params, name)
    try:
        value = default if raw is None else int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"query parameter {name!r} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"query parameter {name!r} must be in {minimum}..{maximum}")
    return value


def _csv(params: Mapping[str, list[str]], name: str) -> list[str]:
    raw = _one(params, name)
    return [item.strip() for item in (raw or "").split(",") if item.strip()]


def _boolean(params: Mapping[str, list[str]], name: str, default: bool = False) -> bool:
    raw = _one(params, name)
    if raw is None:
        return default
    if raw.casefold() in {"1", "true", "yes"}:
        return True
    if raw.casefold() in {"0", "false", "no"}:
        return False
    raise ValueError(f"query parameter {name!r} must be a boolean")


def _public_run(value: Any) -> dict[str, Any]:
    """Redact argv and working directories from unauthenticated REST responses."""
    ready = json_ready(value)
    raw = dict(ready) if isinstance(ready, Mapping) else {}
    command = raw.get("command", [])
    valid_command = (command if isinstance(command, list)
                     and all(isinstance(item, str) and item for item in command)
                     else None)
    elapsed = raw.get("elapsed_s")
    payload = {
        "id": public_identifier(raw.get("id")),
        "snapshot_id": public_identifier(raw.get("snapshot_id")),
        "stage": public_identifier(raw.get("stage")),
        "backend": public_identifier(raw.get("backend")),
        "request_hash": public_sha256(raw.get("request_hash")),
        "toolchain_id": public_identifier(raw.get("toolchain_id")),
        "status": public_enum(raw.get("status"), PUBLIC_RUN_STATUSES),
        "environment_hash": public_sha256(raw.get("environment_hash")),
        "input_artifact_ids": public_identifier_list(raw.get("input_artifact_ids")),
        "output_artifact_ids": public_identifier_list(raw.get("output_artifact_ids")),
        "diagnostics": public_identifier_list(raw.get("diagnostics")),
        "failure_class": public_enum(raw.get("failure_class"), PUBLIC_FAILURE_CLASSES),
        "exit_code": (raw.get("exit_code") if isinstance(raw.get("exit_code"), int)
                      and not isinstance(raw.get("exit_code"), bool) else None),
        "attempt": (raw.get("attempt") if isinstance(raw.get("attempt"), int)
                    and not isinstance(raw.get("attempt"), bool)
                    and raw.get("attempt") >= 1 else None),
        "started_at": public_timestamp(raw.get("started_at")),
        "finished_at": public_timestamp(raw.get("finished_at")),
        "elapsed_s": (elapsed if isinstance(elapsed, (int, float))
                      and not isinstance(elapsed, bool) and math.isfinite(float(elapsed))
                      and elapsed >= 0 else None),
        "metadata": sanitize_run_metadata(raw.get("metadata")),
    }
    gates = raw.get("gates") if isinstance(raw.get("gates"), list) else []
    payload["gates"] = [{
        "kind": public_enum(item.get("kind"), PUBLIC_GATE_KINDS),
        "status": public_enum(item.get("status"), PUBLIC_GATE_STATUSES),
        "evidence_ids": public_identifier_list(item.get("evidence_ids")),
        "reason_redacted": item.get("reason") is not None,
    } for item in gates if isinstance(item, Mapping)]
    argv0 = None
    if valid_command:
        try:
            candidate = Path(valid_command[0]).name
        except (OSError, ValueError):
            candidate = ""
        argv0 = public_identifier(candidate)
    try:
        command_hash = stable_hash(valid_command if valid_command is not None
                                   else {"malformed_legacy_argv": True})
    except (TypeError, ValueError, UnicodeError):
        command_hash = stable_hash({"malformed_legacy_argv": True})
    payload["execution_metadata"] = {
        "argv0": argv0,
        "command_hash": command_hash,
        "command_redacted": True,
        "working_directory_redacted": True,
        "message_redacted": raw.get("message") is not None,
        "backend_details_redacted": True,
    }
    return payload


def _coerce_bundle(value: Project | GraphBundle | CoreService | str | Path) -> tuple[GraphBundle, str | None]:
    if isinstance(value, CoreService):
        return value.bundle, value.snapshot_id
    if isinstance(value, Project):
        return value.bundle, None
    if isinstance(value, GraphBundle):
        return value, None
    return Project.open(value).bundle, None


class RestApplication:
    """Pure request dispatcher, usable directly in tests or through HTTP."""

    def __init__(self, project: Project | GraphBundle | CoreService | str | Path,
                 *, snapshot_id: str | None = None):
        self.bundle, inherited_snapshot = _coerce_bundle(project)
        self.snapshot_id = snapshot_id or inherited_snapshot

    def _service(self, params: Mapping[str, list[str]]) -> CoreService:
        requested = _one(params, "snapshot_id", self.snapshot_id)
        return CoreService(self.bundle, snapshot_id=requested)

    @staticmethod
    def _ok(body: dict[str, Any], status: int = 200) -> ApiResponse:
        return ApiResponse(status, body, {
            "Content-Type": "application/json; charset=utf-8",
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        })

    @staticmethod
    def _error(status: int, code: str, message: str, **headers: str) -> ApiResponse:
        base_headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        }
        base_headers.update(headers)
        return ApiResponse(status, {
            "schema_version": SCHEMA_VERSION,
            "error": {"code": code, "message": message},
        }, base_headers)

    def dispatch(self, method: str, target: str) -> ApiResponse:
        if method.upper() != "GET":
            return self._error(405, "method_not_allowed", "the public REST API is read-only",
                               Allow="GET")
        parsed = urlsplit(target)
        path = parsed.path.rstrip("/") or "/"
        try:
            params = parse_qs(parsed.query, keep_blank_values=True, strict_parsing=False)
            if path in {"/openapi.json", f"{API_PREFIX}/openapi.json"}:
                return self._ok(openapi_document())
            if not path.startswith(API_PREFIX + "/"):
                return self._error(404, "not_found", "unknown API route")
            route = path[len(API_PREFIX):]
            requested = _one(params, "snapshot_id", self.snapshot_id)
            selected = (self.bundle.store.snapshot(requested) if requested else
                        self.bundle.latest_snapshot() or
                        self.bundle.store.latest_candidate(self.bundle.manifest.project_id))
            if route == "/status":
                if selected is None or not self.bundle.store.has_graph(selected.id):
                    value = self.bundle.status(selected.id if selected else None)
                    if selected:
                        value.update({
                            "runs": len(self.bundle.store.runs(selected.id)),
                            "observations": len(self.bundle.store.observations(selected.id)),
                            "diagnostic_history": len(self.bundle.store.diagnostics(selected.id)),
                            "diagnostics": [json_ready(item) for item in
                                            self.bundle.store.diagnostics(selected.id)],
                            "completeness": "unavailable",
                        })
                    return self._ok(value)
            if (route == "/overview"
                    and (selected is None or not self.bundle.store.has_graph(selected.id))):
                # Keep the wire contract and validation stable even when the
                # latest candidate failed before producing a canonical graph.
                # Candidate health is still useful evidence and must not
                # require constructing CoreService, whose graph invariant is
                # intentionally stricter.
                view = _one(params, "view", "architecture") or "architecture"
                if view not in {"architecture", "evidence"}:
                    raise ValueError("view must be architecture or evidence")
                _integer(params, "depth", 1, 0, 8)
                _integer(params, "top_k", 12, 1, 50)
                snapshot_key = selected.id if selected else None
                return self._ok({
                    "schema_version": SCHEMA_VERSION,
                    "snapshot_id": snapshot_key,
                    "status": self.bundle.status(snapshot_key),
                    "architecture": None,
                    "incomplete": True,
                    "message": "no successful canonical graph is available",
                })
            if (route in {"/diagnostics", "/runs"}
                    and (selected is None or not self.bundle.store.has_graph(selected.id))):
                snapshot_key = selected.id if selected else None
                if route == "/diagnostics":
                    values = (self.bundle.store.diagnostics(snapshot_key)
                              if snapshot_key else [])
                    severity = _one(params, "severity")
                    if severity:
                        values = [item for item in values if str(item.severity) == severity]
                    return self._ok(self._page(snapshot_key, values, params))
                values = self.bundle.store.runs(snapshot_key) if snapshot_key else []
                stage = _one(params, "stage")
                status = _one(params, "status")
                if stage:
                    values = [item for item in values if item.stage == stage]
                if status:
                    values = [item for item in values if str(item.status) == status]
                return self._ok(self._page(
                    snapshot_key, [_public_run(item) for item in values], params
                ))
            service = self._service(params)
            return self._dispatch_get(service, route, params)
        except KeyError as exc:
            identifier = str(exc.args[0]) if exc.args else "resource"
            return self._error(404, "not_found", f"resource not found: {identifier}")
        except ValueError as exc:
            return self._error(400, "invalid_request", str(exc))
        except StoreError as exc:
            return self._error(409, "unavailable_snapshot", str(exc))

    def _dispatch_get(self, service: CoreService, route: str,
                      params: Mapping[str, list[str]]) -> ApiResponse:
        if route == "/status":
            return self._ok(service.status().to_dict())
        if route == "/overview":
            explored = service.explore(ExploreSpec(
                view=_one(params, "view", "architecture") or "architecture",
                depth=_integer(params, "depth", 1, 0, 8),
                top_k=_integer(params, "top_k", 12, 1, 50),
            )).to_dict()
            return self._ok({
                "schema_version": SCHEMA_VERSION,
                "snapshot_id": service.snapshot_id,
                "status": service.status().to_dict(),
                "architecture": explored,
            })
        if route == "/graph":
            return self._ok(service.graph().to_dict())
        if route == "/entities":
            query = _one(params, "q")
            if query is not None:
                return self._ok(service.query(QuerySpec(
                    query=query,
                    kinds=_csv(params, "kinds"),
                    scope_id=_one(params, "scope_id"),
                    stages=_csv(params, "stages"),
                    authorities=_csv(params, "authorities"),
                    limit=_integer(params, "limit", 20, 1, 100),
                    cursor=_one(params, "cursor"),
                )).to_dict())
            graph = service.graph()
            values = sorted(graph.entities.values(), key=lambda item: item.id)
            kinds = set(_csv(params, "kinds"))
            stages = set(_csv(params, "stages"))
            authorities = set(_csv(params, "authorities"))
            if kinds:
                values = [item for item in values if item.kind in kinds]
            if stages:
                values = [item for item in values if item.stage in stages]
            if authorities:
                values = [item for item in values if str(item.authority) in authorities]
            return self._ok(self._page(service.snapshot_id, values, params))
        if route.startswith("/entities/"):
            entity_id = unquote(route[len("/entities/"):])
            entity = service.graph().entities.get(entity_id)
            if entity is None:
                raise KeyError(entity_id)
            return self._ok({"schema_version": SCHEMA_VERSION,
                             "snapshot_id": service.snapshot_id,
                             "entity": json_ready(entity)})
        if route == "/observations":
            values = self.bundle.store.observations(
                service.snapshot_id, subject_id=_one(params, "subject_id"),
                predicate=_one(params, "predicate"))
            stage = _one(params, "stage")
            authority = _one(params, "authority")
            if stage:
                values = [item for item in values if item.stage == stage]
            if authority:
                values = [item for item in values if str(item.authority) == authority]
            return self._ok(self._page(service.snapshot_id, values, params))
        if route == "/diagnostics":
            values = (self.bundle.store.diagnostics(service.snapshot_id)
                      if _boolean(params, "history", False)
                      else self.bundle.store.active_diagnostics(service.snapshot_id))
            severity = _one(params, "severity")
            if severity:
                values = [item for item in values if str(item.severity) == severity]
            return self._ok(self._page(service.snapshot_id, values, params))
        if route == "/runs":
            values = self.bundle.store.runs(service.snapshot_id)
            stage = _one(params, "stage")
            status = _one(params, "status")
            if stage:
                values = [item for item in values if item.stage == stage]
            if status:
                values = [item for item in values if str(item.status) == status]
            return self._ok(self._page(
                service.snapshot_id, [_public_run(item) for item in values], params
            ))
        if route == "/artifacts":
            return self._ok(self._page(
                service.snapshot_id, self.bundle.store.artifacts(service.snapshot_id), params
            ))
        if route.startswith("/artifacts/"):
            artifact_id = unquote(route[len("/artifacts/"):])
            artifact = next((item for item in self.bundle.store.artifacts(service.snapshot_id)
                             if item.id == artifact_id), None)
            if artifact is None:
                raise KeyError(artifact_id)
            return self._ok({"schema_version": SCHEMA_VERSION,
                             "snapshot_id": service.snapshot_id,
                             "artifact": json_ready(artifact),
                             "content_embedded": False})
        if route == "/derivations":
            return self._ok(self._page(
                service.snapshot_id, self.bundle.store.derivations(service.snapshot_id), params
            ))
        if route == "/verifications":
            return self._ok(self._page(
                service.snapshot_id, self.bundle.store.verifications(service.snapshot_id), params
            ))
        if route == "/predictions":
            return self._ok(self._page(
                service.snapshot_id, self.bundle.store.predictions(service.snapshot_id), params
            ))
        if route == "/variants":
            parent = _one(params, "parent_snapshot_id") or service.snapshot_id
            return self._ok(self._page(
                service.snapshot_id,
                self.bundle.store.variants(parent), params
            ))
        if route == "/knowledge":
            from ..knowledge import filter_rules
            applicability = {key: value for key, value in {
                "vendor": _one(params, "vendor"), "tool": _one(params, "tool"),
                "tool_version": _one(params, "tool_version"), "stage": _one(params, "stage"),
            }.items() if value is not None}
            values = filter_rules(
                self.bundle.store.knowledge_rules(),
                document_id=_one(params, "document_id"),
                document_version=_one(params, "document_version"),
                applicability=applicability or None,
            )
            query = _one(params, "q")
            if query:
                folded = query.casefold()
                values = [item for item in values if folded in " ".join(filter(None, [
                    item.id, item.title, item.summary, item.section,
                ])).casefold()]
            response = self._page(service.snapshot_id, values, params)
            response.update({"authority_class": "knowledge_rule",
                             "applicability_context": applicability})
            return self._ok(response)
        if route == "/evidence":
            entity_id = _one(params, "entity_id")
            if not entity_id:
                raise ValueError("query parameter 'entity_id' is required")
            return self._ok(service.evidence(entity_id))
        if route.startswith("/evidence/"):
            return self._ok(service.evidence(unquote(route[len("/evidence/"):])))
        if route == "/compare":
            other = _one(params, "other_snapshot_id")
            if not other:
                raise ValueError("query parameter 'other_snapshot_id' is required")
            return self._ok(service.compare(other))
        return self._error(404, "not_found", "unknown API route")

    @staticmethod
    def _page(snapshot_id: str | None, values: list[Any],
              params: Mapping[str, list[str]]) -> dict[str, Any]:
        limit = _integer(params, "limit", 100, 1, 100)
        offset = _integer(params, "offset", 0, 0, 2_147_483_647)
        page = values[offset:offset + limit]
        next_offset = offset + len(page)
        return {
            "schema_version": SCHEMA_VERSION,
            "snapshot_id": snapshot_id,
            "items": [json_ready(item) for item in page],
            "next_offset": next_offset if next_offset < len(values) else None,
            "truncated": next_offset < len(values),
            "total": len(values),
        }


def openapi_document() -> dict[str, Any]:
    """Return a compact OpenAPI 3.1 contract containing read-only operations only."""
    get_paths: dict[str, tuple[str, str]] = {
        "/api/v1/status": ("status", "Bundle, graph, run, and completeness status"),
        "/api/v1/overview": ("overview", "Architecture overview backed by CoreService.explore"),
        "/api/v1/graph": ("graph", "Canonical graph for one immutable snapshot"),
        "/api/v1/entities": ("entities", "List or search canonical entities"),
        "/api/v1/entities/{entity_id}": ("entity", "Read one canonical entity"),
        "/api/v1/observations": ("observations", "Tool observations with stage and authority"),
        "/api/v1/diagnostics": ("diagnostics", "Extraction and tool diagnostics"),
        "/api/v1/runs": ("runs", "Immutable tool-run ledger entries"),
        "/api/v1/artifacts": ("artifacts", "Content-addressed artifact metadata without private bodies"),
        "/api/v1/artifacts/{artifact_id}": ("artifact", "Read one artifact metadata record"),
        "/api/v1/derivations": ("derivations", "Deterministic derivations with cited inputs"),
        "/api/v1/verifications": ("verifications", "Independent correctness verification evidence"),
        "/api/v1/predictions": ("predictions", "Prediction envelopes kept outside fact tables"),
        "/api/v1/variants": ("variants", "Recorded candidate actions without applying edits"),
        "/api/v1/knowledge": ("knowledge", "Versioned guidance rules, not design facts"),
        "/api/v1/evidence": ("evidence", "Evidence and artifact metadata for one entity"),
        "/api/v1/compare": ("compare", "Compare two immutable snapshots"),
        "/api/v1/openapi.json": ("openapi", "This OpenAPI document"),
    }
    query_parameters: dict[str, list[tuple[str, bool, str, str]]] = {
        "/api/v1/overview": [
            ("view", False, "string", "architecture or evidence view"),
            ("depth", False, "integer", "traversal depth, 0..8"),
            ("top_k", False, "integer", "maximum overview roots, 1..50"),
        ],
        "/api/v1/entities": [
            ("q", False, "string", "search text; omit to list"),
            ("kinds", False, "string", "comma-separated exact kinds"),
            ("scope_id", False, "string", "containing entity ID"),
            ("stages", False, "string", "comma-separated stages"),
            ("authorities", False, "string", "comma-separated authority classes"),
            ("cursor", False, "string", "opaque search cursor"),
        ],
        "/api/v1/observations": [
            ("subject_id", False, "string", "exact subject ID"),
            ("predicate", False, "string", "exact predicate"),
            ("stage", False, "string", "exact stage"),
            ("authority", False, "string", "exact authority class"),
        ],
        "/api/v1/diagnostics": [
            ("severity", False, "string", "exact severity"),
            ("history", False, "boolean", "include diagnostics from superseded attempts"),
        ],
        "/api/v1/runs": [
            ("stage", False, "string", "exact run stage"),
            ("status", False, "string", "exact run status"),
        ],
        "/api/v1/variants": [
            ("parent_snapshot_id", False, "string", "parent snapshot; defaults to selected"),
        ],
        "/api/v1/knowledge": [
            ("q", False, "string", "rule title/summary/section search"),
            ("document_id", False, "string", "exact document ID"),
            ("document_version", False, "string", "exact document version"),
            ("vendor", False, "string", "applicability vendor"),
            ("tool", False, "string", "applicability tool"),
            ("tool_version", False, "string", "applicability tool version"),
            ("stage", False, "string", "applicability stage"),
        ],
        "/api/v1/evidence": [
            ("entity_id", True, "string", "entity to trace"),
        ],
        "/api/v1/compare": [
            ("other_snapshot_id", True, "string", "snapshot to compare against"),
        ],
    }
    paged = {
        "/api/v1/entities", "/api/v1/observations", "/api/v1/diagnostics",
        "/api/v1/runs", "/api/v1/artifacts", "/api/v1/derivations",
        "/api/v1/verifications", "/api/v1/predictions", "/api/v1/variants",
        "/api/v1/knowledge",
    }
    paths: dict[str, Any] = {}
    for path, (operation_id, summary) in get_paths.items():
        parameters: list[dict[str, Any]] = []
        path_parameter = "entity_id" if "{entity_id}" in path else (
            "artifact_id" if "{artifact_id}" in path else None
        )
        if path_parameter:
            parameters.append({"name": path_parameter, "in": "path", "required": True,
                               "schema": {"type": "string"}})
        if path != "/api/v1/openapi.json":
            parameters.append({"name": "snapshot_id", "in": "query", "required": False,
                               "schema": {"type": "string"}})
        for name, required, kind, description in query_parameters.get(path, []):
            parameters.append({"name": name, "in": "query", "required": required,
                               "description": description, "schema": {"type": kind}})
        if path in paged:
            parameters.extend([
                {"name": "limit", "in": "query", "required": False,
                 "schema": {"type": "integer", "minimum": 1, "maximum": 100}},
                {"name": "offset", "in": "query", "required": False,
                 "schema": {"type": "integer", "minimum": 0}},
            ])
        paths[path] = {"get": {
            "operationId": operation_id,
            "summary": summary,
            "parameters": parameters,
            "responses": {
                "200": {"description": "Successful read",
                        "content": {"application/json": {"schema": {"type": "object"}}}},
                "400": {"description": "Invalid request"},
                "404": {"description": "Unknown snapshot, entity, or route"},
                "405": {"description": "Mutation attempted against the read-only API"},
                "409": {"description": "Selected snapshot has no successful graph view"},
            },
        }}
    return {
        "openapi": "3.1.0",
        "info": {
            "title": "HLSGraph read-only API",
            "version": __version__,
            "description": ("Deterministic HLS facts and evidence. Predictions are never "
                            "promoted to tool observations."),
        },
        "servers": [{"url": "http://127.0.0.1:8000"}],
        "paths": paths,
    }


def make_handler(application: RestApplication) -> type[BaseHTTPRequestHandler]:
    class HLSGraphRequestHandler(BaseHTTPRequestHandler):
        server_version = "HLSGraph/0.1"
        protocol_version = "HTTP/1.1"

        def _respond(self) -> None:
            response = application.dispatch(self.command, self.path)
            payload = response.encoded()
            self.send_response(response.status)
            for name, value in response.headers.items():
                self.send_header(name, value)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        do_GET = _respond
        do_HEAD = _respond
        do_POST = _respond
        do_PUT = _respond
        do_PATCH = _respond
        do_DELETE = _respond
        do_OPTIONS = _respond

        def log_message(self, format: str, *args: Any) -> None:  # pragma: no cover - host policy
            return

    return HLSGraphRequestHandler


def _is_loopback(host: str) -> bool:
    if host.casefold() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def serve(project_root: str | Path, host: str = "127.0.0.1", port: int = 8000,
          *, snapshot_id: str | None = None, allow_remote: bool = False) -> None:
    """Serve until interrupted; external binding requires explicit opt-in."""
    if not allow_remote and not _is_loopback(host):
        raise ValueError("non-loopback REST binding requires allow_remote=True")
    if not 0 <= int(port) <= 65535:
        raise ValueError("port must be in 0..65535")
    application = RestApplication(project_root, snapshot_id=snapshot_id)
    server = ThreadingHTTPServer((host, int(port)), make_handler(application))
    try:
        server.serve_forever()
    finally:
        server.server_close()


__all__ = ["API_PREFIX", "ApiResponse", "RestApplication", "make_handler",
           "openapi_document", "serve"]
