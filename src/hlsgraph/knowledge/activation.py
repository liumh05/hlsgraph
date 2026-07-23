"""Instance-local authorization for executable knowledge bindings.

Knowledge binding dictionaries and target names are public metadata.  They are
useful for pack authoring, but possession of matching scalar values must never
authorize guidance for a design instance.  This module provides the short-lived
object-capability boundary used by retrieval: raw contexts are registered for
one exact snapshot/view/scope and accepted only by the issuing session.

The session is deliberately process-local and non-serializable.  It retains
canonical bytes and hashes for bindings, rules, and issued values.  Production
matching happens inside :meth:`BindingActivationSession.evaluate_atomically`;
no caller-constructible or caller-mutable snapshot is later accepted as
authority.  Inspection helpers return fresh detached copies only.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from threading import RLock
from types import MappingProxyType
from typing import Any, Callable, Mapping, NamedTuple, Sequence, TypeVar

from ..model import KnowledgeBinding, KnowledgeRule, json_ready, stable_hash


BINDING_ACTIVATION_CONTRACT = "hlsgraph.knowledge_binding_activation.v1"

_ResultT = TypeVar("_ResultT")


def _binding_payload(binding: KnowledgeBinding) -> dict[str, Any]:
    payload = json_ready(binding)
    if not isinstance(payload, dict):
        raise TypeError("knowledge binding must serialize to an object")
    return payload


def _rule_payload(rule: KnowledgeRule) -> dict[str, Any]:
    payload = json_ready(rule)
    if not isinstance(payload, dict):
        raise TypeError("knowledge rule must serialize to an object")
    return payload


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _binding_fingerprint(binding: KnowledgeBinding) -> str:
    return stable_hash(_binding_payload(binding))


def _rule_fingerprint(rule: KnowledgeRule) -> str:
    return stable_hash(_rule_payload(rule))


def _binding_from_bytes(payload: bytes) -> KnowledgeBinding:
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, dict):
        raise TypeError("canonical knowledge binding is not an object")
    return KnowledgeBinding.from_dict(value)


def _rule_from_bytes(payload: bytes) -> KnowledgeRule:
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, dict):
        raise TypeError("canonical knowledge rule is not an object")
    return KnowledgeRule(**value)


def _freeze_context(
    context: Mapping[str, set[str]],
) -> Mapping[str, frozenset[str]]:
    frozen: dict[str, frozenset[str]] = {}
    for raw_key, raw_values in context.items():
        if not isinstance(raw_key, str) or not isinstance(raw_values, set):
            raise TypeError("binding context must map strings to sets of strings")
        if any(not isinstance(item, str) for item in raw_values):
            raise TypeError("binding context values must be strings")
        frozen[raw_key] = frozenset(raw_values)
    return MappingProxyType(frozen)


def _context_payload(
    context: Mapping[str, frozenset[str]],
) -> dict[str, list[str]]:
    return {
        str(key): sorted(str(item) for item in values)
        for key, values in sorted(context.items())
    }


def _context_fingerprint(context: Mapping[str, frozenset[str]]) -> str:
    return stable_hash(_context_payload(context))


def _context_from_bytes(payload: bytes) -> dict[str, set[str]]:
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, dict):
        raise TypeError("canonical binding context is not an object")
    result: dict[str, set[str]] = {}
    for raw_key, raw_values in value.items():
        if not isinstance(raw_key, str) or not isinstance(raw_values, list):
            raise TypeError("canonical binding context has an invalid shape")
        if any(not isinstance(item, str) for item in raw_values):
            raise TypeError("canonical binding context values must be strings")
        result[raw_key] = set(raw_values)
    return result


@dataclass(frozen=True, slots=True)
class AttestedBindingContext:
    """One non-serializable context issued by an active retrieval session.

    Frozen dataclasses are a convenience, not the authority boundary.  Every
    field and the exact object identity are revalidated against private
    canonical bytes before and after the session-local evaluator runs.
    """

    values: Mapping[str, frozenset[str]]
    values_fingerprint: str
    target_kind: str
    target: str
    binding_id: str
    binding_fingerprint: str
    rule_id: str
    rule_fingerprint: str
    snapshot_id: str
    graph_hash: str
    scope_hash: str
    contract: str = BINDING_ACTIVATION_CONTRACT
    _issuer: "BindingActivationSession" = field(
        repr=False, compare=False, hash=False, default=None,  # type: ignore[assignment]
    )


class _IssuedRecord(NamedTuple):
    """Private immutable claims for one issued handle.

    The visible fields on :class:`AttestedBindingContext` are never used to
    choose another binding or rule.  They are only checked against this tuple.
    """

    handle: AttestedBindingContext
    values_fingerprint: str
    values: Mapping[str, frozenset[str]]
    values_bytes: bytes
    raw_context: Mapping[str, set[str]]
    binding_id: str
    binding_fingerprint: str
    rule_id: str
    rule_fingerprint: str
    target_kind: str
    target: str
    snapshot_id: str
    graph_hash: str
    scope_hash: str
    contract: str
    claims_hash: str


def _issued_claims_hash(
    *,
    values_fingerprint: str,
    values_bytes: bytes,
    binding_id: str,
    binding_fingerprint: str,
    rule_id: str,
    rule_fingerprint: str,
    target_kind: str,
    target: str,
    snapshot_id: str,
    graph_hash: str,
    scope_hash: str,
    contract: str,
) -> str:
    return stable_hash({
        "values_fingerprint": values_fingerprint,
        "values_sha256": hashlib.sha256(values_bytes).hexdigest(),
        "binding_id": binding_id,
        "binding_fingerprint": binding_fingerprint,
        "rule_id": rule_id,
        "rule_fingerprint": rule_fingerprint,
        "target_kind": target_kind,
        "target": target,
        "snapshot_id": snapshot_id,
        "graph_hash": graph_hash,
        "scope_hash": scope_hash,
        "contract": contract,
    })


class BindingActivationSession:
    """Short-lived issuer and atomic evaluator for binding contexts."""

    def __init__(
        self,
        *,
        snapshot_id: str,
        graph_hash: str,
        allowed_ids: Sequence[str],
        bindings: Sequence[KnowledgeBinding],
        rules: Sequence[KnowledgeRule],
        expected_binding_fingerprints: Mapping[str, str] | None = None,
        expected_rule_fingerprints: Mapping[str, str] | None = None,
        raw_contexts: Mapping[
            tuple[str, str], Sequence[Mapping[str, set[str]]]
        ],
    ) -> None:
        self._lock = RLock()
        self._snapshot_id = str(snapshot_id)
        self._graph_hash = str(graph_hash)
        self._scope_hash = stable_hash(sorted(str(item) for item in allowed_ids))
        self._active = True
        # issued id -> (object, values hash, exact frozen values, canonical
        # bytes, exact raw source object).  The source is re-hashed at use so
        # issuing cannot be separated from a later caller mutation.
        self._issued: dict[int, _IssuedRecord] = {}
        self._raw: dict[
            tuple[str, str], dict[int, Mapping[str, set[str]]]
        ] = {
            (str(kind), str(target)): {id(item): item for item in contexts}
            for (kind, target), contexts in raw_contexts.items()
        }
        # Registries retain immutable canonical bytes, never a frozen object
        # later exposed to a caller.
        self._bindings: dict[
            str, tuple[KnowledgeBinding, str, bytes]
        ] = {}
        self._rules: dict[
            str, tuple[KnowledgeRule, str, bytes]
        ] = {}
        binding_surface_required = expected_binding_fingerprints is not None
        rule_surface_required = expected_rule_fingerprints is not None
        expected_binding_fingerprints = dict(expected_binding_fingerprints or {})
        expected_rule_fingerprints = dict(expected_rule_fingerprints or {})
        try:
            supplied_binding_ids = {item.id for item in bindings}
            supplied_rule_ids = {item.id for item in rules}
        except Exception as exc:
            self.close()
            raise ValueError("knowledge activation surface is malformed") from exc
        if binding_surface_required and (
            set(expected_binding_fingerprints) != supplied_binding_ids
        ):
            self.close()
            raise ValueError("reviewed binding fingerprint surface is incomplete")
        if rule_surface_required and (
            set(expected_rule_fingerprints) != supplied_rule_ids
        ):
            self.close()
            raise ValueError("reviewed rule fingerprint surface is incomplete")
        for binding in bindings:
            try:
                binding_id = binding.id
                if binding_id in self._bindings:
                    self.close()
                    raise ValueError(
                        "binding activation surface contains duplicate IDs"
                    )
                payload = _binding_payload(binding)
                fingerprint = stable_hash(payload)
                canonical = _canonical_bytes(payload)
            except ValueError:
                raise
            except Exception as exc:
                self.close()
                raise ValueError("knowledge binding is malformed") from exc
            if (binding_surface_required
                    and expected_binding_fingerprints.get(binding_id) != fingerprint):
                self.close()
                raise ValueError("reviewed binding changed before activation")
            self._bindings[binding_id] = (
                binding, fingerprint, canonical,
            )
        for rule in rules:
            try:
                rule_id = rule.id
                if rule_id in self._rules:
                    self.close()
                    raise ValueError(
                        "binding activation surface contains duplicate rule IDs"
                    )
                payload = _rule_payload(rule)
                fingerprint = stable_hash(payload)
                canonical = _canonical_bytes(payload)
            except ValueError:
                raise
            except Exception as exc:
                self.close()
                raise ValueError("knowledge rule is malformed") from exc
            if (rule_surface_required
                    and expected_rule_fingerprints.get(rule_id) != fingerprint):
                self.close()
                raise ValueError("reviewed rule changed before activation")
            self._rules[rule_id] = (
                rule, fingerprint, canonical,
            )

    @property
    def active(self) -> bool:
        with self._lock:
            return self._active

    @property
    def snapshot_id(self) -> str:
        return self._snapshot_id

    @property
    def graph_hash(self) -> str:
        return self._graph_hash

    @property
    def scope_hash(self) -> str:
        return self._scope_hash

    def issue(
        self,
        binding: KnowledgeBinding,
        raw_context: Mapping[str, set[str]],
    ) -> AttestedBindingContext | None:
        """Freeze one exact registered target context for one exact binding."""

        with self._lock:
            if not self._active:
                return None
            try:
                binding_id = binding.id
                rule_id = binding.knowledge_rule_id
                target_kind = str(binding.target_kind)
                target = str(binding.target)
                registered_binding = self._bindings.get(binding_id)
                rule = self._rules.get(rule_id)
                bucket = self._raw.get((target_kind, target))
                binding_fingerprint = _binding_fingerprint(binding)
                live_rule_id = rule[0].id if rule is not None else None
                live_rule_fingerprint = (
                    _rule_fingerprint(rule[0]) if rule is not None else None
                )
                snapshot_id = self.snapshot_id
                graph_hash = self.graph_hash
                scope_hash = self.scope_hash
            except Exception:
                return None
            if (
                registered_binding is None
                or registered_binding[0] is not binding
                or registered_binding[1] != binding_fingerprint
                or rule is None
                or live_rule_id != rule_id
                or rule[1] != live_rule_fingerprint
                or bucket is None
                or bucket.get(id(raw_context)) is not raw_context
            ):
                return None
            try:
                frozen_values = _freeze_context(raw_context)
                values_fingerprint = _context_fingerprint(frozen_values)
                values_bytes = _canonical_bytes(_context_payload(frozen_values))
            except Exception:
                return None
            context = AttestedBindingContext(
                values=frozen_values,
                values_fingerprint=values_fingerprint,
                target_kind=target_kind,
                target=target,
                binding_id=binding_id,
                binding_fingerprint=registered_binding[1],
                rule_id=rule_id,
                rule_fingerprint=rule[1],
                snapshot_id=snapshot_id,
                graph_hash=graph_hash,
                scope_hash=scope_hash,
                _issuer=self,
            )
            try:
                claims_hash = _issued_claims_hash(
                    values_fingerprint=values_fingerprint,
                    values_bytes=values_bytes,
                    binding_id=binding_id,
                    binding_fingerprint=registered_binding[1],
                    rule_id=rule_id,
                    rule_fingerprint=rule[1],
                    target_kind=target_kind,
                    target=target,
                    snapshot_id=snapshot_id,
                    graph_hash=graph_hash,
                    scope_hash=scope_hash,
                    contract=BINDING_ACTIVATION_CONTRACT,
                )
            except Exception:
                return None
            self._issued[id(context)] = _IssuedRecord(
                handle=context,
                values_fingerprint=values_fingerprint,
                values=frozen_values,
                values_bytes=values_bytes,
                raw_context=raw_context,
                binding_id=binding_id,
                binding_fingerprint=registered_binding[1],
                rule_id=rule_id,
                rule_fingerprint=rule[1],
                target_kind=target_kind,
                target=target,
                snapshot_id=snapshot_id,
                graph_hash=graph_hash,
                scope_hash=scope_hash,
                contract=BINDING_ACTIVATION_CONTRACT,
                claims_hash=claims_hash,
            )
            return context

    def _validated_records(
        self,
        binding: KnowledgeBinding,
        context: Any,
    ) -> tuple[
        tuple[KnowledgeBinding, str, bytes],
        tuple[KnowledgeRule, str, bytes],
        _IssuedRecord,
    ] | None:
        """Validate identities and canonical hashes while ``self._lock`` is held."""

        if (
            not self._active
            or not isinstance(context, AttestedBindingContext)
            or context._issuer is not self
        ):
            return None
        issued = self._issued.get(id(context))
        if (
            issued is None
            or issued.handle is not context
            or context.values is not issued.values
            or context.contract != issued.contract
            or context.snapshot_id != issued.snapshot_id
            or context.graph_hash != issued.graph_hash
            or context.scope_hash != issued.scope_hash
            or context.binding_id != issued.binding_id
            or context.binding_fingerprint != issued.binding_fingerprint
            or context.rule_id != issued.rule_id
            or context.rule_fingerprint != issued.rule_fingerprint
            or context.target_kind != issued.target_kind
            or context.target != issued.target
            or binding.id != issued.binding_id
            or binding.knowledge_rule_id != issued.rule_id
            or issued.snapshot_id != self._snapshot_id
            or issued.graph_hash != self._graph_hash
            or issued.scope_hash != self._scope_hash
            or issued.contract != BINDING_ACTIVATION_CONTRACT
        ):
            return None
        registered_binding = self._bindings.get(issued.binding_id)
        rule = self._rules.get(issued.rule_id)
        bucket = self._raw.get((issued.target_kind, issued.target))
        try:
            raw_bytes = _canonical_bytes(
                _context_payload(_freeze_context(issued.raw_context))
            )
        except (TypeError, ValueError, RuntimeError):
            return None
        expected_claims_hash = _issued_claims_hash(
            values_fingerprint=issued.values_fingerprint,
            values_bytes=issued.values_bytes,
            binding_id=issued.binding_id,
            binding_fingerprint=issued.binding_fingerprint,
            rule_id=issued.rule_id,
            rule_fingerprint=issued.rule_fingerprint,
            target_kind=issued.target_kind,
            target=issued.target,
            snapshot_id=issued.snapshot_id,
            graph_hash=issued.graph_hash,
            scope_hash=issued.scope_hash,
            contract=issued.contract,
        )
        if not (
            registered_binding is not None
            and registered_binding[0] is binding
            and registered_binding[1] == issued.binding_fingerprint
            and registered_binding[1] == _binding_fingerprint(binding)
            and registered_binding[2] == _canonical_bytes(_binding_payload(binding))
            and rule is not None
            and rule[0].id == issued.rule_id
            and rule[1] == issued.rule_fingerprint
            and rule[1] == _rule_fingerprint(rule[0])
            and rule[2] == _canonical_bytes(_rule_payload(rule[0]))
            and issued.values_fingerprint == context.values_fingerprint
            and issued.values_fingerprint == _context_fingerprint(issued.values)
            and issued.values_bytes == _canonical_bytes(
                _context_payload(issued.values)
            )
            and bucket is not None
            and bucket.get(id(issued.raw_context)) is issued.raw_context
            and issued.values_bytes == raw_bytes
            and issued.claims_hash == expected_claims_hash
        ):
            return None
        return registered_binding, rule, issued

    def _safe_validated_records(
        self,
        binding: KnowledgeBinding,
        context: Any,
    ) -> tuple[
        tuple[KnowledgeBinding, str, bytes],
        tuple[KnowledgeRule, str, bytes],
        _IssuedRecord,
    ] | None:
        try:
            return self._validated_records(binding, context)
        except Exception:
            # Mutable/corrupt records are untrusted input.  They remove
            # executable guidance instead of aborting the fact retrieval.
            return None

    def validate(
        self,
        binding: KnowledgeBinding,
        context: Any,
    ) -> bool:
        """Prove current-session issuance and reject mutation or replay."""

        with self._lock:
            return self._safe_validated_records(binding, context) is not None

    def evaluate_atomically(
        self,
        binding: KnowledgeBinding,
        context: AttestedBindingContext,
        evaluator: Callable[
            [KnowledgeBinding, KnowledgeRule, Mapping[str, set[str]]], _ResultT
        ],
    ) -> _ResultT | None:
        """Validate, evaluate canonical detached values, then revalidate.

        The evaluator's return value is ordinary output, not a reusable
        authorization envelope.  No method in this module accepts it as
        authority.  The lock closes same-session issue/close races; the second
        validation closes mutation of the live rule or binding during a
        callback.  Evaluation itself receives only fresh decoded copies.
        """

        with self._lock:
            records = self._safe_validated_records(binding, context)
            if records is None:
                return None
            registered_binding, rule, issued = records
            try:
                detached_binding = _binding_from_bytes(registered_binding[2])
                detached_rule = _rule_from_bytes(rule[2])
                detached_values = _context_from_bytes(issued.values_bytes)
            except Exception:
                return None
            result = evaluator(detached_binding, detached_rule, detached_values)
            if self._safe_validated_records(binding, context) is None:
                return None
            return result

    def _validated_rule_record(
        self, rule: KnowledgeRule,
    ) -> tuple[KnowledgeRule, str, bytes] | None:
        if not self._active:
            return None
        registered = self._rules.get(rule.id)
        if (
            registered is None
            or registered[0] is not rule
            or registered[1] != _rule_fingerprint(rule)
            or registered[2] != _canonical_bytes(_rule_payload(rule))
        ):
            return None
        return registered

    def _safe_validated_rule_record(
        self, rule: KnowledgeRule,
    ) -> tuple[KnowledgeRule, str, bytes] | None:
        try:
            return self._validated_rule_record(rule)
        except Exception:
            return None

    def evaluate_rule_atomically(
        self,
        rule: KnowledgeRule,
        evaluator: Callable[[KnowledgeRule], _ResultT],
    ) -> _ResultT | None:
        """Evaluate one reviewed rule without exposing an authority object."""

        with self._lock:
            registered = self._safe_validated_rule_record(rule)
            if registered is None:
                return None
            try:
                detached_rule = _rule_from_bytes(registered[2])
            except Exception:
                return None
            result = evaluator(detached_rule)
            if self._safe_validated_rule_record(rule) is None:
                return None
            return result

    def rule_snapshot_for(
        self, rule: KnowledgeRule,
    ) -> KnowledgeRule | None:
        """Return a fresh non-authorizing copy of one unchanged stored rule."""

        return self.evaluate_rule_atomically(rule, lambda detached: detached)

    def binding_snapshot_for(
        self,
        binding: KnowledgeBinding,
        context: AttestedBindingContext,
    ) -> KnowledgeBinding | None:
        """Return a fresh non-authorizing copy after atomic validation."""

        return self.evaluate_atomically(
            binding, context, lambda detached, _rule, _values: detached,
        )

    def condition_for(
        self,
        binding: KnowledgeBinding,
        context: AttestedBindingContext,
    ) -> Mapping[str, Any] | None:
        """Return a fresh non-authorizing condition mapping."""

        return self.evaluate_atomically(
            binding, context,
            lambda _binding, rule, _values: dict(rule.condition),
        )

    def close(self) -> None:
        with self._lock:
            self._active = False
            self._issued.clear()
            self._raw.clear()
            self._bindings.clear()
            self._rules.clear()


__all__ = [
    "BINDING_ACTIVATION_CONTRACT",
    "AttestedBindingContext",
    "BindingActivationSession",
]
