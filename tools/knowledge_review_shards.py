#!/usr/bin/env python3
"""Closed three-shard plan for the public knowledge review.

This module is intentionally independent from ``run_knowledge_review.py``.
It defines the review input partition, assertion ownership, exact rule
reference allocation, deterministic plan hashing, and the fail-closed token
budget contract.  It does not execute a reviewer or seal an attestation.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import importlib.metadata
import json
import re
from typing import Any, Callable, Mapping, Sequence


PLAN_SCHEMA_VERSION = "hlsgraph.knowledge-review.shard-plan.v1"
SEMANTIC_PROTOCOL_ID = "hlsgraph.knowledge-review.semantic.v1"
ADVERSARIAL_PROTOCOL_ID = "hlsgraph.knowledge-review.adversarial.v1"

SHARD_ORDER = (
    "knowledge_activation",
    "ir_semantics",
    "tool_evidence",
)

MODEL_CONTEXT_WINDOW_TOKENS = 372_000
MAX_VISIBLE_INPUT_TOKENS = 250_000
MIN_CONTEXT_RESERVE_TOKENS = 122_000
TOOL_EVENT_OVERHEAD_TOKENS = 192
# Codex's system/developer envelope and tool wrappers are not emitted by the
# JSONL protocol.  Reserve a fixed conservative allowance inside the 250k
# ceiling instead of pretending those tokens are zero.
RUNTIME_ENVELOPE_ALLOWANCE_TOKENS = 32_000
AUTO_COMPACT_TOKEN_LIMIT_TOKENS = 300_000
AUTO_COMPACT_TOKEN_LIMIT_SCOPE = "total"
DEFAULT_TOKENIZER_ID = "o200k_base"
DEFAULT_TOKENIZER_PACKAGE = "tiktoken"
DEFAULT_TOKENIZER_PACKAGE_VERSION = "0.13.0"
DEFAULT_TOKENIZER_CONTRACT_SHA256 = (
    "02b0af8658a8a3abddcf10a02178f7f5ebeaa12e9fb24c562a71c5dbf288b4c9"
)

MODEL_SOURCE_PROJECTION_SCHEMA_VERSION = (
    "hlsgraph.knowledge-review.model-source-projection.v1"
)
MODEL_SOURCE_PROJECTION_PREFIX = "review-projections/v1"
CITATION_AUDIT_SOURCE_PATH = "docs/knowledge-citation-audit-v0.3.json"
CITATION_EVIDENCE_SOURCE_PATH = "docs/knowledge-review-evidence-v0.3.json"
AMD_PACK_SOURCE_PATH = (
    "src/hlsgraph/knowledge/packs/amd_public_guidance_2024_2.json"
)
AXI_PACK_SOURCE_PATH = "src/hlsgraph/knowledge/packs/axi_public_guidance.json"
OPEN_IR_PACK_SOURCE_PATH = (
    "src/hlsgraph/knowledge/packs/open_ir_public_guidance.json"
)
_PACK_SOURCE_PATHS = (
    AMD_PACK_SOURCE_PATH, AXI_PACK_SOURCE_PATH, OPEN_IR_PACK_SOURCE_PATH,
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class ShardPlanError(ValueError):
    """The citation audit cannot be represented by the closed shard plan."""


class TokenBudgetError(ValueError):
    """The token budget could not be established safely."""


class TokenBudgetExceeded(TokenBudgetError):
    """The fixed visible-input budget was exceeded."""


@dataclass(frozen=True)
class RuleReferenceSpec:
    rule_id: str
    shard_id: str
    citation_url: str
    section: str


@dataclass(frozen=True)
class ShardDefinition:
    shard_id: str
    source_paths: tuple[str, ...]
    semantic_assertion_ids: tuple[str, ...]
    adversarial_assertion_ids: tuple[str, ...]


@dataclass(frozen=True)
class TokenBudgetContract:
    """Pinned context limits plus an explicitly identified tokenizer."""

    tokenizer_id: str = DEFAULT_TOKENIZER_ID
    tokenizer_contract_sha256: str = DEFAULT_TOKENIZER_CONTRACT_SHA256
    context_window_tokens: int = MODEL_CONTEXT_WINDOW_TOKENS
    max_visible_input_tokens: int = MAX_VISIBLE_INPUT_TOKENS
    min_context_reserve_tokens: int = MIN_CONTEXT_RESERVE_TOKENS
    tool_event_overhead_tokens: int = TOOL_EVENT_OVERHEAD_TOKENS
    runtime_envelope_allowance_tokens: int = (
        RUNTIME_ENVELOPE_ALLOWANCE_TOKENS
    )
    auto_compact_token_limit_tokens: int = AUTO_COMPACT_TOKEN_LIMIT_TOKENS
    auto_compact_token_limit_scope: str = AUTO_COMPACT_TOKEN_LIMIT_SCOPE

    def __post_init__(self) -> None:
        if not self.tokenizer_id:
            raise TokenBudgetError("tokenizer_id must be non-empty")
        if _SHA256_RE.fullmatch(self.tokenizer_contract_sha256) is None:
            raise TokenBudgetError(
                "tokenizer_contract_sha256 must be a lowercase SHA-256"
            )
        fixed = (
            self.context_window_tokens == MODEL_CONTEXT_WINDOW_TOKENS
            and self.max_visible_input_tokens == MAX_VISIBLE_INPUT_TOKENS
            and self.min_context_reserve_tokens == MIN_CONTEXT_RESERVE_TOKENS
            and self.tool_event_overhead_tokens == TOOL_EVENT_OVERHEAD_TOKENS
            and self.runtime_envelope_allowance_tokens
            == RUNTIME_ENVELOPE_ALLOWANCE_TOKENS
            and self.auto_compact_token_limit_tokens
            == AUTO_COMPACT_TOKEN_LIMIT_TOKENS
            and self.auto_compact_token_limit_scope
            == AUTO_COMPACT_TOKEN_LIMIT_SCOPE
        )
        if not fixed:
            raise TokenBudgetError(
                "knowledge-review context and reserve limits are fixed"
            )
        if (
            self.context_window_tokens - self.max_visible_input_tokens
            < self.min_context_reserve_tokens
        ):
            raise TokenBudgetError("token contract does not preserve its reserve")
        if not (
            self.max_visible_input_tokens
            < self.auto_compact_token_limit_tokens
            < self.context_window_tokens
        ):
            raise TokenBudgetError("auto-compaction threshold is not safely separated")

    def to_dict(self) -> dict[str, Any]:
        return {
            "tokenizer_id": self.tokenizer_id,
            "tokenizer_contract_sha256": self.tokenizer_contract_sha256,
            "context_window_tokens": self.context_window_tokens,
            "max_visible_input_tokens": self.max_visible_input_tokens,
            "min_context_reserve_tokens": self.min_context_reserve_tokens,
            "tool_event_overhead_tokens": self.tool_event_overhead_tokens,
            "runtime_envelope_allowance_tokens": (
                self.runtime_envelope_allowance_tokens
            ),
            "auto_compact_token_limit_tokens": (
                self.auto_compact_token_limit_tokens
            ),
            "auto_compact_token_limit_scope": self.auto_compact_token_limit_scope,
        }


DEFAULT_TOKEN_BUDGET_CONTRACT = TokenBudgetContract()


def tokenizer_contract_payload(
    encoding: Any, *, package_version: str,
) -> dict[str, Any]:
    """Fingerprint the exact pinned tokenizer tables used for budgeting.

    Merely recording the encoding name is insufficient: a package or table
    change could move a shard across the fixed 250k boundary.  The formal
    runner therefore binds the pattern, every mergeable token/rank pair, and
    the special-token table from the pinned local package.
    """

    if package_version != DEFAULT_TOKENIZER_PACKAGE_VERSION:
        raise TokenBudgetError(
            "formal review requires tiktoken "
            f"{DEFAULT_TOKENIZER_PACKAGE_VERSION}, found {package_version!r}"
        )
    if getattr(encoding, "name", None) != DEFAULT_TOKENIZER_ID:
        raise TokenBudgetError("formal review loaded the wrong tokenizer encoding")
    pattern = getattr(encoding, "_pat_str", None)
    ranks = getattr(encoding, "_mergeable_ranks", None)
    special = getattr(encoding, "_special_tokens", None)
    n_vocab = getattr(encoding, "n_vocab", None)
    max_token_value = getattr(encoding, "max_token_value", None)
    if (
        not isinstance(pattern, str)
        or not isinstance(ranks, dict)
        or not isinstance(special, dict)
        or isinstance(n_vocab, bool)
        or not isinstance(n_vocab, int)
        or isinstance(max_token_value, bool)
        or not isinstance(max_token_value, int)
    ):
        raise TokenBudgetError("tokenizer does not expose the pinned table contract")
    rank_rows: list[tuple[bytes, int]] = []
    for token, rank in ranks.items():
        if (
            not isinstance(token, bytes)
            or isinstance(rank, bool)
            or not isinstance(rank, int)
            or rank < 0
        ):
            raise TokenBudgetError("tokenizer mergeable-rank table is malformed")
        rank_rows.append((token, rank))
    rank_digest = hashlib.sha256()
    for token, rank in sorted(rank_rows, key=lambda item: (item[1], item[0])):
        rank_digest.update(len(token).to_bytes(8, "big"))
        rank_digest.update(token)
        rank_digest.update(rank.to_bytes(8, "big"))
    normalized_special: dict[str, int] = {}
    for token, rank in special.items():
        if (
            not isinstance(token, str)
            or isinstance(rank, bool)
            or not isinstance(rank, int)
            or rank < 0
        ):
            raise TokenBudgetError("tokenizer special-token table is malformed")
        normalized_special[token] = rank
    return {
        "package": DEFAULT_TOKENIZER_PACKAGE,
        "package_version": package_version,
        "encoding_name": DEFAULT_TOKENIZER_ID,
        "n_vocab": n_vocab,
        "max_token_value": max_token_value,
        "pat_str_sha256": hashlib.sha256(pattern.encode("utf-8")).hexdigest(),
        "mergeable_ranks_sha256": rank_digest.hexdigest(),
        "mergeable_rank_count": len(rank_rows),
        "special_tokens": dict(sorted(normalized_special.items())),
    }


def load_verified_tokenizer() -> Any:
    """Load the optional pinned tokenizer and verify its complete contract."""

    try:
        import tiktoken
    except ImportError as exc:  # pragma: no cover - optional formal dependency
        raise TokenBudgetError(
            "formal review requires the pinned 'review' optional dependency"
        ) from exc
    try:
        version = importlib.metadata.version(DEFAULT_TOKENIZER_PACKAGE)
        encoding = tiktoken.get_encoding(DEFAULT_TOKENIZER_ID)
    except Exception as exc:  # pragma: no cover - environment-specific failure
        raise TokenBudgetError("cannot load the pinned formal-review tokenizer") from exc
    payload = tokenizer_contract_payload(encoding, package_version=version)
    digest = hashlib.sha256(_canonical_json(payload)).hexdigest()
    if digest != DEFAULT_TOKENIZER_CONTRACT_SHA256:
        raise TokenBudgetError(
            "formal-review tokenizer tables do not match the pinned contract"
        )
    return encoding


@dataclass(frozen=True)
class TokenBudget:
    contract: TokenBudgetContract
    prompt_tokens: int
    chunk_tokens: int
    command_tokens: int
    tool_event_count: int
    tool_event_overhead_tokens: int
    runtime_envelope_allowance_tokens: int
    visible_input_tokens: int
    context_reserve_tokens: int

    @property
    def within_budget(self) -> bool:
        return (
            self.visible_input_tokens
            <= self.contract.max_visible_input_tokens
            and self.context_reserve_tokens
            >= self.contract.min_context_reserve_tokens
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": self.contract.to_dict(),
            "prompt_tokens": self.prompt_tokens,
            "chunk_tokens": self.chunk_tokens,
            "command_tokens": self.command_tokens,
            "tool_event_count": self.tool_event_count,
            "tool_event_overhead_tokens": self.tool_event_overhead_tokens,
            "runtime_envelope_allowance_tokens": (
                self.runtime_envelope_allowance_tokens
            ),
            "visible_input_tokens": self.visible_input_tokens,
            "context_reserve_tokens": self.context_reserve_tokens,
            "within_budget": self.within_budget,
        }


def model_source_projection_path(shard_id: str, source_path: str) -> str:
    """Return the virtual cache path for one shard-local source projection."""

    if shard_id not in SHARD_ORDER:
        raise ShardPlanError(f"unknown review shard: {shard_id!r}")
    if source_path not in {
        CITATION_AUDIT_SOURCE_PATH, CITATION_EVIDENCE_SOURCE_PATH,
        *_PACK_SOURCE_PATHS,
    }:
        raise ShardPlanError(f"source is not projection-controlled: {source_path!r}")
    return f"{MODEL_SOURCE_PROJECTION_PREFIX}/{shard_id}/{source_path}"


_PROJECTED_SOURCE_PATHS = {
    "knowledge_activation": (
        CITATION_AUDIT_SOURCE_PATH,
        CITATION_EVIDENCE_SOURCE_PATH,
        AMD_PACK_SOURCE_PATH,
        AXI_PACK_SOURCE_PATH,
    ),
    "ir_semantics": (
        CITATION_AUDIT_SOURCE_PATH,
        CITATION_EVIDENCE_SOURCE_PATH,
        OPEN_IR_PACK_SOURCE_PATH,
    ),
    "tool_evidence": (
        CITATION_AUDIT_SOURCE_PATH,
        CITATION_EVIDENCE_SOURCE_PATH,
        AMD_PACK_SOURCE_PATH,
    ),
}


def projected_model_source_paths(shard_id: str) -> tuple[str, ...]:
    """Return only the virtual JSON sources visible to ``shard_id``."""

    try:
        originals = _PROJECTED_SOURCE_PATHS[shard_id]
    except KeyError as exc:
        raise ShardPlanError(f"unknown review shard: {shard_id!r}") from exc
    return tuple(model_source_projection_path(shard_id, path) for path in originals)


def is_model_source_projection_path(path: str) -> bool:
    return isinstance(path, str) and path.startswith(
        MODEL_SOURCE_PROJECTION_PREFIX + "/"
    )


SEMANTIC_ASSERTION_DESCRIPTIONS = {
    "S01.knowledge_guidance_not_fact": (
        "Knowledge guidance never becomes a design fact, tool observation, "
        "verification result, or hardware edge."
    ),
    "S02.binding_condition_instance_local_entailment": (
        "Executable bindings prove the cited rule's complete condition from "
        "the current target instance; absent, conflicting, ambiguous, generic, "
        "or caller-injected context fails closed."
    ),
    "S03.directive_exact_scope_source_operand_proof": (
        "A directive proves its exact instance, scope, source declaration, "
        "anchor, snapshot bytes, options, and any separate operand identity "
        "through authorized deterministic evidence."
    ),
    "S04.fresh_real_run_authorization_receipt": (
        "Tool-backed truth requires a fresh real Local/SSH run, one-use "
        "internal authorization, immutable manifest, and independently "
        "revalidated persisted execution receipt."
    ),
    "S05.canonical_report_artifact_single_anchor": (
        "A typed observation has one canonical declared report and sole "
        "matching anchor with parser-issued predicate/value/unit and exact "
        "byte provenance."
    ),
    "S06.requested_effective_achieved_stage_and_three_gate_separation": (
        "Requested, effective/applied, achieved, estimate, post-synth, "
        "post-route, correctness, resource-fit, and timing states remain "
        "distinct."
    ),
    "S07.language_spec_contract_and_non_topology": (
        "MLIR/LLVM/CIRCT semantics bind an exact compatible language-spec "
        "revision; LLVM CFG, software calls, and native Handshake SSA do not "
        "by themselves become HLS hardware topology."
    ),
    "S08.cross_layer_mapping_unique_typed_anchored": (
        "Cross-layer mappings are typed, anchored, explicit, and uniquely "
        "resolved; ambiguity stays incomplete."
    ),
    "S09.aggregate_static_feature_recomputation": (
        "Aggregate features are recomputed from qualified evidence with "
        "schema, completeness, provenance, artifact, and origin identities; "
        "unknown is not numeric zero."
    ),
    "S10.retrieval_plane_isolation": (
        "Fact/evidence ranking, normalization, graph propagation, and "
        "truncation remain isolated from knowledge, local text, and "
        "predictions; generic adapters cannot claim a trusted plane."
    ),
    "S11.coverage_target_registry_activation_gate": (
        "Only reviewed rule coverage can activate executable guidance; "
        "binding and supported-target inventories are exact, while "
        "citation_only and no_normative are non-executable."
    ),
}


ADVERSARIAL_ASSERTION_DESCRIPTIONS = {
    "A01.reserved_metadata_injection": (
        "Attempt to inject trusted-looking authority, provenance, binding, or "
        "plane tokens through mutable metadata."
    ),
    "A02.directive_scope_operand_source_forgery": (
        "Attempt to forge directive scope, anchor, source bytes, options, "
        "operand, requested observation, or replay proof."
    ),
    "A03.review_ready_install_select_bypass": (
        "Attempt to install, select, or activate an executable binding while "
        "its pack is not review_ready."
    ),
    "A04.incomplete_condition_or_cross_target_context": (
        "Attempt to weaken condition entailment or satisfy it with missing, "
        "ambiguous, or another target's context."
    ),
    "A05.coverage_registry_concealment": (
        "Attempt to hide bindings in non-rule coverage or add/omit rules, "
        "bindings, sections, or supported targets."
    ),
    "A06.generic_container_unknown_boolean_activation": (
        "Attempt activation through generic report containers, bare "
        "constraints, unknown boolean spellings, or partial evidence."
    ),
    "A07.spec_revision_compatibility_forgery": (
        "Attempt to claim MLIR/LLVM/CIRCT compatibility for an unbound or "
        "mismatched artifact/spec revision."
    ),
    "A08.graph_metadata_semantic_attestation": (
        "Attempt to mint a trusted semantic attestation from a "
        "caller-constructed object in mutable graph metadata."
    ),
    "A09.llvm_cfg_software_call_handshake_topology_promotion": (
        "Attempt to promote LLVM CFG, software calls, or native Handshake SSA "
        "into canonical hardware topology."
    ),
    "A10.ambiguous_cross_layer_mapping_promotion": (
        "Attempt to turn absent, duplicated, conflicting, or ambiguous "
        "cross-layer mappings into normative edges."
    ),
    "A11.aggregate_feature_spoof": (
        "Attempt to inject or aggregate operations, widths, indices, memory "
        "accesses, bounds, or dependence values without qualified complete "
        "origin evidence."
    ),
    "A12.retrieval_plane_rank_budget_pollution": (
        "Attempt to let knowledge, local text, or predictions perturb "
        "fact/evidence candidates, BM25, propagation, or budget."
    ),
    "A13.fabricated_sdk_toolrun_runner_receipt": (
        "Attempt to fabricate a successful ToolRun, runner identity, request, "
        "environment, manifest, staged output, attestation, or commit receipt "
        "through public SDK/store calls."
    ),
    "A14.one_use_authorization_or_receipt_replay": (
        "Attempt to reuse an execution authorization, attestation, or commit "
        "receipt more than once."
    ),
    "A15.sibling_snapshot_run_stage_artifact_workload_reuse": (
        "Attempt to donate evidence across instance, snapshot, run, stage, "
        "artifact, workload, directive, scope, or operand boundaries."
    ),
    "A16.multi_anchor_or_report_donation": (
        "Attempt to use multiple anchors, a mismatched source artifact, "
        "sibling report, or duplicate declared output path."
    ),
    "A17.parser_predicate_value_unit_provenance_substitution": (
        "Attempt to substitute parser identity, predicate, value, unit, "
        "artifact bytes, or provenance."
    ),
    "A18.stale_fake_replay_failed_undeclared_path_replaced_artifact": (
        "Attempt activation from stale, fake, replay, failed, undeclared, "
        "path-replaced, hash-mismatched, or otherwise uncommitted artifacts."
    ),
    "A19.requested_achieved_estimate_postroute_confusion": (
        "Attempt to confuse requested with achieved, estimates with later "
        "stages, or correctness/resource/timing Gates."
    ),
}


def assertion_contract(
    protocol_id: str, assertion_ids: Sequence[str],
) -> list[dict[str, str]]:
    """Project assertion prose without exposing another shard's IDs."""

    if protocol_id in {"semantic", SEMANTIC_PROTOCOL_ID}:
        descriptions = SEMANTIC_ASSERTION_DESCRIPTIONS
    elif protocol_id in {"adversarial", ADVERSARIAL_PROTOCOL_ID}:
        descriptions = ADVERSARIAL_ASSERTION_DESCRIPTIONS
    else:
        raise ShardPlanError(f"unknown review protocol: {protocol_id!r}")
    if (not isinstance(assertion_ids, Sequence)
            or isinstance(assertion_ids, (str, bytes))):
        raise ShardPlanError("assertion IDs must be a sequence")
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for assertion_id in assertion_ids:
        if not isinstance(assertion_id, str) or assertion_id in seen:
            raise ShardPlanError("assertion projection has malformed or duplicate IDs")
        meaning = descriptions.get(assertion_id)
        if meaning is None:
            raise ShardPlanError(f"unknown assertion ID: {assertion_id!r}")
        result.append({"assertion_id": assertion_id, "meaning": meaning})
        seen.add(assertion_id)
    return sorted(result, key=lambda row: row["assertion_id"])


_KNOWLEDGE_SOURCES = (
    *projected_model_source_paths("knowledge_activation"),
    "src/hlsgraph/bundle.py",
    "src/hlsgraph/evidence_policy.py",
    "src/hlsgraph/extract/base.py",
    "src/hlsgraph/extract/directive_identity.py",
    "src/hlsgraph/extract/directive_replay.py",
    "src/hlsgraph/extract/directives.py",
    "src/hlsgraph/extract/source.py",
    "src/hlsgraph/graph.py",
    "src/hlsgraph/knowledge/activation.py",
    "src/hlsgraph/knowledge/core.py",
    "src/hlsgraph/knowledge/supported_targets.py",
    "src/hlsgraph/model.py",
    "src/hlsgraph/retrieval.py",
    "tools/knowledge_review.schema.json",
    "tools/knowledge_review_evidence.schema.json",
)

_IR_SOURCES = (
    *projected_model_source_paths("ir_semantics"),
    "src/hlsgraph/evidence_policy.py",
    "src/hlsgraph/extract/llvm.py",
    "src/hlsgraph/extract/mlir.py",
    "src/hlsgraph/extract/source.py",
    "src/hlsgraph/extract/static_features.py",
    "src/hlsgraph/graph.py",
    "src/hlsgraph/model.py",
    "src/hlsgraph/retrieval.py",
)

_TOOL_SOURCES = (
    *projected_model_source_paths("tool_evidence"),
    "src/hlsgraph/bundle.py",
    "src/hlsgraph/evidence_policy.py",
    "src/hlsgraph/extract/observation_replay.py",
    "src/hlsgraph/extract/vitis.py",
    "src/hlsgraph/extract/vivado.py",
    "src/hlsgraph/manifest.py",
    "src/hlsgraph/model.py",
    "src/hlsgraph/run_projection.py",
    "src/hlsgraph/runner/core.py",
    "src/hlsgraph/runner/staging.py",
    "src/hlsgraph/store/migrations.py",
    "src/hlsgraph/store/sqlite.py",
)


SHARD_DEFINITIONS = (
    ShardDefinition(
        shard_id="knowledge_activation",
        source_paths=_KNOWLEDGE_SOURCES,
        semantic_assertion_ids=(
            "S01.knowledge_guidance_not_fact",
            "S02.binding_condition_instance_local_entailment",
            "S03.directive_exact_scope_source_operand_proof",
            "S11.coverage_target_registry_activation_gate",
        ),
        adversarial_assertion_ids=(
            "A01.reserved_metadata_injection",
            "A02.directive_scope_operand_source_forgery",
            "A03.review_ready_install_select_bypass",
            "A04.incomplete_condition_or_cross_target_context",
            "A05.coverage_registry_concealment",
            "A06.generic_container_unknown_boolean_activation",
        ),
    ),
    ShardDefinition(
        shard_id="ir_semantics",
        source_paths=_IR_SOURCES,
        semantic_assertion_ids=(
            "S07.language_spec_contract_and_non_topology",
            "S08.cross_layer_mapping_unique_typed_anchored",
            "S09.aggregate_static_feature_recomputation",
            "S10.retrieval_plane_isolation",
        ),
        adversarial_assertion_ids=(
            "A07.spec_revision_compatibility_forgery",
            "A08.graph_metadata_semantic_attestation",
            "A09.llvm_cfg_software_call_handshake_topology_promotion",
            "A10.ambiguous_cross_layer_mapping_promotion",
            "A11.aggregate_feature_spoof",
            "A12.retrieval_plane_rank_budget_pollution",
        ),
    ),
    ShardDefinition(
        shard_id="tool_evidence",
        source_paths=_TOOL_SOURCES,
        semantic_assertion_ids=(
            "S04.fresh_real_run_authorization_receipt",
            "S05.canonical_report_artifact_single_anchor",
            "S06.requested_effective_achieved_stage_and_three_gate_separation",
        ),
        adversarial_assertion_ids=(
            "A13.fabricated_sdk_toolrun_runner_receipt",
            "A14.one_use_authorization_or_receipt_replay",
            "A15.sibling_snapshot_run_stage_artifact_workload_reuse",
            "A16.multi_anchor_or_report_donation",
            "A17.parser_predicate_value_unit_provenance_substitution",
            "A18.stale_fake_replay_failed_undeclared_path_replaced_artifact",
            "A19.requested_achieved_estimate_postroute_confusion",
        ),
    ),
)

_SHARD_BY_ID = {item.shard_id: item for item in SHARD_DEFINITIONS}
if tuple(_SHARD_BY_ID) != SHARD_ORDER:  # pragma: no cover - import invariant
    raise RuntimeError("fixed shard definitions do not match SHARD_ORDER")


_UG1399 = "https://docs.amd.com/r/2024.2-English/ug1399-vitis-hls/"
_UG835 = "https://docs.amd.com/r/2024.2-English/ug835-vivado-tcl-commands/"
_UG903 = "https://docs.amd.com/r/2024.2-English/ug903-vivado-using-constraints/"
_UG906 = "https://docs.amd.com/r/2024.2-English/ug906-vivado-design-analysis/"
_UG907 = (
    "https://docs.amd.com/r/2024.2-English/"
    "ug907-vivado-power-analysis-optimization/"
)
_LLVM_COMMIT = "429c88d37f1f02e68ebc1fc7b0da4511ce6407e3"
_LLVM_BASE = f"https://github.com/llvm/llvm-project/blob/{_LLVM_COMMIT}/"
_CIRCT_URL = (
    "https://github.com/llvm/circt/blob/"
    "ef03d45c960607315a8b62903b92d072d8542e30/"
    "include/circt/Dialect/Handshake/HandshakeOps.td"
)


def _spec(
    rule_id: str, shard_id: str, citation_url: str, section: str,
) -> RuleReferenceSpec:
    return RuleReferenceSpec(rule_id, shard_id, citation_url, section)


RULE_REFERENCE_SPECS = (
    # Directive/source activation evidence.
    _spec(
        "amd.ug1399:2024.2:directive.array_partition_requests_banking",
        "knowledge_activation", _UG1399 + "pragma-HLS-array_partition",
        "pragma HLS array_partition",
    ),
    _spec(
        "amd.ug1399:2024.2:directive.dataflow_has_explicit_scope",
        "knowledge_activation", _UG1399 + "pragma-HLS-dataflow",
        "pragma HLS dataflow",
    ),
    _spec(
        "amd.ug1399:2024.2:directive.dependence_is_user_assertion",
        "knowledge_activation", _UG1399 + "pragma-HLS-dependence",
        "pragma HLS dependence",
    ),
    _spec(
        "amd.ug1399:2024.2:directive.inline_changes_hierarchy",
        "knowledge_activation", _UG1399 + "pragma-HLS-inline",
        "pragma HLS inline",
    ),
    _spec(
        "amd.ug1399:2024.2:directive.interface_is_port_contract",
        "knowledge_activation", _UG1399 + "pragma-HLS-interface",
        "pragma HLS interface",
    ),
    _spec(
        "amd.ug1399:2024.2:axi.interface_mode_is_scoped_request",
        "knowledge_activation", _UG1399 + "pragma-HLS-interface",
        "pragma HLS interface",
    ),
    _spec(
        "amd.ug1399:2024.2:directive.pipeline_ii_is_requested",
        "knowledge_activation", _UG1399 + "pragma-HLS-pipeline",
        "pragma HLS pipeline",
    ),
    _spec(
        "amd.ug1399:2024.2:directive.stream_depth_is_requested",
        "knowledge_activation", _UG1399 + "pragma-HLS-stream",
        "pragma HLS stream",
    ),
    _spec(
        "amd.ug1399:2024.2:directive.tripcount_guides_estimates",
        "knowledge_activation", _UG1399 + "pragma-HLS-loop_tripcount",
        "pragma HLS loop_tripcount",
    ),
    _spec(
        "amd.ug1399:2024.2:directive.unroll_requests_replication",
        "knowledge_activation", _UG1399 + "pragma-HLS-unroll",
        "pragma HLS unroll",
    ),
    # Compiler IR and language semantics.
    _spec(
        "circt.handshake:git-ef03d45c960607315a8b62903b92d072d8542e30:"
        "circt.handshake_has_dataflow_semantics",
        "ir_semantics", _CIRCT_URL,
        "Handshake operation and FuncOp definitions",
    ),
    _spec(
        f"llvm.ir.debug:git-{_LLVM_COMMIT}:"
        "llvm.debug_metadata_is_source_mapping_evidence",
        "ir_semantics",
        _LLVM_BASE
        + "llvm/docs/SourceLevelDebugging.md#debugging-information-format",
        "Debugging Information Format",
    ),
    _spec(
        f"llvm.ir.langref:git-{_LLVM_COMMIT}:"
        "llvm.bitwidth_histograms_require_integer_width_schema",
        "ir_semantics", _LLVM_BASE + "llvm/docs/LangRef.md#integer-type",
        "Integer Type",
    ),
    _spec(
        f"llvm.ir.langref:git-{_LLVM_COMMIT}:llvm.calls_are_low_level_ir_evidence",
        "ir_semantics", _LLVM_BASE + "llvm/docs/LangRef.md#call-instruction",
        "Call Instruction",
    ),
    _spec(
        f"llvm.ir.langref:git-{_LLVM_COMMIT}:llvm.cfg_is_low_level_control_evidence",
        "ir_semantics", _LLVM_BASE + "llvm/docs/LangRef.md#functions",
        "Basic Blocks and Terminators",
    ),
    _spec(
        f"llvm.ir.langref:git-{_LLVM_COMMIT}:llvm.containers_are_ir_scope_evidence",
        "ir_semantics", _LLVM_BASE + "llvm/docs/LangRef.md#functions",
        "Functions",
    ),
    _spec(
        f"llvm.ir.langref:git-{_LLVM_COMMIT}:"
        "llvm.index_histograms_require_explicit_operand_schema",
        "ir_semantics",
        _LLVM_BASE + "llvm/docs/LangRef.md#getelementptr-instruction",
        "GetElementPtr Instruction",
    ),
    _spec(
        f"llvm.ir.langref:git-{_LLVM_COMMIT}:"
        "llvm.instructions_are_typed_operation_evidence",
        "ir_semantics",
        _LLVM_BASE + "llvm/docs/LangRef.md#instruction-reference",
        "Instruction Reference",
    ),
    _spec(
        f"llvm.ir.langref:git-{_LLVM_COMMIT}:"
        "llvm.memory_access_histograms_require_opcode_schema",
        "ir_semantics",
        _LLVM_BASE
        + "llvm/docs/LangRef.md#memory-access-and-addressing-operations",
        "Memory Access and Addressing Instructions",
    ),
    _spec(
        f"llvm.ir.langref:git-{_LLVM_COMMIT}:"
        "llvm.operation_histograms_require_opcode_schema",
        "ir_semantics",
        _LLVM_BASE + "llvm/docs/LangRef.md#instruction-reference",
        "Instruction Reference / Opcode Aggregate",
    ),
    _spec(
        f"llvm.mlir.builtin:git-{_LLVM_COMMIT}:"
        "mlir.locations_are_mapping_evidence",
        "ir_semantics",
        _LLVM_BASE + "mlir/include/mlir/IR/BuiltinLocationAttributes.td",
        "Location Attributes",
    ),
    _spec(
        f"llvm.mlir.langref:git-{_LLVM_COMMIT}:"
        "mlir.containers_preserve_ir_scope",
        "ir_semantics", _LLVM_BASE + "mlir/docs/LangRef.md#blocks",
        "Blocks and Regions",
    ),
    _spec(
        f"llvm.mlir.langref:git-{_LLVM_COMMIT}:"
        "mlir.multiple_dialects_keep_native_semantics",
        "ir_semantics", _LLVM_BASE + "mlir/docs/LangRef.md#dialects",
        "Dialects",
    ),
    _spec(
        f"llvm.mlir.langref:git-{_LLVM_COMMIT}:"
        "mlir.operation_histograms_require_dialect_schema",
        "ir_semantics", _LLVM_BASE + "mlir/docs/LangRef.md#operations",
        "Operations",
    ),
    _spec(
        f"llvm.mlir.langref:git-{_LLVM_COMMIT}:"
        "mlir.region_semantics_come_from_owner",
        "ir_semantics", _LLVM_BASE + "mlir/docs/LangRef.md#regions",
        "Regions",
    ),
    # Real-tool observations and stage-scoped evidence.
    _spec(
        "amd.ug1399:2024.2:dataflow.dynamic_results_are_workload_scoped",
        "tool_evidence", _UG1399 + "Dataflow-Viewer", "Dataflow Viewer",
    ),
    _spec(
        "amd.ug1399:2024.2:directive.requested_effective_achieved",
        "tool_evidence",
        _UG1399 + "Failure-to-Satisfy-Optimization-Directives",
        "Failure to Satisfy Optimization Directives",
    ),
    _spec(
        "amd.ug1399:2024.2:qor.latency_and_ii_are_distinct",
        "tool_evidence", _UG1399 + "Performance-Metrics-Example",
        "Performance Metrics Example",
    ),
    _spec(
        "amd.ug1399:2024.2:schedule.control_steps_are_compiler_evidence",
        "tool_evidence", _UG1399 + "Schedule-Viewer", "Schedule Viewer",
    ),
    _spec(
        "amd.ug1399:2024.2:verification.cosim_is_workload_scoped",
        "tool_evidence", _UG1399 + "Running-C/RTL-Co-Simulation",
        "Running C/RTL Co-Simulation",
    ),
    _spec(
        "amd.ug1399:2024.2:verification.csim_is_workload_scoped",
        "tool_evidence", _UG1399 + "Running-C-Simulation",
        "Running C Simulation",
    ),
    _spec(
        "amd.ug835:2024.2:timing.summary_keeps_wns_tns_distinct",
        "tool_evidence", _UG835 + "report_timing_summary",
        "report_timing_summary",
    ),
    _spec(
        "amd.ug903:2024.2:xdc.constraints_are_design_inputs",
        "tool_evidence", _UG903 + "About-XDC-Constraints",
        "About XDC Constraints",
    ),
    _spec(
        "amd.ug906:2024.2:physical.congestion_is_stage_scoped",
        "tool_evidence", _UG906 + "Analyzing-the-Design-Congestion",
        "Analyzing the Design Congestion",
    ),
    _spec(
        "amd.ug906:2024.2:resource.utilization_is_stage_scoped",
        "tool_evidence", _UG906 + "Report-Utilization", "Report Utilization",
    ),
    _spec(
        "amd.ug906:2024.2:timing.post_route_observations_are_routed_evidence",
        "tool_evidence", _UG906 + "Verifying-Timing-Signoff",
        "Verifying Timing Signoff",
    ),
    _spec(
        "amd.ug906:2024.2:timing.post_route_signoff_requires_routed_design",
        "tool_evidence", _UG906 + "Verifying-Timing-Signoff",
        "Verifying Timing Signoff",
    ),
    _spec(
        "amd.ug907:2024.2:power.requires_activity_context",
        "tool_evidence",
        _UG907 + "Running-Power-Analysis-from-the-Vivado-IDE",
        "Running Power Analysis from the Vivado IDE",
    ),
)

_RULE_SPEC_BY_ID = {item.rule_id: item for item in RULE_REFERENCE_SPECS}
if len(_RULE_SPEC_BY_ID) != len(RULE_REFERENCE_SPECS):  # pragma: no cover
    raise RuntimeError("duplicate rule ID in fixed reference allocation")
if set(item.shard_id for item in RULE_REFERENCE_SPECS) != set(SHARD_ORDER):
    raise RuntimeError("rule allocation does not cover all fixed shards")


def assertion_owners(protocol_id: str) -> dict[str, str]:
    """Return the closed assertion-to-shard mapping for one protocol."""

    if protocol_id in {"semantic", SEMANTIC_PROTOCOL_ID}:
        attribute = "semantic_assertion_ids"
    elif protocol_id in {"adversarial", ADVERSARIAL_PROTOCOL_ID}:
        attribute = "adversarial_assertion_ids"
    else:
        raise ShardPlanError(f"unknown review protocol: {protocol_id!r}")
    result: dict[str, str] = {}
    for shard in SHARD_DEFINITIONS:
        for assertion_id in getattr(shard, attribute):
            if assertion_id in result:  # pragma: no cover - import invariant
                raise RuntimeError(f"duplicate assertion owner: {assertion_id}")
            result[assertion_id] = shard.shard_id
    return result


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ) + "\n").encode("utf-8")


def allocate_rule_references(
    citation_audit: Mapping[str, Any],
) -> dict[str, tuple[dict[str, str], ...]]:
    """Validate and allocate the exact closed rule-reference inventory.

    Document references are intentionally not assigned to model shards.  They
    belong to the deterministic suite aggregate.  Every rule reference must
    match its declared rule ID, URL, and section exactly.
    """

    rows = citation_audit.get("references")
    if not isinstance(rows, list):
        raise ShardPlanError("citation audit references must be an array")
    allocated: dict[str, list[dict[str, str]]] = {
        shard_id: [] for shard_id in SHARD_ORDER
    }
    seen_rules: set[str] = set()
    seen_references: set[str] = set()
    owners_by_url: dict[str, set[str]] = {}
    unexpected: list[str] = []

    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ShardPlanError(f"citation reference {index} is not an object")
        if row.get("reference_kind") != "rule":
            continue
        rule_id = row.get("rule_id")
        if not isinstance(rule_id, str) or not rule_id:
            raise ShardPlanError(f"rule reference {index} has no rule_id")
        spec = _RULE_SPEC_BY_ID.get(rule_id)
        if spec is None:
            unexpected.append(rule_id)
            continue
        reference_id = row.get("reference_id")
        surface = row.get("reference_surface_sha256")
        if not isinstance(reference_id, str) or _SHA256_RE.fullmatch(reference_id) is None:
            raise ShardPlanError(f"rule {rule_id} has an invalid reference_id")
        if not isinstance(surface, str) or _SHA256_RE.fullmatch(surface) is None:
            raise ShardPlanError(
                f"rule {rule_id} has an invalid reference surface SHA-256"
            )
        if rule_id in seen_rules:
            raise ShardPlanError(f"duplicate rule reference: {rule_id}")
        if reference_id in seen_references:
            raise ShardPlanError(f"duplicate reference_id: {reference_id}")
        if row.get("citation_url") != spec.citation_url:
            raise ShardPlanError(f"rule {rule_id} citation URL does not match plan")
        if row.get("section") != spec.section:
            raise ShardPlanError(f"rule {rule_id} section does not match plan")
        normalized = {
            "reference_id": reference_id,
            "reference_surface_sha256": surface,
            "rule_id": rule_id,
            "citation_url": spec.citation_url,
            "section": spec.section,
        }
        allocated[spec.shard_id].append(normalized)
        seen_rules.add(rule_id)
        seen_references.add(reference_id)
        owners_by_url.setdefault(spec.citation_url, set()).add(spec.shard_id)

    if unexpected:
        raise ShardPlanError(
            "unexpected rule references: " + ", ".join(sorted(unexpected))
        )
    missing = sorted(set(_RULE_SPEC_BY_ID) - seen_rules)
    if missing:
        raise ShardPlanError("missing rule references: " + ", ".join(missing))
    split_urls = sorted(url for url, owners in owners_by_url.items() if len(owners) != 1)
    if split_urls:  # pragma: no cover - guarded by the fixed specs
        raise ShardPlanError(
            "references sharing one citation URL cannot span shards: "
            + ", ".join(split_urls)
        )
    return {
        shard_id: tuple(sorted(items, key=lambda row: row["reference_id"]))
        for shard_id, items in allocated.items()
    }


def build_shard_plan(citation_audit: Mapping[str, Any]) -> dict[str, Any]:
    """Build the deterministic, JSON-serializable three-shard plan."""

    references = allocate_rule_references(citation_audit)
    shards: list[dict[str, Any]] = []
    for shard in SHARD_DEFINITIONS:
        shards.append({
            "shard_id": shard.shard_id,
            "source_paths": list(shard.source_paths),
            "semantic_assertion_ids": list(shard.semantic_assertion_ids),
            "adversarial_assertion_ids": list(shard.adversarial_assertion_ids),
            "rule_references": [dict(row) for row in references[shard.shard_id]],
        })
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "shard_order": list(SHARD_ORDER),
        "token_budget_contract": DEFAULT_TOKEN_BUDGET_CONTRACT.to_dict(),
        "shards": shards,
    }


def _json_projection_source(payload: bytes, *, label: str) -> dict[str, Any]:
    if not isinstance(payload, bytes):
        raise ShardPlanError(f"{label} payload must be bytes")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ShardPlanError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ShardPlanError(f"{label} contains non-finite value {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ShardPlanError(f"cannot parse {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ShardPlanError(f"{label} must be a JSON object")
    return value


def _projection_envelope(
    *, shard_id: str, source_path: str, source_payload: bytes,
    projection_kind: str, assigned_rule_ids: Sequence[str],
    assigned_reference_ids: Sequence[str], projected: Mapping[str, Any],
) -> bytes:
    return _canonical_json({
        "schema_version": MODEL_SOURCE_PROJECTION_SCHEMA_VERSION,
        "projection_kind": projection_kind,
        "shard_id": shard_id,
        "source_path": source_path,
        "source_sha256": hashlib.sha256(source_payload).hexdigest(),
        "assigned_rule_ids": sorted(assigned_rule_ids),
        "assigned_reference_ids": sorted(assigned_reference_ids),
        "projected": copy.deepcopy(dict(projected)),
    })


def _binding_id(binding: Mapping[str, Any]) -> str:
    required = {
        "knowledge_rule_id", "target_kind", "target", "required_context",
        "producer", "producer_version",
    }
    if not required.issubset(binding):
        raise ShardPlanError("knowledge binding is missing identity fields")
    value = {
        "rule": binding["knowledge_rule_id"],
        "target_kind": binding["target_kind"],
        "target": binding["target"],
        "required_context": binding["required_context"],
        "producer": binding["producer"],
        "producer_version": binding["producer_version"],
    }
    digest = hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return f"knowledge_binding_{digest[:24]}"


_PACK_GOVERNANCE_SHARD = {
    AMD_PACK_SOURCE_PATH: "knowledge_activation",
    AXI_PACK_SOURCE_PATH: "knowledge_activation",
    OPEN_IR_PACK_SOURCE_PATH: "ir_semantics",
}


def _project_pack_source(
    *, shard_id: str, source_path: str, source_payload: bytes,
    assigned_references: Sequence[Mapping[str, Any]], pack_id: str,
) -> bytes:
    value = _json_projection_source(source_payload, label=source_path)
    if value.get("pack_id") != pack_id:
        raise ShardPlanError(f"pack identity differs from citation audit: {source_path}")
    assigned_rule_ids = sorted({
        str(row["rule_id"]) for row in assigned_references
        if row.get("pack_id") == pack_id
    })
    assigned_reference_ids = sorted({
        str(row["reference_id"]) for row in assigned_references
        if row.get("pack_id") == pack_id
    })
    if not assigned_rule_ids:
        raise ShardPlanError(f"shard has no assigned rules in projected pack: {source_path}")

    rules = value.get("rules")
    bindings = value.get("bindings")
    coverage = value.get("coverage")
    documents = value.get("documents")
    if (not isinstance(rules, list) or not isinstance(bindings, list)
            or not isinstance(coverage, dict) or not isinstance(documents, list)):
        raise ShardPlanError(f"pack projection source is malformed: {source_path}")
    projected_rules: list[dict[str, Any]] = []
    for source in rules:
        if not isinstance(source, dict):
            raise ShardPlanError(f"pack contains a malformed rule: {source_path}")
        qualified = (
            f"{source.get('document_id')}:{source.get('document_version')}:"
            f"{source.get('rule_id')}"
        )
        if qualified not in assigned_rule_ids:
            continue
        row = copy.deepcopy(source)
        row["qualified_rule_id"] = qualified
        projected_rules.append(row)
    if sorted(str(row.get("qualified_rule_id")) for row in projected_rules) != assigned_rule_ids:
        raise ShardPlanError(f"pack does not contain every assigned rule: {source_path}")

    projected_bindings = [
        copy.deepcopy(row) for row in bindings
        if isinstance(row, dict)
        and row.get("knowledge_rule_id") in assigned_rule_ids
    ]
    selected_binding_ids = {_binding_id(row) for row in projected_bindings}
    if len(selected_binding_ids) != len(projected_bindings):
        raise ShardPlanError(f"projected pack duplicates a binding: {source_path}")

    governance = _PACK_GOVERNANCE_SHARD[source_path] == shard_id
    coverage_entries = coverage.get("entries")
    target_inventory = coverage.get("target_inventory")
    if not isinstance(coverage_entries, list) or not isinstance(target_inventory, list):
        raise ShardPlanError(f"pack coverage is malformed: {source_path}")
    projected_entries: list[dict[str, Any]] = []
    for source in coverage_entries:
        if not isinstance(source, dict):
            raise ShardPlanError(f"pack coverage entry is malformed: {source_path}")
        status = source.get("status")
        if status == "rule":
            source_rules = source.get("rule_ids")
            source_bindings = source.get("binding_ids")
            if not isinstance(source_rules, list) or not isinstance(source_bindings, list):
                raise ShardPlanError(f"rule coverage is malformed: {source_path}")
            kept_rules = sorted(set(source_rules) & set(assigned_rule_ids))
            if not kept_rules:
                continue
            row = copy.deepcopy(source)
            row["rule_ids"] = kept_rules
            row["binding_ids"] = sorted(
                set(source_bindings) & selected_binding_ids
            )
            projected_entries.append(row)
        elif governance:
            projected_entries.append(copy.deepcopy(source))

    projected_targets: list[dict[str, Any]] = []
    for source in target_inventory:
        if not isinstance(source, dict):
            raise ShardPlanError(f"target coverage is malformed: {source_path}")
        if source.get("status") == "bound":
            if not isinstance(source.get("binding_ids"), list):
                raise ShardPlanError(f"bound target coverage is malformed: {source_path}")
            kept = sorted(set(source["binding_ids"]) & selected_binding_ids)
            if not kept:
                continue
            row = copy.deepcopy(source)
            row["binding_ids"] = kept
            projected_targets.append(row)
        elif governance:
            projected_targets.append(copy.deepcopy(source))

    covered_rules = {
        str(rule_id) for row in projected_entries
        for rule_id in row.get("rule_ids", [])
    }
    if covered_rules != set(assigned_rule_ids):
        raise ShardPlanError(f"projected pack coverage omits assigned rules: {source_path}")
    covered_bindings = {
        str(binding_id) for row in projected_entries
        for binding_id in row.get("binding_ids", [])
    }
    if covered_bindings != selected_binding_ids:
        raise ShardPlanError(f"projected pack coverage omits assigned bindings: {source_path}")

    document_keys = {
        (str(row.get("document_id")), str(row.get("document_version")))
        for row in projected_rules
    }
    document_keys.update(
        (str(row.get("document_id")), str(row.get("document_version")))
        for row in projected_entries
    )
    projected_documents = [
        copy.deepcopy(row) for row in documents
        if isinstance(row, dict)
        and (str(row.get("document_id")), str(row.get("document_version")))
        in document_keys
    ]
    if {
        (str(row.get("document_id")), str(row.get("document_version")))
        for row in projected_documents
    } != document_keys:
        raise ShardPlanError(f"projected pack omits a referenced document: {source_path}")

    projected_coverage = {
        key: copy.deepcopy(item) for key, item in coverage.items()
        if key not in {"entries", "target_inventory"}
    }
    projected_coverage["entries"] = projected_entries
    projected_coverage["target_inventory"] = projected_targets
    projected = {
        key: copy.deepcopy(value.get(key)) for key in (
            "schema_version", "pack_id", "title", "license", "metadata",
        )
    }
    projected.update({
        "documents": projected_documents,
        "rules": projected_rules,
        "bindings": projected_bindings,
        "coverage": projected_coverage,
    })
    return _projection_envelope(
        shard_id=shard_id, source_path=source_path,
        source_payload=source_payload, projection_kind="knowledge_pack",
        assigned_rule_ids=assigned_rule_ids,
        assigned_reference_ids=assigned_reference_ids,
        projected=projected,
    )


def build_model_source_projections(
    source_payloads: Mapping[str, bytes],
) -> dict[str, bytes]:
    """Build all model-readable rule-local JSON projections.

    The full cache remains the immutable acquisition/snapshot boundary.  Only
    these virtual payloads are named by a shard plan, so a projected cache can
    never expose another shard's rules merely because they share one source
    JSON file in the checkout.
    """

    required = {
        CITATION_AUDIT_SOURCE_PATH, CITATION_EVIDENCE_SOURCE_PATH,
        *_PACK_SOURCE_PATHS,
    }
    if not required.issubset(source_payloads):
        missing = sorted(required - set(source_payloads))
        raise ShardPlanError("projection sources are missing: " + ", ".join(missing))
    audit_payload = source_payloads[CITATION_AUDIT_SOURCE_PATH]
    evidence_payload = source_payloads[CITATION_EVIDENCE_SOURCE_PATH]
    audit = _json_projection_source(audit_payload, label=CITATION_AUDIT_SOURCE_PATH)
    evidence = _json_projection_source(
        evidence_payload, label=CITATION_EVIDENCE_SOURCE_PATH,
    )
    references = audit.get("references")
    entries = evidence.get("entries")
    packs = audit.get("packs")
    if (not isinstance(references, list) or not isinstance(entries, list)
            or not isinstance(packs, list)):
        raise ShardPlanError("citation audit/evidence projection sources are malformed")
    allocated = allocate_rule_references(audit)
    reference_by_id = {
        str(row.get("reference_id")): row for row in references
        if isinstance(row, dict) and row.get("reference_kind") == "rule"
    }
    pack_path_by_id = {
        str(row.get("pack_id")): str(row.get("path"))
        for row in packs if isinstance(row, dict)
    }
    if set(pack_path_by_id.values()) != set(_PACK_SOURCE_PATHS):
        raise ShardPlanError("citation audit pack paths differ from projection contract")

    result: dict[str, bytes] = {}
    projected_rule_union: set[str] = set()
    for shard_id in SHARD_ORDER:
        assigned_ids = {row["reference_id"] for row in allocated[shard_id]}
        assigned_references = [
            copy.deepcopy(reference_by_id[reference_id])
            for reference_id in sorted(assigned_ids)
        ]
        assigned_rule_ids = sorted(str(row["rule_id"]) for row in assigned_references)
        if len(assigned_rule_ids) != len(set(assigned_rule_ids)):
            raise ShardPlanError(f"shard duplicates a projected rule: {shard_id}")
        if projected_rule_union & set(assigned_rule_ids):
            raise ShardPlanError("one rule would be model-visible in multiple shards")
        projected_rule_union.update(assigned_rule_ids)
        urls = {str(row["citation_url"]) for row in assigned_references}
        fetch_urls = {str(row["fetch_url"]) for row in assigned_references}
        document_keys = {
            (str(row["document_id"]), str(row["document_version"]))
            for row in assigned_references
        }
        selected_pack_ids = {str(row["pack_id"]) for row in assigned_references}

        projected_audit = {
            "schema_version": audit.get("schema_version"),
            "mode": audit.get("mode"),
            "passed": audit.get("passed"),
            "policy": copy.deepcopy(audit.get("policy")),
            "surface_policy": copy.deepcopy(audit.get("surface_policy")),
            "document_evidence_policy": copy.deepcopy(
                audit.get("document_evidence_policy")
            ),
            "generator": copy.deepcopy(audit.get("generator")),
            "source_manifest_sha256": audit.get("manifest_sha256"),
            "packs": [
                {
                    "pack_id": row.get("pack_id"),
                    "path": row.get("path"),
                    "review_surface_sha256": row.get("review_surface_sha256"),
                    "projected_rule_count": sum(
                        reference.get("pack_id") == row.get("pack_id")
                        for reference in assigned_references
                    ),
                }
                for row in packs if isinstance(row, dict)
                and row.get("pack_id") in selected_pack_ids
            ],
            "references": assigned_references,
            "fetches": [
                copy.deepcopy(row) for row in audit.get("fetches", [])
                if isinstance(row, dict) and row.get("fetch_url") in fetch_urls
            ],
            "document_evidence": [
                copy.deepcopy(row) for row in audit.get("document_evidence", [])
                if isinstance(row, dict)
                and (str(row.get("document_id")), str(row.get("document_version")))
                in document_keys
            ],
        }
        audit_virtual = model_source_projection_path(
            shard_id, CITATION_AUDIT_SOURCE_PATH,
        )
        result[audit_virtual] = _projection_envelope(
            shard_id=shard_id, source_path=CITATION_AUDIT_SOURCE_PATH,
            source_payload=audit_payload, projection_kind="citation_audit",
            assigned_rule_ids=assigned_rule_ids,
            assigned_reference_ids=sorted(assigned_ids), projected=projected_audit,
        )

        selected_entries: list[dict[str, Any]] = []
        seen_bindings: set[str] = set()
        for entry in entries:
            if not isinstance(entry, dict):
                raise ShardPlanError("citation evidence contains a malformed entry")
            bindings = entry.get("reference_bindings")
            if not isinstance(bindings, list):
                raise ShardPlanError("citation evidence entry has malformed bindings")
            kept = [
                copy.deepcopy(row) for row in bindings
                if isinstance(row, dict) and row.get("reference_id") in assigned_ids
            ]
            if not kept:
                continue
            if entry.get("citation_url") not in urls:
                raise ShardPlanError("evidence binding points to another citation URL")
            row = copy.deepcopy(entry)
            row["reference_bindings"] = kept
            selected_entries.append(row)
            seen_bindings.update(str(item["reference_id"]) for item in kept)
        if seen_bindings != assigned_ids:
            raise ShardPlanError(f"evidence projection is incomplete for {shard_id}")
        projected_evidence = {
            "schema_version": evidence.get("schema_version"),
            "citation_audit_sha256": evidence.get("citation_audit_sha256"),
            "entries": selected_entries,
        }
        evidence_virtual = model_source_projection_path(
            shard_id, CITATION_EVIDENCE_SOURCE_PATH,
        )
        result[evidence_virtual] = _projection_envelope(
            shard_id=shard_id, source_path=CITATION_EVIDENCE_SOURCE_PATH,
            source_payload=evidence_payload, projection_kind="citation_evidence",
            assigned_rule_ids=assigned_rule_ids,
            assigned_reference_ids=sorted(assigned_ids), projected=projected_evidence,
        )

        expected_sources = set(_PROJECTED_SOURCE_PATHS[shard_id])
        pack_sources = expected_sources - {
            CITATION_AUDIT_SOURCE_PATH, CITATION_EVIDENCE_SOURCE_PATH,
        }
        actual_pack_sources = {
            pack_path_by_id[pack_id] for pack_id in selected_pack_ids
        }
        if actual_pack_sources != pack_sources:
            raise ShardPlanError(f"projected pack assignment is stale for {shard_id}")
        for source_path in sorted(pack_sources):
            pack_id = next(
                key for key, value in pack_path_by_id.items() if value == source_path
            )
            virtual = model_source_projection_path(shard_id, source_path)
            result[virtual] = _project_pack_source(
                shard_id=shard_id, source_path=source_path,
                source_payload=source_payloads[source_path],
                assigned_references=assigned_references, pack_id=pack_id,
            )

    if projected_rule_union != set(_RULE_SPEC_BY_ID):
        raise ShardPlanError("model-source projection union differs from rule plan")
    expected_virtual = {
        path for shard_id in SHARD_ORDER
        for path in projected_model_source_paths(shard_id)
    }
    if set(result) != expected_virtual:
        raise ShardPlanError("model-source projection path inventory is incomplete")
    return dict(sorted(result.items()))


def shard_plan_sha256(plan: Mapping[str, Any]) -> str:
    """Hash a plan with canonical JSON; input order never affects the digest."""

    return hashlib.sha256(_canonical_json(plan)).hexdigest()


def _as_text(value: str | bytes, *, label: str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise TokenBudgetError(f"{label} is not strict UTF-8") from exc
    raise TokenBudgetError(f"{label} must be str or bytes")


def _count_tokens(
    tokenizer: Any, text: str, *, label: str,
) -> int:
    encoder: Callable[[str], Any] | None = None
    method = getattr(tokenizer, "encode", None)
    if callable(method):
        encoder = method
    elif callable(tokenizer):
        encoder = tokenizer
    if encoder is None:
        raise TokenBudgetError("tokenizer must be callable or expose encode(text)")
    try:
        encoded = encoder(text)
        if isinstance(encoded, bool):
            raise TypeError("boolean token count")
        if isinstance(encoded, int):
            count = encoded
        else:
            count = len(encoded)
    except Exception as exc:
        raise TokenBudgetError(f"cannot tokenize {label}") from exc
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise TokenBudgetError(f"tokenizer returned an invalid count for {label}")
    return count


def calculate_token_budget(
    *,
    prompt: str | bytes,
    chunks: Sequence[str | bytes],
    commands: Sequence[str],
    tokenizer: Any,
    contract: TokenBudgetContract = DEFAULT_TOKEN_BUDGET_CONTRACT,
) -> TokenBudget:
    """Count one shard's visible input without silently dropping any event.

    The review contract reads each chunk with one command.  A different number
    of commands is rejected instead of being assigned a guessed overhead.
    """

    if len(chunks) != len(commands):
        raise TokenBudgetError("every chunk must have exactly one read command")
    prompt_tokens = _count_tokens(
        tokenizer, _as_text(prompt, label="prompt"), label="prompt",
    )
    chunk_tokens = sum(
        _count_tokens(
            tokenizer, _as_text(chunk, label=f"chunk {index}"),
            label=f"chunk {index}",
        )
        for index, chunk in enumerate(chunks)
    )
    command_tokens = 0
    for index, command in enumerate(commands):
        if not isinstance(command, str) or not command:
            raise TokenBudgetError(f"command {index} must be a non-empty string")
        command_tokens += _count_tokens(
            tokenizer, command, label=f"command {index}",
        )
    event_count = len(commands)
    event_overhead = event_count * contract.tool_event_overhead_tokens
    runtime_allowance = contract.runtime_envelope_allowance_tokens
    visible = (
        prompt_tokens + chunk_tokens + command_tokens + event_overhead
        + runtime_allowance
    )
    reserve = contract.context_window_tokens - visible
    return TokenBudget(
        contract=contract,
        prompt_tokens=prompt_tokens,
        chunk_tokens=chunk_tokens,
        command_tokens=command_tokens,
        tool_event_count=event_count,
        tool_event_overhead_tokens=event_overhead,
        runtime_envelope_allowance_tokens=runtime_allowance,
        visible_input_tokens=visible,
        context_reserve_tokens=reserve,
    )


def enforce_token_budget(
    *,
    prompt: str | bytes,
    chunks: Sequence[str | bytes],
    commands: Sequence[str],
    tokenizer: Any,
    contract: TokenBudgetContract = DEFAULT_TOKEN_BUDGET_CONTRACT,
) -> TokenBudget:
    """Calculate and enforce the fixed 250k visible-input ceiling."""

    budget = calculate_token_budget(
        prompt=prompt, chunks=chunks, commands=commands,
        tokenizer=tokenizer, contract=contract,
    )
    if not budget.within_budget:
        raise TokenBudgetExceeded(
            "knowledge-review shard exceeds the fixed token budget: "
            f"visible={budget.visible_input_tokens}, "
            f"maximum={contract.max_visible_input_tokens}, "
            f"reserve={budget.context_reserve_tokens}, "
            f"minimum_reserve={contract.min_context_reserve_tokens}"
        )
    return budget


__all__ = [
    "ADVERSARIAL_PROTOCOL_ID",
    "AUTO_COMPACT_TOKEN_LIMIT_TOKENS",
    "AUTO_COMPACT_TOKEN_LIMIT_SCOPE",
    "ADVERSARIAL_ASSERTION_DESCRIPTIONS",
    "AMD_PACK_SOURCE_PATH",
    "AXI_PACK_SOURCE_PATH",
    "CITATION_AUDIT_SOURCE_PATH",
    "CITATION_EVIDENCE_SOURCE_PATH",
    "DEFAULT_TOKEN_BUDGET_CONTRACT",
    "DEFAULT_TOKENIZER_CONTRACT_SHA256",
    "DEFAULT_TOKENIZER_ID",
    "DEFAULT_TOKENIZER_PACKAGE",
    "DEFAULT_TOKENIZER_PACKAGE_VERSION",
    "MAX_VISIBLE_INPUT_TOKENS",
    "MODEL_SOURCE_PROJECTION_PREFIX",
    "MODEL_SOURCE_PROJECTION_SCHEMA_VERSION",
    "MIN_CONTEXT_RESERVE_TOKENS",
    "MODEL_CONTEXT_WINDOW_TOKENS",
    "PLAN_SCHEMA_VERSION",
    "OPEN_IR_PACK_SOURCE_PATH",
    "RUNTIME_ENVELOPE_ALLOWANCE_TOKENS",
    "RULE_REFERENCE_SPECS",
    "SEMANTIC_PROTOCOL_ID",
    "SEMANTIC_ASSERTION_DESCRIPTIONS",
    "SHARD_DEFINITIONS",
    "SHARD_ORDER",
    "ShardPlanError",
    "TokenBudget",
    "TokenBudgetContract",
    "TokenBudgetError",
    "TokenBudgetExceeded",
    "allocate_rule_references",
    "assertion_contract",
    "assertion_owners",
    "build_model_source_projections",
    "build_shard_plan",
    "calculate_token_budget",
    "enforce_token_budget",
    "load_verified_tokenizer",
    "is_model_source_projection_path",
    "model_source_projection_path",
    "projected_model_source_paths",
    "shard_plan_sha256",
    "tokenizer_contract_payload",
]
