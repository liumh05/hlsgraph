"""Deterministic hybrid retrieval over HLS facts, evidence, and guidance.

The retriever deliberately keeps the four public truth planes separate.  Graph
propagation is only a ranking operation over explicit canonical relations; it
never creates an entity, relation, observation, or QoR value.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
import difflib
import hashlib
import json
import math
import os
import re
import stat
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Mapping, Protocol, Sequence, runtime_checkable

from .bundle import GraphBundle
from .diagnostic_projection import public_diagnostic
from .evidence_policy import (
    successful_fresh_tool_run_error,
    tool_evidence_compatibility_error,
    tool_run_manifest_identity_error,
)
from .extract.directive_replay import (
    DirectiveReplayIndex,
    match_directive_replay,
    replay_directive_declarations,
)
from .graph import CanonicalGraph
from .knowledge.core import (
    canonical_context_scalar,
    knowledge_activation_hash,
    target_derived_condition_source,
)
from .knowledge.activation import (
    AttestedBindingContext,
    BindingActivationSession,
)
from .knowledge.supported_targets import canonical_supported_targets
from .model import (
    AuthorityClass, Entity, hash_artifact_bytes, json_ready,
    reject_embedded_body_fields, stable_hash, stable_id,
)
from .version import RETRIEVAL_PROFILE_SCHEMA_VERSION, SCHEMA_VERSION


RETRIEVAL_ALGORITHM_VERSION = "hlsgraph.hybrid_retrieval.v1"
DEFAULT_RETRIEVAL_PROFILE = "hls.default.v1"
DEFAULT_PLANES = ("facts", "evidence", "knowledge", "local")
VALID_PLANES = frozenset({"facts", "evidence", "knowledge", "local", "predictions"})
_PRIVATE_ACCESS_LOG = Path(".hlsgraph/private/retrieval-access.jsonl")

# The context keys below are not metadata.  They are capability-like markers
# emitted only after ``_binding_target_contexts`` closes a current record to
# typed immutable-ledger evidence and revalidates any referenced bytes.
# Versioned values keep generic truthy strings from being mistaken for a
# future, stronger evidence policy.
_GATE_EVIDENCE_CONTEXT_KEY = "gate_evidence_qualified"
_GATE_EVIDENCE_CONTEXT_VALUE = "derived_from_typed_evidence_v1"
_OBSERVATION_EVIDENCE_CONTEXT_KEY = "observation_evidence_qualified"
_OBSERVATION_EVIDENCE_CONTEXT_VALUE = "derived_from_typed_observation_evidence_v1"
_DIRECTIVE_OPERAND_CONTEXT_KEY = "directive_operand_linked"
_DIRECTIVE_OPERAND_CONTEXT_VALUE = "derived_from_current_directive_operand_link_v1"
_DEPENDENCE_OPERAND_CONTEXT_KEY = "dependence_operand_resolved"
_DEPENDENCE_OPERAND_CONTEXT_VALUE = "derived_from_current_dependence_operand_v1"
_DIRECTIVE_SOURCE_CONTEXT_KEY = "directive_source_declaration_qualified"
_DIRECTIVE_SOURCE_CONTEXT_VALUE = (
    "derived_from_current_directive_source_declaration_v1"
)
_PORT_OWNERSHIP_CONTEXT_KEY = "port_ownership_qualified"
_PORT_OWNERSHIP_CONTEXT_VALUE = (
    "derived_from_unique_current_component_port_v1"
)
_REQUESTED_DIRECTIVE_CONTEXT_KEY = "requested_directive_present"
_OBSERVATION_ARTIFACT_KIND_CONTEXT_KEY = "observation_artifact_kind"
_TOOL_ARTIFACT_CONTEXT_KEY = "tool_artifact_evidence_qualified"
_TOOL_ARTIFACT_CONTEXT_VALUE = "derived_from_declared_live_tool_output_v1"
_CONSTRAINT_INPUT_CONTEXT_KEY = "constraint_input_evidence_qualified"
_CONSTRAINT_INPUT_CONTEXT_VALUE = "derived_from_unique_live_snapshot_input_v1"
_SEMANTIC_ARTIFACT_EVIDENCE_CONTEXT_KEY = "semantic_artifact_evidence_qualified"
_SEMANTIC_ARTIFACT_EVIDENCE_CONTEXT_VALUE = (
    "derived_from_immutable_extraction_attestation_v1"
)
_SEMANTIC_ATTESTATION_CONTRACT = "hlsgraph.artifact_semantic_attestation.v1"
_SEMANTIC_ARTIFACT_BYTE_CLOSURE = "snapshot_input_live_sha256_no_link_v1"
_AGGREGATE_EVIDENCE_CONTEXT_KEY = "aggregate_evidence_qualified"
_AGGREGATE_EVIDENCE_CONTEXT_VALUE = (
    "derived_from_recomputed_current_ir_domain_v1"
)
_AGGREGATE_EVIDENCE_CONTRACT = "hlsgraph.recomputed_ir_aggregate.v1"
_RESERVED_DERIVED_CONTEXT_KEYS = frozenset({
    _GATE_EVIDENCE_CONTEXT_KEY,
    _OBSERVATION_EVIDENCE_CONTEXT_KEY,
    "observation_instance_id",
    "observation_artifact_identity",
    "observation_parser_identity",
    "observation_source_identity",
    "observation_run_identity",
    "constraint_artifact_identity",
    _DIRECTIVE_OPERAND_CONTEXT_KEY,
    _DEPENDENCE_OPERAND_CONTEXT_KEY,
    "directive_operand_identity",
    _DIRECTIVE_SOURCE_CONTEXT_KEY,
    "directive_source_identity",
    _PORT_OWNERSHIP_CONTEXT_KEY,
    "port_owner_id",
    "configured_component_id",
    "port_ownership_identity",
    _REQUESTED_DIRECTIVE_CONTEXT_KEY,
    _OBSERVATION_ARTIFACT_KIND_CONTEXT_KEY,
    _TOOL_ARTIFACT_CONTEXT_KEY,
    "tool_artifact_identity",
    "tool_artifact_run_identity",
    _CONSTRAINT_INPUT_CONTEXT_KEY,
    _SEMANTIC_ARTIFACT_EVIDENCE_CONTEXT_KEY,
    _AGGREGATE_EVIDENCE_CONTEXT_KEY,
    "aggregate_evidence_contract",
    "aggregate_evidence_identity",
    "aggregate_semantic_attestation_identity",
    "aggregate_source_artifact_identity",
    "adapter_contract",
    "adapter_version",
    "artifact_byte_closure",
    "artifact_identity",
    "artifact_revision",
    "artifact_sha256",
    "evidence_origin_identity",
    "extraction_manifest_identity",
    "extractor_identity",
    "extractor_name",
    "extractor_version",
    "location_kind",
    "mapping_kind",
    "mapping_provenance",
    "mapping_resolution",
    "mapping_resolution_contract",
    "language_spec_compatibility_contract",
    "language_spec_family",
    "language_spec_revision",
    "language_spec_revision_source",
    "semantic_attestation_contract",
    "semantic_attestation_identity",
    "semantic_artifact_kind",
    "native_ir_artifact_identity",
    "native_ir_evidence",
    "native_ir_evidence_contract",
    "native_ir_relation_provenance",
    "resolved_target_anchor_identity",
    "resolved_target_id",
    "source_anchor_identity_contract",
    "typed_mlir_location_present",
    "typed_source_anchor_identity",
    "unique_mlir_location_mapping_resolved",
    # Rule-condition premises below are produced only by the target-local
    # resolver.  Entity/report metadata cannot self-assert their presence.
    "csim_result_present",
    "cosim_result_present",
    "csynth_report_present",
    "performance_report_present",
    "schedule_artifact_present",
    "dataflow_profiling_enabled",
    "constraint_artifact_present",
    "timing_summary_present",
    "post_route_timing_result_present",
    "utilization_report_present",
    "congestion_report_present",
    "power_report_present",
    "timing_gate_requested",
})

# These pairs describe the current public parser contract, not a filename or
# namespace heuristic.  The generic tool-evidence policy still checks the
# observation stage against its producer run; this narrower table additionally
# prevents (for example) a timing predicate from borrowing a utilization report
# produced by the same Vivado stage.
_OBSERVATION_REPORT_POLICY: dict[
    tuple[str, str], frozenset[tuple[str, str]]
] = {
    **{
        (predicate, "csim"): frozenset({
            ("amd.vitis.csim_result", "verification_evidence"),
        })
        for predicate in (
            "csim.exit_code", "csim.mismatches", "csim.assertions_failed",
        )
    },
    **{
        (predicate, "cosim"): frozenset({
            ("amd.vitis.cosim_rpt", "verification_evidence"),
            ("amd.vitis.cosim_report", "verification_evidence"),
        })
        for predicate in (
            "cosim.status", "cosim.latency_min_cycles",
            "cosim.latency_avg_cycles", "cosim.latency_max_cycles",
            "cosim.interval_min_cycles", "cosim.interval_avg_cycles",
            "cosim.interval_max_cycles",
        )
    },
    **{
        (predicate, "cosim"): frozenset({
            ("amd.vitis.dataflow_profile", "verification_evidence"),
        })
        for predicate in (
            "profile.fifo_max_occupancy", "profile.read_block_cycles",
            "profile.write_block_cycles", "profile.token_count",
        )
    },
    ("clock.estimated_period_ns", "schedule"): frozenset({
        ("amd.vitis.csynth_xml", "tool_observation"),
    }),
    **{
        (predicate, "schedule"): frozenset({
            ("amd.vitis.csynth_xml", "tool_observation"),
        })
        for predicate in (
            "qor.latency_best_cycles", "qor.latency_worst_cycles",
            "qor.interval_min_cycles", "qor.interval_max_cycles",
            "qor.latency_cycles", "qor.iteration_latency_cycles",
        )
    },
    ("qor.target_ii", "schedule"): frozenset({
        ("amd.vitis.schedule_json", "compiler_decision"),
    }),
    ("qor.achieved_ii", "schedule"): frozenset({
        ("amd.vitis.csynth_xml", "tool_observation"),
        ("amd.vitis.schedule_json", "compiler_decision"),
    }),
    **{
        (predicate, "schedule"): frozenset({
            ("amd.vitis.schedule_json", "compiler_decision"),
        })
        for predicate in (
            "schedule.start_cycle", "schedule.end_cycle",
            "schedule.pipeline_stage", "schedule.operation_latency",
        )
    },
    **{
        (predicate, "schedule"): frozenset({
            ("amd.vitis.directive_status", "tool_observation"),
        })
        for predicate in (
            "directive.tool_status", "directive.reported_requested",
            "directive.tool_effective", "directive.achieved",
        )
    },
    **{
        (f"resource.{resource}", "schedule"): frozenset({
            ("amd.vitis.csynth_xml", "tool_observation"),
        })
        for resource in ("lut", "ff", "dsp", "bram_18k", "uram")
    },
    **{
        (predicate, stage): frozenset({
            ("amd.vivado.timing_summary", "tool_observation"),
            ("amd.vivado.post_route_timing", "tool_observation"),
        })
        for predicate in ("timing.wns_ns", "timing.tns_ns")
        for stage in ("post_synth", "post_place", "post_route")
    },
    **{
        (f"resource.{resource}", stage): frozenset({
            ("amd.vivado.utilization", "tool_observation"),
            ("amd.vivado.post_route_utilization", "tool_observation"),
        })
        for resource in ("lut", "ff", "dsp", "bram_18k", "uram")
        for stage in ("post_synth", "post_place", "post_route")
    },
    **{
        (predicate, stage): frozenset({
            ("amd.vivado.physical_summary", "tool_observation"),
            ("amd.vivado.qor_summary", "tool_observation"),
        })
        for predicate in (
            "physical.congestion_level", "timing.critical_path_delay_ns",
            "physical.drc_errors", "physical.cdc_critical",
            "power.dynamic_w", "power.static_w",
        )
        for stage in ("post_synth", "post_place", "post_route")
    },
}

_OBSERVATION_PARSER_POLICY: dict[tuple[str, str], frozenset[str]] = {
    ("amd.vitis.reports", "1"): frozenset({
        kind
        for choices in _OBSERVATION_REPORT_POLICY.values()
        for kind, _authority in choices
        if kind.startswith("amd.vitis.")
    }),
    ("amd.vivado.reports", "1"): frozenset({
        kind
        for choices in _OBSERVATION_REPORT_POLICY.values()
        for kind, _authority in choices
        if kind.startswith("amd.vivado.")
    }),
}

# A report container is executable knowledge evidence only when the current
# artifact itself closes to a fresh real run, an immutable run manifest, one
# declared output, and the same live managed bytes.  The canonical stage is
# intentionally explicit: Vitis CSYNTH output describes the schedule plane,
# while Vivado output keeps its implementation run stage.
_TOOL_ARTIFACT_STAGE_POLICY: dict[str, dict[str, str]] = {
    "amd.vitis.csynth_xml": {"csynth": "schedule", "schedule": "schedule"},
    "amd.vitis.schedule_json": {"csynth": "schedule", "schedule": "schedule"},
    "amd.vivado.timing_summary": {
        stage: stage for stage in ("post_synth", "post_place", "post_route")
    },
    "amd.vivado.post_route_timing": {"post_route": "post_route"},
    "amd.vivado.utilization": {
        stage: stage for stage in ("post_synth", "post_place", "post_route")
    },
    "amd.vivado.post_route_utilization": {"post_route": "post_route"},
}
_AMD_TYPED_OBSERVATION_PREDICATES = frozenset(
    predicate for predicate, _stage in _OBSERVATION_REPORT_POLICY
)
_DYNAMIC_OBSERVATION_PREDICATES = frozenset(
    predicate for predicate in _AMD_TYPED_OBSERVATION_PREDICATES
    if predicate.startswith(("csim.", "cosim.", "profile."))
)
_CSYNTH_ESTIMATE_PREDICATES = frozenset({
    "clock.estimated_period_ns",
    "qor.latency_best_cycles", "qor.latency_worst_cycles",
    "qor.interval_min_cycles", "qor.interval_max_cycles",
    "qor.latency_cycles", "qor.iteration_latency_cycles",
    "qor.achieved_ii",
    "resource.lut", "resource.ff", "resource.dsp",
    "resource.bram_18k", "resource.uram",
})

_CONDITION_OBSERVATION_ARTIFACTS: dict[str, frozenset[str]] = {
    "csim_result_present": frozenset({"amd.vitis.csim_result"}),
    "cosim_result_present": frozenset({
        "amd.vitis.cosim_rpt", "amd.vitis.cosim_report",
    }),
    "csynth_report_present": frozenset({"amd.vitis.csynth_xml"}),
    "performance_report_present": frozenset({
        "amd.vitis.csynth_xml", "amd.vitis.schedule_json",
    }),
    "schedule_artifact_present": frozenset({"amd.vitis.schedule_json"}),
    "dataflow_profiling_enabled": frozenset({"amd.vitis.dataflow_profile"}),
    "timing_summary_present": frozenset({
        "amd.vivado.timing_summary", "amd.vivado.post_route_timing",
    }),
    "post_route_timing_result_present": frozenset({
        "amd.vivado.post_route_timing", "amd.vivado.physical_summary",
        "amd.vivado.qor_summary",
    }),
    "utilization_report_present": frozenset({
        "amd.vivado.utilization", "amd.vivado.post_route_utilization",
    }),
    "congestion_report_present": frozenset({
        "amd.vivado.physical_summary", "amd.vivado.qor_summary",
    }),
    "power_report_present": frozenset({
        "amd.vivado.physical_summary", "amd.vivado.qor_summary",
    }),
}

# Canonical spellings for MLIR Builtin Location subclasses that a native or
# text adapter may preserve on a SourceAnchor.  A merely ``mlir.*``-prefixed
# string is not a typed Location and must not qualify mapping guidance.
_MLIR_LOCATION_KINDS = frozenset({
    "mlir.callsite", "mlir.filelinecol", "mlir.filelinecol.redacted",
    "mlir.fused", "mlir.name", "mlir.opaque",
})
_CONCRETE_MLIR_MAPPING_LOCATION_KINDS = (
    _MLIR_LOCATION_KINDS - {"mlir.filelinecol.redacted"}
)
_MLIR_LOCATION_RESOLUTION_CONTRACT = "hlsgraph.mlir_location_resolution.v1"
_SOURCE_ANCHOR_IDENTITY_CONTRACT = "hlsgraph.source_anchor_identity.v1"
_LLVM_PROJECT_SPEC_REVISION = "git-429c88d37f1f02e68ebc1fc7b0da4511ce6407e3"
_CIRCT_HANDSHAKE_SPEC_REVISION = "git-ef03d45c960607315a8b62903b92d072d8542e30"
_LANGUAGE_SPEC_REVISION_SOURCE = "immutable_extractor_attestation.v1"
_LANGUAGE_SPEC_CONTRACTS: dict[str, tuple[str, str, frozenset[str]]] = {
    "mlir": (
        _LLVM_PROJECT_SPEC_REVISION,
        "hlsgraph.mlir.language_spec_compatibility.v1",
        frozenset({"ir.mlir"}),
    ),
    "llvm": (
        _LLVM_PROJECT_SPEC_REVISION,
        "hlsgraph.llvm.language_spec_compatibility.v1",
        frozenset({"ir.llvm"}),
    ),
    "circt.handshake": (
        _CIRCT_HANDSHAKE_SPEC_REVISION,
        "hlsgraph.circt.handshake_spec_compatibility.v1",
        frozenset({"ir.mlir"}),
    ),
}
_NATIVE_MLIR_SSA_EVIDENCE_CONTRACT = "hlsgraph.mlir.ssa_def_use.v1"
_SOURCE_MAPPING_TARGET_KINDS = frozenset({
    "hls.kernel", "hls.function", "hls.loop", "hls.memory", "hls.port",
    "hls.stream", "source.variable",
})
_EXPLICIT_IR_WIDTH = re.compile(
    r"(?:\b(?:ap_|ac_)?(?:u?int|u?fixed)\s*<\s*([1-9]\d*)|"
    r"\b_?BitInt\s*\(\s*([1-9]\d*)\s*\)|"
    r"\b(?:u?int)([1-9]\d*)_t\b)",
    re.I,
)
_LLVM_INDEX_OPCODES = frozenset({
    "getelementptr", "extractelement", "insertelement",
    "extractvalue", "insertvalue",
})
_LLVM_MEMORY_ACCESS_KINDS = {
    "load": "load",
    "store": "store",
    "getelementptr": "address",
    "atomicrmw": "atomic",
    "cmpxchg": "atomic",
    "fence": "ordering",
}

# A relation can be useful evidence without being hardware topology.  The
# following relations are therefore assigned exactly zero graph-propagation
# weight instead of being silently reinterpreted as architecture edges.
_ZERO_WEIGHT_RELATIONS = frozenset({
    "software.calls", "llvm.calls", "llvm.cfg", "ir.contains",
    "handshake.dataflow",
})

_RELATION_WEIGHTS: dict[str, float] = {
    "hls.streams_to": 1.0,
    "hls.annotates": 0.9,
    "hls.contains": 0.7,
    "cross.maps_to": 0.6,
    "cross.projects_to": 0.6,
    "hls.reads": 0.75,
    "hls.writes": 0.75,
    "hls.connects": 0.8,
    "hls.implements": 0.65,
}

_CHANNEL_WEIGHTS: dict[str, float] = {
    "exact": 2.0,
    "prefix": 1.2,
    "fts": 1.0,
    "bm25": 1.0,
    "fuzzy": 0.5,
    "graph": 1.2,
    "graph_evidence": 0.7,
    "adapter": 1.0,
}

_TERM_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "流水线": ("pipeline", "pipelined"),
    "启动间隔": ("ii", "initiation", "interval"),
    "间隔": ("ii", "interval"),
    "延迟": ("latency", "cycles"),
    "数据流": ("dataflow", "stream", "streams"),
    "流": ("stream", "fifo"),
    "队列": ("fifo", "stream"),
    "循环": ("loop", "trip", "bound"),
    "展开": ("unroll", "parallel"),
    "分区": ("partition", "bank", "memory"),
    "接口": ("interface", "axi", "port"),
    "资源": ("resource", "lut", "ff", "dsp", "bram", "uram"),
    "时序": ("timing", "wns", "tns", "clock"),
    "位宽": ("bitwidth", "width", "precision"),
    "精度": ("precision", "bitwidth"),
    "指令": ("directive", "pragma"),
    "约束": ("constraint", "requested", "effective"),
    "瓶颈": ("bottleneck", "recurrence", "schedule", "stall"),
    "正确性": ("correctness", "csim", "cosim"),
    "拥塞": ("congestion", "route"),
    "功耗": ("power", "activity"),
}


def _is_link_or_reparse(path: Path) -> bool:
    try:
        value = path.lstat()
    except OSError:
        return True
    if stat.S_ISLNK(value.st_mode):
        return True
    attributes = int(getattr(value, "st_file_attributes", 0))
    reparse = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    if attributes & reparse:
        return True
    is_junction = getattr(path, "is_junction", None)
    try:
        return bool(callable(is_junction) and is_junction())
    except OSError:
        return True


def _append_private_access(
    project_root: Any, *, content_sha256: str, anchor: Mapping[str, Any],
    result: str, byte_count: int,
) -> bool:
    """Append one body-free local audit event, failing closed on unsafe paths.

    The deliberately tiny record schema excludes time, query text, filenames,
    symbols, document titles, and excerpts.  A caller that intended to return
    private bytes must withhold them when this function returns ``False``.
    """
    if (not isinstance(content_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", content_sha256)
            or not isinstance(result, str)
            or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", result)
            or not isinstance(byte_count, int) or isinstance(byte_count, bool)
            or not 0 <= byte_count <= 16_000
            or not isinstance(anchor, Mapping)):
        return False
    allowed_anchor_keys = {"kind", "start_line", "end_line", "chunk_id"}
    if set(anchor) - allowed_anchor_keys:
        return False
    normalized_anchor: dict[str, Any] = {}
    for key in sorted(anchor):
        value = anchor[key]
        if key in {"start_line", "end_line"}:
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                return False
        elif (not isinstance(value, str) or not value
              or len(value) > 128 or "\x00" in value):
            return False
        normalized_anchor[key] = value
    record = {
        "content_sha256": content_sha256,
        "anchor": normalized_anchor,
        "result": result,
        "byte_count": byte_count,
    }
    payload = (json.dumps(record, ensure_ascii=True, sort_keys=True,
                          separators=(",", ":")) + "\n").encode("ascii")
    root = Path(project_root).resolve()
    ledger_root = root / ".hlsgraph"
    private_root = ledger_root / "private"
    try:
        if (_is_link_or_reparse(root) or _is_link_or_reparse(ledger_root)
                or not ledger_root.is_dir()):
            return False
        if private_root.exists():
            if _is_link_or_reparse(private_root) or not private_root.is_dir():
                return False
        else:
            private_root.mkdir(mode=0o700)
            if _is_link_or_reparse(private_root) or not private_root.is_dir():
                return False
        if os.name != "nt":
            # A read-only sandbox may expose an already-hardened private
            # directory.  Re-applying the same mode is still a metadata write
            # and fails with EROFS, which would incorrectly suppress an audit
            # append supplied through a narrowly writable file overlay.
            # Preserve fail-closed behavior for any weaker/stricter mode: only
            # the exact 0700 state avoids chmod, and the result is rechecked.
            private_mode = stat.S_IMODE(
                private_root.stat(follow_symlinks=False).st_mode
            )
            if private_mode != 0o700:
                os.chmod(private_root, 0o700)
            if stat.S_IMODE(
                private_root.stat(follow_symlinks=False).st_mode
            ) != 0o700:
                return False
        path = root / _PRIVATE_ACCESS_LOG
        if path.exists() and (_is_link_or_reparse(path) or not path.is_file()):
            return False
        flags = (os.O_WRONLY | os.O_CREAT | os.O_APPEND
                 | int(getattr(os, "O_BINARY", 0))
                 | int(getattr(os, "O_NOFOLLOW", 0)))
        descriptor = os.open(path, flags, 0o600)
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode):
                return False
            written = os.write(descriptor, payload)
            if written != len(payload):
                return False
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            current = path.lstat()
        except OSError:
            return False
        opened_identity = (int(opened.st_dev), int(opened.st_ino))
        current_identity = (int(current.st_dev), int(current.st_ino))
        if (opened_identity != current_identity or not stat.S_ISREG(current.st_mode)
                or _is_link_or_reparse(ledger_root)
                or _is_link_or_reparse(private_root)
                or _is_link_or_reparse(path) or not path.is_file()):
            return False
        if os.name != "nt":
            final_private_mode = stat.S_IMODE(
                private_root.stat(follow_symlinks=False).st_mode
            )
            if final_private_mode != 0o700:
                return False
    except OSError:
        return False
    return True


def normalize_terms(value: str) -> list[str]:
    """Return stable bilingual HLS and identifier terms for one query."""
    expanded = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value)
    expanded = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", expanded)
    raw = re.findall(r"[A-Za-z0-9]+|[\u3400-\u9fff]+", expanded, re.UNICODE)
    result: list[str] = []
    seen: set[str] = set()

    def add(term: str) -> None:
        folded = term.casefold().strip()
        if folded and folded not in seen:
            seen.add(folded)
            result.append(folded)

    for token in raw:
        add(token)
        for chinese, aliases in _TERM_EXPANSIONS.items():
            if chinese in token:
                add(chinese)
                for alias in aliases:
                    add(alias)
    return result


@dataclass(slots=True)
class RetrievalSpec:
    query: str
    snapshot_id: str | None = None
    scope_id: str | None = None
    view: str = "architecture"
    planes: tuple[str, ...] = DEFAULT_PLANES
    profile: str = DEFAULT_RETRIEVAL_PROFILE
    top_k: int = 8
    max_chars: int | None = None
    include_private_snippets: bool = False
    include_predictions: bool = False
    applicability: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.query, str) or not self.query.strip():
            raise ValueError("retrieval query must be a non-empty string")
        if len(self.query) > 4_096 or "\x00" in self.query:
            raise ValueError("retrieval query must not contain NUL and is limited to 4096 characters")
        if self.view not in {"architecture", "evidence"}:
            raise ValueError("view must be architecture or evidence")
        self.planes = tuple(dict.fromkeys(str(item) for item in self.planes))
        unknown = set(self.planes) - VALID_PLANES
        if unknown:
            raise ValueError(f"unsupported retrieval planes: {', '.join(sorted(unknown))}")
        if type(self.include_private_snippets) is not bool:
            raise ValueError("include_private_snippets must be a boolean")
        if type(self.include_predictions) is not bool:
            raise ValueError("include_predictions must be a boolean")
        if "predictions" in self.planes and not self.include_predictions:
            raise ValueError(
                "the predictions plane requires include_predictions=True"
            )
        # ``planes`` is the final channel allow-list consumed by every
        # retriever path.  The separate flag is the explicit capability gate:
        # opting in adds the isolated prediction plane without changing the
        # fact or guidance channels selected by the caller.
        if self.include_predictions and "predictions" not in self.planes:
            self.planes = (*self.planes, "predictions")
        if not 1 <= int(self.top_k) <= 50:
            raise ValueError("top_k must be in 1..50")
        self.top_k = int(self.top_k)
        if self.max_chars is not None:
            if not 1_000 <= int(self.max_chars) <= 24_000:
                raise ValueError("max_chars must be in 1000..24000")
            self.max_chars = int(self.max_chars)
        if self.profile != DEFAULT_RETRIEVAL_PROFILE:
            raise ValueError(
                f"unsupported retrieval profile: {self.profile!r}; "
                f"expected {DEFAULT_RETRIEVAL_PROFILE!r}"
            )
        if not isinstance(self.applicability, dict) or not all(
            isinstance(key, str) and key and isinstance(item, str) and item
            for key, item in self.applicability.items()
        ):
            raise ValueError("applicability must map non-empty strings to non-empty strings")


@dataclass(slots=True)
class RetrievalItem:
    record_id: str
    plane: str
    record_kind: str
    title: str
    summary: str
    score: float = 0.0
    authority_class: str | None = None
    stage: str | None = None
    completeness: str = "complete"
    entity_id: str | None = None
    evidence_ids: list[str] = field(default_factory=list)
    citation: dict[str, Any] | None = None
    score_channels: dict[str, float] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.plane not in VALID_PLANES:
            raise ValueError(f"unsupported retrieval item plane: {self.plane}")
        if not self.record_id or not self.record_kind or not self.title:
            raise ValueError("retrieval item requires record_id, record_kind, and title")
        self.evidence_ids = sorted(set(self.evidence_ids))
        self.score_channels = dict(sorted(self.score_channels.items()))


@dataclass(slots=True)
class RetrievalTrace:
    query_sha256: str
    snapshot_id: str
    profile: str
    profile_hash: str
    graph_hash: str
    profile_schema_version: str = RETRIEVAL_PROFILE_SCHEMA_VERSION
    algorithm_version: str = RETRIEVAL_ALGORITHM_VERSION
    candidate_counts: dict[str, int] = field(default_factory=dict)
    elapsed_ms: dict[str, float] = field(default_factory=dict)
    output_budget_chars: int = 0
    output_chars: int = 0
    truncated: bool = False
    private_snippets_requested: bool = False
    private_snippets_returned: bool = False
    adapter_fingerprints: list[str] = field(default_factory=list)
    semantic_channel: str = "optional_not_enabled"


@dataclass(slots=True)
class RetrievalResult:
    snapshot_id: str
    facts: list[RetrievalItem]
    guidance: list[RetrievalItem]
    predictions: list[RetrievalItem]
    flow: list[dict[str, Any]]
    citations: list[dict[str, Any]]
    ambiguities: list[dict[str, Any]]
    confidence: str
    incomplete: bool
    stale: bool
    warnings: list[str]
    trace: RetrievalTrace
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        value = json_ready(self)
        trace = value.get("trace", {})
        if isinstance(trace, dict) and not trace.get("adapter_fingerprints"):
            trace.pop("adapter_fingerprints", None)
        return value


@runtime_checkable
class RetrievalAdapter(Protocol):
    """Optional local-only retrieval channel.

    Adapters return already policy-projected items.  Merely requesting private
    snippets does not authorize an adapter to return them; authorization remains
    an adapter/project responsibility.
    """

    adapter_id: str

    def search(self, spec: RetrievalSpec, terms: Sequence[str],
               limit: int) -> Iterable[RetrievalItem]: ...


class LocalKnowledgeRetrievalAdapter:
    """Read-only adapter for the private, rebuildable local knowledge sidecar."""

    adapter_id = "hlsgraph.local_knowledge_sidecar.v1"

    def __init__(self, project_root: Any, *, allow_private_snippets: bool = False):
        from .knowledge import LocalKnowledgeSidecar

        self.sidecar = LocalKnowledgeSidecar(project_root)
        self.project_root = Path(project_root).resolve()
        self.allow_private_snippets = bool(allow_private_snippets)
        self.warnings: list[str] = []
        self.document_hashes: dict[str, str] = {}
        try:
            manifest = self.sidecar.manifest()
            self.document_hashes = dict(manifest.document_hashes)
            self.semantic_index_available = manifest.embedder_fingerprint is not None
            self.fingerprint = stable_hash({
                "adapter": self.adapter_id, "index_manifest": manifest.id,
            })
        except Exception:
            self.semantic_index_available = False
            self.fingerprint = stable_hash({"adapter": self.adapter_id, "index": "unavailable"})

    def search(self, spec: RetrievalSpec, terms: Sequence[str],
               limit: int) -> Iterable[RetrievalItem]:
        if "local" not in spec.planes:
            return []
        include_text = spec.include_private_snippets and self.allow_private_snippets
        queries = [spec.query, *terms[:8]]
        by_id: dict[str, Any] = {}
        for query in dict.fromkeys(item.strip() for item in queries if item.strip()):
            try:
                hits = self.sidecar.search(query, limit=limit, include_text=include_text)
            except Exception:
                if spec.include_private_snippets:
                    for document_key, document_hash in sorted(self.document_hashes.items()):
                        if not _append_private_access(
                            self.project_root, content_sha256=document_hash,
                            anchor={
                                "kind": "knowledge_index",
                                "chunk_id": "document_" + stable_hash(document_key)[:24],
                            },
                            result="search_failed", byte_count=0,
                        ):
                            self.warnings.append(
                                "private_access_log_failed:local_knowledge"
                            )
                raise
            for hit in hits:
                previous = by_id.get(hit.chunk_id)
                score = -float(hit.score) if float(hit.score) < 0 else float(hit.score)
                previous_score = (-float(previous.score) if previous is not None
                                  and float(previous.score) < 0
                                  else float(previous.score) if previous is not None else -1.0)
                if previous is None or score > previous_score:
                    by_id[hit.chunk_id] = hit
        result: list[RetrievalItem] = []
        for hit in sorted(
            by_id.values(),
            key=lambda item: (-(-float(item.score) if float(item.score) < 0
                                else float(item.score)), item.chunk_id),
        )[:limit]:
            score = -float(hit.score) if float(hit.score) < 0 else float(hit.score)
            data: dict[str, Any] = {
                "document_id": hit.document_id,
                "document_version": hit.document_version,
                "heading": hit.heading,
                "chunk_sha256": hit.chunk_sha256,
                "channel": hit.channel,
                # A user-owned sidecar is searchable evidence, not a reviewed
                # public KnowledgeRule.  Promotion into a rule pack is a
                # separate, explicit review workflow.
                "review_status": "local_unreviewed",
            }
            excerpt_authorized = False
            if spec.include_private_snippets:
                document_key = f"{hit.document_id}@{hit.document_version}"
                document_hash = self.document_hashes.get(document_key)
                outcome = ("returned" if hit.excerpt is not None and include_text
                           else "denied_policy" if not self.allow_private_snippets
                           else "unavailable")
                excerpt_bytes = (len(hit.excerpt.encode("utf-8"))
                                 if hit.excerpt is not None and include_text else 0)
                logged = bool(document_hash) and _append_private_access(
                    self.project_root,
                    content_sha256=document_hash or "",
                    anchor={"kind": "knowledge_chunk", "chunk_id": hit.chunk_id},
                    result=outcome, byte_count=excerpt_bytes,
                )
                if not logged:
                    self.warnings.append("private_access_log_failed:local_knowledge")
                excerpt_authorized = bool(
                    logged and outcome == "returned" and hit.excerpt is not None
                )
            if excerpt_authorized:
                data.update({
                    "private_excerpt": hit.excerpt,
                    "authorization": "project_bounded",
                })
            result.append(RetrievalItem(
                record_id=hit.chunk_id,
                plane="local",
                record_kind="local_document_chunk",
                title=hit.heading or hit.title or hit.document_id,
                summary=(f"Unreviewed local document metadata for {hit.document_id} "
                         f"{hit.document_version}; body is project-private."),
                score=max(0.000001, score),
                authority_class="local_document_excerpt",
                completeness="complete",
                citation={
                    "document_id": hit.document_id,
                    "document_version": hit.document_version,
                    "section": hit.heading,
                    "local": True,
                },
                data=data,
            ))
        return result


class SourceSnippetRetrievalAdapter:
    """Read anchor-bounded snippets from still-valid project source artifacts."""

    adapter_id = "hlsgraph.source_snippets.v1"
    canonical_capability = "hlsgraph.canonical_source_anchor_projection.v1"
    _MAX_ARTIFACT_BYTES = 16 * 1024 * 1024

    def __init__(self, bundle: GraphBundle, *, allow_private_snippets: bool = False):
        self.bundle = bundle
        self.allow_private_snippets = bool(allow_private_snippets)
        self.warnings: list[str] = []
        self.fingerprint = stable_hash({"adapter": self.adapter_id, "version": "1"})

    def search(self, spec: RetrievalSpec, terms: Sequence[str],
               limit: int) -> Iterable[RetrievalItem]:
        del spec, terms, limit
        return []

    @staticmethod
    def _is_link_or_reparse(path: Path) -> bool:
        return _is_link_or_reparse(path)

    @classmethod
    def _safe_path(cls, root: Path, uri: str) -> Path | None:
        relative = Path(uri)
        if relative.is_absolute() or not relative.parts or any(
            item in {"", ".", ".."} for item in relative.parts
        ):
            return None
        current = root
        if cls._is_link_or_reparse(current):
            return None
        for component in relative.parts:
            current = current / component
            if cls._is_link_or_reparse(current):
                return None
        try:
            current.absolute().relative_to(root.absolute())
        except ValueError:
            return None
        return current

    @classmethod
    def _verified_bytes(cls, root: Path, artifact: Any) -> tuple[bytes | None, str]:
        if str(artifact.retention) != "external":
            return (None, "not_external")
        if not (artifact.kind.startswith("source.")
                or artifact.kind.startswith("testbench.")):
            return (None, "not_source_or_testbench")
        if artifact.size > cls._MAX_ARTIFACT_BYTES:
            return (None, "artifact_too_large")
        path = cls._safe_path(root, artifact.uri)
        if path is None:
            return (None, "unsafe_path")
        flags = os.O_RDONLY | int(getattr(os, "O_BINARY", 0)) | int(getattr(os, "O_NOFOLLOW", 0))
        descriptor = -1
        try:
            before = path.lstat()
            if not stat.S_ISREG(before.st_mode):
                return (None, "not_regular")
            descriptor = os.open(path, flags)
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode):
                return (None, "not_regular")
            chunks: list[bytes] = []
            remaining = cls._MAX_ARTIFACT_BYTES + 1
            while remaining:
                value = os.read(descriptor, min(1024 * 1024, remaining))
                if not value:
                    break
                chunks.append(value)
                remaining -= len(value)
            after = os.fstat(descriptor)
        except OSError:
            return (None, "read_failed")
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        try:
            current = path.lstat()
        except OSError:
            return (None, "changed_during_read")
        identity = lambda item: (
            int(item.st_dev), int(item.st_ino), int(item.st_size), int(item.st_mtime_ns),
        )
        if identity(before) != identity(opened) or identity(opened) != identity(after) \
                or identity(after) != identity(current):
            return (None, "changed_during_read")
        if cls._safe_path(root, artifact.uri) != path:
            return (None, "changed_during_read")
        data = b"".join(chunks)
        if len(data) != artifact.size:
            return (None, "size_mismatch")
        if hashlib.sha256(data).hexdigest() != artifact.sha256:
            return (None, "hash_mismatch")
        return (data, "verified")

    def source_snippets(self, spec: RetrievalSpec, facts: Sequence[RetrievalItem],
                        *, snapshot_id: str, limit: int) -> list[RetrievalItem]:
        if not spec.include_private_snippets or limit <= 0:
            return []
        graph = self.bundle.store.load_graph(snapshot_id)
        artifacts = {item.id: item for item in self.bundle.store.artifacts(graph.snapshot_id)}
        result: list[RetrievalItem] = []
        seen: set[tuple[str, str, int | None, int | None]] = set()
        for fact in facts:
            if fact.record_kind != "entity" or not fact.entity_id:
                continue
            entity = graph.entities.get(fact.entity_id)
            if entity is None:
                continue
            for anchor in entity.anchors:
                key = (entity.id, anchor.artifact_id, anchor.start_line, anchor.end_line)
                if key in seen:
                    continue
                seen.add(key)
                artifact = artifacts.get(anchor.artifact_id)
                if artifact is None:
                    self.warnings.append(f"source_snippet_missing_artifact:{anchor.artifact_id}")
                    continue
                audit_anchor = {
                    "kind": "source_line",
                    "start_line": anchor.start_line or 1,
                    "end_line": anchor.end_line or anchor.start_line or 1,
                }
                if not self.allow_private_snippets:
                    if not _append_private_access(
                        self.bundle.project_root,
                        content_sha256=artifact.sha256,
                        anchor=audit_anchor, result="denied_policy", byte_count=0,
                    ):
                        self.warnings.append("private_access_log_failed:source")
                    continue
                data, status = self._verified_bytes(self.bundle.project_root, artifact)
                if data is None:
                    if not _append_private_access(
                        self.bundle.project_root,
                        content_sha256=artifact.sha256,
                        anchor=audit_anchor, result=status, byte_count=0,
                    ):
                        self.warnings.append("private_access_log_failed:source")
                    if status not in {"not_external", "not_source_or_testbench"}:
                        self.warnings.append(
                            f"source_snippet_{status}:{anchor.artifact_id}"
                        )
                    continue
                try:
                    text = data.decode("utf-8")
                except UnicodeDecodeError:
                    if not _append_private_access(
                        self.bundle.project_root,
                        content_sha256=artifact.sha256,
                        anchor=audit_anchor, result="non_utf8", byte_count=0,
                    ):
                        self.warnings.append("private_access_log_failed:source")
                    self.warnings.append(
                        f"source_snippet_non_utf8:{anchor.artifact_id}"
                    )
                    continue
                lines = text.splitlines()
                if not lines or anchor.start_line is None:
                    continue
                start = max(1, anchor.start_line)
                end = max(start, anchor.end_line or start)
                start = max(1, start - 2)
                end = min(len(lines), end + 2, start + 79)
                if start > len(lines) or end < start:
                    if not _append_private_access(
                        self.bundle.project_root,
                        content_sha256=artifact.sha256,
                        anchor=audit_anchor, result="anchor_out_of_range", byte_count=0,
                    ):
                        self.warnings.append("private_access_log_failed:source")
                    self.warnings.append(
                        f"source_snippet_anchor_out_of_range:{anchor.artifact_id}"
                    )
                    continue
                excerpt = "\n".join(lines[start - 1:end])
                if len(excerpt) > 4_000:
                    excerpt = excerpt[:4_000]
                returned_anchor = {
                    "kind": "source_line", "start_line": start, "end_line": end,
                }
                if not _append_private_access(
                    self.bundle.project_root,
                    content_sha256=artifact.sha256,
                    anchor=returned_anchor, result="returned",
                    byte_count=len(excerpt.encode("utf-8")),
                ):
                    self.warnings.append("private_access_log_failed:source")
                    continue
                snippet_id = "source_snippet_" + stable_hash({
                    "snapshot": graph.snapshot_id,
                    "entity": entity.id,
                    "artifact": artifact.id,
                    "start": start,
                    "end": end,
                    "excerpt": hashlib.sha256(excerpt.encode("utf-8")).hexdigest(),
                })[:24]
                result.append(RetrievalItem(
                    record_id=snippet_id,
                    plane="evidence",
                    record_kind="source_snippet",
                    title=f"Source anchor for {entity.qualified_name or entity.name}",
                    summary=(f"Authorized project-local source lines {start}-{end}; "
                             "the full artifact remains external."),
                    score=max(0.000001, fact.score * 0.99),
                    authority_class="static_fact",
                    stage="source",
                    completeness=str(entity.completeness),
                    entity_id=entity.id,
                    evidence_ids=[entity.id, artifact.id],
                    score_channels={"anchor": 1.0},
                    data={
                        "artifact_id": artifact.id,
                        "artifact_sha256": artifact.sha256,
                        "anchor": {
                            "start_line": start, "end_line": end,
                            "symbol": anchor.symbol,
                        },
                        "projection_provenance": self.adapter_id,
                        "canonical_adapter_capability": self.canonical_capability,
                        "private_excerpt": excerpt,
                        "authorization": "project_bounded",
                        "excerpt_sha256": hashlib.sha256(
                            excerpt.encode("utf-8")
                        ).hexdigest(),
                    },
                ))
                if len(result) >= limit:
                    return result
        return result


def default_retrieval_adapters(bundle: GraphBundle) -> tuple[RetrievalAdapter, ...]:
    """Discover only project-local, non-network retrieval channels.

    Existence of a sidecar enables metadata search.  Returning its text still
    requires both ``RetrievalSpec.include_private_snippets`` and the manifest's
    explicit ``privacy.mcp_source_snippets = bounded`` policy.
    """
    root = bundle.project_root / ".hlsgraph" / "private" / "knowledge"
    metadata = bundle.manifest.metadata
    privacy = metadata.get("privacy", {}) if isinstance(metadata, Mapping) else {}
    allowed = (
        isinstance(privacy, Mapping)
        and privacy.get("mcp_source_snippets") == "bounded"
    )
    result: list[RetrievalAdapter] = []
    result.append(SourceSnippetRetrievalAdapter(
        bundle, allow_private_snippets=allowed,
    ))
    if (root / "manifest.json").is_file() and (root / "chunks.sqlite").is_file():
        result.append(LocalKnowledgeRetrievalAdapter(
            bundle.project_root, allow_private_snippets=allowed,
        ))
    return tuple(result)


@dataclass(slots=True)
class _Document:
    key: str
    text: str
    fields: tuple[str, ...]
    item: RetrievalItem
    entity_id: str | None = None


@dataclass(slots=True)
class _BindingEvaluation:
    """Detached result of one session-local atomic binding evaluation.

    This is ordinary retrieval output, never an authorization token.  No API
    accepts it as authority for another match.
    """

    binding: Any
    rule: Any
    values: Mapping[str, set[str]]
    request_matches: bool
    binding_matches: bool
    revision_unbound: bool
    rule_applicable: bool
    rule_reason: str


@dataclass(slots=True)
class _ReviewedKnowledgeSurface:
    """Exact live records closed to a revalidated review-ready inventory."""

    rule_ids: set[str] = field(default_factory=set)
    binding_ids: set[str] = field(default_factory=set)
    rule_fingerprints: dict[str, str] = field(default_factory=dict)
    binding_fingerprints: dict[str, str] = field(default_factory=dict)
    rejected: bool = False


@dataclass(slots=True)
class _RuleEvaluation:
    rule: Any
    applicable: bool
    reason: str


def _text(value: Any, limit: int = 1200) -> str:
    try:
        rendered = json.dumps(json_ready(value), ensure_ascii=False, sort_keys=True,
                              separators=(",", ":"))
    except (TypeError, ValueError, UnicodeError):
        rendered = ""
    return rendered[:limit]


def _document_tokens(value: str) -> list[str]:
    return normalize_terms(value)


def _bm25(documents: Sequence[_Document], query_terms: Sequence[str]) -> dict[str, float]:
    if not documents or not query_terms:
        return {}
    tokenized = [_document_tokens(item.text) for item in documents]
    average = sum(len(item) for item in tokenized) / max(1, len(tokenized))
    frequencies: dict[str, int] = defaultdict(int)
    for values in tokenized:
        for term in set(values):
            frequencies[term] += 1
    scores: dict[str, float] = {}
    for document, values in zip(documents, tokenized):
        if not values:
            continue
        counts: dict[str, int] = defaultdict(int)
        for token in values:
            counts[token] += 1
        score = 0.0
        for term in query_terms:
            count = counts.get(term, 0)
            if not count:
                continue
            document_frequency = frequencies.get(term, 0)
            inverse = math.log(1.0 + (len(documents) - document_frequency + 0.5)
                               / (document_frequency + 0.5))
            denominator = count + 1.5 * (1.0 - 0.75 + 0.75 * len(values) / max(1.0, average))
            score += inverse * (count * 2.5 / denominator)
        if score > 0:
            scores[document.key] = score
    return scores


class HybridRetriever:
    """Snapshot-pinned deterministic hybrid retriever."""

    def __init__(self, bundle: GraphBundle, snapshot_id: str,
                 adapters: Sequence[RetrievalAdapter] = ()):
        self.bundle = bundle
        self.snapshot_id = snapshot_id
        self.adapters = tuple(adapters)
        self._directive_replay_cache: tuple[str, DirectiveReplayIndex] | None = None

    def retrieve(self, spec: RetrievalSpec) -> RetrievalResult:
        started = perf_counter()
        snapshot_id = spec.snapshot_id or self.snapshot_id
        if snapshot_id != self.snapshot_id:
            return HybridRetriever(self.bundle, snapshot_id, self.adapters).retrieve(spec)
        graph = self.bundle.store.load_graph(snapshot_id)
        view_graph = self._view_graph(graph, spec.view)
        allowed_ids = self._scope_ids(view_graph, spec.scope_id)
        terms = normalize_terms(spec.query)
        query_folded = spec.query.strip().casefold()
        warnings: list[str] = []
        all_documents = self._documents(spec, view_graph, allowed_ids, warnings)
        prediction_documents = [item for item in all_documents
                                if item.item.plane == "predictions"]
        knowledge_documents = [item for item in all_documents
                               if item.item.plane == "knowledge"]
        # Facts/evidence own a closed lexical corpus.  Knowledge-pack rows,
        # private sidecars, and predictions are ranked later in physically
        # separate channels, so adding a rule cannot change fact BM25 IDF,
        # lexical ranks, graph seeds, RRF denominators, or score normalization.
        documents = [item for item in all_documents
                     if item.item.plane in {"facts", "evidence"}]
        documents_by_key = {item.key: item for item in documents}
        documents_by_entity: dict[str, list[_Document]] = defaultdict(list)
        for document in documents:
            if document.entity_id:
                documents_by_entity[document.entity_id].append(document)
        collected_ms = (perf_counter() - started) * 1000.0

        channel_scores: dict[str, dict[str, float]] = defaultdict(dict)
        for document in documents:
            folded_fields = tuple(item.casefold() for item in document.fields if item)
            if query_folded in folded_fields:
                channel_scores["exact"][document.key] = 1.0
            coverage = sum(1 for term in terms if term in _document_tokens(document.text))
            if (any(item.startswith(query_folded) or query_folded in item
                    for item in folded_fields) or (terms and coverage == len(terms))):
                channel_scores["prefix"][document.key] = (
                    0.5 + 0.5 * coverage / max(1, len(terms))
                )

        channel_scores["bm25"].update(_bm25(documents, terms))

        # Reuse the canonical SQLite FTS channel for entities.  Single-term
        # probes supplement the existing all-terms expression without changing
        # its persistence or query semantics.
        fts_queries = [spec.query.strip(), *terms[:8]]
        seen_queries: set[str] = set()
        for fts_query in fts_queries:
            if not fts_query or fts_query.casefold() in seen_queries:
                continue
            seen_queries.add(fts_query.casefold())
            for hit in self.bundle.store.search_entities(snapshot_id, fts_query, limit=100):
                entity_id = str(hit["entity_id"])
                key = f"entity:{entity_id}"
                if key not in documents_by_key:
                    continue
                score = max(0.000001, float(hit.get("score", 0.0)))
                channel_scores["fts"][key] = max(channel_scores["fts"].get(key, 0.0), score)

        # Fuzzy fallback remains bounded and deterministic for large graphs.
        normalized_query = " ".join(terms)
        fuzzy_documents = [item for item in documents if item.item.record_kind == "entity"][:10_000]
        for document in fuzzy_documents:
            if document.key in channel_scores["exact"]:
                continue
            score = difflib.SequenceMatcher(
                None, normalized_query, " ".join(_document_tokens(document.text)),
            ).ratio()
            if score >= 0.58:
                channel_scores["fuzzy"][document.key] = score
        if len([item for item in documents if item.item.record_kind == "entity"]) > 10_000:
            warnings.append("fuzzy_candidate_cap_reached")

        lexical_ms = (perf_counter() - started) * 1000.0 - collected_ms

        seeds: dict[str, float] = defaultdict(float)
        for channel in ("exact", "prefix", "fts", "bm25", "fuzzy"):
            for key, score in channel_scores[channel].items():
                entity_id = documents_by_key[key].entity_id
                if entity_id:
                    seeds[entity_id] += _CHANNEL_WEIGHTS[channel] * max(0.000001, score)
        ppr, selected_relation_ids = self._typed_ppr(view_graph, allowed_ids, seeds)
        for rank, (entity_id, score) in enumerate(
            sorted(ppr.items(), key=lambda item: (-item[1], item[0])), start=1
        ):
            key = f"entity:{entity_id}"
            if key in documents_by_key:
                channel_scores["graph"][key] = score
            for document in documents_by_entity.get(entity_id, ()):
                if document.entity_id == entity_id and document.item.record_kind != "entity":
                    channel_scores["graph_evidence"][document.key] = score
        graph_ms = (perf_counter() - started) * 1000.0 - collected_ms - lexical_ms

        local_adapter_items: dict[str, RetrievalItem] = {}
        local_channel_scores: dict[str, dict[str, float]] = defaultdict(dict)
        adapter_prediction_items: dict[str, RetrievalItem] = {}
        private_returned = False
        adapter_fingerprints: list[str] = []
        semantic_index_available = False

        def collect_adapter_warnings(adapter: Any, fingerprint: str) -> None:
            values = getattr(adapter, "warnings", ())
            if not isinstance(values, list):
                return
            for warning in values:
                if (isinstance(warning, str) and len(warning) <= 256
                        and re.fullmatch(r"[A-Za-z0-9_.:+-]+", warning)):
                    warnings.append(warning)
                else:
                    warnings.append(
                        f"retrieval_adapter_warning_rejected:{fingerprint}"
                    )

        for adapter in self.adapters:
            adapter_id = str(getattr(adapter, "adapter_id", type(adapter).__name__))
            candidate_fingerprint = getattr(adapter, "fingerprint", None)
            adapter_fingerprint = (
                candidate_fingerprint.casefold()
                if isinstance(candidate_fingerprint, str)
                and re.fullmatch(r"[0-9a-fA-F]{64}", candidate_fingerprint)
                else hashlib.sha256(adapter_id.encode("utf-8")).hexdigest()
            )
            trace_adapter = (
                adapter_id != SourceSnippetRetrievalAdapter.adapter_id
                or spec.include_private_snippets
            )
            if trace_adapter:
                adapter_fingerprints.append(adapter_fingerprint)
            semantic_index_available = (
                semantic_index_available
                or getattr(adapter, "semantic_index_available", False) is True
            )
            try:
                values = list(adapter.search(spec, terms, max(20, spec.top_k * 4)))
            except Exception as exc:  # adapter failure is an explicit degradation
                warnings.append(
                    f"retrieval_adapter_failed:{adapter_fingerprint}:{type(exc).__name__}"
                )
                continue
            for position, item in enumerate(values, start=1):
                if not isinstance(item, RetrievalItem):
                    warnings.append(f"retrieval_adapter_invalid_item:{adapter_fingerprint}")
                    continue
                if item.plane not in spec.planes:
                    continue
                # Generic adapters are untrusted retrieval inputs.  They may
                # contribute only project-local unreviewed text or isolated
                # predictions.  Canonical facts/evidence are projected from
                # the ledger above; the sole source-snippet exception is an
                # exact built-in capability validated after fact ranking.
                if item.plane in {"facts", "evidence"}:
                    warnings.append(
                        "retrieval_adapter_canonical_capability_rejected:"
                        + adapter_fingerprint
                    )
                    continue
                if item.plane == "knowledge":
                    warnings.append(
                        "retrieval_adapter_public_knowledge_rejected:"
                        + adapter_fingerprint
                    )
                    continue
                if item.plane == "local" and (
                    item.record_kind != "local_document_chunk"
                    or item.authority_class != "local_document_excerpt"
                    or item.data.get("review_status") != "local_unreviewed"
                ):
                    warnings.append(
                        f"retrieval_adapter_invalid_local_authority:{adapter_fingerprint}"
                    )
                    continue
                if item.plane == "predictions":
                    if (item.record_kind != "prediction_envelope"
                            or item.authority_class != "prediction_hypothesis"):
                        warnings.append(
                            "retrieval_adapter_invalid_prediction_authority:"
                            + adapter_fingerprint
                        )
                        continue
                elif item.plane != "local":
                    warnings.append(f"retrieval_adapter_invalid_item:{adapter_fingerprint}")
                    continue
                private_excerpt = item.data.get("private_excerpt")
                public_data = {key: value for key, value in item.data.items()
                               if key != "private_excerpt"}
                try:
                    reject_embedded_body_fields(public_data, "retrieval adapter item")
                except ValueError:
                    warnings.append(f"retrieval_adapter_body_rejected:{adapter_fingerprint}")
                    continue
                if private_excerpt is not None:
                    authorized = (
                        spec.include_private_snippets
                        and item.data.get("authorization") == "project_bounded"
                        and isinstance(private_excerpt, str)
                        and len(private_excerpt) <= 4_000
                        and len(private_excerpt.splitlines()) <= 80
                    )
                    if not authorized:
                        warnings.append(
                            "retrieval_adapter_private_excerpt_rejected:"
                            + adapter_fingerprint
                        )
                        continue
                key = f"adapter:{adapter_id}:{item.record_id}"
                if item.plane == "predictions":
                    adapter_prediction_items[key] = item
                    continue
                local_adapter_items[key] = item
                local_channel_scores[f"local_adapter:{adapter_fingerprint}"][key] = (
                    max(0.000001, item.score or 1.0 / position)
                )
                if item.data.get("private_excerpt") is not None:
                    private_returned = True
            collect_adapter_warnings(adapter, adapter_fingerprint)
        items = {key: document.item for key, document in documents_by_key.items()}
        ranked, score_channels = self._rrf(channel_scores)
        maximum = max((score for _, score in ranked), default=1.0)
        ordered_items: list[RetrievalItem] = []
        for key, score in ranked:
            item = items.get(key)
            if item is None:
                continue
            item.score = round(score / maximum, 8)
            item.score_channels = {
                name: round(value, 8) for name, value in score_channels[key].items()
            }
            ordered_items.append(item)

        limit = spec.top_k
        facts = ordered_items[:limit]
        source_snippets: list[RetrievalItem] = []
        for adapter in self.adapters:
            read_snippets = getattr(adapter, "source_snippets", None)
            if not callable(read_snippets):
                continue
            adapter_fingerprint = hashlib.sha256(
                str(getattr(adapter, "adapter_id", type(adapter).__name__)).encode("utf-8")
            ).hexdigest()
            candidate_fingerprint = getattr(adapter, "fingerprint", None)
            if (isinstance(candidate_fingerprint, str)
                    and re.fullmatch(r"[0-9a-fA-F]{64}", candidate_fingerprint)):
                adapter_fingerprint = candidate_fingerprint.casefold()
            if type(adapter) is not SourceSnippetRetrievalAdapter:
                warnings.append(
                    "retrieval_adapter_canonical_capability_rejected:"
                    + adapter_fingerprint
                )
                continue
            privacy = (self.bundle.manifest.metadata.get("privacy", {})
                       if isinstance(self.bundle.manifest.metadata, Mapping) else {})
            if (not isinstance(privacy, Mapping)
                    or privacy.get("mcp_source_snippets") != "bounded"):
                # The exact built-in adapter with its own deny flag may run
                # only to append the body-free denial audit event.  Its output
                # is discarded.  A caller-constructed allow=True instance is
                # rejected before any private read.
                if getattr(adapter, "allow_private_snippets", None) is False:
                    try:
                        list(read_snippets(
                            spec, facts, snapshot_id=snapshot_id, limit=limit,
                        ))
                    except Exception as exc:
                        warnings.append(
                            f"retrieval_adapter_failed:{adapter_fingerprint}:"
                            f"{type(exc).__name__}"
                        )
                    collect_adapter_warnings(adapter, adapter_fingerprint)
                else:
                    warnings.append(
                        "retrieval_adapter_private_policy_rejected:"
                        + adapter_fingerprint
                    )
                continue
            try:
                snippets = list(read_snippets(
                    spec, facts, snapshot_id=snapshot_id, limit=limit,
                ))
            except Exception as exc:
                warnings.append(
                    f"retrieval_adapter_failed:{adapter_fingerprint}:{type(exc).__name__}"
                )
                continue
            for item in snippets:
                excerpt = item.data.get("private_excerpt") if isinstance(item, RetrievalItem) else None
                public_data = ({key: value for key, value in item.data.items()
                                if key != "private_excerpt"}
                               if isinstance(item, RetrievalItem) else {})
                body_fields_safe = True
                try:
                    reject_embedded_body_fields(public_data, "source snippet adapter item")
                except ValueError:
                    body_fields_safe = False
                if (not isinstance(item, RetrievalItem) or item.plane != "evidence"
                        or item.record_kind != "source_snippet"
                        or "evidence" not in spec.planes
                        or not spec.include_private_snippets
                        or not isinstance(excerpt, str) or len(excerpt) > 4_000
                        or len(excerpt.splitlines()) > 80
                        or item.data.get("authorization") != "project_bounded"
                        or not body_fields_safe
                        or not self._verified_source_snippet_item(
                            item, snapshot_id=snapshot_id, graph=view_graph,
                        )):
                    warnings.append(
                        f"retrieval_adapter_invalid_item:{adapter_fingerprint}"
                    )
                    continue
                source_snippets.append(item)
            collect_adapter_warnings(adapter, adapter_fingerprint)
        for snippet in source_snippets:
            position = next((index + 1 for index, item in enumerate(facts)
                             if item.record_kind == "entity"
                             and item.entity_id == snippet.entity_id), len(facts))
            facts.insert(position, snippet)
        facts = facts[:limit]
        retained_snippets = [item for item in facts if item.record_kind == "source_snippet"]
        if retained_snippets:
            private_returned = True
            channel_scores["source_snippet"] = {
                item.record_id: item.score for item in retained_snippets
            }
        if spec.include_private_snippets and not private_returned:
            warnings.append("private_snippets_not_authorized_or_unavailable")
        guidance, guidance_channels = self._rank_guidance(
            knowledge_documents, local_adapter_items, local_channel_scores,
            terms, query_folded, limit,
        )
        for name, values in guidance_channels.items():
            channel_scores[name] = values
        if any(item.data.get("binding_status") == "lexical_only" for item in guidance):
            warnings.append("knowledge_guidance_lexical_only_no_applicable_binding")
        if any(item.data.get("binding_status") == "applicable_revision_unbound"
               for item in guidance):
            warnings.append("knowledge_binding_artifact_revision_unbound")
        predictions = (self._rank_predictions(
            prediction_documents, adapter_prediction_items, terms, query_folded, limit,
        ) if "predictions" in spec.planes else [])
        if predictions:
            channel_scores["predictions_isolated"] = {
                item.record_id: item.score for item in predictions
            }
        flow = self._flow_spine(view_graph, ppr, selected_relation_ids, facts, max_edges=8)
        citations = self._citations(guidance)

        exact_entities = sorted(
            documents_by_key[key].entity_id for key in channel_scores["exact"]
            if (key in documents_by_key and documents_by_key[key].entity_id
                and documents_by_key[key].item.record_kind == "entity")
        )
        ambiguities: list[dict[str, Any]] = []
        if len(set(exact_entities)) > 1:
            candidates = sorted(set(exact_entities))
            ambiguities.append({
                "kind": "exact_entity_match", "candidate_ids": candidates,
                "resolved": False,
            })
            warnings.append("ambiguous_exact_entity_match")
        for item in facts:
            anchors = item.data.get("anchors", [])
            if not isinstance(anchors, list):
                continue
            ambiguous = [anchor for anchor in anchors if isinstance(anchor, Mapping)
                         and anchor.get("ambiguity")]
            if ambiguous:
                ambiguities.append({
                    "kind": "evidence_mapping", "record_id": item.record_id,
                    "ambiguous_anchor_count": len(ambiguous), "resolved": False,
                })
        stale = self.bundle.is_stale(self.bundle.store.snapshot(snapshot_id))
        incomplete = stale or not facts or any(item.completeness != "complete" for item in facts)
        if stale:
            warnings.append("snapshot_stale")
        if not facts:
            warnings.append("no_matching_fact_or_evidence")
        strongest = ordered_items[0] if ordered_items else None
        confidence = "low"
        if strongest is not None:
            channels = set(strongest.score_channels)
            if "exact" in channels or len(channels & {"prefix", "fts", "bm25", "graph"}) >= 2:
                confidence = "high"
        if incomplete:
            confidence = "low" if confidence != "high" or not facts else "incomplete"

        budget = spec.max_chars or (13_000 if len(view_graph.entities) < 150
                                    else 18_000 if len(view_graph.entities) < 500
                                    else 24_000)
        profile_payload = {
            "profile_schema_version": RETRIEVAL_PROFILE_SCHEMA_VERSION,
            "profile": spec.profile,
            "restart": 0.25,
            "iterations": 25,
            "depth": 3,
            "max_nodes": 200,
            "rrf_k": 60,
            "plane_channel_partition": {
                "facts": ["facts", "evidence"],
                "knowledge": ["knowledge"],
                "local": ["local"],
                "predictions": ["predictions"],
            },
            "relation_weights": _RELATION_WEIGHTS,
            "zero_weight_relations": sorted(_ZERO_WEIGHT_RELATIONS),
            "channel_weights": _CHANNEL_WEIGHTS,
        }
        trace = RetrievalTrace(
            query_sha256=hashlib.sha256(spec.query.encode("utf-8")).hexdigest(),
            snapshot_id=snapshot_id,
            profile=spec.profile,
            profile_hash=stable_hash(profile_payload),
            graph_hash=graph.graph_hash,
            profile_schema_version=RETRIEVAL_PROFILE_SCHEMA_VERSION,
            candidate_counts={name: len(values) for name, values in sorted(channel_scores.items())},
            elapsed_ms={
                "collect": round(collected_ms, 3),
                "lexical": round(max(0.0, lexical_ms), 3),
                "graph": round(max(0.0, graph_ms), 3),
                "total": round((perf_counter() - started) * 1000.0, 3),
            },
            output_budget_chars=budget,
            private_snippets_requested=spec.include_private_snippets,
            private_snippets_returned=private_returned,
            adapter_fingerprints=sorted(set(adapter_fingerprints)),
            semantic_channel=("indexed_available_but_disabled"
                              if semantic_index_available else "optional_not_enabled"),
        )
        result = RetrievalResult(
            snapshot_id=snapshot_id, facts=facts, guidance=guidance,
            predictions=predictions, flow=flow, citations=citations,
            ambiguities=ambiguities,
            confidence=confidence, incomplete=incomplete, stale=stale,
            warnings=sorted(set(warnings)), trace=trace,
        )
        self._apply_budget(result, budget)
        return result

    def _verified_source_snippet_item(
        self, item: RetrievalItem, *, snapshot_id: str, graph: CanonicalGraph,
    ) -> bool:
        """Revalidate the only built-in adapter capability that emits evidence.

        The adapter output is accepted only as a projection of an existing
        canonical entity and artifact.  Its stable record ID, authority,
        provenance, source hash, anchor, and excerpt digest are all recomputed;
        a caller-defined adapter cannot opt into this path by copying fields.
        """
        if (item.plane != "evidence" or item.record_kind != "source_snippet"
                or item.authority_class != "static_fact" or item.stage != "source"
                or not isinstance(item.entity_id, str)):
            return False
        entity = graph.entities.get(item.entity_id)
        if entity is None:
            return False
        artifact_id = item.data.get("artifact_id")
        artifact_sha256 = item.data.get("artifact_sha256")
        excerpt = item.data.get("private_excerpt")
        excerpt_sha256 = item.data.get("excerpt_sha256")
        anchor = item.data.get("anchor")
        if (not isinstance(artifact_id, str)
                or not isinstance(artifact_sha256, str)
                or not re.fullmatch(r"[0-9a-f]{64}", artifact_sha256)
                or not isinstance(excerpt, str)
                or not isinstance(excerpt_sha256, str)
                or hashlib.sha256(excerpt.encode("utf-8")).hexdigest() != excerpt_sha256
                or not isinstance(anchor, Mapping)
                or set(anchor) - {"start_line", "end_line", "symbol"}
                or item.data.get("projection_provenance")
                != SourceSnippetRetrievalAdapter.adapter_id
                or item.data.get("canonical_adapter_capability")
                != SourceSnippetRetrievalAdapter.canonical_capability):
            return False
        start = anchor.get("start_line")
        end = anchor.get("end_line")
        if (not isinstance(start, int) or isinstance(start, bool) or start < 1
                or not isinstance(end, int) or isinstance(end, bool) or end < start):
            return False
        artifacts = {
            value.id: value for value in self.bundle.store.artifacts(snapshot_id)
        }
        artifact = artifacts.get(artifact_id)
        matching_anchors = [value for value in entity.anchors
                            if value.artifact_id == artifact_id
                            and value.start_line is not None]
        if (artifact is None or artifact.sha256 != artifact_sha256
                or set(item.evidence_ids) != {entity.id, artifact.id}
                or not matching_anchors):
            return False
        data, status = SourceSnippetRetrievalAdapter._verified_bytes(
            self.bundle.project_root, artifact,
        )
        if data is None or status != "verified":
            return False
        try:
            lines = data.decode("utf-8").splitlines()
        except UnicodeDecodeError:
            return False
        anchor_matches = False
        for source_anchor in matching_anchors:
            original_start = int(source_anchor.start_line or 1)
            original_end = max(original_start, int(
                source_anchor.end_line or original_start,
            ))
            expected_start = max(1, original_start - 2)
            expected_end = min(len(lines), original_end + 2, expected_start + 79)
            if (start == expected_start and end == expected_end
                    and anchor.get("symbol") == source_anchor.symbol):
                anchor_matches = True
                break
        if not anchor_matches or excerpt != "\n".join(lines[start - 1:end])[:4_000]:
            return False
        expected_id = "source_snippet_" + stable_hash({
            "snapshot": snapshot_id,
            "entity": entity.id,
            "artifact": artifact.id,
            "start": start,
            "end": end,
            "excerpt": excerpt_sha256,
        })[:24]
        return item.record_id == expected_id

    def _rank_guidance(
        self, documents: Sequence[_Document],
        local_items: Mapping[str, RetrievalItem],
        local_channels: Mapping[str, Mapping[str, float]],
        terms: Sequence[str], query_folded: str, limit: int,
    ) -> tuple[list[RetrievalItem], dict[str, dict[str, float]]]:
        """Rank public knowledge and local text in isolated lexical corpora."""
        public_channels: dict[str, dict[str, float]] = defaultdict(dict)
        public_by_key = {item.key: item.item for item in documents}
        for document in documents:
            folded_fields = tuple(item.casefold() for item in document.fields if item)
            tokens = _document_tokens(document.text)
            if query_folded in folded_fields:
                public_channels["exact"][document.key] = 1.0
            coverage = sum(1 for term in terms if term in tokens)
            if (any(item.startswith(query_folded) or query_folded in item
                    for item in folded_fields) or (terms and coverage == len(terms))):
                public_channels["prefix"][document.key] = (
                    0.5 + 0.5 * coverage / max(1, len(terms))
                )
        public_channels["bm25"].update(_bm25(documents, terms))
        search_rules = getattr(self.bundle.store, "search_knowledge_rules", None)
        if callable(search_rules):
            probes = [" ".join(terms), *terms[:8]]
            seen: set[str] = set()
            for probe in probes:
                folded = probe.strip().casefold()
                if not folded or folded in seen:
                    continue
                seen.add(folded)
                for hit in search_rules(probe, limit=100):
                    rule = hit.get("rule")
                    key = f"knowledge:{getattr(rule, 'id', '')}"
                    if key not in public_by_key:
                        continue
                    raw_score = float(hit.get("score", 0.0))
                    score = max(0.000001, -raw_score if raw_score < 0 else raw_score)
                    public_channels["fts"][key] = max(
                        public_channels["fts"].get(key, 0.0), score,
                    )

        public_ranked, public_evidence = self._rrf(public_channels)
        public_maximum = max((score for _key, score in public_ranked), default=1.0)
        ranked_items: list[RetrievalItem] = []
        for key, score in public_ranked:
            item = public_by_key.get(key)
            if item is None:
                continue
            item.score = round(score / public_maximum, 8)
            item.score_channels = {
                f"knowledge_{name}": round(value, 8)
                for name, value in public_evidence[key].items()
            }
            ranked_items.append(item)

        local_ranked, local_evidence = self._rrf(local_channels)
        local_maximum = max((score for _key, score in local_ranked), default=1.0)
        for key, score in local_ranked:
            item = local_items.get(key)
            if item is None:
                continue
            item.score = round(score / local_maximum, 8)
            item.score_channels = {
                name: round(value, 8) for name, value in local_evidence[key].items()
            }
            ranked_items.append(item)

        ranked_items.sort(key=lambda item: (
            -item.score, 0 if item.plane == "knowledge" else 1, item.record_id,
        ))
        trace_channels = {
            **{f"knowledge_{name}": dict(values)
               for name, values in public_channels.items()},
            **{name: dict(values) for name, values in local_channels.items()},
        }
        return (ranked_items[:limit], trace_channels)

    @staticmethod
    def _rank_predictions(
        documents: Sequence[_Document], adapter_items: Mapping[str, RetrievalItem],
        terms: Sequence[str], query_folded: str, limit: int,
    ) -> list[RetrievalItem]:
        """Rank hypotheses without influencing any fact/guidance channel."""
        channels: dict[str, dict[str, float]] = defaultdict(dict)
        by_key = {item.key: item.item for item in documents}
        for document in documents:
            folded_fields = tuple(item.casefold() for item in document.fields if item)
            tokens = _document_tokens(document.text)
            if query_folded in folded_fields:
                channels["exact"][document.key] = 1.0
            coverage = sum(1 for term in terms if term in tokens)
            if (any(item.startswith(query_folded) or query_folded in item
                    for item in folded_fields) or (terms and coverage == len(terms))):
                channels["prefix"][document.key] = (
                    0.5 + 0.5 * coverage / max(1, len(terms))
                )
        channels["bm25"].update(_bm25(documents, terms))
        for key, item in adapter_items.items():
            by_key[key] = item
            channels["adapter"][key] = max(0.000001, item.score or 1.0)
        ranked, score_channels = HybridRetriever._rrf(channels)
        maximum = max((score for _key, score in ranked), default=1.0)
        result: list[RetrievalItem] = []
        for key, score in ranked:
            item = by_key.get(key)
            if item is None:
                continue
            item.score = round(score / maximum, 8)
            item.score_channels = {
                name: round(value, 8) for name, value in score_channels[key].items()
            }
            result.append(item)
            if len(result) >= limit:
                break
        return result

    def _documents(self, spec: RetrievalSpec, graph: CanonicalGraph,
                   allowed_ids: set[str], warnings: list[str]) -> list[_Document]:
        documents: list[_Document] = []
        entities = [graph.entities[item] for item in sorted(allowed_ids)]
        if "facts" in spec.planes:
            for entity in entities:
                fields = (entity.id, entity.name, entity.qualified_name or "", *entity.aliases)
                summary = f"{entity.kind} {entity.qualified_name or entity.name}"
                documents.append(_Document(
                    key=f"entity:{entity.id}",
                    text=" ".join((*fields, entity.kind, entity.stage, _text(entity.attrs))),
                    fields=fields,
                    entity_id=entity.id,
                    item=RetrievalItem(
                        record_id=entity.id, plane="facts", record_kind="entity",
                        title=entity.qualified_name or entity.name, summary=summary,
                        authority_class=str(entity.authority), stage=entity.stage,
                        completeness=str(entity.completeness), entity_id=entity.id,
                        evidence_ids=[anchor.artifact_id for anchor in entity.anchors],
                        data={
                            "kind": entity.kind,
                            "name": entity.name,
                            "qualified_name": entity.qualified_name,
                            "aliases": list(entity.aliases),
                            "attrs": json_ready(entity.attrs),
                            "anchors": [json_ready(anchor) for anchor in entity.anchors],
                        },
                    ),
                ))
            for relation in sorted(graph.relations.values(), key=lambda item: item.id):
                if relation.src not in allowed_ids or relation.dst not in allowed_ids:
                    continue
                source = graph.entities[relation.src]
                target = graph.entities[relation.dst]
                fields = (
                    relation.id, relation.kind,
                    source.name, source.qualified_name or "",
                    target.name, target.qualified_name or "",
                    relation.mapping_kind or "",
                )
                documents.append(_Document(
                    key=f"relation:{relation.id}",
                    text=" ".join((*fields, relation.stage, _text(relation.attrs))),
                    fields=fields, entity_id=relation.src,
                    item=RetrievalItem(
                        record_id=relation.id, plane="facts", record_kind="relation",
                        title=(f"{source.qualified_name or source.name} "
                               f"{relation.kind} {target.qualified_name or target.name}"),
                        summary=(f"Explicit {relation.kind} relation from {relation.src} "
                                 f"to {relation.dst}."),
                        authority_class=str(relation.authority), stage=relation.stage,
                        completeness=str(relation.completeness), entity_id=relation.src,
                        evidence_ids=[relation.id, *(
                            anchor.artifact_id for anchor in relation.anchors
                        )],
                        data={
                            "src": relation.src, "dst": relation.dst,
                            "kind": relation.kind,
                            "mapping_kind": relation.mapping_kind,
                            "attrs": json_ready(relation.attrs),
                            "anchors": [json_ready(anchor) for anchor in relation.anchors],
                        },
                    ),
                ))
        if "evidence" in spec.planes:
            evidence_subjects: dict[str, str] = {}
            observations = self.bundle.store.observations(self.snapshot_id)
            for observation in observations:
                if observation.subject_id not in allowed_ids:
                    continue
                evidence_subjects[observation.id] = observation.subject_id
                entity = graph.entities[observation.subject_id]
                value = _text(observation.value, 300)
                summary = f"{observation.predicate}={value}"
                if observation.unit:
                    summary += f" {observation.unit}"
                evidence_ids = [observation.id]
                evidence_ids.extend(item for item in (
                    observation.artifact_id, observation.run_id,
                    observation.anchor.artifact_id if observation.anchor else None,
                ) if item)
                fields = (observation.id, observation.predicate, entity.name,
                          entity.qualified_name or "")
                documents.append(_Document(
                    key=f"observation:{observation.id}",
                    text=" ".join((*fields, observation.stage, str(observation.authority),
                                   value, observation.unit or "")),
                    fields=fields, entity_id=observation.subject_id,
                    item=RetrievalItem(
                        record_id=observation.id, plane="evidence",
                        record_kind="observation", title=observation.predicate,
                        summary=summary, authority_class=str(observation.authority),
                        stage=observation.stage, completeness=str(observation.completeness),
                        entity_id=observation.subject_id, evidence_ids=evidence_ids,
                        data={
                            "subject_id": observation.subject_id,
                            "predicate": observation.predicate,
                            "value": json_ready(observation.value),
                            "unit": observation.unit,
                            "run_id": observation.run_id,
                            "artifact_id": observation.artifact_id,
                            "workload_id": observation.workload_id,
                        },
                    ),
                ))
            derivations = self.bundle.store.derivations(self.snapshot_id)
            for derivation in derivations:
                subject_id = str(derivation.get("subject_id", ""))
                if subject_id not in allowed_ids:
                    continue
                identifier = str(derivation.get("id", ""))
                predicate = str(derivation.get("predicate", "derivation.unknown"))
                evidence_subjects[identifier] = subject_id
                value = _text(derivation.get("value"), 300)
                fields = (
                    identifier, predicate,
                    str(derivation.get("algorithm", "")),
                    str(derivation.get("algorithm_version", "")),
                    graph.entities[subject_id].name,
                )
                evidence_refs = derivation.get("evidence_refs", [])
                evidence_ids = [identifier]
                if isinstance(evidence_refs, list):
                    evidence_ids.extend(
                        str(item.get("target_id")) for item in evidence_refs
                        if isinstance(item, Mapping) and item.get("target_id")
                    )
                documents.append(_Document(
                    key=f"derivation:{identifier}",
                    text=" ".join((*fields, str(derivation.get("stage", "unknown")), value)),
                    fields=fields, entity_id=subject_id,
                    item=RetrievalItem(
                        record_id=identifier, plane="evidence", record_kind="derivation",
                        title=predicate,
                        summary=(f"Deterministic derivation {predicate}={value} from "
                                 "explicit evidence references."),
                        authority_class=str(derivation.get("authority", "derived_fact")),
                        stage=str(derivation.get("stage", "unknown")),
                        completeness=str(derivation.get("completeness", "incomplete")),
                        entity_id=subject_id, evidence_ids=evidence_ids,
                        data={
                            "subject_id": subject_id, "predicate": predicate,
                            "value": json_ready(derivation.get("value")),
                            "unit": derivation.get("unit"),
                            "algorithm": derivation.get("algorithm"),
                            "algorithm_version": derivation.get("algorithm_version"),
                            "evidence_refs": json_ready(evidence_refs),
                        },
                    ),
                ))
            runs = {item.id: item for item in self.bundle.store.runs(self.snapshot_id)}
            for verification in self.bundle.store.verifications(self.snapshot_id):
                identifier = str(verification.get("id", ""))
                evidence_ids = [str(item) for item in verification.get("evidence_ids", [])]
                linked_subjects = sorted({
                    evidence_subjects[item] for item in evidence_ids if item in evidence_subjects
                })
                if spec.scope_id is not None and not linked_subjects:
                    continue
                subject_id = linked_subjects[0] if len(linked_subjects) == 1 else None
                kind = str(verification.get("kind", "verification.unknown"))
                status = str(verification.get("status", "unknown"))
                run_id = verification.get("run_id")
                run = runs.get(str(run_id)) if run_id else None
                synthetic_run = bool(run is not None and (
                    str(run.metadata.get("authority", "")).casefold()
                    in {"synthetic", "fake", "replay"}
                    or str(run.backend).casefold() in {"runner.fake", "runner.replay"}
                ))
                stage = run.stage if run is not None else {
                    "csim": "csim", "rtl_cosim": "rtl_cosim",
                }.get(kind, "unknown")
                fields = (identifier, kind, status,
                          str(verification.get("workload_id") or ""))
                documents.append(_Document(
                    key=f"verification:{identifier}", text=" ".join((*fields, stage)),
                    fields=fields, entity_id=subject_id,
                    item=RetrievalItem(
                        record_id=identifier, plane="evidence",
                        record_kind="verification_result", title=kind,
                        summary=f"Verification {kind} is {status}.",
                        authority_class=("synthetic" if synthetic_run
                                         else "verification_evidence"), stage=stage,
                        completeness=("complete" if status in {"pass", "fail"}
                                      else "incomplete"),
                        entity_id=subject_id,
                        evidence_ids=[identifier, *evidence_ids,
                                      *([str(run_id)] if run_id else [])],
                        data={
                            "kind": kind, "status": status, "run_id": run_id,
                            "workload_id": verification.get("workload_id"),
                            "details": json_ready(verification.get("details", {})),
                            "linked_subject_ids": linked_subjects,
                            "tool_truth": bool(
                                run is not None
                                and run.metadata.get("tool_truth") is True
                                and not synthetic_run
                                and self.bundle.store.has_valid_execution_commit(
                                    self.snapshot_id, run.id,
                                )
                            ),
                        },
                    ),
                ))
            for run in runs.values():
                synthetic_run = (
                    str(run.metadata.get("authority", "")).casefold()
                    in {"synthetic", "fake", "replay"}
                    or str(run.backend).casefold() in {"runner.fake", "runner.replay"}
                )
                for gate in run.gates:
                    evidence_ids = [str(item) for item in gate.evidence_ids]
                    linked_subjects = sorted({
                        evidence_subjects[item] for item in evidence_ids
                        if item in evidence_subjects
                    })
                    if spec.scope_id is not None and not linked_subjects:
                        continue
                    subject_id = linked_subjects[0] if len(linked_subjects) == 1 else None
                    gate_kind = str(gate.kind)
                    gate_status = str(gate.status)
                    identifier = "retrieval_gate_" + stable_hash({
                        "run": run.id, "kind": gate_kind,
                        "evidence": sorted(evidence_ids),
                    })[:24]
                    fields = (identifier, gate_kind, gate_status, run.id, run.stage)
                    documents.append(_Document(
                        key=f"gate:{identifier}", text=" ".join(fields), fields=fields,
                        entity_id=subject_id,
                        item=RetrievalItem(
                            record_id=identifier, plane="evidence",
                            record_kind="verification_gate", title=gate_kind,
                            summary=f"Verification gate {gate_kind} is {gate_status}.",
                            authority_class=("synthetic" if synthetic_run
                                             else "verification_evidence"),
                            stage=run.stage,
                            completeness=("complete" if gate_status in {"pass", "fail"}
                                          else "incomplete"),
                            entity_id=subject_id,
                            evidence_ids=[run.id, *evidence_ids],
                            data={
                                "kind": gate_kind, "status": gate_status,
                                "run_id": run.id,
                                # A gate reason is free-form local runner text
                                # and may contain commands, paths, or source.
                                # Other read-only projections redact it too.
                                "reason_redacted": gate.reason is not None,
                                "tool_truth": bool(
                                    run.metadata.get("tool_truth") is True
                                    and not synthetic_run
                                    and self.bundle.store.has_valid_execution_commit(
                                        self.snapshot_id, run.id,
                                    )
                                ),
                                "linked_subject_ids": linked_subjects,
                            },
                        ),
                    ))
            for diagnostic in self.bundle.store.active_diagnostics(self.snapshot_id):
                projected = public_diagnostic(diagnostic)
                subject_id = projected.get("subject_id")
                if subject_id is not None and subject_id not in allowed_ids:
                    continue
                identifier = str(projected.get("id") or diagnostic.id)
                code = str(projected.get("code") or "diagnostic.unknown")
                diagnostic_run = runs.get(str(projected.get("run_id"))) \
                    if projected.get("run_id") else None
                synthetic_run = bool(diagnostic_run is not None and (
                    str(diagnostic_run.metadata.get("authority", "")).casefold()
                    in {"synthetic", "fake", "replay"}
                    or str(diagnostic_run.backend).casefold()
                    in {"runner.fake", "runner.replay"}
                ))
                fields = (identifier, code, str(projected.get("severity") or ""))
                documents.append(_Document(
                    key=f"diagnostic:{identifier}", text=" ".join(fields), fields=fields,
                    entity_id=subject_id,
                    item=RetrievalItem(
                        record_id=identifier, plane="evidence", record_kind="diagnostic",
                        title=code,
                        summary="Diagnostic details are redacted; use its stable ID locally.",
                        authority_class=(
                            "synthetic" if synthetic_run else
                            "tool_observation" if projected.get("run_id") else "static_fact"
                        ),
                        stage=projected.get("stage"), completeness="incomplete",
                        entity_id=subject_id,
                        evidence_ids=[item for item in (
                            identifier, projected.get("artifact_id"), projected.get("run_id"),
                        ) if isinstance(item, str)],
                        data=projected,
                    ),
                ))
        if "knowledge" in spec.planes:
            context = self._applicability_context(spec)
            bindings_by_rule: dict[str, list[Any]] = defaultdict(list)
            try:
                binding_reader = getattr(
                    self.bundle.store, "knowledge_bindings", None,
                )
                rules = list(self.bundle.store.knowledge_rules())
                stored_bindings = (
                    list(binding_reader()) if callable(binding_reader) else []
                )
                reviewed_surface = self._review_ready_surface(
                    bindings=stored_bindings, rules=rules,
                )
            except Exception:
                rules = []
                stored_bindings = []
                reviewed_surface = _ReviewedKnowledgeSurface(rejected=True)
            if reviewed_surface.rejected:
                warnings.append("knowledge_activation_session_rejected")
                # Never touch malformed rows again after the review gate has
                # rejected their identity/content.  Facts and evidence remain
                # independently retrievable.
                rules = []
                stored_bindings = []
                reviewed_surface = _ReviewedKnowledgeSurface()
            reviewed_bindings: list[Any] = []
            for binding in stored_bindings:
                if binding.id not in reviewed_surface.binding_ids:
                    continue
                bindings_by_rule[binding.knowledge_rule_id].append(binding)
                reviewed_bindings.append(binding)
            reviewed_rules = [
                rule for rule in rules
                if rule.id in reviewed_surface.rule_ids
            ]
            target_contexts = self._binding_target_contexts(graph, allowed_ids)
            activation: BindingActivationSession | None = None
            pending_knowledge_documents: list[_Document] = []
            try:
                activation = BindingActivationSession(
                    snapshot_id=self.snapshot_id,
                    graph_hash=graph.graph_hash,
                    allowed_ids=sorted(allowed_ids),
                    bindings=reviewed_bindings,
                    rules=reviewed_rules,
                    expected_binding_fingerprints=(
                        reviewed_surface.binding_fingerprints
                    ),
                    expected_rule_fingerprints=reviewed_surface.rule_fingerprints,
                    raw_contexts=target_contexts,
                )
                for rule in reviewed_rules:
                    def evaluate_rule(detached_rule: Any) -> _RuleEvaluation:
                        applicable, applicability_reason = self._rule_applicable(
                            detached_rule.applicability, context,
                        )
                        return _RuleEvaluation(
                            rule=detached_rule,
                            applicable=applicable,
                            reason=applicability_reason,
                        )

                    rule_evaluation = activation.evaluate_rule_atomically(
                        rule, evaluate_rule,
                    )
                    if rule_evaluation is None:
                        continue
                    rule_snapshot = rule_evaluation.rule
                    bindings = bindings_by_rule.get(rule.id, [])
                    matching_bindings: list[Any] = []
                    revision_unbound_bindings: list[Any] = []
                    if bindings:
                        for binding in bindings:
                            target_key = (
                                str(binding.target_kind), str(binding.target),
                            )
                            for instance_context in target_contexts.get(target_key, ()):
                                if not self._context_matches_request(
                                    instance_context, spec,
                                ):
                                    continue
                                attested_context = activation.issue(
                                    binding, instance_context,
                                )
                                if attested_context is None:
                                    continue
                                evaluation = self._binding_evaluation(
                                    activation, binding, attested_context, spec,
                                )
                                if evaluation is None or not evaluation.request_matches:
                                    continue
                                binding_matches = evaluation.binding_matches
                                revision_unbound = (
                                    not binding_matches
                                    and evaluation.revision_unbound
                                )
                                if not binding_matches and not revision_unbound:
                                    continue
                                if evaluation.rule_applicable:
                                    if binding_matches:
                                        matching_bindings.append(evaluation.binding)
                                    else:
                                        revision_unbound_bindings.append(
                                            evaluation.binding
                                        )
                                    rule_snapshot = evaluation.rule
                                    break
                        if not matching_bindings and not revision_unbound_bindings:
                            continue
                        reason = "applicable"
                    else:
                        reason = rule_evaluation.reason
                        if not rule_evaluation.applicable:
                            continue
                    revision_unbound = bool(
                        revision_unbound_bindings and not matching_bindings
                    )
                    selected_bindings = (
                        matching_bindings if matching_bindings
                        else revision_unbound_bindings
                    )
                    binding_status = (
                        "applicable_revision_unbound" if revision_unbound
                        else "applicable" if matching_bindings else "lexical_only"
                    )
                    fields = (
                        rule_snapshot.id, rule_snapshot.rule_id,
                        rule_snapshot.title, rule_snapshot.section,
                        rule_snapshot.document_id, rule_snapshot.document_version,
                    )
                    pending_knowledge_documents.append(_Document(
                        key=f"knowledge:{rule_snapshot.id}",
                        text=" ".join((
                            *fields, rule_snapshot.summary or "",
                            _text(rule_snapshot.condition),
                            _text(rule_snapshot.effect),
                            _text(rule_snapshot.applicability),
                        )),
                        fields=fields,
                        item=RetrievalItem(
                            record_id=rule_snapshot.id, plane="knowledge",
                            record_kind="knowledge_rule",
                            title=rule_snapshot.title,
                            summary=rule_snapshot.summary or rule_snapshot.title,
                            authority_class="knowledge_rule",
                            stage=rule_snapshot.applicability.get("stage"),
                            completeness=(
                                "complete" if reason == "applicable"
                                and matching_bindings and not revision_unbound
                                else "incomplete"
                            ),
                            citation={
                                "document_id": rule_snapshot.document_id,
                                "document_version": rule_snapshot.document_version,
                                "section": rule_snapshot.section,
                                "url": rule_snapshot.citation_url,
                            },
                            data={
                                "rule_id": rule_snapshot.rule_id,
                                "applicability": json_ready(
                                    rule_snapshot.applicability
                                ),
                                "condition": json_ready(rule_snapshot.condition),
                                "effect": json_ready(rule_snapshot.effect),
                                "binding_status": binding_status,
                                "binding_ids": sorted(
                                    item.id for item in selected_bindings
                                ),
                                "applicability_status": reason,
                            },
                        ),
                    ))
                documents.extend(pending_knowledge_documents)
            except (TypeError, ValueError):
                # Corrupt, duplicated, or mutable activation input removes all
                # executable guidance; it never falls back to raw dictionaries.
                warnings.append("knowledge_activation_session_rejected")
            finally:
                if activation is not None:
                    activation.close()
        if "predictions" in spec.planes:
            for prediction in self.bundle.store.predictions(self.snapshot_id):
                subject_id = str(prediction.get("subject_id", ""))
                if subject_id not in allowed_ids:
                    continue
                identifier = str(prediction.get("id", ""))
                predicate = str(prediction.get("predicate", "prediction.unknown"))
                fields = (identifier, predicate, str(prediction.get("model_id", "")))
                documents.append(_Document(
                    key=f"prediction:{identifier}", text=" ".join((*fields, _text(prediction))),
                    fields=fields, entity_id=subject_id,
                    item=RetrievalItem(
                        record_id=identifier, plane="predictions",
                        record_kind="prediction_envelope", title=predicate,
                        summary=f"Prediction hypothesis for {subject_id}; not tool truth.",
                        authority_class="prediction_hypothesis",
                        completeness="complete", entity_id=subject_id,
                        data=json_ready(prediction),
                    ),
                ))
        return documents

    def _review_ready_surface(
        self, *, bindings: Sequence[Any] | None = None,
        rules: Sequence[Any] | None = None,
    ) -> _ReviewedKnowledgeSurface:
        """Fail closed when mutable or malformed stored knowledge cannot hash."""

        try:
            return self._review_ready_surface_unchecked(
                bindings=bindings, rules=rules,
            )
        except Exception:
            return _ReviewedKnowledgeSurface(rejected=True)

    def _review_ready_surface_unchecked(
        self, *, bindings: Sequence[Any] | None = None,
        rules: Sequence[Any] | None = None,
    ) -> _ReviewedKnowledgeSurface:
        """Return exact records closed to installed review-ready packs.

        This is a read-side security gate, not merely an install-time
        convenience.  Old ledgers or directly injected binding rows remain
        lexical-only unless the installed pack inventory and its immutable
        coverage manifest agree on the reviewed surface.
        """

        pack_reader = getattr(self.bundle.store, "installed_knowledge_packs", None)
        coverage_reader = getattr(self.bundle.store, "knowledge_coverage", None)
        binding_reader = getattr(self.bundle.store, "knowledge_bindings", None)
        rule_reader = getattr(self.bundle.store, "knowledge_rules", None)
        if not all(callable(item) for item in (
            pack_reader, coverage_reader, binding_reader, rule_reader,
        )):
            return _ReviewedKnowledgeSurface()
        installed = {
            str(item.get("pack_id")): item
            for item in pack_reader()
            if isinstance(item, Mapping) and item.get("pack_id")
        }
        stored_bindings = list(bindings) if bindings is not None else list(binding_reader())
        bindings_by_id = {item.id: item for item in stored_bindings}
        stored_rules = list(rules) if rules is not None else list(rule_reader())
        rules_by_id = {item.id: item for item in stored_rules}
        if (len(bindings_by_id) != len(stored_bindings)
                or len(rules_by_id) != len(stored_rules)):
            return _ReviewedKnowledgeSurface()
        surface = _ReviewedKnowledgeSurface()
        for coverage in coverage_reader():
            inventory = installed.get(str(coverage.pack_id))
            if (inventory is None
                    or not coverage.review_ready
                    or inventory.get("review_ready") is not True
                    or inventory.get("review_status") != coverage.review_status
                    or inventory.get("coverage_id") != coverage.id
                    or inventory.get("coverage_scope") != coverage.coverage_scope
                    or inventory.get("target_registry_version")
                    != coverage.target_registry_version):
                continue
            rule_values = inventory.get("rule_ids", [])
            binding_values = inventory.get("binding_ids", [])
            if (not isinstance(rule_values, list)
                    or not isinstance(binding_values, list)
                    or any(not isinstance(item, str) for item in rule_values)
                    or any(not isinstance(item, str) for item in binding_values)
                    or len(rule_values) != len(set(rule_values))
                    or len(binding_values) != len(set(binding_values))):
                continue
            declared_rules = set(rule_values)
            declared_bindings = set(binding_values)
            if (not declared_rules.issubset(rules_by_id)
                    or not declared_bindings.issubset(bindings_by_id)
                    or inventory.get("activation_hash")
                    != knowledge_activation_hash(
                        (rules_by_id[item] for item in declared_rules),
                        (bindings_by_id[item] for item in declared_bindings),
                        coverage,
                    )):
                continue
            rule_counts: Counter[str] = Counter()
            binding_counts: Counter[str] = Counter()
            binding_entries: dict[str, Any] = {}
            for entry in coverage.entries:
                if str(entry.status) != "rule":
                    continue
                rule_counts.update(entry.rule_ids)
                for binding_id in entry.binding_ids:
                    binding_counts[binding_id] += 1
                    binding_entries[binding_id] = entry
            if (rule_counts != Counter({item: 1 for item in declared_rules})
                    or binding_counts
                    != Counter({item: 1 for item in declared_bindings})):
                continue
            if any(
                binding_id not in bindings_by_id
                or bindings_by_id[binding_id].knowledge_rule_id
                not in binding_entries[binding_id].rule_ids
                for binding_id in declared_bindings
            ):
                continue
            try:
                expected_targets = canonical_supported_targets(
                    coverage.coverage_scope,
                    coverage.target_registry_version,
                )
            except KeyError:
                continue
            actual_targets = {
                (item.target_kind, item.target)
                for item in coverage.target_inventory
            }
            if actual_targets != expected_targets:
                continue
            target_counts: Counter[str] = Counter()
            target_mismatch = False
            for target in coverage.target_inventory:
                for binding_id in target.binding_ids:
                    binding = bindings_by_id.get(binding_id)
                    if binding is None or (
                        binding.target_kind, binding.target,
                    ) != (target.target_kind, target.target):
                        target_mismatch = True
                        break
                    target_counts[binding_id] += 1
                if target_mismatch:
                    break
            if (target_mismatch or target_counts
                    != Counter({item: 1 for item in declared_bindings})):
                continue
            # Close the IDs to the exact live bytes checked above.  The second
            # activation-hash computation detects a mutation while the rest of
            # the coverage contract was being verified; the session then
            # rechecks these per-record fingerprints at capture and use.
            rule_fingerprints = {
                item: stable_hash(json_ready(rules_by_id[item]))
                for item in declared_rules
            }
            binding_fingerprints = {
                item: stable_hash(json_ready(bindings_by_id[item]))
                for item in declared_bindings
            }
            if inventory.get("activation_hash") != knowledge_activation_hash(
                (rules_by_id[item] for item in declared_rules),
                (bindings_by_id[item] for item in declared_bindings),
                coverage,
            ):
                continue
            if any(
                item in surface.rule_fingerprints
                and surface.rule_fingerprints[item] != fingerprint
                for item, fingerprint in rule_fingerprints.items()
            ) or any(
                item in surface.binding_fingerprints
                and surface.binding_fingerprints[item] != fingerprint
                for item, fingerprint in binding_fingerprints.items()
            ):
                return _ReviewedKnowledgeSurface()
            surface.rule_ids.update(declared_rules)
            surface.binding_ids.update(declared_bindings)
            surface.rule_fingerprints.update(rule_fingerprints)
            surface.binding_fingerprints.update(binding_fingerprints)
        return surface

    def _review_ready_binding_ids(
        self, *, bindings: Sequence[Any] | None = None,
        rules: Sequence[Any] | None = None,
    ) -> set[str]:
        """Compatibility inspection returning only eligible binding IDs."""

        return set(self._review_ready_surface(
            bindings=bindings, rules=rules,
        ).binding_ids)

    @staticmethod
    def _context_copy(value: Mapping[str, set[str]]) -> dict[str, set[str]]:
        return {key: set(items) for key, items in value.items()}

    @staticmethod
    def _context_add(context: dict[str, set[str]], key: str, value: Any) -> None:
        if isinstance(value, bool):
            context.setdefault(key, set()).add(canonical_context_scalar(value))
        elif isinstance(value, str) and value:
            context.setdefault(key, set()).add(canonical_context_scalar(value))

    @classmethod
    def _context_metadata(
        cls, context: dict[str, set[str]], metadata: Mapping[str, Any],
        *, preserve_existing: Iterable[str] = (),
    ) -> None:
        # Reserved derived values are never copied from entity, relation, run,
        # artifact, observation, derivation, or diagnostic metadata.  This is
        # an explicit deny-list as well as an allow-list below so a future
        # metadata expansion cannot accidentally make Gate qualification
        # user-injectable.
        metadata = {
            key: value for key, value in metadata.items()
            if key not in _RESERVED_DERIVED_CONTEXT_KEYS
        }
        locked = {
            key: set(context[key]) for key in preserve_existing
            if context.get(key)
        }
        for key in (
            "protocol", "spec_version", "interface_mode", "ir", "dialect",
            "producer", "producer_version", "workload_id", "testcase_id",
            "activity_source", "waiver_mode", "vendor", "tool", "tool_version",
            "version",
        ):
            value = metadata.get(key)
            if isinstance(value, str):
                cls._context_add(context, key, value)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    cls._context_add(context, key, item)
        options = metadata.get("options")
        if isinstance(options, Mapping):
            cls._context_add(
                context, "interface_mode",
                options.get("mode") or options.get("interface_mode"),
            )
        operation = metadata.get("operation") or metadata.get("source_operation")
        if isinstance(operation, str) and "." in operation:
            cls._context_add(context, "dialect", operation.split(".", 1)[0])
        # Producer-bound identity beats self-described auxiliary metadata.
        # Metadata can fill a missing value for historical imports, but it
        # cannot add a conflicting tool/version/workload to an already pinned
        # run or artifact context.
        for key, values in locked.items():
            context[key] = values

    @classmethod
    def _context_directive_identity(
        cls, context: dict[str, set[str]], *, subject_id: str,
        metadata: Mapping[str, Any], graph: CanonicalGraph,
    ) -> None:
        """Copy identity only from the current directive record.

        Scope IDs are intentionally excluded from ``_context_metadata``.  They
        can qualify a rule only when the current directive carries a matching
        self ID.  This method copies identity but mints no source/operand
        capability: those stronger markers require independent fixed-parser
        replay in ``_context_directive_source_evidence`` or the DEPENDENCE
        counterpart below.
        """
        directive = graph.entities.get(subject_id)
        if directive is None or directive.kind != "hls.directive":
            return
        # Observation metadata may repeat extractor identities, but may not
        # replace the identities on the canonical directive instance.
        for key in (
            "directive_instance_id", "scope_id", "scope_kind",
            "scope_resolution", "function_id", "loop_id", "variable_id",
            "port_id",
        ):
            value = metadata.get(key)
            if value is not None and directive.attrs.get(key) != value:
                return
        instance_id = metadata.get("directive_instance_id")
        if not isinstance(instance_id, str) or instance_id != subject_id:
            return
        cls._context_add(context, "directive_instance_id", instance_id)
        scope_id = metadata.get("scope_id")
        scope_kind = metadata.get("scope_kind")
        resolution = metadata.get("scope_resolution")
        if not all(isinstance(item, str) and item
                   for item in (scope_id, scope_kind, resolution)):
            return
        scope_entity = graph.entities.get(scope_id)
        if scope_entity is None or scope_entity.kind != scope_kind:
            return
        cls._context_add(context, "scope_id", scope_id)
        cls._context_add(context, "scope_kind", scope_kind)
        cls._context_add(context, "scope_resolution", resolution)
        role_kinds = {
            "function_id": {"hls.kernel", "hls.function"},
            "loop_id": {"hls.loop"},
            "port_id": {"hls.port"},
        }
        for key, allowed_kinds in role_kinds.items():
            # A role is part of this precise scope only when it repeats the
            # same stable ID and its entity kind permits that role.
            if metadata.get(key) == scope_id and scope_kind in allowed_kinds:
                cls._context_add(context, key, scope_id)
        directive_kind = str(
            metadata.get("directive_kind")
            or directive.attrs.get("directive_kind")
            or directive.name
        ).upper()
        variable_id = metadata.get("variable_id")
        if directive_kind != "DEPENDENCE":
            if (variable_id == scope_id
                    and scope_kind in {"hls.memory", "hls.stream", "hls.port"}):
                cls._context_add(context, "variable_id", variable_id)
            return
        # DEPENDENCE's operand is distinct from the enclosing loop/function
        # scope.  Copy it only when both stable IDs exist in the graph and
        # close to the same unique function owner.
        if (scope_kind not in {"hls.kernel", "hls.function", "hls.loop"}
                or not isinstance(variable_id, str) or variable_id == scope_id):
            return
        variable = graph.entities.get(variable_id)
        if variable is None or variable.kind not in {
            "hls.memory", "hls.stream", "hls.port", "source.variable",
        }:
            return

        def function_owners(entity_id: str) -> set[str]:
            entity = graph.entities[entity_id]
            if entity.kind in {"hls.kernel", "hls.function"}:
                return {entity_id}
            frontier = {entity_id}
            visited = set(frontier)
            owners: set[str] = set()
            while frontier:
                parents = {
                    relation.src for relation in graph.relations.values()
                    if relation.kind == "hls.contains" and relation.dst in frontier
                    and relation.src in graph.entities
                } - visited
                visited.update(parents)
                found = {
                    item for item in parents
                    if graph.entities[item].kind in {"hls.kernel", "hls.function"}
                }
                owners.update(found)
                frontier = parents - found
            return owners

        scope_owners = function_owners(scope_id)
        if len(scope_owners) == 1 and function_owners(variable_id) == scope_owners:
            cls._context_add(context, "variable_id", variable_id)

    @classmethod
    def _context_projection_metadata(
        cls, context: dict[str, set[str]], metadata: Mapping[str, Any],
    ) -> None:
        """Copy non-attesting, instance-local IR/projection fields.

        Artifact revision, adapter identity, and language compatibility are
        intentionally excluded.  Only a pipeline-issued semantic attestation
        may add those capability-like context values.
        """
        for key in (
            "projection_mapping", "producer", "producer_version", "operation",
        ):
            value = metadata.get(key)
            if isinstance(value, str):
                cls._context_add(context, key, value)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    cls._context_add(context, key, item)
        source_operation = metadata.get("source_operation")
        if isinstance(source_operation, str):
            cls._context_add(context, "operation", source_operation)

    @classmethod
    def _context_entity_evidence(
        cls, context: dict[str, set[str]], entity: Any, *, current: bool,
    ) -> None:
        """Add condition evidence proved by one typed entity record.

        Open-IR rule conditions are deliberately not inferred from a query,
        symbol name, sibling record, or a snapshot-wide union.  These markers
        are emitted only from the current target entity or from an explicitly
        cited derivation evidence entity.
        """
        kind = str(entity.kind)
        if current:
            cls._context_add(context, "entity_instance_id", entity.id)
            cls._context_add(context, "entity_kind", kind)
        if kind in {"ir.mlir.module", "ir.mlir.function"}:
            cls._context_add(context, "mlir_container_present", True)
        elif kind == "ir.mlir.operation":
            cls._context_add(context, "mlir_operation_present", True)
        elif kind in {"ir.llvm.module", "ir.llvm.function"}:
            cls._context_add(context, "llvm_container_present", True)
        elif kind == "ir.llvm.block":
            cls._context_add(
                context, "basic_blocks_or_branches_present", "true",
            )
        elif kind == "ir.llvm.operation":
            cls._context_add(context, "llvm_instruction_present", True)

        attrs = entity.attrs if isinstance(entity.attrs, Mapping) else {}
        opcode = attrs.get("opcode")
        if isinstance(opcode, str):
            cls._context_add(context, "llvm_opcode", opcode)
        if kind == "ir.llvm.operation" and (
            attrs.get("memory_access") is True
            or isinstance(attrs.get("memory_access_kind"), str)
        ):
            cls._context_add(context, "memory_instruction_present", True)
        index_kinds = attrs.get("index_kinds")
        if (kind == "ir.llvm.operation" and isinstance(index_kinds, list)
                and any(isinstance(item, str) for item in index_kinds)):
            cls._context_add(
                context, "index_operand_instruction_present", True,
            )
        bitwidths = attrs.get("bitwidths")
        if (kind.startswith("ir.llvm.") and isinstance(bitwidths, list)
                and any(isinstance(item, int) and not isinstance(item, bool)
                        and item > 0 for item in bitwidths)):
            cls._context_add(context, "explicit_integer_width_present", True)
        if any(
            str(getattr(anchor, "mapping_kind", "")) == "llvm.debug"
            for anchor in entity.anchors
        ):
            cls._context_add(context, "debug_metadata_present", True)

    @staticmethod
    def _is_concrete_mlir_mapping_location(anchor: Any) -> bool:
        return (
            getattr(anchor, "mapping_kind", None)
            in _CONCRETE_MLIR_MAPPING_LOCATION_KINDS
            and isinstance(getattr(anchor, "ir_location", None), str)
            and anchor.ir_location.startswith("loc(")
            and isinstance(getattr(anchor, "start_line", None), int)
            and not isinstance(anchor.start_line, bool)
            and isinstance(getattr(anchor, "start_column", None), int)
            and not isinstance(anchor.start_column, bool)
            and getattr(anchor, "ambiguity", None) is None
        )

    @staticmethod
    def _mlir_location_source_candidates(
        graph: CanonicalGraph, location: Any,
    ) -> list[tuple[Any, Any, str]]:
        """Resolve one concrete Location to exact supported AST anchor pairs."""
        candidates: dict[tuple[str, str], tuple[Any, Any, str]] = {}
        if not HybridRetriever._is_concrete_mlir_mapping_location(location):
            return []
        for entity in graph.entities.values():
            if (entity.kind not in _SOURCE_MAPPING_TARGET_KINDS
                    or entity.stage != "ast"
                    or str(entity.completeness) != "complete"):
                continue
            for anchor in entity.anchors:
                if (anchor.artifact_id == location.artifact_id
                        and anchor.start_line is not None
                        and anchor.end_line is not None
                        and anchor.ambiguity is None
                        and anchor.start_line <= location.start_line
                        <= anchor.end_line):
                    anchor_identity = stable_hash(anchor)
                    candidates[(entity.id, anchor_identity)] = (
                        entity, anchor, anchor_identity,
                    )
        return [candidates[key] for key in sorted(candidates)]

    @classmethod
    def _context_relation_evidence(
        cls, context: dict[str, set[str]], relation: Any,
        graph: CanonicalGraph, *, current: bool,
    ) -> None:
        """Add relation-local endpoint, mapping, and condition provenance."""
        source = graph.entities.get(relation.src)
        target = graph.entities.get(relation.dst)
        if source is None or target is None:
            return
        if current:
            cls._context_add(context, "relation_instance_id", relation.id)
            cls._context_add(context, "relation_kind", relation.kind)
            cls._context_add(context, "source_entity_kind", source.kind)
            cls._context_add(context, "target_entity_kind", target.kind)
            cls._context_add(context, "source_entity_stage", source.stage)
            cls._context_add(context, "target_entity_stage", target.stage)
        if relation.mapping_kind:
            cls._context_add(context, "mapping_kind", relation.mapping_kind)
        for key in ("hardware_topology", "hardware_instance"):
            value = relation.attrs.get(key)
            if isinstance(value, bool):
                cls._context_add(context, key, value)

        if (relation.stage == "mlir" and relation.kind in {
                "ir.contains", "ir.ssa_use",
        }):
            cls._context_add(
                context, "mlir_containment_or_ssa_relation_present", True,
            )
        concrete_locations = [
            anchor for anchor in relation.anchors
            if cls._is_concrete_mlir_mapping_location(anchor)
        ]
        if (current
                and relation.kind == "cross.maps_to"
                and relation.stage == "mlir"
                and source.kind == "ir.mlir.operation"
                and str(source.completeness) == "complete"
                and target.kind in _SOURCE_MAPPING_TARGET_KINDS
                and target.stage == "ast"
                and str(target.completeness) == "complete"
                and str(relation.completeness) == "complete"
                and relation.attrs.get("hardware_topology") is False
                and relation.mapping_kind == "mlir.location"
                and len(concrete_locations) == 1):
            location = concrete_locations[0]
            candidates = cls._mlir_location_source_candidates(graph, location)
            target_anchor_hashes = {
                stable_hash(anchor) for anchor in relation.anchors
                if anchor is not location
            }
            location_identity = stable_hash(location)
            relation_anchor_hashes = {
                stable_hash(anchor) for anchor in relation.anchors
            }
            attrs = relation.attrs
            candidate_count = attrs.get("mapping_candidate_count")
            resolved = candidates[0] if len(candidates) == 1 else None
            resolution_valid = (
                resolved is not None
                and resolved[0].id == relation.dst
                and resolved[2] in target_anchor_hashes
                and location_identity != resolved[2]
                and len(relation.anchors) == 2
                and relation_anchor_hashes == {
                    location_identity, resolved[2],
                }
                and attrs.get("cardinality") == "many_to_many"
                and attrs.get("mapping_ambiguous") is False
                and isinstance(candidate_count, int)
                and not isinstance(candidate_count, bool)
                and candidate_count == 1
                and attrs.get("mapping_provenance")
                == "mlir.location_anchor"
                and attrs.get("mapping_redacted") is False
                and attrs.get("mapping_resolution") == "unique_exact"
                and attrs.get("mapping_resolution_contract")
                == _MLIR_LOCATION_RESOLUTION_CONTRACT
                and attrs.get("mapping_unresolved") is False
                and attrs.get("resolved_target_anchor_identity")
                == resolved[2]
                and attrs.get("resolved_target_id") == target.id
                and attrs.get("source_anchor_identity_contract")
                == _SOURCE_ANCHOR_IDENTITY_CONTRACT
                and attrs.get("target_layer") == "source_ast"
                and attrs.get("typed_source_anchor_identity")
                == location_identity
            )
            if not resolution_valid:
                return
            cls._context_add(context, "typed_mlir_location_present", True)
            cls._context_add(
                context, "mapping_provenance", "mlir.location_anchor",
            )
            cls._context_add(
                context, "location_kind", location.mapping_kind,
            )
            cls._context_add(context, "mapping_resolution", "unique_exact")
            cls._context_add(
                context, "mapping_resolution_contract",
                _MLIR_LOCATION_RESOLUTION_CONTRACT,
            )
            cls._context_add(
                context, "unique_mlir_location_mapping_resolved", True,
            )
            cls._context_add(
                context, "typed_source_anchor_identity", location_identity,
            )
            cls._context_add(
                context, "resolved_target_anchor_identity", resolved[2],
            )
            cls._context_add(context, "resolved_target_id", target.id)
            cls._context_add(
                context, "source_anchor_identity_contract",
                _SOURCE_ANCHOR_IDENTITY_CONTRACT,
            )

        if (relation.kind == "llvm.cfg"
                and source.kind == target.kind == "ir.llvm.block"):
            cls._context_add(
                context, "basic_blocks_or_branches_present", True,
            )
        if (relation.kind == "llvm.calls"
                and source.kind == "ir.llvm.operation"
                and target.kind == "ir.llvm.function"
                and str(source.attrs.get("opcode", "")) in {
                    "call", "invoke", "callbr",
                }):
            cls._context_add(
                context, "resolved_call_like_instruction_present", True,
            )
        if (current
                and relation.kind == "handshake.dataflow"
                and relation.stage == "mlir"
                and source.kind == target.kind == "ir.mlir.operation"
                and str(source.completeness) == "complete"
                and str(target.completeness) == "complete"
                and str(relation.completeness) == "complete"
                and all(str(item.attrs.get("operation", "")).startswith(
                    "handshake.") for item in (source, target))
                and relation.attrs.get("hardware_topology") is False
                and relation.attrs.get("native_ir_evidence") is True
                and relation.attrs.get("native_ir_evidence_contract")
                == _NATIVE_MLIR_SSA_EVIDENCE_CONTRACT
                and relation.attrs.get("native_ir_relation_provenance")
                == "mlir.ssa_def_use"
                and len(relation.anchors) == 1):
            anchor = relation.anchors[0]
            ssa_value = relation.attrs.get("ssa_value")
            target_operands = target.attrs.get("ssa_operands")
            artifact_id = relation.attrs.get("native_ir_artifact_id")
            native_evidence_valid = (
                isinstance(ssa_value, str)
                and ssa_value.startswith("%")
                and source.attrs.get("ssa_result") == ssa_value
                and isinstance(target_operands, list)
                and ssa_value in target_operands
                and all(isinstance(item, str) and item.startswith("%")
                        for item in target_operands)
                and isinstance(artifact_id, str)
                and artifact_id == anchor.artifact_id
                and anchor.mapping_kind is None
                and anchor.ir_location is None
                and anchor.ambiguity is None
                and any(item.artifact_id == artifact_id
                        for item in source.anchors)
                and any(item.artifact_id == artifact_id
                        for item in target.anchors)
            )
            if not native_evidence_valid:
                return
            cls._context_add(context, "handshake_operation_present", True)
            cls._context_add(context, "native_ir_evidence", True)
            cls._context_add(
                context, "native_ir_evidence_contract",
                _NATIVE_MLIR_SSA_EVIDENCE_CONTRACT,
            )
            cls._context_add(
                context, "native_ir_relation_provenance", "mlir.ssa_def_use",
            )
            cls._context_add(
                context, "native_ir_artifact_identity", artifact_id,
            )

    def _context_derivation_evidence(
        self, context: dict[str, set[str]], derivation: Mapping[str, Any],
        graph: CanonicalGraph, artifacts: Mapping[str, Any],
    ) -> None:
        """Qualify a derivation from its own typed evidence closure only."""
        identifier = derivation.get("id")
        if isinstance(identifier, str):
            self._context_add(context, "derivation_instance_id", identifier)
        self._context_add(
            context, "derivation_algorithm", derivation.get("algorithm"),
        )
        self._context_add(
            context, "derivation_algorithm_version",
            derivation.get("algorithm_version"),
        )
        evidence = derivation.get("evidence_refs", [])
        if not isinstance(evidence, list):
            return
        artifact_ids: set[str] = set()
        evidence_entities: dict[str, Any] = {}
        evidence_relations: dict[str, Any] = {}
        for reference in evidence:
            if not isinstance(reference, Mapping):
                continue
            if reference.get("snapshot_id") not in {None, graph.snapshot_id}:
                continue
            kind = str(reference.get("kind") or "")
            target_id = str(reference.get("target_id") or "")
            if kind == "entity_anchor" and target_id in graph.entities:
                evidence_entities[target_id] = graph.entities[target_id]
                self._context_entity_evidence(
                    context, graph.entities[target_id], current=False,
                )
            elif kind == "relation" and target_id in graph.relations:
                evidence_relations[target_id] = graph.relations[target_id]
                self._context_relation_evidence(
                    context, graph.relations[target_id], graph, current=False,
                )
            elif kind == "artifact" and target_id in artifacts:
                artifact_ids.add(target_id)
        semantic_artifact_id = self._context_semantic_artifact_evidence(
            context, graph, artifact_ids, artifacts,
        )

        predicate = str(derivation.get("predicate") or "")
        if (predicate not in {
                "feature.operation_histogram", "feature.index_histogram",
                "feature.bitwidth", "feature.memory_access",
        } or derivation.get("algorithm") != (
            f"hlsgraph.static.{predicate.removeprefix('feature.')}"
        ) or str(derivation.get("algorithm_version")) != "1"
                or str(derivation.get("completeness")) != "complete"
                or derivation.get("subject_id") not in evidence_entities):
            return
        operations = [
            item for item in evidence_entities.values()
            if item.kind in {"ir.mlir.operation", "ir.llvm.operation"}
        ]
        if not operations:
            return
        if (semantic_artifact_id is None or any(
            not any(anchor.artifact_id == semantic_artifact_id
                    for anchor in item.anchors)
            for item in evidence_entities.values()
            if item.kind in {"ir.mlir.operation", "ir.llvm.operation"}
        )):
            return
        # Recompute the complete operation domain from the qualified graph,
        # rather than accepting a derivation-selected subset of evidence.
        from .extract.static_features import (
            _CONTAINS_KINDS, _EXPLICIT_MAPPING_KINDS, _closure,
            _scope_operations,
        )
        children: dict[str, list[tuple[Any, str]]] = defaultdict(list)
        for relation in sorted(graph.relations.values(), key=lambda item: item.id):
            if relation.kind in _CONTAINS_KINDS:
                children[relation.src].append((relation, relation.dst))
        subject_id = str(derivation.get("subject_id"))
        closure_ids, parent = _closure(subject_id, graph.entities, children)
        expected_operations, expected_entities, expected_relations = _scope_operations(
            subject_id, closure_ids, parent, graph.entities,
            sorted(
                (item for item in graph.relations.values()
                 if item.kind in _EXPLICIT_MAPPING_KINDS),
                key=lambda item: item.id,
            ),
        )
        if ({item.id for item in expected_operations}
                != {item.id for item in operations}
                or not {item.id for item in expected_entities}.issubset(
                    evidence_entities
                )
                or not {item.id for item in expected_relations}.issubset(
                    evidence_relations
                )):
            return
        layers = {
            "mlir" if item.kind == "ir.mlir.operation" else "llvm"
            for item in operations
        }
        if len(layers) != 1:
            return
        layer = next(iter(layers))
        if derivation.get("stage") != layer:
            return
        metadata = derivation.get("metadata", {})
        if not isinstance(metadata, Mapping):
            return

        if predicate != "feature.operation_histogram":
            if (layer != "llvm"
                    or any(not item.kind.startswith("ir.llvm.")
                           for item in evidence_entities.values())):
                return
            opcodes: list[str] = []
            for operation in operations:
                opcode = operation.attrs.get("opcode")
                if not isinstance(opcode, str) or not opcode:
                    return
                opcodes.append(opcode)

            if predicate == "feature.index_histogram":
                values: list[str] = []
                for operation, opcode in zip(operations, opcodes, strict=True):
                    raw = operation.attrs.get("index_kinds")
                    if raw is None:
                        raw = []
                    if (not isinstance(raw, list)
                            or any(item not in {"constant", "dynamic"}
                                   for item in raw)
                            or (opcode in _LLVM_INDEX_OPCODES) != bool(raw)):
                        return
                    values.extend(raw)
                expected = dict(sorted(Counter(values).items()))
                contract = {
                    "index_histogram_schema": (
                        "llvm.explicit_index_operand_kind_histogram.v1"
                    ),
                    "index_histogram_provenance": (
                        "typed_ir_entity_evidence.v1"
                    ),
                    "index_operand_definition": (
                        "llvm.gep_extract_insert_explicit_operand.v1"
                    ),
                    "index_histogram_domain_complete": True,
                }
                condition = "typed_index_histogram_present"
            elif predicate == "feature.bitwidth":
                widths: list[int] = []
                for entity in evidence_entities.values():
                    raw = entity.attrs.get("bitwidths")
                    if raw is not None and (
                        not isinstance(raw, list) or any(
                            not isinstance(item, int) or isinstance(item, bool)
                            or not 0 < item <= 1_048_576 for item in raw
                        )
                    ):
                        return
                    if isinstance(raw, list):
                        widths.extend(raw)
                    for key in ("type", "element_type", "return_type"):
                        value = entity.attrs.get(key)
                        if value is None:
                            continue
                        if not isinstance(value, str):
                            return
                        for match in _EXPLICIT_IR_WIDTH.finditer(value):
                            width = next(
                                (int(item) for item in match.groups() if item),
                                0,
                            )
                            if 0 < width <= 1_048_576:
                                widths.append(width)
                expected = dict(sorted(Counter(
                    str(width) for width in widths
                ).items()))
                contract = {
                    "bitwidth_schema": (
                        "llvm.explicit_integer_width_occurrence_histogram.v1"
                    ),
                    "bitwidth_provenance": "typed_ir_entity_evidence.v1",
                    "bitwidth_definition": (
                        "llvm.explicit_integer_type_occurrence.v1"
                    ),
                    "bitwidth_domain_complete": True,
                }
                condition = "typed_bitwidth_histogram_present"
            else:
                values = []
                for operation, opcode in zip(operations, opcodes, strict=True):
                    expected_kind = _LLVM_MEMORY_ACCESS_KINDS.get(opcode)
                    if operation.attrs.get("memory_access_kind") != expected_kind:
                        return
                    if expected_kind is not None:
                        values.append(expected_kind)
                expected = dict(sorted(Counter(values).items()))
                contract = {
                    "memory_access_schema": (
                        "llvm.memory_access_kind_histogram.v1"
                    ),
                    "memory_access_provenance": (
                        "typed_ir_entity_evidence.v1"
                    ),
                    "memory_access_opcode_definition": (
                        "llvm.load_store_gep_atomic_fence.v1"
                    ),
                    "memory_access_domain_complete": True,
                }
                condition = "typed_memory_access_histogram_present"
            if derivation.get("value") != expected or any(
                metadata.get(key) != value for key, value in contract.items()
            ):
                return
            for key, value in contract.items():
                self._context_add(context, key, value)
            self._context_add(context, condition, True)
            self._context_recomputed_aggregate_evidence(
                context, derivation=derivation, graph=graph,
                semantic_artifact_id=semantic_artifact_id,
                operation_ids=[item.id for item in operations],
                aggregate_contract=contract,
            )
            return

        names: list[str] = []
        for operation in operations:
            if layer == "mlir":
                name = operation.attrs.get("operation")
                dialect = operation.attrs.get("dialect")
                if (not isinstance(name, str) or not isinstance(dialect, str)
                        or "." not in name
                        or name.split(".", 1)[0].casefold() != dialect.casefold()):
                    return
            else:
                name = operation.attrs.get("opcode")
                if not isinstance(name, str) or not name:
                    return
            names.append(name)
        expected = dict(sorted(Counter(names).items()))
        if derivation.get("value") != expected:
            return
        expected_schema = (
            "mlir.dialect_qualified_opcode_histogram.v1"
            if layer == "mlir" else "llvm.opcode_histogram.v1"
        )
        if (metadata.get("operation_histogram_schema") != expected_schema
                or metadata.get("operation_histogram_provenance")
                != "typed_ir_entity_evidence.v1"
                or metadata.get("operation_histogram_domain_complete") is not True):
            return
        self._context_add(
            context, "operation_histogram_schema", expected_schema,
        )
        self._context_add(
            context, "operation_histogram_provenance",
            "typed_ir_entity_evidence.v1",
        )
        self._context_add(
            context, "operation_histogram_domain_complete", True,
        )
        if layer == "mlir":
            self._context_add(
                context, "dialect_qualified_operation_histogram_present", True,
            )
        else:
            self._context_add(
                context, "opcode_qualified_operation_histogram_present", True,
            )
        self._context_recomputed_aggregate_evidence(
            context, derivation=derivation, graph=graph,
            semantic_artifact_id=semantic_artifact_id,
            operation_ids=[item.id for item in operations],
            aggregate_contract={
                "operation_histogram_schema": expected_schema,
                "operation_histogram_provenance": (
                    "typed_ir_entity_evidence.v1"
                ),
                "operation_histogram_domain_complete": True,
            },
        )

    @classmethod
    def _context_recomputed_aggregate_evidence(
        cls, context: dict[str, set[str]], *, derivation: Mapping[str, Any],
        graph: CanonicalGraph, semantic_artifact_id: str,
        operation_ids: Iterable[str], aggregate_contract: Mapping[str, Any],
    ) -> None:
        """Mint a separate origin for a fully recomputed current aggregate.

        The language-spec attestation proves which live IR bytes and adapter
        contract are being interpreted.  It does not prove an aggregate.  A
        successful recomputation therefore receives a distinct deterministic
        origin bound to the derivation, exact operation domain, graph, bytes,
        and semantic attestation.
        """
        attestation_ids = context.get("semantic_attestation_identity", set())
        artifact_ids = context.get("artifact_identity", set())
        artifact_hashes = context.get("artifact_sha256", set())
        if (attestation_ids == set() or len(attestation_ids) != 1
                or artifact_ids != {semantic_artifact_id}
                or len(artifact_hashes) != 1):
            return
        attestation_id = next(iter(attestation_ids))
        aggregate_id = stable_id("aggregate_evidence", {
            "snapshot_id": graph.snapshot_id,
            "graph_hash": graph.graph_hash,
            "derivation_id": derivation.get("id"),
            "subject_id": derivation.get("subject_id"),
            "predicate": derivation.get("predicate"),
            "algorithm": derivation.get("algorithm"),
            "algorithm_version": derivation.get("algorithm_version"),
            "semantic_attestation_id": attestation_id,
            "artifact_id": semantic_artifact_id,
            "artifact_sha256": next(iter(artifact_hashes)),
            "operation_ids": sorted(set(operation_ids)),
            "value_hash": stable_hash(derivation.get("value")),
            "aggregate_contract": dict(aggregate_contract),
            "contract": _AGGREGATE_EVIDENCE_CONTRACT,
        })
        context["evidence_origin_identity"] = {aggregate_id}
        cls._context_add(
            context, _AGGREGATE_EVIDENCE_CONTEXT_KEY,
            _AGGREGATE_EVIDENCE_CONTEXT_VALUE,
        )
        cls._context_add(
            context, "aggregate_evidence_contract",
            _AGGREGATE_EVIDENCE_CONTRACT,
        )
        cls._context_add(context, "aggregate_evidence_identity", aggregate_id)
        cls._context_add(
            context, "aggregate_semantic_attestation_identity", attestation_id,
        )
        cls._context_add(
            context, "aggregate_source_artifact_identity", semantic_artifact_id,
        )

    def _qualified_semantic_attestation(
        self, graph: CanonicalGraph, artifact_ids: set[str],
        artifacts: Mapping[str, Any],
    ) -> None:
        """Reject graph-carried semantic claims until a trusted contract exists.

        ``CanonicalGraph.metadata`` is serialized project data and callers can
        construct it.  A well-formed ``ArtifactSemanticAttestation`` object in
        that dictionary is therefore not a capability or persisted trust root.
        v0.3 has no authorization ledger for language-spec adapters, so OpenIR
        executable guidance remains fail-closed even when the artifact bytes,
        extractor names, and claimed revisions appear self-consistent.
        """

        return None

    def _context_semantic_artifact_evidence(
        self, context: dict[str, set[str]], graph: CanonicalGraph,
        artifact_ids: set[str], artifacts: Mapping[str, Any],
    ) -> str | None:
        qualified = self._qualified_semantic_attestation(
            graph, artifact_ids, artifacts,
        )
        if qualified is None:
            return None
        artifact, attestation = qualified
        self._context_add(
            context, _SEMANTIC_ARTIFACT_EVIDENCE_CONTEXT_KEY,
            _SEMANTIC_ARTIFACT_EVIDENCE_CONTEXT_VALUE,
        )
        self._context_add(
            context, "semantic_attestation_contract",
            _SEMANTIC_ATTESTATION_CONTRACT,
        )
        self._context_add(context, "semantic_attestation_identity", attestation.id)
        self._context_add(context, "evidence_origin_identity", attestation.id)
        self._context_add(context, "artifact_identity", artifact.id)
        self._context_add(context, "semantic_artifact_kind", artifact.kind)
        self._context_add(context, "artifact_sha256", artifact.sha256)
        self._context_add(context, "artifact_revision", attestation.artifact_revision)
        self._context_add(
            context, "artifact_byte_closure", _SEMANTIC_ARTIFACT_BYTE_CLOSURE,
        )
        self._context_add(context, "extractor_name", attestation.extractor_name)
        self._context_add(context, "extractor_version", attestation.extractor_version)
        self._context_add(context, "extractor_identity", attestation.extractor_identity)
        self._context_add(context, "adapter_contract", attestation.adapter_contract)
        self._context_add(context, "adapter_version", attestation.adapter_version)
        self._context_add(
            context, "extraction_manifest_identity", stable_hash({
                "snapshot_id": self.snapshot_id,
                "extraction_hash": attestation.extraction_hash,
                "extractor_identity": attestation.extractor_identity,
                "artifact_id": artifact.id,
                "artifact_sha256": artifact.sha256,
            }),
        )
        for item in attestation.language_spec_contracts:
            self._context_add(context, "language_spec_family", item.family)
            self._context_add(context, "language_spec_revision", item.revision)
            self._context_add(
                context, "language_spec_compatibility_contract",
                item.compatibility_contract,
            )
        self._context_add(
            context, "language_spec_revision_source",
            _LANGUAGE_SPEC_REVISION_SOURCE,
        )
        return artifact.id

    def _context_unique_anchor_artifact(
        self, context: dict[str, set[str]], entity_ids: Iterable[str],
        graph: CanonicalGraph, artifacts: Mapping[str, Any],
        relation_anchors: Iterable[Any] = (),
    ) -> None:
        """Qualify a target only through one attested live IR artifact."""
        artifact_ids = {
            str(anchor.artifact_id)
            for entity_id in entity_ids
            for anchor in graph.entities[entity_id].anchors
            if anchor.artifact_id in artifacts
        }
        artifact_ids.update(
            str(anchor.artifact_id) for anchor in relation_anchors
            if anchor.artifact_id in artifacts
        )
        self._context_semantic_artifact_evidence(
            context, graph, artifact_ids, artifacts,
        )

    @classmethod
    def _context_ir_stage(cls, context: dict[str, set[str]], stage: Any) -> None:
        if stage == "mlir":
            cls._context_add(context, "ir", "mlir")
        elif stage == "llvm":
            cls._context_add(context, "ir", "llvm")

    def _manifest_context(self) -> tuple[dict[str, set[str]], dict[str, Any]]:
        manifest = self.bundle.store.snapshot_manifest(self.snapshot_id)
        context: dict[str, set[str]] = {}
        self._context_add(context, "vendor", manifest.target.vendor)
        # Tool and version are one inseparable identity.  A manifest containing
        # Vitis HLS 2023.2 plus Vivado 2024.2 must not synthesize the nonexistent
        # pair "Vitis HLS 2024.2" by independently pooling both columns.
        if len(manifest.toolchains) == 1:
            toolchain = manifest.toolchains[0]
            self._context_add(context, "tool", toolchain.name)
            self._context_add(context, "tool", toolchain.id)
            self._context_add(context, "tool_version", toolchain.version)
            self._context_add(context, "version", toolchain.version)
        return context, {item.id: item for item in manifest.toolchains}

    def _source_tool_context(
        self, base: Mapping[str, set[str]], toolchains: Mapping[str, Any],
    ) -> dict[str, set[str]]:
        """Resolve exactly one Vitis HLS identity for source semantics."""
        context = {
            key: set(values) for key, values in base.items()
            if key not in {"tool", "tool_version", "version"}
        }
        candidates = [
            item for item in toolchains.values()
            if str(item.name).casefold() == "vitis_hls"
            and str(item.vendor).casefold()
            in context.get("vendor", {str(item.vendor).casefold()})
        ]
        if len(candidates) != 1:
            return context
        toolchain = candidates[0]
        self._context_add(context, "tool", toolchain.name)
        self._context_add(context, "tool", toolchain.id)
        self._context_add(context, "tool_version", toolchain.version)
        self._context_add(context, "version", toolchain.version)
        return context

    def _run_context(
        self, run: Any, base: Mapping[str, set[str]], toolchains: Mapping[str, Any],
    ) -> dict[str, set[str]]:
        # Producer-bound records use the producer's tool identity, not the
        # union of unrelated tools declared elsewhere in the snapshot.
        context = {
            key: set(values) for key, values in base.items()
            if key not in {"tool", "tool_version", "version", "stage",
                           "workload_id", "testcase_id", "activity_source"}
        }
        self._context_add(context, "stage", run.stage)
        if run.toolchain_id:
            self._context_add(context, "tool", run.toolchain_id)
            toolchain = toolchains.get(run.toolchain_id)
            if toolchain is not None:
                self._context_add(context, "tool", toolchain.name)
                self._context_add(context, "tool_version", toolchain.version)
                self._context_add(context, "version", toolchain.version)
        for key in ("workload_id", "testcase_id", "activity_source"):
            self._context_add(context, key, run.metadata.get(key))
        self._context_metadata(
            context, run.metadata,
            preserve_existing=(
                "vendor", "tool", "tool_version", "version", "stage",
                "workload_id", "testcase_id", "activity_source",
            ),
        )
        return context

    def _qualified_tool_artifact_context(
        self, *, artifact: Any, run: Any, artifacts: Mapping[str, Any],
        manifest: Any, vendor_only: Mapping[str, set[str]],
        toolchains: Mapping[str, Any],
    ) -> dict[str, set[str]] | None:
        """Close the current report artifact to its declared live run output."""
        stage_policy = _TOOL_ARTIFACT_STAGE_POLICY.get(str(artifact.kind))
        canonical_stage = (
            stage_policy.get(str(run.stage)) if stage_policy is not None else None
        )
        expected_tool = (
            "vitis_hls" if str(artifact.kind).startswith("amd.vitis.")
            else "vivado" if str(artifact.kind).startswith("amd.vivado.")
            else None
        )
        toolchain = toolchains.get(run.toolchain_id)
        if (canonical_stage is None or expected_tool is None
                or toolchain is None
                or str(toolchain.name).casefold() != expected_tool
                or run.snapshot_id != self.snapshot_id
                or artifact.producer_run_id != run.id
                or successful_fresh_tool_run_error(run) is not None
                or tool_run_manifest_identity_error(run, manifest) is not None
                or not self.bundle.store.has_valid_execution_commit(
                    self.snapshot_id, run.id,
                )):
            return None
        base_ids = {
            item.id for item in artifacts.values()
            if item.producer_run_id is None
        }
        if not base_ids.issubset(set(run.input_artifact_ids)):
            return None
        output_path = artifact.metadata.get("declared_output_path")
        declared = {
            item.path: item for item in manifest.stage_outputs.get(run.stage, [])
        }
        spec = declared.get(output_path) if isinstance(output_path, str) else None
        declared_stage = artifact.metadata.get("stage")
        if (spec is None or spec.kind != artifact.kind
                or spec.role != artifact.role
                or str(spec.access) != str(artifact.access)
                or spec.license != artifact.license
                or artifact.id not in run.output_artifact_ids
                or (declared_stage is not None
                    and declared_stage != canonical_stage)
                or not self._managed_artifact_bytes_valid(artifact)):
            return None

        context = self._run_context(run, vendor_only, toolchains)
        context["stage"] = {canonical_stage.casefold()}
        self._context_add(context, "snapshot_id", self.snapshot_id)
        self._context_add(context, "snapshot_association", "verified")
        self._context_add(
            context, _TOOL_ARTIFACT_CONTEXT_KEY,
            _TOOL_ARTIFACT_CONTEXT_VALUE,
        )
        self._context_add(context, "tool_artifact_identity", stable_hash({
            "artifact_id": artifact.id,
            "kind": artifact.kind,
            "sha256": artifact.sha256,
            "size": artifact.size,
            "declared_output_path": output_path,
        }))
        self._context_add(context, "tool_artifact_run_identity", stable_hash({
            "run_id": run.id,
            "request_hash": run.request_hash,
            "stage": run.stage,
            "toolchain_id": run.toolchain_id,
            "environment_hash": run.environment_hash,
        }))
        return context

    def _artifact_context(
        self, artifact: Any, base: Mapping[str, set[str]], toolchains: Mapping[str, Any],
    ) -> dict[str, set[str]]:
        context = self._context_copy(base)
        expected_tool = (
            "vitis_hls" if artifact.kind.startswith("amd.vitis.")
            else "vivado" if artifact.kind.startswith("amd.vivado.")
            else "vivado" if artifact.kind == "constraint.xdc"
            else None
        )
        if expected_tool is not None and artifact.kind != "constraint.xdc":
            matches = [item for item in toolchains.values()
                       if item.name.casefold() == expected_tool]
            if len(matches) == 1:
                toolchain = matches[0]
                self._context_add(context, "tool", toolchain.name)
                self._context_add(context, "tool", toolchain.id)
                self._context_add(context, "tool_version", toolchain.version)
                self._context_add(context, "version", toolchain.version)
        artifact_metadata = artifact.metadata
        if artifact.kind == "constraint.xdc":
            # A source constraint cannot self-assert the tool identity under
            # which it applies.  That identity is selected atomically below
            # from the manifest's stage/toolchain mapping.
            artifact_metadata = {
                key: value for key, value in artifact.metadata.items()
                if key not in {"vendor", "tool", "tool_version", "version", "stage"}
            }
        self._context_metadata(
            context, artifact_metadata,
            preserve_existing=("vendor", "tool", "tool_version", "version"),
        )
        self._context_projection_metadata(context, artifact.metadata)
        # These values come from the snapshot-scoped ledger row, never from
        # user-provided artifact metadata.  They let knowledge bindings demand
        # byte identity and snapshot association without exposing a path/body.
        self._context_add(context, "snapshot_id", self.snapshot_id)
        self._context_add(context, "artifact_id", artifact.id)
        self._context_add(context, "artifact_sha256", artifact.sha256)
        self._context_add(context, "snapshot_association", "verified")
        if artifact.kind == "constraint.xdc":
            manifest = self.bundle.store.snapshot_manifest(self.snapshot_id)
            snapshot = self.bundle.store.snapshot(self.snapshot_id)
            same_path = [
                item for item in self.bundle.store.artifacts(self.snapshot_id)
                if item.uri == artifact.uri
            ]
            if (self._snapshot_manifest_identity_valid(manifest, snapshot)
                    and artifact.producer_run_id is None
                    and manifest.constraints.xdc_files.count(artifact.uri) == 1
                    and len(same_path) == 1 and same_path[0].id == artifact.id
                    and snapshot.artifact_hashes.get(artifact.uri) == artifact.sha256
                    and self._snapshot_input_artifact_bytes_valid(artifact)):
                self._context_add(context, "constraint_hash", snapshot.constraint_hash)
                self._context_add(
                    context, _CONSTRAINT_INPUT_CONTEXT_KEY,
                    _CONSTRAINT_INPUT_CONTEXT_VALUE,
                )
                self._context_add(
                    context, "constraint_artifact_identity", stable_hash({
                        "artifact_id": artifact.id,
                        "uri": artifact.uri,
                        "sha256": artifact.sha256,
                        "size": artifact.size,
                        "constraint_hash": snapshot.constraint_hash,
                    }),
                )
                # A constraint input is stage-qualified only by a single exact
                # manifest stage/toolchain identity, not by independently
                # pooled stage and tool/version values.  If stages select
                # different Vivado builds, this one-context projection is
                # deliberately incomplete instead of cross-pairing them.
                stage_pairs: list[tuple[str, Any]] = []
                for stage in ("post_synth", "post_place", "post_route"):
                    if stage not in manifest.stage_commands:
                        continue
                    try:
                        toolchain = manifest.toolchain_for_stage(stage)
                    except (KeyError, TypeError, ValueError):
                        continue
                    if toolchain.name.casefold() == "vivado":
                        stage_pairs.append((stage, toolchain))
                selected_toolchains = {
                    item.id: item for _stage, item in stage_pairs
                }
                if len(selected_toolchains) == 1 and stage_pairs:
                    selected = next(iter(selected_toolchains.values()))
                    context["vendor"] = {
                        canonical_context_scalar(selected.vendor)
                    }
                    context["tool"] = {
                        canonical_context_scalar(selected.name),
                        canonical_context_scalar(selected.id),
                    }
                    context["tool_version"] = {
                        canonical_context_scalar(selected.version)
                    }
                    context["version"] = {
                        canonical_context_scalar(selected.version)
                    }
                    context["stage"] = {
                        canonical_context_scalar(stage)
                        for stage, _toolchain in stage_pairs
                    }
        return context

    def _snapshot_manifest_identity_valid(
        self, manifest: Any, snapshot: Any,
    ) -> bool:
        """Close a live manifest row to every persisted snapshot identity hash."""

        try:
            return bool(
                snapshot.id == self.snapshot_id
                and snapshot.id == stable_id(
                    "snapshot", snapshot.identity_payload(), 32,
                )
                and stable_hash(manifest.identity_payload())
                == snapshot.manifest_hash
                and stable_hash(manifest.build) == snapshot.build_hash
                and stable_hash(manifest.target) == snapshot.target_hash
                and stable_hash(manifest.constraints) == snapshot.constraint_hash
                and stable_hash({
                    "toolchains": manifest.toolchains,
                    "stage_toolchains": manifest.stage_toolchains,
                }) == snapshot.toolchain_hash
            )
        except (AttributeError, TypeError, ValueError):
            return False

    @staticmethod
    def _target_device_identity(manifest: Any) -> str | None:
        target = manifest.target
        identity = {
            "vendor": target.vendor, "part": target.part, "package": target.package,
            "speed_grade": target.speed_grade, "board": target.board,
            "platform": target.platform, "platform_hash": target.platform_hash,
        }
        # A vendor name alone is not a device/capacity identity.
        if not any(identity[key] for key in (
                "part", "package", "speed_grade", "board", "platform",
                "platform_hash")):
            return None
        return stable_hash(identity)

    def _managed_artifact_bytes_valid(self, artifact: Any) -> bool:
        """Revalidate one managed artifact without following a link/reparse hop."""
        if str(artifact.retention) != "managed":
            return False
        project_root = self.bundle.project_root.resolve()
        candidate = self.bundle.project_root / artifact.uri
        try:
            relative = candidate.relative_to(self.bundle.project_root)
        except ValueError:
            return False
        current = self.bundle.project_root
        for part in relative.parts:
            current = current / part
            if _is_link_or_reparse(current):
                return False
        try:
            path = candidate.resolve(strict=True)
            path.relative_to(project_root)
            data = path.read_bytes()
        except (OSError, ValueError):
            return False
        return len(data) == artifact.size and hash_artifact_bytes(data) == artifact.sha256

    def _snapshot_input_artifact_bytes_valid(self, artifact: Any) -> bool:
        """Revalidate one immutable snapshot input used by a source fact."""
        if artifact.producer_run_id is not None:
            return False
        snapshot = self.bundle.store.snapshot(self.snapshot_id)
        if snapshot.artifact_hashes.get(artifact.uri) != artifact.sha256:
            return False
        project_root = self.bundle.project_root.resolve()
        candidate = self.bundle.project_root / artifact.uri
        try:
            relative = candidate.relative_to(self.bundle.project_root)
        except ValueError:
            return False
        current = self.bundle.project_root
        for part in relative.parts:
            current = current / part
            if _is_link_or_reparse(current):
                return False
        try:
            path = candidate.resolve(strict=True)
            path.relative_to(project_root)
            data = path.read_bytes()
        except (OSError, ValueError):
            return False
        return len(data) == artifact.size and hash_artifact_bytes(data) == artifact.sha256

    def _fixed_directive_replay(
        self, *, manifest: Any, artifacts: Mapping[str, Any],
    ) -> DirectiveReplayIndex:
        """Return an ephemeral fixed-parser replay over current snapshot bytes.

        The cache contains only hashes and canonical record commitments.  Every
        use still revalidates all immutable input bytes, so source drift or a
        link/reparse replacement immediately disables the cached proof.
        """
        base_artifacts = {
            item.id: item for item in artifacts.values()
            if item.producer_run_id is None
        }
        cache_key = stable_hash({
            "snapshot_id": self.snapshot_id,
            "manifest": manifest.identity_payload(),
            "artifacts": [
                json_ready(base_artifacts[key]) for key in sorted(base_artifacts)
            ],
        })
        if (
            self._directive_replay_cache is not None
            and self._directive_replay_cache[0] == cache_key
            and all(
                self._snapshot_input_artifact_bytes_valid(item)
                for item in base_artifacts.values()
            )
        ):
            return self._directive_replay_cache[1]
        index = replay_directive_declarations(
            project_root=self.bundle.project_root,
            manifest=manifest,
            snapshot=self.bundle.store.snapshot(self.snapshot_id),
            artifacts=artifacts,
            artifact_bytes_valid=self._snapshot_input_artifact_bytes_valid,
        )
        if index.failure_reason is None:
            self._directive_replay_cache = (cache_key, index)
        return index

    def _declared_managed_run_output_valid(
        self, artifact: Any, *, run: Any, manifest: Any,
        artifacts: Mapping[str, Any],
    ) -> bool:
        """Close one retained output to the exact immutable stage declaration.

        A CAS-shaped URI and matching digest are insufficient: the artifact
        must be the declared output produced by this exact run, with the
        declaration's path, kind, role, access, and licence.  Tool evidence is
        retained as managed bytes, and those bytes are re-read through the
        link/reparse-safe validator immediately before a derived marker is
        minted.
        """
        output_path = artifact.metadata.get("declared_output_path")
        if not isinstance(output_path, str) or not output_path:
            return False
        declared = {
            item.path: item for item in manifest.stage_outputs.get(run.stage, [])
        }
        spec = declared.get(output_path)
        matching_output_ids = [
            item_id for item_id in run.output_artifact_ids
            if item_id in artifacts
            and artifacts[item_id].metadata.get("declared_output_path")
            == output_path
        ]
        expected_artifact_id = stable_id("artifact", {
            "kind": artifact.kind, "uri": artifact.uri,
            "sha256": artifact.sha256, "size": artifact.size,
            "media_type": artifact.media_type, "role": artifact.role,
            "license": artifact.license,
            "producer_run_id": artifact.producer_run_id,
            "retention": str(artifact.retention),
            "access": str(artifact.access), "metadata": artifact.metadata,
        })
        return bool(
            spec is not None
            and artifact.id == expected_artifact_id
            and artifact.producer_run_id == run.id
            and run.output_artifact_ids.count(artifact.id) == 1
            and matching_output_ids == [artifact.id]
            and spec.path == output_path
            and spec.kind == artifact.kind
            and spec.role == artifact.role
            and str(spec.access) == str(artifact.access)
            and spec.license == artifact.license
            and str(artifact.retention) == "managed"
            and self._managed_artifact_bytes_valid(artifact)
        )

    def _qualified_real_gate_run_context(
        self, *, run: Any, manifest: Any, snapshot: Any,
        artifacts: Mapping[str, Any], vendor_only: Mapping[str, set[str]],
        toolchains: Mapping[str, Any],
    ) -> dict[str, set[str]] | None:
        """Return context only for one fresh successful real-tool invocation."""
        expected_run_id = stable_id("run", {
            "snapshot": run.snapshot_id, "stage": run.stage,
            "backend": run.backend, "request": run.request_hash,
            "attempt": run.attempt, "started_at": run.started_at,
        }) if run is not None else None
        if (run is None
                or run.snapshot_id != self.snapshot_id
                or run.id != expected_run_id
                or not self._snapshot_manifest_identity_valid(manifest, snapshot)
                or successful_fresh_tool_run_error(run) is not None
                or tool_run_manifest_identity_error(run, manifest) is not None
                or not self.bundle.store.has_valid_execution_commit(
                    self.snapshot_id, run.id,
                )):
            return None
        artifact_ids = set(artifacts)
        input_ids = list(run.input_artifact_ids)
        output_ids = list(run.output_artifact_ids)
        base_ids = {
            item.id for item in artifacts.values()
            if item.producer_run_id is None
        }
        if (len(input_ids) != len(set(input_ids))
                or len(output_ids) != len(set(output_ids))
                or not set(input_ids).issubset(artifact_ids)
                or not set(output_ids).issubset(artifact_ids)
                or not base_ids.issubset(set(input_ids))):
            return None
        context = self._run_context(run, vendor_only, toolchains)
        self._context_add(context, "snapshot_id", self.snapshot_id)
        self._context_add(context, "snapshot_association", "verified")
        return context

    def _directive_source_evidence(
        self, context: Mapping[str, set[str]], *, subject_id: str,
        observations: Sequence[Any], graph: CanonicalGraph,
        artifacts: Mapping[str, Any], directive_replay: DirectiveReplayIndex,
        require_requested: bool = False, require_unique: bool = False,
    ) -> tuple[Any, Any, dict[str, set[str]], Any] | None:
        """Return one exact request reproduced by the fixed source parsers."""
        directive = graph.entities.get(subject_id)
        if (directive is None or directive.kind != "hls.directive"
                or directive.stage != "source"
                or str(directive.authority) != "declared_constraint"
                or str(directive.completeness) != "complete"
                or context.get("directive_instance_id") != {
                    directive.id.casefold()
                }):
            return None
        directive_anchors = {
            stable_hash(anchor): str(anchor.artifact_id)
            for anchor in directive.anchors if anchor.artifact_id in artifacts
        }
        directive_kind = str(
            directive.attrs.get("directive_kind") or directive.name
        ).upper()
        expected_value = directive.attrs.get("options") or True
        replay_proof = match_directive_replay(
            directive_replay,
            graph=graph,
            observations=observations,
            directive_id=directive.id,
        )
        if replay_proof is None:
            return None
        qualified = []
        for candidate in sorted(
            observations,
            key=lambda item: (
                item.predicate != "directive.requested", item.id,
            ),
        ):
            if (candidate.snapshot_id != self.snapshot_id
                    or candidate.subject_id != directive.id
                    or candidate.predicate not in (
                        {"directive.requested"} if require_requested else {
                            "directive.requested", "directive.declared_selected",
                        }
                    )
                    or candidate.stage != "source"
                    or str(candidate.authority) != "declared_constraint"
                    or str(candidate.completeness) != "complete"
                    or candidate.run_id is not None
                    or stable_hash(candidate.value) != stable_hash(expected_value)
                    or (candidate.predicate == "directive.declared_selected"
                        and directive.attrs.get("state") != "selected_declared")
                    or str(candidate.metadata.get(
                        "directive_kind", "",
                    )).upper() != directive_kind
                    or candidate.artifact_id is None
                    or candidate.anchor is None
                    or candidate.anchor.artifact_id != candidate.artifact_id
                    or stable_hash(candidate.anchor) not in directive_anchors
                    or directive_anchors[stable_hash(candidate.anchor)]
                    != candidate.artifact_id):
                continue
            artifact = artifacts.get(candidate.artifact_id)
            same_path = [
                item for item in artifacts.values()
                if artifact is not None and item.uri == artifact.uri
            ]
            if (artifact is None or len(same_path) != 1
                    or same_path[0].id != artifact.id
                    or not self._snapshot_input_artifact_bytes_valid(artifact)):
                continue
            candidate_context: dict[str, set[str]] = {}
            self._context_directive_identity(
                candidate_context, subject_id=candidate.subject_id,
                metadata=candidate.metadata, graph=graph,
            )
            if (candidate_context.get("directive_instance_id")
                    != {directive.id.casefold()}
                    or candidate_context.get("scope_id") != context.get("scope_id")):
                continue
            if (
                stable_hash(json_ready(candidate))
                != replay_proof.requested_observation_hash
                or candidate.artifact_id != replay_proof.source_artifact_id
                or artifact.sha256 != replay_proof.source_artifact_sha256
                or artifact.size != replay_proof.source_artifact_size
                or stable_hash(json_ready(candidate.anchor))
                != replay_proof.source_anchor_hash
            ):
                continue
            qualified.append((candidate, artifact, candidate_context, replay_proof))
        if not qualified or (require_unique and len(qualified) != 1):
            return None
        return qualified[0]

    def _context_interface_port_ownership(
        self, context: dict[str, set[str]], *, graph: CanonicalGraph,
        directive: Any, replay_proof: Any,
    ) -> bool:
        """Bind an INTERFACE declaration to the unique configured top port."""
        manifest = self.bundle.store.snapshot_manifest(self.snapshot_id)
        scope_values = context.get("scope_id", set())
        port_values = context.get("port_id", set())
        if (scope_values != port_values or len(scope_values) != 1
                or context.get("scope_kind", set()) != {"hls.port"}):
            return False
        port_id = next(iter(scope_values))
        port = graph.entities.get(port_id)
        kernels = [
            entity for entity in graph.entities.values()
            if entity.kind == "hls.kernel"
            and entity.name == manifest.build.top
            and entity.snapshot_id == self.snapshot_id
            and entity.stage == "ast"
            and str(entity.authority) == "static_fact"
            and str(entity.completeness) == "complete"
        ]
        if (port is None or port.kind != "hls.port"
                or port.stage != "ast"
                or str(port.authority) != "static_fact"
                or str(port.completeness) != "complete"
                or len(kernels) != 1):
            return False
        owner = kernels[0]
        ownership = [
            relation for relation in graph.relations.values()
            if relation.kind == "hls.contains"
            and relation.dst == port.id
        ]
        if (len(ownership) != 1 or ownership[0].src != owner.id
                or ownership[0].snapshot_id != self.snapshot_id
                or ownership[0].stage != "ast"
                or str(ownership[0].authority) != "static_fact"
                or str(ownership[0].completeness) != "complete"
                or replay_proof.port_owner_id != owner.id
                or replay_proof.port_owner_kind != "hls.kernel"
                or replay_proof.port_owner_record_hash
                != stable_hash(json_ready(owner))
                or replay_proof.port_owner_relation_hash
                != stable_hash(json_ready(ownership[0]))):
            return False
        self._context_add(context, "port_owner_id", owner.id)
        self._context_add(context, "configured_component_id", owner.id)
        self._context_add(
            context, _PORT_OWNERSHIP_CONTEXT_KEY,
            _PORT_OWNERSHIP_CONTEXT_VALUE,
        )
        self._context_add(context, "port_ownership_identity", stable_hash({
            "contract": _PORT_OWNERSHIP_CONTEXT_VALUE,
            "snapshot_id": self.snapshot_id,
            "configured_top": manifest.build.top,
            "directive_id": directive.id,
            "port_id": port.id,
            "owner_id": owner.id,
            "owner_record_hash": replay_proof.port_owner_record_hash,
            "contains_relation_id": ownership[0].id,
            "contains_relation_hash": replay_proof.port_owner_relation_hash,
            "directive_replay_identity": replay_proof.replay_identity,
        }))
        return True

    def _context_directive_source_evidence(
        self, context: dict[str, set[str]], *, directive_id: str,
        observations: Sequence[Any], graph: CanonicalGraph,
        artifacts: Mapping[str, Any], directive_replay: DirectiveReplayIndex,
    ) -> None:
        """Mint one source-declaration capability from a closed current record.

        The capability is intentionally stronger than a directive entity plus
        copied metadata.  It requires exactly one complete ``directive.requested``
        observation for the current instance, its exact resolved ANNOTATES
        scope (and operand where applicable), one common source anchor, and the
        immutable snapshot input's current no-link size/SHA-256 bytes.
        DEPENDENCE retains its separate operand capability and does not use this
        generic token.
        """
        directive = graph.entities.get(directive_id)
        if directive is None:
            return
        directive_kind = str(
            directive.attrs.get("directive_kind") or directive.name
        ).upper()
        if directive_kind == "DEPENDENCE":
            return
        evidence = self._directive_source_evidence(
            context, subject_id=directive_id, observations=observations,
            graph=graph, artifacts=artifacts, directive_replay=directive_replay,
            require_requested=True, require_unique=True,
        )
        if evidence is None:
            return
        observation, artifact, observation_context, replay_proof = evidence
        identity_keys = (
            "directive_instance_id", "scope_id", "scope_kind",
            "scope_resolution", "function_id", "loop_id", "variable_id",
            "port_id", _DIRECTIVE_OPERAND_CONTEXT_KEY,
            "directive_operand_identity",
        )
        if any(
            context.get(key, set()) != observation_context.get(key, set())
            for key in identity_keys
        ):
            return
        scope_values = context.get("scope_id", set())
        if len(scope_values) != 1:
            return
        scope_id = next(iter(scope_values))
        annotations = [
            relation for relation in graph.relations.values()
            if relation.kind == "hls.annotates"
            and relation.src == directive.id and relation.dst == scope_id
            and relation.stage == "source"
            and str(relation.authority) == "declared_constraint"
            and str(relation.completeness) == "complete"
            and relation.attrs.get("scope_node_id") == scope_id
        ]
        directive_anchors = list(directive.anchors)
        if (len(annotations) != 1 or len(annotations[0].anchors) != 1
                or len(directive_anchors) != 1 or observation.anchor is None
                or stable_hash(annotations[0].anchors[0])
                != stable_hash(observation.anchor)
                or stable_hash(directive_anchors[0])
                != stable_hash(observation.anchor)):
            return
        operand_roles = {
            "ARRAY_PARTITION": "variable_id",
            "STREAM": "variable_id",
            "INTERFACE": "port_id",
        }
        operand_role = operand_roles.get(directive_kind)
        if operand_role is not None:
            operand_values = context.get(operand_role, set())
            if (
                operand_values != {scope_id.casefold()}
                or replay_proof.operand_id != scope_id
                or replay_proof.operand_role != operand_role
            ):
                return
            self._context_add(
                context,
                _DIRECTIVE_OPERAND_CONTEXT_KEY,
                _DIRECTIVE_OPERAND_CONTEXT_VALUE,
            )
            self._context_add(
                context,
                "directive_operand_identity",
                stable_hash({
                    "contract": replay_proof.contract,
                    "replay_identity": replay_proof.replay_identity,
                    "directive_id": directive.id,
                    "operand_id": scope_id,
                    "operand_role": operand_role,
                    "annotates_relation_id": annotations[0].id,
                }),
            )
        if directive_kind == "INTERFACE" and not self._context_interface_port_ownership(
            context, graph=graph, directive=directive,
            replay_proof=replay_proof,
        ):
            return
        self._context_add(
            context, _DIRECTIVE_SOURCE_CONTEXT_KEY,
            _DIRECTIVE_SOURCE_CONTEXT_VALUE,
        )
        self._context_add(context, "directive_source_identity", stable_hash({
            "directive_id": directive.id,
            "directive_kind": directive_kind,
            "scope_id": scope_id,
            "annotates_relation_id": annotations[0].id,
            "operand_identity": next(iter(
                context.get("directive_operand_identity", {""})
            )),
            "source_observation_id": observation.id,
            "source_observation_value_hash": stable_hash(observation.value),
            "source_anchor_hash": stable_hash(observation.anchor),
            "source_artifact_id": artifact.id,
            "source_artifact_sha256": artifact.sha256,
            "source_artifact_size": artifact.size,
            "snapshot_id": self.snapshot_id,
            "directive_replay_contract": replay_proof.contract,
            "directive_replay_identity": replay_proof.replay_identity,
            "source_spelling_hash": replay_proof.source_spelling_hash,
        }))

    def _context_requested_directive_evidence(
        self, context: dict[str, set[str]], *, current_observation: Any,
        observations: Sequence[Any], graph: CanonicalGraph,
        artifacts: Mapping[str, Any], directive_replay: DirectiveReplayIndex,
    ) -> None:
        """Prove a directive predicate has one exact source request record."""
        self._context_directive_source_evidence(
            context, directive_id=current_observation.subject_id,
            observations=observations, graph=graph, artifacts=artifacts,
            directive_replay=directive_replay,
        )
        if _DIRECTIVE_SOURCE_CONTEXT_KEY not in context:
            # DEPENDENCE deliberately uses its stronger, separate operand
            # proof because the directive scope and variable operand differ.
            self._context_dependence_operand_evidence(
                context, directive_id=current_observation.subject_id,
                observations=observations, graph=graph, artifacts=artifacts,
                directive_replay=directive_replay,
            )
        if (_DIRECTIVE_SOURCE_CONTEXT_KEY in context
                or _DEPENDENCE_OPERAND_CONTEXT_KEY in context):
            self._context_add(context, _REQUESTED_DIRECTIVE_CONTEXT_KEY, True)

    def _context_dependence_operand_evidence(
        self, context: dict[str, set[str]], *, directive_id: str,
        observations: Sequence[Any], graph: CanonicalGraph,
        artifacts: Mapping[str, Any], directive_replay: DirectiveReplayIndex,
    ) -> None:
        """Prove a DEPENDENCE operand without reusing its scope annotation."""
        directive = graph.entities.get(directive_id)
        if (directive is None or directive.kind != "hls.directive"
                or str(directive.attrs.get(
                    "directive_kind", directive.name,
                )).upper() != "DEPENDENCE"
                or directive.stage != "source"
                or str(directive.authority) != "declared_constraint"
                or str(directive.completeness) != "complete"):
            return
        scope_values = context.get("scope_id", set())
        variable_values = context.get("variable_id", set())
        if len(scope_values) != 1 or len(variable_values) != 1:
            return
        scope_id = next(iter(scope_values))
        variable_id = next(iter(variable_values))
        if variable_id == scope_id:
            return
        scope = graph.entities.get(scope_id)
        variable = graph.entities.get(variable_id)
        if (scope is None or scope.kind not in {
                "hls.kernel", "hls.function", "hls.loop",
            } or scope.stage != "ast"
                or str(scope.authority) != "static_fact"
                or str(scope.completeness) != "complete"
                or variable is None or variable.kind not in {
                    "hls.memory", "hls.stream", "hls.port", "source.variable",
                } or variable.stage != "ast"
                or str(variable.authority) != "static_fact"
                or str(variable.completeness) != "complete"):
            return
        annotations = [
            relation for relation in graph.relations.values()
            if relation.kind == "hls.annotates"
            and relation.src == directive.id and relation.dst == scope_id
            and relation.stage == "source"
            and str(relation.authority) == "declared_constraint"
            and str(relation.completeness) == "complete"
            and relation.attrs.get("scope_node_id") == scope_id
        ]
        if len(annotations) != 1:
            return

        def function_owners(entity_id: str) -> set[str]:
            entity = graph.entities[entity_id]
            if entity.kind in {"hls.kernel", "hls.function"}:
                return {entity_id}
            frontier = {entity_id}
            visited = set(frontier)
            owners: set[str] = set()
            while frontier:
                parents = {
                    relation.src for relation in graph.relations.values()
                    if relation.kind == "hls.contains"
                    and relation.dst in frontier
                    and relation.src in graph.entities
                    and relation.stage == "ast"
                    and str(relation.authority) == "static_fact"
                    and str(relation.completeness) == "complete"
                } - visited
                visited.update(parents)
                found = {
                    item for item in parents
                    if graph.entities[item].kind in {
                        "hls.kernel", "hls.function",
                    }
                }
                owners.update(found)
                frontier = parents - found
            return owners

        scope_owners = function_owners(scope_id)
        if len(scope_owners) != 1 or function_owners(variable_id) != scope_owners:
            return
        options = directive.attrs.get("options")
        requested_name = (
            options.get("variable") if isinstance(options, Mapping) else None
        )
        if (not isinstance(requested_name, str) or not requested_name
                or variable.name != requested_name):
            return
        operand_candidates = {
            entity.id for entity in graph.entities.values()
            if entity.kind in {
                "hls.memory", "hls.stream", "hls.port", "source.variable",
            }
            and entity.name == requested_name
            and entity.stage == "ast"
            and str(entity.authority) == "static_fact"
            and str(entity.completeness) == "complete"
            and function_owners(entity.id) == scope_owners
        }
        if operand_candidates != {variable_id}:
            return
        source_evidence = self._directive_source_evidence(
            context, subject_id=directive.id, observations=observations,
            graph=graph, artifacts=artifacts, directive_replay=directive_replay,
            require_requested=True, require_unique=True,
        )
        if source_evidence is None:
            return
        observation, artifact, observation_context, replay_proof = source_evidence
        if observation_context.get("variable_id") != {variable_id}:
            return
        if (
            replay_proof.directive_kind != "DEPENDENCE"
            or replay_proof.scope_id != scope_id
            or replay_proof.operand_id != variable_id
            or replay_proof.operand_role != "variable_id"
        ):
            return
        if (len(annotations[0].anchors) != 1
                or stable_hash(annotations[0].anchors[0])
                != stable_hash(observation.anchor)):
            return
        owner_id = next(iter(scope_owners))
        self._context_add(
            context, _DEPENDENCE_OPERAND_CONTEXT_KEY,
            _DEPENDENCE_OPERAND_CONTEXT_VALUE,
        )
        self._context_add(context, "directive_operand_identity", stable_hash({
            "directive_id": directive.id,
            "operand_id": variable_id,
            "scope_id": scope_id,
            "function_owner_id": owner_id,
            "annotates_relation_id": annotations[0].id,
            "source_observation_id": observation.id,
            "source_observation_value_hash": stable_hash(observation.value),
            "source_anchor_hash": stable_hash(observation.anchor),
            "source_artifact_id": artifact.id,
            "source_artifact_sha256": artifact.sha256,
            "directive_replay_contract": replay_proof.contract,
            "directive_replay_identity": replay_proof.replay_identity,
            "source_spelling_hash": replay_proof.source_spelling_hash,
        }))

    def _qualified_observation_context(
        self, *, observation: Any, run: Any, artifacts: Mapping[str, Any],
        manifest: Any, vendor_only: Mapping[str, set[str]],
        toolchains: Mapping[str, Any], graph: CanonicalGraph,
        parser_replay_cache: dict[tuple[str, str, str], tuple[Any, ...]],
    ) -> dict[str, set[str]] | None:
        """Close one observation to its own declared, live tool-report evidence.

        The returned reserved markers are capability-like values.  They are
        minted only for the current observation and cannot be supplied by
        metadata or borrowed from another record in the snapshot.
        """
        if (observation.snapshot_id != self.snapshot_id
                or str(observation.completeness) != "complete"
                or run is None or run.snapshot_id != self.snapshot_id
                or observation.run_id != run.id
                or successful_fresh_tool_run_error(run) is not None
                or tool_run_manifest_identity_error(run, manifest) is not None
                or not self.bundle.store.has_valid_execution_commit(
                    self.snapshot_id, run.id,
                )):
            return None
        base_ids = {
            item.id for item in artifacts.values()
            if item.producer_run_id is None
        }
        if not base_ids.issubset(set(run.input_artifact_ids)):
            return None

        source = observation.source
        if (source is None or not observation.artifact_id
                or observation.anchor is None
                or observation.artifact_id != observation.anchor.artifact_id
                or source.artifact_id != observation.artifact_id):
            return None
        artifact = artifacts.get(source.artifact_id)
        if artifact is None or source.artifact_sha256 != artifact.sha256:
            return None
        if source.validation_error(
            predicate=observation.predicate,
            value=observation.value,
            unit=observation.unit,
        ) is not None:
            return None
        parser_kinds = _OBSERVATION_PARSER_POLICY.get(
            (source.parser_name, source.parser_version),
        )
        if parser_kinds is None or artifact.kind not in parser_kinds:
            return None
        reports = [artifact]
        policy = _OBSERVATION_REPORT_POLICY.get(
            (str(observation.predicate), str(observation.stage)),
        )
        authority = str(observation.authority)
        if (not policy
                or any((item.kind, authority) not in policy for item in reports)
                or tool_evidence_compatibility_error(
                    observation, run, reports,
                ) is not None):
            return None

        workload: str | None = None
        testcase: str | None = None
        if str(observation.predicate) in _DYNAMIC_OBSERVATION_PREDICATES:
            workload = observation.workload_id
            if (not isinstance(workload, str) or not workload
                    or run.metadata.get("workload_id") != workload
                    or any(item.metadata.get("workload_id") != workload
                           for item in reports)):
                return None
            testcase_records = (
                observation.metadata.get("testcase_id"),
                run.metadata.get("testcase_id"),
                *(item.metadata.get("testcase_id") for item in reports),
            )
            if any(value is not None for value in testcase_records):
                if any(not isinstance(value, str) or not value
                       for value in testcase_records):
                    return None
                testcase_values = set(testcase_records)
                if len(testcase_values) != 1:
                    return None
                testcase = next(iter(testcase_values))

        output_path = artifact.metadata.get("declared_output_path")
        if not isinstance(output_path, str) or not output_path:
            return None
        matching_specs = [
            item for item in manifest.stage_outputs.get(run.stage, [])
            if item.path == output_path
        ]
        if len(matching_specs) != 1:
            return None
        spec = matching_specs[0]
        owned = [
            item for item in artifacts.values()
            if (item.producer_run_id == run.id
                and item.metadata.get("declared_output_path") == output_path)
        ]
        if (len(owned) != 1 or owned[0].id != artifact.id
                or spec.kind != artifact.kind
                or spec.role != artifact.role
                or str(spec.access) != str(artifact.access)
                or spec.license != artifact.license
                or artifact.producer_run_id != run.id
                or run.output_artifact_ids.count(artifact.id) != 1
                or not self._managed_artifact_bytes_valid(artifact)):
            return None

        # A source receipt is only a content commitment; callers can construct
        # a self-consistent hash for a fabricated value.  Re-run the fixed
        # built-in parser over the exact live report bytes before minting the
        # capability-like observation marker used by knowledge applicability.
        from .extract.observation_replay import replay_observation_source_error
        replay_error = replay_observation_source_error(
            project_root=self.bundle.project_root,
            manifest=manifest,
            snapshot=self.bundle.store.snapshot(self.snapshot_id),
            graph=graph,
            artifact=artifact,
            observation=observation,
            cache=parser_replay_cache,
        )
        if replay_error is not None:
            return None

        context = self._run_context(run, vendor_only, toolchains)
        context["stage"] = {str(observation.stage).casefold()}
        self._context_add(context, "snapshot_id", self.snapshot_id)
        self._context_add(context, "snapshot_association", "verified")
        self._context_add(context, "workload_id", workload)
        self._context_add(context, "testcase_id", testcase)
        self._context_add(
            context, _OBSERVATION_EVIDENCE_CONTEXT_KEY,
            _OBSERVATION_EVIDENCE_CONTEXT_VALUE,
        )
        self._context_add(context, "observation_instance_id", observation.id)
        self._context_add(
            context, "observation_source_identity", stable_hash(source),
        )
        self._context_add(context, "observation_parser_identity", stable_hash({
            "name": source.parser_name,
            "version": source.parser_version,
            "contract": source.contract,
        }))
        report_kinds = {item.kind for item in reports}
        if len(report_kinds) == 1:
            self._context_add(
                context, _OBSERVATION_ARTIFACT_KIND_CONTEXT_KEY,
                next(iter(report_kinds)),
            )
        self._context_add(context, "observation_artifact_identity", stable_hash([
            {"artifact_id": item.id, "sha256": item.sha256}
            for item in reports
        ]))
        self._context_add(context, "observation_run_identity", stable_hash({
            "run_id": run.id,
            "request_hash": run.request_hash,
            "stage": run.stage,
            "toolchain_id": run.toolchain_id,
            "environment_hash": run.environment_hash,
        }))
        return context

    def _qualified_gate_contexts(
        self, *, vendor_only: Mapping[str, set[str]], toolchains: Mapping[str, Any],
        runs: Mapping[str, Any], artifacts: Mapping[str, Any],
        observations: Sequence[Any], derivations: Sequence[Mapping[str, Any]],
        verifications: Sequence[Mapping[str, Any]],
    ) -> list[tuple[str, dict[str, set[str]]]]:
        """Build gate contexts only from one closed, snapshot-local evidence chain.

        This is deliberately independent of a free-standing ``ToolRun.gates``
        claim.  A knowledge rule can qualify a gate only through parser-typed
        observations/verification records or an approved physical derivation;
        unrelated runs and metadata cannot donate missing identity fields.
        """
        manifest = self.bundle.store.snapshot_manifest(self.snapshot_id)
        snapshot = self.bundle.store.snapshot(self.snapshot_id)
        by_observation = {item.id: item for item in observations}
        result: list[tuple[str, dict[str, set[str]]]] = []

        def mark_typed_evidence(context: dict[str, set[str]]) -> None:
            # Deliberately local to this evidence-closing routine.  Generic
            # metadata projection has no path to this reserved value.
            self._context_add(
                context,
                _GATE_EVIDENCE_CONTEXT_KEY,
                _GATE_EVIDENCE_CONTEXT_VALUE,
            )

        def real_context(run: Any) -> dict[str, set[str]] | None:
            return self._qualified_real_gate_run_context(
                run=run, manifest=manifest, snapshot=snapshot,
                artifacts=artifacts, vendor_only=vendor_only,
                toolchains=toolchains,
            )

        def leaf_artifacts(
            observation: Any, run: Any, allowed_kinds: set[str],
        ) -> list[Any] | None:
            artifact_ids = {
                str(item) for item in (
                    observation.artifact_id,
                    observation.anchor.artifact_id if observation.anchor else None,
                ) if item
            }
            if not observation.artifact_id or not artifact_ids:
                return None
            reports = [artifacts.get(item) for item in sorted(artifact_ids)]
            if (any(item is None for item in reports)
                    or any(item.kind not in allowed_kinds for item in reports)
                    or any(not self._declared_managed_run_output_valid(
                        item, run=run, manifest=manifest, artifacts=artifacts,
                    ) for item in reports)
                    or tool_evidence_compatibility_error(
                        observation, run, reports,
                    ) is not None):
                return None
            return reports

        def physical_scope(artifact: Any, *, timing: bool = False) -> tuple[str, ...] | None:
            scope = artifact.metadata.get("scope")
            if not isinstance(scope, Mapping):
                return None
            top = manifest.build.top
            if (scope.get("kind") != "kernel"
                    or str(scope.get("top") or "") != top
                    or str(scope.get("instance") or "") != top):
                return None
            if (manifest.target.part
                    and str(scope.get("part") or "") != manifest.target.part):
                return None
            if (manifest.target.platform
                    and str(scope.get("platform") or "") != manifest.target.platform):
                return None
            if timing:
                clock = str(scope.get("clock") or "")
                known = {item.name for item in manifest.target.clocks}
                if clock != "all" and clock not in known:
                    return None
            return (
                str(scope.get("kind")), str(scope.get("top")),
                str(scope.get("instance")), str(scope.get("part") or ""),
                str(scope.get("platform") or ""),
            )

        verification_policy = {
            "csim": ("csim", {"amd.vitis.csim_result"}),
            "rtl_cosim": (
                "rtl_cosim", {"amd.vitis.cosim_rpt", "amd.vitis.cosim_report"},
            ),
        }
        for verification in verifications:
            kind = str(verification.get("kind", ""))
            if kind not in verification_policy:
                continue
            run = runs.get(str(verification.get("run_id") or ""))
            expected_stage, allowed_kinds = verification_policy[kind]
            if (verification.get("snapshot_id") != self.snapshot_id
                    or run is None or run.stage != expected_stage):
                continue
            context = real_context(run)
            evidence_ids = [str(item) for item in verification.get("evidence_ids", [])]
            leaves = [by_observation.get(item) for item in evidence_ids]
            verification_workload = verification.get("workload_id")
            run_workload = run.metadata.get("workload_id")
            workload = str(verification_workload or "")
            if (context is None or not workload
                    or run_workload != verification_workload
                    or not leaves or len(evidence_ids) != len(set(evidence_ids))
                    or any(item is None for item in leaves)):
                continue
            typed_artifacts: dict[str, Any] = {}
            valid = True
            for observation in leaves:
                if (observation.snapshot_id != self.snapshot_id
                        or observation.run_id != run.id
                        or str(observation.completeness) != "complete"
                        or str(observation.authority) not in {
                            "tool_observation", "verification_evidence",
                            "physical_measurement",
                        }
                        or observation.workload_id != workload
                        or observation.stage not in {"csim", "cosim"}):
                    valid = False
                    break
                current = leaf_artifacts(observation, run, allowed_kinds)
                if (current is None
                        or any(item.metadata.get("workload_id") != workload
                               for item in current)):
                    valid = False
                    break
                typed_artifacts.update((item.id, item) for item in current)
            if not valid:
                continue
            # The executable stage is ``rtl_cosim`` while parser observations
            # and the versioned rule use the semantic evidence stage ``cosim``.
            # Keep that mapping local to this closed verification instance.
            context["stage"] = {
                "cosim" if kind == "rtl_cosim" else "csim"
            }
            self._context_add(context, "workload_id", workload)
            mark_typed_evidence(context)
            self._context_add(
                context, "verification_observation_identity",
                stable_hash(sorted(item.id for item in leaves)),
            )
            self._context_add(
                context, "verification_report_identity",
                stable_hash(sorted(
                    (item.id, item.sha256) for item in typed_artifacts.values()
                )),
            )
            result.append(("correctness", context))

        for derivation in derivations:
            predicate = str(derivation.get("predicate") or "")
            if predicate not in {"gate.resource_fits", "gate.post_route_timing"}:
                continue
            input_ids = [str(item) for item in
                         derivation.get("input_observation_ids", [])]
            if (derivation.get("snapshot_id") != self.snapshot_id
                    or derivation.get("stage") != "post_route"
                    or derivation.get("authority") != "derived_fact"
                    or derivation.get("completeness") != "complete"
                    or len(input_ids) != len(set(input_ids))):
                continue
            allowed_report_kinds = {
                "gate.resource_fits": {
                    "amd.vivado.post_route_utilization", "amd.vivado.utilization",
                },
                "gate.post_route_timing": {
                    "amd.vivado.post_route_timing", "amd.vivado.timing_summary",
                },
            }[predicate]
            leaves = [by_observation.get(item) for item in input_ids]
            if not leaves or any(item is None for item in leaves):
                continue
            producer_ids = {str(item.run_id or "") for item in leaves}
            if len(producer_ids) != 1 or "" in producer_ids:
                continue
            run = runs.get(next(iter(producer_ids)))
            if run is None or run.stage != "post_route":
                continue
            if any(item.subject_id != derivation.get("subject_id") for item in leaves):
                continue
            context = real_context(run)
            if context is None:
                continue
            reports: dict[str, Any] = {}
            valid = True
            for observation in leaves:
                if (observation.snapshot_id != self.snapshot_id
                        or observation.run_id != run.id
                        or observation.stage != "post_route"
                        or str(observation.completeness) != "complete"
                        or str(observation.authority) not in {
                            "tool_observation", "verification_evidence",
                            "physical_measurement",
                        }
                ):
                    valid = False
                    break
                current = leaf_artifacts(observation, run, allowed_report_kinds)
                if current is None:
                    valid = False
                    break
                reports.update((item.id, item) for item in current)
            if not valid:
                continue
            metadata = derivation.get("metadata", {})
            if not isinstance(metadata, Mapping):
                continue
            if predicate == "gate.resource_fits":
                allowed = {"amd.vivado.post_route_utilization", "amd.vivado.utilization"}
                unique_reports = {(item.id, item.sha256) for item in reports.values()
                                  if item.kind in allowed}
                usage_names = [
                    item.predicate.split(".", 1)[1].casefold()
                    for item in leaves if item.predicate.startswith("resource.")
                ]
                usage = set(usage_names)
                numeric_usage = {
                    item.predicate.split(".", 1)[1].casefold(): float(item.value)
                    for item in leaves
                    if (item.predicate.startswith("resource.")
                        and isinstance(item.value, (int, float))
                        and not isinstance(item.value, bool)
                        and math.isfinite(float(item.value))
                        and float(item.value) >= 0)
                }
                capacities = {str(key).casefold(): float(value)
                              for key, value in manifest.target.capacities.items()}
                reserved = {str(key).casefold(): float(value) for key, value in
                            manifest.target.reserved_resources.items()}
                capacity_keys = set(capacities)
                device_identity = self._target_device_identity(manifest)
                expected_target = stable_hash(manifest.target)
                if (len(unique_reports) != 1 or not capacity_keys
                        or usage != capacity_keys
                        or len(usage_names) != len(capacity_keys)
                        or set(numeric_usage) != capacity_keys
                        or any(item.unit != "count" for item in leaves)
                        or any(physical_scope(item) is None
                               for item in reports.values())
                        or (any(item.kind == "amd.vivado.utilization"
                                and item.metadata.get("stage") != "post_route"
                                for item in reports.values()))
                        or derivation.get("algorithm") != "hlsgraph.gate.capacity_compare"
                        or str(derivation.get("algorithm_version")) != "1"
                        or device_identity is None
                        or metadata.get("target_profile_hash") != expected_target):
                    continue
                expected_fit = all(
                    numeric_usage[name] <= capacities[name] - reserved.get(name, 0.0)
                    for name in capacity_keys
                )
                if derivation.get("value") is not expected_fit:
                    continue
                self._context_add(context, "complete_post_route_utilization", "true")
                self._context_add(
                    context, "utilization_report_identity",
                    stable_hash(sorted(unique_reports)),
                )
                self._context_add(context, "target_profile_hash", expected_target)
                self._context_add(context, "target_device_identity", device_identity)
                self._context_add(context, "capacity_identity", stable_hash({
                    "capacities": manifest.target.capacities,
                    "reserved_resources": manifest.target.reserved_resources,
                }))
                mark_typed_evidence(context)
                result.append(("resource_fits", context))
                continue

            timing_reports = {(item.id, item.sha256) for item in reports.values()
                              if item.kind in {
                                  "amd.vivado.post_route_timing",
                                  "amd.vivado.timing_summary",
                              }}
            report_scopes = {
                physical_scope(item, timing=True) for item in reports.values()
                if item.kind in {
                    "amd.vivado.post_route_timing", "amd.vivado.timing_summary",
                }
            }
            routed = [
                item for item in artifacts.values()
                if item.kind == "amd.vivado.routed_checkpoint"
                and item.producer_run_id == run.id
                and item.id in run.output_artifact_ids
                and item.metadata.get("stage") == "post_route"
                and self._declared_managed_run_output_valid(
                    item, run=run, manifest=manifest, artifacts=artifacts,
                )
                and physical_scope(item) in report_scopes
            ]
            declared_xdc_list = list(manifest.constraints.xdc_files)
            declared_xdc = set(declared_xdc_list)
            constraint_artifacts = []
            valid_constraints = len(declared_xdc_list) == len(declared_xdc)
            for uri in sorted(declared_xdc):
                candidates = [
                    item for item in artifacts.values()
                    if item.kind == "constraint.xdc"
                    and item.uri == uri
                    and item.producer_run_id is None
                ]
                if (len(candidates) != 1
                        or snapshot.artifact_hashes.get(uri)
                        != candidates[0].sha256
                        or candidates[0].id not in run.input_artifact_ids
                        or not self._snapshot_input_artifact_bytes_valid(
                            candidates[0]
                        )):
                    valid_constraints = False
                    break
                constraint_artifacts.append(candidates[0])
            wns = [item for item in leaves if item.predicate == "timing.wns_ns"]
            if (len(timing_reports) != 1 or None in report_scopes
                    or len(report_scopes) != 1 or len(routed) != 1
                    or len(wns) != 1
                    or any(item.predicate not in {"timing.wns_ns", "timing.tns_ns"}
                           or item.unit != "ns" for item in leaves)
                    or derivation.get("algorithm") != "hlsgraph.gate.wns_nonnegative"
                    or str(derivation.get("algorithm_version")) != "1"
                    or (any(item.kind == "amd.vivado.timing_summary"
                            and item.metadata.get("stage") != "post_route"
                            for item in reports.values()))
                    or not valid_constraints
                    or {item.uri for item in constraint_artifacts} != declared_xdc
            ):
                continue
            observed_wns = wns[0].value
            if (isinstance(observed_wns, bool)
                    or not isinstance(observed_wns, (int, float))
                    or not math.isfinite(float(observed_wns))
                    or derivation.get("value") is not (float(observed_wns) >= 0.0)):
                continue
            self._context_add(
                context, "timing_report_identity", stable_hash(sorted(timing_reports)),
            )
            self._context_add(context, "routed_design_identity", routed[0].sha256)
            self._context_add(context, "constraint_hash", snapshot.constraint_hash)
            self._context_add(
                context, "constraint_artifact_identity", stable_hash([
                    {
                        "artifact_id": item.id, "uri": item.uri,
                        "sha256": item.sha256,
                    }
                    for item in sorted(
                        constraint_artifacts, key=lambda value: value.uri,
                    )
                ]),
            )
            mark_typed_evidence(context)
            result.append(("post_route_timing", context))
        return result

    def _binding_target_contexts(
        self, graph: CanonicalGraph, allowed_ids: set[str],
    ) -> dict[tuple[str, str], list[dict[str, set[str]]]]:
        """Return contexts attached to individual target records.

        No stage, workload, or producer-tool value is pooled across records.
        This prevents a cosimulation run elsewhere in the snapshot from making
        an unrelated synthesis predicate look workload-qualified.
        """
        manifest_context, toolchains = self._manifest_context()
        source_tool_context = self._source_tool_context(
            manifest_context, toolchains,
        )
        manifest = self.bundle.store.snapshot_manifest(self.snapshot_id)
        vendor_only = {
            key: set(values) for key, values in manifest_context.items()
            if key == "vendor"
        }
        runs = {item.id: item for item in self.bundle.store.runs(self.snapshot_id)}
        artifacts = {item.id: item for item in self.bundle.store.artifacts(self.snapshot_id)}
        observations = self.bundle.store.observations(self.snapshot_id)
        derivations = self.bundle.store.derivations(self.snapshot_id)
        verifications = self.bundle.store.verifications(self.snapshot_id)
        needs_directive_replay = any(
            entity_id in allowed_ids and entity.kind == "hls.directive"
            for entity_id, entity in graph.entities.items()
        ) or any(
            item.subject_id in allowed_ids
            and item.predicate.startswith("directive.")
            for item in observations
        )
        directive_replay = (
            self._fixed_directive_replay(
                manifest=manifest, artifacts=artifacts,
            )
            if needs_directive_replay
            else DirectiveReplayIndex.failed("not_required")
        )
        parser_replay_cache: dict[
            tuple[str, str, str], tuple[Any, ...]
        ] = {}
        result: dict[tuple[str, str], list[dict[str, set[str]]]] = defaultdict(list)

        for entity_id in sorted(allowed_ids):
            entity = graph.entities[entity_id]
            context = self._context_copy(
                source_tool_context
                if entity.kind == "hls.directive" else manifest_context
            )
            context["stage"] = {entity.stage.casefold()}
            self._context_entity_evidence(context, entity, current=True)
            self._context_metadata(context, entity.attrs)
            if entity.kind == "hls.directive":
                self._context_directive_identity(
                    context, subject_id=entity.id, metadata=entity.attrs,
                    graph=graph,
                )
                self._context_dependence_operand_evidence(
                    context, directive_id=entity.id,
                    observations=observations, graph=graph,
                    artifacts=artifacts, directive_replay=directive_replay,
                )
                self._context_directive_source_evidence(
                    context, directive_id=entity.id,
                    observations=observations, graph=graph,
                    artifacts=artifacts, directive_replay=directive_replay,
                )
            self._context_projection_metadata(context, entity.attrs)
            self._context_unique_anchor_artifact(
                context, (entity_id,), graph, artifacts,
            )
            if entity.kind.startswith("ir.mlir.") or entity.stage == "mlir":
                self._context_add(context, "ir", "mlir")
            elif entity.kind.startswith("ir.llvm.") or entity.stage == "llvm":
                self._context_add(context, "ir", "llvm")
            result[("entity_kind", entity.kind)].append(context)
            directive_kind = entity.attrs.get("directive_kind")
            if isinstance(directive_kind, str):
                result[("directive_kind", directive_kind)].append(
                    self._context_copy(context)
                )
        for relation in sorted(graph.relations.values(), key=lambda item: item.id):
            if relation.src not in allowed_ids or relation.dst not in allowed_ids:
                continue
            context = self._context_copy(
                source_tool_context
                if relation.stage == "source" else manifest_context
            )
            context["stage"] = {relation.stage.casefold()}
            self._context_relation_evidence(
                context, relation, graph, current=True,
            )
            self._context_metadata(context, relation.attrs)
            self._context_metadata(context, graph.entities[relation.src].attrs)
            self._context_metadata(context, graph.entities[relation.dst].attrs)
            self._context_projection_metadata(context, relation.attrs)
            self._context_projection_metadata(
                context, graph.entities[relation.src].attrs,
            )
            self._context_projection_metadata(
                context, graph.entities[relation.dst].attrs,
            )
            self._context_unique_anchor_artifact(
                context, (relation.src, relation.dst), graph, artifacts,
                relation.anchors,
            )
            if relation.stage == "mlir":
                self._context_add(context, "ir", "mlir")
            elif relation.stage == "llvm":
                self._context_add(context, "ir", "llvm")
            result[("relation_kind", relation.kind)].append(context)

        for observation in observations:
            if observation.subject_id not in allowed_ids:
                continue
            run = runs.get(observation.run_id) if observation.run_id else None
            artifact = artifacts.get(observation.artifact_id) if observation.artifact_id else None
            qualified = self._qualified_observation_context(
                observation=observation, run=run, artifacts=artifacts,
                manifest=manifest, vendor_only=vendor_only,
                toolchains=toolchains, graph=graph,
                parser_replay_cache=parser_replay_cache,
            ) if run is not None else None
            if qualified is not None:
                context = qualified
            elif run is not None:
                context = self._run_context(run, vendor_only, toolchains)
            elif artifact is not None:
                context = self._artifact_context(
                    artifact,
                    source_tool_context
                    if observation.stage == "source" else vendor_only,
                    toolchains,
                )
            else:
                context = self._context_copy(
                    source_tool_context
                    if observation.stage == "source" else vendor_only
                )
            context["stage"] = {observation.stage.casefold()}
            self._context_ir_stage(context, observation.stage)
            if observation.workload_id:
                context["workload_id"] = {observation.workload_id.casefold()}
            for key in ("testcase_id", "activity_source"):
                value = observation.metadata.get(key)
                if isinstance(value, str) and value:
                    context[key] = {value.casefold()}
            self._context_metadata(
                context, observation.metadata,
                preserve_existing=(
                    "vendor", "tool", "tool_version", "version", "stage",
                    "workload_id", "testcase_id", "activity_source",
                ),
            )
            if observation.predicate.startswith("directive."):
                self._context_directive_identity(
                    context, subject_id=observation.subject_id,
                    metadata=observation.metadata, graph=graph,
                )
                self._context_requested_directive_evidence(
                    context, current_observation=observation,
                    observations=observations, graph=graph,
                    artifacts=artifacts, directive_replay=directive_replay,
                )
            result[("predicate", observation.predicate)].append(context)
        for derivation in derivations:
            if str(derivation.get("subject_id", "")) not in allowed_ids:
                continue
            predicate = derivation.get("predicate")
            if not isinstance(predicate, str):
                continue
            stage = derivation.get("stage")
            context = self._context_copy(
                source_tool_context if stage == "source" else vendor_only
            )
            if isinstance(stage, str) and stage:
                context["stage"] = {stage.casefold()}
                self._context_ir_stage(context, stage)
            metadata = derivation.get("metadata", {})
            if isinstance(metadata, Mapping):
                self._context_metadata(context, metadata)
                for key in ("workload_id", "testcase_id", "activity_source"):
                    value = metadata.get(key)
                    if isinstance(value, str) and value:
                        context[key] = {value.casefold()}
            self._context_derivation_evidence(
                context, derivation, graph, artifacts,
            )
            result[("predicate", predicate)].append(context)

        for diagnostic in self.bundle.store.active_diagnostics(self.snapshot_id):
            if diagnostic.subject_id is not None and diagnostic.subject_id not in allowed_ids:
                continue
            run = runs.get(diagnostic.run_id) if diagnostic.run_id else None
            artifact = artifacts.get(diagnostic.artifact_id) if diagnostic.artifact_id else None
            if run is not None:
                context = self._run_context(run, vendor_only, toolchains)
            elif artifact is not None:
                context = self._artifact_context(artifact, vendor_only, toolchains)
            else:
                context = self._context_copy(vendor_only)
            if artifact is not None:
                self._context_semantic_artifact_evidence(
                    context, graph, {artifact.id}, artifacts,
                )
            context["stage"] = {diagnostic.stage.casefold()}
            self._context_ir_stage(context, diagnostic.stage)
            for key in ("workload_id", "testcase_id", "activity_source"):
                value = diagnostic.metadata.get(key)
                if isinstance(value, str) and value:
                    context[key] = {value.casefold()}
            self._context_metadata(
                context, diagnostic.metadata,
                preserve_existing=(
                    "vendor", "tool", "tool_version", "version", "stage",
                    "workload_id", "testcase_id", "activity_source",
                ),
            )
            self._context_add(
                context, "diagnostic_instance_id", diagnostic.id,
            )
            if diagnostic.code == "mapping.ambiguous_mlir_location":
                mapping_kind = diagnostic.metadata.get("mapping_kind")
                location_kind = diagnostic.metadata.get("location_kind")
                provenance = diagnostic.metadata.get("mapping_provenance")
                if (mapping_kind == "mlir.location"
                        and location_kind in _CONCRETE_MLIR_MAPPING_LOCATION_KINDS
                        and provenance == "mlir.location_anchor"):
                    self._context_add(
                        context, "typed_mlir_location_present", "true",
                    )
                    self._context_add(context, "mapping_kind", mapping_kind)
                    self._context_add(context, "location_kind", location_kind)
                    self._context_add(
                        context, "mapping_provenance", provenance,
                    )
            result[("diagnostic_code", diagnostic.code)].append(context)

        for artifact in artifacts.values():
            run = runs.get(artifact.producer_run_id) if artifact.producer_run_id else None
            qualified_artifact = self._qualified_tool_artifact_context(
                artifact=artifact, run=run, artifacts=artifacts,
                manifest=manifest, vendor_only=vendor_only,
                toolchains=toolchains,
            ) if run is not None else None
            context = (
                qualified_artifact if qualified_artifact is not None
                else self._run_context(run, vendor_only, toolchains)
                if run is not None
                else self._artifact_context(artifact, vendor_only, toolchains)
            )
            stage = artifact.metadata.get("stage")
            if (qualified_artifact is None and artifact.kind != "constraint.xdc"
                    and isinstance(stage, str) and stage):
                context["stage"] = {stage.casefold()}
                self._context_ir_stage(context, stage)
            for key in ("workload_id", "testcase_id", "activity_source"):
                value = artifact.metadata.get(key)
                if isinstance(value, str) and value:
                    context[key] = {value.casefold()}
            public_artifact_metadata = artifact.metadata
            if artifact.kind == "constraint.xdc":
                public_artifact_metadata = {
                    key: value for key, value in artifact.metadata.items()
                    if key not in {
                        "vendor", "tool", "tool_version", "version", "stage",
                    }
                }
            self._context_metadata(
                context, public_artifact_metadata,
                preserve_existing=(
                    "vendor", "tool", "tool_version", "version", "stage",
                    "workload_id", "testcase_id", "activity_source",
                ),
            )
            result[("artifact_kind", artifact.kind)].append(context)

        for run in runs.values():
            context = self._run_context(run, vendor_only, toolchains)
            for gate in run.gates:
                result[("gate_kind", str(gate.kind))].append(
                    self._context_copy(context)
                )
        for gate_kind, context in self._qualified_gate_contexts(
            vendor_only=vendor_only, toolchains=toolchains, runs=runs,
            artifacts=artifacts, observations=observations,
            derivations=derivations, verifications=verifications,
        ):
            result[("gate_kind", gate_kind)].append(context)
        return dict(result)

    def _binding_evaluation(
        self,
        session: BindingActivationSession,
        binding: Any,
        context: AttestedBindingContext,
        spec: RetrievalSpec,
    ) -> _BindingEvaluation | None:
        """Return one terminal decision from the local reviewed session.

        Production passes the session created in the same ``_documents`` call;
        no issuer field or mutable retriever registry can introduce a different
        activation surface.
        """

        def evaluate(
            detached_binding: Any,
            detached_rule: Any,
            detached_values: Mapping[str, set[str]],
        ) -> _BindingEvaluation:
            targets = {
                str(detached_binding.target_kind): {
                    str(detached_binding.target),
                },
            }
            binding_matches = self._binding_constraints_match_values(
                detached_binding, detached_values, targets,
                condition=detached_rule.condition,
            )
            revision_unbound = bool(
                not binding_matches
                and self._binding_constraints_missing_only_artifact_revision(
                    detached_binding, detached_values, targets,
                    condition=detached_rule.condition,
                )
            )
            rule_applicable, rule_reason = self._rule_applicable(
                detached_rule.applicability, detached_values,
            )
            return _BindingEvaluation(
                binding=detached_binding,
                rule=detached_rule,
                values=detached_values,
                request_matches=self._context_matches_request(
                    detached_values, spec,
                ),
                binding_matches=binding_matches,
                revision_unbound=revision_unbound,
                rule_applicable=rule_applicable,
                rule_reason=rule_reason,
            )

        return session.evaluate_atomically(binding, context, evaluate)

    @staticmethod
    def _binding_constraints_match_values(
        binding: Any, context: Mapping[str, set[str]],
        targets: Mapping[str, set[str]], *,
        condition: Mapping[str, Any] | None = None,
    ) -> bool:
        """Inspect scalar constraints without granting activation authority.

        Pack-authoring and unit tests use this pure matcher.  A ``True`` result
        says only that supplied values are mutually compatible; production
        retrieval must still run the local reviewed session's atomic evaluator
        with an issued :class:`AttestedBindingContext`.
        """
        if str(binding.target) not in targets.get(str(binding.target_kind), set()):
            return False
        required = getattr(binding, "required_context", {})
        if not isinstance(required, Mapping):
            return False
        if not HybridRetriever._binding_semantics_complete(binding, required):
            return False
        if not HybridRetriever._binding_context_integrity(required, context):
            return False
        for raw_key, raw_expected in required.items():
            key = "tool_version" if raw_key == "version" else str(raw_key)
            actual = context.get(key, set()) or context.get(str(raw_key), set())
            if not HybridRetriever._constraint_matches(raw_expected, actual):
                return False
        return HybridRetriever._binding_condition_applicable(
            binding, condition or {}, context, targets,
        )

    @staticmethod
    def _binding_constraints_missing_only_artifact_revision(
        binding: Any, context: Mapping[str, set[str]],
        targets: Mapping[str, set[str]],
        *, condition: Mapping[str, Any] | None = None,
    ) -> bool:
        """Pure non-authorizing inspection for the revision-unbound case."""
        if str(binding.target) not in targets.get(str(binding.target_kind), set()):
            return False
        required = getattr(binding, "required_context", {})
        if (not isinstance(required, Mapping)
                or HybridRetriever._constraint_mentions(
                    required.get(_SEMANTIC_ARTIFACT_EVIDENCE_CONTEXT_KEY),
                    _SEMANTIC_ARTIFACT_EVIDENCE_CONTEXT_VALUE,
                )
                or not HybridRetriever._requires_present_value(
                    required.get("artifact_revision")
                )
                or context.get("artifact_revision")):
            return False
        if not HybridRetriever._binding_semantics_complete(binding, required):
            return False
        for raw_key, raw_expected in required.items():
            if raw_key == "artifact_revision":
                continue
            key = "tool_version" if raw_key == "version" else str(raw_key)
            actual = context.get(key, set()) or context.get(str(raw_key), set())
            if not HybridRetriever._constraint_matches(raw_expected, actual):
                return False
        return HybridRetriever._binding_condition_applicable(
            binding, condition or {}, context, targets,
        )

    @staticmethod
    def _binding_condition_applicable(
        binding: Any, condition: Mapping[str, Any],
        context: Mapping[str, set[str]], targets: Mapping[str, set[str]],
    ) -> bool:
        """Evaluate every rule premise against one target-instance context.

        Direct premises require exactly one actual value.  A missing premise
        can be derived only through the closed target mapping audited by the
        knowledge loader, and each source contract rechecks the typed witness
        from this same context.  Ambiguous, missing, or conflicting values are
        never resolved by picking a convenient member of a set.
        """
        if not isinstance(condition, Mapping):
            return False
        if str(binding.target) not in targets.get(str(binding.target_kind), set()):
            return False
        for raw_key, expected in condition.items():
            key = str(raw_key)
            actual = context.get(key, set())
            if actual:
                if len(actual) != 1 or not HybridRetriever._constraint_matches(
                    expected, actual,
                ):
                    return False
                continue
            source = target_derived_condition_source(binding, key, expected)
            if source is None or not HybridRetriever._target_condition_witnessed(
                binding, key, source, context,
            ):
                return False
        return True

    @staticmethod
    def _target_condition_witnessed(
        binding: Any, key: str, source: str,
        context: Mapping[str, set[str]],
    ) -> bool:
        def singleton(name: str) -> bool:
            return len(context.get(name, set())) == 1

        if source == "hlsgraph.target.directive_instance.v1":
            if not all(singleton(name) for name in (
                "directive_instance_id", "scope_id", "scope_kind",
            )):
                return False
            declaration = context.get(_DIRECTIVE_SOURCE_CONTEXT_KEY, set()) == {
                _DIRECTIVE_SOURCE_CONTEXT_VALUE,
            }
            dependence = context.get(_DEPENDENCE_OPERAND_CONTEXT_KEY, set()) == {
                _DEPENDENCE_OPERAND_CONTEXT_VALUE,
            }
            return declaration or dependence
        if source == "hlsgraph.target.qualified_artifact.v1":
            if str(binding.target) == "constraint.xdc":
                return (
                    context.get(_CONSTRAINT_INPUT_CONTEXT_KEY, set())
                    == {_CONSTRAINT_INPUT_CONTEXT_VALUE}
                    and all(singleton(name) for name in (
                        "artifact_sha256", "constraint_hash",
                        "constraint_artifact_identity",
                    ))
                )
            return (
                context.get(_TOOL_ARTIFACT_CONTEXT_KEY, set())
                == {_TOOL_ARTIFACT_CONTEXT_VALUE}
                and all(singleton(name) for name in (
                    "tool_artifact_identity", "tool_artifact_run_identity",
                ))
            )
        if source == "hlsgraph.target.qualified_observation.v1":
            allowed_artifacts = _CONDITION_OBSERVATION_ARTIFACTS.get(key)
            artifact_kinds = context.get(_OBSERVATION_ARTIFACT_KIND_CONTEXT_KEY, set())
            return (
                allowed_artifacts is not None
                and len(artifact_kinds) == 1
                and artifact_kinds <= allowed_artifacts
                and context.get(_OBSERVATION_EVIDENCE_CONTEXT_KEY, set())
                == {_OBSERVATION_EVIDENCE_CONTEXT_VALUE}
                and context.get("snapshot_association", set()) == {"verified"}
                and all(singleton(name) for name in (
                    "stage", "observation_instance_id",
                    "observation_artifact_identity", "observation_run_identity",
                ))
            )
        if source == "hlsgraph.target.qualified_gate.v1":
            if (context.get(_GATE_EVIDENCE_CONTEXT_KEY, set())
                    != {_GATE_EVIDENCE_CONTEXT_VALUE}
                    or context.get("snapshot_association", set()) != {"verified"}
                    or not singleton("stage")):
                return False
            identities = {
                "csim_result_present": (
                    "verification_observation_identity",
                    "verification_report_identity",
                ),
                "cosim_result_present": (
                    "verification_observation_identity",
                    "verification_report_identity",
                ),
                "utilization_report_present": (
                    "utilization_report_identity", "target_profile_hash",
                    "target_device_identity", "capacity_identity",
                ),
                "timing_gate_requested": (
                    "routed_design_identity", "timing_report_identity",
                    "constraint_hash", "constraint_artifact_identity",
                ),
            }.get(key)
            return identities is not None and all(singleton(name) for name in identities)
        return False

    @staticmethod
    def _constraint_mentions(constraint: Any, expected: str) -> bool:
        folded = canonical_context_scalar(expected)
        if isinstance(constraint, (str, bool)):
            return canonical_context_scalar(constraint) == folded
        if isinstance(constraint, (list, tuple)):
            return False
        if isinstance(constraint, Mapping):
            values: list[Any] = []
            if "equals" in constraint:
                values.append(constraint["equals"])
            if isinstance(constraint.get("one_of"), (list, tuple)):
                values.extend(constraint["one_of"])
            return folded in {
                canonical_context_scalar(item) for item in values
            }
        return False

    @staticmethod
    def _requires_present_value(constraint: Any) -> bool:
        return isinstance(constraint, Mapping) and constraint.get("required") is True

    @staticmethod
    def _binding_context_integrity(
        required: Mapping[str, Any], context: Mapping[str, set[str]],
    ) -> bool:
        """Recheck relationships that scalar binding constraints cannot express."""
        if HybridRetriever._constraint_mentions(
            required.get(_PORT_OWNERSHIP_CONTEXT_KEY),
            _PORT_OWNERSHIP_CONTEXT_VALUE,
        ):
            singleton_keys = (
                "scope_id", "port_id", "port_owner_id",
                "configured_component_id", "port_ownership_identity",
            )
            if any(len(context.get(key, set())) != 1 for key in singleton_keys):
                return False
            if (context.get(_PORT_OWNERSHIP_CONTEXT_KEY, set())
                    != {_PORT_OWNERSHIP_CONTEXT_VALUE}
                    or context["scope_id"] != context["port_id"]
                    or context["port_owner_id"]
                    != context["configured_component_id"]
                    or not re.fullmatch(
                        r"[0-9a-f]{64}",
                        next(iter(context["port_ownership_identity"])),
                    )):
                return False
        if not HybridRetriever._constraint_mentions(
            required.get(_SEMANTIC_ARTIFACT_EVIDENCE_CONTEXT_KEY),
            _SEMANTIC_ARTIFACT_EVIDENCE_CONTEXT_VALUE,
        ):
            return True
        singleton_keys = (
            "semantic_attestation_identity", "evidence_origin_identity",
            "artifact_identity", "artifact_sha256", "artifact_revision",
            "extractor_identity", "extractor_name", "extractor_version",
            "adapter_contract", "adapter_version", "extraction_manifest_identity",
        )
        if any(len(context.get(key, set())) != 1 for key in singleton_keys):
            return False
        artifact_id = next(iter(context["artifact_identity"]))
        artifact_sha256 = next(iter(context["artifact_sha256"]))
        artifact_revision = next(iter(context["artifact_revision"]))
        origin = next(iter(context["evidence_origin_identity"]))
        attestation_id = next(iter(context["semantic_attestation_identity"]))
        extractor_identity = next(iter(context["extractor_identity"]))
        aggregate_required = HybridRetriever._constraint_mentions(
            required.get(_AGGREGATE_EVIDENCE_CONTEXT_KEY),
            _AGGREGATE_EVIDENCE_CONTEXT_VALUE,
        )
        if (not re.fullmatch(r"[0-9a-f]{64}", artifact_sha256)
                or artifact_revision != f"sha256:{artifact_sha256}"
                or not re.fullmatch(r"[0-9a-f]{64}", extractor_identity)
                or origin == artifact_id):
            return False
        if aggregate_required:
            aggregate_keys = (
                "aggregate_evidence_identity",
                "aggregate_semantic_attestation_identity",
                "aggregate_source_artifact_identity",
            )
            if any(len(context.get(key, set())) != 1 for key in aggregate_keys):
                return False
            aggregate_id = next(iter(context["aggregate_evidence_identity"]))
            if (origin != aggregate_id
                    or aggregate_id in {artifact_id, attestation_id}
                    or context["aggregate_semantic_attestation_identity"]
                    != {attestation_id}
                    or context["aggregate_source_artifact_identity"]
                    != {artifact_id}):
                return False
        elif origin != attestation_id:
            return False
        native_identity = context.get("native_ir_artifact_identity")
        if native_identity and native_identity != {artifact_id}:
            return False
        return True

    @staticmethod
    def _binding_semantics_complete(binding: Any, required: Mapping[str, Any]) -> bool:
        """Reject an under-qualified binding before it can select guidance.

        This guard is deliberately stricter than generic rule matching.  The
        public 2024.2 AMD pack must pin the tool build and stage; workload- and
        activity-dependent meanings additionally require their dynamic scope.
        Static bindings explicitly declare that status in metadata so omission
        cannot accidentally be interpreted as universally applicable.
        """
        # Arrays are data values in other public contracts.  Binding
        # alternatives must use the explicit ``{"one_of": [...]}`` operator
        # so generic and set-valued retrieval matching cannot diverge.
        if any(isinstance(value, (list, tuple)) for value in required.values()):
            return False

        vendor = required.get("vendor")
        tool = required.get("tool")
        amd_tool = (
            HybridRetriever._constraint_mentions(vendor, "amd")
            and any(HybridRetriever._constraint_mentions(tool, item)
                    for item in ("vitis_hls", "vivado"))
        )
        if amd_tool and not HybridRetriever._constraint_mentions(
            required.get("tool_version"), "2024.2",
        ):
            return False

        target_kind = str(binding.target_kind)
        target = str(binding.target)
        directive_target = (
            target_kind == "directive_kind"
            or (target_kind == "predicate" and target.startswith("directive."))
        )
        if amd_tool and directive_target:
            # Every directive rule applies to one already-resolved declaration
            # or tool observation.  A manifest-level tool/version match is
            # never enough to bind scoped directive guidance.
            if not all(HybridRetriever._requires_present_value(required.get(key))
                       for key in ("directive_instance_id", "scope_id")):
                return False
            if "scope_kind" not in required:
                return False
            if not any(HybridRetriever._constraint_mentions(
                    required.get("scope_resolution"), resolution,
            ) for resolution in ("source_ast", "external_exact")):
                return False
            if target_kind == "directive_kind":
                roles = {
                    "DATAFLOW": ("function_id", "loop_id"),
                    "PIPELINE": ("function_id", "loop_id"),
                    "UNROLL": ("loop_id",),
                    "ARRAY_PARTITION": ("variable_id",),
                    "INTERFACE": ("port_id",),
                    "STREAM": ("variable_id",),
                    "DEPENDENCE": ("function_id", "loop_id", "variable_id"),
                    "LOOP_TRIPCOUNT": ("loop_id",),
                    "INLINE": ("function_id",),
                }.get(target.upper(), ())
                if not roles or not any(
                    HybridRetriever._requires_present_value(required.get(role))
                    for role in roles
                ):
                    return False
                if target.upper() == "DEPENDENCE":
                    if (_DIRECTIVE_OPERAND_CONTEXT_KEY in required
                            or not HybridRetriever._requires_present_value(
                            required.get("variable_id")
                        ) or not any(HybridRetriever._requires_present_value(
                            required.get(role)
                        ) for role in ("function_id", "loop_id"))
                            or not HybridRetriever._constraint_mentions(
                                required.get(_DEPENDENCE_OPERAND_CONTEXT_KEY),
                                _DEPENDENCE_OPERAND_CONTEXT_VALUE,
                            )
                            or not HybridRetriever._requires_present_value(
                                required.get("directive_operand_identity")
                            )):
                        return False
                else:
                    if (not HybridRetriever._constraint_mentions(
                            required.get(_DIRECTIVE_SOURCE_CONTEXT_KEY),
                            _DIRECTIVE_SOURCE_CONTEXT_VALUE,
                        ) or not HybridRetriever._requires_present_value(
                            required.get("directive_source_identity")
                        )):
                        return False
                if target.upper() in {"ARRAY_PARTITION", "INTERFACE", "STREAM"}:
                    if (not HybridRetriever._constraint_mentions(
                            required.get(_DIRECTIVE_OPERAND_CONTEXT_KEY),
                            _DIRECTIVE_OPERAND_CONTEXT_VALUE,
                        ) or not HybridRetriever._requires_present_value(
                            required.get("directive_operand_identity")
                        )):
                        return False
                if target.upper() == "INTERFACE":
                    if (not HybridRetriever._constraint_mentions(
                            required.get(_PORT_OWNERSHIP_CONTEXT_KEY),
                            _PORT_OWNERSHIP_CONTEXT_VALUE,
                        ) or not all(HybridRetriever._requires_present_value(
                            required.get(key)
                        ) for key in (
                            "port_owner_id", "configured_component_id",
                            "port_ownership_identity",
                        ))):
                        return False

            if (target_kind == "predicate"
                    and not HybridRetriever._constraint_mentions(
                        required.get(_REQUESTED_DIRECTIVE_CONTEXT_KEY), True,
                    )):
                return False
            if (target_kind == "predicate" and target in {
                    "directive.tool_status", "directive.reported_requested",
                    "directive.tool_effective", "directive.achieved",
            } and not HybridRetriever._constraint_mentions(
                required.get(_OBSERVATION_ARTIFACT_KIND_CONTEXT_KEY),
                "amd.vitis.directive_status",
            )):
                return False

        if (amd_tool and target_kind == "predicate"
                and target in _AMD_TYPED_OBSERVATION_PREDICATES):
            if (not HybridRetriever._constraint_mentions(
                    required.get(_OBSERVATION_EVIDENCE_CONTEXT_KEY),
                    _OBSERVATION_EVIDENCE_CONTEXT_VALUE,
                ) or not HybridRetriever._constraint_mentions(
                    required.get("snapshot_association"), "verified",
                ) or not all(HybridRetriever._requires_present_value(
                    required.get(key)
                ) for key in (
                    "observation_instance_id", "observation_artifact_identity",
                    "observation_run_identity",
                ))):
                return False

        csynth_estimate_binding = (
            amd_tool and target_kind == "predicate"
            and str(binding.knowledge_rule_id).endswith(":qor.csynth_is_estimate")
        )
        if csynth_estimate_binding and (
                target not in _CSYNTH_ESTIMATE_PREDICATES
                or not HybridRetriever._constraint_mentions(
                    required.get(_OBSERVATION_ARTIFACT_KIND_CONTEXT_KEY),
                    "amd.vitis.csynth_xml",
                )):
            return False

        open_ir = any(
            HybridRetriever._constraint_mentions(required.get("ir"), item)
            for item in ("mlir", "llvm")
        )
        if open_ir:
            metadata = getattr(binding, "metadata", {})
            spec_family = (
                "circt.handshake"
                if target_kind == "relation_kind" and target == "handshake.dataflow"
                else "mlir" if HybridRetriever._constraint_mentions(
                    required.get("ir"), "mlir",
                ) else "llvm"
            )
            spec_revision, spec_contract, _artifact_kinds = (
                _LANGUAGE_SPEC_CONTRACTS[spec_family]
            )
            if (not isinstance(metadata, Mapping)
                    or metadata.get("current_instance_only") is not True
                    or metadata.get("artifact_revision_not_inferred") is not True
                    or metadata.get("language_spec_revision_not_inferred") is not True
                    or not HybridRetriever._requires_present_value(
                        required.get("artifact_revision")
                    )
                    or not HybridRetriever._constraint_mentions(
                        required.get(_SEMANTIC_ARTIFACT_EVIDENCE_CONTEXT_KEY),
                        _SEMANTIC_ARTIFACT_EVIDENCE_CONTEXT_VALUE,
                    )
                    or not HybridRetriever._constraint_mentions(
                        required.get("semantic_attestation_contract"),
                        _SEMANTIC_ATTESTATION_CONTRACT,
                    )
                    or not HybridRetriever._constraint_mentions(
                        required.get("artifact_byte_closure"),
                        _SEMANTIC_ARTIFACT_BYTE_CLOSURE,
                    )
                    or not all(HybridRetriever._requires_present_value(
                        required.get(key)
                    ) for key in (
                        "semantic_attestation_identity", "artifact_identity",
                        "artifact_sha256", "evidence_origin_identity",
                        "extraction_manifest_identity", "extractor_identity",
                        "extractor_name", "extractor_version",
                        "adapter_contract", "adapter_version",
                    ))
                    or not HybridRetriever._constraint_mentions(
                        required.get("language_spec_family"), spec_family,
                    )
                    or not HybridRetriever._constraint_mentions(
                        required.get("language_spec_revision"), spec_revision,
                    )
                    or not HybridRetriever._constraint_mentions(
                        required.get("language_spec_compatibility_contract"),
                        spec_contract,
                    )
                    or not HybridRetriever._constraint_mentions(
                        required.get("language_spec_revision_source"),
                        _LANGUAGE_SPEC_REVISION_SOURCE,
                    )):
                return False
            instance_key = {
                "entity_kind": "entity_instance_id",
                "relation_kind": "relation_instance_id",
                "predicate": "derivation_instance_id",
                "diagnostic_code": "diagnostic_instance_id",
            }.get(target_kind)
            if (instance_key is None
                    or not HybridRetriever._requires_present_value(
                        required.get(instance_key)
                    )):
                return False
            if (target_kind == "entity_kind"
                    and not HybridRetriever._constraint_mentions(
                        required.get("entity_kind"), str(binding.target)
                    )):
                return False
            if (target_kind == "relation_kind"
                    and not HybridRetriever._constraint_mentions(
                        required.get("relation_kind"), str(binding.target)
                    )):
                return False
            if target_kind == "relation_kind" and target == "cross.maps_to":
                location_constraint = required.get("location_kind")
                location_choices = (
                    location_constraint.get("one_of")
                    if isinstance(location_constraint, Mapping) else None
                )
                target_constraint = required.get("target_entity_kind")
                target_choices = (
                    target_constraint.get("one_of")
                    if isinstance(target_constraint, Mapping) else None
                )
                if (not isinstance(location_choices, (list, tuple))
                        or not location_choices
                        or not {str(item) for item in location_choices}.issubset(
                            _CONCRETE_MLIR_MAPPING_LOCATION_KINDS
                        )
                        or not isinstance(target_choices, (list, tuple))
                        or not target_choices
                        or not {str(item) for item in target_choices}.issubset(
                            _SOURCE_MAPPING_TARGET_KINDS
                        )
                        or not HybridRetriever._constraint_mentions(
                        required.get("source_entity_kind"), "ir.mlir.operation"
                    )
                        or not HybridRetriever._constraint_mentions(
                            required.get("mapping_kind"), "mlir.location"
                        )
                        or not HybridRetriever._constraint_mentions(
                            required.get("target_entity_stage"), "ast"
                        )
                        or not HybridRetriever._constraint_mentions(
                            required.get("hardware_topology"), "false"
                        )
                        or not HybridRetriever._constraint_mentions(
                            required.get("typed_mlir_location_present"), "true"
                        )
                        or not HybridRetriever._constraint_mentions(
                            required.get("mapping_provenance"),
                            "mlir.location_anchor",
                        )
                        or not HybridRetriever._constraint_mentions(
                            required.get("mapping_resolution"),
                            "unique_exact",
                        )
                        or not HybridRetriever._constraint_mentions(
                            required.get("mapping_resolution_contract"),
                            _MLIR_LOCATION_RESOLUTION_CONTRACT,
                        )
                        or not HybridRetriever._constraint_mentions(
                            required.get(
                                "unique_mlir_location_mapping_resolved"
                            ), "true",
                        )
                        or not HybridRetriever._constraint_mentions(
                            required.get("source_anchor_identity_contract"),
                            _SOURCE_ANCHOR_IDENTITY_CONTRACT,
                        )
                        or not all(HybridRetriever._requires_present_value(
                            required.get(key)
                        ) for key in (
                            "typed_source_anchor_identity",
                            "resolved_target_anchor_identity",
                            "resolved_target_id",
                        ))):
                    return False
            if target_kind == "relation_kind" and target == "handshake.dataflow":
                if (metadata.get("native_ir_evidence_only") is not True
                        or not HybridRetriever._constraint_mentions(
                            required.get("source_entity_kind"),
                            "ir.mlir.operation",
                        )
                        or not HybridRetriever._constraint_mentions(
                            required.get("target_entity_kind"),
                            "ir.mlir.operation",
                        )
                        or not HybridRetriever._constraint_mentions(
                            required.get("hardware_topology"), "false",
                        )
                        or not HybridRetriever._constraint_mentions(
                            required.get("handshake_operation_present"), "true",
                        )
                        or not HybridRetriever._constraint_mentions(
                            required.get("native_ir_evidence"), "true",
                        )
                        or not HybridRetriever._constraint_mentions(
                            required.get("native_ir_evidence_contract"),
                            _NATIVE_MLIR_SSA_EVIDENCE_CONTRACT,
                        )
                        or not HybridRetriever._constraint_mentions(
                            required.get("native_ir_relation_provenance"),
                            "mlir.ssa_def_use",
                        )
                        or not HybridRetriever._requires_present_value(
                            required.get("native_ir_artifact_identity")
                        )):
                    return False
            aggregate_feature = (
                target_kind == "predicate" and target in {
                    "feature.operation_histogram", "feature.index_histogram",
                    "feature.bitwidth", "feature.memory_access",
                }
            )
            if aggregate_feature and (
                metadata.get("aggregate_evidence_recomputed") is not True
                or not HybridRetriever._constraint_mentions(
                    required.get(_AGGREGATE_EVIDENCE_CONTEXT_KEY),
                    _AGGREGATE_EVIDENCE_CONTEXT_VALUE,
                )
                or not HybridRetriever._constraint_mentions(
                    required.get("aggregate_evidence_contract"),
                    _AGGREGATE_EVIDENCE_CONTRACT,
                )
                or not all(HybridRetriever._requires_present_value(
                    required.get(key)
                ) for key in (
                    "artifact_identity", "artifact_sha256",
                    "evidence_origin_identity", "aggregate_evidence_identity",
                    "aggregate_semantic_attestation_identity",
                    "aggregate_source_artifact_identity",
                ))
            ):
                return False
            if target_kind == "predicate" and target == "feature.operation_histogram":
                if (not HybridRetriever._constraint_mentions(
                        required.get("derivation_algorithm"),
                        "hlsgraph.static.operation_histogram",
                    )
                        or not HybridRetriever._constraint_mentions(
                            required.get("derivation_algorithm_version"), "1"
                        )
                        or not HybridRetriever._constraint_mentions(
                            required.get("operation_histogram_provenance"),
                            "typed_ir_entity_evidence.v1",
                        )
                        or not HybridRetriever._constraint_mentions(
                            required.get("operation_histogram_domain_complete"),
                            "true",
                        )):
                    return False
                if HybridRetriever._constraint_mentions(required.get("ir"), "mlir"):
                    if (not HybridRetriever._constraint_mentions(
                            required.get("operation_histogram_schema"),
                            "mlir.dialect_qualified_opcode_histogram.v1",
                        )
                            or not HybridRetriever._constraint_mentions(
                                required.get(
                                    "dialect_qualified_operation_histogram_present"
                                ), "true",
                            )):
                        return False
                elif HybridRetriever._constraint_mentions(required.get("ir"), "llvm"):
                    if (not HybridRetriever._constraint_mentions(
                            required.get("operation_histogram_schema"),
                            "llvm.opcode_histogram.v1",
                        )
                            or not HybridRetriever._constraint_mentions(
                                required.get(
                                    "opcode_qualified_operation_histogram_present"
                                ), "true",
                            )):
                        return False
                else:
                    return False
            if target_kind == "predicate" and target == "feature.index_histogram":
                if (not HybridRetriever._constraint_mentions(
                        required.get("derivation_algorithm"),
                        "hlsgraph.static.index_histogram",
                    )
                        or not HybridRetriever._constraint_mentions(
                            required.get("derivation_algorithm_version"), "1"
                        )
                        or not HybridRetriever._constraint_mentions(
                            required.get("index_histogram_schema"),
                            "llvm.explicit_index_operand_kind_histogram.v1",
                        )
                        or not HybridRetriever._constraint_mentions(
                            required.get("index_histogram_provenance"),
                            "typed_ir_entity_evidence.v1",
                        )
                        or not HybridRetriever._constraint_mentions(
                            required.get("index_operand_definition"),
                            "llvm.gep_extract_insert_explicit_operand.v1",
                        )
                        or not HybridRetriever._constraint_mentions(
                            required.get("index_histogram_domain_complete"),
                            "true",
                        )
                        or not HybridRetriever._constraint_mentions(
                            required.get("typed_index_histogram_present"),
                            "true",
                        )):
                    return False
            if target_kind == "predicate" and target == "feature.bitwidth":
                if (not HybridRetriever._constraint_mentions(
                        required.get("derivation_algorithm"),
                        "hlsgraph.static.bitwidth",
                    )
                        or not HybridRetriever._constraint_mentions(
                            required.get("derivation_algorithm_version"), "1"
                        )
                        or not HybridRetriever._constraint_mentions(
                            required.get("bitwidth_schema"),
                            "llvm.explicit_integer_width_occurrence_histogram.v1",
                        )
                        or not HybridRetriever._constraint_mentions(
                            required.get("bitwidth_provenance"),
                            "typed_ir_entity_evidence.v1",
                        )
                        or not HybridRetriever._constraint_mentions(
                            required.get("bitwidth_definition"),
                            "llvm.explicit_integer_type_occurrence.v1",
                        )
                        or not HybridRetriever._constraint_mentions(
                            required.get("bitwidth_domain_complete"), "true",
                        )
                        or not HybridRetriever._constraint_mentions(
                            required.get("typed_bitwidth_histogram_present"),
                            "true",
                        )):
                    return False
            if target_kind == "predicate" and target == "feature.memory_access":
                if (not HybridRetriever._constraint_mentions(
                        required.get("derivation_algorithm"),
                        "hlsgraph.static.memory_access",
                    )
                        or not HybridRetriever._constraint_mentions(
                            required.get("derivation_algorithm_version"), "1"
                        )
                        or not HybridRetriever._constraint_mentions(
                            required.get("memory_access_schema"),
                            "llvm.memory_access_kind_histogram.v1",
                        )
                        or not HybridRetriever._constraint_mentions(
                            required.get("memory_access_provenance"),
                            "typed_ir_entity_evidence.v1",
                        )
                        or not HybridRetriever._constraint_mentions(
                            required.get("memory_access_opcode_definition"),
                            "llvm.load_store_gep_atomic_fence.v1",
                        )
                        or not HybridRetriever._constraint_mentions(
                            required.get("memory_access_domain_complete"),
                            "true",
                        )
                        or not HybridRetriever._constraint_mentions(
                            required.get(
                                "typed_memory_access_histogram_present"
                            ), "true",
                        )):
                    return False

        if (amd_tool and target_kind == "artifact_kind"
                and target == "constraint.xdc"):
            if (not HybridRetriever._constraint_mentions(
                    required.get("snapshot_association"), "verified")
                    or not HybridRetriever._constraint_mentions(
                        required.get(_CONSTRAINT_INPUT_CONTEXT_KEY),
                        _CONSTRAINT_INPUT_CONTEXT_VALUE,
                    )
                    or not HybridRetriever._requires_present_value(
                        required.get("artifact_sha256")
                    )
                    or not HybridRetriever._requires_present_value(
                        required.get("constraint_hash")
                    )
                    or not HybridRetriever._requires_present_value(
                        required.get("constraint_artifact_identity")
                    )):
                return False

        if (amd_tool and target_kind == "artifact_kind"
                and target in _TOOL_ARTIFACT_STAGE_POLICY):
            if (not HybridRetriever._constraint_mentions(
                    required.get("snapshot_association"), "verified"
                ) or not HybridRetriever._constraint_mentions(
                    required.get(_TOOL_ARTIFACT_CONTEXT_KEY),
                    _TOOL_ARTIFACT_CONTEXT_VALUE,
                ) or not all(HybridRetriever._requires_present_value(
                    required.get(key)
                ) for key in (
                    "tool_artifact_identity", "tool_artifact_run_identity",
                ))):
                return False

        if amd_tool and target_kind == "gate_kind":
            if (not HybridRetriever._constraint_mentions(
                    required.get("snapshot_association"), "verified")
                    or not HybridRetriever._constraint_mentions(
                        required.get(_GATE_EVIDENCE_CONTEXT_KEY),
                        _GATE_EVIDENCE_CONTEXT_VALUE,
                    )
                    or target not in {
                        "correctness", "resource_fits", "post_route_timing",
                    }):
                return False
            gate_identity_fields = {
                "correctness": (
                    "verification_observation_identity",
                    "verification_report_identity",
                ),
                "resource_fits": (
                    "utilization_report_identity", "target_profile_hash",
                    "target_device_identity", "capacity_identity",
                ),
                "post_route_timing": (
                    "routed_design_identity", "timing_report_identity",
                    "constraint_hash", "constraint_artifact_identity",
                ),
            }[target]
            if not all(HybridRetriever._requires_present_value(required.get(key))
                       for key in gate_identity_fields):
                return False
            if (target == "resource_fits"
                    and not HybridRetriever._constraint_mentions(
                        required.get("complete_post_route_utilization"), "true"
                    )):
                return False

        stage_sensitive = str(binding.target_kind) in {
            "predicate", "artifact_kind", "gate_kind", "diagnostic_code",
            "entity_kind", "relation_kind", "directive_kind",
        }
        if stage_sensitive and "stage" not in required:
            return False

        target = str(binding.target).casefold()
        workload_declared = any(
            HybridRetriever._requires_present_value(required.get(key))
            for key in ("workload_id", "testcase_id")
        )
        workload_dynamic = workload_declared or any(token in target for token in (
            "fifo", "stall", "csim", "cosim", "runtime", "occupancy",
            "deadlock", "workload",
        ))
        activity_declared = HybridRetriever._requires_present_value(
            required.get("activity_source")
        )
        activity_dynamic = activity_declared or any(
            token in target for token in ("activity", "power")
        )
        waiver_declared = HybridRetriever._requires_present_value(
            required.get("waiver_mode")
        )
        waiver_dynamic = waiver_declared or any(
            token in target for token in ("drc", "cdc")
        )
        if workload_dynamic:
            if not workload_declared:
                return False
        elif activity_dynamic:
            if not activity_declared:
                return False
        elif waiver_dynamic:
            if not waiver_declared:
                return False
        else:
            metadata = getattr(binding, "metadata", {})
            if (not isinstance(metadata, Mapping)
                    or metadata.get("dynamic_scope") != "static"):
                return False
        return True

    def _applicability_context(self, spec: RetrievalSpec) -> dict[str, set[str]]:
        context, _toolchains = self._manifest_context()
        for key, value in spec.applicability.items():
            context.setdefault(key, set()).add(value.casefold())
        return context

    @staticmethod
    def _context_matches_request(
        context: Mapping[str, set[str]], spec: RetrievalSpec,
    ) -> bool:
        for raw_key, requested in spec.applicability.items():
            key = "tool_version" if raw_key == "version" else raw_key
            actual = context.get(key, set()) or context.get(raw_key, set())
            if requested.casefold() not in actual:
                return False
        return True

    @staticmethod
    def _rule_applicable(applicability: Mapping[str, Any],
                         context: Mapping[str, set[str]]) -> tuple[bool, str]:
        for raw_key, raw_expected in applicability.items():
            key = "tool_version" if raw_key == "version" else str(raw_key)
            actual = context.get(key, set()) or context.get(str(raw_key), set())
            if not HybridRetriever._constraint_matches(raw_expected, actual):
                return (False, "context_mismatch")
        return (True, "applicable")

    @staticmethod
    def _constraint_matches(constraint: Any, actual: set[str]) -> bool:
        if constraint in (None, "*"):
            return True
        canonical_actual = {canonical_context_scalar(item) for item in actual}
        if not canonical_actual:
            return False
        if isinstance(constraint, (list, tuple)):
            return False
        if isinstance(constraint, Mapping):
            allowed = {
                "equals", "one_of", "min_version", "max_version", "required",
            }
            # A typo or future operator must never weaken applicability.  Pack
            # loading and generic knowledge filtering enforce the same set;
            # retrieval repeats the guard at the final instance-match boundary
            # because model objects remain intentionally mutable.
            if set(constraint) - allowed:
                return False
            if constraint.get("required") not in (None, True):
                return False
            candidates = set(canonical_actual)
            if "equals" in constraint:
                candidates &= {canonical_context_scalar(constraint["equals"])}
            if "one_of" in constraint:
                choices = constraint["one_of"]
                # Active-session conditions are recursively frozen, so their
                # reviewed ``one_of`` arrays arrive as tuples.  Caller-owned
                # pack objects still require lists at load time.
                if not isinstance(choices, (list, tuple)):
                    return False
                candidates &= {
                    canonical_context_scalar(item) for item in choices
                }
            if not candidates:
                return False

            def version_key(value: str) -> tuple[tuple[int, Any], ...]:
                parts = re.findall(r"\d+|[A-Za-z]+", value)
                return tuple((0, int(part)) if part.isdigit()
                             else (1, part.casefold()) for part in parts)

            if "min_version" in constraint:
                minimum = version_key(str(constraint["min_version"]))
                candidates = {item for item in candidates if version_key(item) >= minimum}
            if "max_version" in constraint:
                maximum = version_key(str(constraint["max_version"]))
                candidates = {item for item in candidates if version_key(item) <= maximum}
            return bool(candidates)
        return canonical_context_scalar(constraint) in canonical_actual

    @staticmethod
    def _view_graph(graph: CanonicalGraph, view: str) -> CanonicalGraph:
        if view == "evidence":
            return graph
        projected = CanonicalGraph(
            snapshot_id=graph.snapshot_id, metadata=dict(graph.metadata),
            schema_version=graph.schema_version,
        )
        for entity in graph.entities.values():
            if entity.kind.startswith(("ir.", "source.", "software.", "ast.")):
                continue
            projected.add_entity(entity)
        for relation in graph.relations.values():
            if relation.src not in projected.entities or relation.dst not in projected.entities:
                continue
            if relation.kind in _ZERO_WEIGHT_RELATIONS:
                continue
            if relation.kind == "hls.contains" and relation.stage in {"source", "ast"}:
                continue
            if relation.kind == "hls.streams_to" and (
                relation.stage in {"source", "ast"}
                or str(relation.authority) == AuthorityClass.STATIC_FACT.value
            ):
                continue
            if relation.attrs.get("hardware_topology") is False:
                continue
            if relation.attrs.get("hardware_instance") is False:
                continue
            projected.add_relation(relation)
        connected = {endpoint for relation in projected.relations.values()
                     for endpoint in (relation.src, relation.dst)}
        for entity_id, entity in list(projected.entities.items()):
            if (entity_id not in connected and entity.kind != "hls.kernel"
                    and entity.stage in {"source", "ast"}):
                del projected.entities[entity_id]
        return projected

    @staticmethod
    def _scope_ids(graph: CanonicalGraph, scope_id: str | None) -> set[str]:
        if scope_id is None:
            return set(graph.entities)
        if scope_id not in graph.entities:
            raise KeyError(scope_id)
        keep = {scope_id}
        queue = deque([scope_id])
        outgoing: dict[str, list[str]] = defaultdict(list)
        for relation in graph.relations.values():
            if relation.kind in {"hls.contains", "ir.contains"}:
                outgoing[relation.src].append(relation.dst)
        while queue:
            current = queue.popleft()
            for target in sorted(outgoing.get(current, ())):
                if target not in keep:
                    keep.add(target)
                    queue.append(target)
        return keep

    @staticmethod
    def _relation_weight(kind: str) -> float:
        if kind in _ZERO_WEIGHT_RELATIONS or kind.startswith("software."):
            return 0.0
        if kind in _RELATION_WEIGHTS:
            return _RELATION_WEIGHTS[kind]
        if kind.startswith(("hls.", "handshake.", "cross.", "memory.", "interface.")):
            return 0.5
        return 0.0

    def _typed_ppr(self, graph: CanonicalGraph, allowed_ids: set[str],
                   raw_seeds: Mapping[str, float]) -> tuple[dict[str, float], set[str]]:
        seeds = {key: value for key, value in raw_seeds.items()
                 if key in allowed_ids and value > 0}
        if not seeds:
            return ({}, set())
        total = sum(seeds.values())
        restart = {key: value / total for key, value in seeds.items()}
        adjacency: dict[str, list[tuple[str, float, str]]] = defaultdict(list)
        for relation in sorted(graph.relations.values(), key=lambda item: item.id):
            if relation.src not in allowed_ids or relation.dst not in allowed_ids:
                continue
            weight = self._relation_weight(relation.kind)
            if weight > 0:
                adjacency[relation.src].append((relation.dst, weight, relation.id))

        selected = set(seeds)
        selected_relations: set[str] = set()
        queue = deque((item, 0) for item in sorted(seeds, key=lambda key: (-seeds[key], key)))
        while queue and len(selected) < 200:
            current, depth = queue.popleft()
            if depth >= 3:
                continue
            for target, _weight, relation_id in adjacency.get(current, ()):
                selected_relations.add(relation_id)
                if target not in selected:
                    selected.add(target)
                    queue.append((target, depth + 1))
                    if len(selected) >= 200:
                        break

        scores = {item: restart.get(item, 0.0) for item in selected}
        alpha = 0.25
        for _ in range(25):
            next_scores = {item: alpha * restart.get(item, 0.0) for item in selected}
            dangling = 0.0
            for source in sorted(selected):
                edges = [(target, weight) for target, weight, _relation_id in adjacency.get(source, ())
                         if target in selected]
                if not edges:
                    dangling += scores.get(source, 0.0)
                    continue
                denominator = sum(weight for _target, weight in edges)
                for target, weight in edges:
                    next_scores[target] += (1.0 - alpha) * scores.get(source, 0.0) * weight / denominator
            if dangling:
                for target, weight in restart.items():
                    if target in next_scores:
                        next_scores[target] += (1.0 - alpha) * dangling * weight
            scores = next_scores
        return (scores, selected_relations)

    @staticmethod
    def _rrf(channel_scores: Mapping[str, Mapping[str, float]]) -> tuple[
        list[tuple[str, float]], dict[str, dict[str, float]]
    ]:
        totals: dict[str, float] = defaultdict(float)
        evidence: dict[str, dict[str, float]] = defaultdict(dict)
        for channel in sorted(channel_scores):
            weight = _CHANNEL_WEIGHTS.get(channel, 1.0)
            ordered = sorted(channel_scores[channel].items(), key=lambda item: (-item[1], item[0]))
            for rank, (key, raw_score) in enumerate(ordered, start=1):
                totals[key] += weight / (60.0 + rank)
                evidence[key][channel] = raw_score
        return (sorted(totals.items(), key=lambda item: (-item[1], item[0])), evidence)

    def _flow_spine(self, graph: CanonicalGraph, ppr: Mapping[str, float],
                    selected_relation_ids: set[str], facts: Sequence[RetrievalItem],
                    max_edges: int) -> list[dict[str, Any]]:
        starts = [item.entity_id for item in facts if item.entity_id in ppr]
        if not starts:
            return []
        current = starts[0]
        visited = {current}
        result: list[dict[str, Any]] = []
        for _ in range(max_edges):
            candidates = []
            for relation in graph.relations.values():
                if (relation.id not in selected_relation_ids or relation.src != current
                        or relation.dst in visited):
                    continue
                if (relation.attrs.get("hardware_topology") is False
                        or relation.attrs.get("hardware_instance") is False):
                    continue
                weight = self._relation_weight(relation.kind)
                if weight > 0:
                    candidates.append((ppr.get(relation.dst, 0.0), weight,
                                       relation.dst, relation))
            if not candidates:
                break
            _score, _weight, target, relation = sorted(
                candidates, key=lambda item: (-item[0], -item[1], item[2], item[3].id)
            )[0]
            result.append({
                "relation_id": relation.id,
                "kind": relation.kind,
                "src": relation.src,
                "dst": relation.dst,
                "stage": relation.stage,
                "authority_class": str(relation.authority),
                "completeness": str(relation.completeness),
            })
            current = target
            visited.add(current)
        return result

    @staticmethod
    def _citations(guidance: Sequence[RetrievalItem]) -> list[dict[str, Any]]:
        values: dict[str, dict[str, Any]] = {}
        for item in guidance:
            if not item.citation:
                continue
            key = stable_hash(item.citation)
            values[key] = item.citation
        return [values[key] for key in sorted(values)]

    @staticmethod
    def _apply_budget(result: RetrievalResult, budget: int) -> None:
        trace_compacted = False
        warnings_compacted = False

        def synchronize_private_flag() -> None:
            result.trace.private_snippets_returned = any(
                item.data.get("private_excerpt") is not None
                for item in (*result.facts, *result.guidance, *result.predictions)
            )

        def size() -> int:
            return len(json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True,
                                  separators=(",", ":")))

        synchronize_private_flag()
        result.trace.output_chars = size()
        while result.trace.output_chars > budget:
            if result.predictions:
                result.predictions.pop()
            elif result.guidance:
                result.guidance.pop()
                result.citations = HybridRetriever._citations(result.guidance)
            elif result.flow:
                result.flow.pop()
            elif result.ambiguities:
                result.ambiguities.pop()
            elif not trace_compacted:
                # At the caller-selected 1k floor, the full per-channel trace
                # can itself exceed the response budget after all answer items
                # have been removed.  Preserve the query/profile/graph hashes,
                # truth/privacy flags, total timing and candidate total while
                # explicitly marking the response truncated.
                candidate_total = sum(result.trace.candidate_counts.values())
                total_ms = result.trace.elapsed_ms.get("total", 0.0)
                result.trace.candidate_counts = {"channel_total": candidate_total}
                result.trace.elapsed_ms = {"total": total_ms}
                result.trace.adapter_fingerprints = []
                trace_compacted = True
            elif not warnings_compacted and result.warnings != ["output_truncated_to_budget"]:
                warning_hash = stable_hash(result.warnings)[:16]
                warning_count = len(result.warnings)
                rejected_count = sum(
                    "warning_rejected" in warning for warning in result.warnings
                )
                summary_kind = (
                    "adapter_warning_rejected" if rejected_count else "warnings_compacted"
                )
                result.warnings = [
                    "output_truncated_to_budget",
                    f"{summary_kind}:{warning_count}:{warning_hash}",
                ]
                warnings_compacted = True
            elif result.facts:
                # Canonical facts are the last answer plane trimmed.  Public
                # pack or local-sidecar growth is discarded first, preventing
                # guidance volume from changing which fact ranks survive a
                # fixed response budget.
                result.facts.pop()
                if not result.facts:
                    result.incomplete = True
                    result.confidence = "low"
            else:
                break
            result.trace.truncated = True
            if "output_truncated_to_budget" not in result.warnings:
                result.warnings.append("output_truncated_to_budget")
                result.warnings.sort()
            synchronize_private_flag()
            result.trace.output_chars = size()
        result.trace.output_chars = size()


__all__ = [
    "DEFAULT_RETRIEVAL_PROFILE", "HybridRetriever", "LocalKnowledgeRetrievalAdapter",
    "RetrievalAdapter", "RetrievalItem", "RetrievalResult", "RetrievalSpec",
    "SourceSnippetRetrievalAdapter",
    "RetrievalTrace", "default_retrieval_adapters", "normalize_terms",
]
