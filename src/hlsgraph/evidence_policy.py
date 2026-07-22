"""Versioned compatibility policy for run-backed tool evidence.

The policy is intentionally explicit.  An observation stage is not inferred from
an artifact filename or a convenient namespace prefix, and an unknown plugin
artifact is accepted only when it carries the complete extension contract below.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .model import (
    ArtifactRef,
    AuthorityClass,
    DesignSnapshot,
    ExecutionAttestation,
    ExecutionCommitReceipt,
    ExecutionDeclaredOutput,
    ExecutionOutputAttestation,
    Observation,
    ProjectManifest,
    ToolRun,
    json_ready,
    stable_hash,
)


TOOL_EVIDENCE_POLICY_VERSION = "hlsgraph.tool-evidence.v0.1"
TOOL_EVIDENCE_EXTENSION_METADATA_KEY = "hlsgraph_evidence"
REAL_TOOL_RUN_BACKENDS = frozenset({"runner.local", "runner.ssh"})
REAL_TOOL_RUN_AUTHORITIES = frozenset({
    AuthorityClass.TOOL_OBSERVATION.value,
    AuthorityClass.VERIFICATION_EVIDENCE.value,
    AuthorityClass.PHYSICAL_MEASUREMENT.value,
})
EXECUTION_ATTESTATION_PROTOCOL = "hlsgraph.execution_attestation.v1"
EXECUTION_ATTESTATION_VALIDATOR = "hlsgraph.stage_orchestrator.v1"
EXECUTION_RECEIPT_CONTRACT = "hlsgraph.execution_commit_receipt.v1"
EXECUTION_RECEIPT_VALIDATOR = "hlsgraph.ledger.execution_attestation.v1"
_DEFAULT_MAX_OUTPUT_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class ToolEvidenceStagePolicy:
    """Allowed producer stages and typed reports for one observation stage."""

    run_stages: frozenset[str]
    intrinsic_artifact_kinds: frozenset[str]
    stage_declared_artifact_kinds: frozenset[str] = frozenset()


_VIVADO_STAGE_DECLARED_KINDS = frozenset({
    "amd.vivado.timing_summary",
    "amd.vivado.utilization",
    "amd.vivado.physical_summary",
    "amd.vivado.qor_summary",
})


# Absence is deliberate: source/AST/MLIR/LLVM/unknown observations cannot be
# present ML truth, even if a caller attaches them to a real tool run.  The
# canonical run-stage aliases reflect the public orchestration vocabulary.
TOOL_EVIDENCE_STAGE_POLICY: Mapping[str, ToolEvidenceStagePolicy] = MappingProxyType({
    "csim": ToolEvidenceStagePolicy(
        run_stages=frozenset({"csim"}),
        intrinsic_artifact_kinds=frozenset({"amd.vitis.csim_result"}),
    ),
    "schedule": ToolEvidenceStagePolicy(
        run_stages=frozenset({"csynth", "schedule"}),
        intrinsic_artifact_kinds=frozenset({
            "amd.vitis.csynth_xml",
            "amd.vitis.csynth_report",
            "amd.vitis.schedule_json",
            "amd.vitis.directive_status",
        }),
    ),
    # ``csynth`` is a supported observation-stage extension used when an
    # adapter preserves the tool's native phase instead of projecting to
    # canonical ``schedule``.
    "csynth": ToolEvidenceStagePolicy(
        run_stages=frozenset({"csynth"}),
        intrinsic_artifact_kinds=frozenset({
            "amd.vitis.csynth_xml",
            "amd.vitis.csynth_report",
        }),
    ),
    "cosim": ToolEvidenceStagePolicy(
        run_stages=frozenset({"cosim", "rtl_cosim"}),
        intrinsic_artifact_kinds=frozenset({
            "amd.vitis.cosim_rpt",
            "amd.vitis.cosim_report",
            "amd.vitis.dataflow_profile",
        }),
    ),
    # Keep the native run spelling available to vendor-neutral adapters while
    # the canonical observation spelling remains ``cosim``.
    "rtl_cosim": ToolEvidenceStagePolicy(
        run_stages=frozenset({"cosim", "rtl_cosim"}),
        intrinsic_artifact_kinds=frozenset({
            "amd.vitis.cosim_rpt",
            "amd.vitis.cosim_report",
            "amd.vitis.dataflow_profile",
        }),
    ),
    "rtl": ToolEvidenceStagePolicy(
        run_stages=frozenset({"rtl", "rtl_export"}),
        intrinsic_artifact_kinds=frozenset({
            "amd.vitis.rtl_export",
            "amd.vitis.rtl_report",
        }),
    ),
    "post_synth": ToolEvidenceStagePolicy(
        run_stages=frozenset({"post_synth", "vivado_synth"}),
        intrinsic_artifact_kinds=frozenset(),
        stage_declared_artifact_kinds=_VIVADO_STAGE_DECLARED_KINDS,
    ),
    "post_place": ToolEvidenceStagePolicy(
        run_stages=frozenset({"post_place"}),
        intrinsic_artifact_kinds=frozenset(),
        stage_declared_artifact_kinds=_VIVADO_STAGE_DECLARED_KINDS,
    ),
    "post_route": ToolEvidenceStagePolicy(
        run_stages=frozenset({"post_route"}),
        intrinsic_artifact_kinds=frozenset({
            "amd.vivado.post_route_timing",
            "amd.vivado.post_route_utilization",
        }),
        stage_declared_artifact_kinds=_VIVADO_STAGE_DECLARED_KINDS,
    ),
    "hardware_runtime": ToolEvidenceStagePolicy(
        run_stages=frozenset({"hardware_runtime"}),
        intrinsic_artifact_kinds=frozenset(),
    ),
})

# A kind typed by this policy keeps that meaning in every stage.  The plugin
# extension contract is only an escape hatch for genuinely unknown kinds; it
# must never relabel a csynth/cosim/RTL report as post-route or board evidence.
_KNOWN_POLICY_ARTIFACT_KINDS = frozenset(
    kind
    for policy in TOOL_EVIDENCE_STAGE_POLICY.values()
    for kind in (
        policy.intrinsic_artifact_kinds
        | policy.stage_declared_artifact_kinds
    )
)


_EXTENSION_SEMANTICS_BY_AUTHORITY: Mapping[str, frozenset[str]] = MappingProxyType({
    AuthorityClass.TOOL_OBSERVATION.value: frozenset({"tool_report"}),
    AuthorityClass.VERIFICATION_EVIDENCE.value: frozenset({"verification_report"}),
    AuthorityClass.PHYSICAL_MEASUREMENT.value: frozenset({"physical_measurement"}),
})


def run_claims_tool_truth(run: ToolRun) -> bool:
    """Return whether a run claims tool truth, including an invalid claim."""

    return run.metadata.get("tool_truth") is True


def real_tool_run_claim_error(run: ToolRun) -> str | None:
    """Return why a tool-truth claim lacks a real, fresh execution identity.

    A real invocation that exits non-zero remains valid ledger evidence.  Run
    success is deliberately checked only by label/gate consumers.
    """
    if run.stage == "index":
        return "index runs cannot claim external tool truth"
    if str(run.status) not in {"succeeded", "failed"}:
        return "real-tool truth requires a freshly executed succeeded/failed run"
    if str(run.backend).casefold() not in REAL_TOOL_RUN_BACKENDS:
        return f"backend {run.backend!r} is not an approved real-tool backend"
    authority = str(run.metadata.get("authority", "")).casefold()
    if authority not in REAL_TOOL_RUN_AUTHORITIES:
        return f"run authority {authority!r} is not a real-tool evidence authority"
    if run.metadata.get("fresh_execution") is not True:
        return "run does not attest a fresh execution"
    if run.metadata.get("fresh_tool_truth") is not True:
        return "run does not attest fresh tool truth"
    if run.metadata.get("tool_truth") is not True:
        return "run does not claim tool truth"
    return None


def successful_fresh_tool_run_error(run: ToolRun) -> str | None:
    """Return why a run is not eligible as successful present-label truth."""

    claim_error = real_tool_run_claim_error(run)
    if claim_error is not None:
        return claim_error
    if str(run.status) != "succeeded":
        return f"run status {str(run.status)!r} is not succeeded"
    if str(run.failure_class) != "none" or run.exit_code not in {None, 0}:
        return "run has a failure classification or non-zero exit code"
    return None


def tool_run_manifest_identity_error(
    run: ToolRun, manifest: ProjectManifest,
) -> str | None:
    """Return why a tool-truth run differs from its immutable snapshot manifest."""

    try:
        expected_toolchain = manifest.toolchain_for_stage(run.stage)
        expected_command = manifest.stage_commands[run.stage]
    except (KeyError, TypeError, ValueError):
        return (
            f"stage {run.stage!r} is not declared by the immutable snapshot manifest"
        )
    if (
        run.toolchain_id != expected_toolchain.id
        or run.environment_hash != expected_toolchain.environment_hash
        or list(run.command) != list(expected_command)
        or run.working_directory != "."
    ):
        return "execution identity does not match the immutable snapshot manifest"
    return None


def tool_run_stable_id_error(run: ToolRun) -> str | None:
    """Return an error when a caller supplied a non-canonical ToolRun ID."""

    payload = json_ready(run)
    payload["id"] = ""
    try:
        expected = ToolRun.from_dict(payload).id
    except (KeyError, TypeError, ValueError) as exc:
        return f"run payload cannot be canonically reconstructed: {exc}"
    if run.id != expected:
        return f"run stable ID {run.id!r} does not match {expected!r}"
    return None


def _declared_output_contracts(
    manifest: ProjectManifest, stage: str,
) -> tuple[ExecutionDeclaredOutput, ...]:
    return tuple(sorted((
        ExecutionDeclaredOutput(
            path=item.path,
            kind=item.kind,
            required=item.required,
            max_bytes=item.metadata.get("max_bytes", _DEFAULT_MAX_OUTPUT_BYTES),
        )
        for item in manifest.stage_outputs.get(stage, [])
    ), key=lambda item: item.path))


def execution_attestation_error(
    attestation: ExecutionAttestation,
    run: ToolRun,
    snapshot: DesignSnapshot,
    manifest: ProjectManifest,
    artifacts: Sequence[ArtifactRef],
) -> str | None:
    """Re-verify a persisted execution attestation from public ledger data.

    This validates deterministic content only.  Initial authorization remains a
    separate, process-local capability check at the store commit boundary.
    """

    if attestation.protocol_version != EXECUTION_ATTESTATION_PROTOCOL:
        return "execution attestation uses an unsupported protocol"
    if attestation.validator != EXECUTION_ATTESTATION_VALIDATOR:
        return "execution attestation was not validated by StageOrchestrator"
    stable_error = tool_run_stable_id_error(run)
    if stable_error is not None:
        return stable_error
    expected_runner_fingerprint = run.metadata.get("runner_fingerprint")
    expected = {
        "run_id": run.id,
        "snapshot_id": run.snapshot_id,
        "stage": run.stage,
        "runner_identity": run.backend,
        "runner_fingerprint": expected_runner_fingerprint,
        "request_hash": run.request_hash,
        "run_payload_hash": stable_hash(json_ready(run)),
        "manifest_hash": snapshot.manifest_hash,
        "build_hash": snapshot.build_hash,
        "target_hash": snapshot.target_hash,
        "constraint_hash": snapshot.constraint_hash,
        "toolchain_hash": snapshot.toolchain_hash,
        "toolchain_id": run.toolchain_id,
    }
    for name, value in expected.items():
        if getattr(attestation, name) != value:
            return f"execution attestation {name} does not match its run/snapshot"
    builtin_authority = {
        "runner.local": "hlsgraph.runner_authority.builtin_local.v1",
        "runner.ssh": "hlsgraph.runner_authority.builtin_ssh.v1",
    }.get(str(run.backend).casefold())
    if (attestation.runner_authority != builtin_authority
            and not attestation.runner_authority.startswith(
                "hlsgraph.runner_authority.plugin."
            )):
        return "execution attestation runner authority is not a trusted built-in/plugin origin"
    if snapshot.id != run.snapshot_id or snapshot.project_id != manifest.project_id:
        return "execution attestation snapshot/project identity is inconsistent"
    manifest_hashes = {
        "manifest_hash": stable_hash(manifest.identity_payload()),
        "build_hash": stable_hash(manifest.build),
        "target_hash": stable_hash(manifest.target),
        "constraint_hash": stable_hash(manifest.constraints),
        "toolchain_hash": stable_hash({
            "toolchains": manifest.toolchains,
            "stage_toolchains": manifest.stage_toolchains,
        }),
    }
    for name, value in manifest_hashes.items():
        if getattr(attestation, name) != value:
            return f"execution attestation {name} does not match the immutable manifest"
    identity_error = tool_run_manifest_identity_error(run, manifest)
    if identity_error is not None:
        return identity_error
    expected_declarations = _declared_output_contracts(manifest, run.stage)
    if attestation.declared_outputs != expected_declarations:
        return "execution attestation output declarations differ from the manifest"

    values = list(artifacts)
    by_id = {item.id: item for item in values}
    if len(by_id) != len(values):
        return "execution attestation artifacts contain duplicate IDs"
    if set(by_id) != set(run.output_artifact_ids):
        return "execution attestation artifacts differ from run output_artifact_ids"
    declarations = {item.path: item for item in expected_declarations}
    expected_outputs: list[ExecutionOutputAttestation] = []
    for artifact in values:
        path = artifact.metadata.get("declared_output_path")
        if not isinstance(path, str) or path not in declarations:
            return f"execution output artifact {artifact.id!r} lacks a declared output path"
        declaration = declarations[path]
        if artifact.kind != declaration.kind:
            return f"execution output artifact {artifact.id!r} has the wrong declared kind"
        if artifact.producer_run_id != run.id:
            return f"execution output artifact {artifact.id!r} has the wrong producer run"
        if artifact.size > declaration.max_bytes:
            return f"execution output artifact {artifact.id!r} exceeds its declared limit"
        expected_outputs.append(ExecutionOutputAttestation(
            artifact_id=artifact.id,
            path=path,
            kind=artifact.kind,
            sha256=artifact.sha256,
            size=artifact.size,
        ))
    expected_output_tuple = tuple(sorted(expected_outputs, key=lambda item: item.path))
    if attestation.outputs != expected_output_tuple:
        return "execution attestation output identities differ from committed artifacts"
    if str(run.status) == "succeeded":
        missing_required = sorted(
            item.path for item in expected_declarations
            if item.required and item.path not in {output.path for output in expected_outputs}
        )
        if missing_required:
            return "successful execution omits required declared outputs"
    staged_manifest = run.metadata.get("staged_output_manifest")
    expected_staged_manifest = [
        {"path": item.path, "size": item.size, "sha256": item.sha256}
        for item in expected_output_tuple
    ]
    if (not isinstance(staged_manifest, list)
            or any(not isinstance(item, Mapping) for item in staged_manifest)):
        return "execution runner staged manifest is missing or malformed"
    normalized_staged = sorted(
        (dict(item) for item in staged_manifest),
        key=lambda item: str(item.get("path", "")),
    )
    if normalized_staged != expected_staged_manifest:
        return "execution attestation outputs differ from the runner staged manifest"
    return None


def execution_commit_receipt_error(
    receipt: ExecutionCommitReceipt,
    attestation: ExecutionAttestation,
    run: ToolRun,
) -> str | None:
    """Return why a persisted store receipt does not close to its attestation."""

    if receipt.receipt_contract != EXECUTION_RECEIPT_CONTRACT:
        return "execution commit receipt uses an unsupported contract"
    if receipt.validator != EXECUTION_RECEIPT_VALIDATOR:
        return "execution commit receipt uses an unsupported validator"
    expected = {
        "attestation_id": attestation.id,
        "run_id": run.id,
        "snapshot_id": run.snapshot_id,
        "run_payload_hash": stable_hash(json_ready(run)),
        "attestation_payload_hash": stable_hash(json_ready(attestation)),
    }
    for name, value in expected.items():
        if getattr(receipt, name) != value:
            return f"execution commit receipt {name} does not match its attestation/run"
    return None


def _extension_contract_error(
    artifact: ArtifactRef, observation: Observation, run: ToolRun,
) -> str | None:
    contract = artifact.metadata.get(TOOL_EVIDENCE_EXTENSION_METADATA_KEY)
    if not isinstance(contract, Mapping):
        return (
            f"artifact kind {artifact.kind!r} is not typed by the policy and lacks "
            f"metadata.{TOOL_EVIDENCE_EXTENSION_METADATA_KEY}"
        )
    expected_semantics = _EXTENSION_SEMANTICS_BY_AUTHORITY.get(str(observation.authority))
    if expected_semantics is None:
        return f"authority {str(observation.authority)!r} is not eligible for tool evidence"
    if contract.get("policy_version") != TOOL_EVIDENCE_POLICY_VERSION:
        return (
            f"artifact kind {artifact.kind!r} has an unsupported extension policy version"
        )
    if contract.get("observation_stage") != observation.stage:
        return (
            f"artifact kind {artifact.kind!r} does not explicitly bind observation stage "
            f"{observation.stage!r}"
        )
    if contract.get("run_stage") != run.stage:
        return (
            f"artifact kind {artifact.kind!r} does not explicitly bind producer stage "
            f"{run.stage!r}"
        )
    if contract.get("semantics") not in expected_semantics:
        return (
            f"artifact kind {artifact.kind!r} has incompatible evidence semantics for "
            f"authority {str(observation.authority)!r}"
        )
    return None


def tool_evidence_compatibility_error(
    observation: Observation, run: ToolRun, artifacts: Sequence[ArtifactRef],
) -> str | None:
    """Return a deterministic incompatibility reason, or ``None`` when valid.

    Producer identity, CAS integrity, run status, and freshness are separate
    checks.  This function only answers whether the claimed stage/authority and
    typed report semantics can belong to that producer stage.
    """

    policy = TOOL_EVIDENCE_STAGE_POLICY.get(observation.stage)
    if policy is None:
        return f"observation stage {observation.stage!r} is not eligible for tool truth"
    if run.stage not in policy.run_stages:
        return (
            f"observation stage {observation.stage!r} is incompatible with producer "
            f"stage {run.stage!r}"
        )
    if (observation.authority == AuthorityClass.PHYSICAL_MEASUREMENT
            and observation.stage != "hardware_runtime"):
        return "physical_measurement authority requires hardware_runtime stage"
    if not artifacts:
        return "tool evidence must cite at least one typed report artifact"

    for artifact in artifacts:
        if artifact.kind in policy.intrinsic_artifact_kinds:
            declared_stage = artifact.metadata.get("stage")
            if declared_stage is not None and declared_stage != observation.stage:
                return (
                    f"artifact kind {artifact.kind!r} declares contradictory stage "
                    f"{declared_stage!r}"
                )
            continue
        if artifact.kind in policy.stage_declared_artifact_kinds:
            if artifact.metadata.get("stage") != observation.stage:
                return (
                    f"generic artifact kind {artifact.kind!r} must explicitly declare "
                    f"stage {observation.stage!r}"
                )
            continue
        if artifact.kind in _KNOWN_POLICY_ARTIFACT_KINDS:
            return (
                f"artifact kind {artifact.kind!r} is typed for another evidence stage "
                "and cannot be reinterpreted by an extension contract"
            )
        extension_error = _extension_contract_error(artifact, observation, run)
        if extension_error is not None:
            return extension_error
    return None
