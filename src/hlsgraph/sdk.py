"""High-level SDK for project indexing, querying, execution, rendering, and export."""
from __future__ import annotations

from dataclasses import replace
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Sequence

from .bundle import BundleError, GraphBundle
from .export import export_dataset, export_graph_json
from .graph import CanonicalGraph
from .extract import (
    ExternalDirectiveExtractor,
    ExtractionContext,
    ExtractionPipeline,
    ExtractionResult,
    LibClangExtractor,
    LlvmIrExtractor,
    MlirTextExtractor,
    RegexSourceExtractor,
    VitisReportExtractor,
    VivadoReportExtractor,
)
from .model import (
    DatasetManifest, Diagnostic, DiagnosticSeverity, FailureClass, GateKind, GateStatus,
    PredictionEnvelope, RunStatus, ToolRun, VariantAction, VerificationKind,
    hash_artifact_bytes, stable_hash, utc_now,
)
from .manifest import project_path
from .query import CoreService, ExploreResult, ExploreSpec, QueryResult, QuerySpec, StatusResult
from .runner import Runner, StageOrchestrator, StageResult, ToolRunRequest
from .plugins import load_extractors


DEFAULT_STAGE_ORDER = (
    "index", "csim", "csynth", "rtl_cosim", "rtl_export", "vivado_synth", "post_route",
)


def _canonical_extraction_value(value: Any, path: str = "options") -> Any:
    """Return the exact JSON-domain value used for extraction identity.

    Plugins may read arbitrary option keys and may contribute a runtime
    identity.  Accepting sets, custom containers, non-string mapping keys, or
    non-finite floats would make one logical request hash differently across
    processes (or fail only after work had begun), so the whole extraction
    profile is normalized before snapshot creation.
    """
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return value
    if type(value) in {list, tuple}:
        return [
            _canonical_extraction_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise ValueError(f"{path} mapping keys must be strings")
        return {
            key: _canonical_extraction_value(value[key], f"{path}.{key}")
            for key in sorted(value)
        }
    raise ValueError(
        f"{path} must contain only canonical JSON values; got {type(value).__name__}"
    )


@dataclass(slots=True)
class IndexResult:
    snapshot_id: str
    graph_hash: str
    entities: int
    relations: int
    observations: int
    derivations: int
    verifications: int
    diagnostics: int
    success: bool
    capabilities: list[str]
    parent_snapshot_id: str | None = None
    action_id: str | None = None


class Project:
    def __init__(self, bundle: GraphBundle):
        self.bundle = bundle

    @classmethod
    def create_from_manifest(cls, manifest_path: str | Path, *, force: bool = False) -> "Project":
        return cls(GraphBundle.from_manifest(manifest_path, force=force))

    @classmethod
    def open(cls, project_root: str | Path) -> "Project":
        return cls(GraphBundle.open(project_root))

    def index(self, *, degraded: bool = False,
              options: dict[str, Any] | None = None,
              parent_snapshot_id: str | None = None,
              action_id: str | None = None) -> IndexResult:
        with self.bundle.execution_lock():
            return self._index_locked(
                degraded=degraded, options=options,
                parent_snapshot_id=parent_snapshot_id, action_id=action_id,
            )

    def _index_locked(self, *, degraded: bool = False,
                      options: dict[str, Any] | None = None,
                      parent_snapshot_id: str | None = None,
                      action_id: str | None = None) -> IndexResult:
        self.bundle.refresh_manifest()
        if action_id is not None:
            if not isinstance(action_id, str) or not action_id.strip():
                raise ValueError("action_id must be a non-empty string or None")
            action = self.bundle.store.variant(action_id)
            if action is None:
                raise KeyError(action_id)
            action_parent = action["parent_snapshot_id"]
            if (parent_snapshot_id is not None
                    and parent_snapshot_id != action_parent):
                raise ValueError(
                    "action_id belongs to a different parent_snapshot_id"
                )
            parent_snapshot_id = action_parent
        if options is not None and type(options) is not dict:
            raise ValueError("options must be a dictionary with string keys")
        if (options is not None and "extractor_plugins" in options
                and not isinstance(options["extractor_plugins"], (list, tuple))):
            raise ValueError("extractor_plugins must be an ordered list or tuple")
        index_options = _canonical_extraction_value(options or {})
        # Semantic no-ops must not create a new extraction profile.  In
        # particular, CLI adapters naturally produce an empty list for a
        # repeated --extractor-plugin option while the Python SDK defaults to
        # no key at all.  Canonicalize both to the same profile so SDK/CLI/REST
        # consumers agree on snapshot, entity, and graph identities.
        plugin_option = index_options.get("extractor_plugins")
        if plugin_option is not None:
            if not isinstance(plugin_option, (list, tuple)):
                raise ValueError("extractor_plugins must be an ordered list or tuple")
            raw_plugin_names = list(plugin_option)
            if any(not isinstance(item, str) or not item.strip() or item != item.strip()
                   for item in raw_plugin_names):
                raise ValueError("extractor plugin names must be non-empty trimmed strings")
            plugin_names = list(dict.fromkeys(raw_plugin_names))
            if plugin_names:
                index_options["extractor_plugins"] = plugin_names
            else:
                index_options.pop("extractor_plugins", None)
        else:
            plugin_names = []
        source_extractor = RegexSourceExtractor() if degraded else LibClangExtractor()
        extractors = [
            source_extractor, ExternalDirectiveExtractor(), MlirTextExtractor(), LlvmIrExtractor(),
            VitisReportExtractor(), VivadoReportExtractor(),
        ]
        extractors.extend(load_extractors(plugin_names))
        extractor_identities = []
        for item in extractors:
            identity: dict[str, Any] = {"name": item.name, "version": item.version}
            runtime_identity = getattr(item, "runtime_identity", None)
            if callable(runtime_identity):
                identity["runtime"] = runtime_identity()
            extractor_identities.append(identity)
        extraction_profile = _canonical_extraction_value({
            "profile": "hlsgraph.canonical_index.v1", "degraded": degraded,
            "extractors": extractor_identities,
            "options": index_options,
        }, "extraction_profile")
        extraction_hash = stable_hash(extraction_profile)
        snapshot = self.bundle.snapshot(
            parent_snapshot_id=parent_snapshot_id,
            action_id=action_id,
            extraction_hash=extraction_hash,
        )
        artifacts = self.bundle.store.artifacts(snapshot.id)
        context = ExtractionContext(
            project_root=self.bundle.project_root, manifest=self.bundle.manifest,
            snapshot=snapshot, artifacts={item.id: item for item in artifacts},
            allow_degraded=degraded, options=index_options,
        )
        started = utc_now()
        result = ExtractionPipeline(extractors).run(context)
        result.graph.metadata.update({
            "project_id": self.bundle.manifest.project_id,
            "top": self.bundle.manifest.build.top,
            "target_vendor": self.bundle.manifest.target.vendor,
            "degraded": degraded,
            "extraction_hash": extraction_hash,
            "extractor_identities": extractor_identities,
        })
        entity_ids = set(result.graph.entities)
        artifact_ids = set(context.artifacts)
        observation_ids = {item.id for item in result.observations}
        derivation_ids = {item.id for item in result.derivations}
        evidence_errors: list[str] = []
        for item in result.observations:
            if item.subject_id not in entity_ids | artifact_ids:
                evidence_errors.append(f"observation {item.id} has an unresolved subject")
            if item.artifact_id and item.artifact_id not in artifact_ids:
                evidence_errors.append(f"observation {item.id} cites an unattached artifact")
            if item.anchor and item.anchor.artifact_id not in artifact_ids:
                evidence_errors.append(f"observation {item.id} has an unattached anchor")
        for item in result.derivations:
            if item.subject_id not in entity_ids | artifact_ids:
                evidence_errors.append(f"derivation {item.id} has an unresolved subject")
            if not set(item.input_observation_ids).issubset(observation_ids):
                evidence_errors.append(f"derivation {item.id} has unavailable input observations")
        available_evidence = observation_ids | derivation_ids
        for item in result.verifications:
            if not set(item.evidence_ids).issubset(available_evidence):
                evidence_errors.append(f"verification {item.id} has unavailable evidence")
        kernels = [item for item in result.graph.entities.values()
                   if item.kind == "hls.kernel"]
        if len(kernels) != 1 or kernels[0].name != self.bundle.manifest.build.top:
            result.diagnostics.append(Diagnostic(
                snapshot_id=snapshot.id, code="extractor.kernel_boundary",
                severity=DiagnosticSeverity.ERROR,
                message=("v0.1 canonical indexing requires exactly one hls.kernel whose "
                         "name equals manifest build.top"),
                metadata={"expected_top": self.bundle.manifest.build.top,
                          "kernel_count": len(kernels),
                          "kernel_names": sorted(item.name for item in kernels)},
            ))
        if evidence_errors:
            result.diagnostics.append(Diagnostic(
                snapshot_id=snapshot.id, code="extractor.invalid_evidence",
                severity=DiagnosticSeverity.ERROR,
                message="extraction output failed evidence-integrity validation",
                metadata={"error_count": len(evidence_errors),
                          "categories": sorted({value.split(" has ", 1)[-1]
                                                for value in evidence_errors})},
            ))
        # Revalidate as the last filesystem-dependent step before deciding
        # whether this candidate is eligible for the atomic graph commit.
        final_mismatches = self._artifact_byte_mismatches(artifacts)
        final_stale = self.bundle.is_stale(snapshot)
        if final_mismatches or final_stale:
            result.diagnostics.append(Diagnostic(
                snapshot_id=snapshot.id, code="extractor.snapshot_changed",
                severity=DiagnosticSeverity.ERROR,
                message=("snapshot inputs changed while extraction was running; "
                         "candidate graph was not committed"),
                metadata={"stale": final_stale,
                          "mismatch_ids": final_mismatches},
            ))
        fatal = [item for item in result.diagnostics
                 if item.severity in {DiagnosticSeverity.ERROR, DiagnosticSeverity.CRITICAL}]
        request_hash = stable_hash({"snapshot": snapshot.id,
                                    "extractors": extractor_identities,
                                    "degraded": degraded, "options": index_options})
        run = ToolRun(
            snapshot_id=snapshot.id, stage="index", backend="extractor.local",
            request_hash=request_hash,
            status=RunStatus.FAILED if fatal else RunStatus.SUCCEEDED,
            failure_class=FailureClass.INPUT if fatal else FailureClass.NONE,
            started_at=started, finished_at=utc_now(),
            message=f"{len(fatal)} fatal extraction diagnostics" if fatal else None,
            metadata={"extractors": extractor_identities,
                      "coverage": result.coverage, "capabilities": result.capabilities,
                      "tool_truth": False, "partial_graph_persisted": False,
                      "candidate_graph_hash": result.graph.graph_hash},
        )
        scoped_diagnostics = []
        resolvable_subjects = artifact_ids if fatal else entity_ids | artifact_ids
        for item in result.diagnostics:
            metadata = dict(item.metadata)
            subject_id = item.subject_id
            if subject_id and subject_id not in resolvable_subjects:
                metadata["unresolved_candidate_subject_id"] = subject_id
                subject_id = None
            scoped_diagnostics.append(replace(
                item, id="", run_id=run.id, subject_id=subject_id, metadata=metadata,
            ))
        result.diagnostics = scoped_diagnostics
        run.diagnostics = [item.id for item in scoped_diagnostics]
        # A failed extraction is an immutable run/diagnostic event, not a
        # partially-authoritative canonical graph.  This also permits a retry of
        # the same design snapshot after its environment is repaired.
        if not fatal:
            self.bundle.store.commit_index_success(
                project_id=self.bundle.manifest.project_id,
                graph=result.graph,
                run=run,
                observations=result.observations,
                derivations=result.derivations,
                verifications=result.verifications,
                diagnostics=result.diagnostics,
            )
        else:
            self.bundle.store.commit_index_failure(run=run, diagnostics=result.diagnostics)
        stats = result.graph.stats()
        return IndexResult(
            snapshot_id=snapshot.id, graph_hash=result.graph.graph_hash,
            entities=stats["entities"], relations=stats["relations"],
            observations=len(result.observations), derivations=len(result.derivations),
            verifications=len(result.verifications), diagnostics=len(result.diagnostics),
            success=not fatal, capabilities=result.capabilities,
            parent_snapshot_id=snapshot.parent_snapshot_id,
            action_id=snapshot.action_id,
        )

    def service(self, snapshot_id: str | None = None) -> CoreService:
        return CoreService(self.bundle, snapshot_id=snapshot_id)

    def query(self, query: str | QuerySpec, **kwargs: Any) -> QueryResult:
        spec = query if isinstance(query, QuerySpec) else QuerySpec(query=query, **kwargs)
        return self.service().query(spec)

    def explore(self, spec: ExploreSpec | None = None, **kwargs: Any) -> ExploreResult:
        return self.service().explore(spec or ExploreSpec(**kwargs))

    def status(self) -> StatusResult:
        if self.bundle.latest_snapshot() is None:
            return StatusResult(data=self.bundle.status())
        return self.service().status()

    def traverse(self, entity_id: str, *, depth: int = 1, direction: str = "both",
                 relation_kinds: Sequence[str] = ()) -> dict[str, Any]:
        return self.service().traverse(entity_id, depth=depth, direction=direction,
                                       relation_kinds=relation_kinds)

    def impact(self, entity_id: str, *, depth: int = 2,
               relation_kinds: Sequence[str] = ()) -> dict[str, Any]:
        return self.service().impact(entity_id, depth=depth, relation_kinds=relation_kinds)

    def evidence(self, entity_id: str) -> dict[str, Any]:
        return self.service().evidence(entity_id)

    def compare(self, other_snapshot_id: str) -> dict[str, Any]:
        return self.service().compare(other_snapshot_id)

    def variants(self, *, parent_snapshot_id: str | None = None,
                 action_id: str | None = None) -> dict[str, Any]:
        """Read proposed actions and only their explicitly recorded lineage."""
        return self.service().variants(
            parent_snapshot_id=parent_snapshot_id, action_id=action_id,
        )

    def record_variant_action(self, action: VariantAction) -> str:
        self.bundle.store.add_variant(action)
        return action.id

    def record_prediction(self, prediction: PredictionEnvelope) -> str:
        graph = self.bundle.store.load_graph(prediction.snapshot_id)
        if prediction.subject_id not in graph.entities:
            raise KeyError(prediction.subject_id)
        self.bundle.store.add_prediction(prediction)
        return prediction.id

    def run(self, runner: Runner, stages: Sequence[str] | None = None, *,
            timeout_s: float = 7200.0) -> StageResult:
        with self.bundle.execution_lock():
            return self._run_locked(runner, stages=stages, timeout_s=timeout_s)

    def _run_locked(self, runner: Runner, stages: Sequence[str] | None = None, *,
                    timeout_s: float = 7200.0) -> StageResult:
        snapshot = self.bundle.latest_snapshot()
        if snapshot is None:
            raise ValueError("index the project before running tool stages")
        if self.bundle.is_stale(snapshot):
            raise BundleError(
                "active snapshot is stale; re-index before running tool stages"
            )
        artifacts = self.bundle.store.artifacts(snapshot.id)
        immutable_inputs = [item for item in artifacts if not item.producer_run_id]
        mismatches = self._artifact_byte_mismatches(immutable_inputs,
                                                    verify_declared_paths=True)
        if mismatches:
            raise BundleError(
                "active snapshot artifacts no longer match their recorded bytes; "
                "re-index or restore retained artifacts before running"
            )
        snapshot_manifest = self.bundle.store.snapshot_manifest(snapshot.id)
        if isinstance(stages, (str, bytes)):
            raise ValueError("stages must be a sequence of stage names, not a string")
        selected = list(
            [stage for stage in DEFAULT_STAGE_ORDER
             if stage in snapshot_manifest.stage_commands and stage != "index"]
            if stages is None else stages
        )
        if any(not isinstance(stage, str) or not stage.strip() for stage in selected):
            raise ValueError("selected stages must be non-empty strings")
        if len(set(selected)) != len(selected):
            raise ValueError("selected stages must be unique")
        unknown = [stage for stage in selected if stage not in snapshot_manifest.stage_commands]
        if unknown:
            raise ValueError(
                f"snapshot manifest has no command for stages: {', '.join(unknown)}"
            )
        selected_positions = {stage: index for index, stage in enumerate(selected)}
        for producer, specs in snapshot_manifest.stage_outputs.items():
            for consumer in {item for spec in specs for item in spec.consumed_by}:
                if consumer not in selected_positions:
                    continue
                if producer not in selected_positions:
                    raise ValueError(
                        f"selected stage {consumer!r} requires producer stage {producer!r}"
                    )
                if selected_positions[producer] >= selected_positions[consumer]:
                    raise ValueError(
                        f"stage dependency order requires {producer!r} before {consumer!r}"
                    )
        existing_outputs = sorted(
            spec.path for stage in selected
            for spec in snapshot_manifest.stage_outputs.get(stage, [])
            if project_path(self.bundle.project_root, spec.path).exists()
        )
        if existing_outputs:
            raise BundleError(
                "declared stage outputs must be run-isolated and absent before execution: "
                + ", ".join(existing_outputs)
            )
        if (any(snapshot_manifest.stage_outputs.get(stage) for stage in selected)
                and getattr(runner, "provides_local_output_bytes", False) is not True):
            raise BundleError(
                f"{getattr(runner, 'name', type(runner).__name__)} declared-output ingestion "
                "is unavailable: the runner does not explicitly guarantee that verified "
                "output bytes are synchronously available beneath the local project root"
            )

        # Execute stage-by-stage so a missing/malformed declared report stops
        # the flow before a later vendor stage is launched.
        completed_runs: list[ToolRun] = []
        stopped: str | None = None
        # Historical outputs attached to this snapshot are evidence, not implicit
        # inputs to a new execution. Only immutable design inputs plus explicitly
        # chained outputs participate in each request/cache identity.
        base_artifacts = {item.id: item for item in artifacts if not item.producer_run_id}
        chained_artifacts: dict[str, tuple[Any, set[str]]] = {}
        for stage in selected:
            toolchain = snapshot_manifest.toolchain_for_stage(stage)
            remote_attestation = toolchain.metadata.get("remote_attestation_argv")
            remote_attestation_argv = (list(remote_attestation)
                                       if isinstance(remote_attestation, list) else [])
            output_specs = snapshot_manifest.stage_outputs.get(stage, [])
            current_inputs = sorted(
                [*base_artifacts.values(),
                 *(artifact for artifact, consumers in chained_artifacts.values()
                   if stage in consumers)],
                key=lambda item: item.id,
            )
            stage_metadata: dict[str, Any] = {
                "project_id": snapshot_manifest.project_id,
                "input_artifacts": [
                    {"id": item.id, "uri": item.uri,
                     "sha256": item.sha256, "size": item.size}
                    for item in current_inputs
                ],
                "declared_outputs": [
                    {"path": item.path, "kind": item.kind, "required": item.required,
                     "consumed_by": list(item.consumed_by)}
                    for item in output_specs
                ],
                "remote_attestation_argv": remote_attestation_argv,
            }
            for identity_key in ("campaign_id", "workload_id"):
                values = {str(item.metadata[identity_key]) for item in output_specs
                          if item.metadata.get(identity_key)}
                if len(values) > 1:
                    raise ValueError(
                        f"stage {stage!r} declares multiple {identity_key} values"
                    )
                if values:
                    stage_metadata[identity_key] = next(iter(values))
            request = ToolRunRequest(
                snapshot_id=snapshot.id, stage=stage,
                argv=list(snapshot_manifest.stage_commands[stage]),
                working_directory=".", timeout_s=timeout_s,
                toolchain_id=toolchain.id,
                environment_hash=toolchain.environment_hash,
                input_artifact_ids=[item.id for item in current_inputs],
                nonzero_failure=FailureClass.CORRECTNESS
                if stage in {"csim", "rtl_cosim"} else FailureClass.DESIGN_COMPILE,
                metadata=stage_metadata,
            )
            pre_mismatches = self._artifact_byte_mismatches(
                current_inputs, verify_declared_paths=True,
            )
            pre_stale = self.bundle.is_stale(snapshot)
            if pre_mismatches or pre_stale:
                run = self._input_validation_failure_run(
                    request, runner, mismatches=pre_mismatches,
                    stale=pre_stale, execution_started=False,
                )
                self.bundle.store.add_run(run)
                completed_runs.append(run)
                stopped = request.stage
                break
            partial = StageOrchestrator(runner).execute([request])
            run = partial.runs[0]
            post_mismatches = self._artifact_byte_mismatches(
                current_inputs, verify_declared_paths=True,
            )
            post_stale = self.bundle.is_stale(snapshot)
            if post_mismatches or post_stale:
                run = self._input_validation_failure_run(
                    request, runner, mismatches=post_mismatches,
                    stale=post_stale, execution_started=True,
                    previous=run,
                )
                self.bundle.store.add_run(run)
                completed_runs.append(run)
                stopped = request.stage
                break
            completed_runs.append(run)
            runner_gate_failed = any(gate.status == GateStatus.FAIL for gate in run.gates)
            if run.status not in {RunStatus.SUCCEEDED, RunStatus.CACHED}:
                self.bundle.store.add_run(run)
                stopped = request.stage
                break
            if output_specs:
                extraction = self._commit_declared_run_outputs(
                    snapshot, snapshot_manifest, run, output_specs,
                )
                produced = {item.id: item for item in
                            self.bundle.store.artifacts(snapshot.id)
                            if item.id in set(run.output_artifact_ids)}
                for artifact in produced.values():
                    consumers = {str(item) for item in
                                 artifact.metadata.get("consumed_by_stages", [])}
                    if consumers:
                        chained_artifacts[artifact.id] = (artifact, consumers)
                report_failed = any(
                    item.severity in {DiagnosticSeverity.ERROR, DiagnosticSeverity.CRITICAL}
                    for item in extraction.diagnostics
                )
                gate_failed = any(
                    item.status == GateStatus.FAIL for item in extraction.verifications
                ) or any(
                    item.predicate in {"gate.resource_fits", "gate.post_route_timing"}
                    and item.value is False for item in extraction.derivations
                )
                if report_failed or gate_failed or runner_gate_failed:
                    stopped = request.stage
                    break
            else:
                self.bundle.store.add_run(run)
                if runner_gate_failed:
                    stopped = request.stage
                    break

        gate_payload = self.service(snapshot.id).verification_gates()
        required = {GateKind.CORRECTNESS, GateKind.RESOURCE_FITS,
                    GateKind.POST_ROUTE_TIMING}
        gates = {
            kind: GateStatus(gate_payload[str(kind)]["status"])
            for kind in required
            if gate_payload[str(kind)]["status"] != GateStatus.UNKNOWN.value
        }
        checks_payload = gate_payload[str(GateKind.CORRECTNESS)].get("checks", {})
        correctness_checks = {
            name: GateStatus(value["status"])
            for name, value in checks_payload.items()
            if name in {VerificationKind.CSIM.value, VerificationKind.RTL_COSIM.value}
        }
        required_checks = {VerificationKind.CSIM.value, VerificationKind.RTL_COSIM.value}
        eligible_campaigns = gate_payload[str(GateKind.CORRECTNESS)].get(
            "eligible_campaigns", []
        )
        eligible_run_ids = gate_payload[str(GateKind.CORRECTNESS)].get(
            "eligible_run_ids", {}
        )
        completed_run_ids = {run.id for run in completed_runs}
        invocation_campaigns = [
            cohort for cohort in eligible_campaigns
            if isinstance(eligible_run_ids.get(cohort), dict)
            and any(
                run_id in completed_run_ids
                for run_id in eligible_run_ids[cohort].get(
                    VerificationKind.CSIM.value, []
                )
            )
            and any(
                run_id in completed_run_ids
                for run_id in eligible_run_ids[cohort].get(
                    VerificationKind.RTL_COSIM.value, []
                )
            )
        ]
        invocation_physical_runs = (
            set(gate_payload.get("eligible_physical_runs", []))
            & completed_run_ids
        )
        invocation_complete = (
            bool(selected)
            and stopped is None
            and len(completed_runs) == len(selected)
            and all(run.status in {RunStatus.SUCCEEDED, RunStatus.CACHED}
                    for run in completed_runs)
        )
        invocation_tool_truth = (
            invocation_complete
            and all(run.metadata.get("tool_truth") is True
                    and str(run.metadata.get("authority", "")) != "synthetic"
                    for run in completed_runs)
        )
        gates_complete = (
            required.issubset(gates)
            and all(gates[item] == GateStatus.PASS for item in required)
            and required_checks.issubset(correctness_checks)
            and all(correctness_checks[item] == GateStatus.PASS for item in required_checks)
            and bool(invocation_campaigns)
            and bool(invocation_physical_runs)
            and invocation_complete
        )
        historical_tool_truth = all(
            gate_payload[str(kind)].get("tool_truth") is True for kind in required
        ) and bool(eligible_campaigns) and bool(
            gate_payload.get("eligible_physical_runs", [])
        )
        tool_truth = historical_tool_truth and invocation_tool_truth
        verified = (
            bool(gate_payload.get("verified"))
            and gates_complete
            and tool_truth
        )
        return StageResult(
            runs=completed_runs, gates=gates, correctness_checks=correctness_checks,
            gates_complete=gates_complete, tool_truth=tool_truth,
            verified=verified, stopped_after_stage=stopped,
        )

    def _commit_declared_run_outputs(
        self, snapshot: Any, manifest: Any, run: ToolRun, output_specs: Sequence[Any]
    ) -> ExtractionResult:
        """Attach declared report outputs and parsed evidence in one ledger commit."""
        managed = []
        missing: list[str] = []
        for spec in output_specs:
            source = project_path(self.bundle.project_root, spec.path)
            if not source.is_file():
                if spec.required:
                    missing.append(spec.path)
                continue
            artifact, _stored_path, _created = self.bundle.prepare_managed_artifact(
                source, kind=spec.kind, role=spec.role, access=spec.access,
                producer_run_id=run.id, license=spec.license,
                metadata={**spec.metadata,
                          "declared_output_path": spec.path,
                          "consumed_by_stages": sorted(spec.consumed_by)},
            )
            managed.append(artifact)

        graph = self.service(snapshot.id).graph()
        extraction = ExtractionResult(graph=CanonicalGraph(snapshot.id))
        if missing:
            extraction.diagnostics.append(Diagnostic(
                snapshot_id=snapshot.id, code="runner.required_output_missing",
                severity=DiagnosticSeverity.ERROR, stage=run.stage,
                message=("required declared tool outputs were not produced: "
                         + ", ".join(sorted(missing))),
                metadata={"missing_paths": sorted(missing)},
            ))
        if managed:
            context = ExtractionContext(
                project_root=self.bundle.project_root, manifest=manifest,
                snapshot=snapshot, artifacts={item.id: item for item in managed},
                options={"existing_graph": graph},
            )
            for extractor in (VitisReportExtractor(), VivadoReportExtractor()):
                if not extractor.supports(context):
                    continue
                parsed = extractor.extract(context)
                if parsed.graph.entities or parsed.graph.relations:
                    extraction.diagnostics.append(Diagnostic(
                        snapshot_id=snapshot.id,
                        code="runner.structural_output_requires_reindex",
                        severity=DiagnosticSeverity.ERROR, stage=run.stage,
                        message=("a declared output contains structural graph facts; add it to "
                                 "the manifest artifact set and re-index a new snapshot"),
                        metadata={"extractor": extractor.name,
                                  "entity_count": len(parsed.graph.entities),
                                  "relation_count": len(parsed.graph.relations)},
                    ))
                    continue
                extraction.observations.extend(parsed.observations)
                extraction.derivations.extend(parsed.derivations)
                extraction.verifications.extend(parsed.verifications)
                extraction.diagnostics.extend(parsed.diagnostics)
                extraction.capabilities.extend(parsed.capabilities)

        # Adding run provenance changes stable IDs, so rebuild every downstream
        # reference before the atomic transaction.
        observation_ids: dict[str, str] = {}
        rebound_observations = []
        for item in extraction.observations:
            rebound = replace(item, id="", run_id=run.id)
            observation_ids[item.id] = rebound.id
            rebound_observations.append(rebound)
        extraction.observations = rebound_observations
        derivation_ids: dict[str, str] = {}
        rebound_derivations = []
        for item in extraction.derivations:
            rebound = replace(
                item, id="",
                input_observation_ids=[observation_ids.get(value, value)
                                       for value in item.input_observation_ids],
            )
            derivation_ids[item.id] = rebound.id
            rebound_derivations.append(rebound)
        extraction.derivations = rebound_derivations
        extraction.verifications = [replace(
            item, id="", run_id=run.id,
            evidence_ids=[observation_ids.get(value, derivation_ids.get(value, value))
                          for value in item.evidence_ids],
        ) for item in extraction.verifications]
        extraction.diagnostics = [replace(item, id="", run_id=run.id)
                                  for item in extraction.diagnostics]

        run.output_artifact_ids = [item.id for item in managed]
        run.diagnostics = [item.id for item in extraction.diagnostics]
        if any(item.severity in {DiagnosticSeverity.ERROR, DiagnosticSeverity.CRITICAL}
               for item in extraction.diagnostics):
            run.status = RunStatus.FAILED
            run.failure_class = FailureClass.INPUT
            run.message = "declared output ingestion failed; see run diagnostics"
        # A published CAS file is never rolled back on ledger failure: another
        # process may already reference the same content-addressed bytes.
        self.bundle.store.commit_run_result(
            run=run, artifacts=managed, observations=extraction.observations,
            derivations=extraction.derivations,
            verifications=extraction.verifications,
            diagnostics=extraction.diagnostics,
        )
        return extraction

    def _artifact_byte_mismatches(
        self, artifacts: Sequence[Any], *, verify_declared_paths: bool = False,
    ) -> list[str]:
        """Return opaque artifact/location identifiers whose live bytes drifted."""
        mismatches: set[str] = set()
        for artifact in artifacts:
            locations: list[tuple[str, str]] = [("artifact", artifact.uri)]
            declared = artifact.metadata.get("declared_output_path")
            if (verify_declared_paths and isinstance(declared, str) and declared
                    and declared != artifact.uri):
                locations.append(("declared_output", declared))
            for location_kind, relative in locations:
                try:
                    path = project_path(self.bundle.project_root, relative)
                    data = path.read_bytes()
                except (OSError, ValueError):
                    mismatches.add(f"{artifact.id}:{location_kind}")
                    continue
                if (len(data) != artifact.size
                        or hash_artifact_bytes(data) != artifact.sha256):
                    mismatches.add(f"{artifact.id}:{location_kind}")
        return sorted(mismatches)

    @staticmethod
    def _input_validation_failure_run(
        request: ToolRunRequest, runner: Runner, *, mismatches: Sequence[str],
        stale: bool, execution_started: bool, previous: ToolRun | None = None,
    ) -> ToolRun:
        """Create an immutable failure event without laundering invalid tool output."""
        event_time = utc_now()
        previous_fingerprint = (previous.metadata.get("runner_fingerprint")
                                if previous is not None else None)
        runner_fingerprint = (previous_fingerprint
                              if isinstance(previous_fingerprint, str)
                              and previous_fingerprint else runner.fingerprint)
        inherited_metadata = dict(previous.metadata) if previous else dict(request.metadata)
        inherited_metadata.update({
            "runner_fingerprint": runner_fingerprint,
            "fresh_execution": bool(
                execution_started
                and previous is not None
                and previous.metadata.get("fresh_execution") is True
            ),
            "fresh_tool_truth": False,
            "authority": "infrastructure",
            "tool_truth": False,
            "input_validation_failed": True,
            "input_mismatch_ids": sorted(mismatches),
            "snapshot_stale": bool(stale),
        })
        backend = previous.backend if previous is not None else runner.name
        return ToolRun(
            snapshot_id=request.snapshot_id, stage=request.stage,
            backend=backend,
            request_hash=request.cache_key(runner_fingerprint),
            toolchain_id=request.toolchain_id, status=RunStatus.FAILED,
            command=list(request.argv), working_directory=request.working_directory,
            environment_hash=request.environment_hash,
            input_artifact_ids=list(request.input_artifact_ids),
            output_artifact_ids=[], diagnostics=[], gates=[],
            failure_class=FailureClass.INPUT,
            exit_code=previous.exit_code if previous else None,
            started_at=previous.started_at if previous else event_time,
            finished_at=utc_now(), elapsed_s=previous.elapsed_s if previous else 0.0,
            message=("snapshot inputs changed during stage execution"
                     if execution_started else
                     "snapshot inputs failed validation before stage execution"),
            metadata=inherited_metadata,
        )

    def render(self, output: str | Path, *, format: str = "html",
               scope_id: str | None = None) -> Path:
        from .render import render
        graph = self.service().graph()
        text = render(graph, format=format, scope_id=scope_id,
                      observations=self.bundle.store.observations(graph.snapshot_id),
                      diagnostics=self.bundle.store.active_diagnostics(graph.snapshot_id))
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8", newline="\n")
        return output

    def export_graph(self, output: str | Path) -> Path:
        snapshot = self.bundle.latest_snapshot()
        if not snapshot:
            raise ValueError("bundle has no snapshot")
        return export_graph_json(self.bundle, snapshot.id, output)

    def export_dataset(self, output_dir: str | Path, dataset: DatasetManifest | None = None,
                       *, format: str = "jsonl") -> dict[str, Any]:
        snapshot = self.bundle.latest_snapshot()
        if not snapshot:
            raise ValueError("bundle has no snapshot")
        return export_dataset(self.bundle, snapshot.id, output_dir, dataset, format=format)
