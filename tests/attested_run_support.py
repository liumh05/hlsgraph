"""White-box support for legacy ledger fixtures.

These helpers deliberately cross private module boundaries so old unit tests can
construct a completed execution without launching vendor tools.  Production
callers must use ``Project.run``; no equivalent bypass is exported by hlsgraph.
"""
from __future__ import annotations

from typing import Any, Sequence

from hlsgraph.model import (
    ArtifactRef,
    ExecutionAttestation,
    ExecutionDeclaredOutput,
    ExecutionOutputAttestation,
    ToolRun,
    json_ready,
    stable_hash,
)
import hlsgraph.runner.core as runner_core
from hlsgraph.runner.staging import DEFAULT_MAX_OUTPUT_BYTES


def execution_authorization(bundle: Any, run: ToolRun,
                            artifacts: Sequence[ArtifactRef] = ()) -> object:
    """Issue a white-box authorization for an already-built test fixture."""

    artifacts = list(artifacts)
    run.metadata.setdefault("runner_fingerprint", stable_hash({
        "test_fixture": True, "backend": run.backend, "request": run.request_hash,
    }))
    run.metadata["staged_output_manifest"] = [
        {"path": str(item.metadata.get("declared_output_path") or item.uri),
         "size": item.size, "sha256": item.sha256}
        for item in sorted(
            artifacts,
            key=lambda value: str(
                value.metadata.get("declared_output_path") or value.uri
            ),
        )
    ]
    snapshot = bundle.store.snapshot(run.snapshot_id)
    manifest = bundle.store.snapshot_manifest(run.snapshot_id)
    declarations = tuple(ExecutionDeclaredOutput(
        path=item.path,
        kind=item.kind,
        required=item.required,
        max_bytes=item.metadata.get("max_bytes", DEFAULT_MAX_OUTPUT_BYTES),
    ) for item in manifest.stage_outputs.get(run.stage, []))
    outputs = tuple(ExecutionOutputAttestation(
        artifact_id=item.id,
        path=str(item.metadata.get("declared_output_path") or item.uri),
        kind=item.kind,
        sha256=item.sha256,
        size=item.size,
    ) for item in artifacts)
    authority = (
        "hlsgraph.runner_authority.builtin_ssh.v1"
        if run.backend == "runner.ssh"
        else "hlsgraph.runner_authority.builtin_local.v1"
    )
    attestation = ExecutionAttestation(
        run_id=run.id,
        snapshot_id=run.snapshot_id,
        stage=run.stage,
        runner_identity=run.backend,
        runner_authority=authority,
        runner_fingerprint=run.metadata["runner_fingerprint"],
        request_hash=run.request_hash,
        run_payload_hash=stable_hash(json_ready(run)),
        manifest_hash=snapshot.manifest_hash,
        build_hash=snapshot.build_hash,
        target_hash=snapshot.target_hash,
        constraint_hash=snapshot.constraint_hash,
        toolchain_hash=snapshot.toolchain_hash,
        toolchain_id=str(run.toolchain_id),
        declared_outputs=declarations,
        outputs=outputs,
    )
    authorization = runner_core._ExecutionCommitAuthorization(
        attestation, object(),
        _sentinel=runner_core._CAPABILITY_CONSTRUCTOR_SENTINEL,
    )
    with runner_core._AUTHORIZATION_LOCK:
        runner_core._ACTIVE_EXECUTION_AUTHORIZATIONS[attestation.id] = authorization
    return authorization


def commit_attested(bundle: Any, *, run: ToolRun,
                    artifacts: Sequence[ArtifactRef] | None = None,
                    **values: Any) -> None:
    artifacts = list(artifacts or [])
    authorization = execution_authorization(bundle, run, artifacts)
    bundle.store.commit_run_result(
        run=run,
        artifacts=artifacts,
        execution_authorization=authorization,
        **values,
    )
