"""Execution-location SPI for local, SSH, fake, and replay backends.

Runners do not know Vitis Tcl, reports, QoR, or correctness.  Toolchain adapters
construct argv and gates; runners only execute an immutable request.
"""
from __future__ import annotations

import abc
import copy
import hashlib
import os
import re
import shlex
import subprocess
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..model import (
    FailureClass,
    GateKind,
    GateResult,
    GateStatus,
    RunStatus,
    ToolRun,
    json_ready,
    safe_relative_path,
    stable_hash,
    utc_now,
)


RUNNER_PROTOCOL_VERSION = "hlsgraph.runner.v1"

# Values below are measurements made by a Runner, not caller-provided request
# context.  Reserving them prevents a request from pre-seeding provenance that
# a backend would otherwise have to overwrite.
RUNNER_MEASURED_METADATA_KEYS = frozenset({
    "authority", "bootstrap_environment_hash", "execution_enabled",
    "expected_remote_environment_hash",
    "fresh_execution", "fresh_tool_truth", "inherited_environment_hash",
    "input_mismatch_ids", "input_validation_failed", "output_embedded",
    "remote_environment_verified", "remote_inputs_verified",
    "remote_project_root", "replayed_from_run_id", "replayed_request_hash",
    "runner_fingerprint", "ssh_host", "stderr_bytes", "stderr_sha256",
    "snapshot_stale", "stdout_bytes", "stdout_sha256", "tool_truth",
})


def _environment_digest(environment: Mapping[str, str]) -> str:
    """Hash an environment without persisting names or values in the ledger."""
    return stable_hash({str(key): str(value) for key, value in sorted(environment.items())})


def _local_bootstrap_environment(
    environment: Mapping[str, str] | None = None, *, platform: str | None = None,
) -> dict[str, str]:
    """Return the minimum host variables required to spawn an isolated process.

    Windows requires ``SystemRoot`` when a caller supplies an explicit
    environment to ``CreateProcess`` (notably for side-by-side assemblies).
    This is OS bootstrap state, not general environment inheritance. Keeping
    the helper pure makes the exact allowlist independently testable.
    """

    if (platform or os.name) != "nt":
        return {}
    source = os.environ if environment is None else environment
    for key, value in source.items():
        if key.casefold() == "systemroot":
            return {"SystemRoot": str(value)}
    return {}


def _output_metadata(stdout: str | bytes | None, stderr: str | bytes | None) -> dict[str, Any]:
    result: dict[str, Any] = {"output_embedded": False}
    for name, value in (("stdout", stdout), ("stderr", stderr)):
        if value is None:
            raw = b""
        elif isinstance(value, bytes):
            raw = value
        else:
            raw = value.encode("utf-8", errors="replace")
        result[f"{name}_bytes"] = len(raw)
        result[f"{name}_sha256"] = hashlib.sha256(raw).hexdigest()
    return result


class RunnerProtocolError(ValueError):
    pass


class CacheMiss(KeyError):
    pass


@dataclass(slots=True)
class ToolRunRequest:
    snapshot_id: str
    stage: str
    argv: list[str]
    working_directory: str = "."
    environment: dict[str, str] = field(default_factory=dict)
    environment_hash: str | None = None
    toolchain_id: str | None = None
    input_artifact_ids: list[str] = field(default_factory=list)
    timeout_s: float = 3600.0
    nonzero_failure: FailureClass = FailureClass.DESIGN_COMPILE
    metadata: dict[str, Any] = field(default_factory=dict)
    protocol_version: str = RUNNER_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if self.protocol_version != RUNNER_PROTOCOL_VERSION:
            raise RunnerProtocolError(
                f"runner protocol mismatch: {self.protocol_version!r} != {RUNNER_PROTOCOL_VERSION!r}"
            )
        if not self.snapshot_id or not self.stage:
            raise RunnerProtocolError("snapshot_id and stage are required")
        if (not self.argv or not all(
                isinstance(item, str) and item and "\x00" not in item
                for item in self.argv)):
            raise RunnerProtocolError(
                "argv must be a non-empty list of strings without NUL"
            )
        if self.timeout_s <= 0:
            raise RunnerProtocolError("timeout_s must be positive")
        if any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key)
               for key in self.environment):
            raise RunnerProtocolError("environment variable names must be portable identifiers")
        if any(not isinstance(value, str) or "\x00" in value
               for value in self.environment.values()):
            raise RunnerProtocolError("environment values must be strings without NUL")
        if self.working_directory != ".":
            self.working_directory = safe_relative_path(self.working_directory, "working_directory")
        conflicts = sorted(set(self.metadata) & RUNNER_MEASURED_METADATA_KEYS)
        if conflicts:
            raise RunnerProtocolError(
                "request metadata uses runner-measured keys: " + ", ".join(conflicts)
            )
        self.nonzero_failure = (self.nonzero_failure if isinstance(self.nonzero_failure, FailureClass)
                                else FailureClass(self.nonzero_failure))

    def cache_key(self, runner_fingerprint: str) -> str:
        return stable_hash({
            "protocol": self.protocol_version,
            "snapshot": self.snapshot_id,
            "stage": self.stage,
            "argv": self.argv,
            "cwd": self.working_directory,
            "environment": dict(sorted(self.environment.items())),
            "environment_hash": self.environment_hash,
            "toolchain": self.toolchain_id,
            "inputs": sorted(self.input_artifact_ids),
            "timeout_s": self.timeout_s,
            "nonzero_failure": str(self.nonzero_failure),
            "metadata": self.metadata,
            "runner": runner_fingerprint,
        })


class Runner(abc.ABC):
    name: str
    # Fail closed for declared tool outputs.  A runner may opt in only when
    # ``execute`` returns after the output bytes are synchronously available
    # beneath the local project root.  Project.run still hashes and copies the
    # bytes itself; this capability merely rules out SSH/sync/replay ambiguity.
    provides_local_output_bytes: bool = False

    @property
    @abc.abstractmethod
    def fingerprint(self) -> str:
        raise NotImplementedError

    def capabilities(self) -> dict[str, Any]:
        return {"name": self.name, "fingerprint": self.fingerprint,
                "protocol_version": RUNNER_PROTOCOL_VERSION,
                "provides_local_output_bytes": self.provides_local_output_bytes}

    @abc.abstractmethod
    def execute(self, request: ToolRunRequest) -> ToolRun:
        raise NotImplementedError

    @staticmethod
    def _disabled(request: ToolRunRequest, backend: str, fingerprint: str) -> ToolRun:
        request_hash = request.cache_key(fingerprint)
        event_time = utc_now()
        return ToolRun(
            snapshot_id=request.snapshot_id, stage=request.stage, backend=backend,
            request_hash=request_hash, toolchain_id=request.toolchain_id,
            status=RunStatus.SKIPPED, command=list(request.argv),
            working_directory=request.working_directory,
            environment_hash=request.environment_hash,
            input_artifact_ids=list(request.input_artifact_ids),
            failure_class=FailureClass.UNSUPPORTED,
            started_at=event_time, finished_at=event_time, elapsed_s=0.0,
            message="execution is disabled; enable it explicitly on the runner",
            metadata={**request.metadata,
                      "runner_fingerprint": fingerprint, "execution_enabled": False,
                      "fresh_execution": False, "fresh_tool_truth": False,
                      "tool_truth": False},
        )


class LocalRunner(Runner):
    name = "runner.local"
    provides_local_output_bytes = True

    def __init__(self, project_root: str | Path, *, allow_execution: bool = False,
                 inherit_environment: bool = True):
        self.project_root = Path(project_root).resolve()
        self.allow_execution = bool(allow_execution)
        self.inherit_environment = bool(inherit_environment)

    @property
    def fingerprint(self) -> str:
        inherited_hash = (_environment_digest(os.environ)
                          if self.inherit_environment else None)
        bootstrap = ({}
                     if self.inherit_environment
                     else _local_bootstrap_environment())
        bootstrap_hash = _environment_digest(bootstrap) if bootstrap else None
        return stable_hash({"backend": self.name, "protocol": RUNNER_PROTOCOL_VERSION,
                            "inherit_environment": self.inherit_environment,
                            "inherited_environment_hash": inherited_hash,
                            "bootstrap_environment_hash": bootstrap_hash})

    def execute(self, request: ToolRunRequest) -> ToolRun:
        if not self.allow_execution:
            return self._disabled(request, self.name, self.fingerprint)
        cwd = (self.project_root / request.working_directory).resolve()
        try:
            cwd.relative_to(self.project_root)
        except ValueError as exc:
            raise RunnerProtocolError("working directory escaped project root") from exc
        if not cwd.is_dir():
            raise RunnerProtocolError(f"working directory does not exist: {request.working_directory}")
        bootstrap = ({}
                     if self.inherit_environment
                     else _local_bootstrap_environment())
        inherited_hash = (_environment_digest(os.environ)
                          if self.inherit_environment else None)
        bootstrap_hash = _environment_digest(bootstrap) if bootstrap else None
        env = dict(os.environ) if self.inherit_environment else dict(bootstrap)
        if (not self.inherit_environment
                and any(key.casefold() == "systemroot"
                        for key in request.environment)):
            env.pop("SystemRoot", None)
            bootstrap_hash = None
        env.update(request.environment)
        started_at = utc_now()
        started = time.monotonic()
        runner_fingerprint = self.fingerprint
        request_hash = request.cache_key(runner_fingerprint)
        try:
            process = subprocess.run(
                request.argv, cwd=cwd, env=env, capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=request.timeout_s,
                shell=False,
            )
            elapsed = time.monotonic() - started
            status = RunStatus.SUCCEEDED if process.returncode == 0 else RunStatus.FAILED
            failure = FailureClass.NONE if process.returncode == 0 else request.nonzero_failure
            return ToolRun(
                snapshot_id=request.snapshot_id, stage=request.stage, backend=self.name,
                request_hash=request_hash, toolchain_id=request.toolchain_id,
                status=status, command=list(request.argv),
                working_directory=request.working_directory,
                environment_hash=request.environment_hash,
                input_artifact_ids=list(request.input_artifact_ids),
                failure_class=failure, exit_code=process.returncode,
                started_at=started_at, finished_at=utc_now(), elapsed_s=elapsed,
                message=None if process.returncode == 0 else f"process exited with code {process.returncode}",
                metadata={**request.metadata,
                          "runner_fingerprint": runner_fingerprint,
                          "inherited_environment_hash": inherited_hash,
                          "bootstrap_environment_hash": bootstrap_hash,
                          **_output_metadata(process.stdout, process.stderr),
                          "execution_enabled": True,
                          "fresh_execution": True, "fresh_tool_truth": True,
                          "authority": "tool_observation", "tool_truth": True},
            )
        except subprocess.TimeoutExpired as exc:
            elapsed = time.monotonic() - started
            return ToolRun(
                snapshot_id=request.snapshot_id, stage=request.stage, backend=self.name,
                request_hash=request_hash, toolchain_id=request.toolchain_id,
                status=RunStatus.FAILED, command=list(request.argv),
                working_directory=request.working_directory,
                environment_hash=request.environment_hash,
                input_artifact_ids=list(request.input_artifact_ids),
                failure_class=FailureClass.TIMEOUT, exit_code=None,
                started_at=started_at, finished_at=utc_now(), elapsed_s=elapsed,
                message=f"stage timed out after {request.timeout_s}s",
                metadata={**request.metadata,
                          "runner_fingerprint": runner_fingerprint,
                          "inherited_environment_hash": inherited_hash,
                          "bootstrap_environment_hash": bootstrap_hash,
                          **_output_metadata(exc.stdout, exc.stderr),
                          "execution_enabled": True,
                          "fresh_execution": True, "fresh_tool_truth": False,
                          "authority": "tool_observation", "tool_truth": False},
            )
        except OSError as exc:
            return ToolRun(
                snapshot_id=request.snapshot_id, stage=request.stage, backend=self.name,
                request_hash=request_hash, toolchain_id=request.toolchain_id,
                status=RunStatus.FAILED, command=list(request.argv),
                working_directory=request.working_directory,
                environment_hash=request.environment_hash,
                input_artifact_ids=list(request.input_artifact_ids),
                failure_class=FailureClass.INFRASTRUCTURE,
                started_at=started_at, finished_at=utc_now(),
                elapsed_s=time.monotonic() - started,
                message=str(exc), metadata={**request.metadata,
                                             "runner_fingerprint": runner_fingerprint,
                                             "inherited_environment_hash": inherited_hash,
                                             "bootstrap_environment_hash": bootstrap_hash,
                                              **_output_metadata(None, None),
                                              "execution_enabled": True,
                                              "fresh_execution": False,
                                              "fresh_tool_truth": False,
                                              "authority": "tool_observation",
                                              "tool_truth": False},
            )


class SSHRunner(Runner):
    name = "runner.ssh"

    def __init__(self, host: str, remote_project_root: str, *, allow_execution: bool = False,
                 ssh_options: Sequence[str] = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=10")):
        if not host or host.startswith("-") or any(char.isspace() for char in host):
            raise ValueError("host must be a non-empty SSH destination without whitespace")
        if (not remote_project_root.startswith("/") or "\x00" in remote_project_root
                or any(part == ".." for part in remote_project_root.split("/"))):
            raise ValueError("remote_project_root must be an absolute normalized POSIX path")
        self.host = host
        self.remote_project_root = remote_project_root.rstrip("/") or "/"
        self.allow_execution = bool(allow_execution)
        self.ssh_options = tuple(ssh_options)

    @property
    def fingerprint(self) -> str:
        return stable_hash({"backend": self.name, "protocol": RUNNER_PROTOCOL_VERSION,
                            "host": self.host, "remote_root": self.remote_project_root,
                            "ssh_options": self.ssh_options,
                            "local_environment_hash": _environment_digest(os.environ)})

    @staticmethod
    def _remote_input_manifest(request: ToolRunRequest) -> list[dict[str, Any]]:
        value = request.metadata.get("input_artifacts")
        if not isinstance(value, list) or not value:
            raise RunnerProtocolError(
                "SSH execution requires metadata.input_artifacts for remote byte verification"
            )
        result: list[dict[str, Any]] = []
        for index, item in enumerate(value):
            if not isinstance(item, Mapping):
                raise RunnerProtocolError(f"input_artifacts[{index}] must be an object")
            identifier = str(item.get("id") or "")
            uri = safe_relative_path(str(item.get("uri") or ""), "remote artifact uri")
            digest = str(item.get("sha256") or "").lower()
            size = item.get("size")
            if (not identifier or not re.fullmatch(r"[0-9a-f]{64}", digest)
                    or not isinstance(size, int) or isinstance(size, bool) or size < 0):
                raise RunnerProtocolError(f"input_artifacts[{index}] is incomplete or invalid")
            result.append({"id": identifier, "uri": uri, "sha256": digest, "size": size})
        if (len({item["id"] for item in result}) != len(result)
                or len({item["uri"] for item in result}) != len(result)):
            raise RunnerProtocolError("remote input artifact IDs and URIs must be unique")
        if sorted(item["id"] for item in result) != sorted(request.input_artifact_ids):
            raise RunnerProtocolError(
                "remote input artifact manifest does not match input_artifact_ids"
            )
        return sorted(result, key=lambda item: (item["uri"], item["id"]))

    @staticmethod
    def _remote_attestation(request: ToolRunRequest) -> tuple[list[str], str]:
        expected = str(request.environment_hash or "").lower()
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise RunnerProtocolError(
                "SSH execution requires environment_hash as the expected SHA-256 "
                "of remote attestation stdout"
            )
        argv = request.metadata.get("remote_attestation_argv")
        if (not isinstance(argv, list) or not argv
                or not all(isinstance(item, str) and item for item in argv)):
            raise RunnerProtocolError(
                "SSH execution requires metadata.remote_attestation_argv as a non-empty argv list"
            )
        return list(argv), expected

    def execute(self, request: ToolRunRequest) -> ToolRun:
        if not self.allow_execution:
            return self._disabled(request, self.name, self.fingerprint)
        attestation_argv, expected_environment_hash = self._remote_attestation(request)
        inputs = self._remote_input_manifest(request)
        remote_cwd = self.remote_project_root
        if request.working_directory != ".":
            remote_cwd += "/" + request.working_directory
        env_prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in sorted(request.environment.items()))
        command = shlex.join(request.argv)
        remote_steps = ["set -o pipefail", f"cd {shlex.quote(remote_cwd)} || exit 86"]
        for item in inputs:
            path = shlex.quote(item["uri"])
            remote_steps.extend([
                f"test -f {path} || exit 86",
                f'test "$(wc -c < {path})" -eq {item["size"]} || exit 86',
                (f'''test "$(sha256sum -- {path} | awk '{{print $1}}')" = '''
                f'{shlex.quote(item["sha256"])} || exit 86'),
            ])
        attestation_command = shlex.join(attestation_argv)
        if env_prefix:
            attestation_command = env_prefix + " " + attestation_command
        remote_steps.extend([
            (f'''attestation_hash="$({attestation_command} | sha256sum | '''
             f'''awk '{{print $1}}')" || exit 87'''),
            (f'test "$attestation_hash" = '
             f'{shlex.quote(expected_environment_hash)} || exit 87'),
        ])
        remote_steps.append(((env_prefix + " ") if env_prefix else "") + command)
        remote = "; ".join(remote_steps)
        # OpenSSH joins arguments after the destination into one remote shell
        # command. Quote the *entire* bash -lc command argument so spaces and
        # shell metacharacters cannot change its boundary during that join.
        ssh_argv = ["ssh", *self.ssh_options, self.host, "bash", "-lc", shlex.quote(remote)]
        local_request = replace(request, argv=ssh_argv, working_directory=".",
                                nonzero_failure=FailureClass.SSH)
        # Execute from the current process directory; SSH owns the remote working directory.
        runner = LocalRunner(Path.cwd(), allow_execution=True)
        run = runner.execute(local_request)
        failure = run.failure_class
        if run.exit_code not in (None, 0):
            failure = (FailureClass.SSH if run.exit_code == 255 else
                       FailureClass.INPUT if run.exit_code == 86 else
                       FailureClass.INFRASTRUCTURE if run.exit_code == 87 else
                       request.nonzero_failure)
        remote_inputs_verified = (run.exit_code is not None
                                  and run.exit_code not in {86, 255})
        remote_environment_verified = (run.exit_code is not None
                                       and run.exit_code not in {86, 87, 255})
        tool_truth = remote_inputs_verified and remote_environment_verified
        fresh_execution = (run.exit_code is not None
                           and run.exit_code not in {86, 87, 255})
        return ToolRun(
            snapshot_id=request.snapshot_id, stage=request.stage, backend=self.name,
            request_hash=request.cache_key(self.fingerprint), toolchain_id=request.toolchain_id,
            status=run.status, command=list(request.argv),
            working_directory=request.working_directory,
            environment_hash=request.environment_hash,
            input_artifact_ids=list(request.input_artifact_ids),
            failure_class=failure, exit_code=run.exit_code,
            started_at=run.started_at, finished_at=run.finished_at, elapsed_s=run.elapsed_s,
            message=run.message,
            metadata={**request.metadata, **run.metadata,
                      "runner_fingerprint": self.fingerprint,
                      "ssh_host": self.host, "remote_project_root": self.remote_project_root,
                      "remote_inputs_verified": remote_inputs_verified,
                      "remote_environment_verified": remote_environment_verified,
                      "expected_remote_environment_hash": expected_environment_hash,
                      "fresh_execution": fresh_execution,
                      "fresh_tool_truth": fresh_execution and tool_truth,
                      "authority": "tool_observation", "tool_truth": tool_truth},
        )


@dataclass(slots=True)
class FakeOutcome:
    status: RunStatus = RunStatus.SUCCEEDED
    failure_class: FailureClass = FailureClass.NONE
    exit_code: int | None = 0
    gates: list[GateResult] = field(default_factory=list)
    message: str | None = None
    output_artifact_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class FakeRunner(Runner):
    name = "runner.fake"

    def __init__(self, script: Mapping[str, Sequence[FakeOutcome | Mapping[str, Any]]] | None = None):
        self.script: dict[str, deque[FakeOutcome | Mapping[str, Any]]] = defaultdict(deque)
        for stage, values in (script or {}).items():
            self.script[stage].extend(values)
        self.calls: list[ToolRunRequest] = []

    @property
    def fingerprint(self) -> str:
        return stable_hash({"backend": self.name, "protocol": RUNNER_PROTOCOL_VERSION,
                            "mode": "synthetic"})

    def execute(self, request: ToolRunRequest) -> ToolRun:
        self.calls.append(request)
        value = self.script[request.stage].popleft() if self.script[request.stage] else FakeOutcome()
        outcome = value if isinstance(value, FakeOutcome) else FakeOutcome(**dict(value))
        return ToolRun(
            snapshot_id=request.snapshot_id, stage=request.stage, backend=self.name,
            request_hash=request.cache_key(self.fingerprint), toolchain_id=request.toolchain_id,
            status=outcome.status, command=list(request.argv),
            working_directory=request.working_directory,
            environment_hash=request.environment_hash,
            input_artifact_ids=list(request.input_artifact_ids),
            output_artifact_ids=list(outcome.output_artifact_ids),
            gates=list(outcome.gates), failure_class=outcome.failure_class,
            exit_code=outcome.exit_code, started_at=utc_now(), finished_at=utc_now(),
            elapsed_s=0.0, message=outcome.message,
            metadata={"runner_fingerprint": self.fingerprint,
                      **request.metadata, **outcome.metadata,
                      "fresh_execution": False, "fresh_tool_truth": False,
                      "authority": "synthetic", "tool_truth": False},
        )


class ReplayRunner(Runner):
    name = "runner.replay"

    def __init__(self, runs: Mapping[str, ToolRun], *, source_runner_fingerprint: str):
        self.runs = dict(runs)
        self.source_runner_fingerprint = source_runner_fingerprint

    @property
    def fingerprint(self) -> str:
        return stable_hash({"backend": self.name, "protocol": RUNNER_PROTOCOL_VERSION,
                            "source_runner": self.source_runner_fingerprint})

    def execute(self, request: ToolRunRequest) -> ToolRun:
        key = request.cache_key(self.source_runner_fingerprint)
        original = self.runs.get(key)
        if original is None:
            raise CacheMiss(key)
        if (original.request_hash != key
                or original.snapshot_id != request.snapshot_id
                or original.stage != request.stage
                or original.metadata.get("runner_fingerprint") != self.source_runner_fingerprint
                or original.status not in {RunStatus.SUCCEEDED, RunStatus.CACHED}
                or original.failure_class != FailureClass.NONE
                or original.exit_code not in {None, 0}):
            # A failed/partial run is evidence of failure, never a successful
            # cache entry. Treat it as unavailable instead of laundering it to
            # the CACHED status.
            raise CacheMiss(key)
        return ToolRun(
            snapshot_id=request.snapshot_id, stage=request.stage, backend=self.name,
            request_hash=request.cache_key(self.fingerprint), toolchain_id=request.toolchain_id,
            status=RunStatus.CACHED, command=list(request.argv),
            working_directory=request.working_directory,
            environment_hash=request.environment_hash,
            input_artifact_ids=list(request.input_artifact_ids),
            output_artifact_ids=list(original.output_artifact_ids),
            diagnostics=list(original.diagnostics), gates=list(original.gates),
            failure_class=original.failure_class, exit_code=original.exit_code,
            started_at=original.started_at, finished_at=original.finished_at,
            elapsed_s=0.0, message=original.message,
            metadata={**original.metadata, "replayed_from_run_id": original.id,
                      "replayed_request_hash": original.request_hash,
                      "fresh_execution": False, "fresh_tool_truth": False,
                      "runner_fingerprint": self.fingerprint,
                      "authority": "replay", "tool_truth": False},
        )


@dataclass(slots=True)
class StageResult:
    runs: list[ToolRun]
    gates: dict[GateKind, GateStatus]
    correctness_checks: dict[str, GateStatus]
    gates_complete: bool
    tool_truth: bool
    verified: bool
    stopped_after_stage: str | None = None


class StageOrchestrator:
    def __init__(self, runner: Runner, *,
                 evidence_validator: Callable[[ToolRun, str], bool] | None = None):
        self.runner = runner
        # Gate IDs alone are references, not proof. Callers with a ledger or
        # an atomic run-result bundle can provide recursive evidence validation;
        # absence of such a validator is fail-closed for ``tool_truth``.
        self.evidence_validator = evidence_validator

    def _validate_response(self, request: ToolRunRequest, run: ToolRun,
                           runner_fingerprint: str, runner_name: str) -> None:
        if not isinstance(run, ToolRun):
            raise RunnerProtocolError("runner response must be a ToolRun")
        expected = {
            "snapshot_id": request.snapshot_id,
            "stage": request.stage,
            "backend": runner_name,
            "request_hash": request.cache_key(runner_fingerprint),
            "toolchain_id": request.toolchain_id,
            "command": list(request.argv),
            "working_directory": request.working_directory,
            "environment_hash": request.environment_hash,
            "input_artifact_ids": list(request.input_artifact_ids),
        }
        mismatches = [name for name, value in expected.items()
                      if getattr(run, name) != value]
        if run.metadata.get("runner_fingerprint") != runner_fingerprint:
            mismatches.append("runner_fingerprint")
        missing_metadata = object()
        for key, value in request.metadata.items():
            if run.metadata.get(key, missing_metadata) != value:
                mismatches.append(f"metadata.{key}")
        identity = json_ready(run)
        identity["id"] = ""
        if ToolRun.from_dict(identity).id != run.id:
            mismatches.append("id")
        if mismatches:
            raise RunnerProtocolError(
                "runner response identity mismatch: " + ", ".join(sorted(set(mismatches)))
            )
        if (len(set(run.input_artifact_ids)) != len(run.input_artifact_ids)
                or len(set(run.output_artifact_ids)) != len(run.output_artifact_ids)
                or set(run.input_artifact_ids) & set(run.output_artifact_ids)):
            raise RunnerProtocolError("runner response artifact lists are not unique and disjoint")
        if run.status in {RunStatus.SUCCEEDED, RunStatus.CACHED}:
            if run.failure_class != FailureClass.NONE or run.exit_code not in {None, 0}:
                raise RunnerProtocolError(
                    "successful/cached runner response has a failure class or non-zero exit"
                )
        elif run.status in {RunStatus.FAILED, RunStatus.SKIPPED}:
            if run.failure_class == FailureClass.NONE:
                raise RunnerProtocolError("failed/skipped runner response lacks a failure class")
        else:
            raise RunnerProtocolError("runner response must be terminal")
        if run.metadata.get("fresh_tool_truth") is True and (
            run.metadata.get("fresh_execution") is not True
            or run.metadata.get("tool_truth") is not True
            or str(run.backend).casefold() in {"runner.fake", "runner.replay"}
        ):
            raise RunnerProtocolError("fresh_tool_truth is inconsistent with runner provenance")
        for key in ("fresh_execution", "fresh_tool_truth", "tool_truth"):
            if not isinstance(run.metadata.get(key), bool):
                raise RunnerProtocolError(f"runner response metadata.{key} must be boolean")
        if run.status == RunStatus.CACHED and run.metadata["fresh_execution"] is True:
            raise RunnerProtocolError("cached runner response cannot claim fresh execution")

    def execute(self, requests: Sequence[ToolRunRequest]) -> StageResult:
        runs: list[ToolRun] = []
        gates: dict[GateKind, GateStatus] = {}
        correctness_checks: dict[str, GateStatus] = {}
        gate_truth: dict[GateKind, bool] = {}
        gate_evidence: dict[GateKind, bool] = {}
        correctness_truth: dict[str, bool] = {}
        correctness_evidence: dict[str, bool] = {}
        correctness_campaign: dict[str, str | None] = {}
        stopped: str | None = None
        execution_failed = False

        def merge_status(previous: GateStatus | None, current: GateStatus) -> GateStatus:
            # A FAIL can never be overwritten by a later PASS. UNKNOWN remains
            # conservative when no failure exists.
            priority = {GateStatus.PASS: 0, GateStatus.UNKNOWN: 1, GateStatus.FAIL: 2}
            if previous is None or priority[current] > priority[previous]:
                return current
            return previous

        for request in requests:
            runner_fingerprint = self.runner.fingerprint
            runner_name = self.runner.name
            # Runner is an untrusted SPI boundary.  Give it an isolated request
            # so in-place mutation cannot rewrite the expectation used below or
            # the caller's stage-selection/control-flow state.
            run = self.runner.execute(copy.deepcopy(request))
            self._validate_response(request, run, runner_fingerprint, runner_name)
            runs.append(run)
            if run.status not in {RunStatus.SUCCEEDED, RunStatus.CACHED}:
                execution_failed = True
                stopped = request.stage
                break
            for gate in run.gates:
                gates[gate.kind] = merge_status(gates.get(gate.kind), gate.status)
                evidence_present = (gate.status != GateStatus.PASS
                                    or bool(gate.evidence_ids))
                evidence_trusted = (
                    bool(gate.evidence_ids)
                    and self.evidence_validator is not None
                    and all(self.evidence_validator(run, item)
                            for item in gate.evidence_ids)
                )
                truth = (run.metadata.get("tool_truth") is True
                         and str(run.metadata.get("authority", "")) != "synthetic"
                         and (gate.status != GateStatus.PASS or evidence_trusted))
                gate_truth[gate.kind] = gate_truth.get(gate.kind, True) and truth
                gate_evidence[gate.kind] = (
                    gate_evidence.get(gate.kind, True) and evidence_present
                )
                if gate.kind == GateKind.CORRECTNESS and request.stage in {"csim", "rtl_cosim"}:
                    correctness_checks[request.stage] = merge_status(
                        correctness_checks.get(request.stage), gate.status,
                    )
                    correctness_truth[request.stage] = (
                        correctness_truth.get(request.stage, True) and truth
                    )
                    correctness_evidence[request.stage] = (
                        correctness_evidence.get(request.stage, True) and evidence_present
                    )
                    campaign = (run.metadata.get("campaign_id")
                                or run.metadata.get("workload_id"))
                    correctness_campaign[request.stage] = (
                        str(campaign) if isinstance(campaign, str) and campaign else None
                    )
            if any(gate.status == GateStatus.FAIL for gate in run.gates):
                stopped = request.stage
                break
        required = {GateKind.CORRECTNESS, GateKind.RESOURCE_FITS, GateKind.POST_ROUTE_TIMING}
        required_checks = {"csim", "rtl_cosim"}
        campaign_complete = (
            required_checks.issubset(correctness_campaign)
            and all(correctness_campaign[item] is not None for item in required_checks)
            and len({correctness_campaign[item] for item in required_checks}) == 1
        )
        gates_complete = (required.issubset(gates)
                          and required_checks.issubset(correctness_checks)
                          and required.issubset(gate_evidence)
                          and required_checks.issubset(correctness_evidence)
                          and all(gate_evidence[item] for item in required)
                          and all(correctness_evidence[item] for item in required_checks)
                          and campaign_complete
                          and all(gates[item] == GateStatus.PASS for item in required)
                          and all(correctness_checks[item] == GateStatus.PASS
                                  for item in required_checks))
        tool_truth = (required.issubset(gate_truth)
                      and required_checks.issubset(correctness_truth)
                      and all(gate_truth[item] for item in required)
                      and all(correctness_truth[item] for item in required_checks))
        verified = gates_complete and tool_truth and not execution_failed
        return StageResult(runs=runs, gates=gates,
                           correctness_checks=correctness_checks,
                           gates_complete=gates_complete, tool_truth=tool_truth,
                           verified=verified,
                           stopped_after_stage=stopped)
