"""Execution-location SPI for local, SSH, fake, and replay backends.

Runners do not know Vitis Tcl, reports, QoR, or correctness.  Toolchain adapters
construct argv and gates; runners only execute an immutable request.
"""
from __future__ import annotations

import abc
import base64
import copy
import hashlib
import json
import math
import os
import re
import signal
import shlex
import subprocess
import tempfile
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
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
    stable_hash,
    utc_now,
)
from .staging import (
    DEFAULT_MAX_OUTPUT_BYTES,
    MAX_TOTAL_TRANSFER_BYTES,
    StagingError,
    create_run_directory,
    read_verified_file,
    remove_run_directory,
    runner_relative_path,
    write_new_file,
)


PROTOCOL_VERSION = "hlsgraph.runner.v2"
# Compatibility spelling retained for v0.x callers.  It intentionally points
# at v2; requests cannot opt back into the unsafe in-place v1 protocol.
RUNNER_PROTOCOL_VERSION = PROTOCOL_VERSION

# Values below are measurements made by a Runner, not caller-provided request
# context.  Reserving them prevents a request from pre-seeding provenance that
# a backend would otherwise have to overwrite.
RUNNER_MEASURED_METADATA_KEYS = frozenset({
    "authority", "bootstrap_environment_hash", "execution_enabled",
    "expected_remote_environment_hash",
    "failure_type", "fresh_execution", "fresh_tool_truth", "inherited_environment_hash",
    "input_mismatch_ids", "input_validation_failed", "output_embedded",
    "remote_environment_verified", "remote_inputs_verified",
    "resource_guard_checked", "resource_guard_configured", "resource_guard_passed",
    "runtime_guard_checked", "runtime_guard_configured", "runtime_guard_passed",
    "runtime_guard_triggered", "runtime_guard_exit_code",
    "runtime_guard_failure_class", "runtime_guard_process_group_pid",
    "runtime_resource_guard", "runtime_resource_monitor",
    "remote_project_root", "replayed_from_run_id", "replayed_request_hash",
    "runner_fingerprint", "ssh_host", "stderr_bytes", "stderr_sha256",
    "snapshot_stale", "staged_output_manifest", "staging_isolated",
    "stdout_bytes", "stdout_sha256", "tool_truth",
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


def _protocol_path(value: str, field_name: str) -> str:
    try:
        return runner_relative_path(value, field_name)
    except StagingError as exc:
        raise RunnerProtocolError(str(exc)) from exc


@dataclass(frozen=True, slots=True)
class ResourceGuard:
    """Explicit runner-owned preflight command for host resource availability.

    The command's exit status is the complete contract: stdout/stderr are never
    parsed or persisted.  This keeps an ordinary HLS tool or report from
    relabeling a design/correctness failure as an infrastructure guard event.
    """

    argv: tuple[str, ...]
    timeout_s: float = 30.0

    def __post_init__(self) -> None:
        if isinstance(self.argv, (str, bytes)):
            raise RunnerProtocolError("resource guard argv must be a sequence, not a string")
        values = tuple(self.argv)
        if (not values or not all(
                isinstance(item, str) and item and "\x00" not in item
                for item in values)):
            raise RunnerProtocolError(
                "resource guard argv must be non-empty strings without NUL"
            )
        object.__setattr__(self, "argv", values)
        if (not isinstance(self.timeout_s, (int, float))
                or isinstance(self.timeout_s, bool)
                or not math.isfinite(float(self.timeout_s))
                or float(self.timeout_s) <= 0):
            raise RunnerProtocolError("resource guard timeout_s must be positive and finite")

    def identity_payload(self) -> dict[str, Any]:
        return {"argv": list(self.argv), "timeout_s": float(self.timeout_s)}


@dataclass(frozen=True, slots=True)
class ResourceGuardResult:
    """Structured result created by a trusted runner, never request metadata."""

    checked: bool
    passed: bool
    exit_code: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.checked, bool) or not isinstance(self.passed, bool):
            raise RunnerProtocolError("resource guard checked/passed must be boolean")
        if self.passed and not self.checked:
            raise RunnerProtocolError("an unchecked resource guard cannot pass")
        if (self.exit_code is not None
                and (not isinstance(self.exit_code, int) or isinstance(self.exit_code, bool))):
            raise RunnerProtocolError("resource guard exit_code must be an integer or None")
        if not self.checked and self.exit_code is not None:
            raise RunnerProtocolError("an unchecked resource guard cannot have an exit code")
        if self.passed and self.exit_code != 0:
            raise RunnerProtocolError("a passing resource guard requires exit code zero")
        if self.checked and not self.passed and self.exit_code == 0:
            raise RunnerProtocolError("a rejected resource guard cannot have exit code zero")


def _resource_guard_metadata(
    result: ResourceGuardResult | None,
) -> dict[str, bool]:
    return {
        "resource_guard_configured": result is not None,
        "resource_guard_checked": bool(result and result.checked),
        "resource_guard_passed": bool(result and result.passed),
    }


def _coerce_resource_guard(
    value: ResourceGuard | Mapping[str, Any] | None,
) -> ResourceGuard | None:
    if value is None or isinstance(value, ResourceGuard):
        return value
    if not isinstance(value, Mapping):
        raise RunnerProtocolError("resource_guard must be a ResourceGuard, mapping, or None")
    try:
        return ResourceGuard(**dict(value))
    except TypeError as exc:
        raise RunnerProtocolError(f"invalid resource_guard configuration: {exc}") from exc


PROCESS_GROUP_PID_TOKEN = "{process_group_pid}"


@dataclass(frozen=True, slots=True)
class RuntimeResourceMonitor:
    """Runner-owned resource probe executed while the tool process group runs.

    ``PROCESS_GROUP_PID_TOKEN`` must be one complete argv item.  The runner
    replaces it with the process-group leader PID without invoking a shell.
    Probe output is discarded; only its exit status is part of the contract.
    By default every non-zero exit is an infrastructure resource-guard event.
    A runner owner may explicitly classify selected exit codes as a trusted
    design-process-group memory/resource limit.
    """

    argv: tuple[str, ...]
    interval_s: float = 5.0
    timeout_s: float = 30.0
    resource_exit_codes: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.argv, (str, bytes)):
            raise RunnerProtocolError(
                "runtime resource monitor argv must be a sequence, not a string"
            )
        values = tuple(self.argv)
        if (not values or not all(
                isinstance(item, str) and item and "\x00" not in item
                for item in values)):
            raise RunnerProtocolError(
                "runtime resource monitor argv must be non-empty strings without NUL"
            )
        if values.count(PROCESS_GROUP_PID_TOKEN) != 1:
            raise RunnerProtocolError(
                "runtime resource monitor argv must contain exactly one "
                f"{PROCESS_GROUP_PID_TOKEN!r} item"
            )
        object.__setattr__(self, "argv", values)
        for name, value in (("interval_s", self.interval_s), ("timeout_s", self.timeout_s)):
            if (not isinstance(value, (int, float)) or isinstance(value, bool)
                    or not math.isfinite(float(value)) or float(value) <= 0):
                raise RunnerProtocolError(
                    f"runtime resource monitor {name} must be positive and finite"
                )
        codes = tuple(self.resource_exit_codes)
        if any(not isinstance(code, int) or isinstance(code, bool) or code == 0
               for code in codes):
            raise RunnerProtocolError(
                "runtime resource monitor resource_exit_codes must be non-zero integers"
            )
        object.__setattr__(self, "resource_exit_codes", tuple(sorted(set(codes))))

    def argv_for_process_group(self, process_group_pid: int) -> list[str]:
        if (not isinstance(process_group_pid, int) or isinstance(process_group_pid, bool)
                or process_group_pid <= 0):
            raise RunnerProtocolError("tool process-group PID must be a positive integer")
        return [str(process_group_pid) if item == PROCESS_GROUP_PID_TOKEN else item
                for item in self.argv]

    def failure_for_exit(self, exit_code: int | None) -> FailureClass:
        return (FailureClass.RESOURCE
                if exit_code in self.resource_exit_codes
                else FailureClass.INFRA_RESOURCE_GUARD)

    def identity_payload(self) -> dict[str, Any]:
        return {
            "argv": list(self.argv),
            "interval_s": float(self.interval_s),
            "timeout_s": float(self.timeout_s),
            "resource_exit_codes": list(self.resource_exit_codes),
        }


@dataclass(frozen=True, slots=True)
class RuntimeResourceMonitorResult:
    """Structured runtime-monitor result emitted only by a trusted Runner."""

    checked: bool
    passed: bool
    triggered: bool
    exit_code: int | None = None
    failure_class: FailureClass | None = None

    def __post_init__(self) -> None:
        if any(not isinstance(value, bool)
               for value in (self.checked, self.passed, self.triggered)):
            raise RunnerProtocolError(
                "runtime resource monitor checked/passed/triggered must be boolean"
            )
        if self.passed and (not self.checked or self.triggered):
            raise RunnerProtocolError(
                "a passing runtime resource monitor must be checked and not triggered"
            )
        if self.triggered and (not self.checked or self.passed):
            raise RunnerProtocolError(
                "a triggered runtime resource monitor must be checked and not pass"
            )
        if (self.exit_code is not None
                and (not isinstance(self.exit_code, int) or isinstance(self.exit_code, bool))):
            raise RunnerProtocolError(
                "runtime resource monitor exit_code must be an integer or None"
            )
        if not self.checked and self.exit_code is not None:
            raise RunnerProtocolError(
                "an unchecked runtime resource monitor cannot have an exit code"
            )
        if self.passed and self.exit_code != 0:
            raise RunnerProtocolError(
                "a passing runtime resource monitor requires exit code zero"
            )
        if self.triggered and self.exit_code == 0:
            raise RunnerProtocolError(
                "a triggered runtime resource monitor cannot have exit code zero"
            )
        failure = self.failure_class
        if failure is not None and not isinstance(failure, FailureClass):
            try:
                failure = FailureClass(failure)
            except ValueError as exc:
                raise RunnerProtocolError(
                    "invalid runtime resource monitor failure_class"
                ) from exc
            object.__setattr__(self, "failure_class", failure)
        if self.triggered:
            if failure not in {
                    FailureClass.INFRA_RESOURCE_GUARD,
                    FailureClass.INFRASTRUCTURE,
                    FailureClass.RESOURCE,
            }:
                raise RunnerProtocolError(
                    "triggered runtime monitor requires a trusted guard/resource failure class"
                )
        elif failure is not None:
            raise RunnerProtocolError(
                "an untriggered runtime monitor cannot have a failure class"
            )


def _runtime_guard_metadata(
    result: RuntimeResourceMonitorResult | None,
) -> dict[str, bool]:
    return {
        "runtime_guard_configured": result is not None,
        "runtime_guard_checked": bool(result and result.checked),
        "runtime_guard_passed": bool(result and result.passed),
        "runtime_guard_triggered": bool(result and result.triggered),
    }


def _coerce_runtime_resource_monitor(
    value: RuntimeResourceMonitor | Mapping[str, Any] | None,
) -> RuntimeResourceMonitor | None:
    if value is None or isinstance(value, RuntimeResourceMonitor):
        return value
    if not isinstance(value, Mapping):
        raise RunnerProtocolError(
            "runtime_resource_monitor must be a RuntimeResourceMonitor, mapping, or None"
        )
    try:
        return RuntimeResourceMonitor(**dict(value))
    except TypeError as exc:
        raise RunnerProtocolError(
            f"invalid runtime_resource_monitor configuration: {exc}"
        ) from exc


@dataclass(frozen=True, slots=True)
class RunnerInput:
    """One immutable artifact copied into a run-scoped staging directory."""

    artifact_id: str
    source_path: str
    staged_path: str
    sha256: str
    size: int

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_id, str) or not self.artifact_id:
            raise RunnerProtocolError("runner input artifact_id is required")
        object.__setattr__(self, "source_path",
                           _protocol_path(self.source_path, "runner input source_path"))
        object.__setattr__(self, "staged_path",
                           _protocol_path(self.staged_path, "runner input staged_path"))
        digest = str(self.sha256).casefold()
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise RunnerProtocolError("runner input sha256 must contain 64 hexadecimal characters")
        object.__setattr__(self, "sha256", digest)
        if not isinstance(self.size, int) or isinstance(self.size, bool) or self.size < 0:
            raise RunnerProtocolError("runner input size must be a non-negative integer")

    def identity_payload(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id, "source_path": self.source_path,
            "staged_path": self.staged_path, "sha256": self.sha256, "size": self.size,
        }


@dataclass(frozen=True, slots=True)
class DeclaredOutput:
    """A path the runner is allowed to return from its staging directory."""

    path: str
    required: bool = True
    max_bytes: int = DEFAULT_MAX_OUTPUT_BYTES

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _protocol_path(self.path, "declared output path"))
        if not isinstance(self.required, bool):
            raise RunnerProtocolError("declared output required must be boolean")
        if (not isinstance(self.max_bytes, int) or isinstance(self.max_bytes, bool)
                or self.max_bytes < 0):
            raise RunnerProtocolError("declared output max_bytes must be non-negative")

    def identity_payload(self) -> dict[str, Any]:
        return {"path": self.path, "required": self.required, "max_bytes": self.max_bytes}


@dataclass(frozen=True, slots=True)
class StagedOutput:
    """Verified bytes retained locally until the SDK atomically commits them."""

    path: str
    local_path: Path
    size: int
    sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _protocol_path(self.path, "staged output path"))
        object.__setattr__(self, "local_path", Path(self.local_path).absolute())
        if not isinstance(self.size, int) or isinstance(self.size, bool) or self.size < 0:
            raise RunnerProtocolError("staged output size must be non-negative")
        digest = str(self.sha256).casefold()
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise RunnerProtocolError("staged output sha256 must contain 64 hexadecimal characters")
        object.__setattr__(self, "sha256", digest)


@dataclass(slots=True)
class RunnerExecution:
    """A terminal run event plus its still-isolated, verified output bytes.

    ``cleanup`` is idempotent.  Direct runner users own this lifetime; the SDK
    always cleans it after committing or recording failure.
    """

    run: ToolRun
    staged_outputs: list[StagedOutput] = field(default_factory=list)
    staging_directory: Path | None = None
    staging_parent: Path | None = None
    resource_guard: ResourceGuardResult | None = None
    runtime_resource_monitor: RuntimeResourceMonitorResult | None = None
    _cleaned: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.run, ToolRun):
            raise RunnerProtocolError("RunnerExecution.run must be a ToolRun")
        if any(not isinstance(item, StagedOutput) for item in self.staged_outputs):
            raise RunnerProtocolError("RunnerExecution.staged_outputs must contain StagedOutput")
        if (self.resource_guard is not None
                and not isinstance(self.resource_guard, ResourceGuardResult)):
            raise RunnerProtocolError(
                "RunnerExecution.resource_guard must be a ResourceGuardResult or None"
            )
        if (self.runtime_resource_monitor is not None
                and not isinstance(
                    self.runtime_resource_monitor, RuntimeResourceMonitorResult,
                )):
            raise RunnerProtocolError(
                "RunnerExecution.runtime_resource_monitor must be a "
                "RuntimeResourceMonitorResult or None"
            )
        if self.staging_directory is not None:
            self.staging_directory = Path(self.staging_directory).absolute()
        if self.staging_parent is not None:
            self.staging_parent = Path(self.staging_parent).absolute()

    def cleanup(self) -> None:
        if self._cleaned:
            return
        if self.staging_directory is not None:
            if self.staging_parent is None:
                raise RunnerProtocolError("staging_parent is required when staging_directory is set")
            remove_run_directory(self.staging_directory, self.staging_parent)
        self._cleaned = True

    def __enter__(self) -> "RunnerExecution":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.cleanup()

    def __getattr__(self, name: str) -> Any:
        # Read-only compatibility for v0.1 callers that inspected the ToolRun
        # returned by execute().  New code should use ``execution.run``.
        return getattr(self.run, name)


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
    inputs: list[RunnerInput] = field(default_factory=list)
    declared_outputs: list[DeclaredOutput] = field(default_factory=list)
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
            self.working_directory = _protocol_path(
                self.working_directory, "working_directory",
            )
        conflicts = sorted(set(self.metadata) & RUNNER_MEASURED_METADATA_KEYS)
        if conflicts:
            raise RunnerProtocolError(
                "request metadata uses runner-measured keys: " + ", ".join(conflicts)
            )
        self.nonzero_failure = (self.nonzero_failure if isinstance(self.nonzero_failure, FailureClass)
                                else FailureClass(self.nonzero_failure))
        if self.nonzero_failure in {
                FailureClass.INFRA_RESOURCE_GUARD, FailureClass.RESOURCE}:
            raise RunnerProtocolError(
                "resource guard failure classes are runner-owned and cannot be a "
                "stage nonzero_failure"
            )
        self.inputs = [item if isinstance(item, RunnerInput) else RunnerInput(**dict(item))
                       for item in self.inputs]
        self.declared_outputs = [
            item if isinstance(item, DeclaredOutput) else DeclaredOutput(**dict(item))
            for item in self.declared_outputs
        ]
        input_ids = [item.artifact_id for item in self.inputs]
        if sorted(input_ids) != sorted(self.input_artifact_ids):
            raise RunnerProtocolError(
                "runner inputs must exactly match input_artifact_ids"
            )
        input_paths = [item.staged_path for item in self.inputs]
        output_paths = [item.path for item in self.declared_outputs]
        if len(set(input_ids)) != len(input_ids):
            raise RunnerProtocolError("runner input artifact IDs must be unique")
        if len(set(input_paths)) != len(input_paths):
            raise RunnerProtocolError("runner input staged paths must be unique")
        if len(set(output_paths)) != len(output_paths):
            raise RunnerProtocolError("declared output paths must be unique")
        if set(input_paths) & set(output_paths):
            raise RunnerProtocolError("runner input and declared output paths must be disjoint")
        all_file_paths = [*input_paths, *output_paths]
        for index, path in enumerate(all_file_paths):
            if any(other.startswith(path + "/") or path.startswith(other + "/")
                   for other in all_file_paths[index + 1:]):
                raise RunnerProtocolError(
                    "runner input/output file paths cannot contain one another"
                )

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
            "input_manifest": sorted(
                (item.identity_payload() for item in self.inputs),
                key=lambda item: (item["staged_path"], item["artifact_id"]),
            ),
            "declared_outputs": [
                item.identity_payload() for item in sorted(
                    self.declared_outputs, key=lambda item: item.path,
                )
            ],
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
    run_scoped_staging: bool = False
    can_produce_tool_truth: bool = False
    can_report_resource_guard: bool = False
    can_report_runtime_resource_guard: bool = False

    @property
    @abc.abstractmethod
    def fingerprint(self) -> str:
        raise NotImplementedError

    def capabilities(self) -> dict[str, Any]:
        return {"name": self.name, "fingerprint": self.fingerprint,
                "protocol_version": RUNNER_PROTOCOL_VERSION,
                "provides_local_output_bytes": self.provides_local_output_bytes,
                "run_scoped_staging": self.run_scoped_staging,
                "declared_outputs_only": True,
                "can_produce_tool_truth": self.can_produce_tool_truth,
                "can_report_resource_guard": self.can_report_resource_guard,
                "can_report_runtime_resource_guard": (
                    self.can_report_runtime_resource_guard
                )}

    @abc.abstractmethod
    def execute(self, request: ToolRunRequest) -> RunnerExecution:
        raise NotImplementedError

    @staticmethod
    def _disabled(
        request: ToolRunRequest, backend: str, fingerprint: str, *,
        resource_guard: ResourceGuardResult | None = None,
        runtime_resource_monitor: RuntimeResourceMonitorResult | None = None,
    ) -> RunnerExecution:
        request_hash = request.cache_key(fingerprint)
        event_time = utc_now()
        run = ToolRun(
            snapshot_id=request.snapshot_id, stage=request.stage, backend=backend,
            request_hash=request_hash, toolchain_id=request.toolchain_id,
            status=RunStatus.SKIPPED, command=list(request.argv),
            working_directory=request.working_directory,
            environment_hash=request.environment_hash,
            input_artifact_ids=list(request.input_artifact_ids),
            failure_class=FailureClass.UNSUPPORTED,
            started_at=event_time, finished_at=event_time, elapsed_s=0.0,
            message="execution is disabled; enable it explicitly on the runner",
            metadata={**request.metadata, **_resource_guard_metadata(resource_guard),
                      **_runtime_guard_metadata(runtime_resource_monitor),
                      "runner_fingerprint": fingerprint, "execution_enabled": False,
                      "fresh_execution": False, "fresh_tool_truth": False,
                      "tool_truth": False},
        )
        return RunnerExecution(
            run, resource_guard=resource_guard,
            runtime_resource_monitor=runtime_resource_monitor,
        )


@dataclass(frozen=True, slots=True)
class _MonitoredProcessResult:
    returncode: int | None
    stdout: bytes
    stderr: bytes
    timed_out: bool
    monitor: RuntimeResourceMonitorResult


def _terminate_process_group(process: subprocess.Popen[Any]) -> None:
    """Best-effort bounded termination of the tool's complete process group."""

    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=10.0, check=False, shell=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            try:
                process.kill()
            except OSError:
                pass
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass
        try:
            process.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            pass
        # The group can still contain descendants after its leader exits.
        # Always follow the grace period with a group-wide hard stop.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass
    try:
        process.wait(timeout=5.0)
    except (OSError, subprocess.TimeoutExpired):
        try:
            process.kill()
            process.wait(timeout=2.0)
        except (OSError, subprocess.TimeoutExpired):
            pass


def _run_monitored_process(
    argv: Sequence[str], *, cwd: Path, env: Mapping[str, str], timeout_s: float,
    monitor: RuntimeResourceMonitor,
    monitor_environment: Mapping[str, str],
) -> _MonitoredProcessResult:
    """Run one tool in an isolated process group and poll a trusted monitor."""

    popen_group: dict[str, Any]
    if os.name == "nt":
        popen_group = {
            "creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200),
        }
    else:
        popen_group = {"start_new_session": True}
    with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
        process = subprocess.Popen(
            list(argv), cwd=cwd, env=dict(env), stdout=stdout_file,
            stderr=stderr_file, shell=False, **popen_group,
        )
        deadline = time.monotonic() + float(timeout_s)
        next_check = time.monotonic()
        monitor_result = RuntimeResourceMonitorResult(False, False, False)
        timed_out = False
        while True:
            now = time.monotonic()
            if now >= next_check:
                try:
                    probe = subprocess.run(
                        monitor.argv_for_process_group(process.pid),
                        cwd=cwd, env=dict(monitor_environment),
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL, timeout=monitor.timeout_s,
                        check=False, shell=False,
                    )
                except subprocess.TimeoutExpired:
                    monitor_result = RuntimeResourceMonitorResult(
                        True, False, True, None,
                        FailureClass.INFRA_RESOURCE_GUARD,
                    )
                except OSError:
                    monitor_result = RuntimeResourceMonitorResult(
                        True, False, True, None, FailureClass.INFRASTRUCTURE,
                    )
                else:
                    if probe.returncode == 0:
                        monitor_result = RuntimeResourceMonitorResult(
                            True, True, False, 0,
                        )
                    else:
                        monitor_result = RuntimeResourceMonitorResult(
                            True, False, True, probe.returncode,
                            monitor.failure_for_exit(probe.returncode),
                        )
                if monitor_result.triggered:
                    _terminate_process_group(process)
                    break
                next_check = time.monotonic() + float(monitor.interval_s)
            if process.poll() is not None:
                break
            now = time.monotonic()
            if now >= deadline:
                timed_out = True
                _terminate_process_group(process)
                break
            sleep_s = min(0.05, max(0.001, next_check - now), max(0.001, deadline - now))
            time.sleep(sleep_s)
        if process.poll() is None:
            _terminate_process_group(process)
        stdout_file.seek(0)
        stderr_file.seek(0)
        stdout = stdout_file.read()
        stderr = stderr_file.read()
        return _MonitoredProcessResult(
            process.returncode, stdout, stderr, timed_out, monitor_result,
        )


class LocalRunner(Runner):
    name = "runner.local"
    provides_local_output_bytes = True
    run_scoped_staging = True
    can_produce_tool_truth = True

    def __init__(
        self, project_root: str | Path, *, allow_execution: bool = False,
        inherit_environment: bool = True,
        resource_guard: ResourceGuard | Mapping[str, Any] | None = None,
        runtime_resource_monitor: (
            RuntimeResourceMonitor | Mapping[str, Any] | None
        ) = None,
    ):
        self.project_root = Path(project_root).resolve()
        self.allow_execution = bool(allow_execution)
        self.inherit_environment = bool(inherit_environment)
        self.resource_guard = _coerce_resource_guard(resource_guard)
        self.runtime_resource_monitor = _coerce_runtime_resource_monitor(
            runtime_resource_monitor
        )

    @property
    def can_report_resource_guard(self) -> bool:  # type: ignore[override]
        return self.resource_guard is not None

    @property
    def can_report_runtime_resource_guard(self) -> bool:  # type: ignore[override]
        return self.runtime_resource_monitor is not None

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
                            "bootstrap_environment_hash": bootstrap_hash,
                            "resource_guard": (
                                stable_hash(self.resource_guard.identity_payload())
                                if self.resource_guard is not None else None
                            ),
                            "runtime_resource_monitor": (
                                stable_hash(
                                    self.runtime_resource_monitor.identity_payload()
                                )
                                if self.runtime_resource_monitor is not None else None
                            )})

    def _guard_failure_execution(
        self, request: ToolRunRequest, *, runner_fingerprint: str,
        result: ResourceGuardResult, failure_class: FailureClass,
        message: str, started_at: str, started: float, staging: Path,
        staging_parent: Path, inherited_hash: str | None,
        bootstrap_hash: str | None,
    ) -> RunnerExecution:
        runtime_result = (
            RuntimeResourceMonitorResult(False, False, False)
            if self.runtime_resource_monitor is not None else None
        )
        run = ToolRun(
            snapshot_id=request.snapshot_id, stage=request.stage, backend=self.name,
            request_hash=request.cache_key(runner_fingerprint),
            toolchain_id=request.toolchain_id, status=RunStatus.FAILED,
            command=list(request.argv), working_directory=request.working_directory,
            environment_hash=request.environment_hash,
            input_artifact_ids=list(request.input_artifact_ids),
            failure_class=failure_class, exit_code=result.exit_code,
            started_at=started_at, finished_at=utc_now(),
            elapsed_s=time.monotonic() - started, message=message,
            metadata={
                **request.metadata, **_resource_guard_metadata(result),
                **_runtime_guard_metadata(runtime_result),
                "runner_fingerprint": runner_fingerprint,
                "inherited_environment_hash": inherited_hash,
                "bootstrap_environment_hash": bootstrap_hash,
                **_output_metadata(None, None), "execution_enabled": True,
                "fresh_execution": False, "fresh_tool_truth": False,
                "authority": "infrastructure", "tool_truth": False,
                "staging_isolated": True, "staged_output_manifest": [],
            },
        )
        return RunnerExecution(
            run, staging_directory=staging, staging_parent=staging_parent,
            resource_guard=result, runtime_resource_monitor=runtime_result,
        )

    def _runtime_guard_failure_execution(
        self, request: ToolRunRequest, *, runner_fingerprint: str,
        resource_guard: ResourceGuardResult | None,
        result: RuntimeResourceMonitorResult, started_at: str, started: float,
        staging: Path, staging_parent: Path, inherited_hash: str | None,
        bootstrap_hash: str | None, stdout: bytes, stderr: bytes,
    ) -> RunnerExecution:
        failure = result.failure_class or FailureClass.INFRA_RESOURCE_GUARD
        run = ToolRun(
            snapshot_id=request.snapshot_id, stage=request.stage, backend=self.name,
            request_hash=request.cache_key(runner_fingerprint),
            toolchain_id=request.toolchain_id, status=RunStatus.FAILED,
            command=list(request.argv), working_directory=request.working_directory,
            environment_hash=request.environment_hash,
            input_artifact_ids=list(request.input_artifact_ids),
            failure_class=failure, exit_code=result.exit_code,
            started_at=started_at, finished_at=utc_now(),
            elapsed_s=time.monotonic() - started,
            message=("runtime resource monitor reported a design process-group "
                     "resource limit" if failure == FailureClass.RESOURCE else
                     "runtime resource monitor could not execute" if
                     failure == FailureClass.INFRASTRUCTURE else
                     "runtime resource monitor rejected execution"),
            metadata={
                **request.metadata, **_resource_guard_metadata(resource_guard),
                **_runtime_guard_metadata(result),
                "runner_fingerprint": runner_fingerprint,
                "inherited_environment_hash": inherited_hash,
                "bootstrap_environment_hash": bootstrap_hash,
                **_output_metadata(stdout, stderr), "execution_enabled": True,
                "fresh_execution": True, "fresh_tool_truth": False,
                "authority": "infrastructure", "tool_truth": False,
                "staging_isolated": True, "staged_output_manifest": [],
            },
        )
        return RunnerExecution(
            run, staging_directory=staging, staging_parent=staging_parent,
            resource_guard=resource_guard, runtime_resource_monitor=result,
        )

    def execute(self, request: ToolRunRequest) -> RunnerExecution:
        if not self.allow_execution:
            guard = (ResourceGuardResult(False, False)
                     if self.resource_guard is not None else None)
            runtime_monitor = (
                RuntimeResourceMonitorResult(False, False, False)
                if self.runtime_resource_monitor is not None else None
            )
            return self._disabled(
                request, self.name, self.fingerprint, resource_guard=guard,
                runtime_resource_monitor=runtime_monitor,
            )
        guard_result = (ResourceGuardResult(False, False)
                        if self.resource_guard is not None else None)
        runtime_result = (
            RuntimeResourceMonitorResult(False, False, False)
            if self.runtime_resource_monitor is not None else None
        )
        try:
            staging, staging_parent = create_run_directory(self.project_root)
            for item in request.inputs:
                data, _size, _digest, _source = read_verified_file(
                    self.project_root, item.source_path, expected_size=item.size,
                    expected_sha256=item.sha256, max_bytes=item.size,
                )
                write_new_file(staging, item.staged_path, data)
            cwd = staging if request.working_directory == "." else staging.joinpath(
                *runner_relative_path(
                    request.working_directory, "working_directory",
                ).split("/")
            )
            cwd.mkdir(parents=True, exist_ok=True)
        except (OSError, StagingError) as exc:
            event_time = utc_now()
            failure = (FailureClass.INPUT if isinstance(exc, StagingError)
                       else FailureClass.INFRASTRUCTURE)
            run = ToolRun(
                snapshot_id=request.snapshot_id, stage=request.stage, backend=self.name,
                request_hash=request.cache_key(self.fingerprint),
                toolchain_id=request.toolchain_id, status=RunStatus.FAILED,
                command=list(request.argv), working_directory=request.working_directory,
                environment_hash=request.environment_hash,
                input_artifact_ids=list(request.input_artifact_ids),
                failure_class=failure,
                started_at=event_time, finished_at=event_time, elapsed_s=0.0,
                message=str(exc), metadata={
                    **request.metadata, **_resource_guard_metadata(guard_result),
                    **_runtime_guard_metadata(runtime_result),
                    "runner_fingerprint": self.fingerprint,
                    **_output_metadata(None, None), "execution_enabled": True,
                    "fresh_execution": False, "fresh_tool_truth": False,
                    "authority": "tool_observation", "tool_truth": False,
                    "staging_isolated": True, "staged_output_manifest": [],
                },
            )
            # create_run_directory may have failed before assigning either
            # variable, so only expose a lifetime when it actually exists.
            return RunnerExecution(
                run,
                staging_directory=locals().get("staging"),
                staging_parent=locals().get("staging_parent"),
                resource_guard=guard_result,
                runtime_resource_monitor=runtime_result,
            )
        bootstrap = ({}
                     if self.inherit_environment
                     else _local_bootstrap_environment())
        inherited_hash = (_environment_digest(os.environ)
                          if self.inherit_environment else None)
        bootstrap_hash = _environment_digest(bootstrap) if bootstrap else None
        env = dict(os.environ) if self.inherit_environment else dict(bootstrap)
        monitor_environment = dict(env)
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
        if self.resource_guard is not None:
            try:
                guard_process = subprocess.run(
                    list(self.resource_guard.argv), cwd=cwd, env=env,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    timeout=self.resource_guard.timeout_s, shell=False,
                )
            except subprocess.TimeoutExpired:
                guard_result = ResourceGuardResult(True, False, None)
                return self._guard_failure_execution(
                    request, runner_fingerprint=runner_fingerprint,
                    result=guard_result,
                    failure_class=FailureClass.INFRA_RESOURCE_GUARD,
                    message="resource guard timed out and rejected execution",
                    started_at=started_at, started=started,
                    staging=staging, staging_parent=staging_parent,
                    inherited_hash=inherited_hash, bootstrap_hash=bootstrap_hash,
                )
            except OSError as exc:
                guard_result = ResourceGuardResult(False, False, None)
                return self._guard_failure_execution(
                    request, runner_fingerprint=runner_fingerprint,
                    result=guard_result,
                    failure_class=FailureClass.INFRASTRUCTURE,
                    message=f"resource guard could not start: {exc}",
                    started_at=started_at, started=started,
                    staging=staging, staging_parent=staging_parent,
                    inherited_hash=inherited_hash, bootstrap_hash=bootstrap_hash,
                )
            if guard_process.returncode != 0:
                guard_result = ResourceGuardResult(
                    True, False, guard_process.returncode,
                )
                return self._guard_failure_execution(
                    request, runner_fingerprint=runner_fingerprint,
                    result=guard_result,
                    failure_class=FailureClass.INFRA_RESOURCE_GUARD,
                    message="resource guard rejected execution",
                    started_at=started_at, started=started,
                    staging=staging, staging_parent=staging_parent,
                    inherited_hash=inherited_hash, bootstrap_hash=bootstrap_hash,
                )
            guard_result = ResourceGuardResult(True, True, 0)
        try:
            if self.runtime_resource_monitor is not None:
                monitored = _run_monitored_process(
                    request.argv, cwd=cwd, env=env, timeout_s=request.timeout_s,
                    monitor=self.runtime_resource_monitor,
                    monitor_environment=monitor_environment,
                )
                runtime_result = monitored.monitor
                if runtime_result.triggered:
                    return self._runtime_guard_failure_execution(
                        request, runner_fingerprint=runner_fingerprint,
                        resource_guard=guard_result, result=runtime_result,
                        started_at=started_at, started=started, staging=staging,
                        staging_parent=staging_parent,
                        inherited_hash=inherited_hash,
                        bootstrap_hash=bootstrap_hash,
                        stdout=monitored.stdout, stderr=monitored.stderr,
                    )
                if monitored.timed_out:
                    raise subprocess.TimeoutExpired(
                        request.argv, request.timeout_s,
                        output=monitored.stdout, stderr=monitored.stderr,
                    )
                process = subprocess.CompletedProcess(
                    request.argv, monitored.returncode,
                    stdout=monitored.stdout, stderr=monitored.stderr,
                )
            else:
                process = subprocess.run(
                    request.argv, cwd=cwd, env=env, capture_output=True, text=True,
                    encoding="utf-8", errors="replace", timeout=request.timeout_s,
                    shell=False,
                )
            elapsed = time.monotonic() - started
            status = RunStatus.SUCCEEDED if process.returncode == 0 else RunStatus.FAILED
            failure = FailureClass.NONE if process.returncode == 0 else request.nonzero_failure
            run = ToolRun(
                snapshot_id=request.snapshot_id, stage=request.stage, backend=self.name,
                request_hash=request_hash, toolchain_id=request.toolchain_id,
                status=status, command=list(request.argv),
                working_directory=request.working_directory,
                environment_hash=request.environment_hash,
                input_artifact_ids=list(request.input_artifact_ids),
                failure_class=failure, exit_code=process.returncode,
                started_at=started_at, finished_at=utc_now(), elapsed_s=elapsed,
                message=None if process.returncode == 0 else f"process exited with code {process.returncode}",
                metadata={**request.metadata, **_resource_guard_metadata(guard_result),
                          **_runtime_guard_metadata(runtime_result),
                          "runner_fingerprint": runner_fingerprint,
                          "inherited_environment_hash": inherited_hash,
                          "bootstrap_environment_hash": bootstrap_hash,
                          **_output_metadata(process.stdout, process.stderr),
                          "execution_enabled": True,
                          "fresh_execution": True, "fresh_tool_truth": True,
                          "authority": "tool_observation", "tool_truth": True,
                          "staging_isolated": True, "staged_output_manifest": []},
            )
            outputs: list[StagedOutput] = []
            if run.status == RunStatus.SUCCEEDED:
                try:
                    for declaration in request.declared_outputs:
                        candidate = staging.joinpath(*declaration.path.split("/"))
                        try:
                            candidate.lstat()
                        except FileNotFoundError:
                            continue
                        _data, size, digest, path = read_verified_file(
                            staging, declaration.path, max_bytes=declaration.max_bytes,
                        )
                        outputs.append(StagedOutput(
                            declaration.path, path, size, digest,
                        ))
                except (OSError, StagingError) as exc:
                    run.status = RunStatus.FAILED
                    run.failure_class = FailureClass.INPUT
                    run.message = f"declared output validation failed: {exc}"
                    run.metadata["fresh_tool_truth"] = False
                    run.metadata["tool_truth"] = False
                    outputs = []
            run.metadata["staged_output_manifest"] = [
                {"path": item.path, "size": item.size, "sha256": item.sha256}
                for item in outputs
            ]
            # ToolRun stable identity includes metadata, so measured output
            # metadata must be finalized before handing it to the orchestrator.
            run.id = ""
            run.__post_init__()
            return RunnerExecution(
                run, outputs, staging_directory=staging, staging_parent=staging_parent,
                resource_guard=guard_result,
                runtime_resource_monitor=runtime_result,
            )
        except subprocess.TimeoutExpired as exc:
            elapsed = time.monotonic() - started
            run = ToolRun(
                snapshot_id=request.snapshot_id, stage=request.stage, backend=self.name,
                request_hash=request_hash, toolchain_id=request.toolchain_id,
                status=RunStatus.FAILED, command=list(request.argv),
                working_directory=request.working_directory,
                environment_hash=request.environment_hash,
                input_artifact_ids=list(request.input_artifact_ids),
                failure_class=FailureClass.TIMEOUT, exit_code=None,
                started_at=started_at, finished_at=utc_now(), elapsed_s=elapsed,
                message=f"stage timed out after {request.timeout_s}s",
                metadata={**request.metadata, **_resource_guard_metadata(guard_result),
                          **_runtime_guard_metadata(runtime_result),
                          "runner_fingerprint": runner_fingerprint,
                          "inherited_environment_hash": inherited_hash,
                          "bootstrap_environment_hash": bootstrap_hash,
                          **_output_metadata(exc.stdout, exc.stderr),
                          "execution_enabled": True,
                          "fresh_execution": True, "fresh_tool_truth": False,
                          "authority": "tool_observation", "tool_truth": False,
                          "staging_isolated": True, "staged_output_manifest": []},
            )
            return RunnerExecution(
                run, staging_directory=staging, staging_parent=staging_parent,
                resource_guard=guard_result,
                runtime_resource_monitor=runtime_result,
            )
        except OSError as exc:
            run = ToolRun(
                snapshot_id=request.snapshot_id, stage=request.stage, backend=self.name,
                request_hash=request_hash, toolchain_id=request.toolchain_id,
                status=RunStatus.FAILED, command=list(request.argv),
                working_directory=request.working_directory,
                environment_hash=request.environment_hash,
                input_artifact_ids=list(request.input_artifact_ids),
                failure_class=FailureClass.INFRASTRUCTURE,
                started_at=started_at, finished_at=utc_now(),
                elapsed_s=time.monotonic() - started,
                message=str(exc), metadata={
                                             **request.metadata,
                                             **_resource_guard_metadata(guard_result),
                                             **_runtime_guard_metadata(runtime_result),
                                             "runner_fingerprint": runner_fingerprint,
                                             "inherited_environment_hash": inherited_hash,
                                             "bootstrap_environment_hash": bootstrap_hash,
                                              **_output_metadata(None, None),
                                              "execution_enabled": True,
                                              "fresh_execution": False,
                                              "fresh_tool_truth": False,
                                              "authority": "tool_observation",
                                              "tool_truth": False,
                                              "staging_isolated": True,
                                              "staged_output_manifest": []},
            )
            return RunnerExecution(
                run, staging_directory=staging, staging_parent=staging_parent,
                resource_guard=guard_result,
                runtime_resource_monitor=runtime_result,
            )


_REMOTE_EXECUTOR = r'''import base64,hashlib,json,os,pathlib,shutil,signal,stat,subprocess,sys,tempfile,time
PREFIX="HLSGRAPH_RUNNER_V2:"
PID_TOKEN="{process_group_pid}"
def emit(value):
 print(PREFIX+json.dumps(value,sort_keys=True,separators=(",",":")),flush=True)
def rel(value):
 value=value.replace("\\","/")
 parts=value.split("/")
 if not value or value.startswith("/") or any(x in ("",".","..") for x in parts):
  raise ValueError("unsafe relative path")
 return parts
def regular(root,value,limit):
 current=root
 parts=rel(value)
 for i,part in enumerate(parts):
  current=current/part
  info=current.lstat()
  if stat.S_ISLNK(info.st_mode): raise ValueError("link output")
  if i<len(parts)-1 and not stat.S_ISDIR(info.st_mode): raise ValueError("non-directory parent")
 if not stat.S_ISREG(current.lstat().st_mode): raise ValueError("non-regular output")
 flags=os.O_RDONLY|getattr(os,"O_NOFOLLOW",0)
 fd=os.open(current,flags); chunks=[]; digest=hashlib.sha256(); size=0
 try:
  if not stat.S_ISREG(os.fstat(fd).st_mode): raise ValueError("non-regular output")
  while True:
   chunk=os.read(fd,min(1048576,limit-size+1))
   if not chunk: break
   size+=len(chunk)
   if size>limit: raise ValueError("output too large")
   chunks.append(chunk); digest.update(chunk)
 finally: os.close(fd)
 return b"".join(chunks),size,digest.hexdigest()
def stop_group(tool):
 try: os.killpg(tool.pid,signal.SIGTERM)
 except OSError: pass
 try: tool.wait(timeout=.5)
 except subprocess.TimeoutExpired: pass
 try: os.killpg(tool.pid,signal.SIGKILL)
 except OSError: pass
 try: tool.wait(timeout=5)
 except (OSError,subprocess.TimeoutExpired):
  try: tool.kill()
  except OSError: pass
def encoded_streams(stdout_file,stderr_file):
 stdout_file.seek(0); stderr_file.seek(0)
 return {"stdout":base64.b64encode(stdout_file.read()).decode(),"stderr":base64.b64encode(stderr_file.read()).decode()}
root=None
try:
 payload=json.load(sys.stdin)
 if payload.get("protocol")!="hlsgraph.runner.v2": raise ValueError("protocol mismatch")
 base=pathlib.Path(sys.argv[1])
 if not base.is_absolute(): raise ValueError("remote staging root is not absolute")
 base.mkdir(parents=True,exist_ok=True)
 root=pathlib.Path(tempfile.mkdtemp(prefix="hlsgraph-run-",dir=base))
 manifest=[{k:item[k] for k in ("artifact_id","path","sha256","size")} for item in payload["inputs"]]
 encoded=json.dumps(manifest,sort_keys=True,separators=(",",":")).encode()
 if hashlib.sha256(encoded).hexdigest()!=payload["input_manifest_sha256"]: raise ValueError("input manifest mismatch")
 for item in payload["inputs"]:
  data=base64.b64decode(item["data"],validate=True)
  if len(data)!=item["size"] or hashlib.sha256(data).hexdigest()!=item["sha256"]: raise ValueError("input bytes mismatch")
  target=root.joinpath(*rel(item["path"])); target.parent.mkdir(parents=True,exist_ok=True)
  with target.open("xb") as handle: handle.write(data)
 monitor_env=dict(os.environ)
 env=dict(monitor_env); env.update(payload["environment"])
 att=subprocess.run(payload["attestation_argv"],cwd=root,env=env,capture_output=True,shell=False)
 if att.returncode!=0 or hashlib.sha256(att.stdout).hexdigest()!=payload["environment_hash"]:
  emit({"kind":"attestation","message":"remote environment attestation failed","exit_code":att.returncode}); sys.exit(0)
 cwd=root if payload["working_directory"]=="." else root.joinpath(*rel(payload["working_directory"]))
 cwd.mkdir(parents=True,exist_ok=True)
 guard=payload.get("resource_guard")
 if guard is not None:
  try:
   check=subprocess.run(guard["argv"],cwd=cwd,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,shell=False,timeout=guard["timeout_s"])
  except subprocess.TimeoutExpired:
   emit({"kind":"resource_guard","message":"resource guard timed out and rejected execution","exit_code":None}); sys.exit(0)
  if check.returncode!=0:
   emit({"kind":"resource_guard","message":"resource guard rejected execution","exit_code":check.returncode}); sys.exit(0)
 monitor=payload.get("runtime_resource_monitor")
 with tempfile.TemporaryFile() as stdout_file,tempfile.TemporaryFile() as stderr_file:
  tool=subprocess.Popen(payload["argv"],cwd=cwd,env=env,stdout=stdout_file,stderr=stderr_file,shell=False,start_new_session=True)
  deadline=time.monotonic()+float(payload["timeout_s"])
  next_check=time.monotonic(); runtime_checked=False
  while True:
   now=time.monotonic()
   if monitor is not None and now>=next_check:
    monitor_argv=monitor["argv"]
    if monitor_argv.count(PID_TOKEN)!=1: raise ValueError("invalid runtime monitor PID token")
    monitor_argv=[str(tool.pid) if item==PID_TOKEN else item for item in monitor_argv]
    try:
     check=subprocess.run(monitor_argv,cwd=cwd,env=monitor_env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,shell=False,timeout=monitor["timeout_s"])
    except subprocess.TimeoutExpired:
     runtime_checked=True; stop_group(tool)
     response={"kind":"runtime_guard","message":"runtime resource monitor timed out","exit_code":None,"runtime_guard_checked":True}
     response.update(encoded_streams(stdout_file,stderr_file)); emit(response); sys.exit(0)
    except OSError:
     runtime_checked=True; stop_group(tool)
     response={"kind":"runtime_guard_infrastructure","message":"runtime resource monitor could not execute","exit_code":None,"runtime_guard_checked":True}
     response.update(encoded_streams(stdout_file,stderr_file)); emit(response); sys.exit(0)
    runtime_checked=True
    if check.returncode!=0:
     stop_group(tool)
     response={"kind":"runtime_guard","message":"runtime resource monitor rejected execution","exit_code":check.returncode,"runtime_guard_checked":True}
     response.update(encoded_streams(stdout_file,stderr_file)); emit(response); sys.exit(0)
    next_check=time.monotonic()+float(monitor["interval_s"])
   if tool.poll() is not None: break
   now=time.monotonic()
   if now>=deadline:
    stop_group(tool)
    response={"kind":"timeout","message":"remote stage timed out","runtime_guard_checked":runtime_checked}
    response.update(encoded_streams(stdout_file,stderr_file)); emit(response); sys.exit(0)
   wait_until=deadline if monitor is None else min(deadline,next_check)
   time.sleep(min(0.05,max(0.001,wait_until-now)))
  streams=encoded_streams(stdout_file,stderr_file)
 outputs=[]
 if tool.returncode==0:
  for item in payload["declared_outputs"]:
   candidate=root.joinpath(*rel(item["path"]))
   try: candidate.lstat()
   except FileNotFoundError: continue
   data,size,digest=regular(root,item["path"],item["max_bytes"])
   outputs.append({"path":item["path"],"size":size,"sha256":digest,"data":base64.b64encode(data).decode()})
 response={"kind":"tool","exit_code":tool.returncode,"outputs":outputs,"runtime_guard_checked":runtime_checked}
 response.update(streams); emit(response)
except ValueError as exc:
 emit({"kind":"input","message":str(exc)[:1024]})
except OSError as exc:
 emit({"kind":"infrastructure","message":str(exc)[:1024]})
except Exception as exc:
 emit({"kind":"infrastructure","message":str(exc)[:1024]})
finally:
 if root is not None: shutil.rmtree(root,ignore_errors=True)
'''


class SSHRunner(Runner):
    """SSH backend with explicit manifests and byte transfer in both directions."""

    name = "runner.ssh"
    provides_local_output_bytes = True
    run_scoped_staging = True
    can_produce_tool_truth = True

    def __init__(
        self, host: str, remote_project_root: str, *, project_root: str | Path | None = None,
        allow_execution: bool = False,
        ssh_options: Sequence[str] = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=10"),
        resource_guard: ResourceGuard | Mapping[str, Any] | None = None,
        runtime_resource_monitor: (
            RuntimeResourceMonitor | Mapping[str, Any] | None
        ) = None,
    ):
        if not host or host.startswith("-") or any(char.isspace() for char in host):
            raise ValueError("host must be a non-empty SSH destination without whitespace")
        if (not remote_project_root.startswith("/") or "\x00" in remote_project_root
                or (remote_project_root != "/" and any(
                    part in {"", ".", ".."}
                    for part in remote_project_root.rstrip("/").split("/")[1:]))):
            raise ValueError("remote_project_root must be an absolute normalized POSIX path")
        self.host = host
        self.remote_project_root = remote_project_root.rstrip("/") or "/"
        self.project_root = Path(project_root).resolve() if project_root is not None else None
        self.allow_execution = bool(allow_execution)
        self.ssh_options = tuple(ssh_options)
        self.resource_guard = _coerce_resource_guard(resource_guard)
        self.runtime_resource_monitor = _coerce_runtime_resource_monitor(
            runtime_resource_monitor
        )

    @property
    def can_report_resource_guard(self) -> bool:  # type: ignore[override]
        return self.resource_guard is not None

    @property
    def can_report_runtime_resource_guard(self) -> bool:  # type: ignore[override]
        return self.runtime_resource_monitor is not None

    def bind_project_root(self, project_root: str | Path) -> None:
        selected = Path(project_root).resolve()
        if self.project_root is not None and self.project_root != selected:
            raise RunnerProtocolError("SSH runner is already bound to a different project root")
        self.project_root = selected

    @property
    def fingerprint(self) -> str:
        return stable_hash({
            "backend": self.name, "protocol": PROTOCOL_VERSION,
            "host": self.host, "remote_staging_root": self.remote_project_root,
            "ssh_options": self.ssh_options,
            "resource_guard": (
                stable_hash(self.resource_guard.identity_payload())
                if self.resource_guard is not None else None
            ),
            "runtime_resource_monitor": (
                stable_hash(self.runtime_resource_monitor.identity_payload())
                if self.runtime_resource_monitor is not None else None
            ),
        })

    @staticmethod
    def _remote_attestation(request: ToolRunRequest) -> tuple[list[str], str]:
        expected = str(request.environment_hash or "").casefold()
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise RunnerProtocolError(
                "SSH execution requires environment_hash as the expected SHA-256 "
                "of remote attestation stdout"
            )
        argv = request.metadata.get("remote_attestation_argv")
        if (not isinstance(argv, list) or not argv
                or not all(isinstance(item, str) and item and "\x00" not in item for item in argv)):
            raise RunnerProtocolError(
                "SSH execution requires metadata.remote_attestation_argv as a non-empty argv list"
            )
        return list(argv), expected

    def execute(self, request: ToolRunRequest) -> RunnerExecution:
        if not self.allow_execution:
            guard = (ResourceGuardResult(False, False)
                     if self.resource_guard is not None else None)
            runtime_monitor = (
                RuntimeResourceMonitorResult(False, False, False)
                if self.runtime_resource_monitor is not None else None
            )
            return self._disabled(
                request, self.name, self.fingerprint, resource_guard=guard,
                runtime_resource_monitor=runtime_monitor,
            )
        if self.project_root is None:
            raise RunnerProtocolError("SSH runner must be bound to a local project root")
        attestation_argv, expected_environment_hash = self._remote_attestation(request)
        staging, staging_parent = create_run_directory(self.project_root)
        started_at = utc_now()
        started = time.monotonic()
        guard_result = (ResourceGuardResult(False, False)
                        if self.resource_guard is not None else None)
        runtime_result = (
            RuntimeResourceMonitorResult(False, False, False)
            if self.runtime_resource_monitor is not None else None
        )
        try:
            input_payload: list[dict[str, Any]] = []
            transferred = 0
            for item in sorted(request.inputs, key=lambda value: value.staged_path):
                data, _size, _digest, _path = read_verified_file(
                    self.project_root, item.source_path, expected_size=item.size,
                    expected_sha256=item.sha256, max_bytes=item.size,
                )
                transferred += len(data)
                if transferred > MAX_TOTAL_TRANSFER_BYTES:
                    raise StagingError("explicit SSH input transfer exceeds total byte limit")
                input_payload.append({
                    "artifact_id": item.artifact_id, "path": item.staged_path,
                    "sha256": item.sha256, "size": item.size,
                    "data": base64.b64encode(data).decode("ascii"),
                })
            input_manifest = [
                {key: item[key] for key in ("artifact_id", "path", "sha256", "size")}
                for item in input_payload
            ]
            manifest_bytes = json.dumps(
                input_manifest, sort_keys=True, separators=(",", ":"),
            ).encode("utf-8")
            payload = {
                "protocol": PROTOCOL_VERSION, "inputs": input_payload,
                "input_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                "declared_outputs": [item.identity_payload()
                                     for item in request.declared_outputs],
                "argv": list(request.argv), "working_directory": request.working_directory,
                "environment": dict(request.environment),
                "attestation_argv": attestation_argv,
                "environment_hash": expected_environment_hash,
                "timeout_s": request.timeout_s,
                "resource_guard": (
                    self.resource_guard.identity_payload()
                    if self.resource_guard is not None else None
                ),
                "runtime_resource_monitor": (
                    self.runtime_resource_monitor.identity_payload()
                    if self.runtime_resource_monitor is not None else None
                ),
            }
            remote_command = "python3 -c " + shlex.quote(_REMOTE_EXECUTOR) + " " + shlex.quote(
                self.remote_project_root
            )
            ssh_argv = ["ssh", *self.ssh_options, self.host, remote_command]
            transport = subprocess.run(
                ssh_argv, input=json.dumps(payload, sort_keys=True, separators=(",", ":")),
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=(request.timeout_s
                         + (float(self.resource_guard.timeout_s)
                            if self.resource_guard is not None else 0.0)
                         + (float(self.runtime_resource_monitor.timeout_s)
                            if self.runtime_resource_monitor is not None else 0.0)
                         + 30.0), shell=False,
            )
            response_line = next(
                (line[len("HLSGRAPH_RUNNER_V2:"):] for line in reversed(
                    transport.stdout.splitlines()) if line.startswith("HLSGRAPH_RUNNER_V2:")),
                None,
            )
            if transport.returncode != 0 or response_line is None:
                failure = FailureClass.SSH if transport.returncode in {255, None} else FailureClass.INFRASTRUCTURE
                response: dict[str, Any] = {
                    "kind": "transport", "exit_code": transport.returncode,
                    "message": "SSH transport failed or returned no runner-v2 response",
                    "stdout": "", "stderr": base64.b64encode(
                        transport.stderr.encode("utf-8", errors="replace")
                    ).decode("ascii"), "outputs": [],
                }
            else:
                try:
                    response = json.loads(response_line)
                except json.JSONDecodeError as exc:
                    raise StagingError("remote runner returned invalid JSON") from exc
                kind = response.get("kind")
                failure = (
                    FailureClass.NONE if kind == "tool" and response.get("exit_code") == 0 else
                    request.nonzero_failure if kind == "tool" else
                    FailureClass.TIMEOUT if kind == "timeout" else
                    self.runtime_resource_monitor.failure_for_exit(
                        response.get("exit_code")
                        if isinstance(response.get("exit_code"), int) else None
                    ) if kind == "runtime_guard" and
                    self.runtime_resource_monitor is not None else
                    FailureClass.INFRA_RESOURCE_GUARD
                    if kind == "resource_guard" and self.resource_guard is not None else
                    FailureClass.INPUT if kind == "input" else
                    FailureClass.INFRASTRUCTURE
                )
            kind = str(response.get("kind") or "transport")
            exit_code = response.get("exit_code")
            if not isinstance(exit_code, int):
                exit_code = None
            stdout = base64.b64decode(response.get("stdout") or "", validate=True)
            stderr = base64.b64decode(response.get("stderr") or "", validate=True)
            outputs: list[StagedOutput] = []
            total_outputs = 0
            if kind == "tool" and exit_code == 0:
                declarations = {item.path: item for item in request.declared_outputs}
                for item in response.get("outputs", []):
                    if not isinstance(item, Mapping) or item.get("path") not in declarations:
                        raise StagingError("remote runner returned an undeclared output")
                    declaration = declarations[str(item["path"])]
                    data = base64.b64decode(str(item.get("data") or ""), validate=True)
                    size = item.get("size")
                    digest = str(item.get("sha256") or "").casefold()
                    if (not isinstance(size, int) or isinstance(size, bool)
                            or size != len(data) or size > declaration.max_bytes
                            or hashlib.sha256(data).hexdigest() != digest):
                        raise StagingError("remote output manifest does not match returned bytes")
                    total_outputs += size
                    if total_outputs > MAX_TOTAL_TRANSFER_BYTES:
                        raise StagingError("explicit SSH output transfer exceeds total byte limit")
                    path = write_new_file(staging, declaration.path, data)
                    _data, checked_size, checked_digest, checked_path = read_verified_file(
                        staging, declaration.path, expected_size=size,
                        expected_sha256=digest, max_bytes=declaration.max_bytes,
                    )
                    outputs.append(StagedOutput(
                        declaration.path, checked_path, checked_size, checked_digest,
                    ))
            status = RunStatus.SUCCEEDED if failure == FailureClass.NONE else RunStatus.FAILED
            if self.resource_guard is not None:
                if kind in {
                    "tool", "timeout", "runtime_guard",
                    "runtime_guard_infrastructure",
                }:
                    guard_result = ResourceGuardResult(True, True, 0)
                elif kind == "resource_guard":
                    guard_result = ResourceGuardResult(True, False, exit_code)
            if self.runtime_resource_monitor is not None:
                if kind in {"tool", "timeout"}:
                    checked = response.get("runtime_guard_checked") is True
                    runtime_result = RuntimeResourceMonitorResult(
                        checked, checked, False, 0 if checked else None,
                    )
                elif kind == "runtime_guard":
                    runtime_result = RuntimeResourceMonitorResult(
                        True, False, True, exit_code,
                        self.runtime_resource_monitor.failure_for_exit(exit_code),
                    )
                elif kind == "runtime_guard_infrastructure":
                    runtime_result = RuntimeResourceMonitorResult(
                        True, False, True, None, FailureClass.INFRASTRUCTURE,
                    )
            remote_inputs_verified = kind in {
                "attestation", "resource_guard", "runtime_guard",
                "runtime_guard_infrastructure", "tool", "timeout",
            }
            remote_environment_verified = kind in {
                "resource_guard", "runtime_guard",
                "runtime_guard_infrastructure", "tool", "timeout",
            }
            fresh_execution = kind in {
                "runtime_guard", "runtime_guard_infrastructure", "tool", "timeout",
            }
            metadata = {
                **request.metadata, **_resource_guard_metadata(guard_result),
                **_runtime_guard_metadata(runtime_result),
                "runner_fingerprint": self.fingerprint,
                "remote_inputs_verified": remote_inputs_verified,
                "remote_environment_verified": remote_environment_verified,
                "expected_remote_environment_hash": expected_environment_hash,
                **_output_metadata(stdout, stderr), "execution_enabled": True,
                "fresh_execution": fresh_execution,
                "fresh_tool_truth": (
                    kind == "tool" and remote_environment_verified
                ),
                "authority": ("infrastructure" if kind in {
                    "resource_guard", "runtime_guard",
                    "runtime_guard_infrastructure",
                }
                              else "tool_observation"),
                "tool_truth": (kind == "tool" and remote_inputs_verified
                               and remote_environment_verified),
                "staging_isolated": True,
                "staged_output_manifest": [
                    {"path": item.path, "size": item.size, "sha256": item.sha256}
                    for item in outputs
                ],
            }
            run = ToolRun(
                snapshot_id=request.snapshot_id, stage=request.stage, backend=self.name,
                request_hash=request.cache_key(self.fingerprint),
                toolchain_id=request.toolchain_id, status=status,
                command=list(request.argv), working_directory=request.working_directory,
                environment_hash=request.environment_hash,
                input_artifact_ids=list(request.input_artifact_ids),
                failure_class=failure, exit_code=exit_code,
                started_at=started_at, finished_at=utc_now(),
                elapsed_s=time.monotonic() - started,
                message=None if failure == FailureClass.NONE else str(
                    response.get("message") or (
                        f"remote process exited with code {exit_code}" if kind == "tool"
                        else "remote execution failed"
                    )
                ), metadata=metadata,
            )
            return RunnerExecution(
                run, outputs, staging_directory=staging, staging_parent=staging_parent,
                resource_guard=guard_result,
                runtime_resource_monitor=runtime_result,
            )
        except subprocess.TimeoutExpired as exc:
            run = ToolRun(
                snapshot_id=request.snapshot_id, stage=request.stage, backend=self.name,
                request_hash=request.cache_key(self.fingerprint),
                toolchain_id=request.toolchain_id, status=RunStatus.FAILED,
                command=list(request.argv), working_directory=request.working_directory,
                environment_hash=request.environment_hash,
                input_artifact_ids=list(request.input_artifact_ids),
                failure_class=FailureClass.SSH, started_at=started_at,
                finished_at=utc_now(), elapsed_s=time.monotonic() - started,
                message="SSH transport timed out",
                metadata={**request.metadata, **_resource_guard_metadata(guard_result),
                          **_runtime_guard_metadata(runtime_result),
                          "runner_fingerprint": self.fingerprint,
                          **_output_metadata(exc.stdout, exc.stderr),
                          "execution_enabled": True, "fresh_execution": False,
                          "fresh_tool_truth": False, "authority": "tool_observation",
                          "tool_truth": False, "remote_inputs_verified": False,
                          "remote_environment_verified": False,
                          "expected_remote_environment_hash": expected_environment_hash,
                          "staging_isolated": True, "staged_output_manifest": []},
            )
            return RunnerExecution(
                run, staging_directory=staging, staging_parent=staging_parent,
                resource_guard=guard_result,
                runtime_resource_monitor=runtime_result,
            )
        except (OSError, StagingError, ValueError) as exc:
            run = ToolRun(
                snapshot_id=request.snapshot_id, stage=request.stage, backend=self.name,
                request_hash=request.cache_key(self.fingerprint),
                toolchain_id=request.toolchain_id, status=RunStatus.FAILED,
                command=list(request.argv), working_directory=request.working_directory,
                environment_hash=request.environment_hash,
                input_artifact_ids=list(request.input_artifact_ids),
                failure_class=FailureClass.INPUT if isinstance(exc, StagingError)
                else FailureClass.INFRASTRUCTURE,
                started_at=started_at, finished_at=utc_now(),
                elapsed_s=time.monotonic() - started, message=str(exc),
                metadata={**request.metadata, **_resource_guard_metadata(guard_result),
                          **_runtime_guard_metadata(runtime_result),
                          "runner_fingerprint": self.fingerprint,
                          **_output_metadata(None, None), "execution_enabled": True,
                          "fresh_execution": False, "fresh_tool_truth": False,
                          "authority": "tool_observation", "tool_truth": False,
                          "remote_inputs_verified": False,
                          "remote_environment_verified": False,
                          "expected_remote_environment_hash": expected_environment_hash,
                          "staging_isolated": True, "staged_output_manifest": []},
            )
            return RunnerExecution(
                run, staging_directory=staging, staging_parent=staging_parent,
                resource_guard=guard_result,
                runtime_resource_monitor=runtime_result,
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

    def execute(self, request: ToolRunRequest) -> RunnerExecution:
        self.calls.append(request)
        value = self.script[request.stage].popleft() if self.script[request.stage] else FakeOutcome()
        outcome = value if isinstance(value, FakeOutcome) else FakeOutcome(**dict(value))
        outcome_metadata = {
            key: value for key, value in outcome.metadata.items()
            if key not in RUNNER_MEASURED_METADATA_KEYS
        }
        failure_class = (
            FailureClass.INFRASTRUCTURE
            if outcome.failure_class in {
                FailureClass.INFRA_RESOURCE_GUARD, FailureClass.RESOURCE,
            }
            else outcome.failure_class
        )
        run = ToolRun(
            snapshot_id=request.snapshot_id, stage=request.stage, backend=self.name,
            request_hash=request.cache_key(self.fingerprint), toolchain_id=request.toolchain_id,
            status=outcome.status, command=list(request.argv),
            working_directory=request.working_directory,
            environment_hash=request.environment_hash,
            input_artifact_ids=list(request.input_artifact_ids),
            output_artifact_ids=list(outcome.output_artifact_ids),
            gates=list(outcome.gates), failure_class=failure_class,
            exit_code=outcome.exit_code, started_at=utc_now(), finished_at=utc_now(),
            elapsed_s=0.0, message=outcome.message,
            metadata={"runner_fingerprint": self.fingerprint,
                      **request.metadata, **outcome_metadata,
                      **_resource_guard_metadata(None),
                      **_runtime_guard_metadata(None),
                      "fresh_execution": False, "fresh_tool_truth": False,
                      "authority": "synthetic", "tool_truth": False,
                      "staging_isolated": False, "staged_output_manifest": []},
        )
        return RunnerExecution(run)


class ReplayRunner(Runner):
    name = "runner.replay"

    def __init__(self, runs: Mapping[str, ToolRun | RunnerExecution], *,
                 source_runner_fingerprint: str):
        self.runs = {
            key: value.run if isinstance(value, RunnerExecution) else value
            for key, value in runs.items()
        }
        self.source_runner_fingerprint = source_runner_fingerprint

    @property
    def fingerprint(self) -> str:
        return stable_hash({"backend": self.name, "protocol": RUNNER_PROTOCOL_VERSION,
                            "source_runner": self.source_runner_fingerprint})

    def execute(self, request: ToolRunRequest) -> RunnerExecution:
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
        run = ToolRun(
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
                      **_resource_guard_metadata(None),
                      **_runtime_guard_metadata(None),
                      "fresh_execution": False, "fresh_tool_truth": False,
                      "runner_fingerprint": self.fingerprint,
                      "authority": "replay", "tool_truth": False,
                      "staging_isolated": False, "staged_output_manifest": []},
        )
        return RunnerExecution(run)


@dataclass(slots=True)
class StageResult:
    runs: list[ToolRun]
    gates: dict[GateKind, GateStatus]
    correctness_checks: dict[str, GateStatus]
    gates_complete: bool
    tool_truth: bool
    verified: bool
    stopped_after_stage: str | None = None
    executions: list[RunnerExecution] = field(default_factory=list, repr=False)

    def cleanup(self) -> None:
        for execution in self.executions:
            execution.cleanup()


class StageOrchestrator:
    def __init__(self, runner: Runner, *,
                 evidence_validator: Callable[[ToolRun, str], bool] | None = None):
        self.runner = runner
        # Gate IDs alone are references, not proof. Callers with a ledger or
        # an atomic run-result bundle can provide recursive evidence validation;
        # absence of such a validator is fail-closed for ``tool_truth``.
        self.evidence_validator = evidence_validator

    def _validate_response(self, request: ToolRunRequest, execution: RunnerExecution,
                           runner_fingerprint: str, runner_name: str,
                           runner_capabilities: Mapping[str, Any]) -> None:
        if not isinstance(execution, RunnerExecution):
            raise RunnerProtocolError("runner response must be a RunnerExecution")
        run = execution.run
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
        guard = execution.resource_guard
        runtime_monitor = execution.runtime_resource_monitor
        guard_keys = (
            "resource_guard_configured", "resource_guard_checked",
            "resource_guard_passed",
        )
        if guard is None:
            if (run.failure_class == FailureClass.INFRA_RESOURCE_GUARD
                    and not (runtime_monitor and runtime_monitor.triggered)):
                raise RunnerProtocolError(
                    "resource-guard failure requires a structured runner result"
                )
            if any(run.metadata.get(key) is True for key in guard_keys):
                raise RunnerProtocolError(
                    "resource-guard metadata requires a structured runner result"
                )
        else:
            if runner_capabilities.get("can_report_resource_guard") is not True:
                raise RunnerProtocolError(
                    "runner is not trusted to report resource-guard results"
                )
            if str(run.backend).casefold() in {"runner.fake", "runner.replay"}:
                raise RunnerProtocolError(
                    "fake/replay runners cannot report resource-guard results"
                )
            expected_guard_metadata = _resource_guard_metadata(guard)
            if any(run.metadata.get(key) != value
                   for key, value in expected_guard_metadata.items()):
                raise RunnerProtocolError(
                    "runner resource-guard metadata disagrees with its structured result"
                )
            if not guard.checked:
                if (run.failure_class == FailureClass.INFRA_RESOURCE_GUARD
                        or run.metadata.get("fresh_execution") is True):
                    raise RunnerProtocolError(
                        "unchecked resource guard cannot classify or precede tool execution"
                    )
            elif guard.passed:
                if (run.failure_class == FailureClass.INFRA_RESOURCE_GUARD
                        and not (runtime_monitor and runtime_monitor.triggered)):
                    raise RunnerProtocolError(
                        "passing resource guard cannot classify a stage failure"
                    )
            else:
                if (run.status != RunStatus.FAILED
                        or run.failure_class != FailureClass.INFRA_RESOURCE_GUARD
                        or run.exit_code != guard.exit_code
                        or run.metadata.get("fresh_execution") is not False
                        or run.metadata.get("fresh_tool_truth") is not False
                        or run.metadata.get("tool_truth") is not False
                        or run.metadata.get("authority") != "infrastructure"
                        or execution.staged_outputs
                        or run.output_artifact_ids):
                    raise RunnerProtocolError(
                        "resource-guard rejection has inconsistent provenance"
                    )
        runtime_keys = (
            "runtime_guard_configured", "runtime_guard_checked",
            "runtime_guard_passed", "runtime_guard_triggered",
        )
        if runtime_monitor is None:
            if (run.failure_class == FailureClass.INFRA_RESOURCE_GUARD
                    and not (guard and guard.checked and not guard.passed)):
                raise RunnerProtocolError(
                    "runtime resource-guard failure requires a structured runner result"
                )
            if any(run.metadata.get(key) is True for key in runtime_keys):
                raise RunnerProtocolError(
                    "runtime-guard metadata requires a structured runner result"
                )
            if run.failure_class == FailureClass.RESOURCE:
                raise RunnerProtocolError(
                    "resource failure requires a trusted runtime monitor mapping"
                )
        else:
            if runner_capabilities.get("can_report_runtime_resource_guard") is not True:
                raise RunnerProtocolError(
                    "runner is not trusted to report runtime resource-guard results"
                )
            if str(run.backend).casefold() in {"runner.fake", "runner.replay"}:
                raise RunnerProtocolError(
                    "fake/replay runners cannot report runtime resource-guard results"
                )
            expected_runtime_metadata = _runtime_guard_metadata(runtime_monitor)
            if any(run.metadata.get(key) != value
                   for key, value in expected_runtime_metadata.items()):
                raise RunnerProtocolError(
                    "runner runtime-guard metadata disagrees with its structured result"
                )
            if not runtime_monitor.checked:
                if (runtime_monitor.passed or runtime_monitor.triggered
                        or run.metadata.get("fresh_execution") is True
                        or run.failure_class == FailureClass.RESOURCE):
                    raise RunnerProtocolError(
                        "unchecked runtime resource monitor cannot precede tool execution"
                    )
            elif runtime_monitor.passed:
                if (runtime_monitor.triggered
                        or run.failure_class in {
                            FailureClass.INFRA_RESOURCE_GUARD,
                            FailureClass.RESOURCE,
                        }):
                    raise RunnerProtocolError(
                        "passing runtime resource monitor cannot classify a guard failure"
                    )
            else:
                if (not runtime_monitor.triggered
                        or run.status != RunStatus.FAILED
                        or run.failure_class != runtime_monitor.failure_class
                        or run.exit_code != runtime_monitor.exit_code
                        or run.metadata.get("fresh_execution") is not True
                        or run.metadata.get("fresh_tool_truth") is not False
                        or run.metadata.get("tool_truth") is not False
                        or run.metadata.get("authority") != "infrastructure"
                        or execution.staged_outputs
                        or run.output_artifact_ids
                        or (guard is not None and not guard.passed)):
                    raise RunnerProtocolError(
                        "runtime resource-guard trigger has inconsistent provenance"
                    )
        declarations = {item.path: item for item in request.declared_outputs}
        paths = [item.path for item in execution.staged_outputs]
        if len(paths) != len(set(paths)):
            raise RunnerProtocolError("runner response contains duplicate staged outputs")
        if any(path not in declarations for path in paths):
            raise RunnerProtocolError("runner response contains an undeclared staged output")
        if str(run.backend).casefold() in {"runner.fake", "runner.replay"} and paths:
            raise RunnerProtocolError("fake/replay runners cannot return staged tool outputs")
        for item in execution.staged_outputs:
            declaration = declarations[item.path]
            if item.size > declaration.max_bytes:
                raise RunnerProtocolError("runner response exceeds declared output byte limit")
            if execution.staging_directory is None:
                raise RunnerProtocolError("staged outputs require a staging directory")
            try:
                _data, size, digest, path = read_verified_file(
                    execution.staging_directory, item.path,
                    expected_size=item.size, expected_sha256=item.sha256,
                    max_bytes=declaration.max_bytes,
                )
            except StagingError as exc:
                raise RunnerProtocolError(str(exc)) from exc
            if path != item.local_path or size != item.size or digest != item.sha256:
                raise RunnerProtocolError("runner staged output identity mismatch")

    def execute(self, requests: Sequence[ToolRunRequest]) -> StageResult:
        runs: list[ToolRun] = []
        executions: list[RunnerExecution] = []
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
            runner_capabilities = self.runner.capabilities()
            if (not isinstance(runner_capabilities, Mapping)
                    or runner_capabilities.get("protocol_version") != PROTOCOL_VERSION
                    or runner_capabilities.get("name") != runner_name
                    or runner_capabilities.get("fingerprint") != runner_fingerprint
                    or not isinstance(
                        runner_capabilities.get("can_report_resource_guard"), bool,
                    )
                    or not isinstance(
                        runner_capabilities.get(
                            "can_report_runtime_resource_guard"
                        ), bool,
                    )):
                raise RunnerProtocolError("runner capabilities do not match runner-v2 identity")
            # Runner is an untrusted SPI boundary.  Give it an isolated request
            # so in-place mutation cannot rewrite the expectation used below or
            # the caller's stage-selection/control-flow state.
            execution = self.runner.execute(copy.deepcopy(request))
            try:
                self._validate_response(
                    request, execution, runner_fingerprint, runner_name,
                    runner_capabilities,
                )
            except Exception:
                if isinstance(execution, RunnerExecution):
                    execution.cleanup()
                raise
            run = execution.run
            executions.append(execution)
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
                           stopped_after_stage=stopped, executions=executions)
