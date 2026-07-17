"""Project manifest, compilation database, and immutable snapshot construction."""
from __future__ import annotations

import json
import mimetypes
import os
import re
import shlex
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 compatibility
    import tomli as tomllib
from pathlib import Path
from typing import Any, Iterable

from .model import (
    AccessPolicy,
    ArtifactRef,
    BuildContext,
    ClockConstraint,
    ConstraintSet,
    DesignSnapshot,
    ProjectManifest,
    RetentionPolicy,
    TargetProfile,
    TranslationUnit,
    artifact_hash_map,
    hash_artifact_bytes,
    json_ready,
    safe_relative_path,
    stable_hash,
)


class ManifestError(ValueError):
    pass


SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx"}


def _inside(root: Path, value: Path) -> Path:
    root = root.resolve()
    value = value.resolve()
    try:
        return value.relative_to(root)
    except ValueError as exc:
        raise ManifestError(f"path is outside project root: {value}") from exc


def project_path(root: str | Path, relative: str) -> Path:
    relative = safe_relative_path(relative)
    resolved_root = Path(root).resolve()
    candidate = (resolved_root / Path(relative)).resolve()
    _inside(resolved_root, candidate)
    return candidate


def load_manifest(path: str | Path) -> ProjectManifest:
    path = Path(path)
    if not path.is_file():
        raise ManifestError(f"manifest does not exist: {path}")
    if path.suffix.lower() == ".toml":
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    elif path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        raise ManifestError("manifest must be .toml or .json")
    try:
        return ProjectManifest.from_dict(data)
    except (KeyError, TypeError, ValueError) as exc:
        raise ManifestError(f"invalid manifest {path}: {exc}") from exc


def write_internal_manifest(path: str | Path, manifest: ProjectManifest) -> None:
    Path(path).write_text(json.dumps(json_ready(manifest), ensure_ascii=False, indent=2,
                                    sort_keys=True) + "\n", encoding="utf-8")


def manifest_template(project_id: str, name: str, top: str, source: str) -> str:
    source = safe_relative_path(source, "source")
    return f'''schema_version = "0.1.0"
project_id = "{project_id}"
name = "{name}"
stage_toolchains = {{}}

[build]
top = "{top}"
language = "c++"
flow_target = "vitis"
compile_commands = "compile_commands.json"
include_dirs = []
config_files = []
tcl_files = []
testbench_files = []
golden_files = []

[[build.translation_units]]
file = "{source}"
directory = "."
arguments = ["-std=c++17"]

[target]
vendor = "amd"
part = ""

[[target.clocks]]
name = "default"
period_ns = 5.0
uncertainty_ns = 0.0

[constraints]
xdc_files = []

[[artifact_paths]]
path = "{source}"
kind = "source.cpp"
role = "design_source"
access = "private"
'''


def load_compile_commands(path: str | Path, project_root: str | Path) -> list[TranslationUnit]:
    root = Path(project_root).resolve()
    path = Path(path).resolve()
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ManifestError("compile_commands.json must contain an array")
    result: list[TranslationUnit] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or "file" not in row or "directory" not in row:
            raise ManifestError(f"invalid compile command at index {index}")
        directory = Path(str(row["directory"]))
        if not directory.is_absolute():
            directory = path.parent / directory
        source = Path(str(row["file"]))
        if not source.is_absolute():
            source = directory / source
        source_rel = _inside(root, source).as_posix()
        directory_rel = _inside(root, directory).as_posix() or "."
        if "arguments" in row:
            arguments = [str(item) for item in row["arguments"]]
        elif "command" in row:
            arguments = shlex.split(str(row["command"]), posix=os.name != "nt")
        else:
            raise ManifestError(f"compile command {index} has neither arguments nor command")
        root_text = str(root)
        root_posix = root.as_posix()
        arguments = [arg.replace(root_text, "${PROJECT_ROOT}").replace(root_posix, "${PROJECT_ROOT}")
                     for arg in arguments]
        output = row.get("output")
        if output:
            output_path = Path(str(output))
            if not output_path.is_absolute():
                output_path = directory / output_path
            try:
                output = _inside(root, output_path).as_posix()
            except ManifestError:
                output = None
        result.append(TranslationUnit(file=source_rel, directory=directory_rel,
                                      arguments=arguments, output=output))
    return sorted(result, key=lambda item: (item.file, item.directory, item.arguments))


def hydrate_compilation_database(manifest: ProjectManifest, project_root: str | Path) -> ProjectManifest:
    compile_commands = manifest.build.compile_commands
    if not compile_commands:
        return manifest
    path = project_path(project_root, compile_commands)
    if not path.exists():
        if manifest.build.translation_units:
            return manifest
        raise ManifestError(f"compilation database not found: {compile_commands}")
    units = load_compile_commands(path, project_root)
    build_data = json_ready(manifest.build)
    build_data["translation_units"] = [json_ready(item) for item in units]
    manifest_data = json_ready(manifest)
    manifest_data["build"] = build_data
    return ProjectManifest.from_dict(manifest_data)


def _local_includes(path: Path) -> list[tuple[str, bool]]:
    """Return textual includes as ``(name, is_quoted)`` without preprocessing.

    This deliberately over-approximates conditional includes. Hashing an
    inactive project header is harmless; omitting a header that clang may read
    would make a snapshot unsound.
    """
    result: list[tuple[str, bool]] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return result
    for match in re.finditer(
        r'^\s*#\s*include\s*([<"])([^">]+)[>"]', text, re.MULTILINE
    ):
        result.append((match.group(2).strip(), match.group(1) == '"'))
    return result


def _argument_path(root: Path, base: Path, value: str) -> Path | None:
    value = value.strip().strip('"').replace("${PROJECT_ROOT}", str(root))
    if not value:
        return None
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = base / candidate
    candidate = candidate.resolve()
    try:
        _inside(root, candidate)
    except ManifestError:
        # System include paths are compilation context, but their contents are
        # outside the project snapshot and must not be copied into it.
        return None
    return candidate


def _compiler_path_flags(arguments: Iterable[str]) -> tuple[list[str], list[str]]:
    """Extract include directories and forced includes from clang-style argv."""
    args = list(arguments)
    include_dirs: list[str] = []
    forced: list[str] = []
    index = 0
    while index < len(args):
        arg = args[index]
        next_value: str | None = None
        destination: list[str] | None = None
        if arg in {"-I", "/I", "-isystem", "-iquote", "-idirafter",
                   "--include-directory"}:
            destination = include_dirs
            if index + 1 < len(args):
                next_value = args[index + 1]
                index += 1
        elif arg in {"-include", "-imacros", "/FI"}:
            destination = forced
            if index + 1 < len(args):
                next_value = args[index + 1]
                index += 1
        else:
            joined_prefixes = (
                ("--include-directory=", include_dirs),
                ("-isystem=", include_dirs),
                ("-iquote=", include_dirs),
                ("-idirafter=", include_dirs),
                ("-include=", forced),
                ("-imacros=", forced),
            )
            for prefix, target in joined_prefixes:
                if arg.startswith(prefix) and len(arg) > len(prefix):
                    destination, next_value = target, arg[len(prefix):]
                    break
            if destination is None:
                if arg.startswith("-I") and len(arg) > 2:
                    destination, next_value = include_dirs, arg[2:]
                elif arg.startswith("/I") and len(arg) > 2:
                    destination, next_value = include_dirs, arg[2:]
                elif arg.startswith("/FI") and len(arg) > 3:
                    destination, next_value = forced, arg[3:]
        if destination is not None and next_value:
            destination.append(next_value)
        index += 1
    return include_dirs, forced


def _expand_response_arguments(root: Path, base: Path, arguments: Iterable[str], *,
                               seen: set[Path] | None = None) -> tuple[list[str], set[str]]:
    expanded: list[str] = []
    response_files: set[str] = set()
    seen = set() if seen is None else seen
    for argument in arguments:
        if not argument.startswith("@") or len(argument) == 1:
            expanded.append(argument)
            continue
        response = _argument_path(root, base, argument[1:])
        if response is None or not response.is_file():
            expanded.append(argument)
            continue
        if response in seen:
            raise ManifestError(f"recursive compiler response file: {_inside(root, response).as_posix()}")
        if response.stat().st_size > 4 * 1024 * 1024:
            raise ManifestError(f"compiler response file is too large: {_inside(root, response).as_posix()}")
        seen.add(response)
        relative = _inside(root, response).as_posix()
        response_files.add(relative)
        nested_args = shlex.split(response.read_text(encoding="utf-8", errors="replace"),
                                  posix=os.name != "nt")
        nested, nested_files = _expand_response_arguments(
            # Compiler response-file arguments are textual argv substitution;
            # relative paths retain the compilation working directory.
            root, base, nested_args, seen=seen,
        )
        expanded.extend(nested)
        response_files.update(nested_files)
        seen.remove(response)
    return expanded, response_files


def _compilation_paths(manifest: ProjectManifest, root: Path) -> tuple[list[Path], set[str], set[str]]:
    include_roots: set[Path] = {root}
    forced_includes: set[str] = set()
    response_files: set[str] = set()

    def add_flags(arguments: Iterable[str], base: Path) -> None:
        expanded, discovered_responses = _expand_response_arguments(root, base, arguments)
        response_files.update(discovered_responses)
        include_values, forced_values = _compiler_path_flags(expanded)
        local_roots: list[Path] = []
        for value in include_values:
            candidate = _argument_path(root, base, value)
            if candidate is not None and candidate.is_dir():
                include_roots.add(candidate)
                local_roots.append(candidate)
        search_roots = [base, *local_roots, *sorted(include_roots, key=lambda item: item.as_posix())]
        for value in forced_values:
            direct = _argument_path(root, base, value)
            candidates = [direct] if direct is not None else []
            if not candidates or not candidates[0].is_file():
                candidates = [_argument_path(root, item, value) for item in search_roots]
            for candidate in candidates:
                if candidate is not None and candidate.is_file():
                    forced_includes.add(_inside(root, candidate).as_posix())
                    break

    for item in manifest.build.include_dirs:
        candidate = _argument_path(root, root, item)
        if candidate is not None and candidate.is_dir():
            include_roots.add(candidate)
    add_flags(manifest.build.cflags, root)
    for unit in manifest.build.translation_units:
        base = (root / unit.directory).resolve()
        _inside(root, base)
        add_flags(unit.arguments, base)
    return (sorted(include_roots, key=lambda item: item.as_posix()),
            forced_includes, response_files)


def _discover_includes(root: Path, seeds: Iterable[str], include_roots: Iterable[Path]) -> set[str]:
    include_roots = list(include_roots)
    found: set[str] = set()
    pending = list(seeds)
    while pending:
        relative = safe_relative_path(pending.pop())
        if relative in found:
            continue
        path = (root / relative).resolve()
        _inside(root, path)
        if not path.is_file():
            continue
        found.add(relative)
        for include, is_quoted in _local_includes(path):
            search_roots = ([path.parent] if is_quoted else []) + include_roots
            for base in search_roots:
                candidate = (base / include).resolve()
                try:
                    rel = _inside(root, candidate).as_posix()
                except ManifestError:
                    continue
                if candidate.is_file():
                    pending.append(rel)
                    break
    return found


def _artifact_kind(path: str, role: str | None = None) -> str:
    suffix = Path(path).suffix.lower().lstrip(".") or "binary"
    if role == "testbench":
        return f"testbench.{suffix}"
    if role == "golden":
        return f"golden.{suffix}"
    if suffix in {"c", "cc", "cpp", "cxx", "h", "hh", "hpp", "hxx"}:
        return f"source.{suffix}"
    if suffix == "tcl":
        return "config.tcl"
    if suffix == "xdc":
        return "constraint.xdc"
    return f"file.{suffix}"


def artifact_from_path(project_root: str | Path, relative: str, *, kind: str | None = None,
                       role: str | None = None, license: str | None = None,
                       access: AccessPolicy | str = AccessPolicy.PRIVATE,
                       retention: RetentionPolicy | str = RetentionPolicy.EXTERNAL,
                       metadata: dict[str, Any] | None = None) -> ArtifactRef:
    root = Path(project_root).resolve()
    relative = safe_relative_path(relative)
    path = (root / relative).resolve()
    _inside(root, path)
    if not path.is_file():
        raise ManifestError(f"artifact does not exist: {relative}")
    data = path.read_bytes()
    return ArtifactRef(kind=kind or _artifact_kind(relative, role), uri=relative,
                       sha256=hash_artifact_bytes(data), size=len(data),
                       media_type=mimetypes.guess_type(path.name)[0], role=role,
                       license=license, access=access, retention=retention,
                       metadata=dict(metadata or {}))


def collect_artifacts(manifest: ProjectManifest, project_root: str | Path) -> list[ArtifactRef]:
    root = Path(project_root).resolve()
    explicit: dict[str, dict[str, Any]] = {}
    for item in manifest.artifact_paths:
        if "path" not in item:
            raise ManifestError("artifact_paths entries require path")
        path = safe_relative_path(str(item["path"]), "artifact path")
        explicit[path] = dict(item)
    roles: dict[str, str] = {}
    for unit in manifest.build.translation_units:
        roles.setdefault(unit.file, "design_source")
    for path in manifest.build.testbench_files:
        roles[path] = "testbench"
    for path in manifest.build.golden_files:
        roles[path] = "golden"
    for path in manifest.build.config_files:
        roles[path] = "hls_config"
    for path in manifest.build.tcl_files:
        roles[path] = "hls_tcl"
    for path in manifest.constraints.xdc_files:
        roles[path] = "constraint"
    seeds = set(roles) | set(explicit)
    include_roots, forced_includes, response_files = _compilation_paths(manifest, root)
    for path in forced_includes:
        roles.setdefault(path, "header" if Path(path).suffix.lower() in SOURCE_SUFFIXES
                         else "dependency")
    for path in response_files:
        roles.setdefault(path, "build_context")
    seeds |= forced_includes | response_files
    includes = _discover_includes(root, seeds, include_roots)
    for path in includes:
        roles.setdefault(path, "header" if Path(path).suffix.lower() in SOURCE_SUFFIXES else "dependency")
    result: list[ArtifactRef] = []
    for path in sorted(seeds | includes):
        item = explicit.get(path, {})
        role = str(item.get("role") or roles.get(path) or "input")
        result.append(artifact_from_path(
            root, path, kind=item.get("kind"), role=role, license=item.get("license"),
            access=item.get("access", AccessPolicy.PRIVATE.value),
            retention=item.get("retention", RetentionPolicy.EXTERNAL.value),
            metadata=item.get("metadata", {}),
        ))
    compile_commands = manifest.build.compile_commands
    if compile_commands and (root / compile_commands).is_file() and compile_commands not in seeds:
        result.append(artifact_from_path(root, compile_commands, kind="build.compile_commands",
                                         role="build_context", access=AccessPolicy.PRIVATE))
    unique = {artifact.id: artifact for artifact in result}
    return [unique[key] for key in sorted(unique)]


def make_snapshot(manifest: ProjectManifest, artifacts: Iterable[ArtifactRef], *,
                  parent_snapshot_id: str | None = None, action_id: str | None = None,
                  extraction_hash: str = "") -> DesignSnapshot:
    artifacts = list(artifacts)
    return DesignSnapshot(
        project_id=manifest.project_id,
        manifest_hash=stable_hash(manifest.identity_payload()),
        artifact_hashes=artifact_hash_map(artifacts),
        build_hash=stable_hash(manifest.build),
        target_hash=stable_hash(manifest.target),
        constraint_hash=stable_hash(manifest.constraints),
        toolchain_hash=stable_hash({
            "toolchains": manifest.toolchains,
            "stage_toolchains": manifest.stage_toolchains,
        }),
        extraction_hash=extraction_hash,
        parent_snapshot_id=parent_snapshot_id,
        action_id=action_id,
    )


def minimal_manifest(project_id: str, name: str, top: str, source: str, *,
                     part: str | None = None, clock_ns: float = 5.0) -> ProjectManifest:
    source = safe_relative_path(source, "source")
    return ProjectManifest(
        project_id=project_id,
        name=name,
        build=BuildContext(top=top, translation_units=[
            TranslationUnit(file=source, arguments=["-std=c++17"])
        ]),
        target=TargetProfile(part=part, clocks=[] if clock_ns is None else [
            ClockConstraint(name="default", period_ns=clock_ns)
        ]),
        constraints=ConstraintSet(),
        artifact_paths=[{"path": source, "kind": _artifact_kind(source),
                         "role": "design_source", "access": "private"}],
    )
