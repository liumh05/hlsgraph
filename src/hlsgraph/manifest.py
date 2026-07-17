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
TEXTUAL_INCLUDE_SUFFIXES = SOURCE_SUFFIXES | {
    "", ".def", ".i", ".ii", ".inc", ".inl", ".ipp", ".tpp", ".txx",
}


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


def parse_manifest_text(text: str, *, format: str = "toml") -> ProjectManifest:
    """Validate a manifest before publishing it to the filesystem."""
    try:
        if format == "toml":
            data = tomllib.loads(text)
        elif format == "json":
            data = json.loads(text)
        else:
            raise ManifestError("manifest format must be toml or json")
        return ProjectManifest.from_dict(data)
    except ManifestError:
        raise
    except (KeyError, TypeError, ValueError, tomllib.TOMLDecodeError) as exc:
        raise ManifestError(f"invalid manifest text: {exc}") from exc


def write_internal_manifest(path: str | Path, manifest: ProjectManifest) -> None:
    Path(path).write_text(json.dumps(json_ready(manifest), ensure_ascii=False, indent=2,
                                    sort_keys=True) + "\n", encoding="utf-8")


def _toml_basic_string(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ManifestError(f"{field_name} must be a non-empty string")
    if "\x00" in value:
        raise ManifestError(f"{field_name} must not contain NUL")
    # JSON string escaping is a strict subset of TOML basic-string escaping for
    # the characters emitted by json.dumps.  This keeps quotes, backslashes,
    # controls, and non-ASCII project names valid without hand-built escaping.
    return json.dumps(value, ensure_ascii=False)


def manifest_template(project_id: str, name: str, top: str, source: str) -> str:
    source = safe_relative_path(source, "source")
    project_id_toml = _toml_basic_string(project_id, "project_id")
    name_toml = _toml_basic_string(name, "name")
    top_toml = _toml_basic_string(top, "top")
    source_toml = _toml_basic_string(source, "source")
    return f'''schema_version = "0.1.0"
project_id = {project_id_toml}
name = {name_toml}
stage_toolchains = {{}}

[build]
top = {top_toml}
language = "c++"
flow_target = "vitis"
compile_commands = "compile_commands.json"
include_dirs = []
config_files = []
tcl_files = []
testbench_files = []
golden_files = []

[[build.translation_units]]
file = {source_toml}
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
path = {source_toml}
kind = "source.cpp"
role = "design_source"
access = "private"
'''


def _split_windows_command_line(command: str) -> list[str]:
    """Split one Windows compiler command using CommandLineToArgvW/CRT rules.

    ``shlex(posix=False)`` retains quotes and incorrectly splits common forms
    such as ``-I\"include dir\"``.  Keeping this implementation pure Python also
    lets manifests produced on Windows be audited on another platform.
    """
    if not isinstance(command, str) or not command.strip():
        raise ManifestError("compiler command must be a non-empty string")
    result: list[str] = []
    length = len(command)
    index = 0
    whitespace = " \t\r\n"
    while index < length:
        while index < length and command[index] in whitespace:
            index += 1
        if index >= length:
            break
        argument: list[str] = []
        quoted = False
        started = False
        while index < length:
            if command[index] in whitespace and not quoted:
                break
            backslashes = 0
            while index < length and command[index] == "\\":
                backslashes += 1
                index += 1
            if index < length and command[index] == '"':
                argument.extend("\\" for _ in range(backslashes // 2))
                if backslashes % 2:
                    argument.append('"')
                elif quoted and index + 1 < length and command[index + 1] == '"':
                    argument.append('"')
                    index += 1
                else:
                    quoted = not quoted
                index += 1
                started = True
                continue
            argument.extend("\\" for _ in range(backslashes))
            if backslashes:
                started = True
            if index >= length:
                break
            # A run of ordinary backslashes does not escape unquoted
            # whitespace.  Re-check the delimiter after consuming the run so
            # ``C:\\include\\ -c`` remains two arguments instead of absorbing
            # the following option into the include path.
            if command[index] in whitespace and not quoted:
                break
            argument.append(command[index])
            index += 1
            started = True
        if quoted:
            raise ManifestError("unterminated quote in Windows compiler command")
        if started:
            result.append("".join(argument))
        while index < length and command[index] in whitespace:
            index += 1
    if not result:
        raise ManifestError("compiler command must contain at least one argument")
    return result


def split_compilation_command(command: str, *, platform: str | None = None) -> list[str]:
    platform = os.name if platform is None else platform
    if platform == "nt":
        return _split_windows_command_line(command)
    try:
        result = shlex.split(command, posix=True)
    except ValueError as exc:
        raise ManifestError(f"invalid compiler command: {exc}") from exc
    if not result:
        raise ManifestError("compiler command must contain at least one argument")
    return result


def load_compile_commands(path: str | Path, project_root: str | Path, *,
                          platform: str | None = None) -> list[TranslationUnit]:
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
            arguments = split_compilation_command(str(row["command"]), platform=platform)
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
        r'^\s*#\s*(?:include(?:_next)?|import)\s*([<"])([^">]+)[>"]',
        text, re.MULTILINE,
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
        # Callers decide whether an external path is an unsupported direct
        # input or an optional lookup.  Never silently reinterpret it as a
        # project artifact.
        return None
    return candidate


def _compiler_path_flags(arguments: Iterable[str]) -> tuple[list[str], list[str], list[str]]:
    """Extract include directories, forced includes, and direct compiler inputs."""
    args = list(arguments)
    include_dirs: list[str] = []
    forced: list[str] = []
    direct_inputs: list[str] = []
    index = 0
    while index < len(args):
        arg = args[index]
        slash_arg = arg.casefold() if arg.startswith("/") else arg
        next_value: str | None = None
        destination: list[str] | None = None
        if arg in {
            "-I", "/I", "-isystem", "-iquote", "-idirafter",
            "--include-directory", "-F", "-iframework", "-imsvc",
            "/imsvc", "/external:I", "-cxx-isystem", "-stdlib++-isystem",
            "-isystem-after",
        } or slash_arg in {"/i", "/imsvc", "/external:i"}:
            destination = include_dirs
            if index + 1 < len(args):
                next_value = args[index + 1]
                index += 1
        elif arg in {"-include", "-imacros", "/FI"} or slash_arg == "/fi":
            destination = forced
            if index + 1 < len(args):
                next_value = args[index + 1]
                index += 1
        elif arg in {"-fmodule-map-file", "-include-pch"}:
            destination = direct_inputs
            if index + 1 < len(args):
                next_value = args[index + 1]
                index += 1
        elif arg == "-fmodule-file":
            destination = direct_inputs
            if index + 1 < len(args):
                next_value = _module_file_path(args[index + 1])
                index += 1
        elif arg == "-ivfsoverlay":
            destination = direct_inputs
            if index + 1 < len(args):
                next_value = args[index + 1]
                index += 1
        else:
            joined_prefixes = (
                ("--include-directory=", include_dirs),
                ("-isystem=", include_dirs),
                ("-isystem-after=", include_dirs),
                ("-iquote=", include_dirs),
                ("-idirafter=", include_dirs),
                ("-iframework=", include_dirs),
                ("-cxx-isystem=", include_dirs),
                ("-stdlib++-isystem=", include_dirs),
                ("-include=", forced),
                ("-imacros=", forced),
                ("-fmodule-map-file=", direct_inputs),
                ("-include-pch=", direct_inputs),
                ("-ivfsoverlay=", direct_inputs),
            )
            for prefix, target in joined_prefixes:
                if arg.startswith(prefix) and len(arg) > len(prefix):
                    destination, next_value = target, arg[len(prefix):]
                    break
            if (destination is None and arg.startswith("-fmodule-file=")
                    and len(arg) > len("-fmodule-file=")):
                destination = direct_inputs
                next_value = _module_file_path(arg[len("-fmodule-file="):])
            if destination is None:
                for prefix in (
                    "/external:I", "-stdlib++-isystem", "-isystem-after",
                    "-cxx-isystem", "-iframework", "/imsvc", "-imsvc", "-F",
                    "-I", "/I",
                ):
                    comparable = arg.casefold() if prefix.startswith("/") else arg
                    expected = prefix.casefold() if prefix.startswith("/") else prefix
                    if comparable.startswith(expected) and len(arg) > len(prefix):
                        destination, next_value = include_dirs, arg[len(prefix):]
                        break
                if (destination is None and slash_arg.startswith("/fi")
                        and len(arg) > 3):
                    destination, next_value = forced, arg[3:]
        if destination is not None and next_value:
            destination.append(next_value)
        index += 1
    return include_dirs, forced, direct_inputs


def _expand_response_arguments(root: Path, base: Path, arguments: Iterable[str], *,
                               platform: str | None = None,
                               seen: set[Path] | None = None) -> tuple[list[str], set[str]]:
    expanded: list[str] = []
    response_files: set[str] = set()
    seen = set() if seen is None else seen
    for argument in arguments:
        if not argument.startswith("@") or len(argument) == 1:
            expanded.append(argument)
            continue
        response_value = argument[1:].strip().strip('"').replace(
            "${PROJECT_ROOT}", str(root),
        )
        response = Path(response_value)
        if not response.is_absolute():
            response = base / response
        response = response.resolve()
        try:
            response_relative = _inside(root, response).as_posix()
        except ManifestError as exc:
            raise ManifestError(
                "compiler response files must be project-local: " + response_value
            ) from exc
        if not response.is_file():
            raise ManifestError(
                f"compiler response file does not exist: {response_relative}"
            )
        if response in seen:
            raise ManifestError(f"recursive compiler response file: {_inside(root, response).as_posix()}")
        if response.stat().st_size > 4 * 1024 * 1024:
            raise ManifestError(f"compiler response file is too large: {_inside(root, response).as_posix()}")
        seen.add(response)
        relative = response_relative
        response_files.add(relative)
        raw_response = response.read_bytes()
        try:
            if raw_response.startswith((b"\xff\xfe", b"\xfe\xff")):
                response_text = raw_response.decode("utf-16")
            else:
                response_text = raw_response.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ManifestError(
                f"compiler response file is not valid UTF-8/UTF-16: {relative}"
            ) from exc
        if "\x00" in response_text:
            raise ManifestError(
                f"compiler response file has unsupported NUL bytes: {relative}"
            )
        nested_args = (split_compilation_command(response_text, platform=platform)
                       if response_text.strip() else [])
        nested, nested_files = _expand_response_arguments(
            # Compiler response-file arguments are textual argv substitution;
            # relative paths retain the compilation working directory.
            root, base, nested_args, platform=platform, seen=seen,
        )
        expanded.extend(nested)
        response_files.update(nested_files)
        seen.remove(response)
    return expanded, response_files


_SEPARATE_PATH_FLAGS = frozenset({
    "-I", "/I", "-isystem", "-iquote", "-idirafter", "--include-directory",
    "-F", "-iframework", "-imsvc", "/imsvc", "/external:I",
    "-cxx-isystem", "-stdlib++-isystem", "-isystem-after",
    "-isysroot", "--sysroot", "-fmodule-map-file", "-include-pch",
    "-ivfsoverlay",
})
_MODULE_FILE_FLAG = "-fmodule-file"
_FORCED_INCLUDE_FLAGS = frozenset({"-include", "-imacros", "/FI"})
_JOINED_PATH_PREFIXES = (
    "--include-directory=", "-fmodule-map-file=", "-include-pch=", "--sysroot=",
    "-stdlib++-isystem=", "-isystem-after=", "-cxx-isystem=",
    "-ivfsoverlay=", "-iframework=",
    "-isystem=", "-iquote=", "-idirafter=", "-isysroot=",
    # Longest/raw prefixes must precede their shorter families.  Otherwise
    # ``-isystem-after...`` would be interpreted as ``-isystem`` plus a path
    # beginning with ``-after``.
    "-stdlib++-isystem", "-isystem-after", "-cxx-isystem",
    "/external:I", "-iframework", "/imsvc", "-imsvc",
    "-isystem", "-iquote", "-idirafter", "-F", "-I", "/I",
)
_JOINED_FORCED_PREFIXES = ("-include=", "-imacros=", "/FI")

_UNSUPPORTED_CONTEXT_FLAGS = frozenset({
    # Clang configuration files can inject arbitrary additional argv and have
    # their own search/include semantics.  v0.1 does not parse that language.
    "--config", "--config-system-dir", "--config-user-dir",
    # These alter implicit header discovery through directory trees that are
    # not represented by the project-local artifact closure.
    "-resource-dir", "-isysroot", "--sysroot", "-iwithsysroot",
    "-iprefix", "-iwithprefix", "-iwithprefixbefore",
    "-iframeworkwithsysroot", "--gcc-toolchain", "-gcc-toolchain",
    "-ccc-install-dir", "-B",
    # Precompiled/implicit module and PCH formats can consult original headers,
    # module maps, and caches in addition to their named binary.  Hashing only
    # the top-level PCM/PCH is not a closed dependency envelope.
    "-include-pch", "-include-pth", "-fmodule-map-file", "-fmodule-file",
    "-fmodules", "-fcxx-modules", "-fimplicit-module-maps",
    "-fimplicit-modules", "-fmodules-cache-path", "-fprebuilt-module-path",
    "-fmodules-ts", "-fmodule-header", "-fmodule-output", "-fpch-preprocess",
    # Host-native feature discovery makes predefined macros depend on the
    # indexing machine instead of the manifest.
    "-march=native", "-mcpu=native", "-mtune=native",
    "/yu", "/yc", "/fp",
    # Compiler escape hatches/plugins are both untracked inputs and in-process
    # code execution.  The HLSGraph extractor plugin protocol is the only
    # supported extension boundary.
    "-Xclang", "-Xpreprocessor", "-load", "-plugin", "-add-plugin",
    "-fplugin",
})
_UNSUPPORTED_CONTEXT_PREFIXES = (
    "--config=", "--config-system-dir=", "--config-user-dir=",
    "-resource-dir=", "-isysroot=", "--sysroot=", "-iwithsysroot=",
    "-iprefix=", "-iwithprefix=", "-iwithprefixbefore=",
    "-iframeworkwithsysroot=", "--gcc-toolchain=", "-gcc-toolchain=",
    "-ccc-install-dir=", "-fplugin=", "-fplugin-arg-", "-Wp,", "/clang:",
    "-include-pch=", "-include-pth=", "-fmodule-map-file=", "-fmodule-file=",
    "-fmodules-cache-path=", "-fprebuilt-module-path=", "-fmodule-header=",
    "-fmodule-output=", "/headerunit:", "/reference", "/ifcsearchdir",
    "/module:", "/external:env:",
    "/yu", "/yc", "/fp",
)
_UNSUPPORTED_JOINED_CONTEXT_PREFIXES = (
    "-resource-dir", "-iwithsysroot", "-iprefix", "-iwithprefixbefore",
    "-iwithprefix", "-iframeworkwithsysroot", "-B",
)


def _reject_unsupported_compiler_context(arguments: Iterable[str]) -> None:
    for argument in arguments:
        candidate = argument.casefold() if argument.startswith("/") else argument
        if candidate in _UNSUPPORTED_CONTEXT_FLAGS:
            label = candidate
        else:
            label = next(
                (prefix.rstrip("=") for prefix in _UNSUPPORTED_CONTEXT_PREFIXES
                 if candidate.startswith(prefix)),
                None,
            )
            if label is None:
                label = next(
                    (prefix for prefix in _UNSUPPORTED_JOINED_CONTEXT_PREFIXES
                     if candidate.startswith(prefix) and len(candidate) > len(prefix)),
                    None,
                )
        if label is not None:
            raise ManifestError(
                f"compiler option {label} is unsupported in the deterministic "
                "v0.1 source extractor"
            )


def _absolute_compiler_path(root: Path, base: Path, value: str) -> str:
    value = value.replace("${PROJECT_ROOT}", str(root))
    path = Path(value)
    return str(path if path.is_absolute() else (base / path).resolve())


def _module_file_path(value: str) -> str:
    """Return the path portion of Clang's optional ``name=module.pcm`` value."""
    return value.split("=", 1)[1] if "=" in value else value


def _absolute_module_file(root: Path, base: Path, value: str) -> str:
    if "=" not in value:
        return _absolute_compiler_path(root, base, value)
    name, path = value.split("=", 1)
    return f"{name}={_absolute_compiler_path(root, base, path)}"


def _forced_include_path(root: Path, base: Path, value: str) -> str:
    """Resolve direct forced-include paths but preserve include-search names."""
    expanded = value.replace("${PROJECT_ROOT}", str(root))
    candidate = Path(expanded)
    if candidate.is_absolute():
        return str(candidate)
    resolved = (base / candidate).resolve()
    if resolved.is_file() or "/" in expanded or "\\" in expanded:
        return str(resolved)
    return expanded


def resolve_compiler_arguments(project_root: str | Path, directory: str | Path,
                               arguments: Iterable[str], *,
                               platform: str | None = None) -> tuple[list[str], set[str]]:
    """Expand response files and anchor path-valued flags to their TU directory.

    The returned argv is shared by snapshot closure discovery and libclang, so
    the bytes attributed to a snapshot match the compilation context actually
    parsed.  Project-local response files are returned separately for hashing.
    """
    root = Path(project_root).resolve()
    base = Path(directory)
    if not base.is_absolute():
        base = root / base
    base = base.resolve()
    _inside(root, base)
    expanded, response_files = _expand_response_arguments(
        root, base, arguments, platform=platform,
    )
    _reject_unsupported_compiler_context(expanded)
    result: list[str] = []
    index = 0
    while index < len(expanded):
        argument = expanded[index]
        if argument == _MODULE_FILE_FLAG:
            if index + 1 >= len(expanded):
                raise ManifestError(f"compiler flag {argument} requires a path")
            result.append(argument)
            result.append(_absolute_module_file(root, base, expanded[index + 1]))
            index += 2
            continue
        path_flag = next((
            flag for flag in _SEPARATE_PATH_FLAGS | _FORCED_INCLUDE_FLAGS
            if (argument.casefold() == flag.casefold()
                if flag.startswith("/") else argument == flag)
        ), None)
        if path_flag is not None:
            if index + 1 >= len(expanded):
                raise ManifestError(f"compiler flag {argument} requires a path")
            result.append(argument)
            value = expanded[index + 1]
            resolver = (_forced_include_path if path_flag in _FORCED_INCLUDE_FLAGS
                        else _absolute_compiler_path)
            result.append(resolver(root, base, value))
            index += 2
            continue
        matched = False
        if (argument.startswith("-fmodule-file=")
                and len(argument) > len("-fmodule-file=")):
            result.append("-fmodule-file=" + _absolute_module_file(
                root, base, argument[len("-fmodule-file="):],
            ))
            matched = True
        for prefix in _JOINED_PATH_PREFIXES:
            if matched:
                break
            comparable = argument.casefold() if prefix.startswith("/") else argument
            expected = prefix.casefold() if prefix.startswith("/") else prefix
            if comparable.startswith(expected) and len(argument) > len(prefix):
                result.append(argument[:len(prefix)] + _absolute_compiler_path(
                    root, base, argument[len(prefix):],
                ))
                matched = True
                break
        if not matched:
            for prefix in _JOINED_FORCED_PREFIXES:
                comparable = argument.casefold() if prefix.startswith("/") else argument
                expected = prefix.casefold() if prefix.startswith("/") else prefix
                if comparable.startswith(expected) and len(argument) > len(prefix):
                    result.append(argument[:len(prefix)] + _forced_include_path(
                        root, base, argument[len(prefix):],
                    ))
                    matched = True
                    break
        if not matched:
            if "${PROJECT_ROOT}" in argument:
                raise ManifestError(
                    "${PROJECT_ROOT} is only supported in compiler response-file "
                    "locations and recognized path-valued compiler options"
                )
            result.append(argument)
        index += 1
    return result, response_files


def _compilation_paths(
    manifest: ProjectManifest, root: Path,
) -> tuple[list[Path], set[str], set[str], set[str]]:
    include_roots: set[Path] = {root}
    forced_includes: set[str] = set()
    response_files: set[str] = set()
    direct_inputs: set[str] = set()
    contexts: list[tuple[Path, list[str], list[str]]] = []

    def collect_context(arguments: Iterable[str], base: Path) -> None:
        expanded, discovered_responses = resolve_compiler_arguments(root, base, arguments)
        response_files.update(discovered_responses)
        include_values, forced_values, direct_values = _compiler_path_flags(expanded)
        for value in include_values:
            candidate = _argument_path(root, base, value)
            if candidate is None or not candidate.is_dir():
                # Explicit external or not-yet-created search roots can change
                # the AST without changing any project-local byte.  v0.1 has
                # no hash-only external dependency envelope, so accepting them
                # would allow one snapshot ID to name different graphs.
                raise ManifestError(
                    "compiler include search roots must be existing "
                    "project-local directories: " + value
                )
            include_roots.add(candidate)
        contexts.append((base, forced_values, direct_values))

    for item in manifest.build.include_dirs:
        candidate = _argument_path(root, root, item)
        if candidate is None or not candidate.is_dir():
            raise ManifestError(
                "manifest include directories must be existing project-local "
                "directories: " + item
            )
        include_roots.add(candidate)

    # Resolve every include search root before any forced include.  Global
    # cflags are appended to each translation unit by the extractor, so a
    # global ``-include forced.hpp`` may legitimately resolve through a TU's
    # ``-I`` directory.  A one-pass traversal made that result depend on the
    # order in which contexts happened to be visited.
    collect_context(manifest.build.cflags, root)
    for unit in manifest.build.translation_units:
        base = (root / unit.directory).resolve()
        _inside(root, base)
        collect_context(unit.arguments, base)

    all_search_roots = sorted(include_roots, key=lambda item: item.as_posix())
    for base, forced_values, direct_values in contexts:
        search_roots = [base, *all_search_roots]
        for value in forced_values:
            direct = _argument_path(root, base, value)
            candidates = [direct] if direct is not None else []
            if not candidates or not candidates[0].is_file():
                candidates = [_argument_path(root, item, value) for item in search_roots]
            matched = False
            for candidate in candidates:
                if candidate is not None and candidate.is_file():
                    forced_includes.add(_inside(root, candidate).as_posix())
                    # Hash every project-local match.  This intentionally
                    # over-approximates include-order resolution across TUs;
                    # omitting an alternate match would be unsound.
                    matched = True
            if not matched:
                raise ManifestError(
                    "forced compiler includes must resolve to an existing "
                    "project-local file: " + value
                )
        for value in direct_values:
            candidate = _argument_path(root, base, value)
            if candidate is None or not candidate.is_file():
                raise ManifestError(
                    "direct compiler inputs must be existing project-local files: "
                    + value
                )
            direct_inputs.add(_inside(root, candidate).as_posix())
    return (sorted(include_roots, key=lambda item: item.as_posix()),
            forced_includes, response_files, direct_inputs)


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
        if path.suffix.casefold() not in TEXTUAL_INCLUDE_SUFFIXES:
            continue
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
                    # Hash every project-local candidate rather than trying to
                    # reimplement Clang's ordered quote/I/system/after search
                    # categories. This safe over-approximation also covers
                    # differing search order across translation units.
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
    include_roots, forced_includes, response_files, direct_inputs = _compilation_paths(
        manifest, root,
    )
    for path in forced_includes:
        roles.setdefault(path, "header" if Path(path).suffix.lower() in SOURCE_SUFFIXES
                         else "dependency")
    for path in response_files:
        roles.setdefault(path, "build_context")
    for path in direct_inputs:
        roles.setdefault(path, "compiler_input")
    seeds |= forced_includes | response_files | direct_inputs
    includes = _discover_includes(root, seeds, include_roots)
    # This set is computed from compiler inputs, never from user-provided
    # ArtifactRef kind/role metadata.  The extractor uses it to scan every
    # compiler-reachable textual byte for implicit preprocessing inputs even
    # when an explicit artifact declaration gives a header an arbitrary kind.
    compiler_text_inputs = _discover_includes(
        root,
        {unit.file for unit in manifest.build.translation_units} | forced_includes,
        include_roots,
    )
    for path in includes:
        roles.setdefault(path, "header" if Path(path).suffix.lower() in SOURCE_SUFFIXES else "dependency")
    result: list[ArtifactRef] = []
    for path in sorted(seeds | includes):
        item = explicit.get(path, {})
        role = str(item.get("role") or roles.get(path) or "input")
        metadata = dict(item.get("metadata", {}))
        # Reserved, collector-owned marker: an explicit metadata value cannot
        # opt a reachable source/header out of deterministic safety scanning.
        metadata.pop("hlsgraph.compiler_reachable_text", None)
        if path in compiler_text_inputs:
            metadata["hlsgraph.compiler_reachable_text"] = True
        result.append(artifact_from_path(
            root, path, kind=item.get("kind"), role=role, license=item.get("license"),
            access=item.get("access", AccessPolicy.PRIVATE.value),
            retention=item.get("retention", RetentionPolicy.EXTERNAL.value),
            metadata=metadata,
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
