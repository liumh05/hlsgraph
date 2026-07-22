"""Frozen manifest loading and validation shared by the public A/B tools."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import sqlite3
import stat
import subprocess
import sys
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
SUITE_PATH = HERE / "manifest.json"
QUESTIONS_PATH = HERE / "questions.jsonl"
CORPUS_LOCK_PATH = HERE / "corpus.lock.json"
STATIC_CASES_PATH = HERE / "static_cases.jsonl"
ANSWER_SCHEMA_PATH = HERE / "answer.schema.json"

SCHEMA_VERSION = "hlsgraph.agent_eval.v1"
CORPUS_SCHEMA_VERSION = "hlsgraph.agent_eval.corpus_lock.v1"
QUESTION_SCHEMA_VERSION = "hlsgraph.agent_eval.question.v1"
STATIC_CASE_SCHEMA_VERSION = "hlsgraph.agent_eval.static_case.v1"
ARM_IDS = ("native", "codegraph", "hlsgraph-v02", "hlsgraph-v03")
ENVIRONMENT_SCHEMA_VERSION = "hlsgraph.agent_eval.environment.v3"
RUNTIME_IDENTITY_SCHEMA_VERSION = "hlsgraph.agent_eval.runtime_identity.v2"
SANDBOX_BOUNDARY_SCHEMA_VERSION = "hlsgraph.agent_eval.sandbox_boundary.v1"
CODEGRAPH_BUILD_IDENTITY_SCHEMA_VERSION = "hlsgraph.agent_eval.codegraph_build.v1"
RUNTIME_TREE_ALGORITHM = "hlsgraph.runtime_tree.v1"
OFFICIAL_CODEX_VERSION = "codex-cli 0.144.0"
CODEGRAPH_OFFLINE_ENV = {
    "CODEGRAPH_NO_DAEMON": "1",
    "CODEGRAPH_TELEMETRY": "0",
    "DO_NOT_TRACK": "1",
    "CODEGRAPH_NO_UPDATE_CHECK": "1",
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")

_FORBIDDEN_AUTH_ENV = (
    "OPENAI_API_KEY", "CODEX_API_KEY", "CODEX_ACCESS_TOKEN",
)


class EvalManifestError(ValueError):
    """Raised when a frozen evaluation asset is malformed or inconsistent."""


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            + "\n").encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_tree_identity(root: Path) -> str:
    """Hash one executable tree without following links.

    ``hlsgraph.runtime_tree.v1`` is deliberately independent of tar, platform
    archive metadata, directory mtimes, and enumeration order.  Directory
    records bind relative path, type, and POSIX mode.  Regular-file records bind
    relative path, type, POSIX mode, and the SHA-256 of stable file bytes.
    Symlink records bind relative path, type, and the link text.  A link is
    accepted only when its complete, non-dangling chain resolves inside the
    tree to a regular file or directory.  Absolute, escaping, dangling, cyclic,
    and special-file links fail closed.  Special filesystem entries fail too.
    """

    lexical_root = Path(os.path.abspath(os.fspath(root)))
    _assert_unlinked_within(Path(lexical_root.anchor), lexical_root)
    if _linked(lexical_root) or not lexical_root.is_dir():
        raise EvalManifestError(f"runtime tree root is missing or linked: {lexical_root}")
    resolved_root = lexical_root.resolve(strict=True)
    records: list[tuple[str, bytes]] = []
    pending = [lexical_root]
    while pending:
        directory = pending.pop()
        try:
            _assert_unlinked_within(lexical_root, directory)
            before_directory = directory.lstat()
            if not stat.S_ISDIR(before_directory.st_mode) or _linked(directory):
                raise EvalManifestError(
                    f"runtime-tree directory became linked or non-directory: {directory}"
                )
            directory.resolve(strict=True).relative_to(resolved_root)
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda item: item.name)
            _assert_unlinked_within(lexical_root, directory)
            after_directory = directory.lstat()
        except (OSError, RuntimeError, ValueError) as exc:
            raise EvalManifestError(f"cannot enumerate runtime tree: {directory}: {exc}") from exc
        directory_identity = lambda info: (
            info.st_dev, info.st_ino, info.st_mode, info.st_mtime_ns,
        )
        if directory_identity(before_directory) != directory_identity(after_directory):
            raise EvalManifestError(
                f"runtime-tree directory changed while enumerated: {directory}"
            )
        directory_relative = (
            "." if directory == lexical_root
            else directory.relative_to(lexical_root).as_posix()
        )
        records.append((
            directory_relative,
            b"D\0" + directory_relative.encode("utf-8", errors="strict") + b"\0"
            + f"{stat.S_IMODE(before_directory.st_mode):04o}".encode("ascii") + b"\0",
        ))
        for entry in entries:
            path = Path(entry.path)
            try:
                _assert_unlinked_within(lexical_root, path.parent)
                relative = path.relative_to(lexical_root).as_posix()
                relative_bytes = relative.encode("utf-8", errors="strict")
                before = path.lstat()
            except (OSError, UnicodeError, ValueError) as exc:
                raise EvalManifestError(f"invalid runtime-tree entry: {path}: {exc}") from exc
            mode = before.st_mode
            if stat.S_ISDIR(mode):
                pending.append(path)
                continue
            if stat.S_ISREG(mode):
                data = _stable_file_bytes(path)
                _assert_unlinked_within(lexical_root, path.parent)
                after = path.lstat()
                identity = lambda info: (
                    info.st_dev, info.st_ino, info.st_mode, info.st_size,
                    info.st_mtime_ns,
                )
                if identity(before) != identity(after):
                    raise EvalManifestError(f"runtime-tree file changed while hashed: {relative}")
                record = (
                    b"F\0" + relative_bytes + b"\0"
                    + f"{stat.S_IMODE(mode):04o}".encode("ascii") + b"\0"
                    + hashlib.sha256(data).digest() + b"\0"
                )
                records.append((relative, record))
                continue
            if stat.S_ISLNK(mode):
                try:
                    target = os.readlink(path)
                    target_bytes = target.encode("utf-8", errors="strict")
                except (OSError, UnicodeError) as exc:
                    raise EvalManifestError(
                        f"cannot read runtime-tree symlink: {relative}: {exc}"
                    ) from exc
                if Path(target).is_absolute():
                    raise EvalManifestError(
                        f"runtime-tree symlink is absolute: {relative} -> {target}"
                    )
                try:
                    lexical_target = Path(os.path.normpath(os.fspath(path.parent / target)))
                    lexical_target.relative_to(lexical_root)
                    resolved_target = (path.parent / target).resolve(strict=True)
                    resolved_target.relative_to(resolved_root)
                    target_info = resolved_target.stat()
                except (OSError, RuntimeError, ValueError) as exc:
                    raise EvalManifestError(
                        f"runtime-tree symlink is dangling or escapes: {relative} -> {target}"
                    ) from exc
                if not (stat.S_ISREG(target_info.st_mode) or stat.S_ISDIR(target_info.st_mode)):
                    raise EvalManifestError(
                        f"runtime-tree symlink targets a special file: {relative}"
                    )
                after = path.lstat()
                _assert_unlinked_within(lexical_root, path.parent)
                link_identity = lambda info: (
                    info.st_dev, info.st_ino, info.st_mode, info.st_size,
                    info.st_mtime_ns,
                )
                if link_identity(before) != link_identity(after) or os.readlink(path) != target:
                    raise EvalManifestError(
                        f"runtime-tree symlink changed while hashed: {relative}"
                    )
                records.append((relative, b"L\0" + relative_bytes + b"\0" + target_bytes + b"\0"))
                continue
            raise EvalManifestError(f"runtime tree contains a special file: {relative}")
        try:
            _assert_unlinked_within(lexical_root, directory)
            final_directory = directory.lstat()
        except (OSError, ValueError) as exc:
            raise EvalManifestError(
                f"runtime-tree directory changed while hashed: {directory}: {exc}"
            ) from exc
        if directory_identity(before_directory) != directory_identity(final_directory):
            raise EvalManifestError(
                f"runtime-tree directory changed while hashed: {directory}"
            )

    digest = hashlib.sha256()
    digest.update(RUNTIME_TREE_ALGORITHM.encode("ascii") + b"\0")
    for _relative, record in sorted(records, key=lambda item: item[0]):
        digest.update(record)
    return digest.hexdigest()


def resolve_local_executable(value: str, root: Path | None = None) -> str:
    """Resolve an explicitly path-like executable without rewriting PATH names.

    The evaluation changes working directory for every corpus.  A relative
    ``.venv/Scripts/python.exe`` would otherwise be interpreted inside that
    corpus and silently select the wrong runtime (or fail to start).  Bare
    commands such as ``python`` and ``codex.cmd`` deliberately remain subject
    to the caller's frozen PATH.
    """
    if not isinstance(value, str) or not value.strip():
        raise EvalManifestError("executable must be a non-empty string")
    candidate = Path(value)
    path_like = (
        candidate.is_absolute()
        or candidate.parent != Path(".")
        or os.sep in value
        or (os.altsep is not None and os.altsep in value)
    )
    if not path_like:
        return value
    resolved = candidate if candidate.is_absolute() else (root or Path.cwd()) / candidate
    # Keep the lexical executable path.  POSIX virtual environments commonly
    # expose ``bin/python`` as a symlink; resolving it would silently replace
    # the selected environment with the system interpreter in the frozen
    # command contract.  ``is_file`` deliberately follows the final symlink so
    # broken or non-file launchers still fail closed.
    lexical = Path(os.path.abspath(os.fspath(resolved)))
    if os.name == "posix" and not lexical.is_file():
        raise EvalManifestError(f"executable must be an existing file: {lexical}")
    return str(lexical)


def resolve_command_argv(value: str, root: Path | None = None) -> list[str]:
    """Split a command and freeze its executable before changing directory.

    In particular, Windows ``subprocess`` cannot directly launch the npm
    PowerShell shim selected for a bare ``codex`` command.  ``shutil.which``
    resolves the executable ``codex.cmd`` shim, while retaining normal PATH
    lookup semantics on other platforms.
    """
    parts = shlex.split(value)
    if not parts:
        raise EvalManifestError("command must be non-empty")
    executable = resolve_local_executable(parts[0], root)
    if executable == parts[0]:
        located = shutil.which(executable)
        if located:
            executable = resolve_local_executable(located, root)
    parts[0] = executable
    return parts


def _decode_mount_path(value: str) -> str:
    """Decode the octal escapes used by ``/proc/self/mountinfo``."""

    for escaped, decoded in (
        (r"\040", " "), (r"\011", "\t"), (r"\012", "\n"),
        (r"\134", "\\"),
    ):
        value = value.replace(escaped, decoded)
    return value


def _mount_table() -> list[tuple[Path, str]]:
    try:
        lines = Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise EvalManifestError("official evaluation cannot inspect Linux mountinfo") from exc
    mounts: list[tuple[Path, str]] = []
    for line in lines:
        fields = line.split()
        try:
            separator = fields.index("-")
            mountpoint = Path(_decode_mount_path(fields[4])).resolve()
            filesystem = fields[separator + 1]
        except (ValueError, IndexError, OSError):
            continue
        mounts.append((mountpoint, filesystem))
    mounts.sort(key=lambda item: len(item[0].parts), reverse=True)
    return mounts


def _filesystem_type(path: Path) -> str:
    resolved = path.resolve()
    for mountpoint, filesystem in _mount_table():
        try:
            resolved.relative_to(mountpoint)
        except ValueError:
            continue
        return filesystem
    raise EvalManifestError(f"official evaluation cannot identify filesystem for {resolved}")


def discover_windows_mount_roots() -> list[Path]:
    """Return every currently mounted WSL drive root (``/mnt/<letter>``)."""

    roots = {
        mountpoint for mountpoint, filesystem in _mount_table()
        if filesystem in {"9p", "drvfs"}
        and re.fullmatch(r"/mnt/[A-Za-z]", mountpoint.as_posix())
    }
    return sorted(roots, key=lambda item: item.as_posix().casefold())


def _sensitive_home_roots() -> list[Path]:
    candidates = {Path("/root"), Path("/home")}
    try:
        import pwd

        candidates.add(Path(pwd.getpwuid(os.getuid()).pw_dir))
    except (ImportError, KeyError, OSError):
        pass
    candidates.add(Path.home())
    roots = {item.resolve() for item in candidates if item.is_dir()}
    return sorted(roots, key=lambda item: item.as_posix())


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def require_official_linux_wsl2() -> dict[str, str]:
    """Fail before mutation unless this is the frozen Linux/WSL2 host class."""

    if os.name != "posix" or sys.platform != "linux" or platform.system() != "Linux":
        raise EvalManifestError(
            "official agent evaluation is NO-GO outside POSIX/Linux WSL2"
        )
    release = platform.release()
    if "microsoft-standard-wsl2" not in release.casefold():
        raise EvalManifestError(
            "official agent evaluation is NO-GO unless run inside WSL2"
        )
    os_release: dict[str, str] = {}
    try:
        for raw in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
            if "=" not in raw or raw.lstrip().startswith("#"):
                continue
            key, value = raw.split("=", 1)
            os_release[key] = value.strip().strip('"')
    except OSError as exc:
        raise EvalManifestError("official evaluation cannot read /etc/os-release") from exc
    if os_release.get("ID") != "ubuntu" or os_release.get("VERSION_ID") != "22.04":
        raise EvalManifestError(
            "official agent evaluation requires Ubuntu-22.04 under WSL2"
        )
    if platform.machine().casefold() not in {"x86_64", "amd64"}:
        raise EvalManifestError("official agent evaluation requires x86-64 Linux")
    return {
        "os_name": os.name,
        "system": platform.system(),
        "release": release,
        "machine": platform.machine(),
        "distribution_id": os_release["ID"],
        "distribution_version": os_release["VERSION_ID"],
        "wsl": "2",
        "wsl_distribution": os.environ.get("WSL_DISTRO_NAME", "Ubuntu-22.04"),
    }


def _require_regular_ext4_path(
    path: Path, label: str, *, directory: bool = False, reject_links: bool = False,
) -> Path:
    lexical = Path(os.path.abspath(os.fspath(path)))
    resolved = lexical.resolve()
    expected = resolved.is_dir() if directory else resolved.is_file()
    if not expected:
        kind = "directory" if directory else "file"
        raise EvalManifestError(f"official {label} must be an existing {kind}: {resolved}")
    is_junction = getattr(lexical, "is_junction", None)
    if reject_links and (
        lexical.is_symlink() or bool(callable(is_junction) and is_junction())
    ):
        raise EvalManifestError(f"official {label} must not be a link or junction")
    if _filesystem_type(resolved) != "ext4":
        raise EvalManifestError(
            f"official {label} must live on WSL ext4, not {_filesystem_type(resolved)}"
        )
    return lexical


def require_official_ext4_directory(
    path: Path, label: str, *, allow_missing: bool = False,
) -> Path:
    """Validate an unlinked WSL-ext4 directory without changing its identity.

    ``allow_missing`` is used for the result root before it is created.  In
    that case the closest existing parent determines the filesystem, and every
    existing path component is checked so a symlink cannot redirect the root
    across the declared sandbox boundary.
    """

    lexical = Path(os.path.abspath(os.fspath(path)))
    probe = lexical
    if lexical.exists() or lexical.is_symlink():
        if not lexical.is_dir():
            raise EvalManifestError(f"official {label} must be a directory: {lexical}")
    elif not allow_missing:
        raise EvalManifestError(f"official {label} must be an existing directory: {lexical}")
    else:
        while not probe.exists() and not probe.is_symlink():
            parent = probe.parent
            if parent == probe:
                raise EvalManifestError(
                    f"official {label} has no existing parent: {lexical}"
                )
            probe = parent
        if not probe.is_dir():
            raise EvalManifestError(
                f"official {label} parent must be a directory: {probe}"
            )

    current = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        current /= part
        if not current.exists() and not current.is_symlink():
            break
        is_junction = getattr(current, "is_junction", None)
        if current.is_symlink() or bool(callable(is_junction) and is_junction()):
            raise EvalManifestError(
                f"official {label} must not traverse a link or junction: {current}"
            )
    filesystem = _filesystem_type(probe)
    if filesystem != "ext4":
        raise EvalManifestError(
            f"official {label} must live on WSL ext4, not {filesystem}"
        )
    return lexical


def _resolve_executable(value: str, label: str) -> Path:
    argv = resolve_command_argv(value)
    if len(argv) != 1:
        raise EvalManifestError(f"official {label} must be one direct executable")
    return _require_regular_ext4_path(Path(argv[0]), label)


def official_process_environment() -> dict[str, str]:
    """Return the narrow, credential-free environment for official subprocesses."""

    present_auth = [name for name in _FORBIDDEN_AUTH_ENV if os.environ.get(name)]
    if present_auth:
        raise EvalManifestError(
            "official evaluation forbids auth environment variables; use CODEX_HOME login"
        )
    codex_home = os.environ.get("CODEX_HOME", "").strip()
    if not codex_home:
        raise EvalManifestError("official evaluation requires an explicit CODEX_HOME")
    allowed = (
        "HOME", "PATH", "CODEX_HOME", "LANG", "LC_ALL", "LC_CTYPE", "TERM",
        "TMPDIR", "TMP", "TEMP", "SSL_CERT_FILE", "SSL_CERT_DIR",
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
        "http_proxy", "https_proxy", "all_proxy", "no_proxy",
        "WSL_DISTRO_NAME", "WSL_INTEROP",
    )
    return {key: os.environ[key] for key in allowed if os.environ.get(key)}


def _version_output(command: list[str]) -> str:
    safe_environment = {
        "HOME": os.environ.get("HOME", "/tmp"),
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
    }
    if os.environ.get("CODEX_HOME"):
        safe_environment["CODEX_HOME"] = os.environ["CODEX_HOME"]
    completed = subprocess.run(
        command, capture_output=True, text=True, check=False, timeout=30,
        env=safe_environment,
    )
    if completed.returncode != 0:
        raise EvalManifestError(
            f"runtime identity command failed: {' '.join(command)}: {completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def _binary_identity(path: Path, version: str) -> dict[str, str]:
    return {
        "path": path.as_posix(),
        "filename": path.name,
        "version": version,
        "sha256": sha256_file(path),
    }


def _python_identity(value: str, label: str) -> dict[str, str]:
    executable = _resolve_executable(value, label)
    script = (
        "import json,platform,sys;"
        "print(json.dumps({'version':platform.python_version(),"
        "'implementation':platform.python_implementation(),"
        "'cache_tag':sys.implementation.cache_tag,"
        "'platform':platform.platform()} ,sort_keys=True))"
    )
    try:
        metadata = json.loads(_version_output([str(executable), "-I", "-c", script]))
    except json.JSONDecodeError as exc:
        raise EvalManifestError(f"official {label} returned invalid identity JSON") from exc
    if (not isinstance(metadata, dict)
            or any(not isinstance(metadata.get(key), str) or not metadata[key]
                   for key in ("version", "implementation", "cache_tag", "platform"))):
        raise EvalManifestError(f"official {label} identity is incomplete")
    return {
        "path": executable.as_posix(),
        "filename": executable.name,
        "sha256": sha256_file(executable),
        **{key: str(metadata[key]) for key in (
            "version", "implementation", "cache_tag", "platform",
        )},
    }


def _git_output(repository: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *args], capture_output=True,
        text=True, check=False, timeout=30,
        env={
            "HOME": os.environ.get("HOME", "/tmp"),
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "LANG": os.environ.get("LANG", "C.UTF-8"),
        },
    )
    if completed.returncode != 0:
        raise EvalManifestError(
            f"CodeGraph git identity failed: {' '.join(args)}: {completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def _assert_codegraph_build_matches_manifest(value: dict[str, Any]) -> None:
    expected_arm = load_manifest()["arms"][1]
    expected = expected_arm["build_identity"]
    comparisons = {
        "revision": value["repository"].get("revision"),
        "repository_tree": value["repository"].get("tree"),
        "package_lock_sha256": value["package_lock"].get("sha256"),
        "node_version": value["node"].get("version"),
        "node_sha256": value["node"].get("sha256"),
        "npm_version": value["npm"].get("version"),
        "npm_cli_sha256": value["npm"].get("sha256"),
        "entrypoint_sha256": value["entrypoint"].get("sha256"),
        "runtime_tree_algorithm": value.get("runtime_tree_algorithm"),
        "dist_tree_sha256": value["dist"].get("tree_sha256"),
        "dependency_tree_sha256": value["dependencies"].get("tree_sha256"),
        "reproduction_contract": value.get("reproduction_contract"),
    }
    required = {
        "revision": expected_arm["revision"],
        "repository_tree": expected["repository_tree"],
        "package_lock_sha256": expected["package_lock_sha256"],
        "node_version": expected["node"]["version"],
        "node_sha256": expected["node"]["sha256"],
        "npm_version": expected["npm"]["version"],
        "npm_cli_sha256": expected["npm"]["cli_sha256"],
        "entrypoint_sha256": expected_arm["entrypoint_sha256"],
        "runtime_tree_algorithm": expected["runtime_tree_algorithm"],
        "dist_tree_sha256": expected["dist_tree_sha256"],
        "dependency_tree_sha256": expected["dependency_tree_sha256"],
        "reproduction_contract": expected["reproduction_contract"],
    }
    if comparisons != required:
        changed = sorted(key for key in required if comparisons.get(key) != required[key])
        raise EvalManifestError(
            "CodeGraph build closure differs from the frozen manifest: "
            + ", ".join(changed)
        )


def _validate_codegraph_build_identity(value: Any) -> dict[str, Any]:
    if (not isinstance(value, dict)
            or value.get("schema_version") != CODEGRAPH_BUILD_IDENTITY_SCHEMA_VERSION):
        raise EvalManifestError("environment lock lacks the CodeGraph build closure")
    unhashed = {key: item for key, item in value.items() if key != "identity_sha256"}
    if value.get("identity_sha256") != sha256_bytes(canonical_json(unhashed)):
        raise EvalManifestError("CodeGraph build closure hash is invalid")
    if value.get("runtime_tree_algorithm") != RUNTIME_TREE_ALGORITHM:
        raise EvalManifestError("CodeGraph runtime-tree algorithm is not frozen")
    if value.get("reproduction_contract") != {
        "umask": "0022",
        "commands": [["npm", "ci"], ["npm", "run", "build"]],
    }:
        raise EvalManifestError("CodeGraph declared reproduction contract is invalid")
    repository = value.get("repository")
    if (not isinstance(repository, dict)
            or not isinstance(repository.get("path"), str)
            or not Path(repository["path"]).is_absolute()
            or re.fullmatch(r"[0-9a-f]{40}", str(repository.get("revision", ""))) is None
            or re.fullmatch(r"[0-9a-f]{40}", str(repository.get("tree", ""))) is None):
        raise EvalManifestError("CodeGraph repository identity is invalid")
    for key in ("package_lock", "entrypoint"):
        item = value.get(key)
        if (not isinstance(item, dict)
                or not isinstance(item.get("path"), str)
                or not Path(item["path"]).is_absolute()
                or item.get("filename") != Path(item["path"]).name
                or _SHA256_RE.fullmatch(str(item.get("sha256", ""))) is None):
            raise EvalManifestError(f"CodeGraph {key} identity is invalid")
    for key in ("node", "npm"):
        item = value.get(key)
        if (not isinstance(item, dict)
                or not isinstance(item.get("path"), str)
                or not Path(item["path"]).is_absolute()
                or item.get("filename") != Path(item["path"]).name
                or not isinstance(item.get("version"), str) or not item["version"]
                or _SHA256_RE.fullmatch(str(item.get("sha256", ""))) is None):
            raise EvalManifestError(f"CodeGraph {key} identity is invalid")
    for key in ("dist", "dependencies"):
        item = value.get(key)
        if (not isinstance(item, dict)
                or not isinstance(item.get("path"), str)
                or not Path(item["path"]).is_absolute()
                or item.get("algorithm") != RUNTIME_TREE_ALGORITHM
                or _SHA256_RE.fullmatch(str(item.get("tree_sha256", ""))) is None):
            raise EvalManifestError(f"CodeGraph {key} tree identity is invalid")
    repository_path = Path(repository["path"])
    required_paths = {
        "package_lock": repository_path / "package-lock.json",
        "entrypoint": repository_path / "dist" / "bin" / "codegraph.js",
        "dist": repository_path / "dist",
        "dependencies": repository_path / "node_modules",
    }
    if any(
        Path(value[key]["path"]) != expected
        for key, expected in required_paths.items()
    ):
        raise EvalManifestError("CodeGraph build closure paths do not match the repository")
    _assert_codegraph_build_matches_manifest(value)
    return value


def capture_codegraph_build_identity(
    *, repository: Path, runtime_root: Path, node: Path, npm_cli: Path,
    entrypoint: Path, full: bool = True,
) -> dict[str, Any]:
    """Capture the frozen CodeGraph source, build tools, and executable closure."""

    repository = _require_regular_ext4_path(
        repository, "CodeGraph repository", directory=True, reject_links=True,
    )
    runtime_root = _require_regular_ext4_path(
        runtime_root, "evaluation runtime root", directory=True, reject_links=True,
    )
    node = _require_regular_ext4_path(node, "Node executable", reject_links=True)
    npm_cli = _require_regular_ext4_path(npm_cli, "npm CLI", reject_links=True)
    entrypoint = _require_regular_ext4_path(
        entrypoint, "CodeGraph entrypoint", reject_links=True,
    )
    _assert_unlinked_within(Path(runtime_root.anchor), runtime_root)
    for path, label in (
        (repository, "CodeGraph repository"), (node, "Node executable"),
        (npm_cli, "npm CLI"), (entrypoint, "CodeGraph entrypoint"),
    ):
        if not _path_is_within(path, runtime_root):
            raise EvalManifestError(f"official {label} must be inside the runtime root")
        _assert_unlinked_within(runtime_root, path)
    expected_entrypoint = repository / "dist" / "bin" / "codegraph.js"
    if entrypoint != expected_entrypoint:
        raise EvalManifestError("CodeGraph entrypoint is not the frozen dist/bin path")
    package_lock = _require_regular_ext4_path(
        repository / "package-lock.json", "CodeGraph package lock", reject_links=True,
    )
    _assert_unlinked_within(runtime_root, package_lock)
    dist = _require_regular_ext4_path(
        repository / "dist", "CodeGraph dist tree", directory=True,
    )
    dependencies = _require_regular_ext4_path(
        repository / "node_modules", "CodeGraph dependency tree", directory=True,
    )
    revision = _git_output(repository, "rev-parse", "HEAD")
    tree = _git_output(repository, "rev-parse", "HEAD^{tree}")
    if _git_output(
        repository, "status", "--porcelain=v1", "--untracked-files=all",
    ):
        raise EvalManifestError("CodeGraph source checkout is dirty or has untracked files")
    ignored = _git_output(
        repository, "ls-files", "--others", "--ignored", "--exclude-standard",
    )
    for raw_relative in ignored.splitlines():
        relative = safe_relative_path(raw_relative)
        if relative.parts[0] not in {"dist", "node_modules"}:
            raise EvalManifestError(
                f"CodeGraph ignored file is outside the hashed build closure: {relative}"
            )
    node_version = _version_output([str(node), "--version"])
    npm_version = _version_output([str(node), str(npm_cli), "--version"])
    manifest_build = load_manifest()["arms"][1]["build_identity"]
    value: dict[str, Any] = {
        "schema_version": CODEGRAPH_BUILD_IDENTITY_SCHEMA_VERSION,
        "runtime_tree_algorithm": RUNTIME_TREE_ALGORITHM,
        "repository": {
            "path": repository.as_posix(), "revision": revision, "tree": tree,
        },
        "package_lock": {
            "path": package_lock.as_posix(), "filename": package_lock.name,
            "sha256": sha256_file(package_lock),
        },
        "node": _binary_identity(node, node_version),
        "npm": _binary_identity(npm_cli, npm_version),
        "entrypoint": {
            "path": entrypoint.as_posix(), "filename": entrypoint.name,
            "sha256": sha256_file(entrypoint),
        },
        "dist": {
            "path": dist.as_posix(), "algorithm": RUNTIME_TREE_ALGORITHM,
            "tree_sha256": (
                runtime_tree_identity(dist) if full
                else manifest_build["dist_tree_sha256"]
            ),
        },
        "dependencies": {
            "path": dependencies.as_posix(), "algorithm": RUNTIME_TREE_ALGORITHM,
            "tree_sha256": (
                runtime_tree_identity(dependencies) if full
                else manifest_build["dependency_tree_sha256"]
            ),
        },
        # This is a declared reproduction recipe.  The observed facts above
        # are the resulting source/tool/tree byte identities.
        "reproduction_contract": manifest_build["reproduction_contract"],
    }
    value["identity_sha256"] = sha256_bytes(canonical_json(value))
    return _validate_codegraph_build_identity(value)


def capture_official_runtime_identity(
    *, public_repository: Path, work_root: Path, codex_command: str,
    codegraph_command: str, codegraph_repository: Path, runtime_root: Path,
    npm_cli: Path, v02_python: str, v03_python: str,
) -> dict[str, Any]:
    """Capture the executable and isolation boundary used by an official run."""

    host = require_official_linux_wsl2()
    present_auth = [name for name in _FORBIDDEN_AUTH_ENV if os.environ.get(name)]
    if present_auth:
        raise EvalManifestError(
            "official evaluation forbids auth environment variables; use a dedicated CODEX_HOME"
        )
    raw_codex_home = os.environ.get("CODEX_HOME", "").strip()
    if not raw_codex_home:
        raise EvalManifestError(
            "official evaluation requires an explicit dedicated CODEX_HOME"
        )
    codex_home = _require_regular_ext4_path(
        Path(raw_codex_home), "CODEX_HOME", directory=True, reject_links=True,
    )
    public_repository = public_repository.resolve()
    work_root = _require_regular_ext4_path(
        work_root, "work root", directory=True, reject_links=True,
    )
    runtime_root = _require_regular_ext4_path(
        runtime_root, "evaluation runtime root", directory=True, reject_links=True,
    )
    if _path_is_within(codex_home, work_root) or _path_is_within(work_root, codex_home):
        raise EvalManifestError("CODEX_HOME and work root must be disjoint")
    isolated_roots = {
        "public repository": public_repository,
        "work root": work_root,
        "runtime root": runtime_root,
        "CODEX_HOME": codex_home,
    }
    for left_name, left in isolated_roots.items():
        for right_name, right in isolated_roots.items():
            if left_name < right_name and (
                _path_is_within(left, right) or _path_is_within(right, left)
            ):
                raise EvalManifestError(
                    f"official {left_name} and {right_name} must be disjoint"
                )

    codex = _resolve_executable(codex_command, "Codex CLI")
    with codex.open("rb") as codex_handle:
        codex_magic = codex_handle.read(4)
    if codex_magic != b"\x7fELF":
        raise EvalManifestError("official Codex CLI must be a direct Linux ELF binary")
    codex_version = _version_output([str(codex), "--version"])
    if codex_version != OFFICIAL_CODEX_VERSION:
        raise EvalManifestError(
            f"official evaluation requires {OFFICIAL_CODEX_VERSION}, got {codex_version!r}"
        )

    codegraph_parts = shlex.split(codegraph_command)
    if (len(codegraph_parts) != 2
            or Path(codegraph_parts[0]).name.casefold() not in {"node", "node.exe"}):
        raise EvalManifestError(
            "official CodeGraph command must be exactly node plus its frozen JS entrypoint"
        )
    node = _resolve_executable(codegraph_parts[0], "Node executable")
    node_version = _version_output([str(node), "--version"])
    match = re.fullmatch(r"v(\d+)\.\d+\.\d+", node_version)
    if match is None or not (20 <= int(match.group(1)) < 25):
        raise EvalManifestError("official CodeGraph runtime requires Node >=20 and <25")
    entrypoint = Path(codegraph_parts[1])
    if not entrypoint.is_absolute():
        entrypoint = Path.cwd() / entrypoint
    entrypoint = _require_regular_ext4_path(entrypoint, "CodeGraph entrypoint")
    codegraph_build = capture_codegraph_build_identity(
        repository=codegraph_repository, runtime_root=runtime_root,
        node=node, npm_cli=npm_cli, entrypoint=entrypoint, full=True,
    )

    helper_path = shutil.which("codex-linux-sandbox")
    bwrap_path = shutil.which("bwrap")
    if not helper_path or not bwrap_path:
        raise EvalManifestError("official Codex Linux sandbox helper and bwrap are required")
    helper = _require_regular_ext4_path(Path(helper_path), "Codex Linux sandbox helper")
    bwrap = _require_regular_ext4_path(Path(bwrap_path), "bubblewrap executable")

    drvfs_roots = discover_windows_mount_roots()
    if not drvfs_roots:
        raise EvalManifestError("official WSL evaluation found no Windows drvfs mounts to deny")
    home_roots = _sensitive_home_roots()
    if any(_path_is_within(work_root, item) for item in home_roots):
        raise EvalManifestError(
            "official work root must be outside every denied user-home root"
        )
    external_canary_root = work_root.parent / ".hlsgraph-eval-private-canary"
    if external_canary_root.exists() or external_canary_root.is_symlink():
        raise EvalManifestError(
            f"official external canary root must not pre-exist: {external_canary_root}"
        )
    deny_roots = {
        public_repository, work_root, runtime_root, codex_home, external_canary_root,
        *drvfs_roots, *home_roots,
    }
    corpus_ids = [item["id"] for item in load_corpus_lock()["corpora"]]
    deny_catalog: dict[str, Any] = {
        "arm_roots": {
            arm: (work_root / arm).as_posix() for arm in ARM_IDS
        },
        "workspace_roots": {
            f"{arm}/{corpus_id}": (work_root / arm / corpus_id).as_posix()
            for arm in ARM_IDS for corpus_id in corpus_ids
        },
        "control_roots": [
            (work_root / ".hlsgraph-eval-boundary").as_posix(),
            (work_root / "_cache").as_posix(),
        ],
    }
    boundary: dict[str, Any] = {
        "schema_version": SANDBOX_BOUNDARY_SCHEMA_VERSION,
        "public_repository": public_repository.as_posix(),
        "work_root": work_root.as_posix(),
        "runtime_root": runtime_root.as_posix(),
        "work_root_policy": "explicit_sibling_directory_deny_v1",
        "work_root_deny_catalog": deny_catalog,
        "work_root_deny_catalog_sha256": sha256_bytes(canonical_json(deny_catalog)),
        "codex_home": codex_home.as_posix(),
        "home_canary_root": Path.home().resolve().as_posix(),
        "external_canary_root": external_canary_root.as_posix(),
        "drvfs_roots": [item.as_posix() for item in drvfs_roots],
        "home_roots": [item.as_posix() for item in home_roots],
        "deny_roots": sorted(item.as_posix() for item in deny_roots),
    }
    boundary["identity_sha256"] = sha256_bytes(canonical_json(boundary))
    runtime: dict[str, Any] = {
        "schema_version": RUNTIME_IDENTITY_SCHEMA_VERSION,
        "host": host,
        "sandbox_boundary": boundary,
        "codex": _binary_identity(codex, codex_version),
        "codex_linux_sandbox": _binary_identity(helper, OFFICIAL_CODEX_VERSION),
        "bubblewrap": _binary_identity(
            bwrap, _version_output([str(bwrap), "--version"]),
        ),
        "node": _binary_identity(node, node_version),
        "codegraph_entrypoint": {
            "path": entrypoint.as_posix(),
            "filename": entrypoint.name,
            "sha256": sha256_file(entrypoint),
        },
        "codegraph_build": codegraph_build,
        "python": {
            "harness": _python_identity(sys.executable, "harness Python"),
            "hlsgraph_v02": _python_identity(v02_python, "v0.2 Python"),
            "hlsgraph_v03": _python_identity(v03_python, "v0.3 Python"),
        },
    }
    runtime["identity_sha256"] = sha256_bytes(canonical_json(runtime))
    return runtime


def _validate_runtime_identity(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema_version") != RUNTIME_IDENTITY_SCHEMA_VERSION:
        raise EvalManifestError("environment lock lacks the official runtime identity")
    unhashed = {key: item for key, item in value.items() if key != "identity_sha256"}
    if value.get("identity_sha256") != sha256_bytes(canonical_json(unhashed)):
        raise EvalManifestError("environment runtime identity hash is invalid")
    host = value.get("host")
    if (not isinstance(host, dict) or host.get("os_name") != "posix"
            or host.get("system") != "Linux" or host.get("wsl") != "2"
            or host.get("distribution_id") != "ubuntu"
            or host.get("distribution_version") != "22.04"
            or "microsoft-standard-wsl2" not in str(host.get("release", "")).casefold()):
        raise EvalManifestError("environment lock was not prepared on Ubuntu-22.04 WSL2")
    boundary = value.get("sandbox_boundary")
    if not isinstance(boundary, dict) or boundary.get("schema_version") != SANDBOX_BOUNDARY_SCHEMA_VERSION:
        raise EvalManifestError("environment lock lacks the sandbox boundary")
    boundary_unhashed = {
        key: item for key, item in boundary.items() if key != "identity_sha256"
    }
    if boundary.get("identity_sha256") != sha256_bytes(canonical_json(boundary_unhashed)):
        raise EvalManifestError("environment sandbox boundary hash is invalid")
    for key in (
        "public_repository", "work_root", "runtime_root", "codex_home", "home_canary_root",
        "external_canary_root",
    ):
        if not isinstance(boundary.get(key), str) or not Path(boundary[key]).is_absolute():
            raise EvalManifestError(f"environment sandbox boundary lacks {key}")
    isolated_boundary_roots = [
        Path(boundary[key]) for key in (
            "public_repository", "work_root", "runtime_root", "codex_home",
        )
    ]
    if any(
        _path_is_within(left, right) or _path_is_within(right, left)
        for index, left in enumerate(isolated_boundary_roots)
        for right in isolated_boundary_roots[index + 1:]
    ):
        raise EvalManifestError("environment runtime, work, checkout, and auth roots overlap")
    if boundary.get("work_root_policy") != "explicit_sibling_directory_deny_v1":
        raise EvalManifestError("environment sandbox boundary has the wrong work-root policy")
    catalog = boundary.get("work_root_deny_catalog")
    expected_workspace_roots = {
        f"{arm}/{corpus['id']}" for arm in ARM_IDS
        for corpus in load_corpus_lock()["corpora"]
    }
    if (not isinstance(catalog, dict)
            or set(catalog) != {"arm_roots", "workspace_roots", "control_roots"}
            or not isinstance(catalog["arm_roots"], dict)
            or set(catalog["arm_roots"]) != set(ARM_IDS)
            or not isinstance(catalog["workspace_roots"], dict)
            or set(catalog["workspace_roots"]) != expected_workspace_roots
            or not isinstance(catalog["control_roots"], list)
            or len(catalog["control_roots"]) != 2
            or any(not isinstance(path, str) or not Path(path).is_absolute()
                   for group in (
                       catalog["arm_roots"].values(),
                       catalog["workspace_roots"].values(),
                       catalog["control_roots"],
                   ) for path in group)
            or boundary.get("work_root_deny_catalog_sha256")
            != sha256_bytes(canonical_json(catalog))):
        raise EvalManifestError("environment sandbox boundary has an invalid deny catalog")
    for key in ("drvfs_roots", "home_roots", "deny_roots"):
        paths = boundary.get(key)
        if (not isinstance(paths, list) or not paths
                or any(not isinstance(item, str) or not Path(item).is_absolute()
                       for item in paths)
                or len(paths) != len(set(paths))):
            raise EvalManifestError(f"environment sandbox boundary has invalid {key}")
    required_denies = {
        boundary["public_repository"], boundary["work_root"], boundary["runtime_root"],
        boundary["codex_home"],
        boundary["external_canary_root"],
        *boundary["drvfs_roots"], *boundary["home_roots"],
    }
    if not required_denies.issubset(set(boundary["deny_roots"])):
        raise EvalManifestError("environment sandbox boundary omits a required deny root")
    home_canary = Path(boundary["home_canary_root"])
    if not any(
        home_canary == Path(item) or _path_is_within(home_canary, Path(item))
        for item in boundary["home_roots"]
    ):
        raise EvalManifestError("environment home canary is outside the denied home roots")
    for key in ("codex", "codex_linux_sandbox", "bubblewrap", "node"):
        identity = value.get(key)
        if (not isinstance(identity, dict)
                or not isinstance(identity.get("path"), str)
                or not Path(identity["path"]).is_absolute()
                or identity.get("filename") != Path(identity["path"]).name
                or not isinstance(identity.get("version"), str) or not identity["version"]
                or _SHA256_RE.fullmatch(str(identity.get("sha256", ""))) is None):
            raise EvalManifestError(f"environment lock has invalid {key} identity")
    if value["codex"].get("version") != OFFICIAL_CODEX_VERSION:
        raise EvalManifestError("environment lock has the wrong Codex version")
    python = value.get("python")
    if not isinstance(python, dict) or set(python) != {
        "harness", "hlsgraph_v02", "hlsgraph_v03",
    }:
        raise EvalManifestError("environment lock has incomplete Python identities")
    for key, identity in python.items():
        if (not isinstance(identity, dict)
                or not isinstance(identity.get("path"), str)
                or not Path(identity["path"]).is_absolute()
                or identity.get("filename") != Path(identity["path"]).name
                or _SHA256_RE.fullmatch(str(identity.get("sha256", ""))) is None
                or any(not isinstance(identity.get(field), str) or not identity[field]
                       for field in ("version", "implementation", "cache_tag", "platform"))):
            raise EvalManifestError(f"environment lock has invalid {key} Python identity")
    entrypoint = value.get("codegraph_entrypoint")
    if (not isinstance(entrypoint, dict)
            or not isinstance(entrypoint.get("path"), str)
            or not Path(entrypoint["path"]).is_absolute()
            or entrypoint.get("filename") != Path(entrypoint["path"]).name
            or _SHA256_RE.fullmatch(str(entrypoint.get("sha256", ""))) is None):
        raise EvalManifestError("environment lock has invalid CodeGraph entrypoint identity")
    codegraph_build = _validate_codegraph_build_identity(value.get("codegraph_build"))
    runtime_root = Path(boundary["runtime_root"])
    build_paths = (
        codegraph_build["repository"]["path"],
        codegraph_build["package_lock"]["path"],
        codegraph_build["node"]["path"],
        codegraph_build["npm"]["path"],
        codegraph_build["entrypoint"]["path"],
        codegraph_build["dist"]["path"],
        codegraph_build["dependencies"]["path"],
    )
    if (codegraph_build["node"] != value["node"]
            or codegraph_build["entrypoint"] != entrypoint
            or any(not _path_is_within(Path(path), runtime_root) for path in build_paths)):
        raise EvalManifestError("CodeGraph build closure is inconsistent with runtime identity")
    return value


def verify_official_runtime_identity(
    environment: dict[str, Any], *, public_repository: Path, work_root: Path,
    codex_command: str, codegraph_command: str, v02_python: str, v03_python: str,
) -> dict[str, Any]:
    expected = _validate_runtime_identity(environment.get("runtime_identity"))
    expected_build = expected["codegraph_build"]
    current = capture_official_runtime_identity(
        public_repository=public_repository, work_root=work_root,
        codex_command=codex_command, codegraph_command=codegraph_command,
        codegraph_repository=Path(expected_build["repository"]["path"]),
        runtime_root=Path(expected["sandbox_boundary"]["runtime_root"]),
        npm_cli=Path(expected_build["npm"]["path"]),
        v02_python=v02_python, v03_python=v03_python,
    )
    if current != expected:
        raise EvalManifestError("current runtime differs from environment.lock.json")
    return current


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvalManifestError(f"cannot read JSON asset {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvalManifestError(f"expected JSON object in {path}")
    return value


def load_manifest(path: Path = SUITE_PATH) -> dict[str, Any]:
    manifest = _load_json(path)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise EvalManifestError("unsupported suite manifest schema")
    if manifest.get("model") != {
        "id": "gpt-5.6-sol", "reasoning_effort": "medium"
    }:
        raise EvalManifestError("the public suite must use gpt-5.6-sol / medium")
    if manifest.get("repetitions") != 4:
        raise EvalManifestError("the public suite must use four repetitions")
    if manifest.get("codex_cli", {}).get("timeout_seconds") != 900:
        raise EvalManifestError("the public suite must use the frozen 900-second timeout")
    arms = manifest.get("arms")
    if not isinstance(arms, list) or tuple(item.get("id") for item in arms) != ARM_IDS:
        raise EvalManifestError(f"arms must be fixed in order: {ARM_IDS!r}")
    codegraph = arms[1]
    expected_revision = "286e9ccc2dad45336d4fd67052930322054d64b5"
    if codegraph.get("revision") != expected_revision:
        raise EvalManifestError("CodeGraph comparison revision is not frozen")
    entrypoint_sha256 = codegraph.get("entrypoint_sha256")
    if (not isinstance(entrypoint_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", entrypoint_sha256) is None):
        raise EvalManifestError("CodeGraph comparison entrypoint hash is not frozen")
    expected_build = {
        "schema_version": CODEGRAPH_BUILD_IDENTITY_SCHEMA_VERSION,
        "runtime_tree_algorithm": RUNTIME_TREE_ALGORITHM,
        "repository_tree": "a3536fb69a45c715e2d245e3c9aea80dee187720",
        "package_lock_sha256": "c50188fcf83f951bf1197aa166279f5f6e2d39d1a12410231f116df0e3b3b5e8",
        "node": {
            "version": "v22.17.0",
            "sha256": "8071ae0fca095a272ad698a90c7061801a86fb6392ddb81e922b68a91a4374b9",
        },
        "npm": {
            "version": "10.9.2",
            "cli_sha256": "8e5f6f3429f8cdbe693cdc29904e9d5a7b127a494bd15c804bd54c7403bfcbe7",
        },
        "reproduction_contract": {
            "umask": "0022",
            "commands": [["npm", "ci"], ["npm", "run", "build"]],
        },
        "dist_tree_sha256": "cc0cefe48514fa34a8c3b488efb4377bec2f62ad84e32c57f495e2cd2cb2e61b",
        "dependency_tree_sha256": "20088cced4df7332c2787bf7d281e301a67d8fd831dad53a564a8d50d723a284",
    }
    if codegraph.get("build_identity") != expected_build:
        raise EvalManifestError("CodeGraph comparison build closure is not frozen")
    if entrypoint_sha256 != "03e4c791cc0dd91ed264278461bf9a56c0278aa0670d5942fc4732311c66de03":
        raise EvalManifestError("CodeGraph comparison entrypoint bytes drifted")
    if arms[2].get("revision") != "7b26bfb07aa7c4d1a4705d3076cac684c3561e6f":
        raise EvalManifestError("HLSGraph v0.2 comparison revision is not frozen")
    contracts = manifest.get("claim_contracts")
    required_contracts = {
        "design_fact", "synthetic_observation", "knowledge_guidance", "unknown",
    }
    if not isinstance(contracts, dict) or not required_contracts.issubset(contracts):
        raise EvalManifestError("suite manifest lacks frozen truth-plane claim contracts")
    for plane in required_contracts:
        contract = contracts[plane]
        if (not isinstance(contract, dict)
                or not isinstance(contract.get("authority_pattern"), str)
                or not isinstance(contract.get("allowed_stages"), list)):
            raise EvalManifestError(f"invalid claim contract for {plane}")
    return manifest


def load_corpus_lock(path: Path = CORPUS_LOCK_PATH) -> dict[str, Any]:
    lock = _load_json(path)
    if lock.get("schema_version") != CORPUS_SCHEMA_VERSION:
        raise EvalManifestError("unsupported corpus lock schema")
    corpora = lock.get("corpora")
    if not isinstance(corpora, list) or len(corpora) != 4:
        raise EvalManifestError("the public suite requires exactly four corpora")
    ids: set[str] = set()
    for corpus in corpora:
        corpus_id = corpus.get("id")
        if not isinstance(corpus_id, str) or not corpus_id or corpus_id in ids:
            raise EvalManifestError("corpus IDs must be unique non-empty strings")
        ids.add(corpus_id)
        if corpus.get("license") != "Apache-2.0":
            raise EvalManifestError(f"{corpus_id}: only Apache-2.0 corpus assets are allowed")
        revision = corpus.get("revision")
        if revision is not None and (not isinstance(revision, str) or len(revision) != 40):
            raise EvalManifestError(f"{corpus_id}: revision must be a full Git SHA")
        files = corpus.get("files")
        if not isinstance(files, list) or not files:
            raise EvalManifestError(f"{corpus_id}: files must be non-empty")
        for entry in files:
            digest = entry.get("sha256")
            if not isinstance(digest, str) or len(digest) != 64:
                raise EvalManifestError(f"{corpus_id}: every file requires SHA-256")
            if "url" in entry and not str(entry["url"]).startswith(
                "https://raw.githubusercontent.com/"
            ):
                raise EvalManifestError(f"{corpus_id}: remote files must use pinned GitHub raw URLs")
    return lock


def _iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise EvalManifestError(f"cannot read JSONL asset {path}: {exc}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvalManifestError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise EvalManifestError(f"expected object at {path}:{line_number}")
        yield line_number, value


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    seen: set[str] = set()
    per_corpus: dict[str, int] = {}
    for line_number, question in _iter_jsonl(path):
        if question.get("schema_version") != QUESTION_SCHEMA_VERSION:
            raise EvalManifestError(f"unsupported question schema at line {line_number}")
        question_id = question.get("id")
        corpus_id = question.get("corpus_id")
        if not isinstance(question_id, str) or question_id in seen:
            raise EvalManifestError(f"duplicate or invalid question ID at line {line_number}")
        if not isinstance(corpus_id, str):
            raise EvalManifestError(f"invalid corpus ID at line {line_number}")
        if not question.get("prompt") or not question.get("criteria"):
            raise EvalManifestError(f"question {question_id} lacks prompt or criteria")
        seen.add(question_id)
        per_corpus[corpus_id] = per_corpus.get(corpus_id, 0) + 1
        questions.append(question)
    if len(questions) != 12:
        raise EvalManifestError("the frozen public suite requires exactly 12 questions")
    if set(per_corpus.values()) != {3} or len(per_corpus) != 4:
        raise EvalManifestError("each corpus must contribute exactly three questions")
    return questions


def load_static_cases(path: Path = STATIC_CASES_PATH) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, case in _iter_jsonl(path):
        if case.get("schema_version") != STATIC_CASE_SCHEMA_VERSION:
            raise EvalManifestError(f"unsupported static case schema at line {line_number}")
        identifier = case.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise EvalManifestError(f"duplicate or invalid static case ID at line {line_number}")
        if (not isinstance(case.get("query"), str) or not case["query"].strip()
                or case.get("view") not in {"architecture", "evidence"}
                or case.get("result_section") not in {"facts", "guidance"}
                or not isinstance(case.get("gold"), list) or not case["gold"]):
            raise EvalManifestError(f"static case {identifier} has an invalid retrieval contract")
        gold_ids = [item.get("id") for item in case["gold"]]
        if any(not isinstance(item, str) or not item for item in gold_ids):
            raise EvalManifestError(f"static case {identifier} has invalid gold IDs")
        if len(gold_ids) != len(set(gold_ids)):
            raise EvalManifestError(f"static case {identifier} has duplicate gold IDs")
        seen.add(identifier)
        cases.append(case)
    if len(cases) != 12:
        raise EvalManifestError("the frozen public suite requires exactly 12 static cases")
    return cases


def asset_digest() -> str:
    """Hash every frozen suite input without depending on file timestamps."""
    digest = hashlib.sha256()
    for path in (
        SUITE_PATH, CORPUS_LOCK_PATH, QUESTIONS_PATH, STATIC_CASES_PATH,
        ANSWER_SCHEMA_PATH,
    ):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def harness_digest() -> str:
    """Hash every checked-in byte that executes or scores the public suite."""
    digest = hashlib.sha256()
    excluded = {"work", "runs", "results", "__pycache__"}
    for path in sorted(item for item in HERE.rglob("*") if item.is_file()):
        relative = path.relative_to(HERE)
        if any(part in excluded for part in relative.parts):
            continue
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
        digest.update(b"\0")
    return digest.hexdigest()


def _linked(path: Path) -> bool:
    junction = getattr(path, "is_junction", None)
    if path.is_symlink() or bool(junction and junction()):
        return True
    try:
        attributes = path.lstat().st_file_attributes
    except (AttributeError, FileNotFoundError, OSError):
        return False
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def _absolute(path: Path) -> Path:
    """Return an absolute lexical path without following a link or junction."""
    return Path(os.path.abspath(os.fspath(path)))


def _assert_unlinked_within(root: Path, path: Path) -> None:
    """Reject a link/reparse component between ``root`` and ``path``.

    Calling ``resolve()`` first would erase the evidence that an intermediate
    directory was a symlink or Windows junction.  Evaluation identities must
    therefore inspect every lexical component before opening a byte.
    """
    root = _absolute(root)
    path = _absolute(path)
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise EvalManifestError(f"evaluation path escapes its root: {path}") from exc
    current = root
    if _linked(current):
        raise EvalManifestError(f"evaluation root is linked: {current}")
    for part in relative.parts:
        current = current / part
        if _linked(current):
            raise EvalManifestError(f"evaluation path contains a link: {current}")


def _stable_file_bytes(path: Path, *, max_bytes: int = 512 * 1024 * 1024) -> bytes:
    if _linked(path) or not path.is_file():
        raise EvalManifestError(f"evaluation file is missing or linked: {path}")
    before = path.stat()
    if before.st_size > max_bytes:
        raise EvalManifestError(f"evaluation file exceeds the identity limit: {path}")
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        data = stream.read(max_bytes + 1)
        closed = os.fstat(stream.fileno())
    after = path.stat()
    identity = lambda info: (
        info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns,
    )
    if (len(data) > max_bytes or identity(before) != identity(opened)
            or identity(opened) != identity(closed)
            or identity(closed) != identity(after)):
        raise EvalManifestError(f"evaluation file changed while hashed: {path}")
    return data


def _tree_identity(root: Path, *, excluded: frozenset[str] = frozenset()) -> str:
    root = _absolute(root)
    if _linked(root) or not root.is_dir():
        raise EvalManifestError(f"evaluation index is missing or linked: {root}")
    digest = hashlib.sha256()
    records: list[tuple[str, Path]] = []
    for directory, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        directory_path = Path(directory)
        for name in sorted(dirnames):
            child = directory_path / name
            relative = child.relative_to(root).as_posix()
            if _linked(child):
                raise EvalManifestError(f"evaluation tree contains a linked directory: {relative}")
        for name in sorted(filenames):
            child = directory_path / name
            relative = child.relative_to(root).as_posix()
            if _linked(child):
                raise EvalManifestError(f"evaluation tree contains a linked file: {relative}")
            if relative not in excluded:
                records.append((relative, child))
    for relative, path in sorted(records):
        data = _stable_file_bytes(path)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(data).digest())
        digest.update(b"\0")
    return digest.hexdigest()


def cold_start_input_identity(work_root: Path, arm: str, corpus_id: str) -> dict[str, str]:
    """Hash the immutable workspace input while excluding only its index root.

    The digest is deliberately recomputable after indexing.  It therefore
    binds a pre-index absence observation to the same source/config bytes that
    appear in the final prepared-workspace identity.
    """
    if arm not in ARM_IDS:
        raise EvalManifestError(f"unknown evaluation arm: {arm}")
    if not any(item["id"] == corpus_id for item in load_corpus_lock()["corpora"]):
        raise EvalManifestError(f"unknown evaluation corpus: {corpus_id}")
    work_root = _absolute(work_root)
    workspace = _absolute(work_root / arm / corpus_id)
    _assert_unlinked_within(work_root, workspace)
    if _linked(workspace) or not workspace.is_dir():
        raise EvalManifestError(f"evaluation workspace is missing or linked: {workspace}")
    index_name = (
        ".codegraph" if arm == "codegraph"
        else ".hlsgraph" if arm.startswith("hlsgraph-")
        else ""
    )
    digest = hashlib.sha256()
    records: list[tuple[str, Path]] = []
    for directory, dirnames, filenames in os.walk(workspace, topdown=True, followlinks=False):
        directory_path = Path(directory)
        relative_directory = directory_path.relative_to(workspace)
        if relative_directory == Path(".") and index_name:
            dirnames[:] = [name for name in dirnames if name != index_name]
        for name in sorted(dirnames):
            child = directory_path / name
            relative = child.relative_to(workspace).as_posix()
            if _linked(child):
                raise EvalManifestError(
                    f"cold-start input contains a linked directory: {relative}"
                )
        for name in sorted(filenames):
            child = directory_path / name
            relative = child.relative_to(workspace).as_posix()
            if index_name and (relative == index_name or relative.startswith(index_name + "/")):
                continue
            if _linked(child):
                raise EvalManifestError(
                    f"cold-start input contains a linked file: {relative}"
                )
            records.append((relative, child))
    for relative, path in sorted(records):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(_stable_file_bytes(path)).digest())
        digest.update(b"\0")
    return {
        "index_relative_path": index_name,
        "input_tree_sha256": digest.hexdigest(),
    }


def _hlsgraph_index_metadata(database_path: Path) -> dict[str, Any]:
    data = _stable_file_bytes(database_path)
    connection = sqlite3.connect(":memory:")
    try:
        deserialize = getattr(connection, "deserialize", None)
        if deserialize is None:
            raise EvalManifestError("SQLite deserialize is required for index identity")
        deserialize(data)
        connection.execute("PRAGMA query_only=ON")
        rows = connection.execute(
            "SELECT project_id,active_snapshot_id FROM project_state ORDER BY project_id"
        ).fetchall()
        if len(rows) != 1 or not rows[0][1]:
            raise EvalManifestError("HLSGraph evaluation index lacks one active snapshot")
        snapshot_id = str(rows[0][1])
        graph_view = connection.execute(
            "SELECT schema_version,metadata_json FROM graph_views WHERE snapshot_id=?",
            (snapshot_id,),
        ).fetchone()
        if graph_view is None:
            raise EvalManifestError("HLSGraph evaluation index lacks its graph view")
        try:
            packs = connection.execute(
                "SELECT pack_id,content_hash FROM knowledge_packs ORDER BY pack_id"
            ).fetchall()
        except sqlite3.OperationalError:
            packs = []
        return {
            "project_id": str(rows[0][0]),
            "snapshot_id": snapshot_id,
            "graph_schema_version": str(graph_view[0]),
            "graph_view_sha256": sha256_bytes(str(graph_view[1]).encode("utf-8")),
            "knowledge_catalog_sha256": sha256_bytes(canonical_json(packs)),
            "database_sha256": sha256_bytes(data),
        }
    finally:
        connection.close()


def workspace_identity(work_root: Path, arm: str, corpus_id: str) -> dict[str, Any]:
    """Bind one prepared workspace's frozen corpus and read-only index bytes."""
    if arm not in ARM_IDS:
        raise EvalManifestError(f"unknown evaluation arm: {arm}")
    corpus = next(
        (item for item in load_corpus_lock()["corpora"] if item["id"] == corpus_id),
        None,
    )
    if corpus is None:
        raise EvalManifestError(f"unknown evaluation corpus: {corpus_id}")
    work_root = _absolute(work_root)
    workspace = _absolute(work_root / arm / corpus_id)
    _assert_unlinked_within(work_root, workspace)
    if _linked(workspace) or not workspace.is_dir():
        raise EvalManifestError(f"evaluation workspace is missing or linked: {workspace}")
    from .setup_corpus import _provenance
    provenance_path = workspace / "EVAL_PROVENANCE.json"
    _assert_unlinked_within(workspace, provenance_path)
    try:
        provenance = json.loads(_stable_file_bytes(provenance_path).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvalManifestError("evaluation provenance is malformed") from exc
    if provenance != _provenance(corpus):
        raise EvalManifestError("evaluation provenance differs from the frozen corpus")
    corpus_files: list[dict[str, str]] = []
    for entry in corpus["files"]:
        relative = safe_relative_path(entry["destination"])
        path = workspace / relative
        _assert_unlinked_within(workspace, path)
        digest = sha256_bytes(_stable_file_bytes(path))
        if digest != entry["sha256"]:
            raise EvalManifestError(f"frozen corpus byte mismatch: {relative.as_posix()}")
        corpus_files.append({"path": relative.as_posix(), "sha256": digest})
    result: dict[str, Any] = {
        "schema_version": "hlsgraph.agent_eval.workspace_identity.v1",
        "arm": arm, "corpus_id": corpus_id,
        "corpus_sha256": sha256_bytes(canonical_json({
            "files": corpus_files,
            "provenance_sha256": sha256_bytes(_stable_file_bytes(provenance_path)),
        })),
        "index_kind": "none",
        "index_sha256": sha256_bytes(b"no-index"),
        "workspace_tree_sha256": _tree_identity(
            workspace,
            excluded=frozenset({".hlsgraph/private/retrieval-access.jsonl"}),
        ),
        "cold_start_input_sha256": cold_start_input_identity(
            work_root, arm, corpus_id,
        )["input_tree_sha256"],
    }
    if arm == "codegraph":
        result.update({
            "index_kind": "codegraph.286e9ccc",
            "index_sha256": _tree_identity(workspace / ".codegraph"),
        })
    elif arm.startswith("hlsgraph-"):
        index_root = workspace / ".hlsgraph"
        result.update({
            "index_kind": arm,
            "index_sha256": _tree_identity(
                index_root,
                excluded=frozenset({"private/retrieval-access.jsonl"}),
            ),
            **_hlsgraph_index_metadata(index_root / "graph.db"),
        })
    result["workspace_identity_sha256"] = sha256_bytes(canonical_json(result))
    return result


def verify_prepared_workspace(
    environment: dict[str, Any], work_root: Path, arm: str, corpus_id: str,
) -> dict[str, Any]:
    expected = environment.get("workspaces", {}).get(f"{arm}/{corpus_id}")
    current = workspace_identity(work_root, arm, corpus_id)
    if expected != current:
        raise EvalManifestError(f"prepared workspace identity changed: {arm}/{corpus_id}")
    return current


def verify_evaluation_checkout(environment: dict[str, Any], root: Path | None = None) -> None:
    root = _absolute(root or HERE.parents[1])
    if _linked(root) or not root.is_dir():
        raise EvalManifestError("evaluation checkout root is missing or linked")
    top_level = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], cwd=root,
        capture_output=True, text=True, check=False, timeout=20,
    )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True,
        check=False, timeout=20,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=root,
        capture_output=True, text=True, check=False, timeout=20,
    )
    tracked = subprocess.run(
        ["git", "ls-files", "--full-name", "-s", "-z"], cwd=root,
        capture_output=True, check=False, timeout=20,
    )
    try:
        checkout_is_root = (
            top_level.returncode == 0
            and Path(top_level.stdout.strip()).resolve() == root.resolve()
        )
    except OSError:
        checkout_is_root = False
    if (not checkout_is_root or revision.returncode != 0 or status.returncode != 0
            or tracked.returncode != 0
            or revision.stdout.strip() != environment["hlsgraph_v03"]["revision"]):
        raise EvalManifestError("evaluation checkout is dirty or differs from the candidate revision")
    for record in tracked.stdout.split(b"\0"):
        if not record:
            continue
        try:
            header, raw_path = record.split(b"\t", 1)
            mode = header.split(b" ", 1)[0]
            relative = safe_relative_path(raw_path.decode("utf-8", errors="strict"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise EvalManifestError("evaluation checkout has an invalid tracked path") from exc
        if mode in {b"120000", b"160000"}:
            raise EvalManifestError("evaluation checkout contains a symlink or submodule")
        path = root / relative
        _assert_unlinked_within(root, path)
        if not path.is_file():
            raise EvalManifestError(f"evaluation checkout tracked file is missing: {relative}")
    if status.stdout.strip():
        raise EvalManifestError("evaluation checkout is dirty or differs from the candidate revision")


def load_environment_lock(path: Path) -> dict[str, Any]:
    """Load and validate the prepared, official evaluation environment.

    The lock is an external root of trust for collection and scoring.  Results
    may repeat its digest, but they cannot replace validation of the actual
    lock file used to build the indexes.
    """
    environment = _load_json(path)
    manifest = load_manifest()
    if environment.get("schema_version") != ENVIRONMENT_SCHEMA_VERSION:
        raise EvalManifestError("unsupported evaluation environment schema")
    if environment.get("suite_asset_sha256") != asset_digest():
        raise EvalManifestError("environment lock belongs to different suite assets")
    if environment.get("evaluation_harness_sha256") != harness_digest():
        raise EvalManifestError("environment lock belongs to different evaluation code")
    if environment.get("official_profile") is not True:
        raise EvalManifestError("environment lock is not an official evaluation profile")
    if environment.get("source_backend") != "libclang":
        raise EvalManifestError("official evaluation environment must use libclang")
    runtime_identity = _validate_runtime_identity(environment.get("runtime_identity"))
    codegraph = environment.get("codegraph_entrypoint")
    codegraph_build = environment.get("codegraph_build")
    expected_codegraph = manifest["arms"][1]
    if (environment.get("codegraph_revision") != expected_codegraph["revision"]
            or not isinstance(codegraph, dict)
            or codegraph != runtime_identity["codegraph_entrypoint"]
            or codegraph.get("sha256") != expected_codegraph["entrypoint_sha256"]
            or not isinstance(codegraph_build, dict)
            or codegraph_build != runtime_identity["codegraph_build"]):
        raise EvalManifestError("environment lock has the wrong CodeGraph identity")

    checks = environment.get("identity_checks")
    if not isinstance(checks, list):
        raise EvalManifestError("environment lock lacks identity checks")
    arm_manifest = {
        item["id"]: item for item in manifest["arms"] if isinstance(item, dict)
    }
    for arm, key, version in (
        ("hlsgraph-v02", "hlsgraph_v02", "0.2.0"),
        ("hlsgraph-v03", "hlsgraph_v03", "0.3.0"),
    ):
        declared = environment.get(key)
        if (not isinstance(declared, dict) or declared.get("version") != version
                or _SHA256_RE.fullmatch(str(declared.get("wheel_sha256", ""))) is None):
            raise EvalManifestError(f"environment lock has invalid {arm} wheel identity")
        matching = [
            item.get("identity") for item in checks
            if isinstance(item, dict)
            and item.get("kind") == "verify-hlsgraph-wheel-installation"
            and item.get("arm") == arm
        ]
        if len(matching) != 1 or not isinstance(matching[0], dict):
            raise EvalManifestError(f"environment lock requires one {arm} identity check")
        identity = matching[0]
        hashes = (
            identity.get("wheel_sha256"), identity.get("wheel_payload_sha256"),
            identity.get("installed_payload_sha256"),
        )
        if (identity.get("schema_version") != "hlsgraph.agent_eval.wheel_identity.v1"
                or identity.get("verified") is not True
                or identity.get("version") != version
                or identity.get("wheel_sha256") != declared["wheel_sha256"]
                or any(_SHA256_RE.fullmatch(str(value or "")) is None for value in hashes)
                or identity.get("wheel_payload_sha256")
                != identity.get("installed_payload_sha256")):
            raise EvalManifestError(f"environment lock has inconsistent {arm} payload identity")
        source_hash = identity.get("source_package_sha256")
        package_hash = identity.get("wheel_package_sha256")
        source_revision = identity.get("source_revision")
        expected_revision = arm_manifest.get(arm, {}).get("revision")
        if expected_revision == "record-at-run-time":
            expected_revision = source_revision
        if (not isinstance(source_hash, str)
                or _SHA256_RE.fullmatch(source_hash) is None
                or package_hash != source_hash
                or not isinstance(source_revision, str)
                or re.fullmatch(r"[0-9a-f]{40}", source_revision) is None
                or declared.get("source_package_sha256") != source_hash
                or declared.get("wheel_package_sha256") != package_hash
                or declared.get("source_revision") != source_revision
                or declared.get("revision") != source_revision
                or source_revision != expected_revision):
            raise EvalManifestError(
                f"environment lock does not bind {arm} wheel bytes to its exact source revision"
            )
        label = "v02" if arm == "hlsgraph-v02" else "v03"
        expected_source_checks = {
            f"verify-{label}-repo-clean": "",
            f"record-{label}-revision": source_revision,
            f"verify-{label}-repo-clean-after": "",
            f"record-{label}-revision-after": source_revision,
        }
        for check_kind, stdout in expected_source_checks.items():
            matching_checks = [
                item for item in checks
                if isinstance(item, dict) and item.get("kind") == check_kind
            ]
            if (len(matching_checks) != 1
                    or matching_checks[0].get("stdout") != stdout):
                raise EvalManifestError(
                    f"environment lock lacks the exact {check_kind} source check"
                )
    revision = environment["hlsgraph_v03"].get("revision")
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise EvalManifestError("environment lock lacks the v0.3 source revision")
    workspaces = environment.get("workspaces")
    expected_workspace_keys = {
        f"{arm}/{corpus['id']}" for arm in ARM_IDS
        for corpus in load_corpus_lock()["corpora"]
    }
    if not isinstance(workspaces, dict) or set(workspaces) != expected_workspace_keys:
        raise EvalManifestError("environment lock lacks the complete workspace matrix")
    if any(
        not isinstance(value, dict)
        or value.get("workspace_identity_sha256")
        != sha256_bytes(canonical_json({
            key: item for key, item in value.items()
            if key != "workspace_identity_sha256"
        }))
        for value in workspaces.values()
    ):
        raise EvalManifestError("environment lock has an invalid workspace identity")
    cold_start = environment.get("cold_start_indexing")
    expected_cold_keys = {
        (arm, corpus["id"]) for arm in ARM_IDS
        for corpus in load_corpus_lock()["corpora"]
    }
    if not isinstance(cold_start, list):
        raise EvalManifestError("environment lock lacks cold-start indexing metrics")
    absence_checks = [
        item for item in checks
        if isinstance(item, dict) and item.get("kind") == "cold-start-index-absence"
    ]
    absence_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    expected_absence_fields = {
        "schema_version", "kind", "checkpoint", "arm", "corpus_id",
        "phase", "status", "index_relative_path", "input_tree_sha256",
        "proof_sha256",
    }
    for proof in absence_checks:
        arm = str(proof.get("arm", ""))
        corpus_id = str(proof.get("corpus_id", ""))
        checkpoint = str(proof.get("checkpoint", ""))
        key = (arm, corpus_id, checkpoint)
        expected_index = (
            ".codegraph" if arm == "codegraph"
            else ".hlsgraph" if arm.startswith("hlsgraph-")
            else None
        )
        workspace = workspaces.get(f"{arm}/{corpus_id}")
        payload = {name: value for name, value in proof.items() if name != "proof_sha256"}
        if (set(proof) != expected_absence_fields
                or proof.get("schema_version")
                != "hlsgraph.agent_eval.cold_start_absence.v1"
                or checkpoint not in {"pre_execution", "pre_index"}
                or key in absence_by_key
                or proof.get("phase") != "index"
                or proof.get("status") != "absent"
                or proof.get("index_relative_path") != expected_index
                or _SHA256_RE.fullmatch(str(proof.get("input_tree_sha256", ""))) is None
                or _SHA256_RE.fullmatch(str(proof.get("proof_sha256", ""))) is None
                or proof.get("proof_sha256") != sha256_bytes(canonical_json(payload))
                or not isinstance(workspace, dict)
                or proof.get("input_tree_sha256")
                != workspace.get("cold_start_input_sha256")):
            raise EvalManifestError("cold-start index absence proof is invalid")
        absence_by_key[key] = proof
    expected_absence_keys = {
        (arm, corpus["id"], checkpoint)
        for arm in ARM_IDS if arm != "native"
        for corpus in load_corpus_lock()["corpora"]
        for checkpoint in ("pre_execution", "pre_index")
    }
    if set(absence_by_key) != expected_absence_keys:
        raise EvalManifestError(
            "environment lock lacks complete cold-start index absence proofs"
        )
    cold_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for item in cold_start:
        if not isinstance(item, dict):
            raise EvalManifestError("cold-start indexing metric must be an object")
        key = (str(item.get("arm", "")), str(item.get("corpus_id", "")))
        if key in cold_by_key:
            raise EvalManifestError(f"duplicate cold-start indexing metric: {key}")
        cold_by_key[key] = item
    if set(cold_by_key) != expected_cold_keys:
        raise EvalManifestError("environment lock lacks the complete cold-start indexing matrix")
    expected_phase_checks: list[dict[str, Any]] = []
    expected_phase_names = {
        "codegraph": ["index"],
        "hlsgraph-v02": ["index"],
        "hlsgraph-v03": ["index", "knowledge_sync"],
    }
    for (arm, corpus_id), item in cold_by_key.items():
        if (item.get("schema_version") != "hlsgraph.agent_eval.cold_start_index.v1"
                or item.get("kind") != "cold-start-index"):
            raise EvalManifestError("cold-start indexing metric has an invalid schema")
        if arm == "native":
            if (item.get("status") != "not_applicable"
                    or item.get("phases") != []
                    or item.get("wall_time_seconds") is not None
                    or item.get("reason") != "no_index_required"):
                raise EvalManifestError("native cold-start metric must be explicitly not applicable")
            continue
        phases = item.get("phases")
        if (not isinstance(phases, list)
                or [phase.get("phase") for phase in phases if isinstance(phase, dict)]
                != expected_phase_names.get(arm)):
            raise EvalManifestError("cold-start indexing phases are incomplete or out of order")
        phase_commands: list[dict[str, str]] = []
        phase_duration = 0.0
        for phase in phases:
            duration = phase.get("wall_time_seconds")
            command_sha256 = phase.get("command_sha256")
            if (phase.get("schema_version")
                    != "hlsgraph.agent_eval.cold_start_phase.v1"
                    or phase.get("kind") != "cold-start-index-phase"
                    or phase.get("status") != "measured"
                    or isinstance(duration, bool)
                    or not isinstance(duration, (int, float))
                    or not math.isfinite(duration) or duration < 0
                    or _SHA256_RE.fullmatch(str(command_sha256 or "")) is None):
                raise EvalManifestError("cold-start indexing phase is incomplete or invalid")
            if phase.get("phase") == "index":
                initial_proof = absence_by_key.get((arm, corpus_id, "pre_execution"))
                immediate_proof = absence_by_key.get((arm, corpus_id, "pre_index"))
                if (initial_proof is None or immediate_proof is None
                        or phase.get("input_tree_sha256")
                        != immediate_proof["input_tree_sha256"]
                        or phase.get("pre_execution_absence_proof_sha256")
                        != initial_proof["proof_sha256"]
                        or phase.get("pre_index_absence_proof_sha256")
                        != immediate_proof["proof_sha256"]):
                    raise EvalManifestError(
                        "cold-start index phase is not bound to both absence proofs"
                    )
            elif any(name in phase for name in (
                "input_tree_sha256", "pre_execution_absence_proof_sha256",
                "pre_index_absence_proof_sha256",
            )):
                raise EvalManifestError(
                    "non-index cold-start phase carries an index absence proof"
                )
            phase_duration += float(duration)
            phase_commands.append({
                "phase": phase["phase"], "command_sha256": command_sha256,
            })
            expected_phase_checks.append({
                **phase, "arm": arm, "corpus_id": corpus_id,
            })
        duration = item.get("wall_time_seconds")
        if (item.get("status") != "measured"
                or isinstance(duration, bool) or not isinstance(duration, (int, float))
                or not math.isfinite(duration) or duration < 0
                or duration != round(phase_duration, 9)
                or item.get("command_sha256")
                != sha256_bytes(canonical_json(phase_commands))):
            raise EvalManifestError("cold-start indexing metric is incomplete or invalid")
    measured_cold = [
        item for item in checks
        if isinstance(item, dict) and item.get("kind") == "cold-start-index"
    ]
    if (len(measured_cold) != len(expected_cold_keys) - len(load_corpus_lock()["corpora"])
            or any(item not in cold_start for item in measured_cold)
            or any(
                item.get("arm") != "native" and item not in measured_cold
                for item in cold_start
            )):
        raise EvalManifestError(
            "cold-start indexing metrics differ from the measured preparation observations"
        )
    measured_phases = [
        item for item in checks
        if isinstance(item, dict) and item.get("kind") == "cold-start-index-phase"
    ]
    if (len(measured_phases) != len(expected_phase_checks)
            or any(item not in measured_phases for item in expected_phase_checks)
            or any(item not in expected_phase_checks for item in measured_phases)):
        raise EvalManifestError(
            "cold-start phases differ from the measured preparation observations"
        )
    return environment


def prepared_hlsgraph_identity(
    environment: dict[str, Any], arm: str,
) -> dict[str, str]:
    """Return the immutable public identity for one prepared HLSGraph arm."""
    if arm not in {"hlsgraph-v02", "hlsgraph-v03"}:
        raise EvalManifestError(f"unsupported HLSGraph arm: {arm}")
    key = arm.replace("-", "_")
    declared = environment[key]
    identity = next(
        item["identity"] for item in environment["identity_checks"]
        if item.get("kind") == "verify-hlsgraph-wheel-installation"
        and item.get("arm") == arm
    )
    result = {
        "arm": arm,
        "version": str(declared["version"]),
        "wheel_sha256": str(declared["wheel_sha256"]),
        "installed_payload_sha256": str(identity["installed_payload_sha256"]),
    }
    result.update({
        "revision": str(declared["revision"]),
        "source_revision": str(identity["source_revision"]),
        "source_package_sha256": str(identity["source_package_sha256"]),
        "wheel_package_sha256": str(identity["wheel_package_sha256"]),
    })
    return result


def safe_relative_path(value: str) -> Path:
    path = Path(value.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise EvalManifestError(f"unsafe relative path: {value!r}")
    return path


__all__ = [
    "ANSWER_SCHEMA_PATH", "ARM_IDS", "CODEGRAPH_BUILD_IDENTITY_SCHEMA_VERSION",
    "CODEGRAPH_OFFLINE_ENV", "CORPUS_LOCK_PATH", "EvalManifestError",
    "ENVIRONMENT_SCHEMA_VERSION", "HERE", "QUESTIONS_PATH", "STATIC_CASES_PATH",
    "RUNTIME_TREE_ALGORITHM", "SUITE_PATH", "asset_digest", "canonical_json",
    "capture_codegraph_build_identity", "cold_start_input_identity",
    "load_corpus_lock",
    "capture_official_runtime_identity", "harness_digest", "load_environment_lock",
    "load_manifest", "load_questions", "official_process_environment",
    "load_static_cases", "prepared_hlsgraph_identity", "resolve_command_argv",
    "require_official_ext4_directory", "resolve_local_executable",
    "runtime_tree_identity", "safe_relative_path", "sha256_bytes", "sha256_file",
    "require_official_linux_wsl2", "verify_evaluation_checkout",
    "verify_official_runtime_identity", "verify_prepared_workspace", "workspace_identity",
]
