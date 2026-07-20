"""Run-scoped staging helpers shared by execution backends and the SDK.

The staging area is an evidence boundary, not an operating-system sandbox.  A
runner executes only from a fresh directory and only declared regular files may
cross back into the bundle.  Every crossing is size- and digest-checked without
following symbolic links or Windows reparse points.
"""
from __future__ import annotations

import hashlib
import os
import re
import stat
import tempfile
from pathlib import Path

from ..model import safe_relative_path


DEFAULT_MAX_OUTPUT_BYTES = 256 * 1024 * 1024
MAX_TOTAL_TRANSFER_BYTES = 512 * 1024 * 1024
_WINDOWS_RESERVED = frozenset(
    {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)),
     *(f"lpt{i}" for i in range(1, 10))}
)
_REPARSE_ATTRIBUTE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


class StagingError(ValueError):
    """A path or byte stream violated the run-staging contract."""


def runner_relative_path(value: str, field_name: str = "runner path") -> str:
    """Return a conservative, portable relative path for a staging boundary."""
    try:
        normalized = safe_relative_path(value, field_name)
    except ValueError as exc:
        raise StagingError(str(exc)) from exc
    for component in normalized.split("/"):
        if (any(ord(char) < 32 or char in '<>:"|?*' for char in component)
                or component.endswith((" ", "."))
                or component.split(".", 1)[0].casefold() in _WINDOWS_RESERVED):
            raise StagingError(
                f"{field_name} contains a non-portable path component: {value!r}"
            )
    return normalized


def is_link_or_reparse(path: Path, *, stat_result: os.stat_result | None = None) -> bool:
    value = stat_result if stat_result is not None else path.lstat()
    return stat.S_ISLNK(value.st_mode) or bool(
        getattr(value, "st_file_attributes", 0) & _REPARSE_ATTRIBUTE
    )


def _checked_root(root: str | Path) -> Path:
    value = Path(root).absolute()
    try:
        info = value.lstat()
    except OSError as exc:
        raise StagingError(f"staging root is unavailable: {value}") from exc
    if not stat.S_ISDIR(info.st_mode) or is_link_or_reparse(value, stat_result=info):
        raise StagingError("staging root must be a real directory")
    return value


def checked_path(
    root: str | Path, relative: str, *, require_file: bool = True,
) -> Path:
    """Resolve an existing path beneath ``root`` without following links."""
    base = _checked_root(root)
    normalized = runner_relative_path(relative)
    current = base
    parts = normalized.split("/")
    for index, component in enumerate(parts):
        current = current / component
        try:
            info = current.lstat()
        except OSError as exc:
            raise StagingError(f"staged path does not exist: {normalized}") from exc
        if is_link_or_reparse(current, stat_result=info):
            raise StagingError(f"staged path contains a link or reparse point: {normalized}")
        leaf = index == len(parts) - 1
        if not leaf and not stat.S_ISDIR(info.st_mode):
            raise StagingError(f"staged path parent is not a directory: {normalized}")
        if leaf and require_file and not stat.S_ISREG(info.st_mode):
            raise StagingError(f"staged output is not a regular file: {normalized}")
    return current


def read_verified_file(
    root: str | Path,
    relative: str,
    *,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
    max_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
) -> tuple[bytes, int, str, Path]:
    """Read one bounded regular file and verify its identity twice."""
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes < 0:
        raise StagingError("max_bytes must be a non-negative integer")
    path = checked_path(root, relative, require_file=True)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise StagingError(f"cannot open staged regular file: {relative}") from exc
    chunks: list[bytes] = []
    digest = hashlib.sha256()
    size = 0
    try:
        info = os.fstat(descriptor)
        if (not stat.S_ISREG(info.st_mode)
                or bool(getattr(info, "st_file_attributes", 0) & _REPARSE_ATTRIBUTE)):
            raise StagingError(f"staged output is not a regular file: {relative}")
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, max_bytes - size + 1))
            if not chunk:
                break
            size += len(chunk)
            if size > max_bytes:
                raise StagingError(f"staged file exceeds byte limit: {relative}")
            digest.update(chunk)
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    actual = digest.hexdigest()
    # Recheck the path after reading to catch ordinary replacement races.  The
    # descriptor identity protects the bytes just read; this check protects the
    # path-to-evidence claim returned to the caller.
    after = checked_path(root, relative, require_file=True).lstat()
    identity_before = (
        getattr(info, "st_dev", None), getattr(info, "st_ino", None),
        getattr(info, "st_mtime_ns", None), info.st_size,
    )
    identity_after = (
        getattr(after, "st_dev", None), getattr(after, "st_ino", None),
        getattr(after, "st_mtime_ns", None), after.st_size,
    )
    if identity_after != identity_before or after.st_size != size:
        raise StagingError(f"staged file changed while it was read: {relative}")
    if expected_size is not None and size != expected_size:
        raise StagingError(f"staged file size mismatch: {relative}")
    if expected_sha256 is not None and actual != expected_sha256.casefold():
        raise StagingError(f"staged file hash mismatch: {relative}")
    return b"".join(chunks), size, actual, path


def write_new_file(root: str | Path, relative: str, data: bytes) -> Path:
    """Create an input/output transfer file without replacing any path."""
    base = _checked_root(root)
    normalized = runner_relative_path(relative)
    target = base.joinpath(*normalized.split("/"))
    parent = base
    for component in normalized.split("/")[:-1]:
        parent = parent / component
        try:
            parent.mkdir()
        except FileExistsError:
            pass
        info = parent.lstat()
        if not stat.S_ISDIR(info.st_mode) or is_link_or_reparse(parent, stat_result=info):
            raise StagingError(f"staging parent is not a real directory: {normalized}")
    flags = (os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
             | getattr(os, "O_NOFOLLOW", 0))
    try:
        descriptor = os.open(target, flags, 0o600)
    except OSError as exc:
        raise StagingError(f"staged path already exists or is unsafe: {normalized}") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return target


def create_run_directory(project_root: str | Path) -> tuple[Path, Path]:
    """Create a unique local run directory under the bundle's private area."""
    project = Path(project_root).resolve()
    _checked_root(project)
    # Create one component at a time and inspect it before using it as the
    # parent of the next mkdir.  ``mkdir(parents=True)`` would follow an
    # attacker-replaced .hlsgraph link before we had a chance to reject it.
    current = project
    for component in (".hlsgraph", "runner-staging"):
        current = current / component
        try:
            current.mkdir()
        except FileExistsError:
            pass
        info = current.lstat()
        if not stat.S_ISDIR(info.st_mode) or is_link_or_reparse(current, stat_result=info):
            raise StagingError("runner staging parent must be a real directory")
    parent = current
    run = Path(tempfile.mkdtemp(prefix="run-", dir=parent)).absolute()
    _checked_root(run)
    return run, parent.absolute()


def remove_run_directory(run: str | Path, parent: str | Path) -> None:
    """Delete only one verified runner-owned tree, never following links."""
    root = Path(run).absolute()
    owner = Path(parent).absolute()
    if root.parent != owner or not re.fullmatch(r"run-[A-Za-z0-9_.-]+", root.name):
        raise StagingError("refusing to remove a directory outside runner staging")
    if not root.exists() and not root.is_symlink():
        return

    def remove(path: Path) -> None:
        info = path.lstat()
        if is_link_or_reparse(path, stat_result=info):
            if stat.S_ISDIR(info.st_mode):
                os.rmdir(path)
            else:
                path.unlink()
            return
        if stat.S_ISDIR(info.st_mode):
            with os.scandir(path) as entries:
                children = [Path(entry.path) for entry in entries]
            for child in children:
                remove(child)
            os.rmdir(path)
        else:
            path.unlink()

    remove(root)


__all__ = [
    "DEFAULT_MAX_OUTPUT_BYTES", "MAX_TOTAL_TRANSFER_BYTES", "StagingError",
    "checked_path", "create_run_directory", "is_link_or_reparse",
    "read_verified_file", "remove_run_directory", "runner_relative_path",
    "write_new_file",
]
