"""Verify that an interpreter imports exactly the supplied HLSGraph wheel.

This helper intentionally uses only the Python standard library so it can run
inside each comparison environment before any index is built.  It rejects
editable/source installs and compares every installed HLSGraph package and
distribution-metadata byte (except installer-generated records) with the
candidate wheel.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import io
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys
from typing import Any, Sequence
from urllib.parse import unquote, urlsplit
import zipfile


_GENERATED_DIST_INFO = {"direct_url.json", "INSTALLER", "RECORD", "REQUESTED"}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_member(name: str) -> str:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise RuntimeError(f"unsafe wheel member: {name!r}")
    return path.as_posix()


def _is_payload(name: str) -> bool:
    path = PurePosixPath(name)
    if not path.parts or name.endswith("/"):
        return False
    if "__pycache__" in path.parts or path.suffix == ".pyc":
        return False
    first = path.parts[0].casefold()
    if first == "hlsgraph":
        return True
    if first.startswith("hlsgraph-") and first.endswith(".dist-info"):
        return path.name not in _GENERATED_DIST_INFO
    return False


def _is_package_payload(name: str) -> bool:
    path = PurePosixPath(name)
    return bool(
        path.parts
        and path.parts[0].casefold() == "hlsgraph"
        and not name.endswith("/")
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    )


def _is_wheel_package_member(name: str) -> bool:
    path = PurePosixPath(name)
    return bool(
        path.parts
        and path.parts[0].casefold() == "hlsgraph"
        and not name.endswith("/")
    )


def _is_link(path: Path) -> bool:
    is_junction = getattr(path, "is_junction", None)
    return path.is_symlink() or bool(is_junction and is_junction())


def _zip_member_is_link(info: zipfile.ZipInfo) -> bool:
    # Wheels are ZIP archives.  A producer can encode a POSIX symlink in the
    # upper Unix-mode bits even though ZipFile.read() presents its target as
    # ordinary bytes, so reject that representation before comparing content.
    unix_mode = (info.external_attr >> 16) & 0xFFFF
    return stat.S_IFMT(unix_mode) == stat.S_IFLNK


def _payload_digest(payload: dict[str, bytes]) -> str:
    digest = hashlib.sha256()
    for name, data in sorted(payload.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(data).digest())
        digest.update(b"\0")
    return digest.hexdigest()


def _wheel_payload(wheel: Path) -> tuple[dict[str, bytes], dict[str, bytes], str]:
    if _is_link(wheel) or not wheel.is_file() or wheel.suffix.casefold() != ".whl":
        raise RuntimeError("candidate wheel is missing or does not have a .whl suffix")
    data = wheel.read_bytes()
    payload: dict[str, bytes] = {}
    package_payload: dict[str, bytes] = {}
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        seen: set[str] = set()
        for info in archive.infolist():
            name = _safe_member(info.filename)
            if name in seen:
                raise RuntimeError(f"duplicate wheel member: {name}")
            seen.add(name)
            if _zip_member_is_link(info):
                raise RuntimeError(f"wheel payload contains a linked member: {name}")
            if _is_payload(name) or _is_wheel_package_member(name):
                member_data = archive.read(info)
                if _is_payload(name):
                    payload[name] = member_data
                if _is_wheel_package_member(name):
                    package_payload[name] = member_data
    if "hlsgraph/__init__.py" not in payload:
        raise RuntimeError("wheel does not contain hlsgraph/__init__.py")
    if not any(name.endswith(".dist-info/METADATA") for name in payload):
        raise RuntimeError("wheel does not contain HLSGraph distribution metadata")
    return payload, package_payload, _sha256(data)


def _source_package_payload(source_repo: Path) -> tuple[dict[str, bytes], str, str]:
    """Read the exact public package from one clean Git checkout.

    The comparison intentionally uses the checked-out bytes rather than Git
    object contents: this catches filters, line-ending conversion, generated
    package data and a wheel assembled from a different worktree.  Git status
    supplies the independent requirement that those bytes belong to HEAD.
    """
    supplied_repo = source_repo.absolute()
    if _is_link(supplied_repo) or not supplied_repo.is_dir():
        raise RuntimeError("source repository is missing or linked")
    source_repo = supplied_repo.resolve()

    def git(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args], cwd=source_repo, capture_output=True, text=True,
            check=False, timeout=20,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"cannot inspect source repository: {detail}")
        return completed.stdout.strip()

    top_level = Path(git("rev-parse", "--show-toplevel")).resolve()
    if top_level != source_repo:
        raise RuntimeError("source repository must be the Git worktree root")
    revision = git("rev-parse", "HEAD")
    if len(revision) != 40 or any(ch not in "0123456789abcdef" for ch in revision):
        raise RuntimeError("source repository HEAD is not a full SHA-1 revision")
    if git("status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("source repository must be clean before wheel verification")

    source_root = source_repo / "src" / "hlsgraph"
    if _is_link(source_repo / "src") or _is_link(source_root) or not source_root.is_dir():
        raise RuntimeError("source package root is missing or linked")
    payload: dict[str, bytes] = {}
    for current, directories, filenames in os.walk(source_root, followlinks=False):
        current_path = Path(current)
        kept_directories: list[str] = []
        for directory in directories:
            child = current_path / directory
            if _is_link(child):
                raise RuntimeError(
                    f"source package contains linked directory: "
                    f"{child.relative_to(source_repo).as_posix()}"
                )
            if directory != "__pycache__":
                kept_directories.append(directory)
        directories[:] = kept_directories
        for filename in filenames:
            path = current_path / filename
            if filename.endswith((".pyc", ".pyo")):
                continue
            relative = _safe_member(path.relative_to(source_repo / "src").as_posix())
            if not _is_package_payload(relative):
                continue
            if _is_link(path) or not path.is_file():
                raise RuntimeError(f"source package contains linked or invalid file: {relative}")
            resolved = path.resolve()
            try:
                resolved.relative_to(source_root)
            except ValueError as exc:
                raise RuntimeError(f"source package file escapes package root: {relative}") from exc
            payload[relative] = resolved.read_bytes()
    if "hlsgraph/__init__.py" not in payload:
        raise RuntimeError("source repository does not contain src/hlsgraph/__init__.py")
    tracked_package = {
        _safe_member(PurePosixPath(name).relative_to("src").as_posix())
        for name in git("ls-files", "--", "src/hlsgraph").splitlines()
        if _is_package_payload(
            _safe_member(PurePosixPath(name).relative_to("src").as_posix())
        )
    }
    if set(payload) != tracked_package:
        missing = sorted(tracked_package - set(payload))
        extra = sorted(set(payload) - tracked_package)
        raise RuntimeError(
            "source package differs from the Git-tracked package "
            f"(missing={missing}, extra={extra})"
        )
    if git("rev-parse", "HEAD") != revision:
        raise RuntimeError("source repository revision changed during wheel verification")
    if git("status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("source repository changed during wheel verification")
    return payload, _payload_digest(payload), revision


def _bind_wheel_to_source(
    wheel_package: dict[str, bytes], source_repo: Path,
) -> dict[str, str]:
    source_package, source_sha256, revision = _source_package_payload(source_repo)
    if set(wheel_package) != set(source_package):
        missing = sorted(set(source_package) - set(wheel_package))
        extra = sorted(set(wheel_package) - set(source_package))
        raise RuntimeError(
            "wheel/source package set mismatch "
            f"(missing_from_wheel={missing}, extra_in_wheel={extra})"
        )
    mismatched = sorted(
        name for name in source_package if wheel_package[name] != source_package[name]
    )
    if mismatched:
        raise RuntimeError(f"wheel/source package bytes mismatch: {mismatched}")
    wheel_sha256 = _payload_digest(wheel_package)
    if wheel_sha256 != source_sha256:
        raise RuntimeError("wheel/source package digest mismatch")
    return {
        "source_package_sha256": source_sha256,
        "wheel_package_sha256": wheel_sha256,
        "source_revision": revision,
    }


def _installed_payload(distribution: importlib.metadata.Distribution) -> tuple[dict[str, bytes], str]:
    files = distribution.files
    if files is None:
        raise RuntimeError("installed distribution has no RECORD file list")
    root = Path(distribution.locate_file("")).resolve()
    recorded_dist_info = {
        PurePosixPath(str(item).replace("\\", "/")).parts[0]
        for item in files
        if PurePosixPath(str(item).replace("\\", "/")).parts
        and PurePosixPath(str(item).replace("\\", "/")).parts[0].casefold().startswith(
            "hlsgraph-"
        )
        and PurePosixPath(str(item).replace("\\", "/")).parts[0].casefold().endswith(
            ".dist-info"
        )
    }
    if len(recorded_dist_info) != 1:
        raise RuntimeError("installed distribution has ambiguous HLSGraph metadata")
    actual_dist_info = {
        item.name for item in root.glob("hlsgraph-*.dist-info") if item.is_dir()
    }
    if actual_dist_info != recorded_dist_info:
        raise RuntimeError("installed environment has extra or missing HLSGraph metadata")

    payload: dict[str, bytes] = {}
    scan_roots = [root / "hlsgraph", root / next(iter(recorded_dist_info))]
    for scan_root in scan_roots:
        if _is_link(scan_root) or not scan_root.is_dir():
            raise RuntimeError(f"installed payload root is missing or linked: {scan_root.name}")
        for current, directories, filenames in os.walk(scan_root, followlinks=False):
            current_path = Path(current)
            for directory in directories:
                child = current_path / directory
                if _is_link(child):
                    raise RuntimeError(f"installed payload contains linked directory: {directory}")
            for filename in filenames:
                path = current_path / filename
                relative = _safe_member(path.relative_to(root).as_posix())
                if not _is_payload(relative):
                    continue
                if _is_link(path) or not path.is_file():
                    raise RuntimeError(f"installed payload is missing or linked: {relative}")
                resolved = path.resolve()
                try:
                    resolved.relative_to(root)
                except ValueError as exc:
                    raise RuntimeError(
                        f"installed payload escapes its environment: {relative}"
                    ) from exc
                payload[relative] = resolved.read_bytes()
    record_text = distribution.read_text("RECORD")
    if not record_text:
        raise RuntimeError("installed distribution is missing RECORD")
    # Parse rather than merely hash it, so a malformed RECORD cannot serve as identity.
    list(csv.reader(io.StringIO(record_text)))
    return payload, _sha256(record_text.encode("utf-8"))


def _direct_url_kind(
    distribution: importlib.metadata.Distribution, wheel: Path | None = None,
) -> str:
    raw = distribution.read_text("direct_url.json")
    if not raw:
        return "absent"
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("installed direct_url.json is malformed") from exc
    if value.get("dir_info", {}).get("editable") is True:
        raise RuntimeError("editable HLSGraph installations are forbidden")
    url = value.get("url")
    if not isinstance(url, str):
        raise RuntimeError("installed direct_url.json has no URL")
    source_name = Path(unquote(urlsplit(url).path)).name
    if not source_name.casefold().endswith(".whl"):
        raise RuntimeError("HLSGraph must be installed from a wheel, not a source directory")
    if wheel is not None and source_name.casefold() != wheel.name.casefold():
        raise RuntimeError("installed HLSGraph wheel name differs from the supplied candidate")
    return "wheel"


def inspect_installed(
    expected_version: str, expected_payload_sha256: str | None = None,
) -> dict[str, Any]:
    distribution = importlib.metadata.distribution("hlsgraph")
    if distribution.version != expected_version:
        raise RuntimeError(
            f"installed HLSGraph version is {distribution.version}, expected {expected_version}"
        )
    module = importlib.import_module("hlsgraph")
    if getattr(module, "__version__", None) != expected_version:
        raise RuntimeError("hlsgraph.__version__ disagrees with distribution metadata")
    installed, record_sha256 = _installed_payload(distribution)
    root = Path(distribution.locate_file("")).resolve()
    module_file = Path(getattr(module, "__file__", "")).resolve()
    try:
        module_relative = module_file.relative_to(root).as_posix()
    except ValueError as exc:
        raise RuntimeError("import hlsgraph resolves outside its distribution environment") from exc
    if not module_relative.startswith("hlsgraph/"):
        raise RuntimeError("import hlsgraph does not resolve to the wheel package")
    payload_sha256 = _payload_digest(installed)
    if expected_payload_sha256 is not None and payload_sha256 != expected_payload_sha256:
        raise RuntimeError("installed HLSGraph payload changed after preparation")
    return {
        "schema_version": "hlsgraph.agent_eval.wheel_identity.v1",
        "verified": True,
        "distribution": "hlsgraph",
        "version": distribution.version,
        "installed_payload_sha256": payload_sha256,
        "record_sha256": record_sha256,
        "payload_files": len(installed),
        "module_location_class": "distribution/hlsgraph",
        "direct_url_kind": _direct_url_kind(distribution),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    }


def inspect_installation(
    wheel: Path, expected_version: str, source_repo: Path | None = None,
) -> dict[str, Any]:
    supplied_wheel = wheel.absolute()
    if _is_link(supplied_wheel):
        raise RuntimeError("candidate wheel must not be a symlink or junction")
    wheel = supplied_wheel.resolve()
    expected, wheel_package, wheel_sha256 = _wheel_payload(wheel)
    distribution = importlib.metadata.distribution("hlsgraph")
    installed_identity = inspect_installed(expected_version)
    installed, record_sha256 = _installed_payload(distribution)
    if set(installed) != set(expected):
        missing = sorted(set(expected) - set(installed))
        extra = sorted(set(installed) - set(expected))
        raise RuntimeError(
            f"installed wheel payload set mismatch (missing={missing}, extra={extra})"
        )
    mismatched = sorted(name for name in expected if installed[name] != expected[name])
    if mismatched:
        raise RuntimeError(f"installed wheel payload bytes mismatch: {mismatched}")

    direct_url_kind = _direct_url_kind(distribution, wheel)
    payload_sha256 = _payload_digest(installed)
    result = {
        "schema_version": "hlsgraph.agent_eval.wheel_identity.v1",
        "verified": True,
        "distribution": "hlsgraph",
        "version": distribution.version,
        "wheel_filename": wheel.name,
        "wheel_sha256": wheel_sha256,
        "installed_payload_sha256": payload_sha256,
        "wheel_payload_sha256": _payload_digest(expected),
        "record_sha256": record_sha256,
        "payload_files": len(installed),
        "module_location_class": "distribution/hlsgraph",
        "direct_url_kind": direct_url_kind,
        "python": installed_identity["python"],
    }
    if source_repo is not None:
        result.update(_bind_wheel_to_source(wheel_package, source_repo))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--expected-version", required=True)
    parser.add_argument("--expected-payload-sha256")
    parser.add_argument(
        "--source-repo", type=Path,
        help="clean source checkout whose src/hlsgraph bytes must equal the wheel package",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.wheel is not None and args.expected_payload_sha256 is not None:
        raise SystemExit("choose --wheel or --expected-payload-sha256, not both")
    if args.wheel is None and args.expected_payload_sha256 is None:
        raise SystemExit("one of --wheel or --expected-payload-sha256 is required")
    if args.expected_payload_sha256 is not None and args.source_repo is not None:
        raise SystemExit("--source-repo is only valid together with --wheel")
    if (args.wheel is not None
            and args.expected_version in {"0.2.0", "0.3.0"}
            and args.source_repo is None):
        raise SystemExit(
            f"v{args.expected_version} evaluation wheel verification requires --source-repo"
        )
    value = (inspect_installation(args.wheel, args.expected_version, args.source_repo)
             if args.wheel is not None else
             inspect_installed(args.expected_version, args.expected_payload_sha256))
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
