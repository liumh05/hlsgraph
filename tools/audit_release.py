#!/usr/bin/env python3
"""Fail-closed hygiene audit for the public tree, wheel, and sdist."""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import re
import sys
import tarfile
import zipfile
from email import policy
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from typing import Iterable


FORBIDDEN_NAMES = (
    "/.hlsgraph/", "__pycache__", ".pytest_cache", ".wheel-test",
    ".packaging-test", "/build/", ".egg-info/", ".db", ".sqlite",
    ".pyc", ".pyo", "/.env", ".pem", ".key",
)
ALLOWED_SDIST_EGG_INFO = frozenset({
    "PKG-INFO",
    "SOURCES.txt",
    "dependency_links.txt",
    "entry_points.txt",
    "requires.txt",
    "top_level.txt",
})
SECRET_PATTERNS = (
    ("private key", re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("assigned credential", re.compile(
        rb"(?i)(?:api[_-]?key|(?:access|auth|refresh)[_-]?token|"
        rb"client[_-]?secret|password|passwd|credential)\s*[:=]\s*"
        rb"['\"]?[A-Za-z0-9_./+=:@-]{8,}"
    )),
    ("license server", re.compile(
        rb"(?i)license[_-]?server\s*[:=]\s*[^\s,;]+"
    )),
    ("bearer credential", re.compile(
        rb"(?i)authorization\s*:\s*bearer\s+[A-Za-z0-9._~+/=-]{12,}"
    )),
    ("credential in URL", re.compile(
        rb"(?i)https?://[^/@\s:]+:[^/@\s]+@"
    )),
    ("GitHub token", re.compile(
        rb"(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{40,})"
    )),
    ("cloud/API token", re.compile(
        rb"(?:AKIA[0-9A-Z]{16}|sk-[A-Za-z0-9_-]{20,})"
    )),
    ("Windows absolute path", re.compile(
        rb"(?i)(?<![A-Za-z0-9+.-])[A-Za-z]:[\\/](?![\\/])"
    )),
    ("POSIX user-home path", re.compile(rb"/(?:home|Users)/[^/\s]+/")),
)


def _public_boundary_pattern(*parts: bytes, word: bool = False) -> re.Pattern[bytes]:
    escaped = re.escape(b"".join(parts))
    if word:
        escaped = rb"(?<![A-Za-z0-9_])" + escaped + rb"(?![A-Za-z0-9_])"
    return re.compile(escaped, re.IGNORECASE | re.ASCII)


# The final boolean is true only for short identifiers known to occur as
# unrelated symbols in audited third-party minified files.
PUBLIC_BOUNDARY_PATTERNS = (
    ("non-public repository identifier", _public_boundary_pattern(
        b"hlsgraph", b"-", b"research",
    ), False),
    ("non-public roadmap document", _public_boundary_pattern(
        b"research", b"-", b"integration",
    ), False),
    ("non-public roadmap marker 1", _public_boundary_pattern(
        b"HLS", b"Pilot", word=True,
    ), False),
    ("non-public roadmap marker 2", _public_boundary_pattern(
        b"Timely", b"HLS", word=True,
    ), False),
    ("non-public roadmap marker 3", _public_boundary_pattern(
        b"G", b"NN", word=True,
    ), True),
    ("non-public roadmap marker 4", _public_boundary_pattern(
        b"R", b"CD", word=True,
    ), True),
    ("non-public roadmap marker 5", _public_boundary_pattern(
        b"control", b"ler", word=True,
    ), False),
    ("non-public roadmap marker 6", _public_boundary_pattern(
        b"agent", b"ic", word=True,
    ), False),
    ("historical personal address", _public_boundary_pattern(
        b"1964722203", b"@", b"qq", b".", b"com",
    ), False),
)
PUBLIC_BOUNDARY_SHORT_EXCLUSIONS = frozenset({
    "src/hlsgraph/render/vendor/elk.bundled.js",
    "src/hlsgraph/render/vendor/cytoscape.min.js",
    "hlsgraph/render/vendor/elk.bundled.js",
    "hlsgraph/render/vendor/cytoscape.min.js",
})
REQUIRED_SDIST = {
    "LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md", "SECURITY.md",
    "CONTRIBUTING.md", "CODE_OF_CONDUCT.md", "DCO", "CITATION.cff",
    "sbom.spdx.json", "docs/references.md", "docs/privacy-and-security.md",
}
SOURCE_SKIP_DIRS = frozenset({
    ".git", ".hlsgraph", ".mypy_cache", ".nox", ".packaging-test",
    ".pytest_cache", ".ruff_cache", ".tox", ".venv", ".wheel-test",
    "__pycache__", "build", "dist", "htmlcov",
})
SOURCE_SCAN_EXCLUSIONS = frozenset({
    # This file necessarily contains the credential-detection expressions.
    "tools/audit_release.py",
})


def _allowed_sdist_egg_info(name: str) -> bool:
    parts = PurePosixPath(name.replace("\\", "/")).parts
    return (
        len(parts) == 3
        and parts[0] == "src"
        and parts[1] == "hlsgraph.egg-info"
        and parts[2] in ALLOWED_SDIST_EGG_INFO
    )


def _forbidden(name: str, *, sdist: bool = False) -> str | None:
    relative = name.replace("\\", "/").lstrip("/")
    normalized = "/" + relative
    lowered = normalized.casefold()
    for item in FORBIDDEN_NAMES:
        if item.casefold() not in lowered:
            continue
        if item == ".egg-info/" and sdist and _allowed_sdist_egg_info(relative):
            continue
        return item
    return None


def _scan(name: str, data: bytes) -> list[str]:
    issues: list[str] = []
    if len(data) <= 8 * 1024 * 1024:
        for label, pattern in SECRET_PATTERNS:
            if pattern.search(data):
                issues.append(f"sensitive {label} pattern in {name}")
    normalized_name = name.replace("\\", "/").lstrip("/")
    encoded_name = normalized_name.encode("utf-8", errors="surrogateescape")
    for label, pattern, allow_short_vendor_symbol in PUBLIC_BOUNDARY_PATTERNS:
        if pattern.search(encoded_name):
            issues.append(f"{label} in member name {name}")
        if (
            allow_short_vendor_symbol
            and normalized_name in PUBLIC_BOUNDARY_SHORT_EXCLUSIONS
        ):
            continue
        if pattern.search(data):
            issues.append(f"{label} in {name}")
    return issues


def _audit_source_tree(root: Path) -> list[str]:
    """Scan files intended for the public repository, excluding build state."""
    issues: list[str] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        parts = PurePosixPath(relative).parts
        if any(part in SOURCE_SKIP_DIRS or part.endswith(".egg-info") for part in parts):
            continue
        if relative in SOURCE_SCAN_EXCLUSIONS or not path.is_file():
            continue
        if marker := _forbidden(relative):
            issues.append(f"forbidden public-tree member {relative} ({marker})")
            continue
        try:
            issues.extend(_scan(relative, path.read_bytes()))
        except OSError as exc:
            issues.append(f"cannot read public-tree member {relative}: {exc}")
    return issues


def _package_verification_code(files: Iterable[tuple[str, bytes]]) -> str:
    """Compute the SPDX package verification code from analyzed files."""
    hashes = sorted(
        hashlib.sha1(data).hexdigest()  # noqa: S324 - SPDX 2.3 mandates SHA-1
        for _name, data in files
    )
    concatenated = "".join(hashes)
    # SHA-1 is required by SPDX 2.3 packageVerificationCode.
    return hashlib.sha1(concatenated.encode("ascii")).hexdigest()  # noqa: S324


def _audit_sbom(sbom_data: bytes, root: Path) -> list[str]:
    issues: list[str] = []
    try:
        sbom = json.loads(sbom_data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return [f"invalid root SBOM: {exc}"]
    if sbom.get("spdxVersion") != "SPDX-2.3":
        issues.append("root SBOM is not SPDX-2.3")

    file_data: dict[str, tuple[str, bytes]] = {}
    for item in sbom.get("files", []):
        spdx_id = item.get("SPDXID")
        file_name = item.get("fileName")
        if not isinstance(spdx_id, str) or not isinstance(file_name, str):
            issues.append("SBOM file entry lacks SPDXID or fileName")
            continue
        if spdx_id in file_data:
            issues.append(f"duplicate SBOM file SPDXID: {spdx_id}")
            continue
        pure_name = PurePosixPath(file_name)
        if not file_name.startswith("./") or ".." in pure_name.parts:
            issues.append(f"unsafe SBOM fileName: {file_name}")
            continue
        candidate = root.joinpath(*pure_name.parts)
        if not candidate.is_file():
            issues.append(f"SBOM file is missing: {file_name}")
            continue
        data = candidate.read_bytes()
        checksums = {
            value.get("algorithm", "").upper(): value.get("checksumValue", "").lower()
            for value in item.get("checksums", []) if isinstance(value, dict)
        }
        actual_sha256 = hashlib.sha256(data).hexdigest()
        if checksums.get("SHA256") != actual_sha256:
            issues.append(f"invalid SBOM SHA256 for {file_name}")
        file_data[spdx_id] = (file_name, data)

    for package in sbom.get("packages", []):
        if package.get("filesAnalyzed") is not True:
            continue
        name = package.get("name", "<unnamed>")
        file_ids = package.get("hasFiles")
        if not isinstance(file_ids, list) or not file_ids:
            issues.append(f"analyzed SBOM package has no files: {name}")
            continue
        missing = [spdx_id for spdx_id in file_ids if spdx_id not in file_data]
        if missing:
            issues.append(f"SBOM package {name} has unknown files: {missing}")
            continue
        expected = package.get("packageVerificationCode", {}).get(
            "packageVerificationCodeValue"
        )
        actual = _package_verification_code(file_data[spdx_id] for spdx_id in file_ids)
        if expected != actual:
            issues.append(f"invalid SPDX packageVerificationCode for {name}")
    return issues


def _audit_wheel_metadata(data: bytes) -> list[str]:
    """Validate core metadata using RFC-aware parsing (LF and CRLF safe)."""
    issues: list[str] = []
    metadata = BytesParser(policy=policy.compat32).parsebytes(data)
    if metadata.get("Version") != "0.1.1":
        issues.append("wheel metadata is not final v0.1.1")
    urls = metadata.get_all("Project-URL", [])
    if not any("https://github.com/liumh05/hlsgraph" in url for url in urls):
        issues.append("wheel metadata has stale repository URLs")
    return issues


def _audit_wheel(path: Path, root: Path, root_sbom: bytes) -> list[str]:
    issues: list[str] = []
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        for name in names:
            if marker := _forbidden(name):
                issues.append(f"forbidden wheel member {name} ({marker})")
            issues.extend(_scan(name, archive.read(name)))
        roots = {name.split("/", 1)[0] for name in names if ".dist-info/" in name}
        if len(roots) != 1:
            issues.append("wheel must have exactly one dist-info root")
            return issues
        dist_info = roots.pop()
        required = {
            f"{dist_info}/METADATA", f"{dist_info}/RECORD",
            f"{dist_info}/sboms/sbom.spdx.json",
        }
        missing = required - set(names)
        if missing:
            issues.append(f"wheel is missing: {sorted(missing)}")
            return issues
        license_members = [
            name for name in names if name.startswith(f"{dist_info}/licenses/")
        ]
        if not any(name.endswith("/LICENSE") for name in license_members):
            issues.append("wheel has no Apache-2.0 LICENSE in dist-info/licenses")

        issues.extend(_audit_wheel_metadata(
            archive.read(f"{dist_info}/METADATA")
        ))

        record_rows = list(csv.reader(io.StringIO(
            archive.read(f"{dist_info}/RECORD").decode("utf-8")
        )))
        recorded = {row[0]: row for row in record_rows}
        for name in names:
            if name == f"{dist_info}/RECORD":
                continue
            row = recorded.get(name)
            if not row or len(row) < 3 or not row[1].startswith("sha256="):
                issues.append(f"missing RECORD hash: {name}")
                continue
            data = archive.read(name)
            digest = base64.urlsafe_b64encode(
                hashlib.sha256(data).digest()
            ).rstrip(b"=").decode("ascii")
            if row[1] != "sha256=" + digest or row[2] != str(len(data)):
                issues.append(f"invalid RECORD entry: {name}")

        wheel_sbom = archive.read(f"{dist_info}/sboms/sbom.spdx.json")
        if wheel_sbom != root_sbom:
            issues.append("wheel SBOM does not exactly match root sbom.spdx.json")
        try:
            sbom = json.loads(root_sbom)
            for item in sbom.get("files", []):
                file_name = item.get("fileName", "")
                if not file_name.startswith("./src/"):
                    continue
                member_name = file_name[len("./src/"):]
                source_path = root / file_name[len("./"):]
                if member_name not in names:
                    issues.append(f"wheel is missing SBOM vendor file: {member_name}")
                elif source_path.is_file() and archive.read(member_name) != source_path.read_bytes():
                    issues.append(f"wheel vendor bytes differ from source: {member_name}")
        except json.JSONDecodeError:
            # The root-SBOM audit reports the more specific parse failure.
            pass
    return issues


def _audit_sdist(path: Path, root_sbom: bytes) -> list[str]:
    issues: list[str] = []
    with tarfile.open(path, "r:gz") as archive:
        files = [item for item in archive.getmembers() if item.isfile()]
        stripped = {"/".join(item.name.split("/")[1:]): item for item in files}
        for name, member in stripped.items():
            if marker := _forbidden(name, sdist=True):
                issues.append(f"forbidden sdist member {name} ({marker})")
            stream = archive.extractfile(member)
            if stream:
                issues.extend(_scan(name, stream.read()))
        missing = REQUIRED_SDIST - set(stripped)
        if missing:
            issues.append(f"sdist is missing: {sorted(missing)}")
        sbom_member = stripped.get("sbom.spdx.json")
        if sbom_member:
            stream = archive.extractfile(sbom_member)
            if stream and stream.read() != root_sbom:
                issues.append("sdist SBOM does not exactly match root sbom.spdx.json")
    return issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dist", type=Path)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    root_sbom = (root / "sbom.spdx.json").read_bytes()
    wheels = sorted(args.dist.glob("hlsgraph-0.1.1-*.whl"))
    sdists = sorted(args.dist.glob("hlsgraph-0.1.1.tar.gz"))
    issues = _audit_source_tree(root) + _audit_sbom(root_sbom, root)
    if len(wheels) != 1:
        issues.append(f"expected one v0.1.1 wheel, found {len(wheels)}")
    else:
        issues.extend(_audit_wheel(wheels[0], root, root_sbom))
    if len(sdists) != 1:
        issues.append(f"expected one v0.1.1 sdist, found {len(sdists)}")
    else:
        issues.extend(_audit_sdist(sdists[0], root_sbom))
    if issues:
        for issue in issues:
            print(f"ERROR: {issue}", file=sys.stderr)
        return 1
    print("release source and archives passed privacy, RECORD, and SPDX checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
