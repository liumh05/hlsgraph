"""Thin setuptools backend that places the checked SBOM in wheel metadata.

PEP 770 reserves ``.dist-info/sboms`` but intentionally does not yet define a
static ``pyproject.toml`` field.  Keeping this small wrapper in-tree makes the
injection explicit and ensures RECORD covers the injected file.  When
``SOURCE_DATE_EPOCH`` is set, both release archives use that timestamp and
normalized permissions for reproducible output (with ZIP's 1980 lower bound).
"""
from __future__ import annotations

import base64
import copy
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import io
import os
from pathlib import Path
import stat
import tarfile
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from setuptools.build_meta import (  # noqa: F401 - PEP 517 hook re-exports
    build_editable,
    get_requires_for_build_editable,
    get_requires_for_build_sdist,
    get_requires_for_build_wheel,
    prepare_metadata_for_build_editable,
    prepare_metadata_for_build_wheel,
)
from setuptools.build_meta import build_sdist as _setuptools_build_sdist
from setuptools.build_meta import build_wheel as _setuptools_build_wheel


_ZIP_MIN_EPOCH = 315532800  # 1980-01-01T00:00:00Z
_ZIP_MAX_EPOCH = 4354819198  # 2107-12-31T23:59:58Z
_GZIP_MAX_EPOCH = (1 << 32) - 1


def _record_digest(data: bytes) -> str:
    encoded = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
    return "sha256=" + encoded.decode("ascii")


def _source_date_epoch() -> int | None:
    """Return SOURCE_DATE_EPOCH as a gzip-safe timestamp when configured."""
    raw = os.environ.get("SOURCE_DATE_EPOCH")
    if raw is None:
        return None
    try:
        epoch = int(raw, 10)
    except ValueError as exc:
        raise RuntimeError("SOURCE_DATE_EPOCH must be an integer Unix timestamp") from exc
    return min(max(epoch, 0), _GZIP_MAX_EPOCH)


def _source_date_time() -> tuple[int, int, int, int, int, int] | None:
    """Return SOURCE_DATE_EPOCH as a ZIP-safe UTC timestamp when configured."""
    source_epoch = _source_date_epoch()
    if source_epoch is None:
        return None
    epoch = min(max(source_epoch, _ZIP_MIN_EPOCH), _ZIP_MAX_EPOCH)
    value = datetime.fromtimestamp(epoch, timezone.utc)
    # DOS timestamps stored by ZIP have a two-second resolution.  Floor the
    # value explicitly so reading the archive cannot produce a different time.
    return (value.year, value.month, value.day, value.hour, value.minute,
            value.second - value.second % 2)


def _normalized_info(name: str, original: ZipInfo | None,
                     timestamp: tuple[int, int, int, int, int, int]) -> ZipInfo:
    """Create deterministic wheel metadata while retaining executable intent."""
    is_directory = name.endswith("/")
    executable = False
    if original is not None and original.create_system == 3:
        original_mode = (original.external_attr >> 16) & 0xFFFF
        executable = stat.S_ISREG(original_mode) and bool(original_mode & 0o111)

    info = ZipInfo(name, timestamp)
    info.compress_type = ZIP_DEFLATED
    info.create_system = 3
    info.internal_attr = 0
    if is_directory:
        info.external_attr = (stat.S_IFDIR | 0o755) << 16 | 0x10
    else:
        mode = 0o755 if executable else 0o644
        info.external_attr = (stat.S_IFREG | mode) << 16
    return info


def _inject_sbom(wheel_path: Path) -> None:
    sbom = Path(__file__).with_name("sbom.spdx.json").read_bytes()
    with ZipFile(wheel_path, "r") as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise RuntimeError("wheel must not contain duplicate member names")
        dist_info = sorted({name.split("/", 1)[0] for name in names
                            if name.endswith(".dist-info/METADATA")})
        if len(dist_info) != 1:
            raise RuntimeError("wheel must contain exactly one .dist-info directory")
        root = dist_info[0]
        record_name = f"{root}/RECORD"
        sbom_name = f"{root}/sboms/sbom.spdx.json"
        infos = {name: archive.getinfo(name) for name in names}
        payloads = {name: archive.read(name) for name in names
                    if name not in {record_name, sbom_name}}
        previous_record = archive.read(record_name).decode("utf-8")

    rows = [row for row in csv.reader(io.StringIO(previous_record))
            if row and row[0] not in {record_name, sbom_name}]
    rows.append([sbom_name, _record_digest(sbom), str(len(sbom))])
    rows.append([record_name, "", ""])
    rows.sort(key=lambda row: (row[0] == record_name, row[0]))
    stream = io.StringIO(newline="")
    csv.writer(stream, lineterminator="\n").writerows(rows)
    payloads[sbom_name] = sbom
    payloads[record_name] = stream.getvalue().encode("utf-8")

    configured_timestamp = _source_date_time()
    record_timestamp = infos[record_name].date_time

    temporary = wheel_path.with_suffix(wheel_path.suffix + ".tmp")
    with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as archive:
        for name in sorted(payloads, key=lambda item: (item.startswith(root + "/"), item)):
            original = infos.get(name)
            timestamp = configured_timestamp or (
                original.date_time if original is not None else record_timestamp
            )
            archive.writestr(
                _normalized_info(name, original, timestamp), payloads[name]
            )
    temporary.replace(wheel_path)


def _normalize_sdist(sdist_path: Path) -> None:
    """Normalize a setuptools tar.gz sdist when SOURCE_DATE_EPOCH is set."""
    epoch = _source_date_epoch()
    if epoch is None:
        return

    temporary = sdist_path.with_suffix(sdist_path.suffix + ".tmp")
    with tarfile.open(sdist_path, "r:gz") as source:
        members = source.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            raise RuntimeError("sdist must not contain duplicate member names")
        with temporary.open("wb") as raw:
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=raw, mtime=epoch,
            ) as compressed:
                with tarfile.open(
                    fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT,
                ) as target:
                    for member in sorted(members, key=lambda item: item.name):
                        normalized = copy.copy(member)
                        normalized.mtime = epoch
                        normalized.uid = 0
                        normalized.gid = 0
                        normalized.uname = ""
                        normalized.gname = ""
                        normalized.pax_headers = {
                            key: value for key, value in member.pax_headers.items()
                            if key not in {"atime", "ctime", "mtime"}
                        }
                        if member.isdir():
                            normalized.mode = 0o755
                        elif member.isfile():
                            normalized.mode = 0o755 if member.mode & 0o111 else 0o644
                        elif member.issym() or member.islnk():
                            normalized.mode = 0o777
                        payload = source.extractfile(member) if member.isfile() else None
                        target.addfile(normalized, payload)
    temporary.replace(sdist_path)


def build_sdist(sdist_directory: str, config_settings=None) -> str:
    filename = _setuptools_build_sdist(
        sdist_directory, config_settings=config_settings,
    )
    _normalize_sdist(Path(sdist_directory) / filename)
    return filename


def build_wheel(wheel_directory: str, config_settings=None,
                metadata_directory: str | None = None) -> str:
    filename = _setuptools_build_wheel(
        wheel_directory, config_settings=config_settings,
        metadata_directory=metadata_directory,
    )
    _inject_sbom(Path(wheel_directory) / filename)
    return filename
