from __future__ import annotations

import hashlib
import importlib.util
import io
from pathlib import Path
import stat
import sys
import tarfile
import types
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

DIST_INFO = "hlsgraph-0.1.0.dist-info"


def _load_build_backend(monkeypatch):
    """Load the local backend without requiring build-only setuptools in pytest."""
    setuptools = types.ModuleType("setuptools")
    build_meta = types.ModuleType("setuptools.build_meta")

    def unused_hook(*_args, **_kwargs):  # pragma: no cover - import shim only
        raise AssertionError("setuptools hooks are not used by this unit test")

    for name in (
        "build_editable", "build_sdist", "build_wheel",
        "get_requires_for_build_editable", "get_requires_for_build_sdist",
        "get_requires_for_build_wheel", "prepare_metadata_for_build_editable",
        "prepare_metadata_for_build_wheel",
    ):
        setattr(build_meta, name, unused_hook)
    setuptools.build_meta = build_meta
    monkeypatch.setitem(sys.modules, "setuptools", setuptools)
    monkeypatch.setitem(sys.modules, "setuptools.build_meta", build_meta)

    path = Path(__file__).resolve().parents[1] / "build_backend.py"
    spec = importlib.util.spec_from_file_location("_hlsgraph_test_build_backend", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record_bytes(names: list[str]) -> bytes:
    rows = [f"{name},," for name in names]
    rows.append(f"{DIST_INFO}/RECORD,,")
    return ("\n".join(rows) + "\n").encode("utf-8")


def _write_input_wheel(path: Path, *, timestamp: tuple[int, ...], reverse: bool) -> None:
    payloads = {
        "hlsgraph/__init__.py": b'__version__ = "0.1.0"\n',
        f"{DIST_INFO}/METADATA": b"Metadata-Version: 2.4\nName: hlsgraph\nVersion: 0.1.0\n",
        f"{DIST_INFO}/WHEEL": b"Wheel-Version: 1.0\nTag: py3-none-any\n",
    }
    payloads[f"{DIST_INFO}/RECORD"] = _record_bytes(list(payloads))
    items = list(payloads.items())
    if reverse:
        items.reverse()
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        for name, data in items:
            info = ZipInfo(name, timestamp)
            info.compress_type = ZIP_DEFLATED
            info.create_system = 3
            mode = 0o755 if name == "hlsgraph/__init__.py" else 0o600
            info.external_attr = (stat.S_IFREG | mode) << 16
            archive.writestr(info, data)


def _write_input_sdist(path: Path, *, timestamp: int, reverse: bool) -> None:
    entries = [
        ("hlsgraph-0.1.0", None, 0o777),
        ("hlsgraph-0.1.0/README.md", b"HLSGraph\n", 0o600),
        ("hlsgraph-0.1.0/tools/check.sh", b"#!/bin/sh\n", 0o700),
    ]
    if reverse:
        entries.reverse()
    with tarfile.open(path, "w:gz", format=tarfile.PAX_FORMAT) as archive:
        for name, payload, mode in entries:
            info = tarfile.TarInfo(name)
            info.mtime = timestamp
            info.mode = mode
            info.uid = timestamp % 1000
            info.gid = timestamp % 997
            info.uname = "builder"
            info.gname = "builder"
            if payload is None:
                info.type = tarfile.DIRTYPE
                archive.addfile(info)
            else:
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))


def test_sbom_injection_is_reproducible_and_zip_metadata_is_normalized(
    tmp_path, monkeypatch,
):
    build_backend = _load_build_backend(monkeypatch)
    first = tmp_path / "first.whl"
    second = tmp_path / "second.whl"
    _write_input_wheel(first, timestamp=(2025, 1, 2, 3, 4, 4), reverse=False)
    _write_input_wheel(second, timestamp=(2026, 6, 7, 8, 9, 10), reverse=True)

    # A pre-ZIP epoch must be clamped to 1980-01-01 rather than leaking either
    # input build time into the rewritten archive.
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "0")
    build_backend._inject_sbom(first)
    build_backend._inject_sbom(second)

    assert hashlib.sha256(first.read_bytes()).digest() == hashlib.sha256(
        second.read_bytes()
    ).digest()
    with ZipFile(first) as archive:
        infos = archive.infolist()
        assert infos
        assert {item.date_time for item in infos} == {(1980, 1, 1, 0, 0, 0)}
        assert {item.create_system for item in infos} == {3}
        assert all(item.extra == b"" and item.comment == b"" for item in infos)
        modes = {
            item.filename: (item.external_attr >> 16) & 0xFFFF for item in infos
        }
        assert stat.S_IMODE(modes["hlsgraph/__init__.py"]) == 0o755
        assert all(
            stat.S_IMODE(mode) == 0o644
            for name, mode in modes.items() if name != "hlsgraph/__init__.py"
        )

    # Re-injection with the same epoch must itself be byte-for-byte idempotent.
    before = hashlib.sha256(first.read_bytes()).digest()
    build_backend._inject_sbom(first)
    assert hashlib.sha256(first.read_bytes()).digest() == before


def test_sdist_normalization_is_reproducible_and_idempotent(tmp_path, monkeypatch):
    build_backend = _load_build_backend(monkeypatch)
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"
    _write_input_sdist(first, timestamp=1_700_000_001, reverse=False)
    _write_input_sdist(second, timestamp=1_800_000_002, reverse=True)

    monkeypatch.setenv("SOURCE_DATE_EPOCH", "315532800")
    build_backend._normalize_sdist(first)
    build_backend._normalize_sdist(second)

    assert hashlib.sha256(first.read_bytes()).digest() == hashlib.sha256(
        second.read_bytes()
    ).digest()
    with tarfile.open(first, "r:gz") as archive:
        members = archive.getmembers()
        assert [item.name for item in members] == sorted(item.name for item in members)
        assert {item.mtime for item in members} == {315532800}
        assert {item.uid for item in members} == {0}
        assert {item.gid for item in members} == {0}
        assert {item.uname for item in members} == {""}
        assert {item.gname for item in members} == {""}
        modes = {item.name: item.mode for item in members}
        assert modes["hlsgraph-0.1.0"] == 0o755
        assert modes["hlsgraph-0.1.0/README.md"] == 0o644
        assert modes["hlsgraph-0.1.0/tools/check.sh"] == 0o755

    before = hashlib.sha256(first.read_bytes()).digest()
    build_backend._normalize_sdist(first)
    assert hashlib.sha256(first.read_bytes()).digest() == before
