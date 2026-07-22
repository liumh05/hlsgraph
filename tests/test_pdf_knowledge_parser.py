from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest

from hlsgraph.knowledge import sidecar
from hlsgraph.knowledge.core import KnowledgePackError, LocalDocumentMetadata
from hlsgraph.knowledge.pdf_parser import PdfKnowledgeParser
from hlsgraph.knowledge.sidecar import _parse_with_plugin


class _LargeLocalParser:
    """Spawn-picklable regression fixture larger than an OS pipe buffer."""

    @staticmethod
    def parse(data, metadata):
        return "x" * (1024 * 1024)


_PRIVATE_STDIO_SENTINEL = "PRIVATE_PDF_STDIO_SENTINEL_6e7d61"


class _NoisyLocalParser:
    """Spawn-picklable parser that bypasses Python stream redirectors."""

    @staticmethod
    def parse(data, metadata):
        os.write(1, _PRIVATE_STDIO_SENTINEL.encode("ascii"))
        os.write(2, _PRIVATE_STDIO_SENTINEL.encode("ascii"))
        return "bounded result"


class _Page:
    def __init__(self, text):
        self.text = text
        self.modes = []

    def extract_text(self, *, extraction_mode):
        self.modes.append(extraction_mode)
        return self.text


def _backend(monkeypatch, pages, *, encrypted=False):
    class _Reader:
        def __init__(self, stream, *, strict):
            assert stream.read(5) == b"%PDF-"
            self.pages = pages
            self.is_encrypted = encrypted
            self.strict = strict

    module = SimpleNamespace(__version__="6.10.0", PdfReader=_Reader)
    monkeypatch.setitem(sys.modules, "pypdf", module)
    return module


def _metadata(data: bytes) -> dict:
    return {
        "media_type": "application/pdf",
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def test_pdf_parser_emits_stable_page_headings_and_identity(monkeypatch) -> None:
    pages = [
        _Page("Introduction\r\nII evidence\n# literal PDF text"),
        _Page("  "),
        _Page("Timing\nWNS"),
    ]
    _backend(monkeypatch, pages)
    data = b"%PDF-1.7\nsynthetic"
    parser = PdfKnowledgeParser(
        extraction_mode="layout", max_pages="3", strict="false",
    )

    assert parser.parse(data, _metadata(data)) == (
        "# PDF page 1\n\nIntroduction\nII evidence\n\\# literal PDF text\n\n"
        "# PDF page 3\n\nTiming\nWNS\n"
    )
    assert pages[0].modes == ["layout"]
    assert pages[1].modes == ["layout"]
    assert pages[2].modes == ["layout"]
    assert parser.version == "1+pypdf-6.10.0"
    assert len(parser.fingerprint) == 64
    assert parser.fingerprint == PdfKnowledgeParser(
        extraction_mode="layout", max_pages=3, strict=False,
    ).fingerprint
    assert parser.fingerprint != PdfKnowledgeParser(
        extraction_mode="plain", max_pages=3, strict=False,
    ).fingerprint
    assert parser.capabilities() == {
        "protocol_version": "hlsgraph.knowledge_parser.v1",
        "local_only": True,
        "network_access": False,
        "media_types": ["application/pdf"],
        "page_anchors": True,
        "embedded_source": False,
    }


@pytest.mark.parametrize(
    ("metadata_change", "message"),
    [
        ({"size": 1}, "size does not match"),
        ({"sha256": "0" * 64}, "hash does not match"),
        ({"media_type": "text/plain"}, "requires media_type"),
    ],
)
def test_pdf_parser_revalidates_supplied_bytes(
    monkeypatch, metadata_change, message,
) -> None:
    _backend(monkeypatch, [_Page("text")])
    data = b"%PDF-1.7\nsynthetic"
    metadata = {**_metadata(data), **metadata_change}
    with pytest.raises(ValueError, match=message):
        PdfKnowledgeParser().parse(data, metadata)


def test_pdf_parser_fails_closed_for_encryption_and_page_limit(monkeypatch) -> None:
    data = b"%PDF-1.7\nsynthetic"
    _backend(monkeypatch, [_Page("text")], encrypted=True)
    with pytest.raises(ValueError, match="encrypted PDFs"):
        PdfKnowledgeParser().parse(data, _metadata(data))

    _backend(monkeypatch, [_Page("one"), _Page("two")])
    with pytest.raises(ValueError, match="exceeding max_pages=1"):
        PdfKnowledgeParser(max_pages=1).parse(data, _metadata(data))


def test_pdf_parser_returns_metadata_only_when_no_text_exists(monkeypatch) -> None:
    _backend(monkeypatch, [_Page(None), _Page(" \n")])
    data = b"%PDF-1.7\nsynthetic"
    assert PdfKnowledgeParser().parse(data, _metadata(data)) is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"extraction_mode": "ocr"}, "plain or layout"),
        ({"max_pages": 0}, "max_pages must be"),
        ({"max_pages": "01"}, "max_pages must be"),
        ({"strict": "yes"}, "strict must be"),
    ],
)
def test_pdf_parser_configuration_is_closed(monkeypatch, kwargs, message) -> None:
    _backend(monkeypatch, [])
    with pytest.raises(ValueError, match=message):
        PdfKnowledgeParser(**kwargs)


def test_isolated_parser_drains_large_bounded_output_before_join() -> None:
    data = b"%PDF-1.7\nsynthetic"
    metadata = LocalDocumentMetadata(
        document_id="test.pdf.large",
        document_version="1",
        uri="file:///unused.pdf",
        sha256=hashlib.sha256(data).hexdigest(),
        size=len(data),
        modified_ns=0,
        indexed_at="2026-01-01T00:00:00Z",
        media_type="application/pdf",
    )
    result = _parse_with_plugin(
        _LargeLocalParser(), data, metadata, timeout_s=10.0,
        max_chars=2 * 1024 * 1024,
    )
    assert result is not None
    assert len(result) == 1024 * 1024


def _local_metadata(data: bytes) -> LocalDocumentMetadata:
    return LocalDocumentMetadata(
        document_id="test.pdf.private_stdio",
        document_version="1",
        uri="file:///unused.pdf",
        sha256=hashlib.sha256(data).hexdigest(),
        size=len(data),
        modified_ns=0,
        indexed_at="2026-01-01T00:00:00Z",
        media_type="application/pdf",
    )


def test_parser_os_write_is_suppressed_from_pytest_capture(capfd) -> None:
    data = b"%PDF-1.7\nprivate synthetic bytes"
    assert _parse_with_plugin(
        _NoisyLocalParser(), data, _local_metadata(data),
        timeout_s=10.0, max_chars=1024,
    ) == "bounded result"

    captured = capfd.readouterr()
    assert _PRIVATE_STDIO_SENTINEL not in captured.out
    assert _PRIVATE_STDIO_SENTINEL not in captured.err


def test_parser_os_write_is_suppressed_from_real_subprocess(tmp_path: Path) -> None:
    script = tmp_path / "stdio_probe.py"
    script.write_text(textwrap.dedent(f"""
        import hashlib
        import os

        from hlsgraph.knowledge.core import LocalDocumentMetadata
        from hlsgraph.knowledge.sidecar import _parse_with_plugin

        SENTINEL = {_PRIVATE_STDIO_SENTINEL!r}

        class NoisyParser:
            @staticmethod
            def parse(data, metadata):
                os.write(1, SENTINEL.encode("ascii"))
                os.write(2, SENTINEL.encode("ascii"))
                return "bounded result"

        if __name__ == "__main__":
            data = b"%PDF-1.7\\nprivate synthetic bytes"
            metadata = LocalDocumentMetadata(
                document_id="test.pdf.subprocess",
                document_version="1",
                uri="file:///unused.pdf",
                sha256=hashlib.sha256(data).hexdigest(),
                size=len(data),
                modified_ns=0,
                indexed_at="2026-01-01T00:00:00Z",
                media_type="application/pdf",
            )
            result = _parse_with_plugin(
                NoisyParser(), data, metadata,
                timeout_s=10.0, max_chars=1024,
            )
            assert result == "bounded result"
            print("SAFE_RESULT")
    """), encoding="utf-8", newline="\n")
    source_root = Path(__file__).resolve().parents[1] / "src"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, (
        str(source_root), environment.get("PYTHONPATH", ""),
    )))

    completed = subprocess.run(
        [sys.executable, str(script)], cwd=tmp_path, env=environment,
        capture_output=True, text=True, timeout=30, check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "SAFE_RESULT"
    assert completed.stderr == ""
    assert _PRIVATE_STDIO_SENTINEL not in completed.stdout
    assert _PRIVATE_STDIO_SENTINEL not in completed.stderr


class _BrokenQueue:
    def __init__(self, error_type):
        self.error_type = error_type

    def get(self, *, timeout):
        raise self.error_type("private queue implementation detail")

    def cancel_join_thread(self):
        return None

    def close(self):
        return None


class _FakeProcess:
    def __init__(self):
        self.alive = False

    def start(self):
        self.alive = True

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.alive = False

    def join(self, timeout=None):
        return None

    def kill(self):
        self.alive = False


class _IgnoringTerminateProcess(_FakeProcess):
    """Portable stand-in for a worker that ignores terminate/SIGTERM."""

    def __init__(self):
        super().__init__()
        self.terminate_calls = 0
        self.kill_calls = 0
        self.join_calls = 0

    def terminate(self):
        self.terminate_calls += 1

    def join(self, timeout=None):
        self.join_calls += 1

    def kill(self):
        self.kill_calls += 1
        self.alive = False


class _BrokenContext:
    def __init__(self, error_type):
        self.queue = _BrokenQueue(error_type)
        self.process = _FakeProcess()

    def Queue(self, *, maxsize):
        return self.queue

    def Process(self, **kwargs):
        return self.process


@pytest.mark.parametrize("error_type", [EOFError, OSError, ValueError])
def test_parser_queue_read_failures_are_sanitized(
    monkeypatch, error_type,
) -> None:
    context = _BrokenContext(error_type)
    monkeypatch.setattr(sidecar.multiprocessing, "get_context", lambda method: context)
    data = b"%PDF-1.7\nprivate synthetic bytes"

    with pytest.raises(
        KnowledgePackError, match="knowledge parser result channel failed",
    ) as caught:
        _parse_with_plugin(
            _NoisyLocalParser(), data, _local_metadata(data),
            timeout_s=10.0, max_chars=1024,
        )

    assert caught.value.__cause__ is None


def test_parser_finally_kills_worker_that_ignores_terminate(monkeypatch) -> None:
    context = _BrokenContext(EOFError)
    process = _IgnoringTerminateProcess()
    context.process = process
    monkeypatch.setattr(sidecar.multiprocessing, "get_context", lambda method: context)
    data = b"%PDF-1.7\nprivate synthetic bytes"

    with pytest.raises(
        KnowledgePackError, match="knowledge parser result channel failed",
    ):
        _parse_with_plugin(
            _NoisyLocalParser(), data, _local_metadata(data),
            timeout_s=10.0, max_chars=1024,
        )

    assert process.terminate_calls == 1
    assert process.kill_calls == 1
    assert process.join_calls == 2
    assert process.is_alive() is False
