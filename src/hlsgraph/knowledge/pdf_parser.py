"""Optional, local-only PDF parser for the private knowledge sidecar.

The parser deliberately returns plain Markdown-like text with one heading per
PDF page.  The ordinary sidecar chunker therefore preserves a human-readable
page locator without adding PDF bytes or extracted text to the canonical
ledger.  ``pypdf`` is imported only when the explicit ``pdf`` plugin is
selected, so the base package keeps no PDF runtime dependency.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
from typing import Any, Mapping


_PARSER_CONTRACT = "hlsgraph.local_pdf.pypdf.page_markdown.v1"
_MARKDOWN_HEADING = re.compile(r"^(\s{0,3})(#{1,6})(\s+)")


def _boolean(value: bool | str, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.casefold() in {"true", "false"}:
        return value.casefold() == "true"
    raise ValueError(f"{field} must be true or false")


def _positive_int(value: int | str, *, field: str, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc
    if str(parsed) != str(value).strip() or not 1 <= parsed <= maximum:
        raise ValueError(f"{field} must be in 1..{maximum}")
    return parsed


class PdfKnowledgeParser:
    """Extract text from a user-owned PDF in a bounded worker process.

    This class implements ``hlsgraph.knowledge_parser.v1``.  It never receives
    the source path, never opens a network connection, and never writes a file.
    The sidecar host supplies already-verified bytes and independently enforces
    the parser timeout and maximum output size.
    """

    name = "hlsgraph.pdf.pypdf"

    def __init__(
        self,
        *,
        extraction_mode: str = "layout",
        max_pages: int | str = 4_096,
        strict: bool | str = True,
    ) -> None:
        if extraction_mode not in {"plain", "layout"}:
            raise ValueError("extraction_mode must be plain or layout")
        self.extraction_mode = extraction_mode
        self.max_pages = _positive_int(max_pages, field="max_pages", maximum=10_000)
        self.strict = _boolean(strict, field="strict")
        try:
            import pypdf
        except ImportError as exc:  # pragma: no cover - depends on optional install
            raise RuntimeError(
                "the pdf knowledge parser requires the hlsgraph[pdf] extra"
            ) from exc
        backend_version = str(getattr(pypdf, "__version__", "")).strip()
        if not backend_version:
            raise RuntimeError("pypdf does not expose a version identity")
        self.version = f"1+pypdf-{backend_version}"
        identity = {
            "contract": _PARSER_CONTRACT,
            "backend": "pypdf",
            "backend_version": backend_version,
            "extraction_mode": self.extraction_mode,
            "max_pages": self.max_pages,
            "strict": self.strict,
        }
        self.fingerprint = hashlib.sha256(json.dumps(
            identity, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")).hexdigest()

    @staticmethod
    def capabilities() -> dict[str, Any]:
        return {
            "protocol_version": "hlsgraph.knowledge_parser.v1",
            "local_only": True,
            "network_access": False,
            "media_types": ["application/pdf"],
            "page_anchors": True,
            "embedded_source": False,
        }

    def parse(self, data: bytes, metadata: Mapping[str, Any]) -> str | None:
        if not isinstance(data, bytes) or not data.startswith(b"%PDF-"):
            raise ValueError("PDF parser input lacks the PDF magic header")
        if not isinstance(metadata, Mapping):
            raise ValueError("PDF parser metadata must be an object")
        if metadata.get("media_type") != "application/pdf":
            raise ValueError("PDF parser requires media_type application/pdf")
        expected_size = metadata.get("size")
        if type(expected_size) is not int or expected_size != len(data):
            raise ValueError("PDF parser input size does not match metadata")
        expected_hash = metadata.get("sha256")
        actual_hash = hashlib.sha256(data).hexdigest()
        if expected_hash != actual_hash:
            raise ValueError("PDF parser input hash does not match metadata")

        import pypdf

        reader = pypdf.PdfReader(io.BytesIO(data), strict=self.strict)
        if bool(reader.is_encrypted):
            raise ValueError("encrypted PDFs are not supported")
        page_count = len(reader.pages)
        if page_count > self.max_pages:
            raise ValueError(
                f"PDF has {page_count} pages, exceeding max_pages={self.max_pages}"
            )

        sections: list[str] = []
        for number, page in enumerate(reader.pages, start=1):
            text = page.extract_text(extraction_mode=self.extraction_mode)
            if text is None:
                continue
            if not isinstance(text, str):
                raise ValueError("pypdf returned a non-text page")
            normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
            if not normalized:
                continue
            if "\x00" in normalized:
                raise ValueError("pypdf returned text containing NUL")
            # The sidecar chunker interprets Markdown headings as anchors.  A
            # literal heading-looking line in PDF text must not replace the
            # deterministic page anchor emitted by this parser.
            normalized = "\n".join(
                _MARKDOWN_HEADING.sub(r"\1\\\2\3", line)
                for line in normalized.split("\n")
            )
            sections.append(f"# PDF page {number}\n\n{normalized}\n")
        if not sections:
            return None
        return "\n".join(sections)


__all__ = ["PdfKnowledgeParser"]
