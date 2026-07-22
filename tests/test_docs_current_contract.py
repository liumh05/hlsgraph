from __future__ import annotations

import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_V2_DOCS = (
    ROOT / "docs" / "architecture.md",
    ROOT / "docs" / "interfaces.md",
)


@pytest.mark.parametrize("path", RUNNER_V2_DOCS, ids=lambda path: path.name)
def test_current_runner_docs_preserve_v2_evidence_transport(path: Path) -> None:
    text = re.sub(r"\s+", " ", path.read_text(encoding="utf-8").casefold())
    required_concepts = {
        "runner v2 protocol": ("runner v2", "hlsgraph.runner.v2"),
        "one non-reusable remote run directory": ("non-reusable remote run directory",),
        "frozen declared outputs": ("freezes the declared outputs", "freezes and hashes declared outputs"),
        "remote sha256 manifest": ("remote size/sha-256 manifest", "hashes declared outputs remotely"),
        "direct sdk staging transfer": ("directly into sdk-owned staging",),
        "local second validation": ("sdk then revalidates", "sdk owns a restricted"),
        "atomic ledger commit": ("atomically committing", "atomically commit"),
    }
    for label, alternatives in required_concepts.items():
        assert any(value in text for value in alternatives), f"{path.name} lost {label}"

    assert "the v0.1 ssh runner" not in text
    assert "does not return independently transferred" not in text


def test_current_docs_never_present_async_sync_as_evidence_transport() -> None:
    for path in (ROOT / "README.md", *(ROOT / "docs").glob("*.md")):
        text = path.read_text(encoding="utf-8")
        paragraphs = re.split(r"\n\s*\n", text)
        for paragraph in paragraphs:
            if re.search(r"\b(?:Mutagen|rsync)\b", paragraph, re.IGNORECASE):
                assert re.search(
                    r"\b(?:never|not|cannot|must not)\b", paragraph, re.IGNORECASE,
                ), f"{path.name} presents asynchronous sync without an evidence prohibition"


def test_independent_protocol_versions_are_not_package_staleness() -> None:
    architecture = (ROOT / "docs" / "architecture.md").read_text(encoding="utf-8")
    interfaces = (ROOT / "docs" / "interfaces.md").read_text(encoding="utf-8")
    assert "`hlsgraph.extractors.v1`" in architecture
    assert "`hlsgraph.runners.v2`" in interfaces
    assert "/api/v1" in interfaces


@pytest.mark.parametrize("relative", [
    "docs/schema.md",
    "docs/architecture.md",
    "docs/interfaces.md",
    "docs/privacy-and-security.md",
])
def test_current_docs_explain_single_report_observation_provenance(relative: str) -> None:
    text = re.sub(
        r"\s+", " ", (ROOT / relative).read_text(encoding="utf-8").casefold(),
    )
    assert "observationsource" in text
    assert "parser" in text
    assert "receipt" in text
    assert "not a signature" in text or "rather than a signature" in text
