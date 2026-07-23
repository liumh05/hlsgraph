from __future__ import annotations

import re
from pathlib import Path

import pytest

from hlsgraph.mcp.server import LEGACY_TOOL_NAMES


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


def test_current_runner_docs_do_not_restore_v01_ssh_assumptions() -> None:
    architecture = (ROOT / "docs" / "architecture.md").read_text(encoding="utf-8")
    interfaces = (ROOT / "docs" / "interfaces.md").read_text(encoding="utf-8")
    normalized_interfaces = re.sub(r"\s+", " ", interfaces)
    assert "SSH quotes one complete `bash -lc` command" not in architecture
    assert (
        "remote project files must already be synchronized"
        not in normalized_interfaces
    )
    assert (
        "transfers them explicitly into one unique remote run directory"
        in normalized_interfaces
    )


def test_documented_legacy_mcp_names_match_the_registration_contract() -> None:
    interfaces = (ROOT / "docs" / "interfaces.md").read_text(encoding="utf-8")
    marker = "its exact tool names are "
    start = interfaces.index(marker) + len(marker)
    paragraph = interfaces[start:interfaces.index(". The default", start)]
    documented = tuple(re.findall(r"`([a-z_]+)`", paragraph))
    assert documented == LEGACY_TOOL_NAMES


def test_v5_runbook_uses_importable_module_entry_points() -> None:
    runbook = (ROOT / "docs" / "knowledge-review-runbook.md").read_text(
        encoding="utf-8",
    )
    executor = (ROOT / "tools" / "execute_knowledge_review_suite.py").read_text(
        encoding="utf-8",
    )
    assert "python3 -m tools.execute_knowledge_review_suite" in runbook
    assert "python3 -m tools.apply_knowledge_review_suite_attestation" in runbook
    assert "python3 tools/execute_knowledge_review_suite.py" not in runbook
    assert "python3 tools/apply_knowledge_review_suite_attestation.py" not in runbook
    assert "codex-resources/bwrap" in runbook
    assert "77360cb751ccedc5971391444ac86a8a33c15b04d6b4a6fe45f5d25496e62c4c" in runbook
    assert "$CODEX_RUNTIME/codex-resources:/usr/bin:/bin" in runbook
    assert "TIKTOKEN_CACHE_DIR" in runbook
    assert "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d" in runbook
    assert "fb374d419588a4632f3f557e76b4b70aebbca790" in runbook
    assert "exactly that one direct child" not in runbook
    assert "single-file Codex runtime" not in runbook
    assert "single-file Codex runtime" not in executor
    assert "For every returned Codex process" in runbook
    assert "post-process contract failures remain diagnosable" in runbook
    assert "code_mode_host" in runbook
    assert "/bin/bash -lc 'head -n 100000000 PATH'" in runbook
    assert "its work root must never be reused" in runbook


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
