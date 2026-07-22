"""Test-only construction of review-ready public knowledge packs."""
from __future__ import annotations

import hashlib

from hlsgraph.knowledge import KnowledgeCatalog, load_pack
from hlsgraph.model import json_ready


def reviewed_pack(pack):
    """Attach deterministic test review evidence without changing semantics."""

    value = json_ready(pack)
    value["metadata"]["review_status"] = "machine_repeated_reviewed"
    value["coverage"].update({
        "review_status": "machine_repeated_reviewed",
        "reviewers": [
            "test.model@pinned#invocation-1",
            "test.model@pinned#invocation-2",
        ],
        "source_hashes": {
            f"{item.document_id}@{item.document_version}": hashlib.sha256(
                f"{item.document_id}@{item.document_version}".encode()
            ).hexdigest()
            for item in pack.documents
        },
        "review_evidence": {
            "independent_invocations": True,
            "citation_verified": True,
            "review_agreement": True,
            "unresolved_conflicts": False,
            "same_model_repeated_review": True,
            "distinct_model_families": False,
        },
    })
    # Review provenance participates in the immutable coverage identity.  The
    # test clone must therefore recompute that identity instead of retaining
    # the unreviewed pack's serialized ID.
    value["coverage"].pop("id", None)
    return load_pack(value)


def reviewed_builtin_catalog() -> KnowledgeCatalog:
    return KnowledgeCatalog([
        reviewed_pack(pack) if pack.bindings and not pack.review_ready else pack
        for pack in KnowledgeCatalog.builtin().packs
    ])


def install_reviewed_builtin_packs(bundle) -> None:
    reviewed_builtin_catalog().install(bundle.store)
