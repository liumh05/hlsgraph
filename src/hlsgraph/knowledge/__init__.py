"""Versioned, citation-only knowledge packs for HLSGraph.

Knowledge packs contain project-authored rules and references.  They are not
design observations and never contain redistributed vendor documentation.
"""

from .core import (
    LOCAL_INDEX_SCHEMA_VERSION,
    PACK_SCHEMA_VERSION,
    DocumentReference,
    KnowledgeCatalog,
    KnowledgePack,
    KnowledgePackError,
    LocalDocumentMetadata,
    filter_rules,
    index_local_document,
    load_builtin_packs,
    load_local_index,
    load_pack,
    matches_applicability,
    save_local_index,
)

__all__ = [
    "LOCAL_INDEX_SCHEMA_VERSION",
    "PACK_SCHEMA_VERSION",
    "DocumentReference",
    "KnowledgeCatalog",
    "KnowledgePack",
    "KnowledgePackError",
    "LocalDocumentMetadata",
    "filter_rules",
    "index_local_document",
    "load_builtin_packs",
    "load_local_index",
    "load_pack",
    "matches_applicability",
    "save_local_index",
]
