"""Versioned, citation-only knowledge packs for HLSGraph.

Knowledge packs contain project-authored rules and references.  They are not
design observations and never contain redistributed vendor documentation.
"""

from .core import (
    LEGACY_PACK_SCHEMA_VERSIONS,
    LOCAL_INDEX_SCHEMA_VERSION,
    PACK_SCHEMA_VERSION,
    DocumentReference,
    KnowledgeCatalog,
    KnowledgePack,
    KnowledgePackError,
    LocalDocumentMetadata,
    binding_entails_rule_condition,
    canonical_context_scalar,
    filter_rules,
    index_local_document,
    load_builtin_packs,
    load_local_index,
    load_pack,
    matches_applicability,
    matches_binding,
    migrate_pack,
    pack_migration_plan,
    save_local_index,
    target_derived_condition_source,
)
from .sidecar import (
    DEFAULT_MAX_PARSED_CHARS,
    DEFAULT_PARSER_TIMEOUT_S,
    LocalKnowledgeHit,
    LocalKnowledgeSidecar,
    SIDECAR_RELATIVE_ROOT,
    SIDECAR_SCHEMA_VERSION,
)

__all__ = [
    "LOCAL_INDEX_SCHEMA_VERSION",
    "LEGACY_PACK_SCHEMA_VERSIONS",
    "PACK_SCHEMA_VERSION",
    "DocumentReference",
    "DEFAULT_MAX_PARSED_CHARS",
    "DEFAULT_PARSER_TIMEOUT_S",
    "KnowledgeCatalog",
    "KnowledgePack",
    "KnowledgePackError",
    "LocalDocumentMetadata",
    "LocalKnowledgeHit",
    "LocalKnowledgeSidecar",
    "SIDECAR_RELATIVE_ROOT",
    "SIDECAR_SCHEMA_VERSION",
    "binding_entails_rule_condition",
    "canonical_context_scalar",
    "filter_rules",
    "index_local_document",
    "load_builtin_packs",
    "load_local_index",
    "load_pack",
    "matches_applicability",
    "matches_binding",
    "migrate_pack",
    "pack_migration_plan",
    "save_local_index",
    "target_derived_condition_source",
]
