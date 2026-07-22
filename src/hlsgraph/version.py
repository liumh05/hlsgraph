__version__ = "0.3.0"
SCHEMA_VERSION = "0.3.0"
BUNDLE_VERSION = "0.3.0"
FEATURE_SCHEMA_VERSION = "0.3.0"
RETRIEVAL_PROFILE_SCHEMA_VERSION = "0.3.0"

# A v0.3 ledger may retain immutable v0.2 graph projections.  Their serialized
# schema marker is part of the graph hash, so rewriting it during the additive
# migration would silently change historical identity.  New graphs always use
# ``SCHEMA_VERSION``; the older marker is read-only compatibility.
SUPPORTED_GRAPH_SCHEMA_VERSIONS = frozenset({"0.2.0", SCHEMA_VERSION})
