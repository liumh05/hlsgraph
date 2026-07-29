# HLSGraph 0.4.0

Technical Preview; no comparative performance advantage is claimed.

This release moves the optional stdio MCP integration to Python SDK 2.x. The
default surface remains the single read-only `explore` tool, the 16 narrow
tools remain explicit compatibility opt-ins, and current and legacy MCP clients
are both supported. `explore` now publishes structured output while preserving
the existing bounded compact context.

The canonical schema, bundle format, ML feature schema, and retrieval profile
remain at `0.3.0`; no stored-data migration is required.

The planned Agent A/B evaluation remains deferred, and this preview makes no
comparative performance claim. The bundled knowledge packs remain explicitly
`unreviewed`; this MCP-only release makes no new review claim.
