# HLSGraph 0.3.0

Technical Preview.

This release adds a unified deterministic GraphRAG surface for HLS design
facts, tool evidence, reviewed public guidance, and authorized local knowledge.
The default MCP surface exposes one read-only `explore` tool; the same retrieval
semantics are available through Python, CLI, and REST.

It also tightens external Vitis HLS directive parsing and scope binding,
preserves reviewed knowledge activation as a separate plane from design truth,
and keeps predictions excluded by default.

The planned 192-run Agent A/B evaluation is deferred. This preview makes no
comparative performance-advantage claim.
