# HLSGraph 0.3.0

Technical Preview.

This release adds a unified deterministic GraphRAG surface for HLS design
facts, tool evidence, versioned public guidance, and authorized local knowledge.
The default MCP surface exposes one read-only `explore` tool; the same retrieval
semantics are available through Python, CLI, and REST.

It also tightens external Vitis HLS directive parsing and scope binding,
preserves knowledge activation as a separate plane from design truth,
and keeps predictions excluded by default.

Complete canonical static aggregates are now accepted only through the
authorized indexing path, with recomputation receipts and immutable ledger
commit receipts. Unsupported, ambiguous, or incomplete MLIR/LLVM/source
domains remain explicitly masked instead of being promoted to complete facts.

The planned 192-run Agent A/B evaluation is deferred. This preview makes no
comparative performance-advantage claim.

The bundled knowledge packs remain explicitly `unreviewed` in this Technical
Preview. The complete six-invocation formal review suite was not attested, so
this preview makes no machine-review, human-review, or cross-model-review claim.
