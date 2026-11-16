# HLSGraph 0.3.1 — Technical Preview

Repeated diagnostics retain distinct source locations and stable fact IDs across retries; compile errors fail indexing and variants cleanly.

This additive release makes source argument bindings and DATAFLOW relationships
queryable. It links caller operands to helper parameters, including parameters
with ARRAY_PARTITION directives; records direct source call processes and local
stream/array channels; and summarizes stream accesses within functions and loops.

All new facts use the `source.*` namespace and the same AST-stage static-evidence
labels as software calls, including `hardware_instance=false`. Source evidence
is not hardware topology. Vitis may restructure non-canonical DATAFLOW regions.
Unknown operands, directions and depths remain explicitly unresolved.

Non-fatal diagnostics identify non-canonical DATAFLOW statements and arguments,
multiple channel endpoints, DATAFLOW/PIPELINE conflicts, and source constructs
requiring synthesis review. Synthesis diagnostics exclude testbench files and
`main`, and respect a libclang preprocessing view with `__SYNTHESIS__` defined.
MCP health, evidence exploration/traversal and ML export expose these additions.

The package version is 0.3.1. Canonical schema, bundle format and ML feature
schema remain 0.3.0. No stored-data migration is needed. Source extractor version
5 changes the extraction profile and therefore newly indexed snapshot identities;
historical graph identities and stored records retain their original meaning.

This is a Technical Preview. It makes no performance-advantage claims and does
not provide vendor synthesis validation. Release archive and knowledge-review
gates remain applicable; see the [release runbook](knowledge-review-runbook.md).
The bundled knowledge packs remain `unreviewed`; this update does not supply
the external formal knowledge-review evidence required for release approval.
