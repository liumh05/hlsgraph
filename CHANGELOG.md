# Changelog

All notable changes are recorded here. HLSGraph follows semantic versioning;
the schema and bundle have independent explicit versions.

## 0.3.0 — 2026-07-21

Developer-preview retrieval and knowledge update:

- add a deterministic hybrid retrieval contract over separate fact, evidence,
  knowledge, local-document, and opt-in prediction planes;
- combine exact/FTS/BM25/fuzzy candidates with typed, directed graph
  propagation and versioned reciprocal-rank fusion without creating graph
  facts or upgrading guidance to observations;
- expose one read-only `explore` MCP tool by default while retaining the v0.2
  narrow tools behind explicit compatibility opt-in;
- add versioned knowledge bindings and machine-checkable coverage manifests,
  including fail-closed tool, stage, workload, and activity applicability;
- bind directive guidance to one explicit directive and its exact function,
  loop, variable, or port identity; keep source-stage requested/selected
  declarations separate from schedule-stage tool-effective and achieved data;
- add a private project-local knowledge sidecar and bounded, hash-revalidated
  source excerpts that never enter the canonical ledger, bundle, REST, or ML
  export; and
- provide the explicit additive `0.2.0 -> 0.3.0` migration while preserving
  historical snapshot, entity, observation, artifact, run, and graph-hash
  semantics.

## 0.2.0 — 2026-07-20

Developer-preview contract update:

- add generic typed evidence references and evidence-backed entity
  correspondences without inferring mappings from names;
- record every variant application as an immutable materialized, no-op, or
  failed attempt, and refuse unchanged candidates without advancing the active
  graph;
- add opt-in static feature-evidence and correspondence queries across SDK,
  CLI, REST, and MCP, with fail-closed ambiguous candidate groups;
- export feature evidence, entity correspondences, and action materializations
  as separate ML tables; feature/correspondence tables default to empty and
  outcome-shaped feature evidence is rejected;
- derive versioned, evidence-backed static scope features during indexing,
  retaining unknown values as masked `null` records and keeping software-call
  targets outside hardware topology;
- introduce the `hlsgraph.runner.v2` execution contract with runner-declared,
  run-scoped staged outputs, verified local/SSH transfer, explicit plugin
  discovery, and runner-owned preflight/runtime resource guards; and
- provide an explicit, resumable `0.1.0 -> 0.2.0` bundle/SQLite migration that
  preserves historical snapshot manifests and legacy derivation IDs.

## 0.1.2 — 2026-07-18

Developer-preview completion and release-boundary hardening:

- close action lineage from a proposed `VariantAction`, through an optional
  `PredictionEnvelope.action_id`, to explicitly linked result snapshots across
  the SDK, REST, MCP, and leakage-aware ML export;
- fail closed on inactive/conditionally ambiguous source pragmas, non-literal
  Tcl directive contexts, compiler-reachable macro includes, path-dependent
  compiler builtins, and unsafe project-root placeholder use;
- make semantically empty plugin selections identity-neutral and isolate core
  indexing from unrequested entry-point conflicts;
- honor Windows compilation-database quoting, translation-unit working
  directories, and response-file arguments consistently in snapshot closure
  discovery and standard libclang extraction;
- escape generated TOML safely, make `init` a no-clobber, no-follow,
  rollback-safe creation transaction, and explicitly refuse replacing an
  existing ledger;
- expose diagnostics through a shared positive public projection while keeping
  detailed messages and metadata in the local ledger;
- export only a minimal hashed projection of variant actions by default, so
  candidate deltas, rationale, proposer text, and private source remain local;
- add automated config-directive scope and RTL-cosim mismatch acceptance cases;
  and
- record exact EPL corresponding-source lineage, expanded compiler/IR
  references, release SBOM scope, and the clean public-history provenance
  boundary.

## 0.1.1 — 2026-07-17

Windows compatibility hotfix for isolated local execution:

- retain only the OS-required `SystemRoot` bootstrap variable when ambient
  environment inheritance is disabled;
- record its digest as `bootstrap_environment_hash` without persisting the
  value, while continuing to exclude `PATH`, credentials, and other ambient
  variables.

## 0.1.0 — 2026-07-17

First developer-preview release of the reusable HLS information layer:

- versioned manifest, artifact, snapshot, run, observation, derivation,
  verification, knowledge-rule, prediction, and graph contracts;
- append-oriented SQLite ledger with content hashes, staleness, provenance,
  diagnostics, FTS, and fail-closed schema checks;
- standard libclang AST extraction, explicit degraded regex mode, external
  directive scoping, experimental MLIR/LLVM evidence adapters, and normalized
  Vitis/Vivado report adapters;
- local, SSH, fake, and replay runner contracts with staged execution,
  explicit stage-to-toolchain identity, and manifest-declared run-isolated
  local output ingestion with producer/evidence provenance;
- shared SDK/CLI/REST/MCP query semantics, self-contained layered HTML view,
  and leakage-aware JSONL/Parquet/PyG export adapters with explicit feature
  stage/attribute firewalls, atomic publication, and per-file integrity hashes;
- citation-only knowledge packs, synthetic golden fixtures, Apache-2.0
  governance, third-party notices, SPDX SBOM, and Windows/Linux CI.

The complete v0.1 support boundary is one HLS kernel. Component/system entities
are schema reservations only; host/multi-CU system collection, native MLIR
dialect plugins, and vendor-tool end-to-end CI remain explicitly outside this
release. Synthetic fixtures never satisfy real correctness/resource/post-route
verification.
