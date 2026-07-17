# Changelog

All notable changes are recorded here. HLSGraph follows semantic versioning;
the schema and bundle have independent explicit versions.

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
