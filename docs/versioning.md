# Versioning, snapshots, and migration

HLSGraph versioning protects evidence meaning. Opening newer code must never
silently reinterpret an older observation.

## Versioned layers

HLSGraph tracks several versions independently:

| Layer | Current v0.3 contract |
| --- | --- |
| Python package | Semantic version `0.3.0`. |
| Canonical schema | `schema_version` on manifests, ledgers, graphs, and wire responses. |
| Bundle format | `bundle_version` in `.hlsgraph/bundle.json`. |
| Extraction profile | Hash of extractor names/versions, degraded selection, plugins, and options. |
| Retrieval profile | Independent `RETRIEVAL_PROFILE_SCHEMA_VERSION` (`0.3.0`) plus an algorithm profile name and content hash. |
| ML feature schema | Independent `FEATURE_SCHEMA_VERSION`, recorded as `feature_schema_version` in `DatasetManifest` and `feature_spec.json`. |
| Plugin protocol | Versioned entry-point groups such as `hlsgraph.extractors.v1`. |
| Knowledge pack/rule | Pack schema plus document ID/version/section and stable rule ID. |
| Prediction | Snapshot/subject/predicate, model/version, trainset hash, input schema version, value/unit, uncertainty, applicability, OOD assessment, and semantic metadata. |

Package compatibility does not erase the other version checks. A model trained
on one feature schema or tool/target profile must declare that applicability
even if the Python package can read both datasets.

## Developer-preview policy

v0.x is a developer preview. Public APIs and schema may evolve before 1.0, but a
breaking stored-data change must still ship an explicit migration path. We will
not silently change the meaning of an old authority class, stage, predicate,
unit, or observation.

Namespaced kinds let adapters add new entities/predicates without requiring a
breaking central enum change. New optional fields and new namespaced kinds are
normally additive; changing identity, truth meaning, or required fields is not.

## Snapshot identity

A `DesignSnapshot` is immutable and content-derived. Its ID includes hashes of:

- manifest identity and all referenced artifact bytes;
- build context, including top, translation units, include/define/flag inputs,
  config/Tcl, tests, compilation database, project-local include closure,
  forced includes, and compiler response files;
- target profile and clocks;
- constraints;
- toolchain contexts and the explicit stage-to-toolchain mapping;
- extraction profile, including explicit degraded mode and plugins; and
- an optional `VariantAction` identity when the candidate represents a proposed delta.

An action proposal alone is not result lineage. Every application attempt is an
immutable `ActionMaterialization`. An unchanged candidate records `no_op`, a
failed extraction records `failed`, and only `materialized` names a distinct
result snapshot. Retrying the same action appends an attempt rather than
rewriting history.

`parent_snapshot_id` is immutable lineage metadata but is deliberately not an
identity input. Re-observing identical design/tool/extraction inputs therefore
recovers the same snapshot ID even if reached from a different active view.

Therefore a macro, top, directive, target part, clock, tool build, artifact, or
extractor change creates a new identity. Creation time is recorded but is not an
identity field.

## Active snapshot

Each project has at most one active snapshot in `project_state`. Once that pointer
exists, “latest” in the high-level SDK resolves to the active successful
canonical graph, not simply the most recent attempted row. In a fresh bundle
that has never indexed successfully, status may expose a separate
`latest_candidate_snapshot_id` for diagnostic reporting; default graph readers
never select a candidate without a successful graph view.

Indexing follows this sequence:

1. refresh the manifest during the explicit write operation;
2. create or reuse a deterministic candidate snapshot;
3. run all selected extractors and collect diagnostics;
4. if no fatal diagnostic exists, persist the canonical graph/evidence and set
   the snapshot active;
5. if extraction fails, persist the immutable run and diagnostics but do not
   persist a partial authoritative graph or advance the active pointer.

This makes retrying an environment failure safe: the failed attempt remains
auditable while normal queries keep reading the last successful graph. REST and
MCP default to the active snapshot and can be pinned to an explicit immutable ID.

## Staleness

Staleness compares the selected/active snapshot identity with the current
manifest and artifact hashes. A stale graph can still be queried for audit, but
it no longer describes the current project inputs. Run `hlsgraph status` before
using results and re-index explicitly when inputs change.

Staleness is not a migration. A stale v0.1 graph may be schema-compatible but
out of date with respect to source/config; an old schema may match the source
but still require migration.

## Explicit ledger migration

The SQLite ledger stores its schema marker in `schema_info`. Initialization and
read paths reject unsupported versions; they do not update the marker on open.

Inspect a migration path without changing data:

```bash
hlsgraph migrate --project /path/to/project --to-version 0.3.0
```

Apply only a registered path:

```bash
hlsgraph migrate --project /path/to/project --to-version X.Y.Z --apply
```

The output states `implicit_migration: false` and lists every registered step.
The registered path is `0.1.0 -> 0.2.0 -> 0.3.0`; an existing v0.2 bundle uses
only the second step. The v0.2 step adds correspondence/action materialization
and generic derivation evidence references. The additive v0.3 step adds
knowledge bindings, coverage manifests, and rebuildable FTS support. Historical
snapshot, entity, relation, observation, artifact, run, derivation, and graph
hash semantics are not rewritten. In particular, a retained v0.2 graph view
keeps its serialized schema marker while new graphs use v0.3. The operation is
lock-protected and resumable after a partially completed ledger or bundle step.
If no path exists, HLSGraph fails closed; opening a legacy bundle never migrates
it.

Migration rules:

- preserve the old payload or emit a new observation when semantics change;
- do not rewrite a prediction as an observation or a declaration as an applied
  directive;
- keep run/artifact/evidence provenance and units;
- maintain snapshot immutability and graph-hash reproducibility;
- document any consumer action required for ML feature or label schemas.

## Reproducibility expectations

For the same snapshot and extraction profile, canonical JSON ordering, stable
IDs, graph hash, and query ordering are deterministic. Tool runs are separate
events: their timestamps and run IDs are expected to differ, while imported
observations must retain the exact run/artifact provenance that produced them.

Reproducing a real QoR claim additionally requires the original licensed design,
tool build/environment, target/platform, constraints, workload, and report
artifacts. A schema-compatible synthetic fixture is not a reproduction of a
vendor result.

Prediction IDs cover every field that changes prediction meaning: snapshot and
subject, predicate, model and model version, input schema, trainset hash, value
and unit, uncertainty, applicability, OOD assessment, semantic metadata, and
an optional linked action ID.
`created_at` is event-time metadata rather than prediction semantics and is not
an identity input. Changing any semantic field creates a distinct immutable
`PredictionEnvelope`; predictions never reuse observation or fact authority.
