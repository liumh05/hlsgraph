# Public interfaces

HLSGraph exposes one canonical ledger and query semantics through several
surfaces. Use the Python SDK for programmatic writes/indexing and the read-only
surfaces for agents and services.

## Python SDK

The high-level entry point is `hlsgraph.Project`:

```python
from hlsgraph import Project

project = Project.create_from_manifest("hlsgraph.toml")
indexed = project.index()                 # libclang path
status = project.status()
hits = project.query("compute_loop", stages=["ast"])
context = project.explore(query="compute", depth=2)
project.render("graph.html", format="html")
project.export_graph("graph.json")
project.export_dataset("dataset", format="jsonl")
```

Use `project.index(degraded=True)` only when deliberately accepting the regex
scanner. The result and health diagnostics preserve the degraded status.

Candidate lineage is explicit and does not promote a proposal or prediction to
fact. Record a `VariantAction` against its parent snapshot, optionally bind a
`PredictionEnvelope.action_id`, apply the candidate inputs, and call
`project.index_variant(action_id)`. If source, build, target, constraint, and
toolchain identity are unchanged, indexing fails closed and records a `no_op`
materialization without changing the active graph. `project.variants()` reports
predictions, immutable materialization attempts, and only materialized result
snapshots; `project.materializations()` exposes the attempt ledger directly.

`project.feature_evidence()` returns only deterministic derivations with a
recursively static evidence closure. `project.correspondences()` returns only
stored `EntityCorrespondence` records. Candidate groups with more than one
endpoint are marked ambiguous and have no singular resolved entity. Both public
query projections omit free-form metadata and expose only its presence and
SHA-256 digest.

`Project.run(runner, stages=...)` executes only stage argv declared in the
manifest. `LocalRunner` and `SSHRunner` require explicit execution enablement;
`FakeRunner` and `ReplayRunner` are non-tool-truth paths for CI and cache tests.
The current runner is generic orchestration, not a turnkey vendor flow script
generator. Before and after each selected stage, the SDK rejects a stale active
snapshot and re-hashes every immutable design input plus each output explicitly
chained into that stage; unrelated historical outputs do not become implicit
inputs. It reads argv and toolchain identity from that snapshot's immutable
manifest row. SSH additionally requires a pinned
environment hash plus `toolchain.metadata.remote_attestation_argv`. In the same
remote shell it hashes the probe's exact stdout, compares that value with
`environment_hash`, and verifies each remote input's size and SHA-256 before the
stage command starts.

Toolchain selection never depends on list order. A manifest with one toolchain
may omit the mapping; a manifest with multiple toolchains must map every
executable stage explicitly:

```toml
[stage_toolchains]
csynth = "amd.vitis.2024_2"
post_route = "amd.vivado.2024_2"
```

Duplicate toolchain IDs, missing mappings, and mappings to an unknown stage or
toolchain make the manifest invalid.

`StageResult.verified` is fail-closed: passing gates need evidence IDs, recursive
evidence validation, and CSim/RTL-cosim checks from the same explicit
campaign/workload. The current invocation itself must contain that eligible
CSim/cosim run pair and the single post-route run that simultaneously proves
resource fit and timing; unrelated historical passes cannot verify a partial
rerun. Output discovery is never inferred. The `hlsgraph.runner.v2` protocol
returns a `RunnerExecution` containing the `ToolRun` and only the outputs
explicitly declared by the manifest. The SDK owns a restricted, run-scoped
staging directory and rejects undeclared paths, traversal, links/reparse
points, size or SHA-256 mismatches, duplicate outputs, and changes between
validation and commit. Successful local or SSH runs copy verified bytes into
the content-addressed store, bind `producer_run_id`, parse supported
Vitis/Vivado reports, and atomically commit the run plus observations,
derivations, verifications, diagnostics, and artifacts:

```toml
[[stage_outputs.csim]]
path = "run/csim/result.json"
kind = "amd.vitis.csim_result"
role = "verification_report"
required = true
metadata = { workload_id = "tb.default", campaign_id = "campaign.default" }
```

Declared output paths must be absent before execution, so stale files cannot be
attributed to a new run. A cross-process bundle execution lock covers command
execution, output validation, and ledger commit. `consumed_by` explicitly
chains a generated artifact into a later selected stage and the SDK enforces
producer-before-consumer order. Unsupported structural outputs require a new
manifest artifact and re-indexed snapshot.

`SSHRunner` creates a non-reusable remote run directory, verifies transferred
inputs, freezes and hashes declared outputs remotely, and transfers those bytes
directly into SDK-owned staging. External asynchronous file synchronization is
never treated as evidence. A configured resource guard runs before the tool;
an optional runner-owned runtime monitor receives only a runner-injected process
ID token and terminates the full process group on violation. Guard, transport,
timeout, design, correctness, and trusted resource failures retain distinct
failure classes. Fake and replay runners never claim real-tool evidence.

Runner plugins are discovered only from the `hlsgraph.runners.v2` entry-point
group and only when explicitly selected. A plugin must return the exact v2
protocol identity and capabilities; private host configuration and credentials
belong to the plugin deployment, not to the public bundle or API.

For lower-level consumers, public contracts include `GraphBundle`,
`CanonicalGraph`, `CoreService`, schema dataclasses, extractor/plugin protocols,
and the ledger store. Opening a bundle is read-only with respect to extraction
and plugins; explicitly indexing or running is a separate operation.
`GraphBundle.add_managed_artifact` is only for artifacts without a producer run.
Run-produced bytes must use `prepare_managed_artifact` followed by the atomic
`LedgerStore.commit_run_result` contract; published content-addressed bytes are
never deleted as rollback because another process may already reference them.

## CLI

CLI output is JSON by default and uses the same SDK/query service:

```text
hlsgraph init       create hlsgraph.toml and .hlsgraph/
hlsgraph index      build and activate a deterministic snapshot
hlsgraph status     report graph, runs, gates, health, and staleness
hlsgraph query      query entities, static feature evidence, or correspondence records
hlsgraph explore    return a bounded graph and evidence neighborhood
hlsgraph run        use fake/local/SSH runner for manifest-declared stages
hlsgraph render     write HTML, JSON, Mermaid, DOT, or SVG
hlsgraph export     write canonical graph or leakage-aware ML dataset
hlsgraph doctor     perform read-only environment/bundle checks
hlsgraph migrate    inspect or explicitly apply a registered migration
hlsgraph knowledge  list rules or index local document metadata
hlsgraph serve      start the read-only REST service
```

Minimal fixture session:

```bash
python -m pip install -e ".[clang]"
hlsgraph index --project examples/dataflow_gemm --manifest hlsgraph.toml
hlsgraph status --project examples/dataflow_gemm
hlsgraph query --project examples/dataflow_gemm compute
hlsgraph render --project examples/dataflow_gemm graph.html
```

Variant materialization and the two typed provenance queries are explicit:

```bash
hlsgraph index --project /path/to/project --action-id action_... --degraded
hlsgraph query --project /path/to/project --record-class feature-evidence \
  --predicate feature.operation_histogram
hlsgraph query --project /path/to/project --record-class correspondence \
  --snapshot-id snapshot_... --other-snapshot-id snapshot_... \
  --correspondence-kind mapping.semantic_identity --direction source
```

Local or SSH execution is intentionally more explicit:

```bash
hlsgraph run --project /path/to/project --backend local \
  --allow-execution --stage csynth
```

The command runs the argv already declared for `csynth`; it does not infer or
download a toolchain. For SSH, the selected toolchain must declare
`environment_hash`; remote project files must already be synchronized byte for
byte with the active snapshot. The declared attestation command should report a
stable toolchain/environment identity (for example, a maintained probe script),
not volatile timestamps or secrets.

```toml
[[toolchains]]
id = "amd.vitis.2024_2"
vendor = "amd"
name = "vitis_hls"
version = "2024.2"
# SHA-256 of the exact stdout bytes emitted by the probe below.
environment_hash = "<64 lowercase hex characters>"
metadata = { remote_attestation_argv = ["./tools/attest-vitis-env"] }
```

## Shared query semantics

`CoreService` backs SDK query/explore, CLI, REST, and MCP. Search follows a
deterministic exact -> substring -> SQLite FTS5 -> fuzzy chain, then applies
stable sorting and snapshot-bound cursors. Queries can filter by entity kind,
scope, stage, and authority. Exploration returns a bounded explicit-relation
neighborhood plus observations, diagnostics, and artifact metadata.

Diagnostics on these shared/public surfaces use a positive projection. It
includes stable IDs, code, severity, stage, safe anchor fields, and
`detail_sha256`, while setting `detail_redacted=true`. Tool/plugin messages,
guidance, and metadata remain available only from the trusted local bundle
store and are not serialized by CoreService query results, CLI status, REST, or
MCP.

As of v0.2.0 this projection also applies to previously exposed diagnostic
records on all public surfaces. Consumers should use the stable code, severity,
anchor, and `detail_sha256` fields; raw diagnostic message/metadata access is a
local ledger operation, not a public wire guarantee.

Cursors are tied to the query specification and snapshot. A cursor from another
query or snapshot is rejected rather than silently reused.

The default graph is the project's active successful snapshot. A fresh or
failed-only bundle may retain a candidate snapshot and diagnostics but has no
queryable canonical graph. SDK, REST, and MCP can pin an immutable successful
snapshot ID where their APIs expose that option.

## REST/OpenAPI

Start the dependency-free read-only service:

```bash
hlsgraph serve --project /path/to/project
```

It listens on `127.0.0.1:8000` by default. Non-loopback binding requires the
explicit `--allow-remote` flag; the built-in server does not provide TLS or
authentication, so an authenticated reverse proxy is required for any shared
deployment.

The versioned prefix is `/api/v1`. Implemented GET resources are:

- `/status`, `/overview`, and `/graph`;
- `/entities` and `/entities/{entity_id}`;
- `/observations`, `/diagnostics`, `/runs`, and `/artifacts`;
- `/artifacts/{artifact_id}` (metadata only; no private body);
- `/derivations`, `/verifications`, `/predictions`, `/variants`, `/knowledge`,
  and `/evidence`;
- `/feature-evidence` and `/correspondences`, both backed by the same
  fail-closed `CoreService` queries used by SDK, CLI, and MCP;
- `/compare?other_snapshot_id=...`;
- `/api/v1/openapi.json` and the equivalent root alias `/openapi.json`.

List endpoints support bounded pagination; relevant resources support filters.
Pass `snapshot_id` to select an immutable graph. Every non-GET method returns
`405 Method Not Allowed`.

Example:

```bash
curl "http://127.0.0.1:8000/api/v1/entities?q=compute&stages=schedule"
curl "http://127.0.0.1:8000/api/v1/observations?stage=post_route"
```

## MCP

Install the optional transport and run it over stdio:

```bash
python -m pip install -e ".[mcp]"
hlsgraph-mcp /path/to/project
# optionally: hlsgraph-mcp /path/to/project --snapshot-id snapshot_...
```

The MCP server registers these read-only tools:

- `overview`, `search`, and `context`;
- `module_or_region` and `traverse`;
- `impact`, which reports explicit dependency facts and never invents QoR
  deltas; software-call and LLVM-CFG edges are excluded by default;
- `evidence` and `compare`;
- `feature_evidence`, which exposes selected static derivations only, and
  `correspondences`, which preserves ambiguous candidate groups without
  auto-resolution;
- `health`, including degraded extraction, missing evidence, and staleness;
- `runs`, which returns redacted execution/failure records, and `predictions`,
  which returns explicitly labeled prediction envelopes outside the fact layer;
- `variants`, which reports only explicitly recorded action, prediction, and
  result-snapshot lineage and never infers whether a candidate was applied;
- `render`, which returns an in-memory rendering and does not write files;
- `knowledge`, whose results are labeled `authority_class=knowledge_rule`.

MCP has no tool for writing graph facts or promoting a prediction. Any external
source/config change creates a new state that must be re-indexed or verified by
a real tool before new observations exist.

## Human view

`Project.render()` / `hlsgraph render` can emit a self-contained HTML view using
a layered hardware/dataflow layout. It supports evidence details,
stage/authority/category filters, search, focus, bottleneck coloring, FIFO-depth
edge width, and light/dark themes. Mermaid, DOT, SVG, and projection JSON are
available for simpler integrations.

The renderer consumes the canonical graph; it does not infer topology. The HTML
does not embed private source bodies, but names, source spans, artifact metadata,
and observations may still be sensitive. Review an export before publishing it.

## ML export

JSONL is the baseline format. The destination must not already exist and is
published atomically. An export directory contains separate tables for
nodes, edges, observations, non-truth observations, labels, splits, artifacts,
runs, predictions, minimally projected variants, action materializations,
snapshot lineage, opt-in feature evidence, and opt-in entity correspondences, plus
`feature_spec.json` and a hashed manifest. Labels carry `snapshot_id`, stage,
unit, mask, and censoring state. Present labels reference same-snapshot complete
observations from successful fresh real-tool runs and intact retained reports;
missing labels have no observation reference and require an explicit reason.
Present labels must also satisfy the exact run-stage/report-kind compatibility
policy recorded as `tool_evidence_policy_version` in `feature_spec.json`;
unknown plugin report kinds require the explicit `hlsgraph_evidence` metadata
contract documented in `schema.md`.

`variants.jsonl`, `action_materializations.jsonl`, and
`snapshot_lineage.jsonl` are non-feature lineage tables.
The default export includes action identity/kind/scope links and hashes but not
raw candidate deltas, rationale, proposer text, or source replacements. A
result-only dataset may contain a minimal parent-action stub; it never copies
the undeclared parent snapshot's private action payload.

`feature_evidence.jsonl` and `entity_correspondence.jsonl` are present even
when empty. They remain empty unless the corresponding
`DatasetManifest.feature_evidence_predicates` or
`entity_correspondence_kinds` list explicitly opts records in. Feature evidence
recursively rejects tool outcomes and workload-bound evidence. Correspondence
rows retain all explicit candidates and set `resolved_target_entity_id=null`
when a source/kind/snapshot group is not unique. Neither table is populated by
name inference.

The indexer supplies built-in static feature derivations without consumer-side
injection. Each selected feature-evidence row includes `algorithm`,
`algorithm_version`, `stage`, `completeness`, `evidence_refs`, and `mask`. A
`null` value is retained with `mask=false`; consumers must not coerce it to
zero. `feature.software_call_targets` is an ML-only sorted unique `list[str]`
of explicit software-call target entity IDs (`unit=entity_ids`) and must not be
interpreted as HLS instance topology.

`observations.jsonl` is fail-closed at the public export boundary. A real-tool
authority observation without a producer run rejects the export. A retained
failure/legacy observation whose producer explicitly does not claim tool truth
is removed from that truth table and written to `nontruth_observations.jsonl`
with `tool_truth=false` and a machine-readable reason. Tool-truth runs are also
rechecked against the immutable snapshot's stage, toolchain, environment,
command, and working-directory identity before any observation is published.
Predictions are physically separate. Source text is never embedded. The manifest
records SHA-256 and size for every generated table/spec file, and `export_hash`
covers that integrity map.

`DatasetManifest.feature_stages` and `feature_attribute_allowlist` are explicit
feature-time and attribute firewalls. The default stops at LLVM and excludes
unknown plugin attributes. Opting into schedule or later stages is a visible
dataset decision. The PyG adapter applies the same stage contract.

Nested node-attribute containers remain closed-schema. In particular, `dims`
is exported only when explicitly allowlisted and is a non-empty list of at most
16 positive integers, each no greater than 2,147,483,647. Tuples, booleans,
non-integers, non-positive values, overlong lists, and out-of-range dimensions
reject the entire field rather than being partially copied.

Parquet requires the `parquet` extra. The optional PyG adapter requires the
`pyg` extra and emits only kind/stage indices as `x`; the core package never
depends on Torch and does not place QoR observations in input features.

```bash
hlsgraph export --project /path/to/project dataset \
  --kind dataset --format jsonl --dataset-id public.my_dataset \
  --snapshot-id snapshot_parent --snapshot-id snapshot_result \
  --feature-evidence-predicate feature.operation_histogram \
  --entity-correspondence-kind mapping.semantic_identity
```

## Knowledge interface

Packaged rules can be filtered without obtaining vendor documents:

```bash
hlsgraph knowledge --project /path/to/project --vendor amd \
  --tool vitis_hls --stage schedule
```

The local indexer hashes metadata for a document the user lawfully possesses;
it does not copy or parse the document body:

```bash
hlsgraph knowledge index --project /path/to/project \
  --path /local/path/to/ug1399.pdf \
  --document-id amd.ug1399 --document-version 2024.2
```

Knowledge rules are guidance and remain outside design observations.
