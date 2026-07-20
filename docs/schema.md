# Schema and truth model

HLSGraph's public schema is designed to preserve disagreements and provenance,
not to collapse every value into a single property bag. The implementation is
defined by `hlsgraph.model`, `hlsgraph.graph`, and the schema version recorded in
each bundle.

## Two independent dimensions: authority and stage

Every entity, relation, and observation has a stage. Entities, relations, and
observations also carry an authority class. These dimensions are orthogonal: a
post-route value can be synthetic fixture data, while a source-stage value can
be a declared constraint rather than an observed fact.

### Authority classes

| Value | Meaning |
| --- | --- |
| `declared_constraint` | User/source/config intent, such as a requested II or clock. |
| `static_fact` | Deterministically parsed structure or source anchor. |
| `compiler_decision` | IR, schedule, binding, banking, or other compiler-selected result. |
| `tool_observation` | A value reported by an HLS/implementation tool. |
| `verification_evidence` | CSim, RTL cosim, assertion, formal, mismatch, or deadlock evidence. |
| `physical_measurement` | A measurement from hardware/runtime instrumentation. |
| `derived_fact` | A deterministic algorithm result with cited input observations. |
| `knowledge_rule` | Versioned documentation guidance, not design evidence. |
| `prediction_hypothesis` | A model/heuristic/LLM output; never a tool observation. |
| `synthetic` | CI or example evidence that must not be presented as real tool truth. |

### Standard stages

`source`, `ast`, `mlir`, `hls_ir`, `llvm`, `schedule`, `rtl`, `post_synth`,
`post_place`, `post_route`, `csim`, `cosim`, `hardware_runtime`, and `unknown`.
Kinds and stage strings may use namespaced extensions for vendor/dialect data.

Consumers should filter on both dimensions. For example, `stage=post_route`
alone does not prove that a value came from a real implementation run; its
authority and artifact metadata must also be checked.

### Run/report stage compatibility

Run-backed tool truth is checked by the versioned policy
`hlsgraph.tool-evidence.v0.1` at both ledger commit and ML-export boundaries.
Both boundaries replay the run's stage, toolchain, environment, command, and
working directory against the immutable snapshot manifest; metadata booleans
alone cannot establish tool truth. Index runs cannot claim external tool truth.
The canonical producer mappings include `csynth -> schedule|csynth`,
`rtl_cosim|cosim -> cosim`, `rtl_export|rtl -> rtl`,
`vivado_synth|post_synth -> post_synth`, and exact `post_place`, `post_route`,
and `hardware_runtime` stages. Source, AST, MLIR, LLVM, and `unknown` are not
eligible present-label stages. A `physical_measurement` must be a
`hardware_runtime` observation; a csynth report can never establish post-route
timing or a board measurement.

Built-in Vitis/Vivado report kinds are matched exactly. Generic Vivado report
kinds additionally require `artifact.metadata.stage`; filenames and namespace
prefixes are never used to infer a stage. A vendor-neutral plugin report must
opt into the same policy with this complete metadata contract:

```json
{
  "hlsgraph_evidence": {
    "policy_version": "hlsgraph.tool-evidence.v0.1",
    "observation_stage": "hardware_runtime",
    "run_stage": "hardware_runtime",
    "semantics": "physical_measurement"
  }
}
```

`semantics` is authority-specific: `tool_report`, `verification_report`, or
`physical_measurement`. An unknown kind or incomplete contract fails closed.
Kinds already known to the policy cannot use this extension contract to acquire
a different stage meaning.

## Requested, effective/applied, and achieved

These states must remain distinct:

- **Requested** is design intent. Active source pragmas and literal, top-level
  external directives emit `directive.requested` with
  `authority=declared_constraint`. Inactive preprocessor regions and Tcl whose
  control/substitution context cannot be proven literal are diagnostics, not
  directive facts.
- **Effective declared** is the winner after deterministic precedence among
  declarations. `directive.effective` means HLSGraph resolved the declarations;
  its metadata explicitly says `tool_applied=false`.
- **Applied/effective by tool** requires tool evidence, such as
  `directive.tool_status` and `directive.tool_effective`. A tool may report
  `applied`, `ignored`, `unmet`, `rejected`, or `unknown`.
- **Achieved** is an observed outcome, for example `qor.achieved_ii` or
  `directive.achieved`. It may differ from the requested value.

No precedence rule upgrades a declaration into a compiler decision. Likewise,
an achieved csynth value is not silently replaced by a post-route value.

## Core contracts

### Project and artifact identity

| Type | Purpose |
| --- | --- |
| `ProjectManifest` | Stable project ID plus build, target, constraints, toolchains, artifacts, stage commands/outputs, and explicit stage-to-toolchain identity. |
| `BuildContext` | Top, translation units, include paths, macros, flags, config/Tcl, tests, dependencies, and compilation database. |
| `TargetProfile` | Vendor/part/platform, clocks, capacities, reserved resources, and reserved memory-topology metadata. |
| `ConstraintSet` | Performance, resource, power, numerical, interface, and XDC intent. |
| `ArtifactRef` | Kind, URI, SHA-256, size, license, producer, retention, access, and metadata; no required body. |
| `DesignSnapshot` | Immutable identity over manifest, artifacts, build, target, constraints, toolchain, and extraction profile. |
| `VariantAction` | A proposed delta from a parent snapshot; it is not proof that the delta was applied or succeeded. Result lineage requires an explicit matching action/parent pair in a new snapshot. |
| `ActionMaterialization` | One immutable attempt to apply a recorded action. `materialized`, `no_op`, and `failed` remain distinct; only `materialized` may name a result snapshot. |

Snapshot IDs are stable hashes of identity fields, not creation timestamps.
`created_at` records the ledger event but does not change the ID. Entity and
relation IDs are stable only within their snapshot. Parent lineage is recorded
separately and does not perturb otherwise identical design identity.

Executable stages resolve through `ProjectManifest.toolchain_for_stage(stage)`.
One declared toolchain is an unambiguous compatibility case. With multiple
toolchains, every stage command must have a `stage_toolchains` entry. Missing
toolchains, duplicate IDs, unknown stage/ID mappings, and ambiguous selection
are invalid rather than falling back to list order. The mapping participates in
both manifest and toolchain snapshot hashes.

### Graph and evidence

| Type | Purpose |
| --- | --- |
| `Entity` | Namespaced kind, name/hierarchy, attributes, anchors, stage, authority, and completeness. |
| `Relation` | Explicit typed edge with endpoints, stage, authority, anchors, mapping kind, and completeness. |
| `Anchor` (`SourceAnchor`) | Artifact and source/IR location plus mapping kind and ambiguity. |
| `Observation` | Atomic subject/predicate/value statement with unit, stage, authority, run, artifact, workload, and completeness. |
| `EvidenceRef` | Typed reference to an observation, derivation, artifact, entity anchor, or explicit relation, with an explicit snapshot and optional anchor. |
| `Derivation` | Recomputable deterministic output with algorithm/version and generic `EvidenceRef` inputs. Legacy observation-ID inputs normalize to evidence references without changing legacy stable IDs. |
| `EntityCorrespondence` | Explicit evidence-backed entity mapping across snapshots, with mapping kind, producer/version, authority, and completeness. It never arises from name matching. |
| `Diagnostic` | Structured extraction/tool health event with stage, severity, subject/artifact, and guidance. |
| `VerificationResult` | Independent correctness evidence for a specific method and optional workload. |
| `ToolRun` | Immutable stage request/result with backend, command, status, failure class, artifacts, and gates. |
| `KnowledgeRule` | Versioned documentation rule and citation, stored outside design observations. |
| `PredictionEnvelope` | Model output with model/data/schema version, uncertainty, applicability, OOD metadata, and an optional action link; it remains outside the fact layer. |
| `LabelSpec` | Snapshot-scoped ML label reference to a real observation, including stage, unit, mask, and missing/censoring state. |

When a prediction carries `action_id`, that action must belong to the
prediction's input snapshot. `Project.index_variant(action_id)` refuses an
unchanged candidate as an explicit `no_op`; a successful changed candidate
records a `materialized` attempt and a distinct result snapshot. Failed and
no-op attempts retain diagnostics but never become result lineage. These are
stored links, not evidence that a candidate improved QoR.

Entity, relation, artifact, predicate, action, backend, and plugin identifiers
are namespaced strings (for example `amd.vitis.csynth_xml` or
`circt.handshake.func`). The public model avoids a closed enum that would force
future vendors into AMD-specific concepts.

Anchor text accepts bounded project-relative or symbolic locations, including
`loc("kernel.cpp":18:5)` and `!dbg !4`. Host-absolute Windows/POSIX locations
are normalized to a stable `redacted.sha256:<digest>` marker with no original
path content. Post-construction mutation is rejected as non-canonical before
SQLite persistence; an adapter should still normalize an in-project path or
emit an explicit human-readable redacted external location itself.

## Completeness and mappings

`complete`, `partial`, `missing`, and `ambiguous` are explicit states. A missing
AST-to-IR or IR-to-schedule mapping is not repaired by name matching. Cross-layer
relations may be many-to-many and retain mapping kind, anchors, and ambiguity.
Cross-snapshot mappings use `EntityCorrespondence`. Query and ML surfaces group
multiple explicit candidates and leave the singular resolved entity unset; they
never select the first candidate.

The default architecture projection excludes software-call and LLVM-CFG edges
from impact semantics. Those edges remain queryable evidence but are not
hardware topology.

## Deterministic static feature derivations

Indexing derives versioned scope-level ML evidence directly from canonical
entities and relations. The built-in predicates are
`feature.operation_histogram`, `feature.index_histogram`,
`feature.trip_count`, `feature.loop_bounds`, `feature.bitwidth`,
`feature.memory_access`, and `feature.software_call_targets`.
`feature.dependence_distance` has the same stable per-scope row contract, but
its value remains `null` unless an entity or relation explicitly records a
proven distance.

Every row records its algorithm/version, stage, authority, completeness, and
typed evidence references. Unknown values are `null` with `missing` or
`partial` completeness, and public query/ML projections set `mask=false`.
Zero is never substituted for missing evidence. An empty histogram/map/list is
valid only when a complete evidence plane proves the scope empty. The current
text LLVM adapter can prove its directly contained operation plane complete
unless it reports truncation. An untruncated MLIR artifact whose dialects all
have registered adapters explicitly marks its static-feature domain complete;
unsupported dialects and truncation keep that domain partial. Degraded source
scanning never upgrades operation, bitwidth, memory, call, or loop facts to
complete.

Cross-plane evidence is followed only through `hls.contains`, `ir.contains`,
and explicit `cross.maps_to`/`cross.projects_to` relations. No name matching is
performed by the feature pass. `feature.software_call_targets` is a deduplicated,
stable-sorted list of target entity IDs with unit `entity_ids`; it consumes only
explicit `software.calls` evidence marked `ml_input_evidence=true` and is not
canonical hardware topology.

## Stage-specific observations

The same predicate family can have multiple simultaneous observations:

- requested clock versus achieved clock;
- csynth latency estimate versus post-synth/post-route timing;
- HLS resource estimate versus implemented utilization;
- static FIFO depth versus workload-specific occupancy/stall measurements;
- cosim result for one testbench/workload versus another workload;
- prediction versus a later real observation.

Units, stage, run/artifact ID, workload, and completeness are part of each
observation's identity. Conflicts are evidence to inspect, not rows to overwrite.

## Three independent gates

| Gate | Required evidence |
| --- | --- |
| `correctness` | Separate CSim, RTL cosim, assertion, formal, mismatch, or deadlock records. |
| `resource_fits` | Deterministic comparison of stage-specific utilization against effective capacity. |
| `post_route_timing` | Post-route timing evidence, normally WNS at the relevant clock(s). |

Each gate is `pass`, `fail`, or `unknown` and cites evidence IDs. A design is
reported as verified only when all three gates pass and none is supported only
by synthetic evidence. A successful process exit is not automatically a passed
gate. In v0.1, a trusted correctness `pass` specifically requires both CSim and
RTL cosim `pass` evidence from trusted tool runs, bound to the same explicit
campaign and workload. Other verification methods may coexist as independent
evidence, but they do not silently substitute for either required result. The
typed observations must cite managed report objects whose current bytes still
match the recorded size and SHA-256. Resource fit and post-route timing must be
recomputed from complete, scoped observations produced by the same fresh
post-route run; exact algorithm versions, units, target-profile identity, and
capacity keys are part of that closure. A `StageResult` is verified only when
the current invocation itself contains the eligible CSim/cosim cohort and that
eligible physical run, so a partial rerun cannot inherit unrelated historical
passes.

## ML truth separation

`DatasetManifest` records snapshot IDs, feature schema, family/dedup-aware
splits, licenses, opt-in `feature_evidence_predicates`, and opt-in
`entity_correspondence_kinds`. Both opt-in lists default to empty. Selected
feature evidence must have a recursively static evidence closure; outcome-,
workload-, prediction-, and label-shaped records are rejected. Static node
features exclude QoR-, label-, and prediction-prefixed attributes. Labels are
keyed by `(snapshot_id, label_id)`
and reference observations rather than duplicating truth values. A present label
must resolve to a complete, same-snapshot observation produced by a successful,
fresh, non-synthetic tool run and a retained managed report whose bytes still
match its size and SHA-256. A missing/censored label carries an explicit
namespaced reason and no observation reference. Predictions are exported in
their own table and cannot satisfy a label lookup.

This separation prevents accidental label leakage and prevents any estimate
from being interpreted as a synthesis result.
