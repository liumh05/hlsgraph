# Unified deterministic retrieval

HLSGraph v0.3 provides one retrieval contract across the Python SDK, CLI,
REST, and MCP. It ranks existing evidence; it never creates graph relations,
QoR values, verification results, or knowledge applicability.

## Truth planes

Every result preserves four independent planes:

- `facts` contains canonical entities, explicit relations, observations,
  derivations, diagnostics, and verification evidence;
- `guidance` contains versioned public rules and authorized local-document
  metadata;
- `predictions` contains separately labelled prediction envelopes and is
  excluded unless explicitly requested; and
- `flow` is a bounded path over already-stored, typed relations. It is a
  retrieval explanation, not a new architecture projection.

Knowledge rules never become observations. Software calls and LLVM CFG edges
remain evidence and receive zero architecture-propagation weight.

`RetrievalSpec.planes` is the final channel allow-list. Predictions have an
additional capability gate: `include_predictions=True` explicitly appends the
isolated `predictions` plane. Naming `predictions` in `planes` without that
flag is rejected instead of silently returning or hiding data. Results from
local adapters pass through the same final allow-list; an adapter cannot add a
plane that the request did not enable. The two private-content booleans require
actual boolean values so Python, CLI, REST, and MCP cannot disagree over
string truthiness.

Knowledge applicability is target-instance scoped and fail-closed. Before a
binding is even considered, retrieval revalidates that it belongs to one
installed pack whose immutable coverage and inventory both agree that the pack
is `review_ready`. The inventory's activation hash commits the complete
serialized rules, bindings, and coverage, and retrieval recomputes that hash
from the exact rows it loaded. This is independent of the catalog and SQLite
write gates; old, altered, or directly injected rows remain inert. Pack loading
also proves that each binding entails its rule condition and occurs exactly
once under that rule in coverage. Retrieval then re-proves every condition from
one instance-local context. Missing, conflicting, or multi-valued evidence
cannot borrow a value from a sibling entity, report, run, workload, or scope.

Scalar constraint matching is deliberately non-authorizing. For each retrieval
call, HLSGraph derives and registers the exact target contexts from the pinned
ledger snapshot and current graph, then retains canonical bytes and hashes for
the complete binding, complete rule, and context values inside one session.
Matching decodes fresh local copies and runs inside one atomic session method,
with identity and hashes checked before and after evaluation; no returned
snapshot or result is accepted later as authorization. Only the same retriever
can present the issued opaque object while the session is live. A serialized
or caller-created `DesignSnapshot`, plain dictionary, copied object,
caller-selected target, omitted condition, another retriever, or an object
replayed after session close cannot activate guidance even when all visible
IDs and scalar values match.

OpenIR is intentionally more conservative in v0.3. The public MLIR, LLVM, and
CIRCT material is citation metadata with every supported target explicitly
classified `no_normative`; the built-in OpenIR pack contains no executable
bindings. The text parsers still preserve operations, blocks, CFG, memory
access, typed locations, SSA def-use, mappings, and static derivations as
structural evidence. They do not claim compatibility with any exact language
specification revision. `CanonicalGraph.metadata` is caller-constructable, so
even a well-formed `ArtifactSemanticAttestation` placed there is ignored. A
future executable OpenIR rule requires a separately designed, persisted
capability/authorization contract; parser names, filenames, syntax, dialect
names, artifact hashes, and citation revisions cannot substitute for it.
Consequently, `cross.maps_to`, `cross.projects_to`,
`handshake.dataflow`, and MLIR/LLVM aggregate features remain queryable facts
or evidence but cannot activate language-spec guidance. Unknown, redacted,
degraded, ambiguous, and unresolved mappings remain incomplete. Missing static
evidence remains unknown, never a numeric zero.

Boolean discriminators in pack `required_context` are JSON booleans; both the
catalog matcher and hybrid retriever use the same canonical boolean tokens.

Once a reviewed pack is explicitly installed, the same instance-local boundary
applies to Arm AXI references. `m_axi`, `s_axilite`, or
`axis` on an exactly scoped Vitis `INTERFACE` directive for a unique direct
port of the configured top component proves only the AMD
2024.2 source request. It does not supply an Arm specification revision,
endpoint roles, channel direction, or a realized interface instance. Arm
IHI0022 H.c and IHI0051 B are therefore `citation_only`; generic `hls.port`,
`hls.stream`, and `hls.streams_to` targets cannot activate their guidance.

AMD implementation and verification bindings apply the same rule. An XDC
target qualifies only when its ledger artifact carries the recorded SHA-256,
is attached to the queried snapshot, and is one of that snapshot's declared
constraint inputs for an explicitly mapped Vivado stage. Stage, tool identity,
and version come from the same stage-to-toolchain selection; artifact metadata
cannot donate them, and different selected Vivado builds make the projection
incomplete rather than producing cross-paired contexts. Gate guidance is not
selected from a free-standing run/stage claim: correctness must close to
workload-scoped typed observations and their managed report; resource fit must
close to one complete post-route utilization set plus the matching target,
device, and capacity identities; post-route timing must close to a timing
report, a separately retained routed-checkpoint identity, and the snapshot's
constraint identity. Every report/checkpoint leaf must be an exact
`manifest.stage_outputs` declaration for the same successful, fresh
`runner.local`/`runner.ssh` run: output ID and path, kind, role, access,
license, managed retention, live size, and SHA-256 are checked together, with
link/reparse paths rejected. Post-route timing also re-reads every declared XDC
input without following links and folds each input artifact ID and SHA-256 into
the gate context. The producing run must also have a valid ledger
`ExecutionAttestation` and its matching atomic `ExecutionCommitReceipt`; that
check revalidates the attested output set against the current managed bytes.
Rows imported by an older migration or inserted directly with tool-like
metadata cannot activate a Gate binding without that receipt. Missing identity
leaves the rule inapplicable rather than borrowing evidence from another run or
target instance.

Internally, this closure mints the reserved, versioned context value
`gate_evidence_qualified=derived_from_typed_evidence_v1`. Generic project,
entity, run, artifact, observation, derivation, and diagnostic metadata can
never supply that value. The retriever also rechecks the retained report's
size and SHA-256 before minting it, so stale or replaced managed bytes fail
closed. Capacity-unknown and capacity-incomplete diagnostics are intentionally
`no_normative`: they describe an HLSGraph evidence gap and cannot satisfy the
utilization-report condition of a vendor rule.

Individual AMD QoR and implementation observations use a separate closure.
Clock estimates, latency/II, schedule fields, timing, utilization, congestion,
and power guidance requires
`observation_evidence_qualified=derived_from_typed_observation_evidence_v1`
plus the current observation, run, and report identities. The retriever mints
those values only when the observation cites a same-snapshot, same-run,
parser-typed artifact that was declared for that run stage, retained in the
managed store, and still matches its recorded size and SHA-256. Predicate,
stage, authority, and artifact kind are checked as one policy tuple; a timing
predicate cannot borrow a utilization report merely because both came from a
Vivado run. The same ledger attestation/receipt validation is mandatory before
the observation capability is minted, and the evidence document's public
`tool_truth` flag is recomputed from that live receipt rather than copied from
run metadata. Metadata cannot inject any of these reserved values. The current
physical-summary adapter does not retain a self-validating waiver-set identity,
so DRC and CDC count targets remain `no_normative` rather than presenting
waiver-sensitive guidance from a bare count. The corresponding UG906 waiver
section is therefore `citation_only`, not executable coverage.

The `qor.csynth_is_estimate` rule binds each supported schedule-stage C-synth
latency/interval observation only when that exact closure proves an
`amd.vitis.csynth_xml` artifact. This includes C-synth `qor.achieved_ii`, but
not a schedule-JSON compiler decision carrying the same predicate.
`qor.target_ii` remains a requested scheduling target and is never classified
as an achieved result or C-synth estimate.

CSim, RTL cosimulation, and dynamic dataflow-profile predicates use that same
observation closure. Their workload identity must match exactly across the
current observation, producing run, and declared managed report; testcase
identities, when present, must also agree. The run must be a successful fresh
Local/SSH tool invocation whose command, toolchain, environment, and stage match
the immutable snapshot manifest. The report kind, declaration, producer run,
live size, and SHA-256 are rechecked before the reserved observation token and
three current-instance identities are minted. A standalone CSim, cosim, or
dataflow-profile artifact cannot prove which parsed observation and workload it
supports, so those artifact targets are explicitly `no_normative`; their typed
predicates and correctness gate remain queryable.

`ARRAY_PARTITION`, `STREAM`, and `INTERFACE` guidance likewise requires the
reserved `directive_operand_linked` marker and a derived operand identity. A
stable ID and a self-consistent `hls.annotates` relation are necessary but are
not proof: before minting either value, retrieval independently replays the
fixed `source.libclang` v4 and `directive.external` v3 parsers under
`hlsgraph.directive_parser_replay.v6` over the exact live snapshot inputs. The
current directive, complete option map, source
spelling hash, anchor, resolved scope/operand entity, annotation, and unique
`directive.requested` observation must match the replay byte-for-byte at the
canonical-record level. A copied scope ID, sibling relation, changed option, or
metadata marker does not qualify. `DEPENDENCE` remains intentionally different:
`hls.annotates` identifies its enclosing loop/function scope, not its operand.
Its two bindings require the separate reserved marker
`dependence_operand_resolved=derived_from_current_dependence_operand_v1` and a
derived operand identity. The retriever mints them only after the current
complete DEPENDENCE directive, its exact named variable entity, a distinct
enclosing scope, their same unique function owner, the current complete source
request observation, the parser replay, and the live snapshot-input hashes all
agree. Metadata and another directive cannot donate either value. Binding
alternatives always use the explicit `{"one_of": [...]}` operator; a bare
array is rejected at the binding boundary.

For the UG1399 PIPELINE, UNROLL, ARRAY_PARTITION, and INLINE effect rules,
replay v6 additionally runs a closed AMD 2024.2 option normalizer. Only a
valid enabled form mints `directive_options_qualified`, a replay-bound options
identity, and one explicit `directive_semantic_mode`; `off`, unknown options,
invalid combinations, and non-array ARRAY_PARTITION operands remain lexical
directive facts but cannot activate those effect rules.

`INTERFACE` has an additional owner closure. Source and external-directive
resolution accept only the unique complete AST `hls.contains` relation from the
configured `hls.kernel` to the named `hls.port`; helper-function ports and
graph-wide same-name fallback are rejected. Replay records and rechecks the
owner entity and containment-relation hashes before retrieval mints
`port_ownership_qualified`, `port_owner_id`, `configured_component_id`, and the
derived ownership identity. Tool name and version are also selected from one
Vitis HLS toolchain record, never from independent manifest-wide unions.

Every `directive.*` observation binding additionally requires
`requested_directive_present=true`. The retriever derives that marker only from
the current exact directive instance and its own unique, complete source-stage
`directive.requested` observation after the same fixed-parser replay succeeds.
All immutable preprocessing, header, Tcl, and config inputs are checked before
and after replay. Missing libclang/compiler context, parser errors, byte drift,
or regex-degraded extraction fails closed. `save_graph()` and caller-written
metadata remain ordinary data operations and cannot issue this capability. A
similarly named directive, sibling scope, tool-status record, or copied metadata
flag cannot supply it.

Tcl and Vitis config inputs use separate conservative literal grammars; neither
parser borrows the other's quoting, comment, substitution, escape, or token
rules. Tcl command names and directive target spellings must match exactly.
Syntax-aware grouping may expose a literal word, but arbitrary braces, quotes,
leading/trailing slashes, or case changes are never stripped or normalized into
a different valid scope. Unsupported or ambiguous spellings produce a
diagnostic and no requested-directive fact.

Extractor-specific fields that are merely name-adjacent to a cited rule are
not auto-bound. In particular, requested clock, available capacities,
trip-count, pipeline-depth, and SLR-crossing predicates are explicitly
`no_normative` for the current citations; they remain ordinary facts with
their own provenance.

## Ranking profile

The built-in `hls.default.v1` profile is deterministic. Its name versions the
ranking algorithm/weights; it is not the stored-contract version. Every trace
also carries `profile_schema_version = "0.3.0"` and a `profile_hash` over that
schema marker plus the complete ranking parameters, so a future wire/schema
change cannot silently reuse the same profile hash:

1. normalize English/Chinese HLS terminology and split snake case, CamelCase,
   acronyms, and compound identifiers;
2. collect exact, qualified-name, prefix, SQLite FTS5, BM25, and bounded fuzzy
   candidates in physically separate corpora for canonical facts/evidence,
   public knowledge, local unreviewed text, and predictions;
3. expand only explicit, profile-whitelisted HLS relations to depth three and
   at most 200 nodes;
4. run typed, directed personalized PageRank with restart `0.25` for at most 25
   iterations;
5. fuse channels with weighted reciprocal-rank fusion (`k=60`) and use stable
   record IDs as the final tie-break; and
6. return an evidence-citing flow spine of at most eight hops.

The facts/evidence corpus alone supplies graph seeds, fact BM25 document
frequencies, fact RRF ranks, and fact score normalization. Installing or
removing a knowledge pack, rebuilding a local sidecar, or enabling predictions
therefore cannot perturb canonical fact ordering. Public knowledge and local
chunks are independently normalized before their presentation-only merge;
predictions remain a third isolated ranker. Generic retrieval adapters may
return only `local_unreviewed` chunks or prediction hypotheses. They cannot
emit public knowledge, facts, or evidence. The only current adapter capability
that projects evidence is the built-in bounded source-snippet adapter, whose
exact class, canonical entity/artifact IDs, authority, provenance, live source
hash, anchor, excerpt digest, and stable record ID are all revalidated.

The adaptive serialized-output ceiling is 13,000, 18,000, or 24,000 characters
according to graph size. Truncation, low confidence, ambiguity, missing
evidence, staleness, and unavailable private content are explicit result fields.
The trace stores a SHA-256 of the query, never the raw query.

An optional semantic channel is available only through the local-only
`hlsgraph.embedders.v1` plugin protocol. HLSGraph does not download a model,
enable network access, or silently change the default profile. Selecting an
embedder is nevertheless a trust decision: it is installed in-process code and
receives each private chunk's plaintext. Its `local_only` declaration is
validated as a protocol contract, not enforced as a filesystem/network/memory
sandbox. During each `embed` call HLSGraph holds a process-wide lock, redirects
OS fd 1/2 to a null sink, restores them afterward, and reduces failures to an
exception class name with no message or cause. Those controls cover the call's
standard descriptors and error surface only; reviewed plugins and ordinary OS
process isolation remain necessary.

## Interfaces

```python
from hlsgraph import Project, RetrievalSpec

project = Project.open("/path/to/project")
result = project.retrieve(RetrievalSpec(
    query="Which loop limits II and what report proves it?",
    snapshot_id=project.status().active_snapshot_id,
))
```

```bash
hlsgraph retrieve --project /path/to/project \
  "Which loop limits II and what report proves it?"
curl "http://127.0.0.1:8000/api/v1/retrieve?q=which+loop+limits+II"
```

REST never returns private bodies. MCP exposes only `explore` by default; set
`HLSGRAPH_MCP_TOOLS=all` before server startup to expose the v0.2 narrow tools
for compatibility.

## Private local evidence

The explicit local sidecar lives under `.hlsgraph/private/knowledge/`. Text,
Markdown, and HTML have bounded built-in parsers. Other formats, including PDF,
require an explicitly selected `hlsgraph.knowledge_parsers.v1` local plugin;
OCR and network access are not enabled by default.

### Direct local PDF indexing

The `pdf` extra registers the built-in entry point named `pdf`. It uses pypdf
locally and does not require converting the manual to Markdown first:

```bash
python -m pip install "hlsgraph[pdf]"
hlsgraph knowledge index --project /path/to/project \
  --path /private/docs/ug1399-vitis-hls-en-us-2024.2.pdf \
  --document-id local.amd.ug1399 --document-version 2024.2 \
  --title "Vitis HLS User Guide 2024.2"
hlsgraph knowledge build-local-index --project /path/to/project \
  --parser pdf --parser-config extraction_mode=layout \
  --parser-timeout-s 60 --max-parsed-chars 8388608
```

`knowledge index` records only path metadata, size, timestamp, and SHA-256.
`build-local-index` revalidates those values, gives the parser verified bytes
plus bounded metadata that omits the source URI/path, and invokes its `parse`
method in a spawn child whose OS stdout and stderr descriptors are discarded
first. The original PDF bytes remain at the user-selected file location; the
private sidecar SQLite stores only the resulting chunks and index metadata.

The child is a timeout/output control, not a security sandbox. A generic parser
plugin is trusted installed code: HLSGraph does not enforce its declared
no-network capability, restrict its filesystem, or provide hard memory
isolation. The current host accepts at most 32 MiB per document and at most
8,388,608 Unicode characters of parser output (about 32 MiB in the worst-case
UTF-8 encoding); parser timeout defaults to 10 seconds and is configurable from
0.1 to 60 seconds. The PDF parser defaults to 4,096 pages and permits an explicit
maximum no greater than 10,000. A timeout, malformed/encrypted PDF, hash change,
excessive page count, excessive output, result-channel failure, or
stdio-containment failure is reported as a sanitized error and does not publish
a partial index. OS fd 1/2 are discarded only during `parse`; this does not stop
trusted code from reopening handles or doing background work. These controls do
not guarantee a peak-memory ceiling, so only reviewed parser plugins should be
installed and selected.

Each text-bearing PDF page becomes a private heading named `PDF page N`; this is
the page anchor returned with an authorized hit. Text extraction does not prove
layout fidelity: figures, equations, reading order, scanned pages, and complex
tables may be incomplete. For those documents, MinerU is an explicit local
plugin/preprocessing choice, not a required dependency or an automatic network
fallback. Its Markdown should preserve page/section markers and stay private.

Recall uses the same unified pipeline as other evidence. The local sidecar
contributes FTS5/BM25 candidates (and optional explicitly configured local
embeddings); weighted RRF combines that channel with graph facts, observations,
diagnostics, and public knowledge rules. Local PDF hits remain in the separate
`local_unreviewed` plane. They cannot create graph edges, change authority, or
become knowledge rules. For example:

```bash
hlsgraph retrieve --project /path/to/project --plane local \
  "What does the guide say about pipeline initiation interval?"
```

That command returns metadata and citations by default. Returning the bounded
page excerpt additionally requires project policy
`privacy.mcp_source_snippets = "bounded"` and the trusted local request flag:

```bash
hlsgraph retrieve --project /path/to/project --plane local \
  --include-private-snippets \
  "What does the guide say about pipeline initiation interval?"
```

A user may instead run an external converter such as MinerU on a lawfully held
PDF and index the resulting UTF-8 Markdown with the built-in parser. That is a
user-side preprocessing path, not a build or runtime dependency of HLSGraph.
The converter output must remain outside the public repository and release
artifacts. For useful evidence links, preserve document/version metadata and
page or section markers in the Markdown; conversion does not make the text a
design fact or a reviewed public rule.

Metadata search is safe by default. A bounded excerpt requires both
`privacy.mcp_source_snippets = "bounded"` in project policy and
`include_private_snippets=true` in the trusted local request. Before returning
content, HLSGraph revalidates project containment, link/reparse status, size,
SHA-256, and source/document identity. Each query reads `chunks.sqlite` once
through a stable file descriptor and verifies that exact byte snapshot against
the manifest. When SQLite deserialize is available, those same bytes are loaded
directly into an in-memory connection. Otherwise HLSGraph writes only the
already verified bytes into a fresh private temporary directory (mode `0700`)
and file (mode `0600`), revalidates file identity, bytes, and SHA-256 before and
after use, opens that staged file read-only/immutable, and backs it up into the
in-memory connection. The mutable user sidecar path is never reopened after the
verified read. Canonical SQLite databases, GraphBundles, exports, generated
reports, wheels, source distributions, and releases never contain sidecar
chunks; the private sidecar SQLite does contain them and must be protected as
private data. REST never returns private chunks.

Private excerpt attempts append a local audit record at
`.hlsgraph/private/retrieval-access.jsonl`. Its schema is deliberately limited
to the content SHA-256, bounded anchor, result code, and returned byte count; it
does not record the query, path, title, source text, or document text. If that
safe log path cannot be validated or written, the private excerpt fails closed.

## Design reference and evaluation

The single-tool interaction and deterministic graph-ranking shape were informed
by CodeGraph at commit
[`286e9ccc2dad45336d4fd67052930322054d64b5`](https://github.com/colbymchenry/codegraph/tree/286e9ccc2dad45336d4fd67052930322054d64b5).
HLSGraph is independently implemented and differs deliberately: its graph is
stage- and authority-aware, propagation is typed and directed, and a software
call graph cannot establish HLS hardware topology.

The public retrieval evaluation harness compares native file tools, the pinned
CodeGraph build, the v0.2 narrow HLSGraph surface, and v0.3 unified retrieval.
Published advantage claims require the frozen evidence scorer and paired
bootstrap gates described by that harness.  A v0.3 Technical/Developer Preview
may explicitly omit that Agent A/B evaluation, but its release notes must make
the preview status clear and cannot claim a comparative performance advantage;
all functional, privacy, knowledge, and formal knowledge-review gates still
apply.
