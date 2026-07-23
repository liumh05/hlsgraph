# HLSGraph

HLSGraph is a deterministic information layer for HLS coding agents,
LLM4HLS, and ML4HLS. It turns source and build context, compiler IR,
schedule/binding evidence, verification results, and HLS/implementation
reports into a traceable hardware-architecture graph.

Software call graphs answer “who calls whom.” HLSGraph answers hardware
questions: which processes may run concurrently, how streams and memories
connect them, which directive applies to which scope, what limits initiation
interval, and which result was observed at which tool stage.

HLSGraph v0.3 is a **Technical Preview**. Its complete support boundary is one
HLS kernel/component; public interfaces may still evolve before 1.0.

## Why HLS needs its own graph

A source-level call graph can remain unchanged while `DATAFLOW`, `PIPELINE`,
`UNROLL`, array partitioning, interface configuration, or scheduling decisions
completely change the generated hardware. HLSGraph therefore does not infer a
hardware topology from calls or an LLVM CFG.

Instead, every public result carries:

- an authority class and compilation/tool stage;
- an artifact, source anchor, run, or derivation reference;
- completeness and staleness state; and
- a stable snapshot identity derived from source, configuration, target,
  constraints, and toolchain inputs.

The LLM reads this evidence and proposes hypotheses. It never manufactures
graph edges, QoR, or verification results.

## Truth model

| Plane | What it contributes | What it cannot prove alone |
|---|---|---|
| Source and AST | Functions, loops, variables, calls, source locations, and exact directive scope | Synthesized hardware topology |
| MLIR/HLS IR | Dialect-specific regions, dataflow, buffers, interfaces, widths, and compiler semantics | Semantics not defined by the owning dialect |
| LLVM IR | Typed operations, CFG, memory access, surviving operations, and debug locations | HLS architecture edges or physical implementation |
| Schedule and reports | Requested/applied/achieved values, binding, latency, II, FIFO, utilization, correctness, and timing observations | Results from another stage, target, run, or workload |
| Knowledge packs | Versioned documentation guidance with applicability and citations | Facts about a particular design |
| Predictions | Optional model output with uncertainty and applicability | Real synthesis or verification evidence |

Requested, effective, and achieved values remain separate. CSim/RTL cosim
correctness, resource fit, and post-route timing are three independent gates.
Synthetic fixtures and replay data can exercise the interfaces but never
become fresh vendor-tool truth.

Complete canonical static aggregates from supported MLIR/LLVM domains are
recomputed during indexing and bound to an immutable receipt. Unknown,
ambiguous, injected, or unsupported values remain masked rather than becoming
zero or silently receiving a `complete` label.

## Five-minute start

Python 3.10 or newer is required. From a source checkout:

```bash
python -m pip install -e ".[clang,mcp]"
hlsgraph index --project examples/dataflow_gemm --manifest hlsgraph.toml
hlsgraph status --project examples/dataflow_gemm
hlsgraph retrieve --project examples/dataflow_gemm \
  "why is achieved II above the requested II?"
hlsgraph render --project examples/dataflow_gemm --format html graph.html
```

The included design, IR, and reports are synthetic fixtures. They demonstrate
the complete interface without claiming a successful real implementation.

The standard source frontend is libclang and expects a usable compilation
context, preferably `compile_commands.json`. If that context is unavailable,
the regex scanner must be selected explicitly:

```bash
hlsgraph index --project examples/dataflow_gemm \
  --manifest hlsgraph.toml --degraded
```

Degraded mode is visible in diagnostics and is never presented as equivalent
to a compilation-context-aware AST.

The Python equivalent is:

```python
from hlsgraph import Project

project = Project.open("examples/dataflow_gemm")

result = project.retrieve(
    "Which operation limits II, and what evidence supports that answer?"
)
print(result.snapshot_id, result.confidence)
for item in result.facts:
    print(item.title, item.stage, item.evidence_ids)
```

## GraphRAG and MCP

`Project.retrieve()` and `hlsgraph retrieve` run one deterministic hybrid
retrieval pipeline over separate result planes:

- `facts`: canonical architecture entities and relations;
- `evidence`: observations, diagnostics, artifacts, and derivations;
- `knowledge`: applicable versioned rules and citations;
- `local`: authorized project-local document chunks; and
- `predictions`: opt-in hypotheses, always separated from facts.

Retrieval combines exact and qualified-name matching, SQLite FTS5/BM25, typed
directed graph propagation, and deterministic reciprocal-rank fusion. It
returns evidence IDs, staleness, ambiguity, completeness, truncation, and a
query-hash trace. Optional embeddings are plugin-based, local-only, and off by
default.

Start the read-only MCP server for an indexed project:

```bash
hlsgraph-mcp /absolute/path/to/project
```

The default MCP surface exposes one bounded `explore` tool. It accepts a
natural-language query and returns the same fact/evidence/knowledge separation
used by the SDK and CLI. The legacy narrow tool set remains available only
through explicit compatibility opt-in.

## Data path and interfaces

```text
manifest + source/config hashes + compilation context
        │
        ├── source / AST / MLIR / LLVM extractors
        ├── local or SSH Runner v2 stages
        └── Vitis HLS / Vivado / verification report adapters
                         │
             immutable run ledger + observations
                         │
             canonical architecture projection
                         │
      Python / CLI / REST / MCP / HTML / JSONL / Parquet / PyG
```

The local SQLite bundle stores snapshot and artifact metadata, runs, entities,
relations, observations, derivations, verification results, diagnostics,
variant materializations, and explicit cross-snapshot correspondences.
Private source is referenced by project-relative URI and SHA-256 by default;
its body is not copied into the ledger, REST responses, or ML exports.

Runner v2 uses declared, run-scoped outputs. Local and SSH results are
size/hash checked before an atomic ledger commit. Vendor tools are not bundled,
and importing existing reports does not execute them.

## Knowledge packs and local manuals

Built-in packs contain only project-authored short guidance, document/version
identity, section names, official URLs, applicability selectors, bindings, and
review metadata. They do not redistribute AMD PDFs, complete extracted text,
or local document chunks.

Every pack exposes its review status. Formal machine review validates exact
citations and activation boundaries; deterministic citation and schema checks
remain distinguishable from that semantic review. Inspect the installed state
instead of assuming it:

`machine_repeated_reviewed` means that the same pinned model passed two
physically isolated review protocols over the frozen evidence. It does not
claim human review or cross-model agreement.

The bundled packs in this Technical Preview currently report `unreviewed`;
the complete six-invocation formal review attestation is deferred. This does
not affect retrieval availability, but clients must not present the guidance
as machine-, human-, or cross-model-reviewed.

```bash
hlsgraph knowledge list --project /path/to/project
hlsgraph knowledge coverage --project /path/to/project
```

Users can index manuals they lawfully possess into a private sidecar:

```bash
python -m pip install "hlsgraph[pdf]"
hlsgraph knowledge index --project /path/to/project \
  --path /private/docs/ug1399-vitis-hls-en-us-2024.2.pdf \
  --document-id local.amd.ug1399 --document-version 2024.2
hlsgraph knowledge build-local-index --project /path/to/project \
  --parser pdf --parser-timeout-s 60
```

The original PDF remains in the user-selected location. Extracted
`local_unreviewed` chunks live under `.hlsgraph/private/knowledge/` and never
enter the canonical ledger, bundle export, wheel, or release. Direct PDF
extraction is sufficient for text PDFs; MinerU or another explicitly selected
local preprocessing path is useful for scanned pages, equations, and complex
tables. Local chunks never become reviewed rules or design facts automatically.

## Current boundary

Vitis HLS/Vivado 2024.2 is the first complete adapter family. The schema and
plugin contracts remain vendor-neutral. The following are intentionally not
claimed as complete in v0.3:

- host and full application graphs;
- multiple compute units and platform interconnect;
- DDR/HBM bank connectivity and runtime traces;
- universal native support for every MLIR/HLS dialect; and
- QoR or architecture facts inferred by an LLM.

## Documentation

- [Architecture](https://github.com/liumh05/hlsgraph/blob/main/docs/architecture.md)
- [Schema and truth model](https://github.com/liumh05/hlsgraph/blob/main/docs/schema.md)
- [SDK, CLI, REST, MCP, render, and ML interfaces](https://github.com/liumh05/hlsgraph/blob/main/docs/interfaces.md)
- [Deterministic hybrid retrieval](https://github.com/liumh05/hlsgraph/blob/main/docs/retrieval.md)
- [Knowledge pack policy](https://github.com/liumh05/hlsgraph/blob/main/docs/governance/KNOWLEDGE_PACK_POLICY.md)
- [Privacy and security](https://github.com/liumh05/hlsgraph/blob/main/docs/privacy-and-security.md)
- [Versioning and migrations](https://github.com/liumh05/hlsgraph/blob/main/docs/versioning.md)
- [Formats and upstream references](https://github.com/liumh05/hlsgraph/blob/main/docs/references.md)

## Contributing and license

Read [CONTRIBUTING.md](https://github.com/liumh05/hlsgraph/blob/main/CONTRIBUTING.md),
[SECURITY.md](https://github.com/liumh05/hlsgraph/blob/main/SECURITY.md), and the
[Code of Conduct](https://github.com/liumh05/hlsgraph/blob/main/CODE_OF_CONDUCT.md)
before contributing. Contributions require a Developer Certificate of Origin
sign-off.

HLSGraph is licensed under the
[Apache License 2.0](https://github.com/liumh05/hlsgraph/blob/main/LICENSE).
Third-party and trademark notices are listed in
[THIRD_PARTY_NOTICES.md](https://github.com/liumh05/hlsgraph/blob/main/THIRD_PARTY_NOTICES.md).
