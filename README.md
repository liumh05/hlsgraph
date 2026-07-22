# HLSGraph

HLSGraph is deterministic, evidence-backed graph infrastructure for HLS coding
agents, LLM4HLS, and ML4HLS. It turns source and build context, compiler IR,
schedule/binding evidence, verification results, and real HLS/implementation
reports into a traceable hardware-architecture view.

Code call graphs answer “who calls whom.” HLSGraph instead represents hardware
questions: which processes can run concurrently, how values move through streams
and memories, which directive applies to which scope, what limits initiation
interval, and which result was observed at which tool stage.

HLSGraph is an early **v0.3 developer preview**. Schemas are versioned, but public
interfaces may evolve before 1.0.

## Truth boundaries

HLSGraph keeps unlike evidence separate:

- Source and AST anchor functions, loops, variables, calls, and directive scope;
  they do not by themselves establish the synthesized hardware topology.
- MLIR/HLS IR supplies higher-fidelity region, dataflow, buffer, interface,
  bit-width, and schedule semantics when the producing tool exposes them.
- LLVM IR is low-level operation, control-flow, memory-access, and debug-location
  evidence; an LLVM CFG is not presented as an HLS architecture graph.
- Vitis HLS and Vivado reports are stage-specific observations. Requested,
  applied, and achieved values remain distinct.
- Correctness, resource fit, and post-route timing are independent gates.
- Predicted or hypothesized values stay separate from synthesis observations.
- Versioned knowledge rules interpret documentation; they are not facts about a
  design.

The LLM reads the graph and proposes hypotheses. It does not manufacture graph
edges, QoR, or verification results.

## Data flow

```text
project manifest + content hashes + compile_commands.json
        │
        ├── source / AST / MLIR / LLVM extractors
        ├── local or SSH tool runs and immutable run ledger
        └── Vitis / Vivado / verification report adapters
                         │
                atomic observations + diagnostics
                         │
                canonical architecture projection
                         │
        Python SDK / CLI / REST / MCP / HTML / ML export
```

Private source is referenced by project-relative URI and SHA-256 by default. Its
body is not copied into the SQLite ledger, API responses, or ML exports. Source
snippets require an explicit authorized read.

## Install and try the fixture from a source checkout

Python 3.10 or newer is required. For development with the standard libclang
source extractor:

```bash
python -m pip install -e ".[dev,clang]"
python -m pytest -q
```

The Python API is the primary integration point during the developer preview:

```python
from hlsgraph import Project

project = Project.create_from_manifest(
    "examples/dataflow_gemm/hlsgraph.toml",
    force=True,
)
result = project.index()
print(result.snapshot_id, result.graph_hash)

overview = project.explore(view="architecture", depth=2)
answer = project.retrieve("Which process limits II, and what evidence supports it?")
print(answer.confidence, [item.title for item in answer.facts])
project.render("examples/dataflow_gemm/graph.html")
```

The shortest equivalent CLI session is:

```bash
hlsgraph index --project examples/dataflow_gemm --manifest hlsgraph.toml
hlsgraph status --project examples/dataflow_gemm
hlsgraph query --project examples/dataflow_gemm compute
hlsgraph retrieve --project examples/dataflow_gemm "why is achieved II above target?"
hlsgraph render --project examples/dataflow_gemm graph.html
```

The bundled IR and report-like fixture artifacts are synthetic parser inputs,
not real Vitis/Vivado evidence. They are tagged with synthetic authority and
cannot make the overall verification result pass.

If libclang is unavailable, the regex scanner can be selected only through the
explicit degraded path:

```python
result = project.index(degraded=True)
```

Degraded extraction is recorded in diagnostics and must not be treated as
equivalent to a compilation-context-aware AST.

Vendor tools are not bundled and are not required by the normal test suite. Tool
runs use commands explicitly declared by the project owner; importing existing
reports remains supported without executing a tool.

## Interfaces and storage

- The Python SDK creates immutable design snapshots, indexes evidence, queries and
  explores the graph, orchestrates explicitly configured stages, renders views,
  and exports datasets.
- CLI, read-only REST/OpenAPI, and MCP adapters delegate common query semantics
  to the same service; cross-interface conformance is covered by tests.
- Unified retrieval keeps facts, evidence, applicable knowledge, private local
  documents, and opt-in predictions in separate result planes. The default MCP
  surface exposes one bounded `explore` tool; legacy narrow tools require an
  explicit operator opt-in.
- A local SQLite ledger stores snapshots, artifact metadata, runs, entities,
  relations, observations, deterministic derivations, explicit cross-snapshot
  correspondences, action materialization attempts, diagnostics, and
  verification results.
- JSON/JSONL are the baseline interchange formats. Parquet and PyG are
  optional adapters; the core package does not depend on Torch.

Static feature evidence and entity correspondences are opt-in ML tables. They
default to empty, preserve evidence references, reject outcome-shaped feature
inputs, and never resolve ambiguous mappings by name or row order.

Indexing automatically materializes evidence-backed scope features for IR
operation/index histograms, exact loop bounds/trip counts, explicit integer
bitwidths, memory accesses, and software-call targets. Unknown values remain
`null`/masked rather than becoming zero; dependence-distance rows receive a
non-null value only when explicitly proven. These derivations cite entity,
relation, and artifact evidence and never turn software calls or LLVM CFG into
hardware topology.

The fully supported v0.3 unit is one HLS kernel/component. Component/system entities, host code,
multiple compute units, memory-bank connectivity, and accelerator runtime
traces have reserved schema space but are not yet complete collectors.

## Knowledge packs

Built-in packs contain only versioned document metadata, section references,
official URLs, applicability selectors, and short project-authored paraphrases.
They do not include UG PDFs or extracted vendor text. Users may explicitly build
a private, rebuildable FTS sidecar from documents they lawfully possess. Its
chunks and optional local-only embeddings stay under
`.hlsgraph/private/knowledge/` and never enter the canonical bundle, REST, ML
export, wheel, or release. Search returns metadata by default; bounded excerpts
require both a project policy switch and an explicit request.

See [the knowledge pack policy](https://github.com/liumh05/hlsgraph/blob/main/docs/governance/KNOWLEDGE_PACK_POLICY.md).

## Documentation

- [Architecture and current implementation boundary](https://github.com/liumh05/hlsgraph/blob/main/docs/architecture.md)
- [Schema, authority/stage, and three-gate truth model](https://github.com/liumh05/hlsgraph/blob/main/docs/schema.md)
- [SDK, CLI, REST, MCP, human view, and ML interfaces](https://github.com/liumh05/hlsgraph/blob/main/docs/interfaces.md)
- [Unified deterministic retrieval and truth-plane separation](https://github.com/liumh05/hlsgraph/blob/main/docs/retrieval.md)
- [Privacy and security](https://github.com/liumh05/hlsgraph/blob/main/docs/privacy-and-security.md)
- [Versioning, active snapshots, staleness, and migration](https://github.com/liumh05/hlsgraph/blob/main/docs/versioning.md)
- [Format, compiler, and implementation references](https://github.com/liumh05/hlsgraph/blob/main/docs/references.md)
- [Synthetic fixtures and evidence claims](https://github.com/liumh05/hlsgraph/blob/main/docs/fixtures.md)
- [Knowledge pack policy](https://github.com/liumh05/hlsgraph/blob/main/docs/governance/KNOWLEDGE_PACK_POLICY.md)

## Contributing and license

Read [CONTRIBUTING.md](https://github.com/liumh05/hlsgraph/blob/main/CONTRIBUTING.md),
[SECURITY.md](https://github.com/liumh05/hlsgraph/blob/main/SECURITY.md), and the
[Code of Conduct](https://github.com/liumh05/hlsgraph/blob/main/CODE_OF_CONDUCT.md)
before contributing. Contributions require a
Developer Certificate of Origin sign-off.

HLSGraph is licensed under the [Apache License 2.0](https://github.com/liumh05/hlsgraph/blob/main/LICENSE).
Third-party and trademark notices are listed in
[THIRD_PARTY_NOTICES.md](https://github.com/liumh05/hlsgraph/blob/main/THIRD_PARTY_NOTICES.md).
