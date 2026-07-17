# Fixtures and evidence claims

The repository's normal tests run without Vitis HLS, Vivado, an FPGA, SSH, or a
GPU. Public fixtures validate parsers and contracts; they are not experimental
evidence for real tool QoR.

## `examples/dataflow_gemm`

The developer-preview golden design is a small Apache-2.0 HLS C++ pipeline:

```text
load -> FIFO(depth=8) -> compute -> FIFO(depth=16) -> store
```

It exercises:

- `compile_commands.json` and a local `hls_stream` stub for libclang;
- functions, loops, arrays, streams, ports, and explicitly scoped pragmas;
- an external Tcl directive that conflicts with an inline pipeline request;
- Handshake-style MLIR dataflow/buffer semantics;
- LLVM operations, CFG, memory accesses, and debug locations as evidence only;
- normalized Vitis-style csynth, schedule, directive, cosim, and dataflow data;
- normalized Vivado-style post-route timing, utilization, and physical summary;
- workload binding for dynamic cosim/FIFO/stall evidence;
- the three independent gate computations;
- self-contained human rendering and leakage-aware ML export.

Every IR/report-like artifact in the manifest is marked
`fixture_authority = "synthetic"`. Extraction therefore emits
`authority=synthetic` instead of compiler/tool/verification authority. The
status service also refuses to report the overall design as verified when a
gate is supported only by synthetic evidence.

## Failure scenarios

The fixture includes deliberately non-passing evidence:

- `reports/directive_status.json` represents a requested/effective pipeline
  directive whose achieved II differs and whose tool status is `unmet`;
- `reports/post_route_timing.rpt` has negative WNS to exercise an independent
  post-route timing failure;
- `cases/cosim_fail.rpt` is a standalone synthetic RTL-cosim mismatch input for
  correctness-failure tests.

The standalone case is not automatically part of the main manifest. A test or
example that imports it must attach synthetic authority and an explicit workload.

## Running the golden path

```bash
python -m pip install -e ".[dev,clang]"
python -m pytest -q

hlsgraph index --project examples/dataflow_gemm --manifest hlsgraph.toml
hlsgraph status --project examples/dataflow_gemm
hlsgraph explore --project examples/dataflow_gemm compute --depth 2
hlsgraph render --project examples/dataflow_gemm graph.html
```

If libclang is intentionally unavailable, degraded behavior can be exercised
only with:

```bash
hlsgraph index --project examples/dataflow_gemm \
  --manifest hlsgraph.toml --degraded
```

Degraded success is not equivalent to the libclang path. Health/diagnostics and
graph metadata preserve the difference.

## How to interpret fixture results

Valid claims:

- the parser recognized the normalized format;
- IDs, graph hashes, query ordering, and exports are deterministic for the same
  snapshot/profile;
- source bodies were not stored or exported;
- directive scope and requested/effective/achieved states remained separate;
- workload, stage, authority, units, evidence IDs, and failure classes survived
  the pipeline;
- synthetic evidence cannot satisfy the final verified decision.

Invalid claims:

- AMD Vitis/Vivado actually produced these files;
- the sample kernel achieved the recorded II, resources, or latency;
- the sample passed real cosim or fit a real device;
- the sample met or failed real post-route timing;
- a vendor end-to-end flow was reproduced.

Real claims require the original licensed design and a captured `ToolRun` from
the named tool build, target, constraints, workload, and report artifacts. Such
optional acceptance runs belong in a separately controlled environment and are
not required by Windows/Linux CI.

## Adding a public fixture

1. Use synthetic, authorized, or demonstrably sanitized material.
2. Add SPDX headers and record an explicit license in `artifact_paths`.
3. Mark all imitated IR/report artifacts with
   `fixture_authority = "synthetic"`; do not rely on filenames or prose.
4. Remove absolute paths, usernames, hosts, seeds, proprietary vectors, and
   license-server data.
5. Bind dynamic evidence to a workload/testcase.
6. Include the smallest input needed to exercise one semantic contract.
7. Add assertions for provenance and truth separation, not only parsed numbers.
8. Never add a vendor PDF, tool binary, full private report, DCP/netlist, or
   waveform without an explicit redistribution review.

See [privacy and security](privacy-and-security.md) and the
[knowledge pack policy](governance/KNOWLEDGE_PACK_POLICY.md).
