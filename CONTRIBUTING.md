# Contributing to HLSGraph

Thank you for helping build trustworthy infrastructure for HLS agents and ML4HLS.
Contributions are welcome in code, fixtures, documentation, extractors, report
adapters, knowledge rules, and interoperability tests.

By participating, you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).
Security reports follow [SECURITY.md](SECURITY.md), not the public issue tracker.

## Developer setup

Use Python 3.10 or newer:

```bash
python -m venv .venv
python -m pip install -e ".[dev,clang]"
python -m pytest -q
```

The default test suite must run without Vitis, Vivado, an FPGA, a GPU, SSH access,
or a license server. Add deterministic, redistributable fixtures for parser and
runner tests. Put vendor-tool end-to-end checks behind an explicit opt-in marker.

## Before opening a pull request

Keep changes focused and include tests. In the pull request, state:

- the user-visible behavior and compatibility impact;
- the evidence source and tool/document versions involved;
- whether any schema, bundle, plugin, API, or export contract changes;
- the privacy and licensing status of every new fixture or document reference;
- which low-cost tests were run; and
- which vendor-dependent tests were not run and why.

Do not commit private designs, credentials, license files, generated binary
artifacts, vendor binaries, UG PDFs, extracted manuals, or reports whose
redistribution is not authorized.

## Truth-model checklist

Every graph or observation change should answer all of these questions:

1. Is this a declared constraint, static fact, compiler decision, tool
   observation, verification result, physical measurement, deterministic
   derivation, knowledge rule, or prediction?
2. Which immutable artifact, run, stage, tool version, and source/IR anchor support
   it?
3. If dynamic, which testbench or workload produced it?
4. Are requested, effective, and achieved values stored separately?
5. Does every pragma/directive explicitly reference its scope?
6. Does an ambiguous cross-layer mapping remain ambiguous instead of being joined
   by name?
7. Are correctness, resource fit, and post-route timing evaluated independently?

Software calls are not hardware instances, LLVM CFG edges are not HLS dataflow
edges, and predictions are not QoR observations.

## Extractors and plugins

An extractor should return entities, relations, atomic observations, diagnostics,
coverage, and artifact anchors without mutating another extractor's result.
Unknown syntax and mapping ambiguity must produce diagnostics. Do not silently
fall back to a lower-fidelity parser; degraded modes require explicit user intent
and visible health status.

Vendor- and dialect-specific kinds use namespaced identifiers. Keep the canonical
schema vendor-neutral and avoid adding a core dependency solely for one adapter.

## Knowledge rules

Follow [docs/governance/KNOWLEDGE_PACK_POLICY.md](docs/governance/KNOWLEDGE_PACK_POLICY.md).
Rules require an official HTTPS citation, exact document version and section, a
short original paraphrase, narrow applicability, and human review. Never copy the
referenced document body.

## Commit sign-off and DCO

This project uses the [Developer Certificate of Origin 1.1](DCO). Sign every commit
with:

```bash
git commit -s -m "Describe the change"
```

The sign-off certifies that you have the right to submit the contribution under
the project's license. Use your real name and a reachable email address. A pull
request with unsigned commits cannot be merged until the commits are properly
signed off.
