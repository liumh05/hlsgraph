# HLSGraph experiment bridge

This directory exposes small, immutable summaries of experiments that predate
the public HLSGraph 0.3 package. The summaries do not convert historical
results into fresh HLSGraph 0.3 tool observations.

`catalog.json` is the machine-readable index. Files below `aggregates/` contain
only the minimum metrics, protocol notes, and limitations needed to trace each
public statement. Large checkpoints, licensed datasets, private source,
complete prompts, and vendor-tool work directories are excluded.

Representative contract checks may be repeated against the hash-locked public
0.3 wheel. Those checks establish interface compatibility only; they do not
re-run or strengthen the historical performance claims.

`bridge_checks.json` records the low-cost checks performed for this bridge.
Each check states whether it passed, was only partial, or exposed a migration
item. In particular, fake and replay runners remain non-truth, and a partial
GNN check must not be read as a repeated model-quality result.

`reproduction_manifest.json` is the sanitized public binding for the paper
reproductions. `traces/controller_fake_smoke.json` is the only selected trace;
it is synthetic and carries no tool-truth or QoR claim.
