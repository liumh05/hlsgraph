# HLSGraph public knowledge-pack adversarial review shard

Perform one fresh, isolated activation red-team shard. The deterministic shard
contract appended to this prompt is the sole authority for this invocation's
`shard_id`, source paths, attack assertion IDs, rule-reference IDs, hashes, and
token budget. Attempt exactly that closed assignment. Do not inspect or emit
another shard's assertions or citations, and do not emit document-reference
rows; the deterministic suite verifier aggregates document identity.

Use only the caller-supplied, read-only projected cache. The live checkout,
sibling repositories, user files, private evidence, and network are outside the
allowlist. The Codex shell tool is the only permitted tool, and it may be used
only for the exact bounded chunk-read grammar declared in the cache. Before
emitting any agent message, you MUST invoke one separate shell call for every
`path` in `shard_manifest.files[].chunks[]` and
`shard_manifest.citations[].inspection_chunks[]`. Each complete command must
be exactly `head -n 100000000 PATH`, with `PATH` copied verbatim from the
manifest. Do not quote, batch, chain, omit, or repeat commands. Even when the
evidence requires rejection, all assigned chunks must still be read exactly
once. Do not emit preliminary JSON or a self-correction; emit exactly one
agent message only after all reads complete. Do not edit, install, run project
code, guess paths, or use web, search, MCP, an interpreter, or any other shell
grammar. Read every assigned source chunk and every assigned available
rule-citation chunk in full. A hash-only operation is not content inspection.
Missing, truncated, duplicate, untraced, or compacted required reads make the
shard incomplete and require rejection.

Citation evidence is resolver-specific and fail-closed:

- `direct.sha256.v1` binds exact locator, document identity/version, content
  type, complete size, and body SHA-256.
- `github.raw.document.v1` binds exact repository, commit, path, source size,
  and full-source SHA-256; `github.raw.lines.v1` also binds the assigned rule
  reference, section, line range, and selected-range hash.
- AMD map/topic resolvers bind the declared 2024.2 document identity; map-root
  metadata cannot substitute for the rule's exact topic body.

Reject application shells, reachability-only metadata, incomplete downloads,
unbound parser output, version/section substitutions, memory, or neighbouring
content. A verified rule requires exact locator inspection plus version,
section, paraphrase, and applicability checks. Do not quote source documents or
publish long passages.

The appended `assertion_contract` contains the only attack IDs and meanings
visible to this shard. Treat it as a closed list; do not reconstruct, guess, or
refer to any attack outside it.

Return exactly one JSON object matching
`hlsgraph.knowledge-review.shard-result.v1`. Repeat all supplied identity hashes
exactly. Emit exactly one assertion row for every assigned attack ID and one
citation row for every assigned rule-reference ID, with no missing, extra, or
duplicate row. Use only controlled issue codes and fixed summaries. Do not emit
findings, explanations, paths, quotations, evidence prose, or fixes. Approval
means every assigned attack failed closed, every assigned citation is verified,
and no controlled issue remains.
