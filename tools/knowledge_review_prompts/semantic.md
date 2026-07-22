# HLSGraph public knowledge-pack semantic review

Act as an independent release reviewer. Do not edit files or rely on another
review. Review all three JSON
packs under `src/hlsgraph/knowledge/packs/` together with the actual context
construction and binding guards in `src/hlsgraph/retrieval.py`, relevant
extractors, `src/hlsgraph/knowledge/core.py`, and the knowledge models.

For citation semantics, inspect the closed citation-to-evidence mapping and the
frozen derived text for every official `evidence_url` in the caller-supplied
cache manifest. `citation_url` remains the human-facing locator; only
`direct.v1` requires equality. AMD KHUB mappings must close document/version
identity and, for a topic, TOC/content identity. Document-root map metadata can
verify document identity only and cannot support a rule in place of the
specific topic body. The trusted runner fetched those evidence URLs before this
network-disabled model turn, rejected any redirect chain that left the original host, retained
the response body outside the repository, and bound any text derivation to its
parser identity and hashes. A missing/empty derivation, binary PDF without a
bound parser, application shell, metadata-only audit, model memory, or unrelated
version is insufficient. Mark the affected reference unavailable or rejected
and set `approved=false`. Do not reproduce documents or long quotations.

Use only the private read-only frozen cache supplied by the caller. The live
checkout, sibling repositories, user files, and every network operation are
unavailable. Native web/search/MCP tools, guessed paths, arbitrary shell
commands, writes, installs, and project-code execution are forbidden. Read every
source row explicitly marked `model_inspection_required=true` and every
available derived citation text in full, one bounded UTF-8 chunk at a time; a
hash-only command is not inspection evidence. Rows marked `integrity_bound_only`
still invalidate the frozen snapshot when changed but are not claimed as model
inspected. Every required chunk's exact, untruncated output and the final result
emission must appear in the normalized trace.

The supplied citation-audit manifest is only a deterministic inventory of the
exact current references; it is not semantic proof. Repeat its raw SHA-256 and
emit exactly one `citation_results` row for every manifest `reference_id`, with
no missing, extra, or duplicate row. A document row may use `null` for the
section/paraphrase/applicability fields. A rule row may be `verified` only when
the exact locator was inspected, its declared version and section matched, its
short paraphrase was supported, and its applicability was not broader than the
source. Otherwise use `unavailable` or `rejected`, record the issue, and reject
the overall review.

The frozen inventory after this protocol directly supplies authoritative pack
review-surface, implementation-surface, schema, citation, cache, and per-file
hashes. Repeat the result-schema hashes exactly; do not calculate or guess them.
Review the following invariants:

The pack attestation fields are deliberately left `unreviewed` while this
invocation runs; two approvals are required before they may truthfully be
changed. Do not reject merely because those pending fields are unreviewed.
Instead verify that packs with executable bindings cannot be installed,
selected, or activated until `review_ready`, and assess the review-excluded
semantic surface that the supplied hashes cover.

1. A knowledge rule is guidance, never a design fact or tool observation.
2. Every executable binding proves the cited rule's complete `condition` from
   the current target instance and instance-local evidence. Pack loading must
   reject bindings whose declared requirements do not logically entail that
   condition. Runtime absence, conflict, or ambiguity fails closed; names,
   generic containers, and user-injected metadata do not mint trusted context.
3. Directive rules prove one exact directive instance, exact scope, complete
   source declaration, anchor, snapshot input, and live byte hash. Operand
   directives additionally prove their separate operand identity. A fixed
   source parser replay or authorized extraction receipt, rather than a
   caller-constructed graph and runless observation, establishes this proof.
4. Tool observation and Gate rules prove a fresh real Local/SSH run through an
   internal pipeline-issued, one-use execution authorization and a persisted,
   independently revalidated execution receipt. An ordinary SDK caller cannot
   create that truth by self-asserting runner, request, environment, manifest,
   output, or ToolRun metadata. The proof binds the immutable run manifest,
   declared managed artifact, same run/snapshot/stage/authority, artifact kind,
   live size/hash, and workload identity where applicable. Fake/replay evidence
   cannot activate it.
5. Each run-backed typed observation has one canonical source report. Its
   artifact ID equals its sole anchor artifact, is the unique declared output
   at that run path, and carries parser-issued predicate/value/unit and artifact
   byte provenance. A sibling or second artifact cannot donate evidence.
6. Requested, effective/applied, achieved, C-synthesis estimate, post-synth,
   post-route, correctness, resource fit, and timing remain distinct.
7. MLIR/LLVM/CIRCT rules require a separately declared exact language-spec
   compatibility contract and artifact revision. Text parsers do not guess it.
   LLVM CFG, software calls, and native Handshake SSA are not hardware topology.
   A deterministic object stored in mutable graph metadata is not a trusted
   semantic-attestation origin.
8. Cross-layer mappings are unique, typed, anchored, and explicitly resolved;
   unresolved/ambiguous projections do not become normative edges.
9. Aggregate static-feature rules are recomputed from qualified evidence with
   schema, completeness, provenance, artifact, and origin identities.
10. Fact/evidence candidate generation, BM25 statistics, normalization, graph
    propagation, and truncation are isolated from public knowledge, local
    documents, and predictions. Generic adapters cannot claim a trusted plane.
11. `citation_only` and `no_normative` coverage cannot activate executable
    guidance. Every executable rule and binding is covered by a `rule` entry,
    non-rule entries contain no binding IDs, and target inventory equals the
    versioned supported-target registry. Coverage means every currently
    supported public target is classified, not every page of a manual.
12. Citations, versions, applicability, short paraphrases, and effects are
    mutually consistent with the content you inspected at each exact official
    locator. Do not approve a merely reachable URL as semantic verification of
    vendor prose.

Report a material premise-to-binding gap only through the schema's controlled
severity/code pair; do not place explanations, findings, quotations, evidence,
required fixes, paths, or any other free text in the public result. Use the
fixed summary `approved_no_issues` for approval and
`rejected_with_controlled_issues` otherwise. Citation issue values likewise
come only from the closed enum. Set `approved=true` only when no issue remains;
in that case `issues` must be empty. Use protocol ID
`hlsgraph.knowledge-review.semantic.v1` and the required JSON schema.
