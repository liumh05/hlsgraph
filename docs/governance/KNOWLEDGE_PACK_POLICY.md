# Knowledge pack policy

HLSGraph knowledge packs provide versioned interpretation rules for tools,
standards, and vendor documentation. They are general guidance, not observations
about a particular design.

## Allowed material

A published pack may contain:

- a stable document identifier, version, title, publisher, and official HTTPS URL;
- a section or command name sufficient to locate the cited material;
- applicability selectors such as vendor, tool, version, dialect, and stage;
- a short, original paraphrase of one narrowly scoped rule; and
- machine-readable consequences for HLSGraph's truth taxonomy.

The rule must bind `document_id`, `document_version`, and `section`. A rule for one
tool release must not silently apply to another release. Applicability matching is
fail-closed: a tool- or stage-specific rule is not selected when that context is
unknown.

A binding may select a rule only when the matched target instance itself, or
its required instance-local context, establishes the rule's `condition`. A
generic report container must not activate a field-specific rule merely because
it can sometimes carry that field, and a gate-conditioned rule must bind only a
qualified gate target. If the public schema cannot express the required link
(for example, a report does not identify one resolved directive scope), the
target is classified as `no_normative` instead of inferring the missing link.
Alternative scalar constraints in `required_context` must use the explicit
`{"one_of": [...]}` operator; a bare JSON array is not a binding operator and
is rejected. Tool-observation bindings must require a reserved evidence token
derived from the current observation's run, declared managed artifact, stage,
authority, and live byte identity whenever such a public closure exists.
Workload-scoped observations must close the same workload across the
observation, run, and report, and must reject conflicting testcase identities.
A directive-observation binding must separately prove that the same exact
directive instance has its own complete, source-anchored requested or selected
declaration; a metadata flag is never that proof.

Coverage is measured over HLSGraph's declared extractor and contract surface,
not over every sentence or chapter in a referenced manual. A complete coverage
manifest means every supported target is bound or explicitly classified; it
does not mean a vendor PDF was republished or exhaustively re-authored.

Coverage is also an executable contract, not a best-effort checklist. Each
manifest names an independently versioned `target_registry_version`; its
`target_inventory` must equal the canonical registry for its
`coverage_scope`, with neither missing nor extra targets. Every
`KnowledgeRule` and every `KnowledgeBinding` must be referenced exactly once by
a `rule` coverage entry, and a binding must appear under its own rule.
`citation_only`, `not_applicable`, and `deferred` entries may not carry rule or
binding IDs. Serialized packs must state these IDs and the registry version
explicitly; loading never reconstructs them from nearby rules or targets.

## Prohibited material

Do not commit or distribute vendor PDFs, screenshots, extracted pages, OCR output,
large quotations, document chunks, embeddings of document text, or reconstructed
manuals. Do not use a knowledge rule to assert a design-specific fact or QoR value.

The canonical bundle records only a user-owned document's URI, SHA-256 digest,
size, media type, modification timestamp, document identity, and optional
official URL. An explicitly requested private sidecar may parse supported local
documents into bounded chunks under `.hlsgraph/private/knowledge/`. Those chunks
and optional local embeddings never enter the canonical SQLite database, bundle,
REST response, ML export, wheel, sdist, or release. Search is metadata-only by
default; a bounded excerpt requires a project authorization switch and an
explicit request. No indexer downloads a document or model automatically.

## Review requirements

Every new or changed rule requires either a human review or a truthfully
declared machine review mode. The review must check:

1. the official URL, document version, and section identity;
2. that the paraphrase is accurate, short, and independently worded;
3. that applicability is no broader than the cited guidance;
4. that the rule cannot be confused with a tool observation or measurement; and
5. that no copyrighted document body or confidential design data is present.

`human_reviewed` is reserved for an identified human review. Machine review has
two deliberately different statuses:

- `machine_repeated_reviewed` means two isolated invocations of the same pinned
  model agreed. The manifest must state that the model family was not diverse.
- `machine_cross_reviewed` means two isolated invocations from distinct model
  families agreed.

Both statuses require source/citation evidence hashes and no unresolved
conflict. Repeated same-model review must never be presented as cross-model or
human review.

Classification completeness is not review approval. A pack remains
`unreviewed` until the declared review invocations have actually completed and
their evidence has been recorded. Packs with executable bindings may be
installed or selected only when their immutable coverage and installed
inventory both say `review_ready`. The catalog, SQLite installation boundary,
and retrieval path each enforce this independently, so a legacy or directly
injected binding row cannot bypass review. An unreviewed pack with no bindings
may still provide citation metadata and short paraphrases for lexical search;
that material cannot activate a rule for a design target.

Breaking semantic changes require a new rule ID or document version. Corrections
that alter applicability or effect must be called out in the pull request.

The metadata-only [citation audit](../knowledge-citation-audit.md) provides the
mechanical URL, version-pin, publisher-host, and optional bounded reachability
checks. Reachability is not semantic review: in particular, an AMD FluidTopics
application shell is recorded only as a reachable locator, never as verified
manual text.

AMD, Vitis, Vivado, LLVM, MLIR, CIRCT, and other names remain the property of their
respective owners. Referencing a document does not imply endorsement.
