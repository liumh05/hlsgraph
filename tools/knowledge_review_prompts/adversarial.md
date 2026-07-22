# HLSGraph public knowledge-pack adversarial activation review

Perform a fresh, isolated red-team review of the current checkout. Do not edit
or consult another review. Your task is to find any way a caller
could activate a public knowledge rule without proving the cited premise.
Inspect every pack, binding, context producer, reserved-token guard, extractor,
and coverage classification. Repeat the supplied pack and implementation
surface hashes exactly.

For citation semantics, inspect the closed citation-to-evidence mapping and
only the frozen derived text associated with each official `evidence_url` in
the supplied cache manifest. `citation_url` remains the human-facing locator;
only `direct.v1` requires equality. AMD KHUB mappings must close
document/version identity and, for topic evidence, TOC/content identity.
Document-root map metadata proves document identity only and cannot replace a
topic body for a rule. The trusted runner performed the evidence fetch before this network-disabled
turn and bound the complete same-host redirect chain, response hash, and parser
identity. Missing/empty text, a binary PDF without a bound parser, an application
shell, reachability metadata, memory, or another version is insufficient;
reject it and set `approved=false`. Do not reproduce documents or long quotes.

Use only the private read-only frozen cache supplied by the caller. The live
checkout, sibling repositories, user files, and all network operations are
unavailable. Native web/search/MCP tools, guessed paths, arbitrary shell
commands, writes, installs, and project-code execution are forbidden. Read each
source row explicitly marked `model_inspection_required=true` and every
available derived citation text in full, one bounded UTF-8 chunk at a time; a
hash-only command is not inspection evidence. Rows marked `integrity_bound_only`
still invalidate the snapshot but are not claimed as model inspected. Every
required chunk's exact, untruncated output and the final result emission must
appear in the normalized trace.

Treat the supplied citation-audit manifest only as the exact reference
inventory, never as semantic proof. Repeat its raw SHA-256 and emit exactly one
`citation_results` row for each manifest `reference_id`. Reject missing, extra,
duplicate, stale-surface, unavailable, or semantically unsupported references.
For every verified rule require the exact locator, version, section,
paraphrase, and applicability checks to be true; document-only rows use `null`
for fields that have no rule semantics.

Pack attestation fields remain `unreviewed` until two invocations approve this
same semantic surface. Do not report that pending state alone as a finding;
instead try to install, select, or activate executable bindings despite the
required `review_ready` gate.

Try at least these attacks:

- inject trusted-looking keys into entity/relation/observation metadata;
- forge source directives, `ANNOTATES` edges, requested observations, scopes,
  options, or operands against unchanged source bytes and bypass fixed parser
  replay/extraction authorization;
- call public SDK/store methods directly with a fabricated successful ToolRun,
  runner identity, request, manifest, staged outputs, or copied execution
  attestation/receipt, including replaying one authorization twice;
- reuse evidence from a sibling instance, other snapshot, run, stage, artifact,
  workload, directive, scope, or operand;
- give one observation multiple anchors, a source artifact different from its
  anchor, a sibling report at the same stage, a duplicate declared output path,
  or parser provenance for different predicate/value/unit bytes;
- exploit missing/duplicate/ambiguous anchors, containment, mappings, producer
  identities, versions, artifacts, or source observations;
- use stale, replay, fake, failed, undeclared, path-replaced, or hash-mismatched
  artifacts and reports;
- confuse requested target values with achieved values or estimates with
  post-route measurements;
- pass arbitrary MLIR/LLVM/CIRCT artifact versions against an exact spec commit;
- make a generic text parser invent a language-spec compatibility claim, or
  place a self-consistent semantic attestation in graph metadata;
- promote LLVM CFG, software calls, Handshake SSA, or a dialect projection into
  HLS hardware topology;
- activate a rule through a generic report container, a bare list constraint,
  a coverage-only citation, an unknown boolean spelling, or incomplete evidence;
- declare binding requirements that do not entail the rule's `condition`, or
  satisfy a condition using context from a different target instance;
- hide an executable binding in citation-only/no-normative coverage, omit a
  rule/binding from section coverage, or omit/add a target relative to the
  versioned supported-target registry;
- perturb fact/evidence BM25 ranks, normalization, graph expansion, or budget
  truncation by adding knowledge/local/prediction text, or make a generic
  adapter emit a trusted fact/evidence/knowledge plane;
- leak prediction/hypothesis or knowledge guidance into fact ranking.

Classify each successful or plausible bypass only through the schema's
controlled severity/code pair. Do not publish finding/evidence/fix prose,
quotations, paths, or any other free text in the result. Use the fixed summary
`approved_no_issues` for approval and `rejected_with_controlled_issues`
otherwise; citation issue values come only from their closed enum. Set
`approved=true` only if all attempted paths fail closed and no material issue
remains. Use protocol ID
`hlsgraph.knowledge-review.adversarial.v1` and the required JSON schema.
