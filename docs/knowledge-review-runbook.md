# Knowledge review release runbook

The two v0.3 knowledge reviews are release evidence, not ordinary generated
documentation. Run them from two separate clean ext4 checkouts of the same
candidate commit. Windows, drvfs, a dirty tree, a model-enabled network, or a
missing retained raw stream/cache makes the review invalid.

## Preconditions

- Ubuntu 22.04 under WSL2, with each checkout and evidence cache on ext4 (not
  `/mnt/c`, `/mnt/d`, another drvfs mount, or a home directory).
- `codex-cli 0.144.0`, authenticated through a dedicated `CODEX_HOME`.
- No `OPENAI_API_KEY`, `CODEX_API_KEY`, or `CODEX_ACCESS_TOKEN` environment
  variable. The runner supplies a narrow environment and denies every home,
  `CODEX_HOME`, drvfs mount, and the live checkout to model-issued commands.
- A clean committed public candidate containing the passed online citation
  audit at `docs/knowledge-citation-audit-v0.3.json`.
- Two external evidence directories created with mode `0700`. Raw JSONL and
  stderr files are mode `0600`; cache directories and files are respectively
  `0700` and `0600`. Never place these restricted artifacts in the repository,
  wheel, sdist, SBOM, or GitHub Release.

Generate and commit the fixed citation inventory before cloning the review
trees:

```bash
python3 tools/audit_knowledge_citations.py --online \
  --output docs/knowledge-citation-audit-v0.3.json
git add docs/knowledge-citation-audit-v0.3.json
git commit -m "Add v0.3 citation audit evidence"
```

## Preflight without network or model execution

The `preflight` subcommand freezes the real source inventory and prompt
contract. It performs no citation fetch and does not invoke Codex:

```bash
python3 tools/run_knowledge_review.py preflight \
  --protocol hlsgraph.knowledge-review.semantic.v1
python3 tools/run_knowledge_review.py preflight \
  --protocol hlsgraph.knowledge-review.adversarial.v1
```

Record the reported snapshot, implementation, schema, citation-audit, and pack
surface hashes. Any change before or during a review invalidates that run.

## Run two isolated invocations

Create two ext4 clones/worktrees at the same commit and private external
evidence parents. The cache and raw paths must not already exist.

```bash
export CODEX_HOME="$HOME/.codex-hlsgraph-review"
install -d -m 0700 /var/tmp/hlsgraph-review-evidence

cd /var/tmp/hlsgraph-review-semantic
python3 tools/run_knowledge_review.py review \
  --protocol hlsgraph.knowledge-review.semantic.v1 \
  --raw-output /var/tmp/hlsgraph-review-evidence/semantic.codex.jsonl \
  --cache-root /var/tmp/hlsgraph-review-evidence/semantic.cache \
  --pdftotext-command /usr/bin/pdftotext

cd /var/tmp/hlsgraph-review-adversarial
python3 tools/run_knowledge_review.py review \
  --protocol hlsgraph.knowledge-review.adversarial.v1 \
  --raw-output /var/tmp/hlsgraph-review-evidence/adversarial.codex.jsonl \
  --cache-root /var/tmp/hlsgraph-review-evidence/adversarial.cache \
  --pdftotext-command /usr/bin/pdftotext
```

The trusted runner fetches each exact citation before model execution, enforces
HTTPS and the original hostname at every redirect, applies a fixed response
size limit, and writes content-addressed bodies plus parser-bound inspection
text to the external cache. Codex then runs with network disabled, the live
checkout denied, and the frozen cache read-only. The only model-issued command
forms accepted by replay are a complete `head -n COUNT PATH` read and
`sha256sum PATH ...` over manifest-listed source or inspection files.

PDF content is usable only when a controlled extractor produces non-empty text
and records its parser version, executable/argv contract, and contract hash.
Omit `--pdftotext-command` only if the citation inventory contains no PDFs. A
PDF without controlled text derivation is marked unavailable; an unavailable
or incompletely inspected citation cannot produce `approved: true`.

The retained raw JSONL contains the public source command output but replaces
full citation inspection output with a hash/length marker. Raw response bodies
and derived citation text remain only in the private external cache. Unknown
events/tools, partial reads, missing start/completion status, output mismatch,
interpreters, search, writes, command chaining, substitute URLs, or cache/source
mutation fail closed.

Each invocation stages and then publishes its result, content-free normalized
trace, and v2 receipt. Do not hand-edit any of these public artifacts.

## Deterministically seal pack attestations

Copy the semantic and adversarial result/trace/receipt triples into the release
checkout at their fixed `docs/knowledge-review-v0.3.*` paths. Retain both raw
streams and both complete caches externally. Then run the sealer from the
release checkout:

```bash
python3 tools/run_knowledge_review.py seal \
  --semantic-raw /var/tmp/hlsgraph-review-evidence/semantic.codex.jsonl \
  --adversarial-raw /var/tmp/hlsgraph-review-evidence/adversarial.codex.jsonl \
  --semantic-cache /var/tmp/hlsgraph-review-evidence/semantic.cache \
  --adversarial-cache /var/tmp/hlsgraph-review-evidence/adversarial.cache
```

The sealer replays both invocations, requires independent thread/invocation/raw
identities and exact citation agreement, and updates only the excluded
review-attestation fields. It aborts if any semantic pack surface or frozen
implementation input changes. The frozen closure includes the restricted
runner, citation generator, surface helper, and release auditor themselves, so
changing the verifier after review invalidates the seal. Raw/stderr paths must
also remain disjoint from each cache root. Never populate `reviewers`,
`source_hashes`, or `review_evidence` manually.

## Formal release audit

The default audit is the release gate. `--preflight-only` checks only archive
and privacy hygiene and cannot approve a release.

```bash
python3 tools/audit_release.py dist \
  --semantic-review-raw /var/tmp/hlsgraph-review-evidence/semantic.codex.jsonl \
  --adversarial-review-raw /var/tmp/hlsgraph-review-evidence/adversarial.codex.jsonl \
  --semantic-review-cache /var/tmp/hlsgraph-review-evidence/semantic.cache \
  --adversarial-review-cache /var/tmp/hlsgraph-review-evidence/adversarial.cache \
  --eval-identity /path/to/environment.lock.json \
  --static-report /path/to/static-report.json \
  --bootstrap-report /path/to/bootstrap.json \
  --scores /path/to/scores.jsonl \
  --run-set /path/to/run-set.json \
  --release-notes /path/to/release-notes.md
```

The audit loads the retained private caches, replays both raw lifecycle streams,
recomputes every snapshot/prompt/result/trace/receipt/source hash, and validates
the deterministic pack seal. If an exact official locator cannot expose the
cited content, the release remains NO-GO; reachability metadata alone is not a
semantic review.
