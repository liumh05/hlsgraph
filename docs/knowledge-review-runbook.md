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
  variable. The runner supplies a narrow environment and starts model-issued
  commands from a default-deny filesystem. Only Codex's system-level
  `:minimal` runtime, the exact hashed Codex runtime directory, and the current
  frozen cache are readable; homes, `CODEX_HOME`, drvfs, live checkouts, raw
  evidence, peer caches, and unrelated temporary files are absent from that
  view.
- A clean committed public candidate containing the passed online citation
  audit at `docs/knowledge-citation-audit-v0.3.json` and its closed
  `docs/knowledge-review-evidence-v0.3.json` mapping. The latter binds the raw
  citation-audit SHA-256 and contains exactly one entry per unique human
  citation URL; it contains official identifiers and URLs, never manual text.
  Every identity string is 1--256 characters and rejects ASCII control/newline
  characters; titles remain short identity labels, not a document-text channel.
- One separate private evidence directory and one dedicated cache parent per
  invocation, all created with mode `0700`. Each cache parent must be owned by the invoking
  user and empty before the run. Each evidence directory must likewise be an
  empty, caller-owned `0700` directory dedicated to that one invocation. Raw
  JSONL and stderr files stay there; never put them beside or below a cache.
  Raw JSONL/stderr files are `0600`; after construction, cache directories and
  files are frozen to `0500` and `0400` respectively. Never place these
  restricted artifacts in the repository, wheel, sdist, SBOM, or GitHub Release.
- A dedicated caller-owned, frozen `0500` runtime directory containing exactly
  one direct child: the self-contained, owner-executable Codex binary used by
  both invocations. No other file or subdirectory is allowed. Its exact
  single-file manifest and executable SHA-256 are recorded in each receipt; a
  package-manager shim, hard link, or writable installation is not valid. The
  only accepted executable is the official `rust-v0.144.0` Linux musl ELF with
  SHA-256
  `901923c1808a151f6926d41d703c17ad48815662cefb1c8d832a052c44271429`.
  The official release-asset tar digest independently checked through the
  GitHub API is
  `725883fc20ab4af3072829aaa0edf6d12c216238f9f7315a6656b950fb05c8bb`.

Generate and commit the fixed citation inventory before cloning the review
trees:

```bash
python3 tools/audit_knowledge_citations.py --online \
  --output docs/knowledge-citation-audit-v0.3.json
```

Resolve each AMD human locator through the official KHUB service before the
candidate is frozen. A document root maps to
`/api/khub/maps/{publication_id}`. A topic maps to
`/api/khub/maps/{publication_id}/topics/{content_id}/content?target=DESIGNED_READER`,
with the same map's `/pages` inventory supplying the unique `prettyUrl`,
`tocId`, `contentId`, and topic title. Record that closed identity in
`knowledge-review-evidence-v0.3.json`. Do not store the fetched metadata,
pages response, topic body, PDF, or derived text in Git. Non-AMD references use
`direct.v1` and must keep identical human and evidence URLs.

Validate both files with `preflight`, then commit them together with the closed
evidence-map schema. The evidence map's `citation_audit_sha256` must equal the
SHA-256 of the citation-audit file's original bytes:

```bash
git add docs/knowledge-citation-audit-v0.3.json \
  docs/knowledge-review-evidence-v0.3.json \
  tools/knowledge_review_evidence.schema.json
git commit -m "Add v0.3 citation review evidence map"
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
install -d -m 0700 \
  /var/tmp/hlsgraph-review-semantic-evidence \
  /var/tmp/hlsgraph-review-adversarial-evidence \
  /var/tmp/hlsgraph-review-semantic-cache \
  /var/tmp/hlsgraph-review-adversarial-cache \
  /var/tmp/hlsgraph-review-runtime
# Install the exact self-contained codex-cli 0.144.0 binary here; do not use
# a symlink or package-manager shim.
install -m 0500 /path/to/exact/codex \
  /var/tmp/hlsgraph-review-runtime/codex
chmod 0500 /var/tmp/hlsgraph-review-runtime

cd /var/tmp/hlsgraph-review-semantic
python3 tools/run_knowledge_review.py review \
  --protocol hlsgraph.knowledge-review.semantic.v1 \
  --raw-output /var/tmp/hlsgraph-review-semantic-evidence/semantic.codex.jsonl \
  --cache-root /var/tmp/hlsgraph-review-semantic-cache/cache \
  --codex-command /var/tmp/hlsgraph-review-runtime/codex \
  --pdftotext-command /usr/bin/pdftotext \
  --pdftotext-sha256 "$(sha256sum /usr/bin/pdftotext | cut -d' ' -f1)"

cd /var/tmp/hlsgraph-review-adversarial
python3 tools/run_knowledge_review.py review \
  --protocol hlsgraph.knowledge-review.adversarial.v1 \
  --raw-output /var/tmp/hlsgraph-review-adversarial-evidence/adversarial.codex.jsonl \
  --cache-root /var/tmp/hlsgraph-review-adversarial-cache/cache \
  --codex-command /var/tmp/hlsgraph-review-runtime/codex \
  --pdftotext-command /usr/bin/pdftotext \
  --pdftotext-sha256 "$(sha256sum /usr/bin/pdftotext | cut -d' ' -f1)"
```

The trusted runner fetches each exact machine evidence URL before model
execution. A redirect may repeat the identical URL, but may not change its
scheme, host, port, path, or query. The runner applies a fixed response
size limit, and writes content-addressed bodies plus parser-bound inspection
text to the external cache. For AMD topics it also fetches and validates the
official map metadata and pages inventory, closing public version, logical
document ID, slug, title, and the unique pretty-URL/TOC/content/topic identity.
Map/pages responses are fetched once per publication in an invocation and
referenced as the same content-addressed artifacts by all its topics. A
FluidTopics JavaScript portal shell is unavailable evidence, not document
text. Codex then runs with network disabled and a true
allowlist rather than a full-disk read baseline: `:minimal`, `$CODEX_RUNTIME`,
and `$CACHE` are the only readable roots, and the generated cache is frozen to
directory mode `0500` and file mode `0400`. Cache files and runtime files with
more than one hard link are rejected. The
runner proves the allowed reads, denied checkout/auth/external/peer reads, and
denied cache writes before sending the prompt. The only model-issued command
forms accepted by replay are a complete `head -n COUNT PATH` read and
`sha256sum PATH ...` over manifest-listed chunks only. Each chunk is UTF-8
safe, content addressed, at most 6000 bytes, and binds a contiguous byte range,
parent hash, and exact reconstruction. A source/citation counts as inspected
only after replay sees every chunk's exact output. The Codex command and receipt
fix `tool_output_token_limit=50000`; any CLI truncation therefore still fails
the exact-output replay rather than being mistaken for visibility.

The snapshot intentionally distinguishes two claims. Every frozen source file
is `integrity_bound`: its byte hash contributes to the snapshot and any change
invalidates the review. Only the explicit minimal activation TCB, public packs,
citation inventory/evidence map, and output contract are
`model_inspection_required` and must be read chunk by chunk. Other files are
listed as `integrity_bound_only`; neither the prompt nor receipt claims the LLM
read them. The public manifest and receipt bind this closed inspection-scope
contract.

PDF content is usable only when the fixed absolute `/usr/bin/pdftotext` binary
matches the explicitly supplied SHA-256, runs under the runner's minimal fixed
environment, produces non-empty UTF-8 text, and records controlled parser,
version-output, executable, argv, environment, and contract hashes. Stdout and
stderr are drained concurrently under fixed byte limits; timeout or a limit
breach terminates and then kills the child and exposes only a body-free error.
Omit `--pdftotext-command` only if the citation inventory contains no PDFs. A
PDF without controlled text derivation is marked unavailable; an unavailable
or incompletely inspected citation cannot produce `approved: true`.

The retained raw JSONL contains the public source command output but replaces
full citation inspection output with a hash/length marker. Raw response bodies
and derived citation text remain only in the private external cache. Unknown
events/tools, partial reads, missing start/completion status, output mismatch,
interpreters, search, writes, command chaining, substitute URLs, or cache/source
mutation fail closed.

Each invocation stages and then publishes its controlled result, content-free
normalized trace, and v4 receipt. Approved results have no issues and use only
the fixed summary `approved_no_issues`; rejected results expose controlled
codes, never model-authored finding/evidence/fix prose. Public traces expose a
controlled parser ID plus contract digest, never parser output or arbitrary
version text. The receipt embeds a path-tokenized boundary contract,
the hashed Codex runtime manifest, cache identity, parent policy, canary
outcomes, and a contract hash; it never publishes an absolute private path.
Do not hand-edit any of these public artifacts. These hashes and replay checks
are reproducible integrity evidence for the declared execution inputs. They are
not a third-party cryptographic attestation of OpenAI execution and do not
protect against an already-malicious maintainer who controls all evidence.

## Deterministically seal pack attestations

Copy the semantic and adversarial result/trace/receipt triples into the release
checkout at their fixed `docs/knowledge-review-v0.3.*` paths. Retain both raw
streams and both complete caches externally. Then run the sealer from the
release checkout:

```bash
python3 tools/run_knowledge_review.py seal \
  --semantic-raw /var/tmp/hlsgraph-review-semantic-evidence/semantic.codex.jsonl \
  --adversarial-raw /var/tmp/hlsgraph-review-adversarial-evidence/adversarial.codex.jsonl \
  --semantic-cache /var/tmp/hlsgraph-review-semantic-cache/cache \
  --adversarial-cache /var/tmp/hlsgraph-review-adversarial-cache/cache
```

The sealer replays both invocations, requires independent thread/invocation/raw
identities and exact citation agreement, and updates only the excluded
review-attestation fields. It aborts if any semantic pack surface or frozen
implementation input changes. The frozen closure includes the restricted
runner, citation generator, surface helper, and release auditor themselves, so
changing the verifier after review invalidates the seal. The checkout, both raw
evidence directories, both caches, and the active dedicated `CODEX_HOME` must
be pairwise disjoint. Sealing and release audit require raw files to be
caller-owned `0600` files with one link in caller-owned `0700` parents; they
recheck the link-free parent chain and file/parent identity, including inode,
size, mtime and ctime, across each descriptor read. The runtime host path is
intentionally absent from the public receipt: only the pathless single-file
manifest and the original sandbox boundary are revalidated later. Never populate `reviewers`,
`source_hashes`, or `review_evidence` manually.

## Formal release audit

The default audit is the release gate. `--preflight-only` checks only archive
and privacy hygiene and cannot approve a release.

```bash
python3 tools/audit_release.py dist \
  --semantic-review-raw /var/tmp/hlsgraph-review-semantic-evidence/semantic.codex.jsonl \
  --adversarial-review-raw /var/tmp/hlsgraph-review-adversarial-evidence/adversarial.codex.jsonl \
  --semantic-review-cache /var/tmp/hlsgraph-review-semantic-cache/cache \
  --adversarial-review-cache /var/tmp/hlsgraph-review-adversarial-cache/cache \
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
