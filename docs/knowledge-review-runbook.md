# Knowledge review release runbook

The v0.3 knowledge review is a six-invocation, fail-closed release procedure.
It reviews the same frozen public candidate twice: three semantic shards and
three adversarial shards.  All six invocations use the same pinned model, so a
successful suite is recorded as `machine_repeated_reviewed`; it is not a
cross-model or human review.

This procedure reviews short public rules and their exact citations.  It does
not publish vendor manuals, PDFs, extracted pages, local document chunks, or
model-readable private evidence.

## Fixed formal profile

- Linux/WSL2 on ext4.  A checkout or evidence tree on drvfs (`/mnt/c`,
  `/mnt/d`, and similar mounts) is invalid.
- One clean, committed candidate checkout.  The suite executor must be the
  copy contained in that checkout.
- `codex-cli 0.144.0`, model `gpt-5.6-sol`, reasoning effort `medium`.
- The closed command contract selects a dedicated ChatGPT Responses provider
  with `supports_websockets=false`.  This keeps the authenticated transport on
  HTTP/SSE and avoids losing an otherwise valid six-invocation suite when a
  long-lived WebSocket closes between shards; the provider URL and all
  transport fields are part of the attested argv hash.
- The official self-contained Linux musl Codex ELF, SHA-256
  `901923c1808a151f6926d41d703c17ad48815662cefb1c8d832a052c44271429`.
  The independently checked standalone release asset SHA-256 is
  `725883fc20ab4af3072829aaa0edf6d12c216238f9f7315a6656b950fb05c8bb`.
- `tiktoken==0.13.0`, encoding `o200k_base`, with the complete tokenizer-table
  contract SHA-256 fixed by `tools/knowledge_review_shards.py`.  The formal
  suite reads its BPE table from one explicit frozen offline cache; it never
  downloads tokenizer data during a release run.
- A dedicated authenticated `CODEX_HOME`; API-key environment variables are
  absent.  The model sandbox cannot read `CODEX_HOME`.
- One caller-owned, empty, mode-`0700` external suite work root.  It must be
  disjoint from the checkout, `CODEX_HOME`, and Codex runtime.

The frozen context contract is a 372,000-token window, at most 250,000
accounted visible input tokens per shard, a 122,000-token reserve, and a
32,000-token allowance inside the visible-input ceiling for the runtime
envelope.  Codex is launched with a 300,000-token auto-compaction threshold
whose scope is `total`.  The raw event stream must contain no compaction event;
the four terminal usage fields (`input_tokens`, `cached_input_tokens`,
`output_tokens`, and `reasoning_output_tokens`) are retained as cumulative API
usage and are not misrepresented as instantaneous context length.  Receipts
label `input_tokens + output_tokens` only as a deterministic derived value;
they never present it as a counter reported by Codex.

## What is frozen and what the model can see

The review snapshot integrity-binds the public package, schemas, citation
inventory/evidence map, prompts, suite executor, cache/replay/seal code, release
auditor, and all three knowledge packs.  A byte change invalidates the suite.

Integrity binding does not imply model visibility.  Each invocation receives
only its own generated projection under
`review-projections/v1/<shard>/...`, its own assertion contract, and its own
content-addressed citation chunks.  The three fixed shards are:

1. `knowledge_activation`
2. `ir_semantics`
3. `tool_evidence`

The 38 rule references occur in exactly one shard each.  Another shard's
rules, bindings, coverage rows, assertions, and chunks are absent rather than
merely hidden by prompt instructions.  Full public source files and the full
citation inventory remain integrity inputs but are not mounted as readable
model context.

## Citation acquisition and PDF policy

Before model execution, the trusted runner obtains the semantic protocol's
exact machine-evidence URLs.  Redirect identity, status, media type, charset,
length, and body bytes are checked against the committed evidence map.  AMD
topics are closed against the official KHUB map and pages identities.  GitHub
citations are commit- and byte-pinned, and rules see only their audited line
ranges.

The adversarial full cache is then rebuilt without network access by replaying
the frozen semantic cache.  The release auditor later compares every resolver
response and evidence response, including exact body bytes, across both full
caches.  Model-issued commands have network disabled in all six invocations.

A document-only PDF locator proves document identity, not section meaning.
PDF text may support a future rule only when the rule cites a bounded section
and the runner is given a fixed absolute `pdftotext` executable plus its
SHA-256.  The controlled derivation must succeed and the assigned chunks must
be completely inspected.  PDF files, complete extracted text, MinerU output,
and local sidecar chunks remain outside Git and all release artifacts.

## Prepare the candidate

Generate the online citation inventory before freezing the candidate:

```bash
python3 tools/audit_knowledge_citations.py --online \
  --output docs/knowledge-citation-audit-v0.3.json
```

Resolve and commit the corresponding closed
`docs/knowledge-review-evidence-v0.3.json`.  It contains official identities,
URLs, versions, and byte hashes, never downloaded manual text.  Run the public
test and privacy gates, commit the candidate, then create a fresh ext4 clone.
The formal executor rejects a dirty checkout.

## Prepare the isolated runtime

Example layout (all paths are on ext4):

```bash
export CHECKOUT=/root/hlsgraph-review-v03
export SUITE_WORK=/var/tmp/hlsgraph-review-suite-v03
export CODEX_RUNTIME=/var/tmp/hlsgraph-review-runtime-v03
export CODEX_HOME=/var/tmp/hlsgraph-review-codex-home-v03
export TIKTOKEN_CACHE_DIR=/var/tmp/hlsgraph-review-tokenizer-cache-v03

install -d -m 0700 "$SUITE_WORK" "$CODEX_HOME"
export CODEX_PACKAGE=/path/to/codex-0.144.0-linux-x64.tgz
export CODEX_STAGE="$(mktemp -d)"

printf '%s  %s\n' \
  391a3793d21feff08da2d9132f01107dd56fa5a48a158e23d15c6d56e34f7cb2 \
  "$CODEX_PACKAGE" | sha256sum --check --strict
tar -xzf "$CODEX_PACKAGE" -C "$CODEX_STAGE" \
  package/vendor/x86_64-unknown-linux-musl/bin/codex \
  package/vendor/x86_64-unknown-linux-musl/codex-resources/bwrap

test ! -e "$CODEX_RUNTIME"
install -d -m 0700 "$CODEX_RUNTIME/codex-resources"
install -m 0500 \
  "$CODEX_STAGE/package/vendor/x86_64-unknown-linux-musl/bin/codex" \
  "$CODEX_RUNTIME/codex"
install -m 0500 \
  "$CODEX_STAGE/package/vendor/x86_64-unknown-linux-musl/codex-resources/bwrap" \
  "$CODEX_RUNTIME/codex-resources/bwrap"
printf '%s  %s\n' \
  901923c1808a151f6926d41d703c17ad48815662cefb1c8d832a052c44271429 \
  "$CODEX_RUNTIME/codex" | sha256sum --check --strict
printf '%s  %s\n' \
  77360cb751ccedc5971391444ac86a8a33c15b04d6b4a6fe45f5d25496e62c4c \
  "$CODEX_RUNTIME/codex-resources/bwrap" | sha256sum --check --strict
chmod 0500 "$CODEX_RUNTIME/codex-resources" "$CODEX_RUNTIME"
```

`CODEX_PACKAGE` in the example is the
`@openai/codex@0.144.0-linux-x64` npm platform package; its package-tar SHA-256
is `391a3793d21feff08da2d9132f01107dd56fa5a48a158e23d15c6d56e34f7cb2`.
It is a different container from the standalone release asset identified
above, but both yield the same pinned Codex ELF.  The runtime must contain
exactly `codex` and `codex-resources/bwrap`, with the intermediate directory
shown above and no other entry.  Files may not be symlinks, hard links,
package-manager shims, or writable installations.  The executor freezes both
component hashes and sets the initial process path to
`$CODEX_RUNTIME/codex-resources:/usr/bin:/bin`, so the pinned bundled bwrap is
selected ahead of an unrecorded host version.  `code_mode`, `code_mode_host`,
and `code_mode_only` are explicitly disabled, so tool execution cannot request
an unpinned `codex-code-mode-host`; the fixed direct shell path is used instead.
Codex's ephemeral argv0 aliases remain inside the dedicated `CODEX_HOME`; that
directory is not exposed to the review sandbox, and `auth.json` is still a
negative boundary canary.

Prepare the tokenizer cache once, before the formal run.  The cache must live
on ext4, contain exactly the named table, and remain caller-owned mode `0500`
with a mode-`0400` table:

```bash
test ! -e "$TIKTOKEN_CACHE_DIR"
install -d -m 0700 "$TIKTOKEN_CACHE_DIR"
curl --fail --location --retry 8 --retry-all-errors \
  --output "$TIKTOKEN_CACHE_DIR/fb374d419588a4632f3f557e76b4b70aebbca790" \
  https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken
printf '%s  %s\n' \
  446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d \
  "$TIKTOKEN_CACHE_DIR/fb374d419588a4632f3f557e76b4b70aebbca790" \
  | sha256sum --check --strict
test "$(stat -c %s \
  "$TIKTOKEN_CACHE_DIR/fb374d419588a4632f3f557e76b4b70aebbca790")" \
  -eq 3613922
chmod 0400 \
  "$TIKTOKEN_CACHE_DIR/fb374d419588a4632f3f557e76b4b70aebbca790"
chmod 0500 "$TIKTOKEN_CACHE_DIR"
```

Authenticate the dedicated `CODEX_HOME` before the run.  Do not export
`OPENAI_API_KEY`, `CODEX_API_KEY`, or `CODEX_ACCESS_TOKEN`.

The work root must still be empty when the executor starts.  If a prior run
created anything there, use a newly named empty root; do not reuse or clean an
evidence directory in place.

## Execute exactly six reviews

From the clean checkout:

```bash
cd "$CHECKOUT"
python3 -m tools.execute_knowledge_review_suite \
  --root "$CHECKOUT" \
  --work-root "$SUITE_WORK" \
  --codex "$CODEX_RUNTIME/codex" \
  --timeout-seconds 7200 \
  --fetch-timeout-seconds 60
```

The executor performs, in order:

1. clean-host, ext4, runtime, tokenizer, source, schema, and privacy checks;
2. one online pinned full-cache acquisition for the semantic protocol;
3. one exact offline full-cache replay for the adversarial protocol;
4. three physically isolated semantic Codex subprocesses;
5. three physically isolated adversarial Codex subprocesses;
6. deterministic aggregation, trace/receipt construction, and pair sealing;
7. publication of only the aggregate result, normalized trace, and receipt to
   their fixed public `docs/` paths.

Each private invocation directory retains:

- `raw.jsonl`: exact authoritative Codex stdout;
- `sanitized.jsonl`: deterministic content-redacted derivative;
- `stderr.raw.log` and `stderr.log`: exact stderr and its deterministic
  derivative;
- `process.json`: exact argv, cwd, stdin/stdout/stderr hashes, exit status, and
  command-contract hash;
- `invocation.json`: the closed replay envelope; and
- its own immutable projected cache.

For every returned Codex process, the executor writes exact stdout/stderr,
their deterministic redacted derivatives, and `process.json` before semantic
interpretation.  On a zero exit status it replaces `sanitized.jsonl` with the
verified citation-redacted JSONL before command/replay/static-boundary checks.
Thus nonzero exits and post-process contract failures remain diagnosable
without reading raw content.  A failed diagnostic directory has no
`invocation.json`, aggregate receipt, or pair seal; it is never resumable and
its work root must never be reused.

The work root also contains both full caches, `suite-evidence.json`, and
`pair-seal.json`.  It is private retained release evidence.  Never add any of
it to Git, a wheel, sdist, SBOM, benchmark bundle, or GitHub Release.

The command contract grants read access only to Codex's minimal runtime roots,
the exact frozen Codex+bwrap runtime tree, and the current projected cache.  It
disables networking, user configuration, repository rules, persistence, and
unlisted tools.  The model submits only manifest-listed
`head -n 100000000 PATH` reads; the pinned CLI must report each as the exact
`/bin/bash -lc 'head -n 100000000 PATH'` event wrapper.  Replay unwraps only
that literal form and then verifies output byte-for-byte.  Unknown events,
wrapper or command drift, incomplete chunks, output truncation, writes,
interpreters, search, chaining, source mutation, or cache mutation fail closed.

The raw stream may contain at most 128 exact Codex HTTP reconnect notices when
a proxy interrupts SSE.  Replay accepts only the two pinned HTTP failure
reasons, attempts 1–5, the exact ChatGPT Responses endpoint, and no extra
fields.  Each accepted notice becomes a `transport_retry` summary row, with
its original event index, in the normalized shard trace and remains bound by
the raw-stream hash.  The same stream must still finish
one turn, read every assigned chunk exactly once, emit one valid final result,
and exit zero.  A terminal error, unknown reason or endpoint, retry after the
final result, excessive retries, or incomplete lifecycle remains a hard
failure; known retries do not count as unknown events.

## Audit first, then atomically attest

Do not hand-edit `review_status`, `reviewers`, `source_hashes`, or
`review_evidence`.  The finalizer first invokes the same pure full-evidence
replay used by the release gate.  Only after all six raw streams, both full
caches, six projected caches, six process records, aggregate results, traces,
receipts, pair seal, and pack semantic surfaces validate does it atomically
replace all three pack files:

```bash
python3 -m tools.apply_knowledge_review_suite_attestation \
  --root "$CHECKOUT" \
  --suite-evidence "$SUITE_WORK"
```

Any pre-write failure leaves every pack byte unchanged.  A partial replacement
is rolled back.  Successful output lists only the three attested pack filenames
and their SHA-256 values; private paths or document text are not embedded.

## Formal release audit

Run the complete release audit after attestation.  The normal release inputs
are still required; the v6 review evidence is supplied as one external root:

```bash
python3 tools/audit_release.py dist \
  --knowledge-review-suite-evidence "$SUITE_WORK" \
  --eval-identity /path/to/environment.lock.json \
  --static-report /path/to/static-report.json \
  --bootstrap-report /path/to/bootstrap.json \
  --scores /path/to/scores.jsonl \
  --run-set /path/to/run-set.json \
  --release-notes /path/to/release-notes.md
```

The auditor independently reconstructs the fixed tree and snapshots, validates
the acquisition modes, reproduces the offline full-cache equality, derives
both sanitized streams from raw bytes, validates exact process argv and
stdio, replays every allowed command, rebuilds both aggregate products and the
six-way pair seal, and compares the pack attestation byte-for-byte with the
reconstructed material.  A missing private evidence file or stale hash is a
release NO-GO.

An explicitly labelled Technical Preview or Developer Preview may omit the
Agent A/B evaluation while retaining every archive, privacy, SPDX, knowledge,
and formal knowledge-review gate:

```bash
python3 tools/audit_release.py dist \
  --knowledge-review-suite-evidence "$SUITE_WORK" \
  --technical-preview-without-agent-eval \
  --release-notes /path/to/release-notes.md
```

This opt-in accepts none of `--eval-identity`, `--static-report`,
`--bootstrap-report`, `--scores`, or `--run-set`, including partial evidence.
The release notes must say `Technical Preview` or `Developer Preview` and must
not claim a comparative performance advantage.  This path establishes
functional and knowledge-review readiness only; it provides no Agent A/B
performance approval.  The normal release path above remains unchanged and is
required before publishing any advantage claim.

The older two-call v4 workflow remains accepted only for historical artifact
verification.  It is not the v0.3 formal procedure because the monolithic
input exceeds the fixed review context budget.  `--preflight-only` checks
archive/privacy hygiene only and cannot approve a release.

These receipts provide reproducible integrity evidence for the declared
inputs and execution contract.  They are not a third-party cryptographic
attestation and cannot defend against a maintainer who controls and replaces
the candidate, verifier, and retained evidence together.
