# Knowledge citation audit

`tools/audit_knowledge_citations.py` audits the metadata and locators in every
built-in knowledge pack. Schema v2 binds each result to the review-excluded
semantic surface of its pack, so later review attestations cannot create a
hash cycle. The default command is offline and read-only:

```console
python tools/audit_knowledge_citations.py --output citation-audit.json
```

It checks required `document_id`, version, section, and citation fields; exact
HTTPS publisher hosts; AMD release paths; Arm static-document locators; full
Git commit pins; matching Git-version identities; and official arXiv records.
The JSON is deterministically ordered and includes a canonical hash of every
manifest field except the hash field itself. It records the generator byte
hash, the path and `review_surface_sha256` of every pack, and one stable
`reference_id` plus `reference_surface_sha256` for every document and rule
citation. Each reference also binds the complete document metadata surface;
rule references additionally bind the complete rule surface. The tool does
not change pack review state or source hashes.

The pack surface deliberately excludes only review attestations, using the
same definition as `tools/knowledge_review_surface.py`. It includes rules,
summaries, applicability, effects, citations, bindings, coverage, and other
semantic metadata. The audit therefore does not use the pack `content_hash`,
which would change when an audit hash is recorded in review metadata.

The manifest also contains one `document_evidence` record for every unique
`document_id@document_version`. Its `evidence_sha256` is reproducible from a
strict allowlist of public metadata only:

- the declared document title, publisher, kind, license note, version, and
  query-free official locator;
- the sorted rule IDs, sections, locator kinds, policy results, and query-free
  citation/fetch locators for that document;
- in online mode only, the corresponding sorted status, content type, byte
  count, response SHA-256, final locator, and verification-result metadata.

Offline records use
`hlsgraph.document-citation-evidence.offline-metadata.v2`; online records use
`hlsgraph.document-citation-evidence.online-fetch-metadata.v2`. Both use
canonical UTF-8 JSON plus SHA-256. Unknown fields, query parameters, response
bodies, retry counts, timestamps, and transport diagnostics are excluded from
the hash input. Stable sorting and de-duplication make equivalent input order
produce the same value. `evidence_sha256_is_document_body_hash` is always
false: this hash identifies the citation-evidence metadata envelope, not a
manual, specification, or webpage body.

An explicit online audit performs one bounded GET for each unique locator after
the offline checks pass:

```console
python tools/audit_knowledge_citations.py --online --timeout 12 \
  --max-bytes 8388608 --max-attempts 2 --output citation-audit-online.json
```

For a v0.3 release candidate, the fixed evidence path is
`docs/knowledge-citation-audit-v0.3.json` and it is generated in online mode.
Release validation can run without network access: it validates the stored
response metadata, canonical manifest hash, generator and pack surfaces, and
the exact document/rule reference inventory. It does not repeat the network
requests. A final evidence file is created only after a successful candidate
audit; the command above does not modify knowledge packs.

Only status, normalized content type, byte count, final locator, SHA-256, and
validation status are retained. Response bodies are never written. Redirects
must remain HTTPS and on the official allowlist. Arm static documents must have
both `application/pdf` content type and PDF magic bytes. AMD FluidTopics pages
are dynamic application shells, so a successful request is deliberately marked
`reachable_locator_only`: it proves that the locator resolves, not that the
cited prose or its meaning was verified. Immutable GitHub locators and official
arXiv records receive similarly narrow reachability labels; semantic review is
a separate knowledge-pack gate.

Consequently, an online AMD FluidTopics document-evidence record is explicitly
labelled `reachable_locator_only`. Its response SHA-256 may identify the
dynamic application-shell bytes seen during that request, but neither that
response digest nor the enclosing `evidence_sha256` is a hash of the cited AMD
manual text.

Transient network or chunked-transfer failures may be retried up to the bounded
`--max-attempts` value (1–3); other HTTP or validation failures are not retried.
The online path prefers an installed `curl` executable for its streaming and
transfer-limit support, and otherwise uses Python's standard-library HTTPS
client. Neither transport receives a filesystem output path.

The command exits with status 0 only when every enabled check passes, 1 for an
audit failure, and 2 for invalid audit configuration. It never downloads a
document unless `--online` is supplied and never indexes or redistributes the
referenced material.
