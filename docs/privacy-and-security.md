# Privacy and security

HLSGraph is local-first infrastructure, not a data-loss-prevention or sandbox
product. Its defaults reduce accidental source disclosure, but repository and
deployment owners remain responsible for access control, licenses, secrets, and
tool execution.

Please report vulnerabilities according to [SECURITY.md](../SECURITY.md).

## Private source policy

By default, source and testbench artifacts are represented by:

- a project-relative URI;
- SHA-256 and byte size;
- role/kind, license, producer, access, and retention metadata;
- source locations or IR locations used as evidence anchors.

Their full bodies are not copied into SQLite, REST responses, MCP responses, the
human view, or ML exports. `ArtifactRef` is evidence metadata, not a blob.

Private snippets are disabled by default. The local SDK or MCP retrieval path
may return a bounded, anchor-backed excerpt only when project policy sets
`privacy.mcp_source_snippets = "bounded"` and the individual request sets
`include_private_snippets=true`. Before returning it, HLSGraph revalidates
project containment, link/reparse status, size, SHA-256, and the requested
anchor. REST never exposes this capability. This is an authorization boundary
in the API, not encryption of files on disk.

Project-relative paths, symbol names, hashes, commands, diagnostics, report
values, and source spans can themselves be sensitive. Treat `.hlsgraph/` and
all exports as project data even when no source body is present.

Diagnostic messages, remediation guidance, and extension metadata remain raw
only in the trusted local bundle. Public CoreService query results, CLI status,
REST, and MCP expose a positive diagnostic projection: stable IDs, code,
severity, stage, safe evidence anchor, and a `detail_sha256` correlation digest.
They set `detail_redacted=true` and replace the message with fixed generic text;
raw guidance and metadata are omitted. An operator with local bundle access can
look up the diagnostic ID to inspect the original details. This projection is a
disclosure boundary, not a claim that `.hlsgraph/` itself is non-sensitive.

Public source/IR anchor strings are bounded. Embedded host-absolute Windows,
UNC, rooted-Windows, and POSIX paths are replaced at construction by a stable
`redacted.sha256:<digest>` marker that contains none of the original path; a
post-construction mutation remains non-canonical and is rejected at the SQLite
write boundary. Relative locations such as
`loc("src/kernel.cpp":18:5)` and symbolic locations such as `!dbg !4` remain
valid. Adapters must normalize project locations and replace external absolute
locations with an explicit redacted marker before creating an anchor.

## Artifact retention

- `external` (the default for source) means the artifact remains at its project
  location and only metadata is stored.
- `managed` is an explicit request to copy a content-addressed artifact under
  `.hlsgraph/artifacts/`.
- `ephemeral` describes data that should not be retained as a durable bundle
  artifact.

Do not mark private material public merely to make an integration convenient.
CAS publication and SQLite attribution are separate durability boundaries. A
failed ledger transaction can leave an unreferenced immutable CAS entry; it is
not removed automatically because a concurrent process may already reference
the same bytes. Treat cleanup as an explicit, reference-aware maintenance task.
Do not commit `.hlsgraph/`, proprietary reports, private tests/vectors, tool
binaries, DCPs, netlists, waveforms, or vendor documents unless their license
and disclosure status have been independently reviewed.

## SQLite and exports

The ledger is append-oriented and read connections use SQLite read-only/query
mode, but the database is not encrypted. Filesystem permissions are the primary
local control. Backup and deletion policies must cover the ledger, managed
artifacts, renderings, and ML exports.

ML export refuses `include_source=True` in v0.3. Artifact rows still include
URI, hash, size, role, license, and access policy. Review dataset manifests and
licenses before sharing. The self-contained HTML view embeds graph/evidence
metadata and should receive the same review.

## REST deployment

The built-in REST server:

- is read-only;
- binds to loopback by default;
- requires explicit opt-in for a non-loopback address;
- has no built-in authentication, authorization, rate limiting, TLS, or tenant
  isolation.

Do not expose it directly to an untrusted network. For shared deployments, put
it behind an authenticated TLS reverse proxy, pin the intended snapshot, restrict
filesystem access, and apply process/network resource limits. Read-only does not
mean non-sensitive.

## MCP deployment

MCP tools are read-only with respect to graph facts and predictions. The default
surface is the unified `explore` tool; the v0.2 narrow tools require
`HLSGRAPH_MCP_TOOLS=all`. Responses can include symbol names, artifact metadata,
observations, redacted diagnostic summaries, and—only under the two-part policy
above—bounded private excerpts. Configure the process with the minimum
filesystem access needed for the selected bundle, and do not assume the
LLM/client is authorized for every project on the host.

MCP `impact` returns dependency facts only. It deliberately excludes
software-call and LLVM-CFG relations by default and never emits predicted QoR as
fact.

## Tool runners

Manifest stage commands are arbitrary executables supplied by the project
owner. HLSGraph does not sandbox them. Local and SSH runners are disabled by
default, and the CLI requires `--allow-execution`, but this confirmation is not
a substitute for command review.

With local environment inheritance disabled, Windows still receives only the
host `SystemRoot` value required by `CreateProcess`; no `PATH`, credentials, or
other ambient variables are inherited. Its value is represented only by
`bootstrap_environment_hash` in run provenance.

Recommended controls:

1. Keep argv as an explicit list; never interpolate untrusted request text into
   shell fragments.
2. Use a dedicated unprivileged OS account and project working directory.
3. Keep EDA licenses, SSH credentials, and API tokens out of manifests, command
   lines, logs, and committed fixture data.
4. Restrict SSH destinations and host keys outside HLSGraph.
5. Apply timeouts and disk quotas; classify infrastructure/license/timeout
   failures instead of translating them into bad QoR.
6. Treat generated reports and tool logs as potentially proprietary.

Enabled SSH runs are fail-closed unless the request carries the immutable input
artifact manifest, a pinned toolchain/environment hash, and an explicit
`remote_attestation_argv`. The remote command checks relative path, byte count,
and SHA-256 for every input, then hashes the attestation probe's exact stdout and
compares it to the expected environment hash. Missing or mismatched attestation
is infrastructure failure and cannot be tool truth. The probe is project-owned
executable code and must be reviewed; this mechanism does not replace SSH
host-key policy, license isolation, or host-level attestation.

Fake/replay runners and synthetic fixture reports are marked non-tool-truth.
They must never be used to claim a verified implementation.

Tool-truth writes are capability-gated.  The public Python models are
intentionally serializable for audit, but a caller-created `ToolRun`,
`ExecutionAttestation`, report, or metadata dictionary is not authorization.
Only the execution pipeline can issue the one-shot in-memory capability that
the ledger consumes before publishing an attestation and commit receipt.  An
arbitrary `Runner` object passed directly to the SDK is not trusted merely
because it calls itself `runner.local`/`runner.ssh` or advertises matching
capabilities.  Built-in runner instances are registered internally; an
explicitly selected `hlsgraph.runners.v2` entry point is trusted executable code
and receives a separately bound plugin authority.  Fake and replay paths can
never receive either authority.

This boundary prevents ordinary public-SDK object assembly from manufacturing
fresh tool evidence.  It is not a sandbox against malicious Python already
executing inside the HLSGraph process, nor against the operating-system owner
rewriting the SQLite database or CAS directly.  Protect the environment,
installed plugins, database, and artifact directory with normal OS controls;
post-write readers still revalidate the public receipt and live output hashes
to detect accidental or out-of-model corruption.

`ObservationSource` follows the same boundary. It names exactly one canonical
report and commits that report hash and a fixed parser's predicate/value/unit
payload; `artifact_id`, anchor artifact, and source artifact must agree. It is
not a signature. HLSGraph replays the built-in report parser against the live
managed bytes and requires one exact output plus a valid execution receipt, so
calling the public constructor or copying a sibling report cannot create trusted
QoR or verification evidence.

Source directives use a separate ephemeral replay boundary. Directive entities,
`hls.annotates` relations, runless `directive.requested` observations, graph
metadata, and stable IDs are all caller-constructible and therefore cannot
self-attest. Before retrieval emits `directive_source_declaration_qualified`,
`requested_directive_present`, `directive_operand_linked`, or
`dependence_operand_resolved`, it validates the complete immutable snapshot
input closure and reruns only the built-in `source.libclang` v2 plus literal
`directive.external` v1 parsers. The exact spelling hash, options, source anchor,
scope, operand, annotation, and request record must match one unique replayed
declaration. Inputs are revalidated after parsing and proof construction. A
missing parser/runtime, compilation diagnostic, ambiguous mapping, changed
input, sibling scope, or option/operand mismatch withholds every capability.
The regex scanner is an explicit degraded indexing aid and can never authorize
these retrieval markers. Replay hashes are kept only in memory; source text is
not copied into SQLite, bundles, REST, MCP, or ML exports.

## Plugins

Extractor and runner plugins are executable Python code with the permissions of
the HLSGraph process. Only explicitly named extractor entry points are loaded by
indexing; opening a bundle does not auto-execute installed plugins. Install
plugins from trusted publishers, pin versions/hashes, and review their license
and data handling.

## Knowledge documents

The repository distributes metadata, short project-authored summaries, section
citations, official URLs, applicability rules, bindings, and coverage
manifests—not UG PDFs or extracted full text. An explicit private sidecar may
parse a user-owned document into bounded local chunks. Those chunks and optional
local-only embeddings remain under `.hlsgraph/private/knowledge/`; they never
enter the canonical database, REST, ML export, wheel, sdist, or release.
Possession and use of the original document remain the user's responsibility.
Parsers and embedders are explicitly selected trusted installed code. Parsers
receive verified document bytes, and embedders receive private chunk plaintext;
their local-only declarations are protocol checks, not filesystem, network, or
memory sandboxes. HLSGraph redirects OS fd 1/2 and sanitizes the error surface
only during each `parse`/`embed` call. Plugins can still deliberately reopen
handles or start background work, so they must be reviewed and isolated with OS
controls appropriate to the document sensitivity.

Sidecar queries hash a single stable read of the SQLite file. When SQLite
deserialize is available, those same verified bytes are loaded in memory. When
it is unavailable, HLSGraph writes the verified bytes to a fresh mode-`0700`
temporary directory and mode-`0600` file, revalidates identity, bytes, and hash
before and after use, opens only that staged file read-only/immutable, and backs
it up into memory. It never reopens the mutable user sidecar path after the
verified read, so a database swap followed by path restoration cannot change
the queried snapshot.
Authorized source and local-document excerpt attempts are audited only in the
project-local `.hlsgraph/private/retrieval-access.jsonl` sidecar. Records contain
content hashes, anchors, outcome codes, and byte counts—not queries, paths,
titles, or bodies—and an unsafe or unwritable log path prevents disclosure.

## Reporting and fixture hygiene

Public fixtures must be synthetic, authorized, or demonstrably sanitized, with
an SPDX-compatible license and explicit `fixture_authority = "synthetic"` when
they imitate tool artifacts. Remove customer names, absolute paths, usernames,
hostnames, part inventories, test vectors, seeds, and license-server details.
Sanitization changes provenance: a sanitized fixture is useful for parser tests,
not evidence that a real design met QoR.
