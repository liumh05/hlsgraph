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

Private snippets are available only through an explicit local SDK/bundle call
that sets `allow_private=True`, identifies an artifact, and requests a bounded
line range. REST and MCP do not expose this operation. This is an authorization
boundary in the API, not encryption of files on disk.

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

ML export refuses `include_source=True` in v0.2. Artifact rows still include
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

MCP tools are read-only with respect to graph facts and predictions. They can
return symbol names, artifact metadata, observations, redacted diagnostic
summaries, and an HTML
or text rendering. Configure the MCP process with the minimum filesystem access
needed for the selected bundle, and do not assume the LLM/client is authorized
for every project on the host.

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

## Plugins

Extractor and runner plugins are executable Python code with the permissions of
the HLSGraph process. Only explicitly named extractor entry points are loaded by
indexing; opening a bundle does not auto-execute installed plugins. Install
plugins from trusted publishers, pin versions/hashes, and review their license
and data handling.

## Knowledge documents

The repository distributes metadata, short project-authored summaries, section
citations, official URLs, and applicability rules—not UG PDFs or extracted full
text. The local knowledge indexer hashes a user-owned document and records
metadata without copying or parsing its contents. Possession and use of the
original document remain the user's responsibility.

## Reporting and fixture hygiene

Public fixtures must be synthetic, authorized, or demonstrably sanitized, with
an SPDX-compatible license and explicit `fixture_authority = "synthetic"` when
they imitate tool artifacts. Remove customer names, absolute paths, usernames,
hostnames, part inventories, test vectors, seeds, and license-server details.
Sanitization changes provenance: a sanitized fixture is useful for parser tests,
not evidence that a real design met QoR.
