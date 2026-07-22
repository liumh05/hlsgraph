# Security policy

HLSGraph handles source locations, build commands, report paths, tool execution,
and optional SSH backends. Treat a leakage of private source or credentials, path
escape, command injection, unauthorized artifact access, or remote read/write
bypass as a security issue.

## Supported versions

HLSGraph is currently a developer preview. Security fixes are provided for the
latest release and the default development branch. Older `0.x` releases may be
asked to upgrade because pre-1.0 schema migrations are not maintained indefinitely.

| Version | Security support |
| --- | --- |
| Latest `0.x` release | Yes |
| Default development branch | Yes |
| Older `0.x` releases | No |

## Reporting a vulnerability

Use the repository host's private **Report a vulnerability** / security advisory
feature. Include affected versions, a minimal reproduction, impact, and any known
mitigation. If private advisories are unavailable, contact an active maintainer
through a private address listed in their verified project profile. Do not publish
sensitive details in an issue, discussion, pull request, chat log, or fixture.

Maintainers aim to acknowledge a report within three business days and provide an
initial assessment within ten business days. Timelines for a fix and disclosure
depend on severity, downstream coordination, and whether a vendor is involved.

## Sensitive-data guidance

- Redact source, report paths, hostnames, usernames, tokens, SSH configuration,
  license-server information, and proprietary test vectors from reports.
- A GraphBundle must keep private source external by default. Verify that database,
  JSON/Parquet exports, REST/MCP responses, logs, and diagnostics contain no source
  body or secret environment values.
- Canonical SQLite databases, GraphBundles, exports, generated reports, wheels,
  and release artifacts must not contain private document bodies. The separate,
  project-local `.hlsgraph/private/knowledge/chunks.sqlite` sidecar intentionally
  stores extracted chunks from user-owned documents; protect and exclude that
  whole private directory. Original PDF bytes stay at the user-selected source
  path and are not copied into the sidecar. Never attach a vendor PDF, sidecar,
  or extracted documentation to a report.
- Do not run a proof of concept against infrastructure or designs you do not own or
  have permission to test.

## Deployment guidance

- Keep REST services on loopback unless authentication, transport security, and
  network policy are configured by the deployer.
- Keep REST and MCP query surfaces read-only. Tool execution belongs behind an
  explicit SDK/CLI action and an allowlisted project manifest.
- Use argv arrays rather than shell strings, validate project-relative paths, and
  never interpolate model output into a command.
- Pin and review extractor plugins and model code. Embedding/model adapters must
  not execute unreviewed remote code by default.
- Treat an explicitly selected `hlsgraph.knowledge_parsers.v1` parser or
  `hlsgraph.embedders.v1` embedder as trusted, installed code. A parser receives
  verified document bytes; an embedder receives private chunk plaintext. Their
  capability declarations and the parser worker are not an OS security sandbox:
  HLSGraph does not enforce filesystem or network isolation or a hard memory
  ceiling. HLSGraph redirects process fd 1/2 only for the duration of each
  `parse`/`embed` call and exposes a sanitized, body-free call error. That narrow
  guard is not whole-plugin-lifetime containment and cannot stop trusted code
  from deliberately reopening output handles or starting background work.
  Current parser admission controls limit each document to 32 MiB, parser time
  to 0.1--60 seconds (10 seconds by default), extracted output to 8,388,608
  Unicode characters (about 32 MiB in the worst-case UTF-8 encoding), and the
  PDF parser to 4,096 pages by default (configurable up to 10,000). These bounds
  reduce accidental resource use but do not contain malicious installed code or
  guarantee a peak-memory ceiling. Install and select only reviewed plugins.
- Use least-privilege SSH accounts and isolated run directories. HLSGraph does not
  manage or distribute vendor licenses.

This policy covers the HLSGraph project itself. Vulnerabilities in Vitis, Vivado,
LLVM, MLIR, CIRCT, operating systems, or other dependencies should also be reported
to their respective maintainers.
