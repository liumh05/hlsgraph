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
- Local document indexes may contain metadata and hashes only. Never attach a
  vendor PDF or extracted documentation to a report.
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
- Use least-privilege SSH accounts and isolated run directories. HLSGraph does not
  manage or distribute vendor licenses.

This policy covers the HLSGraph project itself. Vulnerabilities in Vitis, Vivado,
LLVM, MLIR, CIRCT, operating systems, or other dependencies should also be reported
to their respective maintainers.
