# HLSGraph 0.3.3 — Technical Preview

This patch release fixes a 0.3.2 regression in Runner v2 declared-output
ingestion.

In 0.3.2 the Vitis report extractor learned to read
`amd.vitis.csynth_full_xml` module hierarchy. A run that declared that report as
an output then failed with `runner.structural_output_requires_reindex`, because
a run must not add structure to its fixed snapshot. In 0.3.1 the same output was
stored without being parsed, so existing projects that already declared it
started to fail after the upgrade.

A run now keeps only the facts about entities and relations its snapshot already
has, drops the new structure together with the facts that depend on it, and
records an informational `runner.structural_output_deferred` diagnostic. The
snapshot is never changed. To see the module hierarchy, add the report to the
manifest artifact set and re-index a new snapshot, as before.

Source indexing is faster on designs with large constant tables: file-scope
initializer lists are no longer walked literal by literal (they cannot hold loops,
directives or recorded calls), and each source file path is resolved once per
translation unit. On a real design with a 950 KB weight header, indexing went
from about 30 s to about 6 s with an identical graph hash.

The package version is 0.3.3. Canonical schema, bundle format and ML feature
schema remain 0.3.0, and extractor versions are unchanged from 0.3.2, so
extraction profiles and snapshot identities are the same.

This is a Technical Preview. It makes no performance-advantage claims and does
not provide vendor synthesis validation. Release archive and knowledge-review
gates remain applicable; see the [release runbook](knowledge-review-runbook.md).
The bundled knowledge packs remain `unreviewed`; this update does not supply
the external formal knowledge-review evidence required for release approval.
