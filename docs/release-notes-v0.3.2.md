# HLSGraph 0.3.2 — Technical Preview

This additive release adds an explicit source-structure render view, Vitis HLS
2025.1 full synthesis module reports, and global-variable extraction.

The default `auto` view preserves the existing selection. `hardware` displays
hardware-stage structure. `source` displays AST containment, DATAFLOW regions
and processes, channels, argument bindings, and directive badges. Source views
carry a visible notice: source structure evidence is not hardware topology.
Source calls between design functions are drawn as `CALLS` edges labelled as
software calls, never as hardware instances; vendor and standard-library helpers
(`hls::`, `std::`) are omitted. Bound ports and memories are attached to the
function that declares them. Directive badges keep `off` flags, so
`INLINE off` is not shown as `INLINE`.
Schedule observations can supply metrics through exact source mappings;
source directives alone do not establish achieved performance.

Bottleneck selection no longer picks the design top when another node has a
measurement: the top's latency contains its children's, so the slowest measured
child is marked instead. A design-top violation of its target II is still ranked
first.

`amd.vitis.csynth_full_xml` supplies module latency, interval, pipeline II,
resources, loop rows, and an explicit RTL instance hierarchy. Exact source names
map through `cross.maps_to`; unknown or ambiguous names remain diagnostic.
The existing 32 MiB report limit is unchanged. All new tests use small synthetic
reports and generic names.

Global and namespace variables have file-owned identities and `storage="global"`.
ARRAY_PARTITION pragmas bind only to lexically visible, unique operands,
including tracked headers. Existing non-global identity rules are unchanged.

The package version is 0.3.2. Canonical schema, bundle format and ML feature
schema remain 0.3.0. No stored-data migration is needed. Source extractor version
6 and Vitis report extractor version 2 change the extraction profile and newly
indexed snapshot identities. Historical records retain their original meaning;
report replay requires a matching supported parser version.

This is a Technical Preview. It makes no performance-advantage claims and does
not provide vendor synthesis validation. Release archive and knowledge-review
gates remain applicable; see the [release runbook](knowledge-review-runbook.md).
The bundled knowledge packs remain `unreviewed`; this update does not supply
the external formal knowledge-review evidence required for release approval.
