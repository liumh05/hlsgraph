# Synthetic DATAFLOW golden design

This directory is a public, Apache-2.0 parser and interface fixture. It was
authored for HLSGraph tests and is **not** output captured from an AMD Vitis or
Vivado installation.

The manifest marks every report-like and IR artifact with
`fixture_authority = "synthetic"`. Extractors must preserve that downgrade so
the resulting observations, verifications, and derived gates cannot be
presented as real tool evidence. The fixture deliberately contains:

- a `load -> compute -> store` DATAFLOW-style kernel;
- inline/Tcl directive precedence and an unmet requested II;
- synthetic CSim and RTL-cosim pass records;
- synthetic resource-fit evidence; and
- a negative post-route WNS, so the independent timing gate fails.

`cases/cosim_fail.rpt` is a separate negative parser input and is not part of
the default manifest. See `../../docs/fixtures.md` for the evidence-claim policy.

Do not replace these files with proprietary project source, vendor binaries,
licensed test vectors, or reports whose redistribution rights are unknown.
