# Evidence failure cases

These files are synthetic, Apache-2.0-licensed CI fixtures. They are not AMD
tool output and must be imported with `fixture_authority = "synthetic"`.

- `cosim_fail.rpt` exercises an RTL co-simulation correctness failure.
- The main fixture's `reports/post_route_timing.rpt` exercises a post-route
  timing failure (`WNS < 0`).
- `reports/directive_status.json` exercises a declared pipeline request whose
  achieved II differs and whose tool status is `unmet`.
