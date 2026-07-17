# Third-party notices

HLSGraph is licensed under Apache-2.0. The following components, optional
dependencies, documents, and names are not relicensed by that statement.
The machine-readable inventory is `sbom.spdx.json`.

## Vendored browser components

HLSGraph distributions include these files for self-contained graph rendering:

| Component | Included file | License | Upstream |
| --- | --- | --- | --- |
| Cytoscape.js 3.30.2 | `src/hlsgraph/render/vendor/cytoscape.min.js` | MIT | <https://github.com/cytoscape/cytoscape.js/tree/v3.30.2> |
| Eclipse Layout Kernel JavaScript bundle (elkjs) 0.9.3 | `src/hlsgraph/render/vendor/elk.bundled.js` | EPL-2.0 AND Apache-2.0 | <https://github.com/kieler/elkjs/tree/0.9.3> |

The Cytoscape file retains its MIT license and copyright header. The ELK bundle
retains its Eclipse Public License 2.0 SPDX header and embeds Google's `web-worker`
shim under Apache-2.0 (its notice is retained in the bundle). The immutable tag
links above provide the corresponding source releases. Complete applicable
license texts are included as `licenses/CYTOSCAPE-MIT.txt`,
`licenses/ELK-EPL-2.0.md`, and the repository's Apache-2.0 `LICENSE`. These notices
must remain with redistributions of the files.

## Runtime, optional, and development dependencies

These packages are obtained separately from their publishers and are not bundled
by HLSGraph source releases:

- tomli (Python 3.10 compatibility dependency) — MIT;
- LLVM/libclang Python distribution — Apache-2.0 with LLVM Exceptions;
- Model Context Protocol Python SDK 1.x — MIT
  (<https://github.com/modelcontextprotocol/python-sdk/tree/v1.x>);
- Apache Arrow / PyArrow — Apache-2.0;
- PyTorch — BSD-3-Clause;
- PyTorch Geometric — MIT;
- pytest — MIT; and
- Coverage.py — Apache-2.0.

Consult the installed package metadata and upstream release for the exact version
and license that you use. Vendor EDA tools, FPGA device files, and license-manager
software are not dependencies distributed by this project.

## Governance texts

`CODE_OF_CONDUCT.md` is adapted from Contributor Covenant 2.1, licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). `DCO` is the
Developer Certificate of Origin 1.1 published by The Linux
Foundation and is reproduced verbatim under its stated copying permission.

## Documentation references and trademarks

Knowledge packs contain project-authored paraphrases plus citations to official
documentation. Referenced guides, PDFs, and extracted text are not distributed.
Their copyrights remain with their publishers.

AMD, the AMD Arrow logo, Vitis, and Vivado are trademarks of Advanced Micro
Devices, Inc. LLVM, MLIR, CIRCT, Python, GitHub, and other names may be trademarks
of their respective owners. Their use is descriptive and does not imply
affiliation, sponsorship, or endorsement.
