# Third-party notices

HLSGraph is licensed under Apache-2.0. The following components, optional
dependencies, documents, and names are not relicensed by that statement.
The machine-readable release inventory is `sbom.spdx.json`. Its dependency
scope is deliberately limited to files shipped by the base HLSGraph
distribution, vendored components, and mandatory runtime dependencies.
Optional, build, and development dependencies are disclosed separately below;
they are not installed or bundled by the base package and therefore are not
represented as base-package dependencies in that SBOM.

## Vendored browser components

HLSGraph distributions include these files for self-contained graph rendering:

| Component | Included file | License | Upstream |
| --- | --- | --- | --- |
| Cytoscape.js 3.30.2 | `src/hlsgraph/render/vendor/cytoscape.min.js` | MIT | <https://github.com/cytoscape/cytoscape.js/tree/v3.30.2> |
| Eclipse Layout Kernel JavaScript bundle (elkjs) 0.9.3 | `src/hlsgraph/render/vendor/elk.bundled.js` | EPL-2.0 AND Apache-2.0 | [elkjs wrapper source commit `a8304cf`](https://github.com/kieler/elkjs/tree/a8304cf79fde75bc2ab1a89d28320f53f8637436) |

The Cytoscape file retains its MIT license and all embedded copyright notices.
The ELK bundle retains its Eclipse Public License 2.0 SPDX header and embeds
Google's `web-worker` shim under Apache-2.0 (its notice is retained in the
bundle). Complete applicable license texts are included as
`licenses/CYTOSCAPE-MIT.txt`, `licenses/ELK-EPL-2.0.md`, and the repository's
Apache-2.0 `LICENSE`. These notices must remain with redistributions of the
files.

### elkjs corresponding source availability

The source code for the EPL-2.0 portions of the distributed elkjs bundle is
available under EPL-2.0 from the following immutable upstream revisions:

- elkjs 0.9.3 wrapper, build files, and generated bundle: commit
  [`a8304cf79fde75bc2ab1a89d28320f53f8637436`](https://github.com/kieler/elkjs/commit/a8304cf79fde75bc2ab1a89d28320f53f8637436);
- ELK 0.9.1 release baseline: commit
  [`62d5909f96fad541bc101ad52dabaece6b7eab7e`](https://github.com/eclipse-elk/elk/commit/62d5909f96fad541bc101ad52dabaece6b7eab7e);
  and
- the additional ELK change identified by the official elkjs 0.9.3 release,
  ELK PR 955: merge commit
  [`7ca51784e42a24201f29bc13e458728b6fc61cdc`](https://github.com/eclipse-elk/elk/commit/7ca51784e42a24201f29bc13e458728b6fc61cdc).

The [official elkjs 0.9.3 release notes](https://github.com/kieler/elkjs/releases/tag/0.9.3)
describe that source lineage. To reconstruct the upstream build tree, place an
elkjs checkout at the wrapper commit and an `elk` checkout at the PR 955 merge
commit in sibling directories, as required by the
[upstream build instructions](https://github.com/kieler/elkjs/blob/a8304cf79fde75bc2ab1a89d28320f53f8637436/README.md#building),
then run `npm install` and `npm run build` from the elkjs checkout. The ELK
v0.9.1 commit is recorded separately as the release baseline. These instructions
identify the preferred source and build lineage; they do not promise a
byte-for-byte rebuild without also recreating the upstream build environment.

## Runtime, optional, and development dependencies

The only mandatory third-party runtime dependency is `tomli>=2` on Python 3.10.
It is obtained separately from its publisher under MIT and is not bundled by
HLSGraph.

The following optional dependencies are installed only when the corresponding
extra is requested and are not bundled by HLSGraph source or wheel releases:

- LLVM/libclang Python distribution — Apache-2.0 with LLVM Exceptions;
- Model Context Protocol Python SDK 1.x — MIT
  (<https://github.com/modelcontextprotocol/python-sdk/tree/v1.x>);
- Apache Arrow / PyArrow — Apache-2.0;
- PyTorch — a multi-licensed distribution whose exact license expression and
  bundled third-party notices depend on the installed release; consult that
  release's package metadata and license files rather than treating it as a
  single-license dependency;
- PyTorch Geometric — MIT.

The build backend depends on setuptools (MIT). The development-only `dev` extra
contains pytest (MIT) and Coverage.py (Apache-2.0). These build and development
tools are separately obtained and are not part of the HLSGraph runtime
distribution.

Consult the installed package metadata and upstream release for the exact version
and license that you use. Vendor EDA tools, FPGA device files, and license-manager
software are not dependencies distributed by this project.

MLIR/LLVM, CIRCT, ScaleHLS, and Dynamatic source code and binaries are not
distributed by HLSGraph. Their official repositories use Apache-2.0 with LLVM
Exceptions; HLSGraph's references to their IR syntax, dialect names, or research
results do not relicense those projects. See `docs/references.md` for official
source-license links and the specification/version policy.

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
