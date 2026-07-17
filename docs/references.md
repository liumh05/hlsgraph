# Related systems and research references

HLSGraph is an independently implemented infrastructure project. The v0.1
distribution does **not** vendor MLIR, LLVM, CIRCT, ScaleHLS, Dynamatic, Vitis,
or Vivado source code or binaries. Names below identify interchange formats,
optional adapters, and compiler or hardware-IR foundations; they do not imply
endorsement or license compatibility beyond each separately obtained
component's own terms.

When publishing results produced with an adapter, cite the upstream system and
the exact tool/version used in addition to HLSGraph's `CITATION.cff`.

## Implementation specifications and version policy

The built-in MLIR and LLVM readers are project-authored text adapters; they do
not embed those compiler projects. Their syntax and evidence semantics are
implemented with reference to these upstream specifications:

- [MLIR Language Reference](https://mlir.llvm.org/docs/LangRef/) for operations,
  SSA values, regions, blocks, attributes, and textual syntax;
- [MLIR Builtin `Location` attributes](https://mlir.llvm.org/docs/Dialects/Builtin/#location-attributes)
  for source and fused-location evidence;
- [LLVM Language Reference](https://llvm.org/docs/LangRef.html) for LLVM IR
  functions, instructions, basic blocks, CFG operands, and memory operations;
- [LLVM Source-Level Debugging](https://llvm.org/docs/SourceLevelDebugging.html)
  for `DIFile`, `DILocation`, and source mapping metadata; and
- [CIRCT Handshake dialect reference](https://circt.llvm.org/docs/Dialects/Handshake/)
  for Handshake operation and channel terminology.

Those web specifications are rolling documents. A URL is not a version pin.
Every imported artifact should therefore retain its producer, exact tool build
or source commit, pass stage, target, and artifact hash. The HLSGraph extractor
version identifies HLSGraph's parser contract; it must not be presented as an
MLIR, LLVM, CIRCT, ScaleHLS, or Dynamatic specification version. A native plugin
must publish the upstream revision and dialect set against which it was tested.

ScaleHLS and Dynamatic are implementation references rather than bundled
dependencies in v0.1:

- ScaleHLS implementation and license:
  [source](https://github.com/UIUC-ChenLab/scalehls) and
  [Apache-2.0 with LLVM Exceptions](https://github.com/UIUC-ChenLab/scalehls/blob/master/LICENSE);
- Dynamatic implementation and license:
  [source](https://github.com/EPFL-LAP/dynamatic) and
  [Apache-2.0 with LLVM Exceptions](https://github.com/EPFL-LAP/dynamatic/blob/main/LICENSE);
- CIRCT implementation and license:
  [source](https://github.com/llvm/circt) and
  [Apache-2.0 with LLVM Exceptions](https://github.com/llvm/circt/blob/main/LICENSE); and
- LLVM/MLIR implementation and license:
  [source](https://github.com/llvm/llvm-project) and
  [Apache-2.0 with LLVM Exceptions](https://github.com/llvm/llvm-project/blob/main/LICENSE.TXT).

Support for an operation name in the experimental text reader is not a claim of
full conformance to any of these projects. ScaleHLS-, Dynamatic-, or
CIRCT-specific semantic projection should be supplied by a separately versioned
plugin and validated against fixtures produced by its pinned upstream revision.

## Compiler and hardware-IR foundations

- Chris Lattner, Mehdi Amini, Uday Bondhugula, Albert Cohen, Andy Davis,
  Jacques Pienaar, River Riddle, Tatiana Shpeisman, Nicolas Vasilache, and
  Oleksandr Zinenko. “MLIR: Scaling Compiler Infrastructure for Domain
  Specific Computation.” CGO 2021, pp. 2–14.
  <https://mlir.llvm.org/pubs/>
- Chris Lattner and Vikram Adve. “LLVM: A Compilation Framework for Lifelong
  Program Analysis & Transformation.” CGO 2004.
  <https://llvm.org/pubs/2004-01-30-CGO-LLVM.html>
- Schuyler Eldridge et al. “MLIR as Hardware Compiler Infrastructure.”
  Workshop on Open-Source EDA Technology (WOSET), 2021.
  <https://woset-workshop.github.io/PDFs/2021/a06.pdf>
- Hanchen Ye, Cong Hao, Jianyi Cheng, Hyunmin Jeong, Jack Huang, Stephen
  Neuendorffer, and Deming Chen. “ScaleHLS: A New Scalable High-Level
  Synthesis Framework on Multi-Level Intermediate Representation.” 2022 IEEE
  International Symposium on High-Performance Computer Architecture (HPCA),
  pp. 741–755. <https://doi.org/10.1109/HPCA53966.2022.00060>.
  Preprint: <https://arxiv.org/abs/2107.11673>.
- Lana Josipović, Andrea Guerrieri, and Paolo Ienne. “Invited Tutorial
  Dynamatic: From C/C++ to Dynamically Scheduled Circuits.” FPGA 2020.
  <https://doi.org/10.1145/3373087.3375391>
