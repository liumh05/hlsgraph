# Related systems and research references

HLSGraph is an independently implemented infrastructure project. The v0.1
distribution does **not** vendor MLIR, LLVM, CIRCT, ScaleHLS, Dynamatic, Vitis,
or Vivado source code or binaries. Names below identify interchange formats,
optional adapters, and compiler or hardware-IR foundations; they do not imply
endorsement or license compatibility beyond each separately obtained
component's own terms.

When publishing results produced with an adapter, cite the upstream system and
the exact tool/version used in addition to HLSGraph's `CITATION.cff`.

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
  Synthesis Framework on Multi-Level Intermediate Representation.” 2021.
  <https://arxiv.org/abs/2107.11673>
- Lana Josipović, Andrea Guerrieri, and Paolo Ienne. “Invited Tutorial
  Dynamatic: From C/C++ to Dynamically Scheduled Circuits.” FPGA 2020.
  <https://doi.org/10.1145/3373087.3375391>
