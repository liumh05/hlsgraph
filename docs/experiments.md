# Research experiment bridge

Historical experiment results were generated under pre-0.3 research code and
exported as immutable evidence. A representative contract subset is
revalidated against HLSGraph 0.3; the historical results are not
comprehensively rerun.

This page connects the public information layer to the surrounding GNN, RCD,
agent workflow, deadlock analysis, and paper-reproduction studies. It does not
make these research modules part of the public Python package.

## Result map

| Priority | Experiment | Main frozen result | Public 0.3 alignment |
|---:|---|---|---|
| 1 | HLSGraph M1-M5 | ML-KEM/ML-DSA selected context was 20.5x/23.8x smaller | Clean-wheel `index`, `status`, and `retrieve` smoke |
| 2 | GNN v24 | Held-out-kernel tau 0.724; utilization MAE 0.067 | Bit-exact `legacy-g10-1.v1` export/tensor contract plus checkpoint hash; prediction re-execution remains pending |
| 3 | RCD S1 | Hard subset 0.71 versus 0.00 ablation | Historical aggregate and hash |
| 4 | RCD S2 | Violations 27 to 17; regret cases 8 to 2 | Frozen JSON replay |
| 5 | RCD S3 | Two confirmations versus six candidates; regret and violations both zero | Frozen JSON replay passed; the fresh FeatureView fixture still needs canonical index authorization |
| 6 | Cached agent study | Bad-region rate 17.6% versus 48.3%/42.4%; gated 6.7% | Explicitly labeled cached lookup |
| 7 | Live agent study | Bad-synthesis rate 18.9% versus 37.8%/32.3% | Kept separate from `eval/agent_ab` |
| 8 | Controller P0 | Correctness-first state machine; 31 historical tests | Mock end-to-end contract check |
| 9 | HLSPilot reproduction | 22 included runs; 11 verified; 483 csynth calls | Manifest and aggregate audit only |
| 10 | TimelyHLS reproduction | 22 included runs; 19 verified; 90 csynth calls | Manifest and aggregate audit only |
| 11 | HARP reproduction | v18 tau 0.9479; v20 tau 0.8981 | External-baseline summary with missing parts stated |
| 12 | N1/LightningSim | FIFO provenance 5/5; minimum runnable depth 4 | Historical deterministic fixture comparison |
| 13 | GNN data and ablations | Held-out tau 0.287 to 0.536 | Counts and metrics only; no raw dataset |

## Evidence rules

- `historical_pre_v03` means the result predates the public 0.3 contracts.
- `v03_contract_smoke` means only that a bounded interface check was repeated.
- Fake and replay runners never become fresh vendor-tool truth.
- Checkpoints, private source, licensed datasets, full prompts, and vendor work
  directories are excluded from the public bundle.
- Paper reproductions use a substituted K26/Vitis/Vivado 2024.2 environment;
  they do not claim board runtime, global optimality, or exact paper numbers.

The machine-readable catalog and per-family evidence are in
[`experiments/`](../experiments/README.md). Exact smoke outcomes, including
partial checks and the open RCD fixture migration, are recorded in
[`bridge_checks.json`](../experiments/bridge_checks.json).
