# PatchTST/42 reproduction results

Two datasets, four prediction lengths each, one run per configuration (seed 2021).
Reference: Table 3, supervised multivariate PatchTST/42, ICLR 2023.
Metrics are on standardized data, matching the upstream evaluation protocol.

| Dataset | Horizon | MSE | Paper MSE | Delta | MAE | Paper MAE | Minutes |
|---|---:|---:|---:|---:|---:|---:|---:|
| ETTh2 | 96 | 0.274209 | 0.274 | +0.000209 | 0.336070 | 0.336 | 4.6 |
| ETTh2 | 192 | 0.337964 | 0.339 | -0.001036 | 0.378030 | 0.379 | 4.9 |
| ETTh2 | 336 | 0.330770 | 0.331 | -0.000230 | 0.380272 | 0.380 | 4.9 |
| ETTh2 | 720 | 0.378991 | 0.379 | -0.000009 | 0.421567 | 0.422 | 5.4 |
| ETTm2 | 96 | 0.164143 | 0.165 | -0.000857 | 0.254254 | 0.255 | 6.8 |
| ETTm2 | 192 | 0.220445 | 0.220 | +0.000445 | 0.292008 | 0.292 | 7.0 |
| ETTm2 | 336 | 0.277860 | 0.278 | -0.000140 | 0.329397 | 0.329 | 6.7 |
| ETTm2 | 720 | 0.367486 | 0.367 | +0.000486 | 0.384898 | 0.385 | 7.1 |

## Verification and limitations

All eight saved predictions/targets passed shape, finite-value and independently recomputed MSE/MAE checks. Best checkpoints and full training configs are retained.
Only one seed was run; these results do not measure variation across seeds. This is not a full-paper reproduction.
Python/PyTorch and worker-count differences from the upstream setup are documented in README.md and environment/.
The upstream evaluator drops the final incomplete test batch. Metrics should be compared using the same evaluation protocol.
All hyperparameters were fixed before training. Test scores were not used to tune or select a rerun.
