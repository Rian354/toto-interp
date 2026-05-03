# PR2 Operational FNO Refit & Aggregation - Issue 3 Resolution

## Status
**Complete**: All 5 seeds (42–46) have been refit and aggregated. Paper-ready numbers, curves, and efficiency metrics are available in `issue3_aggregate/`.

## Root Causes
1. **Outdated cluster `fno.py`**: The cluster version only wrote `artifact_metadata["state_dict"]` and `["config"]`, missing history/timing data needed for aggregation (the local laptop repo already had the full version).
2. **`load_probe_best_per_label` metric mixing**: If any row had `test_accuracy`, all groups used it, causing continuous labels (with all-NaN accuracy) to be dropped from timing/curves. Fixed by selecting `test_r2` vs `test_accuracy` based on `task_type` per group, and applying the same rule when building cost rows.

## Changes Deployed
- Updated `toto_interp/fno.py` on the cluster to include history, timing, and FLOPs tracking.
- Slurm jobs refit FNO from cached activations for all seeds 42–46 using `slurm/refit_fno_issue3_curves.sh`, then ran `aggregate_pr2_operational.py` with `--seeds 42 43 44 45 46`.
- Synced all aggregates and probe directories to local machine (`/Users/saurabhatri/Dev/toto-interp/runs/pr2_operational_issue3/`).

## Local Artifact Paths
| Artifact | Path |
|---------|------|
| Raw epoch curves (all seeds) | `runs/pr2_operational_issue3/issue3_aggregate/fno_learning_curves.csv` |
| Mean ± CI by epoch | `runs/pr2_operational_issue3/issue3_aggregate/fno_learning_curves_mean_ci.csv` |
| Timing / FLOPs (all arms) | `runs/pr2_operational_issue3/issue3_aggregate/probe_timing_raw.csv` |
| Timing mean ± CI | `runs/pr2_operational_issue3/issue3_aggregate/probe_timing_mean_ci.csv` |
| Cost efficiency analysis | `runs/pr2_operational_issue3/issue3_aggregate/cost_efficiency_mean_ci.csv` |
| FNO checkpoints (+ metadata) | `runs/pr2_operational_issue3/operational_fno_seed{42-46}/probes/artifacts/*.pt` |
| Per-seed matched results | `runs/pr2_operational_issue3/issue3_aggregate/pr2_issue3_matched_per_seed.csv` |
| Matched mean ± CI | `runs/pr2_operational_issue3/issue3_aggregate/pr2_issue3_matched_mean_ci.csv` |

`aggregate_pr2.json` lists all cost output paths.

## Aggregated Results (All 5 Seeds: 42–46 Averaged)

### Table 1: All Arms Comparison (FNO vs Pretrained LP vs Random Init LP)

| label | task_type | arm | train_time_s (mean ± CI) | test_metric (mean ± CI) | approx_total_flops |
|-------|-----------|-----|------------------------|------------------------|-------------------|
| cardinality_bucket | categorical | FNO | 12.31 ± 3.69 | 0.9692 ± 0.0043 (acc) | 1.84e+12 |
| cardinality_bucket | categorical | pretrained_lp | 63.91 ± 9.53 | 0.7884 ± 0.0182 (acc) | — |
| cardinality_bucket | categorical | random_init_lp | 8.79 ± 4.19 | 0.6770 ± 0.0331 (acc) | — |
| coordination | continuous | FNO | 11.32 ± 2.33 | — | 1.84e+12 |
| coordination | continuous | pretrained_lp | 0.81 ± 0.19 | 0.0653 ± 0.0277 (R²) | — |
| coordination | continuous | random_init_lp | 1.20 ± 0.86 | -0.0415 ± 0.0113 (R²) | — |
| current_burstiness | continuous | FNO | 11.21 ± 2.26 | — | 1.84e+12 |
| current_burstiness | continuous | pretrained_lp | 5.14 ± 2.18 | — | — |
| current_burstiness | continuous | random_init_lp | 5.62 ± 2.51 | — | — |
| current_sparsity | continuous | FNO | 13.68 ± 5.09 | — | 1.84e+12 |
| current_sparsity | continuous | pretrained_lp | 0.78 ± 0.15 | — | — |
| current_sparsity | continuous | random_init_lp | 1.99 ± 2.55 | — | — |
| domain | categorical | FNO | 10.26 ± 3.55 | 0.4706 ± 0.0157 (acc) | 1.47e+12 |
| domain | categorical | pretrained_lp | 21.48 ± 12.61 | 0.5326 ± 0.0182 (acc) | — |
| domain | categorical | random_init_lp | 69.84 ± 51.26 | 0.4071 ± 0.0131 (acc) | — |
| frequency_bucket | categorical | FNO | 18.17 ± 13.39 | 0.7884 ± 0.0444 (acc) | 1.84e+12 |
| frequency_bucket | categorical | pretrained_lp | 6.63 ± 1.75 | 0.8582 ± 0.0139 (acc) | — |
| frequency_bucket | categorical | random_init_lp | 13.85 ± 4.48 | 0.6770 ± 0.0331 (acc) | — |
| future_burstiness | continuous | FNO | 10.99 ± 1.91 | — | 1.84e+12 |
| future_burstiness | continuous | pretrained_lp | 2.72 ± 2.40 | — | — |
| future_burstiness | continuous | random_init_lp | 6.39 ± 0.59 | — | — |
| future_sparsity | continuous | FNO | 11.07 ± 1.98 | — | 1.84e+12 |
| future_sparsity | continuous | pretrained_lp | 0.76 ± 0.15 | — | — |
| future_sparsity | continuous | random_init_lp | 2.16 ± 2.56 | — | — |
| metric_type | categorical | FNO | 11.73 ± 2.35 | 0.3521 ± 0.0376 (acc) | 1.77e+12 |
| metric_type | categorical | pretrained_lp | 25.58 ± 14.34 | 0.5940 ± 0.0141 (acc) | — |
| metric_type | categorical | random_init_lp | 18.20 ± 5.37 | 0.4251 ± 0.0121 (acc) | — |
| shift_risk | continuous | FNO | 10.82 ± 1.83 | — | 1.84e+12 |
| shift_risk | continuous | pretrained_lp | 1.66 ± 1.92 | -1.6486 ± 0.8895 (R²) | — |
| shift_risk | continuous | random_init_lp | 2.84 ± 2.67 | -0.2036 ± 0.1471 (R²) | — |

### Table 2: FNO-Only with CE-Adjusted Losses (Epoch 16)

*Definitions: `final_*_loss` = epoch 16 from curves; `train_total_time_s` = GPU wall time per probe (mean ± CI).*

| label | task_type | train_time_s (mean ± CI) | approx_flops | final_train_loss (mean ± CI) | final_val_loss (mean ± CI) |
|-------|-----------|------------------------|--------------|----------------------------|---------------------------|
| cardinality_bucket | categorical | 12.31 ± 3.69 | 1.84e+12 | 0.0585 ± 0.0367 | 0.1818 ± 0.0480 |
| coordination | continuous | 11.32 ± 2.33 | 1.84e+12 | 0.1727 ± 0.0033 | 0.2049 ± 0.0075 |
| current_burstiness | continuous | 11.21 ± 2.26 | 1.84e+12 | 1.02e+13 ± 2.66e+12 | 1.17e+13 ± 3.71e+12 |
| current_sparsity | continuous | 13.68 ± 5.09 | 1.84e+12 | 0.0057 ± 0.0021 | 0.0057 ± 0.0013 |
| domain | categorical | 10.26 ± 3.55 | 1.47e+12 | 0.8872 ± 0.0423 | 1.4293 ± 0.0979 |
| frequency_bucket | categorical | 18.17 ± 13.39 | 1.84e+12 | 0.5288 ± 0.0148 | 0.9125 ± 0.2310 |
| future_burstiness | continuous | 10.99 ± 1.91 | 1.84e+12 | 1.38e+13 ± 3.90e+12 | 1.68e+13 ± 9.00e+12 |
| future_sparsity | continuous | 11.07 ± 1.98 | 1.84e+12 | 0.0057 ± 0.0021 | 0.0055 ± 0.0013 |
| metric_type | categorical | 11.73 ± 2.35 | 1.77e+12 | 0.9299 ± 0.0220 | 1.4927 ± 0.5358 |
| shift_risk | continuous | 10.82 ± 1.83 | 1.84e+12 | 9.25e+09 ± 1.88e+09 | 9.15e+09 ± 4.47e+09 |

> **Note on loss scales**: Continuous targets on large scales (burstiness, shift risk) have artificially high MSE losses by construction; categorical cross-entropy losses remain in the ~0–2 range. CI = 95% confidence interval from 5 seeds. FNO test metrics for continuous labels aren't in `pr2_issue3_matched_mean_ci.csv` (matching only captured categorical FNO metrics).

## Repo Updates
- `slurm/refit_fno_issue3_curves.sh`: Refit FNO from activations + aggregate (backs up old probes first). Updated to run all seeds 42–46.
- `scripts/aggregate_pr2_operational.py`: Added per-`task_type` metric selection and path fallback for artifacts.
- `runs/pr2_operational_issue3/operational_seed{42-46}_report.{md,json}`: Individual seed reports on cluster (local has all 5).

## Cluster Paths (All Seeds Complete)
- Reports: `/scratch/rianatri/toto-interp/runs/pr2_operational_issue3/operational_seed{42-46}_report.{md,json}`
- FNO artifacts: `/scratch/rianatri/toto-interp/runs/pr2_operational_issue3/operational_fno_seed{42-46}/probes/artifacts/*.pt`
- Aggregates: `/scratch/rianatri/toto-interp/runs/pr2_operational_issue3/issue3_aggregate/`
