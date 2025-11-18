# TQE Analysis Pipeline v4.2.0 PRO — Extended Report

- Generated: 2025-11-18 20:07:16
- Target mode: batch_all
- Source directory: `/Users/stevilen/Documents/GitHub/TQE_simulation/TQE_Universe_Simulation_Full_Pipeline_v4.2.0_PRO/TQE_Universe_Simulation_batch_all_20251118_101437`

## Run Inventory
- Total runs analyzed: **11**
- E-only runs: **1**
- E+I runs: **10**

## Highlighted I-Definitions
- **Stability leader**: `entanglement` (100.00% stable)
- **Complexity leader**: `shannon` (score 66.62)
- **Planck proximity leader**: `kl_shannon` (χ²=1073.91)

## Artifact Coverage
- summary_full.json, tqe_runs.csv, Bayesian calibration
- Planck validation (scatter + χ² bar + CSV export)
- Life compatibility, entropy volatility, anomaly diagnostics
- Nested sampling traces, stability sweeps, seed registries

## Suggested Follow-ups
1. Inspect `04_best_model_selection/recommendation_report.md` for model choices.
2. Review `02_detailed_metrics/all_runs_metrics.csv` for downstream ML.
3. Compare E-only vs E+I in `01_comparative_analysis/eonly_vs_ei/` if available.
