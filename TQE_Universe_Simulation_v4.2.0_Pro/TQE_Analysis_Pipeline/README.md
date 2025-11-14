[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX) <!-- TODO: replace with real DOI when minted -->
[![arXiv](https://img.shields.io/badge/arXiv-coming%20soon-b31b1b.svg)](https://arxiv.org/) <!-- TODO: update with actual arXiv entry when available -->
[![GitHub stars](https://img.shields.io/github/stars/SteviLen420/TQE_simulation?style=social)](https://github.com/SteviLen420/TQE_simulation)
[![GitHub forks](https://img.shields.io/github/forks/SteviLen420/TQE_simulation?style=social)](https://github.com/SteviLen420/TQE_simulation)
[![Research Status](https://img.shields.io/badge/status-active%20research-green)](https://github.com/SteviLen420/TQE_simulation)

# TQE ANALYSIS PIPELINE (E,I) — POST-SIMULATION COMPARATIVE SUITE

Author: **Stefan Len**  
Version: **v4.2.0 PRO**

---

## 0. Executive Summary

The analysis pipeline ingests the full output of a `batch_all` or `batch_ei` universe simulation run, harvests every JSON/CSV artifact under each timestamped `Aggregate/` folder, and produces:

- An extended metrics table (≈80 columns) covering stability, Friedmann cosmology, Planck proximity, life compatibility, entropy volatility, sweeps, anomalies, nested sampling, and top-universe metadata.
- 12+ comparative analysis categories with PNG visualizations + CSV exports.
- Triple model rankings (stability, complexity, physical-laws realism) with a recommendation report.
- A clean summary + extended/validation bundle ready for publication or further machine learning exploration.

---

## 1. Requirements & Environment

- Python **3.9+** (same interpreter as the main simulation)
- Core libraries already required by the simulation: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy`, `tqdm`
- Optional (enables extra diagnostics): `healpy`, `shap`, `lime`
- Access to the simulation output root (`TQE_Universe_Simulation_Full_Pipeline_v4.2.0_PRO`)
- Works locally and on Google Colab (script auto-detects Colab, mounts Drive, and rewrites paths)

---

## 2. Configuration Knobs (most common)

| Setting / Env Var | Default | Purpose |
| --- | --- | --- |
| `SIMULATION_ROOT` | `../TQE_Universe_Simulation_Full_Pipeline_v4.2.0_PRO` | Where timestamped `TQE_Universe_Simulation_*` folders live |
| `ANALYSIS_OUTPUT_ROOT` | `analysis_results` | Parent directory for all generated reports |
| `TARGET_MODE` | `"batch_all"` | `batch_all` (E-only + E+I mix) or `batch_ei` (E+I only) |
| `TARGET_TIMESTAMP` | `None` | If set (`YYYYMMDD_HHMMSS`), force that batch, otherwise latest |
| `PLANCK_TARGET_E` / `PLANCK_TARGET_I` | 0.7619 / 0.1309 | Reference marker in Planck-fit plots |
| `FIGURE_DPI` | 200 | Global resolution for generated figures |

Override via environment:
```bash
export TARGET_MODE=batch_ei
export TARGET_TIMESTAMP=20251113_193539
python3 TQE_Analysis_Pipeline_v4.2.0_PRO.py
```

---

## 3. Directory Layout

```
TQE_Analysis_Pipeline/
├── README.md                     # This document
├── TQE_Analysis_Pipeline_v4.2.0_PRO.py
└── analysis_results/             # Created when the script runs
```

Simulation outputs are expected under `../TQE_Universe_Simulation_Full_Pipeline_v4.2.0_PRO/`.

---

## 4. Running the Pipeline

```bash
cd TQE_Analysis_Pipeline
python3 TQE_Analysis_Pipeline_v4.2.0_PRO.py
```

Execution steps:
1. Detect local vs Colab environment.
2. Discover the target batch (latest timestamp unless overridden).
3. Load all artifacts (summary, tqe_runs, Planck, life, entropy, anomalies, etc.).
4. Build the extended metrics DataFrame; cache raw data as pickles.
5. Generate comparative modules, visualizations, rankings, and final reports.
6. Produce extended markdown + metadata exports.
7. Validate that critical artifacts exist and log the status.

Runtime: ~1–3 minutes per batch on a modern laptop, dominated by plotting.

---

## 5. Data Inputs Harvested per Run

| Artifact | Description / Use |
| --- | --- |
| `summary_full.json` | Stability breakdown, Goldilocks window, Bayesian fit, top universes |
| `tqe_runs_EI_*.csv` | Friedmann metrics, lock-in epochs, entropy volatility, quantum fields, entanglement |
| `life_compatibility_summary.json` | Life-score components (Planck/stability/goldilocks sub-scores) |
| `planck_best_fit_summary.json`, `planck_validation*.csv` | Planck best-fit E/I/α, χ², score, ℓ-span |
| `entropy_volatility_summary*.csv` | Global entropy volatility statistics |
| `stability_by_I_eps_sweep*.csv`, `stability_by_I_zero*.csv` | Sensitivity sweeps (log ε slope, zero-baseline) |
| `advanced_anomaly_detection_results*.csv`, `physical_anomalies*.csv`, `cmb_gaussianity_check*.csv`, `cmb_isotropy_check*.csv` | Advanced anomaly diagnostics, Gaussianity/isotropy tests |
| `I_Definitions_Comparison.csv` | I(E) curves for all definitions (mean/std/range) |
| `nested_sampling_samples*.csv` | Nested sampling traces (logZ iterations) |
| `pre_fluctuation_pairs*.csv`, `universe_seeds*.csv` | Seed QA / reproducibility checks |

> If a file is missing, the loader emits a warning and fills metrics with `NaN` so the pipeline can proceed.

---

## 6. Analysis Phases (mirrors the main README structure)

1. **Comprehensive Data Collection** – Intelligent globbing, metadata counters for each artifact type.
2. **Extended Metric Extraction** – Builds a master DataFrame with 50–80 fields per run.
3. **Comparative Categories**  
   - Basic metrics (stability & Goldilocks)  
   - Emergent laws  
   - Friedmann cosmology  
   - Planck fit proximity (scatter + χ²)  
   - CMB anomalies  
   - Lock-in dynamics  
   - Quantum fields  
   - Entanglement  
   - Parameter sensitivity  
   - Statistical fine-tuning  
   - Topology  
   - I-definition curves  
   - E-only vs E+I baseline  
   - Life compatibility & top universes  
   - Entropy volatility & stability sweeps  
   - Advanced physical anomalies
4. **Advanced Visualizations** – Radar charts, complexity panels, heatmaps.
5. **Triple Ranking System** – Stability-, complexity-, and physical-laws-focused CSVs + Markdown recommendation.
6. **Extended Reports** – `extended_report.md` with highlights + follow-up checklist.
7. **Summary Export** – `analysis_summary.txt` + `run_info.json` metadata bundle.
8. **Validation & QC** – `validation_report.txt` enumerates required artifacts.

---

## 7. Output Directory Map

```
analysis_results/<mode>_<timestamp>_analysis/
├── 00_summary/
│   ├── analysis_summary.txt
│   ├── extended_report.md
│   ├── validation_report.txt
│   └── run_info.json
├── 01_comparative_analysis/
│   ├── basic_metrics/
│   ├── emergent_laws/
│   ├── friedmann_cosmology/
│   ├── planck_fit/
│   ├── life_top_universes/
│   ├── entropy_volatility/
│   ├── cmb_anomalies/
│   ├── lockin_dynamics/
│   ├── quantum_fields/
│   ├── entanglement/
│   ├── parameter_sensitivity/
│   ├── finetuning/
│   ├── topology/
│   ├── physical_anomalies/
│   ├── i_definitions_direct/
│   └── eonly_vs_ei/
├── 02_detailed_metrics/
│   ├── all_runs_metrics.csv
│   ├── correlation_matrix.png
│   └── distributions_boxplot.png
├── 03_visualizations/
│   ├── radar_chart_top5.png
│   └── complexity/*.png
├── 04_best_model_selection/
│   ├── ranking_stability_focused.csv
│   ├── ranking_complexity_focused.csv
│   ├── ranking_physical_laws_focused.csv
│   ├── top_3_models_triple.json
│   └── recommendation_report_triple.md
└── 05_raw_data/
    ├── collected_data.pkl
    └── extended_metrics.pkl
```

Every subfolder contains both PNG plots and CSV tables so the results can be re-used elsewhere (e.g., notebooks, dashboards, ML pipelines).

---

## 8. Metrics Cheat Sheet (selected fields)

| Field | Description |
| --- | --- |
| `i_definition`, `run_type` | E-only vs E+I identity |
| `stable_percent`, `lockin_percent`, `goldilocks_width` | Core stability/Goldilocks stats |
| `complexity_score`, `life_compatibility_score`, `information_richness` | Derived scores from summary |
| `life_score_json`, `life_planck_component`, `life_stability_component` | Raw life compatibility components |
| `planck_E`, `planck_I`, `planck_chi2_reduced`, `planck_score` | Planck fit proximity |
| `planck_validation_chi2_mean`, `planck_validation_ell_span` | Per-ℓ validation statistics |
| `entropy_volatility_global_mean`, `stability_eps_slope`, `stability_zero_baseline` | Information volatility + sweep sensitivity |
| `advanced_anomaly_sigma_mean`, `physical_anomaly_count`, `cmb_gaussianity_p_mean` | Advanced anomaly diagnostics |
| `nested_sampling_iterations`, `nested_logZ_final` | Evidence convergence |
| `top_universe_seed`, `top_universe_lock_epoch`, `top_universe_I` | Metadata for the highest-ranked universe |

Full list → `02_detailed_metrics/all_runs_metrics.csv`.

---

## 9. Troubleshooting & Tips

- **“summary_full.json not found”** → ensure the simulation finished; partial runs don’t emit this file.
- **Colab drive issues** → run `from google.colab import drive; drive.mount('/content/drive')` before launching the script.
- **Empty plot or NaN column** → indicates the artifact for that module was missing; check loader warnings in the console.
- **Custom timestamp analysis** → set `TARGET_TIMESTAMP` to re-run an older batch without touching others.
- **Speed tweak** → lower `FIGURE_DPI` or temporarily disable heavy modules if you only need CSVs.
- **Planck marker tweak** → export different fiducials by overriding `PLANCK_TARGET_E` / `PLANCK_TARGET_I`.

---

## 10. Extending the Pipeline

1. Ensure the simulation writes the new artifact under every `Aggregate/` folder.
2. Add a `load_*` helper in the script (pattern-matching + pandas/json load).
3. Store the artifact inside each run’s dictionary (`collected_data`).
4. Map the artifact to new columns in `extract_extended_metrics`.
5. Create an `analyze_*` module for plotting/export and register it in Phase 3, plus update `dirs`.

Use the existing Planck/life/entropy/anomaly modules as templates.

---

## 11. Automation Notes

- Run the analysis pipeline after *every* `batch_all` simulation to keep an archival history (each timestamped output lives in its own folder).
- Commit only the code and README; treat `analysis_results/` as generated artifacts.
- CI / cron idea: monitor the simulation root for new `TQE_Universe_Simulation_single_ei_*` folders → trigger the analysis script automatically → push the resulting summary to a dashboard or Slack.

---

The analysis pipeline mirrors the structure and rigor of the main TQE README, giving you a reproducible, well-documented post-processing and validation workflow. With the full documentation centralized here, you can keep the Python script lean while still having every detail at your fingertips. Happy analyzing! 👩‍🚀🌀