[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![Research Status](https://img.shields.io/badge/status-active%20research-green)](https://github.com/SteviLen420/TQE_simulation)

# TQE Energy Fluctuations Heisenberg v4.2.0 PRO

**Title:** Heisenberg-Limited Vacuum Fluctuation Suppression Under TQE Law Lock-In  
**Author:** Stefan Len  
**Version:** v4.2.0 PRO

---

## Abstract

This module implements a dedicated Monte Carlo + open-quantum-system simulation that probes the TQE hypothesis in the context of the Heisenberg Uncertainty Principle. Two scenarios are evolved in parallel:

1. **Pre-Law / No Lock-In:** Pure vacuum fluctuations without the stabilising f(E,I) coupling.  
2. **With-Law / Lock-In:** Identical quantum system but with active TQE coupling that dynamically modulates dissipation, anharmonicity and bath coupling.

By comparing the energy variance, entropy growth, coherence decay and Δx·Δp evolution between the two scenarios, we quantify how effective “physical law lock-in” is at suppressing universe-spawning fluctuations. The run also tests three hypotheses about the origin of the information parameter I (emergent, inherent, threshold).

---

## Key Capabilities

- **Scenario pairing, shared dynamics:** NO-LAW and WITH-LAW trajectories share the exact Hamiltonian, drives, bath, and seeding so suppression ratios are meaningful.
- **Open-quantum evolution:** QuTiP Lindblad solvers with amplitude damping, dephasing, thermal baths, optional two-mode coupling, anharmonic x⁴/double-well potentials, and driven oscillators.
- **Dynamic TQE lock-in:** The f(E,I)=exp(-(E-Ec)²/(2σ²))(1+αI) coupling rescale dissipation and potentials segment-by-segment so suppression is enacted online.
- **Automatic I-origin sweep:** Emergent, inherent, and threshold information models are all simulated every run; dedicated plots show their behaviour plus a comparison overlay.
- **Heisenberg compliance checks:** Δx, Δp, Δx·Δp are tracked for every trajectory and benchmarked against ℏ/2, with compliance stored in the JSON outputs.
- **Multi-metric statistics:** Energy variance/entropy/coherence/uncertainty ratios, phase-space trajectories, Jensen–Shannon drift, and suppression dashboards are produced out of the box.
- **Optional parameter sweep engine:** Toggle `ENABLE_PARAMETER_SWEEP` to scan EC/SIGMA/ALPHA (or any MASTER_CTRL scalar) with CSV + dual-panel PNG summaries.
- **Reproducible by design:** A single `MASTER_CTRL['SEED']` drives master RNG state, per-trajectory seeds, and metadata recorded in `summary.json`.
- **Analysis-ready outputs:** Timestamped run folders include `summary.json`, `comparative_analysis.json`, three CSVs, and ≥13 PNGs accepted directly by the TQE analysis pipeline.

---

## Requirements & Environment

- Python **3.9+** (tested on CPython 3.9–3.11).  
- `numpy`, `scipy`, `matplotlib`, `tqdm`, `qutip`. The script self-checks these modules at startup and will `pip install` them if missing.  
- Optional (`ENABLE_PARAMETER_SWEEP=True`): `pandas` for CSV aggregation.  
- Local or Google Colab execution—Colab runs auto-mount Drive and write under `/content/drive/MyDrive/TQE_Heisenberg_Fluctuation/`.

> QuTiP compilation needs standard build tools (Xcode CLI tools on macOS, build-essential on Linux, MSVC Build Tools on Windows).

---

## Configuration Overview

All controls live in the `MASTER_CTRL` dictionary near the top of `TQE_Energy_Fluctuations_Heisenberg_v4.2.0_Pro.py`. The most used blocks are:

| Block | Representative keys | Purpose |
| --- | --- | --- |
| Reproducibility & Ensemble | `SEED`, `N_ENSEMBLE`, `N_HILB`, `T_FINAL`, `N_T` | Master RNG seed, ensemble size, Hilbert truncation, runtime grid |
| Quantum features | `ANHARMONIC_X4`, `DOUBLE_WELL`, `TWO_MODE_COUPLING`, `TIME_DEP_DRIVE`, `THERMAL_BATH`, `TRAJECTORIES` | Toggle anharmonic/double-well potentials, two-mode coupling, time-dependent drives, MC trajectories |
| Open-system rates | `GAMMA_PHI_1/2`, `KAPPA_1/2`, `NTH_1/2` | Baseline Lindblad rates for the pre-law scenario |
| Lock-in parameters | `BETA_A/B`, `EC`, `SIGMA`, `ALPHA`, `N_SEGMENTS` | Information prior and Goldilocks window used when TQE lock-in is active |
| I-origin tuning | `I_EMERGENT_*`, `I_INHERENT_*`, `I_THRESHOLD_*` | Shape each information model (all three run every execution) |
| Heisenberg guard | `HEISENBERG_LIMIT_ACTIVE`, `DELTA_X_MIN`, `DELTA_P_MIN`, `UNCERTAINTY_PRODUCT_MIN` | Explicit uncertainty cutoffs for compliance logging |
| Parameter sweep | `ENABLE_PARAMETER_SWEEP`, `SWEEP_VARIABLE`, `SWEEP_VALUES`, `SWEEP_N_ENSEMBLE` | Optional scan over EC / SIGMA / ALPHA (or any scalar key) |
| Output & plotting | `BASE_FOLDER_NAME`, `PLOT_DPI`, `PLOT_FONTSIZE_*` | Directory naming and publication-grade figure styling |

No CLI flags are parsed—edit `MASTER_CTRL` (or import the module and mutate the dict) to change settings.

---

## Running the Simulation

```bash
cd TQE_Universe_Simulation_v4.2.0_Pro/TQE_Energy_Fluctuations_Heisenberg
python TQE_Energy_Fluctuations_Heisenberg_v4.2.0_Pro.py
```

Typical workflow:
1. Edit `MASTER_CTRL` to set the ensemble size, lock-in window, plotting DPI, etc.
2. (Optional) Enable the sweep block to scan `EC`, `SIGMA`, or `ALPHA`.
3. Run the script; a 7-phase progress bar will appear (NO-LAW → WITH-LAW → aggregation → stats → data export → visualisations → optional sweep).
4. Inspect the timestamped folder under `./TQE_Heisenberg_Fluctuation/`.

Runtime on a modern laptop is ~3–6 minutes for `N_ENSEMBLE=100`, scaling roughly linearly with ensemble size and sweep points.

---

## Pipeline Phases

1. **Phase 1 – NO-LAW ensemble:** Monte Carlo Lindblad evolution without f(E,I).  
2. **Phase 2 – WITH-LAW (3× I-origin):** Repeats the same ensemble with emergent, inherent, and threshold information models.  
3. **Phase 3 – Data aggregation:** Aligns time-series arrays, builds mean/variance tensors, and extracts final energy sets.  
4. **Phase 4 – Statistical comparison:** Computes suppression ratios, Heisenberg compliance, and summary tables saved to JSON.  
5. **Phase 5 – Data export & visualization:** Writes CSVs plus 13 PNGs (energy/variance/entropy/coherence, phase space, Δx·Δp, multi-panel, I-mode plots).  
6. **Phase 6 – Reporting:** Console summary plus `summary.json` that includes the full `MASTER_CTRL`.  
7. **Phase 7 – Parameter sweep (optional):** If `ENABLE_PARAMETER_SWEEP=True`, an extra CSV + dual-panel PNG are produced.

## Output Structure

```
TQE_Heisenberg_Fluctuation_<timestamp>/
├── summary.json                 # run info + MASTER_CTRL + stats (ingested downstream)
├── comparative_analysis.json    # NO-LAW vs WITH-LAW tables + suppression ratios
├── data/
│   ├── no_law_timeseries.csv
│   ├── with_law_timeseries.csv
│   ├── ensemble_final_energies.csv
│   └── parameter_sweep_<var>.csv        # only when sweeps are enabled
└── figs/
    ├── 01_energy_comparison.png
    ├── 02_variance_comparison.png
    ├── 03_entropy_comparison.png
    ├── 04_coherence_comparison.png
    ├── 05_final_energy_dist.png
    ├── 06_suppression_summary.png
    ├── 07_heisenberg_uncertainty.png
    ├── 08_phase_space_E_vs_S.png
    ├── 09_multidimensional_tracking.png
    ├── 10_parameter_sweep_<var>.png     # only if sweep enabled
    ├── 11_I_evolution_emergent.png
    ├── 12_I_evolution_inherent.png
    ├── 13_I_evolution_threshold.png
    └── 14_I_mode_comparison.png
```

Every CSV contains a metadata header (run name, timestamp, seed) so the analysis pipeline can link these runs to universe-scale experiments.

---

## Interpretation Checklist

1. **Suppression ratios:** In `comparative_analysis.json`, ratios < 1 for variance / uncertainty / coherence indicate successful lock-in.  
2. **Entropy & coherence plots (Fig. 3–4):** WITH-LAW traces should flatten earlier than NO-LAW.  
3. **Heisenberg compliance (Fig. 7 + JSON block):** Verify both scenarios respect ℏ/2; WITH-LAW typically approaches but does not cross the bound.  
4. **Phase space (Fig. 8):** Energy–entropy trajectories reveal whether laws corral the system into a bounded attractor.  
5. **I-origin panels (Fig. 11–14):** Inspect whether emergent/inherent/threshold models converge or diverge; this feeds into later Goldilocks diagnostics.  
6. **Parameter sweep outputs:** Identify the EC/SIGMA/ALPHA ranges where suppression ratios undergo sharp transitions.

---

## Relationship to Main TQE Pipeline

- `summary.json` includes the exact `MASTER_CTRL` dict, timestamp, and seed so the Monte Carlo universe pipeline can ingest these settings.
- `comparative_analysis.json` matches the schema expected by the v4.2.0 analysis pipeline (Phases 11–20) for downstream visualisation.
- Parameter sweep CSVs can be cross-referenced when calibrating the Bayesian Goldilocks search inside `TQE_Pipeline_Modular`.

---

## How to Cite

> Stefan Len. (2025). *TQE Energy Fluctuations Heisenberg v4.2.0 PRO* [Software]. GitHub. https://github.com/SteviLen420/TQE_simulation

BibTeX:

```bibtex
@software{Len_2025_TQE_Heisenberg_v4_2,
  author    = {Len, Stefan},
  title     = {{TQE Energy Fluctuations Heisenberg v4.2.0 PRO}},
  year      = {2025},
  publisher = {GitHub},
  url       = {https://github.com/SteviLen420/TQE_simulation},
  version   = {4.2.0}
}
```

---

## Support / Issues

- File a GitHub issue in the main repository or contact the author for collaborative research discussions.  
- When reporting bugs, attach the `summary.json` and relevant CSVs for faster diagnosis.