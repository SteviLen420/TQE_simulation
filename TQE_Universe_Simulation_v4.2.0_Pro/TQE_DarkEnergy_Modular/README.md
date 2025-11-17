[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17614948.svg)](https://doi.org/10.5281/zenodo.17614948)
[![Research Status](https://img.shields.io/badge/status-active%20research-green)](https://github.com/SteviLen420/TQE_simulation)

# TQE Dark Energy Modular Pipeline

**Title:** TQE–ΛSim (Modular): Energy–Information Coupling for Dark Energy Evolution  
**Author:** Stefan Len  
**Version:** v4.2.0 PRO (Modular build)

---

## 0. Overview

This directory contains the **modularized** version of the TQE Dark Energy Coupling Simulation.  
`TQE_DarkEnergy_Coupling_Simulation.py` (monolithic script) has been refactored into reusable modules so the pipeline can be imported, tested, or partially executed without modifying a 10k+ line file. The modular build keeps the scientific content identical while exposing clear APIs for:

- Running the full E-only vs E+I dual-mode campaign with Bayesian post-processing
- Swapping/adding coupling models, I-parameter definitions, or observational data loaders
- Calling individual components (cosmology, observables, inference, structure analysis) in notebooks
- Maintaining Colab-first workflows with deterministic seeding and Google Drive output sinks

---

## 1. Directory Layout

```
TQE_DarkEnergy_Modular/
├── __init__.py
├── config.py                 # MASTER_CTRL, COSMO_PARAMS, FIDUCIAL_PARAMS, constants
├── cosmology.py              # FriedmannEvolution + distance/growth utilities
├── data_loader.py            # Pantheon+, BOSS/eBOSS/DESI, Planck loaders (real + mock)
├── inference.py              # BayesianInferenceEngine (MCMC + nested sampling)
├── main.py                   # CLI entry point (mirrors monolithic banner)
├── observables.py            # ObservablePredictions (SNe, BAO, CMB, LSS, likelihood)
├── pipeline.py               # run_automatic_tqe_darkenergy_pipeline orchestrator
├── simulation.py             # TQEDarkEnergyCouplingSimulation class (per-model workflow)
├── structure.py              # GalaxyStructureAnalyzer (cosmic web, catalogues)
├── tqe_core.py               # EnergyInformationContent + CouplingModel definitions
├── utils.py                  # Package installs, Colab/Drive setup, seeding, perf helpers
├── visualization.py          # Cross-mode dashboards, comparison plots
└── (support modules)         # observables, visualization, etc. imported throughout
```

Every module can be imported independently (e.g., `from TQE_DarkEnergy_Modular.cosmology import FriedmannEvolution`) enabling targeted notebooks or tests without executing the entire pipeline.

---

## 2. Module Guide

| Module | Purpose / Key Classes |
| --- | --- |
| `config.py` | Central `MASTER_CTRL`, Planck 2018 cosmological constants, performance knobs, data paths, visualization palette, Google Drive settings. |
| `tqe_core.py` | Implements **EnergyInformationContent** (phenomenological, EFT, and TQE-compliant energy-based I definitions) and **CouplingModel** (covariant pressure, uniform w, geometric, null ΛCDM). |
| `cosmology.py` | `FriedmannEvolution` with TQE-aware Ω_DE(a), distance ladder, log-spaced integrators, growth-factor ODE, and flatness guards. |
| `data_loader.py` | Reusable loaders for Pantheon+, BOSS/eBOSS/DESI BAO, Planck spectra/maps with fallbacks to “enhanced” mock data when `ALLOW_MOCK_DATA=True`. |
| `observables.py` | `ObservablePredictions`: μ(z), BAO D_M/H, CMB C_ℓ (CAMB-backed when available), matter P(k), σ₈ integration, S₈, composite likelihood (χ² + AIC/BIC/DIC). |
| `inference.py` | `BayesianInferenceEngine` with emcee-based MCMC, dynesty nested sampling, corner plots, evidence/Bayes factors, information criteria. |
| `structure.py` | `GalaxyStructureAnalyzer` for synthetic density fields, cosmic-web classification, void/filament/sheet/cluster catalogues, Minkowski metrics. |
| `simulation.py` | `TQEDarkEnergyCouplingSimulation`: per-model execution (H(a), I(a), ρ_DE(a), S₈(z), I–E correlations, observables, galaxy structure, sanity/sensitivity checks, visualization + file export). |
| `pipeline.py` | End-to-end pipeline orchestrator: Goldilocks search, model loops, dual coupling modes, β₀ sweeps, result aggregation, aggregator dashboards, reproducibility snapshots. |
| `visualization.py` | Comparison dashboards (E-only vs E+I), Bayes-factor plots, β₀ sweep figures. |
| `utils.py` | Dependency installation, Colab/Drive detection, deterministic seeding, performance modes, memory cleanup, ZIP archiving. |

---

## 3. Requirements & Environment

- **Python 3.9+** (tested on CPython 3.9–3.11)
- **Google Colab + Google Drive** are the intended runtime (pipeline aborts on purely local execution for reproducibility reasons).
- Core dependencies: `numpy`, `scipy`, `pandas`, `matplotlib`, `tqdm`, `scikit-learn`, `camb`, `emcee`, `dynesty`, `corner`, `healpy` (optional), `astropy`, `h5py`.
- `utils.check_and_install_all_packages()` auto-installs missing packages (quiet pip installs, Colab-safe).
- CAMB, emcee, dynesty, and healpy are loaded lazily; the pipeline attempts installation if imports fail.

> **Tip:** When running in Colab, upload the `TQE_DarkEnergy_Modular` folder, mount Drive, and execute `python -m TQE_DarkEnergy_Modular.main`. The script mounts Drive (if needed) and writes all artifacts under `/content/drive/MyDrive/TQE_DarkEnergy_Coupling_Simulation/`.

---

## 4. Quickstart

### Colab (recommended)

```python
from google.colab import drive
drive.mount('/content/drive')

%cd /content/TQE_DarkEnergy_Modular
!python -m TQE_DarkEnergy_Modular.main
```

### As a module

```python
from TQE_DarkEnergy_Modular.pipeline import run_automatic_tqe_darkenergy_pipeline
from TQE_DarkEnergy_Modular.config import MASTER_CTRL

# Optional overrides
MASTER_CTRL["COUPLING_MODE"] = "dual"
MASTER_CTRL["RUN_BETA0_SWEEP"] = True

results = run_automatic_tqe_darkenergy_pipeline()
```

### Custom experimentation

```python
from TQE_DarkEnergy_Modular.config import MASTER_CTRL, FIDUCIAL_PARAMS
from TQE_DarkEnergy_Modular.tqe_core import EnergyInformationContent, CouplingModel
from TQE_DarkEnergy_Modular.simulation import TQEDarkEnergyCouplingSimulation

i_field = EnergyInformationContent("energy_based", {"epsilon": 1e-6, "normalization": "tanh"})
coupling = CouplingModel("covariant_pressure", i_field, {"alpha": 0.02}, coupling_mode="EplusI")
sim = TQEDarkEnergyCouplingSimulation(coupling, i_field, fiducial_params=FIDUCIAL_PARAMS.copy())
sim.run_cosmological_evolution()
sim.compute_observables()
```

---

## 5. Configuration & Execution Modes

All knobs live in `MASTER_CTRL` (`config.py`). The most-used blocks:

| Block | Representative Keys | Notes |
| --- | --- | --- |
| Cosmology | `H0`, `OMEGA_M`, `OMEGA_LAMBDA`, `SIGMA_8`, `USE_ODE_GROWTH` | Planck 2018 fiducials, high-precision growth solver. |
| Coupling/I-field | `COUPLING_MODE`, `RUN_DUAL_COMPARISON`, `I_FIELD_AMPLITUDE`, `I_FIELD_NORMALIZATION`, `ALPHA_COUPLING`, `BETA0_COUPLING`, `RUN_BETA0_SWEEP` | Switch between E-only / E+I / dual, enable β₀ sweeps, tune I definitions. |
| Bayesian | `RUN_MCMC`, `RUN_NESTED_SAMPLING`, `MCMC_NSTEPS`, `NESTED_NLIVE`, `COMPUTE_EVIDENCE`, `COMPUTE_AIC/BIC/DIC` | Full posterior + evidence workflow. |
| Data paths | `PANTHEON_PLUS_DATA_PATH`, `BOSS_BAO_DATA_PATH`, `PLANCK_CMB_DATA_PATH`, `ALLOW_MOCK_DATA`, `USE_REAL_*` | Toggle between real and enhanced mock datasets. |
| Performance | `PERFORMANCE_MODE`, `ENABLE_VECTORIZATION`, `AUTO_FIND_GOLDILOCKS`, `RUN_GALAXY_STRUCTURE_ANALYSIS` | Reduce grid sizes for quick iterations or ramp up for publication quality. |

**Execution Modes**
- `COUPLING_MODE="Eonly" | "EplusI" | "dual"` — run baseline, coupled, or both.
- `RUN_BETA0_SWEEP=True` — expand Model 3 (geometric) into finely sampled β₀ grid (21 values by default).
- `AUTO_FIND_GOLDILOCKS=True` — Differential-Evolution search for optimal `(E_c, σ, α, β₀)` before model runs.

---

## 6. Pipeline Flow (Modular Build)

1. **Environment setup** – package checks, Drive mount, deterministic seeding (`utils.py`).
2. **Goldilocks optimization (optional)** – `find_goldilocks_zone_bayesian` (from monolithic helper) stores optimal parameters and patches `MASTER_CTRL`.
3. **Model expansion** – `pipeline.py` builds per-model configs (covariant pressure, uniform w(I), geometric, null) + optional β₀ sweep clones.
4. **Dual-mode execution** – For each coupling mode (E-only / E+I) and model:
   - `TQEDarkEnergyCouplingSimulation` runs cosmological evolution, I-field evaluation, ρ_DE modulation, S₈/ρ_DE/D(z) series, I–E correlation, observables, galaxy structure, sanity + sensitivity sweeps, visualization.
   - Optional CMB Planck validation (requires `healpy` + real maps).
   - Bayesian inference: emcee MCMC + dynesty nested sampling per model.
   - Saving: CSV/JSON/PNG/ZIP bundles under model-specific directories.
5. **Cross-model aggregation** – `visualization.compare_eonly_vs_eplusi`, Bayes-factor grids, β₀ dashboards, summary CSVs.
6. **Final summary** – pipeline metadata, reproducibility snapshot (`MASTER_CTRL` + environment info), aggregated rankings.

---

## 7. Outputs

Each run creates a Drive folder:  
`/content/drive/MyDrive/TQE_DarkEnergy_Coupling_Simulation/TQE_DarkEnergy_Coupling_Simulation_v4.2.0PRO_<timestamp>/`

```
├── 00_Pipeline_Summary/
│   ├── Pipeline_Summary.txt
│   ├── Pipeline_Metadata.json
│   └── MASTER_CTRL_Snapshot.json
├── 01_Goldilocks_Optimization/           (if enabled)
├── 02_Model_Runs/
│   ├── Model_<name>_<mode>_<timestamp>/
│   │   ├── evolution/*.csv              (H(a), I(a), ρ_DE(a))
│   │   ├── observables/*.csv            (SNe, BAO, CMB, LSS, likelihood)
│   │   ├── bayesian/*.csv|.h5           (MCMC chains, nested samples)
│   │   ├── galaxy_structure/*.csv       (void/filament/cluster catalogues)
│   │   ├── sanity_sensitivity/*.json
│   │   ├── PNG_Visualizations/*.png
│   │   └── TQE_DarkEnergy_Complete_Results.zip
├── 03_Cross_Model_Analysis/
│   ├── Eonly_Aggregator/*.csv|.png
│   ├── EplusI_Aggregator/*.csv|.png
│   └── Eonly_vs_EplusI_Comparison/*.csv|.png
└── 04_Reproducibility/
    ├── MASTER_CTRL_Final.json
    ├── Environment_Info.txt
    └── Package_Versions.json
```

All CSV/JSON files include metadata (timestamp, seed hash, coupling mode) so downstream notebooks or dashboards can trace provenance.

---

## 8. Working With Individual Modules

- **Swap coupling physics**: create a new class in `tqe_core.CouplingModel` (e.g., add modified gravity term) and reference it in `pipeline.py` when building `models_config`.
- **Custom data ingestion**: extend `data_loader.py` with new survey readers, then toggle paths in `MASTER_CTRL`.
- **Notebook studies**: import `FriedmannEvolution` or `ObservablePredictions` to compute H(z), μ(z), or S₈ figures without running the entire pipeline.
- **Testing**: Each module is small enough (≈200–1500 lines) to be unit-tested independently; use deterministic seeds for reproducible outputs.

---

## 9. Troubleshooting & Tips

- **Pipeline exits immediately** → Running locally without Drive/Colab. Either move to Colab or adapt `pipeline.py` (lines enforcing COLAB) if you accept manual storage.
- **healpy missing** → Install via `pip install healpy` or set `USE_REAL_CMB_PLANCK_MAPS=False`.
- **Memory pressure** → Enable `MEMORY_EFFICIENT_MODE=True`, reduce `A_GRID_N_LOG`, set `PERFORMANCE_MODE="fast"`, or disable galaxy structure analysis.
- **Long Bayesian runs** → Lower `MCMC_NSTEPS`, `NESTED_NLIVE`, or switch off Bayesian blocks until ready for publication-grade sweeps.
- **Mock vs real data** → Set `ALLOW_MOCK_DATA=False` and provide actual Pantheon+/BOSS/Planck files to avoid accidental mock usage.

---

## 10. Citation & License

Please cite the repository when reusing the modular pipeline:

> Stefan Len. (2025). *TQE Dark Energy Coupling Simulation v4.2.0 PRO* (Version v4.2.0\_Pro) [Software]. Zenodo. https://doi.org/10.5281/zenodo.17614948

**BibTeX**
```bibtex
@software{Len_2025_TQE_DarkEnergy_v4_2,
  author    = {Len, Stefan},
  orcid     = {https://orcid.org/0009-0007-0383-7315},
  title     = {{TQE Dark Energy Coupling Simulation v4.2.0 PRO}},
  year      = {2025},
  publisher = {Zenodo},
  url       = {https://github.com/SteviLen420/TQE_simulation},
  doi       = {10.5281/zenodo.17614948},
  version   = {4.2.0}
}
```

**License:** MIT (see repository root `LICENSE`).

---

## 11. Related Files

- Monolithic reference implementation: `../TQE_DarkEnergy_Coupling_Simulation/TQE_DarkEnergy_Coupling_Simulation.py`
- Full pipeline README (non-modular): `../TQE_DarkEnergy_Coupling_Simulation/README.md`
- Universe modular reference: `../TQE_Universe_Modular/README_MODULAR.md`
- Research context: `../../TQE_Research_Notes.md`, `../../AI_METHODOLOGY.md`

---

*This modular build mirrors the scientific behavior of the monolithic TQE–ΛSim script while making it far easier to reason about, extend, and integrate into auxiliary tooling. Contributions and collaborative research are welcome.* 

