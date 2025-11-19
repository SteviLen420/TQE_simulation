SPDX-License-Identifier: MIT

Copyright (c) 2025 Stefan Len

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17627756.svg)](https://doi.org/10.5281/zenodo.17627756)
[![Execution](https://img.shields.io/badge/runtime-local__desktop-orange)](#)
[![Planck Data](https://img.shields.io/badge/CMB%20input-%7E%2FDesktop%2FCMB__MAPS-informational)](#)
[![Diagnostics](https://img.shields.io/badge/diagnostics-preflight%20%2B%20postrun-success)](#)
[![Status](https://img.shields.io/badge/status-active%20development-green)](#)

# TQE Universe Simulation Suite v4.2.0 PRO

**Author:** Stefan Len  
**Contact:** stefan@tqe-theory.space

This repository hosts every active component of the TQE research stack. All pipelines run locally (no Colab), write their outputs either to `~/SIMULATION_RUNS` or to user Desktop folders, and rely on the shared Planck dataset located in `~/Desktop/CMB_MAPS`. The repository also keeps the original monolithic scripts archived for reference.

---

## Repository overview

| Directory | Role | Outputs / Notes |
| --- | --- | --- |
| `TQE_Universe_Simulation_Modular/` | 28-phase universe simulation (E-only, E+I, batch) | `~/SIMULATION_RUNS/universe/...` + `~/Desktop/TQE_Universe_Simulation_Modular_Results/...` |
| `TQE_DarkEnergy_Modular/` | Dark-energy coupling pipeline (Covariant pressure, uniform w, geometric, ΛCDM) | `~/SIMULATION_RUNS/dark_energy/...` + `~/Desktop/TQE_DarkEnergy_Modular_Results/...` |
| `TQE_Heisenberg_Modular/` | Heisenberg fluctuation suppression simulation (NO-LAW vs WITH-LAW) | `~/SIMULATION_RUNS/heisenberg/...` + `~/Desktop/TQE_Heisenberg_Modular_Results/...` |
| `TQE_Analysis_Modular/` | Post-run comparative analysis for batch universes | `~/SIMULATION_RUNS/analysis/...` |
| `TQE_Universe_Simulation_Full_Pipeline/` | Legacy monolithic pipeline (read-only reference) | Original script + Planck TT spectrum |

---

## External data dependencies

- **Planck maps/masks:** `~/Desktop/CMB_MAPS`
  - `CMB_Maps/`: SMICA, NILC, SEVEM, Commander (I/Q/U, R3).
  - `CMB_Raw_Skymap/` + `CMB_Raw_Skymap_New/`: HFI/LFI frequency maps.
  - `CMB_Mask/`: common intensity/polarization masks + misspix masks.
  - `CMB_Anomaly/NHI_HPX.fits`: neutral hydrogen map for anomaly correlation.
  - `planck_data/COM_PowerSpect_CMB-TT-full_R3.01.txt`: TT spectrum for validation.
- Each modular pipeline references this directory explicitly (`PLANCK_DATA_PATH` or `CMB_PLANCK_BASE_PATH`), so no copies live inside the repo.

---

## Pipeline summaries

### Universe modular (`TQE_Universe_Simulation_Modular/`)
- Implements the 28-phase pipeline (Monte Carlo → Goldilocks → CMB generation → anomaly detection → final summary).
- Supports single modes, batch modes, multi-I analyses, and runs diagnostics before/after execution.
- Outputs: aggregated CSV/JSON, categorized results, PNG visualizations, Planck validation artifacts.
- Default run command:
  ```bash
  python -m TQE_Universe_Simulation_v4.2.0_Pro.TQE_Universe_Simulation_Modular.main
  ```

### Dark Energy modular (`TQE_DarkEnergy_Modular/`)
- Runs all four cosmological models (Covariant Pressure, Uniform w, Geometric, Null ΛCDM) in E-only/E+I/dual configurations.
- Handles Pantheon+/BAO/Planck data, MCMC + nested sampling, auto-aggregation, and dual-mode comparisons.
  ```bash
  python -m TQE_Universe_Simulation_v4.2.0_Pro.TQE_DarkEnergy_Modular.main
  ```

### Heisenberg modular (`TQE_Heisenberg_Modular/`)
- Compares NO-LAW vs WITH-LAW trajectories across all three information-origin models (emergent/inherent/threshold).
- Saves comparative JSON, time-series CSVs, 14+ PNG plots, and optional parameter sweep results.
  ```bash
  python -m TQE_Universe_Simulation_v4.2.0_Pro.TQE_Heisenberg_Modular.main
  ```

### Analysis modular (`TQE_Analysis_Modular/`)
- Discovers the latest batch runs inside `~/SIMULATION_RUNS/universe`, builds comprehensive metric tables, and produces triple-ranking outputs plus visualizations.
  ```bash
  python -m TQE_Universe_Simulation_v4.2.0_Pro.TQE_Analysis_Modular.main
  ```

Every README inside these modules documents the configuration knobs (`config.py` / `master_ctrl.py`), output layout, and troubleshooting tips. Diagnostics now run in all modular pipelines to catch missing data, permission issues, or incomplete outputs.

---

## SIMULATION_RUNS structure

```
SIMULATION_RUNS/
├── universe/      # Universe modular runs (per mode/timestamp)
├── dark_energy/   # Dark energy modular runs
├── heisenberg/    # Heisenberg modular runs
└── analysis/      # Analysis results (batch comparisons)
```

Each subdirectory uses timestamped folders to keep runs isolated. For long-term storage or manual inspection, the Desktop folders retain the full artifact sets.

---

## Contact & citation

Please cite the suite when using these pipelines in research:

```bibtex
@software{Len_2025_TQE_Suite_v4_2,
  author    = {Len, Stefan},
  orcid     = {https://orcid.org/0009-0007-0383-7315},
  title     = {{TQE Universe Simulation Suite v4.2.0 PRO}},
  year      = {2025},
  version   = {4.2.0},
  publisher = {Zenodo},
  url       = {https://github.com/SteviLen420/TQE_simulation},
  doi       = {10.5281/zenodo.17627756}
}
```

Questions or collaboration proposals: **stefan@tqe-theory.space**. The modular stack is under active development; check each subdirectory’s README for up-to-date instructions.

