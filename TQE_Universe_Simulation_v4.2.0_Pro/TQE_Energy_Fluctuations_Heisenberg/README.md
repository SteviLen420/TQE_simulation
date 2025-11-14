[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![arXiv](https://img.shields.io/badge/arXiv-coming%20soon-b31b1b.svg)](https://arxiv.org/)
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

- **Scenario Pairing:** Automatically runs NO-LAW vs WITH-LAW using the same Hamiltonian, drives, and bath parameters for apples-to-apples suppression ratios.  
- **Open-Quantum Dynamics:** QuTiP Lindblad integration with amplitude damping, dephasing, and thermal baths; optional quantum trajectory unraveling for Monte Carlo ensembles.  
- **TQE Lock-In Coupling:** f(E,I)=exp(-(E-Ec)²/(2σ²))(1+αI) applied online to dissipation rates, potential landscape, and drives. Energy expectation ⟨n⟩ serves as the live E proxy.  
- **Information-Origin Modes:** Emergent (I driven by |ΔE| and correlations), inherent (I=f(E)), and threshold (I activates above Ec); all three tracked and plotted per run.  
- **Heisenberg Compliance:** Continuous monitoring of Δx, Δp, Δx·Δp vs ℏ/2 with red-line alerts if uncertainty bounds are violated.  
- **Multi-Metric Analysis:** Energy mean/variance/max, von Neumann entropy, coherence, Jensen–Shannon drift, suppression ratios, and phase-space trajectories.  
- **Parameter Sweeps:** Optional sweeps over Ec, σ, α, bath temperatures, or drive strengths with aggregated CSV + PNG exports.  
- **Publication-Ready Outputs:** Standardised folder structure with summary JSON, comparative tables, per-scenario CSVs, and 10–14 plots reusable for manuscripts.  
- **Reproducible Seeding:** Deterministic master seed and per-trajectory seeds ensure bit-for-bit reruns.  
- **Planck-Aware Context:** Designed to complement the main TQE Universe pipeline—summary JSON includes compatibility hooks consumed later by the analysis pipeline.

---

## Installation

Recommended Python **3.10+** (any 3.8+ works). Create a virtualenv or conda env and install requirements:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip wheel setuptools
pip install numpy matplotlib scipy tqdm qutip
```

Optional packages for extended diagnostics:

- `pandas` – CSV aggregation / sweeps  
- `seaborn` – nicer plotting styles  
- `numba` – experimental acceleration of custom propagators

> **QuTiP note:** first install may take ~1 min; ensure build tools (gcc/clang) are available on your platform.

---

## Configuration Overview

All controls reside at the top of `TQE_Energy_Fluctuations_Heisenberg.py` inside the `MASTER_CTRL` dictionary. Key blocks:

- `SCENARIOS`: enable/disable NO-LAW, WITH-LAW, or mixed runs.  
- `QUANTUM_SYSTEM`: oscillator frequencies, anharmonic strength λ, coupling g, drive amplitude/frequency.  
- `OPEN_SYSTEM`: damping rates κ, dephasing γ, bath temperature (nth), unraveling toggles.  
- `LOCKIN`: Ec, σ, α, and information-origin mode weights.  
- `SIMULATION`: number of time steps, dt, ensemble size, checkpoint frequency.  
- `OUTPUT`: run ID prefix, figure DPI, CSV/JSON toggles.

Set the environment variable `TQE_HEISENBERG_PROFILE` to switch between predefined profiles (e.g., `demo`, `paper`).

---

## Running the Simulation

Default single-run:

```bash
cd TQE_Energy_Fluctuations_Heisenberg
python TQE_Energy_Fluctuations_Heisenberg.py
```

Typical CLI options (via env vars):

```bash
export TQE_HEISENBERG_PROFILE=paper
export TQE_HEISENBERG_NUM_TRAJ=256
python TQE_Energy_Fluctuations_Heisenberg.py --sweep alpha --values 0.4 0.8 1.2
```

When sweeps are requested, each parameter value produces its own subfolder under the timestamped run directory.

---

## Output Structure

```
TQE_Heisenberg_Fluctuation_<timestamp>/
├── summary.json
├── comparative_analysis.json
├── config_snapshot.json
├── data/
│   ├── no_law_timeseries.csv
│   ├── with_law_timeseries.csv
│   ├── ensemble_final_energies.csv
│   └── sweep_<param>.csv
└── figs/
    ├── 01_energy_comparison.png
    ├── ...
    └── 14_I_mode_comparison.png
```

Each JSON file includes metadata (commit hash, master seed, scenario settings) so the analysis pipeline can chain results downstream.

---

## Interpretation Checklist

1. **Variance Suppression:** `variance_ratio < 1` indicates laws suppress large fluctuations.  
2. **Entropy Growth:** Compare slopes; WITH-LAW should plateau earlier.  
3. **Heisenberg Compliance:** Δx·Δp stays ≥ ℏ/2; WITH-LAW should hug the limit more tightly.  
4. **Coherence:** Decay slowed in presence of lock-in → indicates stabilised quantum phase.  
5. **Information Origin:** Emergent vs inherent vs threshold plots reveal how I behaves relative to E.  
6. **Parameter Sweeps:** Look for critical Ec/σ where suppression sharply increases.

---

## Relationship to Main TQE Pipeline

- Serves as a pre-law microphysical testbed; suppression ratios feed into the universe-scale `TQE_Universe_Simulation_Full_Pipeline`.  
- Output JSON/CSV files are compatible with the master analysis pipeline (Phase 11–20) for comparative visualisations.  
- Fine-tuning metadata (Ec, σ, α, information mode) is logged to `summary.json` for reproducibility and traceability.

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