SPDX-License-Identifier: MIT

Copyright (c) 2025 Stefan Len

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![arXiv](https://img.shields.io/badge/arXiv-coming%20soon-b31b1b.svg)](#)
[![GitHub stars](https://img.shields.io/github/stars/SteviLen420/TQE_simulation?style=social)](https://github.com/SteviLen420/TQE_simulation)
[![GitHub forks](https://img.shields.io/github/forks/SteviLen420/TQE_simulation?style=social)](https://github.com/SteviLen420/TQE_simulation)
[![Research Status](https://img.shields.io/badge/status-active%20research-green)](https://github.com/SteviLen420/TQE_simulation)

# TQE UNIVERSE SIMULATION SUITE v4.2.0 PRO

**Title:** End-to-End Monte Carlo & Diagnostic Framework for the Theory of the Question of Existence  
**Author:** Stefan Len  
**Version:** v4.2.0 PRO  
**Contact:** stefan@tqe-theory.space

---

## Overview

The TQE (Theory of the Question of Existence) Universe Simulation Suite v4.2.0 PRO is a research-grade software stack designed to test the hypothesis that stable, complexity-permitting physical laws emerge from the coupling of vacuum energy fluctuations (E) with an information-theoretic orientation parameter (I). This monorepo collects every major component needed for large-scale experimentation:

1. **Full Pipeline:** `TQE_Universe_Simulation_Full_Pipeline/` – legacy monolithic script implementing all 28 phases end-to-end.
2. **Modular Pipeline:** `TQE_Pipeline_Modular/` – functionally identical to the monolith, reorganized into modules (config, core, phases, simulation, analysis).
3. **Analysis Suite:** `TQE_Analysis_Pipeline/` – post-simulation comparative analysis, ranking, and visualization.
4. **Heisenberg Microphysics:** `TQE_Energy_Fluctuations_Heisenberg/` – open-quantum-system experiment probing law lock-in vs. the Heisenberg bound.
5. **Diagnostic Harness:** `TQE_Universe_Simulation_Diagnose/` – integrity checks, dependency validation, smoke tests, and structured reporting.

Every submodule carries its own README and script, but this document provides a unifying view, the core theory summary, and cross-cutting usage notes.

---

## Theory Snapshot

The TQE hypothesis asserts that energy and information co-determine the stability of physical laws. The reasoning evolved through three conceptual steps:

1. **Consciousness-field Ansatz**  
   `P′(ψ) = P(ψ) · f(Φ)` – the Born rule augmented with a consciousness field Φ that biases collapse toward coherent, low-entropy states.
2. **Energy-Consciousness Coupling**  
   `P′(ψ) = P(ψ) · f(E, Φ)` – stability emerges when Φ is not an external dimension but an intrinsic property of energy.
3. **Energy–Information Coupling (final TQE equation)**  
   `P′(ψ) = P(ψ) · f(E, I)` – consciousness is replaced by an information orientation parameter I, measurable through information theory and quantum statistics.

This final form is what v4.2.0 PRO implements:

```
P'(ψ) = P(ψ) · f(E, I)
f(E, I) = exp(-(E - E_c)^2 / (2σ^2)) · (1 + α · I)
```

Where:
- `E` is vacuum energy (often mapped to ΩΛ).
- `I` is the intrinsic information orientation parameter (0–1).
- `E_c` is the Goldilocks energy center (stability sweet spot).
- `σ` controls the width of the stability window.
- `α` modulates the influence of information on law lock-in.
- `I` encapsulates directional bias toward complexity, derived from measurable information metrics rather than an undefined “consciousness field.”

### Parameter Breakdown

Practically, the simulations implement:
- **E sampling** via lognormal distributions (Planck-aligned), optionally truncated.
- **I computation** via 10 definitions (KL, Shannon, Rényi, mutual info, entanglement entropy, Fisher info, composite KL×Shannon, KL-Shannon harmonic mean, Fisher-KL fusion, Jensen–Shannon). All include dark-energy modulation `I_final = I_base × √(E_ref / E)`.
- **Coupling modes** (`E_plus_I`, `product`, `E_times_I_pow`) to derive the complexity parameter `X`.
- **Law lock-in detection** when fluctuations fall below `REL_EPS_LOCKIN` for `CALM_STEPS_LOCKIN` epochs (with E-only vs. E+I metrics).
- **CMB generation & anomalies** via CAMB + healpy, producing emergent cold spots and Axis-of-Evil alignments without forced Planck matching.

The suite is thus capable of running Monte Carlo universes, analyzing their lock-in dynamics, validating against Planck 2018 data, ranking information definitions, and probing microphysical suppression mechanisms.

---

## Repository Layout (v4.2.0 PRO)

```
TQE_Universe_Simulation_v4.2.0_Pro/
├── README.md                         # This document
├── README_HU.md                      # Hungarian translation (see below)
├── TQE_Universe_Simulation_Full_Pipeline/
│   ├── README.md                     # Monolithic pipeline docs
│   └── TQE_Universe_Simulation_Full_Pipeline_v4.2.0_Pro.py
├── TQE_Pipeline_Modular/
│   ├── README.md                     # Modular pipeline docs
│   ├── config/master_ctrl.py
│   ├── core/, phases/, simulation/, analysis/, utils/
│   └── main.py
├── TQE_Analysis_Pipeline/
│   ├── README.md
│   └── TQE_Analysis_Pipeline_v4.2.0_PRO.py
├── TQE_Energy_Fluctuations_Heisenberg/
│   ├── README.md
│   └── TQE_Energy_Fluctuations_Heisenberg_v4.2.0_Pro.py
└── TQE_Universe_Simulation_Diagnose/
    ├── README.md
    └── TQE_Universe_Simulation_Diagnose_v4.2.0_Pro.py
```

---

## Component Overview

| Module | Purpose | Highlights |
| --- | --- | --- |
| Full Pipeline | Legacy monolithic script | 4 run modes, 28 phases, deterministic seeding, CAMB/healpy integration |
| Modular Pipeline | Same functionality, structured modules | Maintains identical outputs, friendlier for collaboration and testing |
| Analysis Pipeline | Post-run comparative suite | 80+ metrics, triple rankings (stability / complexity / physical laws), 12+ visualization categories |
| Heisenberg Module | Microphysical experiment | Two-scenario Lindblad evolution, tri-mode I sweep, Δx·Δp monitoring |
| Diagnose Script | Integrity checks & smoke tests | Dependency validation, phase signature auditing, JSON/CSV reports |

Each README contains specific installation, configuration, and usage instructions tuned to that component.

---

## Quickstart Summary

1. **Install requirements** (Python 3.9+, core scientific stack, optional CAMB/healpy/qutip/dynesty/corner).
2. **Run a pipeline** (monolithic or modular) via `python TQE_Universe_Simulation_Full_Pipeline_v4.2.0_Pro.py` or `python -m TQE_Pipeline_Modular.main`.
3. **Post-process** a `batch_all` run using `python TQE_Analysis_Pipeline_v4.2.0_PRO.py`.
4. **Run microphysics** experiments via `python TQE_Energy_Fluctuations_Heisenberg_v4.2.0_Pro.py`.
5. **Validate the codebase** after refactors via `python TQE_Universe_Simulation_Diagnose_v4.2.0_Pro.py --deep --smoke`.

Detailed command snippets live inside each submodule’s README.

---

## Citation

```bibtex
@software{Len_2025_TQE_Suite_v4_2,
  author    = {Len, Stefan},
  title     = {{TQE Universe Simulation Suite v4.2.0 PRO}},
  year      = {2025},
  version   = {4.2.0},
  publisher = {GitHub},
  url       = {https://github.com/<repo>/TQE_simulation},
  doi       = {10.5281/zenodo.XXXXXXX}
}
```

---

## Next Steps

- Use the Diagnose tool before and after major merges to ensure the modular and monolithic pipelines remain synchronized.
- Leverage the Heisenberg module to generate priors for the main Goldilocks search and document the suppression ratios in the analysis reports.
- Contribute additional I-definitions or anomaly detectors by extending the modular pipeline (preferable) and verifying via the diagnostics + smoke run.

For questions or collaboration proposals, email **stefan@tqe-theory.space**. Contributions, issue reports, and scientific discussions are very welcome. Let’s keep pushing on the fundamental question the TQE framework asks: _why do complexity-permitting laws exist, and how did they stabilize?_ 

