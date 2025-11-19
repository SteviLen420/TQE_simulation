# TQE Heisenberg Modular Pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17627756.svg)](https://doi.org/10.5281/zenodo.17627756)
[![Execution](https://img.shields.io/badge/runtime-local__desktop-orange)](#)
[![Diagnostics](https://img.shields.io/badge/diagnostics-preflight%20%2B%20postrun-success)](#)
[![Status](https://img.shields.io/badge/status-active%20development-green)](#)

**Title:** TQE Heisenberg Fluctuation Simulation: Law Suppression of Vacuum Fluctuations (Modular)  
**Author:** Stefan Len  
**Version:** v4.2.0 PRO (Modular build)

---

## 0. Overview

This directory contains the **modularized** version of the TQE Energy Fluctuations Heisenberg Simulation.  
`TQE_Energy_Fluctuations_Heisenberg_v4.2.0_Pro.py` (monolithic script, 1891 lines) has been refactored into reusable modules for clean separation of concerns, easier testing, and better maintainability. The modular build keeps the scientific content identical while exposing clear APIs for:

- Running comparative analysis of NO-LAW vs WITH-LAW quantum fluctuations
- Testing all 3 I-origin models (emergent, inherent, threshold)
- Customizing quantum system parameters and simulation settings
- Generating publication-quality visualizations
- Local execution with Desktop output (no Colab dependency) and automatikus diagnosztika (futás előtt/után ellenőrzések)

---

## 1. Directory Layout

```
TQE_Heisenberg_Modular/
├── __init__.py
├── config.py                 # MASTER_CTRL: All configurable parameters
├── main.py                   # Entry point (package installation, setup, pipeline execution)
├── pipeline.py               # Main pipeline orchestrator (7 phases)
├── visualization.py          # All plotting functions (14 plots)
├── core/
│   ├── __init__.py
│   ├── tqe_functions.py      # TQE helper functions (f_lockin, Heisenberg uncertainty, coherence)
│   ├── quantum_system.py     # Quantum system setup (operators, Hamiltonians, collapse operators)
│   └── information_origin.py # I-origin models (emergent, inherent, threshold)
├── simulation/
│   ├── __init__.py
│   ├── trajectory.py         # Single trajectory evolution (run_single)
│   └── ensemble.py           # Initial state sampling (coherent states, I-parameter)
└── utils/
    ├── __init__.py
    ├── setup.py              # Package installation and reproducibility setup
    └── plotting.py           # Scientific plotting style configuration
```

Every module can be imported independently (e.g., `from TQE_Heisenberg_Modular.core.quantum_system import build_quantum_system`) enabling targeted notebooks or tests without executing the entire pipeline.

---

## 2. Module Guide

| Module | Purpose / Key Functions |
| --- | --- |
| `config.py` | Central `MASTER_CTRL` dictionary: ensemble size, quantum system parameters, TQE lock-in parameters, I-origin model settings, visualization options, performance knobs. |
| `core/tqe_functions.py` | TQE helper functions: `sample_info_beta()`, `f_lockin()`, `lockin_rate_scale()`, `lockin_potential_scale()`, `compute_heisenberg_uncertainty()`, `normalize_coherence()`. |
| `core/quantum_system.py` | `build_quantum_system()`: Creates quantum system operators, Hamiltonians, collapse operators for one/two-mode harmonic oscillators with anharmonic potentials, time-dependent drives, and thermal baths. |
| `core/information_origin.py` | I-origin models: `compute_I_emergent()`, `compute_I_inherent()`, `compute_I_threshold()`. Tests three hypotheses about information origin in physical systems. |
| `simulation/trajectory.py` | `run_single()`: Simulates one member of ensemble. Handles both NO-LAW (single-shot evolution) and WITH-LAW (segmented evolution with dynamic I adaptation) scenarios. |
| `simulation/ensemble.py` | `sample_coherent_states()`: Samples coherent state amplitudes with heavy-tailed lognormal distribution. |
| `pipeline.py` | `run_pipeline()`: Main orchestrator executing 7 phases: (1) NO-LAW simulation, (2) WITH-LAW simulation (3 I-modes), (3) Data aggregation, (4) Statistical analysis, (5) Data saving, (6) Visualization, (7) Parameter sweep (optional). |
| `visualization.py` | `generate_all_plots()`: Creates 14 publication-quality plots: energy/variance/entropy/coherence comparisons, Heisenberg uncertainty, phase space, multi-dimensional tracking, I-evolution plots for all 3 modes. |
| `utils/setup.py` | `check_and_install_packages()`, `setup_reproducibility()`: Dependency installation, deterministic seeding. |
| `utils/plotting.py` | `setup_scientific_plotting_style()`: Configures matplotlib for publication-quality figures. |

---

## 3. Theoretical Foundation

### Core TQE Hypothesis

**Why do stable physical laws prevent universe-spawning fluctuations?**

The TQE framework proposes that in a pre-law state (no stable physical constants), the Heisenberg Uncertainty Principle permits arbitrarily large vacuum fluctuations. Once stable laws emerge and lock in, these same laws **SUPPRESS** large-scale fluctuations, preventing new universes from forming within an existing universe.

**SCENARIO 1: PRE-LAW (NO LOCK-IN)**
- Pure quantum fluctuations without stabilizing mechanism
- Open quantum system with thermal bath, dephasing, damping
- No f(E,I) coupling applied
- Expected outcome: **LARGE, UNBOUNDED fluctuations**

**SCENARIO 2: WITH-LAW (WITH LOCK-IN)**
- Same quantum system + dynamics
- f(E,I) lock-in mechanism **ACTIVE**
- TQE coupling modulates dissipation rates and potential landscape
- Expected outcome: **SUPPRESSED, BOUNDED fluctuations**

### TQE Lock-In Mechanism

The f(E,I) function modulates system parameters online:

```
f(E,I) = exp(-(E-E_c)²/(2σ²)) · (1 + α·I)
```

**Dynamic adaptation:**
- Dissipation rates scaled by √f(E,I)
- Anharmonic potential strength scaled by f(E,I)
- Energy E = ⟨n₁⟩ + ⟨n₂⟩ (photon number proxy)
- Information I ~ Beta(a,b) (intrinsic orientation parameter)

### Information Origin Models

**ALL 3 MODELS RUN AUTOMATICALLY:**

1. **Emergent I**: Information emerges spontaneously from energy fluctuation structure
   ```
   I_{t+1} = γ·I_t + α·|ΔE_t| + β·corr(ΔE_t, ΔE_{t-1})
   ```

2. **Inherent I**: Information is deterministic function of energy
   ```
   I = scale · f(E)  where f = log(E/E0) or (E/E0)^γ or E
   ```

3. **Threshold I**: Information activates above critical energy
   ```
   I = 0 if E < E_c, else I += slope·(E - E_c)
   ```

---

## 4. Requirements & Environment

- **Python 3.9+** (tested on CPython 3.9–3.11)
- **Local execution only** (no Colab support)
- **Core dependencies**: `numpy`, `scipy`, `matplotlib`, `tqdm`, `qutip`
- `utils.setup.check_and_install_packages()` auto-installs missing packages

> **Note:** QuTiP is required for quantum system simulation. Installation may take 30-60 seconds on first run.

---

## 5. Quickstart

### As a module

```python
from TQE_Heisenberg_Modular import main

# Run pipeline with default settings
results = main.main()

# Or with config overrides
results = main.main(config_override={
    "N_ENSEMBLE": 50,  # Smaller ensemble for faster testing
    "T_FINAL": 10.0,   # Shorter evolution time
})
```

### Direct pipeline call

```python
from TQE_Heisenberg_Modular.pipeline import run_pipeline
from TQE_Heisenberg_Modular.config import MASTER_CTRL

# Optional overrides
config_override = {
    "N_ENSEMBLE": 100,
    "ENABLE_PARAMETER_SWEEP": False,
}

results = run_pipeline(config_override)
```

### Command line

```bash
cd TQE_Universe_Simulation_v4.2.0_Pro/TQE_Heisenberg_Modular
python -m TQE_Heisenberg_Modular.main
```

---

## 6. Output Structure

All results are saved to **Desktop**:

```
Desktop/TQE_Heisenberg_Modular_Results/
└── TQE_Heisenberg_Fluctuation_YYYYMMDD_HHMMSS/
    ├── summary.json                    # Complete run metadata + results + Heisenberg compliance
    ├── comparative_analysis.json       # NO-LAW vs WITH-LAW statistics + suppression ratios
    ├── data/
    │   ├── no_law_timeseries.csv      # Scenario 1 time-series data
    │   ├── with_law_timeseries.csv    # Scenario 2 time-series data
    │   ├── ensemble_final_energies.csv # Final states for all realizations
    │   └── parameter_sweep_{var}.csv  # Parameter sweep results (if enabled)
    └── figs/
        ├── 01_energy_comparison.png    # Energy evolution (both scenarios)
        ├── 02_variance_comparison.png  # Variance evolution
        ├── 03_entropy_comparison.png   # von Neumann entropy
        ├── 04_coherence_comparison.png # Quantum coherence (normalized)
        ├── 05_final_energy_dist.png    # Final energy distributions
        ├── 06_suppression_summary.png  # Summary bar chart
        ├── 07_heisenberg_uncertainty.png # Δx·Δp evolution + ℏ/2 limit
        ├── 08_phase_space_E_vs_S.png   # Phase space trajectories
        ├── 09_multidimensional_tracking.png # (E, S, C, Δx·Δp) 4-panel
        ├── 10_parameter_sweep_{var}.png # Parameter sweep (if enabled)
        ├── 11_I_evolution_emergent.png  # I(t) emergent model
        ├── 12_I_evolution_inherent.png  # I(t) inherent model
        ├── 13_I_evolution_threshold.png # I(t) threshold model
        └── 14_I_mode_comparison.png     # All 3 I-modes compared
```

**Total output:** 2 JSON + 3 CSV + 14 PNG (or 15 PNG if parameter sweep enabled)

---

## 7. Configuration

All parameters are defined in `config.py` (`MASTER_CTRL` dictionary). Key settings:

### Core Simulation Controls

```python
"N_ENSEMBLE": 100,              # Number of initial quantum states
"N_HILB": 20,                   # Fock space truncation per mode
"T_FINAL": 12.0,                # Total evolution time
"N_T": 300,                     # Number of time points
```

### Quantum System Features

```python
"ANHARMONIC_X4": True,          # Add λx⁴ term
"TWO_MODE_COUPLING": True,      # Second oscillator + coupling
"TIME_DEP_DRIVE": True,         # H(t) drive term
"THERMAL_BATH": True,           # Thermal Lindblad at nth > 0
```

### TQE Lock-In Parameters

```python
"EC": 25.0,                     # Goldilocks energy center
"SIGMA": 8.0,                   # Stability window width
"ALPHA": 0.8,                   # Information bias strength
"N_SEGMENTS": 12,               # Number of segmented evolution steps
```

### I-Origin Models

```python
"I_ORIGIN_MODE": "emergent",    # Reference mode (all 3 tested automatically)
"I_EMERGENT_ALPHA": 0.3,        # Weight for |ΔE_t| contribution
"I_EMERGENT_BETA": 0.2,         # Weight for autocorrelation contribution
"I_EMERGENT_GAMMA": 0.95,       # Decay factor (I persistence)
```

### Visualization

```python
"PLOT_DPI": 300,                # Figure DPI for high-quality output
"PLOT_FONTSIZE_TITLE": 14,
"PLOT_FONTSIZE_LABEL": 12,
"PLOT_FONTSIZE_LEGEND": 10,
```

---

## 8. Key Results

The pipeline computes **suppression ratios** to quantify law effectiveness:

- **Variance ratio** = σ²(WITH-LAW) / σ²(NO-LAW)
- **Uncertainty ratio** = Δx·Δp(WITH-LAW) / Δx·Δp(NO-LAW)
- **Coherence ratio** = C(WITH-LAW) / C(NO-LAW)

**Ratio < 1.0** → Laws suppress fluctuations (TQE prediction confirmed)

The pipeline also validates **Heisenberg uncertainty principle compliance**:
- Checks if Δx·Δp ≥ ℏ/2 for all trajectories
- Reports compliance status in `comparative_analysis.json`

---

## 9. Reproducibility

Each run generates a unique seed (or uses `MASTER_CTRL["SEED"]` if specified). The seed is saved in:
- `summary.json` → `run_info.seed`
- `comparative_analysis.json` → `run_metadata.master_seed`

To reproduce a run, set:
```python
MASTER_CTRL["SEED"] = <seed_value_from_json>
```

---

## 10. Performance

- **Serial execution** (multiprocessing disabled by default due to global variable dependencies)
- **Tensor operator caching** enabled by default (2-3x speedup)
- **Memory efficient mode** available (`MEMORY_EFFICIENT = True`)

Typical runtime on MacBook Pro M1 (32GB RAM):
- N_ENSEMBLE=100: ~5-10 minutes
- N_ENSEMBLE=200: ~10-20 minutes

---

## 11. Differences from Original Script

| Feature | Original | Modular |
| --- | --- | --- |
| **Colab support** | ✅ Yes | ❌ No (local only) |
| **Output location** | `SIMULATION_RUNS/heisenberg/` | `Desktop/TQE_Heisenberg_Modular_Results/` |
| **Structure** | Monolithic (1891 lines) | Modular (13 files) |
| **Importability** | ❌ No | ✅ Yes (all modules importable) |
| **Testing** | Difficult | Easy (isolated modules) |

---

## 12. Scientific Questions

1. **PRIMARY**: Can we demonstrate quantitatively that stable physical laws (represented by the TQE lock-in mechanism f(E,I)) act as a **SUPPRESSION** mechanism for large vacuum fluctuations, thereby explaining why new universes do not form within our existing cosmos?

2. **INFORMATION ORIGIN**: Where does the information parameter I come from?
   - **EMERGENT**: Does I emerge spontaneously from energy fluctuation structure?
   - **INHERENT**: Is I an inherent property of energy states (I = f(E))?
   - **THRESHOLD**: Does I activate only above a critical threshold (I at E > E_c)?

Each model tests a different hypothesis about the fundamental origin of information in physical systems.

---

## 13. License

MIT License - Copyright (c) 2025 Stefan Len

---

## 14. Citation

If you use this code in your research, please cite:

```
TQE Heisenberg Fluctuation Simulation v4.2.0 PRO (Modular)
Theory of the Question of Existence (TQE)
Author: Stefan Len
Year: 2025
```

---

## 15. Contact

For questions or issues, please refer to the main TQE simulation repository.
