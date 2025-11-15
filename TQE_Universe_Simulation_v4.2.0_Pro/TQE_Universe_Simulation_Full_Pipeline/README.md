SPDX-License-Identifier: MIT

Copyright (c) 2025 Stefan Len

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX) <!-- TODO: replace with real DOI when minted -->
[![GitHub stars](https://img.shields.io/github/stars/SteviLen420/TQE_simulation?style=social)](https://github.com/SteviLen420/TQE_simulation)
[![GitHub forks](https://img.shields.io/github/forks/SteviLen420/TQE_simulation?style=social)](https://github.com/SteviLen420/TQE_simulation)
[![Research Status](https://img.shields.io/badge/status-active%20research-green)](https://github.com/SteviLen420/TQE_simulation)

# TQE UNIVERSE SIMULATION PIPELINE v4.2.0 PRO

**Title: The TQE Framework: A Monolithic, Reproducible Pipeline for Monte Carlo Simulation of Universe Evolution from Energy-Information Principles**

> **Monolithic focus.** This README documents the legacy, single-script implementation (`TQE_Universe_Simulation_Full_Pipeline_v4.2.0_Pro.py`). All 28 phases run sequentially inside one orchestrator, making it ideal for straightforward, end-to-end reproductions or Colab execution. For modularized development, testing, or selective phase execution, see the companion `TQE_Pipeline_Modular/README.md`.

**Author**: Stefan Len

**Version**: v4.2.0 PRO

---

## Abstract

The TQE (Theory of the Question of Existence) Framework v4.2.0 PRO is a comprehensive computational pipeline designed to investigate the hypothesis that stable, complexity-permitting physical laws emerge from the coupling of vacuum energy fluctuations (E) with an information-theoretic orientation parameter (I). This advanced version represents a significant evolution from earlier iterations, featuring a complete 28-phase analysis pipeline, Bayesian adaptive optimization, enhanced physics engine, and comprehensive anomaly and law detection capabilities.

The framework provides an end-to-end environment for conducting large-scale Monte Carlo simulations of universe ensembles, systematically exploring the parameter space of initial conditions where Energy (E) and Information (I) interact to determine cosmic evolution. The core simulation models the complete lifecycle of universes: from an initial pre-collapse phase where physical laws fluctuate stochastically, through a critical "law lock-in" event where stable constants are selected, to a subsequent expansion phase that generates cosmological observables.

**Key Features of v4.2.0 PRO:**

- **28-Phase Comprehensive Pipeline**: From Monte Carlo simulation through advanced statistical analysis, anomaly detection, and Bayesian model selection
- **Bayesian Adaptive Goldilocks Optimization**: Gaussian Process Regression with UCB acquisition function for intelligent parameter space exploration
- **10 I-Parameter Definitions**: KL-divergence, Shannon entropy, Rényi entropy, Mutual Information, Entanglement Entropy, Fisher Information, composite KL×Shannon product, KL-Shannon refined, Fisher–KL fusion, and symmetric Jensen–Shannon divergence (all with E-modulation)
- **4 Execution Modes**: Single E-only baseline, single E+I coupling, batch E+I comparison, and comprehensive batch_all mode
- **Enhanced Physics Engine**: Integration with CAMB for realistic CMB generation, Friedmann evolution, quantum field fluctuations, and cosmic entanglement networks
- **Comprehensive Anomaly Detection**: CMB anomalies (Cold Spot, Axis of Evil) plus advanced quantum, entropy, topological, and information theory anomalies
- **Advanced Law Detection**: Automatic detection of conservation laws, symmetry laws, scaling laws, emergent laws, quantum laws, and more
- **Bayesian Model Selection**: BIC, AIC, and nested sampling for rigorous model comparison and I-definition ranking
- **Emergent CMB Maps**: Fully emergent anomaly patterns (no forced Planck matching) for genuine TQE validation
- **Self-Healing CMB Aggregates**: Variant-aware loaders automatically retrieve run-specific cold spot & AOE catalogues and regenerate aggregate Mollweide density maps (with healpy auto-install fallback)
- **Planck-Aware Setup**: Automatic Planck TT spectrum download or surrogate synthesis (CAMB-backed or analytic) when the reference file is missing, keeping validation fully automated
- **Complexity & Life-Compatibility Analytics**: Integrated component synthesis (lock-in quality, Goldilocks precision, information richness, Planck fit, stability quality, robustness) plus top-universe ranking exports
- **Planck-Constrained Fine-Tuning**: Gradient-based E/I adjustments with adaptive strength, jitter, E–I correlation, historical feedback, and χ²/α calibration to reach Planck targets without forcing trajectories
- **Deterministic Reproducibility & Manifests**: Master + per-universe seeding chain, timestamped run directories, aggregate CSV/JSON/PNG exports, and machine-readable manifests feeding the analysis pipeline
- **Planck Validation Workflow**: Automatic TT spectrum acquisition/synthesis, per-map amplitude calibration (α), χ²/dof computation with priors, Planck proximity scoring, and persistent fine-tuning history logging

The pipeline is architected for reproducibility, extensibility, and scientific rigor, making it a research-grade tool for theoretical cosmology and the investigation of emergent physical laws.

---

## How to Cite

If you use this software in your research, please consider citing it. This helps to acknowledge the work and allows others to discover and reproduce your results. The `CITATION.cff` file in the root of this repository is provided for automated citation management.

**Plain Text Citation:**
> Stefan Len. (2025). *TQE Universe Simulation Pipeline v4.2.0 PRO* (Version 4.2.0) [Software]. GitHub. <!-- TODO: Add actual GitHub URL when available -->

Author ORCID: https://orcid.org/0009-0007-0383-7315

**BibTeX Entry:**

```bibtex
@software{Len_2025_TQE_v4.2.0_PRO,
  author    = {Len, Stefan},
  orcid     = {https://orcid.org/0009-0007-0383-7315},
  title     = {{TQE Universe Simulation Pipeline v4.2.0 PRO}},
  version   = {4.2.0},
  date      = {2025},
  publisher = {GitHub},
  url       = {<!-- TODO: Add actual GitHub URL when available -->},
  doi       = {<!-- TODO: Add actual DOI when available -->}
}
```

---

## Installation & Environment Setup

The TQE Framework v4.2.0 PRO requires **Python 3.9+** and several scientific libraries. We strongly recommend setting up a dedicated virtual environment to ensure reproducibility and avoid dependency conflicts.

### Prerequisites

- **Python**: 3.9, 3.10, or 3.11 (3.9+ required)
- **Operating System**: Linux, macOS, or Windows
- **Build Tools** (for optional dependencies):
  - **Linux**: `build-essential`, `python3-dev`, `gfortran`
  - **macOS**: Xcode Command Line Tools (`xcode-select --install`)
  - **Windows**: Recent Python from python.org with build tools

### Using pip (Recommended)

```bash
# 1. Clone the repository
git clone <!-- TODO: Add actual GitHub repository URL when available -->
cd TQE_simulation/TQE_Universe_Simulation_v4.2.0_Pro/TQE_Universe_Simulation_Full_Pipeline

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate

# 3. Upgrade pip tooling
pip install -U pip wheel setuptools

# 4. Install core dependencies
pip install numpy pandas matplotlib scikit-learn scipy tqdm

# 5. Install optional dependencies (recommended for full functionality)
pip install healpy camb qutip shap lime dynesty corner
```

### Using Conda (Alternative)

```bash
# 1. Clone the repository
git clone <!-- TODO: Add actual GitHub repository URL when available -->
cd TQE_simulation/TQE_Universe_Simulation_v4.2.0_Pro/TQE_Universe_Simulation_Full_Pipeline

# 2. Create conda environment
conda env create -f environment.yml  # TODO: Create environment.yml if needed
conda activate tqe_env

# 3. Install remaining dependencies
pip install -r requirements.txt  # TODO: Create requirements.txt if needed
```

### Optional Dependencies

The following packages enable additional functionality:

- **qutip**: Enables quantum-mechanical modeling of pre-collapse dynamics and superposition phases
- **healpy**: Required for generating CMB-like sky maps in HEALPix format
- **camb**: Enables realistic CMB power spectrum generation using the Code for Anisotropies in the Microwave Background
- **shap, lime**: Required for Explainable AI (XAI) analysis and model interpretability
- **dynesty, corner**: Required for Bayesian nested sampling and corner plots (PRO features)

💡 **Installation Tips:**
- If `healpy` build fails, try installing `astropy` first: `pip install astropy`, then `pip install healpy`
- For Windows users, if venv activation is blocked, run: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
- Conda is often more reliable for binary dependencies like `healpy` and `camb` on some platforms

---

## Quickstart / Usage

The simulation pipeline is launched by running the main script directly. Its behavior is fully customizable through the `MASTER_CTRL` Python dictionary located at the top of the pipeline file.

### 1. Basic Execution

The simplest way to run the pipeline is with default settings:

```bash
cd TQE_Universe_Simulation_v4.2.0_Pro/TQE_Universe_Simulation_Full_Pipeline
python TQE_Universe_Simulation_Full_Pipeline_v4.2.0_Pro.py
```

This will execute a single E+I simulation run with the default I-definition (`jensen_shannon`) and generate all 28 phases of analysis.

### 2. Execution Modes

The pipeline supports four distinct execution modes, controlled by the `RUN_MODE` parameter in `MASTER_CTRL`:

#### Mode 1: Single E-Only (`single_eonly`)
Baseline mode for comparison - Energy-only coupling with I parameter disabled.

```python
MASTER_CTRL = {
    "RUN_MODE": "single_eonly",
    "NUM_UNIVERSES": 250,
    # ... other settings
}
```

#### Mode 2: Single E+I (`single_ei`)
TQE coupling mode with one selected I-definition.

```python
MASTER_CTRL = {
    "RUN_MODE": "single_ei",
    "I_DEFINITION_MODE": "jensen_shannon",  # or any of the 10 definitions
    "NUM_UNIVERSES": 250,
    # ... other settings
}
```

#### Mode 3: Batch E+I (`batch_ei`)
Runs all 10 I-definitions independently for comparison.

```python
MASTER_CTRL = {
    "RUN_MODE": "batch_ei",
    "NUM_UNIVERSES": 250,
    # ... other settings
}
```

#### Mode 4: Batch All (`batch_all`)
Comprehensive comparison: E-only baseline + all 10 I-definitions (11 total runs).

```python
MASTER_CTRL = {
    "RUN_MODE": "batch_all",
    "NUM_UNIVERSES": 250,
    # ... other settings
}
```

### 3. Modify Settings

Open `TQE_Universe_Simulation_Full_Pipeline_v4.2.0_Pro.py` in a text editor and modify the `MASTER_CTRL` dictionary:

```python
MASTER_CTRL = {
    # Core Pipeline Controls
    "RUN_MODE": "single_ei",           # Execution mode
    "I_DEFINITION_MODE": "jensen_shannon",  # I-parameter definition
    "NUM_UNIVERSES": 1000,             # Number of universes to simulate
    "SEED": None,                      # Master seed (None = auto-generate)
    
    # Goldilocks Optimization
    "BAYESIAN_UCB_KAPPA": 2.0,        # Exploration-exploitation trade-off
    "BAYESIAN_GP_NOISE": 0.01,        # GP noise level
    
    # Epoch Settings
    "TIME_STEPS": 1000,                # Stability run epochs
    "LOCKIN_EPOCHS": 500,              # Lock-in dynamics epochs
    "EXPANSION_EPOCHS": 1000,          # Expansion dynamics epochs
    
    # ... see Configuration Parameters section for complete list
}
```

### 4. Output Location

Results are saved to a timestamped directory:
- **Local**: `TQE_Universe_Simulation_Full_Pipeline_v4.2.0_PRO/TQE_Universe_Simulation_Full_Pipeline_{mode}_{timestamp}/`
- **Google Colab**: `/content/drive/MyDrive/TQE_Universe_Simulation_Full_Pipeline_v4.2.0_PRO/...`

Each run generates:
- 55+ PNG visualization plots
- 35+ CSV data files
- 3+ JSON summary files (including complexity & life-compatibility metrics)
- 20+ FITS/NPY CMB maps (for lock-in universes)

---

## Configuration Parameters

The behavior of the TQE Framework is controlled by the central `MASTER_CTRL` Python dictionary within the pipeline script. This configuration-as-code approach enables experimental campaigns, parameter sweeps, and analysis settings to be defined directly within the source code, providing a clear foundation for reproducibility.

### Core Pipeline Controls

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `RUN_MODE` | string | `"single_ei"` | Execution mode: `"single_eonly"`, `"single_ei"`, `"batch_ei"`, or `"batch_all"` |
| `I_DEFINITION_MODE` | string | `"jensen_shannon"` | Active I-definition (used if `RUN_MODE = "single_ei"`). Options: `"kl_divergence"`, `"shannon"`, `"renyi"`, `"mutual_info"`, `"composite"`, `"kl_shannon"`, `"entanglement"`, `"fisher"`, `"fisher_kl_fusion"`, `"jensen_shannon"` |
| `NUM_UNIVERSES` | int | `250` | Number of universes to simulate in the Monte Carlo run (default: 250, recommended: 1000-10000 for production runs) |
| `SEED` | int/None | `None` | Master seed for reproducibility. If `None`, auto-generated |
| `PIPELINE_VARIANT` | string | `"full"` | Auto-set by `RUN_MODE`: `"full"` (E+I) or `"energy_only"` (E-only) |

### Goldilocks Optimization (Bayesian Adaptive)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `CALIBRATION_EPOCHS` | int | `500` | Epochs per stability check during Bayesian sampling |
| `BAYESIAN_UCB_KAPPA` | float | `2.0` | Exploration-exploitation trade-off (higher = more exploration). Recommended: 1.5 (aggressive), 2.0 (balanced), 3.0 (conservative) |
| `BAYESIAN_GP_NOISE` | float | `0.01` | GP noise level for robustness. Recommended: 0.005 (low noise), 0.01 (balanced), 0.05 (high noise) |

**Budget Allocation Strategy:**
- 30% of `NUM_UNIVERSES` used for Bayesian Goldilocks discovery (exploration)
- 70% of `NUM_UNIVERSES` used for full simulation in discovered zone (exploitation)
- Example: `NUM_UNIVERSES=1000` → 300 (Bayesian) + 700 (full sim) = 1000 total

### Epoch Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `TIME_STEPS` | int | `1000` | Stability run epochs |
| `LOCKIN_EPOCHS` | int | `500` | Lock-in dynamics epochs |
| `EXPANSION_EPOCHS` | int | `1000` | Expansion dynamics epochs |
| `FL_EXP_EPOCHS` | int | `2000` | Fluctuation expansion panel epochs |

### Coupling Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `X_SCALE` | float | `20.0` | E-I coupling scale factor |
| `ALPHA_I` | float | `0.9` | I coupling strength |
| `X_MODE` | string | `"E_plus_I"` | Coupling mode: `"E_plus_I"`, `"product"`, or `"E_times_I_pow"` |

### Stability Thresholds

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `REL_EPS_STABLE` | float | `0.035` | Stability threshold (relative variation) |
| `REL_EPS_LOCKIN` | float | `0.040` | Lock-in threshold (relative variation) |
| `CALM_STEPS_STABLE` | int | `4` | Consecutive calm steps required for stability |
| `CALM_STEPS_LOCKIN` | int | `3` | Consecutive calm steps required for lock-in |
| `MIN_LOCKIN_EPOCH` | int | `120` | Minimum epoch for lock-in to occur |

### CMB Generation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `CMB_NSIDE` | int | `128` | HEALPix resolution parameter (higher = finer resolution) |
| `CMB_POWER_SLOPE` | float | `2.5` | Power spectrum slope (Pk ~ k^-slope) |
| `CMB_AMPLITUDE_SCALE` | float | `5e-10` | Overall amplitude of CMB fluctuations |
| `CAMB_INTEGRATION` | bool | `False` | Use CAMB for realistic CMB power spectra (requires `camb` package) |
| `ENABLE_PHYSICAL_ANOMALIES` | bool | `False` | DISABLED: Anomalies should emerge naturally, not be added artificially |

### Bayesian Analysis Parameters (PRO Features)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ENABLE_BAYESIAN_ANALYSIS` | bool | `True` | Enable Bayesian Model Selection (BIC, AIC, Bayes Factor) |
| `ENABLE_NESTED_SAMPLING` | bool | `True` | Enable Nested Sampling for Bayesian Evidence computation |
| `NESTED_SAMPLING_NLIVE` | int | `1000` | Number of live points for nested sampling |
| `NESTED_SAMPLING_DLOGZ` | float | `0.1` | Stopping criterion for nested sampling |
| `ENABLE_CORNER_PLOTS` | bool | `True` | Enable corner plots for parameter posterior distributions |

### Performance Optimization

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `USE_MULTIPROCESSING` | bool | `True` | Enable parallel universe simulation |
| `MAX_WORKERS` | int/None | `None` | Worker count (None = auto-detect CPU cores) |
| `ENABLE_CMB_CACHE` | bool | `True` | Cache CMB maps for faster repeated generation |
| `CMB_CACHE_SIZE` | int | `1000` | LRU cache size for CMB maps |
| `REDUCE_MEMORY_USAGE` | bool | `True` | Reduce memory usage (useful for Colab) |

### Output & I/O Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `SAVE_FIGS` | bool | `True` | Save plots to disk |
| `SAVE_JSON` | bool | `True` | Save summary JSON files |
| `DRIVE_BASE_DIR` | string | `"/content/drive/MyDrive/..."` | Google Drive base directory (Colab) |
| `COLAB_OPTIMIZED` | bool | `True` | Enable Colab-specific optimizations |
| `VERBOSE` | bool | `False` | Extra prints/logs (set to `True` for debugging) |

### Advanced Feature Toggles

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `USE_ENHANCED_PHYSICS` | bool | `True` | Enable enhanced physics engine (Friedmann, quantum fields, entanglement) |
| `RUN_PLANCK_VALIDATION` | bool | `True` | Run Planck 2018 observational comparison (Phase 15) |
| `ENABLE_QUANTUM_ANOMALY_DETECTION` | bool | `True` | Enable quantum field anomaly detection (Phase 25) |
| `ENABLE_CONSERVATION_LAW_DETECTION` | bool | `True` | Enable conservation law detection (Phase 26) |
| `COMPUTE_ALL_I_DEFINITIONS` | bool | `True` | Enable CSV export for all 10 I-definitions |

**Note:** For a complete list of all available parameters, see the `MASTER_CTRL` dictionary in the pipeline script (lines 536-1033). The configuration is extensively documented with inline comments explaining each parameter's purpose and recommended values.

---

## Computational Framework & Methodology

The TQE Framework v4.2.0 PRO is structured as a comprehensive, multi-stage computational pipeline for the systematic investigation of emergent physical laws. Its design emphasizes configuration-driven execution, reproducibility, and modularity, with a sophisticated architecture that supports both single-run experiments and large-scale batch comparisons.

### High-Level Architecture

The workflow is orchestrated by a central Python dictionary named `MASTER_CTRL`, located at the top of the main pipeline script. This configuration-as-code approach enables experimental campaigns—including parameter sweeps, I-definition comparisons, and analysis settings—to be defined directly within the source code. This provides a clear and direct foundation for reproducibility and transparent experimental design.

The pipeline is organized into four sequential conceptual stages:

**1. Generation** – Reads simulation parameters from the `MASTER_CTRL` dictionary and generates initial conditions for an ensemble of universes. Each universe is defined by Energy (E) and Information (I) values drawn from statistical distributions. The `PhysicsEngine` class handles all physical computations, including E-I coupling, I-parameter definitions, and CMB generation.

**2. Simulation** – The computational core of the framework. It evolves universes through pre-collapse, law lock-in, and expansion phases using Monte Carlo methods. This stage is computationally intensive and includes support for parallel execution via multiprocessing. The simulation incorporates Bayesian adaptive optimization to intelligently explore the parameter space and discover optimal Goldilocks zones.

**3. Analysis** – Performs comprehensive post-processing on raw outputs, including calculation of cosmological observables, generation of CMB-like sky maps, execution of diagnostic tests to score universes for fine-tuning, and targeted scans for selected anomalies. The analysis suite spans 28 distinct phases, from basic stability analysis to advanced law detection.

**4. Interpretation** – The pipeline includes advanced analysis modules for applying machine learning, Bayesian model selection, and interpretability techniques. The goal is to uncover causal relationships between initial (E,I) conditions and emergent characteristics, providing insights into the mechanisms driving universe stability and law emergence.

### PipelineContext Architecture

The `PipelineContext` class encapsulates all transient state and global configurations for a single pipeline run, eliminating global variables and providing a clean, object-oriented interface. Key responsibilities include:

- **Seed Management**: Master seed generation and per-universe seed derivation for deterministic reproducibility
- **Path Management**: Automatic directory structure creation with variant tagging (E-only vs E+I)
- **File I/O**: Centralized saving of figures, CSV files, and JSON summaries with automatic categorization
- **Runtime Registries**: Tracking of CMB maps, universe categories, and other runtime data

### PhysicsEngine Architecture

The `PhysicsEngine` class encapsulates all physical computations related to E, I, X, and CMB generation. It provides:

- **Energy Sampling**: Lognormal distribution sampling with physical model support
- **Information Parameter Definitions**: 10 distinct I-definition methods, all with E-modulation (dark energy coupling)
- **E-I Coupling**: Flexible coupling modes (`E_plus_I`, `product`, `E_times_I_pow`)
- **CMB Generation**: Integration with CAMB for realistic power spectra, plus fallback methods
- **Enhanced Physics**: Friedmann evolution, quantum field fluctuations, cosmic entanglement networks

This staged structure conceptually supports re-running individual components (e.g., re-analyzing simulation data with a new anomaly detector) without repeating upstream steps, although the current implementation realizes this in a streamlined, script-based form optimized for end-to-end execution.

---

## Reproducibility by Design: The Seeding Hierarchy

To ensure computational reproducibility, the framework implements a sophisticated two-tiered seeding hierarchy. This design provides deterministic outcomes within a fixed software environment, supporting verifiable and repeatable scientific workflows.

### Master Seed

A single master seed is defined by the `SEED` key in the `MASTER_CTRL` dictionary. If set to an integer, it initializes a master pseudo-random number generator (PRNG) for the entire run. If left as `None`, a random 32-bit seed is automatically generated and saved, ensuring each run has a unique but reproducible identifier.

The master seed is used to:
- Initialize the main `numpy.random.Generator` (modern RNG)
- Synchronize legacy `numpy.random` state (for libraries like QuTiP)
- Set environment variables for strict determinism (`PYTHONHASHSEED=0`, thread limits)

### Per-Universe Seeds

The master PRNG is used to deterministically generate a unique seed for each of the `NUM_UNIVERSES` in the ensemble using `numpy.random.SeedSequence.spawn()`. This ensures that:

- Each universe's stochastic processes are initialized independently
- The same master seed produces the same per-universe seeds
- Individual universes can be reproduced by using their specific seed
- Full traceability from master seed → per-universe seed → universe outcomes

### Deterministic Execution

The framework implements strict determinism controls:

```python
# Environment variables set for reproducibility
os.environ["PYTHONHASHSEED"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
```

This hierarchical system provides two levels of control:
- **Ensemble-level**: Re-using the same `SEED` allows an entire ensemble to be reproduced
- **Universe-level**: Selecting an individual universe's seed (found in output CSV files) enables exact reproduction of that universe's evolution without re-running the full ensemble

This capability is especially valuable for debugging, targeted analysis, and validation of noteworthy cases. While the current implementation guarantees reproducibility under consistent library versions and environments, strict cross-platform bit-level determinism may vary depending on the underlying PRNG implementation.

---

## The Simulation Core: A Universe's Lifecycle

The evolution of each universe in the simulation follows a staged lifecycle, modeling the transition from an indeterminate system to one governed by stable, fixed laws. This lifecycle is implemented across multiple phases of the pipeline.

### Initialization

Each run begins with initial Energy (E) and Information (I) values, sampled from distributions defined in the configuration:

- **Energy (E)**: Sampled from a lognormal distribution with parameters `E_LOG_MU` and `E_LOG_SIGMA`, optionally clipped to `[E_TRUNC_LOW, E_TRUNC_HIGH]`. In physical model mode, E represents Omega_Lambda (dark energy density) with reference value 0.7.

- **Information (I)**: Computed using one of 10 I-definition methods (selected via `I_DEFINITION_MODE`). All definitions include E-modulation: `I_final = I_base × √(E_ref/E)`, coupling the information content to dark energy.

- **Complexity Parameter (X)**: The E-I coupling is computed via `compute_coupling(E, I)`, respecting the `X_MODE` setting:
  - `"E_plus_I"`: X = E + I × X_SCALE
  - `"product"`: X = E × I × X_SCALE
  - `"E_times_I_pow"`: X = E × (I ** X_I_POWER) × X_SCALE

These scalar quantities are the fundamental inputs to the TQE model.

### Pre-Collapse Phase

In this phase, the effective physical laws (represented by a state vector X) fluctuate stochastically around values determined by the initial conditions. Conceptually, this is interpreted as a "quantum-like" regime where the laws are not yet fixed.

The implementation models these fluctuations through:
- **Quantum Fluctuation Dynamics**: Random perturbations with optional use of `qutip` for advanced quantum state evolution
- **Superposition Phase**: Optional quantum superposition modeling using density matrices
- **Noise Decay**: Exponential decay of noise amplitude over time, controlled by `NOISE_DECAY_TAU`

The fluctuation phase establishes a normalized, statistically well-behaved foundation from which the more complex, parameter-dependent evolution can unfold.

### Clarification on the "Lawless" Phase

It is crucial to note that the "pre-collapse" phase, while characterized by fluctuating laws, does not start from an absolutely lawless state. The simulation framework is built upon the `qutip` quantum mechanics library, which fundamentally incorporates **the Schrödinger equation** and **Hamiltonian mechanics** as its operational baseline.

Therefore, the TQE model does not simulate the emergence of quantum mechanics itself. Instead, it investigates the process by which higher-level, effective physical laws and constants (like those governing gravitation, electromagnetism, or cosmological expansion) are selected and stabilized from this fundamental, fluctuating quantum substrate. The simulation models the emergence of stable macro-level laws from an underlying quantum reality.

### Law Lock-In

At a critical point, the universe transitions from fluctuating to stable laws, `X_final`. This lock-in event is governed by stability threshold parameters:

**Lock-In Criterion:**
- The relative variation of X(t) must satisfy: `ΔP/P < REL_EPS_LOCKIN` (default: 0.040)
- This condition must hold for `CALM_STEPS_LOCKIN` consecutive epochs (default: 3)
- Lock-in cannot occur before `MIN_LOCKIN_EPOCH` (default: 120 epochs)
- Lock-in requires prior stability: `LOCKIN_REQUIRES_STABLE = True` (default)

When these conditions are met, the system records a `lock_epoch`. This value is a key output of the simulation, representing the moment when physical laws become permanently fixed.

**Lock-In Mechanism:**
- **E-only mode**: Tracks emergent CMB observables (amplitude A, spectral index ns, Hubble parameter H) for lock-in detection
- **E+I mode**: Tracks the coupling variable X (E-I interaction) for TQE-consistent lock-in detection

### Expansion Phase

After lock-in, the universe evolves deterministically according to the fixed laws `X_final`. This phase simulates large-scale expansion and structure formation, producing cosmological observables that are passed to the analysis stage.

The expansion phase includes:
- **Multiplicative Growth**: Size parameter S(t) evolves via multiplicative growth process
- **Growth Rate Modulation**: Growth rate is modulated by the universe's X value
- **Decaying Noise**: Subject to exponentially decaying noise (controlled by `NOISE_DECAY_TAU`)
- **CMB Generation**: Final state is used to generate CMB-like sky maps for comparison with observational data

---

## The 28-Phase Pipeline

The TQE Framework v4.2.0 PRO executes a comprehensive 28-phase analysis pipeline, organized into 12 logical groups. Each phase builds upon previous results, creating a complete workflow from universe generation through advanced statistical analysis.

### GROUP 1: Core Simulation & Data Generation

#### **Phase 1: Monte Carlo Simulation + Bayesian Adaptive Goldilocks Optimization**

**Purpose**: Generate universe ensemble and discover optimal Goldilocks zone using intelligent Bayesian optimization.

**Method**: 
- **Bayesian Adaptive Sampling**: Uses Gaussian Process Regression (GP) with Upper Confidence Bound (UCB) acquisition function
- **Budget Allocation**: 30% of `NUM_UNIVERSES` for Bayesian exploration, 70% for full simulation in discovered zone
- **3-Iteration Strategy**: 
  1. Exploration phase: Uniform sampling across X space
  2. Exploitation phase: UCB-guided sampling focusing on promising regions
  3. Refinement phase: Peak-focused sampling for precise Goldilocks boundaries

**Outputs**:
- `tqe_runs.csv`: Complete universe data (E, I, X, stability, lock_epoch, etc.)
- `Goldilocks_Results/bayesian_goldilocks_optimization.csv`: GP sampling history
- `Goldilocks_Results/goldilocks_gp_uncertainty.png`: GP mean and uncertainty visualization
- Goldilocks parameters: `X_peak`, `X_peak_uncertainty`, `X_c_low`, `X_c_high`

**Key Features**:
- Works efficiently on any sample size (100 to 10,000+ universes)
- Adaptive sampling focuses universes on promising X regions
- Goldilocks computed FROM simulation universes (no separate calibration step)

### GROUP 2: Basic Analysis & Visualization

#### **Phase 2: Stability Curve Analysis**

**Purpose**: Visualize stability probability as a function of Complexity parameter X, identifying the Goldilocks zone.

**Method**: 
- Bins universes by X value (40 bins by default)
- Computes stability rate (stable/total) for each bin
- Fits cubic spline to bin means
- Identifies peak stability location and half-maximum boundaries

**Outputs**:
- `PNG_Visualizations/stability_curve_EI_Pipeline_v4.2.0_Pro.png`: Stability curve with Goldilocks zone overlay
- Peak X location, Goldilocks window boundaries

#### **Phase 3: E-I Parameter Space Visualization**

**Purpose**: Map universe outcomes in the Energy-Information parameter space.

**Method**: Scatter plot of all universes colored by stability outcome (stable=red, unstable=blue).

**Outputs**:
- `PNG_Visualizations/scatter_EI_EI_Pipeline_v4.2.0_Pro.png`: E-I parameter space map

#### **Phase 4: Fluctuation Dynamics**

**Purpose**: Visualize the quantum fluctuation → superposition → collapse → expansion sequence.

**Method**: Generates time-series panels showing:
- Quantum fluctuation statistics (expectation value, variance)
- Superposition phase (entropy, purity evolution)
- Collapse event (X parameter fixation)
- Expansion dynamics (amplitude A, orientation I)

**Outputs**:
- `PNG_Visualizations/fl_fluctuation_EI_Pipeline_v4.2.0_Pro.png`: Quantum fluctuation statistics
- `PNG_Visualizations/fl_superposition_EI_Pipeline_v4.2.0_Pro.png`: Entropy and purity evolution
- `PNG_Visualizations/fl_collapse_EI_Pipeline_v4.2.0_Pro.png`: Collapse event visualization
- `PNG_Visualizations/fl_expansion_EI_Pipeline_v4.2.0_Pro.png`: Expansion dynamics

### GROUP 3: Stability & Lock-in Analysis

#### **Phase 5: Stability-by-I Analysis**

**Purpose**: Analyze correlation between Information parameter I and stability outcomes.

**Method**: Computes stability statistics for universes with I ≈ 0 (exact zero + epsilon sweep) vs. I > 0.

**Outputs**:
- `PNG_Visualizations/stability_by_I_EI_Pipeline_v4.2.0_Pro.png`: Stability comparison by I value
- Statistical summary of I's role in stability

#### **Phase 6: Lock-in Histogram**

**Purpose**: Visualize the distribution of lock-in epochs across the universe ensemble.

**Method**: Histogram of `lock_epoch` values for universes that achieved lock-in.

**Outputs**:
- `PNG_Visualizations/lockin_histogram_EI_Pipeline_v4.2.0_Pro.png`: Lock-in epoch distribution

#### **Phase 7: Stability Distribution**

**Purpose**: Categorize and visualize universe fates (lock-in, stable, unstable).

**Method**: Bar chart showing counts and percentages for each outcome category.

**Outputs**:
- `PNG_Visualizations/stability_distribution_three_EI_Pipeline_v4.2.0_Pro.png`: Three-category distribution

#### **Phase 8: Average Lock-in Curve**

**Purpose**: Visualize the average evolution leading to lock-in.

**Method**: Time-series of average X(t) for universes that locked in, showing convergence to stable value.

**Outputs**:
- `PNG_Visualizations/avg_lockin_curve_EI_Pipeline_v4.2.0_Pro.png`: Average lock-in trajectory

### GROUP 4: Machine Learning & Emergent Laws

#### **Phase 9: Feature Importance Analysis**

**Purpose**: Use Random Forest models to identify which parameters (E, I, X) most strongly predict outcomes.

**Method**: 
- Trains Random Forest classifiers (stability prediction) and regressors (lock-in epoch prediction)
- Computes feature importance scores
- Optional: SHAP and LIME explanations (if packages installed)

**Outputs**:
- `Aggregate/feature_importance_EI_Pipeline_v4.2.0_Pro.csv`: Feature importance scores
- `PNG_Visualizations/feature_importance_EI_Pipeline_v4.2.0_Pro.png`: Feature importance visualization

#### **Phase 10: Emergent Laws Detection**

**Purpose**: Detect power-law scaling relationships and phase transitions in the simulation data.

**Method**: 
- Searches for power-law relationships: `Y ~ X^α`
- Detects phase transitions (sudden changes in parameter relationships)
- Computes correlation strengths between E, I, X

**Outputs**:
- `Aggregate/emergent_law_summary_EI_Pipeline_v4.2.0_Pro.csv`: Detected laws and their parameters
- `PNG_Visualizations/emergent_laws_EI_Pipeline_v4.2.0_Pro.png`: Law visualization

#### **Phase 11: Statistical Finetuning Detector**

**Purpose**: Analyze fine-tuning relationships, particularly the E-I gap (`|E - I|`) effect on lock-in probability.

**Method**: 
- Computes lock-in probability as a function of `|E - I|`
- Identifies optimal E-I imbalance for law finalization
- Generates finetuning curves and bar charts

**Outputs**:
- `PNG_Visualizations/finetune_gap_curve_EI_Pipeline_v4.2.0_Pro.png`: Lock-in probability vs. E-I gap
- `PNG_Visualizations/finetune_gap_adaptive_EI_Pipeline_v4.2.0_Pro.png`: Adaptive finetuning analysis
- `PNG_Visualizations/lockin_by_eqI_bar_EI_Pipeline_v4.2.0_Pro.png`: Bar chart comparison

### GROUP 5: CMB Generation & Validation

#### **Phase 12: Best Universe Plots & CMB Map Generation**

**Purpose**: Select top-performing universes and generate their CMB maps.

**Method**: 
- Ranks universes by composite score (lock-in speed, stability quality, etc.)
- Selects top 3 from each category: lock-in, stable-only, unstable
- Generates entropy evolution plots for best universes
- Generates CMB maps using CAMB (if available) or fallback method
- Saves CMB maps as FITS files

**Outputs**:
- `Categorized_Results/lock_in/1_FIGURES/best_uni_rank0X_uidXXXXX_EI_Pipeline_v4.2.0_Pro.png`: Entropy evolution plots
- `Categorized_Results/lock_in/2_DATA_FILES/entropy_timeseries_uidXXXXX.csv`: Entropy time-series data
- `Categorized_Results/lock_in/3_CMB_MAPS/cmb_map_uidXXXXX.fits`: CMB map FITS files
- `Categorized_Results/lock_in/3_CMB_MAPS/cmb_map_uidXXXXX.png`: CMB map preview images

#### **Phase 13: Complete CMB Map Coverage**

**Purpose**: Ensure all lock-in universes have CMB maps generated.

**Method**: Iterates through all universes with `lock_epoch >= 0`, generating missing CMB maps.

**Outputs**: Additional CMB FITS files and previews for all lock-in universes

#### **Phase 14: Entropy Volatility Analysis**

**Purpose**: Analyze entropy fluctuations and volatility patterns.

**Method**: Computes entropy volatility metrics and generates distribution plots.

**Outputs**:
- `PNG_Visualizations/entropy_volatility_EI_Pipeline_v4.2.0_Pro.png`: Entropy volatility analysis

#### **Phase 15: Planck Observational Comparison**

**Purpose**: Compare simulated universes to Planck 2018 observational data (ONLY phase using Planck data).

**Method**: 
- Loads Planck 2018 CMB power spectrum data
- Computes chi-squared fit: `χ² = Σ (C_ℓ_sim - C_ℓ_Planck)² / σ_ℓ²`
- Generates comparison plots

**Outputs**:
- `PNG_Visualizations/planck_validation_EI_Pipeline_v4.2.0_Pro.png`: Planck comparison plot
- `Aggregate/planck_chi2_EI_Pipeline_v4.2.0_Pro.csv`: Chi-squared values per universe
- Planck chi-squared value (used in Phase 28 for Bayesian analysis)

#### **Phase 16: CMB Anomaly Detection**

**Purpose**: Detect emergent CMB anomalies on simulated maps (fully emergent, no forced Planck matching).

**Method**: 
- **Cold Spot Detection**: Multi-scale Gaussian smoothing, z-score thresholding, minimum separation filtering
- **Axis of Evil Detection**: Quadrupole-octupole alignment analysis using spherical harmonics

**Outputs**:
- `Aggregate/cmb_coldspots_summary_{i_def}.csv`: Cold spot detections with positions, depths, z-scores
- `Aggregate/cmb_aoe_summary_{i_def}.csv`: Axis of Evil detections with alignment angles
- `Categorized_Results/lock_in/3_CMB_MAPS/coldspots_overlay_uidXXXXX_EI_Pipeline_v4.2.0_Pro.png`: Cold spot overlay (max 3)
- `Categorized_Results/lock_in/3_CMB_MAPS/aoe_overlay_uidXXXXX_EI_Pipeline_v4.2.0_Pro.png`: AOE overlay (max 3)

### GROUP 6: E+I Interaction Analysis

#### **Phase 17: E+I Importance Comparison**

**Purpose**: Compare the relative importance of Energy vs. Information parameters.

**Method**: Statistical analysis and visualization of E vs. I contributions to outcomes.

**Outputs**:
- `PNG_Visualizations/ei_importance_comparison_EI_Pipeline_v4.2.0_Pro.png`: E vs. I importance analysis

#### **Phase 18: Multi-Mode Goldilocks Comparison**

**Purpose**: Compare Goldilocks zones across all I-definitions (if batch mode).

**Method**: Overlays Goldilocks curves for different I-definitions on a single plot.

**Outputs**:
- `PNG_Visualizations/multi_mode_goldilocks_EI_Pipeline_v4.2.0_Pro.png`: Multi-definition Goldilocks comparison

### GROUP 7: Advanced CMB Analysis

#### **Phase 19: CMB Statistical Analysis**

**Purpose**: Aggregate CMB statistics from all simulated maps (does NOT use Planck data).

**Method**: 
- Gaussianity tests (skewness, kurtosis, chi-squared)
- Isotropy tests (directional variance analysis)
- Power spectrum analysis (angular power spectrum C_ℓ)

**Outputs**:
- `PNG_Visualizations/cmb_gaussianity_EI_Pipeline_v4.2.0_Pro.png`: Gaussianity analysis
- `PNG_Visualizations/cmb_isotropy_EI_Pipeline_v4.2.0_Pro.png`: Isotropy analysis
- `PNG_Visualizations/cmb_power_spectrum_EI_Pipeline_v4.2.0_Pro.png`: Power spectrum plot
- `Aggregate/cmb_statistics_EI_Pipeline_v4.2.0_Pro.csv`: Statistical summary

#### **Phase 20: Comprehensive Correlation Analysis**

**Purpose**: Generate comprehensive correlation matrices between all parameters.

**Method**: Computes Pearson/Spearman correlations between E, I, X, stability, lock_epoch, and derived metrics.

**Outputs**:
- `PNG_Visualizations/correlation_heatmap_EI_Pipeline_v4.2.0_Pro.png`: Correlation matrix heatmap
- `Aggregate/correlation_matrix_EI_Pipeline_v4.2.0_Pro.csv`: Full correlation matrix

### GROUP 8: Advanced Statistical Analysis

#### **Phase 21: Advanced Statistical Analysis**

**Purpose**: Compute advanced statistical metrics and distributions.

**Method**: 
- Parameter sensitivity analysis
- Universe classification analysis
- Performance metrics analysis
- Statistical distribution analysis

**Outputs**:
- `PNG_Visualizations/statistical_summary_EI_Pipeline_v4.2.0_Pro.png`: Statistical summary plots
- `Aggregate/advanced_statistics_EI_Pipeline_v4.2.0_Pro.csv`: Advanced metrics

#### **Phase 22: CMB Anomaly Visualization**

**Purpose**: Create aggregate visualization of all detected anomalies.

**Method**: 
- Loads the run-specific cold spot & AOE catalogues (variant-aware); falls back to synthetic draws only if no detections were recorded
- Automatically installs `healpy` on demand and regenerates Mollweide density maps with per-detector overlays
- Produces both histogram/heatmap diagnostics and sky-projected aggregate density plots

**Outputs**:
- `PNG_Visualizations/coldspot_position_heatmap_EI_Pipeline_v4.2.0_Pro.png`: Cold spot position heatmap (variant-tagged)
- `PNG_Visualizations/coldspot_depth_histogram_EI_Pipeline_v4.2.0_Pro.png`: Cold spot depth histogram
- `PNG_Visualizations/aggregate_coldspot_density_map_EI_Pipeline_v4.2.0_Pro.png`: Mollweide aggregate cold spot density map with detections highlighted
- `PNG_Visualizations/aoe_alignment_histogram_EI_Pipeline_v4.2.0_Pro.png`: Axis-of-Evil alignment histogram
- `PNG_Visualizations/aggregate_aoe_density_map_EI_Pipeline_v4.2.0_Pro.png`: Multipole-specific aggregate AOE density maps (ℓ = 2…5)

### GROUP 9: Enhanced Physics Analysis

#### **Phase 23: Enhanced Physics Analysis**

**Purpose**: Analyze Friedmann evolution, quantum fields, and entanglement networks.

**Method**: 
- Friedmann equation integration (universe age, H0, Omegas)
- Quantum field fluctuation analysis
- Cosmic entanglement network computation
- Physical anomaly catalogue generation

**Outputs**:
- `PNG_Visualizations/friedmann_evolution_EI_Pipeline_v4.2.0_Pro.png`: Friedmann evolution plots
- `PNG_Visualizations/quantum_field_analysis_EI_Pipeline_v4.2.0_Pro.png`: Quantum field analysis
- `Aggregate/enhanced_physics_EI_Pipeline_v4.2.0_Pro.csv`: Physics data
- `Aggregate/physics_analysis.json`: Physics summary JSON

#### **Phase 24: Comprehensive Data Extraction**

**Purpose**: Extract comprehensive data from all universes for external analysis.

**Method**: Aggregates all computed metrics into master CSV files.

**Outputs**:
- `Aggregate/comprehensive_data_EI_Pipeline_v4.2.0_Pro.csv`: Complete universe data
- Additional specialized CSV files for different analysis categories

### GROUP 10: Advanced Anomaly & Law Detection

#### **Phase 25: Advanced Anomaly Detection**

**Purpose**: Detect anomalies across multiple physics domains (beyond CMB).

**Method**: Vectorized anomaly detection (100× faster than iterrows) using z-score thresholding:

- **Quantum Field Anomalies**: Unusual quantum fluctuation amplitudes
- **Entropy Anomalies**: Extreme entropy volatility
- **Topological Anomalies**: Anomalous topological defect densities
- **Energy Conservation Anomalies**: Energy conservation violations
- **Information Theory Anomalies**: Extreme information entropy values
- **CMB Statistical Anomalies**: Power spectrum outliers

**Outputs**:
- `Aggregate/advanced_anomaly_detection_results_EI_Pipeline_v4.2.0_Pro.csv`: Anomaly catalogue
- `PNG_Visualizations/advanced_anomaly_detection_analysis_EI_Pipeline_v4.2.0_Pro.png`: Anomaly visualization

#### **Phase 26: Advanced Law Detection**

**Purpose**: Automatically detect physical laws emerging from the simulation data.

**Method**: Statistical analysis to identify law-like patterns:

- **Conservation Laws**: Energy, momentum, charge conservation
- **Symmetry Laws**: E-I symmetry breaking patterns
- **Scaling Laws**: Power-law relationships (X-stability scaling)
- **Emergent Laws**: Lock-in emergence patterns
- **Quantum Laws**: Quantum uncertainty principle manifestations
- **Thermodynamic Laws**: Entropy increase laws
- **Statistical Laws**: Boltzmann distribution patterns
- **Field Theory Laws**: Field correlation laws
- **Geometric Laws**: Geometric scaling relationships
- **Information Laws**: Information conservation patterns

**Outputs**:
- `Aggregate/advanced_law_detection_results_EI_Pipeline_v4.2.0_Pro.csv`: Law catalogue
- `PNG_Visualizations/advanced_law_detection_analysis_EI_Pipeline_v4.2.0_Pro.png`: Law visualization

### GROUP 11: Comprehensive Visualization

#### **Phase 27: Comprehensive Visualization Extraction**

**Purpose**: Generate multi-dimensional visualizations of parameter space and universe distributions.

**Method**: 
- Parameter space heatmaps
- Multi-dimensional analysis plots
- Statistical distribution analysis
- Correlation network visualization
- Phase space dynamics
- Information theory analysis
- Quantum field analysis
- Cosmological evolution analysis

**Outputs**: 10+ additional visualization PNG files covering all analysis dimensions

### GROUP 12: Final Summary & Bayesian

#### **Phase 28: Final Summary, Complexity Synthesis & Bayesian Integration**

**Purpose**: Consolidate all metrics, evaluate complexity and life-compatibility, and perform Bayesian model selection.

**Method**: 
- Aggregates all metrics into `summary_full.json`
- Computes complexity and life-compatibility component scores (lock-in quality, Goldilocks precision, information richness, Planck fit, stability quality, Goldilocks robustness) with threshold checks
- Generates top-universe complexity ranking and component visualization exports
- Computes Bayesian Information Criterion (BIC) and Akaike Information Criterion (AIC)
- Optional: Runs nested sampling for Bayesian Evidence computation
- Generates corner plots for parameter posterior distributions
- Exports I-Definitions comparison CSV and PNG (if enabled)

**Outputs**:
- `summary_full.json`: Complete pipeline summary with all metrics
- `Aggregate/complexity_metrics_summary.csv`: Run-level complexity and life-compatibility record
- `Aggregate/complexity_universe_ranking.csv`: Top universes by complexity (if enabled)
- `Aggregate/life_compatibility_summary.json`: Complexity & life-compatibility component breakdown
- `PNG_Visualizations/complexity_life_components.png`: Component bar chart
- `PNG_Visualizations/complexity_top_universes.png`: Top-universe ranking bar plot (if enabled)
- `Aggregate/bayesian_metrics_{i_def}.csv`: Bayesian model selection metrics
- `PNG_Visualizations/bayesian_comparison_{i_def}.png`: BIC/AIC comparison
- `PNG_Visualizations/corner_plot_{i_def}.png`: Parameter posterior distributions (if nested sampling enabled)
- `Aggregate/I_Definitions_Comparison.csv`: All 10 I-definitions comparison (if enabled)
- `PNG_Visualizations/I_Definitions_Comparison.png`: I-definitions visualization

---

## 10 I-Parameter Definitions

The TQE Framework supports 10 distinct mathematical definitions for the Information parameter (I), each capturing different aspects of information-theoretic content. All definitions are normalized to the range [0, 1] and include E-modulation (dark energy coupling): `I_final = I_base × √(E_ref/E)`.

### Classical Information Theory Definitions

#### **1. KL-Divergence (`kl_divergence`)**

**Definition**: Kullback-Leibler divergence between two random quantum states.

**Formula**: 
```
D_KL(p1||p2) = Σ p1(i) log[p1(i)/p2(i)]
I_kl = D_KL / (1 + D_KL)  [normalized to [0,1]]
```

**Physical Interpretation**: Measures quantum state distinguishability - how different two probability distributions are. Higher values indicate greater information content in distinguishing states.

**E-Modulation**: `I_kl_final = I_kl × √(E_ref/E)`

#### **2. Shannon Entropy (`shannon`)**

**Definition**: Shannon entropy of a random quantum state.

**Formula**:
```
H = -Σ p(i) log[p(i)]
I_shannon = H / log(dim)  [normalized to [0,1]]
```

**Physical Interpretation**: Measures information content and uncertainty quantification. Higher entropy indicates more mixed/uncertain quantum states, representing greater information capacity.

**E-Modulation**: `I_shannon_final = I_shannon × √(E_ref/E)`

#### **3. Rényi Entropy (`renyi`)**

**Definition**: Generalized entropy (collision entropy, α=2).

**Formula**:
```
H_α = (1/(1-α)) log[Σ p(i)^α]
I_renyi = H_2 / log(dim)  [normalized to [0,1]]
```

**Physical Interpretation**: Generalized entropy measure emphasizing different aspects of the probability distribution. α=2 (collision entropy) emphasizes high-probability events.

**E-Modulation**: `I_renyi_final = I_renyi × √(E_ref/E)`

#### **4. Mutual Information (`mutual_info`)**

**Definition**: Correlation between different aspects of the quantum state.

**Formula**: 
```
I(X;Y) = H(X) + H(Y) - H(X,Y)
I_mutual = I(X;Y) / max[H(X), H(Y)]  [normalized to [0,1]]
```

**Physical Interpretation**: Measures how much information one part of the system reveals about another. Higher values indicate stronger correlations and information sharing.

**E-Modulation**: `I_mutual_final = I_mutual × √(E_ref/E)`

### Quantum Information Definitions

#### **5. Entanglement Entropy (`entanglement`)**

**Definition**: Von Neumann entropy of a subsystem (normalized).

**Formula**:
```
S_vN = -Tr[ρ_A log(ρ_A)]
I_ent = S_vN / log(d)  [normalized to [0,1]]
where ρ_A is the reduced density matrix of subsystem A
```

**Physical Interpretation**: Measures quantum entanglement between subsystems. Higher values indicate stronger entanglement, representing quantum information content.

**E-Modulation**: `I_ent_final = I_ent × √(E_ref/E)`

#### **6. Fisher Information (`fisher`)**

**Definition**: Quantum Fisher Information (normalized).

**Formula**:
```
F_Q = 4 × Var[H] = 4 × (⟨H²⟩ - ⟨H⟩²)
I_fisher = F_Q / 4.0  [normalized to [0,1]]
```

**Physical Interpretation**: Measures quantum metrology precision - how well a quantum state can estimate a parameter. Higher values indicate greater sensitivity and information content.

**E-Modulation**: `I_fisher_final = I_fisher × √(E_ref/E)`

### Composite Fusion Definitions

#### **7. Composite Product (`composite`)**

**Definition**: Multiplicative fusion of KL-divergence and Shannon entropy (strict filtering).

**Formula**:
```
I_composite = I_kl × I_shannon
```

**Physical Interpretation**: Strict filtering approach - requires both high distinguishability (KL) and high information content (Shannon). More conservative than individual measures.

**E-Modulation**: Applied to both components before multiplication.

#### **8. KL-Shannon Refined (`kl_shannon`)**

**Definition**: Harmonic mean of KL-divergence and Shannon entropy (balanced, outlier-robust).

**Formula**:
```
I_kl_shannon = 2 × (I_kl × I_shannon) / (I_kl + I_shannon)  [harmonic mean]
OR weighted average: w_kl × I_kl + w_sh × I_shannon
```

**Physical Interpretation**: Balanced fusion that is robust to outliers. Harmonic mean emphasizes lower values, providing conservative estimates.

**E-Modulation**: Applied to both components before fusion.

#### **9. Fisher-KL Fusion (`fisher_kl_fusion`)**

**Definition**: Fusion of quantum metrology (Fisher) and distinguishability (KL).

**Formula**:
```
I_fisher_kl = (I_fisher + I_kl) / 2  [average fusion]
```

**Physical Interpretation**: Combines quantum metrology precision with state distinguishability, capturing both sensitivity and information content.

**E-Modulation**: Applied to both components before fusion.

### Symmetric Information Measures

#### **10. Jensen-Shannon Divergence (`jensen_shannon`)** ⭐ **DEFAULT & VALIDATED**

**Definition**: Symmetric, bounded version of KL-divergence (validated with Planck 2018 CMB data).

**Formula**:
```
JS(p||q) = 0.5 × [D_KL(p||m) + D_KL(q||m)]
where m = (p + q) / 2 is the average distribution
I_js = JS / (1 + JS)  [normalized to [0,1]]
```

**Physical Interpretation**: 
- **Symmetric**: JS(p||q) = JS(q||p), unlike standard KL-divergence
- **Bounded**: Always in [0, 1] range (after normalization)
- **Robust**: More stable than KL-divergence for comparing distributions
- **Validated**: Used in real universe measurements (Planck 2018 CMB data) for optimal I_kl determination
- **Experimental Result**: Real universe I_js ≈ 0.250, matching TQE predictions for stable universes

**E-Modulation**: `I_js_final = I_js × √(E_ref/E)`

**Why Default?**: Jensen-Shannon divergence has been validated against real observational data (Planck 2018 CMB power spectrum), showing remarkable agreement with TQE predictions. The measured value (I_js ≈ 0.25) falls within the range [0.2, 0.4] that produces stable, life-compatible universes in TQE simulations.

### I-Definition Selection

The active I-definition is selected via the `I_DEFINITION_MODE` parameter in `MASTER_CTRL`:

```python
MASTER_CTRL = {
    "I_DEFINITION_MODE": "jensen_shannon",  # or any of the 10 options
    # ...
}
```

**Comparison Mode**: If `COMPUTE_ALL_I_DEFINITIONS = True`, the pipeline exports a CSV file (`I_Definitions_Comparison.csv`) and visualization (`I_Definitions_Comparison.png`) showing all 10 definitions computed for the same E values, enabling direct comparison.

---

## Anomaly Detection

The TQE Framework includes comprehensive anomaly detection capabilities, spanning both CMB-specific anomalies and advanced physics-domain anomalies. All anomalies are **fully emergent** from the simulation—no forced matching to Planck data, allowing genuine TQE validation.

### CMB Anomalies (Phase 16)

#### **Cold Spot Detection**

**Purpose**: Detect unusually cold regions in simulated CMB maps, analogous to the Planck Cold Spot anomaly.

**Method**: 
- **Multi-Scale Gaussian Smoothing**: Applies Gaussian filters at multiple angular scales (default: 180 and 360 arcmin)
- **Z-Score Thresholding**: Identifies regions with z-scores below threshold (default: -0.5)
- **Minimum Separation Filtering**: Ensures detected spots are separated by minimum angular distance (default: 30 arcmin)
- **Top-K Selection**: Selects top K coldest spots per universe (default: 5)

**Detection Algorithm**:
1. Smooth CMB map at multiple scales using HEALPix Gaussian beam convolution
2. Compute z-scores: `z = (T - μ) / σ` for each pixel
3. Identify connected regions below threshold
4. Filter by minimum separation to avoid duplicates
5. Rank by depth (most negative z-score)

**Outputs**:
- `Aggregate/cmb_coldspots_summary_{i_def}.csv`: Detected cold spots with:
  - Universe ID
  - Celestial coordinates (RA, Dec, or HEALPix pixel)
  - Temperature depth (µK)
  - Z-score
  - Angular size (FWHM)
- `Categorized_Results/lock_in/3_CMB_MAPS/coldspots_overlay_uidXXXXX_EI_Pipeline_v4.2.0_Pro.png`: Overlay visualization (max 3 universes)
  - Lime X markers (✗) indicate detected cold spot centers
  - Colored circles show spot extent
- `PNG_Visualizations/coldspots_pos_heatmap_EI_Pipeline_v4.2.0_Pro.png`: Positional distribution heatmap
- `PNG_Visualizations/coldspots_z_hist_EI_Pipeline_v4.2.0_Pro.png`: Depth distribution histogram with Planck reference line (z ≈ -70)

**Key Finding**: The simulation successfully generates rare, large-scale cold spots. In the analyzed E+I cohort, cold spots were detected in only 3 out of 10,000 universes—the same "best" universes that also exhibited Axis of Evil anomalies, suggesting a common physical origin linked to rapid law stabilization.

#### **Axis of Evil (AoE) Detection**

**Purpose**: Detect anomalous alignments between low-multipole moments (quadrupole ℓ=2 and octupole ℓ=3), analogous to the observed "Axis of Evil" in Planck CMB data.

**Method**:
- **Spherical Harmonic Decomposition**: Decomposes CMB map into spherical harmonics using HEALPix `anafast()`
- **Multipole Extraction**: Extracts quadrupole (ℓ=2) and octupole (ℓ=3) components
- **Axis Calculation**: Computes principal axes for each multipole using eigenvector analysis
- **Alignment Analysis**: Computes angle between quadrupole and octupole axes
- **Statistical Significance**: Monte Carlo realizations (default: 20) to assess significance

**Detection Algorithm**:
1. Compute spherical harmonic coefficients `a_ℓm` up to `ℓ_max` (default: 3)
2. Extract quadrupole (ℓ=2) and octupole (ℓ=3) components
3. Compute principal axes via eigenvector decomposition of multipole moment tensors
4. Calculate alignment angle: `θ = arccos(|n̂₂ · n̂₃|)`
5. Compare to random distribution via Monte Carlo

**Outputs**:
- `Aggregate/cmb_aoe_summary_{i_def}.csv`: Detected AOE alignments with:
  - Universe ID
  - Alignment angle (degrees)
  - P-value (statistical significance)
  - Quadrupole axis direction
  - Octupole axis direction
- `Categorized_Results/lock_in/3_CMB_MAPS/aoe_overlay_uidXXXXX_EI_Pipeline_v4.2.0_Pro.png`: Overlay visualization (max 3 universes)
  - Colored star markers (★) for quadrupole (ℓ=2,3,4,5) axes
  - Lines connecting aligned axes
- `PNG_Visualizations/aoe_angle_hist_EI_Pipeline_v4.2.0_Pro.png`: Alignment angle distribution histogram with reference line (≈20°)

**Key Finding**: The simulation generates AOE anomalies in the same rare subset of universes as cold spots. Notably, one simulated alignment (≈20°) quantitatively matches the observed value from our universe's CMB, providing significant encouragement for the model's validity.

### Advanced Anomalies (Phase 25)

The pipeline includes vectorized anomaly detection (100× faster than iterrows) across multiple physics domains:

#### **1. Quantum Field Anomalies**

**Detection**: Identifies universes with unusual quantum fluctuation amplitudes.

**Method**: 
- Computes mean and standard deviation of `quantum_fluctuation` values
- Flags universes with z-score > threshold (default: 3.0)
- Classifies as "high" significance if z-score > 5.0, else "medium"

**Output**: Anomaly catalogue with universe ID, anomaly value, expected value, deviation sigma, and significance level.

#### **2. Entropy Anomalies**

**Detection**: Identifies universes with extreme entropy volatility.

**Method**: 
- Analyzes `entropy_volatility` column
- Z-score thresholding (default: 3.0)
- Flags unusually volatile or stable entropy evolution

**Output**: Entropy anomaly catalogue.

#### **3. Topological Anomalies**

**Detection**: Identifies anomalous topological defect densities.

**Method**: 
- Analyzes `topological_defect_density` values
- Z-score thresholding for defect density outliers
- Flags universes with unusual topological structure

**Output**: Topological anomaly catalogue.

#### **4. Energy Conservation Anomalies**

**Detection**: Identifies potential energy conservation violations.

**Method**: 
- Analyzes Energy (E) parameter distribution
- Flags universes with extreme E values (z-score > threshold)
- May indicate non-physical configurations

**Output**: Energy anomaly catalogue.

#### **5. Information Theory Anomalies**

**Detection**: Identifies extreme information entropy values.

**Method**: 
- Analyzes Information (I) parameter distribution
- Flags universes with unusually high or low I values
- May indicate information-theoretic inconsistencies

**Output**: Information anomaly catalogue.

#### **6. CMB Statistical Anomalies**

**Detection**: Identifies power spectrum outliers.

**Method**: 
- Analyzes `cmb_power_spectrum` values
- Z-score thresholding for power spectrum deviations
- Flags universes with unusual CMB statistics

**Output**: CMB statistical anomaly catalogue.

**Configuration**: All advanced anomaly detection can be toggled via `MASTER_CTRL`:
- `ENABLE_QUANTUM_ANOMALY_DETECTION = True`
- `ENABLE_ENTROPY_ANOMALY_DETECTION = True`
- `ENABLE_TOPOLOGICAL_ANOMALY_DETECTION = True`
- `ENABLE_ENERGY_ANOMALY_DETECTION = True`
- `ENABLE_INFORMATION_ANOMALY_DETECTION = True`
- `ENABLE_CMB_ANOMALY_DETECTION = True`
- `ANOMALY_DETECTION_THRESHOLD = 3.0` (z-score threshold)

---

## Law Detection

The TQE Framework automatically detects physical laws emerging from the simulation data (Phase 26). These are not imposed but discovered through statistical analysis of the universe ensemble, providing insights into the mechanisms driving cosmic evolution.

### Conservation Laws

#### **Energy Conservation Law**

**Detection**: Analyzes Energy (E) parameter stability across the ensemble.

**Method**: 
- Computes coefficient of variation: `CV = σ(E) / μ(E)`
- Law strength: `strength = 1.0 / (1.0 + CV)`
- Quality classification:
  - Excellent: CV < 0.1
  - Good: CV < 0.2
  - Fair: otherwise

**Physical Interpretation**: Measures how well energy is conserved across universes. Higher strength indicates better energy conservation, suggesting a fundamental conservation principle.

### Symmetry Laws

#### **E-I Symmetry Breaking Law**

**Detection**: Analyzes correlation between Energy and Information parameters.

**Method**: 
- Computes Pearson correlation: `corr(E, I)`
- Symmetry breaking strength: `strength = |corr(E, I)|`
- Quality classification:
  - Excellent: |corr| > 0.8
  - Good: |corr| > 0.6
  - Fair: otherwise

**Physical Interpretation**: Measures E-I symmetry breaking. High correlation indicates strong coupling, while low correlation suggests symmetry breaking—a key mechanism in TQE for law emergence.

### Scaling Laws

#### **X-Stability Scaling Law**

**Detection**: Analyzes power-law relationship between Complexity (X) and stability.

**Method**: 
- Computes Spearman correlation: `ρ(X, stable)`
- Law strength: `strength = |ρ|`
- Statistical significance: `significance = 1.0 - p_value`

**Physical Interpretation**: Identifies scaling relationships between complexity and stability. Strong correlations indicate power-law behavior, characteristic of critical phenomena and phase transitions.

### Emergent Laws

#### **Lock-In Emergence Law**

**Detection**: Analyzes the rate of law lock-in across the ensemble.

**Method**: 
- Computes lock-in rate: `rate = (lock_epoch >= 0).mean()`
- Law strength: `strength = rate`
- Quality classification:
  - Excellent: rate > 0.7
  - Good: rate > 0.5
  - Fair: otherwise

**Physical Interpretation**: Measures the emergence of stable physical laws. High lock-in rates indicate that the TQE mechanism successfully produces law-governed universes.

### Quantum Laws

#### **Quantum Uncertainty Principle**

**Detection**: Analyzes E-I uncertainty product.

**Method**: 
- Computes uncertainty product: `product = (E × I).mean()`
- Law strength: `strength = product`
- Quality classification based on product magnitude

**Physical Interpretation**: Tests quantum uncertainty-like relationships between E and I. High products may indicate fundamental quantum limits on simultaneous knowledge of energy and information.

### Thermodynamic Laws

#### **Entropy Increase Law**

**Detection**: Analyzes entropy evolution over time.

**Method**: 
- Computes entropy increase: `ΔS = entropy.diff().mean()`
- Law strength: `strength = max(0, ΔS)`
- Quality classification:
  - Excellent: ΔS > 0.01
  - Good: ΔS > 0.005
  - Fair: otherwise

**Physical Interpretation**: Tests the second law of thermodynamics—entropy should increase over time. Positive values confirm thermodynamic consistency.

### Statistical Laws

#### **Boltzmann Distribution Law**

**Detection**: Analyzes Energy distribution for Boltzmann-like behavior.

**Method**: 
- Compares E distribution to expected Boltzmann form
- Computes quality metric based on distribution shape
- Law strength: `strength = 1.0 / (1.0 + |σ - μ×0.1|)`

**Physical Interpretation**: Tests whether energy follows statistical mechanics distributions, suggesting thermal equilibrium or maximum entropy principles.

### Field Theory Laws

#### **Field Correlation Law**

**Detection**: Analyzes E-I field correlation.

**Method**: 
- Computes field correlation: `corr = |corr(E, I)|`
- Law strength: `strength = corr`
- Quality classification based on correlation strength

**Physical Interpretation**: Measures field-theoretic coupling between energy and information. High correlations suggest unified field behavior.

### Geometric Laws

#### **Geometric Scaling Law**

**Detection**: Analyzes geometric relationships in Complexity parameter.

**Method**: 
- Computes geometric mean: `X_geo = mean(√X)`
- Law strength: `strength = X_geo`
- Quality classification based on geometric mean magnitude

**Physical Interpretation**: Tests geometric scaling relationships, which may indicate fractal or self-similar structures in parameter space.

### Information Laws

#### **Information Conservation Law**

**Detection**: Analyzes Information (I) parameter stability.

**Method**: 
- Computes coefficient of variation: `CV = σ(I) / μ(I)`
- Law strength: `strength = 1.0 / (1.0 + CV)`
- Quality classification similar to energy conservation

**Physical Interpretation**: Tests information conservation principles. High strength suggests information is a conserved quantity, consistent with information-theoretic foundations of physics.

**Configuration**: All law detection can be toggled via `MASTER_CTRL`:
- `ENABLE_CONSERVATION_LAW_DETECTION = True`
- `ENABLE_SYMMETRY_LAW_DETECTION = True`
- `ENABLE_SCALING_LAW_DETECTION = True`
- `ENABLE_EMERGENT_LAW_DETECTION = True`
- `ENABLE_QUANTUM_LAW_DETECTION = True`
- `ENABLE_THERMODYNAMIC_LAW_DETECTION = True`
- `ENABLE_STATISTICAL_LAW_DETECTION = True`
- `ENABLE_FIELD_LAW_DETECTION = True`
- `ENABLE_GEOMETRIC_LAW_DETECTION = True`
- `ENABLE_INFORMATION_LAW_DETECTION = True`

**Outputs**:
- `Aggregate/advanced_law_detection_results_EI_Pipeline_v4.2.0_Pro.csv`: Complete law catalogue with:
  - Law type
  - Law strength (0-1 scale)
  - Law quality (excellent/good/fair)
  - Statistical significance
  - Universe count
- `PNG_Visualizations/advanced_law_detection_analysis_EI_Pipeline_v4.2.0_Pro.png`: Comprehensive law visualization (4-panel plot)

---

## Bayesian Analysis (PRO Features)

The TQE Framework v4.2.0 PRO includes sophisticated Bayesian analysis capabilities for rigorous model comparison and parameter estimation. These features enable quantitative assessment of different I-definitions and provide uncertainty quantification.

### Bayesian Adaptive Goldilocks (Phase 1)

**Purpose**: Intelligently discover optimal Goldilocks zones using Bayesian optimization.

**Method**: **Gaussian Process Regression (GP) with Upper Confidence Bound (UCB) Acquisition**

**Algorithm**:
1. **GP Model**: Fits Gaussian Process to stability observations `P(stable | X)`
   - Kernel: Radial Basis Function (RBF) + White noise
   - Hyperparameters: Length scale, signal variance, noise level
2. **UCB Acquisition**: Selects next X to sample: `X_next = argmax[μ(X) + κ×σ(X)]`
   - `μ(X)`: GP mean (exploitation)
   - `σ(X)`: GP uncertainty (exploration)
   - `κ`: Exploration-exploitation trade-off (default: 2.0)
3. **3-Iteration Strategy**:
   - **Iteration 1 (Exploration)**: Uniform sampling across X space
   - **Iteration 2 (Exploitation)**: UCB-guided sampling focusing on promising regions
   - **Iteration 3 (Refinement)**: Peak-focused sampling for precise boundaries

**Outputs**:
- `Goldilocks_Results/bayesian_goldilocks_optimization.csv`: GP sampling history
- `Goldilocks_Results/goldilocks_gp_uncertainty.png`: GP mean and uncertainty visualization
- Discovered parameters: `X_peak`, `X_peak_uncertainty`, `X_c_low`, `X_c_high`

**Advantages**:
- Works efficiently on any sample size (100 to 10,000+ universes)
- Adaptive sampling focuses universes on promising X regions
- Provides uncertainty quantification for Goldilocks boundaries
- No separate calibration step—integrated into Phase 1

### Bayesian Model Selection (Phase 28)

**Purpose**: Compare different I-definitions using information criteria and Bayesian evidence.

#### **Bayesian Information Criterion (BIC)**

**Formula**:
```
BIC = k × log(n) - 2 × log(L)
    = k × log(n) + χ²
```

Where:
- `k`: Number of free parameters (X_SCALE, ALPHA_I, + I-definition complexity)
- `n`: Number of data points (CMB pixels + Planck observables)
- `L`: Likelihood (from chi-squared: `L = exp(-χ²/2)`)
- `χ²`: Chi-squared fit to Planck 2018 data

**Interpretation**: Lower BIC indicates better model (penalizes complexity). Used for model comparison across I-definitions.

#### **Akaike Information Criterion (AIC)**

**Formula**:
```
AIC = 2k - 2×log(L)
    = 2k + χ²
```

**Corrected AIC (AICc)** for small sample sizes:
```
AICc = AIC + 2k(k+1)/(n-k-1)
```

**Interpretation**: Lower AIC/AICc indicates better model. AICc corrects for small sample bias.

#### **Nested Sampling (Bayesian Evidence)**

**Purpose**: Compute Bayesian Evidence (marginal likelihood) for rigorous model comparison.

**Method**: Uses `dynesty` library for nested sampling algorithm.

**Algorithm**:
1. **Prior Transform**: Uniform priors over parameter space
   - `X_SCALE`: [10.0, 50.0] (default)
   - `ALPHA_I`: [0.1, 2.0] (default)
2. **Log-Likelihood Function**: 
   - Simulates small Monte Carlo with given parameters
   - Computes chi-squared proxy
   - Returns: `log_likelihood = -0.5 × χ²_proxy × scale_factor`
3. **Nested Sampling**: 
   - Number of live points: `NLIVE` (default: 1000)
   - Stopping criterion: `dlogz` (default: 0.1)
   - Maximum iterations: `MAX_ITER` (default: 10000)

**Outputs**:
- `log_evidence`: Bayesian Evidence (marginal likelihood)
- `log_evidence_error`: Uncertainty in evidence estimate
- `n_iterations`: Number of nested sampling iterations
- `n_calls`: Number of likelihood function calls
- `samples`: Parameter posterior samples
- `weights`: Importance weights for samples

**Bayes Factor**: For comparing two models (e.g., I-definition A vs. B):
```
BF = exp(log_evidence_A - log_evidence_B)
```

Interpretation:
- `BF > 1`: Model A is favored
- `BF > 3`: Substantial evidence for A
- `BF > 10`: Strong evidence for A
- `BF > 100`: Decisive evidence for A

#### **Corner Plots (Parameter Posteriors)**

**Purpose**: Visualize parameter posterior distributions from nested sampling.

**Method**: Uses `corner` library to generate corner plots showing:
- 1D marginal distributions (diagonal)
- 2D joint distributions (off-diagonal)
- True parameter values (red lines)
- 16th, 50th, 84th percentiles

**Outputs**:
- `PNG_Visualizations/corner_plot_{i_def}.png`: Parameter posterior visualization

**Configuration**:
```python
MASTER_CTRL = {
    "ENABLE_BAYESIAN_ANALYSIS": True,      # Enable BIC/AIC computation
    "ENABLE_NESTED_SAMPLING": True,        # Enable nested sampling
    "NESTED_SAMPLING_NLIVE": 1000,         # Live points
    "NESTED_SAMPLING_DLOGZ": 0.1,          # Stopping criterion
    "ENABLE_CORNER_PLOTS": True,           # Generate corner plots
    # ...
}
```

**Outputs**:
- `Aggregate/bayesian_metrics_{i_def}.csv`: BIC, AIC, AICc, log_likelihood, chi²
- `PNG_Visualizations/bayesian_comparison_{i_def}.png`: BIC/AIC/chi² bar charts
- `PNG_Visualizations/corner_plot_{i_def}.png`: Parameter posterior distributions
- `Aggregate/nested_sampling_samples.csv`: Posterior samples with weights

---

## CMB Generation & Analysis

The TQE Framework generates realistic Cosmic Microwave Background (CMB) maps for simulated universes, enabling direct comparison with observational data and testing of TQE predictions.

### CMB Map Generation

#### **CAMB Integration (Optional)**

**Purpose**: Generate realistic CMB power spectra using the Code for Anisotropies in the Microwave Background.

**Method**: 
- If `CAMB_AVAILABLE = True` and `CAMB_INTEGRATION = True`:
  - Uses CAMB to compute angular power spectrum `C_ℓ` from cosmological parameters
  - Parameters derived from E-I coupling: `Omega_Lambda = E`, `H0`, `Omega_m`, etc.
  - Generates HEALPix maps from power spectrum using `synfast()`
- If CAMB unavailable or disabled:
  - Uses simplified power-law spectrum: `C_ℓ ~ ℓ^(-CMB_POWER_SLOPE)`
  - Generates Gaussian random fields on HEALPix grid

**Output Format**: HEALPix FITS files (`.fits`) with:
- Resolution: `NSIDE` (default: 128, corresponding to ~27 arcmin pixel size)
- Temperature fluctuations in micro-Kelvin (µK)
- Full-sky coverage in HEALPix pixelization

#### **E-I Coupling in CMB Generation**

**Method**: CMB properties are modulated by E-I coupling:
- **Amplitude**: Scaled by Complexity parameter X
- **Power Spectrum Slope**: Modulated by Information parameter I
- **Anisotropy Pattern**: Influenced by lock-in epoch and stability quality

**Physical Interpretation**: The CMB preserves information about the primordial quantum fluctuations from which the universe emerged. In TQE, these fluctuations are influenced by the E-I coupling, imprinting signatures that can be detected as anomalies.

### CMB Analysis (Phase 19)

#### **Gaussianity Tests**

**Purpose**: Test whether simulated CMB maps follow Gaussian statistics (as expected from inflation).

**Method**:
- **Skewness**: `S = ⟨(T - μ)³⟩ / σ³`
- **Kurtosis**: `K = ⟨(T - μ)⁴⟩ / σ⁴ - 3`
- **Chi-Squared Test**: Compare histogram to expected Gaussian distribution

**Outputs**:
- `PNG_Visualizations/cmb_gaussianity_EI_Pipeline_v4.2.0_Pro.png`: Gaussianity analysis plots
- Statistical summary in CSV

#### **Isotropy Tests**

**Purpose**: Test whether CMB maps are statistically isotropic (no preferred direction).

**Method**:
- **Directional Variance Analysis**: Computes variance in different sky directions
- **Hemispherical Comparison**: Compares statistics in opposite hemispheres
- **Variance Map**: Visualizes directional variance across the sky

**Outputs**:
- `PNG_Visualizations/cmb_isotropy_EI_Pipeline_v4.2.0_Pro.png`: Isotropy analysis plots

#### **Power Spectrum Analysis**

**Purpose**: Compute angular power spectrum `C_ℓ` for comparison with Planck data.

**Method**:
- Uses HEALPix `anafast()` to compute `C_ℓ = (1/(2ℓ+1)) × Σ_m |a_ℓm|²`
- Bins multipoles for visualization
- Compares to theoretical power-law or CAMB prediction

**Outputs**:
- `PNG_Visualizations/cmb_power_spectrum_EI_Pipeline_v4.2.0_Pro.png`: Power spectrum plot
- `Aggregate/cmb_statistics_EI_Pipeline_v4.2.0_Pro.csv`: Statistical summary

### Planck 2018 Comparison (Phase 15)

**Purpose**: Compare simulated universes to real observational data (ONLY phase using Planck data).

**Method**:
- Resolves the Planck 2018 TT spectrum automatically: downloads the official file (or synthesizes a CAMB/analytic surrogate) into `planck_data/` if it is missing
- Computes chi-squared fit: `χ² = Σ_ℓ (C_ℓ_sim - C_ℓ_Planck)² / σ_ℓ²`
- Generates comparison plots

**Outputs**:
- `PNG_Visualizations/planck_validation_EI_Pipeline_v4.2.0_Pro.png`: Power spectrum comparison
- `Aggregate/planck_validation_EI_Pipeline_v4.2.0_Pro.csv`: Chi-squared values (χ² and χ²/dof) per universe
- `planck_data/COM_PowerSpect_CMB-TT-full_R3.01.txt`: Resolved or synthesized Planck TT spectrum used for validation
- Planck chi-squared value (used in Phase 28 for Bayesian analysis)

**Important Note**: Phase 15 is the ONLY phase that uses Planck observational data. All other phases (including anomaly detection) work with simulated maps only, ensuring that anomalies are fully emergent rather than forced to match observations.

---

## Enhanced Physics Engine

The TQE Framework v4.2.0 PRO includes an Enhanced Physics Engine that integrates real-world physics calculations, enabling more realistic simulations and direct comparison with observational cosmology.

### Friedmann Evolution

**Purpose**: Compute cosmological evolution using Friedmann equations with E-I coupling.

**Methods**:

#### **Hubble Parameter Calculation**

**Formula**:
```
H(a) = H₀ × √[Ω_m/a³ + Ω_Λ + Ω_k/a²]
```

Where:
- `H₀`: Hubble constant (default: 67.36 km/s/Mpc from Planck 2018)
- `Ω_m`: Matter density (default: 0.3153)
- `Ω_Λ`: Dark energy density = E (from simulation)
- `Ω_k`: Curvature density = 1 - Ω_m - Ω_Λ
- `a`: Scale factor

**E-I Coupling**: Dark energy density `Ω_Λ` is set by Energy parameter E, modulated by Information I through the coupling function.

#### **Universe Age Calculation**

**Formula**:
```
t(a) = (1/H₀) × ∫[0 to a] da' / [a' × √(Ω_m/a'³ + Ω_Λ + Ω_k/a'²)]
```

**Output**: Universe age in Gyr (Gigayears), compared to Planck 2018 value (13.8 Gyr).

#### **Cosmological Epoch Detection**

**Method**: Identifies cosmological epochs based on scale factor:
- **Radiation Dominated**: `a < a_eq` (matter-radiation equality)
- **Matter Dominated**: `a_eq < a < a_Λ` (dark energy domination)
- **Dark Energy Dominated**: `a > a_Λ`

**E-I Coupling**: Transition epochs depend on E (dark energy density), which is influenced by I through the coupling mechanism.

### Quantum Field Fluctuations

**Purpose**: Analyze quantum field theory aspects of the simulation.

**Methods**:

#### **Vacuum Energy Density**

**Formula**:
```
ρ_vac = (3H₀²/8πG) × Ω_Λ
```

**E-I Coupling**: Vacuum energy is directly related to E parameter, with I modulating the effective cosmological constant.

#### **Zero-Point Energy**

**Method**: Computes zero-point energy contributions from quantum fields.

**Formula**:
```
E_zp = (1/2) × Σ_k ℏω_k
```

#### **Quantum Fluctuation Amplitudes**

**Method**: Analyzes amplitude of quantum fluctuations in different modes.

**Output**: Distribution of fluctuation amplitudes, used for anomaly detection (Phase 25).

### Cosmic Entanglement Network

**Purpose**: Model quantum entanglement networks in the early universe.

**Methods**:

#### **Entanglement Entropy**

**Formula**:
```
S_ent = -Tr[ρ_A log(ρ_A)]
```

Where `ρ_A` is the reduced density matrix of a spatial region A.

**E-I Coupling**: Information parameter I is related to entanglement entropy, providing a physical interpretation of I as quantum information content.

#### **Holographic Entropy**

**Method**: Computes holographic entropy bound (Bekenstein bound):
```
S_holographic ≤ A/(4G)
```

Where A is the area of the boundary.

**Physical Interpretation**: Tests holographic principle—information content is bounded by surface area, not volume.

#### **Entanglement-Stability Correlation**

**Method**: Analyzes correlation between entanglement entropy and universe stability.

**Finding**: Universes with higher entanglement entropy tend to be more stable, suggesting that quantum information plays a role in law stabilization.

### Physical Constants Calculation

**Purpose**: Compute real-world physics constants from E-I parameters.

**Methods**:

The Enhanced Physics Engine calculates:
- **Fine Structure Constant**: `α_EM = 1/137.036...` (modulated by I)
- **Strong Coupling Constant**: `α_S = 0.1181` (at MZ scale)
- **Gravitational Constant**: `G = 6.67430×10⁻¹¹ m³/kg/s²`
- **Planck Constant**: `h = 6.62607015×10⁻³⁴ J⋅s`
- **Speed of Light**: `c = 299792458 m/s`
- **Elementary Charge**: `e = 1.602176634×10⁻¹⁹ C`
- **Particle Masses**: Proton, electron, neutrino masses

**E-I Coupling**: Physical constants are derived from E-I coupling, providing a mechanism for fine-tuning.

### Inflation Parameters

**Purpose**: Compute inflation parameters from E-I coupling.

**Methods**:

#### **Inflation Scale**

**Formula**:
```
E_inflation = INFLATION_SCALE × f(E, I)
```

Default: `INFLATION_SCALE = 1e16 GeV`

#### **Reheating Temperature**

**Formula**:
```
T_reheat = REHEATING_TEMPERATURE × f(E, I)
```

Default: `REHEATING_TEMPERATURE = 1e15 GeV`

**E-I Coupling**: Inflation parameters are modulated by E-I interaction, influencing the early universe dynamics.

### Outputs

**Enhanced Physics Analysis (Phase 23)**:
- `PNG_Visualizations/friedmann_evolution_EI_Pipeline_v4.2.0_Pro.png`: Friedmann evolution plots
- `PNG_Visualizations/quantum_field_analysis_EI_Pipeline_v4.2.0_Pro.png`: Quantum field analysis
- `PNG_Visualizations/cosmic_entanglement_analysis_EI_Pipeline_v4.2.0_Pro.png`: Entanglement network visualization
- `Aggregate/enhanced_physics_EI_Pipeline_v4.2.0_Pro.csv`: Complete physics data
- `Aggregate/physics_analysis.json`: Physics summary JSON

**Configuration**:
```python
MASTER_CTRL = {
    "USE_ENHANCED_PHYSICS": True,          # Enable enhanced physics engine
    "FRIEDMANN_AGE_CALCULATION": True,     # Universe age calculation
    "QUANTUM_FIELD_FLUCTUATIONS": True,    # Quantum field analysis
    "COSMIC_ENTANGLEMENT_NETWORK": True,   # Entanglement network
    # ...
}
```

---

## Results & Output Structure

The TQE Framework generates a comprehensive, organized output structure for each simulation run. All outputs are timestamped and organized by category for easy navigation and analysis.

### Directory Structure

Each simulation run creates a timestamped directory with the following structure:

```
TQE_Universe_Simulation_Full_Pipeline_v4.2.0_PRO/
└── TQE_Universe_Simulation_Full_Pipeline_{EI|E_only}_{YYYYMMDD_HHMMSS}/
    ├── Goldilocks_Results/
    │   ├── bayesian_goldilocks_optimization.csv
    │   ├── goldilocks_zone_{i_def}.png
    │   └── goldilocks_gp_uncertainty.png
    │
    ├── PNG_Visualizations/          # All visualization plots (55+ PNG files)
    │   ├── stability_curve.png
    │   ├── scatter_EI.png
    │   ├── cmb_map_all_anomalies_EI_Pipeline_v4.2.0_Pro.png
    │   ├── planck_validation_EI_Pipeline_v4.2.0_Pro.png
    │   ├── advanced_law_detection_analysis_EI_Pipeline_v4.2.0_Pro.png
    │   └── ... (50+ more plots)
    │
    ├── Aggregate/                   # All CSV and JSON data files (35+ files)
    │   ├── tqe_runs.csv
    │   ├── planck_chi2_EI_Pipeline_v4.2.0_Pro.csv
    │   ├── cmb_coldspots_summary_{i_def}.csv
    │   ├── cmb_aoe_summary_{i_def}.csv
    │   ├── advanced_anomaly_detection_results_EI_Pipeline_v4.2.0_Pro.csv
    │   ├── advanced_law_detection_results_EI_Pipeline_v4.2.0_Pro.csv
    │   ├── bayesian_metrics_{i_def}.csv
    │   ├── nested_sampling_samples.csv
    │   ├── summary_full.json
    │   └── ... (25+ more CSV/JSON files)
    │
    ├── Categorized_Results/         # Best universes by category
    │   ├── lock_in/
    │   │   ├── 1_FIGURES/           # Entropy evolution plots
    │   │   │   ├── best_universe_rank01_uidXXXXX_entropy_evolution.png
    │   │   │   └── ...
    │   │   ├── 2_DATA_FILES/        # Entropy timeseries CSV
    │   │   │   ├── best_universe_rank01_uidXXXXX_entropy_timeseries.csv
    │   │   │   └── ...
    │   │   └── 3_CMB_MAPS/          # CMB maps (FITS) + anomaly overlays (PNG)
    │   │       ├── cmb_uidXXXXX.fits
    │   │       ├── cmb_uidXXXXX_coldspot_overlay_EI_Pipeline_v4.2.0_Pro.png
    │   │       ├── cmb_uidXXXXX_aoe_overlay_EI_Pipeline_v4.2.0_Pro.png
    │   │       └── ...
    │   ├── stable/                  # Same structure as lock_in
    │   └── unstable/                # Same structure as lock_in
    │
    └── summary_full.json            # Complete pipeline summary (all metrics)
```

### Output File Types

#### **1. CSV Files (35+ files)**

**Core Simulation Data**:
- `tqe_runs.csv`: Complete universe ensemble data (E, I, X, stability, lock_epoch, etc.)
- `universe_seeds.csv`: Reproducibility seed mapping (master_seed → per-universe seeds)
- `pre_fluctuation_pairs.csv`: E-I pairs before quantum fluctuations

**CMB Analysis**:
- `cmb_coldspots_summary_{i_def}.csv`: Detected cold spots (coordinates, z-scores, temperatures)
- `cmb_aoe_summary_{i_def}.csv`: Axis of Evil alignments (angles, p-values, axes)
- `cmb_statistics_EI_Pipeline_v4.2.0_Pro.csv`: CMB statistical properties
- `planck_chi2_EI_Pipeline_v4.2.0_Pro.csv`: Chi-squared fits to Planck 2018 data

**Anomaly & Law Detection**:
- `advanced_anomaly_detection_results_EI_Pipeline_v4.2.0_Pro.csv`: Multi-category anomaly catalogue
- `advanced_law_detection_results_EI_Pipeline_v4.2.0_Pro.csv`: Multi-category law catalogue

**Bayesian Analysis**:
- `bayesian_metrics_{i_def}.csv`: BIC, AIC, AICc, log-likelihood, chi²
- `nested_sampling_samples.csv`: Parameter posterior samples with weights

**Timeseries Data**:
- `fl_fluctuation_timeseries.csv`: Quantum fluctuation evolution
- `fl_superposition_timeseries.csv`: Superposition phase evolution
- `fl_collapse_timeseries.csv`: Collapse phase evolution
- `fl_expansion_timeseries.csv`: Expansion phase evolution
- `{universe_id}_entropy_timeseries.csv`: Per-universe entropy evolution

**Analysis Results**:
- `feature_importance_summary.csv`: Random Forest feature importance rankings
- `emergent_law_summary.csv`: Detected power-law and phase transition laws
- `statistical_finetuning_summary.csv`: Finetuning gap analysis
- `stability_distribution_five.csv`: Universe fate categorization

#### **2. PNG Visualization Files (55+ files)**

**Stability & Goldilocks**:
- `stability_curve.png`: Stability vs. Complexity (X) with Goldilocks zone overlay
- `goldilocks_zone_{i_def}.png`: Goldilocks zone visualization with Bayesian GP uncertainty
- `goldilocks_gp_uncertainty.png`: Gaussian Process mean and uncertainty bands

**Parameter Space**:
- `scatter_EI.png`: E-I parameter space mapping (colored by stability)
- `stability_by_I.png`: Stability correlation with Information parameter

**Quantum Fluctuations**:
- `fl_fluctuation.png`: Quantum fluctuation dynamics
- `fl_superposition.png`: Superposition phase visualization
- `fl_collapse.png`: Collapse phase visualization
- `fl_expansion.png`: Expansion phase visualization

**CMB Maps & Anomalies**:
- `cmb_map_all_anomalies_EI_Pipeline_v4.2.0_Pro.png`: Aggregate CMB map with all detected anomalies
  - Dark blue circles (○) = Cold Spots
  - Yellow circles (○) = Axis of Evil alignments
- `coldspots_pos_heatmap_EI_Pipeline_v4.2.0_Pro.png`: Cold spot positional distribution
- `coldspots_z_hist_EI_Pipeline_v4.2.0_Pro.png`: Cold spot depth distribution
- `aoe_angle_hist_EI_Pipeline_v4.2.0_Pro.png`: AOE alignment angle distribution
- `planck_validation_EI_Pipeline_v4.2.0_Pro.png`: Power spectrum comparison with Planck 2018

**CMB Statistical Analysis**:
- `cmb_gaussianity_EI_Pipeline_v4.2.0_Pro.png`: Gaussianity tests (skewness, kurtosis)
- `cmb_isotropy_EI_Pipeline_v4.2.0_Pro.png`: Isotropy tests (directional variance)
- `cmb_power_spectrum_EI_Pipeline_v4.2.0_Pro.png`: Angular power spectrum C_ℓ

**Laws & Anomalies**:
- `advanced_law_detection_analysis_EI_Pipeline_v4.2.0_Pro.png`: Comprehensive law visualization (4-panel)
- `advanced_anomaly_detection_analysis_EI_Pipeline_v4.2.0_Pro.png`: Anomaly distribution plots

**Machine Learning**:
- `feature_importance_classification.png`: Random Forest classification feature importance
- `feature_importance_regression.png`: Random Forest regression feature importance

**Bayesian Analysis**:
- `bayesian_comparison_{i_def}.png`: BIC/AIC/chi² bar charts
- `corner_plot_{i_def}.png`: Parameter posterior distributions (if nested sampling enabled)

**Physics Analysis**:
- `friedmann_evolution_EI_Pipeline_v4.2.0_Pro.png`: Hubble parameter evolution
- `quantum_field_analysis_EI_Pipeline_v4.2.0_Pro.png`: Quantum field fluctuations
- `cosmic_entanglement_analysis_EI_Pipeline_v4.2.0_Pro.png`: Entanglement network visualization

**Per-Universe CMB Overlays** (in `Categorized_Results/{category}/3_CMB_MAPS/`):
- `cmb_uidXXXXX_coldspot_overlay_EI_Pipeline_v4.2.0_Pro.png`: Cold spot overlay (lime X markers)
- `cmb_uidXXXXX_aoe_overlay_EI_Pipeline_v4.2.0_Pro.png`: Axis of Evil overlay (colored star markers)

#### **3. FITS Files (CMB Maps)**

**Location**: `Categorized_Results/{category}/3_CMB_MAPS/`

**Format**: HEALPix FITS files (`.fits`)
- Resolution: `NSIDE` (default: 128, ~27 arcmin pixel size)
- Temperature fluctuations in micro-Kelvin (µK)
- Full-sky coverage in HEALPix pixelization

**Files**: One FITS file per best universe (top 3 in each category: lock_in, stable, unstable)
- `cmb_uidXXXXX.fits`: HEALPix CMB map

**Usage**: Can be opened with standard astronomical tools (HEALPix, healpy, TOPCAT, etc.)

#### **4. JSON Files**

**`summary_full.json`**: Complete pipeline summary containing:
- Pipeline metadata (version, timestamp, execution mode, I-definition)
- Reproducibility information (master_seed, seed generation method)
- Goldilocks parameters (X_peak, X_c_low, X_c_high, uncertainty)
- Universe statistics (total count, stability rates, lock-in rates)
- CMB statistics (cold spot count, AOE count, Planck chi²)
- Bayesian metrics (BIC, AIC, log_evidence if nested sampling enabled)
- File paths (relative paths to all generated outputs)
- Configuration snapshot (MASTER_CTRL parameters used)

**`goldilocks_optimization.json`**: Bayesian Goldilocks optimization results:
- GP hyperparameters (length scale, signal variance, noise level)
- Discovered parameters (X_peak, X_peak_uncertainty, X_c_low, X_c_high)
- Sampling history (X values, stability observations, GP predictions)

**`physics_analysis.json`**: Enhanced physics analysis summary:
- Friedmann evolution parameters
- Quantum field properties
- Entanglement network statistics
- Physical constants (if computed)

### File Naming Conventions

**Variant Tagging**: All output files include variant tags:
- `_EI_Pipeline_v4.2.0_Pro`: E+I coupling mode
- `_E_only_Pipeline_v4.2.0_Pro`: E-only baseline mode

**I-Definition Tagging**: Files specific to I-definitions include the definition name:
- `_{i_def}`: e.g., `_kl_shannon`, `_jensen_shannon`, etc.

**Universe ID Formatting**: Universe IDs are zero-padded to 5 digits:
- `uid00001`, `uid00142`, `uid07804`, etc.

### Output Size Estimates

**Single Run (10,000 universes, E+I mode)**:
- CSV files: ~50-100 MB
- PNG files: ~200-500 MB (55+ plots at 180 DPI)
- FITS files: ~10-20 MB (9 CMB maps: 3 categories × 3 universes)
- JSON files: ~1-5 MB
- **Total**: ~300-600 MB per run

**Batch Mode (11 runs: E-only + 10 I-definitions)**:
- **Total**: ~3-7 GB (11 × single run size)

---

## Mathematical Formalism

The TQE Framework is built on a rigorous mathematical foundation. This section provides the complete mathematical formalism underlying the simulation.

### Core TQE Equations

#### **1. Quantum State Modulation**

The universe's quantum state `P(ψ)` is modulated by energy (E) and information (I):

```
P'(ψ) = P(ψ) · f(E, I)
```

Where the coupling function is:

```
f(E, I) = exp(-(E - E_c)²/(2σ²)) · (1 + α·I)
```

**Parameters**:
- `E_c`: Critical energy (Goldilocks zone center)
- `σ`: Stability window width
- `α`: Information orientation bias strength (ALPHA_I)
- `I`: Information parameter (0 ≤ I ≤ 1)

#### **2. Complexity Parameter (X)**

The Complexity parameter X quantifies the E-I coupling strength:

**E-only mode**:
```
X = E × X_SCALE
```

**E+I mode** (depends on `X_MODE`):
- `X_MODE = "E_plus_I"`: `X = X_SCALE × (E + ALPHA_I × I)`
- `X_MODE = "product"`: `X = X_SCALE × E × (ALPHA_I × I)`
- `X_MODE = "E_times_I_pow"`: `X = X_SCALE × E × (ALPHA_I × I)^X_I_POWER`

#### **3. Stability Criterion**

A universe is considered **stable** if:

```
X ∈ [X_c_low, X_c_high]  (Goldilocks zone)
```

Where the Goldilocks zone is discovered via Bayesian Adaptive Optimization (Phase 1).

#### **4. Lock-In Mechanism**

**E-only mode**: Tracks emergent CMB observables:
- Amplitude: `A = f(E)`
- Spectral index: `n_s = f(E)`
- Hubble parameter: `H = f(E)`

Lock-in occurs when these observables stabilize (coefficient of variation < threshold).

**E+I mode**: Tracks X coupling stability:
- `X(t)` is computed at each time step
- Lock-in occurs when `CV(X) < LOCK_IN_THRESHOLD` (default: 0.01)

**Lock-in epoch**: First time step where lock-in criterion is satisfied.

### I-Parameter Definitions (Mathematical Formulae)

All I-definitions include E-modulation: `I_final = I_base × √(E_ref/E)`, where `E_ref = 0.7` (Planck 2018 value).

#### **1. KL-Divergence**

```
I_KL = D_KL(P(ψ) || P_ref(ψ)) = Σ_i P(ψ_i) × log(P(ψ_i) / P_ref(ψ_i))
```

Where `P_ref(ψ)` is a reference quantum state (typically uniform or vacuum).

#### **2. Shannon Entropy**

```
I_Shannon = -Σ_i P(ψ_i) × log(P(ψ_i))
```

#### **3. Rényi Entropy (α=2, Collision Entropy)**

```
I_Rényi = -log(Σ_i P(ψ_i)²)
```

#### **4. Mutual Information**

```
I_MI = H(X) + H(Y) - H(X, Y)
```

Where `H(X)`, `H(Y)` are marginal entropies, `H(X, Y)` is joint entropy.

#### **5. Entanglement Entropy (Von Neumann)**

```
I_Entanglement = -Tr[ρ_A log(ρ_A)]
```

Where `ρ_A` is the reduced density matrix of subsystem A.

#### **6. Fisher Information**

```
I_Fisher = Σ_i (∂P(ψ_i)/∂θ)² / P(ψ_i)
```

Where `θ` is a parameter (typically E or I).

#### **7. Composite Product**

```
I_Composite = I_KL × I_Shannon
```

Multiplicative fusion (strict filtering).

#### **8. KL-Shannon Refined (Harmonic Mean)**

```
I_KL_Shannon = 2 × (I_KL × I_Shannon) / (I_KL + I_Shannon)
```

Balanced, outlier-robust fusion.

#### **9. Fisher-KL Fusion**

```
I_Fisher_KL = √(I_Fisher × I_KL)
```

Geometric mean fusion (quantum metrology + distinguishability).

#### **10. Jensen-Shannon Divergence**

```
I_JS = 0.5 × [D_KL(P || M) + D_KL(Q || M)]
```

Where `M = 0.5 × (P + Q)` is the midpoint distribution. Symmetric, bounded (0 ≤ I_JS ≤ 1).

**Validation**: This definition was validated with Planck 2018 CMB data for optimal I parameter measurement.

### Bayesian Goldilocks Optimization

#### **Gaussian Process Model**

The stability probability is modeled as:

```
P(stable | X) ~ GP(μ(X), k(X, X'))
```

**Kernel (RBF + White Noise)**:
```
k(X, X') = σ_f² × exp(-(X - X')²/(2ℓ²)) + σ_n² × δ(X, X')
```

**Hyperparameters**:
- `σ_f²`: Signal variance
- `ℓ`: Length scale
- `σ_n²`: Noise variance

#### **Upper Confidence Bound (UCB) Acquisition**

Next X to sample:
```
X_next = argmax[μ(X) + κ × σ(X)]
```

Where:
- `μ(X)`: GP mean (exploitation)
- `σ(X)`: GP uncertainty (exploration)
- `κ`: Exploration-exploitation trade-off (default: 2.0)

### Bayesian Model Selection

#### **Bayesian Information Criterion (BIC)**

```
BIC = k × log(n) - 2 × log(L)
    = k × log(n) + χ²
```

Where:
- `k`: Number of free parameters
- `n`: Number of data points
- `L`: Likelihood = exp(-χ²/2)
- `χ²`: Chi-squared fit to Planck 2018 data

#### **Akaike Information Criterion (AIC)**

```
AIC = 2k - 2×log(L) = 2k + χ²
```

**Corrected AIC (AICc)**:
```
AICc = AIC + 2k(k+1)/(n-k-1)
```

#### **Bayesian Evidence (Nested Sampling)**

```
Z = ∫ L(θ) × π(θ) dθ
```

Where:
- `L(θ)`: Likelihood function
- `π(θ)`: Prior distribution
- `θ`: Parameters (X_SCALE, ALPHA_I)

**Bayes Factor**:
```
BF = Z_A / Z_B
```

### CMB Generation (CAMB Integration)

#### **Angular Power Spectrum**

```
C_ℓ = (1/(2ℓ+1)) × Σ_m |a_ℓm|²
```

Where `a_ℓm` are spherical harmonic coefficients.

#### **E-I Coupling in Power Spectrum**

**Primordial Power Spectrum Parameters**:
- Spectral index: `n_s = 0.965 + 0.05×(I - I_obs) + 0.02×(E - E_obs)`
- Amplitude: `A_s = 2.1e-9 × (E/E_obs)^(-0.3) × (1 + 0.1×(I - I_obs))`
- Tensor-to-scalar ratio: `r = 0.01 × (1 + 0.5×(I - I_obs)) × (E/E_obs)^0.1`

Where `E_obs = 0.7`, `I_obs = 0.5` (reference values).

### Friedmann Evolution

#### **Hubble Parameter**

```
H(a) = H₀ × √[Ω_m/a³ + Ω_Λ + Ω_k/a²]
```

Where:
- `Ω_Λ = E` (dark energy density from simulation)
- `Ω_k = 1 - Ω_m - Ω_Λ` (curvature)
- `a`: Scale factor

#### **Universe Age**

```
t(a) = (1/H₀) × ∫[0 to a] da' / [a' × √(Ω_m/a'³ + Ω_Λ + Ω_k/a'²)]
```

For flat universe (Ω_k ≈ 0):
```
t ≈ (2/3) × (1/H₀) × arcsinh(√(Ω_Λ/Ω_m))
```

### Entropy Evolution

#### **Bekenstein-Hawking Horizon Entropy**

```
S_BH = A/(4G) = π × r_h²
```

Where `r_h = 1/H` is the horizon radius.

#### **Quantum Fluctuation Entropy**

```
S_quantum = -Σ_i P(ψ_i) × log(P(ψ_i))
```

### Anomaly Detection Formulae

#### **Cold Spot Z-Score**

```
z = (T - μ) / σ
```

Where `T` is pixel temperature, `μ` and `σ` are mean and standard deviation of smoothed map.

#### **Axis of Evil Alignment Angle**

```
θ = arccos(|n̂₂ · n̂₃|)
```

Where `n̂₂` and `n̂₃` are unit vectors for quadrupole (ℓ=2) and octupole (ℓ=3) axes.

---

## Assessment & Validation

The TQE Framework includes multiple validation mechanisms to ensure scientific rigor and reproducibility.

### Reproducibility Validation

**Two-Tiered Seeding Hierarchy**:
1. **Master Seed**: Set via `MASTER_CTRL["SEED"]` (default: 42)
2. **Per-Universe Seeds**: `universe_seed = master_seed + universe_id`

**Validation**: Running the same configuration twice with the same master seed produces **identical results** (bit-for-bit reproducibility).

**Output**: `Aggregate/universe_seeds.csv` provides full seed traceability.

### Statistical Validation

**Monte Carlo Convergence**:
- Stability rates should converge as `N_universes` increases
- Goldilocks zone boundaries should stabilize with sufficient sampling

**Validation Metrics**:
- Coefficient of variation of stability rates across independent runs
- Goldilocks zone boundary uncertainty (from Bayesian GP)

### Observational Validation

**Planck 2018 Comparison (Phase 15)**:
- Chi-squared fit to Planck CMB power spectrum
- Comparison of cosmological parameters (H₀, Ω_m, Ω_Λ)
- **Note**: This is the ONLY phase using observational data—all other phases use simulated maps only

**Validation Criteria**:
- Chi-squared per degree of freedom: `χ²/ν < 2.0` (good fit)
- Best universes should have `χ²/ν ≈ 1.0` (excellent fit)

### Anomaly Validation

**Emergent Anomaly Detection**:
- Cold spots and Axis of Evil are **fully emergent** (not forced to match Planck)
- Validation: Compare simulated anomaly rates to observed rates
- **Key Finding**: Rare anomalies (3/10,000 universes) match observed rarity

**Statistical Significance**:
- Monte Carlo p-values for AOE alignments
- Z-score distributions for cold spots

### Bayesian Model Validation

**Information Criteria**:
- Lower BIC/AIC indicates better model
- AICc corrects for small sample bias

**Nested Sampling Validation**:
- Evidence error should be small: `log_evidence_error < 0.1`
- Posterior samples should converge (Gelman-Rubin statistic)

### Code Quality Validation

**File Protection**:
- Empty file detection and automatic cleanup
- Error handling for missing dependencies (CAMB, healpy, etc.)

**Determinism Validation**:
- Dedicated RNG streams for different components (CMB generation, anomaly detection, etc.)
- Seed isolation prevents cross-component contamination

---

## Limitations & Known Issues

The TQE Framework is a research-grade proof-of-concept. The following limitations should be considered when interpreting results:

### Computational Limitations

**Sample Size**:
- Default: 10,000 universes (configurable via `N_UNIVERSES`)
- Larger ensembles (>100,000) require significant computational resources
- Bayesian Goldilocks optimization scales efficiently, but full pipeline runtime scales linearly with `N_UNIVERSES`

**CMB Map Resolution**:
- Default: `NSIDE = 128` (~27 arcmin pixel size)
- Higher resolution (`NSIDE = 256, 512`) requires more memory and computation time
- CAMB integration may fail for very high `NSIDE` values

**Nested Sampling**:
- Computationally expensive (may take hours for full convergence)
- Default: `NLIVE = 1000`, `MAX_ITER = 10000`
- Can be disabled via `ENABLE_NESTED_SAMPLING = False`

### Theoretical Limitations

**Simplified Physics**:
- E-I coupling is phenomenological (not derived from first principles)
- CMB generation uses simplified power spectrum (unless CAMB is available)
- Enhanced physics engine provides approximations, not exact solutions

**I-Parameter Definitions**:
- 10 definitions are tested, but the "correct" definition is unknown
- E-modulation (`√(E_ref/E)`) is an assumption, not derived
- Validation with Planck data suggests Jensen-Shannon divergence is optimal, but this requires further investigation

**Lock-In Mechanism**:
- E-only mode tracks CMB observables (may not capture all stabilization mechanisms)
- E+I mode tracks X coupling (TQE-specific, requires validation)
- Lock-in threshold (`LOCK_IN_THRESHOLD = 0.01`) is arbitrary

### Observational Limitations

**Planck Data Usage**:
- Only Phase 15 uses Planck 2018 data (for chi-squared comparison)
- All other phases use simulated maps only (ensuring emergent anomalies)
- This design choice ensures anomalies are genuine TQE predictions, not forced matches

**Anomaly Detection**:
- Cold spot detection uses multi-scale Gaussian smoothing (may miss non-Gaussian features)
- Axis of Evil detection uses Monte Carlo significance (limited by `NREALIZ` parameter)
- Additional anomalies (HPA, etc.) are not yet implemented

### Software Limitations

**Dependencies**:
- CAMB integration requires `camb` library (optional but recommended)
- HEALPix operations require `healpy` library (optional but recommended)
- Nested sampling requires `dynesty` library (optional)
- Corner plots require `corner` library (optional)

**Error Handling**:
- CAMB errors are aggregated and reported (pipeline continues with fallback)
- Missing dependencies trigger fallback modes (simplified CMB generation, etc.)
- Some errors may be silently caught (check logs for warnings)

**Platform Compatibility**:
- Tested on Linux, macOS, and Google Colab
- Windows compatibility not fully tested
- Google Drive integration requires Colab environment

### Future Improvements

**Planned Enhancements**:
1. **HPC Optimization**: Parallel processing for large ensembles
2. **Additional Anomalies**: Hemispherical Power Asymmetry (HPA), etc.
3. **Modular Architecture**: Refactoring into separate modules for easier maintenance
4. **Extended I-Definitions**: Additional information-theoretic measures
5. **Advanced Physics**: More realistic CMB generation, inflation models, etc.

---

## Troubleshooting

Common issues and solutions:

### Issue: CAMB Errors

**Symptoms**: `[CAMB] ERROR: ...` messages in output

**Solutions**:
1. Install CAMB: `pip install camb`
2. Disable CAMB: Set `CAMB_INTEGRATION = False` in `MASTER_CTRL`
3. Check CAMB version compatibility (tested with `camb >= 1.3.0`)

### Issue: HEALPix Errors

**Symptoms**: `[HEALPY] ERROR: ...` or missing CMB maps

**Solutions**:
1. Install healpy: `pip install healpy`
2. Check `NSIDE` value (must be power of 2: 64, 128, 256, etc.)
3. Reduce `NSIDE` if memory errors occur

### Issue: Memory Errors

**Symptoms**: `MemoryError` or system slowdown

**Solutions**:
1. Reduce `N_UNIVERSES` (default: 10,000 → try 1,000)
2. Reduce `NSIDE` (default: 128 → try 64)
3. Disable CMB map generation: Set `GENERATE_CMB_MAPS = False`
4. Use batch mode with fewer I-definitions

### Issue: Non-Reproducible Results

**Symptoms**: Different results with same seed

**Solutions**:
1. Check `SEED` is set in `MASTER_CTRL`
2. Ensure no parallel execution (use single-threaded mode)
3. Check for floating-point precision issues (rare, but possible)
4. Verify all dependencies are pinned versions

### Issue: Missing Output Files

**Symptoms**: Expected CSV/PNG files not generated

**Solutions**:
1. Check `SAVE_DIR` path exists and is writable
2. Check for empty file detection (files may be deleted if empty)
3. Review error logs for phase-specific failures
4. Verify `ENABLE_*` flags in `MASTER_CTRL` are set correctly

### Issue: Slow Performance

**Symptoms**: Pipeline takes hours to complete

**Solutions**:
1. Reduce `N_UNIVERSES` (largest performance factor)
2. Disable nested sampling: `ENABLE_NESTED_SAMPLING = False`
3. Disable CMB map generation for non-lock-in universes
4. Use `batch_ei` mode instead of `batch_all` (fewer runs)

### Issue: Google Colab Integration

**Symptoms**: Drive mount fails or files not saved

**Solutions**:
1. Manually mount Drive: `from google.colab import drive; drive.mount('/content/drive')`
2. Set `DRIVE_BASE_DIR` explicitly in `MASTER_CTRL`
3. Check Drive storage quota (may be full)
4. Use local execution if Colab issues persist

---

## Contributing

Contributions to the TQE Framework are welcome! This is an open-source research project, and community input is valuable.

### How to Contribute

**1. Bug Reports**:
- Open an issue on GitHub (TODO: add GitHub link)
- Include: error messages, configuration, system information
- Provide minimal reproducible example if possible

**2. Feature Requests**:
- Open an issue with "Feature Request" label
- Describe the feature and its scientific motivation
- Discuss implementation approach before coding

**3. Code Contributions**:
- Fork the repository
- Create a feature branch: `git checkout -b feature/amazing-feature`
- Follow existing code style (PEP 8, docstrings)
- Add tests if applicable
- Submit a pull request with detailed description

**4. Documentation**:
- Improve README clarity
- Add code comments
- Write tutorials or examples
- Translate documentation to other languages

**5. Scientific Contributions**:
- Validate I-parameter definitions
- Propose new anomaly detectors
- Extend physics engine
- Compare with observational data

### Code Style Guidelines

- **Python**: PEP 8 compliant
- **Docstrings**: Google-style docstrings
- **Type Hints**: Use type hints where applicable
- **Comments**: Explain "why", not "what"
- **Naming**: Descriptive variable/function names

### Testing

- Test new features with small `N_UNIVERSES` (100-1000)
- Verify reproducibility with fixed seeds
- Check output file generation
- Validate against known results

---

## License

This project is licensed under the **MIT License**.

See the [LICENSE](../LICENSE) file for details.

**SPDX-License-Identifier**: MIT  
**Copyright (c) 2025 Stefan Len**

### License Summary

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

## Contact & Support

**Author**: Stefan Len

**Email**: stefan@tqe-theory.space

**Repository**: TODO: Add GitHub link

**DOI**: TODO: Add Zenodo DOI link

**arXiv**: TODO: Add arXiv link

### Getting Help

- **Documentation**: Start with this README and the pipeline header comments
- **Issues**: Open a GitHub issue for bugs or feature requests
- **Email**: Contact directly for scientific collaboration or questions
- **Discussions**: TODO: Add GitHub Discussions link

### Citation

If you use the TQE Framework in your research, please cite:

```bibtex
@software{Len_2025_TQE,
  author    = {Len, Stefan},
  title     = {{TQE Universe Simulation Pipeline v4.2.0 PRO}},
  version   = {4.2.0},
  date      = {2025},
  publisher = {GitHub},
  url       = {TODO: Add GitHub link},
  doi       = {TODO: Add Zenodo DOI}
}
```

### Acknowledgments

- **CAMB**: Code for Anisotropies in the Microwave Background (Lewis & Challinor)
- **HEALPix**: Hierarchical Equal Area isoLatitude Pixelization (Górski et al.)
- **dynesty**: Dynamic Nested Sampling (Speagle)
- **scikit-learn**: Machine learning library (Pedregosa et al.)
- **NumPy, SciPy, Matplotlib**: Scientific Python ecosystem

---
