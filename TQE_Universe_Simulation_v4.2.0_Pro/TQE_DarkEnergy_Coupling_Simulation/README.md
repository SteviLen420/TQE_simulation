[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17614948.svg)](https://doi.org/10.5281/zenodo.17614948)
[![Research Status](https://img.shields.io/badge/status-active%20research-green)](https://github.com/SteviLen420/TQE_simulation)

# TQE Dark Energy Coupling Simulation v4.2.0 PRO

**Title:** TQE–ΛSim: Numerical Coupling of the I Parameter and Dark Energy Density in Quantum Cosmology  
**Author:** Stefan Len  
**Version:** v4.2.0 PRO  
**Date:** 2025-10-29

---

## Abstract

The TQE Dark Energy Coupling Simulation is a comprehensive computational pipeline designed to test the Theory of the Question of Existence (TQE) hypothesis in the context of dark energy evolution. This pipeline investigates whether the intrinsic information content (I-parameter) of the universe's energy state affects dark energy density evolution through energy-information coupling, thereby influencing cosmic expansion, structure formation, and large-scale cosmological dynamics.

The simulation implements a dual-mode framework where each cosmological model runs in two configurations: **E-only** (energy magnitude effect only) and **E+I** (energy coupled with its intrinsic information content). By comparing these modes against real observational data from Pantheon+ (SNe Ia), BOSS/eBOSS/DESI (BAO), and Planck (CMB), the pipeline quantifies the coupling strength and provides falsifiable predictions for the TQE hypothesis.

**Key Scientific Questions:**
- Is dark energy's evolution influenced by the intrinsic informational content (I-parameter) of the universe's energy state?
- Can the I-parameter, as an intrinsic property of energy, modulate dark energy density through information-energy coupling?
- Does this coupling influence the universe's trajectory toward complexity and cosmological stability?

---

## Theoretical Foundation

### Core TQE Principle

The Theory of the Question of Existence (TQE) proposes that stable physical laws emerge from the coupling of vacuum energy fluctuations (E) with an information-theoretic orientation parameter (I). The fundamental modulation equation is:

$$
P'(\psi) = P(\psi) \cdot f(E,I)
$$

where $f(E,I) = \exp\left(-\frac{(E-E_c)^2}{2\sigma^2}\right) \cdot (1 + \alpha \cdot I)$

**Key Definitions:**
- **E**: Energy (in cosmology: $E = H(a)/H_0$, normalized expansion rate)
- **I**: Information content **intrinsic** to E (NOT an independent field!)
  - Operational definition: $I = \frac{|dE/da|}{E + |dE/da|}$ (normalized asymmetry in energy evolution)
  - Measures how rapidly the energy system is changing (directional bias)
- **E_c**: Critical energy (Goldilocks zone center)
- **σ**: Stability window width
- **α**: Information orientation bias strength

### Critical Insight

**I is NOT an external field acting on energy.** I is an **INTRINSIC PROPERTY** of the energy state itself—its internal information content, measured by the temporal asymmetry (change rate) of the energy system. High $|dE/da|$ → high I (far from equilibrium, high directional bias). Low $|dE/da|$ → low I (near equilibrium, low asymmetry).

### Cosmological Models Tested

The pipeline tests four distinct coupling mechanisms:

#### Model 1: Covariant E-Pressure Coupling
- **Hypothesis**: Dark energy density responds to expansion rate (E) and information (I)
- **E-only**: $\rho_{DE} = \rho_\Lambda \cdot \exp(-\alpha \cdot E)$ (baseline: energy magnitude effect)
- **E+I**: $\rho_{DE} = \rho_\Lambda \cdot \exp(-\alpha \cdot E \cdot (1-I))$ (information modulates coupling strength)

#### Model 2: Uniform Equation of State
- **Hypothesis**: Dark energy equation of state varies with information content
- **E-only**: $w_{DE} = w_0$ (constant equation of state)
- **E+I**: $w_{DE} = w_0 + w_I \cdot I(a)$ (information-dependent equation of state)

#### Model 3: Geometric Coupling
- **Hypothesis**: Information gradients (spatial/temporal) affect dark energy
- **E-only**: $\rho_{DE} = \rho_\Lambda$ (cosmological constant)
- **E+I**: $\rho_{DE} = \rho_\Lambda \cdot \exp(\beta_0 \cdot F[I, \nabla I, \partial_t I])$ (geometric functional of I and derivatives)
  - $F_I = \text{sigmoid}((I-\langle I \rangle)/\sigma_I)^2 + \text{sigmoid}(dI/da \cdot aH/\kappa)^2$

#### Model 4: Null Model (ΛCDM)
- **Baseline**: Standard cosmology with no I-coupling
- $\rho_{DE} = \rho_\Lambda$, $w = -1$ (cosmological constant)

### Falsifiable Predictions

If TQE is correct:
- **S₈ parameter** differs between E-only and E+I modes
- **CMB anomalies** show non-random statistical signatures
- **Matter power spectrum** $P(k)$ exhibits scale-dependent features
- **I-E correlation** shows temporal lag structure indicating causal coupling

---

## Key Capabilities

### Dual Coupling Mode Framework
- **E-only mode**: Pure energy damping, baseline for comparison
- **E+I mode**: Full TQE coupling with energy-information interaction
- **Automatic comparison**: Statistical significance testing between modes

### Observational Data Integration
- **Type Ia Supernovae**: Pantheon+ survey (1,701 SNe Ia) with full covariance
- **Baryon Acoustic Oscillations**: BOSS DR12, eBOSS DR16, DESI data
- **Cosmic Microwave Background**: Planck 2018 power spectrum and component-separated maps
- **Large-Scale Structure**: Matter power spectrum and growth factor analysis

### Bayesian Inference Engine
- **MCMC sampling**: Parameter posteriors with credible intervals (emcee)
- **Nested sampling**: Bayesian evidence computation for model comparison (dynesty)
- **Bayes Factors**: Quantitative evidence ratios between rival models
- **Information criteria**: AIC, BIC, DIC for model selection

### Galaxy Structure Analysis
- **3D density field** generation from matter power spectrum
- **Cosmic web classification**: Voids, filaments, sheets, clusters
- **Structure cataloging** with size filtering and physical properties
- **Comparison** with observed large-scale structure (SDSS/2dFGRS)

### Goldilocks Optimization
- **Automatic parameter search**: Bayesian optimization (Differential Evolution)
- **Optimized parameters**: Critical energy (E_c), stability width (σ), coupling strengths (α, β₀)
- **Objective function**: Minimizes H(a) deviations while maintaining physical consistency

### CMB Planck Validation
- **Real map loading**: Component-separated maps (SMICA, NILC, SEVEM, Commander)
- **Power spectrum computation**: $C_\ell$ from Planck maps with beam correction
- **Anomaly detection**: Cold/hot spots, hemispherical asymmetries
- **NHI correlation**: Neutral Hydrogen foreground correlation analysis

### Comprehensive Output
- **32-36 files per model run**: CSV, JSON, TXT, ZIP archives
- **11-16 publication-quality plots**: PNG format with LaTeX formatting
- **Reproducibility snapshots**: Complete parameter tracking and environment info
- **Cross-model aggregation**: Automated comparison and ranking

---

## Requirements & Environment Setup

### Prerequisites

- **Python**: 3.9+ (tested on CPython 3.9–3.11)
- **Operating System**: Linux, macOS, or Windows (Google Colab recommended)
- **Google Colab + Google Drive**: Required for full functionality (local execution has limited support)

### Core Dependencies

The pipeline automatically checks and installs required packages:

```python
# Core scientific libraries
numpy >= 1.21.0
scipy >= 1.7.0
pandas >= 1.3.0
matplotlib >= 3.4.0
tqdm >= 4.62.0

# Cosmological calculations
camb >= 1.3.0  # For CMB power spectrum generation

# Bayesian inference
emcee >= 3.1.0  # MCMC sampling
dynesty >= 2.0.0  # Nested sampling
corner >= 2.2.0  # Corner plots

# CMB analysis (optional)
healpy >= 1.15.0  # HEALPix maps (auto-installed if needed)
astropy >= 5.0.0  # FITS file handling

# Google Colab integration
google.colab  # Auto-detected if running in Colab
```

### Installation

#### Google Colab (Recommended)

1. Upload `TQE_DarkEnergy_Coupling_Simulation.py` to Google Colab
2. Mount Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Run the pipeline:
   ```python
   !python TQE_DarkEnergy_Coupling_Simulation.py
   ```

The pipeline automatically:
- Detects Colab environment
- Mounts Google Drive if needed
- Installs missing packages
- Sets up directory structure

#### Local Installation (Limited Support)

```bash
# 1. Clone repository
git clone https://github.com/SteviLen420/TQE_simulation.git
cd TQE_simulation/TQE_Universe_Simulation_v4.2.0_Pro/TQE_DarkEnergy_Coupling_Simulation

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install numpy scipy pandas matplotlib tqdm camb emcee dynesty corner healpy astropy

# 4. Run pipeline
python TQE_DarkEnergy_Coupling_Simulation.py
```

**Note**: Local execution has limited functionality. The pipeline is optimized for Google Colab with Google Drive integration.

---

## Configuration Overview

All pipeline behavior is controlled by the `MASTER_CTRL` dictionary at the top of the script. This configuration-as-code approach enables experimental campaigns, parameter sweeps, and analysis settings to be defined directly within the source code, providing a clear foundation for reproducibility.

### Core Pipeline Controls

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `COUPLING_MODE` | string | `"dual"` | Options: `"Eonly"`, `"EplusI"`, `"dual"` (runs both) |
| `RUN_DUAL_COMPARISON` | bool | `True` | Run both E-only and E+I with same seed for comparison |
| `AUTO_PREFIX_FILES` | bool | `True` | Automatically prefix files with "EplusI_" or "Eonly_" |
| `MASTER_SEED` | string | `"TQE_DarkEnergy_2025"` | Deterministic seed string for reproducibility |
| `AUTO_FIND_GOLDILOCKS` | bool | `False` | Enable automatic Goldilocks zone optimization |

### Cosmological Parameters (Planck 2018 Fiducial)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `H0` | 67.4 | Hubble constant [km/s/Mpc] |
| `OMEGA_M` | 0.315 | Matter density parameter |
| `OMEGA_LAMBDA` | 0.685 | Dark energy density parameter |
| `OMEGA_B` | 0.049 | Baryon density parameter |
| `OMEGA_R` | 9.24e-5 | Radiation density parameter |
| `W0` | -1.0 | Dark energy equation of state |
| `N_S` | 0.965 | Scalar spectral index |
| `SIGMA_8` | 0.811 | Amplitude of matter fluctuations |

### I-Parameter Model Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `I_FIELD_AMPLITUDE` | 0.02 | Amplitude A for phenomenological model: $I(a) = A \cdot a^\gamma$ |
| `I_FIELD_GAMMA` | 0.5 | Power law index γ |
| `I_FIELD_EPSILON` | 1e-6 | Regularization epsilon for energy-based I-parameter |
| `I_FIELD_NORMALIZATION` | `'tanh'` | Normalization method: `'tanh'` or `'rational'` |

### Coupling Model Parameters

#### Model 1: Covariant E-Pressure
| Parameter | Default | Description |
|-----------|---------|-------------|
| `ALPHA_COUPLING` | 0.02 | Coupling strength α (optimized for H(a=1) < 0.3%) |
| `ALPHA` | 0.02 | Current α value (updated by Goldilocks finder) |
| `USE_EXP_COUPLING` | `True` | Use exponential coupling: $\rho_{DE} = \rho_\Lambda \cdot \exp(\beta_0 I - \alpha E)$ |
| `ALPHA_DAMPING` | 0.0008 | Ultra-optimized damping coefficient |

#### Model 2: Uniform w(I)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `W_I_COUPLING` | 0.05 | I-coupling to w: $w_{DE} = w_0 + w_I \cdot I(a)$ |

#### Model 3: Geometric Coupling
| Parameter | Default | Description |
|-----------|---------|-------------|
| `BETA0_COUPLING` | 0.02 | Geometric coupling strength β₀ |
| `BETA0` | 0.010 | Current β₀ value (updated by Goldilocks finder) |
| `FI_USE_SIGMOID` | `True` | Use sigmoid normalization for F_I |
| `FI_KAPPA_SCALE` | 67.4 | κ scale for dI/da normalization (H₀) |

### Numerical Integration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `A_MIN_LOG` | 1e-4 | Minimum scale factor (log-space) |
| `A_MAX_LOG` | 1.0 | Maximum scale factor |
| `A_GRID_N_LOG` | 2048 | Number of scale factor points (optimized: 2× faster than 4096) |
| `USE_LOG_A_GRID` | `True` | Use log-spaced grid (better for early universe) |
| `Z_MIN` | 0.0 | Minimum redshift |
| `Z_MAX` | 5.0 | Maximum redshift (extended from 3.0) |
| `Z_POINTS` | 100 | Number of redshift points |

### Bayesian Inference Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `RUN_MCMC` | `True` | Run MCMC posterior sampling |
| `MCMC_NWALKERS` | 50 | Number of MCMC walkers |
| `MCMC_NSTEPS` | 5000 | Number of MCMC steps (production) |
| `MCMC_BURNIN` | 1000 | Burn-in steps |
| `USE_NESTED_SAMPLING` | `True` | Use nested sampling (dynesty) for Bayes Factor |
| `NESTED_NLIVE` | 250 | Number of live points (optimized: 2× faster than 500) |
| `NESTED_DLOGZ` | 0.05 | Evidence tolerance (optimized: faster convergence) |
| `CREATE_CORNER_PLOTS` | `True` | Create corner plots (posteriors) |
| `COMPUTE_EVIDENCE` | `True` | Compute Bayesian evidence log Z |

### Prior Ranges (ΛCDM-Compatible, Physically Motivated)

| Parameter | Range | Description |
|-----------|-------|-------------|
| `PRIOR_OMEGA_M` | [0.2, 0.4] | Ω_m prior (Planck 2018: 0.315 ± 0.007) |
| `PRIOR_H0` | [60.0, 75.0] | H₀ prior [km/s/Mpc] (Planck: 67.4 ± 0.5, SH0ES: 73.0 ± 1.0) |
| `PRIOR_ALPHA` | [0.0, 0.3] | α prior range (limited to prevent instability) |
| `PRIOR_W0` | [-1.3, -0.7] | w₀ prior (near -1, within energy conditions) |
| `PRIOR_W_I` | [-0.5, 0.5] | w_I prior range (small perturbation) |
| `PRIOR_BETA0` | [0.0, 0.3] | β₀ prior range (limited to prevent instability) |

### Observational Data Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ALLOW_MOCK_DATA` | `True` | Use enhanced mock data (50 SNe, 10 BAO, 47 CMB) for testing |
| `USE_REAL_SNE_DATA` | `False` | Use real Pantheon+ data (requires data file) |
| `USE_REAL_BAO_DATA` | `False` | Use real BOSS/eBOSS/DESI data (requires data file) |
| `USE_REAL_CMB_DATA` | `False` | Use real Planck CMB data (requires data file) |
| `USE_FULL_COVARIANCE` | `True` | Use full covariance matrices (not just diagonal) |
| `USE_REAL_CMB_PLANCK_MAPS` | `True` | Enable real Planck CMB map validation |
| `CMB_PLANCK_BASE_PATH` | `"/content/drive/MyDrive/CMB_Planck_Maps"` | Google Drive path for Planck maps |

### Performance Optimization

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ENABLE_VECTORIZATION` | `True` | Vectorize array computations (2-5× faster) |
| `ENABLE_CACHING` | `True` | Cache expensive computations (3-10× faster) |
| `CACHE_SIZE` | 1000 | LRU cache size for frequently computed values |
| `PERFORMANCE_MODE` | `"balanced"` | Options: `"fast"`, `"balanced"`, `"accurate"` |
| `MEMORY_EFFICIENT_MODE` | `True` | Clean up intermediate arrays |

**Performance Mode Details:**
- **Fast mode**: Reduced resolution (A_GRID_N_LOG=1024, Z_POINTS=50)
- **Balanced mode**: Optimized (A_GRID_N_LOG=2048, Z_POINTS=100, NESTED_NLIVE=250)
- **Accurate mode**: High resolution (A_GRID_N_LOG=8192, Z_POINTS=200)

### Visualization Parameters (Publication Quality)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `PLOT_DPI` | 100 | Display DPI |
| `PLOT_SAVE_DPI` | 300 | Save DPI (publication: 300-600) |
| `PLOT_FONTSIZE_TITLE` | 16 | Title font size |
| `PLOT_FONTSIZE_LABEL` | 14 | Label font size |
| `PLOT_FONTSIZE_LEGEND` | 11 | Legend font size |
| `PLOT_FIGSIZE_DEFAULT` | (10, 7) | Default figure size |
| `PLOT_FIGSIZE_WIDE` | (14, 5) | Wide figure size (2-panel) |

---

## Running the Pipeline

### Basic Execution

The simplest way to run the pipeline is with default settings:

```bash
cd TQE_Universe_Simulation_v4.2.0_Pro/TQE_DarkEnergy_Coupling_Simulation
python TQE_DarkEnergy_Coupling_Simulation.py
```

Or in Google Colab:

```python
!python TQE_DarkEnergy_Coupling_Simulation.py
```

### Execution Modes

The pipeline supports three coupling modes, controlled by `COUPLING_MODE`:

#### Mode 1: E-Only (`"Eonly"`)
Baseline mode for comparison - Energy-only coupling with I parameter disabled.

```python
MASTER_CTRL = {
    "COUPLING_MODE": "Eonly",
    # ... other settings
}
```

#### Mode 2: E+I (`"EplusI"`)
Full TQE coupling mode with energy-information interaction.

```python
MASTER_CTRL = {
    "COUPLING_MODE": "EplusI",
    # ... other settings
}
```

#### Mode 3: Dual (`"dual"`)
Runs both E-only and E+I modes with the same seed for direct comparison.

```python
MASTER_CTRL = {
    "COUPLING_MODE": "dual",
    "RUN_DUAL_COMPARISON": True,
    # ... other settings
}
```

### Parameter Sweeps

#### β₀ Sweep (Model 3 Geometric)

Enable automatic β₀ parameter sweep:

```python
MASTER_CTRL = {
    "RUN_BETA0_SWEEP": True,
    "BETA0_RANGE": [0.0, 0.01, 0.02, 0.03, 0.05, 0.10],  # 21 values from 0.000 to 0.100
    # ... other settings
}
```

This generates 21 model runs (one per β₀ value) for comprehensive parameter space exploration.

### Goldilocks Optimization

Enable automatic Goldilocks zone optimization:

```python
MASTER_CTRL = {
    "AUTO_FIND_GOLDILOCKS": True,
    # ... other settings
}
```

This runs Bayesian optimization (Differential Evolution) to find optimal TQE parameters (E_c, σ, α, β₀) before the main pipeline execution.

### Custom Configuration

Edit the `MASTER_CTRL` dictionary at the top of the script:

```python
MASTER_CTRL = {
    # Core settings
    "COUPLING_MODE": "dual",
    "RUN_DUAL_COMPARISON": True,
    "AUTO_FIND_GOLDILOCKS": False,
    
    # Cosmological parameters
    "H0": 67.4,
    "OMEGA_M": 0.315,
    "OMEGA_LAMBDA": 0.685,
    
    # Coupling parameters
    "ALPHA": 0.02,
    "BETA0": 0.010,
    
    # Bayesian inference
    "RUN_MCMC": True,
    "MCMC_NSTEPS": 5000,
    "USE_NESTED_SAMPLING": True,
    
    # ... see Configuration Overview for complete list
}
```

---

## Pipeline Phases

The automatic pipeline (`run_automatic_tqe_darkenergy_pipeline()`) executes the following stages:

### Phase 0: Goldilocks Zone Optimization (Optional)

**If `AUTO_FIND_GOLDILOCKS=True`:**

- Bayesian optimization (Differential Evolution) to find optimal TQE parameters
- Searches for: E_c (critical energy), σ (stability width), α (coupling strength), β₀ (geometric coupling)
- Objective function: Minimizes H(a) deviations while maintaining physical consistency
- Updates `MASTER_CTRL` with optimal parameters for subsequent model runs

**Output:**
- `Goldilocks_Optimal_Parameters.json`
- `Goldilocks_Optimal_Visualization.png`
- Optimization summary in console

### Phase 1: Model Initialization

- **4 base models**: Covariant Pressure, Uniform w, Geometric, Null ΛCDM
- **Optional β₀ sweep**: 21 values from 0.000 to 0.100 (if `RUN_BETA0_SWEEP=True`)
- **Total models**: 4 base × (1 + 21 sweep) = 88 models (if sweep enabled)

### Phase 2: Dual Coupling Mode Execution

**If `COUPLING_MODE='dual'` or `RUN_DUAL_COMPARISON=True`:**

- Runs each model in both E-only and E+I configurations
- Uses same seed for direct comparison
- Total runs: 88 models × 2 modes = 176 runs (if sweep enabled)

### Phase 3: Per-Model Analysis (12 Phases per Run)

Each model run executes the following analysis phases:

#### 3.1: Cosmological Evolution
- Compute H(a), I(a), ρ_DE(a) evolution
- Scale factor grid: log-spaced from a_min to 1.0
- High-resolution integration with adaptive step size

#### 3.2: Field Statistics
- Compute I_mean, I_std for geometric model
- Mean-centered coupling for stability
- Field smoothness metrics

#### 3.3: Evolution Series
- S₈(z) evolution: Structure growth parameter
- D(z) evolution: Linear growth factor
- ρ_DE(z) evolution: Dark energy density

#### 3.4: I-E Correlation Analysis
- Pearson correlation: Linear relationship
- Spearman correlation: Monotonic relationship
- Mutual Information: Non-linear dependence
- Temporal lag scan: Causal coupling detection

#### 3.5: Observable Predictions
- **SNe Ia Hubble diagram**: Distance-redshift relation μ(z)
- **BAO observables**: D_M(z), H(z) at survey redshifts
- **CMB power spectrum**: C_ℓ using CAMB (if available)
- **LSS power spectrum**: P(k) with growth factor

#### 3.6: Galaxy Structure Analysis
- 3D density field generation from P(k)
- Cosmic web classification: Voids, filaments, sheets, clusters
- Structure cataloging with physical properties
- Size filtering and statistical analysis

#### 3.7: Sanity Checks
- Physical consistency validation
- H(a) > 0 for all a
- ρ_DE(a) > 0 for all a
- Ω_total ≈ 1 (flatness constraint)
- D_L/D_A = (1+z)² relationship

#### 3.8: Sensitivity Test
- ±1% I-parameter perturbation
- Measure impact on observables
- Numerical stability assessment

#### 3.9: Visualizations
- 11-16 publication-quality PNG plots per run
- LaTeX-formatted labels and legends
- Unified color scheme across models
- High-resolution output (300 DPI)

#### 3.10: CMB Planck Validation (If Enabled)
- Load real Planck component-separated maps
- Compute power spectrum C_ℓ
- Anomaly detection (cold/hot spots)
- NHI foreground correlation
- Comparison with TQE simulated C_ℓ

#### 3.11: Bayesian Inference (If Enabled)
- MCMC posterior sampling (emcee)
- Nested sampling for evidence (dynesty)
- Corner plots for parameter posteriors
- Bayes Factors for model comparison
- Information criteria (AIC, BIC, DIC)

#### 3.12: Data Saving
- 32-36 files per run (CSV, JSON, TXT, ZIP)
- Reproducibility snapshot (MASTER_CTRL + environment)
- Complete parameter tracking

### Phase 4: Cross-Model Aggregation

#### 4.1: E-Only Aggregator
- Model comparison across E-only runs
- β₀ sweep analysis (if enabled)
- Parameter ranking and selection

#### 4.2: E+I Aggregator
- Model comparison across E+I runs
- β₀ sweep analysis (if enabled)
- Parameter ranking and selection

#### 4.3: Dual Comparison
- E-only vs E+I statistical analysis
- Significance testing (t-tests, Mann-Whitney U)
- Bayes Factor comparison
- Model ranking by information criteria

### Phase 5: Final Summary

- Pipeline metadata and execution time
- Model rankings and recommendations
- Reproducibility snapshot
- Summary report (TXT + JSON)

---

## Output Structure

### Directory Layout

```
TQE_DarkEnergy_Coupling_Simulation_v4.2.0PRO_<timestamp>/
├── 00_Pipeline_Summary/
│   ├── Pipeline_Summary.txt
│   ├── Pipeline_Metadata.json
│   ├── Reproducibility_Info.txt
│   └── MASTER_CTRL_Snapshot.json
│
├── 01_Goldilocks_Optimization/  (if AUTO_FIND_GOLDILOCKS=True)
│   ├── Goldilocks_Optimal_Parameters.json
│   ├── Goldilocks_Optimal_Visualization.png
│   └── Goldilocks_Optimization_Summary.txt
│
├── 02_Model_Runs/
│   ├── Model_1_Covariant_Eonly_<timestamp>/
│   │   ├── Evolution_Data/
│   │   │   ├── hubble_parameter_evolution.csv
│   │   │   ├── i_parameter_evolution.csv
│   │   │   ├── dark_energy_density_evolution.csv
│   │   │   └── evolution_series.csv
│   │   ├── Observables/
│   │   │   ├── sne_ia_hubble_diagram.csv
│   │   │   ├── bao_observables.csv
│   │   │   ├── cmb_power_spectrum.csv
│   │   │   └── likelihood_results.csv
│   │   ├── Analysis/
│   │   │   ├── I_E_correlation.csv
│   │   │   ├── S8_normalization.csv
│   │   │   └── sensitivity_test_results.csv
│   │   ├── Galaxy_Structure/
│   │   │   ├── Galaxy_Cosmic_Web_Summary.json
│   │   │   ├── Galaxy_Void_Catalogue.csv
│   │   │   └── Galaxy_Wall_Catalogue.csv
│   │   ├── Visualizations/
│   │   │   ├── 01_hubble_parameter_evolution.png
│   │   │   ├── 02_i_parameter_evolution.png
│   │   │   ├── 03_dark_energy_density_evolution.png
│   │   │   ├── 04_sne_ia_hubble_diagram.png
│   │   │   ├── 05_bao_observables.png
│   │   │   ├── 06_cmb_power_spectrum.png
│   │   │   ├── 07_S8_evolution.png
│   │   │   ├── 08_growth_factor_evolution.png
│   │   │   ├── 09_I_vs_E_scatter.png
│   │   │   ├── 10_cosmic_web_fractions.png
│   │   │   └── 11_void_size_distribution.png
│   │   ├── Bayesian_Inference/  (if RUN_MCMC=True)
│   │   │   ├── mcmc_samples.csv
│   │   │   ├── nested_sampling_samples.csv
│   │   │   ├── corner_plot.png
│   │   │   └── bayesian_summary.json
│   │   ├── CMB_Planck_Validation/  (if USE_REAL_CMB_PLANCK_MAPS=True)
│   │   │   ├── planck_power_spectrum.csv
│   │   │   ├── tqe_power_spectrum.csv
│   │   │   ├── power_spectrum_comparison.png
│   │   │   └── anomaly_catalog.csv
│   │   ├── Model_Summary.json
│   │   ├── Full_Summary.txt
│   │   └── TQE_DarkEnergy_Complete_Results.zip
│   │
│   ├── Model_1_Covariant_EplusI_<timestamp>/
│   │   └── ... (same structure as E-only)
│   │
│   ├── Model_2_Uniform_w_Eonly_<timestamp>/
│   │   └── ... (same structure)
│   │
│   └── ... (all model runs)
│
├── 03_Cross_Model_Analysis/
│   ├── Eonly_Aggregator/
│   │   ├── model_comparison.csv
│   │   ├── beta0_sweep_analysis.csv  (if RUN_BETA0_SWEEP=True)
│   │   └── model_ranking.json
│   ├── EplusI_Aggregator/
│   │   ├── model_comparison.csv
│   │   ├── beta0_sweep_analysis.csv
│   │   └── model_ranking.json
│   └── Dual_Comparison/
│       ├── eonly_vs_eplusi_comparison.csv
│       ├── bayes_factor_comparison.csv
│       ├── statistical_significance_tests.csv
│       └── eonly_vs_eplusi_dashboard.png
│
└── 04_Reproducibility/
    ├── MASTER_CTRL_Final.json
    ├── Environment_Info.txt
    └── Package_Versions.json
```

### File Naming Conventions

Files are automatically prefixed based on coupling mode (if `AUTO_PREFIX_FILES=True`):
- **E-only mode**: `Eonly_<filename>`
- **E+I mode**: `EplusI_<filename>`

Example:
- `Eonly_hubble_parameter_evolution.csv`
- `EplusI_hubble_parameter_evolution.csv`

### Output File Types

#### CSV Files
- **Evolution data**: H(a), I(a), ρ_DE(a), S₈(z), D(z)
- **Observables**: SNe Ia, BAO, CMB, LSS predictions
- **Analysis**: I-E correlations, sensitivity tests, likelihood results
- **Galaxy structure**: Void/wall/filament catalogs
- **Bayesian inference**: MCMC/nested sampling samples

#### JSON Files
- **Model summaries**: Complete parameter sets and results
- **Galaxy structure**: Cosmic web classification summaries
- **Bayesian inference**: Posterior summaries and evidence
- **Reproducibility**: MASTER_CTRL snapshots

#### PNG Files
- **Publication-quality plots**: 300 DPI, LaTeX formatting
- **Corner plots**: Parameter posteriors (if Bayesian inference enabled)
- **Dashboard plots**: Multi-panel comparisons

#### ZIP Archives
- **Complete results**: All files for a model run compressed
- **Reproducibility bundles**: Complete run snapshots

---

## Key Classes and Functions

### Core Classes

#### `EnergyInformationContent`
Computes the I-parameter (information content) from energy evolution.

**Key Methods:**
- `compute_information(a, E, dE_da)`: Compute I(a) from energy evolution
- `compute_information_derivative(a, E, dE_da)`: Compute dI/da
- `compute_information_gradient_squared(a)`: Compute |∇I|² (smoothness penalty)
- `I_from_KL_divergence(P_t, P_t_plus_1)`: Alternative I definition (KL divergence)
- `I_from_Shannon_entropy(P_t)`: Alternative I definition (Shannon entropy)

#### `CouplingModel`
Implements dark energy coupling mechanisms.

**Key Methods:**
- `rho_DE(a, rho_Lambda, friedmann)`: Compute dark energy density
- `w_DE(a)`: Compute dark energy equation of state
- `compute_G_field(a, H, H0)`: Compute combined E-I field: G(a) = w_E·(E-1) + w_I·(I-⟨I⟩)

**Coupling Types:**
- `"covariant_pressure"`: Model 1 (exponential E-I coupling)
- `"uniform_w"`: Model 2 (I-dependent equation of state)
- `"geometric"`: Model 3 (geometric functional of I and derivatives)
- `"null"`: Model 4 (pure ΛCDM)

#### `FriedmannEvolution`
Solves Friedmann equations with dark energy coupling.

**Key Methods:**
- `H(a)`: Hubble parameter H(a) [km/s/Mpc]
- `E(a)`: Dimensionless Hubble parameter E(a) = H(a)/H₀
- `comoving_distance(z)`: Comoving distance D_C(z)
- `luminosity_distance(z)`: Luminosity distance D_L(z)
- `angular_diameter_distance(z)`: Angular diameter distance D_A(z)
- `growth_factor(z)`: Linear growth factor D(z) (ODE solver)

#### `TQEDarkEnergyCouplingSimulation`
Main simulation class orchestrating the complete analysis.

**Key Methods:**
- `run_cosmological_evolution(a_min, a_max, n_points)`: Run cosmological evolution
- `compute_observables()`: Compute all observable predictions
- `compute_evolution_series()`: Compute S₈(z), D(z), ρ_DE(z) evolution
- `compute_I_E_correlation()`: I-E correlation analysis
- `run_sanity_checks()`: Physical consistency validation
- `visualize_results(save_plots=True)`: Generate all visualizations
- `save_results()`: Save all output files

#### `ObservablePredictions`
Computes cosmological observables for model comparison.

**Key Methods:**
- `sne_hubble_diagram(z_array)`: SNe Ia distance-redshift relation
- `bao_observables(z_array)`: BAO measurements D_M(z), H(z)
- `cmb_power_spectrum(use_camb=True)`: CMB power spectrum C_ℓ
- `matter_power_spectrum(k_array, z=0)`: Matter power spectrum P(k)
- `S8_parameter(z=0.0)`: Structure growth parameter S₈
- `compute_likelihood()`: Likelihood from SNe, BAO, H₀ prior

#### `BayesianInferenceEngine`
Bayesian parameter estimation with MCMC/Nested Sampling.

**Key Methods:**
- `run_mcmc(n_walkers, n_steps, n_burn)`: MCMC posterior sampling
- `run_nested_sampling(nlive, dlogz)`: Nested sampling for evidence
- `make_corner_plot(save_path)`: Corner plot of posteriors
- `compute_ic()`: Information criteria (AIC, BIC, DIC)
- `compute_bayes_factor(logz_reference)`: Bayes Factor for model comparison

#### `GalaxyStructureAnalyzer`
3D cosmological structure formation analysis.

**Key Methods:**
- `generate_density_field()`: Generate 3D density field from P(k)
- `classify_cosmic_web()`: Classify voids/filaments/sheets/clusters
- `find_voids()`: Find void regions
- `find_clusters()`: Find cluster regions
- `find_filaments()`: Find filament structures
- `find_walls()`: Find wall/sheet structures
- `compute_all_metrics()`: Compute all structure metrics

#### `CMBPlanckValidation`
Real Planck CMB map validation and comparison.

**Key Methods:**
- `compute_planck_power_spectrum()`: Compute C_ℓ from Planck maps
- `compute_tqe_power_spectrum()`: Compute C_ℓ from TQE simulation
- `compare_power_spectra()`: Compare Planck vs TQE
- `detect_anomalies(skymap, threshold)`: Detect cold/hot spots
- `correlate_with_nhi(skymap)`: Correlate with NHI foreground
- `generate_validation_plots(output_dir)`: Generate validation plots

### Utility Functions

#### Data Loading
- `load_pantheon_plus_data(filepath, cov_filepath)`: Load Pantheon+ SNe Ia data
- `load_boss_bao_data(filepath, cov_filepath)`: Load BOSS/eBOSS/DESI BAO data
- `load_planck_cmb_data(filepath, cov_filepath)`: Load Planck CMB data

#### Pipeline Functions
- `run_automatic_tqe_darkenergy_pipeline()`: Main automatic pipeline
- `find_goldilocks_zone_bayesian(run_dir)`: Goldilocks optimization
- `compare_eonly_vs_eplusi(all_results, run_dir)`: E-only vs E+I comparison
- `compute_bayes_factors_all_models(all_results)`: Bayes Factor computation
- `run_integrated_aggregator(run_dir)`: Cross-model aggregation
- `run_unit_tests(friedmann)`: Unit tests for ΛCDM compatibility

#### Reproducibility
- `set_deterministic_seed(seed_string)`: Set deterministic seed
- `save_reproducibility_snapshot(run_dir)`: Save complete reproducibility info

---

## Interpretation Guide

### Key Metrics to Monitor

#### 1. Hubble Parameter Evolution
- **H(a) stability**: Should be smooth, positive, and finite for all a
- **H(a=1) deviation**: Compare E-only vs E+I modes
- **Early universe behavior**: Check for instabilities at small a

#### 2. Dark Energy Density Evolution
- **ρ_DE(a) positivity**: Must be positive for all a
- **ρ_DE(a=1) normalization**: Should match observed Ω_Λ
- **Evolution rate**: Compare E-only vs E+I modes

#### 3. I-E Correlation
- **Pearson correlation**: Linear relationship strength
- **Spearman correlation**: Monotonic relationship strength
- **Mutual Information**: Non-linear dependence
- **Temporal lag**: Causal coupling detection (lag ≠ 0 indicates coupling)

#### 4. Observable Predictions
- **SNe Ia Hubble diagram**: Distance-redshift relation μ(z)
- **BAO observables**: D_M(z), H(z) at survey redshifts
- **CMB power spectrum**: C_ℓ comparison with Planck
- **S₈ parameter**: Structure growth (key discriminator)

#### 5. Bayesian Inference Results
- **Posterior distributions**: Parameter constraints
- **Bayes Factors**: Model comparison (BF > 10: strong evidence)
- **Information criteria**: AIC, BIC, DIC (lower is better)
- **Evidence log Z**: Bayesian evidence (higher is better)

#### 6. Galaxy Structure Analysis
- **Cosmic web fractions**: Voids, filaments, sheets, clusters
- **Void size distribution**: Compare with observations
- **Structure catalog**: Physical properties and statistics

### Success Criteria

#### For TQE Hypothesis Validation:
1. **E+I mode shows different observables than E-only**: S₈, CMB, BAO differ significantly
2. **I-E correlation shows temporal lag**: Indicates causal coupling (not just correlation)
3. **Bayes Factor favors E+I over E-only**: BF > 10 (strong evidence)
4. **Information criteria favor E+I**: Lower AIC/BIC/DIC for E+I mode
5. **CMB anomalies show non-random patterns**: Consistent with TQE predictions

#### For Physical Consistency:
1. **All sanity checks pass**: H(a) > 0, ρ_DE(a) > 0, Ω_total ≈ 1
2. **Smooth evolution**: No discontinuities or instabilities
3. **Observable predictions match data**: χ²/dof < 2 for good fits
4. **Sensitivity test shows stability**: ±1% I-perturbation → small observable changes

### Troubleshooting Common Issues

#### Issue: H(a) becomes negative or infinite
**Cause**: Coupling strength too large, instability in early universe  
**Solution**: Reduce `ALPHA` or `BETA0`, increase `I_FIELD_MAX_DELTA_LN_RHO` constraint

#### Issue: ρ_DE(a) becomes negative
**Cause**: Exponential coupling with wrong sign, instability  
**Solution**: Check coupling formula sign, reduce coupling strength

#### Issue: Bayesian inference fails to converge
**Cause**: Prior ranges too wide, likelihood too flat  
**Solution**: Tighten prior ranges, increase `NESTED_NLIVE`, check likelihood computation

#### Issue: CMB Planck validation fails
**Cause**: Missing Planck map files, incorrect path  
**Solution**: Check `CMB_PLANCK_BASE_PATH`, ensure maps are in Google Drive

#### Issue: Memory errors during large runs
**Cause**: Too many models, high resolution grids  
**Solution**: Enable `MEMORY_EFFICIENT_MODE`, reduce `A_GRID_N_LOG`, run models sequentially

---

## Advanced Usage

### Custom I-Parameter Definitions

The pipeline supports multiple I-parameter definitions. Modify `EnergyInformationContent.compute_information()`:

```python
# Energy-based (default, TQE-compliant)
I = |dE/da| / (E + |dE/da|)

# KL divergence-based
I = D_KL(P_t || P_{t+1}) / (1 + D_KL(P_t || P_{t+1}))

# Shannon entropy-based
I = -Σ P_t · log(P_t)  # normalized

# Composite fusion
I = f(I_KL, I_Shannon)  # product, sum, or weighted combination
```

### Custom Coupling Models

Add new coupling models by extending `CouplingModel`:

```python
class CustomCouplingModel(CouplingModel):
    def rho_DE(self, a, rho_Lambda, friedmann=None):
        # Your custom coupling formula
        I = self.information_content.compute_information(a, ...)
        E = friedmann.E(a)
        return rho_Lambda * custom_function(E, I)
```

### Parameter Sweeps

Enable automatic parameter sweeps:

```python
MASTER_CTRL = {
    "RUN_BETA0_SWEEP": True,
    "BETA0_RANGE": [0.0, 0.01, 0.02, 0.03, 0.05, 0.10],
    
    # Or custom sweep
    "RUN_ALPHA_SWEEP": True,
    "ALPHA_RANGE": [0.0, 0.01, 0.02, 0.03, 0.05],
}
```

### Cross-Validation

Enable cross-validation for model selection:

```python
MASTER_CTRL = {
    "USE_CROSS_VALIDATION": True,
    "TRAIN_TEST_SPLIT": 0.7,
    "N_CV_FOLDS": 5,
    "TRAIN_ON_CMB": True,
    "TRAIN_ON_SNE": True,
}
```

### Ablation Studies

Test individual coupling components:

```python
MASTER_CTRL = {
    "RUN_ABLATION": True,
    "TEST_I_ONLY": True,           # Test ⟨I⟩ only
    "TEST_GRAD_I_ONLY": True,      # Test |∇I|² only
    "TEST_TIME_DERIV_I_ONLY": True,  # Test (∂_t I)² only
}
```

---

## Performance Optimization

### Recommended Settings for Different Use Cases

#### Quick Testing (Fast Mode)
```python
MASTER_CTRL = {
    "PERFORMANCE_MODE": "fast",
    "A_GRID_N_LOG": 1024,
    "Z_POINTS": 50,
    "MCMC_NSTEPS": 1000,
    "NESTED_NLIVE": 100,
}
```
**Runtime**: ~5-10 minutes per model  
**Use case**: Quick parameter exploration, debugging

#### Production Runs (Balanced Mode)
```python
MASTER_CTRL = {
    "PERFORMANCE_MODE": "balanced",
    "A_GRID_N_LOG": 2048,
    "Z_POINTS": 100,
    "MCMC_NSTEPS": 5000,
    "NESTED_NLIVE": 250,
}
```
**Runtime**: ~30-60 minutes per model  
**Use case**: Publication-quality results, model comparison

#### High-Precision Analysis (Accurate Mode)
```python
MASTER_CTRL = {
    "PERFORMANCE_MODE": "accurate",
    "A_GRID_N_LOG": 8192,
    "Z_POINTS": 200,
    "MCMC_NSTEPS": 10000,
    "NESTED_NLIVE": 500,
}
```
**Runtime**: ~2-4 hours per model  
**Use case**: Final validation, parameter estimation

### Memory Optimization

Enable memory-efficient mode:

```python
MASTER_CTRL = {
    "MEMORY_EFFICIENT_MODE": True,
    "ENABLE_CACHING": True,
    "CACHE_SIZE": 500,  # Reduce if memory constrained
}
```

### Parallel Execution

For large parameter sweeps, run models in parallel:

```python
# Run models sequentially (default)
# Or use multiprocessing for parallel execution
from multiprocessing import Pool

def run_model(model_config):
    # Initialize and run model
    pass

with Pool(processes=4) as pool:
    results = pool.map(run_model, model_configs)
```

---

## Citation & Attribution

If you use this software in your research, please cite:

**Plain Text Citation:**
> Stefan Len. (2025). *TQE Dark Energy Coupling Simulation v4.2.0 PRO* (Version v4.2.0\_Pro) [Software]. Zenodo. https://doi.org/10.5281/zenodo.17614948

**BibTeX Entry:**

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

**Author ORCID:** https://orcid.org/0009-0007-0383-7315

---

## License

This project is licensed under the MIT License – see the [LICENSE](../../LICENSE) file in the repository root for details.

---

## Support & Contact

For questions, collaborations, or feedback:

- **Email**: stefan@tqe-theory.space
- **GitHub Issues**: [File an issue](https://github.com/SteviLen420/TQE_simulation/issues)
- **Repository**: https://github.com/SteviLen420/TQE_simulation

---

## Acknowledgments

This pipeline builds upon the theoretical framework of the Theory of the Question of Existence (TQE) and integrates with:

- **CAMB**: Code for Anisotropies in the Microwave Background (Lewis & Challinor 2002)
- **emcee**: MCMC Hammer (Foreman-Mackey et al. 2013)
- **dynesty**: Dynamic Nested Sampling (Speagle 2020)
- **healpy**: HEALPix Python bindings (Górski et al. 2005)
- **Planck Collaboration**: CMB data and maps (Planck 2018)
- **Pantheon+ Collaboration**: SNe Ia data (Scolnic et al. 2022)
- **BOSS/eBOSS/DESI Collaborations**: BAO data

---

## Changelog

### v4.2.0 PRO (2025-10-29)
- **CRITICAL UPDATE**: 16 bug fixes, TQE-compliant I-parameter
- Fixed E-only vs E+I distinction (was identical before)
- Implemented energy-based I-parameter: I = |dE/da| / (E + |dE/da|)
- Added exponential coupling formula: ρ_DE = ρ_Λ·exp(β₀I - αE)
- Optimized performance: 2× faster with balanced mode
- Enhanced Bayesian inference: Nested sampling with optimized parameters
- Added CMB Planck validation: Real map loading and comparison
- Improved galaxy structure analysis: 3D cosmic web classification
- Added Goldilocks optimization: Automatic parameter search
- Enhanced reproducibility: Complete environment tracking

---

## Related Documentation

- **Main TQE Repository README**: [README.md](../../README.md)
- **TQE Research Notes**: [TQE_Research_Notes.md](../../TQE_Research_Notes.md)
- **TQE Foundational Laws**: [TQE_FOUNDATIONAL_LAWS_OF_THE_UNIVERSE.md](../../TQE_FOUNDATIONAL_LAWS_OF_THE_UNIVERSE.md)
- **TQE Cycle Model**: [TQE_CYCLE_MODEL.md](../../TQE_CYCLE_MODEL.md)
- **AI Methodology**: [AI_METHODOLOGY.md](../../AI_METHODOLOGY.md)

---

*This is an active research project. Contributions, feedback, and scientific collaboration are welcome.*

