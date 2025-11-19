SPDX-License-Identifier: MIT

Copyright (c) 2025 Stefan Len

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17627756.svg)](https://doi.org/10.5281/zenodo.17627756)
[![Execution](https://img.shields.io/badge/runtime-local__desktop-orange)](#)
[![SIMULATION_RUNS](https://img.shields.io/badge/input-SIMULATION__RUNS%2Funiverse-informational)](#)
[![Status](https://img.shields.io/badge/status-active%20development-green)](#)

# TQE ANALYSIS PIPELINE v4.2.0 PRO (MODULAR)

**Title:** TQE Comparative Analysis Pipeline: Batch Simulation Results Analysis and Model Selection (Modular)  
**Author:** Stefan Len  
**Version:** v4.2.0 PRO (Modular build)

---

## 0. Overview

This directory contains the **modularized** version of the TQE Analysis Pipeline.  
`TQE_Analysis_Pipeline_v4.2.0_PRO.py` (monolithic script, 3956 lines) has been refactored into reusable modules for clean separation of concerns, easier testing, and better maintainability. The modular build keeps the scientific content identical while exposing clear APIs for:

- Automatically discovering and analyzing batch simulation results (`batch_all` or `batch_ei`)
- Extracting comprehensive metrics (50-80 columns per run)
- Performing 12+ comparative analysis categories
- Generating triple ranking reports (stability, complexity, physical-laws)
- Producing publication-quality visualizations and recommendations
- Local execution with Desktop output (no Colab dependency)

**Key Features:**
- ✅ **Local execution only** (no Colab support)
- ✅ **Desktop output**: Results saved to `Desktop/TQE_Analysis_Modular_Results/`
- ✅ **Modular structure**: Clean separation of concerns (data loading, analysis, visualization)
- ✅ **Comprehensive analysis**: 12+ analysis categories, triple ranking system
- ✅ **Extended metrics**: 50-80 columns per run (stability, complexity, life-compatibility, etc.)
- ✅ **Auto-discovery**: Automatically finds latest batch runs or specific timestamps
- ✅ **Triple ranking**: Three ranking perspectives (stability, complexity, physical-laws)

---

## 1. Directory Layout

```
TQE_Analysis_Modular/
├── __init__.py
├── config.py                    # MASTER_CTRL configuration
├── main.py                      # Main orchestrator (8-phase pipeline)
├── README.md                    # This file
├── core/
│   ├── __init__.py
│   ├── path_setup.py           # Path setup, file finding, directory discovery
│   └── data_collector.py       # Data collection orchestrator
├── data_loaders/
│   ├── __init__.py
│   ├── summary_loader.py       # Summary JSON and TQE runs CSV
│   ├── bayesian_loader.py      # Bayesian calibration data
│   ├── cmb_loader.py           # CMB anomaly data (cold spots, Axis of Evil)
│   └── metrics_loader.py       # Extended metrics loaders (14+ functions)
├── analysis/
│   ├── __init__.py
│   ├── metrics_extractor.py    # Metrics extraction and DataFrame building
│   ├── model_selector.py       # Triple ranking system (stability/complexity/physical-laws)
│   ├── comparative.py          # E+I comparison, E-only vs E+I
│   └── specialized.py          # 14 specialized analysis functions (stubs - to be implemented)
├── visualization/
│   ├── __init__.py
│   ├── detailed_metrics.py     # Detailed metrics, correlation matrix, distributions
│   ├── advanced_plots.py       # Radar charts, heatmaps, complexity analysis
│   └── reports.py              # Extended report generation
└── utils/
    ├── __init__.py
    ├── setup.py                # Package installation utilities
    └── helpers.py              # Helper functions (I-definition extraction, etc.)
```

Every module can be imported independently (e.g., `from TQE_Analysis_Modular.analysis.model_selector import select_best_model`) enabling targeted notebooks or tests without executing the entire pipeline.

---

## 2. Module Guide

| Module | Purpose / Key Functions |
| --- | --- |
| `config.py` | Central `MASTER_CTRL` dictionary: target mode, ranking weights, visualization settings, statistical options, Planck targets. |
| `core/path_setup.py` | `setup_paths()`: Desktop output path setup. `smart_find_file()`: Universal file finder. `find_latest_mode_directory()`: Auto-discovery of batch runs. `validate_target_mode()`, `detect_eonly_presence()`, `collect_run_directories()`. |
| `core/data_collector.py` | `collect_simulation_data()`: Comprehensive data collection orchestrator. Loads all simulation artifacts (summary JSON, CSV files, Planck validation, life compatibility, etc.) from discovered runs. |
| `data_loaders/summary_loader.py` | `load_summary_json()`, `load_tqe_runs_csv()`: Load core simulation outputs. |
| `data_loaders/bayesian_loader.py` | `load_bayesian_calibration_csv()`: Load Bayesian Goldilocks optimization data. |
| `data_loaders/cmb_loader.py` | `load_cmb_coldspots()`, `load_cmb_aoe()`: Load CMB anomaly detection results. |
| `data_loaders/metrics_loader.py` | 10+ loader functions: emergent laws, parameter sensitivity, I-definitions comparison, life compatibility, Planck validation, entropy volatility, stability sweeps, advanced anomalies, nested sampling, pre-fluctuation pairs, universe seeds. |
| `analysis/metrics_extractor.py` | `extract_extended_metrics()`: Extract 30-50 additional metrics from CSV files. `extract_metrics_from_summary()`: Extract basic metrics from summary JSON. `build_metrics_dataframe()`: Build comprehensive DataFrame (50-80 columns). |
| `analysis/model_selector.py` | `select_best_model()`: Triple ranking system. Generates stability-focused, complexity-focused, and physical-laws-focused rankings. Produces recommendation report with usage guidance. |
| `analysis/comparative.py` | `compare_ei_definitions()`: Compare all E+I definitions (stability rates, Goldilocks zones, Planck fit). `compare_eonly_vs_ei()`: Analyze improvements from adding I-coupling. |
| `analysis/specialized.py` | 14 specialized analysis functions: `analyze_emergent_laws()`, `analyze_friedmann_cosmology()`, `analyze_cmb_anomalies()`, `analyze_lockin_dynamics()`, `analyze_quantum_fields()`, `analyze_entanglement()`, `analyze_parameter_sensitivity()`, `analyze_topology()`, `analyze_i_definitions_direct()`, `analyze_planck_fit()`, `analyze_life_top_universes()`, `analyze_entropy_volatility()`, `analyze_physical_anomalies()`, `analyze_statistical_finetuning()`. **Note:** Currently stubs - full implementation pending. |
| `visualization/detailed_metrics.py` | `generate_detailed_metrics()`: Comprehensive metrics table, correlation matrix, distribution box plots. |
| `visualization/advanced_plots.py` | `generate_advanced_visualizations()`: Radar charts, performance heatmaps, scatter plots. `generate_complexity_analysis()`: Complexity vs stability, life-compatibility comparison. |
| `visualization/reports.py` | `generate_extended_reports()`: Extended markdown report summarizing key findings. |
| `utils/setup.py` | `check_and_install_packages()`: Automatic dependency installation. |
| `utils/helpers.py` | `extract_i_definition()`: Extract I-definition name from directory names. |

---

## 3. Theoretical Foundation

### Core Purpose

The TQE Analysis Pipeline provides comprehensive comparative analysis for batch TQE simulations, identifying the best-performing I-parameter definition across multiple analysis categories. It automatically discovers simulation runs, harvests all artifacts, and produces:

- **Extended metrics table** (≈80 columns) covering stability, Friedmann cosmology, Planck proximity, life compatibility, entropy volatility, sweeps, anomalies, nested sampling, and top-universe metadata
- **12+ comparative analysis categories** with PNG visualizations + CSV exports
- **Triple model rankings** (stability, complexity, physical-laws realism) with recommendation report
- **Clean summary + extended/validation bundle** ready for publication or further machine learning exploration

### Triple Ranking System

The pipeline uses **three ranking perspectives** to identify the best I-definition:

1. **Stability-Focused (Traditional)**: Maximizes stable universe percentage
   - Weights: Stability rate (30%), Lock-in rate (20%), Planck χ² fit (20%), Goldilocks precision (15%), CMB anomaly match (10%), Bayesian efficiency (5%)

2. **Complexity-Focused (TQE-Consistent)**: Maximizes structural complexity and life-compatibility
   - Weights: Complexity score (35%), Life-compatibility (25%), Information richness (20%), Stability quality (10%), Observational match (10%)

3. **Physical-Laws-Focused (Observational Realism)**: Maximizes observational realism (Planck, CMB, emergent laws)
   - Weights: Emergent laws quality (30%), Friedmann consistency (25%), CMB anomaly match (20%), Lock-in efficiency (15%), Quantum field realism (10%)

**Critical Insight:** Different goals require different I-definitions!
- For **observational validation**: Use Physical-Laws-Focused winner
- For **TQE theory validation**: Use Complexity-Focused winner
- For **maximum stability**: Use Stability-Focused winner

### Metrics Extracted

**Basic Metrics:**
- `stable_percent`: Percentage of stable universes
- `lockin_percent`: Percentage with law lock-in
- `X_peak`: Goldilocks zone peak position
- `X_peak_uncertainty`: Peak uncertainty
- `chi_squared_reduced`: Planck fit quality

**Advanced Metrics (50+ additional):**
- `complexity_score`: Structural complexity (0-100)
- `life_compatibility_score`: Life-readiness (0-100)
- `information_richness`: I-parameter effectiveness (0-100)
- `power_law_exponent`: Emergent law scaling
- `age_deviation_from_planck`: Cosmological consistency
- `entanglement_entropy_mean`: Quantum entanglement
- `lockin_efficiency`: Fast, decisive law formation
- `vacuum_energy_mean`: Quantum field properties
- And 50+ more...

---

## 4. Requirements & Environment

- **Python 3.9+** (tested on CPython 3.9–3.11)
- **Local execution only** (no Colab support)
- **Core dependencies**: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `tqdm`
- `utils.setup.check_and_install_packages()` auto-installs missing packages
- **Access to simulation results**: Results must exist in `SIMULATION_RUNS/universe/` directory

> **Note:** The pipeline automatically installs required packages on first run. Manual installation: `pip install pandas numpy matplotlib seaborn scipy tqdm`

---

## 5. Quickstart

### As a module

```python
from TQE_Analysis_Modular.main import run_analysis_pipeline

# Run with default settings (auto-detect latest batch_all)
run_analysis_pipeline()
```

### Custom configuration

```python
from TQE_Analysis_Modular.main import run_analysis_pipeline

# Override configuration
config_override = {
    "TARGET_MODE": "batch_all",
    "TARGET_TIMESTAMP": "20251118_204912",  # Specific timestamp
    "RANKING_MODE": "triple",
    "FIGURE_DPI": 300,
}

run_analysis_pipeline(config_override=config_override)
```

### Command line

```bash
cd TQE_Universe_Simulation_v4.2.0_Pro/TQE_Analysis_Modular
python -m main
```

---

## 6. Output Structure

All results are saved to **Desktop**:

```
Desktop/TQE_Analysis_Modular_Results/
└── {mode}_{timestamp}_analysis/
    ├── 00_summary/
    │   ├── analysis_summary.txt          # Execution summary
    │   ├── run_info.json                 # Run metadata
    │   ├── extended_report.md            # Extended findings report
    │   └── validation_report.txt         # Validation status
    ├── 01_comparative_analysis/
    │   ├── basic_metrics/                # E+I definitions comparison
    │   │   ├── stability_rates_comparison.png
    │   │   ├── goldilocks_zones_comparison.png
    │   │   ├── planck_chi2_comparison.png
    │   │   └── ei_ranking_table.csv
    │   ├── eonly_vs_ei/                  # E-only vs E+I comparison
    │   │   ├── stability_improvement.png
    │   │   ├── goldilocks_shift.png
    │   │   └── eonly_vs_ei_metrics.json
    │   ├── emergent_laws/                # Power-laws, phase transitions
    │   ├── friedmann_cosmology/          # Age, H0, Omegas
    │   ├── cmb_anomalies/                # Cold spots, Axis of Evil
    │   ├── lockin_dynamics/              # Lock-in efficiency
    │   ├── quantum_fields/               # Vacuum energy, fluctuations
    │   ├── entanglement/                 # Entanglement entropy
    │   ├── parameter_sensitivity/        # E/I/X sensitivity
    │   ├── topology/                     # Curvature, defects
    │   ├── i_definitions_direct/         # I(E) curves
    │   ├── planck_fit/                   # Planck validation
    │   ├── life_top_universes/           # Life-compatible universes
    │   ├── entropy_volatility/           # Entropy patterns
    │   └── physical_anomalies/           # Physical anomaly detection
    ├── 02_detailed_metrics/
    │   ├── all_runs_metrics.csv          # ⭐ Comprehensive metrics table (50-80 columns)
    │   ├── correlation_matrix.png        # Metric correlations
    │   └── distributions_boxplot.png     # Distribution analysis
    ├── 03_visualizations/
    │   ├── radar_chart_top5.png          # Multi-metric radar chart
    │   ├── heatmap_performance.png       # Performance heatmap
    │   ├── scatter_X_peak_vs_stability.png
    │   └── complexity/
    │       ├── complexity_vs_stability.png
    │       └── life_compatibility_comparison.png
    ├── 04_best_model_selection/
    │   ├── weighted_ranking.csv          # All rankings merged
    │   ├── ranking_stability_focused.csv
    │   ├── ranking_complexity_focused.csv
    │   ├── ranking_physical_laws_focused.csv
    │   ├── top_3_models_triple.json
    │   └── recommendation_report.md      # ⭐ READ THIS FIRST
    └── 05_raw_data/
        ├── collected_data.pkl            # Raw collected data
        └── extended_metrics.pkl          # Extended metrics DataFrame
```

---

## 7. Configuration & Execution Modes

All configuration lives in `config.py` (`MASTER_CTRL` dictionary). The most-used settings:

| Setting | Default | Description |
| --- | --- | --- |
| `TARGET_MODE` | `"batch_all"` | `"batch_all"` (E-only + E+I) or `"batch_ei"` (E+I only) |
| `TARGET_TIMESTAMP` | `None` | Auto-detect latest batch, or specify `"YYYYMMDD_HHMMSS"` |
| `RANKING_MODE` | `"triple"` | `"triple"`, `"stability"`, `"complexity"`, or `"physical_laws"` |
| `RANKING_WEIGHTS_STABILITY` | `{...}` | Weights for stability-focused ranking |
| `RANKING_WEIGHTS_COMPLEXITY` | `{...}` | Weights for complexity-focused ranking |
| `RANKING_WEIGHTS_PHYSICAL_LAWS` | `{...}` | Weights for physical-laws-focused ranking |
| `FIGURE_DPI` | `300` | Plot resolution (300 for publication quality) |
| `TOP_N_MODELS` | `3` | Number of top models to highlight |
| `PLANCK_TARGET_E` | `0.7619` | Reference E value (Planck 2018) |
| `PLANCK_TARGET_I` | `0.1309` | Reference I value (Planck 2018) |

### Execution Modes

The pipeline analyzes results from two batch modes:

1. **`batch_all`**: E-only baseline + all 11 I-definitions (12 total runs)
   - Provides comprehensive comparison including baseline
   - Enables E-only vs E+I improvement analysis

2. **`batch_ei`**: All 11 I-definitions only (11 total runs)
   - Faster analysis (no E-only comparison)
   - Focuses on I-definition comparison

---

## 8. Pipeline Flow (8 Phases)

1. **Phase 1: Comprehensive Data Collection**
   - Auto-discover target batch directory (latest or specified timestamp)
   - Load all simulation artifacts: summary JSON, TQE runs CSV, Bayesian calibration, emergent laws, parameter sensitivity, CMB anomalies, Planck validation, life compatibility, entropy volatility, stability sweeps, advanced anomalies, nested sampling, I-definitions comparison, etc.

2. **Phase 2: Extended Metric Extraction**
   - Extract basic metrics from summary JSON (stability, Goldilocks, Bayesian)
   - Extract extended metrics from CSV files (Friedmann, quantum fields, entanglement, entropy, topology, etc.)
   - Build comprehensive DataFrame (50-80 columns per run)

3. **Phase 3: Comparative Analysis**
   - **3A**: Basic metrics comparison (E+I definitions)
   - **3B-K**: Extended category analyses (14 specialized analyses)
   - **3L**: E-only vs E+I comparison (if available)
   - Detailed metrics generation

4. **Phase 4: Advanced Visualizations**
   - Radar charts (multi-metric comparison)
   - Performance heatmaps
   - Scatter plots (X_peak vs stability)
   - Complexity analysis (complexity vs stability, life-compatibility)

5. **Phase 5: Triple Model Ranking**
   - Stability-focused ranking
   - Complexity-focused ranking
   - Physical-laws-focused ranking
   - Recommendation report generation

6. **Phase 6: Extended Reports**
   - Extended markdown report summarizing key findings
   - Highlighted I-definitions
   - Artifact coverage summary

7. **Phase 7: Summary Export**
   - Execution summary (text)
   - Run metadata (JSON)
   - Output structure documentation

8. **Phase 8: Validation & QC**
   - Validate critical artifacts exist
   - Generate validation report

---

## 9. Key Results & Interpretation

### Triple Ranking Results

The pipeline produces three rankings, each optimized for different goals:

**Stability-Focused Winner:**
- Best for: Maximum stable universe percentage
- Use case: When stability is the primary concern

**Complexity-Focused Winner:**
- Best for: TQE theory validation
- Use case: When structural complexity and life-compatibility matter most
- Typically: `kl_shannon` or `entanglement` (based on analysis results)

**Physical-Laws-Focused Winner:**
- Best for: Observational validation
- Use case: When matching Planck 2018 cosmology is critical
- Typically: `kl_shannon` or `jensen_shannon` (best Planck fit)

### Recommendation Report

The `recommendation_report.md` file provides:
- Detailed methodology for each ranking system
- Top 3 models for each ranking
- Usage recommendations with code examples
- Scientific justification

**Example recommendation:**
```python
# For Planck-consistent simulations:
I_DEFINITION_MODE = "kl_shannon"

# For TQE complexity studies:
I_DEFINITION_MODE = "entanglement"
```

---

## 10. Working With Individual Modules

- **Custom analysis**: Import specific analysis functions:
  ```python
  from TQE_Analysis_Modular.analysis.comparative import compare_ei_definitions
  from TQE_Analysis_Modular.analysis.model_selector import select_best_model
  ```

- **Data loading**: Use loaders independently:
  ```python
  from TQE_Analysis_Modular.data_loaders.summary_loader import load_summary_json
  summary = load_summary_json("/path/to/run")
  ```

- **Metrics extraction**: Extract metrics from collected data:
  ```python
  from TQE_Analysis_Modular.analysis.metrics_extractor import extract_extended_metrics
  extended = extract_extended_metrics(data_dict, "kl_shannon")
  ```

- **Notebook studies**: Import visualization functions for custom plots:
  ```python
  from TQE_Analysis_Modular.visualization.advanced_plots import generate_advanced_visualizations
  ```

---

## 11. Differences from Original Pipeline

1. **Modular structure**: Code split into logical modules (20+ files vs 1 monolithic file)
2. **No Colab support**: Local execution only (removed all Colab/Google Drive code)
3. **Desktop output**: Results saved to Desktop instead of `SIMULATION_RUNS/analysis`
4. **Config-driven**: All settings in `config.py` (MASTER_CTRL)
5. **Stub functions**: Some specialized analysis functions are placeholders (to be implemented from original)
6. **Clean imports**: Each module can be imported independently
7. **Better organization**: Clear separation of data loading, analysis, and visualization

---

## 12. Implementation Status

✅ **Fully Implemented:**
- Data collection and loading (all loaders)
- Metrics extraction (basic + extended)
- Comparative analysis (E+I, E-only vs E+I)
- Model selection (triple ranking system)
- Advanced visualizations (radar charts, heatmaps, complexity analysis)
- Report generation (extended reports, recommendation reports)
- Path setup and file discovery
- Package installation utilities

⚠️ **Partially Implemented (Stubs):**
- Specialized analysis functions in `analysis/specialized.py` (14 functions)
  - These are placeholders with function signatures
  - Full implementation pending (can be copied from original pipeline)
  - Functions: `analyze_emergent_laws`, `analyze_friedmann_cosmology`, `analyze_cmb_anomalies`, `analyze_lockin_dynamics`, `analyze_quantum_fields`, `analyze_entanglement`, `analyze_parameter_sensitivity`, `analyze_topology`, `analyze_i_definitions_direct`, `analyze_planck_fit`, `analyze_life_top_universes`, `analyze_entropy_volatility`, `analyze_physical_anomalies`, `analyze_statistical_finetuning`

---

## 13. Troubleshooting & Tips

### Issue: No simulation data found
- **Solution**: Ensure simulation results exist in `SIMULATION_RUNS/universe/`
- Check that `TARGET_MODE` matches your simulation mode (`batch_all` or `batch_ei`)
- Verify directory structure: `SIMULATION_RUNS/universe/TQE_Universe_Simulation_{mode}_{timestamp}/`

### Issue: Desktop write permissions
- **Solution**: Ensure you have write access to Desktop
- Check that `Desktop/TQE_Analysis_Modular_Results/` can be created
- On macOS/Linux: Check Desktop permissions with `ls -la ~/Desktop`

### Issue: Missing specialized analysis results
- **Solution**: The specialized analysis functions are currently stubs
- Full implementation pending (see `analysis/specialized.py`)
- Core analysis (comparative, model selection, visualizations) works fully

### Issue: Import errors
- **Solution**: Ensure you're running from the correct directory
- Try: `python -m TQE_Analysis_Modular.main` from the parent directory
- Or: `cd TQE_Analysis_Modular && python main.py`

### Issue: Package installation fails
- **Solution**: The pipeline auto-installs packages, but if it fails:
  - Manual installation: `pip install pandas numpy matplotlib seaborn scipy tqdm`
  - Check Python version: `python --version` (requires 3.9+)

---

## 14. Performance & Runtime

- **Typical runtime**: ~1-3 minutes per batch on a modern laptop
- **Bottleneck**: Plotting (matplotlib figure generation)
- **Memory usage**: Moderate (~500MB-2GB depending on batch size)
- **Disk usage**: ~50-200MB per analysis run (PNG files dominate)

**Optimization tips:**
- Reduce `FIGURE_DPI` for faster plotting (default: 300)
- Disable specialized analyses if not needed (currently stubs anyway)
- Process smaller batches if memory constrained

---

## 15. Citation & License

Please cite the repository when reusing the modular pipeline:

**Plain Text Citation:**
> Stefan Len. (2025). *TQE Analysis Pipeline v4.2.0 PRO (Modular)* (Version v4.2.0_PRO) [Software]. Zenodo. https://doi.org/10.5281/zenodo.17627756

**BibTeX Entry:**
```bibtex
@software{Len_2025_TQE_Analysis_v4.2.0_PRO,
  author    = {Len, Stefan},
  orcid     = {https://orcid.org/0009-0007-0383-7315},
  title     = {{TQE Analysis Pipeline v4.2.0 PRO (Modular)}},
  version   = {4.2.0},
  year      = {2025},
  publisher = {Zenodo},
  url       = {https://github.com/SteviLen420/TQE_simulation},
  doi       = {10.5281/zenodo.17627756}
}
```

**License:** MIT License - Copyright (c) 2025 Stefan Len

---

## 16. Next Steps

1. **Review recommendation**: Check `04_best_model_selection/recommendation_report.md` for model choices
2. **Inspect detailed metrics**: Review `02_detailed_metrics/all_runs_metrics.csv` for downstream ML
3. **Compare E-only vs E+I**: Check `01_comparative_analysis/eonly_vs_ei/` if available
4. **Implement specialized analyses**: Complete stub functions in `analysis/specialized.py` from original pipeline
5. **Custom analysis**: Use individual modules in notebooks for targeted studies

---

## 17. Scientific Questions Addressed

The pipeline addresses several key scientific questions:

1. **Which I-definition best matches observational data?** → Physical-Laws-Focused ranking
2. **Which I-definition produces the most complex, life-compatible universes?** → Complexity-Focused ranking
3. **Which I-definition maximizes stability?** → Stability-Focused ranking
4. **How much does I-coupling improve over E-only?** → E-only vs E+I comparison
5. **What are the key metrics driving I-definition performance?** → Correlation matrix, detailed metrics
6. **How do different I-definitions compare across multiple dimensions?** → Radar charts, heatmaps

---

## 18. Reproducibility

The pipeline ensures reproducibility through:
- **Deterministic file discovery**: Consistent directory structure and naming
- **Timestamp-based organization**: Each analysis run is timestamped
- **Raw data preservation**: Collected data and metrics saved as pickles
- **Configuration snapshot**: MASTER_CTRL settings documented in reports
- **Validation checks**: Ensures critical artifacts are generated

---

## 19. Future Enhancements

Potential improvements for future versions:
- Complete implementation of specialized analysis functions
- Machine learning integration for I-definition prediction
- Interactive dashboards (Plotly/Bokeh)
- Batch processing of multiple timestamps
- Parallel analysis execution
- Database integration for large-scale result storage

---

**For questions, issues, or contributions, please refer to the main repository documentation or open an issue on GitHub.**
