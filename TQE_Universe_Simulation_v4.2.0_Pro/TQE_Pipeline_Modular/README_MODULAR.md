# SPDX-License-Identifier: MIT
#
# Copyright (c) 2025 Stefan Len

# TQE Pipeline Modular - Modularized Pipeline Structure

This directory contains the modularized version of the TQE Universe Simulation Pipeline. The original monolithic pipeline file (`TQE_Universe_Simulation_Full_Pipeline_v4.2.0_PRO.py`) has been split into a modular structure for better maintainability, reusability, and easier development.

## Overview

The modular structure organizes the pipeline into logical components, making it easier to:
- **Maintain**: Smaller, focused modules are easier to understand and modify
- **Reuse**: Individual modules can be imported and used independently
- **Test**: Each module can be tested in isolation
- **Navigate**: Easier to find specific functionality
- **Develop**: Multiple developers can work on different modules simultaneously

## Structure

```
TQE_Pipeline_Modular/
├── __init__.py
├── config/
│   ├── __init__.py
│   └── master_ctrl.py          # MASTER_CTRL configuration dictionary
├── core/
│   ├── __init__.py
│   ├── pipeline_context.py     # PipelineContext class
│   └── physics_engine.py       # PhysicsEngine class
├── utils/
│   ├── __init__.py
│   ├── plotting.py             # Plotting utility functions
│   ├── memory.py               # Memory management utilities
│   ├── formatting.py           # Formatting utilities
│   └── cmb_utils.py            # CMB-specific utilities
├── simulation/
│   ├── __init__.py
│   ├── monte_carlo.py          # Phase 1: Monte Carlo + Goldilocks
│   ├── goldilocks.py           # Bayesian Goldilocks optimization logic
│   └── lock_in.py              # Lock-in mechanism
├── phases/
│   ├── __init__.py
│   ├── phase_01_10.py          # Phases 1-10: Core simulation and basic analysis
│   ├── phase_11_20.py          # Phases 11-20: ML, CMB, and correlation analysis
│   └── phase_21_28.py          # Phases 21-28: Advanced analysis and visualization
├── analysis/
│   ├── __init__.py
│   ├── cmb_analysis.py         # CMB analysis functions
│   ├── anomaly_detection.py    # Anomaly detection functions
│   ├── law_detection.py        # Law detection functions
│   └── bayesian.py             # Bayesian analysis functions
└── main.py                     # Main orchestrator: run_pipeline and main execution
```

## Module Organization

### Config Module (`config/`)
- **master_ctrl.py**: Contains the `MASTER_CTRL` configuration dictionary with all pipeline settings, parameters, and execution modes.

### Core Modules (`core/`)
- **pipeline_context.py**: `PipelineContext` class that encapsulates all transient state and global configurations for a single pipeline run. Handles seed management, path management, file I/O, and runtime registries.
- **physics_engine.py**: `PhysicsEngine` class that handles all physical computations related to E (Energy), I (Information), X (coupling parameter), and CMB generation. Includes energy sampling, information parameter definitions, coupling computation, CMB map generation, and Planck-aware gradient fine-tuning hooks (E/I attractors, jitter, E–I correlation, α calibration, χ² scoring).

### Utils Modules (`utils/`)
- **plotting.py**: Scientific plotting style setup (`setup_scientific_plotting_style`) and consistent plot styling functions (`apply_consistent_plot_style`).
- **memory.py**: Memory optimization (`optimize_for_colab`) and cleanup functions (`cleanup_memory`) for Colab and local environments.
- **formatting.py**: Formatting utility functions for labels (`_pretty_label`) and values (`_fmt`).
- **cmb_utils.py**: CMB-specific utilities including:
  - `_axis_from_lmap`: Extract axis from HEALPix maps
  - `detect_cold_spots_healpix`: Multi-scale cold spot detection
  - `detect_axis_of_evil`: Axis of Evil alignment detection
  - `generate_coldspot_overlay`: Generate cold spot overlay plots
  - `generate_aoe_overlay`: Generate Axis of Evil overlay plots
  - `get_cached_cmb_or_generate`: CMB map caching for performance

### Simulation Modules (`simulation/`)
- **monte_carlo.py**: Monte Carlo universe sampling functions:
  - `run_mc`: Main Monte Carlo simulation function
  - `_run_single_universe`: Single universe simulation
  - `phase_01_monte_carlo`: Phase 1 execution with integrated Bayesian Goldilocks
- **goldilocks.py**: Bayesian adaptive Goldilocks optimization:
  - `bayesian_adaptive_goldilocks`: Gaussian Process Regression with UCB acquisition
  - `compute_dynamic_goldilocks`: Dynamic Goldilocks zone computation
  - `sigma_goldilocks`: Goldilocks zone sigma function
  - `simulate_lock_in`: Lock-in simulation function
  - `_check_stability_calibration`: Stability calibration check
  - `_plot_bayesian_goldilocks`: Bayesian Goldilocks visualization
- **lock_in.py**: Lock-in mechanism and stability:
  - `adjust_stability_thresholds`: Dynamic threshold adjustment

### Phases Modules (`phases/`)
- **phase_01_10.py**: Phases 2-10 covering:
  - Phase 2: Stability curve analysis
  - Phase 3: E-I parameter space visualization
  - Phase 4: Fluctuation dynamics panels
  - Phase 5: Stability-by-I analysis
  - Phase 6: Lock-in histogram
  - Phase 7: Stability distribution
  - Phase 8: Average lock-in curve
  - Phase 9: Feature importance analysis
  - Phase 10: Emergent laws detection
- **phase_11_20.py**: Phases 11-20 covering:
  - Phase 11: Statistical finetuning detector
  - Phase 12: Best universe plots & CMB generation
  - Phase 13: Complete CMB map coverage
  - Phase 14: Entropy volatility analysis
  - Phase 15: Planck validation
  - Phase 16: CMB anomaly detection
  - Phase 17: E+I importance comparison
  - Phase 18: Multi-mode Goldilocks comparison
  - Phase 19: CMB analysis plots
  - Phase 20: Comprehensive correlation analysis
  - Plus helper functions for visualization and analysis
- **phase_21_28.py**: Phases 21-28 covering:
  - Phase 21: Advanced statistical analysis
  - Phase 22: CMB anomaly analysis plots
  - Phase 23: Enhanced physics analysis
  - Phase 24: Comprehensive data extraction
  - Phase 25: Advanced anomaly detection
  - Phase 26: Advanced law detection
  - Phase 27: Comprehensive visualization extraction
  - Phase 28: Final summary, complexity & life-compatibility synthesis, and Bayesian integration

### Analysis Modules (`analysis/`)
- **bayesian.py**: Bayesian analysis functions:
  - `compute_bayesian_model_selection`: BIC, AIC calculation
  - `run_nested_sampling`: Nested sampling for Bayesian evidence
  - `generate_corner_plot`: Corner plots for parameter posteriors
  - `save_bayesian_metrics_csv`: Save Bayesian metrics
  - `plot_bayesian_comparison`: Bayesian comparison plots
- **anomaly_detection.py**: Multi-category anomaly detection:
  - `_detect_quantum_field_anomalies`: Quantum field anomalies
  - `_detect_entropy_anomalies`: Entropy fluctuation anomalies
  - `_detect_topological_anomalies`: Topological defect anomalies
  - `_detect_energy_anomalies`: Energy conservation anomalies
  - `_detect_information_anomalies`: Information theory anomalies
  - `_detect_cmb_statistical_anomalies`: CMB statistical anomalies
  - `_create_anomaly_detection_plots`: Anomaly visualization
- **law_detection.py**: Multi-category law detection:
  - `_detect_conservation_laws`: Conservation laws (energy, momentum, charge)
  - `_detect_symmetry_laws`: Symmetry breaking laws
  - `_detect_scaling_laws`: Scaling laws and power laws
  - `_detect_emergent_laws`: Emergent behavior laws
  - `_detect_quantum_laws`: Quantum mechanical laws
  - `_detect_thermodynamic_laws`: Thermodynamic laws
  - `_detect_statistical_laws`: Statistical mechanics laws
  - `_detect_field_laws`: Field theory laws
  - `_detect_geometric_laws`: Geometric and topological laws
  - `_detect_information_laws`: Information theory laws
  - `_create_law_detection_plots`: Law visualization

### Complexity & Life-Compatibility Analytics
- Integrated in `phase_21_28.py` via `integrate_complexity_analysis`
  - Produces `complexity_metrics_summary.csv` (run-level metrics) and `life_compatibility_summary.json`
  - Saves dual-panel visualization `complexity_life_components.png`
  - Optionally exports top-universe ranking (`complexity_universe_ranking.csv` + `complexity_top_universes.png`) when `COMPLEXITY_TOP_N > 0`
- Controlled by new `MASTER_CTRL` keys:
  - `ENABLE_COMPLEXITY_ANALYSIS`, `SAVE_COMPLEXITY_PLOTS`
  - `COMPLEXITY_TOP_N`, `COMPLEXITY_THRESHOLD`, `LIFE_COMPATIBILITY_THRESHOLD`

### Main Module (`main.py`)
- **run_pipeline**: Main pipeline orchestrator that executes all 28 phases sequentially
- **switch_pipeline_type**: Switch between E+I and E-Only pipeline modes
- **run_multi_i_parameter_analysis**: Run pipeline for multiple I parameter definitions
- **run_single_i_parameter_mode**: Run pipeline for a single I parameter definition
- **create_i_parameter_summary_analysis**: Create comprehensive summary analysis
- **_print_pipeline_completion**: Print standardized completion summary
- **Main execution block**: `if __name__ == "__main__"` with 4 execution modes

## Usage

The modular pipeline can be used in the same way as the original monolithic version:

### As a Module

```python
from TQE_Pipeline_Modular.main import run_pipeline
from TQE_Pipeline_Modular.config.master_ctrl import MASTER_CTRL

# Run the pipeline with default settings
result = run_pipeline()

# Run with custom configuration
config = MASTER_CTRL.copy()
config["NUM_UNIVERSES"] = 1000
config["I_DEFINITION_MODE"] = "jensen_shannon"
result = run_pipeline(config_override=config)
```

### Direct Execution

```bash
# From the TQE_simulation_Prototype directory
python -m TQE_Pipeline_Modular.main

# Or directly
cd TQE_Pipeline_Modular
python main.py
```

### Importing Individual Modules

```python
# Import specific modules
from TQE_Pipeline_Modular.core.pipeline_context import PipelineContext
from TQE_Pipeline_Modular.core.physics_engine import PhysicsEngine
from TQE_Pipeline_Modular.simulation.monte_carlo import run_mc
from TQE_Pipeline_Modular.phases.phase_01_10 import phase_02_stability_curve

# Use individual components
ctx = PipelineContext(MASTER_CTRL)
physics = PhysicsEngine(MASTER_CTRL, ctx.rng)
```

## Import Structure

All modules use relative imports to maintain the modular structure:

```python
# Within a module, use relative imports
from ..config.master_ctrl import MASTER_CTRL
from ..core.pipeline_context import PipelineContext
from ..core.physics_engine import PhysicsEngine
from ..utils.plotting import setup_scientific_plotting_style
from ..simulation.monte_carlo import run_mc
```

## Benefits of Modular Structure

1. **Maintainability**: Smaller, focused modules (typically 200-2000 lines) are easier to understand and modify than a single 10,000+ line file
2. **Reusability**: Individual modules can be imported and used independently in other projects
3. **Testability**: Each module can be tested in isolation with unit tests
4. **Readability**: Easier to navigate and find specific functionality using IDE features
5. **Parallel Development**: Multiple developers can work on different modules simultaneously without conflicts
6. **Version Control**: Smaller files make git diffs more meaningful and easier to review
7. **Documentation**: Each module can have focused documentation

## Original Pipeline

The original monolithic pipeline file remains unchanged at:
```
TQE_Universe_Simulation_Full_Pipeline/TQE_Universe_Simulation_Full_Pipeline_v4.2.0_PRO.py
```

This ensures backward compatibility and provides a reference implementation.

## Migration Notes

- The modular version provides **exactly the same functionality** as the original pipeline
- **Backward compatibility**: The `main.py` can be executed the same way as the original
- **Gradual migration**: You can gradually transition to using the modular version
- **Import updates**: All imports have been updated to use the modular structure
- **No breaking changes**: The API remains the same, only the internal structure changed

## File Count

The modular structure consists of **26 Python files** organized into logical modules, compared to the original single 10,454-line file.

## Documentation

For detailed documentation about the pipeline functionality, phases, configuration, and scientific background, see:
- **README.md** - Complete original pipeline documentation (this file's companion)
- Original pipeline header comments in each module file
- Inline code documentation and docstrings

## Development

When adding new features or modifying existing code:

1. **Identify the appropriate module** based on functionality
2. **Use relative imports** to maintain modular structure
3. **Update `__init__.py`** files if adding new public functions
4. **Maintain consistency** with existing code style and patterns
5. **Test modules independently** before integration

---

**Note**: This modular structure was created by splitting the original monolithic pipeline file while preserving all functionality and maintaining backward compatibility.

