# Simulation Runs Directory

This directory contains all TQE simulation execution results, organized by pipeline type.

## Directory Structure

```
SIMULATION_RUNS/
├── universe/          # TQE Universe Simulation Full Pipeline results
├── heisenberg/        # TQE Heisenberg Fluctuation results
├── dark_energy/       # TQE Dark Energy Coupling results
├── analysis/          # TQE Analysis Pipeline comparative analysis results
└── README.md          # This file
```

## Pipeline Output Locations

### 1. Universe Simulation (`universe/`)
- **Pipeline**: `TQE_Universe_Simulation_Full_Pipeline_v4.2.0_Pro.py`
- **Structure**: `universe/TQE_Universe_Simulation_{mode}_{timestamp}/`
- **Modes**: `batch_all`, `batch_ei`, `single_ei`, `single_eonly`
- **Example**: `universe/TQE_Universe_Simulation_batch_all_20251118_101437/`

### 2. Heisenberg Fluctuation (`heisenberg/`)
- **Pipeline**: `TQE_Energy_Fluctuations_Heisenberg_v4.2.0_Pro.py`
- **Structure**: `heisenberg/TQE_Heisenberg_Fluctuation_{timestamp}/`
- **Example**: `heisenberg/TQE_Heisenberg_Fluctuation_20251101_054309/`

### 3. Dark Energy Coupling (`dark_energy/`)
- **Pipeline**: `TQE_DarkEnergy_Coupling_Simulation.py`
- **Structure**: `dark_energy/TQE_DarkEnergy_Coupling_Simulation_v4.2.0PRO_{timestamp}/`
- **Example**: `dark_energy/TQE_DarkEnergy_Coupling_Simulation_v4.2.0PRO_20251030_132700/`

### 4. Analysis Pipeline (`analysis/`)
- **Pipeline**: `TQE_Analysis_Pipeline_v4.2.0_PRO.py`
- **Structure**: `analysis/{mode}_{timestamp}_analysis/`
- **Example**: `analysis/batch_all_20251118_200708_analysis/`

## Current Runs

- **Universe**: `TQE_Universe_Simulation_batch_all_20251118_101437/` (1.1GB)
- **Heisenberg**: `TQE_Heisenberg_Fluctuation_20251101_054309/`
- **Dark Energy**: `Model_3_Geometric_beta0_0.2000_EplusI_20251030_132700/`
- **Analysis**: `batch_all_20251118_200708_analysis/`

## Notes

- Each pipeline automatically creates timestamped directories
- All pipelines are configured to save to their respective subdirectories
- Old runs can be archived or removed to save disk space
- Analysis results reference universe simulation runs from `universe/` directory
