**TQE (E,I) Universe Simulation Pipeline**

Author: Stefan Len

Contact: stefan.len [at] gmail.com

Abstract
In this work, I present a comprehensive computational framework, TQE, for conducting large-scale Monte Carlo simulations of universe evolution. The model is based on the foundational hypothesis of a coupling between abstract 'Energy' (E) and 'Information' (I) principles, which jointly determine the developmental trajectory of each simulated universe. The framework is implemented as a single, powerful Python script designed for rigorous scientific investigation.

The key stages encompass:
(i) a centralized, deterministic seeding mechanism to ensure full reproducibility;
(ii) initialization of E and I parameters for a population of universes;
(iii) a pre-collapse phase modeling quantum superposition and fluctuation dynamics;
(iv) a critical "law lock-in" event where fundamental properties stabilize from a chaotic state;
(v) a post-collapse expansion phase; and
(vi) a suite of advanced analysis modules.

These modules perform deep diagnostics, including the procedural generation of CMB-like maps, the quantification of fine-tuning metrics, and the statistical analysis of cosmological anomalies (e.g., Cold Spot, low-multipole alignments). Furthermore, an integrated Explainable AI (XAI) module employs machine learning models (Random Forest, SHAP, LIME) to determine the causal relationships between initial parameters and emergent universal properties. The entire pipeline is architected for robustness, featuring configuration-driven execution and a final summary-generation stage that consolidates all parameters, data, and metadata into a navigable, publication-ready dataset.

Key Features
End-to-End Simulation: Simulates an entire ensemble of universes—from initial parameter sampling to final analysis—in a single, automated run.

Energy-Information Hypothesis: Implements a novel model where universe evolution is driven by the interplay of Energy (E) and Information (I).

Dynamic "Goldilocks" Zone: Automatically identifies and utilizes the parameter space (the "Goldilocks zone") most conducive to producing stable universes with fixed physical laws.

Cosmological Analysis Suite: Procedurally generates CMB-like sky maps for promising universes and analyzes them for fine-tuning indicators and statistical properties.

Anomaly Detection: Includes built-in modules to algorithmically search simulated universes for cosmological anomalies like the CMB Cold Spot and the Axis of Evil.

Explainable AI (XAI): Uses SHAP and LIME to automatically determine which initial parameters (E, I, X) are the most significant drivers of simulation outcomes (e.g., stability, anomaly presence).

Guaranteed Reproducibility: A master seeding system ensures that any simulation run can be reproduced bit-for-bit for scientific validation.

Zero-Setup Execution: The script automatically detects and installs any missing Python dependencies (healpy, qutip, shap, etc.) on its first run.

Cloud-Ready: Detects when it's running in Google Colab and can automatically mount your Google Drive to save results.

Installation and Dependencies
This pipeline is designed to be run with minimal setup. The script will automatically install any required packages using pip if they are not found in your environment.

Core Dependencies:

numpy

pandas

matplotlib

scikit-learn

tqdm

Auto-Installed Dependencies (for full functionality):

qutip: For quantum superposition stages.

healpy: For spherical CMB map generation and analysis (Axis of Evil).

scipy: For advanced scientific computations.

shap: For the SHAP analysis in the XAI module.

lime: For the LIME analysis in the XAI module.

Simply clone the repository to get started:

Bash

git clone https://github.com/SteviLen420/TQE_simulation.git
cd TQE_simulation
How to Configure the Pipeline
All simulation parameters are controlled from a single, centralized dictionary named MASTER_CTRL at the top of the TQE_Universe_Simulation_Full_Pipeline.py script.

To change any aspect of the simulation, simply edit the values in this dictionary before running the script.

Example: Key Configuration Options in MASTER_CTRL

Python

MASTER_CTRL = {
    # --- Core simulation ---
    "NUM_UNIVERSES":        5000,   # Number of universes to simulate
    "LOCKIN_EPOCHS":        700,    # Duration of the "law lock-in" phase
    "PIPELINE_VARIANT":     "full", # "full" (E+I) or "energy_only" (E only)
    "SEED":                 None,   # Master seed for reproducibility (if None, one is generated)

    # --- E–I coupling (X definition) ---
    "X_MODE":               "product",  # How E and I are combined: "product" | "E_plus_I"
    "ALPHA_I":              0.8,        # Coupling strength of I

    # --- Goldilocks zone controls ---
    "GOLDILOCKS_MODE":      "dynamic",  # "dynamic" (auto-detect) or "heuristic" (manual)

    # --- Anomaly Detectors ---
    "CMB_COLD_ENABLE":      True,       # Enable the Cold Spot detector
    "CMB_AOE_ENABLE":       True,       # Enable the Axis-of-Evil detector

    # --- Machine Learning / XAI ---
    "RUN_XAI":              True,       # Master switch for the XAI analysis section
    "RUN_SHAP":             True,       # Generate SHAP plots
    "RUN_LIME":             True,       # Generate LIME plots

    # --- Outputs / IO ---
    "SAVE_FIGS":            True,       # Save all plots to disk
    "SAVE_DRIVE_COPY":      True,       # In Colab, copy results to Google Drive
    "DRIVE_BASE_DIR":       "/content/drive/MyDrive/TQE_Universe_Simulation_Full_Pipeline",
}
How to Run the Simulation
The entire pipeline is executed by running the main Python script.

On a Local Machine
Navigate to the repository directory and run:

Bash

python TQE_Universe_Simulation_Full_Pipeline.py
The script will start the simulation, print progress updates to the console, and save all outputs into a new, timestamped directory.

In Google Colab
Upload the TQE_Universe_Simulation_Full_Pipeline.py script to your Colab environment.

Create a new code cell and run the script:

Python

!python TQE_Universe_Simulation_Full_Pipeline.py
The script will automatically detect the Colab environment and ask for permission to mount your Google Drive. Once authorized, all results will be saved to the path specified by DRIVE_BASE_DIR in the configuration.

Understanding the Output
For each run, the pipeline creates a unique directory with a name like TQE_Universe_Simulation_Full_Pipeline_E+I_20250914_143000. Inside this directory, you will find:

tqe_runs.csv: The main results file, containing the initial parameters and final outcomes for every simulated universe.

summary_full.json: A detailed JSON file containing all configuration parameters used for the run, a summary of the results, and paths to all generated artifacts.

cmb_coldspots_summary.csv: A table of all Cold Spots detected across the simulated universes.

cmb_aoe_summary.csv: A table detailing the Quadrupole-Octopole alignment (Axis of Evil) for each universe.

A figs/ subdirectory containing all generated plots, including:

stability_curve.png: The "Goldilocks" curve showing stability probability versus the composite X parameter.

scatter_EI.png: A 2D plot of all universes in the (E, I) parameter space, colored by their stability.

best_universes/: A folder with entropy evolution plots for the top-ranked universes.

cmb_best/: A folder with generated CMB map images for the best universes.

xai/: A folder containing all SHAP and LIME plots, organized by target variable (e.g., stability, cold spot presence).

How to Cite
If you use this software in your research, please consider citing it. This helps to acknowledge the work and allows others to discover and reproduce your results.

Plain Text Citation:

Stefan Len. (2025). TQE (E,I) Universe Simulation Pipeline (Version 2.0.0) [Software]. GitHub. https://github.com/SteviLen420/TQE_simulation

BibTeX Entry:

Útržok kódu

@software{TQE_Simulation_2025,
  author = {Len, Stefan},
  title = {{TQE (E,I) Universe Simulation Pipeline}},
  year = {2025},
  publisher = {GitHub},
  version = {2.0.0},
  url = {https://github.com/SteviLen420/TQE_simulation}
}
