# ===================================================================================
#  TQE (E,I) UNIVERSE SIMULATION PIPELINE
# ===================================================================================
# Author: Stefan Len
# ===================================================================================

"""
===================================================================================
Title: The TQE Framework: A Modular, Reproducible Pipeline for Monte Carlo
Simulation of Universe Evolution from Energy-Information Principles
===================================================================================

ABSTRACT
-----------------------------------------------------------------------------------
In this work, I present a comprehensive computational framework, TQE, 
for conducting large-scale Monte Carlo simulations of universe evolution.
The model is based on the foundational hypothesis of a coupling between 
abstract 'Energy' (E) and 'Information' (I) principles, which jointly determine 
the developmental trajectory of each simulated universe. The framework is 
implemented as a modular, multi-stage Python pipeline designed for rigorous 
scientific investigation.

The key stages encompass:
(i) a centralized, deterministic seeding mechanism to ensure full reproducibility;
(ii) initialization of E and I parameters for a population of universes;
(iii) a pre-collapse phase modeling quantum superposition and fluctuation dynamics;
(iv) a critical "law lock-in" event where fundamental properties stabilize
from a chaotic state;
(v) a post-collapse expansion phase; and
(vi) a suite of advanced analysis modules.

These modules perform deep diagnostics, including the procedural generation of
CMB-like maps, the quantification of fine-tuning metrics, and the statistical
analysis of cosmological anomalies (e.g., Cold Spot, low-multipole alignments).
Furthermore, an integrated Explainable AI (XAI) module employs machine learning
models (Random Forest, SHAP, LIME) to determine the causal relationships between
initial parameters and emergent universal properties. The entire pipeline is
architected for robustness, featuring configuration-driven execution via profiles
and a final manifest-generation stage that consolidates all parameters, data,
and metadata into a navigable, publication-ready dataset.


===================================================================================
1. COMPUTATIONAL FRAMEWORK AND METHODOLOGY
===================================================================================

The TQE pipeline is a sequence of 20 interoperable Python modules designed to
simulate and analyze an ensemble of universes. The architecture can be logically
divided into four main components.

-----------------------------------------------------------------------------------
1.1. Framework Architecture and Reproducibility (Modules 00-04)
-----------------------------------------------------------------------------------
The foundation of the pipeline ensures consistency, configurability, and
reproducibility.

- Configuration and Orchestration (00_config, 01_Master_Control): A centralized
  dictionary (`MASTER_CTRL`) contains all parameters for the simulation and
  analysis. This allows for configuration-driven execution, with profiles (e.g.,
  `demo`, `paper`) enabling rapid testing and production runs. The
  `Master_Control` module also serves as the execution harness, dynamically
  calling each stage in sequence.

- I/O and Environment Handling (02_io_paths, 03_imports): A dedicated I/O module
  robustly manages file paths for different environments (e.g., local desktop,
  Google Colab with automated Drive mounting). It generates a unique, timestamped
  run directory for each execution, preventing data overwrites. A central `imports`
  module ensures all other modules share the same library versions and cached
  run paths.

- Seeding and RNG (04_seeding): This is the cornerstone of reproducibility.
  Upon the first execution in a new run directory, a single, high-entropy 64-bit
  "master seed" is generated and saved. The `numpy.random.SeedSequence.spawn()`
  method is then used to deterministically derive a unique, independent seed for
  each of the N universes. All subsequent stochastic processes in the pipeline are
  seeded from either the master seed or the specific per-universe seed,
  guaranteeing bit-for-bit reproducibility of any given run.

-----------------------------------------------------------------------------------
1.2. Simulation Core: Universe Generation and Evolution (Modules 05-10)
-----------------------------------------------------------------------------------
This component simulates the life cycle of each universe.

- Initialization (05_energy_sampling, 06_information_bootstrap): The
  simulation begins by sampling initial `E0` (Energy) and `I0` (Information)
  values for the entire population from specified distributions.

- Pre-Collapse Dynamics (07_fluctuation, 08_superposition): These modules model
  the initial chaotic state before universal laws are fixed. This includes a
  fluctuation phase and an optional quantum superposition stage that can
  leverage the `qutip` library to model quantum states, from which information
  metrics are derived.

- Law Lock-in (09_collapse_LawLockin): A critical event where a fluctuating "law"
  variable `L(t)` evolves under decaying noise. Based on the coupled `X = f(E,I)`
  parameter and stability thresholds, this variable may or may not stabilize
  ("lock-in"), representing the emergence of fixed physical laws.

- Expansion (10_expansion): Universes that successfully lock-in enter a
  post-collapse expansion phase. A size parameter `S(t)` evolves via a
  multiplicative growth process, where the growth rate is modulated by the
  universe's `X` value and subject to decaying noise.

-----------------------------------------------------------------------------------
1.3. Analysis Suite: Diagnostics and Anomaly Detection (Modules 11-18)
-----------------------------------------------------------------------------------
This component analyzes the properties of the simulated population of universes.

- Aggregation and Ranking (11_montecarlo, 12_best_universe): Results from the
  collapse and expansion phases are merged into a master `DataFrame`. A weighted
  scoring function is then applied to rank universes based on metrics like
  growth, stability, and speed of lock-in, identifying the "best" universes.

- Map Generation and Diagnostics (13_cmb_map_generation, 14_finetune_diagnostics):
  For each universe, a 2D CMB-like map is procedurally generated. These maps are
  then analyzed for fine-tuning indicators, including RMS, spectral slope (`alpha`),
  correlation length, skewness, and kurtosis.

- Anomaly Detection (15-18): The simulated CMB maps are algorithmically searched
  for four well-known cosmological anomalies: the Cold Spot, Low-Multipole
  (Quadrupole-Octopole) Alignments, Lack of Large-Angle Correlation, and
  Hemispherical Power Asymmetry.

-----------------------------------------------------------------------------------
1.4. Meta-Analysis and Finalization (Modules 19-20)
-----------------------------------------------------------------------------------
The final component provides a high-level interpretation and summary of the
entire run.

- Explainable AI (XAI) (19_xai): A comprehensive dataset is automatically
  aggregated from all previous stages. Machine learning models (Random Forest)
  are trained to predict various outcomes (e.g., `S_final`, `lockin_at`, anomaly
  presence) from the initial `E` and `I` parameters. SHAP and LIME analyses are
  then performed to calculate feature importance, providing insight into the
  causal drivers within the simulation.

- Manifest Generation (20_results_manifest): The final module scans the entire
  run directory, collecting all generated summary files (JSONs) and data tables
  (CSVs). It produces two key outputs: a master `run_manifest.json` that serves
  as a navigable table of contents for the run, and a single, wide-format
  `per_universe_summary.csv` that consolidates the most critical metrics for each
  universe, ready for final statistical analysis.

"""
