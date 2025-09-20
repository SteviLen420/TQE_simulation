 SPDX-License-Identifier: MIT

 Copyright (c) 2025 Stefan Len

[![CI](https://github.com/SteviLen420/TQE_simulation/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/SteviLen420/TQE_simulation/actions/workflows/ci.yml)  
[![Python](https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11-blue)](https://www.python.org/doc/)  

#  TQE UNIVERSE SIMULATION PIPELINE

**Title: The TQE Framework: A Modular, Reproducible Pipeline for Monte Carlo
Simulation of Universe Evolution from Energy-Information Principles**

**Author**: Stefan Len


### Abstract


The TQE Framework is a novel computational pipeline designed to investigate the hypothesis that the fundamental laws of physics are not axiomatic but emerge from a more primitive interplay of Energy (E) and Information (I). This framework provides an end-to-end, proof-of-concept environment for conducting Monte Carlo simulations of ensembles of universes, enabling the systematic exploration of a wide parameter space of initial conditions. The core of the pipeline is a multi-stage simulation that models the lifecycle of a universe: an initial pre-collapse phase, where physical laws are treated as fluctuating and stochastic; a probabilistic “law lock-in” event, where a stable set of effective physical constants is selected; and a subsequent expansion phase that generates large-scale structures. An integrated analysis suite processes the outputs of these simulations to generate cosmological observables analogous to the Cosmic Microwave Background (CMB) and to perform targeted scans for selected anomalies, such as the Cold Spot and hemispherical asymmetries. A key contribution of this work is the integration of an Explainable AI (XAI) module, which applies machine learning models together with interpretability techniques such as SHAP to explore the relationship between the initial (E,I) conditions and the emergent characteristics of a simulated universe. While some components (e.g., anomaly detection and LIME explanations) are implemented in simplified form, the framework is architected for reproducibility and extensibility, making it a solid research-grade prototype and a foundation for further development in theoretical cosmology.


## How to Cite

If you use this software in your research, please consider citing it. This helps to 
acknowledge the work and allows others to discover and reproduce your results. 
The `CITATION.cff` file in the root of this repository is provided for automated 
citation management.

**Plain Text Citation:**
> Stefan Len. (2025). *TQE (E,I) Universe Simulation Pipeline* (Version 1.0.0) [Software]. GitHub. https://github.com/SteviLen420/TQE_simulation

**BibTeX Entry:**

```bibtex
@software{Len_2025_TQE,
  author    = {Len, Stefan},
  title     = {{TQE Universe Simulation Pipeline}},
  version   = {1.0.0},
  date      = {XXXX.XX.XX},
  publisher = {GitHub},
  url       = {https://github.com/SteviLen420/TQE_simulation}
}
```
________________

## Installation & Environment Setup

The TQE Framework requires **Python 3.9+** and several scientific libraries.  
I recommend setting up a dedicated virtual environment to ensure reproducibility.

### Using pip
```bash
git clone https://github.com/SteviLen420/TQE_simulation.git
cd TQE_simulation
python -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate
pip install -r requirements.txt 
```
### Using Conda
```bash
git clone https://github.com/SteviLen420/TQE_simulation.git
cd TQE_simulation
conda env create -f environment.yml
conda activate tqe_env
```
### Optional Dependencies
	•	**qutip** – enables quantum-mechanical modeling of pre-collapse dynamics
	•	**healpy** – required for generating CMB-like sky maps
	•	**shap, lime** – required for Explainable AI (XAI) analysis

💡 If installation issues occur with healpy or qutip, we recommend using Conda, as it handles binary dependencies more reliably than pip.

## Quickstart / Usage
The simulation is launched by running the script directly. Its behavior can be fully customized by modifying the `MASTER_CTRL` Python dictionary, which is located at the top of the .py file.

### 1. Modify Settings (Optional)
Open the `TQE_Universe_Simulation_Full_Pipeline.py` file in a text editor and change the values in the `MASTER_CTRL` dictionary according to your requirements.
```Python
# Example: Increasing the number of universes
MASTER_CTRL = {
    "NUM_UNIVERSES": 10000,
    "SAVE_FIGS": True,
    # ... other settings
}
```
### 2. Run the Simulation
Run the script from the command line.
```bash
python TQE_Universe_Simulation_Full_Pipeline.py
```

## Configuration Parameters

The behavior of the TQE Framework is controlled by the central `MASTER_CTRL` Python dictionary within the script. 
The table below lists some of the most important parameters:

| Parameter        | Type   | Default | Description                                                           |
|------------------|--------|---------|-----------------------------------------------------------------------|
| NUM_UNIVERSES    | int    | 5000    | Number of universes to simulate in the Monte Carlo run.               |
| PIPELINE_VARIANT | string | "full"  | `"full"` (E+I) or `"energy_only"` (E only) pipeline variant.          |
| SEED             | int    | None    | The master seed for reproducibility. If `None`, it's auto-generated.  |
| SAVE_DRIVE_COPY  | bool   | True    | If True and running in Colab, copies results to Google Drive.         |
| GOLDILOCKS_MODE  | string | "dynamic" | `"dynamic"` or `"heuristic"` mode for stability window calculation. |
| RUN_XAI          | bool   | True    | Master switch for running the entire Explainable AI (XAI) section.    |
| CMB_BEST_FIGS    | int    | 3       | How many CMB maps of the "best" universes to generate.                |


## Computational Framework & Methodology
The TQE Framework is structured as a multi-stage computational pipeline for the systematic investigation of emergent physical laws. Its design emphasizes configuration-driven execution and reproducibility, with modularity implemented at the conceptual level and partially realized in the current prototype.

### High-Level Architecture
The workflow is orchestrated by a central Python dictionary named `MASTER_CTRL`, which is located at the top of the main script (`TQE_Universe_Simulation_Full_Pipeline.py`). This configuration-as-code approach enables experimental campaigns—including parameter sweeps and analysis settings—to be defined directly within the source code. This provides a clear and direct foundation for reproducibility and transparent experimental design.

The pipeline is organized into four sequential stages:

**Generation** – Reads simulation parameters from the `MASTER_CTRL` dictionary and generates initial conditions for an ensemble of universes, each defined by Energy (E) and Information (I) values drawn from statistical distributions.

**Simulation** – The computational core of the framework. It evolves universes through pre-collapse, law lock-in, and expansion phases. This stage is computationally intensive and includes preliminary support for parallel execution.

**Analysis** – Performs post-processing on raw outputs, including calculation of cosmological observables, generation of CMB-like sky maps, and execution of diagnostic tests to score universes for fine-tuning and scan for selected anomalies.

**Interpretation** – Applies Explainable AI (XAI) to synthesize ensemble results. Machine learning models are trained to predict outcomes from initial conditions, providing insights into causal relationships within the TQE model.

This staged structure conceptually supports re-running individual components (e.g., re-analyzing simulation data with a new anomaly detector) without repeating upstream steps, although the current implementation realizes this in a more streamlined, script-based form.



## Reproducibility by Design: The Seeding Hierarchy

To ensure computational reproducibility, the framework implements a two-tiered seeding hierarchy. This design provides deterministic outcomes within a fixed software environment, supporting verifiable and repeatable scientific workflows.

•	**Master Seed** – A single seed is defined by the `SEED` key in the `MASTER_CTRL` dictionary within the Python script. If set to an integer, it initializes a master pseudo-random number generator (PRNG) for the entire run. If left as None, a random seed is automatically generated.
 
•	**Per-Universe Seeds** – The master PRNG is used to deterministically generate a unique seed for each of the `NUM_UNIVERSES` in the ensemble. This ensures that each universe’s stochastic processes are initialized independently and reproducibly.

This hierarchical system provides two levels of control. Re-using the same `SEED` allows an entire ensemble to be reproduced, while selecting an individual universe's seed (found in the output universe_seeds.csv file) enables the exact reproduction of that universe’s evolution without re-running the full ensemble. This capability is especially valuable for debugging, targeted analysis, and validation of noteworthy cases. While the current implementation guarantees reproducibility under consistent library versions and environments, strict cross-platform bit-level determinism may vary depending on the underlying PRNG implementation.



## The Simulation Core: A Universe’s Lifecycle

The evolution of each universe in the simulation follows a staged lifecycle, modeling the transition from an indeterminate system to one governed by stable, fixed laws.
•	**Initialization** – Each run begins with initial Energy (E) and Information (I) values, sampled from distributions defined in the configuration file. These scalar quantities are the fundamental inputs to the TQE model.
 
•	**Pre-Collapse Phase** – In this phase, the effective physical laws (represented by a state vector X) fluctuate stochastically around values determined by the initial conditions. Conceptually, this is interpreted as a “quantum-like” regime where the laws are not yet fixed. The implementation models these fluctuations through random perturbations, with optional use of qutip for more advanced experiments.

### Clarification on the "Lawless" Phase
It is crucial to note that the "pre-collapse" phase, while characterized by fluctuating laws, does not start from an absolutely lawless state. The simulation framework is built upon the qutip quantum mechanics library, which fundamentally incorporates **the Schrödinger equation** and **Hamiltonian mechanics** as its operational baseline.

Therefore, the TQE model does not simulate the emergence of quantum mechanics itself. Instead, it investigates the process by which higher-level, effective physical laws and constants (like those governing gravitation or electromagnetism) are selected and stabilized from this fundamental, fluctuating quantum substrate. The simulation models the emergence of stable macro-level laws from an underlying quantum reality.

•	**Law Lock-In** – At a critical point, the universe transitions from fluctuating to stable laws, $X_{final}$. This lock-in event is governed by a stability_threshold parameter. When the variance of X(t) remains below this threshold for a specified number of epochs, the system records a lock_epoch. This value is a key output of the simulation.

•	**Expansion Phase** – After lock-in, the universe evolves deterministically according to the fixed laws $X_{final}$. This phase simulates large-scale expansion and structure formation, producing cosmological observables that are passed to the analysis stage.


## The Analysis & Diagnostics Suite

This suite of modules quantifies the outcomes of the simulation core, transforming raw data into scientifically meaningful metrics and visualizations.

•	**CMB-like Map Generation** – The framework uses the healpy library to project the final state of the simulated universe onto a spherical grid, producing sky maps analogous to the Cosmic Microwave Background. These maps enable direct visual and statistical comparison with observational data.

•	**Fine-Tuning Diagnostics** – A set of metrics evaluates how “fine-tuned” a universe’s locked-in laws ($X_{final}$) are for the emergence of complexity (e.g., structure formation, stable chemistry). The current implementation provides a continuous fine-tuning score, which can serve as the basis for classifying universes as barren, habitable, or other categories in future extensions.

•	**Anomaly Scanning** – The current implementation includes tools for detecting two key anomalies observed in CMB data:
	
 •	**CMB Cold Spot** – detection of unusually cold regions in the simulated sky.
	
 •	**Low-ℓ Multipole Alignment (Axis of Evil)** – analysis of the alignment between quadrupole ($\ell=2$) and octopole ($\ell=3$) modes.

Additional anomaly modules, such as the Hemispherical Power Asymmetry (HPA) and the Low-ℓ Alignment Correlation (LLAC), are mentioned in the framework design but are not yet implemented in the prototype.



## Explainable AI (XAI) for Cosmological Interpretation

The XAI module transforms the dataset generated by the simulation ensemble into interpretable scientific insights. By moving beyond what happens to why it happens, this module serves as the primary tool for extracting new, testable ideas from the simulation's results.

•	**Models** – The framework employs a multi-model strategy to suit different analytical goals. The primary model for general analysis and classification tasks is a RandomForest, chosen for its robust performance on tabular data and its relative interpretability. For targeted, high-performance regression tasks—such as predicting the fine_tuning_score or modeling the performance gains from including the Information parameter (I)—an **XGBoost (Extreme Gradient Boosting)** model is also utilized. XGBoost is capable of capturing more complex, non-linear patterns in the data, providing a deeper layer of analysis.

•	**Targets** – Models are trained on ensembles of simulated universes. Features include initial conditions (E, I) and emergent properties (e.g., lock_epoch), while target variables include outcomes such as the fine_tuning_score or anomaly flags (e.g., presence of a Cold Spot).

### Interpretability Methods
The framework integrates explainability tools to analyze the trained models:

•	**SHAP (SHapley Additive exPlanations)** – Used to quantify global feature importance. SHAP values answer questions such as: “What factors most influence whether a universe becomes fine-tuned?”

•	**LIME (Local Interpretable Model-agnostic Explanations)** – Support for local explanations is included at a prototype stage. `LIME` can provide case-by-case insights (e.g., explaining why a specific simulated universe exhibits an anomaly), though its integration is less developed than `SHAP`.

By combining ensemble simulation with `XAI`, the framework goes beyond description and supports hypothesis generation. It highlights the quantitative relationships between initial conditions and final observables, enabling new, testable ideas about the underlying physics of the TQE model.


## Results & Run Folder Structure
To ensure organized and traceable results, each execution of the pipeline generates a unique, timestamped output directory inside the runs/ folder (e.g., runs/TQE_Universe_Simulation_Full_Pipeline_full_20250917_083000/). 

Within this main run directory, the following structure is used:

Root Directory: All primary data artifacts, including .csv files (like tqe_runs.csv), .json summaries, and .npy map data, are saved directly into the root of this folder.

figs/ Subdirectory: A dedicated figs/ folder is created to hold all generated plots and visualizations (.png). This directory contains further subfolders for specific analyses, such as:

**best_universes/**

**cmb_best/**

**cmb_coldspots/**

**xai/**

This structure ensures that results from each run are self-contained, reproducible, and easy to navigate. The framework design also anticipates the ability to mirror outputs to a secondary location (e.g., for compute cluster storage), though this functionality is not yet fully implemented in the current prototype.



## Mathematical Formalism

The TQE Framework is guided by a set of mathematical principles that structure the simulation, while current implementations realize these in simplified stochastic form.

Key State Variables
	•	$E$: total energy of the initial state ($E \in \mathbb{R}^+$).
	•	$I$: total information content of the initial state ($I \in \mathbb{R}^+$).
	•	$X(t)$: time-varying state vector representing effective physical laws during the pre-collapse phase ($X(t) \in \mathbb{R}^n$).
	•	$X_{final}$: locked-in state vector of physical laws, $X_{final} = X(t_{lock})$.

Pre-Collapse Dynamics
Conceptually, fluctuations of $X(t)$ can be modeled as a Langevin process:

$$
\frac{dX(t)}{dt} = -\nabla V(X) + \sqrt{2D},\eta(t)
$$

where $V(X)$ is a potential landscape determined by $(E,I)$, $D$ is a diffusion coefficient, and $\eta(t)$ is Gaussian white noise.
In the current implementation, this is realized as stochastic perturbations around $X_{mean} = f(E,I)$ rather than a full potential-based dynamic.

Lock-In Criterion
The lock-in event occurs when the variance of $X(t)$ remains below a threshold $\epsilon$ for $\Delta t$ consecutive epochs. The lock-in epoch $t_{lock}$ is recorded as a key output.

Probabilistic Interpretation
The README formalism includes a Boltzmann-like expression for lock-in probability:

$$
P(\text{lock-in} \mid X) \propto \exp\left(-\frac{V(X)}{kT_{eff}}\right)
$$

This represents a guiding analogy rather than a fully implemented feature.

Fine-Tuning Score
The habitability score is implemented as a multivariate Gaussian around life-friendly target values:

$$
\mathcal{F}(X_{final}) = \prod_{i=1}^n \exp\left(-\frac{(x_i - x_{i,\text{target}})^2}{2\sigma_i^2}\right)
$$

Anomaly Metrics
Current anomaly modules include the CMB Cold Spot and Low-ℓ multipole alignments.
Additional anomaly equations, such as the Hemispherical Power Asymmetry (HPA), are included in the framework’s formal design but are not yet implemented in the prototype.
			



# Assessment of the Codebase

## Maturity Level

The TQE Framework is best described as a research-grade prototype. The codebase demonstrates architectural sophistication beyond a simple script, featuring a configuration-driven pipeline, a hierarchical seeding mechanism for reproducibility, and integration of an XAI module. These are hallmarks of a serious scientific tool. However, the current implementation lacks formal test suites, robust error handling, and polished user-facing APIs, which would be expected in a production-ready system.

**Ratings**
	
 •	**Rigor: 8/10** – The methodological foundations are strong. The two-tiered seeding hierarchy and the integration of explainable machine learning reflect a clear focus on scientific rigor. Some anomaly modules are implemented in simplified form, leaving room for further refinement.
	
 •	**Clarity: 7/10** – The high-level architecture is well structured and documented, but the core algorithms are dense and may pose a learning curve for those outside theoretical cosmology.
	
 •	**Reproducibility: 8/10** – Reproducibility is a central design feature, with its in-script configuration and a deterministic seeding mechanism ensuring consistency across runs in the same environment. Full cross-platform bit-level reproducibility, however, is not yet guaranteed.


## Justification and Suggestions for Improvement

The framework distinguishes itself through its strong scientific motivation, emphasis on reproducibility, and the novel integration of explainable AI. Its architecture is well-suited for large-scale computational experiments and provides a solid foundation for exploring the TQE model’s parameter space.

To further strengthen the project and improve its usability, maintainability, and adoption by the wider research community, the following software engineering improvements are recommended:
	
 **1.	Formal Testing Suite** – Introduce automated testing (e.g., with pytest). Unit tests could cover core functions (e.g., anomaly calculators, stability metrics), while integration tests could validate end-to-end runs with a minimal configuration. This would safeguard against regressions and ensure correctness as the code evolves.
	
 **2.	Containerization** – Provide a Dockerfile and/or environment.yml for Conda. Dependencies such as healpy and qutip are known to be challenging to build across platforms. Containerization would package a working environment, allowing any user to reproduce results reliably with a single command.
	
 **3.	Automated API Documentation** – Supplement the high-level README with detailed API documentation. Using tools like Sphinx with the autodoc extension would automatically generate a documentation site from in-code docstrings. This would support collaborators and enable the framework to be extended or integrated as a research library.

## Limitations & Future Work

The TQE Framework should be regarded as a stochastic research prototype rather than a fully developed cosmological simulator. While the pipeline demonstrates the feasibility of modeling emergent laws of physics through energy–information dynamics, several limitations remain:

* **Simplified Physics:** Current implementations of anomalies (e.g., Cold Spot, multipole alignment) and fine-tuning diagnostics use heuristic or simplified metrics. They do not yet capture the full complexity of astrophysical analyses.
* **Computational Scale:** The framework is designed for moderate ensemble sizes. Large-scale runs may encounter memory and runtime constraints without further optimization.
* **Reproducibility Scope:** Results are reproducible within the same environment and dependency versions, but strict bitwise reproducibility across platforms is not guaranteed.
* **Prototype XAI Integration:** SHAP is well integrated, but LIME and other interpretability methods remain in an early-stage prototype form.
* **Predictive Modeling Constraints:** The analysis shows that while classifying the final state of a universe is moderately successful (AUC ≈ 0.65), predicting more nuanced outcomes is challenging with the current feature set. Specifically, predicting the precise *timing* of the lock-in event is difficult (R² ≈ 0.05), as is predicting second-order effects like the performance gain (`delta`) from including the `I` parameter (R² < 0). This highlights that the initial conditions alone may be insufficient to predict the full stochastic evolution.

### Future Work

The TQE Framework, in its current form, is a prototype, and the following extensions and research directions are planned:

* **Refactoring into a Modular Pipeline:** Restructuring the current prototype script into a fully modular, automated pipeline to support large-scale, reproducible experimental campaigns and easier collaboration.
* **Incorporating Additional Physical Models:** Adding more realism to the simulation by integrating more complex physical principles.
* **Implementing GPU Acceleration:** Utilizing GPU and distributed computing to scale up to very large ensembles (>10⁵ universes).
* **Extending the Anomaly Detection Suite:** Adding modules for additional cosmological anomalies, such as Hemispherical Power Asymmetry (HPA).
* **Enhancing the XAI Toolkit:** Integrating different machine learning models (e.g., neural networks) to improve interpretability.
* **Improving Predictive Accuracy:**
    * Developing time-series features (e.g., early-stage volatility, parameter drift) to better predict the lock-in epoch.
    * Running larger ensembles to capture subtle, second-order effects like the fine-tuning deltas.
* **Introducing a Formal Testing Suite:** Implementing unit and integration tests, as well as containerization (e.g., Docker), to ensure robust cross-platform reproducibility.

**Summary:** The TQE Framework is a living research project, open to expansion, scaling, and deeper physical realism.


## Troubleshooting

Below are some common issues and suggested fixes when working with the TQE Framework.

### 1. Installation / Dependency Issues
- **Problem:** `healpy` fails to compile or install on some platforms.  
- **Fix:**  
  - Use Conda: `conda install -c conda-forge healpy`  
  - Or install system dependencies first (Linux):  
    ```bash
    sudo apt-get update && sudo apt-get install libcfitsio-dev
    ```

### 2. Memory Usage
- **Problem:** Running large ensembles (e.g., `NUM_UNIVERSES > 1000`) causes memory errors.  
- **Fix:**  
  - Reduce `NUM_UNIVERSES` in the `MASTER_CTRL` dictionary.  
  - Run in batches (split runs into smaller ensembles).  
  - Use parallel execution with fewer workers (`SKLEARN_N_JOBS`).  

### 3. Reproducibility
- **Problem:** Results differ across machines/platforms.  
- **Fix:**  
  - Ensure same Python version and library versions (use `environment.yml`).  
  - For strict reproducibility, containerization (Docker) is recommended.  

### 4. Long Runtime
- **Problem:** Simulation takes too long for large parameter sweeps.  
- **Fix:**  
  - Enable parallelization with `SKLEARN_N_JOBS = -1`.  
  - Use smaller `LOCKIN_WINDOW` or fewer epochs for testing/debugging.  

### 5. Visualization Errors
- **Problem:** Plots or maps do not display in headless environments (servers, clusters).  
- **Fix:**  
  - Save outputs to file instead of displaying:  
    ```python
    import matplotlib
    matplotlib.use("Agg")
    ```
  - Check `runs/.../figs/` for saved images.


## Contributing Guidelines

This project is a research-grade prototype. Contributions are welcome, especially in the form of bug fixes, documentation improvements, or experimental module extensions.  

If you would like to contribute to the TQE Framework, please follow these steps:

1. **Fork** the repository to your own GitHub account.  
2. **Create a new branch** for your feature or bugfix.  
   ```bash
   git checkout -b feature/my-new-feature
   ```
3.	Commit your changes with clear messages.
4.	Push the branch to your fork.
5.	Open a Pull Request against the main branch of this repository.

**Code Style**

	•	Follow PEP8 style guidelines.
	•	Use meaningful variable and function names.
	•	Include docstrings for all public functions and classes.

**Testing**

	•	Please ensure that existing tests pass before submitting.
	•	If you add new features, include unit tests (pytest is recommended).

By contributing, you agree that your code will be licensed under the same license as this project.


## License

This project is licensed under the MIT License – see the [LICENSE](./LICENSE) file for details.

## Contact

Got questions, ideas, or feedback?  
Drop me an email at **tqe.simulation@gmail.com** 

## TQE pipeline
[TQE_Universe_Simulation_Full_Pipeline.py](./TQE_Universe_Simulation_Full_Pipeline.py)
