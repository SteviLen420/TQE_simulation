#  TQE UNIVERSE SIMULATION PIPELINE

# Author: Stefan Len


**Title: The TQE Framework: A Modular, Reproducible Pipeline for Monte Carlo
Simulation of Universe Evolution from Energy-Information Principles**


**Abstract**


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
@software{TQE_Simulation_2025,
  author = {Len, Stefan},
  title = {{TQE (E,I) Universe Simulation Pipeline}},
  year = {2025},
  publisher = {GitHub},
  version = {1.0.0},
  url = {[https://github.com/SteviLen420/TQE_simulation](https://github.com/SteviLen420/TQE_simulation)}
}
```
________________

## Installation & Environment Setup

The TQE Framework requires **Python 3.9+** and several scientific libraries.  
We recommend setting up a dedicated virtual environment to ensure reproducibility.

### Using pip
```bash
git clone https://github.com/your_username/TQE_Framework.git
cd TQE_Framework
python -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate
pip install -r requirements.txt 
```
### Using Conda
```bash
git clone https://github.com/your_username/TQE_Framework.git
cd TQE_Framework
conda env create -f environment.yml
conda activate tqe_env
```
### Optional Dependencies
	•	**qutip** – enables quantum-mechanical modeling of pre-collapse dynamics
	•	**healpy** – required for generating CMB-like sky maps
	•	**shap, lime** – required for Explainable AI (XAI) analysis

💡 If installation issues occur with healpy or qutip, we recommend using Conda, as it handles binary dependencies more reliably than pip.

## Quickstart / Usage Example

Once the environment is set up, you can run the TQE Framework directly with the default configuration file.

### Run with default configuration
```bash
python TQE_Universe_Simulation_Full_Pipeline.py --config MASTER_CTRL.yml
```
### Minimal example
```bash
# Example: run the simulation programmatically
from TQE_Universe_Simulation_Full_Pipeline import run_simulation

# Load default configuration
config = "MASTER_CTRL.yml"

# Run simulation
results = run_simulation(config)

# Inspect output
print(results.head())
```
Output
	
 •	All results are stored in a timestamped directory inside runs/ (e.g., runs/TQE_Run_20250914_123000/).
	
 •	Subdirectories contain:
	
**maps/** → CMB-like sky maps
 	
**diag/** → stability curves, anomaly scans
 
**xai/** → Explainable AI outputs (SHAP/LIME)
 
**logs/** → runtime information and metadata

## Configuration Parameters

The behavior of the TQE Framework is controlled by a central YAML configuration file (`MASTER_CTRL.yml`).  
The table below lists the most important parameters:

| Parameter            | Section      | Type    | Default       | Description                                                                 |
|----------------------|-------------|---------|---------------|-----------------------------------------------------------------------------|
| `run_name`           | meta        | string  | `tqe_run`     | Base name for the output directory.                                         |
| `master_seed`        | meta        | int     | `42`          | The master seed for reproducibility of the entire experiment.               |
| `n_universes`        | simulation  | int     | `100`         | Number of universes to simulate in the ensemble.                            |
| `e_dist`             | simulation  | dict    | `{...}`       | Parameters for the initial Energy distribution (e.g., mean, std).           |
| `i_dist`             | simulation  | dict    | `{...}`       | Parameters for the initial Information distribution.                        |
| `stability_threshold`| simulation  | float   | `1e-5`        | The stability metric threshold required for law lock-in.                    |
| `run_hpa_scan`       | analysis    | bool    | `true`        | Enable/disable the Hemispherical Power Asymmetry (HPA) anomaly analysis.    |
| `xai_target`         | xai         | string  | `fine_tuning_score` | Target variable for the XAI model to predict.                         |
| `n_jobs`             | meta        | int     | `-1`          | Number of parallel processes to use (`-1` = all available cores).           |
 


## Computational Framework & Methodology

The TQE Framework is structured as a multi-stage computational pipeline for the systematic investigation of emergent physical laws. Its design emphasizes configuration-driven execution and reproducibility, with modularity implemented at the conceptual level and partially realized in the current prototype.

### High-Level Architecture

The workflow is orchestrated by a central YAML configuration file, MASTER_CTRL.yml. This configuration-as-code approach enables experimental campaigns—including parameter sweeps and analysis settings—to be defined and archived in a single, human-readable text file. This provides a foundation for reproducibility and transparent experimental design.

The pipeline is organized into four sequential stages:

1.	**Generation** – Reads simulation parameters from MASTER_CTRL.yml and generates initial conditions for an ensemble of universes, each defined by Energy (E) and Information (I) values drawn from statistical distributions.
   
2.	**Simulation** – The computational core of the framework. It evolves universes through pre-collapse, law lock-in, and expansion phases. This stage is computationally intensive and includes preliminary support for parallel execution.
   
3.	**Analysis** – Performs post-processing on raw outputs, including calculation of cosmological observables, generation of CMB-like sky maps, and execution of diagnostic tests to score universes for fine-tuning and scan for selected anomalies.
   
4.	**Interpretation** – Applies Explainable AI (XAI) to synthesize ensemble results. Machine learning models are trained to predict outcomes from initial conditions, providing insights into causal relationships within the TQE model.

This staged structure conceptually supports re-running individual components (e.g., re-analyzing simulation data with a new anomaly detector) without repeating upstream steps, although the current implementation realizes this in a more streamlined, script-based form.



## Reproducibility by Design: The Seeding Hierarchy

To ensure computational reproducibility, the framework implements a two-tiered seeding hierarchy. This design provides deterministic outcomes within a fixed software environment, supporting verifiable and repeatable scientific workflows.

•	**Master Seed** – A single master_seed is defined in the MASTER_CTRL.yml configuration file. This initializes a master pseudo-random number generator (PRNG) at the beginning of a run.
 
•	**Per-Universe Seeds** – The master PRNG is used to deterministically generate a unique universe_seed for each of the n_universes in the ensemble. This ensures that each universe’s stochastic processes are initialized independently and reproducibly.

This hierarchical system provides two levels of control. Re-using the same master_seed allows an entire ensemble to be reproduced, while selecting an individual universe_seed enables the exact reproduction of a single universe’s evolution without re-running the full ensemble. This capability is especially valuable for debugging, targeted analysis, and validation of noteworthy cases. While the current implementation guarantees reproducibility under consistent library versions and environments, strict cross-platform bit-level determinism may vary depending on the underlying PRNG implementation.



## The Simulation Core: A Universe’s Lifecycle

The evolution of each universe in the simulation follows a staged lifecycle, modeling the transition from an indeterminate system to one governed by stable, fixed laws.
•	**Initialization** – Each run begins with initial Energy (E) and Information (I) values, sampled from distributions defined in the configuration file. These scalar quantities are the fundamental inputs to the TQE model.
 
•	**Pre-Collapse Phase** – In this phase, the effective physical laws (represented by a state vector X) fluctuate stochastically around values determined by the initial conditions. Conceptually, this is interpreted as a “quantum-like” regime where the laws are not yet fixed. The implementation models these fluctuations through random perturbations, with optional use of qutip for more advanced experiments.

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

The XAI module transforms the dataset generated by the simulation ensemble into interpretable scientific insights.

•	**Models** – The primary machine learning model employed is a RandomForest classifier or regressor, chosen for its performance on tabular data and its interpretability compared to deep neural networks.
 
•	**Targets** – Models are trained on ensembles of simulated universes. Features include initial conditions (E, I) and emergent properties (e.g., lock_epoch), while target variables include outcomes such as the fine_tuning_score or anomaly flags (e.g., presence of a Cold Spot).
 
•	**Interpretability Methods** – The framework integrates explainability tools to analyze the trained models:
 
•	**SHAP (SHapley Additive exPlanations)** – used to quantify global feature importance. SHAP values answer questions such as: “What factors most influence whether a universe becomes fine-tuned?”
 
•	**LIME (Local Interpretable Model-agnostic Explanations)** – support for local explanations is included at a prototype stage. LIME can provide case-by-case insights (e.g., explaining why a specific simulated universe exhibits an anomaly), though its integration is less developed than SHAP.

By combining ensemble simulation with XAI, the framework goes beyond description and supports hypothesis generation. It highlights the quantitative relationships between initial conditions and final observables, enabling new, testable ideas about the underlying physics of the TQE model.


## Results Manifesting & Run Folder Structure

To ensure organized and traceable results, each execution of the pipeline generates a unique, timestamped output directory (e.g., RUN_DIR/run_20231027_153000/). Within this directory, the following standardized subfolders are created:

•	**logs/** – detailed logs of the pipeline’s execution.

•	**data/** – raw, unprocessed outputs from the simulation stage for each universe.

•	**analysis/** – processed results, including metrics, anomaly statistics, and summary tables (e.g., results.csv).

•	**figures/** – generated plots and figures, such as CMB-like maps and SHAP summary plots.

This structure ensures that results remain reproducible and easy to navigate. The framework design also anticipates the ability to mirror outputs to a secondary location (e.g., for compute cluster storage), though this functionality is not yet fully implemented in the current prototype.



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
P(\text{lock-in} \mid X) \propto \exp!\left(-\frac{V(X)}{kT_{eff}}\right)
$$

This represents a guiding analogy rather than a fully implemented feature.

Fine-Tuning Score
The habitability score is implemented as a multivariate Gaussian around life-friendly target values:

$$
\mathcal{F}(X_{final}) = \prod_{i=1}^n \exp!\left(-\frac{(x_i - x_{i,\text{target}})^2}{2\sigma_i^2}\right)
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
	
 •	**Reproducibility: 8/10** – Reproducibility is a central design feature, with YAML configuration and a deterministic seeding mechanism ensuring consistency across runs in the same environment. Full cross-platform bit-level reproducibility, however, is not yet guaranteed.


## Justification and Suggestions for Improvement

The framework distinguishes itself through its strong scientific motivation, emphasis on reproducibility, and the novel integration of explainable AI. Its architecture is well-suited for large-scale computational experiments and provides a solid foundation for exploring the TQE model’s parameter space.

To further strengthen the project and improve its usability, maintainability, and adoption by the wider research community, the following software engineering improvements are recommended:
	
 **1.	Formal Testing Suite** – Introduce automated testing (e.g., with pytest). Unit tests could cover core functions (e.g., anomaly calculators, stability metrics), while integration tests could validate end-to-end runs with a minimal configuration. This would safeguard against regressions and ensure correctness as the code evolves.
	
 **2.	Containerization** – Provide a Dockerfile and/or environment.yml for Conda. Dependencies such as healpy and qutip are known to be challenging to build across platforms. Containerization would package a working environment, allowing any user to reproduce results reliably with a single command.
	
 **3.	Automated API Documentation** – Supplement the high-level README with detailed API documentation. Using tools like Sphinx with the autodoc extension would automatically generate a documentation site from in-code docstrings. This would support collaborators and enable the framework to be extended or integrated as a research library.
