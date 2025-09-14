[![CI](https://github.com/SteviLen420/TQE_simulation/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/SteviLen420/TQE_simulation/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11-blue)](https://www.python.org/doc/)
[![CI](https://img.shields.io/github/actions/workflow/status/SteviLen420/TQE_simulation/ci.yml?branch=main)](https://github.com/SteviLen420/TQE_simulation/actions)
[Read the story behind the project →](STORY.md)

#  TQE (E,I) UNIVERSE SIMULATION PIPELINE

# Author: Stefan Len


**Title: The TQE Framework: A Modular, Reproducible Pipeline for Monte Carlo
Simulation of Universe Evolution from Energy-Information Principles**


**Abstract**


The TQE Framework is a novel computational pipeline designed to investigate the hypothesis that the fundamental laws of physics are not axiomatic but emerge from a more primitive interplay of Energy (E) and Information (I). This framework provides a complete, end-to-end environment for conducting Monte Carlo simulations of universe ensembles, allowing for the systematic exploration of a vast parameter space of initial conditions. The core of the pipeline is a multi-stage simulation that models the lifecycle of a universe: an initial, pre-collapse phase where physical laws exist in a quantum-like superposition; a probabilistic "law lock-in" event, where a stable set of physical constants is selected; and a subsequent expansion phase that generates large-scale structures. A comprehensive analysis suite processes the outputs of these simulations to generate cosmological observables analogous to the Cosmic Microwave Background (CMB) and to perform targeted scans for known CMB anomalies, such as the Cold Spot and low-ℓ alignments. A key contribution of this work is the integration of an Explainable AI (XAI) module, which uses machine learning models and interpretability techniques like SHAP to build a predictive and explanatory bridge between the initial (E,I) conditions and the final, observable characteristics of a simulated universe. The TQE Framework is architected for modularity, scalability, and, most critically, full computational reproducibility, making it a powerful tool for theoretical cosmology research.


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


**Computational Framework & Methodology**


The TQE Framework is engineered as a robust, multi-stage computational pipeline designed for the systematic investigation of emergent physical laws. Its architecture prioritizes modularity, configuration-driven execution, and strict reproducibility to meet the rigorous demands of scientific research.


**High-Level Architecture**


The framework's workflow is orchestrated by a central YAML configuration file, MASTER_CTRL.yml. This configuration-as-code approach allows entire experimental campaigns, including parameter sweeps and analysis settings, to be defined and archived within a single, human-readable text file. This design not only simplifies the execution of complex simulations but also forms the bedrock of the framework's reproducibility.
The pipeline is divided into four distinct, logically sequential stages:
1. 01-Generation: This initial stage reads the simulation parameters from MASTER_CTRL.yml and generates the initial conditions for an entire ensemble of universes. Each universe is defined by its starting Energy (E) and Information (I) values, drawn from specified statistical distributions.
2. 02-Simulation: This is the computational core of the framework. It takes the initial conditions for each universe and executes the evolution algorithm, progressing through the pre-collapse, law lock-in, and expansion phases. This stage is computationally intensive and designed for parallel execution.
3. 03-Analysis: Following the simulation, this stage performs post-processing on the raw output data. It calculates high-level cosmological observables, generates CMB-like sky maps, and executes a suite of diagnostic tests to score universes for fine-tuning and scan for cosmological anomalies.
4. 04-Interpretation: The final stage leverages Explainable AI (XAI) to synthesize the results from the entire ensemble. It trains machine learning models to predict simulation outcomes based on initial conditions, providing deep insights into the causal relationships within the TQE model.
This modular structure is a critical design feature, enabling researchers to re-run specific parts of the pipeline without having to repeat upstream computations. For instance, one can re-analyze existing simulation data with a new anomaly detection algorithm by invoking only the analysis stage, a significant efficiency gain in research workflows.


**Reproducibility by Design: The Seeding Hierarchy**


To ensure full computational determinism, the framework implements a sophisticated two-tiered seeding hierarchy. This system guarantees that any simulation campaign can be reproduced bit-for-bit, a non-negotiable requirement for verifiable scientific claims.
1. Master Seed: A single master_seed is defined in the MASTER_CTRL.yml configuration file. This seed initializes a master pseudo-random number generator (PRNG) at the beginning of a run.
2. Per-Universe Seeds: The master PRNG is used to deterministically generate a unique universe_seed for each of the n_universes in the ensemble. This is achieved through a robust function, such as universe_seed_i = f(master_seed, i), which ensures that the seed for universe $i$ is independent of the total number of universes being simulated.
Each universe_seed is then used to seed the local PRNG responsible for all stochastic processes within that single universe's simulation. This hierarchical design provides two layers of reproducibility. A researcher can reproduce the entire ensemble of results by simply re-using the master_seed. Furthermore, if a single universe exhibits particularly interesting behavior, it can be isolated and its evolution reproduced exactly by using its specific universe_seed, without the need to re-run the entire, potentially massive, ensemble. This capability is invaluable for debugging, detailed analysis, and validating extraordinary results.


**The Simulation Core: A Universe's Lifecycle**


The evolution of each universe within the simulation follows a distinct lifecycle, modeling the transition from a state of physical indeterminacy to a cosmos with stable, fixed laws.
* Initialization: Each simulation instance begins with a set of initial conditions for total Energy (E) and Information (I), sampled from distributions defined in the configuration. These two scalar values are the fundamental inputs to the TQE model.
* Pre-Collapse Phase: The universe enters a state of "quantum-like fluctuation." In this phase, the set of physical laws, represented by a state vector X, is not fixed. Instead, it exists in a dynamic superposition of possibilities, evolving stochastically. The presence of the qutip (Quantum Toolbox in Python) library as an optional dependency suggests this phase is not merely conceptual but can be computationally modeled using the formalisms of quantum mechanics, such as state vectors evolving under a Hamiltonian parameterized by E and I.
* Law Lock-In: This is the pivotal event in the simulation. The universe transitions from the fluctuating pre-collapse phase to a state with a single, stable set of physical laws, Xfinal​. This transition is probabilistic and governed by a stability_threshold parameter. The simulation monitors the variance or fluctuation of the law vector X(t). When this fluctuation remains below the specified threshold for a defined number of consecutive epochs, the laws are considered "locked-in." The epoch at which this occurs, lock_epoch, is a critical output variable.
* Expansion Phase: Once the laws are locked-in, the universe's subsequent evolution is deterministic, governed by the fixed parameter set Xfinal​. This phase simulates the large-scale expansion and structure formation that produce the final cosmological observables passed to the analysis suite.


**The Analysis & Diagnostics Suite**


This suite of modules quantifies the outcomes of the simulation core, translating raw data into scientifically meaningful metrics and visualizations.
* CMB-like Map Generation: The framework uses the healpy library to project the final state of the simulated universe onto a spherical grid, creating a sky map analogous to the Cosmic Microwave Background. This allows for direct visual and statistical comparison with observational data.
* Fine-Tuning Diagnostics: A set of metrics is computed to score how "fine-tuned" a universe's locked-in laws (Xfinal​) are for the emergence of complexity (e.g., structure formation, stable chemistry). This allows for a quantitative classification of universes in the ensemble as "barren," "habitable," or other categories, providing a means to study the anthropic principle from a generative standpoint.
* Anomaly Scanning: The pipeline is explicitly designed to search for patterns in the generated maps that correspond to statistically significant anomalies observed in our own CMB data. The focus on these specific anomalies suggests the TQE framework is positioned as a potential explanatory model for phenomena that are in tension with the standard ΛCDM model of cosmology. The targeted anomalies include:
   * The CMB Cold Spot: An unusually large and cold region of the sky.
   * Low-ℓ Multipole Alignments: The unexpected alignment of the quadrupole (ℓ=2) and octopole (ℓ=3) moments of the CMB, sometimes referred to as the "Axis of Evil."
   * The Low-ℓ Alignment Correlation (LLAC): Correlations between the low multipole alignments.
   * The Hemispherical Power Asymmetry (HPA): A statistically significant difference in the power of temperature fluctuations between two opposing hemispheres of the sky.


**Explainable AI (XAI) for Cosmological Interpretation**


The most advanced component of the framework is its XAI module, which transforms the massive dataset generated by the simulation ensemble into scientific understanding.
* Models: The primary machine learning model employed is a RandomForest classifier or regressor. This model is chosen for its high performance on tabular data and its relative interpretability compared to deep neural networks.
* Targets: The model is trained on the full ensemble of simulated universes. The features are the initial conditions (E, I) and key emergent properties (e.g., lock_epoch). The target variable is a simulation outcome of scientific interest, such as the fine_tuning_score or a boolean flag indicating the presence of a specific anomaly like the Cold Spot.
* Interpretability Methods: The framework integrates state-of-the-art XAI libraries to deconstruct the trained model's logic:
   * SHAP (SHapley Additive exPlanations): Used to determine global feature importance. SHAP values can answer questions like, "Overall, what is the most influential factor in determining whether a universe becomes fine-tuned?" This provides a ranked list of the drivers of specific outcomes across the entire parameter space.
   * LIME (Local Interpretable Model-agnostic Explanations): Used to explain individual predictions. LIME can answer questions like, "Why did this specific simulated universe develop a Hemispherical Power Asymmetry?" It provides a local, case-by-case explanation, highlighting the feature values that pushed a particular prediction.
The integration of this XAI workflow creates a powerful feedback loop for scientific discovery. It elevates the framework from a descriptive tool that merely generates data to a prescriptive one that generates hypotheses. By revealing the quantitative relationships between initial conditions and final observables, the XAI module allows researchers to formulate new, precise, and testable hypotheses about the underlying physics of the TQE model, dramatically accelerating the process of scientific inquiry.


**Results Manifesting & Run Folder Structure**


To ensure organized and traceable results, every execution of the pipeline creates a unique, timestamped output directory, for example: RUN_DIR/run_20231027_153000/. This directory contains a standardized set of subfolders:
* logs/: Contains detailed logs of the pipeline's execution.
* data/: Stores the raw, unprocessed output from the simulation stage for each universe.
* analysis/: Contains processed data, such as calculated metrics, anomaly statistics, and summary tables (e.g., results.csv).
* FIG_DIR/: The designated folder for all generated plots and figures, such as CMB-like maps and SHAP summary plots.
The framework also supports mirroring all outputs to a secondary location, a feature useful for creating backups or for working on compute clusters with separate permanent storage.
________________


## Mathematical Formalism
The TQE Framework is grounded in a set of mathematical principles that govern the evolution of simulated universes. The core concepts are defined below.

### Key State Variables
- **E**: The total energy of the initial state. A scalar value, $E \in \mathbb{R}^+$.
- **I**: The total information content of the initial state. A scalar value, $I \in \mathbb{R}^+$.
- **X(t)**: A time-varying state vector representing the set of physical laws during the pre-collapse phase. $X(t) \in \mathbb{R}^n$, where *n* is the number of fundamental parameters (e.g., coupling constants, particle masses) being simulated.
- **X\_{final}**: The final, locked-in vector of physical laws, $X\_{final} = X(t\_{lock})$, where $t\_{lock}$ is the lock-in epoch.

### Update and Selection Rules

**Coupling Function:**  
The initial state of the physical laws is centered around a value determined by the initial energy and information through a coupling function:

$$
X_{mean} = f(E,I)
$$

This function defines the fundamental hypothesis of the TQE model.

**Pre-Collapse Dynamics:**  
The fluctuation of the law vector $X(t)$ around its mean can be modeled as a stochastic process. A simplified representation using a Langevin equation is:

$$
\frac{dX(t)}{dt} = -\nabla V(X) + \sqrt{2D}\,\eta(t)
$$

Here, $V(X)$ is a potential landscape whose shape is determined by the initial conditions $E$ and $I$.  
The term $-\nabla V(X)$ drives the system towards local minima (stable law configurations),  
$D$ is a diffusion coefficient representing the magnitude of quantum-like fluctuations,  
and $\eta(t)$ is a Gaussian white noise term.

**Lock-In Criterion:**  
The transition to a stable set of laws occurs at epoch $t_{lock}$ if the system's stability metric $S(t)$ remains below a predefined threshold $\epsilon$ for a duration of $\Delta t$ epochs.  
A common choice for the stability metric is the trace of the covariance matrix of the state vector over the recent time window:

$$
S(t) = \mathrm{Tr}\left(\mathrm{Cov}(X(t'))\right)_{t' \in} < \epsilon
$$

### Probabilistic and Stability Definitions
The probability of the system locking into a specific configuration of laws $X$ is related to the depth of the corresponding well in the potential landscape $V(X)$.  
In analogy to statistical mechanics, this can be expressed as:

$$
P(\text{lock-in} \mid X) \propto \exp\left(-\frac{V(X)}{kT_{eff}}\right)
$$

where $T_{eff}$ is an effective temperature of the system during the pre-collapse phase, representing the energy available for fluctuations.


### Anomaly and Scoring Equations
The analysis suite uses standard statistical estimators to quantify cosmological observables and anomalies.

**Fine-Tuning Score ($\mathcal{F}$):**  
A heuristic score to quantify the "habitability" of a universe with laws  

$$
X_{final} = \{x_1, x_2, \dots, x_n\}
$$

This is often modeled as a multivariate Gaussian function centered on known "life-friendly" values $(x_{i,\text{target}})$, with widths $(\sigma_i)$ defining the tolerance for each parameter:

$$
\mathcal{F}(X_{final}) = \prod_{i=1}^n \exp\!\left(-\frac{(x_i - x_{i,\text{target}})^2}{2\sigma_i^2}\right)
$$

**Hemispherical Power Asymmetry (HPA):**  
The asymmetry parameter $A$ is calculated from the angular power spectra $(C_\ell)$ computed independently on two opposing hemispheres of the sky map (North, $N$, and South, $S$):  

$$
A = \frac{\sum_{\ell=\ell_{min}}^{\ell_{max}} \big(C_\ell^N - C_\ell^S\big)}
         {\sum_{\ell=\ell_{min}}^{\ell_{max}} \big(C_\ell^N + C_\ell^S\big)}
$$

This value quantifies the normalized difference in power over a specific range of angular scales (multipoles $\ell$).

			
________________


Assessment of the Codebase




Maturity Level


The TQE Framework is assessed as Research-Grade. The codebase demonstrates a high degree of architectural sophistication, moving well beyond a simple prototype. Key features such as the modular pipeline, configuration-driven design, a robust reproducibility mechanism, and the integration of an advanced XAI analysis suite are hallmarks of a mature tool built for serious scientific inquiry. However, it likely lacks the comprehensive test suites, extensive error handling, and polished user-facing APIs that would characterize a production-grade system intended for a non-expert audience.


Ratings


* Rigor: 9/10. The methodological foundations are exceptionally strong. The two-tiered seeding hierarchy provides a state-of-the-art solution for reproducibility in stochastic simulations. The explicit focus on testing against known cosmological anomalies and the use of a formal XAI loop for interpretation reflect a deep commitment to scientific rigor.
  
* Clarity: 7/10. The modular architecture significantly enhances clarity at a high level. However, as is common with specialized research code, the internal logic of the core simulation algorithms and mathematical formalisms is likely dense and may present a steep learning curve for those outside the immediate domain of theoretical cosmology.
  
* Reproducibility: 10/10. The framework's design makes reproducibility a first-class citizen. The combination of configuration-as-code via YAML files and the deterministic hierarchical seeding mechanism provides a complete and robust solution for ensuring that all results can be independently verified and reproduced, which is the gold standard for computational science.


**Justification and Concrete Suggestions for Improvement**


The framework stands out for its exceptional scientific and methodological design, particularly its unwavering focus on reproducibility and its innovative use of XAI to accelerate hypothesis generation. The architecture is thoughtfully constructed for large-scale computational experiments, allowing for the systematic exploration of the TQE model's parameter space. The primary areas for improvement lie in software engineering best practices that would enhance its robustness, ease of use for collaborators, and long-term maintainability.

Three concrete suggestions to further elevate the project are:
            1. Implement a Formal Testing Suite: Introduce the pytest framework to the repository. This would involve creating a tests/ directory with unit tests for critical, isolated functions (e.g., anomaly calculators, stability metrics) to verify their correctness. Additionally, integration tests could be added to run the minimal demo.yml profile and assert that the expected output files and directories are created correctly. A formal test suite would safeguard against regressions during future development and provide an executable specification of the code's intended behavior.
            2. Containerize the Environment: Provide a Dockerfile and/or a Conda environment.yml file in the repository's root. The framework's dependencies, particularly healpy and qutip, are known to be difficult to build from source due to system-level requirements. Containerization would completely solve this problem by packaging the exact, working software environment. This would guarantee that any user on any machine can run the code with a single command, making the framework's results truly and effortlessly reproducible.
            3. Generate Automated API Documentation: Supplement this high-level README.md with detailed, low-level API documentation. This can be achieved by using a tool like Sphinx with the autodoc extension, which automatically parses docstrings from the Python source code to generate a full HTML documentation website. This would be an invaluable resource for other researchers wishing to understand the codebase in detail, extend the framework with new modules, or use its components as a library in their own projects.
