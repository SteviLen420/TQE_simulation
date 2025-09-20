SPDX-License-Identifier: MIT

Copyright (c) 2025 Stefan Len

[![CI](https://github.com/SteviLen420/TQE_simulation/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/SteviLen420/TQE_simulation/actions/workflows/ci.yml)  
[![Python](https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11-blue)](https://www.python.org/doc/)  

# Theory of The Question of Existance (TQE)

**Title: A Modular, Reproducible Pipeline for Monte Carlo Simulation of Universe Evolution from Energy-Information Principles**

**Author**: Stefan Len

## About The Project

The TQE Framework is a computational pipeline designed to investigate the hypothesis that the fundamental laws of physics emerge from a primitive interplay of Energy (E) and Information (I). It provides a complete, end-to-end environment for conducting Monte Carlo simulations of universe ensembles, analyzing their cosmological properties, and using Explainable AI (XAI) to uncover the underlying principles.

This repository contains the full source code and documentation for the framework.

## Navigating This Repository

To understand the project, I recommend exploring the following files:

➡️ [STORY](STORY.md) : Start here for the scientific motivation, the core concepts, and the narrative behind the TQE framework. This explains the "WHY" of the project.

➡️ [README](./TQE_Universe_Simulation_Full_Pipeline/README.md) : This is the main technical documentation. It contains detailed instructions on installation, configuration, usage, and the complete API reference. This explains the "HOW" of the project.

## Quick Start

To get the project up and running locally, follow these steps.

**1. Clone the repository:**
     ```bash
     git clone [https://github.com/SteviLen420/TQE_simulation.git](https://github.com/SteviLen420/TQE_simulation.git)
     ```
**2. Navigate to the repository directory:**
    ```bash
    cd TQE_Universe_Simulation_Full_Pipeline
    ```
**3. Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
For detailed execution and configuration instructions, please refer to the technical [README](./TQE_Universe_Simulation_Full_Pipeline/README.md) file in the pipeline directory.

## How to Cite

If you use this software in your research, please cite it. The repository includes a CITATION.cff file with all the necessary metadata.

**You can use the following BibTeX entry for convenience:**

```bash
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

## A Note on Methodology and AI Collaboration

This project is the result of an independent intellectual exploration by a self-taught enthusiast, not a formally trained academic scientist. The primary goal was to translate a conceptual idea about the origins of physical laws into a testable computational framework, using the most advanced tools available to an independent researcher.

To achieve this, I relied heavily on a suite of modern AI models and specialized tools as collaborative partners.

### The AI and Tool Stack

Full transparency about the tools used is critical to understanding the project's methodology. The workflow was divided into three core activities:

**1. Code Generation and Architectural Design**
The pipeline's code was developed through a combination of high-level architectural guidance and iterative, hands-on implementation.

* **Google's Gemini 2.5 Pro & OpenAI's GPT-5**: Served as high-level architectural consultants, assisting in the overall pipeline design, complex refactoring, and the implementation of advanced features.

* **DeepSeek-R1:14B (via Ollama)**: Employed as a local, offline model for the core iterative development loop, including real-time code generation, debugging, and algorithm refinement.

**2. Data Analysis and Mathematical Validation**
The raw outputs of the simulation were processed and validated using a combination of general and specialized AI.

* **Gemini 2.5 Pro & DeepSeek-R1:14B**: Used for the primary analysis of simulation outputs, helping to structure statistical tests and interpret the results from the XAI modules.

* **Wolfram|One & Wolfram|Alpha**: Utilized for the rigorous validation of the mathematical formalisms and equations presented in the documentation, ensuring their correctness and internal consistency.

**3. Scientific Context and Documentation**
Placing the project in an academic context and documenting it was a critical step.

* **IBM Watson**: Leveraged for large-scale analysis of scientific literature to identify broader research trends and knowledge gaps, ensuring the TQE hypothesis was positioned correctly within the current scientific landscape.

* **Specialized GPTs (Wolfram GPT, SciSpace, Scholar GPT)**: Employed to perform targeted literature searches, contextualizing the TQE hypothesis against existing scientific research and specific papers.

* **Gemini 2.5 Pro & GPT-5**: Assisted in drafting the comprehensive documentation, abstracts, and formal descriptions found throughout this repository.

## Limitations and Future Work

Given this unique, AI-assisted methodology, the TQE Framework should be viewed as a research-grade **proof-of-concept**. It demonstrates the feasibility of this line of inquiry, but the findings require further investigation, validation, and critique by the scientific community. To facilitate this process and guide the project's evolution, the following key areas are planned for future development:

* **Computational Scaling**: To explore the parameter space more exhaustively, future work will focus on optimizing the pipeline for high-performance computing (HPC) environments. This will enable simulations with significantly larger ensembles of universes (>10^5) and longer evolutionary epochs, providing greater statistical power to the results.

* **Expansion of Anomaly Detectors**: The current framework includes detectors for the CMB Cold Spot and the Axis of Evil. The pipeline is designed to be extensible, and future versions will incorporate additional anomaly modules, such as the Hemispherical Power Asymmetry (HPA) and other statistical measures observed in cosmological data.

* **Architectural Refactoring**: The current single-script implementation is being actively refactored into a fully modular framework. This will improve maintainability, facilitate easier collaboration, and allow researchers to swap or modify individual components (e.g., the simulation core, an analysis module) without altering the rest of the pipeline.

It is my hope that this project might serve as a novel starting point or a useful tool for researchers in theoretical physics and cosmology. Contributions, feedback, and collaborations in all these areas are warmly welcomed.

## License
This project is licensed under the MIT License – see the [LICENSE](./LICENSE) file for details.

## Contact

Got questions, ideas, or feedback?  
Drop me an email at **tqe.simulation@gmail.com** 

## TQE pipeline
[TQE_Universe_Simulation_Full_Pipeline.py](./TQE_Universe_Simulation_Full_Pipeline/TQE_Universe_Simulation_Full_Pipeline.py)
