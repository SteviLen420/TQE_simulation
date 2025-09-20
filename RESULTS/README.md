SPDX-License-Identifier: MIT

Copyright (c) 2025 Stefan Len

# TQE Simulation - Results & Analysis Archive
[![CI](https://github.com/SteviLen420/TQE_simulation/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/SteviLen420/TQE_simulation/actions/workflows/ci.yml) 
[![Python](https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11-blue)](https://www.python.org/doc/)  

**Author:** Stefan Len

## About This Directory

This directory contains the complete set of results, raw data, and analysis scripts for my **Theory of the Question of Existence (TQE)** simulation framework. It serves as a comprehensive archive for the various simulation runs and the tools I used to interpret them.

## Navigating This Repository

This repository is organized to provide a clear and transparent overview of the entire TQE research project. All code, data, and documentation are structured as follows:

* ➡️ **Main Documents (in project root):**
    * [README.md](../README.md): The main entry point and guide to the project.
    * [STORY.md](../STORY.md): Explains the scientific motivation and narrative behind the TQE framework.
    * [TQE_MANUSCRIPT.md](../TQE_MANUSCRIPT.md): The formal scientific paper.

* ➡️ **Code & Pipelines:**
    * [TQE_Universe_Simulation_Full_Pipeline](../TQE_Universe_Simulation_Full_Pipeline/): Contains the primary Python simulation pipeline.
    * [TQE_Wolfram_Math_Check_Pipeline](./TQE_Wolfram_Math_Check_Pipeline/): Contains the Wolfram Language notebooks for analysis. *(Assuming this is a subfolder of results)*

* ➡️ **Simulation Data:**
    * [E-plus-I_Simulation](./E-plus-I_Simulation/): Contains the full output for the `E+I` simulation runs.
    * [E-Only_Simulation](./E-Only_Simulation/): Contains the full output for the `E-only` simulation runs.

* ➡️ **Project Files:**
    * **[CITATION.cff](../CITATION.cff):** Provides the citation information for this work.
    * **[LICENSE](../LICENSE):** The MIT License under which this project is distributed.

## Methodology & Tools

I conducted the analysis presented here with a focus on rigor and transparency, leveraging a suite of modern tools.

* **Mathematical Validation:** The core statistical calculations and mathematical checks on the raw data were performed and validated using **Wolfram Language**.
* **AI-Assisted Analysis & Interpretation:** I performed the interpretation of the results with the assistance of several large language models to provide a multi-faceted analysis and reduce my own cognitive bias. The models I used include:
    * **Gemini 2.5** (Google)
    * **GPT-5** (OpenAI)
    * An offline, locally-run **DeepSeek-R1-Distill-Qwen-14B** for data processing and prompt-based analysis.
* **Reproducibility:** All random seeds and configuration files are archived for reproducibility.
 
## How to Reproduce the Analysis

The quantitative results presented in the manuscript can be reproduced using the tools and data in this archive.

1.  **Data:** The raw `.csv` files from the simulations are located in the `/E-plus-I_Simulation/DATA/` and `/E-Only_Simulation/DATA/` subdirectories.
2.  **Analysis Script:** The Wolfram Language notebook located in the `/RESULTS/` folder (`TQE_Wolfram_Math_Check_Pipeline.nb`) was used to perform the final comparative analysis.
3.  **Execution:** To reproduce the summary tables, open the notebook, set the input directories to point to the respective data folders, and evaluate the notebook.

## Current Status & Known Issues

This project is a **research-grade proof-of-concept** and is under my active development. While the core findings are consistent, users should be aware of the following:

* **Work in Progress:** I am still addressing minor bugs in the Python pipeline's file saving system, and not all intended analysis outputs are generated in every run. I am actively working on improving this.
* **Code Synchronization:** I am doing further work to synchronize the code across different modules to achieve even more precise and consistent results.
* **Data Availability:** Despite these minor issues, all available data from the completed simulation runs can be found in the `/DATA/` directory.

## License
This project is licensed under the MIT License – see the [LICENSE](./LICENSE) file for details.

## Contact

Got questions, ideas, or feedback?  
Drop me an email at **tqe.simulation@gmail.com** 

[RESULTS](../RESULTS)
