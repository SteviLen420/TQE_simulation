# TQE Simulation - Results & Analysis Archive

**Author:** Stefan Len

## About This Directory

This directory contains the complete set of results, raw data, and analysis scripts for my **Theory of the Question of Existence (TQE)** simulation framework. It serves as a comprehensive archive for the various simulation runs and the tools I used to interpret them.

## Navigating This Repository

This repository is organized to provide a clear and transparent overview of the entire TQE research project. All code, data, and documentation are structured as follows, based on the project's root directory:

* ➡️ **Main Documents:**
    * [README.md](./README.md): (This file) The main entry point and guide to the project.
    * [STORY.md](./STORY.md): Explains the scientific motivation and the narrative behind the TQE framework.
    * [TQE_MANUSCRIPT.md](./TQE_MANUSCRIPT.md): The formal scientific paper detailing the theoretical framework and findings.

* ➡️ **Code & Pipelines:**
    * **[TQE_Universe_Simulation_Full_Pipeline.py](./TQE_Universe_Simulation_Full_Pipeline/TQE_Universe_Simulation_Full_Pipeline.py):** Contains the primary Python simulation pipeline used to generate the universe ensembles.
    * **[TQE_Wolfram_Math_Check_Pipeline](./TQE_Wolfram_Math_Check_Pipeline):** Contains the Wolfram Language notebooks used for detailed statistical validation and comparative analysis of the results.

* ➡️ **Simulation Results:**
    * **[E-plus-I_Simulation](./E-plus-I_Simulation):** Each directory with this naming pattern contains the complete output (data, figures, and a summary README) for a specific simulation run where both Energy and Information parameters were active.
    * **[E-Only_Simulation](./E-Only_Simulation):** Each directory with this naming pattern contains the complete output for a specific simulation run where only the Energy parameter was active.

* ➡️ **Project Files:**
    * **[CITATION.cff](./CITATION.cff):** Provides the citation information for this work.
    * **[LICENSE](./LICENSE):** The MIT License under which this project is distributed.

## Methodology & Tools

I conducted the analysis presented here with a focus on rigor and transparency, leveraging a suite of modern tools.

* **Mathematical Validation:** The core statistical calculations and mathematical checks on the raw data were performed and validated using **Wolfram Language**.
* **AI-Assisted Analysis & Interpretation:** I performed the interpretation of the results with the assistance of several large language models to provide a multi-faceted analysis and reduce my own cognitive bias. The models I used include:
    * **Gemini 2.5** (Google)
    * **GPT-5** (OpenAI)
    * An offline, locally-run **DeepSeek-R1-Distill-Qwen-14B** for data processing and prompt-based analysis.

## Current Status & Known Issues

This project is a **research-grade proof-of-concept** and is under my active development. While the core findings are consistent, users should be aware of the following:

* **Work in Progress:** I am still addressing minor bugs in the Python pipeline's file saving system, and not all intended analysis outputs are generated in every run. I am actively working on improving this.
* **Code Synchronization:** I am doing further work to synchronize the code across different modules to achieve even more precise and consistent results.
* **Data Availability:** Despite these minor issues, all available data from the completed simulation runs can be found in the `/DATA/` directory.

## Main Project Documents

For the complete scientific context, theoretical framework, and the narrative behind this project, please refer to the main documents I have written, located in the `/docs/` folder or the root of the repository.
