# TQE Simulation - Results & Analysis Archive

**Author:** Stefan Len

---
## About This Directory

This directory contains the complete set of results, raw data, and analysis scripts for the **Theory of the Question of Existence (TQE)** simulation framework. It serves as a comprehensive archive for the various simulation runs and the tools used to interpret them.

## Navigating This Directory

This archive is organized into several key subdirectories to ensure clarity and reproducibility. All essential components of the analysis can be found within this structure.

* ➡️ **[/TQE_Universe_Simulation_Full_Pipeline/](./TQE_Universe_Simulation_Full_Pipeline/):** Contains the primary Python simulation code. The `README.md` within this folder provides detailed usage instructions for running new simulations.
* ➡️ **[/analysis/](./analysis/):** Contains the Wolfram Language notebooks used for detailed statistical validation and comparative analysis of the simulation results.
* ➡️ **[/DATA/](./DATA/):** Contains the raw and processed `.csv` and `.json` output files from the various simulation runs, organized into subfolders for each run type (e.g., `E+I` and `E-only`).
* ➡️ **[/docs/](./docs/):** Contains supporting documentation, including the main scientific manuscript (`TQE_MANUSCRIPT.md`) and the project's narrative (`STORY.md`).

## Methodology & Tools

The analysis presented here was conducted with a focus on rigor and transparency, leveraging a suite of modern tools.

* **Mathematical Validation:** The core statistical calculations and mathematical checks on the raw data were performed and validated using **Wolfram Language**.
* **AI-Assisted Analysis & Interpretation:** The interpretation of the results was conducted with the assistance of several large language models to provide a multi-faceted analysis and reduce cognitive bias. The models used include:
    * **Gemini 2.5** (Google)
    * **GPT-5** (OpenAI)
    * An offline, locally-run **DeepSeek-R1-Distill-Qwen-14B** for data processing and prompt-based analysis.

## Current Status & Known Issues

This project is a **research-grade proof-of-concept** and is under active development. While the core findings are consistent, users should be aware of the following:

* **Work in Progress:** There are still minor bugs being addressed in the Python pipeline's file saving system, and not all intended analysis outputs are generated in every run. I am actively working on improving this.
* **Code Synchronization:** Further work is being done to synchronize the code across different modules to achieve even more precise and consistent results.
* **Data Availability:** Despite these minor issues, all available data from the completed simulation runs can be found in the `/DATA/` directory.

## Main Project Documents

For the complete scientific context, theoretical framework, and narrative behind this project, please refer to the main documents located in the `/docs/` folder or the root of the repository.
