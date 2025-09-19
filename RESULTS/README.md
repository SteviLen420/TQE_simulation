# TQE Simulation - Results & Analysis Archive

**Author:** Stefan Len

## About This Directory

This directory contains the complete set of results, raw data, and analysis scripts for my **Theory of the Question of Existence (TQE)** simulation framework. It serves as a comprehensive archive for the various simulation runs and the tools I used to interpret them.

## Navigating This Directory

This archive is organized into several key subdirectories to ensure clarity and reproducibility. All essential components of my analysis can be found within this structure.

* ➡️ **[/TQE_Universe_Simulation_Full_Pipeline/](./TQE_Universe_Simulation_Full_Pipeline/):** Contains the primary Python simulation code I developed. The `README.md` within this folder provides detailed usage instructions for running new simulations.
* ➡️ **[/analysis/](./analysis/):** Contains the Wolfram Language notebooks I created for detailed statistical validation and comparative analysis of the simulation results.
* ➡️ **[/DATA/](./DATA/):** Contains the raw and processed `.csv` and `.json` output files from my various simulation runs, organized into subfolders for each run type (e.g., `E+I` and `E-only`).
* ➡️ **[/docs/](./docs/):** Contains my supporting documentation, including the main scientific manuscript (`TQE_MANUSCRIPT.md`) and the project's narrative (`STORY.md`).

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
