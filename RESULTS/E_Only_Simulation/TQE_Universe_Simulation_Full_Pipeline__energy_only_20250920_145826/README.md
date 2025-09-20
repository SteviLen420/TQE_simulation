SPDX-License-Identifier: MIT

Copyright (c) 2025 Stefan Len

[![CI](https://github.com/SteviLen420/TQE_simulation/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/SteviLen420/TQE_simulation/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11-blue)](https://www.python.org/doc/)

# TQE Simulation Results for E-Only Universes
### Author: Stefan Len

## Important Note: Preliminary Data

Please be aware that the results presented in this directory are **preliminary and not conclusive**.

The data generation pipeline is still under active development, may contain bugs, and requires further refinement. These outputs should be considered **representative**—their primary purpose is to serve as a proof-of-concept demonstrating that the theoretical simulations are operational and producing data as expected.

---

### Directory Structure

This directory contains the complete, raw results from the `TQE_Universe_Simulation` runs, specifically for the **E-Only (Energy-Only) universe paradigm**.

Each subdirectory named `TQE_Universe_Simulation_Full_Pipeline__energy_only` contains the full output of a single, independent simulation run. The timestamp in the directory name (`YYYYMMDD_HHMMSS`) indicates the precise execution time.

### Contents of Each Directory

Each simulation folder contains the following file types, documenting different aspects of the run:

* **`CSV files (.csv)`**: Raw numerical data, metrics, and statistics in tabular format. Ideal for further statistical analysis and data processing.
* **`JSON files (.json)`**: Structured output data, parameters, and summaries from the simulations, in a machine-readable format.
* **`Images (.png)`**: Automatically generated visualizations, such as plots, distributions, and other diagrams, illustrating key findings and correlations.
* **`CMB Maps (.fits)`**: Simulated Cosmic Microwave Background (CMB) maps. The `.fits` (Flexible Image Transport System) format is the standard in astronomy, allowing for detailed scientific analysis of the data. 
* **`Wolfram Mathematica Notebooks (.nb)`**: The complete Wolfram Language analyses and code used to process, analyze, and visualize the data. These notebooks ensure the reproducibility of the evaluation. 

## License
This project is licensed under the MIT License – see the [LICENSE](../../LICENSE) file for details.

## Contact

Got questions, ideas, or feedback?  
Drop me an email at **tqe.simulation@gmail.com** 
    
[RESULTS](../../RESULTS)
